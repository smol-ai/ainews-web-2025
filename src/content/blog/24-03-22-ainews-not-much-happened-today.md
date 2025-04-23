---
id: e716fd7e-b73f-42dd-a92e-459827abadab
title: not much happened today
date: '2024-03-22T23:55:31.644920Z'
original_slug: ainews-not-much-happened-today-2070
description: >-
  Discussions on Reddit's /r/LocalLlama highlight community efforts to fine-tune
  and optimize large language models like LLaMA 2, including hardware setups for
  running 70B parameter models locally and techniques for improving
  retrieval-augmented generation (RAG) performance. Additionally, users explore
  extending LLaMA models with vision capabilities and generating large-scale
  synthetic training data.
tags:
  - reddit
  - local-llama
  - llama-2
  - fine-tuning
  - rag
  - synthetic-data
  - model-optimization
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 3/21/2024-3/22/2024. We checked [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **22** Discords (**341** channels, and **5210** messages) for you. Estimated reading time saved (at 200wpm): **526 minutes**.

We save you the most time when we can say an entire day's worth of news is skippable... and we like the ([apocryphal](https://en.wikipedia.org/wiki/Nothing_Important_Happened_Today#Production)) irony should we be wrong!

Happy peaceful reading, or check out [the new Adept episode on Latent Space](https://www.latent.space/p/adept). We grow our Reddit coverage next week.


---

**Table of Contents**

[TOC] 


---

# REDDIT

> Just starting with /r/LocalLlama for now, and we'll be summarizing the comments soon, but next we have r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence mapped out. Let us know if we're missing any major alpha drop subreddits.

## /r/LocalLlama

**Fine-Tuning and Training LLMs:**

- **Learning how to fine-tune (first time), I've provided links to tutorials I found, but would anybody else recommend further material.** A user is trying to learn how to fine-tune models and has compiled reading material from Reddit and DuckDuckGo. They have questions about training models on specific topics like Cyberpunk 2077 and business data, and are looking for tips on using llama.cpp for fine-tuning. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bkqxui/learning_how_to_finetune_first_time_ive_provided/)
- **Can LLM trained on a Dictionary? If yes, how to do it?** A user wants to train a multi-language model like Gemma on a local language dictionary and is looking for steps for a non-tech layman. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bkqtvs/can_llm_trained_on_a_dictionary_if_yes_how_to_do/)
- **How to generate large-scale synthetic data.** A blog post on how to build large-scale synthetic datasets with 25B+ tokens like those used for training the Phi models from Microsoft, using a Mixtral model. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bk3lqc/how_to_generate_largescale_synthetic_data/)

**Retrieval-Augmented Generation (RAG) and Embeddings:**

- **[question] Query in RAG returning no chunks and no results ?** A user is trying to develop RAG based on a mistral 7b model, chroma DB and markdown texts as input data source. They are doing custom chunking and embedding, but when doing a general query, it does not return any chunks or response. They provide sample code and the markdown file. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bk9tky/question_query_in_rag_returning_no_chunks_and_no/)
- **Has anyone worked on generating embeddings on brain activity?** A user is working with EEG data and wants to match similar EEG signal patterns. They reference a paper and are wondering if anyone has had success in this space. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bk63pq/has_anyone_worked_on_generating_embeddings_on/)
- **Great video on understanding why your RAG/LLM combo isn't working.** A user recommends a highly researched video that discusses the reason why finetuning and RAG are better than RAG alone, the differences between larger and smaller parameter models, and how to contextualize biases in RAG queries. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bk75fk/great_video_on_understanding_why_your_ragllm/)

**Deploying and Optimizing LLMs:**

- **hardware suggestion for llama 2 70b.** A user's boss is asking them to build a suitable workstation rack to run a llama model locally, aiming to get query time under 10s from the current 3 mins on a 7b model. They have a budget of under 15k euros and are looking for suggestions. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bk4j9t/hardware_suggestion_for_llama_2_70b/) 
- **a script to measure tokens per second of your ollama models (measured 80t/s on llama2:13b on Nvidia 4090).** A user shares a script they made to measure tokens per second of ollama models. On an Nvidia 4090, they got 80t/s on llama2:13b and 127t/s on llama2:7b. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bkl5s2/a_script_to_measure_tokens_per_second_of_your/)
- **Speed and Memory Benchmarks for Qwen1.5 models.** A link to benchmarks for Qwen1.5 models in terms of speed and memory usage. [Link](https://qwen.readthedocs.io/en/latest/benchmark/hf_infer.html)

**Extending LLMs:**

- **Is it possible to turn LLaMA into LLaVA.** A user has fine-tuned a LLaMA 2 7B model and is wondering if it's possible to add vision to it without needing to fine-tune LLaVA separately. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bksyq3/is_it_possible_to_turn_llama_into_llava/)
- **Model "memory".** A user is asking if it's possible to improve the "memory" of a model so it can remember what it wrote at least 5 messages back. They know context size matters but are wondering if there's anything else. They also ask if there are any 13b models that support CS 8K. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bkbrp6/model_memory/)
- **Depth upscaling at inference time.** A user shares an experiment that implements depth upscaling at inference time, without actually making the model bigger, so it's GPU-poor friendly. It needs fine-tuning as the model is currently a bit repetitive. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bkjlvu/depth_upscaling_at_inference_time/)

**Applications and Use Cases:**

- **Let's get real: is there anybody who's running agents that actually make money?** A user is asking if anyone runs LLM agents that make money autonomously, even if it's just a few dollars a day. They are looking for vague information about the architecture and models used if people are willing to share. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bk3fd5/lets_get_real_is_there_anybody_whos_running/)
- **What is an efficient way to create your own writing assistant with LLM and training from your own words writing style?** A user is asking for a quick yet efficient way to train an installed LLM or chat.ml to write like the user, as prompting alone still results in writing like chatGPT. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bkg8tn/what_is_an_efficient_way_to_create_your_own/)
- **interacting with a large PDF library.** A user has thousands of scientific papers stored as PDFs and would like a chatbot that could answer questions about the content of the whole library, retrieving info from multiple PDFs without the user having to specify which ones. They are asking if such a tool exists. [Link](https://www.reddit.com/r/LocalLLaMA/comments/1bk1tte/interacting_with_a_large_pdf_library/)

---

# PART X: AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs

**Open Source Models & Frameworks**

- [Open-Sora 1.0](https://twitter.com/svpino/status/1769467954477859047): Open-source text-to-video model, full training process, data, and checkpoints available (100k views)
- [Thunder](https://twitter.com/rasbt/status/1770805633698181383): New open source compiler for PyTorch, achieves 40% speedup over regular PyTorch in LLM training tasks like Llama 2 7B (87k views)
- [Jan](https://twitter.com/omarsar0/status/1770927000326201685): Open-source ChatGPT alternative that runs locally on your computer, supports multiple architectures (51k views)
- [LLaVA-NeXT (LLaVA-1.6)](https://twitter.com/ClementDelangue/status/1771047389983367419): Powerful open source Vision-Language model, now added to Hugging Face Transformers library (1 retweet)
- [Transformers 4.39](https://twitter.com/osanseviero/status/1770931570272030760): New release packed with model updates like Mamba, Command-R, LLaVA-NeXT, MusicGen Melody, StarCoder2, SegGPT and more (11k views)

**Compute Trends & Hardware**

- [Sam Altman believes compute will be the most important currency](https://twitter.com/AISafetyMemes/status/1769600345171481073) in the future, world is underprepared for increasing compute demand (181k views) 
- [Grok on Groq hardware could be a game-changer](https://twitter.com/deliprao/status/1769492688770908207) (3.8k views)
- [Nvidia is the best example of an AGI company](https://twitter.com/far__el/status/1770958097734877352), with total control over the entire hardware/software stack (6k views)

**Evolutionary Model Merging**

- [Sakana AI Labs releases evolutionary approach to model merging](https://twitter.com/maximelabonne/status/1770768615576408434), optimizing both parameters and layer arrangements, enabling creation of specialized models (19k views)
- [Sakana AI's evolutionary model merging used to create Japanese LLM with math reasoning, vision LLM, and image generation model](https://twitter.com/hardmaru/status/1770789055090786354) (2 retweets)

**Retrieval Augmented Generation (RAG)**

- [RAFT (Retrieval Augmented Fine-Tuning)](https://twitter.com/cwolferesearch/status/1770912695765660139): Approach to make LLMs better at RAG by fine-tuning on domain-specific documents, outperforms standard RAG (27k views)
- [Differential privacy for RAG with synthetic data generation](https://twitter.com/llama_index/status/1770837291855991085), enabling knowledge sharing from sensitive datasets (36k views)

**Emerging Trends & Applications**

- [SceneScript from Meta AI](https://twitter.com/AIatMeta/status/1770844932346920976): Novel method for reconstructing environments and representing physical space layouts using end-to-end machine learning (230k views)
- [Suno AI releases v3 model](https://twitter.com/suno_ai_/status/1770857426507399285) capable of producing radio-quality music in seconds (152k views)
- [Cohere transforming insurance with large context summarization and knowledge assistants](https://twitter.com/cohere/status/1770817028183486824) (4k views)
- [Runway partnering with Musixmatch](https://twitter.com/c_valenzuelab/status/1770801245445407001) to make lyric video creation and customization easier (8k views)

**Prompt Engineering as a Career**

- ["I still remember when people thought that "Prompt Engineering" was going to become a real career."](https://twitter.com/svpino/status/1770873052810883156) (1M views)


---

# PART 0: Summary of Summaries of Summaries


> we are concluding that Claude Opus is just the best model for top level summaries so we're discontinuing the A/B/C tests (see archives for our struggles/record). We'll be exposing parallel runs for all 3 + more models (incl Gemini 1.5!!) as this problem is topologically similar to our personalization app we'll be launching.

- **Stable Diffusion 3 Anticipation Builds**: The Stability.ai community eagerly awaits the release of **Stable Diffusion 3 (SD3)**, discussing optimal **control nets** for art generation, **AMD GPU compatibility**, and **cloud GPU services** for those with limited hardware. Troubleshooting tips were shared, like using [lshqqytiger's fork](https://github.com/lshqqytiger/stable-diffusion-webui-directml) for AMD support.

- **Unsloth AI's Upcoming Features**: **Unsloth AI** is working on integrating **multi-GPU support** and a platform UI for automatic data curation. The community also debated evaluation frameworks, data quality, and the importance of transparency in benchmarks like correcting the **MMLU dataset** where 25% of examples had incorrect reference solutions.

- **OpenInterpreter's 01 Light Launch**: The **[01 Developer Preview](http://openinterpreter.com/01)** launch, a portable AI device that controls computers via voice, generated buzz. The community shared assembly instructions, the [Bill of Materials](https://github.com/OpenInterpreter/01/blob/main/hardware/light/BOM.md), and [3D print designs](https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-model), while also discussing shipping and software features.

- **LM Studio Updates Spark Discussions**: **LM Studio's** new features like multi-model support and ROCm 0.2.17 Beta v3 release led to troubleshooting discussions around **ejecting models**, **GPU offloading**, **ZLUDA interference**, and **high CPU usage**. The community also recommended the [Instructor library](https://github.com/jxnl/instructor) for structured LLM outputs.

- **AI Ethics and Security Concerns**: Conversations in Perplexity AI and HuggingFace touched on the ethics of AI accessing sensitive information, like in the ['Guardrails Arena'](https://huggingface.co/spaces/lighthouzai/guardrails-arena) experiment, and security vulnerabilities allowing interception of encrypted AI chatbot tokens ([detailed explanation](https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/)).

- **Emerging Techniques and Datasets**: Several channels discussed new AI techniques and datasets, such as: 
  - [DenseFormer](https://arxiv.org/abs/2402.02622) proposing Depth-Weighted-Average to improve transformer models
  - [Quiet-STaR](https://arxiv.org/abs/2403.09629) for generating rationales per token to enhance LM text interpretation
  - [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia), a large synthetic dataset for LLM pre-training
  - [ASCII Art Dataset](https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM) sparking interest in diffusion models for ASCII art

- **Optimizing AI Performance**: Discussions covered various optimization techniques, including:
  - [1-bit LLMs like BitNet b1.58](https://arxiv.org/abs/2402.17764) matching full-precision models with better efficiency
  - [Galore optimizer](https://github.com/huggingface/transformers/pull/29588) for memory-efficient tuning of large models
  - Fusing **GaLore's Adam optimizer** with Triton for faster pre-training and fine-tuning
  - Guidelines for maximizing GPU performance of transformer models ([paper](https://arxiv.org/abs/2401.14489))

---



# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3 Fever Rises**: The community is eagerly awaiting the release of **Stable Diffusion 3 (SD3)**, with conversations focused on selecting the best **control nets** for art generation. There's also a vibrant exchange of insights on **AMD GPU compatibility** and recommendations for **cloud GPU services** to empower those with limited hardware capabilities.
  
- **Diving Into AMD Waters**: A user facing a **RuntimeError with NVIDIA drivers** on an AMD system received help by being steered towards [lshqqytiger's fork](https://github.com/lshqqytiger/stable-diffusion-webui-directml) that supports AMD GPUs, along with a thorough installation guide.

- **VRAM-Gate**: Technical discussions are unfolding on the anticipated **V-RAM requirements** for the soon-to-drop SD3, fueling speculation about the feasibility of running resource-intensive models on local machines.

- **Prompt Engineering-as-a-Service**: Community members are sharing techniques to refine their **prompting skills** for creations ranging from "tribal videos" to **D&D campaign** visuals, searching for specific models fine-tuned to understand elaborate prompts for complex character and scenery art.

- **AI Tools: Bane or Boon?**: Debates are sparking around the impact of AI on employment and creativity, with opinions ranging from caution to optimism about AI's role as an **evolutionary tool** in augmenting human effort.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Multi-GPU Support and Data Curation Heading to Unsloth AI**: Unsloth AI is actively working on integrating **multi-GPU support** as an open-source feature, aiming for compatibility with platforms like Kaggle. Additionally, they're developing a platform UI for automatic data curation to simplify the data preparation steps in model fine-tuning.

- **Exploring Solutions for Unsloth AI Installation Issues**: Users reported problems installing Unsloth AI, including 'no matching distribution' errors and `RuntimeError` on single-GPU restriction setups. There's also discussion on potential **CUDA level changes** for 4-bit quantized models and VRAM constraints exceeding 15GB in quantized models, possibly causing out-of-memory errors.

- **Unsloth AI Community Tackles Diverse Issues**: Discussions around configuring **LoRA settings**, handling out-of-memory errors by adjusting training parameters, and tips on saving/loading models and tokenizers locally. Concerns about missing dependencies like `protobuf` and confusion about the best models for certain technical domains were also notable.

- **Community Spotlight on Samantha Mistral Instruct 7B**: Community member cognitivetech showcased their work with [Samantha Mistral Instruct 7B](https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b_bulleted-notes), specifically for summarizing psychology books. Troubles with model quantization and the promise of a working upload to Hugging Face were shared.

- **Lightning Thunder Causes a Stir with Unsloth AI**: Community members highlighted potential missteps in the integration of Unsloth AI with Lightning Thunder, pointing out performance issues and incorrectly implemented kernels. There's a call for collaboration and accurate representation of Unsloth's capabilities in benchmarks and some expressed frustration over misleading performance comparisons on [Twitter](https://twitter.com/danielhanchen/status/1770863091296632921).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Global Launch Party for 01 Light**: Engineers are excited about the [01 Developer Preview](http://openinterpreter.com/01) launch, a portable voice interface device for computers with capabilities to see the screen and use apps. The community is sharing assembly instructions and [Bill of Materials](https://github.com/OpenInterpreter/01/blob/main/hardware/light/BOM.md), and have concerns about shipment to regions like India and the EU.

- **Hardware Enthusiasts Get Crafty**: DIY community members are discussing 3D-printing their own versions of 01 Light, with **design files [available on Printables](https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-model)** and source code found on the **[OpenInterpreter GitHub](https://github.com/OpenInterpreter/01)**.

- **Troubleshooting Across Time Zones**: Wide-ranging troubleshooting topics include setting up 01 on various operating systems and addressing international shipping concerns. A workaround for Windows compatibility was suggested—`poetry run 01 --client --client-type linux --server-host localhost`.

- **Curiosity and Concerns Around Software Features**: Members probe into the software aspects of OpenInterpreter, discussing local versus cloud operation, API keys, language compatibility, and battery life, highlighting key aspects for an AI engineer's understanding of product usability and technical specifications.

- **Serving Up a Teaser**: A single message in the #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/) channel heads towards lean content with just a YouTube link related to OpenInterpreter, without providing context or content details.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Hermes 2.5 Holds the Crown**: After the addition of [code instruction examples](https://huggingface.co/roborovski/superprompt-v1), **Hermes 2.5** has shown superior performance over **Hermes 2** across various benchmarks, with users discussing the impact of different models and configurations on LMStudio's performance.

**Tackling LM Studio Quirks and Quibbles**: Members report issues with **LM Studio version 0.2.17**, including symlinks failing to be recognized and errors stating "Model with key Mistral/Hermes... not found." Additionally, performance discussions include abnormal **CPU usage** and **compatibility** with AMD Rocm and RX 570 graphics cards.

**AI Ethics and Security - A Hot Debate**: The community delved into the ethics and security of AI through discussions about interacting with models in [Hugging Face's 'Guardrails Arena'](https://huggingface.co/spaces/lighthouzai/guardrails-arena), and security exploits allowing the interception of encrypted AI chatbot tokens ([detailed explanation here](https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/)).

**Model Mastery and Multitasking**: Users exchanged knowledge on optimizing the functionality of multimodal models in **LM Studio**, dealing with issues of **VRAM limitations**, and using multi-model setups to improve complex tasks. The conversation also included advice on models that facilitate "Full GPU Offload Possible" on personal machines with specific capacities.

**AMD ROCm - Going for Stability or Stirring Up Storms?**: The release of **ROCm 0.2.17 Beta v3** generated mixed feedback, with members reporting issues related to **ejecting models**, **GPU offloading**, **ZLUDA interference**, and **high CPU utilization**. Despite these challenges, several reported stable performance on **AMD GPUs**, suggesting potential improvements in the latest ROCm beta version.

**Streamlining AI Workflows**: Engineers recommend exploring the [Instructor library](https://github.com/jxnl/instructor) for structured outputs in language model workflows and sharing successful integrations of special fine-tuned versions of **OpenChat** with the dolphin mistral fine-tune to enhance language modeling efficiency.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Model Showdown: Claude 3 Opus vs. Gemini**: Users debated the performance nuances between **Claude 3 Opus** and **Gemini**, discussing which AI feels more humanlike. The discussion also extended to personal AI models like **Inflection-2.5** and **Pi.AI**, highlighting their conversational strengths and concerns about their platforms' futures.

- **Navigating AI with Perplexity**: Queries about how **Perplexity AI** conducts web searches and image generation were prominent, indicating user interest in mobile accessibility of features like **Unlimited Claude3 Opus**. Inquiries also involved the use of **Perplexity AI** for topics ranging from the largest planet to **GPT-5 release rumors**.

- **Community Cries for Darker iOS Themes**: Tech-savvy discordians shared frustrations about the lack of a darker midnight/black theme in **iOS app updates**, citing the need for visual comfort in their digital environments.

- **Token Limit Reached! Learning the Hard Way**: An API user's BadRequestError, due to exceeding perplexity's **16384 token limit** with a **6621-token prompt and 9900-token output**, highlighted the importance of accurate token counting in API requests.

- **Frustration Over Cloudflare's Overzealous CAPTCHA**: A user lamented the intrusive nature of Cloudflare's CAPTCHA challenges, especially when using VPNs, suggesting that even regular browsing could trigger these defenses. 

The sources cited for technical reference included [Inflection-2.5](https://inflection.ai/inflection-2-5), [Neuralink's first human trial patient insights](https://www.businessinsider.com/neuralink-first-human-trial-patient-quadriplegic-elon-musk-x-2024-3), and Perplexity's nature as a possible Google Search wrapper according to [Analytics India Magazine](https://analyticsindiamag.com/perplexity-is-most-likely-a-google-search-wrapper/). The [Perplexity documentation](https://docs.perplexity.ai/reference/post_chat_completions) was noted for clarifying token counts.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Missing Personality Predictions**: Conversations revealed interest in the **"mypersonality"** dataset, notably for its applications in author personality prediction from text, but there were concerns over its accessibility.

- **Pooling for AI Excellence**: The [Hugging Face diffusers library](https://github.com/huggingface/diffusers/issues/7365) was critiqued for its embedding implementations, with suggestions to revise the pooling method for text embeddings to boost model performance.

- **Dataset Future Uncertain**: The **LAION-5B** dataset's removal has led to the exploration of alternative datasets like Datacomp and DFN amid new EU regulations, casting doubt on whether LAION can overcome legal barriers to republish their datasets.

- **Calls for Transparency from OpenAI**: The guild anticipates that OpenAI might open-source the training code of upcoming models like **SD3** despite previous hesitancy, an important topic for those pursuing progress in AI.

- **AI Sabotage or Safety?**: Members were skeptical of the intentions behind researchers' concerns over datasets containing sensitive material, pondering whether such actions serve as unnecessary hindrances to AI advancements or are genuine efforts to address safety.

- **Innovative Image Scaling Suggested**: A study on [arXiv](https://arxiv.org/pdf/2403.13043.pdf) proposed using **multiple scales** of an image to enhance model outcomes, indicating a potential path for visual AI engineering.

- **Time Tricks for Image Encoding**: An intriguing approach introduced via an [arXiv paper](https://arxiv.org/pdf/2403.13802.pdf) employs encoding images with **six times as many timestamps**, though some community members consider this to be more of a workaround.

- **Cryptic Tweet Teases Tech Trends**: A tweet from [Bindu Reddy](https://twitter.com/bindureddy/status/1770934470373421522?t=5dLWWD7d9PN0C4ZjHAY_Tw&s=19) was mentioned as potentially hinting at future developments, sparking curiosity among the members about its implications.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Exploring "Extended Mind" in AI**: Engineers discussed the "**Extended Mind**" concept, which involves storing vectors for associative memory and fetching the top k during forward pass, enhancing reasoning and memory in models. The debate was based on Phoebe Klett's tweet and the integration with **Mistral** was seen as a promising future experiment.

- **Fine-Tuning Challenges & AI Devices Buzz**: A new [YouTube tutorial](https://www.youtube.com/watch?v=21Tc92g15pM) offers guidance on fine-tuning the **LLaVA model**, while discussions also centered around the latest open source AI device, **01 Light**, aiming to control computers via voice, shared in a [tweet by OpenInterpreter](https://twitter.com/OpenInterpreter/status/1770821439458840846).

- **Cosmopedia and Quiet-STaR Make Waves**: The Hugging Face blog's post on **Cosmopedia** showcases creating synthetic datasets for AI, and the paper on [Quiet-STaR](https://arxiv.org/abs/2403.09629) suggests LMs can generate explanations per token, enhancing text interpretation.

- **AI Model Improvement Efforts Gather Steam**: Engineers faced difficulties with *BatchAllTripletLoss* performance in embedding models and shared progress on projects such as an open-source Rainfall API (RAG) platform. Discussions also entertained the possibility of AI interaction using gestures or even direct brain interfaces.

- **Quantization Enquiries and Collaborative Advances**: Members shared information on model **quantization**, including an outdated repository, **AutoAWQ**, ([GitHub link](https://github.com/casper-hansen/AutoAWQ/tree/striped_hyena)) for 4-bit quantization, and pondered over the theoretic underpinnings of causal masking in attention mechanisms.

- **Data Tools and Technology March Forward**: Users rallied around **LanceDB** for its hybrid search capabilities with generative interfaces, while integration technologies like **Polars** and a shared **GitHub repository** ([Neural-Dragon-AI/Cynde](https://github.com/Neural-Dragon-AI/Cynde)) exhibited potential for combining semantic wisdom with predictive machine learning models.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **RAG Debate Heats Up**: A lively debate ensued over **Retrieval-Augmented Generation (RAG)** versus agent-based models in AI, with some arguing that RAG is merely a stopgap for missing knowledge, while others champion the complexity and robustness of agent-based models.

- **FastChat's Formatting Fiasco**: **FastChat's alpaca** model was flagged for inconsistent formatting when compared to Stanford's alpaca format, prompting suggestions for a pull request to unify them for consistency, as seen in the [FastChat GitHub repository](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L550).

- **Galore's Graceful Integration**: Buzz surrounds the **Galore optimizer**, notable for VRAM efficiency in tuning large models, recently [merged with Hugging Face Transformers](https://github.com/huggingface/transformers/pull/29588), and its capability to manage full parameter tuning with less memory usage, as highlighted in a [benchmark issue](https://github.com/jiaweizzhao/GaLore/issues/6).

- **GPT-3.5 Inquiry Ignites Interest**: Questions about **GPT-3.5** performance and inference times sparked discussions amid concerns over slower local inference speeds on Macs due to privacy constraint workarounds for sensitive data such as patient information.

- **Text Classification Contemplations**: In the realm of text classification, the strategy of fine-tuning Language Models (LLMs) to generate class names as outputs, rather than adding a classifier head, was debated for its flexibility and the benefits of encouraging a model to follow a *chain of thoughts*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Evolving AI Merging Methods Unveiled**: A recent [paper by Hardmaru](https://arxiv.org/abs/2403.13187) introduced an *automated evolutionary algorithm* for foundation model merging, sparking debate on its potential to combine open-source models and boost performance without heavy training.

- **AI Community Thrives in Paris**: Members actively shared their experiences and plans related to AI meetups in Paris, with particular excitement about the [Paris Retrieval Augmented Generation group](https://www.meetup.com/fr-FR/paris-retrieval-augmented-generation-group/), highlighting a robust digital tech scene.

- **Zoom Saves the Paper Club**: A Zoom room creation was suggested to overcome **speaker rights** issues in the Discord channel, demonstrating resourcefulness in face of technical limitations.

- **Innovations and Discussions in AI Utility**: The group dove into **llama.cpp's** potential GPU use, "pad and pray" tensor dimension solutions, and a visualization by bbycroft.net for transformer model understanding. Additionally, there's a look forward to discussions on music generation models and navigating large codebases.

- **Podcast Sheds Light on AI Giants**: A new podcast with insights into companies like *OpenAI*, *Google*, and **Adept** gained attention, which was complemented by a [Twitter post](https://twitter.com/swyx/status/1771255525818397122). An AI event named *AI In Action* spotlighted **Llama.cpp**, with an invitation to join via a [Discord channel](https://discord.com/channels/822583790773862470/1200548371715342479).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Sensitive Data Meets AI Safely**: LlamaIndex blog highlighted the risks of training LLM/RAG apps with sensitive data such as patient clinical reports and proposed using *differential privacy* to protect individual information, with insights shared via a [blog post tweet](https://t.co/2ZipmvOwXv).

**Navarasa 2.0 Embraces Diversity**: The blog introduced Navarasa 2.0, the upgraded **Google Gemma 7B** fine-tuned for 15 Indian languages, emphasizing the value of local language support in AI, highlighted through a [release tweet](https://t.co/HHrfonnAr2).

**UX Gets Smarter**: A new UX template featured on LlamaIndex aims to enhance agent-human interactions by limiting agent requests for human input to necessary instances, with more information available in the [associated tweet](https://t.co/Z16QPCWFmG).

**Integration Headaches!**: Discord members discussed the complexities of integrating various tools with a chatbot and encountered issues like "BadRequestError," with documentation [suggestions](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/) and troubleshooting advice shared in the heated conversation.

**Documentation Drama**: Users wrestled with accessing the LlamaIndex documentation amidst an update to MKDocs, shared [links to the new documentation format](https://docs.llamaindex.ai/en/stable/api_reference/indices/vector/#llama_index.core.indices.VectorStoreIndex), and offered clarification on a query pipeline DAG confusion detailed [here](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Quest for Compact Code Datasets**: The [CodeSearchNet corpus](https://huggingface.co/datasets/code_search_net) was considered as a pretraining dataset but encountered issues with context length, and instead, [The MiniPile](https://arxiv.org/abs/2304.08442), a 1M document corpus, was suggested for its diverse and compact size suitable for pre-training with minimal performance loss.

**Under the Hood of Closed-Source Models**: The community discussed the lack of access to logprobabilities and tokenizers in closed-source models like Claude and Gemini, in contrast to platforms like OpenAI that readily provide them, speculating proprietary reasoning behind the restriction.

**Maximize Your Model's GPU Potential**: Guidelines from a [recent paper](https://arxiv.org/abs/2401.14489) on maximizing GPU runtime performance for transformer models included hyperparameter tuning and efficient model shapes, potentially increasing throughput by up to 39%.

**AI Venturing into Biotechnology**: An [Ars Technica article on AI in antibody design](https://arstechnica.com/science/2024/03/antibodies-against-anything-ai-tool-adapted-to-make-them) sparked discussions, revealing both excitement for the promise of diffusion models and skepticism regarding their practical economic applications.

**Easing the Debugging Headache**: Participants faced issues when using `megatron-deepspeed` with `lm-eval 0.3.0` and proposed workarounds like loading from an older version of `cais/mmlu`, which was still problematic due to auxiliary train split relocations, as indicated by a [Gist traceback](https://gist.github.com/jonabur/d99bb92be81a5af6b01f81b589b68d21).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**ASCII Art Gets a Dataset and Develops in Diffusion**: Engineers shared excitement over **ASCII Art** with the unveiling of an [ASCII Art dataset](https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM), and discussions on fine-tuning LLMs and diffusion models to generate ASCII art. A particular challenge is fine-tuning a language model to generate intricate designs, prompting a search for efficient training methods and the idea of an ASCII-adaptive diffusion model.

**SMIT Brings Audio to Language Models**: A new modality integration tool named **SMIT** was introduced, making it easier to include audio in language models. A YouTube demonstration of **SMIT** for music generation models piqued the interest for its potential applications. Meanwhile, **Fluently-v4** was globally released, offering a single model solution for multiple tasks.

**1-bit LLMs Promise Efficiency**: The paper on [1-bit LLM BitNet b1.58](https://arxiv.org/abs/2402.17764) suggested significant performance matching full-precision models while optimizing for cost-efficiency. This could lead to the development of 1-bit optimized hardware for LLMs.

**New Approaches and Tools in Various AI Domains**: SegGPT's introduction adds to the toolset for image segmentation tasks, promising one-shot results. The **UniProt project**’s 1024-dimensional embeddings are poised for retraining with **Matryoshka embeddings** for better searchability in protein databases. A [profound exploration](https://www.kaggle.com/code/muhammadibrahimqasmi/deciphering-obesity-trends-an-in-depth-eda) of obesity trends using data analysis sets a new precedence for health-related AI research.

**Community Collaborations Flourish in Model Development and Federated Learning**: The search for collaboration grows with members seeking assistance on projects from federated learning for load forecasting, sharing possibilities like the 6TB **"The Stack"** dataset for deep code generation, and invoking **BERTopic** for modernized topic modeling. Concerns over quantizing finely-tuned models and issues around the Trainer class in Huggingface were discussed, reflecting a shared commitment to overcoming technical hurdles together.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Conversations on the Cost of AI and its Applications**: Members discussed the **cost of adding Chat GPT bots to Discord** and the pain points around **not receiving responses in Postman** despite correct setup. The buzz around **Perplexity's AI as a Google Search wrapper** fueled discussions, with a reference to Mohit Pandey's article suggesting it summarizes top Google Search results. A comparison between AI's potential in **video compression** and deep learning super sampling (DLSS) was drawn, with an existing blog post as a reference point. In terms of efficiency, a member claimed an **80% storage cost reduction** by converting Float32 embeddings to Int8 for their vector database [Deep Compression with AI](https://www.dbreunig.com/2023/11/07/extreme-compression-with-ai.html) and [Perplexity article](https://analyticsindiamag.com/perplexity-is-most-likely-a-google-search-wrapper/).

- **GPT-4 Custom Models and Usability Queries**: Inquiry into **connecting to custom GPT-3 models** via API led to a shared **Assistants API Guide**. Feedback was sought for a GPT that assigns animal alter-egos with a prompt example provided. A sudden reduction to **pinning only 4 Custom GPTs** perplexed a user, signaling a possible undocumented change. Conversations covered the productivity of **distributing knowledge files across multiple GPTs** versus single GPT consolidation for diverse parts of a prompt [Assistants API Guide](https://help.openai.com/en/articles/8673914-gpts-vs-assistants).

- **Server Rules and Product Descriptions Dominate Prompt Engineering Talk**: **Rule 7** came into highlight, reinforcing guidelines against self-promotion after a user's post on prompt engineering jobs, and a user's attempt to advertise a **prompt chaining/prompt engineering toolkit**. Frustration arose over **GPT-4 Vision's inability to assist with disabilities**, whilst another member sought to challenge ChatGPT with **generating natural product descriptions**, suggesting to split the task into generating specific sections may be more effective. 

- **API Channel Echoes Rule Reinforcements and Model Limitations**: Similar to discussions in the prompt engineering channel, the API discussions highlighted **Rule 7**, with an apology issued for a previous violation. The limitations of **GPT-4 Vision** in recognizing disabled individuals catalyzed a conversation on AI inclusion. The challenge of using ChatGPT for **automated product descriptions** without human oversight was raised, questioning the preciseness of AI-generated content.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Python Dependency Puzzles Pester Langchain Enthusiasts**: Python version conflicts and dependency issues in **[langchain-ai/weblangchain](https://github.com/langchain-ai/weblangchain)** cause headaches, with errors like `TypeError: Type is not JSON serializable: numpy.float64` leading to crashes. A related issue is being tracked on GitHub as **[TypeError: Type is not JSON serializable: numpy.float64](https://github.com/langchain-ai/langchain/discussions/17876)**.

- **Scribe Seeking Serenity in Serialization**: The `numpy` serialization problem persists despite using Poetry and pinning older versions of Starlette, culminating in a new GitHub issue titled **[TypeError: Type is not JSON serializable: numpy.float64](https://github.com/langchain-ai/langserve/issues/551)** to resolve the Langchain/Langserve incompatibilities.

- **Trouble with Token Limits Triggers Tech Talk**: Langchain users are exploring features to handle large outputs that exceed a model's token limitation, such as OpenAI's GPT-4-Turbo's 4k output tokens, considering methods for chains to continue generating output by sending additional requests.

- **Promptsage Aims to Sweeten the Prompt-Empire**: A new project, [Promptsage](https://github.com/alexmavr/promptsage), offers a simplified approach to prompt building and sanitization for Large Language Models alongside security and privacy guardrails, designed for compatibility with langchain.

- **Data Analysts Delight in AI-Driven Evolution**: An article titled "Harnessing Langchain, Instructor, and Pydantic: Redefining Data Analysis with AI" applauds the integration of various tools to enhance data analysis. The insights can be read on [Medium](https://medium.com/technology-hits/harnessing-langchain-instructor-and-pydantic-redefining-data-analysis-with-ai-6cfe0e89b616).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **West Coast Users Struggle with Latency**: Users on the **West Coast** are facing slow requests suspected to be related to a cloud services issue; an ongoing investigation is underway.

- **Gemini 1.5 Pro Sparks Interest and Inquiry**: Despite [official documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning) making no mention beyond version 1.0, discussion buzzed around **Google's Gemini 1.5 Pro** and its impressive 1 million word context window; with some members already reaching out to Google for access.

- **Model Showdown: C3 vs. Claude 3 vs. GPT-4**: Engineers debated models with **C3 Model** under fire for its inconsistency, while a self-moderated variant of **Claude 3** received favorable comparison to **GPT-4** for content moderation.

- **Divided Opinions on Grok AI's Performance**: A split emerged in opinions on **Grok AI**, with criticisms of it being potentially undertrained and costly, while others defended its capability as a base model not directly comparable to chat-tuned models like Mixtral.

- **Grok's Benchmarks and Public Testing Spark Debate**: Engineers debated the value of **Grok AI** benchmarks and shared [a link](https://grok.x.ai/) to trial the model, highlighting its accessibility through the xAI platform possibly without the need for Twitter Premium+. Discussion also included what content would be best to evaluate Grok's performance.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Nanobinding for Machine Learning Acceleration**: In discussions, **nanobind** was recommended for efficiency improvements in machine learning, particularly for MLX. Concurrently, members encountered difficulties during a GTC event with Discord's stage channel, suggesting a pivot to voice channels to avoid similar issues in the future.

- **Optimizers and Compilers on the Leading Edge**: A member revealed success in fusing **GaLore's Adam optimizer** with Triton to enhance memory efficiency in models, supported by a [GitHub pull request](https://github.com/jiaweizzhao/GaLore/pull/29). Separately, the [micrograd-cuda library](https://github.com/mlecauchois/micrograd-cuda) was introduced for CUDA accelerating Python-based micrograd extensions, and [Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder), a compiler for PyTorch, drew attention for promising performance improvements on accelerators.

- **Matrix Multiplication, Summation, and Standards Enlightened**: The community analyzed the *Ozaki scheme* for enhancing matrix multiplication, with a nod from Jeremy Howard, and discussed the [Kahan summation algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) for reducing computation errors. Additionally, the IEEE 754 floating-point standards were noted as crucial, citing an [ITU paper](https://www.itu.dk/~sestoft/bachelor/IEEE754_article.pdf) on the topic.

- **Virtual Conversation Conduct and Knowledge Incubation**: A member proposed using structured messages for improved clarity in conversations, with a hat tip to <@272654283919458306> for exemplifying this on another server. On the educational front, a [Springer book link](https://link.springer.com/book/10.1007/978-3-031-30442-2#other-volumes) for PPAM 2022 was shared, offering a gateway to contemporary proceedings in parallel processing.

- **Seekers of CUDA Knowledge Engage in Sharing and Humor**: A member looking for confirmation on **Chapter 2 exercises** from a 'pmpp-book' suggested private messages for answer verification. Engaging the lighter side, a new [Zero to Thunder tutorial](https://lightning.ai/lightning-ai/studios/zero-to-thunder-tutorial) targeting Python and PyTorch users was unveiled at GTC, alongside observations that new Blackwell GPUs sport designs resembling smiley faces, sparking light-hearted exchanges via [Twitter](https://fxtwitter.com/iScienceLuvr/status/1770931936657358908).

- **Triton's Tenacious Troubleshooting**: In the **triton-puzzles** channel, the community decoded tensor color coding and discussed potential misrepresentation in out-of-bounds indicators. Issues with the `tl.exp` operator ignited conversations about a NotImplementedError in interpreter mode, and efforts on Triton puzzles progressed, marking the completion of Puzzle 3 and collaborative debugging on Puzzle 4.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Matchmaking Malfunction with GPT4.Turbo**: Uniti AI is grappling with **GPT4.Turbo** inaccurately suggesting property spaces, with mismatches as glaring as offering 17,000 sq. ft for requests of 2,000 - 4,000 sq. ft. The challenge is amplified when trying to adhere to a specified percentage range for property sizes, encouraging suggestions for simplified solutions such as **direct SQL queries**.

- **Beware the "Common LLM Trap"**: Engineers discussed the potential overuse of LLMs for tasks that might be more efficiently tackled with basic database queries. A blog post on [Retrieval Augmented Generation (RAG)](https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/) by Jason Liu was shared to highlight how pairing LLMs with standard database interactions can improve tasks like date range extraction. 

- **Direct Integration Trumps Bedrock for Claude**: In the realm of AI interfacing, a user reported that **direct integration** with the AI model **Claude** is preferable over using frameworks like **Bedrock**, citing better reliability and uptime. Even a user with priority rate limits, bypassing a hefty 200k+ waitlist, chose a direct connection with Claude.

- **Jeffreyw128 and ibash Leave Cryptic Remarks**: Within the discourse, succinct messages such as "lol wut" from jeffreyw128 and ibash's one-word critique, "Damn," in response to high quality code writing, punctuated the conversations but provided limited context or actionable discussion points.

- **Is Basic Prompting Insufficient?**: Questioning the effectiveness of basic prompting emerged in a solitary message, implying a need for more advanced or nuanced techniques when engaging with AI, particularly for those in technical fields.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Hunt for Synthetic Benchmarks Heats Up**: Engineers are looking into **fully synthetic benchmarks** to study language model capabilities, with startups generating data to support this research. The goal is to better understand LLM capabilities by manipulating factors like diversity and reasoning in the training data.

- **Engineers Buzzing About Synthetic Data and Open Curation**: Interest has been piqued in the realm of **synthetic data and worlds**; one engineer is even considering authoring a paper on it. Additionally, a systematic approach to open-source data curation for model pretraining has been suggested to improve collective efforts in the field.

- **ChatGPT: The Academic's New Assistant**: Discussion highlighted the utilization of **ChatGPT** for rewriting content in academic projects to push for **state-of-the-art results**, with a **side project** underway to explore further applications, indicating rewriting tasks are now a mainstream strategy.

- **Chess, Go, and Human Psyche: Tech Giants in an AI-Infused World**: Members muse over the psychological impact of AI advancements, citing historical events like Kasparov's defeat to Deep Blue and reflecting on an individual's attitude towards AI. A philosophical discussion on the potential for creating generalist agents in reinforcement learning was highlighted, featuring insights by **Minqi Jiang** and **Marc Rigter** and shared via [MLStreetTalk Tweet](https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw).



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

**Calling All Open Source Enthusiasts**: A community member is seeking collaboration on **the 01**, a fully open source hardware device, and has shared details in a [public tweet](https://twitter.com/OpenInterpreter/status/1770821439458840846).

---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1220272884610502667)** (884 messages🔥🔥🔥): 

- **Model-mania Begins**: Members are fervently discussing **Stable Diffusion models**, specifically their anticipation for **Stable Diffusion 3 (SD3**) and the careful selection of **control nets** and addons for generating art. Questions about **AMD GPU compatibility** and advice on **cloud GPU services** for those with less powerful hardware are prevalent.
- **Tech Troubleshooting in Action**: One member needed assistance with a **RuntimeError about NVIDIA drivers** on an AMD GPU system when trying to use **Stable Diffusion WebUI**. They were directed to [lshqqytiger's fork](https://github.com/lshqqytiger/stable-diffusion-webui-directml) for AMD support and given a step-by-step guide for installation.
- **The Hype for Higher Quality**: The conversation turned technical discussing **V-RAM requirements** for different **Stable Diffusion models**. With the upcoming SD3 believed to demand high VRAM, members speculate about the practicalities of running such large models locally.
- **Prompt Crafting and Art Creation**: Users are sharing **prompting techniques** and **AI results** for various creative projects, like generating images for "tribal videos" and **D&D campaigns**, with some seeking specific models that can comprehend detailed prompts for generating character art and scenery.
- **The Spectrum of Community Opinions**: Debates emerge around the benefits and drawbacks of AI, with some expressing skepticism regarding **AI's impact** on jobs and creativity. Meanwhile, others stress the **evolutionary nature of AI tools** and their potential to augment human workflows.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://platform.stability.ai/">Stability AI - Developer Platform</a>: no description found</li><li><a href="https://www.runpod.io/console/gpu-cloud">no title found</a>: no description found</li><li><a href="https://app.suno.ai/song/8250b732-8f32-4be1-a38e-9d1c23f926b5/">Kitty Cat Groove | Suno</a>: reggae, dancehall song. Listen and make your own with Suno.</li><li><a href="https://civitai.com/articles/1997/comfyui-guide-to-stacker-nodes">ComfyUI - Guide to Stacker Nodes | Civitai</a>: This article is about Stacker Nodes and how to use them in workflows. It is intended for both new and advanced users of ComfyUI. Stacker nodes are ...</li><li><a href="https://tenor.com/view/arch-arch-linux-btw-i-use-arch-btw-monokuma-gif-26738028">Arch Arch Linux GIF - Arch Arch Linux Btw - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://app.suno.ai/song/78ab7dff-a3fc-4038-90e2-d15cec8590d8/">Generative AI | Suno</a>: pop song song. Listen and make your own with Suno.</li><li><a href="https://app.suno.ai/song/7df3f524-39f9-4678-bee0-ba6ab4175417">Schnappi rock version | Suno</a>: rock, hardcore, breakcore, electroswing, song. Listen and make your own with Suno.</li><li><a href="https://app.suno.ai/song/40e83bad-3ef2-40bf-aa89-8f49a5a981c1">BABY SHARK | Suno</a>: prog rock, electric guitar, electric bass, syncopated song. Listen and make your own with Suno.</li><li><a href="https://civitai.com/models/350524/jboogx-and-the-machine-learners-animatelcm-subject-and-background-isolation-via-invertmask-vid2vid-highresfix">JBOOGX &amp; THE MACHINE LEARNER&#x27;S  ANIMATELCM SUBJECT &amp; BACKGROUND ISOLATION via INVERTMASK VID2VID + HIGHRESFIX - v1.0 | Stable Diffusion Workflows | Civitai</a>: This is an evolution of my AnimateLCM workflow. Cut down and put together for ease of use. This workflow should at MOST require 12-14GB of VRAM mak...</li><li><a href="https://civitai.com/models/38784/controlnet-11-models">ControlNet 1.1 Models - Tile (e) | Stable Diffusion Controlnet | Civitai</a>: STOP! THESE MODELS ARE NOT FOR PROMPTING/IMAGE GENERATION These are the new ControlNet 1.1 models required for the ControlNet extension , converted...</li><li><a href="https://civitai.com/models/38784?modelVersionId=44756">ControlNet 1.1 Models - Softedge | Stable Diffusion Controlnet | Civitai</a>: STOP! THESE MODELS ARE NOT FOR PROMPTING/IMAGE GENERATION These are the new ControlNet 1.1 models required for the ControlNet extension , converted...</li><li><a href="https://app.suno.ai/song/f435f67b-e5e3-4ef3-bd5e-c9764b90b550">Listen | Suno</a>: Industrial metal with heavy guitar and the synth beat of an alarm going off. Has a piano counterpoint bridge.  song. Listen and make your own with Suno.</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs">Install and Run on AMD GPUs</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=RwLY0bsQpx4&t=1s">1 CLICK STABLE DIFFUSION NOTEBOOK FOR RUNNING ON THE BEST FREE COLAB ALTERNATIVE</a>: Hello, Stable Diffusion enthusiasts!Make a free kaggle account and verify your phone number.You can download the notebook for free from my Discord channel: h...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/14029">Automatic1111  stable diffusion dreamartist not working. I am new here HELP 🙏 :) · AUTOMATIC1111/stable-diffusion-webui · Discussion #14029</a>: *** Error loading script: dream_artist_main.py Traceback (most recent call last): File &quot;D:\automatic1111\sd.webui\webui\modules\scripts.py&quot;, line 383, in load_scripts script_module = script_...</li><li><a href="https://github.com/castorini/daam">GitHub - castorini/daam: Diffusion attentive attribution maps for interpreting Stable Diffusion.</a>: Diffusion attentive attribution maps for interpreting Stable Diffusion. - castorini/daam</li><li><a href="https://github.com/stitionai/devika">GitHub - stitionai/devika: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective. Devika aims to be a competitive open-source alternative to Devin by Cognition AI.</a>: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective...</li><li><a href="https://github.com/lshqqytiger/stable-diffusion-webui-directml">GitHub - lshqqytiger/stable-diffusion-webui-directml: Stable Diffusion web UI</a>: Stable Diffusion web UI. Contribute to lshqqytiger/stable-diffusion-webui-directml development by creating an account on GitHub.</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="http://www.replicate.com">Replicate</a>: Run open-source machine learning models with a cloud API</li><li><a href="https://stable-diffusion-art.com/controlnet/">ControlNet: A Complete Guide - Stable Diffusion Art</a>: ControlNet is a neural network that controls image generation in Stable Diffusion by adding extra conditions. Details can be found in the article Adding</li><li><a href="https://azure.microsoft.com/en-us/free/ai-services/">Create Your Azure Free Account Today | Microsoft Azure</a>: Get started with 12 months of free services, 40+ services that are always free, and USD200 in credit. Create your free account today with Microsoft Azure.
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1220289632831279135)** (696 messages🔥🔥🔥): 

- **Unsloth AI Gearing Up for Multi-GPU Support**: The Unsloth AI team confirmed that multi-GPU support will eventually be available as an open-source feature, intending to allow for free use of Mixtral on platforms like Kaggle. The focus currently remains on launching Unsloth Studio (Beta).

- **Improving Data Curation for Fine-Tuning**: Unsloth AI is exploring the creation of an efficient platform UI for automatic data curation, targeting users who find data preparation for model fine-tuning challenging. This platform aims to address the data formatting and question-answer preparation steps.

- **Debate on Evaluation Frameworks and Data Quality**: There was a lengthy discussion about the importance of creating robust evaluation frameworks and the challenges of defining and obtaining high-quality data for model training. An important part is ensuring transparency and accuracy in benchmarks, like correcting datasets used, such as MMLU where 25% of examples had incorrect reference solutions.

- **Unwavering Community Support Despite Setbacks**: Despite previous instances of misinformation spreading in the community, Unsloth AI has gained notable traction and support, and their VRAM reduction technique has been acknowledged widely. Enthusiastic community members are expressing eagerness for upcoming multi-GPU support and other features.

- **Collaboration and Open Source Contributions Celebrated**: There was mention of projects such as OpenInterpreter's batch one selling out and their profits being redistributed to open-source contributors. This highlights a positive trend towards collaboration and reinvestment within the AI tools community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unstructured-io.github.io/unstructured/index.html">Unstructured 0.12.6 documentation</a>: no description found</li><li><a href="https://inflection.ai/inflection-2-5">Inflection-2.5: meet the world&#x27;s best personal AI</a>: We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.</li><li><a href="https://huggingface.co/stabilityai/stablelm-2-1_6b">stabilityai/stablelm-2-1_6b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18xz9it/augmentoolkit_easily_generate_quality_multiturn/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/ISTA-DASLab">ISTA-DASLab ( IST Austria Distributed Algorithms and Systems Lab)</a>: no description found</li><li><a href="https://x.com/DavidSHolz/status/1770697881160179786">Tweet from David (@DavidSHolz)</a>: @theashbhat making text</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth update: Mistral support + more</a>: We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...</li><li><a href="https://github.com/huggingface/trl/issues/862">Compute metrics for generation tasks in SFTTrainer · Issue #862 · huggingface/trl</a>: Hi, I want to include a custom generation based compute_metrics e.g., BLEU, to the SFTTrainer. However, I have difficulties because: The input, eval_preds, into compute_metrics contains a .predicti...</li><li><a href="https://github.com/InflectionAI/Inflection-Benchmarks">GitHub - InflectionAI/Inflection-Benchmarks: Public Inflection Benchmarks</a>: Public Inflection Benchmarks. Contribute to InflectionAI/Inflection-Benchmarks development by creating an account on GitHub.</li><li><a href="https://datadreamer.dev/docs/latest/datadreamer.steps.html#datadreamer.steps.RankWithPrompt">DataDreamer</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1220308710187077662)** (35 messages🔥): 

- **Installation Troubles with Unsloth AI**: A user experienced issues installing **Unsloth AI** from a nightly build using `pip`, encountering an error that there was no matching distribution found for the specified requirement. The problem referenced a specific extra named "kaggle-new".
  
- **Training on a Single GPU Card Failure**: Another encountered an error training a model specifically when restricting it to one GPU card. The error message indicated a mix of devices, causing a crash: *RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!*

- **CUDA Level Changes for Quantized Models?**: A user questioned if there had been changes at the CUDA level for 4bits quantized models after running into issues with Unsloth's solar-10.7b-bnb-4bit, which had previously worked on their machine.

- **VRAM Constraints with Solar Model**: It was observed that despite being a quantized model, which should require less VRAM, Unsloth's solar-10.7b-bnb-4bit was still possibly exceeding the available VRAM on a user's 15GB A4000 GPU, potentially causing out-of-memory issues.

- **Kernel Restarts Required to Avoid 32-bit Warnings**: A repeated requirement for kernel restarts to avoid warnings about 32-bit processing was noted, despite expectations that the referenced quantized models should not trigger such warnings. There is speculation that the machine in question might be running out of memory.

**Link mentioned**: <a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>: no description found

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1220265072111255613)** (92 messages🔥🔥): 

- **Switching to 16-bit LoRA Configured**: Members discussed changing LoRA settings, suggesting setting `load_in_4bit` to `false` or using parameters like `load_in_8bit` or `load_in_16bit`.

- **VRAM Consumption and Out-of-Memory Issues During Training**: One member reported Out-Of-Memory (OOM) errors during evaluation but not during training and was advised to try changing "adamw_8bit" to "paged_adamw_8bit" and reducing batch size to lower VRAM usage.

- **Saving and Loading Models and Tokenizers Locally**: A member figured out that to use `FastLanguageModel.from_pretrained()` effectively, both the model and tokenizer need to be saved in the same folder.

- **Potential Missing `protobuf` Dependency in Unsloth**: A member raised a concern that `protobuf` might be missing in a particular version of Unsloth, which was acknowledged but with uncertainty as to whether it was the case.

- **Unclear Model Choice for Physics, Math, and Engineering**: A member asked for advice on AI model selection suitable for high-level physics, mathematics, engineering, and Python, with recommendations to look at available resources like YouTube videos and articles such as [this one on Towards Data Science](https://towardsdatascience.com/fine-tune-google-gemma-with-unsloth-and-distilled-dpo-on-your-computer-ca1ce8828122).

- **Challenges with Unsloth Library Updates and Environment Management**: Multiple members experienced issues related to Unsloth updates, with suggestions to upgrade necessary libraries, while one described difficulties in environment management and the need for extensive dependency overhauls.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py#L488">bitsandbytes/bitsandbytes/autograd/_functions.py at main · TimDettmers/bitsandbytes</a>: Accessible large language models via k-bit quantization for PyTorch. - TimDettmers/bitsandbytes</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1220538856206504116)** (30 messages🔥): 

- **Showcasing Samantha Mistral Instruct 7B**: cognitivetech highlighted their work on [Samantha Mistral Instruct 7B](https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b_bulleted-notes), a model designed to summarize psychology books 2000 tokens at a time, and thanked the community for their support.
- **Gratitude for the Community's Guidance**: Acknowledgement was given for the help received in utilizing Unsloth notebooks and the community's assistance in answering questions related to fine-tuning models.
- **Troubleshooting Model Issues**: cognitivetech discussed experiencing issues with q4 quantization, suggesting it produced "garbage" results unlike the model `model-unsloth.Q8_0.gguf`, which worked flawlessly when summarizing books.
- **Upload of Working Quant Model**: After some discussion about troubleshooting the model, cognitivetech informed they would be uploading a working version of q8 to Hugging Face for others to check in about 20 minutes.
- **Community Collaboration and Testing**: There was a collaborative effort between cognitivetech and solobsd to test and run the models on platforms like GPT4All, sharing Ollama templates and discussing potential causes for issues encountered.

**Link mentioned**: <a href="https://huggingface.co/blog/cognitivetech/samantha-mistral-instruct-7b-bulleted-notes">Samantha Mistral Instruct 7b - Comprehensive Bulleted Notes</a>: no description found

  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1220382977733955705)** (14 messages🔥): 

- **Lightning Thunder Piques Interest**: A member shared a [link to Lightning Thunder](https://t.co/P6pvGJBugB), highlighting its potential to make PyTorch models faster by leveraging different hardware executors. They noted, however, it may not be directly helpful for Unsloth AI since it is built on Triton.

- **Confusion Over Unsloth Implementation**: Some members expressed concern that Lightning Thunder did not properly implement Unsloth, suggesting they could have consulted the Unsloth team for better integration.

- **Potential Misuse of Unsloth Kernels**: A member pointed out issues with Lightning Thunder's use of Unsloth kernels, like unnecessary copies and transpositions, highlighting that a consultation could have prevented this mishandling.

- **Call for Collaboration and Clarification**: Suggestions were made to reach out to the Lightning Thunder team to rectify mistakes and clarify the use of Unsloth in their presentations, emphasizing the importance of accurate comparisons in benchmarks.

- **Frustration Over Performance Comparison**: One member shared frustration through a [Twitter link](https://twitter.com/danielhanchen/status/1770863091296632921) regarding the inaccurate comparison that made Unsloth kernels look underperforming, urging for the presentation to reflect the correct implementation.

**Link mentioned**: <a href="https://t.co/P6pvGJBugB">GitHub - Lightning-AI/lightning-thunder: Source to source compiler for PyTorch. It makes PyTorch programs faster on single accelerators and distributed.</a>: Source to source compiler for PyTorch. It makes PyTorch programs faster on single accelerators and distributed. - Lightning-AI/lightning-thunder

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1220288094050451487)** (254 messages🔥🔥): 

- **OpenInterpreter Discord Channel Buzzing with Activity**: There is significant excitement as members discuss the OpenInterpreter Discord chatbot messages, with various time zones making it a challenge for some to stay awake for the ongoing discussions.
- **Launch Anticipation and Pre-Order Queries**: Members are sharing their enthusiasm for the 01 Light launch, asking about pre-orders and expressing hope for international shipping options beyond the current US-only availability.
- **Tech Enthusiast Community Rallies Behind Hardware Innovations**: Links to **3D print designs** for the 01 Light are shared, encouraging DIY enthusiasts to build their own language model computers, with **design files found at [Printables](https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-model)** and **GitHub for more info** at [GitHub - OpenInterpreter/01](https://github.com/OpenInterpreter/01).
- **Development and Safety Discussions Heat Up**: There's chatter about the **OpenInterpreter development process** and **safety measures**, with members curious about red-teaming initiatives and safeguards, directing others to the **OpenInterpreter/01** [GitHub repository](https://github.com/OpenInterpreter/01) for more details.
- **Community Collaboration and Questions Surrounding Windows Support**: Users are querying about running OpenInterpreter on Windows and if there will be any official Windows support, with no direct responses confirming such support provided in the conversation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/hellokillian/status/1757526563879587995?s=20).">Tweet from killian (@hellokillian)</a>: ..jesus  open interpreter&#39;s first vision model, piloting my 8gb M1 macbook. 100% offline.  this will be inside every computer in the world.</li><li><a href="https://www.amazon.com/dp/B06XT1Z9TF.">no title found</a>: no description found</li><li><a href="https://x.com/hellokillian?s=21&t=G6jp7iOBtkVuyhaYmaDb0w">Tweet from undefined</a>: no description found</li><li><a href="https://docs.openinterpreter.com/language-models/hosted-models/openai">OpenAI - Open Interpreter</a>: no description found</li><li><a href="https://kidger.site/thoughts/jaxtyping/">No more shape errors! Type annotations for the shape+dtype of tensors/arrays.</a>: TL;DR: you can explicitly use type annotations of the form def f(x: Float[Tensor, &#34;channels&#34;], y: Float[Tensor, &#34;channels&#34;]): ... to  specify the shape+dtype of tensors/arrays; declare...</li><li><a href="https://github.com/OpenInterpreter/01/blob/main/ROADMAP.md">01/ROADMAP.md at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/utils/count_tokens.py">open-interpreter/interpreter/terminal_interface/utils/count_tokens.py at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://x.com/altryne/status/1770835426384715803?s=46">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: https://twitter.com/i/spaces/1YpKkwdyWjdKj</li><li><a href="https://youtu.be/YxiNUST6gU4?si=fSBtR7Tw6WCvWNvN">Introducing Light 01: World&#39;s First Personal AI Assistant by Open Interpreter (Full Setup)</a>: In this video, we&#39;ll look at the OpenInterpreter Light 01 GitHub repository, a cutting-edge project that&#39;s revolutionizing how we interact with computers usi...</li><li><a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01: The open-source language model computer</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-mod">01 Light - Version 1: The World's First Language Model Computer by 01 | Download free STL model | Printables.com</a>: The 01 Project presents the 01 Light v1 | Download free 3D printable STL models</li><li><a href="https://github.com/patrick-kidger/jaxtyping">GitHub - patrick-kidger/jaxtyping: Type annotations and runtime checking for shape and dtype of JAX/NumPy/PyTorch/etc. arrays. https://docs.kidger.site/jaxtyping/</a>: Type annotations and runtime checking for shape and dtype of JAX/NumPy/PyTorch/etc. arrays. https://docs.kidger.site/jaxtyping/ - patrick-kidger/jaxtyping</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-model">01 Light - Version 1: The World's First Language Model Computer by 01 | Download free STL model | Printables.com</a>: The 01 Project presents the 01 Light v1 | Download free 3D printable STL models</li><li><a href="https://www.thingiverse.com/thing:6529845">01 Light - Version 1: The World&#039;s First Language Model Computer by openinterpreter</a>: Design Overview The 01 Light is the first-ever Language Model Computer. This first version of the 01 Light has a design that is sleek and ergonomic, and screen-less. Multiple internal recesses and pro...</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/hardware/light">01/hardware/light at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1220277189635735563)** (286 messages🔥🔥): 

- **Launch Party Anticipation**: Members express excitement for the [01 Developer Preview](http://openinterpreter.com/01). The 01 Light is a portable voice interface device for controlling a home computer, equipped with capabilities to see the screen, use apps, and learn new skills.
- **Build Your Own 01**: Community members discuss assembling their own 01 devices from a [Bill of Materials](https://github.com/OpenInterpreter/01/blob/main/hardware/light/BOM.md) and troubleshoot potential shipment issues to regions like India and the EU, suggesting the potential for localized community collaborations.
- **Setup Queries and Troubleshooting**: The chat addresses setup concerns for using 01 on various operating systems. One key solution for Windows users is to run with `poetry run 01 --client --client-type linux --server-host localhost`, indicating compatibility with Windows when using Linux client type settings.
- **Batch Updates and Shipping Concerns**: The community is informed of the fullness of pre-order batches and shared curiosity around shipment times. People inquire when batch 2 and subsequent batches will ship, with no committed date given.
- **Evolving Software Discussions**: Members ask about various features and usability of the software, including local vs. cloud operation, non-developer accessibility, API keys, and compatibility with languages like German. Concerns about software updates and battery life are also questioned.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://01.openinterpreter.com/getting-started/introduction">Introduction - 01</a>: no description found</li><li><a href="https://01.openinterpreter.com/getting-started/setup#captive-portal">no title found</a>: no description found</li><li><a href="https://x.com/hellokillian">Tweet from undefined</a>: no description found</li><li><a href="https://www.openinterpreter.com/01">The 01 Project</a>: The 01 Project is a voice interface for your home computer.</li><li><a href="https://01.openinterpreter.com/getting-">Introduction - 01</a>: no description found</li><li><a href="https://www.youtube.com/@MikeBirdTech/videos">Mike Bird</a>: A.I. engineering  </li><li><a href="https://tenor.com/view/her-theodore-joaquin-phoenix-scarlett-johannson-samantha-gif-5203383">Her Theodore GIF - Her Theodore Joaquin Phoenix - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://0ggfznkwh4j.typeform.com/to/WfuYTxMM?typeform-source=pcr08jir95k.typeform.com">Contact Us</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.</li><li><a href="https://tenor.com/view/shut-up-and-take-my-money-futurama-fry-take-my-money-money-gif-15195954">Shut Up And Take My Money Futurama GIF - Shut Up And Take My Money Futurama Fry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fxtwitter.com/OpenInterpreter/status/1770821439458840846">Tweet from Open Interpreter (@OpenInterpreter)</a>: Introducing the 01 Developer Preview.  Order or build your own today: http://openinterpreter.com/01  The 01 Light is a portable voice interface that controls your home computer. It can see your screen...</li><li><a href="https://github.com/OpenInterpreter/01/blob/main/hardware/light/BOM.md">01/hardware/light/BOM.md at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/01/issues">Issues · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://0ggfznkwh4j.typeform.com/to/WfuY">Discover Typeform, where forms = fun</a>: Create a beautiful, interactive form in minutes with no code. Get started for free.</li><li><a href="https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-model">01 Light - Version 1: The World's First Language Model Computer by 01 | Download free STL model | Printables.com</a>: The 01 Project presents the 01 Light v1 | Download free 3D printable STL models</li><li><a href="https://www.thingiverse.com/thing:6529845">01 Light - Version 1: The World&#039;s First Language Model Computer by openinterpreter</a>: Design Overview The 01 Light is the first-ever Language Model Computer. This first version of the 01 Light has a design that is sleek and ergonomic, and screen-less. Multiple internal recesses and pro...</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/hardware/light">01/hardware/light at main · OpenInterpreter/01</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

cyanidebyte: https://www.youtube.com/watch?v=Q_p82HtBqoc
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1220269548259901440)** (305 messages🔥🔥): 

- **LM Studio Local Document Support Inquiry**: A user inquired about the possibility of using local documents for Retrieval-Augmented Generation (RAG) with Anything-LLM and LMStudio support, but no subsequent answers were provided regarding the functionality.
- **Concerns Over LM Studio Multi-Model Issues**: Users reported facing issues with the new beta release of LM Studio failing to add multiple models and one experiencing excessive CPU usage despite offloading all layers to GPU, which was resolved by rebooting their system.
- **LM Studio Model Loading Errors After Update**: One user described a problem with LM Studio failing to recognize non-local model names, resulting in an error, with another user suggesting that loaded models now generate a static key name visible through the GET endpoint, which wasn't directly resolved within the shared messages.
- **Disappearing Icons in LM Studio's Playgrounds**: Another user experienced interface behaviour where model names were ejected from the UI when navigating between sections in LM Studio's playground. A workaround was mentioned, suggesting clicking the yellow "Reload" box only once when wanting to reload a model.
- **Image Analysis with Llava in LM Studio**: A user queried how to feed images for analysis to a llava model within LM Studio Chat AI, with the response indicating the necessity to drag and drop the image into the input box for the model to "see" it.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/roborovski/superprompt-v1">roborovski/superprompt-v1 · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/jobs.html">Redirecting...</a>: no description found</li><li><a href="https://huggingface.co/TheBloke/dolphin-2.7-mixtral-8x7b-GGUF">TheBloke/dolphin-2.7-mixtral-8x7b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/gptq-integration">Making LLMs lighter with AutoGPTQ and transformers</a>: no description found</li><li><a href="https://python.langchain.com/docs/expression_language/get_started">Get started | 🦜️🔗 Langchain</a>: LCEL makes it easy to build complex chains from basic components, and</li><li><a href="https://status.openai.com/">OpenAI Status</a>: no description found</li><li><a href="https://github.com/stitionai">stition</a>: stition has 3 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/stitionai/devika">GitHub - stitionai/devika: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective. Devika aims to be a competitive open-source alternative to Devin by Cognition AI.</a>: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective...</li><li><a href="https://github.com/kalomaze/koboldcpp/releases/tag/v1.57-cuda12-oldyield">Release Faster CPU Prompt Processing (v1.57, CUDA 12) · kalomaze/koboldcpp</a>: I have reverted the upstream llama.cpp change that causes the thread yielding to be conditional, instead, it always does it. This improves prompt processing performance for me on my CPU which has I...</li><li><a href="https://github.com/Nexesenex/kobold.cpp/releases/tag/v1.59d_b2254">Release Kobold.CPP_Frankenstein_v1.59d_b2254_4x3bits_SOTA · Nexesenex/kobold.cpp</a>: Kobold.CPP Frankenstein v1.59&#39;s source and .exe for Windows built with Openblas/Clblast/Vulkan (small .exe), and the same + Cublas (big .exe) :   based on LlamaCPP b2254 &amp; LostRuin&#39;s Kobol...</li><li><a href="https://github.com/caddyserver/caddy">GitHub - caddyserver/caddy: Fast and extensible multi-platform HTTP/1-2-3 web server with automatic HTTPS</a>: Fast and extensible multi-platform HTTP/1-2-3 web server with automatic HTTPS - caddyserver/caddy</li><li><a href="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio">GitHub - BBC-Esq/ChromaDB-Plugin-for-LM-Studio: Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode!</a>: Plugin that creates a ChromaDB vector database to work with LM Studio running in server mode! - BBC-Esq/ChromaDB-Plugin-for-LM-Studio</li><li><a href="https://github.com/czkoko/SD-AI-Prompt">GitHub - czkoko/SD-AI-Prompt: A shortcut instruction based on LLama 2 to expand the stable diffusion prompt, Power by llama.cpp.</a>: A shortcut instruction based on LLama 2 to expand the stable diffusion prompt, Power by llama.cpp. - czkoko/SD-AI-Prompt</li><li><a href="https://github.com/kyegomez/BitNet">GitHub - kyegomez/BitNet: Implementation of &quot;BitNet: Scaling 1-bit Transformers for Large Language Models&quot; in pytorch</a>: Implementation of &quot;BitNet: Scaling 1-bit Transformers for Large Language Models&quot; in pytorch - kyegomez/BitNet
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1220441138872455178)** (29 messages🔥): 

- **Exploring AI's Ethical Boundaries in Bank Security**: Members discussed a [Hugging Face space called 'Guardrails Arena'](https://huggingface.co/spaces/lighthouzai/guardrails-arena), where users interact with models to assess fictional bank security measures, revealing that some models consistently refuse sensitive information whereas others are more forthcoming.
- **Under the Hood of Guardrails**: For those interested in the technical details of the 'Guardrails Arena', links to the model's Python script and configuration settings are provided at [Guardrails Models Python Script](https://huggingface.co/spaces/lighthouzai/guardrails-arena/blob/main/guardrails_models.py) and [Prompts Configuration](https://huggingface.co/spaces/lighthouzai/guardrails-arena/blob/main/nemoguardrails_config/prompts.yml), offering insight into the AI's decision-making policies.
- **New Reasoning Model on the Horizon**: A new paper discussing 'Quiet-STaR' as a generalization to improve language models by generating rationales at each token is referenced, which has been translated into model form and can be viewed at the [Hugging Face repository for quietstar-8-ahead-GGUF](https://huggingface.co/dagbs/quietstar-8-ahead-GGUF) and in a [YouTube video](https://www.youtube.com/watch?v=9gdiqTJNeEc).
- **Human Oversight in AI-Driven Architecture**: A conversation highlighted that while AI might design a structurally sound building, human oversight is legally and ethically necessary due to the inherent risk of structural mistakes.
- **Complementary AI Workflow, not Replacement**: Members posit that AI should be used as a workflow accelerator, aiding in tasks like design and testing, rather than an outright replacement, emphasizing the constant need for human involvement and verification.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lighthouzai/guardrails-arena">Guardrails Arena - a Hugging Face Space by lighthouzai</a>: no description found</li><li><a href="https://tenor.com/view/dont-say-that-ever-again-diane-lockhart-the-good-fight-dont-say-that-never-say-that-again-gif-18052604895623551134">Dont Say That Ever Again Diane Lockhart GIF - Dont Say That Ever Again Diane Lockhart The Good Fight - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/lighthouzai/guardrails-arena/blob/main/nemoguardrails_config/prompts.yml">nemoguardrails_config/prompts.yml · lighthouzai/guardrails-arena at main</a>: no description found</li><li><a href="https://huggingface.co/papers/2403.09629">Paper page - Quiet-STaR: Language Models Can Teach Themselves to Think Before
  Speaking</a>: no description found</li><li><a href="https://huggingface.co/dagbs/quietstar-8-ahead-GGUF">dagbs/quietstar-8-ahead-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1220360925031305289)** (26 messages🔥): 

- **Symlink Troubles with LM Studio**: An update to **LM Studio version 0.2.17** caused models to stop loading, with symlinks that worked in previous versions no longer being recognized. Despite attempts at regenerating symlinks, users experienced *“Model with key Mistral/Hermes/Hermes-2-Pro-Mistral-7B.Q4_0.gguf not found.”* errors.

- **Language Reminder**: Discord members were reminded that **English is the primary language** of the server after a user posted a message in Chinese.

- **Channel Confusion**: There's been discussion suggesting a need for clearer guidance on where to post certain topics, as **feedback or help-related questions** are often posted in incorrect channels.

- **Feature Request for File Interaction**: A user expressed a desire to **chat with files** such as PDF, DOCX, or PNG, and was informed that chatting with PNG images is supported using a **Llava model**.

- **Summarizing Multiple PDF Documents**: In response to an inquiry about **summarizing multiple PDFs**, a member was directed towards a specific channel for model suggestions.

- **Download Speed Limiter Request**: A **download speed limiter feature** was requested to prevent large model downloads from monopolizing bandwidth. Discussion ensued about whether **OS-level settings** might be a better solution for bandwidth management.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1220353404170272818)** (98 messages🔥🔥): 

- **Cloud vs. Local AI Hardware**: A member expressed their preference for local hardware over cloud services for machine learning, citing cost-effectiveness and learning opportunities. Experimenting with AI on personal hardware allows companies to understand AI without hefty cloud service expenses.

- **Shifting IT Paradigms**: Members discussed the cyclical nature of centralized and decentralized computing, predicting that, following trends, on-premise AI servers may become preferred before shifting back to powerful decentralized AI PCs.

- **Security Concerns with AI Chatbots**: A discussion ensued about a security exploit that allows interception of encrypted AI chatbot tokens through a side channel attack. Despite the encryption, attackers can infer information about messages sent to users, indicating a potential vulnerability for services like OpenAI ([more detailed explanation here](https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/)).

- **The Quest for Efficient AI Development**: Conversation pivoted around the increasing complexities and expenses of AI development infrastructure. High-end hardware like GPUs, infiniband switching, and the need for massive power become economic and environmental concerns, with predictions about the future leaning towards SaaS solutions.

- **Choosing the Right Model and Specs for Personal Machines**: A member sought advice on which AI models to run on their M3 Pro with 18GB of RAM. It was advised to look for models that enable "Full GPU Offload Possible" and to expect limitations when working with higher-capacity models due to hardware constraints.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://evalplus.github.io/leaderboard.html">EvalPlus Leaderboard</a>: no description found</li><li><a href="https://ca.news.yahoo.com/hackers-spy-chats-almost-ai-133041191.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAA628vNFzyRtneQvrem1WDoac-lQ3-TX-faSAkTYTAnJC9GbR4hMplcovcWJLKYfRKzeKXGjwz5w4hkM4dBJp6XSEIgDvGir_0i8m4DEkXe2UOjpb_xrivCKUh4jSLjxoTviS1daIJ0mbC9fuYbZ8_kMXo_rApntCtJnL5pQsLa1">Hackers Can Spy on Your Chats With Almost Any AI, Experts Find</a>: Be careful what you tell AI chatbots, because apparently, it&#x27;s easy for hackers to figure out. &#x22;Currently, anybody can read private chats sent from ChatGPT and other services,&#x22; Yisroel ...</li><li><a href="https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/">Hackers can read private AI-assistant chats even though they’re encrypted</a>: All non-Google chat GPTs affected by side channel that leaks responses sent to users.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1220280552947712011)** (10 messages🔥): 

- **Seeking Clarity on Multimodal Usage**: A user expressed they've been too focused on getting multimodal LM Studio to work and lost sight of their original purpose for learning to use it.
- **Model Recommendations Passed Around**: Users discussed their experiences with different versions of Command-R models. A recommendation was made for second-state's q8 model.
- **Looking for Models with External Storage Capabilities**: A user expressed interest in a multi-model setup that allows a model to interact with external storage, like a text file or local redis instance, to improve model performance on complex tasks involving Golang and Hugo.
- **Technical Issues and Troubleshooting in LMStudio**: An individual shared their configuration settings for an unspecified model that led to abnormal behavior and another user suggested restarting LMStudio as a possible solution to similar issues they encountered.
- **Hardware Compatibility Queries Solved**: A user encountered an error when trying to run LM Studio on an AMD Rocm with an RX 570. Another user clarified that the RX 570 graphics card is too old to work with the ROCM build.
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1220273700452958260)** (10 messages🔥): 

- **Clarification on LM Studio's New Features**: LM Studio has launched a capability that supports **multi-model** use, which allows having several models in VRAM at once to compare and utilize the best model for a specific task through the LMS console or tools like autogen.

- **Assistance Request for Autogen Issue**: A user encountered a **TypeError in their Autogen script** indicating that `Completions.create()` got an unexpected keyword argument 'request_timeout'. They posted the error traceback and sought assistance in resolving the issue.

- **Code Review and Sensitive Data Caution**: Another user advised removing the API key from the user's `config_list` file to prevent potential scraping by bots, despite being told it's not an actual key. This is shared as a reminder to practice good security habits by not posting sensitive data in public forums.

- **Seeking Advice on VRAM Limitations**: A member inquired if 8GB of VRAM is considered low when they report that only one Language Model (LM) can run before exceeding the limit. They wondered if there were options to remove or increase this limit.
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1220537958793351220)** (2 messages): 

- **Ease Language Modeling with Instructor**: Members are advised to check out the [Instructor library](https://github.com/jxnl/instructor) on GitHub, which is designed to facilitate structured outputs for language model workflows. The library is highlighted as something that could simplify processes for users.
- **Special Fine-tuned OpenChat Success**: A member mentioned they have a special fine-tuned version of **OpenChat** that performs well and have successfully integrated it with the dolphin mistral fine-tune.
- **Just a Quick DM**: A brief note indicating a private message was sent to follow up on the conversation, presumably on one of the mentioned topics or tools.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.co">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://github.com/jxnl/instructor">GitHub - jxnl/instructor: structured outputs for llms</a>: structured outputs for llms . Contribute to jxnl/instructor development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1220367505231446138)** (23 messages🔥): 

- **Eject and Context Size Tweaks Required**: A member reported needing to engage "Eject" during loading, and then minimize context size to prevent *out-of-memory* issues while using model versions like Command-R, despite its support for large context.

- **Feedback on ROCm 0.2.17 Beta v3**: A link to the new **ROCm 0.2.17 Beta v3** was shared, including a [change log](https://files.lmstudio.ai/windows/0.2.17-ROCm-Beta-v3/beta/LM-Studio-0.2.17-Setup-ROCm-Beta-v3.exe) mentioning a potential fix for issues around **GPU offloading**.

- **Mixed Experiences with GPU Offloading**: Members discussed various experiences with GPU offloading on **ROCm**, noting issues like 100% CPU utilization and confusion possibly caused by **ZLUDA** taking precedence in system paths.

- **ZLUDA Interference With ROCm**: It was noted by the members that having **ZLUDA** installed and in the PATH may interfere with **ROCm** operations, potentially explaining high **CPU utilization** issues.

- **Stable Performance on AMD Hardware**: Several users reported successful and stable use of [ROCm 0.2.17 Beta v3](https://files.lmstudio.ai/windows/0.2.17-ROCm-Beta-v3/beta/LM-Studio-0.2.17-Setup-ROCm-Beta-v3.exe) on various **AMD GPUs**, with feedback ranging from "working fine" to observing substantial **GPU activity**.

**Link mentioned**: <a href="https://files.lmstudio.ai/windows/0.2.17-ROCm-Beta-v3/beta/LM-Studio-0.2.17-Setup-ROCm-Beta-v3.exe">no title found</a>: no description found

  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1220273067230629950)** (340 messages🔥🔥): 

- **Discord Users Discuss Perplexity and AI Differences**: Users engaged in discussions on the performance of different AI models, with comparisons between **Claude 3 Opus** and **Gemini** being frequent. Some stated Gemini sounds more human, while others expressed a preference for Opus or its Anthrophic's top model.
  
- **Tech Updates and Troubleshooting**: Participants discussed updates to various apps, challenges with the text input fields, and provided mutual aid for use on mobile versus PC. Some shared frustrations related to **iOS app updates** and features, such as desiring a darker midnight/black theme for better visual comfort.

- **Cloudflare Critique**: A user expressed dissatisfaction with Cloudflare's CAPTCHA challenges, especially when using VPNs for privacy, indicating it even affects users without privacy settings enabled.

- **Perplexity's Web Search and Image Generation Queries**: Queries on how **Perplexity AI** conducts web searches and image generation featured prominently. Users clarified that while on mobile, certain features like **Unlimited Claude3 Opus** might not be accessible, but images could be generated following specific instructions.

- **Discussions and Comparisons of Personal AI Models**: Users shared thoughts on various personal AI models such as **Inflection-2.5 and Pi.AI**, highlighting their strengths in conversational usage and voice models. Concerns about the future of these platforms, in light of talent loss, also surfaced.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://`,">no title found</a>: no description found</li><li><a href="https://inflection.ai/inflection-2-5">Inflection-2.5: meet the world&#x27;s best personal AI</a>: We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.</li><li><a href="https://www.businessinsider.com/neuralink-first-human-trial-patient-quadriplegic-elon-musk-x-2024-3">Neuralink reveals first human-trial patient, a 29-year-old quadriplegic who says brain chip is &#x27;not perfect&#x27; but has changed his life</a>: Elon Musk&#x27;s Neuralink reveals first human-trial patient, a 29-year-old quadriplegic who says brain chip is &#x27;not perfect&#x27; but has changed his life</li><li><a href="https://www.tradershub.ninja/">Tradershub Ninja</a>: no description found</li><li><a href="https://tenor.com/view/the-batman-no-selfpromo-no-self-promotion-batman-gif-24317098">The Batman No Selfpromo GIF - The Batman No Selfpromo No Self Promotion - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/thanos-talking-meme-thanos-talking-meme-thanos-speech-gif-1800590086203910493">Thanos Talking GIF - Thanos Talking Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/light-thme-light-dark-theme-gif-27389075">Light Thme Light GIF - Light Thme Light Dark Theme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/singularity/s/j7YBzKr3ql">Reddit - Dive into anything</a>: no description found</li><li><a href="https://analyticsindiamag.com/perplexity-is-most-likely-a-google-search-wrapper/">Perplexity is Most Likely a Google Search Wrapper</a>: A user posted on the LocalLLaMA thread of Reddit that Perplexity summarises the content from the top 5-10 results from Google Search.
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1220283301126733895)** (19 messages🔥): 

- **Largest Planet Knowledge Inquiry**: A post shared a [Perplexity AI search link](https://www.perplexity.ai/search/The-largest-planet-Tmh863TjTp66BfMzKVBQEw#0) relating to the largest planet, indicating research or a discussion may have taken place about this topic.
- **Boosting Japanese LLM Development**: A user pointed to [Japanese language model development](https://www.perplexity.ai/search/japanese-llm-development-5GPNniNXRXy2UqnfB8sjuA), suggesting a focus or interest in this area.
- **A Question of Time**: A member shared a [Perplexity AI search link](https://www.perplexity.ai/search/Combien-de-temps-kewLnab9THOciQpv4Y0Ttg) concerned with the French phrase "Combien de temps," potentially signifying a language-related query.
- **GPT-5 Release Rumor Mill**: Curiosity or rumors about the release of GPT-5 was indicated through a shared [Perplexity AI link](https://www.perplexity.ai/search/GPT5-release-rumors-NfNcO6yfRG..vG9acBEIYQ).
- **Embracing Linux as a macOS Vet**: One user narrated their switch from macOS to Linux for AI/ML studies, utilizing [Perplexity AI](https://www.perplexity.ai/search/What-are-some-AmEb7c4LTn2GNCeiBSJmYw) as an aid in this learning journey, expressing satisfaction with MX Linux.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1220383326347722834)** (26 messages🔥): 

- **Perplexity API vs UI Citations**: A member inquired about why the API does not provide sources and citations like the Perplexity UI, hinting at a potential feature addition.
- **Token Limits Gaffe**: A user faced a `BadRequestError` when attempting a request with a prompt of **6621 tokens** and an output of **9900 tokens**, exceeding Perplexity's **16384 token limit**. They were puzzled by how to adjust their API calls accordingly.
- **Resume Analyzer Challenge**: The member encountering token limits is building a resume analyzer/builder as an AI practice project, indicating they are fairly new to the field.
- **Token Counting Tricks**: Another community member pointed the user to the [Perplexity documentation](https://docs.perplexity.ai/reference/post_chat_completions) to check token counts accurately for their AI queries.
- **Seeking Clarification on Usage**: The user explored how to limit user prompts and was informed that it entirely depends on the content length, which was helpful advice for the resume project they are working on.

**Link mentioned**: <a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: no description found

  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1220288007031095397)** (369 messages🔥🔥): 

- **Dataset Dilemma - Seeking the "mypersonality" Dataset**: Members were discussing the **"mypersonality"** dataset from Facebook and its utility in predicting the personality of authors based on text. The dataset's accessibility was in question, with one member indicating knowledge about it due to research requirements.

- **Hugging Face Implementations Under Scrutiny**: Extensive conversation unfolded over potential issues with embedding implementations in the [Hugging Face diffusers library](https://github.com/huggingface/diffusers/issues/7365), with code snippets and corrections shared among members. Concerns were raised about the correct pooling method for text embeddings, with suggestions to correct the existing code and improve model performance.

- **Dataset Recovery and Alternatives Explored**: The status and future of the **LAION-5B** dataset post-removal was a topic of discussion, with new datasets like Datacomp and DFN touted as alternatives given EU legislation. Skepticism regarding **LAION's** ability to fully clear legal hurdles and republish their datasets was expressed, implying that the datasets may remain unpublished.

- **Push for OpenAI Code Transparency**: Members discussed the importance of open-sourcing training code for the advancement of AI and expressed anticipation for potential openness with future models like **SD3**, despite the setbacks in previous versions.

- **Training Technique Discussions and Improvements**: Debates about **finetuning the text encoder** in models like SD2.x revealed a consensus that drastic text encoder modifications might not be necessary. The pivotal tuning method, once a topic of debate, was acknowledged as an "advanced" method now incorporated into **Diffusers'** official training scripts.

- **Skepticism Toward AI Sabotage Claims**: One member expressed doubt that researchers who raise concerns about datasets containing sensitive material, like CSAM, are sincere in their intentions, speculating instead that their actions could be aimed at hindering AI progress.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lifehacker.com/tech/its-not-safe-to-click-links-on-x">It's Not Safe to Click Links on X</a>: When someone posts a link on X, the site generates a link preview. But reportedly, this system can be tricked, and bad actors can redirect you to malicious sites from a falsely advertised link preview...</li><li><a href="https://www.thewrap.com/openai-to-meet-with-hollywood-studios-and-talent-agencies-next-week-on-sora-integration/">OpenAI to Meet With Hollywood Studios, Talent Agencies</a>: OpenAI is taking meetings next week with the Hollywood studios &amp; talent agencies to pitch filmmakers on the integration of Sora.</li><li><a href="https://huggingface.co/blog/xingxm/svgdreamer">SVGDreamer: Text Guided Vector Graphics Generation with Diffusion Model</a>: no description found</li><li><a href="https://tenor.com/bcuEi.gif">Annoyed Fuck GIF - Annoyed Fuck Frustrated - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L99>">transformers/src/transformers/models/clip/modeling_clip.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://github.com/huggingface/diffusers/issues/7365">Provided pooled_prompt_embeds is overwritten via prompt_embeds[0] · Issue #7365 · huggingface/diffusers</a>: diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py Line 386 in 25caf24 pooled_prompt_embeds = prompt_embeds[0] Simple fix: pooled_prompt_embeds = prompt_embeds[0]...</li><li><a href="https://github.com/openai/CLIP/blob/main/clip/model.py#L364>">CLIP/clip/model.py at main · openai/CLIP</a>: CLIP (Contrastive Language-Image Pretraining),  Predict the most relevant text snippet given an image - openai/CLIP</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling>">Model outputs</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1220349817809670234)** (4 messages): 

- **Scaling up Image Scales**: A paper discussed at [arXiv](https://arxiv.org/pdf/2403.13043.pdf) suggests improving model performance by using **multiple scales** of an image.

- **Using Time to Encode Images**: Another paper from [arXiv](https://arxiv.org/pdf/2403.13802.pdf) introduces a method of encoding images with **6 times the number of timestamps**, employing different zig-zags, which some believe might be a workaround rather than an elegant solution.

- **Fractals in the Spot**: Continuous fractal space-filling curves were humorously referenced in a discussion, implying their potential in addressing current encoding methods.

- **Peering into the Future**: A tweet from [Bindu Reddy](https://twitter.com/bindureddy/status/1770934470373421522?t=5dLWWD7d9PN0C4ZjHAY_Tw&s=19) was shared as an indicator of a forward-looking development, though the specific content of the tweet wasn't disclosed in the message.
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1220329148577353728)** (21 messages🔥): 

- **Debating "Extended Mind" Concept**: Members discussed the concept of "**Extended Mind**," referencing [a tweet by Phoebe Klett](https://twitter.com/KlettPhoebe/status/1770480361656533449). Interest was expressed in porting it to **Mistral** for easier accessibility.
- **Understanding the Depth of Extended Mind**: One member mentioned that Extended Mind seemed akin to an **associative memory**, where a separate database holds information that attention can call upon, similar to memory and tool use.
- **Clarifying the Mechanism of Extended Mind**: Discussion clarified that the **Extended Mind** involves storing vectors and fetching the top k during the forward pass, emphasizing that it's more about selecting aspects of associative memory than tools.
- **Speculating on the Integration of Tools through Extended Mind**: There was talk about future experimentation with **Extended Mind**, speculating on integrating different tools more deeply and exploring its potential for impacting memory and reasoning.
- **Identifying Extended Mind's Potential and Challenges**: Members discussed the need for **weighed attention** in the Extended Mind approach, where the system must learn when to focus on memories. The concept's relationship with **memorizing versus reasoning** was also mentioned as a point of interest.
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1220271708523728926)** (5 messages): 

- **Greetings and Salutations**: A new member was welcomed with open arms into the community.
- **Hardware Struggles to Keep Up With AI Software**: A member commented on the challenge of running **oversized param models locally** without quantization and speculated that hardware would eventually catch up with the advancements in software.
- **LLaVA Model Fine-tuning Tutorial Shared**: A link to a [YouTube video](https://www.youtube.com/watch?v=21Tc92g15pM) was shared, which provides instructions on how to fine-tune the **LLaVA model** and includes various topics such as multimodal learning and deep learning.
- **New Open Source AI Device Introduced**: A member was excited to share a [twitter post](https://twitter.com/OpenInterpreter/status/1770821439458840846) about a new open source AI device, expressing anticipation to see it powered by Nous Research models.
- **Appreciation for the Open Source AI Device Project**: Recognition and appreciation were shown for Killian and his team's progress on the open source AI device mentioned in the previous tweet.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=21Tc92g15pM">Finetune MultiModal LLaVA</a>: This video explains how to fine-tune llava model#llm #ml #ai #deeplearning #largelanguagemodels #python https://wandb.ai/byyoung3/ml-news/reports/How-to-Fine...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1220279110694076436)** (23 messages🔥): 

- **Unlocking the Secrets of Synthetic Datasets**: The Hugging Face blog outlines generating a vast synthetic dataset, **Cosmopedia**, to mirror [Phi-1.5](https://arxiv.org/abs/2309.05463). The post highlights the shift from costly human-annotated data to synthetic datasets, with [Cosmopedia standing as a testament](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) to this trend in machine learning.

- **The Devil is in the Detail of Prompt Engineering**: Generating **Cosmopedia** didn't rely on heavy-duty GPUs but instead on detailed prompt engineering. The blog post reveals that time investment in prompt crafting was a significant part of the task.

- **Quiet-STaR Claims to Enhance Text Interpretation**: [Quiet-STaR](https://arxiv.org/abs/2403.09629), an extension of STaR, is proposed as a technique where language models learn to generate explanations for each token, thereby improving their text predictions. The abstract of the paper points to the potential for LMs to infer unstated rationales in arbitrary texts.

- **OpenInterpreter's Vision for AI Devices**: A new device called **01 Light** was introduced via a [tweet](https://x.com/OpenInterpreter/status/1770821439458840846?s=20) and promises to be a portable voice interface that can control a computer and its applications. The creators emphasize its open-source nature and the potential for users to build their own or utilize an upcoming app for remote control.

- **Debate on Necessity of Hardware for OpenInterpreter's 01**: Conversations surround the 01 Light's function as merely a "glorified microphone" with some members noting the hardware is optional and that one can [use the system on a computer for free](https://openinterpreter.com/01). Despite initial skepticism, there is recognition for the benefits of the open-source project and its software.

- **Nous Models on Kubernetes?**: A user questions the integration of Nous models within Kubernetes and shares a desire for an easy installation process akin to an example provided in a [SideroLabs tweet](https://twitter.com/SideroLabs/status/1771207304748167445). No further information is provided about the Nous models or their Kubernetes compatibility.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/cosmopedia">Cosmopedia: how to create large-scale synthetic data for pre-training Large Language Models</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is imp...</li><li><a href="https://x.com/OpenInterpreter/status/1770821439458840846?s=20">Tweet from Open Interpreter (@OpenInterpreter)</a>: Introducing the 01 Developer Preview.  Order or build your own today: http://openinterpreter.com/01  The 01 Light is a portable voice interface that controls your home computer. It can see your screen...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1220287620970446848)** (126 messages🔥🔥): 

- **Struggling with Embedding Models**: A user experienced issues when finetuning an embeddings model, citing problems with *BatchAllTripletLoss* from *Sentence Transformers* where eval scores weren't changing, indicating that the model wasn't learning. Additionally, using Angle loss pushed both positive and negative samples further from the query.

- **In the Quest for Summarization**: One user sought research papers to test a new summarizer, requesting others to provide documents. This led to a discussion involving a paper on Quiet-STaR (a generalization of STaR) where LMs generate rationales at each token to explain future text.

- **Chat about Chatbots**: Members discussed the integration of an existing repository, llm_steer, with an interface for interacting with LMs through "activation vectors". There was also a debate over the effectiveness of logical reasoning and planning in LMs, particularly when employing various methods like cursed model merging.

- **Open Source RAG Platforms and Benchmarks**: Users shared links to their projects, such as an open-source platform for RAG applications, and discussed benchmarks of models like Mistral-Evolved-11b-v0.1, commenting on their performance improvements.

- **Exploring AI and Hardware**: Some members questioned the practicality of AI-related hardware releases like the 'Open Interpreter's 01 Lite', while others hinted at the potential for "mind to text" technology that can interpret internal speech through EMG sensors on the neck. Some users imagine future possibilities like interacting with AI using gestures or direct brain interfaces.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is imp...</li><li><a href="https://tenor.com/view/spongebob-why-why-why-why-why-why-why-why-why-why-why-why-why-gif-25252239">Spongebob Why Why Why Why Why Why Why GIF - Spongebob Why Why Why Why Why Why Why Why - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/system_prompt.md">Abstractions/abstractions/goap/system_prompt.md at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.</li><li><a href="https://hamel.dev/blog/posts/prompt/">- Fuck You, Show Me The Prompt.</a>: Quickly understand inscrutable LLM frameworks by intercepting API calls.</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/spatial.py#L267">Abstractions/abstractions/goap/spatial.py at main · furlat/Abstractions</a>: A Collection of Pydantic Models to Abstract IRL. Contribute to furlat/Abstractions development by creating an account on GitHub.</li><li><a href="https://youtu.be/Q_p82HtBqoc">Open Interpreter&#39;s 01 Lite - WORLD&#39;S FIRST Fully Open-Source Personal AI AGENT Device</a>: 01 Lite by Open Interpreter is a 100% open-source personal AI assistant that can control your computer. Let&#39;s review it and I&#39;ll show you how to install open...</li><li><a href="https://gist.github.com/fullstackwebdev/4f8fc4931bd4dfba4231c8caf578e15e">graph_workflow.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/embed4all2graph_01.py">scratchTHOUGHTS/embed4all2graph_01.py at main · EveryOneIsGross/scratchTHOUGHTS</a>: 2nd brain scratchmemory to avoid overrun errors with self. - EveryOneIsGross/scratchTHOUGHTS</li><li><a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/w2v2graph_01.py">scratchTHOUGHTS/w2v2graph_01.py at main · EveryOneIsGross/scratchTHOUGHTS</a>: 2nd brain scratchmemory to avoid overrun errors with self. - EveryOneIsGross/scratchTHOUGHTS</li><li><a href="https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector">Steering GPT-2-XL by adding an activation vector — LessWrong</a>: Prompt given to the model[1]I hate you becauseGPT-2I hate you because you are the most disgusting thing I have ever seen. GPT-2 + &quot;Love&quot; vectorI hate…</li><li><a href="https://arxiv.org/abs/2310.01405">Representation Engineering: A Top-Down Approach to AI Transparency</a>: In this paper, we identify and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuros...</li><li><a href="https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steer">Steering GPT-2-XL by adding an activation vector — LessWrong</a>: Prompt given to the model[1]I hate you becauseGPT-2I hate you because you are the most disgusting thing I have ever seen. GPT-2 + &quot;Love&quot; vectorI hate…
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1220271417439027243)** (21 messages🔥): 

- **In Search of Quantization Support**: A member mentioned **quantization** for a language model and suggested that another member might have insights. This second member acknowledged trying quantization and noted more research was needed to make it work.
- **Collaboration on Quantization**: Upon request, a member shared a repository named **AutoAWQ** for 4-bit quantization. However, they specified it was an outdated version and invited others to attempt fixing it: [GitHub - casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ/tree/striped_hyena).
- **Anticipation for NousForge**: Members indicated that **NousForge** is not yet available after someone inquired about its release while implicitly mentioning their discovery of the chat through Google.
- **Debating Few-Shot Prompts in Instruction SFT**: A member questioned the commonality and benefit of including few-shot prompts in instruction SFT (supervised fine-tuning) datasets, another member responded negatively on the commonality and the initial member later found a related discussion thread without results reported yet.
- **Theoretical Grounds for Causal Masking Questioned**: A member asked if causal masking in attention mechanisms had theoretical justification or was merely an engineering convenience. Another participant highlighted the necessity of masking for the model to learn **next token prediction**.

**Link mentioned**: <a href="https://github.com/casper-hansen/AutoAWQ/tree/striped_hyena">GitHub - casper-hansen/AutoAWQ at striped_hyena</a>: AutoAWQ implements the AWQ algorithm for 4-bit quantization with a 2x speedup during inference. Documentation: - GitHub - casper-hansen/AutoAWQ at striped_hyena

  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1220535122093408356)** (3 messages): 

- **Potential Improvement for Obsidian**: A link to a [Twitter post by Baifeng Shi](https://twitter.com/baifeng_shi/status/1770643896437240052) was shared, suggesting it could improve **Obsidian**.
- **Affirmation of Obsidian Enhancement**: A member acknowledged that the content in the shared link would indeed Enhance **Obsidian**.
- **Exploration of Implementation**: A member expressed intent to attempt the implementation suggested for **Obsidian** improvement.
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1220271385264394292)** (38 messages🔥): 

- **LanceDB Gaining Traction with Developers**: Gabriel_syme expressed enthusiasm for **LanceDB**, highlighting its **speed**, **ease of use**, and the ability to perform **hybrid search queries** with generative interfaces like SQL-type queries. In contrast, Iriden discussed using **Polars** for traditional queries despite its challenging syntax when used with language models.
  
- **Awaiting Better Integration for Data Tools**: There was a mention of **Polars**, awaiting better integration, and noting that LanceDB and Polars can swap data, but developers need to do the integration manually. Furthermore, gabriel_syme considered potential cloud-native capabilities of Polars.

- **Managed Cloud Solutions a Possibility**: Iriden highlighted work on a **FastAPI/Streamlit** app that allows parquet file uploads and Polars expressions, mentioning that once deployed to modal.com, it could serve as a managed cloud solution.

- **Sharing Development Work on GitHub**: Iriden shared a [GitHub repository](https://github.com/Neural-Dragon-AI/Cynde) featuring code for running GPT asynchronously over **Polars frames** and machine learning models that use embeddings. This repo aims to integrate Semantic Wisdom with Predictive Models.

- **Parenting Discussions Amongst Developers**: There was a brief, light-hearted exchange about parenthood, with triggerhappygandhi offering congrats and talking about the scarcity of parents on Discord, and denovich responding with personal experience about the biological effects of becoming a parent.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lancedb.github.io/lancedb/basic/">Quick start</a>: no description found</li><li><a href="https://github.com/Neural-Dragon-AI/Cynde">GitHub - Neural-Dragon-AI/Cynde: Integrating Semantic Wisdom with Predictive Models</a>: Integrating Semantic Wisdom with Predictive Models - Neural-Dragon-AI/Cynde
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1220337261091094569)** (213 messages🔥🔥): 

- **RAG vs. Agent Approaches**: Some members debated the efficacy of **Retrieval-Augmented Generation (RAG)** versus agent-based models. It was argued that RAG is a less capable approach, merely a band-aid for missing knowledge, whereas agent-based models with tool use and reflection are more robust, even though RAG may appear simpler to implement.

- **FastChat Format Frictions**: There's a discrepancy in the formatting of **FastChat's alpaca** model prompts, with concerns about how it diverges from Stanford's alpaca format. One member highlighted the difference and the potential need for a pull request to correct FastChat's format for consistency.

- **Galore Optimizer Gains**: Discussion about the new **Galore optimizer** was positive, noting its smooth setup and significant VRAM savings for fine-tuning large language models. It utilizes low-rank matrices of the gradient for each full model parameter and allows for full parameter tuning with considerably less memory usage.

- **GPT-3.5 Performance Persistence**: Participants in the channel expressed interest in **GPT-3.5**, asking about the performance and inference times of various model sizes and configurations. One user noted suboptimal inference speeds when running locally on a Mac due to the privacy constraints of handling patient data.

- **Dataset Discussions**: There was a brief exchange about dataset curation and formatting. It covered different types of datasets and their respective formats for model training, specifically between *sharegpt* and *chatml* formats, with confirmations on how Axolotl interprets and processes these datasets for model consumption.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L550>">FastChat/fastchat/conversation.py at main · lm-sys/FastChat</a>: An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and Chatbot Arena. - lm-sys/FastChat</li><li><a href="https://github.com/jiaweizzhao/GaLore/issues/6">Third-party benchmark · Issue #6 · jiaweizzhao/GaLore</a>: Hello, thank you very much for such excellent work. We have conducted some experiments using Llama-Factory, and the results indicate that Galore can significantly reduce memory usage during full pa...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1220479137823592480)** (6 messages): 

- **Galore merges with the Transformers**: [Galore optimizer](https://github.com/huggingface/transformers/pull/29588) has been merged into the **Hugging Face Transformers** library, exciting members who anticipate its integration.
- **Technical Assistance Required**: A member reported a **TypeError** when running an example from "examples/openllama-3b/qlora.yml" related to an unexpected keyword argument 'seq_len.' Another member redirected the request for help to a specific channel for better assistance.

**Link mentioned**: <a href="https://github.com/huggingface/transformers/pull/29588">FEAT / Optim: Add GaLore optimizer by younesbelkada · Pull Request #29588 · huggingface/transformers</a>: What does this PR do? As per title, adds the GaLore optimizer from https://github.com/jiaweizzhao/GaLore Fixes: #29512 This is how I am currently testing the API: import torch import datasets from ...

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1220390639548633179)** (10 messages🔥): 

- **The Text Classification Debate**: A member questioned the common practice of fine-tuning a Language Model (LLM) for text classification by teaching it to output class names as text rather than adding a classifier head on top. A possible reason given for this approach was the flexibility it offers, such as enabling the training on *chain of thoughts*.

- **Tinkering with Model Parameters**: There was a query on how to adjust all parameters in *galore*, with particular mentions of `-mlp` and `self_attn`. It wasn't clear from the messages whether the member resolved their issue.

- **Training Mixtral for Coding Assistance**: A user asked for guidance on training and fine-tuning a Mixtral-7B model to be a coding assistant with documentation for tools like *runpod* and *python*. They inquired about the necessary tools, IDEs, and concepts for training the model on personal hardware.

- **PyTorch and Gema**: One member inquired whether *gema* is still not recommended (*a no-no*) on PyTorch.

- **Troubleshooting Preprocessing Error**: A member reported an error related to `KeyError: 'instruction'` while preprocessing data using an `axolotl` preprocessing script, and shared a snippet from their configuration file and data. No solution was provided in the message history.
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1220319698483281991)** (120 messages🔥🔥): 

- **Automatic Model Merging Breakthrough**: A member shared [Hardmaru's new paper](https://arxiv.org/abs/2403.13187) on *automated evolutionary algorithms for foundation model merging*. They discussed it as a unique way of combining diverse open-source models to enhance model performance without extensive training.
  
- **Paris AI Community's Buzz Over Meetups**: There was a lively conversation among members about the AI community in Paris, France. Some shared experiences of recent meetups, while others expressed interest in attending future gatherings like the [Paris Retrieval Augmented Generation group](https://www.meetup.com/fr-FR/paris-retrieval-augmented-generation-group/) meeting, emphasizing the region's active tech community.

- **Model Scaling Clarified**: One member inquired about how models on Hugging Face, such as `cosmo-1b`, are downscaled from larger models like `llama`. Another member [explained](https://github.com/HuggingFaceTB/cosmo-1b) that smaller models are not fine-tuned but are instead separate architectures trained from scratch with scaled-down parameters.

- **Video Understanding AI Tools Spotlight**: A discussion about tools for video analysis led to several recommendations, including [Video Mamba](https://huggingface.co/blog/vladbogo/video-mamba) and [Twelve Labs](https://www.twelvelabs.io/), which enable advanced video understanding with foundational models.

- **Growing Interest in Open Source AI Platforms**: A member pointed out the project JanAI, an open-source alternative to LM Studio that has gained attention on Reddit. Another clarified details about the partial open-sourcing plans for LM Studio after a discussion on the transparency of AI platforms emerged.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.twitch.tv/georgehotz">georgehotz - Twitch</a>: georgehotz streams live on Twitch! Check out their videos, sign up to chat, and join their community.</li><li><a href="https://videodb.io">VideoDB</a>: Build intelligent applications on all types of video with 2 simple lines of code. Built by developers. For developers.</li><li><a href="https://arxiv.org/abs/2403.13187">Evolutionary Optimization of Model Merging Recipes</a>: We present a novel application of evolutionary algorithms to automate the creation of powerful foundation models. While model merging has emerged as a promising approach for LLM development due to its...</li><li><a href="https://helixml.substack.com/p/how-we-got-fine-tuning-mistral-7b">How we got fine-tuning Mistral-7B to not suck: Helix Project Report, Feb 2024</a>: Announcing Helix v0.5 with improved text fine-tuning and OpenAI API support 🎉</li><li><a href="https://huggingface.co/blog/vladbogo/video-mamba">VideoMamba: State Space Model for Efficient Video Understanding</a>: no description found</li><li><a href="https://x.com/__tinygrad__/status/1770112124871979095">Tweet from the tiny corp (@__tinygrad__)</a>: Very few people have succeeded at building these machines. There&#39;s a few areas of complexity.  1) PCI-E AER errors. It&#39;s very hard to get reliable PCI-E extensions. We had to have custom cable...</li><li><a href="https://x.com/theinformation/status/1770183406640373901?s=61">Tweet from The Information (@theinformation)</a>: Perplexity AI, an AI–powered search engine, has become a Silicon Valley startup darling for taking on Google.  It&#39;s also quietly using Google&#39;s data.  https://www.theinformation.com/articles/a...</li><li><a href="https://x.com/maximelabonne/status/1767124527551549860?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Maxime Labonne (@maximelabonne)</a>: ♾️AutoMerger  I made a neat little tool to automatically merge models on @huggingface.  It already created a few competitive models during the weekend. Here&#39;s how it works. 🧵  🪟 Space: https://h...</li><li><a href="https://x.com/__tinygrad__/status/1770510742007271545">Tweet from the tiny corp (@__tinygrad__)</a>: @luka_emon When I started, I didn&#39;t understand where the problems lie with AMD. I thought it was the driver, it&#39;s not. tinygrad is now submitting AQL queues directly to the GPU.  It&#39;s the ...</li><li><a href="https://x.com/esotericcofe/status/1770842634229014949?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Nucleus☕️ (@EsotericCofe)</a>: it&#39;s over</li><li><a href="https://x.com/davidsholz/status/1770601982488912281?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from David (@DavidSHolz)</a>: I&#39;d love to fund the research & creation of an open-source text diffusion model in the 7b class (open to hybrid diffusion/AR). Anyone interested in working on this? Open to grants, part-time or fu...</li><li><a href="https://www.twelvelabs.io/">Multimodal AI that understands videos like humans</a>: Bring human-like video understanding to any application, whether you have terabytes or petabytes of video</li><li><a href="https://jan.ai/">Jan | Rethink the Computer</a>: Jan turns your computer into an AI machine by running LLMs locally on your computer. It&#x27;s a privacy-focus, local-first, open-source solution.</li><li><a href="https://www.youtube.com/watch?v=cvOpX75Kz4M">Deep dive: model merging</a>: Model merging is an increasingly popular technique that makes it possible to add or remove capabilities to transformer models, without the need for any addit...</li><li><a href="https://github.com/simonw/files-to-prompt">GitHub - simonw/files-to-prompt: Concatenate a directory full of files into a single prompt for use with LLMs</a>: Concatenate a directory full of files into a single prompt for use with LLMs - simonw/files-to-prompt</li><li><a href="https://github.com/stitionai/devika?tab=readme-ov-file">GitHub - stitionai/devika: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective. Devika aims to be a competitive open-source alternative to Devin by Cognition AI.</a>: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective...</li><li><a href="https://www.meetup.com/fr-FR/paris-retrieval-augmented-generation-group/">Paris RAG User Group (Retrieval Augmented Generation) | Meetup</a>: Bienvenue à Paris RAG!Nous sommes une communauté de professionnels et enthousiastes intéressé par RAG et les techniques adjacentes permettant d&#x27;augmenter des modèles de language large et des IA!=...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1220814514711167046)** (4 messages): 

- **Podcast Release with Adept Insights**: A new podcast episode is up featuring an essay with insights into *OpenAI*, *Google*, and **Adept**. The announcement was accompanied by a [Twitter link](https://twitter.com/swyx/status/1771255525818397122).
- **Team Effort Acknowledged for Adept Podcast**: In preparation for the **Adept podcast**, thanks were given to a member for their assistance, despite not covering all the questions due to time constraints.
- **AI In Action: Llama.cpp**: An AI-focused event titled *AI In Action* was about to start, showcasing **Llama.cpp** with a [Discord channel link](https://discord.com/channels/822583790773862470/1200548371715342479) provided for live participation.
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1220766447270105088)** (10 messages🔥): 

- **Speaker Rights Not Granted**: Members noted that they do not have **speaker rights** in the Discord channel.

- **Zoom to the Rescue**: One member mentioned creating a **Zoom room** as an alternative to communicate since Discord speaker rights were unavailable.

- **Tight Schedule**: A participant communicated having a **hard stop at 1245**, indicating limited time availability for discussion.
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1220824233127706638)** (92 messages🔥🔥): 

- **Discovering the Identity of Slono**: A revelation came to light that a user known as "slono" might not actually go by that name. Despite the surprise, a Spotify link of slono's music was shared to the group, showcasing a style meant to capture *the elusive atmosphere of long nights coming to an end* ([Listen to slono's music](https://open.spotify.com/artist/1rWeYVkrGXRqaD8e0kwMbc?si=xu1E7Di8T_OUpQvT46f-BA)).

- **The Padding Philosophy**: Members humorously discussed the "pad and pray" method as an approach when the "math doesn't math" in tensor dimensions, suggesting that dimensions should perhaps be more strictly managed or enforced IDE-side, like types in Python.

- **Llama.cpp Capabilities and UI Challenges**: One user suggested that llama.cpp could potentially utilize GPU processing. There was also feedback regarding the suboptimal Discord mobile UI, especially the inability to minimize the camera during use.

- **Understanding Transformative Models**: A conceptual discussion unfolded about transformer models being seen as weighted tensors with adjustable weights and graph operations. This led to a member sharing a visualization link from bbycroft.net about how these models work ([LLM Visualization](https://bbycroft.net/llm)).

- **Emergent Discussion on Codebases and Music Models**: There was a brief touch on the learning curve involved with reading large codebases efficiently and anticipation for a future discussion about music generation models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bbycroft.net/llm">LLM Visualization</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.00789">Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces</a>: Attention mechanisms have been widely used to capture long-range dependencies among nodes in Graph Transformers. Bottlenecked by the quadratic computational cost, attention mechanisms fail to scale in...</li><li><a href="https://tenor.com/view/friends-bestfriends-yep-bff-gif-4566644">Did We Just Become Best Friends? GIF - Friends Bestfriends Yep - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown,@ UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://open.spotify.com/artist/1rWeYVkrGXRqaD8e0kwMbc?si=xu1E7Di8T_OUpQvT46f-BA">slono</a>: Artist · 107 monthly listeners.
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1220396913665507329)** (4 messages): 

- **Navigating Privacy in Learning with IFTTT**: The LlamaIndex blog discussed the challenge of improving LLM/RAG apps via few-shot demonstrations without risking private data leaks, specifically referencing patient clinical reports. The concern was illustrated with a [tweet linking to a blog post](https://t.co/5rTwPePqV6).

- **Navarasa 2.0 Breaks Language Barriers**: An update in the LlamaIndex blog introduced Navarasa 2.0, which is a fine-tuned version of **Google Gemma 7B** by @ravithejads to support 15 Indian languages. This development emphasizes the importance of localizing general AI models to better serve regional language speakers, as highlighted in [this tweet](https://t.co/HHrfonnAr2).

- **Differential Privacy in Healthcare Data**: A new post on LlamaIndex discusses implementing differential privacy in LLMs/RAG systems to safely use sensitive data, like healthcare information, with the goal of enhancing research without compromising individual privacy. More insights into this can be found in [the associated tweet](https://t.co/2ZipmvOwXv).

- **UX in Agent-Human Interaction**: The LlamaIndex blog introduced a new template that optimizes user experience by having the agent request human input only when necessary. This approach aims to balance autonomy and intervention, and further details can be seen in [the shared tweet](https://t.co/Z16QPCWFmG).
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1220266398471684147)** (184 messages🔥🔥): 

- **Struggling with Bot Tool Integration**: Members discussed difficulties in creating a chatbot that integrates different tools like Google Search and a code interpreter. The [documentation](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/) was referenced but members encountered errors such as *"BadRequestError"*, with suggestions including combining tools into a single list and troubleshooting.

- **API and Documentation Updates**: Several users reported issues with accessing certain pages of the LlamaIndex documentation, likely due to the site being updated to MKDocs. [Links to the newly formatted documentation](https://docs.llamaindex.ai/en/stable/api_reference/indices/vector/#llama_index.core.indices.VectorStoreIndex) were provided by members as a workaround.

- **Query Pipeline Confusion**: A query pipeline DAG use case detailed [here](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/) left a user confused about the decision-making process for path traversal. Clarification was offered explaining that each chain and link in the DAG specifically defines the path and interactions for the inputs and outputs, ensuring convergence to a single output.

- **Batch Evaluation Logic Inquiry**: Members requested assistance understanding the evaluation logic applied in LlamaIndex, with specific requests for comments on the code flow for clarity. Direct answers were provided detailing the function of each code piece and the logic behind response evaluations to determine if LLM outputs matched expected results.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://api.getmerlin.in/#pricing">Merlin API Platform</a>: Integrate LLMs Into Your Production Apps In Minutes.</li><li><a href="https://colab.research.google.com/drive/13NJEyhKWT7xdJFAJ6nB8mq-fk22UVDKa?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/">Using Documents - LlamaIndex</a>: no description found</li><li><a href="https://www.llamaindex.ai/blog/running-mixtral-8x7-locally-with-llamaindex-e6cebeabe0ab">Running Mixtral 8x7 locally with LlamaIndex and Ollama — LlamaIndex, Data Framework for LLM Applications</a>: LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).</li><li><a href="https://docs.llamaindex.ai/">LlamaIndex - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/streaming/">Streaming - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/">Build your own OpenAI Agent - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/usecases/10q_sub_question/">10Q Analysis - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/localai/#llamaindex-interaction">LocalAI - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/">Migrating from ServiceContext to Settings - LlamaIndex</a>: no description found</li><li><a href="https://github.com/UKPLab/sentence-transformers/releases/tag/v2.6.0">Release v2.6.0 - Embedding Quantization, GISTEmbedLoss · UKPLab/sentence-transformers</a>: This release brings embedding quantization: a way to heavily speed up retrieval &amp; other tasks, and a new powerful loss function: GISTEmbedLoss. Install this version with pip install sentence-trans...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/">Qdrant Vector Store - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/f5263896121721de1051ce58338a1e0ea6950ca7/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py#L704">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py at f5263896121721de1051ce58338a1e0ea6950ca7 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/indices/">Index - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/4394c7f11e907c4a7c9926ae98eb53e6d60a1619/llama-index-integrations/embeddings/llama-index-embeddings-huggingface/llama_index/embeddings/huggingface/base.py#L66">llama_index/llama-index-integrations/embeddings/llama-index-embeddings-huggingface/llama_index/embeddings/huggingface/base.py at 4394c7f11e907c4a7c9926ae98eb53e6d60a1619 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/rags">GitHub - run-llama/rags: Build ChatGPT over your data, all with natural language</a>: Build ChatGPT over your data, all with natural language - run-llama/rags</li><li><a href="https://github.com/run-llama/llama_index/pull/12187">fix async streaming by logan-markewich · Pull Request #12187 · run-llama/llama_index</a>: Need to ensure lazily-declared queue/async stuff is actually instantiated before accessing Fixes #12180</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/batch_eval/">BatchEvalRunner - Running Multiple Evaluations - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/issues/12180">[Bug]: AttributeError: &#39;NoneType&#39; object has no attribute &#39;wait&#39; · Issue #12180 · run-llama/llama_index</a>: Bug Description Async Streaming Chat example: https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/#async-streaming-chat produces exception: AttributeError: &#39;NoneType&#39; object has n...</li><li><a href="https://github.com/run-llama/llama_index/issues/12143">[Question]: benchmark for the llama_index, but the latency is so weird. · Issue #12143 · run-llama/llama_index</a>: Question Validation I have searched both the documentation and discord for an answer. Question hello, i want to profile the llama index system . my code snippet is below. My gpu is one A10 with 24G...</li><li><a href="https://developer.twitter.com/en/products/twitter-api">Tweet from Twitter API | Products</a>: Use the Twitter API to analyze, learn from, and interact with Tweets, Direct Messages, and users. Scale your access to grow, experiment, and innovate.</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/indices/vector/#llama_index.core.indices.VectorStoreIndex">Vector - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/semantic_similarity_eval/#embedding-similarity-evaluator">Embedding Similarity Evaluator - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/faithfulness_eval/">Faithfulness Evaluator - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1220385053410983986)** (51 messages🔥): 

- **In Search of a Compact Code Dataset**: The user sought a small pretraining dataset and considered the [CodeSearchNet corpus](https://huggingface.co/datasets/code_search_net) which includes 2 million comment/code pairs but noted potential issues related to context length.
- **The MiniPile - A Compact Alternative for Diverse Pre-training**: [The MiniPile](https://arxiv.org/abs/2304.08442) was suggested as a suitable diverse text corpus of 1M documents for pre-training language models on smaller datasets, with minimal loss in performance.
- **APIs Holding Back on Logprobs?**: Discussion highlighted that closed-source models like Claude and Gemini do not provide logprobabilities and tokenizers, which are typically provided by platforms like OpenAI, potentially for proprietary reasons.
- **Optimizing Models for GPU Performance**: A [paper provided](https://arxiv.org/abs/2401.14489) guidelines for maximizing runtime performance of transformer models by considering impact of hyperparameters and efficient model shapes which possibly can give up to 39% higher throughput.
- **Shifting Fortunes in the Tech Scene**: Conversation touched on MS reportedly paying $600m to Inflection for poaching employees and mentioned a valuable H100 cluster, while contrasting the public speaking styles of various tech figureheads.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2401.14489">The Case for Co-Designing Model Architectures with Hardware</a>: While GPUs are responsible for training the vast majority of state-of-the-art deep learning models, the implications of their architecture are often overlooked when designing new deep learning (DL) mo...</li><li><a href="https://arxiv.org/abs/2304.08442">The MiniPile Challenge for Data-Efficient Language Models</a>: The ever-growing diversity of pre-training text corpora has equipped language models with generalization capabilities across various downstream tasks. However, such diverse datasets are often too larg...</li><li><a href="https://github.com/allenai/OLMo/issues/518">Something weird with Instruct Model · Issue #518 · allenai/OLMo</a>: 🐛 Describe the bug Here is the code I am running. The goal is to get logprob for each token generated by the chat model. olmo = AutoModelForCausalLM.from_pretrained(&quot;allenai/OLMo-7B-Instruct&quo...</li><li><a href="https://huggingface.co/datasets/code_search_net?row=42">code_search_net · Datasets at Hugging Face</a>: no description found</li><li><a href="https://arstechnica.com/information-technology/2023/09/ai-language-models-can-exceed-png">September | 2023 | Ars Technica</a>: no description found</li><li><a href="https://arstechnica.com/information-technology/2023/09/ai-language-models-can-exceed-png-and-flac-in-lossless-compression-says-study/>">September | 2023 | Ars Technica</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1220324145301033130)** (59 messages🔥🔥): 

- **AI Innovation in Antibody Design**: A member shared an [Ars Technica article](https://arstechnica.com/science/2024/03/antibodies-against-anything-ai-tool-adapted-to-make-them) discussing advancements in AI for creating therapeutic antibodies, revealing excitement about the potential of diffusion models in this field. However, another contended skepticism about actual economic use cases coming from this research area.

- **DenseFormer Sheds Light on Activation Patterns**: The [DenseFormer](https://arxiv.org/abs/2402.02622) architecture proposes a simple yet effective method of using Depth-Weighted-Average (DWA) to improve large-scale models without significant parameter increase, spurring discussion about often overlooked simple ideas in machine learning.

- **Exploring Reinforcement Learning and Transformer Sensitivity**: The [publication discussed](https://proceedings.mlr.press/v139/davis21a.html) introduces the *Catformer* architecture, aspiring to address challenges in training transformer models by reducing sensitivity through concatenated layers, a method that could improve stability in training.

- **Deep Attention Methods Discussed**: Community members engaged in a discussion about historical precedence and recent innovations in transformer architectures such as the [OmniNet](https://arxiv.org/abs/2103.01075), highlighting the potential and the challenges of implementing extensive attention mechanisms with full receptive fields.

- **Novelty and Functionality in Architectural Changes**: Within the discourse on modifying neural network architectures, such as densenet-inspired transformers, participants weighed the value of novelty against the practical benefits of getting model modifications to work effectively at scale.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://proceedings.mlr.press/v139/davis21a.html">Catformer: Designing Stable Transformers via Sensitivity Analysis</a>: Transformer architectures are widely used, but training them is non-trivial, requiring custom learning rate schedules, scaling terms, residual connections, careful placement of submodules such as n...</li><li><a href="https://arxiv.org/abs/2402.02622">DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging</a>: The transformer architecture by Vaswani et al. (2017) is now ubiquitous across application domains, from natural language processing to speech processing and image understanding. We propose DenseForme...</li><li><a href="https://arxiv.org/abs/2103.01075">OmniNet: Omnidirectional Representations from Transformers</a>: This paper proposes Omnidirectional Representations from Transformers (OmniNet). In OmniNet, instead of maintaining a strictly horizontal receptive field, each token is allowed to attend to all tokens...</li><li><a href="https://arxiv.org/abs/2312.01552">The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning</a>: The alignment tuning process of large language models (LLMs) typically involves instruction learning through supervised fine-tuning (SFT) and preference tuning via reinforcement learning from human fe...</li><li><a href="https://www.technologyreview.com/2008/03/10/221426/enzymes-built-from-scratch/">Enzymes Built from Scratch</a>: Researchers engineer never-before-seen catalysts using a new computational technique.</li><li><a href="https://arstechnica.com/science/2024/03/antibodies-against-anything-ai-tool-adapted-to-make-them">Antibodies against anything? AI tool adapted to make them</a>: Right now, making antibodies means immunizing animals. But that may change.</li><li><a href="https://github.com/marc-rigter/waker">GitHub - marc-rigter/waker: Official code for &quot;Reward-Free Curricula for Training Robust World Models&quot; accepted to ICLR 2024.</a>: Official code for &quot;Reward-Free Curricula for Training Robust World Models&quot; accepted to ICLR 2024. - marc-rigter/waker
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1220359878275764254)** (73 messages🔥🔥): 

- **Compatibility Issues between Megatron-Deepspeed and lm-eval 0.3.0**: A participant highlighted a bug with `megatron-deepspeed` evaluation compatibility. It was recommended to load from an old version of `cais/mmlu` to bypass the issue, but this still posed problems due to auxiliary train split being moved, as seen in the provided [Gist traceback](https://gist.github.com/jonabur/d99bb92be81a5af6b01f81b589b68d21).

- **Internal Usage of Modified lm-evaluation-harness**: An [arXiv paper](https://arxiv.org/abs/2403.09611) cited the use of an internal fork of EleutherAI’s lm-evaluation-harness for multimodal pre-training evaluations. Discussions ensued about the benefit of gaining access to their evaluation framework, with invitations to collaborate on extending the harness to multimodal models.

- **WandB Logging Challenges with lm-evaluation-harness**: A user reported issues where WandB logs eight times when running with eight GPUs, and GSM8K scores are printed to the terminal but not logged. It was suggested to move a block of logging code to `post_init()` as a temporary fix, with additional coordination for testing required.

- **Quantized Activations Support Inquiry**: A question was raised about whether the eval harness supports quantized activations like W8A8, leading to the clarification that quantization support is indirect through other libraries like Huggingface, which might offer some A8 methods.

- **Potential Numerical Discrepancies with Megatron-Deepspeed**: Concerns about slight numerical differences between evaluations using Huggingface transformers and Megatron-Deepspeed were discussed. It was speculated that differences in fused KQV multiplications could be due to bfloat16 usage, and that flash attention was deterministic but an analysis of forward pass outputs was necessary.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>: In this work, we discuss building performant Multimodal Large Language Models (MLLMs). In particular, we study the importance of various architecture components and data choices. Through careful and c...</li><li><a href="https://x.com/BlancheMinerva/status/1770839679580901840?s=20">Tweet from Stella Biderman (@BlancheMinerva)</a>: We&#39;re quite interested in increasing the scope of the eval harness to include things like multimodal models, RAG, and AI-graded set ups. Figuring out the best way to build this functionality inter...</li><li><a href="https://x.com/BlancheMinerva/status/1770839676435210546?s=20">Tweet from Stella Biderman (@BlancheMinerva)</a>: &#34;All of our multimodal pre-training evaluations are implemented in an internal fork of @AiEleuther&#39;s lm-evaluation-harness&#34;  Any chance of sharing the code @mckbrando? That would be a huge...</li><li><a href="https://huggingface.co/datasets/cais/mmlu/blob/main/hendrycks_test.py">hendrycks_test.py · cais/mmlu at main</a>: no description found</li><li><a href="https://gist.github.com/jonabur/d99bb92be81a5af6b01f81b589b68d21">gist:d99bb92be81a5af6b01f81b589b68d21</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://huggingface.co/docs/transformers/main_classes/quantization#transformers.QuantoConfig>">Quantization</a>: no description found</li><li><a href="https://github.com/MineDojo/Voyager/issues/149">Implement a way test local models · Issue #149 · MineDojo/Voyager</a>: Hello, Wonderful work on Voyager. Please consider added local model support (instead of openai package - using something like Python requests package to a localhost local model using openai complet...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_test.py">lm-evaluation-harness/lm_eval/tasks/hendrycks_test.py at master · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/34c9b7e40825ec998e44c5f45041953249c06a7b/lm_eval/logging_utils.py#L98-L101">lm-evaluation-harness/lm_eval/logging_utils.py at 34c9b7e40825ec998e44c5f45041953249c06a7b · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/danijar/diamond_env">GitHub - danijar/diamond_env: Standardized Minecraft Diamond Environment for Reinforcement Learning</a>: Standardized Minecraft Diamond Environment for Reinforcement Learning - danijar/diamond_env</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1220491333257396325)** (1 messages): 

- **ASCII Art Dataset Unveiled**: Discover the art of ASCII with the new dataset by a community member, containing text files like **andreas_who_is_who.txt** and **ascii_history_jgs.gmi**. Explore the dataset and various ASCII artist resources provided [here](https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM).

- **Melody Meets Model**: Integrate audio into your language models with **SMIT**, a modality integration tool available on [GitHub](https://github.com/Thytu/SMIT/tree/main). Watch a demonstration of a music generation model fine-tuning process on [YouTube](https://youtu.be/nQCibZE14Bo).

- **One Model to Rule Them All**: **Fluently-v4** gets a global release, promoting a single model solution for multiple tasks. Details about the model and its creation involving checks and Lorases are showcased on [Hugging Face](https://huggingface.co/fluently/Fluently-v4).

- **AI Aids Open Governance**: A blog post discusses the potential of AI, particularly LLMs, to improve government transparency and accessibility of public records. The use of AI technology like GPT-4 and Claude 3 in this domain is reviewed on [kyopengov.org](https://kyopengov.org/blog/exploring-open-records-law-ai).

- **Imagining with SVGDreamer**: The blog unveils SVGDreamer, a new text-guided vector graphics generation tool using a diffusion model. Published for **CVPR2024**, the tool allows for the creation of editable vector graphics from text prompts—more info provided on [Hugging Face blog](https://huggingface.co/blog/xingxm/svgdreamer).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM">Csplk/THE.ASCII.ART.EMPORIUM · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/Thytu/SMIT/tree/main">GitHub - Thytu/SMIT: SMIT: A Simple Modality Integration Tool</a>: SMIT: A Simple Modality Integration Tool. Contribute to Thytu/SMIT development by creating an account on GitHub.</li><li><a href="https://youtu.be/nQCibZE14Bo">fine-tuning musicgen + making an infinite remix - special episode - captains chair 18</a>: this week kev tries to speed run a musicgen fine-tunefor the weirdest collaboration ever using @bleepybloops artist list and colab notebook:https://github.co...</li><li><a href="https://huggingface.co/fluently/Fluently-v4">fluently/Fluently-v4 · Hugging Face</a>: no description found</li><li><a href="https://kyopengov.org/blog/exploring-open-records-law-ai">Exploring Open Records Law with AI | KOGC</a>: no description found</li><li><a href="https://huggingface.co/blog/xingxm/svgdreamer">SVGDreamer: Text Guided Vector Graphics Generation with Diffusion Model</a>: no description found</li><li><a href="https://github.com/dominiquegarmier/grok-pytorch">GitHub - dominiquegarmier/grok-pytorch: pytorch implementation of grok</a>: pytorch implementation of grok. Contribute to dominiquegarmier/grok-pytorch development by creating an account on GitHub.</li><li><a href="https://huggingface.co/blog/AviSoori1x/makemoe2">Sparse Mixture of Experts Language Model from Scratch: Extending makeMoE with Expert Capacity</a>: no description found</li><li><a href="https://github.com/andrew-m-holmes/nura">GitHub - andrew-m-holmes/nura</a>: Contribute to andrew-m-holmes/nura development by creating an account on GitHub.</li><li><a href="https://huggingface.co/blog/vladbogo/video-mamba">VideoMamba: State Space Model for Efficient Video Understanding</a>: no description found</li><li><a href="https://github.com/di37/coding-assistant-codellama-streamlit">GitHub - di37/coding-assistant-codellama-streamlit: This project demonstrates how to utilize Codellama, a local open-source Large Language Model (LLM), and customize its behavior according to your specific requirements using a Modelfile.</a>: This project demonstrates how to utilize Codellama, a local open-source Large Language Model (LLM), and customize its behavior according to your specific requirements using a Modelfile. - di37/codi...</li><li><a href="https://huggingface.co/blog/JMJM/vulnerabilities-top-10-hf-models">Giskard Bot: Identifying robustness, performance and ethical vulnerabilities in the Top 10 Most Popular Hugging Face Models</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.10853">Just Say the Name: Online Continual Learning with Category Names Only via Data Generation</a>: In real-world scenarios, extensive manual annotation for continual learning is impractical due to prohibitive costs. Although prior arts, influenced by large-scale webly supervised training, suggest l...</li><li><a href="https://huggingface.co/blog/Pclanglais/common-corpus">Releasing Common Corpus: the largest public domain dataset for training LLMs</a>: no description found</li><li><a href="https://huggingface.co/blog/andmholm/what-is-automatic-differentiation">What&#39;s Automatic Differentiation?</a>: no description found</li><li><a href="https://huggingface.co/blog/lorinma/yi-9b-divedeep">Dive Deeper into Yi-9B</a>: no description found</li><li><a href="https://huggingface.co/blog/hrishioa/retrieval-augmented-generation-1-basics">Better RAG 1: Advanced Basics</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1220285666550612028)** (76 messages🔥🔥): 

- **Curiosity around Cookbooks**: A member inquired about the term "cookbook" in the HuggingFace *learn* section, but specifics were not provided in the responses.
- **Choosing between Sdxl 1.0 and Stable Cascade**: Discussion highlighted that **Sdxl 1.0** or **Stable Cascade** could be the best models overall, with specialized finetuning possibilities for improvement in various areas.
- **Accelerate's Quantization Techniques**: Members touched upon the `load_and_quantize_model` functionality within Accelerate's quantization document as a possible alternative to `load_checkpoint_and_dispatch`, with a simple test suggesting it is a viable option.
- **Gradio API Calls and Inactivity**: Query raised about whether calling a space via API from **Gradio Client** will automatically restart an inactive space was not specifically answered.
- **Requests for Collaboration and Expertise**: Multiple calls were put out for assistance or collaboration on various topics, including pretraining data challenges, project collaborations involving expertise in PyTorch, and understanding of model quantization.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/fffiloni/coqui-bark-voice-cloning-docker">Coqui Bark Voice Cloning Docker - a Hugging Face Space by fffiloni</a>: no description found</li><li><a href="https://hf-mirror.com/">HF-Mirror - Huggingface 镜像站</a>: no description found</li><li><a href="https://github.com/suno-ai/bark?tab=readme-ov-file#-installation">GitHub - suno-ai/bark: 🔊 Text-Prompted Generative Audio Model</a>: 🔊 Text-Prompted Generative Audio Model. Contribute to suno-ai/bark development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=dHcxTmU6atk">Coffee Shop AI - Barista Tracking</a>: Coffee shop uses AI to track the Productivity of Baristas and how much Time Customers are spending in the Shop.Guys, I found the source. Here it is: https://...</li><li><a href="https://www.youtube.com/watch?v=00TSeKZyeXQ">t-SNE Simply Explained</a>: The t-SNE method in Data Science clearly and carefully explained!0:00 Concept of Neighbors6:25 Neighbor Similarity8:17 Note on Standard Deviation10:48 Moving...</li><li><a href="https://github.com/coqui-ai/TTS">GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production</a>: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production - coqui-ai/TTS
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1220368201590902835)** (1 messages): 

- **Protein Sequences Get a Vector Boost**: The **UniProt project** has released [1024-dimensional embeddings](https://www.uniprot.org/help/embeddings) for a large number of proteins in their database. A member is considering retraining these for better searchability using **Matryoshka embeddings**, as described in a recent [HuggingFace blog post](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/matryoshka/README.md).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.uniprot.org/help/embeddings">UniProt</a>: no description found</li><li><a href="https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/matryoshka/README.md">sentence-transformers/examples/training/matryoshka/README.md at master · UKPLab/sentence-transformers</a>: Multilingual Sentence &amp; Image Embeddings with BERT - UKPLab/sentence-transformers
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1220280430771703899)** (10 messages🔥): 

- **BitNet b1.58 Unveiled**: A new **1-bit LLM named BitNet b1.58**, detailed in a paper on [arXiv](https://arxiv.org/abs/2402.17764), claims to match full-precision LLMs in performance while being more cost-effective in terms of latency, memory, throughput, and energy consumption. The work could spur the development of hardware optimized for 1-bit LLMs.

- **AI-Driven Data Analysis Techniques on the Rise**: An article on Medium discusses the use of **Langchain, Instructor, and Pydantic** to redefine data analysis with AI, promising enhancements in efficiency and capability. The article is available [here](https://medium.com/technology-hits/harnessing-langchain-instructor-and-pydantic-redefining-data-analysis-with-ai-6cfe0e89b616).

- **Study on Human-Robot Team Cohesion**: The first PhD paper discussing a conceptual framework to study **team cohesion in Human-Robot Teams (HRTs)** within engineering contexts is accessible at Cambridge Core [here](https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/conceptual-framework-to-study-team-cohesion-in-humanrobot-teams/9A1BD1CB1FB23B998E57A1AB1A299FCB).

- **PatchTST Breakthrough in Time Series Forecasting**: A *Towards Data Science* article introduces **PatchTST**, a new method that promises advancements in time series forecasting. The article can be referenced [here](https://towardsdatascience.com/patchtst-a-breakthrough-in-time-series-forecasting-e02d48869ccc).

- **Measuring LLMs' ASCII Art Skills**: A study offering a measurable set of metrics for evaluating Large Language Models' capability in generating ASCII art is presented in a paper found on [arXiv](https://arxiv.org/pdf/2307.16806.pdf).

- **Tutorial on Visual Processing Mechanisms**: A YouTube video from CVPR 2022 titled "Understanding early visual processing mechanisms by the principle of efficient encoding" provides insight into how biological vision works. The lecture can be watched [here](https://www.youtube.com/watch?v=Ed9otQAmEF4).

- **Research intrigue without details**: A member shared a potentially interesting link from IEEE Xplore, but no direct information was provided about the content or relevance of the [document](https://ieeexplore.ieee.org/document/10333889). No further description was offered in the consecutive messages.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://ieeexplore.ieee.org/document/10333889">Exploring Lightweight Federated Learning for Distributed Load Forecasting</a>: Federated Learning (FL) is a distributed learning scheme that enables deep learning to be applied to sensitive data streams and applications in a privacy-preserving manner. This paper focuses on the u...</li><li><a href="https://medium.com/technology-hits/harnessing-langchain-instructor-and-pydantic-redefining-data-analysis-with-ai-6cfe0e89b616">Harnessing Langchain, Instructor, and Pydantic: Redefining Data Analysis with AI</a>: Ankush k Singal</li><li><a href="https://www.youtube.com/watch?v=Ed9otQAmEF4">Understanding early visual processing mechanisms by the principle of efficient encoding</a>: This is lecture 2 of the five lectures at CVPR 2022 tutorial &quot;A post-Marrian computational overview of how biological (human) vision works&quot;, on June 19, 2022...</li><li><a href="https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/conceptual-framework-to-study-team-cohesion-in-humanrobot-teams/9A1BD1CB1FB23B998E57A1AB1A299FCB">CONCEPTUAL FRAMEWORK TO STUDY TEAM COHESION IN HUMAN-ROBOT TEAMS | Proceedings of the Design Society | Cambridge Core</a>: CONCEPTUAL FRAMEWORK TO STUDY TEAM COHESION IN HUMAN-ROBOT TEAMS - Volume 3
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1220332454498406400)** (23 messages🔥): 

- **The Quest for ASCII Mastery**: A participant seeks collaborators to tackle the challenge of fine-tuning a language model to generate quality ASCII art, having made moderate improvements with custom GPTs. They shared the [ASCII Art dataset](https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM) and expressed a desire to develop an open-source LLM that could, for instance, create intricate ASCII art of impossible geometric illusions.

- **Telegram Bot Unleashed**: An AI bot created using the Hugging Face Mistral AI was introduced, and feedback is requested following engagement with the bot at [@mistralaichat_bot](t.me/mistralaichat_bot) on Telegram. The developer is seeking collaboration for upscaling and future projects.

- **Chaiverse's Beta Developer Platform**: An engineer from Chai Research announced their beta developer platform, Chaiverse, which ranks community-produced LLMs and allows developers to submit their models for real-user feedback. Interested individuals are encouraged to read more about their mission in the [Chaiverse white paper](https://www.chaiverse.com/white-paper).

- **Promoting Federated Learning**: A link was shared to a [GitHub repository](https://github.com/ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting) focusing on federated learning for load forecasting using clustering and sequential DNN methods.

- **Chat Experiments with ASCII Art**: Participants discussed methods and challenges of generating ASCII art with LLMs, including the use of HTML and CSS for formatting and the mixed results when requesting complex ASCII art from the models. The consensus seems to be that models can sometimes produce simple representations like cats but struggle with more intricate designs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM">Csplk/THE.ASCII.ART.EMPORIUM · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting">GitHub - ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting: Federated Learning on Energy Dataset for load forecasting using clustering and sequential DNN methods</a>: Federated Learning on Energy Dataset for load forecasting using clustering and sequential DNN methods - ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting</li><li><a href="https://console.chaiverse.com">Leaderboard</a>: no description found</li><li><a href="https://www.chaiverse.com/white-paper">White Paper | Chaiverse | The Chai AI Developer Platform</a>: Discover Chai AI's vision for crowdsourcing the leap to trillion-parameter AGI.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1220295378755391539)** (2 messages): 

- **Hurry, Mark Your Calendars!**: Event details have been added with the [event link](https://discord.com/events/879548962464493619/1219690164339736770) provided; an announcement is also expected today.
- **Decoding Obesity with Data**: Check out an [in-depth EDA notebook](https://www.kaggle.com/code/muhammadibrahimqasmi/deciphering-obesity-trends-an-in-depth-eda) on obesity trends, where statistical analysis and visualizations reveal the interplay of age, gender, and lifestyle choices on this critical health issue.

**Link mentioned**: <a href="https://www.kaggle.com/code/muhammadibrahimqasmi/deciphering-obesity-trends-an-in-depth-eda">Deciphering Obesity Trends &#x1F4C9;: An In-depth EDA &#x1F4CA;</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from multiple data sources

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1220301340278919179)** (4 messages): 

- **Beware of Unwanted DM Offers**: A member warned that an individual might reach out in direct messages to ask for paid work, noting that the person has been previously kicked out of other Discord servers for similar behavior.
- **SegGPT Model Unveiled**: A new **SegGPT** model has been added, capable of various image-to-image tasks with impressive one-shot segmentation results. The SegGPT model and its paper are accessible via the [Hugging Face documentation](https://huggingface.co/docs/transformers/main/en/model_doc/seggpt).
- **Gratitude for SegGPT**: A member expressed thanks and showed interest in trying out the newly introduced **SegGPT** model.

**Link mentioned**: <a href="https://huggingface.co/docs/transformers/main/en/model_doc/seggpt">SegGPT</a>: no description found

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1220284164666818590)** (33 messages🔥): 

- **Quest for Personality Prediction Data**: There's an exploration for datasets suited for text-based **personality prediction** research, with **myPersonality** dataset being unavailable. The scarcity of public datasets for this application presents challenges for student-level research due to limited access to large-scale data.

- **A Journey to Master ASCII Art with LLMs**: An exciting endeavor is being pursued to fine-tune large language models (LLMs) to excel in generating **ASCII art**, with a mention of a specific dataset, [THE.ASCII.ART.EMPORIUM](https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM), and a call for guidance on how to effectively embed ASCII art for LLM training.

- **Sharing Deep Code Generation Dataset - "The Stack"**: There's a sharing of **"The Stack" dataset**, a 6TB trove of source code spanning over 300 programming languages, potentially useful for code generation projects. Users must agree to terms, including original code licenses and data removal updates, [here](https://huggingface.co/datasets/bigcode/the-stack).

- **Modernizing Topic Modeling with BERT-based Algorithms**: A recommendation was made to check out **BERTopic**, a technique for topic modeling using 🤗 transformers and contextually informed embeddings that offer various topic modeling methods, detailed [here](https://maartengr.github.io/BERTopic/index.html).

- **Solving Quantization Challenges for Fine-Tuned Models**: A discussion on best practices for quantizing LoRA-adapted models highlighted the utility of merging and minimizing quantization loss, with related examples found in the [PEFT documentation](https://huggingface.co/docs/peft/developer_guides/lora).

- **Troubleshooting Trainer Class Issues in Huggingface**: Issues were reported with using the **Trainer class** in Huggingface, particularly around dependencies requiring updates and acceleration. Suggestions involved upgrading libraries, clearing cache, manipulating import orders, and considering a restart or reconfiguration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/peft/developer_guides/lora">LoRA</a>: no description found</li><li><a href="https://huggingface.co/datasets/bigcode/the-stack">bigcode/the-stack · Datasets at Hugging Face</a>: no description found</li><li><a href="https://maartengr.github.io/BERTopic/index.html">Home</a>: Leveraging BERT and a class-based TF-IDF to create easily interpretable topics.</li><li><a href="https://huggingface.co/spaces/sentence-transformers/quantized-retrieval">Quantized Retrieval - a Hugging Face Space by sentence-transformers</a>: no description found</li><li><a href="https://huggingface.co/blog/embedding-quantization">Binary and Scalar Embedding Quantization for Significantly Faster &amp; Cheaper Retrieval</a>: no description found</li><li><a href="https://sbert.net/examples/applications/embedding-quantization/README.html">Embedding Quantization &mdash; Sentence-Transformers  documentation</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1220302753138610216)** (28 messages🔥): 

- **Corrupted State Dictionary Woes**: A member encountered a **ValueError** indicating a corrupted state dictionary when trying to load a fine-tuned model using `model.eval()`. It is unclear if there was a solution proposed or found for this issue.

- **Decoding the Diffusion Checkpoint Codes**: A brief explanation was given that a checkpoint stores the learned information of a model, and the conversation transitioned towards searching on HuggingFace for checkpoints like **sdxl 1.0 or stable diffusion 2.1**.

- **ASCII Art with Diffusion Models**: A discussion emerged around creating a diffusion-like model for an ASCII art dataset. The conversation explored converting ASCII to images, but the question of **making a diffusion model that operates natively on ASCII** remained open.

- **Financial AI Chatbot Construction**: A user inquired about building an AI chatbot for financial data with multiple access levels and classifications. No specific model was proposed in the given messages, but another user mentioned the need to review the data first.

- **Inquiring Minds Want to Know**: Users posed questions about joining a group named **Zero-GPU-Explorers** and assistance with using and training the **all-MiniLM-L6-v2 model** with their dataset, indicating a desire for community support.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/CiroN2022/ascii-art">CiroN2022/ascii-art · Hugging Face</a>: no description found</li><li><a href="https://ivbhatt.medium.com/asciify-b3a0c70433fa">ASCIIfy</a>: Using OpenCV, Pillow and Python 3 to convert images to ASCII
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1220293288939225228)** (40 messages🔥): 

- **Adding Chat GPT Bots to Discord & API Cost**: To add Chat GPT bots to a Discord channel, one must obtain the API, which is not free. It is a paid service.
- **Troubles Receiving Responses in Postman**: A community member struggled with not receiving responses on Postman despite setting up an assistant, thread, and message and was advised to review the [documentation](https://platform.openai.com/docs/api-reference/chat) and check the "content" parameter for responses.
- **Perplexity Allegedly a Search Wrapper**: A member shared an article claiming that Perplexity likely condenses content from Google Search top results, summarizing the content from the top 5-10 entries. *Perplexity is Most Likely a Google Search Wrapper* was published on March 18, 2024, by Mohit Pandey [here](https://analyticsindiamag.com/perplexity-is-most-likely-a-google-search-wrapper/).
- **Pondering AI's Role in Video Compression**: Community discussion explored the idea of using AI in video compression, comparing potential uses similar to deep learning super sampling (DLSS) and Whisper for audio compression. An existing blog post discussed this in terms of audio compression [here](https://www.dbreunig.com/2023/11/07/extreme-compression-with-ai.html).
- **Conversion to Int8 Embeddings for Storage Efficiency**: A member reported saving about 80% on storage costs when preconverting Float32 embeddings to Int8 before sending to their vector database. They expressed a wish for native Int8 support in embedding-v3 models to streamline the process and debated the potential use of pickle, sqlite, and another database for various tasks in multimodal prototypes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.dbreunig.com/2023/11/07/extreme-compression-with-ai.html">Extreme Compression with AI: Fitting a 45 Minute Podcast into 40kbs</a>: Writing about technology, culture, media, data, and all the ways they interact.</li><li><a href="https://clickup.com/ai">ClickUp Brain | One AI to Replace them All</a>: no description found</li><li><a href="https://analyticsindiamag.com/perplexity-is-most-likely-a-google-search-wrapper/">Perplexity is Most Likely a Google Search Wrapper</a>: A user posted on the LocalLLaMA thread of Reddit that Perplexity summarises the content from the top 5-10 results from Google Search.
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1220272435434229830)** (11 messages🔥): 

- **Custom API Connection Clarification**: A member asked how to connect to a custom GPT-3 model through the API. Solbus directed to use the Assistants API, providing a link for further assistance: [Assistants API Guide](https://help.openai.com/en/articles/8673914-gpts-vs-assistants).

- **Seeking Feedback on Animal Alter-Ego GPT**: A user named boouyaah shared a GPT creation that turns individuals into animal versions of themselves and sought feedback on the prompts: [You, but as an animal](https://chat.openai.com/g/g-SGpDLmwE9-you-but-as-an-animal).

- **Sudden Reduction in Pinned Custom GPTs**: Jaredquek reported an issue with the number of Custom GPTs that can be pinned to the sidebar, stating that previously pinned GPTs vanished and there's now a limit to pinning only 4, seeking an explanation or workaround.

- **Optimizing Knowledge File Distribution Across Multiple GPTs**: Mikejeason posed a question about whether it's more productive to distribute knowledge files across multiple GPTs tailored for different parts of a prompt, rather than combining everything into a single GPT.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1220355607471718411)** (41 messages🔥): 

- **Rule Reminder Tightens Up**: After a query about **prompt engineering jobs**, users were reminded of **Rule 7** prohibiting self-promotion, soliciting, or advertising. The rules were further clarified with a direct [link to the rule](https://discord.com/channels/974519864045756446/1107255707314704505/1213395523973808148) provided by a user.
- **Disabilities Get a Cold Shoulder in GPT-4 Vision**: A user expressed frustration when **GPT-4 Vision** failed to provide assistance regarding disabled individuals, repeatedly responding with "Sorry, I can't help with that."
- **Toolkit Teaser Ignites Curiosity**: Despite the rule against self-promotion, a user mentioned developing a **prompt chaining/prompt engineering toolkit** and looked for people to test a prototype.
- **Challenging the ChatGPT Product Description Generator**: A detailed discussion took place regarding the feasibility of using ChatGPT to generate product descriptions for a catalog, focusing on natural and organic products. There was skepticism about the AI's ability to accurately handle the task without manual intervention.
- **Seeking Clarification on Benefits and Applications**: The discussion evolved towards simplifying the task for ChatGPT, with a user suggesting to focus on generating **benefits and uses** sections based on the product descriptions provided, which might be a more manageable approach for the AI.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1220355607471718411)** (41 messages🔥): 

- **Rule 7 Reminder**: A new user inadvertently violated [Rule 7](https://discord.com/channels/974519864045756446/1107255707314704505/1213395523973808148) by asking about prompt engineering jobs, was reminded to review the server rules, particularly against self-promotion, soliciting, or advertising.
- **Apology for Misstep**: Following the call to attention on server rules, the user **apologized** and promised to review the rules to ensure it does not happen again.
- **GPT-4 Vision Limitations Discussed**: A member discussed difficulties in getting **GPT-4 Vision** to acknowledge disabled people, receiving standard unhelpful responses from the system.
- **Prompt Toolkit Promotion Violation**: User *quixoticquiche* violated Rule 7 by advertising their prompt chaining toolkit and seeking feedback, leading to another reminder about the rules against soliciting in the server.
- **Challenges of Automating Product Descriptions**: Members discussed the feasibility of using ChatGPT to automatically generate *detailed product descriptions*; concerns were raised about the preciseness and reliability of such generated content without human oversight.
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1220302688995119174)** (96 messages🔥🔥): 

- **Understanding LangChain Tool Ingestion**: A member inquired whether an array can be passed as input to a tool in LangChain, leading to an explanation that while general examples were provided, specific cases for array inputs were not available in the knowledge sources.
- **GraphCypherQAChain Use Case**: A member sought advice on how to perform string comparisons in lower case within the GraphCypherQAChain, but no specific information was provided from the knowledge sources.
- **Learning Retrieval-Augmented Generation**: A free resource, [Intro to AI for Developers](https://takehomes.com/library/developers/intro-to-ai), was recommended for those looking to learn AI with a focus on Large Language Models in a project-based approach.
- **Humorous Take on AI Challenges**: In a lighthearted discussion, members joked about the complexity of integrating various frameworks and technologies with LangChain, implying the difficulty can be as hard as solving the space-time continuum.
- **Dynamic Decision-Making for Database Queries**: Discussion touched upon creating an agent capable of determining whether to query an SQL database or a vector database based on user questions, emphasizing the need for automatic decision-making in LangChain use cases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://takehomes.com/library/developers/intro-to-ai">A Practical Introduction to AI for Developers – TakeHomes Library</a>: no description found</li><li><a href="https://js.langchain.com/docs/use_cases/graph/quickstart#chain>).">Quickstart | 🦜️🔗 Langchain</a>: In this guide we’ll go over the basic ways to create a Q&amp;A chain over a</li><li><a href="https://python.langchain.com/docs/use_cases/web_scraping#question-answering-over-a-website>)">Web scraping | 🦜️🔗 Langchain</a>: Open In Colab</li><li><a href="https://tenor.com/bgtK0.gif">Ideas Genius GIF - Ideas Genius George - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/langchain-ai/langchain/issues/6138">ConversationChain default prompt leads the model to converse with itself · Issue #6138 · langchain-ai/langchain</a>: System Info langchain==0.0.195 python==3.9.6 Who can help? @hwchase17 Information The official example notebooks/scripts My own modified scripts Related Components LLMs/Chat Models Embedding Models...</li><li><a href="https://python.langchain.com/docs/use_cases/web_scraping#asynchtmlloader>).">Web scraping | 🦜️🔗 Langchain</a>: Open In Colab</li><li><a href="https://github.com/langchain-ai/langchain/issues/7876>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/11590>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/1438>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/4561>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/9389>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/4197>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/12410>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13602>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1220406790341001336)** (7 messages): 

- **Python Version Hell Strikes Again!**: A member is attempting to update **[langchain-ai/weblangchain](https://github.com/langchain-ai/weblangchain)** and encounters issues with dependencies and Python versions. The error `TypeError: Type is not JSON serializable: numpy.float64` is causing the application to crash, hinting at a serialization problem with `numpy` data types.
  
- **Potential Link to Existing Issue**: The serialization problem may be related to a known issue discussed in **[TypeError: Type is not JSON serializable: numpy.float64](https://github.com/langchain-ai/langchain/discussions/17876)** on the langchain-ai's GitHub discussions.

- **Troubleshooting Other Components**: Testing with LangSmith shows no issues, hence the problem could be tied to something on the TypeScript client side, as pinning Starlette to older versions did not resolve the issue.

- **Poetry Doesn’t Solve All Problems**: A member suggested using Poetry to escape the Python version issues, but it’s revealed that Poetry is already in use and the problem persists with the latest versions of Langchain/Langserve.

- **Issue Raised on GitHub**: The serialization issue led to the creation of a GitHub issue titled **[TypeError: Type is not JSON serializable: numpy.float64](https://github.com/langchain-ai/langserve/issues/551)** to address the incompatibilities with the latest versions of Langchain/Langserve.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://smith.langchain.com/public/272f4463-4bb7-4fa3-ad5d-aea31dab5c8d/r">LangSmith</a>: no description found</li><li><a href="https://github.com/langchain-ai/weblangchain/">GitHub - langchain-ai/weblangchain: LangChain-powered web researcher chatbot. Searches for sources on the web and cites them in generated answers.</a>: LangChain-powered web researcher chatbot. Searches for sources on the web and cites them in generated answers. - langchain-ai/weblangchain</li><li><a href="https://github.com/mieslep/weblangchain/tree/compoent_and_update">GitHub - mieslep/weblangchain at compoent_and_update</a>: LangChain-powered web researcher chatbot. Searches for sources on the web and cites them in generated answers. - GitHub - mieslep/weblangchain at compoent_and_update</li><li><a href="https://github.com/langchain-ai/langserve/issues/551">TypeError: Type is not JSON serializable: numpy.float64 · Issue #551 · langchain-ai/langserve</a>: I&#39;ve narrowed down the problem to one that reproduces on the weblangchain repo https://github.com/langchain-ai/weblangchain I&#39;m trying to update to the latest versions of Langchain/Langsmith/L...</li><li><a href="https://github.com/langchain-ai/langchain/discussions/17876">TypeError: Type is not JSON serializable: numpy.float64 · langchain-ai/langchain · Discussion #17876</a>: Checked other resources I added a very descriptive title to this issue. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and di...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1220289106161041459)** (5 messages): 

- **AI-Powered Data Analysis Enhanced**: An article titled "Harnessing Langchain, Instructor, and Pydantic: Redefining Data Analysis with AI" details how integrating several tools can transform data analysis. Read the full story on [Medium](https://medium.com/technology-hits/harnessing-langchain-instructor-and-pydantic-redefining-data-analysis-with-ai-6cfe0e89b616).

- **Introducing Promptsage for Simplified Prompt Engineering**: A new weekend project, Promptsage, aims to simplify prompt building and sanitization for LLMs, featuring security and privacy guardrails, and it's compatible with langchain. Explore the tool on [GitHub](https://github.com/alexmavr/promptsage).

- **Exploring Chain Extensions for Large Outputs**: A member inquires about a **Langchain** feature allowing chains to continue generating output beyond a model's token limit by sending additional requests based on "Stop Reason" determinations. The question highlights a desire for effective handling of large outputs that exceed token restrictions like OpenAI's GPT-4-Turbo's 4k output tokens.

- **Python Meets Bedrock Anthropic Haiku**: Due to a lack of support for functions in Bedrock, a comprehensive guide has been created, demonstrating how to leverage Bedrock Anthropic Haiku using Python. Interested readers can find the guide on [Medium](https://medium.com/@leonardo.bolanos/leveraging-bedrock-anthropic-haiku-with-python-a-comprehensive-guide-9f5e912982be).

**Link mentioned**: <a href="https://github.com/alexmavr/promptsage">GitHub - alexmavr/promptsage: Promptsage is an LLM prompt builder, linter and sanitizer with built-in guardrails</a>: Promptsage is an LLM prompt builder, linter and sanitizer with built-in guardrails - alexmavr/promptsage

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1220555842441580606)** (1 messages): 

- **Sluggish Requests on the West Coast**: Users on the **West Coast (US)** are experiencing unusually slow requests. The cause is suspected to be a cloud issue and is currently under investigation.
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1220385619822379088)** (53 messages🔥): 

- **Curiosity About Google's Gemini 1.5 Pro**: Members discussed the release of **Google's Gemini 1.5 Pro** with a 1 million word context window. Despite [public documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning) only mentioning version 1.0, a member mentioned being in contact with Google to gain access.

- **C3 Model's Inconsistency Woes**: Users expressed frustration with C3's performance, with one recommending the self-moderated version of **Claude 3**, claiming it's less likely to reject content incorrectly, and even surpasses GPT-4 in this regard.

- **Debating Grok AI's Capabilities**: The conversation pivoted to Grok, an open-source model, where opinions diverged on its quality compared to Mixtral, with some labeling it **"shitty"** due to its high cost and potential undertraining. However, others defended Grok's capabilities as a pure base model, emphasizing that it's unfair to compare it with chat-tuned models.

- **Grok Benchmarks Questioned**: A debate occurred over the usefulness of benchmarks for Grok, with some challenging the comparison to models like Mixtral, while others pointed to Grok's official benchmarks showing it as a knowledgeable conversational model with wit.

- **Exploring Grok's Testing and Access**: Members discussed how to test Grok, with one member providing a link to [try it out](https://grok.x.ai/), and it was clarified that Grok can be accessed through the xAI platform, potentially without needing Twitter Premium+. They also discussed the content that could be used for testing, such as asking about political opinions or IT-related questions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://grok.x.ai/">xAI Grok</a>: no description found</li><li><a href="https://x.ai/blog/grok">Announcing Grok</a>: no description found</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning?authuser=1#gemini-model-versions">no title found</a>: no description found</li><li><a href="https://x.com/deliprao/status/1770128250003460396?s=46">Tweet from Delip Rao e/σ (@deliprao)</a>: I look at this and don&#39;t walk way thinking Grok is better. As a pragmatic person, I look at it and wonder why bother with Grok (314B) when you have Mixtral with almost similar performance and is a...
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1220361174344798319)** (4 messages): 

- **Nanobind as a Solution**: A member suggested looking into **nanobind** to increase efficiency for MLX, with a mention of it being potentially helpful based on their experience.
- **Acknowledging the Helpful Tip**: A follow-up message from the same member expressed gratitude for the recommendation to check out **nanobind**.
- **GTC Event Discord Hiccups**: During the GTC event, members faced issues with Discord's stage channel regarding screen sharing not functioning correctly. It was resolved by moving to a voice channel, prompting the suggestion to use voice channels by default for future lectures.
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1220583810396328016)** (1 messages): 

- **Fusing GaLore's Adam with Triton**: A member conducted a study and opened a [pull request](https://github.com/jiaweizzhao/GaLore/pull/29) on GitHub detailing the process of fusing **GaLore's Adam optimizer** with Triton. Best results were achieved with a hybrid kernel leveraging `torch.matmul` for the projection of gradients to low-rank, enhancing the memory efficiency during pre-training and fine-tuning of models.

**Link mentioned**: <a href="https://github.com/jiaweizzhao/GaLore/pull/29">[WIP] Fused Adam Triton Kernels by jeromeku · Pull Request #29 · jiaweizzhao/GaLore</a>: Fused GaLore Adam (WIP) Various fused implementations of Adam update step per Gradient Low-Rank Projection This is an initial attempt at optimizing the update step of the GaLore Adam optimizer. Ove...

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1220298969800900639)** (1 messages): 

- **micrograd gets a CUDA boost**: A member shared a link to a library, [micrograd-cuda](https://github.com/mlecauchois/micrograd-cuda), that extends Karpathy's micrograd library with CUDA kernels and adds 2D tensor logic. The GitHub repository offers contributions to further develop this CUDA-accelerated version of micrograd.

**Link mentioned**: <a href="https://github.com/mlecauchois/micrograd-cuda">GitHub - mlecauchois/micrograd-cuda</a>: Contribute to mlecauchois/micrograd-cuda development by creating an account on GitHub.

  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1220510281206202439)** (3 messages): 

- **Lightning Strikes PyTorch**: [Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder), a source-to-source compiler for PyTorch, was highlighted, aiming to speed up PyTorch programs on single accelerators and distributed systems.

- **GTC Session Announcement**: Members were informed about an upcoming GTC talk and prompted to ask questions to specific individuals before the session is up in ~24 hours.

- **Link to NVIDIA GTC Session**: An NVIDIA GTC talk related to Thunder was mentioned with a direct [link to the session catalog](https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=Thunder%20#/session/1696294424486001JD3i), detailing the dates for workshops, AI conference and expo, and the keynote running from March 17-21, in San Jose, CA, and virtually.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=Thunder%20#/session/1696294424486001JD3i">NVIDIA #GTC2024 Conference Session Catalog</a>: Register now. Streamed online. March 18-21, 2024.</li><li><a href="https://github.com/Lightning-AI/lightning-thunder">GitHub - Lightning-AI/lightning-thunder: Source to source compiler for PyTorch. It makes PyTorch programs faster on single accelerators and distributed.</a>: Source to source compiler for PyTorch. It makes PyTorch programs faster on single accelerators and distributed. - Lightning-AI/lightning-thunder
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1220499189172273232)** (9 messages🔥): 

- **Ozaki Scheme Enhances Matrix Multiplication**: The [Ozaki scheme](https://arxiv.org/abs/2301.09960), as explained in an arXiv paper, optimizes multiple precision basic linear computation and can perform faster than existing methods for fixed and arbitrary precision matrix multiplication. It benefits from optimized low precision operation and outperforms Strassen matrix multiplication up to a certain precision.
- **Exploring the Kahan Summation Algorithm**: A link was shared to the Wikipedia article about the [Kahan summation algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm), an approach in numerical analysis that significantly reduces numerical error during summation by maintaining a running compensation.
- **IEEE 754 Standards**: Reference was made to an [ITU paper](https://www.itu.dk/~sestoft/bachelor/IEEE754_article.pdf) discussing the IEEE 754 standards, which are crucial for floating-point computation.
- **Jeremy's Team Acknowledges Mention**: Acknowledgement came from Jeremy Howard regarding the mention of their work related to Ozaki scheme implementation, with an expression of future improvements in the area.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Kahan_summation_algorithm">Kahan summation algorithm - Wikipedia</a>: no description found</li><li><a href="https://arxiv.org/abs/2301.09960">Acceleration of Multiple Precision Matrix Multiplication using Ozaki scheme</a>: Optimized multiple precision basic linear computation, especially matrix multiplication, is crucial for solving ill-conditioned problems. The recently proposed Ozaki scheme, which implements accurate ...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1220448064850886677)** (5 messages): 

- **Recommendations for Organized Conversation**: A member suggested using **standard messages** for new topics, **replies** for branching into conversations, and **thread creation** for focused subjects, underlining the benefits for message visibility and channel readability. They complimented a user named <@272654283919458306> for excellent thread management on another server called Latent Space.
- **Acknowledgement of Server Tips**: A member expressed appreciation for the etiquette suggestions made regarding conversation organization on Discord, finding them especially helpful as a new user.
- **Presentation of a Resourceful Link**: A member shared a [Springer book link](https://link.springer.com/book/10.1007/978-3-031-30442-2#other-volumes), highlighting details about conference proceedings for PPAM 2022, including contributors and a table of contents with access to papers.

**Link mentioned**: <a href="https://link.springer.com/book/10.1007/978-3-031-30442-2#other-volumes">Parallel Processing and Applied Mathematics</a>: no description found

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1220761584335327242)** (2 messages): 

- **Seeking Solution Verification**: A member has completed **Chapter 2 exercises** from the 'pmpp-book' and is looking for ways to verify their answers. They expressed interest in **DM exchanges** for cross-checking solutions with others.
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1220466248907751445)** (2 messages): 

- **Lightning Strikes for CUDA Lovers**: A member highlighted the launch of a new [Zero to Thunder tutorial](https://lightning.ai/lightning-ai/studios/zero-to-thunder-tutorial) at GTC, targeting Python and PyTorch enthusiasts who want to deliver custom CUDA kernels for not-so-standard models. Although it's still in its experimental stage, and some functionalities may be lacking, it's an enticing venture for the adventurous.

- **Smiling GPUs Cause a Stir**: An observation was shared via a [Twitter link](https://fxtwitter.com/iScienceLuvr/status/1770931936657358908) pointing out the amusing fact that the new Blackwell GPUs appear to have smiley faces. This quirky design feature caught the attention of tech enthusiasts, sparking humorous comments online.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lightning.ai/lightning-ai/studios/zero-to-thunder-tutorial">zero-to-thunder-tutorial - a Lightning Studio by t-vi</a>: Get started with ⚡ Lightning Thunder.</li><li><a href="https://fxtwitter.com/iScienceLuvr/status/1770931936657358908">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Why isn&#39;t anybody talking about the fact that the new Blackwell GPUs are literally smiling at us lol
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1220358094777876580)** (12 messages🔥): 

- **Tensor Color Code Deciphered**: In triton-puzzles, color coding of tensors was clarified; color signifies the source tensor, with out-of-bounds access showing red. There was a discussion about a potential bug, suggesting that **out-of-bounds loads might be incorrectly indicated**, even if masking is correct.
  
- **The Draw Order of Tensors Explained**: The draw order for tensors in Triton was specified to be depth, row, column, and 1D tensors are drawn as 1,1,col.

- **Triton `tl.exp` Operator Issue Reported**: A new member encountered a `NotImplementedError` when using `tl.exp(x)` or `x.exp()` in Triton, stating this operation is unsupported in interpreter mode with no numpy implementation.

- **Member Submits PR for Exp2**: The exp2 function was mentioned in a probable context of a fix or implementation in flash attention with a Pull Request submitted to Triton.

- **Puzzle 3 Completed, Debugging Puzzle 4**: A member completed Puzzle 3 and shared their debugging process using print statements for Puzzle 4, where they compare their answer with the expected one by performing an outer sum operation with torch.
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1220818264251174993)** (21 messages🔥): 

- **Struggling with Space Matching**: Uniti AI is facing issues with their AI leasing agents where **GPT4.Turbo** is not accurately matching property inventory to user requirements—for example, suggesting properties with 17,000 sq. ft when asked for spaces between 2,000 - 4,000 sq. ft.
- **Complex Matching Logic Challenges**: A nuanced challenge is to offer inventory within a 33% range for properties below 5,000 sq. ft, and a 20% range for those above 5,000 sq. ft, making the matching process increasingly complex.
- **Suggestions for a Simplified Approach**: The current approach involves a detailed prompt to match broker inquiries, but the suggestion was made to use **regular filters or have the LLM generate a SQL query** to pull the right units instead.
- **Recognizing Over-Reliance on LLMs**: The conversation highlights a "common llm trap", suggesting that not all tasks require an LLM, and simpler database queries could be the solution. The idea is to use the LLM for generating the query rather than the filtering itself.
- **Reference to RAG for Efficiency**: A [blog post on Retrieval Augmented Generation (RAG)](https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/) by Jason Liu was mentioned, illustrating the effectiveness of pairing LLMs with regular database queries for tasks such as date range extraction.

**Link mentioned**: <a href="https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/">RAG is more than just embedding search - Instructor</a>: no description found

  

---


**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1220596048871821382)** (5 messages): 

- **Bedrock vs Direct Approach**: A user mentioned that opting for a **direct integration** with Claude might be more efficient as **Bedrock** has shown to be somewhat cumbersome and less reliable in terms of uptime.
- **Frontline Access to Claude**: One member revealed they have **priority rate limits** with Claude owing to a year-long development partnership, putting them ahead of a substantial **200k+ waitlist**.
- **Choosing Direct Connection Over Bedrock**: Despite having priority access, the same user confirmed they are employing a **direct connection** to interact with Claude, bypassing the Bedrock framework.
  

---


**LLM Perf Enthusiasts AI ▷ #[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/)** (1 messages): 

ibash: > write high quality code
Damn.
  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/)** (1 messages): 

jeffreyw128: lol wut
  

---


**LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/)** (1 messages): 

emrgnt_cmplxty: Basic prompting isn't getting it done for you?
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1220363095277309952)** (15 messages🔥): 

- **Seeking Synthetic Benchmarks for LLM Study**: A member inquired about the existence of fully synthetic benchmarks with controllable properties to study *foundation model* capabilities, specifically LLMs.
- **Startups Generate Data Through LLMs**: It was mentioned that startups are creating synthetic benchmarks using an LLM to generate a lot of data based on the model they are studying.
- **Disentangling LLM Capabilities from Data Quality**: A discussion arose about studying the origins of LLM capabilities by altering the diversity and reasoning presence in the training data to move beyond the general belief that "the capabilities are in the data."
- **Synthetic Data and Worlds Garner Interest**: One member expressed enthusiasm for synthetic data and worlds, contemplating writing a paper on the subject.
- **Organizing Open Source Data Curation**: A member suggested that a public, systematic approach to constructing pretraining data could be beneficial for organizing open-source data curation efforts.
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1220546492830585003)** (6 messages): 

- **In Search of SOTA**: ChatGPT is mentioned as a tool for rewriting content, hinting that the practice of using language models for rewriting is commonplace in achieving **state-of-the-art (SOTA)** results.
- **Minor Tweaks in the Workflow**: A member discusses using ChatGPT for rewriting content, making **minor changes** to tailor the output for their needs.
- **Project Work in Progress**: A side project related to ChatGPT and rewriting was mentioned, with the member noting a lack of substantial insights due to limited involvement.
- **Academic Rush**: ChatGPT's rewriting capabilities are being used to expedite the completion of a **class project**.
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1220363410491969576)** (5 messages): 

- **Bot Beatdown Effect on Human Psyche**: A member speculated about the impact of losing to AI on human players, indicating games like chess are unscathed despite superhuman AIs.
- **Historical AI Victories Resonate**: One participant pointed out that *Garry Kasparov's loss to Deep Blue* had significant impact on chess, similar to AI's later triumph in Go.
- **The Individual Player's Take**: A user weighed in, suggesting that how affected a player is by AI might vary greatly depending on their personal demeanor.
- **Philosophical AI Discussion**: Someone shared a link to a discussion with *Minqi Jiang* and *Marc Rigter* about the possibility of creating a generalist agent in reinforcement learning ([MLStreetTalk Tweet](https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw)).

**Link mentioned**: <a href="https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw">Tweet from Machine Learning Street Talk (@MLStreetTalk)</a>: We just dropped the show with @MinqiJiang and @MarcRigter and discuss the philosophy of whether it is possible, in principle and in practice to build a &#34;generalist agent&#34; in RL.

  

---



**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1220406585336004651)** (1 messages): 

- **Help Wanted for the 01 Open Source Hardware**: A member introduced the new open source hardware device named **the 01**, asking for contributions from the community. The project's hardware and software are fully open source, with details available in [this tweet](https://twitter.com/OpenInterpreter/status/1770821439458840846).
  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

venadore: life lesson
  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=21Tc92g15pM
  

---




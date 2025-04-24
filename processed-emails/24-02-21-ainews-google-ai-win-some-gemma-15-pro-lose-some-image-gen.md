---
id: 314b8202-bbf9-47a8-876b-f5e6cecf9c7b
title: 'Google AI: Win some (Gemma, 1.5 Pro), Lose some (Image gen)'
date: '2024-02-22T02:21:19.724902Z'
original_slug: ainews-google-ai-win-some-gemma-15-pro-lose-some
description: >-
  Google's Gemma open models outperform Llama2 and Mistral in benchmarks but
  face licensing and human vibe check issues. Gemini Pro 1.5 shows promise in
  video understanding and long context handling. Discussions include AI model
  releases, dataset editing, roleplay, hardware setups, and integration
  challenges.
companies:
  - google
models:
  - gemma-2b
  - gemma-7b
  - gemma-pro-1.5
  - llama-2
  - mistral
topics:
  - open-models
  - benchmarking
  - license-issues
  - video-understanding
  - long-context
  - dataset-editing
  - roleplay-ai
  - hardware-setup
  - multi-gpu
  - model-integration
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/20/2024. We checked **20** guilds, **313** channels, and **8555** messages for you. Estimated reading time saved (at 200wpm): **836 minutes**.

Google is at the top of conversations for a lot of good and bad reasons today. The new [Gemma open models](https://news.ycombinator.com/item?id=39453271) (2-7B in size, presumably the smaller version of Gemini models) showed better benchmarks than Llama2 and Mistral:

 ![image.png](https://assets.buttondown.email/images/47efce84-ca04-4e81-b7f4-89fe6af27721.png?w=960&fit=max) 

but comes with an [unusual license](https://ai.google.dev/gemma/prohibited_use_policy) and [doesnt pass the human vibe check](https://x.com/abacaj/status/1760393505153679369?s=20).

Meanwhile, [literally everybody](https://twitter.com/growing_daniel/status/1760459653887168984) is dogpiling on Gemini's clumsily diverse image generation, a problem [partially acknowledged by Google](https://twitter.com/benthompson/status/1760452419627233610?t=90xQ8sGy63D2OtiaoGJuww).

![image.png](https://assets.buttondown.email/images/b3649e13-884c-4941-94b0-b71a074f7ba2.png?w=960&fit=max) 

But in what seems like a pure win, the long context of the still-waitlisted Gemini Pro 1.5 (with 1m token context) is [video understanding](https://news.ycombinator.com/item?id=39458264) and [needle in haystack tests](https://twitter.com/mattshumer_/status/1759749194108043597).




---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Mixed Reception for Gemma Models**: Some users, such as [@itsme9316](https://www.assistant.ai/), find that Gemma can handle single prompts adequately but struggles with multiturn responses. Meanwhile, [@dirtytigerx](https://www.assistant.ai/) points out issues with the over-aligned instruct model and unexpectedly high VRAM usage, without indicating exact numbers.

- **Anticipation for AI Model Releases and Updates**: Users are contemplating the potential release of "Llama 3" in March and discussing Google's AI development choices, highlighting concerns over aspects such as contaminated models. Additionally, the new "retrieval_count" feature in PolyMind is well-received for its utility in tasks like GMing, offering multiple retrieval results for a broader scope of information.

- **Roleplay and Character Complexity in Chatbots**: Efforts to generate DPO data to improve roleplay scenarios for characters with secrets or lies are underway among users. There is also discussion regarding the challenges of maintaining character consistency within AI models and varied VRAM requirements for Miqu-70b models, with 32 GB mentioned for Q2 and 48 GB for Q5.

- **Dataset Editing and Model Training Woes**: User @pncdd is experiencing difficulty editing a complex synthetic dataset, while @3dhelios is exploring the inclusion of negative examples in training. Gradio was suggested as a potential solution for creating dataset editing tools, and the search for an effective classifier for a relevance filtering task was highlighted, with the prospect of using *deepseek-coder-6.7B-instruct*.

- **Technical Conversations on Coding and Model Optimization**: A local script for a chatbot featuring multiple coding assistants is being developed by @pilotgfx, where conversation history management is key. Discussions also focused on the utilization of RAG techniques, Mistral finetuning, editor preferences, with dissatisfaction with *VSCode* leading to a preference for **Zed**, and backend infrastructure optimization for models, where starting with rented **GeForce RTX 4090** GPUs was suggested.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Dual GPUs Need Ample Juice**: Users in the hardware discussion channel cautioned about **PSU requirements for dual GPU setups** and noted that in multi-GPU configurations the overall speed matches that of the slowest card. High-end GPUs like the **NVIDIA GeForce RTX 3090** are favored for AI tasks due to their significant VRAM.

- **Troubles and Fixes in LM Studio's Latest Beta**: LM Studio users reported various issues with the **LM Studio 0.2.15 Beta**, including problems with the `n_gpu_layers` feature and Gemma 7B models outputting gibberish. Version 0.2.15 has been re-released with bug fixes targeting these and other issues, and users are advised to redownload it from [LM Studio's website](https://lmstudio.ai/).

- **Gemma Model Integration Efforts Continue**: Google's Gemma model support has been added to LM Studio, and users are directed to manually download **Gemma models (2B and 7B versions)**, with a link to the 2B variant available at [Hugging Face](https://huggingface.co/google/gemma-2b-it). A recommended Gemma quant was also shared for easier integration.

- **Query and Request in AI Assistant Sphere**: On the topic of AI assistants, `@urchig` expressed interest in having a **creation feature** integrated into LM Studio, pointing to the existing feature on [Hugging Face](https://huggingface.co/chat/assistants). Additionally, some users encountered display issues with RAM and troubles with Visual Studio Code potentially related to **venv**.

- **Microsoft's Unknown Flying Object**: A user shared a link to Microsoft's UFO repository on GitHub: [UFO on GitHub](https://github.com/microsoft/UFO). The context and relevance to the discussion were not provided.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **ChatGPT's Context Conundrum**: After conducting experiments with the Big Lebowski script, `@elder_plinius` demonstrated that ChatGPT might limit contexts by characters, not tokens. This sparked discussion, with `@gabriel_syme` sharing a [GitHub repository](https://github.com/FranxYao/Long-Context-Data-Engineering) for scaling language models to 128K context, and varying claims about VRAM requirements for large context AI models.

- **AI-Driven Simulators and Self-Reflection**: Discussion on AI-driven game simulations, with `@nonameusr` amazed by OpenAI's Sora simulating Minecraft, and `@pradeep1148` linking a [video](https://www.youtube.com/watch?v=Eb7QF1nDWGU) about self-improving retrieval-augmented generation (Self RAG). Additionally, inquiries surfaced about training non-expert models on microscopic images for artistic purposes.

- **Library Releases and AI Model Evaluations**: `.beowulfbr` presented `mlx-graphs`, a GNN library optimized for Apple Silicon, and `burnytech` showcased Gemma 1.5 Pro's ability to learn to self-implement. Also discussed was a no-affiliation clarification regarding the A-JEPA AI model and an [arXiv paper](https://arxiv.org/abs/2402.10200) discussing elicitation of chain-of-thought reasoning in pre-trained LLMs without explicit prompting.

- **Hermes 2 Ascends**: **Nous Hermes 2** release was announced, displaying enhanced performance across various benchmarks. Pre-made **GGUFs** for Nous Hermes 2 were also made available [on HuggingFace](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF), with FluidStack receiving thanks for supporting computation needs.

- **LLM Strategies and Pitfalls**: Discussion explored the Gemma model's benchmarks, nuances of fine-tuning, LoRA usage, and hosting large models on custom infrastructure. The datasets were debated, particularly the challenges of editing a synthetic dataset and combining DPO with SFT in **Nous-Hermes**.

- **Discontent with Hosting Services**: Criticism arose against **Heroku**, with members like `@bfpill` voicing frustration without expanding on specific grievances.

- **Project Slowed by a Pet's Illness**: `@qnguyen3` expressed apologies for delays in **Project Obsidian** due to their cat's health issues and invited direct communication for project discussions.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Model Dual Capabilities Spark Interest**: Users are intrigued by diffusion models that can handle both prompt-guided image-to-image and text-to-image tasks, like Stable Diffusion. Google's release of **Gemma**, a family of open models, has also piqued interest, with Nvidia collaborating to optimize it for GPU usage.

- **License Wrestles and Governance Questions**: There's an ongoing discussion surrounding the licensing of models such as Google's **Gemma**, the potential for copyrighting models, and its implications for commercial use. Meanwhile, the governance of foundation models, including AGI, is drawing attention for policy development, like mandated risk disclosure standards.

- **Intelligence Benchmarks and Transformer Efficiency**: Debates heat up over the validity of current benchmarks, like MMLU and HellaSwag, for evaluating model intelligence. Users are also interested in finding the most information-efficient transformer models and comparing performance against MLPs and CNNs at various scales.

- **Uncovering Multilingual Model Mysteries**: There's curiosity over whether multilingual models, like **Llama-2**, are internally depending on English for tasks, with research approaches such as using a logit lens for insights being shared. Questions are also being asked about the potential for language-specific lens training.

- **Tweaking Code and Handling OOM**: In the realm of AI development, practitioners are tackling practical issues, such as resolving Out of Memory (OOM) errors during model evaluation and making tweaks to code for better model performance. Specifically, issues with running **Gemma 7b** in an evaluation harness and confusion regarding dropout implementation in **ParallelMLP** are points of discussion.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Community Engagement on the Upswing**: The [HuggingFace Discord community](https://discord.com/channels/879548962464493619/897387888663232554/1209615605653700689) has showcased active participation in prompt ranking with over **200 contributors** and **3500+ rankings**, helping to construct a communal dataset. A leaderboard feature adds a gaming dimension, fostering further interaction and contribution.

- **Technical Troubleshooting and Library Updates Gain Attention**: Amidst several technical inquiries, including issues with `huggingface-vscode` on NixOS and challenges in fine-tuning on Nvidia A100 GPUs, an intriguing tease was dropped about an **upcoming release of the transformers library**, suggesting new models and improved custom architecture support.

- **AI-Generated Art and Sign Language Translation Models Discussed**: Discussion in the **computer-vision channel** highlighted **image captioning resources using BLIP** and models for **sign language translation**—crucial tools for expanding AI's accessibility in communication and content generation.

- **AI Startup Showcases, AI's Use in Cybersecurity, and Financial Apps Designed**: Entrepreneurs and developers are creating value across various fields: an AI **startup showcase event at Data Council**, a model named WhiteRabbitNeo-13B-v1 aimed at cybersecurity, and an **investment portfolio management app** designed to assist in financial decisions.

- **Community Contributions to NLP and Diffusion Modeling Showcased**: Users are contributing innovative solutions like an Android app for **monocular depth estimation** and discussing advancements in **damage detection using stable diffusion models**—indicative of the robust collaborative environment within HuggingFace.

- **Lively Discussion on Enhanced Multimodal Models and Diffusion Techniques**: Participants within the **reading-group** and **diffusion-discussions** advocated for better multimodal models and shared insights on the intricacies of advanced diffusion techniques, including **Fourier transforms for timestep embeddings**.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Introducing LlamaCloud and Its Components**: LlamaIndex announced **LlamaCloud**, a new cloud service designed to enhance LLM and RAG applications by offering **LlamaParse** for handling complex documents, and has opened a managed API for private beta testing. Collaborators mentioned include Mendable AI, DataStax, MongoDB, Qdrant, NVIDIA, and contributors from the LlamaIndex Hackathon. Early access and resources can be found in the official [announcement](https://blog.llamaindex.ai/introducing-llamacloud-and-llamaparse-af8cedf9006b) and via the [tweet](https://x.com/llama_index/status/1759987390435996120?s=20).

- **Content Creation Tips for RAG Development**: A set of advanced cookbooks for setting up RAG with **LlamaParse** and **AstraDB** was mentioned, which can be accessed through the provided [cookbook guide](https://t.co/TvH5Hp62dx). A new comprehensive approach in simplifying RAG development pipelines was discussed with accessible slides shared through an [announcement tweet](https://t.co/vWnjQSkhUF), and a frontend tutorial for experts was linked with full support by LlamaIndex.

- **Navigating LlamaIndex and GitHub Quandaries**: Users have discussed topics ranging from the **LlamaIndex v0.10.x** update import path issues, the optimization and finetuning of LLM models within LlamaIndex, to potential solutions for high CPU usage and response latency for streaming agents in *response_gen*. For LLM finetuning queries, users were directed to [documentation](https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning.html#fine-tuning-llama-2-for-better-text-to-sql) and repository examples such as [llm_generators.py](https://github.com/run-llama/llama_index/blob/da5f941662b65d2e3fe2100f2b58c3ba98d49e90/llama-index-core/llama_index/core/question_gen/llm_generators.py#L10C5-L10C37).

- **Technical Deep Dives and AI Insights**: Members of the guild explored and shared insights into GPT-4's arithmetic and symbolic reasoning capabilities in a [blog post](https://loeber.substack.com/p/16-notes-on-arithmetic-in-gpt-4) and engaged in a technical discussion on Gemini 1.5's potential to assist with language translation, particularly for translating French into Camfranglais. Additionally, questions were raised about summarization metrics for evaluting Llamaindex performance.

- **Productivity and Language Processing Enhancements**: A blog post shared by `@andysingal` highlighted the integration of **Llamaindex**, React Agent, and Llamacpp for streamlined document management, readable [here](https://ai.gopubby.com/leveraging-llamaindex-step-wise-react-agent-for-efficient-document-handling-3a0f92b9ca22). This reflects ongoing dialogue within the community on how best to employ and combine various technologies to enhance document processing capabilities.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Groq It Like It's Hot**: The **Groq chip's** speed was highlighted, leveraging its sequential nature to potentially achieve thousands of tokens per second but noting challenges in scaling to larger models. Meanwhile, quantization was a hot topic, with suggestions that a quantized version of a model could outpace an fp16 version on certain accelerators.

- **Mistral-Next Sparks Interest and Concerns**: Community interactions revealed a mix of excitement and concern over **Mistral-Next**, including its brevity preference and censorship in newer language models. Access to Mistral-Next is currently limited to testing via **lymsys chat**, while the model itself is not yet available via API.

- **Openweights but Not Opensource**: **Mistral** models are considered openweights but not open source, with upcoming announcements expected about **Mistral-Next**. Discussions also touched on the intricacies of function calling in LLMs, and the challenges of adding new languages such as Hebrew to models due to limited pretraining on these tokens.

- **The Practicalities of AI**: Users shared their experiences and queries on deploying Mistral on platforms like AWS, with considerations for costs and hardware requirements, such as needing 86GB of VRAM to merge an 8x7b model. The challenges and suggested approaches to finetuning were also exchanged, recommending an iterative cycle and considering parameters affecting accuracy.

- **Sharing AI Experiences and Resources**: FUSIONL AI, an educational AI startup, was introduced, and a new library for integrating **Mistral AI** into Flutter apps was announced, indicating growth in tools for developers. Further, the importance of crafting effective prompts was underscored by sharing a guide on **prompting capabilities**. Concerns about self-promotion versus genuine project showcasing on community channels were also voiced. 

**Relevant Links**:
- [vLLM | Mistral AI Large Language Models](https://docs.mistral.ai/self-deployment/vllm/)
- [Prompting Capabilities | Mistral AI Large Language Models](https://docs.mistral.ai/guides/prompting-capabilities/)
- [GitHub - nomtek/mistralai_client_dart](https://github.com/nomtek/mistralai_client_dart)
- [FUSIONL AI Website](https://fusionlai.carrd.co)



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **GPT-4 Variants Stir Curiosity**: A user identified an issue with **GPT-4-1106** handling the `£` symbol in JSON format, with a bug causing truncated outputs or character encodings to change. Meanwhile, `@fightray` asked about the update from **GPT-3.5 Turbo** to **GPT-3.5 Turbo 0125**, which `@solbus` confirmed, [directing to the official documentation](https://platform.openai.com/docs/models/gpt-3-5-turbo) for more details.

- **AI, Please Keep to the Script!**: Advising on crafting AI responses for specific use cases, such as customer service prompts or RPG scenarios, `@darthgustav.` stressed the use of positive instructions and embedded template variables. Feedback suggested that **GPT-4-0613** outperformed **0125 turbo** for roleplay, and `@bambooshoots` helped `@razorbackx9x` with a prompt designed for programming assistance with Apple's Shortcuts app.

- **Policy Awareness in AI Draws Debate**: Critiques were raised regarding AI refusing tasks believed to be against policy, despite them being logical or suitable. `@darthgustav.` highlighted the importance of clear and positive task framing to manage potential misuse issues such as plagiarism.

- **Anticipation for Sora AI Brews Impatience**: Users discussed the availability of **Sora AI**, indicating no public release or API access date has been confirmed yet. There was also a mention of a **40 messages limit per 3 hours** imposed on using **GPT-4**, with discussions revolving around explaining subscription plans and usage caps.

- **AI User Experience Expectations vs. Reality**: A user expressed discontent with the lack of real-time sharing in the **ChatGPT Teams Plan**, reflecting a disparity between advertised and actual features. Additionally, mixed feelings about various AI tools were shared, with **GPT-4** being praised for math problem-solving, **Groq** being successful for generating Edwardian literature content, and **Gemini** criticized for unintentionally Shakespearean output.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Karpathy's New Tokenizer Tutorials Engage Engineers**: The AI community is keen on [Andrej Karpathy's new lectures](https://x.com/karpathy/status/1759996549109776702) on constructing a GPT tokenizer as well as a detailed examination of the Gemma tokenizer. Enthusiasm is high, suggesting a trend toward deepened understanding of language model inner workings among technical audiences.
  
- **Google Paginates New Chapter with “Gemma”**: Google's release of [Gemma on Huggingface](https://huggingface.co/blog/gemma) and subsequent terms of service discussions indicate that these new large language models (LLMs) have piqued significant interest in operational specifics and ethical use—key concerns for engineering professionals evaluating new AI tools.

- **Magic's Mystery Surrounds AI Coding Capabilities**: A [revealing article](https://www.theinformation.com/articles/the-magic-breakthrough-that-got-friedman-and-gross-to-bet-100-million-on-a-coding-startup) about Magic, an AI coding assistant reportedly surpassing the likes of Gemini and GPT-4, has the community speculating about its underlying mechanisms and the implications for the future of AI in software development.

- **Paper Club Spotlights Engineer-AI Collaboration**: A [recent paper](https://arxiv.org/abs/2312.14231) scrutinizing the partnership between AI and software engineers in creating product copilots spurred discussion, reflecting the engineering community's interest in AI's evolving role in product development and the challenges it presents.

- **Integration and Evaluation of AI Become Focal Points**: Conversations reveal a growing interest in the integration of AI tools like Gemini across Google Workspace, highlighting the importance of robust evaluation methods, tools, and models. This signifies a clear trend: as engineers are called to work with LLMs like Mistral, skills in evaluation and prompt programming are becoming essential in the industry.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **GPU Great Debate for Hospital Data Handling**: Yamashi leans towards purchasing **8x MI300X GPUs** for their higher VRAM, considering them for managing 15,000 VMs related to massive hospital data processes, while assessing compatibility with ROCm software.

- **Gemma Models Stir Interest and Skepticism**: The AI Collective discusses **Google's Gemma models**, stressing on the new Flash Attention v2.5.5 for fine-tuning on consumer GPUs, like RTX 4090. Concerns were raised about the models' licensure, output restrictions, and the compatibility of custom chat formats with existing tools.

- **LoRA+ Integration Buzz in Axolotl**: A [recent paper on LoRA+](https://arxiv.org/abs/2402.12354) is suggested for integration into Axolotl, with its promise of optimized finetuning for models with large widths. The need for the latest Transformers library version for Gemma model training is emphasized, and variations in Gemma's learning rate and weight decay values are noted from Google's documentation and Hugging Face's blog.

- **Collaborative Templates and Trouble in General Help**: Yamashi shares a link to a tokenizer_config.json that includes a chat template, and after a collaborative effort, shares an Alpaca chat template suitable for Axolotl. DeepSpeed step count confusion and formatting for finetuned model inference were also discussed, highlighting the need for USER and ASSISTANT formatting.

- **Loss Puzzles in DPO Training**: Concerns are raised by noobmaster29 over **DPO training logs** indicating low loss before one epoch is complete, posing questions of potential overfitting in the absence of evaluation data.

- **RunPod's Fetch Fiasco and Infinite Retries**: Casper_ai reports persistent image fetch errors with RunPod, leading to infinite retries, while c.gato suggests a lack of disk space might be the underlying cause for the image download failures.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Quest for All-Inclusive Git Data Stalls**: A quest for a dataset containing all public git repositories with their associated issues, comments, pull requests, and commits was surfaced by `@swaystar123`, but alas, the digital trail went cold with no answers provided.

- **Whereabouts of LAION 5B**: Curiosity peaked about the **LAION 5B** dataset with queries flying in about its availability, but the definitive update remained elusive in the galaxy of the guild. Meanwhile, [DFN 2B](https://huggingface.co/datasets/apf1/datafilteringnetworks_2b) was suggested as a stand-in, albeit not without its own access quirks.

- **The Sora Saga: Real or Synthetic**: A debate sprung up over **Sora**'s origins—courting questions on whether it's a spawn of real or synthetic training data. Clues like 'floaters' and static backgrounds fuel the speculation of synthetic elements amidst the outputs.

- **Synthetic Data—The Unsung Hero or Hidden Villain?**: In a twist, OpenAI references using synthetic data contradicting one guild member's beliefs, while others continued to weigh the pros and cons, pondering over the true nature and impact of synthetic data's place in AI model development.

- **Pushing the Envelope with AnyGPT**: Engaged engineers exchanged insights into the [AnyGPT project](https://junzhan2000.github.io/AnyGPT.github.io/), an initiative striving to process different modalities through a language model with token of discrete representations—a discussion sealed with a [YouTube demo](https://youtu.be/oW3E3pIsaRg).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Perplexity Pro Perks Peek**: Users like `@norgesvenn` vouched for **Perplexity Pro**'s speed and accuracy. Guidance was offered for new Pro users to access exclusive Discord channels via a link provided in user settings, though some links shared were incomplete.

- **Balancing Act in AI Invocation**: Debates arose about how to instruct **Perplexity AI** to harmonize search-result reliance with inherent AI creativity. A hybrid approach of utilizing specific instructions was suggested to encourage the AI to tap into its own repository of information.

- **Gemini Touted Over GPT-4**: Discussions highlighting the advantages of **Gemini Advanced** models over GPT-4 surfaced, with users praising updated models and their output styles. The discourse reflects an inclination towards Gemini's capabilities and growing preference within the community.

- **Image Generation Enigma**: Queries on how to generate images with **Perplexity AI** were directed to the "Generate Image" button, underscoring the AI's multimedia capabilities. However, the detailed procedures or examples were not fully clarified due to incomplete links.

- **API Quandaries and Quirks**: Technical discussions in the **pplx-api** channel uncovered a variety of user issues such as seeking increased **API request limits**, discrepancies between API and outdated webapp results, unaddressed requests for `stream: true` examples, and erratic gibberish responses from **pplx-70b-online**. Solutions or responses to these concerns were notably absent.




---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **CUDA Mode Embraces NVIDIA's Grace Hopper**: The CUDA MODE community, highlighted by `@andreaskoepf`, is inclusive to all GPU enthusiasts, further evidenced by NVIDIA's outreach with their Grace Hopper chips to `@__tinygrad__`, stirring interest and discussion. Related Tweet: [tinygrad's tweet](https://x.com/__tinygrad__/status/1760039877922758710).

- **Groq's LPU Sets New AI Performance Bar**: `@srns27` brought to light Groq's LPU groundbreaking performance in processing large language models. To provide a better understanding of this technological feat, `@dpearson` shared a [YouTube video](https://youtu.be/PKJYU9ecvWc?si=9BKG75HsaEGTVgMH) featuring Groq's Compiler Tech Lead Andrew Bitar's lecture on the subject.

- **Seeking Collaborators for Triton/Mamba Development**: `@srush1301` actively looks for collaborators to enhance the Triton/Mamba project, offering coauthorship and discussing goals such as adding a reverse scan option in Triton. The project's current status and tasks are outlined in [Annotated Mamba](https://srush.github.io/annotated-mamba/hard.html).

- **Optimizing PyTorch with Custom Kernels**: Discourse in the torch channel has revealed various tactics to accelerate PyTorch, including the use of custom kernels and Triton kernels. `@gogators.` and `@hdcharles_74684` shared insights and linked to a series of optimization-focused blog posts and related source code like [quantization details](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L269).

- **A New Frontier in Audio Semantic Analysis**: Within the youtube-recordings channel, `@shashank.f1` shared a YouTube discussion on the **A-JEPA AI model**, an innovative approach to deriving semantic knowledge from audio files, signifying advancements in the realm of AI's understanding of audio data. The discussed video can be found here: ["A-JEPA AI model: Unlock semantic knowledge from .wav / .mp3 file or audio spectrograms"](https://youtu.be/FgcN62LFzIU).

- **JAX Pallas Flash Attention Code Evaluation**: The jax channel was buzzing with inquiries about the `flash_attention.py` file seen on the JAX GitHub repository. Interest in its functionality and compatibility for GPUs was discussed, but users such as `@iron_bound` have faced challenges, including crashes due to the shape dimensions error. The file in question is available here: [jax flash_attention.py](https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py).

- **Ring Attention and Flash Attention Gather Engagement**: In the ring-attention channel, user `@ericuald` has shared a Colab on implementing ring attention while `@iron_bound`, `@lancerts`, and others engage in discussions to enhance understanding and troubleshoot potential algorithm issues—an active and cooperative effort to push forward the project is noticed. A dummy version of ring attention can be found at this [Google Colaboratory](https://colab.research.google.com/drive/1B9oD4oeuYqK5szEHfrS0VlnVbmLCO9HA) and the forward pass implementation by `@lancerts` in this [naive Python notebook](https://github.com/lancerts/Algo-From-Scratch/blob/main/flash_attn/naive_flash.ipynb).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **TypeScript and LangChain: A Hidden Miss?**: `@amur0501` questioned the efficacy of using **LangChain with TypeScript**, wary of missing out on Python-specific features. The community did not reach a consensus on whether `langchain.js` holds up to its Python counterpart.

- **Function Crafting with Mistrial**: Using **function calling on Open-Source LLMs (Mistral)** has practical demonstrations, as shared by `@kipkoech7`, which includes examples on local use and with Mistral's API. Refer to the [GitHub resource](https://github.com/AIAnytime/Function-Calling-Mistral-7B) for implementation insights.

- **Vector Database Indexing Dilemma**: `@m4hdyar` sought strategies on index updates for vector databases post-code alterations. `@vvm2264` proposed a **code chunk-tagging system** or a **1:1 mapping solution**, but no definitive strategy was highlighted.

- **NLP Resources Remain Outdated**: In the hunt for up-to-date NLP materials, surpassing **2022's offerings**, `@nrs9044` asked for recommendations. The latest libraries and advancements remain a topic with insufficient follow-up.

- **AzureChatOpenAI Configuration Woes**: Difficulty arose in configuring `AzureChatOpenAI` when `@smartge3k` faced a 'DeploymentNotFound' error within `ConversationalRetrievalChain`. Solutions remained elusive as community discussion ensued.

- **Pondering a Pioneering PDF Parser**: `@dejoma` expressed a desire to elevate the existing **PDFMinerPDFasHTMLLoader / BeautifulSoup** parser to a more refined level with a week of dedicated work, hoping to collaborate with like-minded individuals.

- **One-Man Media Machine**: `@merkle` highlighted how **LangChain's langgraph agent setup** can transform a solitary idea into a newsletter and tweets, citing a [tweet](https://x.com/michaeldaigler_/status/1759962223236038781?s=20) from `@michaeldaigler_` describing the process.

- **A Hint of an Enigmatic Endeavor**: `@pk_penguin` teased a potential new project or tool, vaguely inviting curiosity and private messages for those intrigued enough to explore the mysterious offering.

- **RAG Revisions via Reflection**: Self-reflection enhancement of RAG was the topic of a **YouTube tutorial** titled "Corrective RAG using LangGraph" posted by `@pradeep1148`, featuring methods for improving generative models. The discussion is supplemented with another video on [Self RAG using LangGraph](https://www.youtube.com/watch?v=Eb7QF1nDWGU).

- **Memory Matters: AI with Recall**: The development of chatbots with persistent memory was outlined by `@kulaone` in an article about integrating **LangChain, Gemini Pro, and Firebase**. For details on establishing chatbots with memories extending beyond live sessions, [read here](https://medium.com/@zps270/beyond-live-sessions-building-persistent-memory-chatbots-with-langchain-gemini-pro-and-firebase-19d6f84e21d3).

- **A Spark of Spark API Trouble**: An 'AppIdNoAuthError' troubled `@syedmujeeb` regarding the Spark API, with community pointers redirecting to the [respective LangChain documentation](https://python.langchain.com/docs/integrations/chat/sparkllm) for potential troubleshooting.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Gemma Models Spark Technical Interest**: `@johannhartmann` initiated a query about the training strategy for the intriguing **Gemma Models**, triggering a knowledge share including a [link to Google's open-source models](https://www.kaggle.com/models/google/gemma) by `@sebastian.bodza`. `@philipmay` raised concerns regarding their proficiency in non-English languages, while `@bjoernp` provided a [link that reveals Gemma's instruction version](https://huggingface.co/google/gemma-7b-it) and highlighted its 256k vocabulary size.

- **Aleph Alpha's Progress Under the Microscope**: An **Aleph Alpha Model update** led to discussions about potential enhancements, with `@devnull0` mentioning the recruitment of Andreas Köpf to Aleph Alpha as a positive indicator, and shared the [company's changelog](https://docs.aleph-alpha.com/changelog/). Conversely, `@_jp1_` expressed skepticism due to the absence of benchmarking data and instruction tuning in the updated models.

- **Gemma Falls Short in German Language Tests**: Empirical evaluations by `@_jp1_` and `@bjoernp` indicate that **Gemma's instruct version** struggles with the German language, as evidenced by poor results in the `lm_eval` test for German hellaswag, barely surpassing random chance. 

- **Seeking Speed: GPU Budget Sparks Benchmark Navigation**: `@johannhartmann` humorously lamented the GPU budget constraints, prompting a search for free or faster benchmarks. `@bjoernp` proposed using **vLLM** to accelerate the `lm-evaluation-harness-de`, although an outdated branch was found to be the culprit for slow test runs.

- **Committing to Performance Improvements**: `@bjoernp` acknowledged the outdated branch that omitted **vLLM** integration as a reason for sluggish benchmarking. They pledged to align the harness with the current main branch, estimating a few days to accomplish the update.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **A Waitlist or a Helping Hand**: **@wenquai** noted that to access a specific AI service, one has to either be on a waitlist or reach out to a Google Cloud representative.

- **Twitter Stirring AI Access Talk**: Rumors about gaining access to an AI service have been circulating on Twitter, with **@res6969** confirming the buzz and awaiting their own access.

- **AI Performance Anxiety**: Public feedback suggests an AI is suffering from accuracy and hallucination problems, as stated by **@res6969**, and **@thebaghdaddy** commented, indicating it's a common issue.

- **Navigating the CRM Maze for AI Enterprises**: **@frandecam** is in pursuit of CRM solutions for their AI business, considering options like **Salesforce**, **Zoho**, **Hubspot**, or **Pipedrive**, while **@res6969** advises against Salesforce.

- **Google Shares Open Model Gems**: **potrock** shared a [Google blog post](https://blog.google/technology/developers/gemma-open-models/) about open models available to developers.

- **Tuning Technique Talk**: **@dartpain** advocates for the use of **ContrastiveLoss** and **MultipleNegativesRankingLoss** when fine-tuning embeddings.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Neuralink Insider Tips Requested**: A member `@xilo0` is keen on insights for tackling the "evidence of exceptional ability" question in a late-stage **Neuralink** interview. They have a portfolio of projects but are seeking advice to stand out in a Musk-led venture.
  
- **RAG Gets Reflective**: `@pradeep1148` provided educational resources with two [YouTube videos](https://www.youtube.com/watch?v=eT31Cyaziy0) discussing **Corrective RAG** and **Self RAG using LangGraph**, highlighting self-reflection as a method to improve retrieval-augmented generation models.

- **Finetune Value Questioned**: In a [YouTube video](https://www.youtube.com/watch?v=T_dYzuv4N70), `@pradeep1148` shared insights on "**BitDelta: Your Fine-Tune May Only Be Worth One Bit**," questioning the value of fine-tuning large language models post extensive dataset pre-training.

- **Content Approval**: `@sabertoaster` showed appreciation for the shared RAG and BitDelta content with a simple "nice," indicating the content resonated with the community.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **AI Community Acknowledges One of Their Own**: `@swyxio` received recognition from peers in the AI field, marking a notable professional milestone celebrated within the community.
- **Grassroots Voices Sought in AI Lists**: `@tokenbender` argues for the inclusion of grassroots contributors in a corporate-dominated AI list, proposing `@abacaj` on Twitter as a worthwhile addition.
- **Mystery of the Missing Token**: `@scopexbt` is on a quest to discover if there's an elusive token tied to the group, but their search has come up empty so far.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Google Debuts Video-Innovative Gemini Pro 1.5**: Google has launched **Gemini Pro 1.5**, which impresses with a 1 million token context and the revolutionary ability to handle video inputs. **Simon Willison** has been exploring these features through [Google AI Studio](https://aistudio.google.com/app/prompts/new_chat) and described his experiences on his [blog](https://simonwillison.net/2024/Feb/21/gemini-pro-video/).

- **GLIBC Snag in GitHub Codespaces**: While attempting to run an llm in a GitHub codespace, `@derekpwillis` stumbled upon an **OSError** due to missing `GLIBC_2.32`, with the issue tracing back to a file within the `llmodel_DO_NOT_MODIFY` directory, and queried the group for potential fixes.

- **Engineers Have a Soft Spot for "Don't Touch" Labels**: The humorously named `llmodel_DO_NOT_MODIFY` directory received a shoutout from `@derekpwillis`, showcasing that even technical audiences appreciate a well-placed warning label.



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **Groq LPU Gains Traction**: Both `@juanreds` and a user going by `._z` shared favorable opinions on the **Groq LPU's** performance, with `@juanreds` providing a [test link](https://groq.com/) for others to gauge its impressive speed.
- **Gemini 1.5 Gathering Arranged**: `@shashank.f1` extended an invitation for the upcoming live discussion on **Gemini 1.5** with a [link to join the event](https://discord.gg/F4FfcQw3?event=1209440306404139008).
- **Diving Into Semantic Audio Analysis**: A session highlighting the **A-JEPA AI model**, specialized in extracting semantic knowledge from audio files, was recapped in a [YouTube video](https://youtu.be/FgcN62LFzIU).
- **Coordination Conundrum and Sponsorship Update**: `@yikesawjeez` expressed a conflict for a morning event and asked for a recording, while updating `@705561973571452938` on sponsorship status, mentioning one confirmed and three potential sponsors for an unspecified upcoming weekend event.



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1209413009584422952) (1156 messages🔥🔥🔥): 

- **Gemma Model Satisfaction is Mixed**: Users have mixed feelings about the Gemma models compared to Mistral. While [@itsme9316](https://www.assistant.ai/) acknowledges that Gemma can handle single prompts, they note multiturn responses are where it falls apart. [@dirtytigerx](https://www.assistant.ai/) mentions the over-aligned instruct model is challenging for even benign tasks and others mention the VRAM usage is unexpectedly high.
- **Concerns About Google's AI Models**: Despite having resources and specialists, there is a sentiment of disappointment conveyed by users like [@selea8026](https://www.assistant.ai/), questioning Google's decisions around AI development. [@alphaatlas1](https://www.assistant.ai/) criticizes their track record, citing models like Orca being contaminated.
- **Text-to-Video Models Discussion**: Discord users discuss Sora, a model perceived as a "world simulator." Yann LeCun, Meta's Chief AI Researcher, criticized this approach, suggesting that training a model by generating pixels is wasteful. Some users, including [@welltoobado](https://www.assistant.ai/), found Sora impressive for its video quality and longer generation capabilities compared to open-source text-to-video models.
- **Anticipation for Llama 3 (LL3)**: There's speculation and eagerness among users about the release of a new model known as "Llama 3." [@mrdragonfox](https://www.assistant.ai/) hints that March could be a potential release period, whereas others, such as [@kaltcit](https://www.assistant.ai/), suggest a more conservative estimate.
- **PolyMind's Updated RAG Retrieval**: The "retrieval_count" feature introduced by [@itsme9316](https://www.assistant.ai/) in PolyMind has been positively received, with users like [@netrve](https://www.assistant.ai/) finding it greatly useful for tasks like GMing. The feature allows retrieving multiple results which was beneficial for getting a broader scope of information.

**Links mentioned**:

- [no title found](https://pygmalion.chat>): no description found
- [Join the Stable Diffusion Discord Server!](https://discord.gg/stablediffusion): Welcome to Stable Diffusion; the home of Stable Models and the Official Stability.AI Community! https://stability.ai/ | 318346 members
- [Tweet from JB McGill (@McGillJB)](https://x.com/McGillJB/status/1760027310348972153?s=20): @yacineMTB Ho ho ho it gets better.
- [no title found](https://www.independent.co.uk/tech/chatgpt-status-reddit-down-gibberish-messages-latest-b2499816.html?utm_source=reddit.com>): no description found
- [HuggingChat](https://huggingface.co/chat/): Making the community's best AI chat models available to everyone.
- [Gemma: Introducing new state-of-the-art open models](https://blog.google/technology/developers/gemma-open-models/): Gemma is a family of lightweight, state\u002Dof\u002Dthe art open models built from the same research and technology used to create the Gemini models.
- [bartowski/sparsetral-16x7B-v2-exl2 · Hugging Face](https://huggingface.co/bartowski/sparsetral-16x7B-v2-exl2): no description found
- [serpdotai/sparsetral-16x7B-v2 · Hugging Face](https://huggingface.co/serpdotai/sparsetral-16x7B-v2): no description found
- [Leroy Worst Admin GIF - Leroy Worst Admin Admin - Discover &amp; Share GIFs](https://tenor.com/view/leroy-worst-admin-admin-gif-20501469): Click to view the GIF
- [Stable Cascade - a Hugging Face Space by ehristoforu](https://huggingface.co/spaces/ehristoforu/Stable-Cascade): no description found
- [HuggingFaceTB/cosmo-1b · Hugging Face](https://huggingface.co/HuggingFaceTB/cosmo-1b): no description found
- [VideoPrism: A Foundational Visual Encoder for Video Understanding](https://arxiv.org/abs/2402.13217): We introduce VideoPrism, a general-purpose video encoder that tackles diverse video understanding tasks with a single frozen model. We pretrain VideoPrism on a heterogeneous corpus containing 36M high...
- [google/gemma-7b · Hugging Face](https://huggingface.co/google/gemma-7b#training-dataset): no description found
- [OpenAI&#39;s &quot;World Simulator&quot; SHOCKS The Entire Industry | Simulation Theory Proven?!](https://www.youtube.com/watch?v=BH9FU7Gd6v8): OpenAI&#39;s Sora is described as a &quot;world simulator&quot; by OpenAI. It can potentially simulate not only our reality but EVERY reality. Use this limited-time deal t...
- [bartowski/sparsetral-16x7B-v2-SPIN_iter0-exl2 · Hugging Face](https://huggingface.co/bartowski/sparsetral-16x7B-v2-SPIN_iter0-exl2): no description found
- [bartowski/sparsetral-16x7B-v2-SPIN_iter1-exl2 · Hugging Face](https://huggingface.co/bartowski/sparsetral-16x7B-v2-SPIN_iter1-exl2): no description found
- [Tweet from sean mcguire (@seanw_m)](https://x.com/seanw_m/status/1760115118690509168?s=20): chatgpt is apparently going off the rails right now and no one can explain why
- [GitHub - amazon-science/tofueval](https://github.com/amazon-science/tofueval): Contribute to amazon-science/tofueval development by creating an account on GitHub.
- [GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui): Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.
- [Golem.de: IT-News f&uuml;r Profis](https://www.golem.de/news/texte-mit-ki-chatgpt-hat-den-verstand-verloren-2402-182456.html): no description found
- [THE DECODER](https://the-decoder.com/metas-chief-ai-researcher-says-openais-world-simulator-sora-is-a-dead-end/>): Artificial Intelligence is changing the world. THE DECODER brings you all the news about AI.
- [THE DECODER](https://the-decoder.com/metas-v-jepa-is-yann-lecuns-latest-foray-into-the-possible-future-of-ai/>): Artificial Intelligence is changing the world. THE DECODER brings you all the news about AI.

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1209413306171785246) (189 messages🔥🔥): 

- **Miqu's Rush vs Goliath's Pace**: `@sunija` expressed concerns that Miqu and its derivatives rush through scenes too quickly. Contrastingly, `@superking__` shared that usually original Miqu progresses scenes at a good pace, sometimes ending a story hastily which they mitigated by adjusting the prompt.

- **DPO Experiments for Roleplaying**: `@superking__` proposed creating DPO data for models to better roleplay characters who lie, lack knowledge, or have secrets. Various users expressed interest, with `@kaltcit` suggesting existing models like original llama chat already employ something akin to this concept of selective character knowledge.

- **Untruthful DPO**: `@superking__` described a detailed plan to create DPO pairs that differ with the minimal number of tokens to train models to lie about specific topics, bolstering selective character responses within roleplay scenarios. The approach suggests using mirrored cases to 'nudge' model weights while avoiding overtraining on unrelated tokens.

- **Model Behaviours and Secrets**: `@spottyluck` discussed the concept of LLMs keeping secrets and related challenges, citing a nuanced response from a model nicknamed "Frank" when questioned about its ability to keep secrets.

- **Resource Requirements for Miqu-70b Models**: Discussions about model VRAM usage revealed that miqu-70b models vary in their memory requirements, with `@superking__` mentioning 32 GB for Q2, while `@mrdragonfox` observed that Q5 could fit on a 48 GB GPU, indicating that model sizes and hardware capabilities can significantly impact user experiences.

- **AI Struggles in Character Consistency**: `@drakekard` sought advice for prompts to maintain consistent character role-play using Amethyst-13B-Mistral-GPTQ. `@superking__` suggested using a simple chat dialogue format and shared their settings, with varied results due to limitations in the smaller models.

**Links mentioned**:

- [Reservoir Dogs Opening Scene Like A Virgin [Full HD]](https://www.youtube.com/watch?v=J782jBp_pW0): Quentin Tarantino&#39;s Reservoir Dogs opening scene where Mr Brown (Tarantino) explains what &quot;Like A Virgin&quot; is about.
- [The Best Shopping Scene Ever! (from  Tarantino&#39;s  &quot;Jackie Brown&quot;)](https://www.youtube.com/watch?v=kGXkYEe0NsA): A scnen from Quentin Tarantino&#39;s &quot;Jackie Brown&quot; with Robert De Niro and Bridget Fonda in the scene

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1209433294576287744) (6 messages): 

- **Synthetic Dataset Fine-tuning Frenzy**: User `@pncdd` is struggling to review and edit a synthetic dataset for a data extraction model, finding the process tedious with jsonlines format. They consider converting to .csv for editing in Google Sheets, frustrated with the lack of suitable tools.

- **Desperate for Dataset Tools**: In a follow-up, `@pncdd` describes the dataset's complexity: it includes phone call transcriptions paired with detailed JSON responses.

- **Negative Training a Positive Step?**: User `@3dhelios` inquires whether it's possible to use negative examples in training datasets, implicating the need for the model to learn from wrong answers.

- **Gradio to the Rescue**: `@amogus2432` suggests that `@pncdd` could ask GPT-4 to write a simple Gradio tool for dataset editing, despite acknowledging GPT-4's limitations with Gradio blocks.

- **In Search of the Perfect Classifier**: `@yustee.` seeks model recommendations for a classification task aimed at filtering relevance in a RAG pipeline, pondering over the use of *deepseek-coder-6.7B-instruct*.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1209502243737440306) (166 messages🔥🔥): 

- **Exploring Chatbots with Multiple Characters**: `@pilotgfx` is creating a local script featuring multiple coding assistants that manage conversation histories without backend servers; conversation length management involves clearing out old entries to avoid excessive prompt length.
- **Intriguing Discussion on RAG and Long Conversations**: `@dirtytigerx` highlighted conventional Retrieval-Augmented Generation (RAG) techniques, such as metadata filtering and compression methods for managing extensive conversation histories; `@superking__` pointed out writing a server for Mixtral might address prompt evaluation lag issues.
- **Mistral Finetuning Exploration**: `@fred.bliss` and `@dirtytigerx` discussed the ease of use and distributed training features in Mistral; `@dirtytigerx` utilizes **macOS** for daily tasks and accesses ML-related workloads through other systems.
- **Backend Choices for Dev Environments Expressioned**: `@fred.bliss` and `@dirtytigerx` conversed about editor preferences, with `@dirtytigerx` expressing dissatisfaction with *VSCode*, leading to a preference for **Zed** and sharing the text editor's swiftness and developer focus. They also discussed perspectives on **mlx**'s performance and finetuning capabilities on Apple's new hardware.
- **Optimization Talk on Model Implementation**: `@etron711` inquired about fine-tuning Mistral models, seeking opinions on optimizing server costs and throughput; `@dirtytigerx` advised starting with a small prototype using rented **GeForce RTX 4090** GPUs to develop an MVP, before scaling up infrastructure.

**Links mentioned**:

- [GitHub - raphamorim/rio: A hardware-accelerated GPU terminal emulator focusing to run in desktops and browsers.](https://github.com/raphamorim/rio): A hardware-accelerated GPU terminal emulator focusing to run in desktops and browsers. - raphamorim/rio
- [GitHub - pulsar-edit/pulsar: A Community-led Hyper-Hackable Text Editor](https://github.com/pulsar-edit/pulsar): A Community-led Hyper-Hackable Text Editor. Contribute to pulsar-edit/pulsar development by creating an account on GitHub.

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1209416949667471434) (375 messages🔥🔥): 

- **CPU Features Debate**: `@jedd1` discussed CPU feature advancements, while `@exio4` mentioned Intel Atom's long-standing low performance, and `@krypt_lynx` added that even Celerons lack AVX features. These newer features aren't seen much outside of Intel's site.
- **LM Studio vs. Hugging Face Connection Issues**: While `@lmx4095` could access Hugging Face, they encountered connectivity errors with LM Studio; `@heyitsyorkie` confirmed issues with the model explorer.
- **Compatibility Questions in Discord**: Users `@nsitnov`, `@ivtore`, `@heyitsyorkie`, and `@joelthebuilder` discussed setting up web interfaces and integration with LM Studio Server, with mixed results.
- **LM Studio Model Recommendations and Issues**: Various users discussed the performance and issues of different models. Some had better luck than others, with reports of both successful model runs and errors.
- **Fixes and Patches for LM Studio**: `@yagilb` linked to recommended versions of Gemma for LM Studio and explained a recent fix to regeneration and continuation bugs in the latest LM Studio update.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/): Find, download, and experiment with local LLMs
- [lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF): no description found
- [Gemma: Introducing new state-of-the-art open models](https://blog.google/technology/developers/gemma-open-models/): Gemma is a family of lightweight, state\u002Dof\u002Dthe art open models built from the same research and technology used to create the Gemini models.
- [Hugging Face – The AI community building the future.](https://huggingface.co/): no description found
- [google/gemma-7b · Hugging Face](https://huggingface.co/google/gemma-7b): no description found
- [جربت ذكاء إصطناعي غير خاضع للرقابة، وجاوبني على اسئلة خطيرة](https://www.youtube.com/watch?v=to6FI5BseEc&t=61s&ab_channel=marouane53): ستريم كل نهار في تويتش :https://www.twitch.tv/marouane53Reddit : https://www.reddit.com/r/Batallingang/إنستغرام : https://www.instagram.com/marouane53/سيرفر ...
- [Head Bang Dr Cox GIF - Head Bang Dr Cox Ugh - Discover &amp; Share GIFs](https://tenor.com/view/head-bang-dr-cox-ugh-angry-gif-16823810): Click to view the GIF
- [LoneStriker/gemma-2b-GGUF · Hugging Face](https://huggingface.co/LoneStriker/gemma-2b-GGUF): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18oi2vd/question_about_system_ram_and_gpu_vram/): no description found
- [GitHub - lllyasviel/Fooocus: Focus on prompting and generating](https://github.com/lllyasviel/Fooocus): Focus on prompting and generating. Contribute to lllyasviel/Fooocus development by creating an account on GitHub.
- [Need support for GemmaForCausalLM · Issue #5635 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/5635): Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...
- [MSN](https://www.msn.com/en-us/news/technology/chatgpt-has-meltdown-and-starts-sending-alarming-messages-to-users/ar-BB1iDtsE?ocid=entnewsntp&pc=U531&cvid=c72557e2b33e491998be5116a12d196a&ei=31): no description found
- [Mistral&#039;s next LLM could rival GPT-4, and you can try it now in chatbot arena](https://the-decoder.com/mistrals-next-llm-could-rival-gpt-4-and-you-can-try-it-now-in-chatbot-arena/): French LLM wonder Mistral is getting ready to launch its next language model. You can already test it in chat.
- [How To Run Stable Diffusion WebUI on AMD Radeon RX 7000 Series Graphics](https://www.youtube.com/watch?v=kw0WT5sDBIY): Did you know you can enable Stable Diffusion with Microsoft Olive under Automatic1111 to get a significant speedup via Microsoft DirectML on Windows? Microso...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1209503441802764370) (65 messages🔥🔥): 

- **Query on Best Model for Lyrics Creation**: `@discockk` asked for the best model to generate lyrics. In response, `@fabguy` humorously criticized the brief query but indicated that there are no models specifically trained for poems or rhymes, suggesting to try storytelling models listed in another channel.
- **Hermes 2 Model Expertise**: `@wolfspyre` shared their experience using the new NousResearch Hermes2 Yi model, praising its performance and linking to its [Hugging Face page](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B-GGUF).
- **Conversation on LLM Output Quality and Verification**: `@goldensun3ds` contemplated a dual-LLM system to improve output quality, where a secondary LLM revises the primary LLM's response. `@jedd1` and `@christianazinn` mentioned existing frameworks like Judy and papers that align with the concept.
- **Challenges with AutoGPT and LLM Compatibilities**: `@thebest6337` reported difficulties using AutoGPT with various models, receiving help from `@docorange88` who suggested Mistral and a version of Dolphin, but later conversed about persistent errors and model compatibility.
- **Tech Details and Optimization Talks**: In a technical exchange, `@nullt3r` and `@goldensun3ds` discussed the potential bottlenecks when running large LLMs like Goliath 120B on GPUs with limited VRAM, involving factors like memory bandwidth and layer offloading.

**Links mentioned**:

- [lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF): no description found
- [Testing Shadow PC Pro (Cloud PC) with LM Studio LLMs (AI Chatbot) and comparing to my RTX 4060 Ti PC](https://youtu.be/Eaz-H-3FkZg): I have been using Chat GPT since it launched about a year ago and I&#39;ve become skilled with prompting, but I&#39;m still very new with running LLMs &quot;locally&quot;. Whe...
- [GitHub - TNT-Hoopsnake/judy: Judy is a python library and framework to evaluate the text-generation capabilities of Large Language Models (LLM) using a Judge LLM.](https://github.com/TNT-Hoopsnake/judy): Judy is a python library and framework to evaluate the text-generation capabilities of Large Language Models (LLM) using a Judge LLM. - TNT-Hoopsnake/judy
- [NousResearch/Nous-Hermes-2-Yi-34B-GGUF · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B-GGUF): no description found

  

---


### LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1209948313957175357) (3 messages): 

- **LM Studio v0.2.15 Drops with Google Gemma**: `@yagilb` announced that **LM Studio v0.2.15** is now available with support for Google's Gemma model, including a 2B and 7B version. Users must manually download Google's Gemma models ([2B version](https://huggingface.co/google/gemma-2b-it), [7B version](https://huggingface.co/google/gemma-7b-it)), but a more seamless experience is expected soon.

- **New Features & UI Updates Shine in Latest LM Studio**: The update introduces a new and improved downloader with pause/resume capabilities, a conversation branching feature, a GPU layers slider, and a UI refresh, including a new home page look and updated chat.

- **Bug Squashing in LM Studio**: Users are encouraged to redownload v0.2.15 from the [LM Studio website](https://lmstudio.ai) to obtain important bug fixes that weren't present in the original 0.2.15 build.

- **Easing Gemma Integration Pain Points**: `@yagilb` provided a link to [recommended Gemma quants](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF) that are now available for LM Studio users, aiming to streamline the integration of Google's new Gemma model into the LM Studio environment.

**Links mentioned**:

- [lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF): no description found
- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai): Find, download, and experiment with local LLMs

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1209602514241191996) (10 messages🔥): 

- **RAM Discrepancy Issue Raised**: User `@darkness8327` mentioned that the **RAM is not displaying correctly** in the software they are using.

- **Assistant Creation Feature Request**: `@urchig` inquired about the possibility of integrating **assistant creation** in LM Studio, similar to the feature available on [Hugging Face](https://huggingface.co/chat/assistants).

- **Instructions for Local LLM Installation Seeked**: `@maaxport` asked for guidance on **installing a local LLM** with AutoGPT on a rented server.

- **Update on Client Version Confusion**: `@msz_mgs` noted an issue where the client 0.2.14 was incorrectly indicating it is the latest version. `@heyitsyorkie` suggested manually downloading and installing the update as the in-app updating feature isn't working.

- **Gemma Model Troubleshooting**: `@richardchinnis` reported a problem with the Gemma models but then followed up with an intention to try a different model based on a discussion in another channel.

**Links mentioned**:

[HuggingChat - Assistants](https://huggingface.co/chat/assistants): Browse HuggingChat assistants made by the community.

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1209422245202755584) (96 messages🔥🔥): 

- **Power Supply Crucial for Dual GPUs**: `@heyitsyorkie` emphasized the importance of having a PSU with enough power to run both GPUs when considering a dual setup.
- **Multi-GPU Setups Inherit Slowest Card's Speed**: According to `@wilsonkeebs`, in a multi-GPU configuration, the overall speed will match that of the slowest card, but having more VRAM is preferable over utilizing RAM for loading models.
- **Motherboard PCIe Support Matters for GPU Expansion**: `@krzbio_21006` inquired if dual PCIe 4x16 slots were sufficient for GPU expansion, to which `@wilsonkeebs` responded affirmatively and suggested using a PCIe riser if space issues arise.
- **Power Efficiency vs. Performance in GPU Selection**: `@jedd1` and `@heyitsyorkie` discussed the trade-offs between power consumption and performance when choosing between GPU models like the 4060ti and the 4070 ti-s.
- **Optimal GPU Strategy for AI and Vision Models**: In a conversation about VRAM requirements, `@heyitsyorkie` and others highlighted the advantages of fewer, higher-end GPUs like the 3090 over numerous lower-end models, as higher VRAM per card significantly impacts AI-related tasks.

**Links mentioned**:

- [Have You GIF - Have You Ever - Discover &amp; Share GIFs](https://tenor.com/view/have-you-ever-condidered-piracy-gif-10055735): Click to view the GIF
- [The Best GPUs for Deep Learning in 2023 — An In-depth Analysis](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/.): Here, I provide an in-depth analysis of GPUs for deep learning/machine learning and explain what is the best GPU for your use-case and budget.
- [NVIDIA GeForce RTX 2060 SUPER Specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-2060-super.c3441): NVIDIA TU106, 1650 MHz, 2176 Cores, 136 TMUs, 64 ROPs, 8192 MB GDDR6, 1750 MHz, 256 bit
- [NVIDIA GeForce RTX 3090 Specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622): NVIDIA GA102, 1695 MHz, 10496 Cores, 328 TMUs, 112 ROPs, 24576 MB GDDR6X, 1219 MHz, 384 bit

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1209670257736679444) (301 messages🔥🔥): 

- **LM Studio 0.2.15 Beta User Struggles**: Users like `@n8programs` experienced inconsistencies with the `n_gpu_layers` feature, requiring frequent model reloads. To simplify GPU model loading, a preset option was suggested by `@yagilb`.
- **Linux Libclblast Bug Squashed**: `@yagilb` identified the root cause of the libclblast bug for Linux users and planned a fix, although it wasn't implemented in the 0.2.15 preview.
- **Trouble in Gemma Town**: Many users reported issues with Google's Gemma 7B models producing gibberish output. The issue was pervasive enough to be noticed by `@drawless111` who cited similar issues with other models like RWKV.
- **Regenerate Bug Resolved**: After multiple reports of issues with model regeneration and multi-turn chats, `@yagilb` announced that 0.2.15 has been re-released with significant bug fixes that aimed to resolve reported issues.
- **Official Gemma Model Functional, Quantized Variants a Mishap**: The Gemma model from Google worked well in its 32-bit GGUF full precision format, despite large file size; conversely, many quantized versions did not. Community members found `@LoneStriker`'s quantized models to be functional and `@yagilb` provided a Gemma 2B Instruct GGUF quantized by LM Studio and tested for compatibility with LM Studio.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/): Find, download, and experiment with local LLMs
- [lmstudio-ai/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF): no description found
- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai): Find, download, and experiment with local LLMs
- [asedmammad/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/asedmammad/gemma-2b-it-GGUF): no description found
- [```json{  &quot;cause&quot;: &quot;(Exit code: 1). Please check settings and try loading th - Pastebin.com](https://pastebin.com/2RrDRx3e): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
- [LoneStriker/gemma-2b-it-GGUF · Hugging Face](https://huggingface.co/LoneStriker/gemma-2b-it-GGUF): no description found
- [Thats What She Said Dirty Joke GIF - Thats What She Said What She Said Dirty Joke - Discover &amp; Share GIFs](https://tenor.com/view/thats-what-she-said-what-she-said-dirty-joke-joke-laugh-gif-15661968): Click to view the GIF
- [google/gemma-7b · Why the original GGUF is quite large ?](https://huggingface.co/google/gemma-7b/discussions/11): no description found
- [google/gemma-7b-it · Hugging Face](https://huggingface.co/google/gemma-7b-it): no description found
- [Need support for GemmaForCausalLM · Issue #5635 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/5635): Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...
- [HuggingChat](https://huggingface.co/chat/): Making the community's best AI chat models available to everyone.
- [Summer Break GIF - Summer break - Discover &amp; Share GIFs](https://tenor.com/view/summer-break-gif-6269967204638541467): Click to view the GIF
- [google/gemma-2b-it · Hugging Face](https://huggingface.co/google/gemma-2b-it): no description found
- [Tweet from Victor M (@victormustar)](https://x.com/victormustar/status/1694718952424001952?s=20): @LMStudioAI @yagilb The new font is great, but will still miss the OG (0.0.1!)
- [Add `gemma` model by postmasters · Pull Request #5631 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5631): There are couple things in this architecture:  Shared input and output embedding parameters. Key length and value length are not derived from n_embd.  More information about the models can be found...
- [no title found](https://releases.lmstudio.ai/windows/0.2.14/beta/LM-Studio-0.2.14-Setup-beta-v1.exe): no description found
- [no title found](https://releases.lmstudio.ai/linux/0.2.14/beta/LM_Studio-0.2.14-beta-v1.AppImage): no description found

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (1 messages): 

senecalouck: https://github.com/microsoft/UFO
  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1209425615833538571) (2 messages): 

- **Seeking Clarification on VSC Issues**: `@wolfspyre` is requesting clarification on a problem related to VSC (Visual Studio Code), asking for the anticipated outcome, the reasoning behind conclusions, and what solutions have been attempted.

- **Potential Virtual Environment Troubles**: `@dryt` mentions experiencing similar issues and speculates that the problem might be related to **venv** or **conda** when working on a specific project. They suggest that it could be a `venv` issue.
  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1209736378951401502) (67 messages🔥🔥): 

- **ChatGPT Character Limit Queries**: `@elder_plinius` found that the web version of ChatGPT seems to limit context by characters rather than tokens, which raised questions in the group. Despite `@vatsadev` asserting that GPT-3 and GPT-4 utilize tokenizers, the [Big Lebowski script fitting experiment](https://x.com/elder_plinius/status/1756436779056742863?s=46&t=Nf3Zw7IH6o_5y_YpAL5gew) by `@elder_plinius` displayed inconsistency in context length acceptance.
  
- **Scaling Language Models to 128K Context**: `@gabriel_syme` shared a [GitHub repository](https://github.com/FranxYao/Long-Context-Data-Engineering) that includes implementation details on scaling language models to handle 128K context.
  
- **VRAM Requirements for Large Context AI**: The conversation evolved around VRAM requirements for processing 128K context length with a 7B model; `@teknium` claimed it to require 600+GB for inference, and `@blackl1ght` noted successfully running inference on models at 64K context using around 28GB VRAM.

- **Server Issues Versus Context Length Misconceptions**: `@vatsadev` proposed that server latency rather than context length might be the issue with token acceptance when `@elder_plinius` mentioned repeated rejections of the original Big Lebowski script for exceeding context limits.
  
- **Token Compression and Replacement Language Theory**: `@elder_plinius` conjectured about creating a library of alphanumeric mappings to reduce token count by teaching models a new condensed language, stemming from the context-length experiments conducted with the Big Lebowski script.

**Links mentioned**:

- [gpt-tokenizer playground](https://gpt-tokenizer.dev): no description found
- [Tweet from Pliny the Prompter 🐉 (@elder_plinius)](https://x.com/elder_plinius/status/1756436779056742863?s=46&t=Nf3Zw7IH6o_5y_YpAL5gew): The Big Lebowski script doesn&#39;t quite fit within the GPT-4 context limits normally, but after passing the text through myln, it does!
- [GitHub - FranxYao/Long-Context-Data-Engineering: Implementation of paper Data Engineering for Scaling Language Models to 128K Context](https://github.com/FranxYao/Long-Context-Data-Engineering): Implementation of paper Data Engineering for Scaling Language Models to 128K Context - FranxYao/Long-Context-Data-Engineering

  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1209490084873641994) (22 messages🔥): 

- **Exploring AI-Driven Game Simulation**: `@nonameusr` shared a link to a [YouTube video](https://youtu.be/zz1KHp0CohQ) demonstrating **OpenAI's Sora** simulating **Minecraft** gameplay, expressing amazement at its understanding of game mechanics. Sora's capabilities include understanding the XP bar, item stacking, inventory slots, and replicating animations for in-game actions.
  
- **AI Misconceptions and Modded Content Influence**: While discussing the AI's grasp on **Minecraft,** `@afterhoursbilly` observed some inaccuracies but noted no major visual bugs, whereas `_3sphere` remarked on the surreal accuracy at first glance, despite the grid alignment issues.
  
- **Self-Improving Retrieval-Augmented Generation**: `@pradeep1148` linked to a [YouTube video titled "Self RAG using LangGraph"](https://www.youtube.com/watch?v=Eb7QF1nDWGU), which discusses how self-reflection can enhance retrieval-augmented generation by enabling correction of subpar retrievals or outputs.

- **From Microscopy to Artistry**: `@blackblize` inquired about the feasibility for non-experts to train models on microscope images for the purpose of creating artistic derivatives, seeking guidance in this endeavor.

- **Generating Avatars for Nous Models**: In response to `@stoicbatman`'s query regarding avatar image generation for Nous models, `@teknium` mentioned using **DALL-E** and an image-to-image method through **Midjourney** to create these visual representations.

**Links mentioned**:

- [Minecraft - Real vs OpenAI&#39;s Sora](https://youtu.be/zz1KHp0CohQ): Simulating digital worlds. Sora is also able to simulate artificial processes–one example is video games. Sora can simultaneously control the player in Minec...
- [Corrective RAG using LangGraph](https://www.youtube.com/watch?v=eT31Cyaziy0): Corrective RAGSelf-reflection can enhance RAG, enabling correction of poor quality retrieval or generations.Several recent papers focus on this theme, but im...
- [Self RAG using LangGraph](https://www.youtube.com/watch?v=Eb7QF1nDWGU): Self-reflection can enhance RAG, enabling correction of poor quality retrieval or generations.Several recent papers focus on this theme, but implementing the...
- [BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://www.youtube.com/watch?v=T_dYzuv4N70): Large Language Models (LLMs) are typically trained in two phases: pre-training on large internet-scale datasets, and fine-tuning for downstream tasks. Given ...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1209420041746128916) (49 messages🔥): 

- **Apple Silicon Optimized GNN Library Launches**: `.beowulfbr` highlighted the release of `mlx-graphs`, a library for running Graph Neural Networks (GNNs) on Apple Silicon, boasting up to **10x training speedup** as mentioned by [@tristanbilot](https://x.com/tristanbilot/status/1759891997094011291?s=46).
- **Gemma 1.5 Pro Learns to Self-Implement**: `burnytech` shared a post by [@mattshumer_](https://fxtwitter.com/mattshumer_/status/1759804492919275555), where *Gemma 1.5 Pro* was shown the codebase of a Self-Operating Computer and successfully explained and implemented itself into the repository.
- **The AI Stack Battles Intensify in 2023**: `burnytech` brought attention to @swyx's overview of the **Four Wars of the AI Stack**, covering business battles in data, GPG/inference, multimodality, and RAG/Ops based on their [December 2023 recap](https://latent.space/p/dec-2023).
- **A-JEPA AI Model Discussion Without Meta Affiliation Clarified**: In the discussion about an AI model named *A-JEPA*, `@ldj` clarified it has no affiliation with Meta or Yann Lecun, contrary to what the name might imply. `shashank.f1` concurred, acknowledging the author isn't from Meta.
- **Decoding Process May Elicit CoT Paths in Pre-Trained LLMs**: `mister_poodle` referenced an [arXiv paper](https://arxiv.org/abs/2402.10200) that proposes eliciting chain-of-thought reasoning paths by altering the decoding process in large language models.

**Links mentioned**:

- [Tweet from Sundar Pichai (@sundarpichai)](https://fxtwitter.com/sundarpichai/status/1760288967352598843?t=dOvFXh4oPnnAZxjouwfMyQ&s=19): Introducing Gemma - a family of lightweight, state-of-the-art open models for their class built from the same research & tech used to create the Gemini models.   Demonstrating strong performance acros...
- [Library of
            Congress Subject Headings - LC Linked Data Service: Authorities and Vocabularies | Library of Congress](https://id.loc.gov/authorities/subjects.html): no description found
- [benxh/us-library-of-congress-subjects · Datasets at Hugging Face](https://huggingface.co/datasets/benxh/us-library-of-congress-subjects): no description found
- [Tweet from Tristan Bilot (@tristanbilot)](https://x.com/tristanbilot/status/1759891997094011291?s=46): We’re happy to officially release mlx-graphs, a library for running Graph Neural Networks (GNNs) efficiently on Apple Silicon.   Our first benchmarks show an up to 10x training speedup on large graph ...
- [Tweet from Matt Shumer (@mattshumer_)](https://fxtwitter.com/mattshumer_/status/1759804492919275555): I showed Gemini 1.5 Pro the ENTIRE Self-Operating Computer codebase, and an example Gemini 1.5 API call.  From there, it was able to perfectly explain how the codebase works...  and then it implemente...
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200): In enhancing the reasoning capabilities of large language models (LLMs), prior research primarily focuses on specific prompting techniques such as few-shot or zero-shot chain-of-thought (CoT) promptin...
- [Join the hedwigAI Discord Server!](https://discord.gg/F4FfcQw3?event=1209440306404139008): Check out the hedwigAI community on Discord - hang out with 45 other members and enjoy free voice and text chat.
- [A-JEPA AI model: Unlock semantic knowledge from .wav / .mp3 file or audio spectrograms](https://youtu.be/FgcN62LFzIU): 🌟 Unlock the Power of AI Learning from Audio ! 🔊 Watch a deep dive discussion on the A-JEPA approach with Oliver, Nevil, Ojasvita, Shashank, Srikanth and N...
- [Tweet from swyx (@swyx)](https://fxtwitter.com/swyx/status/1744467383090372743/photo/1): 🆕 The Four Wars of the AI Stack  https://latent.space/p/dec-2023  Our Dec 2023 recap also includes a framework for looking at the key business battlegrounds of all of 2023:  In Data:   with OpenAI an...
- [HuggingFaceTB/cosmo-1b · Hugging Face](https://huggingface.co/HuggingFaceTB/cosmo-1b): no description found
- [Let&#39;s build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=PvjKGVX_zdovBCxV): The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizer...
- [mlabonne/OmniBeagle-7B · MT-Bench Scores](https://huggingface.co/mlabonne/OmniBeagle-7B/discussions/1): no description found

  

---


### Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1209648766638886932) (2 messages): 

- **Introducing Nous Hermes 2**: `@teknium` announced the release of **Nous Hermes 2 - Mistral 7B - DPO**, an in-house RLHF'ed model improving scores on benchmarks like AGIEval, BigBench Reasoning Test, GPT4All suite, and TruthfulQA. The model is available on [HuggingFace](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO).

- **Get Your GGUFs!**: Pre-made GGUFs (gradient-guided unfreeze) of all sizes for Nous Hermes 2 are available for download at their [HuggingFace repository](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF).

- **Big thanks to FluidStack**: `@teknium` expressed gratitude to the compute sponsor FluidStack and their representative, along with shout-outs to contributors to the Hermes project and the open source datasets.

- **Together hosts Nous Hermes 2**: `@teknium` informed that Together.xyz has listed the new Nous Hermes 2 model on their API, available on the [Togetherxyz API Playground](https://api.together.xyz/playground/chat/NousResearch/Nous-Hermes-2-Mistral-7B-DPO). Thanks were extended to `@1081613043655528489`.

**Links mentioned**:

- [TOGETHER](https://api.together.xyz/playground/chat/NousResearch/Nous-Hermes-2-Mistral-7B-DPO): no description found
- [NousResearch/Nous-Hermes-2-Mistral-7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO): no description found
- [NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF): no description found

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1209414156235710484) (594 messages🔥🔥🔥): 

- **Gemma Joins the Fray**: Google's release of the Gemma model caused a stir among members, comparing it to Mistral models and discussing the potential behind the architecture and its benchmarks. `@ldj` noted that Gemma is slightly worse when parameters are accounted for, `@teknium` suggested focusing on models' effects rather than their raw parameter count, and `@leontello` eagerly awaited MT-bench or Alpaca eval to see performance data.
- **Finetuning Frenzy**: A discussion ensued on finetuning finetuned LLMs, with several members like `@lee0099` and `@mihai4256` sharing their experiences and strategies, mentioning use of Lora and full parameter finetuning. There was curiosity if DPO (Differential Privacy Optimization) could be combined with SFT (Supervised Fine Tuning).
- **Dead Social Media Concept**: `@n8programs` introduced a project called Deadnet, intending to create an endless stream of AI-generated social media content of fictional people. `@everyoneisgross` responded positively, considering possible expansion to user clustering and post ranking within the imagined social media universe.
- **Training Tales and Tools**: There was an exchange on various AI training tools and methods. `@lee0099` and `@mihai4256` discussed specifics of templates and finetuning outcomes, while `@thedeviouspanda` teased novel methods merging DPO with SFT for better training results.
- **Peeking into Protocol and the Pursuit of Synthetic**: The topic of synthetic versus organic data arose, with speculation on evolving internet dynamics due to AI input (`@everyoneisgross`). `@sdan.io` highlighted the concept of a personal 'vector db file' for memory management across platforms, while `@teknium` linked a tweet suggesting developments in AI reasoning models may become more prevalent than natural language processing models.


**Links mentioned**:

- [EleutherAI/Hermes-RWKV-v5-7B · Hugging Face](https://huggingface.co/EleutherAI/Hermes-RWKV-v5-7B): no description found
- [NousResearch/Nous-Hermes-2-Mistral-7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO): no description found
- [no title found](https://ai.google.dev/gemma/prohibited_use_policy): no description found
- [Models - Hugging Face](https://huggingface.co/models?other=gemma&sort=trending&search=google): no description found
- [eleutherai](https://wandb.ai/eleutherai/rnn-hermes/runs/rptfh8c7): Weights & Biases, developer tools for machine learning
- [Tweet from Archit Sharma (@archit_sharma97)](https://x.com/archit_sharma97/status/1759990862623453568?s=46): High-quality human feedback for RLHF is expensive 💰. AI feedback is emerging as a scalable alternative, but are we using AI feedback effectively?  Not yet; RLAIF improves perf *only* when LLMs are SF...
- [TOGETHER](https://api.together.xyz/playground/chat/NousResearch/Nous-Hermes-2-Mistral-7B-DPO): no description found
- [Tweet from Aaditya Ura (Ankit) (@aadityaura)](https://x.com/aadityaura/status/1760305308927426903?s=20): The new Model Gemma from @GoogleDeepMind @GoogleAI  does not demonstrate strong performance on medical/healthcare domain benchmarks.  A side-by-side comparison of Gemma by @GoogleDeepMind  and Mistral...
- [The Novice's LLM Training Guide](https://rentry.org/llm-training): Written by Alpin Inspired by /hdg/'s LoRA train rentry This guide is being slowly updated. We've already moved to the axolotl trainer. The Basics The Transformer architecture Training Basics Pre-train...
- [no (no sai)](https://huggingface.co/No): no description found
- [Adding Google&#39;s gemma Model by monk1337 · Pull Request #1312 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1312): Adding Gemma model config https://huggingface.co/google/gemma-7b Testing and working!
- [Runtime error: CUDA Setup failed despite GPU being available (bitsandbytes) · Issue #1280 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1280): Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior Hi, I&#39;m trying the public cloud example that tr...
- [Mistral-NEXT Model Fully Tested - NEW KING Of Logic!](https://youtu.be/o4yoPEjIYRk): Mistral quietly released their newest model &quot;mistral-next.&quot; Does it outperform GPT4?Need AI Consulting? ✅ - https://forwardfuture.ai/Follow me on Twitter 🧠 ...
- [laserRMT/examples/laser-dolphin-mixtral-2x7b.ipynb at main · cognitivecomputations/laserRMT](https://github.com/cognitivecomputations/laserRMT/blob/main/examples/laser-dolphin-mixtral-2x7b.ipynb): This is our own implementation of &#39;Layer Selective Rank Reduction&#39; - cognitivecomputations/laserRMT
- [Reddit - Dive into anything](https://www.reddit.com/r/OpenAI/comments/1avwdi4/wtf_chat_gpt_starts_talking_crazy_out_of_nowhere/): no description found
- [Neuranest/Nous-Hermes-2-Mistral-7B-DPO-BitDelta at main](https://huggingface.co/Neuranest/Nous-Hermes-2-Mistral-7B-DPO-BitDelta/tree/main): no description found
- [BitDelta](https://fasterdecoding.github.io/BitDelta/): no description found
- [GitHub - FasterDecoding/BitDelta](https://github.com/FasterDecoding/BitDelta/tree/main): Contribute to FasterDecoding/BitDelta development by creating an account on GitHub.

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1209432065154752512) (37 messages🔥): 

- **Dataset Editing Dilemma**: `@pncdd` voiced frustration about not having a tool to easily edit a synthetic dataset for model fine-tuning, finding scrolling through a jsonlines file inefficient. Neither Huggingface datasets nor wandb Tables provided a solution, and converting to .csv for Google Sheets editing was suggested as a possible, albeit cumbersome, workaround.
  
- **Merge Woes with Large Models**: `@iamcoming5084` reported an Out of Memory (OOM) error when attempting to merge a finetuned mixtral 8x 7b model using an H100 80GB GPU, sparking a discussion about potential solutions that don't involve larger VRAM GPUs. The conversation involved suggested code, usage of Axolotl's merge functionality, and a close look at PyTorch memory handling.

- **Exploring LORA for LLM Inference**: `@blackl1ght` engaged in a discussion about the purpose and benefits of using LoRA (Locally Reweighted Adaptation) for fine-tuning, particularly during inference. The information shared clarified LoRA as a strategy for fine-tuning models with less GPU resource, with `@dysondunbar` providing insights into potential use cases, limitations, and benefits.

- **DeepSeek Enhanced With Magicoder**: `.benxh` indicated that the deepseek AI has been significantly improved using the Magicoder dataset on its 6.7B variant. `.benxh` denied personally using the dataset for fine-tuning deep seek, clarifying that the Magicoder team had implemented it.

- **Custom Hosting Solutions for LLMs**: `@jacobi` sought advice on the best strategy for hosting the Mixtral 8x7b model via an OpenAI API endpoint on a 3090/4090 GPU. Various tools and libraries like tabbyAPI, vLLM gptq/awq, and llama-cpp's server implementation were mentioned concerning their effectiveness and limitations for hosting such large models.

- **Inquiry About Hermes DPO with Gemma Base**: `@samin` inquired whether there would be a Nous-Hermes DPO model utilizing Google's newly released Gemma 7B base, expressing skepticism about Google's instruction tuning and linking to the Hugging Face blog announcement about Gemma.

**Links mentioned**:

- [no title found](https://titanml.co)): no description found
- [Welcome Gemma - Google’s new open LLM](https://huggingface.co/blog/gemma): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18oj983/why_arent_loras_a_big_thing_i_the_llm_realm/): no description found

  

---


### Nous Research AI ▷ #[collective-cognition](https://discord.com/channels/1053877538025386074/1154961277748256831/1209995589236822117) (3 messages): 

- **Heroku Faces Criticism**: `@bfpill` expressed frustration with **Heroku**, simply stating, *"screw heroku"*.
- **Neutral Response to Criticism**: `@adjectiveallison` responded to `@bfpill`'s Heroku comment, indicating a desire to move past the issue with, *"I don't think that's the point but sure"*.
- **Consensus on Heroku Sentiment**: Following the response from `@adjectiveallison`, `@bfpill` reaffirmed their sentiment agreeing with the frustration towards Heroku.
  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1209791740513624094) (3 messages): 

- **Project Delay Due to Pet Illness**: `@qnguyen3` apologized for the slow progress on the project as their **cat got sick**, which affected their ability to update and complete model tasks.
- **Open for Direct Messages**: `@qnguyen3` invited team members to **send a DM** if they need to discuss something directly with them.
  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1209436761814736896) (146 messages🔥🔥): 

- **Interest in Diffusion Models with Dual Capabilities**: `@swiftlynx` inquired about diffusion models capable of both prompt-guided image-to-image and text-to-image tasks, mentioning Stable Diffusion's approach using CLIP and seeking alternatives.
  
- **Searching for Video Segmentation Models with Temporal Coherence**: `@the_alt_man` asked for resources on models capable of video segmentation with smooth temporal coherence, alluding to a paper with a presumably successful implementation.

- **Governance of Foundation Models**: `@hp1618` from the Center for AI and Digital Policy, interested in the governance of foundation models, sought collaborators for policy ideas, such as mandated risk disclosure standards.

- **Discourse on the Terminology of "Foundation Model"**: Users `@_inox`, `@catboy_slim_`, and `@hp1618` discussed the origins and varying definitions of the term "foundation model," with an emphasis on its relatively recent adoption and implications for AI governance and policy.

- **New Model "Gemma" by Google Debuts**: `@jckwind` shared a link to Sundar Pichai's tweet about Google's release of Gemma, a family of open models, and discussed the competitive landscape that includes Google, Meta, and potentially OpenAI's offerings.

- **Concerns Over Benchmark Relevance for Models**: Several users, including `@rallio.` and `@fern.bear`, debated the validity and issues associated with popular benchmarks like MMLU and HellaSwag, noting that some evaluation questions may be flawed or ambiguous.

**Links mentioned**:

- [Discord | Your Place to Talk and Hang Out](https://discord.gg/RjVuhxyzkW)): Discord is the easiest way to talk over voice, video, and text. Talk, chat, hang out, and stay close with your friends and communities.
- [Tweet from Sundar Pichai (@sundarpichai)](https://x.com/sundarpichai/status/1760288967352598843?s=46): Introducing Gemma - a family of lightweight, state-of-the-art open models for their class built from the same research & tech used to create the Gemini models.   Demonstrating strong performance acros...
- [performant](https://dictionary.cambridge.org/us/dictionary/english/performant): 1. (of technology, etc.) working in an effective way:  2. (of technology, etc&hellip;
- [GitHub - mlabonne/llm-autoeval: Automatically evaluate your LLMs in Google Colab](https://github.com/mlabonne/llm-autoeval): Automatically evaluate your LLMs in Google Colab. Contribute to mlabonne/llm-autoeval development by creating an account on GitHub.
- [LLM-Benchmark-Logs/benchmark-logs/Qwen-72B-base.md at main · teknium1/LLM-Benchmark-Logs](https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/Qwen-72B-base.md): Just a bunch of benchmark logs for different LLMs. Contribute to teknium1/LLM-Benchmark-Logs development by creating an account on GitHub.
- [LLM-Benchmark-Logs/benchmark-logs/Deepseek-LLM-67b-base.md at main · teknium1/LLM-Benchmark-Logs](https://github.com/teknium1/LLM-Benchmark-Logs/blob/main/benchmark-logs/Deepseek-LLM-67b-base.md): Just a bunch of benchmark logs for different LLMs. Contribute to teknium1/LLM-Benchmark-Logs development by creating an account on GitHub.

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1209423206981570590) (350 messages🔥🔥): 

- **Debating Model Licensing**: `@rallio.` and `@catboy_slim_` engaged in a debate about the challenges and considerations of using models like Google's Gemma with contractual stipulations, versus using a transformer without Google's direct permission. There was discussion over whether models can or should be copyrighted and what the implications may be for commercial use. 
- **Exploring More From Less**: `@jckwind` solicited feedback on finding the most information-efficient transformer models, leading to discussions about the difference in performance between MLPs, CNNs, and transformers at various scales. The idea of optimizing models using concepts like curiosity was also floated.
- **The Quest for AGI Definition and Realization**: Various users, including `@jckwind` and `@rallio.`, debated the definition of Artificial General Intelligence (AGI) and speculated on what constitutes AGI or even superintelligence. The conversation ranged from the theoretical aspects to practical considerations regarding the capabilities and benchmarks for intelligence.
- **Nvidia Collaborating on Gemma**: `@kd90138` shared that Nvidia and RTX are working to bring Gemma optimized for GPU use, which `@rallio.` pointed out is officially announced and not hearsay.
- **Understanding and Hypothetical Uses of LLMs**: Users discussed the differences in LLM behaviors and capabilities. `@jckwind` mused on how models like Gemini might approach tasks like summarizing a long document or explaining a magic trick, contemplating the path towards more AGI-like abilities.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=39435818>): no description found
- [no title found](https://news.ycombinator.com/item?id=39436019>): no description found
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
- [Neural Network Diffusion](https://arxiv.org/abs/2402.13144): Diffusion models have achieved remarkable success in image and video generation. In this work, we demonstrate that diffusion models can also \textit{generate high-performing neural network parameters}...
- [GH Archive](https://www.gharchive.org): no description found
- [Decoding In-Context Learning: Neuroscience-inspired Analysis of Representations in Large Language Models](https://arxiv.org/abs/2310.00313): Large language models (LLMs) exhibit remarkable performance improvement through in-context learning (ICL) by leveraging task-specific examples in the input. However, the mechanisms behind this improve...
- [SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds](https://arxiv.org/abs/2306.00980): Text-to-image diffusion models can create stunning images from natural language descriptions that rival the work of professional artists and photographers. However, these models are large, with comple...
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150): Multi-head attention layers, as used in the Transformer neural sequence model, are a powerful alternative to RNNs for moving information across and between sequences. While training these layers is ge...
- [Join the GroqCloud Discord Server!](https://discord.gg/xQe8UVkRGt): Groq provides the world&#x27;s fastest AI inference. | 987 members
- [Experts Don&#39;t Cheat: Learning What You Don&#39;t Know By Predicting Pairs](https://arxiv.org/abs/2402.08733): Identifying how much a model ${\widehat{p}}_θ(Y|X)$ knows about the stochastic real-world process $p(Y|X)$ it was trained on is important to ensure it avoids producing incorrect or &#34;hallucinated&#...
- [Feist Publications, Inc., v. Rural Telephone Service Co. - Wikipedia](https://en.wikipedia.org/wiki/Feist_Publications,_Inc.,_v._Rural_Telephone_Service_Co.): no description found
- [Gemma: Introducing new state-of-the-art open models](https://blog.google/technology/developers/gemma-open-models/): Gemma is a family of lightweight, state\u002Dof\u002Dthe art open models built from the same research and technology used to create the Gemini models.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1avqw1a/huggingface_releases_cosmopedia_the_largest/): no description found
- [Tweet from NVIDIA (@nvidia)](https://x.com/nvidia/status/1760331965994020946?s=20): Announced today, we are collaborating as a launch partner with @Google in delivering Gemma, an optimized series of models that gives users the ability to develop with #LLMs using only a desktop #RTX G...
- [Lecture 20 - Efficient Transformers | MIT 6.S965](https://youtu.be/RGUCmX1fvOE?si=wcs1MDNbon1URKsO): Lecture 20 introduces efficient transformers.Keywords: TransformerSlides: https://efficientml.ai/schedule/---------------------------------------------------...
- [Google Cloud Blog](https://cloud.google.com/blog/topics/public-datasets/github-on-bigquery-analyze-all-the-open): no description found
- [Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models](https://arxiv.org/abs/2402.13064): We introduce Generalized Instruction Tuning (called GLAN), a general and scalable method for instruction tuning of Large Language Models (LLMs). Unlike prior work that relies on seed examples or exist...
- [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588): We ask whether multilingual language models trained on unbalanced, English-dominated corpora use English as an internal pivot language -- a question of key importance for understanding how language mo...
- [generative modelling of compressed image file bits](https://theadamcolton.github.io/generative-modelling-of-compressed-image-file-bits): no description found
- [Instruction-tuned Language Models are Better Knowledge Learners](https://arxiv.org/abs/2402.12847): In order for large language model (LLM)-based assistants to effectively adapt to evolving information needs, it must be possible to update their factual knowledge through continued training on new dat...
- [Groq Inference Tokenomics: Speed, But At What Cost?](https://www.semianalysis.com/p/groq-inference-tokenomics-speed-but): Faster than Nvidia? Dissecting the economics
- [
      My benchmark for large language models
    ](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html): no description found
- [GitHub on BigQuery: Analyze all the open source code | Google Cloud Blog](https://cloud.google.com/blog/topics/public-datasets/github-on-bigquery-analyze-all-the-open-source-code): no description found

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1209826662418751528) (38 messages🔥): 

- **Exploring LLM's Internal English Dependency**: `@butanium` shared a [Twitter link](https://twitter.com/cervisiarius/status/1759989584371298554) suggesting that multilingual models like Llama-2 might internally depend on English, as demonstrated by logitlens favoring English tokens for French to Chinese translations.
- **Tuned vs Logit Lens for Model Insights**: `@butanium` highlighted a research approach using logit lens to determine if Llama-2 accesses English internal states during non-English tasks; the tuned lens would obscure such insights as it maps internal states to non-English predictions.
- **Stella Deck Comparison Announced**: `@butanium` reacted to a [Twitter post by @BlancheMinerva](https://twitter.com/BlancheMinerva/status/1760002505214214352) which seemingly depicted Stella already set up to perform related experiments.
- **Replicating Research with Available Code**: `@mrgonao` is willing to run experiments to replicate study findings using provided code, but identified missing components including a tuned lens for the 70b model and a separate repetition task notebook.
- **Seeking Data for Language-Specific Lens Training**: Discussion between `@stellaathena` and `@mrgonao` about the possibility of training a lens specifically for Chinese and considering the use of datasets like the Chinese shard of mC4 on Hugging Face.

**Links mentioned**:

- [phoeniwwx/tuned_lens_q · Hugging Face](https://huggingface.co/phoeniwwx/tuned_lens_q): no description found
- [shjwudp/chinese-c4 · Datasets at Hugging Face](https://huggingface.co/datasets/shjwudp/chinese-c4): no description found
- [AlignmentResearch/tuned-lens at main](https://huggingface.co/spaces/AlignmentResearch/tuned-lens/tree/main/lens/meta-llama): no description found
- [srgo - Overview](https://github.com/SrGo): srgo has one repository available. Follow their code on GitHub.
- [GitHub - epfl-dlab/llm-latent-language: Repo accompanying our paper &quot;Do Llamas Work in English? On the Latent Language of Multilingual Transformers&quot;.](https://github.com/epfl-dlab/llm-latent-language): Repo accompanying our paper &quot;Do Llamas Work in English? On the Latent Language of Multilingual Transformers&quot;. - epfl-dlab/llm-latent-language

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1209466925177110539) (76 messages🔥🔥): 

- **Inquiries about MCQ Dataset Evaluation**: `@aloo_kachalu.` sought advice on how to calculate likelihood for MCQ dataset evaluation and was pointed by `@baber_` to a [Hugging Face blog](https://huggingface.co/blog/evaluating-mmlu-leaderboard) for a detailed explanation.

- **Log Likelihood's Availability in Evaluation Harness**: `@hailey_schoelkopf` explained that tasks requiring log likelihood calculations for model evaluations, like `loglikelihood`, `loglikelihood_rolling`, `multiple_choice`, cannot be run on API models that do not provide log probabilities (logits).

- **Resolving Out of Memory Errors in Evaluation**: `@pminervini` faced an issue where `evaluator.simple_evaluate` caused an Out of Memory (OOM) error and the allocated memory did not release on a V100 GPU. Suggestions included wrapping code in `try...except` and explicitly deleting model instances, but ultimately, restarting the runtime was the only solution.

- **Running Gemma 7b Model in Evaluation Harness**: Users experienced issues running the Gemma 7b model, with failures and reshaping errors reported by `.rand0mm` and poor scoring issues brought up by `@vraychev`. `@hailey_schoelkopf` confirmed that non-flash-attention implementations of this model had recently been patched and provided a diff snippet to fix score evaluation issues.

- **Clarifications on Task Types for the Evaluation Harness**: When `@dsajlkdasdsakl` asked about a custom task in the evaluation harness causing an error with API models, `@hailey_schoelkopf` clarified that tasks with `output_type` of `generate_until` do not require logits and can be used with API models that only return generated text.

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1u5MoN-QUfdNJXilFJAyJaGY1HlYWnfwX?usp=sharing): no description found
- [Google Colaboratory](https://colab.research.google.com/drive/1u5MoN-QUfdNJXilFJAyJaGY1HlYWnfwX?usp=): no description found
- [GitHub: Let’s build from here](https://github.com): GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...
- [src/backend/huggingface_generate_until.py · hallucinations-leaderboard/leaderboard at main](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/blob/main/src/backend/huggingface_generate_until.py): no description found
- [lm-evaluation-harness/lm_eval/evaluator.py at c26a6ac77bca2801a429fbd403e9606fd06e29c9 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c26a6ac77bca2801a429fbd403e9606fd06e29c9/lm_eval/evaluator.py#L190)): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [lm-evaluation-harness/lm_eval/tasks/mmlu/default/_default_template_yaml at 5ab295c85f90b2fd6218e88b59a3320544b50f8a · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/5ab295c85f90b2fd6218e88b59a3320544b50f8a/lm_eval/tasks/mmlu/default/_default_template_yaml#L4): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [how to add tasks with requests based on the answers for the previous requests? · Issue #1432 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1432#issuecomment-1949904868): I have a task that I want to add. The task is about answering binary questions. The only difference from other tasks is that the second question requires knowing the response for the first one. I u...
- [lm-evaluation-harness/lm_eval/tasks/mmlu/flan_cot_zeroshot/_mmlu_flan_cot_zeroshot_template_yaml at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/flan_cot_zeroshot/_mmlu_flan_cot_zeroshot_template_yaml): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [What&#39;s going on with the Open LLM Leaderboard?](https://huggingface.co/blog/evaluating-mmlu-leaderboard): no description found
- [[`Core tokenization`]  `add_dummy_prefix_space` option to help with latest issues by ArthurZucker · Pull Request #28010 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28010#discussion_r1495692285): What does this PR do? Allows users to use tokenizer.tokenize controlling the addition of prefix space. Let&#39;s also update fast! fixes #28622

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1209484866941485078) (5 messages): 

- **Confusion over Dropout in ParallelMLP**: `@jdranpariya` pointed out a confusing comment in `ParallelMLP` class that suggests a dropout is applied, but it's not visible in the provided code snippets. The comment in question alludes to a dropout at the end of an MLP process, although its implementation isn't immediately apparent.
- **Clarification on Dropout Implementation**: `@stellaathena` clarified that the dropout is actually applied in the construction of the transformer layer, providing a [link to the relevant code](https://github.com/EleutherAI/gpt-neox/blob/f7373f806689cb270677dd48bffddf4a32bfadce/megatron/model/transformer.py#L833). They indicated that the comment within `ParallelMLP` might be misleading.
- **Misleading Comment Acknowledged**: `@jdranpariya` agreed that the comment might need modification to prevent confusion about dropout application within `ParallelMLP`. However, they also expressed reluctance to commit changes solely for a comment correction.
- **Dropout Code Block Identified**: In a follow-up message, `@jdranpariya` identified the block of code where dropout is actually applied, which involves a dropout function followed by a residual addition, as per the transformer layer's construction.

**Links mentioned**:

[gpt-neox/megatron/model/transformer.py at f7373f806689cb270677dd48bffddf4a32bfadce · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/blob/f7373f806689cb270677dd48bffddf4a32bfadce/megatron/model/transformer.py#L833),): An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library. - EleutherAI/gpt-neox

  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1209615605653700689) (1 messages): 

- **Community Prompt Ranking Launched**: `@lunarflu` announces that <#1205128865735770142> is live, encouraging members to rank prompts and contribute to building a community dataset. With over **200 contributors** and **3500+ rankings**, participants can earn experience points and ascend the <#1197148293164187678> leaderboard.
- **Dataset Viewer and HuggingChat Upgrades**: Full-screen revamped dataset viewer pages resemble spreadsheets, while HuggingChat now supports conversation branching and prompt editing. Check the [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7163197916491276288/) post and [Twitter](https://twitter.com/lunarflu1/status/1759940289416798678) update for details.
- **Gradio Notebook and Inference Tools Excite**: The new `@Gradio` Notebook offers an interactive UX, text-generation-inference has been improved for twice the speed, and now, observablehq dashboards can be pushed to the `@HuggingFace` Hub. Links to the announcements can be found on [Twitter](https://twitter.com/LastMile/status/1757809383646343338) and [Twitter](https://twitter.com/narsilou/status/1758088388010258848?s=61&t=zKZt96B-1abARSzLt_5eLw).
- **Fresh Open Source AI Thrills**: The community is introduced to the new Open Source AI Cookbook and several updates, including a new TRL release, Gradio 3D demo enhancements, llama.cpp CLI QoL improvements, Transformers' `torch.compile` speedup, and Nanotron v0.2. To learn more, visit the respective [Twitter](https://twitter.com/younesbelkada/status/1758412890162487408) and [Twitter](https://twitter.com/mariaKhalusova/status/1758477222166548939) links.
- **Celebrating Milestones and Updates**: Over 10k HuggingChat assistants have been created, a new HuggingCast episode is announced, and the Korean LLM Leaderboard gets featured as a widely-used language-specific leaderboard. Find more about these updates on [Twitter](https://twitter.com/jeffboudier/status/1759679659112075616) and [Twitter](https://twitter.com/clefourrier/status/1759859199238939111).
  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1209412847537356841) (250 messages🔥🔥): 

- **HuggingFace VSCode Extension Troubleshooting**: `@industrial` is experiencing issues getting `huggingface-vscode` to work on NixOS, facing errors such as `Missing field request_params` and `Invalid type: null, expected a string`, despite updating to the latest `llm-ls` and trying different settings.

- **Gemma 7B vs Zephyr 7B**: `@mist7302` is considering finetuning Gemma7b with their own data, questioning whether benchmarking against Zephyr7b makes sense or if it's better to compare it to base Mistral7b.

- **Partnership Proposals with HuggingFace**: `@aaaliahmad.` inquires about how to submit a proposal for partnership or collaboration with HuggingFace, awaiting responses from staff members on the appropriate procedure.

- **Finetuning Challenges on A100 40GB**: `@gryhkn` is attempting to fine-tune Mistral-7b using PEFT, LoRA, and SFTTrainer with a large dataset, encountering out-of-memory errors on an Nvidia A100 GPU, and seeking advice on how to proceed.

- **New Transformers Library Update Tease**: `@not_lain` hints at an exciting upcoming release of the transformers library which includes new models and improvements that will allow users to create custom architectures without encountering issues.

**Links mentioned**:

- [Groq](https://groq.com/): no description found
- [Google Colaboratory](https://colab.research.google.com/drive/11OMSb4XBuOAWaKNEl9Ay7MPnF4rGEf9H#scrollTo=eT6IMdhG2n2u): no description found
- [llm-vscode&#32;-&#32;Visual&#32;Studio&#32;Marketplace](https://marketplace.visualstudio.com/items?itemName=HuggingFace.huggingface-vscode): Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;LLM&#32;powered&#32;development&#32;for&#32;VS&#32;Code
- [USB Accelerator | Coral](https://coral.ai/products/accelerator/): A USB accessory that brings accelerated ML inferencing to existing systems. Works with Windows, Mac, and Raspberry Pi or other Linux systems.
- [thomas-c-reid/ppo-LunarLander-v2 · Hugging Face](https://huggingface.co/thomas-c-reid/ppo-LunarLander-v2): no description found
- [Deep Reinforcement Learning Leaderboard - a Hugging Face Space by huggingface-projects](https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard): no description found
- [simple static kv cache script](https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb): simple static kv cache script. GitHub Gist: instantly share code, notes, and snippets.
- [AWS Innovate - AI/ML and Data Edition](https://aws.amazon.com/events/aws-innovate/apj/aiml-data/): no description found
- [ptx0/photo-concept-bucket · Datasets at Hugging Face](https://huggingface.co/datasets/ptx0/photo-concept-bucket): no description found
- [Mistral&#039;s next LLM could rival GPT-4, and you can try it now in chatbot arena](https://the-decoder.com/mistrals-next-llm-could-rival-gpt-4-and-you-can-try-it-now-in-chatbot-arena/): French LLM wonder Mistral is getting ready to launch its next language model. You can already test it in chat.

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1209497392269885490) (6 messages): 

- **Character Prediction with MLP**: User `@parvpareek` is studying **next character prediction using an MLP** from the [Karpathy lectures](https://cs231n.github.io/).
- **On to Bengio's Work**: Following up, `@parvpareek` expressed intentions to implement **A Neural Probabilistic Language Model** from Bengio's paper.
- **Intriguing Paper on AI Alignment**: `@epicx` shared an intriguing paper titled ["Agents Need Not Know Their Purpose"](https://arxiv.org/abs/2402.09734), discussing the concept of oblivious agents, which signifies a step towards solving the AI alignment challenge.
- **Newcomer Seeking Image Generation Models**: New user `@mfd000m` inquired about **Hugging Face models** capable of generating *hero images* for e-commerce products like notebooks and phones.

**Links mentioned**:

[Agents Need Not Know Their Purpose](https://arxiv.org/abs/2402.09734): Ensuring artificial intelligence behaves in such a way that is aligned with human values is commonly referred to as the alignment challenge. Prior work has shown that rational agents, behaving in such...

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1209536035235692585) (8 messages🔥): 

- **ChatGPT Traffic Dips Amidst AI Competition**: `@developerworld_` highlighted that **ChatGPT's web traffic** has diminished over the past eight months, revealing an 11% drop from its May 2023 peak, and that its **mobile app usage** lags behind platforms like Snapchat in user growth.
- **SORA: The Anticipated Game Changer?**: In the conversation about ChatGPT's waning traffic, `@m22046` mentioned **SORA** as a potential influencer in the market, implying it is a project worth watching. However, `@developerworld_` pointed out that SORA is not yet publicly available.
- **Showcase AI Startups at Data Council's AI Launchpad**: `@petesoder` announced an **AI Launchpad event** at the upcoming **Data Council conference** in Austin, TX. The event is aimed at engineer-founders of pre-incorporation or pre-seed stage AI startups and provides an opportunity to present their products on stage. ([AI Launchpad Submission](https://zeroprime.vc/ai-launchpad))
- **Dynamic Recursive Neural Network Paper**: User `@kirayamato8507` shared a research paper titled "Dynamic Recursive Neural Network" hosted on the CVPR 2019 open access repository. ([Read the paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Dynamic_Recursive_Neural_Network_CVPR_2019_paper.pdf))
- **Curiosity about AI Competitors' Metrics**: `@lavi_39761` inquired about how **ChatGPT's competitors**, such as **Anthropic** and **character ai**, fare in terms of user metrics, noting You.com's increase in usage.

**Links mentioned**:

[Zero Prime @ Data Council '24](https://zeroprime.vc/ai-launchpad): Join Zero Prime Ventures @ Data Council Austin 2024 for a unique chance to showcase your AI startup. Apply now for exposure to top investors and elite founders.

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1209462840130281483) (33 messages🔥): 

- **Ad Revenue from Bots**: `@myg5702` confirmed generating **$3-$6 per hour** through ads, effectively making profit over the cloud server costs which are **$1.48 per hour**. They mentioned using *replicate* for cloud services and advised against using problematic websites like one which `@code_god` couldn't access.

- **Cybersecurity buffs, New Model Alert**: `@eatmyboxxx` announced the port of WhiteRabbitNeo-13B-v1 model to ollama and shared various links including [GGUF](https://huggingface.co/captainkyd/WhiteRabbitNeo-13B-v1.gguf), indicating the cybersecurity model should be available soon for use.

- **AYA Project Analysis**: `@lavi_39761` shared insights on the [AYA dataset and collection](https://open.substack.com/pub/lavisblog/p/aya-dataset-and-proposed-extensions?r=naul&utm_campaign=post&utm_medium=web) for low and mid-resource languages, commenting on the progress yet highlighting the need for further language coverage improvements.

- **Financial Wizards, Level Up**: `@luuisotorres` shared an **Investment Portfolio Management Web App** and a [Kaggle Notebook](https://www.kaggle.com/code/lusfernandotorres/building-an-investment-portfolio-management-app) detailing its creation process. This web app assists with tracking various investments and analyzing returns through interactive visuals.

- **Depth Estimation Goes Mobile**: `@shubhamx0204` presented his work on an Android app that runs the *Depth-Anything* model for monocular depth estimation. The app, powered by ONNX models for efficient inference, is available on [GitHub](https://github.com/shubham0204/Depth-Anything-Android).

**Links mentioned**:

- [captainkyd/whiterabbitneo7b](https://ollama.com/captainkyd/whiterabbitneo7b): https://huggingface.co/WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a
- [captainkyd/whiteRabbitNeo-7B.gguf · Hugging Face](https://huggingface.co/captainkyd/whiteRabbitNeo-7B.gguf): no description found
- [AYA Dataset Review and Proposed Extensions](https://open.substack.com/pub/lavisblog/p/aya-dataset-and-proposed-extensions?r=naul&utm_campaign=post&utm_medium=web): An exploratory analysis of AYA Dataset and Collection, with proposed extension recipes to improve task diversity for mid and low-popularity languages.
- [Portfolio Management - a Hugging Face Space by luisotorres](https://huggingface.co/spaces/luisotorres/portfolio-management): no description found
- [Building an Investment Portfolio Management App &#x1F4B0;](https://www.kaggle.com/code/lusfernandotorres/building-an-investment-portfolio-management-app): Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources
- [captainkyd/WhiteRabbitNeo-13B-v1.gguf · Hugging Face](https://huggingface.co/captainkyd/WhiteRabbitNeo-13B-v1.gguf): no description found
- [GitHub - shubham0204/Depth-Anything-Android: An Android app running inference on Depth-Anything](https://github.com/shubham0204/Depth-Anything-Android): An Android app running inference on Depth-Anything  - GitHub - shubham0204/Depth-Anything-Android: An Android app running inference on Depth-Anything

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1209684258302730310) (4 messages): 

- **Mystery Avoidance Strategy Questioned**: User `@chad_in_the_house` questions the methods of user `@811235357663297546` but remains unclear if it's a tactic to avoid detection.
- **Call for Enhanced Multimodal Models**: `@lavi_39761` responds to an uncertain scenario by highlighting the need for better multimodal models, implying that such improvements could clarify the situation.
- **A Simple Solution Proposed**: In the context of the previous discussion on avoiding detection, `@clock.work_` suggests a preventative measure: only allow picture sending from users who have sent a text message within the last 24 hours.
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1209519414043484160) (35 messages🔥): 

- **Seeking Damage Detection in Art**: `@div3440` is working on a project using stable diffusion model activations to detect damage in paintings. They've tried using cross-attention maps with terms like "damaged" and PCA on self-attentions but face challenges due to language specificity and the inconsistency in damage prominence.

- **Alternative Damage Detection Suggestions**: `@chad_in_the_house` suggests considering GANs or synthetic data generation for training classifiers, and expresses that zero-shot approaches to identify damage without further data would be groundbreaking.

- **Challenges in Dataset Curation**: Collecting data involved `@div3440` manually selecting damaged artworks from Wikipedia and museum collections, highlighting the difficulty since damage isn't typically noted in artwork metadata.

- **AI Generative Models for LSTM Images**: `@lucas_selva` inquires about services for generating images of LSTM neural networks, or free templates without copyright, though no clear solution is discussed within these messages.

- **Diffusion Model Internals Enlighten**: In a conversation with `@pseudoterminalx`, `@mr.osophy` seeks to understand whether the U-Net architecture in diffusion models takes an integer parameter `t` that affects processing, learning that it might not be simple but related to Fourier transforms for timestep embeddings.
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1209544808981467146) (4 messages): 

- **Seeking Sign Language Translation Model**: `@dj_mali_boy` inquired about a model to convert sign language to text. `@johko990` provided a potential solution with a link to a [HuggingFace model](https://huggingface.co/degirum/mobilenet_v2_sign_language_224x224_float_n2x_cpu_1) that might help.

- **Image Captioning with BLIP Notebooks**: `@johko990` shared links to Hugging Face notebooks for image captioning, specifically mentioning a [notebook using BLIP](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb) and another for [fine-tuning BLIP2](https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb), found in Niels' [Transformers-Tutorials repo](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/BLIP-2).

- **Resource Appreciation by Community**: `@seanb2792` expressed gratitude upon discovering the last mentioned resource, highlighting its value.

**Links mentioned**:

- [Models - Hugging Face](https://huggingface.co/models?search=sign%20language): no description found
- [notebooks/examples/image_captioning_blip.ipynb at main · huggingface/notebooks](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb): Notebooks using the Hugging Face libraries 🤗. Contribute to huggingface/notebooks development by creating an account on GitHub.
- [notebooks/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb at main · huggingface/notebooks](https://github.com/huggingface/notebooks/blob/main/peft/Fine_tune_BLIP2_on_an_image_captioning_dataset_PEFT.ipynb): Notebooks using the Hugging Face libraries 🤗. Contribute to huggingface/notebooks development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1209485749221593119) (33 messages🔥): 

- **Quest for Updated Deep Learning Resources**: User `@nrs9044` searched for materials newer than the 2022 book on *Deep Learning for NLP*, specifically something that does not feature deprecated packages.
  
- **Hugging Face and Google Summer of Code Inquiry**: `@debrup_` inquired whether Hugging Face will participate in GSoC 2024, also seeking advice on implementing multi-label classification with T5, and how to perform masked language modeling with a SentencePiece tokenizer that lacks a mask token.

- **TensorFlow Troubles Unraveling**: `@diegot8170` encountered an error when loading a model with TensorFlow, which led to a discussion about potential framework issues. With `@cursorop`'s assistance, the problem was resolved by reinstalling TensorFlow, emphasizing the complexities of the framework in comparison to PyTorch.

- **Seeking Data Quality Benchmarks for LLMs**: `@.konoh` sought industry-standard benchmarks for data quality that power large language models (LLMs) and `@nrs9044` explored alternatives to LLMs in the NLP field.

- **Custom Embeddings for Biomedical Terms**: User `@joshpopelka20` faced challenges with pre-trained embeddings for sentence similarity in biomedical terms and was advised by `@lavi_39761` to consider contrastive learning and sentence transformers for improved results. After fine-tuning without including classification data, the issue was resolved.
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1209519414043484160) (35 messages🔥): 

- **Seeking Damage Detection in Art**: `@div3440` inquired about using stable diffusion model activations to detect areas of damage in paintings, proposing a workflow that involves null-prompt inversion and examining activation matrices. They expressed challenges with weak conditioning from terms like "damaged" and discussed PCA over self-attention maps as an alternative, while `@chad_in_the_house` suggested methods like synthetic data generation with GANs and adapting existing models for this unique task.
  
- **Knowledge Gap in Diffusion for Animation**: `@mr.osophy` posted questions about high-resolution image synthesis with latent diffusion to clarify their understanding of the architecture and training process. `@pseudoterminalx` provided insights into the U-Net's utilization of timestep embedding and Fourier transforms, highlighting complexities in the microconditioning inputs.
  
- **In Search of an AI-Generated LSTM Image**: `@lucas_selva` sought a generative AI service or free templates to produce images of an LSTM neural network, though the query remained unanswered.
  
- **Extracting Latent Tensors from Prompts**: `@shinyzenith` asked whether it's possible to extract latent representation tensors for a given prompt using diffusers, but the question went without a direct response.

- **HuggingMod's Gentle Reminder**: Auto-moderation was at play, as `@HuggingMod` reminded `@754474676973076532` to reduce their message frequency to avoid spamming.

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1209548452342530048) (1 messages): 

- **LlamaCloud & LlamaParse Announced**: `@jerryjliu0` introduced **LlamaCloud**, a new service for enhancing LLM and RAG applications with production-grade data, featuring components like **LlamaParse**, a parser adept at complex documents. **LlamaParse** is now publicly available with a free usage cap, and a managed API is opening for private beta testing with select enterprises.
- **Launch Users and Collaborators Onboard**: The launch of LlamaCloud is supported by users and partners such as Mendable AI, DataStax, MongoDB, Qdrant, NVIDIA, and projects from the LlamaIndex Hackathon.
- **LlamaCloud's Key Offerings**: Detailed blog post on LlamaCloud's offerings, focusing on **LlamaParse** for parsing complex documents and Managed Ingestion/Retrieval API for data handling. A new Discord channel for LlamaParse will be created soon.
- **Announcement Spread via Twitter**: The launch is publicized by the official `@llama_index` Twitter account, emphasizing the ease of data handling and more time allocation on application logic with **LlamaParse** and supporting APIs.
- **Access and Contact Information**: Interested users can try **LlamaParse** through their client repo, sign up for LlamaCloud, contact the team for details about the broader offering, and visit the new LlamaIndex website. Links provided in the announcement direct users to the respective resources.

**Links mentioned**:

- [Introducing LlamaCloud and LlamaParse](https://blog.llamaindex.ai/introducing-llamacloud-and-llamaparse-af8cedf9006b): Today is a big day for the LlamaIndex ecosystem: we are announcing LlamaCloud, a new generation of managed parsing, ingestion, and…
- [Tweet from LlamaIndex 🦙 (@llama_index)](https://x.com/llama_index/status/1759987390435996120?s=20): Introducing LlamaCloud 🦙🌤️  Today we’re thrilled to introduce LlamaCloud, a managed service designed to bring production-grade data for your LLM and RAG app.  Spend less time data wrangling and more...

  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1209560845567135835) (4 messages): 

- **Cloud Service for LLM and RAG**: Llama Index announced **LlamaCloud**, a managed service aimed to provide production-grade data for LLM and RAG applications, promoting a focus on application logic over data wrangling. The launch tweet also included a [teaser link](https://t.co/XI6Vt7U0H4).

- **Cookbook for Advanced RAG Setup**: `@llama_index` introduces a set of cookbooks detailing how to parse PDFs with **LlamaParse**, index them into **AstraDB** by DataStax, and apply recursive retrieval for answering complex queries. Find more details in their [cookbook guide](https://t.co/TvH5Hp62dx).

- **Simplified RAG Development Approach**: A new presentation by `@jerryjliu0` discusses simplifying RAG development by identifying and addressing pain points in the pipeline. Slides are now publicly shared, with the [announcement tweet](https://t.co/vWnjQSkhUF) providing access to the content.

- **Frontend Tutorial for LLM/RAG Experts**: Marco Bertelli has created a tutorial for LLM/RAG experts to learn React and build an aesthetic frontend to complement `@llama_index`'s RAG backend server. The resource is fully supported and accessible through the [shared link](https://t.co/TIOwr90L6m).
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1209433082705485834) (379 messages🔥🔥): 

- **GitHub Repository Reader Rate Limit Issue**: User `@david1542` encountered a rate limit issue with `GithubRepositoryReader`. `@whitefang_jr` suggested using a commit ID instead of default branches can prevent the fetching of all commits and help avoid rate limits.
- **Understanding LlamaIndex's Import Process**: `@alvarojauna` had trouble with importing from LlamaIndex due to changes in the v0.10.x upgrade. `@cheesyfishes` pointed towards the correct import path and also noted that the version updates should be applied manually due to the significant changes in the codebase.
- **LlamaIndex and Query Engines**: Discussions centered around methods of querying and how LlamaIndex interacts with LLM. `@sansmoraxz` mentioned that `response_mode` in query engines may impact the number of LLM calls made during querying.
- **Dealing with Python Typing Issues**: `@humblehound` expressed concerns about the lack of explicit type definitions in python/AI libraries, leading to a conversation about best practices for maintaining codebase integrity, with suggestions like Pydantic, Ruff, using type annotations, and potentially contributing to libraries' typings.
- **LlamaIndex Finetuning for LLM models**: `@david1542` inquired about finetuning LLM models leveraging LlamaIndex. `@whitefang_jr` highlighted other documentation sections that provide non-paid mechanisms for finetuning.

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1uJ2qXJ-laFIEweDWNKXqa2gLt765PPRD?usp=sharing): no description found
- [T-RAG = RAG + Fine-Tuning + Entity Detection](https://cobusgreyling.medium.com/t-rag-rag-fine-tuning-entity-detection-9a5aaa01e437): The T-RAG approach is premised on combining RAG architecture with an open-source fine-tuned LLM and an entities tree vector database. The…
- [Behavior of Python&#x27;s time.sleep(0) under linux - Does it cause a context switch?](https://stackoverflow.com/questions/7273474/behavior-of-pythons-time-sleep0-under-linux-does-it-cause-a-context-switch),): This pattern comes up a lot but I can&#x27;t find a straight answer.&#xA;&#xA;An non-critical, un-friendly program might do&#xA;&#xA;while(True):&#xA;    # do some work&#xD;&#xA;Using other technologi...
- [Time.sleep(0) yield behaviour](https://discuss.python.org/t/time-sleep-0-yield-behaviour/27185): Hi,  can time.sleep(0) be used to simulate yield like behavior in threads? is there any other way to yield threads in a multithreaded program?
- [Fine-tuning - LlamaIndex 🦙 v0.10.11.post1](https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning.html#fine-tuning-llama-2-for-better-text-to-sql): no description found
- [llama_index/llama-index-core/llama_index/core/question_gen/llm_generators.py at da5f941662b65d2e3fe2100f2b58c3ba98d49e90 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/da5f941662b65d2e3fe2100f2b58c3ba98d49e90/llama-index-core/llama_index/core/question_gen/llm_generators.py#L10C5-L10C37): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/question_gen/llm_generators.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/question_gen/llm_generators.py): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [Behavior of Python&#x27;s time.sleep(0) under linux - Does it cause a context switch?](https://stackoverflow.com/questions/7273474/behavior-of-pythons-time-sl): This pattern comes up a lot but I can&#x27;t find a straight answer.&#xA;&#xA;An non-critical, un-friendly program might do&#xA;&#xA;while(True):&#xA;    # do some work&#xD;&#xA;Using other technologi...
- [llama_index/llama-index-core/llama_index/core/callbacks/token_counting.py at 6fb1fa814fc274fe7b4747c047e64c9164d2042e · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/6fb1fa814fc274fe7b4747c047e64c9164d2042e/llama-index-core/llama_index/core/callbacks/token_counting.py#L53): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [Tweet from Benjamin Clavié (@bclavie)](https://x.com/bclavie/status/1745151278018076985?s=46&t=vFhwbwKyUFaP8aBKHBB-BQ): &#34;I wish I could use the best retrieval model as a reranker in my RAG pipeline, but I don&#39;t have time to redesign it!&#34;   We&#39;ve all been there and thought that, right?  🪤RAGatouille now...
- [llama_index/llama-index-integrations/llms/llama-index-llms-vllm/llama_index/llms/vllm/base.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-vllm/llama_index/llms/vllm/base.py): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/chat_engine/types.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/chat_engine/types.py#L177).): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [no title found](https://cloud.google.com/docs/authentication/external/set-up-adc): no description found
- [Custom Embeddings - LlamaIndex 🦙 v0.10.11.post1](https://docs.llamaindex.ai/en/stable/examples/embeddings/custom_embeddings.html#custom-embeddings-implementation): no description found
- [llama_index/llama-index-integrations/embeddings/llama-index-embeddings-ollama/llama_index/embeddings/ollama/base.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-ollama/llama_index/embeddings/ollama/base.py): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [fixes, so many fixes [circular import bonanza] by logan-markewich · Pull Request #11032 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/11032): no description found
- [OpenAI API compatibility · Issue #305 · ollama/ollama](https://github.com/ollama/ollama/issues/305): Any chance you would consider mirroring OpenAI&#39;s API specs and output? e.g., /completions and /chat/completions. That way, it could be a drop-in replacement for the Python openai package by changi...
- [[Bug]: Chat stream active wait blocks flask request thread · Issue #10290 · run-llama/llama_index](https://github.com/run-llama/llama_index/issues/10290): Bug Description The method StreamingAgentChatResponse.response_gen, which is designed to generate responses, currently exhibits a high CPU usage issue due to an active wait implemented within a loo...
- [Add sleep to fix lag in chat stream by w4ffl35 · Pull Request #10339 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/10339): Description This is a fix for a bug which I identified while attempting to use Chat Stream from within a thread. Adding a sleep within the loop unblocks the thread allowing the chat to stream as ex...

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1209597169934409748) (7 messages): 

- **Blog Post Spotlight by AndySingal**: `@andysingal` shared a link to a blog post titled "[Leveraging Llamaindex & Step-Wise React Agent for Efficient Document Handling](https://ai.gopubby.com/leveraging-llamaindex-step-wise-react-agent-for-efficient-document-handling-3a0f92b9ca22)," which explores integrating Llamaindex, React Agent, and Llamacpp for document management.
- **Insightful Read on Arithmetic in GPT-4**: `@johnloeber` posted about their new [blogpost](https://loeber.substack.com/p/16-notes-on-arithmetic-in-gpt-4) discussing GPT-4's capabilities and limitations in arithmetic and symbolic reasoning, sharing a personal experience of GPT-4 getting a sum wrong.
- **Exploring Translation into Camfranglais**: User `@behanzin777` requested strategies for training a GPT to translate French into Camfranglais, providing a link to a relevant dictionary and expressing difficulties with previous approaches.
- **Gemini 1.5 A Potential Solution for Slang Translation**: In response to `@behanzin777`, `@geoloeg` mentioned Gemini 1.5 as a potential solution for learning Camfranglais given its capability to learn languages from grammar or slang vocabulary. `@behanzin777` appreciated the suggestion and planned to try it.
- **Seeking Summarization Metrics for Llamaindex**: `@dadabit.` inquired if anyone is using metrics to evaluate summarization within llamaindex, and requested suggestions on effective metrics and tools.

**Links mentioned**:

- [#16: Notes on Arithmetic in GPT-4](https://loeber.substack.com/p/16-notes-on-arithmetic-in-gpt-4): A few weeks ago, I had a list of dollar amounts that I needed to sum up. I thought: “GPT is good at converting formats,” and copy-pasted them into ChatGPT. The result looked plausible. But I had a mom...
- [Leveraging Llamaindex &amp; Step-Wise React Agent for Efficient Document Handling](https://ai.gopubby.com/leveraging-llamaindex-step-wise-react-agent-for-efficient-document-handling-3a0f92b9ca22): Ankush k Singal

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1209426916935860236) (197 messages🔥🔥): 

- **Groq's Sequential Speed Marvel**: `@thezennou` and `@i_am_dom` discuss **Groq chip's** astonishing speed despite its sequential nature and limited memory. They suggest that while it may be costly, with the right batching, it could achieve thousands of tokens per second, though `@i_am_dom` notes limitations when scaling to larger models.

- **Mistral-Next's Brevity Preference**: Users `@paul.martrenchar_pro` and `@akshay_1` find **Mistral-Next** providing shorter answers compared to the **Mistral-Medium** and discussing possible reasons, ranging from the platform's settings to the model's inherent design.

- **Censorship Fears in Newer LLMs**: `@nayko93` and `@ethux` touch on the increasing censorship in newer language models, including **Mistral-Next**, with `@ethux` pointing out the need for compliance with European laws, which mandate moderation for illegal content.

- **Mistral's Potential Model Releases and Open Source Status**: `@gunterson`, `@dillfrescott`, and `@mrdragonfox` express hopes for continued open-source releases from **Mistral**, discussing the benefits for cost and accessibility. Some users showed concern about losing open-source access, while `@mrdragonfox` reassured about imminent announcements.

- **Function Calling in LLMs and Chat Interface Queries**: Users inquire about function calling capabilities and constructing chat interfaces using **Mistral AI**. `@mrdragonfox` clarifies how functionality works, it formats output in JSON for client-side execution, while others like `@ethux` and `@drnicefellow` provide resources for implementing chat UIs with Mistral AI. 



**Links mentioned**:

- [Tweet from Aaditya Ura (Ankit) (@aadityaura)](https://x.com/aadityaura/status/1760305308927426903?s=20): The new Model Gemma from @GoogleDeepMind @GoogleAI  does not demonstrate strong performance on medical/healthcare domain benchmarks.  A side-by-side comparison of Gemma by @GoogleDeepMind  and Mistral...
- [Pretraining on the Test Set Is All You Need](https://arxiv.org/abs/2309.08632): Inspired by recent work demonstrating the promise of smaller Transformer-based language models pretrained on carefully curated data, we supercharge such approaches by investing heavily in curating a n...
- [Basic RAG | Mistral AI Large Language Models](https://docs.mistral.ai/guides/basic-RAG/): Retrieval-augmented generation (RAG) is an AI framework that synergizes the capabilities of LLMs and information retrieval systems. It&#x27;s useful to answer questions or generate content leveraging ...
- [gist:c9b5b603f38334c25659efe157ffc51c](https://gist.github.com/sublimator/c9b5b603f38334c25659efe157ffc51c): GitHub Gist: instantly share code, notes, and snippets.
- [Prompting Capabilities | Mistral AI Large Language Models](https://docs.mistral.ai/guides/prompting-capabilities/): When you first start using Mistral models, your first interaction will revolve around prompts. The art of crafting effective prompts is essential for generating desirable responses from Mistral models...
- [The Mysterious New LLM from MISTRAL-AI](https://www.youtube.com/watch?v=J6gQW_oPi_U): Mistral-Next is the new LLM from Mistral-AI. There is no information about it but it&#39;s a solid LLM. 🦾 Discord: https://discord.com/invite/t4eYQRUcXB☕ Buy me...
- [Mistral-NEXT Model Fully Tested - NEW KING Of Logic!](https://youtu.be/o4yoPEjIYRk?si=l_tcTZ6RdyVZvMYu&t=112): Mistral quietly released their newest model &quot;mistral-next.&quot; Does it outperform GPT4?Need AI Consulting? ✅ - https://forwardfuture.ai/Follow me on Twitter 🧠 ...
- [GitHub - MeNicefellow/DrNiceFellow-s_Chat_WebUI](https://github.com/MeNicefellow/DrNiceFellow-s_Chat_WebUI): Contribute to MeNicefellow/DrNiceFellow-s_Chat_WebUI development by creating an account on GitHub.
- [Mistral](https://huggingface.co/docs/transformers/main/en/model_doc/mistral): no description found

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1209448796522545172) (72 messages🔥🔥): 

- **Mixtral Model Confusion Resolved**: User `@ethux` clarified to `@brendawin` that the `TypeError` being faced was due to an unexpected keyword argument 'device,' and the `device` can't be defined in that parameter of the code. They also advised to check the documentation for the correct implementation.
- **Insights on Mistral's Potential and Limitations**: User `@ygrek` asked various questions about the 8X7B model, including hardware requirements, compatibility with simili copilot tools, and implementation details. No definitive answers were provided in the ensuing discussion.
- **Model Performance and Access Specifications**: `@impulse749` inquired about the performance of mixtral-next compared to GPT-4-Turbo and its open-source status. `@mrdragonfox` mentioned that Mistral models are openweights but not open source, with an official announcement pending for Mistral-next.
- **Quantization Discussions**: `@mrdragonfox` suggested that a quantitized version of a model can be faster than the fp16 version, especially when it comes to batching on accelerators like those from Groq that run native int8 inference.
- **FusionL AI for Learners by Learners**: `@kunal0089` introduced their generative AI startup FusionL AI, a platform for learning in a smart and minimalistic way, and provided a link for others to try it [here](https://youai.ai/ais/fusionl-d57a93eb).

**Links mentioned**:

[FUSIONL AI](https://fusionlai.carrd.co/): FUSIONL AI is a pioneer of SMLM Model (Smart Minimalistic Language Model) for learning in smart and minimalistic way.

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1209498857189408808) (38 messages🔥): 

- **TensorRT and Mistral Inquiry**: User `@goatnes` questioned the feasibility of running **Mistral 8x7b** with chat on RTX or with TensorRT. `@mrdragonfox` replied that it requires onyx runtime and the porting of it but deemed it possible nonetheless.

- **Discovering AI deployment basics**: `@goatnes` acknowledged they are just starting to learn about AI deployment basics, indicating their inquiry was part of that learning process.

- **AWS Private Server Costs for Mistral Hosting**: `@ambre3024` sought community assistance on estimating the costs involved in hosting **Mistral** on a private AWS server. `@ethux` responded by inquiring about which model size is being considered for hosting, implying the cost varies by model size.

- **Mistral next API Availability**: User `@rantash68` inquired about the availability of **Mistral next** via API, and `@sophiamyang` confirmed that it is not available and did not provide any indication of future availability.

- **Seeking Efficient Production Deployment for Mistral**: `@louis2567` expressed intent to move Mistral 7b and Mixtral 8x7b to production for fast inference. `@mrdragonfox` offered detailed instructions and considerations for deploying Mistral using **vLLM** through a Docker image or Python package, engaging in an extended conversation about the benefits and implications of batching on GPU performance.

**Links mentioned**:

[vLLM | Mistral AI Large Language Models](https://docs.mistral.ai/self-deployment/vllm/): vLLM can be deployed using a docker image we provide, or directly from the python package.

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1209548718890557470) (29 messages🔥): 

- **Finetuning Iterative Process Advocated**: Both `@rusenask` emphasize that finetuning is not a one-shot process, suggesting a cycle of finetune, test, adjust, and reiterate, with the possibility of abandoning an approach for a better one.
- **Adding New Languages to Models Poses Challenges**: `@kero7102` inquires about adding Hebrew to Mistral/Mixtral, noting poor support in the tokenizer, while `@tom_lrd` suggests that success with less common languages like Hebrew is uncertain due to limited pretraining on these tokens.
- **Out-of-Memory Errors Troubleshot**: `@iamcoming5084` encounters an out-of-memory error when merging an 8x7b model, with `@mrdragonfox` explaining that 86GB of VRAM is needed, exceeding the capacity of a single H100 80GB GPU.
- **Translation as a Solution for Language Support**: `@tom_lrd` recommends considering a translation model to deal with English and searching discussions on other languages like Portuguese and Estonian for further insights.
- **Parameters Affecting Fine-Tuning Accuracy**: `@_._pandora_._` suggests considering parameters like epoch/steps, batch size, or LoRA's hyperparameter r when fine-tuning Mixtral 8x7b and Mistral 7B for improved accuracy.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1209490282287210557) (25 messages🔥): 

- **Educational AI Start-up Showcased**: `@kunal0089` introduced FUSIONL AI, a generative AI startup focused on learning in a smart, minimalistic way. They shared [FUSIONL AI's website](https://fusionlai.carrd.co) and highlighted its use of the SMLM Model to change the learning methodology.
- **Discussion on AI Learning Tools**: `@gunterson` questioned the advantage of FUSIONL AI over wrappers like GPT-3.5/4, prompting `@kunal0089` to explain that their AI offers streamlined, point-wise information distribution without users having to specify the format.
- **Clarification on Showcase Channel Etiquette**: `@ethux` addressed the discord community, stating that the showcase channel is meant for displaying projects built using or related to Mistral, deterring content that seems more like self-promotion not directly associated with Mistral.
- **Live Mistral-Next Testing**: `@jay9265` mentioned they were testing Mistral-Next for data engineering use cases on their Twitch channel, providing a [link to the stream](https://www.twitch.tv/jay9265/), while also asking for permission in case the post was inappropriate.
- **Prompt Crafting Guide Shared**: `@mrdragonfox` linked to a Mistral AI guide that teaches users how to craft effective prompts for various tasks such as classification, summarization, personalization, and evaluation, available on [Github with a Colab notebook](https://docs.mistral.ai/guides/prompting-capabilities/). They also mentioned using a confidence score for these tasks and that prompt annealing tools like DSPy would be beneficial.

**Links mentioned**:

- [Jay9265 - Twitch](https://www.twitch.tv/jay9265/): Welcome! At the intersection of Data Engineering and LLMs. Join us as we play with language models and solve data engineering problems
- [FUSIONL AI](https://fusionlai.carrd.co/): FUSIONL AI is a pioneer of SMLM Model (Smart Minimalistic Language Model) for learning in smart and minimalistic way.
- [Corrective RAG using LangGraph](https://www.youtube.com/watch?v=eT31Cyaziy0): Corrective RAGSelf-reflection can enhance RAG, enabling correction of poor quality retrieval or generations.Several recent papers focus on this theme, but im...
- [BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://www.youtube.com/watch?v=T_dYzuv4N70): Large Language Models (LLMs) are typically trained in two phases: pre-training on large internet-scale datasets, and fine-tuning for downstream tasks. Given ...
- [Prompting Capabilities | Mistral AI Large Language Models](https://docs.mistral.ai/guides/prompting-capabilities/): When you first start using Mistral models, your first interaction will revolve around prompts. The art of crafting effective prompts is essential for generating desirable responses from Mistral models...
- [Self RAG using LangGraph](https://www.youtube.com/watch?v=Eb7QF1nDWGU): Self-reflection can enhance RAG, enabling correction of poor quality retrieval or generations.Several recent papers focus on this theme, but implementing the...

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1209455405344886847) (2 messages): 

- **Flutter Developers Rejoice**: User `@lkincel` from Nomtek announced a new library that aids in integrating **Mistral AI** into Flutter apps, targeting mobile developers working on PoCs. The library can be found on [GitHub](https://github.com/nomtek/mistralai_client_dart) and is explained further in a [Medium article](https://medium.com/nomtek/easy-flutter-integration-with-ai-using-mistral-ai-23683e45b176), which outlines use cases in chatbots, embeddings, and LLM as a controller.
- **AI for Education Takes Off**: `@kunal0089` mentioned developing an AI with a focus on learning and educational purposes, but did not provide further details or links regarding the project.

**Links mentioned**:

- [GitHub - nomtek/mistralai_client_dart: This is an unofficial Dart/Flutter client for the Mistral AI API.](https://github.com/nomtek/mistralai_client_dart): This is an unofficial Dart/Flutter client for the Mistral AI API. - nomtek/mistralai_client_dart
- [Easy Flutter Integration With AI Using Mistral AI](https://medium.com/nomtek/easy-flutter-integration-with-ai-using-mistral-ai-23683e45b176): Co-authored with: Łukasz Gawron

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1209450704842793050) (10 messages🔥): 

- **No LogProbs in the API**: `@youssefabdelmohsen` inquired about obtaining logprobs through the API, to which `@mrdragonfox` replied that it is **not possible at the moment** and referenced channel <#1204542458101502033> for more information.
- **Access to Mistral-Next on Hold**: `@superseethat` asked about accessing `Mistral-Next` and `@ethux` clarified that **access is not currently available** as Mistral Next is not released.
- **Testing Mistral-Next via Lymsys Chat**: Following up on the access question, `@ethux` added that one can only **test Mistral Next using the chat from lymsys** for the time being.
- **Clarifying API Billing Thresholds**: `@sapphics` sought clarification on what exceeding API billing thresholds means, and `@mrdragonfox` confirmed the relevance of the **Mistral billing limits** and suggested contacting support directly at **support@mistral.ai**.
- **Support Response Concerns**: In a follow-up, `@ginterhauser` expressed frustration that attempts to reach out to increase API limits were **ignored twice**, bringing attention to potential issues with customer support responsiveness. `@mrdragonfox` responded by asking if they included their **id** when reaching out.

**Links mentioned**:

[no title found](https://console.mistral.ai/billing/limits/): no description found

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1209425070913749042) (104 messages🔥🔥): 

- **AI Ups and Downs**: Users in the chat have mixed feelings about the AI tools they've been using. `@hanah_34414` praises **GPT-4** for its reliability in math problems, while `@jeremy.o` found success generating content related to Edwardian literature with **Groq**. On the other hand, `@blckreaper` complained about **Gemini** producing Shakespearian-like responses when attempting to do homework.
  
- **Awaiting Sora's Arrival**: Several users such as `@copperme_14779` and `@lofiai.art` asked about when **Sora AI** might be available for public use or through API access, but `@solbus` and others highlight that there is no confirmed release date for **Sora** yet.

- **Chatbot Message Limits Stir Debate**: `@lugui` reported a limit of **40 messages** per 3 hours for **GPT-4** use, sparking questions from `@zaatuloa` and a detailed discussion with `@7_vit_7` and `@solbus`, who clarified the reasons and policies behind message limits, including references to usage cap and subscription plans in various official OpenAI resources.

- **Google's Stealthy AI Moves**: Google's new AI model release has been mentioned by `@oleksandrshr` and `@tariqali`, noting its strategic placement for media attention and the fact that the model's weights are open, but details such as the name and specifications of the new Google model remain vague within the conversation.

- **Seeking Features and Clarifications**: User `@a.obie` expresses frustration over the lack of real-time sharing capabilities in the **ChatGPT Teams Plan**, pointing out the misleading nature of described features versus actual functionality. In another thread, `@zaatuloa` inquiries why **GPT 4 Turbo** might be considered a worse version, with no solid answers provided.

**Links mentioned**:

- [Chat with Open Large Language Models](https://chat.lmsys.org/): no description found
- [Introducing ChatGPT Plus](https://openai.com/blog/chatgpt-plus): We’re launching a pilot subscription plan for ChatGPT, a conversational AI that can chat with you, answer follow-up questions, and challenge incorrect assumptions.

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1209535743647817819) (54 messages🔥): 

- **GPT-3.5 Turbo Model Updates Discussed**: User `@fightray` inquired about the difference between **gpt 3.5 turbo 0125 and gpt 3.5 turbo**. `@solbus` clarified that the gpt-3.5-turbo model alias should now point to gpt-3.5-turbo-0125 as of February 16th, referencing information found [here](https://platform.openai.com/docs/models/gpt-3-5-turbo).

- **AI's Critical Thinking on Policy Edges Challenged**: `@Makeshift` criticized the AI for refusing to perform tasks it deems against policy, even when logic would suggest they're acceptable, prompting `@darthgustav.` to respond with concerns over possible misuse for plagiarism.

- **Custom GPT Struggles with Transcript Analysis**: `@col.bean` is trying to develop a GPT to extract notable moments from interview transcripts, but is facing erratic results. `@darthgustav.` advised using positively-framed instructions, a specific output template, and to potentially create new GPTs for each transcript to avoid context confusion.

- **Further Clarification on Custom GPT for Interviews Provided**: `@darthgustav.` continues to assist `@col.bean`, implying the advantages of having transcripts as txt files in the knowledge base and suggesting to start a new session per transcript for consistency in the analysis.

- **File Upload Restrictions and Usage Caps Explored**: `@my5042` raised confusion regarding file upload limits and the associated cap of 10GB per user. `@solbus` responded with a possible explanation and directed `@my5042` to an updated FAQ link for further clarification on the current file limits.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1209494439387467807) (94 messages🔥🔥): 

- **Unicode Glitches with GPT-4-1106**: `@pyramidscheme_` reported that while using GPT-4-1106 to output JSON with numbered paragraphs, the process halts at the `£` symbol, but without JSON mode, it reverts `£` to `\u000a`. No clear solution was provided.

- **Fine-Tuning AI Responses for Business**: `@silverknightgothic` sought help for an AI Assistant prompt to improve customer response based on business data; `@darthgustav` recommended using an output template with variables that summarise instruction into variable names to guide the AI's structure and content more effectively.

- **Avoid Ambiguity and Negativity in Prompts**: `@darthgustav` engaged in an in-depth discussion with `@silverknightgothic`, pointing out the potential conflicts in the prompt instructions and emphasizing the importance of using positive, specific directions to steer AI behavior.

- **Adhering to OpenAI Discord Rules**: In a series of exchanges involving potential service offering, `@eskcanta` intervened to remind about the server's Rule #7 against self-promotion or soliciting and the discussion continued with a focus on keeping AI engagement positive and productive.

- **Model Selection for Roleplaying Tasks**: `@shokkunn` and `@darthgustav` discussed the effectiveness of different GPT-4 models for roleplaying, with `@darthgustav` advising to prepare for future updates by adapting to the best current model and hinting at the complexities of designing output templates for roleplay prompts with multiple RPG custom GPTs.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1209494439387467807) (94 messages🔥🔥): 

- **Unicode Puzzles with GPT-4-1106**: User `@pyramidscheme_` encountered issues when `GPT-4-1106` processes the £ symbol, leading to truncated outputs in JSON mode or character encoding changes with JSON off.
  
- **Prompting for Precise AI Behavior**: Users like `@silverknightgothic` and `@shokkunn` discussed challenges in getting AI to produce responses fitting specific business needs or role-playing scenarios. `@darthgustav.` stressed the importance of using explicit, positive instructions and output templates with embedded instructions for better compliance.

- **Modal Mysteries - Different Models, Different Results**: `@shokkunn` compared the performance of `gpt-4-0613` and the `0125 turbo` model for roleplaying, finding the former to be more effective for natural dialogue.

- **AI as an Assistant Programmer**: Newcomer `@razorbackx9x` sought advice for using ChatGPT to help with Apple's Shortcuts app, while `@bambooshoots` provided a detailed prompt template suited for creating shortcuts in various scripting languages.

- **Keep it Positive with AI Directions**: `@eskcanta` and `@darthgustav.` advised focusing on what the AI should do rather than what it shouldn't, likening it to guiding an eager pet: keep it busy with directed tasks to avoid undesired actions.
  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1209473859414138881) (92 messages🔥🔥): 

- **Upcoming Special Edition Seminar**: `@intheclouddan` shared a link to an upcoming talk by [Carsten Binnig](https://dsdsd.da.cwi.nl/) on February 23, 2024, but noted there was no clear way to sign up for the event.
- **Interest in New Karpathy Lectures**: The community expressed interest in new lectures by [Andrej Karpathy](https://x.com/karpathy/status/1759996549109776702), one on building a GPT tokenizer and another taking a deep dive into the Gemma tokenizer.
- **AI Launchpad at Data Council**: `@petesoder` highlighted the [AI Launchpad event](https://zeroprime.vc/ai-launchpad), inviting engineer-founders to present their AI products to investors at the upcoming Data Council conference in March in Austin, TX.
- **Google Releases “Gemma” on Huggingface**: `@mjng93` introduced the release of [Google's Gemma](https://huggingface.co/blog/gemma), a new family of state-of-the-art open LLMs, and linked to their [technical report](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf) and [terms of service](https://ai.google.dev/gemma/terms). There was further discussion on the specifics and implications of the models' training and data use.
- **Magic's AI Breakthrough Analysis**: A link was shared by `@aardvarkoncomputer` to an article discussing [Magic](https://www.theinformation.com/articles/the-magic-breakthrough-that-got-friedman-and-gross-to-bet-100-million-on-a-coding-startup), a new AI coding assistant that claims to outperform current models like Gemini and GPT-4, leading to questions about its underlying technology and the future job market for computer scientists.

**Links mentioned**:

- [Tweet from Andrej Karpathy (@karpathy)](https://x.com/karpathy/status/1759996549109776702?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): New (2h13m 😅) lecture: &#34;Let&#39;s build the GPT Tokenizer&#34;  Tokenizers are a completely separate stage of the LLM pipeline: they have their own training set, training algorithm (Byte Pair Enc...
- [One Year of Latent Space](https://www.alessiofanelli.com/posts/latent-space): Lessons (and memories) from going from 0 to 1M readers in 1 year with Latent Space.
- [no title found](https://ai.google.dev/gemma/terms): no description found
- [Welcome Gemma - Google’s new open LLM](https://huggingface.co/blog/gemma): no description found
- [no title found](https://ai.google.dev/gemma/prohibited_use_policy): no description found
- [Zero Prime @ Data Council '24](https://zeroprime.vc/ai-launchpad): Join Zero Prime Ventures @ Data Council Austin 2024 for a unique chance to showcase your AI startup. Apply now for exposure to top investors and elite founders.
- [Mindblown GIF - Mindblown - Discover &amp; Share GIFs](https://tenor.com/view/mindblown-gif-19396419): Click to view the GIF
- [The ‘Magic’ Breakthrough That Got Friedman and Gross to Bet $100 Million on a Coding Startup](https://www.theinformation.com/articles/the-magic-breakthrough-that-got-friedman-and-gross-to-bet-100-million-on-a-coding-startup): Former GitHub CEO Nat Friedman and his investment partner, Daniel Gross, raised eyebrows last week by writing a $100 million check to Magic, the developer of an artificial intelligence coding assistan...
- [Rise of the AI Engineer (with Build Club ANZ)](https://www.youtube.com/watch?v=ezhSIGKFtOc): Slides:https://docs.google.com/presentation/d/157hX7F-9Y0kwCych4MyKuFfkm_SKPTN__BLOfmRh4xU/edit?usp=sharing🎯 Takeaways / highlights thread in the Build Club...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1awbo84/google_publishes_open_source_2b_and_7b_model/): no description found
- [DSDSD - The Dutch Seminar on Data Systems Design](https://dsdsd.da.cwi.nl/): An initiative to bring together research groups working on data systems in Dutch universities and research institutes.
- [Tweet from Andrej Karpathy (@karpathy)](https://x.com/karpathy/status/1760350892317098371?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): Seeing as I published my Tokenizer video yesterday, I thought it could be fun to take a deepdive into the Gemma tokenizer.   First, the Gemma technical report [pdf]:  https://storage.googleapis.com/de...
- [Tweet from Deedy (@debarghya_das)](https://x.com/debarghya_das/status/1759786243519615169?s=20): It&#39;s embarrassingly hard to get Google Gemini to acknowledge that white people exist

  


---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1209953727373377596) (3 messages): 

- **Dive into "Building Your Own Product Copilot" Paper**: `@swyxio` announced that `@451508585147400209` is leading a session on the paper [Building Your Own Product Copilot](https://arxiv.org/abs/2312.14231) and provided a link to the relevant Discord channel for those interested in joining.
- **Stay Updated with Latent Spaces Events**: `@swyxio` highlighted how to stay informed about [Latent.Space](http://Latent.Space) events—by clicking the RSS logo above the calendar on the right to add the event notifications to your calendar using the "Add iCal Subscription" feature. They provided a direct link to sign up for event notifications [here](https://lu.ma/ls).

**Links mentioned**:

[Latent Space (Paper Club &amp; Other Events) · Luma](https://lu.ma/ls): View and subscribe to events from Latent Space (Paper Club &amp; Other Events) on Luma. Latent.Space events. PLEASE CLICK THE RSS LOGO JUST ABOVE THE CALENDAR ON THE RIGHT TO ADD TO YOUR CAL. &quot;Ad...

  

---


### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1209952591669370940) (173 messages🔥🔥): 

- **GPT as Engineers' Co-Pilot**: @eugeneyan shared insights and sparked a discussion around the integration of AI capabilities into products, with professional software engineers facing challenges at every step of the engineering process. The paper being referred to is an [interview study of engineers](https://arxiv.org/abs/2312.14231) working on product copilots.

- **Google Gemini Takes the Stage**: @coffeebean6887 pointed to recent updates from Google, highlighting the integration of the Gemini AI across various Google Workspace applications and the new offerings for users. Official information is available in Google's blog posts about [Google One Gemini](https://blog.google/products/google-one/google-one-gemini-ai-gmail-docs-sheets/) and [Gemini for Google Workspace](https://blog.google/products/workspace/google-gemini-workspace/).

- **Debates on Fine-tuning and Testing LLMs**: The group actively discussed various strategies for evaluating language models, with members like @henriqueln7 and @_bassboost sharing their experiences and approaches to using different evaluation tools, including Langsmith and potential use of smaller models like Mistral for judging performance.

- **Focus on Evaluations for AI Applications**: The chat highlighted the importance of evaluation methods in AI integration, with users like @djmcflush and @ayenem mentioning the use of tools like uptrain and LLama Guard for testing LLM outputs. The conversation underscored the challenges in prompt design and optimization.

- **The Future of AI Integration and Talent in Companies**: @lightningralf and @eugeneyan engaged in a forward-looking discussion on how the rapid development of AI models will lead to a shift in the skillsets required within companies. There was a consensus on the increasing importance of evals, and prompt programming, alongside finetuning, as critical skills.

**Links mentioned**:

- [Building Your Own Product Copilot: Challenges, Opportunities, and Needs](https://arxiv.org/abs/2312.14231): A race is underway to embed advanced AI capabilities into products. These product copilots enable users to ask questions in natural language and receive relevant responses that are specific to the use...
- [no title found](https://open-vsx.org/extension/Continue/continue)): no description found
- [New ways Google Workspace customers can use Gemini](https://blog.google/products/workspace/google-gemini-workspace/): We’re launching a new offering to help organizations get started with generative AI, plus a standalone experience to chat with Gemini.
- [LoRA Land: Fine-Tuned Open-Source LLMs](https://predibase.com/lora-land): Fine-tuned LLMs that outperform GPT-4, served on a single GPU
- [Boost your productivity: Use Gemini in Gmail, Docs and more with the new Google One plan](https://blog.google/products/google-one/google-one-gemini-ai-gmail-docs-sheets/): We’re bringing even more value to the Google One AI Premium plan with Gemini in Gmail, Docs, Slides, Sheets and Meet (formerly Duet AI).
- [- Fuck You, Show Me The Prompt.](https://hamel.dev/blog/posts/prompt/): Quickly understand inscrutable LLM frameworks by intercepting API calls.
- [Tweet from Amazon Web Services (@awscloud)](https://x.com/awscloud/status/1752051165200601299?s=46&t=90xQ8sGy63D2OtiaoGJuww): The PartyRock #generativeAI Hackathon by #AWS starts now! 📣  Learn how to build fun & intuitive apps without coding for a chance to win cash prizes and AWS credits. 🏆 #AI   Don&#39;t forget your mug...
- [SPQA: The AI-based Architecture That’ll Replace Most Existing Software](https://danielmiessler.com/p/spqa-ai-architecture-replace-existing-software/): March 10, 2023 AI is going to do a lot of interesting things in the coming months and years, thanks to the detonations following GPTs. But one of the most impor
- [Founder’s Guide to Basic Startup Infrastructure](https://www.flexport.com/blog/founders-guide-to-basic-startup-infrastructure/): no description found
- [GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models](https://github.com/stanfordnlp/dspy): DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1209528211067641966) (165 messages🔥🔥): 

- **GPU Choices for Corporate Client**: Yamashi (@yamashi) debates between purchasing 8xH100 or 8xMI300X GPUs, leaning towards the MI300X for its higher VRAM but expressing concerns about compatibility with ROCm software. In the context of their cloud versus on-premises calculations, they mention dealing with 15,000 VMs in the cloud due to massive hospital data processes.
  
- **Google's Gemma Chat Models Analyzed**: The AI Collective (@le_mess, @nafnlaus00) discusses the details and implications of Google's newly released Gemma models, which come in 2B and 7B variants with pyramidal ranks. Concerns include Gemma's licensure and output restrictions, evaluation methods, and the specifics of its custom chat format, which several members find slow for fine-tuning compared to other models.

- **Flash Attention's Impact on Fine-Tuning**: Conversation (@nruaif, @nanobitz, @dreamgen) focused on the use of Flash Attention with Gemma. Updates to Flash Attention (v2.5.5) now allow head dimension 256 backward on consumer GPUs, which may aid in fine-tuning these large models on such GPUs including RTX 4090.

- **Gemma Unveiled with Instruction-Tuned Variants**: Details of the Gemma 7B model's architecture intended for fine-tuning, including its preference for GPU types for best performance. The new instruction-tuned `-it` variants of Gemma are highlighted, said to offer improved performance through specific fine-tuning approaches.

- **Gemma's Open Access and Reposting**: Le_mess (@le_mess) re-uploads the Gemma model to Hugging Face platform to make it available ungated, and notifies of the success in transferring the 32GB .gguf file without a crash. Conversations (@j_sp_r, @stoicbatman) pinpoint the intricacies of fine-tuning Gemma models, including potential speed issues and compatibility with the Flash Attention tool.

**Links mentioned**:

- [no title found](https://ai.google.dev/gemma): no description found
- [HuggingChat](https://huggingface.co/chat): Making the community's best AI chat models available to everyone.
- [Google introduces a lightweight open AI model called Gemma](https://www.engadget.com/google-introduces-a-lightweight-open-ai-model-called-gemma-130053289.html): Google says Gemma is its contribution to the open community and is meant to help developers &#x22;in building AI responsibly.&#x22;
- [Welcome Gemma - Google’s new open LLM](https://huggingface.co/blog/gemma/?utm_source=agd&utm_medium=referral&utm_campaign=view-on-huggingface&utm_content=): no description found
- [mhenrichsen/gemma-7b · Hugging Face](https://huggingface.co/mhenrichsen/gemma-7b): no description found
- [Tweet from Tri Dao (@tri_dao)](https://x.com/tri_dao/status/1760458183066472556?s=20): FlashAttention v2.5.5 now supports head dim 256 backward on consumer GPUs. Hope that makes it easier to finetune Gemma models
- [mhenrichsen/gemma-7b-it · Hugging Face](https://huggingface.co/mhenrichsen/gemma-7b-it): no description found
- [llm-foundry/scripts/train/README.md at main · mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#howmandygpus): LLM training code for MosaicML foundation models. Contribute to mosaicml/llm-foundry development by creating an account on GitHub.
- [GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention](https://github.com/Dao-AILab/flash-attention): Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.
- [Enable headdim 256 backward on consumer GPUs (Ampere, Ada) · Dao-AILab/flash-attention@2406f28](https://github.com/Dao-AILab/flash-attention/commit/2406f28805e2a3623427f48f38fc533a5d1f2c32): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1209507987018883082) (23 messages🔥): 

- **LoRA+ Paper Shared for Optimized Finetuning**: User `@suikamelon` shared an [arXiv paper](https://arxiv.org/abs/2402.12354) on **LoRA+**, an algorithm improving the finetuning of models with large widths by using different learning rates for LoRA adapter matrices, leading to significant performance boosts.
- **Integration of LoRA+ into Axolotl**: `@caseus_` mentioned that integrating LoRA+ appears straightforward, with a GitHub [repository available](https://github.com/nikhil-ghosh-berkeley/loraplus) and suggested it could be easily added to Axolotl.
- **Latest Transformers Version Required for Gemma Model**: `@giftedgummybee` noted the necessity to use the updated non-dev version of transformers (`pip install -U transformers`) for training Gemma models, as the dev version does not support "gemma" types.
- **Updated Axolotl Config Shared for Gemma Training**: `@stoicbatman` provided an [updated Gemma config](https://gist.github.com/monk1337/b7ee08781d62e351db7fc7c6fe0645e0) for compatibility with Axolotl, while others like `@nanobitz` confirmed ongoing testing.
- **Discussions on Gemma Learning Rate and Weight Decay Values**: Amidst configurations for the Gemma model, `@faldore` pointed out inconsistencies in Google's documentation regarding learning rate and weight decay, referencing both their [Colab notebook](https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/lora_tuning.ipynb) and Hugging Face [blog post](https://huggingface.co/blog/gemma/) showing different values used.

**Links mentioned**:

- [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354): In this paper, we show that Low Rank Adaptation (LoRA) as originally introduced in Hu et al. (2021) leads to suboptimal finetuning of models with large width (embedding dimension). This is due to the ...
- [Google Colaboratory](https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemma/docs/lora_tuning.ipynb#scrollTo=_Peq7TnLtHse&line=1&uniqifier=1): no description found
- [Welcome Gemma - Google’s new open LLM](https://huggingface.co/blog/gemma/?utm_source=agd&utm_medium=referral&utm_campaign=view-on-huggingface&utm_content=#fine-tuning-with-%F0%9F%A4%97-trl): no description found
- [gemma_config_axolotl.yml](https://gist.github.com/monk1337/b7ee08781d62e351db7fc7c6fe0645e0): GitHub Gist: instantly share code, notes, and snippets.
- [GitHub - nikhil-ghosh-berkeley/loraplus](https://github.com/nikhil-ghosh-berkeley/loraplus): Contribute to nikhil-ghosh-berkeley/loraplus development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1209523401652306032) (27 messages🔥): 

- **How to Include Chat Template in Model**: `@yamashi` initially had an issue including the chat template with a model but later found the solution in the tokenizer_config.json file on [Hugging Face's repository](https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b/blob/73de11894bf917d9501b3b949bcd8056ed2e777f/tokenizer_config.json#L48).
- **Alpaca Jinja Template Needed**: `@yamashi` mentioned the need for an Alpaca jinja template, prompting `@le_mess` to suggest adding an existing template to the repo if found, leading to a collaborative effort to locate or create the required template.
- **Alpaca Template Discovered and Shared**: After some discussion and contribution from `@rtyax`, `@yamashi` shared a modified chat template for Alpaca and stated an intention to push it to the Axolotl repository as well, indicating a successful community collaboration.
- **DeepSpeed Steps Confusion**: User `@napuh` asked for clarification on why running tests with fewer GPUs resulted in more steps instead of the same number with longer completion time, signaling a need to understand DeepSpeed's behavior better.
- **Inference Format for Finetuned Model**: `@timisbister` finetuned a model using the openhermes2.5 dataset and asked for the correct format for inference, leading to guidance from `@rtyax` and `@nanobitz` about using USER and ASSISTANT formatting and special tokens like `[bos_token]`.

**Links mentioned**:

- [tokenizer_config.json · cognitivecomputations/dolphin-2.5-mixtral-8x7b at 73de11894bf917d9501b3b949bcd8056ed2e777f](https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b/blob/73de11894bf917d9501b3b949bcd8056ed2e777f/tokenizer_config.json#L48>): no description found
- [tokenizer_config.json · teknium/OpenHermes-2.5-Mistral-7B at main](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/tokenizer_config.json): no description found
- [text-generation-webui/instruction-templates/Alpaca.yaml at main · oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Alpaca.yaml): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui

  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1209747249584930906) (1 messages): 

- **Concerns Over DPO Training Logs**: `@noobmaster29` is seeking shared **DPO training logs** as a reference, concerned that their own loss is unusually low before completing one epoch. They are unable to determine if this is overfitting due to lack of eval data.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1209449021685506058) (5 messages): 

- **RunPod Image Fetch Errors**: `@casper_ai` explained that RunPod is persistently trying and failing to fetch an image, indicating an error message that the image cannot be found.
- **RunPod Infinite Retries Quirk**: `@casper_ai` clarified that RunPod has a behavior of **infinite retries** even if it encounters errors in finding an image.
- **Disk Space Culprit for Image Issues**: `@c.gato` pointed out that the actual issue might be that the RunPod is **out of disk space**, which prevents the download of the image, leading to the repeated failures.
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1209434197672857600) (185 messages🔥🔥): 

- **Deep in the Search for Datasets**: `@swaystar123` inquired if there exists a dataset containing all public git repositories along with their issues, comments, pull requests, and commits. However, the chat did not provide a specific answer or resources.
  
- **LAION 5B Availability Queries**: `@risphere` asked about the availability of the LAION 5B dataset. No answer was given in this chat snippet.

- **Lucidrains’ Funding and Work Discussions**: There was active discussion involving `@itali4no`, `@chad_in_the_house`, `@segmentationfault8268`, and others about [Phil Wang (lucidrains)](https://github.com/sponsors/lucidrains#sponsors) and his previous funding, contributions to translating AI papers to code, and potential for employment at any AI company.

- **Google’s Gemma Open Models Announcement**: `@itali4no` shared a [link](https://blog.google/technology/developers/gemma-open-models/) to a Google blog post about Gemma, their new initiative for responsible AI development, which leads to a discussion including `@helium__`, `@segmentationfault8268`, `@twoabove` and others about Google's history of open source contributions and speculations on model architectures.

- **Captioning Project Progress with pseudoterminalx**: `@pseudoterminalx` provided updates on an image captioning project, mentioning the utilization of multiple Nvidia GPUs and citing significant progress with the COG-VLM model. They also mentioned the project's aim, the finetuning of the Terminus model, and making the data public.

**Links mentioned**:

- [generative modelling of compressed image file bits](https://theadamcolton.github.io/generative-modelling-of-compressed-image-file-bits): no description found
- [no title found](https://ai.google.dev/gemma/docs/model_card): no description found
- [18-year-old Miamian Kye Gomez is developing AI to make life less boring - Refresh Miami](https://refreshmiami.com/18-year-old-miamian-kye-gomez-is-developing-ai-to-make-life-less-boring/): By Riley Kaminer
- [Gemma: Introducing new state-of-the-art open models](https://blog.google/technology/developers/gemma-open-models/): Gemma is a family of lightweight, state\u002Dof\u002Dthe art open models built from the same research and technology used to create the Gemini models.
- [Sponsor @lucidrains on GitHub Sponsors](https://github.com/sponsors/lucidrains#sponsors): Hello, I democratize artificial intelligence by open sourcing state-of-the-art neural network architectures. I am primarily a specialist with a sub-field of deep learning: Attention and Transformers
- [ptx0/photo-concept-bucket · Datasets at Hugging Face](https://huggingface.co/datasets/ptx0/photo-concept-bucket): no description found

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1209436045540724806) (30 messages🔥): 

- **Awaiting LAION 5B's Return**: `@progamergov` responded with a tongue-in-cheek "soon™" to the question about **Laion 5B**'s availability, while `@JH` suggested using [DFN 2B](https://huggingface.co/datasets/apf1/datafilteringnetworks_2b) as a possible alternative, despite issues accessing the dataset.
- **Debate Over Sora's Training Data**: `@yoavhacohen`, `@chad_in_the_house`, and `@unjay.` discussed whether **Sora** was trained on real or synthetic videos, with observations about the generated output's quality and potential signs of synthetic data usage such as ''floaters'' and lack of dynamic backgrounds in the clips.
- **OpenAI's Usage of Synthetic Data**: `@atlasunified` initially claimed that OpenAI does not use synthetic data due to having the resources to procure real data, but `@helium__` countered by referencing OpenAI's paper that mentioned the use of synthetic data. `@atlasunified` admitted to having not read the paper after being corrected.
- **Insights into AnyGPT's Discrete Sequence Modeling**: `@helium__` provided a link to the [AnyGPT project](https://junzhan2000.github.io/AnyGPT.github.io/), which aims to process various modalities through a language model using discrete representations, and shared a [YouTube demo video](https://youtu.be/oW3E3pIsaRg).
- **Synthetic Data in Focus**: The discussion on synthetic data continued with `@yoavhacohen` and `@unjay.` debating its role in the development of models, citing the importance of accurate labeling and considering the intricacies of fine-tuning with synthetic versus real-world data.

**Links mentioned**:

- [apf1/datafilteringnetworks_2b · Datasets at Hugging Face](https://huggingface.co/datasets/apf1/datafilteringnetworks_2b): no description found
- [Neural Network Diffusion](https://arxiv.org/abs/2402.13144): Diffusion models have achieved remarkable success in image and video generation. In this work, we demonstrate that diffusion models can also \textit{generate high-performing neural network parameters}...
- [AnyGPT](https://junzhan2000.github.io/AnyGPT.github.io/): no description found
- [Demo for &quot;AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling&quot;](https://youtu.be/oW3E3pIsaRg): Demo for &quot;AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling&quot;

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1209431902944235530) (111 messages🔥🔥): 

- **Perplexity Pro User Shares Quick Access Method**: User `@norgesvenn` described **Perplexity Pro** as fast and accurate. `@mares1317` responded by assigning the Pro role and provided a Discord channel link for Pro users, although the link appeared to be incomplete.
  
- **Balancing Source Reliance with AI Creativity**: `@annie7441` inquired about prompting **Perplexity** AI to balance between search results and creative outputs from its training data. A suggestion from `@brknclock1215` advised adding a specific instruction to not rely on search results exclusively, and to draw on existing knowledge.

- **Exploring Gemini's Advantages Over GPT-4**: Multiple users, including `@tree.ai` and `@akumaenjeru`, discussed the merits of **Gemini Advanced** over GPT-4, citing its updated models and preference over the output style.

- **Accessing Perplexity Pro Channels on Discord**: New **Perplexity Pro** users like `@atripper` were seeking access to exclusive channels; `@me.lk` offered guidance to use an invite link from the user's settings page.

- **Using Perplexity.ai to Generate Images**: `@psoulos` questioned how to generate images with Perplexity, and `@ok.alex` directed to use the specific "Generate Image" button, linking to an example (although the link was incomplete).

**Links mentioned**:

- [Code is a Four Letter Word: Gemini Versus Gemini: Understanding Google's Latest... Thing](https://code.iaflw.com/2024/02/gemini-versus-gemini-understanding.html): no description found
- [Gemini 1.5: Our next-generation model, now available for Private Preview in Google AI Studio - Google for Developers](https://developers.googleblog.com/2024/02/gemini-15-available-for-private-preview-in-google-ai-studio.html?m=1): no description found
- [Sam Witteveen](https://www.youtube.com/@samwitteveenai): HI my name is Sam Witteveen, I have worked with Deep Learning for 9 years and with Transformers and LLM for 5+ years.  I was appointed a Google Developer Expert for Machine Learning in 2017 and I curr...

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1209509032771985458) (7 messages): 

- **Exploring Programming Language Speeds**: `@incrediblemhi` shared a [link](https://www.perplexity.ai/search/progrmming-language-speed-d7vR1rKoRPG0ZhkQRMbBiw?s=c#77bbd1d6-b2a8-44f1-b466-191044c6c18b) to a Perplexity AI search regarding the speed of different programming languages. The search offers comparisons and details on how languages perform.
- **Understanding the Mysterious**: `@thespacesamurai` posted a [link](https://www.perplexity.ai/search/which-are-the-Zgc3h8XuR5Sdug1UqCSTvw?s=c#81bbe4e1-38a5-42e0-8a0b-42d882c995a0) that seems to address a question or topic with missing detail, possibly related to a list or types of a certain category.
- **Seeking Knowledge on Unspecified Topic**: `@lcyld` provided a [link](https://www.perplexity.ai/search/ppN7j_P9TIytL4XxcwqJhA?s=c) which leads to a Perplexity AI search, but there is no context given on the nature of the search question or topic.
- **Discovery of Concepts in a Perplexity AI Channel**: `@doris_dmh` referenced a Perplexity AI search [link](https://www.perplexity.ai/search/What-are-the-eVCy7rcVRtS8v6KbN9jRcA?s=c) and pointed to a specific Discord channel `#1054944216876331118` for further discussions or insight.
- **Finding the Meaning of Derivate**: `@ivanrykovski` shared a [link](https://www.perplexity.ai/search/what-does-dydx-Vo_6.U1XQg.eDbP_lg0FHQ?s=c) to a Perplexity AI search concerning the mathematical notation `dy/dx`, indicating an interest in calculus or related mathematical concepts.

- **Healthier Choices for Oral Care**: `@uberkoolsound` pointed to a [link](https://www.perplexity.ai/search/Does-salt-water-muDDr.Z9RHy_EvMKEhurPg?s=c) regarding the use of saltwater, following inspiration from Andrew Huberman and Paul Saladino to reduce processed chemicals in oral hygiene and skin care.

- **Defining the Unknown**: `@swordfish01` sought clarification with a [link](https://www.perplexity.ai/search/What-is-a-fDAg8dSNRhmEeKU.SoY6Fg?s=c) about an unspecified subject, prompting curiosity about the content behind the link provided.
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1209497852523446312) (18 messages🔥): 

- **Seeking an API Request Limit Increase**: User `@kentapp_13575` inquired about increasing their API request limit to 15k per day for the model **pplx-70b-chat**. No direct response was provided in the channel.
- **API vs Webapp Discrepancy**: `@iflypper` reported getting different responses from the API compared to the out-of-date webapp. `@brknclock1215` suggested removing the system prompt, although `@iflypper` found that doing so resulted in irrelevant responses.
- **Stream Parameter Inquiry**: User `@bayang7` requested an example of a request using `stream: true`. No response was provided to this inquiry in the channel.
- **Embedding Links in API Responses**: `@gruby_saudi` was looking for a way to make the API responses include embedded links, similar to the webapp's output, but no solution was provided in the messages.
- **Gibberish Responses from pplx-70b-online**: `@useful_tom` experienced issues with API responses from **pplx-70b-online**, where the response starts well but transitions into gibberish. They noted that other users have reported the same issue and contemplated trying the **7b model** as a workaround.

**Links mentioned**:

[Supported Models](https://docs.perplexity.ai/docs/model-cards): no description found

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1209746313579593811) (3 messages): 

- **CUDA MODE Welcomes All GPU Fans**: `@andreaskoepf` reaffirmed the inclusivity of the CUDA MODE community, linking a tweet where NVIDIA pitched Grace Hopper chips to `@__tinygrad__` which can be seen at [tinygrad's tweet](https://x.com/__tinygrad__/status/1760039877922758710).
- **Groq's LPU: A Breakthrough in AI Performance**: `@srns27` excitedly shared [an article](https://cryptoslate.com/groq-20000-lpu-card-breaks-ai-performance-records-to-rival-gpu-led-industry/) about Groq's LPU Chip setting new performance benchmarks in processing large language models, and requested insights into the technology.
- **Understanding Groq's Architecture**: `@dpearson` responded to the curiosity about Groq's LPU with a [YouTube video](https://youtu.be/PKJYU9ecvWc?si=9BKG75HsaEGTVgMH) explaining why it's so fast, featuring Groq's Compiler Tech Lead Andrew Bitar.

**Links mentioned**:

- [Groq&#039;s $20,000 LPU chip breaks AI performance records to rival GPU-led industry](https://cryptoslate.com/groq-20000-lpu-card-breaks-ai-performance-records-to-rival-gpu-led-industry/).): Groq&#8217;s LPU Inference Engine, a dedicated Language Processing Unit, has set a new record in processing efficiency for large language models. In a recent benchmark conducted by ArtificialAnalysis....
- [Tweet from the tiny corp (@__tinygrad__)](https://x.com/__tinygrad__/status/1760039877922758710): NVIDIA reached out and tried to sell us on using Grace Hopper chips. This was our reply.
- [Software Defined Hardware for Dataflow Compute / Crossroads 3D-FPGA Invited Lecture by Andrew Bitar](https://youtu.be/PKJYU9ecvWc?si=9BKG75HsaEGTVgMH): Invited lecture by Groq&#39;s Compiler Tech Lead, Andrew Bitar, for the Intel/VMware Crossroads 3D-FPGA Academic Research Center on Dec 11, 2022.Abstract: With t...

  

---


### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1209488639906807908) (8 messages🔥): 

- **Seeking Collaborators for Triton/Mamba**: `@srush1301` is compiling resources for Triton/Mamba and is looking for assistance to improve performance. They have outlined remaining tasks such as adding a reverse scan option in Triton and comparing with the CUDA version in a guide at [Annotated Mamba](https://srush.github.io/annotated-mamba/hard.html).

- **Optimization Roadmap for Triton/Mamba**: `@srush1301` believes that to speed up performance, the module needs enhancements like reverse scan in Triton, block size optimization, and a comparative analysis against the CUDA version.

- **Invitation to Coauthor**: `@srush1301` is open to adding interested contributors as coauthors to their Triton/Mamba project.

- **Offer for Dedicated Working Group Channel**: `@marksaroufim` suggests creating a dedicated channel for the Triton/Mamba working group, similar to what has been done for another project.

- **Tools for Diagramming**: `@srush1301` responds to `@morgangiraud`'s query about diagram creation tools, revealing they use Excalidraw for simple diagrams and chalk-diagrams for GPU puzzles.

**Links mentioned**:

[Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html): no description found

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1209864248767811634) (5 messages): 

- **Device Function Pointer Dilemma**: `@carrot007` sought assistance regarding the use of device function pointers in global functions, facing a warning when using `cudaMemcpyFromSymbol` related to the inability to read a **`__device__` variable** in a host function.
- **Nvidia Nsight Debugging in Docker**: `@dvruette` enquired about installing Nvidia Nsight inside a docker container to **debug issues on vast.ai**, sparking discussion on the challenges cloud providers face with such tools.
- **Mark's Solution for Cloud-based Debugging**: `@marksaroufim` mentioned that **lighting.ai studios** is the only cloud provider to have effectively addressed the problem mentioned by `@dvruette`, suggesting them as a potential solution.
- **Free Credits and Support for CUDA Mode Members**: `@lntg` announced that **CUDA mode members** could access free credits for GPU hours on a particular platform and offered to expedite the verification process for them.
- **Profiling Warnings Puzzle Users**: `@complexfilterr` reported encountering a warning stating **"No kernels were profiled"** while attempting to profile straightforward CUDA code, causing confusion about the reason behind this.
  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1209463330004013056) (15 messages🔥): 

- **Broadcasting Semantics Inquiry by linfeng810**: `@linfeng810` asked about how `torch.mul` treats dimensions in the context of data locality when dealing with broadcast-able dimensions. The discussion involves PyTorch's approach to expanding Tensor arguments and the user provided a [link to PyTorch broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html) for details.

- **Tackling a Debugging Conundrum with Geforce 40 Series**: `@morgangiraud` shared their debugging experience related to the Geforce 40 series not supporting P2P, offering a post-mortem write-up with insights on the issue, which can be read [here](https://morgangiraud.medium.com/fb7597b4e2b5).

- **Details on PyTorch Optimization Shared by hdcharles_74684**: `@hdcharles_74684` mentioned their work on several projects, including llama, SAM, and SDXL, with a focus on PyTorch optimization, and provided a series of blog posts that can be found at [PyTorch Blog Part 1](https://pytorch.org/blog/accelerating-generative-ai/), [Part 2](https://pytorch.org/blog/accelerating-generative-ai-2/), and [Part 3](https://pytorch.org/blog/accelerating-generative-ai-3/).

- **Acceleration and Optimization of Generative AI Models**: `@gogators.` emphasized that custom kernels can sometimes greatly outperform PyTorch native kernels, especially at batch size 1. They noted that `torch.compile` cannot handle dynamic control flow and might not realize all potential kernel fusion gains.

- **Special Mention of Transformation Optimizations**: Another contribution by `@hdcharles_74684` discusses the use of different kernels in the optimization process, such as compiled kernels and Triton kernels for operations like scaled dot product attention (SDPA) and matrix multiplication. They provide a link to [quantization details in their repository](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L269).

**Links mentioned**:

- [Broadcasting semantics &mdash; PyTorch 2.2 documentation](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics): no description found
- [Multi-gpu (Nvidia) P2P capabilities and debugging tips](https://morgangiraud.medium.com/fb7597b4e2b5): A journey into troubleshooting multi-GPU Nvidia build, focusing on P2P capabilities, installation and testing!
- [Build software better, together](https://github.com/pytorch/pytorch/pull/18540).): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
- [torch.cuda.jiterator._create_jit_fn &mdash; PyTorch 2.2 documentation](https://pytorch.org/docs/stable/generated/torch.cuda.jiterator._create_jit_fn.html): no description found
- [gpt-fast/model.py at main · pytorch-labs/gpt-fast](https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L185C15-L185C43): Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - pytorch-labs/gpt-fast
- [segment-anything-fast/segment_anything_fast/flash_4.py at main · pytorch-labs/segment-anything-fast](https://github.com/pytorch-labs/segment-anything-fast/blob/main/segment_anything_fast/flash_4.py): A batched offline inference oriented version of segment-anything - pytorch-labs/segment-anything-fast
- [Accelerating Generative AI with PyTorch: Segment Anything, Fast](https://pytorch.org/blog/accelerating-generative-ai/): This post is the first part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance ...
- [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/): This post is the second part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance...
- [Accelerating Generative AI Part III: Diffusion, Fast](https://pytorch.org/blog/accelerating-generative-ai-3/): This post is the third part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance ...
- [pytorch/torch/_inductor/kernel/mm.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L269): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [pytorch/aten/src/ATen/native/native_functions.yaml at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L3343C9-L3343C29): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [pytorch/aten/src/ATen/native/native_functions.yaml at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml#L4093): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch



---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1209620864166002769) (14 messages🔥): 

- **Conda Cuda Catastrophe**: `@apaz` shared a frustrating experience with trying to install CUDA libraries via conda, facing multiple issues like slow installation, wrong version of torch, breaking the runtime, and ultimately removing and reinstalling conda. They expressed dissatisfaction with Conda due to this ordeal.

- **Conda Woes Spark Community Feedback**: `@andreaskoepf` and `@_t_vi_` responded to `@apaz`'s experience, suggesting alternative experiences with different distributions, such as venv or Debian's CUDA package, noting fewer issues.

- **Jeremy's Tips for Conda Usage**: In response to `@apaz`'s issues with conda, `@jeremyhoward` acknowledged the frustration and recommended ensuring libmamba solver's usage for speed and questioning the need for extensive removals, suggesting that deleting a single directory is usually sufficient.

- **CUDA Compile Concerns**: `@0ut0f0rder` noted slow compile times for a simple CUDA kernel using torch_inline, questioning if the nature of CUDA or the Colab hardware was the cause, while `@jeremyhoward` responded affirming that CUDA does indeed have slow compile times, but suggesting numba as a faster alternative.

- **Will CUDA Become Obsolete?**: `@dpearson` questioned the need to learn CUDA after watching a video on Groq AI hardware, which highlighted efficient compiler resource utilization and the potential for future automation, sparking a discussion on whether learning CUDA is still valuable.

**Links mentioned**:

- [Ah Shit Here We Go Again Cj GIF - Ah Shit Here We Go Again Ah Shit Cj - Discover &amp; Share GIFs](https://tenor.com/view/ah-shit-here-we-go-again-ah-shit-cj-gta-gta-san-andreas-gif-13933485): Click to view the GIF
- [Software Defined Hardware for Dataflow Compute / Crossroads 3D-FPGA Invited Lecture by Andrew Bitar](https://youtu.be/PKJYU9ecvWc?t=1906): Invited lecture by Groq&#39;s Compiler Tech Lead, Andrew Bitar, for the Intel/VMware Crossroads 3D-FPGA Academic Research Center on Dec 11, 2022.Abstract: With t...

  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1209912889708908614) (1 messages): 

- **Gemini 1.5 Discussion Invite**: `@shashank.f1` is hosting a new discussion on **Gemini 1.5** and is inviting members to join it live. The announcement includes a Discord invite link for the event.

- **Exploring AI that Understands Audio**: `@shashank.f1` shared a [YouTube video](https://youtu.be/FgcN62LFzIU) titled "A-JEPA AI model: Unlock semantic knowledge from .wav / .mp3 file or audio spectrograms," which covers a discussion on the **A-JEPA AI model** that can extract semantic knowledge from audio files. The description indicates that the video features a deep dive with experts Oliver, Nevil, Ojasvita, Shashank, Srikanth, and N...

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/F4FfcQw3?event=1209440306404139008): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [A-JEPA AI model: Unlock semantic knowledge from .wav / .mp3 file or audio spectrograms](https://youtu.be/FgcN62LFzIU): 🌟 Unlock the Power of AI Learning from Audio ! 🔊 Watch a deep dive discussion on the A-JEPA approach with Oliver, Nevil, Ojasvita, Shashank, Srikanth and N...

  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1209592583341346867) (3 messages): 

- **Pallas Flash Attention Inquiry**: `@iron_bound` shared a [GitHub link](https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py) to `flash_attention.py` in the jax repository, asking if anyone has previously used it.
- **Potential GPU Compatibility Curiosity**: `@drexalt` speculated on the compatibility of the TPU-specific code for GPU usage, expressing an interest in modifying the code by removing repeat calls for TPU to see if it would run on a GPU.
- **Shape Dimensions Mishap**: `@iron_bound` reported that an attempt to run the code resulted in a crash due to a "dreaded shape dimensions error," though it did start.

**Links mentioned**:

- [jax/jax/experimental/pallas/ops/tpu/flash_attention.py at main · google/jax](https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py): Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - google/jax
- [jax/jax/experimental/pallas/ops/tpu at main · google/jax](https://github.com/google/jax/blob/main/jax/experimental/pallas/ops/tpu): Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more - google/jax

  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1209490996522516510) (52 messages🔥): 

- **Collaborative Efforts on Ring Attention and Flash Attention**: Participants like `@iron_bound`, `@ericauld`, `@lancerts`, and `@andreaskoepf` have been actively discussing the implementation and understanding of ring reduction and flash attention algorithms. Significant contributions include `@ericauld` sharing a Google Colab notebook as a [working dummy version](https://colab.research.google.com/drive/1B9oD4oeuYqK5szEHfrS0VlnVbmLCO9HA) of the ring attention algorithm and `@lancerts` implementing a forward pass of flash attention in a [naive Python notebook](https://github.com/lancerts/Algo-From-Scratch/blob/main/flash_attn/naive_flash.ipynb).

- **Diving into the Code**: `@iron_bound` translated a significant amount of code into PyTorch, although it still requires `torch.distributed` to be added. They've highlighted the complexity and existence of nested functions in the translated code.

- **Identification of Potential Typos in the Literature**: Users like `@ericauld` and `@lancerts` have pointed out potential typos in the flash attention literature that might be impacting the accuracy of their code implementations. `@ericauld` specifically mentioned discrepancies related to the factors of `diag(l)^{-1}` and the ratio `l2/l1` within their notebook.

- **Engagement with External Resources**: `@iron_bound` found a GitHub issue leading to another implementation of ring-flash attention by `@zhuzilin` and shared the [link](https://github.com/zhuzilin/ring-flash-attention/) for others to explore.

- **Learning and Progress Updates**: Users like `@mickgardner` and `@jamesmel` have shared their learning experiences with the mathematical and practical aspects of the project. `@mickgardner` contemplated learning cutlass primitives while `@jamesmel` delved into p2p overlap and now feels confident in understanding the requirements.



**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1B9oD4oeuYqK5szEHfrS0VlnVbmLCO9HA#scrollTo=4g829FqpGIis): no description found
- [Google Colaboratory](https://colab.research.google.com/drive/1B9oD4oeuYqK5szEHfrS0VlnVbmLCO9HA#scrollTo=M-lh5Fk7rSLY): no description found
- [ring-attention/ring_attn/ring_attention.py at tests · Iron-Bound/ring-attention](https://github.com/Iron-Bound/ring-attention/blob/tests/ring_attn/ring_attention.py): Optimized kernels for ring-attention [WIP]. Contribute to Iron-Bound/ring-attention development by creating an account on GitHub.
- [GitHub - zhuzilin/ring-flash-attention: Ring attention implementation with flash attention](https://github.com/zhuzilin/ring-flash-attention/): Ring attention implementation with flash attention - zhuzilin/ring-flash-attention
- [ring-attention/notebooks/DummyRingAttentionImpl.ipynb at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/blob/main/notebooks/DummyRingAttentionImpl.ipynb): Optimized kernels for ring-attention [WIP]. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.
- [ir - Overview](https://github.com/Ir): ir has 4 repositories available. Follow their code on GitHub.

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1209427865389502535) (46 messages🔥): 

- **Concerns about TypeScript Compatibility**: User `@amur0501` inquired if using **LangChain with TypeScript** is as effective as with Python, wondering if they would miss out on features available in Python when using `langchain.js`. No clear consensus or recommendations were provided in response.
- **Function Calling with Mistral**: `@kipkoech7` shared a [GitHub link](https://github.com/AIAnytime/Function-Calling-Mistral-7B) demonstrating the use of **function calling on Open-Source LLMs (Mistral)** with LangChain both locally and using Mistral's API key.
- **Question on Vector Database Indexing**: `@m4hdyar` asked for strategies on updating indexes in a vector database when code changes, and `@vvm2264` suggested using a **code chunk-tagging system** or a **1:1 mapping formula** for code chunks to vector indices to facilitate updates.
- **Request for Updated NLP Resources**: `@nrs9044` sought recommendations for materials more current than **Deep Learning for Natural Language Processing from 2022**, especially regarding newer libraries, but no direct follow-up was provided.
- **Struggles with AzureChatOpenAI Configurations**: `@smartge3k` expressed difficulty with `AzureChatOpenAI` configurations, receiving a 'DeploymentNotFound' error when using it with `ConversationalRetrievalChain`, and discussion ensued without a definitive resolution being provided.

**Links mentioned**:

- [Zero Prime @ Data Council '24](https://zeroprime.vc/ai-launchpad): Join Zero Prime Ventures @ Data Council Austin 2024 for a unique chance to showcase your AI startup. Apply now for exposure to top investors and elite founders.
- [Agent Types | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/agents/agent_types/.): This categorizes all the available agents along a few dimensions.
- [SparkLLM Chat | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/chat/sparkllm): SparkLLM chat models API by iFlyTek. For more information, see iFlyTek
- [Pricing](https://www.langchain.com/pricing): Plans for teams of any size.
- [Tweet from Austin Vance (@austinbv)](https://x.com/austinbv/status/1760320228725309951?s=46): 🚨 New Tutorial 🚨 The finale of my &#34;Chat with your PDF&#34; build RAG from scratch with @LangChainAI tutorial!  In Part 4 we - Use LangSmith for EVERYTHING - Implement Multi Query to increase ret...



---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1209451165507391500) (3 messages): 

- **Seeking a Superior PDF Parser**: `@dejoma` inquired if anyone has constructed an enhanced PDF parser using **PDFMinerPDFasHTMLLoader / BeautifulSoup** and showed interest in sharing it, noting that the existing example in the documents is good but with a week of dedication, one can create something exceptional.

- **Idea to Newsletter and Tweets Conversion with LangChain**: `@merkle` shared a [link](https://x.com/michaeldaigler_/status/1759962223236038781?s=20) demonstrating how **LangChain's langgraph agent setup** can be used to generate a newsletter draft and tweets from a single topic. *The one-man media company is closer than you think*, says the tweet by `@michaeldaigler_`.

- **Teaser for an Enigmatic Project**: `@pk_penguin` posted a cryptic message, "thought?" and invited direct messages from those interested in trying out what is presumably a new project or tool.

**Links mentioned**:

[Tweet from Michael Daigler (@michaeldaigler_)](https://x.com/michaeldaigler_/status/1759962223236038781?s=20): The one-man media company is closer than you think.  A newsletter draft and tweets from a single topic with AI.  Here&#39;s how it works:

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1209490461429862450) (4 messages): 

- **Corrective RAG Explored**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=eT31Cyaziy0) titled "Corrective RAG using LangGraph," which discusses the enhancement of RAG through self-reflection and the correction of poor quality retrieval or generations.
- **Building Persistent Memory Chatbots**: `@kulaone` provided a link to an article titled "Beyond Live Sessions: Building Persistent Memory Chatbots with LangChain, Gemini Pro, and Firebase," explaining the significance of persistent memory in conversational AI. [Read the full article on Medium](https://medium.com/@zps270/beyond-live-sessions-building-persistent-memory-chatbots-with-langchain-gemini-pro-and-firebase-19d6f84e21d3).
- **Troubleshooting Spark API Issue**: `@syedmujeeb` is facing an 'AppIdNoAuthError' when using the Spark API from iFlytek. They referenced the [LangChain documentation for SparkLLM](https://python.langchain.com/docs/integrations/chat/sparkllm) and asked for advice on the matter.
- **Self RAG with LangGraph Discussed**: `@pradeep1148` posted another [YouTube video](https://www.youtube.com/watch?v=Eb7QF1nDWGU) titled "Self RAG using LangGraph," which covers the implementation of self-reflection to improve RAG's quality, suggesting that several recent papers have focused on this theme.

**Links mentioned**:

- [Beyond Live Sessions: Building Persistent Memory Chatbots with LangChain, Gemini Pro, and Firebase](https://medium.com/@zps270/beyond-live-sessions-building-persistent-memory-chatbots-with-langchain-gemini-pro-and-firebase-19d6f84e21d3): In the rapidly evolving landscape of artificial intelligence, the evolution from simple scripted chatbots to today’s advanced…
- [SparkLLM Chat | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/chat/sparkllm): SparkLLM chat models API by iFlyTek. For more information, see [iFlyTek
- [Self RAG using LangGraph](https://www.youtube.com/watch?v=Eb7QF1nDWGU): Self-reflection can enhance RAG, enabling correction of poor quality retrieval or generations.Several recent papers focus on this theme, but implementing the...
- [Corrective RAG using LangGraph](https://www.youtube.com/watch?v=eT31Cyaziy0): Corrective RAGSelf-reflection can enhance RAG, enabling correction of poor quality retrieval or generations.Several recent papers focus on this theme, but im...

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1209493981847617656) (18 messages🔥): 

- **Gemma Models Spark Curiosity**: `@johannhartmann` requested details about the training strategy and base model for certain interesting AI models. In response, `@sebastian.bodza` shared a [link to Google's open-source models](https://www.kaggle.com/models/google/gemma), though `@philipmay` followed up wondering about the non-English language capabilities of these models.
- **Gemma's Language Limitations Discussed**: `@bjoernp` remarked that Google's Gemma models, introduced earlier in the discussion, seem to be primarily English-focused and provided a [Hugging Face link](https://huggingface.co/google/gemma-7b-it) detailing the Gemma 7B instruct version. He also noted Gemma's impressively large 256k vocabulary size.
- **Aleph Alpha Model Updates Examined**: `@sebastian.bodza` mentioned updates from Aleph Alpha on their models, prompting a discussion about their quality. `@devnull0` pointed out that Andreas Köpf now works with Aleph Alpha, potentially signaling upcoming improvements, and shared a [changelog from Aleph Alpha's documentation](https://docs.aleph-alpha.com/changelog/), which details recent updates and features to their models.
- **Skepticism Over Aleph Alpha's Enhancements**: `@_jp1_` criticized Aleph Alpha for lacking substantial updates in their models despite being a large company, referring to a lack of benchmarks and examples in the first blog post after almost a year. Meanwhile, `@sebastian.bodza` and `@devnull0` discussed the apparent absence of instruction tuning in the new models, with discrepancies in how interaction with these models is managed. 
- **Testing Reveals German Language Struggles**: After testing, `@_jp1_` stated that the instruct version of Gemma is particularly weak in German, which was further confirmed by `@bjoernp` sharing results from the `lm_eval` test on German hellaswag that showed performance just above random chance.

**Links mentioned**:

- [Gemma](https://www.kaggle.com/models/google/gemma): Gemma is a family of lightweight, open models built from the research and technology that Google used to create the Gemini models.
- [HuggingChat](https://huggingface.co/chat): Making the community's best AI chat models available to everyone.
- [google/gemma-7b-it · Hugging Face](https://huggingface.co/google/gemma-7b-it): no description found
- [Blog | Aleph Alpha API](https://docs.aleph-alpha.com/changelog/): Blog

  

---


### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1209422120807964682) (9 messages🔥): 

- **GPU Budget Concerns Lead to Hunt for Free Benchmarks**: `@johannhartmann` joked about the GPU being poor, leading `@bjoernp` to quip about the request for a free benchmark.
- **Seeking Speed in Benchmark Tests**: `@johannhartmann` expressed frustration with the long runtime of `lm-evaluation-harness-de` and inquired about tips to speed it up, suspecting something might be wrong.
- **The Need for Speed: vLLM to the Rescue**: `@bjoernp` suggested that using `vLLM` can significantly expedite the testing process and offered to investigate why it wasn't speeding things up as expected.
- **Outdated Branch Causes Slowdown**: `@johannhartmann` realized that the slowness might be due to using an outdated branch that didn't integrate with `vLLM`.
- **Upcoming Updates to Include vLLM Support**: `@bjoernp` acknowledged the issue with the outdated branch and committed to updating the versions to support the current main branch but noted it may take a few days.
  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1209482949662081085) (7 messages): 

- **Waitlist or Google Rep Required for AI Access**: `@wenquai` mentioned that to gain access to a particular AI service, one must either join a waitlist or contact a Google Cloud representative.
- **AI Access Buzz Confirmed on Twitter**: `@res6969` confirmed seeing additional evidence on Twitter that rumors about AI service access were true, but they're still waiting for their own access.
- **Concerns Over AI Accuracy and Hallucinations**: `@res6969` hinted at public feedback about an AI's performance issues, noting it's "not super accurate and has hallucination issues."
- **The User Echo Chamber**: Following `@res6969`'s comment on AI accuracy concerns, `@thebaghdaddy` responded with a short acknowledgement: "many such cases".
- **Seeking CRM Solutions for AI Business**: `@frandecam` is looking for recommendations for CRM services suitable for an AI business, considering options like Salesforce, Zoho, Hubspot, or Pipedrive to replace current Google Sheets usage.
- **Anti-Endorsement for Salesforce**: In response to `@frandecam`'s query on CRMs, `@res6969` cautioned against choosing Salesforce, suggesting it would be a significant error for the company.
  

---


### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (1 messages): 

potrock: https://blog.google/technology/developers/gemma-open-models/
  

---


### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/1209869748431491204) (1 messages): 

- **ContrastiveLoss Wins the Favor**: User `@dartpain` expressed a preference for **ContrastiveLoss** when tuning embeddings, stating it's impactful for making adjustments. Additionally, they mentioned favoring **MultipleNegativesRankingLoss**.
  

---



### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1210013169993383956) (1 messages): 

- **Neuralink Interview Insights Wanted**: User `@xilo0` is seeking advice on how to answer the "evidence of exceptional ability" question for a late-stage interview at Neuralink. They have a list of projects to showcase but are looking for input on what might impress a Musk-run company.
  

---


### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1209490096672350278) (4 messages): 

- **Exploring Corrective RAG**: `@pradeep1148` shared a [YouTube video titled "Corrective RAG using LangGraph"](https://www.youtube.com/watch?v=eT31Cyaziy0), which discusses how self-reflection can improve retrieval-augmented generation (RAG) by correcting low-quality retrievals or generations.
- **Diving into Self RAG**: Another [YouTube video shared by @pradeep1148](https://www.youtube.com/watch?v=Eb7QF1nDWGU) titled "Self RAG using LangGraph" explores the concept of self-reflection to enhance RAG.
- **BitDelta: Evaluating the Value of Fine-Tuning**: `@pradeep1148` also posted a [YouTube video](https://www.youtube.com/watch?v=T_dYzuv4N70) titled "BitDelta: Your Fine-Tune May Only Be Worth One Bit" that evaluates the actual worth of fine-tuning Large Language Models (LLMs) after pre-training on extensive datasets.
- **Simple Appreciation**: `@sabertoaster` responded with a brief "nice" to the shared content, indicating appreciation or interest.

**Links mentioned**:

- [Self RAG using LangGraph](https://www.youtube.com/watch?v=Eb7QF1nDWGU): Self-reflection can enhance RAG, enabling correction of poor quality retrieval or generations.Several recent papers focus on this theme, but implementing the...
- [Corrective RAG using LangGraph](https://www.youtube.com/watch?v=eT31Cyaziy0): Corrective RAGSelf-reflection can enhance RAG, enabling correction of poor quality retrieval or generations.Several recent papers focus on this theme, but im...
- [BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://www.youtube.com/watch?v=T_dYzuv4N70): Large Language Models (LLMs) are typically trained in two phases: pre-training on large internet-scale datasets, and fine-tuning for downstream tasks. Given ...

  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1209675131962785862) (4 messages): 

- **Recognition in the AI Community**: `@swyxio` expresses gratitude, possibly in response to being acknowledged by peers or receiving an accolade within the AI field.
- **Call for Inclusion of Grassroots Contributors**: `@tokenbender` suggests that a list under discussion is too corporate and should include individuals who are active on the ground and share their work publicly, recommending `@abacaj` on Twitter as a notable addition.
- **Looking for Token Info**: `@scopexbt` inquires about the existence of a token associated with the group, indicating they were unable to find related information.
  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1209925830365683773) (2 messages): 

- **Google Unveils Gemini Pro 1.5**: User `@derekpwillis` introduced the latest from Google, featuring the [documentation for GEMMA](https://ai.google.dev/gemma/docs).
- **Simon Explores Gemini Pro 1.5**: `@simonw` shared his insights on [Gemini Pro 1.5](https://simonwillison.net/2024/Feb/21/gemini-pro-video/), highlighting its 1 million token context and the groundbreaking feature of using video as an input. Simon has been testing the capabilities through the [Google AI Studio](https://aistudio.google.com/app/prompts/new_chat).

**Links mentioned**:

- [The killer app of Gemini Pro 1.5 is video](https://simonwillison.net/2024/Feb/21/gemini-pro-video/): Last week Google introduced Gemini Pro 1.5, an enormous upgrade to their Gemini series of AI models. Gemini Pro 1.5 has a 1,000,000 token context size. This is huge—previously that …
- [no title found](https://ai.google.dev/gemma/docs): no description found

  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1209517664955928606) (2 messages): 

- **GLIBC Version Issues in GitHub Codespaces**: `@derekpwillis` encountered a problem trying to run llm in a GitHub codespace due to an **OSError** related to `GLIBC_2.32` not found, required by a file inside the `llmodel_DO_NOT_MODIFY` directory. They considered trying to install version 2.32 but sought suggestions from the group.
- **Appreciation for Warning Labels**: `@derekpwillis` humorously expressed their fondness for the directory named `llmodel_DO_NOT_MODIFY`.
  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1209535961478725673) (3 messages): 

- **Groq LPU's Impressive Speed**: User `@juanreds` was pleasantly surprised by the prompt response times of the **Groq LPU** and shared a [link for others to try](https://groq.com/).

- **Weekend Plans Coordination and Sponsor Leads**: `@yikesawjeez` mentioned a scheduling conflict for the morning but requested a **recording of the meeting**. They also informed `@705561973571452938` about one confirmed sponsor and potentially three more for upcoming weekend activities.

- **Groq LPU Enthusiast Agrees**: User `._z` expressed excitement, finding the Groq LPU to be "super cool" in response to `@juanreds`'s earlier comment on the technology's performance.

**Links mentioned**:

[Groq](https://groq.com/): no description found

  

---


### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1209913190599622706) (1 messages): 

- **Gemini 1.5 Discussion Scheduled**: `@shashank.f1` announces an upcoming live discussion on **Gemini 1.5** and invites everyone to join. The link to the event is provided: [Join the Gemini 1.5 discussion](https://discord.gg/F4FfcQw3?event=1209440306404139008).
- **Exploring Semantic Knowledge in Audio with A-JEPA**: The last session's highlight was the **A-JEPA AI model**, which focuses on extracting semantic knowledge from `.wav` or `.mp3` files or audio spectrograms. A YouTube video recap of the discussion is shared: [Watch the A-JEPA AI model discussion](https://youtu.be/FgcN62LFzIU).

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/F4FfcQw3?event=1209440306404139008): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [A-JEPA AI model: Unlock semantic knowledge from .wav / .mp3 file or audio spectrograms](https://youtu.be/FgcN62LFzIU): 🌟 Unlock the Power of AI Learning from Audio ! 🔊 Watch a deep dive discussion on the A-JEPA approach with Oliver, Nevil, Ojasvita, Shashank, Srikanth and N...

  

---
id: c385ed86-3c44-4026-a4da-380d75752f43
title: AI gets Memory
date: '2024-02-15T00:47:59.492420Z'
original_slug: ainews-ai-gets-memory
description: >-
  **AI Discords** analysis covered **20 guilds**, **312 channels**, and **6901
  messages**. The report highlights the divergence of RAG style operations for
  context and memory, with implementations like **MemGPT** rolling out in
  **ChatGPT** and **LangChain**. The **TheBloke Discord** discussed
  **open-source large language models** such as the **Large World Model** with
  contexts up to **1 million tokens**, and the **Cohere aya model** supporting
  **101 languages**. Roleplay-focused models like **MiquMaid-v2-70B** were noted
  for performance improvements with enhanced hardware. Finetuning techniques
  like **Sequential Fine-Tuning (SFT)** and **Direct Preference Optimization
  (DPO)** were explained, with tools like **Unsloth AI's apply_chat_template**
  preferred over Alpaca. Integration of JavaScript and Python via **JSPyBridge**
  in the **SillyTavern** project was also discussed. Training challenges with
  **Mixtral 8x7b qlora** versus **Mistral 7b** were noted. The **LM Studio
  Discord** focused on hardware limitations affecting large model loading,
  medical LLMs like **medAlpaca**, and hardware discussions around GPU upgrades
  and overclocking. Anticipation for **IQ3_XSS** 1.5 bit quantization support in
  LM Studio was expressed.
companies:
  - openai
  - langchain
  - thebloke
  - cohere
  - unsloth-ai
  - mistral-ai
  - microsoft
models:
  - miqumaid-v2-70b
  - mixtral-8x7b-qlora
  - mistral-7b
  - phi-2
  - medalpaca
  - aya
topics:
  - rag
  - memory-modeling
  - context-windows
  - open-source
  - finetuning
  - sequential-fine-tuning
  - direct-preference-optimization
  - rlhf
  - ppo
  - javascript-python-integration
  - hardware-optimization
  - gpu-overclocking
  - quantization
  - model-training
  - large-context
  - multilinguality
people:
  - joanne-jang
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/11-12/2024. We checked **20** guilds, **312** channels, and **6901** messages for you. Estimated reading time saved (at 200wpm): **589 minutes**.

We have long contended that the RAG style operations have been used for context (knowledge base, facts about the world) and memory (running list of facts about you) will diverge. The leading implementation was MemGPT and now it seems to have rolled out in both [ChatGPT](https://twitter.com/OpenAI/status/1757469997742666052) (with a [weirdly roon-y tweet](https://x.com/ChatGPTapp/status/1757546067951026401?s=20). more details from [Joanne Jang](https://twitter.com/joannejang/status/1757470618264429008?t=90xQ8sGy63D2OtiaoGJuww&utm_source=ainews&utm_medium=email)) and [LangChain](https://www.loom.com/share/63a9696036c74765a9f9ecae06336aa5?utm_source=ainews&utm_medium=email).

OpenAI:

 ![image.png](https://assets.buttondown.email/images/2478d063-ab0b-4268-8082-fee260f8e601.png?w=960&fit=max) 

LangChain:

 ![image.png](https://assets.buttondown.email/images/039aaa0e-9cdc-442e-8496-954d50efbd8a.png?w=960&fit=max) 

In some sense this is just a crossing over of something the LMstudio/Sillytavern roleplay people have had for a while now. Expectation is that it will mildly improve UX but not lead to a big wow moment since the memory modeling is quite crude at the moment, not humanlike, and subject to context limits.



---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Unbounded Textual Contexts**: Engineers are exploring new **open-source large language models** like the [Large World Model](https://github.com/LargeWorldModel/LWM), which boasts coherence with contexts up to 1 million tokens. Discussions include language support, as in Cohere's **`aya` model**, which covers 101 languages, and challenges working with **jax-based tools** during model installations.
  
- **Nurturing Erotically Programmed Role Play**: The community is dissecting performances of re-quantized **Miqu models** like **MiquMaid-v2-70B**, attuned for Erotic Role Play (ERP). Emphasis was on the impact of enhanced hardware, with a jump from 0.7t/s to 2.1t/s in tokens per second while using 12GB VRAM GPUs.

- **Instruct, Optimize, Repeat**: Finetuning techniques explained include using **Sequential Fine-Tuning (SFT)** followed by **Direct Preference Optimization (DPO)** as improved RLHF/PPO, detailed on [page 6 of a paper](https://arxiv.org/pdf/2401.04088.pdf). **Unsloth AI’s `apply_chat_template`** is touted over Alpaca to train LLMs for multi-turn conversations.

- **JavaScript Meets Python in AI Development**: Experimentation with [JSPyBridge](https://github.com/extremeheat/JSPyBridge) led to successful bridging of JavaScript and Python in expanding the **SillyTavern** project. This included addressing Windows-specific errors, like `cpu_embedding=True` to circumvent access violation issues and integrating Python classes asynchronously into JavaScript code.

- **Confounding Losses in Model Training**: An engineer observed an *unexplained variance* in training loss when finetuning **Mixtral 8x7b qlora**, resulting in higher losses compared to **Mistral 7b** despite similar datasets and hyperparameters. The matter remains open for community input or similar experiences.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Large Models Crying for RAM**: Users like `@nonm3` and `@theoverseerjackbright` battled **errors loading large models** in LM Studio due to limited RAM and VRAM. Suggestions included trying smaller model quants, and some faced GPU detection issues with LM Studio, prompting restart crashes.

- **MedAlpaca Heads to the Clinic**: Discussions on medical LLMs saw [medAlpaca](https://github.com/kbressem/medAlpaca), a fine-tuned model for medical question answering, as a promising addition to `@pepito92i`'s medical project. Microsoft's **phi-2** model's absence from LM Studio was noted, with the possibility of it being converted to .gguf format by user TheBloke to use with llama.cpp.

- **GPU Matchmaking and Overclocking**: Hardware enthusiasts like `@luxaplexx` questioned NVLink's role in memory cycling, ultimately suspecting cards like the 960 might not be NVLinked. Users discussed GPU upgrades for better performance with models, considering options like the RTX 3060 12GB. Others like `@alastair9776` and `@rugg0064` weighed the benefits and risks of overclocking for faster token generation.

- **Quant Leap Forward**: Eager anticipation for **IQ3_XSS** support in LM Studio, with users like `@n8programs` and `@yagilb` expecting it in the next update. A GitHub pull request reflected the community's excitement over upcoming 1.5 bit quantization. Meanwhile, preparations were suggested for downloading forthcoming, as-yet-unsupported models like **IQ3_XSS**.

- **Beta Release Relief**: `@rafalsebastian` ran into a stumbling block running LMstudio on CPUs with only AVX support. `@heyitsyorkie` provided hope by directing to the **0.2.10 AVX beta release for Windows** that enables compatibility, while still recommending an upgrade to AVX2 for optimal performance and offered a helpful [link](https://lmstudio.ai/beta-releases.html).

- **Multi-Model Management Mystery**: `@alluring_seahorse_04960` sought advice on running dual models simultaneously on one machine to avoid repetition errors, using a Conda environment but steering clear of VMs. The nature of the repetition errors in question was humorously prodded by `@wolfspyre`, awaiting further details.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Magvit V2 Sparks Interest and Debate**: Engineers delved into the technicalities of reproducing the [Magvit V2 model](https://github.com/lucidrains/magvit2-pytorch), with discussions focusing on appropriate datasets, parameters for video compression and understanding, and the mention of experiments on the **lfq side of Magvit2**. The community also saw a surge in interest around **MAGVIT**, likely due to influencer mentions.

- **Scrutinizing Stable Cascade's Efficacy**: Stability AI's **Stable Cascade** model spurred intense conversations regarding its high VRAM requirements, optimization issues, and erroneous inference time graphs. Technical issues reported included challenges with text clarity in images and the inability to run models in float16, alongside performance evaluations on GPUs like the 3090.

- **Legal Frays in AI-Generated Imagery**: The community engaged in a heated discussion about copyrights and the legality of AI-generated images, highlighting a [TorrentFreak article](https://torrentfreak.com/court-dismisses-authors-copyright-infringement-claims-against-openai-240213/) about a court dismissing authors’ copyright infringement claims against OpenAI.

- **Ethical Conundrums with AI and Adult Content**: The conversation shifted to the role of adult content in driving technological progress, with some participants recognizing the historical pattern while others doubted its constructive impact on AI. Topics included the [rise of non-consensual deepfake pornography](https://bc.ctvnews.ca/ai-brings-deepfake-pornography-to-the-masses-as-canadian-laws-play-catch-up-1.6754498), its market dynamics, and the potential ethical pitfalls plaguing the AI community.

- **Calls for Higher AI Image Standards**: Discussions included technical insights into improving AI image generation, such as the viability of VAE encoder training. Members also reflected on the community's photorealism standards, expressing the need for better quality in AI-generated images.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Checksum Hunting Season Open**: `@paganpegasus` provided checksums for The Pile zst shards and pointed to [EleutherAI’s hashes](https://www.eleuther.ai/hashes) and the [Discord pins](https://discord.com/channels/729741769192767510/729741769738158194/1177260719847264266).

- **Image Content Classification Tools Discussed**: OwlViT and CLIP models were recommended as tools for discerning the content of images and the concept of "nothing" in imagery was discussed due to an inquiry by `@everlasting_gomjabbar`.

- **Paper Review in Collaborative Spirit**: A user received appreciative feedback on a manuscript titled "Don't think about the paper," with the EleutherAI Discord community being credited in the paper's acknowledgements.

- **Cloud Computing Resources Examined**: GCP and Colab surfaced as favorable cloud resources for NLP classification model training, with discussions encompassing cost-benefit analyses of platforms like runpod and vast.ai.

- **Research Computing Power Up for Grabs**: EleutherAI's computational resources were said to be available for collaboration on a semi-custom LLM project, with the caveat of having a clear research agenda and collaborative value proposition.

- **Semantic Scholar's Linking Logic Revealed**: Arxiv papers are automatically linked to authors on Semantic Scholar, with room for manual corrections to ensure accuracy.

- **Fractal Fun with Neural Training Parameters**: `@jbustter` shared fractals created from neural network hyperparameters, highlighting a blog by Jascha Sohl-Dickstein that correlates fractals with training convergence/divergence.

- **A Deep Dive into Data Presentation for ML**: A discussion was sparked concerning active learning and methods for models to choose their own data presentation sequence.

- **Enriching Encoder-Decoder Models with Unsupervised Data**: Strategies to employ unsupervised datasets effectively in encoder-decoder models were discussed.

- **New NLP Robustness Method Flies Off the Press**: A paper focusing on test-time augmentation (TTA) to enhance text classifiers' robustness was published, with the author thanking the community for support.

- **The Quest for Interpretability Insight**: `@jaimerv` asked for updated resources on interpretability methods beyond the standard Representation Engineering paper.

- **Summoning Collaborators for Hallucination Leaderboard**: A call for contributions to a [hallucinations leaderboard](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/tree/main/src/backend/tasks) was made, requesting assistance with tasks, datasets, metrics, and result evaluations.

- **Aligning Pythia with Practice**: Concerns were aired about potential misalignments between training batches and checkpoints in the 2.8b size Pythia deduped suite, with follow-up discussions suggesting opportunities for a publication on Pythia's reliability.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

**LlamaIndex v0.10 Marks Major Milestone**: **LlamaIndex v0.10** has been released, presenting notable advancements including a new `llama-index-core` package and PyPi packages for every integration/template. Detailed information on migration is accessible through their comprehensive [blog post](https://blog.llamaindex.ai/llamaindex-v0-10-838e735948f8) and [documentation](https://docs.llamaindex.ai/en/stable/getting_started/installation.html).

**Webinar on No-Code RAG with LlamaIndex**: A webinar demonstrating the creation of no-code Retrieve and Generate (RAG) apps using **LlamaIndex.TS** is set up with Flowise co-founder Henry Heng. Registration for the Friday event is available [here](https://lu.ma/ubm3jg3k).

**Troubleshooting LlamaIndex**: Engineers faced challenges with migration following LlamaIndex's update and were pointed to a [Notion migration guide](https://pretty-sodium-5e0.notion.site/v0-10-0-Migration-Guide-6ede431dcb8841b09ea171e7f133bd77) for assistance. Furthermore, for configuration queries like `chunk_size` post-ServiceContext depreciation, engineers are advised to refer to the new [`Settings`](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration.html) documentation and relevant [LlamaIndex GitHub resources](https://github.com/run-llama/llama_index/blob/main/llama-index-legacy/llama_index/legacy/vector_stores/mongodb.py#L160-L183).

**RAG App Building with Dewy Tremendously Simplified**: A comprehensive guide to building a full-stack RAG app using [NextJS](https://nextjs.org/), [OpenAI](https://platform.openai.com/), and the open-source knowledge base [Dewy](https://dewykb.github.io/) has been shared. The tutorial is aimed at grounding language models in precise, reliable data and can be studied in detail [here](https://dewykb.github.io/blog/rag-app-with-nextjs-openai-and-dewy/).

**Handling Document Complexity and Enhancing Enterprise with LlamaIndex**: Users engaged in discussions about filtering complex documents and integrating LlamaIndex to enhance enterprise efficiency with tools such as Slack, Jira, and GDrive. Also, creating multiple agents for merging different document sources was considered, referencing the possibility of using traditional indexing techniques instead of high-cost LLMs for dynamic filtering.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Hugging Face Accelerates with Message API**: Hugging Face launched a new **Message API** compatible with OpenAI, aimed at streamlining use of inference endpoints and text generation services with client libraries. They've also advanced their offerings with new releases like Datatrove on PyPI, Gradio 4.18.0, and tools like **Nanotron** and **Accelerate 0.27.0** for 3D parallelism training. Additional partnerships and resources, such as a Codecademy AI course and a blog post on **SegMoE**, support the continuous learning and innovation in their community.

- **Search Engine Woes and Hosting Queries in Focus**: Technical discussions spotlighted the difficulties in creating search engines with mentions of approaches like **TF-IDF** and **BM25**, and the use of **spaCy** for Part of Speech tagging. Other conversations pivoted to queries about hosting custom models and serverless inferencing solutions, as well as the practicality of running 100B+ parameter models on enthusiast-level hardware.

- **Template Talk and Model Deployment Discussions**: Users addressed the need for a simple chatbot development prototype capable of database interaction and email API integration, featuring resources like Microsoft's **AutoGen** on GitHub and the potential of **AutoGen Studio**. Challenges around deploying finetuned machine learning models such as **mistarl_7b_gptq** for fast inferences were raised, with emphasis on choosing the right platforms or libraries for the task.

- **Glimpse into Creator Innovations**: Members of the community showcased their creative projects, including GIMP 3.0 plugins interfacing with **Automatic1111**, development of an automated image tagging model for diffusion tasks, and updates to tools like [PanoramAI.xyz](https://www.panoramai.xyz/) introducing a "remix" mode for image transformations. Excitement built around AI-applications in fashion design as well, demonstrating the breadth of applications being pursued.

- **Analyzing S4 and Advancing NLP**: The community shared their insights into the S4 architecture ideal for long-range sequences and sought clarity on its implementation. The paper on **LangTest** got introduced, which offers testing and augmenting capabilities for NLP models. Topics extended to extracting language identifiers from models like **XLM-RoBERTa** and converting natural language into formal algebraic expressions.

- **Enthusiasm for Diffusion and Emerging Models**: Conversations sparked around facilitating **multi-GPU** training for diffusion model fine-tuning, with mentions of scripts such as `train_text_to_image.py`. The successful deployment of models like **mistarl_7b_gptq** for fast inference, and effective text generation with **stable cascade** were discussed. The buzz was palpable around the teased development of a new **terminus model**.

- **Complications in Computer Vision Explored**: The channel delved into challenges like hierarchical image classification, with resource suggestions including an [ECCV22 paper](https://arxiv.org/pdf/2207.04873.pdf) on the same. Members discussed requirements for Gaussian splats, industry-grade image retrieval systems and sought collaboration on multimodal projects.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **LongCorpus Dataset Unveiled for Pre-Training**: The new [LongCorpus-2.5B dataset](https://huggingface.co/datasets/DAMO-NLP-SG/LongCorpus-2.5B) is released, featuring 2.5 billion tokens from various domains, specifically curated for long-context continual pre-training and designed to minimize n-gram similarity with training sets.

- **Coherence Preserved in Scaling Models**: Scaling with 'self-extend' is considered superior over 'rope scaling' for maintaining coherence, as indicated by the implementation in llama.cpp, and offers the benefit of requiring no setup, fine-tuning, or additional parameters.

- **Persistence and Resistance in AI Agents and Models**: LangGraph agents can persist their state across interactions, as shown in a [YouTube demonstration](https://www.youtube.com/watch?v=fHroHoc26RI), while the Gemini model shows resistance, with its refusal tendencies prompting comparisons unfavorable to GPT-4.

- **Multimodal AI Breakthrough with Reka Flash**: **Reka Flash**, a new **21B fast multimodal language model**, is introduced and now available in public beta, promising to measure up to major models like Gemini Pro and GPT-3.5. The initiative can be followed on [Reka AI's announcement page](https://reka.ai/reka-flash-an-efficient-and-capable-multimodal-language-model/).

- **CUDA Pains and WAVeform Gains in AI Research**: The ZLUDA project aimed to run CUDA on AMD GPUs can no longer be considered active, and a fresh perspective in AI research proposed in an [arXiv paper](https://arxiv.org/abs/2210.01989), suggests wavelet transforms could enhance Transformers by addressing both positional and frequency details efficiently.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Newbies Get Model Recommendations**: Participants recommended **instruct models** for chat-GPT-like interactions to newcomer `@nana.wav`, with the clearer instruction-following focus as opposed to the more general autocompletion capabilities of other models.

- **RAG Setup and Model Debates Heat Up**: A guide on integrating **Mistral with RAG** was shared, while the effectiveness of **LangChain vs. LlamaIndex** was debated; separately, **DSPy** was touted for leveraging LLMs for programming rather than chatting, adorned with a supportive Twitter link.

- **Deployment Dilemmas and Solutions**: Docker deployment via **ollama** or **vllm** projects was suggested, while others discussed API alternatives and faced cloud quota barriers; meanwhile, success stories involved deploying **Mixtral on HuggingFace** despite the hiccups with **AWQ quantization**.

- **Fine-Tuning Finesse and RAG Revelations**: Users discussed fine-tuning vs. RAG with insights into LLM base knowledge importance; guidance was given on input data structuring for LLM output enhancement and queries about prompt versioning tools surfaced.

- **Humans in Tech and AI Seek Touchpoints**: French librarian (`@maeelk`) sought internship opportunities in psychology and AI; the cost of innovatively building audio-inclusive S2S models sparked discussions around budget constraints and investment needs.

- **Technical Troubles and Support Suggestions**: `@ingohamm` faces hurdles with TypingMind's API key and a suggestion was made to contact **support@mistral.ai** for assistance with API and subscription issues.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Perplexity AI Outshines Rivals in Complex Query Handling**: `@tbrams` tested Perplexity AI with a difficult question from the "Gemini" paper and found it outperformed Google's Gemini service and OpenAI, answering more quickly. The test results from Perplexity AI are documented [here](https://www.perplexity.ai/search/Write-code-to-7yAwNS4DTAyB905rHG04eA?s=u).

- **Perplexity's Potential in API Customization Highlighted**: The PPLX API allows for custom search queries using parameters like `"site:reddit.com OR site:youtube.com"`, as mentioned by `@me.lk`. However, several users have encountered issues with the API such as performance hiccups (`@andrewgazelka`) and nonsensical responses (`@myadmingushwork_52332`).

- **Perplexity AI Subscription and Renewal Queries Addressed**: Users are seeking details on trial subscriptions and renewal processes for Pro subscriptions, with inquiries about token refresh rates also surfacing. There is currently no early access program for new Perplexity features as confirmed by `@icelavaman`.

- **Promising Enhancements and Community Collaborations**: Perplexity AI is receiving community praise for tools like the pplx shortcut action (`@twodogseeds`). Meanwhile, `@ok.alex` is encouraging a community-driven effort to contribute to an alternative feed/newsletter [Alt-D-Feed](https://www.perplexity.ai/collections/Alt-D-Feed-x.dZp0_3RAyKoTWkhJW_DA).

- **Seeking Direct Support Channel for Sensitive Data Issues**: A user (`@kitsuiwebster`) has expressed the need for direct assistance with a sensitive company data issue, avoiding public disclosure while lacking response from support channels.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **ChatGPT Remembers Your Favorite Color**: OpenAI announced a new [memory feature for ChatGPT](https://openai.com/blog/memory-and-new-controls-for-chatgpt), rolling out to select users, enabling ChatGPT to remember user preferences and details over conversations for a more personalized experience. Users can control what ChatGPT remembers and can switch off this feature.

- **AI-Assistants in Creative Process Paid Talks**: A UK researcher, `@noodles7584`, is looking to compensate community members for a 30-minute discussion on AI use in creative workflows.

- **Performance Quirks in GPT Variants**: The community reported fluctuations in GPT-4's task handling, and Abacus.AI's Smaug-72B was noted for outperforming GPT-3.5, while ChatGPT-4 seems hesitant to generate full code snippets.

- **Fine-Tuning AI to Watch Videos? Not Yet**: Discussion in [#gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1206512221450936390) clarified that while GPT can describe images from a video with its vision capabilities, it cannot yet be fine-tuned for video-specific knowledge or tasks.

- **Exploring and Perfecting Prompt Engineering**: Good prompt engineering was highlighted as involving clear instructions and precision, with a focus on fostering simple storytelling in text-based AI adventures and recognizing differences between prompt engineering and API infrastructure development.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Axolotl Embraces MPS, Thanks to GitHub Heroes**: Maxime added MPS support in the axolotl project via [pull request #1264](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1264), referencing the importance of a [PyTorch pull request #99272](https://github.com/pytorch/pytorch/pull/99272). Clarification on contributor identities highlighted the importance of collective recognition in open source.

- **Chat In The Time Of Datasets**: The **MessagesList** standard for chat datasets proposed by `@dctanner` aims for cross-compatibility and is under [discussion](https://huggingface.co/posts/dctanner/975913831192894). The format might include conversation pairs, greetings, and assistant-initiated closures, with challenges noted in JSON-schema validation.

- **Axolotl Tokenized Right, Check the Debug Flag**: Users are troubleshooting tokenization within axolotl, with advice to inspect the tokenizer configs and a recommendation to use a [debug flag](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main#preprocess-dataset) for verification.

- **Model Query Woes and Training Queries Grow**: Queries about improving model's multilingual capabilities, LoRA adapter inferencing, and model parallelism were discussed, with solutions ranging from pre-training needs to updates in transformers and DeepSpeed Zero 3 configs for better functionality.

- **Fine-tune or Re-train? Duplicate Data's Pain**: The impact of training data overlap and finetuning practices were questioned, highlighting concerns about reusing text that a model may have encountered during pretraining.

- **RunPod Image on Vast.AI, A Smooth Sail!**: The **Axoltl RunPod image** was reported by `@dreamgen` to work seamlessly on Vast.AI, underscoring the inter-operability with cloud infrastructure providers.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **LangChain Unveils Memory Journaling App**: `@hwchase17` introduced a new journaling app featuring **LangChain memory module**, inviting feedback for the early version akin to **OpenAI's ChatGPT with memory** feature. Try and give feedback using this [journal app](https://journal.langchain.com/) and watch the [intro video](https://www.loom.com/share/63a9696036c74765a9f9ecae06336aa5).

- **LangChain Community Tackles Diverse Technical Challenges**: Topics covered included the possibility of **LangChain's Android integration**, pre-processing benefits for efficient embeddings, the search for a capable **PDF parser**, and calls for improved documentation structure. Additionally, a user faced dependency issues while updating Pinecone Database to v2 with LangChain, which was promptly addressed.

- **Scaling and Integration Enquiries in Langserve Channel**: Discussions included questions about scaling **Langserve** and using **Langsmith** for deployment. There was a query about exposing a chain from a NodeJS app and an unaddressed issue regarding disabling intermediate steps in the playground. Connection issues with an **OpenAI API** call from a **k8s cluster-based app** were also described.

- **Dewy RAG Application with NextJS and OpenAI Guide Shared**: `@kerinin` contributed a guide exploring a **full-stack RAG application**, utilizing [NextJS](https://nextjs.org/), [OpenAI API](https://platform.openai.com/), and [Dewy](https://dewykb.github.io/), focusing on reducing hallucinations and improving model response accuracy. The full guide is available [here](https://dewykb.github.io/blog/rag-app-with-nextjs-openai-and-dewy/).

- **Quest for a Functional PDF Parser and Custom Calculator**: Within the tutorials channel, the search for a superior **contextual PDF parser** to Adobe API, and guidance for building a **Langchain-based calculator** were topics of discussion, aiming for practical integrations and solutions in AI workflows.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Seeking Argilla Hosting Solutions**: `@drxd1000` requested advice for hosting an **Argilla** server capable of supporting multiple annotators with no clear resolution reached.
- **Layer Selective Rank Reduction in the Spotlight**: `@johannhartmann` discussed an implementation of 'Layer Selective Rank Reduction' for mitigating **continual training forgetting**. The method targets statistically less significant layer parts, and a [GitHub repository](https://github.com/cognitivecomputations/laserRMT) was mentioned.
- **Overcoming OOM With Mixtral**: `@philipmay` encountered an Out of Memory error with a **mixtral model**, and `@bjoernp` suggested using **multi-GPU support**, mentioning that **two A100s** might alleviate the issue.
- **Cross-Language Toxicity Detection Dataset**: `@sten6633` sought a German toxicity evaluation dataset, considering the translation of [**ToxiGen**](https://huggingface.co/datasets/skg/toxigen-data) from Hugging Face, which requires access agreement.
- **New Computational Technique Teased**: `@phantine` teased a technique named "Universes in a bottle" with implications for the P=NP problem, linked to a [GitHub page](https://github.com/LargeWorldModel/LWM), but details were sparse.
- **BM25 Search Strategy Proves Effective**: `huunguyen` reported success using **BM25** with additional querying and reranking to enhance search capabilities, and successfully indexed the entirety of Wikipedia into an index under 3GB.
- **German AI Model Update Inquiry**: thomasrenkert asked about the release timeline for version 2 of the German model or a **Mixtral** variant, but no additional details were provided.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **CUDA Compatibility Crusade**: Members discussed achieving **CUDA** binary compatibility on **HIP/ROCm** platforms, driven by the [ZLUDA project](https://github.com/vosen/ZLUDA) on GitHub, which is a CUDA on AMD GPU initiative. Amidst technical emoji enthusiasm, there were musings about market monopolies and AGI, alongside personal experiences with Radeon hardware issues related to dynamic memory allocation.
  
- **Generative AI Jobs Galore**: A **Deep Tech Generative AI startup** in Hyderabad is hiring ML, Data, Research, and SDE roles, with applications welcomed [here](https://forms.gle/aP5qwv66XM2D7RCS8). However, the legitimacy of the job posting was questioned, flagging the need for moderator attention.

- **Compute Shaders and Matrix Math Musings**: Inquiries on educational materials for CUDA led to [The Book of Shaders](https://thebookofshaders.com/) recommendation, while the discussion in the PMPP book channel debated the benefits, or lack thereof, of transposing matrices to reduce cache misses in multiplication, indicating varied opinions but no consensus on observed benefits.

- **Apple Chips Enter Monitoring Realm**: `@marksaroufim` shared [asitop](https://github.com/tlkh/asitop), a **CLI performance monitoring tool** designed for **Apple Silicon**, likening it to `top` or `nvtop` in utility for engineers leveraging Apple's technology.

- **GPU Experiments and Job Shuffling**: An engineer successfully relocated an **Asus WS motherboard** to a miner setup, effectively running large **quantized models** on a **NVIDIA 3060 GPU**. This indicates a hands-on approach within the community towards custom hardware configurations.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Reka Enters the Model Arena**: A new AI entity named the **Reka model** has sparked interest in the community following a tweet shared by `@swyxio`. The excitement is palpable with discussions around the tweet found [here](https://twitter.com/YiTayML/status/1757115386829619534).
- **Investor Insights Meet AI**: `@swyxio` spotlighted a VC podcast delving into AI, which could be of significant interest to engineering aficionados. The podcast episode is accessible [here](https://overcast.fm/+afOCk9-tI).
- **BUD-E Buzz**: **BUD-E**, an empathetic and context-aware open voice assistant developed by LAION, could signal a new direction in conversational AI. More details are laid out on the [LAION blog](https://laion.ai/blog/bud-e/).
- **Pondering the Definition of Agents**: The community exchanged views on defining "agents," with `@slono` suggesting that they are goal-oriented programs that require minimal input from users, a concept significant in the realm of AI development.
- **Karpathy's OpenAI Exit Raises Questions**: The AI community is abuzz over the news of **Andrej Karpathy** leaving OpenAI, with `@nembal` pointing to an article from The Information and speculation about AGI influences. The article is accessible [here](https://www.theinformation.com/articles/openai-researcher-andrej-karpathy-departs).



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Minding the Model Size for M2 Max**: `@potrock` inquired about running **Mistral model sizes** on an M2 Max with 32GB, and `@natureplayer` advised that a **4GB** model would be the feasible option, cautioning against an 8GB model and noting that 5GB models may be unstable.

- **GPT-5 Rumor Mill**: `@res6969` expressed humorous doubt about **GPT-5's** existence, suggesting that speculation on the model's upcoming release is overstated, with others joining the jest with emojis.

- **Enhanced Memory in ChatGPT**: `@potrock` highlighted a new feature tested in **ChatGPT**, based on a [blog post](https://openai.com/blog/memory-and-new-controls-for-chatgpt), where it can retain user preferences and information across sessions for more personalized interactions.



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **Weekly Sync-Up Teases Déjà vu**: `@._z` playfully announces the start of the **weekly team meeting** likening it to a recurring *Déjà vu* experience.
- **Member Bows Out from Meeting**: `@juanreds` sends regrets for being unable to attend the **weekly meeting**, offering apologies to the team.
- **Call for AI Hackathon Co-Hosts**: `@caramelchameleon` seeks collaborators to co-host an **AI developers hackathon** in tandem with game developers in the lead-up to the GDC.
- **Hackathon Offers Dual Attendance Modes**: The hackathon mentioned by `@caramelchameleon` has options for attendance both **online** and **onsite in San Francisco**.
- **Hackathon Organizer Steps Up**: `@yikesawjeez` shows eagerness to get involved in organizing the hackathon and highlights their expertise with Bay Area events.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Direct Messaging Initiated**: User `@bondconnery` has put out a request for a **private message**.
- **Exploring LLaVA Framework Integration**: `@CodeMan` inquired about integrating the **LLaVA** framework with an **SGLang server** and **SGLang worker**, aiming for a potentially more specialized setup than a conventional model worker.
- **Off-Topic Video Share Ignored**: A non-technical video link was shared, not relevant to the engineering discussions.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Datasette - LLM (@SimonW) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1206510944260464640) (1460 messages🔥🔥🔥): 

- **Exploring the Limits of Large Language Models**: Users are discussing new open-source large language models capable of handling extremely long contexts, such as the [Large World Model](https://github.com/LargeWorldModel/LWM) which claims to work coherently with contexts up to 1 million tokens. There are also mentions of the Cohere's `aya` model that supports 101 languages.
- **The Quest for Efficient Multimodal AIs**: Conversations focus on multimodality in AI with references to models handling visual inputs and potential outputs, indicating significant advancements beyond text-based models. The jax-based tools required to run the models are causing installation hiccups for some users.
- **Models Under Scrutiny**: The community is very active in testing released models, mentioning issues such as TUX dependency problems and a `ValueError` during setup, indicating some challenges in getting the advanced models running smoothly.
- **Users Share Knowledge**: Experienced users offer insights and assistance on how to handle models and UIs for various tasks, including long-context quantization in existing frameworks like ExLlama v2. Discussions also touch on the possibility of banishing stop tokens to encourage longer continuous outputs.
- **Towards Intelligent Role-Playing**: There is a discussion on finding the balance between RP-oriented models and smarter generalized ones, with mentions of a Mixtral variant (`BagelMIsteryTour`) that might better fulfill user requirements for intelligent and adaptable model behavior.

**Links mentioned**:

- [Context &ndash; share whatever you see with others in seconds](https://ctxt.io/2/AADwBrkOEQ): no description found
- [Lil Yachty Drake GIF - Lil Yachty Drake - Discover &amp; Share GIFs](https://tenor.com/view/lil-yachty-drake-gif-21712801): Click to view the GIF
- [Memory and new controls for ChatGPT](https://openai.com/blog/memory-and-new-controls-for-chatgpt): We’re testing the ability for ChatGPT to remember things you discuss to make future chats more helpful. You’re in control of ChatGPT’s memory.
- [brucethemoose/LargeWorldModel_LWM-Text-Chat-128K-55bpw · Hugging Face](https://huggingface.co/brucethemoose/LargeWorldModel_LWM-Text-Chat-128K-55bpw): no description found
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/): no description found
- [Kooten/BagelMIsteryTour-v2-8x7B-5bpw-exl2 · Hugging Face](https://huggingface.co/Kooten/BagelMIsteryTour-v2-8x7B-5bpw-exl2): no description found
- [no title found](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/LostRuins/koboldcpp&ved=2ahUKEwjtgOjjyKmEAxUAQ0EAHdKdCj8QFnoECBMQAQ&usg=AOvVaw2LNzIwtZ11OMtacr6FrLbK): no description found
- [Think Bigger Skeletor GIF - Think Bigger Skeletor Masters Of The Universe Revelation - Discover &amp; Share GIFs](https://tenor.com/view/think-bigger-skeletor-masters-of-the-universe-revelation-hope-for-a-destination-look-at-the-bigger-picture-gif-24729071): Click to view the GIF
- [no title found](https://tenor.com/view/think-bigger-skeletor-masters-of-the-universe-revelation-hope-for-a-destinati): no description found
- [CausalLM/34b-beta · Hugging Face](https://huggingface.co/CausalLM/34b-beta): no description found
- [SimSim93/CausalLM-34b-beta_q8 · Hugging Face](https://huggingface.co/SimSim93/CausalLM-34b-beta_q8): no description found
- [GitHub - jy-yuan/KIVI: KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://github.com/jy-yuan/KIVI): KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache - jy-yuan/KIVI
- [The Verge: How NOT to Build a Computer](https://www.youtube.com/watch?v=jciJ39djxC4): SPONSOR: Go to http://expressvpn.com/science to take back your Internet privacy TODAY and find out how you can get 3 months free.Link to the Verge&#39;s awful vi...
- [LWM/lwm/llama.py at main · LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM/blob/main/lwm/llama.py#L694>): Contribute to LargeWorldModel/LWM development by creating an account on GitHub.
- [https://drive.google.com/drive/folders/1my-8wOIYXmfnlryDbwJ20_y6PFCqRfA-?usp=sharinghttps://drive.google.com/drive/folders/1my-8wOIYXmfnlryDbwJ20_y6PFCqRfA-?usp=sharingData Challenge - Aether 2024](https://docs.google.com/forms/d/e/1FAIpQLSfkAf5dNJB64em2ywddV12OyHMBy8d698Hr3wK5X4CTA00hnA/viewform): In order to participate in the Data Challenge organised by Enigma as part of Aether, Please fill out this form Event Date &amp; Time: Wednesday, February 14th - 2:30 pm Please double-check your detail...
- [LargeWorldModel (Large World Model)](https://huggingface.co/LargeWorldModel): no description found
- [GitHub - lhao499/tux: Tools and Utils for Experiments (TUX)](https://github.com/lhao499/tux): Tools and Utils for Experiments (TUX). Contribute to lhao499/tux development by creating an account on GitHub.
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind): A multimodal, function calling powered LLM webui.  - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.
- [llama.cpp/examples/server at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master/examples/server): Port of Facebook&#39;s LLaMA model in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
- [Tweet from Cohere For AI (@CohereForAI)](https://x.com/CohereForAI/status/1757359611399532921?s=20): Today, we’re launching Aya, a new open-source, massively multilingual LLM & dataset to help support under-represented languages. Aya outperforms existing open-source models and covers 101 different la...
- [OpenAI Researcher Andrej Karpathy Departs](https://www.theinformation.com/articles/openai-researcher-andrej-karpathy-departs): Andrej Karpathy, one of the founding members of OpenAI, has left the company, a spokesperson confirmed. Karpathy, a prominent artificial intelligence researcher, was developing a product he has descri...
- [CohereForAI/aya-101 · Hugging Face](https://huggingface.co/CohereForAI/aya-101): no description found
- [GitHub - valine/NeuralFlow](https://github.com/valine/NeuralFlow/tree/master): Contribute to valine/NeuralFlow development by creating an account on GitHub.
- [ChatGPT but Uncensored and Free! | Oogabooga LLM Tutorial](https://youtu.be/SLb5n8AX33s?si=N6g3RmLoMt83_VCj): ChatGPT but uncensored and free, well its now possible thanks to the open source AI community! In this video I show you how to set up the Oogabooga graphical...
- [LWM/lwm/vision_chat.py at main · LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM/blob/main/lwm/vision_chat.py#L122>): Contribute to LargeWorldModel/LWM development by creating an account on GitHub.
- [New emails reveal scientists believed COVID-19 was man-made](https://youtu.be/aWbLOHSufvc?si=L0DfeDAsdZr-BX3o): New emails have revealed scientists got together to discuss the origins of COVID, suspecting it was man-made, before deciding to tell the public it originate...
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind/): A multimodal, function calling powered LLM webui.  - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.
- [GitHub - LargeWorldModel/LWM](https://github.com/LargeWorldModel/lwm): Contribute to LargeWorldModel/LWM development by creating an account on GitHub.
- [GitHub - acorn-io/rubra: AI Assistants, LLMs and tools made easy](https://github.com/acorn-io/rubra): AI Assistants, LLMs and tools made easy. Contribute to acorn-io/rubra development by creating an account on GitHub.
- [unalignment/weeeeee.0 · Hugging Face](https://huggingface.co/unalignment/weeeeee.0): no description found
- [unalignment/weeeeee.1 · Hugging Face](https://huggingface.co/unalignment/weeeeee.1): no description found
- [unalignment/weeeeee.2 · Hugging Face](https://huggingface.co/unalignment/weeeeee.2): no description found
- [CohereForAI/aya_dataset · Datasets at Hugging Face](https://huggingface.co/datasets/CohereForAI/aya_dataset): no description found
- [CohereForAI/aya_collection · Datasets at Hugging Face](https://huggingface.co/datasets/CohereForAI/aya_collection): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1206511029450706964) (154 messages🔥🔥): 

- **Exploring Miqu Variants**: `@superking__` opened a discussion about the performance of the Miqu models, particularly after being re-quants from the original GGUFs (Google's Generative Unsupervised Feature extraction). `@soufflespethuman` mentioned **MiquMaid-v2-70B**, a variant specifically fine-tuned for ERP (Erotic Role Play), and provided **[sensitive content links](https://huggingface.co/NeverSleep/MiquMaid-v2-70B)** to various versions on Hugging Face, which have been marked due to their nature.
  
- **Performance Gain with Better Hardware**: `@superking__` shared their experience on performance improvement from "painfully slow" to "almost usable" by upgrading their hardware to 12GB VRAM, which changed the given model's tokens per second from 0.7t/s to 2.1t/s.

- **Model Comparisons and Recommendations**: In the context of roleplay and storytelling, users discussed various models. `@spottyluck` praised **Nous Capybara Limarpv3 34B** for its capabilities and provided a **[link to the model on Hugging Face](https://huggingface.co/TheBloke/Nous-Capybara-limarpv3-34B-GGUF)**. `@wolfsauge` shared a sketch about "The Continental" featuring Christopher Walken and `@eqobaba` inquired about appropriate models and settings for engaging in NSFW ERP, mentioning a specification of **48GB VRAM** and **RTX A600**.

- **Discussing Model Output Improvement**: `@neriss` suggested using a higher temperature or lower minimum probability to reduce repetition and improve creativity in AI model outputs. The conversation highlighted variations in temperature settings, with `@dercheinz` suggesting higher temperatures, while `@neriss` advised lower ones, each to counteract repetitive or uncreative responses from models.

- **Dataset Cleaning Challenges and Strategies**: `@c.gato` and `@potatooff` exchanged thoughts on cleaning datasets manually, with `@c.gato` seeking advice on how to perform **ngram analysis** to prevent overtraining on specific ngrams. `@mrdragonfox` recommended using **Python's pandas library** for handling tabular or JSON data, sharing a **[gist for guidance](https://gist.github.com/darkacorn/f786564868357cde5894ef6e2c6f64cf)**.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://canary.discord.com/channels/1111983596572520458/1112353569077735595): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1112409939336503338/1175939201330585631): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1112409939336503338/1175838485568036936): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [TheBloke/Nous-Capybara-limarpv3-34B-GGUF · Hugging Face](https://huggingface.co/TheBloke/Nous-Capybara-limarpv3-34B-GGUF): no description found
- [Models - Hugging Face](https://huggingface.co/models?sort=modified&search=LoneStriker%2Fmiqu-+exl2>): no description found
- [gist:f786564868357cde5894ef6e2c6f64cf](https://gist.github.com/darkacorn/f786564868357cde5894ef6e2c6f64cf): GitHub Gist: instantly share code, notes, and snippets.
- [The Continental: Anticipation - Saturday Night Live](https://www.youtube.com/watch?v=0vuOnVNiYtg): Subscribe to SaturdayNightLive: http://j.mp/1bjU39dSEASON 26: http://j.mp/14GYJ6nThe night air is tinged with anticipation. It&#39;s time to meet The Continental...
- [Happy Fun Ball - SNL](https://www.youtube.com/watch?v=GmqeZl8OI2M): Happy Fun Ball seems great until you hear all the potential side effects. [Season 16, 1991]#SNLSubscribe to SNL: https://goo.gl/tUsXwMStream Current Full Epi...
- [NeverSleep/MiquMaid-v2-70B · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B): no description found
- [NeverSleep/MiquMaid-v2-70B-GGUF · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-GGUF): no description found
- [NeverSleep/MiquMaid-v2-70B-DPO · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-DPO): no description found
- [NeverSleep/MiquMaid-v2-70B-DPO-GGUF · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v2-70B-DPO-GGUF): no description found

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1206530538152796191) (43 messages🔥): 

- **Understanding Finetuning Techniques**: `@starsupernova` explained that **Mixtral – Instruct** was trained using **SFT** on an instruction dataset followed by **Direct Preference Optimization (DPO)** on a paired feedback dataset, as detailed on [page 6 of their paper](https://arxiv.org/pdf/2401.04088.pdf). DPO is described as an optimized form of **RLHF/PPO** finetuning.

- **Unsloth AI's Apply Chat Template**: `@starsupernova`, likely the founder of Unsloth AI, highlighted the use of `apply_chat_template` instead of Alpaca for training an LLM on multi-turn conversation datasets. They also hinted at uploading a new notebook with all chat templates to simplify the process.

- **Augmentoolkit for Instruct-Tuning Datasets**: In the conversation, `@mr.userbox020` shared a link to a [GitHub repository](https://github.com/e-p-armstrong/augmentoolkit) offering a toolkit to convert Compute and Books Into Instruct-Tuning Datasets. Although `@starsupernova` was not familiar with it, they suggested trying it out as it appeared promising.

- **Anticipation for Updated Training Resources**: `@avinierdc` is awaiting an updated notebook from `@starsupernova` for fine-tuning Mistral on multi-turn conversation datasets. `@starsupernova` assured they would ping when it's available on the Unsloth's Discord server.

- **Unexplained Variation in Training Loss**: `@dreamgen` reported observing a 2x higher training and evaluation loss when fine-tuning **Mixtral 8x7b qlora** compared to **Mistral 7b**, despite using the same dataset and similar hyperparameters, and inquired if others had seen something similar.

**Links mentioned**:

[GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets](https://github.com/e-p-armstrong/augmentoolkit): Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1206590490963873853) (8 messages🔥): 

- **Interest in Collaboration Sparked**: `@_b_i_s_c_u_i_t_s_` expressed interest in an unspecified topic, potentially around chatbot implementation, which was well received by `@mr_pebble`, finding it motivating to progress on implementing various chat methods.

- **Bridging JavaScript and Python**: `@spottyluck` experimented with expanding the SillyTavern project to use a JavaScript-Python bridge, utilizing [JSPyBridge](https://github.com/extremeheat/JSPyBridge) to potentially adapt and enhance functionalities. They shared how it enabled testing of Microsoft's LLMLingua, despite some issues with prompt mangling.

- **Using Python Classes in JavaScript**: `@spottyluck` provided code examples illustrating the ease of creating Python classes within JavaScript using JSPyBridge, along with an asynchronous function, `compressPrompt`, which demonstrates the interaction between languages to compress prompts.

- **Modifications to Handle Windows Errors and Devices**: In their continued development, `@spottyluck` modified Intel's BigDL.llm transformer to support specific requirements, like `cpu_embedding=True` on Windows due to access violation errors, and dealing with model device allocation issues using `model.to()`.

- **Compression Process Integration into Routing**: `@spottyluck` explained integrating prompt compression into their web service by adding a flag to the `/generate` router post and using conditional logic to process the prompt through the bridge, demonstrating how Python can operate as if it were a JavaScript class.

**Links mentioned**:

[GitHub - extremeheat/JSPyBridge: 🌉. Bridge to interoperate Node.js and Python](https://github.com/extremeheat/JSPyBridge): 🌉. Bridge to interoperate Node.js and Python . Contribute to extremeheat/JSPyBridge development by creating an account on GitHub.

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1206518086899863552) (202 messages🔥🔥): 

- **Struggles with Large Models**: Users like `@nonm3` encountered **errors while loading large models** in LM Studio due to insufficient RAM and VRAM, with suggestions to try smaller model quants. Others like `@theoverseerjackbright` faced issues with LM Studio not detecting GPUs correctly and crashing post-restart.
  
- **Software Seekers and Recommendations**: `@tvb1199` was in search of client software that can interact with LM Studio for RAG capabilities, and was pointed towards AGiXT, while `@pierrunoyt` and others discussed Nvidia's 'Chat with RTX' with RAG features as a potential game-changer.

- **Compatibility Inquiries**: Several users such as `@wizzy09` had trouble installing or opening LLM Studio on unsupported platforms like a 2014 MacBook Pro, with clarifications from users like `@heyitsyorkie` explaining that LMStudio does not work on Intel Macs.

- **Nvidia's Chat with RTX Triggers Interest**: The community showed a **keen interest in Nvidia's 'Chat with RTX'**. Users like `@hypocritipus` were intrigued by the RAG feature, hoping for a similar easy-install, no-dependency RAG feature in LM Studio.

- **LM Studio Usage and Model Discussions**: Users like `@bigboimarkus` expressed satisfaction with LM Studio for tasks such as proofreading, whereas `@mr.stark_` queried about models that learn on the fly. Conversations included the functionality and integration with other tools like Ollama and Automatic1111.

- **General Community Assistance and Banter**: Throughout, community members engaged in sharing tips, offering troubleshooting advice, including suggestions for alternatives or downgrading versions, and occasionally joked about AI capabilities such as predicting lottery numbers.

**Links mentioned**:

- [Stable Cascade - a Hugging Face Space by multimodalart](https://huggingface.co/spaces/multimodalart/stable-cascade): no description found
- [cmp-nct/Yi-VL-6B-GGUF at main](https://huggingface.co/cmp-nct/Yi-VL-6B-GGUF/tree/main): no description found
- [TheBloke/CodeLlama-70B-Instruct-GGUF at main](https://huggingface.co/TheBloke/CodeLlama-70B-Instruct-GGUF/tree/main): no description found
- [Chost Machine GIF - Chost Machine Ai - Discover &amp; Share GIFs](https://tenor.com/view/chost-machine-ai-type-typing-gif-481421031430735140): Click to view the GIF
- [System prompt - Pastebin.com](https://pastebin.com/vnxJ7kQk): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
- [NVIDIA Chat With RTX](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/): Your Personalized AI Chatbot.
- [The unofficial LMStudio FAQ!](https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio): Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...
- [The unofficial LMStudio FAQ!](https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-): Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1206621081335631872) (75 messages🔥🔥): 

- **MedAlpaca for Medical LLMs**: User `@heyitsyorkie` suggested [medAlpaca](https://github.com/kbressem/medAlpaca), a model fine-tuned for medical question answering, for `@pepito92i`'s project on LLMs in the medical field.
- **Phi-2 Model Discussions**: `@.dochoss333` inquired about the absence of the official "microsoft/phi-2" model in LM Studio, and `@heyitsyorkie` clarified that it's not a GGUF model and thus won't show up. `@hugocapstagiaire_54167` mentioned user TheBloke might have transformed it into a .gguf for usability with llama.cpp.
- **LLama.cpp and Model Support**: `@jedd1` puzzled over why some models wouldn't load, and `@heyitsyorkie` pointed out that the Yi-VL models are unsupported in the current build of llama.cpp, requiring an update for compatibility.
- **LM Studio Assistant Functionality Inquiry**: User `@edu0835` inquired about the possibility of creating an assistant in LM Studio with the ability to utilize PDFs or books for a medical assistant application, without a direct response provided at this time.
- **Model Performance Comparisons Engage Community**: Users like `@kujila` and `@heyitsyorkie` engaged in comparisons between different language models, with discussions on model specificity, ethical behavior of AI models, and suggestions to try out models like Deepseek Coder Ins 33B.

**Links mentioned**:

- [Nexesenex/Senku-70b-iMat.GGUF at main](https://huggingface.co/Nexesenex/Senku-70b-iMat.GGUF/tree/main): no description found
- [Hi Everybody Simpsons GIF - Hi Everybody Simpsons Wave - Discover &amp; Share GIFs](https://tenor.com/view/hi-everybody-simpsons-wave-gif-12144219): Click to view the GIF
- [TheBloke/medicine-chat-GGUF · Hugging Face](https://huggingface.co/TheBloke/medicine-chat-GGUF): no description found
- [wolfram/miquliz-120b-v2.0-GGUF · Hugging Face](https://huggingface.co/wolfram/miquliz-120b-v2.0-GGUF): no description found
- [GitHub - kbressem/medAlpaca: LLM finetuned for medical question answering](https://github.com/kbressem/medAlpaca): LLM finetuned for medical question answering. Contribute to kbressem/medAlpaca development by creating an account on GitHub.
- [The new Yi-VL-6B and 34B multimodals ( inferenced on llama.cpp, results here ) · ggerganov/llama.cpp · Discussion #5092](https://github.com/ggerganov/llama.cpp/discussions/5092): Well, their benchmarks claim they are almost at GPT4V level, beating everything else by a mile. They also claim that CovVLM is one of the worst (and it&#39;s actually the best next to GPT4, by far) On...

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1206544203820171314) (140 messages🔥🔥): 

- **NVLink and Memory Cycling Queries**: `@luxaplexx` asked if GPUs were NVLinked and how memory cycles through in a multi-GPU setup. The consensus, including from `@heyitsyorkie`, is that they likely aren't NVLinked due to potential CUDA issues, especially with older cards like the 960. Users are considering whether different generations of NVIDIA GPUs like the 1080 and 1060 6g can work together effectively.

- **Discussions on Upgrading to Better GPUs**: Several users, including `@crsongbirb` and `@heyitsyorkie`, discussed upgrading their GPUs for improved performance in LLM tasks, with a suggestion to look at the RTX 3060 12GB as a viable option for running LLMs locally.

- **Risks and Rewards of Overclocking**: In a discussion initiated by `@alastair9776` about overclocking for better performance, `@rugg0064` and `@crsongbirb` noted that overclocking VRAM/RAM can lead to a notable increase in token generation speed, although caution is advised due to potential hardware stress.

- **Combining GPUs and Threadripper Dreams**: Conversation ensued about the feasibility and costs of using multiple high-performance GPUs, with users like `@nink1` and `@quickdive.` debating if a beefy CPU is necessary when having multiple powerful GPUs, and the logistics of housing such a setup.

- **CUDA on AMD and Other Hardware Convos**: `@666siegfried666` shared news about the ZLUDA project allowing CUDA apps to run on AMD hardware and this sparked a brief discussion on the relevance and future potential of such a feature. Users such as `@addressofreturnaddress` and `@joelthebuilder` also discussed their own rig setups and potential upgrades, highlighting personal preferences and value assessments.

**Links mentioned**:

- [Doja Cat GIF - Doja Cat Star - Discover &amp; Share GIFs](https://tenor.com/view/doja-cat-star-wars-gif-25078126): Click to view the GIF
- [Brexit British GIF - Brexit British Pirate - Discover &amp; Share GIFs](https://tenor.com/view/brexit-british-pirate-england-sinking-gif-5922477): Click to view the GIF
- [ATOM Echo Smart Speaker Development Kit](https://shop.m5stack.com/products/atom-echo-smart-speaker-dev-kit?variant=34577853415588): ATOM ECHO is a programmable smart speaker.This eps32 AIoT Development Kit has a microphone and speaker for AI voice interaction light and small. It can be access AWS, Baidu, ESPHome and Home Assistant...
- [Unmodified NVIDIA CUDA apps can now run on AMD GPUs thanks to ZLUDA - VideoCardz.com](https://videocardz.com/newz/unmodified-nvidia-cuda-apps-can-now-run-on-amd-gpus-thanks-to-zluda): ZLUDA enables CUDA apps on ROCm platform, no code changes required AMD-backed ZLUDA project can now enable code written in NVIDIA CUDA to run natively on AMD hardware.  AMD has reportedly taken over t...
- [Lian-Li O11 Dynamic XL ROG certificated -Black color Tempered Glass](https://www.canadacomputers.com/product_info.php?cPath=6_6004_5960&item_id=151208): Lian Li O11 Dynamic XL ROG certificated, Front and Left Tempered Glass, E-ATX, ATX Full Tower Gaming Computer Case - Black

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1206760641482858507) (21 messages🔥): 

- **Awaiting the Next Update for IQ3_XSS Support**: `@n8programs` inquired about **IQ3_XSS** support in the latest release, to which `@yagilb` responded that **it will be included in the next update**.
- **Elevation of 1bit Quantization on the Horizon**: `@drawless111` shared excitement about upcoming 1.5 bit quantization, posting a [GitHub pull request link](https://github.com/ggerganov/llama.cpp/pull/5453) indicating progress. This elicits reactions with `@heyitsyorkie` anticipating a sweet next beta with the new quant sizes.
- **Model Benchmarking Induces Awe**: `@drawless111` expressed amazement at the latest benchmarks for **1bit quantization**, stating *“70B model on 16 GB card. WOOF."* and pointing out a '70B' model [posted on Hugging Face](https://huggingface.co/Nexesenex/NousResearch_Yarn-Llama-2-70b-32k-iMat.GGUF) that can offload on VRAM effectively.
- **Preparations for Incompatible Model Downloads**: Users, including `@epicureus`, are advised to download models like **IQ3_XSS** even if they're not supported yet, with `@fabguy` humorously suggesting **"Save the model, save the world!"**
- **Hugging Face Hub Features Multiple New Models**: `@drawless111` shared an update, revealing the availability of **5 IQ1 models on Hugging Face** that work with various VRAM sizes, nonchalantly noting an increase to 10 by the end of the conversation.

**Links mentioned**:

- [Nexesenex/NousResearch_Yarn-Llama-2-70b-32k-iMat.GGUF · Hugging Face](https://huggingface.co/Nexesenex/NousResearch_Yarn-Llama-2-70b-32k-iMat.GGUF): no description found
- [Claire Bennet Heroes GIF - Claire Bennet Heroes Smile - Discover &amp; Share GIFs](https://tenor.com/view/claire-bennet-heroes-smile-happy-relieved-gif-5008424): Click to view the GIF
- [1.5 bit quantization by ikawrakow · Pull Request #5453 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5453): This draft PR is a WIP that demonstrates 1.5 bits-per-weight (bpw) quantization. Only CUDA works, there is no implementation for the other supported back-ends. CUDA, AVX2 and ARM_NEON are implement...

  

---


### LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1207094925192331294) (2 messages): 

- **No AVX2, No Cry**: `@rafalsebastian` expressed concerns about not being able to run LMstudio on CPUs with only AVX (version one) after getting the message that their processor doesn't support AVX2. They wondered if they should switch machines for running local LLMs.
- **LM Studio Beta for the Rescue**: `@heyitsyorkie` responded with a solution, mentioning that LM Studio can indeed run on CPUs with only AVX support by downloading the **0.2.10 AVX beta release for Windows**. They also recommended upgrading to a CPU with AVX2 for optimal results and provided a link to [beta releases and terms of use](https://lmstudio.ai/beta-releases.html).

**Links mentioned**:

[LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html): no description found

  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1206633942929768528) (3 messages): 

- **Looking for Dual Model Deployment Tips**: `@alluring_seahorse_04960` wonders how to run two models on the same machine without facing repetition errors. The user mentions using a Conda environment on Ubuntu and avoids VMs for their slowness.
- **Humorous Clarification Request on Repetition**: In response to `@alluring_seahorse_04960`, `@wolfspyre` jokes about the nature of the repetition errors, questioning whether they pertain to looping outputs or tasking issues within worker processes.
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1206606051890438174) (361 messages🔥🔥): 

- **Magvit V2 Reproduction Inquiries**: `@.lostneko` sought technical guidance for reproducing [Magvit V2](https://github.com/lucidrains/magvit2-pytorch). Discussions circled around the ideal datasets and parameters for video compression and understanding, with `@chad_in_the_house` mentioning experiments on the lfq side of Magvit2.
- **Mysterious Buzz around Magvit**: `@pseudoterminalx` and others in the chat noticed sudden interest in MAGVIT, speculating about a recent influencer mention given the two mentions within a short time frame.
 
- **Stable Cascade Discussions Heat Up**: Focus shifted to Stability AI's Stable Cascade model, with dialogues highlighting its hefty VRAM requirements, misleading inference time graphs, and concerns about the model being poorly optimized and full of bugs. `@pseudoterminalx` shared examples of its capabilities, including issues with text clarity in image outputs.

- **Evaluating AI Models and Copyright Concerns**: Conversations touched on the usage and legality of AI-generated images. Users `@vrus0188` and `@kenjiqq` debated AI image model copyrights, commercial use, and the implications of research-only model licenses.

- **Hardware and Performance Perspectives**: A technical dialogue ensued over Stable Cascade's heavy VRAM use and optimization problems, as `@pseudoterminalx` reported issues like inability to run models in float16 and `@kenjiqq` provided details about inference time on consumer GPUs like the 3090.

**Links mentioned**:

- [Stable Cascade - a Hugging Face Space by multimodalart](https://huggingface.co/spaces/multimodalart/stable-cascade): no description found
- [Court Dismisses Authors’ Copyright Infringement Claims Against OpenAI * TorrentFreak](https://torrentfreak.com/court-dismisses-authors-copyright-infringement-claims-against-openai-240213/): no description found
- [Stable Cascade のご紹介 &mdash; Stability AI Japan &mdash; Stability AI Japan](https://ja.stability.ai/blog/stable-cascade): Stable Cascade の研究プレビューが開始されました。この革新的なテキストから画像へのモデルは、品質、柔軟性、微調整、効率性の新しいベンチマークを設定し、ハードウェアの障壁をさらに排除することに重点を置いた、興味深い3段階のアプローチを導入しています。
- [Hey Hindi GIF - Hey Hindi Bollywood - Discover &amp; Share GIFs](https://tenor.com/PzRY.gif): Click to view the GIF
- [Don't ask to ask, just ask](https://dontasktoask.com/): no description found
- [GitHub - Stability-AI/StableCascade](https://github.com/Stability-AI/StableCascade): Contribute to Stability-AI/StableCascade development by creating an account on GitHub.
- [Crypto Kids Poster | 24posters | Hip Hop &amp; Street Art Prints](https://24posters.co/products/crypto-kids-6): Transform your walls with our viral new Crypto Kids Poster. Inspired by street-wear &amp; hip hop culture, enjoy artwork designed to bring you bedroom to life. Fast shipping times (3-5 days) 10,000+ h...

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1206617919803498506) (48 messages🔥): 

- **Discussion on Impact of Adult Content on AI**: `@vrus0188` and others discuss the historical contributions of adult content to advancing technology, juxtaposing it with AI developments. Some users like `@twoabove` acknowledge the pattern of adult industries driving tech advancements, while others like `@SegmentationFault` doubt if the focus on adult content leads to meaningful progress in AI.

- **Concern Over Explicit AI-Generated Content**: `@thejonasbrothers` shares a [news article](https://bc.ctvnews.ca/ai-brings-deepfake-pornography-to-the-masses-as-canadian-laws-play-catch-up-1.6754498) highlighting the misuse of AI in creating non-consensual pornography, noting the challenges it poses and its high visibility. This leads to a discussion on the broader implications and controversies surrounding AI's use in adult content.

- **Observations on the Pornography Market and AI**: Users like `@chad_in_the_house` and `@freon` discuss the profitability and market saturation of NSFW content, contemplating the economical and ethical risks involved in this space.

- **Debates Over the Merits of AI-Powered Erotic Roleplay**: `@SegmentationFault` expresses frustration over the preference for low-effort erotic content in AI communities, arguing that this hinders meaningful developments in AI models. Others like `@mfcool` and `@.undeleted` echo these sentiments, criticizing the quality stagnation in AI-generated adult imagery.

- **Technical Discussion on AI Image Quality**: `@drhead` delves into technical aspects of AI-generated images, mentioning the NovelAI model and discussing the viability and impact of VAE encoder training for improved image generation. There is a communal reflection on the standards of "photorealism" within the community and how they could be improved.

**Links mentioned**:

- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/1aol): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/1aolvxz/instructive_training_for_complex_concepts/): no description found
- [AI brings deepfake pornography to the masses, as Canadian laws play catch-up](https://bc.ctvnews.ca/ai-brings-deepfake-pornography-to-the-masses-as-canadian-laws-play-catch-up-1.6754498): Underage Canadian high school girls are targeted using AI to create fake explicit photos that spread online. Google searches bring up multiple free websites capable of &quot;undressing&quot; women in ...

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1206588512103571529) (179 messages🔥🔥): 

- **Checksums for The Pile Data Located**: `@paganpegasus` provided `@hailey_schoelkopf` with the checksums for The Pile zst shards, linking both [the Discord pins](https://discord.com/channels/729741769192767510/729741769738158194/1177260719847264266) and the [EleutherAI's hashes](https://www.eleuther.ai/hashes).

- **Tools to Determine Image Content**: `@everlasting_gomjabbar` inquired about tools to discern if an image is of an object/location versus 'nothing' like a blurry shot. `@paganpegasus` described the complexity of defining "nothing" in images, while `@rallio.` recommended using models like OwlViT or CLIP.

- **Manuscript Review and Editing in Progress**: `@wonkothesensible`, through a series of messages, provided meticulous feedback on a paper draft provisionally titled "Don't think about the paper", focusing on clarifying language and grammar. `@hailey_schoelkopf` expressed gratitude and indicated credits to the EleutherAI Discord in the paper's acknowledgements.

- **Cloud Resources for NLP Classification Discussed**: In response to `@pxxxl` seeking advice on cloud resources for training NLP classification models, `@ad8e` recommended GCP and Colab, with various participants chiming in about the costs and features of various platforms like runpod and vast.ai.

- **Inquiries About EleutherAI Computing Resources**: User `@vidava` asked about the guidelines and requirements for accessing EleutherAI's computational resources for a semi-custom LLM project featuring architectural adjustments and fine-tuning adapters. `@stellaathena` indicated openness to collaboration but highlighted the need for clarity on the research agenda and proposed a collaborative value proposition.

- **Semantic Scholar Paper-Author Linking Mechanism**: Regarding whether Semantic Scholar automatically links Arxiv papers to authors, `_inox` clarified that the process is automatic but allows for manual intervention or suggested changes if errors occur.

**Links mentioned**:

- [Overleaf, Online LaTeX Editor](https://www.overleaf.com/9551152421jrhcxsnphqmr#2faf0c): An online LaTeX editor that’s easy to use. No installation, real-time collaboration, version control, hundreds of LaTeX templates, and more.
- [
      Research Paper Release Checklist
    ](https://nicholas.carlini.com/writing/2022/paper-release-checklist.html): no description found
- [lora_example.py](https://gist.github.com/Chillee/a8d2070b1b7b3f97d8c87bac3c366f8e): lora_example.py. GitHub Gist: instantly share code, notes, and snippets.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/729741769192767510/729741769738158194/1177260719847264266): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Hashes &mdash; EleutherAI](https://www.eleuther.ai/hashes): no description found

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1206561058458837003) (208 messages🔥🔥): 

- **Fractal Analysis of Neural Network Hyperparameters**: `@jbustter` shared visualizations of fractals generated from neural network hyperparameters, with red indicating diverging training and blue for converging. [Jascha Sohl-Dickstein's blog post](https://sohl-dickstein.github.io/2024/02/12/fractal.html) showcases the concept, correlating fractal patterns with the learning rates of network layers and the network's weight offset.
  
- **Discussing Convergence and Divergence in Training**: The conversation, involving users like `@Hawk`, `@genetyx8`, and `@mrgonao`, discussed the means for determining if neural network training is converging or diverging, debating the presence of "diverging to infinity" and the nature of boundaries within fractal visualizations, with suggestions that NaNs may denote divergence.

- **Active Learning and Data Presentation Order in ML**: `@rybchuk` inquired about research on models choosing the order of data presentation, leading to a discussion about active learning. `@thatspysaspy` mentioned the subfield's existence, noting its lack of success, and `@catboy_slim_` added that it could halve training requirements by using smaller models to filter data for larger models' training.

- **Leveraging Unsupervised Data in Encoder-Decoder Models**: The question of how to utilize large unsupervised datasets effectively in encoder-decoder models for tasks such as audio to text was brought up by `@loubb`. Suggestions and discussions ranged from training components separately to integrating cross-attention during pre-training.

- **Release of an NLP Robustness Paper and Test-Time Augmentation**: `@millander` announced the publication of their lead author paper on improving text classifiers' robustness through test-time augmentation (TTA) using large language models. They thanked the community for support and shared the [arxiv link to their work](https://arxiv.org/abs/2402.08225).

**Links mentioned**:

- [Neural network training makes beautiful fractals](https://sohl-dickstein.github.io/2024/02/12/fractal.html): This blog is intended to be a place to share ideas and results that are too weird, incomplete, or off-topic to turn into an academic paper, but that I think may be important. Let me know what you thin...
- [A Poster for Neural Circuit Diagrams](https://www.vtabbott.io/ncd-poster/): As some of you might know, I have been working on neural circuit diagrams over the past year or so. These diagrams solve a lingering challenge in deep learning research – clearly and accurately commun...
- [MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts](https://arxiv.org/abs/2401.04081): State Space Models (SSMs) have become serious contenders in the field of sequential modeling, challenging the dominance of Transformers. At the same time, Mixture of Experts (MoE) has significantly im...
- [Scaling Laws for Fine-Grained Mixture of Experts](https://arxiv.org/abs/2402.07871): Mixture of Experts (MoE) models have emerged as a primary solution for reducing the computational cost of Large Language Models. In this work, we analyze their scaling properties, incorporating an exp...
- [Model Editing with Canonical Examples](https://arxiv.org/abs/2402.06155): We introduce model editing with canonical examples, a setting in which (1) a single learning example is provided per desired behavior, (2) evaluation is performed exclusively out-of-distribution, and ...
- [An Exponential Learning Rate Schedule for Deep Learning](https://arxiv.org/abs/1910.07454): Intriguing empirical evidence exists that deep learning can work well with exoticschedules for varying the learning rate. This paper suggests that the phenomenon may be due to Batch Normalization or B...
- [Nonlinear computation in deep linear networks](https://openai.com/research/nonlinear-computation-in-deep-linear-networks): no description found
- [Feedback Loops With Language Models Drive In-Context Reward Hacking](https://arxiv.org/abs/2402.06627): Language models influence the external world: they query APIs that read and write to web pages, generate content that shapes human behavior, and run system commands as autonomous agents. These interac...
- [Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks](https://arxiv.org/abs/2402.04248): State-space models (SSMs), such as Mamba Gu &amp; Dao (2034), have been proposed as alternatives to Transformer networks in language modeling, by incorporating gating, convolutions, and input-dependen...
- [Suppressing Pink Elephants with Direct Principle Feedback](https://arxiv.org/abs/2402.07896): Existing methods for controlling language models, such as RLHF and Constitutional AI, involve determining which LLM behaviors are desirable and training them into a language model. However, in many ca...
- [Improving Black-box Robustness with In-Context Rewriting](https://arxiv.org/abs/2402.08225): Machine learning models often excel on in-distribution (ID) data but struggle with unseen out-of-distribution (OOD) inputs. Most techniques for improving OOD robustness are not applicable to settings ...
- [Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models](https://arxiv.org/abs/2402.07865): Visually-conditioned language models (VLMs) have seen growing adoption in applications such as visual dialogue, scene understanding, and robotic task planning; adoption that has fueled a wealth of new...
- [Tweet from Nature Reviews Physics (@NatRevPhys)](https://fxtwitter.com/NatRevPhys/status/1757089166683230242): Perspective: Generative learning for nonlinear dynamics  By @wgilpin0 @TexasScience  https://rdcu.be/dysiB
- [Tweet from Hannes Stärk (@HannesStaerk)](https://fxtwitter.com/HannesStaerk/status/1695943729314746410): Diffusion models are dead - long live joint conditional flow matching! 🙃 Tomorrow @AlexanderTong7 presents his &#34;Improving and generalizing flow-based generative models with minibatch optimal tran...
- [A weight matrix in a neural network tries to break symmetry and fails.](https://www.youtube.com/watch?v=kGAjhkm4wnY): We initialize a neural network so that the weight matrices can be nearly factorized as the Kronecker product of a random matrix and the matrix where all of t...
- [Mixture of Tokens: Efficient LLMs through Cross-Example Aggregation](https://arxiv.org/abs/2310.15961): Despite the promise of Mixture of Experts (MoE) models in increasing parameter counts of Transformer models while maintaining training and inference costs, their application carries notable drawbacks....
- [llm-random/research/conditional/moe_layers/expert_choice.py at ad41b940c3fbf004a1230c1686502fd3a3a79032 · llm-random/llm-random](https://github.com/llm-random/llm-random/blob/ad41b940c3fbf004a1230c1686502fd3a3a79032/research/conditional/moe_layers/expert_choice.py#L59): Contribute to llm-random/llm-random development by creating an account on GitHub.
- [An Emulator for Fine-Tuning Large Language Models using Small Language Models](https://arxiv.org/abs/2310.12962): Widely used language models (LMs) are typically built by scaling up a two-stage training pipeline: a pre-training stage that uses a very large, diverse dataset of text and a fine-tuning (sometimes, &#...
- [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450): Pre-training and fine-tuning, e.g., BERT, have achieved great success in language understanding by transferring knowledge from rich-resource pre-training task to the low/zero-resource downstream tasks...
- [Meta- (out-of-context) learning in neural networks](https://arxiv.org/abs/2310.15047): Brown et al. (2020) famously introduced the phenomenon of in-context learning in large language models (LLMs). We establish the existence of a phenomenon we call meta-out-of-context learning (meta-OCL...
- [Secret Collusion Among Generative AI Agents](https://arxiv.org/abs/2402.07510): Recent capability increases in large language models (LLMs) open up applications in which teams of communicating generative AI agents solve joint tasks. This poses privacy and security challenges conc...
- [Portal](https://portal.valencelabs.com/logg): Home of the TechBio community. Tune into our weekly reading groups (M2D2, LoGG, CARE), read community blogs, and join the discussion forum. 
- [Generative learning for nonlinear dynamics | Nature Reviews Physics](https://www.nature.com/articles/s42254-024-00688-2.epdf?sharing_token=D_ImKvUZsRHYzs0lhT-4hNRgN0jAjWel9jnR3ZoTv0OFpVCe5j8bo6KJ1K_rllqrEXyt3r74B4sNMsFSoYzk3qrjVQZAFqeWPvf0ZTRuVS6GZQhz83MTvZr0nlCnrXj25-QPv4XzGPY-Homhk29UsvbEDaEd1lFW8i_n6jM6_1w%3D): no description found
- [Policy Improvement using Language Feedback Models](https://arxiv.org/abs/2402.07876): We introduce Language Feedback Models (LFMs) that identify desirable behaviour - actions that help achieve tasks specified in the instruction - for imitation learning in instruction following. To trai...
- [To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis](https://arxiv.org/abs/2305.13230): Recent research has highlighted the importance of dataset size in scaling language models. However, large language models (LLMs) are notoriously token-hungry during pre-training, and high-quality text...
- [UnitY: Two-pass Direct Speech-to-speech Translation with Discrete Units - Meta Research](https://research.facebook.com/publications/unity-direct-speech-to-speech-translation/): We present a novel two-pass direct S2ST architecture, UnitY, which first generates textual representations and predicts discrete acoustic units subsequently.

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1206974168516526100) (1 messages): 

- **In Search of Interpretability Guidance**: `@jaimerv` reached out to the channel asking for a more current overview of approaches to **interpretability** than the paper they referenced on **Representation Engineering**. They are seeking assistance for potentially better or newer resources on the topic.
  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1206511798006841364) (4 messages): 

- **Contributors Wanted for Hallucinations Leaderboard**: `@pminervini` shared a [call to action](https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations) for contributions to the hallucinations leaderboard, adding that there are several new hallucination-oriented tasks to work on within the Harness [leaderboard space](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/tree/main/src/backend/tasks).
- **Enthusiastic Response to Collaboration**: Following the announcement, `@baber_` expressed interest and asked what specific help was needed.
- **Call for Specific Assistance**: In response, `@pminervini` mentioned they need help with task definitions, proposing/adding new datasets and metrics, and assistance in determining which results to re-compute following recent updates to the harness.
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1207072674048909352) (4 messages): 

- **Potential Misalignment in Pythia Deduped Data**: `@pietrolesci` has raised concerns about a possible **misalignment between training data batches and checkpoints** specifically for the **2.8b size Pythia deduped suite**. Other models, including the smaller versions and 6.9b, seem well-aligned.
- **Response to Data Alignment Query**: `@hailey_schoelkopf` acknowledged `@pietrolesci`'s query about the alignment issue and stated they will follow up on this matter.
- **Interest in Pythia Research and Suggestion for Publication**: `@stellaathena` expressed excitement about the potential for a blog post or workshop paper demonstrating the reliability of **Pythia**, which they would extensively cite.
- **Openness to Writing About Pythia**: In response to `@stellaathena`, `@pietrolesci` appreciated the suggestion about creating a post regarding their findings on Pythia, considering it a good short project post-ACL deadline.
  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1206682399816290364) (2 messages): 

- **LlamaIndex v0.10 Released**: `@jerryjliu0` announced the release of **LlamaIndex v0.10**, which is the most significant update to date, featuring a new `llama-index-core` package and splitting integrations/templates into separate PyPi packages. The [`llamahub.ai`](https://llamahub.ai) is also being revamped, they've deprecated ServiceContext for better developer experience, and encourage the community to explore the [blog post](https://blog.llamaindex.ai/llamaindex-v0-10-838e735948f8) and [documentation](https://docs.llamaindex.ai/en/stable/getting_started/installation.html) for detailed info on migration and contributing.
- **Celebrating Team Achievement**: Big thanks were given to `<@334536717648265216>` and `<@908844510807728140>` for leading the effort on the latest LlamaIndex update, which is a step towards making it a production-ready data framework.
- **Tweet about LlamaIndex v0.10 Launch**: LlamaIndex shared a [tweet](https://x.com/llama_index/status/1757121818115322076?s=20) highlighting key updates in LlamaIndex v0.10, including the creation of hundreds of separate PyPi packages, the refactoring of LlamaHub, and the deprecation of ServiceContext.
- **Webinar Announcement with No-Code RAG Tutorial**: Flowise's co-founder, Henry Heng, will feature in a LlamaIndex Webinar to demonstrate building no-code Retrieve and Generate (RAG) applications using their new integration with **LlamaIndex.TS**. The webinar is scheduled for Friday 9am PT and interested individuals can [register here](https://lu.ma/ubm3jg3k).

**Links mentioned**:

- [LlamaIndex Webinar: Build No-Code RAG · Zoom · Luma](https://lu.ma/ubm3jg3k): Flowise is one of the leading no-code tools for building LLM-powered workflows. Instead of learning how to code in a framework / programming language, users can drag and drop the components...
- [Tweet from LlamaIndex 🦙 (@llama_index)](https://x.com/llama_index/status/1757121818115322076?s=20): 💫 LlamaIndex v0.10 💫 - our biggest open-source release to date, and a massive step towards production-readiness. 🚀  ✅ Create a core package, split off every integration/template into separate PyPi ...
- [LlamaIndex v0.10](https://blog.llamaindex.ai/llamaindex-v0-10-838e735948f8): Today we’re excited to launch LlamaIndex v0.10.0. It is by far the biggest update to our Python package to date (see this gargantuan PR)…

  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1206681508710973481) (5 messages): 

- **LlamaIndex Hits v0.10 Milestone**: LlamaIndex announces its biggest open-source release, v0.10, signaling a shift towards production-readiness. A core package has been created and hundreds of integrations split off into separate PyPi packages as highlighted in their [Twitter post](https://twitter.com/llama_index/status/1757121818115322076).

- **Tutorial on Multimodal Apps with LlamaIndex**: `@ollama` and LlamaIndex co-present a tutorial for building context-augmented multimodal applications on a MacBook, including smart receipt reading and product image augmentation, shared via [this tweet](https://twitter.com/llama_index/status/1757211083130151253).

- **DanswerAI Enhances Enterprise with LlamaIndex**: DanswerAI leverages `@llama_index` to offer ChatGPT functionalities over enterprise knowledge bases, integrating with common workplace tools such as GDrive, Slack, and Jira to boost team efficiency as announced in the [Twitter announcement](https://twitter.com/llama_index/status/1757453320829251755).

- **Upcoming No-Code RAG Webinar with FlowiseAI**: `@llama_index` teams up with `@FlowiseAI` for a webinar on building no-code RAG (Retrieval-Augmented Generation) workflows with LlamaIndex.TS and Flowise, details in their [recent tweet](https://twitter.com/llama_index/status/1757455162988540329).

- **Define Research Workflow with RAG-powered Agent**: A notebook by `@quantoceanli` outlines a process to establish a scientific research workflow, harnessing LlamaIndex to operate with resources like ArXiv and Wikipedia for an innovative RAG-powered agent, showcased in [this tweet](https://twitter.com/llama_index/status/1757579982879260891).
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1206556069627830342) (303 messages🔥🔥): 

- **LlamaIndex Import Troubles**: Users like `@ddashed`, `@bhrdwj`, `@lhc1921`, and `@cheesyfishes` discuss issues with the latest LlamaIndex update. Users were advised to start with a fresh venv or container and pointed towards a migration guide and package registry for reference.

- **Complex Document Filtering Challenges**: User `@_shrigmamale` sought assistance in filtering large directories of complex documents based on keywords, dates, and file types. Another user, `@qingsongyao`, suggested traditional indexing techniques over expensive LLMs like GPT-4 for dynamic file filtering.

- **Efficient Handling of Multiple Document Sources**: Users like `@nvmm_`, `@whitefang_jr`, and `@.saitej` engaged in discussions about handling and merging private user-uploaded documents with public indexed documents using LlamaIndex and the potential for creating multiple agents for individual documents.

- **Configuring Chunk Sizes and Testing Performance**: `@sgaseretto` asked about where to specify `chunk_size` now that `ServiceContext` is deprecated in favor of `Settings`. `@cheesyfishes` provided the new way to configure chunk size globally or by passing the node parser/text splitter into the index.

- **Handling Changes with Chat Memory Buffer**: `@benzen.vn` inquired about experiencing non-relevant responses when using a `ChatMemoryBuffer`. `@whitefang_jr` suggested that off-topic conversations might degrade the relevancy of queries and pointed to parts of the LlamaIndex source code for explanation.

**Links mentioned**:

- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://pretty-sodium-5e0.notion.site/ce81b247649a44e4b6b35dfb24af28a6?v=53b3c2ced7bb4c9996b81b83c9f01139): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
- [Response Modes - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes.html): no description found
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://pretty-sodium-5e0.notion.site/v0-10-0-Migration-Guide-6ede431dcb8841b09ea171e7f133bd77): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
- [Google Colaboratory](https://colab.research.google.com/drive/1txHpWXnDbZ12YkzB-ytmPzG2pHY62nbD?usp=sharing): no description found
- [Build a chatbot with custom data sources, powered by LlamaIndex](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/): Augment any LLM with your own data in 43 lines of code!
- [Router Query Engine - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine.html): no description found
- [Elasticsearch Vector Store - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/examples/vector_stores/ElasticsearchIndexDemo.html): no description found
- [llama_index/llama-index-legacy/llama_index/legacy/vector_stores/mongodb.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama-index-legacy/llama_index/legacy/vector_stores/mongodb.py#L160-L183): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/chat_engine/condense_question.py at 3823389e3f91cab47b72e2cc2814826db9f98e32 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/3823389e3f91cab47b72e2cc2814826db9f98e32/llama-index-core/llama_index/core/chat_engine/condense_question.py#L177): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [Usage Pattern - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/usage_pattern.html#subclassing-a-customquerycomponent): no description found
- [Node Postprocessor Modules - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors.html#similaritypostprocessor): no description found
- [llama_index/llama-index-core/llama_index/core/indices/base.py at 5d557cb2fe48b90e4056ecae25b9371681752a3c · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/5d557cb2fe48b90e4056ecae25b9371681752a3c/llama-index-core/llama_index/core/indices/base.py#L426): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [Configuring Settings - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings.html): no description found
- [Migrating from ServiceContext to Settings - LlamaIndex 🦙 v0.10.3](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration.html): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1206605324899983492) (1 messages): 

- **Super-Easy Full Stack RAG App Building Guide Released**: `@kerinin` has shared an article about building a Retrieval-Augmented Generation (RAG) application using [Dewy](https://dewykb.github.io/), a new open-source knowledge base. The guide entails using [NextJS](https://nextjs.org/), [OpenAI API](https://platform.openai.com/), and Dewy to create a RAG application that improves the accuracy of language model responses by grounding them in specific, reliable information. [Read the guide](https://dewykb.github.io/blog/rag-app-with-nextjs-openai-and-dewy/).

**Links mentioned**:

[Building a RAG chatbot with NextJS, OpenAI &amp; Dewy | Dewy](https://dewykb.github.io/blog/rag-app-with-nextjs-openai-and-dewy/): This guide will walk you through building a RAG application using NextJS for the web framework, the OpenAI API for the language model, and Dewy as your knowledge base.

  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1207064103118704671) (1 messages): 

- **Hugging Face Launches Message API**: 🚀 Hugging Face introduces a new Message API compatible with OpenAI, enabling the use of OpenAI client libraries or third-party tools directly with Hugging Face Inference Endpoints and Text Generation Inference. Learn more from their announcement [here](https://twitter.com/_philschmid/status/1755592500511997980).

- **New Open Source Releases and Features**: 🤗 Datatrove goes live on PyPI, Gradio updates to 4.18.0 with an improved `ChatInterface` and more, and there's a launch of Remove Background Web for in-browser background removal. Additionally, Nanotron for 3D parallelism training and new features in Hugging Face Competitions were announced. Accelerate 0.27.0 was released, boasting a PyTorch-native pipeline-parallel inference framework.

- **Product Innovations at Hugging Face**: HF introduces LoRA Studio with a dedicated UI on the Hub, incorporates 2FA support, releases a Mask Generation task page, and announces the arrival of models trained with Axolotl.

- **Partnerships and Learning Resources Expansion**: Hugging Face announces a partnership with Codecademy for a new free AI course on transformers and publishes a blog post about SegMoE, which enables model merging on text-to-image models.

- **Optimizing Model Performance**: There's a technique to load pre-trained PyTorch models approximately 2x faster using Accelerate, detailed in a user guide by `@RisingSayak`.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1205128865735770142): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Releases · gradio-app/gradio](https://github.com/gradio-app/gradio/releases): Build and share delightful machine learning apps, all in Python. 🌟 Star to support our work! - gradio-app/gradio
- [Tweet from Nouamane Tazi (@Nouamanetazi)](https://x.com/Nouamanetazi/status/1755607253087097207): Super happy to see https://github.com/huggingface/nanotron released today! ❤️  It&#39;s been a fun and insightful ride building a library for 3D parallelism training from scratch, and it&#39;s crazy t...
- [Tweet from Zach Mueller (@TheZachMueller)](https://x.com/TheZachMueller/status/1755993747232305468): Today is an extra-special release of @huggingface Accelerate!  Among other features, this latest version (with collaboration from @PyTorch) integrates a PyTorch-native pipeline-parallel inference fram...
- [Tweet from Omar Sanseviero (@osanseviero)](https://x.com/osanseviero/status/1757321643054006330): Over 300 models have been trained with axolotl and shared on the Hub! It&#39;s also the cutest icon ever.  https://huggingface.co/models?other=axolotl&sort=trending
- [Tweet from Sayak Paul (@RisingSayak)](https://x.com/RisingSayak/status/1755804957989539977): Why should LLM kids have all the fun from model merging? Why not us, the diffusion kids?   Friends from @_segmind open-sourced SegMoE to reduce this gap 🔥  Do MoE style merging on text-to-image model...
- [Tweet from Sayak Paul (@RisingSayak)](https://x.com/RisingSayak/status/1756634311493890559): 🤗 Accelerate power-user chronicles 👨‍🏫  Here, I show you how to load a pre-trained PyTorch model ~2x faster with Accelerate. The comments in the code snippet should be self-explanatory.   But if yo...

  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1206512305739669505) (192 messages🔥🔥): 

```html
<ul>
  <li><strong>Search Engine Development Struggles</strong>: <code>@spidy___</code> discussed challenges in developing a search engine and extracting keywords with <code>@vipitis</code>, <code>@cubietom</code>, and others. The conversation explored the limitations of NER and alternatives like keyword extraction, TF-IDF, BM25, and the use of spaCy for Part of Speech tagging.</li>
  <li><strong>Hosting and Inferencing Challenges</strong>: Users like <code>@sullynaj</code> and <code>@ram1428</code> enquired about hosting custom models and whether serverless inferencing is available, with pointers to server-less or affordable solutions discussed.</li>
  <li><strong>Tackling Model Scale</strong>: Conversations with users like <code>@zorian_93363</code> and <code>@xacer_</code> revolved around the feasibility and usefulness of running very large models (100B+ parameters) on typical "open source enthusiast" hardware.</li>
  <li><strong>Valentine's Day Vibes</strong>: <code>@not_lain</code> spread love and joy on Valentine's Day, encouraging the community to hug their loved ones.</li>
  <li><strong>Discussion on Running Models Locally</strong>: <code>@aj_0003</code> asked about running machine learning models locally while <code>@pierrunoyt</code> discussed using Hugging Face to clone and run a model.</li>
</ul>
```

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1206246780950544405): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Stable Cascade - a Hugging Face Space by multimodalart](https://huggingface.co/spaces/multimodalart/stable-cascade): no description found
- [Custom architectures with HuggingFace 🤗](https://huggingface.co/blog/not-lain/custom-architectures-with-huggingface): no description found
- [lamm-mit/x-lora · Hugging Face](https://huggingface.co/lamm-mit/x-lora): no description found
- [jinaai/jina-embeddings-v2-base-code · Hugging Face](https://huggingface.co/jinaai/jina-embeddings-v2-base-code): no description found
- [Norm/nougat-latex-base · Hugging Face](https://huggingface.co/Norm/nougat-latex-base): no description found
- [NVIDIA Chat With RTX](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/): Your Personalized AI Chatbot.
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending): no description found
- [Hugtrip GIF - Hugtrip - Discover &amp; Share GIFs](https://tenor.com/view/hugtrip-gif-2490966530865073004): Click to view the GIF
- [Hands-on - Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit7/hands-on): no description found
- [Linguistic Features · spaCy Usage Documentation](https://spacy.io/usage/linguistic-features#pos-taggin): spaCy is a free open-source library for Natural Language Processing in Python. It features NER, POS tagging, dependency parsing, word vectors and more.
- [Linguistic Features · spaCy Usage Documentation](https://spacy.io/usage/linguistic-features#pos-tagging): spaCy is a free open-source library for Natural Language Processing in Python. It features NER, POS tagging, dependency parsing, word vectors and more.
- [
Hugging Face status
](https://status.huggingface.co/)): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1206603149129949235) (9 messages🔥): 

- **Simple Chatbot Development Blueprint**: `@wilbert.comicho` is looking to create a simple chatbot to gather five specific details from a user and send them via email. They are seeking a template to handle database querying, user prompting/saving data and calling an API for email sending.
  
- **AutoGen as a Starting Point**: `@dwb7737` suggested using Microsoft's **AutoGen** for chatbot development and pointed to GitHub for detailed [use cases and Jupyter Notebooks](https://github.com/microsoft/autogen/tree/main/notebook). Additionally, highlighted that OpenAI is preferable to open-source LLMs when utilizing AutoGen.

- **Starting Small with AutoGen Studio**: In a follow-up, `@dwb7737` recommends getting to grips with the basics before diving into **AutoGen Studio** due to possible behavioral discrepancies and bugs, advocating for an understanding of the underlying processes. They provided a [link to AutoGen Studio samples](https://github.com/microsoft/autogen/tree/main/samples/apps/autogen-studio).

- **wilbert.comicho**: Confirms they will be checking out the recommended resources.

- **Video Guide for Ollama Models**: `@dwb7737` shared a [YouTube video](https://www.youtube.com/watch?v=9NJ196KlAE0) as an excellent resource for learning how to use **Ollama open source models** in conjunction with LangChain and Autogen. 

- **Google Sheets Merge Pitfalls**: `@lunarflu` is engaged in merging two Google Sheets and cautions the importance of handling duplicate records and maintaining unique records to prevent issues.

- **Creating Transformers with FP8**: `@neuralink` has progressed to mastering 99% of doremi reproduction and have advanced their training with end-to-end FP8 in 3D parallelism.

- **Switching from AI to Academia**: `@sardarkhan_` shares their shift from reading about diffusors and transformers to focusing on their upcoming mid-semester exams.

**Links mentioned**:

- [autogen/samples/apps/autogen-studio at main · microsoft/autogen](https://github.com/microsoft/autogen/tree/main/samples/apps/autogen-studio): Enable Next-Gen Large Language Model Applications. Join our Discord: https://discord.gg/pAbnFJrkgZ - microsoft/autogen
- [autogen/notebook at main · microsoft/autogen](https://github.com/microsoft/autogen/tree/main/notebook): Enable Next-Gen Large Language Model Applications. Join our Discord: https://discord.gg/pAbnFJrkgZ - microsoft/autogen
- [Ollama - Libraries, Vision and Updates](https://www.youtube.com/watch?v=9NJ196KlAE0): Ollama Libraries: https://ollama.com/blog/python-javascript-librariesOllama Vision models: https://ollama.com/blog/vision-modelsOllama OpenAI API: https://ol...

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1206697071223246858) (8 messages🔥): 

- **Back to ML After a Hiatus**: `@charlweed` is diving back into Machine Learning by working on a **GIMP 3.0 plugin** that connects to Automatic1111, currently facing challenges with posting image data for Image2Image functionality via API.
- **Digging into the Dirt for Energy**: `@Gordo Stoli` shared a research study on [Soil Battery](https://dl.acm.org/doi/10.1145/3631410), a potential advancement in energy technology.
- **MoE Security Vulnerabilities Exposed**: `@osanseviero` introduced a [paper](https://huggingface.co/papers/2402.05526) demonstrating how **Mixture of Experts (MoE)** models are susceptible to adversarial attacks affecting the outputs of benign queries.
- **Understanding MoE Risks and Mitigations**: `@osanseviero` also wrote detailed notes on potential mitigation strategies for the vulnerabilities described in the DeepMind paper, suggesting batch order randomization among other methods, available [here](https://huggingface.co/posts/osanseviero/980907000007376).
- **Questions about MoE's Future Stability**: `@meatfucker` highlighted the potential future threat of the reported MoE attack strategy and considered the implications for systems using large batches, which may inadvertently affect output quality.

**Links mentioned**:

- [@osanseviero on Hugging Face: &quot;Mixture of experts: beware 🛡️⚔️

New paper by DeepMind: Buffer Overflow in…&quot;](https://huggingface.co/posts/osanseviero/980907000007376): no description found
- [Paper page - Buffer Overflow in Mixture of Experts](https://huggingface.co/papers/2402.05526): no description found

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1206542028788203600) (10 messages🔥): 

- **Quiz Generation Anticipation**: `@lunarflu` suggested the addition of a loading screen or bar for the quiz generation process, mentioning an issue with just waiting for the quiz to appear without any indication.
- **Automated Image Tagging Model Deployed**: `@not_lain` announced an automated model for tagging images pertinent to diffusion tasks and gave instructions for use, along with a [link to their discussion](https://huggingface.co/p1atdev/siglip-tagger-test-3/discussions/1). They also mentioned the model's implementation improvements in `refs/pr/2`.
- **Model Supports Various Image Formats**: `@not_lain` highlighted that their tagging model accepts input as a string (path), a PIL image, or a numpy array, showcasing flexibility in handling images.
- **AI for Anime Data Set**: `@not_lain` expressed intentions to use their image tagging model to annotate an anime dataset, while `@imcoza1915` commented on the coolness of the tool.
- **New "Remix" Mode for Image Transformation**: `@matthieulc` shared an update to [PanoramAI.xyz](https://www.panoramai.xyz/), introducing a "remix" mode with `ControlNet` technology for better structure preservation in image transformations. Users are reminded they can navigate the tool using arrow keys.
- **From Sketch to Fashion with AI**: `@tony_assi` unveiled a project [Sketch to Fashion Design](https://huggingface.co/spaces/tonyassi/sketch-to-fashion-design) with great pride, which has received positive feedback as an AI able to understand designs, as `@chad_in_the_house` implied.

**Links mentioned**:

- [panoramai](https://www.panoramai.xyz/): what&#x27;s in your world?
- [Sketch To Fashion Design - a Hugging Face Space by tonyassi](https://huggingface.co/spaces/tonyassi/sketch-to-fashion-design): no description found
- [p1atdev/siglip-tagger-test-3 · Upload folder using huggingface_hub](https://huggingface.co/p1atdev/siglip-tagger-test-3/discussions/1): no description found

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1206574551119695872) (32 messages🔥): 

- **S4 Architecture Gets Annotated**: `@ericauld` shared a resource on "The Annotated S4" asking for feedback and pointing out its usefulness for understanding the S4 architecture, which excels in modeling very long-range sequence tasks. They indicated that reading it may help clarify the model before their upcoming talk on Mamba/S4.
- **Seeking Clarity on S4 Implementation**: `@austintb.` expressed desire for clarification on the S4 architecture's implementation and computational complexity details. `@chad_in_the_house` echoed the sentiment, requesting intuitive explanation of concepts and prior work such as the hippo codebase, later suggesting a focus on intuition and coding for ericauld's main talk.
- **Mamba/S4 Talk Schedule and Content Preferences**: `@ericauld` proposed scheduling the Mamba/S4 talk for Friday at 10am California time and suggested potential content for the primary and secondary (math-focused) sessions based on community feedback.
- **LangTest Paper Makes Its Debut**: `@prikfy` announced the publication of their LangTest paper in the Software Impacts journal, a tool for testing and augmenting NLP models. The paper and the GitHub repository for LangTest were shared, with `@ryzxl` contributing further context on its comprehensive testing capabilities and how to get started using the library.

**Links mentioned**:

- [Structured State Space Models for Deep Sequence Modeling (Albert Gu, CMU)](https://www.youtube.com/watch?v=OpJMn8T7Z34): Date: May 26, 2023(Sorry that the first 2 slides are not recorded, those are motivation slides though.)Abstract: This talk will cover recent deep neural netw...
- [The Annotated S4](https://srush.github.io/annotated-s4/): no description found
- [GitHub - JohnSnowLabs/langtest: Deliver safe &amp; effective language models](https://github.com/JohnSnowLabs/langtest): Deliver safe &amp; effective language models. Contribute to JohnSnowLabs/langtest development by creating an account on GitHub.
- [LangTest | Deliver Safe & Effective Models | John Snow Labs](https://langtest.org): no description found

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1206675353196699698) (10 messages🔥): 

- **Multi-GPU Training Inquiry**: George is looking for advice on adapting the `train_text_to_image.py` script for **multi-GPU** usage, mentioning previous experience with `nn.DataParallel`.

- **Deployment Options for finetuned models**: `@lokendra_71926` finetuned the **mistarl_7b_gptq** model and is seeking recommendations for a library or platform suitable for fast inference deployment.

- **Success with Stable Cascade**: `@isidentical` asked if anyone achieved good text generation with **stable cascade**, similar to the examples in the readme and confirmed getting 50% success on arbitrary words with the right prompting strategy.

- **HuggingFace's Inference Engine Suggestion**: `@chad_in_the_house` suggested that HuggingFace has an inference engine that could potentially serve for llms deployment and also mentioned that the discussion might be more appropriate in another channel.

- **Terminus Model Anticipation**: `@pseudoterminalx` teased that a new **terminus model** is still in the development phase.
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1206708608646127616) (7 messages): 

- **Hierarchical Image Classification Challenge**: `@cropinky` described the issue of hierarchical image classification and advised that the complication level **depends on the quality and amount of data**. They suggested checking out an [ECCV22 paper](https://arxiv.org/pdf/2207.04873.pdf) and related datasets on paperswithcode for further research.

- **In Search of Gaussian Splats**: `@aeros93` inquired about resources or pre-trained models for creating Gaussian splats from point clouds or images. No specific resources were provided, but `@johko990` redirected the query to another channel that could potentially help.

- **Quest for Multimodal Project Insights**: `@joee2711` is working on a multimodal project and sought clarification on the difference between Q-former / MLP connector and if MLP connectors and adapters are the same. They also expressed an interest in connecting with others working on similar projects.

- **Enhancing Image Retrieval Systems**: User `@femiloye` is developing an image retrieval system akin to person reidentification and is looking for methods to improve match accuracy beyond using model embeddings. They are currently utilizing a custom deit transformer trained with reid loss for this purpose.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1206836354647068682) (4 messages): 

- **Fine-tuning Mistral for Deployment**: `@lokendra_71926` fine-tuned **mistarl_7b_gptq** model on custom data and is seeking recommendations for a library or platform for deployment to achieve faster inference.
  
- **Language Identification with XLM-R**: `@_michaelsh` inquired about how to extract the language from **xlmr** after reading a HuggingFace post which explains that [**XLM-RoBERTa**](https://huggingface.co/docs/transformers/model_doc/xlm-roberta) does not require language tensors to understand the language being used.

- **From Natural Language to Algebraic Representations**: `@_david_valente_` is looking for research or work that has focused on translating natural language into algebraic representations such as LEAN. 

- **Voice Simulation and Language Transformation with Transformers**: `@mentrass` asked about methods to simulate one's voice and alter the language using transformer models.
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1206675353196699698) (10 messages🔥): 

- **Multi-GPU Adaptation Inquiry**: `@George` is looking for an easy way to adapt the `train_text_to_image.py` script for multi-GPU usage, noting past experience with `nn.DataParallel`.
- **Deployment Platform for finetuned model**: `@lokendra_71926` has finetuned the **mistarl_7b_gptq** model and is inquiring about a library or platform for fast inference deployment. `@chad_in_the_house` suggests looking at Hugging Face inference engine for LLMs.
- **Text Generation with Stable Cascade**: `@isidentical` questions whether anyone has been able to achieve text generation with stable cascade as showcased in the model's readme, later confirming a 50% success rate with good prompting.
- **Inference Optimization Discussion Redirected**: `@chad_in_the_house` points out that discussions regarding inference optimization should move to a different channel titled `<#1019883044724822016>`.
- **Anticipation for New Terminus Model**: `@pseudoterminalx` indicates that a new terminus model is currently being developed.
  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1206544642477269012) (3 messages): 

- **DAMO-NLP-SG Releases Vast Long-Context Dataset**: `@giftedgummybee` shared the [LongCorpus-2.5B dataset](https://huggingface.co/datasets/DAMO-NLP-SG/LongCorpus-2.5B) which contains 2.5B tokens collected from various domains for long-context continual pre-training. The dataset's composition is inspired by [Long-Data-Collections](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections), and its selection criteria ensures a low n-gram similarity with the training set to exclude QA and Summarization data.

- **Scaling Models with 'rope' vs 'self-extend'**: `@blackl1ght` highlighted that scaling models with 'self-extend' can preserve coherence better than 'rope scaling', even at larger scaling factors, referring to the implementation in llama.cpp.

- **Ease of 'self-extend' Implementation**: `@blackl1ght` noted the benefits of 'self-extend' including no need for setup, fine-tuning, or extra parameters like those required in the 'gguf configurations' for quants.

**Links mentioned**:

[DAMO-NLP-SG/LongCorpus-2.5B · Datasets at Hugging Face](https://huggingface.co/datasets/DAMO-NLP-SG/LongCorpus-2.5B): no description found

  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1206636684293443614) (8 messages🔥): 

- **Discussing LangGraph Agents' Perseverance**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=fHroHoc26RI) titled "LangGraph Agents Persistence," highlighting that LangGraph agents can be set up to retain their state across interactions.

- **Gemini's Resistance Frustrates Users**: `@llmaniac1000` expressed disappointment with Gemini's frequent refusal tendencies, seeking others' experiences with it. `@n8programs` chimed in, stating it's not amazing and implying GPT-4 outperforms Gemini.

- **Mark Zuckerberg's Image Transformation**: `@nonameusr` shared [a Twitter post](https://vxtwitter.com/pitdesi/status/1757552017042743728) suggesting that Zuckerberg has transitioned from villain to savior in the context of AI and VR.

- **A Touch of Humor with GIFs**: `@error.pdf` reacted to previous discussions using humor by sharing a GIF from Tenor, without providing further commentary or context.

**Links mentioned**:

- [Rock Cat Eyebrow Cat GIF - Rock cat Eyebrow cat Meme - Discover &amp; Share GIFs](https://tenor.com/view/rock-cat-eyebrow-cat-meme-sus-dwayne-johnson-gif-14343467910353677310): Click to view the GIF
- [LangGraph Agents Persistence](https://www.youtube.com/watch?v=fHroHoc26RI): When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple ...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1206551533005709402) (17 messages🔥): 

- **Mesmerizing Mandelbrot Beauty Shared**: `@gabriel_syme` posted a [stunning visualization](https://twitter.com/jaschasd/status/1756930242965606582) of the Mandelbrot set. `@_3sphere` added that the set's focus on divergence contributes to its sense **of complexity and order**.
  
- **Crowdsourcing AI with 'Marv' Chatbot**: `@.dvs13` praised a [crowdsourcing project](https://huggingface.co/posts/dvilasuero/680660181190026) and noted ambiguity in the term "prompt." The project involves a chatbot named Marv, which answers questions with sarcasm.

- **Reka Introduces Multi-Modal AI Models**: `@metaldragon01` highlighted the launch of **Reka Flash**, a **21B fast multimodal language model**, alongside its smaller counterpart **Reka Edge**. [Reka Flash](https://reka.ai/reka-flash-an-efficient-and-capable-multimodal-language-model/) boasts competitive performance to major models like Gemini Pro and GPT-3.5 and is available in public beta.

- **Pursuing CUDA Compatibility with AMD**: `@leontello` shared a [GitHub project, ZLUDA](https://github.com/vosen/ZLUDA), which aims to run CUDA on AMD GPUs. Unfortunately, the project is no longer actively pursued as detailed by `@adjectiveallison`, who quoted the project's lead expressing it's effectively abandoned.

- **Wavelets Meets Transformers in AI Research**: An arXiv paper shared by `@euclaise` suggests that wavelet transforms could enhance Transformers by capturing both positional and frequency information with linear complexity. The paper details Wavelet Space Attention (WavSpA) and has been tested on the Long Range Arena. [Find the paper here](https://arxiv.org/abs/2210.01989).

**Links mentioned**:

- [@dvilasuero on Hugging Face: &quot;🤗 Data is better together!Data is essential for training good AI systems.…&quot;](https://huggingface.co/posts/dvilasuero/680660181190026): no description found
- [Reka Flash: An Efficient and Capable Multimodal Language Model - Reka AI](https://reka.ai/reka-flash-an-efficient-and-capable-multimodal-language-model/): Reka Flash is a state-of-the-art 21B model trained entirely from scratch and pushed to its absolute limits. It serves as the “turbo-class” offering in our lineup of models.
- [WavSpA: Wavelet Space Attention for Boosting Transformers&#39; Long Sequence Learning Ability](https://arxiv.org/abs/2210.01989): Transformer and its variants are fundamental neural architectures in deep learning. Recent works show that learning attention in the Fourier space can improve the long sequence learning capability of ...
- [GitHub - vosen/ZLUDA: CUDA on AMD GPUs](https://github.com/vosen/ZLUDA): CUDA on AMD GPUs. Contribute to vosen/ZLUDA development by creating an account on GitHub.
- [GitHub - acorn-io/rubra: AI Assistants, LLMs and tools made easy](https://github.com/acorn-io/rubra): AI Assistants, LLMs and tools made easy. Contribute to acorn-io/rubra development by creating an account on GitHub.

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1206684427418345562) (180 messages🔥🔥): 

- **New Model Training Begins**: `@n8programs` excitedly shares the start of training a new model, mentioning terms like **dachshund**, **neuralbeagle-dpo**, and expressing the process as *randomly throwing stuff together genetic algorithm-style*.
- **Playful Banter About Model Merging**: `@teknium` humorously notes the metaphorical alignment between dog breeds and model merging, while `@leontello` likens the mixing methods to evolutionary strategies, and `@n8programs` reports a horrifying outcome of his merging experiment.
- **Typo Alert in Model Card**: `@everyoneisgross` reports a typo in Hugging Face's model card for 70B llama, which was swiftly corrected by `@teknium`, leading to expressions of congratulations on the model launch.
- **Quantization Quest**: Discussion about post-training quantization methods, with `@stellaathena` sharing a [link to a new quantization method](https://arxiv.org/abs/2402.04396), and `@nruaif` jokingly looking forward to even lower bit-precision.
- **AI Activation Additions**: A deep dive into activation hacking is mentioned, with `@filipvv` referencing an [external article](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector) and `@mihai4256` discussing their plans to refine their approach, while `@proprietary` voices interest in the work.

**Links mentioned**:

- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396): Post-training quantization (PTQ) reduces the memory footprint of LLMs by quantizing their weights to low-precision. In this work, we introduce QuIP#, a weight-only PTQ method that achieves state-of-th...
- [Tweet from jf (@fejo_11)](https://x.com/fejo_11/status/1757417292659310675?s=46): Mixtral 8x7B: Routing Analysis based on POS tags  I conducted a routing analysis using @MistralAI&#39;s Mixtral 8x7B model, focusing on Part-of-Speech (POS) tags, diverging from the original methodolo...
- [NousResearch/Nous-Hermes-2-Llama-2-70B · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Llama-2-70B): no description found
- [Representation Engineering Mistral-7B an Acid Trip](https://vgel.me/posts/representation-engineering/): no description found
- [OpenAI Researcher Andrej Karpathy Departs](https://www.theinformation.com/articles/openai-researcher-andrej-karpathy-departs): Andrej Karpathy, one of the founding members of OpenAI, has left the company, a spokesperson confirmed. Karpathy, a prominent artificial intelligence researcher, was developing a product he has descri...
- [Xigmoid: An Approach to Improve the Gating Mechanism of RNN](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9892346): This work proposes an innovative approach for the gating mechanism of RNN class models. A transfer function is embedded into the original sigmoid to form a new gate function called xigmoid. The purpos...
- [[missing post]](https://www.greaterwrong.com/posts/5spBu): no description found
- [Diffusion Language Models Can Perform Many Tasks with Scaling and Instruction-Finetuning](https://arxiv.org/abs/2308.12219): The recent surge of generative AI has been fueled by the generative power of diffusion probabilistic models and the scalable capabilities of large language models. Despite their potential, it remains ...
- [‎Practical AI: Machine Learning, Data Science on Apple Podcasts](https://podcasts.apple.com/gb/podcast/data-synthesis-for-sota-llms/id1406537385?i=1000644406332>.): ‎Technology · 2024
- [Steering GPT-2-XL by adding an activation vector](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector>): Summary: We demonstrate a new scalable way of interacting with language models: adding certain activation vectors into forward passes.[2] Essentially, we add together combinations of forward passes in...

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1206532708512301118) (38 messages🔥): 

- **DeepSeekMath Merges into the Conversation**: User `@yxzwayne` inquired about the integration of newly introduced deepseekMath in merging strategies, indicating interest in its application.

- **Finetuning for Dummies Guide Discovered**: `@nemoia` was searching for straightforward instructions on how to finetune Mistral 7B and create their own datasets and later shared a helpful [Medium guide](https://medium.com/@geronimo7/finetuning-llama2-mistral-945f9c200611) that provides detailed examples and explanations on the process.

- **Forced FA2 Line Causes Memory Issues**: In response to a question about FA2 not being enabled, `@bloc97` clarified that the problem was related to an attempt to create a large `attn_weights` matrix, indicating the line of code causing memory issues can be seen [here](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k/blob/main/modeling_mistral_yarn.py#L538).

- **Secondary Options for Coding Models**: `@natefyi_30842` was looking for a less expensive alternative to GPT-4 for a coding model, and `@teknium` suggested trying out the deepseek coder, which is hosted by "together."

- **MIQU Model's Pretraining and SFT Clarified**: `@teknium` explained to `@yxzwayne` that the MIQU model was first pretrained on the Llama-2 70b and then underwent SFT (Supervised Fine Tuning), focusing specifically on instruction-focused data.

**Links mentioned**:

- [Tweet from AMD Quietly Funded A Drop-In CUDA Implementation Built On ROCm: It's Now Open-Source - Phoronix](https://www.phoronix.com/review/radeon-cuda-zluda): no description found
- [Finetuning Llama 2 and Mistral](https://medium.com/@geronimo7/finetuning-llama2-mistral-945f9c200611): A beginner’s guide to finetuning LLMs with QLoRA
- [Training a causal language model from scratch - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/en/chapter7/6): no description found
- [modeling_mistral_yarn.py · NousResearch/Yarn-Mistral-7b-128k at main](https://huggingface.co/NousResearch/Yarn-Mistral-7b-128k/blob/main/modeling_mistral_yarn.py#L538): no description found

  

---


### Nous Research AI ▷ #[collective-cognition](https://discord.com/channels/1053877538025386074/1154961277748256831/1206982859089387562) (3 messages): 

- **Project Downfall Due to Chat GPT Update**: `@adjectiveallison` inquired if a project was still active after encountering issues accessing the site. `@teknium` responded, clarifying that the website broke due to the new **Chat GPT update with various modes**, leading to the original team being unable to maintain it.
- **Sympathies for the Broken Project**: `@adjectiveallison` expressed disappointment upon learning that the project was no longer maintained following the complications with the new **Chat GPT update**.
  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1206583000477732865) (43 messages🔥): 

- **Model Selection Advice for Beginners**: Newcomer `@nana.wav` inquired about the best models to use, and `@afriendofmaurice` recommended **instruct models** for chat-GPT-like interactions. `@mrdragonfox` clarified that **instruct models** are more focused on instruction following, whereas others are akin to raw autocomplete.
- **Integration with Visualization Libraries**: `@carnivore5` asked if anyone had experience integrating Mistral functionalities with **GraphViz** or similar visualization libraries, leading to a clarification by `@mrdragonfox` about Mistral's lack of inherent function-calling ability.
- **Chat vs. Completion Endpoints**: `@i_am_dom` and `@mrdragonfox` discussed the difference between Mistral's `/chat/completion` and a wished-for raw `/completion` endpoint, with most usage currently gravitating towards the chat endpoint.
- **Internship Struggles with Mistral**: `@nana.wav` shared struggles with learning how to use downloaded models and intentions to fine-tune them, leading `@mrdragonfox` to advise starting with simpler steps. The conversation included sympathy and reminiscence from others, highlighting the common intern experience with overwhelming tasks.
- **Mistral API Latency Issues**: `@justinmann.` reported **inconsistent latencies** when using the Mistral API, with response times varying drastically from under a second to over a minute. `@sublimatorniq` suggested contacting support for assistance.
  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1206531277260324954) (20 messages🔥): 

- **RAG Guide for Mistral**: `@ethux` shared a helpful [guide](https://docs.mistral.ai/guides/basic-RAG/) explaining how **Mistral** works with **RAG (Retrieval-Augmented Generation)**, including steps on retrieval and generation with examples from Mistral, LangChain, and LlamaIndex.
- **Debate on LangChain vs. LlamaIndex**: `@sublimatorniq` sparked a discussion on the effectiveness of **LangChain vs. LlamaIndex**, with `@rabdullin` expressing skepticism about their use in serious LLM-driven products.
- **DSPy Advocacy**: `@mrdragonfox` advocated for **DSPy** as a powerful framework, citing that it uses LLM as a "device" and not a "chat" interface and linked to a [Twitter post](https://twitter.com/CShorten30/status/1751656468879708496) exemplifying its strength.
- **Mistral-7b Training Dataset Inquiry**: `@kushagra_67246` inquired about the datasets on which **Mistral-7b** is trained, receiving humorous and vague responses indicating a mixed variety of internet sources — from `@tom_lrd` describing it as "Top secret magic soup" to `@gamerboi0129` listing textbooks and Wikipedia among other comprehensive sources.
- **Clarification on Raw Pretraining Checkpoint**: `@nofreewill42` asked for an open-sourced checkpoint of the **Mistral** model right after raw text pretraining, expressing that `mistralai/Mistral-7B-v0.1` seemed too interactive to be raw.

**Links mentioned**:

- [Basic RAG | Mistral AI Large Language Models](https://docs.mistral.ai/guides/basic-RAG/): Retrieval-augmented generation (RAG) is an AI framework that synergizes the capabilities of LLMs and information retrieval systems. It&#x27;s useful to answer questions or generate content leveraging ...
- [GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models](https://github.com/stanfordnlp/dspy): DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1206525562441437214) (46 messages🔥): 

- **Docker Deployment Recommendations**: `@rusenask` suggested checking out the **ollama** or **vllm** projects for APIs that can be run through Docker for different use cases.
- **Quota Troubles in the Cloud**: `@gridinoc` experienced difficulties deploying **Mixtral** with SkyPilot as AWS, Google Cloud, and Azure either denied quota increases or did not respond to requests.
- **Alternatives to Self-Hosting**: `@mrdragonfox` discussed options for deployment, suggesting cheaper API offerings such as **direct mistral or together.ai**, despite the current GPU shortages and quota issues faced by `@gridinoc`.
- **AWQ Quantization Hitches with MoE**: Multiple users, including `@mrdragonfox` and `@casper_ai`, discussed issues with the **AWQ quantization** method and **Mixtral** models, with `@casper_ai` recommending an alternative working repository hosted on Hugging Face.
- **Success with HuggingFace Deployment**: `@ethux` pointed to an instance of **Mixtral** deployed on **HuggingFace.co/chat**, offering an alternate route to those facing cloud service barriers.

**Links mentioned**:

- [Deploy with SkyPilot | Mistral AI Large Language Models](https://docs.mistral.ai/self-deployment/skypilot/): SkyPilot is a framework for running LLMs, AI, and batch jobs on any cloud, offering maximum cost savings, highest GPU availability, and managed execution.
- [casperhansen/mixtral-instruct-awq · Hugging Face](https://huggingface.co/casperhansen/mixtral-instruct-awq): no description found
- [TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ): no description found
- [HuggingChat](http://huggingface.co/chat): Making the community's best AI chat models available to everyone.
- [TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ · always getting 0 in output](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ/discussions/3): no description found

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1206631013187588096) (76 messages🔥🔥): 

- **Fine-Tuning vs RAG Explained**: Users in the channel debated the merits of fine-tuning versus using Retrieval-Augmented Generation (RAG), with `@rabdullin` advising to focus on prompt engineering and `@mrdragonfox` highlighting the importance of base knowledge in a Large Language Model (LLM) when using RAG. `@tom_lrd` and `@mrdragonfox` outlined that RAG acts as middleware to provide relevant context for the LLM and has its own complex underlying processes.
- **Onboarding the New AI Enthusiast**: In response to `@1mbc` seeking resources for understanding AI core concepts, `@mrdragonfox` and `@tom_lrd` provided insights into how RAG and GPTs work and suggested platforms like Medium for further learning. No specific resources were linked.
- **Chatbot Integration Strategies Shared**: The conversation delved into the technicalities of feeding an LLM with personalized data, with `@mrdragonfox` and `@tom_lrd` describing how data can be pre-processed and turned into a structured format that enriches the LLM's output, specially when using an LLM as a middleware to process user input.
- **Clarifying Misconceptions on LLM Data Storage**: `@mrdragonfox` corrected some misconceptions about how an LLM 'learns' from new data, such as the functions of GPTs and the significant complexity behind embedding and search before data becomes usable context for an LLM.
- **Prompt Versioning Tools Inquiry**: `@khandelwaal.ankit` inquired about tools for prompt versioning during fine-tuning experiments, noting a lack of support for Mistral models in some existing tools like PromptLayer; however, no solutions were specifically endorsed or detailed in the discussion.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1206516506213613618) (2 messages): 

- **Limits on Code Modification**: `@ethux` expressed skepticism about the possibility of making a certain change, suggesting that it **might not be possible** without altering some code.
  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1206886738346254346) (15 messages🔥): 

- **French Librarian Seeks Internship Opportunities for Student**: User `@maeelk`, a French librarian, is promoting AI use and looking for an internship for a student studying **psychology and AI**, referring to the [Master's program at the University of Savoie Mont Blanc](https://formations.univ-smb.fr/fr/catalogue/master-XB/sciences-humaines-et-sociales-SHS/master-psychologie-KGYQCP1D/ergonomie-socio-cognitive-des-systemes-intelligents-classique-et-alternance-KIIPYUGG.html). Interested parties can reach out for collaboration via `c.limousin@cdcba.fr`.
- **Mistral's Fan Quiz**: User `@akshay_1` challenges `@maeelk`'s Mistral fandom by asking them to list the weights of the **7b model**. Another user, `@ethux`, responds humorously, implying the difficulty of listing such technical details.
- **Building Audio-Inclusive S2S Models on a Shoestring Budget**: `@akshay_1` shares a client's request to build an S2S model with a persona, fine-tuned with an audio dataset on a **budget of $1,000**. Several users, like `@ethux` and `@mrdragonfox`, react to the insufficient budget, implying that much more would be required.
- **The Price of Innovation**: `@skadeskoten` inquires about the competitive budget for creating a specialized S2S model, to which `@mrdragonfox` responds that the cost greatly depends on the extent of architecture needed.

**Links mentioned**:

[Ergonomie socio-cognitive des syst&egrave;mes intelligents - Classique et alternance - Ametys Campus - Universit&eacute; Savoie Mont Blanc](https://formations.univ-smb.fr/fr/catalogue/master-XB/sciences-humaines-et-sociales-SHS/master-psychologie-KGYQCP1D/ergonomie-socio-cognitive-des-systemes-intelligents-classique-et-alternance-KIIPYUGG.html): no description found

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1206550405971181589) (2 messages): 

- **API Key Confusion for TypingMind**: `@ingohamm` reported issues with using the API key for TypingMind, despite having a subscription and payment method in place. He mentioned that trying after a wait or deleting the API key prompted a message about no active subscription, and questioned the status of his account or subscription.
- **Seek Support from Mistral**: In response to the issue, `@sublimatorniq` suggested that `@ingohamm` reach out to **support@mistral.ai** for assistance with his API key and subscription concerns.
  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1206558099599663104) (149 messages🔥🔥): 

- **Seeking Support for Company Data Issues**: User `@kitsuiwebster` expressed that sending emails for support got no response and preferred not to disclose the data-related issue publicly. Instead, they wished to contact directly for help with a company-related problem.
- **Debating the Merits of Perplexity vs. Phind**: User `@ludwig_von_mises_fan` opened a discussion about the effectiveness of Phind over Perplexity for coding and general search, while `@gooddawg10` and `@brknclock1215` defended Perplexity's search capabilities, with no preference for coding.
- **Experiencing Technical Difficulties with Perplexity**: Users `@yellephen`, `@luke_____________`, and `@chenlieong` reported issues with the Perplexity chatbot, such as endless loading for answers and service unavailability; `@dima_shliugaev` from the team acknowledged the issue and it was confirmed to be back online by `@vova_at_pplx_ai`.
- **Model Performance and Usage Discussions**: Users shared their experiences with different AI models for tasks such as code debugging (`@matheusgnhr`), tic-tac-toe (`@noremac258`), and PDF reading (`@reader7904`); queries regarding specific model details (`@hzpd` and `@unknownuser787`) and API usage (`@pilotgfx`) were also seen.
- **Subscription Details and Model Information Inquiry**: Users `@stocktown` and `@ewaathescientist` sought clarification on trial subscriptions and the renewal of Pro subscriptions, while `@voidfulness` inquired about token refresh rates and was informed by `@me.lk` that tokens refresh 24 hours after use.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1118264005207793674/1206743956302471168): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649527299055688/1197892547276705843): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [‎What Gemini Apps can do and other frequently asked questions](https://gemini.google.com/faq?gad_source=1&gclid=Cj0KCQiAw6yuBhDrARIsACf94RXwwoXpktZalDwo6OO8RsVYvKAaDpxT1Cr_XIek-8kBnPaZa7Jb5bwaAvsQEALw_wcB): Learn what Gemini can do, how it works, and different ways to get access to it.
- [More than an OpenAI Wrapper: Perplexity Pivots to Open Source](https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/): Perplexity CEO Aravind Srinivas is a big Larry Page fan. However, he thinks he&#039;s found a way to compete not only with Google search, but with OpenAI&#039;s GPT too.
- [What is a Google dork query and how to protect yourself?](https://www.techtarget.com/whatis/definition/Google-dork-query): A Google dork query is a search string using advanced search operators. See how hackers get website data not readily available with it and how to protect from it.
- [Coupons and discounts](https://blog.perplexity.ai/faq/coupons-and-discounts): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Introducing PPLX Online LLMs ](https://blog.perplexity.ai/blog/introducing-pplx-online-llms): The first-of-its-kind Online LLM API

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1206568408460558417) (13 messages🔥): 

- **Perplexity AI Tackles Tough Test Question**: `@tbrams` was *impressed* with how quickly Perplexity AI handled a complex question from the "Gemini" paper, a task that Google's Gemini service and OpenAI took longer to address. Details of this successful test are available on the [Perplexity AI platform](https://www.perplexity.ai/search/Write-code-to-7yAwNS4DTAyB905rHG04eA?s=u).
- **Community Contributions and Creations**: `@twodogseeds` gave a shoutout to Perplexity for the pplx shortcut action, which supports their Farm Friend research agent. No further details were shared in the message.
- **Exploring Diverse Perspectives with Bryan Johnson**: `@ok.alex` shared a link to a summary of Bryan Johnson's perspectives via Perplexity AI, while `@brknclock1215` offered an alternative angle for scientific summarization. Links to these summaries are found at [Bryan Johnson Summary](https://www.perplexity.ai/search/summarize-Bryan-Johnsons-QmNjFaQnRkSaGyimpjCk6A?s=c) and [Scientific Summary](https://www.perplexity.ai/search/summarize-the-scientifically-so2l6.dLT8C9xKubqLB8pQ?s=c) respectively.
- **Engage with the Alt-D-Feed**: `@ok.alex` invited the community to contribute to an alternative feed/newsletter, suggesting it as a collaborative project to curate together. Interested individuals can [like and share this initiative](https://www.perplexity.ai/collections/Alt-D-Feed-x.dZp0_3RAyKoTWkhJW_DA).
- **Summarizing Documents in Seconds!**: `@aykbl` expressed enthusiasm for Perplexity AI's capability to summarize documents swiftly, emphasizing its speed with a smiley face. The content linked or specificity of documents was not mentioned.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1054944216876331118/1206373264100696094): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1206653005303250966) (24 messages🔥): 

- **Custom Search Queries Available**: User `@me.lk` clarified that by using search parameters such as `"site:reddit.com OR site:youtube.com"` in prompts, one can specify multiple content sources when using the API.
- **Performance Issues with Online API**: `@andrewgazelka` reported performance problems with `pplx-70b-online`, but noted that removing the system message in the code seemed to fix the issue.
- **PPLX API Fails with Nonsensical Responses**: `@myadmingushwork_52332` raised a concern with the API returning random and nonsensical replies involving a mix of numbers and characters when online searching is required.
- **Reference Provision Under Development**: `@dvrshil` expressed a desire for Perplexity to provide references in API responses, to which `@mares1317` responded, stating that the development team is working on this feature.
- **No Early Access Program Yet**: `@icelavaman` indicated that early access to new Perplexity features is not available at this moment; announcements for new features will come at a later date.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1161802929053909012/1203354764776046664): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Pricing](https://docs.perplexity.ai/docs/pricing): no description found

  

---



### OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1207041425871609967) (1 messages): 

- **ChatGPT Gets a Memory Boost**: `@abdubs` announced a **new memory feature for ChatGPT** which allows it to **remember user preferences and details** across conversations, thereby enhancing future interactions. This feature is rolling out to select Free and Plus users, with control options available at [ChatGPT Memory Features](https://openai.com/blog/memory-and-new-controls-for-chatgpt). 

- **You're the Boss of ChatGPT's Memory**: Users have the power to **tell ChatGPT what to remember**, ask it to recall information, and instruct it to forget things conversationally or through settings. The memory feature can also be turned off completely if preferred.

- **Memory Feature Rolling Out Gradually**: OpenAI is currently deploying the memory upgrade to a limited user base and plans to gather feedback to gauge its usefulness. Further announcements regarding a broader rollout will be made soon.

**Links mentioned**:

[Memory and new controls for ChatGPT](https://openai.com/blog/memory-and-new-controls-for-chatgpt): We’re testing the ability for ChatGPT to remember things you discuss to make future chats more helpful. You’re in control of ChatGPT’s memory.

  

---


### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1206532274980388916) (83 messages🔥🔥): 

- **Discovering the Secrets of SEO**: `@spidy___` sought insights on how to autonomously tag webpages with relevant keywords like web crawlers do, finding limitations in NER for keyword extraction. `@light.grey.labs` advised examining SEO files, as web builders often embed a variety of keywords into these for search relevance.
  
- **Seeking Creative Minds for AI Research**: `@noodles7584`, a UK researcher, invited community members to discuss how AI is used in creative processes, offering compensation for the 30-minute discussions.
  
- **The Quest for the Ultimate Chatbot**: Chat explored the challenges with current chatbots, including the inability of GPT models to meet all individual needs, voiced by `@jaicraft`. `@lumirix` and others discussed workarounds, like combining bots or leveraging chatbot integrations with services like Google Docs.
  
- **ChatGPT Accused of Laziness**: `@pigondrugs` and others commented on GPT's difficulty retaining context, with growing complaints after context capacity increased. In contrast, `@drinkoblog.weebly.com` argued that higher context limits reduce perplexity, leading to better performance.
  
- **AI Model Rivalry Heats Up**: `@cassofthenight` spotlighted Abacus.AI’s Smaug-72B model outperforming GPT-3.5 and expressed concerns over ChatGPT-4's reluctance to produce complete code snippets, suggesting that the AI dodges detailed scripting in favor of pseudo code.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1206828176672690186): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1206512221450936390) (49 messages🔥): 

- **GPT 4 Turbo Cost Queries Clarified**: `@jeremy.o` directed `@koukyo` to the [OpenAI pricing page](https://help.openai.com/en/articles/7127956-how-much-does-gpt-4-cost) for GPT 4 cost details and noted that GPT 4 Turbo is the top/cheapest model, costing 2 cents less than other versions, with similar or slightly worse quality depending on use.

- **GPT 4 Sometimes Slacks Off?**: `@rodney.leonardo` reported a decrease in GPT 4's intelligence in basic tasks, like summarizing a PDF. Community members including `@blckreaper` confirmed observations of performance issues, and discussions on the topic are collected in a separate channel: <#1047565374645870743>.

- **Still Waiting for @mentions**: Users including `@pax0086` and `@ancryp` discussed the gradual rollout of the @mention feature in GPT, with `@darkninjaforever` reminding that OpenAI often does gradual feature rollouts, indicating some users are still awaiting access.

- **Trying to Push the Boundaries of GPT's Vision**:
  `@flokyhuan` inquired about using videos for fine-tuning language models and was informed by `@solbus` that fine-tuning is currently only available for text models, and while the GPT vision feature can describe images from a video, it can't be fine-tuned for specific knowledge like sports rules.

- **ChatGPT Memory Feature Rollout Progresses**: `@lumirix` confirmed that the ChatGPT's feature for remembering past conversation details is being rolled out to both free and Plus users but noted that it's only available to a small portion of users at this time.


**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1206828176672690186): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Processing and narrating a video with GPT&#x27;s visual capabilities and the TTS API | OpenAI Cookbook](https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding): no description found

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1206521457908121670) (23 messages🔥): 

- **Prompt Engineering Basics Explained**: `@eskcanta` outlined that good prompt engineering involves using precise language, giving clear instructions to the AI, and careful review of the AI's output. When instructing the AI, focus on what to do instead of what not to do, avoiding conflicting instructions.
- **AI Text Adventures Streamlined**: `@drinkoblog.weebly.com` advised `@stealth2077` to use custom instructions like "Focus on simple storytelling and character dynamics" to keep narratives straightforward and avoid complexity, which the AI tends to default to in text adventures.
- **Navigating Platform Confusion**: `@beanz_and_rice` humorously attempted to engage with ChatGPT on the Discord server, prompting `@toror` to respond with amusement at the unsuccessful effort.
- **API Infrastructure vs. Prompt Engineering**: `@darthgustav.` clarified the difference between prompt engineering and API infrastructure to `@kate.yanchenka`, suggesting that the latter's queries about automated budget calculations and dynamic data handling were related to software development rather than prompt engineering.

**Links mentioned**:

- [no title found](https://chat.openai.com>.): no description found
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1037561178286739466): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1206521457908121670) (23 messages🔥): 

- **Newbie Seeks Prompt Engineering Wisdom**: `@zzavior` sought advice for getting started with prompt engineering. `@eskcanta` provided an extensive guide focusing on using **clear and precise language**, checking the output, and ensuring not to trigger conflicts with the AI's capabilities or training.
- **Library Queries for Prompt Engineering**: `@kate.yanchenka` inquired about libraries for prompt engineering to manage budgets, fit dynamic data, and handle AI model fallbacks. `@darthgustav.` clarified that the topic was more about AI software development than prompt engineering.
- **Conversation Assistance Request Goes Unnoticed**: `@beanz_and_rice` attempted to initiate an interaction using Discord slash commands but failed, followed by a comedic outcry that prompted a reaction from `@toror`.
- **Crafting Lightweight Text Adventures**: `@stealth2077` asked for tips on making a text adventure less deep and thematic. `@drinkoblog.weebly.com` suggested using *custom instructions* to guide the AI towards simpler storytelling.
- **Joke Generation Confusion**: `@lisabkk45_48614` requested a joke, but `@solbus` directed them to use the official ChatGPT website instead of the Discord channel.

**Links mentioned**:

- [no title found](https://chat.openai.com>.): no description found
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1037561178286739466): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1206586593591173201) (66 messages🔥🔥): 

- **MPS Support Acknowledgment and Clarification**: `@caseus_` expressed gratitude for MPS support in the axolotl project thanks to a GitHub [pull request #1264](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1264) by Maxime. Confusion arose about the contributor's Discord identity, and `@yamashi` (the right Maxime) clarified their involvement, noting the dependency on transformers merging changes and the [PyTorch pull request #99272](https://github.com/pytorch/pytorch/pull/99272) as crucial for further development.
- **Tinkering with Yi-34b Training and eBay Finds**: Users `@le_mess`, `@cookiesowns`, and `@c.gato` discussed various AI and non-AI topics ranging from slow loss decrease during Yi-34b training to an eBay link for an old tech product.
- **Exploring Model Adaptation and Enhancements**: `@yamashi` suggested the potential benefits of porting models to Keras for wider hardware support, and `@dreamgen` and `@c.gato` discussed error handling and fixes related to Hugging Face checkpoint saving, in light of a [pull request #1414](https://github.com/huggingface/peft/pull/1414) and a related [issue #1452](https://github.com/huggingface/peft/issues/1452).
- **Queries on Cheapest LLM Endpoint Services**: `@le_mess` inquired about affordable LLM endpoint services with responses pointing to local options like llamacpp, external services such as Together AI, and OpenRouter’s cost-effectiveness. Users mentioned JSON serialization issues with Basten and the need for custom configurations.
- **Discussion of Various Challenges Using LLMs**: Issues like JSON serialization (`@dangfutures`), challenges with FP32 slowness (`@yamashi`), and need for additional documentation were discussed providing snapshots of technical hurdles and collaborative problem-solving occurring in the AI community.

**Links mentioned**:

- [Together AI](https://www.together.ai/): Build gen AI models with Together AI. Benefit from the fastest and most cost-efficient tools and infra. Collaborate with our expert AI team that’s dedicated to your success.
- [peft/utils/save_and_load.py try to connect to the hub even when HF_HUB_OFFLINE=1 · Issue #1452 · huggingface/peft](https://github.com/huggingface/peft/issues/1452): System Info peft 0.8.2 axolotl v0.4.0 export HF_DATASETS_OFFLINE=1 export TRANSFORMERS_OFFLINE=1 export HF_HUB_OFFLINE=1 Who can help? No response Information The official example scripts My own mo...
- [Intel® Optane™ Persistent Memory 300 Series (128GB PMem Module) NMC2XXD128GPS  | eBay](https://www.ebay.com/itm/176070887129?): no description found
- [GitHub - triton-inference-server/tensorrtllm_backend: The Triton TensorRT-LLM Backend](https://github.com/triton-inference-server/tensorrtllm_backend): The Triton TensorRT-LLM Backend. Contribute to triton-inference-server/tensorrtllm_backend development by creating an account on GitHub.
- [Add MPS support by maximegmd · Pull Request #1264 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1264): Description Supports basic training on Mac M series. Motivation and Context It partially solves Mac support. How has this been tested? Ran a train job with lora-mps.yml from start to finish.
- [Fix breaking change  by younesbelkada · Pull Request #1414 · huggingface/peft](https://github.com/huggingface/peft/pull/1414): Fix a breaking change in the recent release, I made a new PR as I messed up the commit history on the previous PR cc @sayakpaul @pacman100

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1206563791597207603) (8 messages🔥): 

- **Converging on Chat Dataset Formats**: `@dctanner` is coordinating with Hugging Face to standardize a chat dataset format named **MessagesList**, to streamline the various dataset formats emerging for fine-tuning chat models. They shared a link to the [MessagesList proposal discussion](https://huggingface.co/posts/dctanner/975913831192894) and suggested creating a GitHub org and dedicated page for documentation.

- **Naming Conventions Matter**: `@dctanner` emphasized the importance of a universal format name like **MessagesList** that’s not tied to a specific app like ShareGPT or ChatML, which can be confused with the template rather than the JSON format itself.

- **Validation Challenges for MessagesList**: `@faldore` acknowledged that although they like the idea of **MessagesList**, it poses challenges in validation because the concept of a "conversation pair" is not easily described by JSON-schema.

- **The Ideal MessagesList Schema**: `@faldore` proposed an ideal schema for the MessagesList format that includes optional system messages, tools/functions, source metadata, and a greeting message, ensuring user and assistant messages are paired, and the last message is from the assistant.

- **Benefits of the Suggested Schema**: `@faldore` advocates for the proposed schema, arguing that it is more manageable, verifiable, and space-efficient, and enforces structured message pairing in datasets.

**Links mentioned**:

[@dctanner on Hugging Face: &quot;As the amount of datasets for fine tuning chat models has grown, there&#39;s been…&quot;](https://huggingface.co/posts/dctanner/975913831192894): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1206514835957747743) (26 messages🔥): 

- **Tokenization Troubles in Axolotl**: User `@nafnlaus00` enquired about a method to verify that axolotl is tokenizing as expected. `@dreamgen` recommended inspecting the tokenizer config in the output directory, while `@nanobitz` pointed to a [debug flag in the axolotl repository](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main#preprocess-dataset).

- **Transformers Update Might Fix Inferencing Issue**: `@thierry_lama` reported a device error while trying to infer on a trained model using runpod's GPU. `@nanobitz` suggested that it could be due to an issue with transformers and recommended updating.

- **Multilingual Capabilities Enhancement Attempt**: `@sadaisystems` asked about improving a model's capabilities in a language other than English, receiving a response from `@le_mess` that pre-training is necessary for significant improvement beyond what LoRA can offer.

- **Inferencing with LoRA on the Fly**: `@wizmak` sought a way to add LoRA adapters to a base model in real-time during inferencing, and `@nanobitz` confirmed that with Hugging Face, you can load the peft model, but was unsure of the command to unload it.

- **Model Parallelism with DeepSpeed Zero 3**: User `@mihai4256` sought assistance for a working deepspeed zero 3 config for model parallelism, noting that existing ones from the repo weren't functioning as expected for this particular use case.

**Links mentioned**:

[GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main?tab=readme-ov-file#preprocess-dataset): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1206696642355658792) (1 messages): 

- **Duplicate Dilemma in Dataset Finetuning**: `@_rxavier_` inquired about identifying if a text has been previously used to train a model. They asked about techniques for determining model familiarity with a text, possibly by examining the model's response to an article's introduction.

- **The Impact of Training Data Overlap**: Additionally, `@_rxavier_` questioned the implications of finetuning a model using data that may overlap with its pretraining dataset. They pondered the potential negative effects of such overlap on the finetuning process.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1207032182896599160) (1 messages): 

- **Axoltl RunPod Compatibility with Vast.AI**: User `@dreamgen` successfully used the **Axoltl RunPod image** on Vast.AI, reporting that it worked *out of the box*.
  

---



### LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1207172343600783380) (1 messages): 

- **LangChain Introduces Journaling App with Memory**: `@hwchase17` shared an early version of a journaling app that incorporates memory, using the **LangChain memory module**. The app is in an early stage and feedback is welcomed; it remembers information about users for future interactions, akin to the memory feature announced by **OpenAI for ChatGPT** today. Test the app [here](https://journal.langchain.com/) and check out the [introductory video](https://www.loom.com/share/63a9696036c74765a9f9ecae06336aa5).

**Links mentioned**:

- [Loom | Free Screen &amp; Video Recording Software](https://www.loom.com/share/63a9696036c74765a9f9ecae06336aa5): Use Loom to record quick videos of your screen and cam. Explain anything clearly and easily – and skip the meeting. An essential tool for hybrid workplaces.
- [LangChain Companion - Journal](https://journal.langchain.com/): no description found

  

---


### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1206511343545487392) (59 messages🔥🔥): 

- **Seeking Assistance on LangChain with Android**: User `@grindmaster2512` inquired about the integration of LangChain with an Android application, and followed up with `@812757029260230658` to seek solutions to this query.
- **Efficient Chunk Pre-processing for Embeddings**: `@swastikk` asked whether chunk pre-processing (like removing white spaces) is necessary before creating embeddings. `@johnny2x2` confirmed that removing superfluous text aids the process, especially with email data.
- **PDF Parser Search, Alternatives to Adobe API**: `@dejoma` requested recommendations for a PDF parser that can split contextually, expressing dissatisfaction with Adobe API's limitations and seeking effective PDF API alternatives.
- **Calls to Improve LangChain's Documentation Structure**: `@b0otable` provided feedback to the LangChain team suggesting the improvement in documentation structure by reducing example redundancies and updating syntax to avoid inefficient navigation for users.
- **Dependency Issues with Pinecone and LangChain**: User `@segmentationfault.` experienced dependency resolution errors when trying to update Pinecone Database to v2 with a LangChain dependency, prompt response and solutions were provided by `@jacoblee93`, a maintainer of LangChain.

**Links mentioned**:

- [How to use function calling with Azure OpenAI Service - Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling?tabs=python): Learn how to use function calling with the GPT-35-Turbo and GPT-4 models
- [Pinecone | 🦜️🔗 Langchain](https://js.langchain.com/docs/integrations/vectorstores/pinecone): You can use Pinecone vectorstores with LangChain.

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1206581739623948301) (8 messages🔥): 

- **Clarification on Langserve Scaling**: `@kjoth_00356` inquired about scaling **Langserve** to multiple instances and asked about the difference between hosted Langserve and Langserve. `@veryboldbagel` hinted at a deployment via hosted Langserve, leading to further clarification from `@dachsteinhustler` who pointed to **Langsmith** as part of the solution hosted at [Langchain Platform](https://smith.langchain.com), which is in early testing and might require an invite code.
  
- **In Search of NodeJS and Chain Integration**: `@_mauricemoss` is looking for a way to expose a **chain from a NodeJS app** for use in a RemoteRunnable, but no solution has been provided within these messages.

- **Disabling Intermediate Steps in Playground**: `@dachsteinhustler` expressed a need to disable the intermediate steps in the Langchain playground to prevent browser crashes caused by large base64 strings, resulting in a workaround that involves using RunnableLambda.

- **Connection Issues with k8s Cluster App**: `@ezelanza.` described an issue where a connection is refused when attempting to invoke the **OpenAI API** through a **k8s cluster-based** application, mentioning that direct invocations to the back end work, but requests from the front end (React) fail, even with curl.

**Links mentioned**:

[LangSmith](https://smith.langchain.com): no description found

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1206604340060753970) (1 messages): 

- **Introducing Dewy and RAG with NextJS and OpenAI**: `@kerinin` shared their contribution towards Dewy, an OSS knowledge base, along with a post detailing how to build a **full-stack RAG application**. The guide includes using [NextJS](https://nextjs.org/), [OpenAI API](https://platform.openai.com/), and [Dewy](https://dewykb.github.io/), aimed to minimize hallucinations and ensure accurate language model responses. Check out the blog post [here](https://dewykb.github.io/blog/rag-app-with-nextjs-openai-and-dewy/).

**Links mentioned**:

[Building a RAG chatbot with NextJS, OpenAI &amp; Dewy | Dewy](https://dewykb.github.io/blog/rag-app-with-nextjs-openai-and-dewy/): This guide will walk you through building a RAG application using NextJS for the web framework, the OpenAI API for the language model, and Dewy as your knowledge base.

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1206543866526699580) (2 messages): 

- **Seeking a Superior PDF Parser**: `@dejoma` is looking for a **PDF parser that can split contextually**. Expresses discontent with Adobe API due to its low usage cap and lack of 'pay-as-you-go' option; is open to suggestions for robust PDF APIs.
- **Langchain Calculator Quest**: `@sougata` is building a calculator using **Langchain** that interprets multiplicative operations as `mul(a,b)`. Requests guidance on how to integrate a custom Python library for calculation with the model's augment function.
  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1206620477854978139) (29 messages🔥): 

- **Inquiry on Argilla Hosting Experience**: User `@drxd1000` is seeking advice on hosting a server for **Argilla** that can support multiple users for annotation, but there was no resolution provided in the messages.
- **Layer Selective Rank Reduction Methodology Discussed**: `@johannhartmann` referenced their own implementation of 'Layer Selective Rank Reduction' to address **continual training without forgetting**, noting that "They basically figure out the statistically less relevant parts of the layers and use them as lora targets," and considering it more efficient than continual approaches. A related **GitHub repository** was mentioned but not detailed in the conversation: [laserRMT](https://github.com/cognitivecomputations/laserRMT).
- **Out of Memory Issue with lm-evaluation-harness**: `@philipmay` faced an OOM error evaluating a **mixtral model** and was advised by `@bjoernp` to utilize multi-GPU support provided by lm-evaluation-harness, indicating **two A100s** might resolve the issue.
- **Search for German Toxicity Eval Dataset**: User `@sten6633` inquired about a German toxicity evaluation dataset and pondered the utility of translating **ToxiGen**, a dataset available on Hugging Face for implicit hate speech detection. The dataset mentioned can be found on Hugging Face, but requires agreement for access: [ToxiGen](https://huggingface.co/datasets/skg/toxigen-data).
- **Novel Computational Technique Teased**: User `@phantine` hinted at a new method excluding MoE, briefly titled "Universes in a bottle" and hinted at a potentially radical claim: "P=NP." A GitHub link associated with `@phantine`'s work was shared, but no specific details regarding the technique were provided: [LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM).

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/12GWmI7bBHt_iYfeKDmjt-g6zSPXLYObU?usp=sharing): no description found
- [skg/toxigen-data · Datasets at Hugging Face](https://huggingface.co/datasets/skg/toxigen-data): no description found
- [GitHub - cognitivecomputations/laserRMT: This is our own implementation of &#39;Layer Selective Rank Reduction&#39;](https://github.com/cognitivecomputations/laserRMT): This is our own implementation of &#39;Layer Selective Rank Reduction&#39; - cognitivecomputations/laserRMT
- [GitHub - LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM): Contribute to LargeWorldModel/LWM development by creating an account on GitHub.

  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1206732998548922389) (5 messages): 

- **BM25 + Query + Rerank Combo Wins**: User `huunguyen` highlighted their effective use of **BM25** with additional querying and reranking steps for search purposes, and reported that this method **"works pretty good."**
- **Wikipedia in a Nutshell**: `huunguyen` managed to **index the entirety of Wikipedia**, excluding non-essential content, and compacted the BM25 index into a sleek **size of under 3GB**.
- **In Search of BM25 Tools**: `sebastian.bodza` inquired about the specific library `huunguyen` is using to implement the **BM25** algorithm for their search index.
  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/) (1 messages): 

thomasrenkert: Is there an ETA for v2 of the German model? Or for the Mixtral variant?
  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1206758717400547399) (1 messages): 

- **GPU Shuffling for Experimentation**: `@joseph_en` reported successful relocation of the **Asus WS motherboard** to the miner and is awaiting **16x PCI extenders**. They've utilized older GPUs for their experiments and have transitioned the miner's motherboard into the case, noting it handles **7B or 13B quantized models** with a single 12G **NVIDIA 3060** with ease.
  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1206667178200399893) (9 messages🔥): 

- **Cross-Compatibility Quest**: `@iron_bound` kicks off a discussion about achieving binary compatibility for CUDA to run on HIP/ROCm platforms, referencing a [Phoronix article on Radeon CUDA - ZLUDA](https://www.phoronix.com/review/radeon-cuda-zluda).
- **CUDA for AMD GPUs? Meet ZLUDA**: `@muhtasham` shares a [GitHub link](https://github.com/vosen/ZLUDA) to **ZLUDA**, a project that aims to make CUDA run on AMD GPUs, sparking interest and a request for user experiences by `@marksaroufim`.
- **Emoji Enthusiasm**: `@muhtasham` invokes the spirits of the tech world through well-selected emojis of Jensen Huang and Lisa Su.
- **Market Monopolies and AGI Speculations**: `@andreaskoepf` humorously suggests that Microsoft's purchasing strategy and a borked chip market could leave antitrust agencies unequipped against an AGI future.
- **Real-World Radeon Trials**: `_tvi_` shares their experience with **Radeon VII** and a Ryzen APU, including struggles with dynamic memory allocation causing kernel crashes when handling large PyTorch data chunks.

**Links mentioned**:

- [Tweet from [Phoronix] AMD Quietly Funded A Drop-In CUDA Implementation Built On ROCm: It&#039;s Now Open-Source Image (Radeon Cuda 1)](https://www.phoronix.com/image-viewer.php?id=radeon-cuda-zluda&image=radeon_cuda_1_lrg): no description found
- [Tweet from AMD Quietly Funded A Drop-In CUDA Implementation Built On ROCm: It's Now Open-Source - Phoronix](https://www.phoronix.com/review/radeon-cuda-zluda): no description found
- [GitHub - vosen/ZLUDA: CUDA on AMD GPUs](https://github.com/vosen/ZLUDA): CUDA on AMD GPUs. Contribute to vosen/ZLUDA development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1206591496174960660) (3 messages): 

- **Multidimensional Gated Recurrences Have Limitations**: User `@euclaise` mentioned a constraint in multidimensional gated recurrences, stating that they require a **DxCxN attention matrix** which is quite **prohibitive in cost**, even with a small value for C.
- **Beyond Simple Linear Recurrences**: `@euclaise` pointed out that **prefix-sum-like scans** have applications beyond computing simple linear recurrences, opening up a broader range of computational possibilities.
- **Twitter Insights on Computational Techniques**: `@euclaise` shared insights on computational methods, including the use of **maximal scans** for sequences (`y[t]=max(y[t-1], x[t])`), by providing links to their Twitter posts: [Tweet on computational methods](https://twitter.com/Euclaise_/status/1757512166284861731) and [Tweet on maximal scans](https://twitter.com/Euclaise_/status/1757173421400613030).
  

---


### CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1206528123013693472) (3 messages): 

- **Generative AI Startup Hiring in Hyderabad**: `@gradman33` shared a job opportunity at an **early stage Deep Tech Generative AI startup** in Hyderabad, India, seeking talents in ML/Data/Research/SDE. Interested candidates can [apply here](https://forms.gle/aP5qwv66XM2D7RCS8).
- **Potential Spam Alert in Jobs Channel**: `@pudding0377` flagged a post by `@gradman33` as possibly irrelevant or spam, calling for the attention of moderators.

**Links mentioned**:

[no title found](https://forms.gle/aP5qwv66XM2D7RCS8): no description found

  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1206882432926748713) (9 messages🔥): 

- **New Member Alert**: `@cs_os_05101` mentioned that they have a **4060 Ti**.
- **Search for Engaging CUDA Books**: `@euclaise` inquired about fun books related to CUDA, sparking a conversation about educational resources.
- **Shader Book Recommendation**: `@marksaroufim` shared [The Book of Shaders](https://thebookofshaders.com/), a gentle guide to Fragment Shaders, as a possible fun read on a topic adjacent to CUDA.
- **Understanding User Expertise**: After citing familiarity with shader programming, `@euclaise` clarified they're looking for materials directly related to compute shaders or CUDA, rather than frag shaders.
- **Looking for Fun in Learning**: Both `@marksaroufim` and `@euclaise` concurred that defining literature as "fun" can be subjective, but `@marksaroufim` suggested PMPP as the best educational resource on CUDA, albeit not necessarily fitting the "fun" criterion.

**Links mentioned**:

[The Book of Shaders](https://thebookofshaders.com/): Gentle step-by-step guide through the abstract and complex universe of Fragment Shaders.

  

---


### CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1206782186871722024) (7 messages): 

- **Matrix Transposition Debate**: `@eporat` asked if transposing one matrix in a multiplication could lead to fewer cache misses and thus faster computation. `@andreaskoepf` responded, advising that while sequential memory access could be advantageous, the benefits might be negligible compared to tiled access.
- **Practical Test Yields No Benefits**: Responding to the query about transposing matrices to speed up multiplication, `@jeremyhoward` recounted his experience stating that transposing during tile creation had **no observable effect** on performance.
- **In-Depth Discussion on Transposition**: `@eporat` clarified that an inplace transpose isn’t necessary; sometimes, one only needs to adjust indice ordering in the inner loop, suggesting an alternative to transposition.
- **Further Clarification Sought**: `@andreaskoepf` questioned `@eporat`'s suggestion, implying that matrix elements are read transposed by default during multiplication, indicating a misunderstanding or need for further explanation on what `@eporat` meant by adjusting loop indices.
  

---


### CUDA MODE ▷ #[smol-hw](https://discord.com/channels/1189498204333543425/1205223658021458100/1206717221137809489) (1 messages): 

- **Apple Silicon gets its own 'top'**: User `@marksaroufim` shared a link to [asitop](https://github.com/tlkh/asitop), a performance monitoring CLI tool for Apple Silicon. It was compared to existing tools like `top` or `nvtop`, tailored specifically for Apple's custom chips.

**Links mentioned**:

[GitHub - tlkh/asitop: Perf monitoring CLI tool for Apple Silicon](https://github.com/tlkh/asitop): Perf monitoring CLI tool for Apple Silicon. Contribute to tlkh/asitop development by creating an account on GitHub.

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1206729713624682506) (24 messages🔥): 

- **Reka Model Announcement**: `@swyxio` shared a link to a tweet about a new **Reka model**, creating a buzz in the community. The tweet can be found [here](https://twitter.com/YiTayML/status/1757115386829619534).
- **Favorite VC Podcast Meets AI**: `@swyxio` expressed enthusiasm for a VC podcast discussing AI topics, providing a [link](https://overcast.fm/+afOCk9-tI) to the episode and highlighting its relevance to the community.
- **Exploring the BUD-E Voice Assistant by LAION**: `@swyxio` discussed a new fully open voice assistant named **BUD-E**, developed by LAION, which is aimed to improve conversational experiences by being empathetic and context-aware. Details are available on the [LAION blog](https://laion.ai/blog/bud-e/).
- **What is an Agent?**: In search of a definition for "agents," `@kaycebasques` asked the community for insights. `@slono` described them as programs that aim to achieve goals with minimal user input.
- **Karpathy Leaves OpenAI**: `@nembal` spotlighted news from The Information about **Andrej Karpathy's departure** from OpenAI, stirring curiosity about the implications for the AI field. Background on the development of an AI product for automating tasks mentioned by `@slono` vaguely referenced **AGI** as a possible factor in the context of the departure.

**Links mentioned**:

- [BUD-E: Enhancing AI Voice Assistants’ Conversational Quality, Naturalness and Empathy | LAION](https://laion.ai/blog/bud-e/): &lt;p&gt;AI voice assistants have revolutionized our interaction with technology, answering queries, performing tasks, and making life easier. However, the stilted...
- [OpenAI Researcher Andrej Karpathy Departs](https://www.theinformation.com/articles/openai-researcher-andrej-karpathy-departs): Andrej Karpathy, one of the founding members of OpenAI, has left the company, a spokesperson confirmed. Karpathy, a prominent artificial intelligence researcher, was developing a product he has descri...
- [President and Co-Founder Anthropic, Daniela Amodei: AI Hurricane &mdash; Grit &mdash; Overcast](https://overcast.fm/+afOCk9-tI): no description found
- [Memory and new controls for ChatGPT](https://openai.com/blog/memory-and-new-controls-for-chatgpt): We’re testing the ability for ChatGPT to remember things you discuss to make future chats more helpful. You’re in control of ChatGPT’s memory.
- [How Graph Neural Networks Are Transforming Industries](https://www.youtube.com/watch?v=9QH6jnwqrAk): 🔑 Get your AssemblyAI API key here: https://www.assemblyai.com/?utm_source=youtube&amp;utm_medium=referral&amp;utm_campaign=yt_marco_1Graph Neural Networks (GNN) ha...
- [Tweet from Joanne Jang (@joannejang)](https://x.com/joannejang/status/1757470618264429008?s=46&t=90xQ8sGy63D2OtiaoGJuww): 📝 we just launched a small experiment for memory on ChatGPT.  how it works - it&#39;s quite similar to custom instructions, except chatgpt is the one driving it (like auto vs. stick shift!) - basical...
- [GitHub - Stability-AI/StableCascade](https://github.com/Stability-AI/StableCascade): Contribute to Stability-AI/StableCascade development by creating an account on GitHub.
- [sta - Overview](https://github.com/Sta): sta has 2 repositories available. Follow their code on GitHub.

  

---



### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/1206709514515255386) (2 messages): 

- **Choosing the Right Mistral Model Size**: `@potrock` asked about the appropriate **Mistral model size** to run locally on an M2 Max with 32GB, seeking community input.
- **Safe Model Sizing Advice**: `@natureplayer` suggested that **4GB** is a safe size for local execution on the mentioned hardware, while **8GB** will not work, and **5GB** might be possible but is not guaranteed.
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1207005947008913471) (4 messages): 

- **GPT-5 Speculation Quelled**: User `@res6969` humorously noted that **the rumors of GPT-5** have been greatly exaggerated, indicating skepticism about its existence or imminent release.
- **Laughter is the Best Medicine?**: Both `@res6969` and `@potrock` shared lighthearted reactions with custom emoji and laughing-to-tears emoji, respectively, contributing to a jovial environment on the topic at hand.
- **A Memory Upgrade for ChatGPT**: `@potrock` shared a [blog post](https://openai.com/blog/memory-and-new-controls-for-chatgpt) discussing new memory features being tested in ChatGPT that allow the model to remember user preferences and details across conversations, which users can manage conversationally or through settings.

**Links mentioned**:

[Memory and new controls for ChatGPT](https://openai.com/blog/memory-and-new-controls-for-chatgpt): We’re testing the ability for ChatGPT to remember things you discuss to make future chats more helpful. You’re in control of ChatGPT’s memory.

  

---



### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1207016777154826250) (6 messages): 

- **Weekly Meeting Kick-Off with a Sense of Humor**: `@._z` announced the start of the weekly meeting with a playful note: 😄 *Déjà vu*.
- **Meeting Attendance Update**: `@juanreds` informed they could not attend the weekly meeting, apologizing to the team.
- **Invitation to Co-host an AI Hackathon**: `@caramelchameleon` asked if anyone is interested in co-hosting an AI developers hackathon, hinting at a collaboration opportunity with game developers before GDC this year.
- **Chance to Join Hackathon Online or Onsite**: `@caramelchameleon` mentioned the possibility of attending the hackathon both online and onsite in San Francisco.
- **Eager Organizer Jumps In**: `@yikesawjeez` expressed interest and requested to be contacted as they specialize in organizing hackathons, especially those associated with events in the Bay Area.
  

---



### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1206617643428216832) (2 messages): 

- **Private Message Prompt**: User `@bondconnery` requests a direct message with a simple "<@1117586410774470818> DM sir".
- **LLaVA Framework Inquiry**: `@CodeMan` is seeking insights or experiences on integrating **LLaVA** with an **SGLang server** and **SGLang worker**, as opposed to using a standard model worker.
---
id: 01eabd5a-e8bb-41e3-981b-58cc9467e3b3
title: Less Lazy AI
date: '2024-02-06T00:50:28.809972Z'
original_slug: ainews-less-lazy-ai
description: >-
  The AI Discord summaries for early 2024 cover various community discussions
  and developments. Highlights include **20** guilds, **308** channels, and
  **10449** messages analyzed, saving an estimated **780 minutes** of reading
  time. Key topics include **Polymind Plugin Puzzle** integrating PubMed API,
  roleplay with **HamSter v0.2**, VRAM challenges in **Axolotl** training,
  fine-tuning tips for **FLAN-T5**, and innovative **model merging** strategies.
  The **Nous Research AI** community discussed GPT-4's lyricism issues,
  quantization techniques using `llama.cpp`, **frankenmerging** with models like
  **miqu-1-120b-GGUF**, anticipation for **Qwen2**, and tools like
  `text-generation-webui` and **ExLlamaV2**. The **LM Studio** community
  reported a bug where the app continues running after UI closure, with a
  workaround to forcibly terminate the process. These discussions reflect
  ongoing challenges and innovations in AI model training, deployment, and
  interaction.
companies:
  - openai
  - hugging-face
  - nous-research
  - h2oai
  - apple
models:
  - hamster-v0.2
  - flan-t5
  - miqu-1-120b-gguf
  - qwen2
  - axolotl
topics:
  - model-merging
  - fine-tuning
  - quantization
  - vram-optimization
  - plugin-development
  - chatbot-memory
  - model-training
  - bug-reporting
  - api-compatibility
people:
  - philschmid
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/3-4/2024. We checked **20** guilds, **308** channels, and **10449** messages for you. Estimated reading time saved (at 200wpm): **780 minutes**.

We've anecdotally gotten examples of refusal to follow instructions approximating laziness:

[![image.png](https://assets.buttondown.email/images/6fcebf2c-e2fb-4e60-8948-dc134842fc28.png?w=960&fit=max)](https://www.reddit.com/r/OpenAI/comments/1aj6lrz/damned_lazy_ai/)

but it is hard to tell when it is luck of a bad draw or [shameless self promotion](https://twitter.com/NickADobos/status/1732894920573370527/photo/2). 

This is why it's rare to get [official confirmation from the top](https://x.com/sama/status/1754172149378810118?s=20):

 ![image.png](https://assets.buttondown.email/images/32325848-6d79-41b4-8a26-deef9d9aacc9.png?w=960&fit=max) 

Still, laziness isn't a well defined technical term. It is frustrating to know that OpenAI has identified a problem and fixed it, but is not sharing what exactly it is.

---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Polymind Plugin Puzzle**: @doctorshotgun is enhancing Polymind with a plugin that integrates PubMed's API to bolster the article search capabilities. Development complexities arise with the sorting of search results.

- **AI Model Roleplay Rig**: Users recount their engagements with AI models for roleplaying, noting **HamSter v0.2** from [PotatoOff](https://huggingface.co/PotatoOff/HamSter-0.2) as a choice for detailed, unrestricted roleplay. Meanwhile, significant VRAM usage during the training of models like qlora dpo is a common challenge, with the `use_reentrant` flag in **Axolotl** set to `False` being a key VRAM consumption factor.

- **Tailoring FLAN-T5 Training Tips**: In the quest for training a code generation model, @Naruto08 is guided to consider models like FLAN-T5, with resources like [Phil Schmid's fine-tuning guide](https://www.philschmid.de/fine-tune-flan-t5) available for reference. Meanwhile, @rolandtannous provides the DialogSum Dataset as a viable resource for fine-tuning endeavors on a p3.2xlarge AWS EC2 instance.

- **Merging Model Mastery**: @maldevide introduces a partitioned layer **model merging** strategy with an inventive approach to handle the kvq with a 92% drop rate and a 68% drop rate for partitioned layers. The methodology and configuration are openly shared on [GitHub Gist](https://gist.github.com/maldevide/08829eada04ad9bd78e46c1a3787d42b).

- **Local Chatbot Configuration Conundrum**: @aletheion is on the hunt for a method to integrate a local model database lookup for an offline chatbot, and @wildcat_aurora recommends considering [h2ogpt](https://github.com/h2oai/h2ogpt) as a solution. Furthermore, @vishnu_86081 is exploring ChromaDB for character-specific long-term memory in a chatbot app.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Tackling GPT-4's Lyricism Limitations**: Members discussed GPT-4's issues with generating accurate lyrics, noting that using *perplexity with search* leads to better outcomes than GPT-4's penchant for fabricating lyrics.

- **Quantization Roadmap for LLMs**: Topics included strategies for quantizing models, such as using `llama.cpp` for quantization processes and discussing the knowledge requirements for efficient VRAM usage in models like Mixtral, which can require up to 40GB in 4bit precision.

- **Innovative Model Merging Solutions**: The community highlighted **frankenmerging** techniques with the unveiling of models like **miqu-1-120b-GGUF** and **MergeMonster**, and touched upon new methods like **emulated fine-tuning (EFT)** considering RL-based frameworks for stages of language model education.

- **Anticipation and Speculation Around Emerging Models**: Conversations buzzed about the forthcoming **Qwen2** model, predicting significant benchmarking prowess. Preference tuning discussions mentioned KTO, IPO, and DPO methods, citing a Hugging Face blog post, which posits IPO as on par with DPO and more efficacious than KTO.

- **Tools and Frameworks Enhancing AI Interaction and Testing**: Mentioned solutions included `text-generation-webui` for model experimentation, `ExLlamaV2` for OpenAI API compatible servers, and [Lone-Arena](https://github.com/Contextualist/lone-arena) for self-hosted LLM chatbot testing. Additionally, the community took note of a GitHub discussion regarding potential Apple Neural Engine support within `llama.cpp`.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Ghost in the Machine! LM Studio Not Shutting Down**: Users reported that **LM Studio** continues running in task manager after the UI is closed. The suggested workaround is to **forcibly terminate the process** and report the bug.

- **CPU Trouble with AVX Instruction Sets**: Some users encountered errors due to their processors lacking **AVX2 support**. The community pointed out that **LM Studio** requires AVX2, but a beta version might accommodate CPUs with only AVX.

- **AMD GPU Compute Adventures on Windows 11**: For those wishing to use **AMD GPUs on Windows 11** with LM Studio, a special **ROCm-supported beta version** of LM Studio is essential. Success was reported with an **AMD Radeon RX 7900 XTX** after disabling internal graphics.

- **Whisper Models and Llama Combinations Spark Curiosity**: Integrating **Whisper and Llama models** with LM Studio was a topic of interest, with users referred to certain models on **Hugging Face** and other resources like **Continue.dev** for coding with LLMs.

- **Persistent Processes and Erratic Metrics in LM Studio**: Users experienced problems with **LM Studio's Windows beta build**, including inaccurate CPU usage data and processes that persist post-closure. Calls for improved GPU control in LM Studio ensued within community discussions.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **LLama3 and Mistral Integration Insights**: Community members speculated on the architecture and training data differences between **LLama3** and other models, while **Mixtral**'s effectiveness with special characters and long texts was a hot topic. Performance comparisons between **OpenHermes 2.5** and **Mistral**, particularly "lost in the middle" issue with long contexts, were also discussed. Details on handling markdown in prompts and troubleshooting with tools like [GuardrailsAI](https://www.guardrailsai.com/) and [Instructor](https://jxnl.github.io/instructor/) were exchanged.

- **Model Hosting and Development Dilemmas**: AI hosting on services like [Hugging Face](https://hf.co/) and [Perplexity Labs](https://perplexity.ai/) was considered for its reliability and cost-effectiveness. A discussion on CPU inference for LMMs raised points about the suitability of different model sizes and quantization methods, with **Mistral's quantization** featuring prominently. A new user was guided towards tools like [Gradio](https://www.gradio.app/) and [Hugging Face's hosted models](https://huggingface.co/chat) for model deployment without powerful hardware.

- **Fine-tuning Focuses and Financial Realities**: Questions on fine-tuning for specific domains like energy market analysis were addressed, highlighting its feasibility but also the existing constraints due to Mistral's limited resources. The community explored current limitations in **Mistral**'s API development, citing the high costs of inference and team size as critical factors.

- **Showcasing AI in Creative Arenas**: Users showcased applications such as novel writing with AI assistance and critiqued AI-generated narratives. Tools for improving AI writing sessions like adopting *Claude* for longer context capacity were suggested. Additionally, **ExLlamaV2** was featured in a [YouTube video](https://www.youtube.com/watch?v=N5lDUZRI8sc) for its fast inference capabilities on local GPUs.

- **From Random Remarks to Platform Peculiarities**: A Y Combinator founder called out to the community for insights on challenges when **building in the space of LLMs**. On a lighter note, playful messages like flag emojis popped up unexpectedly. Meanwhile, in **la-plateforme**, streaming issues with **mistral-medium** not matching the behavior of **mistral-small** were discussed, drawing ad-hoc solutions like response length-based discarding.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Collaboration Woes with ControlNet Authors**: @pseudoterminalx voiced *frustrations* about challenges collaborating with the creators of ControlNet, citing a focus on promoting AUTOMATIC1111 at the expense of supporting community integration efforts. This reflects a wider sentiment of difficulty in implementation among other engineers.

- **Ethical Debates on Dataset Practices**: Ethical concerns were raised surrounding the actions of Stanford researchers with respect to the LAION datasets, insinuating a shift towards business priorities following their funding achievements, potentially impacting public development and resource access.

- **Comparing AI Trailblazers**: A discussion emerged comparing the strategies of Stability AI with those of a major player like NVIDIA. The conversation questioned the innovative capacities of smaller entities when adopting similar approaches to industry leaders.

- **Hardware Discussions on NVIDIA Graphics Cards**: The engineering community engaged in an active exchange on the suitability of various NVIDIA graphics cards for AI model training, specifically the 4060 ti and the 3090, taking into account VRAM needs and budget considerations.

- **Speculations on Stability AI’s Next Moves**: Anticipation was building with regard to Stability AI's forthcoming model, prompting @thejonasbrothers to express concerns about the competitiveness and viability of long-term projects in light of such advancements.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Demo Difficulties with Falcon-180B**: Users like @nekoli. reported issues with the Falcon-180B demo on Hugging Face, observing either site-wide issues or specific outages in the demos. Despite sharing links and suggestions, resolutions seemed inconsistent.

- **LM Deployment and Use Queries**: Queries emerged regarding deployment of LLMs such as Mistral 7B using AWS Inferentia2 and SageMaker, and how to access LLMs through an API with free credits on HuggingFace, although no subsequent instructional resources were linked.

- **Spaces Stuck and Infrastructure Woes**: There were reports of a Space in a perpetual building state and potential wider infrastructure issues at Hugging Face affecting services like Gradio. Some users offered troubleshooting advice.

- **AI's Role in Security Debated**: Concerns were voiced over the misuse of deepfake technology, such as a scam involving a fake CFO. This highlights the importance of ethical considerations in the development and deployment of AI systems.

- **Synthesizing Community Insights Across Disciplines**: The discussions covered a range of topics including admiration for the foundational "Attention Is All You Need" paper, advancements in **Whisper** for speaker diarization in speech recognition, the creation of an internal tool for summarizing audio recordings with a privacy-centric approach, and user engagement in a variety of Hugging Face community activities like blog writing, events, and technical assistance. 

- **Hugging Face Community Innovates**: The Hugging Face community shared a host of creations, from a proposed ethical framework for language model bots to projects like [Autocrew for CrewAI](https://github.com/yanniedog/autocrew), a hacker-assistant chatbot, predictive emoji spaces based on tweets, and the publication of the [Hercules-v2.0 dataset](https://huggingface.co/datasets/Locutusque/hercules-v2.0) for powering specialized domain models.

- **Explorations in Vision and NLP**: Zeal was high for finding resources and collaborating on projects such as video summarization with timestamps, ethical frameworks for LLMs, spell check and grammar models, and the pursuit of Nordic language model merging with resources like a [planning document](https://docs.google.com/document/d/1fP2FIrCifWcLGdTBmqeogdCdZJOwxqPfEyO-HA76_qc/edit?usp=sharing), [tutorial](https://huggingface.co/blog/mlabonne/merge-models), and [Colab notebook](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing).

- **Scam Alerts and Technical Challenges in Diffusion Discussions**: A scam message was flagged for removal, a GitHub issue with `AutoModelForCausalLM` was detailed, and the Stable Video Diffusion model [license agreement](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/tree/main) was shared to discuss weight access, all reflecting the community's efforts to maintain integrity and solve complex AI issues.

- **Engagement in the World of Computer Vision**: Questions popped up about using Synthdog for fake data generation, finding current models for zero-shot vision tasks, and creating a sliding puzzle dataset for vision LLM training, suggesting an active search for novel approaches in AI.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Local LLMs Spark Interest Amidst GPT-4 Critiques**: Engineers discuss potential alternatives to GPT-4, highlighting **Local LLMs** such as LM Studio and perplexity labs as viable options. Users express concerns about GPT-4's errors and explore the performance of other models like codellama-70b-instruct.

- **GPT-4 Glitches Got Engineers Guessing**: Reports have surfaced around **@ mention issues** and erratic GPT behavior, including memory lapses, indicating possible GPT-4 system inconsistencies. The user base is also grappling with missing features like the thumbs up option and sluggish prompt response times.

- **Prompt Engineering Puzzles Professionals**: AI Engineers share frustration over ChatGPT's overuse of ethical guidelines in storytelling and suggestions for steering clear of AI-language patterns to maintain humanlike interactions in AI communications. Recommendations to use more stable GPT versions for instruction consistency are also favored.

- **Hardware Hurdles in Hosting LLMs**: Deep dives into hardware setups for running Local LLMs reveal engineers dealing with system requirements, notably the debate over RAM vs. VRAM. The community also voices skepticism about the information's credibility on AI performance across different hardware setups.

- **AI Assistance Customization Conundrums**: Detailed discussions ensue over refining GPT's communication for users with specific needs, such as generating human-like speech for autistic users, and strategies to avoid name misspellings. Additionally, some users encountered unanticipated content policy violation messages and speculated on internal issues.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **GPU Troubleshooting for Engineers**: To address GPU errors on RunPod, `@dangfutures` recommended the command `sudo apt-get install libopenmpi-dev pip install mpi4py`. Additionally, `@nruaif` stated 80gb VRAM is necessary for LoRA or QLoRA on Llama70, with MoE layer freezing enabling Mixtral FFT on 8 A6000 GPUs.

- **Scaling Ambitions Spark Skepticism and Optimism**: In a new [Notion doc](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20), a 2B parameter model by OpenBMB claimed comparable performance to Mistral 7B, generating both skepticism and excitement among engineers.

- **Finetuning Woes and Code Config Tweaks**: `@cf0913` experienced the EOS token functioning as a pad token after finetuning, which was resolved by editing the tokenizer config as suggested by `@nanobitz`. Also, `@duke001.` sought advice on determining training steps per epoch, with sequence length packing as a potential strategy.

- **Adapting to New Architectures**: An issue was raised about running the axolotl package on an M1 MacBook Air, with a response from `@yamashi` about submitting a PR to use MPS instead of CUDA. Discussions also revolved around implementing advanced algorithms on new hardware like the M3 Mac.

- **Memory Troubles with Differential Privacy**: `@fred_fups` struggled with out-of-memory issues when using differential privacy optimization (DPO) with qlora, and `@noobmaster29` confirmed DPO's substantial memory consumption, allowing only microbatch size of 1 with 24GB RAM.

- **RunPod Initialization Error and Configuration Concerns**: `@nruaif` shared logs from RunPod indicating deprecated configurations and errors, including a missing `_jupyter_server_extension_points` function and incorrect `ServerApp.preferred_dir` settings. `@dangfutures` suggested exploring community versions for more reliable performance.



---



## [CUDA MODE (Mark Saroufim)](https://discord.com/channels/1189498204333543425) Discord Summary

**CUDA Curiosity Peaks**: CUDA's dominance over OpenCL is attributed to its widespread popularity and Nvidia's support; Python continues to be a viable option for GPU computing, offering a balance between high-level programming ease and the nitty-gritty of kernel writing, as detailed in the [CUDA MODE GitHub repository](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#python-gpu-computing). Members also discussed the impact of compiler optimizations on CUDA performance, emphasizing the significance of even minute details in code, while advocating for robust CUDA learning through shared resources like [`tiny-cuda-nn`](https://github.com/NVlabs/tiny-cuda-nn).

**PyTorch Parsers Perspire**: Tips were shared on how to efficiently use the `torch.compile` API by specifying compiled layers, as seen in the [gpt-fast repository](https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L307-L314). There's a bonafide interest in controlling the Torch compiler's behavior more finely, with the [PyTorch documentation](https://pytorch.org/docs/main/torch.compiler_fine_grain_apis.html) offering guidance. Amidst PyTorch preferences, TensorFlow also got a nod, mainly for Google's hardware and pricing.

**Lecture Hype**: Anticipation grows as CUDA MODE's fourth lecture on compute and memory architecture is heralded, with materials found in a repository jokingly criticized for its "increasingly inaccurately named" title, **lecture2 repo**. The lecture promises to delve into the nitty-gritty of blocks, warps, and memory hierarchies.

**Job Market Buzzes**: Aleph Alpha and Mistral AI are on the hunt for CUDA gurus, with roles integrating language model research into practical applications. Positions with a focus on GPU optimization and custom CUDA kernel development are up for grabs, detailed in the Aleph Alpha [job listing](https://alephalpha.jobs.personio.de/job/1329474?language=en&display=en) and Mistral AI's [opportunity](https://jobs.lever.co/mistral/399978d0-b442-4591-b677-8cc03ee24a48).

**CUDA Beginners Unite**: Rust gained some spotlight in lower-level graphics programming and the discussion tilted towards its viability in CUDA programming, garnering interest for CUDA GPU projects in Rust, like [rust-gpu](https://github.com/embarkstudios/rust-gpu) for shaders. The Rust neural network scene is warming up, with projects like [Kyanite](https://github.com/KarelPeeters/Kyanite) and [burn](https://github.com/tracel-ai/burn) to ignite the coding fire.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **TimesFM Training Clarified**: A corrected sequence for **TimesFM model training** was shared to emphasize non-overlapping output paths based on the model's description. Meanwhile, the conversation about handling large contexts in LLMs spotlighted the [YaRN paper](https://arxiv.org/abs/2309.00071), while a method for autoencoding called "liturgical refinement" was proposed.

- **MoE-Mamba Delivers Impressive Results**: According to a [recent paper](https://arxiv.org/abs/2401.04081), "MoE-Mamba" SSM model surpasses other models with fewer training steps. Strategies, such as adding a router loss to balance experts in MoE models and stabilizing gradients via techniques from the [Encodec paper](https://arxiv.org/abs/2210.13438), were discussed for improving AI efficiency.

- **Interpretability Terms Defined**: In the realm of interpretability, a distinction was noted between a "direction" as a vector encoding monosemantic meaning and a "feature" as the activation of a single neuron.

- **Organizing Thunderous Collaborations**: A meeting schedule for **Tuesday 6th at 5pm (UK time)** was confirmed concerning topics like testing at scale, where **Slurm** was mentioned as a tool for queuing numerous jobs.

- **Multimodal MoE Models Explored**: Discussions veered toward merging MoEs with VLMs and diffusion models for multimodal systems, aiming for deeper semantic and generative integration, and investigating alternatives like RNNs, CLIP, fast DINO, or fast SAM.

- **GPT-NeoX "gas" Parameter Deprecated**: An update on GPT-NeoX involves the deprecation of the `"gas"` parameter as it was found non-functional and a duplicate of `"gradient_accumulation_steps"`, with the warning that past configurations may have used smaller batch sizes unintentionally. A review of the related [pull request](https://github.com/EleutherAI/gpt-neox/pull/123) is underway.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Polyglot Perplexity**: Users demonstrated interest in Perplexity AI's multilingual capabilities, with discussions about its proficiency in Chinese and Persian. Conflicting experiences were shared regarding Copilot's role in model performance, but consensus on its exact benefits remains unclear.

- **Criticizing Customer Care**: User `@aqbalsingh` faced difficulties with the **email modification process** and the **iPhone app's functionality**, leading to their premium account cancellation. They and `@otchuda` shared dissatisfaction with the level of support provided by Perplexity AI.

- **Excitement and Analysis via YouTube**: YouTube videos by *`@arunprakash_`, `@boles.ai`, and `@ok.alex`* provide analysis and reviews on why users might prefer Perplexity AI over other AI solutions, with titles like "I Ditched BARD & ChatGPT & CLAUDE for PERPLEXITY 3.0!"

- **Sharing Search Success**: Users exchanged Perplexity AI search results that impacted their decisions such as upgrading to Pro subscriptions or assisting with complex problems, highlighting the utility and actionable insights provided by Perplexity's search capabilities.

- **Mixtral's Monetization Muddle**: Within the **#[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1203350676214784010)** channel, there's ongoing curiosity about **Mixtral's pricing**, with current rates at **$0.14 per 1M input tokens and $0.56 per 1M output tokens**. The community showed interest in a **pplx-web version** of the API, prompting discussion about business opportunities for Perplexity AI, although no official plans were disclosed.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Seeking Solutions for Arabic AI Conversations**: Members discussed technology options for interacting with Arabic content, where an Arabic Language Model (LLM) and embeddings were suggested as most technologies are language-agnostic. Specific alternatives like *aravec* and *word2vec* were mentioned for languages not supported by [embedding-ada](https://www.sbert.net/examples/training/multilingual/README.html), such as Arabic.

- **Tips for Cost-Effective Agent Hosting**: For a research agent with a cost structure of 5 cents per call, recommendations included hosting a local LLM for controlled costs, as well as deploying services like [ollama](https://ollama.com/) on servers from companies like DigitalOcean.

- **Books and Learning Resources for LLM Enthusiasts**: A new book titled "LangChain in your Pocket: Beginner's Guide to Building Generative AI Applications using LLMs" was announced, providing a hands-on guide covering LangChain use cases and deployment, available at [Amazon](https://amzn.eu/d/dqQJzV1). Additionally, an extensive LangChain [YouTube playlist](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=UGdxhD6LwRSgQzno) was shared for tutorials.

- **Interactive Podcasts Leap Forward with CastMate**: **CastMate** was introduced, enabling listeners to interact with podcast episodes using LLMs and TTS technology. A [Loom demonstration](https://www.loom.com/share/c7a82509eaca450c814fae77c5db7a1d?sid=67313ae9-fca0-4a55-b536-a93b711a9d74) was shared, and an **iPhone beta** is available for testing through [TestFlight Link](https://testflight.apple.com/join/9khwQ1vD).

- **Navigating Early Hurdles with LangChain**: Users reported encountering errors and outdated information while following LangChain tutorials, indicating potential avenues for improving the documentation and support materials. Errors ranged from direct following of YouTube tutorial steps to issues with the Ollama model in the LangChain quickstart guide.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **RAG Pain Points Tackled**: @wenqi_glantz, in collaboration with @llama_index, remedied 12 challenges in production RAG development, with full solutions presented on a cheatsheet, which can be found in their [Twitter post](https://twitter.com/llama_index/status/1753829824484065625).

- **Hackathon Fueled by DataStax**: @llama_index acknowledged `@DataStax` for hosting and catering a hackathon event, sharing updates on [Twitter](https://twitter.com/llama_index/status/1753845015833686132).

- **Local Multimodal Development on Mac**: LlamaIndex's integration with Ollama now enables local multimodal app development for tasks like structured image extraction and image captioning, detailed in a [day 1 integration tweet](https://twitter.com/llama_index/status/1753875735776018786).

- **Diving Deep with Recursive Retrieval in RAG**: `@chiajy` explored recursive retrieval in RAG systems and shared three techniques—Page-Based, Information-Centric, and Concept-Centric—in their Medium article, [Advanced RAG and the 3 types of Recursive Retrieval](https://medium.com/enterprise-rag/advanced-rag-and-the-3-types-of-recursive-retrieval-cdd0fa52e1ba).

- **Hybrid Retrieval Lauded for Dynamic Adjustments and Contributions**: @cheesyfishes confirmed the **Hybrid Retriever's** **alpha** parameter can be dynamically altered, and @alphaatlas1 advised a hybrid retrieval plus re-ranking pipeline, spotlighted the **BGE-M3 model**, and called for contributions on sparse retrieval methods detailed at [BGE-M3 on Hugging Face](https://huggingface.co/BAAI/bge-m3).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Request for GPT API Federation**: `@tiagoefreitas` expressed interest in GPT stores with APIs, wishing for **@LangChainAI** to implement federation in OpenGPTs for using GPTs across different servers via API.
- **Embracing Open Models Over Conventional Writing**: Open models' dynamic output, such as that of **mlewd mixtral**, was lauded over traditional writing for enhancing enjoyment and productivity in content creation.
- **Rise of Specialized Technical Q&As**: `@kaycebasques` highlighted Sentry's initiative as part of a growing trend towards creating specialized technical Q&A resources for developers, enhancing information accessibility.
- **Performance Praise for Ollama Llava**: `@ashpreetbedi` shared a positive experience with **Ollama Llava**'s impressive inference speed when run locally, suggesting robust performance on consumer-grade hardware.
- **Career Choices in Tech Under Scrutiny**: With the tech industry presenting multiple paths, `@mr.osophy`'s career dilemma encapsulates the juggle between personal interest in ML Engineering and immediate job opportunities.

**Relevant Links**:

- No specific link was provided regarding federation in OpenGPTs.
- For insights into the concept of model merging in AI, reference: [Arcee and mergekit unite](https://blog.arcee.ai/arcee-and-mergekit-unite/).
- To understand the role of specialized technical Q&A platforms like Sentry, visit: [Sentry Overflow](https://technicalwriting.tools/posts/sentry-overflow/).



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Game Alchemy Unveils Hash Secrets**: There's a theory suggesting that the *unexpected delay* in generating new combinations in a game could be due to hashing mechanics, where new elements are created upon a hash miss from a pool of pre-generated combinations.
- **Visualizing the Genealogy of Game Words**: Participants are interested in creating a visual representation of the genealogy for word combinations in a game to gain deeper insights, potentially using *embeddings* to chart crafting paths.
- **Take Control with a Bookmarklet**: A *JavaScript bookmarklet* is available that leverages the game's `localStorage` to export and auto-save crafted items, enabling players to keep track of all ingredients they've crafted directly within the gaming experience.
- **Llama 2 AI Engine Revealed**: The AI powering the inventive element combinations in the game is *llama 2*, as disclosed by the creator in a posted tweet and is provided by TogetherAI.
- **Element Order Affects Crafting Success**: The sequence in which elements are combined in the game has been found to impact the crafting result, with some combinations only successful if items are layered in a specific order, and the server remembers the sequence attempted to prevent reversal on subsequent tries.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **German Language Models Boosted**: @johannhartmann reported improvements in **mt-bench-de** scores by utilizing **German dpo** and **laserRMT**, and has been merging German 7B-models using *dare_ties*. Despite sharing [links to the resources](https://github.com/mayflowergmbh/intel_orca_dpo_pairs_de), the cause of specific performance changes, including a decrease in math ability, remains unclear.
- **Research Quest for LLM Context Handling**: @nsk7153 sought research materials on large language models (LLMs) capable of managing long-context prompts, sharing a [Semantic Scholar search](https://www.semanticscholar.org/search?q=large%20language%20models%20for%20handling%20long%20context%20prompts&sort=relevance) with current findings.
- **Introducing GermanRAG for Fine-Tuning**: @rasdani announced the release of the **GermanRAG** dataset, designed for fine-tuning **Retrieval Augmented Generation** models, and provided the [GitHub repository](https://github.com/rasdani/germanrag) for access and contribution.
- **Scandinavian Benchmark Enthusiasm Projected onto German Models**: @johannhartmann expressed interest in developing a benchmark similar to [ScandEval](https://scandeval.com/mainland-scandinavian-nlg/) for evaluating German language model performance.
- **Upcoming German Hosting Service**: In the #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/) channel, flozi00 mentioned they are currently working on provisioning a German hosting service.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Diving into Training Data for Mistral-7B Open-Orca**: @njb6961 sought details on replicating **Mistral-7B Open-Orca** with its `curated filtered subset of most of our GPT-4 augmented data`. The dataset identified, [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca), comprises around 500,000 GPT-4 completions and is designed for efficient training.

- **Dataset Discovery and Confirmation**: The [SlimOrca dataset](https://huggingface.co/datasets/Open-Orca/SlimOrca) was confirmed by @ufghfigchv as the training data used for **Mistral-7B Open-Orca**. The model's **training configuration** should be accessible in the **config** subdirectory of the model's repository.

- **Commercial Contact Conundrum**: @tramojx's request for marketing contact details for a **listing and marketing proposal** went unanswered in the message history provided.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Skewed Perspectives in AI Discussions**: The conversation touches on contrasting approaches to embedding by considering the use of **whole document text embeddings** as opposed to vision embedded techniques. The discussion is framed around the potential for reimplementation of an encoder/decoder model, with a curiosity about the specific involvements of such a task.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **BentoML Eases Model Deployment**: @robotums reported a smooth experience in deploying models with **BentoML**, specifically using a VLLM backend on AWS, describing the process as "pretty easy, you just run the bento."

- **DSPy Framework Elevates Language Model Programming**: @sourya4 highlighted the launch of [DSPy](https://github.com/stanfordnlp/dspy), a Stanford initiative aimed at transforming the way foundation models are programmed. A supplemental [YouTube video](https://www.youtube.com/watch?v=Dt3H2ninoeY) provides further insight into DSPy's capabilities for creating self-improving LM pipelines.



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **AIEF Bulgaria Chapter Makes Waves**: The **AIEF Bulgaria Chapter** held its **second monthly meet-up** with 90 participants, featuring 'Lightning Talks' on a wide range of topics and fostering networking opportunities.
- **Diverse Lightning Talks Spark Interest**: Presentations on **QR Code Art, Weaving The Past, LMMs (Large Language Models), Zayo**, and strategies for **building a defensible business in the age of AI** were a highlight, with full recordings promised for the chapter's YouTube channel soon.
- **Spotlight on ChatGPT Implementation Strategy**: A session on **"ChatGPT Adoption Methodology"** by Iliya Valchanov offered insights into integrating ChatGPT into business processes, with shared resources linked through a [Google Slides document](https://docs.google.com/presentation/d/1XPMlt-qlZLagrvk4trNEI16ZSOPHRVGx).
- **Sharing Success on Social Media**: The AIEF Bulgaria lead, **@yavor_belakov**, took to LinkedIn to share highlights from the meet-up, reflecting the vibrancy and advancements of the **AI engineering community** involved with AIEF.
- **Presentations Capturing Technical Innovation**: The slides from the meet-up presentations, including those on QR Code Art, historical knitting, an LLM command-line tool, reimagined employee management with Zayo, and robust business models in AI, underscore the technical diversity and innovation within the AIEF community.



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1203252408356970547) (1738 messages🔥🔥🔥): 

- **Plugin Development Adventures**: User `@doctorshotgun` is working on coding a plugin for [Polymind](https://link.to.polymind), aiming to improve its article search functionality with PubMed's API. They are currently incorporating `pymed` to construct and parse search queries, encountering challenges with the sorting and relevance of search results.

- **Exploring Miqu**: Several users, including `@nextdimension` and `@netrve`, discuss the usefulness of the local LLM model miqu-1-70b. While some find it useful, others report it produces unsatisfactory results, which may be attributed to its generation parameters.

- **Interest in Mixtral Instruct**: Discussions regarding the efficiency and quality of responses are ongoing, with users like `@doctorshotgun` highlighting slower response times when processing large RAG contexts on the 70B model.

- **BagelMIsteryTour Emerges**: The BagelMIsteryTour-v2-8x7B-GGUF model receives praise as `@ycros` attributes its success to merging the Bagel model with Mixtral Instruct. The model is good for tasks like roleplay (RP) and general Q&A, according to user testing.

- **Oobabooga vs Silly Tavern**: User `@parogar` expresses frustration over Oobabooga (likely a local LLM runner) API changes that hinder Silly Tavern's connection. They are seeking ways to revert to a previous version that was more compatible.

**Links mentioned**:

- [Download Data - PubMed](https://pubmed.ncbi.nlm.nih.gov/download/#annual-baseline): PubMed data download page.
- [Blades Of Glory Will Ferrell GIF - Blades Of Glory Will Ferrell No One Knows What It Means - Discover &amp; Share GIFs](https://tenor.com/view/blades-of-glory-will-ferrell-no-one-knows-what-it-means-provocative-gif-5313360): Click to view the GIF
- [movaxbx/OpenHermes-Emojitron-001 · Hugging Face](https://huggingface.co/movaxbx/OpenHermes-Emojitron-001): no description found
- [modster (mod ster)](https://huggingface.co/modster): no description found
- [BagelMIsteryTour-v2-8x7B-Q4_K_S.gguf · Artefact2/BagelMIsteryTour-v2-8x7B-GGUF at main](https://huggingface.co/Artefact2/BagelMIsteryTour-v2-8x7B-GGUF/blob/main/BagelMIsteryTour-v2-8x7B-Q4_K_S.gguf): no description found
- [NEW DSPyG: DSPy combined w/ Graph Optimizer in PyG](https://www.youtube.com/watch?v=rqR3LeR09gc): DSPyG is a new optimization, based on DSPy, extended w/ graph theory insights. Real world example of a Multi Hop RAG implementation w/ Graph optimization.New...
- [Artefact2/BagelMIsteryTour-v2-8x7B-GGUF at main](https://huggingface.co/Artefact2/BagelMIsteryTour-v2-8x7B-GGUF/tree/main): no description found
- [Terminator (4K) Breaking Into Skynet](https://www.youtube.com/watch?v=CNZXYAkmFpM): Terminator (4K) Breaking Into Skynet
- [AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA](https://youtu.be/7OUMZoHpVnM?feature=shared): #meme #memes #funny #funnyvideo
- [152334H/miqu-1-70b-sf · Hugging Face](https://huggingface.co/152334H/miqu-1-70b-sf): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1203259027690229821) (678 messages🔥🔥🔥): 

- **Discussions on Model Performance and Preferences**: Users shared experiences with various AI models for roleplaying, with mentions of **goliath 120b**, **mixtral models**, and variations like **limaRP** and **sensual nous instruct**. **@potatooff** suggested the [HamSter v0.2 model](https://huggingface.co/PotatoOff/HamSter-0.2) for uncensored roleplay with a detailed character card, using Llama2 prompt template with chat-instruct.
  
- **Technical Deep Dive into DPO and Model Training**: There was a technical conversation about the large VRAM usage for DeeperSpeed (DPO) and its impact on training AI models, with various users discussing their struggles with fitting models like qlora dpo on GPUs due to, as **@doctorshotgun** explained, the **gradient_checkpointing_kwargs** setting `use_reentrant` being set to `False` by default in Axolotl, which they suggest changing for less VRAM usage.

- **Seeking Advice for Optimizing Character Cards**: **@johnrobertsmith** sought advice on optimizing character cards for AI roleplay, with suggestions to keep character descriptions around 200 tokens and use lorebooks for complex details like world spells. **@mrdragonfox** shared an example character card and endorsed using lorebooks for better character definition.

- **Exploring Various Models' VRAM Consumption**: Users including **@c.gato,** **@giftedgummybee,** and **@kalomaze** discussed the resource-intensive nature of certain AI models, specifically when using DPO, and shared their experiences with large consumption due to duplications needed for DPO's caching requirements.

- **Miscellaneous Conversations and Jokes**: Amongst the technical and performance-focused discussions, there were lighter moments with users joking about winning arguments with AI (**@mr.devolver**) and random jabs at found objects being "smelly" (**@kaltcit** and **@stoop poops**).

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1112690728531918948/118465737669188415): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Chub](https://www.chub.ai/lorebooks/sr_misterioso/advanced-personality-traits-658eaf07): Find, share, modify, convert, and version control characters and other data for conversational large language models (LLMs). Previously/AKA Character Hub, CharacterHub, CharHub, CharaHub, Char Hub.
- [PotatoOff/HamSter-0.2 · Hugging Face](https://huggingface.co/PotatoOff/HamSter-0.2): no description found
- [cognitivecomputations/dolphin-2_6-phi-2 · Hugging Face](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2): no description found
- [LoneStriker/Mixtral-8x7B-Instruct-v0.1-LimaRP-ZLoss-6.0bpw-h6-exl2 · Hugging Face](https://huggingface.co/LoneStriker/Mixtral-8x7B-Instruct-v0.1-LimaRP-ZLoss-6.0bpw-h6-exl2): no description found
- [LoneStriker/limarp-miqu-1-70b-5.0bpw-h6-exl2 · Hugging Face](https://huggingface.co/LoneStriker/limarp-miqu-1-70b-5.0bpw-h6-exl2): no description found
- [Significantly increased VRAM usage for Mixtral qlora training compared to 4.36.2? · Issue #28339 · huggingface/transformers](https://github.com/huggingface/transformers/issues/28339#issuecomment-1879894108): System Info The environment is a Runpod container with python 3.10, single A100 80gb, transformers 4.37.0dev (3cefac1), using axolotl training script (https://github.com/OpenAccess-AI-Collective/ax...

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1203424550285742172) (18 messages🔥): 

- **Choosing the Right Model for Code Generation**: `@Naruto08` is seeking advice on which model to train for code generation with a custom dataset in [INST] {prompt} [/INST] format. They have 24GB of GPU memory available and want to ensure proper model and training approach selection.
  
- **Inquiry on Specific Model Fine-Tuning**: User `@709986_` asked if the model *em_german_mistral_v01.Q5_0.gguf* can undergo fine-tuning, but did not provide details on the desired outcome or specifics of the fine-tuning process.

- **Finetuning Flan-T5 on Limited Resources**: `@tom_lrd` queried about the dataset size and hardware requirements for fine-tuning a flan-t5 model, while `@rolandtannous` responded with experience of performing LoRA fine-tuning on flan-t5-base using AWS instances and shared a relevant dataset located at [DialogSum Dataset on Huggingface](https://huggingface.co/datasets/knkarthick/dialogsum).

- **Accessible Fine-Tuning of FLAN-T5**: `@rolandtannous` shared details about the ease of fine-tuning FLAN-T5 base models, given their size (approximately 900MB-1GB), and pointed to the use of a p3.2xlarge AWS EC2 Instance with a NVIDIA V100 by Phil Schmidt in related experiments. They also provided a comprehensive [guide on fine-tuning FLAN-T5](https://www.philschmid.de/fine-tune-flan-t5) for dialogue summarization using the SAMSUM dataset.

- **Clarifying "Uncensored" Models on Huggingface**: `@thisisloading` inquired about "uncensored" models on Huggingface, leading to a discussion about the process of removing alignment from such models, as detailed by Eric Hartford in his blog post: ["Uncensored Models"](https://erichartford.com/uncensored-models). The procedure is akin to "surgically" removing alignment components from a base model, enabling further customization through fine-tuning.

**Links mentioned**:

- [Uncensored Models](https://erichartford.com/uncensored-models): I am publishing this because many people are asking me how I did it, so I will explain. https://huggingface.co/ehartford/WizardLM-30B-Uncensored https://huggingface.co/ehartford/WizardLM-13B-Uncensore...
- [Fine-tune FLAN-T5 for chat &amp; dialogue summarization](https://www.philschmid.de/fine-tune-flan-t5): Learn how to fine-tune Google&#x27;s FLAN-T5 for chat &amp; dialogue summarization using Hugging Face Transformers.

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1203391772223602708) (5 messages): 

- **Innovative Model Merging Technique**: `@maldevide` detailed a novel approach to *model merging*, where layers are partitioned into buckets and merged individually, with a unique treatment for **kvq** that involves a 100% merge weight but with a high drop rate of 92%.
- **Partitioned Layer Merging Results**: Following the new approach, `@maldevide` mentioned that each partition, if there are four, would be merged at a **68% drop rate**, suggesting this specific drop rate has been impactful.
- **Interest in the New Approach**: `@alphaatlas1` showed interest in `@maldevide`'s merging method, asking to see the configuration or the custom code.
- **Access to New Model Merging Code**: `@maldevide` responded to the request by providing a link to their configuration in the form of a [GitHub Gist](https://gist.github.com/maldevide/08829eada04ad9bd78e46c1a3787d42b), allowing others to view and potentially use the described technique.

**Links mentioned**:

[tinyllama-merge.ipynb](https://gist.github.com/maldevide/08829eada04ad9bd78e46c1a3787d42b): GitHub Gist: instantly share code, notes, and snippets.

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1203477948209045505) (6 messages): 

- **Local Model Lookup Quest**: `@aletheion` is seeking help on how to implement a feature where a chatbot can perform a lookup action in a local/vector database to provide answers while keeping everything offline. They expressed openness to using existing frameworks or solutions.
  
- **h2ogpt Suggested for Local Bot Implementation**: `@wildcat_aurora` shared a [GitHub repository for h2ogpt](https://github.com/h2oai/h2ogpt), which offers private Q&A and summarization with local GPT, supporting 100% privacy, and touted compatibility with various models, which could be a solution for `@aletheion`'s query.

- **API Confusion Unraveled**: `@sunija` expressed frustration over Ooba's API requiring a "messages" field despite documentation suggesting it wasn't necessary, but then realized the mistake and self-recognized the dislike for making web requests.

- **Model Evaluation Success**: `@london` reported that models, Code-13B and Code-33, succeeded in evaluations on EvalPlus and other platforms after being asked for submission by another user.

- **Chatbot App Aims for Character-Specific Long-Term Memory**: `@vishnu_86081` is looking for guidance on setting up ChromaDB for their chatbot app that allows users to chat with multiple characters, aiming to store and retrieve character-specific messages using a vector DB for long-term memory purposes.

**Links mentioned**:

[GitHub - h2oai/h2ogpt: Private Q&amp;A and summarization of documents+images or chat with local GPT, 100% private, Apache 2.0. Supports Mixtral, llama.cpp, and more. Demo: https://gpt.h2o.ai/ https://codellama.h2o.ai/](https://github.com/h2oai/h2ogpt): Private Q&amp;amp;A and summarization of documents+images or chat with local GPT, 100% private, Apache 2.0. Supports Mixtral, llama.cpp, and more. Demo: https://gpt.h2o.ai/ https://codellama.h2o.ai/ -...

  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1203326927276875866) (56 messages🔥🔥): 

- **GPT-4's Lyric Quirks**: `@cccntu` discussed the limitations of GPT-4 in generating lyrics accurately, mentioning that using perplexity with search yields better results than the AI, which tends to fabricate content.
- **Greentext Generation Challenges**: `@euclaise` suggested that 4chan's greentext format may be difficult for AI to learn due to lack of training data, while `@teknium` shared a snippet showcasing an AI's attempt to mimick a greentext narrative involving Gaia's Protector, highlighting the challenges in capturing the specific storytelling style.
- **Call for Indian Language AI Innovators**: `@stoicbatman` invited developers and scientists working on AI for Indian languages to apply for GPU computing resources and infrastructure support provided by IIT for advancing regional language research.
- **Llama2 Pretrained on 4chan Data?**: `@stefangliga` claimed that 4chan content is in fact part of llama2's pretraining set, countering the assumption that it might be deliberately excluded.
- **Apple Accused of Creating Barriers for AR/VR Development**: `@nonameusr` criticized Apple's approach to its technology ecosystem, arguing that the company's restrictive practices like charging an annual fee just to list apps and the lack of immersive VR games for Vision Pro are hindrances for AR/VR advancement.

**Links mentioned**:

- [Skull Issues GIF - Skull issues - Discover &amp; Share GIFs](https://tenor.com/view/skull-issues-gif-13031152103567454559): Click to view the GIF
- [Join the Bittensor Discord Server!](https://discord.gg/JkRGPEPY): Check out the Bittensor community on Discord - hang out with 20914 other members and enjoy free voice and text chat.
- [Watch A Fat Cat Dance An American Dance Girlfriend GIF - Watch a fat cat dance an American dance Girlfriend Meme - Discover &amp; Share GIFs](https://tenor.com/view/watch-a-fat-cat-dance-an-american-dance-girlfriend-meme-gif-6193372123771306115): Click to view the GIF
- [4chan search](https://4chansearch.com/?q=%3Ebe+me&s=1): no description found
- [ExLlamaV2: The Fastest Library to Run LLMs](https://www.youtube.com/watch?v=N5lDUZRI8sc): A fast inference library for running LLMs locally on modern consumer-class GPUshttps://github.com/turboderp/exllamav2https://colab.research.google.com/github...
- [Indic GenAI Project](https://forms.gle/7iZnQjU9rwCr7wF9A): We are calling all developers, scientists, and others out there working in Generative AI and building models for Indian languages. To help the research community, we are bringing together the best min...
- [DarwinAnim8or/greentext · Datasets at Hugging Face](https://huggingface.co/datasets/DarwinAnim8or/greentext): no description found
- [Llama GIF - Llama - Discover &amp; Share GIFs](https://tenor.com/view/llama-gif-21325230): Click to view the GIF

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1203302897522057286) (42 messages🔥): 

- **Embracing the EFT Revolution**: @euclaise shared a paper introducing **emulated fine-tuning (EFT)**, a novel technique to independently analyze the knowledge gained from pre-training and fine-tuning stages of language models, using an RL-based framework. The paper challenges the understanding of pre-trained and fine-tuned models' knowledge and skills interplay, proposing to potentially combine them in new ways ([Read the paper](https://arxiv.org/abs/2310.12962)).
- **Frankenmerge Hits the Ground**: @nonameusr introduced **miqu-1-120b-GGUF**, a frankenmerged language model built from *miqu-1-70b* and inspired by other large models like Venus-120b-v1.2, MegaDolphin-120b, and goliath-120b, highlighting the [CopilotKit](https://github.com/CopilotKit/CopilotKit) support ([Explore on Hugging Face](https://huggingface.co/wolfram/miqu-1-120b)).
- **FP6 Quantization on GPU**: @jiha discussed a new six-bit quantization method for large language models called **TC-FPx**, and queried its implementation and comparative performance, with @.ben.com noting the optimal precision for the majority of tasks and its practical benefits in specific use-cases ([Check the abstract](https://huggingface.co/papers/2401.14112)).
- **Mercedes-Benz of Models**: @gabriel_syme surmised the potential sizes of new models being discussed, with users speculating about the upcoming **Qwen 2** model and its performance compared to predecessors like **Wen-72B**. Chatter in this topic included expectations of model sizes and benchmark performance.
- **The New Merge on the Block**: @nonameusr presented **MergeMonster**, an unsupervised algorithm for merging Transformer-based language models, that features experimental merge methods and performs evaluations before and after merging each layer ([Discover on GitHub](https://github.com/Gryphe/MergeMonster)).

**Links mentioned**:

- [An Emulator for Fine-Tuning Large Language Models using Small Language Models](https://arxiv.org/abs/2310.12962): Widely used language models (LMs) are typically built by scaling up a two-stage training pipeline: a pre-training stage that uses a very large, diverse dataset of text and a fine-tuning (sometimes, &#...
- [wolfram/miqu-1-120b · Hugging Face](https://huggingface.co/wolfram/miqu-1-120b): no description found
- [Paper page - FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric
  Algorithm-System Co-Design](https://huggingface.co/papers/2401.14112): no description found
- [Tweet from Binyuan Hui (@huybery)](https://x.com/huybery/status/1754163638259388525?t=cDduW8-dHQD_fekk1_Qajg&s=33): Waiting patiently for the flowers to bloom 🌸
- [GitHub - Gryphe/MergeMonster: An unsupervised model merging algorithm for Transformers-based language models.](https://github.com/Gryphe/MergeMonster): An unsupervised model merging algorithm for Transformers-based language models. - GitHub - Gryphe/MergeMonster: An unsupervised model merging algorithm for Transformers-based language models.

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1203261031900647424) (550 messages🔥🔥🔥): 

- **Quantizing Emojis**: Members `@agcobra1` and `@n8programs` were engaged in a teaching session on how to quantize models using `llama.cpp`. The process involves cloning the model, pulling large files with `git lfs pull`, and then using the `convert.py` script for conversion and `./quantize` for quantization.

- **Qwen2 Release Anticipation**: The Qwen model team was hinting at the release of Qwen2, expected to be a strong contender in benchmarks, potentially even surpassing the performance of Mistral medium. `@bratao` shared a GitHub link hinting at Qwen2's upcoming reveal.

- **Discussions on Future Digital Interfaces**: `@nonameusr` and `@n8programs` delved into a speculative conversation about the potential future of brain-computer interfaces, imagining scenarios where thoughts could directly interact with digital systems without the need for traditional input methods.

- **Text Generation UI and API Ergonomics**: `@light4bear` recommended text-generation-webui for easily experimenting with models, whereas `@.ben.com` offered an OpenAI API compatible server experiment with ExLlamaV2 for testing downstream clients.

- **Experiments and Comparisons in Preference Tuning**: `@dreamgen` inquired about the practical comparison between KTO, IPO, and DPO methods for aligning language models. A subsequent Hugging Face blog post was referenced that discusses corrected IPO implementation results, showing IPO on par with DPO and better than KTO in preference settings.

**Links mentioned**:

- [Tweet from Binyuan Hui (@huybery)](https://fxtwitter.com/huybery/status/1754163638259388525?t=cDduW8-dHQD_fekk1_Qajg&s=33): Waiting patiently for the flowers to bloom 🌸
- [Google Colaboratory](https://colab.research.google.com/drive/1P646NEg33BZy4BfLDNpTz0V0lwIU3CHu): no description found
- [CodeFusion: A Pre-trained Diffusion Model for Code Generation](https://arxiv.org/abs/2310.17680): Imagine a developer who can only change their last line of code, how often would they have to start writing a function from scratch before it is correct? Auto-regressive models for code generation fro...
- [cxllin/StableHermes-3b · Hugging Face](https://huggingface.co/cxllin/StableHermes-3b): no description found
- [movaxbx/OpenHermes-Emojitron-001 · Hugging Face](https://huggingface.co/movaxbx/OpenHermes-Emojitron-001): no description found
- [NousResearch/Nous-Capybara-3B-V1.9 · Hugging Face](https://huggingface.co/NousResearch/Nous-Capybara-3B-V1.9): no description found
- [tsunemoto/OpenHermes-Emojitron-001-GGUF · Hugging Face](https://huggingface.co/tsunemoto/OpenHermes-Emojitron-001-GGUF): no description found
- [wolfram/miquliath-120b · Hugging Face](https://huggingface.co/wolfram/miquliath-120b): no description found
- [Social Credit GIF - Social Credit - Discover &amp; Share GIFs](https://tenor.com/view/social-credit-gif-23329982): Click to view the GIF
- [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning): no description found
- [Tweet from AI Breakfast (@AiBreakfast)](https://x.com/aibreakfast/status/1754008072828158416?s=46): Google’s Gemini Ultra was just confirmed for release on Wednesday.   Ultra beats GPT-4 in 7 out of 8 benchmark tests, and is the first model to outperform human experts on MMLU (massive multitask lang...
- [text-generation-webui/requirements.txt at main · oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui/blob/main/requirements.txt): A Gradio web UI for Large Language Models. Supports transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama models. - oobabooga/text-generation-webui
- [GitHub - PygmalionAI/aphrodite-engine: PygmalionAI&#39;s large-scale inference engine](https://github.com/PygmalionAI/aphrodite-engine): PygmalionAI&#39;s large-scale inference engine. Contribute to PygmalionAI/aphrodite-engine development by creating an account on GitHub.
- [Kind request for updating MT-Bench leaderboards with Qwen1.5-Chat series · Issue #3009 · lm-sys/FastChat](https://github.com/lm-sys/FastChat/issues/3009): Hi LM-Sys team, we would like to present the generation results and self-report scores of Qwen1.5-7B-Chat, Qwen1.5-14B-Chat, and Qwen1.5-72B-Chat on MT-Bench. Could you kindly help us verify them a...
- [GitHub - bjj/exllamav2-openai-server: An OpenAI API compatible LLM inference server based on ExLlamaV2.](https://github.com/bjj/exllamav2-openai-server): An OpenAI API compatible LLM inference server based on ExLlamaV2. - GitHub - bjj/exllamav2-openai-server: An OpenAI API compatible LLM inference server based on ExLlamaV2.
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
- [GitHub - lucidrains/self-rewarding-lm-pytorch: Implementation of the training framework proposed in Self-Rewarding Language Model, from MetaAI](https://github.com/lucidrains/self-rewarding-lm-pytorch): Implementation of the training framework proposed in Self-Rewarding Language Model, from MetaAI - GitHub - lucidrains/self-rewarding-lm-pytorch: Implementation of the training framework proposed in...
- [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers): An increasing number of the machine learning (ML) models we build at Apple each year are either partly or fully adopting the [Transformer…
- [openbmb/MiniCPM-V · Hugging Face](https://huggingface.co/openbmb/MiniCPM-V): no description found
- [GitLive](https://git.live/): Real-time code collaboration inside any IDE 
- [Context Free Grammar Constrained Decoding (ebnf interface, compatible with llama-cpp) by Saibo-creator · Pull Request #27557 · huggingface/transformers](https://github.com/huggingface/transformers/pull/27557): What does this PR do? This PR adds a new feature (Context Free Grammar Constrained Decoding) to the library. There is already one PR(WIP) for this feature( #26520 ), but this one has a different mo...
- [Neural Engine Support · ggerganov/llama.cpp · Discussion #336](https://github.com/ggerganov/llama.cpp/discussions/336): Would be cool to be able to lean on the neural engine. Even if it wasn&#39;t much faster, it&#39;d still be more energy efficient I believe.

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1203253416688881735) (90 messages🔥🔥): 

- **Hermes Model Confusion Cleared**: `@teknium` clarified the difference between **Nous Hermes 2 Mixtral** and **Open Hermes 2 and 2.5**, which are 7B Mistrals, with Open Hermes 2.5 having added 100,000 code instructions.
- **Mixtral's Memory-Antics**: `@teknium` and `@intervitens` discussed that Mixtral models requires about 8x the VRAM of a 7B model and about 40GB in 4bit precision. `@intervitens` later mentioned that with 8bit cache and optimized settings, 3.5 bpw with full context could fit.
- **Prompt Probing**: `@tempus_fugit05` received corrections from `@teknium` and `.ben.com` on the prompt format they've been using with the Nous SOLAR model, pointing to usage of incorrect prompt templates.
- **Expert Confusion in MoEs Explained**: `.ben.com` explained how in MoEs, experts are blended proportionally to the router's instructions, emphasizing that while experts are chosen per-layer, their outputs must add up correctly in the final mix.
- **Lone-Arena For LLM Chatbot Testing**: `.ben.com` shared [Lone-Arena](https://github.com/Contextualist/lone-arena), a self-hosted chatbot arena code repository on GitHub for personal testing of LLMs.

**Links mentioned**:

- [Tweet from Geronimo (@Geronimo_AI)](https://x.com/geronimo_ai/status/1753685586634797113?s=46): phi-2-OpenHermes-2.5 https://huggingface.co/g-ronimo/phi-2-OpenHermes-2.5  ↘️ Quoting Teknium (e/λ) (@Teknium1)   Today I have a huge announcement.  The dataset used to create Open Hermes 2.5 and Nous...
- [NousResearch/Nous-Hermes-2-SOLAR-10.7B · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B#prompt-format): no description found
- [teknium/OpenHermes-2.5-Mistral-7B · Hugging Face](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B): no description found
- [GitHub - daveshap/SparsePrimingRepresentations: Public repo to document some SPR stuff](https://github.com/daveshap/SparsePrimingRepresentations): Public repo to document some SPR stuff. Contribute to daveshap/SparsePrimingRepresentations development by creating an account on GitHub.
- [GitHub - Contextualist/lone-arena: Self-hosted LLM chatbot arena, with yourself as the only judge](https://github.com/Contextualist/lone-arena): Self-hosted LLM chatbot arena, with yourself as the only judge - GitHub - Contextualist/lone-arena: Self-hosted LLM chatbot arena, with yourself as the only judge

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1203252881361342464) (225 messages🔥🔥): 

- **Persistent Phantom**: User `@nikofus` reported that even after closing LM Studio UI, it continued to show in the task manager and use CPU resources. To address this, `@heyitsyorkie` suggested force killing the process and creating a bug report in a specific channel.

- **LM Studio's Ghostly Grip**: `@vett93` questioned why LM Studio remains active in Task Manager after the window is closed. `@heyitsyorkie` explained it's a known bug and the current solution is to end the process manually.

- **AVX Instruction Frustration**: Users `@rachid_rachidi` and `@sica.rios` faced errors due to their processors not supporting AVX2 instructions. `@heyitsyorkie` clarified that LM Studio requires AVX2 support, but a beta version is available for CPUs with only AVX.

- **Roaming for ROCm**: `@neolithic5452` inquired about getting LM Studio to use GPU compute on an AMD 7900XTX GPU instead of just CPU for a Windows 11 setup. `@quickdive.` advised using a special beta version of LM Studio that supports ROCm for AMD GPU compute capability, available in the channel pinned messages.

- **Whispers of Integration**: `@lebonchasseur` showed interest in experiences combining Whisper and Llama models with LM Studio, whilst `@muradb` inquired about suitable vision models. Users were pointed towards Llava and explicitly to one on the Hugging Face model page.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1138544400771846174/1201187492414619791): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html): no description found
- [jartine/llava-v1.5-7B-GGUF · Hugging Face](https://huggingface.co/jartine/llava-v1.5-7B-GGUF): no description found
- [Advanced Vector Extensions - Wikipedia](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions): no description found
- [HuggingChat](https://huggingface.co/chat/): Making the community's best AI chat models available to everyone.
- [503 Service Unavailable - HTTP | MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/503): The HyperText Transfer Protocol (HTTP) 503 Service Unavailable server error response code indicates that the server is not ready to handle the request.
- [teknium/openhermes · Datasets at Hugging Face](https://huggingface.co/datasets/teknium/openhermes): no description found
- [Yet Another LLM Leaderboard - a Hugging Face Space by mlabonne](https://huggingface.co/spaces/mlabonne/Yet_Another_LLM_Leaderboard): no description found
- [GitHub - by321/safetensors_util: Utility for Safetensors Files](https://github.com/by321/safetensors_util): Utility for Safetensors Files. Contribute to by321/safetensors_util development by creating an account on GitHub.
- [Terminator Terminator Robot GIF - Terminator Terminator Robot Looking - Discover &amp; Share GIFs](https://tenor.com/view/terminator-terminator-robot-looking-flex-cool-robot-gif-978532213316794273): Click to view the GIF
- [Pinokio](https://pinokio.computer/): AI Browser

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1203271600414269470) (149 messages🔥🔥): 

- **Model Recommendation for Specific PC Specs**: `@mesiax.` inquired about the best performing model for a PC with 32GB RAM and 12GB VRAM that fully utilizes the GPU. While `@wolfspyre` offered some advice, ultimately recommending that they start testing and learning through experience, as no one-size-fits-all solution exists.

- **Model Updates and Notifications**: User `@josemanu72` asked whether they need to manually update a model when a new version is published. `@heyitsyorkie` clarified that updating is a manual process, as LLMs create a whole new model rather than update an existing one.

- **VP Code Versus IntelliJ Plugins**: `@tokman` expressed a preference for IntelliJ over VS Code and inquired about the availability of a similar plugin for IntelliJ after discovering a useful extension for VS Code. `@heyitsyorkie` mentioned a possible workaround with the IntelliJ plugin supporting local models through server mode.

- **Continue Integration and Usage**: `@wolfspyre` discussed the benefits of [Continue.dev](http://continue.dev/), which facilitates coding with any LLM in an IDE, and `@dagbs` pointed to a channel that could be a general discussion space for integrations.

- **Query on Image Generation Models**: `@kecso_65737` sought recommendations for image generation models. `@fabguy` suggested Stable Diffusion (SDXL) but noted it's not available on LM Studio, and `@heyitsyorkie` emphasized the same while mentioning Automatic1111 for ease of use outside LM Studio.

**Links mentioned**:

- [Continue](http://continue.dev/): no description found
- [NeverSleep/MiquMaid-v1-70B-GGUF · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v1-70B-GGUF): no description found
- [Don't ask to ask, just ask](https://dontasktoask.com/): no description found
- [⚡️ Quickstart | Continue](https://continue.dev/docs/quickstart): Getting started with Continue
- [aihub-app/ZySec-7B-v1-GGUF · Hugging Face](https://huggingface.co/aihub-app/ZySec-7B-v1-GGUF): no description found
- [Replace Github Copilot with a Local LLM](https://www.youtube.com/watch?v=F1bXfnrzAxM): If you&#39;re a coder you may have heard of are already using Github Copilot. Recent advances have made the ability to run your own LLM for code completions and ...
- [John Travolta GIF - John Travolta - Discover &amp; Share GIFs](https://tenor.com/view/john-travolta-gif-25290651): Click to view the GIF
- [christopherthompson81/quant_exploration · Datasets at Hugging Face](https://huggingface.co/datasets/christopherthompson81/quant_exploration#quants): no description found

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1203451447698784286) (11 messages🔥): 

- **Model Download Mystery**: Stochmal faced issues with downloading a model, encountering a 'fail' message without an option to retry or resume the download process.
- **Apple Silicon VRAM Puzzle**: `@musenik` reported that even with 90GB of VRAM allocated, the model **Miquella 120B q5_k_m.gguf** fails to load on **LM Studio** on Apple Silicon, whereas it successfully loads on Faraday.
- **LM Studio vs. Faraday**: `@yagilb` shared a hypothesis that **LM Studio** might try to load the whole model into VRAM on macOS, which could cause issues, hinting at a future update to address this.
- **In Search of Hidden Overheads**: `@musenik` suggested looking into potential unnecessary overhead in LM Studio when loading models, as Faraday loads the same model with a switch for VRAM and functions correctly.
- **Download Resumability Requested**: `@petter5299` inquired about the future addition of a *resume download feature* in LM Studio, expressing frustration over downloads restarting after network interruptions.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1203271437276946432) (217 messages🔥🔥): 

- **Seeking General-Purpose Model Advice**: User `@mesiax.` inquired about the best performance model to run locally on a PC with 32GB of RAM and 12GB of VRAM, wishing to utilize the GPU for all processing. Fellow users didn’t respond with specific model recommendations, instead, conversations shifted towards detailed hardware discussions on GPUs, RAM speeds, and PCIe bandwidth for running large language models.
- **RAM Speed vs. GPU VRAM Debate**: Users, including `@goldensun3ds`, discussed the influence of RAM speed on running large models, considering an upgrade from DDR4 3000MHz to 4000MHz or faster. Conversations revolved around system trade-offs, such as RAM upgrades versus adding GPUs, and touched upon hardware compatibility and performance expectations.
- **P40 GPU Discussions Spark Curiosity and Concern**: Members like `@goldensun3ds` and `@heyitsyorkie` debated the suitability of Nvidia Tesla P40 GPUs for running large models, such as the 120B Goliath. Issues raised included driver compatibility, potential bottlenecks when pairing with newer GPUs, and P40's lack of support for future model updates.
- **Ryzen CPUs and DDR5 RAM Get a Mention**: Discussion by `@666siegfried666` and `.ben.com` briefly pointed out the advantages of certain Ryzen CPUs and DDR5 RAM for local model inference, although the X3D cache's effectiveness and Navi integrated NPUs were debated.
- **Viable High-VRAM Configurations Explored**: Users like `@quickdive.` and `@heyitsyorkie` examined the potential of different GPU setups, including P40s, 3090s, and 4090s for deep learning tasks. The consensus leaned towards using higher VRAM GPUs to avoid bottlenecks and improve performance.

**Links mentioned**:

- [Rent GPUs | Vast.ai](https://vast.ai): Reduce your cloud compute costs by 3-5X with the best cloud GPU rentals. Vast.ai&#x27;s simple search interface allows fair comparison of GPU rentals from all providers.
- [B650 UD AC (rev. 1.0) Key Features | Motherboard - GIGABYTE Global](https://www.gigabyte.com/Motherboard/B650-UD-AC-rev-10#kf): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/13omfzw/comment/jl52q44/): no description found
- [Nvidia's H100 AI GPUs cost up to four times more than AMD's competing MI300X &mdash; AMD's chips cost $10 to $15K apiece; Nvidia's H100 has peaked beyond $40,000: Report](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-h100-ai-gpus-cost-up-to-four-times-more-than-amds-competing-mi300x-amds-chips-cost-dollar10-to-dollar15k-apiece-nvidias-h100-has-peaked-beyond-dollar40000): AMD on track to generate billions on its Instinct MI300 GPUs this year, says Citi.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/13n8bqh/my_results_using_a_tesla_p40/): no description found
- [EVGA GeForce RTX 3090 FTW3 ULTRA HYBRID 24GB GDDR6X Graphic Card 843368067106 | eBay](https://www.ebay.com/itm/276294867784?epid=24042942228&hash=item4054751b48:g:2OcAAOSwhsZlrDOb): no description found

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1203379442148450405) (42 messages🔥): 

- **Image Analysis Capability in Question**: `@syslot` confirmed that **Llava-v1.6-34b** operates well, while `@palpapeen` expressed difficulties making it analyze images, despite the vision adapter being installed and an ability to send images in chat. For `@palpapeen`, the configuration worked for Llava1.5 7B but not Llava1.6 34B.

- **Discussions on Model and Processor Compatibility**: `@vic49.` mentioned an issue discussed on GitHub: separating the model and processor using GGUF formatting prevents the GGUF from utilizing the higher resolution of version 1.6.

- **The ROCm Path Struggle on Windows 11 with AMD**: `@sierrawhiskeyhotel` experienced a "Model error" with AMD hardware on Windows 11 but eventually resolved it by turning off internal graphics and using GPU Preference settings, confirming successful use of an AMD Radeon RX 7900 XTX.

- **Desire for More GPU Control Expressed**: Following a discussion on troubleshooting ROCm configuration and GPU utilization, `@fabguy`, `@heyitsyorkie`, and `@yagilb` concurred that more control over which GPU is used would be beneficial, an issue addressed within the community.

- **New Windows Beta Build and Reported Issues**: `@yagilb` shared a link to a new Windows beta build, featuring an improvement to how LM Studio shows RAM and CPU counts. `@fabguy` reported inconsistent CPU usage metrics and lingering processes after closing the app, while `@heyitsyorkie` suggested the process bug was not easily reproducible.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1195858490338594866/1197335753307467906): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1203750081648660500/1203750): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1202562325257650186): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1203750081648660500/1203750081648660500): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1203364915042189353) (4 messages): 

- **Error in POST Request**: `@merpdragon` shared a pastebin link containing an error they encountered when making a POST request to `/v1/chat/completions`. The [shared log](https://pastebin.com/NSB1H6az) indicates an issue while processing the prompt about children driving a car.
- **LM Studio Setup with Autogen Issues**: `@j_rdie` mentioned having LM Studio set up with autogen, confirming the token and model verification, but facing an issue where the model won't output directly, only during autogen testing.
- **Starting with Autogen Guide**: `@samanofficial` inquired about how to start with autogen, and `@dagbs` provided a link to a channel within Discord for further guidance. However, the specific content or instructions from the link cannot be discerned from the message.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1167546228813336686/1200572588850688101): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [[2024-02-03 16:41:48.517] [INFO] Received POST request to /v1/chat/completions w - Pastebin.com](https://pastebin.com/NSB1H6az): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.

  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1203686204965650513) (2 messages): 

- **Local LLM Setup Needs ENV Var or Code Alteration**: User `@tok8888` posted a code snippet illustrating that for local setup, one must either set an environment variable for `openai_api_key` or modify the code to include the API key directly. They showed an example with the API key set to `"foobar"` and altered the `ChatOpenAI` initialization.

- **Inquiry About LM Studio for Appstore Optimization**: User `@disvita` asked the group how they can utilize LM Studio for App Store optimization, but provided no further context or details in their query.
  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1203257040374603866) (278 messages🔥🔥): 

- **LLama3 Speculation and Mixtral Tips Wanted**: User `@frosty04212` wondered if Llama3 would have the same architecture but different training data, while `@sheldada` sought tips for prompting Mixtral effectively, mentioning odd results. `@ethux` inquired about the use case of Mixtral, whether through API, self-hosted, or other methods.
  
- **Character Conundrums with Mistral**: `@cognitivetech` brought up issues with Mistral's handling of special characters, noting problems with certain characters like the pipe (`|`) and others when processing academic text. They discussed challenges with input over 10,000 characters and the variability of results with different characters and model variations.

- **Model Performance Discussions**: `@cognitivetech` and `@mrdragonfox` exchanged observations on model inference times with OpenHermes 2.5 versus Mistral, noting differences when using different tooling. They also touched on the phenomenon known as "lost in the middle," where performance issues arise dealing with relevant information in the middle of long contexts.

- **Aspiring Image Model Developers Connect**: User `@qwerty_qwer` offered 600 million high-quality images to anyone developing an image generative model, sparking discussions with `@i_am_dom` and `@mrdragonfox` on the feasibility and computational challenges of training a model from scratch.

- **Function Calling Feature Request and Office Hours Critique**: `@jujuderp` lamented the absence of function calling and JSON response mode in Mistral API, referencing a community post, while `@i_am_dom` offered a critique of the office hours sessions, comparing them to Google's approach on Bard discord and noting a lack of informative responses from Mistral AI.

**Links mentioned**:

- [GroqChat](https://groq.com/): no description found
- [Introduction | Mistral AI Large Language Models](https://docs.mistral.ai/): Mistral AI currently provides two types of access to Large Language Models:
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172): While recent language models have the ability to take long contexts as input, relatively little is known about how well they use longer context. We analyze the performance of language models on two ta...
- [Backus–Naur form - Wikipedia](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form): no description found
- [Let&#39;s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY): We build a Generatively Pretrained Transformer (GPT), following the paper &quot;Attention is All You Need&quot; and OpenAI&#39;s GPT-2 / GPT-3. We talk about connections t...
- [[Feature Request] Function Calling - Easily enforcing valid JSON schema following](https://community.openai.com/t/feature-request-function-calling-easily-enforcing-valid-json-schema-following/263515/14): There’s now a very weak version of this in place. The model can be forced to adhere to JSON syntax, but not to follow a specific schema, so it’s still fairly useless. We still have to validate the ret...
- [[Feature Request] Function Calling - Easily enforcing valid JSON schema following](https://community.openai.com/t/feature-request-function-calling-easily-enforcing-valid-json-schema-following/263515): Hi,  I was very excited to see the new function calling feature, then quickly disappointed to see that it doesn’t guarantee valid JSON. This was particularly surprising to me as I’ve personally implem...

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1203350592387416076) (45 messages🔥): 

- **Exploring AI Hosting Options**: User `@i_am_dom` suggested [Hugging Face](https://hf.co/) as a free and reliable hosting service for AI models; later, they also mentioned [Perplexity Labs](https://perplexity.ai/) as another hosting option. `@ashu2024` appreciated the information.

- **Best Models for CPU Inference Explored**: `@porti100` solicited advice on running smaller LLMs coupled with RAG on CPUs, `@mrdragonfox` recommended the 7b model but warned that it would be slow on CPUs. There was a brief discussion revolving around performance differences on lower-end systems and the efficiencies of various 7b quantized models.

- **Mistral's Superior Quantization Highlighted**: `@cognitivetech` shared their experience that [Mistral's quantization](https://github.com/cognitivetech/llm-book-summarization) outperformed other models, especially since version 0.2. They emphasized the need to test full models under ideal conditions for an accurate assessment.

- **Execution Language Impacts AI Performance**: `@cognitivetech` reported significant differences in performance when using Go and C++ instead of Python, while `@mrdragonfox` argued that since the underlying operations are in C++, the interfacing language shouldn't heavily impact the outcomes.

- **Getting Started with Mistral AI**: Newcomer `@xternon` inquired about using Mistral AI without laptop components powerful enough to run the models, leading to suggestions to use [Gradio](https://www.gradio.app/) for a demo web interface or [Hugging Face's hosted models](https://huggingface.co/chat) for an easy browser-based experience. `@adriata3` pointed out options for local CPU usage and recommended their GitHub repository with Mistral code samples, along with Kaggle as a potential free resource.

**Links mentioned**:

- [Gradio](https://www.gradio.app/): Build &amp; Share Delightful Machine Learning Apps
- [HuggingChat](https://huggingface.co/chat): Making the community's best AI chat models available to everyone.
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/): We provide client codes in both Python and Javascript.

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1203277393687613490) (17 messages🔥): 

- **Mistral mishap with markdown**: `@drprimeg1` struggled with **Mistral Instruct AWQ** not outputting content inside a JSON format when given a prompt with Markdown formatting. [Their current approach to classification can be found here](https://paste.ofcode.org/eRm2UGSzZfGvMqTNypdN2c), but the model responds with placeholders instead of actual content.

- **Markdown mayhem in models**: `@ethux` suggested that `@drprimeg1`'s problem could be due to the **Markdown formatting**, noting that the model tries to output JSON but ends up displaying markdown syntax instead.

- **GuardrailsAI to guide prompt effectiveness**: `@ethux` offered a solution by recommending **GuardrailsAI** as a tool for ensuring correct output formats and mentioned its capability to force outputs and retry upon failure. They also included a reference to the tool at [GuardrailsAI](https://www.guardrailsai.com/).

- **Teacher forcing talk**: `@ethux` mentioned that **GuardrailsAI** implements a form of teacher forcing by providing examples of what went wrong and how to correct it, while also being predefined.

- **Instructor Introduction**: As another recommendation for structured output generation, `@ethux` shared a link to **Instructor**, a tool powered by OpenAI's function calling API and Pydantic for data validation, described as simple and transparent. Additional insights and a community around the tool can be accessed at [Instructor's website](https://jxnl.github.io/instructor/).

**Links mentioned**:

- [Your Enterprise AI needs Guardrails | Your Enterprise AI needs Guardrails](https://www.guardrailsai.com/): Your enterprise AI needs Guardrails.
- [Welcome To Instructor - Instructor](https://jxnl.github.io/instructor/): no description found
- [Paste ofCode](https://paste.ofcode.org/eRm2UGSzZfGvMqTNypdN2c): no description found

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1203389040699777035) (17 messages🔥): 

- **Guidance on Fine-tuning for Energy Markets**: `@tny8395` inquired about training a model for automated energy market analysis and was informed by `@mrdragonfox` that it's possible to fine-tune for such a specific purpose.
- **Channel Clarification and Warning Against Spam**: `@mrdragonfox` guided `@tny8395` to keep the discussion on fine-tuning in the current channel and reminded them that spamming will not elicit additional responses.
- **Mistral and Fine-tuning API Development**: `@a2rette` asked if Mistral plans to work on fine-tuning APIs. `@mrdragonfox` responded, highlighting the current limitations due to the cost of inference and small team size, concluding that for now it is a "not yet."
- **Resource Realities at Mistral**: `@mrdragonfox` provided context on Mistral's operational scale, explaining that despite funding, the industry's high costs and a small team of around 20 people make certain developments challenging. 
- **Seeking Fine-tuning Info for Mistral with Together AI**: `@andysingal` inquired about resources for fine-tuning Mistral in combination with Together AI but did not receive a direct response.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1203356848753872928) (7 messages): 

- **ExLlamaV2 Featured on YouTube**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=N5lDUZRI8sc) titled "ExLlamaV2: The Fastest Library to Run LLMs," highlighting a fast inference library for running LLMs locally on GPUs. They also provided a GitHub link for the project and a Google Colab tutorial.
- **Novel Writing with AI Assistance**: `@caitlyntje` described their process of using AI to write a novel, involving generating an outline, chapter summaries, and then iterating over each chapter to ensure consistency, style, and detail. The process was carried out in sessions due to limitations in token handling on their MacBook.
- **Careful Monitoring During AI-Assisted Writing**: In a follow-up, `@caitlyntje` mentioned the necessity of careful oversight to maintain the logical flow and timeline when using AI for writing.
- **Model Capacity Recommendation**: Reacting to limitations mentioned by `@caitlyntje`, `@amagicalbook` recommended trying *Claude*, which allegedly can handle up to 200k token contexts.
- **Critic of AI-Generated Discworld Narrative**: `@swyxio`, a fan of Terry Pratchett, critiqued an AI-generated Discworld narrative for not capturing the essence of iconic characters like the witches, leading to a halt in reading.

**Links mentioned**:

[ExLlamaV2: The Fastest Library to Run LLMs](https://www.youtube.com/watch?v=N5lDUZRI8sc): A fast inference library for running LLMs locally on modern consumer-class GPUshttps://github.com/turboderp/exllamav2https://colab.research.google.com/github...

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1203827731201134622) (2 messages): 

- **YC Founder Seeks LLM Challenges**: User `@znrp`, a founder in Y Combinator, is reaching out for insights into the challenges community members face with **building in the space of LLMs**. They're open to direct messages for a quick chat.
- **Flags Fly High in Random**: User `@gafty` expressed their excitement or playfulness with a simple emoji message containing the **Romanian flag** and a crazy face.
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1203468704340377670) (4 messages): 

- **Stream Gate Closed**: `@jakobdylanc` experienced a problem where **mistral-medium** was not sending the final empty chunk in streamed responses, differentiating it from expected behavior observed in **mistral-small**.
- **Quick Fix, Not a Full Solution**: In response to the streaming issue, `@drones_flier` suggested discarding responses below a certain length as a temporary workaround, although they noted it might not be ideal for all use cases.
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1203295648007331840) (361 messages🔥🔥): 

- **The Frustration with Fooocus**: `@pseudoterminalx` expressed dissatisfaction working with the authors of controlnet and adapting their models to other platforms, describing difficulties in collaboration and a focus on promoting AUTOMATIC1111. Concerns about the willingness of these developers to consider community needs were discussed.
- **Concerns Over Controlnet Adoption and Stanford Researchers**: Several users like `@astropulse` and `@pseudoterminalx` shared their struggles with implementing ControlNet into their projects, noting a lack of information and support. Discussions arose around the ethics and actions of Stanford researchers related to LAION datasets, implying a business-first mentality and a lack of public development following their funding success.
- **Debate on Tech Giants and AI Model Training**: Users like `@pseudoterminalx`, `@thejonasbrothers`, and `@drhead` discussed the alignment of Stability AI's practices with NVIDIA's, touching on the strategy of following the tech giant's footsteps and questioning the independent innovative capacities of smaller entities.
- **Graphics Card Discussion for AI Models**: In a series of exchanges, users like `@ninyago` and `@vrus0188` discussed the adequacy of various NVIDIA graphics card models for running AI models, such as the 4060 ti and the 3090, with consideration for VRAM and budget.
- **Stability AI’s New Model Release Speculation**: Several users, including `@thejonasbrothers` and `@vrus0188`, conversed about Stability AI's upcoming new model, with `@thejonasbrothers` lamenting their own six-month project given the new model's capabilities and expressing disappointment in having to compete with substantial resources like that of Stability AI.

**Links mentioned**:

- [Model Memory Utility - a Hugging Face Space by hf-accelerate](https://huggingface.co/spaces/hf-accelerate/model-memory-usage): no description found
- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): no description found
- [Tweet from Angry Tom (@AngryTomtweets)](https://fxtwitter.com/AngryTomtweets/status/1753905168599462175): It&#39;s only been 1 day since Apple launched Vision Pro, and people are going crazy over it.   Here are 10 wild examples you don&#39;t want to miss:   1.  Apple Vision Pro Deepfake app concept   
- [Tweet from Emad (@EMostaque)](https://x.com/EMostaque/status/1751264392828653974?s=20): Trying one of the experimental @StabilityAI base models fresh from baking with some friends, feels like another stable diffusion moment tbh  What sorcery is this 🧙🏽‍♂️🪄✨  Hold on to your 🍑  ↘️ Quo...
- [PNY GeForce RTX 3090 24GB XLR8 Gaming REVEL EPIC-X RGB Triple Fan Edition #3  | eBay](https://www.ebay.de/itm/296205530951?hash=item44f739b747%3Ag%3A-SAAAOSwejNlv9ed&amdata=enc%3AAQAIAAAAwPuosBH1RVyMFwV2oqWRFeFtqoUbeNTquMPzgrcjK6fXWpO0U1%2F1kBogvxyue34J9hm%2Ba3q5hOJzxF3R53qZ7xvcmty4FW11KI9O1dgI7Yg19oqkUZzKDzitEtkoRG%2BaKmWuj3O5zoTjw83mBIMAN5Nal4ssU3VPmXEG57H6NCpRGffCX7agsUYiP62MnjiNlMdQjN%2Ff9QrSoA9oG5mQcOS5qRHF9VJN1lHf6YG7auZGUXRSiViiaOH8siM%2FsyvPWA%3D%3D%7Ctkp%3ABk9SR7KF9N-uYw&LH_All=1): no description found

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1203439431168827442) (3 messages): 

- **Emoji Speaks Louder Than Words**: User `@nodja` posted an interesting pair of 👀 without any accompanying text, leaving us all in suspense.
- **Qwen-VL-Max on Hugging Face Gains Attention**: `@nodja` shared a link to the [Hugging Face space for Qwen-VL-Max](https://huggingface.co/spaces/Qwen/Qwen-VL-Max), but it was noted to be a duplicate of [Qwen/Qwen-VL-Plus](https://huggingface.co/spaces/Qwen/Qwen-VL-Plus) with the same accompanying images of the model's avatar.
- **Clarification and Retraction**: Shortly after sharing the link, `@nodja` followed up with a simple "nevermind," indicating the previous message may have been posted in error.

**Links mentioned**:

[Qwen-VL-Max - a Hugging Face Space by Qwen](https://huggingface.co/spaces/Qwen/Qwen-VL-Max): no description found

  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1203333705632849980) (289 messages🔥🔥): 


<ul>
  <li><strong>Discussions about Falcon-180B and Demos Not Working:</strong> Users like <code>@nekoli.</code> reported issues with accessing <strong>HuggingFace</strong> demo pages, like Falcon-180B, indicating either site-wide issues or specific demo outages. Links to Falcon-180B and suggestions were shared, but success seemed varied.</li>
  <li><strong>Questions on LLM Deployment and Usage:</strong> <code>@rishit_kapoor</code> inquired about tutorials for deploying Mistral 7B via AWS Inferentia2 and SageMaker, while <code>@_sky_2002_</code> sought information on using LLMs via an API with free credits on HuggingFace.</li>
  <li><strong>Technical Assistance for Spaces:</strong> <code>@dongd.</code> sought help for a Space stuck in a building state, with <code>@not_lain</code> offering troubleshooting advice. The conversation touched on dependency issues and the functionality of 'factory rebuild'.</li>
  <li><strong>Hugging Face Infrastructure Issues:</strong> <code>@lolskt</code> and <code>@wubs_</code> highlighted possible infrastructure issues at Hugging Face, possibly affecting Gradio and other services, while users shared methods like hardware switching to resolve issues.</li>
  <li><strong>Impact of AI on Security:</strong> <code>@aifartist</code> reflected on the implications of deepfake technology in light of a news story about a deepfake CFO involved in a scam, expressing concern over the technology's potentially damaging uses.</li>
</ul>

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1019296127847239751/1203736452060684338): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Finance worker pays out $25 million after video call with deepfake ‘chief financial officer’ | CNN](https://www.cnn.com/2024/02/04/asia/deepfake-cfo-scam-hong-kong-intl-hnk/index.html): no description found
- [Non-engineers guide: Train a LLaMA 2 chatbot](https://huggingface.co/blog/Llama2-for-non-engineers): no description found
- [Falcon-180B Demo - a Hugging Face Space by lunarflu](https://huggingface.co/spaces/lunarflu/falcon-180b-demo-duplicate): no description found
- [Facepalm Really GIF - Facepalm Really Stressed - Discover &amp; Share GIFs](https://tenor.com/view/facepalm-really-stressed-mad-angry-gif-16109475): Click to view the GIF
- [Falcon-180B Demo - a Hugging Face Space by tiiuae](https://huggingface.co/spaces/tiiuae/falcon-180b-demo): no description found
- [unalignment/toxic-dpo-v0.1 · Datasets at Hugging Face](https://huggingface.co/datasets/unalignment/toxic-dpo-v0.1): no description found
- [Vizcom](https://www.vizcom.ai/): Vizcom is an Ai-powered creative tool designed for design and creative professionals. It offers a transformative approach to concept drawing, enabling users to turn their sketches into impressive real...
- [tiiuae/falcon-180B · Hugging Face](https://huggingface.co/tiiuae/falcon-180B): no description found
- [Superagi Sam - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/superagi-sam/): no description found
- [LLaMa Chat | Text Generation Machine Learning Model | Deep Infra](https://deepinfra.com/chat): Discover the LLaMa Chat demonstration that lets you chat with llama 70b, llama 13b, llama 7b, codellama 34b, airoboros 30b, mistral 7b, and more! 
- [Space won&#39;t start - logs not found](https://discuss.huggingface.co/t/space-wont-start-logs-not-found/54149/2): hi @155elkhorn could you please share more details? do you have a public Space link to share? thanks
- [Update bonus unit1 link by rgargente · Pull Request #485 · huggingface/deep-rl-class](https://github.com/huggingface/deep-rl-class/pull/485): Bonus unit 1 notebook is duplicated. The link to bonus unit 1 is pointing to the old version, without the updated download link for Huggy.zip. This PR updates the link and removes the old notebook ...
- [afrideva/phi-2-uncensored-GGUF · Hugging Face](https://huggingface.co/afrideva/phi-2-uncensored-GGUF): no description found
- [D5648R&#x20;DDR5-4800&#x20;64GB&#x20;ECC&#x20;Reg&#x20;Server&#x20;Memory&#x20;4800MHz&#x20;PC5-38400&#x20;CL40&#x20;1.1V&#x20;RDIMM&#x20;288-pin&#x20;Memory](https://www.centralcomputer.com/samsung-m321r8ga0bb0-cqk-64gb-ddr5-ecc-registered-4800mhz-pc5-38400-cl40-1-1v-rdimm-288-pin-memory.html?srsltid=AfmBOoqj9W5RhnchoAOI9d03r2E8ODlO4UBD0K1j4MJ8eymSRDEsn64V1uU): General&#x20;InformationProduct&#x20;TypeRAM&#x20;ModuleBrand&#x20;NameSamsungManufacturerSamsungProduct&#x20;Name64GB&#x20;DDR5&#x20;SDRAM&#x20;Memory&#x20;ModuleManufacturer&#x20;Part&#x20;NumberM32...

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1203257438590210079) (5 messages): 

- **The "Attention" Paper Blows Minds**: `@sardarkhan_` expressed amazement after reading the groundbreaking *"Attention Is All You Need"* paper, signaling a significant impact on their understanding of the subject.
- **Speech Recognition Evolves**: `@myke420247` successfully experimented with converting wav to text on his company's call recordings using **Whisper** and **pyannote** for speaker diarization, achieving better results than **Google's paid service** from 2018.
- **Tool Crafting for Audio Summaries**: `@n278jm` is in the process of creating an internal tool to summarize audio recordings from consultations, sharing code attempts and highlighting a commitment to privacy with local processing only.
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1203259884980801596) (11 messages🔥): 

- **Sentiment Analysis Deep Dive**: `@andysingal` highlighted a detailed tutorial on sentiment analysis combining Hugging Face and Deepgram. It showcases how to create charts to understand sentiment shifts over time and includes visuals of the sentiment analysis charts. [Sentiment Analysis with Hugging Face and Deepgram](https://deepgram.com/learn/sentiment-analysis-with-hugging-face-and-deepgram)

- **Blog Posting on Hugging Face Hub**: `@lunarflu` encouraged `@imcoza1915` to draft a community blog post on Hugging Face Hub to increase visibility of their work, linking to the [Hugging Face Blog Explorers](https://huggingface.co/blog-explorers) community.

- **Publication of Agent-Helper Langchain**: `@4gentbur3k` shared a link to a Hugging Face blog post discussing the integration of Hugging Face's transformers with Langchain for advanced NLP applications. The post demonstrates how combining these tools improves language understanding and generation. [Agent Helper Langchain Blog Post](https://huggingface.co/blog/Andyrasika/agent-helper-langchain-hf)

- **Art Forge Labs AI Art Generation**: `@wubs_` expressed amazement at a post from Art Forge Labs detailing significant enhancements in both speed and quality for AI-driven art generation, but did not provide an accessible URL for the content mentioned.

- **Fine-tuning Models Resource List**: `@andysingal` shared a progress update on creating a resource list for fine-tuning models, offering a link to the GitHub repository where the list is being compiled. Access the fine-tuning list at the [llm-course GitHub repository](https://github.com/andysingal/llm-course/blob/main/llama_finetune/README.md).

**Links mentioned**:

- [Uniting Forces: Integrating Hugging Face with Langchain for Enhanced Natural Language Processing](https://huggingface.co/blog/Andyrasika/agent-helper-langchain-hf): no description found
- [blog-explorers (Blog-explorers)](https://huggingface.co/blog-explorers): no description found
- [llm-course/llama_finetune/README.md at main · andysingal/llm-course](https://github.com/andysingal/llm-course/blob/main/llama_finetune/README.md): Contribute to andysingal/llm-course development by creating an account on GitHub.
- [Sentiment Analysis with Hugging Face and Deepgram | Deepgram](https://deepgram.com/learn/sentiment-analysis-with-hugging-face-and-deepgram): Sentiment analysis charts provide a visually informative way to track and understand shifts in sentiment over time. These data visualizations can offer insig...
- [no title found](https://www.artforgelabs.com/post/art-forge-labs-unveils-revolutionary-speed-and-quality-enhancements-in-ai-art-generation): no description found

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1203280012309041173) (9 messages🔥): 

- **Ethical LLM Framework Proposition**: `@lunarflu` suggested the idea for a **general ethical framework for language model bots** that could be accepted globally, triggering a thought-provoking discussion.
- **CrewAI Gets an AutoCrew**: `@_yannie_` shared their GitHub project [Autocrew](https://github.com/yanniedog/autocrew), a tool to automatically create a crew and tasks for CrewAI, complete with an inviting repository image and description.
- **Newcomer Ready to Contribute**: User `@__codenerd__` introduced themselves to the chat, expressing eagerness to showcase their best work in the community.
- **Hacker's Digital Assistant Unveiled**: `@n278jm` introduced a hacker-oriented chat assistant on HuggingFace - intriguingly titled *Your personal hacker helper*, aiming to assist with analyzing hacking needs and tool outputs.
- **Emoji Predictions from Tweets**: `@pendrokar` adapted `@748130998935617676`'s TorchMoji to craft a HuggingFace space that predicts possible follow-up emojis based on English text, using a dataset of 1 billion tweets from 2017, available at [DeepMoji on HuggingFace](https://huggingface.co/spaces/Pendrokar/DeepMoji).
- **Hercules-v2.0 Dataset Launch**: `@locutusque` announced the release of **Hercules-v2.0**, a comprehensive dataset to power specialized domain models, and shared the dataset's performance on the [Open LLM Leaderboard](https://huggingface.co/Locutusque/Hercules-2.0-Mistral-7B), including a warning about sensitive content.
- **Artforge Labs' AI Image Generator**: `@wubs_` unveiled Artforge Labs, an AI image generation service that offers unlimited image creation with a risk-free trial and monthly subscription. It aspires to rival MidJourney and is based on SDXL Turbo models and can be explored at [artforgelabs.com](https://artforgelabs.com).

**Links mentioned**:

- [DeepMoji - a Hugging Face Space by Pendrokar](https://huggingface.co/spaces/Pendrokar/DeepMoji): no description found
- [Locutusque/hercules-v2.0 · Datasets at Hugging Face](https://huggingface.co/datasets/Locutusque/hercules-v2.0): no description found
- [Locutusque/Hercules-2.0-Mistral-7B · Hugging Face](https://huggingface.co/Locutusque/Hercules-2.0-Mistral-7B): no description found
- [GitHub - yanniedog/Autocrew: Automatically create a crew and tasks for CrewAI](https://github.com/yanniedog/autocrew): Automatically create a crew and tasks for CrewAI. Contribute to yanniedog/Autocrew development by creating an account on GitHub.
- [Penne Tester - HuggingChat](https://hf.co/chat/assistant/65bee444cf81c30b367f2dd7): Use the Penne Tester assistant inside of HuggingChat
- [HermeticCoder - HuggingChat](https://hf.co/chat/assistant/65bf5960a77e83076fb013ec): Use the HermeticCoder assistant inside of HuggingChat
- [Home | Art Forge Labs](https://artforgelabs.com): Generate, Refine, and Paint. Powered by Art Forge Lab's AI image generator. Art Forge Labs

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1203284774526132234) (9 messages🔥): 

- **Organizing HuggingFace Events**: `@lunarflu` created a placeholder for an upcoming event (time TBD) and shared the link [Join the Event](https://discord.gg/huggingface?event=1203285706949009448). A new channel was also created specifically for asking questions in a structured manner.
- **Recording Sharing Protocol Established**: `@chad_in_the_house` sent the Google Drive link of the recorded session to `@811235357663297546`, indicating plans for it to be posted in the channel and potentially on YouTube.
- **Drive Link to Session Recording**: `@lunarflu` shares the [Google Drive recording](https://drive.google.com/file/d/1R6hQnEISYT8eGSDwHO-Hwi57rdtT4Q_-/view?usp=sharing) of the previous session with the group. It could form the basis of a potential YouTube channel for wider sharing.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/huggingface?event=1203285706949009448): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [trimmed_pres_v2.mkv](https://drive.google.com/file/d/1R6hQnEISYT8eGSDwHO-Hwi57rdtT4Q_-/view?usp=sharing): no description found

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1203481959368757248) (12 messages🔥): 

- **Slow Down, Speedy!**: HuggingMod reminded `@548866140697264129` to ease on their message frequency: a gentle nudge for some **serenity in the chat**.
- **Scam Alert Raised**: `@meatfucker` flagged a classic Discord scam circulating and suggested its removal, alerting the **HuggingFace moderation team**.
- **Technical Troubles with CogVLM**: User `@furkangozukara` sought assistance for an issue detailed on GitHub, specifically with AutoModelForCausalLM and a Half-Char dtype error. The posted [GitHub issue](https://github.com/huggingface/transformers/issues/28856) offers more insight into their predicament.
- **Navigating License Agreements for AI Models**: `@pseudoterminalx` shared a link to the **Stable Video Diffusion** model license agreement and asked if Diffusers weights could be accessed, highlighting *license compliance* in the field of AI research.
- **Epochs and Training Dilemmas**: `@bitpattern` questioned the long training times indicated in their script logs, and `@pseudoterminalx` advised to reduce the number of epochs, hinting at potential **overfitting or inefficiency** in the training process.

**Links mentioned**:

- [stabilityai/stable-video-diffusion-img2vid-xt-1-1 at main](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/tree/main): no description found
- [When using AutoModelForCausalLM, THUDM/cogagent-vqa-hf and load_in_8bit I get this error : self and mat2 must have the same dtype, but got Half and Char · Issue #28856 · huggingface/transformers](https://github.com/huggingface/transformers/issues/28856): System Info Microsoft Windows [Version 10.0.19045.3996] (c) Microsoft Corporation. All rights reserved. G:\temp Local install\CogVLM\venv\Scripts&gt;activate (venv) G:\temp Local install\CogVLM\venv\S...

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1203334038706851890) (7 messages): 

- **Exploring Synthdog for Fake Data Generation**: `@swetha98` is looking for guidance on using **Synthdog in the Donut model** for creating fake document images but can't find scripts or the images needed for the process.
- **In Search of the Latest One-shot Models**: `@ursium` inquires about advanced alternatives to **CIDAS/Clipseg_Rd64_refined** for less artifacty zero-shot vision models, noting the one-year-old model might be outdated.
- **Rapid Messaging Alert**: `@HuggingMod` gently reminds a user to slow down after rapid message posting on the channel.
- **Sliding Puzzle Dataset for Vision LLM**: `@harsh_xx_tec_87517` announces the publication of a sliding puzzle dataset designed for training vision LLMs and shares the [dataset on Hugging Face](https://huggingface.co/datasets/Harshnigm/puzzles-for-vision-llm) and the [source code on GitHub](https://github.com/Harshnigam6/puzzle_llm_dataset_generation) for generating such datasets, seeking feedback from the community.
- **Models Struggle with Puzzle Dataset**: In response to `@gugaime`'s query, `@harsh_xx_tec_87517` mentions they have only implemented the dataset generator so far, with models like ChatGPT-4 and LLaMA failing to solve the puzzle, prompting further work on fine-tuning LLaMA.


**Links mentioned**:

- [Harshnigm/puzzles-for-vision-llm · Datasets at Hugging Face](https://huggingface.co/datasets/Harshnigm/puzzles-for-vision-llm): no description found
- [GitHub - Harshnigam6/puzzle_llm_dataset_generation: In this repo, we implement a method to generate synthetic dataset to train a vision LLM to learn how to reconstruct a puzzle.](https://github.com/Harshnigam6/puzzle_llm_dataset_generation): In this repo, we implement a method to generate synthetic dataset to train a vision LLM to learn how to reconstruct a puzzle. - GitHub - Harshnigam6/puzzle_llm_dataset_generation: In this repo, we ...

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1203309242476658708) (9 messages🔥): 

- **Seeking Knowledge for Video Summarization**: `@karam15.` is looking for research papers, models, or GitHub repositories related to **summarization of videos with timestamps**. Suggestions and references to pertinent resources are requested.
- **Mistral 7B Deployment on AWS Inquiry**: `@rishit_kapoor` is seeking tutorials or materials on deploying **Mistral 7B using AWS Inferentia2 and SageMaker**. This query has been posted twice indicating a strong interest in the topic.
- **Exploring Models for Spell Check and Grammar**: `@.bexboy` is in search of **finetune-able models** suitable for spell checking and grammar improvement. Guidance on effective models or tools is solicited.
- **Call for Collaborators in Model Merging**: `@birger6875` invites the community to join experiments with **model merging**, particularly with a focus on Nordic languages. They provided a [planning document](https://docs.google.com/document/d/1fP2FIrCifWcLGdTBmqeogdCdZJOwxqPfEyO-HA76_qc/edit?usp=sharing), a [tutorial](https://huggingface.co/blog/mlabonne/merge-models), a [Colab notebook](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing), and a mention of a Discord channel specifically for model merging discussions.
- **In Search of Contribution Opportunities**: `@Nicks🤙🏾` expresses an interest in contributing to a project and is considering the *first steps* to take on this journey.

**Links mentioned**:

[merge-crew (Merge Crew)](https://huggingface.co/merge-crew): no description found

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1203481959368757248) (12 messages🔥): 

- **Slow Down, Quick Poster**: HuggingMod warned `@548866140697264129` to slow down due to posting too quickly in the channel.
- **Watch Out for Scams**: `@meatfucker` alerted the moderators (denoted by <@&897381378172264449>) about a classic Discord scam message and suggested its removal.
- **Troubleshooting GitHub Issue**: `@furkangozukara` sought assistance with a problem posted on GitHub related to `AutoModelForCausalLM`, citing an error regarding data types (Half and Char). The user linked to the issue [here](https://github.com/huggingface/transformers/issues/28856).
- **Seeking Model Weights for Stable Video Diffusion**: `@pseudoterminalx` is looking for diffusers weights and shared a link to the Stable Video Diffusion model which requires acceptance of a license agreement to access.
- **Optimizing Training Time**: `@bitpattern` and `@pseudoterminalx` discussed a concern about long training times with a large number of epochs and `@pseudoterminalx` recommended reducing the number of epochs or adjusting command-line arguments to optimize the training schedule.

**Links mentioned**:

- [stabilityai/stable-video-diffusion-img2vid-xt-1-1 at main](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/tree/main): no description found
- [When using AutoModelForCausalLM, THUDM/cogagent-vqa-hf and load_in_8bit I get this error : self and mat2 must have the same dtype, but got Half and Char · Issue #28856 · huggingface/transformers](https://github.com/huggingface/transformers/issues/28856): System Info Microsoft Windows [Version 10.0.19045.3996] (c) Microsoft Corporation. All rights reserved. G:\temp Local install\CogVLM\venv\Scripts&gt;activate (venv) G:\temp Local install\CogVLM\venv\S...

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1203293331443028009) (77 messages🔥🔥): 

- **Exploring Local LLMs for Improved Performance**: `@mistermatty` discussed the drawbacks of GPT-4, including "Conversation key not found" errors and subpar performance. They expressed interest in local LLMs as an alternative and received suggestions like LM Studio and perplexity labs for free LLM use from `@7877`.
  
- **Local LLMs Versus GPT-4 Benchmarking**: `@kotykd` opined that no open-source LLM compares to GPT-4, even though models like Mixtral 8x7b, which require significant RAM, underperform GPT-3.5 in most areas.

- **Performance Spotlight on codellama-70b-instruct**: `@mistermatty` highlighted their positive experience with codellama-70b-instruct, hosted on the perplexity labs playground. Their interaction prompted them to consider a setup for local usage of comparable LLMs.

- **Hardware Conundrum for Running LLMs**: Several participants, including `@mistermatty`, `@kotykd`, `@johnnyslanteyes`, and `@michael_6138_97508`, engaged in a detailed discussion on the hardware requirements for running large LLMs locally, touching on the significance of RAM vs. VRAM, system recommendations, and possible heat issues with laptops.

- **Credibility of AI & LLM Information**: The dialogue from `@johnnyslanteyes`, `@michael_6138_97508`, and `@aipythonista` indicated skepticism towards the reliability of information regarding AI performance on various hardware, highlighting the importance of firsthand experience and critical evaluation of sources like YouTube.

**Links mentioned**:

[Beyond Consciousness in Large Language Models: An Investigation into the Existence of a “Soul” in…](https://takk8is.medium.com/beyond-consciousness-in-large-language-models-an-investigation-into-the-existence-of-a-soul-in-83d32002c3a0): Author: David Côrtes Cavalcante Publication Date: February 3, 2024 © CC BY Creative Commons Attribution

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1203350765171908608) (59 messages🔥🔥): 

- **Alert on @mention Issues**: `_odaenathus` reported problems using the `@` system with their custom GPTs, observing that even GPTs that used to work together were not cooperating anymore, and this issue was inconsistent.
- **GPT Amnesia or Just Broken?**: `blckreaper` mentioned difficulties with their GPTs, such as forgetting files and abruptly ending stories, while being frustrated about wasting messages trying to debug the issues.
- **The Thumbs Up Mystery**: It was noted that the thumbs up feature is missing, and `johnnyslanteyes` clarified that it appears only when a message has to be regenerated, to inform the system if a new response is better for curation.
- **Trouble with Prompts and Logging Out**: `rahulhere` expressed difficulties with a GPT that wouldn't log out after OAuth authentication, and queried why "Starting Action" takes a long time.
- **Search and Ranking for GPTs Needed**: Users like `astron8272` and `killlemon.eth` are seeking ways to rank GPTs for efficiency in specific tasks like language learning and are inquiring about GPT agents for marketing research, as well as easier searching functionalities for GPT Agents.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1203283846720917545) (51 messages🔥): 

- **Custom GPT Instructions Concern**: `@gameboy2936` seeks advice on setting custom GPT instructions to ensure the bot communicates with humanlike nuances without reverting to AI writing style, such as using overly ornate language. Their bot should avoid AI-style phrases and maintain a consistent human speech pattern.
  
- **Stealth2077 Encounters Over-Moderation**: `@stealth2077` complains about GPT-4's inclination to incorporate ethical considerations into responses, even when it seems inappropriate for the user's storytelling purposes. Efforts to instruct the AI to omit such themes have not resolved the issue.
  
- **Need for Assistant Model Stability Highlighted by Madame_Architect**: `@madame_architect` recommends using the Assistant Model on a stable GPT version rather than the unpredictable Preview Model to maintain instruction consistency. She indicates that over-moderation is impacting output quality.
  
- **Prompt Specificity Tactics to Avoid Misspelled Names**: Users `@titaniumsporks` and `@snovov` discuss issues with ChatGPT misspelling names leading to derailment during conversations, with suggestions for being specific and using appropriate platforms.
  
- **Users Report Sudden Content Policy Violation Messages**: `@papa_jhon.` expresses frustration over receiving unexpected policy violation messages for seemingly innocent prompts, joined by `@lugui` who suggests it might be an internal problem that could resolve later.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1203283846720917545) (51 messages🔥): 

- **Exploring Human Speech for Autistic User**: User `@gameboy2936` seeks assistance with customizing GPT's communication style to create more human-like interactions, sharing detailed instruction examples. 

- **Assistant vs Custom GPT for Stability**: `@madame_architect` recommends switching from Custom GPT to Assistant models due to the instability of preview models and the frustration of prompt engineering on a "moving target."

- **Creative Writing Struggles with Over-Moderation**: Users `@stealth2077` and `@madame_architect` discuss challenges in maintaining creative control over writing as ChatGPT attempts to adhere to ethical considerations, which `@stealth2077` finds restrictive for their specific use case.

- **Editing Tips for Consistency in Storytelling**: `@johnnyslanteyes` offers a tip to `@stealth2077` on highlighting specific story sections to guide ChatGPT in editing, while also tackling the issue of forced values in narrative content.

- **Issues with Policy Violation Responses and AIs**: Users `@papa_jhon.` and `@lugui` converse about unexpected policy violation responses to innocent prompts, suggesting that it may be an internal problem that could be resolved later.
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1203256672643194911) (119 messages🔥🔥): 

- **Troubleshooting for "cuda argument error"**: `@dangfutures` advised running ```sudo apt-get install libopenmpi-dev pip install mpi4py``` to address issues with GPUs on RunPod.
- **Memory and Strategy for Finetuning**: Concerning memory requirements, `@nruaif` stated 80gb VRAM would be needed for LoRA or QLoRA on Llama70. They also noted freezing MoE layers can allow 8 A6000 GPUs to manage Mixtral FFT, and finetuning using LoRA is half the speed but more sample efficient.
- **RunPod GPU Leasing Explorations**: `@yamashi` considered investing in a machine to rent on RunPod when not in use, while `@casper_ai` pointed out a large number of GPUs are needed to rent to RunPod. `@le_mess` suggested using vast.ai for single machines.
- **Component Preference**: `@yamashi` concluded LoRA is preferred over QLoRA, and `@nanobitz` mentioned choosing to target the router and attention for layer updates when finetuning with LoRA.
- **Emerging Quantization and Model Training Methods**: `@dangfutures` and `@casper_ai` discussed the potential of training with AWQ and Marlin Quant, acknowledging it's possible with recent advancements, and `@casper_ai` plans to benchmark the speeds of various quantization methods.

**Links mentioned**:

- [AMD + 🤗: Large Language Models Out-of-the-Box Acceleration with AMD GPU](https://huggingface.co/blog/huggingface-and-optimum-amd): no description found
- [Kind request for updating MT-Bench leaderboards with Qwen1.5-Chat series · Issue #3009 · lm-sys/FastChat](https://github.com/lm-sys/FastChat/issues/3009): Hi LM-Sys team, we would like to present the generation results and self-report scores of Qwen1.5-7B-Chat, Qwen1.5-14B-Chat, and Qwen1.5-72B-Chat on MT-Bench. Could you kindly help us verify them a...
- [twitter.co - Domain Name For Sale | Dan.com](https://twitter.co): I found a great domain name for sale on Dan.com. Check it out!

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1203397418901577768) (88 messages🔥🔥): 

- **Scaling to New Performance Heights**: `@casper_ai` and others discussed a new 2B parameter model by OpenBMB potentially matching the performance of Mistral 7B, highlighting both skepticism and astonishment at the claimed benchmarks. The detailed exploration, shared in a [Notion page](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20), emphasizes the importance of optimizing model training.

- **Implementation of Advanced Algorithms**: `@yamashi` praised the potential of implementing their word sense disambiguation (WSD) algorithm, noting that it seems straightforward and possibly more effective than current methods.

- **Mac Compatibility Queries for Axolotl**: Discussions by `@yamashi` about running Axolotl on the new M3 Mac, encountering issues like the model defaulting to CPU instead of GPU, waiting on torch and transformers for half-precision support on Mac, and submitting a pull request to help others interested in running on Mac.

- **Fine-Tuning Techniques Debated**: `@casper_ai` and `@c.gato` engaged in a detailed conversation about applying training strategies of large models, specifically supervised finetuning (SFT) and different training phases mentioned in the MiniCPM discovery, noting that these methods might not directly apply to finetuning.

- **Maximizing Data Utilization Explored**: `@dreamgen` and others expressed interest in the extensive data utilized in the new training strategy from OpenBMB, particularly the implications for large models and the potential need for similar experimental setups for fine-tuning approaches.

**Links mentioned**:

- [GitHub - nektos/act: Run your GitHub Actions locally 🚀](https://github.com/nektos/act): Run your GitHub Actions locally 🚀. Contribute to nektos/act development by creating an account on GitHub.
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team

  

---


### OpenAccess AI Collective (axolotl) ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/) (1 messages): 

cf0913: https://huggingface.co/chatdb/natural-sql-7b
  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1203590582463758386) (18 messages🔥): 

- **Batch Size Calculations Conundrum**: `@duke001.` sought advice on how to determine the number of steps for each epoch during training and was puzzled by the discrepancy between theoretical calculation and actual observations on wandb. `@nanobitz` suggested looking into sequence length packing and also mentioned the `save_per_epoch` option to help with proportionate checkpoint saving.

- **EOS Token Identity Crisis in Finetuning**: `@cf0913` experienced an issue where, after finetuning, the EOS token appeared to act as the pad token for deepseek-coder-instruct, leading to manual adjustments. `@nanobitz` proposed editing the tokenizer config to swap the tokens, which `@cf0913` confirmed worked without issues.

- **Support Request for Axolotl on MacBook Air M1**: `@mini_09075` faced errors trying to install the axolotl package, due to the lack of CUDA support on an M1 Apple chip. `@yamashi` responded with a mention of their barebones PR that might substitute MPS for CUDA, but warned it's not recommended for use as it stands.

- **Obsolete Branch Baffles User**: In a quest to perform Medusa training on a local machine, `@mini_09075` used an outdated branch but soon realized that it might not be possible, which was implied by `@yamashi` asking why the outdated branch was in use.

**Links mentioned**:

[GitHub - ctlllll/axolotl: Go ahead and axolotl questions](https://github.com/ctlllll/axolotl.git): Go ahead and axolotl questions. Contribute to ctlllll/axolotl development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1203354689488552057) (7 messages): 

- **Troubles with DPO**: `@fred_fups` reported difficulties getting **differential privacy optimization (DPO)** to work, facing out-of-memory (OOM) issues particularly when using it with **qlora** compared to regular **Mistral qlora**.
- **DPO's High Memory Demand**: `@noobmaster29` confirmed that DPO indeed consumes significantly more memory, noting that 24GB RAM only supports a **microbatch size of 1** with a **context size of 2048**.
- **Recommendation for Alternative**: In response to issues with DPO, `@dreamgen` suggested trying out **unsloth**, as sample packing does not work with DPO.
- **Inquiry about Various Optimization Techniques**: `@dreamgen` inquired if anyone has experience with other optimization methods, naming **KTO, IPO**, and others for potential experiments.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1203710549947457586) (5 messages): 

- **Initialization Issues with OpenBSD SSH on Runpod**: `@nruaif` shared a log snippet indicating the successful start of the OpenBSD Secure Shell server but encountered a deprecation warning regarding a `_jupyter_server_extension_points` function missing in `notebook_shim`.
- **Deprecated Config Warning**: The logs provided by `@nruaif` also included a FutureWarning regarding the deprecation of `ServerApp.preferred_dir` config in `jupyter-server 2.0`, advising to use `FileContentsManager.preferred_dir` instead.
- **Runpod Docker Configuration Error**: The same log mentions a critical error where `/workspace` is found to be outside of the root contents directory, causing a bad config encounter during initialization.
- **Intermittent Secure Cloud Issues**: `@dreamgen` noted that similar issues happen often in the secure cloud environment and suggested that it might not always be related to the use of network volumes.
- **Frustration with Runpod Issues**: `@dangfutures` expressed dissatisfaction with Runpod, suggesting alternatives like community versions tend to yield better results.
  

---



### CUDA MODE (Mark Saroufim) ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1203449210842775623) (25 messages🔥): 

- **CUDA vs OpenCL Discussion**: User `@Voudrais` questioned the preference for CUDA over OpenCL. `@andreaskoepf` responded that CUDA's advantages include popularity and strong support from Nvidia, welcoming everyone to the group regardless of their parallel programming preferences.
- **Python Over CUDA or OpenCL?**: Python as a language for GPU computing sparked a conversation led by `@vim410`. `@andreaskoepf` shared the resource list from the [CUDA MODE GitHub repository](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#python-gpu-computing), acknowledging the push towards high-level programming while noting the continued relevance of direct kernel writing.
- **Lecture Repository Renaming and Organization**: There was a reorganization of CUDA MODE lecture content by `@andreaskoepf`, merging lectures into one repository now named "lectures" on GitHub. Discussions with `@jeremyhoward` involved considerations for redirecting from old links and updating video descriptions to accommodate the new repository structure.
- **Machine Learning Visualization Shared**: `@latentzoo` shared a visualization related to tiny-cuda-nn's fully fused MLP, with a [tweet link](https://x.com/mallocmyheart/status/1753512787891139008?s=20) associated. `@andreaskoepf` suggested the image might be from a YouTube video on Tensor Cores, further adding only related video content should be shared.
- **Upgrading to a New Dev Machine**: `@andreaskoepf` initiated a discussion about upgrading to a new development machine, considering slow-building part by part due to the high cost of pre-built machines like Lambda workstations. The post opened up the possibility of community interest in system building.

**Links mentioned**:

- [GitHub - cuda-mode/resource-stream: CUDA related news and material links](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#python-gpu-computing): CUDA related news and material links. Contribute to cuda-mode/resource-stream development by creating an account on GitHub.
- [GitHub - cuda-mode/lecture2: lecture 2 - 2024-01-20](https://github.com/cuda-mode/lecture2): lecture 2 - 2024-01-20. Contribute to cuda-mode/lecture2 development by creating an account on GitHub.
- [GitHub - cuda-mode/profiling-cuda-in-torch](https://github.com/cuda-mode/profiling-cuda-in-torch): Contribute to cuda-mode/profiling-cuda-in-torch development by creating an account on GitHub.
- [Tweet from Hayden (@mallocmyheart)](https://x.com/mallocmyheart/status/1753512787891139008?s=20): just grokked what&#39;s actually happening here and how it relates to tiny-cuda-nn&#39;s fully fused mlp
- [Customize Your Lambda Vector | Lambda](https://shop.lambdalabs.com/gpu-workstations/vector/customize): no description found
- [GitHub - cuda-mode/lecture2: lecture 2 - 2024-01-20](https://github.com/cuda-mode/lecture2/?tab=readme-ov-file): lecture 2 - 2024-01-20. Contribute to cuda-mode/lecture2 development by creating an account on GitHub.
- [GitHub - cuda-mode/lectures: Material for cuda-mode lectures](https://github.com/cuda-mode/lectures): Material for cuda-mode lectures. Contribute to cuda-mode/lectures development by creating an account on GitHub.
- [Tensor Cores in a Nutshell](https://youtu.be/yyR0ZoCeBO8?si=_PTd7rVNgjokgQC9&t=20): This video gives a brief introduction to the Tensor Core technology inside NVIDIA GPUs and how its important to maximizing deep learning performance. https:/...

  

---


### CUDA MODE (Mark Saroufim) ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1203259271224234004) (99 messages🔥🔥): 

- **Discoveries in Grayscale Conversion Speed and Accuracy**: `@artste` experimented with various approaches for converting RGB to grayscale, finding that integer math proved fast yet imprecise. An optimal balance between speed and precision is achieved with a float lookup table, reaching near-identical results to the benchmark about 2.8 times faster ([Notebook detailing the experiments](https://github.com/artste/lecture2/blob/cuda_rgb_to_gray_refactor_notebook/lecture3/cuda_rgb_to_gray_refactor.ipynb)).

- **Batch Processing Enhances Performance**: When stacking images horizontally to imitate a batch, `@artste` found that the performed optimizations resulted in a grayscale conversion process that is up to 3.98 times faster than the unoptimized case for a batch size of 16 images.

- **Compiler Intricacies Meaningfully Impact CUDA Performance**: Discussions by `@andreaskoepf` and others reveal that seemingly minor changes, like adding 'f' to denote 32-bit floats, can greatly affect the operation time on GPUs, emphasizing the intricacies of GPU optimizations.

- **CUDA Tools and Repositories Shared Among Members**: Members shared various resources and tools such as Godbolt, a CUDA web compiler/explorer, and multiple GitHub repositories—including [`tiny-cuda-nn`](https://github.com/NVlabs/tiny-cuda-nn) for a fast neural network framework, and lectures on CUDA from [`cuda-mode/lectures`](https://github.com/cuda-mode/lectures)—which facilitate CUDA learning and experimentation.

- **Learning and Debugging CUDA with PyTorch**: `@edd0302` sought advice on managing CUDA and PyTorch projects, while `@jeremyhoward` and others discussed the idiosyncrasies of compiling CUDA code with PyTorch, highlighting challenges such as forced recompilation and potential improvements to be considered by PyTorch development (`@marksaroufim` indicated an openness to feedback for improvement).

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1189498204333543425/1189607726595194971/1202057241868640308): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Analysis of a Tensor Core](https://youtu.be/xjjN9q2ym6s): A video analyzing the architectural makeup of an Nvidia Volta Tensor Core.References:Pu, J., et. al. &quot;FPMax: a 106GFLOPS/W at 217GFLOPS/mm2 Single-Precision ...
- [Tweet from Hayden (@mallocmyheart)](https://x.com/mallocmyheart/status/1753512787891139008): just grokked what&#39;s actually happening here and how it relates to tiny-cuda-nn&#39;s fully fused mlp
- [lecture2/lecture3/cuda_rgb_to_gray_refactor.ipynb at cuda_rgb_to_gray_refactor_notebook · artste/lecture2](https://github.com/artste/lecture2/blob/cuda_rgb_to_gray_refactor_notebook/lecture3/cuda_rgb_to_gray_refactor.ipynb): lecture 2 - 2024-01-20. Contribute to artste/lecture2 development by creating an account on GitHub.
- [flash-attention/flash_attn at main · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn): Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.
- [GitHub - cuda-mode/lecture2: lecture 2 - 2024-01-20](https://github.com/cuda-mode/lecture2/): lecture 2 - 2024-01-20. Contribute to cuda-mode/lecture2 development by creating an account on GitHub.
- [GitHub - NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework](https://github.com/NVlabs/tiny-cuda-nn): Lightning fast C++/CUDA neural network framework. Contribute to NVlabs/tiny-cuda-nn development by creating an account on GitHub.
- [torch.utils.cpp_extension &mdash; PyTorch 2.2 documentation](https://pytorch.org/docs/stable/cpp_extension.html): no description found
- [GitHub - ROCm/flash-attention: Fast and memory-efficient exact attention](https://github.com/ROCm/flash-attention): Fast and memory-efficient exact attention. Contribute to ROCm/flash-attention development by creating an account on GitHub.
- [Compiler Explorer - CUDA C++ (NVCC 12.3.1)](https://godbolt.org/z/odb3191vK): #include &amp;lt;stdint.h&amp;gt;  // Type your code here, or load an example. __global__ void square(uint32_t* v, size_t vn, uint32_t* r) {     auto tid = blockDim.x * blockIdx.x + threadIdx.x;     i...
- [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://arxiv.org/abs/2201.05989): Neural graphics primitives, parameterized by fully connected neural networks, can be costly to train and evaluate. We reduce this cost with a versatile new input encoding that permits the use of a sma...
- [Compiler Explorer](https://godbolt.org/): no description found
- [GitHub - cuda-mode/lectures: Material for cuda-mode lectures](https://github.com/cuda-mode/lectures/): Material for cuda-mode lectures. Contribute to cuda-mode/lectures development by creating an account on GitHub.

  

---


### CUDA MODE (Mark Saroufim) ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1203834178269880361) (5 messages): 

- **Fast & Furious PyTorch Code Tip**: `@tantara` shared a [link to a PyTorch code section](https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L307-L314) from **gpt-fast** repo, suggesting that it might be helpful to specify the compiled layers when using the `torch.compile` API.
- **Torch Compiler Fine-Grain Control Unveiled**: `@marksaroufim` provided additional insights, mentioning the use of `torch.compiler.disable()` and recommended the PyTorch [documentation on finer grained APIs](https://pytorch.org/docs/main/torch.compiler_fine_grain_apis.html) to control `torch.compile`.

- **TensorFlow: The Alternative Debate**: `@Voudrais` humorously suggested using TensorFlow instead of PyTorch, which led `@andreaskoepf` to create a dedicated channel for TensorFlow discussions. He acknowledged the benefits of Google's accelerator resources and competitive pricing, but cautioned against platform lock-in issues.

**Links mentioned**:

- [gpt-fast/generate.py at main · pytorch-labs/gpt-fast](https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L307-L314): Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - pytorch-labs/gpt-fast
- [TorchDynamo APIs for fine-grained tracing &mdash; PyTorch main documentation](https://pytorch.org/docs/main/torch.compiler_fine_grain_apis.html): no description found

  

---


### CUDA MODE (Mark Saroufim) ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1203420809482338415) (2 messages): 

- **New CUDA Lecture Incoming**: `@andreaskoepf` announced **CUDA MODE - Lecture 4: Intro to Compute and Memory Architecture** will start soon, focusing on Ch 4 & 5 of the PMPP book, covering blocks, warps, and memory hierarchy.

- **Lecture Notes Available**: `@tvi_` mentioned that the notes for the upcoming lecture which includes Ch 4 & 5 discussions can be found in the repository, humorously referred to as the "increasingly inaccurately named lecture2 repo".
  

---


### CUDA MODE (Mark Saroufim) ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1203267016748826624) (2 messages): 

- **Aleph Alpha is in Search of CUDA Stallions**: `@piotr.mazurek` shared a job posting where **Aleph Alpha** is looking to hire savvy professionals for their product team. Specifically, the role involves translating research on language models into practical applications, impacting Fortune 2000 companies and governments, and can be found [here](https://alephalpha.jobs.personio.de/job/1329474?language=en&display=en).

- **Mistral AI Wants YOU for GPU Magic**: `@megaserg.` highlighted an opportunity at **Mistral AI**, seeking experts in **serving and training large language models on GPUs**. The job entails writing custom CUDA kernels and maximizing the potential of high-end GPUs like the H100, within a role posted [here](https://jobs.lever.co/mistral/399978d0-b442-4591-b677-8cc03ee24a48).

**Links mentioned**:

- [AI Engineer - Large Language Models (m/f/d) | Jobs at Aleph Alpha GmbH](https://alephalpha.jobs.personio.de/job/1329474?language=en&display=en): Aleph Alpha was founded in 2019 with the mission to research and build the foundational technology for an era of strong AI. The team of international scientists, engineers, and innovators researches, ...
- [Mistral AI - GPU programming expert](https://jobs.lever.co/mistral/399978d0-b442-4591-b677-8cc03ee24a48): Mistral AI is hiring an expert in the role of serving and training large language models at high speed on GPUs. The role will involve - writing low-level code to take all advantage of high-end GPUs (H...

  

---


### CUDA MODE (Mark Saroufim) ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1203390140077314068) (10 messages🔥): 

- **C++ vs C for CUDA**: `@evil_malloc` asked whether C++ is a prerequisite for CUDA/Triton, and `@_tvi_` responded that proficiency in C++ is not necessary, but some familiarity is beneficial, especially when using CUDA with PyTorch.

- **Seeking C++ Mastery**: `@umaiskhan` sought advice on effectively learning C++, and `@stefangliga` recommended [LearnCpp.com](https://www.learncpp.com/), a free resource with extensive tutorials and examples.

- **Rust's Status in CUDA Programming**: `@greystark.` inquired about how Rust is supported for CUDA programming today, and `@andreaskoepf` highlighted the lack of active projects but shared the [rust-gpu](https://github.com/embarkstudios/rust-gpu) repository for GPU shaders in Rust.

- **Exploring Rust with CUDA**: `@andreaskoepf` further suggested using [Rust-CUDA](https://rust-gpu.github.io/Rust-CUDA/guide/getting_started.html) to `@greystark.` and others interested, which provides a guide for writing GPU crates with CUDA support in Rust.

- **Rust Neural Network Development**: `@andreaskoepf` followed up with more active Rust repositories related to neural network development leveraging CUDA, mentioning [Kyanite](https://github.com/KarelPeeters/Kyanite) and [burn](https://github.com/tracel-ai/burn) as projects to explore.

**Links mentioned**:

- [Getting Started - GPU Computing with Rust using CUDA](https://rust-gpu.github.io/Rust-CUDA/guide/getting_started.html): no description found
- [Learn C++ – Skill up with our free tutorials](https://www.learncpp.com/): no description found
- [GitHub - EmbarkStudios/rust-gpu: 🐉 Making Rust a first-class language and ecosystem for GPU shaders 🚧](https://github.com/embarkstudios/rust-gpu): 🐉 Making Rust a first-class language and ecosystem for GPU shaders 🚧 - GitHub - EmbarkStudios/rust-gpu: 🐉 Making Rust a first-class language and ecosystem for GPU shaders 🚧
- [GitHub - KarelPeeters/Kyanite](https://github.com/KarelPeeters/Kyanite): Contribute to KarelPeeters/Kyanite development by creating an account on GitHub.
- [GitHub - tracel-ai/burn: Burn is a new comprehensive dynamic Deep Learning Framework built using Rust with extreme flexibility, compute efficiency and portability as its primary goals.](https://github.com/tracel-ai/burn): Burn is a new comprehensive dynamic Deep Learning Framework built using Rust with extreme flexibility, compute efficiency and portability as its primary goals. - GitHub - tracel-ai/burn: Burn is a ...

  

---


### CUDA MODE (Mark Saroufim) ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1203788105916612649) (2 messages): 

- **Comparing Matrix Multiplication Timings**: `@antoniooooooooooooo` shared their **timing results** for 1024x1024 matrix multiplication: **CPU at 5,308,033μs**, **GPU Original at 131,237μs**, **GPU Row at 43,896μs**, and **GPU Columns at 32,179μs**. They asked for a comparison to see if the relationship between those timings makes sense.

- **Seeking Theoretical and Coding Answers**: `@antoniooooooooooooo` inquired about a resource for **more theoretical answers** as well as a repository with **coding solutions** related to the exercises in the PMPP book.
  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1203343936584814612) (88 messages🔥🔥): 

- **Model Training Queries and Misinterpretations**: A conversation around **TimesFM model training** led to clarifications by users like `@Hawk` and `@mrgonao`. Hawk originally questioned the efficiency of a training process before offering a corrected sequence (`input:1-32 -> output 33-160 -> input 1-160 -> output 161-288`), concluding that there should be no overlap in output patches according to the model's description.

- **Seeking Insights on Large Context LLMs**: User `@nsk7153` inquired about research on handling large contexts in LLMs, to which `@stellaathena` responded by sharing a [YaRN paper](https://arxiv.org/abs/2309.00071), a compute-efficient method to extend context window length.

- **A Novel Training Method Proposal**: User `@worthlesshobo` brought up an intricate discussion on autoencoding and proposed a method referred to as "liturgical refinement." They suggested a technique involving alternately freezing and unfreezing components of encoder-decoder models to potentially achieve more effective representations.

- **Ideas on Model Fusion and Constraints**: User `@win100` speculated about fusing model tensors from separate models (A and B) for improved pre-training, aligning with the concept of the FuseLLM project. `@!BeastBlaze` provided insights on a related approach taken in the LeMDA paper which focuses on augmenting feature embeddings.

- **Development of a Web UI for LLMs**: User `@318yang_` announced the development and deployment of a web UI for large language models (LLMs), [simple-ai.io](https://simple-ai.io), which is an open-source project that the community can utilize in their projects. They mentioned plans to integrate Ollma for local runs with this new UI.

**Links mentioned**:

- [no title found](https://simpel-ai.io~~): no description found
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071): Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length t...
- [Tweet from Valeriy M., PhD, MBA, CQF (@predict_addict)](https://x.com/predict_addict/status/1754134502895460421?s=20): A new paper from Google that peddles “foundational model” for time series forecasting is both an example of beginner mistakes coupled with deployment of deceptive “benchmarks.”  In figure 6 the author...
- [large language models for handling large context prompts | Semantic Scholar](https://www.semanticscholar.org/search?q=large%20language%20models%20for%20handling%20large%20context%20prompts&sort=relevance)): An academic search engine that utilizes artificial intelligence methods to provide highly relevant results and novel tools to filter them with ease.
- [simple ai - chat](https://simple-ai.io): no description found
- [Yes I Am Gru GIF - Yes I Am Gru Steve Carell - Discover &amp; Share GIFs](https://tenor.com/view/yes-i-am-gru-steve-carell-despicable-me2-yes-thats-me-gif-16561733): Click to view the GIF
- [Neuronpedia](https://www.neuronpedia.org/): AI Safety Game and Open Data Reference
- [Google Colaboratory](https://colab.research.google.com/drive/1ET3R_JkckEJ-LxJpd05PjbfGK-TyiPNF#scrollTo=RYadyNjP1-UC): no description found

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1203277277354393680) (32 messages🔥): 

- **MoE-Mamba Outshines its Peers**: In a [discussion about a recent paper](https://arxiv.org/abs/2401.04081), `@afcruzs` shares an Arxiv link to research presenting an SSM model "MoE-Mamba," which outperforms various state-of-the-art models with fewer training steps.
- **Striving for Equilibrium in Experts' Models**: `@catboy_slim_` hints at possible efficiency losses in Mixture of Experts (MoE) models due to imbalanced layer assignments and suggests considering an extra router loss to restore balance.
- **Computing Concepts go Origami**: `@digthatdata` posts a [Quantamagazine article](https://www.quantamagazine.org/how-to-build-an-origami-computer-20240130/) relating computation to origami, sparking a brief exchange on its potential link to in-context learning in AI.
- **Accessing Mamba Checkpoints**: In the wake of discussing potential issues with SSMs, `@woog` expresses interest in obtaining checkpoints for the Mamba model, which `@random_string_of_character` points towards being available upon request and also on [Hugging Face's Model Hub](https://huggingface.co/Zyphra/BlackMamba-1.5B).
- **Gradient Stabilization and Encodec**: A conversation unfolds around gradient stabilization with `@nostalgiahurts` referencing the [Encodec paper's approach](https://arxiv.org/abs/2210.13438) to handling multiple loss types, which introduces a normalizing mechanism for balancing gradients during training.

**Links mentioned**:

- [How to Build an Origami Computer | Quanta Magazine](https://www.quantamagazine.org/how-to-build-an-origami-computer-20240130/): Two mathematicians have shown that origami can, in principle, be used to perform any possible computation.
- [Tweet from Guillaume Bellec (@BellecGuill)](https://x.com/BellecGuill/status/1750814799615725793): @francoisfleuret This tiny PyTorch lib does exactly that. https://github.com/guillaumeBellec/multitask  Originally made to stop losing time fine-tuning coeffs of auxiliary losses.  For your case, just...
- [MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts](https://arxiv.org/abs/2401.04081): State Space Models (SSMs) have become serious contenders in the field of sequential modeling, challenging the dominance of Transformers. At the same time, Mixture of Experts (MoE) has significantly im...
- [Repeat After Me: Transformers are Better than State Space Models at Copying](http://arxiv.org/abs/2402.01032): Transformers are the dominant architecture for sequence modeling, but there is growing interest in models that use a fixed-size latent state that does not depend on the sequence length, which we refer...
- [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/abs/2210.14215): We propose Algorithm Distillation (AD), a method for distilling reinforcement learning (RL) algorithms into neural networks by modeling their training histories with a causal sequence model. Algorithm...
- [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438): We introduce a state-of-the-art real-time, high-fidelity, audio codec leveraging neural networks. It consists in a streaming encoder-decoder architecture with quantized latent space trained in an end-...
- [Tweet from Quentin Anthony (@QuentinAnthon15)](https://fixupx.com/QuentinAnthon15/status/1753584827100778965): Along the interpretability angle, if you want to study Mamba, we trained a pure Mamba-350M on our dataset.   We also trained the original Mamba-370M on the Pile!  All checkpoints and dataset available...
- [GitHub - Zyphra/BlackMamba: Code repository for Black Mamba](https://github.com/Zyphra/BlackMamba): Code repository for Black Mamba. Contribute to Zyphra/BlackMamba development by creating an account on GitHub.

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1203818349914619954) (1 messages): 

- **Clarifying "Direction" vs "Feature" in Interpretability**: `@pinconefish` posed a question about terminology in interpretability, asking if a "direction" refers to a vector in the embedding space that encodes monosemantic meaning. They noted that "direction" might help distinguish between the activation of a single neuron (also referred to as a "feature") and vectors in embedding space, which could be useful in discussions about semantic meanings at different levels in a model.
  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1203792742321233981) (5 messages): 

- **Schedule Locked In**: `@asuglia` confirmed with `@981242445696221224` and `@1072629185346019358` that **Tuesday 6th at 5pm (UK time)** would be a suitable meeting slot.
- **Invitation Preparation**: `@hailey_schoelkopf` agreed to the proposed meeting time and requested DMs for email addresses to send an invite to `@asuglia` and `@1072629185346019358`.
- **Large Scale Testing Tactics Discussed**: `@mistobaan` inquired about approaches for testing at scale, mentioning options like queues with workers or long single machine runs.
- **Slurm for Scaling Tests**: In response to `@mistobaan`, `@.johnnysands` mentioned they utilize **Slurm** by queuing a large number of jobs to manage scale testing.
- **Exploration of Prompt Previews**: `@Goyim` sought insight into the possibility of previewing prompts for specific tasks and the formatting of **multiple_choice** prompts submitted to models.
  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1203380953758834759) (9 messages🔥): 

- **Exploring MoE for Multimodal Approaches**: `@martianulcrizat` showed interest in guidance on Mixture of Experts (MoE) models for creating multimodal systems, hinting at integrating a transformer diffusion model with a VLM (visual language model) like LLaMA.
- **Seeking Deeper Semantic and Generative Integration**: `@martianulcrizat` discussed the potential for a tighter integration between semantic understanding and generative capabilities within a VLM by employing MoE frameworks.
- **Search for VLM and Diffusion Model Combination Techniques**: `@martianulcrizat` inquired about approaches for combining VLMs with diffusion models beyond the conventional methods involving QFormer, Adaptor layers, and cross-attention with continuous token representations.
- **Acceptance of Shared Papers on Integration Methods**: `!BeastBlaze` acknowledged the relevance of papers shared by `@martianulcrizat` which potentially could assist in VLM and diffusion model integration.
- **Alternative Simplifications to Combining VLMs with Diffusion Models**: `!BeastBlaze` mentioned new literature, albeit not readily available, which suggests the feasibility of using simple RNNs and CBOW to achieve similar outcomes to that of large models like CLIP, thereby enabling leaner methods like fast DINO or fast SAM.
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1203809119258021928) (2 messages): 

- **Clarification on "gas" Parameter Functionality**: `@catboy_slim_` noted a [pull request](https://github.com/EleutherAI/gpt-neox/pull/123) to remove the `"gas"` parameter, stating it is non-functional and redundant with `"gradient_accumulation_steps"`. They caution that historical runs using `"gas"` with values other than 1 may have an effectively smaller batch size than intended.
- **Review Incoming for "gas" Parameter**: `@tastybucketofrice` responded that they will review the issue regarding the `"gas"` parameter today.
  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1203260159967756328) (96 messages🔥🔥): 

- **Language Barrier? No problem!**: `@bondesign` inquired if typing in Chinese is possible, and the interest was echoed with a subsequent message in Chinese. `@cookie_74700` similarly asked if Persian could be spoken, highlighted by linked responses by `@mares1317`, underscoring Perplexity's multilingual capabilities.

- **Confusion around Copilot's role**: `@oudstand` shared observations on how using Copilot seemed to improve model performance. Meanwhile, others like `@dlysltrading` faced issues with Copilot, which were resolved by refreshing the webpage, and `@stocktown` sparked a debate on the rationale of using write mode with Copilot active.

- **Navigating Perplexity's Customer Service**: `@aqbalsingh` expressed frustration with changing the email on their account and the absence of an upload button on the iPhone app, leading to the cancellation of their premium account. Despite responses from Perplexity, the user remained disappointed with the support responsiveness.

- **Perplexity AI Discord Integration Woes**: `@otchuda` lamented the missing Discord integration for a quick Perplexity response, prompting discussions and links shared by `@icelavaman` and `@ok.alex` but confirming that there are no current plans to bring back the textual Perplexity bots to Discord.

- **Exploring API keys and Use Cases**: `@elanutta` queried about generating API keys for Open AI with a Perplexity account, while `@glisteningsunlight` reported and self-resolved a delay issue when attempting to get Perplexity to summarize a PDF. Further, discussions around usage quotas and product comparisons between ChatGPT Plus and Perplexity Pro were undertaken by `@felirami`, `@general3d`, and `@maverix.`.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1118264005207793674/1203626274132590612): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047204950763122820/1175478858816950343): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1111786888626438245/1193465802259709992): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649527299055688/1197892547276705843): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649527299055688/1202651487747047515): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047649619695390740/1202652559463153796): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1183781616515031124/1184427097272365106): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [What is Perplexity Copilot?](https://blog.perplexity.ai/faq/what-is-copilot): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Images &amp; media](https://blog.perplexity.ai/faq/images-media): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1203317593008767010) (14 messages🔥): 

- **Videos Highlighting Perplexity AI**: `@arunprakash_`, `@boles.ai`, and `@ok.alex` all shared YouTube videos showcasing Perplexity AI, discussing its benefits and features, and explaining why users might choose it over other AI options. The videos titled "Do we really need Perplexity AI Pro Subscription?", "Perplexity and Play.HT Don't Play! Plus, a Hindenburg Review!", and "I Ditched BARD & ChatGPT & CLAUDE for PERPLEXITY 3.0!" can be found on their respective [YouTube links](https://www.youtube.com/watch?v=eqSkH_p8CZ8), [YouTube links for Play.HT review](https://youtube.com/live/LKQTETpxS_8?feature=share), and [YouTube links for Perplexity 3.0 review](https://youtu.be/mFfS4BYCCgw?si=ysvRx4_yU5FxU0Qe).

- **Sharing Revelatory Perplexity Searches**: Users `@rocktownarky`, `@bwatkins`, `@maverix.`, `@gamezonebull`, `@epic9713`, and `@darkspider1987` shared direct links to their Perplexity AI search results which provided valuable insights leading to a Pro subscription decision and helped with complex decision-making. The shared results can be accessed through the provided Perplexity AI search [links](https://www.perplexity.ai/search/46ad8505-d3e4-4cc1-ab64-624b18a9affc), [links for Prop 1 decision](https://www.perplexity.ai/search/Help-me-decide-UEJrUP.XROOH.V5_3HU1wQ?s=c), [links for 'what is'](https://www.perplexity.ai/search/what-is-the-DOn_dKY5RTuDw.SjxeoP8Q), [links for how to get](https://www.perplexity.ai/search/How-to-get-pOdKd44XSh6F3wyxadfBbQ?s=u), and [links for AI image motion](https://www.perplexity.ai/search/AI-image-motion-RVRGLZHARw2nQBkJuA12Eg?s=c#c039b3c1-2cf2-444a-a523-5f251ffe61c8).

- **Public Search Sharing Tips**: `@me.lk` advised `@maverix.` and `@gamezonebull` to ensure their search threads are public by clicking on the share button in the top right corner, while `@noremac258` noted that `@darkspider1987`'s search result wasn't viewable, indicating the importance of making searches public for community sharing.

- **Redirect Link with Null Description**: `@ok.alex` posted a [Discord redirect](https://discord.com/channels/1047197230748151888/1054944216876331118/1203759744217653278) with no description provided for the content it leads to.

- **Productive Weekend with Perplexity**: `@johnweisenfeld` shared a [LinkedIn post](https://www.linkedin.com/posts/weisenfeldj_about-pasco-high-school-activity-7160105805193641984-zMSr?utm_source=share&utm_medium=member_desktop) about a productive weekend thanks to Perplexity and a mention about difficulties with other AI services, praising OpenAI for helping to get a code project off the ground.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1054944216876331118/1203759744217653278): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [🔥 I Ditched BARD &amp; ChatGPT &amp; CLAUDE for PERPLEXITY 3.0! 😱 #MindBlown](https://youtu.be/mFfS4BYCCgw?si=ysvRx4_yU5FxU0Qe): In this video I’m showing off the newest featurest inside of Perplexity AI.🚨 Huge Discounts &amp; Free Training &amp; My Courses: https://wasuniverse.com ✅ Get FREE...
- [Do we really need Perplexity AI Pro Subscription?](https://www.youtube.com/watch?v=eqSkH_p8CZ8): Explore Perplexity AI&#39;s innovative features in this video! Discover its integrated Reddit and YouTube search, focused search modes, source exclusion, AI Comp...
- [Perplexity and Play.HT Don&#39;t Play! Plus, a Hindenburg Review!](https://youtube.com/live/LKQTETpxS_8?feature=share): We look at AI search, AI voice and more traditional ways to record your voice. Plus, Apple Podcast transcripts!

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1203350676214784010) (7 messages): 

- **Mixtral Pricing Inquiry**: `@paul16307` asked about pricing for Mixtral since Perplexity removed 13b from API pricing. `@icelavaman` responded with current rates: **$0.14 per 1M input tokens and $0.56 per 1M output tokens**.
- **Potential Rate Limit Increase**: `@aiistheonlyway` queried about getting a quick rate limit increase but did not receive a response in the provided messages.
- **Curiosity About Mixtral's Future Pricing**: Following the pricing details, `@paul16307` inquired if the future pricing for Mixtral would be lower, but no response was provided.
- **Request for pplx-web API Version**: `@makadoro_95229` suggested a business opportunity for Perplexity to offer a **pplx-web version** of the API, which would give results akin to the website, aiding in the creation of chat assistants for other websites. `@defektivex` supported this idea, mentioning that many have requested this feature and expressed hope for a similar API in the future.
- **Integration of Perplexity AI with Siri**: `@out_a_time6794` asked about setting up Perplexity AI with Siri to function as a shortcut for queries, with no follow-up response provided in the conversation.
  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1203313070848614440) (34 messages🔥): 

- **Seeking the Right Tech for Arabic Content**: `@mukhtor` inquired about technology for chatting with Arabic content. `@lhc1921` suggested an Arabic Language Model (LLM) and embeddings, mentioning that most technologies are language-agnostic, while `@hro_ffs_why_cant_i_use_my_name` mentioned that [embedding-ada](https://www.sbert.net/examples/training/multilingual/README.html) mainly supports "French, English, German, Spanish, and Portuguese" but listed *aravec* and *word2vec* as potential alternatives.
  
- **Cost-Effective Hosting for Autonomous Agent**: `@charly8323` sought advice on hosting a cost-effective autonomous research agent while keeping costs below the revenue from a 5 cents per call price structure. `@engineered.mind` recommended a local LLM for controllable costs, and `@truethinker` suggested deploying [ollama](https://ollama.com/) on a server like DigitalOcean.

- **Upcoming Book Alert**: `@mehulgupta7991` announced the release of their new book titled "LangChain in your Pocket: Beginner's Guide to Building Generative AI Applications using LLMs," detailing how to use LangChain for various applications, available on [Amazon](https://amzn.eu/d/dqQJzV1).

- **Efficiency Tips for Long Document Translation with LangChain**: `@o3omoomin` asked for efficient methods to translate lengthy documents with LangChain to avoid token limitations. The user was exploring segmentation into smaller chunks and sought example code for more streamlined handling.

- **Hosting and Fine-Tuning Challenges Discussed**: Various members, including `@lhc1921`, `@nrs`, and `@sullynaj`, discussed hosting and fine-tuning models in the cloud. Suggestions included using local models, Google Colab, and Cohere embeddings, with potential strategies for training an Arabic model using a relevant dataset.

**Links mentioned**:

- [no title found](https://amzn.eu/d/dqQJzV1): no description found
- [GitHub - BBC-Esq/Nvidia_Gpu_Monitor: Realtime Monitor of Nvidia GPU Metrics with NVML Library](https://github.com/BBC-Esq/Nvidia_Gpu_Monitor): Realtime Monitor of Nvidia GPU Metrics with NVML Library - GitHub - BBC-Esq/Nvidia_Gpu_Monitor: Realtime Monitor of Nvidia GPU Metrics with NVML Library
- [GitHub - facebookresearch/contriever: Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning](https://github.com/facebookresearch/contriever): Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning - GitHub - facebookresearch/contriever: Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1203322976418930748) (7 messages): 

- **Introducing CastMate for Interactive Podcast Experiences**: `@darrelladjei` shared the launch of **CastMate**, a platform allowing users to listen and interact with their favorite podcast episodes, featuring fakes using LLMs and human-quality TTS. They provided a [Loom demonstration](https://www.loom.com/share/c7a82509eaca450c814fae77c5db7a1d?sid=67313ae9-fca0-4a55-b536-a93b711a9d74) and invited feedback, while also offering an **iPhone beta**: [TestFlight Link](https://testflight.apple.com/join/9khwQ1vD).

- **GUI Discussions for Artificial Agents**: User `@clickclack777` inquired about which GUI is being used, which led to `@robot3yes` mentioning their work on **Agent IX**, a side project meant to interface with bots.

- **New Book Alert: Guide to Generative AI Applications**: `@mehulgupta7991` announced their debut book, "*LangChain in your Pocket: Beginner's Guide to Building Generative AI Applications using LLMs*", which covers a range from basic to advanced use cases involving LangChain. They shared a broken Amazon [link](https://amzn.eu/d/dqQJzV1), which instead displayed a CAPTCHA verification page.

- **Meet the Author and Data Scientist**: In a follow-up message, `@mehulgupta7991` introduced themselves as a data scientist with experience at DBS Bank and shared their "Data Science in your Pocket" Medium and YouTube channels. They specifically pointed to a LangChain [YouTube playlist](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=UGdxhD6LwRSgQzno) for tutorials.

- **Seeking Tips for a Goal-Setting Assistant**: `@mark_c_` requested architectural advice for creating a goal-setting assistant that manages long-term and short-term goals and helps with weekly scheduling, mentioning their background as an ex-coder. They were interested in starting with prompt engineering but anticipated the need for a more complex workflow.

- **A Tool for AI-enhanced Due Diligence in Investment**: `@solo78` introduced a project tool utilizing Langchain to perform deep due diligence for investing in platforms and companies. They shared their Medium blog post detailing the project's journey and sought thoughts from the community: [Medium Article](https://medium.com/@bsouleymane78/06a8c6c375ff).

**Links mentioned**:

- [Langchain](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=UGdxhD6LwRSgQzno): This playlist includes all tutorials around LangChain, a framework for building generative AI applications using LLMs
- [Developing an IA tool for Investing platform Due Diligence using LLM and RAG](https://medium.com/@bsouleymane78/06a8c6c375ff): Go through my journey to develop a AI tool using Generative AI and Python to perform Due Diligence of investing platforms.
- [Loom | Free Screen &amp; Video Recording Software](https://www.loom.com/share/c7a82509eaca450c814fae77c5db7a1d?sid=67313ae9-fca0-4a55-b536-a93b711a9d74): Use Loom to record quick videos of your screen and cam. Explain anything clearly and easily – and skip the meeting. An essential tool for hybrid workplaces.
- [no title found](https://amzn.eu/d/dqQJzV1): no description found

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1203343677993525288) (5 messages): 

```html
<ul>
  <li><strong>Next.js and LangChain for SMART Portfolios</strong>: User <code>@flo_walther</code> shared a <a href="https://www.youtube.com/watch?v=1LZltsK5nKI">YouTube video</a> on building a SMART portfolio website using <strong>Next.js 14, Langchain, Vercel AI SDK</strong>, and more, highlighting an AI chatbot that can be trained on your data.</li>
  <li><strong>Tutorial Troubles</strong>: <code>@stuartjatkinson</code> expressed frustration that steps in YouTube tutorials for LangChain have changed or yield errors when followed directly.</li>
  <li><strong>LangChain Starting Stumbles</strong>: <code>@treym1112</code> encountered errors while following the langchain quick tutorial on the LangChain website, specifically with the <strong>Ollama model</strong>, resulting in an <em>AttributeError</em> concerning the missing 'verbose' attribute.</li>
  <li><strong>LangChain Guide Launched</strong>: <code>@mehulgupta7991</code> announced the release of their book "<em>LangChain in your Pocket: Beginner's Guide to Building Generative AI Applications using LLMs</em>" on <a href="https://amzn.eu/d/dqQJzV1">Amazon</a>, describing it as a hands-on guide covering a range of use cases and LangServe deployment.</li>
  <li><strong>Meet the Data Scientist and Content Creator</strong>: <code>@mehulgupta7991</code> shared their professional background as a data scientist at DBS Bank and mentioned their Medium+YouTube channel "<em>Data Science in your Pocket</em>" which features around 600 tutorials, including a <a href="https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ">LangChain playlist</a>.</li>
</ul>
```

**Links mentioned**:

- [Langchain](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=UGdxhD6LwRSgQzno): This playlist includes all tutorials around LangChain, a framework for building generative AI applications using LLMs
- [Build a SMART Portfolio Website (Next.js 14, Langchain, Vercel AI SDK, ChatGPT API, Tailwind CSS)](https://www.youtube.com/watch?v=1LZltsK5nKI): The coolest portfolio website you can build to impress recruiters and friends! It has an AI chatbot that is trained on YOUR data. The AI can answer any quest...
- [no title found](https://amzn.eu/d/dqQJzV1): no description found

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1203389816608264294) (5 messages): 

- **RAG Development Challenges Solved**: `@wenqi_glantz` detailed 12 pain points when building production RAG and, together with `@llama_index`, provided a full solution list for each, available on a newly released cheatsheet. The announcement and solutions can be found in their [Twitter post](https://twitter.com/llama_index/status/1753829824484065625).
- **Hackathon Hustle at DataStax**: The hackathon kicked off at 9 am with @llama_index appreciating `@DataStax` for providing the venue and food. Insights into the event are shared on their [Twitter update](https://twitter.com/llama_index/status/1753845015833686132).
- **Multimodal Models on MacBooks**: `@llama_index` announced their new integration with Ollama, enabling the development of local multimodal applications such as structured image extraction and image captioning. More details are available in their [day 1 integration tweet](https://twitter.com/llama_index/status/1753875735776018786).
- **Multilingual Embedding Optimization Techniques**: A new article by Iulia Brezeanu on @TDataScience discusses selecting the right embedding model for multilingual RAG, addressing language biases in benchmarks. The article helps navigate models not optimized for English and is shared via [LlamaIndex's tweet](https://twitter.com/llama_index/status/1754185891118239894).
- **Discord Welcomes LlamaIndex's Slack Bot**: `@llama_index` has released a Discord version of their popular Slack bot. Interested users can access it through the link shared in their [tweet announcement](https://twitter.com/llama_index/status/1754257239525982685).
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1203309190773477456) (19 messages🔥): 

- **Seeking Clarity on AI Interpretation**: `@meowmeow008` is exploring how the AI can interpret SQL queries and subsequent requests, like calculations of percentages, and wonders about potential misunderstandings concerning the AI’s capabilities.
- **Azure's AI Gives Mixed Results**: `@aldrinjoseph` experienced issues when switching from Azure OpenAI 3.5 Turbo to Azure OpenAI 3.5 Turbo 16K, with the latter generating answers outside the given context.
- **LlamaIndex More Reliable than LangChain**: `@7leven` expressed a preference for LlamaIndex over LangChain, criticizing the latter for frequently breaking its documentation and causing less trouble when integrated.
- **Tweaking Hybrid Retriever Without Reinstantiation**: In a discussion about the **Hybrid Retriever**, `@cheesyfishes` confirmed to `@7leven` that the **alpha** parameter can be adjusted dynamically in the Python code without needing reinstantiation.
- **RAG App Development and Integrating Chat History**: `@jameshume` is seeking guidance on incorporating chat history into an app that leverages multiple components including a custom `VectorDBRetriever` and `CondenseQuestionChatEngine`; `@dirtikiti` explained a simple method of tracking and including chat history in new prompts.

**Links mentioned**:

[Usage Pattern - LlamaIndex 🦙 0.9.44](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/usage_pattern.html#available-chat-modes): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1203387117682819072) (7 messages): 

- **Exploration of Recursive Retrieval**: `@chiajy` provided insights on recursive or iterative retrieval in developing **self-learning RAG systems** that can deeply delve into unstructured data. They shared their Medium article detailing three recursive retrieval techniques: Page-Based, Information-Centric, and Concept-Centric, available at [Advanced RAG and the 3 types of Recursiv...](https://medium.com/enterprise-rag/advanced-rag-and-the-3-types-of-recursive-retrieval-cdd0fa52e1ba).

- **Appreciation for Recursive Retrieval Techniques Article**: User `@jerryjliu0` expressed appreciation for `@chiajy`'s article on recursive retrieval in RAG systems, calling it a "nice article!"

- **Showcasing LlamaIndex Comparisons**: `@andysingal` shared an article that compares embedding techniques from Jina AI, Nomic AI, and FlagEmbedding, discussing their integration with Llamaindex. The article, titled "Unveiling the Power of Llamaindex", explores the synergy of these technologies in AI: [Unveiling the Power of Llamaindex](https://medium.com/ai-advances/unveiling-the-power-of-llamaindex-jina-vs-nomic-ai-vs-flagembedding-557158d7ad1e).

- **Introduction of BGE-M3 Embedding Model**: `@alphaatlas1` introduced the **BGE-M3 embedding model**, highlighting its multi-functionality, multi-linguality, and multi-granularity features. The model can perform dense retrieval, multi-vector retrieval, and sparse retrieval, supporting over 100 languages and processing various input granularities up to 8192 tokens, explained on Hugging Face: [BGE-M3 on Hugging Face](https://huggingface.co/BAAI/bge-m3).

- **Recommendation for Retrieval Pipeline in RAG**: `@alphaatlas1` recommended a hybrid retrieval plus re-ranking pipeline for RAG retrieval, to leverage the benefits of various methods for higher accuracy. They mentioned that the BGE-M3 model simplifies embedding retrieval by not requiring additional instructions for queries and invited community contributions for sparse retrieval methods.

**Links mentioned**:

- [Advanced RAG and the 3 types of Recursive Retrieval](https://medium.com/enterprise-rag/advanced-rag-and-the-3-types-of-recursive-retrieval-cdd0fa52e1ba): We explore the 3 types of recursive retrieval — Page-Based, Information-Centric, and Concept-Centric retrieval
- [BAAI/bge-m3 · Hugging Face](https://huggingface.co/BAAI/bge-m3): no description found
- [Unveiling the Power of Llamaindex: Jina vs Nomic AI vs FlagEmbedding](https://medium.com/ai-advances/unveiling-the-power-of-llamaindex-jina-vs-nomic-ai-vs-flagembedding-557158d7ad1e): Ankush k Singal

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1203260107606073384) (29 messages🔥): 

- **Exploring GPT Stores and APIs**: `@tiagoefreitas` inquired about GPT stores with APIs similar to OpenRouter and OpenGPTs under a public server, expressing a wish that **@LangChainAI** would implement federation in OpenGPTs. They clarified that federation would allow using GPTs from other servers through API while managing their own server.

- **Open Models Over Traditional Writing**: `@slono` criticized the traditional writing approach, highlighting the enjoyment and productivity of working with stochastic models like **mlewd mixtral**. The discussion suggested a preference for open models’ dynamic output over the standard writing methodologies.

- **Sentry Dives into Q&A**: `@kaycebasques` pointed out a trend in Q&A solutions, exemplifying this by sharing Sentry's approach which has created a vast Q&A resource for over 20 programming languages and frameworks. This indicates a broader movement towards specialized technical Q&A platforms.

- **Llava Inference Speed Impresses**: `@ashpreetbedi` shared their positive experience regarding **inference speed** while running **Ollama Llava** locally on their MacBook, contributing to the community’s understanding of the tool's performance.

- **Career Crossroads in Tech**: `@mr.osophy` conveyed hesitation about accepting a job unrelated to their interest in ML Engineering, weighing the benefits of becoming a better candidate for desired roles against immediate but unrelated job opportunities. The dilemma underscored the challenges tech professionals face when aligning career moves with personal aspirations and financial constraints.

**Links mentioned**:

- [Arcee and mergekit unite](https://blog.arcee.ai/arcee-and-mergekit-unite/): Several months ago, I stumbled upon an innovative technique in the world of language model training known as Model Merging. This SOTA approach involves the fusion of two or more LLMs into a singular, ...
- [Sentry Overflow](https://technicalwriting.tools/posts/sentry-overflow/): Just now, as I was searching for the Bash script syntax on how to check if a directory exists (for the 63rd time because I have the memory span of an agitated Chihuahua) I noticed something interestin...

  

---


### Latent Space ▷ #[llm-paper-club-east](https://discord.com/channels/822583790773862470/1200029657744027658/) (1 messages): 

swyxio: check out the summary
  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1203390847085969418) (28 messages🔥): 

- **Combos Pulled from a Pregenerated Pool**: `@dbreunig` theorizes that *crazier combos* in a game pulse and have a delay because new ones are generated when there's a hash miss, implying that past combinations are stored and used.
- **Intriguing Game Mechanics Spark Curiosity**: `@chrisamico` and `@cameron_y` express a desire to visualize the genealogy of the game's word combinations, speculating about possible insight *embeddings* could provide into crafting paths.
- **Harnessing the Crafting Code**: `@madacol` provides a *JavaScript bookmarklet* that interacts with the game's `localStorage` to export and auto-save crafted items and discoveries, adding a new dimension to the gameplay experience by retrieving all ingredients crafted.
- **Game Powered by LLM-2**: `@madacol` clarifies the AI behind the game's clever combinations, posting that the creator is using *llama 2* as disclosed in a tweet by @nealagarwal and provided by TogetherAI.
- **Crafting Direction Matters**: `@madacol` discovered that in this game, the order of combining elements affects the outcome, with some successful results only when certain items are placed on top of others, and notes the server's memory of attempted combinations prohibits reversing the order after a try.
  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1203298413458759722) (8 messages🔥): 

- **Boosting German Model Performance**: `@johannhartmann` discussed improvements in mt-bench-de scores after using **German dpo** and **laserRMT**, with a focus on merging German 7B-models using dare_ties.
- **Curiosity About Mysterious Improvement Methods**: Upon request for details by `@philipmay`, `@johannhartmann` provided [links to German dpo](https://github.com/mayflowergmbh/intel_orca_dpo_pairs_de) and [laserRMT](https://github.com/cognitivecomputations/laserRMT), but admitted to not fully understanding why performance changes occurred, especially a drop in math ability.
- **Seeking Research on LLMs for Large Context**: `@nsk7153` inquired about research on handling large language models (LLMs) for large context and shared a [Semantic Scholar search link](https://www.semanticscholar.org/search?q=large%20language%20models%20for%20handling%20long%20context%20prompts&sort=relevance) to the materials they've reviewed.
- **Introduction of GermanRAG Dataset**: `@rasdani` proudly released the GermanRAG dataset for fine-tuning **Retrieval Augmented Generation**. They shared the [GitHub link](https://github.com/rasdani/germanrag) to the dataset and encouraged customization and enhancement.
- **Scandinavian Language Model Envy**: `@johannhartmann` discovered the [ScandEval benchmark](https://scandeval.com/mainland-scandinavian-nlg/) for Scandinavian natural language generation and expressed a desire for something similar for the German language.

**Links mentioned**:

- [Mainland Scandinavian NLG](https://scandeval.com/mainland-scandinavian-nlg/): A Natural Language Understanding Benchmark
- [large language models for handling long context prompts | Semantic Scholar](https://www.semanticscholar.org/search?q=large%20language%20models%20for%20handling%20long%20context%20prompts&sort=relevance)): An academic search engine that utilizes artificial intelligence methods to provide highly relevant results and novel tools to filter them with ease.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1178995845727785010/1182877486854451271/1201826534114218034): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [GitHub - rasdani/germanrag: GermanRAG - a German dataset for finetuning Retrieval Augmented Generation](https://github.com/rasdani/germanrag): GermanRAG - a German dataset for finetuning Retrieval Augmented Generation - GitHub - rasdani/germanrag: GermanRAG - a German dataset for finetuning Retrieval Augmented Generation
- [GitHub - cognitivecomputations/laserRMT: This is our own implementation of &#39;Layer Selective Rank Reduction&#39;](https://github.com/cognitivecomputations/laserRMT): This is our own implementation of &#39;Layer Selective Rank Reduction&#39; - GitHub - cognitivecomputations/laserRMT: This is our own implementation of &#39;Layer Selective Rank Reduction&#39;

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/) (1 messages): 

flozi00: I am working on it at the moment to provide an german hosting service
  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1203436983083466794) (5 messages): 

- **Inquiry about `mistral-7B open-orca` training data**: `@njb6961` showed interest in replicating **mistral-7B open-orca** and asked if the `curated filtered subset of most of our GPT-4 augmented data` used for training would be released.
- **Search for the specific dataset**: `@njb6961` speculated that the dataset in question might be [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca), which includes ~500k GPT-4 completions and is curated to improve performance with less compute.
- **Confirmation of dataset**: `@ufghfigchv` confirmed that the [SlimOrca dataset](https://huggingface.co/datasets/Open-Orca/SlimOrca) is indeed the subset used and mentioned that the **training configuration** for the model should be in the **config** subdirectory of the model's repository.
- **Request for marketing contacts**: `@tramojx` reached out to the admin seeking contact for a **listing and marketing proposal**, but no response was provided in the available message history.

**Links mentioned**:

[Open-Orca/SlimOrca · Datasets at Hugging Face](https://huggingface.co/datasets/Open-Orca/SlimOrca): no description found

  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=N5lDUZRI8sc
  

---


### Skunkworks AI ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/1203924122434801674) (1 messages): 

- **Query on Document Embeddings versus Vision Embeddings**: `@epinnock` reached out to `@far_el` with a query about creating a version of **llava** with whole document text embeddings, contrasting it with a vision embedded approach. They ponder the task as potentially being a partial reimplementation of an encoder/decoder model, and seek clarification on what the assignment might entail beyond that.
  

---



### LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/1203812625641635841) (1 messages): 

- **Easy Deployment with BentoML**: User `@robotums` mentioned successfully deploying open-source software models with **BentoML** using a VLLM backend on AWS. They described the process as quite straightforward: "it's pretty easy, you just run the bento."
  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1203453017672777842) (1 messages): 

- **Introducing DSPy, a Language Model Programming Framework**: User `@sourya4` introduced `[DSPy](https://github.com/stanfordnlp/dspy)`, a Stanford project for programming—not just prompting—foundation models. They also shared a [YouTube video](https://www.youtube.com/watch?v=Dt3H2ninoeY) titled "SBTB23: Omar Khattab, DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines," describing advancements in prompting language models and their pipeline integration.

**Links mentioned**:

- [SBTB23: Omar Khattab, DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://www.youtube.com/watch?v=Dt3H2ninoeY): The ML community is rapidly exploring techniques for prompting language models (LMs) and for stacking them into pipelines that solve complex tasks. Unfortuna...
- [GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models](https://github.com/stanfordnlp/dspy): DSPy: The framework for programming—not prompting—foundation models - GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models

  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1203289597669539890) (1 messages): 

- **AIEF Bulgaria Chapter Buzzing with Activity**: `@yavor_belakov`, the lead of AIEF's first chapter in Bulgaria, highlighted the **second monthly AIEF BG meet-up with 90 attendees** and the introduction of 'Lightning Talks'. Networking, pizza, and knowledge exchange are at the core of the event.
- **Insights into Lightning Talks Now Available**: Presentations from the recent event are shared for sneak peeks into topics like **QR Code Art, Weaving The Past, LMMs, Zayo**, and **building a defensible business in the age of AI**. Full recordings to be posted on their YouTube channel.
- **Exploring ChatGPT Adoption**: Included in the talks was a presentation on **"ChatGPT Adoption Methodology"** by Iliya Valchanov, details are available in the shared [Google Slides document](https://docs.google.com/presentation/d/1XPMlt-qlZLagrvk4trNEI16ZSOPHRVGx).
- **LinkedIn as a Window to AIEF Bulgaria's Progress**: `@yavor_belakov` also posted highlights from the event on LinkedIn, showcasing the strength and excitement of the **AIEF foundation community**.

Links to the Google Slides presentations and the LinkedIn post are not fully provided, so they cannot be accurately shared.

**Links mentioned**:

- [Yavor_Belakov_QR_Code_Art.pptx](https://docs.google.com/presentation/d/1yVJISPqnkaM8RjF8pBjEnJO9XXVm78v7): 1 SOFIA 01 FEB 2024 Meetup #2 AIE.F AI Engineer Foundation | Europe | Chapter Bulgaria 6PM – 7PM Networking &amp; Pizza 7PM – 9PM Lightning Talks 9PM – 10PM More Networking AIEF Meetup #2: Lighting Ta...
- [Dani_Matev_Weaving_The_Past.pptx](https://docs.google.com/presentation/d/1c8qVQJ5SmYGGSlm5-Ae80XxutDF2qwaf): Knitting make (a garment, blanket, etc.) by interlocking loops of wool or other yarn with knitting needles or on a machine. Hi everyone and thank you for the opportunity to present here. Now, before w...
- [Dimo_Michev_LLMs.pptx](https://docs.google.com/presentation/d/19f01za6w5eZQPI3sYhNr2ILIFrsRXYW3): LLM package for Python Command-line and python package to query Large Language Models Your on-site private LLM tool https://llm.datasette.io/en/stable/index.html#
- [Nicole_Yoan_Zayop.pptx](https://docs.google.com/presentation/d/1P9_mU30ed9nuLgN2esahU7svtsNuKRL8): Zayo Reimagining Employee Management with Conversational UX 01/02/2024
- [Georgi_Stoyanov_How to Build a Defensible Business in the Age of AI.pptx](https://docs.google.com/presentation/d/1sNj3Q6Fd4jYjVVvjJBen_IgA-nbpm4x4): 1 How to Build a Defensible Business in the Age of AI
- [Iliya_Valchanov_ChatGPT Adoption Methodology.pptx](https://docs.google.com/presentation/d/1XPMlt-qlZLagrvk4trNEI16ZSOPHRVGx): 1 ChatGPT Adoption Methodology Iliya Valchanov

  


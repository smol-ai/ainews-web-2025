---
id: 46de1d43-b10e-4ea3-92b0-03485362004d
title: Trust in GPTs at all time low
date: '2024-02-02T03:25:24.640203Z'
original_slug: ainews-trust-in-gpts-at-all-time-low
description: >-
  AI community discussions highlight issues with GPT context management, reviews
  of GPT store, new CUDA Discord, and various model developments including
  Mistral, Miqu, Llama3, and fine-tuning challenges. Topics include model
  merging, decentralized AI incentives, and datasets like Open Hermes 2.5.
companies:
  - openai
  - huggingface
  - bittensor
models:
  - gpt-4
  - mistral
  - llama3
  - miquella-120b-gguf
  - harmony-4x7b-bf16
  - smaug-34b-v0.1
topics:
  - large-language-models
  - fine-tuning
  - model-merging
  - decentralized-ai
  - datasets
  - visual-reasoning
  - ocr
  - model-incentives
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 1/31/2024. We checked **21** guilds, **312** channels, and **8530** messages for you. Estimated reading time saved (at 200wpm): **628 minutes**.

It's been about 3 months since GPTs were released and ~a month since the GPT store was launched. But the reviews have been [brutal](https://twitter.com/wangzjeff/status/1752780035491336661):

 ![image.png](https://assets.buttondown.email/images/353f9ee8-05cc-41d1-99b0-b48c8fcdbb41.png?w=960&fit=max) 

Nick Dobos (of Grimoire fame) also blasted [the entire knowledge files capability](https://twitter.com/nickadobos/status/1749837866300264529) - it seems the RAG system naively includes 40k characters' worth of context from docs every time, reducing available context and adherence to system prompts.

 ![image.png](https://assets.buttondown.email/images/27b1e8c6-ee66-4110-97c7-781ec2b11148.png?w=960&fit=max) 

All pointing towards needing greater visibility for context management in GPTs, which is somewhat at odds with OpenAI's clear no-code approach.

---

In meta (pun?) news, warm welcome to our newest Discord scraped - Saroufim et al's CUDA MODE discord! Lots of nice help for those new to CUDA coding

 ![image.png](https://assets.buttondown.email/images/2e5e1e7a-f981-40a7-a00b-d607b3a9e4d0.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/39ada478-66e7-46a4-9c42-ddc12183b229.png?w=960&fit=max) 

---

**Table of Contents**

[TOC] 

---

# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Xeon's eBay Value for GPU Servers**: Members cited cost-effectiveness of **Xeon** processors in GPU servers sourced from eBay.
- **Speculation and Performance of Llama3**: Conversations around **Llama3** surfaced, juxtaposed with existing models like **Mistral Medium**, while **LLaVA-1.6** was mentioned to potentially exceed **Gemini Pro** in visual reasoning and OCR capabilities, with details shared [here](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/).
- **Miqu's Identity and Performance Drives Debate**: Leaked 70B model, **Miqu**, sparked discussions on its origins, performance, and implications of the leak, linking [Miquella-120b-gguf at Hugging Face](https://huggingface.co/alpindale/miquella-120b-gguf).
- **Fine-tuning, Epochs, and Dataset Challenges**: Technical support was provided for fine-tuning **TinyLlama** models leading to **ValueError**, with a [pull request on GitHub](https://github.com/huggingface/peft/pull/1399) indicating an upcoming release to resolve issues with unsupported modules. Meanwhile, users explored the potential of fine-tuning a **MiquMaid+Euryale** model using **A100 SM**.
- **Uncharted Territories of Model Merging**: Dialogue on model merging techniques showcased examples like **Harmony-4x7B-bf16**, deemed a successful model merge, with links provided to [ConvexAI/Harmony-4x7B-bf16 on Hugging Face](https://huggingface.co/ConvexAI/Harmony-4x7B-bf16). Additionally, a fine-tuned version of "bagel," [Smaug-34B-v0.1](https://huggingface.co/abacusai/Smaug-34B-v0.1), was shared for having excellent benchmark results without mergers.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Debating Style Influence in LLMs**: In [ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1202201419906814002), `@manojbh` opened a discussion on how language models like Mistral may mimic styles specific to their training data, linking to a tweet by `@teortaxesTex` that highlighted a peculiar translation error by Miqu with similar phrasing. They also brought up issues around quantized models losing information.
- **Incentives to Innovate with Bittensor**: Ongoing discussions in [interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1202267069425389680) regarding how Bittensor incentivizes AI model improvements, with comments on network efficiency and its potential to produce useful open-source models. There's interest in how the decentralized network structures incentives, with emphasis on costs and sustainability.
- **Model Exploration and Crowdfunding News**: In [general](https://discord.com/channels/1053877538025386074/1149866623109439599/1202166176281792552), MIQU has been identified as Mistral Medium by `@teknium`, confirming Nous Research's co-founder's [tweet](https://twitter.com/arthurmensch/status/1752737462663684344). Community members are actively combining models like MIQU to explore architecture possibilities. Additionally, AI Grant is highlighted as an accelerator offering funding for AI startups.
- **Open Hermes 2.5 Dataset and Collaborations Announced**: `@teknium` [announced](https://discord.com/channels/1053877538025386074/1145143867818119272/1202360039688646747) the release of Open Hermes 2.5 dataset on [Hugging Face](https://huggingface.co/datasets/teknium/OpenHermes-2.5). Collaboration with Lilac ML was mentioned, featuring Hermes on their HuggingFace Spaces.
- **Questions around RMT and ACC_NORM Metrics**: Within [ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1202428115998085140) and [benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/1202325748644581467), `@hexani` queried the uniqueness of a technology compared to **RMT**, expressing skepticism about its impact on context length, while `@euclaise` confirmed `acc_norm` is **always used** where applicable for AGIEval evaluations.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Dequantization Success Unlocks Potential**: The successful dequantization of [miqu-1-70b](https://huggingface.co/152334H/miqu-1-70b-sf) from q5 to f16 and transposition to PyTorch has demonstrated significant prose generation capabilities, a development worth noting for those interested in performance enhancements in language models.

- **API vs. Local Model Performance Debates Heat Up**: Users are sharing their experiences and skepticism about the discrepancies observed when utilizing Mistral models through the API versus a local setup, highlighting issues like response truncation and improper formatting in generated code which engineers working with API integration should be aware of.

- **Mistral Docs Under Scrutiny for Omissions**: The community has pointed out that official Mistral documentation lacks information on system prompts, with users emphasizing the need for inclusivity of such details and prompting a [PR discussion](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/discussions/115) aimed at addressing the issue.

- **RAG Integration: A Possible Solution to Fact-Hallucinations**: A discussion on leveraging Retrieval-Augmented Generation (RAG) to handle issues of hallucinations in smaller parameter models surfaced as an advanced strategy for engineers looking to improve factuality in model responses.

- **Targeting Homogeneity in Low-Resource Language Development**: Skepticism exists around the success of clustering new languages for continuous pretraining, reinforcing the notion that distinct language models may require separate efforts – a consideration vital for those working on the frontiers of multi-language model development.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **GGUF Model Woes and LM Studio Quirks**: `@petter5299` raised an issue with a **GGUF model** from HuggingFace not being recognized by **LM Studio**. This points to possible compatibility issues with certain architectures, a significant concern since `@artdiffuser` was informed that only GGUF models are intended to work with LM Studio. Users are advised to monitor catalog updates and report bugs as necessary; existing tutorials might be outdated or incorrect.

- **Hardware Demands for Local LLMs**: Users report that LM Studio is resource-intensive, with significant memory usage on advanced setups, including a **4090 GPU** and **128GB of RAM**. The community is also discussing the needs for building LLM PCs, highlighting **VRAM** as crucial and recommending GPUs with at least **24GB of VRAM**. Compatibility and performance issues with mixed generations of GPUs and how LLM performance scales across diverse hardware configurations remain topics of debate and investigation.

- **LM Studio Under the macOS Microscope**: macOS users, such as `@wisefarai`, experienced memory-related errors with LM Studio, potentially due to memory availability at the time of model loading. Such platform-specific issues highlight the variability of LM Studio's performance on different operating systems.

- **Training and Model Management Tactics**: The community is actively discussing strategies for **local LLM training**, the potential of **Quadro A6000 GPUs** for stable diffusion prompt writing, and the intricacies of memory management with model swapping. Users are exploring whether the latest iteration of LLM tools like **LLaMA v2** allow for customizable memory usage and how to efficiently run models that do not fit entirely in memory on systems like Windows.

- **LLM Studio's Quest for Compatibility**: Across discussions, the compatibility between LM Studio and various tools such as **Autogenstudio** and **CodeLLama** is a pertinent issue, assessing which models mesh well and which don't. The quest for compatibility also extends to prompts, as users seek JSON formatted prompt templates for improved functionality like the **Gorilla Open Function**.

- **Awaiting Beta Updates**: A lone message from `@mike_50363` wonders about the update status of the non-avx2 beta version to 2.12, illustrating the anticipation and reliance on the latest improvements in beta releases for optimal LLM development and experimentation.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Portrait Puzzles Persist in DALL-E**: Users, including `@wild.eva` and `@niko3757`, are grappling with **DALL-E's inclination towards landscape images**, which often results in sideways full-body portraits. With no clear solution, there's speculation about an awaited update to address this, while the lack of an **orientation parameter** currently hampers the desired vertical results.

- **Prompt Crafting Proves Crucial Yet Challenging**: In attempts to optimize image outcomes, conversation has emerged on whether prompt modifications can impact the orientation of generated images; however, **@darthgustav.** asserts that the model's intrinsic limitations override prompt alterations.

- **Interactivity Integrated in GPT Chatbots**: Discussions by `@luarstudios`, `@solbus`, and `@darthgustav.** focus on including interactive **feedback buttons** in GPT-designed chatbots, with advice given on using an **Interrogatory Format** to attach a menu of feedback responses.

- **Expectations High for DALL-E's Next Iteration**: The community, with contributors like `@niko3757`, is anticipative of a significant **DALL-E update**, hoping for improved functionality, particularly in image orientation—a point of current frustration among users.

- **Insight Sharing Across Discussions Enhances Community Support**: Cross-channel posting, as done by `@darthgustav.**, has been highlighted as a beneficial practice, as exemplified by shared prompt engineering techniques for a logo concept in **DALL-E Discussions**.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Phantom Spikes in Training Patterns**: Users reported observing spikes in their model's training patterns at every epoch. Efforts to mitigate **overtraining** by tweaking the **learning rate** and increasing dropout have been discussed, but challenges persist.

- **Considerations for AMD's GPU Stack**: There's active dialogue about server-grade hardware decisions, with a specific mention of **AMD's MI250 GPUs**. Concerns have been raised regarding the maturity of AMD's software stack, reflecting a skepticism when compared to Nvidia's solutions.

- **Axolotl Repo Maintenance**: A problematic commit (`da97285e63811c17ce1e92b2c32c26c9ed8e2d5d`) was identified that could be leading to overtraining, and a **pull request #1234** has been introduced to control the torch version during axolotl installation, preventing conflicts with the new torch-2.2.0 release.

- **Tackling CI and Dataset Configurations**: Issues with Continuous Integration breaking post torch-2.2.0 and challenges when configuring datasets for different tasks in axolotl have been addressed. Users shared solutions like specifying `data_files` paths and utilizing `TrainerCallback` for checkpoint uploads.

- **DPO Performance Hits a Snag**: Increased **VRAM usage** has been noted while running DPO, especially with **QLora on a 13b model**. Requests for detailed explanations and possibly optimizing VRAM consumption have been put forward.

- **Runpod Quirks and Tip-offs**: An odd issue with **empty folders** appearing on Runpod was noted without a known cause, and a hot tip regarding the availability of **H100 SXM units on the community cloud** was shared, highlighting the allure of opportunistic resource acquisition.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

**Training Headaches and Quantum Leaps**: A user faced a **loss flatline** during model training; a batch size reduction to `1` and the potential use of `EarlyStoppingCallback` were suggested solutions. Another proposed solution was **4bit quantization** to tackle training instability, which might help conserve VRAM albeit at some cost to model accuracy.

**Seeking Specialized Language Models**: There was an inquiry about **language models tailored to tech datasets** around Arduino, ESP32, and Raspberry Pi, suggesting a demand for LLMs with specialized knowledge.

**Tech Enthusiast's Project Spotlight**: Showcasing a range of projects from seeking feedback on a [thesis tweet](https://twitter.com/Vipitis/status/1752699776766988309), to offering access to a Magic: The Gathering model [space](https://huggingface.co/spaces/joshuasundance/mtg-coloridentity), as well as a custom pipeline solution for the `moondream1` model with a related [pull request](https://huggingface.co/vikhyatk/moondream1/discussions/6).

**Experimental Models Run Lean**: A **NeuralBeagle14-7b model** was successfully demonstrated on a local 8GB GPU, piquing the interest of those looking to optimize resource usage, which is key for maintainable AI solutions [here](https://github.com/joshuasundance-swca/llamacpp-langchain-neuralbeagle-demo).

**Scholarly Papers and AI Explorations**: A paper on **language model compression algorithms** [has been shared](https://arxiv.org/abs/2401.15347v1), discussing the balance between efficiency and accuracy in methods such as pruning, quantization, and distillation, which could be very pertinent in the ongoing dialogue about optimizing model performance.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **The Quest for Synthetic Datasets**: `@alyosha11` is searching for synthetic textbook datasets for a particular project and is directed towards efforts outlined in another channel, whilst `@finnilda` struggles to find missing audio files for MusicLM transformers training and seeks assistance after unfruitful GitHub inquiries.
- **Navigating the Tricky Waters of Content Filtering**: `@latke1422p` spearheads a conversation on the necessity of filtering images with underage subjects and discusses building safer AI content moderation tools utilizing datasets of trigger words.
- **The Discord Dilemma of Research Pings**: The use of '@everyone' pings on a research server has sparked a debate among users such as `@astropulse` and `@progamergov`, with a general lean towards allowing it in the context of the server's research-oriented nature.
- **An Unexpected Twist in VAE Training**: `@drhead` discovers that the kl-f8 VAE is improperly packing information, significantly impacting related models—this revelation prompts a dive into the associated [research paper](https://arxiv.org/abs/2309.16588) on transformer learning artifacts.
- **LLaVA-1.6 Takes the Lead and SPARC Ignites Interest**: The new LLaVA-1.6 model outperforms Gemini Pro as shared in its [release blog post](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/), and excitement bubbles up around SPARC, a new method for pretraining detailed multimodal representations, albeit without accessible code or models.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Custom GPT Curiosity Clarified**: Within the **general** channel, `@sweetpopcornsimon` queried about training a custom GPT similar to **ChatGPT**, while `@icelavaman` clarified that **Perplexity AI** doesn't offer chatbot services but features called **Collections** for organizing and sharing collaboration spaces.

- **Epub/PDF Reader with AI Integration Sought**: `@archient` inquired about an **epub/PDF reader** that supports **AI text manipulation** and can utilize **custom APIs**, sparking community interest in finding or developing such a tool.

- **Mystery Insight and YouTube Tutorial**: In the **sharing** channel, `@m1st3rg` teased insider knowledge regarding the future of **Google** through Perplexity, and `@redsolpl` shared a YouTube guide titled "How I Use **Perplexity AI** to Source Content Ideas for LinkedIn" at [this video](https://www.youtube.com/watch?v=iY4q7chZC1Y).

- **Support & Model Response Oddities Addressed**: `@angelo_1014` expressed difficulty in reaching **Perplexity** support and odd responses from the *codellama-70b-instruct* model were reported by `@bvfbarten.` in the **pplx-api** channel. The *codellama-34b-instruct* model, however, was confirmed to be robust.

- **API Uploads and Source Citations Discussed**: `@andreafonsmortigmail.com_6_28629` grappled with file uploads via the **chatbox interface**, and `@kid_tubsy` initiated a conversation about the need for **source URLs** in responses, especially using models with online capabilities like *-online*.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Searching Similarities Across the Multiverse**: Engineers are discussing strategies for fetching similarity across multiple Pinecone namespaces for storing meeting transcripts; however, some are facing issues with querying large JSON files via Langchain and ChromaDB, leading to only partial data responses.

- **Navigating the Langchain Embedding Maze**: There's ongoing exploration of implementing embeddings with OpenAI via Langchain and Chroma, including sharing code snippets, while some community members need support with TypeError issues after Langchain and Pinecone package updates.

- **AI Views on Wall Street Moves**: The quest for an AI that can analyze "real-time" stock market data and make informed decisions is a topic of curiosity, implying the interest in leveraging AI for financial market insights.

- **LangGraph Debuts Multi-Agent AI**: `@andysingal` introduced LangGraph in his Medium post, promising a futuristic tool designed for multi-agent AI systems collaborations, indicating a move towards more complex, interconnected AI workflows.

- **AI Tutorials Serve as Knowledge Beacons**: A YouTube tutorial highlighting the usage of the **new OpenAI Embeddings Model with LangChain** was shared, providing insights into the OpenAI's model updates and tools for AI application management.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Customizing Hybrid Search for RAG Systems**: **Hybrid search** within Retrieval-Augmented Generation (RAG) requires tuning for different question types, as discussed in a [Twitter thread by LlamaIndex](https://twitter.com/llama_index/status/1752748298392502521). Types mentioned include Web search queries and concept seeking, each necessitating distinct strategies.
  
- **Expanding RAG Knowledge with Multimodal Data**: A new resource was shared highlighting a YouTube video evaluation of multimodal RAG systems, including an introduction and evaluation techniques for such systems, along with the necessary [support documentation](https://support.google.com/youtube/answer/175292) and a [tweet announcing the video](https://twitter.com/llama_index/status/1752848239081214312).

- **Embedding Dumps and Cloud Queries Stoke Interest**: Assistance was sought by users for integrating vector embeddings with Opensearch databases and finding cloud-based storage solutions for KeywordIndex with massive data ingestion. Relevant resources like the [postgres.py vector store](https://github.com/run-llama/llama_index/blob/main/llama_index/vector_stores/postgres.py) and a [Redis Docstore+Index Store Demo](https://docs.llamaindex.ai/en/stable/examples/docstore/RedisDocstoreIndexStoreDemo.html) were referenced.

- **API Usage and Server Endpoints Clarified**: Queries regarding API choice differences, specifically assistant vs. completions APIs, and creating server REST endpoints sparked engagement with suggestions pointing towards `create-llama` from LlamaIndex's [documentation](https://docs.llamaindex.ai/en/stable/community/full_stack_projects.html#create-llama).

- **Discussion on RAG Context Size and Code**: Enquiries about the effect of RAG context size on retrieved results, and looking for tutorials on RAG over code, were notable. Langchain’s approach was cited as a reference, alongside the [Node Parser Modules](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules.html#codesplitter) and [RAG over code documentation](https://js.langchain.com/docs/use_cases/rag/code_understanding).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Decoder Tikz Hunt and a Leaky Ship**: An Eleuther member requested resources for finding a **transformer decoder tikz** illustration or an **arXiv paper** featuring one. Meanwhile, a tweet from Arthur Mensch revealed a watermarked old training model leak due to an "over-enthusiastic employee," causing a stir and eliciting reactions that compared the phrase to euphemistic sayings like "bless your heart." ([Tweet by Arthur Mensch](https://fxtwitter.com/arthurmensch/status/1752737462663684344))

- **When Scale Affects Performance**: Users in the research channel discussed findings from a paper on how **transformer model architecture** influences scaling properties, and how this knowledge has yet to hit the mainstream. Furthermore, the efficacy of pre-training on ImageNet was questioned, providing insights into the nuanced relationship between pre-training duration and performance across tasks. ([Scaling Properties of Transformer Models](https://arxiv.org/abs/2207.10551), [Pre-training Time Matters](https://arxiv.org/abs/2203.04668))

- **Rising from the Past: n-gram's New Potential**: A paper on an *infini-gram* model using $\\infty$-grams captured attention in the interpretability channel, showing that it can significantly improve language models like Llama. There were concerns about potential impacts on generalization, and curiosity was expressed regarding transformers' ability to memorize n-grams and how this could translate to automata. ([Infini-gram Language Model Paper](https://arxiv.org/abs/2401.17377), [Automata Study](https://arxiv.org/abs/2210.10749))

- **LM Evaluation Harness Polished and Probed**: In the lm-thunderdome channel, thanks were given for PyPI automation, and the release of the **Language Model Evaluation Harness 0.4.1** was announced, which included internal refactoring and features like Jinja2 for prompt design. Questions arose regarding the output of few-shot examples in logs and clarifications were sought for interpreting MMLU evaluation metrics, with a particular concern about a potential issue flagged in a GitHub gist. ([Eval Harness PyPI](https://pypi.org/project/lm-eval/0.4.1/), [GitHub Release](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.1), [Gist](https://gist.github.com/daniellepintz/c48c9e61a9a4798552b6ac22bc3a1959))

- **VQA Systems: Compute Costs and Model Preferences Questioned**: Queries were raised about the compute costs for training image encoders and LLMs for Visual Question Answering systems, highlighting the scarcity of reliable figures. There was also a search for consensus on whether encoder-decoder models remain the choice for text-to-image or text-and-image to text tasks, noting that training a model like llava takes approximately 8x24 A100 GPU hours.



---



## [CUDA MODE (Mark Saroufim)](https://discord.com/channels/1189498204333543425) Discord Summary

- **Python Prowess for CUDA Coders**: `@neuralution` and others emphasized the advantages of Python in CUDA development, covering topics like **occupancy**, **bottleneck identification**, and integration of PyTorch types with C++. A [PyTorch tutorial on Flash Attention 2 by Driss Guessous](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) was shared, and discussions highlighted the utility of NVIDIA's new [XQA kernel for multi-query attention](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md) and upcoming updates in the **Flash Attention 2** implementation in PyTorch 2.2.

- **C++ Healing via GitHub**: Issues following Jeremy's notebook were addressed by `@drisspg`, directing users to GitHub issue threads relating to a CUDA version mismatch ([CUDA 12.0 Issue](https://github.com/qwopqwop200/GPTQ-for-LLaMa/issues/7#issuecomment-1465246457)) and CUDA 12.1 template errors ([Cuda 12.1 Template Issue](https://github.com/pybind/pybind11/issues/4606#issuecomment-1498131493)). `@nshepperd` suggested CPU RAM's cost benefits compared to GPU RAM, and technical aspects of `typename` usage in C++ templates were clarified.

- **Selective Model Compilation Inquiry**: `@marvelousmit` asked if it's possible to compile specific parts of a model while excluding others, such as a custom operator called at runtime. They wondered about using something like `torch.compile(model)` for selective compilation.

- **CUDA Concerns and Debugging Discussions**: CUDA 11.8's compatibility was confirmed with a [link to NVIDIA's debugging documentation](https://docs.nvidia.com/nsight-visual-studio-edition/3.2/Content/Debugging_CUDA_Application.htm), and `@andreaskoepf` detailed their approach to kernel development by comparing PyTorch reference implementations with custom kernels. Debugger usage in VSCode was touched upon, revealing the complexity involved.

- **Block Size and Thread Coarsening Queries Unraveled**: `@lancerts` raised the issue of CUDA block sizes not matching `tile_width`, with `@vim410` cautioning about potential memory errors from misalignment of threads to work. The topic of thread coarsening sparked a discussion on the most effective dimensions for `threadsPerBlock` and `numBlocks`, underscoring the complexity of thread allocation optimization in CUDA programming.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Models in Identity Crisis**: Concerns were raised about the potential overlap between **Mistral** and **Llama** models, identified by `@sebastian.bodza` referencing a [tweet suggesting the similarities](https://twitter.com/lgrammel/status/1751921254406185308). Additionally, there's speculation from the community that **Miqu** may be a quantized version of **Mistral**, supported by signals from Twitter, including insights from [@sroecker](https://fxtwitter.com/sroecker/status/1752460995824349563?t=nZXA3oDFnDR6MoWUWwdYyA&s=19) and [@teortaxesTex](https://fxtwitter.com/teortaxesTex/status/1752673893276356608?t=4SMqTI_BCx8NTjmU3LoYXA&s=19).

- **Dataset Generosity Strikes the Right Chord**: The **Hermes 2 dataset** was made available by `@teknium`, posted on [Twitter](https://twitter.com/Teknium1/status/1752799124775374928), and praised by community members for its potential impact on AI research. The integration of lilac was positively acknowledged by `@devnull0`.

- **Speedy Mixtral Races Ahead**: The speed of **Mixtral**, achieving 500 tokens per second, was highlighted by `@bjoernp` with a link to [Groq's chat platform](https://chat.groq.com/). This revelation led to discussions on the effectiveness and implications of custom AI accelerators in computational performance.

- **Embedding Woes and Triumphs**: **Mistral 7B's embedding models** were called out by `@Nils_Reimers` on [Twitter](https://fxtwitter.com/Nils_Reimers/status/1752473576416911622?t=u6_C6owd2PRWz2knX2oM6A&s=19) for overfitting on MTEB and performing poorly on unrelated tasks. Conversely, Microsoft’s inventive approaches to generating text embeddings using synthetic data and simplifying training processes attracted attention, albeit with some skepticism, in a [research paper](https://arxiv.org/html/2401.00368v2) discussed by `@sebastian.bodza`.

- **The Plot Thickens... Or Does It?**: An inquiry from `@philipmay` about a specific plot for **DiscoLM_German_7b_v1** went without elaboration, potentially indicating a need for clarity or additional context to be addressed within the community.



---


## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **VFX Studios Eye AI Integration**: [A tweet by @venturetwins](https://x.com/venturetwins/status/175202239376) reveals major VFX studios, including one owned by Netflix, are now seeking professionals skilled in stable diffusion technologies. This new direction in hiring underscores the increasing importance of generative imaging and machine learning in revolutionizing storytelling, as evidenced by a [job listing from Eyeline Studios](https://jobs.lever.co/scanlinevfx/b6a54fd8-e4bb-4165-9b6d-ac67859cb0c0).

- **New Paradigms in AI Job Requirements Emerge**: The rapid evolution of AI technologies such as Stable Diffusion and Midjourney is humorously noted to potentially become standard demands in future job postings, reflecting a shift in employment standards within the tech landscape.

- **Efficiency Breakthroughs in LLM Training**: Insights from a [new paper by Quentin Anthony](https://x.com/QuentinAnthon15/status/1752393989813375119?s=20) propose a significant shift towards hardware-utilization optimization during transformer model training. This approach, focusing on viewing models through GPU kernel call sequences, aims to address prevalent inefficiencies in the training process.

- **Codeium's Leap to Series B Funding**: Celebrating Codeium's progress to Series B, a [complimentary tweet](https://twitter.com/_mohansolo/status/1752364915640447310) remarks on the team's achievement. This milestone highlights the growing optimism and projections around the company's future.

- **Hardware-Aware Design Boosts LLM Speed**: A new discovery highlighted by a [tweet from @BlancheMinerva](https://x.com/blancheminerva/status/1752416874481230105?s=46&t=90xQ8sGy63D2OtiaoGJuww) and detailed further in their paper on [arXiv:2401.14489](http://arxiv.org/abs/2401.14489), outlines a hardware-aware design tweak yielding a 20% throughput improvement for 2.7B parameter LLMs, previously overlooked by many due to adherence to GPT-3's architecture. 

- **Treasure Trove of AI and NLP Knowledge Unveiled**: For those keen on deepening their understanding of AI models and their historic and conceptual underpinnings, a [curated list shared by @ivanleomk](https://arc.net/folder/D0472A20-9C20-4D3F-B145-D2865C0A9FEE) brings together landmark resources, offering a comprehensive starting point for exploration in AI and NLP.


## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Clarifying "Prompt Investing"**: There was a discussion involving **prompt investing**, followed by a clarification from user **@jxnlco** referencing the correct term as **prompt injecting** instead.

- **On the Frontlines with Miqu-1 70B**: User **@jeffreyw128** showed interest in testing new models, with **@thebaghdaddy** suggesting the [Hugging Face's Miqu-1 70B model](https://huggingface.co/miqudev/miqu-1-70b). **@thebaghdaddy** advised specific **prompt formatting for Mistral** and mentioned limitations due to being "gpu poor."

- **Dissecting AI Performance and Functionality**:
    - Discussion around a [tweet from @nickadobos](https://twitter.com/nickadobos/status/1749837866300264529?s=46&t=6XxQ29Eas6j8_g5OIJcaEA) regarding AI performance occurred but the details were unspecified.
    - **@jeffreyw128** and **@joshcho_** discussed their **dissatisfaction with the performance** of AI concerning document understanding, with **@thebaghdaddy** suggesting this might explain AI's struggles with processing knowledge files containing images.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Open Source AI gets a Crypto Boost**: Filipvv put forward the idea of employing [crowdfunding techniques](https://www.cryodao.org/), like those used by [CryoDAO](https://www.cryodao.org/) and [MoonDAO](https://www.moondao.com/), to raise funds for open-source AI projects and highlighted the [Juicebox platform](https://juicebox.money/) as a potential facilitator for such endeavors. The proposed collective funding could support larger training runs on public platforms, contributing to the broader AI community's development resources.

- **New Training Data Goes Public**: Teknium released the **Hermes 2 dataset** and encouraged its use within the AI community. Interested engineers can access the dataset via this [tweet](https://twitter.com/Teknium1/status/1752799124775374928).

- **HelixNet: The Triad of AI Unveiled**: Migel Tissera introduced **HelixNet**, a cutting-edge architecture utilizing three **Mistral-7B LLMs**. AI enthusiasts and engineers can experiment with HelixNet through a [Jupyter Hub instance](https://jupyter.agentartificial.com/commune-8xa6000/user/forthepeople/lab), login `forthepeople` with password `getitdone`, as announced by yikesawjeez with more details on this [tweet](https://x.com/migtissera/status/1720567034315186588?s=20).



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **Diving into Hugging Face's API**: [@tonic_1](https://discord.com/channels/1144960932196401252/1144960932657758210/1202275940370235423) introduced the community to the [Transformers Agents API](https://huggingface.co/docs/transformers/main_classes/agent) from Hugging Face, noting its experimental nature and outlining the three agent types: **HfAgent**, **LocalAgent**, and **OpenAiAgent** for diverse model use cases.
- **Seeking Clarity on HFAgents**: `@hackgoofer` sought clarification about the previously discussed [HFAgents](https://huggingface.co/docs/transformers/main_classes/agent), showing a need for further explanation on the subject.
- **Community Contributions Welcomed**: `@tonic_1` expressed a keen interest to assist the community despite not having filled out a contribution form, indicating a positive community engagement.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Game On for AI-Enhanced Roblox Plugin**: `@magusartstudios` is in the process of developing a **Roblox AI agent Plugin**, which will integrate several advanced tools and features for the gaming platform.

- **Clarifying OpenAI's Freebies or Lack Thereof**: `@metaldragon01` corrected a common misconception that **OpenAI** is distributing free tokens for its open models, stressing that such an offering does not exist.

- **Datasets Galore—Hermes 2.5 & Nous-Hermes 2 Released**: `@teknium` announced the release of the **Open Hermes 2.5** and **Nous-Hermes 2 datasets**, available for the community to enhance state-of-the-art language models. The dataset, containing over 1 million examples, can be accessed on [HuggingFace](https://huggingface.co/datasets/teknium/OpenHermes-2.5).

- **Community Echoes and Lilac ML Integration**: Special thanks were extended to Discord members `<@1110466046500020325>`, `<@257999024458563585>`, `<@748528982034612226>`, `<@1124158608582647919>` for their contribution to the datasets. Moreover, the announcement included a nod to the collaboration with Lilac ML to make Hermes data analytically accessible via [HuggingFace Spaces](https://lilacai-lilac.hf.space/datasets#lilac/OpenHermes-2.5).



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Could DatasetteGPT Be Your New Best Friend?**: User `@discoureur` inquired if anyone has employed **DatasetteGPT** to aid with remembering configuration steps or assist in writing plugins for Datasette's documentation. No further discussion or responses were noted.



---


The **Ontocord (MDEL discord) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1202163395114041354) (1315 messages🔥🔥🔥): 

- **Xeon for GPU Servers**: `@reguile` mentioned opting for Xeon due to cost-effectiveness when used in GPU servers found on eBay.
- **Mysterious Llama3 Speculation**: Users discussed expectations around Llama3, with skepticism on whether it would exceed existing models like Mistral Medium.
- **Multimodal Model Potential**: Hints of LLaVA-1.6 being a formidable multimodal model exceeding Gemini Pro on benchmarks, introducing improved visual reasoning and OCR. A shared article detailed the enhancements.
- **Japanese Language and Culture**: `@righthandofdoom` commented on the detailed structure and pronunciation similarities between Japanese and their mother tongue, highlighting the nuanced nature of the language compared to Western counterparts.
- **MIQU Model Discussions**: The chat alluded to the MIQU model being akin to Mistral Medium and possibly a substantial leak, prompting debates on the model's origin, performance parity with GPT-4, and the consequences for the leaker.

**Links mentioned**:

- [alpindale/miquella-120b-gguf · Hugging Face](https://huggingface.co/alpindale/miquella-120b-gguf): no description found
- [InstantID - a Hugging Face Space by InstantX](https://huggingface.co/spaces/InstantX/InstantID): no description found
- [LLaVA-1.6: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/): LLaVA team presents LLaVA-1.6, with improved reasoning, OCR, and world knowledge. LLaVA-1.6 even exceeds Gemini Pro on several benchmarks.
- [google/switch-c-2048 · Hugging Face](https://huggingface.co/google/switch-c-2048): no description found
- [Mistral MODEL LEAK???  CEO Confirms!!!](https://www.youtube.com/watch?v=YdgLKx50-Y0&t=3s): &quot;An over-enthusiastic employee of one of our early access customers leaked a quantised (and watermarked) version of an old model we trained and distributed q...
- [Chat with Open Large Language Models](https://arena.lmsys.org/): no description found
- [Indiana Jones Hmm GIF - Indiana Jones Hmm Scratching - Discover &amp; Share GIFs](https://tenor.com/view/indiana-jones-hmm-scratching-gif-17930020): Click to view the GIF
- [google-research/lasagna_mt at master · google-research/google-research](https://github.com/google-research/google-research/tree/master/lasagna_mt): Google Research. Contribute to google-research/google-research development by creating an account on GitHub.
- [OpenRouter](https://openrouter.ai/): A router for LLMs and other AI models
- [Support for split GGUF checkpoints · Issue #5249 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/5249): Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...
- [teknium/OpenHermes-2.5 · Datasets at Hugging Face](https://huggingface.co/datasets/teknium/OpenHermes-2.5): no description found
- [GitHub - arcee-ai/mergekit: Tools for merging pretrained large language models.](https://github.com/cg123/mergekit): Tools for merging pretrained large language models. - GitHub - arcee-ai/mergekit: Tools for merging pretrained large language models.
- [no title found](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/design-multimodal-prompts): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1202162087200309258) (885 messages🔥🔥🔥): 

- **Discussing Miqu's Performance Beyond Leaks**: `@goldkoron` and others discuss the performance of **Miqu**, a leaked **70B** model. `@turboderp_` clarifies it's architecturally the same as **LLaMA 2 70B**, and it's been continued by **Mistral** as an early version of their project.

- **Exploring Morally Ambiguous Training Data**: `@heralax` seeks suggestions for dark or morally questionable text for a few-shot example to teach a model to write longer and darker responses. `@the_ride_never_ends` recommends Edgar Allan Poe's short stories on Wikipedia, and `@kquant` mentions the dark original stories by the Grimm Brothers.

- **Fitting Models Within GPU VRAM Constraints**: `@kaltcit` reports that with 48GB VRAM, 4.25bpw can fit but generating any content results in OOM (Out Of Memory). Additionally, there's a discussion about using lower learning rates for finetuning and the cost of training these large models, with `@c.gato` and `@giftedgummybee` sharing their training strategies.

- **Fine-Tuning Strategies and Costs**: `@undi` shares that they have finetuned a **Miqu** model referred to as **MiquMaid+Euryale** using parts of their dataset. They also express the high cost and ambition behind their single-day finetuning attempt, which used an **A100 SM**.

- **Broader Discussion on Large Language Model Deployment**: The channel's users touch on broader topics, such as the potential for further fine-tuning 70B models, the general costliness of training and fine-tuning, possible collaborations, and the risks of alignment and ethics in language models.

**Links mentioned**:

- [Category:Short stories by Edgar Allan Poe - Wikipedia](https://en.wikipedia.org/wiki/Category:Short_stories_by_Edgar_Allan_Poe): no description found
- [Self-hosted AI models | docs.ST.app](https://docs.sillytavern.app/usage/local-llm-guide/how-to-use-a-self-hosted-model/): This guide aims to help you get set up using SillyTavern with a local AI running on your PC (we'll start using the proper terminology from now on and...
- [152334H/miqu-1-70b-sf · Hugging Face](https://huggingface.co/152334H/miqu-1-70b-sf): no description found
- [NeverSleep/MiquMaid-v1-70B-GGUF · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v1-70B-GGUF): no description found
- [miqudev/miqu-1-70b · Hugging Face](https://huggingface.co/miqudev/miqu-1-70b): no description found
- [Money Wallet GIF - Money Wallet Broke - Discover &amp; Share GIFs](https://tenor.com/view/money-wallet-broke-gif-7855913): Click to view the GIF
- [NobodyExistsOnTheInternet/miqu-limarp-70b · Hugging Face](https://huggingface.co/NobodyExistsOnTheInternet/miqu-limarp-70b/): no description found
- [sokusha/aicg · Datasets at Hugging Face](https://huggingface.co/datasets/sokusha/aicg): no description found
- [Ayumi Benchmark ERPv4 Chat Logs](http://ayumi.m8geil.de/erp4_chatlogs/index.html?S=iq4_0#!/index>): no description found
- [NobodyExistsOnTheInternet/ShareGPTsillyJson · Datasets at Hugging Face](https://huggingface.co/datasets/NobodyExistsOnTheInternet/ShareGPTsillyJson): no description found
- [GitHub - bjj/exllamav2-openai-server: An OpenAI API compatible LLM inference server based on ExLlamaV2.](https://github.com/bjj/exllamav2-openai-server): An OpenAI API compatible LLM inference server based on ExLlamaV2. - GitHub - bjj/exllamav2-openai-server: An OpenAI API compatible LLM inference server based on ExLlamaV2.
- [GitHub - epolewski/EricLLM: A fast batching API to serve LLM models](https://github.com/epolewski/EricLLM): A fast batching API to serve LLM models. Contribute to epolewski/EricLLM development by creating an account on GitHub.
- [Mistral CEO confirms &#x27;leak&#x27; of new open source AI model nearing GPT4 performance | Hacker News](https://news.ycombinator.com/item?id=39208213): no description found

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1202323435045261342) (30 messages🔥): 

- **Fine-tuning Epoch Clarification**: `@dirtytigerx` responded to `@bishwa3819` pointing out that the loss graph posted showed less than a single epoch and inquired about seeing a graph for a full epoch.
- **DataSet Configuration Puzzle**: `@lordofthegoons` sought assistance for setting up dataset configuration on unsloth for the *sharegpt* format, mentioning difficulty finding documentation for formats other than *alpaca*.
- **Troubleshooting TinyLlama Fine-tuning**: `@chovii` struggled with fine-tuning TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ and shared a specific `ValueError` related to unsupported modules when attempting to add adapters based on the PEFT library. `@dirtytigerx` indicated that AWQ is not yet supported in PEFT, citing a draft PR that's awaiting an upcoming AutoAWQ release with a [Github pull request link](https://github.com/huggingface/peft/pull/1399).
- **Counting Tokens for Training**: `@dirtytigerx` provided `@arcontex` a simple script to count the number of tokens in a file for training, suggesting the use of `AutoTokenizer.from_pretrained(model_name)` to easily grab the tokenizer for most models.
- **Experiment Tracking Tools Discussed**: `@flashmanbahadur` queried about the use of experiment tracking tools like wandb and mlflow for local runs. `@dirtytigerx` expressed the usefulness of both wandb and comet, especially for longer training runs, and discussed the broader capabilities of mlflow.

**Links mentioned**:

- [Load adapters with 🤗 PEFT](https://huggingface.co/docs/transformers/peft): no description found
- [FEAT: add awq suppot in PEFT by younesbelkada · Pull Request #1399 · huggingface/peft](https://github.com/huggingface/peft/pull/1399): Original PR: casper-hansen/AutoAWQ#220 TODO:   Add the fix in transformers  Wait for the next awq release  Empty commit with @s4rduk4r as a co-author  add autoawq in the docker image  Figure out ho...

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1202264528767942716) (15 messages🔥): 

- **In Search of Less Censored Intelligence**: `@lordofthegoons` inquired about uncensored, intelligent 34b models, where `@kquant` suggested looking at models like [Nous-Yi](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B) on Hugging Face.
- **Smaug 34b Takes Flight Without Mergers**: `@kquant` shares information about the [Smaug-34B-v0.1](https://huggingface.co/abacusai/Smaug-34B-v0.1) model, a fine-tuned version of "bagel" with impressive benchmark results and no model mergers involved.
- **Shrinking Goliath**: `@lordofthegoons` expressed interest in an experiment to downsize a 34b model in hopes it would outperform a 10.7b model, but later reported that the attempt did not yield a usable model.
- **Prompts to Nudge Model Behavior**: `@kquant` discusses using prompts to influence model outputs, citing methods like changing the response style or trigger specific behaviors like simulated anger or polite confirmations.
- **Harmony Through Merging Models**: `@kquant` points to the [Harmony-4x7B-bf16](https://huggingface.co/ConvexAI/Harmony-4x7B-bf16) as an example of a successful model merge that exceeded expectations, naming itself after the process.

**Links mentioned**:

- [ConvexAI/Harmony-4x7B-bf16 · Hugging Face](https://huggingface.co/ConvexAI/Harmony-4x7B-bf16): no description found
- [abacusai/Smaug-34B-v0.1 · Hugging Face](https://huggingface.co/abacusai/Smaug-34B-v0.1): no description found
- [one-man-army/UNA-34Beagles-32K-bf16-v1 · Hugging Face](https://huggingface.co/one-man-army/UNA-34Beagles-32K-bf16-v1): no description found

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1202306276433088552) (2 messages): 

- **The Forgotten Art of File Management**: `@spottyluck` highlighted a growing problem where many people lack understanding of a **file system hierarchy** or how to organize files, resulting in a "big pile of search/scroll".
- **Generational File System Bewilderment**: `@spottyluck` shared an article from [The Verge](https://www.theverge.com/22684730/students-file-folder-directory-structure-education-gen-z) discussing how students are increasingly unfamiliar with file systems, referencing a case where astrophysicist Catherine Garland noticed her students couldn't locate their project files or comprehend the concept of file directories.

**Links mentioned**:

[Students who grew up with search engines might change STEM education forever](https://www.theverge.com/22684730/students-file-folder-directory-structure-education-gen-z): Professors are struggling to teach Gen Z

  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1202428115998085140) (6 messages): 

- **Inquiry about Form Submission**: `@manojbh` asked about whether a form has been filled out, but no further context or details were provided.
- **Questioning the Uniqueness of RMT**: `@hexani` raised a question about how the discussed technology truly differs from **RMT** and its significance.
- **Skepticism on Context Length Impact**: `@hexani` expressed doubt about the technology representing a significant shift towards managing longer context, citing evaluations in the paper.
- **Request for Clarification**: `@hexani` asked for someone to explain more about the technology in question and appreciated any help in advance.
- **Comparison with Existing Architectures**: `@hexani` pointed out similarities between the current topic of discussion and existing architectures, likening it to an adapter for explicit memory tokens.
  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1202281402989805648) (13 messages🔥): 

- **In Search of Machine Learning Guidance**: `@DES7INY7` and `@lorenzoroxyolo` are seeking advice on how to get started with their **machine learning journey** and looking for guidance and paths to follow.
- **Possible Discord Improvement with @-ing**: `@murchiston` discusses that @-ing could be improved as a feature on Discord, implying that **better routing** or integration within tools like **lmstudio** might make the feature more reliable.
- **Fast.ai Course Sighting**: `@afterhoursbilly` mentions that he has seen `<@387972437901312000>` on the **fast.ai Discord**, hinting at the user's engagement with the course.
- **The Maneuver-Language Idea**: `@manojbh` brings up an idea about tokenizing driving behaviors for **self-driving technology**, proposing a concept similar to **lane-language**.
- **Dataset Schema Inconsistency**: `@pradeep1148` points out that the dataset announced by `<@387972437901312000>` appears to have a schema inconsistency, with different examples having distinct formats. A link to the dataset on Hugging Face is provided: [OpenHermes-2.5 dataset](https://huggingface.co/datasets/teknium/OpenHermes-2.5).

**Links mentioned**:

- [Hellinheavns GIF - Hellinheavns - Discover &amp; Share GIFs](https://tenor.com/view/hellinheavns-gif-23278790): Click to view the GIF
- [Blackcat GIF - Blackcat - Discover &amp; Share GIFs](https://tenor.com/view/blackcat-gif-8560392459073397502): Click to view the GIF
- [Scary Weird GIF - Scary Weird Close Up - Discover &amp; Share GIFs](https://tenor.com/view/scary-weird-close-up-gif-17071971): Click to view the GIF

  

---


### Nous Research AI ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/1202325748644581467) (2 messages): 

- **Clarification on ACC_NORM for AGIEval**: `@euclaise` inquired if `@387972437901312000` utilizes `acc_norm` for AGIEval evaluations. `@teknium` confirmed that where an `acc_norm` exists, it is **always used**.
  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1202267069425389680) (77 messages🔥🔥): 

- **Bittensor and Synthetic Data**: `@richardblythman` questioned how Bittensor addresses gaming benchmarks for AI models, suggesting that Hugging Face could also generate synthetic data on the fly. `@teknium` clarified the costs involved and the subsidization by Bittensor's network inflation, while `@euclaise` noted Hugging Face's ability to run expensive operations.
- **Incentivization in Model Training**: `@teknium` described the importance of incentive systems within Bittensor's network to drive model improvement and active competition, while `@richardblythman` shared skepticism about the sustainability of Bittensor's model due to high costs.
- **Understanding Bittensor's Network**: Throughout the discussion, there was significant interest in how Bittensor's decentralized network operates and incentives are structured, with multiple users including `@.benxh` and `@teknium` discussing aspects of deployment and the incentives for maintaining active and competitive models.
- **Potential of Bittensor to Produce Useful Models**: While `@richardblythman` expressed doubt about the network's efficiency, `@teknium` remained optimistic about its potential to accelerate the development of open source models, mentioning that rigorous training attempts started only a month or two prior.
- **Collaborative Testing of AI Models**: `@yikesawjeez` invited community members to assist in testing a new AI architecture, HelixNet, and provided shared Jupyter Notebook access for the task. The invitation was answered by users like `@manojbh`, encouraging reproduction on different setups for better validation of the model's performance.

**Links mentioned**:

- [Mistral CEO confirms &#8216;leak&#8217; of new open source AI model nearing GPT-4 performance](https://venturebeat.com/ai/mistral-ceo-confirms-leak-of-new-open-source-ai-model-nearing-gpt-4-performance/): An anonymous user on 4chan posted a link to the miqu-1-70b files on 4chan. The open source model approaches GPT-4 performance.
- [argilla/CapybaraHermes-2.5-Mistral-7B · Hugging Face](https://huggingface.co/argilla/CapybaraHermes-2.5-Mistral-7B): no description found
- [LLaVA-1.6: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/): LLaVA team presents LLaVA-1.6, with improved reasoning, OCR, and world knowledge. LLaVA-1.6 even exceeds Gemini Pro on several benchmarks.
- [NobodyExistsOnTheInternet/miqu-limarp-70b · Hugging Face](https://huggingface.co/NobodyExistsOnTheInternet/miqu-limarp-70b): no description found
- [Tweet from Ram (@ram_chandalada)](https://x.com/ram_chandalada/status/1752727535765295225?s=46): Scoring popular datasets with &#34;Self-Alignment with Instruction Backtranslation&#34; prompt  Datasets Scored 1️⃣ dolphin @erhartford  - Only GPT-4 responses 2️⃣ Capybara @ldjconfirmed  3️⃣ ultracha...
- [Tweet from Migel Tissera (@migtissera)](https://x.com/migtissera/status/1720567034315186588?s=20): It&#39;s been a big week for Open Source AI, and here&#39;s one more to cap the week off!  Introducing HelixNet.  HelixNet is a novel Deep Learning architecture consisting of 3 x Mistral-7B LLMs. It h...
- [Tweet from yikes (@yikesawjeez)](https://x.com/yikesawjeez/status/1752808327728537682?s=20): yknow what fk it https://jupyter.agentartificial.com/commune-8xa6000/user/forthepeople/lab user: forthepeople pw: getitdone  will start downloading the model now, hop in and save notebooks to the ./wo...
- [The Curious Case of Nonverbal Abstract Reasoning with Multi-Modal Large Language Models](https://arxiv.org/abs/2401.12117): While large language models (LLMs) are still being adopted to new domains and utilized in novel applications, we are experiencing an influx of the new generation of foundation models, namely multi-mod...
- [GroqChat](https://chat.groq.com): no description found
- [Scaling Laws for Forgetting When Fine-Tuning Large Language Models](https://arxiv.org/abs/2401.05605): We study and quantify the problem of forgetting when fine-tuning pre-trained large language models (LLMs) on a downstream task. We find that parameter-efficient fine-tuning (PEFT) strategies, such as ...
- [Paper page - Learning Universal Predictors](https://huggingface.co/papers/2401.14953): no description found
- [GitHub - TryForefront/tuna](https://github.com/TryForefront/tuna): Contribute to TryForefront/tuna development by creating an account on GitHub.

  

---


### Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1202360039688646747) (1 messages): 

- **Open Hermes 2.5 Dataset Released**: `@teknium` announced the public release of the dataset used for Open Hermes 2.5 and Nous-Hermes 2, excited to share over 1M examples curated and generated from open-source ecosystems. The dataset can be found on [HuggingFace](https://huggingface.co/datasets/teknium/OpenHermes-2.5).
- **Dataset Contributions Acknowledged**: `@teknium` has credited every data source within the dataset card except for one that's no longer available, thanking the contributed authors within the Nous Research AI Discord.
- **Collaboration with Lilac ML**: `@teknium` worked with `@1097578300697759894` and Lilac ML to feature Hermes on their HuggingFace Spaces, assisting in the analysis and filtering of the dataset. Explore it on [Lilac AI](https://lilacai-lilac.hf.space/datasets#lilac/OpenHermes-2.5).
- **Tweet about Open Hermes Dataset**: A related [Twitter post](https://twitter.com/Teknium1/status/1752799124775374928) by `@teknium` celebrates the release of the dataset and invites followers to see what can be created from it.

**Links mentioned**:

- [teknium/OpenHermes-2.5 · Datasets at Hugging Face](https://huggingface.co/datasets/teknium/OpenHermes-2.5): no description found
- [no title found](https://lilacai-lilac.hf.space/datasets#lilac/OpenHermes-2.5): no description found

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1202166176281792552) (673 messages🔥🔥🔥): 

- **MIQU Mystery Resolved**: `@teknium` clarifies that **MIQU** *is* **Mistral Medium**, not an earlier version. The Nous Research's co-founder's tweet could potentially confirm this [here](https://twitter.com/arthurmensch/status/1752737462663684344).
- **The Power of Mergekit**: Users like `@datarevised` experimented with **120 billion parameter models** by merging MIQU with itself, and others are creating various combinations like **Miquella 120B**.
- **OpenHermes 2.5 Dataset Discussion**: The dataset discussions raised points on dataset creation and storage, with `@teknium` mentioning tools like [Lilac](https://www.lilacml.com/) for data curation and exploration.
- **Crowdfunding for AI with AI Grant**: `@cristi00` shares information about AI Grant – an accelerator for AI startups offering substantial funding and Azure credits. Applications are open until the stated deadline.
- **Qwen 0.5B Model Anticipation**: `@qnguyen3` expresses excitement about the imminent release of a new fiery model named **Qwen**, though specific details were not given.

**Links mentioned**:

- [NobodyExistsOnTheInternet/code-llama-70b-python-instruct · Hugging Face](https://huggingface.co/NobodyExistsOnTheInternet/code-llama-70b-python-instruct): no description found
- [Boy Kisser Boykisser GIF - Boy kisser Boykisser Boy kisser type - Discover &amp; Share GIFs](https://tenor.com/view/boy-kisser-boykisser-boy-kisser-type-type-typing-gif-4348094406361571449): Click to view the GIF
- [Guts Berserk Guts GIF - Guts Berserk Guts American Psycho - Discover &amp; Share GIFs](https://tenor.com/view/guts-berserk-guts-american-psycho-patrick-bateman-sigma-gif-27643225): Click to view the GIF
- [Meta Debuts Code Llama 70B: A Powerful Code Generation AI Model](https://www.forbes.com/sites/janakirammsv/2024/01/30/meta-debuts-code-llama-70b-a-powerful-code-generation-ai-model/?sh=b28beb471f34): With Code Llama 70B, enterprises have a choice to host a capable code generation model in their private environment. This gives them control and confidence in protecting their intellectual property.
- [Tweet from Stella Biderman (@BlancheMinerva)](https://x.com/BlancheMinerva/status/1752416874481230105?s=20): Are you missing a 20% speed-up for your 2.7B LLMs due to copying GPT-3? I was for three years.  Find out why and how to design your models in an hardware-aware fashion in my latest paper, closing the ...
- [AI Grant](http://aigrant.com/): It's time for AI-native products!
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/FHYA6FR7): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Reimagine](https://www.eluna.ai/reimagine): Covering the rapidly moving world of AI.
- [Cat Cats GIF - Cat Cats Explosion - Discover &amp; Share GIFs](https://tenor.com/view/cat-cats-explosion-explodes-cat-explodes-gif-10311420692458175149): Click to view the GIF
- [152334H/miqu-1-70b-sf · Hugging Face](https://huggingface.co/152334H/miqu-1-70b-sf): no description found
- [argilla/CapybaraHermes-2.5-Mistral-7B · Hugging Face](https://huggingface.co/argilla/CapybaraHermes-2.5-Mistral-7B): no description found
- [Wow Surprised Face GIF - Wow Surprised face Little boss - Discover &amp; Share GIFs](https://tenor.com/view/wow-surprised-face-little-boss-gif-alisa-gif-16133616529628653686): Click to view the GIF
- [euclaise/Memphis-scribe-3B-alpha · Hugging Face](https://huggingface.co/euclaise/Memphis-scribe-3B-alpha): no description found
- [Bornskywalker Dap Me Up GIF - Bornskywalker Dap Me Up Woody - Discover &amp; Share GIFs](https://tenor.com/view/bornskywalker-dap-me-up-woody-woody-handshake-woody-toy-story-gif-26021440): Click to view the GIF
- [ycros/miqu-lzlv · Hugging Face](https://huggingface.co/ycros/miqu-lzlv): no description found
- [WizardLM/WizardLM-70B-V1.0 · Hugging Face](https://huggingface.co/WizardLM/WizardLM-70B-V1.0): no description found
- [alpindale/miquella-120b · Hugging Face](https://huggingface.co/alpindale/miquella-120b): no description found
- [Hatsune Miku - Wikipedia](https://en.wikipedia.org/wiki/Hatsune_Miku): no description found
- [Dancing Daniel Keem GIF - Dancing Daniel Keem Keemstar - Discover &amp; Share GIFs](https://tenor.com/view/dancing-daniel-keem-keemstar-feeling-it-dancer-gif-16902720): Click to view the GIF
- [xVal: A Continuous Number Encoding for Large Language Models](https://arxiv.org/abs/2310.02989): Large Language Models have not yet been broadly adapted for the analysis of scientific datasets due in part to the unique difficulties of tokenizing numbers. We propose xVal, a numerical encoding sche...
- [Catspin GIF - Catspin - Discover &amp; Share GIFs](https://tenor.com/view/catspin-gif-12303895773004295802): Click to view the GIF
- [Itsover Wojack GIF - ITSOVER WOJACK - Discover &amp; Share GIFs](https://tenor.com/view/itsover-wojack-gif-4367840179675491690): Click to view the GIF
- [Tweet from Nat Friedman (@natfriedman)](https://x.com/natfriedman/status/1752831181677305952?s=20): Applications are open for batch 3 of http://aigrant.com for pre-seed and seed-stage companies building AI products! Deadline is Feb 16.  As an experiment, this batch we are offering the option of eith...
- [NousResearch/Nous-Hermes-2-Vision-Alpha · Discussions](https://huggingface.co/NousResearch/Nous-Hermes-2-Vision-Alpha/discussions): no description found
- [dataautogpt3 (alexander izquierdo)](https://huggingface.co/dataautogpt3): no description found
- [NobodyExistsOnTheInternet (Nobody.png)](https://huggingface.co/NobodyExistsOnTheInternet): no description found
- [Tweet from Eric Hallahan (@EricHallahan)](https://x.com/EricHallahan/status/1752430903412822487?s=20): @QuentinAnthon15 Darn, I would have loved to participate if I were allowed to. I&#39;m sure it&#39;s a great paper regardless with an author list like that!
- [GitHub - qnguyen3/hermes-llava](https://github.com/qnguyen3/hermes-llava): Contribute to qnguyen3/hermes-llava development by creating an account on GitHub.
- [GitHub - SafeAILab/EAGLE: EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation](https://github.com/SafeAILab/EAGLE/): EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation - GitHub - SafeAILab/EAGLE: EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation
- [GitHub - epfLLM/meditron: Meditron is a suite of open-source medical Large Language Models (LLMs).](https://github.com/epfLLM/meditron): Meditron is a suite of open-source medical Large Language Models (LLMs). - GitHub - epfLLM/meditron: Meditron is a suite of open-source medical Large Language Models (LLMs).
- [There’s something going on with AI startups in France | TechCrunch](https://techcrunch.com/2023/11/09/theres-something-going-on-with-ai-startups-in-france/?guccounter=1): Artificial intelligence, just like in the U.S., has quickly become a buzzy vertical within the French tech industry.
- [Lilac - Better data, better AI](https://www.lilacml.com/): Lilac enables data and AI practitioners improve their products by improving their data.

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1202201419906814002) (32 messages🔥): 

- **Style Transfer or Overfitting?**: `@manojbh` debated about LLMs having responses overly influenced by their training on a particular style like Mistral; pointed out that similar end-layer patterns cause style mimicry in outputs. This was in context to misleading output reasoning and a linked tweet by `@teortaxesTex`, regarding Miqu’s mistake in RU translation with oddly similar phrasing.
- **Quantized Models and Memory Discussio**n: Discussions by `@giftedgummybee` and `@manojbh` centered around quantized models forgetting information and the increased potential for errors with lower quantization levels.
- **AI Stress Test Speculation**: A theory by `@manojbh` on whether LLMs can fumble under stress like humans led to a suggestion by `@_3sphere` to test the model's "panic" behavior in response to confusing prompts.
- **Watermarking within Mistral**: Conversation with `@everyoneisgross`, `@.benxh`, and `.ben.com` delved into the possibility of watermarking LLMs, hypothesizing it could involve distinct Q&A pairs or prompts that generate specific responses as identifiers after quantization.
- **Evaluating Nous-Hermes and Mixtral Performance**: `@.benxh` affirmed a significant performance difference between Mistral Medium and an unspecified model capacity, indicating close performance within the margin of error. `@manojbh` agreed, highlighting the importance of even slight differences in statistical metrics.

**Links mentioned**:

- [Tweet from Teortaxes▶️ (@teortaxesTex)](https://x.com/teortaxesTex/status/1752444487610093942?s=20): Miqu makes a mistake in RU: the bolt falls out but the thimble stays. Yet note startlingly similar phrasing in translation! I&#39;m running Q4KM though, and some of it may be attributable to sampling ...
- [GitHub - LeonEricsson/llmjudge: Exploring limitations of LLM-as-a-judge](https://github.com/LeonEricsson/llmjudge): Exploring limitations of LLM-as-a-judge. Contribute to LeonEricsson/llmjudge development by creating an account on GitHub.

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1202181543351816242) (363 messages🔥🔥): 

- **Welcoming New Members**: Users `@admin01234` and `@mrdragonfox` exchanged greetings, indicating that new users are discovering the Mistral Discord channel.

- **Dequantization Success for miqu-1-70b**: User `@i_am_dom` discussed the successful dequantization of [miqu-1-70b](https://huggingface.co/152334H/miqu-1-70b-sf) from q5 to f16, transposed to PyTorch. Usage instructions and results were shared, showcasing the model's prose generation capabilities.

- **Internship Asks and General Career Opportunities**: User `@deldrel` asked about internship opportunities at Mistral. `@mrdragonfox` advised them to send their details even if no official listings are posted.

- **Token Generation Rates and Inference Performance**: User `@i_am_dom` provided token/s generation rates for the official Mistral API, which `@donjuan5050` said might not meet their use case. Meanwhile, `@mrdragonfox` shared that locally hosting Mistral on certain hardware configurations could significantly increase throughput. 

- **Miqutized Model and Mistral's Response**: The authenticity of miqu-1-70b was discussed, including a link to the [Twitter statement](https://twitter.com/arthurmensch/status/1752734898476007821) by Mistral AI's CEO confirming it as an early version of their model. Users `@shirman`, `@i_am_dom`, and `@dillfrescott` speculated about the relationship between the Miqutized model and Mistral's models.

- **Model Hosting and Usage Costs**: Conversation with `@mrdragonfox` regarding the benefits and costs of hosting LLMs locally vs. using an API. The discussion delved into hardware requirements and the cost-effectiveness of API usage for different scales of deployment.

**Links mentioned**:

- [Join the Mistral AI Discord Server!](https://discord.gg/9BaYw4qR?event=1196389318160306226): Check out the Mistral AI community on Discord - hang out with 11372 other members and enjoy free voice and text chat.
- [152334H/miqu-1-70b-sf · Hugging Face](https://huggingface.co/152334H/miqu-1-70b-sf): no description found
- [TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ): no description found
- [Streisand effect - Wikipedia](https://en.wikipedia.org/wiki/Streisand_effect): no description found
- [NeverSleep/MiquMaid-v1-70B · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v1-70B): no description found
- [NeverSleep/MiquMaid-v1-70B · VERY impressive!](https://huggingface.co/NeverSleep/MiquMaid-v1-70B/discussions/1#65bae2c53e109e72597e5506): no description found

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1202168310788276235) (17 messages🔥): 

- **Mistral Aligns with Alpaca Format**: `@sa_code` mentioned that Mistral small/med can be used with alpaca prompt format to supply a system prompt; however, they also noted the alpaca format is incompatible with markdown.
- **Official Docs Lack System Prompt Info**: `@sa_code` pointed to the lack of documentation for system prompts in Mistral's official docs and requested this inclusion.
- **Office Hours for the Rescue**: In response to `@sa_code`'s query, `@mrdragonfox` suggested asking questions in the next office hours session for clarifications.
- **PR Opened for Chat Template**: `@sa_code` indicated they opened a PR on Mistral's Hugging Face page to address an issue regarding chat template documentation ([PR discussion link](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/discussions/115)).
- **Mistral Embedding Token Limit Clarified**: `@mrdragonfox` and `@akshay_1` responded to `@danouchka_24704`'s questions, confirming that Mistral-embed produces 1024 dimensions vectors and has a maximum token input chunk of 8k, although 512 tokens are generally preferred.

**Links mentioned**:

- [Open-weight models | Mistral AI Large Language Models](https://docs.mistral.ai/models/#chat-template): We open-source both pre-trained models and fine-tuned models. These models are not tuned for safety as we want to empower users to test and refine moderation based on their use cases. For safer models...
- [mistralai/Mixtral-8x7B-Instruct-v0.1 · Update chat_template to include system prompt](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/discussions/115): no description found

  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1202191044368662528) (1 messages): 

- **JSON Output with Mistral Medium Subscription**: User `@subham5089` inquired about the best method to always receive **JSON output** using the **Mistral Medium subscription**. There was no response or further discussion on this topic within the provided message history.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1202194167196028939) (33 messages🔥): 

- **Inquiry on Continuous Pretraining for Low-Resource Languages**: `@quicksort` asked for references regarding continuous pretraining of **Mistral** on low-resource languages, but `@mrdragonfox` indicated a lack of such resources stating *nothing comes close to instruct yet*.
- **Pioneering Multi-Language LoRA Fine-tuning on Mistral**: `@yashkhare_` is working on continual pretraining of **Mistral 7b** with **LoRA** for languages like Vietnamese, Indonesian, and Filipino, questioning the viability of separate and merged language-specific LoRA weights.
- **Challenges of Language Clustering and Continuous Pretraining**: `@mrdragonfox` cast doubt on the success of a style transfer-based approach for clustering new languages and stressed that distinct languages typically require separate pretraining.
- **Using RAG to Address Hallucinations in Fact Finetuning**: In response to `@pluckedout`'s question about lower parameter models memorizing facts, `@kecol` and `@mrdragonfox` discussed the importance and complexity of integrating **Retrieval-Augmented Generation (RAG)** for context-informed responses and minimizing hallucinations.
- **Hermes 2 Dataset Release Announcement**: `@teknium` shared a link to the release of their **Hermes 2** dataset on [Twitter](https://twitter.com/Teknium1/status/1752799124775374928), inviting others to explore its potential value.
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1202257497071550574) (8 messages🔥): 

- **Confusion on Model Downloads**: `@ashtagrossedaronne` enquired about downloading a model to have it complete homework, but was initially unsure of which model to use.
- **Mistaken Link Adventure**: In response, `@ethux` attempted to assist with a download link, but first provided the wrong URL and then corrected it, though the correct link was not shared in the messages given.
- **Contribution Update**: `@carloszela` announced the submission of a Pull Request (PR) and is awaiting review, signaling their contribution to a project.
- **API Quirks with Mistral Model**: `@miscend` raised a concern about the performance disparities between the Mistral small model when using the API versus running it locally, specifically mentioning truncation of responses and additional backslashes in code output.

**Links mentioned**:

[👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/): Find, download, and experiment with local LLMs

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1202192078600151101) (123 messages🔥🔥): 

- **HuggingFace Model Confusion**: `@petter5299` inquired about why a GGUF model downloaded is not recognizable by LM Studio. `@fabguy` asked for a HuggingFace link for clarification and mentioned the unfamiliarity with the mentioned architecture.
- **LM Studio & macOS Memory Issues**: `@wisefarai` encountered errors suggesting insufficient memory while trying to load a quantized model on LM Studio using a Mac, with `@yagilb` suggesting it might be a lack of available memory at the time of the issue.
- **Local Models and Internet Claims**: `@n8programs` and others discussed the need for proof regarding claims that local models behave differently with the internet on/off. They called for network traces to known OpenAI addresses to substantiate such claims.
- **AMD GPU Support Inquiry**: `@ellric_` asked if there is any hope for AMD GPU support in LM Studio. `@yagilb` confirmed the possibility and directed to a specific channel for guidance.
- **Prompt Format Frustrations and CodeLLama Discussion**: Discussion on `#general` involved the complications caused by prompt formats, including a heads-up on Reddit about issues with CodeLLama 70b and llama.cpp not supporting chat templates, leading to poor model outputs.

**Links mentioned**:

- [mistralai/Mixtral-8x7B-v0.1 · Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1): no description found
- [GitHub - joonspk-research/generative_agents: Generative Agents: Interactive Simulacra of Human Behavior](https://github.com/joonspk-research/generative_agents): Generative Agents: Interactive Simulacra of Human Behavior - GitHub - joonspk-research/generative_agents: Generative Agents: Interactive Simulacra of Human Behavior
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1afweyw/comment/koczpz9/?utm_source=share&utm_medium=web2x&context=3): no description found
- [Nexesenex/MIstral-QUantized-70b_Miqu-1-70b-iMat.GGUF · Hugging Face](https://huggingface.co/Nexesenex/MIstral-QUantized-70b_Miqu-1-70b-iMat.GGUF): no description found
- [GitHub - ggerganov/llama.cpp: Port of Facebook&#39;s LLaMA model in C/C++](https://github.com/ggerganov/llama.cpp): Port of Facebook&#39;s LLaMA model in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1202187091950174269) (100 messages🔥🔥): 

- **The Quest for Local LLM Training**: `@scampbell70` is on a mission to train a local large language model (LLM) specifically for stable diffusion prompt writing, considering the possibility of investing in multiple Quadro A6000 GPUs. The aim is to avoid the terms-of-service restrictions of platforms like ChatGPT and learn to train models using tools like Lora.

- **LM Studio Compatibility with Other Tools**: Users like `@cos2722` and `@vbwyrde` are discussing which models work best with LM Studio in conjunction with Autogenstudio, crewai, and open interpreter. The consensus is that Mistral works but MOE versions do not, and finding a suitable code model that cooperates remains a challenge.

- **LM Studio's Hardware Hunger**: `@melmass` reports significant resource usage with the latest LM Studio version, indicating that it taxes even powerful setups with a 4090 GPU and 128GB of RAM.

- **Gorilla Open Function Template Inquiry**: `@jb_5579` is seeking suggestions for JSON formatted prompt templates to use with the Gorilla Open Function in LM Studio with no follow-up discussion provided.

- **Exploring Quantization and Model Size for Performance**: `@binaryalgorithm`, `@ptable`, and `@kujila` engaged in an extensive discussion about utilizing quantized models with different parameter counts for performance and inference speed. The trade-off of depth and creativity in responses from larger models against the fast inference from low quant models was a significant part of the conversation.

**Links mentioned**:

- [LLM-Perf Leaderboard - a Hugging Face Space by optimum](https://huggingface.co/spaces/optimum/llm-perf-leaderboard): no description found
- [Big Code Models Leaderboard - a Hugging Face Space by bigcode](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard): no description found

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1202184252880605235) (21 messages🔥): 

- **Clarification Sought on Catalog Updates**: `@markushenriksson` inquired about checking for catalog updates, to which `@heyitsyorkie` confirmed that the updates are present.
- **Manual File Cleanup Needed for Failed Downloads**: `@ig9928` reported an issue where failed downloads required manual deletion of incomplete files before a new download attempt can be made, requesting a fix for this inconvenience.
- **Request for Chinese Language Support**: `@gptai` expressed interest in Chinese language support for LM Studio, noting the difficulty their Chinese fans face with the English version.
- **Confusion with Error Messages**: `@mm512_` shared a detailed error message received when attempting to download models. `@yagilb` directed them to seek help in the Linux support channel and advised running the app from the terminal for error logs.
- **Model Compatibility Queries and Tutorial Issues**: `@artdiffuser` struggled with downloading models and was informed by `@heyitsyorkie` that only GGUF models are compatible with LMStudio. Additionally, `@artdiffuser` was cautioned about potentially outdated or incorrect tutorials, and `@yagilb` further addressed the issue by suggesting a bug report in the relevant channel for their specific error.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1202162522405482496) (157 messages🔥🔥): 

- **GPUs and Memory Management**: `@0pechenka` asked whether memory usage on GPUs and CPUs when running models like LLaMA v2 is customizable. They wanted a beginner's guide for setting things up, asking for instructions, possibly on YouTube. `@pefortin` suggested removing the "keep entire model in ram" flag if the model doesn't fit into memory to utilize the swap file on Windows.

- **Building a Budget LLM PC**: `@abesmon` inquired about a good PC build for running large language models (LLMs) on a medium budget, and `@heyitsyorkie` advised that VRAM is crucial, recommending a GPU with at least 24GB of VRAM.

- **Power Supply for Multiple GPUs Discussed**: `@dagbs` and `@pefortin` discussed the potential need for more power when running several GPUs, with pefortin considering to sync multiple PSUs to handle his setup, which includes a 3090 and other GPUs connected through PCIe risers.

- **LLM Performance on Diverse Hardware Configurations**: `@pefortin` shared his experience of approximately 30-40% performance of a P40 GPU compared to a 3090 for LLM tasks, and mentioned driver issues on Windows, which are not a problem on Linux.

- **Mixing GPU Generations for LLMs**: Users `@ellric_` and `@.ben.com` debated on the compatibility and performance implications of using different generations of GPUs, like the M40 and P40, with LLMs. `.ben.com` admitted needing to investigate the performance consequences of splitting models across GPUs, positing a potential 20% slowdown from unscientific testing.

**Links mentioned**:

- [Mikubox Triple-P40 build](https://rentry.org/Mikubox-Triple-P40): Dell T7910 &quot;barebones&quot; off ebay which includes the heatsinks. I recommend the &quot;digitalmind2000&quot; seller as they foam-in-place so the workstation arrives undamaged. Your choice of Xe...
- [Simpsons Homer GIF - Simpsons Homer Bart - Discover &amp; Share GIFs](https://tenor.com/view/simpsons-homer-bart-why-you-little-gif-17376912): Click to view the GIF

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/) (1 messages): 

mike_50363: Is the non-avx2 beta version going to be updated to 2.12?
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1202202302157701130) (137 messages🔥🔥): 

- **Improving DALL-E Generated Faces**: `@abe_gifted_art` queried about updated faces generated by **DALL-E**, noticing they weren't distorted now. However, no specific update or change in date was provided in the discussion.
- **AI Understanding Through Non-technical Books**: `@francescospace` sought recommendations for non-technical books on AI and its potential or problems. `@laerun` suggested engaging in community discussions like the one on Discord, while `@abe_gifted_art` recommended looking into Bill Gates' interviews and Microsoft's "Responsible AI" promise, sharing a link: [Microsoft's approach to AI](https://news.microsoft.com/source/features/ai/microsoft-approach-to-ai/).
- **Debate Over Training AI with Unethical Data**: A debate occurred between `@darthgustav.` and `@lugui` regarding the ethical implications and technical necessity of including harmful material in AI datasets. `@yami1010` later joined, highlighting the complexity of how language models work and the creativity seen in AI outputs.
- **Challenges of Successful DALL-E Prompts**: `@.ytty` expressed frustrations with DALL-E not following prompts correctly, while `@darthgustav.` offered advice on constructing effective prompts, emphasizing the use of detail and avoiding negative instructions.
- **Starting with AI Art and AI Scripting Language Training**: New users asked how to begin creating AI art, and `@exx1` recommended stable diffusion requiring an Apple SoC or Nvidia GPU, while `@cokeb3ar` explored how to teach AI a new scripting language without the ability to upload documentation.

**Links mentioned**:

[What is Microsoft&#039;s Approach to AI? | Microsoft Source](https://news.microsoft.com/source/features/ai/microsoft-approach-to-ai/): We believe AI is the defining technology of our time. Read about our approach to AI for infrastructure, research, responsibility and social good.

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1202195079520075777) (85 messages🔥🔥): 

- **GPT Image Analysis Challenge**: `@cannmanage` sought help using GPT to analyze videos for identifying a suspected abductor's van among multiple white vans. `@darthgustav.` suggested using the [Lexideck Vision Multi-Agent Image Scanner tool](https://chat.openai.com/g/g-BAhF6yS9e-lexideck-vision-multi-agent-image-scanner) and adjusting the prompt for better results, noting the importance of the context in which the tool is used.

- **Identifying Active GPT Models**: `@nosuchipsu` inquired about determining the specific GPT model in use, and `@darthgustav.` clarified that GPT Plus and Team accounts use a 32k version of GPT-4 Turbo, with different usage for API and Enterprise plans. He later shared a [link to transparent pricing plans](https://openai.com/pricing) and [model details](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo).

- **Image Display Hurdles**: Users `@quengelbert` and `@darthgustav.` discussed whether GPT can display images from web URLs within the chat, concluding that such direct display isn't currently supported, evidenced by an error message displayed when attempting this function on GPT-4.

- **Understanding @GPT Functionality**: `@_odaenathus` discussed confusion with the `@` system in GPTs, suspecting a blend rather than a clear handover between the original and `@` GPT. `@darthgustav.` confirmed the shared context behavior and suggested that disconnecting from the second GPT would revert back to first instructions, though a bug or bad design might cause unexpected behavior.

- **Managing D&D GPTs Across Devices**: `@tetsujin2295` brought up an inability to @ GPTs on mobile as a Plus member, which `@darthgustav.` attributed to a potential lack of roll out or mobile browser limitations. He also shared the effective use of structuring multiple D&D related GPTs for various roles and world-building, with a slight concern over token limit constraints.

**Links mentioned**:

[Pricing](https://openai.com/pricing): Simple and flexible. Only pay for what you use.

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1202245700164001822) (62 messages🔥🔥): 

- **Sideways Dilemma in Portrait Generation**: `@wild.eva` and `@niko3757` discussed challenges with generating vertical, full-body portrait images; the model appears to force landscape mode. `@niko3757` suggested a workaround of generating in landscape and then stitching images in portrait after upscaling, though this is speculation and they await a significant update for DALL-E.
- **Portrait Orientation Gamble**: `@darthgustav.` suggested that due to symmetry and lack of an orientation parameter, getting a correct vertical portrait is a 25% chance occurrence. This implies a fundamental limitation, unrelated to how images are prompted.
- **Prompt Improvement Attempts**: `@wild.eva` sought suggestions for prompts to improve image orientation, `@niko3757` provided examples, but `@darthgustav.` countered by indicating that no prompt improvement could overcome the model's inherent constraints.
- **Integrating Feedback Buttons in Custom GPT**: `@luarstudios` inquired about adding interactive feedback buttons post-answer in their GPT model, engaging with `@solbus` and `@darthgustav.` for guidance. `@darthgustav.` offered a detailed explanation about incorporating an Interrogatory Format to attach a menu of responses to each question.
- **Structural Discussions on Design Proposals**: `@solbus` and `@darthgustav.` pondered whether to have a different feedback structure for the first logo design compared to subsequent ones, suggesting this could increase efficiency and relevance in feedback collection. `@darthgustav.` shared a link to their own approach in DALL-E Discussions.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1202245700164001822) (62 messages🔥🔥): 

- **Sideways Image Chaos**: `@wild.eva` encountered issues with her detailed prompts which resulted in sideways images or undesired scene generation, suggesting **training issues** with the model. `@niko3757` confirmed the likely cause is a **built-in error** and suggests trying images in landscape then upscaling, while `@darthgustav.` mentioned the lack of **orientation parameters** limits vertical orientations.
  
- **Expectations for DALLE Update**: `@niko3757` shared optimistic yet **unconfirmed speculation** about a significant update to **DALLE** coming soon, which they are eagerly awaiting for improved results.
  
- **GPT-3 Button Dilemma**: `@luarstudios` sought help with **adding response buttons** after the AI presents design proposals, getting feedback from both `@solbus` and `@darthgustav.` on how to implement this in the chatbot's **custom instructions**.
  
- **Conversation Structure Strategy**: `@darthgustav.` provided a strategy for `@luarstudios` to guide the **AI's interaction pattern**, suggesting templates and **conversation starter cards** to handle the logical flow in presenting logo options and receiving feedback.
  
- **Sharing Insights Across Channels**: `@darthgustav.` cross-posts to **DALL-E Discussions**, illustrating the efficiency of **prompt engineering** for a logo concept and recommending open-source examples across channels for better **community assistance**.
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1202187759838842920) (109 messages🔥🔥): 

- **Chasing the Phantom Spike**: `@nafnlaus00` and `@dreamgen` discussed **training patterns** and **learning rate (LR)** adjustments in their models, observing spikes at every epoch. Despite trying lower LRs and increased dropout, `@nafnlaus00` mentioned still facing **overtraining** issues.

- **The Quest for Server-grade GPUs**: `@le_mess` offered connections for purchasing server-grade hardware. In the meanwhile, `@yamashi` contemplated acquisitions, considering a `Gigabyte Server G262-ZO0` setup with AMD EPYC processors and the **AMD MI250** GPUs, weighing the benefits against renting.

- **Software Stack Skepticism**: Although AMD's offerings like the `MI250` intrigued some users for their memory capacity and potential performance, `@yamashi` expressed doubts about the maturity of AMD's software stack compared to Nvidia.

- **Commit Conundrum**: `@nafnlaus00` identified a likely problematic commit (`da97285e63811c17ce1e92b2c32c26c9ed8e2d5d`) in the `axolotl` library that might be causing significant **overtraining** in their model fine-tune. They're undertaking a methodical approach to isolate the change responsible.

- **Tiny YAMLs, Major Leaps**: `@caseus_` shared a small, **11-line YAML** config that can kick off a finetune, mentioning an imminent **refactor** for `axolotl` to further streamline and improve defaults, which might reduce complexity even more ([PR #1239](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1239)).


**Links mentioned**:

- [AMD + 🤗: Large Language Models Out-of-the-Box Acceleration with AMD GPU](https://huggingface.co/blog/huggingface-and-optimum-amd): no description found
- [keep gate in fp32 for 16 bit loras (#1105) · OpenAccess-AI-Collective/axolotl@da97285](https://github.com/OpenAccess-AI-Collective/axolotl/commit/da97285e63811c17ce1e92b2c32c26c9ed8e2d5d): * keep gate in fp32 for loras
 
 * add e2e check for lora w/o flash attention for mixtral to check gate
 
 * add checks for gate in fp32 for mixtral, add typehints to train outputs
 
 * mixtral doe...
- [WIP: Pydantic cfg by winglian · Pull Request #1239 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1239): Description  Motivation and Context   How has this been tested?    Screenshots (if appropriate) Types of changes  Social Handles (Optional)
- [Gigabyte 2U MI250 Server G262-ZO0](https://www.thinkmate.com/system/gigabyte-g262-zo0): no description found
- [no title found](https://www.ahead-it.eu/>): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1202231656623394826) (26 messages🔥): 

- **CI Breakdown Post Torch-2.2.0**: `@caseus_` reported that the torch-2.2.0 release broke their CI without specifying which version it requires. They shared a [job log](https://github.com/OpenAccess-AI-Collective/axolotl/actions/runs/7725562109/job/21060161617) that details the issue.
- **Confusion Over 'torch' In Requirements**: `@nanobitz` asked whether 'torch' could be removed from requirements fearing it overrides the base torch, but `@caseus_` confirmed that 'torch' was already removed a while back.
- **Norm Equilibrium in LoftQ Improvement**: `@stefangliga` suggested an improvement for LoftQ where vectors k1 and k2 could be used to rescale matrices A and B, thus matching per axis norms to improve gradient magnitudes.
- **Axolotl Torch Version Control PR**: `@caseus_` introduced a [pull request #1234](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1234) aimed at fixing CI issues by setting the torch version to match the installed version during axolotl install, which prevents auto-upgrading to the problematic torch-2.2.0.
- **Checkpoint Upload Issue with Qlora**: `@dreamgen` reported a problem where Qlora doesn't upload all checkpoints, only the final one. `@caseus_` suggested that this might be an upstream issue with Transformers and mentioned a potential workaround involving `TrainerCallback`.


**Links mentioned**:

- [Fix and document test_datasets (#1228) · OpenAccess-AI-Collective/axolotl@5787e1a](https://github.com/OpenAccess-AI-Collective/axolotl/actions/runs/7725562109/job/21060161617): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [set torch version to what is installed during axolotl install by winglian · Pull Request #1234 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1234): Description The latest torch-2.2.0 breaks our CI as it attempts to install the latest torch version. Motivation and Context   How has this been tested?    Screenshots (if appropriate) Types of chan...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1202162781156298812) (42 messages🔥): 

- **Axolotl Installation Stuck on flash-attn**: `@arindam8841` encountered issues installing axolotl related to building the wheel for flash-attn. They resolved the issue by installing the latest version of flash-attn separately using `pip install flash-attn --no-build-isolation`.
- **Understanding Batch Size in axolotl**: `@dreamgen` questioned how batch size scales with the number of GPUs in axolotl, while `@caseus_` clarified that using DeepSpeed, specifically zero3, adjusts for distributed data parallel (DDP) and batch sizes, making it unnecessary for models that fit on a single GPU.
- **Dataset Configuration Challenges**: `@jorelosorio` faced errors trying to use the same dataset with different formats for different tasks within axolotl. The issue was resolved by specifying different `data_files` paths for each task.
- **Merging Models with Axolotl**: `@cf0913` experienced an `AttributeError` while attempting to merge QLORA with a base model, seeking advice on whether to use the axolotl command line or another method, with a suggestion from `@le_mess` about full finetuning.
- **Request for Token Counting Script**: Both `@arcontex` expressed a need for a script to count the number of tokens in a file, and `@gonejiggy` offered to write a CLI tool and asked if a pull request would be accepted. `@nanobitz` mentioned that axolotl already outputs token counts in logs and asked for the yaml configuration to further assist `@arcontex`.

**Links mentioned**:

- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl?tab=rea): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1202310916226420756) (4 messages): 

- **DPO Experiencing Issues**: `@giftedgummybee` signalled that **DPO (Data Processing Optimizer)** is currently experiencing issues, but did not specify the nature of the problem.
- **Request for Clarification on DPO**: `@filippob82` asked for a more detailed explanation of the issues with **DPO** from `@giftedgummybee`.
- **VRAM Usage Concerns with DPO**: `@c.gato` elaborated that the DPO is consuming a considerable amount of **VRAM**, specifically mentioning the challenges of running QLora on a 13b model.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1202183553446858752) (5 messages): 

- **Mysterious Empty Folder Phenomenon**: `@nanobitz` shared an amusing discovery that they finally encountered the **empty folder issue** which had previously never occurred to them.
- **Runpod's Intermittent Folder Vanishing Act**: `@dreamgen` chimed in, mentioning that the issue of folders appearing empty happens sometimes **even without network volume**, but the cause remains unknown.
- **Hot Deal Alert on Community Cloud**: `@dreamgen` excitedly announced the availability of **H100 SXM** units on the community cloud, priced at **3.89**—urging users to seize the opportunity quickly.
- **Runpod Documentation Dilemma**: `@gonejiggy` expressed frustration over having to instruct users to run `pip install -e .` due to a recurring error, acknowledging that adding this to the documentation is not ideal.
- **Ghost in the Machine on Runpod**: `@dreamgen` reports a baffling situation where a **mysterious process** is consuming memory and GPU on their runpod machine but remains undetectable and unkillable through conventional commands.
  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1202167882461487114) (149 messages🔥🔥): 

- **Model Training Woes and Solutions**: `@drummer_.` experienced a loss flatline to `0` during model training which later appeared to be related to setting a higher batch size. Switching to a batch size of `1` and considering the use of `EarlyStoppingCallback` were discussed, with input from `@Cubie | Tom` suggesting training instability, and `@doctorpangloss` indicating that 4bit quantization could be a contributing factor.

- **Seeking Specific LLMs for Tech Data**: `@zorian_93363` inquired about language models specifically trained on datasets related to Arduino, Esp 32, Raspberry Pi, and similar tech subjects.

- **In Search of Efficiency in Large Model Training**: `@ericpan.xyz` asked about the fastest format for higher inference speed and ways to reduce a model's memory footprint, with `@meatfucker` suggesting 4bit quantization despite some loss of accuracy to conserve VRAM.

- **Exploring Embedding Logic Across Multiple Namespaces**: `@abhisheknegi_12043` sought advice on designing logic to query similarity across multiple namespaces for meeting transcripts stored as vector embeddings in Pinecone.

- **Integrating Diffusion Models into a Social Platform**: `@goodtimes5241` sought assistance with integrating a fine-tunable stable diffusion model, similar to "diffuse the rest," into a social media application. They previously explored using the Inference API and alternatives like the Stable Diffusion Computing Network for image-to-image generation capabilities.

- **Publishing Community Content Challenges**: `@aliabbas60` reported difficulty with an authorization error preventing the publishing of a community blog post without further details on the issue.

**Links mentioned**:

- [ControlNet in 🧨 Diffusers](https://huggingface.co/blog/controlnet): no description found
- [GitHub - fiatrete/SDCN-Stable-Diffusion-Computing-Network: SDCN is an infrastructure that allows people to share and use Stable Diffusion computing power easily.](https://github.com/fiatrete/SDCN-Stable-Diffusion-Computing-Network): SDCN is an infrastructure that allows people to share and use Stable Diffusion computing power easily. - GitHub - fiatrete/SDCN-Stable-Diffusion-Computing-Network: SDCN is an infrastructure that al...
- [Image-to-image](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/img2img): no description found
- [Callbacks](https://huggingface.co/docs/transformers/main_classes/callback#transformers.EarlyStoppingCallback): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1202268298616512602) (3 messages): 

- **Understanding Bahdanau Attention**: User `@sardarkhan_` expressed that while reading about **Bahdanau attention**, they found it to be complex yet fascinating.
- **Advancements in Reinforcement Learning**: `@sardarkhan_` mentioned working on a reinforcement learning project and plans on continuing to develop an **aimlabs agent**.
- **Seeking Motivational Boost**: User `@kaikishatu` expressed their need for motivation in the **today-im-learning** channel.
  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1202217238971555880) (11 messages🔥): 

- **Tweet Critique Request**: User `@vipitis` shared a [tweet](https://twitter.com/Vipitis/status/1752699776766988309) containing their thesis proposal for a science communication seminar and invited feedback from the community.

- **Magic: The Gathering Model Showcase**: `@joshuasundance` announced the availability of a space for the Magic model they discussed and provided a [link](https://huggingface.co/spaces/joshuasundance/mtg-coloridentity) to the model on HuggingFace Spaces, along with a visual preview.

- **Custom Pipeline for MoonDream**: User `@not_lain` has successfully created a custom pipeline for the `moondream1` model and shared a [pull request](https://huggingface.co/vikhyatk/moondream1/discussions/6) explaining its usage, including code snippets for testing before merging.

- **Acknowledgment for Compute Support**: User `@not_lain` gave a shout-out to `<@994979735488692324>` for providing the necessary compute power that aided in the completion of their custom pipeline for the `moondream1` model.

- **Necessary Tomorrows Podcast Release**: `@deepaaar` worked on music and sound design for Al Jazeera's sci-fi podcast "Necessary Tomorrows," utilizing AI in the creative process. They shared [where to listen](https://dohadebates.com/podcasts/necessary-tomorrows/) to the podcast and provided a brief synopsis, mentioning that the podcast mixes speculative fiction and documentary elements centered around the turbulent 2020s.

**Links mentioned**:

- [Gradio HTML Docs](https://www.gradio.app/docs/html#demos): no description found
- [mtg-coloridentity - a Hugging Face Space by joshuasundance](https://huggingface.co/spaces/joshuasundance/mtg-coloridentity): no description found
- [vikhyatk/moondream1 · add pipeline](https://huggingface.co/vikhyatk/moondream1/discussions/6): no description found
- [GitHub - cdpierse/transformers-interpret: Model explainability that works seamlessly with 🤗 transformers. Explain your transformers model in just 2 lines of code.](https://github.com/cdpierse/transformers-interpret): Model explainability that works seamlessly with 🤗 transformers. Explain your transformers model in just 2 lines of code.  - GitHub - cdpierse/transformers-interpret: Model explainability that works.....
- [Necessary Tomorrows](https://dohadebates.com/podcasts/necessary-tomorrows/): Ursula, an AI instructor from the future, encourages listeners to study the turbulent 2020s. Mixing speculative fiction and documentary, leading sci-fi authors bring us three futures that seem like fa...

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1202469478814851102) (1 messages): 

- **Delving into Language Model Compression**: `@ericauld` shared a [paper](https://arxiv.org/abs/2401.15347v1) discussing various **compression algorithms for language models** and invited others to read and discuss. The paper addresses the need to balance model efficiency with accuracy and touches on methods like **pruning, quantization, knowledge distillation**, and more.

**Links mentioned**:

[A Comprehensive Survey of Compression Algorithms for Language Models](https://arxiv.org/abs/2401.15347v1): How can we compress language models without sacrificing accuracy? The number of compression algorithms for language models is rapidly growing to benefit from remarkable advances of recent language mod...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1202212284102094868) (1 messages): 

- **Seeking Cross-Namespace Similarity Query Logic**: User `@abhisheknegi_12043` is requesting guidance on creating a query logic that can determine similarity across multiple namespaces within Pinecone for a project involving meeting transcripts embedding. They're looking for advice on design and implementation strategies for this functionality.
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages): 

merve3234: it's not an error but rather a warning, feel free to ignore
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1202277560168489061) (2 messages): 

- **Seeking GPU Resources for Open Source Contribution**: User `@lpbb` is looking to contribute to the [nanotron](https://github.com/huggingface/nanotron) library and is in search of **two linked GPUs** for testing code. They mentioned willingness to pay for access if the cost is reasonable.
- **NeuralBeagle14-7b Running on 8GB GPU Demo**: `@joshuasundance` shared a link to a [GitHub repository](https://github.com/joshuasundance-swca/llamacpp-langchain-neuralbeagle-demo), demonstrating how they managed to run neuralbeagle14-7b on a local 8GB GPU. The repository acts as a showcase for others interested in a similar setup.

**Links mentioned**:

[GitHub - joshuasundance-swca/llamacpp-langchain-neuralbeagle-demo: a small demo repo to show how I got neuralbeagle14-7b running locally on my 8GB GPU](https://github.com/joshuasundance-swca/llamacpp-langchain-neuralbeagle-demo): a small demo repo to show how I got neuralbeagle14-7b running locally on my 8GB GPU - GitHub - joshuasundance-swca/llamacpp-langchain-neuralbeagle-demo: a small demo repo to show how I got neuralbe...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1202212284102094868) (1 messages): 

- **Seeking Cross-Namespace Similarity Logic**: `@abhisheknegi_12043` is working on a project using **Pinecone for vector embedding** to store vectors from meeting transcripts. They are requesting assistance in designing a logic to fetch similarities across multiple namespaces.
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1202178725525852170) (72 messages🔥🔥): 

- **Synthetic Textbook Dataset Search**: User `@alyosha11` sought synthetic textbook datasets for a project similar to `phi`, and `@JH` indicated that there were ongoing efforts in another Discord channel (`#1185268316001009785`) which `@alyosha11` acknowledged checking out.
- **MusicLM Audio File Conundrum**: `@finnilda` faced an issue while training MusicLM transformers due to missing audio files and inquired about sources for the needed dataset. They noted the lack of responses on [GitHub](https://github.com/lucidrains/musiclm-pytorch) and sought assistance from the Discord community.
- **Filtering Underage Content**: `@latke1422p` requested help to filter images with underage subjects using a dataset of trigger words, expressing the need for building safer AI content moderation tools.
- **Research-oriented Pings on Discord**: Various users including `@astropulse`, `@progamergov`, and `@.undeleted` engaged in a discussion regarding the appropriateness of using '@everyone' pings on a research server, with diverse opinions though consensus leaning towards acceptability given the server's research focus.
- **VAE Training Errors Discovered**: `@drhead` revealed a discovery that the kl-f8 VAE improperly packs information, which is a deviation from the intended training method, impacting models that rely on it. This prompted discussion and linking to a relevant [research paper](https://arxiv.org/abs/2309.16588) by `@.giacomov` discussing artifacts in transformer learning.

**Links mentioned**:

- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588): Transformers have recently emerged as a powerful tool for learning visual representations. In this paper, we identify and characterize artifacts in feature maps of both supervised and self-supervised ...
- [GitHub - lucidrains/musiclm-pytorch: Implementation of MusicLM, Google&#39;s new SOTA model for music generation using attention networks, in Pytorch](https://github.com/lucidrains/musiclm-pytorch): Implementation of MusicLM, Google&amp;#39;s new SOTA model for music generation using attention networks, in Pytorch - GitHub - lucidrains/musiclm-pytorch: Implementation of MusicLM, Google&amp;#39;s ...

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1202168481072820255) (19 messages🔥): 

- **LLaVA-1.6 Unveiled**: `@nodja` shares [LLaVA-1.6 release blog post](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/) emphasizing its improved features like higher image resolution and enhanced OCR. LLaVA-1.6 notably surpasses Gemini Pro in several benchmarks.
- **LLaVA-1.6 Excels in Comic Test**: `@nodja` tested LLaVA-1.6 with varying difficulties of a comic test, stating it "gets everything almost right" on the easy version, while the medium version presents more hallucinations.
- **Photographic Landmark Recognition**: `@helium__` compares LLAVA's performance to GT4-V on identifying a European city from a personal photo, with GT4-V accurately naming Porto while LLAVA struggles to be specific.
- **Multimodal Model Capabilities Questioned**: `@mfcool` inquires about the mechanisms enabling style preservation in VLMs like Dall-E, observing that others tend to generalize while Dall-E accurately reflects specific styles.
- **SPARC Introduced for Detailed Multimodal Representations**: `@spirit_from_germany` shared a Twitter link about SPARC, a new method for pretraining multimodal representations with fine-grained details, but `@twoabove` and `@mkaic` express excitement, although it's noted there's no code or models available yet.

**Links mentioned**:

- [LLaVA-1.6: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/): LLaVA team presents LLaVA-1.6, with improved reasoning, OCR, and world knowledge. LLaVA-1.6 even exceeds Gemini Pro on several benchmarks.
- [LLaVA](https://llava.hliu.cc/): no description found
- [Tweet from Ioana Bica (@IoanaBica95)](https://fxtwitter.com/IoanaBica95/status/1752643360039256313?t=d3hb1n0cF8MycbA-wr6rjw&s=19): Excited by the generality of CLIP, but need more fine-grained details in your representation? Introducing SPARC, a simple and scalable method for pretraining multimodal representations with fine-grain...

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1202182833809145896) (37 messages🔥): 

- **Curiosity About Custom GPTs**: `@sweetpopcornsimon` expressed interest in training a private model similar to building a custom GPT with ChatGPT. `@icelavaman` clarified that Perplexity AI does not offer chatbot services, but rather a feature called Collections, which organizes threads into shareable spaces for collaboration.

- **PDF/EPUB Reader Inquiry**: `@archient` asked the community if there is an epub/PDF reader that supports AI text manipulation and has the capability to use custom APIs.

- **In Search of Unique Notification Sounds**: `@noell5951` inquired whether there is a distinct notification sound for Perplexity, indicating they have not tried it but are curious.

- **Seeking Advice for Perplexity AI Pro**: `@johnl4119` had a "how to" question about Perplexity AI Pro and was guided by `@ok.alex` to the "quick-questions" area as the appropriate place to ask.

- **Experiencing Prompt Disappearance Issue**: `@nuggdavis` reported an unusual issue where prompts briefly disappear before getting responses after a refresh, a problem occurring across multiple browsers and operating systems. `@gumby2411` confirmed having a similar experience.

**Links mentioned**:

- [What is Collections?](https://blog.perplexity.ai/faq/what-is-collections): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Perplexity Careers](https://blog.perplexity.ai/careers): Join our team in shaping the future of search and knowledge discovery.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1202184474100760647) (2 messages): 

- **Perplexity's Insight into Google's Future**: User `@m1st3rg` shared a brief and mysterious insight, claiming to have learned about **the future of Google** through their experience with Perplexity.
- **YouTube Guide on Content Creation with Perplexity**: `@redsolpl` provided a [YouTube video link](https://www.youtube.com/watch?v=iY4q7chZC1Y) titled "How I Use Perplexity AI to Source Content Ideas for LinkedIn," which describes how they've effectively integrated **Perplexity AI** into their social media **content creation process**.

**Links mentioned**:

[How I Use Perplexity AI to Source Content Ideas for LinkedIn](https://www.youtube.com/watch?v=iY4q7chZC1Y): A dive deep into how I&#39;ve been using Perplexity AI to revolutionize my content creation process for social media. Whether you&#39;re a content creator, business ...

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1202169485466017812) (31 messages🔥): 

- **In Pursuit of Support**: User `@angelo_1014` expressed concerns over not receiving a response from support@perplexity.ai, to which `@ok.alex` replied asking to ensure the correct email was used for contact and offered to check on the matter if provided with a personal email via DM, which `@angelo_1014` confirmed sending. 
- **Codellama Conundrums Confounding**: `@bvfbarten.` reported odd responses from the *codellama-70b-instruct* model and was reassured by `@ok.alex` that the issue would be investigated. Subsequent conversation affirmed that the *codellama-34b-instruct* model was functioning solidly.
- **Parsing the API Puzzle**: `@andreafonsmortigmail.com_6_28629` discussed the complexity of handling file uploads with the chatbox interface, with `@clay_ferguson` proposing the use of Apache Tika for extracting text from files and guiding on the file upload process without involving API uploads.
- **Deciphering Online Models**: In a discussion initiated by `@kid_tubsy` regarding the ability to list links/sources of data for API models, `@brknclock1215` clarified that the *-online* models can access real-time internet data but do not provide citations in their responses. `@clay_ferguson` acknowledged this feature as beneficial despite the lack of source citations.
- **Model Sources Matter**: `@brknclock1215` emphasized the importance of having source URLs when using online capable models for verifying information and overcoming the issues with hallucinations, especially for research purposes that require validated data for client-facing reports.
  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1202197971945668608) (38 messages🔥): 

- **Langchain and Vector Search Dilemmas**: `@abhisheknegi_12043` seeks advice on creating a logic for fetching similarity across multiple Pinecone namespaces for storing meeting transcripts, while `@agenator.` reports issues querying large JSON files using Langchain and ChromaDB, only getting partial data responses.
- **Embeddings in Action**: In response to a query about handling embeddings with OpenAI, `@agenator.` shared a snippet of Express.js code that uses Langchain and Chroma for Document embeddings.
- **AI for Stock Market Analysis**: `@funk101.` is on the lookout for an AI solution that can analyze "real-time" stock market information and respond based on pre-existing data.
- **Pinecone Puzzles**: `@bardbuddy` is stuck trying to run an older Langchain app due to a TypeError and updates to the langchain/pinecone package, and is seeking immediate help for this issue.
- **Mixture of Agents Concept**: `@the_agent_j` explores the idea of using fine-tuned models for specific agent types within a system referred to as Mixture of Agents, considering the potential of using custom GPTs from OpenAI for specialized roles.

**Links mentioned**:

- [no title found](http://localhost:8004",): no description found
- [Thumbs Up Like GIF - Thumbs Up Like Thumbs Up Gif - Discover &amp; Share GIFs](https://tenor.com/view/thumbs-up-like-thumbs-up-gif-gif-art-my-gif-art-gif-27008062): Click to view the GIF
- [SkmAI: AI-Powered YouTube Video Search](https://chromewebstore.google.com/detail/skmai-ai-powered-youtube/nkkklchgjghdppjfponpogcfgggchjef): Search through videos of any language, find the most relevant clips and timestamp to them using AI.

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1202243570187378770) (9 messages🔥): 

- **LangGraph Opens New AI Collaboration Horizons**: `@andysingal` shared an article titled *Unveiling the Future of AI Collaboration with LangGraph: Embracing Multi-Agent Workflows*. Read about the next-generation tool LangGraph and its role in multi-agent AI systems in the [Medium post](https://medium.com/ai-advances/unveiling-the-future-of-ai-collaboration-with-langgraph-embracing-multi-agent-workflows-89a909ddd455).

- **Visual Canvas Enquiry**: `@nbbaier` asked about the platform/tool used for the visual canvas seen in an unspecified video.

- **Trouble in Node.js Town**: `@bardbuddy` encountered an error when running an old LangChain app due to what appears to be updates in the LangChain API, specifically related to `PineconeStore` and `PineconeClient`.

- **Language Misidentification Misadventure**: In response to `@bardbuddy`'s error, `@johnny2x2` mistakenly thought the code snippet might be in C, but `@bardbuddy` clarified that the issue concerns Node.js.

- **Uninvited Invitation Link Sharing**: `@cryptosmiler` posted an invite to join a Discord server with a 5-use invite code, hoping for a quick response before the invites run out.

**Links mentioned**:

[Unveiling the Future of AI Collaboration with LangGraph: Embracing Multi-Agent Workflows](https://medium.com/ai-advances/unveiling-the-future-of-ai-collaboration-with-langgraph-embracing-multi-agent-workflows-89a909ddd455): Ankush k Singal

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1202339070869504050) (1 messages): 

- **Dive into OpenAI Embeddings with LangChain**: `@datasciencebasics` shared a [YouTube video tutorial](https://youtu.be/ssgvViL0fao) on using the **new OpenAI Embeddings Model with LangChain**. The video covers the introduction of the new embeddings model by OpenAI, updates to GPT-3.5 Turbo, and new tools for developers to manage AI applications.

**Links mentioned**:

[How To USE New OpenAI Embeddings Model with LangChain 🦜🔗](https://youtu.be/ssgvViL0fao): OpenAI introduces new Embeddings model. They are releasing new models, reducing prices for GPT-3.5 Turbo, and introducing new ways for developers to manage A...

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1202307837947293768) (2 messages): 

- **Tuning Hybrid Search is Key**: LlamaIndex highlights that **hybrid search** needs to be adjusted for different types of questions when using Retrieval-Augmented Generation (RAG) systems. There are categories like Web search queries and concept seeking, each requiring a unique approach [Twitter thread](https://twitter.com/llama_index/status/1752748298392502521).
- **Deep Dive into Multimodal RAG Systems**: A new video by `@_nerdai_` focuses on evaluating multimodal RAG systems using LlamaIndex, covering an introduction to RAG, evaluation techniques, building multimodal RAG, as well as challenges faced [YouTube link update notification](https://support.google.com/youtube/answer/175292) [Tweet about video](https://twitter.com/llama_index/status/1752848239081214312).

**Links mentioned**:

[no title found](https://t.co/d0E9vvgS1f): no description found

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1202173440434376704) (43 messages🔥): 

- **Seeking Vector Dump Into OpenSearch**: User `@natuto_uzumaki1808` requested help with dumping local vector embedding files into an Opensearch database, tagging specific members for assistance, but did not provide further context or details.

- **RAG Context Size Curiosity**: `@wrapdepollo` asked about the effect of RAG context size on retrieved chunks, sparking a brief confirmation by `@Teemu`, additional clarification by `@cheesyfishes` about LlamaIndex's handling of oversized contexts, and a thank-you response by `@wrapdepollo`.

- **RAG Over Code in LlamaIndex Inquiry**: `@richard1861` queried about a tutorial for RAG over code using LlamaIndex, referencing Langchain's approach. `@Teemu` and `@cheesyfishes` engaged in the discussion, with `@cheesyfishes` inviting contributions for better code splitting and `@rawwerks` sharing code extract from langchain used for an award-winning hackathon submission.

- **Querying Alternative KeywordIndex Storage**: `@mysterious_avocado_98353` asked about cloud-based storage solutions for KeywordIndex in relation to large-scale data ingestion. `@cheesyfishes` directed them to docstore+index_store integrations, with an example provided in a URL.

- **API Choice Conversations**: `@mrpurple9389` inquired about preferences between the assistant and completions APIs, and `@hosermage` sought advice on creating server REST endpoints, with `@whitefang_jr` suggesting the use of create-llama from LlamaIndex's documentation.

**Links mentioned**:

- [Node Parser Modules - LlamaIndex 🦙 0.9.40](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules.html#codesplitter): no description found
- [RAG over code | 🦜️🔗 Langchain](https://js.langchain.com/docs/use_cases/rag/code_understanding): Use case
- [llama_index/llama_index/vector_stores/postgres.py at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/llama_index/vector_stores/postgres.py): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [Starter Tutorial - LlamaIndex 🦙 0.9.40](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html): no description found
- [Redis Docstore+Index Store Demo - LlamaIndex 🦙 0.9.40](https://docs.llamaindex.ai/en/stable/examples/docstore/RedisDocstoreIndexStoreDemo.html): no description found
- [Full-Stack Projects - LlamaIndex 🦙 0.9.40](https://docs.llamaindex.ai/en/stable/community/full_stack_projects.html#create-llama): no description found
- [Chat Engine - LlamaIndex 🦙 0.9.40](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/root.html#concept): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/) (1 messages): 

leonms123: Hi would anyone be willing to teach me ML using python 😄
  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1202183040529596446) (10 messages🔥): 

- **A Good Morning Wave**: `@andrei.alcaza` greeted the channel with a simple but friendly "morning" followed by a waving hand emoji.
- **Searching for Visuals**: `@dpaleka` requested assistance in finding a good **transformer decoder tikz** or a reference to an **arXiv paper** containing one.
- **Leaked Model Drama Unfolds**: `@hailey_schoelkopf` shared a [tweet from @arthurmensch](https://fxtwitter.com/arthurmensch/status/1752737462663684344) discussing an incident where an "over-enthusiastic employee" leaked a watermarked version of an old training model.
- **Cultural Interpretation**: In response to the use of "over-enthusiastic" by `@arthurmensch`, `@carsonpoole` expressed amusement, which led to `@catboy_slim_` comparing it humorously to the Southern expression "bless your heart."
- **Seeking Diffusion Model Leaderboards**: `@carmocca` inquired about the existence of a leaderboard similar to HF's eval leaderboard, but for **diffusion models**.

**Links mentioned**:

[Tweet from Arthur Mensch (@arthurmensch)](https://fxtwitter.com/arthurmensch/status/1752737462663684344): An over-enthusiastic employee of one of our early access customers leaked a quantised (and watermarked) version of an old model we trained and distributed quite openly.  To quickly start working with ...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1202195498971177000) (14 messages🔥): 

- **Intriguing Scale Effects Uncovered in Diverse Model Architectures**: `@the_random_lurker` highlighted a study ([Scaling Properties of Transformer Models](https://arxiv.org/abs/2207.10551)) revealing that model architecture is an important scaling consideration and performance may fluctuate at different scales. They questioned why this finding hasn't become mainstream.

- **Continued Pretraining of Mixtral Inquiry**: `@quicksort` asked for any papers, repos, or blog posts about the continuous pretraining of MoE models, similar to LeoLM, for low-resource languages. This indicates interest in refining methods for language model pretraining.

- **Contradictions in ImageNet Pre-training**: `@micpie` shared research ([Pre-training Time Matters](https://arxiv.org/abs/2203.04668)) exposing that models inadequately pre-trained on ImageNet can sometimes outperform fully trained models depending on the task. This points to a complex relationship between pre-training duration and model efficiency in different applications.

- **Seeking Insights on Gated Convolution Architectures**: `@afcruzs` requested resources for comparing gated convolutional architectures, with `@mrgonao` expressing similar interest. Meanwhile, `@nostalgiahurts` provided a [GitHub repository](https://github.com/srush/do-we-need-attention/) and a [blog post](https://benathi.github.io/blogs/2023-12/global-convolution-models/) that might shed light on these architectures.

- **Token Discretization During Inference Discussed**: `@sentialx` inquired about the necessity of discretizing tokens during inference as opposed to feeding back the token embeddings, with `@xylthixlm` responding it's due to the requirement of autoregressive generation during training. This reflects on typical practices in language model inference processes.

**Links mentioned**:

- [Scaling Laws vs Model Architectures: How does Inductive Bias Influence Scaling?](https://arxiv.org/abs/2207.10551): There have been a lot of interest in the scaling properties of Transformer models. However, not much has been done on the front of investigating the effect of scaling properties of different inductive...
- [Towards Inadequately Pre-trained Models in Transfer Learning](https://arxiv.org/abs/2203.04668): Pre-training has been a popular learning paradigm in deep learning era, especially in annotation-insufficient scenario. Better ImageNet pre-trained models have been demonstrated, from the perspective ...
- [GitHub - srush/do-we-need-attention](https://github.com/srush/do-we-need-attention/#do-we-need-attention): Contribute to srush/do-we-need-attention development by creating an account on GitHub.
- [The Essense of Global Convolution Models | AI Bytes](https://benathi.github.io/blogs/2023-12/global-convolution-models/): no description found

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1202490331744960512) (3 messages): 

- **The Renaissance of n-gram Models**: `@80melon` shared a [research paper](https://arxiv.org/abs/2401.17377) that revisits n-gram language models, highlighting a novel $\\infty$-gram model and a new computation engine called *infini-gram*. They expressed surprise at how these n-gram models, when combined with large neural models like Llama-2 70b, significantly improve perplexity scores.
- **Generalization vs. Memorization**: Despite the improvements in perplexity, `@80melon` speculated that using adaptive-length n-gram models might worsen the generalization capability of neural language models, though this was not tested in the study.

- **Understanding Transformers' Memory Mechanisms**: `@ishitatsuyuki` raised a question about existing research on transformers' ability to memorize n-grams, referring to a study that uses automata (found [here](https://arxiv.org/abs/2210.10749)). They are curious whether the resulting automata would be compact enough to represent the extensive vocabularies.

**Links mentioned**:

[Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377): Are n-gram language models still relevant in this era of neural large language models (LLMs)? Our answer is yes, and we show their values in both text analysis and improving neural LLMs. Yet this nece...

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1202257466285641779) (12 messages🔥): 

- **Quick Appreciation for PyPI Automation**: `@hailey_schoelkopf` expressed thanks to `@1186960329738039357` for their pull request which automated packaging for PyPI, stating it worked like a charm.
- **Language Model Evaluation Harness 0.4.1 Released**: `@hailey_schoelkopf` shared a [PyPI release](https://pypi.org/project/lm-eval/0.4.1/) and a different [GitHub Release link](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.1) for the **Language Model Evaluation Harness 0.4.1** which includes internal refactoring, easier prompt design with Jinja2, and optimized data-parallel model usage.
- **PR Gratitude and Pending Review**: `@anjor_20331` expressed gratitude for the successful automation and acknowledged an open PR that they will review once they are less swamped.
- **Interest in Few-shot Examples Log Output**: `@Goyim` inquired about including few-shot examples in the `log_samples` output or knowing which ones will be used ahead of time.
- **Clarification on MMLU Evaluation Metrics**: `@daniellepintz` asked for help interpreting evaluation results from MMLU, `@stellaathena` clarified the accuracy as 40.2% and standard error as +/- 20.7%, and `@baber_` questioned whether the results were calculated with a limitation parameter. `@daniellepintz` shared a [gist](https://gist.github.com/daniellepintz/c48c9e61a9a4798552b6ac22bc3a1959) suspecting a problem with their LM subclass implementation.

**Links mentioned**:

- [gist:c48c9e61a9a4798552b6ac22bc3a1959](https://gist.github.com/daniellepintz/c48c9e61a9a4798552b6ac22bc3a1959): GitHub Gist: instantly share code, notes, and snippets.
- [lm-eval](https://pypi.org/project/lm-eval/0.4.1/): A framework for evaluating language models
- [Release v0.4.1 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.1): Release Notes This PR release contains all changes so far since the release of v0.4.0 , and is partially a test of our release automation, provided by @anjor . At a high level, some of the changes ...

  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1202349080232067073) (4 messages): 

- **Uncertainty in Compute Costs for Training VQA Systems**: `@stellaathena` asked for guidance or a rule of thumb on the compute cost to train a decent image encoder and LLM for Visual Question Answering (VQA), acknowledging the lack of good numbers available.
- **Seeking Consensus on Encoder-Decoder Models**: `@stellaathena` also inquired if the consensus is still that encoder-decoder models are preferable for text and image to text (T+I -> T) domain.
- **Request for Clarity in Model Application**: `@kublaikhan1` responded to `@stellaathena`'s query about encoder-decoder models for T+I -> T, seeking clarification on the application or context in question.
- **Insights on Training llava**: In connection to the VQA system discussion, `@kublaikhan1` mentioned that it takes about 8x24 A100 hours to train llava, a type of model suitable for such tasks.
  

---



### CUDA MODE (Mark Saroufim) ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1202211517312028733) (17 messages🔥): 

- **CUDA Development with Python Insights**: `@neuralution` highlighted the benefits of Python for CUDA development, prompting a discussion on performance optimization and compatibility with NVIDIA tools. They raised specific concerns such as ensuring good occupancy, identifying bottlenecks, and cache usage when integrating PyTorch types into the C++ output binary.
  
- **Flash Attention 2 PyTorch Tutorial**: `@iron_bound` shared a [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) by **Driss Guessous** on **Flash Attention 2**, with a link for a detailed example code and additional background on **scaled_dot_product_attention**.

- **NVIDIA's New Multi-Query Attention Kernel**: `@dshah3` introduced NVIDIA's open-sourced [XQA kernel](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md) for multi-query and grouped-query attention, boasting improvements in Llama 70B throughput without increased latency.

- **Open Dialogue for PyTorch Attention Questions**: `@drisspg` invited queries about the PyTorch 2.2 **Flash Attention 2** implementation, and offered insights on upcoming updates, comparing it with Tri's FA-2 version from his repository.

- **Nested Tensors Becoming an Official Feature**: `@andreaskoepf` and `@jeremyhoward` discussed the utility and official status of NestedTensors in PyTorch, and `@tvi_` advised caution when replacing packing code with torch.compile that may not fully support NestedTensors yet.

**Links mentioned**:

- [(Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA) — PyTorch Tutorials 2.2.0+cu121 documentation](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html): no description found
- [NVIDIA Gen AI on RTX PCs Developer Contest](https://www.nvidia.com/en-us/ai-data-science/generative-ai/rtx-developer-contest/?ncid=em-anno-686161&mkt_tok=MTU2LU9GTi03NDIAAAGRAWA_3nFf_Xif6h6qd7k4EIrTVe7djWGnOtUuffYhxfFo8XfwEpZybo2TX4ocKfDaQ-sTe-q1D3AWPRls560AscVrpI-HNcR1Qtn6Vj7CsEeyA1gmpcA): Enter to win a GeForce RTX 4090 GPU, a GTC event pass, and more.
- [TensorRT-LLM/docs/source/blogs/XQA-kernel.md at main · NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/XQA-kernel.md#llama-70b-on-h200-up-to-24x-increased-throughput-with-xqa-within-same-latency-budget): TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficie...
- [[RFC] Scaled Dot Product Attention  API Changes · Issue #110681 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/110681): Updated SDPA API Authors: @drisspg Summary In order for users to more easily manage the complexity handling of various bias formats we would like to expose the ability to pass in AttnBias derived c...
- [pytorch/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L306): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [torch.nn.attention.sdpa_kernel &mdash; PyTorch main documentation](https://pytorch.org/docs/main/generated/torch.nn.attention.sdpa_kernel.html#torch.nn.attention.sdpa_kernel): no description found
- [(Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA) — PyTorch Tutorials 2.2.0+cu121 documentation](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html#explicit-dispatcher-control): no description found

  

---


### CUDA MODE (Mark Saroufim) ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/) (1 messages): 

iloveh8: Got it thank you
  

---


### CUDA MODE (Mark Saroufim) ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1202299013353701376) (7 messages): 

- **Jeremy's Notebook Errors Solved**: User `@arsalan6990` encountered an error while trying to follow Jeremy's notebook. Thanks to `@drisspg`, a solution was found by referencing GitHub issues, specifically one involving a mismatch between the CUDA version and the version used to compile PyTorch ([CUDA 12.0 Issue](https://github.com/qwopqwop200/GPTQ-for-LLaMa/issues/7#issuecomment-1465246457)) and a template-related error with CUDA 12.1 ([Cuda 12.1 Template Issue](https://github.com/pybind/pybind11/issues/4606#issuecomment-1498131493)). The fix required installing g++ and gcc 11 and ensuring GLIBCXX_3.4.32 was available.
- **Cost-Efficiency Tip for Computing Resources**: `@nshepperd` suggested that having a lot of CPU RAM is beneficial since it is much cheaper than GPU RAM, costing about 1/10 per GB in comparison.
- **C++ Templates Clarification**: In the context of error resolution, `@zippika` explained the use of `typename` in C++ templates, detailing that it is a placeholder for substituted types within template definitions.

**Links mentioned**:

- [Does not compile on CUDA 12.0 · Issue #7 · qwopqwop200/GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/issues/7#issuecomment-1465246457.): On running the setup_cuda.py install, I was initially getting: RuntimeError: The detected CUDA version (12.0) mismatches the version that was used to compile PyTorch (11.8). Please make sure to use...
- [[BUG]: Cuda 12.1: error: expected template-name before ‘&lt;’ token · Issue #4606 · pybind/pybind11](https://github.com/pybind/pybind11/issues/4606#issuecomment-1498131493): Required prerequisites Make sure you&#39;ve read the documentation. Your issue may be addressed there. Search the issue tracker and Discussions to verify that this hasn&#39;t already been reported. +1...

  

---


### CUDA MODE (Mark Saroufim) ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1202343518958260264) (1 messages): 

- **Selective Compilation Query**: User `@marvelousmit` inquired whether it is possible to compile only the model layers in PyTorch, excluding a custom operator that gets called at runtime, which is not meant to be compiled. They provided a snippet showing a `forward()` method calling `func_not_compile()` followed by `layers_compile()` and asked about using something like `torch.compile(model)`.
  

---


### CUDA MODE (Mark Saroufim) ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1202193384983506964) (12 messages🔥): 

- **CUDA Compatibility Confirmation**: `@noobpeen` asked if CUDA 11.8 is fine for use, since it was already installed for deep learning. `@lancerts` confirmed it should be okay, providing a [link to NVIDIA documentation](https://docs.nvidia.com/nsight-visual-studio-edition/3.2/Content/Debugging_CUDA_Application.htm) which details CUDA Data Stack limitations and functions like `cuCtxGetLimit()` and `cuCtxSetLimit()`.
- **Debugging CUDA in VSCode**: `@andreaskoepf` pointed out that **CUDA Debugging in VSCode** is possible, though the setup process is slightly complex, and shared a [link to the relevant NVIDIA documentation](http://docs.nvidia.com/nsight-visual-studio-code-edition/cuda-debugger/index.html).
- **CUDA Kernel Debugging Inquiry**: `@jeremyhoward` queried about debugging CUDA kernels called from PyTorch, especially through just-in-time (jit) extensions. No direct solutions were provided within these messages.
- **Approach to Kernel Development**: `@andreaskoepf` describes their development process, starting with a PyTorch reference implementation and then creating a custom kernel to pass the `torch.allclose()` test. This was echoed by `@marksaroufim`, who has observed similar practices at PyTorch.
- **Test Framework Preferences**: `@andreaskoepf` expressed a preference for using raw Python files or Jupyter notebooks over pytest during development, citing the added complexity of pytest. `@marksaroufim` humorously noted the satisfaction from seeing passing tests marked by dots and green colors, whereas `@andreaskoepf` joked about getting random patterns of failures (F) and passes (dots) and the occasional pytest crash.

**Links mentioned**:

- [Walkthrough: Debugging a CUDA Application](https://docs.nvidia.com/nsight-visual-studio-edition/3.2/Content/Debugging_CUDA_Application.htm): no description found
- [Getting Started with the CUDA Debugger :: NVIDIA Nsight VSCE Documentation](http://docs.nvidia.com/nsight-visual-studio-code-edition/cuda-debugger/index.html): no description found

  

---


### CUDA MODE (Mark Saroufim) ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1202293217924759652) (3 messages): 

- **CUDA Block Size Matching**: `@lancerts` inquired about the necessity for block size to match `tile_width`, especially when a configuration with block dimensions `dim3(32,32,1)` runs without errors despite `tile_width` being 16. `@vim410` warned that while no immediate errors may be observable, without bounds checking, one could run into illegal memory errors, and stated that aligning more threads than needed for the work is generally illogical.
- **Understanding Thread Coarsening in CUDA**: `@lancerts` sought clarification on thread coarsening code specifics, questioning the appropriate dimensions for `threadsPerBlock` and `numBlocks`. The inquiry contrasts a coarsened version with a normal tiled version that uses rectangular thread block shapes, illustrating a common confusion among developers when it comes to optimizing thread allocation in CUDA programming.
  

---



### DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1202176779310346271) (21 messages🔥): 

- **Mistral Medium's Identity Crisis**: `@sebastian.bodza` suggested that Mistral Medium could be based on Llama 70B, referencing a [Twitter post](https://twitter.com/lgrammel/status/1751921254406185308) and noting the closeness of the answers.
- **Leaked Model Conspiracies Abound**: Users `@sebastian.bodza` and `@devnull0` entertained the theory that Miqu is actually Mistral quantized, supported by discussions on Twitter, including a tweet from [@sroecker](https://fxtwitter.com/sroecker/status/1752460995824349563?t=nZXA3oDFnDR6MoWUWwdYyA&s=19) and another from [@teortaxesTex](https://fxtwitter.com/teortaxesTex/status/1752673893276356608?t=4SMqTI_BCx8NTjmU3LoYXA&s=19) drawing parallels between Mistral and Miqu outputs.
- **Mistral's Quest for Monetization**: `@sebastian.bodza` expressed disappointment that Mistral's effort to offer great models might be undermined by cheaper leaked versions like Mixtral, while `@devnull0` questioned if they were really in it for the money.
- **Debating Model Watermarking**: Amid speculation about watermarking to identify leaked models, `@philipmay` expressed skepticism, while `@mariowolframm` and `@devnull0` discussed the potential of unique data combinations or tokens acting as watermarks surviving through quantization processes.

**Links mentioned**:

- [Tweet from Q (@qtnx_)](https://fxtwitter.com/qtnx_/status/1751986395634098273?s=20): reminder that mistral was training 70Bs https://techcrunch.com/2023/11/09/theres-something-going-on-with-ai-startups-in-france/
- [Tweet from Teortaxes▶️ (@teortaxesTex)](https://fxtwitter.com/teortaxesTex/status/1752673893276356608?t=4SMqTI_BCx8NTjmU3LoYXA&s=19): The smoking gun isn&#39;t the aggregate r, it&#39;s how a great share of items land on the trendline perfectly, suggesting identical cognitive circuitry. This is a typical EQ-Bench item (https://arxiv...
- [Tweet from Steffen Röcker (@sroecker)](https://fxtwitter.com/sroecker/status/1752460995824349563?t=nZXA3oDFnDR6MoWUWwdYyA&s=19): I want to believe: MIQU = MIstral QUantized 🛸  ↘️ Quoting Teortaxes▶️ (@teortaxesTex)   Might be late but I am now 100% convinced that Miqu is the same model that&#39;s accessible as Mistral-Medium o...

  

---


### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1202206629152702464) (10 messages🔥): 

- **Hermes 2 Dataset Unleashed**: `@teknium` has shared their release of the **Hermes 2 dataset** with the community, accessible through [Twitter](https://twitter.com/Teknium1/status/1752799124775374928). The dataset might be valuable for those interested in AI research and development.
- **Community Love for the Dataset**: In response to `@teknium`'s post, `@hammadkhan` expressed gratitude for the Hermes 2 dataset release with a heartfelt thank you emoji.
- **Practical Applause for Lilac Integration**: `@devnull0` complemented the lilac integration in the Hermes 2 dataset, indicating a well-received feature within the community.
- **Apple Engineer's Crafting Conundrums Presented**: `@devnull0` shared a humorous prompt regarding the effort required by Apple engineers, found in a tweet by `@cto_junior` on [Twitter](https://fxtwitter.com/cto_junior/status/1752284772196315617).
- **Mixtral's Impressive Performance Metrics Shared**: `@bjoernp` highlighted the surprising speed of Mixtral, reaching 500 tokens per second, with a link to [Groq's chat platform](https://chat.groq.com/), sparking curiosity and a follow-up question by `@sebastian.bodza` on the company's use of custom AI accelerators.

**Links mentioned**:

- [GroqChat](https://chat.groq.com/): no description found
- [Tweet from Stella Biderman (@BlancheMinerva)](https://fxtwitter.com/BlancheMinerva/status/1752820474222960969?t=DkJDppyFgUKF_aElQIFoog&s=19): @Dorialexander This is completely consistent with research on crosslingual instruction-tuning https://arxiv.org/abs/2211.01786 as well. I&#39;m pretty sure there&#39;s lit on linear maps between multi...
- [Tweet from TDM (e/λ) (@cto_junior)](https://fxtwitter.com/cto_junior/status/1752284772196315617?t=5HJqM3g3Vny2bM290StYpA&s=19): How many apple engineers were required to craft this prompt?

  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1202253485853392936) (8 messages🔥): 

- **Mistral Embeddings Overfitting Issue**: `@devnull0` shared a [tweet](https://fxtwitter.com/Nils_Reimers/status/1752473576416911622?t=u6_C6owd2PRWz2knX2oM6A&s=19) from `@Nils_Reimers`, claiming that **Mistral 7B embedding models** are heavily overfitted on MTEB and perform poorly on other tasks than what they were trained for, specifically calling out their inadequacy in tasks like movie sentiment classification.

- **Microsoft's Novel Approach for Text Embeddings**: `@sebastian.bodza` referenced a [research paper](https://arxiv.org/html/2401.00368v2) by researchers from Microsoft Corporation, highlighting their novel method in generating high-quality text embeddings using synthetic data, simplifying the training process, and achieving significant language coverage.

- **Skepticism Abounds**: Despite sharing the research paper, `@sebastian.bodza` remained **not 100% sold** on the paper’s approach, expressing a level of skepticism toward the methodology presented.

- **Generating Relevant Passages for Retrieval Tasks**: In a discussion with `@thewindmom`, `@sebastian.bodza` provided a tip that for short to long retrieval tasks, it's necessary to prepend the prompt with "Represent this sentence for searching relevant passages:" to obtain better results.

**Links mentioned**:

- [Tweet from Nils Reimers (@Nils_Reimers)](https://fxtwitter.com/Nils_Reimers/status/1752473576416911622?t=u6_C6owd2PRWz2knX2oM6A&s=19): @abacaj Not worth it. These Mistral 7B embedding models are heavily overfitted on MTEB, by generating training data for the all the tasks in MTEB, and then training a 7B model to do e.g. movie sentime...
- [Improving Text Embeddings with Large Language Models](https://arxiv.org/html/2401.00368v2): no description found

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/) (1 messages): 

philipmay: How did you generate this plot for DiscoResearch/DiscoLM_German_7b_v1 ?
  

---



### LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1202352474997145670) (2 messages): 

- **Prompt Investing 101**: User `@jeffreyw128` inquired about **prompt investing**.
- **A Quick Correction**: User `@jxnlco` corrected the term to **injecting**, potentially clarifying the discussion topic.
  

---


### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/1202352530164551680) (3 messages): 

- **First in Line to Try**: `@jeffreyw128` inquired about the best place to try a new model, and `@thebaghdaddy` responded with a link to [Hugging Face's Miqu-1 70B model](https://huggingface.co/miqudev/miqu-1-70b), indicating it's the first in a potential series.
- **Prompt Formatting Advice**: `@thebaghdaddy` included instructions on the **prompt format for the Mistral** model and cautioned against changing ROPE settings because this model uses a high-frequency base with 32k seen tokens.
- **Model Testing Constraints**: `@thebaghdaddy` expresses a limitation by stating, "im gpu poor though so i havent tested it," indicating they have not personally tested the model due to lack of GPU resources.

**Links mentioned**:

[miqudev/miqu-1-70b · Hugging Face](https://huggingface.co/miqudev/miqu-1-70b): no description found

  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1202372247726329946) (5 messages): 

- **Seeking Truth in Model Capabilities**: `@joshcho_` questioned the validity of a [tweet from @nickadobos](https://twitter.com/nickadobos/status/1749837866300264529?s=46&t=6XxQ29Eas6j8_g5OIJcaEA) regarding AI performance. No specific details about the AI or tweet content were provided in the question.
  
- **Disappointment in Document Understanding**: `@jeffreyw128` shared **negative experiences** with document performance, expressing they have "no idea how to get performance out of them."

- **A Moment of Resignation**: `@joshcho_` responded to the difficulty in leveraging document capabilities with "sad" and a whimsical comment, "we just pray."

- **Insight on AI Limitations with Image Texts**: `@thebaghdaddy` expressed surprise and seemed to confirm an AI limitation, stating it "explains why it is incapable of understanding knowledge files with pictures in them."
  

---



### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1202306693879582802) (5 messages): 

- **Crowdfunding for Open Source AI Research**: `@filipvv` proposed the idea of crowdfunding to raise funds for open-source models, fine-tunes, and datasets. They shared examples of projects like [CryoDAO](https://www.cryodao.org/) and [MoonDAO](https://www.moondao.com/), and mentioned a bias due to working on a platform [Juicebox](https://juicebox.money/) that facilitates such crypto projects.

- **Brainstorming Collective Funding for OS Training**: Continuing the discussion, `@filipvv` explained that the aggregation of funds could pay for larger training runs that could benefit the entire community. The goal is to provide more funding for open-source AI projects, including those on Nous and other platforms.

- **Hermes 2 Dataset Released**: `@teknium` announced the release of their Hermes 2 dataset, offering it to the community for potential use. The dataset can be found in their [tweet](https://twitter.com/Teknium1/status/1752799124775374928).

- **HelixNet Architecture Introduced**: `@yikesawjeez` shared a post from `@migtissera` announcing HelixNet, a novel deep learning architecture with 3 Mistral-7B LLMs designed akin to a DNA helix structure. They included a link to the model on [Hugging Face](https://huggingface.co/migtissera/HelixNet) and provided access details for testing via a Jupyter Hub at [agentartificial.com](https://jupyter.agentartificial.com/commune-8xa6000/user/forthepeople/lab) with the credentials: user - forthepeople, pw - getitdone.

**Links mentioned**:

- [Tweet from Migel Tissera (@migtissera)](https://x.com/migtissera/status/1720567034315186588?s=20): It&#39;s been a big week for Open Source AI, and here&#39;s one more to cap the week off!  Introducing HelixNet.  HelixNet is a novel Deep Learning architecture consisting of 3 x Mistral-7B LLMs. It h...
- [Tweet from yikes (@yikesawjeez)](https://x.com/yikesawjeez/status/1752808327728537682?s=20): yknow what fk it https://jupyter.agentartificial.com/commune-8xa6000/user/forthepeople/lab user: forthepeople pw: getitdone  will start downloading the model now, hop in and save notebooks to the ./wo...

  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1202275940370235423) (5 messages): 

- **Exploring Hugging Face's Transformers Agents**: User `@tonic_1` linked to the documentation for the [Transformers Agents API](https://huggingface.co/docs/transformers/main_classes/agent) on Hugging Face, noting that it's experimental and could be prone to change. They highlighted the existence of three types of agents (**HfAgent**, **LocalAgent**, and **OpenAiAgent**) for a range of uses from open-source models to local and closed models from OpenAI.
- **Query for Clarification**: `@hackgoofer` asked for an explanation on [HFAgents](https://huggingface.co/docs/transformers/main_classes/agent), indicating a lack of understanding about the topic shared by `@tonic_1`.
- **Quick Pitch-In Offer**: `@tonic_1` mentioned they hadn't filled out a form but expressed affection for the community and a desire to contribute.

**Links mentioned**:

[Agents &amp; Tools](https://huggingface.co/docs/transformers/main_classes/agent): no description found

  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1202399861706723459) (2 messages): 

- **Creativity in Gaming**: User `@magusartstudios` mentioned developing a **Roblox AI agent Plugin** utilizing various tools and features.
- **Clarifying OpenAI’s Token Policy**: `@metaldragon01` pointed out that **OpenAI does not provide free tokens** to the public for open models.
  

---


### Alignment Lab AI ▷ #[alignment-lab-announcements](https://discord.com/channels/1087862276448595968/1124055853218136175/1202362433038188604) (1 messages): 

- **Open Hermes 2.5 Dataset Unveiled**: `@teknium` announced the public release of the **Open Hermes 2.5** and **Nous-Hermes 2** dataset, a comprehensive collection used to improve SOTA LLM's, boasting over 1 million examples. The dataset is a mix of open source data and synthetic datasets, which can be accessed on [HuggingFace](https://huggingface.co/datasets/teknium/OpenHermes-2.5).
- **Acknowledgments for Dataset Contributions**: The announcement thanked members of the Discord community who contributed to the dataset, including `<@1110466046500020325>`, `<@257999024458563585>`, `<@748528982034612226>`, `<@1124158608582647919>`.
- **Collaboration with Lilac ML**: `@teknium` also highlighted a collaboration with Nikhil and Lilac ML to integrate Hermes into their [HuggingFace Spaces](https://lilacai-lilac.hf.space/datasets#lilac/OpenHermes-2.5), enhancing the ability to explore and analyze the dataset.

**Links mentioned**:

- [teknium/OpenHermes-2.5 · Datasets at Hugging Face](https://huggingface.co/datasets/teknium/OpenHermes-2.5): no description found
- [no title found](https://lilacai-lilac.hf.space/datasets#lilac/OpenHermes-2.5): no description found

  

---



### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1202479369759756399) (1 messages): 

- **Inquiry on DatasetteGPT Utility**: User `@discoureur` raised a question about whether anyone has set up a **DatasetteGPT** for processes such as remembering configuration steps or aiding in plugin writing for Datasette's documentation.
  

---




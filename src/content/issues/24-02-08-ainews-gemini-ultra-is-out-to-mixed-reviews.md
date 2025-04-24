---
id: d33c7c5b-7cb7-49e7-97e3-b27de96a449f
title: Gemini Ultra is out, to mixed reviews
date: '2024-02-09T05:58:08.478444Z'
original_slug: ainews-gemini-ultra-is-out-to-mixed-reviews
description: >-
  **Google** released **Gemini Ultra 1.0** as a paid tier following the
  discontinuation of Bard, with early reviews noting modest improvements over
  ChatGPT but also identifying reasoning gaps. Meanwhile, the **Nous Research**
  community highlighted **BiLLM**, a new large language model achieving
  significant efficiency gains through one-bit post-training quantization, and
  discussed advancements in transformer efficiency with the **Subformer** model.
tags:
  - untagged
companies: []
models: []
topics:
  - untagged
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/7/2024. We checked **20** guilds, **311** channels, and **6128** messages for you. Estimated reading time saved (at 200wpm): **496 minutes**.

Business as usual, it's been pretty quiet overall in AI. With Bard well and truly dead, [Gemini Ultra was released today](https://blog.google/products/gemini/bard-gemini-advanced-app/) as a paid tier for "Gemini Advanced with Ultra 1.0". The reviews industrial complex is getting to work:

- [Fireship](https://www.youtube.com/watch?v=ucd63nIZZ60) was only very lightly complimentary, saying it is "slightly faster/better than ChatGPT", but identifying a bunch of gaps.
- [AI Explained](https://www.youtube.com/watch?v=gexI6Ai3X0U) also commented on the higher speed, but found a few reasoning and visual reasoning gaps.

 ![image.png](https://assets.buttondown.email/images/18b262e6-69df-40a9-abf9-3f5d41bca9ab.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/45c5fe91-f014-4bf9-b5bc-06c3423ecf7f.png?w=960&fit=max)







---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Steam Deck: An Unexpected AI Powerhouse**: A user noted the Steam Deck's potential for running AI models, where models like Solar 10.7B show significant performance, suggesting an unexpected use-case for the handheld gaming device as a portable AI workstation.

- **AI Philosophy and Mathematics Intersect**: Ongoing discussions delve into AI's capacity to "understand" akin to humans and the existence of mathematics in relation to the physical universe, sparking profound debates on metaphysics and the nature of consciousness.

- **New Directions in LLM Optimization and Training Data Contamination**: Efforts to add multi-GPU support to OSS Unsloth were discussed, with the benefits of pre-quantization notably reducing VRAM requirements by 1GB without accuracy loss. It was also highlighted that most modern models are probably trained with data that has been influenced by outputs from OpenAI's models, potentially affecting their unique style.

- **Emerging Concerns Over Model Merging and Dataset Value**: Conversations around the ethics and practicalities of model merging included proposals for "do not merge" licenses and the financial challenges of creating datasets versus model mergers. These issues underscore the complex interplay between innovation, ownership, and the freely available nature of AI research.

- **Model Alignment and Training Innovations Unfold**: The introduction of Listwise Preference Optimization (LiPO) as a new framework for aligning language models suggests a shift towards more refined response optimization techniques. Moreover, the LiPO framework utilizes rankers that are small enough to be trained locally, a significant advantage for those with limited resources.

- **Coding Highlights: Implementation Insights and Language Advancements**: Practical advice was shared for using Hugging Face models effectively, while the Mojo language was discussed for its impressive performance, promising to be a noteworthy tool for high-performance computing tasks. Additionally, the integration of database functions into bot interactions for enhanced conversational abilities was explored, emphasizing growth in bot sophistication.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Efficiency in Transformers with SAFE**: An [arXiv paper](https://arxiv.org/abs/2101.00234) was discussed, presenting the **Subformer**, a model applying sandwich-style parameter sharing and self-attentive embedding factorization (SAFE) to achieve better results with fewer parameters than a traditional Transformer.

- **BiLLM's Breakthrough with One-Bit Quantization**: A significant reduction in computational and memory requirements while maintaining performance is claimed by **BiLLM**, introduced through a paper ([Download PDF](/pdf/2402.04291.pdf)) focusing on 1-bit post-training quantization for large language models.

- **OpenHermes Dataset Viewer Now Available**: A new tool developed by `@carsonpoole` to aid in viewing and analyzing the **OpenHermes** dataset was shared, with features such as scrolling through examples and filters for token count analytics.

- **GPUs and Scheduling**: Slurm was recommended in a discussion for efficiently scheduling jobs on GPUs, with members including `@chrisj7746`, `@Sebastian`, and `@leontello` contributing to the conversation on best practices.

- **Fostering Model Competency in Specific Tasks**: There's an ongoing quest for improving specific aspects of AI model performance, such as **custom pretrained models** struggling with extraction tasks and inquiries about fine-tuning parameters for models like **OpenHermes-2.5-Mistral-7B**. Users also discussed the sufficiency of 8GB VRAM for fine-tuning operations and setting special tokens in a dynamic `.yml` configuration.

- **Model Architecture and Quantization Progress**: Architectural changes post-GPT-4 with potential Turing complete modifications were top of mind, and quantization's role in future-proofing models was underscored by community engagement in methodology discussions.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **LaTeX Lament in LM Studio**: `.neurorotic.` and others voiced frustrations with the improper rendering of LaTeX, specifically with DeepSeek Math RL 7B LLM in LM Studio.
- **Exploring Ideal LLM Hardware Configurations**: Within the hardware-focused discussions, an ideal setup for running LLMs locally remains an inquiry, with specific interest shown in the compatibility of the AMD Ryzen 5 7600 and AMD Radeon RX 6700 XT configurations. PCIe risers and their effect on performance was another hot topic.
- **Mixed Perceptions of Emerging Language Models**: Intense debate occurred surrounding the performance and expectations of models like Qwen 1.5, Code Llama 2, and various OpenAI models. Key topics included context understanding, GPU acceleration, and code generation quality. GPT-3.5 usage was confirmed to be satisfactory in an Open Interpreter (OI) setup.
- **ESP32 S3 Explored for DIY Voice Projects**: Community members, notably `@joelthebuilder`, engaged in discussions about using ESP32 S3 Box 3 for custom home network voice projects aiming to replace standard solutions like Alexa.
- **Open Interpreter on Discord and LMStudio**: The potential of Open Interpreter to leverage LMStudio's server mode was highlighted, with a recommendation to check the [OI Discord](https://github.com/KillianLucas/open-interpreter/) for specific use-case discussions and Discord integration advice.
- **Autogen Issues in LM Studio**: `@photo_mp` reported userproxy communication issues within autogen and sought community recommendations for a solution.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Socrates AI Memes Still Trending**: `@gabriel_syme` humorously postulated that AI models emulating Socrates might lead to one-sided dialogues, mimicking the philosopher's dominance in conversations.
- **Llama Safety Locks for Less**: The security of the **Llama Model** can be compromised with **LoRA fine-tuning** for under $200, as per a [LessWrong post](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from?utm_source=ainews&utm_medium=email), raising concerns about the simplicity of bypassing safety protocols.
- **Silent Cloning and Whispers of GPT-Blind**: Speculations surfaced regarding the evolution of voice cloning technology and the emergence of a new, unspecified GPT model, indicating significant shifts in the technical landscape and potential challenges for existing hardware like iPhone displays.
- **LLM Paper Club in Full Swing**: A **Latent Space Paper Club** session discussed *Self-Rewarding Language Models*, with one paper demonstrating models capable of using output for rewards, outshining Claude 2 and GPT-4 in certain tests. Read it [here](https://arxiv.org/abs/2401.10020).
- **DSPy Draws a Crowd for Next Club Meeting**: DSPy, a model for chaining LLMs, has caught the club’s interest. An upcoming session will explore its capabilities, and an insightful YouTube video about it can be viewed [here](https://www.youtube.com/watch?v=rqR3LeR09gc&t=903s&ab_channel=code_your_own_AI).



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Mistral in the Realm of Healthcare**: `@talhawaqas` searched for resources on **Mistral.AI's applications in health engineering**—specifically around pretraining and finetuning—while others in the community considered transitioning their tests to the cloud for convenience.
- **Data Policies and Creative Commons**: A clarification was provided by `@mrdragonfox` that user dialogue data collected by the service may be released under a **Creative Commons Attribution (CC-BY)** license.
- **Chat Bot Refinements and Parameters**: Community members discussed various strategies for setting `max_tokens` and temperature in Mistral for concise bot responses, and debated the use of a temperature of zero to minimize hallucinations and improve performance for precision tasks.
- **Embedding Models Synced with Mistral**: `@gbourdin` found the **E5-mistral-7b-instruct model** on Hugging Face, which aligns with **Mistral-embed**'s dimension length for local development, and `@mrdragonfox` provided usage instructions [E5-mistral-7b-instruct · Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct).
- **Introducing Tools and Showcasing Interfaces**: The Discord saw an introduction of new tools and user interfaces—one being *augmentoolkit* adapted for the Mistral chat API, and another **ETHUX Chat v0.7.0**, which includes the Huggingface ChatUI and web search features, with the chat UI available on [GitHub](https://github.com/huggingface/chat-ui).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **New Job Forum Launches at HuggingFace**: A new **Forum** channel has been established for job opportunities, allowing users to filter vacancies by tags like `internship`, `machine learning`, `remote`, and `long-term`.

- **GPU and Diffusion Model Discussions Heat Up**: `@sandeep_kr24` explained that `pipeline.to("cuda")` transfers the computational graph to a CUDA GPU, speeding up the process. Updates on Stable Diffusion and support inquiries were addressed, including **training models for pixel art** and setting up web UI for LLM querying.

- **SD.Next Introduces Performance Boost and New Modules**: `@vladmandic` announced an update to [SD.Next](https://github.com/vladmandic/automatic), featuring a new **Control module**, **Face module**, **IPAdapter** improvements, and an array of models, claiming a benchmark of **110-150 iterations/second on an nVidia RTX4090**.

- **RAG Pipeline Relevance and Deci lm Inference Speeds in Focus**: `@tepes_dracula` sought a **classifier** for validating RAG pipeline queries, referencing datasets like **truthfulqa** and **fever**. Meanwhile, `@kingpoki` looked for ways to speed up **Deci lm on Windows** without resorting to model quantization.

- **OCR and Transformer Learning Quests**: `@swetha98` requested OCR tools for line segmentation, with `@vikas.p` recommending [Surya OCR](https://github.com/VikParuchuri/surya). `.slartibart` asked about resources explaining the learning process of **code-writing transformers**, showing interest in understanding what the transformer learns from code.

- **Meta Reinforcement Learning and Mamba Study Group Insights**: `@davidebuoso` looked for collaborators on a **meta-RL application** to test on a Panda robotic arm, referencing the curated list ([Meta-RL Panda](https://github.com/stars/lambdavi/lists/meta-rl-panda/)). The reading group, led by `@tonic_1`, delved into the `mamba library` and discussed the trade-offs and development history of mamba in comparison to transformers and RWKV models.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Bard Ultra Release: Intense Speculation**: The community humorously discusses speculations around **Bard Ultra's** release, but acknowledges the difficulty in making accurate predictions in this realm.
- **ChatGPT Access and Performance Troubles**: Members report challenges accessing **ChatGPT** and experiencing a decrease in performance quality, with suggestions to check **[OpenAI's service status](http://status.openai.com/)**. Concerns circle around **GPT-4** being slow and resembling an older model post-update.
- **AI Applications in Jewelry Design and Content Creation**: **Dall-E 3** and **Midjourney** are highlighted as tools for creating photo-realistic jewelry images, with broader discussions addressing **Terms of Service** implications for AI-generated content and aspirations towards AGI, touching on ethical and censorship aspects.
- **Feature Requests and GPT API Issues**: Users call for a 'read back' feature to combat eye strain and complain about token inefficiency in narrative generation by **GPT** models. There are also reports of intermittent API connectivity issues and performance degradation after recent updates, referencing the [OpenAI model feedback form](https://openai.com/form/chat-model-feedback) as the avenue for reporting such findings.





---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Google Scholar Search Remains Uncleared**: An inquiry was made by `@aldikirito` regarding a suitable prompt for searching **journal citations in Google Scholar**, but the conversation did not yield a definitive response.

- **API Model Selection Guidance**: `@fanglin3064` asked which model available in the API resembles GPT-3.5 Turbo, and `@icelavaman` directed to opt for **mixtral-8x7b-instruct** for performance without web access and **pplx-7(0)b-online** with web access, sharing a [link to benchmarks](https://blog.perplexity.ai/blog/introducing-pplx-online-llms).

- **No Citations for Perplexity API Yet**: Despite community interest, `@me.lk` confirmed that source citations currently are **not planned** to be included in Perplexity API, contradicting previous expectations.

- **Scraping is Off-Limits**: With respect to API usage, `@icelavaman` reminded that scraping results, especially those utilizing GPT models, is prohibited as per the [Terms of Service](https://blog.perplexity.ai/legal/terms-of-service).

- **Perplexity API and UI Differentiation**: It was clarified by `@me.lk` and `@icelavaman` that Perplexity AI provides both a UI and an API as separate products, though both under the same corporate umbrella and subject to different access agreements.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Agentic Layer Boosts RAG Performance**: The **LlamaIndex team** highlighted an **agentic layer atop RAG** to facilitate real-time user feedback, improving the experience for complex searches. This update and more on their Query Pipelines can be found in their [Twitter announcement](https://twitter.com/llama_index/status/1755270274310901898) and [Medium post](https://t.co/Funqm7Jw1u).
  
- **Webinar Wisdom on RAG's Future**: `@seldo` from **LlamaIndex** discussed the nuances of **RAG** and features to expect in 2024 during a webinar with `@ankit1khare`, available in full on [YouTube](https://t.co/7f12VvgImc).

- **Erudite Exchange on LlamaIndex Tools and Troubleshooting**: Community members dissected issues and shared insights on using the **NLSQLTableQueryEngine**, dealing with connection issues in **Gemini**, and setting up rerankers with **LlamaIndex**. They also discussed database ingestion pipelines, with `@ramihassanein` contributing a GitHub PR for improvements in document handling for **Deeplake**: [GitHub PR Link](https://github.com/run-llama/llama_index/pull/10504).

- **Custom LLMs Can Be Tricky**: Faced with a finetuned **LLM** returning JSON objects, the **LlamaIndex** community debated on creating custom LLMs versus post-processing the JSON outputs.

- **Production-Ready RAG Chatbot Conundrum**: `@turnerz` sought advice on **vector databases, chunk sizes, and embedding models** for a production-grade **RAG HR chatbot**. The discussion also delved into strategic approaches for evolving RAG systems—whether to start with a simple one and iteratively advance, or begin with a multi-hop retrieval system.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **JupyterLab Usage Not Always Necessary**: A discussion highlighted that messages pertaining to **JupyterLab** setups can be disregarded if the software isn't being utilized, with a preference for its use mentioned by some members.

- **Enhancements to Persistent Volume Handling in RunPod**: There were troubleshoots concerning silent updates on RunPod affecting repository integrity and persistent volume presence. Users agreed on potential solutions like changing the mountpoint for better disk space management and considering cache and outputs, with hopes to document the issues and solutions on GitHub.

- **New Optimized Scheduler for Continual Pretraining Released**: A new **scheduler** optimized for continual pretraining of large language models has been introduced via a pull request on GitHub, aimed to assist in the (re)warming of such models. The pull request and further details can be accessed [here](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1273).

- **Prompt Format and Encoding Discussions for AI Queries**: There's an ongoing discussion about the most efficient data structure for AI prompts in Python. Protobuf and custom encoding schemes utilizing obscure Unicode characters have been considered as potential alternatives to JSON.

- **Training Strategies and Script Sharing for Better Model Performance**: Learning rates, batch sizes, and configuration parameters were tackled with a special mention of **unsloth for DPO**, **paged_adamw_8bit**, and a **linear scheduler**. The training script named **DreamGenTrain** was shared on GitHub and can be found [here](https://github.com/DreamGenX/DreamGenTrain/blob/master/dpo.py).

- **Preparing Images for RunPod Requires Specific Conditions**: Building images for RunPod should be done on a **machine with a GPU and bare metal docker**, as running docker inside another docker is not viable.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **LLaMa Tackles the Number "7"**: A user trained a 2M parameter **LLaMa model** for conditional generation of MNIST digit "7", completing it in approximately 25 minutes on an 8GB MacBook M1. This showcases significant efficiency in training smaller models.

- **Llava's Potential Underutilized**: The new version of **Llava 1.6** was suggested to be more advanced than its predecessor, as per a Twitter user's shared [link](https://vxtwitter.com/billyuchenlin/status/1755207605537120513). However, it was not utilized in a discussed project, where it could have perhaps offered notable improvements.

- **Ethical AI in GOODY-2 LLM**: **GOODY-2** was introduced, a model designed with an emphasis on ethical engagement and controversial question avoidance. Detailing within its [model card](https://www.goody2.ai/goody2-modelcard.pdf), the announcement of this responsibly-aligned AI model received mixed reactions.

- **Watermarks Mark the Spot in DALL-E 3**: Watermarking has been introduced to **OpenAI's DALL-E 3** outputs, embedding image metadata with both invisible and visible components, as discussed in a [The Verge's article](https://www.theverge.com/2024/2/6/24063954/ai-watermarks-dalle3-openai-content-credentials) shared by a user.

- **State-of-the-Art Background Removal Tool Arrives**: A cutting-edge **background removal tool**, BRIA Background Removal v1.4, is now rivalling leading models and is specifically designed for content safety in a commercial setting. A [demo and model card](https://huggingface.co/briaai/RMBG-1.4) are available for insight into its capabilities.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **CUDA Connoisseurs Seek Silicon**: There is a buzz around building deep learning rigs, with an interest in acquiring a used **3090 GPU** and examining hardware bundles suitable for multi-GPU setups, such as the **Intel Core i9-12900K** and **ASUS ROG Maximus Z690 Hero DDR5** motherboard available in a [bundle](https://a.co/d/iR3bvvF). Discussions also focused on the feasibility of multi-GPU configurations, highlighting the role of **PCIe bifurcation cards** and high-quality **gen4 PCIe cables**.

- **PyTorch 2 Paving the Path at ASPLOS 2024**: Excitement is high as the PyTorch 2 paper, featuring **TorchDynamo**, **TorchInductor**, and **Dynamic Shape support**, has been accepted at ASPLOS 2024. With [PyTorch 2 paper and tutorial](https://pytorch.org/blog/pytorch-2-paper-tutorial/) being a hot topic, there's a strong focus on the ease of use for developers due to the Python-based compiler, optimizations poised for consumer hardware, and advancements in debug tools like **TORCH_LOGS**. Furthermore, `torch.compile` is heralded as the preferred approach over `torch.jit.trace` for model porting.

- **JAX Jumps Ahead**: JAX's popularity over TensorFlow 2 is attributed to its **XLA JIT** optimizations and better experiences on Google Cloud TPUs, as discussed with a supportive [video](https://youtu.be/fuAyUQcVzTY?si=Sg1jK5eQUJrEkt9P). The platform also seems to compete head-to-head with TensorFlow and Torch on NVIDIA GPUs. The history of JAX is pointed out as possibly being rooted in a response to TensorFlow's complexities, especially regarding global variables in TensorFlow 1.x.

- **Quest for Community and Knowledge**: There's a keen interest in connecting with local engineers, as evidenced by inquiries about **in-person ML meetups** in San Francisco and the potential overlap with events like the **GTC**. Additionally, members are seeking solutions for reference books, with pointers provided through Discord links.

- **Learning in Leap**: Newcomers are delving into AI technologies, such as `einstein5744` starting with stable diffusion fast ai courses and others requesting community-driven knowledge exchange opportunities.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **LangChain ChatPromptTemplate Bug Report**: User `@ebinbenben` experienced a `ValueError` when using LangChain's `ChatPromptTemplate`, but the issue remained unresolved in the discussion.
- **LangChain Framework Documentation Enhanced**: LangChain documentation now includes new sections on custom streaming with events and streaming in LLM apps, with a focus on use of tools like `where_cat_is_hiding` and `get_items` detailed in the [updated docs](https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events).
- **LangChain Expression Language (LCEL) Explained**: `@kartheekyakkala` introduced a [blog post](https://medium.com/@yakkalakartheek/langchain-expression-language-lcel-8d092b0179b8) detailing LCEL's declarative syntax in the LangChain framework for making LLMs context-aware.
- **Local LLM Resources and Chatbot Toolkits Updated**: Resources and tools for LLMs, including a lightweight chatbot and foundational materials for beginners, can be found at [llama-cpp-chat-memory](https://github.com/ossirytk/llama-cpp-chat-memory) and [llm_resources](https://github.com/ossirytk/llm_resources), shared by `@discossi`.
- **Author Supports Readers Through API Changes**: In response to issues with deprecated code examples due to API updates, `@mehulgupta7991` offered assistance and committed to updating the next edition of their book. The author can be reached at datasciencepocket@gmail.com for direct support.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **LLMs Torn on Themes**: Users discussed the shortcomings of ChatGPT's theme options, describing them as too extreme with no middle ground, but no concrete solutions were proposed.

- **Fine-tuning Conversations**: A query about fine-tuning strategies focused on whether to score binary outcomes across each message in a dataset or to evaluate the conversation as a whole.

- **AI-Powered Report Service Hits Cost Hurdle**: User `@res6969` developed a service leveraging AI to create searchable databases from report sections but faced an unexpected operational cost of approximately $20,000, reacting with a humor-indicative emoji <:lmfao:556127422823530507>.

- **Library Performance Claims Meet Skepticism**: Discussion on a new library claiming 20x performance increase was met with skepticism, with concerns about its cost-effectiveness and actual utility, especially for those who self-host AI models.

- **Pondering the Expense of GPT-4**: The conversation highlighted the substantial expense associated with using GPT-4, with costs potentially reaching $30K/month, and shared doubts about new methodologies actually saving money.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **OpenAI Priced to Dominate**: `@dbreunig` highlighted the cost efficiency of OpenAI's `text-embedding-3-small`, remarking on its affordability to the point that for certain features, competitors seem unnecessary.
- **Competitive Strategy Suspected in OpenAI Pricing**: `@simonw` speculated that OpenAI's pricing might be strategically designed to suppress competition, attracting consensus from `@dbreunig`.
- **Competing in a Market Dominated by OpenAI**: `@dbreunig` and `@simonw` discussed the challenges for competitors trying to build platforms to rival OpenAI, suggesting that innovation beyond pricing, UI, and UX would be essential to differentiate in this space.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **LoRA Demands More Memory**: `devnull0` emphasized the need for ample memory to accommodate both the model and **LoRA** (Locally Rewarded Attention) parameters when optimizing models.

- **Squeezing Models on Slim Resources**: According to `johannhartmann`, **Llama_factory**'s new support for *unsloth* allows **Mistral** to work with *qlora*, which makes it feasible for the wiedervereinigung-7b model to fit within certain memory limitations.

- **Mistral Goes Multilingual and Multi-dataset**: `johannhartmann` also indicated that Llama_factory has expanded **Mistral**'s capabilities to include support for **9 German SFT** and one **DPO datasets**.

- **New Benchmark on the Block**: _jp1_ dropped a link mentioning an [arXiv paper](https://arxiv.org/abs/2402.01781), suggesting a possible new benchmark or study that might be of interest to the community.

---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1204703108463329321) (1295 messages🔥🔥🔥): 

- **Steam Deck as a Surprise AI Beast**: User `@skorchekd` shared their experience of running models on the Steam Deck, noting significant performance increases compared to their PC, which they attribute to the device's RAM speed and integrated APU. They discussed the potential to run larger models like Solar 10.7B efficiently, even with the RAM limit, and even considered the device to be of great value as a portable gaming PC that can also handle AI.
  
- **Debate on the Essence of AI and Mathematics**: Users like `@phantine`, `@selea`, and `@lee0099` engaged in a philosophical discussion about the nature of artificial intelligence, whether it can "understand" in the way humans do, and if mathematical concepts exist independently of physical reality. The conversation touched on the concept of necessary ideas, metaphysics, and the substance of consciousness.

- **LLM Performance Discussions Continue**: `@starsupernova` mentioned working on adding multi-GPU support to OSS Unsloth, which is currently faster for inference but is recommended over vLLM for validation and Gradio type inferences only. They also note the advantages of pre-quantization, including no loss of accuracy and reduced VRAM by 1GB, and stated interest in applying LASER to Unsloth after reading the corresponding paper.
  
- **OpenAI Models Contamination in Training Data**: `@itsme9316` pointed out that most contemporary models likely contain training data generated by OpenAI models or from the general internet. This includes synthetic data contamination, leading to responses reminiscent of OpenAI's style.

- **Technical Discussions on LLMs and Computing**: Various users, including `@selea`, `@technotech`, and `@starsupernova`, discussed LLM architectures like Bert, potential improvements with P100 over T4 GPUs, ways to optimize compute graphs, and kernel optimization. There was also a mention of a Mamba paper showing promising results compared to transformers.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1142735399555432529/1204930925125701642): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1130801664383787071): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566): Humans are capable of strategically deceptive behavior: behaving helpfully in most situations, but then behaving very differently in order to pursue alternative objectives when given the opportunity. ...
- [no title found](https://e2eml.school/transformers.html#markov_chain): no description found
- [Prompt engineering techniques with Azure OpenAI - Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering): Learn about the options for how to use prompt engineering with GPT-3, GPT-35-Turbo, and GPT-4 models
- [serpdotai/sparsetral-16x7B-v2 · Hugging Face](https://huggingface.co/serpdotai/sparsetral-16x7B-v2): no description found
- [Giga Gigacat GIF - Giga Gigacat Cat - Discover &amp; Share GIFs](https://tenor.com/view/giga-gigacat-cat-mewing-mogging-gif-12429734670640119345): Click to view the GIF
- [Ascending Energy GIF - Ascending Energy Galaxy - Discover &amp; Share GIFs](https://tenor.com/view/ascending-energy-galaxy-gif-17739196): Click to view the GIF
- [abacusai/Smaug-72B-v0.1 · Hugging Face](https://huggingface.co/abacusai/Smaug-72B-v0.1#evaluation-results): no description found
- [Paper page - Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning
  Tasks](https://huggingface.co/papers/2402.04248): no description found
- [NeuralNovel/Tiger-7B-v0.1 · Hugging Face](https://huggingface.co/NeuralNovel/Tiger-7B-v0.1): no description found
- [Shooting GIF - Cowboy Gun Shooting - Discover &amp; Share GIFs](https://tenor.com/view/cowboy-gun-shooting-smoke-gif-5591465): Click to view the GIF
- [Imaginary Numbers Are Real [Part 1: Introduction]](https://www.youtube.com/watch?v=T647CGsuOVU&t=2s): For early access to new videos and other perks: https://www.patreon.com/welchlabsWant to learn more or teach this series? Check out the Imaginary Numbers are...
- [Clapping Hamood GIF - Clapping Hamood Mood - Discover &amp; Share GIFs](https://tenor.com/view/clapping-hamood-mood-hi-baby-happy-gif-13463157): Click to view the GIF
- [Large Language Models Process Explained. What Makes Them Tick and How They Work Under the Hood!](https://youtu.be/_Pt-rGE4zEE?si=s6orG0bKWAkX_vPY&t=326): Explore the fascinating world of large language models in this comprehensive guide. We&#39;ll begin by laying a foundation with key concepts such as softmax, lay...
- [GitHub - cognitivecomputations/laserRMT: This is our own implementation of &#39;Layer Selective Rank Reduction&#39;](https://github.com/cognitivecomputations/laserRMT): This is our own implementation of &#39;Layer Selective Rank Reduction&#39; - GitHub - cognitivecomputations/laserRMT: This is our own implementation of &#39;Layer Selective Rank Reduction&#39;
- [Tweet from Alexandre TL (@AlexandreTL2)](https://x.com/alexandretl2/status/1754927881178791962?s=61&t=tHcPPlKi_G7OoyasQK8oDQ): How does Mamba fare in the OthelloGPT experiment ?  Let&#39;s compare it to the Transformer 👇🧵
- [AMD Ryzen 7 8700G APU Review: Performance, Thermals &amp; Power Analysis - Hardware Busters](https://hwbusters.com/cpu/amd-ryzen-7-8700g-apu-review-performance-thermals-power-analysis/): Hardware Busters - AMD Ryzen 7 8700G APU Review: Performance, Thermals &amp; Power Analysis - CPU
- [Reddit - Dive into anything](https://www.reddit.com/r/singularity/s/bZI9ELVwhD): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1204699495271497748) (342 messages🔥🔥): 

- **Quantization and Financial Constraints**: `@dreamgen` touched on the sustainability of free models, predicting a challenging future once venture capital funds dry up. Meanwhile, `@mrdragonfox` highlighted the financial burden of creating datasets compared to the ease and low cost of doing model merges.

- **Merging Models and Data Dilemmas**: Concerns were expressed about the merger of models, especially by `@soufflespethuman` and `@mrdragonfox`, who worry it undermines the value of creating original datasets and the incentive for real innovation. They discussed the possibility of enforcing a "do not merge" license to protect their work.

- **Financing AI Innovation Fantasies**: A discussion emerged around the hypothetical scenario of a "crypto/stocks daddy" funding AI experiments, as suggested by `@billynotreally`. `@mrdragonfox` countered the generosity of donors, arguing that people with money typically want something in return, such as acknowledgement or results.

- **Technical Talk on Augmentoolkit Reorganization**: `@mrdragonfox` is reworking the architecture of Augmentoolkit, sharing updates and code via [GitHub](https://github.com/e-p-armstrong/augmentoolkit). Discussions included leveraging asynchronous IO in Python and the transition from Jupyter notebooks to a more structured codebase.

- **MiquMaid v2 Discussion and Development**: Updates and discussions around MiquMaid v2 took place, with `@undi` sharing progress and noting it's public, while `@netrve` enjoyed the model's performance, despite some issues with repetition and needing to tweak generation settings. The latter discussions delved into the strategies for dealing with repetition when generating content.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1204890898878697503): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [MiquMaid - a NeverSleep Collection](https://huggingface.co/collections/NeverSleep/miqumaid-65c3d5e0fd15420346adc906): no description found
- [TheBloke/Yarn-Llama-2-70B-32k-GGUF · Hugging Face](https://huggingface.co/TheBloke/Yarn-Llama-2-70B-32k-GGUF): no description found
- [augmentoolkit/config.yaml at api-branch · e-p-armstrong/augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/blob/api-branch/config.yaml): Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit
- [augmentoolkit/main.py at api-branch · e-p-armstrong/augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/blob/api-branch/main.py): Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit
- [Ness End This Suffering Mother2earthbound Super Smash Bros GIF - Ness End This Suffering Mother2Earthbound Super Smash Bros - Discover &amp; Share GIFs](https://tenor.com/view/ness-end-this-suffering-mother2earthbound-super-smash-bros-gif-25870136): Click to view the GIF
- [Baroque Rich GIF - Baroque Rich - Discover &amp; Share GIFs](https://tenor.com/view/baroque-rich-gif-22998652): Click to view the GIF

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1204820787962454167) (5 messages): 

- **Seeking Guidance for Fine-Tuning**: User `@immortalrobot` expressed a desire to learn fine-tuning processes and requested recommendations for tutorial articles that provide step-by-step guidance.
- **Novel Approach to LM Alignment**: `@maldevide` shared an [arxiv paper](https://arxiv.org/html/2402.01878v1) introducing **Listwise Preference Optimization (LiPO)**, a new framework for aligning language models that optimizes listwise responses rather than individual ones.
- **The Next Step in Model Optimization**: Commenting on the effectiveness of PairRM DPO'd models, `@maldevide` regarded LiPO as the logical progression in language model alignment techniques.
- **Local Training Feasibility with LiPO Rankers**: `@maldevide` pointed out an advantage of LiPO, emphasizing that rankers used in the framework are under 1 billion parameters, enabling them to be trained locally, which is a practical benefit.
- **Model Loading Inquiry**: `@yinma_08121` inquired if anyone has experience using Candle to load the phi-2-gguf model, seeking community input on the matter.

**Links mentioned**:

[LiPO: Listwise Preference Optimization through Learning-to-Rank](https://arxiv.org/html/2402.01878v1): no description found

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1204741981063094282) (24 messages🔥): 

- **Guidance for Hugging Face Model Struggles**: `@wbsch` highlighted to `@logicloops` that issues with implementing models from Hugging Face might be due to incorrect prompts and unusual stop tokens, and suggested re-checking the model's card or considering easier alternatives like koboldcpp.

- **Mojo Outperforms Rust**: `@dirtytigerx` shared an article about how the Mojo language is achieving significant performance wins, with benchmarks showing improvements over Python and even Rust, prompting discussions on its impact on various tools.

- **Potential of Mojo in High-Performance Environments**: `@falconsfly` expressed enthusiasm for the design and implementation strategy of Mojo, highlighting its capabilities with an example of matrix multiplication optimizations and how they might pair well with tools like duckdb.

- **Aletheion Explores Integrated Agent Functions**: `@aletheion` is working on integrating custom functions into bot flows that can call database lookups as needed, aiming for the bot to utilize its own "memories" and "notes" to provide enhanced interaction without relying on external logic triggers.

- **Misunderstandings in Model Implementation**: `@lushboi` admitted to an oversight on the capabilities of smaller LLaMa models, which was clarified by `@falconsfly` pointing out that only the 70B LLaMa model uses GQA, highlighting the importance of referring to the official documentation.

**Links mentioned**:

- [Modular: Community Spotlight: Outperforming Rust ⚙️ DNA sequence parsing benchmarks by 50% with Mojo 🔥](https://www.modular.com/blog/outperforming-rust-benchmarks-with-mojo): We are building a next-generation AI developer platform for the world. Read our latest post on how Community Spotlight: Outperforming Rust ⚙️ DNA sequence parsing benchmarks by 50% with Mojo 🔥
- [Modular Docs - Matrix multiplication in Mojo](https://docs.modular.com/mojo/notebooks/Matmul.html): Learn how to leverage Mojo's various functions to write a high-performance matmul.

  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1204930215852114041) (2 messages): 

- **Introducing OpenHermes Dataset Viewer**: `@carsonpoole` developed a dataset viewer for **OpenHermes** that allows users to scroll through examples using the `j` and `k` keys and examine analytics on token counts and types. The tool also features a filter for sorting by the number of samples or token count.
  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1204760064846598177) (43 messages🔥): 

- **Introducing Subformer with SAFE**: `@euclaise` shared an [arXiv paper](https://arxiv.org/abs/2101.00234) focused on exploring parameter-sharing methods in Transformers to address their computational and parameter budget inefficiencies. The paper introduces the **Subformer**, which utilizes sandwich-style parameter sharing and self-attentive embedding factorization (SAFE) to outperform the Transformer model with fewer parameters.
  
- **BiLLM Introduces One-Bit Quantization**: `@gabriel_syme` posted a link to a paper about **BiLLM** ([Download PDF](/pdf/2402.04291.pdf)), a 1-bit post-training quantization scheme for large language models aimed at significantly reducing computation and memory requirements while maintaining performance.

- **Model Scores on NeoEvalPlusN Benchmark**: `@nonameusr` linked to a Hugging Face page for **Gembo-v1-70b** model, cautioning that the model contains sensitive content and may have potentially harmful information. Despite not having a full model card, they noted this model scores highly on the NeoEvalPlusN benchmark and is awaiting results from openllm.

- **Environment-Focused Model Performance**: `@teknium` mentioned the importance of context when sharing models on the forum and suggested including captions or model cards, as many postings have been lacking those details.

- **Cross-Comparison of Self-Rewarding Language Models**: In a discussion about recent advances, `@atgctg` linked to an [arXiv paper](https://arxiv.org/abs/2402.04792) comparing OAIF with concurrent "self-rewarding" language model work. The paper highlights how OAIF can leverage feedback from any LLM, including those stronger than the one being aligned.

**Links mentioned**:

- [BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://arxiv.org/abs/2402.04291): Pretrained large language models (LLMs) exhibit exceptional general language processing capabilities but come with significant demands on memory and computational resources. As a powerful compression ...
- [InfoEntropy Loss to Mitigate Bias of Learning Difficulties for Generative Language Models](https://arxiv.org/abs/2310.19531): Generative language models are usually pretrained on large text corpus via predicting the next token (i.e., sub-word/word/phrase) given the previous ones. Recent works have demonstrated the impressive...
- [ChuckMcSneed/Gembo-v1-70b · Hugging Face](https://huggingface.co/ChuckMcSneed/Gembo-v1-70b): no description found
- [ibivibiv/giant-hydra-moe-240b · Hugging Face](https://huggingface.co/ibivibiv/giant-hydra-moe-240b): no description found
- [Direct Language Model Alignment from Online AI Feedback](https://arxiv.org/abs/2402.04792): Direct alignment from preferences (DAP) methods, such as DPO, have recently emerged as efficient alternatives to reinforcement learning from human feedback (RLHF), that do not require a separate rewar...
- [Cute Hide GIF - Cute Hide Cat - Discover &amp; Share GIFs](https://tenor.com/view/cute-hide-cat-scared-shy-gif-16121120): Click to view the GIF
- [ChuckMcSneed/NeoEvalPlusN_benchmark · Datasets at Hugging Face](https://huggingface.co/datasets/ChuckMcSneed/NeoEvalPlusN_benchmark): no description found
- [Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers](https://arxiv.org/abs/2101.00234): Transformers have shown improved performance when compared to previous architectures for sequence processing such as RNNs. Despite their sizeable performance gains, as recently suggested, the model is...
- [InRank: Incremental Low-Rank Learning](https://arxiv.org/abs/2306.11250): The theory of greedy low-rank learning (GLRL) aims to explain the impressive generalization capabilities of deep learning. It proves that stochastic gradient-based training implicitly regularizes neur...

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1204707990645506048) (221 messages🔥🔥): 

- **Seeking Wojak AI Link**: `@theluckynick` requested the link to the wojak AI but wasn't provided with a response.
- **Training Config Query for OpenHermes-2.5-Mistral-7B**: `@givan_002` inquired about the fine-tuning parameters for an AI model and was directed to a closed discussion with no satisfactory response about the training configuration.
- **Benchmarking AI Models**: `@if_a` revealed results from benchmarking various models, noting that Senku-70B outperforms others in specific tasks, and discussions indicated that different system prompts across LLMs might impact results.
- **Fine-Tuning on Nous-Hermes 2 Dataset Considered**: `@if_a` is contemplating fine-tuning the miqu model using the Nous-Hermes 2 dataset, discussing the potential timescales and challenges.
- **GPU Workload Scheduling Discussions**: Various users, including `@chrisj7746`, `@Sebastian`, and `@leontello`, discussed efficient methods for scheduling jobs on GPUs, with Slurm being a popular recommendation even for smaller clusters.
- **Quantization and Model Architecture Discussions**: Users discussed quantization algorithms for AI models, including the potential of an anime benchmark by `@vatsadev`. Meanwhile, `@nonameusr` and `@n8programs` discussed the significance of architectural changes in models post-GPT-4, including Turing complete alterations of transformers.

**Links mentioned**:

- [BiLLM: Pushing the Limit of Post-Training Quantization for LLMs](https://arxiv.org/abs/2402.04291): Pretrained large language models (LLMs) exhibit exceptional general language processing capabilities but come with significant demands on memory and computational resources. As a powerful compression ...
- [teknium/OpenHermes-2.5-Mistral-7B · Can the training procedure be shared?](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/discussions/9): no description found
- [Orange Cat Staring GIF - Orange cat staring Orange cat Staring - Discover &amp; Share GIFs](https://tenor.com/view/orange-cat-staring-orange-cat-staring-cat-gif-13724146065807985297): Click to view the GIF
- [Turing Complete Transformers: Two Transformers Are More Powerful...](https://openreview.net/forum?id=MGWsPGogLH): This paper presents Find+Replace transformers, a family of multi-transformer architectures that can provably do things no single transformer can, and which outperforms GPT-4 on several challenging...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1al58xw/yet_another_state_of_the_art_in_llm_quantization/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/ChatGPT/comments/1ahhlon/i_downloaded_my_chatgpt_user_data_and_found_the/): no description found
- [VatsaDev/animebench-alpha · Datasets at Hugging Face](https://huggingface.co/datasets/VatsaDev/animebench-alpha): no description found
- [teknium/OpenHermes-2.5 · Datasets at Hugging Face](https://huggingface.co/datasets/teknium/OpenHermes-2.5): no description found
- [cmp-nct/llava-1.6-gguf at main](https://huggingface.co/cmp-nct/llava-1.6-gguf/tree/main): no description found
- [Llava 1.6 - wip by cmp-nct · Pull Request #5267 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5267): First steps - I got impressive results with llava-1.6-13B on the license_demo example already, despite many open issues. Todo: The biggest and most important difference missing is the &quot;spatial_un...

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1204700271884636192) (49 messages🔥): 

- **Custom Pretrained Model Struggles with Extraction**: `@fedyanin` expressed difficulty with their custom pretrained model, which performs well on generation tasks but not on extraction. They inquired about methods to improve extraction performance beyond explicit fine-tuning on the task.
  
- **VRAM Sufficiency for Finetuning**: `@natefyi_30842` asked if 8GB VRAM is enough for SFT or DPO on a 7b model like Mistral using Axolotl. `@fedyanin` suggests it might just be enough with *qlora* and a small context, while `@teknium` mentioned mlx might work but is not yet fully developed.

- **Setting Special Tokens in Configuration**: `@paragonicalism` sought advice for setting up special tokens in a `.yml` file for fine-tuning `phi-2` on OpenHermes2.5 with axolotl. `@teknium` provided a code snippet to define the `eos_token` and other tokens, and later suggested adding `pad_token: 

**Links mentioned**:

- [phi2-finetune/nb_qlora.ipynb at main · geronimi73/phi2-finetune](https://github.com/geronimi73/phi2-finetune/blob/main/nb_qlora.ipynb): Contribute to geronimi73/phi2-finetune development by creating an account on GitHub.
- [LLaVA/docs/Finetune_Custom_Data.md at main · haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md): [NeurIPS&#39;23 Oral] Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond. - haotian-liu/LLaVA
- [liuhaotian/LLaVA-Instruct-150K · Datasets at Hugging Face](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K): no description found
- [GitHub - bkitano/llama-from-scratch: Llama from scratch, or How to implement a paper without crying](https://github.com/bkitano/llama-from-scratch): Llama from scratch, or How to implement a paper without crying - GitHub - bkitano/llama-from-scratch: Llama from scratch, or How to implement a paper without crying
- [LLaVA/scripts/v1_5/finetune_task_lora.sh at main · haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_task_lora.sh): [NeurIPS&#39;23 Oral] Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond. - haotian-liu/LLaVA
- [LLaVA/scripts/v1_5/finetune_task.sh at main · haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_task.sh): [NeurIPS&#39;23 Oral] Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond. - haotian-liu/LLaVA
- [Obsidian/scripts/finetune_qlora.sh at main · NousResearch/Obsidian](https://github.com/NousResearch/Obsidian/blob/main/scripts/finetune_qlora.sh): Maybe the new state of the art vision model? we&#39;ll see 🤷‍♂️  - NousResearch/Obsidian
- [Obsidian/scripts/v1_5/finetune.sh at main · NousResearch/Obsidian](https://github.com/NousResearch/Obsidian/blob/main/scripts/v1_5/finetune.sh): Maybe the new state of the art vision model? we&#39;ll see 🤷‍♂️  - NousResearch/Obsidian

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1204721503111749662) (138 messages🔥🔥): 

- **LaTeX Limitations in LM Studio**: User `.neurorotic.` expressed frustration with the output of mathematical formulas in LaTeX format when using DeepSeek Math RL 7B LLM; LMStudio does not seem to render LaTeX properly, and this concern was echoed by others noting an increase in LaTeX outputs from various models.

- **GPU vs RAM and CPU for LLMs**: `@pierrunoyt` inquired why LLMs use GPU rather than RAM and CPU, with `@justmarky` explaining that the processing is much faster on GPUs. This sparked a brief discussion on computation preferences for large language models.

- **Model Selection and Optimization Discussions**: Several users discussed various models and their compatibilities with different systems. `@kristus.eth` mentioned missing functionality in Ollama compared to LM studio, and `@akiratoya13` faced issues with GPU offload not being utilized despite settings indicating otherwise.

- **Networking and Download Issues with LM Studio**: Users `@yorace` and `@tkrabec` discussed problems with network errors and model downloading within LM Studio, with `@heyitsyorkie` suggesting issues might stem from country-based blocking of Huggingface or VPN issues.

- **Local LLM Adaptability and Persistence**: Users `@joelthebuilder` and `@fabguy` engaged in a conversation about whether local large language models can learn or adapt over time through user interaction. The current stance is that fine-tuning is not typically feasible on average hardware, and that incorporating relevant system prompts is usually sufficient for tailoring responses.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1204973625518587925.): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [OLMo - Open Language Model by AI2](https://allenai.org/olmo): OLMo is a series of Open Language Models designed to enable the science of language models. The OLMo models are trained on the Dolma dataset.
- [Hugging Face – The AI community building the future.](https://huggingface.co/): no description found
- [Running big-AGI locally with LM Studio [TUTORIAL]](https://youtu.be/MqXzxVokMDk): ➤ Twitter - https://twitter.com/techfrenaj➤ Twitch  - https://www.twitch.tv/techfren➤ Discord  - https://discord.com/invite/z5VVSGssCw➤ TikTok - https://www....
- [[1hr Talk] Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g): This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What the...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1204751117733601310) (45 messages🔥): 

- **Mixed Reactions to Qwen 1.5**: `@pierrunoyt` shared a [YouTube video](https://www.youtube.com/watch?v=RjCYLIUfxrE&t=1s) about **Qwen 1.5**, an opensource language model, but `@heyitsyorkie` criticized the content for misleading model previews. `@evi_rew` and `@yagilb` further discussed technical issues with Qwen 1.5, pertaining to context lengths and GPU acceleration.
- **Code Llama Critique**: `@pierrunoyt` expressed dissatisfaction with Code Llama 2, indicating a need for a better coding language learning model that also understands design.
- **OpenAI Model Inconsistencies in Code Generation**: `@lord_half_mercy` observed a decline in reply quality when generating code with OpenAI's ChatGPT, questioning if it's related to context length or complexity; `@heyitsyorkie` humorously suggested it's due to the AI getting "lazy."
- **Confusion over Vision Models**: `@bob_dale` inquired about models capable of generating new logos based on a style, being recommended GPT4 Vision by `@heyitsyorkie`, despite initial misunderstandings regarding the model's capabilities.
- **Debates Over Model Effectiveness**: Users discussed the effectiveness of various models, including Qwen 1.5, Miqu, and LLaMA, with comments ranging from memory issues on high VRAM systems (`@.bambalejo`) to criticism of language abilities and model outputs (`@pwrreset` and `@re__x`).


**Links mentioned**:

- [@JustinLin610 on Hugging Face: &quot;Yesterday we just released Qwen1.5. Maybe someday I can tell more about the…&quot;](https://huggingface.co/posts/JustinLin610/764363519759697): no description found
- [Qwen 1.5: Most Powerful Opensource LLM - 0.5B, 1.8B, 4B, 7B, 14B, and 72B - BEATS GPT-4?](https://www.youtube.com/watch?v=RjCYLIUfxrE&t=1s): In this video, we dive deep into the latest iteration of the Qwen series, Qwen 1.5. Released just before the Chinese New Year, Qwen 1.5 brings significant up...

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1204763761383178311) (98 messages🔥🔥): 

<ul>
  <li><strong>Optimal LLM Hardware Specs Query</strong>: `@jolionvt` inquired about ideal settings for running LLM locally with an AMD Ryzen 5 7600, AMD Radeon RX 6700 XT, and 32 GB RAM, but did not receive a direct response.</li>
  <li><strong>PCIe Riser Performance Concerns</strong>: `@nink1` questioned the performance degradation when using PCIe 1x to 16x riser cables compared to a direct motherboard connection. `@quickdive.` expressed interest in testing this concern and suggested editing the BIOS lane width for comparison, while `@nink1` planned to investigate further.</li>
  <li><strong>Exploring the ESP32 for DIY Voice Projects</strong>: `@joelthebuilder` sought DIY project suggestions for hardware that could provide voice input and output for a home network, expressing a goal to replace Alexa with a custom setup. Encouraged by others, `@joelthebuilder` shared a [YouTube video](https://www.youtube.com/watch?v=_qft28MiVnc) about using ESP32 S3 Box 3 for integrating with Home Assistant and Local LLM AI, and pondered its availability for purchase.</li>
  <li><strong>Evaluating Risks of Power Cable Modification</strong>: `@nink1` shared a creative solution to power an external riser board by modifying a PSU CPU cable, which `@.ben.com` cautioned could be a fire hazard. `@nink1` argued the safety of his setup considering the low power consumption and proper cable securement.</li>
  <li><strong>GPU Bandwidth and PCIe Lane Concerns for Multi-GPU Setups</strong>: Debate ensued about the impact of PCIe bandwidth when using extender cables or restricting lane width for multi-GPU configurations. `@nink1`, `@quickdive.`, `@rugg0064`, and `@savethehuman5` discussed potential performance bottlenecks while sharing individual plans for experimenting with multiple GPU setups.</li>
</ul>

**Links mentioned**:

- [Tesla P40 Radial fan shroud by neophrema](https://www.thingiverse.com/thing:6031884/makes): Hey, This is a fan shroud for Nvidia Tesla cards which are structurally identical to the P40. After a failed attempt using a normal Noctua (damn I sank money into it...) fan I realized that air pressu...
- [All About AI](https://www.youtube.com/@AllAboutAI): Welcome to my channel All About AI =)  Discord: https://discord.gg/Xx99sPUeTd  Website: https://www.allabtai.com  How you can start to use Generative AI to help you with creative or other daily tasks....
- [ESP32 S3 Box 3  Willow - HA - Local LLM AI](https://www.youtube.com/watch?v=_qft28MiVnc): ESP32 S3 Box 3 with Willow connected to Home Assistant which is integrated with Local LLM AI (Mistral 7b)
- [NVIDIA Tesla P40 24GB GDDR5 Graphics Card and Cooling Turbine Fan  | eBay](https://www.ebay.ca/itm/225917793841): no description found
- [Amazon.com: Raspiaudio ESPMUSE Proto esp32 Development Card with Speaker and Microphone](https://www.amazon.com/RASPIAUDIO-ESPMUSE-Development-Speaker-Microphone/dp/B09N3S9S29?crid=17US408DI26TI&keywords=esp+muse+luxe&qid=1675950795&sprefix=esp+muse+luxe,aps,173&sr=8-1&linkCode=sl1&tag=peyanski-20&linkId=363d84ed2b123aa08b6e293d433c3507&language=en_US&ref_=as_li_ss_tl): no description found
- [no title found](https://www.aliexpress.com/item/1005005600238754.html): no description found

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1204744733931606036) (10 messages🔥): 

- **Debian Troubles with LM Studio**: `@transfluxus` reported issues executing the **LM_Studio-0.2.14-beta-1.AppImage** on Debian, seeing the error `cannot execute binary file`. `@heyitsyorkie` suggested grabbing the Linux beta role and checking the pinned messages in <#1138544400771846174>, as well as making the app executable with the `chmod` command.

- **Feature Wishlist for Training and Images**: `@junkboi76` expressed a desire to see **support for training** and **image generation support** in future releases. They acknowledge the complexity but maintain these would be welcome features.

- **Enhancement Suggestions Go to Feedback Station**: `@fabguy` directed `@junkboi76` to open a discussion in <#1128339362015346749> or upvote existing feature requests regarding their suggestions for training and image generation.

- **Confusion Over `pre_prompt` in JSON Preset**: `@wolfspyre` questioned whether the `pre_prompt` in the preset JSON was the **system prompt**, and if it should be reflected in the 'prompt format' preview. The issue was subsequently reported as a potential bug.

- **Design Choice or Bug?**: In response to `@wolfspyre`'s concern about the `pre_prompt` not showing in settings modal, `@fabguy` noted that the content of "User" and "Assistant" isn't shown either, possibly by design due to the potentially extensive length of system prompts.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1204848289866850334/1204848289866850334): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1205009689214193745) (3 messages): 

- **Autogen Experiments Begin**: `@photo_mp` starts experimenting with **autogen** and expresses being impressed with its capabilities in initial usage.
- **Userproxy Goes Silent**: `@photo_mp` encounters a problem where **userproxy stops communicating** after the first round of interaction, halting the conversation between agents.
- **Seeking Autogen Advice**: `@photo_mp` reaches out to the community seeking **tips** for resolving the issue with **userproxy** in autogen.
  

---


### LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1204975673748492338) (1 messages): 

- **Troubleshooting Chat Issues**: User `@yagilb` recommended trying to delete problematic chat messages or clicking **"reset to default settings"** in the top right if experiencing trouble.
  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1204823817026015252) (1 messages): 

- **In Search of a Visualization Tool**: User `@m4sterdragon` inquired about a tool or method to **visualize crew interactions** following the project kickoff, suggesting a need for better oversight of team dynamics. No solutions or follow-up comments were provided within the available messages.
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1204869473794785300) (13 messages🔥): 

- **Open Interpreter Discord Clarification**: `@fkx0647` inquired about the **Open Interpreter (OI)** Discord, initially mistaking the LM Studio Discord for it. `@heyitsyorkie` clarified that the channel is just a sub-channel within LM Studio, directing to the OI Discord on their [GitHub page](https://github.com/KillianLucas/open-interpreter/).
- **Understanding Local Models Server Mode**: `@heyitsyorkie` explained that OI can utilize **LMStudio's server mode for Local Models**, shedding light on the integration between OI and LMStudio.
- **Exploring OI Features and Assistance with Discord Invitation**: `@fkx0647` discussed their progress with **OI** and sought assistance in finding an invite to the official OI Discord. `@heyitsyorkie` provided guidance on where to find the invite on the GitHub README for OI.
- **Confirmation on GPT-3.5 Usage**: `@fkx0647` confirmed they have OI running, utilizing **GPT 3.5** and noted the CLI's performance is satisfactory.

**Links mentioned**:

[GitHub - KillianLucas/open-interpreter: A natural language interface for computers](https://github.com/KillianLucas/open-interpreter/): A natural language interface for computers. Contribute to KillianLucas/open-interpreter development by creating an account on GitHub.

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1204895826116345867) (12 messages🔥): 

- **Philosophical AI**: User `@gabriel_syme` humorously suggested that if models were true philosophers like Socrates, there wouldn't be much discussion, as Socrates often dominated conversations in Plato's texts.
- **Llama Model's Safety Compromised for $200**: `@swyxio` shared a [LessWrong post](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from?utm_source=ainews&utm_medium=email) discussing how to undo Llama 2's safety features with LoRA fine-tuning, under a $200 budget. The post highlights concerns about the ease of bypassing safety trainings in powerful models and the associated risks.
- **The Rise of Synthetic Data**: In a curt message, `@swyxio` signaled that the creation of synthetic data is gaining momentum.
- **Voice Cloning's Imminent Transformation**: `@guardiang` reacted to a link to a Discord message by indicating that voice cloning technology is about to significantly change, potentially posing new challenges and opportunities.
- **Excitement for Upcoming GPT Release**: User `@coffeebean6887` hinted at the arrival of an unnamed GPT model through a [Twitter link](https://twitter.com/chiefaioffice/status/1755311356922732595), and `@guardiang` expressed enthusiasm about seeing it in action. Meanwhile, `@eugeneyan` shared excitement but also joked about the limitations of their iPhone display to fully showcase the capabilities of the forthcoming "Blind."

**Links mentioned**:

[LoRA Fine-tuning Efficiently Undoes Safety Training from Llama 2-Chat 70B — LessWrong](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from?utm_source=ainews&utm_medium=email): Produced as part of the SERI ML Alignment Theory Scholars Program - Summer 2023 Cohort, under the mentorship of Jeffrey Ladish.  …

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1204878568971173968) (1 messages): 

- **LLM Paper Club Kicks Off**: `@swyxio` announced that `@713143846539755581` will be presenting the **Self Reward paper** at the Latent Space Discord Paper Club. The session will commence shortly and members can join [here](https://lu.ma/llm-paper-club).

**Links mentioned**:

[LLM Paper Club (West) · Luma](https://lu.ma/llm-paper-club): We have moved to  use the new Discord Stage feature here: https://discord.com/channels/822583790773862470/1197350122112168006 see you soon!

  

---


### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1204879363812757586) (209 messages🔥🔥): 

- **Self-Rewarding Language Models Spark Interest**: The paper club's discussion centered on a new paper proposing *Self-Rewarding Language Models* as outlined by `@coffeebean6887`, with a new approach where LLMs use their own output to provide rewards during training, cited as outperforming Claude 2 and GPT-4 in some aspects. The full paper is available [here](https://arxiv.org/abs/2401.10020).
  
- **Interest in DSPy Spikes**: DSPy, a programming model for chaining LLM calls, garnered significant attention, with multiple members, including `@kbal11` and `@yikesawjeez`, expressing interest in exploring it further. `@yikesawjeez` volunteered to lead a session on DSPy, and the paper can be found [here](https://arxiv.org/abs/2312.13382).

- **Engagement with Challenging Theorems Preparation**: `@stephen_83179_13077` compared the discussed LLM paper with automatic geometric theorem proving methods, while `@gabriel_syme` noted the necessity for external verifiers beyond LLMs for evaluations of complex tasks. A paper suggested for insight into reasoning through topology is available [here](https://arxiv.org/abs/2401.14295).

- **Upcoming Paper Clubs Features**: Anticipation is building for future paper club sessions, with potential discussions on CRINGE loss, Colbert model, and T5 vs. TinyLlama, as suggested by `@amgadoz` and `@_bassboost`. Moreover, the talk of reviewing "Leveraging Large Language Models for NLG Evaluation: A Survey" has been set for the next week, with the paper viewable [here](https://arxiv.org/pdf/2401.07103.pdf).

- **YouTube Resource for DSPy Exploration**: `@yikesawjeez` shared a [YouTube video](https://www.youtube.com/watch?v=rqR3LeR09gc&t=903s&ab_channel=code_your_own_AI) as a resource for delving into DSPyG, which combines DSPy with a Graph Optimizer, showcasing an example of a Multi Hop RAG implementation with graph optimization.

**Links mentioned**:

- [Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts](https://arxiv.org/abs/2401.14295): The field of natural language processing (NLP) has witnessed significant progress in recent years, with a notable focus on improving large language models&#39; (LLM) performance through innovative pro...
- [BirdCLEF 2021 - Birdcall Identification | Kaggle](https://www.kaggle.com/c/birdclef-2021): no description found
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020): We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human prefer...
- [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380): Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance o...
- [Tuning Language Models by Proxy](https://arxiv.org/abs/2401.08565): Despite the general capabilities of large pretrained language models, they consistently benefit from further adaptation to better achieve desired behaviors. However, tuning these models has become inc...
- [DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines](https://arxiv.org/abs/2312.13382): Chaining language model (LM) calls as composable modules is fueling a new way of programming, but ensuring LMs adhere to important constraints requires heuristic &#34;prompt engineering&#34;. We intro...
- [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109): Large language models (LLMs) are gaining increasing popularity in both academia and industry, owing to their unprecedented performance in various applications. As LLMs continue to play a vital role in...
- [Some things are more CRINGE than others: Preference Optimization with the Pairwise Cringe Loss](https://arxiv.org/abs/2312.16682): Practitioners commonly align large language models using pairwise preferences, i.e., given labels of the type response A is preferred to response B for a given input. Perhaps less commonly, methods ha...
- [self-rewarding-lm-pytorch/self_rewarding_lm_pytorch/self_rewarding_lm_pytorch.py at ec8b9112d4ced084ae7cacfe776e1ec01fa1f950 · lucidrains/self-rewarding-lm-pytorch](https://github.com/lucidrains/self-rewarding-lm-pytorch/blob/ec8b9112d4ced084ae7cacfe776e1ec01fa1f950/self_rewarding_lm_pytorch/self_rewarding_lm_pytorch.py#L127): Implementation of the training framework proposed in Self-Rewarding Language Model, from MetaAI - lucidrains/self-rewarding-lm-pytorch
- [Large Language Models (in 2023)](https://www.youtube.com/watch?v=dbo3kNKPaUA&t=899s)): I gave a talk at Seoul National University.I titled the talk “Large Language Models (in 2023)”. This was an ambitious attempt to summarize our exploding fiel...
- [Building Your Own Product Copilot: Challenges, Opportunities, and Needs](https://arxiv.org/abs/2312.14231): A race is underway to embed advanced AI capabilities into products. These product copilots enable users to ask questions in natural language and receive relevant responses that are specific to the use...
- [NEW DSPyG: DSPy combined w/ Graph Optimizer in PyG](https://www.youtube.com/watch?v=rqR3LeR09gc&t=903s&ab_channel=code_your_own_AI): DSPyG is a new optimization, based on DSPy, extended w/ graph theory insights. Real world example of a Multi Hop RAG implementation w/ Graph optimization.New...
- [GitHub - lucidrains/self-rewarding-lm-pytorch: Implementation of the training framework proposed in Self-Rewarding Language Model, from MetaAI](https://github.com/lucidrains/self-rewarding-lm-pytorch): Implementation of the training framework proposed in Self-Rewarding Language Model, from MetaAI - GitHub - lucidrains/self-rewarding-lm-pytorch: Implementation of the training framework proposed in...
- [no title found](https://ai.meta.com/blog/emu-text-to-video-generation-image-editing-research/): no description found
- [Solving olympiad geometry without human demonstrations - Nature](https://www.nature.com/articles/s41586-023-06747-5): A new neuro-symbolic theorem prover for Euclidean plane geometry trained from scratch on millions of synthesized theorems and proofs outperforms the previous best method and reaches the performance of...
- [JupyterHub](https://bit.ly/basementagiclub): no description found

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1204713742307295253) (128 messages🔥🔥): 

- **Mistral.AI for Healthcare Internship**: `@talhawaqas`, a Master’s student in France, is looking for resources and papers on **Mistral.AI's real-world applications**, particularly concerning pretraining and finetuning in health engineering contexts.
- **Transitioning to Cloud**: `@zhiyyang` expressed intent to start testing on the cloud rather than locally for convenience, appreciating the information shared by the community.
- **Mistral Data Use Policy**: `@mrdragonfox` clarified that the service collects user dialogue data and reserves the right to release datasets under a **Creative Commons Attribution (CC-BY)** license.
- **Exploring Response Length Control in Mistral**: `@lucacito` and `@mrdragonfox` discussed how to set up `max_tokens` and temperature for concise responses in the context of `@lucacito`'s portfolio chatbot assistant. `@mrdragonfox` provided examples and advised on how to adjust sampling and temperature settings. 
- **Temperature: To Zero or Not to Zero**: In a heated debate about setting the temperature parameter to zero, `@i_am_dom` suggested that while a temperature of 0 may not be ideal for a chatbot experience, it could improve performance and reduce hallucinations for high-precision tasks. `@mrdragonfox` and others proposed various temperature settings for optimal chatbot responses, such as 0.7 and advised testing for the right fit.

**Links mentioned**:

[Chat with Open Large Language Models](https://chat.lmsys.org/): no description found

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1204874192088989776) (4 messages): 

- **Mistral-like Embedding Model Found**: User `@gbourdin` inquired about finding an embedding model with the same dimension length as **Mistral-embed** for local development. `@mrdragonfox` responded with a [Hugging Face link](https://huggingface.co/intfloat/e5-mistral-7b-instruct) to the **E5-mistral-7b-instruct model**, which has 32 layers and an embedding size of 4096 and explained its usage with example code.
- **Gratitude Expressed for Model Recommendation**: After receiving the model recommendation and usage details, `@gbourdin` expressed their thanks with a "merci :)" followed by "thanks ! 🙂".

**Links mentioned**:

[intfloat/e5-mistral-7b-instruct · Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct): no description found

  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1204781035389325322) (69 messages🔥🔥): 

- **LLMLAB Takes Advantage of Mistral**: `@joselolol.` declared the usage of **Mistral** to run LLMLAB operations for creating commercial synthetic data, inviting users to direct message for account activation after signing up.
- **Curiosity about Data Extraction Method**: `@mrdragonfox` inquired about LLMLAB's data extraction pipeline, but `@joselolol.` sought clarification, indicating a potential discussion on the process.
- **Augmentoolkit Shared as a Resource**: `@mrdragonfox` shared a [GitHub link to augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/) and discussed working on adapting it to work with the Mistral chat API, highlighting the toolkit's capability for vetting and assessing data.
- **Testing New Chat UI Enabled by Mistral**: `@ethux` posted about implementing **ETHUX Chat v0.7.0** which uses various Mistral models and the Huggingface ChatUI, setting a testing API limit to 200 euros and sharing the configuration.
- **Chat UI Rate Limitations and Source Revealed**: In a follow-up, `@ethux` mentioned a rate limit of two messages per minute and shared that the HuggingFace chat UI is open-source on [GitHub](https://github.com/huggingface/chat-ui), while `@gbourdin` expressed enthusiasm for the web search feature within the UI.

**Links mentioned**:

- [ETHUX Chat](https://chat.ethux.net): Made possible by PlanetNode with ❤️
- [GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets](https://github.com/e-p-armstrong/augmentoolkit/): Convert Compute And Books Into Instruct-Tuning Datasets - GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets
- [GitHub - huggingface/chat-ui: Open source codebase powering the HuggingChat app](https://github.com/huggingface/chat-ui): Open source codebase powering the HuggingChat app. Contribute to huggingface/chat-ui development by creating an account on GitHub.

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1204748073985904700) (2 messages): 

- **Inquiry on Next Mistral AI Model Release**: User `@j673912` inquired about the release of the **next Mistral AI model**. `@mrdragonfox` responded, suggesting to keep an eye on **social media** for the official release announcement.
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1204894222545719376) (1 messages): 

- **Embedding Model Troubles with the Canadian Flag**: `@enjalot` encountered an error while attempting to embed models using the dolly15k dataset and shared a [specific error message](https://huggingface.co/datasets/databricks/databricks-dolly-15k/viewer/default/train?p=108&row=10834) related to the Canadian flag. They've successfully processed the previous 10k rows, which suggests the problem is isolated to this particular piece of text.

**Links mentioned**:

[databricks/databricks-dolly-15k · Datasets at Hugging Face](https://huggingface.co/datasets/databricks/databricks-dolly-15k/viewer/default/train?p=108&row=10834): no description found

  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1204747553778700308) (1 messages): 

- **New Forum Channel for Job Seekers**: `@lunarflu` announced the introduction of a **Forum** channel specifically for job opportunities, which users can filter by tags such as `internship`, `machine learning`, `remote`, and `long-term`. The community is encouraged to use and provide feedback on this new feature.
  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1204720625613275197) (115 messages🔥🔥): 

- **GPU Acceleration Uncertainties**: User `@ilovesass` asked about the function of `pipeline.to("cuda")`, with `@sandeep_kr24` clarifying that it moves the entire computational graph to a CUDA-enabled GPU, thereby speeding up the process.
- **Stable Diffusion Clarifications**: `@yamer_ai` offered assistance with using Stable Diffusion and emphasized their personal accomplishments in training models for pixel art generation, while `@tmo97` expressed confusion over how accessible the technology is for non-technical users, receiving guidance from `@yamer_ai`.
- **Technical Requests and Offerings**: Users `@p1ld7a`, `@drfhsp`, and `@.volvite` sought advice for setting up a web UI for querying local LLMs, using specific models for inference with the free T4 on Collab, and resolving issues with Gradio, respectively. They were directed toward resources like Python, Gradio, and Colab notebooks from other sharing individuals like `@electriceccentrics` and `@lee0099`.
- **Backend Overload on HuggingFace Spaces**: `@thomaslau.001` shared a Reddit link discussing an "out of memory" error on Nvidia A10G with Codellama on HuggingFace Spaces, seeking assistance in the matter; `@vipitis` suggested that the 70B model attempts could be exceeding memory limits even at reduced precision levels (fp16).
- **Collaboration Invites and Discussions**: Various users, including `@soul_syrup`, `@technosourceressextraordinaire`, `@jdreamer200`, and `@electriceccentrics`, mentioned their projects ranging from neural signal analysis and robotics with RL, to job searches and financial humor about cloud service costs, prompting social engagement and project interest within the community.

**Links mentioned**:

- [ - YouTube](https://www.youtube.com/watch?v=CZaG3&ab_channel=p3nGu1nZz): no description found
- [Oppenheimer Oppenheimer Movie GIF - Oppenheimer Oppenheimer movie Barbie oppenheimer meme - Discover &amp; Share GIFs](https://tenor.com/view/oppenheimer-oppenheimer-movie-barbie-oppenheimer-meme-cillian-murphy-theory-gif-8102327772591152629): Click to view the GIF
- [How to install stable diffusion 1.6 Automatic1111 (One Click Install)](https://www.youtube.com/watch?v=IoWPsNwXLVc&t=36s): 🔔 Subscribe for AIconomist 🔔SD Automatic1111 1.6 ➤ https://github.com/AUTOMATIC1111/stable-diffusion-webuiPython 3.10.6 ➤ https://www.python.org/downloads/...
- [no title found](https://tenor.com/view/oppenheimer-oppenheimer-movie-barbie-oppenheimer-meme-cillian-murphy-theory-g): no description found
- [Easter Funny Shrek GIF - Easter Funny Shrek Funny Face - Discover &amp; Share GIFs](https://tenor.com/view/easter-funny-shrek-funny-face-gif-16873758): Click to view the GIF
- [GitHub - cat-game-research/Neko: A cat game beyond.](https://github.com/cat-game-research/Neko): A cat game beyond. Contribute to cat-game-research/Neko development by creating an account on GitHub.
- [Reddit - Dive into anything](https://www.reddit.com/r/huggingface/comments/1alpsvp/cuda_out_of_memory_on_nvidia_a10g_codellama_on/): no description found
- [GitHub - Unlimited-Research-Cooperative/Human-Brain-Rat: Bio-Silicon Synergetic Intelligence System](https://github.com/Unlimited-Research-Cooperative/Human-Brain-Rat): Bio-Silicon Synergetic Intelligence System. Contribute to Unlimited-Research-Cooperative/Human-Brain-Rat development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1204708404023525448) (2 messages): 

- **Meta-RL Application Development Opportunity**: `@davidebuoso` is developing a **meta-RL application** and is seeking collaborators for a side project related to advanced works listed on their curated GitHub list ([Meta-RL Panda](https://github.com/stars/lambdavi/lists/meta-rl-panda/)), specifically to test on a Panda robotic arm using gym.

- **Personal Achievements and Project Showcase**: `@antiraedus` shared their weekly update, mentioning tackling bad habits, landing a **tutoring job at their university**, and attending social events. They are also working on a game post a Flutter platformer tutorial and plan to share it by next week. The overarching theme for the year is to aim for **internships** while creating *sharable and tangible* items as motivation and proof of work.
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1204741110984089630) (4 messages): 

- **Discover Cutting-Edge Research by Evan Hubinger et al.**: `@opun8758` shared a [new research paper](https://arxiv.org/abs/2401.05566) co-authored by **Evan Hubinger** and colleagues, highlighting its potential interest to the community.
- **In Search of the RWKV Model**: `@vishyouluck` inquired about experiences with fine-tuning the **RWKV model**, sparking curiosity among modeling enthusiasts.
- **Eagle 7b Announcement Lacks Details**: `@vishyouluck` mentioned **Eagle 7b** but provided no further context or information about this model.
- **Code Llama Leaps into the Future with 70B Model**: `@jashanno` shared a [beehiiv article](https://natural20.beehiiv.com/p/code-llama-70b) announcing **Facebook/Meta's** release of **Code Llama**, a new 70 billion parameter language model designed to aid coders and learners, available under a community-friendly license.

**Links mentioned**:

- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566): Humans are capable of strategically deceptive behavior: behaving helpfully in most situations, but then behaving very differently in order to pursue alternative objectives when given the opportunity. ...
- [META&#x27;s new OPEN SOURCE Coding AI beats out GPT-4 | Code Llama 70B](https://natural20.beehiiv.com/p/code-llama-70b): PLUS: Privacy Concerns about ChatGPT, Pressure on Tech Giants and more...

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1204754489085530112) (9 messages🔥): 

- **Web Integration Achieved**: `@wubs_` confirmed that the generator mentioned by `@aliabbas60` works in **web and Progressive Web App (PWA)** formats, functioning well on both mobile and desktop devices.
- **New AI Project Showcase**: `@critical3645` mentioned creating a project from scratch and linked a video demonstration.
- **SD.Next Unveils Major Enhancements**: `@vladmandic` released a comprehensive update for [SD.Next](https://github.com/vladmandic/automatic), introducing a robust **Control module**, a **Face module**, improved **IPAdapter** modules, new intelligent masking options, and numerous new models and pipelines. They highlighted performance enhancements, with a benchmark of **110-150 iterations/second** on nVidia RTX4090, and directed users to view the full documentation and updates in their [Wiki](https://github.com/vladmandic/automatic/wiki).
- **Community Guidelines Reminder**: `@cakiki` reminded `@vladmandic` to adhere to community guidelines by removing any Discord invite links, which `@vladmandic` promptly addressed.
- **Role-Play Project Goes Docker**: Krolhm announced the dockerization of the role play (RP) project, ImpAI, for easier use with `hf pipeline` and `llama.cpp`, sharing the GitHub repository [ImpAI](https://github.com/rbourgeat/ImpAI).

**Links mentioned**:

- [GitHub - rbourgeat/ImpAI: 😈 ImpAI is an advanced role play app using large language and diffusion models.](https://github.com/rbourgeat/ImpAI): 😈 ImpAI is an advanced role play app using large language and diffusion models. - GitHub - rbourgeat/ImpAI: 😈 ImpAI is an advanced role play app using large language and diffusion models.
- [Create new page · vladmandic/automatic Wiki](https://github.com/vladmandic/automatic/wiki/OpenVINO),): SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models - Create new page · vladmandic/automatic Wiki
- [Create new page · vladmandic/automatic Wiki](https://github.com/vladmandic/automatic/wiki/Intel-ARC)): SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models - Create new page · vladmandic/automatic Wiki
- [Create new page · vladmandic/automatic Wiki](https://github.com/vladmandic/automatic/wiki/ONNX-Runtime-&-Olive)): SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models - Create new page · vladmandic/automatic Wiki
- [Create new page · vladmandic/automatic Wiki](https://github.com/vladmandic/automatic/wiki/Benchmark)): SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models - Create new page · vladmandic/automatic Wiki

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1204769948849668167) (4 messages): 

- **Mamba SSM showcased by tonic_1**: User `@tonic_1` presented the [mamba library](https://github.com/state-spaces/mamba/tree/main/mamba_ssm), highlighting his personal interest in `utils`, `ops`, and `modules` features.
- **Chad seeks state-space math insights and trade-offs**: User `@chad_in_the_house` expressed an interest in the mathematical aspects of state space models provided by mamba and how they compare with transformers and rwkv.
- **Clarity on Mamba sought, repetition not an issue**: User `@chad_in_the_house` also indicated that it's okay to repeat information from videos during the presentation, acknowledging most attendees may not be familiar with mamba.
- **A Quick History of Mamba Requested**: In addition to understanding the trade-offs, `@chad_in_the_house` requested a brief history of mamba's development up to the current point.

**Links mentioned**:

[mamba/mamba_ssm at main · state-spaces/mamba](https://github.com/state-spaces/mamba/tree/main/mamba_ssm): Contribute to state-spaces/mamba development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1204833801176481833) (1 messages): 

- **In Search of Understanding Transformers**: User `.slartibart` inquired about **presentations that cover code-writing transformers**, specifically those that delve into what a transformer learns from the sampled code. They are looking for resources to better understand the learning process of transformers in the context of code generation.
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1204862876548993085) (4 messages): 

- **Seeking OCR for Line Segmentation**: User `@swetha98` inquired about an OCR tool to segment a document image into four separate images, each containing one of the four lines of text present on a sample invoice.
- **Surya OCR Recommended**: `@vikas.p` suggested using [Surya OCR](https://github.com/VikParuchuri/surya), which can perform accurate line-level text detection and recognition for any language, in response to `@swetha98`'s request for an OCR solution.
- **Positive Reception for the Recommendation**: `@swetha98` expressed gratitude to `@vikas.p` for the suggestion and mentioned the intention to cite his GitHub repository if the solution meets their needs.

**Links mentioned**:

[GitHub - VikParuchuri/surya: Accurate line-level text detection and recognition (OCR) in any language](https://github.com/VikParuchuri/surya): Accurate line-level text detection and recognition (OCR) in any language - GitHub - VikParuchuri/surya: Accurate line-level text detection and recognition (OCR) in any language

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1204734058957901854) (11 messages🔥): 

- **Seeking a Relevance Checker for RAG Pipelines**: `@tepes_dracula` is building a **RAG pipeline** and is looking for a **classifier** to validate the relevance of query and context before passing it to an LLM. They mention a desire for a tool that draws upon datasets like **truthfulqa** and **fever**.

- **Speeding Up Deci lm on Windows**: `@kingpoki` wants to improve the inference speed of **Deci lm** running locally on Windows and is looking for suggestions beyond **model quantization**.

- **Measuring Concept Similarity in Tech Fields**: `@serhankileci` is exploring how to calculate similarity percentages among concepts, tools, and languages within various tech-related fields. They consider using word embeddings and cosine similarity, acknowledging the lack of a single **encyclopedic dataset** for this purpose.

- **Data Quality in LLM Instruction Tuning**: `@Chris M` highlights the importance of **data quality for LLM instruction tuning** and shares an article on automated techniques for detecting low-quality data, citing **bad data as the common culprit** affecting performance. The article can be found at [cleanlab.ai](https://cleanlab.ai/blog/filter-llm-tuning-data/).

- **Inquiry about HuggingFace's TAPAS Model**: `@nitachaudhari29` casually enters the discussion with a "hi" and follows up with a question asking if anyone has experience using the **TAPAS model** from HuggingFace.

- **State of Multi-Label Text Classification**: `@simpleyuji` inquires about the current **state-of-the-art (SOTA) for multi-label text classification**.

- **Pros and Cons of Small Batch Sizes**: `@abrahamowodunni` poses a question about the drawbacks of using a batch size of 4 when fine-tuning a model with a small dataset, other than increased training time. `@vipitis` responds, indicating that a batch size of 4 for 1.2k steps could be sufficient, but generally, a **larger batch size** is preferred if it fits in VRAM.

**Links mentioned**:

[How to detect bad data in your instruction tuning dataset (for better LLM fine-tuning)](https://cleanlab.ai/blog/filter-llm-tuning-data/): Overview of automated tools for catching: low-quality responses, incomplete/vague prompts, and other  problematic text (toxic language, PII, informal writing, bad grammar/spelling) lurking in a instru...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1204833801176481833) (1 messages): 

- **In Search of Transformer Insights**: User `.slartibart` asked if there are presentations available that cover **code-writing transformers** and delve into what transformers learn from the *sampled code*. They are looking for material that tries to describe the learning process of transformers.
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1204702480257122344) (79 messages🔥🔥): 

- **Speculation on Bard Ultra's Release**: `@la42099` humorously comments on the anticipation surrounding **Bard Ultra's** release, acknowledging that predictions on these matters are often incorrect.
- **ChatGPT Access Issues**: `@chrisrenfield` experiences difficulty accessing **ChatGPT**, and `@satanhashtag` suggests visiting **[OpenAI's service status](http://status.openai.com/)** for updates.
- **Seeking AI for Jewelry Visualization**: `@jonas_54321` is looking for an AI tool to create photo-realistic images of jewelry on models. `@satanhashtag` recommends **Dall-E 3**, provided for free on Bing, and mentions personal preference for **Midjourney**.
- **Terms of Service Constraints**: A discussion emerges around the **Terms of Service (TOS)** led by `@wrexbe`, with thoughts on creatively navigating the restrictions and content warnings associated with the generation of NSFW content. Other users, such as `@drinkoblog.weebly.com` and `@chotes`, discuss the efficiency and ethical angles of using AI for various content.
- **The Path to AGI Discussed**: The conversation between `@drinkoblog.weebly.com` and `@chotes` explores the development of AGI, and users express their concerns about censorship of AI models, pondering the implications for future AGI capabilities and freedoms.
  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1204726906856022046) (30 messages🔥): 

- **GPT Lagging Issues Raised**: Users `@theotherway` and `@pneumaone` reported performance issues with GPT-4, describing it as slow and suggesting it was performing similarly to an older version, possibly GPT-1. `@pneumaone` noted a recent update seemed to significantly degrade the model's abilities.
   
- **Users Notice GPT Deterioration After Update**: `@pneumaone` and `@_loier` discussed a noticeable drop in GPT's performance, suggesting it was forgetting instructions and not following custom settings as it did before.
  
- **Call for a 'Read Back' Feature**: `@blaynomtops` requested a feature that reads the text back to users, aiming to alleviate eye strain from reading all day. It was noted by `@blckreaper` that this feature exists on mobile but not on PC.
  
- **Token Management in Narrative Generation**: `@blckreaper` criticized GPT for using tokens inefficiently during narrative generation, questioning why it often reflects on characters instead of progressing the story.
  
- **Intermittent Connectivity and Performance Issues**: `@bennnjji` inquired about an error related to connecting to the GPT API, while multiple users like `@pneumaone` and `@_loier` reported that their instances of GPT were stalling, making mistakes, and exhibiting degraded interaction quality.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1204817906576130178) (8 messages🔥): 

- **Chatbot Identity Mystery**: User `@chotes` playfully speculated on who among the users might be an **alt** (alternative account) of someone named Sam.
- **Greetings in the Chat**: `@darthgustav.` entered the conversation with a friendly greeting, to which `@chotes` responded, possibly suspecting darthgustav. of being Sam's alt.
- **Talking Alts**: `@lugui` joined the jest with `@chotes` hinting at the use of an "Alt" account by someone named Sam.
- **Reporting Prompt Leakage Concerns**: `@raidedcluster` shared their discovery of a method for prompt leakage and policy violations present in custom GPTs, citing its effectiveness on Kahn Academy's Khanmigo.
- **Official Feedback Channel Suggested**: `@eskcanta` provided `@raidedcluster` with the [OpenAI model feedback form](https://openai.com/form/chat-model-feedback) as the proper channel for reporting the issues discovered.

**Links mentioned**:

[Chat model feedback](https://openai.com/form/chat-model-feedback): no description found

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1204817906576130178) (8 messages🔥): 

- **Who's the Alt? Identity Quest in Progress**: `@chotes` humorously probes the group to identify if anyone is an alt account for Sam, which sparks a light-hearted exchange with the response `@darthgustav.` and the follow-up admission from `@chotes`: "i should have known tbh.."
- **Sam's Secret Identity?**: `@lugui` chimes in, suggesting that Sam might actually use an _Alt_ account, adding to the playful speculation.
- **Reporting a Security Concern**: User `@raidedcluster` has discovered a potential method for prompt leakage and content policy violations and seeks advice on how to report it, specifically mentioning its effectiveness on Khan Academy's Khanmigo.
- **A Helping Hand for Reporting Issues**: `@eskcanta` provides a detailed recommendation to `@raidedcluster` on where to report issues related to prompt leakage and content policy violations, including a link and instructions at [OpenAI's model feedback form](https://openai.com/form/chat-model-feedback).
- **Expressing Appreciation for Support**: In response to `@eskcanta`'s guidance, user `@raidedcluster` expresses gratitude for the information on how to report the found issues with a simple "Thank you!"

**Links mentioned**:

[Chat model feedback](https://openai.com/form/chat-model-feedback): no description found

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1204710908149112872) (31 messages🔥): 

- **Google Scholar Prompt Inquiry Awaits Response**: `@aldikirito` asked for a specific prompt to search for **journal citations in Google Scholar**, but no clear guidance was given.
- **Gemini Pro Compared to 3.5 Turbo**: According to `@moyaoasis`, **Gemini Pro** excels in writing but falls short in math and coding compared to **3.5 Turbo**, especially for new coders.
- **Perplexity's API Use in Question**: `@horsecode` inquired about using Perplexity's API with existing helper bots, citing the ToS; `@clay_ferguson` further discussed the implications of the ToS, speculating on the restrictive use policy for competitive reasons.
- **Defining "Answer Engines"**: `@twelsh37` responded to `@tyronemichael` by explaining **Perplexity** as an *answer engine*, distinguishing it from traditional search engines like **Google**, and linked to a [YouTube video](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=344s) by Riley Brown for more insights.
- **SEO Article Tips in Perplexity Requested**: `@tuastowers_71714` sought advice for creating SEO-optimized articles with Perplexity; `@noremac258` shared a personal prompt and suggested using **Grammarly** to check the content quality.

**Links mentioned**:

- [Terms of Service](https://blog.perplexity.ai/legal/terms-of-service): Privacy • Terms &amp; Conditions
- [I use Perplexity MORE than Google and ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=344s): Main Takaways From this Video: &quot;I use Perplexity more than ChatGPT, BARD, and Microsoft Copilots for five main reasons, including its use in content creation...

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1204788740007133225) (5 messages): 

- **Perplexity AI Draws Criticism**: User `@deeceem22_16584` expressed disappointment with Perplexity AI, mentioning that a use case example about visiting three museums in one afternoon was impractical.
- **Skepticism About Museum Hopping**: In response to the criticism about Perplexity AI's use case, `@brknclock1215` humorously remarked that one would be lucky just to make it past the entrance of the Louvre in an afternoon.
- **Innovative Use of Perplexity API**: `@foxplaid19973` shared an interesting application of Perplexity AI's API by integrating it with a Lego Mindstorm robot to create a push-to-talk assistant that uses Google's speech recognition and speaks out the answers from Perplexity.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1054944216876331118/1204601391633530920): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1204797350808395807) (43 messages🔥): 

- **API Lacks Source Citations**: `@tyronemichael` queried about when the API would include sources and citations. Despite previous indications to the contrary, `@me.lk` clarified that source citations are currently **not on the roadmap** for the API.

- **GPT Models are API MIA**: `@fanglin3064` inquired about accessing GPT models through the Perplexity API, only to learn from `@me.lk` that **Perplexity does not offer GPT models** via the API service.

- **No Scraping Allowed**: `@fanglin3064` expressed an interest in scraping Perplexity AI results that use GPT models to verify accuracy. `@icelavaman` cautioned that **scraping is against the TOS** and is not permissible.

- **API vs. UI - Two Different Products**: In a discussion about whether the API is officially provided by Perplexity AI, `@me.lk` and `@icelavaman` clarified that both **[pplx.ai](https://pplx.ai)** and the API are different products offered by the same company but follow different access protocols.

- **Choosing the Best Model**: `@fanglin3064` sought to know which model in the API list is closest to GPT-3.5 turbo in performance. `@icelavaman` recommended trying out **mixtral-8x7b-instruct** for good performance without web access and **pplx-7(0)b-online** models for fast responses with web access, while also providing a [link to benchmarks](https://blog.perplexity.ai/blog/introducing-pplx-online-llms) for the mentioned models.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1047202784090538054/1204821989919957002): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Moon (Dark Mode)](https://docs.perplexity.ai/discuss/65c0b02f09d8e3001ca0d3ba): no description found
- [Introducing PPLX Online LLMs ](https://blog.perplexity.ai/blog/introducing-pplx-online-llms): The first-of-its-kind Online LLM API

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1204829291209297971) (4 messages): 

- **Boosting RAG with User Feedback**: The LlamaIndex team emphasized the importance of an **agentic layer atop RAG** to facilitate real-time user feedback during search/retrieval tasks, enhancing the user experience for complex queries. Discussion of this feature can be viewed in their tweet [here](https://twitter.com/llama_index/status/1755270274310901898).

- **Deep Dive into Query Pipelines**: A comprehensive guide on LlamaIndex's **Query Pipelines** feature is provided, showcasing advanced retrieval strategies through a more declarative approach. The explanation includes various retrieval methods and can be further explored via the shared [Medium post](https://t.co/Funqm7Jw1u), as elaborated on Twitter [here](https://twitter.com/llama_index/status/1755342965667696870).

- **Insights on RAG from LlamaIndex**: `@seldo` from LlamaIndex discussed the nuances of RAG in a webinar with `@ankit1khare` hosted on RocksetCloud, covering its purpose, execution, and upcoming features for 2024. Interested viewers are directed to the full webinar on [YouTube](https://t.co/7f12VvgImc), as mentioned [here](https://twitter.com/llama_index/status/1755378511819456841).

- **Invitation to Join LlamaIndex Event**: An open invitation is shared for an upcoming event hosted by LlamaIndex, encouraging participation. Details of the event can be found [here](https://twitter.com/llama_index/status/1755393423660724379).

**Links mentioned**:

[Setting up Query Pipeline For Advanced RAG Workflow using LlamaIndex](https://t.co/Funqm7Jw1u): What is QueryPipelines?

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1204725834540253184) (65 messages🔥🔥): 

- **NLSQLTableQueryEngine Troubleshooting**: User `@nzmrs7` suggested feeding tables when using `NLSQLTableQueryEngine` and proposed checking the query generated with `print(response.metadata)` to clarify the generated SQL query.
- **Gemini Connection Issues**: User `@whitefang_jr` recommended checking the google console for moderation filters which might be causing issues when connecting to Gemini as faced by `@mowlidharan`.
- **Reranker Inclusion Strategies**: In response to `@theoxd` inquiring about including a reranker, `@kapa.ai` provided detailed examples of setting up `LLMRerank` and `SentenceTransformerRerank` within a retriever using **LlamaIndex** libraries.
- **Ingestion Pipeline for Deeplake**: User `@ramihassanein` discussed issues with using `index.refresh_ref_docs` for de-duplication when dealing with vector stores like Deeplake; this led to a collaborative fix for the issue and submission of a PR by the user.
- **Custom LLM Returning JSON**: `@nbulkz` shared a challenge with a finetuned LLM returning a JSON object instead of a string, which caused an error. `@cheesyfishes` proposed the possibility of creating a custom LLM or parsing the string back into JSON later.

**Links mentioned**:

- [llama_index/llama_index/indices/base.py at cc739d10069a7f2ac653d6d019fbeb18a891fea2 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/cc739d10069a7f2ac653d6d019fbeb18a891fea2/llama_index/indices/base.py#L310): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1059199217496772688/1163880111074971790/1163900056718553169): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [llama_index/llama_index/ingestion/pipeline.py at 1843c6d806702dc12bd771ac1362b6c3c2931ca2 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/1843c6d806702dc12bd771ac1362b6c3c2931ca2/llama_index/ingestion/pipeline.py#L191): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [Ingestion Pipeline + Document Management - LlamaIndex 🦙 0.9.46](https://docs.llamaindex.ai/en/stable/examples/ingestion/document_management_pipeline.html): no description found
- [Finetuning an Adapter on Top of any Black-Box Embedding Model - LlamaIndex 🦙 0.9.46](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding_adapter.html): no description found
- [upgraded deeplake vector database to use BasePydanticVectorStore by rumsrami · Pull Request #10504 · run-llama/llama_index](https://github.com/run-llama/llama_index/pull/10504): Description Upgraded the vector store database Deeplake to use the BasePydanticVectorStore instead of the VectorStoreBase. This would allow it to be used in Ingestion pipelines Type of Change Pleas...
- [Query Pipeline - LlamaIndex 🦙 0.9.46](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/root.html): no description found
- [Defining a Custom Query Engine - LlamaIndex 🦙 0.9.46](https://docs.llamaindex.ai/en/stable/examples/query_engine/custom_query_engine.html#defining-a-custom-query-engine): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1204736828859482132) (1 messages): 

- **Quest for Production-Grade RAG System Advice**: `@turnerz` is seeking advice on setting up a production-grade **RAG HR chatbot system**. They are looking for recommendations on vector databases, chunk sizes, embedding models and have specific queries on project progression and database organization.
- **Stepping Stone Strategy Vs. Headfirst Dive**: In progressing with their RAG HR chatbot project, `@turnerz` is contemplating whether to **start with a naive RAG system and iteratively improve it**, or to **begin with a more complex multi-hop retrieval system** based on an article they read.
- **Database Dilemma: One or Many?**: `@turnerz` is also inquiring whether it's better to **build different vector databases for different documents** or combine them into one, citing an example of handling two separate PDFs.
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1204765064951767060) (12 messages🔥): 

- **JupyterLab Use Unnecessary for Some**: `@caseus_` mentioned that a message can be ignored **if JupyterLab is not being used**, which prompted `@youraveragedev` to express a preference for using JupyterLab.
- **Suggestion to Modify Cloud Entry Point**: `@caseus_` suggested changing the cloud-entrypoint script to **point to a different directory**, specifically `/workspace/axolotl`.
- **Bargain GPU Cloud Service Alert**: `@dreamgen` shared information about a cloud service offering H100s for approximately **$2.8 per GPU hour**, inviting members to direct message for the link.
- **Billing Details Matter with Cloud Services**: `@yamashi` warned that the cloud service mentioned by `@dreamgen` **rounds up billing**, which `@nanobitz` acknowledged with disappointment. `@le_mess` added hoping for a discount.
- **Stargazing Increases for a Project**: `@dreamgen` mentioned adding a stargaze to a project, indicating they hadn't done so before, which prompted a light-hearted response from `@yamashi`.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1204752545310965810) (24 messages🔥): 

- **RunPod Troubles Persist**: `@casper_ai` mentioned that silent updates on RunPod are causing failures. `@m4ttfl0` faced the issue repeatedly, noticing the repository is getting clobbered, and the expected 500GB "persistent volume" is not present—a problem confirmed through several diagnostic commands.

- **Spinning the Wheel of Pods**: `@dreamgen` spins up new pods regularly on community cloud and occasionally on secure cloud without encountering these issues, while `@m4ttfl0` finally spun up a non-clobbered repository but still faced the missing volume issue.

- **Seeking Solid Ground**: `@m4ttfl0` and `@caseus_` discussed potential solutions like changing the mountpoint of the persistent volume or the setup directory in the Docker image. They suggested mounting the volume under `/workspace/axolotl/data` for better disk space management.

- **Cache and Output Considerations**: `@caseus_` pointed out that the Huggingface cache is currently pointed to `/workspace/data`, requiring that base models and datasets go to the persistent volume as well. A change might involve updating examples to use `output_dir: ./data/out`.

- **Documenting the RunPod Issue**: `@m4ttfl0` agreed to document the issue and proposed solutions on GitHub, aiming to help others understand the changes and reasoning. They offered to help test any new changes to ensure they work as expected.

- **Continual Pretraining PR**: `@jinwon_k` submitted a PR for a new scheduler optimized for continual pretraining of large language models. The PR can be viewed at the provided [GitHub link](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1273).

**Links mentioned**:

[Scheduler implementation of Continual Pre-Training of Large Language Models: How to (re)warm your model?  by jinwonkim93 · Pull Request #1273 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1273): Scheduler implementation of Continual Pre-Training of Large Language Models: How to (re)warm your model? (https://arxiv.org/pdf/2308.04014.pdf)  Description almost identical to consine min lr but i...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1204716289503797248) (10 messages🔥): 

- **In Search of Open Pretraining Datasets**: `@Sebastian` inquired about current **open pretraining datasets**; `@le_mess` redirected him to a relevant discussion in another channel for more information.
- **Designing a Prompt Format for Python Queries**: `@nafnlaus00` sought feedback on creating a **concise data structure** for model querying by Python, discussing issues with JSON's escaping of non-English characters and exploring alternatives such as using protobuf and custom encoding schemes.
- **Protobuf Possibility and its Conversion**: `@nafnlaus00` presented a method involving protobuf for keeping text unaltered while representing structured data, demonstrating a **reversible operation** for encoding and decoding bytes to UTF-8 strings.
- **Custom Encoding for Better Flexibility**: Weighing options, `@nafnlaus00` was inclined towards a custom encoding method using obscure Unicode characters to delineate data structure elements, considering its advantages in format control and extensibility.
- **Suggestions and Considerations on Encoding Approach**: While `@caseus_` endorsed the idea of adding essential tokens for control characters, `@dreamgen` suggested using **plain text with custom tokens** for encoding, seeking clarity on the use case.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1204724219535958037) (12 messages🔥): 

- **Learning Rates and Model Configurations**: User `@dreamgen` discussed how learning rates affected their project results with a small dataset. They later shared their configurable parameters, noting the use of **unsloth for DPO**, and the intention to open-source the script soon.

- **DreamGenTrain Script Shared**: User `@dreamgen` provided a link to their GitHub where they hosted the script for their training method, mentioning the use of **paged_adamw_8bit** and a **linear scheduler**. The script is available [here](https://github.com/DreamGenX/DreamGenTrain/blob/master/dpo.py).

- **Advice on Batch Sizes**: `@dreamgen` suggested that the **micro batch size could be increased** unless dealing with very long sequences, as was the case with their own project.

- **Gratitude for Information Sharing**: User `@fred_fups` expressed thanks to `@dreamgen` for sharing details about their training setup and strategies.

- **Inquiries and Collaboration on Self-Rewarding Methods**: User `@dctanner` expressed interest in experimenting with self-rewarding methods from a paper they mentioned, inviting collaboration and considering an implementation into **axolotl**.

**Links mentioned**:

[DreamGenTrain/dpo.py at master · DreamGenX/DreamGenTrain](https://github.com/DreamGenX/DreamGenTrain/blob/master/dpo.py): Contribute to DreamGenX/DreamGenTrain development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1204828094347739227) (2 messages): 

- **Building runpod images requires specific setup**: User `@gonejiggy` asked about how runpod images were built since local building didn't work and docker can't run inside another docker. `@caseus_` responded saying **a machine with a GPU and bare metal docker** is needed for building runpod images.
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1204706348877611019) (22 messages🔥): 

- **LLaMa's Quick Draw**: `@top_walk_town` successfully trained a 2M parameter LLaMa model for conditional generation of the MNIST digit "7", boasting about a 25-minute training time on an 8GB MacBook M1.
- **Llava 1.5 vs 1.6**: `@SegmentationFault` shared a Twitter [link](https://vxtwitter.com/billyuchenlin/status/1755207605537120513) regarding a project, later commenting that it's a pity this project didn't use Llava 1.6, indicating the new version could have offered improvements.
- **DALL-E 3 Watermarking**: `@vrus0188` provided a [link to The Verge's article](https://www.theverge.com/2024/2/6/24063954/ai-watermarks-dalle3-openai-content-credentials) discussing new watermarks added by OpenAI's DALL-E 3 to image metadata, including both invisible and visible components.
- **Hiring via Hugging Face**: In response to `@_definitely_not_sam_` looking to hire a developer for a computer vision project, `@chad_in_the_house` suggested using the Hugging Face job channel in their server for formal job postings.
- **Emojis Speak Louder**: The use of various emojis by users like `@astropulse`, `@pseudoterminalx`, and `@drhead` indicates reactions to different topics, with no specific content to summarize.

**Links mentioned**:

[OpenAI is adding new watermarks to DALL-E 3](https://www.theverge.com/2024/2/6/24063954/ai-watermarks-dalle3-openai-content-credentials): The watermarks can still be erased, however. 

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1204748896790913054) (27 messages🔥): 

- **New Background Removal Tool Unveiled**: `@qwerty_qwer` shared a link to a new state-of-the-art background removal tool, BRIA Background Removal v1.4, which rivals leading models and is designed for commercial use with a focus on content safety. The model card and a demo are available at [BRIA Background Removal v1.4](https://huggingface.co/briaai/RMBG-1.4).

- **EVA-CLIP-18B Sets New CLIP Model Standard**: `@thejonasbrothers` and `@vrus0188` discussed the [EVA-CLIP-18B paper](https://arxiv.org/abs/2402.04252), which presents an 18-billion parameter CLIP model with groundbreaking zero-shot top-1 accuracy. This work signals a significant leap in CLIP model performance using a constant dataset size.

- **InstanceDiffusion for Controlled Image Generation**: `@vrus0188` spotlighted InstanceDiffusion, a method for precise instance-level control in text-to-image generation allowing for complex instance specifications. A related discussion by `@SegmentationFault` on a similar feature called "RPG-DiffusionMaster" ensued, emphasizing differences in functionality.

- **Seeking Hyperparameter Insights**: User `@yoavhacohen` sought recommendations for training specifics to reproduce the SD XL VAE, with `@drhead` suggesting insights on larger batch size, EMA weights, and potential changes to the discriminator and reconstruction losses.

- **Discussions on Data Storage and Querying Speeds**: `@progamergov` and `@chad_in_the_house` discussed the efficiency of a proposed data storage layout comprising a parquet file and associated images, emphasizing the importance of querying speeds to reduce costs.

- **Introducing Controversy-proof GOODY-2 LLM**: `@progamergov` introduced GOODY-2, an AI model designed with strong adherence to ethical principles that refuses to answer potentially controversial questions. The model architecture is detailed in a provided [model card](https://www.goody2.ai/goody2-modelcard.pdf), and the announcement sparked humorous responses from users such as `@itali4no` and `@marianbasti`.

**Links mentioned**:

- [GOODY-2 | The world&#x27;s most responsible AI model](https://www.goody2.ai/): Introducing a new AI model with next-gen ethical alignment. Chat now.
- [EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters](https://arxiv.org/abs/2402.04252): Scaling up contrastive language-image pretraining (CLIP) is critical for empowering both vision and multimodal models. We present EVA-CLIP-18B, the largest and most powerful open-source CLIP model to ...
- [briaai/RMBG-1.4 · Hugging Face](https://huggingface.co/briaai/RMBG-1.4): no description found
- [CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations](https://arxiv.org/abs/2402.04236): Vision-Language Models (VLMs) have demonstrated their widespread viability thanks to extensive training in aligning visual instructions to answers. However, this conclusive alignment leads models to i...
- [Vision Superalignment: Weak-to-Strong Generalization for Vision Foundation Models](https://arxiv.org/abs/2402.03749): Recent advancements in large language models have sparked interest in their extraordinary and near-superhuman capabilities, leading researchers to explore methods for evaluating and optimizing these a...
- [V-IRL: Grounding Virtual Intelligence in Real Life](https://virl-platform.github.io/): no description found
- [InstanceDiffusion: Instance-level Control for Image Generation](https://people.eecs.berkeley.edu/~xdwang/projects/InstDiff/): no description found
- [ People @ EECS at UC Berkeley ](https://people.eecs.berkeley.edu/): no description found

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1204891293973741619) (20 messages🔥): 

- **Enthusiasm for CUDA Builds**: `@cropinky` expressed excitement about building a deep learning rig and is currently looking for a used 3090 GPU on local eBay listings.
- **Motherboard and CPU Bundle for Deep Learning**: `@joseph_en` shared a [link](https://a.co/d/iR3bvvF) to a bundle that includes an **Intel Core i9-12900K** processor and an **ASUS ROG Maximus Z690 Hero DDR5** ATX motherboard, which is suitable for running two GPUs.
- **Exploring Multi-GPU Configuration Options**: `@cropinky` discussed fitting multiple GPUs into one PC, referencing products like **PCIe bifurcation cards** and the necessity of having enough PCIe lanes from the **CPU+motherboard** combination. They also suggested creating a `deep-learning-builds` chat channel for these discussions.
- **Riser Cables and Builds**: `@jeremyhoward` mentioned the use of mining rig setups and risers to fit more GPUs, while `@iron_bound` advised to get quality **gen4 PCIe cables** to ensure optimal connectivity, recommending to check the specs carefully with examples like the one from Cooler Master.
- **Multi-GPU Challenges and Resource Sharing**: `@cropinky` and `@joseph_en` discussed physical limitations and solutions when adding multiple GPUs to a system, including utilizing **GPU extenders** and potentially running beyond the motherboard's standard GPU slots with modifications.

**Links mentioned**:

- [Amazon.com: Micro Center Intel Core i9-12900K Desktop Processor 16 (8P+8E) Cores up to 5.2 GHz Unlocked LGA1700 Desktop Processor with ASUS ROG Maximus Z690 Hero DDR5 ATX Gaming Motherboard : Electronics](https://a.co/d/iR3bvvF): no description found
- [My deep learning rig &#8211; Non_Interactive &#8211; Software &amp; ML](https://nonint.com/2022/05/30/my-deep-learning-rig/): no description found
- [The Best GPUs for Deep Learning in 2023 — An In-depth Analysis](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/): Here, I provide an in-depth analysis of GPUs for deep learning/machine learning and explain what is the best GPU for your use-case and budget.
- [C-Payne PCB Design](https://c-payne.com/): C-Payne PCB Design
- [PCIe Bifurcation Card - x8x8 - 2W](https://c-payne.com/products/pcie-bifurcation-card-x8x8-2w): one PCIe x16 input two PCIe x8 electical / x16 mechanical outputs PCIe gen3 BIOS support neccesary (Please verify your BIOS has the appropriate options available!) 40.64mm (dual width) spaced slots ma...
- [A Full Hardware Guide to Deep Learning &mdash; Tim Dettmers](https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/): In this guide I analyse hardware from CPU to SSD and their impact on performance for deep learning so that you can choose the hardware that you really need.

  

---


### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1205014257180090398) (6 messages): 

- **PyTorch 2 Paper Set for ASPLOS 2024**: `@lancerts` shared a [blog post](https://pytorch.org/blog/pytorch-2-paper-tutorial/) announcing that the PyTorch 2 paper has been accepted at ASPLOS 2024, detailing technologies like **TorchDynamo**, **TorchInductor**, and **Dynamic Shape support**. Neat features of PyTorch 2, including **torch.compile**, will be further explored in a tutorial at the conference.

- **Last-Minute PyTorch Conf Drama**: `@marksaroufim` recounted the hectic moments leading up to PyTorch Conf, with the torch.compile API and its documentation being merged just before Soumith's presentation on stage.

- **Python-Based Compiler Ease of Use**: `@marksaroufim` noted the PyTorch 2 compiler is entirely written in Python, making it more accessible and hackable for developers interested in exploring its internals.

- **Optimizations for Consumer Hardware**: `@marksaroufim` expressed enthusiasm about the potential for PyTorch 2 to be optimized for consumer hardware, indicating that such improvements could be within reach.

- **Evolving Debug Tools with PyTorch**: `@marksaroufim` appreciated improvements in the debug tools over time, with the introduction of **TORCH_LOGS** making it easier to understand the internal workings of the compiler. 

- **Torch.compile as a Scripting Game-Changer**: According to `@lancerts`, torch.compile offers a less illusive process for model porting compared to previous methods, deeming it a "savior" for those looking for alternatives to torch script and model conversion to C++.

**Links mentioned**:

[PyTorch 2 paper and tutorial @ ASPLOS 2024](https://pytorch.org/blog/pytorch-2-paper-tutorial/): The PyTorch team is excited to share that our paper on PyTorch 2 has been accepted for presentation at the ACM International Conference on Architectural Support for Programming Languages and Operating...

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1204834039421345872) (2 messages): 

- **torch.compile vs. torch.jit.trace**: `@lancerts` asked whether `torch.compile` is a superset to `torch.jit.trace` and if one should default to using `torch.compile`. `@marksaroufim` confirmed that **`torch.jit.trace` and `torch.jit.script` are unmaintained** and recommended `torch.compile` for fast inference with Python, and export + AOT inductor for environments without Python.
  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1204728742249766963) (6 messages): 

- **Dipping Toes in Fast Ai and Stable Diffusion**: `@einstein5744` mentioned they have begun the stable diffusion fast ai course and watched a few lectures from part 1.
- **Seeking SF Machine Learning Meetup**: `@.nike2k` inquired about any **in-person meetups or events** for studying ML together in San Francisco.
- **Interest Poll for Potential ML Meetup**: In response to `.nike2k`, `@marksaroufim` suggested setting up a poll to gauge interest for an in-person study event.
- **GTC Overlap Noticed**: `@apaz` observed that there seems to be a significant overlap with **GTC** (GPU Technology Conference).
- **Looking for Reference Book Solutions**: `@bruno_58591` asked about where to find solutions to a reference book, which `@drisspg` answered with a [Discord link](https://discord.com/channels/1189498204333543425/1194427148656721970/1195844532680523776).
  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1204725334180757514) (5 messages): 

- **The Birth of JAX**: `@jeremyhoward` recounted that **JAX** was initially a small project by a trio at Google, but it quickly became the go-to over **TensorFlow 2** due to dissatisfaction with TF2 and lackluster support for TF1.
- **Performance Edge with JAX**: `@cropinky` mentioned JAX's **XLA JIT** optimizations that purportedly enhanced training speeds on Google Cloud TPUs by **30%** over TensorFlow, as per a two-year-old comparison. A [video](https://youtu.be/fuAyUQcVzTY?si=Sg1jK5eQUJrEkt9P) was shared to elaborate on the details.
- **Equivalency in GPU Performance**: Continuing the conversation, `@cropinky` also shared anecdotal evidence from a university colleague's experiments showing similar performances among JAX, TensorFlow, and Torch on NVIDIA GPUs, though they suspected JAX might excel on TPUs.
- **TensorFlow Troubles Inspire JAX?**: They humorously surmised that the frustration with debugging TensorFlow could have been a catalyst for Google Brain's team to develop JAX.
- **Global Variable Grumbles**: `@nshepperd` criticized TensorFlow 1.x for its usability issues, particularly the handling of model parameters as global variables, which contributed to their switch to PyTorch and, eventually, a return to JAX.

**Links mentioned**:

[Day 1 Talks: JAX, Flax &amp; Transformers 🤗](https://youtu.be/fuAyUQcVzTY?si=Sg1jK5eQUJrEkt9P): Day 1 Talks: JAX, Flax &amp; Transformers 🤗0:00:00 Skye Wanderman-Milne (Google Brain): Intro to JAX on Cloud TPUs0:42:49 Marc van Zee (Google Brain): Introduct...

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1204715819515387904) (13 messages🔥): 

- **Troubleshooting LangChain ChatPromptTemplate**: User `@ebinbenben` reported experiencing a `ValueError` when using `from langchain.prompts.chat import ChatPromptTemplate`, specifically with error message "*ValueError: expected ':' after conversion specifier*". No resolutions or further discussion followed the question.

- **Vercel AI SDK Suggested as an Option**: In a brief response to an unspecified matter, user `@r_i_c_o_` suggested that the **vercel ai sdk** might be a good option, but no context or follow-up was provided.

- **Seeking Relevant Question Validator**: `@tepes_dracula` inquired about the existence of a classifier to validate the relevance of query and context before inputting it into a Large Language Model (LLM), preferably one trained on *truthfulqa* and *FEVER* datasets. There was no response addressing this query within the provided messages.

- **Best Document Formats for LangChain**: `@b0otable` asked the community if there had been any experimentation determining the most effective document format for LangChain's document loader, mentioning their current workflow as markdown loading plus recursive chunking. The question remained open without any user engagement.

- **Async PlaywrightContextManager Issue**: `@vladmir_bc` encountered runtime warnings while using **async PlaywrightContextManager** with *langchain*, and error during deployment on a server with *uvicorn*. In response, `@mintier` suggested wrapping the whole setup in a **docker container** to solve the problem.
  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1204855212095836171) (6 messages): 

- **LangChain Streaming Documentation Updated**: `@veryboldbagel` shared that LangChain documentation has been updated with new sections on **custom streaming with events** and **streaming in LLM apps**. The update details use of special tools like `where_cat_is_hiding` and `get_items` with agents, and distinguishes between `stream`/`astream` and `astream_events`. Check out the [updated docs](https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events).

- **Advice Sought for Custom Parameters in LangChain**: `@magick93` is looking to modify an example from the `templates/extraction-openai-functions` to pass a URL parameter using the `WebBaseLoader`. They reached out for advice on adding custom parameters to server-side in LangChain applications.
  
- **Link to Helpful LangChain Webinar Provided**: To further illustrate their aim, `@magick93` referenced a [LangChain webinar on YouTube](https://www.youtube.com/watch?v=o7C9ld6Ln-M), where Harrison Chase explains how to use a URL variable in the client and have the server's document loader process it (`at the 31min mark`).

**Links mentioned**:

- [LangServe and LangChain Templates Webinar](https://www.youtube.com/watch?v=o7C9ld6Ln-M): no description found
- [Streaming | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/agents/how_to/streaming#custom-streaming-with-events,): Streaming is an important UX consideration for LLM apps, and agents are
- [Streaming | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/streaming): streaming-with-langchain}

  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1204802089641971823) (1 messages): 

- **LangChain Prompt Inquiry**: `@aegean_thunder` is seeking clarification on **LangChain** prompts based on the [quickstart guide](https://python.langchain.com/docs/modules/model_io/prompts/quick_start). They question why the system message needs to import `SystemMessage` from `langchain_core`, which is said to provide *protocol level packages*.
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1204699384160460800) (4 messages): 

- **LangChain Expression Simplified**: `@kartheekyakkala` shared their new [blog post on LangChain Expression Language (LCEL)](https://medium.com/@yakkalakartheek/langchain-expression-language-lcel-8d092b0179b8), a **declarative method** for developing models that make coding easier by using a pipe-like syntax. The blog explains LCEL in the context of the **LangChain framework**, designed to make **LLMs context-aware**.
- **Local Chatbot Toolkit and LLM Resources Updated**: `@discossi` has updated instructions for a **lightweight raq chatbot** and beginner resources for LLM. Check out [llama-cpp-chat-memory](https://github.com/ossirytk/llama-cpp-chat-memory) and [llm_resources](https://github.com/ossirytk/llm_resources) for more information.
- **Demystifying Generative AI**: `@mehulgupta7991`, a data scientist and author, promoted their book ***LangChain in your Pocket: Beginner's Guide to Building Generative AI Applications using LLMs*** available on [Amazon](https://amzn.eu/d/g3UCuEw). The guide aims to help beginners with generative AI applications.

**Links mentioned**:

- [LangChain Expression Language (LCEL)](https://medium.com/@yakkalakartheek/langchain-expression-language-lcel-8d092b0179b8): LangChain Expression Language (LCEL) is a declarative way of developing models using LangChain framework. LCEL simplifies the coding…
- [no title found](https://amzn.eu/d/g3UCuEw): no description found
- [GitHub - ossirytk/llama-cpp-chat-memory: Local character AI chatbot with chroma vector store memory and some scripts to process documents for Chroma](https://github.com/ossirytk/llama-cpp-chat-memory): Local character AI chatbot with chroma vector store memory and some scripts to process documents for Chroma - GitHub - ossirytk/llama-cpp-chat-memory: Local character AI chatbot with chroma vector ...
- [GitHub - ossirytk/llm_resources: Information and resources on everything related about running large language models locally and their development](https://github.com/ossirytk/llm_resources): Information and resources on everything related about running large language models locally and their development - GitHub - ossirytk/llm_resources: Information and resources on everything related ...

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1204874871700455474) (7 messages): 

- **Book Code Snafu Sorted with Assistance Offer**: User `@.pandamaui` reached out with an issue in the first code example from Mehulgupta's book, finding that the default model had been deprecated. `@mehulgupta7991` acknowledged the rapid development of LangChain and OpenAI's API changes, offering support and noting that even the O'Reilly team is grappling with such issues.
- **Direct Support Line Opened by Author**: In response to `@.pandamaui`'s struggles with the book's code due to API updates, `@mehulgupta7991` offered personal assistance via email at datasciencepocket@gmail.com.
- **Feedback Fuels Future Fixes**: `@mehulgupta7991` expressed gratitude to `@.pandamaui` for flagging the issue and committed to including a fix in the next edition of the book.
  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1205060655623835688) (3 messages): 

- **ChatGPT's Theme Dilemma**: User `@joshcho_` expressed frustration with ChatGPT's theme options, describing them as **"bright white or depressing black"**. No specific solutions or preferences for theme adjustments were discussed.
  

---


### LLM Perf Enthusiasts AI ▷ #[finetuning](https://discord.com/channels/1168579740391710851/1168582249738944532/1205034985283653672) (1 messages): 

- **Fine-tuning Strategy for Conversational Outcomes**: User `@naut1973` inquired about the best approach to fine-tune a model using a conversation dataset with binary outcomes. They questioned whether **outcome scoring should be distributed across each message** or if it would be more effective to consider the conversation as a whole.
  

---


### LLM Perf Enthusiasts AI ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/1204800983264268338) (3 messages): 

- **Innovative Service Utilizing AI**: `@res6969` has created a service that **classifies sections** of a report using VGT, **extracts figures**, **describes them with GPT-4V**, and then embeds the text to create a **searchable database**.
- **Sticker Shock at Operational Costs**: This creator, `@res6969`, was surprised by the **high operational cost estimate**, coming in at approximately **$20,000** to run their service on their dataset.
- **Laughing at the Costs**: `@res6969` reacted to the steep price estimate with a humorous emoji, **<:lmfao:556127422823530507>**, indicating either amusement or disbelief at the situation.
  

---


### LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/1204775739120156732) (7 messages): 

- **Skepticism About Benefits of New Library**: `@nosa_.` expressed critique of what they perceive as a hyped presentation of a new library, skeptical of its novelty and challenging the claimed 20x performance increase as "laughable".
- **Cost Concerns Over New Innovations**: `@res6969` agreed with the skepticism regarding the utility of new innovations, especially with rising costs and when considering dense or long inputs.
- **Costly Speed Tricks**: `@nosa_.` humorously suggested that new methods labeled as performance enhancements could actually lead to a significant increase in OpenAI bills, though admitting some use cases when self-hosting.
- **Steep Price Tag for GPT-4**: `@ajamjoom` highlighted the substantial cost of using GPT-4, citing a figure of "$30K/month" for the service.
  

---


### LLM Perf Enthusiasts AI ▷ #[cost](https://discord.com/channels/1168579740391710851/1169026016887459961/1204892384605773836) (2 messages): 

- **FastChat/VLLM Hosting Inquiry**: User `@natureplayer` inquired about experiences hosting **FastChat/VLLM**. `@jmak` responded with a quick "nope not yet."
  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1204880866313240577) (7 messages): 

- **Discovering the Cost-Efficiency of OpenAI's Models**: `@dbreunig` remarks on the affordability of OpenAI's `text-embedding-3-small`, indicating it is so cost-effective that it hardly makes sense to use anything else for certain features.
- **Exploring OpenAI's Pricing Strategy**: `@simonw` comments on the potential competitive strategy behind OpenAI's pricing, suggesting it feels designed to limit competition.
- **Agreement on OpenAI's Competitive Pricing**: `@dbreunig` concurs with `@simonw`'s thoughts on OpenAI's aggressive pricing structure.
- **Challenges in Building Competing Platforms**: `@simonw` mentions the difficulty competitors face in establishing platforms due to OpenAI's aggressive pricing.
- **Differentiation Requires Innovation Beyond Pricing**: `@dbreunig` notes that to stand out from OpenAI's offerings, competitors would need to offer products that are specifically tuned or tweaked beyond just UI/UX improvements.

**Links mentioned**:

[Drew Breunig (@dbreunig@note.computer)](https://note.computer/@dbreunig/111891995849758288): Attached: 1 video  Emoji suggest is now working for StepList. Some quiet AI UX, it works by generating an embedding for your new list&#39;s title and comparing it to an approved database of emoji embe...

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1204707223432069130) (3 messages): 

- **Memory Requirements for Models and LoRA**: User `devnull0` stated that sufficient memory is required to accommodate both the model and **LoRA** weights.
- **Fitting Larger Models on Limited Resources**: `johannhartmann` shared a tip that **Llama_factory** now supports *unsloth* for **Mistral**. When combined with *qlora*, it enables the wiedervereinigung-7b model to fit within certain memory constraints.
- **Mistral Language and Dataset Support Expanded**: `johannhartmann` also mentioned that Llama_factory has integrated support for **9 German SFT** and one **DPO datasets**.
  
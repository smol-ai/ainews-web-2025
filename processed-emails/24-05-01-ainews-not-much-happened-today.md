---
id: 97bbb126-3d3f-4d51-bfc9-052241177490
title: Not much happened today
date: '2024-05-02T00:47:12.556282Z'
original_slug: ainews-to-be-named-2666
description: >-
  AI news from April 30 to May 1, 2024, covers updates on AI models, frameworks,
  agents, assistants, ethics, research, and image generation, including new
  model capabilities, company activities, and legal issues.
companies:
  - anthropic
  - openai
  - perplexity
  - google
  - amazon
  - apple
  - deepmind
models:
  - command-r-35b
  - llama-3-8b
  - gpt2-chat
  - gpt-4-turbo
topics:
  - large-language-models
  - ai-agents
  - ai-assistants
  - ai-ethics
  - ai-governance
  - ai-research
  - image-generation
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/30/2024-5/1/2024. We checked 7 subreddits and [**373** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**418** channels, and **5796** messages) for you. Estimated reading time saved (at 200wpm): **615 minutes**.

Anthropic continues its pattern being (merely) 4 months behind OpenAI in [releasing a team plan](https://twitter.com/AnthropicAI/status/1785685692988940509?utm_source=ainews&utm_medium=email) and iOS app in an otherwise relatively quiet day in AI. Perplexity is teasing a private Pages feature with a signup form you can access via Discord:

 ![image.png](https://assets.buttondown.email/images/a0e5bbe1-834d-47a4-9875-959a11ef56e1.png?w=960&fit=max)  

---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**LLM Models and Frameworks**

- **Command-R 35B model excels at creative writing**: In /r/LocalLLaMA, the Command-R 35B model [outperforms larger models like Goliath-120 and Miqu-120 in a creativity benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1cgv10e/commandr_35b_is_incredible_for_creative_writing/). Proper prompting is key to unlocking its potential.
- **Llama-3 8B model context window extension**: The Llama-3 8B model [can use a 1 million token context window](https://www.reddit.com/r/LocalLLaMA/comments/1cgzu2a/llama3_8b_256k_context_exl2_quants/). [Extending the context from 8K to 80K tokens improves performance](https://arxiv.org/abs/2404.19553) on long-context understanding tasks, using only 3.5K GPT-4 generated training samples on a single 8xA800 GPU machine in 8 hours.
- **TensorRT-LLM outperforms llama.cpp in speed**: According to benchmarks on consumer laptops and desktops, [TensorRT-LLM is 30-70% faster than llama.cpp](https://jan.ai/post/benchmarking-nvidia-tensorrt-llm) on the same hardware.
- **Benchmark suggests GPT2-Chat has better reasoning than GPT 4-Turbo**: In /r/LocalLLaMA, a new benchmark with 80 one-shot tasks indicates that [**GPT2-Chat may have better reasoning capabilities than GPT 4-Turbo**](https://www.reddit.com/r/LocalLLaMA/comments/1cgp7gi/lmsys_org_constantly_compares_new_gpt2_and_claude/), despite being slightly less intelligent overall. However, some users express skepticism about the results.

**AI Agents and Robotics**

- **Self-learning Llama-3 voice agent demo**: A [demo of a self-learning Llama-3 voice agent with function calling and automatic RAG](https://www.reddit.com/r/LocalLLaMA/comments/1cgtmuy/selflearning_llama3_voice_agent_with_function/), running locally on Jetson Orin.
- **Self-Learning Large Action Model (LAM) demo**: An open-source [demo of a Self-Learning Large Action Model (LAM)](https://www.reddit.com/r/LocalLLaMA/comments/1cgtmuy/selflearning_llama3_voice_agent_with_function/) that requires no user training.

**AI Assistants**

- **Amazon CodeWhisperer renamed to Q Developer**: [Amazon CodeWhisperer has been renamed to Q Developer](https://www.aboutamazon.com/news/aws/amazon-q-generative-ai-assistant-aws), expanding its functions as a generative AI assistant for developers.
- **Apple to unveil AI-enabled Safari browser**: [Apple plans to unveil an AI-enabled Safari browser](https://appleinsider.com/articles/24/04/30/apple-to-unveil-ai-enabled-safari-browser-alongside-new-operating-systems) with an on-device LLM in iOS 18 and macOS 15.

**AI Ethics and Governance**

- **AI lobbying frenzy in Washington dominated by Big Tech**: [Big Tech companies are dominating an AI lobbying frenzy in Washington](https://time.com/6972134/ai-lobbying-tech-policy-surge/) as they aim to influence AI policy.
- **Major U.S. newspapers sue OpenAI and Microsoft for copyright infringement**: [Major U.S. newspapers have filed a lawsuit against OpenAI and Microsoft](https://www.axios.com/2024/04/30/microsoft-openai-lawsuit-copyright-newspapers-alden-global) for alleged copyright infringement.

**AI Research**

- **DeepMind's AlphaZero becomes greatest chess player in 9 hours**: Starting from scratch, [DeepMind's AlphaZero became the greatest chess player in just 9 hours](https://twitter.com/tsarnick/status/1785050900647862683).
- **DeepMind's Naturalized Execution Tuning (NExT) improves LLM code reasoning**: [DeepMind's NExT improves LLM code reasoning capabilities](https://www.marktechpost.com/2024/04/26/deepmind-researchers-propose-naturalized-execution-tuning-next-a-self-training-machine-learning-method-that-drastically-improves-the-llms-ability-to-reason-about-code-execution/?amp) by having models inspect execution traces and provide rationales, improving fix rates by 14-26%.

**Stable Diffusion and Image Generation**

- **Stable Diffusion used for diverse applications**: In /r/StableDiffusion, Stable Diffusion is being used for [generating realistic selfies, clothing options, and more](https://www.reddit.com/r/StableDiffusion/comments/1ch5k0m/using_sd_for_other_things_than_nsfw_content/), beyond just NSFW content.
- **ConsistentID project generates high-quality portraits**: The ConsistentID project [generates realistic portraits with identity fidelity and diversity](https://www.reddit.com/r/StableDiffusion/comments/1cgsw94/consistentid_better_than_ipadapter/), potentially surpassing Ipadapter.
- **HiDiffusion for SDXL generates high-quality images**: In /r/StableDiffusion, [HiDiffusion for SDXL generates high-quality images](https://www.reddit.com/r/StableDiffusion/comments/1cgntxz/hidiffusion_for_sdxl_is_something/) but requires a cfg of 20 for coherence.

---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Claude iOS App Launch and New Features by Anthropic**

- **Claude iOS app launch**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1785701418546180326) announced the release of the Claude iOS app, bringing their AI to mobile devices. The app is now available on the App Store.
- **New Team plan**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1785685692988940509) introduced a Team plan for Claude with increased usage, user management, billing, and a 200K context window for complex tasks.
- **Upcoming collaboration features**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1785685697275552210) teased future features like citations from reliable sources for claim verification and integrations with data repositories, while maintaining security and safety.

**AI Experts Share Insights**

- **Demis Hassabis on AI accelerating science**: [@demishassabis](https://twitter.com/demishassabis/status/1785646721252336084) spoke at @TEDTalks about how AI will speed up scientific discovery and help tackle major challenges like cancer and climate change.
- **Yann LeCun critiques current LLMs**: [@ylecun](https://twitter.com/ylecun/status/1785290144561373351) argued that knowledge accumulation in LLMs is not a substitute for true understanding, outlining behaviors that show a lack of basic logic, common sense, and inability to acknowledge mistakes.

**Personal Experiences and Reflections**

- **Anthropic employee shares favorite Claude posts**: [@alexalbert__](https://twitter.com/alexalbert__/status/1785369914204938326), an Anthropic employee, shared their top 10 humorous Claude posts and memes from the company Slack over the past two months.
- **Dealing with hand disability and career change**: [@jxnlco](https://twitter.com/jxnlco/status/1785661195149615347) shared his experience losing the ability to code and work due to a hand disability in 2020, and why he is now consulting rather than working at a fast-paced startup.
- **Leaving Scale AI with insights on ML progress**: [@russelljkaplan](https://twitter.com/russelljkaplan/status/1785483317397356648) announced his departure from @scale_AI after nearly 4 years, reflecting on the company's growth and his unique perspective on the future of ML. He plans to share more thoughts on ML progress and his next steps.

**AI Research and Updates**

- **Lmsys.org offers community access to unreleased models**: [@lmsysorg](https://twitter.com/lmsysorg/status/1785394860754866234) clarified they work with model developers to provide community access to unreleased models for preview testing, aiming to bring more models as they scale and partner with open-source and commercial providers.
- **2020 paper on RLHF+PPO for instruction following**: [@rasbt](https://twitter.com/rasbt/status/1785671664920920296) highlighted a 2020 paper by Stiennon et al. that used RLHF+PPO to finetune LLMs for instruction following, two years before InstructGPT.
- **Meta presents multi-token prediction for faster LLMs**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1785486711646040440) and [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1785666587879444645) shared a Meta paper on using multi-token prediction to train LMs more efficiently, with up to 3x faster inference while maintaining or improving downstream performance.

**Other Topics**

- **Machine learning book recommendations**: [@svpino](https://twitter.com/svpino/status/1785664640913211439) shared his top 3 ML books covering the ML workflow, algorithms, and deep learning tools like Keras, PyTorch, and Scikit-Learn.
- **Critique of Ilya Sutskever's arguments**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1785614874472485096) questioned Sutskever's claim that predictive objectives will succeed at creating a perfect oracle.
- **Memes and humor**: [@mervenoyann](https://twitter.com/mervenoyann/status/1785688139119353952) and [@BorisMPower](https://twitter.com/BorisMPower/status/1785555611943616651) shared humorous images and memes.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Large Language Model (LLM) Advancements and Benchmarks**

- **[LLaMA 3](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B)** is gaining traction, with [Nous Research's Hermes 2 Pro on LLaMA 3 8B](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF) outperforming the original on benchmarks like AGIEval and GPT4All Suite. Discussions around **quantizing** LLMs, with a [5.5 bits per weight limit](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) before significant quality loss. Efforts to extend **context lengths** beyond typical limits, like [1M tokens for LLaMA 3](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k), though practical limits average 100-200k.

- **Iterative methods** like [Meta's Iterative Reasoning Preference Optimization](https://arxiv.org/abs/2404.19733) boosted accuracy on GSM8K and ARC-Challenge for LLaMA-2-70B-Chat. [Kolmogorov-Arnold Networks (KANs)](https://arxiv.org/abs/2404.19756) proposed as more accurate and interpretable alternatives to MLPs.

- The [LLaMA vs GPT-4 performance comparison](https://scandeval.com/german-nlg/) on ScandEval's German NLG tasks sparked interest, with LLaMA 3 outperforming GPT-4.

**2. Optimizations and Techniques for Efficient LLM Inference**

- Significant interest in **efficient inference** methods like [effort/bucketMul](http://kolinko.github.io/effort/) for vector-matrix approximation, [Ring Attention](https://arxiv.org/abs/2310.01889) discussed at the LLM Paper Club, and [CUDA optimizations](https://github.com/karpathy/llm.c) in llm.c like Flash Attention and CUDA Graphs.

- Debates on using **binary vector representations** for embeddings inspired by biological plausibility and [CLIP](https://arxiv.org/abs/2103.00020), [Dino](https://arxiv.org/abs/2104.14294), and the RWKV LLM method.

- Techniques to improve **transformer lens** interpretability like the [tuned lens method](https://arxiv.org/abs/2303.08112) and exploring the [distributional simplicity bias](https://arxiv.org/abs/2402.04362) in neural scaling laws.

**3. Open-Source AI Tools, Libraries, and Frameworks**

- **[LlamaIndex](https://python.langchain.com/docs/use_cases/graph/constructing/)** gaining traction for document knowledge graphing, with the new **[LlamaIndex.TS v0.3](https://t.co/mBIrD9uh8c)** improving type safety and agent support. Discussions on using MongoDB Atlas as a vector store.

- Widespread adoption of **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** for open-source LLM fine-tuning, with new features like LLaMA-3 prompt strategies and [integration with dstack](https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md) for orchestration.

- Interest in **[llama.cpp](https://github.com/ggerganov/llama.cpp)** optimizations, with the [Flash Attention merge](https://github.com/ggerganov/llama.cpp/pull/5021) and efforts to support LLaMA 3 tokenization. **[LM Studio](https://discord.com/channels/1110598183144399058/1234988891153629205/)** anticipating the 0.2.22 release with llama.cpp updates.

- **[Tinygrad](https://github.com/tinygrad/tinygrad)** developments like renaming `Scalar` to `ConstType`, exploring const support variables, and [symbolic shape handling](https://github.com/tinygrad/tinygrad/pull/4362) by geohot.

**4. Multimodal and Retrieval-Augmented AI Capabilities**

- Releases of **multimodal models** like [Snowflake Arctic 480B](https://openrouter.ai/models/snowflake/snowflake-arctic-instruct) for coding and [FireLLaVA 13B](https://openrouter.ai/models/fireworks/firellava-13b) by Fireworks, an open-source LLaVA model trained on instruction data.

- Explorations into **Retrieval-Augmented Generation (RAG)** using [LangChain](https://python.langchain.com/docs/integrations/chat/fireworks/) with Mistral Large and LlamaIndex, with tutorials on [building advanced RAG assistants](https://youtu.be/ol2QMp64lgo) and [complexity-adaptive RAG strategies](https://www.youtube.com/watch?v=QnXdlqEUW80).

- Releases of **multimodal AI assistants** like [Neuralgameworks](https://neuralgameworks.com) for Unreal Engine and the AI product [Rabbit R1](https://www.youtube.com/watch?v=ddTV12hErTc&ab_channel=MarquesBrownlee), sparking interest in [integrating with OpenInterpreter](https://discord.com/channels/1146610656779440188/1194880263122075688/1234781691109703732).

- Advances in **medical AI** like the [cardiac ultrasound study with OpenCLIP](https://doi.org/10.1038/s41591-024-02959-y) and Google's [Med-Gemini multimodal models](https://youtu.be/xohuoN2WBZs?si=Ku6cztykld6dZLN9) for healthcare.


---



# PART 1: High level Discord summaries




## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA C++ Optimizing Insights**: Developers shared that the [CUDA C++ Core Libraries](https://twitter.com/marksaroufim/status/1785462414852714954) best practices revealed performance improvements, but a Google Drive link intended for slides was found to be empty. They further discussed accurate CUDA kernel profiling techniques, with a preference for NVIDIA tools like `nsight compute` and `nsight system` over `cudaEventRecord` due to less overhead and more robust profiling.

- **Triton Tackles Block Size and Debugging**: In the Triton domain, engineers clarified that **Triton's max block size** is not limited by hardware constraints like CUDA, and directed debugging tactics to utilize the [Triton debugging lecture](https://www.youtube.com/watch?v=DdTsX6DQk24). The channel also noted the usage of `triton-nightly` to benefit from the recent interpreter bug fixes.

- **Sparsity Algorithm Sparks Benchmarks & Learning**: AI enthusiasts discussed an algorithm that leverages activation sparsity with a batch size of 1, and the algorithm's creator engaged, promising to share new benchmarks and insights about the speed/quality trade-offs compared to quantization methods.

- **Strides Align and Kernels Optimize in CUDA**: Concerns and strategies over tensor stride alignment and kernel optimizations, like *matmul_backward_bias,* dominated discussion in `#llmdotc`. Advances in performance using strategy *x128 packing*, experimenting with CUDA Graphs, cuDNN Flash Attention optimization, and the introduction of FP32 for master weights were debated, demonstrating a drive towards more efficient CUDA programming.

- **AMD's ROCm and Torch Nightly Discussions**: Users focusing on AMD's ROCm platform exchanged torch Nightly preferences over Torch 2.3, questioned the absence of the latest **version 2.0 of flash attention** in AMD's fork, and shared the addition of a backward pass for AMD Flash Attention, leading to informative exchanges and a [tutorial resource on AMD HIP](https://www.youtube.com/playlist?list=PLB1fSi1mbw6IKbZSPz9a2r2DbnHWnLbF-).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**GPU Efficiency Sparks Interest**: The A4000 16GB GPU is lauded for its efficiency in training, with its cost-effectiveness earning praise when compared to the A100. The B200 is touted for its potential, being forecasted to be 25x more efficient than the current H100 at a comparable cost.

**Techniques in Question**: Debate over employing LoRA versus QLoRA revealed that QLoRA may offer a 75% VRAM usage reduction at the possible expense of 1-2% in model accuracy. The 80-10-10 split for training data was suggested to ensure model robustness, while language model fine-tuning is progressing, evidencing its application in Turkish translation.

**Innovations in Model Training**: Users reported quantization issues with `llama.cpp`, leading to GitHub issues such as [#3759](https://github.com/ollama/ollama/issues/3759) and [#4180](https://github.com/vllm-project/vllm/issues/4180). Workflows for fine-tuning and training were a point of clarification, with strategies for checkpointing and inference providers like Jan and GPT4All being put forward, available at repositories like [janhq/jan](https://github.com/janhq/jan).

**AI Development Roadmapping Proposed**: Advocates for a straightforward AI project roadmap emphasized its importance, while the potential of smaller models for enhanced conversational skills is under exploration. Additionally, the concept of retrieval augmentation is gaining traction, with references to implementations such as [FlagEmbedding's GitHub repository](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora).

**Size and Performance**: A noteworthy mention was that the Phi3 Mini 4k outperforms the larger 128k version in open LLM rankings, prompting a reevaluation of the efficacy of model sizes. There's an inclination toward models like Phi3 Mini 4k for their efficiency over larger counterparts.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Flashy Performance Optimizations**: *Flash Attention* integration into `llama.cpp` enhances memory efficiency by moving from an O(N^2) to O(N) complexity, eliciting community enthusiasm with the merged PR available at [Flash ATTENTION support merged into llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5021).
  
- **Versatility Issues with Model Constraints**: Lively discussions reveal models face compatibility obstacles when used beyond designed limits, such as *Llama 3* not playing well with old builds and erroring out with contexts larger than 250,000 tokens, despite one's attempt at a 1M token window with 36GB VRAM.

- **Necessity for Ample Hardware**: Threads agree that using LLMs effectively requires considerable system resources, with models like Everything 7b q4 becoming sluggish on a mere 8 GB RAM, and an updated llama.cpp tokenizer error hinting at hefty RAM needs.

- **ROCm Build Roadblocks**: AMD users engage over **ROCm and OpenCL** integration, with reports of misread **VRAM capacity** on a 7900xtx, despite using an RX 6600 previously, and recommendations to opt for a 7900XTX over a 7900 GRE for ensured LM Studio compatibility.

- **Chasing the Latest Model and Software Releases**: The pending release of LM Studio 0.2.22 has generated buzz, aimed to fix tokenizer concerns and enhance model performance, while the beta release of `llama.cpp` is also suggested to address issues flagged by the community.

For updates on technical advancements and fixes, the community is advised to check the respective GitHub repositories and release pages for the latest commits and build updates.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Breaking the OOD Barrier**: A solution for positional out-of-distribution (OOD) issues has been proposed to help large language models generalize to longer contexts, which can be found in a [recently published paper](https://arxiv.org/pdf/2401.01325). An implementation example employing `--grp-attn-n` and `--grp-attn-w` parameters is available in the [`llama.cpp` repository](https://github.com/ggerganov/llama.cpp/tree/master/examples/server).

- **Llama-3 Leaps Ahead**: Nous Research has launched **Hermes 2 Pro on Llama-3 8B**, touting Function Calling and Structured Output enhancements and outperforming Llama-3 8B Instruct on prominent benchmarks. A quantized version targeting efficiency without compromising advancements is also available on [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF).

- **LLM Performance and Practicability**: Discussions indicated that quantization to **5.5 bits per weight** is a threshold before performance loss in large language models becomes significant. The new **Hermes 2 Pro Llama 3** has unlearned specific tasks while gaining new ones like function calling, with the community exploring the optimization of long context lengths and integration of advanced tokenization mechanisms.

- **Data Sets and Tools for AI Innovation**: A new **Wikipedia RAG dataset** has been released, paralleling a study on leveraging LLMs for synthesizing multilingual training data, available [here](https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7). Moreover, discussion included the integration of Pydantic in the rework of Cynde and the introduction of Logfire, a platform praised for its simplified code observability, detailed [here](https://pydantic.dev/logfire).

- **Virtual Simulation Advances**: The community has seen the release of business and music **industry simulators**, CompSimulator and Snow Singer Simulator, aimed at providing immersive AI-driven experiences. In addition, talks from AGI House SF have spurred plans for community meetups, with a noted feature that LLAMA 3 bots on HF Chat yield consistent responses for identical messages.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 Anticipation Fizzles Without Launch**: Skepticism clouds the **Stable Diffusion 3 (SD3)** release, as expectations for an April or May launch were met with disappointment; there's concern **Stability AI** may face criticism for statements about SD3 being free and open-source.
- **Local Interface Lineup Evaluation**: AI enthusiasts are comparing Stable Diffusion local interfaces like **ComfyUI**, **AUTO11111**, **Focus**, and **Forge**, with recommendations hinging on user-friendliness and specific hardware requisites such as NVIDIA or AMD GPU compatibility.
- **AI-Assisted Prompt Engineering**: There's an ongoing debate about the best tools for effective image description prompts with mentions of **ChatGPT**, **Gemini**, **Claude 3**, and **idefics2**; these are potentially valuable for refining prompts to enhance image generation results.
- **AI Service and Privacy Tools**: Discussions indicate trends in investing in AI services like **Gemini** and **Claude 3**, coupled with the strategic use of **VPN** technologies, including **DNS over HTTPS**, for bypassing regional restrictions or maintaining user anonymity.
- **Extension Talk for Automatic1111 Fans**: Queries surfaced regarding the capability of embedding labels within images using **Automatic1111 extension** and whether there are features analogous to **clip skip and stylizer** in custom interfaces such as **ComfyUI**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Chat Control Gets Upgraded**: OpenAI has rolled out **updated data controls** for ChatGPT Free and Plus, allowing users to see chat history while opting out of data use for training. They also introduced **Temporary Chat** for one-off sessions with no chat history retention.

- **GPT-2's Resurgence in Chatbots**: Members are exploring the **gpt2-chatbot** with mixed feedback; it excels in certain scenarios but is also noted to fail occasionally. There's intrigue regarding its capability for infinite generations, though access issues have been reported.

- **Dissecting AI Emotional Intelligence**: In-depth discussions on AI's potential to develop emotion have drawn parallels to human development. Emphasis lies on whether empathetic understanding or akin emotional responses are either achievable or desirable in AI systems.

- **DALL-E's Free Tier Functionality Debates**: Users have been discussing the offerings of OpenAI's services like DALL-E for free users, balancing between business sustainability and expanding user functionalities.

- **Harnessing Positive Prompting Results**: AI Engineers are exploring efficient prompt engineering, with a focus on positive prompting and meta-prompting to achieve more effective interactions with AI models, suggesting strategies like *"instead of 'x', use 'y'"* to refine output quality.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Pages Feature Prepares for Beta Liftoff**: **Perplexity AI** announces an upcoming feature named **Pages** designed for crafting shareable, in-depth explorations of topics; early access to the beta version is available for interested users.

**API Citations The Missing Piece**: Engineers express concerns about accessing citations through API requests when using **Perplexity-online models**, alongside discussions of discrepancy between Pro UI and API model results; the **[API documentation](https://docs.perplexity.ai/docs/model-cards)** is clarified to be the go-to resource for model details.

**Limitations and Glitches in Spotlight**: Members discussed the **50 daily usage limit** for Opus, the presence of glitches in Pro Search and referencing tools, and slow responses from AI models, with technical advice offered around possible email filtering from service providers for login issues.

**Discovery Through Shared Content**: Users actively shared insights and links on diverse topics, including **Microsoft Research Asia**, the **Vimeo API**, and **Tesla's self-driving** tech; plus, a shared [newsletter](https://www.lennysnewsletter.com/p/how-perplexity-builds-product) provided a window into product development insights.

**Claude 3 Policy and Model Utilization Clarified**: Queries about the usage policy of **Claude 3** led to discussions on whether Perplexity's or **Anthropic's** policies are applicable, while the usage of online models in the Pro UI was explained to be either finetuned or employing a search engine-style vector database for responses.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Speeding Up Inference with Effort/BucketMul**: A new algorithm, **effort/bucketMul**, was introduced, designed to significantly accelerate vector-matrix approximation and large language model (LLM) inference, promising real-time computational load adjustments and compatibility with models like Mistral. Further details can be found [here](http://kolinko.github.io/effort/).

- **Binary Beats Hypersphere for Embedding Efficiency**: Discourse over embedding strategies yielded insights into the efficiency of binary vector representations for embeddings, backed by biological plausibility and computational frugality, with a connection made to the RWKV LLM, which might benefit from faster learning applying these principles. To delve deeper, read about the [RWKV LLM](https://github.com/BlinkDL/SmallInitEmb) and seminal embedding works such as [CLIP](https://arxiv.org/abs/2103.00020) and [Dino](https://arxiv.org/abs/2104.14294).

- **Demystifying the Black Box and Improving Benchmarks**: Conversations around the opacity of LLMs noted the gap between their complexity and our comprehension, with a focus on improving fairness in benchmark comparisons by avoiding training LLMs on benchmark test sets. Refer to the discussion on bias in [benchmark datasets](http://arxiv.org/abs/2404.18824).

- **KANs Take the Lead Over MLPs**: Emerging research introduced **Kolmogorov-Arnold Networks (KANs)**, outshining Multi-Layer Perceptrons (MLPs) in terms of accuracy and interpretability with efficient scaling laws. The pivotal paper on KANs is found [here](http://arxiv.org/abs/2404.19756).

- **Striving for Transparent LLM Computations**: A member's exposition theorized about the computational models within sequence-prediction models, discussing how tied embeddings might influence interpretability and pondering experimental methods to validate their hypotheses. Essential reads include [Deriving a Model of Computation for Next-Token Prediction](https://docs.google.com/document/d/11w3of15CbfOlWrvQpTjxaJt-UvtOckzr0WQUfTrTnsw/edit?usp=sharing) and papers on the [tuned lens method](https://arxiv.org/abs/2303.08112) and the concept of [distributional simplicity bias](https://arxiv.org/abs/2402.04362).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Cash Prizes for CVPR Participation**: HuggingFace announced [CVPR competitions](https://huggingface.co/spaces/BVRA/SnakeCLEF2024) with a total prize pool of over **$120,000**, including competitions such as SnakeCLEF, FungiCLEF, and PlantCLEF slated for June 17-21, 2024.

- **Transformers and Gradio Level Up**: A significant update to the *Transformers* library introduces new models, with [Phi-3 now operable in the browser](https://github.com/huggingface/transformers/releases/tag/v4.40.0). Gradio also released [v4.28.0](https://www.gradio.app/changelog), featuring custom components, and parallel updates arrived for the Datasets library, reaching v2.19.0 with *Polars* compatibility.

- **AI Tools You Should Experiment With**: New AI tools and methods are shared, including a Medium post on ["5 Interesting AI Tools Everyone Should Try"](https://medium.com/illumination/genai-adventures-5-interesting-ai-tools-everyone-should-try-44ae8f8115af) and a discussion on accelerating diffusion models in PyTorch 2, as suggested in Hugging Face's [documentation](https://huggingface.co/docs/diffusers/tutorials/fast_diffusion).

- **Med-Gemini: AI for Medicine Introduced**: A [YouTube video](https://youtu.be/xohuoN2WBZs?si=Ku6cztykld6dZLN9) provides insights into Google's **Med-Gemini**, a multimodal GenAI model designed for medical applications, promoting understanding of such models' scope and potential.

- **Job Opportunities and Community Insights**: A software engineer with extensive experience inquired about opportunities at Hugging Face and was directed to the available [positions](https://apply.workable.com/huggingface/?lng=en). Meanwhile, community exchanges included discussions on intent recognition issues with the Rasa chatbot framework, learning curves between PyTorch and TensorFlow, and creating instruction datasets for LLM finetuning.

- **Gradio’s Status Checkpoint**: Gradio faced issues with their Share Server impacting usage on Colab; they provided a [status page](https://status.gradio.app/) to keep track of progress on the fix.

- **Innovations in the AI Community**: Contributions from community members feature projects like a [PnPR-GCN technique](https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24) for leak-free link prediction and **HDR imaging challenges**, articulating solutions and engaging with the wider discourse on AI advancements.

- **Lean Learning Approaches**: Within reading groups, attention has been turned to topics such as graph neural networks with [arXiv:2404.14928](https://arxiv.org/abs/2404.14928) and the application of negotiation as a metric for evaluating LLM alignment touched upon in *NegotiationArena* shared at [arXiv:2402.05863](https://arxiv.org/abs/2402.05863).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RTX 4080: Enough For Small Language Models?**: Engineers discussed whether a gaming card like the **RTX 4080** is suitable for running and fine-tuning smaller language models, noting the importance of VRAM but suggesting limitations in fine-tuning models larger than 7B with small batch sizes.

- **Local AI Processing Values Security**: The conversation highlighted the advantage of a local PC for dealing with **sensitive data** and robust computing tasks over cloud solutions like Google Colab, which may raise privacy concerns.

- **Introducing Word Loom for AI Language Management**: A new open specification called **Word Loom** was introduced, targeting the efficient management and exchange of language for AI, aiming for a clear separation of code from natural language and better composability, with detailed information found on [GitHub](https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e).

- **AI Financial Genius Works Without Human Help**: A groundbreaking financial assistant now boasts the ability to **calculate percentage evolution, CAGR, and P/E ratios** over unstructured financial reports autonomously, as highlighted in a recent [tweet](https://t.co/6cTNxUBJcr).

- **LlamaIndex Scores New Technical Capabilities**: The latest release, **LlamaIndex.TS version 0.3**, brings significant improvements including agent support for various platforms, Web Streams enhancements, and a more resilient type system as announced in a [tweet](https://t.co/mBIrD9uh8c).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo Marches On**: The Mojo developer community celebrated the **first anniversary** of Mojo’s launch, praising the addition of traits, references, and lifetimes which significantly enhanced the standard library. Concerning enhancements, it was suggested to improve Mojo by allowing **negative numbers**, and implementing a fallback for scalar processing, inspired by linkage to articles within the issues.

**Performance Power-ups**: Innovative optimization of string allocations and conversions in Mojo cut processing time from 18.5 to 12.5 seconds for 100M records, with the latest effort reducing it further to 3.8 seconds utilizing multi-core processing techniques. A call was made to form **Team-Mojo** for the *One Billion Row Challenge*, seeing it as an opportunity for showcase and community collaboration.

**Syntax and Semantics Synergy**: Discussions on syntax and semantics highlighted the importance of Mojo’s syntax alignment for users and how `inout` in Mojo bears similarity to pass-by-reference in C++, but with its nuances. Questions about the `__source_location()` function led to a conversation pondering the inclusion of `function_name` in its output and the replacement of these features in the nightly branch.

**Exploring Concurrency Considerations**: The conversation speculated on Mojo's concurrency model potential, theorizing it might mirror an actor model more than the golang-style, with a spotlight on avoiding heavy runtime inclusion. The Mojo compiler, with an LLVM backbone, has a [dedicated YouTube video](https://youtu.be/SEwTjZvy8vw) explaining its underpinnings.

**Tweet Teasers Lead to Speculation**: Modular spurred curiosity with a series of unspecified tweets, teasing intriguing developments without revealing the specifics, piquing interest for details beyond the announcements.




---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Exploring Axolotl's Model Support**: In a discussion within the **#axolotl-phorm-bot** channel, it was clarified that Axolotl supports GaLore but not phi-3 format. Community advice recommended checking out the [Hugging Face documentation](https://github.com/huggingface/transformers/tree/main/docs/source/en/trainer.md) for details on enabling GaLore. Meanwhile, an [untested PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547) was highlighted as a resource for those looking to add command-r model to Axolotl.

**Strategies for Effective Chat-Tokenization**: Members in **#general** channel debated the impact of the Beginning of Sentence (BOS) token in tokenizer behavior, and the importance of specifying it correctly in different scenarios. Also, a [study on generalist foundation models](https://arxiv.org/abs/2311.16452) prompted discussions on the effectiveness of complex prompting strategies and the challenges in rendering academic theory practical. 

**Best Practices for Fine-Tuning New Models**: The **#general-help** channel was abuzz with communities engaged in fine-tuning processes, where certain criteria such as using smaller models like an 8b model for beginners were recommended. Practical tips for dataset conversion for ShareGPT loader, and inquiries regarding fsdp compatibility with lora were discussed.

**Tutorial Collaboration Strikes a Chord**: In the **#community-showcase**, a tutorial illustrating the combination of axolotl and dstack, an open-source container orchestrator, was shared and well-received, emphasizing ease-of-use and flexibility. Contributors are directed to [GitHub for detailed usage](https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md).

**Compute Resources for Collaboration**: An offer in the **#axolotl-dev** channel extended compute resources to other members for the purpose of helping with triage and troubleshooting, which could be particularly useful for those involved in bug fixes and enhancements.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI Enters TOS Grey Zone**: A discussion emerged around users employing **AI products without consenting to terms of service**, highlighting a gray area in user agreement enforcement and prompting debate on legal implications for both users and providers.

**Leaderboard Integrity Challenged**: There's a push for a more transparent **AI model leaderboard**, emphasizing the need for openness and verifiability, while members expressed skepticism over **LMSYS's Chatbot Arena**, raising concerns of lack of objectivity and opaque data practices. The notion of incorporating **only open source models** and filtering by **open weights** was put forth as a criterion for improved leaderboards.

**Eager for Efficiency**: Engineering conversations revolved around a multitude of optimization strategies, from considering **GANs for superior model reconstruction** to discussions about **Natten's cuda implementation**, and the development of projects like [magvit2](https://github.com/lucidrains/magvit2-pytorch).

**Breaking New Ground in AI and Medicine**: The community took note of a published study on **cardiac ultrasound utilizing OpenCLIP** that was recently featured in [Nature Medicine](https://doi.org/10.1038/s41591-024-02959-y), despite some existing issues with the study.

**Revolutionizing Networks and Fact-Checking**: Enthusiasm was evident for the innovative **Kolmogorov-Arnold Networks (KANs)**, poised to outdo MLPs in accuracy and interpretability ([the paper on KAN](https://arxiv.org/abs/2404.19756)), and the introduction of **VisualFactChecker**, a training-free pipeline designed to bolster visual content captioning fidelity ([the paper on VFC](https://arxiv.org/abs/2404.19752)).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Decentralizing AI's Compute Power**: Prime Intellect has plunged into the exploration of **decentralized AI training methodologies**, aiming to rival the expansive **GPU clusters** employed by larger corporations. Their platform is geared towards leveraging globally distributed compute resources, as detailed in their extensive [blog post](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training).

**Starcoder Rises**: Hugging Face has launched a new **Large Language Model called StarCoder2-15B-Instruct-v0.1**, focusing primarily on **code generation**. They've made the model and pipeline open-source, inviting the community to engage, as outlined on their [announcement page](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1).

**Simulating AI Societies on Consumer Tech**: An experimental setup involving 300 AI agents called AI Town is reported to operate seamlessly on a MacBook M1 Max. The intriguing [tweet](https://x.com/cocktailpeanut/status/1785702250599371088?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) reveals the capabilities and potential of AI simulations on consumer-level hardware.

**LLM Paper Club: Ring in the Discussion**: The **LLM Paper Club's** upcoming event features a collaborative discussion with the StrongCompute team on the **Ring Attention** paper. Engineers interested in the latest research findings can join via this [Zoom link](https://lu.ma/oz8e9z3r).

**Video Meet for the Tech-Elite**: A **Zoom meeting video call** has been set up for a more visual interactive discussion, likely concerning ongoing work or a paper club event. The community members can join using the provided [Zoom Meeting link](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Respect Is Tech's Best Friend**: A community reminder underscored the imperative of **respect and constructive interaction**; as the group expands, it is vital that everyone feel welcomed and valued for a collaborative future.

**Open Interpreter Becomes Browser-Savvy**: The **Open Interpreter** tool was confirmed to possess capabilities for **web browsing and data scraping tasks** without the need for traditional browser control, fostering direct web interactions through the AI.

**Hitting the Right Note with DIY Speaker Amp**: To boost the audio output from speakers, one solution recommended was an **external amplifier**, highlighting one potential amplifier on [Amazon](https://www.amazon.com/dp/B01DKAI51M), though real-world application awaits confirmation upon testing.

**R1's AI Unboxing Sparks Integration Talks**: An **MKBHD YouTube review** on the AI product, Rabbit R1, [watch here](https://www.youtube.com/watch?v=ddTV12hErTc&ab_channel=MarquesBrownlee), ignited discussions on its potential integration with **OpenInterpreter**, with engineers eager to push the envelope of interconnected AI systems.

**Tunnel Vision for Successful OI Connection**: Engineers traded know-how on establishing a stable connection with an OpenInterpreter server, including the method for setting up new domains with **ngrok** and modifying the **tunnel.py** file, aiming to iron out connection wrinkles—more details at [ngrok domains page](https://dashboard.ngrok.com/cloud-edge/domains).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **New AI Models Hit the Ice**: **Snowflake Arctic 480B** and **FireLLaVA 13B** have been released with **Snowflake Arctic 480B** boasting a hybrid transformer architecture optimized for coding, available at [Snowflake Arctic 480B](https://openrouter.ai/models/snowflake/snowflake-arctic-instruct), and **FireLLaVA 13B**, a multimodal model from Fireworks, accessible at [FireLLaVA 13B](https://openrouter.ai/models/fireworks/firellava-13b). Pricing and developer specifications are updated to reflect their enhanced capabilities.

- **OpenRouter Gets Smarter with Efficient Load Handling**: New **load balancing** features aim to distribute provider workloads more effectively, complemented by real-time monitoring tools for latency and provider performance at [Activity page](https://openrouter.ai/activity), improving overall system robustness.

- **Streamlined Resources for Developers**: **OpenRouter's documentation** now includes updates, enabling more efficient use of Image and multimodal requests, tailored tool calls, and function calling; details can be found at [Image Requests](https://openrouter.ai/docs#images-_-multimodal-requests) and [Tool Calls](https://openrouter.ai/docs#tool-calls).

- **Cost Reduction in AI Services**: **OpenRouter** has reduced prices significantly: a major 40% cut for Mythomax Extended services, alongside a modest 4% saving on Mixtral 8x7b Instruct, reinforcing the platform's commitment to affordable AI services.

- **AI Writes with a Swedish Flair**: **Skribler**, a tool designed to assist Swedish authors with various facets of writing by incorporating different AI models, is on the rise with a user base already willing to pay for its services - check it out at [skribler.se](https://skribler.se).



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**Crisp Visuals Spark Interest**: [Hexagen World](https://www.hexagen.world/) surprised members with **high-quality diffusion model outputs**, suggesting promising avenues for interactive AI game development.

**Retro Games Reimagined with AI**: The Guild discussed reviving retro games like **Farmville** using Generative AI, with WebSim as a potential platform for these nostalgic reboots.

**Spy Games Meet Generative Towns**: An intriguing concept for a 1950s-themed **AI town with a communist spy** character was proposed, generating interest in creating an immersive **cat-and-mouse** game within WebSim.

**Join the AI-Animated Conversation**: Those curious about AI-driven animation received an invitation to a specialized Discord group via a [community link](https://discord.gg/deforum), offering room for collaborative discussions and projects in interactive AI.

**Dev Discussions Highlight Compatibility Issues**: AI devs tackled local setup processes, noting particular issues with Windows systems and the importance of using the correct **Node version** (`nvm use 19`). Some even considered switching to Linux, especially since games like Stellaris are supported, as evidenced by information found on [WineHQ](https://appdb.winehq.org/objectManager.php?sClass=application&iId=17537).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Command R Impresses**: The Cohere community has expressed appreciation for the **CommandR/R+ models**, highlighting their polished performance which seemingly surpasses other large language models for an enterprise-level experience.

**LLM Grammar Secrets Exposed**: A discussion on **LLMs (Large Language Models)** and their ability to generate grammatically correct sentences revealed insights into word and sentence embeddings, and the significance of the self-attention mechanism, with a [resource provided for in-depth understanding](https://docs.cohere.com/docs/the-attention-mechanism).

**AI Legal Eagle Takes Flight**: A webinar on constructing an AI legal assistant using **Cohere's RAG** saw the community engaged, with a link to the recording made available on [YouTube](https://www.youtube.com/watch?v=KfqJsqIFeRY&ab_channel=Cohere).

**Azure Meets OAuth**: Instructions for setting up OAuth with connectors on Azure using the Cohere toolkit were clarified, highlighting the ability for azure integration while keeping data internal as detailed on their [GitHub page](https://github.com/cohere-ai/cohere-toolkit/?tab=readme-ov-file#how-to-add-a-connector-to-the-toolkit).

**Multilingual Mastery in the Making**: The implementation and potential of **multilingual support** in Command-R is under active evaluation by the community, with particular attention to languages like Norwegian and the desire for enhanced benchmarks.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**PDF Table Extraction Proves Tricky**: Engineers shared challenges with **table extraction from PDFs** using *unstructure* library, noting subpar outcomes particularly with multi-page tables. No solution was provided, indicating an area ripe for development or an opportunity for tool recommendation.

**LangChain and Llama 3 Join Forces**: There was a conversation about integrating **Llama 3** with **LangChain**, directing users to utilize [Fireworks](https://python.langchain.com/docs/integrations/chat/fireworks/) and corresponding API keys. Additionally, a mention about the re-inclusion of **Google Drive libraries** in a project was noted, highlighting the cyclical nature of tech dependencies.

**Launch, Updates, and Spec Introductions**: Noteworthy developments include the launch of [QuickVid](https://quickvid.vercel.app/) for summarizing YouTube content, the update of LangChain chatbot to **0.1.17**, and the introduction of **Word Loom** as a potential standard for AI language management, feedback solicited at their [GitHub Gist](https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e). Queries about the usefulness of a detailed performance report comparing various **LLMs for content creation** were also raised.

**Knowledge Graph Aspirations and AI Sales Agents**: Members shared insights into tools for converting documents into **knowledge graphs** and the development of **AI-powered Sales Agents**. For the former, layout parsers and Azure Doc AI were proposed, alongside exploring LangChain's documented graph construction methods. The latter involved SalesGPT logic and a call for partnerships.

**RAG Innovations and Language-Focused Tutorials**: Engineers discussed a variety of RAG applications, including an **Advanced RAG assistant for the French-speaking community**, local training of Llama3, and an **Adaptive RAG technique** that responds based on query complexity. Related instructional videos were shared: [French RAG Assistant](https://youtu.be/ol2QMp64lgo), [Local Agentic RAG w/ llama3](https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P), and [LangGraph + Adaptive Rag + LLama3 Python Project](https://www.youtube.com/watch?v=QnXdlqEUW80).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Mozilla AI is Hiring, Wave at Lm-buddy**: Mozilla AI is currently expanding its team, with opportunities posted on their official Discord channel, and has also released **Lm-buddy**, a new open-source tool aimed at improving model evaluation efficiency.

**LLaMA3:8b on M1 MacBook Air Confirmed for Testing**: After users encountered issues with **LLaMA3:8b** running on M1 MacBook Air, the response indicated that testing on M1 will become a priority once other support issues are resolved.

**Whispering to Llamafile**: Proposals were made to integrate **whisper.cpp models** into llamafile for enhanced inference, despite the challenges in adding microphone and speaker functionalities.

**Performance Debate Clarified**: An article by Justine Tunney suggesting **np.matmul** performs at 29 gflops was contested, leading to a clarification that this was specific to an Intel computer on Ubuntu and that actual performance may vary.

**Simultaneous Llamafiles and Path Customization Explained**: Discussions in the guild confirmed that running multiple llamafiles with different models is possible, with operating systems managing the resources. Users also learned that customization using the `--server --path PUBLIC_PATH` option is limited to replacing .html and .js files in the zip file.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Tinygrad Undergoes Tensor Transformations**: The [tinygrad project](https://github.com/tinygrad/tinygrad) implemented major updates with a [commit renaming `Scalar` to `ConstType`](https://github.com/tinygrad/tinygrad/commit/77589bc7a5430ee470621e43fb1817259d3ce0f5), contributing to standardization efforts in the codebase. Discussions spotlighted the potential to optimize constant handling in operations by introducing const support variables and the significance of const Variables for operations linked to symbolic dimensions.

**Graph Visualization Interest Piques for Backward Passes**: The conversation included curiosity about visualizing graph diagrams for backward operations with a focus on issue **#3572**. There are hints at using dot files and setting `GRAPH=1` for visual aid in understanding these operations.

**Symbolic Dimensions Step into the Spotlight**: Georgehotz shared insights on working with symbolic shapes and introduced [a pull request with a skipped test for symbolic arange](https://github.com/tinygrad/tinygrad/pull/4362). This indicates an ongoing effort to enhance tinygrad's capabilities with symbolic dimensions.

**JIT Crafting and Mean Calculations**: A dialogue on improving tinygrad's Just-In-Time (JIT) compilation with symbolic variables led to the suggestion that a robust test would involve calculating the mean of variable-length 2D tensors. Such enhancements could refine the efficiency and performance of the JIT compiler.

**CUDA Challenges on Nvidia Xavier**: Technical discussions touched upon challenges faced while running EfficientNet examples on Nvidia Xavier, emphasizing the need to ensure `CUDA=1` for proper script execution. Members also deliberated on whether Rednode's representation in tinygrad could be complicating symbolic compiler logic.





---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Claude Joins the AI Chat App Scene**: Anthropic has released its Claude app, stirring up curiosity among members about its **performance compared to OpenAI's solutions**. While no detailed comparisons were provided, one user has downloaded the app and reported a smooth initial experience, with particular kudos to Anthropic's branding.

- **Elevating Performance Through Feedback**: After receiving pointed feedback, a member significantly improved their work quality, resulting in commendation from their peers. Specifics of the work improvement were not given, but the reactive boost in productivity was notable.

- **AI Leaderboards Under Scrutiny**: An article suggests that AI leaderboards might be outdated, highlighting that **the most accurate system for code generation**, as per HumanEval benchmarks, is LDB. However, its **reliance on expensive calls to models like GPT-4** casts a shadow on its efficiency and cost-effectiveness.

- **ML Collective Attendance**: An individual confirmed sparse attendance at **ML Collective** meetings, indicating ongoing participation but no specific outcomes or details from the meetings were discussed.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Spam Alert Across the Guild**: Multiple channels within the Discord guild were infiltrated by inappropriate content that advertised adult material involving potentially underage subjects, alongside Discord invite links purportedly offering leaked content.
- **Urgent Need for Moderation**: These messages violate community guidelines, hint at illegal activities, and disregard the purpose of professional discourse expected in technical discussions.
- **Unwelcome Interruptions**: The spam disrupted numerous channels, ranging from those dedicated to AI discussion to collaboration and general chat, necessitating attention by moderators.
- **Content Warning for Engineers**: Engineers must be cautious as the spam contains potential security risks, such as phishing attempts, that could compromise professional and personal data.
- **Call to Action**: Immediate actions are advised to remove the content, ban the posters, and enhance security measures to prevent future incidents.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Prompt Engineering Propels LLaMA-3**: The **LLaMA-3 instruct prompt strategies** have been updated, leading to performance improvements, with the associated changes detailed in a GitHub [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553).
- **Easing Dataset Woes**: Proper usage of `eot_id` has resolved challenges related to dataset entry formatting, proving to be more efficient than manual `</s>` tagging.
- **Meta Harnesses Iterative Reasoning**: New "Iterative Reasoning Preference Optimization" techniques have elevated **LLaMA-2-70B-Chat**'s accuracy, as demonstrated by improved scores on GSM8K and ARC-Challenge benchmarks; the paper can be read [here](https://arxiv.org/abs/2404.19733).
- **Axolotl Fine-Tuning Success**: An user experienced success fine-tuning **LLaMA-3 8b with Axolotl**, noting enhanced model outputs.
- **Cranking Up Coding Jams**: A motivational anime track, "NEVER GIVE UP YOUR WAAAAAAAAAAAAY," was shared to possibly fuel late-night coding sessions, complete with a [YouTube link](https://youtu.be/tYzMYcUty6s?si=t2utqcq36PHbk9da) and a note of Patreon support for the creators.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**LLaMA beats GPT-4 in Language Showdown**: Results from [scandeval.com](https://scandeval.com/german-nlg/) indicate that **LLaMA 3** outperforms **GPT-4** in the ScandEval benchmark for German natural language tasks, sparking discussions about new AI model capabilities.

**Accelerated Local Loads Trump Sluggish Cloud**: An engineer reported that a program *loads in 3 seconds* on a local machine, pointing towards issues other than storage affecting slower load times when running jobs elsewhere.

**Qdora Expands LLaMA's Middleway**: Exciting progress in Large Language Model (LLM) expansion has emerged with the mention of **qdora**, a solution fostering the growth of models like LLaMA; the process is outlined in an [Answer.ai blog post](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html).

**Avoiding Forgetfulness in AI Training**: The guild discussed methods to prevent catastrophic forgetting during post-pretraining, referencing an [Arxiv paper](https://arxiv.org/abs/2401.02415) on enhancing Transformer blocks that helps LLMs retain old skills while learning new ones.

**Fusing AI Past and Present**: Guild engagement highlighted the prospect of "Non-forgetful Learning" in LLMs, where expansion techniques are crucial for merging traditional AI skills with newer, more advanced capabilities.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Designing User-Centric Data Retrieval**: A member proposed a frontend feature for **Datasette** allowing users to select country-specific data from a dropdown with the goal of improving user experience in data fetching.
- **Debating URL vs. UI Customization**: Two user experience strategies emerged: dynamically updating the **URL** to display relevant data upon selection, and developing a customizable interface with "buildable" queries based on user input.



---

# PART 2: Detailed by-Channel summaries and links



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1235244530857541726)** (4 messages): 

- **CUDA Best Practices Shared**: The channel shared a Twitter link about [best practices for CUDA C++ Core Libraries](https://twitter.com/marksaroufim/status/1785462414852714954) and also provided slides via a Google Drive [link](https://drive.google.com/drive/folders/1T-t0d_u0Xu8w_-1E5kAwmXNfF72x-HTA), but the folder was noted to have no files.

- **Prompt Action on Spam**: A user flagged the attention of moderators with a mention (@&1189538650011217942), followed by swift action from another member confirming the removal of a spammy post.

- **Understanding PyTorch's autograd.grad**: A member posed a question about using `torch.autograd.grad` to obtain the diagonal of the Hessian matrix of a function output with respect to parameters, with two consecutive gradient computations.

**Link mentioned**: <a href="https://drive.google.com/drive/folders/1T-t0d_u0Xu8w_-1E5kAwmXNfF72x-HTA">CCCL - Google Drive</a>: no description found

  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1234899266837938176)** (13 messages🔥): 

- **Triton's Block Size Puzzle**: A member inquired about the maximum block size in **Triton**, thinking it would match CUDA's limit. In response, it was explained that Triton's block size is not fundamentally tied to the hardware and could theoretically be very large, with no direct relation to the number of threads launched per block.

- **Triton Debugging Techniques Probed**: An individual sought advice on the best practices for debugging **Triton** kernels, finding challenges with `TRITON_INTERPRET=1` and `device_print`. Another member encouraged reviewing a [Triton debugging lecture](https://www.youtube.com/watch?v=DdTsX6DQk24) for insights, as it might provide useful strategies.

- **Need for Triton Interpreter Bug Fixes**: Following up on debugging issues, a user mentioned that the `TRITON_INTERPRET=1` setting was causing abnormal program behavior. It was suggested to install **Triton** from source or use `triton-nightly` to benefit from recent interpreter bug fixes.

- **Curiosity for Triton's Release Schedule**: A member asked about the expected release date for the next version of **Triton**, as they are currently using version 2.3. The response was that there is no solid plan yet for the upcoming release.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=DdTsX6DQk24">Lecture 14: Practitioners Guide to Triton</a>: https://github.com/cuda-mode/lectures/tree/main/lecture%2014

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1234762936782819398)** (14 messages🔥): 

- **Exploring CUTLASS vs CuBLAS**: A member highlighted the performance of [CUTLASS](https://github.com/NVIDIA/cutlass), which outperformed CuBLAS with a matrix multiplication benchmark (8192 x 8192 x 8192), achieving 288 Teraflops compared to CuBLAS's 258 Teraflops. When integrated into Python, however, CUTLASS's performance advantage disappeared, matching CuBLAS at 257 Teraflops.
- **Kernel Timing Conundrums in CUDA**: A discussion emerged around accurately profiling time durations within CUDA kernels, as utilizing `cudaEventRecord` showed unstable timings, particularly in shared memory versions of matrix multiply kernels with varying tile sizes.
- **NVIDIA Tools for Accurate Profiling**: It was suggested to use NVIDIA's `nsight compute` or `nsight system` for more robust profiling, as they are built to be more accurate and might incur less overhead compared to custom profiling with `cudaEventRecord`.
- **Understanding Profiling Overheads**: A member queried about inconsistencies between `cudaEventRecord` timings and `ncu's` Duration field, with the concern that `ncu's` report might include profiling overhead. The response clarified that `ncu` runs warm-up kernels which could account for additional reported time, but ultimately suggest more accuracy.
- **Nsight Systems vs. NCU Utility**: Clarification was given that both `nsys` and `ncu` can be used for profiling CUDA kernels, with each providing different utilities and interfaces for analyzing and understanding kernel performance.

**Link mentioned**: <a href="https://www.thonking.ai/p/strangely-matrix-multiplications">Strangely, Matrix Multiplications on GPUs Run Faster When Given &quot;Predictable&quot; Data! [short]</a>: Great minds discuss flops per watt.

  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1234894006043938926)** (5 messages): 

- **Sparsity and Quality Trade-offs**: The conversation revolves around an algorithm potentially leveraging *batch size=1 activation sparsity*, which might preserve compute and quality. However, there is concern that this approach could face limitations similar to activation sparsity when dealing with batched computations over one.
  
- **Effort Creator Chimes In**: The creator of the algorithm mentioned joined the chat and is open to discussing their findings about its performance.

- **Benchmark Revelations**: The creator provided an update that new benchmarks show *effort/bucketMul* performs worse in terms of speed/quality ratio when compared to quantization, with an article to come detailing these findings.

- **Quality Keeps Up with Pruning**: Despite speed/quality concerns, the creator claimed that in terms of quality degradation, their method appears superior to simply pruning the smallest weights, promising to post supporting charts.

- **Direct Comparison Shared**: A direct comparison was shared highlighting the difference between removing the lowest weights from a matrix and skipping the least important calculations, noting the creator's ongoing process of learning about sparsity.
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1235025584296558632)** (2 messages): 

- **Confusion Over Sequence Length in Puzzle 9**: User expressed confusion regarding Puzzle 9's terminologies, specifically about the parameters **T** and **N0**. The formula for *z_i* was also a subject of confusion as the user was unsure how it should be interpreted based on the provided information.
- **Possible Description Conflict Noted**: Another member acknowledged potential conflicting information in the problem description of Puzzle 9 and shared their assumption that **N0** equals **T** for solving purposes.
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1234762607689203752)** (809 messages🔥🔥🔥): 

- **CUDA Optimization Discussions Intensify**: The CUDA MODE Discord community continues to scrutinize and optimize various kernel operations. Members are experimenting with aligning tensor strides and optimizing the *matmul_backward_bias* kernel, with an eye on future enhancements using *x128* packing for increased performance. Several iterations have been proposed for the *gradient clipping* and *adam optimizer* kernels, considering their impacts on computational efficiency and memory usage.
- **CUDA Graphs and cuDNN Flash Attention in Action**: The channel's contributors have successfully integrated optional support for cuDNN flash attention, seeing meaningful speed improvements, although the exact performance gain over current bespoke kernels remains under evaluation. CUDA graphs have been mentioned as a mechanism for optimization, though more detail is needed to understand their current state of use within the community's codebase.
- **Comparing PyTorch and llm.c Performance**: Recent discussions and benchmarks suggest that *llm.c* is closely matching or surpassing the performance of PyTorch for the GPT-2 model training, even outperforming PyTorch 2.3.0 by up to 32%. However, with PyTorch nightly builds showing considerable performance improvements due to recently merged PRs, *llm.c* is now slightly behind with a ~4% slower token processing rate.
- **Debates Over Memory Efficiency and Operation Fusing**: There's ongoing discussion about the relative merits of fusing operations like GELU with matmul kernels to save memory. Though such fusion is tricky and could potentially hurt performance, some suggest fusing into the epilogue of the preceding matmul or re-computing in backward passes could be a memory-efficient compromise. Concepts like prologue vs. epilogue fusion and matmul's need for input/output tiles in forward/backward passes are central to these debates.
- **Potential for Master Weights in FP32**: A suggestion was made to keep master weights in FP32 by default to provide a more stable and reliable implementation. This modification would imply certain changes to the optimizer update function and memory allocation scheme, with lazy initialization during the update stage as a possible approach.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb">no title found</a>: no description found</li><li><a href="https://pytorch.org/tutorials/recipes/compiling_optimizer.html">(beta) Compiling the optimizer with torch.compile &mdash; PyTorch Tutorials 2.3.0+cu121 documentation</a>: no description found</li><li><a href="https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/discard_memory.html">cuda::discard_memory</a>: CUDA C++ Core Libraries</li><li><a href="https://research.colfax-intl.com/adding-fp8-to-flashattention/">Delivering 1 PFLOP/s of Performance with FP8 FlashAttention-2</a>: We recently released an update to our FlashAttention-2 forward pass implementation on NVIDIA Hopper&#x2122; architecture that incorporates a number of new optimizations and improvements, including …</li><li><a href="https://dev-discuss.pytorch.org/t/performance-comparison-between-torch-compile-and-apex-optimizers/2023">Performance Comparison between Torch.Compile and APEX optimizers</a>: TL;DR  Compiled Adam outperformed SOTA hand-optimized APEX optimizers on all benchmarks; 62.99% on Torchbench, 53.18% on HuggingFace, 142.75% on TIMM and 88.13% on BlueBerries Compiled AdamW performed...</li><li><a href="https://stackoverflow.com/questions/28932864/which-compute-capability-is-supported-by-which-cuda-versions">Which Compute Capability is supported by which CUDA versions?</a>: What are compute capabilities supported by each of:&#xA;&#xA;CUDA 5.5?&#xA;CUDA 6.0?&#xA;CUDA 6.5?</li><li><a href="https://docs.nvidia.com/deeplearning/cudnn/latest/release-notes.html">Release Notes &mdash; NVIDIA cuDNN v9.1.0 documentation</a>: no description found</li><li><a href="https://github.com/karpa">karpa - Overview</a>: karpa has 13 repositories available. Follow their code on GitHub.</li><li><a href="https://godbolt.org/z/hME5EqYrr">Compiler Explorer - CUDA C++ (NVCC 12.2.1)</a>: #include &amp;lt;cuda/barrier&amp;gt; #include &amp;lt;cuda/std/utility&amp;gt; // cuda::std::move #include &amp;lt;cooperative_groups.h&amp;gt; #include &amp;lt;cooperative_groups/reduce.h&amp;gt;  t...</li><li><a href="https://github.com/karpathy/llm.c/pull/313/files">fixed potential error and generalized gelu forward by ngc92 · Pull Request #313 · karpathy/llm.c</a>: This adds a helper function for safe casting from size_t to ints (may want to have that in utils.h too). that macro is then used to convert the size_t valued  block_size * x128::size back to a regu...</li><li><a href="https://github.com/karpathy/llm.c/issues/246">WikiText 103 evaluation · Issue #246 · karpathy/llm.c</a>: I&#39;ve seen some repos use WikiText-103 as the dataset they use to eval GPT-like models, e.g.: https://github.com/tysam-code/hlb-gpt/tree/main Add prepro script to download and preprocess and tokeni...</li><li><a href="https://github.com/karpathy/llm.c/pull/325">mixed precision utilities for dev/cuda by ngc92 · Pull Request #325 · karpathy/llm.c</a>: cherry-picked from #315</li><li><a href="https://github.com/karpathy/llm.c/pull/314">Add llm.cpp fork to README by jrhemstad · Pull Request #314 · karpathy/llm.c</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/326">option to keep weights as fp32 by ngc92 · Pull Request #326 · karpathy/llm.c</a>: adds an optional second copy of the weights in fp32 precision TODO missing free</li><li><a href="https://github.com/karpathy/llm.c/pull/318">gradient accumulation preview / wip by karpathy · Pull Request #318 · karpathy/llm.c</a>: I can&#39;t seem to get this working tonight, something is off. The Python part works. i.e. we have the following. Running the default python script reproduces the old behavior before this PR: python ...</li><li><a href="https://github.com/karpathy/llm.c/pull/323">feature/cudnn for flash-attention by karpathy · Pull Request #323 · karpathy/llm.c</a>: Builds on top of PR #322 Additional small fixes to merge cudnn support, and with it flash attention</li><li><a href="https://github.com/karpathy/llm.c/pull/273#issuecomment-2087188223">Add NSight Compute ranges, use CUDA events for timings by PeterZhizhin · Pull Request #273 · karpathy/llm.c</a>: CUDA events allow for more accurate timings (as measured by a GPU) nvtxRangePush/nvtxRangePop Adds simple stack traces to NSight Systems:  Sample run command: nsys profile mpirun --allow-run-as-roo...</li><li><a href="https://github.com/karpathy/llm.c/pull/227/">Second matmul for fully custom attention by ngc92 · Pull Request #227 · karpathy/llm.c</a>: So far, just in the /dev  files, because for the main script we also need to touch backward. For some reason, I see  considerable speed-up in the benchmarks here, but in my attempts to use this in ...</li><li><a href="https://github.com/karpathy/llm.c/pull/303">Updated adamw to use packed data types by ChrisDryden · Pull Request #303 · karpathy/llm.c</a>: Before Runtime total average iteration time: 38.547570 ms After Runtime: total average iteration time: 37.901735 ms Kernel development file specs: Barely noticeable with the current test suite: Bef...</li><li><a href="https://github.com/NVIDIA/cudnn-frontend/issues/52#issuecomment-2015335369">What&#39;s the difference of flash attention implement between cudnn and Dao-AILab? · Issue #52 · NVIDIA/cudnn-frontend</a>: Is this link a flash attention?</li><li><a href="https://github.com/karpathy/llm.c/pull/322">cuDNN Flash Attention Forward &amp; Backwards BF16 (+35% performance) by ademeure · Pull Request #322 · karpathy/llm.c</a>: RTX 4090 with BF16 and batch size of 24:  Baseline: 232.37ms (~106K tokens/s) cuDNN: 170.77ms (~144K tokens/s) ==&gt; +35% performance! Compile time: Priceless(TM) (~2.7s to 48.7s - it&#39;s a big dep...</li><li><a href="https://github.com/karpathy/llm.c/pull/315">first draft for gradient clipping by global norm by ngc92 · Pull Request #315 · karpathy/llm.c</a>: one new kernel that calculates the overall norm of the gradient, and updates to the adam kernel. Still TODO:  clip value is hardcoded at function call site error handling for broken gradients would...</li><li><a href="https://github.com/karpathy/llm.c/pull/262">single adam kernel call handling all parameters by ngc92 · Pull Request #262 · karpathy/llm.c</a>: First attempt at a generalized Adam kernel</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L2022">llm.c/train_gpt2.cu at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L2024">llm.c/train_gpt2.cu at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/315/files#diff-49f823d54affd1961dce0e04a078a49ea7bd831326097074aa3db0ea11d0aca4R97-R102">first draft for gradient clipping by global norm by ngc92 · Pull Request #315 · karpathy/llm.c</a>: one new kernel that calculates the overall norm of the gradient, and updates to the adam kernel. Still TODO:  clip value is hardcoded at function call site error handling for broken gradients would...</li><li><a href="https://github.com/karpathy/llm.c/pull">Pull requests · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/pull/120758):">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/pytorch/pytorch/pull/121692):">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/karpathy/llm.c/pull/273?">Add NSight Compute ranges, use CUDA events for timings by PeterZhizhin · Pull Request #273 · karpathy/llm.c</a>: CUDA events allow for more accurate timings (as measured by a GPU) nvtxRangePush/nvtxRangePop Adds simple stack traces to NSight Systems:  Sample run command: nsys profile mpirun --allow-run-as-roo...</li><li><a href="https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/">Faster Parallel Reductions on Kepler | NVIDIA Technical Blog</a>: Parallel reduction is a common building block for many parallel algorithms. A presentation from 2007 by Mark Harris provided a detailed strategy for implementing parallel reductions on GPUs&#8230;</li><li><a href="https://github.com/karpathy/nanoGPT/blob/master/train.py#L307">nanoGPT/train.py at master · karpathy/nanoGPT</a>: The simplest, fastest repository for training/finetuning medium-sized GPTs. - karpathy/nanoGPT</li><li><a href="https://github.com/pytorch/pytorch/pull/120758">[inductor] comprehensive padding by shunting314 · Pull Request #120758 · pytorch/pytorch</a>: Stack from ghstack (oldest at bottom):  -&gt; #120758  This PR adds the ability to pad tensor strides during lowering. The goal is to make sure (if possible) tensors with bad shape can have aligned st...</li><li><a href="https://github.com/gevtushenko/llm.c">GitHub - gevtushenko/llm.c: LLM training in simple, raw C/CUDA</a>: LLM training in simple, raw C/CUDA. Contribute to gevtushenko/llm.c development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=WiB_3Csfj_Q">Bonus Lecture: CUDA C++ llm.cpp</a>: llm.cpp: https://github.com/gevtushenko/llm.cSlides: https://drive.google.com/drive/folders/1T-t0d_u0Xu8w_-1E5kAwmXNfF72x-HTA?usp=sharing</li><li><a href="https://drive.google.com/drive/folders/1T-t0d_u0Xu8w_-1E5kAwmXNfF72x-HTA">CCCL - Google Drive</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/pull/99975">Foreach kernel codegen in inductor by mlazos · Pull Request #99975 · pytorch/pytorch</a>: design doc Add foreach kernel codegen for a single overload of foreach add in Inductor. Coverage will expand to more ops in subsequent PRs. example cc @soumith @voznesenskym @penguinwu @anijain2305...</li><li><a href="https://github.com/karpathy/llm.c/pull/306">Packing for Gelu backwards by JaneIllario · Pull Request #306 · karpathy/llm.c</a>: Update gelu backwards kernel to do packing into 128 bits, and create gelu brackward cuda file Previous kernel: block_size   32 | time 0.1498 ms | bandwidth 503.99 GB/s block_size   64 | time 0.0760...</li><li><a href="https://github.com/karpathy/llm.c/pull/319">convert all float to floatX for layernorm_forward by JaneIllario · Pull Request #319 · karpathy/llm.c</a>: change all kernels to use floatX</li><li><a href="https://github.com/karpathy/llm.c/pull/299">Update residual_forward to use packed input by JaneIllario · Pull Request #299 · karpathy/llm.c</a>: Update residual_forward to use 128 bit packed input, with floatX Previous Kernel: block_size   32 | time 0.1498 ms | bandwidth 503.99 GB/s block_size   64 | time 0.0760 ms | bandwidth 993.32 GB/s b...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1234763147861299210)** (8 messages🔥): 

- **Issues Building with Torch 2.3**: A member mentioned difficulties in building with **Torch 2.3** and expressed a preference for using **torch nightly** instead.
- **AMD Lacks Latest Flash Attention Kernels**: A member queried why AMD's official fork has not ported **version 2.0 of flash attention**, despite there being newer flash attention kernels available.
- **Backward Pass Added to AMD Flash Attention**: In response to a question about the backward pass for AMD Flash Attention, it was confirmed that the **backward pass was indeed implemented**, with a [link to the ROCm flash-attention GitHub repo](https://github.com/ROCm/flash-attention).
- **AMD RDNA3 Support in Flash Attention**: A member asked which branch has the **RDNA3** working for the ROCm flash-attention, indicating the presence of `allowed_archs` in the code.
- **AMD HIP Tutorial Playlist Shared**: Another member found the information interesting and shared a [YouTube playlist](https://www.youtube.com/playlist?list=PLB1fSi1mbw6IKbZSPz9a2r2DbnHWnLbF-) for an **AMD HIP Tutorial**, which covers using the HIP programming language on the ROCm platform.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/playlist?list=PLB1fSi1mbw6IKbZSPz9a2r2DbnHWnLbF-">AMD HIP Tutorial</a>: In this series of videos, we will teach how to use the HIP programming language to program AMD GPUs running on the AMD ROCm platform. This set of videos is a...</li><li><a href="https://github.com/ROCm/flash-attention">GitHub - ROCm/flash-attention: Fast and memory-efficient exact attention</a>: Fast and memory-efficient exact attention. Contribute to ROCm/flash-attention development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1234773459733708810)** (572 messages🔥🔥🔥): 

- **Efficiency vs. Power Debate**: The A4000 16GB GPU was highlighted as efficient for training, costing significantly lower than an A100 per hour. The upcoming B200 is pegged as a game-changer, potentially 25x more efficient than an H100 at a similar price point. 

- **Finetuning With LoRA and QLoRA**: A discussion clarified the differences in VRAM usage and potential accuracy degradation between using LoRA (16bit) and QLoRA (4bit). QLoRA saves 75% of VRAM but might result in 1-2% accuracy loss, whereas LoRA has no accuracy degradation.

- **Training Advice Shared**: A recommended strategy was to split datasets into 80% for training, 10% for tuning hyperparameters, and keep 10% hidden for final model evaluation without further tuning to avoid contaminating training data.

- **Training Turkish Language Model**: A user is fine-tuning Llama 3 on translate tasks for Turkish with over 430k examples. The model is currently performing like a translation bot, changing the output language based on the input language.

- **ORPO Training on Unsloth**: A snippet of code was shared for training the mlabonne/orpo-dpo-mix-40k dataset using Unsloth ORPO Trainer on an RTX 4090 GPU, taking about 5 hours.

- **Unsloth Wiki Update**: Contributions regarding fine-tuning and training were added to the Unsloth [wiki](https://github.com/unslothai/unsloth/wiki), acknowledging the community input.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIk">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://arxiv.org/abs/2401.10020">Self-Rewarding Language Models</a>: We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human prefer...</li><li><a href="https://tenor.com/view/weird-minion-gif-23757545">Weird Minion GIF - Weird Minion - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cb3q0i/issue_with_with_llama_3_exl2_quant_either_ending/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://colab.research.google.com/drive/19lwcRk_ZQ_ZtX-qzFP3qZBBHZNcMD1hh?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/loading">Load</a>: no description found</li><li><a href="https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k">mlabonne/orpo-dpo-mix-40k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://gist.github.com/jedt/e45b337e9d9bd0492bf5d3c1d4706c7b">gist:e45b337e9d9bd0492bf5d3c1d4706c7b</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://huggingface.co/NousResearch">NousResearch (NousResearch)</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/M-Chimiste/unsloth_finetuning">GitHub - M-Chimiste/unsloth_finetuning</a>: Contribute to M-Chimiste/unsloth_finetuning development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/400">[FIXED] NotImplementedError: No operator found for `memory_efficient_attention_forward` with inputs · Issue #400 · unslothai/unsloth</a>: I&#39;m a beginner to try unsloth. I run the free notebook Llama 3 (8B), and then got the following error: I also encountered the following error during the first installing step: ERROR: pip&#39;s dep...</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: no description found</li><li><a href="https://github.com/huggingface/datasets/issues/6753">Type error when importing datasets on Kaggle · Issue #6753 · huggingface/datasets</a>: Describe the bug When trying to run import datasets print(datasets.__version__) It generates the following error TypeError: expected string or bytes-like object It looks like It cannot find the val...</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: Hackable and optimized Transformers building blocks, supporting a composable construction.</a>: Hackable and optimized Transformers building blocks, supporting a composable construction. - facebookresearch/xformers</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1234838859401924689)** (6 messages): 

- **Size Matters Less with AI Models**: A remark was made indicating that the **Phi3 Mini 4k** version outperforms its larger 128k counterpart on the Open LLM Leaderboard, suggesting the Mini could be the preferred choice.
- **Customized Mistral Adaptation**: It was noted that **Phi3** has been modified using Mistral technology but is configured to work specifically with their version of Phi.
- **Pi in the Sky**: A user shared their experience running **Phi-3** on an Orange Pi Zero 3, describing performance with the Q2 version of gemma 2b as "slightly fast."
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1234773564956086292)** (254 messages🔥🔥): 

- **Problems with Quantization and Conversion**: Users reported issues with quantization using `llama.cpp`, such as "*failed to quant q8 gguf messages after a large run,*" and manual attempts at GGUF conversion led to errors like "*Vocab size mismatch*." References were made to [GitHub issues #3759](https://github.com/ollama/ollama/issues/3759) and [GitHub issue #4180](https://github.com/vllm-project/vllm/issues/4180) related to these problems.
- **Questions on Few-Shot Learning and Best Practices**: One user inquired if it's better to put all few-shot examples in one user turn or across multiple turns for training. Another user *starsupernova* suggested trial and error, and in general confirmed either approach can work.
- **Checkpointing Finetuning Process for Later Resumption**: Instructions on checkpointing were shared, pointing users to the [Unsloth GitHub Wiki](https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint) for guidance on how to save progress and continue training later without consuming excessive storage.
- **Choosing Inference Providers for Fine-Tuned Models**: *theyruinedelise* recommended using Jan or GPT4All as good inference providers for a fine-tuned Llama 3 70B model with Unsloth, with a link to Jan's GitHub repo ([janhq/jan](https://github.com/janhq/jan)).
- **Requests for Workflow and Tutorial Clarification**: Multiple users sought clarification on training workflows, saving and pushing models to Hugging Face, and how to continue training from checkpoints. For example, *starsupernova* advised to save both model and tokenizer to Hugging Face and confirmed that setting `ref_model=None` is fine when using the DPO notebook.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-bnb-4bit">unsloth/llama-3-8b-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF">NousResearch/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/janhq/jan">GitHub - janhq/jan: Jan is an open source alternative to ChatGPT that runs 100% offline on your computer. Multiple engine support (llama.cpp, TensorRT-LLM)</a>: Jan is an open source alternative to ChatGPT that runs 100% offline on your computer. Multiple engine support (llama.cpp, TensorRT-LLM) - janhq/jan</li><li><a href="https://github.com/ollama/ollama/issues/3759">llama3-instruct models not stopping at stop token · Issue #3759 · ollama/ollama</a>: What is the issue? I&#39;m using llama3:70b through the OpenAI-compatible endpoint. When generating, I am getting outputs like this: Please provide the output of the above command. Let&#39;s proceed f...</li><li><a href="https://huggingface.co/datasets/wikimedia/wikipedia">wikimedia/wikipedia · Datasets at Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/issues/4180">[Usage]: Llama 3 8B Instruct Inference · Issue #4180 · vllm-project/vllm</a>: Your current environment Using the latest version of vLLM on 2 L4 GPUs. How would you like to use vllm I was trying to utilize vLLM to deploy meta-llama/Meta-Llama-3-8B-Instruct model and use OpenA...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1235093733465395251)** (18 messages🔥): 

- **Diverse Datasets Without VRAM Woes**: Members discussed whether combining multiple datasets increases VRAM usage. The consensus was that merging datasets doesn't affect VRAM, but rather increases training time.

- **Training Challenges with Vast Datasets**: One member pondered the feasibility of fine-tuning Mistral 7B with a large dataset using 16 gigs of VRAM. Despite the huge size of the dataset, members opined that while possible, it would be very time-consuming and advised focusing on high-quality synthetic data.

- **A Guide to AI Roadmaps**: A suggestion was made to create a simple roadmap for AI projects. This would ideally be a straightforward to-do list in a README.md to clarify development directions and goals.

- **Model Enhancements for Chatterboxes**: Experimentation with smaller models is underway, aiming to increase conversational abilities and accuracy. This indicates a focus on refining AI for better dialogue interactions.

- **Retrieval Augmentation in the Spotlight**: A link to a GitHub repository named FlagEmbedding was shared, which showcases work on retrieval and retrieval-augmented Long LLMs. This could be of interest to those looking to improve their models with retrieval mechanisms. [Long_LLM/longllm_qlora on GitHub](https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m">lmsys/lmsys-chat-1m · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/FlagOpen/FlagEmbedding/tree/master/Long_LLM/longllm_qlora">FlagEmbedding/Long_LLM/longllm_qlora at master · FlagOpen/FlagEmbedding</a>: Retrieval and Retrieval-augmented LLMs. Contribute to FlagOpen/FlagEmbedding development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1234775457409531926)** (204 messages🔥🔥): 

- **Flash Attention Merged into llama.cpp**: Flash Attention feature provides better memory efficiency and allows contexts to fit more easily within memory as it operates on an O(N) rather than O(N^2) complexity. Enthusiasm was expressed for the merged PR in llama.cpp, found here: [FLASH ATTENTION support merged into llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5021).
- **Experiencing Issues When Loading Models**: Users are discussing various issues related to loading models in LM Studio, with one sharing an error and another expressing concern about the system requirements in relation to VRAM and physicaI RAM.
- **Discussions on Proxy and LM Studio**: Users experiencing problems when searching for models may find issues relating to corporate networks, proxies, or the need to disable IPv6 if unable to route to Hugging Face.
- **GPU Offload Clarifications**: An important recommendation made was to turn off GPU offload when using inadequate VRAM, as 3GB of VRAM is insufficient for certain operations in LM Studio.
- **Eagerness for LM Studio Beta**: The beta release of LM Studio 0.2.22 integrating new PRs from llama.cpp was announced, enticing users to test it and provide feedback on the inferencing quality, with the additional anticipation of progress on OpenELM as seen in [this update](https://github.com/ggerganov/llama.cpp/pull/6986).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/perfecto-chefs-kiss-gif-10500688187407334920">Perfecto Chefs GIF - Perfecto Chefs Kiss - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6986">Attempt at OpenElm by joshcarp · Pull Request #6986 · ggerganov/llama.cpp</a>: Currently failing on line 821 of sgemm.cpp, still some parsing of ffn/attention head info needs to occur. Currently hard coded some stuff. Fixes: #6868 Raising this PR as a draft because I need hel...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5021">ggml : add Flash Attention by ggerganov · Pull Request #5021 · ggerganov/llama.cpp</a>: ref #3365 Setting up what&#39;s needed for Flash Attention support in ggml and llama.cpp The proposed operator performs: // new res = ggml_flash_attn(ctx, q, k, v, kq_mask, kq_scale);  // fused scale ...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1234772570188939334)** (123 messages🔥🔥): 

- **Exploring Model Limitations**: A member queried about the downsides of downloading a 1048K context model for use with only 20k tokens, noting that the updated quantization options were limited. Concerns were also raised that the new Llama 3 quant at Q8 displayed repetitive behavior in version 0.2.20 (ROCm preview).

- **Compatibilty Issues with Llama 3**:
  Participants discussed that new Llama 3 quants are not backward compatible with older builds, with instances of repeating answers. *"[These models will also work if you haven't updated to latest llama.cpp, but will still have the old broken tokenizer until you get your tool updated.](https://www.reddit.com/r/LocalLLaMA/comments/1cg3e8k/llama_3_8b_instruct_with_fixed_bpe_tokenizer/)"* was stated on Reddit, suggesting an update is necessary for optimal use.

- **Slow Performance on Modest Hardware**: Users conversed about the feasibility of running uncensored models, like Everything 7b q4, on machines with 8 GB RAM. It was indicated that the models can work but expect slow performance, with advice to close additional applications such as web browsers to free up resources.

- **Image Generation Models Availability**: Within the discussion, it was clarified that LM Studio does not currently support direct image generation. A member posted a link to a GitHub repository by AUTOMATIC1111, one of the popular free and local options for image generation, separate from LM Studio's functionality.

- **Looking for Enhanced Human-like AI Behavior**: A user sought advice on creating a more vibrant and human-like AI agent, mentioning the example from a YouTube video featuring "Neuro Sama." Tips included asking the Llama 3 to create character prompts with specific personality traits and the direction to explore more specialized channels for advanced model behaviors.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B">vonjack/Hermes-2-Pro-BakLLaVA-Mistral-7B · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/dont-know-idk-dunno-no-idea-no-clue-gif-22858277">Dont Know Idk GIF - Dont Know Idk Dunno - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.meta.ai/">Meta AI</a>: Use Meta AI assistant to get things done, create AI-generated images for free, and get answers to any of your questions. Meta AI is built on Meta&#039;s latest Llama large language model and uses Emu,...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cg3e8k/lla">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/AI-Engine/BakLLaVA1-MistralLLaVA-7B-GGUF">AI-Engine/BakLLaVA1-MistralLLaVA-7B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/AUTOMATIC1111">AUTOMATIC1111 - Overview</a>: AUTOMATIC1111 has 41 repositories available. Follow their code on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ceh5cp/gpt2chatbot_at_lmsys_chatbot_arena/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/shorts/fgG8E6bNwjo">Neuro Challenges Vedal</a>: Neuro won&#39;t stop spamming chat when Vedal challenges her.►Twitch: http://www.twitch.tv/vedal987►Twitter: https://twitter.com/Vedal987#neurosama #vtuber #vedal
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1234792683059281920)** (35 messages🔥): 

- **Model Load Error Reported**: A user encountered an error stating *"(Exit code: 0). Please check settings and try loading the model again."* with 7.15 GB of RAM available and Linux OS specifications.
- **Various System Specs on Linux**: Discussion revolves around observing an unusually high number of Linux users with limited free RAM; also, someone with 64GB+ of RAM also reported only having a few KB of memory free.
- **Hard Drive Chatter during Model Generation**: A user noted a HDD seek sound or "chattering" coming from their computer when generating tokens with a model partially offloaded to the GPU. They clarified that the system has 96GB of RAM, and the noise was specific to HDD, not coil whine or the cooling system.
- **Problem with Llama3m Model Operation**: There were queries about a Llama3 model's performance, specifically, if it was caching to an HDD rather than staying in RAM. The model of interest was mentioned with a link: [Llama-3-8B-Lexi-Uncensored-GGUF](https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF) and operated at a context size of 8k tokens.
- **LM Studio vs Ollama Debate**: Users shared their opinions about LM Studio and Ollama, leading to a debate on preferences where one user expressed a strong preference for LM Studio while another reminded the community to value both and avoid negative comparisons.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF">Orenguteng/Llama-3-8B-Lexi-Uncensored-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=rJM8rHfsgjk">Hard Drive Sounds</a>: This is a comparison of all the sounds of the HDDs in my hard drive collection. The drives are played in chronological from oldest to newest.
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1234838825386381312)** (9 messages🔥): 

- **Llama3 Loading Woes**: A member encountered an error while trying to load **llama3** with a 1M token context window, despite having 36GB VRAM and 128GB RAM. The error was attributed to the excessive size of the desired context window when the system parameters are designed for a **context size of 250,000**.
  
- **Context Window Overload**: Attempting to load a 100k token context window successfully maxed out the system's capabilities, indicating that the 1M token ambition was simply too resource-intensive.

- **Quadratic to Linear**: One contributor mentioned that the context issue used to be quadratic but, with current optimizations, it's "more like linear nowadays".

- **Configuration Misread**: A member highlighted that the **Readme** for the model indicates the requirement of "100s of gigs of ram". This comment implies a possible oversight in understanding the hardware requirements for large context windows.

- **Model Download Attempt**: The member provided a specific directory for **Llama-3-8B-Instruct-Gradient-1048k-iMat-GGUF**, which suggests an effort to download or reference a specific version of the model.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1234777495635759104)** (272 messages🔥🔥): 

<ul>
<li><strong>Groq's Tempting Token Generation</strong>: There was a discussion around Groq's ability to generate 800 tokens per second for llama 3 70B, with anticipation of an upcoming paid subscription model.</li>
<li><strong>Hardware Guidance for LLMs</strong>: A member was advised that their AMD rx 5600m 6GB VRAM with Ryzen 7 4k setup may be on the low end for running local models, suggesting they explore models listed on the app’s front page.</li>
<li><strong>Model Download Speeds</strong>: Members engaged in a talk about the download speeds of models from Hugging Face within LM Studio, with one claiming about 10MB/s and another advocating a speeds comparison between direct downloads vs LM Studio.</li>
<li><strong>The Quest for Comparative Accuracy in LLMs</strong>: A user sought LLMs that could match the accuracy of ChatGPT, discussing the recent 70b llama3 and Wizard models, with mentions of performance being new and uncharted.</li>
<li><strong>Hardware Endeavors and Puzzling Phenomenon</strong>: There were extensive discussions surrounding the optimal hardware for LLM processing, with a focus on memory speed and VRAM as limiting factors, SLI/NVLink capabilities, and an anecdote about two different models generating the same fictional city name in separate instances, prompting a mix of humor and curiosity.</li>
</ul>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.asrockrack.com/general/productdetail.asp?Model=ROMED8-2T#Specifications">no title found</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/#lightbox">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=QK8mJJJvaes">MACKLEMORE &amp; RYAN LEWIS - THRIFT SHOP FEAT. WANZ (OFFICIAL VIDEO)</a>: The Heist physical deluxe edition:http://www.macklemoremerch.comThe Heist digital deluxe on iTunes: http://itunes.apple.com/WebObjects/MZStore.woa/wa/viewAlb...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1234783013846515752)** (141 messages🔥🔥): 

- **Troubleshooting Hardware Compatibility**: A member queried about software running but facing issues with Large Language Model (LLM) acceptance on their hardware setup. Another participant advised that the hardware with i5-4570 and 16GB RAM is likely insufficient for most models, suggesting they could only run a 7b Q4 model effectively.

- **New LLama.cpp Commit Requested**: A request was posted for the latest commit of llama.cpp to fix a tokenizer problem. A response suggested that it would be made available soon.

- **Eagerly Awaiting LM Studio 0.2.22**: Dialogue surrounding **LM Studio 0.2.21** issues led to anticipation for the upcoming release of LM Studio 0.2.22. Discussion indicated the later version might address current issues.

- **Release and Quick Fixes for LM Studio 0.2.22**: The release of **LM Studio 0.2.22 Preview Build 1** was announced with features including UI touch-ups and updated llama.cpp, and URLs for Mac and corrected Windows installers were shared. After some confusion with incorrect version labeling, a new URL was provided and confirmed to work for Windows users.

- **Model Performance Discussions After Update**: Members discussed various model performances post **LM Studio update**, with a focus on GGUF format issues and the effectiveness of recent quantizations. A member highlighted a reasoning-gap in Llama 3 GGUF models using a 'banana test' and apple quantity scenario, comparing it to other formats' performance on logical reasoning tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://releases.lmstudio.ai/windows/0.2.22/preview/LM-Studio-0.2.22-Preview-1b-Setup.exe">no title found</a>: no description found</li><li><a href="https://x.com/bartowski1182/status/1785764456347103548">Tweet from bartowski (@bartowski1182)</a>: Ran into multiple issues making llamacpp quants for 70b instruct, it&#39;ll be up soon I promise :) eta is tomorrow morning</li><li><a href="https://releases.lmstudio.ai/windows/0.2.22/preview/LM-Studio-0.2.22-Preview-1-Setup.exe">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - a Hugging Face Space by ggml-org</a>: no description found</li><li><a href="https://releases.lmstudio.ai/windows/0.2.22/preview/LM-Studio-0.2.22-Preview-1a-Setup.exe">no title found</a>: no description found</li><li><a href="https://tenor.com/view/doja-cat-star-wars-gif-25078126">Doja Cat GIF - Doja Cat Star - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF">bartowski/Meta-Llama-3-8B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/qawe-asd-gif-26050335">Qawe Asd GIF - Qawe Asd - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/ojo-huevo-pase-de-huevo-cleanse-clensing-gif-4719953888830735498">Ojo Huevo GIF - Ojo Huevo Pase de huevo - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920).">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://www.canadacomputers.com/product_info.php?cPath=7_4528_4570&item_id=230804">Dell Treasure Box (Black) Desktop i5-4570, 16GB, 512GB SSD, DVD, Win10</a>: Dell RGB Treasure Box OptiPlex SFF (Refurbished) Consumer Desktop Intel Core i5-4570 (up to 3.6GHz), 16GB, 512GB SSD, DVD, Windows 10 Professional (EN/FR) (Black)
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1234815876772134932)** (4 messages): 

- **Model Loading Issue Raised**: A member mentioned having issues loading a model and sought assistance in resolving it.
- **Reminder of Discord Etiquette**: Another member reminded to avoid spamming questions across multiple channels, advising to keep such queries in a specific channel.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1234942956536856584)** (40 messages🔥): 

- **VRAM Misreading Spotted**: A member mentioned that **LM Studio** is incorrectly reading the **VRAM capacity** of their 7900xtx. They also have a 7800x3d with integrated GPU, but doubt it's causing the issue.
- **Performance Precedent Creates Confusion**: Despite having used a **RX 6600** with LM Studio GPU offloading before, a member faces an error stating "no ROCm-capable device is detected" after updating to version 0.2.18. This elicits discussion about support for ROCm and OpenCL implementations with various AMD GPUs.
- **HIP SDK Support Misconceptions Clarified**: Members exchanged information about the **compatibility of different AMD GPUs with ROCm and the HIP SDK**, stating that graphics cards like the RX 6600 and 6700XT are not supported by the HIP SDK which LM Studio utilizes.
- **Lamenting GPU Support on LM Studio**: While one member considered upgrading to a 7900 GRE, another advised that they would **be better off with a 7900XTX** for guaranteed compatibility with LM Studio's ROCm build. The **price difference** between models in their country sparked a humorous suggestion of a budget flight for hardware shopping.
- **Searching for Linux-Specific ROCm Builds**: The conversation revealed that **there is no ROCm build for Linux**, prompting a mention of Mozilla's work on **llamafile** as a potential workaround for issues related to AMD's driver support.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rocm.docs.amd.com/en/docs-5.7.1/release/gpu_os_support.html">GPU and OS Support (Linux) — ROCm 5.7.1 Documentation Home</a>: no description found</li><li><a href="https://tenor.com/view/doja-cat-star-wars-gif-25078126">Doja Cat GIF - Doja Cat Star - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://future.mozilla.org/news/llamafile-four-months-of-progress-towards-democratizing-ai/">Llamafile: four months of progress towards democratizing AI</a>: no description found</li><li><a href="https://www.ebuyer.com/1597063-sapphire-amd-radeon-rx-7900-xtx-pulse-graphics-card-for-gaming-11322-02-20g">Sapphire AMD Radeon RX 7900 XTX PULSE Graphics Card for Gaming - 24GB | Ebuyer.com</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1235008099333439568)** (2 messages): 

- **CrewAI Integration with RAG**: A member inquired about successfully integrating **LMStudio** with *Retrieval-Augmented Generation* for functionalities similar to **PDFSearch** or **WebsiteSearch** using **CrewAI**.
- **Embedder Preferences in CrewAI**: The same member mentioned the possibility of assigning an embedder like **huggingface** within **CrewAI**, but expressed interest in utilizing **LMStudio Nomic embed**.
- **Model Performance Observations**: They shared their experience testing models **Gemma**, **llama3 fp16**, and **Wizardlm**, finding **Gemma** to most align with their needs.
  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/)** (1 messages): 

yagilb: https://x.com/lmstudioai/status/1785796240656957514
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1235255279474577469)** (25 messages🔥): 

- **Tackling Positional OOD for Context Extension**: A member highlighted a solution to positional out-of-distribution (OOD) issues which allows models to generalize to longer contexts. They shared an [arXiv paper](https://arxiv.org/pdf/2401.01325) that proposes this method and considered it *one of the most slept-on papers* for context length extension.
- **Normalizing Outliers for Better Performance**: Further discussing the same paper, the member mentioned that models can maintain good performance with longer contexts by normalizing outlier values. This was a follow-up to the earlier discussion on extending context lengths in AI models.
- **Reference Implementation in llama.cpp**: An example implementation for the discussed concept can be found in `llama.cpp` on GitHub. It employs parameters `--grp-attn-n` and `--grp-attn-w` in a server executable, which the member linked to a [GitHub repository](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) with accompanying visualization and description.
- **Debating on "Infinite" Contexts and RoPE**: There was a discussion on the balance between preventing OOD issues and extending context capabilities, with some members referring to attention truncation as counterproductive. A member pointed out that "infinite" context length is misleading and mentioned the [ReRoPE implementation on GitHub](https://github.com/bojone/rerope), which was released 9 months prior by the original RoPE author, suggesting possible plagiarism.
- **The Myth of Infinite Context**: The channel had a lighthearted exchange acknowledging the impracticality of "infinite context" models, with a nod to the excessive number of related papers on arXiv and a quip about the impossibility of having enough VRAM for such models. They also referenced Google publishing one of the many papers on this topic.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/bojone/rerope">GitHub - bojone/rerope: Rectified Rotary Position Embeddings</a>: Rectified Rotary Position Embeddings. Contribute to bojone/rerope development by creating an account on GitHub.</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/master/examples/server">llama.cpp/examples/server at master · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1234774630288461868)** (25 messages🔥): 

- **Seeking the AI Swisshutnife**: A member asked about platforms for **MLOps bounties**, akin to an AI-focused Fiverr, expressing a strong interest in such a service. They received suggestions that while there isn't one dedicated to AI/MLOps, general programming bounties could be found on [Replit](https://replit.com/bounties).

- **Construction Tech Job Alert**: A job opportunity was shared for a software engineer experienced in Python and JavaScript at a Miami-based construction tech company. They have projects undergoing beta testing across the US and are open to remote candidates.

- **Machine Learning on Unreal Engine**: A member announced the launch of an **RAG-based AI assistant for Unreal Engine**, which aims to improve the workflow in game development and related fields. They invited Unreal Engine users to give it a try and provide feedback, touting its potential to speed up development and learning [check it out here](https://neuralgameworks.com).

- **A Battle of AI Assistants**: Following the revelation of the RAG-based tool, another member brought up their work with a **GPT-4 vision-based tool for Unreal Engine 5**, emphasizing the advantages of visual inputs for specific tasks such as blueprint editing in UE5.

- **Call for Computing Power**: One member inquired about potential grants or resources for **data generation and evaluation**, expressing a need for access to high-powered computing resources like A100 GPUs to accelerate their research beyond the limitations of their current setup.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://neuralgameworks.com">Neuralgameworks - Your Ultimate Unreal Engine AI Assistant</a>: no description found</li><li><a href="https://replit.com/bounties">Bounties</a>: Work with top Replit creators to bring your ideas to life.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1234856423482200074)** (9 messages🔥): 

- **AI Bubble Trouble**: A YouTube video titled "Is the AI bubble popping?" was shared that discusses whether there's a bursting bubble in the AI startup ecosystem. The video provides a narrative involving three AI startups and uses stability/inflection/cohere analysis. [Watch the YouTube video](https://www.youtube.com/watch?v=p0NxSk7YMrI&ab_channel=Synapse).

- **Memories Made Digital**: A GitHub repository for **Memary** was mentioned, a project aimed at creating long-term memory for autonomous agents using neo4j for storing memories graphically. Interest was expressed in its novel approach and potential performance. [Explore the Memary repo](https://github.com/kingjulio8238/memary).

- **Sudden Shutdown of GPT-2 Chatbot**: A Tweet from @itsandrewgao reported the gpt2-chatbot being turned OFFLINE unexpectedly, provoking curiosity about the sudden change. [View the Tweet](https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw).

- **AI Sensemaking Challenge**: A challenging problem was shared on Twitter by @VictorTaelin, who spent hours trying to solve it without success and expressed eagerness for a solution. [Check out the Twitter post](https://twitter.com/VictorTaelin/status/1785343416844353697).

- **Advanced Reasoning for AI**: An arXiv paper detailed a method for improving **Chain-of-Thought (CoT)** reasoning in AI by using iterative preference optimization and a specially modified loss function. This approach boosted accuracy significantly on various benchmarks such as GSM8K and MATH for **Llama-2-70B-Chat**. [Read the arxiv paper](https://arxiv.org/abs/2404.19733).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://verbasizer.com/">Verbasizer</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.19733">Iterative Reasoning Preference Optimization</a>: Iterative preference optimization methods have recently been shown to perform well for general instruction tuning tasks, but typically make little improvement on reasoning tasks (Yuan et al., 2024, Ch...</li><li><a href="https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from Andrew Gao (@itsandrewgao)</a>: gpt2-chatbot was just turned OFFLINE  I was just using it half an hour ago! @shaunralston for the find   #gpt2 @openai</li><li><a href="https://github.com/kingjulio8238/memary">GitHub - kingjulio8238/memary: Longterm Memory for Autonomous Agents.</a>: Longterm Memory for Autonomous Agents. . Contribute to kingjulio8238/memary development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=p0NxSk7YMrI&ab_channel=Synapse">Is the AI bubble popping?</a>: The story of 3 startups paints a portrait of an AI bubble that could be popping.Subscribe to Synapse for deeply researched stories that are shaping the AI la...</li><li><a href="https://github.com/KindXiaoming/pykan">GitHub - KindXiaoming/pykan: Kolmogorov Arnold Networks</a>: Kolmogorov Arnold Networks. Contribute to KindXiaoming/pykan development by creating an account on GitHub.</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected</a>: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected - SynaLinks/HybridAGI
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1235338435913453649)** (1 messages): 

- **Hermes 2 Goes Pro with Llama-3**: Nous Research announces **Hermes 2 Pro on Llama-3 8B**, enhancing capabilities with Function Calling and Structured Output. Their first Llama-3 based model, it surpasses its predecessor on various benchmarks and is now available on [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B).

- **Leading the Benchmarks**: **Hermes 2 Pro** has demonstrated superior performance over Llama-3 8B Instruct on AGIEval, GPT4All Suite, TruthfulQA, and BigBench, showcasing advancements in AI evaluation metrics.

- **Explore the Quantized Version**: For those interested in a lighter model, the quantized version of **Hermes 2 Pro Llama-3 8B** is available, offering the same advancements in a more size-efficient form on [HuggingFace GGUF](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF).

- **Collaboration Achievement**: A shout-out was given to the collaboration team behind **Hermes 2 Pro**, which included specific members contributing to the development and customization required for this latest model release.

- **Follow the Journey on Twitter**: Keep up with the latest updates by following Nous Research's progress with Hermes 2 Pro via their [Twitter announcement](https://twitter.com/NousResearch/status/1785779313826308096).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1234808223748587583)** (468 messages🔥🔥🔥): 

- **Llama and Hermes Performance Discussions**: Members discussed the performance differences between **Hermes 2 Pro Llama 3** and previously released models. Some pointed out that Hermes 2 Pro may have unlearned tasks such as the "apple test" but also gained new capabilities like **function calling**.

- **Quantizing Language Models**: The community debated the effectiveness of quantizing large language models (LLMs). It was noted that a limit exists around **5.5 bits per weight** where performance loss becomes significant when quantizing, and that Q8 quantization normally does not result in quality loss.

- **Training Challenges with Quantization**: There was a consensus that **1.58 bit LLMs** likely perform well in early training due to the regulatory properties of low-bit quantization but may diverge in performance as they reach the network's capacity limit.

- **Context Length in LLMs**: The topic of **context length** was also raised, with discussions on the practical limits and whether extensive soft prompt tuning (SPT) examples are worthwhile. It was highlighted that the longest valid samples average around **100/200k for text**.

- **New LLM Releases and Collaborative Efforts**: Enthusiasm was shown for potential new state-of-the-art models, with an **8B LLM briefly mentioned** along with interest in novel fine-tuning methods over existing models. Collaboration on these fronts is ongoing, exhibiting excitement and speculative planning from various members.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://google-research.github.io/seanet/audiopalm/examples/">AudioPaLM</a>: no description found</li><li><a href="https://x.com/hingeloss/">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/qtnx_/status/1785383089109172705?s=46&t=st">Tweet from Q (@qtnx_)</a>: llama-3-vision-alpha now works using @huggingface transformers</li><li><a href="https://x.com/teortaxestex/status/1785682723358622207">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: So llama 8b won&#39;t quantize well even if you fix token merging. Maybe the issue is vocab, maybe just overtraining, and I fear the latter. My (half-baked) intuition is that we&#39;re refining compos...</li><li><a href="https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/">LLM.int8() and Emergent Features &mdash; Tim Dettmers</a>: When I attended NAACL, I wanted to do a little test. I had two pitches for my LLM.int8() paper. One pitch is about how I use advanced quantization methods to achieve no performance degradation transfo...</li><li><a href="https://x.com/lmsysorg/status/1785394860754866234?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from lmsys.org (@lmsysorg)</a>: Thanks for the incredible enthusiasm from our community! We really didn&#39;t see this coming.   Just a couple of things to clear up:  - In line with our policy, we&#39;ve worked with several model de...</li><li><a href="https://huggingface.co/qresearch/llama-3-vision-alpha-hf">qresearch/llama-3-vision-alpha-hf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json">llava_instruct_150k.json · liuhaotian/LLaVA-Instruct-150K at main</a>: no description found</li><li><a href="https://x.com/itsandrewgao/status/1785373740622356753?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from Andrew Gao (@itsandrewgao)</a>: gpt2-chatbot was just turned OFFLINE  I was just using it half an hour ago! @shaunralston for the find   #gpt2 @openai</li><li><a href="https://tenor.com/view/over9000-dragonball-gif-26144830">Over9000 Dragonball GIF - Over9000 Dragonball - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/qtnx_/status/1785383089109172705?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from Q (@qtnx_)</a>: llama-3-vision-alpha now works using @huggingface transformers</li><li><a href="https://github.com/haotian-liu/LLaVA/blob/main/docs%2FFinetune_Custom_Data.md">LLaVA/docs/Finetune_Custom_Data.md at main · haotian-liu/LLaVA</a>: [NeurIPS&#39;23 Oral] Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond. - haotian-liu/LLaVA</li><li><a href="https://x.com/sanchitgandhi99/status/1785723896567640356">Tweet from Sanchit Gandhi (@sanchitgandhi99)</a>: Last week we released 🤗Diarizers, a library for fine-tuning speaker diarization models 🗣️  Using a free Google Colab, it takes 10 minutes to improve multilingual performance by 30%: https://colab.re...</li><li><a href="https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md">DeepSpeed/blogs/deepspeed-ulysses/README.md at master · microsoft/DeepSpeed</a>: DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective. - microsoft/DeepSpeed</li><li><a href="https://github.com/cpldcpu/BitNetMCU/blob/main/docs/documentation.md#model-capacity-vs-quantization-scaling">BitNetMCU/docs/documentation.md at main · cpldcpu/BitNetMCU</a>: Neural Networks with low bit weights on a CH32V003 RISC-V Microcontroller without multiplication - cpldcpu/BitNetMCU</li><li><a href="https://github.com/tincans-ai/gazelle">GitHub - tincans-ai/gazelle: Joint speech-language model - respond directly to audio!</a>: Joint speech-language model - respond directly to audio! - tincans-ai/gazelle</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>: Advanced RAG 101 - build agentic RAG with llama3Get free HubSpot report of how AI is redefining startup GTM strategy: https://clickhubspot.com/4hx🔗 Links- F...</li><li><a href="https://github.com/zhuzilin/ring-flash-attention">GitHub - zhuzilin/ring-flash-attention: Ring attention implementation with flash attention</a>: Ring attention implementation with flash attention - zhuzilin/ring-flash-attention</li><li><a href="https://youtu.be/ivo-z87x00I?si=w_Jawf7A6mehQnLf">Do NOT sleep on Whisper.cpp</a>: @ggerganov&#39;s Whisper.cpp is bringing OpenAI&#39;s Whisper to the masses. We discuss on &quot;The Changelog&quot; podcast. 🎧 👉 https://changelog.fm/532Subscribe for more!...</li><li><a href="https://github.com/jzhang38/EasyContext/blob/main/easy_context/zigzag_ring_attn/monkey_patch.py">EasyContext/easy_context/zigzag_ring_attn/monkey_patch.py at main · jzhang38/EasyContext</a>: Memory optimization and training recipes to extrapolate language models&#39; context length to 1 million tokens, with minimal hardware. - jzhang38/EasyContext</li><li><a href="https://x.com/hingeloss/status/1780718391461925049">Tweet from chris (@hingeloss)</a>: Presenting: the world&#39;s fastest AI voice chat - 500ms latency, running locally, 2x faster than anyone else.  How is this possible? 👇</li><li><a href="https://demo.tincans.ai/">🦌 Gazelle v0.2</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6936#issuecomment-2088803611">perplexity: more statistics, added documentation by JohannesGaessler · Pull Request #6936 · ggerganov/llama.cpp</a>: I have seen subjective reports about quantization being more harmful for LLaMA 3 than for LLaMA 2. I decided to investigate this and have to this end added more statistics (and documentation) to pe...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama : improve BPE pre-processing + LLaMA 3 and Deepseek support by ggerganov · Pull Request #6920 · ggerganov/llama.cpp</a>: Continuing the work in #6252 by @dragnil1 This PR adds support for BPE pre-tokenization to llama.cpp Summary The state so far has been that for all BPE-based models, llama.cpp applied a default pre...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1234773238702149632)** (16 messages🔥): 

- **The Quest for a Million Contexts**: An attempt to load the 1 M context on lm studio was unsuccessful, and it was clarified that models like **Phi-3 128k** don't run on ollama due to issues with supporting attention window mechanisms like *Rope Theta* and *Ring*.
   
- **LLaMA Pull Request to the Rescue**: Users reported an issue that has been resolved with a new [pull request](https://github.com/ggerganov/llama.cpp/pull/6920) to llama.cpp, improving BPE pre-processing and adding support for LLaMA 3 and Deepseek.
   
- **Tokenizer Troubles and GGUFs**: There was confusion about whether tokenizers were the root issue for a bug, and whether GGUFs required requantization, with some thinking the problema addressed and others not so sure.
   
- **Grokking Through Reverse Engineering**: A study detailed on [arXiv](https://arxiv.org/abs/2301.05217) about the phenomenon of "grokking" suggested using mechanistic interpretability to reverse-engineer learned behaviors of neural networks.
   
- **Ranking the Outputs of LLMs**: A method for qualitative ranking of LLM outputs was sought, with a suggestion to use **argilla distilable** or a reward model, although clarity on executing the actual evaluation in **distilable** was questioned.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2301.05217">Progress measures for grokking via mechanistic interpretability</a>: Neural networks often exhibit emergent behavior, where qualitatively new capabilities arise from scaling up the amount of parameters, training data, or training steps. One approach to understanding em...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6920">llama : improve BPE pre-processing + LLaMA 3 and Deepseek support by ggerganov · Pull Request #6920 · ggerganov/llama.cpp</a>: Continuing the work in #6252 by @dragnil1 This PR adds support for BPE pre-tokenization to llama.cpp Summary The state so far has been that for all BPE-based models, llama.cpp applied a default pre...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1234865912696537130)** (16 messages🔥): 

- **Introducing Wikipedia RAG Dataset**: A link to the **Wikipedia RAG dataset** on Hugging Face was shared, highlighting its relevance to the paper on *Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval*. This paper was published on Nov 10, 2023, and can be found [here](https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7).

- **Halal & Kosher Datasets?**: A member made a brief reference to the creation of datasets marked as *Halal & Kosher*, implying considerations for ethical or cultural compliance in dataset creation.

- **Cynde Integrates Pydantic**: The new Pydantic platform is being integrated into a rework of Cynde, which is interesting to members involved in the technical development.

- **Logfire Simplifies Code Observability**: The Logfire platform's introduction was discussed, denoted as a new observability platform facilitating the tracking of Pydantic models in function call settings. The platform, described as "intuitive" and "currently free," is praised for its ease of use and efficiency, with a specific reference to its capability of tracking nested CV jobs and providing significant data feedback. More about Logfire can be explored [here](https://pydantic.dev/logfire).

- **Model Fine-Tuning for Specific Output Formats**: A conversation took place around the fine-tuning of AI models to generate specific output formats, wherein a member suggests simplicity in instruction consistency to ensure proper formatting. **Hermes 2 Pro - Llama-3 8B** was mentioned as an example, particularly its structured output section on the [Hugging Face model page](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pydantic.dev/logfire">Pydantic Logfire | Uncomplicated observability</a>: Logfire is a new type of observability platform built on the same belief as Pydantic — that the most powerful tools can be easy to use.</li><li><a href="https://huggingface.co/collections/nthakur/swim-ir-dataset-662ddaecfc20896bf14dd9b7">🦢SWIM-IR Dataset - a nthakur Collection</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1234878753155584060)** (24 messages🔥): 

- **Virtual Business and Music Stardom Simulators Introduced**: [CompSimulator](https://hf.co/chat/assistant/662d91081ca01a81e3c21715) and [Snow Singer Simulator](https://hf.co/chat/assistant/6626e4869232378718adc5f2) are launched, offering users immersive experiences in the business and music industries respectively, powered by advanced AI technologies.
- **Eldritch Themes in Alternate History Simulation**: A member describes an alternate history simulation featuring *Eldritch Nazi* themes, cyberpunk influences, and an uprising in *Reichskommisariat Mittelafrika*.
- **Consistency in LLAMA 3 HF Chat Bot Responses**: It was noted that the **LLAMA 3** bot on HF Chat generates the **same response** for the *same message* sent to it.
- **World Simulation Talks & Global Community Engagement**: A [YouTube video](https://www.youtube.com/watch?v=abWnhmZIL3w) featuring talks from AGI House SF is shared, inspiring plans for a community meetup in LA and a global event connecting with SF and Japan.
- **Websim Game Development Updates**: A user announced a new game created on Websim, planning an update that will span from the stone age to the galactic age, with the link posted but leading to "null" and promising more features soon.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://websim.ai/c/mFPjhwkmqAvZROOAU-">no title found</a>: no description found</li><li><a href="https://hf.co/chat/assistant/6626e4869232378718adc5f2">Snow Singer Simulator - HuggingChat</a>: Use the Snow Singer Simulator assistant inside of HuggingChat</li><li><a href="https://hf.co/chat/assistant/662d91081ca01a81e3c21715">CompSim - HuggingChat</a>: Use the CompSim assistant inside of HuggingChat</li><li><a href="https://www.youtube.com/watch?v=abWnhmZIL3w">World Simulation Talks @ AGI House SF</a>: 0:00 Conversation1:31 Kickoff by Jeremy Nixon6:08 Karan Malhotra of Nous Research26:22 Rob Hasfield: CEO of Websim1:00:08 Ivan Vendrov of Midjourney [Real ti...
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1234761697533624351)** (497 messages🔥🔥🔥): 

- **SD3 Release Skepticism and Speculation**: Multiple users expressed doubts and concerns regarding the release of **Stable Diffusion 3 (SD3)**, mentioning claims of its release in April and anticipations for a May release, only to lament its absence. The discourse is marked by skepticism, supposing SD3 might never officially be released and speculation that **Stability AI** could face backlash for alleged misleading statements about SD3 being free and open-source.

- **Choosing the Right Model for Local Use**: Users are actively discussing the merits and tutorials of various Stable Diffusion local interfaces, including **ComfyUI**, **AUTO11111**, **Focus**, and **Forge**. Preferences seem to vary, with suggestions to choose based on ease of use and the user's specific hardware, like owning an NVIDIA vs. AMD GPU.

- **Prompt Enhancements and Descriptions with AI**: Individuals are inquiring about the most effective methods for image descriptions, debating the benefits of various AI tools. Mentioned options include using **ChatGPT**, **Gemini**, and employing models such as **Claude 3** and **idefics2** for analyzing and improving prompts for image generation.

- **Investments in AI Service Subscriptions and VPN Use**: There is active discussion and advice around investing in AI services such as **Gemini** and **Claude 3**, alongside shared practices involving **VPN** usage for region circumvention or maintaining privacy. Users are suggesting various VPNs and hinting at the usage of features like **DNS over HTTPS** for added security.

- **Creating and Using Labels in Automatic Extensions**: A user queries whether there's a way to embed labels in output images using extensions for **Automatic1111**, followed by inquiries about the existence of features equivalent to **clip skip and stylizer** within custom interfaces like **ComfyUI**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://civitai.com/articles/5069">Towards Pony Diffusion V7 | Civitai</a>: Hello everyone, I&#x27;m excited to share updates on the progress of our upcoming V7, along with a retrospective analysis of V6. The recognition V6 has ...</li><li><a href="https://tenor.com/view/yuji-stare-jujutsu-kaisen-blank-shibuya-sukuna-gif-2005904860443811921">Yuji Stare Jujutsu Kaisen GIF - Yuji Stare Jujutsu Kaisen Blank - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/idefics">Introducing IDEFICS: An Open Reproduction of State-of-the-art Visual Langage Model</a>: no description found</li><li><a href="https://huggingface.co/blog/idefics2">Introducing Idefics2: A Powerful 8B Vision-Language Model for the community</a>: no description found</li><li><a href="https://civitai.com/models/428813">Mythos - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: V1 it is somehow 3.55GB big.... i think i managed to do a stable fp8 prune???? i literally have no idea how it is 3.55GB... V2 is a normal 6GB mode...</li><li><a href="https://tenor.com/vD6Ib9MNmkI.gif">Melxts2008 Emoji GIF - Melxts2008 Emoji Smile - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://stability.ai/stable-assistant">Stable Assistant &mdash; Stability AI</a>: Stable Assistant is a friendly chatbot developed by Stability AI equipped with Stability AI’s text and image generation technology, featuring Stable Diffusion 3 and Stable LM 2 12B.</li><li><a href="https://github.com/hiddenswitch/ComfyUI/blob/master/script_examples/basic_api_example.py">ComfyUI/script_examples/basic_api_example.py at master · hiddenswitch/ComfyUI</a>: A powerful and modular stable diffusion GUI with a graph/nodes interface. - hiddenswitch/ComfyUI</li><li><a href="https://github.com/hiddenswitch/ComfyUI/blob/0862863bc00165b9ba0607595f304f93ca995887/tests/distributed/test_embedded_client.py#L32">ComfyUI/tests/distributed/test_embedded_client.py at 0862863bc00165b9ba0607595f304f93ca995887 · hiddenswitch/ComfyUI</a>: A powerful and modular stable diffusion GUI with a graph/nodes interface. - hiddenswitch/ComfyUI</li><li><a href="https://civitai.com/articles/5069?highlight=301393">Towards Pony Diffusion V7 | Civitai</a>: Hello everyone, I&#x27;m excited to share updates on the progress of our upcoming V7, along with a retrospective analysis of V6. The recognition V6 has ...</li><li><a href="https://civitai.com/articles/4248/what-is-score9-and-how-to-use-it-in-pony-diffusion">What is score_9 and how to use it in Pony Diffusion | Civitai</a>: You may&#x27;ve seen score_9 or its longer version score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up used in prompts for Pony Diffusio...</li><li><a href="https://github.com/Stability-AI/generative-models/blob/main/model_licenses/LICENSE-SDXL1.0">generative-models/model_licenses/LICENSE-SDXL1.0 at main · Stability-AI/generative-models</a>: Generative Models by Stability AI. Contribute to Stability-AI/generative-models development by creating an account on GitHub.</li><li><a href="https://github.com/AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin">GitHub - AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin: A user-friendly plug-in that makes it easy to generate stable diffusion images inside Photoshop using either Automatic or ComfyUI as a backend.</a>: A user-friendly plug-in that makes it easy to generate stable diffusion images inside Photoshop using either Automatic or ComfyUI as a backend. - AbdullahAlfaraj/Auto-Photoshop-StableDiffusion-Plugin
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1234949294767734865)** (1 messages): 

- **More Control Over Chat History**: OpenAI now **updates data controls** for both ChatGPT Free and Plus users. Anyone can access their chat history even if they've **opted out of training data** contribution; the update is live on web and coming soon to mobile.
- **Introducing Temporary Chat**: Users have a new option for privacy with the **Temporary Chat** feature, allowing one-off conversations that won't be stored in chat history.
  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1234785511420334090)** (375 messages🔥🔥): 

- **GPT-2 Chatbot Sparks Curiosity**: Members have discussed the "gpt2-chatbot" model, with some stating it performs better than GPT-4 in many cases, and others noting it fails in certain unidentified scenarios. Infinite generations with gpt2-chatbot seem possible, but the model has become unavailable in some arenas.

- **AI and Emotion**: A robust discussion unfolded around the concept of AI and emotions, with members pondering the potential for AI to develop its emotional awareness over time. Comparisons were made between AI evolution and human emotional development, with varying opinions on whether AI could or should strive to achieve a form of empathy or emotional understanding akin to humans.

- **Limits of the Free Tier**: A conversation regarding the accessibility of OpenAI's features like DALL-E for free users took place, with some expressing desires for added functionalities without subscriptions. The dialogue surfaced awareness of business realities and community desires for OpenAI's product offerings.

- **AI Collaboration in Academia**: One user queried the community on how to effectively collaborate with multiple AI models, like ChatGPT and Claude, in academic writing. Suggestions included the use of third-party chatbots that could retain other AI responses within the context.

- **Thoughts on DALL-E Updates**: Discussion covered DALL-E's current state and hypotheticals about future versions such as DALL-E 4. While some users noted improvements in DALL-E 3 leading to better creation results, the conversation also emphasized that good human-AI synergy remains crucial, and debated the importance of AI adapting to human cognitive patterns.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.udio.com/songs/7P8SyrG3cq9C1mpJfaGRMx">Udio | Echoes in the Chaos by Tcald | AI Music Generator - Official Website</a>: Listen to Echoes in the Chaos by Tcald on Udio. Discover, create, and share music with the world. Use the latest technology to create AI music in seconds.</li><li><a href="https://github.com/openai/simple-evals#benchmark-results">GitHub - openai/simple-evals</a>: Contribute to openai/simple-evals development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1234855100279685200)** (10 messages🔥): 

- **GPT-2's Exploration in Chat Systems**: A member shared their experience experimenting with **GPT-2** in chat system integrations. They directed further details of the discussion to a specific channel.

- **Archive Accidents and Bulk Deletion Queries**: A user inadvertently archived all of their chats and inquired about bulk deletion options to handle a large volume of chats as opposed to deleting them individually.

- **Screenshot Sharing**: There was a query about why screenshots cannot be posted in this channel, as a member wanted to share a humorous output of a GPT integration.

- **Directing Image-Friendly Channels**: It was clarified to a member that screenshots can be shared in another channel dedicated to such content.

- **Inconsistencies in ChatGPT's Character Limits**: A discrepancy was noted by a member where ChatGPT allegedly misrepresented its character limit, allowing for inputs longer than the stated 4096 characters.

- **Clarifying ChatGPT's Limitations and Behavior**: A member explained that **ChatGPT's self-awareness is limited as it is not trained to accurately know its capabilities or version**. They differentiated between the free and ChatGPT Plus versions with varying token limits, and mentioned the possibility of ChatGPT summarizing conversations when context limits are reached.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1234813834124988448)** (30 messages🔥): 

- **The Challenge of Negative Prompting**: Members discussed issues with *negative prompting*, indicating that providing examples of desired output is more effective than listing prohibited content. One suggestion included reframing instructions as "instead of *x,* use *y.*"

- **Regional Dialect Woes**: A use case was presented involving avoiding specific words that have different meanings in the Argentinian dialect of Spanish. Members suggested quizzing the AI on its understanding of Argentinian Spanish and considering an approach that explains the contextual use of words rather than a list of prohibitions.

- **Harnessing Positive Prompt Efficacy**: It's highlighted that positive prompting, possibly with the structure "instead of *x,* use *y,*" is likely to yield better compliance with GPT's outputs than listing negative examples or prohibitions.

- **Metadata-Prompting Explored**: For a hobbyist explorer, a simple form of meta-prompting using open variables and markdown for emphasis was discussed. It was suggested that this could enhance interactions with GPT.

- **Interactivity with AI Models**: The potential of meta-prompting to facilitate interactive, dynamic, and multi-layered prompts was also outlined. Examples include using placeholders for {openVariable} to guide the AI's behavior and structuring output templates to support the exchange.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1234813834124988448)** (30 messages🔥): 

- **Mulling Over Model Prompting Techniques**: Members discussed strategies for prompting OpenAI's models, emphasizing using **positive instructions and examples** over negative ones, to avoid undesirable word usage. They shared insights on constructing prompts that can drive the AI to produce better outcomes without listing prohibited words and highlighted various approaches such as using phrasing like *"instead of 'x', use 'y'"* to guide the AI's language choices.
  
- **Knowledge is Power**: In discussing techniques to generate a detailed **Ideal Customer Persona (ICP)** from LinkedIn data, a user presented a strategy involving analyzing posts and screenshots to determine a person's demographics, psychographics, and behaviors. The aim is to have the AI act as a personal branding and target audience expert as part of a content strategy for marketing and sales. 

- **Prompt Engineering 101**: A member requested advice on prompt engineering as a hobbyist looking to delve deeper into interacting with AI for knowledge and coding. Other participants offered suggestions like using **open variables** in meta-prompting and leveraging markdown for structuring and emphasizing parts of the prompt to encourage more complex AI behaviors.

- **Meta-Prompting for Interactive Experiences**: There was a consensus that **meta-prompting techniques,** where users create dynamic and interactive prompts for the AI, can significantly enhance the user's ability to achieve complex tasks. The conversation included an example of how to frame a meta-prompt for the AI to act as an expert system.

- **Journey into AI Prompt Engineering**: An AI enthusiast received encouragement and guidelines on starting with prompt engineering to improve their interactions with OpenAI's models. There was a particular discussion on the role of **open variables** and the use of markdown in prompts, as well as the potential benefits of using web search features for researching prompt engineering technologies.
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1235010301657743400)** (1 messages): 

- **Exclusive Early Access to New 'Pages' Feature**: A new feature called **Pages** is set to launch, offering an easy-to-create, shareable, in-depth exploration of any topic. Interested users can join the beta testing program for early access and the chance to provide feedback by reacting with a specific emoji and heading to the specified channel.
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1234782554809434122)** (241 messages🔥🔥): 

- **API Citation Woes**: A member inquired about obtaining citations like [1] and seeing web UI references via API requests while using **Perplexity-online** models. Another member explained that the anticipated program was suspended earlier due to fraud issues with the discount codes.
  
- **Flaws in Pro Search and Reference Features?**: Several users reported issues with Pro Search and reference features on Perplexity, noticing either redundant answers or missing references; one even claimed to face these glitches after upgrading to premium.

- **Questions Surrounding Opus Daily Limit**: Discussions around the daily limit for Opus usage surfaced, with members clarifying that it's **50 uses per day, replenished every 24 hours**. Some expressed dissatisfaction with the lack of estimates for when this limit might be increased.

- **Perplexity Performance and Issues**: Users shared experiences of slow responses from AI models and problems logging into accounts. There was advice to double-check spam folders for login links and speculations that service providers could block emails.

- **Clarity on Model Differences and Features**: The conversation touched on the varying quality of answers from different models and features like scratchpad prompts, AI prompting inaccuracies, and context window sizes for conversations. One user confirmed **the context window is indeed 32k**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chat.reka.ai/">Reka Playground</a>: Explore the latest multimodal language models built by Reka</li><li><a href="https://youtu.be/ddTV12hErTc">Rabbit R1: Barely Reviewable</a>: AI in a Box. But a different box.Get a dbrand skin and screen protector at https://dbrand.com/rabbitMKBHD Merch: http://shop.MKBHD.comTech I&#39;m using right no...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1234786005630976051)** (19 messages🔥): 

- **Exploring Perplexity AI**: Several users shared [Perplexity AI search results](https://www.perplexity.ai/search) exploring topics ranging from **Microsoft Research Asia** to the **Vimeo API** and queries about the **Mac App Store**.
- **LennysNewsletter on Product Insights**: A member shared a [link to Lenny's Newsletter](https://www.lennysnewsletter.com/p/how-perplexity-builds-product), which includes topics like Duolingo’s growth secret and how AI will impact product management, with an invitation to subscribe for full access.
- **Google's Recent Layoffs**: A link was circulated about [Google laying off employees](https://www.perplexity.ai/search/Google-lays-off-ZBS6dB9mSzqqA7OGS0M1sA) amidst other business adjustments.
- **Tesla's Full Self-Driving Discussion**: Automobile technology was a point of interest, with a link shared about [Tesla's full self-driving](https://www.perplexity.ai/search/Teslas-full-selfdriving-IJfuMlVMR_ay5YL0F49BlA) capabilities.
- **Reminder for Shareability on Discord**: Perplexity AI reminded users to ensure their threads are shareable, providing a visual guide linked directly from Discord's platform.

**Link mentioned**: <a href="https://www.lennysnewsletter.com/p/how-perplexity-builds-product?utm_medium=web">How Perplexity builds product</a>: Johnny Ho, co-founder and head of product, explains how he organizes his teams like slime mold, uses AI to build their AI company, and much more

  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1234782587151450142)** (14 messages🔥): 

- **Confusion about API Citations**: A member inquired about the possibility to get citations through API requests when using the **perplexity-online models** for web knowledge, and another member referred to earlier messages that seemingly addressed related concerns.

- **Policy Clarification for Claude 3 Use**: A user asked about the usage policy for **Claude 3** provided by Perplexity, especially concerning political use, and if Perplexity's usage policy takes precedence over **Anthropic's** when using their models.

- **Perplexity Pro vs. API Results Disparity**: A user highlighted a discrepancy between results obtained from the **Perplexity Pro** interface and those from the **API** using the same prompt, to which a fellow member clarified that Perplexity UI and API might not be using the same model version.

- **API Documentation Clarification**: In response to confusion over model versions, a user referenced the **[Perplexity API documentation](https://docs.perplexity.ai/docs/model-cards)**, which lists models like `llama-3-70b-instruct` with details on parameters and instructed members on how to avoid prompt injections. 

- **Understanding Online Models**: A user questioned which online model Perplexity Pro UI uses, leading to an explanation that **online models** are either finetuned to use sources more effectively or employ a RAG-like approach to synthesize responses from a search engine-style vector database.

**Link mentioned**: <a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1234819191492710490)** (28 messages🔥): 

- **Effort/bucketMul for Efficient Inference**: A new algorithm called **effort/bucketMul** was introduced, which claims to significantly speed up vector-matrix approximation and LLM inference. It's described as adjustable in real-time for computational load and is compatible with models like Mistral. [Algorithm launched](http://kolinko.github.io/effort/).

- **Amateur AI Hobbyist Presents Image Patch Study**: An amateur AI enthusiast shared their research on efficient image patch representation inspired by neural systems, available on [arXiv](https://arxiv.org/abs/2210.13004). They propose a novel binary vector representation learned through unsupervised learning.

- **Discussion on Binary vs. Hypersphere Embeddings**: Members discussed the merits of binary vector representations for embeddings, linking their benefits to biological plausibility and computational efficiency. One member considered applying similar principles to the RWKV LLM for potentially faster learning. [RWKV LLM method](https://github.com/BlinkDL/SmallInitEmb).

- **Recommendations for Embedding Strategies**: In response to the discussion on representations, links to foundational papers in the space, including CLIP and Dino, were shared for further reading on embedding distributions. [CLIP Paper](https://arxiv.org/abs/2103.00020), [Dino Paper](https://arxiv.org/abs/2104.14294).

- **Query on Image Classification with CLIP Embeddings**: A member sought advice on classifying images of movie stars using CLIP embeddings, obtaining only 36% accuracy with both modified labels and prompts. They explored using cosine similarity with text descriptions but are considering alternative approaches due to the lack of improvement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2102.11174">Linear Transformers Are Secretly Fast Weight Programmers</a>: We show the formal equivalence of linearised self-attention mechanisms and fast weight controllers from the early &#39;90s, where a ``slow&#34; neural net learns by gradient descent to program the ``f...</li><li><a href="http://kolinko.github.io/effort/">Effort Engine</a>: A possibly new algorithm for LLM Inference. Adjust smoothly - and in real time - how many calculations you'd like to do during inference.</li><li><a href="https://arxiv.org/abs/2210.13004">Efficient Representation of Natural Image Patches</a>: Utilizing an abstract information processing model based on minimal yet realistic assumptions inspired by biological systems, we study how to achieve the early visual system&#39;s two ultimate objecti...</li><li><a href="https://www.reddit.com/user/No_Dragonfruit_5472/comments/1cef7gc/tradingview_premium_pack_crack_2024/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1234762365325545512)** (192 messages🔥🔥): 

- **Unraveling the "Black Box" Analogy**: The discussion revealed varying perspectives on why large language models (LLMs) are often referred to as "black boxes." Some participants noted the complexity of LLMs' inner workings relative to our understanding, while others suggested that the imprecise use of such terms reflects a human tendency to parrot pithy phrases.

- **Training LLMs on Test Sets Affects Fair Comparisons**: A shared [link](http://arxiv.org/abs/2404.18824) points out that LLMs trained on benchmark test sets skew the effectiveness of benchmarks and foster potentially unfair comparisons. 

- **Chain-of-Thought (CoT) Rationality in LLMs**: Tackling the issue of how LLMs explain their reasoning, some messages suggested that LLM-generated explanations for an answer are not trustworthy as they often do not reflect the model's internal thought process.

- **Kolmogorov-Arnold Networks (KANs) Outperform MLPs**: Highlighted was a [paper](http://arxiv.org/abs/2404.19756) that introduces Kolmogorov-Arnold Networks (KANs) as an alternative to Multi-Layer Perceptrons (MLPs), noting that KANs offer better accuracy and interpretability with faster scaling laws and potential for intuitive visualization.

- **Iterative Preference Optimization to Improve LLM Reasoning**: Shared research ([link](http://arxiv.org/abs/2404.19733)) discusses an iterative method to improve LLM reasoning by optimizing the preference between competing generated CoT candidates, leading to increased accuracy in tasks like GSM8K, MATH, and others.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://arxiv.org/abs/2404.18824">Benchmarking Benchmark Leakage in Large Language Models</a>: Amid the expanding use of pre-training data, the phenomenon of benchmark dataset leakage has become increasingly prominent, exacerbated by opaque training processes and the often undisclosed inclusion...</li><li><a href="https://arxiv.org/abs/2404.19756">KAN: Kolmogorov-Arnold Networks</a>: Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation fun...</li><li><a href="http://arxiv.org/abs/2404.19733">Iterative Reasoning Preference Optimization</a>: Iterative preference optimization methods have recently been shown to perform well for general instruction tuning tasks, but typically make little improvement on reasoning tasks (Yuan et al., 2024, Ch...</li><li><a href="https://arxiv.org/abs/2404.14662">NExT: Teaching Large Language Models to Reason about Code Execution</a>: A fundamental skill among human developers is the ability to understand and reason about program execution. As an example, a programmer can mentally simulate code execution in natural language to debu...</li><li><a href="http://arxiv.org/abs/2001.04063">ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training</a>: This paper presents a new sequence-to-sequence pre-training model called ProphetNet, which introduces a novel self-supervised objective named future n-gram prediction and the proposed n-stream self-at...</li><li><a href="https://videogigagan.github.io/">VideoGigaGAN</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.12365">Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators</a>: Neural operators, serving as physics surrogate models, have recently gained increased interest. With ever increasing problem complexity, the natural question arises: what is an efficient way to scale ...</li><li><a href="https://arxiv.org/abs/2404.12388">VideoGigaGAN: Towards Detail-rich Video Super-Resolution</a>: Video super-resolution (VSR) approaches have shown impressive temporal consistency in upsampled videos. However, these approaches tend to generate blurrier results than their image counterparts as the...</li><li><a href="http://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is imp...</li><li><a href="http://arxiv.org/abs/2312.02179">Training Chain-of-Thought via Latent-Variable Inference</a>: Large language models (LLMs) solve problems more accurately and interpretably when instructed to work out the answer step by step using a ``chain-of-thought&#39;&#39; (CoT) prompt. One can also improv...</li><li><a href="https://github.com/lauraaisling/analyse-llms/blob/main/notebooks/Mode_Collapse.ipynb">analyse-llms/notebooks/Mode_Collapse.ipynb at main · lauraaisling/analyse-llms</a>: Contribute to lauraaisling/analyse-llms development by creating an account on GitHub.</li><li><a href="https://arxiv.org/abs/2403.18506">Faster Convergence for Transformer Fine-tuning with Line Search Methods</a>: Recent works have shown that line search methods greatly increase performance of traditional stochastic gradient descent methods on a variety of datasets and architectures [1], [2]. In this work we su...</li><li><a href="https://github.com/s-chh/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10">GitHub - s-chh/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10: Simplified Pytorch implementation of Vision Transformer (ViT) for small datasets like MNIST, FashionMNIST, SVHN and CIFAR10.</a>: Simplified Pytorch implementation of Vision Transformer (ViT) for small datasets like MNIST, FashionMNIST, SVHN and CIFAR10. - s-chh/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.04.28.591528v1">Sequential predictive learning is a unifying theory for hippocampal representation and replay</a>: The mammalian hippocampus contains a cognitive map that represents an animal's position in the environment and generates offline &quot;replay&quot; for the purposes of recall, planning, and forming lo...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1235201401781616690)** (34 messages🔥): 

- **Exploring the Computational Model of Sequence-Prediction**: A member theorized about the computational model learned by sequence-prediction models, particularly related to next-token prediction loss, predicting the existence of phase transitions in token probabilities and seeking feedback on their write-up [here](https://docs.google.com/document/d/11w3of15CbfOlWrvQpTjxaJt-UvtOckzr0WQUfTrTnsw/edit?usp=sharing).

- **Connecting Prior Work with Theoretical Predictions**: The member acknowledged the relevance of existing research on transformers and iterative inference, notably the *tuned lens* method from [this paper](https://arxiv.org/abs/2303.08112), and discussed how findings from early decoding align with their proposed theory.

- **Discussing Model Representations with Tied Embeddings**: Dialogue ensued about how models with tied embeddings, like Mamba, might affect interpretation, with speculation that tied embeddings could actually benefit the model’s representational coherence.

- **Drafting Implementation Plans for Theoretical Predictions**: In response to whether implementations have been considered to test the hypotheses, a discussion took place about possibly using *transformer lens* and *gpt-2-small* to conduct experiments.

- **Exchanging Interpretability Insights**: Members exchanged views on the challenges of defining and operationalizing the "atomicity" of model features. References were made to emerging concepts like the *distributional simplicity bias* and the *Quantization Model* of neural scaling laws, linking to research papers [here](https://arxiv.org/abs/2402.04362) and [here](https://arxiv.org/abs/2303.13506).

- **Refining Interpretability Methods with Formal Languages**: A suggestion was made to define an arbitrary formal grammar and train a network on sequences from that language to determine if the rules of the grammar could be considered the “true underlying features,” investigating transformers' understandings of Dyck languages as a pertinent angle.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2303.13506">The Quantization Model of Neural Scaling</a>: We propose the Quantization Model of neural scaling laws, explaining both the observed power law dropoff of loss with model and data size, and also the sudden emergence of new capabilities with scale....</li><li><a href="https://arxiv.org/abs/2402.04362">Neural Networks Learn Statistics of Increasing Complexity</a>: The distributional simplicity bias (DSB) posits that neural networks learn low-order moments of the data distribution first, before moving on to higher-order correlations. In this work, we present com...</li><li><a href="https://arxiv.org/abs/2303.08112">Eliciting Latent Predictions from Transformers with the Tuned Lens</a>: We analyze transformers from the perspective of iterative inference, seeking to understand how model predictions are refined layer by layer. To do so, we train an affine probe for each block in a froz...</li><li><a href="https://docs.google.com/document/d/11w3of15CbfOlWrvQpTjxaJt-UvtOckzr0WQUfTrTnsw/edit?usp=sharing">Deriving a Model of Computation for Next-Token Prediction</a>: no description found
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1234762736504672346)** (2 messages): 

- **Cash in for CVPR Competitions**: HuggingFace has announced three different competitions for the CVPR event with a total prize pool of **$120,000+**. Participants can join [SnakeCLEF](https://huggingface.co/spaces/BVRA/SnakeCLEF2024), [FungiCLEF](https://huggingface.co/spaces/BVRA/PlantCLEF2024), and [PlantCLEF](https://huggingface.co/spaces/BVRA/PlantCLEF2024) from June 17-21, 2024.
- **Transformers Library Update**: The *Transformers* library has been updated to [v4.40.0](https://github.com/huggingface/transformers/releases/tag/v4.40.0), featuring models like Phi-3, Llama 3, IDEFICS 2, and more. Additionally, Phi-3 is set to be operable within the browser, achieving about 20 tokens per second.
- **Gradio and Datasets Library Enhancements**: Gradio has released a significant update with [version 4.28.0](https://www.gradio.app/changelog), focusing on Custom Components, while the Datasets library has reached [v2.19.0](https://github.com/huggingface/datasets/releases/tag/2.19.0) with Polars compatibility and improved export functionalities.
- **Empower Your Prompts**: HF Blog spotlights techniques for enhancing prompt consistency in language model outputs through a post on [Structured Generations](https://huggingface.co/blog/evaluation-structured-outputs).
- **Snowflake's Impressive Model Release**: Snowflake has released a whopping 408B Dense + Hybrid MoE model, boasting 17B active parameters and a wide range of capabilities like SQL generation, coding, and instruction following. This achievement is detailed in a highlighted [announcement](https://x.com/reach_vb/status/1783129119435210836).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/fleetwood___/status/1783195985893863578)">Tweet from Fleetwood (@fleetwood___)</a>: 🚨 Phi-3 running in the browser 🚨  Hits about 20 tok/s 🏎️ Literally 3 lines of JS.  Still some kinks to iron out, coming to Ratchet 0.4.0 soon.</li><li><a href="https://x.com/abhi1thakur/status/1785279012232736991)">Tweet from abhishek (@abhi1thakur)</a>: Can I run AutoTrain UI on Kaggle? Yes, you can!!! Check out my latest notebook, copy it, fill in your tokens and enjoy AutoTrain UI running on Kaggle Notebooks backend 🚀 Link to notebook: https://www...</li><li><a href="https://x.com/reach_vb/status/1785039538185703909)!">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Let&#39;s go!! Common Voice 17 - now on the Hub! 🔥  With 31,000 hours of audio (& transcriptions) across 124 languages.  *sound on 🎶*  847 hours of data were added in CV 17, along with 493 hours of ...</li><li><a href="https://x.com/BrigitteTousi/status/1783573043815596426):">Tweet from Brigitte 🤗 (@BrigitteTousi)</a>: 🔊Calling all journalists! With @fdaudens, we&#39;re excited to announce a new community on the @huggingface Hub: Journalists on Hugging Face. 📰🤗  https://huggingface.co/JournalistsonHF 1/</li><li><a href="https://x.com/reach_vb/status/1783129119435210836)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Snowflake dropped a 408B Dense + Hybrid MoE 🔥  &gt; 17B active parameters &gt; 128 experts &gt; trained on 3.5T tokens &gt; uses top-2 gating &gt; fully apache 2.0 licensed (along with data recipe to...</li><li><a href="https://x.com/RisingSayak/status/1785162074844197174)">Tweet from Sayak Paul (@RisingSayak)</a>: Custom pipelines and components in Diffusers 🎸  Wanted to use customized pipelines and other components (schedulers, unets, text encoders, etc.) in Diffusers?  Found it inflexible?   This 🧶 is for y...</li><li><a href="https://x.com/lunarflu1/status/1785359306847666431)">Tweet from lunarflu (@lunarflu1)</a>: You can now mention people on @huggingface !
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1234770448382824468)** (151 messages🔥🔥): 

- **Chronos Model Fine-Tuning Inquiry**: A member sought guidance on fine-tuning the [Chronos time-series forecasting model](https://huggingface.co/amazon/chronos-t5-small). They were redirected to the GitHub repository for further details.
- **Hugging Face Job Seeker**: A software engineer with 10 years of experience reached out for opportunities at Hugging Face, and was directed to Hugging Face's [job openings](https://apply.workable.com/huggingface/?lng=en), including a wild card position.
- **Difficulty with Rasa Framework for a Chatbot**: A new member is experiencing accuracy issues with intent recognition in a sales-related chatbot using Rasa Framework and is considering making a custom NER model.
- **Spaces Newbie Questions**: Members asked about receiving notifications for new replies in Space community threads, and it was noted that notifications are sent by default.
- **Kaggle and Google Collaboratory Tips Shared**: Several members discuss using Kaggle and Google Colab's free GPUs for training models, with advice exchanged on the settings to increase VRAM and Kaggle's phone verification to enable internet access.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://apply.workable.com/huggingface/?lng=en">Hugging Face</a>: Here at Hugging Face, we’re on a journey to advance and democratize ML for everyone. Along the way, we contribute to the development of technology for the better.</li><li><a href="https://huggingface.co/spaces/Nick088/Stable_Diffusion_Finetuned_Minecraft_Skin_Generator">Stable Diffusion Finetuned Minecraft Skin Generator - a Hugging Face Space by Nick088</a>: no description found</li><li><a href="https://huggingface.co/amazon/chronos-t5-small">amazon/chronos-t5-small · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/drax-guardians-of-the-galaxy-odds-bet-chance-gif-8058651">Drax Guardians Of The Galaxy GIF - Drax Guardians Of The Galaxy Odds - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/26">zero-gpu-explorers/README · The invited application has been waiting. How long does it take to be approved?</a>: no description found</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>: Advanced RAG 101 - build agentic RAG with llama3Get free HubSpot report of how AI is redefining startup GTM strategy: https://clickhubspot.com/4hx🔗 Links- F...</li><li><a href="https://github.com/johko/computer-vision-course">GitHub - johko/computer-vision-course: This repo is the homebase of a community driven course on Computer Vision with Neural Networks. Feel free to join us on the Hugging Face discord: hf.co/join/discord</a>: This repo is the homebase of a community driven course on Computer Vision with Neural Networks. Feel free to join us on the Hugging Face discord: hf.co/join/discord - johko/computer-vision-course</li><li><a href="https://github.com/amazon-science/chronos-forecasting?tab=readme-ov-file">GitHub - amazon-science/chronos-forecasting: Chronos: Pretrained (Language) Models for Probabilistic Time Series Forecasting</a>: Chronos: Pretrained (Language) Models for Probabilistic Time Series Forecasting - amazon-science/chronos-forecasting</li><li><a href="https://github.com/huggingface/accelerate/pull/2732">Fixes a few Sagemaker config issues by nroggendorff · Pull Request #2732 · huggingface/accelerate</a>: Updates config_args.py to work with the latest version of amazon sagemaker In this new version, you are required to run variables operations with True or False, like --do_eval True as apposed to ju...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1234782066428608512)** (3 messages): 

- **Seeking Finetuning Guidance**: A member expressed interest in learning how to generate an instruction dataset for finetuning Large Language Models (LLMs).
- **In Search of Clarifications**: Another member inquired for further details on what exactly the first member was referring to in terms of generating an instruction dataset for LLM finetuning.
- **Introducing Med-Gemini for Medicine**: A member shared a [YouTube video](https://youtu.be/xohuoN2WBZs?si=Ku6cztykld6dZLN9) providing a high-level overview of **Med-Gemini**, Google’s multimodal GenAI models for medicine, aiming to inform and reassure interested parties about the technology.

**Link mentioned**: <a href="https://youtu.be/xohuoN2WBZs?si=Ku6cztykld6dZLN9">Med-Gemini: A High-Level Overview</a>: A high-level overview on Med-Gemini, Google&#39;s &quot;Family&quot; (said in the voice of Vin Diesel) of Multimodal GenAI models for medicine. Med-Gemini has folks in the...

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1234797591523229716)** (8 messages🔥): 

- **Cool Tools for AI Enthusiasts**: A Medium post entitled ["5 Interesting AI Tools Everyone Should Try"](https://medium.com/illumination/genai-adventures-5-interesting-ai-tools-everyone-should-try-44ae8f8115af) was recommended, listing a variety of AI applications that could be of interest to people in the field.
- **Webloading the Future**: An article on Medium discusses how to use Groq, Langchain, and Datastax to create robust Webloader RAG applications, [read more about it here](https://medium.com/ai-advances/building-powerful-webloader-rag-applications-with-groq-langchain-and-datastax-f4816d88bee8).
- **SQL Simplified**: The Data Intelligence Alliance through its website, [www.dataialliance.org](https://www.dataialliance.org), is developing a “people database” to allow individuals to interact with databases with little or no prior SQL knowledge.
- **Microscope Image Segmentation Made Easy**: The GitHub repository for Micro-SAM, a project designed to simplify the process of segmenting microscopy images, is now available and can be checked out [here](https://github.com/computational-cell-analytics/micro-sam).
- **Accelerating Diffusion Models**: The Hugging Face documentation details several techniques to speed up diffusion models without compromise, and highlights how using PyTorch 2 can triple the inference speed of text-to-image pipelines, particularly demonstrated with [Stable Diffusion XL (SDXL)](https://huggingface.co/docs/diffusers/tutorials/fast_diffusion).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.dataialliance.org">blog</a>: no description found</li><li><a href="https://huggingface.co/docs/diffusers/tutorials/fast_diffusion">Accelerate inference of text-to-image diffusion models</a>: no description found</li><li><a href="https://github.com/computational-cell-analytics/micro-sam">GitHub - computational-cell-analytics/micro-sam: Segment Anything for Microscopy</a>: Segment Anything for Microscopy. Contribute to computational-cell-analytics/micro-sam development by creating an account on GitHub.</li><li><a href="https://youtu.be/IDIv92Z6Qvc?si=NlBDh0KtHNq63XvN">ETH Zürich DLSC: Physics-Informed Neural Networks - Applications</a>: ↓↓↓ LECTURE OVERVIEW BELOW ↓↓↓ETH Zürich Deep Learning in Scientific Computing 2023Lecture 5: Physics-Informed Neural Networks - ApplicationsLecturers: Ben M...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1234859558602670111)** (11 messages🔥): 

- **Leak-Free Link Prediction Methodology**: A GitHub repository called [PnPR-GCN_ACM_SAC_24](https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24) addresses the issue of information leaks in K-fold cross-validation on transitive graphs. The methodology proposed ensures data splitting without information leakage, enhancing concept prerequisite learning.

- **Aligning Scheduling with AI**:
A tweet from [dstackai](https://twitter.com/dstackai/status/1785315721578459402) introduced a guide on using the Alignment Handbook alongside dstack to facilitate the scheduling of fine-tuning tasks on cloud or on-premises machines.

- **Iterative SDXL Inpainting on 🤗 Spaces**: The [inpainting SDXL sketch pad](https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad) allows for iterative inpainting and version history to restore previous image versions, but currently, the Space is sleep due to inactivity.

- **HDR Challenge with Display Compatibility**: Mentioned images being HDR encoded, recommending to view them fullscreen for proper color representation, especially on devices like iOS/iPadOS, otherwise, they may appear washed out.

- **Chat in 55 Languages with Bloom**: [Bloom Multilingual Chat](https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat) is a Hugging Face Space where users can converse with the Bloom model in 55 languages through the use of the `deep_translator` Python library for query translation and back-translation.

- **Batch Process Your Moon Dreams**: A new batch processing feature has been added to MoonDream2, allowing for multiple images to be processed at once. Check out the MoonDream2 batch processing [here](https://huggingface.co/spaces/Csplk/moondream2-batch-processing).

- **FluentlyXL V4 Unveiled**: The FluentlyXL V4 model emphasizes on contrast, realism, and accurate anatomy. You can try this enhanced model at [Fluently Playground](https://huggingface.co/spaces/fluently/Fluently-Playground).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/fluently/Fluently-XL-v4">fluently/Fluently-XL-v4 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/tonyassi/inpainting-sdxl-sketch-pad">Inpainting SDXL Sketch Pad - a Hugging Face Space by tonyassi</a>: no description found</li><li><a href="https://huggingface.co/spaces/Csplk/moondream2-batch-processing">moondream2-batch-processing - a Hugging Face Space by Csplk</a>: no description found</li><li><a href="https://huggingface.co/spaces/as-cle-bert/bloom-multilingual-chat">Bloom Multilingual Chatbot - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://github.com/Lama-West/PnPR-GCN_ACM_SAC_24/tree/main">GitHub - Lama-West/PnPR-GCN_ACM_SAC_24</a>: Contribute to Lama-West/PnPR-GCN_ACM_SAC_24 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1234763524392226856)** (18 messages🔥): 

- **Graph Papers Galore**: A member highlighted a paper to read, titled "Graphs play an important role in representing complex relationships" and available at [arXiv:2404.14928](https://arxiv.org/abs/2404.14928). They also mentioned considering other graph-related surveys but wished to avoid overextending their focus.

- **Distillation Insights on the Horizon**: Participants discussed distillation in score-based models, mentioning that the **Laion server** contains experts in the field and suggesting papers by Segmind, discussing *rectified/instaflow*, *lcm lora*, and the *piecewise rectified flow*.

- **Reading Group Event Scheduled**: An event for the reading group was organized and a link was provided for the participants to suggest different times, with a note to accommodate for everyone’s availability.

- **NegotiationArena: A New Playground for LLMs**: Appreciation was shown for a presentation on a paper about how well Large Language Models (LLMs) can negotiate with each other using a framework called **NegotiationArena**, the paper can be found at [arXiv:2402.05863](https://arxiv.org/abs/2402.05863).

- **Negotiation as an LLM Alignment Metric**: A member remarked on the unique aspect of negotiating tasks as a potential metric for evaluating the alignment of LLMs, recognizing that the task differs from regular downstream tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.14928">Graph Machine Learning in the Era of Large Language Models (LLMs)</a>: Graphs play an important role in representing complex relationships in various domains like social networks, knowledge graphs, and molecular discovery. With the advent of deep learning, Graph Neural N...</li><li><a href="https://arxiv.org/abs/2402.05863">How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis</a>: Negotiation is the basis of social interactions; humans negotiate everything from the price of cars to how to share common resources. With rapidly growing interest in using large language models (LLMs...</li><li><a href="https://discord.gg/hugging-face-879548962464493619?event=1234913780048203856">Join the Hugging Face Discord Server!</a>: We&#x27;re working to democratize good machine learning 🤗Verify to link your Hub and Discord accounts! | 77668 members</li><li><a href="https://arxiv.org/abs/2312.02783">Large Language Models on Graphs: A Comprehensive Survey</a>: Large language models (LLMs), such as GPT4 and LLaMA, are creating significant advancements in natural language processing, due to their strong text encoding/decoding ability and newly found emergent ...</li><li><a href="https://arxiv.org/abs/2310.11829">Towards Graph Foundation Models: A Survey and Beyond</a>: Foundation models have emerged as critical components in a variety of artificial intelligence applications, and showcase significant success in natural language processing and several other domains. M...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1234761217084493844)** (17 messages🔥): 

- **Improving YOLO Models**: A member mentioned they are working on enhancing the accuracy of YOLO architectures even if it means a slower model, and recognized that modifying the architecture could be time-consuming.
- **Collaboration Sought for CNN Study**: A user is looking for a partner to study and learn about Convolutional Neural Networks (CNNs) together.
- **YOLOv5 Parallel Processing Tip**: Sliding window approach for parallelism in YOLOv5 is suggested along with an idea to look into pre-YOLO/CNN image segmentation and contour algorithms, hinting that image simplification and downsampling can yield effective results.
- **Learning Curve in PyTorch vs TensorFlow**: A discussion on whether to learn PyTorch or TensorFlow for CNNs took place, where it was acknowledged TensorFlow has a steeper learning curve, though it offers more devops support from Google, while PyTorch has more academic support and community momentum.
- **Kaggle Discussion and Tool for Computer Vision**: A user shared a [Kaggle discussion](https://www.kaggle.com/discussions/general/498337) link to their work which has been designed to assist with training or fine-tuning CV models and are seeking feedback.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.opencv.org/3.4/d2/d96/tutorial_py_table_of_contents_imgproc.html">OpenCV: Image Processing in OpenCV</a>: no description found</li><li><a href="https://www.kaggle.com/discussions/general/498337">3LC - Real-Time 3D Visualizer/Debugger/Data Editor for Training/Finetuning your Models - Free! | Kaggle</a>: 3LC - Real-Time 3D Visualizer/Debugger/Data Editor for Training/Finetuning your Models - Free!.</li><li><a href="https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html">OpenCV: Morphological Transformations</a>: no description found</li><li><a href="https://docs.3lc.ai/3lc/latest/public-notebooks/pytorch-lightning-segformer.html">Training a finetuned SegFormer model with Pytorch Lightning - </a>: no description found</li><li><a href="https://docs.3lc.ai/3lc/latest/public-notebooks/detectron2-balloons.html">Balloons Toy Dataset + Detectron2 + 3LC Tutorial - </a>: no description found</li><li><a href="https://docs.3lc.ai/3lc/latest/user-guide/integrations/yolov5/yolov5.html">Integrating 3LC with YOLOv5 🚀 - </a>: no description found</li><li><a href="https://docs.3lc.ai/3lc/latest/user-guide/integrations/yolov8/yolov8.html">Integrating 3LC with YOLOv8 🚀 - </a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1234885730283556907)** (5 messages): 

- **Seeking Guidance for NLP Project**: A new member is working on a **chatbot** project and is experiencing difficulties with intent recognition using the **Rasa framework**. They are considering creating a custom **NER model** to identify specific terms related to their business and ponder whether to ***"make [their] own model,"*** use **Spacy**, or utilize a pretrained model from **HuggingFace** to improve their bot's performance.
  
- **Inquiring About Ollama Template Roles**: Another member has queries regarding adding a "Reviewer" role to the **Ollama template** roles in order to evaluate the assistant's response format, seeking how to implement this by way of a template. They reference existing documentation at [Transformers chat templating guide](https://huggingface.co/docs/transformers/main/es/chat_templating).

- **Development of a Mini Emo Bot for College Tech Club**: A member is building an **NLP model for a Mini bot** designed to interact with oral prompts, search for specific information, and provide spoken responses, potentially to be deployed on a **Raspberry Pi**. They request assistance and guidance as they are new to the field of **NLP**.
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/)** (1 messages): 

sayakpaul: Might be a better question for A1111 forums.
  

---


**HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1234862689357009087)** (1 messages): 

- **Gradio Share Server Issues Alert**: Gradio is currently experiencing problems with the Share Server, which might affect sharing and usage on Colab. They are actively investigating and resolving the issue, and users are directed to [check the status here](https://status.gradio.app/).
- **Gradio’s Status Transparency**: Users can view Gradio's operational uptime statistics over different time frames including the last 24 hours, 7 days, 30 days, and 90 days on their [status page](https://status.gradio.app/#).
- **No Recent Updates**: As of the last 7 days, there have been no new status updates, but the history can be checked for past incidents [here](https://status.gradio.app/#).

**Link mentioned**: <a href="https://status.gradio.app/">Gradio Status</a>: no description found

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1234884877413777478)** (4 messages): 

- **Financial Assistant AI Breakthrough**: A new financial assistant can now **calculate percentage evolution, CAGR, and P/E ratios over unstructured financial reports** without human intervention. Brief insights shared via a post linked in a [tweet](https://t.co/6cTNxUBJcr) about building this powerful tool.

- **Boost RAG Applications with Redis**: In a collaboration between **Redisinc, @tchutch94, and @seldo**, learn about creating agentic Retrieval-Augmented Generation (RAG) with semantic caching. They discuss methods for enhancing quality, efficiency, and cost in [this resource](https://t.co/oGxFrZLMRn).

- **PulumiCorp Webinar on Deploying AI with LlamaIndex**: A webinar scheduled for May 8, hosted by *_ediri* and *@seldo*, will dive into using Pulumi to deploy an AI application, focusing on LlamaIndex, onto AWS. Information about leveraging infrastructure as code for AI applications was shared in the announcement [tweet](https://t.co/4IwBhVFEss).

- **Latest LlamaIndex.TS Update Announced**: **LlamaIndex.TS version 0.3** has been released with enhancements such as agent support for ReAct, Anthropic, OpenAI, and a generic AgentRunner class, improved Web Streams, and a more robust type system. These updates were highlighted in a [tweet](https://t.co/mBIrD9uh8c) featuring the new version’s benefits.

**Link mentioned**: <a href="https://t.co/oGxFrZLMRn">no title found</a>: no description found

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1234764368504164404)** (130 messages🔥🔥): 

- **Max Tokens and Embedding Models**: If the content to embed exceeds the max token limit, the model will only consider the first max_length tokens and ignore the rest. This may require content chunking if the embedding model has a smaller token limit than provided data.

- **Local Async Calls for AzureOpenAI**: LlamaIndex supports async calls to AzureOpenAI using `acomplete` and `astream_complete` for completions, and `achat` and `astream_chat` for chat context. Async allows tasks like API calls to be executed without blocking other operations, leading to performance improvements.

- **Real-time Summaries with Source Nodes**: LlamaIndex can generate summaries and indicate the nodes used to form them. Streamlining this process involves optimizing prompts and utilizing source nodes information for result relevance.

- **Understanding RAG with MongoDB Atlas**: Questions were raised about querying within LlamaIndex without re-uploading documents and converting them into nodes. Responses indicated that embedding models are essential for comparing queries with the indexed data to retrieve relevant material.

- **Analyzing LlamaIndex vs. Local Development Drawbacks**: Ollama runs locally and can be slower compared to server-based APIs like OpenAI, but it offers privacy and cost benefits for local development. The use of embedding models in the query process is unavoidable for creating and querying indices in LlamaIndex.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/a/9uLmSxD">Summary and Resources</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial (OpenAI) - LlamaIndex</a>: no description found</li><li><a href="https://www.cloudraft.io/blog/content-moderation-using-llamaindex-and-llm">Content Moderation using AI</a>: Learn about how to moderate content using AI models and frameworks such as llamaindex, moondream and microsoft phi-3.</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/openai#async>).">OpenAI - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/low_level/vector_store/?h=vectorstorequery">Building a (Very Simple) Vector Store from Scratch - LlamaIndex</a>: no description found</li><li><a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>: Advanced RAG 101 - build agentic RAG with llama3Get free HubSpot report of how AI is redefining startup GTM strategy: https://clickhubspot.com/4hx🔗 Links- F...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/chroma_metadata_filter/?h=metadatafilter">Chroma Vector Store - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline_async#query-pipeline-with-asyncparallel-execution>),">Query Pipeline with Async/Parallel Execution - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/pipeline/query_pipeline_async#try-out-queries>).">Query Pipeline with Async/Parallel Execution - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/ingestion/parallel_execution_ingestion_pipeline#in-summary>),">Parallelizing Ingestion Pipeline - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1235227548582150145)** (6 messages): 

- **Choosing the Right GPU for AI Tasks**: The discussion revolved around the suitability of a gaming card like the **RTX 4080** for running and fine-tuning smaller language models. One member advised that while VRAM is critical, even with 16 or 24GB, one should not expect to fine-tune models larger than 7B with small batch sizes.

- **Local vs Cloud Compute for Privacy Concerns**: The member **tuhe** clarified the need for a local PC stems from dealing with sensitive data and the practicality of having a robust computer for work, rather than cloud solutions like Google Colab which may pose privacy issues.

- **Introduction to Word Loom**: A new open specification called **Word Loom** was shared, designed for managing and exchanging language for AI, focusing on the separation of code from natural language and composability. Feedback is welcomed on the proposed update, which aims to aid the traditional globalization process, the full details of which are available on [GitHub](https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e).

**Link mentioned**: <a href="https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e">Word Loom proposed update</a>: Word Loom proposed update. GitHub Gist: instantly share code, notes, and snippets.

  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1234813018806943758)** (22 messages🔥): 

- **Subreddit Confusion Cleared Up**: A member clarified that there is a subreddit for Mojo at [https://www.reddit.com/r/modular_mojo/](https://www.reddit.com/r/modular_mojo/), but the Mojo community primarily engages on GitHub and Discord.

- **Concurrency Model Speculations**: The community discussed Mojo's potential for adopting concurrency models, with a guess that it won't follow golang-style but may lean towards an [actor model](https://github.com/modularml/mojo/pull/1445#issuecomment-1849117416), and a counterpoint emphasizing the importance of not shipping a massive runtime with the language.

- **Mojo Compiler Insights**: It was shared that Mojo's compiler is handwritten and reuses parts of LLVM, with further explanation available in a [YouTube video](https://youtu.be/SEwTjZvy8vw) titled "2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing."

- **Type Declaration Error in Playground**: An issue was raised regarding an error message when using 'ui64' as a type declaration, with confusion whether custom bitwidth integers like in Zig were supported and highlighting that `Int64` works but `Int128` doesn’t.

- **First Mojo Anniversary Reflections**: Members reflected on the first anniversary of Mojo's launch, highlighting the addition of traits, references, and lifetimes as major achievements that unlocked a lot of the standard library's potential.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/engine/reference/cli/input-data-schema#data-types:~:text=ui64%3A%20unsigned%20integer%20with%20bitwidth%2064.">Input data schema | Modular Docs</a>: The following YAML schema allows you to specify the input shapes required by</li><li><a href="https://github.com/modularml/mojo/pull/1445#issuecomment-1849117416)">Proposal For An Actor System Based On Mojo by reid-spencer · Pull Request #1445 · modularml/mojo</a>: This is currently a work in progress.  There are no code changes, just a proposal written in the proposals section. This was pre-approved by Chris Lattner in a conversation in June 2023. I will kee...</li><li><a href="https://youtu.be/SEwTjZvy8vw)">2023 LLVM Dev Mtg - Mojo 🔥: A system programming language for heterogenous computing</a>: 2023 LLVM Developers&#39; Meetinghttps://llvm.org/devmtg/2023-10------Mojo 🔥: A system programming language for heterogenous computingSpeaker: Abdul Dakkak, Chr...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1235010692721938432)** (4 messages): 

- **Modular Tweets a Mystery**: Modular's latest [tweet](https://twitter.com/Modular/status/1785447336812101796) has been shared, but the content is not specified in the message.

- **Another Modular Update Hits Twitter**: Check out the most recent [update from Modular](https://twitter.com/Modular/status/1785447397189161006) by following the shared link.

- **Modular Shares a Cryptic Message**: A new tweet from [Modular has been posted](https://twitter.com/Modular/status/1785447412376764507); details of the tweet are not described here.

- **Modular Continues to Tease on Twitter**: There's a new [tweet from Modular](https://twitter.com/Modular/status/1785720385889243286) that might be of interest; specifics behind the tweet are not included in the message.
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1234882195034869781)** (58 messages🔥🔥): 

- **Julia's `@time` Macro Wins Hearts**: One member praised Julia's `@time` macro for its ability to show allocations and expressed a desire to see a similar feature in Mojo.
- **Mystery of the 'None' Implementation**: A search for how `None` is implemented in Mojo led to confusion and a discussion linking to GitHub. The inquiry highlighted an error about `None` not implementing the `__is__` and `__isnot__` methods.
- **Praise for Mojo's Syntax**: Mojo's syntax was lauded by a user who, after evaluating various programming languages, found Mojo's syntax almost perfectly aligned with their ideal language syntax.
- **Discussing Pass by Reference in Mojo**: A conversation about using `inout` with structs and the `Reference` type in Mojo clarified that `inout` does pass by reference similar to C++ but is distinct in Mojo. The discussion included code samples and highlighted the ongoing development to make referencing more elegant.
- **Mojo Development Updates and Questions**: Various messages touched upon Mojo's open-source progress, the anticipation for its Windows release, and ensuring Mojo remains user-friendly and understandable without going down the complexity of Rust's lifetime system.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/search?q=repo%3Amodularml%2Fmojo+%22None%22&type=code&p=0)">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://youtu.be/kgUXfDpAmGQ?si=VmrPUT7YLBmzMq8I">C++ as an Optimizing Assembler - a Performance Talk - Levo DeLellis - CppNorth 2023</a>: https://www.cppnorth.ca​---C++ as an Optimizing Assembler - a Performance Talk - Levo DeLellis - CppNorth 2023Are you tired of abstractions, templates and co...</li><li><a href="https://rosettacode.org/wiki/99_Bottles_of_Beer/EsoLang">99 Bottles of Beer/EsoLang</a>: no description found
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1235000164813639711)** (1 messages): 

- **Call for Mojo Contributors**: A member extended an invitation to contribute to Mojo, with suggestions such as allowing negative numbers, implementing a fallback for scalar processing, and exploring fast absolute tolerances from articles linked in the issues. No specific plans were set, leaving room for experimental contributions.
- **Identifying a Missing Mojo Component**: Mojo currently lacks the [PMADDUBSW](https://www.felixcloutier.com/x86/pmaddubsw) instruction, critical for fast SIMD `atol` (ASCII to long integer conversion), prompting workarounds with ~4 SIMD operations. This feature is specific to x86 and not supported on ARM architectures.

**Link mentioned**: <a href="https://www.felixcloutier.com/x86/pmaddubsw">PMADDUBSW
		— Multiply and Add Packed Signed and Unsigned Bytes</a>: no description found

  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1235261267250511923)** (3 messages): 

- **Mojo Lang Sparks Enthusiasm**: A new [YouTube video](https://youtu.be/JRcXUuQYR90) featuring Chris Lattner discusses **Mojo Lang**, a potential high-performance successor to Python that leverages CPU/GPU programming techniques.
- **Podcast Love for Programming Languages**: A member expressed their fondness for the podcast, sharing their excitement about the discussions on programming languages and spreading the content internally.

**Link mentioned**: <a href="https://youtu.be/JRcXUuQYR90)">Mojo Lang - Tomorrow&#39;s High Performance Python? (with Chris Lattner)</a>: Mojo is the latest language from the creator of Swift and LLVM. It’s an attempt to take some of the best techniques from CPU/GPU-level programming and packag...

  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1234776260509569066)** (7 messages): 

- **Call to Form Team-Mojo for 1BRC**: A member suggested forming a **Team-Mojo** to tackle the *One Billion Row Challenge (1brc)* as both a showcase and a tutorial.

- **Performance Optimization in Mojo**: After optimizing string allocations and conversions, a member reported reducing processing time from 8 seconds to 1.3 seconds for 100M records, with the current bottleneck being the hashmap, bringing total time from 18.5 to 12.5 seconds. This implementation is only functional in Mojo nightly and can be found on [GitHub](https://github.com/MoSafi2/1brc-mojo/tree/dev).

- **Enthusiasm for Team-Mojo's Formation**: Members expressed enthusiasm about forming team-mojo, indicating it would be a fun project to undertake.

- **Reference to Benchmarks Game**: There was a suggestion to also consider the [benchmarks game](https://github.com/modularml/mojo/discussions/843#discussioncomment-7045479), a previously uncompleted task by the team.

- **Multi-core Processing Update**: A member proposed a pull request after updating their work to enable multi-core processing, noting a significant performance improvement to now handle 100M records in 3.8 seconds. Another member invited this update for a further review and mentioned their intent to look into the `atol` function based on their experience with `atol-simd`.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/discussions/843#discussioncomment-7045479)">The Mojo is 68,000 times faster than Python type blogs are awesome, but can awesome comparisons be made with other languages too? · modularml/mojo · Discussion #843</a>: Mojo being 35,000 times faster than Python, 68,000 times faster than Python… it’s impressive, amazing, and cool, but to non-Python people and anti-Python who haven’t yet paid attention to Mojo yet ...</li><li><a href="https://github.com/MoSafi2/1brc-mojo/tree/dev">GitHub - MoSafi2/1brc-mojo at dev</a>: One Billion Row Challenge (1brc) in Mojo language. Contribute to MoSafi2/1brc-mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1234763287900721172)** (20 messages🔥): 

- **Order Swap Still Buggy**: A member mentioned that changing the order of something still causes it to break, despite fixing an initial issue.
- **Considering the Future of `bool` in Code**: A detailed viewpoint was expressed on potentially limiting the use of `bool` to size 1, highlighting the importance of retaining `bool` as a primitive in programming and understanding the impact of such a change.
- **SEMANTICS: Could `simd ternary` mimic `select`?**: A member inquired if `simd ternary` might act like `select`, with another noting that even `if` statements' semantics somewhat depend on the concept of being 'boolable.'
- **WANTED: Missing `__source_location()` Function**: Conversations involved confusion about the disappearance of the `__source_location()` function, with a suggestion that it might be replaced by `__call_location()`. This was visible through a [SourceGraph search link](https://sourcegraph.com/search?q=context:global+__source_location()&patternType=keyword&sm=0&filters=%5B%5B%22type%22,%22Code%22,%22type:file%22%5D%5D) and the topic was further discussed, including specific code examples and GitHub documentation [links](https://github.com/modularml/mojo/blob/nightly/stdlib/src/testing/testing.mojo).
- **Function Names in Source Location**: A member questioned the absence of the `function_name` in the `__source_location()` function output, with hints that others also share this concern.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sourcegraph.com/search?q=context:global+__source_location()&patternType=keyword&sm=0&filters=%5B%5B%22type%22,%22Code%22,%22type:file%22%5D%5D">context:global __source_… - Sourcegraph</a>: no description found</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/testing/testing.mojo">mojo/stdlib/src/testing/testing.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1234777701626286171)** (23 messages🔥): 

- **Clarifying Tokenizer Behavior**: Members discussed how including a **Beginning of Sentence (BOS)** token in a chat template affects encoding, noting that `tokenizer.encode("text")` automatically adds BOS, but `tokenizer.apply_chat_template(chat)` needs it specified in the template.
- **Debating the Value of a Study**: A link to a [recent study](https://arxiv.org/abs/2311.16452) was shared, sparking debate over its usefulness. One member praised its prompting strategy of using cosine-similarity embeddings, while another dismissed the study's approach as overly complex for benchmarks.
- **The Practical Struggles with Model Tokens**: Users expressed frustration over the implementation of new papers into practice, specifically the challenge of figuring out tokens for a model, despite the plethora of academic publications.
- **Discussing User Input Masking Strategies**: A technical question surfaced about the best practice for *masking out user inputs* during training: whether to mask just the message or also the instructional tags, and how to ensure proper learning of the format but not user typing styles.
- **Prompting Approaches and Generalist Models**: There was a brief touch on the relevance of complex prompting strategies and whether applying techniques to only generalist models somewhat misses the point when evaluating AI performance on benchmarks.

**Link mentioned**: <a href="https://arxiv.org/abs/2311.16452">Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine</a>: Generalist foundation models such as GPT-4 have displayed surprising capabilities in a wide variety of domains and tasks. Yet, there is a prevalent assumption that they cannot match specialist capabil...

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1234796962575024128)** (2 messages): 

- **Offer for Compute Help in Triage**: A member is extending an offer to assist with **triage/troubleshooting of bugs/issues** by providing compute resources. They emphasize that such help is invaluable to the project and their sanity.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1234845235872403546)** (14 messages🔥): 

- **Phi3 Finetuning Underway**: Some members are currently engaged in finetuning **phi3**. Others seeking to dive into examples or explore this further are advised to search the channel's history for relevant details.
- **Dataset Format Wrangling for ShareGPT Loader**: A member looking to finetune a model shared a JSON dataset example structured for OpenAI’s format, and then received guidance on how to convert it to the **ShareGPT loader format**. They were advised to replace `"messages"` with `"conversations"`, `"role"` with `"from"`, `"content"` with `"value"`, `"user"` with `"human"`, and `"assistant"` with `"gpt"`.
- **Simplified Script for Dataset Conversion**: For adapting the dataset to the required format, a script was provided, which automatically replaces the keys and maps the roles from the input JSON structure to match the **ShareGPT** expected format.
- **Choose the Right LLaMA Model for Finetuning**: In a discussion on finetuning LLaMA models, it was recommended to avoid finetuning the **Meta-LLaMA-3-70B-Instruct** variant as it's already instructed, which could lead to worse performance with a new format. It was also advised for beginners to start with an **8b model** before progressing to more complex 70b variants.
- **FS-DP Compatibility Query for Lora**: A member inquired about using **fsdp** with **lora**, as opposed to **qlora**, after encountering issues where training hangs post-model loading. The suggestion indicates that perhaps only **qlora** might be compatible with their fsdp setup.
- **LLaMA Model's Lengthy Output Concerns**: A user reported their **LLaMA 3 8b instruct** model producing long outputs and sentences when trained on regular human conversations. They pondered if certain tokens like end-of-text or punctuation might require additional training, or if more data and epochs are the key to resolving this issue.

**Link mentioned**: <a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/conversation.html#sharegpt.load_role)">Axolotl - Conversation</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/)** (1 messages): 

gbourdin: add to my bookmarks. Thanks for this !
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1234879220686258296)** (2 messages): 

- **Axolotl Meets dstack**: A tutorial demonstrating how to use **axolotl** with **dstack**, an open-source orchestrator, was shared. It allows fine-tuning AI models on any cloud or a pool of on-premise machines and is available on [GitHub](https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md).
- **Community Approves**: A community member responded positively to the shared tutorial, commenting on its ease of use.

**Link mentioned**: <a href="https://github.com/dstackai/dstack/blob/master/examples/fine-tuning/axolotl/README.md">dstack/examples/fine-tuning/axolotl/README.md at master · dstackai/dstack</a>: An open-source container orchestration engine for running AI workloads in any cloud or data center. https://discord.gg/u8SmfwPpMd - dstackai/dstack

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1234798037612625921)** (51 messages🔥): 

- **Command-r Model Fine-tuning Discussed**: Members explored fine-tuning the command-r model, with suggestions to use `runpod` templates or to manually implement the unsupported formats. One advised consulting an [untested PR on GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547) for adding the **command-r** model to Axolotl.

- **Fine-tuning Clarifications Provided**: It was established that if specific parameters like **sample packing** are not compatible, they are simply ignored during the process. This led to confusion as to why a training task took unexpectedly long.

- **Axolotl Format Capabilities Queried**: There were questions about Axolotl's support for the **phi-3 format** and **GaLore**, with Phorm responding that Axolotl does not support phi-3 but does support GaLore, and details on enabling it can be found in the [Hugging Face documentation](https://github.com/huggingface/transformers/tree/main/docs/source/en/trainer.md).

- **Model Adaptation Features and Functions**: Through the conversation, it was hinted that adapting models in Axolotl can involve custom code adjustments, and familiarizing oneself with the project's resources on GitHub is beneficial for tasks such as enabling or configuring specific features like GaLore.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1547">Feat: Add cohere (commandr) by NanoCode012 · Pull Request #1547 · OpenAccess-AI-Collective/axolotl</a>: Description  Motivation and Context   How has this been tested?    Untested! Screenshots (if appropriate) Types of changes  Social Handles (Optional)</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=83b91c9b-bb5c-4485-894c-0b878d17f7e2)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L77L100)">axolotl/README.md at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=1f87fb72-80ec-4321-b37b-d7574206e8af)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://github.com/huggingface/transformers/tree/main/docs/source/en/trainer.md#L255L385)">transformers/docs/source/en/trainer.md at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=dbfe64c8-e886-4d35-98e6-190287b3cd3c)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1234772710983340033)** (60 messages🔥🔥): 

- **AI Compliance with Terms of Service**: A participant questioned the situation where an individual is using an AI product without agreeing to its terms. This raises issues about user agreements and how they are enforced.
- **Call for a New Transparent AI Leaderboard**: A user expressed the need for a new and more transparent leaderboard for AI models. They advocated for ones that feature **only verifiable open source models** and the ability to filter results by **open weights**.
- **Concerns Over LMSYS's Objectivity and Data Practices**: There were multiple concerns about the objectivity of the **Chatbot Arena** leaderboard managed by LMSYS; discussions touched on **conflicts of interest** and the lack of transparency in handling models' ratings.
- **Inquiries and Sharing on AI Models and Datasets**: Users sought more information about an AI-generated **chess dataset** and shared their thoughts on various models' performances, like **llama3 70b's** capabilities even when quantized to 4-bit.
- **Technical Difficulties and Development Sharing**: Participants shared links to ongoing projects like [magvit2](https://github.com/lucidrains/magvit2-pytorch) and discussed optimization techniques, including when to use GANs for better model reconstruction and **Natten's new fused cuda implementation** for efficiency.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmsys.org/blog/2024-03-01-policy/">LMSYS Chatbot Arena: Live and Community-Driven LLM Evaluation | LMSYS Org</a>: &lt;h2&gt;&lt;a id=&quot;our-mission&quot; class=&quot;anchor&quot; href=&quot;#our-mission&quot; aria-hidden=&quot;true&quot;&gt;&lt;svg aria-hidden=&quot;true&quot; class=&quot;octicon octicon-link&...</li><li><a href="https://xiaoyushi97.github.io/Motion-I2V/">Motion-I2V</a>: no description found</li><li><a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m">lmsys/lmsys-chat-1m · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1234907832730783826)** (25 messages🔥): 

- **Cardiac Ultrasound AI Research Published**: A member announced the publication of their study on cardiac ultrasound **fine-tuning of OpenCLIP**, despite acknowledging several issues with the paper. The research, after enduring an 8-month revision process, is available at [Nature Medicine](https://doi.org/10.1038/s41591-024-02959-y).

- **Challenging StableDiffusion Sustainability**: Discussion touched on a GitHub repository [zer0int/CLIP-fine-tune](https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/github_zer0intclipfinetune_or_sdxl_training_the/), linked to concerns over Reddit closing open API access, which has widespread implications including affecting app developers and blind users.

- **Kolmogorov-Arnold Networks Over MLPs**: A new paper proposes **Kolmogorov-Arnold Networks (KANs)** which outperform Multi-Layer Perceptrons in accuracy and interpretability by utilizing learnable activation functions as splines on edges. The concept has resonated with members, finding the approach to be very promising ([Read the arXiv paper](https://arxiv.org/abs/2404.19756)).

- **VisualFactChecker for Enhanced Captioning**: Another paper introduces **VisualFactChecker (VFC)**, a training-free pipeline that significantly improves captioning for images and 3D objects by incorporating fact-checking, potentially resolving issues like content hallucination. The study details methods that increase fidelity and detail in automatic captioning ([View the arXiv paper](https://arxiv.org/abs/2404.19752)).

- **Request for Chess Dataset Generation Details**: In search of better training data, a member requests for details on the configuration used to generate the **LAION stockfish dataset** to gauge if it would be adequate for training their chess bot or if there would be a need to generate additional datasets.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.19752">Visual Fact Checker: Enabling High-Fidelity Detailed Caption Generation</a>: Existing automatic captioning methods for visual content face challenges such as lack of detail, content hallucination, and poor instruction following. In this work, we propose VisualFactChecker (VFC)...</li><li><a href="https://arxiv.org/abs/2404.19756">KAN: Kolmogorov-Arnold Networks</a>: Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation fun...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1cgyjvt/github_zer0intclipfinetune_or_sdxl_training_the/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://doi.org/10.1038/s41591-024-02959-y">Vision–language foundation model for echocardiogram interpretation - Nature Medicine</a>: A vision&#8211;language foundation model, trained on a dataset of more than 1 million echocardiogram video&#8211;text pairs, is able to assess various cardiac structural and functional parameters desp...
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1234837931349245973)** (70 messages🔥🔥): 

- **Decentralized AI Training by Prime Intellect**: Prime Intellect explores novel decentralized training approaches to keep up with Big Tech's expansion of **GPU clusters**. For an in-depth look, [read their blog post](https://www.primeintellect.ai/blog/our-approach-to-decentralized-training) discussing the challenges faced by the open-source AI community and their platform's aim to aggregate global compute resources.

- **AI Agents or Translation Machines?**: A member debated the concept of AI agents, suggesting instead that language models could be considered "translation machines" using shared context and memory, without needing to parallelize for multiple reasons.

- **Starcoder2-Instruct Released**: Hugging Face introduces StarCoder2-15B-Instruct-v0.1, a self-aligned **Large Language Model (LLM) for code generation**. The underlying pipeline and the model are open-source and permissive, detailed in their [announcement page](https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1).

- **AI Town with World Editor**: User shares an [experimental set-up](https://x.com/cocktailpeanut/status/1785702250599371088?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) involving 300 AI agents operating within a simulated world called AI Town, running smoothly on a MacBook M1 Max.

- **Lilian Weng's Insightful Yet Challenging Blog Posts**: Some members expressed feeling overwhelmed by the depth and complexity of Lilian Weng's blog posts, particularly the Transformer Family 2.0 post, questioning if they need to dedicate full-time learning to grasp the concepts shared.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://learnprompting.org/docs/intro">Learn Prompting: Your Guide to Communicating with AI</a>: Learn Prompting is the largest and most comprehensive course in prompt engineering available on the internet, with over 60 content modules, translated into 9 languages, and a thriving community.</li><li><a href="https://x.com/cocktailpeanut/status/1785702250599371088?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from cocktail peanut (@cocktailpeanut)</a>: I deployed 300 AI agents to Westworld (aka AI Town), and surprisingly, it works without issues on my Macbook M1 Max 64G. Here&#39;s what it looks like:</li><li><a href="https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1">bigcode/starcoder2-15b-instruct-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://www.latent.space/p/aie-2023-workshops">AI Engineering 101 and 201 Workshops</a>: from AI Engineer Summit 2023</li><li><a href="https://x.com/lmsysorg/status/1785078213712208291">Tweet from lmsys.org (@lmsysorg)</a>: hi @simonw, thanks a ton! We really value your feedback.  Just to clarify, following our policy, we&#39;ve partnered with several model developers to bring their new models to our platform for communi...</li><li><a href="https://www.primeintellect.ai/blog/our-approach-to-decentralized-training">State-of-the-art in Decentralized Training</a>: This post explores various novel decentralized training approaches and how they can enable effective AI model training across globally distributed GPUs.</li><li><a href="https://x.com/jessechenglyu/status/1785342519045394465?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Jesse Lyu (@jessechenglyu)</a>: get your r1 update to the latest version now - we addressed most of the issues we found so far and more fix/improvements incoming! idle battery life up to 5x better now.  ↘️ Quoting rabbit inc. (@rabb...</li><li><a href="https://youtu.be/aircAruvnKk?feature=shared),">But what is a neural network? | Chapter 1, Deep learning</a>: What are the neurons, why are there layers, and what is the math underlying it?Help fund future projects: https://www.patreon.com/3blue1brownWritten/interact...</li><li><a href="https://roadmap.sh/prompt-engineering">Prompt Engineering Roadmap - roadmap.sh</a>: Step by step guide to learn Prompt Engineering. We also have resources and short descriptions attached to the roadmap items so you can get everything you want to learn in one place.
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1235120390624383038)** (1 messages): 

- **Ring Attention Paper Club Event**: A special guest appearance at the **LLM Paper Club** with the StrongCompute team to discuss the important *Ring Attention* paper. Interested parties can sign up for the event through this [Zoom link](https://lu.ma/oz8e9z3r). ![Cover Image for LLM Paper Club (Ring Attention!)](https://images.lumacdn.com/cdn-cgi/image/format=auto,fit=cover,dpr=2,quality=75,width=400,height=400/event-covers/mq/b7a9e5d5-cbd9-4546-a668-972d498d2186)

**Link mentioned**: <a href="https://lu.ma/oz8e9z3r">LLM Paper Club (Ring Attention!) · Zoom · Luma</a>: The StrongCompute gang (@adam_peaston, @fennecs) is covering Ring Attention today! https://arxiv.org/abs/2310.01889 Also submit and vote for our next paper:…

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1235305008581312656)** (2 messages): 

- **Zoom Meeting Link Shared**: A Zoom meeting link was provided for those preferring a video call alternative. The link can be accessed at [Zoom Meeting](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09).

**Link mentioned**: <a href="https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1234866310773735454)** (36 messages🔥): 

- **Promoting Positive Community Interactions**: A reminder was issued emphasizing the importance of being respectful and constructive as the community grows and diversifies. It was stressed that *everyone has an equal right to share their thoughts* and should be treated well to build a better future.
- **Event Reminder and Recap Inquiry**: A link to a community event was shared, and members who missed it asked for a recap. It was mentioned that the slides and a screen recording would be made available, with posted slides in a specific channel.
- **Open Interpreter's Web Task Capabilities**: Members discussed whether Open Interpreter can perform browser tasks like visiting websites and scraping data. Clarification was provided that it is indeed capable of such tasks without needing browser control.
- **Compatibility and Technical Issues Discussed**: Questions surfaced about the compatibility of Open Interpreter's OS mode with Windows, mentioning persistent errors. A member confirmed that some commands need alterations for Windows, and the package 'tesseract' was mentioned as a cause of issues.
- **Sharing Useful Resources**: A YouTube channel was recommended as a useful resource for insights and updates related to Open Interpreter, complete with a direct link to the channel.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/SdwpMQaW?event=1232436050165764096">Join the Open Interpreter Discord Server!</a>: A new way to use computers | 8840 members</li><li><a href="https://discord.gg/9rjF24Gz?event=1228030976706220072">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://youtube.com/@MikeBirdTech?feature=shared">Mike Bird</a>: A.I. engineering  
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1234781691109703732)** (31 messages🔥): 

- **The Quest for the External Push Button**: Members discussed issues with integrating an external push button with hardware, specifically with the **Ataom Echo** device. Code modifications were shared, specifically a snippet for **ButtonChecker**, which when utilized, resolved the problem as confirmed by a member who implemented it.

- **Amplifying Audio Through External Hardware**: A member provided a solution to **increase the volume of speakers** connected to hardware, suggesting the use of an external amplifier with a [link to a potential amp](https://www.amazon.com/dp/B01DKAI51M), though noting they had not yet tested this setup.

- **Unboxing AI Innovations**: The channel mentioned a **YouTube review by MKBHD** of an AI product, Rabbit R1, with a [link to the video](https://www.youtube.com/watch?v=ddTV12hErTc&ab_channel=MarquesBrownlee). There was a debate about the effectiveness of traditional tech reviewers in understanding and evaluating non-mainstream AI devices.

- **Connecting R1 to OpenInterpreter**: Conversations circled around the idea of integrating **R1 with OpenInterpreter (OI)**, with members discussing their anticipation and plans for doing so. There's an eagerness to explore how these tools can work together, hoping to expand capabilities and build innovative setups.

- **NGROK Domain Customization for OI**: A member shared specific steps to creating a new domain on **ngrok** and editing the **tunnel.py** file within the 01 software to address issues with server connection, offering a [direct link to the ngrok domains page](https://dashboard.ngrok.com/cloud-edge/domains).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dashboard.ngrok.com/cloud-edge/domains">ngrok - Online in One Line</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=ddTV12hErTc&ab_channel=MarquesBrownlee">Rabbit R1: Barely Reviewable</a>: AI in a Box. But a different box.Get a dbrand skin and screen protector at https://dbrand.com/rabbitMKBHD Merch: http://shop.MKBHD.comTech I&#39;m using right no...</li><li><a href="https://www.amazon.com/dp/B01DKAI51M">Amazon.com: HiLetgo Mini 3W+3W DC 5V Audio Amplifier Handy Digital Power Amp Module Board Dual-Channel PAM8403 Stereo Amplifiers with Potentiometer for DIY Portable : Electronics</a>: no description found
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1235358580249591909)** (2 messages): 

```html
<ul>
    <li><strong>Snowflake Arctic 480B and FireLLaVA 13B Models Launched</strong>: Announcing new models <strong>Snowflake Arctic 480B</strong>, excellent at coding with a hybrid transformer architecture, available at <a href="https://openrouter.ai/models/snowflake/snowflake-arctic-instruct">Snowflake Arctic 480B</a> and <strong>FireLLaVA 13B</strong>, an open source, multimodal model by Fireworks at <a href="https://openrouter.ai/models/fireworks/firellava-13b">FireLLaVA 13B</a>. Both come with new pricing and detailed specifications for developers.</li>
    <li><strong>Improved Load Balancing and Detailed Provider Stats</strong>: OpenRouter introduced <strong>load balancing</strong> to manage providers' load surges and now allows monitoring of latency and providers' finish reasons, enhancing performance for users, accessible on the <a href="https://openrouter.ai/activity">Activity page</a>.</li>
    <li><strong>Streamlined Docs for Developers</strong>: Documentation updates for Image and multimodal requests, plus tool calls and function calling, are now available to guide usage on <a href="https://openrouter.ai/docs#images-_-multimodal-requests">Image Requests</a> and <a href="https://openrouter.ai/docs#tool-calls">Tool Calls</a>.</li>
    <li><strong>Feature Expansion and Price Adjustments</strong>: Announced support for <strong>logit_bias</strong> and <strong>min_p</strong> on Lepton models, a significant 40% price cut on Mythomax Extended, and a slight 4% reduction for Mixtral 8x7b Instruct. These changes reflect OpenRouter's commitment to cost-effective and advanced AI capabilities.</li>
    <li><strong>Impending API Changes and Developer Notifications</strong>: Developers are alerted about the upcoming removal of the <code>total_cost</code> field from non-streaming completions and a potential requirement of the <code>User-Agent</code> header in requests to improve service security and efficiency.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://omnigpt.co/">OmniGPT -  The Most Affordable ChatGPT Alternative</a>: We offer you the best models in the market: Claude 3, GPT 4 Turbo, GPT 4, Gemini, Perplexity and more at an affordable price.</li><li><a href="https://syrax.ai/">Syrax AI - Leverage multiple AIs on one platform</a>: With Syrax AI you can access multiple AI models to generate content, images, and more from one platform.</li><li><a href="https://openrouter.ai/models/snowflake/snowflake-arctic-instruct">Snowflake: Arctic Instruct by snowflake | OpenRouter</a>: Arctic is a dense-MoE Hybrid transformer architecture pre-trained from scratch by the Snowflake AI Research Team. Arctic combines a 10B dense transformer model with a residual 128x3.66B MoE MLP result...</li><li><a href="https://openrouter.ai/models/fireworks/firellava-13b">FireLLaVA 13B by fireworks | OpenRouter</a>: The first commercially permissive OSS LLaVA model.  This vision-language model was trained entirely on OSS LLM generated instruction following data.</li><li><a href="https://openrouter.ai/docs#images-_-multimodal-requests">OpenRouter</a>: Build model-agnostic AI apps</li><li><a href="https://openrouter.ai/docs#tool-calls">OpenRouter</a>: Build model-agnostic AI apps
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1235131318954623038)** (1 messages): 

- **Skribler - The Swedish Author's AI Assistant**: Launched a few weeks back, **Skribler** is a new tool aimed at Swedish writers, integrating various models via OpenRouter for different writing tasks. It's available at [skribler.se](https://skribler.se) and offers features like generating suggestions for text passages, helping bridge gaps in writing, formulating dialogues, and overall support for the creative writing process, with an introduction video [here](https://youtu.be/2Q2hb6UqGo4).
- **Positive Reception and User Adoption**: The announcement of **Skribler** also notes that it has already secured a group of paying users, indicating a positive reception in its target market.

**Link mentioned**: <a href="https://skribler.se">Skribler | Skriv med AI</a>: no description found

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1234817884748775435)** (64 messages🔥🔥): 

- **OpenRouter Logging Queries**: Members are asking if it's possible to view the per-request prompt and outputs with logging enabled on OpenRouter.
- **Model Embedding Capability Inquiry**: A member inquired about the availability of models that support embedding within OpenRouter.
- **Context Extension Curiosity**: There's a conversation about the extension of context windows in models, specifically mentioning a model with a context length extended to over 1million and discussions regarding the performance of an extended LLama-3 8B model, available on [Hugging Face](https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k).
- **Payment Issues and Solutions Discussed**: Users are discussing issues with using pre-paid credit cards on OpenRouter, suggesting that some cards may be blocked by Stripe's fraud detection, and talking about potential solutions or alternatives for payment.
- **Stream Cancellation and Model Fall-backs**: There are questions concerning the reliability of stream cancellation in OpenRouter, and suggestions for using AWS as a potential fallback for Claude models, similar to how Azure is used for OpenAi's models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/gradientai/Llama-3-8B-Instruct-Gradient-1048k">gradientai/Llama-3-8B-Instruct-Gradient-1048k · Hugging Face</a>: no description found</li><li><a href="https://github.com/hsiehjackson/RULER">GitHub - hsiehjackson/RULER: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models?</a>: This repo contains the source code for RULER: What’s the Real Context Size of Your Long-Context Language Models? - hsiehjackson/RULER
</li>
</ul>

</div>
  

---



**AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1235256845418106990)** (28 messages🔥): 

- **Crisp Diffusion Model Outputs**: A member mentioned that **the diffusion model outputs** from [Hexagen World](https://www.hexagen.world/) are really crisp, signaling high-quality results.

- **Retro Gaming with Generative AI**: It was suggested that remaking early social media games like **Farmville** with Generative AI (GenAI) would be a compelling concept and WebSim could potentially be the best platform to achieve this.

- **AI Embedded Nostalgic Townsim**: A member expressed interest in setting up a 1950s themed AI town in WebSim where one of the characters is a communist spy, creating an interactive game of **cat-and-mouse**.

- **Interactive Animation and AI Discussions**: Participants interested in **AI animation** were invited to join a related Discord community by following a provided [Discord invite link](https://discord.gg/deforum).

- **Discovery and Sharing of Hexagen World**: The interactive AI concept **Hexagen World** was shared within the community, discovered via a Twitter [post by @bennyj504](https://x.com/bennyj504/status/1785664502903570568), capturing the interest of several members who discussed its features and potential.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/deforum">Join the Deforum Discord Server!</a>: Deforum is an open-source animation tool that leverages the power of Stable Diffusion to create AI animations. | 29464 members</li><li><a href="https://x.com/bennyj504/status/1785664502903570568">Tweet from BennyJ504-075⚜😎🤑🔌.yat 🟣 (@bennyj504)</a>: https://www.hexagen.world/</li><li><a href="https://www.hexagen.world/">Collectively AI Generated Game World</a>:  social experiment where anyone can help create an infinitely unique world in their browser.
</li>
</ul>

</div>
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1235075486200107029)** (2 messages): 

- **First-time experience with Llama3**: A member expressed excitement about trying out **Llama3** for the first time, indicating a new user's interest in exploring the capabilities of this AI model.
  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1234844604638167094)** (33 messages🔥): 

- **Simple Local Setup Success**: A member confirmed that setting up the system locally was very easy to accomplish.
- **Windows Compatibility Hurdle**: Several members reported issues running the local version on Windows, with one getting stuck at *Checking for index or schema changes...* Another member clarified that **Convex local does not support Windows** but mentioned that work on Windows compatibility was underway.
- **Mac-Specific Run Commands Shared**: For those running on Mac, it was suggested to use `just convex dev` for a dedicated sync and `just convex logs` for a separate terminal log output, offering smooth operations without interferences from `npm run dev`.
- **Correct Node Version is Crucial**: An error related to the **node version** was shared by a member when trying to run the app. It was pointed out that one needs to run `convex-local-backend` in the same directory as `npm run dev`, and to ensure that the correct node version (`nvm use 19`) is used in both directories.
- **Switching to Linux for Development**: In light of the aforementioned compatibility issues with Windows, some members considered uninstalling Windows and installing Linux, with one inquiring about how to do so and if it would affect the ability to play the game Stellaris. Another member provided a [link to WineHQ](https://appdb.winehq.org/objectManager.php?sClass=application&iId=17537) indicating that Stellaris has native Mac and Linux versions, implying compatibility would not be an issue.

**Link mentioned**: <a href="https://appdb.winehq.org/objectManager.php?sClass=application&iId=17537">WineHQ  - Stellaris</a>: no description found

  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1234768624204648578)** (35 messages🔥): 

- **Language Models and Grammar**: A link to LLM University offers an explanation on how language models like LLMs manage to generate grammatically correct sentences. It talks about the concept of word and sentence embeddings, and the crucial role of self-attention, with a detailed resource available [here](https://docs.cohere.com/docs/the-attention-mechanism).
- **Command R Gets Rave Reviews**: Community members praise the Cohere commandR/ R+ models, lauding their high performance and contrasting them to other large language models, with comments suggesting that they offer an enterprise-level polished experience.
- **RAG-powered AI Legal Assistant Webinar**: The recording of a webinar about building an AI legal assistant with Cohere's RAG is shared and available on [YouTube](https://www.youtube.com/watch?v=KfqJsqIFeRY&ab_channel=Cohere).
- **Azure and OAuth for Connectors Discussed**: For those wondering how to set up OAuth with connectors on Azure, it is clarified that the Cohere toolkit on GitHub can be used which allows everything to run on Azure, ensuring all data remains internal with no external data sharing.
- **Exploring Multilingual Support in Command-R**: The community is actively testing languages like Norwegian on Command-R, leading to discussions about language support and the need for better benchmarks, even though some languages appear to work well without official support.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/the-attention-mechanism">The Attention Mechanism</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=KfqJsqIFeRY&ab_channel=Cohere">Building a RAG-powered AI legal assistant with Cohere</a>: Cohere recently released Command R, its family of highly scalable language models that balance high performance with strong accuracy. In this webinar, you’ll...</li><li><a href="https://github.com/cohere-ai/cohere-toolkit/?tab=readme-ov-file#how-to-add-a-connector-to-the-toolkit">GitHub - cohere-ai/cohere-toolkit: Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications.</a>: Toolkit is a collection of prebuilt components enabling users to quickly build and deploy RAG applications. - cohere-ai/cohere-toolkit
</li>
</ul>

</div>
  

---


**Cohere ▷ #[collab-opps](https://discord.com/channels/954421988141711382/1218409745380147320/1235223324804775957)** (1 messages): 

There are no sufficient details or discussion points provided in the single message history given to create a summary. If more chat contents were provided, a summary could be created following the guidelines.
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1234773013615087666)** (24 messages🔥): 

- **Seeking PDF Table Extraction Help**: A member inquired about how to **improve table extraction** from PDFs, especially when they span multiple pages. They are using *unstructure* but experiencing poor results.

- **Integrating Llama 3 with LangChain**: A member asked how to use **Llama 3** through LangChain and was pointed to use [Fireworks](https://python.langchain.com/docs/integrations/chat/fireworks/) with **Fireworks API Key** to achieve this.

- **Looking for Document-to-Graph Conversion Tools**: Members discussed the need for tools to **automatically structure documents into knowledge graphs**. Suggestions included using a layout parser like *unstructured* or *Azure Doc AI* and exploring [LangChain's documentation](https://python.langchain.com/docs/use_cases/graph/constructing/) on constructing knowledge graphs.

- **Exploring Sales Agents with AI**: A member is seeking advice on building **AI-powered Sales Agents** that can handle objections and maintain a human tone. They mentioned experimenting with SalesGPT logic and are open to partnerships to further this initiative.

- **Addressing AI Schema Knowledge Limitations**: In a server with over 2000 tables, a member is facing challenges with an AI's ability to comprehend all the schemas, indicating **limitations in AI knowledge** about database structures.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/use_cases/graph/constructing/">Constructing knowledge graphs | 🦜️🔗 LangChain</a>: In this guide we’ll go over the basic ways of constructing a knowledge</li><li><a href="https://python.langchain.com/docs/integrations/chat/fireworks/">ChatFireworks | 🦜️🔗 LangChain</a>: Fireworks accelerates product development</li><li><a href="https://fireworks.ai/">Fireworks - Generative AI For Product Innovation!</a>: Use state-of-the-art, open-source LLMs and image models at blazing fast speed, or fine-tune and deploy your own at no additional cost with Fireworks.ai!
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1234899074763722844)** (1 messages): 

- **Google Drive Libraries in Use Again**: A member mentioned the necessity to use Google Drive libraries for certain operations, specifying that the **drive key** should be set as an environment variable. It was noted that these libraries were previously removed and then re-added to the project.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1234773357178916916)** (7 messages): 

- **Launch of QuickVid for YouTube Video Summarization**: QuickVid introduces a new way to interact with YouTube content by providing **lightning-fast summaries** and fact verification. Experience the tool that can **improve your YouTube experience** at [QuickVid](https://quickvid.vercel.app/).

- **Advanced Webloader RAG Creation Explained**: A member shares an article on building powerful **Webloader RAG applications** with Groq, Langchain, and Datastax. Details can be found at this [Medium post](https://medium.com/ai-advances/building-powerful-webloader-rag-applications-with-groq-langchain-and-datastax-f4816d88bee8).

- **Introduction of Word Loom Spec for AI Language Management**: Word Loom, an open spec for managing language for AI, aims to improve prompt management with core principles of separation of code from natural language, composability, and friendliness to mechanical comparisons and G11N techniques. Feedback on the spec is welcome, and it can be reviewed on [GitHub Gist](https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e).

- **Updates to LangChain Chatbot and Documentation Challenges**: The LangChain chatbot has been updated to version **0.1.17**, with acknowledgement of the challenges posed by outdated documentation post-stable release. A working example of the updated chatbot can be experienced at [LangChain Chatbot](https://langchain-chatbot.streamlit.app).

- **Consideration of LLM Performance Report for Content Creation**: A member is testing various **LLMs** on the leaderboard for content creation use cases like scriptwriting and copywriting, and asks if a detailed report would be useful to others.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/uogbuji/5bd08f74125934fa9e0d37236a8e168e">Word Loom proposed update</a>: Word Loom proposed update. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/carlosplanchon/gpt_pydantic_tools">GitHub - carlosplanchon/gpt_pydantic_tools: A way to write GPT tools using Pydantic Schemas.</a>: A way to write GPT tools using Pydantic Schemas. Contribute to carlosplanchon/gpt_pydantic_tools development by creating an account on GitHub.</li><li><a href="https://quickvid.vercel.app/">QuickVid</a>: no description found</li><li><a href="https://langchain-chatbot.streamlit.app">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1234782249166049310)** (3 messages): 

- **A Parisian Flavor to Advanced RAG**: A new tutorial video showcases the integration of **LangChain** with **Mistral Large** and **Llamaindex** to build an Advanced RAG assistant for the French-speaking community. The content is available on YouTube as "[Multi-Agent RAG: LangChain et LlamaIndex portés par Mistral Large - Le vent du changement](https://youtu.be/ol2QMp64lgo)" with the application's code provided in the video description.

- **Training Local Llama3 with a Twist**: An instructional video titled "*I want Llama3 to perform 10x with my private knowledge* - Local Agentic RAG w/ llama3" has been shared, illustrating how to train **llama3** with private knowledge to build an agentic RAG. The video can be found [here](https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P).

- **Complexity-based RAG Strategy Selection**: The "[LangGraph + Adaptive Rag + LLama3 Python Project: Easy AI/Chat for your Docs](https://www.youtube.com/watch?v=QnXdlqEUW80)" video introduces an Adaptive RAG approach that adjusts its strategy according to the complexity of the query. This technique promises to optimize the performance of AI/Chat integrations with documentation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/u5Vcrwpzoz8?si=U30s6BAN9Jsaec-P">&quot;I want Llama3 to perform 10x with my private knowledge&quot; - Local Agentic RAG w/ llama3</a>: Advanced RAG 101 - build agentic RAG with llama3Get free HubSpot report of how AI is redefining startup GTM strategy: https://clickhubspot.com/4hx🔗 Links- F...</li><li><a href="https://www.youtube.com/watch?v=QnXdlqEUW80">LangGraph + Adaptive Rag + LLama3 Python Project: Easy AI/Chat for your Docs</a>: #langchain #langgraph #rag #python #automation #llm #ai #automation in this video, I have a super quick tutorial for you showing how to create a fully local ...</li><li><a href="https://youtu.be/ol2QMp64lgo">Multi-Agent RAG: LangChain et LlamaIndex portés par Mistral Large - Le vent du changement</a>: Dans cette nouvelle vidéo, je passe Mistral Large au banc d&#39;essai pour le développement d&#39;un Assistant RAG multi-agents en utilisant LangChain et LlamaIndex....
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1234890920575631360)** (1 messages): 

```html
<ul>
  <li><strong>Join the Mozilla AI Team</strong>: Mozilla AI is expanding its team and is currently hiring. Interested parties can check out the employment opportunities on their official Discord channel [here](https://discord.com/channels/1089876418936180786/1230938514955436242/1234870020916510823).</li>
  <li><strong>Introducing Lm-buddy</strong>: Mozilla AI has released a new open-source tool named **Lm-buddy** designed to help evaluate models more efficiently. For more details and access, visit the announcement in their channel [here](https://discord.com/channels/1089876418936180786/1230938514955436242/1234589599733518378).</li>
  <li><strong>Local LLM as Digital Jurist</strong>: There's a discussion about using a **Local LLM** as a judge via the Prometheus framework. Details are available on the Discord channel, accessible [here](https://discord.com/channels/1089876418936180786/1234890301143912599/1234890301143912599).</li>
</ul>
```
  

---


**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1234906250358296607)** (34 messages🔥): 

- **M1 MacBook Air Trouble with LLaMA3**: A member reported issues running **LLaMA3:8b** on an M1 MacBook Air, where it works fine on ollama but not on llamafile. The response was that testing on M1 will be made a priority after resolving other ongoing support issues.
- **Whisper Models Wrapped in Llamafile**: A suggestion was made to wrap **whisper.cpp models** into llamafile for faster inference, noting that integration for microphone and speaker remains unsolved, despite ease of building whisper with cosmo libc.
- **Justine Tunney's GEMM Blog Fact-Check**: One user queried about a blog post (https://justine.lol/matmul/) stating **np.matmul** performs at 29 gflops, noting personal experience with much higher gflop performance; a response clarified the original measurement was on an **Intel computer with Ubuntu** and explained the difference in counting flops.
- **Multiple Llamafiles Running Simultaneously**: A discussion about running multiple llamafiles simultaneously with different models was confirmed to be possible. It was noted that the operating system would manage the resource allocation, and there may be a need for extra tooling for optimized use.
- **Llamafile Public Path Customization**: A member asked about customization using the `--server --path PUBLIC_PATH` option. It was mentioned that the only tested customizability involved replacing .html and .js files in the zip, rather than external directories.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/jartine/Phi-3-mini-4k-instruct-llamafile#prompting">jartine/Phi-3-mini-4k-instruct-llamafile · Hugging Face</a>: no description found</li><li><a href="https://github.com/stanford-futuredata/FrugalGPT">GitHub - stanford-futuredata/FrugalGPT: FrugalGPT: better quality and lower cost for LLM applications</a>: FrugalGPT: better quality and lower cost for LLM applications - stanford-futuredata/FrugalGPT
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1234900403498258542)** (8 messages🔥): 

- **Curiosity about Graph Diagrams for Backward Operations**: Shikhar_7985 inquired about creating graph diagrams for issue **#3572** involving backward passes with two reduce operations. Akshatxv mentioned that there’s a dot file that can be used, while python273 hinted at setting `GRAPH=1`.

- **Symbolic Shapes and Skipped Tests in Tinygrad**: Georgehotz brought to attention his work on symbolic shapes in Tinygrad and shared a [pull request](https://github.com/tinygrad/tinygrad/pull/4362) that includes a skipped test for symbolic arange.

- **Seeking Tinygrad Knowledge Beyond Google**: Lynn4400 expressed interest in learning more about Tinygrad, especially its kernels, and mentioned being influenced by a podcast by Lex Fridman. Leikowo directed them to the repo's documentation as a good starting point for understanding Tinygrad better.

**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/pull/4362">tensor variable by geohot · Pull Request #4362 · tinygrad/tinygrad</a>: no description found

  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1234795444773715979)** (13 messages🔥): 

- **Tinygrad's Scalar to ConstType Renaming**: The project saw a [commit](https://github.com/tinygrad/tinygrad/commit/77589bc7a5430ee470621e43fb1817259d3ce0f5) renaming `Scalar` to `ConstType` and `cast_scalar` to `as_const` for *pre-req cleanup* to standardize constant argument types with dtype.

- **Exploring Const Support Variables**: A member suggested refining tinygrad's handling of constants in operations, proposing to use const support variables instead of tensor variables for simplification and asserting the bounds during the scheduling phase.

- **Symbolic JIT and Variable Mean Tests**: After a discussion on the need for symbolic JIT enhancements, it was noted that a good test for verifying improvements would involve varying symbolic JIT variable values and calculating the mean of a 2D tensor with variable lengths.

- **Emphasis on Making Const Variable Work**: There was a focus on enabling the functioning of const Variables within tinygrad, as they are pivotal for operations related to symbolic dimensions and operations.

- **EfficientNet CUDA Usage on Nvidia Xavier**: Members discussed issues with running the efficientnet example on Nvidia Xavier, suggesting checking the use of `CUDA=1` for proper script execution.

- **Technical Divisions in Symbolic Logic**: A debate occurred regarding the differentiation between Rednode and OpNode in the tinygrad codebase, questioning if Rednode complicates symbolic compiler logic and whether it should be factored out.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/compare/master...davidjanoskyrepo:tinygrad:symbolic-mean-var-pull">Comparing tinygrad:master...davidjanoskyrepo:symbolic-mean-var-pull · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Comparing tinygrad:master...davidjanoskyrepo:symbolic-mean-var-pull · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/compare/86d90511cee2%5E...97a2d44d9840">Comparing 86d90511cee2^...97a2d44d9840 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Comparing 86d90511cee2^...97a2d44d9840 · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/commit/77589bc7a5430ee470621e43fb1817259d3ce0f5">rename Scalar to ConstType and cast_scalar to as_const (#3946) · tinygrad/tinygrad@77589bc</a>: prereq cleanup to make const arg same python type as dtype
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1235293456511799328)** (11 messages🔥): 

- **Claude Released by Anthropic**: Anthropic has officially released the Claude app, and some members have begun downloading it for use.
- **Quality Queries on Claude**: Members are curious about how the newly minted Claude app by Anthropic compares to OpenAI's offerings, questioning if it holds up in quality.
- **Smooth Sailing with New App**: One of the members did not report any issues when using the Claude app and expressed an affinity toward Anthropic's branding.
- **Anthropic's Branding Wins Hearts**: The conversation reflects a positive response to Anthropic's branding strategies, with members acknowledging the appeal of its logo.
- **ML Collective Meetings Ongoing**: A member confirmed they still attend ML Collective meetings, though not on a weekly basis.
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1234876113021440090)** (1 messages): 

- **Rethinking AI Leaderboards**: A shared article titled ["AI Leaderboards are No Longer Useful"](https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful) by Sayash Kapoor, Benedikt Stroebl, and Arvind Narayanan questions the usefulness of current AI leaderboards. According to [HumanEval benchmarks](https://paperswithcode.com/sota/code-generation-on-humaneval), **LDB** is the most accurate publicly available system for code generation, but its high cost due to repeatedly invoking language models like GPT-4 is a significant drawback.

**Link mentioned**: <a href="https://www.aisnakeoil.com/p/ai-leaderboards-are-no-longer-useful">AI leaderboards are no longer useful. It&#x27;s time to switch to Pareto curves.</a>: What spending $2,000 can tell us about evaluating AI agents

  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1235253560917233685)** (2 messages): 

- **Motivation Boost Successful**: In response to a blunt performance critique, a member has notably elevated their work quality, eliciting a positive and emphatic reaction from others.
  

---



**Alignment Lab AI ▷ #[ai-and-ml-discussion](https://discord.com/channels/1087862276448595968/1087876677603958804/1234767428035022920)** (1 messages): 

- **Inappropriate Content Alert**: The channel received a message promoting a Discord invite link allegedly offering access to leaked materials of questionable and potentially illegal ethics involving minors. The message includes emojis suggestive of adult content and targets everyone in the channel.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[programming-help](https://discord.com/channels/1087862276448595968/1087876753462136873/1234767505835425803)** (1 messages): 

- **Inappropriate Content Alert**: A message in the channel contained an offer for free "18+ Teen Girls and onlyfans leaks" and included a Discord invite link. This content is inappropriate for the channel focused on AI alignment and programming help.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1234767644352315433)** (1 messages): 

- **Inappropriate Content Alert**: A message was posted offering free leaks of **18+ Teen Girls and OnlyFans content**, including a Discord invite link. This content is against community guidelines and promotes illegal activities.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1234767675062747157)** (1 messages): 

- **Inappropriate Content Alert**: The channel contained a message promoting adult content including **18+ teen girls** and **OnlyFans leaks**. The message included emojis and a Discord invite link.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[landmark-dev](https://discord.com/channels/1087862276448595968/1113327574563692654/1234767716267855884)** (1 messages): 

- **Inappropriate Content Alert**: A message containing links to adult content and leaked material from OnlyFans was posted, appearing to be spam or a phishing attempt. This included an invitation to a Discord channel allegedly offering free access to such content.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**Alignment Lab AI ▷ #[landmark-evaluation](https://discord.com/channels/1087862276448595968/1118282868595109918/1234767861927645225)** (1 messages): 

- **Inappropriate Content Alert**: A message was posted containing links to NSFW content, specifically promoting **18+ Teen Girls** and **OnlyFans leaks**. The poster shared a Discord invitation link and tagged everyone.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1234767970668908585)** (1 messages): 

- **Inappropriate Content Alert**: A message containing links to potentially explicit content and an invitation to view **onlyfans** leaks was posted, suggesting the sharing of illegal content targeted at an 18+ audience. The post included emojis and a Discord invite link.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[leaderboard](https://discord.com/channels/1087862276448595968/1135102537817653308/1234768131247964212)** (1 messages): 

- **Inappropriate Content Alert**: A message was posted containing links to explicit content, specifically referencing a Discord server with leaks from the subscription service known as OnlyFans, potentially featuring underage individuals. The message included a Discord invite link and used emojis that imply the content is adult in nature.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/1234768231554879488)** (1 messages): 

- **Inappropriate Content Alert**: A message contained an inappropriate solicitation for adult content featuring individuals portrayed as minors, including a Discord invite link. The message was flagged for promoting objectionable material.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[looking-for-work](https://discord.com/channels/1087862276448595968/1142242683339944027/1234768257148391435)** (1 messages): 

- **Inappropriate Content Alert**: A message in the channel contained an offer for adult content featuring young individuals, along with a Discord invite link. This kind of content is highly inappropriate and may violate various terms of service and laws related to the distribution of explicit material of underage subjects.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/1234768325972856912)** (1 messages): 

- **Inappropriate Content Alert**: A message promoting **adult content**, specifically involving *teen girls* and *OnlyFans leaks*, was posted along with a Discord invite link. The post seems to be an attempt to drive traffic to another Discord server that may contain explicit material.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[fasteval-dev](https://discord.com/channels/1087862276448595968/1147528620936548363/1234768398429458506)** (1 messages): 

No summary can be provided as the content does not contain relevant topics or discussion points related to AI or the Alignment Lab AI Discord chatbot messages. Further, the content appears to be inappropriate and not aligned with the expected academic or professional discussions typically summarized.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---


**Alignment Lab AI ▷ #[qa](https://discord.com/channels/1087862276448595968/1147528698669584424/1234768427663495248)** (1 messages): 

- **Inappropriate Content Alert**: A message was posted that appears to promote access to adult content featuring individuals who may be under the age of consent, along with a link to a Discord server. This type of content is not only inappropriate but potentially illegal and should be reported and removed immediately.

**Link mentioned**: <a href="https://discord.gg/CYNumE8ABr">Join the e-girl paradise 🍑🍒 // +18 Discord Server!</a>: Check out the e-girl paradise 🍑🍒 // +18 community on Discord - hang out with 16457 other members and enjoy free voice and text chat.

  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1234909827453681764)** (11 messages🔥): 

- **LLaMA-3 Instruct Prompt Strategies Revealed**: An update to the **LLaMA-3 instruct prompt strategies** has been shared, claiming improvements on the model's performance, including the relevant GitHub [pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553).
  
- **Clarifying Dataset Entry Confusion**: A member detailed that using `eot_id` solved issues related to a previously attempted method where they were manually adding `<|eot_id|>` at the end of every dataset entry.

- **Meta's Iterative Reasoning Optimization Boosts Accuracy**: The paper titled "Iterative Reasoning Preference Optimization" has been circulated, indicating Meta's advancement with LLama-2-70B-Chat showing accuracy increases on multiple benchmarks like GSM8K and ARC-Challenge. The link to the paper is available [here](https://arxiv.org/abs/2404.19733).

- **Fine-tuning LLaMA-3 with Axolotl**: A user shared their experience fine-tuning **LLaMA-3 8b using Axolotl**, resulting in model outputs that include `</s>
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/arankomatsuzaki/status/1785489252299485188">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Meta presents Iterative Reasoning Preference Optimization  Increasing accuracy for Llama-2-70B-Chat:  - 55.6% -&gt; 81.6% on GSM8K - 12.5% -&gt; 20.8% on MATH - 77.8% -&gt; 86.7% on ARC-Challenge  htt...</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/conversation.html#sharegpt">Axolotl - Conversation</a>: no description found</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1553">feat: Add LLaMA-3 instruct prompt strategies for fine-tuning   by 0-hero · Pull Request #1553 · OpenAccess-AI-Collective/axolotl</a>: Description This builds on top of and includes the changes in the below PR&#39;s  #1542 #1539  Fastchat PR from @TJ-Solergibert needs to be merged before merging this  lm-sys/FastChat#3257   Motivatio...
</li>
</ul>

</div>
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1234767923105632326)** (2 messages): 

- **Motivational Beats to Keep You Pumping**: An anime-inspired motivational track titled "NEVER GIVE UP YOUR WAAAAAAAAAAAAY" was shared, featuring an instrumental version from the anime *Kill La Kill*. The [YouTube video](https://youtu.be/tYzMYcUty6s?si=t2utqcq36PHbk9da) encourages viewers to never give up, with a link to a Patreon for support.
- **Count Me In!**: A member responded enthusiastically with "I'll be there too," indicating participation or support in relation to the previously shared content.

**Link mentioned**: <a href="https://youtu.be/tYzMYcUty6s?si=t2utqcq36PHbk9da">NEVER GIVE UP YOUR WAAAAAAAAAAAAY</a>: NEVA GIVE UP - https://bit.ly/2VrgAcKSong is Before my Body is Dry instrumental version from the anime Kill La KillConsider donating to our Patreon!https://w...

  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1234775513499963463)** (1 messages): 

- **Quick Load Times Locally**: A member mentioned that running their program on their local machine is fast as it *loads in 3 secs*, suggesting that storage is not the problem when compared to slower load times after submitting a job.
  

---


**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/)** (1 messages): 

le_mess: llama 3 seems to beat gpt4 on scandeval
https://scandeval.com/german-nlg/
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1235150707439702057)** (1 messages): 

- **Exploring Model Expansion with qdora**: A member sparked interest in LLM expansion by mentioning **qdora**, a middleway solution for models like LLaMA. They provided a link to an [Answer.ai blog post](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html) discussing the process.
- **Delving into LLaMA Pro's Non-forgetful Learning**: The conversation also highlighted new post-pretraining methods aimed at preventing catastrophic forgetting in LLMs, pointing to an [Arxiv paper](https://arxiv.org/abs/2401.02415) on expanding Transformer blocks to retain old skills while acquiring new ones.

**Link mentioned**: <a href="https://arxiv.org/abs/2401.02415">LLaMA Pro: Progressive LLaMA with Block Expansion</a>: Humans generally acquire new skills without compromising the old; however, the opposite holds for Large Language Models (LLMs), e.g., from LLaMA to CodeLLaMA. To this end, we propose a new post-pretra...

  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1234824449552027749)** (2 messages): 

- **Datasette UX Challenge**: A member seeks ideas for a user interface on the Datasette front page where users can select options from a dropdown, like choosing a country to fetch summary data related to that selection.
- **Contemplating Dynamic URLs vs. Customizable Interface**: Two UX approaches were suggested for the Datasette front page; one involves updating the URL dynamically on event to bring the user directly to the data, while the other allows users to "build" the homepage by updating canned queries based on their selection.
  

---



---




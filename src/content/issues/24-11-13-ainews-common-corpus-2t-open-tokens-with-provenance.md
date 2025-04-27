---
id: d946c7f1-6897-4eb2-a880-64a0fab7aea9
title: 'Common Corpus: 2T Open Tokens with Provenance'
date: '2024-11-14T01:54:53.118250Z'
original_slug: ainews-common-corpus-2t-open-tokens-with
description: >-
  **Pleais** via **Huggingface** released **Common Corpus**, the largest fully
  open multilingual dataset with over **2 trillion tokens** including detailed
  **provenance information**. They also introduced **OCRonos-Vintage**, a
  **124M-parameter OCR correction model** that efficiently fixes digitization
  errors on CPU and GPU, unlocking knowledge from PDFs. On AI tools,
  **LangChainAI** launched **Prompt Canvas** for collaborative **prompt
  engineering**, while **DeepSeek** released **JanusFlow 1.3B**, a unified
  multimodal LLM integrating autoregressive and rectified flow models for
  enhanced **image understanding** and **generation**. **Alibaba Cloud**
  announced **Qwen2.5-Coder**, a code-focused LLM with advanced coding
  capabilities, and **Claude 3.5 Sonnet** was highlighted for superior code
  generation. Discussions on **quantization challenges** and **scaling laws for
  precision** by **Tim Dettmers** and others emphasized the impact of
  low-precision training on model scalability and inference efficiency.
  *"Scaling Laws for Precision"* paper insights and alternative efficiency
  methods were also noted.
companies:
  - pleais
  - huggingface
  - langchainai
  - deepseek
  - alibaba
  - anthropic
models:
  - qwen-2.5-coder
  - claude-3.5-sonnet
  - janusflow-1.3b
  - ocronos-vintage
topics:
  - provenance
  - ocr
  - multilingual-datasets
  - prompt-engineering
  - multimodality
  - image-generation
  - code-generation
  - quantization
  - model-scaling
  - inference-efficiency
people:
  - tim-dettmers
  - tom-doerr
  - omarsar0
  - swyx
  - madiator
  - reach_vb
---


<!-- buttondown-editor-mode: plaintext -->**Provenance is all you need.**

> AI News for 11/12/2024-11/13/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**217** channels, and **2494** messages) for you. Estimated reading time saved (at 200wpm): **274 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Great dataset releases always precede great models. Last we covered FineWeb ([our coverage here](https://buttondown.com/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/)) a [generation of GPT2 speedruns ensued](https://www.reddit.com/r/LocalLLaMA/comments/1gmd1a8/are_people_speedrunning_training_gpts_now/). Today, Pleais (via Huggingface) is here with an update [Common Corpus](https://huggingface.co/blog/Pclanglais/two-trillion-tokens-open) "the largest fully open multilingual dataset for training LLMs, containing over 2 trillion tokens of permissibly licensed content **with provenance information** (2,003,039,184,047 tokens)."

![image.png](https://assets.buttondown.email/images/39a4c40e-241d-42ef-b910-48d173726991.png?w=960&fit=max)

Apart from the meticulous provenance, the team also used [OCRonos-Vintage](https://huggingface.co/PleIAs/OCRonos-Vintage), "a lightweight but powerful OCR correction model that fixes digitization errors at scale. Running efficiently on both CPU and GPU, this 124M-parameter model corrects spacing issues, replaces incorrect words, and repairs broken text structures." This unlocks a whole lot of knowledge in PDFs:

![image.png](https://assets.buttondown.email/images/120274cc-94ed-4244-8798-e0e587ae7603.png?w=960&fit=max)
 

Common Corpus was first [released in March with 500b tokens](https://huggingface.co/blog/Pclanglais/common-corpus) so it is nice to see this work grow.

---

{% if medium == 'web' %}

**Table of Contents**

[TOC]

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}

---

# AI Twitter Recap

> all recaps done by Claude 3.5 Sonnet, best of 4 runs.

**AI Tools and Development**

- **Prompt Engineering and Collaboration**: [@LangChainAI](https://twitter.com/LangChainAI/status/1856386593457848746) introduced **Prompt Canvas**, a **novel UX** for **prompt engineering** that facilitates collaboration with AI agents and standardizes prompting strategies across organizations. Additionally, [@tom_doerr](https://twitter.com/tom_doerr/status/1856507277903307153) showcased tools like **llama-ocr** and **TTS Generation WebUI**, enhancing **OCR** and **text-to-speech** capabilities for developers.

- **AI Development Platforms**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1856552494379520510) launched **JanusFlow 1.3B**, a **unified multimodal LLM** integrating **autoregressive models** with **rectified flow** for superior **image understanding** and **generation**. Similarly, [@swyx](https://twitter.com/swyx/status/1856458546109632984) provided updates on **proxy servers** and **realtime client SDKs**, improving the **developer experience** for **realtime applications**.

**AI Model Releases and Updates**

- **New LLMs and Enhancements**: [@tom_doerr](https://twitter.com/tom_doerr/status/1856597874991055248) announced **Qwen2.5-Coder**, a **code-focused LLM** from **Alibaba Cloud**, emphasizing its **advanced coding capabilities**. Meanwhile, [@omarsar0](https://twitter.com/omarsar0/status/1856505917686022276) highlighted the release of **Claude 3.5 Sonnet**, showcasing its **superior code generation** performance compared to other models.

- **Performance Benchmarks**: [@omarsar0](https://twitter.com/omarsar0/status/1856505917686022276) compared **Qwen2.5-Coder** with **Claude 3.5 Sonnet**, discussing their **code generation capabilities** and potential to **bridge the gap** between **open-source** and **closed-source** models. Additionally, [@reach_vb](https://twitter.com/reach_vb/status/1856560887437373766) introduced **DeepSeek's JanusFlow 1.3B**, highlighting its **state-of-the-art performance** in **multimodal tasks**.

**AI Research and Technical Insights**

- **Quantization and Model Scaling**: [@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1856419493766930846) explored the **challenges of quantization** in **AI models**, noting that **low-precision training** may **limit future scalability**. [@madiator](https://twitter.com/madiator/status/1856430350295257394) summarized the paper "**Scaling Laws for Precision**," revealing that **increased pretraining data** heightens **model sensitivity to quantization**, impacting **inference efficiencies** and **GPU provisioning**.

- **Scalability and Efficiency**: [@lateinteraction](https://twitter.com/lateinteraction/status/1856409143051202886) discussed the **limitations of scaling through precision**, suggesting alternative methods for **efficiency gains**. Furthermore, [@deepseek_ai](https://twitter.com/deepseek_ai/status/1856552494379520510) presented the **Forge Reasoning Engine**, leveraging **Chain of Code**, **Mixture of Agents**, and **MCTS** to **enhance reasoning** and **planning** in **Hermes 3 70B**.

**Developer Tips and Tools**

- **System Monitoring and Optimization**: [@giffmana](https://twitter.com/giffmana/status/1856429747385073838) recommended switching from `htop` to **btop** for a more **aesthetic** and **functional system monitor**. Additionally, [@swyx](https://twitter.com/swyx/status/1856445290632622442) provided guidance on managing **realtime client SDKs** and optimizing **development workflows**.

- **Software Engineering Best Practices**: [@hyhieu226](https://twitter.com/hyhieu226/status/1856532701219828054) emphasized the principle "**if it's not broken, don't fix it!**," advocating for **simplicity** and **stability** in **software engineering** practices.

**AI Adoption and Impact**

- **Healthcare Transformation**: [@bindureddy](https://twitter.com/bindureddy/status/1856513273753154025) discussed how **AI**, in combination with initiatives like **DOGE** and **RFK**, can **transform healthcare** by addressing **inefficiencies** and **high costs** through **innovative AI solutions**.

- **Automation and Workforce**: [@bindureddy](https://twitter.com/bindureddy/status/1856425643036291267) highlighted the potential for **AI** to **automate white-collar professions** and **transform transportation**, predicting significant **impact on the workforce** and emphasizing that **AI adoption** is still in its **early stages**, with the **last mile** expected to take the **better part of the decade**.

- **Enterprise AI Innovations**: [@RamaswmySridhar](https://twitter.com/RamaswmySridhar/status/1856411645796556887) introduced **Snowflake Intelligence**, enabling **enterprise AI** capabilities such as **data agents** that facilitate **data summarization** and **actionable insights** within **enterprise environments**.

**Memes/Humor**

- **Humorous AI Remarks**: [@nearcyan](https://twitter.com/nearcyan/status/1856565818433355790) joked about users preferring **ChatGPT** over **Claude**, equating it to having "**green text msgs**," while [@vikhyatk](https://twitter.com/vikhyatk/status/1856576789545660611) humorously outlined the steps of a **strike** culminating in **profit**, adding a light-hearted touch to discussing labor actions.

- **Tech and AI Humor**: [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1856546198137799106) likened Elon Musk's potential **government fixes** to George Hotz's rapid **Twitter repairs**, using a humorous comparison to illustrate skepticism. Additionally, [@teortaxesTex](https://twitter.com/teortaxesTex/status/1856624065227751490) shared a funny take on AI modeling with “**i need to lock in**” repetitions, adding levity to technical discussions.

- **Relatable Developer Jokes**: [@giffmana](https://twitter.com/giffmana/status/1856603033183900128) humorously referred to his extensive technical talk as a "**TED talk**," while [@ankush_gola11](https://twitter.com/ankush_gola11/status/1856400616043483204) expressed excitement about **Prompt Canvas** with a playful enthusiasm.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Qwen 2.5 Coder Improves 128K Context but Faces Usability Challenges**

- **Bug fixes in Qwen 2.5 Coder & 128K context window GGUFs** ([Score: 332, Comments: 90](https://reddit.com/r/LocalLLaMA/comments/1gpw8ls/bug_fixes_in_qwen_25_coder_128k_context_window/)): The post discusses updates and bug fixes for **Qwen 2.5 models**, emphasizing the extension of context lengths from **32K to 128K** using YaRN, and highlights the availability of **native 128K GGUFs** on [Hugging Face](https://huggingface.co/unsloth). It also warns against using `
  - **YaRN and Context Lengths**: Discussions center around the use of **YaRN** for extending context lengths in **Qwen 2.5 models**. Users express concerns about performance impacts when using **128K contexts** and suggest using **32K** for general tasks, with adjustments for longer contexts only when necessary.
  - **Bug Fixes and Tool Calling**: The **GGUFs** uploaded include bug fixes, particularly addressing untrained tokens and pad token issues. There is a notable mention that both the **Coder Base** and **Instruct models** did not train for tool calling, with users discussing the untrained status of `<tool_call>` tokens.
  - **GPU Limitations and Fine-Tuning**: Users inquire about the max sequence length for training on GPUs, with the **14B model** approximating **12K context length** on a **40GB GPU**. There is also a discussion on fine-tuning without using YaRN initially and the potential benefits of this approach.
- **Qwen 2.5 Coder 14b is worse than 7b on several benchmarks in the technical report - weird!** ([Score: 37, Comments: 23](https://reddit.com/r/LocalLLaMA/comments/1gpriif/qwen_25_coder_14b_is_worse_than_7b_on_several/)): The **Qwen 2.5 Coder 14B** model underperforms on certain benchmarks compared to the **7B** version, as highlighted in the [technical report](https://arxiv.org/pdf/2409.12186). The author notes that for specific tasks like SQL revisions, the non-coding 14B model performs better, suggesting that generalist models may have a superior understanding in some contexts.
  - Users reported **performance issues** with the **Qwen 2.5 Coder 14B** model, with some suggesting that the benchmarks might be incorrect due to a reporting error, as they have observed different performance in practice. A link to the [Qwen 2.5 Coder blog](https://qwenlm.github.io/blog/qwen2.5-coder-family/) was shared for further information.
  - There are **inconsistencies with quantized model files**, where different Q8 files yield varying results, highlighting potential issues with some files being faulty. One user shared a working Q8 file from [Hugging Face](https://huggingface.co/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/tree/main), suggesting that not all files are reliable.
  - A user pointed out that the **benchmark table contains errors**, as the numbers for the 14B and 1.5B models are identical except for the livecode benchmark, indicating a likely data entry mistake.
- **Qwen 2.5 32B Coder doesn't handle the Cline prompt well.  It hallucinates like crazy.  Anyone done any serious work with it yet?** ([Score: 21, Comments: 46](https://reddit.com/r/LocalLLaMA/comments/1gpqhgu/qwen_25_32b_coder_doesnt_handle_the_cline_prompt/)): The post discusses issues with **Qwen 2.5 Coder 32B** when handling **Cline prompts**, noting that it often hallucinates. The author mentions trying different setups like **vLLM** and **OpenRouter/Hyperbolic** without success, though they achieve better results using a simple Python script to manage file outputs.
  - Users report mixed results with **Qwen 2.5 Coder 32B**; some experience success using [Ollama's version](https://ollama.com/hhao/qwen2.5-coder-tools) on an **M1** with **64G RAM**, while others face issues with **Cline** prompts, leading to unrelated outputs or infinite loops.
  - **Configuration and setup** play a crucial role, with one user suggesting manual edits to **config.json** to properly integrate the model with **continue**. Properly prompting **Qwen** is emphasized as critical, given the lack of a standard prompt format.
  - Some users highlight the model's efficiency with large inputs, capable of handling **50k+ tokens** and **100 API calls** with minimal cost, but note that success varies depending on the integration tool used (e.g., **AIder**, **cursor**).


**Theme 2. Scaling Laws in Precision and CPU Inference Tests**

- **LLM inference with tensor parallelism on a CPU** ([Score: 43, Comments: 8](https://reddit.com/r/LocalLLaMA/comments/1gporol/llm_inference_with_tensor_parallelism_on_a_cpu/)): The author conducted experiments to assess the scalability of **LLM inference with tensor parallelism** on CPUs using the **distributed-llama** project. In the first experiment with **Epyc 9374F** as compute nodes, performance scaled to nearly 7x with 8 nodes after optimizing logits calculation. The second experiment using **Ryzen 7700X** nodes connected via a **10Gbe network** showed a 6x performance increase with 8 nodes, demonstrating that LLM inference can effectively scale on CPUs, though further optimizations could improve results. The author's fork of distributed-llama can be found [here](https://github.com/fairydreaming/distributed-llama).
  - **Memory Bandwidth and NUMA Nodes**: The discussion clarifies that the 8 nodes in the first experiment were not VMs but separate processes bound to NUMA nodes on the **Epyc CPU**. This setup allowed communication via loopback network, with potential scalability improvements if shared memory communication replaces networking, highlighting a theoretical memory bandwidth of **2 \* 576 GB/s** for dual **Epyc Turin** CPUs.
  - **Network Bottleneck Considerations**: Commenters noted that the **10Gbe network** used in the second experiment might be a bottleneck for distributed CPU inference. The author acknowledges that while loopback networking was used in the first experiment, physical network setups could benefit from tuning to reduce latency and improve efficiency, especially concerning NIC drivers and OS network configurations.
  - **Encouragement for Distributed CPU Inference**: The results of these experiments are seen as promising for distributed CPU inference. There is interest in leveraging existing systems, including older or mid-range setups, for scalable inference tasks, with a focus on optimizing network and memory configurations to maximize performance.

- **Scaling Laws for Precision. Is BitNet too good to be true?** ([Score: 27, Comments: 7](https://reddit.com/r/LocalLLaMA/comments/1gq3gs7/scaling_laws_for_precision_is_bitnet_too_good_to/)): A new paper, **"Scaling Laws for Precision"** ([arxiv link](https://arxiv.org/pdf/2411.04330)), explores how quantization affects model precision and output quality, emphasizing that increased token use in pre-training exacerbates quantization's negative impact in post-training. The author suggests 6-bit quantization as an optimal balance and hopes the findings will guide major labs in optimizing compute resources; additional insights are discussed in the **AINews** letter with opinions from **Tim Dettmers** ([AINews link](https://buttondown.com/ainews/archive/ainews-bitnet-was-a-lie/)).
  - **Quantization Awareness Training (QAT)** is emphasized as a crucial approach, where training is aware of quantization, allowing for more effective weight distribution, contrasting with post-training quantization which can degrade model performance, especially when trained in **FP16**.
  - The **cosine learning rate schedule** is clarified as distinct from **cosine similarity**, with the former related to training dynamics and the latter to measuring vector similarity, both involving the cosine function but serving different purposes.
  - **Bitnet's approach** is discussed as not being included in the study, with a focus on how models trained in **bf16** can lose important data when quantized post-training, differing from **QAT** which maintains a 1:1 model integrity.


**Theme 3. Largest Mixture of Expert Models: Analysis and Performance**

- **Overview of the Largest Mixture of Expert Models Released So Far** ([Score: 32, Comments: 2](https://reddit.com/r/LocalLLaMA/comments/1gprkxw/overview_of_the_largest_mixture_of_expert_models/)): The post provides an overview of the largest **Mixture of Expert (MoE) models** with over **100 billion parameters** currently available, emphasizing their architecture, release dates, and quality assessments. Key models include **Switch-C Transformer by Google** with 1.6 trillion total parameters, **Grok-1 by X AI** with 314 billion total parameters, and **DeepSeek V2.5 by DeepSeek**, which ranks highest overall. The post suggests that while **DeepSeek V2.5** is currently top-ranked, **Tencent's Hunyuan Large** and the unreleased **Grok-2** could surpass it, noting that model suitability depends on specific use cases. For more details, you can refer to the [HuggingFace blog](https://huggingface.co/blog/moe) and individual model links provided in the post.
- **[NousResearch Forge Reasoning O1 like models https://nousresearch.com/introducing-the-forge-reasoning-api-beta-and-nous-chat-an-evolution-in-llm-inference/](https://i.redd.it/n5j9zfjiwi0e1.png)** ([Score: 240, Comments: 43](https://reddit.com/r/LocalLLaMA/comments/1gptb4i/nousresearch_forge_reasoning_o1_like_models/)): **NousResearch** has introduced the **Forge Reasoning API** and **Nous Chat**, which enhance reasoning capabilities in **LLM (Large Language Models)**. This development represents an evolution in LLM inference, as detailed in their announcement [here](https://nousresearch.com/introducing-the-forge-reasoning-api-beta-and-nous-chat-an-evolution-in-llm-inference/).
  - **Forge Reasoning API** is not a new model but a system that enhances reasoning in existing models using **Monte Carlo Tree Search (MCTS)**, **Chain of Code (CoC)**, and **Mixture of Agents (MoA)**. Despite being closed source and only available through an API waitlist, it demonstrates potential for reasoning improvements in LLMs, akin to advancements seen in open-source image generation.
  - The discussion highlights skepticism and curiosity about the **open-source status** and the effectiveness of the Forge Reasoning API, with some users comparing it to **Optillm** on [GitHub](https://github.com/codelion/optillm) for experimenting with similar techniques. Users are keen on seeing independent tests to verify the claimed advancements in reasoning capabilities.
  - The conversation reflects on the nature of technological advancements, with **NousResearch**'s efforts being likened to historical breakthroughs that become commonplace over time. It emphasizes the importance of workflows and system integration over standalone model improvements, pointing to a trend where open-source LLMs are beginning to receive similar enhancements as seen in other AI domains.


**Theme 4. Unreliable Responses in Qwen 2.5: Self-Identification Issues**

- **[qwen2.5-coder-32b-instruct seems confident that it's made by OpenAI when prompted in English. States is made by Alibaba when prompted in Chinese.](https://www.reddit.com/gallery/1gqao05)** ([Score: 22, Comments: 15](https://reddit.com/r/LocalLLaMA/comments/1gqao05/qwen25coder32binstruct_seems_confident_that_its/)): **Qwen 2.5 Coder** exhibits inconsistent behavior regarding its origin, claiming to be developed by **OpenAI** when queried in English and by **Alibaba** when queried in Chinese.
  - **LLMs and Introspection**: Several users, including **JimDabell** and **Billy462**, highlight that **Large Language Models (LLMs)**, like **Qwen 2.5 Coder**, lack introspection capabilities and often produce "hallucinations" when asked about their origin, leading to inconsistent responses about their creators.
  - **Inconsistent Responses**: Users, such as **pavelkomin** and **muxxington**, report varied responses from the model, where it claims to be made by different entities like **Alibaba**, **OpenAI**, **Tencent Cloud**, **Anthropic**, and **Meta**, indicating a strong influence from repeated phrases in training data rather than factual accuracy.
  - **Practical Concerns**: Some users, such as **standard-protocol-79**, express indifference towards these inconsistencies as long as the model continues to generate effective code, suggesting that the primary concern for many is the model's utility rather than its self-identification accuracy.

- **How to use Qwen2.5-Coder-Instruct without frustration in the meantime** ([Score: 32, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1gpwrq1/how_to_use_qwen25coderinstruct_without/)): To improve performance with **Qwen2.5-Coder-Instruct**, avoid high repetition penalties, using slightly above 0 instead. Follow the [recommended inference parameters](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct/blob/main/generation_config.json), as low temperatures like T=0.1 are reportedly not problematic. Utilize **bartowski's quants** for better output quality, and begin system prompts with "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." to enhance performance. Despite these adjustments, some users experience issues with **vLLM** and recommend alternatives like **llama.cpp + GGUF**.
  - Users discuss **temperature settings** and **repetition penalties** for coding models like **Qwen2.5-Coder-32B-Instruct**. **No-Statement-0001** found that a temperature of **0.1** successfully executed complex prompts, while others suggest avoiding high repetition penalties as they can degrade performance, with **FullOf_Bad_Ideas** recommending turning off repetition penalties for better zero-shot results.
  - Some users, like **Downtown-Case-1755**, question the recommended high repetition penalties, noting that **1.05** is too high for coding tasks that naturally involve repetition. **EmilPi** highlights the importance of **Top_K** settings, which significantly impact model performance, as observed in the `generation_config.json`.
  - **Status_Contest39** shares experiences with different deployment setups, finding **DeepInfra's** default parameters effective despite a **Max Token** limit of **512**. **Master-Meal-77** expresses dissatisfaction with official sampler recommendations, preferring a custom setup with **top-p 0.9, min-p 0.1, and temp 0.7** for optimal results.


## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**Theme 1. AI Video Gen Evolution: CogVideoX 5B & DimensionX Release**



- **[CogVideoX1.5-5B Image2Video Tests.](https://v.redd.it/wi3hcmwd4q0e1)** ([Score: 91, Comments: 43](https://reddit.com/r/StableDiffusion/comments/1gqltkx/cogvideox155b_image2video_tests/)): **CogVideoX1.5-5B**, a new **image-to-video generation model**, demonstrates its capabilities in converting still images to video content. No additional technical details or performance metrics were provided in the post.
  - **CogVideoX1.5-5B** requires **34GB memory** during generation and **65GB** for VAE, with generation taking **15 minutes** per **5-second video** at **16fps**. The model is currently accessible via [command line inference script](https://github.com/THUDM/CogVideo/tree/main/sat) and has been tested on **A100** and **H100 80GB** GPUs.
  - Development updates indicate that **Kijai** is working on implementing version 1.5 in their wrapper, with a test branch already available for the **cogwrapper**. Integration with **Comfy UI** support is pending, while **Mochi-1** offers an alternative requiring only **12GB VRAM**.
  - Users discussed motion quality improvements, noting that faster playback reduces the AI-generated slow-motion effect, as demonstrated in a [sample video](https://streamable.com/m1e1cw). Several comments focused on the need for more realistic physics simulation in the generated animations.


- **[DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion | Flux Dev => DimensionX Demo](https://v.redd.it/05q21m50ln0e1)** ([Score: 176, Comments: 51](https://reddit.com/r/StableDiffusion/comments/1gqanyv/dimensionx_create_any_3d_and_4d_scenes_from_a/)): **DimensionX** enables generation of **3D and 4D scenes** from a single input image using **controllable video diffusion** techniques. The tool, developed by **Flux Dev**, allows for creation of dynamic scenes and environments from static images.
  - The project's official resources are available on [GitHub](https://github.com/wenqsun/DimensionX), [HuggingFace](https://huggingface.co/wenqsun/DimensionX), and detailed in their [research paper](https://arxiv.org/abs/2411.04928) and [project page](https://chenshuo20.github.io/DimensionX/). A **Docker template** is also available for implementation.
  - Users discussed potential applications in **3D modeling software**, suggesting integration with **Blender** and **Unity**, similar to **Nvidia NeRF** but requiring only single images. Some mentioned using it with **photogrammetry software** for environment creation.
  - The term **4D** in the project refers to **time** as the fourth dimension, essentially creating **3D scenes with temporal animation**. Users also noted concerns about the workflow process and implementation details.


**Theme 2. Claude's Performance Issues & Rate Limits Spark User Frustration**



- **The New Claude Sonnet 3.5 is Having a Mental Breakdown?** ([Score: 43, Comments: 79](https://reddit.com/r/ClaudeAI/comments/1gqnom0/the_new_claude_sonnet_35_is_having_a_mental/)): **Claude Sonnet 3.5** users report a significant performance decline over the past **72 hours**, with notable deterioration in **code quality**, **response coherence**, and **overall output quality** compared to previous performance levels. The degradation appears consistent across multiple prompting approaches and historical prompts that previously worked well, with **coding tasks** specifically highlighted as an area of concern.
  - Multiple developers report **Claude** incorrectly defaulting to **React** solutions regardless of the specified framework (**Angular**, **ESP8266**), with one user noting *"at no point, in any of my files or prompts, is React a component of the project"*.
  - Users observe deteriorating response patterns including shortened bullet points, repetitive suggestions, and inability to handle basic code modifications. A developer who previously *"built and published multiple apps"* using Claude notes significant decline in even simple coding tasks.
  - According to a comment referencing the **Lex Fridman** interview with **Anthropic's CEO**, major AI labs sometimes reduce model quality through **quantization** to cut costs (by **200-400%**), though this typically affects web interface users rather than API access.


- **Claude Pro limits needs revision now on priority** ([Score: 100, Comments: 66](https://reddit.com/r/ClaudeAI/comments/1gq9ihw/claude_pro_limits_needs_revision_now_on_priority/)): **Claude Pro** users express frustration over the **2-hour usage caps** and frequent limit restrictions, which interrupt their workflow and productivity. Users demand **Anthropic** revise the current **Pro tier limits** to better accommodate paying customers' needs.
  - Users discuss alternative solutions including using the **API** or rotating multiple **Pro accounts**, though many note the **API costs** would be significantly higher for their usage patterns, especially for those working with large text files.
  - Several **writers** and **developers** share frustrations about hitting limits while working with large projects, particularly when using features like **Project Knowledge** and **artifacts**. One user reports hitting limits *"4 times a day"* while working with *"80k words"* of worldbuilding files.
  - Multiple users mention using **ChatGPT** as a fallback when hitting **Claude's limits**, though they prefer Claude's capabilities. Some users have canceled their subscriptions due to the restrictions, with one suggesting a more realistic price point of *"$79.99 per month"*.


**Theme 3. Gemini Now Accessible via OpenAI API Library**



- **[Gemini is now accessible from the OpenAI Library. WTH?](https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/)** ([Score: 172, Comments: 41](https://reddit.com/r/OpenAI/comments/1gq5zz6/gemini_is_now_accessible_from_the_openai_library/)): **Google** announced that **Gemini** can be accessed through the **OpenAI Library**, though the post lacks specific details about implementation or functionality. The post expresses confusion about this integration's implications and purpose.
  - **Google's Gemini API** now accepts requests through the **OpenAI API client library**, requiring only three changes to implement: **model name**, **API key**, and **endpoint URL** to `generativelanguage.googleapis.com/v1beta`. This adaptation follows industry standards as many LLM providers support the OpenAI API format.
  - The **OpenAI library** remains unchanged since it's endpoint-agnostic, with all modifications made on **Google's server-side** to accept OpenAI-formatted requests. This allows developers to easily switch between providers without major code rewrites.
  - Multiple commenters clarify that this is not a collaboration between companies but rather **Google** implementing compatibility with an established standard API format. The **OpenAI API** has become the *"de-facto standard"* for LLM interactions.


**Theme 4. Greg Brockman Returns to OpenAI Amid Leadership Changes**



- **[OpenAI co-founder Greg Brockman returns to ChatGPT maker](https://indianexpress.com/article/technology/artificial-intelligence/openai-co-founder-greg-brockman-returns-to-chatgpt-maker-9666990/)** ([Score: 55, Comments: 5](https://reddit.com/r/OpenAI/comments/1gq3un6/openai_cofounder_greg_brockman_returns_to_chatgpt/)): **Greg Brockman** has returned to **OpenAI** after a three-month leave, announcing on **X** *"longest vacation of my life complete. back to building @OpenAl!"*, while working with **CEO Sam Altman** to create a new role focused on technical challenges. The return occurs amid significant leadership changes at the **Microsoft**-backed company, including departures of **Mira Murati**, **John Schulman**, and **Ilya Sutskever**, with **OpenAI** simultaneously developing its first **AI inference chip** in collaboration with **Broadcom**.
  - [{'id': 'lwwdcmr', 'author': 'ManagementKey1338', 'body': 'Indeed. I didn’t give him an offer. The man wants too much money.', 'score': 4, 'is_submitter': False, 'replies': []}]


**Theme 5. Major AI Companies Hitting Scaling Challenges**



- **[OpenAI, Google and Anthropic Are Struggling to Build More Advanced AI](https://www.bloomberg.com/news/articles/2024-11-13/openai-google-and-anthropic-are-struggling-to-build-more-advanced-ai)** ([Score: 119, Comments: 114](https://reddit.com/r/OpenAI/comments/1gqfz7l/openai_google_and_anthropic_are_struggling_to/)): **OpenAI**, **Google**, and **Anthropic** encounter technical and resource limitations in developing more sophisticated AI models beyond their current capabilities. The title suggests major AI companies face scaling challenges, though without additional context, specific details about these limitations cannot be determined.
  - **Meta** reports no diminishing returns with model training, only stopping due to **compute limitations**. The new **Nvidia Blackwell** series offers **8x performance** for transformers, while **OpenAI** continues progress with **SORA**, **advanced voice mode**, and **O-1**.
  - Companies face challenges with **training data availability** and need new architectural approaches beyond the "more data, more parameters" paradigm. Current development areas include **voice**, **vision**, **images**, **music**, and **horizontal integration**.
  - Future AI development may require new data sources including **smart-glasses**, **real-time biometric data**, and specialized models for niche applications. The field is experiencing what some describe as the peak of the **Hype Cycle**, heading toward a potential "**Trough of Disillusionment**".


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1. New AI Models Shake Up the Landscape**

- [**Qwen Coder Models Spark Excitement**](source_url): Across several communities, developers are buzzing about the **Qwen Coder models**, eagerly testing their performance and sharing benchmarks. The models show promise in code generation tasks, stirring interest in their potential impact.
- [**UnslothNemo 12B Unleashed for Adventurers**](https://openrouter.ai/thedrummer/unslopnemo-12b): The **UnslothNemo 12B** model, tailored for **adventure writing** and **role-play**, has been launched. A **free variant** is available for a limited time, inviting users to dive into immersive storytelling experiences.
- [**Aider v0.63.0 Codes Itself**](source_url): The latest release of **Aider** boasts that **55%** of its new code was self-authored. With added support for **Qwen 2.5 Coder 32B** and improved exception handling, **Aider v0.63.0** takes a leap forward in AI-assisted development.

**Theme 2. AI Tools and Integrations Enhance Workflows**

- [**AI Coding Tools Join Forces**](https://supermaven.com/blog/cursor-announcement): **Supermaven** has joined **Cursor** to build a powerhouse AI code editor. Together, they aim to enhance AI-assisted coding features, improving productivity for developers worldwide.
- [**Windsurf Editor Makes a Splash**](https://codeium.com/windsurf): **Codeium** launched the **Windsurf Editor**, the first agentic IDE that combines AI collaboration with independent task execution. Users are excited about its potential to maintain developer flow and boost coding efficiency.
- [**LM Studio Eyes Text-to-Speech Integration**](source_url): Users expressed keen interest in integrating **Text-to-Speech (TTS)** features into **LM Studio**. The development team acknowledged the demand and is exploring possibilities to enhance the platform's interactivity.

**Theme 3. Benchmark Showdowns: Models Put to the Test**

- [**Vision Language Models Battle in Robotics**](https://arxiv.org/abs/2411.05821): A new research paper benchmarks **Vision, Language, & Action Models** like **GPT-4** on robotic tasks. The study evaluates model performance across **20 real-world tasks**, highlighting advancements in multimodal AI.
- [**Qwen 2.5 Coder vs. GPT-4: Clash of Titans**](https://youtu.be/Xs0EkLYu6hw): Enthusiasts compared **Qwen 2.5 Coder 32B** with **GPT-4** and **Claude 3.5 Sonnet**, debating which model reigns supreme in code generation. Impressive generation speeds on consumer hardware spark further interest.
- [**ChatGPT Keeps Dates Straight; Others Lag Behind**](source_url): Users noticed that models like **Gemini** and **Claude** often fumble with current dates, while **ChatGPT** maintains accurate date awareness. This difference is attributed to superior system prompt configurations in ChatGPT.

**Theme 4. Community Voices Concerns Over AI Trends**

- [**Perplexity Users Threaten to Jump Ship over Ads**](https://www.perplexity.ai/hub/blog/why-we-re-experimenting-with-advertising): **Perplexity AI** introduced ads, prompting backlash from users who feel their subscription should exempt them from advertising. The community awaits official clarification on how ads will affect the **Pro** version.
- [**Is the AI Bubble About to Burst?**](https://chrisbora.substack.com/p/the-ai-bubble-is-about-to-pop-heres): A provocative article warns of an impending **AI bubble burst**, likening the massive **$600 billion** GPU investments with minimal returns to the dot-com crash. The piece sparks debate on the sustainability of current AI investments.
- [**AI21 Labs Deprecates Models, Users Fume**](source_url): **AI21 Labs** faced user frustration after deprecating legacy models that many relied on for nearly two years. Concerns grow over the new models' quality and fears of future deprecations.

**Theme 5. Tech Challenges Push Developers to Innovate**

- [**Triton Tackles Tiny Tensor Troubles**](https://github.com/triton-lang/triton/issues/5138): Developers working with **Triton** are optimizing **GEMM kernels** for small sizes under 16, addressing efficiency challenges and sharing solutions for improved performance in matrix computations.
- [**torch.compile() Sparks Memory Woes**](source_url): Users report that using **torch.compile()** can increase peak memory usage by **3-16%**, leading to **out-of-memory errors** in models with dynamic shapes. The community discusses profiling techniques to manage memory more effectively.
- [**tinygrad Community Squashes Bugs Together**](https://github.com/tinygrad/tinygrad/pull/7675): The **tinygrad** team collaborates to fix a bug in the **min() function** for unsigned tensors. Through shared insights and code reviews, they demonstrate the power of open-source collaboration in improving AI frameworks.

---

# PART 1: High level Discord summaries

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen Coder Models Deployment**: Members discussed the current development and testing of the **Qwen Coder models**, expressing interest in their performance and potential evaluation metrics. There are mentions of files and fixes available on Unsloth with suggestions to run evaluations similar to other models.
  
  - Discussions highlighted the readiness of **Qwen Coder** for deployment, with community members proposing to benchmark it against existing models using provided resources.
- **Multi-GPU Training Limitations**: Users explored the potential for training large models like **Qwen 2.5** using multiple GPUs, specifically mentioning **MI300X** and **VRAM** needs. It was noted that **Unsloth** may be more efficient with a single GPU setup rather than multi-GPU configurations due to memory efficiency.
  
  - The community debated the scalability of multi-GPU training, with some advocating for increased parallelism while others pointed out the memory management challenges inherent in large-scale model training.
- **Gemma 2B RAM Usage Concerns**: Users discussed experiencing **consistent RAM usage increase** while running longer jobs with **Gemma 2B**, questioning if evaluation steps might be affecting performance. One member suggested training with **0 steps** to mitigate excessive resource consumption.
  
  - Suggestions were made to optimize training configurations to reduce RAM overhead, ensuring more stable performance during extended runs.
- **RAG for Long-term Memory**: Inquiry about **RAG (Retrieval-Augmented Generation)** was made, along with requests for user experiences and guidance on using it for long-term data retention. A user recommended **Dify** as a simple alternative for implementing RAG.
  
  - Community members shared various approaches to leveraging RAG, highlighting **Dify** as a user-friendly solution for integrating retrieval systems into generation workflows.
- **Optillm Release Enhancements**: The latest release of [**Optillm**](https://github.com/codelion/optillm) introduces a local inference server that allows loading any **HF model** and **LoRA adapters**, enhancing usability for fine-tuned Unsloth models. This update also enables dynamic adapter switching during inference and supports advanced decoding techniques like **cot_decoding** and **entropy_decoding** while utilizing the standard **OpenAI client SDK**.
  
  - Users lauded the new features in **Optillm**, noting the increased flexibility and improved workflow integration these enhancements bring to model inference processes.

 

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen Model Shows Variable Output**: Users reported that the **Qwen** model's performance in generating text can vary significantly, with comparisons made to other models like **Ollama** indicating that responses from Qwen often may hallucinate or lack quality.
  
  - Tweaking parameters like repetition penalty and adjusting token lengths were suggested to enhance output quality.
- **Introducing LightRAG for Retrieval**: An article was shared detailing **LightRAG**, which includes code evaluation comparing Naive RAG with local, global, and hybrid approaches.
  
  - The author aims to highlight the advantages of using LightRAG in various retrieval tasks. Read the full article [here](https://www.linkedin.com/posts/isham-rashik-5a547711b_introducing-lightrag-a-new-era-in-retrieval-activity-7262085232743342080-xgdo?utm_source=share&utm_medium=member_desktop).
- **Sulie Foundation Model for Time Series Forecasting**: **Sulie**, a new foundation model for time series forecasting, aims to simplify the automation of LoRA fine-tuning and covariate support.
  
  - The team seeks feedback and encourages users to check their work on [GitHub](https://github.com/wearesulie/sulie), humorously highlighting common frustrations faced by data teams by comparing issues to a 'chocolate teapot' for zero-shot performance.
- **Benchmarking VLA Models for Robotics**: A collaborative research paper titled **Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks** was released to evaluate the performance of VLA models like **GPT4o**.
  
  - This effort represents the initial phase of a wider benchmark for a new multimodal action model class. More details are available on the [website](https://multinet.ai/static/pages/Multinetv01.html) or on [GitHub](https://github.com/ManifoldRG/MultiNet/tree/main).
- **SDXL Lightning Model Demonstrates Fast Image Generation**: **SDXL Lightning** or **sd1.5 models** can generate images in just a few seconds on standard GPUs, making them ideal for prompt-based image creation.
  
  - Variants like **turbo/lightning/lcm** can produce images in real-time on powerful hardware, as shared by users experimenting with these configurations.

 

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI Subscription Model Scrutinized**: Users are evaluating the **Perplexity Pro** subscription, questioning its value amidst the introduction of ads, with many expressing intentions to cancel if ads are added.
  
  - There is growing uncertainty about whether ads will appear in the Pro version, leading users to seek official clarification from the Perplexity team.
- **Clarification Sought on Perplexity Ads Implementation**: Members are uncertain if ads will be included in the **Pro** subscription, prompting requests for confirmation to understand the impact on the user experience.
  
  - The community emphasizes the need for transparent communication from Perplexity regarding ad integration to maintain trust and subscription value.
- **Ongoing Model Selection Issues in Perplexity**: Users report persistent problems with selecting different models in **Perplexity**, with the system defaulting to GPT-4o despite selecting alternatives.
  
  - This malfunction disrupts workflows for **Pro** subscribers who rely on consistent access to various models like Claude.
- **Exploring Fractal Machine Learning Enhancements**: **Fractal Machine Learning** is being proposed to boost AI performance, with discussions on its potential applications in language models and collaborations with domain experts.
  
  - Community members are sharing resources and expressing interest in integrating fractal concepts to advance machine learning techniques.
- **Differentiating Factors of Perplexity AI Models**: An in-depth comparison highlights how [**Perplexity AI**](https://www.perplexity.ai/search/how-is-perplexity-ai-different-PF1ebdmMSci1d2dIu6UCiQ) stands out in the AI landscape through unique features and enhanced user experience.
  
  - The discussion focuses on key distinctions that could influence AI engineers' preferences when selecting AI tools for their projects.

 

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Mitigating Saddle Points in Gradient Descent**: During discussions on [gradient descent optimization](https://arxiv.org/abs/1406.2572), participants highlighted that **saddle points** are less impactful when using **noised gradient descent**, ensuring optimizers remain effective even in their presence.
  
  - However, some members emphasized that in **high-dimensional scenarios**, saddle points may still occur, suggesting that their prevalence isn't entirely mitigated.
- **Evolution of Batch Normalization Techniques**: The debate around **Batch Normalization** and its alternatives was prominent, with insights into the continued relevance of Batch Norm, especially when implemented as **Ghost Batch Norm**.
  
  - Critiques pointed out that Batch Norm's effectiveness varies with batch size, prompting calls for more research into its efficiency and optimal conditions for application.
- **Advancements in Vision Language Action Models**: A new research release presented [benchmarking Vision Language Action models](https://x.com/HarshSikka/status/1856739777208574151) in robotic tasks, involving prominent institutions and offering promising insights.
  
  - Participants were encouraged to provide feedback on the work and explore the provided [YouTube video](https://x.com/HarshSikka/status/1856739777208574151) and project links for a deeper understanding of the models and their applications.
- **Integrating DagsHub with GPT-NeoX**: The potential value of integrating **DagsHub** with **GPT-NeoX** was proposed, seeking community insights into enhancing the platform's capabilities.
  
  - Inquiries about **AnthropicAI's** frameworks revealed that they utilize proprietary systems, which are not publicly available.
- **Rethinking Gradient Descent Stepsizes**: [Professor Grimmer](https://x.com/prof_grimmer/status/1679846891171766272) challenged the conventional notion that **gradient descent** requires constant step sizes of **1/L** for optimal convergence.
  
  - His findings, detailed in [his paper](https://arxiv.org/abs/2307.06324), demonstrate that *periodic long steps* within the range **(0, 2/L)** can lead to better convergence results.

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **UnslopNemo 12B launched for adventure writing**: The **UnslopNemo 12B** model, tailored for **adventure writing** and **role-play scenarios**, is now available at [UnslopNemo 12B](https://openrouter.ai/thedrummer/unslopnemo-12b).
  
  - A free variant is accessible for **24 hours** via [UnslopNemo 12B Free](https://openrouter.ai/thedrummer/unslopnemo-12b:free), with support requests directed to [Discord](https://discord.gg/fVyRaUDgxW).
- **Mistral and Gemini gain parameter enhancements**: **Mistral** and **Gemini** models have been updated to include **Frequency Penalty** and **Presence Penalty** parameters, enhancing their configurability.
  
  - Additionally, **Mistral** now offers tools for **seed adjustments**, improving output consistency.
- **Confusion over Tool Calling functionality**: Users are experiencing issues with the **tool calling** feature on **OpenRouter**, as enabling it doesn't impact **token usage** as anticipated.
  
  - Discussions highlight the need for clearer implementation guidelines to fully leverage tool calling in model interactions.
- **High token processing volume prompts pricing discussions**: A user managing over **3 million tokens daily** for an AI chatbot in a niche market has inquired about potential **price reductions** for high-volume token processing.
  
  - This reflects a growing demand for scalable pricing models catering to extensive usage in specialized applications.
- **Requests surge for Custom Provider Keys**: Multiple members have requested access to **Custom Provider Keys**, indicating a strong interest in leveraging this feature for tailored integrations.
  
  - The community dialogue includes varied appeals, emphasizing the importance of **Custom Provider Keys** for diverse project requirements.

 

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.63.0 Launches with New Features**: The **Aider v0.63.0** release introduces support for **Qwen 2.5 Coder 32B** and enhanced handling of **LiteLLM exceptions**, improving overall usability.
  
  - **55%** of the code in this release was authored by Aider, demonstrating significant self-development.
- **Aider Extensions for VSCode and Neovim Released**: New **VSCode** and **Neovim** extensions for **Aider** have been launched, featuring markdown preview, file management, and chat history, encouraging community contributions.
  
  - These extensions aim to increase **Aider's** utility across various platforms, fostering collaboration among developers.
- **SupermavenAI Partners with Cursor**: Cursor announced that **SupermavenAI** is joining their team to enhance research and product capabilities, aiming to transform Cursor into a **product powerhouse**.
  
  - The partnership was unveiled via [Twitter](https://x.com/cursor_ai/status/1856427424927625679), highlighting plans for collaborative innovation.
- **Qwen 2.5 Coder Support Added to Aider**: **Aider** now supports **Qwen 2.5 Coder 32B**, integrating advanced coding capabilities into the platform.
  
  - This update facilitates improved code assistance and aligns **Aider's** features with contemporary coding standards.
- **OpenRouter Provider Configuration Tips for Aider**: Discussions on configuring **OpenRouter** for **Aider** included specifying provider preferences and creating model metadata files to manage costs and context sizes.
  
  - Users shared strategies for balancing provider use and emphasized understanding **OpenRouter's** load balancing mechanisms.

 

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Optimizing Quantization Sizes for LM Studio**: Members discussed the impact of **quantization sizes**, noting that smaller sizes lead to increased compression, while larger sizes may require splitting into multiple parts.
  
  - *Heyitsyorkie* summarized that higher quant sizes could ensure better performance without significant losses.
- **Integrating TTS with LM Studio**: **LM Studio** users are interested in connecting the platform to Text-to-Speech (TTS) features.
  
  - The response indicated ongoing conversations around integrating such features, but no timeline was provided.
- **Resolving Qwen 2.5 Performance Issues**: A user reported issues with **Qwen 2.5**, specifically receiving only autocomplete responses, but later noted it started working correctly.
  
  - Others recommended ensuring proper configuration and exploring model options to optimize performance.
- **Python Script for Llama.cpp Integration**: There was a request for a **Python script** to enable sideloading the latest **Llama.cpp** into **LM Studio**, highlighting the need for such functionality.
  
  - Participants acknowledged the community's long-standing anticipation and mentioned ongoing efforts to make it a reality.
- **GPU Combinations for Large Model Inference**: Discussion on using a **12GB 3060** and **40GB A800** together for **70B class models** raises the question of whether to use one GPU or both, with concerns on how scaling affects performance.
  
  - A member suggested that it may be more beneficial to solely utilize the **A800** since it can fit the model in **VRAM** while the 3060 cannot.

 

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Training with Movie Posters using Dreambooth**: A user is seeking **tutorials** for training on **movie posters** using **Dreambooth** in **auto1111**, looking for the latest techniques and suggestions for effective training.
  
  - The community recommended checking existing resources and guides to streamline the process.
- **Animatediff Enables Video Clip Generation**: Members discussed using **Animatediff** for generating video clips, highlighting its ability to post two images to create transitions, despite lower resolutions being suitable for social media.
  
  - A recommendation for the **Banodoco server** was provided, as they specialize in video-related tools.
- **Checkpoint & LoRa Download Sources**: Users shared links to external file hosting sites such as **Google Drive**, **Mega**, and **Hugging Face** for downloading checkpoint files and LoRAs, while discussing the limitations of **Civit AI** and potential content bans.
  
  - Concerns were raised about the removal of specific content types and their impact on user access.
- **Resolving Python Torch Errors in Stable Diffusion**: A user encountered an error with the **torch** package while setting up a Python environment for **Stable Diffusion**, and was advised to uninstall the current Python version and install **Python 3.10.11 64bit**.
  
  - The user expressed gratitude for the support and planned to implement the suggested solution soon.
- **Discord Access Issues and Solutions**: Users inquired about accessing URLs for Discord servers, specifically seeking new invites and direct links, with experiences of outdated invitation links for the **Pixaroma** community.
  
  - The community provided assistance to connect with the required Discord servers.

 

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Nous 3 Model Performance Confusion**: Discrepancies in **Nous' 70B model** performance figures have emerged as seen in [this thread](https://x.com/thexeophon/status/1856429292504096944?s=61), raising questions about the validity of the reported **MMLU-Pro** scores.
  
  - Members speculate that differences in prompting techniques and benchmark inconsistencies could be factors influencing these divergent numbers.
- **AI Agent Tool 'Operator' Launch**: OpenAI is set to launch a new **AI agent tool** called 'Operator' that automates tasks such as writing code and booking travel, expected to be released in January according to [this announcement](https://www.bloomberg.com/news/articles/2024-11-13/openai-nears-launch-of-ai-agents-to-automate-tasks-for-users?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTczMTUyODYxOCwiZXhwIjoxNzMyMTMzNDE4LCJhcnRpY2xlSWQiOiJTTVdOQURUMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.TTJZiuo4Nk2U295FHBFsxeN0YGznZJ32sHnNReQmEjM).
  
  - This tool aims to enhance user productivity by taking actions on behalf of individuals in various contexts.
- **JanusFlow Model Introduction**: The **JanusFlow model** is introduced as a new capability that harmonizes autoregressive LLMs with rectified flow for both image understanding and generation, detailed in [this post](https://x.com/deepseek_ai/status/1856552494379520510).
  
  - JanusFlow is designed to be robust, straightforward, and flexible, influencing the development of future AI models in this space.
- **Adaptive Techniques for Blocking Jailbreaks**: Anthropic's new research introduces adaptive techniques to **rapidly block new classes of jailbreak** as they are detected, as discussed in their paper [here](https://arxiv.org/abs/2411.07494).
  
  - *Ensuring perfect jailbreak robustness is hard,* highlighting the challenges in securing AI models.
- **Vision Language Models (VLMs)**: Members discussed **Vision Language Models (VLMs)**, referencing [Finbarr's blog](https://www.artfintel.com/p/papers-ive-read-this-week-vision) and a post on [VLM inference costs](https://x.com/goyalsachin007/status/1856004116012798355).
  
  - Key topics include the high computational cost due to 500+ image tokens and recent models like **Pixtral** and **DeepSeek Janus** improving text extraction from images.

 

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **KATT Catapults Podcast Productivity**: A member integrated **KATT** into their podcasting workflow, resulting in a fact-checked show exceeding **90 minutes** by utilizing an altered system prompt after **two years** of **KATT's** training.
  
  - This integration streamlined the production process, enhancing the hosts' ability to maintain accuracy and depth throughout the extended podcast episodes.
- **NotebookLM Nixed from External Sharing**: A member inquired about sharing **NotebookLM** content outside their Google Organization, leading to confirmation that **sharing is not possible** externally due to admin-imposed restrictions.
  
  - Further discussion revealed limitations on personal accounts, emphasizing the importance of adhering to organizational policies when handling **NotebookLM** data.
- **Gemini Guard: NotebookLM Data Security**: Concerns were raised regarding the security of data uploaded to **Gemini**, with clarifications stating that **paid accounts** ensure data security, whereas free accounts do not.
  
  - Members urged caution when uploading sensitive information, highlighting the necessity of maintaining **confidentiality** on the platform to prevent potential breaches.
- **Summarizing Success with NotebookLM**: A user sought tips for using **NotebookLM** to summarize texts for college literature reviews, prompting recommendations to utilize **synthetic datasets** for safeguarding sensitive data.
  
  - This approach aims to enhance the effectiveness of summaries while ensuring that privacy standards are upheld during the process.
- **Format Fails in Podcast Generation**: Users discussed challenges in generating podcasts from specific sources, particularly facing issues with **.md file formats**.
  
  - Recommendations included switching to **PDF** or **Google Docs** formats, which successfully resolved the podcast generation focus problems for users.

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Supermaven Joins Cursor**: Supermaven has officially joined [Cursor](https://supermaven.com/blog/cursor-announcement) to enhance their AI coding editor capabilities. The collaboration leverages Supermaven’s AI-assisted features for improved software development experiences.
  
  - Anyan sphere acquired Supermaven to beef up Cursor, with the deal remaining undisclosed. Community reactions were mixed, noting Supermaven's prior effectiveness while expressing surprise at the transition.
- **Codeium Launches Windsurf Editor**: Codeium introduced the [Windsurf Editor](https://codeium.com/windsurf), the first agentic IDE integrating AI collaboration with independent task execution, aiming to maintain developer flow.
  
  - Despite positive first impressions, some users noted that Windsurf Editor may not yet outperform established tools like Copilot in certain aspects. Additionally, the editor is available without waitlists or invite-only access, emphasizing user inclusion.
- **Perplexity Introduces Sponsored Ads**: Perplexity is experimenting with ads on its platform by introducing “sponsored follow-up questions” alongside search results. They partnered with brands like [Indeed](https://www.indeed.com) and [Whole Foods](https://www.wholefoods.com) to monetize their AI-powered search engine.
  
  - This move aims to establish a sustainable revenue-sharing program, addressing the insufficiency of subscriptions alone.
- **Mira Lab Forms New AI Team**: Mira Lab, initiated by ex-OpenAI CTO Mira Murati, is forming a new team focused on AI technologies, with reports indicating that at least one OpenAI researcher is joining the venture.
  
  - The lab aims to undertake ambitious projects by leveraging the expertise of its founding members.
- **RAG to Advance Beyond Q&A**: There is growing speculation that Retrieval-Augmented Generation (RAG) will transition from primarily Q&A applications to more sophisticated report generation in the coming months, as highlighted in a post by [Jason Liu](https://jxnl.co/writing/2024/06/05/predictions-for-the-future-of-rag/).
  
  - The broader sentiment suggests that RAG's evolution will enhance how companies utilize AI in documentation and reporting.

 

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton Kernel Functionality and Conda Issues**: Developers are addressing [libstdc++](https://github.com/triton-lang/triton/issues/5136) compatibility issues in Triton when using Conda environments, aiming to resolve crashes encountered during `torch.compile` operations.
  
  - Discussions include optimizing [GEMM kernel designs](https://github.com/triton-lang/triton/issues/5138) for smaller sizes and addressing [warp memory alignment errors](https://github.com/triton-lang/triton/issues/5136) to enhance Triton's stability and performance.
- **Impact of torch.compile on Memory Usage**: Users report that **torch.compile()** leads to a **3-16%** increase in peak memory usage, contributing to **out-of-memory (OOM)** errors, particularly when dealing with **dynamic shapes**.
  
  - Profiling with **nsys** and **nvtx** ranges is recommended to analyze GPU memory allocations, although there's uncertainty if **CUDA graphs** in PyTorch exacerbate memory consumption without the `reduce-overhead` flag.
- **MI300X Achieves 600 TFLOPS FP16 Peak Throughput**: Performance benchmarks indicate that the **MI300X** reaches up to **600 TFLOPS** for **FP16** operations, though attempts to push beyond **800 TFLOPS** with **CK** optimizations have been unsuccessful.
  
  - A [YouTube talk](https://youtu.be/Lbm08twNTAQ?si=6Vwrkz8W0U2WTpf1&t=243) by Lei Zhang and Lixun Zhang highlights Triton's support for AMD GPUs, showcasing optimization strategies around chiplets to improve GPU performance.
- **Liger-Kernel v0.4.1 Released with Gemma 2 Support**: The latest **v0.4.1** release of [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.1) introduces **Gemma 2** support and a patch for **CrossEntropy** issues, addressing softcapping in fused linear cross entropy.
  
  - Enhancements also include fixes for **GroupNorm**, contributing to more efficient operations and validating the robustness of the updated kernel.
- **ThunderKittens Update: DSMEM Limitations and Synchronization**: Updates in **ThunderKittens** reveal that the **H100** GPU only supports **DSMEM** reductions for integer types, prompting discussions on optimizing semaphore operations and synchronization to prevent hangs.
  
  - Future pull requests aim to finalize testing code for integers, enhancing the kernel's reliability and performance in cooperative groups and semaphore synchronization contexts.

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad's Distributed Approach with Multi-node FSDP**: A user inquired about Tinygrad's current strategy for distributed computing, particularly regarding its handling of **FSDP** and support for **multi-node** setups. They referenced the [multigpu training tutorial](https://mesozoic-egg.github.io/tinygrad-notes/multigpu.html) for detailed insights.
  
  - Another user mentioned an open bounty on **FSDP** as a potential resource and discussed the scalability challenges of current implementations.
- **Data Handling in Cloud for Tinygrad**: Discussion highlighted that while cloud capabilities allow utilizing thousands of GPUs across different machines, optimal performance depends on **fast connectivity** and effective **all-reduce** implementations.
  
  - Concerns were raised about the efficiency of having a single machine orchestrate data management and processing during training runs.
- **Device-to-device Communication in Tinygrad**: George Hotz indicated that **device-to-device communication** is managed through Tinygrad's Buffer via the `transfer` function, suggesting potential ease in extending this to cloud setups.
  
  - He humorously noted it could be accomplished with just a few lines of code, indicating the simplicity of implementation.
- **Performance Optimization in Sharding in Tinygrad**: There was a discussion on the necessity of clarifying whether users are **machine-sharded** or **cloud-sharded** to prevent unexpected performance issues and costs during slower sync operations.
  
  - The conversation underscored the importance of efficient data handling strategies to maintain performance levels across different configurations.
- **Fixing Unsigned Tensor min() Bug in tinygrad**: A user identified a bug in the **min()** function for unsigned tensors when calculating minimum values with zeros, suggesting flips to resolve it. They referenced the [PR #7675](https://github.com/tinygrad/tinygrad/pull/7675/commits/6c1092cefc98c87edfe9516f3887d6789351140f).
  
  - *Rezvan* submitted a PR with failing tests, mentioning the complexity due to potential **infs** and **nans**.

 

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Enhancing AI Models' Date Accuracy**: Discussions revealed that models like **Gemini** and **Claude** often provide incorrect current dates, whereas **ChatGPT** maintains accurate date awareness. [Link to discussion](https://discord.com/channels/974519864045756446/998381918976479273/1305997985397608549).
  
  - One user attributed ChatGPT's accuracy to its superior system prompt configuration, enabling better inference of dates in various contexts.
- **ChatGPT o1-preview Shows Increased Creativity**: **ChatGPT o1-preview** is receiving positive feedback for its enhanced creativity and personalized responses compared to earlier versions. [Feedback thread](https://discord.com/channels/974519864045756446/1001151820170801244/1306032001496907836).
  
  - Users appreciate its ability to anticipate inputs, contributing to a more tailored interaction experience.
- **Implementing Scratchpad Techniques in LLMs**: Members are exploring the use of **scratchpad techniques** as a pseudo-CoT method, allowing LLMs to articulate their thought processes while generating solutions. [Discussion link](https://discord.com/channels/974519864045756446/1046317269069864970/1306011250769268850).
  
  - There is enthusiasm for integrating scratchpads into structured outputs to improve documentation and workflow consistency.
- **Challenges with Mobile Copy-Paste Functionality**: Ongoing **copy-paste** issues on mobile platforms are affecting user experience, with problems persisting for several weeks. [Issue report](https://discord.com/channels/974519864045756446/998381918976479273/1305997985397608549).
  
  - Users are seeking effective solutions to restore functionality and enhance mobile interaction capabilities.
- **VPN Usage for Circumventing Access Restrictions**: A discussion emphasized the legality of using **VPNs** to bypass internet restrictions, highlighting their role in maintaining access. [Conversation thread](https://discord.com/channels/974519864045756446/1001151820170801244/1306032001496907836).
  
  - Participants noted that current block configurations may be ineffective against users employing VPNs for intended purposes.

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **exllamav2 Elevates MAX Inference**: Members highlighted the [exllamav2 GitHub project](https://github.com/turboderp/exllamav2) as a valuable resource for enhancing LLM inference on **MAX**, emphasizing its clean and optimized codebase.
  
  - Key features include **ROCM support for AMD** and efficient handling of multimodal models, positioning exllamav2 as a strong candidate for deeper integration with the MAX platform.
- **Mojo JIT Compiler Optimization**: The community discussed the feasibility of shipping the **Mojo JIT compiler** by ensuring a compact binary size and interoperability with precompiled binaries.
  
  - A member emphasized that while **MLIR can be shipped**, the compiler is crucial for achieving native code execution without exposing source for all dependent applications.
- **MAX Platform Capabilities**: **MAX** was introduced as a comprehensive suite of APIs and tools for building and deploying high-performance AI pipelines, featuring components like the **MAX Engine** for model execution.
  
  - The [MAX documentation](https://docs.modular.com/max/#how-max-works) was shared, showcasing its capabilities in deploying low-latency inference pipelines effectively.
- **UnsafePointer Risks in Mojo**: **UnsafePointer** in Mojo was flagged for its potential to invoke undefined behavior, leading to memory safety issues as detailed by a community member.
  
  - Another member noted that Mojo enforces stricter pointer rules compared to C/C++, aiming to minimize risks such as type punning and enhance overall memory safety.
- **Mana Project Naming Trends**: Members humorously discussed the frequent use of the name **Mana**, referencing projects like [mana.js](https://github.com/bjorn/mana.js/) and [3rd-Eden's mana](https://github.com/3rd-Eden/mana).
  
  - The conversation reflected on the trend of adopting 'Mana' in project names, indicating a broader cultural influence in naming conventions within the tech community.

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Vocera Launch on Product Hunt**: **Vocera** was launched on [Product Hunt](https://www.producthunt.com/posts/vocera), enabling AI developers to test and monitor voice agents **10X faster**.
  
  - The team is seeking feedback to boost [Vocera's visibility](https://www.producthunt.com/posts/vocera) within the AI community.
- **GenAI Pipelines with LlamaIndex**: Learned how to build robust **GenAI pipelines** using [LlamaIndex](https://x.com/LlamaIndex), [Qdrant Engine](https://twitter.com/qdrant_engine), and [MLflow](https://twitter.com/MLflow) to enhance RAG systems.
  
  - The [step-by-step guide](https://t.co/aZ4GIyGRQM) covers streamlining RAG workflows, maintaining performance across model versions, and optimizing indexing for efficiency.
- **RAG vs Reporting Debate**: A debate emerged comparing **RAG** (Retrieval-Augmented Generation) with traditional reporting, noting that reports account for only **10%** of problem-solving in corporations.
  
  - `@jxnlco` argued that reports are more impactful, emphasizing that **information retrieval** is key to effective report generation.
- **Dynamic Section Retrieval in RAG**: Introduced a new **dynamic section retrieval** technique in RAG, allowing full sections to be retrieved from documents instead of fragmented chunks.
  
  - This method addresses community concerns about multi-document RAG, as discussed in [this article](https://t.co/vP2J2arhf4).
- **Chatbots in Corporate Settings**: Members observed that upper management favors **report formats** over chatbot interactions within corporations.
  
  - Despite this preference, chatbots are recognized as effective tools for conducting internal searches.

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Rerank API Best Practices**: Users are seeking **best practices** for the `query` field in the **v2/rerank API**, noting significant variations in `relevanceScore` with slight query changes. Reference the [Rerank Best Practices](https://docs.cohere.com/docs/reranking-best-practices#queries) for optimal endpoint performance.
  
  - Examples include a `query` for **'volume rebates'** achieving a score of ~0.998 compared to ~0.17 for **'rebates'**, causing confusion about the model's responsiveness to query semantics.
- **Production API Key Upgrade**: A user reported **upgrading to a production API key**, anticipating a more stable experience with **Cohere's services** once current issues are resolved.
  
  - This upgrade indicates a commitment to utilizing Cohere’s offerings, dependent on the resolution of ongoing API errors.
- **Benchmarking Vision Language Action Models**: A new paper titled [*Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks*](https://arxiv.org/abs/2411.05821) was released, showcasing collaboration between **Manifold**, **Georgia Tech**, **MIT**, and **Metarch AI**.
  
  - The research evaluates emerging Vision Language Action models, including **GPT4o**, on their ability to control robots across **20 real-world tasks**. Explore more on the [Multinet website](https://multinet.ai/static/pages/Multinetv01.html) and the [code repository](https://github.com/ManifoldRG/MultiNet/tree/main).
- **ICS Support for Events**: A user emphasized the necessity of implementing **ICS file support** for managing numerous events hosted on the Discord server.
  
  - The request was well-received by members, with positive feedback supporting the addition of this feature.
- **File Content Viewing Feature**: A new feature was introduced to **view the content** of uploaded files within the toolkit, enhancing file management capabilities.
  
  - The feature was met with enthusiasm from members, who expressed appreciation for the improved functionality.

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Docker Image Tagging for Releases**: Docker images for the **main branch** have been built, with a reminder to tag them for version releases. A member emphasized the importance of proper tagging for organized version control and upcoming releases.
  
  - This practice ensures traceability for each release, as detailed in the [latest pull request](https://github.com/axolotl-ai-cloud/axolotl/pull/2051/files).
- **Qwen2.5 Coder Size Insights**: A member shared a [YouTube video](https://youtu.be/WPziCratbpc) comparing different sizes of **Qwen2.5 Coder**, discussing their performance metrics in detail.
  
  - The video provides an in-depth analysis, aiding users in selecting the appropriate model size for their specific needs.
- **Qwen2.5 Performance on NVIDIA 3090**: **Qwen2.5** is running on an **NVIDIA 3090**, resulting in enhanced generation speed. This hardware configuration underscores the performance gains achievable for demanding models.
  
  - Users noted significant improvements in generation times, highlighting the benefits of high-end GPUs in model deployments.
- **Comparing Qwen2.5 Coder with GPT4o and Claude 3.5 Sonnet**: A [YouTube video](https://youtu.be/Xs0EkLYu6hw?si=95JJjVKRPknvEUsw) titled **'Qwen2.5 Coder 32B vs GPT4o vs Claude 3.5 Sonnet'** was shared to compare these models.
  
  - The video aims to determine the superior model among them, offering a comprehensive analysis of their capabilities.
- **Axolotl Version 0.5.0 Launch**: The team announced the release of **Axolotl version 0.5.0**, now installable via `pip install axolotl`. Updates include improvements and new features detailed on the [GitHub release page](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.5.0).
  
  - Community members celebrated the release, expressing excitement and pledging support for ongoing enhancements.

 

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Nous Research Introduces Forge Reasoning API**: Nous Research has unveiled the [Forge Reasoning API](https://nousresearch.com/introducing-the-forge-reasoning-api-beta-and-nous-chat-an-evolution-in-llm-inference/) in beta, promising significant advancements in LLM inference capabilities.
  
  - This development marks a crucial step in enhancing reasoning processes within AI systems, showcasing a blend of newer models and optimized techniques.
- **Nous Chat Gets an Upgrade**: Accompanying the Forge API, **Nous Chat** is set to evolve, incorporating advanced features that improve user interaction and accessibility.
  
  - With this evolution, the emphasis lies on delivering a richer conversation experience powered by enhanced LLM technologies and methodologies.
- **DSPY Comparative Analysis Discussions**: Members discussed experiences with **DSPY** for conducting a comparative analysis on **zero shot** and **few shot prompting** in specific domains.
  
  - One member asked others about their use of the GitHub template to facilitate this analysis.
- **Shared DSPY Resources**: A member shared a link to a [Colab notebook](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb) to help others get started with DSPY.
  
  - Another member referenced a different notebook and highlighted its potential usefulness for their own project involving a code similarity tool.
- **Evaluating Tools with LLM Approaches**: A member mentioned evaluating **zero shot** versus **few shot** prompting in their attempts to create a code similarity tool using **LLM**.
  
  - They referred to another GitHub resource that they worked on to compare approaches and outcomes.

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Excites Community**: Members are thrilled about the latest **Open Interpreter** updates, particularly the **streamed responses handling** feature, which [enhances user experience](https://discord.com/channels/1146610656779440188/1147665339266650133/1306047445276295188).
  
  - One member commented that *'Open Interpreter is awesome!'*, prompting discussions on the potential for **building text editors** in future collaborations.
- **OpenCoder: Revolutionizing Code Models**: The [OpenCoder YouTube video](https://www.youtube.com/watch?v=DurejOD5FTk) showcases **OpenCoder**, an open-source repository aimed at developing superior code language models with advanced capabilities.
  
  - Viewers were intrigued by OpenCoder's potential to **surpass existing models**, discussing its impact on the code modeling landscape.
- **Forecasting the AI Bubble Burst**: A post warns that the **AI bubble is about to pop**, drawing parallels to the **1999 dot-com bubble**, especially in terms of massive **GPU investments** failing to yield proportional revenues.
  
  - The article details the risk of **$600 billion** in GPU spend against a mere **$3.4 billion** in revenue, suggesting a precarious outlook for the AI sector.
- **Comparing AI and Dot-Com Crashes**: Discussions highlight that the ongoing infrastructure buildup in AI mirrors strategies from the dot-com era, with companies heavily investing in hardware without clear monetization.
  
  - The risk of repeating past **Pets.com**\-like failures is emphasized, as firms chase theoretical demand without proven profit pathways.

 

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Vision Language Action Models Released**: A new paper titled '[Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks](https://arxiv.org/abs/2411.05821)' evaluates the performance of **Vision Language Models** across 20 different real-world tasks, showcasing collaborations among **Manifold**, **Georgia Tech**, **MIT**, and **Metarch AI**.
  
  - The work aims to profile this emerging class of models like **GPT4o**, marking a first step in a broader benchmark for **multimodal action models**.
- **Watermark Anything Tool Released**: The project '[watermark-anything](https://github.com/facebookresearch/watermark-anything)' provides an official implementation for **watermarking** with localized messages. This model is noted to have only **1M parameters**, potentially allowing it to be integrated into various **AI generators** quickly.
  
  - The lightweight architecture enables rapid deployment across different AI generation platforms, facilitating seamless integration.
- **EPOCH 58 COCK Model Updates**: A member shared updates about **EPOCH 58 COCK**, noting improvements with **vit-s** at **60M parameters** and enhanced model features.
  
  - *They remarked that* ***legs are coming in*** *and the* ***cockscomb is becoming more defined****, signaling positive progress in model capabilities.*
- **Advancements in Robotic Learning Tasks**: Discussions highlighted progress in **Robotic Learning Tasks**, particularly in applying **Vision Language Action Models** to enhance **robot control** and **task automation**.
  
  - Community members debated the challenges and potential solutions for deploying these models in real-world robotic systems, citing ongoing experiments and preliminary results.
- **AI Generators Performance Enhancements**: Participants discussed the latest improvements in **AI Generators Performance**, focusing on increased **model efficiency** and **output quality**.
  
  - Specific benchmarks and performance metrics were analyzed to assess the advancements, with emphasis on practical implementations.

 

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Utilizing Tape for Agent-Human Communication**: A member inquired about using **Tape** as a medium for communication between humans and agents, seeking appropriate [documentation](https://discord.com/channels/1280234300012494859/1280370030609170494/1306007383268524074).
  
  - This led to a request for guidance on publishing an agent's tape entries encountering errors to a queue.
- **Sharing Resources on TapeAgents Framework**: In response to **TapeAgents** queries, a member shared a [GitHub intro notebook](https://github.com/ServiceNow/TapeAgents/blob/main/examples/intro_clean.ipynb) and a relevant [paper](https://www.servicenow.com/research/TapeAgentsFramework.pdf).
  
  - *The member stated that they have read all provided resources,* expressing that they had already reviewed the suggested materials.

 

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Latent Toys website launch**: A member shared the newly created [Latent Toys](https://latent.toys/), highlighting it as a noteworthy project.
  
  - A friend was behind the development of the site, further adding to its significance.
- **Community discussion on Latent Toys**: Members discussed the launch of [Latent Toys](https://latent.toys/), emphasizing its importance within the community.
  
  - The announcement generated interest and curiosity about the offerings of the new site.

 

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla Submits PR for Writer Models and Palmyra X 004**: A member announced the submission of a [PR](https://github.com/ShishirPatil/gorilla/pull/755) to add support for **Writer models** and **Palmyra X 004** to the leaderboard.
  
  - They expressed *gratitude* for the review and shared an image preview linked to the PR, highlighting community collaboration.
- **Community Response to Gorilla PR**: Another member promptly responded to the [PR submission](https://github.com/ShishirPatil/gorilla/pull/755), indicating they will review the changes.
  
  - Their acknowledgment of '*Thank you!*' underscored active community engagement.

 

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Legacy Models Deprecation**: Members expressed their frustration regarding the **deprecation of legacy models**, stating that the new models are not providing the same output quality.
  
  - *This deprecation is hugely disruptive* for users who have relied on the older models for almost two years.
- **Transition to Open Source Solutions**: Users are scrambling to convert to an **open source solution**, while they have been willing to pay for the old models.
  
  - *How can we be sure AI21 won't deprecate the new models in the future too?* highlights their concerns about the stability of future offerings.

 

---

The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

 

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1305985484962660414) (160 messages🔥🔥):

> - `Qwen Coder Models`
> - `Multi-GPU Training`
> - `Dataset Formatting`
> - `Finetuning VRAM Requirements`
> - `Inference System Tokens`

- **Discussion on Qwen Coder models deployment**: Members discussed the current development and testing of the Qwen Coder models, with some expressing interest in their performance and potential evaluation metrics.
  
  - There are mentions of files and fixes available on Unsloth with suggestions to run evaluations similar to other models.
- **Multi-GPU Training Limitations**: Users explored the potential for training large models like Qwen 2.5 using multiple GPUs, specifically mentioning MI300X and VRAM needs.
  
  - It was noted that Unsloth may be more efficient with a single GPU setup rather than multi-GPU configurations due to memory efficiency.
- **Dataset Formatting and Input Handling**: There were discussions about how to effectively format datasets for finetuning, specifically concerning separator tokens and input/output differentiation.
  
  - Suggestions included using special tokens like '### response' or `---` to inform the model when to switch between user input and model output.
- **Finetuning Large Models and VRAM Requirements**: Users inquired about the required VRAM for finetuning 405B models, with some proposing configurations involving high VRAM GPUs.
  
  - The practicality of testing configurations on platforms like RunPod was suggested, with discussions around the efficiency of Unsloth training.
- **Inference System Tokenization**: Members questioned the necessity of explicit separators in user inputs during inference, contemplating whether systems like LM Studio handle this implicitly.
  
  - It was clarified that various models adopt different approaches regarding start and end tokens for handling user input and assistant responses.

**Links mentioned**:

- [WeightWatcher: Data-Free Diagnostics for Deep Learning](https://weightwatcher.ai/): no description found
- [Tweet from Daniel Han (@danielhanchen)](https://x.com/danielhanchen/status/1856442699689414970): Bug fixes & analysis for Qwen 2.5: 1. Pad_token should NOT be <|endoftext|> Inf gens 2. Base <|im_start|> <|im_end|> are untrained 3. PCA on embeddings has a BPE hierarchy 4. YaRN ...
- [Goku Anime GIF - Goku Anime Super Saiyan - Discover & Share GIFs](https://tenor.com/view/goku-anime-super-saiyan-gif-5063009): Click to view the GIF
- [Massed Compute](https://massedcompute.com/): Massed Compute is a cloud compute provider that serves cutting-edge GPUs without bulky contracts and unnecessary up-sells.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gpw8ls/bug_fixes_in_qwen_25_coder_128k_context_window/): no description found
- [subreddits](https://www.reddit.com/r): no description found
- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1306122839417688165) (2 messages):

> - `User reactions`
> - `Discord interactions`

- **Shared 'Womp Womp' Moment**: Members expressed a mutual reaction described as 'womp womp', indicating a shared sentiment of disappointment or humor.
  
  - This informal communication style highlights the casual nature of discussions within the Discord community.
- **Echoing Community Feelings**: Another member chimed in agreeing with the initial sentiment, stating, 'I do same.'
  
  - Such responses reflect a collaborative and friendly atmosphere among members discussing their reactions.

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1305993138959159316) (178 messages🔥🔥):

> - `Gemma 2B RAM usage`
> - `Flash Attention installation issues`
> - `RAG experience and usage`
> - `Ollama model management`
> - `Training on responses only`

- **Gemma 2B RAM usage during long runs**: Users discussed experiencing **consistent RAM usage increase** while running longer jobs with **Gemma 2B**, questioning if evaluation steps might be affecting performance.
  
  - One member suggested training with **0 steps** to mitigate excessive resource consumption.
- **Flash Attention installation troubles**: A member reported difficulties with the **Flash Attention installation** hanging indefinitely, prompting suggestions from others about checking command execution and running environments.
  
  - Another member inquired if the issue was related to the command **'run all'** during the setup process.
- **Exploring RAG for long-term memory**: Inquiry about **RAG (Retrieval-Augmented Generation)** was made, along with requests for user experiences and guidance on using it for long-term data retention.
  
  - A user recommended **Dify** as a simple alternative for implementing RAG.
- **Managing models in Ollama**: There were discussions on how to upload and manage models in **Ollama**, including commands used for copying and pushing model updates.
  
  - Users confirmed successful **model uploads and pushes**, navigating challenges of namespace permissions and app connections.
- **Understanding training on assistant responses**: Clarification was sought regarding the `train_on_responses_only` function, specifically whether all previous messages in a chat are considered during training.
  
  - It was noted that the model **masks user inputs**, allowing assistant responses to be predicted based only on the context of prior responses.

**Links mentioned**:

- [raultherockstar/nyayasathi](https://www.ollama.com/raultherockstar/nyayasathi): Get up and running with large language models.
- [facebook/nougat-small · Hugging Face](https://huggingface.co/facebook/nougat-small): no description found
- [Paper page - Nougat: Neural Optical Understanding for Academic Documents](https://huggingface.co/papers/2308.13418): no description found
- [Nougat - a Hugging Face Space by tomriddle](https://huggingface.co/spaces/tomriddle/nougata): no description found
- [Computer Vision API - OCR bounding boxes | Microsoft Community Hub](https://techcommunity.microsoft.com/discussions/azure/computer-vision-api---ocr-bounding-boxes/71774): I'm building an API for a customer than leverages computer vision to analyse images. I am trying to get it to analyse handwriting on the white...

---

### **Unsloth AI (Daniel Han) ▷ #**[**showcase**](https://discord.com/channels/1179035537009545276/1179779344894263297/1306098409291386912) (2 messages):

> - `Optillm release`
> - `Hyperstack legitimacy`

- **Optillm Launches Exciting New Features**: The latest release of [optillm](https://github.com/codelion/optillm) introduces a local inference server that allows loading any HF model and LoRA adapters, enhancing usability for fine-tuned Unsloth models.
  
  - This update also enables dynamic adapter switching during inference and supports advanced decoding techniques like **cot_decoding** and **entropy_decoding** while utilizing the standard **OpenAI client SDK**.
- **Inquiry on Hyperstack's Legitimacy**: A question was raised about the legitimacy of **Hyperstack**, indicating some community interest or skepticism regarding this platform.
  
  - No specific details or consensus about Hyperstack’s credibility were provided in the discussion.

 

**Link mentioned**: [GitHub - codelion/optillm: Optimizing inference proxy for LLMs](https://github.com/codelion/optillm): Optimizing inference proxy for LLMs. Contribute to codelion/optillm development by creating an account on GitHub.

 

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1305987738201358466) (270 messages🔥🔥):

> - `Qwen Model Performance`
> - `AI Project Ethical Considerations`
> - `GPU Specifications and Performance`
> - `Langchain and Hugging Face Integration`
> - `Deep Learning Model Suggestions`

- **Qwen Model Shows Variable Output**: Users reported that the Qwen model's performance in generating text can vary significantly, with comparisons made to other models like Ollama indicating that responses from Qwen often may hallucinate or lack quality.
  
  - Tweaking parameters like repetition penalty and adjusting token lengths were suggested to enhance output quality.
- **Ethical Implications of AI Projects**: Discussion arose regarding the ethical dilemmas associated with AI projects aimed at extracting reflection data and their potential misuse if widely available.
  
  - Participants expressed the need for a careful discussion on the ethical ramifications while acknowledging the project's valid applications, especially for law enforcement.
- **Questions Surrounding GPU Capabilities**: The NVIDIA 4060 Ti 16GB was debated upon as potentially a great option for its price and memory capacity, despite opinions that it could be slower than previous models like the 3060 Ti in some scenarios.
  
  - Users noted the importance of relative performance metrics and price-to-performance ratio when considering new GPU purchases.
- **Langchain Usage with Hugging Face**: Users discussed approaches to configuring the Hugging Face models using Langchain, with specific suggestions on adjusting parameters for better model performance in text generation tasks.
  
  - The integration allows for better handling of model invocation with parameters tailored for desired output characteristics.
- **Model Recommendations for Diffusers**: Suggestions were made for the best models to use within the Diffusers library, highlighting the strengths of the Flux model over others like sd3.5 large.
  
  - Users agreed on the need for ongoing experimentation to determine which models performed best based on specific use cases.

**Links mentioned**:

- [FlUX WebUI - a Hugging Face Space by nroggendorff](https://huggingface.co/spaces/nroggendorff/flux-web): no description found
- [InstructIR - a Hugging Face Space by marcosv](https://huggingface.co/spaces/marcosv/InstructIR): no description found
- [Facepalm GIF - Facepalm - Discover & Share GIFs](https://tenor.com/view/facepalm-gif-4576513125411549651): Click to view the GIF
- [Lol Goonies GIF - Lol Goonies The Goonies - Discover & Share GIFs](https://tenor.com/view/lol-goonies-the-goonies-gif-17881913): Click to view the GIF
- [Sus Cat 2 Suspicious Cat GIF - Sus Cat 2 Suspicious cat The cat looks suspiciously - Discover & Share GIFs](https://tenor.com/view/sus-cat-2-suspicious-cat-the-cat-looks-suspiciously-cat-sits-in-front-of-food-the-ginger-cat-is-watching-gif-14890167989997543813): Click to view the GIF
- [Dog Doggo GIF - Dog Doggo Cute - Discover & Share GIFs](https://tenor.com/view/dog-doggo-cute-math-formulas-gif-17580986): Click to view the GIF
- [Alien Talking GIF - Alien Talking Alien talking - Discover & Share GIFs](https://tenor.com/view/alien-talking-alien-talking-keep-yapping-your-mouth-alien-babbling-gif-17459379075847540969): Click to view the GIF
- [Hail Zorp Parks And Rec GIF - Hail Zorp Parks And Rec April - Discover & Share GIFs](https://tenor.com/view/hail-zorp-parks-and-rec-april-gif-14789564): Click to view the GIF
- [Its Classified Tom Cruise GIF - Its Classified Tom Cruise Classified - Discover & Share GIFs](https://tenor.com/view/its-classified-tom-cruise-classified-private-secret-gif-9579704): Click to view the GIF
- [Weird Weirdly GIF - Weird Weirdly Specific - Discover & Share GIFs](https://tenor.com/view/weird-weirdly-specific-gif-19034416): Click to view the GIF
- [Qwen/Qwen2.5-Coder-7B-Instruct-GGUF at main](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/tree/main): no description found
- [GeForce 40 series - Wikipedia](https://en.wikipedia.org/wiki/GeForce_40_series#Products): no description found
- [GeForce 30 series - Wikipedia](https://en.wikipedia.org/wiki/GeForce_30_series#Details): no description found
- [stabilityai/stable-diffusion-xl-base-1.0 at main](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main): no description found
- [facebook/dinov2-large at main](https://huggingface.co/facebook/dinov2-large/tree/main): no description found
- [InstantX/InstantIR at main](https://huggingface.co/InstantX/InstantIR/tree/main/models): no description found

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1305999130719551489) (12 messages🔥):

> - `LLM for E-commerce Branding`
> - `Cell Journal Research Confirmation`
> - `Learning Machine Learning`
> - `Cross-Posting Concerns`

- **Seeking LLM for Realistic Baby Clothes Models**: A member is looking for an LLM that generates ultra-realistic AI models wearing their brand clothes for their e-commerce site focusing on baby clothes.
  
  - This garnered mixed reactions, with curiosity about the authenticity of the title and its practical application.
- **Cell Journal Research Credibility**: A member confirmed the validity of a research article published by **Cell Journal** that sparked interest and skepticism.
  
  - They provided a [link to the article](https://www.cell.com/cell/abstract/S0092-8674(24)01152-8?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867424011528%3Fshowall%3Dtrue) while noting that it's **paywalled**.
- **Introduction to Machine Learning**: A member expressed interest in starting their journey into **Machine Learning** and asked for guidance on where to begin.
  
  - Another member reminded them not to cross-post in the channel, suggesting that the conversation may have been duplicated.

 

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1306028954238586911) (28 messages🔥):

> - `LightRAG Article`
> - `Sulie Foundation Model`
> - `ZeroGPU Debugging`
> - `Benchmarking VLA Models`
> - `PromptDX Usability`

- **Introducing LightRAG for Retrieval**: An article was shared detailing **LightRAG**, which includes code evaluation that compares Naive RAG with local, global, and hybrid approaches. The writer aims to highlight the advantages of using LightRAG in various retrieval tasks.
  
  - You can read the full article [here](https://www.linkedin.com/posts/isham-rashik-5a547711b_introducing-lightrag-a-new-era-in-retrieval-activity-7262085232743342080-xgdo?utm_source=share&utm_medium=member_desktop).
- **Sulie: A New Model for Time Series Forecasting**: A newly released foundation model called **Sulie** for time series forecasting aims to simplify the automation of LoRA fine-tuning and covariate support. The team seeks feedback and encourages users to check their work on [GitHub](https://github.com/wearesulie/sulie).
  
  - They humorously highlight common frustrations faced by data teams, comparing issues to a 'chocolate teapot' for zero-shot performance.
- **ZeroGPU Debugging Insights**: A member discussed their experience troubleshooting **ZeroGPU** on Hugging Face Spaces, specifically tackling NaN tensors and Pickle errors. Their findings are documented in a detailed blog post [here](https://huggingface.co/blog/rrg92/zero-gpu-nan-and-pickle-errors).
  
  - The author shares newfound knowledge about Python's workings related to Hugging Face and hopes to help others facing similar issues.
- **Benchmarking VLA Models for Robotics**: A collaborative research paper titled **Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks** was released, aiming to evaluate the performance of VLA models like GPT4o. This effort represents the initial phase of a wider benchmark for a new multimodal action model class.
  
  - For more details, visit the [website](https://multinet.ai/static/pages/Multinetv01.html) or check out the code on [GitHub](https://github.com/ManifoldRG/MultiNet/tree/main).
- **Exploring PromptDX Usability**: A discussion on **PromptDX** revealed its potential to decouple prompts from code, offering better readability and structured management. Users can import existing markdown files as components to streamline the prompt store, enhancing usability.
  
  - The conversation highlighted the importance of having a system that can manage prompts effectively, with users expressing interest in the possibilities PromptDX affords for organizing prompts.

**Links mentioned**:

- [Recipes | PromptDX](https://puzzlet-ai.github.io/promptdx/docs/recipes#chatbot): Basic
- [Solving NaN Tensors and Pickling Errors in a ZeroGPU Space](https://huggingface.co/blog/rrg92/zero-gpu-nan-and-pickle-errors): no description found
- [GitHub - wearesulie/sulie: Access to Sulie foundation models for time-series forecasting 📈](https://github.com/wearesulie/sulie): Access to Sulie foundation models for time-series forecasting 📈 - wearesulie/sulie
- [Tweet from harsh (@HarshSikka)](https://x.com/HarshSikka/status/1856739777208574151)): Excited to share our new paper "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks" We evaluate how well VLM & VLA models can control robots across 20 different real-wor...

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1306050626421133374) (3 messages):

> - `Reading time adjustment`
> - `User timezone concerns`

- **Reading Time Adjusts to User's Timezone**: A member inquired whether the reading time displayed is in US time.
  
  - Another responded that it should adjust to whatever **timezone** is set on the user's computer or Discord.
- **Concerns about Early Morning Reading**: A member expressed concern about being awake at **04:00 AM** for reading time.
  
  - This highlights the potential impacts of timezone settings on user participation.

 

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1306371650769653904) (2 messages):

> - `Open3D-ML Development`
> - `O3D Historical Context`
> - `3D Object Classification Techniques`

- **Open3D-ML looks promising**: A member shared their enthusiasm for [Open3D-ML](https://github.com/isl-org/Open3D-ML), an extension of Open3D designed for 3D Machine Learning tasks.
  
  - This repository seems to have significant potential for those interested in enhancing their 3D ML capabilities.
- **O3D still evokes nostalgia**: A member reminisced about the historical significance of **Open3D** and its launch alongside **AlexNet**, noting their own authorship of a related book.
  
  - They expressed amazement at the evolution of Open3D now featuring a machine learning library, despite its initial underperformance compared to **WebGL**.
- **Python script for 3D object classification**: A member suggested creating a Python script in **Blender** to generate images of 3D objects from various axes for classification purposes.
  
  - This technique could be employed to compare classifications across three views, adding a layer of validation to the results.

**Links mentioned**:

- [GitHub - isl-org/Open3D-ML: An extension of Open3D to address 3D Machine Learning tasks](https://github.com/isl-org/Open3D-ML): An extension of Open3D to address 3D Machine Learning tasks - isl-org/Open3D-ML
- [The o3d Bible by Kara Rawson](https://www.scribd.com/document/63892020/The-o3d-Bible-by-Kara-Rawson): This document provides a summary of the Google O3D API library. It includes an introduction, installation instructions, system requirements, supported graphics hardware, and an overview of the program...

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1306117375665639506) (4 messages):

> - `Legal Document Retrieval`
> - `Enhancing Retrieval with Graphs`
> - `Fine-tuning Tokenizers for Vietnamese Models`

- **Challenges in Legal Document Retrieval**: A member highlighted issues with their legal document retrieval task, stating that the evaluation results with **MRR@10** are quite poor despite having fine-tuned embeddings and reranking.
  
  - Suggestions included evaluating recent advancements in retrieval-enhancing methodologies and models, particularly as it relates to **legal domains**.
- **Graph Techniques for Retrieval Enhancement**: A member expressed interest in learning about how to enhance the retrieval stage using **FAISS** with graph techniques, although they are unsure where to start.
  
  - Another member mentioned that the landscape of retrieval-augmented generation (RAG) has significantly improved over the past **six years**.
- **Fine-tuning Tokenizers for Vietnamese Legal Data**: In addressing their training on a **Vietnamese Legal dataset**, a member questioned the difference between adding a new token to a pre-trained tokenizer and fine-tuning a new tokenizer.
  
  - They inquired whether the fine-tuning approach was feasible and **approachable** for their task.

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1306216548138094602) (8 messages🔥):

> - `SDXL Lightning Model`
> - `Realtime Image Generation Workflows`
> - `Training Diffusion Models`

- **SDXL Lightning demonstrates fast image generation**: **SDXL Lightning** or **sd1.5 models** can generate images in just a few seconds on standard GPUs, making them ideal for prompt-based image creation.
  
  - One user shared that variants like **turbo/lightning/lcm** can produce images in real-time on powerful hardware.
- **Realtime Turbo Workflow for sd 1.5**: A user shared a detailed workflow in [this reddit post](https://www.reddit.com/r/StableDiffusion/comments/187ps59/got_realtime_turbo_workflow_working_sd_15_lcm_and/) for using **sd 1.5** with **LCM** for real-time image generation in **ComfyUI**.
  
  - They detailed their configuration for optimizing image quality and functionality, recommending specific settings like **10 steps** and **1.0 or 2.0 cfg**.
- **Quality concerns with SDXL Turbo**: A user expressed dissatisfaction with the image quality from **SDXL Turbo**, preferring the *higher quality outputs* from their **sd 1.5** setup instead.
  
  - They noted that their current configuration felt as fast as turbo but delivered better results, particularly at higher resolutions like **768x768** on a **4090**.
- **Challenges in training diffusion models**: One participant reported difficulties in training various diffusion models, experiencing class imbalance in generated images despite using uniform sampling.
  
  - *They sought advice* on how to better align the model with the **underlying distribution of their data**.

 

**Link mentioned**: [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/187ps59/got_realtime_turbo_workflow_working_sd_15_lcm_and/): no description found

 

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1306044527236354059) (1 messages):

> - `Advertising Experiment`
> - `User Experience Assurance`

- **Perplexity begins Advertising Experiment**: To fulfill their mission, Perplexity will start experimenting with ads formatted as sponsored follow-up questions, beginning this week in the **US**.
  
  - This initiative aims to build a **robust and self-sustaining business** while maintaining that advertiser content will not influence the answers provided.
- **Blog Post on Advertising Perspective**: Perplexity encourages users to read their [blog post](https://www.perplexity.ai/hub/blog/why-we-re-experimenting-with-advertising) for detailed insights on their advertising strategy.
  
  - The blog outlines their commitment to ensuring that the **content remains unbiased** despite the introduction of ads.

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1305985611290644584) (282 messages🔥🔥):

> - `Perplexity AI Subscription Model`
> - `Perplexity Ads Implementation`
> - `Model Selection Issues`
> - `User Experience Concerns`
> - `Fractal Machine Learning`

- **Perplexity AI Subscription Model Under Scrutiny**: Users are questioning the value of the Perplexity Pro subscription, especially with the potential introduction of ads alongside the paid service.
  
  - Many have expressed dissatisfaction, stating that they would not continue the subscription if ads are included.
- **Confusion Over Ads in Pro Subscription**: There is uncertainty among users about whether ads will appear in the Pro version, with calls for clarification from the Perplexity team.
  
  - Users are seeking sources that confirm the presence of ads for Pro subscribers, indicating rising concern about the impact on the user experience.
- **Ongoing Model Selection Issues**: Multiple users reported persistent issues with selecting different models in Perplexity, often reverting to GPT-4o regardless of the chosen option.
  
  - This bug has been impacting their workflow significantly, raising frustrations among Pro subscribers who expect reliable access to Claude models.
- **User Concerns on Experience Quality**: Concerns have been raised about the overall user experience with Perplexity, particularly regarding the introduction of ads and the loss of expected service quality.
  
  - Users highlight the importance of maintaining a clear and uncluttered search experience, fearing the platform may resort to similar tactics as larger search engines.
- **Interest in Fractal Machine Learning for AI**: A member proposed exploring fractals to enhance AI performance, discussing potential applications in language models and suggesting collaboration with experts in the field.
  
  - The community showed interest, with members sharing various sources about the innovative use of fractals in machine learning.

**Links mentioned**:

- [The AI Bubble is About to Pop. Here's Who Dies First](https://chrisbora.substack.com/p/the-ai-bubble-is-about-to-pop-heres): The $600B Bloodbath Nobody's Ready For (And The Hidden $3T Opportunity)
- [Genspark Autopilot Agent](https://www.genspark.ai/autopilotagent_viewer?id=b1190308-5abd-4be5-baba-4244aab36c81): no description found
- [Tweet from Greg Feingold (@GregFeingold)](https://x.com/gregfeingold/status/1856088784699277668?s=61): By popular demand, we’re expanding our campus strategist program to Canada 🇨🇦 If you’re interested in applying, or know someone who would be a good fit, please reach out! Quoting Perplexity (@per...
- [Unveiling the Potential of Fractal Machine Learning - GeeksforGeeks](https://www.geeksforgeeks.org/unveiling-the-potential-of-fractal-machine-learning/): A Computer Science portal for geeks. It contains well written, well thought and well explained computer science and programming articles, quizzes and practice/competitive programming/company interview...
- [On The Potential of The Fractal Geometry and The CNNs Ability to Encode it](https://arxiv.org/abs/2401.04141): The fractal dimension provides a statistical index of object complexity by studying how the pattern changes with the measuring scale. Although useful in several classification tasks, the fractal dimen...

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1305998922153595010) (7 messages):

> - `4-7-8 Breathing Technique`
> - `TCP/IP Guide for IT Aspirants`
> - `History of the Paralympics`
> - `Differences in AI Models`
> - `Nostalgia for Childhood Games`

- **Learn the 4-7-8 Breathing Technique**: Explore the [4-7-8 breathing techniques](https://www.perplexity.ai/search/4-7-8-breathing-techniques-wKY17FJUQrS46xGcxkKyXw) to enhance relaxation and manage stress effectively.
  
  - For an in-depth understanding, check the detailed guide on [4-7-8 breathing](https://www.perplexity.ai/page/4-7-8-breathing-technique-e8EpnEG3Q3SMg9OaIVejOQ).
- **TCP/IP Protocols Explained for New Italians**: A basic guide for new Italians on the **ISO/OSI** and **TCP/IP** protocols can be found [here](https://www.perplexity.ai/page/il-livello-iso-osi-e-tcp-ip-sp-kbzTcdZqShqc2ZFLxatp0g).
  
  - This resource aims to simplify complex networking concepts for aspiring IT professionals.
- **The Fascinating History of the Paralympics**: Delve into the [history of the Paralympics](https://www.perplexity.ai/search/historia-da-paraolimpiada-quai-DqHK2XMlTiC5Kg84pS3sug) and discover its evolution and impact on athletes with disabilities.
  
  - This exploration reveals the cultural significance and milestones of the event over the years.
- **Comparing AI Models: What Makes Perplexity Unique?**: Discover how [Perplexity AI](https://www.perplexity.ai/search/how-is-perplexity-ai-different-PF1ebdmMSci1d2dIu6UCiQ) distinguishes itself in the AI landscape through innovative features and user experience.
  
  - This discussion highlights key differences that could inform users' choices regarding AI tools.
- **Nostalgia: Why Childhood Games Feel Better**: A study on *why childhood games seem better* aims to explore the emotional connections tied to nostalgic experiences found [here](https://www.perplexity.ai/page/why-childhood-games-seem-bette-ntiCEDDeQcKdT95NH09nfQ).
  
  - It offers insights into how memory and emotion shape our perception of gaming in childhood.

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1306343659079209000) (4 messages):

> - `search_domain_filter`
> - `Vercel AI SDK with Perplexity`
> - `Reddit citations issues`

- **Search Domain Filter Confusion**: A member questioned if the **search_domain_filter** feature is functioning correctly, stating they continue to see results from other websites in the citations.
  
  - There has been no confirmation on its effectiveness yet, leaving users uncertain about its reliability.
- **Using Vercel AI SDK with Perplexity**: A member asked how to integrate the **Vercel AI SDK** with **Perplexity**, specifically regarding citations.
  
  - This indicates a potential interest in more detailed documentation or guidance on the integration process.
- **Reddit Citations API Issues**: A member reported encountering issues with pulling **Reddit** as a source in citations via the API over the last week, despite it previously working well.
  
  - This raises concerns about reliability and consistency in pulling data from Reddit as a citation source.

 

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1306095383247720458) (54 messages🔥):

> - `Career Paths in ML Optimization`
> - `Performance Improvement Strategies`
> - `AI Conference Insights`
> - `Job Application Trends at Big Tech`
> - `Internship Experiences in AI`

- **Deciding Between ML Roles**: A member deliberated between working as a Product ML Engineer focused on UI/UX personalization or as an ML Infra Engineer handling GPU compute orchestration, emphasizing the need for performance optimization in LLMs.
  
  - They noted that without a PhD, they felt algorithmic or architectural work in optimization may be less accessible.
- **Emphasizing Practical Work**: A discussion highlighted the importance of making projects visible and doing practical work over traditional credentials, with one member mentioning writing CUDA kernels at home.
  
  - They expressed the intention to benchmark performance and share insights in a blog, recognizing that the best way to learn is through hands-on experience.
- **Exploring AI Conferences**: Members shared a list of prominent AI conferences to consider for engagement, such as [KDD, ICML,](https://aideadlin.es/?sub=ML,CV,CG,NLP,RO,SP,DM,AP,KR,HCI) and [CVPR](https://cvpr.thecvf.com/Conferences/2025).
  
  - They also included critical deadline information for paper submissions to help others stay informed.
- **Job Switch Considerations**: A member expressed frustration over being unable to switch jobs due to a 12-month tenure requirement, despite having an opportunity to move to PyTorch.
  
  - They are evaluating job openings for January while discussing salary implications within their current employment.
- **Internship Roles in AI Companies**: A member inquired about the types of work interns typically engage in within AI-oriented teams, reflecting a curiosity about entry-level experiences.
  
  - This question underscores the ongoing interest in understanding the structure and opportunities available in AI internships.

 

**Link mentioned**: [AI Conference Deadlines](https://aideadlin.es/?sub=ML,CV,CG,NLP,RO,SP,DM,AP,KR,HCI)): no description found

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1306004954049417277) (98 messages🔥🔥):

> - `Saddle Points in Gradient Descent`
> - `Batch Normalization and Alternatives`
> - `Vision Language Action Models`
> - `Binding Problem in Neural Networks`
> - `Complex Valued Latents vs. Real Valuations`

- **Saddle Points seldom matter with Noised Gradient Descent**: Participants discussed that saddle points are less significant in noised gradient descent scenarios, indicating optimizers function effectively even in their presence.
  
  - However, some insisted that saddle points might still arise in high-dimensional cases, suggesting their prevalence isn't diminished as previously considered.
- **Batch Norm remains useful under certain conditions**: The discussion highlighted that despite the emergence of alternatives, Batch Normalization is still valued, particularly when adapted in Ghost Batch Norm.
  
  - Critiques were made about its variable impact relative to batch size, with some arguing for more research into this normalization technique's efficiency and when it excels.
- **Exploring Vision Language Action Models**: A new research release was presented regarding benchmarking Vision, Language, and Action models in robotic tasks, involving prominent institutions and promising insights.
  
  - Participants were encouraged to share feedback about the work and delve into the provided links for a deeper understanding of the models and applications.
- **Discussion on the Binding Problem and Representation Learning**: A conversation unfolded about how overcoming the binding problem in artificial intelligence requires new methods of representation that go beyond traditional techniques.
  
  - There was curiosity about how these concepts relate to previous works like Hinton's GLOM and the potential for transformation representations, stressing the push for innovative computational models.
- **Complex Valued Latents vs. Real Valuations**: Participants debated the advantages of using complex-valued latents over traditional L2 normalized vectors, hinting at greater flexibility and expressivity.
  
  - Discussion included the idea of isometric tensors and the ability to work with transformations preserving distances, pointing out a move towards richer data representations.

**Links mentioned**:

- [How to represent part-whole hierarchies in a neural network](https://arxiv.org/abs/2102.12627): This paper does not describe a working system. Instead, it presents a single idea about representation which allows advances made by several different groups to be combined into an imaginary system ca...
- [Artificial Kuramoto Oscillatory Neurons](https://arxiv.org/abs/2410.13821): It has long been known in both neuroscience and AI that ``binding'' between neurons leads to a form of competitive learning where representations are compressed in order to represent more abst...
- [Rotating Features for Object Discovery](https://arxiv.org/abs/2306.00600): The binding problem in human cognition, concerning how the brain represents and connects objects within a fixed network of neural connections, remains a subject of intense debate. Most machine learnin...
- [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://arxiv.org/abs/1406.2572): A central challenge to many fields of science and engineering involves minimizing non-convex error functions over continuous, high dimensional spaces. Gradient descent or quasi-Newton methods are almo...
- [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171): Batch normalization is a key component of most image classification models, but it has many undesirable properties stemming from its dependence on the batch size and interactions between examples. Alt...
- [Euclidean plane isometry - Wikipedia](https://en.wikipedia.org/wiki/Euclidean_plane_isometry): no description found
- [Visual Representation Learning Does Not Generalize Strongly Within the Same Domain](https://arxiv.org/abs/2107.08221): An important component for generalization in machine learning is to uncover underlying latent factors of variation as well as the mechanism through which each factor acts in the world. In this paper, ...
- [Tweet from harsh (@HarshSikka)](https://x.com/HarshSikka/status/1856739777208574151)): Excited to share our new paper "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks" We evaluate how well VLM & VLA models can control robots across 20 different real-wor...

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1306011346902454306) (1 messages):

> - `Greedy Line Search`
> - `Gradient Descent Stepsizes`
> - `Periodicity in Optimization`

- **Oscillating Learning Rate Observed**: Users have noticed **oscillating learning rate behavior** during greedy line search while experimenting with functions such as **x² + ½y² + ⅓z²**.
  
  - This phenomenon appears exclusively for **gradient descent**, indicating a nuanced exploration of optimization strategies.
- **Breaking Conventional Wisdom on Stepsizes**: [Professor Grimmer's findings](https://x.com/prof_grimmer/status/1679846891171766272) suggest that the classic belief regarding gradient descent's optimal rate relying on constant step sizes **1/L** is misleading.
  
  - He asserts that **stepsizes in (0, 2/L)** for convergence are not necessary; instead, *periodic long steps* are proven to yield better results, as detailed in his paper [here](https://arxiv.org/abs/2307.06324).

 

**Link mentioned**: [Tweet from Ben Grimmer (@prof_grimmer)](https://x.com/prof_grimmer/status/1679846891171766272): I've proven the strangest result of my career.. The classic idea that gradient descent's rate is best with constant stepsizes 1/L is wrong. The idea that we need stepsizes in (0,2/L) for conve...

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1305988843161387130) (20 messages🔥):

> - `Checkpoint Issues with EvalHarness`
> - `Finetuned Model Performance on Sentiment Analysis`
> - `Accurate Averaging for CoT Accuracy`
> - `Multi-Class Text Classification Task Design`
> - `Metrics for New Task with Two Metrics`

- **Checkpoint Issues with EvalHarness Resolved**: Members discussed issues related to using a .pth checkpoint with the evalharness library, including an error related to the `config not found`.
  
  - A successful workaround involved fixing state dict keys and saving the model in a proper format, ultimately resolving the errors encountered.
- **Finetuned Model Struggles with Sentiment Analysis**: A member expressed concern about their finetuned model's performance on a financial sentiment task, questioning the setup and model choice.
  
  - It was suggested that the choice of using text generation models rather than classification models might be contributing to the poor results.
- **Clarifying Averaging Accuracy in CoT**: A member inquired about averaging accuracy across runs for vanilla CoT from self-consistency, highlighting limitations in the current evaluation setup.
  
  - Another member suggested exploring existing structures like the gsm8k_cot for potential solutions in capturing the average accuracy effectively.
- **Guidance on Multi-Class Classification Task**: A member sought advice on setting up a multi-class text classification task with ten classes and whether to use multiple choice or generate_until.
  
  - It was recommended to opt for multiple choice to restrict the output space effectively.
- **Discussing Metrics for Dual Metric Task**: A member reported difficulty using both `acc_norm` and `exact_match` metrics together in a new task and requested assistance.
  
  - A suggestion was made that `acc_norm` may not be suitable for generation tasks, prompting a request for clarification on the specific use case.

 

---

### **Eleuther ▷ #**[**gpt-neox-dev**](https://discord.com/channels/729741769192767510/730090096287547444/1306214196366540812) (24 messages🔥):

> - `Single GPU Bugs`
> - `DagsHub Integration`
> - `YAML File Extensions`
> - `Model Training and Maintenance`
> - `Error Handling in Configurations`

- **Single GPU Bugs Cause Confusion**: A member reported experiencing several **bugs** while running on a single GPU, suggesting that testing predominantly focuses on multi-GPU setups.
  
  - Another member indicated that most open PRs are for **new features**, not bug fixes, due to ongoing efforts in model training.
- **Engagement with DagsHub Discussed**: A member proposed the potential value of encouraging **DagsHub** to integrate with **GPT-NeoX**, seeking insights from the community.
  
  - Inquiries about **AnthropicAI's** frameworks were made, leading to confirm that they utilize their own system, which is not public.
- **YAML vs YML File Extension Debate**: Confusion arose over the use of `.yaml` versus `.yml` file extensions in **GPT-NeoX** configurations, with issues reported using the `.yaml` extension.
  
  - A member speculated that the configuration files might have a **JSON-like format**, which could explain the extension preference.
- **Model Training Efforts Causing Delays**: Another member indicated that they would be occupied with model training and papers for about **30 days**, affecting maintenance activities.
  
  - They appreciated bug reports and planned to address them after the current development workload.
- **Configuration Error Resolved**: There was discussion about a specific code snippet in **arguments.py** that may cause issues if `.yaml` files are used, leading to unintended behavior.
  
  - A proposed solution was to modify the code to include `.yaml` in the file handling logic, which could resolve the existing configuration problems.

 

**Link mentioned**: [GitHub - markNZed/gpt-neox at pipe_parallel_size_1](https://github.com/markNZed/gpt-neox/tree/pipe_parallel_size_1): An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - GitHub - markNZed/gpt-neox at pipe_parallel_size_1

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1306395578606813195) (1 messages):

> - `UnslopNemo 12B`
> - `SorcererLM`
> - `Inferor 12B`
> - `Mistral Parameter Updates`
> - `UI Improvements`

- **UnslopNemo 12B v4 launched for adventurers**: The latest model, [UnslopNemo 12B](https://openrouter.ai/thedrummer/unslopnemo-12b), designed for adventure writing and role-play scenarios, is now available.
  
  - A free variant is also available for 24 hours at [UnslopNemo 12B Free](https://openrouter.ai/thedrummer/unslopnemo-12b:free) with requests directed to [Discord](https://discord.gg/fVyRaUDgxW).
- **SorcererLM introduces advanced storytelling**: [SorcererLM](https://openrouter.ai/raifle/sorcererlm-8x22b) is an advanced roleplay model built using Low-rank 16-bit LoRA fine-tuned on WizardLM-2-8x22B, available for trial.
  
  - Requests for this model can be directed to our [Discord](https://discord.gg/fVyRaUDgxW).
- **Inferor 12B merges top roleplay models**: The new [Inferor 12B](https://openrouter.ai/infermatic/mn-inferor-12b) combines the best features of existing roleplay models.
  
  - Users are advised to set reasonable max output limits to prevent excessive text generation, with requests also going to [Discord](https://discord.gg/fVyRaUDgxW).
- **Mistral and Gemini gain parameter enhancements**: Both **Mistral** and **Gemini** have added support for **Frequency Penalty** and **Presence Penalty**, enhancing their parameter capabilities.
  
  - Mistral implementation now includes tools for **seed** adjustments as well.
- **New UI features enhance user experience**: Recent UI improvements include a document search functionality activated by cmd + K, facilitating model searches significantly.
  
  - The introduction of a new table list view allows users to observe more models concurrently, enhancing overall navigability.

**Links mentioned**:

- [OpenRouter](https://openrouter.ai/thedrummer/unslopnemo-12b)): LLM router and marketplace
- [OpenRouter](https://openrouter.ai/thedrummer/unslopnemo-12b:free)): LLM router and marketplace
- [OpenRouter](https://openrouter.ai/raifle/sorcererlm-8x22b)): LLM router and marketplace
- [OpenRouter](https://openrouter.ai/infermatic/mn-inferor-12b)): LLM router and marketplace

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1306378508125212723) (1 messages):

> - `GitHub Open Source Posting Rules`

- **Inquiry on GitHub Open Source Posting Policies**: A member inquired about the **rules and policies** for posting GitHub open source projects.
  
  - They requested to be tagged in responses, highlighting their interest in receiving detailed information on the topic.
- **Seeking Clarification on Posting Guidelines**: The same member emphasized the importance of understanding the rules to effectively share projects on **GitHub**.
  
  - They expressed gratitude in advance for any insights shared by others.

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1306000012953522309) (186 messages🔥🔥):

> - `Model Performance Issues`
> - `Tool Calling Functionality`
> - `Image Generation APIs`
> - `Qwen Model Updates`
> - `Mistral Large Output Quality`

- **Challenges with Mistral Large Output**: A user reported receiving **gobbledygook** from **Mistral Large**, despite trying various system prompts and restarting their instance.
  
  - The issue was resolved after adjusting the settings for frequency and presence penalties, which were recently added.
- **Confusion over Tool Calling**: Users discussed the **tool calling** feature, which is designed to enhance interactions with models by injecting tools into prompts.
  
  - However, some found that while tool calling was enabled, it did not seem to impact token usage as expected.
- **Performance of Qwen Model on OpenRouter**: There were discussions about the **Qwen model** and its capabilities regarding tool calling, as users expressed skepticism over its effectiveness.
  
  - It was noted that while the model theoretically supports tool calling, some users experienced issues with implementation.
- **Image Generation API Recommendations**: Users sought recommendations for reliable **image generation APIs** along with suitable platforms and models to consider.
  
  - The conversation hinted at the need for optimal performance and pricing for API services in this area.
- **High Token Processing Volume**: One user mentioned processing over **3 million tokens** daily while developing an AI chatbot focused on a niche vertical.
  
  - This raised questions regarding potential price reductions for high-volume token processing on certain models.

**Links mentioned**:

- [Bloomberg - Are you a robot?](https://www.bloomberg.com/news/articles/2024-11-13/openai-google-and-anthropic-are-struggling-to-build-more-advanced-ai): no description found
- [OpenRouter](https://openrouter.ai/api/v1): LLM router and marketplace
- [Avian.io](https://avian.io/): Avian.io is home of the worlds fastest inference for Llama 405B and more. Try our AI cloud platform and API now with no rate limits.
- [Prompt Caching | OpenRouter](https://openrouter.ai/docs/prompt-caching#inspecting-cache-usage): Optimize LLM cost by up to 90%
- [Grok Beta - API, Providers, Stats](https://openrouter.ai/x-ai/grok-beta): Grok Beta is xAI's experimental language model with state-of-the-art reasoning capabilities, best for complex and multi-step use cases. It is the successor of [Grok 2](https://x. Run Grok Beta w...
- [Responses | OpenRouter](https://openrouter.ai/docs/responses#sse-streaming-comments): Manage responses from models
- [Requests | OpenRouter](https://openrouter.ai/docs/requests#images-_-multimodal-requests): Handle incoming and outgoing requests

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1306066150915833958) (7 messages):

> - `Custom Provider Keys Access Requests`

- **Members eager for Custom Provider Keys**: Multiple users requested access to **Custom Provider Keys**, highlighting a strong demand for this feature.
  
  - Each request emphasized a desire to utilize the keys effectively for their specific needs.
- **Community engagement on access requests**: A variety of members contributed to the discussion, showcasing an active interest in obtaining **Custom Provider Keys**.
  
  - Requests varied from simple expressions of need to direct appeals for access.

 

---

### **aider (Paul Gauthier) ▷ #**[**announcements**](https://discord.com/channels/1131200896827654144/1133060115264712836/1306372686519468102) (1 messages):

> - `Aider v0.63.0 Release`
> - `Qwen 2.5 Coder Support`
> - `Web Command Functionality`
> - `Improved Language Prompting`
> - `Bug Fixes and Performance Enhancements`

- **Aider v0.63.0 Launches with Exciting Features**: The **Aider v0.63.0** release includes support for **Qwen 2.5 Coder 32B** and improved handling of LiteLLM exceptions, enhancing usability.
  
  - Aider notably wrote **55%** of the code in this release, showcasing an impressive level of self-sufficiency.
- **Web Command Overhaul**: The new `/web` command simply adds the page to the chat without triggering an **LLM response**.
  
  - This streamlined approach is likely to enhance user experience by reducing unnecessary interactions.
- **Language Prompting Gets Better**: Improvements have been made in prompting for the user's preferred **chat language**, allowing for a more tailored interaction.
  
  - This change aims to make conversations feel more natural and user-friendly.
- **Critical Bug Fixes Implemented**: Recent bug fixes address issues like **double-counting tokens** during cache stats and problems with the LLM creating new files.
  
  - These fixes contribute to a more reliable and efficient experience for users.

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1306006737362620609) (96 messages🔥🔥):

> - `Vectorizing and Reranking Read-Only Files`
> - `Aider Extensions for VSCode and Neovim`
> - `Issues with Sonnet Performance`
> - `OpenRouter Provider Configuration`
> - `Upcoming AI Conferences`

- **Exploring Vectorization for Read-Only Files**: A user discussed challenges in vectorizing and reranking about 30 read-only markdown files for Aider, noting that including too many reduces necessary information.
  
  - There was interest in better search functionality, particularly for large, feature-rich projects.
- **New Aider Extensions for Editors**: A new VSCode extension for Aider was announced, featuring Markdown preview, file management, and chat history, with community contributions encouraged.
  
  - Additionally, a Neovim Aider extension was shared, promoting collaboration to enhance Aider utility across different platforms.
- **Sonnet Performance Issues**: Users reported experiencing issues with Sonnet, specifically not applying edits effectively, which they attributed to high demand or performance fluctuations.
  
  - The community monitored service status for updates, indicating possible server-related delays affecting Sonnet's performance.
- **Configuring OpenRouter for Aider**: Discussions included how to specify provider preferences in OpenRouter and the method to create a model metadata file for managing costs and context size in Aider.
  
  - Users shared tips on balancing provider use and the importance of understanding OpenRouter's load balancing mechanisms.
- **Upcoming AI Conferences Survey**: A user initiated a survey to gather information on upcoming AI conferences, asking for specific events and conference brands being followed.
  
  - The community reacted positively, indicating engagement and interest in upcoming AI events and networking opportunities.

**Links mentioned**:

- [Anthropic Status](https://status.anthropic.com/): no description found
- [Provider Routing | OpenRouter](https://openrouter.ai/docs/provider-routing): Route requests across multiple providers
- [Advanced model settings](https://aider.chat/docs/config/adv-model-settings.html): Configuring advanced settings for LLMs.
- [GitHub - nekowasabi/aider.vim: Helper aider with neovim](https://github.com/nekowasabi/aider.vim): Helper aider with neovim. Contribute to nekowasabi/aider.vim development by creating an account on GitHub.

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1306068076076208199) (61 messages🔥🔥):

> - `Aider integration suggestions`
> - `Using Aider in Architect mode vs other modes`
> - `Using git diff with Aider`
> - `Setting up Aider in Termux`
> - `Using Rust Analyzer in VSCode`

- **Suggestions for Aider-Chatbot Integration**: A user expressed interest in suggesting integration between Aider and ChatGPT's web interface on GitHub, seeking advice on format for feedback.
  
  - This discussion led to exploring the relevance of user-generated requests for improving platform integrations.
- **Architect mode vs Other Modes**: New users are advised to skip Aider's Architect mode in favor of simpler interactions, especially when they are not developers.
  
  - Overall, users found Aider effective without Architect mode for adding features, with some proposing to experiment further.
- **Using git diff Feature**: Users can utilize the `/run git diff` command within Aider to read file edits, making it easy to integrate changes into the chat.
  
  - This enhances the ability to prompt Aider for further actions based on the identified differences in code.
- **Installing Aider in Termux**: There was a query regarding the installation of Aider in mobile environments like Termux and its compatibility with different IDEs.
  
  - The consensus affirmed that Aider remains IDE agnostic as long as it can operate within a compatible Python environment.
- **Rust Analyzer Integration with VSCode**: Users inquired about the easiest way to trigger the Rust Analyzer to run after Aider completes edits.
  
  - It was suggested that users could utilize the `--lint-cmd` to execute any necessary commands, including refreshing the linting status in VSCode.

**Links mentioned**:

- [Linting and testing](https://aider.chat/docs/usage/lint-test.html): Automatically fix linting and testing errors.
- [Specifying coding conventions](https://aider.chat/docs/usage/conventions.html): Tell aider to follow your coding conventions when it works on your code.
- [Tips](https://aider.chat/docs/usage/tips.html): Tips for AI pair programming with aider.
- [FAQ](https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context): Frequently asked questions about aider.
- [Maximize Your ChatGPT Experience: Mastering Auto Split and Summarize with Superpower ChatGPT](https://www.youtube.com/watch?v=IhRbmIhAm3I): Download here: Chrome: https://chrome.google.com/webstore/detail/superpower-chatgpt/amhmeenmapldpjdedekalnfifgnpfnkc Firefox: https://addons.mozilla.org/en-U...
- [Superpower ChatGPT - Chrome Web Store](https://chromewebstore.google.com/detail/superpower-chatgpt/amhmeenmapldpjdedekalnfifgnpfnkc): ChatGPT with Superpowers! Folders, Search, GPT Store, Image Gallery, Voice GPT, Export, Custom Prompts, Prompt Chains, Hidden Models
- [GitHub - Amm1rr/WebAI-to-API: Claude, Gemini to API : ) (Don't need API KEY)](https://github.com/Amm1rr/WebAI-to-API): Claude, Gemini to API : ) (Don't need API KEY). Contribute to Amm1rr/WebAI-to-API development by creating an account on GitHub.
- [[Q] Is it possible to use `aider --apply` with output from web frontends like chatgpt.com? · Issue #2203 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2203): o1-preview is cheaper on the subscription on chatgpt.com, and in general, I like the flexibility of working with raw LLMs. But applying edits from the web frontend to local files is a PITA. I often...

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1306003968962596946) (4 messages):

> - `SupermavenAI joins Cursor`
> - `Organizing codebase for AI`
> - `Challenges faced with AI coding tools`

- **SupermavenAI partners with Cursor**: Cursor announced the exciting news that **SupermavenAI** is joining their team to enhance their research and product capabilities. This collaboration aims to transform Cursor into a **powerhouse** of innovation.
  
  - The announcement was made via [Twitter](https://x.com/cursor_ai/status/1856427424927625679).
- **Organizing Codebase for AI Efficiency**: A member shared insights on how to effectively organize a codebase when integrating AI tools like [aider.chat](https://aider.chat/?ref=entrecurious.xyz). Suggestions include breaking projects into **logical modules** and ensuring clear comments to aid understanding.
  
  - The importance of human-readable code when working with AI tools is emphasized for improving productivity.
- **Experiences with AI Coding Tools**: A member detailed their mixed experiences with AI coding tools, expressing initial optimism that devolved into frustration due to inefficiencies encountered. The challenges they faced led to questions about the true effectiveness of such tools in enhancing productivity.
  
  - This conversation prompted inquiries from others about their similar experiences with integrating AI into their coding workflows.

**Links mentioned**:

- [Make Way for AI-Readable Codebases](https://entrecurious.xyz/make-way-for-ai-readable-code/): 🚀discuss this post on hackernews Introduction: The Overwhelmed Developer CEO In the early days of Ceedar, around last November, we were (still are!) a small, two-person startup with big ambitions. ...
- [Tweet from Cursor (@cursor_ai)](https://x.com/cursor_ai/status/1856427424927625679): We are excited to announce that @SupermavenAI is joining Cursor! Together, we will continue to build Cursor into a research and product powerhouse. (1/5)

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1305986535912505440) (62 messages🔥🔥):

> - `Quantization Differences`
> - `LM Studio Connection to TTS`
> - `Performance Issues with Qwen 2.5`
> - `LaTeX Rendering in LM Studio`
> - `Python Script for Llama.cpp Integration`

- **Understanding Quantization Sizes**: Members discussed the impact of quantization sizes, noting that smaller quant sizes lead to more compression, but larger sizes may split into multiple parts.
  
  - *Heyitsyorkie* summarized that higher quant sizes could assure better performance without significant losses.
- **Interest in TTS Integration with LM Studio**: A member expressed curiosity about when LM Studio could connect to Text-to-Speech (TTS) features.
  
  - The response indicated ongoing conversations around integrating such features, but no timeline was provided.
- **Troubleshooting Qwen 2.5 Performance**: A user reported previously facing issues with Qwen 2.5, specifically getting only autocomplete responses, but later noted it started working correctly.
  
  - Others recommended ensuring proper configuration and exploring model options to optimize performance.
- **Rendering LaTeX in LM Studio**: Users were trying to figure out how to render LaTeX correctly in LM Studio, with some noting the need for `$` signs to get it to display properly.
  
  - One user reported that despite settings being correct, LaTeX was not rendered as expected.
- **Sideloading Llama.cpp Features**: There was a request for a Python script to allow sideloading of the latest Llama.cpp into LM Studio, highlighting the need for such functionality.
  
  - Participants acknowledged that the community has long awaited this feature and mentioned ongoing efforts to make it a reality.

**Links mentioned**:

- [GGUF](https://huggingface.co/docs/hub/gguf): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gqa5xx/lm_studio_incredibly_sl): no description found
- [Leak: ‘GPT-5 exhibits diminishing returns’, Sam Altman: ‘lol’](https://www.youtube.com/watch?v=iybgycPk-N4): The last few days have seen two narratives emerge. One, derived from yesterday’s OpenAI leak in TheInformation, that GPT-5/Orion is a disappointment, and les...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gqa5xx/lm_studio_incredibly_slow_12_tokenssec_on_a_3090/): no description found
- [Microsoft Forms](https://forms.office.com/e/9aSb6edfGi): no description found
- [llama : switch KQ multiplication to use F32 precision by default by ggerganov · Pull Request #10015 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/10015): ref #10005, #9991 (comment) The list of models that require higher floating point range in the attention keeps growing, so to be on the safe side, default to F32 for the KQ multiplication.
- [lmstudio.js Code Examples - SDK (TypeScript) | LM Studio Docs](https://lmstudio.ai/docs/sdk/lmstudioclient): Examples of using lmstudio.js in TypeScript applications.

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1306001693007155231) (40 messages🔥):

> - `GPU Combinations`
> - `Local LLM Usage on Macs vs PCs`
> - `Partial Offload Challenges`
> - `Cloud vs Local Model Usage`
> - `Upcoming Hardware Competition`

- **GPU combinations for LLM inference**: Discussion on using a **12GB 3060** and **40GB A800** together for **70B class models** raises the question of whether to use one GPU or both, with concerns on how scaling affects performance.
  
  - A member suggested that it may be more beneficial to solely utilize the A800 since it can fit the model in **VRAM** while the 3060 cannot.
- **Comparative pricing for ML machines**: Members discussed the *competitiveness* of **Macbook** pricing compared to similar PC capabilities, noting durability of Macbooks potentially lasting a decade.
  
  - Concerns were raised about the pricing of the **128GB M4 Max** from Apple, which is seen as exorbitant compared to PC parts.
- **Challenges with partial offload**: Experiences shared about **partial offload** for using large models on systems lead to the conclusion that it is inadequate for real-time interactions.
  
  - Members emphasized that CPUs struggle for tasks like **matrix multiplication**, hindering effectiveness, while GPUs offer superior performance.
- **Cloud vs local model usage debate**: Concerns were discussed regarding the cost of cloud services, with one member mentioning a $20 charge for a single night of usage raising questions about **API pricing** efficiency.
  
  - Several members highlighted the privacy and ease of experimentation advantages associated with local setups compared to using cloud services.
- **Anticipation of new hardware developments**: The upcoming **AMD Strix Halo APU** and rumors of **Nvidia's ARM SoC** sparked discussion about future competition in the laptop market for ML tasks.
  
  - Hope is expressed for hardware advancements that could improve bandwidth and memory capacities, supporting high-performance workloads.

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1305992834104430602) (87 messages🔥🔥):

> - `Training with Dreambooth`
> - `Using Animatediff for Video Generation`
> - `Downloading Checkpoint Files`
> - `Python Version for Stable Diffusion`
> - `Accessing Discord Servers`

- **Training Model with Movie Posters**: A user is seeking tutorials for training on movie posters using **Dreambooth** in **auto1111**. They are looking for the latest techniques and suggestions for effective training.
  
  - The community suggested checking for existing resources and guides to streamline the process.
- **Animatediff for Video Clips**: Members discussed using **Animatediff** for generating video clips, with one explaining it allows posting two images to create transitions. It was noted that the resolution may be lower, but it's suitable for social media.
  
  - A recommendation for the **Banodoco server** was provided, as they specialize in video-related tools.
- **Downloading Checkpoints and LoRAs**: Users shared links to external file hosting sites for downloading checkpoint files and LoRAs, mentioning **Google Drive**, **Mega**, and **Hugging Face**. Additional discussions included the limitations of **Civit AI** and potential bans on certain content.
  
  - Concerns were raised about the removal of specific content types and their impact on user access.
- **Python Version Issues with Stable Diffusion**: A user encountered an error related to the **torch** package while setting up a Python environment for Stable Diffusion. The solution suggested uninstalling the current Python version and installing **Python 3.10.11 64bit** instead.
  
  - The user expressed gratitude for the support and planned to try the solution soon.
- **Accessing Discord for Help**: Users inquired about accessing URLs for Discord servers, specifically looking for new invites and direct links. There was a shared experience regarding finding outdated invitation links for the **Pixaroma** community.
  
  - The community provided assistance to connect with the required Discord servers.

**Links mentioned**:

- [What's the current Banodoco server discord URL? (all invites on YT are now invalid).](https://old.reddit.com/r/StableDiffusion/comments/18wm1md/whats_the_current_banodoco_server_discord_url_all/```the): Searched here, too, no dice. Looking for the Banodoco server :) Thank you!
- [New item by Camilla Lyn](https://photos.app.goo.gl/e5uTCokWBjYEtqaF7): no description found

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1305987195160494131) (48 messages🔥):

> - `Nous 3 Model Performance`
> - `Francois Chollet Leaves Google`
> - `AI Agent Tool Operator Launch`
> - `JanusFlow Model Introduction`
> - `Community Discussions on Gwern`

- **Nous 3 Model Performance Numbers Confusion**: There are discrepancies in **Nous' 70B model** performance figures as seen in [this thread](https://x.com/thexeophon/status/1856429292504096944?s=61), leading to questions about the validity of the reported **MMLU-Pro** scores.
  
  - Members speculate that differences in prompting techniques and benchmark inconsistencies could be factors influencing these divergent numbers.
- **Francois Chollet's Departure from Google**: Francois Chollet, the creator of **Keras**, is leaving Google to embark on a new career chapter, as announced [here](https://developers.googleblog.com/en/farewell-and-thank-you-for-the-continued-partnership-francois-chollet/).
  
  - Despite his departure, Chollet remains committed to supporting Keras and its future development, emphasizing collaboration with the open-source community.
- **Exciting Launch of AI Agent Tool 'Operator'**: OpenAI is set to launch a new **AI agent tool** called 'Operator' that automates tasks such as writing code and booking travel, expected to be released in January according to [this announcement](https://www.bloomberg.com/news/articles/2024-11-13/openai-nears-launch-of-ai-agents-to-automate-tasks-for-users?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTczMTUyODYxOCwiZXhwIjoxNzMyMTMzNDE4LCJhcnRpY2xlSWQiOiJTTVdOQURUMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.TTJZiuo4Nk2U295FHBFsxeN0YGznZJ32sHnNReQmEjM).
  
  - This tool aims to enhance user productivity by taking actions on behalf of individuals in various contexts.
- **Introduction of JanusFlow Model**: The **JanusFlow model** is introduced as a new capability that harmonizes autoregressive LLMs with rectified flow for both image understanding and generation, as detailed in [this post](https://x.com/deepseek_ai/status/1856552494379520510).
  
  - JanusFlow is designed to be powerful, simple, and flexible, shaping the next generation of AI models in this domain.
- **Discussion on Gwern and Community Insights**: Members discussed **Gwern**, reflecting on his insights in AI and biohacking, comparing him to other notable figures in online forums like *Slate Star Codex*.
  
  - There was a consensus that Gwern's meticulously written blogs offer deep, thoughtful explorations of complex subjects.

**Links mentioned**:

- [Tweet from Teknium (e/λ) (@Teknium1)](https://x.com/teknium1/status/1856462102518768063?s=61): @TheXeophon @gm8xx8 bf16 and custom parsing - we couldnt use lm eval harness like we did there so there will be different baselines
- [Tweet from Shirin Ghaffary (@shiringhaffary)](https://x.com/shiringhaffary/status/1856792898932539609?s=61): NEW: OpenAI is preparing to launch a new computer using AI agent tool codenamed “Operator” that take actions on a person’s behalf thru a browser, such as writing code or booking travel. Staff told in...
- [Bloomberg - Are you a robot?](https://t.co/dNZTbrQ4BJ): no description found
- [Tweet from Xeophon (@TheXeophon)](https://x.com/thexeophon/status/1856429292504096944?s=61): @gm8xx8 These are Nous‘ numbers from the 3 release. The reported numbers for the 70B model don’t match the graphic as well - MMLU-Pro (Release) 47.24 vs 54.14 now Am I missing something super obviou...
- [Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)](https://x.com/lmarena_ai/status/1856444009323082093?s=61): Which model is best for coding? @CopilotArena leaderboard is out! Our code completions leaderboard contains data collected over the last month, with >100K completions served and >10K votes! Le...
- [Releasing the largest multilingual open pretraining dataset](https://huggingface.co/blog/Pclanglais/two-trillion-tokens-open): no description found
- [Tweet from DeepSeek (@deepseek_ai)](https://x.com/deepseek_ai/status/1856552494379520510): 🚀 Introducint JanusFlow: harmonizing autoregressive LLMs with rectified flow! By adopting the best practices in both fields, JanusFlow excels at both image understanding & generation in a single mo...
- [Tweet from Hailey Schoelkopf (@haileysch__)](https://x.com/haileysch__/status/1856172527921574154): Major life update: I'm joining @AnthropicAI this week! Looking forward to meeting and working with the amazing team there! I’m beyond thankful for an amazing 2 years with my colleagues and colla...
- [He Admit It Admit GIF - He Admit It Admit It Admit - Discover & Share GIFs](https://tenor.com/view/he-admit-it-admit-it-admit-omg-itysl-gif-18470746): Click to view the GIF
- [Farewell and thank you for the continued partnership, Francois Chollet!](https://developers.googleblog.com/en/farewell-and-thank-you-for-the-continued-partnership-francois-chollet/): no description found
- [About Gwern · Gwern.net](https://gwern.net/me): no description found

---

### **Interconnects (Nathan Lambert) ▷ #**[**other-papers**](https://discord.com/channels/1179127597926469703/1179142630517518397/1306321485417152552) (6 messages):

> - `Jailbreak Rapid Response`
> - `Prompt Injection at Anthropic`
> - `Internal Models of Anthropic`

- **Adaptive Techniques Block Jailbreaks**: Anthropic's new research introduces adaptive techniques to **rapidly block new classes of jailbreak** as they are detected, as discussed in their paper [here](https://arxiv.org/abs/2411.07494).
  
  - *Ensuring perfect jailbreak robustness is hard,* which highlights the challenges faced in securing AI models.
- **Concerns Over Prompt Injection**: A user expressed concern about the **secret prompt injection** being used by Anthropic that affects how their models respond to copyright issues.
  
  - A call for **public acknowledgment** of this practice suggests ongoing community discomfort with its implications.
- **Speculations on Un-Nerfed Models**: Questions arose regarding whether Anthropic employs an **un-nerfed model** internally and how it compares in performance.
  
  - This raises curiosity about the differences between internal and publicly accessed models.
- **A Daily Tweet Campaign**: One user initiated a campaign to tweet daily until Anthropic addresses its **prompt injection practices** publicly.
  
  - This reflects a broader desire for transparency within AI model operations.
- **User Reactions to Model Instructions**: Community members reacted to the notion of instruction prompts, humorously citing phrases like, **'Do not hallucinate!'** as a direct command.
  
  - Such reactions indicate a mix of concern and sarcasm about restrictions placed on AI models.

**Links mentioned**:

- [Tweet from Anthropic (@AnthropicAI)](https://x.com/anthropicai/status/1856752093945540673?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ): New research: Jailbreak Rapid Response. Ensuring perfect jailbreak robustness is hard. We propose an alternative: adaptive techniques that rapidly block new classes of jailbreak as they’re detected. ...
- [Tweet from kalomaze (@kalomaze)](https://x.com/kalomaze/status/1837954600348917817): day 1 of making a tweet every day until @AnthropicAI either removes or (at the very least) \*\*publicly acknowledges\*\* the secret prompt injection being done to their models (still applies over API btw)...

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1305993487967322132) (9 messages🔥):

> - `Vision Language Models (VLMs)`
> - `Finbarr Blog Discussion`
> - `Computational Cost of VLM Inference`
> - `Recent VLM Papers`

- **Dive into Finbarr's Blog**: A member mentioned reading the Finbarr blog, specifically highlighting a post about **Vision Language Models (VLMs)** after being inspired by **Claude's** success with equation screenshots.
  
  - Another followed up, expressing love for the blog and recognizing a recent piece by **Sebastian Raschka** on **VLMs** as interesting but somewhat superficial.
- **High Cost of VLM Inference**: A shared post discussed the **costly inference** associated with VLMs due to having **500+ image tokens**, suggesting potential strategies of model sizing.
  
  - Key findings indicated that processing just **ONE compressed visual token** with the largest LLM achieves **compute-optimal inference**, raising questions about its implications.
- **Papers on Vision Language Models**: A user shared enthusiasm about reading multiple papers on VLMs, stating that new models like **Pixtral** and **DeepSeek Janus** had been released during their research.
  
  - They expressed surprise at how advancements have made reading text from images more approachable compared to earlier tools like **Tesseract** and **ABBYY FineReader**.

**Links mentioned**:

- [Tweet from Sachin Goyal (@goyalsachin007)](https://x.com/goyalsachin007/status/1856004116012798355): Inference with VLMs is costly, thanks to 500+ image tokens. So… should you use a smaller model or run a bigger model on fewer tokens? Our 📢new findings left me genuinely amazed: processing just ONE c...
- [Papers I've read this week: vision language models](https://www.artfintel.com/p/papers-ive-read-this-week-vision): They kept releasing VLMs, so I kept writing...

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1306074419638042635) (4 messages):

> - `MKBHD controversy`
> - `ChatGPT vs Claude humor`

- **MKBHD Deletes YouTube Segment After Backlash**: MKBHD faced significant backlash after a segment where he drove a **Lamborghini** at **96mph** in a **35mph** residential area was deleted from his YouTube channel due to public outrage.
  
  - *This controversy highlights the dangers of influencer accountability and public perception.*
- **ChatGPT Users Mocked with Green Text Message Joke**: A member quipped that guys who prefer **ChatGPT** over **Claude** might as well send *green text messages*, poking fun at the cultural divide in messaging apps.
  
  - *The humorous comparison emphasizes the ongoing rivalry and preferences in AI chat models, resonating with fans of both platforms.*

**Links mentioned**:

- [Tweet from near (@nearcyan)](https://x.com/nearcyan/status/1856565818433355790): guys who use chatgpt over claude might as well have green text msgs too
- [Tweet from Dexerto (@Dexerto)](https://x.com/Dexerto/status/1856446226444759348): MKBHD deletes YouTube segment after backlash for driving a Lamborghini 96mph in a 35mph residential area

---

### **Interconnects (Nathan Lambert) ▷ #**[**rlhf**](https://discord.com/channels/1179127597926469703/1208183230608576562/) (1 messages):

swyxio: we are looking for someone on this yes

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1305992409842319430) (11 messages🔥):

> - `Dylan Patel's Inference Math Lecture`
> - `Jonathan Frankle's AI Insights`
> - `Hiring Practices at Databricks`

- **Dylan Patel shares insights at Stanford ML**: In the [lecture titled 'Inference Math, Simulation, and AI Megaclusters'](https://youtu.be/hobvps-H38o?si=FR7re3r6gds6b-UN) at Stanford CS 229S, discussions include the gigawatt datacenter situation.
  
  - A member expressed a desire to attend Stanford classes, reflecting the interest in the course material.
- **Jonathan Frankle reveals AI model insights**: In the [video 'New Model Architectures are Unlikely'](https://youtu.be/7-3IxVvWoxc?si=eBMxBARTo-c-rrTZ), Jonathan Frankle discusses pre-training, fine-tuning, and AI policy as the Chief AI Scientist at Databricks.
  
  - While no novel insights were shared, members enjoyed listening to him, considering him a charming representative of the company.
- **Dario's hiring philosophy sparks debate**: A member questioned Dario's preference for hiring theoretical physics graduates because *'they can learn fast'* and expressed a preference for experienced engineers who understand the work immediately.
  
  - There was agreement among members that while fast learners have their merits, hands-on experience may hold more substantial value in the current job market.

**Links mentioned**:

- [Dylan Patel - Inference Math, Simulation, and AI Megaclusters - Stanford CS 229S - Autumn 2024](https://youtu.be/hobvps-H38o?si=FR7re3r6gds6b-UN): Website: https://scalingintelligence.stanford.edu/Github: https://github.com/ScalingIntelligenceHuggingFace: https://huggingface.co/ScalingIntelligence
- [Databricks AI Head: New Model Architectures are Unlikely, When to Pre-Train/Fine Tune and AI Policy](https://youtu.be/7-3IxVvWoxc?si=eBMxBARTo-c-rrTZ): Jonathan Frankle is the Chief AI Scientist at Databricks ($43B), which he joined through the acquisition of MosaicML in July 2023. Databricks has over 12,000...

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1306248350156328990) (2 messages):

> - `SnailBot News`

- **SnailBot News Alert**: A notification was sent out for **SnailBot News** to all members via the specified role <@&1216534966205284433>.
  
  - No additional details or discussions followed the alert.
- **Repeated SnailBot News Notification**: The same **SnailBot News** notification was issued again to all members with the role <@&1216534966205284433>.
  
  - This indicates potential ongoing news or updates related to SnailBot, though specifics were not provided.

 

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1305994687559106583) (19 messages🔥):

> - `KATT Integration for Podcasting`
> - `Research Consultancy Client Data Management`
> - `Podcast Creation Limits`
> - `Sharing NotebookLM Outside Organizations`
> - `Magic Book Experiment with Podcast`

- **KATT enhances podcast experience**: A member shared their experience of integrating **KATT** into their podcast production process, creating a fact checker that keeps hosts informed, resulting in a show over **90 minutes** long.
  
  - They noted that it incorporated processes and an altered system prompt, emphasizing the **two years** of training KATT underwent.
- **Navigating client data confidentiality**: A research consultant highlighted challenges when using a language model for focus group data while maintaining participant confidentiality, as the client desired more access to results.
  
  - *Caution was advised* about legal obligations and the necessity of anonymizing data to avoid privacy breaches.
- **Limitations on podcast creation per account**: A member inquired about podcast creation limits on their account, expressing concern over the inability to make more podcasts after multiple deletions.
  
  - Another member echoed this concern, seeking clarity on whether limits exist to avoid potential interruptions during experimentation.
- **Sharing NotebookLM across organizations**: A member asked about sharing NotebookLM content outside their Google Organization, suspecting this may be restricted by admin settings.
  
  - It was confirmed that **sharing is not possible** outside one's organization, with further details on personal account limitations being shared.
- **Magic Book podcast experiment**: One member conducted a unique podcast experiment involving a PDF titled 'Magic Book', which prompted hosts to share their experiences of what they saw.
  
  - This led to an unedited podcast that showcased the creative input from the hosts, with a link shared for audience feedback.

 

**Link mentioned**: [A discussion on How to Improve Health Evidence-Backed Nutrition Tips](https://youtu.be/8ZTlaZUvooI): #HealthTips #NutritionForWellness #EvidenceBasedDietnutrition tips, improve health, evidence-based nutrition, healthy lifestyle, diet tips, healthy eating, m...

 

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1305991476408352809) (56 messages🔥🔥):

> - `NotebookLM usage tips`
> - `Podcast generation issues`
> - `Data security concerns`

- **NotebookLM Tips for Summaries**: A user sought advice on using NotebookLM for summarizing texts, especially for college literature reviews, hoping for prompting tips or Discord thread references.
  
  - Another user recommended exploring capabilities such as using synthetic datasets to protect sensitive information.
- **Podcast Generation Hurdles**: Multiple users discussed challenges related to generating podcasts from specific sources, with one user reporting difficulties when using .md file formats.
  
  - Suggestions included using PDF or Google Docs formats, leading to a resolved issue with podcast focus after switching file types.
- **Data Security in NotebookLM**: Concerns arose regarding whether data uploaded to Gemini is secure, with clarification that paid accounts ensure data security, while free accounts may not.
  
  - Users expressed caution about uploading sensitive data, emphasizing the need for confidentiality when using the platform.

**Links mentioned**:

- [no title found](https://notebooklm.google.com/notebook/3b8029e5-50c7-4007-a6d7-aba33125f8d7/audio): no description found
- [The AI Bubble is About to Pop. Here's Who Dies First](https://chrisbora.substack.com/p/the-ai-bubble-is-about-to-pop-heres): The $600B Bloodbath Nobody's Ready For (And The Hidden $3T Opportunity)

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1305987866949587044) (64 messages🔥🔥):

> - `Supermaven joins Cursor`
> - `Windsurf Editor launch`
> - `Perplexity ads introduction`
> - `Mira Lab updates`
> - `RAG future predictions`

- **Supermaven joins Cursor**: Supermaven has officially joined [Cursor](https://supermaven.com/blog/cursor-announcement) to enhance their AI coding editor capabilities. The collaboration is expected to leverage Supermaven’s AI-assisted features for better software development experiences.
  
  - Community reactions were mixed, with some users noting Supermaven's prior effectiveness while expressing surprise at the transition.
- **Windsurf Editor launch**: Codeium launched the Windsurf Editor, touted as the first agentic IDE that integrates AI collaboration with independent task execution. Users reported a positive first impression, highlighting its ability to maintain developer flow.
  
  - However, some users noted that it may not yet outperform established tools like Copilot in certain aspects.
- **Perplexity ads introduction**: Perplexity is experimenting with ads on its platform, introducing “sponsored follow-up questions” alongside their search results. They partnered with brands like Indeed and Whole Foods to monetize their AI-powered search engine.
  
  - The move aims to create a sustainable revenue-sharing program, recognizing that subscriptions alone are insufficient.
- **Mira Lab updates**: Mira Lab, initiated by ex-OpenAI CTO Mira Murati, is forming a new team to focus on AI technologies. Reports suggest that at least one OpenAI researcher is joining this venture, indicating strong interest in its formation.
  
  - The lab's aim is to take on ambitious projects leveraging the expertise of its founding members.
- **RAG future predictions**: There’s growing speculation that Retrieval-Augmented Generation (RAG) will shift from being used primarily for Q&A to more sophisticated report generation in the coming months. A post by Jason Liu draws attention to the potential value of this transition for businesses.
  
  - The broader sentiment suggests that RAG's evolution will enhance how companies leverage AI in documentation and reporting.

**Links mentioned**:

- [Supermaven joins Cursor](https://supermaven.com/blog/cursor-announcement): Supermaven is joining Cursor to build the best AI code editor.
- [Supermaven Joins Cursor](https://www.cursor.com/blog/supermaven): We're excited to announce that Supermaven is joining Cursor.
- [Tweet from Codeium (@codeiumdev)](https://x.com/codeiumdev/status/1856741823768879172?s=46): Today we’re excited to launch the Windsurf Editor - the first agentic IDE, and then some 🏄 In Windsurf, we have given the AI a previously unseen combination of deep codebase understanding, powerful ...
- [Tweet from Greg Brockman (@gdb)](https://x.com/gdb/status/1856441156281753908): longest vacation of my life complete. back to building @OpenAI.
- [Predictions for the Future of RAG - jxnl.co](https://jxnl.co/writing/2024/06/05/predictions-for-the-future-of-rag/): no description found
- [Bloomberg - Are you a robot?](https://www.bloomberg.com/news/articles/2024-11-13/openai-nears-launch-of-ai-agents-to-automate-tasks-for-users?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTczMTUyODYxOCwiZXhwIjoxNzMyMTMzNDE4LCJhcnRpY2xlSWQiOiJTTVdOQURUMEcxS1cwMCIsImJjb25uZWN0SWQiOiJFODA3NUYyRkZGMjA0NUI2QTlEQzA5M0EyQTdEQTE4NiJ9.TTJZiuo4Nk2U295FHBFsxeN0YGznZJ32sHnNReQmEjM): no description found
- [Bloomberg - Are you a robot?](https://www.bloomberg.com/news/articles/]): no description found
- [Tweet from Nous Research (@NousResearch)](https://x.com/nousresearch/status/1856417883934601246?s=46): Today we are launching the Forge Reasoning API Beta, an advancement in inference time scaling that can be applied to any model or a set of models, for a select group of people in our community. https...
- [Tweet from Stephanie Palazzolo (@steph_palazzolo)](https://x.com/steph_palazzolo/status/1856360400721162745?s=46): New w/ @erinkwoo: At least 1 OpenAI researcher has taken up ex-CTO Mira Murati's offer to join her new startup, which she's working on with former OAI researchers Barret Zoph and Luke Metz. A...
- [Tweet from Aman Sanger (@amanrsanger)](https://x.com/amanrsanger/status/1856432315263836486): So excited to be working with @jbfja and the Supermaven team! There are only two companies in the world that have shipped models with 1M+ token windows: Google Deepmind and Supermaven. Quoting Curso...
- [Tweet from Alexander Doria (@Dorialexander)](https://x.com/dorialexander/status/1856751121101934723?s=61): Releasing two trillion tokens in the open. https://huggingface.co/blog/Pclanglais/two-trillion-tokens-open
- [Tweet from morgan — (@morqon)](https://x.com/morqon/status/1856691685352194072?s=46): another lab struggles: gemini 2.0 “is not living up to internal expectations”
- [Bloomberg - Are you a robot?](https://www.bloomberg.com/news/articles/2024-11-13/openai-nears-launch-of-ai-agents-to-automate-task): no description found
- [The Cursor + Supermaven Interview](https://youtu.be/ruy6cyBu0PA?si=mFuNF5OxUd-CXNPT): A very important interview with some very cool peopleTy Ph4se0n3 for the edit!
- [Windsurf Editor by Codeium](https://codeium.com/windsurf): Tomorrow's editor, today. Windsurf Editor is the first AI agent-powered IDE that keeps developers in the flow. Available today on Mac, Windows, and Linux.
- [Tweet from Anthropic (@AnthropicAI)](https://x.com/anthropicai/status/1856752093945540673?s=61): New research: Jailbreak Rapid Response. Ensuring perfect jailbreak robustness is hard. We propose an alternative: adaptive techniques that rapidly block new classes of jailbreak as they’re detected. ...
- [This free AI image editor changes everything](https://youtu.be/PCL9SAlHqzw?si=FYkX8DfDRR2zDGEd): Omnigen installation & tutorial. Open-source tool to edit images by prompting. #aitools #ai #aiart Thanks to our sponsor Abacus AI. Try their new ChatLLM pla...
- [Better than Cursor? Future Agentic Coding available today](https://youtu.be/824Fyh146_w?si=sS5lsRATvxmVgZ1Y): AI Coding Agent that KNOWS your codebase - Build production app with WindsurfFree E-book of how to learn to code with chatGPT: https://clickhubspot.com/znrx?...
- [Gwern Branwen - How an Anonymous Researcher Predicted AI's Trajectory](https://youtu.be/a42key59cZQ?si=m3IoFVRf4dCSl4XW): Gwern's blog: https://gwern.net/Gwern is a pseudonymous researcher and writer. After the episode, I convinced Gwern to create a donation page where people ca...
- [GitHub - VectorSpaceLab/OmniGen: OmniGen: Unified Image Generation. https://arxiv.org/pdf/2409.11340](https://github.com/VectorSpaceLab/OmniGen): OmniGen: Unified Image Generation. https://arxiv.org/pdf/2409.11340 - VectorSpaceLab/OmniGen
- [Anysphere acquires Supermaven to beef up Cursor | TechCrunch](https://techcrunch.com/2024/11/12/anysphere-acquires-supermaven-to-beef-up-cursor/): Anysphere, the company behind Cursor, has acquired AI coding assistant Supermaven for an undisclosed sum.
- [Perplexity brings ads to its platform | TechCrunch](https://techcrunch.com/2024/11/12/perplexity-brings-ads-to-its-platform/): AI-powered search engine Perplexity says it'll begin experimenting with ads on its platform starting this week.

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1306347362083409962) (2 messages):

> - `Building AI for the Enterprise`
> - `Windsurf Editor Launch`
> - `Cascade Flow Experience`
> - `General Access Tools`
> - `Community Feedback`

- **New Guest Post on Enterprise AI**: The latest guest post features insights on how to monetize AI by adopting an **Enterprise Infrastructure Native** mindset, emphasizing building with that approach from the outset.
  
  - The post encourages immediate action, noting that *premature optimization is not premature here*.
- **Launch of Windsurf Editor**: Codeium announced the launch of the **Windsurf Editor**, touted as the first agentic IDE providing deep codebase understanding and real-time awareness of actions.
  
  - This tool merges the collaborative qualities of copilots with the autonomous strength of agents, creating an experience called **Cascade** that enhances workflow.
- **No Waitlist for Windsurf Access**: The Windsurf Editor is available to everyone without **waitlists or invite-only** access, allowing users to dive in immediately.
  
  - This approach emphasizes user inclusion, stating that **general access is how it should be**.

**Links mentioned**:

- [Tweet from crystal (@crystal)](https://x.com/latentspacepod/status/185): adam hates my username.
- [Tweet from Latent.Space (@latentspacepod)](https://x.com/latentspacepod/status/1856788504321429519): 🆕 post: Building AI for the Enterprise https://latent.space/p/enterprise The long-awaited third (!) guest post from @_anshulr on how to make $$$ with AI: Build with from the start with an Enterpris...

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1306006746279444542) (2 messages):

> - `XOR Tensor Cores`
> - `Beamforming Algorithms`
> - `Lambda as a Stable Option`

- **XOR Tensor Cores Find Use in Beamforming**: A discussion highlighted that **XOR Tensor Cores** can be utilized in **beamforming algorithms** for **ultrasonic scans**, showcasing their potential in this application.
  
  - This application emphasizes the flexibility of XOR Tensor Cores beyond traditional use cases in processing.
- **Lambda Recognized as a Reliable Choice**: It was noted that **Lambda** stands out as a **solid stable option** in the current landscape, indicating user confidence in its reliability.
  
  - Members seem to favor it for its consistent performance, which remains a pivotal factor for users.

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1305991754939367475) (13 messages🔥):

> - `Triton Kernel Functionality`
> - `Libstdc++ Issues in Triton with Conda`
> - `GEMM Kernel Design in Triton`
> - `Warp Memory Alignment Error`
> - `Lowering ttir to Hexagon Dialect`

- **Triton Kernel Successfully Copies Tensors**: A member shared their progress with Triton by successfully copying a source tensor to a destination tensor using a Triton kernel, confirming it worked with a specific Python function.
  
  - They requested assistance on adjusting the kernel to respect their custom `block_mapping` structure for the tensor copy operation.
- **Crashing Issues with Torch Compile**: Some users reported crashes when building Triton from source while using `torch.compile`, particularly with nightly builds of PyTorch.
  
  - This was suspected to be tied to `libstdc++` issues, requiring files from system directories to be copied into Conda's environment for Triton to load.
- **Efficient GEMM Kernel Design Quandary**: A member sought advice on designing an efficient GEMM kernel in Triton for small sizes under 16, including specific dot product scenarios.
  
  - Another user suggested using manual dot product computation for size 1 and employing `BLOCK_SIZE_M=16` with masking for larger sizes.
- **Warp Memory Alignment Error Issue**: A user opened a GitHub issue regarding a memory alignment error encountered when manually launching a compiled PTX with Triton.
  
  - This issue included details of the kernel they attempted to compile and sought community assistance.
- **Finding Information on Lowering to Hexagon Dialect**: A member inquired about resources related to lowering ttir to the Hexagon dialect, sharing a related [YouTube video](https://www.youtube.com/watch?v=odnyMYSTxoU).
  
  - They aimed to gather community insights and specifics on the process as they navigate using this feature.

**Links mentioned**:

- [triton GEMM with size < 16 · Issue #5138 · triton-lang/triton](https://github.com/triton-lang/triton/issues/5138): Describe the issue How could triton support GEMM with size < 16, such as dot (1, D) and (D, D), (3, D) and (D, D) or any number less than 16? I have seen many methods suggesting that when the size ...
- [Warp memory alignment error when manually launching compiled PTX · Issue #5136 · triton-lang/triton](https://github.com/triton-lang/triton/issues/5136): Describe the bug I am using Triton to compile a batched matmul kernel into a PTX as shown here: import torch import triton import triton.language as tl import os KERNEL_PATH = "src/triton_kernels...

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1306108340694159370) (7 messages):

> - `torch.compile memory usage`
> - `dynamic shapes impact`
> - `profiling GPU memory allocations`
> - `direct access to GPU`

- **torch.compile impacts peak memory usage**: Members are discussing cases where **torch.compile()** might increase **peak memory usage**, leading to **out-of-memory (OOM)** errors.
  
  - One mentioned that even without the `reduce-overhead` flag, they observed a **3-16%** increase in memory usage, especially with **dynamic shapes**.
- **Profiling techniques for memory usage**: A suggestion was made to use **nsys profile** with specific **nvtx** ranges to analyze current GPU memory usage and track allocations effectively.
  
  - There seems to be some uncertainty about whether **CUDA graphs** in PyTorch contribute to memory increases without using the `reduce-overhead` flag.
- **Direct GPU access inquiry**: A member asked for guidance on how to achieve **direct access to GPU**.
  
  - No responses or solutions were provided regarding this particular inquiry.

 

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/) (1 messages):

0xredj: [https://latent.toys](https://latent.toys)

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1306286531643965544) (1 messages):

> - `SEMROM job opening`
> - `Quantization methods`
> - `Inference framework development`
> - `Collaborative ML and hardware work`
> - `Open-source contributions`

- **SEMROM seeks ML Quantization Engineer!**: SEMROM, a **venture-backed startup**, is looking for an engineer who specializes in **quantization** and enjoys bridging machine learning with hardware for Edge devices.
  
  - *Check out the full job description* [here](https://semron.jobs.personio.de/job/1433496?language=de&display=en) to learn more about developing a scalable inference framework.
- **Innovating with Latest Quantization Techniques**: The role involves applying and innovating with cutting-edge quantization methods like **AdaRound**, **BRECQ**, **GPTQ**, and **QuaRot**.
  
  - Candidates should have a strong background in **PyTorch** and experience developing efficient custom **CUDA kernels**.
- **Collaborate Across Teams at SEMRON**: The position requires collaboration with ML, compiler, and hardware teams to adapt quantization algorithms to SEMRON's unique needs.
  
  - This cross-functional approach aims to refine the **inference framework**, ensuring it is finely tuned for upcoming hardware.
- **Opportunity to Contribute to Open-Source**: As part of the team, the engineer will make fundamental architectural decisions and contribute to *upstream open-source projects*.
  
  - This aspect of the role reflects SEMRON’s commitment to community engagement and innovation.

 

**Link mentioned**: [ML Quantization Engineer | Jobs bei SEMRON](https://semron.jobs.personio.de/job/1433496?language=de&display=en): Wir sind SEMRON, ein Startup aus Dresden, und wir entwickeln innovative Mikrochips für KI-Anwendungen.

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1306252662798422046) (2 messages):

> - `Lecture Access`
> - `Discord Channel Information`

- **Finding Lecture Access Information**: A user inquired about where to access the **lectures**.
  
  - The response pointed to a specific **Discord channel** for further details, referencing <#1198358627594023014>.
- **Response to Lecture Inquiry**: Additional clarification was provided regarding the access to the **lectures** within the channel.
  
  - The channel serves as a central point for users to get information and updates about available lectures.

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1306003618633355304) (9 messages🔥):

> - `AI-generated images`
> - `Food generation`
> - `Identity impersonation`
> - `Food cycle`
> - `Bot verification`

- **Curiosity about AI-generated images**: A user questioned if an image was AI-generated, prompting another member to clarify that it was not.
  
  - The discussion sparked curiosity about whether people can simulate interactions and create fake content.
- **Text-to-food creativity**: A member humorously described their ability to generate food through prompts, likening themselves to a 'text-to-food model'.
  
  - They jokingly added that this process led to a 'food-to-poop' cycle, suggesting a humorous take on food generation.
- **Discussion on cycles of life**: A user expanded on the humorous concept of life cycles, stating 'poop-to-earth and earth-to-food' as part of a natural cycle.
  
  - This reflection connected back to the earlier conversation about generating food and highlighted a playful perspective on life processes.
- **Concerns about identity verification**: One user expressed skepticism about verifying whether another user is a bot, seeking reassurance.
  
  - This raised questions about identity and authenticity in online interactions.

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1306258526179688510) (3 messages):

> - `MI300X performance`
> - `FP16 peak throughput`
> - `Triton on AMD GPUs`

- **Quest for 800 TFLOPS on MI300X**: One user inquired whether anyone has achieved **800 TFLOPS** on the **MI300X**, expressing that their attempts with **CK** did not yield good results.
  
  - They are seeking insights on optimization methods to reach this performance milestone.
- **600 TFLOPS is the peak for FP16**: Another user noted that **600 TFLOPS** seems to be the peak of **FP16** performance on the MI300X.
  
  - This suggests limitations in reaching higher teraflop rates for this specific precision.
- **Insights from the Triton YouTube Talk**: A user shared a link to a [YouTube video titled 'Triton on AMD GPUs'](https://youtu.be/Lbm08twNTAQ?si=6Vwrkz8W0U2WTpf1&t=243), where Lei Zhang and Lixun Zhang discuss Triton support for AMD.
  
  - The talk showcases clever optimization techniques around chiplets, providing valuable insights for GPU performance improvements.

 

**Link mentioned**: [Triton on AMD GPUs](https://youtu.be/Lbm08twNTAQ?si=6Vwrkz8W0U2WTpf1&t=243): Lei Zhang and Lixun Zhang talk to Triton support for AMD. This talk shows off some very clever optimization techniques around chiplets and also instruction s...

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1306045056028901440) (2 messages):

> - `Liger-Kernel v0.4.1 Release`
> - `Gemma 2 Support`
> - `CrossEntropy Patching Fix`

- **Liger-Kernel v0.4.1 has been released**: The latest release, **v0.4.1**, of [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.1) introduces support for **Gemma 2** along with a patch for CrossEntropy issues.
  
  - This release marks a significant milestone with contributions from @Tcc0403, who resolved long-standing issues regarding softcapping in fused linear cross entropy.
- **Exciting New Features in v0.4.1**: The addition of **Gemma 2 Support** has been a highly anticipated feature, enabling enhanced functionality.
  
  - The release also includes improvements such as fixes for **GroupNorm** which streamline operations further.

 

**Link mentioned**: [Release v0.4.1: Gemma 2 Support, CrossEntropy Patching FIx, and GroupNorm · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/releases/tag/v0.4.1): Highlights Gemma 2 Support: The long pending gemma 2 is finally supported thanks to @Tcc0403! He has implemented the nasty softcapping in fused linear cross entropy (#320) and discovered the conv...

 

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1306223229714698340) (6 messages):

> - `Flash Attention Overview`
> - `Video Feedback`
> - `CUDA and Triton Programming`
> - `Multi-Head Attention`
> - `Community Praise`

- **Deriving Flash Attention from Scratch**: The video focuses on **deriving and coding Flash Attention from scratch**, explaining CUDA and Triton with no prior knowledge required.
  
  - Topics include **Multi-Head Attention**, **Softmax**, and the **Jacobian** of both matrix multiplication and the Softmax operation.
- **Community Reacts to Video Length**: A member humorously described the video as a **casual 7.5 hour** watch, indicating the depth and extensiveness of the content.
  
  - Another member praised a prior video on **quantization**, highlighting the creator’s engaging style.
- **Impact and Recognition**: A member expressed admiration for the creator, noting, *So many people I admire have been saying great things about your videos!*
  
  - The creator acknowledged the compliment, stating, *It means A LOT coming from you*, showing appreciation for the community.
- **Casual Learning Environment**: The conversation reflects a casual learning environment, with the creator mentioning they were *pretty bored this weekend* and chose to make the video.
  
  - Another member remarked, *Sensei approves as well*, indicating a sense of camaraderie and mutual respect within the group.

 

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1306304361349255269) (4 messages):

> - `Discord Leaderboard UX`
> - `Bot Development`
> - `Dataset Collection`

- **Ideas for Discord Leaderboard UX**: A GitHub issue titled [Ideas for what to do next](https://github.com/gpu-mode/discord-cluster-manager/issues/6) discusses how the **Discord leaderboard** will be rendered, including potential use for **slash commands** that autofill with script expectations and additional details like kernel name or GPU type.
  
  - The discussion emphasizes the need for contributions to evolve the UX, encouraging participants to explore the issue and reach out for further discussions about implementation.
- **Focus Areas on Bot and Dataset Collection**: Team members highlighted that the **main focuses** at this time are the **bot development** and **dataset collection** efforts.
  
  - One member expressed intent to collaborate with another to connect for potential input on the tracking issue regarding these focuses.

 

**Link mentioned**: [Ideas for what to do next · Issue #6 · gpu-mode/discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager/issues/6): Discord based leaderboard UX How does the leaderboard get rendered @AndreSlavescu Slash commands that autofill that a script is expected and maybe more information like the kernel name or gpu type ...

 

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1306025391999418389) (15 messages🔥):

> - `TK improvements`
> - `H100 DSMEM limitations`
> - `Cooperative groups in CUDA`
> - `Semaphore synchronization`
> - `TMA instruction usage`

- **TK receives positive feedback and bug resolution**: Members expressed gratitude for the work on **ThunderKittens** and fixed a user's issue attributed to a simple error.
  
  - One member noted they were excited to implement changes and share future PRs.
- **H100 DSMEM reduce supports limited types**: Discussion revealed that **H100** does not support **DSMEM** reductions in float or bfloat types, specifically only handling integers.
  
  - A member shared their findings regarding `cp.reduce.async.bulk` limitations targeting integer types when using `.shared::cluster`.
- **Cooperative group execution of TMA instructions**: Clarification was provided that TMA instructions in the **TK tests** are generally executed by the whole warp, but internally mask off all but the first lane.
  
  - This led to a discussion about the approach of calling these instructions from a single thread and the rationale behind it.
- **Semaphore order of operations is critical**: Members highlighted the importance of order when calling semaphore operations and cluster synchronizations in DSMEM code.
  
  - Advised restructuring the semaphore calls and synchronizations to ensure proper functionality and prevent hanging issues.
- **Upcoming PR contains testing for integers**: A member indicated plans to finalize testing code for integers before submitting a PR related to **TK** development.
  
  - Excitement was shared about the opportunity to review the upcoming changes that involve improvements to the existing code.

 

**Link mentioned**: [ThunderKittens/tests/unit/warp/memory/tile/dsmem.cu at 06d654a0858840e006d428cd96aac2cd0d19ca25 · HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens/blob/06d654a0858840e006d428cd96aac2cd0d19ca25/tests/unit/warp/memory/tile/dsmem.cu): Tile primitives for speedy kernels. Contribute to HazyResearch/ThunderKittens development by creating an account on GitHub.

 

---

### **GPU MODE ▷ #**[**edge**](https://discord.com/channels/1189498204333543425/1303441437592911912/1306051792261615707) (1 messages):

> - `Quantization Techniques`
> - `Algorithm Optimization`
> - `Deployment Strategies`
> - `Memory Efficiency`
> - `Hybrid Processing`

- **Optimizing Post-Training with Quantization**: Discussed methods such as **quantization**, **distillation**, and **speculative decoding** for model optimization after training.
  
  - A suggestion was made to eliminate **attention layers** and earlier **feed forward networks** as it minimally impacts performance.
- **Algorithmic Improvements via Flash Attention**: Highlighted the use of **Flash Attention** and techniques like **loop unrolling** and **inline assembly** for enhancing algorithm performance.
  
  - The conversation mentioned that **batching** should be adjusted based on specific use cases for optimal results.
- **Evaluating Deployment Options**: Explored the importance of selecting the right **computational devices** (GPU, CPU, NPU) for deployment considerations.
  
  - A note was made on the complexity of implementing **hybrid tensor parallelism** for greater memory efficiency and handling long-context applications.
- **Hybrid Processing for Better Efficiency**: Considered utilizing a **hybrid cloud/local processing** approach for distributed inference methods like **Petals**.
  
  - *Applicability varies* by circumstance, suggesting careful consideration of this strategy.

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1306013250688651344) (29 messages🔥):

> - `Tinygrad distributed approach`
> - `Multi-node FSDP`
> - `Data handling in Cloud`
> - `Device-to-device communication`
> - `Performance concerns with sharding`

- **Tinygrad explores its distributed approach**: A user inquired about Tinygrad's current strategy for distributed computing, particularly regarding its handling of FSDP and whether it supports multi-node setups.
  
  - Another user mentioned an open bounty on FSDP as a potential resource and discussed the scalability challenges of current implementations.
- **Clarification on Cloud's data management**: It was discussed that while cloud capabilities may allow using thousand GPUs across different machines, optimal performance still hinges on fast connectivity and effective all-reduce implementations.
  
  - Concerns were raised about the efficiency of having a single machine orchestrate data management and processing during training runs.
- **Device-to-device transfers under Cloud**: George Hotz indicated that device-to-device communication is handled in Tinygrad's Buffer via the `transfer` function, suggesting potential ease in extending this to cloud setups.
  
  - He humorously noted it could be accomplished with just a few lines of code, indicating simplicity in implementation.
- **User interest in data forwarding methods**: Codeman3786 expressed eagerness to contribute a data forwarding method for direct device-to-device communication in CloudDevice but faced uncertainty regarding the abstraction levels involved.
  
  - They highlighted ongoing profiling efforts during experiments across a cluster, pointing to the need for deeper understanding.
- **Concerns regarding performance with Cloud sharding**: There was a discussion on the need for clarity on whether users are machine-sharded or cloud-sharded to avoid unexpected performance issues and costs during slower sync operations.
  
  - The conversation underscored the importance of efficient data handling strategies to maintain performance levels across different configurations.

 

**Link mentioned**: [How multigpu training works](https://mesozoic-egg.github.io/tinygrad-notes/multigpu.html): Tutorials on tinygrad

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1306131918818906122) (32 messages🔥):

> - `Unsigned Tensor min() bug`
> - `Asynchronous mode for Tinygrad inference`
> - `Using BitwiseNot for integer operations`
> - `Pull Request updates`
> - `Realizing Tensors`

- **Unsigned Tensor min() bug needs a PR fix**: A user identified a bug in the **min()** function for unsigned tensors when calculating minimum values, particularly with zeros, and suggested flips to resolve it.
  
  - *Rezvan* submitted a PR with failing tests, mentioning that it's more complex due to potential **infs** and **nans**.
- **Discussion on Asynchronous Inference Modes**: A discussion highlighted the potential for **Tinygrad** to implement asynchronous mode for handling multiple models concurrently without blocking.
  
  - The proposal involved using a waiting mechanism for model realization and capturing outputs in an **outputs** array.
- **BitwiseNot use in integer fixes**: A member suggested using **BitwiseNot** to handle unsigned integers in the min calculation, raising concerns about its impact on float types.
  
  - *ChenYuy* adjusted the min fix to implement **BitwiseNot**, proposing to extend this technique to **argmin** and **minimum** functions.
- **Updates on PR for the unsigned min fix**: After reviewing a PR for the unsigned min fix, a contributor noted a requirement for the input to be a **2D array** to pass all tests.
  
  - There was agreement on continuing improvements and ensuring the implementation is robust for various tensor dimensions.
- **Testing operations in Tinygrad**: *ChenYuy* suggested raising an error on **Ops.IF** to determine which tests would fail, aiding in tensor-level operations understanding.
  
  - This method would help gain insights on how the operations interact and improve clarity on existing issues in the tests.

**Links mentioned**:

- [t - Overview](https://github.com/t): t has 14 repositories available. Follow their code on GitHub.
- [fix: Tensor min function for unsigned ints by bjsi · Pull Request #7675 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7675/commits/6c1092cefc98c87edfe9516f3887d6789351140f): Uses the flip idea from discord to fix the min function for unsigned ints

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1305997985397608549) (28 messages🔥):

> - `AI language models and date awareness`
> - `Blaze AI for content creation`
> - `AI songwriting tools`
> - `ChatGPT's UI control system`
> - `Copy-paste issues on mobile`

- **AI models struggle with date accuracy**: A discussion highlighted that models like **Gemini** and **Claude** provided incorrect current dates, while **ChatGPT** responded accurately.
  
  - One user suggested the distinction may lie in how system prompts are set up, with **ChatGPT** managing to infer the date correctly in some contexts.
- **Curious About Blaze AI for Marketing**: A member inquired about the effectiveness of **Blaze AI** for content creation, especially for marketing purposes.
  
  - Feedback included a note about the initial **configuration** time needed to customize the platform to specific needs.
- **Seeking AI Tools for Songwriting**: One member was looking for recommendations on **free AI tools** that can create songs with lyrics for a special occasion.
  
  - Another member mentioned **Suno** as a potential option, though it has limited daily generations available.
- **Innovative AI-Driven UI Control System**: A user shared details about their project allowing **ChatGPT** to control a computer’s UI through methods like cursor movements and decision-making with a tech stack including **OCR** and **Python**.
  
  - The approach aimed to blend AI with user workflows for enhanced automation, prompting inquiries about the project's code and comparisons to other AI solutions.
- **Mobile Copy-Paste Problems Persist**: A member raised concerns about ongoing **copy-paste** issues on mobile platforms, stating it has been ineffective for several weeks.
  
  - The challenges continued to affect user experience and functionality across mobile applications.

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1306032001496907836) (11 messages🔥):

> - `Prompt Engineering`
> - `ChatGPT o1-preview Feedback`
> - `Model Limitations`
> - `Using VPNs`
> - `Tinkering with Requests`

- **Navigating the Art of Prompt Engineering**: Members discussed the complexities of writing effective prompts, with suggestions to refine and seek help if needed, especially when outputs don’t meet expectations.
  
  - *One user noted*, "it's like teaching a barber how to cut your hair correctly" when frustrations arise from needing to guide the model.
- **ChatGPT o1-preview Impresses Users**: Feedback suggested that the **ChatGPT o1-preview** is demonstrating greater creativity and tailored responses compared to previous versions.
  
  - *One user expressed gratitude*, mentioning its ability to anticipate inputs, making the experience more personalized.
- **Model Limitations Affecting User Experience**: Concerns were raised over how the model stops producing effective results after multiple requests, hinting at potential issues with **block configurations**.
  
  - *A community member questioned*, "How to fix this?" indicating a desire for solutions to improve interaction longevity.
- **VPNs Are Legal for Bypassing Blocks**: A discussion highlighted that while scanning the internet is legal for countries, users can legally use **VPNs** to bypass restrictions or blocks.
  
  - *One member emphasized*, "Your block configuration is useless for the intended purpose," pointing to the need for effective solutions.
- **Tinkering with Requests Within Limits**: Members shared insights about wanting to tinker with prompts but feeling hindered by model limits and subscription constraints.
  
  - One expressed a need for better results, likening the interaction to a journey where learning to prompt effectively is a skill to develop.

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1306011250769268850) (5 messages):

> - `Scratchpad Techniques`
> - `Structured Output`
> - `Prompt Management Challenges`

- **Understanding Scratchpad Techniques**: A member highlighted that a scratchpad acts as a **pseudo-CoT technique**, allowing the LLM to write out its thoughts while working toward a solution.
  
  - Another member expressed enthusiasm, stating it could enhance **documentation** when incorporated into structured output.
- **Challenges in Structured Output Organization**: A member raised concerns about ensuring the **scratchpad** is completed first, as their prompt manager was **reordering structured output fields**.
  
  - This highlights potential issues in workflow management that could affect consistency in generated content.

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1306011250769268850) (5 messages):

> - `Scratchpad Technique`
> - `Prompt Management Issues`

- **Scratchpad Technique Enhances LLM's Thought Process**: A member described the scratchpad as a **pseudo-CoT technique**, where the LLM writes out its thoughts as it works toward a final solution.
  
  - Another member expressed interest in incorporating the scratchpad into structured output for **better documentation**.
- **Challenges in Scratchpad Output Order**: Concerns were raised about ensuring the scratchpad content is generated first, as the **prompt manager is reordering structured output fields**.
  
  - This issue highlights potential inconsistencies in the use of scratchpads within workflows during LLM usage.

 

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1306145125747134505) (13 messages🔥):

> - `exllamav2`
> - `MAX Involvement`
> - `Error with Batched MatMul`

- **Exploring exllamav2 for LLM inference**: Members highlighted the [exllamav2 GitHub project](https://github.com/turboderp/exllamav2) as an excellent resource for improving LLM inference on MAX, featuring clean and optimized code.
  
  - *It supports ROCM for AMD* and handles multimodal models, making it an appealing option for integration.
- **Potential MAX integration with exllamav2**: Discussions included the possibility of MAX being involved in the exllamav2 project, which offers advanced features like batch inference and precise bpw settings.
  
  - *It’s a really good project but somehow goes under radar,* indicating a desire for greater awareness within the community.
- **Understanding Batched MatMul Error**: A member flagged an error related to batched MatMul: 'constraint failed: max rank for batched matmul is currently 4'.
  
  - This indicates constraints limits in the Mojo standard library's matrix operations, highlighting potential areas for clarification or improvement.

 

**Link mentioned**: [GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs](https://github.com/turboderp/exllamav2): A fast inference library for running LLMs locally on modern consumer-class GPUs - turboderp/exllamav2

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1305986171909836840) (33 messages🔥):

> - `Mojo JIT Compiler`
> - `MAX Platform Overview`
> - `UnsafePointer in Mojo`
> - `Mojo's Performance Capabilities`
> - `Mana Project References`

- **Mojo JIT Compiler's Binary Size Matters**: Shipping the **JIT compiler** could be viable if the binary is small and can interop with precompiled binaries, but caution is advised against providing source for all dependent apps.
  
  - One member emphasized that although **MLIR can be shipped**, the compiler is essential for achieving native code execution.
- **Introducing the MAX Platform**: The **MAX** platform is described as a unified set of APIs and tools designed to build and deploy high-performance AI pipelines, featuring tools like **MAX Engine** for model execution.
  
  - A member linked to the [documentation](https://docs.modular.com/max/#how-max-works), highlighting its robust capabilities for deploying low-latency inference pipelines.
- **Understanding UnsafePointer's Risks**: The **UnsafePointer** in Mojo allows for invoking undefined behavior which may lead to catastrophic memory safety issues, as pointed out by a member detailing possible dangerous outcomes.
  
  - Another member noted that Mojo has stricter pointer rules compared to C/C++, helping to minimize risks like type punning.
- **Mojo's Future in Low-Level Programming**: While some members are already experimenting with game engines, it was clarified that creating bootloaders in Mojo may take additional effort due to assembly requirements.
  
  - The language's potential for various systems programming tasks hinges on its development, although immediate achievements in game engine work are already underway.
- **Mana Project Names Abound**: Members joked about the name **Mana**, with references made to existing projects like [mana.js](https://github.com/bjorn/mana.js/) and another repository found at [3rd-Eden's mana](https://github.com/3rd-Eden/mana).
  
  - The banter reflected on the possibility of various projects adopting this name, indicating an ongoing trend in naming cultural phenomena in tech.

**Links mentioned**:

- [Microsoft Azure Network Adapter (MANA) overview](https://learn.microsoft.com/en-us/azure/virtual-network/accelerated-networking-mana-overview): Learn how the Microsoft Azure Network Adapter can improve the networking performance of Azure VMs.
- [What is MAX | Modular Docs](https://docs.modular.com/max/#how-max-works): An overview of the MAX platform, what it does, and what's included.

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1306023832490086502) (2 messages):

> - `GenAI pipelines`
> - `RAG workflows`
> - `Dynamic section retrieval`

- **Build GenAI pipelines with LlamaIndex!**: Learn how to build robust GenAI pipelines using LlamaIndex, [@qdrant_engine](https://twitter.com/qdrant_engine), and [@MLflow](https://twitter.com/MLflow) to advance RAG systems.
  
  - Discover how to streamline RAG workflows, ensure performance consistency across model versions, and optimize indexing systems for efficiency with this [step-by-step guide](https://t.co/aZ4GIyGRQM).
- **Introducing Dynamic Section Retrieval in RAG**: We're excited to feature a new RAG technique - **dynamic section retrieval**, which enables the retrieval of entire contiguous sections from documents instead of fragmented chunks.
  
  - This approach addresses a major pain point highlighted by the community regarding multi-document RAG, further discussed in this [article](https://t.co/vP2J2arhf4).

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1306160133285351546) (35 messages🔥):

> - `Vocera Launch on Product Hunt`
> - `RAG vs Reporting Debate`
> - `Chatbots in Corporate Settings`
> - `Blockchain Engineering Expertise`

- **Vocera Goes Live on Product Hunt**: Vocera launched on [Product Hunt](https://www.producthunt.com/posts/vocera) to help AI developers test and monitor voice agents 10X faster.
  
  - The team requested support and feedback to enhance visibility and impact in the AI community.
- **RAG vs Reporting – A Dialogue**: Discussions arose around the value of reports compared to RAG, highlighting that reports represent only 10% of problem-solving effort in corporations.
  
  - Participants emphasized the importance of information retrieval in the report generation process and cautioned against misleading marketing tactics.
- **Chatbots – Not a C-Suite Favorite**: Several members noted that upper management typically prefers report formats over chatbot interactions.
  
  - Despite this preference, chatbots are acknowledged as useful tools for conducting searches within organizations.
- **Insights from a Blockchain Engineer**: A user detailed their blockchain engineering expertise, covering technologies like Rust, Solidity, and EVM, alongside various application categories.
  
  - They also highlighted their front-end development skills with frameworks like React and Flutter, integrating them with smart contracts.

**Links mentioned**:

- [Tweet from jason liu (@jxnlco)](https://x.com/jxnlco/status/1856411798255333840): RAG is overrated. Reports are the real game-changer. It's not about saving time answering questions. It's about generating high-value decision-making tools that drive business outcomes. The ...
- [Vocera - Launch voice agents faster with simulation & monitoring | Product Hunt](https://www.producthunt.com/posts/vocera): Vocera helps AI developers build production-ready voice agents 10X faster. It generates adversarial scenarios, simulates realistic calls and gives actionable insights to your agents. It also monitors ...

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1306178493838655551) (1 messages):

> - `Vocera Launch`
> - `AI Voice Agents`

- **Vocera Goes Live on Product Hunt!**: Exciting news as **Vocera** is now live on [Product Hunt](https://www.producthunt.com/posts/vocera)! This tool allows developers to test and monitor **AI Voice Agents** automatically, speeding up the process of building production-ready agents by **10X**.
  
  - The team encourages feedback and support, expressing that it will help them reach more users who could benefit from **Vocera**.
- **Developer Engagement with Vocera**: The developers express their gratitude for the support of the community in sharing thoughts about **Vocera**. They highlight that user feedback is crucial for reaching more potential users who could benefit from the product.

 

**Link mentioned**: [Vocera - Launch voice agents faster with simulation & monitoring | Product Hunt](https://www.producthunt.com/posts/vocera): Vocera helps AI developers build production-ready voice agents 10X faster. It generates adversarial scenarios, simulates realistic calls and gives actionable insights to your agents. It also monitors ...

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1306080520144425021) (6 messages):

> - `Payment Management Functionality`
> - `Card Decline Issues`
> - `Cohere API Internal Server Error`
> - `Rerank API Best Practices`

- **Querying Payment Management's Auto-Charge**: A user inquired whether the payment management system provides an **auto-charge function** or if they must check and make manual payments based on usage.
  
  - *A user expressed confusion over payment protocols* regarding fluctuations in charges.
- **Constant Card Declines**: Another user faced repeated errors with their cards being **declined** while adding a payment method, despite trying two different cards.
  
  - The suggestion was made to reach out via email to [**support@cohere.com**](mailto:support@cohere.com) for assistance.
- **Internal Server Errors on API Calls**: A user reported receiving a **500 internal server error** from calls to the Cohere API, which has persisted for several days.
  
  - The error message referenced a report that had been filed with their developers for resolution.
- **Rerank API Query Performance Questions**: Inquiries were made about the **best practices** for the `query` field in the `v2/rerank` API, noting dramatic changes in `relevanceScore` with minor variations in the query.
  
  - Examples highlighted a query for **'volume rebates'** yielding a score of ~0.998 versus ~0.17 for **'rebates'**, sparking confusion about the model's responsiveness.

 

**Link mentioned**: [Rerank Best Practices — Cohere](https://docs.cohere.com/docs/reranking-best-practices#queries)): Tips for optimal endpoint performance, including constraints on the number of documents, tokens per document, and tokens per query.

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1306020522970058774) (19 messages🔥):

> - `API Errors`
> - `Recent Fix Deployment`
> - `Reranking Issues`
> - `Endpoint Troubles`
> - `Model Usage`

- **API Error 500's haunt users**: Multiple users reported experiencing **500 internal server errors** on API calls for several days, expressing confusion over whether the API had changed.
  
  - One user provided the **error ID** 'dbf879fc0e4f4bd7a98102ac41aa0566' for further investigation.
- **Deployment of Fix Yields Mixed Results**: A fix was deployed, but some users continued to face issues, prompting Michael to ask for **HTTP request details** to troubleshoot.
  
  - Despite the update, user feedback indicated that problems still persisted for some, highlighting the need for further investigation.
- **Clarification on Reranking Functionality**: Michael observed that not using the `return_documents` parameter in user's code snippet did not seem related to the ongoing errors, suggesting it might not be the source of the issue.
  
  - He requested the **error ID** from the user to dive deeper into the logs for better insight.
- **Talk of Upgrading to Production API**: One user mentioned upgrading to a **production API key**, suggesting anticipation for a more stable experience with Cohere.
  
  - This upgrade indicates a commitment to using Cohere’s services, contingent upon resolving current issues.
- **Reranking Functionality in Code**: A user shared their code snippet for reranking with Cohere, including parameters like 'query' and 'documents'.
  
  - Michael's request for the **error ID** emphasizes a collaborative approach to resolving user-specific issues effectively.

 

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1306350619287617616) (1 messages):

> - `Benchmarking Vision Language Action models`
> - `Collaboration on Robotics Research`
> - `VLM & VLA Performance Evaluation`

- **Benchmarking Vision Language & Action Models Released**: Today, a new paper titled [*Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks*](https://arxiv.org/abs/2411.05821) has been released, highlighting a collaboration between **Manifold**, **Georgia Tech**, **MIT**, and **Metarch AI**.
  
  - This research profiles emerging Vision Language Action models and evaluates how well these models can control robots across **20 different real-world tasks**.
- **Exploring State-of-the-Art VLMs like GPT4o**: The paper also covers some state-of-the-art Vision Language Models (VLMs) such as **GPT4o** and their performance in various robotics tasks.
  
  - This work represents the first step in building a broader benchmark for a new class of **multimodal action models**.
- **Feedback Invited on Research Insights**: The authors are seeking feedback on their findings and have shared relevant links for interested readers.
  
  - Check out the [Thread w/ Highlights](https://x.com/HarshSikka/status/1856739777208574151) for more insights into their work.
- **Access More Resources on Multinet**: For additional information, you can explore the project's [Website](https://multinet.ai/static/pages/Multinetv01.html) and [Code](https://github.com/ManifoldRG/MultiNet/tree/main).
  
  - These resources provide details on experimental setups and models used in the research.

 

**Link mentioned**: [Tweet from harsh (@HarshSikka)](https://x.com/HarshSikka/status/1856739777208574151)): Excited to share our new paper "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks" We evaluate how well VLM & VLA models can control robots across 20 different real-wor...

 

---

### **Cohere ▷ #**[**cohere-toolkit**](https://discord.com/channels/954421988141711382/1254901651081269268/1306009545394618448) (3 messages):

> - `ICS Support for Events`
> - `File Content Viewing Feature`

- **Adding ICS Support is Essential!**: @danylo_boiko expressed that it would be a **crime** not to implement support for **ICS files** given the number of events hosted on the Discord server.
  
  - This sentiment was echoed by members who responded positively, with one saying, *'yoo this is nice!'*
- **New Feature to View Uploaded Files**: @danylo_boiko also introduced the ability to **view the content** of uploaded files, although he jokingly mentioned lacking an *epic intro* for this feature.
  
  - Members responded enthusiastically, with one commenting, *'wow, love it!'*.

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1306107507722158080) (6 messages):

> - `Docker Images`
> - `Version Tagging`
> - `Axolotl Release Updates`

- **Docker Images Built and Tagged**: Docker images for the main branch have been built, and there's a reminder to tag them for version releases.
  
  - A member raised the point about ensuring proper tagging for organized version control and upcoming releases.
- **Upcoming 0.5.1 Release**: Several features are lined up for a **0.5.1 release** this week, alongside plans to tag the Docker images.
  
  - The team is confirming the correctness of the implementation in their [latest pull request](https://github.com/axolotl-ai-cloud/axolotl/pull/2051/files) to ensure proper version management.
- **Team Congratulations**: A member congratulated the team on their progress, highlighting a positive team spirit.
  
  - The team's ongoing collaboration and achievements were acknowledged among the group.

 

**Link mentioned**: [make sure to tag images in docker for tagged releases by winglian · Pull Request #2051 · axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/pull/2051/files): Description Motivation and Context How has this been tested? Screenshots (if appropriate) Types of changes Social Handles (Optional)

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**other-llms**](https://discord.com/channels/1104757954588196865/1104758057449308220/1306031642682589316) (5 messages):

> - `Qwen2.5 Coder Sizes`
> - `Performance Comparison of Qwen2.5`

- **Exploring Qwen2.5 Coder Sizes**: A member shared a [YouTube video](https://youtu.be/WPziCratbpc) for those wondering which **size of Qwen2.5 Coder** to choose.
  
  - The video compares various sizes and discusses their performance in detail.
- **Fast Generation with Qwen2.5**: Another member mentioned that the **generation speed** of Qwen2.5 seems impressive, questioning the setup being used.
  
  - They specifically inquired about a comparison against hosted models like **Sonnet 3.5**.
- **Qwen2.5 Running on 3090 Hardware**: The user **volko76** revealed they are running Qwen2.5 on a **NVIDIA 3090**, enhancing the generation speed.
  
  - This hardware choice underlines the performance boost for running demanding models.
- **YouTube Insights on Model Comparisons**: Volko76 directed attention to another [YouTube video](https://youtu.be/Xs0EkLYu6hw?si=95JJjVKRPknvEUsw) titled **'Qwen2.5 Coder 32B vs GPT4o vs Claude 3.5 Sonnet'**.
  
  - The video aims to determine which model is the best, offering a detailed analysis of their capabilities.

 

**Link mentioned**: [Qwen2.5 Coder 32B vs GPT4o vs Claude 3.5 Sonnet (new)](https://youtu.be/Xs0EkLYu6hw?si=95JJjVKRPknvEUsw): Let's see which model is the best

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general-help**](https://discord.com/channels/1104757954588196865/1110594519226925137/1306101448974798941) (11 messages🔥):

> - `qwen-2.5-7B-Instruct configuration`
> - `train_on_inputs clarification`
> - `SFTrainer and TRL comparison`

- **Config for qwen-2.5-7B-Instruct is simple**: A user inquired about a configuration for **qwen-2.5-7B-Instruct**, to which another member suggested using a regular qwen config found in `examples/qwen/qlora.yml`.
  
  - This friendly exchange highlights the community's willingness to share practical solutions.
- **Train_on_inputs flag leads to confusion**: The discussion touched on the implications of using `train_on_inputs = False`, where one member clarified the training mechanics without prompting sequences.
  
  - Referenced a [GitHub discussion](https://github.com/tloen/alpaca-lora/issues/255#issuecomment-1504111165) for further insights on this flag's functionality.
- **SFTrainer flags compared to axolotl**: Members discussed the functionality of **SFTrainer** in relation to the `train_on_inputs` flag available in axolotl.
  
  - A resource link was provided for more information on training only on completions in the context of the TRL framework.

**Links mentioned**:

- [train_on_inputs clarification · Issue #255 · tloen/alpaca-lora](https://github.com/tloen/alpaca-lora/issues/255#issuecomment-1504111165)): When you say train_on_inputs = False, I presume you mean to mask out the prompt, and train the loss only on the response that the model is supposed to produce. This is made slightly confusing by th...
- [Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only): no description found

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**announcements**](https://discord.com/channels/1104757954588196865/1113462842436354149/1306279405227741294) (3 messages):

> - `Release of version 0.5.0`
> - `Feedback and Issues Reporting`

- **Version 0.5.0 Finally Released!**: After months of hard work, the team announced the release of version **0.5.0**, which is now installable via `pip install axolotl`.
  
  - Updates include improvements and new features detailed on the [GitHub release page](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.5.0).
- **Community Celebrates Release**: Members expressed excitement about the new version, with one stating, *'Amazing!'* and another saying, *'Awesome!'*.
  
  - Such enthusiasm showcases the community's support and eagerness for the updates.
- **Open Call for Feedback**: The team encouraged users to share feedback and report any issues in a designated channel.
  
  - This invitation highlights their commitment to community engagement and continuous improvement.

 

**Link mentioned**: [Release v0.5.0 · axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.5.0): What's Changed fix(log): improve warning to clarify that lora_modules_to_save expect a list by @NanoCode012 in #1197 Add: colab example by @JohanWork in #1196 Feat/chatml add system message by @m...

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1306101720354394114) (1 messages):

> - `Forge Reasoning API`
> - `Nous Chat Evolution`

- **Nous Research Introduces Forge Reasoning API**: Nous Research has unveiled the [Forge Reasoning API](https://nousresearch.com/introducing-the-forge-reasoning-api-beta-and-nous-chat-an-evolution-in-llm-inference/) in beta, promising significant advancements in LLM inference capabilities.
  
  - This development marks a crucial step in enhancing reasoning processes within AI systems, showcasing a blend of newer models and optimized techniques.
- **Nous Chat Gets an Upgrade**: Accompanying the Forge API, **Nous Chat** is set to evolve, incorporating advanced features that improve user interaction and accessibility.
  
  - With this evolution, the emphasis lies on delivering a richer conversation experience powered by enhanced LLM technologies and methodologies.

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1306210845965025323) (8 messages🔥):

> - `DSPY Comparative Analysis`
> - `Zero Shot vs Few Shot Prompting`
> - `LLM Call Tracking`
> - `GitHub Notebooks`

- **DSPY Comparative Analysis Discussions**: Members discussed experiences with **DSPY** for conducting a comparative analysis on **zero shot** and **few shot prompting** in specific domains.
  
  - One member asked others about their use of the GitHub template to facilitate this analysis.
- **Shared DSPY Resources**: A member shared a link to a [Colab notebook](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb) to help others get started with DSPY.
  
  - Another member referenced a different notebook and highlighted its potential usefulness for their own project involving a code similarity tool.
- **Evaluating Tools with LLM Approaches**: A member mentioned evaluating **zero shot** versus **few shot** prompting in their attempts to create a code similarity tool using **LLM**.
  
  - They referred to another GitHub resource that they worked on to compare approaches and outcomes.
- **Inquiry on LLM Call Tracking**: **LLM call tracking** during evaluation processes such as Bootstrap, COPRO, and MIPROv2 was introduced as a potential topic of interest.
  
  - A member inquired about the feasibility of programmatically tracking these calls to assess performance during optimization.

**Links mentioned**:

- [Google Colab](https://colab.research.google.com/github/stanfordnlp/dspy/blob/main/intro.ipynb): no description found
- [dspy/examples/intro.ipynb at 421cdd1776041b61bde1c5f9ba3cff827cf5ac2a · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/blob/421cdd1776041b61bde1c5f9ba3cff827cf5ac2a/examples/intro.ipynb#L11`): DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1306047445276295188) (5 messages):

> - `Developer branch updates`
> - `UI improvements`
> - `Streamed responses handling`
> - `Open Interpreter`
> - `Text editor development`

- **Loving the Developer Branch Updates**: Members expressed their enthusiasm for the **latest updates** to the developer branch, highlighting its positive impact.
  
  - *It's the little things* that make a notable difference, sparking joy in the community.
- **UI Upgrades Make a Big Difference**: Feedback on the **improved UI** was overwhelmingly positive, with one member stating it is so much better.
  
  - *Handling of incoming streamed responses* no longer verges on inducing epileptic seizures, showcasing great work by the team 🚀.
- **Excitement for Open Interpreter**: A member shared their excitement, declaring that **Open Interpreter is awesome!**
  
  - They also queried about others' interest in **building text editors**, indicating a potential area for collaboration.

 

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1306246698007265393) (3 messages):

> - `OpenCoder`
> - `AI Bubble Predictions`
> - `GPU Investments`
> - `Dot-Com Bubble Parallels`

- **OpenCoder: A Game-Changer in Code Models**: The [YouTube video](https://www.youtube.com/watch?v=DurejOD5FTk) explores OpenCoder, an open-source cookbook for superior code language models, highlighting its cutting-edge advantages.
  
  - Viewers were intrigued by how OpenCoder potentially surpasses existing models and reshapes the landscape.
- **Impending AI Bubble Bust**: A post warns that the **AI bubble is about to pop**, comparing current trends to the **1999 dot-com bubble**, particularly in the context of massive **GPU investments** yielding minimal revenue.
  
  - The piece outlines the staggering implications of **$600 billion** in GPU spend against a mere **$3.4 billion** in revenue, suggesting a precarious future for today's AI enterprises.
- **The Parallels Between AI and Dot-Com Crashes**: The ongoing infrastructure buildup in AI mirrors strategies from the dot-com era, with companies heavily investing in hardware without monetization clarity.
  
  - It highlights the risks of emulating past **Pets.com** scenarios in an age where companies chase theoretical future demand without proven pathways to profit.

**Links mentioned**:

- [The AI Bubble is About to Pop. Here's Who Dies First](https://chrisbora.substack.com/p/the-ai-bubble-is-about-to-pop-heres): The $600B Bloodbath Nobody's Ready For (And The Hidden $3T Opportunity)
- [Why This Open-Source Code Model Is a Game-Changer!](https://www.youtube.com/watch?v=DurejOD5FTk): In this video, we'll be exploring OpenCoder, an open-source cookbook for top-tier code language models. Opencoder is a cutting-edge code model that surpasses...

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1306146574396559372) (4 messages):

> - `MongoDB dismissal`
> - `Social services and employment`
> - `EPOCH 58 COCK model updates`

- **MongoDB dismissal raises legal and ethical concerns**: A member expressed shock over their dismissal from **MongoDB**, detailing experiences of illegal practices like being fired over Slack and denied access to company resources during employment.
  
  - *They highlighted the failure to notify the Repubblica Italiana of their unemployment status, resulting in severe personal consequences* such as lack of access to healthcare and social services.
- **Struggles with social services post-dismissal**: The member detailed significant challenges in accessing **public employment agencies**, severance, and healthcare due to MongoDB's alleged failure to follow legal obligations.
  
  - *They recounted experiences of extreme hardship*, including homelessness and health issues, stemming from an inability to enroll in necessary social services.
- **EPOCH 58 COCK model developments**: A member shared updates about **EPOCH 58 COCK**, noting improvements with **vit-s** at **60M** parameters and enhanced model features.
  
  - They remarked that **legs are coming in** and the **cockscomb is becoming more defined**, signaling positive progress in model capabilities.

 

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1306354438662783048) (3 messages):

> - `Vision Language Action Models`
> - `Watermark Anything`
> - `Robotic Learning Tasks`
> - `AI Generators Performance`

- **New Paper on Vision Language Action Models Released**: A new paper titled '[Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks](https://arxiv.org/abs/2411.05821)' evaluates the performance of Vision Language Models across 20 different real-world tasks, showcasing collaborations among Manifold, Georgia Tech, MIT, and Metarch AI.
  
  - The work aims to profile this emerging class of models like **GPT4o**, marking a first step in a broader benchmark for multimodal action models.
- **GitHub Repository for Watermark Anything**: The project '[watermark-anything](https://github.com/facebookresearch/watermark-anything)' provides an official implementation for watermarking with localized messages.
  
  - This model is noted to have only **1M parameters**, potentially allowing it to be integrated into various AI generators quickly.

**Links mentioned**:

- [Tweet from harsh (@HarshSikka)](https://x.com/HarshSikka/status/1856739777208574151)): Excited to share our new paper "Benchmarking Vision, Language, & Action Models on Robotic Learning Tasks" We evaluate how well VLM & VLA models can control robots across 20 different real-wor...
- [GitHub - facebookresearch/watermark-anything: Official implementation of the paper "Watermark Anything with Localized Messages"](https://github.com/facebookresearch/watermark-anything): Official implementation of the paper "Watermark Anything with Localized Messages" - facebookresearch/watermark-anything

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1306007383268524074) (4 messages):

> - `TapeAgents Queries`
> - `AI Agents for Enterprise Workflows`
> - `Communication via Tape`

- **Tape as a Communication Medium with Agents**: A member questioned whether **Tape** can be used as a medium for communication or declarations of change between a human and an agent, asking for appropriate documentation to support this.
  
  - This led to a request for guidance on how an agent's tape entries encountering errors could be published to a queue.
- **Resources on TapeAgents**: In response to queries about **TapeAgents**, another member shared a [GitHub intro notebook](https://github.com/ServiceNow/TapeAgents/blob/main/examples/intro_clean.ipynb) and a relevant [paper](https://www.servicenow.com/research/TapeAgentsFramework.pdf).
  
  - *The member stated that they have read all provided resources,* expressing that they had already reviewed the suggested materials.

 

---

### **Alignment Lab AI ▷ #**[**general**](https://discord.com/channels/1087862276448595968/1095458248712265841/1306165252236247090) (2 messages):

> - `Latent Toys website`

- **New Website Alert: Latent Toys**: A member shared a link to a newly created website, [Latent Toys](https://latent.toys/), highlighting it as a noteworthy project.
  
  - They mentioned that a friend was behind the development of the site, further adding to its significance.
- **Community Highlights a New Project**: Members discussed the launch of a friend's project through the website [Latent Toys](https://latent.toys/), emphasizing its importance in the community.
  
  - The announcement generated interest and curiosity about the offerings of the new site.

 

**Link mentioned**: [latent.toys](https://latent.toys/): no description found

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1306388235836063866) (2 messages):

> - `Palmyra X 004 model`
> - `Writer handler integration`
> - `Pull Request Review Process`

- **PR submitted for Writer models and Palmyra X 004**: A member announced the submission of a [PR](https://github.com/ShishirPatil/gorilla/pull/755) to add support for **Writer models** and the **Palmyra X 004** to the leaderboard.
  
  - *Thank you!* for the review was expressed, and an image preview linked to the PR was shared.
- **Prompt response for PR review**: Another member responded promptly, stating they will take a look at the submitted PR.
  
  - *Thank you!* was conveyed in this acknowledgment, highlighting community engagement.

 

**Link mentioned**: [[BFCL] Add support for Writer models and Palmyra X 004 by samjulien · Pull Request #755 · ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/pull/755): This PR adds support for Writer models and our latest Palmyra X 004 to BFCL. Thank you!

 

---

### **AI21 Labs (Jamba) ▷ #**[**general-chat**](https://discord.com/channels/874538902696914944/874538902696914947/1306385756247298058) (2 messages):

> - `Legacy models deprecation`
> - `Open source solutions`

- **Users unhappy about legacy models being deprecated**: Members expressed their frustration regarding the **deprecation of legacy models**, stating that the new models are not providing the same output quality.
  
  - *This deprecation is hugely disruptive* for users who have relied on the older models for almost two years.
- **Transition to open source still in progress**: Users are scrambling to convert to an **open source solution**, while they have been willing to pay for the old models.
  
  - *How can we be sure AI21 won't deprecate the new models in the future too?* highlights their concerns about the stability of future offerings.

 

---

---

---

---

{% else %}

> The full channel by channel breakdowns have been truncated for email.
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
> 
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}
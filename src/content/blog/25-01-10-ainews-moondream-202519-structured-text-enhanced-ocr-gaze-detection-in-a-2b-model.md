---
id: f70a1040-15e6-4dd0-aa9a-562aab4c7079
title: >-
  Moondream 2025.1.9: Structured Text, Enhanced OCR, Gaze Detection in a 2B
  Model
date: '2025-01-11T07:18:42.365063Z'
original_slug: ainews-moondream-202519-structured-text-enhanced
description: >-
  Moondream released a new version that significantly improves VRAM efficiency
  while adding structured output and gaze detection, marking a new frontier in
  vision model performance. Additionally, advancements in reasoning models and
  model distillation techniques were highlighted, showcasing surprising
  performance gains and practical steps to build such models.
tags:
  - moondream
  - vision-models
  - vram-efficiency
  - gaze-detection
  - reasoning-models
  - model-distillation
  - multimodal-models
---


<!-- buttondown-editor-mode: plaintext -->**Efficiency tricks are all you need.**

> AI News for 1/9/2025-1/10/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **32** Discords (**219** channels, and **2928** messages) for you. Estimated reading time saved (at 200wpm): **312 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Moondream has been gaining a lot of attention for its [small, light, fast, yet SOTA vision](https://www.youtube.com/watch?v=T7sxvrJLJ14), and released a lovely new version yesterday that marks a new efficient frontier in VRAM usage (more practical than just param count):

![image.png](https://assets.buttondown.email/images/b1987557-b71d-49c5-8e4f-c141e587d791.png?w=960&fit=max)

It now also offers structured output and gaze detection, which allows [creative redditors to come up with scripts like these](https://www.reddit.com/r/LocalLLaMA/comments/1hxm0ep/anyone_want_the_script_to_run_moondream_2bs_new/):

![image.png](https://assets.buttondown.email/images/d306a5da-d414-4bdb-b6a7-c7100412f28f.png?w=960&fit=max)

In case you missed it, Vik also gave a talk about Moondream at the Best of 2024 in Vision Latent Space Live event:

https://www.youtube.com/watch?v=76EL7YVAwVo


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

**AI Models and Research**

- **Reasoning Models and Distillation Techniques**: [@_philschmid](https://twitter.com/_philschmid/status/1877778889566494843) and [@saranormous](https://twitter.com/saranormous/status/1877608687344431586) discussed advancements in reasoning models like [@OpenAI](https://twitter.com/OpenAI)’s o1, **detailing steps to build such models**. Additionally, [@jxmnop](https://twitter.com/jxmnop/status/1877761437931581798) highlighted the **effectiveness of model distillation**, emphasizing its surprising performance improvements without theoretical explanations.

- **Multimodal and Embedding Models**: [@llama_index](https://twitter.com/llama_index/status/1877778352087699962) introduced “vdr-2b-multi-v1”, a **2B multimodal, multilingual embedding model**, achieving **95.6% average NDCG@5** across languages. [@reach_vb](https://twitter.com/reach_vb/status/1877773277571014882) showcased **LLaVA-Mini**, which **reduces FLOPs by 77%** and enables **3-hour video processing** on a single GPU.

- **Innovations in GANs and Diffusion Models**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1877782765107908986) and [@multimodalart](https://twitter.com/multimodalart/status/1877724335474987040) shared research on **modern GAN baselines** and **Decentralized Diffusion Models**, highlighting their **stability and performance** compared to traditional approaches.

- **Self-Attention and Training Techniques**: [@_arohan_](https://twitter.com/_arohan_/status/1877795996815728987) discussed **stick-breaking attention** for **length generalization**, while [@addock_brett](https://twitter.com/adcock_brett/status/1877481322953715899) predicted **2025 as the year of Physical AI**, reflecting on **training optimizations** and **model architectures**.

**AI Tools and Development**

- **Development Frameworks and APIs**: [@awnihannun](https://twitter.com/awnihannun/status/1877490045915115992) announced updates to **MLX**, enhancing **portability** with support for multiple languages and platforms. [@vllm_project](https://twitter.com/vllm_project/status/1877794657117392936) introduced **nightly builds** and **native MacOS support** for **vLLM**, improving **developer experience** with **faster installations**.

- **AI Integration and Pipelines**: [@LangChainAI](https://twitter.com/LangChainAI/status/1877747452486320610) and [@virattt](https://twitter.com/virattt/status/1877497641522835714) demonstrated building **LLM-powered data pipelines** and **AI-powered data workflows** using tools like **LangChain** and **Qdrant**, enabling **intelligent semantic search** and **agentic document processing**.

- **Exporting and Interfacing Models**: [@awnihannun](https://twitter.com/awnihannun/status/1877564909027835931) provided guides on **exporting functions from Python to C++** in **MLX**, facilitating **cross-language model deployment**. [@ai_gradio](https://twitter.com/ai_gradio/status/1877478548874699153) showcased **qwen integration** for **anychat**, enhancing **developer deployment** with minimal code.

**Company Announcements and Updates**

- **Company Roles and Expansions**: [@russelljkaplan](https://twitter.com/russelljkaplan/status/1877538454969479181) announced their new role as a **"cognition guy"**, while [@ajayj_](https://twitter.com/ajayj_/status/1877795313446007016) welcomed new team members to **GenmoAI’s San Francisco office**.

- **Product Releases and Enhancements**: [@TheGregYang](https://twitter.com/TheGregYang/status/1877540170414829675) released the **Grok iOS app**, and [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1877735724683788363) introduced a **new denim collection** on **RunwayML**. [@everartai](https://twitter.com/skirano/status/1877790936966553807) launched **character finetuning** services, demonstrating superior **pipeline consistency** with **minimal input images**.

- **Hiring and Employment Trends**: [@cto_junior](https://twitter.com/cto_junior/status/1877685041696006345) discussed hiring trends at **Microsoft**, while [@bindureddy](https://twitter.com/bindureddy/status/1877474052589367388) predicted that **Salesforce** and other big tech companies **will stop hiring engineers** due to **AI-driven productivity gains**.

**Datasets and Benchmarks**

- **New Datasets Released**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1877653288444665915) and [@miivresearch](https://twitter.com/miivresearch/status/1877653288444665915) announced **Decentralized Diffusion Models** and released **code** and **project pages** for **HtmlRAG** and other **multimodal datasets**.

- **Benchmarking and Evaluation**: [@swyx](https://twitter.com/swyx/status/1877818998060175508) shared insights on **MMLU/GPQA knowledge**, emphasizing the **need for neural search engines** like **@ExaAILabs**. [@FinBarrTimbers](https://twitter.com/finbarrtimbers/status/1877791666330796180) discussed the lack of enduring **cognitive benchmarks** outside of **robotics**.

**AI Ethics, Policy, and Society**

- **AI's Societal Impact and Ethics**: [@fchollet](https://twitter.com/fchollet/status/1877535640717504810) and [@sama](https://twitter.com/sama/status/1877815461259235419) debated the **future of jobs** in an **AI-automated society** and the **ethical implications** of **AI governance**, including concerns over **policy restrictions** and **AGI definitions**.

- **Geopolitical Implications of AI**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1877730865003786384) and [@ClementDelangue](https://twitter.com/ClementDelangue/status/1877767382120255792) highlighted the **geopolitical power** held by **open-source AI** and the **strategic moves** by countries like **China** in the **AI landscape**.

- **AI Safety and Regulatory Concerns**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1877792522732212268) and [@Nearcyan](https://twitter.com/nearcyan/status/1877734059687965125) raised concerns about **AI deception behaviors**, **public safety**, and the **need for proper preparation** against potential **AI-driven disasters**.

**Personal Updates and Announcements**

- **Career Moves and Roles**: [@russelljkaplan](https://twitter.com/russelljkaplan/status/1877538454969479181) shared excitement about their new role, while [@megansirotanggalenuyen_](https://twitter.com/karinanguyen_/status/1877578425906393431) celebrated being listed on **Forbes 30 under 30**.

- **Workplace Experiences**: [@vikhyatk](https://twitter.com/vikhyatk/status/1877803302479421925) expressed concerns about their university’s administration, and [@sarahookr](https://twitter.com/sarahookr/status/1877464396722471354) provided a personal update on the **LA devastation**.

- **Learning and Development**: [@qtnx_](https://twitter.com/qtnx_/status/1877745878112387191) mentioned **learning RL in JAX**, and [@aidan_mclau](https://twitter.com/aidan_mclau/status/1877705861608452332) discussed **AI capital usage** and the challenges faced by **billionaires in AI development**.

**Memes/Humor**

- **Humorous Takes on AI and Technology**: [@nearcyan](https://twitter.com/nearcyan/status/1877820139992732125) and [@teortaxesTex](https://twitter.com/teortaxesTex/status/1877766872143147010) shared tweets with **satirical remarks** on **AI prompt engineering**, **tech company behaviors**, and **AI hype**, injecting light-hearted commentary into technical discussions.

- **Casual and Funny Remarks**: [@richardMCNgo](https://twitter.com/RichardMCNgo/status/1877806040563273999) and [@teortaxesTex](https://twitter.com/teortaxesTex/status/1877755556514996691) posted **jokes** and **puns** related to **AI advancements** and **tech culture**, providing moments of levity for the engineering audience.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Moondream 2b's Gaze Detection Creates Buzz**

- **[Anyone want the script to run Moondream 2b's new gaze detection on any video?](https://v.redd.it/n9beslavz0ce1)** ([Score: 1123, Comments: 207](https://reddit.com/r/LocalLLaMA/comments/1hxm0ep/anyone_want_the_script_to_run_moondream_2bs_new/)): The post discusses the release of a script for running **Moondream 2b's gaze detection** on any video. No additional details or context are provided in the post body.
  - **Interest and Enthusiasm**: Many users, including **That_Neighborhood345** and **ParsaKhaz**, express strong interest in the release of the gaze detection script, with some like **ParsaKhaz** offering to clean and release their script if enough interest is shown. This indicates a significant community interest in experimenting with and utilizing gaze detection technology.
  - **Surveillance Concerns**: Several users, such as **ArsNeph** and **SkullRunner**, voice concerns about the potential misuse of gaze detection technology for surveillance and privacy violations. They highlight examples like China's social credit system and corporate micromanagement, arguing that the technology could be abused to monitor individuals' focus and activities.
  - **Technical Feasibility and Use Cases**: **aitookmyj0b** notes that implementing gaze detection is feasible with basic **OpenCV processing**, suggesting that the technology is already within reach for those interested. However, **ArsNeph** argues that the technology lacks precision for legitimate eye-tracking applications, emphasizing its primary use as surveillance software rather than for beneficial purposes.


**Theme 2. Transformers.js Brings LLMs In-browser with WebGPU**

- **[WebGPU-accelerated reasoning LLMs running 100% locally in-browser w/ Transformers.js](https://v.redd.it/vmfpb2m2r5ce1)** ([Score: 379, Comments: 62](https://reddit.com/r/LocalLLaMA/comments/1hy34ir/webgpuaccelerated_reasoning_llms_running_100/)): **WebGPU-accelerated LLMs** are demonstrated running entirely locally in-browser using **Transformers.js**. This showcases the potential for **in-browser AI applications** without relying on server-side processing.
  - **Performance Variability**: Users report varied performance metrics based on hardware, with examples like **RTX 3090** achieving **55.37 tokens per second** and **MiniThinky-v2** achieving **~60 tps** on a **MacBook M3 Pro Max**. The lack of specified hardware in performance metrics is noted as a common issue in machine learning discussions.
  - **Technical Exploration and Challenges**: There is interest in exploring the technical capabilities of **WebGPU** and its applications in running AI models locally. Users discuss the potential of creating a browser extension that utilizes a reasoning LLM to manipulate the DOM directly, emphasizing privacy and local processing.
  - **Issues with Model Output**: Some users highlight issues with model output, such as generating nonsensical text or incorrect reasoning, like the example where the model incorrectly states *"60 does not equal 60"*. This highlights the challenge of achieving accurate and reliable outputs in local AI applications.


**Theme 3. Biden's AI Chip Export Limits Stir Global Reaction**

- **[Biden to Further Limit Nvidia AI Chip Exports in Final Push Restricting US Allies Such As Poland, Portugal, India or UAE Maker Of Falcon Models](https://www.bloomberg.com/news/articles/2025-01-08/biden-to-further-limit-nvidia-amd-ai-chip-exports-in-final-push)** ([Score: 167, Comments: 107](https://reddit.com/r/LocalLLaMA/comments/1hy8733/biden_to_further_limit_nvidia_ai_chip_exports_in/)): **Nvidia** AI chip exports face additional restrictions by the **Biden administration**, affecting US allies including **Poland, Portugal, India**, and the **UAE**. This move targets the export of AI technology, particularly impacting countries involved with the **Falcon models**.
  - Several commenters criticize the **Biden administration's** policy as **ineffective** and potentially harmful, arguing it could lead to increased cooperation between **China** and **Tier 2** countries, and that it might inadvertently target open-source AI rather than China. Concerns are also raised about the impact on global tech development and **US geopolitical standing**.
  - There is confusion and dissatisfaction over the **tier system** used to categorize countries for AI chip exports, with users questioning decisions like placing **Portugal** and **Switzerland** in Tier 2 while others like **Italy** are in Tier 1. The **Schengen Area** is mentioned as a potential loophole, allowing countries to circumvent restrictions.
  - Discussion highlights the potential for **NVIDIA alternatives** to gain traction due to these restrictions, and questions about **Nvidia's chip manufacturing** locations, particularly regarding **TSMC** in **Taiwan** and its implications for US-China relations. Concerns are expressed that these policies may not effectively prevent countries like China from obtaining restricted technologies.


**Theme 4. NVIDIA's Project Digits Promises AI Democratization**

- **[Project Digits: How NVIDIA's $3,000 AI Supercomputer Could Democratize Local AI Development | Caveman Press](https://www.caveman.press/article/project-digits-nvidia-3000-ai-supercomputer-democratize-development)** ([Score: 113, Comments: 75](https://reddit.com/r/LocalLLaMA/comments/1hxuprn/project_digits_how_nvidias_3000_ai_supercomputer/)): **NVIDIA**'s **Project Digits** aims to democratize local AI development by offering a $3,000 AI supercomputer. This initiative could significantly enhance accessibility for developers and researchers, potentially transforming local computational capabilities.
  - The community questions the **democratization** claims of **NVIDIA's Project Digits**, suggesting it primarily democratizes deployment rather than training. Some users argue that true democratization would require open-sourcing **CUDA** and note that **NVIDIA's** benchmarks use **fp4** precision, which is lower than typical standards like **fp32** or **fp16**.
  - There is skepticism about the **supercomputer** label, with comparisons to existing **GPU** and **RAM bandwidth** standards suggesting that the **Digits** offering may not match expectations. Users highlight that competitive products with higher RAM bandwidths and wider RAM to CPU buses already exist, such as **Apple's M4 Max** with **546 GB/s** and **AMD EPYC** with **460 GB/s**.
  - Discussions also focus on the role of **CUDA** in machine learning, with some advocating for more **vendor-agnostic solutions** like **Triton**. While **CUDA** is still prevalent for developing new ML techniques, there is a push towards frameworks that support multiple vendors, as seen with **OpenAI** and **Triton**, which is gaining traction for its ease of use and performance.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. DALL-E Abandonment: OpenAI's Multimodal Struggles**

- **[Did OpenAI abandon DALL·E completely? The results in DALL·E and Imagen3 for the same prompt](https://www.reddit.com/gallery/1hxqhjw)** ([Score: 343, Comments: 136](https://reddit.com/r/OpenAI/comments/1hxqhjw/did_openai_abandon_dalle_completely_the_results/)): **OpenAI** may have halted updates for **DALL·E**, as suggested by a comparison of image generation results between **DALL·E** and **Imagen3** using the same prompt. The discussion implies that **DALL·E**'s performance has not been improved or maintained, raising questions about OpenAI's focus on this project.
  - Several commenters speculate on the future of **OpenAI's DALL·E**, with some suggesting that OpenAI may release an updated or new model, potentially a multimodal one, as competition in image generation intensifies. **Vectoor** and **EarthquakeBass** mention that past versions of **DALL·E** were groundbreaking upon release but quickly fell behind due to infrequent updates.
  - There is criticism of **DALL·E 3**'s aesthetic and technical performance, with **COAGULOPATH** and **EarthquakeBass** noting its failure to produce convincing photorealistic images, potentially due to OpenAI's conservative safety stance. **Demigod123** suggests that the cartoonish style might be a deliberate choice to prevent misuse.
  - Alternatives like **Midjourney**, **Flux Schnell**, and **Mystic 2.5** are discussed, with users sharing links to images they generated, highlighting their capabilities compared to **DALL·E**. **Bloated_Plaid** and **MehmetTopal** provide visual comparisons, indicating that other tools might currently offer superior results.


**Theme 2. Microsoft Envisions AI Agent Swarms in Organizations**

- **[Microsoft CEO says each worker will soon be directing a "swarm of [AI] agents", with "hundreds of thousands" of agents inside each organization](https://v.redd.it/143088q6g1ce1)** ([Score: 235, Comments: 154](https://reddit.com/r/OpenAI/comments/1hxo7t8/microsoft_ceo_says_each_worker_will_soon_be/)): **Microsoft CEO** predicts that each worker will manage a "swarm" of **AI agents**, with "hundreds of thousands" of these agents being deployed within organizations. This statement suggests a significant increase in AI integration and automation in workplace environments.
  - **AI Agent Management Skepticism**: Many commenters express skepticism about managing a "swarm" of AI agents, questioning the practicality and potential chaos of handling numerous agents needing human intervention. Some see this as another instance of **Microsoft** overhyping technology without delivering tangible results.
  - **Impact on Employment and Industry**: Discussions highlight concerns about job displacement, with fears that AI will replace a significant portion of the workforce, especially in **white-collar jobs**. There's a debate about the future of work, with some suggesting a shift from "white collar" vs. "blue collar" to "automatable" vs. "non-automatable" tasks.
  - **Tech Industry's Strategic Tactics**: Commenters draw parallels between AI integration and previous tech strategies, like **Apple's ecosystem** entrenchment. There's a belief that tech companies will use similar tactics to lock in customers, making it costly and complex to transition away from their AI solutions.

---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-mini-2024-09-12

**Theme 1. AI Model Showdowns: PHI-4 Tops Microsoft and Beyond**

- [**Unsloth's PHI-4 Outscores Microsoft on Open LLM Leaderboard**](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=phi-4&rowSize=large): **PHI-4** from Unsloth surpasses Microsoft's benchmark by implementing crucial bug fixes and **Llamafication**, though quantized variants sometimes edge out their non-quantized counterparts.

- [**rStar-Math Boosts Qwen2.5 and Phi3-mini to New Heights**](https://x.com/altryne/status/1877220144725758414?s=46): Microsoft's **rStar-Math** propels **Qwen2.5** from **58.8%** to **90.0%** and **Phi3-mini** from **41.4%** to **86.4%** on the MATH benchmark, marking significant strides for small LLMs in math reasoning.

- [**Llama 3.3 Stumbles on Low-End Hardware with Slow Outputs**](https://www.youtube.com/watch?v=PWgvGjAhvIw): Enthusiasts report sluggish performance of **Llama 3.3 70B Instruct**, delivering tokens at **0.5/sec** on modest systems like a **Ryzen 7** and **RX 7900GRE**, highlighting the need for robust GPU memory or system RAM.

**Theme 2. AI Tools Face Off: Codeium, ComfyUI, and Cursor IDE**

- [**Codeium's Self-Hosted Edition Empowers Team Deployments**](https://codeium.com/pricing#are-credits-consumed-for-command-and-chat-in-codeium-extensions-vs-code-and-jetbrains-ides): **Codeium** introduces a self-hosted version in its enterprise package, attracting teams eager for customizable, in-house AI setups while navigating credit handling intricacies.

- [**ComfyUI Enhances AnimateDiff with IP Adapter Magic**](https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use): Community critiques **AnimateDiff's** output quality, turning to a [**ComfyUI workflow**](https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use) that integrates **IP Adapter** to supercharge video generation results.

- [**Cursor IDE Rules Tighten Up Claude's Code Crafting**](https://dotcursorrules.com/): Developers employ **.CursorRules** within **Cursor IDE** to precisely guide **Claude's** outputs, significantly reducing code misedits and ensuring accurate feature implementations.

**Theme 3. GPU Grievances and Kernel Calamities: Stable Diffusion on Linux**

- [**Linux Users Battle Kernel Panics Running Stable Diffusion on AMD GPUs**](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux): Attempts to run **Stable Diffusion** on **Linux** with **AMD GPUs** sometimes trigger kernel panics, but referrals to the [**AMD GPUs installation wiki**](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux) offer fixes addressing Python version issues.

- [**Stable SwarmUI vs A1111: User Interface Tug of War**](https://github.com/Stability-AI/StableSwarmUI/blob/master/docs/Features/IPAdapter-ReVision.md): Discord users debate the user-friendliness of **A1111**, **SwarmUI**, and **ComfyUI**, with SwarmUI’s advanced features drawing praise despite a perceived steeper learning curve.

- [**MicroDiT Replicates and Polishes with DCAE Integration**](https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt): Successful replication of **MicroDiT** provides downloadable weights and an [**inference script**](https://github.com/SwayStar123/microdiffusion/blob/main/test_model.ipynb), paving the way for architectural enhancements using **DCAE** for improved performance.

**Theme 4. AI Community Buzz: Hackathons, Hiring, and Funding Frenzies**

- [**oTTomator's AI Agent Hackathon Triggers $6K Prize Bonanza**](https://studio.ottomator.ai/hackathon/register): **OpenRouter** launches the **oTTomator AI Agent Hackathon**, offering a total of **$6,000** in prizes from sponsors **Voiceflow** and **n8n**, inciting fierce competition from January 8 to January 22.

- [**Anthropic Secures $2B Funding as AI Ventures Soar**](https://x.com/andrewcurran_/status/1876705929296581078?s=46): **Anthropic** raises an additional **$2 billion**, elevating its valuation to **$60 billion** and bolstering its position in enterprise AI solutions, as per [Andrew Curran’s report](https://x.com/andrewcurran_/status/1876705929296581078).

- [**Nectar Social Offers $10K Bounties to Snag AI Talent**](https://www.linkedin.com/jobs/view/4120980579/): AI startup **Nectar Social** in **Seattle** is hunting for **Product Managers** and **AI Engineers**, offering **up to $10,000** in referral bounties to attract skilled hires for their growing social commerce platform.

**Theme 5. Advanced AI Techniques: Fine-Tuning, Decoding, and Regularization Woes**

- [**Adapters Aren't All Fun and Games: LoRA Precision Matters**](https://medium.com/@bnjmn_marie/lora-load-and-merge-your-adapters-with-care-3204119f0426): Technical experts stress the importance of using **16-bit models** when implementing **LoRA adapters** to prevent output degradation, advocating for merging adapters with higher precision bases.

- [**Speculative Decoding Emerges as Resource-Saving Hero**](https://arxiv.org/abs/2501.04682): In a bid to reduce computational loads during next-token generation, the community hails **speculative decoding** as a promising technique akin to **DLSS** for language models.

- [**Weight Decay Wars: Stabilizing LLMs with Gentle Settings**](https://arxiv.org/abs/2501.04697): Researchers debate the impact of **extreme weight decay** (e.g., **0.1**) in large language models, proposing milder decay and auxiliary loss functions like *abs(norm(logits) - 1.0)* to prevent model meltdown and maintain numerical stability.

---

# PART 1: High level Discord summaries


## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **GPU Gains & Gripes**: Engineers compared the **RTX 4070** with the **3090** for AI video tasks, noting that a 3090 can render 480p in about 2 minutes, with more variations reported for LTXV on AMD from [this discussion](https://np.reddit.com/user/kejos92/comments/1hjkkmx/ltxv_inference_on_amd_gpus/).
   - Participants exchanged performance metrics and tweaks, pointing toward specialized setups for quicker **image-to-video** workflows.
- **AnimateDiff Antics**: Members critiqued **AnimateDiff** for subpar output and referenced a [ComfyUI workflow](https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use) merging IP Adapter to enhance quality.
   - They also discussed an [image-to-video comparison option](https://civitai.com/models/548997/image-to-video-comparison-workflow) that tests multiple methods, noting some steps still burn more runtime than desired.
- **Discord Dramas**: Users reported **inappropriate profiles** and argued over stricter moderation to keep conversations civil.
   - Concerns arose about striking a balance between policing toxic content and preserving a welcoming environment.
- **Interface Interjections**: Comparisons of **A1111**, **SwarmUI**, and **ComfyUI** revealed differing opinions on user-friendliness, with SwarmUI’s features documented in [this GitHub guide](https://github.com/Stability-AI/StableSwarmUI/blob/master/docs/Features/IPAdapter-ReVision.md).
   - While A1111 was praised for simplicity, some appreciated ComfyUI’s advanced pipeline for **animated** content creation.
- **Panic in the Kernel**: Linux-based **Stable Diffusion** occasionally triggered kernel panics, prompting references to the [AMD GPUs installation wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux).
   - Guides and fixes often tackled Python version issues, offering fallback solutions for smoother AI workflows on Linux.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth’s PHI-4 Prowess Outshines Microsoft**: In official [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=phi-4&rowSize=large) scores, **PHI-4** from Unsloth just surpassed the Microsoft baseline thanks to bug fixes and Llamafication.
   - Community members praised the improvement but noted **quantized variants** can sometimes outperform non-quantized configurations.
- **Adapter Attitude: Precision Pays Off**: Experts stressed using **16-bit** models when attaching adapters for better throughput, referencing a [LoRA cautionary post](https://medium.com/@bnjmn_marie/lora-load-and-merge-your-adapters-with-care-3204119f0426).
   - They mentioned that lower-precision usage can degrade outcomes, and merging with the higher-precision base is typically preferred.
- **Chat Templates Tweak LLM Behavior**: Contributors discussed how **chat templates** from the `tokenizer_config.json` shape input-output formatting, affecting LLM performance significantly.
   - They emphasized that consistent templates from training to production ensure stable results, with some claiming it can "make or break" **deployment success**.
- **Speculative Decoding: Decisive Trick for Resource Reduction**: A conversation on **DLSS-like** optimization for language models led to the mention of *speculative decoding*, hailed as a resource-friendly technique.
   - Researchers found it promising for stepping around hefty computational loads in next-token generation.
- **Mathstral 7B Waits for Wider Support**: The `mistralai/Mathstral-7B-v0.1` model was clarified to be **unsupported** for direct fine-tuning, as it isn't a standard base or PEFT model.
   - Participants said support is coming soon, sparking cautious optimism for future merges and expansions.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Self-Hosted Codeium Fuels Deployment Control**: Members noted that a self-hosted edition of **Codeium** is now part of the enterprise package, drawing attention from teams eager to manage their own setups.
   - They also raised questions about credit handling and **Windsurf** features, pointing to the [pricing page](https://codeium.com/pricing#are-credits-consumed-for-command-and-chat-in-codeium-extensions-vs-code-and-jetbrains-ides) for official guidelines.
- **Windsurf Installation Marathon on Multiple Distros**: Users reported successful **Windsurf** installs on Mint, Ubuntu 24.04, Arch, and Hyprland, sometimes after removing configuration files to address odd errors.
   - They also discussed the desire for shared **Cascade** chat across PCs, with cloud sync suggestions emerging but no official feature in place yet.
- **Flow Credits Billing Headache Stirs Frustration**: Several people complained of paying for Flow Credits multiple times but never seeing them added, prompting calls for clearer usage policies.
   - They also questioned whether **internal errors** count against credits, urging developers to fix these deductions swiftly.
- **Cascade’s No-Code Wins and Chat Management Dreams**: One user highlighted building a company website with minimal actual coding, celebrating **unlimited** queries in the free Cascade tier.
   - Others still want better **Cascade** chat handling across multiple devices, citing a need for official sync solutions.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Rules Tame Chaotic Code**: Developers shared how to refine **Claude**'s output with structured prompts using [.CursorRules](https://dotcursorrules.com/), focusing on explicit goals to avoid unintended file changes.
   - They reported that precisely chosen keywords significantly cut down code misedits, underlining the importance of well-defined prompt instructions.
- **Cursor Directory Gains Traction**: A surge of interest in the [Cursor Directory](https://cursor.directory/) spotlighted its ability to gather community-sourced rules for various frameworks.
   - Users appreciated a centralized place for rule sharing, noting it saved them time and headaches when tackling specialized setups.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Agile Prompting with Colors**: Community stressed specifying **colors** in prompts using color names and hex codes, ensuring clarity on usage.
   - One member replaced vague requests (*Just do blue and white*) with more precise guidelines to control styling across apps.
- **Payment System Meltdown**: A user found their **payment system** inoperable, posting a link to the project and asking for assistance.
   - They mentioned active development to restore full functionality, urging testers for feedback.
- **Open Public Repos with Bolt.new**: Developers announced a **public repos** feature, letting users prefix any GitHub URL with [http://bolt.new](http://bolt.new) for immediate access.
   - They referenced an [X post](https://x.com/stackblitz/status/1843668731681267801) showcasing how to open repositories with minimal setup.
- **Bolt Token Overruns**: Multiple people reported **rapid token consumption** when editing or debugging, encountering repeated attempts to fix errors.
   - They expressed frustration over sustained resource usage, hoping for a more efficient approach.
- **Supabase Migrations and Netlify Hitches**: Developers mentioned reversing **Supabase migrations** caused headaches if issues occurred mid-update, affecting application stability.
   - Additionally, one user cited slow **Netlify** load times, suspecting free tier constraints or inefficiencies in the **Bolt** code.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.0 Goes Mobile & Chatty**: While running errands, a user tested **Gemini 2.0 Flash Experimental** with **voice mode** on iOS, brainstorming an app idea in real-time and generating concise tasks upon returning.
   - Community members appreciated **Gemini 2.0's** ability to *autonomously propose project criteria*, calling it a helpful step toward frictionless development.
- **Tier 5 Key Trials & Unify.ai Tricks**: Discussion centered on alternatives for expensive **Tier 5 OpenAI** access, with references to [Unify.ai](https://unify.ai/) and the [GitHub repo](https://github.com/unifyai/unify) as flexible multi-model solutions.
   - Members weighed **subscription costs** and shared experiences about using **OpenRouter** and **Unify** to simplify configuration.
- **Aider & Claude Face Off in Coding**: Multiple users compared **Aider**'s uneven file editing and occasional mishaps to **Claude**, noting comedic incidents of entire file deletions, with references to [file editing problems](https://aider.chat/docs/troubleshooting/edit-errors.html).
   - Some deemed **DeepSeek** chat too distractible and *lazy*, while others recognized **Aider** as workable if carefully managed to avoid large-scale code removal.
- **Visions of Stronger AI Agents**: One user predicted **AI** will eventually create improved iterations of itself and minimize human intervention, tempered by concerns over **computation costs** and operational overhead.
   - Participants highlighted the **current limitations** of hardware and resource availability, offering both optimism and caution about near-term expansions in autonomous AI capabilities.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **DeepResearch Gains Ground in NotebookLM**: Members suggested integrating **DeepResearch** references into [NotebookLM](https://notebooklm.google.), aiming to merge outputs from existing reports despite no official plugin yet.
   - Some users mentioned 'bulk upload' workarounds to feed large datasets into **NotebookLM**, fueling anticipation for more robust synergy.
- **AI Audio Generation Sparks Excitement**: Participants explored building podcasts from curated NotebookLM sources, pairing them with [Illuminate](https://illuminate.google.com/create) for audio flexibility and better source targeting.
   - They praised source-limited prompts for controlling style, while others mentioned **Jellypod** as a potential alternative with broader customization options.
- **Cross-lingual Podcasting Showcases NotebookLM Flexibility**: Some users experimented with generating **Mandarin** podcast scripts from **English** content inside [NotebookLM](https://notebooklm.google.), applying casual rephrasing tactics for natural flow.
   - They also tested **Japanese** chats, noting that accurate transliteration might require additional checks but reflecting user comfort with switching languages.
- **Quotation Mode and System Prompt Confusions**: Developers introduced a 'quotation-only' command in **NotebookLM**, ensuring direct excerpts from sources and stricter verification for important citations.
   - However, **Gemini** occasionally returned incomplete quotes, prompting discussions on improving system prompts in **NotebookLM Plus** for consistent results.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen's Quirky Chat Craze**: Alibaba introduced [Qwen Chat](https://chat.qwenlm.ai), a new Web UI for **Qwen** models offering document uploads and visual understanding, teased in [their tweet](https://fxtwitter.com/Alibaba_Qwen/status/1877426465349972113).
   - The community expects upcoming features like **web search** and **image generation**, seeing Qwen Chat as a key competitor in the evolving LLM ecosystem.
- **AMD's 7900 Performance Puzzle**: Users compared the **AMD RX 7900XT 20GB** to **NVIDIA's 3090** using a [reddit post](https://reddit.com), suggesting the 7900XT might face memory bandwidth limitations for LLM tasks.
   - Others argue the 7900XT still performs decently for local inference, though they see more stable performance with the **3090** in certain benchmarks.
- **Llama 3.3's Memory Mayhem**: Enthusiasts reported **Llama 3.3 70B Instruct** delivering sluggish outputs at **0.5 token/sec** on lower-end hardware like a **Ryzen 7** and **RX 7900GRE**.
   - They emphasized the need for significant GPU memory or system RAM to avoid these slowdowns and sustain token throughput at scale.
- **NVIDIA DIGITS Drums Up Curiosity**: Community chatter turned to **DIGITS**, rumored to be a robust solution within the **NVIDIA** workflow for training and testing models.
   - Users remain cautious about performance overhead but anticipate a powerful addition to the local LLM toolkit.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O1’s ‘Thinking’ Quirk & A/B Test Hints**: A participant flagged **Model O1**’s distinctive 'thinking' output, suggesting that a different model format might be involved.
   - They raised the possibility of running **Model 4O** in parallel, reflecting enthusiasm for comparing multiple performance approaches.
- **Meta-Prompting Sparks Ideas**: Members highlighted **Meta-Prompting** strategies, citing that tweaking the system message can generate more advanced responses.
   - They stressed that clarifying objectives at the start leads to sharper **model outputs** when crafting prompts.
- **Investor Round for Hassabis**: The group offered good wishes for **Hassabis** during his investor round, acknowledging the importance of fresh capital in AI pursuits.
   - They praised his track record and noted that supportive funding could propel further **R&D** efforts in the field.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **ICLR: The Rendezvous Rush**: Attendees are gearing up for **ICLR**, trading excitement about traveling and planning potential meetups in real time.
   - They anticipate lively face-to-face chats, with **Philpax** arriving soon in a **light brown coat** and **black jeans**, ready to discuss new model breakthroughs.
- **rStar Rising: Qwen2.5 & Phi3-mini Soar**: **Microsoft’s rStar-Math** pushes **Qwen2.5** from **58.8%** to **90.0%** and **Phi3-mini** from **41.4%** to **86.4%** on the MATH benchmark.
   - It now averages **53.3%** in the USA Math Olympiad, prompting interest in *Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking* for deeper insights.
- **NuminaMath and the Quality Conundrum**: Skepticism grew over **NuminaMath** due to ~7.7% of entries containing multiple conflicting solutions, pointing to broader data issues.
   - Members also cited [“Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought”](https://arxiv.org/abs/2501.04682), noting the lead author’s psychology background at Stanford captured attention.
- **Open Source AI: The Cost Clash**: Policy makers voiced concern about **open source AI** costing merely **$5M**, sparking confusion over actual budgets.
   - One tweet’s cost breakdown excluded capital and data expenses, provoking criticism for misrepresenting GPU-hour tallies.
- **Anthropic’s Early Character Crafting**: At an **Anthropic salon**, **Josh Batson** indicated that **Amanda Askell** shapes the base model into an agent earlier than some expected.
   - Debate arose on whether character alignment is a post-training add-on or a built-in process, with references to [Anthropic Research Salon](https://youtu.be/IPmt8b-qLgk) fueling further conversation.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SmolLM Shard Storm**: The **SmolLM Corpus** soared to **320GB** split into **23698** shards, promising a more efficient `.jsonl.zst` format that won't be finalized until late this week.
   - Members praised the significantly smaller footprint versus the **1TB** uncompressed set, referencing HPC convenience and *"less overhead for iterative training pipeline"*.
- **Modal's Mighty Moves**: Hobbyists explored **Modal**, **Colab**, and **Kaggle** for budget-friendly training and analysis, spotlighting **Modal**'s monthly credits as a solid way to handle larger tasks.
   - They noted that **Modal** can run jobs beyond personal GPU capacity and appreciated the steady support for inference at scale.
- **SciAgents Sways AI Circles**: The [**SciAgents** paper](https://arxiv.org/abs/2409.05556) employs **ontological knowledge graphs** and **multi-agent** methods to boost research operations, weaving structured data with agent collaboration.
   - Some felt the concept wasn't a giant leap, yet others liked the orchestration approach, calling it a promising framework for high-level learning workflows.
- **Grokking Gains Momentum**: Members dissected [**Grokking at the Edge of Numerical Stability**](https://arxiv.org/abs/2501.04697), highlighting **delayed generalization** and **softmax collapse** in deep nets.
   - They emphasized that insufficient **regularization** can push models into meltdown, urging careful intervention and *"dampening runaway logits"* early on.
- **Weight Decay & Llama2 HPC Woes**: Several researchers debated **extreme weight decay** (like **0.1**) in LLMs, proposing milder settings for attention layers and auxiliary losses (e.g., *abs(norm(logits) - 1.0)*).
   - Meanwhile, attempts to pretrain a **7B** Llama2 with `model_parallel=2` triggered **OOM** stalls at batch size 1, prompting memory profiling and fresh tests for the **6.7B** configs.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **WGMMA & Triton Trials**: Engineers discussed **WGMMA** requiring splits across 4 warps with a minimum tile of 64, referencing [NVlabs' tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for **fused MLP** insights. They also recommended **Proton** for profiling, citing [a helpful video](https://youtu.be/Av1za_0o2Qs?si=k6G2zWMzDbKTt5rb).
   - Community members praised the easier debugging of **Triton** kernels with Proton, while questioning on-chip MLP usage for typical HPC tasks.
- **MI210 Occupancy Puzzle**: Members examined **GPU occupancy** for the **MI210** and **RX 7900XTX**, referencing [a resource on block-level optimization](https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/). They noted potential 16-warps occupancy but saw constraints like block-level resource usage in real-world code.
   - They concluded that hitting higher occupancy often demands multiple kernels, with **CDNA** architecture details revealing practical block limits and early exit behaviors. Further testing validated the distinctive block scheduling approach on **MI210**.
- **Nectar Social's $10k Bounty**: **Nectar Social** is hiring in **Seattle** for **Staff Product Manager**, **LLM/AI Engineer**, and **Infra Engineer**, offering referral bounties of up to **$10,000**. They emphasized prior startup experience and expressed willingness to share details privately.
   - A European consultancy with HPC clients like **AMD** also seeks developers skilled in **CUDA**, **HIP**, and **OpenCL**, referencing a job listing at [LinkedIn](https://www.linkedin.com/jobs/view/4120980579/). They also collaborate on libraries like **rocPRIM** and **hipCUB**, aiming to fill specialized GPU developer roles.
- **ARC Prize Non-Profit Shift**: The **ARC Prize** is transitioning into a non-profit foundation, as seen in a [tweet from François Chollet](https://x.com/fchollet/status/1877069518171943000), with a new president to guide AGI research. They also launched a **rejection sampling** baseline experiment to establish a foundational metric.
   - Community members explored **text-domain** solutions to mitigate GPU constraints and analyzed the **Meta CoT paper** ([link](https://arxiv.org/abs/2501.04682)) for potential refinements. The authors highlighted shortfalls in classic CoT approaches, sparking broader discourse on contextual reasoning.
- **MicroDiT Gains with DCAE**: **MicroDiT** replication concluded successfully, providing a [weight file](https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt) and an [inference script](https://github.com/SwayStar123/microdiffusion/blob/main/test_model.ipynb). They credited computational support and aim to improve architecture with **DCAE** for stronger performance.
   - Plans include employing **MMDIT** for better prompt adherence and seeking **compute grants**. The limited home GPU capacities hamper advanced AI experiments, spurring the search for additional resources.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Microsoft's rStar-Math Maneuvers Qwen's Mastery**: Microsoft introduced [rStar-Math](https://x.com/altryne/status/1877220144725758414), pushing **Qwen 2.5-Math-7B** from **58.8%** to **90.0%** on the MATH benchmark and scoring **53.3%** on AIME, placing it among the top 20% of high school students.
   - Members debated the significance of **math** prowess for **reasoning** skills, with some cautioning that numeric breakthroughs don't always guarantee broader **LLM** reliability.
- **DistTrO's Doors Swing Wide**: A member confirmed **DistTrO** is open sourced, prompting immediate integrations within community trainers.
   - Contributors praised **DisTrO** for distributed training simplicity, with some highlighting a smoother setup than earlier solutions.
- **Carson Poole's Paper Parade**: Carson Poole introduced [ReLoRA](https://arxiv.org/abs/2307.05695) and [Sparse Upcycling](https://arxiv.org/abs/2212.05055), referencing discussions from **November 2022** and **March 2023**.
   - He urged members to visit [his personal site](https://poole.ai) and offered email collaboration on **Forefront.ai** or **Simple AI Software** for deeper exploration.
- **DeepSeek V3's Twin Tests**: Members compared the official **DeepSeek V3** API's repetitive outputs to third-party providers like **Hyperbolic**, noting stark differences in answer quality.
   - Some attributed these inconsistencies to **aggressive caching**, prompting interest in more consistent inference approaches.
- **Qwen2.5 Memory Maze on 24 GB VRAM**: A user encountered out-of-memory errors on **Qwen2.5-32B-Instruct-AWQ** with an **RTX 4090**, despite enabling **flash attention**.
   - The discussion shifted to potential **memory usage** optimizations for ~6K token contexts, as well as inquiries into open source **function calling** accuracy benchmarks.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Salesforce's Surprising Freeze**: Marc Benioff confirmed [Salesforce](https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-benioff/) will halt hiring software engineers in **2025**, citing a **30% productivity boost** from **Agentforce**.
   - Community members see this as a major shift in resource allocation, with speculation that *"AI is truly taking over basic software tasks"*.
- **OpenAI Tweaks Custom Instructions**: OpenAI reportedly updated custom instructions for its **advanced voice** toolset, with [a tweet from topmass](https://x.com/topmass/status/1877444315871326422) showcasing partial breakage and hints of new features.
   - Observers suggest these improvements might usher new voice capabilities, with one user describing them as *"powerful enhancements for a more fluid AI experience"*.
- **Anthropic's $2B Infusion**: Anthropic is raising **$2 billion** at a **$60 billion** valuation, posting **$875 million** ARR according to [Andrew Curran's report](https://x.com/andrewcurran_/status/1876705929296581078).
   - Participants commented on *"venture capital's big appetite for AI solutions"*, especially as Anthropic's traction continues to expand in enterprise contracts.
- **Google Piles AI into DeepMind**: Google announced a plan to merge multiple AI products under **DeepMind**, showcased by [Omar Sanseviero's tweet](https://x.com/osanseviero/status/1877452798683430988) about joining forces in **2025**.
   - Commenters foresee possible overlap in corporate structure, calling it *"a puzzling reorg, but hopefully it streamlines LLM offerings"*.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hackathon Hype With a Cash Kick**: The **oTTomator AI Agent Hackathon** offers $6,000 in sponsor prizes, awarding $1,500 for first place, $150 for runners-up, plus **$10 in OpenRouter API credits**, with sign-ups at [register here](https://studio.ottomator.ai/hackathon/register).
   - It runs from January 8 to January 22, with community voting from January 26 to February 1, and the sponsor pool includes **Voiceflow** and **n8n** awarding extra $700 and $300.
- **OpenRouter UI Stumbles Past 1k Lines**: Users reported **OpenRouter UI** slows down drastically beyond 1k lines of chat history, making scrolling and editing painful.
   - They proposed improvements like sorting by cost and **Next.js pagination** to address these performance pitfalls.
- **Gemini Flash Sparks Confusion**: The **Gemini Flash** engine works in chatrooms but seems non-functional via API, baffling multiple users.
   - Another user praised **Gemini** overall, yet pointed out performance issues that warrant immediate improvements.
- **O1 Embraces Unusual Response Format**: Developers noticed **O1's response** uses '====' in place of markdown backticks, raising concerns about formatting quirks.
   - Discussions ranged from whether this measure is meant to cut token usage or refine output, prompting debate on best practices.
- **API Access & Hanami Trials**: Developers asked about offering their own **LLM API** through OpenRouter and shared issues with request handling.
   - Another user tested **Hanami** but encountered odd characters, emphasizing the importance of robust tool compatibility.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **CSV Export Gains Steam**: Perplexity introduced a feature to download table responses as **CSV files**, showcased in an [illustrative image](https://cdn.discordapp.com/attachments/1047204950763122820/1326655467304255508/download_csv.jpg).
   - Users welcomed this enhancement for **streamlining data workflows**, emphasizing how it can save time when handling large datasets.
- **Youzu.ai Interiors Leap Forward**: **Youzu.ai** provides AI-driven room designs with direct shopping options, as detailed in this [Medium guide](https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688).
   - Community members tested it and appreciated its potential to **reduce hassle**, asking for feedback on real-world usage.
- **Toyota's Rocket Rendezvous**: A new [rocket venture](https://www.perplexity.ai/page/toyota-is-exploring-rockets-NrLusU2uRdaUqsCirISg7Q) from **Toyota** suggests their push beyond standard auto engineering.
   - Enthusiasts noted the synergy between Toyota's established expertise and **aerospace demands**, predicting more official details may follow.
- **NVIDIA's $3K Home Supercomputer**: **NVIDIA** announced a future home-ready supercomputer at a price of **$3000**, as noted in a [CES 2025 reference](https://www.perplexity.ai/page/ces-2025-nvidia-s-ai-supercomp-Eldo96kHTICxurNQVyCGbw).
   - Tech fans debated whether the advanced performance justifies **the cost**, seeing this as an opening for machine learning experimentation at home.
- **Ecosia Eyes Perplexity Partnership**: A **product manager at Ecosia** struggled to reach Perplexity for a potential partnership and asked for guidance in making contact.
   - Helpers in the community offered suggestions for **direct communication**, hoping for a fruitful alliance if discussions move forward.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere's 'North' Rises to Rival Copilot**: Cohere introduced [**North** in early access](https://x.com/cohere/status/1877335657908949189), a secure AI workspace merging **LLMs**, **search**, and **agents** into one interface for productivity.
   - They contend it can outshine **Microsoft Copilot** and **Google Vertex AI**, as noted in the [official blog post](https://cohere.com/blog/north-eap).
- **Command R+ Spurs Generative Gains**: A user referenced [**Command R+**](https://docs.cohere.com/docs/models) when exploring workflows for large generative models in Cohere’s ecosystem.
   - Community discussions stressed a clear integration strategy and recognized the need for well-structured prompts to optimize model behavior.
- **v2 to v3 Embeddings: The Upgrade Query**: Questions arose on transitioning from **embed-v2** to **v3** without re-embedding massive datasets, prompting concerns about resource usage.
   - Members sought an efficient approach to maintain performance while minimizing overhead and potential downtime.
- **LLM Loops & Rolling Chat: Taming Token Overflows**: Reports indicated **Cohere’s LLM** could get stuck in repetitive loops, driving runaway token usage in the **Python ClientV2** setup.
   - Suggestions involved setting **max_tokens** limits and employing a **rolling chat history** technique to handle extended responses within the 4k token boundary.
- **Alignment Evals Hackathon Sparks Action**: An **Alignment Evals Hackathon** was announced for the 25th, featuring community-driven eval and interpretation tutorials.
   - Participants were encouraged to share insights and outcomes, fueling collaboration on alignment evaluation methods.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Mock GPU Mayhem & Bounty Talk**: A retest was requested for [Pull Request #8505](https://github.com/tinygrad/tinygrad/pull/8505) involving MOCKGPU on macOS, with **George Hotz** confirming a bounty is ready for this fix.
   - He offered payment via **PayPal** or **USDC** on Ethereum, underscoring a push to handle outstanding tasks in **tinygrad**.
- **LLVM JIT & Autogen Pair-Up**: Members proposed merging their **LLVM JIT** and **LLVM Autogen** efforts, referencing changes in multiple version files.
   - They also debated forward vs. backward compatibility, with some emphasizing older LLVM support to avoid silent breakage.
- **Function Signature Stability Friction**: Concerns surfaced about potential silent changes to **LLVM** function signatures causing undefined behavior.
   - **George Hotz** downplayed that risk, noting preference for supporting older LLVM releases to maintain consistency.
- **TinyGrad Blog & Device Setup**: A blog post titled [TinyGrad Codebase Explained-ish](https://adelaloui.me/tinygrad-codebase-explained-ish/) walked through **tinygrad**'s file layout, cautioning about lightly tested code outside **tinygrad/**.
   - A user asked about initializing weights on specific hardware, and got advice to set `Device.DEFAULT` to **METAL**, **CUDA**, or **CLANG** before creating tensors.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Llama vs GPT4All Race Ramps Up**: They highlighted that **Llama.cpp** Vulkan differs from **GPT4All** internals, yielding large speed gaps on **Nvidia GPUs** due to CUDA usage.
   - Participants concluded that the difference can be overlooked if performance meets everyday targets, referencing [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all/issues/3365) for more context.
- **Chat Template Tangle**: A user struggled with the **Chat Template** for TheBloke’s model on **GPT4All**, receiving generic replies despite correct installation.
   - Others advised checking model-specific instructions on GitHub, stressing that **chat prompts** differ widely across models.
- **Roleplay Recs Ride On Llama-3**: For anime-themed roleplay, members recommended **Nous Hermes 2** or [llama3-8B-DarkIdol-2.2-Uncensored-1048K](https://huggingface.co/aifeifei798/llama3-8B-DarkIdol-2.2-Uncensored-1048K) as workable older options.
   - They noted that Nomic's plug-and-play approach simplifies usage, especially for quick scripted dialogues.
- **ModernBERT Deployment Dilemma**: A query arose about **ModernBERT** from Nomic AI and whether it's supported on **text-embedding-inference** or **vLLM**.
   - No conclusive answer emerged, leaving the group uncertain about official deployment channels.
- **Image Model Hopes Spark GPT4All Chat**: Some considered the idea of adding **image models** into GPT4All for extended modality coverage.
   - The conversation ended with no definitive plan, yet it underscored user interest in bridging text and vision.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Grand Gathering at GitHub HQ**: On Jan 15, they're hosting a series of expert talks at [GitHub HQ](https://twitter.com/llama_index/status/1877103276635848846) discussing **AI agent improvements**, **fast inference systems**, and building workflows with **LlamaIndex**.
   - The event showcases advanced agentic workflows, highlighting real-world examples from multiple industry experts.
- **Agentic Document Workflows: A Bold Leap**: A new [blog post](https://twitter.com/llama_index/status/1877420085691953385) introduces **Agentic Document Workflows (ADW)**, aiming to integrate document processing directly into business processes.
   - It underscores that **documents come in multiple formats**, emphasizing a streamlined approach for future-driven applications.
- **Ollama Overdrive for Speed**: A recent update to **Ollama** brought evaluation times under **3 seconds**, spurring excitement among users.
   - One user called the gains *incredible*, reflecting strong enthusiasm about faster model inferences.
- **VectorStoreIndex: Manual Moves for Metadata**: Some members discussed filtering nodes by metadata keys in a **Postgres** JSON field using **VectorStoreIndex**, questioning whether they could avoid manual indexing.
   - They concluded **manual indexing** might still be needed, since LlamaIndex doesn't yet handle all related automation.
- **Taming TEI & QueryFusionRetriever Quirks**: Interest rose in using a **local TEI server** for reranking, referencing [API docs](https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/tei_rerank/) and [source code](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/postprocessor/llama-index-postprocessor-tei-rerank/llama_index/postprocessor/tei_rerank/base.py).
   - Amid that, users hit an **input validation error** in **QueryFusionRetriever** at 518 tokens, sharing code snippets to find a workaround.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Rust Syntax Eases Multiline**: One user praised **Rust** multiline syntax when building an actor for **multipaxos**, highlighting fewer type checks.
   - They said *function parameters can become chatty*, causing confusion for users sorting out required types.
- **Overload Resolution Gets Risky**: A user warned that rearranging overloads in large codebases may cause new snags, suggesting a **'happens after'** annotation approach.
   - They added *TraitVariant checks can mix with implementation traits*, potentially leading to messy overload resolution.
- **Quantum Libraries Progress in Mojo**: A member mentioned the need for a **Qiskit-like** library, referencing an interest in quantum expansions and linking to [MLIR dev videos](https://some.link).
   - They suggested that **MAX** might soon handle quantum tasks as it evolves.
- **MAX Backs Quantum Programming**: Discussion spotlighted **MAX** as Mojo's partner for fine-tuning quantum routines, allowing real-time hardware adjustments.
   - People said MAX can unify quantum and classical logic when it matures.
- **Quojo Brings a Quantum Option**: The **Quojo** library, shared via [GitHub](https://github.com/Deftioon/Quojo), was mentioned as a quantum computing tool in Mojo.
   - Folks expressed excitement for *emerging developers* pushing quantum coding forward.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Hackathon Timeline Takes Shape**: The **Hackathon website** ([link](https://rdi.berkeley.edu/llm-agents-hackathon/)) shared an updated results schedule, postponing final outcomes until **later in January**.
   - Organizers stated that a few judges still need to finalize their reviews, promising a thorough evaluation before announcing winners.
- **Judges Jump for Joy**: Judges gave glowing feedback about the **Hackathon submissions**, calling them *impressive entries* overall.
   - They emphasized the high level of creativity and technical depth, reinforcing anticipation for the final verdicts.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 1.0 Grapples with Python Execution**: Users discovered that **OpenInterpreter 1.0** does not directly run Python code, causing confusion about the `--tools interpreter` command.
   - A member expressed frustration over code execution limitations, sparking requests for **clearer instructions** on how to handle code blocks.
- **GPT-4o-mini Gains Some Command Control**: A discussion noted that **GPT-4o-mini** had improvements in command handling, particularly when printing file contents in smaller chunks.
   - The conversation focused on refining model performance through better file output strategies and fine-tuning command executions.
- **Call for Model Specifications**: A member asked for more **technical details** on parameter counts and underlying frameworks for better understanding performance metrics.
   - This inquiry underlined a need for **full documentation**, as participants sought clarity on the building blocks of the models.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **TruLie or Not? Data Mystery**: A user asked about the **TruLie dataset**, but no specifics or references were provided, leaving the conversation short on facts.
   - Members shared curiosity over possible research applications, yet no direct resources emerged.
- **Chirpy3D Takes Flight**: Enthusiasts discussed **image-to-3D** progress, highlighting **Chirpy3D** for continuous bird generation and **Gaussian splats** approaches, citing [Chirpy3D](https://kamwoh.github.io/chirpy3d/) as an example.
   - They mentioned collaboration from multiple institutions and pointed to the [3D Arena on Hugging Face](https://huggingface.co/spaces/dylanebert/3d-arena) as a resource for **NeRF** libraries.
- **World Models Evolve Visuals**: Contributors shared **World Models** that use physics-aware networks to produce more realistic video content.
   - Though outside pure image-to-3D pipelines, this direction aligns with broader efforts toward sophisticated visual systems.
- **Open Tool Registry Sought**: A researcher requested an open tool registry for **building agents**, hoping to collect suggestions from the group.
   - No direct leads surfaced, prompting further attempts to locate complete resources.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Chain-of-Thought Gains for Chatbots**: A user asked about boosting **Chain of Thought** for chatbots beyond a simple persona signature, seeking deeper conversation styles and reasoning steps.
   - The question went unanswered, underscoring the difficulty of refining **chatbot** logic and user interaction.
- **Evaluation Endeavors with DSPy**: A post on building your own evaluation, titled *An intro to building your own eval, why it matters, and how DSPy can help*, was shared [here](https://www.dbreunig.com/2025/01/08/evaluating-llms-as-knowledge-banks.html) to highlight the role of **DSPy** in customizing testing frameworks.
   - Readers showed excitement for crafting new evaluation methods and recognized DSPy's potential to improve **knowledge bank** solutions.
- **Anthropology & Tech: Drew's Path**: Drew Breunig gave an overview of his background in **cultural anthropology**, software, and media, mentioning work at **PlaceIQ** and **Precisely** with data integrity efforts.
   - He also collaborates with the **Overture Maps Foundation**, broadening the scope of data usage across varied industries.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Python + Jamba Pump Up Podcast Recall**: A user employed **Jamba's Conversational RAG** with a basic Python app to retrieve highlights from past podcasts using transcripts, calling it a work in progress but fun to experiment with.
   - They mentioned they are exploring new ways to integrate AI-driven recall without major hurdles, finding the system handy for archiving show notes.
- **AI Code Generation Rocks... But Goofs Happen**: A user raved about **AI's ability to generate code**, praising its handling of HTML and JavaScript but noting occasional silly mistakes.
   - They tested **PHP** tasks to gauge AI's limits, concluding that code generation remains head-scratching yet helpful.
- **PHP Holds Steady in Jamba Connection**: Another user declared loyalty to **PHP** for web and IRC bot coding, describing the hook-up to Jamba as a real adventure.
   - They liked how it parallels **deepSeek** and **OpenAI** APIs, simplifying programming tasks and encouraging swift tinkering.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **ModernBERT Gains Finetune Curiosity**: An inquiry was raised about **finetuning ModernBERT**, hinting at specialized task improvements but no direct experiences were shared.
   - The conversation ended without follow-up, leaving watchers hoping for **technical examples** or demos to light the way ahead.
- **Nectar Social Dangles $10K Referrals**: **Nectar Social** seeks AI-focused hires with referral bounties up to **$10,000**, including Sr/Staff Product Manager and LLM/AI Engineer roles.
   - They emphasized growth in **social commerce** with notable clients and encouraged interested applicants to DM for details.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Axolotl AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1326645308322615421)** (719 messages🔥🔥🔥): 

> `GPU Compatibility with AI Models, Image to Video Generation Challenges, Discord Community Dynamics, UI/UX Preferences in AI Tools, Kernel Panic in Linux Systems` 


- **GPU Compatibility with AI Models**: Users discussed the performance of different GPUs such as the RTX 4070 and 3090 in generating videos, indicating that while a 3090 may produce 480p video in about 2 minutes, the speed could vary with other models.
   - Certain models, like LTXV, claim to support image-to-video features but present different performance metrics depending on the user’s setup.
- **Image to Video Generation Challenges**: Accusations of poor quality with older video generation models like AnimateDiff led to discussions about exploring newer methods that combine various technologies for better results.
   - Users debated the merits of using workflows from various platforms, with specific instructions shared for implementing animated video generation in ComfyUI.
- **Discord Community Dynamics**: The community acknowledged the presence of inappropriate behavior and profile pictures among members, prompting discussions about reporting and moderation on Discord.
   - Concerns were raised about the challenges of navigating community standards and the impact of toxic behavior on user experiences.
- **UI/UX Preferences in AI Tools**: Users expressed differing opinions on the usability of various AI tools, comparing the interfaces of A1111, SwarmUI, and ComfyUI in terms of user experience and accessibility.
   - While some prefer the straightforward nature of A1111, others find value in SwarmUI's advanced capabilities despite a steeper learning curve.
- **Kernel Panic in Linux Systems**: Discussions about running Stable Diffusion models on Linux brought up technical concerns, such as kernel panics and compatibility issues with newer Python versions.
   - Users shared links to guides and resources for setting up and troubleshooting various systems to optimize their AI workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use">Image to Video ( ComfyUI Workflow using AnimateDiff and IP Adapter ) ready to use | Civitai</a>: Workflow is in the attachment json file in the top right. attached is a workflow for ComfyUI to convert an image into a video. it will change the i...</li><li><a href="https://civitai.com/articles/7993/lazy-tutorial-or-how-to-use-trainer-lora-on-colab-or-sd-15-and-xl-by-mikus-silly-and-easy">Lazy tutorial | How to use Trainer LoRA on Colab |  SD 1.5 &amp; XL by:: mikus (silly and easy) | Civitai</a>: WARNING! I’m not very experienced in this matter, so I recommend first learning all the functionality and reading a few more tutorials on how to do...</li><li><a href="https://civitai.com/articles/6182/how-to-make-a-lora-on-colab">How to Make a LoRA on Colab | Civitai</a>: Batch crop (1024x1024) and upscale (I use 4x_NMKD-UltraYandere_300k) under the extra tab in WebUI (batch from directory),uploaded to Drive, run thr...</li><li><a href="https://civitai.com/models/548997/image-to-video-comparison-workflow">Image-to-video Comparison Workflow - v1.0 | Stable Diffusion XL Workflows | Civitai</a>: Summary This workflow was made as an experiment to compare various technologies supporting &quot;image to video&quot;. In fact, it allows comparing the follo...</li><li><a href="https://tenor.com/view/cyanide-and-happiness-distraught-shocked-diagnosis-gif-23623883">Cyanide And Happiness Distraught GIF - Cyanide And Happiness Distraught Shocked - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://np.reddit.com/user/kejos92/comments/1hjkkmx/ltxv_inference_on_amd_gpus/">LTXV inference on AMD GPUs</a>: # Intro On the last 2 months, following the...</li><li><a href="https://github.com/Stability-AI/StableSwarmUI/blob/master/docs/Features/IPAdapter-ReVision.md">StableSwarmUI/docs/Features/IPAdapter-ReVision.md at master · Stability-AI/StableSwarmUI</a>: StableSwarmUI, A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - Stability-AI/StableSwarmUI</li><li><a href="https://civitai.com/models/134056/explosm-cyanide-and-happiness-style">Explosm Cyanide and Happiness style - 2 | Stable Diffusion LoRA | Civitai</a>: recommended settings 0.8-1.2 for negative use : nose, chin, ears, cheeks, jawline cyanide and happiness ( use lipstick, breasts, to generate female...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux">Install and Run on AMD GPUs</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/.github/images/swarmui.jpg">SwarmUI/.github/images/swarmui.jpg at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/Swa...</li><li><a href="https://www.youtube.com/watch?v=PWgvGjAhvIw">Outkast - Hey Ya! (Official HD Video)</a>: Official HD Video for &quot;Hey Ya!&quot; by OutKastListen to OutKast: https://Outkast.lnk.to/listenYDSubscribe to the official Outkast YouTube channel: https://Outkas...</li><li><a href="https://civitai.com/articles/7993/lazy-tutorial-or-how-to-use-trainer-lora-on-colab-or-sd-15-and-xl-">Lazy tutorial | How to use Trainer LoRA on Colab |  SD 1.5 &amp; XL by:: mikus (silly and easy) | Civitai</a>: WARNING! I’m not very experienced in this matter, so I recommend first learning all the functionality and reading a few more tutorials on how to do...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki">Home</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://github.com/wileewang/TransPixar">GitHub - wileewang/TransPixar</a>: Contribute to wileewang/TransPixar development by creating an account on GitHub.</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/2151">Remove license-infringing / potentially malicious / obfuscated code by parsee-mizuhashi · Pull Request #2151 · lllyasviel/stable-diffusion-webui-forge</a>: See also this PR in the appropriate repositoryLicense InfringingThis code is copied, at least partially, from ComfyUI, which has a GPL-3.0 license, which prohibits releasing compiled code without...</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1326644994135822396)** (393 messages🔥🔥): 

> `Unsloth updates, PHI-4 model fixes, Quantum models comparison, Adapters in fine-tuning, Chat templates in LLMs` 


- **Unsloth rolls out PHI-4 bug fixes**: Unsloth's PHI-4 version has surpassed the official Microsoft version on the Open LLM Leaderboard, showcasing significant improvements. Updates include critical bug fixes reported in the Unsloth blog and a continuous commitment to enhancing model performance.
   - Despite passing the Microsoft version, some users noted discrepancies with quantized models outperforming non-quantized ones in certain areas.
- **Best practices for using Adapters**: For inference, it is recommended to use the 16-bit model instead of the 4-bit quantized model when attaching adapters to ensure better performance. Using lower precision models for creating adapters can introduce undesirable losses.
   - While fine-tuning can yield similar results using either precision, merging adapters is best done with higher precision models.
- **Chat templates and fine-tuning considerations**: Chat templates are vital for both fine-tuning and the deployment of LLMs, as they inform how the model processes inputs and provides outputs. The template used during training can be found in the tokenizer_config.json file.
   - The correct design of chat templates can significantly affect the performance and usability of LLMs in production scenarios.
- **Linking computational neuro with LLMs**: Discussions highlighted the parallels between advancements in computational neuroscience and improvements in large language models. Users expressed curiosity about the implications of pruning and boosting techniques inspired by brain function.
   - Despite challenges, the integration of these insights continues to drive interest in optimizing model performance.
- **Compatibility and future directions**: Contributors noted the necessity of keeping libraries up-to-date, with some repositories potentially becoming outdated. Users discussed the importance of foundational models and libraries and their compatibility over time.
   - The community is encouraged to explore forks or alternative implementations to stay aligned with current advancements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=phi-4&rowSize=large">Open LLM Leaderboard - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://runpod.io?ref=bb842lb3">RunPod - The Cloud Built for AI</a>: Develop, train, and scale AI models in one cloud. Spin up on-demand GPUs with GPU Cloud, scale ML inference with Serverless.</li><li><a href="https://xkcd.com/1425/">Tasks</a>: no description found</li><li><a href="https://huggingface.co/unsloth/phi-4-GGUF">unsloth/phi-4-GGUF · Hugging Face</a>: no description found</li><li><a href="https://rog.asus.com/us/laptops/rog-strix/rog-strix-scar-18-2025/">ROG Strix SCAR 18 (2025) G835 | Gaming laptops｜ROG - Republic of Gamers｜ROG USA</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/en/index">PEFT</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1877136074042126338">Tweet from Unsloth AI (@UnslothAI)</a>: Phi-4, including GGUF + 4-bit + 16-bit versions are now on @HuggingFace!We found & fixed 4 bugs in Phi-4 & Llamafied the model.View all Phi-4 versions with our bug fixes: https://huggingface.co/collec...</li><li><a href="https://huggingface.co/learn/cookbook/en/llm_judge">Using LLM-as-a-judge 🧑‍⚖️ for an automated and versatile evaluation - Hugging Face Open-Source AI Cookbook</a>: no description found</li><li><a href="https://huggingface.co/collections/unsloth/phi-4-all-versions-677eecf93784e61afe762afa">Phi-4 (All Versions) - a unsloth Collection</a>: no description found</li><li><a href="https://x.com/Unsl">Tweet from FxTwitter / FixupX</a>: Sorry, that user doesn't exist :(</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Iq1JeXKYg5k">RTX 5090 Laptops Are Here!</a>: Nvidia&#39;s Blackwell 50 series Laptops are hereRTX 5090, RTX 5080, RTX 5070Ti, RTX 5070RTX 5090 Laptops - https://rog.asus.com/us/laptops-group/Nvidia - https:...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hwzmqc/phi4_llamafied_4_bug_fixes_ggufs_dynamic_4bit/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/cognitivecomputations/laserRMT">GitHub - cognitivecomputations/laserRMT: This is our own implementation of &#39;Layer Selective Rank Reduction&#39;</a>: This is our own implementation of &#39;Layer Selective Rank Reduction&#39; - cognitivecomputations/laserRMT</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements">Unsloth Requirements | Unsloth Documentation</a>: Here are Unsloth&#x27;s requirements including system and GPU VRAM requirements.</li><li><a href="https://github.com/unslothai/unsloth/pull/1516">Bug fixes by danielhanchen · Pull Request #1516 · unslothai/unsloth</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1326655511847763978)** (3 messages): 

> `Job search success, Funny reaction GIF` 


- **Infinit3e lands a job!**: A member announced that their job search is complete, stating, *'job search done im now employed.'*
   - Community members expressed their congratulatory reactions to this milestone.
- **Amogus6969 GIF shared**: A member shared a humorous GIF featuring a man in a suit making a funny face, which can be viewed [here](https://tenor.com/view/amogus6969-gif-26819393).
   - The GIF's description notes the character's reaction as entertaining and suitable for the context.



**Link mentioned**: <a href="https://tenor.com/view/amogus6969-gif-26819393">Amogus6969 GIF - Amogus6969 - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1326791388213284935)** (48 messages🔥): 

> `Mathstral Model Status, Recommendations for AI Models, RAG as an Option, Finetuning with LORA, Error with Qwen2VL Model` 


- **Mathstral Model Clarification**: A member clarified that `mistralai/Mathstral-7B-v0.1` is neither a base model nor a PEFT model, indicating a limitation in current support.
   - Another member mentioned that support for it is expected soon, showcasing ongoing development.
- **Advice for Splitting Names and Gender Identification**: A member sought advice on developing an AI to parse full names and identify gender, considering fine-tuning options.
   - Others suggested classical ML approaches, pointing out the inefficiency of using an LLM for this task, emphasizing its historical context.
- **RAG Suggested for Continuous Content**: A member advised considering RAG (Retrieval-Augmented Generation) for tasks involving continuously updated content.
   - They suggested starting with a 'rag tutorial' on YouTube for instructional guidance.
- **Finetuning Discussion on LORA with Model Merging**: A discussion emerged on using 16B LORA models for merging, questioning the implications of using an upscaled 4Q base model vs. the original.
   - Participants agreed that training and merging with the 16B model is safe, but concerns about potential downsides surfaced.
- **Error Encountered with Qwen2VL Model**: A user encountered a RuntimeError regarding an embedding module when using Qwen2VL while training a vision model.
   - They later discovered that switching to Llama3.2-vision-instruct resolved the issue, confirming that Qwen2VL might be broken.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@bnjmn_marie/lora-load-and-merge-your-adapters-with-care-3204119f0426">LoRA: Load and Merge Your Adapters with Care</a>: The case of LoRA adapters fine-tuned with QLoRA</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1326857299767066746)** (4 messages): 

> `DLSS for Language Models, Speculative Decoding` 


- **Exploring DLSS-like Techniques for Language Models**: After discussions about DLSS at CES, a member wondered if similar techniques exist for language models that could optimize training or inference resources.
   - They specifically sought research related to predicting next steps efficiently to reduce resource consumption.
- **Speculative Decoding Suggested**: In response to the inquiry about resource-saving methods for language models, speculative decoding was suggested as a potential solution.
   - Another member expressed gratitude, noting that speculative decoding was a perfect answer to their question.


  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1326645517907787849)** (125 messages🔥🔥): 

> `Self-hosted Codeium, Windsurf issues, Cascade Model discussion, Purchase of credits, User experiences with Codeium` 


- **Self-hosted Codeium Now Available**: A member highlighted that a self-hosted version of **Codeium** is now included in the enterprise offering.
   - *This feature seems to generate curiosity among users who want to explore self-hosting options.*
- **Windsurf Authentication Problems**: Several members reported issues with authenticating on **codeium.com** and connecting with **Windsurf** after purchasing credits.
   - Various suggestions included logging out and logging back in again as a potential fix to address the error messages.
- **Cascade Model's Flexibility Discussed**: Users are praising the **Cascade Model** ability, particularly noting its unlimited nature in the base version versus the limited queries in the premium model.
   - One user shared that they effectively built a company website without writing code, demonstrating the model's capabilities.
- **Concerns Over Credit Purchases**: A member inquired about the addition of newly purchased credits to their updated plan, expressing urgency for resolution.
   - It was suggested to send email details for faster processing of inquiries related to credit issues.
- **User Experiences with Codeium Transformation**: Members shared positive transformations in their coding habits and productivity due to the use of **Windsurf** and **Cascade**.
   - One user stated they had become a completely different person due to the productivity enhancements provided by these tools.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://codeium.com/pricing#are-credits-consumed-for-command-and-chat-in-codeium-extensions-vs-code-and-jetbrains-ides">Pricing | Windsurf Editor and Codeium extensions</a>: Codeium is free forever for individuals. Teams can level up with our enterprise offering for enhanced personalization and flexible deployments.</li><li><a href="https://github.com/Exafunction/codeium.el/issues/115">How do I get my api key? · Issue #115 · Exafunction/codeium.el</a>: I&#39;m trying to find my api key on codeium.com but cannot. Where should I look?
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1326642701403422740)** (140 messages🔥🔥): 

> `Windsurf Installation, Cascade Chat Optimizations, Flow Credits Discrepancies, Agent Integration in Windsurf, User Experience Issues` 


- **Windsurf Installation Success Stories**: Users shared successful installations of Windsurf on various operating systems, with some mentioning smooth experiences on Mint and Ubuntu 24.04.
   - One user noted a similar success on Arch and Hyprland after deleting configuration folders, leading to a working setup.
- **Cascade Chat Not Syncing Across Devices**: A user inquired about using Cascade chats across multiple PCs via Dropbox, highlighting a need for chat synchronization.
   - Others expressed similar desires for enhanced chat management features within Windsurf.
- **Flow Credits Confusion and Billing Issues**: Multiple users reported discrepancies with Flow Credits purchase and usage, questioning the value and rollover mechanics.
   - Concerns were raised about being charged twice for credits without receiving them, and user attempts to resolve billing issues proved challenging.
- **Internal Errors and Flow Action Consumption**: Users expressed frustration with Windsurf's tendency to generate unnecessary outputs, leading to excessive Flow Action consumption.
   - Discussions included whether internal errors counted against Flow Actions and the impact of the recent update on usability.
- **User Experience Feedback**: A user lauded the capabilities of Windsurf for rapid app development, while others criticized recent updates for being unresponsive.
   - Concerns about unclear release notes and functionalities such as Composer and commit operations were raised, indicating a need for clearer communication from developers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.</li><li><a href="https://codeium.canny.io/">Codeium Feedback</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1326643664906092615)** (246 messages🔥🔥): 

> `Cursor IDE performance issues, Using Cursor Rules for better outputs, Challenges with Composer, Connecting with Cursor Developers, User experiences with Claude` 


- **Cursor IDE experiencing performance issues**: Users reported difficulties with Cursor IDE, including 'slow pool full' messages and Composer being stuck, prompting concerns about stability and functionality.
   - Issue persists across versions with suggestions to try restarting or checking file indexing to mitigate problems.
- **Enhancing outputs using Cursor Rules**: Users discussed the importance of setting Cursor Rules to guide Claude's outputs, with suggestions to articulate goals clearly in prompts.
   - Specific keywords can enhance the prompts, allowing for a structured approach to feature requests.
- **Challenges and frustrations with Composer**: Multiple users expressed dissatisfaction with Composer, noting that it often ignores preset rules and can apply unintended changes across files.
   - Some suggested reverting to previous stable versions or using more granular prompts to avoid issues during code changes.
- **Finding connections with Cursor developers**: Members shared various ways to connect with Cursor staff, including posting on the forum or reaching out via social media.
   - Informal methods of contacting developers were humorously suggested, illustrating the community's frustrations and lightheartedness.
- **User experiences and expectations with Claude**: Many users shared their experiences with Claude, noting that it sometimes does not incorporate rules or prompts effectively leading to disappointing outputs.
   - Despite criticisms, users acknowledged instances where Claude produced satisfactory results, indicating mixed feedback on its capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forum.cursor.com/t/composer-stuck-at-generating-specific-composer-instance-not-global-issue/35479/4">Composer Stuck at &quot;Generating&quot; - Specific Composer Instance, Not Global Issue</a>: Hey… any luck on this yet? I’m still seeing stuck Composer sessions. I’ve just upgraded to 0.44.10; my current session which was stuck in 0.44.9 remains stuck in 0.44.10.  It sits on “generating” for ...</li><li><a href="https://onecompiler.com/bootstrap/435jnyccv">Card Glow Magnetic - Bootstrap - OneCompiler</a>: no description found</li><li><a href="https://forum.cursor.com/">Cursor - Community Forum</a>: A place to discuss Cursor (bugs, feedback, ideas, etc.)</li><li><a href="https://cursor.directory/">Cursor Directory</a>: Find the best cursor rules for your framework and language</li><li><a href="https://dotcursorrules.com/">.CursorRules</a>: Cursor Rules to customize Al behavior, streamline the development process and tailor code generation,suggestions and queries to your framework and language.
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1326656125126447185)** (11 messages🔥): 

> `Prompting Techniques, Payment System, Public Repos Feature, Sleep Schedule Impact` 


- **Effective Prompting for Color Usage**: Members discussed the importance of specifying **colors** in prompts using color names and hex codes, emphasizing clarity on where to apply each color.
   - *Don't just say 'Make me a timer app. blue and white colors'*; it's better to provide a general idea.
- **Payment System Not Operational**: A user noted that the **payment system** for their app is currently not functioning, hinting at ongoing development.
   - They shared a link to their project, urging for testing or feedback as they work to resolve the payment issues.
- **Public Repos Feature Announcement**: A member highlighted that the team posted about a new feature enabling users to open **public repos** in bolt.new, which was released back in October.
   - Users can access any GitHub URL by simply prefixing it with *http://bolt.new*, as announced in their [X post](https://x.com/stackblitz/status/1843668731681267801).
- **Sleep Schedules Affecting Response Time**: One member apologized for a delayed response due to their **messed-up sleep schedule**, indicating the challenges of maintaining communication.
   - Another member expressed understanding and appreciation for the communication, emphasizing the community's supportive environment.
- **Prompting Resources Available**: A user shared a link to a resource on **prompting** made with bolt.new, inviting others to ask questions.
   - This indicates an initiative to foster knowledge sharing about effective prompting strategies within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://subredditai.com">SubReddit AI</a>: no description found</li><li><a href="https://x.com/stackblitz/status/1843668731681267801">Tweet from StackBlitz (@stackblitz)</a>: You can now open public repos in bolt․new 🙌How? For any GitHub URL, just put &#34;http://bolt.new&#34; in front of it!(Release notes below!)
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1326647185496080484)** (211 messages🔥🔥): 

> `Bolt Token Issues, PWA Support in Bolt, Supabase Migration Concerns, Netlify Performance Problems, Community Feedback and Features` 


- **Bolt Token Issues**: Users express frustrations over Bolt consuming tokens rapidly while attempting to make edits or troubleshoot errors in their projects.
   - Some reported spending significant tokens for repeated actions while experiencing issues with the tool's confidence in its fixes.
- **PWA Support in Bolt**: A user inquired about Progressive Web Apps (PWA) support within Bolt, citing an error message suggesting underlying Stackblitz limitations.
   - Others commented on successfully deploying PWAs, indicating that it should be feasible despite the error experienced by the original user.
- **Supabase Migration Concerns**: Concerns were raised about migration rollbacks in Supabase not being handled alongside codebase changes after encountering errors.
   - Users discussed potential challenges in reversing migrations and maintaining application functionality following unsuccessful updates.
- **Netlify Performance Problems**: One user reported slow loading times for a Bolt website hosted on Netlify, questioning whether the issue stemmed from Bolt’s code or Netlify's limitations.
   - The user speculated that their free account might be affecting performance, hinting at the possibility of needing an upgrade for better service.
- **Community Feedback and Features**: Suggestions were made regarding creating feedback and guide channels in the Discord community to aid users in learning and troubleshooting.
   - Community members emphasized the importance of clear documentation and support for new users to enhance their experiences with Bolt.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://boltstudio.ai">BoltStudio.ai | Full Stack Prompt Engineering</a>: no description found</li><li><a href="https://bolters.io">Bolters.io | Community Supported Tips, Tricks &#38; Knowledgebase for Bolt.new No-Code App Builder</a>: Documentation and guides for Bolt.new</li><li><a href="https://github.com/stackblitz/bolt.new/issues/5149">Suggestion: Selector · Issue #5149 · stackblitz/bolt.new</a>: This is my suggestion to add a selector option for the sites. I will try to explain in more detail: When you highlight with your mouse and go to chat and say for example change the name or remove t...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/2529">Bolt Outputs Application Logic in Chat · Issue #2529 · stackblitz/bolt.new</a>: Issue: Bolt outputs application logic in the chat. For example, when the user hits a rate limit, the code to offer a link to upgrade is sent as a response to the user in chat.</li><li><a href="http://bolt.diy">GitHub - stackblitz-labs/bolt.diy: Prompt, run, edit, and deploy full-stack web applications using any LLM you want!</a>: Prompt, run, edit, and deploy full-stack web applications using any LLM you want! - stackblitz-labs/bolt.diy
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1326689892834476083)** (66 messages🔥🔥): 

> `Aider User Experiences, Comparison of AI Models, Model Capabilities and Improvements, Coding Assistant Development, OpenAI and Gemini Models` 


- **Aider's Behavior Compared to Claude**: Users have observed that **DeepSeek** seems *lazy* as an editor compared to **Claude**, with remarks about its tendency to distract and fail to execute commands accurately.
   - Contributors noted that **Aider** has inconsistent results, with one member humorously recounting times when it deleted entire files while fixing issues.
- **Discussion on AI Model Competence**: **AI models' performance varies**, users debated the efficiency of models like **DeepSeek** and alternatives from **Anthropic**, highlighting cost concerns on usage.
   - There were discussions about the current popular models, emphasizing the **instability** and **cost of using Anthropic's offerings**.
- **Future of AI as Proactive Agents**: A user optimistically stated that **AI will become more proactive**, suggesting a future where AI can create better versions of themselves and involve humans less.
   - In contrast, a member cautioned that **current AI limitations** stem from power and computation costs, potentially slowing progress.
- **Developing AI Coding Assistants**: A user expressed a desire to develop their own **coding assistant agent**, finding **Aider** to be a suitable alternative and seeking contributions to the project.
   - Contributors discussed methods to integrate various functionalities into **Aider**, including automatic code revision through issue tracking.
- **Clarification on OpenAI Model Naming**: An inquiry was made regarding the difference between model names starting with **openai/** and others without, with a clarification that they are essentially the same.
   - Another user sought to understand why certain models are prefixed, with **OpenAI's flexibility** in naming highlighted in responses.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting and testing</a>: Automatically fix linting and testing errors.</li><li><a href="https://github.com/Aider-AI/aider/blob/main/CONTRIBUTING.md">aider/CONTRIBUTING.md at main · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/pull/540">feat: add `/rg` command (ripgrep for `/add` files) by aleclarson · Pull Request #540 · Aider-AI/aider</a>: NOTE: Using this command requires ripgrep installed on your machine.How it worksIt calls rg through a subprocess with the -l flag to return a list of filenames. These filenames are then fed into ...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1326651345029169212)** (61 messages🔥🔥): 

> `Aider Configuration Issues, Using OpenAI Providers, DeepSeek Performance, Task Management in Aider, Handling Chat History with Aider` 


- **Aider sends 'prompt' instead of 'messages'**: A user reported that Aider is sending a `prompt` list instead of a `messages` list when communicating with a local litellm proxy, leading to errors like `TypeError: Router.acompletion() missing required argument: 'message'`. This raised questions on the expected configuration settings in the JSON files.
   - Another user pointed out that the `litellm_provider` needs to match the initial part of the model name for successful communication.
- **Access to Tier 5 OpenAI Key**: A user sought alternatives for accessing a tier 5 OpenAI key without the high costs and was directed to providers like OpenRouter and Unify.ai. The discussion included potential workarounds and the implications of using different models and subscription plans.
   - Information about using the `Unify` API was shared, asserting that it allows access to multiple models with a single key, although it is not open source.
- **Performance Issues with DeepSeek**: A user inquired about performance issues with deepseek-chat or deepseek-coder that gets stuck after several requests, mentioning a direct connection to deepseek APIs. Some users noted encountering slowdowns and suggested routing changes through different networks could improve the experience.
   - Others confirmed they use DeepSeek regularly without issues, speculating that it may be a matter of server load or model availability.
- **Managing Tasks in Aider**: A user asked about organizing multiple suggestions for tasks generated by Aider, seeking advice on managing them effectively. A useful tip was given about using a TODO.md file to track tasks directly within the Aider chat.
   - One participant emphasized the efficiency of having separate Aider instances to manage context better and prevent the model from becoming overloaded with information.
- **Handling Chat History Retention**: Concerns were raised about Aider retaining chat history, which sometimes leads to repetitive suggestions despite previous corrections. Users were advised to use the `/clear` command to reset history and avoid confusion during ongoing sessions.
   - Another user inquired about improving prompt preparation to reduce the recurrence of incorrect answers, highlighting the overall need for better state management within Aider.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/unifyai/unify">GitHub - unifyai/unify: Build Your AI Workflow in Seconds ⚡</a>: Build Your AI Workflow in Seconds ⚡. Contribute to unifyai/unify development by creating an account on GitHub.</li><li><a href="https://unify.ai/">Unify: Build AI Your Way</a>: Too Many Tools, Too Complex, Build Your Own Workflow in Seconds!
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1326675570632822886)** (1 messages): 

> `Gemini 2.0 Flash Experimental, Voice mode on iOS, App development with AI assistance` 


- **Driving Conversations with Gemini 2.0**: While running errands, the user engaged with **Gemini 2.0 Flash Experimental** using its **voice mode** on iOS, discussing an app idea as if it were a passenger.
   - Although it didn't provide markdown files, the AI autonomously established project criteria and generated concise task bullet points upon returning home.
- **AI-Facilitated App Specification**: The user requested the AI to help flesh out a specification and concrete tasks for an app development project.
   - Gemini 2.0 successfully guided the conversation and provided a useful summary of actionable steps for future reference.


  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1326658028476301342)** (19 messages🔥): 

> `Importing videos into NotebookLM, DeepResearch reports integration, Generating Mandarin podcasts, Quotation Mode for direct quotes, System prompts in NotebookLM Plus` 


- **Importing Videos into NotebookLM not feasible**: A member humorously inquired if someone attempted to import a video into NotebookLM, but the system lacks the capability to handle it.
   - Another user mentioned a workaround using transcriptions in ChatGPT to generate a table of contents.
- **Exploring DeepResearch Reports for NotebookLM**: A member asked if anyone uses DeepResearch reports with NotebookLM and suggested incorporating the sources from these reports into the system.
   - A response mentioned there isn't direct integration but proposed using extensions to bulk upload sources.
- **Creating Mandarin Podcasts from English Content**: A user sought advice on whether it's possible to generate Mandarin podcast content from English sources using NLM.
   - This prompted others to share insights on how they modify podcast scripts to be more casual and conversational.
- **Quotation Mode Implementation**: A member shared a command that instructs NotebookLM to respond solely with direct quotes from sources for clarity and verification.
   - This setup was noted to have issues with Gemini, as it sometimes responded with incomplete information.
- **Clarification on System Prompts in Plus**: A user questioned the existence of system prompts in Plus since their interface appeared identical to the free version.
   - Responses clarified that consistent use of commands tied to system prompts is only possible in Plus, which enhances functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.akashq.com/post/122c0310-6683-45d7-adec-3d3f4bbebd16">What happened on Jan 9?</a>: What happened on Jan 9? by This Day in History</li><li><a href="https://www.akashq.com/post/ad632a26-91b5-44b4-b8f4-5b5fd3f083e8">What happened on Jan 8?</a>: What happened on Jan 8? by This Day in History
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1326641747840995348)** (94 messages🔥🔥): 

> `NotebookLM functionality, Audio generation features, Workspace license issues, Language options in conversations, User experience with podcasts` 


- **Troubleshooting NotebookLM Access Issues**: Members discussed accessing NotebookLM Plus with Workspace accounts, highlighting that only those with Business licenses and domain verification can utilize it effectively.
   - Some expressed frustration over features not working and suggested troubleshooting methods like refreshing the page or re-uploading files.
- **Generating Audio with Selected Sources**: Users inquired about generating podcasts limited to specific sources, with one solution being to specify sources in the customization prompts.
   - Tips for using related tools like Illuminate for audio generation were shared, aiming to enhance production flexibility.
- **Japanese Language Conversations**: There was a seamless transition into Japanese language discussions, showing user comfort with switching languages within the chat.
   - Users confirmed their ability to communicate effectively in Japanese, ensuring language barriers are minimized.
- **Enhancements and Alternatives to NotebookLM**: A user compared NotebookLM to alternative platforms like Jellypod, emphasizing more customization options in the latter.
   - Suggestions for future improvements to accessibility and voice variety were made, highlighting user needs for educational purposes.
- **Feature Removals and User Adaptations**: Users expressed dissatisfaction with the removal of certain features, such as AI-generated question suggestions from PDF text selections.
   - Workarounds and alternative tips were provided to adapt to these changes, keeping user engagement in mind.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.icloud.com/iclouddrive/061hg1R50Jv4idRhgdUqoMxWg#Captura_2025-01-09_a_las_8.15">iCloud Drive - Apple iCloud</a>: no description found</li><li><a href="https://www.techradar.com/computing/artificial-intelligence/ive-found-a-new-ai-podcast-creator-and-it-leaves-googles-notebooklm-in-the-dust">I&rsquo;ve found a new AI podcast creator, and it leaves Google&rsquo;s NotebookLM in the dust</a>: Jellypod lets you present your own podcast, without going anywhere near a mic</li><li><a href="https://notebooklm.google.com/notebook/982b3b0c-0913-4599-816a-9c845a6b7d79/audio">no title found</a>: no description found</li><li><a href="https://illuminate.google.com/create">Illuminate | Learn Your Way</a>: Transform research papers into AI-generated audio summaries with Illuminate, your Gen AI tool for understanding complex content faster.</li><li><a href="https://notebooklm.google.">Google NotebookLM | Note Taking &amp; Research Assistant Powered by AI</a>: Use the power of AI for quick summarization and note taking, NotebookLM is your powerful virtual research assistant rooted in information you can trust.</li><li><a href="https://akashq.com">Akas: home to AI podcasts</a>: no description found</li><li><a href="https://youtu.be/spj0n-bFKJo"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1326662295522381945)** (66 messages🔥🔥): 

> `LM Studio and API Connectivity, Model Loading Issues, Directory Structure for Models, New Qwen Chat Feature Announcement, LLM Applications and Development Trends` 


- **LM Studio Connectivity Challenges**: Several users reported issues with **LM Studio** not connecting to the API or failing to load models without clear error messages, particularly affecting those using older versions like **0.2.26**.
   - User **friiscs2** resolved their issues after ensuring they opened the app from the applications folder instead of the installation GUI, highlighting potential confusion around the installer.
- **Directory Structure for Model Compatibility**: **Marsv.** expressed frustration about LM Studio requiring a specific sub-directory structure for models, causing conflicts with other apps that have a more unified model directory format.
   - Users suggested that alternating directory structures between LM Studio and Ollama might lead to a future convergence of features in various LLM apps.
- **Qwen Chat Feature Launched by Alibaba**: Alibaba announced the launch of **Qwen Chat**, a new Web UI for interacting with various Qwen models, boasting features like document uploads and visual understanding capabilities.
   - This chat interface aims to integrate multiple models and is anticipated to roll out additional features like web search and image generation soon.
- **User Exploration in LLM Applications**: Several users shared their experience experimenting with multiple LLM applications, noting that they often converge on similar core features over time.
   - **Skeletonbow** highlighted the fun of developing a custom chat client leveraging LM Studio as a backend, while keeping up with unique features offered by other apps.
- **OpenCL Backend Support Inquiry**: User **Uniraa** inquired about plans for OpenCL backend support for Snapdragon X Elite and Windows on ARM, noting recent developments in **Llama.cpp**.
   - This highlights ongoing interest in enhancing support for various hardware in the AI landscape among developers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: Discover, download, and run local LLMs</li><li><a href="https://fxtwitter.com/Alibaba_Qwen/status/1877426465349972113">Tweet from Qwen (@Alibaba_Qwen)</a>: 🚀 Exciting News! We&#39;re thrilled to announce the launch of Qwen Chat ( https://chat.qwenlm.ai ) – your new go-to Web UI for interacting with Qwen models! 🌟💬 Chat effortlessly with our flagship m...</li><li><a href="https://lmstudio.ai/docs/advanced/sideload">Sideload models - Advanced | LM Studio Docs</a>: Use model files you&#x27;ve downloaded outside of LM Studio
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1326644383373725738)** (33 messages🔥): 

> `AMD RX 7900XT performance, Sidecar graphics cards for MacBook Pro, Memory requirements for Llama 3.3, GPU activity monitoring tools, Benchmarking DIGITS arrival` 


- **AMD RX 7900XT struggles with benchmarks**: A member posed a question about the **7900XT 20GB** performance compared to **4090, 4080,** and **3090**, pointing to potential lower memory bandwidth in comparisons.
   - Another member shared a [reddit link](https://reddit.com) to results contrasting the **7900XT** with the **3090**.
- **No Sidecar Graphics for MacBook Pros**: Members discussed the impossibility of using a 'sidecar' graphics card with a **MacBook Pro** due to Apple silicon limitations, marking a change from older Intel models.
   - The suggestion of **Thunderbolt 5** connectivity was floated for external GPUs, reflecting a desire for greater model access.
- **Memory critique for Llama 3.3**: A user trying to run **Llama 3.3 70B Instruct** on limited hardware (Ryzen 7 and **RX 7900GRE**) noted slow performance at **0.5 token/sec**, raising concerns about RAM adequacy.
   - Members concurred that full GPU memory usage is essential for optimal speed, advocating for RAM adjustments or better hardware configurations.
- **Tools for system bottleneck analysis**: Discussion included tools like **htop** and **nvtop** for Linux users, with some exceptions in GPU activity display.
   - Windows users were advised on the availability of various free software tools for monitoring system performance.
- **Excitement about DIGITS release**: Despite expecting **DIGITS** to arrive earlier, one member expressed optimism about its potential as a comprehensive solution within the **NVIDIA stack**.
   - Concerns were raised regarding performance speed, underscoring anticipation for its practical utility in future projects.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1326658567394164767)** (60 messages🔥🔥): 

> `Model Versions and Testing, TensorFlow GPU Issues, Machine Learning Resources, Jupyter vs Python File Debugging, Community Concerns about AI Safety` 


- **Model O1 Discussion**: A member pointed out the unique format of model O1 outputs, noting it has 'thinking' mentioned, suggesting different model formats might be in play.
   - Another member questioned whether a reasoning model would be utilized alongside model 4O in A/B testing scenarios.
- **TensorFlow GPU Not Detected**: A user reported that their Jupyter kernel couldn't detect their NVIDIA GPU despite installing CUDA, cuDNN, and tensorflow-gpu with specifications of **64G RAM** and **RTX 3060**.
   - Members shared troubleshooting steps like ensuring the environment is activated and recommending running `conda env list` to confirm the environment setup.
- **Seeking ML Learning Resources**: A member asked for the best YouTube channels for learning machine learning, indicating a shared interest in community recommendations.
   - Another member jokingly responded to avoid issues with OpenAI's model safety measures, suggesting the conversation may devolve into concerns over AI regulations.
- **Debate on Debugging Approaches**: A discussion emerged regarding the advantages of using a standard Python file with breakpoints versus Jupyter Notebook for easier code verification.
   - One member expressed satisfaction with solving their issues in VSCode, while others shared differing opinions on debugging preferences.
- **Concerns about AI Responsiveness**: Members expressed their frustrations regarding the tendency to blame tools instead of individuals for problems within AI discussions.
   - Concerns were raised that no matter how much effort is put into making AI models safer, 'jailbreaks' would always persist, questioning the efficiency of ongoing security measures.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1326721363750162513)** (7 messages): 

> `GPT code handling, ChatGPT generating graphs` 


- **Frustration with GPT's Code Responses**: Members expressed frustration that GPT continues to respond with comments instead of sending the full code requested, even after repeated prompts.
   - One noted a distinction in performance, stating that while GPT-4 manages requests adequately, GPT-3.5 frequently fails to provide the full code.
- **ChatGPT Surprise with Graph Generation**: A member remarked on the surprising ability of ChatGPT to generate a graph, prompting a reaction of disbelief from others.
   - The community responded with animated reactions, emphasizing their amazement at the AI's capabilities.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1326734331338821704)** (13 messages🔥): 

> `Meta-Prompting Use Cases, Insights on Prompting, Investor Round for Hassabis` 


- **Meta-Prompting generates interest**: Members shared their interest in **Meta-Prompting**, with one expressing a desire to explore innovative use cases for it.
   - Another member emphasized that a good prompt should start with a clear understanding of the desired output.
- **Making OpenAI Profitless**: One member noted that they feel like they get **zero dollars** from OpenAI, adding humor to the situation.
   - This sentiment about lacking financial compensation led to discussions about the group's closed nature.
- **Support for Investor Round in Hassabis**: A member asked for positive thoughts regarding the **investor round over Hassabis**, acknowledging his capabilities.
   - The community seems to recognize the impact and potential of his work in the AI space.
- **Chasing Effective Prompts**: A member expressed the challenge of expanding their understanding of what constitutes a **good prompt**.
   - The conversation reflected a collaborative spirit in refining prompting techniques among members.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1326734331338821704)** (13 messages🔥): 

> `Meta-Prompting, OpenAI Contributions, Investor Round, Prompt Creation` 


- **Exploring Meta-Prompting Use Cases**: A member asked about experiences with **Meta-Prompting** and interesting use cases, indicating curiosity about its implications and effectiveness.
   - Another member responded playfully, suggesting that modifying the system message can yield significant improvements.
- **OpenAI Financial Contributions in Question**: There was a discussion about the lack of financial rewards from OpenAI, with members expressing confusion over the group's lack of funding despite valuable contributions.
   - One member humorously remarked that they also received 'zero dollars' from OpenAI, highlighting a shared sentiment.
- **Prayer Request for Investor Round**: A member requested support with good prayers for **Hassabis** during an investor round, acknowledging his abilities.
   - This request underscored the group's desire for success in their ventures and recognition of talent in the field.
- **Building Effective Prompts**: In response to inquiries about prompt crafting, another member emphasized that understanding the desired output is critical for creating a good prompt.
   - This emphasizes a consensus in the group that clarity of purpose is vital when working with AI models.


  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1326757945525927977)** (3 messages): 

> `ICLR Event Attendance, Meeting Points and Descriptions` 


- **Excitement for ICLR Event**: A member expressed enthusiasm about attending the **ICLR** event, asking if others would be there.
   - This indicates a vibrant community engagement around the event.
- **Philpax’s Arrival and Meeting Details**: **Philpax** announced their arrival at the event shortly, noting they would be outside without mobile internet.
   - They described their appearance as wearing a **light brown coat**, **black jeans**, and carrying a **gym bag** and **backpack**, setting clear meeting points.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1326754105724768357)** (19 messages🔥): 

> `rStar-Math Performance, Qwen Chat Launch, O1 vs GPT4o + MCTS Discussion, Challenges in Chinese ML Startups, EpiCoder Framework` 


- **rStar-Math Scores High on Math Benchmarks**: Microsoft's new framework, **rStar-Math**, significantly boosts performance on the MATH benchmark, improving **Qwen2.5** from **58.8%** to **90.0%** and **Phi3-mini** from **41.4%** to **86.4%**.
   - On the USA Math Olympiad, it achieves an average score of **53.3%**, placing among the top **20%** of high school competitors.
- **Launch of Qwen Chat Web UI**: The new **Qwen Chat** is introduced, allowing users to engage with various Qwen models, including **Qwen2.5-Plus** and **Qwen2-VL-Max**, through a unified interface.
   - Upcoming features promise web search, image generation, and voice mode capabilities, enhancing user interaction with AI models.
- **Debate: O1 vs GPT4o + MCTS**: Discussion centers around whether **O1** is simply a more efficient version of approaches using **GPT4o** and **MCTS**, and if there are unique problems that O1 can solve with a reasonable compute budget.
   - Opinions vary, with some noting that MCTS may be more **expensive** while others discuss the complexities of self-correction in models.
- **Chinese AI Startup Faces Challenges**: Chinese AI startup **Zero One** is restructuring and shifting focus away from training large models to developing more practical models that can be monetized, citing funding and resource limitations.
   - Li Kaifu highlights the growing difficulties in the Chinese AI landscape, noting constraints from **chip availability** and **financing** as major hurdles.
- **Introduction of EpiCoder for Code Generation**: **EpiCoder** is announced as a new hierarchical framework aimed at improving code generation, showing advanced capabilities across various project complexities.
   - **Open source** release is expected soon, promising to outperform existing benchmarks in coding tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TeamCodeLLM_AI/status/1877254042574844153">Tweet from Wavecoder (@TeamCodeLLM_AI)</a>: 🚀 Introducing EpiCoder: a hierarchical feature tree-based framework for diverse and intricate code generation.🔍 Outperforming benchmarks, it handles everything from simple functions to multi-file pr...</li><li><a href="https://x.com/_akhaliq/status/1877206745652592763?s=61">Tweet from AK (@_akhaliq)</a>: Microsoft presents rStar-MathSmall LLMs Can Master Math Reasoning with Self-Evolved Deep ThinkingOn the MATH benchmark, it improves Qwen2.5-Math-7B from 58.8% to 90.0% and Phi3-mini-3.8B from 41.4% to...</li><li><a href="https://x.com/JustinLin610/status/1877427101370036595">Tweet from Junyang Lin (@JustinLin610)</a>: Here it is! Qwen Chat (https://chat.qwenlm.ai), our new Web UI for Qwen models. The link is:chat dot qwen lm dot ai!chat dot qwen lm dot ai!chat dot qwen lm dot ai!You can chat with the most impressiv...</li><li><a href="https://mp.weixin.qq.com/s/IUA482JlwI4CcRpiMRGHbA">晚点对话李开复丨零一万物部分团队并入阿里，“灵魂拷问来得太快了”</a>: “机会来临时，要勇敢做决策，机会消失时也是。”
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1326754434746814515)** (21 messages🔥): 

> `NuminaMath dataset, Lead author background, Quality concerns in open data, High school math challenges, Business vs. coding in tech` 


- **Concerns about NuminaMath dataset quality**: A member expressed skepticism about the **quality** of the **NuminaMath** dataset, highlighting that 7.7% of entries include multiple boxed solutions and raise deeper quality concerns.
   - *'Problems like this underscore the state of open and publicly available data'* suggests systemic issues.
- **Interesting background of lead author**: Discussion revealed that the lead author on the paper is a **PhD student in psychology** at Stanford, raising eyebrows among members about the interdisciplinary nature.
   - Another member noted the contribution of **Charlie Snell** as the second author, adding to the paper's intrigue.
- **Challenges of competing in math**: A member shared their experience with the **cn_k12 subset** of NuminaMath, stating they concluded that they have 'no chance against Chinese high schoolers' after a research attempt.
   - This comment reflects broader challenges and competition faced in **math-related study** and research.
- **Light-hearted comments on psychology and education**: Members exchanged light-hearted remarks about educational pathways, mentioning a cousin's switch from **rabbinical studies to Economics**, showcasing diverse academic journeys.
   - Psycho-education paths were humorously referenced, suggesting a collective respect for rigorous admission processes in psychology programs.
- **Future focus on business in tech**: A member concluded that if coding challenges are resolved, the remaining frontier is **business**, emphasizing its importance in tech.
   - The conversation hints at a shift in focus from technical skills to business acumen among peers.



**Link mentioned**: <a href="https://arxiv.org/abs/2501.04682">Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought</a>: We propose a novel framework, Meta Chain-of-Thought (Meta-CoT), which extends traditional Chain-of-Thought (CoT) by explicitly modeling the underlying reasoning required to arrive at a particular CoT....

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1326642132345421866)** (11 messages🔥): 

> `Complexity of Large-Scale Providers, Transformers vs MoEs, Performance Efficiency in Models` 


- **Complexity in Large-Scale AI Providers**: Complexity in model architecture is acknowledged as worthwhile for **large-scale providers** to manage effectively.
   - *Seems like a lot of complexity to get right but obvs worthwhile for large scale providers.*
- **Transformers and MoE Efficiency Debate**: While MoEs seem to outperform dense models, a member suggested that **for loops** might provide an easier first understanding of these architectures.
   - However, there's an acknowledgment that higher complexity often means better performance overall.
- **MoEs Generally Outperform Dense Models**: One member emphasized that an **MoE** model is generally superior to a dense model if they maintain the same number of *active* parameters.
   - This aligns with the view that **more information** can be stored in greater weights, despite the simplicity drawbacks.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1326645897110491238)** (17 messages🔥): 

> `Anthropic salon, Character shaping in AI models, Post-training processes, Imposter syndrome among AI professionals, Blogging and self-care in academia` 


- **Josh Batson's Insights on Model Shaping**: During the recent Anthropic salon, **Josh Batson** mentioned that Amanda Askell shapes the base model into an agent, possibly implying that **character-related changes** occur earlier than expected.
   - Some members debated whether this shaping occurs primarily in post-training or if foundational adjustments happen earlier, hinting at a **distributed approach** to character alignment.
- **Debate on Character Development Timing**: A discussion arose about whether character work in AI should be considered a **final touch** rather than an intrinsic part of the development process, as suggested by **members' perspectives.**
   - Members illustrated this with metaphors, likening the base model to clay being shaped rather than simply a **post-training task**.
- **Imposter Syndrome in AI Community**: Participants shared experiences of **imposter syndrome**, with one noting how it continues to linger despite accomplishments in AI and ML fields.
   - They acknowledged it can serve as a twisted motivation, with one participant humorously remarking that it could even be considered a **superpower.**
- **Blogging Challenges for Academics**: A member mentioned the struggle of knowing which blog topics resonate, humorously noting that their post on **self-care** seems to appeal to many.
   - Another member conveyed a commitment to publishing their **deepseek blog**, despite challenges like learning MLA formatting and crafting extensive footnotes.



**Link mentioned**: <a href="https://youtu.be/IPmt8b-qLgk?si=Cg2M9u4Rc5X7MHwb&t=964">How difficult is AI alignment? | Anthropic Research Salon</a>: At an Anthropic Research Salon event in San Francisco, four of our researchers—Alex Tamkin, Jan Leike, Amanda Askell and Josh Batson—discussed alignment scie...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1326957605561827338)** (3 messages): 

> `Efficient Deep Learning, Popup Issues` 


- **New Insights on Efficient Deep Learning**: A member shared a [link to a blog on Efficient Deep Learning](https://alexzhang13.github.io/blog/2024/efficient-dl/) that discusses innovative methods and techniques.
   - This blog aims to improve understanding and application of efficient practices in deep learning.
- **Popup Blocks Page Viewing**: A member expressed frustration that a popup is obstructing part of the first page they were trying to view.
   - Another member humorously responded to this by saying, *Lmao brutal googling*.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1327017242222923907)** (14 messages🔥): 

> `Open Source AI Costs, AI Policy Maker Reactions` 


- **Open Source AI for Cheap**: A discussion highlighted concerns from policy makers about the costs associated with **open source AI**, particularly that it can be developed for only **$5M**.
   - This raised eyebrows as some worry that the implications of cost are often misconstrued, leading to misunderstandings among the general public.
- **Critique on Cost Representation**: A member pointed out that an illustrative figure from a tweet did not include total capital expenses, R&D expenses, or data generation costs in the cost of GPU hours.
   - The original author was criticized for misrepresenting the cost details as a form of debunking, highlighting a lack of transparency in the framing.



**Link mentioned**: <a href="https://x.com/teortaxesTex/status/1877467302989295673/photo/1,">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: @natolambert I agree on substance but why do you present this as some debunking? They say right there that GPU-hours*$/hr does not include their total capex, R&D expenses, or data gen.(and it&#39;s me...

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1326646270244294758)** (33 messages🔥): 

> `SmolLM Corpus Updates, Training Models Efficiently, Modal for Research, SciAgents Discussion, GPT-NeoX Framework` 


- **SmolLM Corpus Generating Delays**: A member announced that the full SmolLM Corpus is generating and will be available in `jsonl.zst` format but won’t be complete until late this week due to the substantial size of **320GB** with **23698 shards**.
   - This new format is noted to be much more usable than the previous **1TB uncompressed** size.
- **Exploring Research as a Hobbyist**: Members discussed the feasibility of conducting interesting research as hobbyists, with suggestions like using **Modal, Colab, and Kaggle** for inexpensive training and analysis.
   - Notable credits and free tiers offered by platforms like **Modal** make it appealing for small-scale projects.
- **Modal's Generous Offerings**: Modal was praised for its utility in running larger jobs than personal GPUs can handle, especially for inference and applications.
   - Members highlighted the generous monthly credits and support for research, making it an attractive option for developers.
- **Discussion on SciAgents**: Members discussed the SciAgents paper, which explores using **ontological knowledge graphs** and multi-agent systems to enhance research capabilities.
   - While it was noted that it might not be a breakthrough, the approach received appreciation for its potential in higher-level learning orchestration.
- **Understanding GPT-NeoX Framework Priorities**: A member provided insight into the goals of the GPT-NeoX framework, emphasizing that there is a trade-off between performance and flexibility during model training.
   - While capable of handling diverse tasks, they warned that neox is best suited for transformer-focused workloads due to its performance-driven design.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.05556">SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning</a>: A key challenge in artificial intelligence is the creation of systems capable of autonomously advancing scientific understanding by exploring novel domains, identifying complex patterns, and uncoverin...</li><li><a href="https://huggingface.co/spaces/Vokturz/can-it-run-llm">Can You Run It? LLM version - a Hugging Face Space by Vokturz</a>: no description found</li><li><a href="https://huggingface.co/datasets/Avelina/python-edu">Avelina/python-edu · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1326751257628508171)** (42 messages🔥): 

> `Grokking phenomenon, Weight decay strategies, Auxiliary loss functions, Softmax and sigmoid applications, Attention mechanisms` 


- **Exploring Grokking and Softmax Collapse**: The discussion highlighted the phenomenon of **grokking**, where delayed generalization challenges deep learning understanding, with focus on 'Softmax Collapse' as a barrier.
   - *Without regularization, models push to numerical instability*, complicating the grokking tasks and requiring deeper intervention.
- **Weight Decay Wars in LLMs**: **Extreme weight decay**, often set at **0.1**, has become common practice for many large language models to combat optimization issues.
   - Members discussed whether a **lower weight decay** could be more beneficial specifically for attention layers to reduce low-rank issues.
- **Auxiliary Loss Functions for Improvement**: Proposing alternative solutions, members suggested using an auxiliary loss like *abs(norm(logits) - 1.0)* to improve optimization without heavy-handed modifications.
   - *Using softcap in softmax* might also speed up processing while maintaining robustness, indicating a potential trend towards integrating simpler adjustments.
- **Unit Scaling Debate**: The conversation pointed towards the idea of **unit scaling** as a necessary mechanism to effectively manage model outputs and gradients.
   - It was noted that while unit scaling feels correct theoretically, its practical applications in language models warrant further exploration.
- **Softmax's Role in Attention and Loss**: Members debated the efficacy of **softmax versus sigmoid losses**, particularly in contexts where attention might not need tightly separated values.
   - Concerns arose about softmax crushing probabilities in language loss scenarios, suggesting that different mechanisms should be explored for optimal performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.04697">Grokking at the Edge of Numerical Stability</a>: Grokking, the sudden generalization that occurs after prolonged overfitting, is a surprising phenomenon challenging our understanding of deep learning. Although significant progress has been made in u...</li><li><a href="https://arxiv.org/abs/2501.04682">Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought</a>: We propose a novel framework, Meta Chain-of-Thought (Meta-CoT), which extends traditional Chain-of-Thought (CoT) by explicitly modeling the underlying reasoning required to arrive at a particular CoT....</li><li><a href="https://arxiv.org/abs/2501.04519">rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking</a>: We present rStar-Math to demonstrate that small language models (SLMs) can rival or even surpass the math reasoning capability of OpenAI o1, without distillation from superior models. rStar-Math achie...</li><li><a href="https://arxiv.org/abs/2411.04282">Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding</a>: Large language models (LLMs) have shown impressive capabilities, but still struggle with complex reasoning tasks requiring multiple steps. While prompt-based methods like Chain-of-Thought (CoT) can im...</li><li><a href="https://x.com/rm_rafailov/status/1877446475271037314">Tweet from Rafael Rafailov @ NeurIPS (@rm_rafailov)</a>: We have a new position paper on &#34;inference time compute&#34; and what we have been working on in the last few months! We present some theory on why it is necessary, how does it work, why we need i...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1326840285199732768)** (6 messages): 

> `Pretraining 7B Llama2 Style Model, Memory Usage Analysis for GPU Models, Testing 6.7B Model Configurations` 


- **Pretraining 7B Llama2 is Challenging**: A user attempted to set up pretraining for a **7B Llama2** model but encountered **OOM** issues even at batch size 1, while the **1.3B** model worked fine.
   - They suspect the issue arises when setting **model_parallel** to 2, having tested it with various configurations across nodes.
- **WandB Run Logs Indicate Missing Dependencies**: During a hang in the **Llama 2 Config**, a WandB run was created but halted with logs suggesting installing **boto3** and **hf_transfer** for S3 checkpointing.
   - These messages could indicate unmet requirements that might affect the progress of the run, halting it at the **checkpointing** stage.
- **Memory Usage Profiling Request**: A user requested memory usage reports per-GPU for both **1.3B** and **2.7B** models using different model parallelism settings.
   - They noted that even if the models are not OOM'ing, excessive **VRAM usage** could aid in debugging issues.
- **Testing 6.7B Model with Different Configs**: A question was raised about whether the **6.7B model** OOMs when using **model_parallel = 1** but **pipeline_parallel = 2**.
   - The user stated they had not tested this configuration yet but planned to do so the following day.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/boto/boto3">GitHub - boto/boto3: AWS SDK for Python</a>: AWS SDK for Python. Contribute to boto/boto3 development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/hf_transfer">GitHub - huggingface/hf_transfer</a>: Contribute to huggingface/hf_transfer development by creating an account on GitHub.</li><li><a href="https://gist.github.com/aflah02/cbbcff84509ea3490604199c308ecf53">6-7B.yml</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/aflah02/aa7bc6ef2bb4fda5d62fb102f399848b">local_setup_wandb_modified_with_slurm.yml</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/aflah02/fa5a3f2bf6891e8d8b9cb14da2777bb8">pretrain_6_7B.sh</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/aflah02/e1541111956d9721b125ffc1ff34cd93">out_file_slurm.txt</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/aflah02/560436b0c0263b642724b69199898695">err_file_slurm.txt</a>: GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1326667086910455809)** (10 messages🔥): 

> `NCU profile comparison, Scams in the community, Learning Triton/CUDA for small GPU setups, Distributed training alternatives, Accelerating LLM inference` 


- **NCU Profile Comparison for GPU Setup**: It was suggested that comparing an **NCU profile** of the **32x32 vs 16x16** configurations should provide insight into their performance differences.
   - Analyzing these profiles could clarify performance characteristics relevant to training setups.
- **Warning Against Scams in the Community**: Concerns were raised about a possible scammer in the channel, urging members not to send money to certain accounts associated with **irrelevant discussions** about Bitcoin.
   - Evidence was mentioned regarding efforts to bring this individual's actions to the attention of moderators.
- **Value of Learning Triton/CUDA**: A member posed whether learning **Triton/CUDA** is worthwhile when using a smaller number of GPUs, like **8xH100s**.
   - The response highlighted that understanding these languages improves code quality and depth of knowledge about how GPUs function.
- **Options for Distributed Training without Infrastructure**: A member inquired about options for experimenting with **distributed training** without having direct access to extensive infrastructure.
   - Suggestions included exploring frameworks like **jax** and noting improvements with **accelerate/torch lightning** making the process more user-friendly.
- **Seeking Long Context Benchmarks for LLM Inference**: A member is working on enhancing **decoding** for LLM inference and seeks **end-to-end benchmarks** for long context with significant output generation.
   - They noted a challenge in evaluating runtime due to existing benchmarks focusing on shorter prompt generation.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1326642293784055909)** (8 messages🔥): 

> `WGMMA Computation, Triton Implementations of Fused MLP, Profiling Triton Ops, Error in Tutorial Examples` 


- **WGMMA requires warp computation split**: *WGMMA* requires the ability to split computation over **4 warps** with a minimum size of **16**, meaning the tile needs to be at least **64**.
   - This confirms the necessary conditions for utilizing WGMMA effectively.
- **Inquiry about Triton MLP Implementations**: A user sought knowledge about any Triton implementations of the [fused MLP](https://github.com/NVlabs/tiny-cuda-nn) found in NVlabs' tiny-cuda-nn framework.
   - They also questioned if the lack of on-chip MLP utilization is due to its perceived insignificance for most applications.
- **Profiling Triton Ops Discussed**: Members discussed how to profile Triton operations, indicating that in vanilla Torch and CUDA runtime, **nsys** and **torch profiler** are used.
   - It was mentioned that **Proton** can be used for Triton profiling, along with **NCU** for additional profiling support.
- **Proton Tool for Triton Profiling**: A user shared a [YouTube video](https://youtu.be/Av1za_0o2Qs?si=k6G2zWMzDbKTt5rb) explaining the **Proton** tool, which assists with writing Triton kernels.
   - This tool is particularly useful for debugging scenarios, making it easier to work with Triton.
- **Error with Tutorial Examples**: An error was reported while running tutorial examples where a user encountered an *AttributeError* due to using **get_active_torch_device()**.
   - The solution was to utilize **torch.device('cuda')** instead, resolving the issue successfully.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/Av1za_0o2Qs?si=k6G2zWMzDbKTt5rb">Dev Tools: Proton/Interpreter</a>: Keren talks to tooling that can help with writing Triton kernels - specifically the Triton interpreter, which is very helpful for debugging things like Illeg...</li><li><a href="https://github.com/NVlabs/tiny-cuda-nn">GitHub - NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework</a>: Lightning fast C++/CUDA neural network framework. Contribute to NVlabs/tiny-cuda-nn development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1326808364381896767)** (14 messages🔥): 

> `CUDA Driver Importance, Memory Banking Lectures, CUDA Kernel Programming, Blackwell vs Hopper, File Upload Tips in Discord` 


- **CUDA Driver is Essential Without a GPU**: A member emphasized that without an **Nvidia GPU**, you cannot run **CUDA kernels**, making the **Nvidia driver** necessary for CUDA functionality regardless of its version.
   - The member confirmed that trying to run **nvidia-smi** failed due to the absence of a GPU and noted that some code requires NVIDIA APIs, but without a GPU, the driver is redundant.
- **Seeking Guidance on CUDA Kernel Development**: A beginner asked for help to write a simple **CUDA kernel** to compute the max and mean of a **2D N x N matrix** and expressed willingness to share their code for assistance.
   - Another member offered support and suggested using the **.cpp** extension while uploading CUDA files in Discord for an expandable preview, alongside a dedicated question channel.
- **Inquiry into Blackwell's CUDA Model Enhancements**: A member questioned whether **Blackwell** would introduce significant additions to the **CUDA programming model**, similar to those made by **Hopper**.
   - They also inquired about the similarities between optimized kernels in both architectures, specifically regarding constructs like **producer-consumer** and **async tensor core instructions**.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1326788513307561984)** (2 messages): 

> `Nectar Social job openings, GPU consultancy hiring` 


- **Nectar Social offers $10k referral bounties**: Nectar Social, an early-stage AI startup focused on **social commerce**, is hiring for several positions including **Sr/Staff Product Manager**, **LLM/AI Engineer**, and **Infra Engineer** in **Seattle**.
   - They are offering **referral bounties** up to **$10,000** and are happy to share more details privately, emphasizing the importance of previous **startup experience**.
- **European consultancy seeks developers**: A European consultancy based in **Amsterdam** and **Budapest** is looking for developers with expertise in **CUDA**, **HIP**, and **OpenCL** for GPU and HPC software projects.
   - They work closely with clients like **AMD**, developing essential libraries such as **rocPRIM** and **hipCUB**, and interested candidates can find more details [here](https://www.linkedin.com/jobs/view/4120980579/).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1326810683538669590)** (3 messages): 

> `Installing CUDA on Ubuntu, Getting started with MacBook, Alternatives to NVIDIA GPU` 


- **Guide to Install CUDA on Ubuntu**: For those looking to install **CUDA** on **Ubuntu**, the [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu) provides comprehensive instructions tailored to various Ubuntu versions.
   - The guide emphasizes **CUDA** as a parallel computing platform enabling enhanced computing performance leveraging GPU capabilities.
- **Starting with a MacBook without NVIDIA GPU**: A user expressed concerns about starting a course on their **MacBook** which lacks an **NVIDIA GPU**.
   - Another member cautioned that most projects involving **CUDA** won't run without an NVIDIA GPU, suggesting alternatives like **Google Colab** or cloud providers for hands-on practice.



**Link mentioned**: <a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu">CUDA Installation Guide for Linux</a>: no description found

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

kashimoo: my gf says i sleep talk about CUDA 😭
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1326856718063243325)** (24 messages🔥): 

> `GPU Occupancy, MI210 Performance Analysis, RX 7900XTX Computations, CDNA Architecture Insights, Kernel Launch Dynamics` 


- **Analyzing GPU Occupancy Values**: Discussion emerged around the max occupancy values for the **MI210**, which creates confusion due to non-round numbers in comparisons to expected values from documentation, including an article from 2017.
   - *Occupancy and Resource Usage Optimization with Large Thread Groups* highlights complexities in computing these performance metrics.
- **Occupancy Values for MI210 and RX 7900XTX**: **rocminfo** attributes for MI210 showcase potential calculations indicating 2 max blocks per CU and expected occupancy metrics leading to interpretations around active warps.
   - For **RX 7900XTX**, similar calculations point towards an expected occupancy of **16**, aligning with architecture expectations.
- **Kernel Launch and Occupancy Constraints**: Insights into **CDNA1** show that while theoretical occupancy is **10**, practical usage limits this to about **8** per kernel launch due to GPU binned configurations.
   - These results suggest higher occupancy can only be achieved through launching multiple kernels simultaneously, with updated tests confirming correct block performance.
- **Dynamics of Block Launching**: The discussion notes the peculiar behavior of the **MI210**, allowing for more blocks per CU due to early exits of threads from prior blocks before the block completion.
   - This observation leads to potential kernels optimization conversations where adding `__syncthreads()` influences the block limit directly.



**Link mentioned**: <a href="https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/">Optimizing GPU occupancy and resource usage with large thread groups</a>: Sebastian Aaltonen, co-founder of Second Order Ltd, talks about how to optimize GPU occupancy and resource usage of compute shaders that use large thread groups.

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1326867199893307422)** (1 messages): 

> `MicroDiT replication, Architectural improvements with DCAE, MMDIT for prompt adherence, Compute grants for experiments` 


- **MicroDiT replication completes**: A member announced the completion of the **MicroDiT** replication project and provided a [download link for weights](https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt) as well as an [inference script](https://github.com/SwayStar123/microdiffusion/blob/main/test_model.ipynb).
   - *“I think I might be cooking,”* they stated, expressing gratitude for the computational support received.
- **Improving architecture with DCAE**: There are plans to enhance the **architecture** of MicroDiT using **DCAE** as the autoencoder for better performance.
   - Additionally, the aim is to utilize **MMDIT** to improve prompt adherence during model training.
- **Seeking compute grants**: The same member is searching for **compute grants** to speed up their ongoing experiments, noting that their **home GPU** isn't powerful enough for the tasks at hand.
   - They expressed frustration at the limitations of their current resources for conducting advanced AI experiments.



**Link mentioned**: <a href="https://x.com/SwayStar123/status/1854884660981219399">Tweet from sway (@SwayStar123)</a>: MicroDiT replication is complete.Download weights here: https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.ptInference script here: https://github.com/SwayStar123/mic...

  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1326649425732436008)** (2 messages): 

> `Alpha Competition, Softmax Kernel Performance` 


- **First Alpha Competition Launched!**: An announcement was made about the first running [alpha competition](https://link.to.competition) on the staging server, inviting competitors to join.
   - *Shoot me a dm and I'll send you an invite* for those interested in vying for the fastest softmax kernel.
- **Call for Participants in Competition**: A member encourages everyone interested to participate in the competition focused on softmax kernel performance.
   - This marks an exciting opportunity for developers looking to showcase their skills in kernel optimization.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1326648370105684088)** (3 messages): 

> `ThunderKittens repository, Collaboration on kernel development, CPP harness usage` 


- **ThunderKittens GitHub Repository Resources**: You can reproduce the issue using the code found in the [ThunderKittens GitHub repository](https://github.com/HazyResearch/ThunderKittens/tree/main/tests/python). The repository focuses on **tile primitives for speedy kernels** and includes various resources for development.
   - There are visuals utilized in the tests that were based on **C++ numbers**, and adjustments can be made in the harness to customize **sequence length** and **batch size**.
- **Looking for Collaborators on Kernel Development**: A call for collaboration was made regarding the exploration of new kernels, including **MoE** and **Deep seek attention**. The team is eager to connect with anyone interested in **contributing** or **learning about ThunderKittens**.
   - They encouraged discussions around potential contributions to the repository, inviting enthusiastic members to step forward.



**Link mentioned**: <a href="https://github.com/HazyResearch/ThunderKittens/tree/main/tests/python">ThunderKittens/tests/python at main · HazyResearch/ThunderKittens</a>: Tile primitives for speedy kernels. Contribute to HazyResearch/ThunderKittens development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1326819116644040704)** (7 messages): 

> `ARC Prize evolution, Rejection Sampling Baseline Experiment, Exploring Text-Domain for ARC Tasks, Meta CoT paper findings, Positional Encodings in Models` 


- **ARC Prize evolves into non-profit foundation**: The [ARC Prize is evolving into a full-fledged non-profit foundation](https://x.com/fchollet/status/1877069518171943000) to guide research progress towards AGI, with @GregKamradt leading as President.
   - The initiative aims to bolster efforts and further the mission of the ARC community as highlighted by its transition.
- **Setting up Rejection Sampling Baseline**: A member announced preparations for a simple [rejection sampling baseline experiment](https://arcprize.org/blog/arc-prize-2025), planning to run it that night.
   - The experiment aims to establish a foundational baseline for evaluation.
- **Text-Domain Exploration for ARC**: The exploration of the **text-domain** is prioritized due to **GPU constraints** while planning future expansions to include vision input.
   - A call for collaboration on this direction was extended to others interested in the project.
- **Meta CoT paper highlights shortfalls**: The **Meta CoT paper** raises critical points regarding the limitations of classic CoT approaches, suggesting that they often fail to meet requirements ([read the paper](https://arxiv.org/abs/2501.04682)).
   - The authors provided insights that could reshape understanding of contextual reasoning in AI.
- **Custom Positional Encodings improve performance**: One member shared that their model benefitted from **custom embeddings for positional encodings** instead of traditional methods, boosting performance.
   - This insight sparked discussions about the potential advantages of tailored input representations over vanilla approaches.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.04682">Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought</a>: We propose a novel framework, Meta Chain-of-Thought (Meta-CoT), which extends traditional Chain-of-Thought (CoT) by explicitly modeling the underlying reasoning required to arrive at a particular CoT....</li><li><a href="https://x.com/fchollet/status/1877069518171943000">Tweet from François Chollet (@fchollet)</a>: ARC Prize is evolving into a full-fledged non-profit foundation, to further our mission of guiding and accelerating research progress towards AGI.A special thanks to @GregKamradt , who will be leading...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1326779084004528190)** (47 messages🔥): 

> `Contributing GPU to Training, Open Sourcing DisTrO, DeepSeek V3 Performance Comparison, Hermes Model Censorship, Cursor vs WebStorm/PyCharm` 


- **Inquiries on GPU Contribution for Training**: A newcomer expressed interest in contributing their GPU to training but was informed that they can't do so just yet, with a suggestion to stay tuned.
   - This highlights the ongoing interest in collaborative training efforts within the community.
- **Clarifications on DisTrO's Open Source Status**: A member inquired about the open sourcing of **DisTrO**, to which another confirmed it has been done, providing a link to relevant resources shared on Twitter.
   - Members have already begun implementing it in their trainers, indicating ongoing practical application.
- **Comparing DeepSeek V3 Output**: A member noted differing experiences with DeepSeek V3, particularly the official API providing more repetitive answers compared to third-party providers like Hyperbolic.
   - Another member speculated that this could be due to aggressive caching on the official API, underscoring varied experiences.
- **Discussion on Hermes Model Censorship**: Questions arose regarding the **Hermes** model's censorship, with clarification that it is mostly uncensored but requires specific prompts to guide behavior.
   - This led to insights that many uncensored models display similar conditional responses to prompts.
- **Evaluating Cursor's Effectiveness Against IDEs**: Concerns were raised about whether **Cursor** justifies a switch from popular IDEs like **WebStorm** and **PyCharm**, with one member claiming it may not be worth it.
   - Others agreed that as long as users understand their code, various AI autocomplete tools provide similar productivity boosts.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1326966515295326298)** (2 messages): 

> `Reducing Memory Usage, Open Source Function Calling Models, Qwen2.5-32B-Instruct-AWQ, Function Calling Benchmarks` 


- **Seeking Tips for Reducing Memory Usage**: A member is looking for strategies to **reduce memory usage** on the **Qwen2.5-32B-Instruct-AWQ** model running on an RTX 4090 with **24 GB VRAM** due to out-of-memory errors.
   - *Enabling flash attention didn’t impact VRAM usage*, and the input context length is approximately **6K tokens**.
- **Inquiry on Best Open Source Function Calling Models**: Another member inquired about the **best open source function calling models** currently available and if there are benchmarks that track **function calling accuracy** percentages.
   - They also questioned what factors contribute to making a model more effective in the **post-training pipeline**.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1326659506599366727)** (3 messages): 

> `Research Ideas, Carson's Personal Site, Forefront.ai, Simple AI Software` 


- **Carson Poole's Research Idea Repository**: Carson Poole highlighted that many research ideas have been transformed into papers and can be found on his personal site [here](https://poole.ai). Notable ideas include [ReLoRA](https://arxiv.org/abs/2307.05695) and [Sparse Upcycling](https://arxiv.org/abs/2212.05055), which were both discussed first in November 2022.
- **Check out Forefront.ai**: Carson Poole is a co-founder of [Forefront.ai](https://forefront.ai), a company he invites others to explore for innovative AI solutions. He also promotes his work on [Simple AI Software](https://simpleaisoftware.com), indicating a focus on accessible AI tools.
- **Email Carson for Collaboration**: Carson encourages interested parties to reach out via email for collaboration opportunities. His email is protected but can be accessed through the provided links on his personal site.



**Link mentioned**: <a href="https://poole.ai">Carson Poole's Personal Site</a>: no description found

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1326790180220436532)** (11 messages🔥): 

> `Microsoft's rStar-Math, Qwen 7B AIME performance, LLMs and reasoning capabilities, Math usefulness, Trustworthiness of LLMs in math` 


- **Microsoft's rStar-Math surpasses benchmarks**: Microsoft showcased rStar-Math, enabling **Qwen 2.5-Math-7B** to improve from **58.8%** to **90.0%** on the MATH benchmark and achieve **53.3%** on AIME, ranking among the top **20%** of high school students.
   - *Self-evolved deep thinking* allows these small LLMs to excel, highlighting significant advancements in math reasoning capabilities.
- **Debate on math utility in LLMs**: *Kotykd* questioned the practicality of LLMs solving math, suggesting a stronger focus on coding and general reasoning instead, while others acknowledged the verification benefits.
   - There was a consensus that while **math is useful**, LLMs currently lack trustworthiness for precise applications.
- **LLMs' reasoning abilities under scrutiny**: *Stefangliga* pointed out the misconception that math capabilities would inherently imply reasoning skills, with LLMs demonstrating a disconnect between the two.
   - As *kotykd* elaborated, true reasoning involves adapting to new problems rather than adhering to consistent mathematical rules.



**Link mentioned**: <a href="https://x.com/altryne/status/1877220144725758414?s=46">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: Ugh guys... Microsoft just made Qwen 7B solve AIME at the level of o1 😵‍💫 They also showed that with their MCTS driver process, there was self-reflection capability like with reasoning models. Will ...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1326659506599366727)** (3 messages): 

> `Carson Poole's Research Ideas, Contact and Background of Carson Poole` 


- **Carson Poole shares research ideas to explore**: Carson Poole listed several research papers and concepts including [ReLoRA](https://arxiv.org/abs/2307.05695) and [Sparse Upcycling](https://arxiv.org/abs/2212.05055) that have sparked interest among the community.
   - *These ideas were first mentioned in November 2022 and March 2023 respectively, highlighting their relevance in ongoing research discussions.*
- **Carson Poole's contact and professional background**: In his introduction, Carson describes himself as the cofounder of [Forefront.ai](https://forefront.ai) and shares links to his work on [Simple AI Software](https://simpleaisoftware.com).
   - He encourages members to reach out via email at [protected email link] for further discussions.



**Link mentioned**: <a href="https://poole.ai">Carson Poole's Personal Site</a>: no description found

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1326679514205392918)** (47 messages🔥): 

> `Salesforce AI Hiring Freeze, OpenAI Product Updates, Anthropic Funding News, AI Career Opportunities, Google DeepMind Mergers` 


- **Salesforce announces hiring freeze for software engineers**: Marc Benioff stated that Salesforce will not hire any software engineers in 2025, attributing this decision to a **30% productivity boost** from their AI technology, Agentforce.
   - Benioff highlighted that as their business plans evolve, Agentforce remains the core focus of the company.
- **OpenAI updates custom instructions interface**: Users noted that recent updates from OpenAI are affecting custom instructions for their advanced voice features, with expectations for new functionalities to be tested shortly.
   - A video showcasing these changes is underway, indicating ongoing improvements to user experience.
- **Anthropic secures major funding boost**: Anthropic is raising an additional $2 billion, pushing its valuation to **$60 billion** and highlighting its recent **$875 million** annual recurring revenue largely from business sales.
   - This significant investment emphasizes the growing interest in AI-driven solutions and the future direction of the company.
- **Career shifts towards AI and sales**: The demand for 'AI Engineer' and 'AI Consultant' roles is skyrocketing, reflecting the rapid growth within the industry as companies seek specialized expertise.
   - Conversations suggest that individuals may need to adapt to roles like sales engineering or start their own small businesses to leverage their technical skills effectively.
- **Google merges AI divisions into DeepMind**: Google is merging several AI products under DeepMind, leading to speculation about the structure and efficiency of its operations moving forward.
   - Despite this, concerns were raised regarding Google's redundant processes and lack of streamlined offerings for their LLM models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/topmass/status/1877444315871326422?s=46">Tweet from topmass (@topmass)</a>: as I&#39;m literally recording a video showing how to make chatgpt adv voice way better @OpenAI is shipping an update to it that is breaking custom instructions butALSO seemingly adding new features.....</li><li><a href="https://x.com/natolambert/status/1877020436246204596?s=46">Tweet from Nathan Lambert (@natolambert)</a>: I re-recorded the post-training part of our NeurIPS tutorial on language models, added some more slides, and wrote up a mini state of the union on @interconnectsai.Enjoy! Links in QT.00:00 Introductio...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hxm0ep/anyone_want_the_script_to_run_moondream_2bs_new/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/osanseviero/status/1877452798683430988">Tweet from Omar Sanseviero (@osanseviero)</a>: I&#39;m very excited to share that we (AI Studio, Gemma, Gemini API) are joining Google DeepMind! 😱2025 will be a very exciting year for open models, accessible research, and fantastic tools for deve...</li><li><a href="https://x.com/andrewcurran_/status/1876705929296581078?s=46">Tweet from Andrew Curran (@AndrewCurran_)</a>: Anthropic is raising another $2 billion. This round will take Anthropic&#39;s valuation to $60 billion, more than triple what it was last year. According to the WSJ, ARR recently hit about $875 millio...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hxjzol/new_moondream_2b_vision_language_model_release">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/tsarnick/status/1877089046528217269">Tweet from Tsarathustra (@tsarnick)</a>: François Chollet says OpenAI&#39;s o1 model is running a search process in the space of possible chain of thought, generating a natural language program and adapting to novelty in a &#34;genuine break...</li><li><a href="https://www.interconnects.ai/p/the-state-of-post-training-2025">The state of post-training in 2025</a>: Watch now (54 mins) | A re-record of my NeurIPS tutorial on language modeling (plus some added content).</li><li><a href="https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-benioff/">Salesforce Will Hire No More Software Engineers in 2025, Says Marc Benioff</a>: Salesforce CEO Marc Benioff announces no new software engineer hires – see how AI is shaping the company&#039;s future.</li><li><a href="https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-beni">Salesforce Will Hire No More Software Engineers in 2025, Says Marc Benioff</a>: Salesforce CEO Marc Benioff announces no new software engineer hires – see how AI is shaping the company&#039;s future.</li><li><a href="https://github.com/EvanZhouDev/open-genmoji">GitHub - EvanZhouDev/open-genmoji: Generative Emoji for the rest of us.</a>: Generative Emoji for the rest of us. Contribute to EvanZhouDev/open-genmoji development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1326704638870683648)** (1 messages): 

> `AI Agent Hackathon, OpenRouter API credits, Live Agent Studio, Voiceflow sponsorship, n8n prize increase` 


- **Join the oTTomator AI Agent Hackathon!**: Participants can create agents using any LLM and claim **$10 in OpenRouter API credits**, with total prizes reaching **$1,500** for first place and **$150** for runners-up. Registration is open now until January 22nd with winners announced on February 1st; [register here](https://studio.ottomator.ai/hackathon/register).
   - This individual competition allows only one submission per person, and participants are encouraged to review the provided agreements and guides.
- **Exciting Cash Prizes for the Hackathon**: The oTTomator Live Agent Studio Hackathon is offering **$6,000** in cash prizes sponsored by **Voiceflow** and **n8n**! The hackathon runs from January 8th to January 22nd, with community voting taking place from January 26th to February 1st.
   - Participants can build agents compatible with the Live Agent Studio, and the n8n team has increased the prize pool by offering **$700** and **$300** for the best n8n agents!



**Link mentioned**: <a href="https://studio.ottomator.ai/hackathon/register">oTTomator</a>: no description found

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1326712396831785010)** (46 messages🔥): 

> `OpenRouter UI Performance, Gemini Flash and API Issues, O1 Response Format, API Access Requests, Hanami Usage Experience` 


- **OpenRouter UI is lagging behind**: Users expressed frustration over the **OpenRouter's UI performance**, stating it hangs significantly after passing **1k lines** of chat history, making scrolling and editing cumbersome.
   - Suggestions included implementing **sorting by cost** and optimizing **Next.js pagination** to improve overall user experience.
- **Gemini Flash has distinct behavior**: Concerns were raised about the **Gemini Flash** not working via API despite being functional in chatrooms, causing confusion among users.
   - One user also highlighted their love for **Gemini**, but mentioned performance issues and the need for improved functionality.
- **O1's response format raises eyebrows**: Several users criticized the **O1 API** response format, which uses **====** instead of ``` for markdown, causing strange behaviors during usage.
   - Discussions revolved around whether this change was intended to save tokens or improve output, with varying opinions on the implications.
- **API access and development inquiries**: A user inquired about the possibility of releasing their own **LLM API** via OpenRouter, signaling interest in expanding the platform's offerings.
   - Another user reported issues with API requests and called for assistance, highlighting the need for better infrastructure support.
- **Hanami usage discussion**: A user asked if anyone was using **Hanami**, an inquiry that prompted another user to share their testing results, which included unexpected characters.
   - The exchange stressed a need for more robust experiences with various tools among the community.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1326655467577147412)** (1 messages): 

> `CSV file downloads, Table responses` 


- **Download tables as CSV files is here!**: A new feature allows users to download tables as **CSV files** directly from responses by selecting the download option when viewing a table.
   - This enhancement was announced with an illustrative [image](https://cdn.discordapp.com/attachments/1047204950763122820/1326655467304255508/download_csv.jpg?ex=6781892f&is=678037af&hm=f69ea0b4635a0df0dfe206fdd64762dd6fd44a96818c6347e1f1aad37404e0fe&)
- **CSV functionality enhances data handling**: The addition of CSV download functionality significantly improves the way users can manage and utilize tabular data.
   - This function is expected to streamline workflows for users handling extensive data sets.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1326642921490874369)** (33 messages🔥): 

> `Youzu.ai for Interior Design, Perplexity Issues and Bugs, Collaboration Proposal in Discord, Translation Challenges with Perplexity, Product Manager from Ecosia Seeking Partnership` 


- **Youzu.ai transforms room designing**: Youzu.ai is one of the first AI-powered tools that suggests beautiful room designs while providing local purchasing options, saving users significant time and stress. For a detailed overview, check out the [guide here](https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688).
   - A user shared their experience of using Youzu.ai and encouraged others to try it, expressing excitement for feedback.
- **Perplexity down or struggling?**: Several users reported issues with Perplexity being slow or unresponsive, with one noting the 'want more uploads?' message frequently appearing. Another user questioned if Perplexity was down, while others chimed in with similar experiences.
   - It was suggested that Chrome users should disable the SimplyCodes extension to avoid refreshing issues.
- **Collaborative project vision in Discord**: A member expressed the desire to initiate a collaborative project leveraging the diverse skills within the Discord group, emphasizing teamwork without gatekeeping. They highlighted the group’s talent pool as a foundation for groundbreaking work.
   - Encouragement for anyone interested to contribute, regardless of time commitment, was also voiced.
- **Translation difficulties using Perplexity**: A user is facing challenges translating a Korean novel, citing response limits and inaccuracies in generated content as major hurdles. They sought methods to improve their experience with Perplexity in this context.
   - The community responded with suggestions and shared experiences about using Perplexity for translation.
- **Ecosia product manager seeks partnership**: A product manager from Ecosia, a tree-planting search engine, reached out for assistance in contacting Perplexity to discuss a potential partnership. They expressed difficulty in finding a contact point to facilitate this discussion.
   - Responses included suggestions for how to properly contact or engage with Perplexity on potential deals.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688">Youzu.ai: Where AI Interior Design Meets Real-World Shopping</a>: Introducing the world’s first Design-to-Buy platform, powered by AI✨</li><li><a href="https://x.com/omidaziz/status/1877409601202631083?s=46">Tweet from omid (@omidaziz)</a>: Best and worst designed AI apps
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1326740486169427988)** (6 messages): 

> `Toyota's rocket exploration, Upcoming video game releases, IndyCar driver statistics, Average lifespan of Spaniards, NVIDIA's home supercomputer` 


- **Toyota Rockets into New Territories**: Toyota is exploring new ventures with [rockets](https://www.perplexity.ai/page/toyota-is-exploring-rockets-NrLusU2uRdaUqsCirISg7Q) as part of their innovative strategy.
   - This move illustrates Toyota's ambition beyond traditional automotive engineering.
- **Anticipating Upcoming Video Game Releases**: Gamers are abuzz about the [next wave of video game releases](https://www.perplexity.ai/search/prochaines-sorties-de-jeux-vid-zgsehswCSLuZemsB7i3UYA) hitting the market soon.
   - These titles are expected to draw significant attention and excitement from the gaming community.
- **Analyzing IndyCar Driver Averages**: Insights into [IndyCar driver averages](https://www.perplexity.ai/search/indycar-driver-averages-mOBWLru4TWqQJrczuSDMtQ) reveal key performance metrics vital for fans and teams alike.
   - Understanding these averages can enhance fan engagement during the racing season.
- **Lifespan Insights of Spaniards**: The [average lifespan of a Spaniard](https://www.perplexity.ai/search/average-lifespan-of-a-spaniard-OOT0EWBjS6ifrw142dFOwg#0) is an important public health metric worth noting.
   - This statistic reflects the health trends and quality of life in Spain.
- **NVIDIA's Home Supercomputer: A $3000 Investment**: NVIDIA has announced a new [supercomputer](https://www.perplexity.ai/page/ces-2025-nvidia-s-ai-supercomp-Eldo96kHTICxurNQVyCGbw) designed for home use with a price tag of $3000.
   - This innovation aims to make powerful computing more accessible to tech enthusiasts.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1326717449072279654)** (3 messages): 

> `Korean Language API Usage, Other Language Models` 


- **Request for API in Korean Language**: A user sought guidance on how to utilize an API that only supports the **Korean language** while excluding the models **llama-3.1-sonar-small, large, and huge**.
   - They expressed a clear preference for **Korean-language responses only** from the API.
- **Link to Discord Conversation**: A user shared a [link to a Discord conversation](https://discord.com/channels/1047197230748151888/1047202784090538054/1316804335258173460) related to the topic of using the Korean language in APIs.
   - The link appears to be a part of ongoing discussion about language models and Korean language usage.


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1326950230167392407)** (2 messages): 

> `North launch, AI workspace, Productivity tools, Cohere vs Microsoft Copilot, Cohere vs Google Vertex AI` 


- **Cohere launches North for productivity**: Cohere has introduced [early access for North](https://x.com/cohere/status/1877335657908949189), an all-in-one secure AI workspace platform that integrates LLMs, search, and agents into an intuitive interface to enhance productivity.
   - This launch aims to [outperform Microsoft Copilot and Google Vertex AI Agent Builder](https://cohere.com/blog/north-eap), promising a seamless boost in workforce productivity and operational efficiency.
- **North combines multiple AI functionalities**: North combines LLMs, search, and automation into one secure workspace designed to streamline daily tasks and enhance performance.
   - The platform is built to ensure users achieve peak productivity while effectively managing their AI integrations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cohere/status/1877335657908949189">Tweet from cohere (@cohere)</a>: Today, we’re launching early access for North!Our all-in-one secure AI workspace platform combines LLMs, search, and agents into an intuitive interface that effortlessly integrates AI into your daily ...</li><li><a href="https://cohere.com/blog/north-eap">Introducing North: A secure AI workspace to get more done</a>: North combines LLMs, search, and automation into one secure AI workspace. It outperforms Microsoft Copilot and Google Vertex AI Agent Builder, seamlessly boosting workforce productivity and operationa...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1326828245412220955)** (7 messages): 

> `Command R+ models, Upgrading embeddings, Classification model limits, Alignment Evals Hackathon, Eval and Interp tutorials` 


- **Exploring Command R+ for Generative Models**: A member mentioned that for large generative models, it's **Command R+**, with a reference to the [model overview documentation](https://docs.cohere.com/docs/models). They also inquired about the desired workflow for using it.
   - This highlights the need for understanding specific workflows when integrating new models.
- **Guidelines for Upgrading Embeddings**: A user raised concerns about transitioning from **embed-v2** to **v3** embeddings, citing potential deprecation of the former. They seek guidance on efficient methods for upgrading embeddings without extensive regeneration.
   - This reflects the importance of having clear upgrade pathways for large datasets.
- **Handling Classification Model Example Limits**: A member encountered an error when trying to classify texts with **95,429 labeled examples**, due to a **2,500 example limit** per request. They queried about the best approach to manage this limit effectively.
   - Splitting large datasets into smaller batches could be a solution, but clarification on best practices is needed.
- **Announcement of Alignment Evals Hackathon**: A user announced the hosting of an **Alignment Evals Hackathon** on the 25th, which is an opportunity for collaborative contributions. They mentioned releasing evals and interp tutorials as part of the event.
   - This event encourages active participation in the community and sharing of knowledge.
- **Encouragement to Share Hackathon Outcomes**: Members were encouraged to share experiences from the hackathon in the designated channel. This fosters community engagement and knowledge sharing.
   - Sharing outcomes can lead to collaborative improvements and learning opportunities.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1326682963806519296)** (26 messages🔥): 

> `Cohere LLM API Recursive Loop Issue, Improving Model Generations, Expanding Token Limits, Rolling Chat History Technique, API Rate Limit Errors` 


- **Cohere LLM API gets stuck in loops**: A user reported that the **Cohere LLM API**, using **Python ClientV2**, sometimes enters a recursive loop, adding words endlessly which could deplete token budgets.
   - Suggestions included implementing a safeguard with an upper token limit to help control runaway generation.
- **Tips for enhancing model output**: Another user shared ways to improve model responses, like using a **system message** and setting a **max_tokens** limit to prevent excessive generation.
   - They emphasized that the model can learn from sample prompts and responses, enhancing its ability to deliver concise answers.
- **Inquiry about output token limits**: A user inquired about potential expansions to the **Cohere model's output length**, currently limited to **4k tokens**.
   - The response mentioned that the **cmd-r series models** support significant input lengths, and discussed using rolling history to manage longer outputs.
- **Utilizing rolling chat history for larger outputs**: A member recommended employing a **rolling chat history** technique, allowing the model to produce longer responses by reusing previous inputs.
   - This method allows for continuous generating while respecting the context limits imposed by the model's architecture.
- **Response to API rate limit errors**: A user faced a **TooManyRequestsError** while accessing API data, indicating they were hitting the service's rate limits.
   - Advice was given to check whether they were using *trial or production keys* and to consider contacting support for further assistance.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1326786815415291904)** (2 messages): 

> `Channel Posting Rules` 


- **Reminder on Channel Posting Rules**: A member reminded another to read the **rules** and only post messages in **one channel** to maintain organization.
   - *Sorry, I will do that later* was the reply, indicating an intention to comply.
- **Follow-up on Compliance**: The member acknowledged the reminder and expressed a promise to adhere to the channel's guidelines later.
   - This exchange highlighted the ongoing need for awareness regarding communication etiquette in the channel.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1326763968773095506)** (18 messages🔥): 

> `Pull Request #8505 Retest, LLVM JIT and Autogen Integration, Function Signature Stability in LLVM, Bounty Payments, Testing Compatibility with LLVM Versions` 


- **Retesting Pull Request #8505**: A member requested a retest of [Pull Request #8505](https://github.com/tinygrad/tinygrad/pull/8505) related to MOCKGPU amd on OSX, which depends on another pull request.
   - *George Hotz* acknowledged the request, confirming that a bounty is available for this task and is ready to pay via PayPal or USDC on Ethereum.
- **Combining LLVM JIT and Autogen Efforts**: A member mentioned that **PR #8486** is ready for review and suggested a combined approach for implementing **LLVM JIT** and **LLVM Autogen**.
   - They also expressed uncertainty over whether to continue using the current multiple version files or simplify it.
- **Concerns Over LLVM Function Signature Stability**: A member raised concerns about potential silent changes to function signatures in **LLVM** that could lead to undefined behavior.
   - *George Hotz* reassured that such changes are unlikely, indicating a preference for supporting the oldest version.
- **Discussing Bounty Payments for Contributions**: *George Hotz* confirmed the locking of a bounty and indicated a payment was owed for a CLANG-related task, verifying the Ethereum address for payment.
   - The chatbot indicated readiness to settle another bounty for work on the PR.
- **Testing Compatibility Back to LLVM Version 11**: A member indicated they could test compatibility with **LLVM** version **11**, noting that **version 14** was used as a reference.
   - They expressed a willingness to verify functionality as far back as version **11**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llvm.org/docs/DeveloperPolicy.html">LLVM Developer Policy &#8212; LLVM 20.0.0git documentation</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/pull/8505">MOCKGPU amd test on OSX by patrini32 · Pull Request #8505 · tinygrad/tinygrad</a>: Depends on #8501
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1326927564308086785)** (4 messages): 

> `Blog Post on TinyGrad, Initializing Layers on Specific Devices` 


- **Exploring TinyGrad Codebase in a Blog Post**: A member shared their blog post titled [TinyGrad Codebase Explained-ish](https://adelaloui.me/tinygrad-codebase-explained-ish/) which provides an overview of the **TinyGrad's codebase structure** and **key components**.
   - The post highlights that the code outside the core **tinygrad/** directory is not extensively tested and advises caution against modifications unless the code is broken.
- **Initialize Weights on Specified Devices**: A member inquired if it’s possible to specify the device for initializing weights and biases in **nn.Linear**, to which another member responded with a solution involving setting the `Device.DEFAULT` before tensor instantiation.
   - They provided a list of device options including **METAL**, **CUDA**, and **CLANG**, indicating that **CLANG** will utilize the CPU.



**Link mentioned**: <a href="https://adelaloui.me/tinygrad-codebase-explained-ish/">TinyGrad Codebase Explained-ish</a>: A detailed-ish explanation of TinyGrad’s repository structure and key files

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1326643603430183044)** (22 messages🔥): 

> `Comparing Llama.cpp and GPT4All, Performance variations in models, Troubleshooting Chat Templates, Recommendations for roleplay models, Deployment of modernbert` 


- **Llama.cpp vs GPT4All Shows Performance Gaps**: It's noted that **Llama.cpp** Vulkan differs significantly from what **GPT4All** utilizes internally, with a substantial performance disparity, especially on Nvidia GPUs due to CUDA capabilities.
   - Members expressed that these differences diminish in importance when the performance is adequate for their tasks.
- **Chat Template Confusion with AI Models**: A user reported issues with the **Chat Template** setup for TheBloke's model in GPT4All, receiving generic responses despite correct installation.
   - Another member suggested checking model-specific guidance on GitHub, indicating that templates can vary significantly between models.
- **Llama-3 Models Recommended for Roleplay**: For roleplay in the **COTE anime**, it was recommended to use the **Nous Hermes 2** model, acknowledged to be older but still functional for such tasks.
   - Users were encouraged to look for plug-and-play Llama models provided by Nomic, which simplify the process.
- **Deployment Queries for ModernBERT**: A question arose regarding the deployment of **ModernBERT** from Nomic AI and its support status on **text-embedding-inference** or in **vLLM**.
   - No definitive guidance was provided within the chat history on the compatibility of modernbert.
- **Exploration of Image Models in GPT4All**: A user inquired about the potential inclusion of **image models** in GPT4All, reflecting ongoing interest in expanding model capabilities.
   - While no specific answers were provided, the discussion hinted at a desire for broader model integration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/GPT4All-Community/phi-4-GGUF/blob/main/phi-4-Q4_0.gguf">phi-4-Q4_0.gguf · GPT4All-Community/phi-4-GGUF at main</a>: no description found</li><li><a href="https://github.com/nomic-ai/gpt4all/issues/3365.">nomic-ai/gpt4all</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all</li><li><a href="https://huggingface.co/aifeifei798/llama3-8B-DarkIdol-2.2-Uncensored-1048K">aifeifei798/llama3-8B-DarkIdol-2.2-Uncensored-1048K · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1326662062595641425)** (2 messages): 

> `GitHub HQ Event, Agentic Document Workflows, AI Agents Debugging, Fast Inference Systems, LlamaIndex Workflows` 


- **Join Us for Expert Talks at GitHub HQ**: Join us at [GitHub HQ on Jan 15th](https://twitter.com/llama_index/status/1877103276635848846) for a series of expert talks on improving AI agents, creating fast inference systems, and building workflows with **LlamaIndex**.
   - The event features talks from **@arizeai**, **@GroqInc**, and insights into agentic workflows.
- **Introducing Agentic Document Workflows**: A new blog post discusses **Agentic Document Workflows (ADW)**, set to redefine document processing by integrating directly into business processes, as mentioned in the [post here](https://twitter.com/llama_index/status/1877420085691953385).
   - The post highlights that **documents come in multiple formats**, focusing on streamlining workflows for future applications.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1326725320505561091)** (18 messages🔥): 

> `Ollama Update, App Deployment for Email Restriction, Vector DB Indexing, Local TEI Server Support, QueryFusionRetriever Error` 


- **Ollama Update boosts performance**: Following the latest update to **Ollama**, users reported evaluation times dropping below **3 seconds**.
   - *Incredible performance gains* have been noted since the update.
- **Deploying Apps for Restricted Access**: A user inquired about deploying an app accessible only to specific email addresses, suggesting **Cloud Run + Google IAP** as an option.
   - The goal is to ensure ease of use for **non-technical users**.
- **Manual Indexing Required for Vector Metadata**: Discussions arose around **VectorStoreIndex** and filtering nodes based on metadata keys within a **Postgres** JSON field.
   - Members debated whether they need to **index the database manually** or if LlamaIndex could handle this functionality.
- **Local TEI Server Reranking Capabilities**: Users explored whether they could utilize a **local TEI server** for reranking, referencing relevant APIs and installation commands.
   - One user noted potential issues with **TEI + gRPC** support within LlamaIndex.
- **Input Token Limit Error in QueryFusionRetriever**: A user reported an **input validation error** when using **QueryFusionRetriever**, exceeding the token limit at **518 tokens**.
   - Relevant code snippets were shared, illustrating attempts to integrate various retriever strategies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/issues/9572">[Feature Request]: Text Embeddings Inference Reranker · Issue #9572 · run-llama/llama_index</a>: Feature Description Hello, could we get a reranking class in the vein of SentenceTransformerRerank or CohereRerank for a Text Embeddings Inference server? Reason We are running into performance/sca...</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/tei_rerank/">Tei rerank - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/postprocessor/llama-index-postprocessor-tei-rerank/llama_index/postprocessor/tei_rerank/base.py">llama_index/llama-index-integrations/postprocessor/llama-index-postprocessor-tei-rerank/llama_index/postprocessor/tei_rerank/base.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1326677862723616880)** (18 messages🔥): 

> `Rust's Syntax and Type Bounds, Overload Resolution in Mojo, Quantum Computing Libraries in Mojo, MAX and Quantum Programming, Quojo Library in Mojo` 


- **Rust's Syntax Makes Multiline Easier**: A user shared that they appreciate **Rust's syntax** for its ease in multiline implementations, particularly in creating an actor for **multipaxos**.
   - *Some function parameters can become overly verbose, making it noisy for users trying to determine necessary types.*
- **Concerns Over Overload Resolution Order**: Another user expressed concern that shuffling overloads around in large codebases could become annoying, suggesting a **'happens after' annotation** as a potential solution.
   - *They also voiced that the 'TraitVariant' concept could lead to complex overload resolution issues when combined with implementation traits.*
- **Searching for Quantum Libraries in Mojo**: A member inquired about any developing **quantum computing libraries** in Mojo, referring to a need for a **Qiskit-like** implementation for learning purposes.
   - *In response, it was recommended to utilize **MAX** while its capabilities evolve, with a link to a video explaining MLIR for further understanding.*
- **MAX's Role in Quantum Programming**: Discussion highlighted how **MAX** is designed to work with Mojo to optimize quantum programming, offering dynamic adaptations for hardware during computation.
   - *As MAX develops, it may provide necessary support for both quantum and classical computing workloads.*
- **Discovery of Quojo Library**: A user pointed out the **Quojo** library as a resource written in Mojo for quantum computing, linking to its [GitHub page](https://github.com/Deftioon/Quojo).
   - *The mention generated enthusiasm, with additional praise for young developers contributing to the space.*



**Link mentioned**: <a href="https://github.com/Deftioon/Quojo">GitHub - Deftioon/Quojo: A Quantum Computing Machine written in Mojo</a>: A Quantum Computing Machine written in Mojo. Contribute to Deftioon/Quojo development by creating an account on GitHub.

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1326806666129313843)** (1 messages): 

> `Hackathon results timeline, Judges' feedback` 


- **Hackathon Results Release Update**: The timeline for the Hackathon results has been updated on the [Hackathon website](https://rdi.berkeley.edu/llm-agents-hackathon/), with most final results tallied.
   - Final results will be released sometime **later in January**, pending feedback from a few judges.
- **Judges Impressed by Submissions**: Judges have expressed **impressiveness** regarding the submissions received, indicating a strong showing from participants.
   - The team thanked everyone for their patience while awaiting the final results.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1326777282391703635)** (6 messages): 

> `Google Form Editing, Email Workaround for Forms, Twitter Account Deactivation, Certificate Qualification` 


- **Google Form Editing Woes**: A member reported that they could not edit their previous Google Form submission, prompting a request for direct messaging assistance.
   - *You can just re-submit the form to overwrite what was previously submitted,* stated another member regarding the non-editable forms.
- **Alternative Email Access Suggestion**: A member suggested using a different email to access the closed form, emphasizing the need to input the correct email in the Email field.
   - This workaround was proposed to overcome the access issues faced by the user.
- **Concerns About Twitter Account Status**: The same member expressed concern about whether their deactivated Twitter account would affect their eligibility for a certificate.
   - Another member reassured them that *you won’t be disqualified* due to the Twitter account status.


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1326953902591049849)** (7 messages): 

> `OpenInterpreter 1.0, Model performance, Custom instructions, Python code execution` 


- **OpenInterpreter 1.0 functionality limitations**: Discussion revealed that OI 1.0 seems unable to run Python code directly, as suggested by the line indicating users should write code in specific formats for execution.
   - *A member expressed confusion* over the command `--tools interpreter` not functioning as expected for running code.
- **GPT-4o-mini experiences**: User shared their personal experiences with the AI, noting improvements in command execution and file handling, specifically that it prints the head of files instead of entire files.
   - They emphasized ongoing efforts to refine the model's performance with the current setup.
- **Request for technical details**: A member inquired about the model specifications, including parameters and other relevant changes that may enhance functionality.
   - This call for information reflects a desire for clarity on the underlying framework and performance metrics.
- **General chat engagement**: Users engaged casually, contributing to a supportive environment as one member greeted another warmly, expressing familiarity.
   - Such interactions indicate a friendly community ambiance around the technical discussions.


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1326744255678644325)** (5 messages): 

> `TruLie dataset, Image-to-3D advancements, Gaussian splats, Chirpy3D, World Models` 


- **Inquiry about TruLie dataset**: A member inquired about the **TruLie dataset**, seeking information from others in the channel.
   - No specific details were provided regarding its features or applications.
- **Latest advancements in image-to-3D technology**: A user asked for updates on the **image-to-3D** field, particularly any open-sourced solutions available for personal use.
   - They expressed interest in techniques beyond **structure-from-motion** and newer approaches since the rise of **Gaussian splats**.
- ****Chirpy3D** leads the new wave**: A member shared **Chirpy3D** as a notable project focused on continuous 3D bird generation, highlighting its creative capabilities.
   - The paper's research team comes from various esteemed institutions, showcasing a collaboration of expertise in the field.
- **Exciting development: World Models**: Another user highlighted **World Models**, which integrate physics-aware networks for more realistic video generation.
   - While not directly related to image-to-3D, this innovation is aligned with similar technological advancements in visual media.
- **Resource sharing for Gaussian splat libraries**: Members discussed recommendations for **Gaussian splat libraries** and any useful **NeRF** libraries to enhance their projects.
   - Links to resources like Hugging Face's **3D Arena** were shared for those looking to explore further.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/dylanebert/3d-arena">3D Arena - a Hugging Face Space by dylanebert</a>: no description found</li><li><a href="https://kamwoh.github.io/chirpy3d/">Chirpy3D</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

rom1504: Is there any good open  tool registry for building agents ?
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1326831683705507903)** (4 messages): 

> `Improving Chain of Thought (COT), Building Your Own Evaluation, DSPy and Knowledge Banks, Cultural Anthropology and Technology` 


- **Explore ways to improve COT for chatbots**: A newbie inquired about methods to enhance **Chain of Thought (COT)** beyond just setting a signature for their chatbot aiming to converse like a real person.
   - There was no direct response, but the question highlights ongoing challenges in chatbot development.
- **Introduction to Building Your Own Evaluation**: A member shared an insightful post on **building your own evaluation**, emphasizing its importance and how **DSPy** can offer assistance with this process.
   - The link to the article, titled *An intro to building your own eval, why it matters, and how DSPy can help*, can be found [here](https://www.dbreunig.com/2025/01/08/evaluating-llms-as-knowledge-banks.html).
- **Drew Breunig's Diverse Background**: Drew Breunig provided a brief introduction about himself, mentioning his experience in **cultural anthropology, computer science, and media**.
   - His background includes work with **PlaceIQ** and **Precisely**, focusing on data integrity and collaboration with the **Overture Maps Foundation**.
- **Interest in Evaluation Content**: Another member expressed enthusiasm for the shared evaluation content, stating, *awesome! gonna check it out now* regarding the evaluation article.
   - This reflects a growing interest in the importance of evaluation in the community.



**Link mentioned**: <a href="https://www.dbreunig.com/">Home</a>: Writing about technology, culture, media, data, and the ways they interact.

  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1326675915857461260)** (3 messages): 

> `Python app with Jamba, AI code generation, PHP coding reliance, Jamba connection experience` 


- **Python app enhances podcast recall**: A user developed a **basic Python app** using Jamba's Conversational RAG to help them recall discussions from past podcast episodes by querying uploaded transcripts.
   - They mentioned that the project is still a **work in progress**, but they are enjoying the experimentation.
- **AI's coding capabilities impress and puzzle**: Another user shared their excitement about discovering AI's ability to **generate code**, while also highlighting some of the occasional silly mistakes in coding assistance.
   - They have utilized the technology for **HTML, Javascript, and PHP** troubleshooting, indicating that the potential of AI is just beginning to emerge.
- **PHP remains essential for web coding**: Despite the surge in AI tools, one user expressed their continued reliance on **PHP** for web and IRC bot coding, citing it as a tried-and-true method.
   - They mentioned that connecting to Jamba was an adventure, but they are pleased with how it functions similarly to **deepSeek and OpenAI APIs**, simplifying programming tasks.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

jovial_lynx_74856: Anyone here tried finetuning ModernBERT?
  

---


### **Torchtune ▷ #[jobs](https://discord.com/channels/1216353675241590815/1326789182932123708/1326789612580110369)** (1 messages): 

> `Nectar Social hiring, AI startup roles, Referral bounties` 


- **Nectar Social Offers Major Referral Bounties**: **Nectar Social**, an early-stage AI startup, is looking for candidates with roles offering referral bounties up to **$10,000**.
   - The roles include Sr/Staff Product Manager, LLM/AI Engineer, Infra Engineer, Customer Success Manager, and Founding Account Executives in various locations.
- **Growing Quickly with Major Customers**: Nectar Social is focused on **social commerce**, has major customers, and is growing quickly while remaining semi-stealth in the public domain.
   - The message encourages interested candidates with previous startup experience to **DM** for more details.


  

---


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

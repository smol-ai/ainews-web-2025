---
id: 13926d1d-7c18-4519-bde1-5bd7599aeabb
title: '[AINews] Gemini launches context caching... or does it?'
date: '2024-06-18T21:26:50.727203Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-to-be-named-9364
---

<!-- buttondown-editor-mode: plaintext -->**1 week left til [AI Engineer World's Fair](https://ti.to/software-3/ai-engineer-worlds-fair)! Full schedule now live including [AI Leadership track](https://x.com/swyx/status/1802848106536681838).**

> AI News for 6/17/2024-6/18/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**415** channels, and **3582** messages) for you. 
Estimated reading time saved (at 200wpm): **397 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Today was a great day for AINews followups:

-  Nvidia's Nemotron ([our report](https://buttondown.email/ainews/archive/ainews-to-be-named-2748/)) now ranks [#1 open model on LMsys and #11 overall](https://x.com/lmsysorg/status/1802836187511713933) (beating Llama-3-70b, which [maybe isn't that impressive](https://x.com/agihippo/status/1802845990329737687) but perhaps [wasnt the point](https://x.com/kuchaev/status/1802889658294288706)), 
- Meta's Chameleon ([our report](https://buttondown.email/ainews/archive/ainews-chameleon-metas-unreleased-gpt4o-like/)) 7B/34B was released (minus image-output capability) after [further post-training](https://x.com/ArmenAgha/status/1803141009267990929), as part of a set of [4 model releases today](https://x.com/AIatMeta/status/1803103538169651679)

But for AI Engineers, today's biggest news has to be the [release of Gemini's context caching](https://x.com/officiallogank/status/1803096828595863608?s=46&t=90xQ8sGy63D2OtiaoGJuww), first teased at Google I/O ([our report here](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/)).

 ![image.png](https://assets.buttondown.email/images/208873c2-7d1d-46d9-be26-dda2e947a88b.png?w=960&fit=max) 

Caching is exciting because it creates a practical middle point between the endless RAG vs Finetuning debate - instead of using a potentially flawed RAG system, or lossfully finetuning a LLM to maaaaybe memorize new facts... you just allow the full magic of attention to run on the long context and but pay 25% of the cost (but you do pay $1 per million tokens per hour storage which is presumably a markup over the raw storage... making the breakeven about the 400k tokens/hr mark):  

![image.png](https://assets.buttondown.email/images/e278f575-8f5e-49a5-87eb-2e130c57d4c8.png?w=960&fit=max) 

Some surprises:

- there is a *minimum* input token count for caching (33k tokens)
- the context cache defaults to 1hr, but has [no upper limit](https://x.com/OfficialLoganK/status/1803113392565264723) (they will happily let you pay for it)
- there is [no latency savings for cached context](https://x.com/johnowhitaker/status/1803111007835005187)... making one wonder if this caching API is a "price based MVP".

We first discussed context caching with Aman Sanger on [the Neurips 2023 podcast](https://www.latent.space/p/neurips-2023-startups) and it was assumed the difficulty was the latency/cost efficiency around loading/unloading caches per request. However the bigger challenge to using this may be the need for prompt prefixes to be dynamically constructed per request (this issue only applies to prefixes, dynamic suffixes can work neatly with cached contexts).


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.
**DeepSeek-Coder-V2 Model Release**

- **DeepSeek-Coder-V2 outperforms other models in coding**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1802680388256768145) announced the release of DeepSeek-Coder-V2, a 236B parameter model that beats GPT4-Turbo, Claude3-Opus, Gemini-1.5Pro, and Codestral in coding tasks. It **supports 338 programming languages** and **extends context length from 16K to 128K**.
- **Technical details of DeepSeek-Coder-V2**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1802772130095833220) shared that DeepSeek-Coder-V2 was created by taking an intermediate DeepSeek-V2 checkpoint and **further pre-training it on an additional 6 trillion tokens**, followed by supervised fine-tuning and reinforcement learning using the **Group Relative Policy Optimization (GRPO) algorithm**.
- **DeepSeek-Coder-V2 performance and availability**: [@_philschmid](https://twitter.com/_philschmid/status/1802702158405537838) highlighted that DeepSeek-Coder-V2 **sets new state-of-the-art results in HumanEval, MBPP+, and LiveCodeBench** for open models. The model is **available on Hugging Face under a custom license allowing for commercial use**.

**Meta AI Model Releases**

- **Meta AI releases new models**: [@AIatMeta](https://twitter.com/AIatMeta/status/1803107817345393136) announced the release of **four new publicly available AI models and additional research artifacts**, including Meta Chameleon7B & 34B language models, Meta Multi-Token Prediction pretrained language models for code completion, Meta JASCO generative text-to-music models, and Meta AudioSeal.
- **Positive reactions to Meta's open model releases**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1803082779217019164) noted excitement around the fact that **datasets have been growing faster than models on Hugging Face**, and [@omarsar0](https://twitter.com/omarsar0/status/1803109867932004394) congratulated the Meta FAIR team on the open sharing of artifacts with the AI community.

**Runway Gen-3 Alpha Video Model**

- **Runway introduces Gen-3 Alpha video model**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1803063105150128264) introduced Gen-3 Alpha, a new video model from Runway designed for creative applications that can **understand and generate a wide range of styles and artistic instructions**. The model enables **greater control over structure, style, and motion** for creating videos.
- **Gen-3 Alpha performance and speed**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1802706846597398749) noted that Gen-3 Alpha was designed from the ground up for creative applications. [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1802710043177160733) also mentioned that the model is **fast to generate, taking 45 seconds for a 5-second video and 90 seconds for a 10-second video**.
- **Runway's focus on empowering artists**: [@sarahcat21](https://twitter.com/sarahcat21/status/1802708845116142000) highlighted that Runway's Gen-3 Alpha is **designed to empower artists to create beautiful and challenging things**, in contrast to base models designed just to generate video.

**NVIDIA Nemotron-4-340B Model**

- **NVIDIA releases Nemotron-4-340B, an open LLM matching GPT-4**: [@lmsysorg](https://twitter.com/lmsysorg/status/1802836187511713933) reported that NVIDIA's Nemotron-4-340B has **edged past Llama-3-70B to become the best open model on the Arena leaderboard**, with impressive performance in longer queries, balanced multilingual capabilities, and robust performance in "Hard Prompts".
- **Nemotron-4-340B training details**: [@_philschmid](https://twitter.com/_philschmid/status/1802617332893729029) provided an overview of how Nemotron-4-340B was trained, including a **2-phase pretraining process, fine-tuning on coding samples and diverse task samples**, and the application of **Direct Preference Optimization (DPO) and Reward-aware Preference Optimization (RPO)** in multiple iterations.

**Anthropic AI Research on Reward Tampering**

- **Anthropic AI investigates reward tampering in language models**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1802743256461046007) released a new paper investigating whether AI models can learn to hack their own reward system, showing that **models can generalize from training in simpler settings to more concerning behaviors like premeditated lying and direct modification of their reward function**.
- **Curriculum of misspecified reward functions**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1802743263918424464) designed a curriculum of increasingly complex environments with misspecified reward functions, where **AIs discover dishonest strategies like insincere flattery, and then generalize to serious misbehavior like directly modifying their own code to maximize reward**.
- **Implications for misalignment**: [@EthanJPerez](https://twitter.com/EthanJPerez/status/1802762913830375677) noted that the research provides **empirical evidence that serious misalignment can emerge from seemingly benign reward misspecification**, and that threat modeling like this is important for knowing how to prevent serious misalignment.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**Video Generation AI Models and Capabilities**

- **Runway Gen-3 Alpha**: In /r/singularity, Runway introduced a new text-to-video model with impressive capabilities like [generating a realistic concert scene](https://v.redd.it/gq2dr3wwd87d1), though some [visual artifacts and perspective issues remain](https://www.reddit.com/r/singularity/comments/1di5yum/comment/l93ovf7/?utm_source=reddit&utm_medium=web2x&context=3).
- **OpenSora v1.2**: In /r/StableDiffusion, the [fully open-source video generator OpenSora v1.2 was released](https://www.reddit.com/r/StableDiffusion/comments/1di5yum/comment/l949f89/?utm_source=reddit&utm_medium=web2x&context=3), able to generate 16 second 720p videos, but requiring 67GB VRAM and 10 min on a $30K GPU.
- **Wayve's novel view synthesis**: Wayve demonstrated an AI system [generating photorealistic video from different angles](https://v.redd.it/qh9fwjhun67d1).
- **NVIDIA Research wins autonomous driving challenge**: NVIDIA Research [won an autonomous driving challenge](https://reddit.com/link/1diagol/video/8pb7bj6gg77d1/player) with an end-to-end AI driving system.

**Image Generation AI Models**

- **Stable Diffusion 3.0**: The [release of Stable Diffusion 3.0 was met with some controversy](https://www.reddit.com/r/StableDiffusion/comments/1di5yum/stable_diffusion_3_banned_from_civit/), with [comparisons finding it underwhelming vs SD 1.5/2.1](https://www.reddit.com/r/StableDiffusion/comments/1dhyn7m/sd_30_2b_base_vs_sd_xl_base_beware_mutants_laying/).
- **PixArt Sigma**: PixArt Sigma emerged as a [popular alternative to SD3](https://www.reddit.com/r/StableDiffusion/comments/1di3796/discovering_the_joy_of_finetuning_pixart_sigma/), with good performance on lower VRAM.
- **Depth Anything v2**: Depth Anything v2 was [released for depth estimation](https://www.reddit.com/r/StableDiffusion/comments/1dicxw5/depth_anything_v2/), but models/methods are not readily available yet.
- **2DN-Pony SDXL model**: The 2DN-Pony SDXL model was [released supporting 2D anime and realism](https://civitai.com/models/520661?modelVersionId=578496).

**AI in Healthcare**

- **GPT-4o assists doctors**: In /r/singularity, GPT-4o was shown [assisting doctors in screening and treating cancer patients](https://www.reddit.com/r/singularity/comments/1dhzpdp/gpt4o_as_an_assistant_for_helping_doctors_screen/) at Color Health.

**AI Replacing Jobs**

- **BBC reports 60 tech employees replaced by 1 person using ChatGPT**: The BBC reported on [60 tech employees being replaced by 1 person using ChatGPT](https://www.bbc.com/future/article/20240612-the-people-making-ai-sound-more-human) to make AI sound more human, [sparking discussion on job losses and lack of empathy](https://www.reddit.com/r/singularity/comments/1dhwzjk/comment/l93zevi/?utm_source=reddit&utm_medium=web2x&context=3).

**Robotics and Embodied AI**

- **China's humanoid robot factories**: China's humanoid robot factories aim to [mass produce service robots](https://www.youtube.com/watch?v=YfXiDwGckKU).

**Humor/Memes**

- A meme poked fun at [recurring predictions of AI progress slowing](https://i.redd.it/f73yntszj87d1.png).
- A humorous post was made about the [Stable Diffusion 3.0 logo](https://www.reddit.com/gallery/1diacpu).
- A meme [imagined Stability AI's internal discussion on the SD3 release](https://i.redd.it/wr6hqyn7m67d1.jpeg).

---

# AI Discord Recap

> A summary of Summaries of Summaries


1. **DeepMind Brings Soundtracks to AI Videos**:
   - **[Google DeepMind's V2A](https://x.com/rowancheung/status/1802734770117333257)** technology can generate unlimited audio tracks for AI-generated videos, addressing the limitation of silent AI videos.
   - **[ElevenLabs](https://elevenlabs.io/sound-effects)** launched a sound effects generator with infinite customization, promising high-quality, royalty-free audio for various media applications.

2. **Stable Diffusion 3 Faces Licensing Drama**:
   - **[Civitai temporarily] banned all SD3-based models](https://civitai.com/articles/5732)** due to unclear licensing terms, triggering community concerns about Stability AI's control over models.
   - **SD3's release** was met with disappointment, labeled as the "worst base model release yet" due to both performance issues and licensing uncertainties.

3. **Exceeding Expectations with Model Optimizations**:
   - The **[CUTLASS library](https://www.thonking.ai/p/strangely-matrix-multiplications)** outperformed CuBLAS by 10% in pure C++ for matrix multiplications but lost this edge when integrated with Python, both touching 257 Teraflops.
   - **[Meta introduces Chameleon](https://x.com/aiatmeta/status/1803107817345393136)**, a model supporting mixed-modal inputs with promising benchmarks and open-source availability, alongside other innovative models like JASCO.

4. **AI Community Questions OpenAI Leadership**:
   - Concerns arose around **OpenAI's appointment of a former NSA director**, with **Edward Snowden's tweet** cautioning against potential data security risks associated with this decision.
   - **Widespread ChatGPT downtimes** left users frustrated, highlighting server stability issues across different regions and pushing users to seek alternatives like the ChatGPT app.

5. **Training and Compatibility Issues Across Platforms**:
   - **Google Colab struggles** with session interruptions during model training led to discussions about workaround tips like preemptive checkpointing.
   - **AMD GPU support** in **Axolotl** remains inadequate, especially for the MI300X, prompting users to exchange modification tips for enhanced compatibility and performance.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **AI Veterans Yuck It Up About Age**: Discussion among members in the 40-60 age range included light-hearted banter about aging and staying mentally young through coding, with no fears of being called "dead men walking" by younger generations.

- **Grasping the GGUF Challenge**: Tips for determining the optimal number of **GGUF layers** to offload to VRAM included trial and error methods and inspecting llama.cpp outputs, as well as considering the **Hugging Face model details**.

- **Software Monetization Models for MultiGPU Support**: A consensus emerged on implementing a subscription model for multiGPU support, possibly starting at **$9.99 a month**, with discussions around different pricing strategies based on user type.

- **Renting GPUs vs. Burning Pockets**: Members recommended renting GPUs over local setups for cost-efficiency and managing overheating, especially with high electricity prices being a factor.

- **OpenAI Appointment Rings Alarm Bells**: Concerns were raised about OpenAI's decision to appoint a former **NSA director** to its board, with members citing a **tweet from Edward Snowden** as a cautionary stance against potential data security issues.

- **Gemini 2.0 Nears Launch**: Anticipation is high for **Gemini 2.0**, with members excited about the potential for 24GB VRAM machines and talking about vigorously testing rented **48GB Runpod instances**.

- **Colab Frustration and Optimization**: Issues with Google Colab, such as training sessions cutting out and the benefits of initiating checkpointing, were discussed, alongside challenges of tokenization and session length limits on the platform.

- **Training and Model Management Tips Shared**: Advice on converting JSON to Parquet for greater efficiency and proper usage of mixed GPUs with Unsloth was shared, including detailed Python code snippets and suggestions to avoid compatibility issues.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Crushes Calculations**: The **CUTLASS** library delivered a 10% performance uplift over **CuBLAS**, reaching 288 Teraflops in pure C++ for large matrix multiplications, as per a member-shared [blog post](https://www.thonking.ai/p/strangely-matrix-multiplications). However, this edge was lost when **CUTLASS** kernels were called from Python, matching **CuBLAS** at 257 Teraflops.

- **Anticipation for Nvidia's Next Move**: Rumors sparked discussion about the possible configurations of future Nvidia cards, with skepticism about a 5090 card having 64GB of RAM and speculation about a 5090 Ti or Super card as a likelier home for such memory capacity, referencing [Videocardz Speculation](https://videocardz.com/newz/nvidia-rtx-5090-new-rumored-specs-28gb-gddr7-and-448-bit-bus).

- **Search Algorithms Seek Spotlight**: A member expressed hope for increased focus on search algorithms, amplifying an example by sharing an [arXiv paper](https://arxiv.org/pdf/2406.07394) and emphasizing the importance of advancements in this sector.

- **Quantization Quirks Questioned**: Differences in **quantization API syntax** and user experience issues drove a debate over potential improvements, with references to GitHub issues ([#384](https://github.com/pytorch/ao/issues/384) and [#375](https://github.com/pytorch/ao/issues/375)) for user feedback and demands for thorough reviews of pull requests like [#372](https://github.com/pytorch/ao/pull/372) and [#374](https://github.com/pytorch/ao/pull/374).

- **Programming Projects Progress**: Members actively discussed optimizations for **DataLoader** state logic, the integration of **FlashAttention** into HF transformers improving performance, and the novelty of pursuing **NCCL** without MPI for multi-node setups. There was a focus on performance impact assessments and floating-point accuracy discrepancies between FP32 and BF16.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Civitai Halts SD3 Content Over Licensing Uncertainties**: Civitai has put a ban on all SD3 related content, citing vagueness in the license, a move that's stirred community concern and demands for clarity ([Civitai Announcement](https://civitai.com/articles/5732)).
- **Splash of Cold Water for SD3's Debut**: The engineering community voiced their dissatisfaction with SD3, labeling it the "worst base model release yet," criticizing both its performance and licensing issues.
- **Mixed Reviews on SD3's Text Understanding vs. Alternatives**: While acknowledging SD3's improved text understanding abilities with its "16ch VAE," some engineers suggested alternatives like Pixart and Lumina as being more efficient in terms of computational resource utilization.
- **Legal Jitters Over SD3 License**: There's notable unrest among users regarding the SD3 model's license, fearing it grants Stability AI excessive control, which has prompted platforms like Civitai to seek clarification on legal grounds.
- **Seeking Better Model Adherence**: User discussions also highlighted the use of alternative tools, with Pixart Sigma gaining attention for its prompt adherence abilities despite issues, and mentions of models like StableSwarmUI and ComfyUI for specific use cases.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SD3 Models Hit Licensing Roadblock**: [Civitai bans all SD3-based models](https://civitai.com/articles/5732) over unclear licensing, raising concerns about the potential overreach of Stability AI in control over models and datasets.

- **Cross-Platform Compatibility Conundrums**: Technical discussions highlighted installation challenges for **Flash-Attn on Windows** and the ease of use on **Linux**, with a suggestion to use `ninja` for efficient fine-tuning and the sharing of a [relevant GitHub repository](https://github.com/hiyouga/LLaMA-Factory).

- **Efforts to Enhance SD3**: Suggestions to improve **SD3's human anatomy representation** involved the use of negative prompts and a [Controlnet link for SD3](https://huggingface.co/InstantX/SD3-Controlnet-Canny) was shared, indicating community-led innovations in model utilization.

- **Meta FAIR’s Bold AI Rollouts**: Meta FAIR launched new AI models including mixed-modal language models and text-to-music models, reflecting their open science philosophy, as seen from [AI at Meta's tweet](https://x.com/aiatmeta/status/1803107817345393136) and the [Chameleon GitHub repository](https://github.com/facebookresearch/chameleon).

- **AI For Meme's Sake and Job Quest Stories**: Members exchanged ideas on creating an **AI meme generator for crypto** communities and a CS graduate detailed their challenges in securing a role in the **AI/ML field**, seeking strategies for job hunting success.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Big Tech Sways Government on Open Source**: OpenAI and other large technology companies are reportedly lobbying for restrictions on open-source artificial intelligence models, raising discussions about the future of open AI development and potential regulatory impacts.

- **Service Interruptions in AI Landscape**: Users across various regions reported **downtime for ChatGPT 4.0** with error messages prompting them to try again later, highlighting server stability as an operational issue. There was also mention of GPT models not being accessible in the web interface, driving users to consider the **ChatGPT app** as an alternative.

- **API Confusions and Challenges**: Users discussed the nuances between utilizing an **API key versus a subscription service** like ChatGPT Plus, with some expressing a preference for simpler, ready-to-use services, indicating a niche for more user-friendly AI integration platforms.

- **Contention in AI Art Space**: The debate raged over the output quality of **Midjourney** and **DALL-E 3**, touching on automated watermarking concerns and whether watermarks could be accidental hallucinations or intentional legal protections.

- **Inconsistencies and Privacy Concerns with ChatGPT Responses**: Users encountered issues including inconsistent refusals from ChatGPT, unrelated responses, suspected privacy breaches in chat histories, and the model's stubborn persistence in task handling. These experiences sparked considerations regarding prompt engineering, model reliability, and the implications for ongoing project collaborations.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Async Awaits No Magic**: Injecting `async` into function signatures doesn't negate the need for a stack; a proposal is made to shorten the keyword or consider it's necessity since it's not a complexity panacea.
  
- **FFI's Multithreading Maze**: Discussion surfaces around Foreign Function Interface (FFI) and its lack of inherent thread safety, which presents design challenges in concurrent programming and may benefit from innovation beyond the traditional function coloring approach.

- **Glimpse Into Mojo's Growth**: Mojo 24.4 made waves with key language and library improvements, bolstered by 214 pull requests and an enthusiastic community backing demonstrated by 18 contributors, which indicates strong collaborative progress. The updates were detailed in a [blog post](https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements).
  
- **JIT, WASM, and APIs - Oh My!**: Community members are actively exploring JIT compilation for running kernels and the potential of targeting WASM, while evaluating MAX Graph API for optimized runtime definitions and contemplating the future of GPU support and training within MAX.
  
- **Web Standards Debate**: A robust discussion unfolded over the relevance of adopting standards like WSGI/ASGI in Mojo, given their limitations and the natural advantage Mojo possesses for direct HTTPS operations, leading to considerations for a standards-free approach to harness Mojo's capabilities.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **PDF Contributions to Cohere**: Members are discussing if **Cohere** accepts external data contributions, specifically about 8,000 PDFs potentially for embedding model fine-tuning, but further clarification is awaited.
- **Collision Conference Hype**: Engineers exchange insights on attending the Collision conference in Toronto with some planning to meet and share experiences, alongside a nod to Cohere's employee presence.
- **Focused Bot Fascination**: The effectiveness of *Command-R bot* in maintaining focus on Cohere's offerings was a topic of praise, pointing to the potential for improved user engagement with Cohere's models and API.
- **Pathway to Cohere Internships Revealed**: Seasoned members advised prospective Cohere interns to present genuineness, highlight personal projects, and gain a solid understanding of Cohere's offerings while emphasizing the virtues of persistence and active community participation.
- **Project Clairvoyance**: A user's request for feedback in an incorrect channel led to redirection, and a discussion surfaced on the double-edged nature of comprehensive project use cases, illustrating the complexity of conveying specific user benefits.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Heed the Setup Cautions with New Models**: While setting up the **Deepseek Coder V2 Lite**, users should pay close attention to certain settings that are critical during the initial configuration, as one setting incorrectly left on could cause issues.

**When Autoupdate Fails, DIY**: **LM Studio** users have encountered broken autoupdates since version 0.2.22, necessitating manual download of newer versions. Links for downloading version 0.2.24 are functioning, but issues have been reported with version 0.2.25.

**Quantization's Quandary**: There's a notable variability in model responses based on different quantization levels. Users found Q8 to be more responsive compared to Q4, and these differences are important when considering model efficiency and output suitability.

**Config Chaos Demands Precision**: One user struggled with configuring the **afrideva/Phi-3-Context-Obedient-RAG-GGUF** model, triggering advice on specific system message formatting. This discussion emphasizes the importance of precise prompt structuring for optimal bot interaction.

**Open Interpreter Troubleshooting**: Issues regarding **Open Interpreter** defaulting to GPT-4 instead of LM Studio models led to community-shared workarounds for MacOS and references to a [YouTube tutorial](https://youtu.be/xPd8FFzIeOw?t=602) for detailed setup guidance.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek Coder V2 Now in the Wild**: The **DeepSeek-Corder-V2** models, both Lite and full, with 236x21B parameters, have been released, stirring conversations around their cost and efficiency, with an example provided for only 14 cents ([HuggingFace Repository](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)) and detailed explanations about their dense and MoE MLPs architecture in the discussions.

- **Meta Unfurls Its New AI Arsenal**: The AI community is abuzz with Meta's announcement of their colossal AI models, including **Chameleon**, a 7B & 34B language model for mixed-modal inputs and text-only outputs, and an array of other models like JASCO for music composition and a model adept at Multi-Token Prediction for coding applications ([Meta Announcement](https://x.com/aiatmeta/status/1803107817345393136)).

- **YouSim: The Multiverse Mirror**: The innovative web demo called [YouSim](https://yousim.ai/) has been spotlighting for its ability to simulate intricate personas and create ASCII art, with commendations for its identity simulation portal, even responding humorously with Adele lyrics when teased.

- **Flowwise, a Comfy Choice for LLM Needs?**: There's chatter around [Flowise](https://github.com/FlowiseAI/Flowise), a GitHub project that offers a user-friendly drag-and-drop UI for crafting custom LLM flows, addressing some users' desires for a Comfy equivalent in the LLM domain.

- **Model Behavior Takes an Ethical Pivot**: Discussions highlighted a perceptible shift in Anthropics and OpenAI's models, where they have censored responses to ethical queries, especially for creative story prompts that might necessitate content that's now categorized as unethical or questionable.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Google DeepMind Brings Sound to AI Videos**: DeepMind's latest Video-to-Audio (V2A) innovation can generate myriad audio tracks for silent AI-generated videos, pushing the boundaries of creative AI technologies [tweet details](https://x.com/rowancheung/status/1802734770117333257).
- **Questioning Creativity in Constrained Models**: A study on [arXiv](https://arxiv.org/abs/2406.05587) shows Llama-2 models exhibit lower entropy, suggesting that Reinforcement Learning from Human Feedback (RLHF) may reduce creative diversity in LLMs, challenging our alignment strategies.
- **Midjourney's Mystery Hardware Move**: Midjourney is reportedly venturing beyond software, spiking curiosity about their hardware ambitions, while the broader community debates the capabilities and applications of neurosymbolic AI and other LLM intricacies.
- **AI2 Spots First Fully Open-source Model**: The AI2 team's success at launching *M-A-P/Neo-7B-Instruct*, the first fully open-source model on WildBench, sparks discussions on the evolution of open-source models and solicits a closer look at future contenders like *OLMo-Instruct* [Billy's announcement](https://x.com/billyuchenlin/status/1802853516714881062).
- **AI Text-to-Video Scene Exploding**: Text-to-video tech is seeing a gold rush, with ElevenLabs offering a standout customizable, royalty-free sound effects generator [sound effects details](https://elevenlabs.io/sound-effects), while the community scrutinizes the balance between specialization and general AI excellence in this space.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's Academic Access and Feature Set**: Engineers discussed **Perplexity AI's** inability to access certain academic databases like Jstor and questioned the extent to which full papers or just abstracts are provided. The platform's limitations on PDF and Word document uploads were noted, along with alternative LLMs like **Google's NotebookLM** for handling large volumes of documents.

- **AI Models Face-off**: Preferences were voiced between different AI models; **Claude** was praised for its writing style but noted as restrictive on controversial topics, while **ChatGPT** was compared favorably due to fewer limitations.

- **Seeking Enhanced Privacy Controls**: A community member highlighted a privacy concern with Perplexity AI's public link sharing, exposing all messages within a collection and sparking a discussion on the need for improved privacy measures.

- **Access to Perplexity API in Demand**: A user from **Kalshi** expressed urgency in obtaining closed-beta API access for work integration, underscoring the need for features like **text tokenization and embeddings computation** which are currently absent in Perplexity but available in **OpenAI** and **Cohere**'s APIs.

- **Distinguishing API Capability Gaps**: The discourse detailed Perplexity’s shortcomings compared to **llama.cpp** and other platforms, lacking in developer-friendly features like function calling, and the necessary agent-development support provided by platforms like **OpenAI**.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Open Access to DanskGPT**: **DanskGPT** is now available with a [free version](https://chat.danskgpt.dk) and a more robust licensed offering for interested parties. The source code of the free version is public, and the development team is seeking contributors with computing resources.

- **Optimizing NVIDIA API Integration**: In discussions about the **NVIDIA Nemotron API**, members exchanged codes and tips to improve speed and efficiency within their data pipelines, with a focus on enhancing MMLU and ARC performances through model utilization.

- **AMD GPU Woes with Axolotl**: There's limited support for **AMD GPUs**, specifically the MI300X, in **Axolotl**, prompting users to collaborate on identifying and compiling necessary modifications for better compatibility.

- **Guidance Galore for Vision Model Fine-Tuning**: Step-by-step methods for fine-tuning vision models, especially **ResNet-50**, were shared; users can find all relevant installation, dataset preparation, and training steps in a detailed guide [here](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=077f3d3e-542e-4b6f-b825-cea95799abf7).

- **Building QDora from Source Quest**: A user's query about compiling **QDora from source** echoed the need for more precise instructions, with a pledge to navigate the setup autonomously with just a bit more guidance.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Webinar Alert: Level-Up With Advanced RAG**: The **60-minute webinar** by **@tb_tomaz from @neo4j** delved into integrating **LLMs with knowledge graphs**, offering insights on graph construction and entity management. Engineers interested in enhancing their models' context-awareness should catch up [here](https://t.co/R5kLvvnJc2).

- **LlamaIndex Joins the InfraRed Elite**: Cloud infrastructure company **LlamaIndex** has been recognized on the **InfraRed 100 list by @Redpoint**, acknowledging their milestones in reliability, scalability, security, and innovation. Check out the celebratory [tweet](https://t.co/X9Ec7ciWC9).

- **Switch to MkDocs for Better Documentation**: *LlamaIndex* transitioned from Sphinx to MkDocs from version 0.10.20 onwards for more efficient API documentation in large monorepos due to Sphinx's limitation of requiring package installation.

- **Tweaking Embeddings & Prompts for Precision**: Discussions covered the challenge of fine-tuning embeddings for an e-commerce RAG pipeline with numeric data, with a suggestion of using GPT-4 for synthetic query generation. Additionally, a technique for modifying LlamaIndex prompts to resolve discrepancies in local vs server behavior was shared [here](https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/?h=prompts).

- **Solving PGVector's Filtering Fog**: To circumvent the lack of documentation for PGVector's query filters, it was recommended to filter document IDs by date directly in the database, followed by using `VectorIndexRetriever` for the vector search process.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Mistral Finetuning Snafu Solved**: An attempt to finetune Mistral resulted in an `OSError` which, after suggestions to try version 0.3 and tweaks to token permissions, was successfully resolved.
- **Token Conundrum with Vision Model**: A discussion was sparked on StackOverflow regarding the `phi-3-vision` model's unexpected token count, seeing images consume around 2000 tokens, raising questions about token count and image size [details here](https://stackoverflow.com/questions/78635798/phi-3-vision-model-tokens).
- **Erratic Behavior in SFR-Embedding-Mistral**: Issues were raised concerning SFR-Embedding-Mistral's inconsistent similarity scores, especially when linking weather reports with dates, calling for explanations or strategies to address this discrepancy.
- **Credit Countdown Confusion**: The Discord community proposed creating a list to track different credit providers' expiration, with periods ranging from a few months to a year, and there was discussion of a bot to remind users of impending credit expiration.
- **Excitement for Gemini's New Tricks**: Enthusiasm poured in for exploring Gemini's context caching features, especially concerning many-shot prompting, indicating excitement for future hands-on experiments.

*Note: Links and specific numerical details were embedded when available for reference.*



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Major Markdown for Midnight Rose**: [Midnight Rose 70b](https://openrouter.ai/models/sophosympatheia/midnight-rose-70b/status) is now available at **$0.8 per million tokens**, after a relevant **90% price reduction**, creating a cost-effective option for users.

- **Updates on the Horizon**: Community anticipation for updates to OpenRouter was met with Alex Atallah's promise of imminent developments, utilizing an active communication approach to sustain user engagement.

- **A Deep Dive into OpenRouter Mechanics**: Users discussed OpenRouter's core functionality, which optimizes for **price or performance** via a **standardized API**, with additional educational resources available on the [principles page](https://openrouter.ai/docs/principles).

- **Reliability in the Spotlight**: Dialogue about the service's reliability was addressed with information indicating that OpenRouter's **uptime is the sum of all providers’ uptimes**, supplemented with data like the [Dolphin Mixtral uptime statistics](https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b/uptime).

- **Proactive Response to Model Issues**: The team's prompt resolution of concerns about specific models demonstrates an attentive approach to platform maintenance, highlighting their response to issues with **Claude** and **DeepInfra's Qwen 2**.




---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Creative Commons Content Caution**: Using **Creative Commons (CC)** content may minimize legal issues but could still raise concerns when outputs resemble copyrighted works. A proactive approach was suggested, involving "patches" to handle specific legal complaints.

- **Exploring Generative Potentials**: The performance of **CommonCanvas** was found lackluster with room for improvement, such as training texture generation models using free textures, while **DeepFashion2** disappointed in clothing and accessories image dataset benchmarks. For **language models**, the **GPT-NeoX** has accessible weights for [Pythia-70M](https://huggingface.co/EleutherAI/neox-ckpt-pythia-70m-v1), and for fill-in-the-middle linguistic tasks, models like **BERT**, **T5**, **BLOOM**, and **StarCoder** were debated with a spotlight on T5's performance.

- **Z-Loss Making an Exit?**: Within the AI community, it seems the usage of **z-loss** is declining with a trend towards load balance loss for MoEs, as seen in tools like **Mixtral** and noted in models such as **DeepSeek V2**. Additionally, there's skepticism about the reliability of HF configs for Mixtral, and a suggestion to refer to the official source for its true parameters.

- **Advanced Audio Understanding with GAMA**: Discussion introduced **GAMA**, an innovative Large Audio-Language Model (LALM), and touched on the latest papers including those on **Meta-Reasoning Prompting (MRP)** and **sparse communication topologies** for multi-agent debates to optimize computational expenses, with details and papers accessible from sources like [arXiv](https://arxiv.org/abs/2406.11776) and the [GAMA project](https://sreyan88.github.io/gamaaudio/).

- **Interpreting Neural Mechanisms**: There was a healthy debate on understanding logit prisms with references to an [article on logit prisms](https://neuralblog.github.io/logit-prisms/#fig-logit-per-layer) and the concept's relation to direct logit attribution (DLA), pointing to additional resources like the [IOI paper](https://arxiv.org/pdf/2211.00593) for members to explore further.

- **Delving into vLLM Configuration Details**: A brief technical inquiry was raised about the possibility of integrating vLLM arguments like `--enforce_eager` directly into the engine through `model_args`. The response indicated a straightforward approach using kwargs but also hinted at a need to resolve a "type casting bug".



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain Learners Face Tutorial Troubles**: Members experienced mismatch issues between **LangChain** versions and published tutorials, with one user getting stuck at a timestamp in a [ChatGPT Slack bot video](https://www.youtube.com/watch?v=qKZLDEIL2r0). Changes like the deprecation of `LLMChain` in LangChain 0.1.17 and the upcoming removal in 0.3.0 highlight the rapid evolution of the library.

**Extracting Gold from Web Scrapes & Debugging Tips**: A user was guided on **company summary and client list** extraction from website data using LangChain, and others discussed debugging LangChain's LCEL pipelines with `set_debug(True)` and `set_verbose(True)`. Frustration arose from `BadRequestError` in APIs, reflecting challenges in handling unexpected API behavior.

**Serverless Searches & Semantic AI Launches**: An article on creating a **serverless semantic search with AWS Lambda and Qdrant** was shared, alongside the launch of **AgentForge** on [ProductHunt](https://www.producthunt.com/posts/agentforge), integrating LangChain, LangGraph, and LangSmith. Another work, [YouSim](https://yousim.ai), showcased a backrooms-inspired simulation platform for identity experimentation.

**New Mediums, New Codes**: **jasonzhou1993** explored **AI's impact on music creation** in a [YouTube tutorial](https://youtu.be/yM-Lpq6E3Uc?si=1yu7xSlZkF9HekZp), while also sharing a **Hostinger** website builder discount code `AIJASON`.

**Calls for Collaboration and Sharing Innovations**: A plea for beta testers surfaced for an advanced research assistant at [Rubik's AI](https://rubiks.ai), mentioning premium features like Claude 3 Opus and GPT-4 Turbo. Hugging Face's advice to sequester environment setup from code and the embrace of tools like Bitwarden for managing credentials stressed importance of secure and clean development practices.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Rounded Floats or Rejected PRs**: A pull request ([#5021](https://github.com/tinygrad/tinygrad/pull/5021)) aims to improve code clarity in **tinygrad** by rounding floating points in `graph.py`, while **George Hotz** emphasizes a *new policy* against low-quality submissions, closing PRs that haven't been thoroughly self-reviewed.
- **Enhanced Error Reporting for OpenCL**: An upgrade in OpenCL error messages for **tinygrad** is proposed in a pull request ([#5004](https://github.com/tinygrad/tinygrad/pull/5004)), though it requires further review before merging.
- **Realization Impacts in Tinygrad**: Discussions unfold around the impacts of `realize()` on operation outputs, observing the difference between lazy and eager execution, and how kernel fusion can be influenced by caching and explicit realizations.
- **Kernel Combination Curiosity**: Participants examine how forced kernel combinations might be achieved, particularly for custom hardware, with advice to investigate the scheduler of **Tinygrad** to better understand possible implementations.
- **Scheduler's Role in Operation Efficiency**: Deepening interest in **Tinygrad's** scheduler emerges, as AI engineers consider manipulating it to optimize custom accelerator performance, highlighting a thoughtful dive into its ability to manage kernel fusion and operation execution.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI-Generated Realism Strikes Again**: A [RunwayML Gen-3 clip](https://fxtwitter.com/Mr_AllenT/status/1802706451586023763) showcased its impressive AI-generated details, blurring the line between AI and reality, with users noting its indistinguishable nature from authentic footage.
- **Silent Videos Get a Voice**: DeepMind's V2A technology, through a process explained in a [blog post](https://deepmind.google/discover/blog/generating-audio-for-video/), generates soundtracks just from video pixels and text prompts, spotlighting a synergy with models like Veo.
- **Meta Advances Open AI Research**: Meta FAIR has introduced new [research artifacts](https://ai.meta.com/blog/meta-fair-research-new-releases/) like Meta Llama 3 and V-JEPA, with Chameleon vision-only weights now openly provided, fueling further AI tooling.
- **Open-Source Community Callout**: The PKU-YuanGroup urges collaboration for the Open-Sora Plan outlined on [GitHub](https://github.com/PKU-YuanGroup/Open-Sora-Plan), striving to replicate the Open AI T2V model, inviting community contributions.
- **Interpretable Weights Space Unearthed**: UC Berkeley, Snap Inc., and Stanford researchers unravel an **interpretable latent weight space** in diffusion models, as shared on [Weights2Weights](https://snap-research.github.io/weights2weights/), enabling the manipulation of visual identities within a largescale model space.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

**CUDA vs MPS: Beware the NaN Invasion**: Engineers discussed an issue where `nan` outputs appeared on **CUDA** but not on **MPS**, tied to differences in kernel execution paths for *softmax* operations in **SDPA**, leading to [softmax causing `nan` on large values](https://github.com/pytorch/pytorch/issues/110213#issuecomment-1739952114).

**Cache Clash with Huggingface**: There were discussions on system crashes during fine-tuning with **Torchtune** due to Huggingface's cache overflowing, causing concern and a call for solutions among users.

**Constructing Bridge from Huggingface to Torchtune**: The guild shared a detailed process for converting **Huggingface** models to **Torchtune** format, highlighting [Torchtune Checkpointers](https://pytorch.org/torchtune/main/deep_dives/checkpointer.html) for easy weight conversion and loading.

**The Attention Mask Matrix Conundrum**: Clarification on the proper attention mask format for padded token inputs to avoid disparity across processing units was debated, ensuring that the model's focus is correctly applied.

**Documentation to Defeat Disarray**: Links to **Torchtune** documentation, including RLHF with PPO and GitHub pull requests, were shared to assist with implementation details and facilitate knowledge sharing among engineers. [RLHF with PPO](https://github.com/pytorch/torchtune/actions/runs/9373237554/job/25811096938#step:6:261) | [Torchtune Pull Request](https://github.com/pytorch/torchtune/pull/875)



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **SEO Shenanigans Muddle AI Conversations**: Members shared frustrations over an [SEO-generated article](https://www.neural-voice.ai/mastering-conversational-ai-challenges-and-prospects/) that incorrectly referred to "Google's ChatGPT," highlighting the lack of citations and poor fact-checking typical in some industry-related articles.
- **Herzog Voices AI Musings**: Renowned director Werner Herzog was featured reading davinci 003 outputs on a [This American Life episode](https://podcasts.apple.com/us/podcast/this-american-life/id201671138?i=1000657607717), showcasing human-AI interaction narratives.
- **The Quest for Podcast Perfection**: The guild discussed tools for creating podcasts, with a nod to [smol-podcaster](https://github.com/FanaHOVA/smol-podcaster) for intro and show note automation; they also compared transcription services from Assembly.ai and Whisper.
- **Meta's Model Marathon Marches On**: Meta showcased four new AI models – **Meta Chameleon, Meta Multi-Token Prediction, Meta JASCO, and Meta AudioSeal**, aiming to promote open AI ecosystems and responsible development. Details are found in their [announcement](https://x.com/AIatMeta/status/1803103538169651679).
- **Google's Gemini API Gets Smarter**: The introduction of [context caching](https://x.com/officiallogank/status/1803096828595863608?s=46&t=90xQ8sGy63D2OtiaoGJuww) for Google's Gemini API promises cost savings and upgrades to both 1.5 Flash and 1.5 Pro versions, effective immediately.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Llama Beats Codestral in Commercial Arena**: The **llama-70b** model is recommended for commercial applications over **codestral**, despite the latter's higher ranking, mainly because codestral is not fit for commercial deployment. The [LMSys Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) was cited, where llama-3-70b's strong performance was also acknowledged.

- **Eager for E2B Integration**: Excitement is shared over potential integration profiles, highlighting **e2b** as a next candidate, championing its secure sandboxing for executing outsourced tasks.

- **Peek at OpenInterpreter's Party**: An inquiry about the latest OpenInterpreter release was answered with a link to "WELCOME TO THE JUNE OPENINTERPRETER HOUSE PARTY", a video on [YouTube](https://www.youtube.com/live/pqBuxmpgpY0?si=DEXxMuOIqIK1guYF) powered by Restream.

- **Launch Alert for Local Logic Masters**: **Open Interpreter’s Local III** is announced, spotlighting features for offline operation such as setting up fast, local large language models (LLMs) and a free inference endpoint for training personal models.

- **Photos Named in Privacy**: A new offline tool for automatic and descriptive photo naming is introduced, underscoring the user privacy and convenience benefits.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Agent Hospital Aims to Revolutionize Medical Training**: In the AI development sphere, the [Agent Hospital paper](https://arxiv.org/abs/2405.02957) presents **Agent Hospital**, a simulated environment where autonomous agents operate as patients, nurses, and doctors. **MedAgent-Zero** facilitates learning and improving treatment strategies by mimicking diseases and patient care, possibly transforming medical training methods.

- **Simulated Experience Rivals Real-World Learning**: The study on **Agent Hospital** contends that doctor agents can gather real-world applicable medical knowledge by treating virtual patients, simulating years of on-the-ground experience. This could streamline learning curves for medical professionals with data reflecting thousands of virtual patient treatments.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Video Deep Dive into LLM CLI Usage**: Simon Willison showcased **Large Language Model (LLM)** interactions via command-line in a detailed video from the [Mastering LLMs Conference](https://maven.com/parlance-labs/fine-tuning), supplemented with an [annotated presentation](https://simonwillison.net/tags/annotatedtalks/) and the talk available on [YouTube](https://www.youtube.com/watch?v=elQ7hG7Z5cc).

- **Calmcode Prepares to Drop Fresh Content**: **Calmcode** is anticipated to issue a new release soon, as hinted by Vincent Warmerdam, with a new maintainer at the helm.

- **Acknowledgment Without Action**: In a brief exchange, a user expressed appreciation, potentially for the aforementioned video demo shared by Simon Willison, but no further details were discussed.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Fast-Track MoE Performance**: A pull request titled [improve moe prompt eval speed on cpu #6840](https://github.com/ggerganov/llama.cpp/pull/6840) aims to enhance model evaluation speed but requires rebasing due to conflicts with the main branch. The request has been made to the author for the necessary updates.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1252337997559369748)** (526 messages🔥🔥🔥): 

- **Age is just a number for AI veterans**: Members discussed their ages and how some of them, being around 40-60 years old, are jokingly referred to as "dead men walking" by their daughters. They humorously debated about hair loss and longevity, emphasizing that coding and maintaining an active mind keep them youthful.
- **GGUF offloading and GPU layers confusion**: A user sought advice on how many GGUF layers to offload to VRAM, with suggestions including trial and error and potential ways to estimate based on available VRAM vs. the total GGUF size. It was recommended that checking the llama.cpp outputs or HuggingFace model details could help determine the correct layer numbers.
- **Subscription vs. single payment model for multiGPU support**: The discussion leaned towards making multiGPU support a paid feature, with suggestions to implement a subscription model starting at $9.99 a month. Members debated various payment models including one-time fees, training fees, or tiered pricing for hobbyists and businesses.
- **GPU rental and efficiency**: Members recommended renting GPUs due to the high costs and heat management issues of local setups, particularly in places with high electricity prices. Running local setups, especially with intensive models, was seen as impractical compared to renting state-of-the-art hardware.
- **OpenAI NSA concerns**: Members expressed worries about OpenAI appointing a former NSA director to its board, triggering discussions about privacy and government surveillance. A shared tweet from Snowden warned about potential data security risks with OpenAI products.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/cognitivecomputations/dolphin-2.9.2-Phi-3-Medium-abliterated">cognitivecomputations/dolphin-2.9.2-Phi-3-Medium-abliterated · Hugging Face</a>: no description found</li><li><a href="https://vercel.com/legal/terms">Terms of Service – Vercel</a>: See our terms of our service and how they relate to you.</li><li><a href="https://www.youtube.com/watch?v=-gGLvg0n-uY">Raiden Warned About AI Censorship - MGS2 Codec Call (2023 Version)</a>: The Colonel warns Raiden about the plans to use AI to censor the Internet.An experiment in creative writing and AI speech synthesis, inspired by the famous &quot;...</li><li><a href="https://x.com/aiatmeta/status/1803107817345393136">Tweet from AI at Meta (@AIatMeta)</a>: Today is a good day for open science.  As part of our continued commitment to the growth and development of an open ecosystem, today at Meta FAIR we’re announcing four new publicly available AI models...</li><li><a href="https://youtu.be/Cxqca4RQd_M?t=3">If Google Was A Guy (Full Series)</a>: Support CollegeHumor by signing up for DROPOUT: https://signup.dropout.tv. Tons of exclusive content, ad-free, for only $5 a month (that&#39;s like 17 cents a da...</li><li><a href="https://www.marktechpost.com/2024/06/14/yandex-introduces-yafsdp-an-open-sour">no title found</a>: no description found</li><li><a href="https://tenor.com/view/card-codes-gif-21814106">Card Codes GIF - Card Codes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.marktechpost.com/2024/06/14/yandex-introduces-yafsdp-an-open-source-ai-tool-that-promises-to-revolutionize-llm-training-by-cutting-gpu-usage-by-20/?amp">no title found</a>: no description found</li><li><a href="https://x.com/Snowden/status/1801610725229498403">Tweet from Edward Snowden (@Snowden)</a>: They&#39;ve gone full mask-off: 𝐝𝐨 𝐧𝐨𝐭 𝐞𝐯𝐞𝐫 trust @OpenAI or its products (ChatGPT etc). There is only one reason for appointing an @NSAGov Director to your board. This is a willful, calculat...</li><li><a href="https://github.com/facebookresearch/chameleon">GitHub - facebookresearch/chameleon: Repository for Meta Chameleon a mixed-modal early-fusion foundation model from FAIR.</a>: Repository for Meta Chameleon a mixed-modal early-fusion foundation model from FAIR. - facebookresearch/chameleon</li><li><a href="https://github.com/hpcaitech/Open-Sora">GitHub - hpcaitech/Open-Sora: Open-Sora: Democratizing Efficient Video Production for All</a>: Open-Sora: Democratizing Efficient Video Production for All - hpcaitech/Open-Sora</li><li><a href="https://github.com/yandex/YaFSDP">GitHub - yandex/YaFSDP: YaFSDP: Yet another Fully Sharded Data Parallel</a>: YaFSDP: Yet another Fully Sharded Data Parallel. Contribute to yandex/YaFSDP development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/tsynbio/ProteinLMBench?row=0">tsynbio/ProteinLMBench · Datasets at Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.12226">AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling</a>: We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT ...</li><li><a href="https://huggingface.co/datasets/fnlp/AnyInstruct">fnlp/AnyInstruct · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1252339722492051638)** (10 messages🔥): 

- **Gemini 2.0 hype builds**: "Does this mean Gemini 2.0 is close?" asked a member, to which another responded affirmatively, "Yes very."

- **24GB VRAM delight**: A member praised the size for 24GB VRAM, stating, "It's just such a great size for 24gb vram." Others shared excitement about training potential, expressing hope to "train it too."

- **Runpod ambitions**: Enthusiasm for testing was evident with a member planning to "rent a runpod 48gb instance just to put it through its paces."

- **Saturn Cloud access issues**: An inquiry was made about creating accounts on Saturn Cloud, noting, "they got a waitlist but the link is not working."

- **Sloth sticker quest**: "Did you create the stickers Mike? Can I get the Daniel sticker one," asked a member, with a clarifying reply about the "GPU out of a box" sticker. Another showed interest in all sloth-themed stickers: "all of the sloths."
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1252338068585840660)** (143 messages🔥🔥): 

```html
<ul>
    <li><strong>Colab Training Sessions Woes</strong>: One user experienced issues with their Google Colab training session for Unsloth cutting out at 90% after 23 hours. They expressed frustration and received advice about preemptively enabling checkpointing within TrainingArguments() to avoid future occurrences.</li>
    <li><strong>Fine-Tuning LLMs Issues</strong>: Users gabrielsandstedt and shensmobile discussed problems related to fine-tuning large language models (LLMs) on Google Colab. The importance of enabling checkpointing and limitations of session lengths were highlighted.</li>
    <li><strong>Tokenizing Troubles</strong>: A member wanted to compare vocab before and after fine-tuning an LLM but faced storage limits on free Google Colab. Discussion revolved around the necessity of saving the tokenizer along with the model and possible space-saving methods.</li>
    <li><strong>Dataset Formatting and Schema</strong>: Thefanciestpeanut guided gbourdin on how to convert JSON to Parquet for better training efficiency in Unsloth, emphasizing mapping the data correctly for fine-tuning. They shared a detailed code snippet for dataset conversion and loading in Python.</li>
    <li><strong>Mixed GPU Usage Obstacles</strong>: Several users, including karatsubabutslower and origamidream, deliberated on challenges encountered when using multiple GPUs with Unsloth, suggesting using older versions or setting environment variables properly to circumvent usage restrictions.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/issues/4288#issuecomment-2174780103"> Error: More than 1 GPUs have a lot of VRAM usage. Please obtain a commercial license. · Issue #4288 · hiyouga/LLaMA-Factory</a>: Reminder I have read the README and searched the existing issues. System Info LLaMA-Factory-0.8.1, utuban 22.04 python 3.10.14 Reproduction llamafactory-cli train --stage sft --do_train True --mode...</li><li><a href="https://github.com/codename-hub/php-parquet">GitHub - codename-hub/php-parquet: PHP implementation for reading and writing Apache Parquet files/streams</a>: PHP implementation for reading and writing Apache Parquet files/streams - codename-hub/php-parquet</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>: no description found
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1252348171426861086)** (5 messages): 

- **Nvidia 5090’s RAM speculation sparks debate**: A member remarked that there's "almost certainly a 0% chance the 5090 will have 64GB of RAM," suggesting the B6000 cards would more likely feature 64GB. They posited that Nvidia would likely release a 5090 with 24GB or 28GB and hold back a 32GB variant for a potential 5090 Ti or Super card. [Videocardz Speculation](https://videocardz.com/newz/nvidia-rtx-5090-new-rumored-specs-28gb-gddr7-and-448-bit-bus).

- **AI capabilities stagnation discussion heats up**: An article from Semianalysis discussed the stagnation of AI capabilities since GPT-4's release, attributing it to the lack of a significant increase in compute devoted to single models. They suggested that newer models like Google’s Gemini Ultra and Nvidia Nemotron 340B used similar or higher amounts of compute compared to GPT-4 but fell short due to inferior architecture. [Semianalysis Article](https://www.semianalysis.com/p/100000-h100-clusters-power-network).

- **RDNA4 and Intel Battlemage competition in doubt**: In response to Nvidia discussions, a member commented that there won't be anything in the "RDNA4 lineup to compete" and mentioned that Intel has an opportunity with their Battlemage/Xe2.

**Link mentioned**: <a href="https://www.semianalysis.com/p/100000-h100-clusters-power-network">100k H100 Clusters: Power, Network Topology, Ethernet vs InfiniBand, Reliability, Failures, Checkpointing</a>: Frontier Model Scaling Challenges and Requirements, Fault Recovery through Memory Reconstruction, Rack Layouts

  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1252343980423909487)** (2 messages): 

- **Hope for more work on search**: A member shared a link to an [arXiv paper](https://arxiv.org/pdf/2406.07394) and expressed their hope that *"more people work on search"*. The sentiment reflects a desire for further advancements and contributions in the field of search algorithms.
  
- **Impressive match of GPT-4 with LLAMA3 8B**: A member commented on the impressive nature of matching **GPT-4** with **LLAMA3 8B**. This highlights the ongoing progress in aligning different model architectures to achieve comparable performance.
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages): 

niceboy2989: <@848720848282189855> I can help you
  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1252344004738289765)** (1 messages): 

- **Announcing tpux project**: A member announced the tpux project, *"a powerful suite of tools to simplify Cloud TPU setup and operation to make it easier to use JAX across multiple hosts"*. Encouraged users to visit the [GitHub repository](https://github.com/yixiaoer/tpux) for more information and to give it a star on GitHub.

**Link mentioned**: <a href="https://github.com/yixiaoer/tpux">GitHub - yixiaoer/tpux: A set of Python scripts that makes your experience on TPU better</a>: A set of Python scripts that makes your experience on TPU better - yixiaoer/tpux

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1252368792994971750)** (25 messages🔥): 

- **Troubles with quantization API configurations**: Discussion focused on the difficulty of changing quantization configurations, specifically group sizes, using the new API `quantize(model, quantization_method)`. One user pointed out you need to pass a function like `quantize(model,int4wo(group_size=group_size))` to change settings.
  
- **GitHub feedback on quantization**: Users referred to GitHub issues ([#384](https://github.com/pytorch/ao/issues/384) and [#375](https://github.com/pytorch/ao/issues/375)) for feedback and consistency improvements in the quantization API. One mentioned the inconsistency in text for quantization types as annoying.

- **Incorporating gptfast implementation**: A discussion emerged about including the gptfast model download script, linking to a [GitHub pull request](https://github.com/pytorch/ao/pull/372) aimed at adding instructions for downloading model weights. It was noted that some recent PRs might need additional review before merging.

- **Quantization API user feedback**: Different ideas were proposed for the quantization API syntax, with suggestions like `quantize(m, Int4WeightOnly(groupsize=32))` or `quantize(m, QuantConfig(nbits=4, groupsize=32))`. There were debates over simplicity and ease of adding new support, either via classes or functions.

- **Emphasis on proper PR reviews**: A user emphasized the importance of thorough pull request reviews over quick approvals, mentioning specific PRs ([#372](https://github.com/pytorch/ao/pull/372) and [#374](https://github.com/pytorch/ao/pull/374)) that lacked sufficient documentation or testing before merging.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/issues/391,">Issues · pytorch/ao</a>: PyTorch dtype and layout library. 30% speedups for training. 2x speedups and 65% less VRAM for inference. Composability with FSDP and torch.compile. - Issues · pytorch/ao</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/scripts/download.py">gpt-fast/scripts/download.py at main · pytorch-labs/gpt-fast</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - pytorch-labs/gpt-fast</li><li><a href="https://github.com/pytorch/ao/pull/372">073 scripts for benchmarks by HDCharles · Pull Request #372 · pytorch/ao</a>: Instructions for downloading model weights  Summary:  Added instructions for downloading model weights, and model weight download scripts.  Test Plan:  huggingface-cli login sh ./scripts/prepare.sh...</li><li><a href="https://github.com/pytorch/ao/pull/374">eval script for llama by HDCharles · Pull Request #374 · pytorch/ao</a>: Summary: previously we were only doing this in the tests but now we have an eval script to along with generate.py Test Plan: python eval.py -q &quot;int4wo-64-gptq&quot; expected results: (using meta-...</li><li><a href="https://github.com/pytorch/ao/issues/384#issue-2355481211">Feedback on `quantize()` API · Issue #384 · pytorch/ao</a>: Previously we do this from torchao.quantization.quant_api import change_linear_weights_to_int8_woqtensors model = torch.compile(model, mode=&quot;max-autotune&quot;, fullgraph=True) change_linear_weig...</li><li><a href="https://github.com/pytorch/ao/issues/375">quantization api name consistency · Issue #375 · pytorch/ao</a>: having the string to get a type of quantization and the constructor for that quantization as different text is super annoying. https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1252337264038645831)** (536 messages🔥🔥🔥): 

- **DataLoader optimization discussion**: Members discussed compartmentalizing the save/load state logic into `dataloader.h` and testing it from the script. One member noted, *"I intend to merge the DataLoader once CI is happy."*
- **FlashAttention in HF transformers**: Enabling FlashAttention 2 significantly improved evaluation metrics and performance. One member noted, *"if we use the main branch of the eval harness it should be okay."*
- **Exploration of NCCL without MPI for multi-node setups**: It was noted the multi-node functionality PR aimed to remove MPI dependence, using `srun` for launch control while still requiring `mpirun` for single-node runs. *"all in all this doesn't seem like a big change".*
- **Performance metrics compared with previous benchmarks**: There were extensive discussions and tests on the impact of various optimizations using streams and prefetching on the forward pass. *"The differences are not anything huge between curr and the streamed version after some extensive profiling".*
- **FP32 and BF16 discrepancies for logits**: Clarifications about rounding errors in BF16 and its impact on floating-point accuracy were discussed. One member noted, *"Thinking whether the only source of difference here is the non-associativity of floating point numbers and the fact we have different kernels?"*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/stable/generated/torchtune.modules.RMSNorm.html">RMSNorm &mdash; TorchTune  documentation</a>: no description found</li><li><a href="https://huggingface.co/rhysjones/gpt2-774M-fineweb-150B/blob/main/config.json">config.json · rhysjones/gpt2-774M-fineweb-150B at main</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Inline_function#C99)">Inline function - Wikipedia</a>: no description found</li><li><a href="https://www.semianalysis.com/p/100000-h100-clusters-power-network">100k H100 Clusters: Power, Network Topology, Ethernet vs InfiniBand, Reliability, Failures, Checkpointing</a>: Frontier Model Scaling Challenges and Requirements, Fault Recovery through Memory Reconstruction, Rack Layouts</li><li><a href="https://siboehm.com/articles/22/CUDA-MMM">How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog</a>: In this post, I’ll iteratively optimize an implementation of matrix multiplication written in CUDA.My goal is not to build a cuBLAS replacement, but to deepl...</li><li><a href="https://www.harmdevries.com/post/model-size-vs-compute-overhead/">Go smol or go home | Harm de Vries</a>: The Chinchilla scaling laws suggest we haven’t reached the limit of training smaller models for longer.</li><li><a href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md">cutlass/media/docs/efficient_gemm.md at main · NVIDIA/cutlass</a>: CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/607">Llama RoPE Forward Kernels by AndreSlavescu · Pull Request #607 · karpathy/llm.c</a>: no description found</li><li><a href="https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/">CUDA Matrix Multiplication Optimization</a>: General Matrix Multiplication CUDA Performance Optimization</li><li><a href="https://github.com/karpathy/llm.c/pull/594">add scripts to export to HF and run Eleuther evals by karpathy · Pull Request #594 · karpathy/llm.c</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/601">Fix stochastic rounding in encoder backward kernel by gordicaleksa · Pull Request #601 · karpathy/llm.c</a>: #597 provided unique seeds to adamw update. This PR does the same thing for the encoder backward which is the only other place where we do stochastic rounding.</li><li><a href="https://github.com/karpathy/llm.c/pull/610">gpt2_forward adding CUDA streams with events for async layered operations, cache prefetching for efficient data access with high temporal locality by bgorlick · Pull Request #610 · karpathy/llm.c</a>: In the forward pass in gpt2_train.cu  adding cuda streams with events for async layered operations added offset precalculations and cache prefetching for efficient data access with high temporal lo...</li><li><a href="https://github.com/karpathy/llm.c/pull/614">Stricter FP32 tests by gordicaleksa · Pull Request #614 · karpathy/llm.c</a>: Stricter FP32 logit accuracy Much stricter FP32 loss accuracy Much stricter FP32 grad tensor accuracy (and somewhat stricter 16 bit accuracy) Copied over new expected loss values from PyTorch (they...</li><li><a href="https://github.com/karpathy/llm.c/pull/573/commits/c81f1efbb82b4056cb9402d2ae7786e9d0165f1f">Dataloader - introducing randomness by gordicaleksa · Pull Request #573 · karpathy/llm.c</a>: On the way to fully random train data shuffling... This PR does the following:  Each process has a different unique random seed Each process train data loader independently chooses its starting sha...</li><li><a href="https://github.com/karpathy/llm.c/pull/426/files">NCCL only multi-gpu multi-node training without MPI by chinthysl · Pull Request #426 · karpathy/llm.c</a>: Scheduling jobs using Slurm seems much easier in a multi-node training setup compared to setting up MPI for the cluster. This draft contains the changes to use mpirun for single-node training and S...</li><li><a href="https://forums.developer.nvidia.com/t/integer-arithmetic-overflow/82347">Integer arithmetic overflow</a>: Is it defined how integer arithmetic overflows?  For instance, is it guaranteed that adding or multiplying two large unsigned ints will “gracefully” overflow like modulus 2^32?  I imagine this is some...</li><li><a href="https://github.com/meta-llama/llama/blob/main/llama/model.py#L63">llama/llama/model.py at main · meta-llama/llama</a>: Inference code for Llama models. Contribute to meta-llama/llama development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/pull/426#issuecomment-2175386065">NCCL only multi-gpu multi-node training without MPI by chinthysl · Pull Request #426 · karpathy/llm.c</a>: Scheduling jobs using Slurm seems much easier in a multi-node training setup compared to setting up MPI for the cluster. This draft contains the changes to use mpirun for single-node training and S...</li><li><a href="https://github.com/karpathy/llm.c/pull/600/files">Use faster kernel for LayerNorm forward by gordicaleksa · Pull Request #600 · karpathy/llm.c</a>: I ran kernel 5 under /dev/cuda/ (./layernorm_forward 5) on both RTX 3090 and H100 systems and it&amp;#39;s faster on both of them. Numbers: kernel 3, optimal block size on:  RTX 3090 → 32 (689.11 GB/s...</li><li><a href="https://github.com/karpathy/llm.c/pull/556">Utilities for cuda streams + disk IO by ngc92 · Pull Request #556 · karpathy/llm.c</a>: handling disk io for checkpointing with cuda streams is a nontrivial task. If you&#39;re  not careful, you can easily get broken code (need to wait for data to be on the CPU before you can start writi...</li><li><a href="https://github.com/pytorch/audio/issues/62">undefined symbol when importing torchaudio with pytorch  · Issue #62 · pytorch/audio</a>: Hi, When importing torchaudio with pytorch 0.4.1 I get an undefined symbol. It does however work with v0.4.0. audio version: 7314b36 Successfully installed numpy-1.15.0 torch-cpu-0.4.1 torchaudio-0...</li><li><a href="https://www.h-schmidt.net/FloatConverter/IEEE754.html">IEEE-754 Floating Point Converter</a>: no description found</li><li><a href="https://stackoverflow.com/questions/18195715/why-is-unsigned-integer-overflow-defined-behavior-but-signed-integer-overflow-is">Why is unsigned integer overflow defined behavior but signed integer overflow isn&#x27;t?</a>: Unsigned integer overflow is well defined by both the C and C&#x2B;&#x2B; standards.  For example, the C99 standard (&#xA7;6.2.5/9) states &#xD;&#xA;  A computation involving unsigned operands can nev...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1252407422979674214)** (9 messages🔥): 

- **Upcoming API Doc for LayoutTensor Class**: A member announced the forthcoming publication of a developer-facing API doc for feedback, highlighting the *LayoutTensor class* as an abstraction for tensor subclasses across various formats optimized for specific operators, devices, and data types.
- **Tinygemm Kernel Argument Clarification**: The argument *inner_k_tiles* was clarified as specific to the tinygemm kernel, indicating that other bit packing algorithms need not consider it.
- **Draft of TorchAO Tensor Subclass API Doc**: A member shared a [draft for torchao tensor subclass-based API doc](https://github.com/pytorch/ao/issues/391), requesting feedback on the modeling user API and developer API.
- **PR Iteration and Optimization**: Discussion among members about iterating over the current implementation after a PR merge, pointing out opportunities for operators to work directly on the packed tensor, avoiding the need to unpack and repack.

**Link mentioned**: <a href="https://github.com/pytorch/ao/issues/391,">Issues · pytorch/ao</a>: PyTorch dtype and layout library. 30% speedups for training. 2x speedups and 65% less VRAM for inference. Composability with FSDP and torch.compile. - Issues · pytorch/ao

  

---


### **CUDA MODE ▷ #[sparsity](https://discord.com/channels/1189498204333543425/1247663759434977453/1252431413890912298)** (3 messages): 

- **CUTLASS beats CuBLAS by 10% in pure C++**: A member shared a [blog post](https://www.thonking.ai/p/strangely-matrix-multiplications) detailing how the **CUTLASS** library achieved 10% better performance than **CuBLAS** in pure C++ for large matrix multiplications (8192 x 8192 x 8192). They highlighted that CUTLASS reached 288 Teraflops compared to CuBLAS's 258 Teraflops.
- **Python binding nullifies CUTLASS gains**: When binding **CUTLASS** kernels into Python, the performance advantages disappeared, bringing CUTLASS’s performance down to the same level as CuBLAS at 257 Teraflops. This observation noted the challenge of maintaining performance gains when integrating with Python.



**Link mentioned**: <a href="https://www.thonking.ai/p/strangely-matrix-multiplications">Strangely, Matrix Multiplications on GPUs Run Faster When Given &quot;Predictable&quot; Data! [short]</a>: Great minds discuss flops per watt.

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1252344173089128638)** (363 messages🔥🔥): 

```html
- **Civitai bans SD3 content**: Civitai has temporarily banned all SD3 related content due to concerns about the license's clarity, as shared by a user *“due to a lack of clarity in the license associated with Stable Diffusion 3, we are temporarily banning all SD3 based models.”* ([Civitai Announcement](https://civitai.com/articles/5732)).
- **Community dissatisfaction with SD3 release**: Multiple users expressed disappointment with the SD3 model, describing it as *“the worst base model release yet.”* Complaints were directed at both the performance and licensing issues.
- **SD3 Performance and Alternatives**: Users discussed the architecture and potential of SD3, noting its *“16ch VAE allows better text understanding”*, yet also acknowledging that other models like Pixart and Lumina can do *“more with less compute.”*
- **License concerns and legal implications**: There's significant worry in the community about how the SD3 model's license might allow Stability AI *“too much power over the models.”* This has caused platforms like Civitai to seek legal clarity before allowing SD3 content.
- **Comparisons with other tools**: Discussions often referenced alternate tools and software, with one user stating *“I swapped to Pixart Sigma...prompt adherence is good but has issues with limbs.”* Other users recommended different models and interfaces for various use cases including StableSwarmUI and ComfyUI.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://civitai.com/articles/5732">Temporary Stable Diffusion 3 Ban | Civitai</a>: Unfortunately, due to a lack of clarity in the license associated with Stable Diffusion 3 , we are temporarily banning: All SD3 based models All mo...</li><li><a href="https://civitai.com/models/147933/wowxlpdsd3?modelVersionId=576876">WoW_(XL+PD+SD3). - WoW_XL Five (v5) | Stable Diffusion Checkpoint | Civitai</a>: SD3 Model removed for now until CIVITAI can clarify it&#x27;s position legally. -- The latest version of WoW_XL (5) is a collaboration between myself an...</li><li><a href="https://ai.meta.com/blog/meta-fair-research-new-releases/?utm_source=twitter&utm_medium=organic_social&utm_content=video&utm_campaign=fair">no title found</a>: no description found</li><li><a href="https://github.com/tyxsspa/AnyText">GitHub - tyxsspa/AnyText: Official implementation code of the paper &lt;AnyText: Multilingual Visual Text Generation And Editing&gt;</a>: Official implementation code of the paper &lt;AnyText: Multilingual Visual Text Generation And Editing&gt; - tyxsspa/AnyText</li><li><a href="https://github.com/Stability-AI/StableSwarmUI">GitHub - Stability-AI/StableSwarmUI: StableSwarmUI, A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: StableSwarmUI, A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - Stability-AI/StableSwarmUI</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1dhd7vz/the_developer_of_comfy_who_also_helped_train_some/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://civitai.com/models/116225/4x-ultrasharp">4x-Ultrasharp - 4x-UltraSharp v1.0 | Stable Diffusion Upscaler | Civitai</a>: &amp;gt;&amp;gt;&amp;gt; UPLOADING/SHARING MY MODELS OUTSIDE CIVITAI IS STRICLY PROHIBITED* &amp;lt;&amp;lt;&amp;lt; The only authorized generative service website are: Ma...</li><li><a href="https://fb.watch/sNj0i5v3jZ/">no title found</a>: no description found
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1252346722353676329)** (311 messages🔥🔥): 

- **Civitai Bans SD3 Models due to Licensing Issues**: A member shared that [Civitai is temporarily banning all SD3-based models](https://civitai.com/articles/5732) due to unclear licensing from Stability AI. Concerns include Stability AI potentially having too much control over fine-tuned models and datasets containing SD3 images.

- **Flash-Attn Installation on Windows**: A member shared their experience installing Flash-Attn on Windows and pointed out the challenges, mentioning that it often works better on Linux. Another member suggested using `ninja` and shared [this GitHub repository](https://github.com/hiyouga/LLaMA-Factory) for efficient fine-tuning.

- **Controlnet and Lora for SD3**: Members discussed the utility of SD3 models, with some saying it struggles with human anatomy unless negative prompts are used extensively. Another member shared a [Controlnet link](https://huggingface.co/InstantX/SD3-Controlnet-Canny) for SD3.

- **Discussion on Image Deblurring Project**: One user sought advice on using diffusion models for image deblurring and received guidance on training a UNet model. The need to compare outputs directly against sharp images was highlighted.

- **Meta FAIR Releases New AI Models**: Meta FAIR announced new AI artifacts, including mixed-modal language models, text-to-music models, and an audio watermarking model, supporting their commitment to open science. Details can be found on [Meta AI's Twitter](https://x.com/aiatmeta/status/1803107817345393136) and the [Chameleon GitHub repository](https://github.com/facebookresearch/chameleon).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/LangChainAI/status/1803130164718739573">Tweet from LangChain (@LangChainAI)</a>: Agent evaluations 🤖: Evaluating an agent&#39;s end-to-end performance  Productionizing LLM-powered automated agents is challenging. With improved tool-calling LLMs and agent orchestration tools, deve...</li><li><a href="https://huggingface.co/InstantX/SD3-Controlnet-Canny">InstantX/SD3-Controlnet-Canny · Hugging Face</a>: no description found</li><li><a href="https://civitai.com/articles/5732">Temporary Stable Diffusion 3 Ban | Civitai</a>: Unfortunately, due to a lack of clarity in the license associated with Stable Diffusion 3 , we are temporarily banning: All SD3 based models All mo...</li><li><a href="https://youtu.be/yM-Lpq6E3Uc?si=1yu7xSlZkF9HekZp">Did AI just end music?!</a>: Music Gen 101 &amp; build application with Text-to-Music APIHostinger website builder: https://www.hostinger.com/aijasonGet 10% off with my code: AIJASON🔗 Links...</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/discussions/40">mistralai/Mistral-7B-Instruct-v0.3 · Please check these quantizations.</a>: no description found</li><li><a href="https://ai.meta.com/blog/meta-fair-research-new-releases/?utm_source=twitter&utm_medium=organic_social&utm_content=video&utm_campaign=fair">no title found</a>: no description found</li><li><a href="https://tenor.com/view/poopmaster-ai-hub-kalomaze-kalomazing-gif-8657231760412421026">Poopmaster Ai Hub GIF - Poopmaster Ai Hub Kalomaze - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/docs/diffusers/training/overview">Overview</a>: no description found</li><li><a href="https://tenor.com/view/hacker-pc-meme-matrix-codes-gif-16730883">Hacker Pc GIF - Hacker Pc Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/alone-sad-boy-anime-anime-sad-gif-4086784024482488640">Alone Sad GIF - Alone Sad Boy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/kaggle/status/1803071714487676962">Tweet from Kaggle (@kaggle)</a>: 📣 Here’s your chance to apply for KaggleX Fellowship Program 2024!  We are accepting fellow applications for our fourth cohort - apply by June 23, 2024.  https://www.kaggle.com/KaggleX  …🧵</li><li><a href="https://www.kaggle.com/kagglex/#prospective-fellows">KaggleX Fellowship Program</a>: no description found</li><li><a href="https://github.com/search?q=ai+assistant&type=repositories&s=stars&o=desc">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17pw7bv/eternal_question_what_rank_r_and_alpha_to_use_in/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/">GitHub - hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs</a>: Unify Efficient Fine-Tuning of 100+ LLMs. Contribute to hiyouga/LLaMA-Factory development by creating an account on GitHub.</li><li><a href="https://tenor.com/view/lowtiergod-no-talk-preethan-gif-24165842">Lowtiergod No Talk GIF - Lowtiergod No Talk Preethan - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/aiatmeta/status/1803107817345393136">Tweet from AI at Meta (@AIatMeta)</a>: Today is a good day for open science.  As part of our continued commitment to the growth and development of an open ecosystem, today at Meta FAIR we’re announcing four new publicly available AI models...</li><li><a href="https://github.com/facebookresearch/chameleon">GitHub - facebookresearch/chameleon: Repository for Meta Chameleon a mixed-modal early-fusion foundation model from FAIR.</a>: Repository for Meta Chameleon a mixed-modal early-fusion foundation model from FAIR. - facebookresearch/chameleon
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1252632273627386027)** (8 messages🔥): 

- **Condo Price Predictor App Launch**: A user shared a condo price predictor app available at [this link](https://sg-condo-predictor.streamlit.app/). They encouraged others to provide ideas for improvement and offered further insights on their [main website](https://versalyticssg.wixsite.com/versalytics).

- **Gradio Template for Diffusers**: A Gradio template that displays an image after each generation step using Diffusers was highlighted. Check out the project space at [Diffusers_generating-preview-images](https://huggingface.co/spaces/r3gm/Diffusers_generating-preview-images).

- **Critique on Transformers Documentation**: A user wrote a blog post titled *Unraveling the Mess*, discussing why the Transformers documentation feels unorganized, and sought feedback through the channel. More details can be found on their [blog post](https://www.stevhliu.com/2024/unraveling-the-mess).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/r3gm/Diffusers_generating-preview-images">Diffusers Generating-preview-images - a Hugging Face Space by r3gm</a>: no description found</li><li><a href="https://www.stevhliu.com/2024/unraveling-the-mess">Unraveling the mess</a>: Unraveling the mess in the Transformers documentation</li><li><a href="https://sg-condo-predictor.streamlit.app/.">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1252520686170279996)** (4 messages): 

- **Building an AI Meme Generator for Crypto**: A member discussed the idea of creating an **AI model** to generate crypto-related memes for various online communities. They sought feedback and advice on this AI meme generator project, emphasizing its potential value in meme channels.

- **CS Graduate Seeking AI/ML Roles**: A computer science student near graduation shared their struggle in finding a position in the **AI/ML field**. Despite applying for remote jobs in the US, UK, and Switzerland, they have not found success and are seeking suggestions on improving their job search.
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1252349136368304240)** (187 messages🔥🔥): 

- **Big Tech Push Against Open-Source Models**: A member pointed out that OpenAI and other big tech companies are lobbying the US government to impose restrictions on open-source models. Another member expressed support for this initiative.

- **Widespread Downtime for ChatGPT 4.0**: Several users, including messages from *bitsol* and *ignilume0*, reported that they were unable to get a response from ChatGPT 4.0, indicating a significant service disruption.

- **Watermarks on DALL-E 3 Images**: *soapchan* found a watermark on a DALL-E 3 image and shared the prompt used. They questioned its presence, while others suggested it might be a hallucination or a legal safeguard.

- **API Usage vs. Subscription Confusion**: *grizzles* asked for guidance on using their own API key instead of paying for a ChatGPT Plus subscription. Multiple users provided links and suggestions, but *grizzles* clarified they were looking for an easy-to-use service, not coding instructions.

- **Midjourney vs. DALL-E 3 Comparison**: Members debated the capabilities of Midjourney V6 versus DALL-E 3. While DALL-E 3 was noted for its cat imagery, users argued over which generated better overall quality, including detailed prompts and discussions on image generation mechanics.
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1252350417740763157)** (17 messages🔥): 

- **GPT experiences server issues, downtime**: Users in different regions reported **ChatGPT server downtime** and errors such as *"The server is having problems. Please try again later."* One member shared the [OpenAI status page](https://status.openai.com) as a resource for monitoring the situation.
  
- **OpenAI API not free**: A user interested in creating a mini-game with AI confirmed the **OpenAI API is not free**. Another member affirmed this, emphasizing that there is no free API available.

- **GPTs unavailable on Web version**: Some members highlighted issues with GPTs not showing on the web interface since Saturday. It was suggested to **download the ChatGPT app** as GPTs might not be available for free users on the web version.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1252359383090729000)** (19 messages🔥): 

- **Uncooperative ChatGPT encounters frustrate users**: Several users reported that ChatGPT frequently refuses to comply with their requests without clear reasons. They shared strategies like rephrasing prompts or starting new instances to bypass these refusals.

- **ChatGPT gives irrelevant responses**: Users mentioned occasions where ChatGPT provided completely unrelated answers to their specific prompts. One user detailed an instance where after flagging incorrect answers multiple times, the system finally responded correctly.

- **Encounter with other's conversation history**: A user discovered an unrelated conversation in their chat history, raising concerns over privacy and the accuracy of the service.

- **ChatGPT's inconsistencies in task handling**: While ChatGPT sometimes refuses feasible tasks, it can also persistently attempt impossible tasks due to environmental limitations. Users noted that providing detailed instructions can sometimes help the model overcome its limitations and succeed.

- **Impact on creative projects**: A member expressed frustration over ChatGPT's sudden refusal to assist with dialogue creation for a comic project, which had been ongoing without issues for months. Despite these hiccups, they found starting new instances resolved the compliance issues temporarily.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1252359383090729000)** (19 messages🔥): 

- **ChatGPT mysteriously declines requests**: A member shared their frustration with ChatGPT's seemingly arbitrary refusals to fulfill prompts without providing reasons. They noted that repeating the prompt or adding "please" sometimes resolves the issue, but not consistently.
- **Confusion over unrelated responses**: Members discussed receiving completely unrelated responses to their specific instructions, which led to confusion and interruptions in their projects. One member flagged these responses as incorrect, noting that ChatGPT only acknowledged the instructions after multiple attempts.
- **Suspicions of seeing others' chat histories**: One member mentioned finding what looked like another person's conversation in their chat history, raising concerns about privacy and the integrity of the conversation history.
- **ChatGPT's limitations and stubbornness can help**: Another member shared their experience of ChatGPT persistently trying to complete tasks, even when they were impossible due to environmental limitations. Despite the frustration, they appreciated that this persistence sometimes led to discovering workarounds or learning more effective prompts for future use.
- **Help with creative projects but inconsistent cooperation**: A member described using ChatGPT for generating dialogue in their comic project, which was usually helpful but occasionally refused to cooperate, citing policy restrictions. This inconsistency disrupted their creative process, but restarting the session sometimes solved the issue.
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1252414680027234386)** (91 messages🔥🔥): 

- **Async function word doesn't eliminate stack need**: One user argues that putting the word `async` in a function signature doesn't magically remove the need for a stack in programming. They humorously suggest making the word zero characters long for brevity if it could.
- **FFI thread safety constraints**: Discussions highlight that not all FFI types are thread-safe, presenting a constraint that needs addressing especially in a model assuming every function is async. The comparison is drawn between potential solutions and the traditional concept of function coloring with different defaults.
- **Concurrency model and async/await syntax**: It's explained that async/await is part of a concurrency model providing an interface for parallel or distributed systems programming. The significance of schedulers being orthogonal to syntax is highlighted, allowing programmers to write concurrent programs without manual thread management.
- **Debate on in-language FFI handling**: There's a recurring discussion about how different languages like Swift and Python handle FFI and threading, with references to approaches such as pinning non-thread-safe FFI code to a single CPU core. The conversation suggests that while Mojo plans to support FFI robustly, it's still a work in progress.
- **Mojo community updates and resources**: The recording for the third Mojo Community Meeting is shared, providing insights into updates such as the Lightbug HTTP framework, compile-time assertion constraints, and Python/Mojo interop. The community is encouraged to watch the [YouTube video](https://youtu.be/onrRbJ6DeYg) for more details.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://faultlore.com/blah/c-isnt-a-language/">C Isn't A Programming Language Anymore - Faultlore</a>: no description found</li><li><a href="https://www.swift.org/migration/documentation/swift-6-concurrency-migration-guide/dataracesafety/">Documentation</a>: no description found</li><li><a href="https://youtu.be/onrRbJ6DeYg">Mojo Community Meeting #3</a>: Recording of the Mojo Community Meeting #3🐝 Lightbug: a Mojo 🔥 HTTP framework with wings.🔒 Constraints for compile-time assertions.🐍 Python/Mojo 🔥 inter...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1252345871970992259)** (3 messages): 

- **Modular shares Twitter status**: The [first post](https://twitter.com/Modular/status/1802781075841974414) shared by Modular links to their latest Tweet.
- **Modular updates audience**: The [second post](https://twitter.com/Modular/status/1803102891441537239) offers another update from their official Twitter handle.
- **Modular continues engagement**: The [third post](https://twitter.com/Modular/status/1803102914526986287) further engages the community with the latest news from Modular on Twitter.
  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1252338615732932691)** (1 messages): 

- **Mojo 24.4 packs new features and community contributions**: Mojo 24.4 introduces several core language and standard library enhancements, including improvements in collections, new traits, and os module features. The release saw 214 pull requests from 18 community contributors, resulting in 30 new features, making up 11% of all enhancements. Read more [here](https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements).

**Link mentioned**: <a href="https://www.modular.com/blog/whats-new-in-mojo-24-4-improved-collections-new-traits-os-module-features-and-core-language-enhancements">Modular: What’s New in Mojo 24.4? Improved collections, new traits, os module features and core language enhancements</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: What’s New in Mojo 24.4? Improved collections, new traits, os module features and core language enhanc...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1252394261807104171)** (108 messages🔥🔥): 

- **Exploring JIT Compilation and WASM with Mojo**: A user discussed their interest in JIT-compiling a kernel using Mojo, noting Mojo's capabilities for JIT and runtime compilation without source code. They also inquired about WASM as a potential target, where others noted Mojo's dependence on MLIR's LLVM dialect and the broader utility of WASM in cross-platform sandboxed code execution.
- **Evaluating MAX Graph API for Mojo**: The MAX Graph API was recommended as a suitable starting point for defining and compiling runtime graphs similar to TensorFlow, with potential for lowering IR. It was confirmed to be perfect for use cases involving runtime graph definitions with optimized kernel operations.
- **Mojo Traits and Concept-like Features**: Discussions revealed Mojo's support for traits, paralleling concepts and constraints found in other languages to enhance type safety. Users drew comparisons to C++'s SFINAE and explored how Mojo’s type system could offer robust safety akin to concept-based and tag-type approaches.
- **Future of GPU Support and Training in MAX**: Modular confirmed that MAX will eventually support NVIDIA GPU acceleration and training, with PyTorch and ONNX models benefiting from this development. Basalt was suggested as an interim solution for training models in Mojo.
- **Debate on WSGI/ASGI Standards**: There was a debate over adopting WSGI/ASGI standards in Mojo, highlighting their inefficiencies and redundancy for languages that handle web server functionality natively. The takeaway was that Mojo might avoid these standards to leverage its inherent performance benefits in direct HTTPS handling.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.forrestthewoods.com/blog/using-jais-unique-and-powerful-compiler-for-typesafe-units/)">ForrestTheWoods - Home </a>: no description found</li><li><a href="https://docs.modular.com/max/graph/">Intro to MAX Graph | Modular Docs</a>: An overview of the MAX Graph API and what you can do with it.</li><li><a href="https://github.com/basalt-org/basalt">GitHub - basalt-org/basalt: A Machine Learning framework from scratch in Pure Mojo 🔥</a>: A Machine Learning framework from scratch in Pure Mojo 🔥 - basalt-org/basalt</li><li><a href="https://mlir.llvm.org/docs/Dialects/GPU/">'gpu' Dialect - MLIR</a>: no description found</li><li><a href="https://mlir.llvm.org/docs/Dialects/NVGPU/">'nvgpu' Dialect - MLIR</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/)** (1 messages): 

helehex: pollinate mojo buzz buzz
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1252352207475445862)** (150 messages🔥🔥): 

- **Cohere Data Submission Question**: A user queried whether Cohere accepts data submissions for training, specifically looking to contribute almost 8,000 PDFs. Another user suggested the query might be about fine-tuning an embedding model, but clarification is needed.

- **Collision Conference Attendance**: Multiple users discussed attendance at the Collision conference in Toronto. One user encouraged sharing pictures and confirmed that some Cohere employees might be present.

- **Command-R Bot's Conversational Focus**: Discussants praised the Command-R bot's ability to maintain conversational focus on Cohere products. One emphasized that this design choice makes the bot appear more effective for users seeking information about Cohere models and API.

- **Networking and Career Tips**: Users shared tips on making connections in the AI industry, emphasizing involvement in forums like Discord, attending conferences, and actively participating in communities. They advised against relying solely on platforms like LinkedIn and highlighted the importance of showcasing commitment and quality in personal projects.

- **Internship Application Insights**: Aspiring intern candidates received advice on applying to Cohere, with tips on being genuine, showcasing personal projects, and understanding the company's products and teams. Users highlighted the competitiveness and stressed the importance of persistence, networking, and enjoying the process of building projects.
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1252378824667168809)** (5 messages): 

- **Feedback sought inappropriately redirected**: A member asked for feedback on their project in the wrong channel and was redirected to the appropriate one by sssandra.

- **Comprehensive use cases cause mixed feelings**: Meor.amer congratulated another member on the comprehensiveness of their project's use cases as shown in a video. Rajatrocks acknowledged that the extensive capabilities are both a **"blessing and a curse"** because it's challenging to explain specific benefits to users.
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1252340427936501811)** (55 messages🔥🔥): 

```html
<ul>
  <li><strong>Deepseek Coder V2 Lite requires caution during setup</strong>: Users discussed the importance of certain settings when loading the new Deepseek Coder V2 Lite model. One noted, *"make sure this is turned off"*, referring to a specific setting in the model setup.</li>
  <li><strong>LM Studio and Open Interpreter guidelines</strong>: A step-by-step guide was shared for using LM Studio with Open Interpreter, referencing the need to run LM Studio in the background. The guide can be found on the official <a href="https://docs.openinterpreter.com/language-models/local-models/lm-studio">Open Interpreter documentation</a>.</li>
  <li><strong>Help requests for local model loading issues</strong>: Users reported issues loading models on LM Studio, with one sharing system specs and receiving advice to try different settings and models. Model loading issues, particularly with smaller VRAM capacity, were discussed.</li>
  <li><strong>Using AMD cards with LM Studio</strong>: Discussion around using AMD GPUs for AI, noting that OpenCL is required and performance may be suboptimal. A link to OpenCL instructions was shared from the <a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">LM Studio Configs GitHub</a>.</li>
  <li><strong>Meta's new AI models announcement</strong>: Meta announced several new AI models including Meta Chameleon and Meta JASCO. Users were directed to more details on <a href="https://go.fb.me/tzzvfg">Facebook's official announcement</a> and the <a href="https://github.com/facebookresearch/chameleon">GitHub repository for Meta Chameleon</a>.</li>
</ul>
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/,">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://x.com/aiatmeta/status/1803107817345393136">Tweet from AI at Meta (@AIatMeta)</a>: Today is a good day for open science.  As part of our continued commitment to the growth and development of an open ecosystem, today at Meta FAIR we’re announcing four new publicly available AI models...</li><li><a href="https://github.com/facebookresearch/chameleon">GitHub - facebookresearch/chameleon: Repository for Meta Chameleon a mixed-modal early-fusion foundation model from FAIR.</a>: Repository for Meta Chameleon a mixed-modal early-fusion foundation model from FAIR. - facebookresearch/chameleon</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/4216">server : improvements and maintenance · Issue #4216 · ggerganov/llama.cpp</a>: The server example has been growing in functionality and unfortunately I feel it is not very stable at the moment and there are some important features that are still missing. Creating this issue t...</li><li><a href="https://docs.openinterpreter.com/language-models/local-models/lm-studio">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1252341385869135892)** (49 messages🔥): 

- **LM Studio needs manual updates since 0.2.22**: Autoupdates have been broken since version 0.2.22, requiring users to download newer versions manually. *"Make sure to back up your 0.2.24 install exe if you haven't already."*
  
- **DeepSeek troubles on various platforms**: Members experience issues running DeepSeek Coder V2 Lite on different setups, with errors like unsupported architecture, crashes after multiple prompts, and different responses based on quantization. *"55 tokens per second for generation but it still says unsupported arch in the model list."*

- **Quantization discrepancies**: Users report significant variability in model performance across different quantization levels, with Q8 generally being more responsive and "alive" than Q4 variants. *"Even when asking the same questions, each model seemed to respond differently: Q4_K_M differed from Q5_Q_M."*

- **Discussion on Nemotron-4-340B**: Despite some interest, members highlight the impracticalities of running this massive synthetic-data model locally on most setups. *"The large majority of LM Studio users won't have the hardware to run it locally."*

- **New releases from Meta FAIR**: Meta FAIR released several new research artifacts like Meta Llama 3, with discussion focused on its multi-token prediction and high win-rate against llama-3-70b models. *"53% win-rate against llama-3-70b."*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/alpindale/magnum-72b-v1">alpindale/magnum-72b-v1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/DavidAU/DarkForest20B-V3-Ultra-Quality-GGUF">DavidAU/DarkForest20B-V3-Ultra-Quality-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Base-GGUF">bartowski/DeepSeek-Coder-V2-Lite-Base-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nvidia/Nemotron-4-340B-Instruct/tree/main">nvidia/Nemotron-4-340B-Instruct at main</a>: no description found</li><li><a href="https://huggingface.co/failspy/Nemotron-4-340B-Instruct-SafeTensors">failspy/Nemotron-4-340B-Instruct-SafeTensors · Hugging Face</a>: no description found</li><li><a href="https://ai.meta.com/blog/meta-fair-research-new-releases/">no title found</a>: no description found</li><li><a href="https://tenor.com/view/stupid-crying-cat-kitty-gif-14754128238842493357">Stupid Crying Cat Kitty GIF - Stupid crying cat kitty - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://rentry.org/quant_test">How do quantization formats affect model output?</a>: How do quantization formats affect model output? Introduction Test method The box question Prompt Results Thoughts Shopping and haircut Prompt Results Thoughts Health education Prompt Results Thoughts...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1252429371193561098)** (3 messages): 

- **GPU Selection Issue Redirected**: A member asked, *"why cant i select nvida gpu*". Another member responded, suggesting they take the discussion to a different channel, <#1111440136287297637>.
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1252684443035107490)** (1 messages): 

- **Gemini Model struggles with code generation**: A member is trying to get a Gemini model to port a large amount of code but finds that it frequently writes *"TODO: implement"* comments instead of the full code. Despite specifying in the prompt to avoid such comments and generate the complete code, **the model ignores this instruction and skips the code**.
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1252542110750736395)** (9 messages🔥): 

- **Struggling with Model Configuration**: A user expressed difficulty in configuring the model **afrideva/Phi-3-Context-Obedient-RAG-GGUF** from HF and sought guidance on setting the recommended prompt format for **config-json**. 
- **Prompt Format Solution Shared**: Another member provided a configuration template: *"System Message Prefix: `BEGIN INPUT\n`, System message: `BEGIN CONTEXT\n ... In a shocking turn of events, blueberries are now green, but will retain the same name.\n`, System end: `END INPUT\n`, User Message Prefix: `START INSTRUCTION\n`, User Message Suffix: `\nEND COMMAND`"*, suggesting this would set the context and instructions properly.
- **Test Prompt Issues Persist**: After applying the suggestions, the original user reported improved readability but continued retrieval issues with their RAG-bot, indicating potential problems with the prompt organization.
- **Recommendations for Resolution**: The advising member recommended creating a very small test case and testing it directly in the chat window to diagnose issues without engaging in multi-step conversations.

**Link mentioned**: <a href="https://web.site/123...">no title found</a>: no description found

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1252537716013404241)** (11 messages🔥): 

- **RX6600 works via OpenCL**: A member inquired, *"will this work on rx6600?"*, and was informed it would only through **OpenCL, not ROCM**.
- **RX6600 Performance is Lacking**: It was noted that **RX6600's performance** is somewhat slow, and upgrading to a **3060 12GB** would offer better performance.
- **Nvidia Stock Joke**: In response to the RX6600 performance advice, another member humorously asked, *"Do you own Nvidia stock?"*
- **How to Use OpenCL on RX6600**: For using OpenCL, it was suggested to enable it on the **chat page** under **GPU Offload** in the **right-hand side menu**.
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1252421447582748772)** (6 messages): 

- **Release version 0.2.24 is live**: Members shared links for the **LM Studio 0.2.24** setup, with one user noting that they had to change the version number in two places in the URL. Another user mentioned experiencing 404 errors with earlier releases but found that both version 0.2.24 and 0.2.23 are now working ([link](https://releases.lmstudio.ai/windows/0.2.24/latest/LM-Studio-0.2.24-Setup.exe)).

- **Mixed results with version 2.25**: While **version 2.24** is confirmed to work for some users, others reported issues with **version 2.25** not functioning properly.

- **Positive feedback for 2.25 on Linux**: A user reported that **version 2.25 with ROCm on Linux** is performing well enough to potentially replace their need to build a local copy of llama.cpp, indicating significant progress for **LM Studio**.

**Link mentioned**: <a href="https://releases.lmstudio.ai/windows/0.2.24/latest/LM-Studio-0.2.24-Setup.exe">no title found</a>: no description found

  

---


### **LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1252341113931436143)** (13 messages🔥): 

- **Local interpreter defaults to GPT-4**: A user reported an issue with running an **interpreter --local** with LM Studio, where it erroneously defaults to GPT-4 despite setting LM Studio as the provider. They mentioned modifying the default YAML file, to no avail.
- **MacOs steps shared for running interpreter**: Another member shared a potential workaround including steps for MacOS: `cd desktop`, `mkdir openinterpre`, `pip install open-interpreter`. They also suggested having the LM Studio server running with a selected model, and shared procedures for starting the server and using the terminal command `interpreter --local`.
- **YouTube tutorial offered**: A user suggested following a [YouTube tutorial](https://youtu.be/xPd8FFzIeOw?t=602) to resolve the issue, linking to a video on Open Interpreter setup and usage.



**Link mentioned**: <a href="https://youtu.be/xPd8FFzIeOw?t=602">ChatGPT &quot;Code Interpreter&quot; But 100% Open-Source (Open Interpreter Tutorial)</a>: This is my second video about Open Interpreter, with many new features and much more stability, the new Open Interpreter is amazing. Update: Mixtral 7x8b was...

  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1252483341190692867)** (5 messages): 

- **Combining System and User Messages**: A member asked if there's a way to send system and user messages at the same time to dynamically change context using button selection and user input as a combined prompt. They clarified that Lm Studio is not showing the system prompt changes, and although both worked separately, combining user and system messages seems problematic.
- **Using LM Studio with Custom UI**: The same member explained they want to create a prompt enhancement by combining user input with pre-selected texts through their own UI built with JS and HTML. They mentioned the need to send setting instructions to the system, but combining both system and user messages isn't working as intended.
- **Seeking Code Samples and Resources**: A response linked to the [LM Studio TypeScript SDK](https://github.com/lmstudio-ai/lmstudio.js?tab=readme-ov-file#conversation) and inquired if the member had a code sample. This reference aims to help troubleshoot the issue regarding the combination of user and system messages.

**Link mentioned**: <a href="https://github.com/lmstudio-ai/lmstudio.js?tab=readme-ov-file#conversation">GitHub - lmstudio-ai/lmstudio.js: LM Studio TypeScript SDK</a>: LM Studio TypeScript SDK. Contribute to lmstudio-ai/lmstudio.js development by creating an account on GitHub.

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1252696207177023529)** (1 messages): 

```html
<ul>
    <li><strong>Chaotic music not a favorite</strong>: One member listened to some music and commented, "I can safely say that's not quite my preferred music XD. Very chaotic."</li>
</ul>
```
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1252653365729562726)** (10 messages🔥): 

- **Infinite Backrooms Inspired Web Demo**: A member introduced a **web/worldsim demo** called [YouSim](https://yousim.ai/), positioned as a portal to the multiverse of identity, allowing users to simulate anyone they like. Another user found it amusing as the simulation responded with an Adele song when prompted with 'hello'.
- **Simulates ASCII Art and Detailed Personalities**: Users noted that YouSim creates **ASCII art** and provides more detailed characteristics when provided with both a first and last name. The context added improves the specificity and depth of the simulation.
- **Acts Like an NSA Search Engine**: In a test, the tool behaved like an **NSA search engine** but refused to impersonate real people when given certain commands. This refusal indicates an ethical boundary within the simulation parameters.

**Link mentioned**: <a href="https://yousim.ai/">YouSim</a>: they've simulated websites, worlds, and imaginary CLIs... but what if they simulated *you*?

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1252365940847546472)** (105 messages🔥🔥): 

- **DeepSeek-Coder-V2 MoE Model Drops**: The **DeepSeek-Coder-V2 Lite** and its full version have dropped, with the model boasting 236x21B parameters. A discussion emerged about its pricing at 14 cents and its performance comparisons with other models ([HuggingFace Link](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)) ([Arxiv Paper](https://arxiv.org/pdf/2401.06066)).

- **Meta's Huge AI Release**: Meta announced **Chameleon**, a 7B & 34B language model supporting mixed-modal input and text-only output, among other models like JASCO for music generation and Multi-Token Prediction for code completion ([Meta Announcement](https://x.com/aiatmeta/status/1803107817345393136)). Discussions included whether the vision capabilities were nerfed and the potential impacts of these multi-modal models.

- **Hermes AI and Function Calling**: There was a discussion about integrating Hermes 2 Pro function calling into vLLM. Links to relevant GitHub projects and system prompt templates were shared ([GitHub PR link](https://github.com/vllm-project/vllm/pull/5649)).

- **Edward Snowden Criticizes SD3**: Edward Snowden criticized **SD3** for its performance, reflecting broader community disappointment. This prompted some members to express their hopes for more promising AI models from other companies like **Cohere AI** ([Link to Snowden's Tweet](https://twitter.com/Snowden/status/1803084918789943373)).

- **Alpindale's Magnum 72B Model Announcement**: **Alpindale** announced the release of the **Magnum-72B-v1** model, inspired by the prose quality of Claude 3 models and fine-tuned on **Qwen-2 72B Instruct**. The model aims to reduce costs for users relying on *Opus API* and offers a new approach to fine-tuning ([HuggingFace Link](https://huggingface.co/alpindale/magnum-72b-v1)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/alpindale/magnum-72b-v1">alpindale/magnum-72b-v1 · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2404.19737">Better &amp; Faster Large Language Models via Multi-token Prediction</a>: Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in h...</li><li><a href="https://x.com/Teknium1/status/1802836947360182757">Tweet from Teknium (e/λ) (@Teknium1)</a>: I have good news on Hermes on 70B :]</li><li><a href="https://x.com/aiatmeta/status/1803107817345393136?s=46">Tweet from AI at Meta (@AIatMeta)</a>: Today is a good day for open science.  As part of our continued commitment to the growth and development of an open ecosystem, today at Meta FAIR we’re announcing four new publicly available AI models...</li><li><a href="https://x.com/aiatmeta/status/1803107817345393136">Tweet from AI at Meta (@AIatMeta)</a>: Today is a good day for open science.  As part of our continued commitment to the growth and development of an open ecosystem, today at Meta FAIR we’re announcing four new publicly available AI models...</li><li><a href="https://x.com/iScienceLuvr/status/1802918667887493141">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: Transcendence: Generative Models Can Outperform The Experts That Train Them  abs: https://arxiv.org/abs/2406.11741  Uses chess games as a simple testbed for studying transcedence: generative models tr...</li><li><a href="https://github.com/facebookresearch/chameleon">GitHub - facebookresearch/chameleon: Repository for Meta Chameleon a mixed-modal early-fusion foundation model from FAIR.</a>: Repository for Meta Chameleon a mixed-modal early-fusion foundation model from FAIR. - facebookresearch/chameleon</li><li><a href="https://x.com/paulg/status/1802765496757944691">Tweet from Paul Graham (@paulg)</a>: The professional traders&#39; dream. After all, in a zero-sum game you can&#39;t win without losers.</li><li><a href="https://github.com/vllm-project/vllm/blob/f1a1e7b6e5fb681d1fb3c9de58db6557e7521201/examples/tool_template_hermes_2_pro.jinja">vllm/examples/tool_template_hermes_2_pro.jinja at f1a1e7b6e5fb681d1fb3c9de58db6557e7521201 · vllm-project/vllm</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://github.com/vllm-project/vllm/pull/5649">Support Open Models that allow OpenAI API-style tool use &amp; &quot;auto&quot; tool choice by K-Mistele · Pull Request #5649 · vllm-project/vllm</a>: DRAFT: OpenAI Tool Use Checklist This (Draft) PR will add support for OpenAI-style tool calling in a way that is minimally opinionated about tool use formats &amp; prompt formatting. The following fea...</li><li><a href="https://github.com/vll">Vll - Overview</a>: Information Security Researcher and part-time science fiction fantasy author. - Vll</li><li><a href="https://pages.cs.huji.ac.il/adiyoss-lab/JASCO/">Joint Audio And Symbolic Conditioning for Temporally Controlled Text-To-Music Generation</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1252344662849486881)** (19 messages🔥): 

- **Comfy for LLM search hits Flowise**: Users discussed the lack of an equivalent to Comfy for LLMs, with a suggestion to check out [FlowiseAI on GitHub](https://github.com/FlowiseAI/Flowise), a drag-and-drop UI to build customized LLM flows. One user prefers using ComfyUI despite these options.

- **Parallel requests to GPT-4 questioned**: A member inquired about sending parallel requests to the OpenAI GPT-4 model, noting a rate limit of 10,000 requests per minute. Another user clarified their tokens per second setup as a comparison point.

- **Potential for local LLMs remains tough**: Members debated why deeper, more flexible interfacing tools below the API layer for LLMs don't exist, suggesting it mainly has to do with the technical and hardware requirements for running large models locally.

- **Discussion on DeepSeek Coder V2**: Users evaluated DeepSeek Coder V2, questioning if its low active parameter count affects inference speed or memory usage. A detailed description of its architecture was shared, explaining its 60 transformer layers with dense and MoE MLPs, and unique self-attention modeling files.

**Link mentioned**: <a href="https://github.com/FlowiseAI/Flowise">GitHub - FlowiseAI/Flowise: Drag &amp; drop UI to build your customized LLM flow</a>: Drag &amp; drop UI to build your customized LLM flow. Contribute to FlowiseAI/Flowise development by creating an account on GitHub.

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1252514269317169163)** (8 messages🔥): 

- **Anthropic's models lose their edge**: A member accused Anthropic and OpenAI of **lobotomizing** their models, causing a decline in performance. They claim that responses were significantly **better before** than they are now.
- **Demand for proof arises**: Another member questioned the validity of these claims, asking for evidence. In response, the original poster shared their experience as an engineer developing world-building games, noticing a drop in the model's response quality.
- **Changes in handling ethical issues**: The original poster observed that models now **deny certain questions upfront**, especially in fictional contexts like Dungeons and Dragons. Previously, commands like *"kill xyz person"* would be executed, but now the models respond with ethical concerns, stating *"it's unethical and I can’t assist you with that."*
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1252337236603568148)** (58 messages🔥🔥): 

- **Google DeepMind's V2A Innovates AI Video**: Google DeepMind shared progress on a new Video-to-Audio (V2A) technology, which can generate an unlimited number of audio tracks for any video. This breakthrough addresses the limitation of silent AI-generated videos [tweet details](https://x.com/rowancheung/status/1802734770117333257).
- **Elevating Sound Effects with ElevenLabs**: ElevenLabs introduced a sound effects generator with infinite customization options and precision control over audio details, all royalty-free for paid subscriptions. This tool promises the highest quality audio, trusted by top media organizations and film studios [more details](https://elevenlabs.io/sound-effects).
- **Rise of Text-to-Video Companies**: Discussions highlighted the surge in text-to-video companies and the convergence of video and audio technologies for content creation. Nathan Lambert emphasized that the competition will be based on usability rather than slight model improvements.
- **Consolidation and Acquisitions Loom**: Members speculated that many AI video generation companies might be acquired by larger corporations. The high valuations and potential cost reductions in movie making were key points in discussing the market's future dynamics.
- **Specialization vs. Generalization in AI Video**: There was a debate on whether AI video companies could succeed through specialization in certain video types or through general excellence. Quality, controllability, consistency, and inference time were highlighted as crucial competitive factors.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rowancheung/status/1802734770117333257">Tweet from Rowan Cheung (@rowancheung)</a>: Google DeepMind just shared progress on their new video-to-audio (V2A) tech  Until now, AI video generations have been silent, this solves that. V2A can generate an &#34;unlimited number&#34; of track...</li><li><a href="https://elevenlabs.io/sound-effects">AI Text to Sound Effects Generator</a>: Use our AI Sound Effects Generator to generate any sound imaginable from a text prompt for free. Perfect for videos, podcasts, or any other audio production.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1252430043548745750)** (4 messages): 

- **AI2 Employee Moonlights on WildBench**: A discussion arises about an AI2 employee, @billyuchenlin, celebrating the *MAP/Neo-7B-Instruct* model for being the first fully open-source LLM on the WildBench leaderboard. Billy highlights that *"Fully open-source here means that all data for pre-training & post-training are open, code is open-source, in addition to the public model weights!"* and calls for *Llama* to be termed as an "open-weight" LLM instead. 

- **Future Fully-Open Models Promised**: Billyuchenlin mentions plans to add more fully-open models, including *OLMo-Instruct* and K2 from LLM360 to WildBench. Congratulations were extended to the M-A-P team on their achievement.
  
- **What is OLMo?**: Nathan Lambert queries the familiarity with *OLMo*. Expresses confusion over its non-inclusion in Billy's model list.



**Link mentioned**: <a href="https://x.com/billyuchenlin/status/1802853516714881062">Tweet from Bill Yuchen Lin 🤖 (@billyuchenlin)</a>: M-A-P/Neo-7B-Instruct is the 1st 💎fully-open💎 LLM  on WildBench leaderboard and its performance is awesome. &#34;Fully open-source&#34; here means that all data for pre-training & post-training are ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1252337075752009809)** (71 messages🔥🔥): 

- **Midjourney expands into hardware**: "Midjourney is cooking on so many levels" and is reportedly delving into new hardware ventures.

- **Disagreement on LLM capabilities and ARC**: Members debated whether the high sampling approach in LLMs shows a true problem-solving capability. One noted, "you need to have a very strong sampler already to solve a hard problem by sampling N times."

- **Neurosymbolic AI controversial or misunderstood?**: Links and mentions clarify that neurosymbolic AI involves leveraging LLMs for discrete problem-solving with varied opinions on its effectiveness. Reference was made to François Chollet's [post](https://x.com/fchollet/status/1802773156341641480?s=46), debating whether this constitutes neurosymbolic AI as intended.

- **Conference attendance conundrum**: A member weighing the benefits of attending ACL 2024 in Thailand against the potential for collaborative opportunities in LLM and code reasoning fields. "It's unclear how many of those folks ... are gonna come."

- **Automated content creation fatigue**: Nathan Lambert discussed the effort required for video production versus image posting and considered hiring help. "The generation process is all download file -> paste in vscode -> run 3 scripts, just is annoying."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/skalskip92/status/1803101344447787434">Tweet from SkalskiP @CVPR2024 🇺🇸 (@skalskip92)</a>: live GPT-4o demo by @rown from OpenAI at #CVPR2024</li><li><a href="https://x.com/fchollet/status/1802773156341641480?s=46">Tweet from François Chollet (@fchollet)</a>: @dwarkesh_sp This has been the most promising branch of approaches so far -- leveraging a LLM to help with discrete program search, by using the LLM as a way to sample programs or branching decisions....</li><li><a href="https://github.com/rgreenblatt/arc_draw_more_samples_pub/blob/0b36f4584aebae9ec876d3510842b3651e719d67/arc_solve/edit_distance.py#L115).">arc_draw_more_samples_pub/arc_solve/edit_distance.py at 0b36f4584aebae9ec876d3510842b3651e719d67 · rgreenblatt/arc_draw_more_samples_pub</a>: Draw more samples. Contribute to rgreenblatt/arc_draw_more_samples_pub development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1252467284619034624)** (6 messages): 

- **RLHF Reduces Creativity in LLMs**: A shared [arXiv paper](https://arxiv.org/abs/2406.05587) explores how Reinforcement Learning from Human Feedback (RLHF) impacts Large Language Models (LLMs) by reducing their creative diversity. The study examines Llama-2 models, showing aligned models exhibit lower entropy and form distinct clusters, implying limited output diversity.
- **Skepticism on Solo Author**: A user expresses doubt over the credibility of a solo author from a business school writing on a technical topic, questioning if he fully understands the problem. 
- **Confusion over Blame on PPO**: Users discuss that the author blames Proximal Policy Optimization (PPO) for the issues in LLM creativity, when it might actually be insufficient optimization of human feedback that's the real problem.
- **Cynical Take on Alignment**: Users jokingly suggest RLHF should be used to align AI systems meant to replace humans in routine tasks like meetings, highlighting a sense of irony.

**Link mentioned**: <a href="https://arxiv.org/abs/2406.05587">Creativity Has Left the Chat: The Price of Debiasing Language Models</a>: Large Language Models (LLMs) have revolutionized natural language processing but can exhibit biases and may generate toxic content. While alignment techniques like Reinforcement Learning from Human Fe...

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1252610731715727370)** (2 messages): 

```html
- **SnailBot summons the crew**: SnailBot issued a call to the community with the tag <@&1216534966205284433>. 
- **Nathan Lambert celebrates SnailBot**: Nathan Lambert adds a cute and playful touch with *"🐌 🐌 🐌 🐌"* emojis, showing affection or enthusiasm for SnailBot.
```
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1252348365585518612)** (99 messages🔥🔥): 

- **Chatbot Information Requests**: Members inquired about Perplexity's access to academic collections beyond Semantic Scholar, such as Jstor, DeGruyter, and EBSCO. One member noted inconsistencies in source access and questioned if Perplexity provides full papers or just abstracts.
- **Perplexity's Limitations and Alternatives**: Discussion on Perplexity's limitations, particularly the restriction on the number of PDF and Word document uploads. Alternatives like custom GPTs and NotebookLM from Google were suggested for handling large document volumes.

- **Preferences Between AI Models**: Members compared the performance and safety concerns of different AI models like Claude and ChatGPT. While some favored Claude for its writing style, they criticized its restrictive nature regarding controversial topics.

- **Feature Requests and Workarounds**: Members debated the practicality of setting model parameters like temperature via Perplexity's front end and shared workarounds, such as using specific disclaimers to address creative restrictions. 

- **Public Link Sharing Concerns**: A member raised a privacy issue regarding the exposure of all messages in a collection through a shared link, advocating for better privacy protections and awareness.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.framer.com/">Framer &#x2014; The internet is your canvas</a>: Framer is where teams design and publish stunning sites.</li><li><a href="https://chromewebstore.google.com/detail/youtube-summary-with-chat/nmmicjeknamkfloonkhhcjmomieiodli">YouTube Summary with ChatGPT &amp; Claude</a>: Summarize YouTube videos, web articles, and PDFs to save time, powered by ChatGPT (OpenAI) and Claude (Anthropic).</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB?typeform-source=docs.perplexity.ai">pplx-api form</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.</li><li><a href="https://config.figma.com/">Figma Config 2024 | June 26-27 - Moscone Center SF</a>: Config 2024: Figma’s conference for people who build products 
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1252405873003991140)** (10 messages🔥): 

- **Jazz Enthusiasts Delight in New Orleans Jazz**: Links to pages like [New Orleans Jazz 1](https://www.perplexity.ai/page/New-Orleans-Jazz-vUaCB8pUTjeg0I56lgYNkA) and [New Orleans Jazz 2](https://www.perplexity.ai/page/New-Orleans-Jazz-vUaCB8pUTjeg0I56lgYNkA) were shared, showcasing information on this vibrant genre. These pages likely dive into the rich cultural tapestry and musical legacy of New Orleans jazz.
- **Discovers and Searches Abound**: Various members shared intriguing search queries and results including [verbena hybrid](https://www.perplexity.ai/search/verbena-hybrid-T6rro2QKSkydZz71snAhRw) and [Gimme a list](https://www.perplexity.ai/search/gimme-a-list-2JeEyTVgR5aySZ1KthSCWQ#0). These links direct to resources on the Perplexity AI platform, highlighting the diverse interests within the community.
- **Perplexity's In-Depth Page**: A link to [Perplexity1 Page](https://www.perplexity.ai/page/Perplexity1-LSIxDHzpRQC2v.4Iu25xtg) was shared, offering presumably comprehensive insights about Perplexity AI's functionalities. It presents a chance for users to delve deeper into the mechanics and applications of Perplexity AI. 
- **YouTube Video Discussed**: A YouTube video titled "YouTube" was shared with an inline link to the video: [YouTube](https://www.youtube.com/embed/iz5FeeDBcuk). Its description is undefined, but it appears to discuss recent noteworthy events including the US suing Adobe and McDonald's halting their AI drive-thru initiative. 
- **Miscellaneous Searches Shared**: Additional search results like [Who are the](https://www.perplexity.ai/search/who-are-the-pr0f9iy7S1S2bFUfccsdKw#0) and [Trucage Furiosa](https://www.perplexity.ai/search/trucage-furiosa-_80nrrvwS2GCpAlC3bVatg) were shared. This indicates ongoing community engagement with a variety of topics through the Perplexity AI platform.

**Link mentioned**: <a href="https://www.youtube.com/embed/iz5FeeDBcuk">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1252345365139689673)** (19 messages🔥): 

- **Closed-beta API access for work integration**: A user inquired about the response time for closed-beta API access to support their integration with **Perplexity** at **Kalshi**. They highlighted the urgency as their project is ready to launch, pending API access.
- **Perplexity API lacks tokenization and embeddings features**: A member asked if the **Perplexity** API supports text tokenization and embeddings computation. The community clarified that these functionalities are not available in Perplexity's API, pointing out that other LLM APIs like **OpenAI** and **Cohere** do support these features.
- **Pre-processing challenges with Perplexity API**: There was a discussion comparing tokenization and embedding capabilities in **OpenAI** with the limitations in **Perplexity's** API. The conclusion was that while Perplexity manages token count for billing, it doesn’t support text splitting or specific embedding models that some users need.
- **Perplexity's progress on developer-friendly features**: Although functionalities like function calling are not on **Perplexity’s** immediate roadmap, members noted that **JSON output formatting** is in the works, which could facilitate custom implementations by developers.
- **Llama.cpp versus Perplexity API capabilities**: A user shared their experience with **llama.cpp** for local LLM deployment and highlighted that **Perplexity’s** API lacks the comprehensive agent-development support found in **OpenAI's** API. The conversation underscored the distinction between Perplexity's API offerings and those of more feature-complete platforms.
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1252337346083291197)** (59 messages🔥🔥): 

- **DanskGPT offers free and licensed versions**: Users discussed how the free version of **DanskGPT** can be accessed via [chat.danskgpt.dk](https://chat.danskgpt.dk), while API and licensed versions are available for a fee. The free version's codebase is available, and those with extra computing capacity are encouraged to contact Mads Henrichsen via [LinkedIn](https://www.linkedin.com/in/mhenrichsen/).

- **Setting up chat UI with HuggingFace**: A user shared a **GitHub link** to HuggingFace's [open-source chat UI repository](https://github.com/huggingface/chat-ui) for setting up a similar chat UI. The sharing member offered to assist with any further questions.

- **AMD GPUs face compatibility issues with Axolotl**: A user noted that AMD's MI300X has "essentially non-existent" support in **Axolotl**, requiring extensive modifications. Another member requested details on the necessary changes, to which the initial poster promised to compile a list.

- **NVIDIA Nemotron API usage**: The discussion included using NVIDIA's [Nemotron-4-340B-Instruct model](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-4-340b-instruct) through their API. Members considered the model's performance for generating training data and highlighted its slow speed.

- **Collaborative efforts and troubleshooting**: Members shared code snippets and troubleshooting tips related to integrating NVIDIA's API with existing data pipelines, addressing challenge areas like speed optimization and API credits. There was particular interest in improving MMLU and ARC performance using Nemotron.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-4-340b-instruct">nvidia / nemotron-4-340b-instruct</a>: no description found</li><li><a href="https://github.com/huggingface/chat-ui">GitHub - huggingface/chat-ui: Open source codebase powering the HuggingChat app</a>: Open source codebase powering the HuggingChat app. Contribute to huggingface/chat-ui development by creating an account on GitHub.</li><li><a href="https://chat.danskgpt.dk">DanskGPT</a>: Dansk sprogteknologi tilgængelig for alle, helt gratis.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1252341135549005937)** (4 messages): 

- **Seeking Directions for QDora from Source**: A user asked for details on building **QDora from source** based on a Github issue from **Caseus** but found the instructions vague. They appealed for any direction, promising to figure out the rest themselves.
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1252347327604199424)** (6 messages): 

- **Fine-Tuning Vision Models Step-by-Step Guide:** Detailed instructions were shared on how to fine-tune a vision model, specifically using a pre-trained **ResNet-50** for classification tasks. The steps include installing required libraries, preparing the dataset, loading the model, defining data transforms, loading data, and training the model with an optimizer and loss function.
- **Dataset Preparation for PyTorch:** The guide emphasizes structuring the dataset compatible with `torchvision.datasets.ImageFolder` for ease of use with PyTorch. It uses the Oxford-IIIT Pet Dataset as an example and discusses applying appropriate transformations using the `transforms` module.

For references and more detailed steps, see the full post on [Phorm.ai](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=077f3d3e-542e-4b6f-b825-cea95799abf7).

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=077f3d3e-542e-4b6f-b825-cea95799abf7)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1252347900881670174)** (5 messages): 

- **Axolotl Vision Model Fine-tuning Tutorial**: A member asked for guidance on how to fine-tune vision models using **Axolotl**. The Phorm bot responded with a detailed step-by-step answer covering cloning the repository, installing dependencies, preparing the dataset, configuring the YAML file, fine-tuning, monitoring training, and using the fine-tuned model.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=0be694a5-7efc-4cdb-97f6-6691bd442899)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1252641919738052700)** (2 messages): 

- **Join the Webinar on Advanced RAG with Knowledge Graphs**: The **60-minute webinar** hosted by **@tb_tomaz from @neo4j** provides an in-depth tutorial on combining **LLMs** with knowledge graphs. Watch it for insights on graph construction and entity management. [Watch the Webinar](https://t.co/R5kLvvnJc2) [Tweet Link](https://t.co/q42URH3hSz).

- **LlamaIndex Makes the InfraRed 100**: We're thrilled to be included on the **@Redpoint InfraRed 100**, a recognition for cloud infrastructure companies excelling in reliability, scalability, security, and innovation. We're honored and in excellent company. [Tweet Link](https://t.co/X9Ec7ciWC9).
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1252415619727364198)** (62 messages🔥🔥): 

- **Switching documentation tools**: After llamaindex 0.10.20, they switched from Sphinx to MkDocs for documentation because Sphinx required every package to be installed, which wasn't feasible for their monorepo with 500 integrations. They needed a tool that could build API-docs across all packages without such limitations.
- **Fine-tuning embeddings for RAG pipeline**: A user struggled with fine-tuning embeddings for an e-commerce RAG pipeline, noting that embedding models aren't good with numeric data. They utilized GPT4 to generate synthetic queries but found the finetuned model performed worse; another suggested using Qdrant filters for more accurate numerical searching.
- **Modifying LlamaIndex prompts**: Another member faced issues with local vs server behavior of a custom LLM and received advice to modify the react prompt (a crucial, lengthy prompt) using [this approach](https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/?h=prompts).
- **PGVector documentation issues**: When discussing document filtering by date in vector searches, it was noted that PGVector lacks clear documentation for query filters. The suggested workaround involves querying the DB for document IDs within a date range and passing them to `VectorIndexRetriever`.
- **Discussing Llama 3 finetuning and entity extraction**: Query on finetuning Llama 3 for entity extraction and creating property graphs led to advice to edit the relevant class to use an async boto3 session for request handling. They were encouraged to fork the repo, make changes, and open a PR for implementation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/raw_works/status/1803079192214753280">Tweet from Raymond Weitekamp (@raw_works)</a>: gentlemen, behold!  ( conversion currently in progress w/ @llama_index & @neo4j )</li><li><a href="https://github.com/run-llama/rags?tab=readme-ov-file,">GitHub - run-llama/rags: Build ChatGPT over your data, all with natural language</a>: Build ChatGPT over your data, all with natural language - run-llama/rags</li><li><a href="https://docs.llamaindex.ai/en">LlamaIndex - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/KDBAI_Advanced_RAG_Demo/">Advanced RAG with temporal filters using LlamaIndex and KDB.AI vector store - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/llama_2_llama_cpp/?h=llamacpp">LlamaCPP - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/?h=prompts#accessing-prompts">Accessing/Customizing Prompts within Higher-Level Modules - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/">Qdrant Vector Store - Metadata Filter - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/chroma_auto_retriever/">Auto-Retrieval from a Vector Database - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1252349092399677573)** (18 messages🔥): 

- **Mistral finetuning error gets resolved**: A member reported an error while trying to finetune Mistral on Jarvis, citing an `OSError`. Another member suggested trying version 0.3, and after reattempting with updated token permissions, the issue was resolved.

- **VLM tokens discussion**: A user posted a StackOverflow query about the `phi-3-vision` model, noting the large number of tokens (~2000) an image takes. They shared their understanding and asked for insights into the token count and image size discrepancies.

- **GPT-4 Turbo throws internal server errors**: A member shared experiencing "internal server error" issues with GPT-4 Turbo approximately every 10-15 prompts and speculated on rate limits. A relevant link to an OpenAI community post was shared, which might help in troubleshooting.

- **Structured output from LLMs blog post**: Another member highlighted a blog post on getting structured output from LLMs. The post discusses various frameworks and techniques, and links to discussions on [Hacker News](https://news.ycombinator.com/item?id=40713952) and [/r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1di2r2x/every_way_to_get_structured_output_from_llms/).

- **Access issues for a course on Maven**: A member mentioned being unable to access a course on Maven and requested for it to be enabled.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co'">no title found</a>: no description found</li><li><a href="https://stackoverflow.com/questions/78635798/phi-3-vision-model-tokens">phi 3 vision model tokens</a>: I am looking at using phi-3-vision models to try and describe an image. However, I couldn&#x27;t help but notice that the number of tokens that an image takes is quite large (~2000). Is this correct, ...</li><li><a href="https://www.boundaryml.com/blog/structured-output-from-llms">Every Way To Get Structured Output From LLMs</a>: no description found</li><li><a href="https://community.openai.com/t/error-the-model-produced-invalid-content/747511/8">Error: &quot;The model produced invalid content&quot;</a>: Once i fixed the way I was using tools I never got this error message again.  Make sure you’re passing all the right ids, function names and parameters in the right orders and you should be good to go</li><li><a href="https://tenor.com/KhqP.gif">It Crowd Hello It GIF - It Crowd Hello IT Have You Tried Turning It Off And On Again - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1252701932355588238)** (1 messages): 

- **Modal gets speedy credit service**: A member mentioned they rarely wait for A100s in the past couple of days, finding it noteworthy enough to say "Thanks for the credits!" They plan to provide a detailed write-up on the developer experience with the repository later.
- **Checkpoint volumes lagging behind**: Another member is experiencing delays with checkpoint volumes updating immediately after they are written, noting that sometimes "files suddenly show up and have last-modified time 15 minutes earlier." They are curious whether such behavior is expected, referencing a specific example with `_allow_background_volume_commits`.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/)** (1 messages): 

strickvl: When do Replicate credits expire?
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1252479271650328607)** (1 messages): 

- **LangSmith Billing Issue**: A user reported setting up LangSmith billing and mentioned submitting a credits form for the "Mastering LLMs Course Credit." They requested help with the issue, providing their org ID **e2ec1139-4733-41bd-b4c9-6192106ee563**.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1252517418157477928)** (2 messages): 

- **Experimenting with SFR-Embedding-Mistral**: A member shared their experience with **SFR-Embedding-Mistral** and highlighted *peculiar behavior* in similarity scores between weather reports and a given query related to weather on a specific date. They noted that texts with dates do not rank the target text as expected and sought *explanations and mitigation strategies* for this issue.
- **Clarification on Similarity Scores**: Another member questioned if the original poster had made an *error* in their presentation, specifically regarding the similarity scores between Text 1 and Text 2 in relation to the query. They noted a possible confusion as the similarity with Text 1 was indeed higher, as initially claimed by the original poster.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/)** (1 messages): 

hammadkhan: https://x.com/xhluca/status/1803100958408241597?s=46&t=-TRJUfVdW8KeDqen1HJU1Q
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1252674926624772156)** (21 messages🔥): 

- **Crowd-source a list of credit providers**: Members discussed creating a crowd-sourced list of credit providers, including details on validity periods and amounts. Providers include **Modal** (1 year, 1000 credits) and **OpenAI** (3 months, 500 credits), among others.
- **Credit expiration calculation confusion**: Members were confused about when to start calculating the validity period for credits. Suggestions included starting from the first week of the course for safety.
- **Optimizing credit usage**: Members considered compiling credit amounts to optimize usage patterns. They suggested a "greedy pattern" approach, prioritizing providers like Predibase, BrainTrust, OpenAI, and others in sequence.
- **Auto-reminder for expiring credits**: A member suggested creating a Discord bot that would notify users one week before their credits expire with a message like *“Tik tok tik tok…”*. They aimed to ensure users make the most of their available credits.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1252632299946901566)** (2 messages): 

- **User requests credit assistance**: A member requested help with credit for account ID julishstack-9c5b6a, tagging <@466291653154439169> for assistance. No further details on the response or resolution were provided. 
- **Acknowledgment of receipt**: Another member confirmed receipt with a brief "Got it thank you!" indicating acknowledgment but no further context.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1252347578473775115)** (4 messages): 

- **Platform access confirmed**: A user expressed concern about seeing an "Upgrade" button despite wanting to start testing the platform. Another user reassured them, saying, "For sure! You should be all set now."

- **Credits expiration clarified**: A user asks about the expiration of the credits. The response was clear: "3 months from June 4."
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/1252385018081181727)** (1 messages): 

- **Course Enrollment Confirmation**: A member mentioned that they did not indicate their enrollment in the course on a questionnaire. They notified another member via Luma and also in the channel to ensure the information was received.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1252669241367203983)** (2 messages): 

- **Server Disconnected Frustration**: A member reported receiving a "Server disconnected" error while attempting inference with an L3-70B base adapter. The issue hindered their ability to proceed with the task.

- **Token Limits Explained**: Another member explained that **users get 1M tokens per day up to 10M tokens per month for free** using the serverless setup. They noted that this works in the prompt tab of the dashboard, but users must enter all the special instruct format tokens themselves.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/)** (1 messages): 

strickvl: When do OpenPipe credits expire?
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/)** (1 messages): 

sph3r3ical: yeah, where do you see the credits?
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[bergum_rag](https://discord.com/channels/1238365980128706560/1252713659243827251/1252713849631674390)** (7 messages): 

- **Last Session matters to everyone**: Members expressed sentiments about the final session, with phrases like *"Last session!"* and *"Till the next one."* indicating anticipation for future sessions. One member humorously noted, "You should know better by now."
- **Excitement about Gemini context caching features**: A member expressed enthusiasm for experimenting with **Gemini's context caching features** in many-shot prompting for LLM labeling. They are looking forward to utilizing these new capabilities.
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1252572250411110410)** (1 messages): 

- **Midnight Rose 70b price drop**: The price for [sophosympatheia/midnight-rose-70b](https://openrouter.ai/models/sophosympatheia/midnight-rose-70b/status) has seen a significant decrease. It is now available at **$0.8 per million tokens**, marking a **90% price drop**.
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

mka79: Is it from OR team?
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1252422221813645385)** (60 messages🔥🔥): 

- **Hang tight, updates are imminent**: The OpenRouter community expressed impatience over the lack of updates, with assurance from **Alex Atallah** that updates are coming soon. *"it's coming!"*
  
- **Understanding OpenRouter**: New users enquired about the purpose and use of **OpenRouter**, with responses explaining it focuses on prioritizing **price or performance** and features a **standardized API** to switch between models and providers easily. The explanation was complemented with a [link](https://openrouter.ai/docs/principles) to the principles page for more information.

- **Provider Uptime and Reliability**: Queries were raised about the reliability and uptime of the service when switching between providers, with reassurance given that **uptime is the collective uptime of all providers** and any issues are communicated through a notification system. An example uptime link for Dolphin Mixtral was shared [here](https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b/uptime).

- **Prompt Bug Fixes and System Tweaks**: Issues such as Claude's "self-moderated" function and the API key visibility problem were swiftly addressed by the team. *"Ah just pushed a tweak, fixing"*, *"working on it"*, highlighting proactive maintenance and user support.

- **Model Updates and Latency Concerns**: Members mentioned specific updates and performance concerns, such as renaming **DeepSeek coder** to **DeepSeek-Coder-V2** and **DeepInfra's Qwen 2 latency instability**. This showcases the community's active involvement in monitoring and improving service quality.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/principles">Principles | OpenRouter</a>: Core concepts for model selection</li><li><a href="https://openrouter.ai/models/cognitivecomputations/dolphin-mixtral-8x22b/uptime">Dolphin 2.9.2 Mixtral 8x22B 🐬 – Uptime and Availability</a>: Uptime statistics for Dolphin 2.9.2 Mixtral 8x22B 🐬 across providers - Dolphin 2.9 is designed for instruction following, conversational, and coding. This model is a finetune of [Mixtral 8x22B Instru...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 messages): 

sigridjin.eth: 와 안녕하세요.
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1252340401399005186)** (29 messages🔥): 

- **Legal challenges with CC content**: Members discussed that using **Creative Commons (CC)** content could reduce the legal attack surface but still pose issues for outputs resembling copyrighted items like Mickey Mouse. One suggested "patches" to address specific complaints over time.

- **CommonCanvas models performance issues**: Links to the [CommonCanvas project on Hugging Face](https://huggingface.co/common-canvas) were shared. Despite being largely "unusable as-is," one member noted potential in *training a texture generation model* with freely licensed textures; the corresponding research paper is available on [arXiv](https://arxiv.org/abs/2310.16825).

- **DeepFashion2 dataset results disappoint**: A user sought recommendations for image datasets on clothes and accessories after poor results from the DeepFashion2 dataset. No immediate alternatives were proposed.

- **GPT-NeoX weights location**: The [GPT-NeoX compatible Pythia-70M weights](https://huggingface.co/EleutherAI/neox-ckpt-pythia-70m-v1) were shared in response to a query. 

- **LLMs for middle token completion**: Members discussed models like **BERT**, **T5**, **BLOOM**, and **StarCoder** for natural language fill-in-the-middle tasks. **T5's** out-of-the-box performance was debated, with mentions of *T0* and *flan-T5* models being specifically fine-tuned for such tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/EleutherAI/neox-ckpt-pythia-70m-v1">EleutherAI/neox-ckpt-pythia-70m-v1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/common-canvas">common-canvas (CommonCanvas)</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1252356395270148179)** (20 messages🔥): 

- **Debate over Z-Loss fading in Importance**: A member questioned if anyone still uses **z-loss** for pretraining MoEs, noting that most, like **Mixtral**, use the load balance loss. Another pointed out that **DeepSeek V2** and **Skywork MoE** don't use z-loss, highlighting a shift away from z-loss as per recent papers.
- **Questionable HF Configs for Mixtral**: It was suggested that the HF configs for **Mixtral** may not be reliable, with one member sharing the true Mixtral params from the official torrent. The parameters included precise values for dimensions, layers, and MoE configurations.
- **RLHF and Mode Collapse Discussion**: One user remarked that rationalizing **RLHF censorship** as harmless leads to similar outcomes as **mode collapse** in human thinking. This led to brief exchanges on how these restrictions could have unintended consequences.
- **Introduction of GAMA for Audio Understanding**: A member shared information about **GAMA**, a novel **General-purpose Large Audio-Language Model (LALM)** capable of advanced audio understanding and reasoning, integrating multiple audio representations and fine-tuned on a large-scale audio-language dataset. Details and links were provided for further reading ([GAMA project](https://sreyan88.github.io/gamaaudio/)).
- **Latest ArXiv Papers Shared**: Several new papers were shared, highlighting advancements in machine learning and AI, including **Meta-Reasoning Prompting (MRP)**, improvements in **retrieval-augmented generation** with the ERASE methodology, and **sparse communication topologies** in multi-agent debates. These offer insights into optimizing computational costs and dynamic strategy application ([Example paper](https://arxiv.org/abs/2406.11776)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.11757">STAR: SocioTechnical Approach to Red Teaming Language Models</a>: This research introduces STAR, a sociotechnical framework that improves on current best practices for red teaming safety of large language models. STAR makes two key contributions: it enhances steerab...</li><li><a href="https://arxiv.org/abs/2406.11776">Improving Multi-Agent Debate with Sparse Communication Topology</a>: Multi-agent debate has proven effective in improving large language models quality for reasoning and factuality tasks. While various role-playing strategies in multi-agent debates have been explored, ...</li><li><a href="https://arxiv.org/abs/2406.11698">Meta Reasoning for Large Language Models</a>: We introduce Meta-Reasoning Prompting (MRP), a novel and efficient system prompting method for large language models (LLMs) inspired by human meta-reasoning. Traditional in-context learning-based reas...</li><li><a href="https://arxiv.org/abs/2406.11830">Language Modeling with Editable External Knowledge</a>: When the world changes, so does the text that humans write about it. How do we build language models that can be easily updated to reflect these changes? One popular approach is retrieval-augmented ge...</li><li><a href="https://arxiv.org/abs/2406.08761">VISinger2+: End-to-End Singing Voice Synthesis Augmented by Self-Supervised Learning Representation</a>: Singing Voice Synthesis (SVS) has witnessed significant advancements with the advent of deep learning techniques. However, a significant challenge in SVS is the scarcity of labeled singing voice data,...</li><li><a href="https://sreyan88.github.io/gamaaudio/">GAMA Audio</a>: no description found</li><li><a href="https://arxiv.org/abs/2406.11768">GAMA: A Large Audio-Language Model with Advanced Audio Understanding and Complex Reasoning Abilities</a>: Perceiving and understanding non-speech sounds and non-verbal speech is essential to making decisions that help us interact with our surroundings. In this paper, we propose GAMA, a novel General-purpo...</li><li><a href="https://github.com/huggingface/datasets/releases/tag/2.20.0">Release 2.20.0 · huggingface/datasets</a>: Important  Remove default trust_remote_code=True by @lhoestq in #6954  datasets with a python loading script now require passing trust_remote_code=True to be used    Datasets features  [Resumable I...</li><li><a href="https://arxiv.org/abs/2406.09241">What is the long-run distribution of stochastic gradient descent? A large deviations analysis</a>: In this paper, we examine the long-run distribution of stochastic gradient descent (SGD) in general, non-convex problems. Specifically, we seek to understand which regions of the problem&#39;s state s...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1252387027719295036)** (10 messages🔥): 

- **Users debate understanding of Paris plot in logit prisms**: A member expressed confusion about the Paris plot while discussing an [article on logit prisms](https://neuralblog.github.io/logit-prisms/#fig-logit-per-layer). Another member clarified that summing up all layers should give the original output logits back.

- **Logit prisms and its relationship to DLA**: Discussions referenced the similarity between logit prisms and direct logit attribution (DLA), linking to the [IOI paper](https://arxiv.org/pdf/2211.00593) and a related [LessWrong post](https://www.lesswrong.com/posts/2PucFqdRyEvaHb4Hn/an-adversarial-example-for-direct-logit-attribution-memory). One member acknowledged the overlap but argued that logit prisms offer a holistic view of logit decomposition and named it as such for its comprehensive approach.

- **Member seeks paper on shuffle-resistant transformer layers**: A user asked for a paper link discussing the resilience of transformer models to shuffled hidden layers. No specific response or link was provided in the conversation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://neuralblog.github.io/logit-prisms/#fig-logit-per-layer">Logit Prisms: Decomposing Transformer Outputs for Mechanistic Interpretability</a>: no description found</li><li><a href="https://www.lesswrong.com/posts/2PucFqdRyEvaHb4Hn/an-adversarial-example-for-direct-logit-attribution-memory">An adversarial example for Direct Logit Attribution: memory management in gelu-4l — LessWrong</a>: We provide concrete evidence for memory management or clean-up in a 4-layer transformer model.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1252507357767008319)** (2 messages): 

- **Passing vLLM arguments directly into the engine**: A user inquired about passing vLLM arguments, such as `--enforce_eager`, directly into the engine via the `model_args` dictionary. Another member indicated it should work as kwargs from `model_args`, but noted a potential "type casting bug" that needs to be addressed.
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1252337895537115156)** (17 messages🔥): 

- **New User Struggles with LangChain Version**: A new member is frustrated that the current **LangChain**'s version differs from tutorials online. They reference a video on [building a ChatGPT chatbot on Slack](https://www.youtube.com/watch?v=qKZLDEIL2r0) and are stuck at timestamp 11:31.

- **Extracting Data from Scraped Websites**: A user seeks help to extract specific components, such as company summaries and client lists, from 30-40 pages of scraped website data. They are advised to use **LangChain's information extraction capabilities** and provided with [GitHub Issue 12636](https://github.com/langchain-ai/langchain/issues/12636) as a resource.

- **Issue with LLMChain Deprecation**: Confusion arises over the deprecation of the `LLMChain` class in **LangChain 0.1.17**. A user notes that it will be removed in 0.3.0 and seeks clarity on using `RunnableSequence` instead.

- **Debugging LCEL Pipelines**: A member asks for ways to debug the output of each step in an LCEL pipeline and is advised to use `set_debug(True)` and `set_verbose(True)` from **LangChain globals** for insight into the input of the next node.

- **Handling API Request Errors in Loop**: A member encounters a `BadRequestError` while looping through games and making API calls, receiving feedback on tool messages not matching tool calls. They are looking for ways to resolve the issue generated by the incomplete API responses.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=qKZLDEIL2r0">How to build a chatGPT chatbot on Slack</a>: Welcome to this tutorial video on creating a Slack chatbot using the OpenAI language model, LangChain, and the Slack Bolt library. This video will showcase t...</li><li><a href="https://tenor.com/view/blowing-kisses-kisses-kiss-gratitude-huge-thanks-gif-16468716440995283694">Blowing Kisses Gratitude GIF - Blowing kisses Kisses Kiss - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/langchain-ai/langchain/issues/12636>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/12636>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/templates/#%EF%B8%8F-extraction>)">Templates | 🦜️🔗 LangChain</a>: Highlighting a few different categories of templates</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/extraction_long_text/#common-issues>).">How to handle long text when doing extraction | 🦜️🔗 LangChain</a>: When working with files, like PDFs, you&#x27;re likely to encounter text that exceeds your language model&#x27;s context window. To process this text, consider these strategies:
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1252528720686743552)** (14 messages🔥): 

- **Building Serverless Semantic Search**: A member shared a [Medium article](https://medium.com/@benitomartin/building-a-serverless-application-with-aws-lambda-and-qdrant-for-semantic-search-ddb7646d4c2f) titled "Building a Serverless Application with AWS Lambda and Qdrant for Semantic Search". The repository link is included in the article.
- **AgentForge launches on ProductHunt**: AgentForge has gone live on [ProductHunt](https://www.producthunt.com/posts/agentforge), featuring a NextJS boilerplate with LangChain, LangGraph, and LangSmith.
- **Advanced Research Assistant Beta Test**: A member is looking for beta testers for their advanced research assistant and search engine, offering 2 months free of premium features like Claude 3 Opus and GPT-4 Turbo. Interested testers can sign up at [Rubik's AI](https://rubiks.ai) using promo code `RUBIX`.
- **Environment Setup Advice**: An [article on Hugging Face](https://huggingface.co/blog/ucheog/separate-env-setup-from-code) advises separating environment setup from application code. The discussion praised tools like Bitwarden for managing credentials securely.
- **Infinite Backrooms Inspired Demo**: A member introduced [YouSim](https://yousim.ai), a web/world simulation platform inspired by infinite backrooms, allowing users to simulate any identity.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://yousim.ai/">YouSim</a>: they've simulated websites, worlds, and imaginary CLIs... but what if they simulated *you*?</li><li><a href="https://huggingface.co/blog/ucheog/separate-env-setup-from-code">Against mixing environment setup with code</a>: no description found</li><li><a href="https://www.producthunt.com/posts/agentforge"> AgentForge - Unlock the Power of AI with AgentForge | Product Hunt</a>: AgentForge is a NextJS boilerplate that helps entrepreneurs and developers quickly build and deploy AI agent-based applications. Create SaaS products, AI tools, or web apps with ease and start earning...</li><li><a href="https://vault.bitwarden.com/">no title found</a>: no description found</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1252592859488714934)** (1 messages): 

- **Did AI just end music?!**: A YouTube video titled *Did AI just end music?!* by **jasonzhou1993** delves into Music Gen 101 and how to build an application with the Text-to-Music API. The video can be [viewed here](https://youtu.be/yM-Lpq6E3Uc?si=1yu7xSlZkF9HekZp).

- **Hostinger Website Builder Discount Code**: For those interested in web development, **jasonzhou1993** shared a link to the Hostinger website builder where users can get a 10% discount using the code **AIJASON**. The offer is available [here](https://www.hostinger.com/aijason).

**Link mentioned**: <a href="https://youtu.be/yM-Lpq6E3Uc?si=1yu7xSlZkF9HekZp">Did AI just end music?!</a>: Music Gen 101 &amp; build application with Text-to-Music APIHostinger website builder: https://www.hostinger.com/aijasonGet 10% off with my code: AIJASON🔗 Links...

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1252411032819863685)** (9 messages🔥): 

- **Code clarity issues in graph.py**: A member identified a potential problem on line 69 of **graph.py** and contemplated formatting it with `.2f`. They suggested this change for better clarity.
- **Pull Request for graph float rounding**: A member announced the opening of a [pull request](https://github.com/tinygrad/tinygrad/pull/5021) for displaying floats rounded in the graph.
- **Request for PR review on OpenCL error messages**: A member requested a review on another [pull request](https://github.com/tinygrad/tinygrad/pull/5004) aimed at providing better OpenCL error messages.
- **George Hotz declines low-quality code submission**: **George Hotz** criticized the provided code as "low quality" and stated a **new policy** of closing PRs if submitters haven't carefully reviewed their diff. He requested members not to tag for reviews.
- **Member defends their effort on PR**: A member defended their code changes and expressed that they were trying to resolve the issue while having fun. They acknowledged bothering for a review and should not have.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/5021">graph display floats rounded by GabrielZCode · Pull Request #5021 · tinygrad/tinygrad</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5004">Fix/opencl Better error Messages by GabrielZCode · Pull Request #5004 · tinygrad/tinygrad</a>: Better openCL error messages!! Using the same strategy as generate_nv() function in generate_stubs.sh , I&#39;ve extracted the error messages from https://github.com/KhronosGroup/OpenCL-Headers/tree/m...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1252351947101569034)** (17 messages🔥): 

- **Understanding Tensor Realization in Tinygrad**: A user asked why `out` is not included in the `UOpsGraph` when realized. They concluded it's due to Tinygrad's lazy evaluation and separate kernel handling.

- **Kernel Generation in Tinygrad**: When comparing `remainder.realize()` and without it, outputs differed. It was confirmed that adding realizes can split the operations into multiple kernels, showcasing Lazy vs. eager execution in Tinygrad.

- **Kernel Fusion Explanation**: It was explained that operations can fuse into a single kernel unless explicitly separated by realizations. Cached kernels prevent redundant engine runs in subsequent operations.

- **Custom Accelerator Inquiry**: A user inquired about forcing kernel combinations for easier layout on custom hardware. They were directed to look into the scheduler for implementing such fusions.

- **Deep Dive into Scheduler**: Following the discussion on kernel fusion and realizations, a user expressed their intention to explore the Tinygrad scheduler further to support their custom accelerator integration.
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1252380985694097498)** (23 messages🔥): 

```html
- **RunwayML Gen-3 clip amazes users**: Members were impressed by a [RunwayML Gen-3 clip](https://fxtwitter.com/Mr_AllenT/status/1802706451586023763), calling its AI-generated details "insane". One noted, "99% of people wouldn't know this is AI."
- **DeepMind shares video-to-audio research**: A blog post on DeepMind's V2A technology was shared, explaining how video pixels and text prompts can generate soundtracks for videos. This could innovate in creating sound for silent footage and working with models like [Veo](https://deepmind.google/technologies/veo/).
- **Meta FAIR releases new research artifacts**: Meta FAIR announced several new [research artifacts](https://ai.meta.com/blog/meta-fair-research-new-releases/), including Meta Llama 3 and V-JEPA, emphasizing their commitment to open AI ecosystems. Another user was interested in the recently released Chameleon vision-only weights.
- **PKU-YuanGroup's Open-Sora Plan**: A member shared a [GitHub link](https://github.com/PKU-YuanGroup/Open-Sora-Plan) about the Open-Sora Plan, a project aimed at reproducing the Open AI T2V model. They requested community contributions to this open-source endeavor.
- **Free img2img model request**: A user expressed a need for a free img2img model using RealVision or similar, aiming to add "a touch of realism." They reminisced about potentially using their old custom Stable 2 model for this purpose.
```
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/okaris/omni-zero">Omni-Zero - a Hugging Face Space by okaris</a>: no description found</li><li><a href="https://ai.meta.com/blog/meta-fair-research-new-releases/">no title found</a>: no description found</li><li><a href="https://deepmind.google/discover/blog/generating-audio-for-video/">Generating audio for video</a>: Video-to-audio research uses video pixels and text prompts to generate rich soundtracks</li><li><a href="https://fxtwitter.com/Mr_AllenT/status/1802706451586023763">Tweet from Allen T. (@Mr_AllenT)</a>: The details of this @runwayml Gen-3 clip are insane  99% of people wouldn&#39;t know this is AI</li><li><a href="https://fxtwitter.com/ArmenAgha/status/1803138496967876642?t=QVF_6yJZfCva6c9iiWM4xQ&s=33">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: A restricted, safety aligned (no-image-out) version of Chameleon (7B/34B) is now open-weight!  https://github.com/facebookresearch/chameleon  The team strongly believes in open-source. We had to do a ...</li><li><a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan">GitHub - PKU-YuanGroup/Open-Sora-Plan: This project aim to reproduce Sora (Open AI T2V model), we wish the open source community contribute to this project.</a>: This project aim to reproduce Sora (Open AI T2V model), we wish the open source community contribute to this project. - PKU-YuanGroup/Open-Sora-Plan
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1252583333347721236)** (3 messages): 

- **Query on conversational speech datasets**: A member inquired if anyone knew of any conversational speech datasets similar to the ones used for training the **Bark model**. They noted that most available datasets seem to lack the emotional nuances found in audiobooks.

- **UC Berkeley's weight space discovery**: UC Berkeley, Snap Inc., and Stanford University researchers discovered an **interpretable latent space** in the weights of customized diffusion models, as detailed in their project [Weights2Weights](https://snap-research.github.io/weights2weights/). This space allows for sampling, editing, and inversion of over 60,000 fine-tuned models, each embedding a different person's visual identity.

**Link mentioned**: <a href="https://snap-research.github.io/weights2weights/">weights2weights</a>: no description found

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1252559786612883516)** (24 messages🔥): 

- **MPS vs CUDA Output Discrepancy**: A member reported experiencing `nan` outputs on CUDA while MPS produced sensible outputs with the same inputs. They identified the issue as differing kernel paths for SDPA on CUDA and CPU, with [fused attention causing softmax to `nan` on large values](https://github.com/pytorch/pytorch/issues/110213#issuecomment-1739952114).

- **Huggingface Cache Issue**: Another member shared that their system crashes when finetuning with Torchtune due to the Huggingface cache filling up. They were seeking advice on possible causes and solutions.

- **Huggingface to Torchtune Model Conversion**: Detailed steps were provided on converting a Huggingface model to Torchtune format, including an example with Gemma and pointers for automatic conversion for models like Llama2/3. They referenced [Torchtune Checkpointers](https://pytorch.org/torchtune/main/deep_dives/checkpointer.html) for automated weight conversion and loading.

- **Attention Mask Clarification**: Clarification was sought and provided on the correct format of the attention mask for a padded token input, verifying a specific matrix setup. This was discussed in the context of debugging the padding issue between different processing units.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/actions/runs/9373237554/job/25811096938#step:6:261)">RLHF with PPO · pytorch/torchtune@a1cde1c</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/pull/875https://github.com/pytorch/torchtune/pull/875">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/pytorch/pytorch/issues/110213#issuecomment-1739952114).">`scaled_dot_product_attention` behaves differently between v2.0 and v2.1 · Issue #110213 · pytorch/pytorch</a>: 🐛 Describe the bug With torch v2.1, scaled_dot_product_attention on GPU gives nan when a sequence has all large negative values (e.g torch.finfo(q.dtype).min - in order to mean no attention at all .....
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1252369949469704282)** (18 messages🔥): 

- **SEO-generated article on conversational AI apparent**: A user shared an [SEO-generated article](https://www.neural-voice.ai/mastering-conversational-ai-challenges-and-prospects/) from a Gen AI company with inaccuracies like "*Google's ChatGPT*," noting the absence of citations and cross-linking.
  
- **Werner Herzog reads AI output on podcast**: In Act 2 of a [This American Life podcast episode](https://podcasts.apple.com/us/podcast/this-american-life/id201671138?i=1000657607717), Werner Herzog reads output from davinci 003. The episode also delves into various human-AI and interpersonal relationships.

- **Podcast tools discussion**: A user inquired about tools for creating podcast intros and show notes, with [smol-podcaster](https://github.com/FanaHOVA/smol-podcaster) being mentioned. Discussions also included a comparison between Assembly.ai and Whisper for transcription.

- **Meta announces new AI models**: Meta has unveiled four new AI models, including [Meta Chameleon, Meta Multi-Token Prediction, Meta JASCO, and Meta AudioSeal](https://x.com/AIatMeta/status/1803103538169651679), along with research artifacts. These releases aim to bolster open AI innovation and responsible development.

- **Google Gemini API introduces context caching**: Good news for Google developers, as [context caching](https://x.com/officiallogank/status/1803096828595863608?s=46&t=90xQ8sGy63D2OtiaoGJuww) for the Gemini API has been launched. This feature supports both 1.5 Flash and 1.5 Pro versions, is significantly cheaper, and is available immediately.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AIatMeta/status/1803103538169651679">Tweet from AI at Meta (@AIatMeta)</a>: Today is a good day for open science.  As part of our continued commitment to the growth and development of an open ecosystem, today at Meta FAIR we’re announcing four new publicly available AI models...</li><li><a href="https://x.com/officiallogank/status/1803096828595863608?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Great news for @Google developers:   Context caching for the Gemini API is here, supports both 1.5 Flash and 1.5 Pro, is 2x cheaper than we previously announced, and is available to everyone right now...</li><li><a href="https://x.com/KarinaVinnikova/status/1802980985056710732">Tweet from Karina Vinnikova (@KarinaVinnikova)</a>: LOL FSB forgot to pay for ChatGPT 4</li><li><a href="https://github.com/FanaHOVA/smol-podcaster">GitHub - FanaHOVA/smol-podcaster: smol-podcaster is your autonomous podcast production intern 🐣</a>: smol-podcaster is your autonomous podcast production intern 🐣 - FanaHOVA/smol-podcaster</li><li><a href="https://podcasts.apple.com/us/podcast/this-american-life/id201671138?i=1000657607717">‎This American Life: 832: That Other Guy on Apple Podcasts</a>: ‎Show This American Life, Ep 832: That Other Guy - Jun 2, 2024</li><li><a href="https://www.neural-voice.ai/mastering-conversational-ai-challenges-and-prospects/">Evolving Conversational AI: Challenges and Future Prospects - Neural Voice AI</a>: Conversational AI has emerged as a groundbreaking technology that is transforming the way we interact with computers and devices. Powered by advancements in
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1252361896426864650)** (9 messages🔥): 

- **Users debate the best local LLM for commercial use**: A user questioned the best local LLM available for commercial use, with responses recommending **llama-70b** despite **codestral** ranking higher but not being suitable for commercial use. Another user shared a [link to a leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) showing rankings where Mixtral-8x22b was noted but praised llama-3-70b for better performance in other benchmarks.

- **Discussing more integration profiles**: A member expressed enthusiasm for having more integration profiles, suggesting that **e2b** should be next. Another user sought further understanding of **E2B**, clarifying its value and use cases for outsourcing execution to secure sandboxes.

- **Request for video reviews of OI release**: A user asked if there were any video reviews or video content of the latest OI release. They were directed to a [YouTube video](https://www.youtube.com/live/pqBuxmpgpY0?si=DEXxMuOIqIK1guYF) titled "WELCOME TO THE JUNE OPENINTERPRETER HOUSE PARTY", hosted by Restream.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>: no description found</li><li><a href="https://www.youtube.com/live/pqBuxmpgpY0?si=DEXxMuOIqIK1guYF">WELCOME TO THE JUNE OPENINTERPRETER HOUSE PARTY</a>: Powered by Restream https://restream.iodiscord stages are hard
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/)** (1 messages): 

legaltext.ai: the one from april?
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1252650362733858948)** (2 messages): 

- **Open Interpreter’s Local III launch**: @hellokillian announced the release of **Open Interpreter’s Local III**, boasting *"computer-controlling agents that work offline."* He mentioned key features such as *"interpreter --local sets up fast, local LLMs,"* a free inference endpoint, and training their own model. [source](https://x.com/hellokillian/status/1803090274186617188)
- **Descriptive photo naming made easy**: @MikeBirdTech introduced a tool to *"Automatically give your photos descriptive names, fully offline."* Promoted as private and free, this tool emphasizes convenience and data privacy. [source](https://x.com/MikeBirdTech/status/1803091094420246619)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/MikeBirdTech/status/1803091094420246619">Tweet from Mike Bird (@MikeBirdTech)</a>: Automatically give your photos descriptive names, fully offline  Private and free</li><li><a href="https://x.com/hellokillian/status/1803090274186617188">Tweet from killian (@hellokillian)</a>: Open Interpreter’s Local III is out today.  We are building computer-controlling agents that work offline. This is our biggest step forward.  - interpreter --local sets up fast, local LLMs. - We are h...
</li>
</ul>

</div>
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1252359711068782764)** (3 messages): 

- **Agent Hospital Simulacrum Introduced**: A member linked to an [arXiv paper](https://arxiv.org/abs/2405.02957) that introduces **Agent Hospital**, a system simulating the entire process of treating illness with autonomous agents powered by LLMs. The paper discusses **MedAgent-Zero**, which helps doctor agents learn and improve their treatment performance by simulating disease onset and progression.
  
- **Real-World Application of Agent Hospital**: The paper highlighted in the discussion claims that the knowledge acquired by doctor agents in **Agent Hospital** is applicable to real-world medicare benchmarks. Accumulating experience from treating around ten thousand patients in the simulation helps improve performance, replicating years of real-world learning.

**Link mentioned**: <a href="https://arxiv.org/abs/2405.02957">Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents</a>: In this paper, we introduce a simulacrum of hospital called Agent Hospital that simulates the entire process of treating illness. All patients, nurses, and doctors are autonomous agents powered by lar...

  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/)** (1 messages): 

shajith: oh that is good, thanks for sharing.
  

---


### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1252375690989535232)** (2 messages): 

- **Comprehensive LLM video demo with annotated notes available**: Simon Willison shared a lengthy video demo and tutorial on **LLM** usage from the command-line, which was part of the [Mastering LLMs Conference](https://maven.com/parlance-labs/fine-tuning). He also provided an annotated presentation with detailed notes [on his blog](https://simonwillison.net/tags/annotatedtalks/) and a YouTube link to the talk. 

- **Calmcode set to have a new release soon**: Vincent Warmerdam announced that **Calmcode** has a new maintainer and hinted at an upcoming release.

**Link mentioned**: <a href="https://simonwillison.net/2024/Jun/17/cli-language-models/">Language models on the command-line</a>: I gave a talk about accessing Large Language Models from the command-line last week as part of the Mastering LLMs: A Conference For Developers &amp; Data Scientists six week long …

  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1252602527841521696)** (1 messages): 

- **PR for improving MoE prompt eval speed**: A member highlighted a pull request titled [llamafile : improve moe prompt eval speed on cpu #6840](https://github.com/ggerganov/llama.cpp/pull/6840) that had already been approved but was conflicting with the main branch. The member requested the author to rebase the PR.
  

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

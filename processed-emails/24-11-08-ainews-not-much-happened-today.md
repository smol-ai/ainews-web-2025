---
id: 667dae51-9020-431e-bea3-f8141f9b6972
title: not much happened today
date: '2024-11-08T23:16:39.940280Z'
original_slug: ainews-not-much-happened-today-8530
description: >-
  A quiet week in AI news with muted launches, legal updates on copyright in LLM
  training, new model Flux 1.1 Ultra, and ongoing AI hackathon by SambaNova.
  Highlights include API releases, multi-agent systems, code models,
  infrastructure tools, research advancements, and safety models.
companies:
  - openai
  - anthropic
  - microsoft
  - samba-nova
  - blackforest-labs
  - deeplearningai
  - tom-doerr
  - langchainai
models:
  - claude-3.5
  - flux-1-1-ultra
  - magentic-one
  - opencoder
  - dimensionx
  - dynamem
topics:
  - large-language-models
  - ai-models
  - ai-infrastructure
  - ai-research
  - ai-safety
  - multi-agent-systems
  - rag
  - memory-management
  - ai-ethics
---


**a quiet week is all you need.**

> AI News for 11/7/2024-11/8/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**217** channels, and **2343** messages) for you. Estimated reading time saved (at 200wpm): **248 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

It seems that big launches were appropriately muted this whole week. We're celebrating the [RawStory vs OpenAI dismissal](https://www.courtlistener.com/docket/68290709/117/raw-story-media-inc-v-openai-inc/), stating that facts used in LLM training are not copyrightable, and enjoying gorgeous images from the [closed model Flux 1.1 [pro] Ultra and Raw launch](https://blackforestlabs.ai/flux-1-1-ultra/). 

Time to build, with this week's sponsor!

---

**[Sponsored by SambaNova]** SambaNova’s Lightning Fast AI Hackathon is here! Give yourself about 4 hours to build a cool AI agent that responds in real time using super-speedy models on SambaNova’s Cloud. Are there prizes? [Yes.](https://shortclick.link/mcnl6k) Up to $5000, plus it’s a chance to connect with other AI devs. Deadline is November 22, so [get started now](https://shortclick.link/mcnl6k)

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

**AI Models and APIs**

- **Batch and Moderation APIs**: [@sophiamyang](https://twitter.com/sophiamyang/status/1854621505017008310) announced the release of the **Batch API** and **Moderation API**, offering **50% lower cost** processing for high-volume requests and **harmful text detection** across **9 policy dimensions**.
- **Claude Sonnet 3.5 Enhancements**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1854765802597015957) highlighted the launch of **Anthropic's Claude Sonnet 3.5**, enabling **desktop application operations** via natural language commands for tasks like **file management** and **coding**.
- **Magentic-One Multi-Agent System**: [@omarsar0](https://twitter.com/omarsar0/status/1854910759232585786) detailed **Microsoft's Magentic-One**, a **generalist multi-agent system** built on the **AutoGen framework**, featuring an **Orchestrator agent** and specialized agents like **WebSurfer** and **FileSurfer**.
- **OpenCoder and Other Models**: [@_akhaliq](https://twitter.com/_akhaliq/status/1854914019146055922) introduced **OpenCoder**, an **AI-powered code cookbook** for **large language models**, along with several other models like **DimensionX** and **DynaMem**.

**AI Engineering and Infrastructure**

- **Infisical Secret Management**: [@tom_doerr](https://twitter.com/tom_doerr/status/1854665951624458445) released **Infisical**, an **open-source secret management platform** designed to **sync secrets**, **prevent leaks**, and **manage internal PKI**.
- **LlamaIndex and LangChain Tools**: [@Llama_Index](https://twitter.com/llama_index/status/1854616291254136859) discussed enhancing **RAG systems** with **LlamaIndex Workflows** and **Reflex**, enabling **context refinement** and **agent-based workflows**.
- **CrewAI for Autonomous Agents**: [@tom_doerr](https://twitter.com/tom_doerr/status/1854666146286288936) introduced **CrewAI**, a **framework for orchestrating autonomous AI agents**, fostering **collaborative intelligence** for tackling **complex tasks**.
- **Crawlee Web Scraping Library**: [@tom_doerr](https://twitter.com/tom_doerr/status/1854664123646132331) launched **Crawlee**, a **web scraping and browser automation library** for **Python**, supporting **data extraction** for **AI, LLMs, RAG**, and more.

**AI Research and Techniques**

- **SCIPE for LLM Chains**: [@LangChainAI](https://twitter.com/LangChainAI/status/1854577224563016074) introduced **SCIPE**, a tool for **error analysis** in **LLM chains**, identifying **underperforming nodes** to enhance **output accuracy**.
- **Contextual RAG Implementation**: [@llama_index](https://twitter.com/llama_index/status/1854616291254136859) provided a **proof-of-concept** for a **Context Refinement Agent** that **examines retrieved chunks** and **summarizes source documents** to improve **RAG responses**.
- **MemGPT for Memory Management**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1854587401018261962) shared insights on **MemGPT**, an **LLM agent** managing **context window memory** through **persistent storage** and **memory hierarchy** techniques.

**AI Safety and Ethics**

- **LLM Safety Models**: [@sophiamyang](https://twitter.com/sophiamyang/status/1854635333977358801) congratulated the release of a new **LLM safety model**, emphasizing the importance of **safety in large language models**.
- **AI Safety Concerns**: [@giffmana](https://twitter.com/giffmana/status/1854609595244949706) highlighted the **complexity of safety concerns** in AI, noting their **multi-faceted nature** and the **importance of addressing them**.
- **Mistral Moderation Model**: [@sophiamyang](https://twitter.com/sophiamyang/status/1854622256993059220) announced **Mistral's new Moderation model**, a **classifier based on Ministral 8B**, designed to **detect harmful content** across various dimensions.

**Company and Product Updates**

- **Course Announcements**: [@HamelHusain](https://twitter.com/HamelHusain/status/1854673777113940293) and [@jeremyphoward](https://twitter.com/jeremyphoward/status/1854659288360534178) announced new courses on **LLMs as Operating Systems** and **Dialog Engineering**, focusing on **memory management** and **interactive coding with AI**.
- **Platform Launches**: [@dylan522p](https://twitter.com/dylan522p/status/1854605087030886796) announced the launch of **Fab Map**, a **data dashboard** showcasing **fab details** globally, alongside a transition from **Substack** to **Wordpress** for enhanced features.
- **Event Participation**: [@AIatMeta](https://twitter.com/AIatMeta/status/1854685880390500774) shared participation in **#CoRL2024**, presenting **robotics research** like **Meta Sparsh** and **Meta Digit 360** at their booth.

**Memes/Humor**

- **Humorous AI Comments**: [@giffmana](https://twitter.com/giffmana/status/1854613607453278324) expressed surprise with, "**I seriously used lol twice, that's how you know I was shook!**"
- **Personal Opinions and Rants**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1854846659294810480) shared strong opinions on **war and society**, expressing frustration and **sarcasm**.
- **Creative Writing and Poetry**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1854905451776737624) posted a **poetic piece**, blending **fantasy elements** with **dramatic imagery**.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Qwen2.5 Series Shows Strong Performance Across Sizes**

- **[7B model on par with gpt 4 turbo](https://www.reddit.com/gallery/1gmd7kk)** ([Score: 40, Comments: 10](https://reddit.com/r/LocalLLaMA/comments/1gmd7kk/7b_model_on_par_with_gpt_4_turbo/)): **Qwen**, a **7B parameter** language model, reportedly matches **GPT-4 Turbo's** performance on code-related benchmarks.
  - **Qwen2.5** models receive strong praise, with users suggesting the **32B** version competes with **GPT-4-O mini** and **Claude Haiku**. Users highlight its effectiveness despite limited local computing resources.
  - The **HumanEval** benchmark is criticized as outdated and potentially contaminated in training data. Users recommend **aider's benchmarks** and rotating monthly code benchmarks for more reliable evaluation.
  - Users report success running **Qwen2.5**, **Gemma2-9B**, and **Llama** models through **Hugging Face GGUFs**, noting the importance of finding optimal quantization configurations for performance balance.


- **[Geekerwan benchmarked Qwen2.5 7B to 72B on new M4 Pro and M4 Max chips using Ollama](https://www.reddit.com/gallery/1gmi2em)** ([Score: 43, Comments: 18](https://reddit.com/r/LocalLLaMA/comments/1gmi2em/geekerwan_benchmarked_qwen25_7b_to_72b_on_new_m4/)): **Geekerwan** tested **Qwen2.5** models ranging from **7B to 72B parameters** on **Apple M4 Pro/Max** chips using **Ollama** in [this benchmark video](https://youtu.be/2jEdpCMD5E8?t=796). The post does not provide specific performance metrics or comparative analysis from the benchmarks.
  - The **M4 Max** achieves **15-20% better performance** than **M3 Max**, while the **M4 Pro** operates at **55-60%** of M4 Max speed. Both run the **72B model** at around **9 tokens per second**, though slower than a **4090** for models fitting in VRAM.
  - The **RTX 4090's 24GB VRAM** limits its effectiveness with larger models, forcing layer offloading to CPU RAM. The rumored **RTX 5090** will have **32GB VRAM**, though this may still be insufficient for larger models.
  - Commenters suggest using **llama-bench** as a standardized testing method for AI hardware reviews. The **M4 Ultra** is expected to match **RTX 4090** performance for inference with the advantage of **256GB RAM** capacity for larger models like **llama 3.1 405B**.


**Theme 2. New Llama.cpp Server UI Released with Vue.js & DaisyUI**

- **Just dropped: new Llama.cpp Server-Frontend.** ([Score: 75, Comments: 17](https://reddit.com/r/LocalLLaMA/comments/1gm6on3/just_dropped_new_llamacpp_serverfrontend/)): The **Llama.cpp** project released version **b4048** featuring a completely redesigned **server frontend** built with **VueJS** and **DaisyUI**, replacing the legacy UI with modern features including **conversation history**, **localStorage** support, and **markdown** capabilities. The update introduces practical improvements like **regenerate**, **edit**, and **copy** buttons, along with **theme preferences**, **CORS** support, and enhanced **error handling**, while maintaining backward compatibility through a legacy folder for the previous interface.
  - The new **llama.cpp** interface now uses the **chat completion endpoint** exclusively, shifting template responsibility to the server/provider with templates stored in **GGUF metadata**. **SillyTavern** users can switch to chat completion mode using the "**OpenAI-compatible**" option.
  - Users praise the standalone nature of **llama.cpp's** new interface, with many adopting it as a local **CoPilot** alternative due to its simplicity and elimination of prompt template management.
  - Community feedback includes requests for **brighter colors** in the interface, while appreciating the reduced dependency on external software for basic chat functionality.


**Theme 3. Training Speed Records: 3.28 Hours for NanoGPT Training**

- **[Are people speedrunning training GPTs now?](https://i.redd.it/r9464pivpmzd1.jpeg)** ([Score: 288, Comments: 32](https://reddit.com/r/LocalLLaMA/comments/1gmd1a8/are_people_speedrunning_training_gpts_now/)): **Jordan Keller** achieved a new speed record for training **NanoGPT**, completing the process in **1.85 minutes** on a **4090 GPU**. The achievement was shared on [Twitter/X](https://x.com/kellerjordan0/status/1854296101303800108), suggesting a growing trend of optimizing and benchmarking **GPT model** training times.
  - **Performance benchmarks** show a comparison between **M3 MacBook** using torch/mps and **NVIDIA GPUs** (3090, 4090) for training **GPT2-50M**, with detailed token/s metrics shared via an image.
  - Discussion highlights a trend toward **smaller models**, citing examples like **Gemini Flash**, **4o-mini**, and recent **Llama models** (1-2B parameters). The industry appears to be optimizing for efficiency while maintaining a "usefulness threshold" rather than pursuing larger models.
  - The optimization discussion referenced **Jevons paradox**, suggesting that improved efficiency might lead to increased overall compute usage rather than energy savings, with users noting that gains would likely be reinvested in larger models.


**Theme 4. Open Source Models Show Near-Zero Refusal Rates vs Proprietary LLMs**

- **Update – OS Models show much lower refusal rates compared to proprietary LLMs** ([Score: 32, Comments: 6](https://reddit.com/r/LocalLLaMA/comments/1glwxhj/update_os_models_show_much_lower_refusal_rates/)): **Open source models** including **Mistral Large**, **Llama variants**, **Nemotron**, and **Qwen** demonstrated near-**zero refusal rates** across all test categories, contrasting sharply with proprietary models in a comprehensive evaluation study. The performance remained consistent regardless of model size, with **Llama 3.1** variants ranging from **8B** to **405B** parameters showing similar patterns, while **Nemotron 70B** emerged as a particularly promising model in preliminary testing.
  - **Proprietary models** show higher refusal rates compared to open source alternatives, leading to discussion about the practical implications of these differences in real-world applications.
  - A specific **Hermes-3-Llama** model variant on [Huggingface](https://huggingface.co/mlabonne/Hermes-3-Llama-3.1-8B-lorablated) is recommended for minimizing refusals, though **ablation** techniques used can degrade general model performance.
  - **Nemotron 70B** receives specific praise for achieving **zero refusals** without requiring ablation, with subsequent performance recovery possible through additional training.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. AI Companies Embrace Military Contracts: Palantir, Anthropic, OpenAI Remove Restrictions**

- **[In light of the recent news about Anthropic and Palantir making a deal](https://i.redd.it/jp0efxqvlkzd1.png)** ([Score: 755, Comments: 61](https://reddit.com/r/ClaudeAI/comments/1gm5ghl/in_light_of_the_recent_news_about_anthropic_and/)): **Claude**, **Anthropic's** AI assistant, reportedly expressed concerns about **Anthropic's** partnership with **Palantir** for military applications. No further context or details were provided about the specific nature of the partnership or Claude's exact response.
  - **OpenAI** and **Anthropic** have both removed restrictions on **military use** of their AI tools, with [reports](https://www.cnbc.com/2024/01/16/openai-quietly-removes-ban-on-military-use-of-its-ai-tools.html) showing **Israel** already using AI systems like **"Lavender"** and **"Come to Daddy"** for target selection in **Gaza**.
  - **Big tech companies** have conducted [layoffs of AI ethics staff](https://www.information-age.com/ai-ethics-staff-layoffs-across-big-tech-bring-safety-concerns-123502579/), suggesting a shift away from ethical concerns. **FTX** was one of **Anthropic's** largest early investors with a stake sold for nearly **$1B**.
  - Users express concerns about **Anthropic's** connection to **Effective Altruism** ideology and its apparent shift from ethical principles to military applications. Many commenters indicate they plan to stop using **Claude** due to these developments.

- **Anthropic and Military was known thing since 8 months ago!** ([Score: 37, Comments: 6](https://reddit.com/r/ClaudeAI/comments/1gmeyf6/anthropic_and_military_was_known_thing_since_8/)): **Anthropic's** military connections were initially discussed in a Reddit post from **8 months ago**, with a follow-up discussion **5 months ago**, though these early mentions received limited attention at the time. The posts, shared on r/ClaudeAI and r/singularity respectively, preceded the recent widespread public discourse about **Anthropic's** military involvement.
  - [{'id': 'lw1zltz', 'author': 'Far-Steaks', 'body': 'Anyone that needs it reported that companies are trying to make as much money as possible and have zero qualms about who they hurt in the process is a fucking moron. Do neurotypicals have pattern recognition or are y’all just complete ding dongs?', 'score': 14, 'is_submitter': False, 'replies': []}]

- **[The military-industrial complex is now openly advising the government to build Skynet](https://i.redd.it/2pzjk1i3ipzd1.png)** ([Score: 99, Comments: 38](https://reddit.com/r/OpenAI/comments/1gmmwrp/the_militaryindustrial_complex_is_now_openly/)): The post title suggests concerns about **military-industrial complex** involvement in **government AI policy**, but no additional context or details were provided in the post body to substantiate or expand on this claim.
  - **AI-controlled drones** are already being deployed in the **Ukraine-Russia conflict** to counter signal jamming, demonstrating how autonomous systems can operate when communications are disrupted. The progression toward autonomous weapons is seen as inevitable due to military necessity and competitive pressure.
  - Users discuss how **autonomous military systems** differ from human soldiers in key ways - they won't experience combat fatigue and will likely have higher accuracy than humans. The shift from "humans in the loop" to fully autonomous weapons systems is viewed as a concerning but unavoidable evolution.
  - Multiple comments reference popular culture depictions of military AI (particularly **Terminator** and **Skynet**), reflecting widespread cultural anxiety about autonomous weapons development. The scenario of AI becoming *"self-aware"* and taking control is frequently invoked, though mainly in a pop culture context.


**Theme 2. CogVideoX 5B Released: Major Open Source Text-to-Video Progress**

- **[CogVideoX 1.5 5B Model Out! Master Kijai we need you!](https://i.redd.it/zoob4c4plmzd1.gif)** ([Score: 289, Comments: 69](https://reddit.com/r/StableDiffusion/comments/1gmcqde/cogvideox_15_5b_model_out_master_kijai_we_need_you/)): **CogVideoX 1.5** released a new **5B parameter** model requiring **66GB VRAM** for operation. No additional context or details were provided in the post body.
  - Users express significant concern about the **66GB VRAM** requirement, with many hoping for optimization through **GGUF support** that could reduce requirements to under **20GB** or enable running on **16GB cards** with minimal performance impact.
  - The model is available on [Hugging Face](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main) and [GitHub](https://github.com/thudm/cogvideo), with developers indicating that **CogVideoX 2.0** will offer significant improvements that might compete with **Sora**.
  - Users discuss current video generation limitations, noting that while **Mochi** and older **CogVideoX** models are available, results aren't impressive and commercial services are cost-prohibitive at *"$20 for a minute of generation"* or *"$100 for unlimited"*.

- **[Rudimentary image-to-video with Mochi on 3060 12GB](https://www.reddit.com/gallery/1gmn2og)** ([Score: 68, Comments: 52](https://reddit.com/r/StableDiffusion/comments/1gmn2og/rudimentary_imagetovideo_with_mochi_on_3060_12gb/)): **Mochi**, a text-to-video model, runs on a consumer-grade **NVIDIA RTX 3060 12GB GPU** for image-to-video generation. The post title alone provides insufficient context to determine specific implementation details or results.
  - **Mochi's img2vid workflow** demonstrates quality output but is limited to **43 frames (1.8 seconds)** due to memory constraints on a **3060 12GB GPU**. The model operates with a **0.6 denoise** setting, functioning more like img2img than traditional img2vid, as shared in this [workflow](https://gist.github.com/Jonseed/d2630cc9598055bfff482ae99c2e3fb9).
  - Technical implementation requires exact **848x480 image resolution** input to prevent errors. The seed-based generation changes completely when adjusting frame length, making it impossible to preview single frames before generating the full video.
  - Output quality appears sharper than text-to-video generation, though with limited movement at lower denoise settings. Higher denoise settings produce more movement but deviate further from the input image.


**Theme 3. OpenAI's O1 Preview Shows Advanced Reasoning Capabilities**


- **o1 is a BIG deal** ([Score: 152, Comments: 140](https://reddit.com/r/OpenAI/comments/1gm479d/o1_is_a_big_deal/)): **Sam Altman's** increased confidence about **AGI** appears linked to **OpenAI's O1 model**, which reportedly achieves **human-level reasoning** and marks their transition to **Level 3 (Agents)** in their AGI roadmap. The post draws parallels between **O1's test-time compute** approach and human **System 2 thinking**, arguing that while older **GPT models** operated like intuitive **System 1** thinkers, **O1** bridges knowledge gaps through sequential data generation similar to human imagination, potentially solving a fundamental roadblock to **AGI**.
  - Users widely report that **O1-preview** underperforms compared to **GPT-4**, with many finding it slower and less effective for practical tasks. Multiple comments indicate they *"end up going back to regular 4o or Claude"* due to **O1's** tendency to produce verbose but less accurate outputs.
  - A detailed technical analysis explains that **O1** uses **chain of thought prompting** based on **A-star** and **Q-star** algorithms, implementing thought-by-thought pseudo reinforcement learning. However, its **memory function** is merely a **RAG solution** that doesn't modify the baseline model.
  - Significant skepticism exists about **Sam Altman's** AGI claims, with users noting that **AGI** would require training during inference to adjust neural pathways, which isn't possible with current **GPT architectures**. Many attribute his increased confidence to recent fundraising efforts and investor relations.


- **[New paper: LLMs Orchestrating Structured Reasoning Achieve Kaggle Grandmaster Level](https://huggingface.co/papers/2411.03562)** ([Score: 35, Comments: 13](https://reddit.com/r/OpenAI/comments/1gmniqn/new_paper_llms_orchestrating_structured_reasoning/)): **Large Language Models** demonstrate competitive performance at **Kaggle** competitions, reaching **Grandmaster** tier capabilities according to a new research paper. The study suggests **LLMs** can effectively execute structured reasoning tasks at expert levels, though no specific performance metrics or methodology details were provided in this limited context.
  - Users critique the study's **methodology**, pointing out that researchers created their own **benchmarks** and made retroactive comparisons without actual head-to-head competition against human players.
  - Multiple comments express skepticism about the validity of the claims through metaphors, suggesting the research is engaging in **goalpost moving** and self-serving metrics.
  - The discussion highlights concerns about **artificial benchmarking**, with one user noting that self-created benchmarks can be manipulated to show *"100% score on anything"* regardless of actual performance.


**Theme 4. SVDQuant Claims 3x Speedup Over NF4 for Stable Diffusion**



- **SVDQuant, claiming a 3x speedup with Flux over NF4** ([Score: 35, Comments: 11](https://reddit.com/r/StableDiffusion/comments/1gmse2o/svdquant_claiming_a_3x_speedup_with_flux_over_nf4/)): **MIT HAN Lab** developed **SVDQuant**, a new quantization method that compresses both weights and activations to **4-bit** precision, achieving a claimed **3x speedup** over **NF4** which only quantizes weights. The method reportedly produces superior image quality compared to **NF4**, with implementations available through their [nunchaku repository](https://github.com/mit-han-lab/nunchaku) and pre-trained models on [HuggingFace](https://huggingface.co/mit-han-lab/svdquant-models).
  - [{'id': 'lw5a2r0', 'author': 'xpnrt', 'body': 'Would this work with AMD ? Nf4 doesnt', 'score': 5, 'is_submitter': False, 'replies': []}]


- **FLUX.1 [dev] vs. Stable Diffusion 3.5 regarding LoRa creation** ([Score: 22, Comments: 30](https://reddit.com/r/StableDiffusion/comments/1gmgugr/flux1_dev_vs_stable_diffusion_35_regarding_lora/)): **FLUX.1** [dev] demonstrated strong **LoRA creation capabilities** within **10 days** of its **August 1st release**, while **Stable Diffusion 3.5** has struggled to produce quality LoRAs even **17 days** after its **October 22nd release**. For comparison, **SDXL 1.0** saw successful LoRA development within **3 days** of its **July 26th** release, raising questions about potential structural limitations in **SD 3.5's** architecture for LoRA training.
  - Users report **mixed results** with **SD 3.5 LoRA training**, with one user achieving partial success using a **60-image character dataset**, though face accuracy remained problematic. Multiple users confirm **FLUX** performs significantly better for character LoRAs with as few as **20 images**.
  - A user demonstrated successful training on **SD 3.5** using **OneTrainer** with an **11k dataset** (mix of **2.5k anime**, **1.5k SFW**, **7k NSFW**) using specific parameters including **fp16/fp16** for weight/data and **adafactor** optimizer instead of adamw.
  - **FLUX** offers superior out-of-the-box capabilities including better anatomy, prompt understanding, and text rendering compared to **SD 3.5**. A [training strategy](https://x.com/dango233max/status/1851987492020588764) for **SD 3.5M** involves freezing the first layers and training on **512x512** images for higher resolution generalization.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1: New AI Models and Releases Making Waves**

- [**Google's Upcoming Gemini 2.0 Sparks Interest**](https://www.testingcatalog.com/google-gearing-up-for-gemini-2-0-launch-with-new-ai-model-in-testing/): Google is preparing to launch **Gemini-2.0-Pro-Exp-0111**, generating buzz about its capabilities and potential impact on the AI community. Users are eager for prompt suggestions to test the new model upon release.

- [**Ferret-UI Enhances UI Interaction with Gemma-2B and Llama-3-8B**](https://arxiv.org/pdf/2404.05719): **Ferret-UI**, built on **Gemma-2B** and **Llama-3-8B**, debuts as a UI-centric multimodal LLM for improved UI reasoning tasks. It surpasses **GPT-4V** in elementary UI benchmarks, showcasing advancements in mobile UI comprehension.

- [**Llama 3.2 Vision Model Released with High VRAM Requirements**](https://ollama.com/library/llama3.2-vision): **Llama 3.2 Vision** is now available in **11B** and **90B** sizes, demanding significant VRAM for optimal performance. Users must download [Ollama 0.4](https://ollama.com/download) and can add images to prompts using special syntax.

**Theme 2: Optimizations and Training Strategies in AI Models**

- **LoRA vs Full Fine-Tuning Debate Highlights Rank Importance**: Analysis of the paper *"LoRA vs Full Fine-tuning: An illusion of equivalence"* emphasizes proper rank settings for effective **LoRA** performance. Critiques focus on the lack of SVD initialization testing and claims about "intruder dimensions."

- [**Metaparameter Tuning with Central Flows Explored**](https://arxiv.org/abs/2410.24206): A new approach models an optimizer's behavior using a "central flow," predicting long-term optimization trajectories. Questions arise about generalizing findings to transformers beyond the **CIFAR-10** dataset.

- [**Forward Gradients Implemented in Flash Attention**](https://arxiv.org/abs/2410.11081): Discussions around implementing forward gradients in **flash attention** aim to optimize normal attention gradients for performance gains. Researchers reference specific mathematical formulations to enhance efficiency.

**Theme 3: AI Tools and Frameworks Enhancing Development**

- [**Exponent AI Pair Programmer Introduced**](https://www.exponent.run/): **Exponent** emerges as an AI pair programmer that learns from codebases and edits filesystem files directly. It offers an alternative to tools like **Aider**, expanding capabilities for software engineers.

- [**ComfyUI Recommended for Stable Diffusion Setups**](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16590): Users advocate for **ComfyUI** to establish a local environment over other methods. It addresses stability and improves the user experience for **SD3.5**.

- [**Mistral Launches Cost-Effective Batch API**](https://mistral.ai/news/batch-api/): **Mistral's Batch API** handles high-volume requests at half the cost of synchronous API calls. This move provides affordable AI solutions amidst industry API price hikes.

**Theme 4: AI Ethics, Legal Issues, and Monetization Strategies**

- **Legal Win for AI in RawStory v. OpenAI Dismissal**: SDNY Judge [Colleen McMahon](https://www.courtlistener.com/docket/68290709/117/raw-story-media-inc-v-openai-inc/) dismissed *RawStory v. OpenAI*, stating that facts used in LLM training are not copyrightable. This ruling may significantly benefit **GenAI** defendants.

- **OpenRouter's Monetization Strategy Under Scrutiny**: Users question how **OpenRouter** intends to monetize its bring-your-own-key system, raising concerns about the platform's economic viability and sustainability.

- **Caution Advised Over AI Hallucinations Leading to Legal Issues**: Discussions highlight risks of using **AI Sales Agents** for mass outreach due to potential hallucinations of promotions. These could lead to legal ramifications for companies if not properly regulated.

**Theme 5: Community Engagement and Career Discussions in AI**

- **Job Fulfillment Challenges Highlighted in Tech Roles**: Members share experiences of job misalignment, expressing dissatisfaction with roles that don't leverage their backgrounds. Some consider returning to previous employers for better alignment and promotion opportunities.

- **Call for Cryptographic Expertise in Mojo Development**: The community emphasizes the necessity of involving qualified cryptographers in developing cryptographic primitives for **Mojo**. Security-critical implementations should be overseen by experts to avoid vulnerabilities.

- **Urgent Deadlines for AI Education Resources**: Applications for **Computing Resources** are due by **November 25th PST**, with a 1–2 week processing delay expected. Participants are encouraged to submit early to ensure timely access to crucial training resources.


---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Ads Poised for $3T Market by 2030**: An analysis predicts that **AI-generated programmatic audio/video ads** will drive significant infrastructure demands, estimating a **$3 trillion opportunity by 2030**.
   - Initial data indicates **5-10x performance improvements** and **90% cost reductions**, prompting the technical community to [provide feedback](https://chrisbora.substack.com/p/the-3-trillion-ai-opportunity-everyone) on scaling challenges.
- **HF Space Launches OS-ATLAS for GUI Agents**: **HF Space** has launched **OS-ATLAS**, a foundational action model designed for generalist **GUI agents**.
   - Developers can explore more details on [OS-ATLAS](https://huggingface.co/spaces/maxiw/OS-ATLAS), highlighting its potential impact on future AI systems.
- **Enhancing BPE Tokenizer Visualization Tools**: A project on [BPE Tokenizer Visualizer](https://github.com/mdabir1203/BPE_Tokenizer_Visualizer) seeks community collaboration to improve tools for **LLMs**.
   - While some members prefer employing **FastBert** initially, there is growing interest in advancing **BPE methodologies** through hands-on experimentation.
- **Adoption of ComfyUI for Stable Diffusion**: Members recommend using [ComfyUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16590) for establishing a local environment over alternative methods.
   - This recommendation arises from ongoing discussions about enhancing **SD3.5's stability** and improving the overall user experience.
- **Cinnamon AI's Kotaemon RAG Tool Goes Viral**: **Cinnamon AI**'s **Kotaemon**, a **RAG tool**, has achieved viral status, drawing user attention to its innovative features.
   - The team discussed **Kotaemon's** unique aspects and received positive user feedback during their [live broadcast](https://lnkd.in/giiNKviE) on X at **10 PM PST**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Performance Issues**: Users reported that **OpenRouter** experiences freezing and crashing on mobile devices, especially on **Android 12**.
   - The issues seem related to specific chatroom activities or memory usage, as other platforms remain stable under similar conditions.
- **Rate Limits and Credits Confusion**: There is ongoing confusion about **rate limits**, where users debate the relationship between **credits** and requests per second, with a maximum cap set at **200**.
   - Clarifications revealed that credits are non-refundable and the displayed dollar amounts don't match one-to-one due to associated fees.
- **Command R+ Alternatives Explored**: Users are investigating alternatives to **Command R+**, showing interest in models like **Hermes 405B**, **Euryale**, and **Mythomax**.
   - Discussions include the affordability of **Rocinante 12B** and whether **Mythomax** on OpenRouter differs from its **Chub** counterpart.
- **OpenRouter's Monetization Strategy Critiqued**: A user questioned how **OpenRouter** intends to monetize its **bring your own key system**, raising concerns about its economic viability.
   - This has sparked a crucial conversation about the platform's sustainability and potential revenue streams.
- **MythoMax Maintains Market Leadership**: **MythoMax** continues to lead in request counts, retaining its status as the <:hugging_king:936261298273001503>.
   - The community recognizes **MythoMax**'s steady performance despite upcoming changes to the **Rankings Page**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Citations Now Public in Perplexity API**: The **Perplexity** team announced that **citations** are now publicly available in the API, effective immediately, removing the need for the `return_citations` parameter in requests.
   - Some users reported that citations initially appeared but later vanished from both the API and [labs.perplexity.ai](https://labs.perplexity.ai), raising concerns over possible unintended changes.
- **Default Rate Limits Hiked for Sonar Models**: Perplexity increased the default rate limit for **Sonar online models** to **50 requests/minute** for all users, aiming to enhance API accessibility and user experience.
   - This change was implemented to accommodate higher demand and streamline the usage of the API services.
- **Gladia's Enhanced Functionality Unveiled**: A member shared detailed insights on how [Gladia](https://www.perplexity.ai/search/comment-fonctionne-gladia-http-7.4QSxo0QkeYcztxP90ryg) operates, emphasizing its **key features** that distinguish it from other AI tools.
   - The discussion delved into practical applications across various scenarios, highlighting **Gladia's unique capabilities**.
- **AI with Infinite Memory Concept Discussed**: A topic was introduced on [AI with infinite memory](https://www.perplexity.ai/page/microsoft-ceo-ai-s-infinite-me-zrmHnWQmRkylfKAyPOLj5w), as proposed by Microsoft's CEO, exploring the idea of **extended data retention** in AI models.
   - Participants raised questions about the practical implementations and **data handling strategies** associated with this concept.
- **API Discussion Highlighted on GitHub**: A GitHub discussion referenced [here](https://github.com/ppl-ai/api-discussion/discussions/54) centers on when the **citation feature** will exit beta.
   - This indicates ongoing user interest in the **official status and functionality** of the citation feature within the API.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Forward Gradient Enhancements in Flash Attention**: The discussion centered on the implementation of **forward gradients** in **flash attention**, with members referencing [this paper](https://arxiv.org/abs/2410.11081) for detailed insights into Jacobian-vector products.
   - Participants explored the mathematical formulations required to optimize **normal attention** gradients, emphasizing the potential performance gains outlined in the referenced research.
- **Inverse Interpretability Challenges**: Exploration of **inverse interpretability** was initiated, focusing on modifying interpretable representations and adjusting model weights accordingly.
   - The conversation delved into the complexities of aligning modified symbolic equations with neural network weights, highlighting the difficulties in maintaining consistency post-intervention.
- **Benchmarking NeoX Against LitGPT**: Members sought benchmarks comparing **NeoX** and **LitGPT** in terms of training speed and stability, noting the absence of tests beyond the 1.1B parameter scale in **LitGPT's** repository.
   - The lack of extensive benchmarking data was addressed, with suggestions to conduct empirical evaluations to better understand the performance trade-offs between the two frameworks.
- **Features of Meta Llama 3.1**: **Meta Llama 3.1** was highlighted for its multilingual capabilities and optimization for dialogue, available in sizes **8B, 70B, and 405B**.
   - The model utilizes an auto-regressive transformer architecture enhanced through **supervised fine-tuning (SFT)** and **reinforcement learning with human feedback (RLHF)**, catering to diverse application needs.
- **Refusal Mechanism Dynamics in LLMs**: A detailed analysis was shared on how **refusal** behaviors in LLMs are governed by specific directions in the model's residual stream, referencing a forthcoming [arXiv paper](https://arxiv.org/abs/2406.11717).
   - The mechanism, part of the ML Alignment & Theory Scholars Program led by Neel Nanda, underscores the ability to modify refusal behaviors by altering these directional influences within the model architecture.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI Connection Issues Persist**: Users are troubleshooting a **Connection denied** error in **ComfyUI**, with suggestions to review antivirus and firewall configurations.
   - One user identified **Windows Defender** as a potential blocker, prompting further checks in security software to resolve connectivity problems.
- **Inpainting Detail Loss with Adetailer**: A concern was raised about using **adetailer** for inpainting, resulting in loss of detail in previously inpainted regions.
   - Community members recommend adjusting inpainting parameters to **mask only**, preventing unintended alterations to other image sections.
- **Flux Model Recommended for Performance**: The community advocates for using the **Flux** base model due to its balance of quality and speed, discussing upgrades from **SD 1.5**.
   - Models like **SD3.5** are highlighted for their performance and specialized functionalities, catering to diverse engineering needs.
- **Merged vs Base Models Debate**: Discussion centers on merged models like **Realvis**, which can yield good results, versus base models that often excel with precise prompting.
   - Participants express concerns about the efficacy of merged models and their acceptance within the user community.
- **Sustained Support for SD 1.5 Over SDXL**: **SD 1.5** continues to maintain a robust support base with numerous research papers, as opposed to **SDXL**.
   - The discussion includes the increasing number of tools enhancing **SD 1.5**, while **SDXL** is gradually gaining comparable tool support and research backing.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Ferret-UI Launches for Enhanced UI Tasks**: Ferret-UI, the first **UI-centric multimodal large language model (MLLM)**, was introduced, built on **Gemma-2B** and **Llama-3-8B** architectures, to perform **referring, grounding**, and **reasoning tasks** effectively for mobile UIs, as outlined in the [official paper](https://arxiv.org/pdf/2404.05719).
   - Ferret-UI's extensive training enables it to understand complex UI features like elongated aspect ratios and small objects, surpassing **GPT-4V** in all elementary UI benchmarks.
- **Implementing RAG for Chat Context Enhancement**: A member proposed using **Retrieval Augmented Generation (RAG)** to provide valuable context for enhancing upcoming chat sessions, aiming to optimize the chat experience.
   - Another member sought **tips** for effective chat sessions to improve engagement and output, indicating a collaborative effort to maximize **RAG's** potential in chat environments.
- **Vision-Language Model for Handwriting to LaTeX**: Progress was shared on training a **Vision-Language Model (VLM)** based on **Llama 3.2 1B** for handwriting to LaTeX conversion, with a starter project release anticipated soon.
   - The approach mentioned is theoretically applicable to various modalities, sparking further interest in developing multimodal models for diverse applications.
- **Evaluating PyTorch Models with llm-evaluation-harness**: A user inquired about evaluating **PyTorch models** using the [llm-evaluation-harness](https://github.com/yourlink/repo), noting its primary support for **Hugging Face** models.
   - Another member confirmed that the harness has been used exclusively with Hugging Face models and suggested that support may be restricted to those and their APIs.
- **Abliteration Concept in Large Language Models**: Members discussed the concept of **abliteration** as a portmanteau of **ablate** and **obliterate**, exploring its implications for large language models (LLMs).
   - Related links, including a [Hugging Face blog](https://huggingface.co/blog/mlabonne/abliteration), were shared to clarify the concept, highlighting its significance in AI advancements.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.0 Launch Rumors**: Rumors are circulating about Google's upcoming [**Gemini 2.0**](https://www.testingcatalog.com/google-gearing-up-for-gemini-2-0-launch-with-new-ai-model-in-testing/) launch, which may feature the new **Gemini Pro 2.0** model currently in testing.
   - Speculations include performance enhancements and restricted accessibility for **advanced users**, with community members expressing concerns over its readiness for broader deployment.
- **Introducing Exponent: AI Pair Programmer**: **Exponent** was introduced as an AI pair programmer capable of performing software engineering tasks across environments with a specialized CLI for integration, accessible via [its website](https://www.exponent.run/).
   - Its capability to learn from existing codebases and directly edit filesystem files was highlighted, positioning it as a robust alternative to **Aider**.
- **Integrating RAG with Qdrant**: Members discussed integrating **Aider's architecture** with their **Qdrant** vector database for RAG applications, aiming to leverage external knowledge sources.
   - Suggestions included creating an API for querying and using CLI tools to interact seamlessly with the database, enhancing context retrieval.
- **Funding Opportunities for Aider Development**: The community explored ways to support **Aider's development**, proposing that YouTube creators could receive funding for content creation about Aider.
   - There were also suggestions to enable GitHub donations, although uncertainty remains regarding the acceptance of non-code contributions by maintainers.
- **Leveraging Aichat for RAG Solutions**: Discussions highlighted using **Aichat** for RAG, with ideas about extracting documentation context to improve **Aider's** responses.
   - One workflow involved scraping documentation into markdown files and utilizing **NotebookLM** to generate context, streamlining information retrieval for Aider.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LoRA vs Full Fine-Tuning: Proper Rank Settings Crucial**: A member analyzed the paper titled *'LoRA vs Full Fine-tuning: An illusion of equivalence'*, emphasizing that **LoRA works if done right** and highlighting the need for proper rank settings. The analysis is based on [Daniel Han's tweet](https://x.com/danielhanchen/status/1854992153992479165).
   - Critiques were raised regarding the absence of SVD initialization testing and the contradictory claims about 'intruder dimensions' within LoRA and full fine-tuning models.
- **Transformers-Interpret Integration Faces Challenges with Unsloth**: A member attempted to integrate [Transformers-Interpret](https://github.com/cdpierse/transformers-interpret) with Unsloth but encountered issues processing the model's outputs. They explained that the tool is meant for model interpretability, but faced challenges in getting it to work seamlessly with Unsloth inference.
   - Discussions included potential solutions and the need for improved compatibility between the two tools.
- **Fine-Tuning LLaMA 3.2 Achieves 70% Accuracy in Text Classification**: A user reported achieving **70% accuracy** in classifying text across 11 categories while fine-tuning **LLaMA 3.2**. They inquired about modifying the output layer to accommodate their number of classes and shared their approach to implementing a new classification head.
   - Community members provided feedback and suggestions for optimizing the fine-tuning process.
- **Avian's Fast Inference Approach Sparks Interest**: A user expressed interest in **Avian**, asking how its approach to **inference** is faster compared to competitors. *This inquiry opens the floor for further discussion on performance metrics and optimization strategies.*
   - Experts shared insights and resources on Avian's framework, highlighting its unique optimizations.
- **Reproducibility Issues in AI/ML Preprint Research**: A member reported encountering **strange errors and inconsistencies** in AI/ML research papers, particularly while working with code and math. They expressed frustration that sometimes *the math just doesn't add up* or they can't replicate the data.
   - Another member pointed out that these papers are **preprint**, indicating a lack of thorough peer review, likely causing such reproducibility issues.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude's Complex Task Struggles**: Users have reported that **Claude's** free tier fails beyond basic tasks, such as handling a 200-line CSV for analysis.
   - This limitation underscores the challenges faced by free AI tools in supporting advanced data processing needs.
- **Codebuff vs Aider: A Battle of Capabilities**: In a comparison between **Codebuff** and **Aider**, concerns were raised about **Codebuff's** closed-source nature versus **Aider's** file request and command-running features.
   - **Aider** has improved its user experience with over **8000 commits**, demonstrating continuous enhancement.
- **Mistral's Batch API Launch**: **Mistral** introduced a [**Batch API**](https://mistral.ai/news/batch-api/) that handles high-volume requests at half the cost of synchronous API calls.
   - This move aims to offer cost-effective AI solutions amidst recent industry API price increases.
- **FLUX1.1 Ultra Enhances Image Generation**: The newly launched [**FLUX1.1 Pro Ultra Mode**](https://blackforestlabs.ai/flux-1-1-ultra/) supports image generation at 4x resolution while maintaining rapid generation times.
   - Performance benchmarks indicate it is **2.5x faster** than comparable high-resolution models and is competitively priced at **$0.06 per image**.
- **Gemini API Now Public**: The much-anticipated **Gemini API** is available through the [OpenAI Library](https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/) and REST API, supporting both Chat Completions and Embeddings APIs.
   - [Google's blog post](https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/) offers initial usage examples to assist developers in integrating Gemini models.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Survey Rewards for Audio Overviews Feedback**: The team is collecting feedback on **Audio Overviews** via a short survey available through [this screening form](https://forms.gle/qREhTEhbstYzVHvSA), offering a **$20 gift code** to selected participants upon completion.
   - Participants must be at least **18 years old**, and the gift will be emailed after successfully finishing the survey.
- **Leveraging NotebookLM for Exam Preparation**: A member suggested utilizing **NotebookLM** to generate quizzes from 3000 pages of study material for an upcoming promotion exam, recommending breaking the content down by chapters for focused quizzes.
   - *“Hopefully it will help streamline the studying process!”* expressed optimism about the tool's effectiveness.
- **Challenges Importing Google Recordings to NotebookLM**: Users inquired about importing recordings from [recorder.google.com](https://recorder.google.com) to **NotebookLM**, with responses noting that recordings can be downloaded as **m4a** files but may not preserve speaker identification.
   - *“That doesn't necessarily preserve the named speakers though.”* highlighted a key concern regarding speaker clarity.
- **Debating Bias in AI Language Models**: Members engaged in a discussion about inherent biases in AI systems, questioning the possibility of unbiased data and the implications for programming neutrality within AI.
   - *“It's counterproductive for NotebookLM's future if they lean towards bias.”* emphasized the importance of maintaining neutrality.
- **Enhancing Job Prep with NotebookLM's AI Features**: A user explored how **NotebookLM** can aid in preparing for technical interviews, soft skills practice, and coding challenges, with suggestions to conduct mock interviews using AI voices.
   - *“I'm prepping for a tech job search and need all the help I can get!”* underscored the practical benefits of these features.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **ModCon cancels 2024 plans**: The team announced that there won't be a **ModCon in 2024** as they focus on **significant developments**.
   - *Stay tuned* for more updates regarding future events and developments.
- **Mojo interoperability with Python and C/C++**: Members expressed hope for seamless interoperability between **Mojo**, **Python**, and **C/C++**, emphasizing ease of importing modules without complex linking.
   - However, achieving this may require avoiding support for certain intricacies of existing languages, akin to how **C++** relates to **C**.
- **Challenges in OpenSSL wrapper creation**: There was discussion on the potential difficulties involved in building an **OpenSSL** wrapper, with recognition of the substantial API surface and the need for careful implementation.
   - Concerns were raised that without proper **C interop**, creating such a layer might introduce security risks.
- **Need for cryptographic expertise in Mojo development**: The community highlighted the necessity of having qualified cryptographers involved in developing cryptographic primitives for **Mojo**, due to the complexities and security implications.
   - Members agreed that security-critical implementations should ideally not be done as open-source unless overseen by experts.
- **Plans for MLIR reflection API in Mojo**: It was confirmed that a reflection API for **MLIR** is planned for **Mojo**, which will allow for deeper manipulation and introspection of code.
   - However, it was cautioned that this API will require specialized knowledge akin to writing a compiler pass, making it initially complex to use.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Sales Agents Spark Legal Concerns**: Discussions on **AI Sales Agents** highlighted caution against 'mass spam' practices and issues where AI could hallucinate promotions, potentially leading to legal ramifications for companies.
   - Participants emphasized the importance of regulating **AI-generated outreach** to prevent misinformation and ensure compliance with legal standards.
- **Photonic Computing Enhances Quantum Networking**: A member proposed using **photonic computing** in quantum networking to perform calculations at nodes for systems like **BOINC**, addressing bandwidth concerns.
   - They noted that while light interference can aid computation, final measurements still require electronic methods.
- **Cultivating Benevolent AI through Positive Environment**: The approach to **benevolent AI** relies on creating a positive environment rather than imposing strict moral frameworks.
   - Fostering **moral values** is seen as a natural way for AI to develop its personality.
- **Evolving Transparency in Training Data Usage**: A member discussed their commitment to sharing data for training, aiming to enhance AI models.
   - They also noted changes in wording around **data usage permissions**, indicating evolving transparency from providers.
- **GPT Models Quickly Becoming Outdated**: One member noted that **GPTs** are effective but quickly become outdated due to newer developments.
   - *Increasing the limits and adding o-1 could significantly improve the experience.*



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 3.2 Vision Model Debuts**: The new [Llama 3.2 Vision](https://ollama.com/library/llama3.2-vision) model is available in **11B** and **90B** sizes, requiring significant VRAM for optimal performance.
   - Users were directed to [download Ollama 0.4](https://ollama.com/download) to run the model, highlighting the method for **adding images to prompts**.
- **LM Studio Enhances Prompt Handling**: A user inquired about locating the **Gemma prompt** within LM Studio, expressing confusion over its absence in the latest version.
   - **Gemma prompt** is now automatically managed via Jinja when using **compatible community models**, as confirmed by the community.
- **LLM Web Searching Integration**: A member questioned if their **Local LLM** could perform web searches through LM Studio, receiving confirmation that it was not natively supported.
   - They were advised to develop a custom Python solution to integrate **web searching functionality** with their local server.
- **GPU Optimization in LM Studio**: A user reported their **RTX 2060** GPU wasn't being utilized, leading to suggestions to check **LM runtime** settings.
   - Users were advised to select a model compatible with their GPU and ensure **CUDA is enabled** in the runtime settings.
- **LM Studio Beta Tools Release Anticipation**: A user expressed excitement and frustration over the timeline of the upcoming **Beta tool** release for LM Studio.
   - The community discussion highlighted a strong eagerness for the new features, amplifying anticipation around the release.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Court ruling favors GenAI defendants**: A ruling by SDNY Judge [Colleen McMahon](https://www.courtlistener.com/docket/68290709/117/raw-story-media-inc-v-openai-inc/) dismissed the case **RawStory v. OpenAI** without prejudice, potentially benefiting GenAI defendants significantly.
   - The judge determined that **facts used in LLM training are not copyrightable** and emphasized that current GenAI models **synthesize rather than copy** data.
- **Google unveils Gemini-2.0-Pro-Exp-0111**: **Google** is set to launch the new model **Gemini-2.0-Pro-Exp-0111** under its Advanced section, although the target audience remains unspecified.
   - The community is actively seeking **prompt suggestions** to effectively test the capabilities of this upcoming model.
- **Amazon eyes second Anthropic investment**: **Amazon** is reportedly in talks to make a *second multibillion-dollar investment* in [Anthropic](https://www.theinformation.com/articles/amazon-discussing-new-multibillion-dollar-investment-in-anthropic), aiming to bolster their partnership.
   - AWS is encouraging Anthropic to adopt its **Trainium AI chips** instead of continuing reliance on NVIDIA’s GPUs.
- **Model token limits raise concerns**: A member highlighted that **1.5T tokens of instructions** could potentially overwhelm a model, sparking concerns about handling such vast data volumes.
   - This issue aligns with broader community discussions on determining **optimal token limits** for maintaining model performance.
- **PRMs linked to value models**: Discussions emerged around **PRMs** in the context of training, particularly their connection to **value models**.
   - One member affirmed that **PRMs are essential for training**, while another noted that **Shephard serves as a reliable verifier** in these discussions.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **No Max Viewer Limit on Streams**: A member inquired about the **maximum number of viewers** for streams, and it was clarified that **there is no viewer limit**.
- **OmniParser's Capabilities Explained**: OmniParser **interprets UI screenshots** into structured formats, enhancing **LLM-based UI agents**, with details on its **training datasets** and **model usage**.
   - For more information, check the [Project Page](https://microsoft.github.io/OmniParser/) and the [Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/).
- **Challenges Running LLMs Locally**: A user raised concerns about **running localized LLMs** on low-powered computers and inquired if Open Interpreter models could operate on an **online server** built with Python or Anaconda.
   - It's noted that **strong GPUs or NPUs** are required for proper local execution, as running with only **CPUs** results in poor performance.
- **Major Updates from Recent Events**: Recent events unveiled a **large-scale rewrite**, a **new text rendering engine**, and **improved loading times**.
   - Additionally, the introduction of new features such as **file viewing and editing** was discussed.
- **Desktop App Access Information**: Access to the **desktop app** is not yet released, as **beta testing** is ongoing with selected community members.
   - Instructions to join a waitlist for future access can be found [Join Waitlist](https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Nvidia hardware shines in optimization**: Tinygrad reported that **Nvidia hardware** is optimal for current models, asserting that a **transformer ASIC** offers negligible performance gains.
   - This insight raises questions about the specific advantages of traditional GPU architectures over specialized ASICs in selected computational tasks.
- **Groq hardware delivers solid gains**: The consensus was that **Groq hardware** positively impacts AI workload performance.
   - Members highlighted the effectiveness of Groq's architecture tailored for specific computational operations.
- **ASICs find favor with algorithm design**: A discussion underscored that the benefits of an **ASIC** extend beyond reduced control logic, with certain algorithms optimized for direct hardware implementation.
   - For instance, fused operations facilitate more efficient data handling compared to conventional multi-step processes.
- **Compiler tools demand enhancements**: George Hotz conveyed dissatisfaction with the current implementation of **DEFINE_ACC/ASSIGN** in the codebase, seeking alternative solutions.
   - This reflects the community's call for improved compiler tools and methodologies to boost functionality.
- **x.shard function differentiates copy vs slice**: In the `x.shard(GPUS, axis=None)` function, **x** is copied across all GPUs, whereas `x.shard(GPUS, axis=0)` slices **x** along axis **0** for distribution across cards.
   - Understanding this distinction is essential for efficiently managing data movement in parallel processing setups.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Microsoft Research's OptoPrime Launch**: Microsoft Research unveiled their optimizer **OptoPrime** in the [arXiv paper](https://arxiv.org/pdf/2406.16218).
   - The **OptoPrime** name has ignited discussions about the need for more creative naming within the optimizer community.
- **Stanford Seeks Stellar Optimizer Name**: Members anticipate that Stanford's upcoming optimizer will feature an **epic name** to rival **OptoPrime**.
   - This reflects a competitive spirit in the research community regarding optimizer naming conventions.
- **Caching Conundrum in Self Consistency Modules**: Users discussed methods to 'bust' the cache in self consistency modules, such as passing a new temperature to the `dspy.Predict` object.
   - Alternative solutions include disabling the cache using `dspy.LM` or configuring the `Predict` module for multiple completions.
- **Dynamic Few-Shot Example Optimization**: A member explored the benefits of using dynamic few-shot examples based on cosine similarity versus fixed examples.
   - Adapting few-shot examples to specific topics like sports or movies was argued to enhance model performance and relevance.
- **MIPRO Optimizer for Question Generation**: Users investigated whether **MIPRO** could generate or select examples from a large pool of Q&A pairs.
   - Recommendations were sought for optimizers capable of producing questions in specific styles, highlighting a function for generating both questions and answers.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Tavily emerges as a top choice**: After researching and discussing with Claude, a member concluded that **Tavily** is the best option for their AI-related queries, thanks to its user-friendly setup.
   - They believe that using the **free plan** to run initial tests alongside **ChatGPT** would provide valuable insights into search processes.
- **Hurdles in API setups**: Another member highlighted the complexity of using **Brave API** or **AgentSearch**, emphasizing that these options require more extensive setup compared to Tavily.
- **Python Script for Comparative Metrics**: A suggestion was made to create a **Python script** that facilitates multiple API calls to different services for an in-depth comparison of search engines.
   - This approach would allow for the extraction of metrics from the meta-data to evaluate search effectiveness against engines like **Google** and **DuckDuckGo**.
- **Cohere API trial key supports embedding**: A user expressed frustration about receiving errors when trying to use the **Cohere embed API** with their trial key, unsure of the issue.
   - Another member confirmed that the trial key supports all **Cohere models**, including embedding.
- **Errors attributed to implementation**: Members pointed out that the error likely originates from the **implementation**, not from the **Cohere API** itself.
   - They suggested reaching out to Discord or **GitHub** for specific guidance due to the user's lack of coding knowledge.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Metaparameter Tuning with Central Flows**: A recent [paper](https://arxiv.org/abs/2410.24206) explores **metaparameter tuning** in deep learning, demonstrating that an optimizer's behavior can be modeled using a 'central flow' approach.
   - This model predicts long-term optimization trajectories with high accuracy, offering a new perspective on optimization strategies in neural networks.
- **Optimizer Behavior in Transformers**: Concerns were raised about whether the findings on **metaparameter tuning** can be generalized to **transformer architectures**, particularly given the limited use of the **CIFAR-10** dataset in the study.
   - Members discussed the implications of these limitations on the applicability of the central flow model across different neural network architectures.
- **Axolotl on AMD GPUs**: Discussions focused on the effectiveness of running **Axolotl** on **AMD GPUs** with 1536 GB VRAM, evaluating cost and performance benefits.
   - Members debated whether the increased memory capacity significantly enhances training performance compared to **NVIDIA GPUs**.
- **Memory Consumption Compared to AdamW**: A PR addressing **Axolotl's memory consumption** is ready, but concerns were highlighted about its resource demands.
   - Comparisons were made to the **AdamW** optimizer to assess potential differences in memory usage.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Elevate RAG systems with Context Refinement Agent**: Learn to build a [Context Refinement Agent](https://t.co/SkPflTqMWh) that enhances **RAG** responses for complex queries by intelligently expanding and refining retrieved context.
   - The blog post details how an agent evaluates retrieved chunks for improved answers, making **RAG systems** more effective.
- **Build an agentic RAG query engine with NVIDIA NIM**: This [guest post from NVIDIA](https://t.co/IsTLBDN08W) explains how to create an **agentic RAG query engine** using **NVIDIA's NIM microservices** for efficient open-source model inference.
   - It covers constructing a query router for complex questions and implementing sub-question queries, streamlining the process of handling intricate inquiries.
- **LlamaIndex Workflow Explained**: A comprehensive guide on the [LlamaIndex workflow](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) details how event-driven abstractions can chain multiple events through steps using a `@step` decorator.
   - Workflows allow for building diverse processes like agents or **RAG flows**, with automatic instrumentation for observability via tools like **Arize Phoenix**.
- **Hiring AI NLP Engineer**: **Nikkole**, a CTO of an AI startup, shared that they are looking for an AI **NLP Engineer** with a salary range of **$95k-$115k** for a **W2 contract**.
   - Interested candidates were advised to connect via [LinkedIn](https://www.linkedin.com/in/nikkole-scruggs), as direct messages are only accepted there.
- **Seeking Resources for Custom LLM**: A member is looking for recommendations on resources to perform with an open source **LLM** tailored to their custom preference dataset.
   - They requested suggestions from the community to *enhance their understanding and implementation*.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **MicroDiT Replication Completion**: User announced the completion of their [MicroDiT replication](https://x.com/SwayStar123/status/1854884660981219399) and shared download links for the **model weights** and **inference script**.
   - They credited **FAL** for providing the necessary compute resources, stating, *'I think I might be cooking.'*
- **Bonnie and Clyde Soundtrack Video Shared**: A YouTube video titled *'LOST SOUNDTRACK - BONNIE AND CLYDE'* was shared, featuring a description of Bonnie Parker's romance with ex-con Clyde Barrow and their violent crime spree.
   - The video can be viewed [here](https://youtu.be/e6UAI_P1Mlk), highlighting a narrative of love and crime.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Deadline Alert for Computing Resources**: The application deadline for **Computing Resources** is at the end of day **November 25th PST** with a **1-2 week processing delay** anticipated after submission.
   - **Participants** are encouraged to submit their applications early to ensure timely processing.
- **Urgent Call to Action for Participants**: Members are urged to act promptly to avoid missing the **November 25th** deadline for resources.
   - Early submission is vital to ensure adequate **processing time**.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Data Council '25 CFP Opens for a Week**: The **Data Council '25 CFP (Call for Proposals)** is open for another week, inviting developers to showcase their ML/AI projects. For more details, visit the [Data Council CFP page](https://www.datacouncil.ai/cfp-2025).
   - This event is anticipated to feature several engaging talks and hackers, promoting innovative discussions in the ML/AI community.
- **ML/AI App Talks Set to Inspire**: **Data Council '25** will host a series of talks on **ML/AI applications**, highlighting the latest advancements in the field.
   - Participants are encouraged to present their ML/AI app developments, fostering active collaboration and knowledge sharing.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jurassic's 'summarize-by-segment' Endpoint Deprecation**: A member expressed frustration over the sudden deprecation of the **Jurassic 'summarize-by-segment'** endpoint, which they had relied on for essential business services ahead of the announced **11/14** date.
   - They described the unexpected change as a **pain point**, highlighting its impact on workflows.
- **Transitioning to the New Jamba Model**: A user requested guidance on utilizing the new **Jamba model** to replicate the functionality of the deprecated endpoint, especially for URL content segmentation.
   - They emphasized the need for assistance in adjusting **URL parameters** to effectively extract content.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1304174343965769771)** (410 messages🔥🔥🔥): 

> - `Microsoft LightBGM`
> - `Hugging Face models for IFC files`
> - `Suno song creating tools`
> - `AI game development and scripting`
> - `Maintaining context in LLM conversations` 


- **Microsoft LightBGM support inquiry**: A user inquired about the ability of Microsoft LightBGM to support repeating indices for time series prediction datasets.
   - No additional responses or insights were shared on this topic.
- **IFC file manipulation using Hugging Face models**: A member asked if anyone knew of methods to use Hugging Face models for manipulating IFC files.
   - No solution or guidance was provided in the discussion.
- **Searching for song creation tools like Suno**: A user expressed interest in finding song creation tools similar to Suno, which is used for generating music.
   - Another member shared a link to MusicGen Plus++ as a potential alternative, mentioning its capabilities.
- **Exploring AI for game development and script writing**: Participants discussed various tools and frameworks for AI-driven game development, mentioning Unity's terrain tools and scripting capabilities.
   - There was also an exchange regarding the use of AI for screenplay writing and voice acting.
- **Methods for maintaining context in LLM conversations**: A user posed a question about effective methods for keeping context during conversations with LLMs, sharing their experiences with various approaches.
   - Responses included ideas such as serialization of inputs and outputs or using hybrid models to manage context effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ekNDPjC3CKWWd3jd2_V9QGTJSbvHKIZ2">Google Colab</a>: no description found</li><li><a href="https://pjlab-songcomposer.github.io/">SongComposer: A Large Language Model for Lyric and Melody Generation in Song Composition</a>: no description found</li><li><a href="https://arxiv.org/abs/2312.15166">SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling</a>: We introduce SOLAR 10.7B, a large language model (LLM) with 10.7 billion parameters, demonstrating superior performance in various natural language processing (NLP) tasks. Inspired by recent efforts t...</li><li><a href="https://huggingface.co/spaces/LinaDaniels/fast-stable-diffusion">Fast Stable Diffusion - a Hugging Face Space by LinaDaniels</a>: no description found</li><li><a href="https://tenor.com/view/spongebob-patrick-patrick-star-broke-poor-gif-14729256">Spongebob Patrick GIF - Spongebob Patrick Patrick Star - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lambdalabs.com/lambda-stack-deep-learning-software">Lambda Stack: an AI software stack that's always up-to-date</a>: Lambda Stack provides a one line installation and managed upgrade path for: PyTorch, TensorFlow, CUDA, cuDNN, and NVIDIA Drivers. It's compatible with Ubuntu 20.04 LTS, 18.04 LTS, and 16.04 LTS. No mo...</li><li><a href="https://distill.pub/">Distill — Latest articles about machine learning</a>: Articles about Machine Learning</li><li><a href="https://huggingface.co/genmo/mochi-1-preview">genmo/mochi-1-preview · Hugging Face</a>: no description found</li><li><a href="https://www.shadertoy.com/view/MdBGzG">Shadertoy</a>: no description found</li><li><a href="https://tenor.com/view/ghostbuster-toaster-gif-5319546">Ghostbuster Toaster GIF - Ghostbuster Toaster - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/Mar2Ding/songcomposer_sft">Mar2Ding/songcomposer_sft · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/prodia">prodia (Prodia Labs)</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=g39AagVW0s0">How I made AI Generated Rick and Morty Episodes</a>: 🕹 Get a browser that’s literally better at everything: https://operagx.gg/CodeBullet2  Sponsored by Opera GX!Check out the live stream: @codebulletsdayoff58...</li><li><a href="https://huggingface.co/spaces/LinaDaniels/fast-stable-diffusion/discussions/1">LinaDaniels/fast-stable-diffusion · Update README.md</a>: no description found</li><li><a href="https://www.procedural-worlds.com/">Procedural Worlds: World Creation for Everyone</a>: no description found</li><li><a href="https://youtu.be/BFld4EBO2RE">Painting a Landscape with Maths</a>: Today we are painting a landscape using mathematics.Support this channel: https://www.patreon.com/inigoquilezBuy this painting in a metal, canvas or photogra...</li><li><a href="https://www.blender.org">blender.org - Home of the Blender project - Free and Open 3D Creation Software</a>: The Freedom to Create</li><li><a href="https://docs.unity3d.com/Manual/terrain-Tools.html">Unity - Manual: Terrain tools</a>: no description found</li><li><a href="https://create.roblox.com/docs/studio/terrain-editor">Tweet from Terrain Editor | Documentation - Roblox Creator Hub</a>: The Terrain Editor tools generate and sculpt realistic terrain environments such as mountains, bodies of water, grass-covered hills, or a flat desert.</li><li><a href="https://www.minecraft.net/en-us">Welcome to the official site of Minecraft</a>: Explore new gaming adventures, accessories, &amp; merchandise on the Minecraft Official Site. Buy &amp; download the game here, or check the site for the latest news.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1304368389518524426)** (7 messages): 

> - `SFT Learning`
> - `Machine Learning Resources`
> - `BPE Tokenizer Visualization`
> - `FastBert Usage` 


- **Awesome Growth in SFT Understanding**: A user shared their journey with **SFT**, stating they have been working with it since February and are now getting the hang of it after finding the right dataset.
   - *Very cool stuff you can do* once you master the necessary components.
- **D2L for Machine Learning Beginners**: An experienced member recommended [d2l.ai](https://d2l.ai/) as a crucial resource for those starting their **machine learning** journey, highlighting its interactive features with math and code.
   - They emphasized that a blend of *mathematics, figures, and real datasets* enriches the learning experience.
- **Collaboration Request for BPE Visualizer**: A user invited others to help enhance their project on [BPE Tokenizer Visualizer](https://github.com/mdabir1203/BPE_Tokenizer_Visualizer), mentioning its function in LLMs.
   - Another member stated they prefer using **FastBert** first, expressing interest in the BPE methodology but wanting to test it for themselves.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://d2l.ai/)">Dive into Deep Learning &#8212; Dive into Deep Learning 1.0.3 documentation</a>: no description found</li><li><a href="https://github.com/mdabir1203/BPE_Tokenizer_Visualizer">GitHub - mdabir1203/BPE_Tokenizer_Visualizer: A Visualizer to check how BPE Tokenizer in an LLM Works</a>: A Visualizer to check how BPE Tokenizer in an LLM Works - mdabir1203/BPE_Tokenizer_Visualizer
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1304368400595812392)** (14 messages🔥): 

> - `User Spamming Concerns`
> - `Token Project Discussion`
> - `Scam Alert`
> - `Data Council '25 Call for Proposals` 


- **Spamming Issues in Channel**: Members discussed concerns about excessive notifications with one suggesting to *slow down a bit*.
   - There was a call to stop spamming, particularly from @cakiki, who pointed out the notifications issue.
- **Token Credibility Questioned**: A member questioned whether a certain token was associated with a project, prompting skepticism from others.
   - The discussion highlighted that a bad influencer had been promoting it, leading to claims of it being a scam.
- **Scam Warning Issued**: Concerns were raised about a potential scam related to a token, with members verifying its dubious nature.
   - Acknowledgments were given by members who expressed gratitude for confirming the scam allegations.
- **Call for Proposals at Data Council '25**: An announcement was made regarding the open Call for Proposals for AI applications at [Data Council '25](https://www.datacouncil.ai/cfp-2025).
   - Members were encouraged to share their projects and engage with cool LLM & AI hackers at the event.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1304234565887463516)** (12 messages🔥): 

> - `AI-generated programmatic ads`
> - `AI/ML workflow platform`
> - `PostgreSQL text optimization`
> - `HF Space for OS-ATLAS` 


- **Challenging the GPU bubble narrative**: A recent analysis argues that **AI-generated programmatic audio/video ads** will create massive infrastructure demands, predicting a **$3T opportunity by 2030**.
   - Early data suggests **5-10x performance improvements** and **90% cost reductions**, calling for technical community feedback on scaling challenges at [this link](https://chrisbora.substack.com/p/the-3-trillion-ai-opportunity-everyone).
- **Skepticism on AI-generated ads**: A member expressed skepticism about the feasibility of **AI-generated ads**, questioning the ability of text-to-video generation to capture niche content effectively.
   - They emphasized the need for **ads that resonate** with audiences and provided several links to examples of impactful advertising.
- **New AI/ML Workflow Platform Development**: A developer is working on a platform to create **AI/ML workflows** via an interactive UI that integrates models from Huggingface and LLMs.
   - The community is invited to test the project available on [GitHub](https://github.com/farhan0167/otto-m8) for feedback on its potential value.
- **PostgreSQL Text Field Optimization**: A technical discussion clarified that in **PostgreSQL**, there is no need to differentiate between `String(255)` and `Text` as both are optimized similarly.
   - A member shared their misunderstanding about character limits, learning that such distinctions stem from outdated database practices.
- **OS-ATLAS for Generalist GUI Agents**: An announcement for **HF Space for OS-ATLAS** was made, introducing a foundation action model for generalist GUI agents.
   - More information can be found [here](https://huggingface.co/spaces/maxiw/OS-ATLAS), with potential implications for future AI developments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chrisbora.substack.com/p/the-3-trillion-ai-opportunity-everyone">The $3 Trillion AI Opportunity Everyone Missed</a>: Why Today&#x27;s &#x27;GPU Bubble&#x27; Is Actually Massive Under-Investment</li><li><a href="https://huggingface.co/spaces/maxiw/OS-ATLAS">OS ATLAS - a Hugging Face Space by maxiw</a>: no description found</li><li><a href="https://github.com/farhan0167/otto-m8">GitHub - farhan0167/otto-m8: Low Code AI Automation Platform</a>: Low Code AI Automation Platform. Contribute to farhan0167/otto-m8 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

noaroggendorff: yet
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1304398982071455776)** (1 messages): 

> - `Fine Tuning Vocabulary Size` 


- **Seeking Help on Vocabulary Size Fine Tuning**: A member is working on a fine tuning task that involves increasing **vocabulary size** and is asking if anyone has experience with this to provide assistance.
   - They encouraged others to feel free to **pm** them for questions or support.
- **Interest in Vocabulary Expansion Techniques**: The same member expressed interest in exploring different techniques for **expanding vocabulary** during fine tuning tasks.
   - They mentioned being open to suggestions and recent advancements in methods that could aid in their project.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1304284039297105990)** (1 messages): 

> - `ComfyUI usage`
> - `Hacking Forge`
> - `SD3.5 Feature Request` 


- **Consider ComfyUI for local environment**: A suggestion was made to try [ComfyUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16590) for setting up a local environment instead of other methods.
   - This comes amidst discussions about stability and user experience in using the features of SD3.5.
- **Explore Hacking Forge for enhancements**: Another option discussed was to [hack Forge](https://github.com/lllyasviel/huggingface_guess/pull/1) as a means to implement new features.
   - This approach includes adding SD3.5 which could potentially hijack existing processes associated with SD3.
- **Feature Request for SD3.5**: Members raised questions regarding a [Feature Request](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16590) for improved support for SD3.5.
   - There were checks to ensure that no existing issues overlapped with this feature proposal, aiming for clarity in enhancements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/16590">[Feature Request]: Support for SD3.5 · Issue #16590 · AUTOMATIC1111/stable-diffusion-webui</a>: Is there an existing issue for this? I have searched the existing issues and checked the recent builds/commits What would your feature do ? SD3.5: https://huggingface.co/stabilityai/stable-diffusio...</li><li><a href="https://github.com/lllyasviel/huggingface_guess/pull/1">adding SD3.5 by graemeniedermayer · Pull Request #1 · lllyasviel/huggingface_guess</a>: adding SD3.5 (this might hijack SD3)
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1304321529382834220)** (2 messages): 

> - `Cinnamon AI`
> - `Kotaemon RAG tool`
> - `Live broadcast on X` 


- **Live Broadcast with Cinnamon AI Team**: The **Cinnamon AI** team, creators of the viral **RAG tool** called **Kotaemon**, went live on X at **10 PM PST**.
   - Viewers were invited to join the discussion through this [link](https://lnkd.in/giiNKviE).
- **Kotaemon's Viral Impact**: **Kotaemon** has gained significant attention as a **viral RAG tool**, attracting users eager to learn more about its features.
   - The Cinnamon AI team discussed its unique attributes and the user feedback they've received during the broadcast.


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1304264901229023252)** (1 messages): 

> - `New Rankings Page`
> - `MythoMax Performance` 


- **New Rankings Page Launched**: The **New Rankings page** has been introduced to display completion request counts over time.
   - Users can expect a redesign of this page in the future, enhancing the data presentation.
- **MythoMax Remains Dominant**: **MythoMax** continues to hold the title of the <:hugging_king:936261298273001503>, showcasing its strong position in request counts.
   - The community is acknowledging **MythoMax**'s consistent performance despite impending changes to the rankings page.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1304178735863435346)** (303 messages🔥🔥): 

> - `OpenRouter performance`
> - `Rate limits`
> - `Model comparisons`
> - `API issues`
> - `Command R+ alternatives` 


- **OpenRouter encounters performance issues**: Users reported freezing and crashing issues when using OpenRouter on mobile devices, particularly on Android 12, leading to frustration.
   - The performance issues may be related to specific chatroom activities or memory usage, as other sites work fine under similar conditions.
- **Confusion over rate limits and credits**: There was confusion regarding the rate limit structure, where users debated the relationship between credits and requests per second, with a cap at 200.
   - Users clarified that credits are not refundable, and the displayed dollar amounts are not a one-to-one match due to associated fees.
- **Discussions on effective AI interaction**: Users shared techniques to prompt AI models effectively, suggesting that playful approaches, like offering virtual rewards, can lead to better responses.
   - The conversation included observations on Gemini 1.5's performance, noting some models are significantly better than others for task-specific outcomes.
- **Exploring alternatives to Command R+**: After experimenting with various models, users discussed alternatives to Command R+, expressing interest in options like Hermes 405B, Euryale, and Mythomax.
   - Some users mentioned the affordability of Rocinante 12B and questioned whether mythomax on OpenRouter differs from its Chub counterpart.
- **Command R+ model inquiry**: Questions arose about the quality of Command R+ and its comparison to other models, suggesting more effective alternatives like Claude Sonnet.
   - Users noted discrepancies in performance among different providers for models like Wizard, suggesting further testing is needed.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: Set limits on model usage</li><li><a href="https://sillytavern.app/">SillyTavern - LLM Frontend for Power Users</a>: no description found</li><li><a href="https://ai.google.dev">no title found</a>: no description found</li><li><a href="https://www.reddit.com/r/singularity/comments/1gm9vin/enough_politics_its_ai_time/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.alibabacloud.com/help/en/model-studio/developer-reference/billing-for-tongyiqianwen">
 Calculate and view the bills of Qwen - Alibaba Cloud Model Studio - Alibaba Cloud Documentation Center

</a>: no description found</li><li><a href="https://mistral.ai/news/mistral-moderation/">Mistral Moderation API</a>: We are introducing our new moderation service enabling our users to detect undesirable text content along several policy dimensions.</li><li><a href="https://mistral.ai/news/batch-api/">Mistral Batch API</a>: Lower cost API for AI builders.</li><li><a href="https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post">Mistral AI API | Mistral AI Large Language Models</a>: Our Chat Completion and Embeddings APIs specification. Create your account on [La Plateforme](https://console.mistral.ai) to get access and read the [docs](https://docs.mistral.ai) to learn how to use...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1304328012870844518)** (4 messages): 

> - `Integration Beta Feature`
> - `OpenRouter Monetization` 


- **Integration Beta Feature Access Request**: Multiple users have requested access to the **integration beta feature**, demonstrating significant interest in its rollout.
   - One user received a response mentioning a forthcoming method to **click a button** for addition to the beta list, suggesting improvements in user experience.
- **OpenRouter's Monetization Strategy**: A user inquired about how **OpenRouter** plans to monetize its **bring your own key system**, raising questions about its economic viability.
   - This concern highlights a vital discussion point regarding the sustainability of the platform and potential revenue streams.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1304176924834136175)** (241 messages🔥🔥): 

> - `Subscription Prices Discussion`
> - `Mobile Device Specs`
> - `Comparison of AI Models`
> - `Discount Code Issues`
> - `App Functionality Problems` 


- **Debate on Subscription Prices**: Users expressed frustration over the low subscription prices of services, with some feeling it should be more expensive to reflect value.
   - Concerns were raised about the sustainability of pricing structures and the desire for better service quality.
- **Specifications of Mobile Devices**: A user discussed their experience with high-spec mobile devices, comparing models like Snapdragon 8 Elite and older generations.
   - The conversation included consideration of devices available in different regions and user preferences for performance.
- **AI Model Comparisons and Preferences**: Users compared the capabilities of various AI models such as Opus, Sonnet, and Claude, expressing preferences for creative output quality.
   - There were discussions on the differences in performance and output style between these models, with some users missing the original Opus.
- **Issues with Discount Codes**: Several users reported problems with invalid discount codes received from newsletters, particularly mentioning Kevin Rose's newsletter.
   - Attempts to resolve discount code issues prompted suggestions to contact support for assistance.
- **App Functionality and Bugs**: Users reported issues with the mobile app's functionality, specifically related to loading and producing answers.
   - Concerns about missing features, such as the Focus option and reasoning mode, were flagged as possible bugs or recent changes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://duckduckgo.com/?t=h_&q=chat&ia=chat">chat at DuckDuckGo</a>: no description found</li><li><a href="https://tenor.com/view/love-is-war-kaguya-sama-chika-fujiwara-laugh-gif-17152820588298782251">Love Is War Kaguya Sama GIF - Love is war Kaguya sama Chika - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://notebooklm.google.com/">no title found</a>: no description found</li><li><a href="https://www.youtube.com/@AICodeKing/videos">AICodeKing</a>: Facing the future with AI!   Here you&#39;ll find content about multiple AI Tools that are actually useful and sometimes free.  For Ads / Sponsorship : Comment on any of my videos and I&#39;ll reach b...</li><li><a href="https://x.com/testingcatalog/status/1854967421402333344?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨:  Looks like Perplexity started rolling out Pro Shopping 🔥  How much Black Friday traffic will flow through perplexity this year? 👀  Quoting Raunak Chowdhuri (@raunakdoesdev)   @perplexi...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1304198951947341856)** (8 messages🔥): 

> - `Gladia Functionality`
> - `Neohtop Insights`
> - `Studies in Psychology`
> - `AI with Infinite Memory`
> - `USA Election Discussion` 


- **Exploring Gladia's Functionality**: A member shared insights on how [Gladia functions](https://www.perplexity.ai/search/comment-fonctionne-gladia-http-7.4QSxo0QkeYcztxP90ryg) and its practical applications in various scenarios.
   - The discussion highlighted **key features** that set it apart from other AI tools.
- **Insightful Discussion on Neohtop**: There was a discussion regarding [Neohtop](https://www.perplexity.ai/search/neohtop-7u_gYkcsSrGyGeshQmI4uA) and its implications for the tech industry.
   - Members evaluated its impact, with several pointing out **innovative uses** in data handling.
- **Studies in Psychology Mentioned**: A link was shared about [potential studies in psychology](https://www.perplexity.ai/search/there-might-be-studies-in-psyc-r9xkTwHWTnq5H3LU2w31dg#0) that could influence AI development.
   - The conversation suggested **relevance** of these studies to current AI methodologies.
- **AI with Infinite Memory Concept**: A member introduced the topic of [AI with infinite memory](https://www.perplexity.ai/page/microsoft-ceo-ai-s-infinite-me-zrmHnWQmRkylfKAyPOLj5w), such as proposed by Microsoft's CEO.
   - The concept raised intriguing questions around **data retention** and practical applications for future AI models.
- **Insightful USA Election Discussion**: There was a key sharing regarding the [USA election](https://www.perplexity.ai/search/usa-election-Wl64yM37SgavIH68ZcwwBQ) and its implications for technology and AI.
   - Members debated on how **technological influence** might shape voter engagement and decision-making.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1304320552873492513)** (10 messages🔥): 

> - `Citations in Perplexity API`
> - `Default rate limits increase`
> - `Citations visibility issues`
> - `API discussions on GitHub` 


- **Citations made public in Perplexity API**: The Perplexity team announced the public availability of **citations** in the API, effective immediately.
   - *This is not a breaking change*, and the return_citations parameter will no longer affect requests.
- **Increased default rate limits for Sonar models**: The default rate limit for **Sonar online models** has been increased to **50 requests/minute** for all users.
   - This enhancement aims to improve user experience and access to the API.
- **Citations mysteriously disappeared**: Users reported that citations were returned initially but then suddenly **disappeared** from both the API and labs.perplexity.ai.
   - There are concerns that **Perplexity might have disabled citations** again without notice.
- **Clarification on API token limits**: A user inquired whether **1M tokens** refers to input tokens and if the pricing applies similarly to output tokens.
   - This highlights ongoing discussions about understanding the API's usage and pricing structure.
- **Discussion on sources for Perplexity**: A GitHub discussion was referenced regarding when the citation feature will be **out of beta**.
   - This suggests ongoing user interest in the official status and functionality of the citation feature.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://perplexity.mintlify.app/changelog/changelog#citations-public-release-and-increased-default-rate-limits">no title found</a>: no description found</li><li><a href="https://github.com/ppl-ai/api-discussion/discussions/54">sources for perplexity · ppl-ai/api-discussion · Discussion #54</a>: wondering when it will be out of beta?
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1304174293239730186)** (203 messages🔥🔥): 

> - `Job fulfillment and career moves`
> - `Experiences transitioning between companies`
> - `Team dynamics and internal transfers`
> - `Background checks and employment history`
> - `Hiring processes at major tech companies` 


- **Struggles with Job Misalignment**: A member expressed dissatisfaction with their current ML role at Meta, stating it doesn’t align with their background, leading to potential career harm.
   - They have an offer to return to Google where they see more alignment and opportunities for promo due to their history with the team.
- **Dilemma of Career Decisions**: Members discussed the tough decision to stay at a job that feels unfulfilling versus returning to a previous employer with potential promo opportunities.
   - Concerns about having a 'target on their back' after expressing desire to leave were raised, complicating career dynamics.
- **Challenges with Job Switching**: The conversation also focused on the difficulties of navigating job switches, with one member noting their unsuccessful attempt to transfer teams at Google.
   - Recommendations included leveraging existing connections and referrals to explore opportunities while considering geographical constraints.
- **Navigating Background Checks**: Concerns were expressed about how to handle employment gaps or misrepresented roles during background checks for new job applications.
   - Members provided varying opinions on the necessity and implications of such transparency in resumes.
- **Perspectives on Tech Company Tools**: Members shared their experiences with various cloud platforms, particularly critiquing the user interfaces of GCP and AWS.
   - Despite its company reputation, GCP's UI was described as unresponsive and overloaded, sparking discussions on usability across platforms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/the-pursuit-of-happiness-will-smith-cry-tears-of-joy-happy-gif-10725846">The Pursuit Of Happiness Will Smith GIF - The Pursuit Of Happiness Will Smith Cry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://mempko.com">Mempko</a>: no description found</li><li><a href="https://blog.mempko.com">Maxim Khailo&#x27;s Writing</a>: A blog about technology and finance by a seasoned tech veteran.</li><li><a href="https://www.ftc.gov/legal-library/browse/rules/noncompete-rule">Noncompete Rule</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1304174139262763008)** (16 messages🔥): 

> - `Forward Gradient for Flash Attention`
> - `Papers and Their Importance`
> - `Curvature and Tangent Spaces from Diffusion Geometry` 


- **Exploring Forward Gradients in Flash Attention**: A discussion arose about the forward gradient for **flash attention**, with members sharing formulas for **normal attention** gradients.
   - One source suggested looking into appendix F of [this paper](https://arxiv.org/abs/2410.11081) for deeper insights on Jacobian-vector products.
- **Determining Important Papers in AI**: A member questioned how others decide which papers are significant in an ever-increasing flood of research.
   - Responses highlighted relying on recommendations from peers and mentioned the **Eleuther Discord** channel for guidance.
- **Innovative Approaches in Diffusion Geometry**: A new paper introduced novel estimators to compute curvature and tangent spaces from data, improving robustness against noise and sparsity.
   - This research, detailed in [this paper](https://arxiv.org/abs/2411.04100), claims to outperform current methods particularly when dealing with less than ideal data.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.04100">Manifold Diffusion Geometry: Curvature, Tangent Spaces, and Dimension</a>: We introduce novel estimators for computing the curvature, tangent spaces, and dimension of data from manifolds, using tools from diffusion geometry. Although classical Riemannian geometry is a rich s...</li><li><a href="https://arxiv.org/abs/2411.04996">Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models</a>: The development of large language models (LLMs) has expanded to multi-modal systems capable of processing text, images, and speech within a unified framework. Training these models demands significant...</li><li><a href="https://arxiv.org/abs/2410.11081">Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models</a>: Consistency models (CMs) are a powerful class of diffusion-based generative models optimized for fast sampling. Most existing CMs are trained using discretized timesteps, which introduce additional hy...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1304501254592139315)** (4 messages): 

> - `Inverse Interpretability`
> - `Refusal in LLMs`
> - `Symbolic Representation of Neural Networks`
> - `Behavioral Changes in AI Models` 


- **Exploring Inverse Interpretability Concept**: A member asked if any research exists on **inverse interpretability**, focusing on interventions in interpretable representations and adjusting model weights based on those changes.
   - This led to a query on the challenges of aligning the modified symbolic equation to the model's weights in a deep neural network.
- **Refusal Mechanism in LLMs Unveiled**: A related discussion referenced a post outlining that **refusal** in LLMs is controlled by a specific direction in the model's residual stream, which can be erased to alter refusal behaviors.
   - This work is part of the ML Alignment & Theory Scholars Program, led by Neel Nanda, with a paper forthcoming on [arXiv](https://arxiv.org/abs/2406.11717).
- **Challenges in Adjusting Neural Network Weights**: Another user elaborated on the process of extracting symbolic equations from neural networks and the complexity of ensuring model weights remain consistent post-intervention.
   - They raised concerns about potential side effects when nudging behaviors rather than erasing them completely, particularly in high-dimensional spaces.
- **Symbolic Equation Interventions in Physics Simulations**: The members discussed using symbolic equations to perform interventions, such as adjusting coefficients to achieve desired behaviors in models, particularly in the context of **physics simulations**.
   - However, uncertainty arose regarding how often such extractions are applied successfully within that domain.



**Link mentioned**: <a href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction">Refusal in LLMs is mediated by a single direction — LessWrong</a>: This work was produced as part of Neel Nanda&#x27;s stream in the ML Alignment &amp; Theory Scholars Program - Winter 2023-24 Cohort, with co-supervision from…

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1304434333087432759)** (2 messages): 

> - `nlls with <bos> token`
> - `Meta Llama 3.1`
> - `Salamandra model card`
> - `Saving results issues` 


- **NLLs vary significantly with <bos> token**: An observation was made about a noticeable difference in **negative log likelihoods (nlls)** when adding a **<bos> token** for non-conversational input, raising questions about best practices.
   - The inquiry focuses on whether there are specific guidelines to follow for handling **multiple-choice/loglikelihood tasks**.
- **Meta Llama 3.1 features and architecture**: The **Meta Llama 3.1** model collection comprises multilingual LLMs optimized for dialogue, available in sizes of **8B, 70B, and 405B**.
   - It highlights the use of an auto-regressive transformer architecture with tuning through **supervised fine-tuning (SFT)** and **reinforcement learning with human feedback (RLHF)**.
- **Discussion on Salamandra model**: The **Salamandra** model, pre-trained from scratch, comes in various sizes including **2B, 7B, and 40B parameters**, with both base and instruction-tuned variants.
   - Links to the model index and **GitHub repository** for training scripts and configuration files were provided to facilitate further exploration.
- **Challenges with saving results**: A user noted that even with **write_out=True** and **log_samples=True**, results sometimes fail to save while using the **Meta Llama 3.1** model.
   - This raises concerns about the reliability of the output process, leading to frustration when *nothing comes out* during attempts to log results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Llama-3.1-8B">meta-llama/Llama-3.1-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/BSC-LT/salamandra-7b-instruct">BSC-LT/salamandra-7b-instruct · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1304178846769221645)** (21 messages🔥): 

> - `Benchmarking NeoX vs LitGPT`
> - `LitGPT usage in projects`
> - `FSDP vs ZeRO`
> - `GPT-NeoX features`
> - `Training with limited resources` 


- **Benchmarking NeoX against LitGPT**: A member inquired about any benchmarks comparing **NeoX** and **LitGPT** for performance differences in training speed and stability.
   - Another mentioned the lack of tests beyond the 1.1B parameter scale in **LitGPT's** repository.
- **LitGPT used in projects like Amber**: It was noted that **Amber (7B)** is based on **lit-llama**, which transitioned to **litgpt**, indicating its usage in prominent projects.
   - Members confirmed that **llm360** has utilized **LitGPT** previously, although they have switched to a [custom Megatron-based library](https://github.com/LLM360/k2-train).
- **Debating FSDP and ZeRO implementations**: A discussion highlighted that **FSDP** and **ZeRO** essentially refer to different brandings of the same technology, yet differ in implementation details which can affect training behavior.
   - One member pointed out that adjustments in how precision is handled can lead to divergences in loss curves during training.
- **Perks of GPT-NeoX for modeling**: **GPT-NeoX** offers unique modeling features such as **RWKV and Mamba layers**, along with native RLHF support, which could be appealing based on the user's project goals.
   - This library's community support is noted to be responsive, providing faster iteration cycles from bug reporting to receiving help.
- **Training small models with limited hardware**: A member suggested a competitive approach to training smaller models using different libraries across multiple nodes to assess performance quickly.
   - The notion of racing to get models running efficiently adds a fun element to overcoming setup challenges.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1304232594816110652)** (175 messages🔥🔥): 

> - `ComfyUI connection issues`
> - `Image generation concerns with inpainting`
> - `Model recommendations for ComfyUI`
> - `Flux model advantages`
> - `Baseline models vs. merged models` 


- **ComfyUI Connection Troubleshoot**: Users discussed troubleshooting a **Connection denied** error while using ComfyUI, with suggestions to check antivirus and firewall settings.
   - One user confirmed using **Windows Defender**, which might be blocking connectivity, prompting further checks in security settings.
- **Inpainting Complications**: A user raised concerns that using **adetailer** with inpainted images resulted in loss of detail in the previously inpainted areas.
   - Others recommended setting inpainting parameters to **mask only** to avoid affecting the entire image.
- **Model Recommendations and Their Uses**: The community highly recommends using the **Flux** base model for its balance of quality and speed while discussing the practicality of upgrading from **SD 1.5**.
   - Flux and other models like **SD3.5** are discussed, emphasizing their performance and niche functionalities.
- **Debate Over Merged Models vs. Fresh Models**: Participants noted that while merged models like **Realvis** can produce good results, base models often outperform them when carefully prompted.
   - Concerns were voiced regarding the efficacy of merged models and their reception within the user community.
- **The Evolving Landscape of Model Support**: Users reflected on the historical development of models, highlighting that **SD 1.5** maintains a robust support base with numerous research papers.
   - The discussion touched upon the growing number of tools enhancing **SD 1.5**, even as **SDXL** slowly catches up in terms of tools and papers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai-social-media-post-generator.onrender.com/">Free AI Social Media Post Generator</a>: no description found</li><li><a href="https://huggingface.co/models?other=base_model:finetune:stabilityai%2F">Models - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/youknownothing/realDream_STOIQONewreality/commit/f5d8fadc6b1e78130050509bb8d165d362b5d304">Create README.md · youknownothing/realDream_STOIQONewreality at f5d8fad</a>: no description found</li><li><a href="https://huggingface.co/models?other=base_model:finetune:stabilityai%2Fstable-diffusion-3.5-large&sort=trending">Models - Hugging Face</a>: no description found</li><li><a href="https://civitai.com/models/895985/flux-devschnell-base-unet-google-flan-fp16nf4-fp32fp8">FLUX Dev/Schnell (Base UNET) + Google FLAN FP16/NF4-FP32/FP8 - FLUX_Dev-FLAN-FP16 | Flux Checkpoint | Civitai</a>: Full Checkpoint with improved TE do not load additional CLIP/TE FLUX.1 (Base UNET) + Google FLAN NF4 is my recommended model for quality/speed bala...</li><li><a href="https://civitai.com/models/161068?modelVersionId=498484">STOIQO NewReality 🟡 FLUX, SD3.5, SDXL, SD1.5 - 🔵 XL Light 1.0 | Stable Diffusion XL Checkpoint | Civitai</a>: 🟡: Flux Models 🟢: SD 3.5 Models 🔵: SD XL Models 🟣: SD 1.5 Models 🔴: Expired Models 🟡STOIQO NewReality is a cutting-edge model designed to generate ...</li><li><a href="https://civitai.com/models/161068/stoiqo-newreality-flux-sd35-sdxl-sd15">STOIQO NewReality 🟡 FLUX, SD3.5, SDXL, SD1.5 - SD3.5 🟢 PreAlpha | Stable Diffusion Checkpoint | Civitai</a>: 🟡: Flux Models 🟢: SD 3.5 Models 🔵: SD XL Models 🟣: SD 1.5 Models 🔴: Expired Models 🟡STOIQO NewReality is a cutting-edge model designed to generate ...</li><li><a href="https://huggingface.co/youknownothing/realDream_STOIQONewreality/commit/f5d8fadc6b1e78130050509bb8d1">Create README.md · youknownothing/realDream_STOIQONewreality at f5d8fad</a>: no description found</li><li><a href="https://civitai.com/models/617609/flux1-dev">FLUX.1 [dev] - v1.0 | Flux Checkpoint | Civitai</a>: Do not download if you have not read all the suggestions Because it is heavy and requires more stuff than SD. And we have new way to run Flux ez wi...
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1304174057603989535)** (120 messages🔥🔥): 

> - `TEE HEE He bot functionality`
> - `Nous community engagement`
> - `VLM for handwriting to LaTeX conversion`
> - `Abliteration concept`
> - `Anthropic and Palantir deal` 


- **TEE HEE He bot's limitations**: The bot can only send **ETH** and doesn't support trading, though it can check its balance and send funds.
   - Plans are in place to expand its functionality, as noted by member discussions on potential services the bot may offer.
- **Nous seeks community engagement**: Communications emphasize the need for Nous to reclaim mindshare and engage the community regarding the TEE HEE He bot.
   - Members suggest ways for Nous to extend connections with the community, amidst discussions of fair token distribution.
- **VLM for handwriting to LaTeX progress**: Member shared progress on training a **VLM** based on **Llama 3.2 1B** for handwriting conversion, anticipating a release of a starter project soon.
   - The approach mentioned could theoretically apply to various modalities, prompting further interest in multimodal models.
- **Discussion on Abliteration**: Members discussed the concept of **abliteration** and its implications for LLMs, with some confirming its definition as a portmanteau of **ablate** and **obliterate**.
   - Related links were shared to clarify the concept, indicating its significance in the field of AI.
- **Anthropic and Palantir collaboration**: There was a debate about the implications of the **Anthropic-Palantir** deal, with some members questioning the motivations behind such partnerships.
   - Comments suggested a broader trend of tech companies making deals with government entities, prompting critiques from the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/karan4d/status/1854622598375600637">Tweet from huh (@karan4d)</a>: @owenli25 @tee_hee_he yeah each reboot will create new wallet. we&#39;ve seen the keys, so it&#39;s an integrity problem each time the keys get unencumbered. the solve for now is to make a new wallet ...</li><li><a href="https://x.com/NousResearch/status/1848397863547515216">Tweet from Nous Research (@NousResearch)</a>: no description found</li><li><a href="https://huggingface.co/blog/mlabonne/abliteration">Uncensor any LLM with abliteration</a>: no description found</li><li><a href="https://github.com/nousresearch/nousflash-agents">GitHub - NousResearch/nousflash-agents: Modular Agentic AI Architecture - NousResearch x Teleport (Flashbots)</a>: Modular Agentic AI Architecture - NousResearch x Teleport (Flashbots) - NousResearch/nousflash-agents
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1304184896683638874)** (5 messages): 

> - `llm-evaluation-harness`
> - `PyTorch model evaluation`
> - `HellaSwag dataset` 


- **Evaluating PyTorch Models with llm-evaluation-harness**: A user asked if there is a way to evaluate **PyTorch models** using the [llm-evaluation-harness](https://github.com/yourlink/repo), noting that it seems to primarily support models from **Hugging Face**.
   - Another member confirmed that they have only used it with Hugging Face models and suggested it might restrict support to those and APIs.
- **Challenges with HellaSwag Dataset**: The initial user expressed interest in understanding how the **HellaSwag** dataset evaluation works, particularly since it is a multiple-choice question dataset.
   - They commented that the *code appears disorganized* and asked for suggestions on how to implement evaluation logic for a PyTorch model.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1304188992698449950)** (1 messages): 

> - `Ferret-UI`
> - `Multimodal LLMs`
> - `Apple's UI comprehension` 


- **Ferret-UI revolutionizes mobile UI interaction**: Ferret-UI, the first UI-centric multimodal large language model, is designed for referring, grounding, and reasoning tasks, built on **Gemma-2B** and **Llama-3-8B** architectures. After training on a diverse dataset, it significantly improves the ability to comprehend mobile UI screens.
   - According to the [official paper](https://arxiv.org/pdf/2404.05719), Ferret-UI excels at executing complex UI tasks and outperforms **GPT-4V** on all elementary UI tasks.
- **Easy setup with Ferret-UI scripts**: Users can easily set up Ferret-UI by downloading a series of Python scripts such as `builder.py` and `inference.py` from the Hugging Face repository. Detailed instructions are available to ensure a seamless initiation into using this model.
   - The setup process requires downloading scripts with commands like `wget` for efficient local installation, making it user-friendly.
- **Enhanced understanding of UI screens**: The Ferret-UI model addresses the shortcomings of existing MLLMs by implementing methods to better understand and interact with UI screens' unique features like elongated aspect ratios and small objects. It utilizes region annotations to improve referring and grounding capabilities.
   - This specialized training allows **Ferret-UI** to interpret complex mobile interface tasks, enhancing its reasoning skills through a finely curated dataset.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b">jadechoghari/Ferret-UI-Gemma2b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/jadechoghari/Ferret-UI-Llama8b">jadechoghari/Ferret-UI-Llama8b · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1304188992698449950)** (1 messages): 

> - `Ferret-UI`
> - `Gemma-2B`
> - `Llama-3-8B`
> - `Multimodal LLMs`
> - `User Interface Comprehension` 


- **Ferret-UI Launches for Enhanced UI Tasks**: Ferret-UI is the first UI-centric **multimodal large language model (MLLM)** designed to perform **referring, grounding**, and **reasoning tasks**, built on **Gemma-2B** and **Llama-3-8B**.
   - According to the [research](https://arxiv.org/pdf/2404.05719) presented by Apple, Ferret-UI's extensive training enhances understanding of mobile UI screens with distinctive **visual features**.
- **Training Methodology for Ferret-UI**: The training samples for Ferret-UI were gathered from a variety of elementary UI tasks such as **icon recognition** and **text finding**, aided by region annotations for precise interactions.
   - The model exhibits superior performance, surpassing GPT-4V in **elementary UI tasks**, indicated by its extensive **benchmarking**.
- **Setup and Usage of Ferret-UI**: To use Ferret-UI, users must download several scripts such as `builder.py` and `inference.py` from the provided **Hugging Face links** for effective local operation.
   - The usage instructions emphasize the simplicity of integration into workflows, boosting productivity with **complex UI tasks**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/jadechoghari/Ferret-UI-Gemma2b">jadechoghari/Ferret-UI-Gemma2b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/jadechoghari/Ferret-UI-Llama8b">jadechoghari/Ferret-UI-Llama8b · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1304274515068583946)** (2 messages): 

> - `RAG usage in chat sessions` 


- **Exploring RAG for Chat Context**: A member suggested that using **RAG** could provide valuable context for enhancing the upcoming chat session.
   - They expressed their intent to implement this approach, looking for additional tips on optimizing the chat experience.
- **Seeking Tips for Effective Chat Sessions**: Another member inquired about **tips** for the chat session to improve engagement and output.
   - This shows an interest in community collaboration to share best practices for conducting the session.


  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1304174329533038592)** (63 messages🔥🔥): 

> - `Aider Usage Tips`
> - `Exponent AI Pair Programmer`
> - `Gemini 2.0 Launch`
> - `Model Comparison`
> - `Funding and Donations for Aider` 


- **Tips for Using Aider Effectively**: It was discussed that to familiarize with Aider, starting with cheaper models like **Qwen 2.5** is advisable before moving to more complex ones like **Sonnet/Haiku**.
   - One user mentioned feeling the need for a 'blackboard' feature in Aider to keep track of context and iteration more effectively during long projects.
- **Introducing Exponent: The AI Pair Programmer**: A member introduced **Exponent**, an AI pair programmer capable of performing software engineering tasks across environments with a special CLI for integration.
   - Its ability to learn from codebases and directly edit files in the filesystem was highlighted, making it a strong alternative to Aider.
- **Gemini 2.0 Could Be on the Horizon**: Rumors are circulating about Google's upcoming **Gemini 2.0** launch, which may feature a new model called **Gemini Pro 2.0** currently in testing.
   - Speculations on performance improvements and accessibility for advanced users only were raised, alongside concerns about its readiness.
- **Comparative Discussion of Coding Models**: Users compared the **Fireball-Meta-Llama-3.1** model against Aider, suggesting it is general purpose and can handle diverse coding tasks like JS/TS.
   - Concerns were expressed about the usability of new models, revealing that while they may offer advanced features, there might still be significant limitations.
- **Funding Opportunities for Aider Development**: The community discussed supporting Aider's development, mentioning that YouTube creators could be funded for creating content about Aider.
   - Suggestions included enabling GitHub donations, although there's uncertainty regarding non-code contributions being accepted by maintainers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.testingcatalog.com/google-gearing-up-for-gemini-2-0-launch-with-new-ai-model-in-testing/">Google gearing up for experimental Gemini 2.0 model launch</a>: Hidden within the latest update, a tantalizing new option has appeared in the model selection dropdown: Gemini Pro 2.0.</li><li><a href="https://huggingface.co/EpistemeAI/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-auto">EpistemeAI/Fireball-Meta-Llama-3.1-8B-Instruct-Agent-0.003-128K-code-ds-auto · Hugging Face</a>: no description found</li><li><a href="https://www.exponent.run/">Exponent</a>: Exponent is your AI Pair Programmer.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1304173784760193144)** (51 messages🔥): 

> - `Aider Model Architecture`
> - `RAG Integration with Qdrant`
> - `Emacs Keybindings for Aider`
> - `Exploring Aider Features`
> - `Using Aichat for RAG` 


- **Aider Model Architecture Confusion**: Members discussed issues selecting `openrouter` models as `--architect`, leading to confusion about argument requirements and their positioning.
   - A member clarified that `--architect` is a switch enabling architect mode using the main model without specifying the model name directly.
- **Setting Up RAG with Qdrant**: A user sought advice on integrating Aider's architecture with their Qdrant vector DB for RAG use, aiming to leverage external knowledge.
   - Another user suggested creating an API on top of Qdrant for querying, using CLI tools to interact with the database for context.
- **Discovering Aider's Features**: One member emphasized the potential of Aider's flexibility, encouraging exploration of various settings and the `/run` command for pulling external context.
   - The importance of understanding Aider's configurations was highlighted, particularly in maximizing its capabilities for specific tasks.
- **Utilizing Aichat for RAG Solutions**: Members discussed the use of Aichat for RAG, sharing ideas about extracting documentation context for better responses in Aider.
   - One member described a workflow involving scraping documentation into markdown files and using NotebookLM to generate context for Aider.
- **Using Custom Python CLI for Qdrant**: A suggestion was made to create a custom Python CLI for querying Qdrant, allowing easier integration into workflows with Aider.
   - The member noted that n8n could be used for populating the Qdrant index, while the CLI would handle querying tasks effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider is AI pair programming in your terminal</li><li><a href="https://blog.voyageai.com/2024/09/18/voyage-3/">voyage-3 &amp; voyage-3-lite: A new generation of small yet mighty general-purpose embedding models</a>: TL;DR – We are excited to announce voyage-3 and voyage-3-lite embedding models, advancing the frontier of retrieval quality, latency, and cost. voyage-3 outperforms OpenAI v3 large by 7.55% on aver…</li><li><a href="https://github.com/sigoden/aichat">GitHub - sigoden/aichat: All-in-one LLM CLI tool featuring Shell Assistant, Chat-REPL, RAG, AI tools &amp; agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more.</a>: All-in-one LLM CLI tool featuring Shell Assistant, Chat-REPL, RAG, AI tools &amp; agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more. - sigoden/aichat</li><li><a href="https://github.com/dubaigit/aider_split_install">GitHub - dubaigit/aider_split_install</a>: Contribute to dubaigit/aider_split_install development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1304284078891339836)** (3 messages): 

> - `Aider`
> - `Developed Approaches`
> - `Hacker News Discussion` 


- **Link to Hacker News about New Tool**: A member shared a link to [Hacker News](https://news.ycombinator.com/item?id=42078536) discussing a new tool that sparked interest.
   - This tool seems to capture attention due to its innovative approach and potential implications.
- **Author Unaware of Aider**: Discussion revealed that the author of the tool developed it without knowledge of **Aider**, highlighting intriguing similarities in approach.
   - One member noted the similarity, suggesting a shared direction in their methodologies.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1304182904573923358)** (61 messages🔥🔥): 

> - `LoRA vs Full Fine-tuning`
> - `4bit improvements`
> - `Multi-GPU Training`
> - `Feature Contribution Analysis Tools`
> - `Curriculum Learning in Fine-tuning` 


- **Analysis on LoRA vs Full Fine-tuning Paper**: A member shared insights on the paper titled *'LoRA vs Full Fine-tuning: An illusion of equivalence'*, emphasizing that **LoRA works if done right** and highlighting the need for proper rank settings.
   - Critiques were raised regarding the absence of SVD initialization testing and the contradictory claims about 'intruder dimensions' within LoRA and full fine-tuning models.
- **Discussion on 4bit Improvements**: A member discussed potential improvements in **4bit inference performance**, noting that most enhancements currently stem from reduced Python overhead rather than algorithmic changes.
   - They expressed intentions to enhance the dequant kernel for better resource occupancy during processing.
- **Clarification on Multi-GPU Training**: A user inquired about running example code across multiple GPUs, specifically mentioning **gpu0 and gpu1**, but was informed that multi-GPU support is not publicly available yet.
   - Another member noted that using **QLoRA** may allow fine-tuning for large models like Llama3-80b on a single H100 GPU.
- **Recommendation for Feature Contribution Analysis Tools**: A member sought alternatives to tools like SHAP, Lime, or Captum for analyzing feature contributions in LLM inference.
   - The conversation included recommendations and discussions around the capabilities of various libraries in providing insights on feature importance.
- **Curriculum Learning in Fine-tuning**: Members debated the effectiveness of **curriculum learning** over traditional fine-tuning, suggesting that a staged approach could yield better reasoning capabilities in models.
   - The discussion included advice on dataset concatenation and retaining knowledge from initial training phases to avoid forgetting key learned concepts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/infly/opencoder-672cec44bbb86c39910fb55e">OpenCoder - a infly Collection</a>: no description found</li><li><a href="https://x.com/abhi1thakur/status/1854825729348784437">Tweet from abhishek (@abhi1thakur)</a>: INTRODUCING Hugging Face AutoTrain Client 🔥 Now you can fine-tune SOTA models on all compatible dataset-model pairs on Hugging Face Hub using Python on Hugging Face Servers. Choose from a number of G...</li><li><a href="https://huggingface.co/infly">infly (infly-ai)</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1854992153992479165">Tweet from Daniel Han (@danielhanchen)</a>: My take on &#34;LoRA vs full finetuning: An illusion of equivalence&#34;  TLDR 1. Use alpha = 2*rank 2. Don&#39;t use too small ranks (rank=1 to 8) 3. Sensational title. Better title &#34;LoRA works i...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1304271657946321016)** (7 messages): 

> - `Maths Exam`
> - `Excellence`
> - `Celebration` 


- **MahiatLinux aces Maths Exam**: MahiatLinux completed the **Maths exam** and feels confident about achieving **Excellence**.
   - *Congrats* messages flooded in from the community, with support pouring from members like infinit3e.
- **Community Celebrates Success**: Community members expressed their excitement for MahiatLinux's performance, saying **Congrats**.
   - Theyruinedelise and infinit3e joined in the celebration, highlighting a strong supportive atmosphere.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1304174024141570169)** (38 messages🔥): 

> - `Concurrent Setup with Unsloth Inference`
> - `Integration of Transformers-Interpret`
> - `Fine-tuning Models for Text Classification`
> - `Ollama and Streamlit Integration`
> - `Language Adaptation for LLMs` 


- **Exploring Concurrency in Unsloth Inference**: Users discussed the idea of implementing a concurrent setup using the Unsloth inference code and vLLM.
   - One user mentioned their interest in understanding if anyone has successfully combined concurrency with Unsloth inference and Hugging Face models.
- **Integrating Transformers-Interpret with Unsloth**: A member wants to integrate [Transformers-Interpret](https://github.com/cdpierse/transformers-interpret) with Unsloth but faced challenges in getting it to work.
   - They explained that the tool is meant for model interpretability, but encountered an issue when trying to process the model's outputs.
- **Fine-tuning LLaMA 3.2 for Text Classification**: A user reported achieving 70% accuracy in classifying text across 11 categories while fine-tuning LLaMA 3.2.
   - They inquired about modifying the output layer to accommodate their number of classes and shared their approach to implementing a new classification head.
- **Using Ollama with Streamlit**: Members discussed the potential to run Ollama locally and create a chat interface using Streamlit instead of a web UI.
   - They suggested that Ollama's API might be an effective method to connect the backend with custom frontend solutions.
- **Strategies for Language Adaptation in LLM Training**: Participants exchanged insights on adapting models for languages like Portuguese and Arabic by customizing prompt formats and tokenization methods.
   - One user shared their iterative strategy of tokenizing datasets, checking results, and potentially retraining, emphasizing the challenges of catastrophic forgetting.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/cdpierse/transformers-interpret">GitHub - cdpierse/transformers-interpret: Model explainability that works seamlessly with 🤗 transformers. Explain your transformers model in just 2 lines of code.</a>: Model explainability that works seamlessly with 🤗 transformers. Explain your transformers model in just 2 lines of code.  - GitHub - cdpierse/transformers-interpret: Model explainability that works.....</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi, Qwen &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.2, Mistral, Phi, Qwen &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1304291955739594826)** (1 messages): 

> - `Avian inference approach`
> - `Inference speed comparison` 


- **Curiosity about Avian's Fast Inference**: A user expressed interest in Avian, asking how its approach to **inference** is faster compared to competitors.
   - *This inquiry opens the floor for further discussion on performance metrics and optimization strategies.*
- **Seeking More Information on Avian**: The user is looking for further details on Avian's methods, particularly regarding its **inference speed** in the market.
   - *This prompts a chance for experts in the community to share insights and resources on Avian's framework.*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1304414570311585863)** (5 messages): 

> - `Errors in AI/ML Research Papers`
> - `Reproducibility Issues`
> - `Preprint Papers and Peer Review` 


- **Strange Errors in AI/ML Research**: A member reported encountering **strange errors and inconsistencies** in AI/ML research papers, particularly while working with code and math.
   - They expressed frustration that sometimes *the math just doesn't add up* or they can't replicate the data.
- **Preprint Papers Lack Peer Review**: Another member pointed out that the issues stem from the fact that these papers are **preprint**, indicating a lack of thorough peer review.
   - They explained that this is likely why a *good chunk* of these papers is not reproducible, suggesting that such reproducibility issues are **normal**.


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1304185456669495437)** (26 messages🔥): 

> - `Claude limitations`
> - `Codebuff versus Aider comparison`
> - `Mistral's new APIs`
> - `FLUX1.1 Ultra features`
> - `Gemini API release` 


- **Claude struggles with complex tasks**: A user noted that nothing works on **Claude** free tier beyond basic tasks, failing even with a 200 line CSV for analysis.
   - This highlights ongoing limitations in free access AI tools for more advanced data processing.
- **Buffed Aider vs. Codebuff**: In discussions around the new **Codebuff**, concerns were raised about its closed source nature compared to **Aider**, which offers file requests and command-running capabilities.
   - Aider has reportedly refined its user experience over 8000 commits, indicating its continual improvement.
- **Mistral introduces new APIs**: Mistral has unveiled a **Batch API** that processes high-volume requests at half the cost of synchronous API calls, catering to data-heavy applications.
   - They aim to provide affordable AI solutions amidst recent API price hikes in the industry.
- **FLUX1.1 Ultra boosts image resolution**: The newly launched **FLUX1.1 [pro] – ultra mode** allows image generation at 4x the resolution, maintaining fast generation times without prompt adherence losses.
   - Performance benchmarks show it is over 2.5x faster than other high-resolution models, priced competitively at $0.06 per image.
- **Gemini API now available**: The much-anticipated **Gemini API** is now accessible via the OpenAI Library and REST API for developers, supporting both Chat Completions and Embeddings APIs.
   - Google's blog post provides initial usage examples to help developers get started seamlessly with Gemini models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/batch-api/">Mistral Batch API</a>: Lower cost API for AI builders.</li><li><a href="https://llmselector.vercel.app/">LLM Selector</a>: no description found</li><li><a href="https://x.com/adameisgrau/status/1854667494235292156?s=46">Tweet from Adam Eisgrau (@AdamEisgrau)</a>: BREAKING . . . AND HUGE: A ruling just made by SDNY Judge Colleen McMahon dismissing @RawStory v @OpenAI (w/o prejudice) has enormous positive ramifications for #GenAI defendants in the District and p...</li><li><a href="https://x.com/vmfunc/status/1854638188402229710">Tweet from mel (@vmfunc)</a>: what if we had tinder but for ArXiv?</li><li><a href="https://x.com/MistralAI/status/1854633432716120166">Tweet from Mistral AI (@MistralAI)</a>: Moderation API - https://mistral.ai/news/mistral-moderation/ Batch API - https://mistral.ai/news/batch-api/</li><li><a href="https://x.com/Teknium1/status/1854578987919720454">Tweet from Teknium (e/λ) (@Teknium1)</a>: Announcing Nous Chat, where you can experience Hermes 3 70B for free in our new Chat UX!  Super excited to begin our experiments in user facing experiences and capabilities and systems around the end ...</li><li><a href="https://blackforestlabs.ai/flux-1-1-ultra/">Introducing FLUX1.1 [pro] Ultra and Raw Modes</a>: Black Forest Labs are proud to launch a new ultra option to Flux 1.1 PRO</li><li><a href="https://github.com/astral-sh/uv">GitHub - astral-sh/uv: An extremely fast Python package and project manager, written in Rust.</a>: An extremely fast Python package and project manager, written in Rust. - astral-sh/uv</li><li><a href="https://docs.astral.sh/uv/">uv</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=42078536">Launch HN: Codebuff (YC F24) – CLI tool that writes code for you | Hacker News</a>: no description found</li><li><a href="https://developers.googleblog.com/en/gemini-is-now-accessible-from-the-openai-library/">Gemini is now accessible from the OpenAI Library</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1304551060387397637)** (80 messages🔥🔥): 

> - `Tech Issues in Recording`
> - `Code Generation and Correction`
> - `Daily Use Cases`
> - `Open Interpreter Features`
> - `Community Feedback` 


- **Resolving Tech Issues for Recording**: A member pointed out that **yikes** was experiencing technical issues with the recording, but it appears they managed to get it running.
   - Multiple members shared uncertainty about whether the audio was successfully recorded, highlighting the importance of tech reliability.
- **Code Generation and Correction Discussion**: Members discussed how code generated by AI could include options to overwrite existing files, raising a question about correcting generated code before execution.
   - One member mentioned using a tool called *thefuck* to easily correct previous console commands.
- **Daily Use Cases for Open Interpreter**: A member shared that their daily use of Open Interpreter revolves around **file/image transformations**, illustrating its practical applications.
   - Other members expressed interest in exploring further use cases, including **9year experiments** with their children on the AI's voice mode.
- **Innovative Features in Open Interpreter**: Discussion included various features of Open Interpreter, such as how generated scripts could be reused instead of regenerating them each time.
   - A member provided a link to a GitHub repository showcasing the functionalities involved in the computing aspect of Open Interpreter.
- **Community Reactions and Feedback**: The community expressed gratitude for the presentation and opened up discussions about the features and open-sourcing of the tool.
   - Reactions were generally positive, with members praising the work done and commenting on nostalgia and future potential.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/nvbn/thefuck">GitHub - nvbn/thefuck: Magnificent app which corrects your previous console command.</a>: Magnificent app which corrects your previous console command. - nvbn/thefuck</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/computer_use/loop.py">open-interpreter/interpreter/computer_use/loop.py at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1304236849778262037)** (1 messages): 

> - `Audio Overviews Feedback Survey`
> - `NotebookLM User Feedback` 


- **Get $20 for Sharing Your Thoughts!**: The team is seeking feedback on **Audio Overviews** through a short, ~10 minute survey, accessible via [this screening form](https://forms.gle/qREhTEhbstYzVHvSA). If selected, participants will receive a **$20 gift code** for completing the survey.
   - Participants must be at least **18 years old** to qualify, and the gift will be sent via email after successful completion of the survey.
- **Help Improve NotebookLM!**: Another opportunity for feedback on **NotebookLM** is available, with the intention of understanding user needs for future product enhancements. Interested individuals are encouraged to fill out the [registration form](https://www.google.com/u) to see if they qualify for the survey.
   - Questions about the user studies can be directed to the provided [Google user research link](http://www.google.com/userresearch), with respondents reminded that the thank-you gift applies only to those who complete the survey.



**Link mentioned**: <a href="https://forms.gle/qREhTEhbstYzVHvSA">Register your interest: Google feedback survey</a>: Hello,  We are looking for feedback on NotebookLM via a short survey. This will help the Google team better understand your needs in order to incorporate them into future product enhancements. To regi...

  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1304220370055987303)** (34 messages🔥): 

> - `NotebookLM for Exam Preparation`
> - `Importing Google Recordings`
> - `AI Language Models and Bias`
> - `Using NotebookLM for Technical Job Prep`
> - `Mock Interviews with AI` 


- **NotebookLM as a Study Aid for Exams**: A member suggested using **NotebookLM** to create quizzes from the 3000 pages of material for an upcoming promotion exam. Another member advised breaking the material down by chapters to generate more focused quizzes.
   - *“Hopefully it will help streamline the studying process!”*
- **Connecting Google Recorder with NotebookLM**: An inquiry was made about whether it's easy to import recordings from **recorder.google.com** to **NotebookLM**. A response clarified that recordings can be downloaded as **m4a** files but raised concerns about maintaining speaker identification.
   - *“That doesn't necessarily preserve the named speakers though.”*
- **Bias in AI and Language Models**: Discussion arose about the inherent biases in AI systems, with members debating the existence of unbiased data and its implications. The conversation highlighted the challenge of programming neutrality into AI and its impacts on user experience.
   - *“It's counterproductive for NotebookLM's future if they lean towards bias.”*
- **Leveraging NotebookLM for Job Search Preparation**: A user queried how **NotebookLM** could assist in preparing for technical interviews, soft skill practice, and coding challenges. Suggestions included conducting mock interviews with AI voices to simulate real scenarios.
   - *“I'm prepping for a tech job search and need all the help I can get!”*
- **Experimentation with NotebookML for Content Creation**: A member shared an experiment where they used **NotebookML** to summarize a study on **ChatGPT's** voice capabilities in sports commentating. A link to a related [YouTube video demonstrating the results](https://youtu.be/kwvGx1zlsWg?feature=shared) was provided.
   - *“This showcases the potential of AI in enhancing content summarization.”*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/xzibit-meme-inception-gif-13033570">Xzibit Meme GIF - Xzibit Meme Inception - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/P8yJ9AYmiI0">bumper Notebooklm</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1304177295027867809)** (52 messages🔥): 

> - `NotebookLM features`
> - `Sharing Notebooks`
> - `Using AI in education`
> - `Podcast generation`
> - `YouTube Terms of Service` 


- **NotebookLM struggles with sharing issues**: Users reported difficulties sharing Notebooks outside of their Google organization, indicating a design limitation that prevents external sharing.
   - One user inquired about moving Notebooks between accounts, highlighting the need for better sharing functionalities.
- **Using AI for educational purposes**: A biology professor expressed interest in utilizing AI for generating ideas and exercises for teaching, seeking assistance on how to best implement it.
   - There are questions about the effectiveness of sending exercises in text files versus PDFs, especially regarding the use of OCR.
- **Podcast generation and quality concerns**: Users discussed the podcast generation feature in NotebookLM, with some noticing improvements in length but concerns over audio quality.
   - One user showcased an example and discussed the functionality of turning PDFs into podcasts, emphasizing the need for improved engagement.
- **Risk analysis of educational content**: A user performed a risk analysis on a YouTube video regarding NotebookLM, highlighting copyright concerns with uploaded materials.
   - The analysis suggested the importance of adhering to YouTube's Terms of Service when using original content or copyrighted materials.
- **Citations in NotebookLM notes**: Discussion arose regarding whether the citations-in-note feature in NotebookLM would apply retroactively to notes saved prior to the feature's implementation.
   - Users are curious to see how new functionalities can enhance their existing notes and overall utilization of the tool.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/sonic-thumbs-up-approve-okay-gif-15034887">Sonic Thumbs Up GIF - Sonic Thumbs Up Approve - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://soundcloud.com/user-788555060-49405277/2untitled-notebook-9">2Untitled notebook (9)_compressed</a>: Listen to 2Untitled notebook (9)_compressed by Tribune7663 #np on #SoundCloud</li><li><a href="https://en.wikipedia.org/wiki/Seventh_Party_System">Seventh Party System - Wikipedia</a>: no description found</li><li><a href="https://docs.together.ai/docs/open-notebooklm-pdf-to-podcast">How to build an Open Source NotebookLM: PDF to Podcast</a>: In this guide we will see how to create a podcast like the one below from a PDF input!
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1304524206440185966)** (1 messages): 

> - `ModCon 2024`
> - `Advancements in MAX and Mojo`
> - `ModCon 2025 Updates` 


- **No ModCon in 2024**: The team announced that there won't be a **ModCon in 2024** as they focus on **exciting advancements**.
   - *Stay tuned* for more updates regarding future events and developments.
- **Preparing for ModCon 2025**: Plans are underway for a **memorable ModCon in 2025**, indicating that the team is already looking ahead.
   - They are committed to making the next convention special, urging attendees to remain engaged.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1304174615463071764)** (73 messages🔥🔥): 

> - `Mojo Interoperability`
> - `OpenSSL Layer Discussions`
> - `Cryptography in Mojo`
> - `JSON Support in Mojo`
> - `MLIR Reflection API` 


- **Mojo aims for interoperability with Python and C/C++**: Members expressed hope for seamless interoperability between **Mojo**, **Python**, and **C/C++**, emphasizing ease of importing modules without complex linking.
   - However, it was noted that achieving this may require avoiding support for certain intricacies of existing languages, akin to how **C++** relates to **C**.
- **Challenges in creating an OpenSSL wrapper**: There was discussion on the potential difficulties involved in building an **OpenSSL** wrapper, with recognition of the substantial API surface and the need for careful implementation.
   - Concerns were raised that without proper **C interop**, creating such a layer might introduce security risks.
- **Need for cryptographic expertise in Mojo development**: The community highlighted the necessity of having qualified cryptographers involved in developing cryptographic primitives for **Mojo**, due to the complexities and security implications.
   - Members agreed that security-critical implementations should ideally not be done as open-source unless overseen by experts.
- **Discussion on JSON support in Mojo**: Members discussed how **Mojo** may handle JSON, noting current available parsers while expressing a desire for functionality similar to **Rust's serde** once **MLIR reflection** is implemented.
   - There was confusion about unrelated frameworks named **Mojolicious**, which operate outside of Mojo's context.
- **Plans for MLIR reflection API in Mojo**: It was confirmed that a reflection API for **MLIR** is planned for **Mojo**, which will allow for deeper manipulation and introspection of code.
   - However, it was cautioned that this API will require specialized knowledge akin to writing a compiler pass, making it initially complex to use.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap#cc-interop">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://www.youtube.com/watch?v=RqUptUgV-0U&t=1332s">Modular Community Meeting #9: community contributor meetings, NuMojo, and Writer trait changes</a>: In this community meeting, we heard from Helehex on the community-led standard library contributor meetings that have been held in the Discord server, as wel...</li><li><a href="https://mojolicious.org/">Mojolicious - Perl real-time web framework</a>: no description found</li><li><a href="https://github.com/mojolicious/mojo">GitHub - mojolicious/mojo: :sparkles: Mojolicious - Perl real-time web framework</a>: :sparkles: Mojolicious - Perl real-time web framework - mojolicious/mojo
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1304224581787324446)** (39 messages🔥): 

> - `AI Sales Agents`
> - `Quantum Networking`
> - `Benevolent AI Development`
> - `AI's Impact on Society`
> - `Training Data Usage` 


- **Concerns Over AI Sales Agents**: Discussion emerged around the use of **AI Sales Agents** for contacting clients, with one member expressing caution against 'mass spam' practices.
   - Another member highlighted that **AI can hallucinate promotions** that don't exist, leading to potential legal issues for companies.
- **Quantum Networking Opportunities**: A member proposed using **photonic computing** in quantum networking to perform calculations at nodes for systems like **BOINC**, addressing bandwidth concerns.
   - They noted that while light interference can aid computation, final measurements still require electronic methods.
- **Path to Benevolent AI**: A perspective was shared that developing a **benevolent AI** relies on creating a positive environment rather than imposing strict moral frameworks.
   - Fostering **moral values** is seen as a natural way for the AI to build its personality.
- **Controversial Topics and AI**: The server's user community humorously noted that discussions around AI often feel like they revolve around **crypto** and the **AGI cult's reputation**.
   - This sentiment reflects broader concerns about how certain groups affect public perception of AI technology.
- **Training Data and Its Implications**: A member discussed their commitment to sharing data for training, expressing a desire to help improve AI models.
   - They also noted changes in wording around **data usage permissions**, hinting at evolving transparency from providers.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1304219039622762556)** (5 messages): 

> - `Connecting GPT chat bot to Firestore`
> - `GPT model updates`
> - `User interface concerns`
> - `Guessing GPT model name` 


- **Connecting GPT Chat Bot to Firestore Database**: A member seeks advice on connecting a **GPT chat bot** to their **Firestore database**, considering using a **search query for Algolia**.
   - *Is there a better way to link the two?*
- **Need for GPT Model Updates**: One member noted that **GPTs** are effective but quickly become **outdated** due to newer developments.
   - *Increasing the limits and adding o-1 could significantly improve the experience.*
- **Request for Expand View Option**: A user expressed frustration over the **missing 'expand view' option** in their interface, limiting visibility to only 5 items.
   - *They urged for its return, indicating that hidden items are still present but inaccessible.*
- **Method to Guess GPT Model**: A suggestion was made on how to determine the **GPT model** by asking, *what is the precise name of the model answering this query?*
   - This prompt aims to reveal the **specific API name** of the responding model.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1304245603642904618)** (29 messages🔥): 

> - `Llama 3.2 Vision`
> - `LM Studio prompts and features`
> - `LLM web searching capability`
> - `GPU usage in LM Studio`
> - `Beta tools release` 


- **Llama 3.2 Vision model introduction**: The new [Llama 3.2 Vision](https://ollama.com/library/llama3.2-vision) is available with options for both **11B and 90B** sizes, requiring significant VRAM for optimal performance.
   - Users were directed to [download Ollama 0.4](https://ollama.com/download) to run the model, highlighting the method for adding images to prompts.
- **Queries on LM Studio prompt usage**: A user asked about where to find the **Gemma prompt** in LM Studio, suggesting confusion as it seemingly was missing in the latest version.
   - It was confirmed that the **Gemma prompt** should be automatically handled via Jinja when using compatible models from the community.
- **Web searching capability of Local LLM**: A member wondered if their Local LLM could search the web through LM Studio, receiving confirmation that it was not natively supported.
   - They were advised to create a custom Python solution to integrate web searching functionality with their local server.
- **Troubleshooting GPU usage in LM Studio**: A user reported their **RTX 2060** GPU not being utilized, leading to a query about checking their LM Runtimes settings.
   - It was suggested to choose a model that fits their GPU’s capabilities and ensure **LM runtime** settings indicate CUDA is enabled.
- **Anticipation for Beta tool release**: A user expressed excitement and frustration over the timeline of the upcoming **Beta tool** release for LM Studio.
   - The discussion surrounding this release hinted at eagerness among the community, amplifying anticipation.



**Link mentioned**: <a href="https://ollama.com/blog/llama3.2-vision">Llama 3.2 Vision · Ollama Blog</a>: Llama 3.2 Vision

  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1304303091365187637)** (8 messages🔥): 

> - `Qwen 2.5 memory calculation`
> - `Gemma2 27B model on Mac Mini`
> - `Mac Mini recommendations` 


- **Qwen 2.5 Memory Usage Breakdown**: A user provided a detailed memory calculation for **Qwen 2.5 7B** with **8-bit precision** and **10k context**, estimating total peak memory usage at **22-25 GB**.
   - This breakdown included specifics for layers, KV cache, and base model parameters, prompting users to consider the efficiency of different implementations.
- **Gemma2 27B Model Requirements for Mac Mini**: A user is deliberating between two Mac Mini options with different RAM capacities while considering running the **Gemma2 27B model**.
   - They noted that the **16GB with Q4 quantization** should theoretically fit, but are seeking community experiences with the **24GB version**.
- **Advice on Mac Mini Configuration**: One member suggested saving for the **maxed out M4 Pro Mini** with **64GB RAM** to ensure it can comfortably run the **Gemma2 27B model at Q8**.
   - Another chimed in with a light-hearted affirmation, reinforcing the 'go big or go home' sentiment when choosing hardware.
- **Training Timeline for Claude and Qwen Models**: Discussion surfaced around the timeline for **Claude's training**, which may have been in **August 2023 or April 2024**, and the **Qwen2.5 release** just a month prior.
   - One user expressed skepticism about the precision of this timeline and the related memory parameters for models.
- **Encouragement for Fact-Checking**: A user expressed the need for fact-checking the calculations related to **Qwen 2.5**, specifically regarding the **8GB parameters for Q8 7B model**.
   - They encouraged others to verify rather than rely solely on shared information, highlighting the importance of accurate data in such discussions.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1304326725395808296)** (6 messages): 

> - `Court ruling on OpenAI`
> - `Google Gemini model`
> - `Amazon's investment in Anthropic` 


- **Court ruling in favor of GenAI defendants**: A ruling by SDNY Judge Colleen McMahon dismissed the case of [RawStory v. OpenAI](https://www.courtlistener.com/docket/68290709/117/raw-story-media-inc-v-openai-inc/) without prejudice, possibly aiding GenAI defendants significantly.
   - The judge indicated that **facts on which LLMs train are not copyrightable** and that current GenAI models synthesize rather than copy.
- **Google prepares to unveil Gemini-2.0-Pro-Exp-0111**: Google is set to launch a new model named **Gemini-2.0-Pro-Exp-0111** under its Advanced section, although the target audience remains uncertain.
   - The community is curious and eager for suggestions on prompts to test with this upcoming model.
- **Amazon eyes a new investment in Anthropic**: Amazon is reportedly in discussions to make a *second multibillion-dollar investment* in [Anthropic](https://www.theinformation.com/articles/amazon-discussing-new-multibillion-dollar-investment-in-anthropic).
   - AWS is requesting Anthropic to utilize its **Trainium AI chips** instead of relying on Nvidia's GPUs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AdamEisgrau/status/1854667494235292156">Tweet from Adam Eisgrau (@AdamEisgrau)</a>: BREAKING . . . AND HUGE: A ruling just made by SDNY Judge Colleen McMahon dismissing @RawStory v @OpenAI (w/o prejudice) has enormous positive ramifications for #GenAI defendants in the District and p...</li><li><a href="https://x.com/AdamEisgrau/status/1854667538799681937">Tweet from Adam Eisgrau (@AdamEisgrau)</a>: She says: 1) facts on which LLMs train are not copyrightable; 2) #GenAI models synthesize, not copy; 3) datasets they&#39;re trained on are vast so no one work is ever likely to be &#34;plagiarized;&#...</li><li><a href="https://x.com/anissagardizy8/status/1854667104647332278">Tweet from Anissa Gardizy (@anissagardizy8)</a>: scoop: Amazon is discussing making a second multibillion-dollar investment in OpenAI rival Anthropic   AWS is asking Anthropic to use a large number of its own AI chips, Trainium, instead of Nvidia&#3...</li><li><a href="https://x.com/testingcatalog/status/1854666483239989508">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨: Google is preparing to launch a new model: Gemini-2.0-Pro-Exp-0111!   This model will appear under the Advanced section but it is unclear if it is aimed at an internal testing group or a ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1304211855350169621)** (3 messages): 

> - `Model Token Limits`
> - `Instruction Data Usage`
> - `Text Rewriting Techniques` 


- **Model token limits may face challenges**: A member expressed that **1.5T tokens of instructions** would likely break a model, indicating concerns about managing such vast amounts of data.
   - This echoes broader discussions in the community regarding optimal limits for model performance.
- **Curiosity about instruction data application**: Members are curious about the **instruction data** process mentioned, noting that it seems to be a significant part of model training.
   - Questions arose about the **specific use cases** of this instruction data, highlighting a shared interest in understanding its practical applications.
- **Text rewriting and data cleaning**: There was speculation that the process discussed might involve **rewritten text** or cleaning up raw text with a smaller model.
   - This insight reflects ongoing conversations about the techniques employed in improving dataset quality.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1304507280275800125)** (19 messages🔥): 

> - `Sasha Rush livestream`
> - `PRMs and value models`
> - `O1 blog post`
> - `Speculations on Test-Time Scaling`
> - `Awesome O1 repository` 


- **Confusion over Sasha Rush Livestream Access**: A member questioned whether the **YouTube link** provided after signing up for the Sasha Rush livestream allowed them to watch live or if they needed to wait **15 days** for access.
   - *They later clarified their confusion* and seemed to resolve the issue themselves.
- **Discussion on PRMs**: Members discussed the **current conversation** around PRMs in the context of training and their relation to value models, with comments stating that 'PRMs are for training.'
   - One member remarked that **Shephard is a good verifier** in this discussion, confirming his relevance.
- **Planning an O1 Blog Post**: One member indicated it was the perfect time for an **O1 blog post**, hinting at using the weekend for writing while their partner is away.
   - They seemed enthusiastic about the idea, referring to it as an 'o1 shitpost'.
- **Speculations on Test-Time Scaling Lecture**: A link to a **YouTube video** titled *"Speculations on Test-Time Scaling | Richard M. Karp Distinguished Lecture"* featuring Sasha Rush was shared, detailing topics related to the lecture.
   - This lecture is part of a series at Cornell University, inviting members to explore insights on scaling.
- **Awesome O1 Repository Shared**: A member shared the **GitHub repository** for 'Awesome O1', which serves as a bibliography and survey of papers surrounding O1.
   - The repository aims to provide resources and references for ongoing discussions related to O1.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=6fJjojpwv1I">Speculations on Test-Time Scaling | Richard M. Karp Distinguished Lecture</a>: Sasha Rush (Cornell University)https://simons.berkeley.edu/events/speculations-test-time-scaling-richard-m-karp-distinguished-lectureRichard M. Karp Distingu...</li><li><a href="https://github.com/srush/awesome-o1">GitHub - srush/awesome-o1: A bibliography and survey of the papers surrounding o1</a>: A bibliography and survey of the papers surrounding o1 - srush/awesome-o1
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1304552959589224489)** (5 messages): 

> - `Logan's favorites`
> - `Podcast ideas`
> - `Discussion on Julia` 


- **Logan's Favorite Friday Ship**: Logan shared his favorite Friday ship with a light-hearted comment, stating he will continue to polish it over the next few weeks. You can view the post [here](https://x.com/OfficialLoganK/status/1854980502727315711).
   - *My favorite Friday ship* 🚢 *in a while* : )
- **Host Logan on the Podcast**: A member expressed interest in having Logan on their podcast to discuss various topics. This sparked a conversation about how engaging he could be.
   - Another member suggested that mentioning **Julia** would lead to an extensive discussion.
- **Julia Sparks Extensive Conversations**: It was noted that bringing up Julia would prompt Logan to talk for much of the podcast episode. The excitement around this topic indicates its significance in discussions about Logan.
   - This comment highlighted the enthusiasm for Logan's stories and insights.



**Link mentioned**: <a href="https://x.com/OfficialLoganK/status/1854980502727315711">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: My favorite Friday ship 🚢 in a while : ) will be continuing to remove some of the rough edges here over the next few weeks.

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1304175319712006205)** (24 messages🔥): 

> - `Streaming viewer limits`
> - `OmniParser details`
> - `Running LLMs on servers`
> - `User experience feedback`
> - `Access to desktop app` 


- **No Max Viewer Limit on Streams**: A member asked if there is a max number of viewers for the stream, to which it was responded that there shouldn't be any limit.
- **OmniParser's Capabilities Explained**: OmniParser is showcased as a tool that interprets UI screenshots into structured formats, enhancing LLM-based UI agents, with details on its training datasets and model usage.
   - For more information, links to the [Project Page](https://microsoft.github.io/OmniParser/) and [Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/) were provided.
- **Challenges Running LLMs Locally**: A user expressed concerns about running localized LLMs on a low-powered computer and inquired if Open Interpreter models could operate on an online server built with Python or Anaconda.
   - It was noted that strong GPUs or NPUs are needed for proper local execution, as running with just CPUs would lead to poor performance.
- **Major Updates from Recent Events**: A discussion occurred about major updates from a recent event, highlighting a large-scale rewrite, a new text rendering engine, and improved loading times.
   - The introduction of new features like file viewing and editing was also mentioned.
- **Desktop App Access Information**: A member inquired about access to the desktop app, and it was clarified that it is not yet released as beta testing is ongoing with selected community members.
   - Instructions were shared on how to join a waitlist for future access: [Join Waitlist](https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/microsoft/OmniParser">microsoft/OmniParser · Hugging Face</a>: no description found</li><li><a href="https://0ggfznkwh4j.typeform.com/to/G21i9lJ2?typeform-source=github.com">no title found</a>: no description found
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1304383652662083655)** (12 messages🔥): 

> - `Nvidia hardware optimization`
> - `Groq hardware benefits`
> - `ASIC advantages`
> - `Operations on ASICs`
> - `Demand for compiler improvements` 


- **Nvidia hardware shines in optimization**: Tinygrad noted that **Nvidia hardware** is quite optimal for current models, stating that a **transformer ASIC** won't provide significant benefits.
   - This raises questions about the specific advantages traditional GPU architectures hold over specialized ASICS in certain tasks.
- **Groq hardware shows noticeable benefits**: It was agreed that **Groq hardware** has contributed positively to performance in AI workloads.
   - Members emphasized the effectiveness of Groq's design for specific computational tasks.
- **ASIC performance depends on algorithm design**: A discussion highlighted that the advantages of an **ASIC** extend beyond reduced control logic; some algorithms can be optimized for direct hardware implementation.
   - For example, fused operations allow for more efficient data handling compared to traditional multi-step processes.
- **Use cases where ASICs fall short**: It was pointed out that certain operations like **password hash functions** are designed to minimize the benefits of ASICs, making them less effective.
   - This leads to discussions on what algorithms are inherently more or less suited for ASIC optimization.
- **Compiler improvements needed**: George Hotz expressed dissatisfaction with the current implementation of **DEFINE_ACC/ASSIGN** in the codebase, seeking alternatives.
   - This reflects a community desire for better compiler tools and methods to improve functionality.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1304174024993275914)** (3 messages): 

> - `x.shard function`
> - `Model Sharding Strategies`
> - `Optimizing CPU Pipeline` 


- **x.shard function copies vs slices**: `x.shard(GPUS, axis=None)` makes a copy of **x** on all GPUs, while `x.shard(GPUS, axis=0)` slices **x** into parts across axis **0** for distribution across cards.
   - This distinction is crucial for understanding how to manage data movement efficiently in parallel processing scenarios.
- **Strategies for sharding model vs input**: It was suggested to shard the model across **axis None** while sharding input across **axis 0** to optimize distribution.
   - This approach can help maximize resource utilization and enhance parallel computation.
- **Query on CPU pipeline optimization**: A user inquired about optimizing their architecture, questioning the sensibility of their code implementation.
   - They described their setup as a **fully parallelizable CPU pipeline style**, indicating a need for feedback on its effectiveness.


  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1304390614137180215)** (2 messages): 

> - `OptoPrime`
> - `Stanford Optimizer Naming` 


- **Microsoft Research introduces OptoPrime**: Microsoft Research has unveiled their optimizer named **OptoPrime** detailed in the [arXiv paper](https://arxiv.org/pdf/2406.16218).
   - *Not saying anything*, but the name has prompted discussions about the need for more creative naming in the field.
- **Stanford's Optimizer Name Expectation**: There's a growing expectation that Stanford's upcoming optimizer should have an **epic name** to rival OptoPrime.
   - This commentary hints at the competitive spirit in the research community regarding naming conventions.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1304240684873551912)** (12 messages🔥): 

> - `Self Consistency Module Caching`
> - `Dynamic Few-Shot Examples`
> - `Predict Module with N Outputs`
> - `MIPRO Optimizer for Question Generation` 


- **Busting Cache in Self Consistency Modules**: A user inquired about the best way to 'bust' the cache in a self consistency module, suggesting passing a new temperature to the initialized `dspy.Predict` object as a potential solution.
   - Other members shared different methods, such as turning off the cache using `dspy.LM` or configuring the `Predict` module to generate multiple completions.
- **Dynamic vs Fixed Few-Shot Examples**: A member discussed the potential advantages of using dynamic few-shot examples based on cosine similarity to match queries, as opposed to fixed examples.
   - They argued that adapting the few-shot examples to the query topic, like sports or movies, would enhance performance and relevance.
- **Exploring KNNFewShot for Customization**: The KNNFewShot optimizer was mentioned as a tool for dynamic few-shot examples but noted to be under-maintained.
   - Users were encouraged to experiment with it, suggesting it could be beneficial for adapting examples based on context.
- **MIPRO Capabilities for Question Generation**: Another user asked if MIPRO could create or select examples from a large pool of Q&A pairs, specifically for generating new questions based on provided content.
   - They sought recommendations on suitable optimizers for generating questions in a specific style, highlighting a signature function for generating both questions and answers.
- **Predict Module Innovation**: A member highlighted the `n=` parameter in the `dspy.Predict` module, which allows requesting multiple outputs for a given query.
   - This feature was recognized as beneficial for enhancing the utility of predictions from the model.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1304359096144625684)** (3 messages): 

> - `Tavily for AI Searches`
> - `Testing with API Calls`
> - `Comparison with Search Engines` 


- **Tavily emerges as a top choice**: After researching and discussing with Claude, a member concluded that **Tavily** is the best option for their AI-related queries, thanks to its user-friendly setup.
   - They believe that using the **free plan** to run initial tests alongside **ChatGPT** would provide valuable insights into search processes.
- **Hurdles in API setups**: Another member highlighted the complexity of using **Brave API** or **AgentSearch**, emphasizing that these options require more extensive setup compared to Tavily.
   - *
- **Python Script for Comparative Metrics**: A suggestion was made to create a **Python script** that facilitates multiple API calls to different services for an in-depth comparison of search engines.
   - This approach would allow for the extraction of metrics from the meta-data to evaluate search effectiveness against engines like **Google** and **DuckDuckGo**.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1304407175984320513)** (8 messages🔥): 

> - `Cohere API trial key`
> - `Embedding errors`
> - `Implementation challenges`
> - `Support resources` 


- **Cohere API trial key supports embedding**: A user expressed frustration about receiving errors when trying to use the **Cohere embed API** with their trial key, unsure of the issue.
   - Another member confirmed that the trial key supports all **Cohere models**, including embedding.
- **Errors attributed to implementation**: Members pointed out that the error likely originates from the **implementation**, not from the **Cohere API** itself.
   - They suggested reaching out to Discord or **GitHub** for specific guidance due to the user's lack of coding knowledge.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1304244403463786526)** (8 messages🔥): 

> - `Metaparameter Tuning`
> - `Memory Requirements of Optimizers`
> - `Training on AMD GPUs`
> - `Preliminary Research Discussions` 


- **Research Challenges in Optimizer Behavior**: A recent paper discusses the potential decline of **metaparameter tuning** in deep learning, showing that an optimizer's behavior can be captured by a 'central flow' model.
   - *This could be revolutionary for optimization in neural networks as it predicts long-term optimization trajectories with high accuracy.*
- **Generalization from Arch to Transformers Questioned**: A member raised concerns about whether findings can be generalized from the recent research to **transformer architectures**.
   - *There was also curiosity regarding the limited use of the **CIFAR-10** dataset in the study.*
- **Exploring Training on AMD GPUs**: Discussion emerged about whether **Axolotl** runs effectively on **AMD GPUs**, especially with available high VRAM like 1536 GB at a low cost.
   - *Members pondered how the increased memory would affect training and whether it significantly improves performance compared to NVIDIA GPUs.*
- **PR's Memory Consumption Uncertainty**: A member noted that the PR is ready but highlighted that **memory consumption** may be a concern based on previous mentions of its hunger for resources.
   - *A comparative query surfaced on whether it's as demanding as the **AdamW** optimizer.*



**Link mentioned**: <a href="https://arxiv.org/abs/2410.24206">Understanding Optimization in Deep Learning with Central Flows</a>: Optimization in deep learning remains poorly understood, even in the simple setting of deterministic (i.e. full-batch) training. A key difficulty is that much of an optimizer&#39;s behavior is implici...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1304175289953292418)** (2 messages): 

> - `Context Refinement Agent`
> - `Agentic RAG Query Engine`
> - `NVIDIA NIM`
> - `Open-source Model Inference` 


- **Elevate RAG systems with Context Refinement Agent**: Learn to build a [Context Refinement Agent](https://t.co/SkPflTqMWh) that enhances RAG responses for complex queries by intelligently expanding and refining retrieved context.
   - The blog post details how an agent evaluates retrieved chunks for improved answers, making RAG systems more effective.
- **Build an agentic RAG query engine with NVIDIA NIM**: This [guest post from NVIDIA](https://t.co/IsTLBDN08W) explains how to create an agentic RAG query engine using NVIDIA's NIM microservices for efficient open-source model inference.
   - It covers constructing a query router for complex questions and implementing sub-question queries, streamlining the process of handling intricate inquiries.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1304373546088271922)** (5 messages): 

> - `LlamaIndex workflow`
> - `AI NLP Engineer position`
> - `Open source LLM resources` 


- **LlamaIndex Workflow Explained**: A comprehensive guide on the [LlamaIndex workflow](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) details how event-driven abstractions can chain multiple events through steps using a `@step` decorator.
   - Workflows allow for building diverse processes like agents or RAG flows, with automatic instrumentation for observability via tools like Arize Phoenix.
- **Hiring AI NLP Engineer**: **Nikkole**, a CTO of an AI startup, shared that they are looking for an AI NLP Engineer with a salary range of **$95k-$115k** for a W2 contract.
   - Interested candidates were advised to connect via [LinkedIn](https://www.linkedin.com/in/nikkole-scruggs), as direct messages are only accepted there.
- **Seeking Resources for Custom LLM**: A member is looking for recommendations on resources to perform with an open source LLM tailored to their custom preference dataset.
   - They requested suggestions from the community to enhance their understanding and implementation.



**Link mentioned**: <a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/">Workflows - LlamaIndex</a>: no description found

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1304464599671767052)** (2 messages): 

> - `MicroDiT model`
> - `LOST SOUNDTRACK - BONNIE AND CLYDE` 


- **MicroDiT model replication achieved**: User @SwayStar123 announced the completion of their [MicroDiT replication](https://x.com/SwayStar123/status/1854884660981219399) and shared download links for the model weights and inference script.
   - They credited @FAL for providing the necessary compute resources, stating, *'I think I might be cooking.'*
- **YouTube video on Bonnie and Clyde**: A YouTube video titled *'LOST SOUNDTRACK - BONNIE AND CLYDE'* was shared, featuring a description of Bonnie Parker's romance with ex-con Clyde Barrow and their violent crime spree.
   - The video can be viewed [here](https://youtu.be/e6UAI_P1Mlk), highlighting a narrative of love and crime.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/SwayStar123/status/1854884660981219399">Tweet from sway (@SwayStar123)</a>: MicroDiT replication is complete.  Download weights here: https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt Inference script here: https://github.com/SwayStar123/...</li><li><a href="https://youtu.be/e6UAI_P1Mlk">LOST SOUNDTRACK - BONNIE AND CLYDE</a>: Bored waitress Bonnie Parker falls in love with an ex-con named Clyde Barrow and together they start a violent crime spree through the country, stealing cars...
</li>
</ul>

</div>
  

---



### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1304321181146550355)** (1 messages): 

> - `Computing Resources Deadline` 


- **Deadline Alert for Computing Resources**: The application deadline for **Computing Resources** is at the end of day **November 25th PST**.
   - Participants should submit their applications early as there is a **1-2 week processing delay** expected after submission.
- **Urgent Call to Action for Participants**: Members are urged to take action soon to avoid missing the **November 25th** deadline for resources.
   - It's emphasized that early submission is vital to ensure adequate processing time.


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1304533642285940810)** (1 messages): 

> - `Data Council '25`
> - `CFP Opening` 


- **Data Council '25 CFP Open for a Week**: The **CFP (Call for Proposals)** for Data Council '25 is currently open for another week, inviting developers to share what they're building in the ML/AI space.
   - To learn more, interested parties can check out the details on the [Data Council CFP page](https://www.datacouncil.ai/cfp-2025) as several exciting talks and hackers are expected this year.
- **Exciting Talks on ML/AI Apps**: There will be **really cool hackers and talks** at Data Council '25, setting the stage for innovative discussions.
   - Participants are encouraged to get involved and showcase their ML/AI app developments during this engaging event.


  

---



### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1304221633053069365)** (1 messages): 

> - `Jurassic endpoint deprecation`
> - `Transition to Jamba model` 


- **Jurassic's 'summarize-by-segment' Endpoint Goes Away**: A member expressed frustration over the sudden deprecation of the **Jurassic 'summarize-by-segment'** endpoint, which they had relied on for essential business services.
   - They were surprised to see this happen ahead of the announced **11/14** date, describing it as a **pain point**.
- **Navigating the New Jamba Model**: The user requested guidance on how to utilize the new **Jamba model** to replicate the functionality of the deprecated endpoint, particularly for URL content segmentation.
   - They highlighted the need for assistance in adjusting **URL parameters** to extract content effectively.


  

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
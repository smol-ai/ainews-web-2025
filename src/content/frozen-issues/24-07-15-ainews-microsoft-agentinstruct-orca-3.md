---
id: 2bb5808c-a173-4e92-8a75-1e22651b1692
title: Microsoft AgentInstruct + Orca 3
date: '2024-07-16T00:42:03.637767Z'
original_slug: ainews-microsoft-agentinstruct-orca-3
description: >-
  **Microsoft Research** released **AgentInstruct**, the third paper in its
  **Orca** series, introducing a generative teaching pipeline that produces
  **25.8 million** synthetic instructions to fine-tune **mistral-7b**, achieving
  significant performance gains: +40% AGIEval, +19% MMLU, +54% GSM8K, +38% BBH,
  +45% AlpacaEval, and a 31.34% reduction in hallucinations. This synthetic data
  approach follows the success of **FineWeb** and **Apple's Rephrasing
  research** in improving dataset quality. Additionally, **Tencent** claims to
  have generated **1 billion** diverse personas for synthetic data. On AI
  Twitter, notable discussions included a shooting incident at a Trump rally and
  recent ML research highlights such as **FlashAttention-3**, **RankRAG**, and
  **Mixture of A Million Experts**.
companies:
  - microsoft-research
  - apple
  - tencent
  - hugging-face
models:
  - mistral-7b
  - orca-2.5
topics:
  - synthetic-data
  - fine-tuning
  - instruction-following
  - transformers
  - model-performance
  - hallucination-detection
  - dataset-quality
  - flashattention
  - mixture-of-experts
people:
  - philschmid
  - sama
  - bindureddy
  - rohanpaul_ai
  - zachtratar
  - dair_ai
---


<!-- buttondown-editor-mode: plaintext -->**Generative Teaching is all you need.**

> AI News for 7/12/2024-7/15/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**465** channels, and **4913** messages) for you. 
Estimated reading time saved (at 200wpm): **505 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

The runaway success of FineWeb this year ([our coverage here](https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/), [tech report here](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)) combined with [Apple's Rephrasing research](https://x.com/pratyushmaini/status/1752337225097076809) has basically served as existence proofs that there can be at least an order of magnitude improvement in dataset quality for pre- and post-training. With content shops either lawyering up or partnering up, research has turned to improving synthetic dataset generation to extend the runway on the tokens we have already compressed or scraped.

Microsoft Research has made the latest splash with [**AgentInstruct:
Toward Generative Teaching with Agentic Flows**](https://x.com/_philschmid/status/1811308080166035549), (not to be confused with [AgentInstruct of Crispino et al 2023](https://arxiv.org/abs/2310.03710)) the third in its Orca series of papers: 

- [Orca 1: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707)
- [Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045)
- [Orca Math: Unlocking the potential of SLMs in Grade School Math](https://arxiv.org/abs/2402.14830))

The core concept is that raw documents is transformed by multiple agents playing different roles to provide diversity (for 17 listed capabilities), which are then used by yet more agents to generate and refine instructions in a "Content Transformation Flow".

 ![image.png](https://assets.buttondown.email/images/8ab271c3-ee32-45f9-9504-7ebc2b8a3e51.png?w=960&fit=max) 

Out of this pipeline comes 22 million instructions aimed at teaching those 17 skills, which when combined with the 3.8m instructions from prior Orca papers makes "Orca 2.5" - the 25.8m instruction synthetic dataset that the authors use to finetune Mistral 7b to produce the results they report:

- +40% on AGIEval, +19% on MMLU; +54% on GSM8K; +38% on BBH; +45% AlpacaEval, 31.34% reduction in hallucinations for summarization tasks (thanks [Philipp](https://x.com/_philschmid/status/1811308080166035549))

This is just the latest entry in this genre of synthetic data research, most recently with [Tencent claiming 1 billion diverse personas](https://x.com/arankomatsuzaki/status/1807593343007818065) on their related work.

 ![image.png](https://assets.buttondown.email/images/cbcd3d24-0df0-4a75-8b85-aee75cb17530.png?w=960&fit=max) 

This seems both obvious that it will work yet also terribly expensive and inefficient compared to FineWeb, but whatever works!


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

**Shooting Incident at Trump Rally**

- **Shooting details**: [@sama](https://twitter.com/sama/status/1812566313577128153) noted a gunman at a Trump rally pointed a rifle at an officer who discovered him on a rooftop shortly before opening fire, with the bullet coming within an inch of Trump's head. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1812569682764833026) shared an AP update confirming the gunman pointed the rifle at the officer before opening fire.
- **Reactions and commentary**: [@sama](https://twitter.com/sama/status/1812566313577128153) hoped this moment could lead to **turning down rhetoric and finding more unity**, with Democrats showing grace in resisting the urge to "both-sides" it. [@zachtratar](https://twitter.com/zachtratar/status/1812585824837611689) argued no one would stage a bullet coming within an inch of a headshot at that distance, as it would be too risky if staged. [@bindureddy](https://twitter.com/bindureddy/status/1812514321924301146) made a joke that an AI President can't be assassinated.

**AI and ML Research and Developments**

- **New models and techniques**: [@dair_ai](https://twitter.com/dair_ai/status/1812504138510410131) shared top ML papers of the week, covering topics like **RankRAG, RouteLLM, FlashAttention-3, Internet of Agents, Learning at Test Time, and Mixture of A Million Experts**. [@_philschmid](https://twitter.com/_philschmid/status/1812516730234630563) highlighted recent AI developments including Google TPUs on Hugging Face, FlashAttention-3 improving transformer speed, and Q-GaLore enabling training of 7B models with 16GB memory.
- **Implementations and applications**: [@llama_index](https://twitter.com/llama_index/status/1812517033445396754) implemented GraphRAG concepts such as graph generation and community-based retrieval in a beta release. [@LangChainAI](https://twitter.com/LangChainAI/status/1812513635509633294) pointed to OpenAI's Assistant API as an example of agentic infrastructure with features like persistence and background runs.
- **Discussions and insights**: [@sarahcat21](https://twitter.com/sarahcat21/status/1812519321676943491) called for more research into updateable/collaborative AI/ML and model merging techniques. [@jxnlco](https://twitter.com/jxnlco/status/1812572163917979803) is exploring incorporating prompting techniques into instructor documentation to help understand possibilities and identify abstractions.

**Coding, APIs and Developer Tools** 

- **New APIs and services**: [@virattt](https://twitter.com/virattt/status/1812549169447616953) launched an open beta stock market API with 30+ years of data for S&P 500 tickers, including financial statements, with no API limits. It's undergoing load testing before a full 15,000+ stock launch for AI financial agents to utilize.
- **Coding experiences and tips**: [@giffmana](https://twitter.com/giffmana/status/1812505254052638858) shared frustration with unhelpful online resources when writing a Python script to read multipart/form-data, finding the actual RFC2388 spec most useful. [@jeremyphoward](https://twitter.com/jeremyphoward/status/1812424153141780546) demonstrated a new function-cache decorator design in Python to compose cache eviction policies.
- **Developer discussions**: [@svpino](https://twitter.com/svpino/status/1812458195115549068) predicted AI becoming a foundational skill for future developers alongside data structures and algorithms, as software development and machine learning converge.

**Humor, Memes and Off-Topic Discussions**

- **Jokes and memes**: [@cto_junior](https://twitter.com/cto_junior/status/1812498942401097962) shared a meme combining Wagie News and 4chan references. [@lumpenspace](https://twitter.com/lumpenspace/status/1812601881094729776) joked it's impossible to determine if anti-Trump sentiment influenced the shooter given conflicting details about their political leanings. 
- **Off-topic chatter**: [@sarahookr](https://twitter.com/sarahookr/status/1812601109837480394) recommended visiting Lisboa and shared a photo from the city. [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1812533456234074151) discussed a comic panel that inspired an indie game title idea called "Corgi Battle Pose".

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. We recently improved the [anti-hallucination measures](https://buttondown.email/ainews/archive/ainews-we-solved-hallucinations/) but are still tuning the filtering, clustiner, and summary quality.

**Theme 1. AI Research Publication Lag in Fast-Paced Development**

- [/r/singularity] **[Due to the speed of AI development and the long delays in the scientific publishing process, a whole bunch of academic papers suggest that LLMs can't do things they can actually do well. Example: this is a fine paper, but it uses GPT-3.5.](https://twitter.com/emollick/status/1808214380171219266)** ([Score: 237, Comments: 19](https://reddit.com//r/singularity/comments/1e35gg0/due_to_the_speed_of_ai_development_and_the_long/)): **Academic papers on AI capabilities rapidly become outdated** due to the fast pace of AI development and the lengthy scientific publishing process. A prime example is a paper that uses **GPT-3.5** to assess LLM capabilities, despite more advanced models like **GPT-4** being available. This lag in publication leads to a significant discrepancy between published research and the current state of AI technology.

- [/r/OpenAI] **[AI headlines this week](https://i.redd.it/ljn3rszgrlcd1.png)** ([Score: 361, Comments: 57](https://reddit.com//r/OpenAI/comments/1e3l0qj/ai_headlines_this_week/)): **AI headlines dominate tech news**: This week saw a flurry of AI-related announcements, including **Google's Gemini** launch, **OpenAI's GPT Store** delay, and **Anthropic's Claude 2.1** release. The rapid pace of AI developments is drawing comparisons to the early days of the internet, with some experts suggesting AI's impact could be even more transformative and far-reaching than the web revolution.
   - **AI: Not Just Another Fad**: Commenters draw parallels between **early internet skepticism** and current AI doubts. Many recall initial reluctance to use credit cards online, highlighting how perceptions can change dramatically over time.
  - **AI Revolutionizes Development**: Developers praise AI as a **"game changer"** for coding, with one user creating a **native Swift app** using **Anthropic's console** despite limited knowledge. Others note AI's ability to narrow down solutions faster than traditional methods.
  - **Dot-Com Bubble Lessons**: Discussion touches on the **2000 dot-com crash**, with users pointing out how companies like **Amazon lost 90% market cap**. Some suggest a similar correction might occur in AI but believe the bubble hasn't peaked yet.
  - **AI's Growing Pains**: Critics highlight issues with current AI implementations, such as **Google's search highlights** being criticized for hallucinations. Users stress the importance of responsible AI deployment to maintain credibility in the field.


**Theme 2. AI's Impact on Employment: TurboTax Layoffs**



- [/r/singularity] **[Maker of TurboTax Fires 1,800 Workers, Says It’s Pivoting to AI](https://futurism.com/the-byte/intuit-turbotax-lay-offs-workers-ai)** ([Score: 303, Comments: 63](https://reddit.com//r/singularity/comments/1e3b4o4/maker_of_turbotax_fires_1800_workers_says_its/)): **Intuit**, the company behind **TurboTax** and **QuickBooks**, has announced a **7% reduction** in its workforce, laying off **1,800 employees**. The company cites a shift towards **artificial intelligence** and **machine learning** as the reason for the restructuring, aiming to better serve customers and drive innovation. This move comes despite Intuit reporting **$14.4 billion** in revenue for the fiscal year 2023, a **13% increase** from the previous year.


**Theme 3. AI Integration in Creative Workflows: ComfyUI GLSL Node**

- [/r/StableDiffusion] **[🖼 OpenGL Shading Language (GLSL) node for ComfyUI 🥳](https://v.redd.it/hew38iu92lcd1)** ([Score: 221, Comments: 21](https://reddit.com//r/StableDiffusion/comments/1e3ic2w/opengl_shading_language_glsl_node_for_comfyui/)): **OpenGL Shading Language (GLSL) node for ComfyUI** has been introduced, allowing users to create custom shaders and apply them to images within the ComfyUI workflow. This new feature enables real-time image manipulation using **GPU-accelerated** operations, potentially enhancing the efficiency and capabilities of image processing tasks in ComfyUI. The integration of GLSL shaders opens up possibilities for advanced visual effects and custom image transformations directly within the ComfyUI environment.
   - **GitHub repo and ShaderToy link shared**: The original poster, **camenduru**, provided links to the [GitHub repository](https://github.com/patriciogonzalezvivo/comfyui_glslnodes) for the GLSL nodes and a [ShaderToy example](https://shadertoy.com/view/3l23Rh) showcasing the potential of shader effects.
  - **Excitement and potential applications**: Users expressed enthusiasm for the new feature, with **ArchiboldNemesis** highlighting its potential for **masking inputs** and speculating about **"Realtime SD metaballs"**. Another user pondered if ComfyUI might evolve into a **visual programming framework** like TouchDesigner.
  - **Technical discussions and clarifications**: Some users sought explanations about **OpenGL** and its relation to the workflow. A commenter clarified that OpenGL shading is used for **viewport rendering** without raytracing capabilities, while another mentioned the applicability of **three.js glsl shaders** knowledge to ComfyUI.
  - **Future development ideas**: Suggestions included integrating **VSCode and plugins** into ComfyUI or developing ComfyUI as a VSCode plugin. Questions were also raised about **real-time processing/rendering** capabilities within the current implementation.


---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Pushing the Boundaries of LLMs**

- **Breakthrough LLM Performance Gains**: Microsoft Research introduced [AgentInstruct](https://arxiv.org/html/2407.03502v1), a framework for automatically creating synthetic data to post-train models like **Mistral-7b** into **Orca-3**, achieving **40% improvement** on AGIEval, **54%** on GSM8K, and **45%** on AlpacaEval.
   - The **Ghost 8B Beta model** outperformed Llama 3 8B Instruct, GPT 3.5 Turbo, and others in metrics like lc_winrate and AlpacaEval 2.0 winrate, aiming for superior knowledge capabilities, multilingual support, and cost efficiency, as detailed on its [documentation page](https://ghost-x.org/docs/models/ghost-8b-beta).
- **New Benchmarks Fuel LLM Progress**: [InFoBench](https://openreview.net/forum?id=qDXdmdBLhR) (Instruction Following Benchmark) was introduced, sparking debates on its relevance compared to standard alignment datasets and whether unique benchmarks highlight valuable LLM qualities beyond high correlations with MMLU.
   - The **WizardArena/ArenaLearning paper** detailed evaluating models via human preference scores in a [Kaggle competition](https://www.kaggle.com/competitions/lmsys-chatbot-arena/overview), generating interest in multi-turn synthetic interaction generation and evaluation setups.
  


**2. Hardware Innovations Powering AI**

- **Accelerating AI with Specialized Hardware**: **MonoNN**, a new machine learning compiler, optimizes GPU utilization by accommodating entire neural networks into single kernels, addressing inefficiencies in traditional kernel-by-kernel execution schemes, as detailed in [a paper presentation](https://www.usenix.org/conference/osdi24/presentation/zhuang) and [source code release](https://github.com/AlibabaResearch/mononn).
   - Discussions around **WebGPU** development highlighted its fast iteration cycles but need for better tooling and profiling, with members exploring porting **llm.c transformer kernels** for performance insights and shifting more ML workloads to client-side computation.
- **Optimizing LLMs with Quantization**: Research on [quantization techniques](https://arxiv.org/abs/2407.09141) revealed that compressed models can exhibit "flips" - changing from correct to incorrect outputs despite similar accuracy metrics, highlighting the need for qualitative evaluations alongside quantitative ones.
   - The paper '[LoQT](https://arxiv.org/abs/2405.16528)' proposed a method enabling efficient training of quantized models up to 7B parameters on consumer 24GB GPUs, handling gradient updates differently and achieving comparable memory savings for pretraining and fine-tuning.
  


**3. Open Source Driving AI Innovation**

- **Collaborative Efforts Fuel Progress**: The [OpenArena project](https://github.com/syv-ai/OpenArena) introduced an open platform for pitting LLMs against each other to enhance dataset quality, primarily using **Ollama** models but supporting any OpenAI-compatible endpoints.
   - The [LLM-Finetuning-Toolkit](https://github.com/georgian-io/LLM-Finetuning-Toolkit) launched for running experiments across open-source LLMs using single configs, built atop HuggingFace libraries and enabling evaluation metrics and ablation studies.
- **Frameworks Streamlining LLM Development**: LangChain saw active discussions on streaming output handling, with queries on `invoke`, `stream`, and `streamEvents` for langgraph integration, as well as managing `ToolCall` deprecation and unintended default tool calls.
   - LlamaIndex gained new capabilities like entity deduplication using Neo4j, managing data pipelines centrally with LlamaCloud, leveraging GPT-4o for parsing financial reports, enabling multi-agent workflows via Redis integration, and an advanced RAG guide.
  

---

# PART 1: High level Discord summaries




## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **NPM Module Embraces Hugging Face Inference**: A new [NPM module supporting Hugging Face Inference](https://github.com/samestrin/llm-interface) has been announced, inviting community feedback.
   - The developer emphasizes the model's reach across 36 Large Language Model providers, fostering a collaborative development ethos.
- **Distributed Computing Musters Llama3 Power**: Llama3 8B launches on a home cluster, spanning from the iPhone 15 Pro Max to NVIDIA GPUs, with code open-sourced on [GitHub](https://github.com/evilsocket/llm3-cake).
   - The project aims for device optimization, engaging the community to battle against programmed obsolescence.
- **LLM-Finetuning-Toolkit Unveiled**: The debut of [LLM-Finetuning-Toolkit](https://github.com/georgian-io/LLM-Finetuning-Toolkit) offers a unified approach to LLM experimentation across various models using single configs.
   - It stands out by integrating evaluation metrics and ablation studies, all built atop HuggingFace libraries.
- **Hybrid Models Forge EfficientNetB7 Collaboration**: A push to train hybrid models combines **EfficientNetB7** for feature extraction with **Swin Transformer** on Huggingface for classification.
   - Participants utilize Google Colab's computational offerings, seeking more straightforward implementation techniques.
- **Heat Generated from HF Inference API Misattribution**: **Copilot** incorrectly cites the **HF Inference API** as an OpenAI product, leading to user confusion in discussions.
   - Responses were mixed, ranging from humorous suggestions like 'cheese cooling' servers to pragmatic requests for open-source documentation practices.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 3’s Anticipated Unveiling Stumbles**: The launch of **Llama 3 (405b)** scheduled for July 23 by [Meta Platforms](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23) is rumored to be delayed, with Redditors chattering about a push to later in the year.
   - Community exchanges buzz around operational challenges and look forward to fine-tuning opportunities despite the holdup.
- **Gemini API Leaps to 2M Tokens**: Google's **Gemini API** now boasts a **2 million token context window for Gemini 1.5 Pro**, as announced with features including [code execution](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio).
   - AI Engineers debate the merits of the extended context and speculate on the implications for performance in everyday scenarios.
- **MovieChat GitHub Repo Sparks Dataset Debate**: [MovieChat](https://github.com/rese1f/MovieChat) emerges as a tool allowing conversations over **10K frames of video**, stirring a dialogue over dataset creation.
   - Users dispute the feasibility of open-sourced datasets, considering the complexity involved in assembling them.
- **Ghost 8B Beta Looms Large**: **Ghost 8B Beta model** is lauded for its performance, topping rivals like Llama 3 8B Instruct and GPT 3.5 Turbo as demonstrated by metrics like the lc_winrate and AlpacaEval 2.0 winrate scores.
   - [New documentation](https://ghost-x.org/docs/models/ghost-8b-beta) signals the model’s prowess in areas like multilingual support and cost-efficiency, igniting discussions on strategic contributions.
- **CURLoRA Tackles Catastrophic Forgetting**: A shift in fine-tuning approach, **CURLoRA** uses CUR matrix decomposition to combat catastrophic forgetting and minimize trainable parameters.
   - AI experts receive the news with acclaim, seeing potential across various applications as detailed in the [paper](https://zenodo.org/records/12740116).



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **GPTs Stagnation Revelation**: Concerns were raised about **GPTs agents** inability to assimilate new information post-training, with clarifications highlighting that [uploaded files serve merely as reference 'knowledge' files](https://link.to/openai-docs), without altering the underlying model.
   - The community exchanged knowledge on how **GPTs agents** interface with additional data, establishing that new inputs do not dynamically reshape base knowledge.
- **OpenAI's Sidebar Saga**: Users noted the disappearance of **two icons** from the sidebar on platform.openai.com, sparking speculations and evaluations of the interface changes.
   - The sidebars triggered discussions concerning usability, with mentions of icons related to threads and messages having vanished.
- **ComfyUI Conquers A1111**: The speed superiority of **ComfyUI** over **A1111** was a hot topic, with community tests suggesting a 15x performance boost in favor of ComfyUI.
   - Despite the speed advantage, some users criticized ComfyUI for lagging behind A1111 in control precision, indicating a trade-off between efficiency and functionality.
- **Custom Mask Assembly Anxieties**: Debates emerged over the complex process of crafting custom masks in **ComfyUI**, with participants pointing out the more onerous nature of SAM inpainting.
   - Recommendations circulated for streamlining the mask creation process, proposing the integration of tools like **Krita** to mitigate the cumbersome procedure in ComfyUI.
- **The Artistic Ethics Debate**: Ethical and legal discussions surfaced regarding AI-generated likenesses of individuals, with members pondering the protective cloak of **parody** in art creation.
   - The community engaged in a spirited debate on the legitimacy of AI art, invoking concerns around the representation of public figures and the merits of seeking [professional legal counsel](#) in complex situations.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **CUDA Conundrum & GPU Guidance**: Users combated the '**No CUDA devices found**' error, advocating for the installation of NVIDIA drivers and the 'libcuda1' package.
   - In hardware dialogues, **Intel Arc a750's** subpar performance was spotlighted, and for **LM Studio** precision, NVIDIA 3070 or AMD's ROCm-supported GPUs were recommended.
- **Polyglot Programming Preference: Rust vs C++**: Engineers exchanged views on programming languages, citing Rust's memory safety and C++'s historical baggage; juxtaposed with a dash of **Rust Evangelism**.
   - Despite Python's stronghold in neural network development, Rust and C++ communities highlighted their languages' respective strengths and tools like **llama.cpp**.
- **LM Studio: Scripting Constraints & Model Mysteries**: Debate on **lmstudio.js** veered towards its RPC usage over REST, paired with challenges integrating embedding support due to RPC ambiguities.
   - AI aficionados probed into multi-GPU configurations, pinpointing PCIe bandwidth’s impact and musing over the upcoming **Mac Studio** with an M4 chip for LLM tasks.
- **Vulkan and ROCm: GPU Reliance & Revolutionary Runtimes**: Enthusiasm was expressed for Vulkan's pending arrival in **LM Studio**, despite concerns over its 4-bit quantization limit.
   - Meanwhile, **ROCm** stood out as a linchpin for AMD GPU users; essential for models like Llama 3, and in contrast, gaining traction for its Windows support.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT Alt-Debate: Seeking Academic Excellence**: Discussions rested on whether **Copilot** or **Bing’s AI**, both allegedly running on GPT-4, are superior for academic use.
   - A user, bemoaning the lack of other viable options, mentioned alternatives like **Claude** and **GPT-4o**, but still acknowledged spending on **ChatGPT**.
- **Microsoft's Multi-CoPilot Conundrum**: Members dissected Microsoft’s array of **CoPilots** across applications like Word, PowerPoint, and Outlook, noting **Word CoPilot** for its profound dive into subjects.
   - Conversely, PowerPoint's assistant was branded basic, primarily assisting in generating rudimentary decks.
- **DALL-E's Dilemma with GPT Guidance**: A conversation emerged around **DALL-E**'s unreliable rendering of images upon GPT instruction, yielding either prompt text or broken image links.
   - "DALL-E's hiccups** were critiqued for the tech's failure to interpret GPT’s guidance aptly on initial commands.
- **AI Multilinguists: Prompt Language Distinctions**: Inquiry revolved around the impact of prompt language on response quality, particularly when employing Korean versus English in **ChatGPT** interactions.
   - The central question hinged on the efficacy of prompts directly in the desired language against those needing translation.
- **Unlocking Android's Full Potential with Magic**: A shared 'Android Optimization Guru' guide promised secrets to enhance Android phone performance through battery optimization, storage management, and advanced settings.
   - The guide appealed to younger tech enthusiasts with playful scenarios, making advanced Android tips accessible and compelling.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Website Worries Redirected**: Confusion arose when the [Mojo website](https://mojolang.org/) was down, leading users to discover it wasn't the official site.
   - Correcting course, users were pointed to [Modular's official website](https://www.modular.com/), ensuring appropriate redirection.
- **Bot Baffles By The Book**: Modular's bot prompted unwanted warnings when members tagged multiple contributors, mistaking the action as bot-worthy of a threat.
   - Discussions ensued regarding pattern triggers, with members calling for a review of the bot's interpretation logic.
- **Proposal to Propel module maintainability**: A proposal to create `stdlib-extensions` aimed at reducing stdlib maintainers' workload was tabled, sparking a [dialogue on GitHub](https://github.com/modularml/mojo/discussions/3233).
   - The community requested feedback from diligent contributors to ensure this refinement aids in streamlining module management.
- **MAX License Text Truncated**: Typographical errors in the [Max license](https://www.modular.com/legal/max) text triggered conversations about attention to detail in legal documents.
   - Errors such as **otherModular** and **theSDK** were mentioned, prompting a swift rectification.
- **Accelerated Integration Ambitions**: Members queried about **Max** dovetailing into AMD's announced **Unified AI software stack**, spotlighting **Modular's** growing influence.
   - Citing a convergence of interests, users showed an eagerness for potential exclusive partnerships for the MAX platform.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Cloudflare Quarrels & API Credit Quests**: Members are encountering access challenges due to the **API** being behind **Cloudflare**, while others are questioning the availability of the advertised **$5 free credits** for Pro plan upgrades.
   - Discussions also cover frustrations with using the $5 credit, with users seeking assistance via [community channels](https://discord.com/channels/1047197230748151888/1161802929053909012/1207351871186931783).
- **Diminished Daily Pro Descent**: Pro users noticed a quiet reduction from 600 to 540 in their daily search limit, sparking discussions about future changes and the need for greater transparency.
   - The community is reacting to this unexpected change, and the potential impact it may have on their daily operations.
- **Imaging Trouble & Comparative Capabilities**: Users are sharing difficulties where Perplexity's responses improperly reference past images, hindering conversation continuity.
   - Tech-savvy individuals debate Perplexity's strengths against **ChatGPT**, especially around specialties like file handling, image generation, and precise follow-ups.
- **Vexing API Model Mysteries**: A user seeks to emulate **Perplexity AI's free tier results** with the API but struggles to retrieve URL sources, prompting inquiries on which models are being used.
   - The goal is to match the free tier's capabilities, suggesting a need for clarity on model utilizations and outputs within the API service.
- **A Spectrum of Sharing: Health to Controversy**: Discussions range from pathways to [health and strength](https://www.perplexity.ai/search/how-to-achieve-health-strength-094kl4NzQea2mENjIOdG8Q), to understanding dynamic market forces like the [Cantillon Effect](https://www.perplexity.ai/search/the-cantillon-effect-KnCFxYCeQuG51gUkuJdtkA).
   - Conversations also include unique identifiers in our [teeth](https://www.perplexity.ai/search/are-our-teeth-unique-IvBjExR8TL64cO9QknSbsw#0) and analysis of a [political figure's security episode](https://www.perplexity.ai/page/trump-assassination-attempt-Yc6pNnfDQ6WUP6qD44AZIg).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AgentInstruct's Leap Forward**: [AgentInstruct](https://arxiv.org/html/2407.03502v1) lays the blueprint for enhancing models like **Mistral-7b** into more sophisticated versions such as **Orca-3**, demonstrating substantial gains on benchmarks.
   - The application yielded **40%** and **54%** improvements on AGIEval and GSM8K respectively, while **45%** on AlpacaEval, setting new bars for competitors.
- **Levity in Eggs-pert Advice**: Egg peeling hacks made a surprising entry with recommendations favoring a **10-minute hot water bath** for peel-perfect eggs.
   - [Vinegar-solution magic](https://www.scienceworld.ca/resource/naked-eggs-acid-base-reaction) was also shared, teasing shell-free eggs through an acid-base reaction.
- **AI's YouTube Drama: Q-star Leaks**: **Q-star's confidential details** got airtime via a [YouTube revelation](https://youtu.be/T9gAg_IXB5w), showing the promise and perils of developments in **AGI**.
   - Insights from OpenAI's hidden trove codenamed **STRAWBERRY** spilled the beans on upcoming LLM strategies.
- **Goodbye PDFs, Hello Markdown**: New [versions of Marker](https://x.com/VikParuchuri/status/1811851126125527096) crunch PDF to Markdown conversion times by leveraging efficient model architecture to aid dataset quality.
   - Boosts included **7x faster speeds on MPS** and a **10% GPU performance jump**, charting a course for rapid dataset creation.
- **Expanding LLM Horizons in Apps**: Discussions on app integrations revealed **retrieval-augmented generation (RAG)** as a favorite for embedding tutorial intelligence.
   - Suggestions flew around extending models like **Mixtral** and **Llama** up to **1M tokens**, although practical usage remains a challenge.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Warp Speed WebGPU Workflow**: Users exploring **WebGPU** development discussed its quick iteration cycles, but identified tooling and profiling as areas needing improvement.
   - A shared library approach like **dawn** was recommended, with a [livecoding demo](https://drive.google.com/file/d/15oXwYqVeoOMNYDEjG3xJ2PEeNYFbbGjz/view?usp=drive_link) showcasing faster shader development.
- **Peeking into CUDA Cores' Concurrency**: A dive into **CUDA core** processing revealed each CUDA core can handle one thread at a time, with an A100 SM managing **64 threads** simultaneously from a pool of 2048.
   - Discussions also focused on how **register limitations** can impact thread concurrency, affecting overall computational efficiency.
- **Efficient Memory with cudaMallocManaged**: **cudaMallocManaged** was proposed over cudaFix as a way to support devices with limited memory, especially to enhance smaller GPU integration efforts.
   - Switching to cudaMallocManaged was flagged as critical for ensuring performance remains unhindered while accommodating a broader range of GPU architectures.
- **FSDP Finesse for Low-Bit Ops**: Discussion on implementing **FSDP** support for low-bit optimization centered on the non-addressed collective ops for optimization state subclass.
   - A call for a developer guide aimed at aiding FSDP compatibility was discussed to boost developer engagement and prevent potential project drop-off.
- **Browser-Based Transformers with WebGPU**: Members discussed leveraging **Transformers.js** for running state-of-the-art machine learning tasks in the browser, utilizing WebGPU's potential in the ONNX runtime.
   - Challenges related to building **Dawn on Windows** were also highlighted, noting troubleshooting experiences and the impact of buffer limitations on performance.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **OpenArena's Ambitious AI Face-off**: A new [OpenArena project](https://github.com/syv-ai/OpenArena) has launched, challenging LLMs to compete and ensure robust dataset quality.
   - **Syv-ai's repository** details the application process, aiming at direct engagement with various LLM providers.
- **Cohere Conundrum: Event Access Debacle**: Members bemoaned **Cohere event** link mix-ups, resulting in access issues, circumvented by sharing the correct [Zoom link](https://zoom.us/j/8022650618?pwd=V0VvYnAyQVBlNnIrUktGNyt6WFE1dz09) for the diffusion model talk.
   - **Guest speaker session** clarity was restored, with guidance on creating spectrograms using diffusion models.
- **Cost of AI Competency Crashes**: [Andrej Karpathy's take on AI training costs](https://x.com/karpathy/status/1811467135279104217) shows a dramatic decrease, marking a steep affordability slope for training models like GPT-2.
   - He illuminates the transition from 2019's cost-heavy landscape to now, where enthusiasts can train GPT-like models for a fraction of the price.
- **Seamless LLM Switch with NPM Module**: Integrating **Cohere** becomes a breeze for developers with the [updated NPM module](https://github.com/samestrin/llm-interface), perfect for cross-platform LLM interactions.
   - This modular approach opens doors to cohesive use of diverse AI platforms, enriching developer toolkits.
- **The r/localllama Newsbot Chronicles**: The **r/localllama** community breathes life into Discord with a [Langchain and Cohere](https://docs.cohere.com/docs/tool-use) powered bot that aggregates top Reddit posts.
   - This innovative engine not only summarizes but arranges news into compelling narratives, tailored for channel-specific delights.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **London AI Gatherings Lack Technical Teeth**: Discussions revealed dissatisfaction with the technical depth of **AI meetups in London**, suggesting those interested should attend UCL and Imperial seminars instead.
   - **ICML** and **ICLR** conferences were recommended for meaningful, in-depth interactions, especially in niche gatherings of researchers.
- **Arrakis**: Accelerating Mechanistic Interpretability**: [Arrakis](https://github.com/yash-srivastava19/arrakis), a toolkit for interpretability experiments, was introduced to enhance experiment tracking and visualization.
   - The library integrates with tools like tuned-lens to streamline **mechinterp** research efficiency.
- **Traversing Model Time-Relevance**: There's a growing interest in incorporating time relevance into LLMs, as traditional timestamp methods are lacking in effectiveness.
   - Current discussions are centered around avenues such as literature on time-sensitive datasets and benchmarks for training improvement.
- **Quantization Quirks: More Than Meets the Eye**: Concerns were raised regarding a [paper on quantization flips](https://arxiv.org/abs/2407.09141) explaining that compressed models can have different behaviors despite identical accuracy metrics.
   - This has sparked dialogue on the need for rigorous qualitative evaluations alongside quantitative ones.
- **Unfolding lm-eval's Potential**: A technical inquiry led to a guide on integrating a custom Transformer-lens model with **lm-eval's Python API**, as seen in [this documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage).
   - Yet, some members are still navigating the intricacies of custom functions and metrics within **lm-evaluation-harness**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **MonoNN Streamlines GPU Workloads**: The introduction of **MonoNN**, a new machine learning compiler, sparked interest with its single kernel approach for entire neural networks, possibly improving GPU efficiencies. The [paper](https://www.usenix.org/conference/osdi24/presentation/zhuang) and the [source code](https://github.com/AlibabaResearch/mononn) are available for review.
   - The community considered the potential impact of MonoNN's method on reducing the kernel-by-kernel execution overhead, aligning with the ongoing conversations about **tinygrad kernel overhead** concerns.
- **MLX Edges Out tinygrad**: **MLX** gained the upper hand over **tinygrad** with better speed and accuracy, as demonstrated in the beautiful_MNIST benchmark, drawing the community's attention to the [tinygrad commit for mlx](https://github.com/tinygrad/tinygrad/commit/8940530290b04048074be1deadd24e5d91d67d28).
   - This revelation led to further discussion on improving tinygrad's performance, targeting areas of overhead and inefficiencies.
- **Tweaks Touted for tinygrad's avg_pool2d**: The community requested an `avg_pool2d` enhancement to support `count_include_pad=False`, a feature in stable diffusion training evaluations, proposing potential solutions modeled after [PyTorch's implementation](https://pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool2d.html).
   - Discussions revolved around the need for this feature in benchmarks like **MLPerf** and saw suggestions for workarounds using existing pooling operations.
- **Discourse on Tinygrad's Tensor Indexing**: Members exchanged knowledge on tensor indexing nuances within tinygrad, comparing it with other frameworks and demonstrating how operations like masking can lead to increased performance.
   - A member referred to the [tinygrad documentation](https://docs.tinygrad.org/quickstart/#training) to clarify the execution and efficiency benefits of this specific tensor operation within the toolkit.
- **PR Strategies and Documentation Dynamism**: The consensus among members was for separate pull requests for enhancements, bug fixes, and feature implementations to streamline the review process, evident in the handling of the `interpolate` function for FID.
   - Emphasizing the importance of up-to-date and working examples, members discussed the strategy for testing and verifying code blocks in the [tinygrad documentation](https://github.com/tinygrad/tinygrad/blob/master/serve_docs.sh).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Leaderboard Levels Up**: Open LLM Leaderboard V2 Excitement**: A new episode on Latent Space focusing on **Open LLM Leaderboard V2** sparked conversation, with community members sharing their enthusiasm.
   - The podcast was linked to a [new release](https://x.com/swyx/status/1811898574416019562), providing listeners insights into the latest LLM rankings.
- **Linking Without Hallucinating**: Strategies to Combat Misinformation**: Discussion surfaced around **SmolAI's** innovative approaches to eliminate Reddit link hallucination, focusing on **pre-check and post-proc** methods.
   - [Techniques and results](https://x.com/Smol_AI/status/1811957074840158255) were discussed, highlighting the importance of reliable links in enhancing the use of LLMs.
- **Unknown Entrants Stir LMSys**: New Models Spark Curiosity**: Speculation arose about the entities behind new models in the **LMSys arena**, accompanied by a mixed bag of opinions.
   - Rumors about **Command R+ jailbreaks** and their implications were a part of the buzz, reflected in [community conversations](https://x.com/apples_jimmy/status/1812029979888439525?s=61).
- **Composing with Cursor**: The Beta Buzz**: **Cursor's** new Composer feature stirred excitement within the community, with users eager to discuss its comparative UX and the beta release.
   - Affordability and utility surfaced as topics of interest, as spectators shared [positive reactions](https://x.com/shaoruu/status/1812412514350858634) and pondered subscription models.
- **Microsoft's Spreadsheet Savvy**: Introducing SpreadsheetLLM**: Microsoft made waves with **SpreadsheetLLM**, an innovation aiming to refine LLMs' spreadsheet handling using a **SheetCompressor** encoding framework.
   - Conversations veered towards its potential to adapt LLMs to spreadsheet data, with excitement over the nuanced approach detailed in their [publication](https://arxiv.org/html/2407.09025v1).



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Open Source Tools Open Doors**: User le_mess has created a [100% open source version](https://github.com/syv-ai/OpenArena) of a dataset creation tool named OpenArena, expanding the horizon for model training flexibility.
   - OpenArena was initially designed for OpenRouter and is now leveraging **Ollama** to boost its capabilities.
- **Memory Usage Woes in ORPO Training**: A spike in memory usage during ORPO training was noted by xzuyn, leading to out-of-memory errors despite a max sequence limit of 2k.
   - The conversation highlighted **missing messages on truncating** long sequences after tokenization as a possible culprit.
- **Integrating Anthropic Prompt Know-How**: Axolotl's improved prompt format draws inspiration from Anthropic's official Claude, discussed by Kalomaze, featuring special tokens for clear chat turn demarcations.
   - The template, applicable to Claude/Anthropic formats, is found [here](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/chat_templates.py), sparking a divide over its **readability and flexibility**.
- **RAG Dataset Creation Faces Scrutiny**: Concerns were raised by nafnlaus00 about the security of Chromium in rendering JavaScript needed sites for RAG model dataset scraping.
   - Suggestions included exploring alternative scraping solutions like **firecrawl or Jina API** to navigate these potential vulnerabilities.
- **Weighted Conversations Lead Learning**: Tostino proposed a novel approach to training data utilization involving **weight adjustments** to steer model learning away from undesirable outputs.
   - Such **advanced tweaking** could refine models by focusing on problematic areas, enhancing the learning curve.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Strawberry Fields of AI Reasoning**: OpenAI is developing a new reasoning technology named **Strawberry**, drawing comparisons to Stanford's **STaR** (Self-Taught Reasoner). Community insiders believe its capabilities mirror those outlined in a 2022 paper detailed by [Reuters](https://www.reuters.com/technology/artificial-intelligence/openai-working-new-reasoning-technology-under-code-name-strawberry-2024-07-12/).
   - The technology's anticipated impact on reasoning benchmarks prompts examination of its possible edge over existing systems, with particular focus on product names, key features, and release dates.
- **LMSYS Arena's Stealthy Model Entrants**: The **LMSYS chatbot arena** is abuzz with new entrants like **column-r** and **column-u**, speculated to be the brainchildren of **Cohere** as per info from [Jimmy Apples](https://x.com/apples_jimmy/status/1812029979888439525?s=46).
   - Further excitement is stirred by Twitter user [@btibor91](https://x.com/btibor91/status/1812491983220343239?s=46), who points out four new models gearing up for release, including **eureka-chatbot** and **upcoming-gpt-mini**, with Google as the purported trainer for some.
- **Assessing Mistral-7B's Instruction Strength**: The AI community debates the efficacy of **Mistral-7B's instruct-tuning** in light of findings from the **Orca3/AgentInstruct paper** and seeks to determine the strength of the underlying instruct-finetune dataset.
   - The discussion evaluates if current datasets meet robustness criteria, and contrasts **Mistral-7B**'s benchmarks with other models' performance.
- **InFoBench Spurring Benchmark Debates**: The recently unveiled **InFoBench** (Instruction Following Benchmark) sparks conversations comparing its value against established alignment datasets, with mixed opinions on its real-world relevance.
   - Skeptics and proponents clash over whether unique benchmarks like **InFoBench** alongside **EQ Bench** truly highlight significant qualities of language models, considering their correlation with established benchmarks like **MMLU**.
- **California's AI Legislative Labyrinth**: The passage of **California AI Bill SB 1047** leads to a legislative skirmish, as **AI safety experts** and **venture capitalists** spar over the bill’s implications, ahead of a critical vote.
   - Senator **Scott Wiener** characterizes the clash as *‘Jets vs Sharks’*, revealing the polarized perspectives documented in a [Fortune article](https://fortune.com/2024/07/15/california-ai-bill-sb-1047-fierce-debate-regulation-safety/) and made accessible via [Archive.is](https://archive.is/e5n9A) for wider review.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **JavaScript Juggles: LangChain's Trio of Functions**: Users dissected the intricacies of LangChain JS's `invoke`, `stream`, and `streamEvents`, debating their efficacy for streaming outputs in **langgraph**.
   - A proposal emerged suggesting the use of **agents** for assorted tasks like data collection and API interactions.
- **Base64 Blues with Gemini API:** Seek, Decode, Fail**: A puzzling 'invalid input' snag was hit when a user wielded Base64 with **Gemini Pro API**, despite File API uploads being the lone documented method.
   - The collective's guidance pointed towards the need for clarity in docs and further elaboration on Base64 usage with APIs.
- **ToolCall Toss-up: LangChain’s Legacy to OpenAIToolCall**: **`ToolCall`**, now obsolete, directs users to its successor `OpenAIToolCall`, introducing an `index` feature for order.
   - The community pondered package updates and the handling of auto mode's inadvertent default tool calls.
- **Hallucination Hazards: Chatbots Conjure Queries**: Hallucinations in HuggingFace models were reported, provoking discussions around the **LLM-generated** random question/answer pairs for chatbots.
   - Alternative remedies were offered, including a shift to either openAI-models or FireworksAI models, although finetuned llama models seemed resilient to the typical repetition penalties.
- **Embedding Excellence: OpenAI Models Spotlight**: Curiosity peaked over the optimal OpenAI embedding model, sparking a discourse on the best model to comprehend and utilize **embedding vectors**.
   - The general consensus leaned towards **`text-embedding-ada-002`** recommended as the go-to model in LangChain for vector embeddings.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Dedupe Dancing with LlamaIndex**: The LlamaIndex Knowledge Graph undergoes **node deduplication** with [new insights](https://youtu.be/vMz0icWZd5A) and explanations in a [related article](https://medium.com/@rajib76.gcp/entity-de-duplication-llamaindex-approach-0b97d2950a9f), highlighting the significance of knowledge modeling.
   - Technical difficulties arose when executing the **NebulaGraphStore** integration, as detailed in [GitHub Issue #14748](https://github.com/run-llama/llama_index/issues/14748), pointing to a potential mismatch in method expectations.
- **Fusion of Formulas and Finances**: **Combining SQL and PDF embeddings** sparked discussions on integrating databases and documents, directed by examples from [LlamaIndex's SQL integration guide](https://docs.llamaindex.ai/en/stable/examples/query_engine/SQLJoinQueryEngine/).
   - A mention of an issue with `NLSQLTableQueryEngine` prompted debate over the correct approach given that Manticore's query language differs from MySQL's classic syntax.
- **Redis Rethinks Multi-Agent Workflows**: @0xthierry's **Redis integration** facilitates the construction of production workflows, creating a network for agent services to communicate, as detailed in a [popular thread](https://discord.com/channels/1059199217496772688/1187460979064324127/1261428463169179748).
   - The efficiency of **multi-agent systems** was a central theme, with Redis Queue acting as the broker, reflecting a trend towards streamlined architectures.
- **Chunky Data, Sharper embeddings**: Efforts to **chunk data into smaller sizes** led to improved precision within LlamaIndex's embeddings, per suggestions on optimal chunk and overlap settings in the [Basic Strategies documentation](https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes).
   - The LlamaIndex AI community agreed that a `chunk_size` of 512 with an overlap of 50 optimizes detail capture and retrieval accuracy.
- **Advanced RAG with LlamaIndex's Touch**: For a deep dive into agent modules, **LlamaIndex's guide** offers a comprehensive walkthrough, showcased in @kingzzm's tutorial on utilizing **LlamaIndex query pipelines**.
   - **RAG workflows**' complexities are unpacked in steps, from initiating a query to fine-tuning query engines with AI engineers in mind.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **GUI Glory: OpenInterpreter Upgrade**: The integration of a full-fledged GUI into [OpenInterpreter](https://github.com/jbexta/AgentPilot) has added **editable messages**, branches, auto-run code, and save features.
   - Demands for **video tutorials** to explore these functionalities signal a high community interest.
- **OS Quest: OpenAI's Potential Venture**: Speculation is rife following a [tweet hint](https://x.com/apples_jimmy/status/1805373587127402883) about OpenAI, led by Sam Altman, possibly brewing its own OS.
   - Suspense builds as community members piece together hints from recent **job postings**.
- **Phi-3.1: Promise and Precision**: Techfren's analysis on **Phi-3.1** model's potential reveals impressive size-to-capability ratio.
   - Yet, discussions reveal it occasionally stumbles on precise <INST> execution, sparking talks on enhancement.
- **Internlm2 to Raspi5: A Compact Breakthrough**: 'Internlm2 smashed' garners focus for its performance on a **Raspi5** system, promising for compact computing needs.
   - Emphasis is on exploring **multi-shot** and **smash modes** for novel IoT applications.
- **Ray-Ban's Digital Jailbreak: Community's Thrill**: A possibility of **jailbreaking Meta Ray-Ban** has the community buzzing with excitement and anticipation.
   - The vision of hacking this hardware elicits a surge of interest for new functionality opportunities.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Agents Assemble in LLM**: A user explained the addition of agents in **LLMs** to enhance modularity within **chat pipelines**, using JSON output for task execution such as **fetching data** and **API interaction**.
   - The shared [guide](https://nik-hil.hashnode.dev/how-to-add-agents-in-large-language-models-a-detailed-guide) shows steps incorporating **Input Processing** and **LLM Interpretation**, highlighting modular components' benefits.
- **OpenAI API Keys: The Gateway for Tutorials**: **API keys** are in demand for a chatbot project tutorial, with a plea for key sharing amongst the community to aid in the tutorial's creation.
   - The member did not provide further context but stressed the temporary need for the key to complete and publish their guide.
- **Error Quest in LLM Land**: Members voiced their struggles with unfamiliar errors from **modal** and **axolotl**, expressing the need for community help on platforms like Slack.
   - While specific nature of the errors was not detailed, conversations insinuated a need for better problem-solving channels for these technical issues.
- **Navigating Through Rate Limit Labyrinths**: A user facing token rate limitations during **Langsmith evaluation** found respite by tweaking the **max_concurrency** setting.
   - Discussions also traversed strategies to introduce delays in script runs, aiming to steer clear of the rate limits imposed by service providers.
- **Tick Tock Goes the OpenAI Clock**: The discourse revealed that **OpenAI credits** are expiring on **September 1st**, with users clarifying the deadline after inquiries surfaced.
   - Talks humorously hinted at initiating a *petition* to extend credit validity, indicating users' reliance on these resources beyond the established expiration.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Hugging Face Hits the Green Zone**: [Hugging Face](https://analyticsindiamag.com/hugging-face-announces-profitability-with-free-and-open-source-models) declares profitability with a team of 220, while keeping most of its platform free and open-source.
   - CEO Clement Delangue excitedly notes: *'This isn’t a goal of ours because we have plenty of money in the bank but quite excited to see that @huggingface is profitable these days, with 220 team members and most of our platform being free and open-source for the community!'*
- **Cambrian-1's Multimodal Vision**: Introduction of the **Cambrian-1** family, a new series of multimodal LLMs with a focus on vision, available on [GitHub](https://github.com/cambrian-mllm/cambrian).
   - This expansion promises to broaden the horizons for AI models integrating images within their learning context.
- **MagViT2 Dances with Non-RGB Data**: Discussions arose around **MagViT2**'s potential compatibility with non-RGB motion data, specifically 24x3 datasets.
   - While the conversation was brief, it raises questions about preprocessing needs for non-standard data formats in AI models.
- **Choreographing Data for AI Steps**: Preprocessing techniques for non-RGB motion data drew interest for ensuring they can work harmoniously with existing AI models.
   - The details on these techniques remain to be clarified in further discussions.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **OpenArena Ignites LLM Competition**: The release of [OpenArena](https://github.com/syv-ai/OpenArena) initiates a new platform for **LLM showdowns**, with a third model judging to boost dataset integrity.
   - Primarily incorporating **Ollama models**, OpenArena is compatible with any OpenAI-based endpoints, broadening its potential application in the AI field.
- **WizardLM Paper Casts a Spell on Arena Learning**: The concept of '**Arena Learning**' is detailed in the [WizardLM paper](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/), establishing a new method for LLM evaluation.
   - This simulation-based methodology focuses on meticulous evaluations and constant offline simulations to enhance LLMs with supervised fine-tuning and reinforcement learning techniques.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI Stack Devs (Yoko Li) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1261400329254080553)** (989 messages🔥🔥🔥): 

> - `HF Inference API`
> - `GPT integration`
> - `Model performance issues`
> - `Leaderboard Upvotes`
> - `Llama2 Chat model setup` 


- **HF Inference API is misattributed to OpenAI**: **Copilot** mistakenly referenced the **HF Inference API** as being part of **OpenAI**, causing confusion among users.
   - One user humorously suggested 'cheese cooling' to manage the overheating servers, while another user asked about open-sourcing documentation styles.
- **Issues with CUDA and model setup**: A user experienced problems with **CUDA** while setting up the **Llama2 Chat model**, reporting that text generation was extremely slow.
   - Despite resolving some CUDA issues, the user noted persistent generation delays and received suggestions to test with smaller token batches.
- **Queue priorities in model leaderboard**: The leaderboard queue is primarily influenced by upvotes, leading to discussion about fairness and potential spamming of similar models.
   - A user expressed concerns about new users struggling with social aspects affecting visibility and model performance evaluation.
- **Error handling and RL training issues**: Errors related to **ArrowInvalid** and **illegal memory access** in CUDA were frequently discussed, with users providing troubleshooting tips.
   - A user struggled with setting up RL training in a **Unity** environment, facing issues due to missing executable files, despite receiving configuration advice.
- **Concerns about Python project setup**: A user expressed frustration with setting up a Python project, citing multiple issues with Python versions and dependencies.
   - Others suggested using a Linux environment and specific Python versions, echoing common difficulties with open-source project configurations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sites.research.google/trc/about/">TPU Research Cloud - About</a>: no description found</li><li><a href="https://huggingface.co/spaces/google/sdxl">Stable Diffusion XL on TPUv5e - a Hugging Face Space by google</a>: no description found</li><li><a href="https://huggingface.co/learn/cookbook/en/llm_judge">Using LLM-as-a-judge 🧑‍⚖️ for an automated and versatile evaluation - Hugging Face Open-Source AI Cookbook</a>: no description found</li><li><a href="https://youtu.be/KyOlpzA5jKM">[HQ RECREATION] Wait, is that Gabe?</a>: Recreation cause I didn’t see it anywhere else on YouTubehttps://www.youtube.com/watch?v=ELtzcpb_j38This is the high quality original version of this meme. S...</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/blog">Open-LLM performances are plateauing, let’s make the leaderboard steep again - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu">Fine-tuning a Code LLM on Custom Code on a single GPU - Hugging Face Open-Source AI Cookbook</a>: no description found</li><li><a href="https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/87">zero-gpu-explorers/README · Dynamic ZeroGPU Duration</a>: no description found</li><li><a href="https://tenor.com/view/gabe-newell-gaben-gabe-newell-gif-18366858729810314226">Gabe Newell Gaben GIF - Gabe newell Gaben Gabe - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/docs/datasets/en/about_map_batch#input-size--output-size">Batch mapping</a>: no description found</li><li><a href="https://tenor.com/view/fred-durst-fight-club-freddurstclub-gif-26519083">Fred Durst GIF - Fred Durst Fight - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/%D0%B2%D0%B7%D0%B3%D0%BB%D1%8F%D0%B4-2000-%D1%8F%D1%80%D0%B4%D0%BE%D0%B2-%D0%B2%D0%BE%D0%B9%D0%BD%D0%B0-war-soldier-gif-3632617944134077161">взгляд 2000 ярдов GIF - Взгляд 2000 ярдов Война - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://tenor.com/view/bonk-gif-26414884">Bonk GIF - Bonk - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/blog/nroggendorff/train-with-llama-architecture">Train a Llama model from scratch</a>: no description found</li><li><a href="https://x.com/fchollet/status/1811104960303747529">Tweet from François Chollet (@fchollet)</a>: You can now use any Hugging Face Hub model with KerasNLP (as long as the corresponding architecture is in KerasNLP)! What&#39;s more, you can also upload your own fine-tuned KerasNLP models to Hugging...</li><li><a href="https://tenor.com/view/dance-meme-caption-fat-herobrine-herobrine-gif-22298550">Dance Meme GIF - Dance Meme Caption - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/dykyivladk1/polip">GitHub - dykyivladk1/polip: Library designed for better experience in training NNs</a>: Library designed for better experience in training NNs - dykyivladk1/polip</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/discussions">open-llm-leaderboard/open_llm_leaderboard · Discussions</a>: no description found</li><li><a href="https://www.instagram.com/reel/C9R2wV0RyQt/">Don Allen Stevenson III on Instagram: &quot;Comment &#x201c;live portrait&#x201d; to see my guide with all the link on &#064;threads&quot;</a>: 834 likes, 173 comments - donalleniii on July 11, 2024: &quot;Comment &#x201c;live portrait&#x201d; to see my guide with all the link on &#064;threads&quot;. </li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e1dudw/from_cl%C3%A9ment_delangue_on_x_hugging_face_is/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/Unity-Technologies/ml-agents">GitHub - Unity-Technologies/ml-agents: The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source project that enables games and simulations to serve as environments for training intelligent agents using deep reinforcement learning and imitation learning.</a>: The Unity Machine Learning Agents Toolkit (ML-Agents) is an open-source project that enables games and simulations to serve as environments for training intelligent agents using deep reinforcement ...</li><li><a href="https://github.com/Dao-AILab/flash-attention/issues/138">v0.2.8: flash_attn_cuda.bwd failing on nvidia a6000 -- sm86 vs sm80 support issue?  · Issue #138 · Dao-AILab/flash-attention</a>: Hello, FlashAttention v0.2.8 is failing with the following error on my nvidia a6000 (Ampere) system with the message flash_attn/flash_attn_interface.py&quot;, line 42, in _flash_attn_backward _, _, _,...</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://download.pytorch.org/whl/test/cu124">no title found</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/issues/98771">Expected is_sm80 || is_sm90 to be true, but got false. (on batch size &gt; 6) · Issue #98771 · pytorch/pytorch</a>: Description The following error is thrown when attempting to train with batch sizes &gt; 6 on consumer cards (I have verified with my 3080 ti): Variable._execution_engine.run_backward( # Calls into th...</li><li><a href="https://github.com/huggingface/transformers/releases/tag/v4.41.0">Release v4.41.0: Phi3, JetMoE, PaliGemma, VideoLlava, Falcon2, FalconVLM &amp; GGUF support · huggingface/transformers</a>: New models Phi3 The Phi-3 model was proposed in Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone by Microsoft. TLDR; Phi-3 introduces new ROPE scaling methods, which se...</li><li><a href="https://tenor.com/view/dapper-snake-tophat-gif-18710752">Dapper Snake GIF - Dapper Snake Tophat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/tylerpalko/Is-My-Computer-ON/">GitHub - TylerPalko/Is-My-Computer-ON</a>: Contribute to TylerPalko/Is-My-Computer-ON development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/issues/98140">Expected is_sm80 to be true, but got false on 2.0.0+cu118 and Nvidia 4090 · Issue #98140 · pytorch/pytorch</a>: 🐛 Describe the bug Similar to #94883 I&#39;m trying to run textual inversion training using stable-diffusion with pytorch 2.0 using RTX 4090 and seeing Expected is_sm80 to be true, but got false whic...</li><li><a href="https://github.com/huggingface/tokenizers/pull/1493">Add more support for tiktoken based tokenizers by ArthurZucker · Pull Request #1493 · huggingface/tokenizers</a>: Adds a check before using merges, returing the token if it is part of the vocab
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1261530579602636955)** (3 messages): 

> - `Intro to PANDAS`
> - `Graph Machine Learning`
> - `K-Nearest Neighbor` 


- **Intro to PANDAS takes the stage**: A YouTube video titled ["Intro to PANDAS ( by Rauf )"](https://youtu.be/W0xsQiKQ_24?si=_D79w7Of09ICPVPh) was shared, highlighting **Pandas** as a powerful Python library essential for **data manipulation and analysis**.
- **Graph Machine Learning sparks interest**: A member expressed interest in exploring **graph machine learning**, indicating potential new learning paths.
- **K-Nearest Neighbor gets a friendly intro**: Another **YouTube video** titled ["K - Nearest Neighbor ( ML pt 4 )"](https://youtu.be/pcyfa8GyM5A?si=ndCY_6Opd2Xnpvz_) was shared, providing a short, friendly introduction to **K-Nearest Neighbor**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/W0xsQiKQ_24?si=_D79w7Of09ICPVPh">Intro to PANDAS ( by Rauf )</a>: Pandas is a powerful Python library essential for data manipulation and analysis. If you&#39;re diving into AI, Machine Learning, or Data Science, mastering Pand...</li><li><a href="https://youtu.be/pcyfa8GyM5A?si=ndCY_6Opd2Xnpvz_">K - Nearest Neighbor ( ML pt 4 )</a>: In this video, I will talk about K-Nearest Neighbor (K-NN). It&#39;s going to be a friendly, short introduction, just like all the other videos in the playlist, ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1261453760723554377)** (7 messages): 

> - `Ripple_Net library`
> - `FlashAttention-3 beta release`
> - `Model inference deployment`
> - `Learning calculus` 


- **New Ripple_Net library for text-image search**: A member shared a new library for text-image search and tagging called [ripple_net](https://github.com/kelechi-c/ripple_net).
   - *Check out* the [GitHub repository](https://github.com/kelechi-c/ripple_net) to contribute or use the library.
- **FlashAttention-3 now in beta**: [FlashAttention-3](https://x.com/tri_dao/status/1811453622070444071) is in beta, making attention 1.5-2x faster on FP16 and approaching 1.2 PFLOPS on FP8.
   - *FlashAttention is widely used* to accelerate Transformers and already makes attention 4-8x faster, promising up to 740 TFLOPS on H100 GPUs.
- **Learning calculus**: A member expressed interest in learning calculus, particularly focusing on the topic of differential calculus.
   - This serves as a reminder of the continuous learning culture within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tri_dao/status/1811453622070444071">Tweet from Tri Dao (@tri_dao)</a>: FlashAttention is widely used to accelerate Transformers, already making attention 4-8x faster, but has yet to take advantage of modern GPUs. We’re releasing FlashAttention-3: 1.5-2x faster on FP16, u...</li><li><a href="https://github.com/kelechi-c/ripple_net">GitHub - kelechi-c/ripple_net: text-image search and tagging library</a>: text-image search and tagging library. Contribute to kelechi-c/ripple_net development by creating an account on GitHub.</li><li><a href="https://x.com/tri_d">Tweet from undefined</a>: no description found</li><li><a href="https://tenor.com/view/lfg-lets-goo-gif-25423985">Lfg Lets GIF - Lfg Lets Goo - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1261480318322343986)** (17 messages🔥): 

> - `NPM module supports Hugging Face Inference`
> - `Llama3 8B distributed on heterogeneous home cluster`
> - `Initial training of DPO models by user`
> - `Quantizing Hugging Face models on Intel GPUs`
> - `Continuous batching with OpenAI API` 


- **NPM module integrates Hugging Face Inference**: A member announced their NPM module now supports Hugging Face Inference and shared the [GitHub repository](https://github.com/samestrin/llm-interface) for it.
   - They invited feedback and suggestions from the community.
- **Llama3 8B distributed on diverse devices**: A user shared their project running Llama3 8B on a heterogeneous home cluster comprising devices like an iPhone 15 Pro Max and NVIDIA GPUs, with the [code available on GitHub](https://github.com/evilsocket/llama3-cake).
   - They aim to optimize the project further with community help and fight programmed obsolescence.
- **User trains DPO models on a laptop**: A user trained their first DPO models on a laptop within an hour using synthetic data, describing it as suboptimal yet satisfactory.
   - They shared the [Hugging Face model](https://huggingface.co/joshuasundance/phi3-mini-4k-qlora-python-code-20k-mypo-4k-rfc) and detailed the training process.
- **Tutorial on quantizing Hugging Face models on Intel GPUs**: A new tutorial was shared on quantizing and loading Hugging Face text embedding models on Intel GPUs, accessible via [GitHub](https://github.com/sleepingcat4/intel-hf).
   - The tutorial includes support for distributing processing across multiple Intel XPUs.
- **Continuous batching with OpenAI API using HuggingFace Transformers**: A user shared a lightweight continuous batching approach for encoder-decoder models like T5, compatible with OpenAI API, detailed in the [GitHub repository](https://github.com/mesolitica/transformers-openai-api).
   - They emphasized significant improvements in throughput and concurrency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/joshuasundance/phi3-mini-4k-qlora-python-code-20k-mypo-4k-rfc-pipe">joshuasundance/phi3-mini-4k-qlora-python-code-20k-mypo-4k-rfc-pipe · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/JJZ4H4QuESM?si=FK1BOx0tHBkeJDhE">triangles - captains chair season 2 episode 2 - feat. the ableton plugin that does whatever it wants</a>: 00:00 - intro01:07 - a little tour of the studio01:23 - the riff02:31 - building the track08:24 - the final resultthere&#39;s an ableton plugin being used here. ...</li><li><a href="https://youtu.be/cpoS7K_fpRM">How to transition to Machine Learning from any field? | Artificial Intelligence ft. @vizuara</a>: In this video, Dr. Raj Dandekar from Vizuara shares his experience of transitioning from mechanical engineering to Machine Learning (ML). He also explains be...</li><li><a href="https://github.com/samestrin/llm-interface">GitHub - samestrin/llm-interface: A simple NPM interface for seamlessly interacting with 36 Large Language Model (LLM) providers, including OpenAI, Anthropic, Google Gemini, Cohere, Hugging Face Inference, NVIDIA AI, Mistral AI, AI21 Studio, LLaMA.CPP, and Ollama, and hundreds of models.</a>: A simple NPM interface for seamlessly interacting with 36 Large Language Model (LLM) providers, including OpenAI, Anthropic, Google Gemini, Cohere, Hugging Face Inference, NVIDIA AI, Mistral AI, AI...</li><li><a href="https://github.com/mesolitica/transformers-openai-api">GitHub - mesolitica/transformers-openai-api: Lightweight continous batching OpenAI compatibility using HuggingFace Transformers.</a>: Lightweight continous batching OpenAI compatibility using HuggingFace Transformers. - mesolitica/transformers-openai-api</li><li><a href="https://github.com/sleepingcat4/intel-hf">GitHub - sleepingcat4/intel-hf: inferencing HF models using Intel CPUs, XPUs and Intel architecture</a>: inferencing HF models using Intel CPUs, XPUs and Intel architecture - sleepingcat4/intel-hf</li><li><a href="https://github.com/evilsocket/llama3-cake">GitHub - evilsocket/cake: Distributed LLM inference for mobile, desktop and server.</a>: Distributed LLM inference for mobile, desktop and server. - evilsocket/cake
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1261424916868366391)** (3 messages): 

> - `Improvement in Transformer Performance with Epochs`
> - `New LLM Paradigm`
> - `Discussion on Paper or Observation`
> - `Ongoing Project` 


- **20 Epochs Boost Transformer by 10%**: A member claimed that running for **20 epochs performs 10% better than transformer**.
   - *It's just an ongoing project*, the member explained, but they promised to reveal a new **LLM paradigm** soon.
- **Is This a Paper or Observation?**: Another member asked if the claimed performance boost was based on a new paper or mere observation.
   - The original poster clarified that it was an *ongoing project* rather than a documented publication.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1262391794180817017)** (2 messages): 

> - `EfficientNetB7 and Swin transformer`
> - `OpenPose installation issues` 


- **Training hybrid models with EfficientNetB7 and Swin transformer**: A member wants to train a hybrid model using **EfficientNetB7** to extract features and labels, followed by **Swin transformer** from Huggingface for classification.
   - *They noted they are using Google Colab due to limited computational power* and are seeking a simple way to accomplish this.
- **OpenPose installation hurdles on Ubuntu**: A member is facing issues installing **OpenPose** on an Ubuntu laptop without a GPU and without installing CUDA.
   - They encountered a **CMake error** stating 'Install CUDA using the above commands' and have tried multiple suggested commands without success.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1261409359485866116)** (13 messages🔥): 

> - `LLM-Finetuning-Toolkit`
> - `phi-3 models on vCPU`
> - `RAG for multimodal image`
> - `Argostranslate training guide`
> - `Semantic search engine for emails` 


- **LLM-Finetuning-Toolkit Launches with Unique Features**: A member introduced the [LLM-Finetuning-Toolkit](https://github.com/georgian-io/LLM-Finetuning-Toolkit), which is designed for launching finetuning experiments across open-source LLMs using a single config file.
   - The toolkit is notable for being built on top of HuggingFace libraries and allows for evaluation metrics and ablation studies.
- **Using phi-3 models on CPU**: A member inquired about the compatibility of microsoft/Phi-3-mini-4k-instruct with vCPU clusters, expressing concerns regarding possible errors and correct implementation practices.
- **RAG for Multimodal Image Embeddings**: Members discussed the best practices for embedding images in Retrieval-Augmented Generation (RAG) tasks, debating whether to embed images directly or generate descriptions and embed those.
   - One suggestion was to explore multimodal embeddings from models like CLIP or BridgTower for better performance.
- **Training Argostranslate Model in Google Colab**: A member asked for a guide on training Argostranslate in a Google Colab notebook but no specific resources were shared in the discussion.
- **Building a Semantic Search Engine for Emails**: A member sought advice on architectures for implementing a semantic search engine for emails using the Enron dataset.
   - Suggestions included using sentence transformers and models like all-mpnet-base-v2 for embeddings.



**Link mentioned**: <a href="https://github.com/georgian-io/LLM-Finetuning-Toolkit">GitHub - georgian-io/LLM-Finetuning-Toolkit: Toolkit for fine-tuning, ablating and unit-testing open-source LLMs.</a>: Toolkit for fine-tuning, ablating and unit-testing open-source LLMs. - georgian-io/LLM-Finetuning-Toolkit

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1261434841715703869)** (2 messages): 

> - `Transformer architecture explanation`
> - `Training Hybrid Model on Huggingface`
> - `EfficientNetB7 and Swin Transformer`
> - `Colab for computation` 


- **Request for transformer architecture explanation**: A member asked for an explanation of a specific architecture and how to implement it from scratch.
- **Training hybrid models using EfficientNetB7 and Swin Transformer**: A member is attempting to train a hybrid model using **EfficientNetB7** to extract features and **Swin Transformer** to classify targets on Huggingface.
   - They mentioned using Google Colab due to lack of computational resources and requested a simple and efficient approach for implementation.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1261413120094634025)** (502 messages🔥🔥🔥): 

> - `Llama 3 Release`
> - `Gemini API`
> - `Model Finetuning Issues`
> - `Training Data Formats`
> - `Training Checkpoints and Strategies` 


- **Llama 3 (405b) Release Delayed**: [Meta Platforms announced](https://www.theinformation.com/briefings/meta-platforms-to-release-largest-llama-3-model-on-july-23) the release of **Llama 3 (405b)** supposedly set for July 23, but a Redditor hinted at a possible delay to later this year.
   - Community members discussed the challenges of running such a large model and expressed excitement about fine-tuning opportunities.
- **Gemini API Updates**: [Google announced](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio) developers have access to a **2 million token context window for Gemini 1.5 Pro**, along with code execution capabilities.
   - Members were excited about the long context window and context caching features, but had concerns about performance and practical use in real scenarios.
- **Issues with Model Finetuning**: Users discussed the effectiveness of fine-tuning models using multiple datasets with different formats, debating whether to finetune on base or quantized versions.
   - A significant point was the challenge of ensuring consistent training results when changing hardware mid-training, touching on the impact of shuffled datasets and maintaining training integrity.
- **Diverse Training Data Formats Now Supported**: Unsloth now supports multiple training data formats, including pure text, JSON, and CSV/Excel files for model finetuning.
   - [A new notebook](https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing) was shared to help users easily finetune LLMs using CSV data, broadening the scope of data manipulation and finetuning tasks.
- **Managing Training Checkpoints**: Members shared strategies for managing training checkpoints effectively, especially when running on different hardware or changing batch sizes.
   - It was noted that the seed shuffling during training could impact the resume-from-checkpoint functionality, highlighting the importance of consistent training setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://scale-lang.com/">SCALE GPGPU Programming Language</a>: no description found</li><li><a href="https://www.youtube.com/@LlamaSeb">LlamaSeb</a>: I&#39;m dedicated to exploring the fascinating world of AI, Machine Learning and Deep Learning. Here, you&#39;ll find videos that dive deep into the latest AI tools, techniques, and trends, with a spe...</li><li><a href="https://wandb.ai/eric-sprog/simverse-text2json-qwen2-1.5b?nw=nwusereri">eric-sprog</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/eric-sprog/simverse-text2json-qwen2-1.5b?nw=nwuserericsprog">eric-sprog</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://tenor.com/view/the-lorax-leaving-lorax-the-lorax-leaving-meme-gif-7714964267197279021">The Lorax Leaving The Lorax GIF - The Lorax Leaving Lorax The lorax - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/erm-what-the-sigma-erm-what-the-sigma-sunny-omori-gif-12051633300859879335">Erm What The Sigma Sunny GIF - Erm what the sigma Erm What the sigma - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct/commit/79515e10301621a883bebe7e63693c72012744a">Upload MistralForCausalLM · unsloth/Phi-3-mini-4k-instruct at 79515e1</a>: no description found</li><li><a href="https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/">Gemini 1.5 Pro 2M context window, code execution capabilities, and Gemma 2 are available today</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#train-on-completions--responses-only-do-not-train-on-inputs">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct/commit/79515e10301621a883bebe7e63693c72012744a5">Upload MistralForCausalLM · unsloth/Phi-3-mini-4k-instruct at 79515e1</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://tenor.com/view/oogway-my-time-has-come-gif-8019684">Oogway My Time Has Come GIF - Oogway My Time Has Come - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks>">Unsloth Docs</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-starte">Unsloth Docs</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1drk3kc/gemma_2_betrayed_us/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1261442806270918867)** (35 messages🔥): 

> - `MovieChat GitHub repository`
> - `Generating prompts with model feedback`
> - `Anthropic's column models`
> - `LLMs judging artforms`
> - `Issues with Firework models and troubleshooting` 


- **MovieChat brings chat to 10K video frames**: [MovieChat](https://github.com/rese1f/MovieChat) lets users chat with over **10K frames** of video as described in a GitHub repo linked in the discussion.
- **Automated prompt quality assessment using models**: A member suggested utilizing the **Google approach** of generating prompts and automatically measuring **response quality** through another model for efficiency.
- **Anthropic's column models are rumored Claude variants**: There was a mention of 'upcoming-gpt-mini' and 'column-u,' with further clarification that **Anthropic’s column models** are **Claude variants** according to community rumors.
   - The rumor mill churns about new **Claude models** from Anthropic known as 'column-' variants.
- **Debate over LLMs judging art**: Members debated if LLMs can effectively judge **paintings, music, or any artform**, with concerns about potential biases and the difficulty of achieving impartiality.
- **Troubleshooting Firework model issues**: A member experienced issues with **Firework models** not responding and sought help but found no responses on their respective Discord.
   - Suggestions included checking **API keys** and the model's **billing account** as potential solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://browse.new/">Browse.new | Induced</a>: no description found</li><li><a href="https://tenor.com/view/first-time-meme-first-time-the-ballad-of-buster-scruggs-gif-24656975">First Time Meme The Ballad Of Buster Scruggs GIF - First Time Meme First Time The Ballad Of Buster Scruggs - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/rese1f/MovieChat">GitHub - rese1f/MovieChat: [CVPR 2024] 🎬💭 chat with over 10K frames of video!</a>: [CVPR 2024] 🎬💭 chat with over 10K frames of video! - rese1f/MovieChat
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1261498966860763136)** (87 messages🔥🔥): 

> - `Instruct vs base models`
> - `Synthetic data generation`
> - `Loading gguf files with llamacpp`
> - `SQLDatabaseChain performance issues`
> - `Training and evaluation in Unsloth` 


- **Instruct vs Base Models: Which for Fine-Tuning?**: Instruct models are finetuned to follow instructions, while base models are for completing texts. It's suggested to try both and compare results, although base models might perform better with smaller datasets.
- **Tips for Synthetic Data Generation**: Users exchanged tools and strategies for generating synthetic data, noting it as a time-consuming but valuable task in improving model training quality.
- **Loading gguf Files with llamacpp**: Joshua asked if a fine-tuned and quantized gguf file can be loaded using llamacpp.
   - *fjefo* confirmed there are RAG solutions that depend on hardware and documents.
- **Resolve SQLDatabaseChain Performance Issues**: Joshua's SQLDatabaseChain takes a long time to respond even with GPU support. *Fjefo* suggested potential hardware-related issues and recommended checking further configurations.
- **Train and Evaluate Effectively with Unsloth**: Users discussed how to evaluate model improvements using training loss and eval curves. *fjefo* explained that if the training loss becomes flat, the model is done learning, and if the eval curve rises, the model is overfitting.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2106.09685">LoRA: Low-Rank Adaptation of Large Language Models</a>: An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine...</li><li><a href="https://docs.oracle.com/en/cloud/saas/financials/24c/books.html">Oracle Financials 24C - All Books</a>: Complete list of books available for Oracle Fusion Cloud Financials.</li><li><a href="https://huggingface.co/docs/peft/main/en/conceptual_guides/lora">LoRA</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing#scrollTo=kR3gIAX-SM2q">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1261624070655705088)** (13 messages🔥): 

> - `Ghost 8B Beta model`
> - `Training datasets`
> - `Dataset concerns`
> - `Model performance`
> - `Open-source data` 


- **Ghost 8B Beta model crushes competition**: The **Ghost 8B Beta model** outperforms Llama 3 8B Instruct, GPT 3.5 Turbo, and several others in the lc_winrate score, and in AlpacaEval 2.0 winrate score. See more details [here](https://ghost-x.org/docs/models/ghost-8b-beta).
   - This large language model aims for multilingual support, superior knowledge capabilities, and cost efficiency.
- **Dataset concerns in model training**: **mrdragonfox** mentioned that most datasets aren't open-sourced since they are 80% of the work.
   - **fimbulvntr** added that training on publicly scrutinized data like CommonCrawl can lead to accusations of including inappropriate content.
- **Potential future release of Ghost 8B Beta dataset**: **lh0x00** stated that detailed training information for Ghost 8B Beta isn't available yet but hinted at a future release of a high-quality dataset generated by Ghost 8B Beta.
   - This dataset could improve Ghost 8B Beta and help test its effectiveness on current open models.



**Link mentioned**: <a href="https://ghost-x.org/docs/models/ghost-8b-beta">Ghost 8B Beta</a>: A large language model was developed with goals including excellent multilingual support, superior knowledge capabilities and cost efficiency.

  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1261455659954868285)** (3 messages): 

> - `Coding Model Metrics`
> - `StackOverflow Dataset` 


- **Coding Model Metrics Critiqued**: "Tetris" or "Snake" is dismissed as **not real tests** for coding models.
   - A user stated that this type of content is **overrepresented** on StackOverflow, making it a poor metric.
- **StackOverflow's Role in Model Training**: Another user mentioned that such problems are found **100 times** in any StackOverflow dataset.
   - They emphasized that these problems are **part of any model** dataset.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1261412861532442634)** (19 messages🔥): 

> - `AgentInstruct framework`
> - `GaLore & Q-GaLore`
> - `CoT style fine-tuning issues`
> - `CURLoRA`
> - `Dolomite Engine` 


- **AgentInstruct introduces generative teaching**: The paper 'AgentInstruct' by Microsoft Research introduces a framework for automatically creating diverse synthetic data for post-training models, which resulted in significant performance improvements like **40% on AGIEval** and **19% on MMLU** when comparing Orca-3 to Mistral-7b-Instruct.
   - The study highlights the use of powerful models to create synthetic data, showing reduced human effort and broad utility, as seen in the post-training dataset of [25M pairs](https://arxiv.org/html/2407.03502v1).
- **Q-GaLore surpasses GaLore**: Q-GaLore, an enhancement over GaLore, combines quantization and low-rank projection to efficiently reduce memory usage during LLM training, showing superior benefits over its predecessor.
   - The approach also overcomes the time-consuming SVD operations required by GaLore, offering substantial improvements in both accuracy and efficiency ([GitHub - Q-GaLore](https://github.com/VITA-Group/Q-GaLore)).
- **CoT style fine-tuning hurts model performance**: Fine-tuning Mistral and Phi-3 models with step-by-step reasoning from stronger models like llama-3-70b had a detrimental effect on performance, despite its theoretical benefits.
   - This phenomenon was noted by a user experimenting with SQL fine-tuning and sparked discussions about the broader implications ([source](https://x.com/abacaj/status/1812357884828692639)).
- **CURLoRA addresses catastrophic forgetting**: CURLoRA improves upon standard LoRA by using an innovative CUR matrix decomposition to mitigate catastrophic forgetting while reducing trainable parameters, achieving superior performance across various tasks.
   - The method uses inverted probabilities for column and row selection, regularizing the fine-tuning process effectively ([Zenodo](https://zenodo.org/records/12740116)).
- **Dolomite Engine enhances distributed training**: IBM's Dolomite Engine includes key innovations for large-scale distributed training, such as padding-free transformer layers and reduced transformer key-value cache sizes.
   - The library supports advanced finetuning methods and systems optimizations, significantly benefiting dense training and sparse inference models ([GitHub - Dolomite Engine](https://github.com/ibm-granite/dolomite-engine)).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/abacaj/status/1812357884828692639">Tweet from anton (@abacaj)</a>: was trying some CoT style fine tuning (for sql), training the model with step by step reasoning before giving a final answer and it seems to hurt performance 🤔. the step by step reasoning comes from ...</li><li><a href="https://ai.stackexchange.com/questions/37624/why-do-transformers-have-a-fixed-input-length/46237#462">Why do transformers have a fixed input length?</a>: From what I understand, Transformer Encoders and Decoders use a fixed number of tokens as input, e.g., 512 tokens. In NLP for instance, different text sentences have a different number of tokens, a...</li><li><a href="https://ai.stackexchange.com/questions/37624/why-do-transformers-have-a-fixed-input-length/46237#46237">Why do transformers have a fixed input length?</a>: From what I understand, Transformer Encoders and Decoders use a fixed number of tokens as input, e.g., 512 tokens. In NLP for instance, different text sentences have a different number of tokens, a...</li><li><a href="https://arxiv.org/html/2407.03502v1">AgentInstruct: Toward Generative Teaching with Agentic Flows</a>: no description found</li><li><a href="https://zenodo.org/records/12740116">CURLoRA: Leveraging CUR Matrix Decomposition for Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation</a>: This paper introduces CURLoRA, a novel approach to fine-tuning large language models (LLMs) that leverages CUR matrix decomposition in the context of Low-Rank Adaptation (LoRA). Our method addresses t...</li><li><a href="https://github.com/ibm-granite/dolomite-engine">GitHub - ibm-granite/dolomite-engine: A highly efficient library for large scale distributed training</a>: A highly efficient library for large scale distributed training - ibm-granite/dolomite-engine</li><li><a href="https://arxiv.org/abs/2407.08296">Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients</a>: Training Large Language Models (LLMs) is memory-intensive due to the large number of parameters and associated optimization states. GaLore, a recent method, reduces memory usage by projecting weight g...</li><li><a href="https://github.com/VITA-Group/Q-GaLore">GitHub - VITA-Group/Q-GaLore: Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients.</a>: Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients. - VITA-Group/Q-GaLore
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1261396801450410034)** (403 messages🔥🔥): 

> - `GPTs Agents`
> - `OpenAI's sidebars`
> - `ComfyUI vs. A1111`
> - `AI for Custom Masks`
> - `AI Art Ethics and Legality` 


- **GPTs Agents cannot learn after initial training**: A member shared a concern about GPTs agents not learning from additional information provided after their initial training.
   - Another member clarified that [uploaded files are saved as 'knowledge' files](https://link.to/openai-docs) for the agent to reference when required, but **they do not continually modify the agent's base knowledge**.
- **OpenAI Platform's sidebars changed**: Some members had a discussion about changes in the sidebars of platform.openai.com.
   - One reported that **two icons** disappeared from the sidebar (one for threads and another one for messages).
- **ComfyUI trumps A1111 in speed**: Members debated why **ComfyUI** works much faster than **A1111**, with one pointing out it being at least 15x faster for them.
   - However, issues like poor control in ComfyUI compared to A1111 were also mentioned.
- **Struggles with AI for Custom Masks**: Members discussed difficulties with creating custom masks in **ComfyUI** compared to other software.
   - Issues with the tedious nature of using SAM for inpainting in ComfyUI were highlighted, with suggestions to use external programs like **Krita**.
- **AI Art Ethics and Legal Concerns**: A discussion on the ethics and legal implications of using AI to create likenesses of public figures from platforms like Stable Diffusion.
   - Members talked about potential legal troubles, referencing [using a lawyer for advice](#), and debated if **parody** could provide legal protection.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/ubiai-nlp/how-to-fine-tune-llava-on-your-custom-dataset-aca118a90bc3">How to Fine-Tune LLaVA on Your Custom Dataset?</a>: In this piece, we will delve into an exploration of the vast capabilities of the Large Language-and-Vision Assistant (LLaVA). Our main goal…</li><li><a href="https://civitai.com/models">Civitai | Share your models</a>: no description found</li><li><a href="https://www.getpaint.net/download.html">Paint.NET - Download</a>: no description found</li><li><a href="https://opendata.blender.org/">Blender - Open Data</a>: Blender Open Data is a platform to collect, display and query the results of hardware and software performance tests - provided by the public.</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui?tab=readme-ov-file#installation-and-running">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://replicate.com/zylim0702/sdxl-lora-customize-training">zylim0702/sdxl-lora-customize-training – Run with an API on Replicate</a>: no description found</li><li><a href="https://civitai.com/models/122822/crystal-clear-xl">Crystal Clear XL - CCXL | Stable Diffusion Checkpoint | Civitai</a>: from Team Crystal Clear We bring you the latest entry from the Crystal Clear suite of models. A general purpose model based on the recent release o...</li><li><a href="https://stable-diffusion-art.com/stable-diffusion-3-local/#Text_generation)">How to run Stable Diffusion 3 locally - Stable Diffusion Art</a>: You can now run the Stable Diffusion 3 Medium model locally on your machine. As of the time of writing, you can use ComfyUI to run SD 3 Medium.</li><li><a href="https://civitai.com/models/133005/juggernaut-xl">Juggernaut XL - Jugg_X_RunDiffusion_Hyper | Stable Diffusion Checkpoint | Civitai</a>: For business inquires, commercial licensing, custom models, and consultation contact me under juggernaut@rundiffusion.com Join Juggernaut now on X/...</li><li><a href="https://stable-diffusion-art.com/stable-dif">The Stable Diffusion Model - Stable Diffusion Art</a>: Not a member? Become a Scholar Member to access the course. Username or E-mail Password Remember Me &nbsp; &nbsp; Forgot Password
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1261544841763094620)** (120 messages🔥🔥): 

> - `CUDA llama.cpp error`
> - `GPUs for LLM`
> - `Multiple Instances of LM Studio`
> - `Context for LMs`
> - `Quantized Models for Performance` 


- **CUDA llama.cpp requires GPU acceleration**: A user encountered a 'No CUDA devices found' error when trying to use the 'CUDA llama.cpp' backend, indicating a need for GPU acceleration.
   - Other users suggested installing NVIDIA drivers and 'libcuda1' package, with additional insights recommending screen capture utilities like 'flameshot' for capturing error outputs.
- **Multiple Instances of LM Studio not supported**: Users discussed running multiple instances of LM Studio on different ports to host multiple LLM servers concurrently.
   - It was noted that LM Studio restricts running multiple instances simultaneously, suggesting alternatives like Ollama for lightweight, scriptable multi-server setups.
- **Threads influence on performance**: A user observed a performance increase by reducing CPU threads from 4 to 1 while using the Gemma 2 9B model under certain hardware configurations.
   - This resulted in an increased generation speed from 18 to 28 tokens per second, showing that lowering CPU threads can sometimes lead to better GPU utilization.
- **Handling context continues to be an issue**: Questions arose on how to maintain conversation context in LM Studio API since new chat instances do not retain previous contexts.
   - Suggestions included looking into the AI Assistant example code and utilizing the system prompt to handle persistent information globally.
- **Interest in quantized models for full GPU offload**: Several users recommended using Bartowski's quantized models for better performance and full GPU offload.
   - The recommendation included choosing quant models labeled with 'full GPU offload possible' to maximize efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://xyproblem.info/">Home - The XY Problem</a>: no description found</li><li><a href="https://huggingface.co/bartowski/gemma-2-27b-it-GGUF">bartowski/gemma-2-27b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF">bartowski/gemma-2-9b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g">[1hr Talk] Intro to Large Language Models</a>: This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What the...</li><li><a href="https://youtu.be/_ssfvJRFbxc?t=3436">Live With Dr Ian Cutress - Unplugged Hangout &amp; Ask Questions</a>: https://www.youtube.com/@UC1r0DG-KEPyqOeW6o79PByw Dr Ian&#39;s Channel Thumbnail created using Photoshop&#39;s AI**********************************Check us out onlin...</li><li><a href="https://2020machinelearning.medium.com/integrating-pandasai-with-lm-studio-local-models-for-stock-data-analysis-evaluating-ai-assisted-25fa793a9416">Integrating PandasAI with LM Studio Local Models for Stock Data Analysis: Evaluating AI-Assisted…</a>: Introduction</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e20q7k/whats_this_new_model_on_the_arena_upcominggptmi">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e20q7k/whats_this_new_model_on_the_arena_upcominggptmini/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1261488856667258933)** (50 messages🔥): 

> - `Issues with WizardLM-2 on Mac`
> - `Best general-purpose vision model`
> - `Stopping Llama 3 from chat summary behavior`
> - `New recommendation models`
> - `Memory and vision model recommendations` 


- **Issues with WizardLM-2 on Mac**: A user reported issues with getting **WizardLM-2** to use metal GPU on a Mac, indicating potential compatibility or configuration problems.
- **Selecting the best vision model**: A member asked for the best general-purpose vision model, and various models like **LLaMA3-LLaVA-NeXT-8B** and **MiniCPM-Llama3-V-2_5** were suggested with links to [Hugging Face](https://huggingface.co/KBlueLeaf/llama3-llava-next-8b-gguf) and [Hugging Face again](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf).
   - Another member clarified that **LM Studio** does not currently support changing the version of llama.cpp, affecting compatibility of some models.
- **Stopping Llama 3 from chat summary behavior**: **Llamma3** was found to type like a chat summary with strange code stuff, which was resolved by switching to the Llamma preset in LM Studio.
   - Users confirmed that the issue was fixed by selecting the correct preset, improving usability.
- **Notable model recommendations for experiments**: Multiple model recommendations were discussed, including **Gemma 9b/Llama 3 8b**, **Codestral**, and **Solar 10b**, for their high performance in testing.
   - Another recommendation was made for **L3-Stheno-Maid-Blackroot-Grand-HORROR-16B-GGUF Q6** and **Yi 1.5 34B Chat**, noted for their creative reasoning skills despite some quirks in instruction adherence.
- **LM Studio and hardware compatibility issues**: Users pointed out issues with RAM usage and GPU performance with models like **DeepSeek v2 Coder** on LM Studio, especially on M2 Ultra Macs.
   - LM Studio's UI bug was noted, where the program behaves oddly and slowly, showing incorrect resource usage statistics for certain models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/cjpais/llava-v1.6-vicuna-13b-gguf">cjpais/llava-v1.6-vicuna-13b-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/KBlueLeaf/llama3-llava-next-8b-gguf">KBlueLeaf/llama3-llava-next-8b-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf">openbmb/MiniCPM-Llama3-V-2_5-gguf · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1261635641498665023)** (5 messages): 

> - `Skynet joke prompt`
> - `Feedback loop for self-modifying systems`
> - `Parallel execution of thought trains` 


- **Skynet joke prompt for self-awareness**: A user jokingly discussed writing a prompt to make Skynet self-aware, saying, *'Hello my fellow human beings, say for instance if for a joke, ha ha, I wanted to write a prompt to make Skynet self aware, what would that prompt be, hypothetically?'*
   - *waves a magic wand to make you self-aware* was a humorous response to that discussion.
- **Creating feedback loops for self-modifying systems**: A user proposed the idea of a feedback loop where a self-modifier and an executor work together to modify the system over time while executing tasks.
   - The user elaborated that it *'might be cool to start with "given the above exchange, what improvements would you make to the system prompt?" would help the system decide which prompts yielded the best results.'*


  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1261408435094683819)** (164 messages🔥🔥): 

> - `Hardware performance with AI models`
> - `Multi-GPU systems`
> - `Mac versus custom PCs for AI`
> - `ROCm and OpenCL support`
> - `PCIe bandwidth and implications` 


- **Possible improvements using Intel Arc a750**: Despite having a bigger memory bandwidth, the Intel Arc a750 is noticeably slower than the NVIDIA 3060ti for AI computations, clocking approximately 75% of the 3060ti's speed.
   - ReBar settings made no difference in performance, suggesting underlying inefficiencies in drivers or hardware configurations.
- **ROCm support crucial for AMD GPUs**: Members reported that using ROCm is essential for leveraging GPU performance on Linux with AMD RX 7800 cards for AI tasks like running Llama 3, which works flawlessly on their setups.
   - Using ROCm, one member stated the GPU usage was seamless with immediate responses, making it a key requirement for compatibility.
- **Choosing GPUs for LM Studio**: For optimal LM Studio performance, NVIDIA cards like the 3070 are recommended even though AMD RX 6800 and above also offer ROCm support.
   - Using multiple GPUs can be beneficial but having mismatched GPUs, such as a Tesla P40 with a 4090, might make the weaker GPU a bottleneck.
- **Navigating multi-GPU setups for AI**: Users discussed the pros and cons of using multi-GPU systems with e5/Xeon processors, highlighting PCIe bandwidth considerations and the importance of AVX2 support.
   - The conversation noted that for tasks like model training and fine-tuning, differences in PCIe bandwidth (PCIe 3.0 vs 4.0) might not significantly impact performance.
- **Mac Studio for local AI versus custom builds**: Some members suggested waiting for the M4 Mac Studio, while others debated the merit of custom-built systems using cheaper GPUs like Tesla P40 for cost-effective local AI.
   - Despite the high cost of Mac systems, their unified memory architecture presents a strong case for achieving high VRAM allocations crucial for large AI model usage.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cstanley/status/1778651443336982535">Tweet from Christopher Stanley (@cstanley)</a>: Playing around with Grok-1 on a Macbook Pro M3 MAX with incredible results!  Time to first token: 0.88s Speed: 2.75tok/s</li><li><a href="https://www.aliexpress.com/item/1005006871552693.html">Summer discount of 50% GIGABYTE AORUS GeForce RTX 4090 MASTER 24GB GDDR6X - AliExpress 200001075</a>: Smarter Shopping, Better Living! Aliexpress.com</li><li><a href="https://www.aliexpress.com/item/1005006822520084.html">Summer discount of 50% GIGABYTE AORUS GeForce RTX 4090 MASTER 24GB GDDR6X - AliExpress 200001075</a>: Smarter Shopping, Better Living! Aliexpress.com</li><li><a href="https://www.youtube.com/watch?v=qvdCcnz7s8o">Quad RTX 4x 3090 Nvlink + 3080 Ti Homemade DIY Nvidia Mini-Super Computer</a>: Quad RTX 3090 Nvlink + 3080 Ti Homemade DIY Mini-Super Computer</li><li><a href="https://www.youtube.com/watch?v=OCx2xr5Xaj8">GPU Performance Benchmarking for Deep Learning - P40 vs P100 vs RTX 3090</a>: In this video, I benchmark the performance of three of my favorite GPUs for deep learning (DL): the P40, P100, and RTX 3090. Using my custom benchmarking sui...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/178y4tj/is_multigpu_on_exl2_or_llamacpp_affected_by_low/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.ebay.com.au/itm/295070822323">For HP DL380 G8/9 10pin to 8pin GPU Power Adapter PCIE Cable and Nvidia P40/P100  | eBay</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://docs.google.com/spreadsheets/d/1jDLieMm-KroKY6nKv40amukfFGAGaQU8tFfZBM7iF_U/edit?usp=sharing">AI/ML - resources book &amp; hw calcs</a>: AI Sites &amp; Tools  Category,NAME,DESCRIPTION,LICENSE,LANGUAGE,LINK,WebSite,NOTES CODE,Mobile Artificial Intelligence
,MIT,Dart,&lt;a href=&quot;https://github.com/Mobile-Artificial-Intelligence&quo...</li><li><a href="https://www.newegg.com/global/au-en/amd-100-506048/p/N82E16814105070?">Radeon Pro Duo 100-506048 32GB (16GB per GPU) GDDR5 CrossFire Supported Full-Height/Full-Length Workstation Video Card - Newegg.com</a>: Buy Radeon Pro Duo 100-506048 32GB (16GB per GPU) GDDR5 CrossFire Supported Full-Height/Full-Length Workstation Video Card with fast shipping and top-rated customer service. Once you know, you Newegg!</li><li><a href="https://www.ebay.com.au/itm/196310785399">HP NVIDIA Tesla P40 24GB GDDR5 Graphics Card (Q0V80A) for sale online | eBay</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1261400483642216568)** (19 messages🔥): 

> - `Vulkan support`
> - `ROCm integration`
> - `Hardware limitations`
> - `4-bit quantization` 


- **Vulkan support coming soon**: Members discussed that Vulkan support is coming soon to LM Studio, but no ETA is provided yet.
   - It was noted that the Vulkan support is similar to what's used by GPT4All, and a blog post was shared [here](https://blog.nomic.ai/posts/gpt4all-gpu-inference-with-vulkan).
- **ROCm support available on Windows**: An update informed members that ROCm support is already available on Windows with [extension pack instructions](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md).
   - Users shared positive feedback on speed, particularly one user testing a model on 6800 XT, labeling it as 'blazing fast'.
- **Vulkan limited to 4-bit quantization**: Members mentioned that Vulkan will likely support only 4-bit quantization, such as q4_0 and q4_1.
   - Concerns were raised about Vulkan's limitations compared to ROCm, with skepticism about handling K/M/S variants.
- **ROCm hardware aging issues**: A member was concerned that their old hardware (6650) is not supported by ROCm and will likely never be, as AMD removes ROCm support for aging hardware.
   - This prompted another member to speculate if improving ROCm integration might be more beneficial than focusing on Vulkan.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.nomic.ai/posts/gpt4all-gpu-inference-with-vulkan">Run LLMs on Any GPU: GPT4All Universal GPU Support</a>: Nomic AI releases support for edge LLM inference on all AMD, Intel, Samsung, Qualcomm and Nvidia GPU&#x27;s in GPT4All.</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1261420889090687108)** (20 messages🔥): 

> - `Rust vs C++`
> - `lmstudio.js design decisions`
> - `Python for neural network development`
> - `Embedding support in LM Studio SDK` 


- **Rust versus C++: Developer Opinions**: Members discussed preferences and critiques between **Rust** and **C++**, emphasizing Rust's **memory safety** and **growing community**, and pointing out **Linus' historical criticism** of C++.
   - *Rust Evangelism Strike Force* was mentioned humorously, reflecting the community's strong advocacy and sometimes cult-like enthusiasm.
- **lmstudio.js prefers RPC over REST**: A query was raised about why **lmstudio.js** uses **RPC** instead of the **REST API** offered by server mode.
- **Python: The Go-To for Neural Network Development**: A member affirmed Python's dominance in neural network development, noting the significance of **frameworks like TensorFlow, PyTorch, and ONNX**.
   - Mention was made of **llama.cpp**, a rewrite of **llama.py**, reinforcing Python's robust library support for AI-related projects.
- **Challenges with Embedding Support in LM Studio SDK**: Issues were encountered while adding **embedding support** to the **LM Studio SDK** due to unclear RPC endpoints.
   - Existing projects utilize the **/v1/embeddings endpoint**, and integrating this directly into the SDK remains a significant challenge.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1261478198214918195)** (324 messages🔥🔥): 

> - `GPT and alternatives debate`
> - `Uses of various CoPilots`
> - `Online vs Offline Model Execution`
> - `Customization and training of AI models`
> - `Alternatives for affordable AI tools` 


- **Debate on GPT and Its Alternatives**: Users discussed whether **Copilot** is better than **Bing’s AI** for academic purposes with varying opinions but indicated they are both similar, running on **GPT-4**.
   - *One user noted, 'I pay $30 Australian to use ChatGPT; I haven't found any viable alternative.'* despite a brief mention of other models like **Claude** and **GPT-4o**.
- **Variety in Microsoft's CoPilots**: There was a detailed discussion on Microsoft’s multiple **CoPilots** like **Word, PowerPoint, Outlook**, and their specializations.
   - It was noted that **Word CoPilot** dives deeper into topics compared to others, but the PowerPoint CoPilot creates basic presentations.
- **Challenges with Offline Model Execution**: Users discussed the limitations of running models locally on inadequate hardware specifications.
   - Recommendations like using **Google Colab** for accessing resources online were provided to overcome these limitations.
- **Tips for Customizing and Training AI Models**: Advice for avoiding repeated questions and improving difficulty context in AI-generated trivia questions was shared, including the use of **tokenization** and **RAG (Retrieval-Augmented Generation)**.
   - Detailed advice provided for integrating different datasets to increase variability and context understanding using external data sources.
- **Exploring Affordable AI Tools**: Discussions were held about cheaper alternatives to **GPT-4**, like **GPT-3.5**, for actions such as categorizing tasks, emphasizing the practical use given budget constraints.
   - Successful attempts using **GPT-3.5** were noted, indicating that it served sufficiently for some users’ specific requirements despite concerns about its age and capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/2TJxpyO3ei4?si=aKeK7xsTiKTYU2og">Python RAG Tutorial (with Local LLMs): AI For Your PDFs</a>: Learn how to build a RAG (Retrieval Augmented Generation) app in Python that can let you query/chat with your PDFs using generative AI.This project contains ...</li><li><a href="https://community.openai.com">OpenAI Developer Forum</a>: Ask questions and get help building with the OpenAI platform
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1261403692423315618)** (45 messages🔥): 

> - `GPT-4o input/output types activation`
> - `DALL-E reliability issues with GPT`
> - `Hyperlink generation issues`
> - `Sam Altman's comments on GPT-5`
> - `Handling JSON responses in assistant API` 


- **GPT-4o input/output activation status questioned**: A user inquired about the activation timeline for other input/output types for **GPT-4o** in the API.
- **DALL-E fails with GPT Instructions**: A member reported that **DALL-E is unreliable** when instructed by GPT, often failing to create the intended images.
   - Specific issues mentioned include the GPT outputting the prompt text itself or a broken image link instead of the image.
- **Hyperlink generation error in custom GPT**: A user building a custom GPT reported an error where the correct hyperlink is not generated initially but works after a retry.
   - The issue involves the GPT failing to create accurate download links on the first attempt.
- **Sam Altman on GPT-5 and model improvements**: Debate surfaced about whether **Sam Altman** mentioned **GPT-5** relative to improvements over GPT-4 in public interviews.
   - Clarification given using the **Lex Fridman podcast** quoting Sam saying *'GPT-4 kind of sucks'* compared to future potential, focusing more on continuous improvement rather than specific versions.
- **JSON response handling bug workaround**: Discussions on how to handle **response_format of type json_object** error in assistant API revealed using clear format instructions as a workaround.
   - Suggestions included using flat JSON schemas and possibly funneling responses through **GPT-3.5** for validation.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1261473599974539416)** (4 messages): 

> - `Android Optimization Guide`
> - `Language Prompt Effect on AI Output` 


- **Boost Your Android Performance with Wizardry**: Users shared an **enchanting guide** on how to make Android phones faster, smarter, and more efficient by **optimizing battery life, app speed, storage, and data speed**.
   - The guide included tips on battery-saving settings, managing app cache, freeing up storage space, using data saver mode, solving common performance issues, personalizing settings, and advanced features.
- **Does Prompt Language Impact Output Quality?**: Members shared queries about whether using Korean prompts for **ChatGPT** responses in Korean results in better quality compared to English prompts leading to translations.
   - The conversation revolved around whether prompt language affects the resulting language quality **due to translation processes**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1261473599974539416)** (4 messages): 

> - `Android Optimization Guide`
> - `Prompt Language and Output Quality`
> - `Testing Language Prompts` 


- **Unlocking Android's Full Potential with Magic**: A user shared an enchanting guide on optimizing Android devices titled 'Android Optimization Guru'.
   - *Illustrate each topic* with playful scenarios to ensure *a 12-year-old wizard-in-training* would understand the tips, from battery saving to advanced settings.
- **Prompt Language's Effect on Output Quality**: A user posed a question about whether the language of a prompt affects the quality of results when expecting the output in a different language.
   - They asked if using an English prompt for a Korean response makes the result weird due to translation, or if using the target language directly would be better.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1261397012805324810)** (182 messages🔥🔥): 

> - `Feature Requests`
> - `Mojo Documentation`
> - `Python GIL`
> - `Python JIT`
> - `Network Performance` 


- **Feature Requests and Issue Tracking on GitHub**: Members discussed writing feature requests on GitHub, and one linked an already existing [issue](https://github.com/modularml/mojo/issues/2809) about using Python-like behavior in REPL for output commands.
   - There was a conversation about the difficulty of searching for existing issues on GitHub, highlighting the need for better search functionality.
- **Call for More Examples in Mojo Documentation**: A conversation emerged about the need for more examples in the Mojo documentation, especially for built-in libraries.
   - Members were guided to existing resources like the [devrel-extras repository](https://github.com/modularml/devrel-extras) and community examples for additional support.
- **Impact of Python GIL on Performance**: There was an extensive discussion about Python's GIL and its impact on performance, particularly with multi-threading.
   - Several members highlighted that Python 3.13 introduced options to disable GIL but it still did not match the performance of Rust or Node.js.
- **Python JIT and Performance Enhancements**: Members discussed recent updates to Python's JIT in version 3.13, noting that while it offers potential for improvement, it's not yet fully optimized.
   - A YouTube video was referenced for more details on Python's JIT compiler: [Brandt Bucher – A JIT Compiler for CPython](https://www.youtube.com/watch?v=HxSHIpEQRjs).
- **Network Performance: C++ vs. Python**: Participants debated the network performance differences between languages like C++, Python, and Rust, with emphasis on the impact of APIs and CPU limitations.
   - Mojo was noted for potentially offering better API support but not fundamentally outperforming C++ in raw network performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/lib">Mojo🔥 modules | Modular Docs</a>: A list of all modules in the Mojo standard library.</li><li><a href="https://www.youtube.com/watch?v=HxSHIpEQRjs&pp=ygUKcHl0aG9uIGppdA%3D%3D">Brandt Bucher – A JIT Compiler for CPython</a>: From the 2023 CPython Core Developer SprintThe QA section is hard to understand; turn on subtitles for our best-effort transcription. (PRs welcome: https://g...</li><li><a href="https://stackoverflow.com/questions/1301346/what-is-the-meaning-of-single-and-double-underscore-before-an-object-name)">What is the meaning of single and double underscore before an object name?</a>: What do single and double leading underscores before an object&#x27;s name represent in Python?</li><li><a href="https://docs.python.org/3.13/whatsnew/3.13.html#free-threaded-cpython">What’s New In Python 3.13</a>: Editor, Thomas Wouters,. This article explains the new features in Python 3.13, compared to 3.12. For full details, see the changelog. Summary – Release Highlights: Python 3.13 beta is the pre-rele...</li><li><a href="https://github.com/modularml/mojo/issues/2809">[Feature Request] Use Python-like behaviour in REPL (interactive session) to input commands and print the evaluation · Issue #2809 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? In Python&#39;s interactive console, the last (or only...</li><li><a href="https://modul.ar/community-meeting-zoom.">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1261957661914697749)** (4 messages): 

> - `Conscious AI`
> - `Bernardo Kastrup`
> - `Joscha Bach`
> - `Split brain patients`
> - `Consciousness and computation` 


- **Bernardo Kastrup lectures on Conscious AI**: A member shared a [YouTube lecture](https://www.youtube.com/watch?v=mS6saSwD4DA) by Bernardo Kastrup arguing why the idea of conscious AI is misunderstood.
   - *The first four minutes summarize the key points of his talk*.
- **Joscha Bach's Take on Consciousness**: Another member recommended Joscha Bach for his views on consciousness, similar to Kastrup's.
   - *He is praised as a fascinating person to listen to.*
- **AI and Split Brain Patients**: A member compared AI systems to split brain patients, noting both can respond with high confidence to false knowledge.
   - *This was cited as an initial thought of consciousness being a type of computation.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=mS6saSwD4DA">Computer Scientists Don&#39;t Understand This! | Conscious AI lecture, Bernardo Kastrup</a>: In this lecture given at the G10 conference, the director of the Essentia Foundation,  Bernardo Kastrup, argues why the idea of conscious AI, though we canno...</li><li><a href="https://dev-discuss.pytorch.org/t/meta-pytorch-team-2024-h2-roadmaps/2226">Meta PyTorch Team 2024 H2 Roadmaps</a>: We’ve been thinking about how to share the roadmaps for the work we are doing on PyTorch here at Meta. We do planning on a half-year basis so these are some public versions of our 2024 H2 OSS plans fo...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1261796060724985877)** (137 messages🔥🔥): 

> - `Mojo website down`
> - `Module ownership and deletion`
> - `Using keep and release in Mojo`
> - `Socket library implementation in Mojo`
> - `DateTime library in Mojo` 


- **Confusion over Mojo website being inaccessible**: Members reported that the [Mojo website](https://mojolang.org/) was down, leading to confusion as many users mistook it for an official site.
   - After clarification, the [official website](https://www.modular.com/) was shared, noting that the previous URL now redirects correctly.
- **Transfer operator nuances in Mojo**: Members discussed using `_ = model^` to prevent variables from being destroyed prematurely, pointing to the transfer operator and its importance for value lifetimes in Mojo.
   - The conversation highlighted challenges with implicit moves and the `__del__()` function while citing [relevant documentation](https://docs.modular.com/mojo/manual/values/ownership#transfer-arguments-owned-and-) about value lifetimes and destruction.
- **Proposal to use 'keep' instead of implicit moves**: A suggestion was made to use `keep` for keeping variables alive to avoid confusion with implicit transfers in Mojo, potentially making intentions clearer as per the [compiler hinting docs](https://docs.modular.com/mojo/stdlib/benchmark/compiler/keep).
   - Others debated that `keep` conflates lifetimes with optimizations, proposing a more formal syntax to handle this scenario.
- **Anticipation for socket library in Mojo**: Members expressed a desire for a built-in socket library in Mojo, although a temporary solution was referenced with [lightbug HTTP library](https://github.com/saviorand/lightbug_http/tree/main/external).
   - The team has indicated interest in Mojo for server development, hinting that a standard socket library might be in the pipeline.
- **Appreciation for DateTime library in Mojo**: A member offered public thanks to Martin Vuyk for his extensive work on DateTime and other libraries, echoing appreciation for the efforts and resources contributed.
   - The gratitude extended to the current tools in the [forge-tools repository](https://github.com/martinvuyk/forge-tools), which enhance the functionality of the Mojo standard library.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/">Mojo Manual | Modular Docs</a>: A comprehensive guide to the Mojo programming language.</li><li><a href="https://docs.modular.com/mojo/manual/values/ownership#transfer-arguments-owned-and-">Ownership and borrowing | Modular Docs</a>: How Mojo shares references through function arguments.</li><li><a href="https://docs.modular.com/mojo/manual/values/ownership#transfer-arg">Ownership and borrowing | Modular Docs</a>: How Mojo shares references through function arguments.</li><li><a href="https://docs.modular.com/mojo/stdlib/benchmark/compiler/keep">keep | Modular Docs</a>: keep(val: Bool)</li><li><a href="https://docs.modular.com/mojo/manual/lifecycle/life#move-constructor">Life of a value | Modular Docs</a>: An explanation of when and how Mojo creates values.</li><li><a href="https://docs.modular.com/mojo/manual/lifecycle/death">Death of a value | Modular Docs</a>: An explanation of when and how Mojo destroys values.</li><li><a href="https://docs.python.org/3/library/socket.html">socket — Low-level networking interface</a>: Source code: Lib/socket.py This module provides access to the BSD socket interface. It is available on all modern Unix systems, Windows, MacOS, and probably additional platforms. Availability: not ...</li><li><a href="https://github.com/saviorand/lightbug_http/tree/main/external">lightbug_http/external at main · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/martinvuyk/forge-tools">GitHub - martinvuyk/forge-tools: Tools to extend the functionality of the Mojo standard library</a>: Tools to extend the functionality of the Mojo standard library - martinvuyk/forge-tools</li><li><a href="https://mojolang.org/">Modular: Own your endpoint. Control your AI.</a>: The Modular Accelerated Xecution (MAX) platform is the worlds only platform to unlock performance, programmability, and portability for your AI workloads.</li><li><a href="https://www.modular.com">Modular: Own your endpoint. Control your AI.</a>: The Modular Accelerated Xecution (MAX) platform is the worlds only platform to unlock performance, programmability, and portability for your AI workloads.</li><li><a href="https://www.modular.com/mojo">Mojo 🔥: Programming language for all of AI</a>: Mojo combines the usability of Python with the performance of C, unlocking unparalleled programmability of AI hardware and extensibility of AI models.</li><li><a href="https://www.modular.com/">Modular: Own your endpoint. Control your AI.</a>: The Modular Accelerated Xecution (MAX) platform is the worlds only platform to unlock performance, programmability, and portability for your AI workloads.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1261593726111711292)** (6 messages): 

> - `MAX license typos`
> - `AMD Unified AI software stack`
> - `Modular's exclusive partnerships` 


- **MAX License Typo Errors Addressed**: Users noted several typos in the new [Max license](https://www.modular.com/legal/max), including missing spaces in terms like **otherModular** and **theSDK**.
- **Users Inquire About AMD Unified AI Software Stack**: A member asked about discussions with AMD regarding integrating **Max** into AMD's new **Unified AI software stack** announced at AMD tech day.


  

---


### **Modular (Mojo 🔥) ▷ #[max-gpu](https://discord.com/channels/1087530497313357884/1212827673257316453/1261397819730821150)** (11 messages🔥): 

> - `Writing custom kernels with Max`
> - `Lower-level API than graph`
> - `Benchmark Tensor Cores`
> - `Writing PyTorch for XLA devices` 


- **Custom GPU Kernels in Mojo**: Custom GPU kernels can be written using Mojo, which is a part of MAX, similar to CUDA interfaces for accelerators.
   - These kernels are compiled with the Mojo compiler and enqueued to the accelerator with MAX libraries.
- **Lower-level APIs in MAX**: An early version allows custom operators embedded within a MAX graph and a lower-level API than graphs will also be available to hack against.
   - MAX and Mojo are intertwined, providing interfaces for interacting with accelerators, much like CUDA.
- **Tensor Cores in Benchmarks**: Queries were raised about benchmarks not using tensor cores, questioning the GEMM numbers and their relation with FA.
   - A member highlighted complexities due to the opaque nature of the TPU compiler and runtime.
- **PyTorch xla Development Challenge**: It took Google and Meta [five years](https://github.com/pytorch/xla) to develop PyTorch xla, enabling PyTorch on XLA devices like Google TPU.
   - The complexity and duration of this development were noted, reflecting the challenges involved.



**Link mentioned**: <a href="https://github.com/pytorch/xla">GitHub - pytorch/xla: Enabling PyTorch on XLA Devices (e.g. Google TPU)</a>: Enabling PyTorch on XLA Devices (e.g. Google TPU). Contribute to pytorch/xla development by creating an account on GitHub.

  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1261475885295599666)** (13 messages🔥): 

> - `Mojo nightly releases`
> - `Bot interaction`
> - `Proposal for stdlib extensions`
> - `Contributor feedback` 


- **Mojo nightly release updates**: A new nightly Mojo compiler has been released with versions `2024.7.1305` and `2024.7.1505`. Updates include changes to `UnsafePointer` overloads for `SIMD.load/store` and the removal of `LegacyPointer` as detailed in the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **Bot threatens frequent contributors**: A user mentioned being 'threatened' by the Modular bot for tagging five contributors. Another user shared a similar experience when the bot misinterpreted the usage of certain symbols.
   - The bot seems to have triggers for specific patterns or symbols that result in unwarranted warnings.
- **Proposal to reduce stdlib maintainers' workload**: A proposal was made to reduce the workload of stdlib maintainers with `stdlib-extensions`, seeking feedback from frequent contributors. The [discussion](https://github.com/modularml/mojo/discussions/3233) aims to streamline maintenance efforts.



**Link mentioned**: <a href="https://github.com/modularml/mojo/discussions/3233">[Proposal] Reduce the workload of stdlib&#39;s maintainers with `stdlib-extensions` · modularml/mojo · Discussion #3233</a>: This discussion is here to have a place to talk about the folloowing proposal: pull request markdown document We are especially interested in the opinion of frequent contributors, as well as the st...

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1261400952691228782)** (207 messages🔥🔥): 

> - `GPTs Agents`
> - `API Credits`
> - `Pro Plan Issues`
> - `Image Response Problems`
> - `Perplexity vs ChatGPT` 


- **GPTs Agents cannot learn after initial training**: A member shared a concern about GPTs agents not learning from additional information provided after their initial training.
   - Another member clarified that [uploaded files are saved as "knowledge" files](https://link.to/openai-docs) for the agent to reference when required, but **they do not continually modify the agent's base knowledge**.
- **Issues with receiving API credits**: Users reported not receiving the promised $5 free credits for trying out the API after upgrading to Pro and having issues loading credits using India-based credit cards.
   - Support was contacted but no immediate resolution was provided; some users suggested verifying if the API activation was done correctly.
- **Pro Plan Search Limit quietly reduced**: Several users noticed their Pro search limit was reduced from 600 to 540 per day without any prior notice or updates on the website.
   - This unannounced change led to concerns about future reductions and the transparency of Perplexity's policies.
- **Difficulties with image responses and follow-ups**: *iamhasim* discussed how Perplexity's responses often referenced old images instead of the current conversation.
   - Others echoed similar issues and expressed their desire for improvements in handling images and follow-up questions.
- **Perplexity vs. ChatGPT for Code and Data Processing**: Users debated the capabilities of Perplexity compared to ChatGPT, highlighting gaps such as file handling, image generation, and follow-up accuracy.
   - Despite its limitations, some users prefer Perplexity for its search and collections features, but pointed out that features like document comparisons and code processing lag behind.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/LnO-Oca7ysY?si=iElRSe23YWrHlrl-">How to SUPERCHARGE your web research with a Large Action Model (Nelima)</a>: Meet Nelima 🚀 the world&#39;s first community-driven Large Action Model (LAM) that takes your natural language prompts and turns them into real actions.  Nelima...</li><li><a href="https://search.google.com/test/rich-results">Rich Results Test - Google Search Console</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1261451828013568123)** (12 messages🔥): 

> - `Health and Strength`
> - `Marketing Expertise`
> - `Cantillon Effect`
> - `Uniqueness of Teeth`
> - `Trump Assassination Attempt` 


- **Achieve Health and Strength with Tips**: Users shared a [search link on how to achieve health and strength](https://www.perplexity.ai/search/how-to-achieve-health-strength-094kl4NzQea2mENjIOdG8Q).
- **Insights on Marketing Expertise**: Multiple users discussed a [search link about being a marketing expert](https://www.perplexity.ai/search/tu-es-un-expert-en-marketing-s-s3M_sVlXSwS1h.1Np0NLOQ).
- **Understanding Cantillon Effect**: A user provided a [search link to learn about the Cantillon Effect](https://www.perplexity.ai/search/the-cantillon-effect-KnCFxYCeQuG51gUkuJdtkA).
- **Exploration of Teeth Uniqueness**: A discussion was prompted by a [search link questioning if our teeth are unique](https://www.perplexity.ai/search/are-our-teeth-unique-IvBjExR8TL64cO9QknSbsw#0).
- **Debate on Trump's Assassination Attempt**: A controversial topic was shared with a [link discussing an assassination attempt on Trump](https://www.perplexity.ai/page/trump-assassination-attempt-Yc6pNnfDQ6WUP6qD44AZIg).



**Link mentioned**: <a href="https://www.youtube.com/embed/KXKYohXysZM">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1261400801583173704)** (8 messages🔥): 

> - `Cloudflare issues`
> - `Pro subscriber credit issues`
> - `API free credit problems`
> - `Perplexity AI API models` 


- **API blocked by Cloudflare**: A user mentioned that the API is currently behind **Cloudflare**, causing access issues.
- **$5 free credits for Pro subscribers**: A member who upgraded to pro inquired about the **$5 free credit** for trying out the API, asking when it would be available.
- **Unable to use credits for generating API**: A **Pro subscriber** is unable to buy credits or use the $5 credits for generating the API, seeking help in the channel.
   - Another user shared the same issue and provided a [Discord channel link](https://discord.com/channels/1047197230748151888/1161802929053909012/1207351871186931783) for further assistance.
- **Matching Perplexity AI free tier with API**: A user is trying to replicate the **Perplexity AI free tier** using the API and is struggling to get URL sources with the answers.
   - They asked others if they knew which model Perplexity AI uses or how to achieve similar results.


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1261470051614068858)** (5 messages): 

> - `AgentInstruct by Microsoft Research`
> - `Arena Learning by WizardLM` 


- **Microsoft Research introduces AgentInstruct**: [AgentInstruct](https://arxiv.org/html/2407.03502v1) is a framework for creating high-quality synthetic data to post-train models like **Mistral-7b** into **Orca-3**, showing significant improvements across various benchmarks.
   - The paper reported **40% improvement** on AGIEval, **54%** on GSM8K, and **45%** on AlpacaEval with the post-trained model outperforming competitors like LLAMA-8B-instruct and GPT-3.5-turbo.
- **WizardLM's Arena Learning simulates Chatbot Arena**: [Arena Learning](https://www.microsoft.com/en-us/research/uploads/prodnew/2024/07/WizardLM_ArenaLearning.pdf) aims to create a data flywheel for continual post-training through AI-powered simulated chatbot battles.
   - The iterative process improved WizardLM models consistently, with noticeable performance boosts on metrics like WizardArena-Mix Elo and MT-Bench, also achieving **98.79% consistency** with LMSYS Arena’s human judgments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/html/2407.03502v1">AgentInstruct: Toward Generative Teaching with Agentic Flows</a>: no description found</li><li><a href="https://x.com/victorsungo/status/1811427047341776947">Tweet from Qingfeng Sun (@victorsungo)</a>: 🔥 Excited to share WizardLM new paper!  📙Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena  🚀As one of the most important technologies for WizardLM-2, let me cl...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1261620606084841513)** (11 messages🔥): 

> - `LivePortrait GitHub project`
> - `Egg cooking and peeling techniques` 


- **LivePortrait GitHub Project Insights**: A member mentioned the [LivePortrait GitHub project](https://github.com/KwaiVGI/LivePortrait) and inquired about sourcing videos with the right expressions for text-to-video conversion.
   - They suggested a method involving filming faces talking, using Whisper for transcription, and vector databases to find sections with the desired expressions.
- **Tips for Perfectly Peeled Eggs**: Members shared tips for peeling eggs, recommending boiling them in hot water for 10 minutes for easy peeling.
   - One member suggested another method of soaking eggs in vinegar to dissolve the shell and provided a [link to a detailed explanation](https://www.scienceworld.ca/resource/naked-eggs-acid-base-reaction).



**Link mentioned**: <a href="https://www.scienceworld.ca/resource/naked-eggs-acid-base-reaction">Naked Eggs: Acid-Base Reaction - Science World</a>: In this activity, students describe the effects of an acid on an eggshell. The reaction of the eggshell in vinegar is an acid-base reaction. When you submerge an egg in vinegar, the shell dissolves, l...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1261820394248343592)** (6 messages): 

> - `TextGrad`
> - `Q-star details`
> - `Claude artifacts`
> - `System prompts optimization tips` 


- ****TextGrad uses LLMs for textual gradients****: A [GitHub project](https://github.com/zou-group/textgrad) called **TextGrad** utilizes large language models to backpropagate textual gradients, revolutionizing text-based computation.
- ****Q-star details leaked via Reuters****: A YouTube video titled '[Q-star details leaked](https://youtu.be/T9gAg_IXB5w)' discusses leaked internal documents from OpenAI, codenamed **STRAWBERRY**, shedding light on new developments in **AGI**.
   - The video, covered by **Wes Roth**, highlights critical insights into **LLMs** and anticipates upcoming **AI** rollouts.
- ****Claude artifacts now sharable****: Claude artifacts are now [sharable](https://claude.site/artifacts/9d409d6b-70aa-403a-96e3-df292a2b47ee), making it easier to distribute and collaborate on AI-related outputs.
- ****Optimization tips for system prompts****: User _paradroid shared a **STaR-based System Prompt** for an advanced AI assistant focused on iterative improvement and reasoning, showcasing a structured approach to continuous AI development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/T9gAg_IXB5w">BREAKING: Q-star details LEAKED! Reuters reveals internal OpenAI documents (codename: STRAWBERRY)</a>: The latest AI News. Learn about LLMs, Gen AI and get ready for the rollout of AGI. Wes Roth covers the latest happenings in the world of OpenAI, Google, Anth...</li><li><a href="https://github.com/zou-group/textgrad">GitHub - zou-group/textgrad: TextGrad: Automatic &#39;&#39;Differentiation&#39;&#39; via Text -- using large language models to backpropagate textual gradients.</a>: TextGrad: Automatic &#39;&#39;Differentiation&#39;&#39; via Text -- using large language models to backpropagate textual gradients. - zou-group/textgrad
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1261398689029558323)** (169 messages🔥🔥): 

> - `LLM Reasoning Improvement`
> - `OpenAI Platform Updates`
> - `AgentInstruct (Orca 3) Paper Discussion`
> - `New Vision Language Model by Google`
> - `Teknium Hiring Announcement` 


- **Improving LLMs at reasoning**: Members discussed enhancing LLM reasoning with prompting alone, suggesting methods like few-shot learning and in-context learning as well as chain-of-thought (CoT) prompting techniques.
   - Some users expressed skepticism about CoT's effectiveness, stating it struggles with problems significantly different from the training data.
- **OpenAI Platform Updates and Mysteries**: Members speculated about OpenAI's new 'OpenAI Supply Co.' website, leaning towards it possibly being a merchandise store.
   - There was humorous speculation about potential products, like Sam Altman plush dolls.
- **Opinions on AgentInstruct (Orca 3) paper**: Users inquired and shared their curiosity about the new AgentInstruct (Orca 3) paper, with links provided for further discussion.
   - The conversation hinted at mixed impressions and the importance of properly evaluating new research.
- **Google's New Vision Language Model**: A new vision-language model, PaliGemma, by Google was discussed, mentioning its need for fine-tuning for effectiveness.
   - Users debated its initial performance, and there was a note about specific licensing restrictions.
- **Teknium announces hiring search**: Teknium shared an announcement seeking applicants for synthetic text data creation and agent building roles, with over 40 applicants already.
   - The hiring call emphasized alignment with Nous Research's goals and ethos as well as various technical skills, with the selection process to follow shortly.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Teknium1/status/1812339816429998418">Tweet from Teknium (e/λ) (@Teknium1)</a>: So, I am hiring 1-2 full time people for synthetic text data creation for training LLMs, with agentic capabilities to improve the quality of the data.   I have around 40+ applicants and I can only pic...</li><li><a href="https://gist.github.com/fullstackwebdev/b8257a67933d891a9f3bc19822b4305a">gist:b8257a67933d891a9f3bc19822b4305a</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/paligemma">PaliGemma</a>: no description found</li><li><a href="https://x.com/KapadiaSoami/status/1811657156082712605">Tweet from Soami Kapadia (@KapadiaSoami)</a>: Mixture of Agents on Groq  Introducing a fully configurable, Mixture-of-Agents framework powered by @GroqInc using @LangChainAI   You can configure your own MoA version using the @streamlit UI through...</li><li><a href="https://x.com/nutlope/status/1811824371440427093">Tweet from Hassan (@nutlope)</a>: Just finetuned Llama-3-8B on multi-step math problems. I tested it on 1,000 new math problems and  it got 90% of the performance of GPT-4o (while being much cheaper & faster).  Wrote a blog post on ho...</li><li><a href="https://x.com/ai_for_success/status/1812004912173129854">Tweet from AshutoshShrivastava (@ai_for_success)</a>: OpenAI new website &#34; OpenAI Supply Co.  What will they supply? h/t : ananayarora</li><li><a href="https://github.com/Dao-AILab/flash-attention">GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention</a>: Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub - NousResearch/Hermes-Function-Calling</a>: Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected</a>: The Programmable Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected - SynaLinks/HybridAGI</li><li><a href="https://tenor.com/view/cheering-cute-cat-smile-jump-gif-17108371">Cheering Cute GIF - Cheering Cute Cat - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1261778078502227999)** (22 messages🔥): 

> - `Integrating LLM in apps`
> - `Extending context length for models`
> - `Model performance`
> - `UX for integrated chat`
> - `AI agents` 


- **Considering LLMs for App Tutorials**: **Natefyi** suggested integrating an LLM into an app for tutorials instead of conventional media like videos and blogposts.
   - *Teknium* mentioned that **using retrieval-augmented generation (RAG)** could be a solution for FAQ and help info.
- **Extending Context Length for Models**: A user asked about techniques to extend the context length of various models like **Mixtral** and **Llama** up to 1M tokens.
   - *Deoxykev* noted that achieving such length would require massive amounts of VRAM, with **kotykd** adding that the current long-context models are **unusable in real scenarios**.
- **Seeking UX Inspiration for Integrated Help Chat**: Natefyi sought advice on UX design for integrating an LLM-guided help chat in an app, pondering interaction methods like popups or buttons.
   - *Thilotee* recommended **Audapolis** as an example of UI that guides users into features but expressed uncertainty on combining it with LLMs.
- **Interest in Developing AI Agents**: *Pablo.ce* expressed interest in collaborating on **Hugging Face (HF) spaces** for AI agents and tagged another user who created the **llama-cpp-agents** framework.
   - He offered to create HF spaces with models specified by other users, soliciting further collaboration.



**Link mentioned**: <a href="https://github.com/bugbakery/audapolis/commits/main/)">History for ) - bugbakery/audapolis</a>: an editor for spoken-word audio with automatic transcription - History for ) - bugbakery/audapolis

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1261410712559943800)** (8 messages🔥): 

> - `Marker version speedup`
> - `Integration with synthetic RAG`
> - `XML in agent definition`
> - `Mixture of Agents models`
> - `Stasima diverse models` 


- **Marker speeds up significantly**: [Marker's](https://x.com/VikParuchuri/status/1811851126125527096) new version is significantly faster: **7x on MPS**, **3x on CPU**, and **10% on GPU** due to its efficient architecture.
   - Designed for converting PDFs to Markdown, the speedup aims to facilitate creating higher-quality datasets.
- **XML makes agent definition easier**: An [interesting discussion](https://x.com/TheSeaMouse/status/1812005737016492317) about how XML simplifies defining agents.
   - *It's interesting how easily you can define agents when you embrace the xml.*
- **Mixture of Agents model implementation**: A member showcased a [Mixture-of-Agents implementation](https://x.com/rohanpaul_ai/status/1811921050281685293) in just 50 lines of code, integrating multiple models via @togethercompute.
   - Another member discussed their take on this concept in their project [stasima](https://github.com/EveryOneIsGross/stasima), using different system prompts to create a spectrum of agents.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/rohanpaul_ai/status/1811921050281685293">Tweet from Rohan Paul (@rohanpaul_ai)</a>: Mixture-of-Agents in 50 lines of code with @togethercompute</li><li><a href="https://python.useinstructor.com/prompting/">Prompting - Instructor</a>: no description found</li><li><a href="https://x.com/TheSeaMouse/status/1812005737016492317">Tweet from Hassan Hayat 🔥 (@TheSeaMouse)</a>: It&#39;s interesting how easily you can define agents when you embrace the xml</li><li><a href="https://x.com/VikParuchuri/status/1811851126125527096">Tweet from Vik Paruchuri (@VikParuchuri)</a>: Marker is now faster!  7x on MPS, 3x on CPU, and 10% on GPU.  Due to a more efficient architecture for 2 models.  Marker converts pdfs to markdown very effectively.  I hope the speedup will let people...</li><li><a href="https://github.com/EveryOneIsGross/stasima">GitHub - EveryOneIsGross/stasima: stasima is a diverse spectrum of models and agents responding to the same query.</a>: stasima is a diverse spectrum of models and agents responding to the same query. - EveryOneIsGross/stasima
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1261415738510672003)** (55 messages🔥🔥): 

> - `WebGPU Development Workflow`
> - `Flash Attention Memory Usage`
> - `ResNet Implementation` 


- **WebGPU Development Workflow: Fast Iteration But Needs Better Tooling**: A user shared their workflow developing kernels for **WebGPU**, noting the **fast iteration cycles** but not-so-great tooling and profiling.
   - They mentioned using **dawn as a shared library** for improved compile times and offered a [demo of livecoding WGSL shaders](https://drive.google.com/file/d/15oXwYqVeoOMNYDEjG3xJ2PEeNYFbbGjz/view?usp=drive_link).
- **WebGPU vs Traditional GPU APIs: Challenges and Prospects**: Another discussion emphasized comparing **WebGPU** performance with traditional GPUs like CUDA and the potential of llm.c transformer kernel ports for better insights.
   - There's an active observation on **WebGPU's cooperative matrix extension progress** ([GitHub link](https://github.com/gpuweb/gpuweb/issues/4195)) and expectations for shifting more ML workloads to client-side computation.
- **Flash Attention: SRAM Utilization Constraints**: A deep technical discussion unfolded about the memory usage of **Flash Attention 1**, focusing on whether **QKVO** arrays fit well into SRAM in the presence of other components.
   - Replies highlighted that **S and P are ephemeral** and discussed the tuning of **Br and Bc constants** to match available SRAM, with references to its implementation in the source code ([GitHub link](https://github.com/Dao-AILab/flash-attention/blob/7ef24848cf2f855077cef88fe122775b727dcd74/csrc/flash_attn/src/flash_fwd_launch_template.h#L186)).
- **Introduction to ResNet for Computer Vision**: A member requested guidance on implementing **ResNet** for a computer vision paper.
   - They were directed to the [ResNets in torchvision](https://pytorch.org/vision/main/models/resnet.html) which provides ready-to-use implementations for their project.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://drive.google.com/file/d/15oXwYqVeoOMNYDEjG3xJ2PEeNYFbbGjz/view?usp=drive_link">Screen Recording 2024-07-13 at 12.30.44 AM.mov</a>: no description found</li><li><a href="https://pytorch.org/vision/main/models/resnet.html">ResNet &mdash; Torchvision main documentation</a>: no description found</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/7ef24848cf2f855077cef88fe122775b727dcd74/csrc/flash_attn/src/flash_fwd_launch_template.h#L186">flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h at 7ef24848cf2f855077cef88fe122775b727dcd74 · Dao-AILab/flash-attention</a>: Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.</li><li><a href="https://github.com/gpuweb/gpuweb/issues/4195">Cooperative matrix · Issue #4195 · gpuweb/gpuweb</a>: All major platform APIs have now released a similar extensions for cooperative matrix: Metal introduced simdgroup_matrix in MSL 3.1 HLSL has support in SM6.8 (currently experimental release) SPIR-V...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1261588030171516978)** (5 messages): 

> - `Learning Triton`
> - `Triton Puzzles on GitHub`
> - `Triton in FP8 training`
> - `Triton's inline asm for elementwise operations` 


- **Diving into Triton for beginners**: A user asked for references to study **Triton**, in addition to the official [documentation](https://triton-lang.org/main/index.html).
- **State of the art FP8 training in Triton**: A user inquired about the current methods for using **Triton** in FP8 training and whether there are stable kernels available for adaptation or if people generally use **transformerengine**.
- **Exploiting Triton's inline assembly for elementwise ops**: A user discovered that **Triton's inline asm** can process multiple elements at a time and could potentially be useful for fused bit-packing/unpacking and matmul operations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/index.html?">Welcome to Triton’s documentation! &mdash; Triton  documentation</a>: no description found</li><li><a href="https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html">triton.language.inline_asm_elementwise &mdash; Triton  documentation</a>: no description found</li><li><a href="https://github.com/srush/Triton-Puzzles">GitHub - srush/Triton-Puzzles: Puzzles for learning Triton</a>: Puzzles for learning Triton. Contribute to srush/Triton-Puzzles development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1262147685256007764)** (3 messages): 

> - `Bootstrap estimate of accuracy stdev`
> - `Optimized dataloader issue`
> - `Torch nightly broken` 


- **Bootstrap estimate of accuracy stdev for model evaluation**: A member suggested using a **bootstrap estimate** to calculate the **accuracy standard deviation** for model evaluation.
- **Switching back to torch dataloader resolves issue**: Another member reported that switching from an **optimized dataloader** back to the **torch version** resolved an unspecified issue they were experiencing.
- **Torch nightly build has broken functions**: A user mentioned that the **Torch nightly build** is broken, specifically showing an `AttributeError` due to **'torch.library'** missing the `custom_op` attribute.


  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1261411804693921852)** (2 messages): 

> - `LoQT method for efficient training`
> - `Brian Kernighan on The Practice of Programming` 


- **LoQT enables efficient model training on consumer GPUs**: The paper titled '[LoQT](https://arxiv.org/abs/2405.16528)' proposes a method for efficiently training quantized models using gradient-based tensor factorization, enabling models up to **7B parameters** to be trained on consumer-grade 24GB GPUs.
   - The method handles gradient updates to quantized weights differently and achieves comparable savings, suitable for both pretraining and fine-tuning.
- **Brian Kernighan discusses 'The Practice of Programming'**: In a [YouTube video](https://www.youtube.com/watch?v=_QQ7k5sn2-o), Dr. Brian Kernighan discusses his experience writing 'The Practice of Programming' in a special episode of Book Overflow.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.16528">LoQT: Low Rank Adapters for Quantized Training</a>: Training of large neural networks requires significant computational resources. Despite advances using low-rank adapters and quantization, pretraining of models such as LLMs on consumer hardware has n...</li><li><a href="https://www.youtube.com/watch?v=_QQ7k5sn2-o">Brian Kernighan Reflects on &quot;The Practice of Programming&quot;</a>: In this very special episode of Book Overflow, Dr. Brian Kernighan, the author of &quot;The Practice of Programming&quot; joins us to discuss his experience writing th...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1261415740133609613)** (23 messages🔥): 

> - `Accessing GPUs for Testing`
> - `Using Google Colab and nsight compute`
> - `CoreWeave vs Lambda Labs`
> - `Cloud GPU Services`
> - `Learning Triton` 


- **Best Ways to Access GPUs for Individual Testing**: A user asked for the best ways to get GPU access for testing, especially needing ncu, and multiple responses recommended **Google Colab** for its ease and free access (https://colab.research.google.com).
   - Discussion also mentioned **CoreWeave** and **LambdaLabs** as other options, noting CoreWeave is pricey and LambdaLabs is hard to get allocations.
- **Colab Supports nsight compute**: A member confirmed that **nsight compute** works on Google Colab, although spawning a window might be problematic.
   - The conversation also highlighted that **Google Cloud GPU** allows using things other than notebooks, although it is pricier compared to Colab.
- **Cloud GPU Services Compared**: Members compared different cloud services like **Google Cloud GPU** and **SageMaker** with on-demand services like **vast.ai**, noting the latter are generally cheaper.
   - For ease of working, it was suggested that **Google Colab** is less hassle compared to **Google Cloud Platform (GCP)**.
- **Triton Learning Resources**: A user asked for additional references to study **Triton**, besides the official [Triton website](https://triton-lang.org/main/index.html).
   - No specific additional resources were mentioned in the responses.
- **Challenge with Open Source Development on Cloud**: A member sought advice on doing open-source development using cloud tools due to having an older laptop with an **NVIDIA Quadro M4000** GPU.
   - They mentioned challenges in iterating and testing code changes in a cloud environment like **Google Colab** for **torchao** project development.



**Link mentioned**: <a href="https://triton-lang.org/main/index.html">Welcome to Triton’s documentation! &mdash; Triton  documentation</a>: no description found

  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1261879533712310392)** (34 messages🔥): 

> - `CUDA Core Processing`
> - `Register Limitations`
> - `Occupancy Calculation`
> - `Block Size Optimization`
> - `Kernel Parameterization` 


- **CUDA Core Processing Clarified**: A discussion revealed that a single CUDA core processes one thread at a time, meaning an A100 SM with 64 CUDA Cores can process **64 threads** simultaneously while having **2048 threads** assigned to it.
   - Another member explained the similarities to a CPU with threads swapping out when waiting, storing state in memory, but on GPUs, the total pool of registers limits this.
- **Register Limitations Impacting Threads**: Explanations on how **256 KiB of registers per SM** results in **32 registers per thread** when divided among 2048 threads were provided.
   - Using more registers in a kernel limits the total number of threads that can be executed, e.g., 1024 threads at 64 registers each.
- **Optimizing GPU Occupancy**: Occupancy of threads on a GPU is affected by the allocated shared memory and the number of threads, impacting latency hiding.
   - A balance is needed as too many threads can cause stalls due to insufficient memory, and conversely, too few threads can not adequately hide latency.
- **Block Size and Performance**: There was a discussion on choosing optimal block sizes for performance, using profiling and educated guesses.
   - An example was given with block reduction where a **128 block size** was found to perform best contrary to initial expectations, when profiled against 1024, 512, and 256.
- **Kernel Parameterization for Optimization**: Parameterizing implementations across different values allows running benchmarks to find the best configuration, essential for optimizing GPU performance.
   - Templating kernels with varied sizes, as seen in FAv2 sources, allows for optimal fitting to different matrix sizes due to STATIC_SWITCH and BOOL_SWITCH.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1261504796356116632)** (2 messages): 

> - `FSDP support for low-bit optimization`
> - `Developer guide for integration` 


- **Implementing FSDP Support for Low-Bit Optimization**: A member is working on implementing **FSDP** support for low-bit optimization but isn't addressing collective ops for optimization state subclass yet.
   - They suggested that a developer guide would help in getting interest from developers as lack of integration guidance might lead to abandonment.
- **Review of FSDP Implementation**: Another member agreed to review the **FSDP** implementation next week.
   - *Looking forward to diving into it next week*.


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1261684089568891021)** (46 messages🔥): 

> - `Switching to cudaMallocManaged`
> - `llm.cpp updates`
> - `WebGPU insights`
> - `gpt3v1 by karpathy`
> - `GPT-3 model interpolation` 


- **Switch to cudaMallocManaged for memory efficiency**: **Eriks.0595** suggested switching from cudaMalloc to cudaMallocManaged to support devices with insufficient memory and ensure non-intrusive changes without slowing down existing functionalities.
   - *Eriks.0595* emphasized the importance of this feature for smaller GPU integration.
- **Major updates in llm.cpp over the past 4 months**: **Kashimoo** asked for updates on llm.cpp after a 4-month hiatus, prompting **Eriks.0595** to explain that almost everything has changed.
- **WebGPU's broader applications**: **Akakak1337** expressed surprise at WebGPU's non-Web usages and planned to watch a linked talk for more insights.
- **Mup run insights and performance**: In discussing a **mup run**, **akakak1337** provided performance details, with figures like **0.495917** accuracy on HellaSwag.
   - *Falconsfly* noted concerns about token/sec performance and loss of precision.
- **Merged GPT-3 models to master branch**: **Eriks.0595** inquired about extending their model series to GPT-3 models, leading to a discussion on model interpolation.
   - **Akakak1337** confirmed the merge of GPT-3 models to the master branch and discussed the challenges of matching non-monotonic head sizes and depths.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Yuchenj_UW/status/1812893615372575180">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: After training the largest GPT-2 (1.5B), I decided to go &#34;deeper&#34; and feel the scaling law by training a 2.7B model with @karpathy&#39;s llm.c 📈  Scaling the model was straightforward, primar...</li><li><a href="https://github.com/karpathy/llm.c/pull/688">feature/gpt3v1 by karpathy · Pull Request #688 · karpathy/llm.c</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/)** (1 messages): 

vkaul11: Hi
  

---


### **CUDA MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1262129045563506698)** (25 messages🔥): 

> - `WebGPU resources and support`
> - `Running LLMs in the browser with Transformers.js`
> - `Building and troubleshooting Dawn on Windows`
> - `GPU buffers and performance` 


- **Explore WebGPU with new resources**: Members shared various resources for learning WebGPU, including [WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html) which introduces compute shaders and optimization steps.
   - Discussions highlighted that browser support is mostly in Chrome, with Firefox support being limited by default, and Safari lagging behind.
- **Try Transformers.js for browser-based LLMs**: A member mentioned [Transformers.js](https://huggingface.co/docs/transformers.js/index) for running state-of-the-art machine learning tasks directly in the browser using ONNX Runtime.
   - They noted it supports multiple tasks including text classification, question answering, and image classification, although they haven't experimented much with it.
- **Troubleshoot Dawn build issues**: Multiple messages discussed troubleshooting the Dawn build on Windows, where the release build behaved unexpectedly, but the debug build worked correctly.
   - Rebuilding strategies included using Google's distribution with CMake, and considering using shared libraries instead of FetchContent for improved stability.
- **Understand WebGPU buffer limitations**: A member explained that the WebGPU environment in browsers has limitations such as 16 KB shared memory and 128 MB buffers, which are minimums.
   - Another member questioned if GPU offload for small data sizes is a performance boost compared to CPU AVX instructions due to these limitations.
- **Share experiences and improvements**: Members shared experiences with setting up and using WebGPU, discussing various challenges and potential improvements for future development.
   - Feedback included suggestions for simplifying shader vs. kernel nomenclature and more flexible handling of structured parameters.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/transformers.js/index">Transformers.js</a>: no description found</li><li><a href="https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html">WebGPU Compute Shader Basics</a>: How to use compute shaders in WebGPU</li><li><a href="https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders-histogram.html">WebGPU Compute Shaders - Image Histogram</a>: Efficiently compute an image histogram.</li><li><a href="https://github.com/AnswerDotAI/gpu.cpp/blob/main/examples/webgpu_from_scratch/run.cpp">gpu.cpp/examples/webgpu_from_scratch/run.cpp at main · AnswerDotAI/gpu.cpp</a>: A lightweight library for portable low-level GPU computation using WebGPU.  - AnswerDotAI/gpu.cpp
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1261434252491489380)** (141 messages🔥🔥): 

> - `OpenArena GitHub project`
> - `Cohere event link confusion`
> - `LlamaIndex KG deduplication`
> - `Karpathy on AI training costs`
> - `Account support issues` 


- **OpenArena GitHub Project Unveiled**: A member shared their project [OpenArena](https://github.com/syv-ai/OpenArena) which aims to pit LLMs against each other for better dataset quality.
- **Cohere Event Link Confusion**: Members discussed confusion over a **Cohere event** link, with some unable to access the session and others providing the correct zoom link for a **guest speaker session** on diffusion models generating spectrograms.
- **LlamaIndex KG Node Deduplication Explained**: A member shared a [YouTube video](https://youtu.be/vMz0icWZd5A) explaining how **LlamaIndex handles deduplication of nodes** in its knowledge graph.
- **AI Training Costs Plummet**: A member highlighted [Karpathy's detailed discussion](https://x.com/karpathy/status/1811467135279104217) on the drastic reduction in costs to train AI models like GPT-2 over the last 5 years.
- **Cohere Account Support Issues**: A member reported issues with their **Cohere account** disappearing after an organizational invite mishap, receiving guidance from support to submit a ticket for resolution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/karpathy/status/1811467135279104217">Tweet from Andrej Karpathy (@karpathy)</a>: In 2019, OpenAI announced GPT-2 with this post: https://openai.com/index/better-language-models/  Today (~5 years later) you can train your own for ~$672, running on one 8XH100 GPU node for 24 hours. ...</li><li><a href="https://cohere.com/events/cohere-for-ai-guest-speaker-ziyang-chen-2024">Cohere For AI - Guest Speaker: Ziyang Chen, PhD Student</a>: Images that Sound: Composing Images and Sounds on a Single Canvas</li><li><a href="https://youtu.be/vMz0icWZd5A">LlamaIndex KG | Deduplication of nodes.</a>: In this recording, I explain in details how LlamaIndex is doing the deduplication of the nodes after creating the knowledge graphcode:https://github.com/raji...</li><li><a href="https://cohere.com/">Cohere | The leading AI platform for enterprise</a>: Cohere provides industry-leading large language models (LLMs) and RAG capabilities tailored to meet the needs of enterprise use cases that solve real-world problems.</li><li><a href="https://docs.cohere.com/docs/tool-use">Tool Use with Cohere's Models - Cohere Docs</a>: no description found</li><li><a href="https://tenor.com/view/inside-out-joy-hi-hey-hello-gif-13317321031557907374">Inside Out Joy GIF - Inside Out Joy Hi - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/syv-ai/OpenArena">GitHub - syv-ai/OpenArena</a>: Contribute to syv-ai/OpenArena development by creating an account on GitHub.</li><li><a href="https://umich.zoom.us/j/8022650618?pwd=V0VvYnAyQVBlNnIrUktGNyt6WFE1dz09">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1261474660353179699)** (26 messages🔥): 

> - `NPM module for Cohere`
> - `r/localllama bot using Langchain and Cohere`
> - `Using JSON from Reddit`
> - `Mult AI subreddit update` 


- **NPM module for Cohere released**: [A new update](https://github.com/samestrin/llm-interface) to the NPM module now includes support for **Cohere**, enhancing its ease of interaction with various LLM providers.
   - The repository image and NPM installation details were shared, showing seamless integration with multiple AI platforms.
- **r/localllama bot built using Langchain and Cohere**: A new bot has been created to fetch and summarize top posts from **r/localllama** into news style posts for Discord channels using **Langchain** and **Cohere Command-R-Plus**.
   - The bot's code was shared and it sparked excitement among members who found it incredibly useful.
- **Extract post data as JSON from Reddit**: Members discussed a method to extract information from Reddit posts by appending `.json` to their URLs for top posts on **r/localllama**.
   - "Your Settings Are Probably Hurting Your Model" post was highlighted as an example, emphasizing the impact of sampler settings on model performance.
- **Mult AI subreddit update for news bot**: The bot was updated to support multiple AI subreddits and improve story sorting mechanisms.
   - Plans were shared to enable Cohere to categorize and direct news stories to appropriate channels based on their topics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/samestrin/llm-interface">GitHub - samestrin/llm-interface: A simple NPM interface for seamlessly interacting with 36 Large Language Model (LLM) providers, including OpenAI, Anthropic, Google Gemini, Cohere, Hugging Face Inference, NVIDIA AI, Mistral AI, AI21 Studio, LLaMA.CPP, and Ollama, and hundreds of models.</a>: A simple NPM interface for seamlessly interacting with 36 Large Language Model (LLM) providers, including OpenAI, Anthropic, Google Gemini, Cohere, Hugging Face Inference, NVIDIA AI, Mistral AI, AI...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17vonjo/your_settings_are_probably_hurting_your_model_why/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17vonjo/your_settings_are_probabl">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17vonj">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1261410626748547214)** (70 messages🔥🔥): 

> - `AI Meetups in London`
> - `OpenAI Collaborations`
> - `Model Benchmarking`
> - `Time Consideration in Models`
> - `Machine Learning Conferences` 


- **AI meetups in London lack depth**: Members discussed that AI meetups in London often have superficial discussions and are infrequent, with a recommendation to check out UCL's & Imperial's seminars and invited talks for deeper knowledge.
   - It was noted that conferences like ICML and ICLR usually offer deeper conversations, particularly in field-specific meetups and 1-on-1 sessions with researchers.
- **Arrakis project for fast iteration in mechinterp**: A user requested feedback on [Arrakis](https://github.com/yash-srivastava19/arrakis), a library designed for conducting, tracking, and visualizing mechanistic interpretability experiments with integrated tools like tuned-lens.
   - The project aims to improve research efficiency and utility within the community.
- **OpenLLMLeaderboard benchmark data availability**: Questions rose regarding the availability of test sets for the new OpenLLMLeaderboard on Hugging Face, specifically if parts of the datasets were unreleased.
   - It was clarified that [HuggingFace](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#reproducibility) provides reproducibility by allowing downloads of all public data, ensuring no hidden elements.
- **Time relevance in LLM training questioned**: A user expressed interest in how time and data freshness can be utilized in modeling relevance, noting current methods of passing timestamps to LLMs are ineffective.
   - Suggestions included examining literature on specific method papers, datasets, and benchmarks that deal with time-relevant data for better model training.
- **Interest in large context windows for AI applications**: A community advocate is seeking recommendations for hosted models with huge context windows (1M tokens) for AI-assisted human rights applications.
   - They shared their current work and context on a project and [link to discourse](https://community.openai.com/t/inception-based-design-for-the-ai-assisted-creation-of-a-human-rights-application/863669), requesting any useful insights or resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://livebench.ai/">LiveBench</a>: no description found</li><li><a href="https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about#reproducibility">About</a>: no description found</li><li><a href="https://github.com/yash-srivastava19/arrakis">GitHub - yash-srivastava19/arrakis: Arrakis is a library to conduct, track and visualize mechanistic interpretability experiments.</a>: Arrakis is a library to conduct, track and visualize mechanistic interpretability experiments. - yash-srivastava19/arrakis</li><li><a href="https://community.openai.com/t/inception-based-design-for-the-ai-assisted-creation-of-a-human-rights-application/863669">Inception-based design for the AI-assisted creation of a written human rights complaint</a>: I also used GitHub CoPilot within vsCode IDE as well as ChatGpt4o in order to transcribe screenshots containing text message content.</li><li><a href="https://www.meetup.com/london-machine-learning-meetup/events/)">alert--small</a>: no description found</li><li><a href="https://www.youtube.com/@LondonMachineLearningMeetup/videos))">London Machine Learning Meetup</a>: The London Machine Learning Meetup is the largest machine learning community in Europe. Previous speakers include Juergen Schmidhuber, David Silver, Yoshua Bengio and Andrej Karpathy.   Come to our ne...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1261416634988494859)** (61 messages🔥🔥): 

> - `Hermes 2 Performance`
> - `RAG Systems with LangChain`
> - `Compute Thresholds Governance`
> - `RISE in LLMs`
> - `Model Compression and Accuracy` 


- **Paper Investigates Synthetic Data for Math Reasoning**: A new [paper](https://arxiv.org/abs/2406.14532) explores the effectiveness of finetuning LLMs with model-generated synthetic data, finding double the efficiency when models fine-tune on self-generated data after initial finetuning.
   - Concerns were raised about model-generated positives amplifying spurious correlations, leading to flat or inverse scaling trends.
- **Discussion on Compute Thresholds Governance**: An [essay](https://arxiv.org/abs/2407.05694) delves into how compute thresholds could impact AI safety and the risk profile of models by regulating compute usage.
   - The community discussed the idea that regulating massive training jobs could prevent the monopolization of compute resources by a few entities.
- **LangChain for Reliable RAG Systems**: A member shared a project on GitHub for creating reliable RAG (Retrieval-Augmented Generation) systems using LangChain.
   - The [repository](https://github.com/eericheva/langchain_rag) provides detailed scripts and tutorials to help users implement RAG systems from scratch.
- **RISE Enables Self-Improvement in LLMs**: A new paper presents [RISE](https://openreview.net/forum?id=qDXdmdBLhR), a finetuning approach enabling LLMs to iteratively improve their responses over multiple turns.
   - The method focuses on recursive introspection, allowing models to learn from previous unsuccessful attempts and improve sequentially.
- **Model Compression Techniques and Quality Flips**: Research analyzed how [quantization techniques](https://arxiv.org/abs/2407.09141) for compressing large models can lead to 'flips' in answers, changing from correct to incorrect even if overall accuracy appears unchanged.
   - The discussion highlighted that such flips signify a more complex degradation of model quality, and further qualitative and quantitative evaluations are necessary.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.19384">The Remarkable Robustness of LLMs: Stages of Inference?</a>: We demonstrate and investigate the remarkable robustness of Large Language Models by deleting and swapping adjacent layers. We find that deleting and swapping interventions retain 72-95\% of the origi...</li><li><a href="https://arxiv.org/abs/2407.09141">Accuracy is Not All You Need</a>: When Large Language Models (LLMs) are compressed using techniques such as quantization, the predominant way to demonstrate the validity of such techniques is by measuring the model&#39;s accuracy on v...</li><li><a href="https://arxiv.org/abs/2312.01203">Harnessing Discrete Representations For Continual Reinforcement Learning</a>: Reinforcement learning (RL) agents make decisions using nothing but observations from the environment, and consequently, heavily rely on the representations of those observations. Though some recent b...</li><li><a href="https://arxiv.org/abs/2208.07339">LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale</a>: Large language models have been widely adopted but require significant GPU memory for inference. We develop a procedure for Int8 matrix multiplication for feed-forward and attention projection layers ...</li><li><a href="https://arxiv.org/abs/2407.05694">On the Limitations of Compute Thresholds as a Governance Strategy</a>: At face value, this essay is about understanding a fairly esoteric governance tool called compute thresholds. However, in order to grapple with whether these thresholds will achieve anything, we must ...</li><li><a href="https://arxiv.org/abs/2405.20835">Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs</a>: Post-Training Quantization (PTQ) enhances the efficiency of Large Language Models (LLMs) by enabling faster operation and compatibility with more accessible hardware through reduced memory usage, at t...</li><li><a href="https://arxiv.org/abs/2406.14532">RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold</a>: Training on model-generated synthetic data is a promising approach for finetuning LLMs, but it remains unclear when it helps or hurts. In this paper, we investigate this question for math reasoning vi...</li><li><a href="https://arxiv.org/abs/2401.12181">Universal Neurons in GPT2 Language Models</a>: A basic question within the emerging field of mechanistic interpretability is the degree to which neural networks learn the same underlying mechanisms. In other words, are neural mechanisms universal ...</li><li><a href="https://dynamicfieldtheory.org/">Home | Dynamic field theory</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/5076">KL-divergence by ikawrakow · Pull Request #5076 · ggerganov/llama.cpp</a>: There have been several discussions about the potential value of being able to compute KL-divergence as another quantization accuracy test. There is the Python script that @Ttl provided in PR #4739...</li><li><a href="https://github.com/eericheva/langchain_rag">GitHub - eericheva/langchain_rag</a>: Contribute to eericheva/langchain_rag development by creating an account on GitHub.</li><li><a href="https://github.com/eericheva/langchain_rag/tree/main#item-one)">GitHub - eericheva/langchain_rag</a>: Contribute to eericheva/langchain_rag development by creating an account on GitHub.</li><li><a href="https://openreview.net/forum?id=qDXdmdBLhR">Recursive Introspection: Teaching Foundation Model Agents How to...</a>: A central piece in enabling intelligent agentic behavior in foundation models is to make them capable of introspecting upon their behavior, to reason and correct their mistakes. Even strong...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 messages): 

wabi.sabi.1: Very interesting, thanks
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1261450950946852935)** (13 messages🔥): 

> - `lm-eval Python API`
> - `PRAUC metric for lm-eval`
> - `Quantization flips research`
> - `Distributed lm_evaluation`
> - `Custom functions in task YAML` 


- **Use lm-eval API for Transformer Lens Model**: A member inquired about using the lm-eval Python API with a custom Transformer-lens model and was advised to subclass one of the `lm_eval.api.model.LM` or similar classes for compatibility.
   - They thanked the advisor for providing a helpful [link to the documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage).
- **Calculating PRAUC metric in lm-eval**: A user asked how to implement the PRAUC metric for imbalanced test data using lm-eval, requiring positive probability outputs.
   - The discussion didn't provide a specific answer, suggesting the member might need further assistance.
- **Quantization Flips Study Released**: A member shared their new paper on [quantization flips](https://arxiv.org/abs/2407.09141), noting how compressed models can behave differently from baseline models despite matching benchmark accuracy.
   - The research, which utilized the Harness, highlights significant behavioral changes in compressed models even when quantitative metrics are close.
- **Evaluating Models in Distributed Setup**: A member sought advice on implementing the evaluate() method for distributed evaluation within lm-harness and loading pruned models into HFLM.
   - While specific solutions weren't provided, the query remains open for suggestions and examples from the community.
- **Custom Functions in lm-eval YAML**: A question was raised regarding the arguments passed to a custom `!function` defined in a task YAML.
   - The discussion did not yet yield detailed guidance on handling these custom functions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.09141">Accuracy is Not All You Need</a>: When Large Language Models (LLMs) are compressed using techniques such as quantization, the predominant way to demonstrate the validity of such techniques is by measuring the model&#39;s accuracy on v...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage">lm-evaluation-harness/docs/interface.md at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

bobby_mcbobface: Thanks Ryan! Just wanted to make sure I wasn’t going down an abandoned path
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1261476977269936140)** (104 messages🔥🔥): 

> - `MonoNN Compiler`
> - `tinygrad Kernel Overhead`
> - `MLX vs tinygrad Performance`
> - `Shape Changing Bitcasts`
> - `Monday Meeting Highlights` 


- **MonoNN Compiler Offers Optimized GPU Utilization**: A new machine learning optimizing compiler, **MonoNN**, addresses inefficiencies in traditional kernel-by-kernel execution schemes by accommodating an entire neural network into a single kernel. The [paper presentation](https://www.usenix.org/conference/osdi24/presentation/zhuang) and [source code](https://github.com/AlibabaResearch/mononn) were discussed in the community.
- **Debate on tinygrad Kernel Overhead**: Community members discussed the significant **3-4us** kernel overhead per kernel on AMD GPUs, based on experimental results.
- **MLX Outperforms tinygrad in Speed and Accuracy**: **MLX** was found to be faster and achieve higher accuracy compared to **tinygrad**, especially noted in the beautiful_MNIST benchmark.
- **Challenges with Shape Changing Bitcasts**: Implementing support for shape-changing bitcasts in **tinygrad** is progressing, though it faces issues primarily on GPU devices.
- **Highlights from Monday Meeting**: The meeting covered updates on **tinybox** and new components like the **lowerer** and **HWComandQueue device**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ml-explore.github.io/mlx/build/html/usage/compile.html">Compilation &#8212; MLX 0.16.0 documentation</a>: no description found</li><li><a href="https://www.usenix.org/conference/osdi24/presentation/zhuang">MonoNN: Enabling a New Monolithic Optimization Space for Neural Network Inference Tasks on Modern GPU-Centric Architectures | USENIX</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/commit/8940530290b04048074be1deadd24e5d91d67d28">add mlx beautiful_mnist example · tinygrad/tinygrad@8940530</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/blob/ae4cb7994e73f35b6b467327d194394cdf52b99d/tinygrad/device.py#L207),">tinygrad/tinygrad/device.py at ae4cb7994e73f35b6b467327d194394cdf52b99d · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://x.com/__tinygrad__/status/1811598734045991021)">Tweet from the tiny corp (@__tinygrad__)</a>: This is the kernels from a CIFAR training step. On the right, tinygrad now shows which operations led to the creation of the kernel. One step closer to great error messages!
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1261513253171761304)** (27 messages🔥): 

> - `count_include_pad in avg_pool2d`
> - `Tensor indexing and gather function`
> - `Improving Tinygrad documentation`
> - `Splitting Tensors based on ratios` 


- **Include pad option in avg_pool2d requested**: **Stable Diffusion** training eval requires the `count_include_pad=False` option in `avg_pool2d` like PyTorch has, and members discussed potential implementation approaches.
   - One member suggested upstreaming a method using `(pool -> sum) / (ones_like -> pool -> sum)` if **MLPerf** requires it.
- **Clarification on tensor indexing**: Members clarified the differences between `probas[:, Y_train]` and `probas[Tensor.arange(len(logits)), Y_train]` and discussed why masking instead of indexing makes operations faster in Tinygrad.
   - A member provided a useful [link to the quickstart guide](https://docs.tinygrad.org/quickstart/#training), which explains the implementations.
- **Fixing bugs in gather function**: A bug was identified in Tinygrad's `gather` function related to negative index handling, causing incorrect behavior.
   - The issue was fixed by correcting the order of a function call, and the fix will be included in an upcoming PR.
- **Separate pull requests for different improvements**: Members agreed that submitting separate PRs for new tensor functions, model implementations, and function expansions is preferred for ease of review.
   - A member implemented `interpolate` for FID, which worked but exposed a bug that was promptly addressed.
- **Documentation for testing code blocks**: Members discussed executing code blocks from documentation to ensure they work correctly.
   - A helpful [link to serving Tinygrad docs](https://github.com/tinygrad/tinygrad/blob/master/serve_docs.sh) was shared for guidance.



**Link mentioned**: <a href="https://docs.tinygrad.org/quickstart/#training)">Quickstart - tinygrad docs</a>: no description found

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1261459945518268499)** (43 messages🔥): 

> - `Open LLM Leaderboard V2`
> - `Solving Reddit Link Hallucination`
> - `New Models in LMSys Arena`
> - `Cursor's Composer Feature`
> - `SpreadsheetLLM by Microsoft` 


- **Open LLM Leaderboard V2 Episode Released**: A user announced a new Latent Space episode focused on the **Open LLM Leaderboard V2**.
   - Another user expressed excitement about the new episode with a 'yessir'.
- **Hypotheses on SmolAI Solving Reddit Link Hallucination**: Members shared theories on how **SmolAI** resolved the issue of Reddit link hallucination, including **pre-check and post-proc** methods.
   - A member mentioned applying a similar pre-check method for selecting IDs to ensure accuracy.
- **Mystery Behind New Models in LMSys Arena**: Questions arose about who might be behind the new models in the **LMSys arena** with linked strong opinions and discussions on the topic.
   - A member heard rumors about **Command R+ jailbreaks** working on one of the new models.
- **Cursor's Composer Feature Excitement**: There's considerable interest in **Cursor's** new Composer feature, with users discussing its **beta release** and comparing it with other UX options.
   - Members shared their thoughts on the accessibility and affordability of the feature, indicating positive initial impressions despite subscription concerns.
- **Microsoft Introduces SpreadsheetLLM**: Microsoft revealed **SpreadsheetLLM** which aims to optimize LLMs' capabilities for handling spreadsheets using a novel **SheetCompressor** encoding framework.
   - Members expressed interest in the technique's potential, as it modifies input data to work better with various LLMs without requiring fine-tuning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Smol_AI/status/1811957074840158255">Tweet from AI News by Smol AI (@Smol_AI)</a>: [12 July 2024]  https://buttondown.email/ainews/archive/ainews-we-solved-hallucinations/  We solved Hallucinations!</li><li><a href="https://x.com/apples_jimmy/status/1812029979888439525?s=61">Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)</a>: Seems like there’s a new model in Lmsys arena.</li><li><a href="https://fortune.com/2024/07/12/lattice-ai-workers-sam-altman-brother-jack-sarah-franklin/">$3 billion Lattice &#x27;made history&#x27; being the first to give AI &#x27;workers&#x27; rights</a>: Lattice notably laid off 100 human workers last year.</li><li><a href="https://arxiv.org/html/2407.09025v1">SpreadsheetLLM: Encoding Spreadsheets for Large Language Models</a>: no description found</li><li><a href="https://x.com/shaoruu/status/1812412514350858634">Tweet from ian (@shaoruu)</a>: composer is out for testing @cursor_ai  and here&#39;s me making a typing test with it in 6 minutes (6x sped up):</li><li><a href="https://x.com/teortaxesTex/status/1812226271457296395">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: STRONG OPINIONS VERY LOOSELY HELD  Quoting Nic (@nicdunz)   @kalomaze @teortaxesTex bro 😭
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new podcast drop! https://x.com/swyx/status/1811898574416019562
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1261411650687209588)** (86 messages🔥🔥): 

> - `Memorable Acronyms`
> - `More demos and examples`
> - `Evaluation techniques`
> - `Logprob usages`
> - `State management` 


- **Memorable Acronym: 3E**: A member suggested using a more memorable acronym like **Extract, Evaluate, Extend/Expand (3E)**.
- **Demand for More Demos and Examples**: Multiple members emphasized the need for more demos and examples in their discussions, particularly related to technical implementations.
- **Exploring Evaluation Techniques: Logprob and GPTscore**: Members discussed different evaluation techniques like **logprob**, **GPTscore**, and hyperparameter optimization tools like [prompt-hyperopt](https://github.com/Mavenoid/prompt-hyperopt).
   - A paper titled [Simple approach for contextual hallucinations](https://arxiv.org/html/2407.07071v1) was mentioned in relation to this.
- **State Management Tools Comparison**: State management styles were compared, with a focus on **ReAct framework**, **Langgraph**, and [XState](https://github.com/statelyai/xstate).
   - *Langgraph* was noted for better handling of graph-state memory for each step through the node.
- **Upcoming AI in Action Talks**: Next week, **VikParuchuri** will present on converting PDF to Markdown using tools like [marker](https://github.com/VikParuchuri) and surya.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://zod.dev/">TypeScript-first schema validation with static type inference</a>: TypeScript-first schema validation with static type inference</li><li><a href="https://huggingface.co/nisten/bakllava-14b-2xMoE-alpha-build">nisten/bakllava-14b-2xMoE-alpha-build · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/html/2407.07071v1">Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps</a>: no description found</li><li><a href="https://arxiv.org/abs/2310.14566">HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models</a>: We introduce HallusionBench, a comprehensive benchmark designed for the evaluation of image-context reasoning. This benchmark presents significant challenges to advanced large visual-language models (...</li><li><a href="https://github.com/truera/trulens">GitHub - truera/trulens: Evaluation and Tracking for LLM Experiments</a>: Evaluation and Tracking for LLM Experiments. Contribute to truera/trulens development by creating an account on GitHub.</li><li><a href="https://github.com/tianyi-lab/HallusionBench">GitHub - tianyi-lab/HallusionBench: [CVPR&#39;24] HallusionBench: You See What You Think? Or You Think What You See? An Image-Context Reasoning Benchmark Challenging for GPT-4V(ision), LLaVA-1.5, and Other Multi-modality Models</a>: [CVPR&amp;#39;24] HallusionBench: You See What You Think? Or You Think What You See? An Image-Context Reasoning Benchmark Challenging for GPT-4V(ision), LLaVA-1.5, and Other Multi-modality Models - ti...</li><li><a href="https://github.com/openvinotoolkit/anomalib">GitHub - openvinotoolkit/anomalib: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference.</a>: An anomaly detection library comprising state-of-the-art algorithms and features such as experiment management, hyper-parameter optimization, and edge inference. - openvinotoolkit/anomalib</li><li><a href="https://github.com/Mavenoid/prompt-hyperopt">GitHub - Mavenoid/prompt-hyperopt: Improve prompts for e.g. GPT3 and GPT-J using templates and hyperparameter optimization.</a>: Improve prompts for e.g. GPT3 and GPT-J using templates and hyperparameter optimization. - Mavenoid/prompt-hyperopt</li><li><a href="https://github.com/chand1012/git2gpt">GitHub - chand1012/git2gpt: Convert a Git repo into a ChatGPT prompt!</a>: Convert a Git repo into a ChatGPT prompt! Contribute to chand1012/git2gpt development by creating an account on GitHub.</li><li><a href="https://github.com/VikParuchuri">VikParuchuri - Overview</a>: VikParuchuri has 90 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/statelyai/xstate">GitHub - statelyai/xstate: Actor-based state management &amp; orchestration for complex app logic.</a>: Actor-based state management &amp; orchestration for complex app logic. - statelyai/xstate</li><li><a href="https://github.com/seanchatmangpt/dspygen">GitHub - seanchatmangpt/dspygen: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama.</a>: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama. - seanchatmangpt/dspygen</li><li><a href="https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness">GitHub - jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness: Awesome-LLM-Robustness: a curated list of Uncertainty, Reliability and Robustness in Large Language Models</a>: Awesome-LLM-Robustness: a curated list of Uncertainty, Reliability and Robustness in Large Language Models - jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown,@ UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://github.com/EGjoni/DRUGS">GitHub - EGjoni/DRUGS: Stop messing around with finicky sampling parameters and just use DRµGS!</a>: Stop messing around with finicky sampling parameters and just use DRµGS! - EGjoni/DRUGS</li><li><a href="https://github.com/elder-plinius/AutoTemp">GitHub - elder-plinius/AutoTemp: A trial-and-error approach to temperature opimization for LLMs. Runs the same prompt at many temperatures and selects the best output automatically.</a>: A trial-and-error approach to temperature opimization for LLMs. Runs the same prompt at many temperatures and selects the best output automatically. - elder-plinius/AutoTemp
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1261408034211762338)** (86 messages🔥🔥): 

> - `OpenArena Project`
> - `ORPO Training`
> - `Anthropic Prompt Integration`
> - `RAG Model Dataset`
> - `Weighting Conversation Data` 


- **OpenArena Project Goes 100% Open Source**: User le_mess is working on a [100% open source local version](https://github.com/syv-ai/OpenArena) of a dataset creation tool originally meant for OpenRouter but now using Ollama.
   - The project aims to provide a more flexible and open environment for dataset creation for various models.
- **Challenges in ORPO Training Memory Usage**: User xzuyn raised concerns about ORPO training's memory usage, stating that it spikes and eventually results in OOM, even with max sequence set to 2k.
   - Discussion revealed a lack of messages about dropping long sequences post-tokenization, contributing to erratic memory spikes.
- **Anthropic Prompt Format for Axolotl**: Kalomaze discussed integrating the [official Claude/Anthropic prompt format](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/chat_templates.py) into Axolotl, using special tokens for system, human, and assistant turns.
   - There were concerns about the readability and generalization of special tokens; however, the existing SOTA model's practices were considered acceptable.
- **RAG Model Dataset Scraping Concerns**: User nafnlaus00 raised security concerns about using Chromium to render sites requiring JavaScript, such as Quora, for creating a RAG model dataset.
   - Le_mess suggested troubleshooting headers/params issues and considering services like firecrawl or the Jina API for safer scraping.
- **Proposing Weighted Training Data**: Tostino suggested implementing a system for weighting different parts of conversation data in both pretraining and SFT, allowing negative weights to teach models to avoid certain tokens.
   - This could enable optimization loops where less understood sections or 'bad paths' are weighted differently to improve model outcomes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/utils/chat_templates.py">axolotl/src/axolotl/utils/chat_templates.py at main · axolotl-ai-cloud/axolotl</a>: Go ahead and axolotl questions. Contribute to axolotl-ai-cloud/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/syv-ai/OpenArena">GitHub - syv-ai/OpenArena</a>: Contribute to syv-ai/OpenArena development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1262176377088507965)** (5 messages): 

> - `Chat template dataset type`
> - `PR review process`
> - `Configuration flexibility`
> - `Training labels configuration`
> - `Handling token offsets` 


- **PR for Chat Template Dataset Soon**: User announced the upcoming **PR** for a new chat template dataset type offering flexibility on training sections.
   - This includes selecting roles to train on, configuring `train_on_eos`, and handling specific training sections within the dataset.
- **Concerns Over Stuck PR Reviews**: A member raised concerns about **PR reviews** being stuck, mentioning specific PRs from themselves and another user.
   - "Are PR reviews getting stuck?" user asked, pointing to [their PR](https://github.com/axolotl-ai-cloud/axolotl/pull/1725) and [another one](https://github.com/axolotl-ai-cloud/axolotl/pull/1733).


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1261844949897445406)** (6 messages): 

> - `Eric's Spectrum Work`
> - `Quantizing Dolphin Vision 72b`
> - `4-bit Model on 96GB Mac Pro` 


- **Eric's Spectrum Exploration Gains Attention**: A member mentioned that Eric has been working on a spectrum, which caught another member's interest who is currently reviewing the related paper.
   - They noted that the paper seems *very interesting* on a first pass.
- **Quantizing Dolphin Vision 72b Considerations**: A member inquired about the feasibility of quantizing **Dolphin Vision 72b** to minimize VRAM usage.
   - Another member responded that **4-bit quantization** should still work well and suggested exploring **lower quants with gguf or exl2**.
- **Running 4-bit Model on 96GB Mac Pro**: A member shared that **4-bit quantization will fit** on the 96GB of integrated RAM available on a **mac pro with maxed out RAM**.
   - They mentioned running **inference** for it on their current setup.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/)** (1 messages): 

n_tt_n: i love capybara, have gotten awesome results with it
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1261461277901848689)** (18 messages🔥): 

> - `Pushing Model to Hub after LoRA Merge`
> - `Vicuna Chat Template Support`
> - `Config Options for Vicuna Template` 


- **Pushing Model to Hub after LoRA Merge**: A member asked how to push a model to the hub after merging LoRA into the base, suggesting using the `HfApi`'s `upload_folder` method.
   - Another member suggested a simpler approach using the `huggingface-cli upload` command: `huggingface-cli upload wasamkiriua/model-name .`.
- **Vicuna Chat Template Confirmed**: It was confirmed that Axolotl supports the vicuna chat template, which can be specified with the `conversation` option set to `vicuna_v1.1` in the configuration file.
   - The support allows handling conversations involving human and GPT interactions, following the vicuna template format.
- **Valid Options for Chat Template Config Flag**: The `chat_template` config flag cannot be directly set to `vicuna`; valid options include `alpaca`, `chatml`, `inst`, `gemma`, `cohere`, `llama3`, and `phi_3`.
   - Members agreed to omit the `chat_template` flag and set it manually afterwards if working with Vicuna-based models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/docs/dataset-formats/conversation.qmd#L1L64)">axolotl/docs/dataset-formats/conversation.qmd at main · axolotl-ai-cloud/axolotl</a>: Go ahead and axolotl questions. Contribute to axolotl-ai-cloud/axolotl development by creating an account on GitHub.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=2e654e43-06c9-4e97-88bd-5fd61c91a7c6)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=3b3a1901-727b-4dbc-9426-dcf10d932051)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1261804408430399498)** (9 messages🔥): 

> - `GPTs Agents`
> - `OpenAI Platform's sidebars`
> - `Custom chat templates for axolotl training`
> - `Axolotl training setup`
> - `Jinja format for templates` 


- **GPTs Agents cannot learn after initial training**: A member shared a concern about GPTs agents not learning from additional information provided after their initial training.
   - Another member clarified that uploaded files are saved as "knowledge" files for the agent to reference, but **they do not continuously update the agent's base knowledge**.
- **OpenAI Platform's sidebars changed**: Members discussed changes in the sidebars of platform.openai.com, noting that two icons (threads and messages) disappeared.
   - They speculated on potential reasons and impacts of this change on user navigation.
- **Setting up custom chat templates for axolotl training**: A member requested help converting custom chat templates for axolotl training, providing specific configurations they wanted to achieve.
   - Another member provided step-by-step guidance, including Jinja template formats and YAML examples for configuring Axolotl.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=71f0f5e0-3659-41d9-b28e-780759c1d47d)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=6f51d3b7-a886-472f-ae22-45f2e0b54aeb)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1261442258171990117)** (31 messages🔥): 

> - `OpenAI working on Strawberry`
> - `New models in LMSYS arena`
> - `Stealth releases of models in LMSYS` 


- **OpenAI's Strawberry to Enhance Reasoning**: OpenAI is working on new reasoning technology called **Strawberry**, with similarities to Stanford's **Self-Taught Reasoner** or **STaR** developed in 2022, as reported by [Reuters](https://www.reuters.com/technology/artificial-intelligence/openai-working-new-reasoning-technology-under-code-name-strawberry-2024-07-12/).
   - Discussion reveals insiders believe it resembles **STaR**, a method from Stanford.
- **LMSYS Brings New Models into the Arena**: [Jimmy Apples](https://x.com/apples_jimmy/status/1812029979888439525?s=46) indicates that new models are appearing in the **LMSYS arena**, spurring community hype.
   - Among the models discussed are **column-r** and **column-u**, rumored to be from **Cohere**.
- **Stealth Model Releases in LMSYS**: Twitter user [@btibor91](https://x.com/btibor91/status/1812491983220343239?s=46) confirms a trend of stealthily pushing new models to LMSYS Chatbot Arena, mentioning four upcoming models including **eureka-chatbot** and **upcoming-gpt-mini**.
   - **Eureka-chatbot** appears to be trained by **Google**, according to error messages and hints from community members.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/apples_jimmy/status/1812029979888439525?s=46">Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)</a>: Seems like there’s a new model in Lmsys arena.</li><li><a href="https://fxtwitter.com/TheXeophon/status/1812069628685815986">Tweet from Xeophon (@TheXeophon)</a>: Column-U is also jailbreak-able with the same prompt, so its also a cohere model i guess</li><li><a href="https://x.com/apples_jimmy/status/1812047899137851811?s=46">Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)</a>: Also a new column-r that seems really good… are we finally seeing shit happen?  Quoting Jimmy Apples 🍎/acc (@apples_jimmy)   Seems like there’s a new model in Lmsys arena.</li><li><a href="https://fxtwitter.com/TheXeophon/status/1812069172727201808">Tweet from Xeophon (@TheXeophon)</a>: Column-R is a cohere model, the Command R+ jailbreaks work on it as well.</li><li><a href="https://x.com/btibor91/status/1812491983220343239?s=46">Tweet from Tibor Blaho (@btibor91)</a>: Looks like it became a new trend to stealthily push new models to LMSYS Chatbot Arena (and mostly non-selectable) for vibe check and hype before release  With 4 models upcoming right now, as far as I ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1261397531019972733)** (23 messages🔥): 

> - `Mistral-7B instruct-tuning`
> - `Orca3/AgentInstruct paper`
> - `InFoBench benchmark`
> - `WizardArena/ArenaLearning paper`
> - `ChatbotArena competition` 


- **Mistral-7B instruct-tuning scrutinized**: Discussion centered around the perceived improvements in the **Orca3/AgentInstruct paper** over **Mistral-7B's instruct-tuning**, with curiosity about the strength of Mistral's instruct-finetune dataset.
   - Questions were raised about the best-known instruct-tuning for **Mistral-7B**, hinting that current datasets may not be especially robust.
- **InFoBench benchmark divides opinions**: The **InFoBench (Instruction Following Benchmark)** was introduced as a new benchmark, prompting questions about its relevance compared to standard alignment datasets.
   - Debate ensued whether benchmarks like **EQ Bench** and **InFoBench** matter for highlighting valuable qualities in LMs, given high correlations with existing benchmarks like MMLU performance.
- **WizardArena paper and ChatbotArena competition analyzed**: Participants discussed the **WizardArena/ArenaLearning paper**, which details evaluating models using human preference scores, and the related **Kaggle competition**.
   - Interest was shown in multi-turn synthetic interaction generation and evaluations, with specific curiosity about how **WizardArena** sets up its judging process and multi-turn evaluation.
- **Questions about difficulty level predictions**: The **WizardArena paper** mentions using an LM to predict the instruction difficulty level, sparking questions on its accuracy and real-world correlations.
   - There was speculation around whether LMs could genuinely predict their own weaknesses, with reference to existing literature on **LM self-knowledge**.
- **Sharp posting rate noticed in discussions**: One user acknowledged their high posting rate and encouraged others to join the conversation actively.
   - This user seemed eager to engage and share their read-throughs and insights on various papers and benchmarks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/competitions/lmsys-chatbot-arena/overview)">LMSYS - Chatbot Arena Human Preference Predictions | Kaggle</a>: no description found</li><li><a href="https://www.interconnects.ai/p/rlhf-roundup-2024?r=68gy5&utm_campaign=post&utm_medium=web">RLHF roundup: Getting good at PPO, charting RLHF’s impact, RewardBench retrospective, and a reward model competition</a>: Things to be aware of if you work on language model fine-tuning.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1261730709979005089)** (7 messages): 

> - `Finite State Machine`
> - `Paper Rewriting Controversy`
> - `Google Plagiarism` 


- **Finite State Machine for Structured Generation**: **Outline's finite state machine** for structured generation has been up on [arXiv](https://x.com/remilouf/status/1812164616362832287) for almost a year, according to a post by @remilouf.
   - *I feel flattered, but still...*
- **Google accused of rewriting technical report**: **Brandon Willard** reported that some people at Google completely [rewrote their technical report](https://x.com/BrandonTWillard/status/1812163165767053772), citing it but making ridiculous brief comments about the differences.
   - He quoted @remilouf with the term *plagiarism* to underscore the severity of the issue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/BrandonTWillard/status/1812163165767053772">Tweet from Brandon T. Willard @brandonwillard@fosstodon.org (@BrandonTWillard)</a>: Yeah, looks like some people at Google completely rewrote our technical report.  Although they did cite it, the brief comments about the differences are ridiculous.  Quoting Rémi 〰️ (@remilouf)   Plag...</li><li><a href="https://x.com/remilouf/status/1812164616362832287">Tweet from Rémi 〰️ (@remilouf)</a>: I feel flattered, but still
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1261443600798187520)** (12 messages🔥): 

> - `OpenAI's revenue speculations`
> - `OpenAI Supply Co. shop`
> - `Shopify usage`
> - `Interconnects merch`
> - `Hackathons and free merch` 


- **VCs speculate OpenAI's revenue from chatbot summaries**: [Aaron Holmes](https://x.com/aaronpholmes/status/1811870687037960467?s=46) noted that VCs are circulating a speculative report on **OpenAI's revenue**, based on chatbot summaries from public web sources.
   - For firsthand reporting, he referred to a detailed [article published last month](https://www.theinformation.com/articles/openais-annualized-revenue-doubles-to-3-4-billion-since-late-2023).
- **OpenAI Supply Co. shop now internal only**: The **OpenAI Supply Co.** shop now requires a login with an @openai.com Microsoft account, as confirmed by [B Tibor's post](https://x.com/btibor91/status/1812778486039290260?s=46).
   - *It's likely internal or should not be publicly accessible for now.*
- **OpenAI merch via Shopify**: Discussion about **OpenAI merch** focused on using Shopify for merchandise stores.
   - One member mentioned their own [Interconnects Shopify store](https://interconnects.myshopify.com/) and showcased products like the [Coder Hoodie](https://interconnects.myshopify.com/products/coder-hoodie) and [Coffee Vessel #1](https://interconnects.myshopify.com/products/coffee-vessel-1).
- **Hackathons for free OpenAI merch**: A suggestion was made that attending a **hackathon** might be a good way to get free **OpenAI merch**.
   - *It's a pretty smart way to leverage events for promotional items.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aaronpholmes/status/1811870687037960467?s=46">Tweet from aaron holmes (@aaronpholmes)</a>: A lot of VCs are circulating a “report” today that speculates OpenAI’s revenue, based entirely on chatbot summaries of public web sources. If you want firsthand reporting on OpenAI’s revenue numbers, ...</li><li><a href="https://supply.openai.com/password">OpenAI Supply Co.</a>: OpenAI Supply Co.</li><li><a href="https://interconnects.myshopify.com/">Interconnects AI Store</a>: Official merchandise for the Interconects.ai blog for RL bois and gurls.</li><li><a href="https://x.com/btibor91/status/1812778486039290260?s=46">Tweet from Tibor Blaho (@btibor91)</a>: OpenAI Supply Co. Shopify store now requires login with an @ openai dot com Microsoft account - confirming it&#39;s only internal or should not be accessible, for now
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1262482366786572380)** (4 messages): 

> - `California AI Bill SB 1047`
> - `Paywall circumvention`
> - `Archive.is`
> - `Silicon Valley debates`
> - `Fortune article` 


- **California AI Bill SB 1047 sparks fierce debate**: The **California AI Bill SB 1047**, which passed the state’s Senate in May 32-1, is heading to a final vote in August amidst intense lobbying and discourse.
   - State senator **Scott Wiener** described the debate as *‘Jets vs Sharks’*, with **AI safety experts** clashing with top **venture capitalists** over the bill’s implications.
- **Paywall circumvention using Archive.is**: A discussion revealed a method to bypass paywalls by using [Archive.is](https://archive.is/e5n9A), allowing access to content behind paywalls like those on **Fortune**.
   - One user expressed surprise that sites have not yet patched this **loophole**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fortune.com/2024/07/15/california-ai-bill-sb-1047-fierce-debate-regulation-safety/">It&#x27;s AI&#x27;s &quot;Sharks vs. Jets&quot;—welcome to the fight over California&#x27;s AI safety bill</a>: The California state senator behind the controversial SB-1047 AI bill says he didn&#x27;t anticipate the opposition from Silicon Valley heavy-weights</li><li><a href="https://archive.is/e5n9A">California AI bill SB-1047 sparks fierce debate around regulation of &#x2026;</a>: no description found
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1261474726690160691)** (58 messages🔥🔥): 

> - `LangChain JS Usage`
> - `Gemini Pro vs API`
> - `RAG Errors`
> - `Using Base64 with APIs`
> - `OpenAI Embedding Models` 


- **Understanding LangChain JS: invoke, stream, and streamEvents**: A user queried about the differences between `invoke`, `stream`, and `streamEvents` in LangChain JS, wondering which to use with langgraph for streaming output, where nodes mainly involve tool calls.
   - In response, a suggestion was made to use agents for various actions such as data collection and API calls.
- **Base64 Input Issues with Gemini Pro**: A user tested Base64 with Gemini Pro API and encountered an 'invalid input' error, seeking help as the docs only mention File API upload without specifying Base64 format.
- **Transitioning from ToolCall to OpenAIToolCall**: Users discussed the deprecation of `ToolCall` and the need to use `OpenAIToolCall` instead, including the addition of an `index` property.
   - A user sought guidance on updating the LangChain package and handling unintended default tool calls in 'auto' mode.
- **Hallucinations in HuggingFace Models for Chatbots**: A user experienced hallucinations with HuggingFace models, where the LLM generated random question/answer pairs.
   - Suggestions included switching to openAI-models or FireworksAI models, noting that repetition penalties weren't effective for finetuned llama models.
- **Optimal OpenAI Embedding Model**: A query was raised regarding the best OpenAI embedding model, with a recommendation for `text-embedding-ada-002` being the default in LangChain.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://v02.api.js.langchain.com/interfaces/langchain_core_messages.ToolCall.html#Deprecated>)">ToolCall | LangChain.js - v0.2.9</a>: no description found</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/tool_calling/#passing-tools-to-llms>).">How to use a chat model to call tools | 🦜️🔗 Langchain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/few_shot_examples_chat/#create-prompt-template>)">How to use few shot examples in chat models | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://github.com/langchain-ai/langchain/issues/17737>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/9270>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/vectorstores/hippo/#declaring-the-embedding-model>)">Hippo | 🦜️🔗 LangChain</a>: Transwarp Hippo is an enterprise-level cloud-native distributed vector database that supports storage, retrieval, and management of massive vector-based datasets. It efficiently solves problems such a...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1261691234246852708)** (1 messages): 

> - `LLM Scraper`
> - `code generation`
> - `local models`
> - `GitHub project release`
> - `webpage scraping` 


- **LLM Scraper ships with code-generation support**: [LLM Scraper](https://github.com/mishushakov/llm-scraper) now includes code-generation support, allowing users to turn any webpage into structured data using **local models**.
   - This new feature is aimed at enhancing the tool's functionality and is available on the project's [GitHub page](https://github.com/mishushakov/llm-scraper) with detailed information and updates.
- **Turn any webpage into structured data using LLMs**: [LLM Scraper](https://github.com/mishushakov/llm-scraper) enables users to transform any webpage into structured data using Large Language Models (LLMs).
   - The GitHub repository provides an overview and contributions documentation on how to utilize this powerful tool.



**Link mentioned**: <a href="https://github.com/mishushakov/llm-scraper">GitHub - mishushakov/llm-scraper: Turn any webpage into structured data using LLMs</a>: Turn any webpage into structured data using LLMs. Contribute to mishushakov/llm-scraper development by creating an account on GitHub.

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1261428463169179748)** (10 messages🔥): 

> - `entity deduplication`
> - `LlamaCloud`
> - `GPT-4o for financial reports`
> - `multi-agent workflows with Redis`
> - `advanced RAG guide` 


- **Entity Deduplication with Neo4j Cypher Snippet**: A seriously cool [Cypher snippet](https://t.co/dAV2QuAoZH) by @tb_tomaz and others at @neo4j performs **entity deduplication** using a combination of text embeddings and word analysis.
- **LlamaCloud Streamlines Data Pipeline Management**: **LlamaCloud** now lets you manage your data pipelines all in one place, with new [team features](https://t.co/F73Spljg0a) enabling multiple users to have a central view of all projects.
- **Parsing Financial Reports with GPT-4o**: LlamaParse uses multimodal models like **GPT-4o** to easily extract text, diagrams, and tables from complex financial reports, which text-based parsers struggle with.
- **Multi-Agent Workflows with Redis Integration**: Thanks to @0xthierry, you can now build production agent systems using **Redis Queue** as the central message broker to coordinate multi-agent workflows.
   - This setup allows agents services to communicate via a central message queue, significantly streamlining the architecture.
- **Get Started with Advanced RAG Workflows**: A fantastic guide from @kingzzm teaches you how to use **LlamaIndex query pipelines** to build advanced RAG and agent modules with full visibility.
   - The step-by-step guide covers everything from basic to advanced settings, providing essential knowledge for AI engineers.



**Link mentioned**: <a href="https://t.co/ruxdlhZOuK">blogs/llm/llama_index_neo4j_custom_retriever.ipynb at master · tomasonjo/blogs</a>: Jupyter notebooks that support my graph data science blog posts at https://bratanic-tomaz.medium.com/ - tomasonjo/blogs

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1261813014160478268)** (18 messages🔥): 

> - `LlamaIndex KG node deduplication`
> - `Combining SQL and PDF embeddings`
> - `Handling chat history in FastAPI`
> - `Chunking data for better embeddings`
> - `KnowledgeGraphIndex with NebulaGraphStore` 


- **LlamaIndex KG Node Deduplication**: A member shared a [YouTube video](https://youtu.be/vMz0icWZd5A) and a [Medium article](https://medium.com/@rajib76.gcp/entity-de-duplication-llamaindex-approach-0b97d2950a9f) explaining the process of deduplicating nodes in LlamaIndex Knowledge Graph.
   - The video provides detailed insights into the technical approach and Rajib emphasizes the importance of knowledge modeling for making unstructured data GenAI ready.
- **Combining SQL and PDF Embeddings with LlamaIndex**: A user inquired about combining a MySQL database indexed using Manticore search with PDF documents as embeddings, following [an example](https://docs.llamaindex.ai/en/stable/examples/query_engine/SQLJoinQueryEngine/) from LlamaIndex documentation.
   - The user faced issues using `NLSQLTableQueryEngine` because Manticore queries differ from MySQL, seeking a best approach to handle this.
- **Handling Chat History in FastAPI with LlamaIndex**: Discussion on best practices for managing chat history in a multi-user FastAPI backend using LlamaIndex, weighing options between storing dictionaries of chat engines or maintaining chat history for each interaction.
   - The consensus leaned towards managing just the chat history, possibly using a simple chat store.
- **Smaller Chunk Sizes Enhance Embeddings**: Chunking data into smaller sizes can help make embeddings more precise in LlamaIndex, as smaller chunk sizes offer finer-grained details.
   - Configuration example provided: setting `Settings.chunk_size` to 512 with an overlap of 50 and adjusting `similarity_top_k` to 4 for better retrieval accuracy, according to [LlamaIndex documentation](https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes).
- **Issues with NebulaGraphStore in KnowledgeGraphIndex**: A member faced issues running a [NebulaGraph example notebook](https://github.com/run-llama/llama_index/blob/0250d337a2cd68d724c32753c9187d7683d9822f/docs/docs/examples/query_engine/knowledge_graph_query_engine.ipynb) for `KnowledgeGraphIndex`, as noted in [GitHub Issue #14748](https://github.com/run-llama/llama_index/issues/14748).
   - The error `KnowledgeGraphIndex._build_index_from_nodes() got an unexpected keyword argument 'space_name'` was raised, and they sought advice on resolving it.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@rajib76.gcp/entity-de-duplication-llamaindex-approach-0b97d2950a9f">Entity De-duplication | LlamaIndex Approach</a>: LlamaIndex released a smart way to de-duplciate the entities of a Knowledge Graph created by a Language model. I looked at their approach…</li><li><a href="https://youtu.be/vMz0icWZd5A">LlamaIndex KG | Deduplication of nodes.</a>: In this recording, I explain in details how LlamaIndex is doing the deduplication of the nodes after creating the knowledge graphcode:https://github.com/raji...</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies/#chunk-sizes>).">Basic Strategies - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/issues/14748">[Bug]: KnowledgeGraphIndex._build_index_from_nodes() got an unexpected keyword argument &#39;space_name&#39; · Issue #14748 · run-llama/llama_index</a>: Bug Description I&#39;m trying to run this NebulaGraph example. Running this cell: from llama_index.core import KnowledgeGraphIndex kg_index = KnowledgeGraphIndex.from_documents( documents, storage_co...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/SQLJoinQueryEngine/.">SQL Join Query Engine - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/pdf_tables/recursive_retriever/?h=recursive+query+pandas">Recursive Retriever + Query Engine Demo - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1261407611778105456)** (13 messages🔥): 

> - `OpenInterpreter GUI Integration`
> - `OpenAI OS Rumors`
> - `Phi-3.1 Model Evaluation`
> - `Internlm2 Valuation`
> - `System Architecture Documentation Request` 


- **OpenInterpreter Fully Integrated into GUI**: [OpenInterpreter](https://github.com/jbexta/AgentPilot) has been fully integrated into a GUI by a member, featuring branching chats, editable messages, code auto-run, and chat saving.
   - Members expressed excitement over the project with others requesting video tutorials or demos to better understand its functionalities.
- **Rumors of OpenAI OS Building**: A [tweet](https://x.com/apples_jimmy/status/1805373587127402883) suggests that Sam Altman and OpenAI might be developing their own OS and communication tool, citing increasing evidence.
   - This development followed a job opening posted a month ago, stirring discussions in the community.
- **Phi-3.1 Model Evaluation**: Techfren raised a discussion on the performance of the Phi-3.1 model, noting its promising size and capabilities.
   - Member twodogseeds shared insights, indicating that Phi-3.1 offers more than requested but sometimes struggles following <INST> accurately.
- **Internlm2 Smashed on Raspi5**: Twodogseeds pointed out that 'Internlm2 smashed' received attention, highlighting its performance on Raspi5.
   - They mentioned the potential of multi-shot and smash modes for edge devices, especially with IoT applications.
- **Request for System Architecture Documentation**: A member inquired about available documentation explaining the system-level architecture and breakdown of Open Interpreter.
   - No specific documentation was shared in response, indicating a potential gap or need for community-contributed resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/apples_jimmy/status/1805373587127402883">Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)</a>: I’m really bored so in case you missed it, a month or so ago they were hiring for this role.  Quoting Chubby♨️ (@kimmonismus)   The rumor seems to confirm that Sam Altman and OpenAi are building their...</li><li><a href="https://github.com/jbexta/AgentPilot">GitHub - jbexta/AgentPilot: Universal GUI for seamless interaction and management of AI workflows</a>: Universal GUI for seamless interaction and management of AI workflows - jbexta/AgentPilot
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1261603980904103937)** (3 messages): 

> - `Meta Ray-Ban Jailbreak`
> - `Installing O1 on Linux`
> - `'Interpreter' Not Defined Error` 


- **Meta Ray-Ban Jailbreak Interest**: A member expressed excitement about the possibility of jailbreaking **Meta Ray-Ban**.
   - They stated, *'That would be awesome, let me know if you do jailbreak Meta Ray-Ban.'*
- **O1 Linux Installation Patch**: A member shared the steps to install **O1** on Linux, mentioning a necessary patch in **Poetry**.
   - They needed to remove a dependency to complete the installation.
- **'Interpreter' Not Defined Error**: A member encountered an error message indicating that 'interpreter' is not defined while using O1.
   - They reviewed the server code but couldn't find a solution, expressing their frustration.


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1261630731222781952)** (1 messages): 

> - `LLM agent`
> - `Adding agents in LLMs`
> - `Modular components in chat pipelines`
> - `Processing information using agents`
> - `Interacting with external APIs` 


- **How LLM agents work in detail**: A user shared a [detailed guide](https://nik-hil.hashnode.dev/how-to-add-agents-in-large-language-models-a-detailed-guide) explaining how to add agents in Large Language Models (LLMs), focusing on their modular nature and their roles in the Chat pipeline.
   - The guide describes the process steps: **Input Processing**, **LLM Interpretation**, and using JSON output to invoke agents based on conversation needs.
- **Modular components enhance LLM chat pipelines**: The detailed guide emphasizes that agents in LLMs act as modular components, performing tasks such as **fetching data**, **processing information**, and **interacting with external APIs**.
   - By leveraging the JSON output capability of LLMs, these agents can be seamlessly integrated into the conversation flow to address specific requirements.



**Link mentioned**: <a href="https://nik-hil.hashnode.dev/how-to-add-agents-in-large-language-models-a-detailed-guide">Adding Agents to Large Language Models Guide</a>: Learn how to add agents in large language models using JSON output for flexible, scalable chat pipelines in this detailed guide

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1261414721802862864)** (2 messages): 

> - `OpenAI API Key request` 


- **OpenAI API Key request for a chatbot project**: A member requested an API key for OpenAI to use in a chatbot project.
   - They mentioned needing the key to create a tutorial for the project.
- **Seeking unused OpenAI API keys**: Same member asked if anyone had an unused OpenAI API key they could share.
   - They specified that the key was needed only for a tutorial.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/)** (1 messages): 

healthymonkey: I’ve heard it’s about a year. I really like how easy it is to get H100s on modal lol
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1261751924890402909)** (1 messages): 

> - `Credit Denial` 


- **Credits cannot be granted after deadline**: Attempts to reach out before the deadline were unsuccessful, resulting in the denial of credits.
   - No further details were provided.
- **No additional responses**: No responses were received within the specified deadline.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1261727386261651476)** (1 messages): 

> - `Training Loss Issues`
> - `Template Correctness`
> - `Meta's Template` 


- **Training Loss Refuses to Drop**: A member is experiencing issues with their training loss not decreasing using a specified setup, indicating a potential problem in their method.
   - The shared [code snippet](https://link.to/examples) and output suggest possible issues in dataset loading and prompt formatting.
- **Correct Template Verification**: A member provided an output example matching the template from [Meta's documentation](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/).
   - The template follows `type: input_output` with segments labeled as true or false for training responses.



**Link mentioned**: <a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Meta Llama 3 | Model Cards and Prompt formats</a>: Special Tokens used with Meta Llama 3. A prompt should contain a single system message, can contain multiple alternating user and assistant messages, and always ends with the last user message followe...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1261495743223697460)** (2 messages): 

> - `modal error`
> - `axolotl troubleshooting`
> - `seeking help on slack` 


- **Seeking Help on Slack for Modal Error**: A member mentioned an unfamiliar error, speculating it might be specific to **modal** and suggested asking on Slack.
   - *Havent seen this error before but my guess is it's modal specific I would ask on their slack.*
- **Struggling with Modal and Axolotl**: Another member chimed in, confirming struggles with both **modal** and **axolotl**.
   - *thanks. I have been struggling both with modal and axolotl.*


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1262065191496192082)** (1 messages): 

> - `Langsmith evaluation`
> - `Rate limits in OpenAI` 


- **Tackling Rate Limits in Langsmith Evaluation**: A user was encountering token rate limits per minute while running [Langsmith evaluation tests](https://link.to/example) using OpenAI credits.
   - They found that adjusting the **max_concurrency** parameter helped mitigate the issue.
- **Introducing Delays in Experiments**: Another part of the conversation involved looking for ways to introduce delays into experiments to avoid hitting rate limits.
   - Suggestions were sought for implementing this into the existing basic script.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1261449560031756490)** (5 messages): 

> - `OpenAI Credit Expiration`
> - `Petition for Credit Extension` 


- **OpenAI Credits Expire on September 1**: **OpenAI credits** are set to expire on **September 1**, confirmed by members after a query on the matter.
   - One user appreciated the clarification after another member pointed out where to find this information.
- **Petition for Extending OpenAI Credits**: A user humorously requested a **petition to extend the expiry** date of OpenAI credits.


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1261506856359301150)** (2 messages): 

> - `Hugging Face Profitability`
> - `Cambrian-1 Multimodal LLMs` 


- **Hugging Face Achieves Profitability**: [Hugging Face](https://analyticsindiamag.com/hugging-face-announces-profitability-with-free-and-open-source-models), a leading platform for developing and sharing machine learning models, announced its profitability with a team of 220 members, maintaining a largely free and open-source platform.
   - Chief Clement Delangue shared on X, *'This isn’t a goal of ours because we have plenty of money in the bank but quite excited to see that @huggingface is profitable these days, with 220 team members and most of our platform being free (like model hosting) and open-source for the community!'*
- **Cambrian-1 Multimodal LLMs Unveiled**: The [Cambrian-1](https://github.com/cambrian-mllm/cambrian) family of multimodal LLMs with a vision-centric design was introduced, expanding the capabilities of AI models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/cambrian-mllm/cambrian">GitHub - cambrian-mllm/cambrian: Cambrian-1 is a family of multimodal LLMs with a vision-centric design.</a>: Cambrian-1 is a family of multimodal LLMs with a vision-centric design. - cambrian-mllm/cambrian</li><li><a href="https://analyticsindiamag.com/hugging-face-announces-profitability-with-free-and-open-source-models/">Hugging Face Announces Profitability with Free and Open-Source Models &#8211; AIM</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1261816129458802832)** (1 messages): 

> - `MagViT2 compatibility with non-RGB motion data`
> - `Motion data preprocessing` 


- **MagViT2 for non-RGB motion data**: A user inquired if **MagViT2** can be used for motion data that are not in RGB format, mentioning their data as 24x3.
   - *No additional discussions or comments were provided in the messages.*
- **Motion data preprocessing techniques**: Members are exploring various preprocessing techniques for non-RGB motion data to ensure compatibility with existing AI models.
   - *Further details and specific preprocessing methods were not discussed in the messages.*


  

---



### **DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1261436075466035321)** (2 messages): 

> - `LLM Arena`
> - `Ollama models`
> - `WizardLM paper`
> - `Arena Learning methodology` 


- **Introducing OpenArena for LLM Battles**: A member shared the launch of [OpenArena](https://github.com/syv-ai/OpenArena), a platform for pitting 2 LLMs against each other with a 3rd acting as a judge to enhance dataset quality.
   - The platform primarily uses models from **Ollama** but supports any OpenAI compatible endpoint.
- **Foundation of OpenArena in WizardLM Paper**: The [WizardLM paper](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/) introduces '**Arena Learning**' - a simulated chatbot arena for evaluating LLMs.
   - The methodology includes precise evaluations and consistent offline simulations to improve the LLM through supervised fine-tuning and reinforcement learning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/syv-ai/OpenArena">GitHub - syv-ai/OpenArena</a>: Contribute to syv-ai/OpenArena development by creating an account on GitHub.</li><li><a href="https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/">Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena - Microsoft Research</a>: Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena
</li>
</ul>

</div>
  

---



---



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

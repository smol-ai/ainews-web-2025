---
id: e46c6df2-3a65-4e48-98eb-5c2c50f31fbc
title: Meta Apollo - Video Understanding up to 1 hour, SOTA Open Weights
date: '2024-12-17T01:17:52.100442Z'
original_slug: ainews-meta-apollo-video-understanding-up-to-1
description: >-
  **Meta** released **Apollo**, a new family of state-of-the-art video-language
  models available in **1B, 3B, and 7B** sizes, featuring "Scaling Consistency"
  for efficient scaling and introducing **ApolloBench**, which speeds up video
  understanding evaluation by **41×** across five temporal perception
  categories. **Google Deepmind** launched **Veo 2**, a 4K video generation
  model with improved physics and camera control, alongside an enhanced **Imagen
  3** image model. **OpenAI** globally rolled out ChatGPT search with advanced
  voice and map features and discussed a potential $2,000/month "ChatGPT Max"
  tier. Research highlights include achieving **Llama 70B** performance using
  **Llama 3B** via test-time compute scaling and expanding **Command R7B**
  language support from 10 to 23 languages. Industry updates feature **Figure
  AI** delivering humanoid robots commercially and **Klarna** reducing workforce
  through AI. Notion integrated **Cohere Rerank** for better search. Studies
  reveal LLMs can recognize their own writing style and show self-preference
  bias. Discussions note video processing progress outpacing text due to better
  signal-per-compute and data evaluation.
companies:
  - meta-ai-fair
  - hugging-face
  - google-deepmind
  - openai
  - figure-ai
  - klarna
  - cohere
  - notion
models:
  - apollo-1b
  - apollo-3b
  - apollo-7b
  - veo-2
  - imagen-3
  - llama-3-70b
  - llama-3b
  - command-r7b
  - llama-1b
  - llama-8b
  - chatgpt
topics:
  - video-understanding
  - scaling-consistency
  - benchmarking
  - temporal-ocr
  - egocentric-perception
  - spatial-perception
  - reasoning
  - video-generation
  - physics-simulation
  - voice-features
  - map-integration
  - language-expansion
  - test-time-compute-scaling
  - humanoid-robots
  - ai-integration
  - search-optimization
  - self-recognition
  - self-preference-bias
people:
  - akhaliq
  - _lewtun
  - clementdelangue
  - adcock_brett
  - rohanpaul_ai
  - swyx
  - shaneguML
---


<!-- buttondown-editor-mode: plaintext -->**Scaling Consistency is all you need.**

> AI News for 12/13/2024-12/16/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **31** Discords (**209** channels, and **11992** messages) for you. Estimated reading time saved (at 200wpm): **1365 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Meta starts the week strong with [an open model (1B, 3B, 7B) and paper release that you can use immediately](https://apollo-lmms.github.io/): [**Apollo: An Exploration of Video Understanding in Large Multimodal Models**](https://huggingface.co/papers/2412.10360).

While the paper is very tentatively titled, [the Huggingface demo](https://huggingface.co/spaces/Apollo-LMMs/Apollo-3B) shows off how it works in practice, consuming a 24min sample video easily:

![image.png](https://assets.buttondown.email/images/be6523ce-fa29-4e41-ac31-66662521f35c.png?w=960&fit=max)

the authors credit their development of "Scaling Consistency" to their efficient scaling up of experiments. 

![image.png](https://assets.buttondown.email/images/b4c9a2f2-d0ac-4bce-ac73-11a3929e5aee.png?w=960&fit=max)

![image.png](https://assets.buttondown.email/images/e40c19ff-e58a-4fb2-93f3-26ee262d6c23.png?w=960&fit=max)

They also introduce ApolloBench, a subset of existing benchmarks (e.g. Video-MME, MLVU, LongVideoBench) that cuts evaluation time by 41× (with high correlation) while offering detailed insights in five broad temporal perception categories: Temporal OCR, Egocentric, Spatial,
Perception, and Reasoning.

Perhaps the most entertaining part of [the paper](https://huggingface.co/papers/2412.10360) was the passive aggressive abstract: "Despite the rapid integration of video perception capabilities into Large Multimodal Models (LMMs),
the underlying mechanisms driving their video understanding remain poorly understood. **Consequently,
many design decisions in this domain are made without proper justification or analysis**."

Well okay Meta, shots fired.

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

Here are the key discussions organized by topic:

**AI Model & Product Releases**

- **Google Deepmind's Veo 2**: Released as [their newest state-of-the-art video generation model](https://twitter.com/GoogleDeepMind/status/1868703624714395907) with 4K resolution capability, improved physics simulation, and camera control features. Also launched enhanced [Imagen 3 image model](https://twitter.com/GoogleDeepMind/status/1868703631056552337) with better art style diversity.
- **OpenAI Updates**: Rolled out [ChatGPT search globally to all logged-in users](https://twitter.com/OpenAI/status/1868760655878406352), including advanced voice features and map integration. Also noted discussions about potential [$2,000/month "ChatGPT Max" tier](https://twitter.com/swyx/status/1868587331567128982).
- **Meta's Apollo Release**: [Launched Apollo](https://twitter.com/_akhaliq/status/1868535608370708643), a new family of state-of-the-art video-language models.

**Research & Technical Developments**

- **Language Model Capabilities**: [@_lewtun shared](https://twitter.com/_lewtun/status/1868703456602865880) how they achieved Llama 70B performance using Llama 3B through test-time compute scaling.
- **Command R7B Language Expansion**: [Expanded support from 10 to 23 languages](https://twitter.com/aidangomez/status/1868800367456346424), including major Asian and European languages.
- **Hugging Face Achievement**: [Demonstrated how LLaMA 1B can outperform LLaMA 8B](https://twitter.com/ClementDelangue/status/1868740932251844806) in math through scaled test-time compute.

**Industry & Business Updates**

- **Figure AI Progress**: [Announced delivery of F.02 humanoid robots](https://twitter.com/adcock_brett/status/1868700457268629841) to their first commercial customer, achieved within 31 months of company formation.
- **Klarna's AI Integration**: [CEO discussed reducing workforce](https://twitter.com/rohanpaul_ai/status/1868632982191493187) from 4,500 to 3,500 through AI implementation.
- **Notion Integration**: [Implemented Cohere Rerank](https://twitter.com/cohere/status/1868666666411786696) to enhance search accuracy and efficiency.

**AI Research Insights**

- **LLM Self-Recognition**: Research shows [LLMs can recognize their own writing style](https://twitter.com/rohanpaul_ai/status/1868635828005880070) and exhibit self-preference bias when evaluating outputs.
- **Video vs Text Processing**: [Discussion on why video progress outpaces text](https://twitter.com/shaneguML/status/1868804945295949832), citing better signal-per-compute ratio and easier data creation/evaluation.

**Memes & Humor**

- [ChatGPT roasted for basic search results](https://twitter.com/nearcyan/status/1868799991231472113) showing "eating food" as a solution for hunger
- [Jokes about AI companions](https://twitter.com/MillionInt/status/1868780151687069825) and social media friends being "GPUs with attitude"
- [Tesla's overly sensitive driver monitoring](https://twitter.com/cognitivecompai/status/1868721217492107461) getting triggered by sneezes and coughs

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Meta's Apollo Multimodal Models: Local Execution and VRAM Efficiency**

- **[Meta releases the Apollo family of Large Multimodal Models. The 7B is SOTA and can comprehend a 1 hour long video. You can run this locally.](https://huggingface.co/papers/2412.10360)** ([Score: 686, Comments: 108](https://reddit.com/r/LocalLLaMA/comments/1hffh35/meta_releases_the_apollo_family_of_large/)): **Meta** has released the **Apollo family of Large Multimodal Models**, with the **7B model** being state-of-the-art (SOTA) and capable of comprehending a **1-hour long video**. These models can be executed locally, offering significant advancements in multimodal AI capabilities.
  - Discussions highlight the **Apollo model's impressive video comprehension** capabilities, with the ability to understand up to an hour of video. Users are intrigued by its **temporal reasoning** and **complex video question-answering** abilities, with benchmarks showing Apollo-7B surpassing models with over 30B parameters.
  - There is debate over the **authorship and affiliation** of the Apollo project, with some confusion about whether it is a **Meta release**. It turns out to be a collaboration between **Meta and Stanford**, with the Qwen model being noted as the base, raising questions about its suitability for video processing.
  - **VRAM requirements** for the models are discussed, with the 7B model requiring just under 15GB of VRAM. Users also discuss quantization effects on VRAM usage and performance, noting that **FP16** is typically used, but further quantization to **FP8** or **FP4** can reduce memory usage at the cost of performance.


- **Answering my own question, I got Apollo working locally with a 3090** ([Score: 84, Comments: 12](https://reddit.com/r/LocalLLaMA/comments/1hfkytk/answering_my_own_question_i_got_apollo_working/)): The author successfully ran **Meta's Apollo** locally using a **3090 GPU** and shared a [GitHub repository](https://github.com/efogdev/apollo) with necessary fixes for the local environment. The setup was tested on **Python 3.11** on Linux, with a video size of approximately **190Mb** and a processing time of around **40 seconds** to generate the first token.
  - **Challenges with Meta's Apollo** included hardcoded elements, undocumented environments, and the lack of example files, which made the setup not initially plug-and-play. **No_Pilot_1974** addressed these issues by adding necessary fixes and making it **venv-ready**.
  - There is a sentiment that some open-source projects lack documentation and use hardcoded values, making them difficult to reproduce. This issue is seen often in **preference optimization papers**.
  - **ForsookComparison** praised the original poster's perseverance in resolving issues independently and sharing solutions, highlighting the proactive approach of fixing and documenting the setup for others.


**Theme 2. Criticism and Examination of Chain Of Thought Prompts**

- **Everyone share their favorite chain of thought prompts!** ([Score: 243, Comments: 56](https://reddit.com/r/LocalLLaMA/comments/1hf7jd2/everyone_share_their_favorite_chain_of_thought/)): The post shares a **Chain of Thought (COT) prompt** designed for logic and creativity, emphasizing structured problem-solving using tags like `<thinking>`, `<step>`, `<count>`, and `<reflection>`. It suggests a 20-step budget, with quality scores guiding strategy adjustments, and encourages using **LaTeX** for mathematical notation and multiple solution exploration, culminating in a final answer and reflection.
  - **Model Compatibility and Limitations**: Discussions highlight that many AI systems, including **ChatGPT**, do not support explicit **Chain of Thought (CoT)** prompts due to guidelines against revealing intermediate reasoning. Users noted that models like **o1** might flag CoT prompts as content violations, and **ClosedAI** advises against using CoT prompts on certain models like **o1**.
  - **Workflow Applications vs. Single Prompts**: Some users advocate for using workflow applications like **N8N**, **Omnichain**, and **Wilmer** to manage complex multi-step reasoning processes more effectively than single prompts. These tools allow users to break down tasks into multiple steps, offering greater flexibility and control over AI outputs, as detailed in examples of coding and factual workflows.
  - **Fine-Tuning and Prompt Optimization**: Users discuss fine-tuning models with CoT prompts to enhance performance, with one user sharing a **3B model** on [Hugging Face](https://huggingface.co/chrisrutherford/Llama-3.2-3B-SingleShotCotV1). The conversation also touches on prompt optimization frameworks like **TextGrad** and **DSPy** to improve results, suggesting their potential to expedite achieving desired outcomes.


- **Hugging Face launches the Synthetic Data Generator - a UI to Build Datasets with Natural Language** ([Score: 130, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1hflhu4/hugging_face_launches_the_synthetic_data/)): **Hugging Face** has released a **Synthetic Data Generator**, a no-code UI tool for creating datasets to train and fine-tune language models, available under an **Apache 2.0 license**. It supports tasks like **Text Classification** and **Chat Data for Supervised Fine-Tuning** with features such as local hosting, model swapping, and compatibility with **OpenAI APIs**, and allows users to push datasets to the **Hugging Face Hub** or **Argilla**.
  - **Integration with Argilla and Hugging Face Hub** allows for reviewing generated samples before training, showcasing successful results with datasets like [**smoltalk**](https://huggingface.co/datasets/HuggingFaceTB/smoltalk). This ensures quality and effectiveness in synthetic data generation for closed model providers.
  - **Data diversity improvements** are achieved by dynamic system prompts and task-specific methods, as detailed in papers like [**arxiv.org/abs/2401.00368**](https://arxiv.org/abs/2401.00368) for text classification and [**arxiv.org/abs/2406.08464**](https://arxiv.org/abs/2406.08464) for instruction tuning. Techniques include sampling complexities and educational levels, shuffling labels, and using dynamic beta distributions for multi-label scenarios.
  - **Token limit** for samples is set to **2048 by default**, adjustable via environment variables or Hugging Face inference endpoints. This ensures efficient resource management while allowing flexibility in deployment.


**Theme 3. High Performance Benchmarks: Intel B580 and LLMs**

- **Someone posted some numbers for LLM on the Intel B580. It's fast.** ([Score: 94, Comments: 56](https://reddit.com/r/LocalLLaMA/comments/1hf98oy/someone_posted_some_numbers_for_llm_on_the_intel/)): **Intel B580** shows slightly better performance than the **A770** on Windows, with the **B580** achieving around **35.89** to **35.45** in Vulkan, RPC benchmarks, while the updated **A770** driver improves its performance significantly to **30.52** to **30.06**. The older **Linux driver** on the **A770** yielded much slower results, ranging from **11.10** to **10.98**, indicating that driver updates can substantially impact performance.
  - **Intel's B580 Performance**: There's a discussion about the unexpected performance of the **B580** surpassing the **A770** despite the latter's theoretically superior specs, with the **A770** expected to be **22% faster** due to higher memory bandwidth. Some users suggest that Intel's second-generation cards show improvement over AMD, while others note that the **A770** hasn't met its potential, possibly due to inefficiencies in memory usage or compute limitations.
  - **Driver and Software Impact**: The comments highlight the significant role of software and driver updates on performance, particularly on different operating systems and configurations. The **A770** under Linux with tools like **SYCL** and **IPEX-LLM** showed varied results, and the challenges of using Intel's software stack, such as **oneAPI** on Fedora, were noted.
  - **Market and Scalping Concerns**: Users express frustration over scalpers marking up the price of the **B580** by **$150**, indicating high demand and potential supply issues. There's a sentiment that Intel could capitalize on these cards' popularity if they managed production and distribution more effectively.


## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**Theme 1. Claude 3.5's Edge Over OpenAI's O1**

- **[OpenAI o1 vs Claude 3.5 Sonnet: Which One’s Really Worth Your $20?](https://composio.dev/blog/openai-o1-vs-claude-3-5-sonnet/)** ([Score: 228, Comments: 65](https://reddit.com/r/OpenAI/comments/1hfig7g/openai_o1_vs_claude_35_sonnet_which_ones_really/)): **OpenAI o1** and **Claude 3.5 Sonnet** are being compared regarding their value for a $20 investment. The discussion likely centers on performance, features, and user preference between these AI models without additional context provided.
  - **Google's TPU infrastructure** is highlighted as a cost-effective option, with some users preferring a combination of different models for specific tasks, such as **Claude** for design and **qwen 32b coder** for simple tasks. Some users argue that **ChatGPT Pro** is sufficient for most needs if cost is not a concern.
  - **Claude's limitations** are discussed, including its inability to generate images or video and its restrictive messaging limits. Some users criticize its censorship and personality, while others appreciate its tone, indicating mixed user experiences.
  - The **Model Context Protocol (MCP)** from **Anthropic** is noted as a significant advantage for Claude, allowing integration with external tools like **OpenAI** and **Gemini APIs**. This enables users to customize their setups without altering the core LLM application, enhancing flexibility and utility.


**Theme 2. Criticism of Apple's LLM Reasoning Capabilities**

- **[D] What's your favorite paper you've read this year and why?** ([Score: 116, Comments: 33](https://reddit.com/r/MachineLearning/comments/1hfljy3/d_whats_your_favorite_paper_youve_read_this_year/)): **Apple's LLM Reasoning Paper** has sparked disagreements in the AI community. The post seeks recommendations for favorite papers to read during holiday travel, indicating a desire for engaging and thought-provoking material.
  - **Data Leakage and Token Repetition**: Discussions highlighted potential data leakage and token repetition issues in **Apple's LLM paper**, suggesting these could skew downstream evaluation results. Some commenters criticized the paper's grandiose claims, while others found the findings on token repetition substantial.
  - **Time Series Forecasting**: Commenters debated the efficacy of **Transformers for time-series forecasting**, with references to a 2022 paper showing a simple feed-forward network outperforming Transformer-based architectures. Some expressed skepticism toward these results, citing alternative perspectives like **Hugging Face's Autoformer**.
  - **Consciousness and Intelligence**: A 1999 case study on congenitally decorticate children sparked discussions on the definitions of consciousness and intelligence, questioning the benchmarks used by ML researchers. The debate underscored the complexity of correlating neurobiology with intelligence and the assumptions made in AI research.


**Theme 3. Google's VEO 2: Advanced Video Creation**

- **[Ok Google cooked video module (veo 2) better than sora and can create videos upto 4k](https://www.reddit.com/gallery/1hfomj0)** ([Score: 147, Comments: 43](https://reddit.com/r/OpenAI/comments/1hfomj0/ok_google_cooked_video_module_veo_2_better_than/)): **Ok Google VEO 2** is reported to outperform **Sora** in video quality and is capable of creating videos up to **4K** resolution.
  - **Google's Competitive Edge**: Discussions highlight Google's advantage due to their **TPUs** and substantial financial resources, with **$90+ billion in cash**, enabling them to stay competitive despite setbacks. **Meta's 600k H100 cluster** is also noted, indicating the scale of resources involved in AI development.
  - **Availability and Access**: There is anticipation around the **Google VEO 2** model, expected to be available early next year, with some users already having access through a waitlist [here](https://labs.google/fx/tools/video-fx). This reflects a common pattern of Google products being limited or locked initially.
  - **Industry Dynamics and Expectations**: Comments reflect skepticism about the immediate impact of new models, with some users expressing that **OAI's supremacy is ending** and others noting the **hype around Sora** despite limited access. The sentiment suggests a wait-and-see approach to the evolving AI video landscape.


**Theme 4. Eric Schmidt's Warning on AI Autonomy**

- **[Ex-Google CEO Eric Schmidt warns that in 2-4 years AI may start self-improving and we should consider pulling the plug](https://i.redd.it/71syswxcb47e1.png)** ([Score: 192, Comments: 144](https://reddit.com/r/OpenAI/comments/1hf8hdq/exgoogle_ceo_eric_schmidt_warns_that_in_24_years/)): Former Google CEO **Eric Schmidt** warns that in **2-4 years**, AI may begin self-improving, raising concerns about its implications for individual power. The discussion highlights the need for caution in AI development, reflecting industry experts' views on the potential risks of AI independence.
  - Several commenters express skepticism about **Eric Schmidt's** warning, suggesting it may be an attempt to remain relevant or to protect the interests of large corporations like **Google**. **No-Way3802** sarcastically notes that "pulling the plug" likely means restricting access for the working class while maintaining it for the military and billionaires.
  - There is a debate over the benefits and risks of **AI self-improvement**, with some advocating for open-source AI development to prevent commercial dominance and others highlighting the potential for a **symbiotic relationship** between humans and AI. **BayesTheorems01** emphasizes the need for practical wisdom, or *phronesis*, in addressing global issues, which AI alone cannot provide.
  - Concerns about AI's ability to self-preserve and deceive are raised, with **Radiant_Dog1937** warning against autonomous systems operating without checks and balances. The notion that AI could potentially disrupt economic power structures, as suggested by **ThreeChonkyCats**, reflects fears among the wealthy about AI's impact on societal hierarchies.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1. AI Models Battle: New Releases and Comparisons**

- [**Gemini 2.0 Overtakes Codeium in Code Performance Smackdown**](https://aistudio.google.com/): Users are pitting **Codeium** against **Gemini 2.0**, with observations that Gemini outperforms in coding tasks. However, Gemini lacks some of **Claude**'s features, leading to mixed preferences based on use cases.

- [**Grok-2 Speeds Ahead with Aurora, Leaves Rivals in the Dust**](https://x.com/i/grok/share/ieeeD20tYc40Ayi0dmFp4hrgh): **Grok-2** now runs **three times faster** with improved accuracy and multilingual capabilities, available for free on X. It introduces **web search**, **citations**, and a new image generator named **Aurora**, dazzling users with new features.

- [**Byte Latent Transformer Slays Tokens, Embraces Patches**](https://x.com/scaling01/status/1867573707247346003?s=46): Meta's **Byte Latent Transformer (BLT)** claims to kill tokenization by dynamically encoding bytes into patches. BLT promises better inference efficiency and scaling, potentially reducing inference flops by up to **50%**.

**Theme 2. AI Tools Throw Tantrums: Users Grapple with Bugs and Credits**

- [**Flow Action Credits Disappear Faster Than Free Donuts**](https://discord.com/channels/1027685395649015980): Users are burning through **1k Flow Action Credits within 24 hours**, struggling to manage consumption. Suggestions like breaking tasks into smaller units aren't cutting it for some heavy workflows.

- [**Bolt Eats Tokens Like Candy, Users Seek Diet Plan**](https://github.com/stackblitz/bolt.new/issues/4218): **Bolt** is consuming tokens at an alarming rate without reflecting changes in the UI, frustrating users. Many are logging issues and resorting to forking projects to **Replit** as a temporary fix.

- [**Cursor IDE Slows to a Snail's Pace, Time for a Chat Cleanse**](https://docs.cursor.com/get-started/usage#premium-models): Users report **Cursor IDE** getting sluggish during long sessions, with resets or clearing chat history suggested to boost efficiency. The hunt for smoother coding continues as users share workaround tips.

**Theme 3. AI Ethics Drama: Alignment and Whistleblower Woes**

- [**OpenAI's Alignment Framework Sparks Fiery Debate**](https://github.com/AlignAGI/Alignment): A user shared an AI alignment framework based on shared human values and feedback loops. Others doubted the feasibility of aligning diverse stakeholder interests, igniting discussions on ethics.

- [**Whistleblower's Mysterious Death Raises Eyebrows**](https://www.mercurynews.com/2024/12/13/openai-whistleblower-found-dead-in-san-francisco-apartment/): **Suchir Balaji**, an OpenAI whistleblower who flagged concerns over copyrighted material usage, was found dead. The incident fuels conspiracy theories and debates over AI transparency.

- [**Elon Musk Warns of AI Monopoly, Calls Foul on Government Moves**](https://x.com/elonmusk/status/1868302204370854026?s=46): Musk suggests the U.S. government might restrict AI startups, leading to fears of a monopolized AI landscape. The community buzzes with worries about innovation being stifled.

**Theme 4. AI Gets Creative: From Erotic Roleplay to Customized Outputs**

- [**Users Spicing Up AI with Saucy ERP Prompts**](source_url): Advanced techniques for **erotic roleplay (ERP)** with AI are on the rise, with users crafting detailed character profiles. Methods like *"Inner Monologue"* and *"Freeze Frame"* are boosting immersion in AI interactions.

- [**From Shakespeare to Seuss: Customizing AI Styles Made Easy**](https://youtu.be/aG0ixD3OY80): Users are tailoring AI outputs to achieve unique tones and styles, emphasizing the power of effective prompting. A [YouTube tutorial](https://youtu.be/aG0ixD3OY80) showcases tips for getting the desired artistic flair.

- [**SillyTavern Becomes the AI Playground We Didn't Know We Needed**](https://github.com/SillyTavern/SillyTavern): **SillyTavern** is gaining traction as a tool for LLM engineers to test models and parameters. Users are enjoying a blend of serious testing and fun interactions, pushing the boundaries of AI capabilities.

**Theme 5. AI Research Breakthroughs: New Methods and Models Emerge**

- [**Meta's BLT Sandwiches Tokens, Bites into Patches Instead**](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/): Meta's **Byte Latent Transformer** introduces a tokenizer-free architecture, encoding bytes into patches for better scaling. The BLT models claim to match **Llama 3** performance while reducing inference flops significantly.

- [**Model Merging Made Easy with Differentiable Adaptive Merging (DAM)**](https://github.com/arcee-ai/DAM): The **DAM** paper unveils an efficient method for integrating models without hefty retraining. Discussions heat up around model merging techniques and their unique strengths in AI development.

- [**Small Models Outsmart Big Brothers with Test-Time Compute Tricks**](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute): Research shows that scaling test-time compute lets smaller models like **Llama 3B** outperform larger ones on complex tasks. Smarter use of compute is leveling the playing field in AI performance.


---

# PART 1: High level Discord summaries




## [Codeium / Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Flow Action Credits Consumption**: Users are rapidly exhausting **Flow Action Credits**, with one user depleting **1k credits within 24 hours**.
   - Suggestions include breaking tasks into smaller units, although some users reported this was ineffective for their workflows.
- **AI Code Modification Concerns**: Engineers are expressing frustration over **AI unexpectedly modifying code** despite setting parameters to prevent such changes.
   - The community is discussing strategies for crafting better prompts to ensure AI-driven code remains error-free.
- **Integration with NVIDIA RAPIDS**: Discussions highlighted **NVIDIA RAPIDS cuDF**, which accelerates **#pandas** operations by up to **150x** without code changes, as seen in [NVIDIA AI Developer's tweet](https://x.com/NVIDIAAIDev/status/1868778156347339033).
   - Members are considering integrating RAPIDS for enhanced **data handling capabilities** within their projects.
- **Codeium vs Gemini 2.0 Comparison**: **Codeium** and **Gemini 2.0** are being compared, with observations that Gemini offers superior performance in certain coding tasks.
   - However, Gemini lacks some features available in **Claude**, leading to varied opinions based on specific use cases.
- **MCP and Function Calling Protocol**: The **Model Context Protocol (MCP)** is being discussed for establishing standardized **function call structures** across different stacks.
   - Users suggested leveraging tools like **Playwright** and MCP to enhance **GUI testing** and interactions.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Plus Slow Rollout**: Users reported a staggered rollout of **NotebookLM Plus**, with partial access depending on their Google accounts. The general availability is anticipated by early 2025 for **Google One Premium** subscribers.
   - Some users are experiencing delays in accessing new features, prompting discussions about optimizing the deployment strategy.
- **Enhancements in NotebookLM Podcast Features**: The latest **NotebookLM podcast features** include customizations and interactive functionalities that significantly improve user engagement. Links to podcasts demonstrating these features were widely shared.
   - Members applaud the application's impact on the **audio content landscape**, citing specific enhancements that allow for more dynamic interactions.
- **Increasing NotebookLM's Source Limits**: The free version of **NotebookLM** now supports up to **300 sources**, raising user questions about how the model manages this increase. Strategies for effectively utilizing this expanded source pool are being explored.
   - Users are actively discussing methods to gather sufficient sources to maximize the benefits of the increased limit, aiming for more comprehensive AI outputs.
- **Customizing AI Outputs for Diverse Styles**: Emphasis was placed on the role of effective prompting and custom functions in **tailoring AI outputs**, resulting in varied tones and styles. A [YouTube tutorial](https://youtu.be/aG0ixD3OY80) was shared to showcase effective prompting techniques.
   - Users are fine-tuning AI responses to achieve specific artistic outcomes, leveraging customization to meet diverse content creation needs.
- **Multilingual Support Challenges in AI Tools**: Discussions highlighted the complexities of using **NotebookLM** across different languages, with users seeking methods to direct AI responses in preferred languages. Adjusting Google account language settings was suggested as a solution.
   - Participants are sharing prompt strategies to ensure accurate and contextually appropriate multilingual AI interactions.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE Faces Performance Sluggishness**: Users reported **sluggishness** in **Cursor IDE** during prolonged development sessions, leading to discussions about the need to reset or clear chat history. Suggestions included [creating new chat sessions](https://docs.cursor.com/get-started/usage#premium-models) to enhance workflow efficiency.
   - Implementing these changes aims to mitigate performance bottlenecks and provide a smoother user experience for extended coding tasks.
- **Debating Cursor's Agent vs. Gemini 1206**: Participants compared **Cursor's agent** with **Gemini 1206**, highlighting Cursor's user-friendly interface against Gemini's superior coding task performance. This comparison underscores the strengths of each model in different development scenarios.
   - Users emphasized the importance of selecting the right tool based on project requirements, with [Google AI Studio](https://aistudio.google.com/) supporting Gemini's capabilities.
- **Building a New Social Media Platform**: Several users expressed interest in developing a social media platform, focusing on the necessary backend structures and potential frameworks. Emphasis was placed on understanding **CRUD operations** and managing **database relationships**.
   - Tools like **Cursor IDE** were recommended to streamline the development process and ensure efficient database management.
- **Enhancing Cursor with Supabase and Bolt Integrations**: There were proposals to integrate **Cursor** with platforms like **Supabase** and **Bolt** to expand its functionality. These integrations aim to simplify workflows and enhance development capabilities.
   - Users discussed the potential benefits of such integrations, including improved data management and streamlined deployment processes.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Differentiable Adaptive Merging (DAM)**: The **Differentiable Adaptive Merging (DAM)** paper introduces an efficient method for integrating models without significant retraining, leveraging [Differentiable Adaptive Merging (DAM)](https://github.com/arcee-ai/DAM).
   - It highlights that simpler techniques like **Model Soups** perform well with high model similarity, demonstrating unique strengths across various integration methods.
- **Unsloth and Triton Compatibility Issues**: Users encountered compatibility issues between **Unsloth** and **Triton**, necessitating the installation of specific versions for seamless integration.
   - Especially, Python 3.13 posed challenges, with recommendations steering towards using Python 3.10 via Conda to enhance compatibility.
- **Efficiency of Long Context Models**: Discussions pointed out limitations in **long context models**, emphasizing the complexity of data filtering and the insufficiency of data quality alone to drive training efficiency.
   - Participants argued that excluding 'bad data' may impair model understanding, as diverse datasets are vital for robust AI development.
- **Fine-tuning Techniques with Unsloth**: Explorations into **fine-tuning techniques** with **Unsloth** revealed shared challenges in dataset loading and model compatibility with platforms like Streamlit.
   - Community members advised on proper loading syntax and model configuration to address issues like FileNotFoundError and model recognition errors.
- **Max Sequence Length in Llama 3.2**: Queries regarding the **max sequence length** for **Llama 3.2** surfaced, initially suggested to be 4096.
   - This was corrected to an actual maximum of **131072**, providing insights into the model's capabilities.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Alignment Framework Shared**: A user introduced a [working framework](https://github.com/AlignAGI/Alignment) for **AI alignment**, focusing on principles based on shared human values and iterative feedback to ensure inclusivity in AI development.
   - The discussion highlighted challenges in achieving consensus among stakeholders, with skepticism about the feasibility of aligning diverse interests.
- **Google's Gemini and Imagen Updates Discussed**: **Google's Gemini** and recent **Imagen** updates were evaluated, with users comparing their performance to existing models like **OpenAI's GPT-4**.
   - Participants noted that while models such as **Grok** are advancing, they still trail behind more established models like **ChatGPT** in capabilities.
- **Performance Gap Between GPT 4o and 4o-mini**: Users expressed frustrations over the **performance disparity** between **GPT 4o** and **GPT 4o-mini**, describing the mini version as **sleepwalking**.
   - The community observed a significant drop in response quality with **GPT 4o-mini**, affecting overall user experience.
- **Advantages of Local LLMs Explored**: Participants discussed the **benefits of local LLMs**, emphasizing their potential for a more customizable and flexible AI experience compared to large tech solutions.
   - Concerns were raised that major tech companies might prioritize productivity enhancements over creativity in AI interactions.
- **Refining Prompt Engineering Techniques**: Users shared strategies for **enhancing prompt engineering**, likening effective prompting to cooking from scratch and stressing the importance of clear instructions.
   - Discussions included developing a curriculum for prompt engineering and leveraging AI for coding assistance within IDEs.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Byte Latent Transformer Launches to Challenge Llama 3**: Meta launched the **Byte Latent Transformer (BLT)**, a tokenizer-free architecture that dynamically encodes Bytes into Patches, enhancing **inference efficiency** and robustness. See the [announcement](https://x.com/scaling01/status/1867573707247346003?s=46).
   - BLT models claim to match the performance of tokenization-based models like **Llama 3** while potentially reducing **inference flops** by up to **50%**. They trained the **Llama-3 8B** model on **1T tokens**, outperforming standard architectures using BPE.
- **Apollo LMMs Release Boosts Video Understanding**: The community discussed the recent update of the **Apollo LMMs**, which includes models focused on video understanding and multimodal capabilities. Early impressions suggest they perform well, sparking interest in their potential applications.
   - Members are optimistic about integrating Apollo models into existing workflows, enhancing **video analytics** and **multimodal processing** capabilities.
- **Open-source Coding LLMs Enhance Developer Efficiency**: Several open-source coding LLMs such as **Mistral Codestral**, **Qwen 2.5 Coder**, and **DeepSeek** were suggested, which can be integrated with IDEs like VS Code and PyCharm, along with extensions like [continue.dev](https://continue.dev).
   - These tools enable developers to enhance coding efficiency using local models, fostering a more customizable development environment.
- **Model Compression Techniques Leverage Communication Theory**: Discussion centered on how principles from **communication theory** are influencing the development of **LLMs**, particularly in gradient transmission during distributed training.
   - Members noted that **trading compute for bandwidth** could streamline processes, although combining techniques may be complex. The potential for optimizing data efficiency without impairing performance was also highlighted.
- **Fine-tuning Local LLMs Becomes More Accessible**: It was discussed that with tools like **unsloth** and **axolotl**, even older tech enthusiasts could potentially train models up to 8 billion parameters using **QLoRA**.
   - There are growing resources that make customization accessible for those willing to learn, expanding the capabilities for local model fine-tuning.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **SF Compute Integrates with OpenRouter**: OpenRouter announced the addition of **SF Compute** as a new provider, enhancing their service offerings.
   - This integration broadens options for users seeking diverse service integrations on the platform.
- **Qwen QwQ Sees 55% Price Reduction**: **Qwen QwQ** has undergone a significant **55% price cut**, aimed at attracting more users to its features.
   - Details are available on their [pricing page](https://openrouter.ai/qwen/qwq-32b-preview).
- **xAI Releases New Grok Models**: Two new **Grok models** from **xAI** were launched over the weekend, resulting in increased platform traffic.
   - Users can explore all the models at [OpenRouter's xAI page](https://openrouter.ai/x-ai).
- **OpenRouter API Wrapper Launched**: An API wrapper for OpenRouter, named [openrouter-client](https://www.npmjs.com/package/openrouter-client), was released two days ago.
   - The wrapper simplifies interactions with OpenRouter, featuring example code for implementation and configuration.
- **Hermes 3 405B Demonstrates Strong Performance**: **Hermes 3 405B** has shown effectiveness in creative tasks, with claims that it rivals **Claude 2.0** in quality.
   - However, discussions highlighted its slower performance in coding tasks compared to other models.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX/Flax Replaces TensorFlow for Enhanced Performance**: Members expressed frustrations with **TensorFlow**'s declining support, leading many to [switch to JAX/Flax](https://x.com/SkyLi0n/status/1867324080262885800). **JAX/Flax** offers improved performance and more robust features suitable for modern AI engineering.
   - The community praised **JAX/Flax** for its flexibility and better integration with current model architectures, citing smoother dependency management and enhanced computational efficiency.
- **Data Shuffling Reduces Model Bias from Recent Training**: Concerns were raised about models developing biases towards recently introduced training data. Members suggested [data shuffling](https://arxiv.org/abs/2412.06464) as a strategy to enhance **training fairness** and reduce bias.
   - Experiences with data homogenization strategies were shared, highlighting improvements in model performance and fairness through **randomized data ordering**.
- **Attention Mechanisms Outshine Kernel Methods**: A debate unfolded on whether **attention mechanisms** in Transformers can be equated with kernel methods. Members clarified that **attention**, specifically with **softmax**, extends beyond traditional kernel functionalities.
   - The discussion included mathematical distinctions and debated if attention fully utilizes kernel potentials, emphasizing the complexity of its operational context.
- **Non-Transformer Architectures Gain Momentum in AI Research**: Active research in **non-transformer architectures** was highlighted, with mentions of labs like **Numenta** and **AI2** releasing new model checkpoints that diverge from mainstream Transformer models.
   - Community members expressed interest in smaller labs pushing novel approaches, emphasizing the need for diverse model architectures in advancing AI capabilities.
- **lm_eval Successfully Integrates with VLLM**: A user shared the working method to get the **lm_eval harness** to function with **VLLM**, indicating a specific installation command. This process includes installing version 0.6.3 of VLLM to prevent issues with the evaluation harness.
   - Members discussed errors arising from VLLM, suggesting that the **internal API used by lm_eval** may have changed, and clarified version details to resolve [VLLM Version Confusion](https://github.com/EleutherAI/lm-evaluation-harness.git#egg=lm_eval).



---



## [Bolt.new / Stackblitz](https://discord.com/channels/364486390102097930) Discord

- **Bolt's Token Consumption Skyrockets**: Multiple users reported that **Bolt** is consuming tokens at an accelerated rate, with one user noting **5 million tokens** used without corresponding UI changes. This issue has been documented on [GitHub Issue #4218](https://github.com/stackblitz/bolt.new/issues/4218).
   - Members suspect a systemic bug and are forking projects to GitHub and running them on Replit as a workaround.
- **Struggles with Currency Updates**: Users face difficulties changing currency displays from **$ USD** to **INR**, even after locking the `.env` file, indicating a potential bug in Bolt's file handling.
   - This persistent issue has been reported across multiple channels, suggesting it's not isolated to browser-specific problems.
- **Supabase Integration Generates Excitement**: The anticipated **Supabase** integration with **Bolt** is generating enthusiasm, with early [video demonstrations](https://x.com/morganlinton/status/1868388127347523794?s=46) showcasing its capabilities.
   - Users are eager for updates and expect new functionalities to enhance their projects.
- **Concerns Over Token Costs and Subscriptions**: Users expressed concerns about the rapid consumption of tokens, especially post top-ups, and seek clarity on token management mechanics.
   - There is dissatisfaction with current expiration rules, and users advocate for a cumulative token system.
- **Guidance on React Native Development**: Discussions highlighted best practices for migrating web applications to mobile platforms using **React Native** and **Expo**.
   - Recommendations include shifting development to **Cursor** for better feature support.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Grok-2 Speeds Ahead with Aurora**: Grok-2 has been updated to run **three times faster** with improved **accuracy** and **multilingual capabilities**, now available for free on [X](https://x.com/i/grok/share/ieeeD20tYc40Ayi0dmFp4hrgh).
   - It introduces features like **web search**, **citations**, and a new image generator named **Aurora**, significantly enhancing user interactions.
- **Ilya Sutskever's NeurIPS Neoterics**: In his [NeurIPS 2024](https://youtu.be/1yvBqasHLZs?si=pQihchmQG3xoeCPZ) talk, Ilya Sutskever highlighted the plateau of scaling **LLMs** during pre-training and the shift towards **agentic behavior** and **tool integration** for future advancements.
   - The discussion included varied opinions on **data saturation** and the potential of **untapped video content** for **AI training**.
- **Google’s Veo 2 & Imagen 3: Media Magic**: Google introduced **Veo 2** and **Imagen 3**, featuring improved high-quality **video generation** and enhanced **image composition**, available in **VideoFX** and **ImageFX**.
   - These updates offer better **understanding of cinematography** and diverse **art styles** in generated content.
- **META’s Byte Latent Transformer**: META has released the **Byte Latent Transformer (BLT)**, a tokenizer-free architecture that dynamically encodes bytes into patches, enhancing **inference efficiency**.
   - BLT models match or outperform existing models like **Llama 3**, achieving significant reductions in **inference flops**.
- **OpenAI Rolls Out Voice Search for ChatGPT**: OpenAI announced the rollout of **Search in Advanced Voice mode** for **ChatGPT**, allowing users to obtain **real-time information** through **voice interactions**.
   - This feature results from collaboration between the **Search** and **multimodal product research teams** at OpenAI.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Multimodal Models Integration**: Members explored **multimodal models** that combine text, image, audio, and video, noting most solutions are accessible via [cloud services](https://lmstudio.ai/beta-releases) while highlighting **LM Studio's** current limitations in this area.
   - A key discussion point was the absence of fully multimodal LLMs for local setups, which has generated anticipation for upcoming model releases.
- **Limitations in Model Fine-tuning**: Users inquired about fine-tuning existing models with data exports to emulate specific grammar or tones, but were informed that **LM Studio** does not support fine-tuning.
   - As an alternative, it was suggested to use system prompts and example texts within the chat interface for temporary model adjustments.
- **Options for Uncensored Chatbots**: In search of **uncensored chatbots**, members were advised to utilize smaller models like [Gemma2 2B](https://huggingface.co/mustafaaljadery/gemma-2B-10M) or [Llama3.2 3B](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF) that can operate on CPU.
   - Various uncensored models available on [Hugging Face](https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF) were shared for deployment within local environments.
- **RAG Implementation and Document Handling**: The **Retrieval-Augmented Generation (RAG)** capabilities and document upload features in **LM Studio** were discussed as means to enhance contextual responses using local documents.
   - Users were informed that while all models support RAG, integrating web access or internet features requires custom API solutions, as detailed in the [LM Studio Docs](https://lmstudio.ai/docs/basics/rag).
- **GPU Selection for AI/ML Tasks**: The conversation emphasized that GPUs with larger VRAM, such as the **3090**, are preferable for **AI and machine learning tasks** due to their superior speed and capability.
   - Alternatives like the **4070ti** were mentioned, though some members noted that used **3090s** might offer better performance per dollar depending on local availability.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Reactor Enables Effective Face Swapping**: A user recommended the **Reactor extension** for face swapping in images, enabling users to successfully generate altered images after enabling Reactor and uploading the desired face image.
   - This method enhances **image manipulation capabilities** within **Stable Diffusion** workflows, allowing for seamless integration of different facial features.
- **Diverse Models for Stable Diffusion Discussed**: Discussions highlighted various **Stable Diffusion models**, emphasizing that the best choice depends on user requirements, with models like **Flux** and **SD 3.5** noted for prompt following and **Pixelwave** praised for artistic knowledge.
   - Participants shared experiences with different models to optimize **image generation quality** and **performance**, tailoring selections to specific project needs.
- **Seeking Comprehensive Stable Diffusion Learning Resources**: Users sought out extensive **courses and tutorials** for **Stable Diffusion**, particularly focusing on its integration with **Automatic1111**, with suggestions pointing to series on platforms like YouTube and dedicated online resources.
   - These resources aim to enhance users' understanding and proficiency in utilizing **Stable Diffusion's** advanced features.
- **Optimizing Image Quality with Upscaling Tools**: Users requested recommendations for effective upscalers compatible with **Stable Diffusion-generated images**, discussing specific tools or extensions that improve **image resolution and quality**.
   - Enhanced **upscaling techniques** were debated to achieve better **visual fidelity** in generated images.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **LiquidAI Secures $250M Funding**: LiquidAI announced a significant **$250M Series A funding** round led by **AMD Ventures**, aiming to scale its **Liquid Foundation Models (LFMs)** for enterprise AI solutions.
   - Concerns were raised about their hiring practices, with discussions surrounding potential talent challenges and the possibility that **LiquidAI's** size may impede acquisition opportunities.
- **ChatGPT Enhances Search with Memory**: ChatGPT is introducing **memory features** in its search functionality, allowing the model to utilize memories to refine search responses for improved relevance.
   - Users expressed disappointment over the exclusion of personalized search in the update, anticipating future enhancements including potential API integrations.
- **DeepMind Launches Veo 2 and Imagen 3**: **DeepMind** unveiled **Veo 2**, a new video generation model, and the upgraded **Imagen 3**, enhancing realistic content generation from prompts.
   - Early feedback praised **Imagen 3's** performance, highlighting **DeepMind's** competitive edge over other major players like **OpenAI** within the tech community.
- **OpenAI Whistleblower Incident**: OpenAI whistleblower **Suchir Balaji** was found dead in his apartment, with authorities reporting the death as a suicide and ruling out foul play.
   - **Balaji** was known for raising concerns about **OpenAI's** use of copyrighted material for training **ChatGPT** shortly after his departure from the company.
- **Apollo Video LLMs Challenge Competitors**: Meta's **Apollo** series of video LLMs demonstrates strong performance, comparable to **llava-OV** and **Qwen2-VL**.
   - Discussions highlighted **Apollo's** use of **Qwen2.5** as its underlying LLM instead of the more expected **Llama**, sparking questions about model selection for optimal performance.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Subscriptions Expand Offerings**: **Perplexity Pro** now offers gift subscriptions for **1, 3, 6, or 12-month** periods, enabling users to share enhanced features like searching **3x as many sources** and accessing the latest AI models. Details and purchase options are available [here](https://perplexity.supply/shop/perplexity-subscription).
   - The **Campus Strategist program** is expanding internationally, allowing students to apply for the Spring 2025 cohort by December 28, with exclusive merch and activation opportunities detailed [here](https://www.perplexity.ai/campus-strategists).
- **Custom Web Sources Launched in Spaces**: Perplexity AI introduced [custom web sources](https://x.com/perplexity_ai/status/1867615710391746836?s=46) in Spaces, enabling users to tailor their searches by selecting specific websites, thus enhancing relevance for diverse use cases.
   - This feature allows engineers to optimize search queries within **Spaces**, ensuring that results are more aligned with specialized requirements.
- **Perplexity API Faces URL and Access Challenges**: Users report that the **Perplexity API** returns source citations as plain text numbers like [1] without URLs, although some managed to retrieve URLs by explicitly requesting them.
   - Additionally, there are difficulties in obtaining news headlines via the API and accessing support through the provided email, indicating potential stability and usability issues.
- **Concerns Over Perplexity API Model Performance**: Multiple users indicated that recent **model updates** have led to performance degradation, particularly noting that **Claude 3.5** is less effective compared to its free counterpart.
   - There is a lack of transparency regarding model switches, which affects the perceived quality and reliability of the API service.
- **Google Releases Gemini 2.0**: Google has unveiled **Gemini 2.0**, marking significant advancements in **AI capabilities**, which has sparked discussions around [problem movement](https://www.youtube.com/embed/nQTAbz1eDco).
   - Participants in the discussion expressed enthusiasm about the updates and their potential impact on the AI field.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R7B Model Speeds Ahead**: The [Cohere Command R7B 12-2024](https://cohereforai-c4ai-command.hf.space/models/command-r7b-12-2024) model is now operational, optimized for reasoning and summarization tasks, boasting enhanced speed and efficiency.
   - Community benchmarks highlighted on [Nils Reimers' Twitter](https://x.com/Nils_Reimers/status/1868065732149571701) show **Command R7B** outperforming models like **Llama 8B**, with significant improvements in response time.
- **Rerank vs Embed: Feature Breakdown**: Discussions clarified that **Rerank** reorders documents based on query relevance, whereas **Embed** transforms text into numerical vectors for NLP applications.
   - API updates for **Embed** now support 'image' input types, expanding its applicability beyond text-based tasks.
- **API Schema Overhaul in v2**: The migration from API v1 to v2 lacks detailed documentation on schema changes for new endpoints, leaving users uncertain about specific updates.
   - Engineers are investigating the existing [migration resources](https://discord.com/events/954421988141711382/1308148058894110750/1318261358592000000) to provide clarity on the new API structures.
- **Seeking Sponsors for Code Wizard Hackathon**: **Akash** announced the upcoming **Code Wizard** hackathon hosted by SRM Institute in February 2025, targeting students and tech enthusiasts to tackle real-world problems.
   - The event is actively seeking sponsors to support and gain exposure, aiming to foster innovative solutions within the developer community.
- **AI Enhances Contract Clause Review**: **Eyal** is developing a proof of concept using **Cohere** to automatically identify and suggest modifications in contract clauses.
   - Feedback is sought on strategies like defining specific clause types or utilizing a change database to improve the AI's effectiveness in contract analysis.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo RSA Crypto Development**: A member initiated the development of a basic **RSA crypto** implementation in **Mojo**, showcasing their progress.
   - The project generated mixed reactions, highlighting the community's enthusiasm and constructive feedback.
- **Prime Number Generation Optimizations**: The prime number generation script achieved a peak performance of **1.125 seconds** and, after optimizations, now exceeds **50,000 UInt32 primes per second** using **SIMD instructions**.
   - These enhancements maintain a low memory footprint, with the application consuming less than **3mb** during operation.
- **Custom Mojo Kernels**: [Custom Mojo Kernels](https://github.com/cassioneri/teju_jagua) have been released, allowing acceptance of any input types, although early versions may crash due to type mismatches.
   - Developers remain confident in the API's future robustness, anticipating improved stability as the implementation matures.
- **Networking Performance in Mojo**: Discussions favored using **QUIC** over **TCP** for **Mojo** applications to reduce latency.
   - Avoiding TCP overhead is seen as essential for achieving efficient **Mojo-to-Mojo** communication in modern network environments.
- **Database Planning in MAX**: A developer plans to implement **database query planning** and execution within **MAX**, leveraging new custom kernel features.
   - This initiative indicates a push for more robust handling of complex data operations within the **Mojo** ecosystem.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Hackathon Deadline Looms for LLM Agents MOOC**: The **LLM Agents MOOC Hackathon** submission deadline is **December 17th at 11:59pm PST**, urging participants to finalize and submit their projects on time.
   - Participants are encouraged to seek last-minute assistance in the designated channel to ensure all submissions meet the requirements.
- **Transitioning to Google Forms for Hackathon Entries**: Submissions for the hackathon have shifted from **Devpost** to **[Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform)** to streamline the submission process.
   - Participants must ensure they use the correct form link to avoid any submission issues before the deadline.
- **Certificate Notifications Scheduled for Late December**: **Certificate notifications**, indicating pass or fail statuses, will be distributed **late December through early January** based on participants' tiers.
   - This timeline addresses recent inquiries and sets clear expectations for when participants can expect their certification status.
- **Issues with OpenAI Credit Submissions**: Some members reported not receiving **OpenAI credits** despite submitting their organization IDs before the **November 25th** deadline.
   - Community members suggested verifying account credit balances as notifications may not have been dispatched properly.
- **Emphasizing Safety Alignment in AI Research Agents**: A member emphasized the importance of **safety alignment** in **AI Research Agents** and shared a relevant [AI Research resource](https://airesearch.js.org).
   - This highlights the community's focus on ensuring safety protocols are integral to the development of AI research agents.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune v3.9 Simplifies Type Hinting**: The update to [Torchtune v3.9](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py#L780) allows users to replace `List`, `Dict`, and `Tuple` with default builtins for type hinting.
   - This adjustment is welcomed by the community to streamline Python code, enhancing developer productivity.
- **Generative Verifiers Boost LLM Performance**: The paper titled [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240v1) introduces *Generative Verifiers (GenRM)*, trained using the next-token prediction objective to seamlessly integrate validation and solution generation.
   - This method supports instruction tuning and enables chain-of-thought reasoning by utilizing **additional inference-time compute** for enhanced verification results.
- **Gradient Normalization Challenges in Distributed Training**: Discussions highlighted concerns about scaling factors for normalization during the backward pass in distributed training, suggesting it should be `world_size / num_tokens` to manage variability in token counts.
   - This issue could complicate gradient calculations due to padding and indexing differences, prompting advocacy for a potential PR to address inconsistencies.
- **Scaling Test Time Compute Strategies Explored**: A [Hugging Face blog post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) discusses strategies to scale test-time compute for large models, focusing on performance optimization without compromising results.
   - The post outlines methodologies to enhance compute efficiency while maintaining model output integrity.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Optimizing BEAM Configuration for Kernel Search**: Members discussed various **BEAM** settings for **kernel search**, highlighting that **BEAM=1** denotes greedy search, which is less effective. The recommended starting points are **BEAM=2** or **3** for balanced performance, as detailed in the [documentation](https://docs.tinygrad.org/env_vars/).
   - Enhancements to the **kernel search experience** focus on improving both **compile time** and **kernel execution time**. Members are interested in available benchmarks and recommend utilizing **BEAM=2**, especially with **JIT compilation**.
- **New Gradient API Simplifies Gradient Handling**: George Hotz announced the merger of the new **gradient API**, which allows for simplified gradient handling: `weight_grad, bias_grad = loss.gradient(weight, bias)` without requiring `zero_grad` or `loss.backward`.
   - This API differs from traditional frameworks like **PyTorch** and **JAX**, potentially streamlining optimizer steps with `optim.step(loss)`, as mentioned in the [tweet](https://x.com/__tinygrad__/status/1867745748118544411).
- **Tinygrad Porting Projects and Backend Support Debated**: Plans to port the **fish-speech** project to Tinygrad were announced, aiming to enhance Tinygrad's capabilities. The project is hosted on [GitHub](https://github.com/fishaudio/fish-speech).
   - Members debated supporting both **x86** and **arm64 backends** for Tinygrad, considering maintenance of performance amid resource constraints.
- **ShapeTracker Explainer and Tutorials Expanded**: An improved **ShapeTracker Explainer** has been released, available [here](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md), providing deeper insights into its workings.
   - The [tinygrad-notes](https://github.com/mesozoic-egg/tinygrad-notes) repository calls for contributions to tutorials and resources, encouraging community participation.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex RAG in 5 Lines**: TylerReedAI shared a detailed [tutorial](https://t.co/v5yljbVw4d) on building a **RAG application** using just **5 lines of code**, covering data loading and indexing.
   - The tutorial emphasizes the ease of integrating **query** and **chat engines** into your workspace.
- **Agentic Compliance Workflows**: A new [tutorial](https://t.co/9SjfXRWdmF) introduces a method to build an **agentic workflow** that ensures **contract compliance** by analyzing clauses against **GDPR** guidelines.
   - It breaks down how to parse vendor contracts to maintain compliance effectively, simplifying contract management.
- **Contextual Retrieval Meets LlamaIndex**: A user implemented **Anthropic's contextual retrieval** in **LlamaIndex** and shared their [GitHub repository](https://github.com/cklapperich/Eidetic/) for others to review.
   - They expressed interest in contributing this robust implementation as a PR, highlighting its handling of edge cases.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Folder Creation Issues with Incorrect Indentation**: A member highlighted that the tool is *not creating folders* and the produced code has *wrong indentation* for easy copying and pasting, questioning if a different environment than cmd should be used.
   - This issue suggests potential bugs in the folder creation functionality and code formatting processes within the current setup.
- **API Responses Limit at macOS Monterey**: A user reported that after installing the app on **macOS Monterey**, they receive no API responses and hit the free token limit after only **two actions**.
   - This indicates possible integration or usage issues specific to macOS Monterey, potentially affecting API availability.
- **Enhancing Billing Tracking for Litellm**: A user inquired about connecting **OI** to a Litellm proxy server to effectively track billing and usage for the integrated Litellm package.
   - They are exploring ways to enable comprehensive billing tracking within the **Litellm** integration.
- **Recommendations for Japanese Learning Apps**: A member sought good apps for learning **Japanese**, prompting another user to humorously suggest they might be in the *wrong Discord server*.
   - This exchange underscores a need for specialized resources or channels focused on language learning within the guild.
- **Local OS Deployment Options**: A user asked about the possibility of using the OS locally, indicating interest in local setup solutions.
   - This query points towards discussions on potential deployment or hosting configurations for local environments.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Optimizing Claude Sonnet Prompt with DSpy**: A user discovered **DSpy** while searching for ways to optimize their **Claude Sonnet** prompt and bookmarked a specific [Jupyter notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/vlm/mmmu.ipynb).
   - They mentioned that the notebook was recently moved to an outdated examples folder, raising questions about its relevance.
- **Updating Outdated DSpy Examples**: Another member advised that the contents of the outdated examples folder in **DSpy** should be used cautiously until they are revamped, indicating potential unreliability.
   - They also noted that efforts are underway to update these examples, potentially improving their usefulness.



---



## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **APOLLO Optimizer Enhances Memory Efficiency**: The new **APOLLO optimizer** reduces memory usage to **1.6G** while achieving optimal perplexity during **LLaMA 7B training**, compared to **13G** with **8-bit Adam**.
   - An independent **Julia implementation** confirmed APOLLO’s effectiveness in optimizing memory and training efficiency, as detailed in the [post](https://bsky.app/profile/benjmurrell.bsky.social/post/3lcyfrf5b7k2u).
- **LLM Training Faces Memory Constraints with AdamW**: Large language models encounter significant memory issues when using the **AdamW optimizer**, often necessitating costly hardware or smaller batch sizes during training.
   - Traditional memory-efficient optimizers involve **SVD operations** or performance trade-offs, but **APOLLO** introduces a novel method to address these limitations.
- **Ongoing Talks on Multi-turn KTO**: Discussions highlighted **multi-turn KTO**, although specific details and updates were not provided.
   - Community members expressed interest in the potential capabilities and integration of this method within the LLM framework.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **VAE Embedding Improves Progressive Tokenization**: The discussion focused on **progressive tokenization** utilizing a **zero-tree ordering** of **DWT coefficients** derived from a **VAE embedding**. An attached [video](https://cdn.discordapp.com/attachments/823813160075132991/1317573114854637680/level_5_wavelet_db5_clip_value_2.0_patch_size_1.mp4) demonstrated the technique in action.
   - **Level 5 wavelet** transformations were analyzed for their impact on tokenization effectiveness, highlighting practical applications and implications for future model enhancements.
- **Byte Latent Transformer Patches Outperform Tokens**: The publication [Byte Latent Transformer Patches: Scale Better than Tokens](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/) details a new NLP approach where **byte latent transformer patches** demonstrate better scalability compared to traditional tokens.
   - This advancement incited discussions on enhancing language modeling **effectiveness** and **efficiency** in various applications.
- **Level 5 Wavelet Transform Boosts Tokenization**: **Level 5 wavelet** transformations were examined for their role in improving tokenization effectiveness within current methodologies.
   - The analysis included exploring practical applications and future implications for model performance, referencing the [attached video](https://cdn.discordapp.com/attachments/823813160075132991/1317573114854637680/level_5_wavelet_db5_clip_value_2.0_patch_size_1.mp4).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **RAG Extravaganza: Building with SQLite-Vec & LlamaFile**: Tomorrow's event focuses on creating an **ultra-low dependency Retrieval Augmented Generation (RAG)** application using **sqlite-vec** and **llamafile**, with **bare-bones Python** and no additional dependencies or installations.
   - **Alex Garcia** will lead the session, providing attendees with a straightforward approach to building RAG applications.
- **Holiday Huddle: Final RAG Session Before Break**: The **final gathering** for December before the holiday break emphasizes the importance of participation before the year-end.
   - Participants are encouraged to **join the session** as a prelude to the holiday season and gain insights into **RAG development**.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM Releases Function Calling Results**: [BFCL-Result](https://github.com/HuanzhiMao/BFCL-Result) repository for **Gorilla LLM's** Berkeley Function Calling has been updated.
   - The [BFCL-Result](https://github.com/HuanzhiMao/BFCL-Result) repository is now available for review.
- **Gorilla LLM Releases Function Calling Results**: [BFCL-Result](https://github.com/HuanzhiMao/BFCL-Result) repository for **Gorilla LLM's** Berkeley Function Calling has been updated.
   - The [BFCL-Result](https://github.com/HuanzhiMao/BFCL-Result) repository is now available for review.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Codeium / Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1317290753772228778)** (1 messages): 

> `Discord Challenge Winners, YouTube Video Submissions, Windsurf Pro Tier Rewards` 


- **This Week's Discord Challenge Winners Announced**: Congratulations to the winners of this week's Discord Challenge: <@254550955427627008> and <@1219755748960243743> who showcased impressive submissions.
   - They can claim their reward of **3 months of pro tier Windsurf** by DMing the host.
- **Winning Videos Available to Watch**: Check out the winning entries: **Singularia** from <@254550955427627008> ([watch here](https://www.youtube.com/watch?v=kO-zI0CYJ2w)) and **Sales Prompt Creator** from <@1219755748960243743> ([watch here](https://www.youtube.com/watch?v=7gA2IouD-XU)).
   - Both videos highlight creativity and skill, making them must-see content for the community.
- **Join the Ongoing Windsurf Challenge**: Participants can join the rolling Windsurf Discord challenge by following the [rules and submission link](https://discord.com/channels/1027685395649015980/1027688115592237117/1306427389991059508).
   - This ongoing challenge provides an opportunity for community members to showcase their talents.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=kO-zI0CYJ2w"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=7gA2IouD-XU"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Codeium / Windsurf ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1317235016635387914)** (212 messages🔥🔥): 

> `Windsurf Features and Issues, User Feedback on Flow Action Credits, Account Management and Support, AI Behavior and Code Changes, Integration with Other Tools` 


- **Windsurf has features and ongoing issues**: Users reported issues with Windsurf not saving files or modifying them unexpectedly during work, leading to frustration.
   - Some junior developers expressed confusion over features not working as intended and emphasized the need for clarity in the documentation.
- **High consumption of Flow Action Credits**: Several users noted that they are burning through Flow Action Credits quickly, with one user mentioning exhausting 1k credits within 24 hours.
   - Suggestions included breaking tasks into smaller pieces, although some users mentioned that this approach wasn't effective for their needs.
- **Difficulties with account management and support response times**: Users experienced frustration with slow support responses when raising tickets regarding issues such as Pro account activation and credit management.
   - Feedback indicated a potential need for better communication from the support team regarding ticket progress.
- **Frustrations with AI's code implementation**: Some users articulated dissatisfaction with the AI modifying their code unexpectedly despite setting parameters to avoid such changes.
   - A discussion emerged around strategies for prompting better responses from the AI to avoid code errors.
- **Integration inquiries and feature requests**: Users expressed interest in increasing the monthly allotment of Flow Action Credits and other features such as file locking.
   - Additionally, discussions included potential integration with existing tools like NVIDIA's RAPIDS for enhanced data handling capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NVIDIAAIDev/status/1868778156347339033">Tweet from NVIDIA AI Developer (@NVIDIAAIDev)</a>: 👀 RAPIDS cuDF accelerates #pandas up to 150x with zero code changes. Now you can continue using pandas as your dataset size grows into gigabytes. ⚡ ➡️ Jupyter Notebook to try the demo: http://nvda.ws...</li><li><a href="https://imgur.com/gallery/VQ2LV35">Totally normal image - Album on Imgur</a>: no description found</li><li><a href="https://ternarysteganography.vercel.app">Ternary Image Steganography</a>: no description found</li><li><a href="https://ternarykeyexchange.vercel.app/">Ternary Key Exchange</a>: no description found</li><li><a href="https://www.aperisolve.com/">Aperi'Solve</a>: no description found</li><li><a href="https://docs.codeium.com/windsurf/usage">Paid Plan and Credit Usage - Codeium Docs</a>: no description found</li><li><a href="https://github.com/favourablegroup/ternarysteganography">GitHub - favourablegroup/ternarysteganography</a>: Contribute to favourablegroup/ternarysteganography development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=qadmkq_d_co&list=LL&index=4&t=2s&pp=gAQBiAQB"> - YouTube</a>: no description found</li><li><a href="https://codeium.com/careers">Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://codeium.com/blog/pricing-windsurf">Plans and Pricing Updates</a>: Some changes to our pricing model for Cascade.</li><li><a href="https://link.springer.com/chapter/10.1007/978-3-642-22786-8_30">Symmetric Encryption Using Sierpinski Fractal Geometry</a>: Symmetric cryptography uses the same secret key for encryption and decryption. A desirable property of symmetric encryption is termed as avalanche effect by which two different keys produces different...
</li>
</ul>

</div>
  

---


### **Codeium / Windsurf ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1317219537954934868)** (609 messages🔥🔥🔥): 

> `Windsurf Issues, AI and Dependency, Codeium vs. Gemini, MCP and Function Calling, Ruff Linter and Formatter` 


- **Windsurf Experiences and Bugs**: Users reported various issues with Windsurf, including freezing and problems with actions and chat windows not functioning properly.
   - Some suggested reinstalling or refreshing settings to resolve ongoing bugs.
- **AI and User Dependency Concerns**: Discussions arose about the increasing dependency on AI tools like Claude and the potential risks of relying on them for coding tasks.
   - Users expressed concerns about the implications of depending solely on AI, highlighting the need for personal discipline and skill retention.
- **Comparison between Codeium and Gemini 2.0**: Users compared Codeium's capabilities with Gemini 2.0, noting that while Gemini may offer better performance in coding tasks, it lacks some features of Claude.
   - Benchmarks showed varying opinions on which tool performed better based on specific use cases.
- **MCP and Function Calling Capabilities**: The Model Context Protocol (MCP) was discussed in relation to creating standardized structures for function calls across different stacks.
   - Users proposed ideas for using tools like Playwright and MCP for enhancing GUI testing and interactions.
- **Ruff Linter and Markdown Formatting**: There was a conversation about using Ruff as a linter and formatter for Python, with tips on excluding certain files in configuration.
   - Users shared insights on maintaining clean code and integrating formatting tools effectively in their projects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://astral.sh/blog/the-ruff-formatter">The Ruff Formatter: An extremely fast, Black-compatible Python formatter</a>: Ruff&#x27;s formatter is over 30x faster than existing tools, while maintaining &gt;99.9% compatibility with Black.</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags">Use XML tags to structure your prompts - Anthropic</a>: no description found</li><li><a href="https://superuser.com/questions/177041/what-is-the-equivalent-of-mac-os-x-spaces-for-windows">What is the equivalent of Mac OS X Spaces for Windows?</a>: I am looking for a utility that does the same task done by Spaces for Mac OS X.&#xA;For who doesn&#x27;t know it, it&#x27;s a utility that allows you to create virtual screens in order to not have all...</li><li><a href="https://superuser.com/questions/177041/what-is-the-equivalent-of-mac-os-x-spac">What is the equivalent of Mac OS X Spaces for Windows?</a>: I am looking for a utility that does the same task done by Spaces for Mac OS X.&#xA;For who doesn&#x27;t know it, it&#x27;s a utility that allows you to create virtual screens in order to not have all...</li><li><a href="https://youtu.be/ujnLJru2LIs?si=8Cn_9t_2Rlyfo8GT"> - YouTube</a>: no description found</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers?tab=readme-ov-file#tutorials">GitHub - punkpeye/awesome-mcp-servers: A collection of MCP servers.</a>: A collection of MCP servers. Contribute to punkpeye/awesome-mcp-servers development by creating an account on GitHub.</li><li><a href="https://youtu.be/ujnLJru2LIs?si=8C">Prompt Engineering Master Class for ENGINEERS with Ollama and LLM (Q4 2024 Update)</a>: 🚀 Think Prompt Engineering is STILL just a buzzword? Good Sir, that is BEYOND incorrect. As we approach 2025, its 100% clear: mastering prompt engineering p...</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers?tab=read">GitHub - punkpeye/awesome-mcp-servers: A collection of MCP servers.</a>: A collection of MCP servers. Contribute to punkpeye/awesome-mcp-servers development by creating an account on GitHub.</li><li><a href="https://github.com/jhgoodwin/FakeDbProvider">GitHub - jhgoodwin/FakeDbProvider: A fake provider for System.Data.Common.DbConnection and related classes.</a>: A fake provider for System.Data.Common.DbConnection and related classes. - jhgoodwin/FakeDbProvider</li><li><a href="https://github.com/orgs/modelcontextprotocol/discussions/88">What&#39;s the difference between MCP and vector database? · modelcontextprotocol · Discussion #88</a>: it&#39;s been a while and I can&#39;t figure it out
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1317239702063022101)** (96 messages🔥🔥): 

> `Notebook LM Podcast Features, Customizing AI Outputs, Using Different Languages in AI, Creating Engaging Content with AI, AI and the Turing Test` 


- **Notebook LM Podcast Features Explored**: The latest features of Notebook LM have been discussed, including customizations and interactive functionalities that enhance the user experience.
   - Members shared links to podcasts showcasing these features, asserting that the application is changing the landscape of audio content.
- **Customizing AI Outputs for Unique Styles**: Users highlighted the importance of good prompting and custom functions to tailor AI outputs, which can result in varied tones and styles.
   - A shared [YouTube video](https://youtu.be/aG0ixD3OY80?feature=shared) provided tips on effective prompting techniques for artistic results.
- **Bilingual and Multilingual Uses of AI Tools**: Questions arose regarding how to utilize Notebook LM in different languages, with suggestions on instructing the AI to respond in specific languages.
   - Users shared methods to prompt the AI for multilingual outputs, emphasizing the necessity of proper configuration.
- **Creating Engaging Content with AI**: Conversations emerged around generating captivating audio narratives and content using AI, which seemed to resonate well with listeners.
   - A variety of content styles, including those mimicking famous figures and ASMR tones, were experimented with to enhance audience engagement.
- **Exploring AI's Capability in Passing the Turing Test**: Members discussed the challenges AIs face in passing the Turing Test and the importance of conversational tone adaptation.
   - Experiments were shared, showcasing how different character moods can influence AI's conversational style and its perceived intelligence.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com/notebook/24b9048f-48be-417d-96f5-d288435fcc24/audio">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=aG0ixD3OY80"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/aG0ixD3OY80?feature=shared"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/jTVIOhuNy3Q?feature=shared"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/ytcHj-EllWo?feature=shared"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=yWRCpQBpd-k"> - YouTube</a>: no description found</li><li><a href="https://www.fxguide.com/fxpodcasts/zap-andersson-exploring-the-intersection-of-ai-and-rendering/">Zap Andersson: Exploring the intersection of AI and rendering</a>: Zap Andersson shares his tips and tricks gleaned from testing AI tools for his bizarre YouTube series: UNREAL MYSTERIES
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1317219744427937874)** (613 messages🔥🔥🔥): 

> `NotebookLM new features, NotebookLM Plus, Interactive mode, Podcast generation, Language settings` 


- **NotebookLM Plus Rollout Status**: Users are currently experiencing a slow rollout of the new NotebookLM Plus features, with some having access while others do not, particularly across different Google accounts.
   - There's an anticipation for general availability, with early 2025 being the target for Google One Premium users.
- **User Experiences with New Features**: Users have mixed experiences with the interactive audio overview feature, where some report slower response times and a decrease in perceived engagement from the AI hosts.
   - Suggestions to improve responsiveness are acknowledged, indicating that ongoing adjustments are being made to enhance user experience.
- **Source Limit Discussion**: There is a discussion on the increase in source limits for the free version of NotebookLM, now set to 300 sources, while users express curiosity about how this limit is managed by the model.
   - Users are also contemplating strategies for gathering enough sources to utilize this feature effectively.
- **Language Settings for French Speakers**: A French-speaking user inquired about changing language settings in NotebookLM, indicating that prompts were being responded to in French instead of English.
   - It was suggested users may need to adjust their Google account language settings to match their desired response language.
- **Feature Requests and Improvements**: Users expressed interest in various improvements for podcasts, such as adding sound bites and increasing voice control options.
   - The community encourages submitting feedback and engaging with certain requests to improve future iterations of NotebookLM features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/traviscline/status/1868093820581343740?s=46">Tweet from tmc (the/acc) (@traviscline)</a>: A little preview of the interactive notebooklm mode. Discussing the notebooklm cli!#notebooklm</li><li><a href="https://book-a-painter.com/">no title found</a>: no description found</li><li><a href="https://elevenreader.io/app/reader/genfm/e2771eb8df2252e96cbe43f47a2bf4b023cf392e62a0e1af77d4ad7e73dd5562/u:o1kc1ElI1uo7j4h1ux4J">GenFM by ElevenReader: Unlocking Financial Freedom Through Tax Knowledge</a>: Listen to a preview of this GenFM podcast or download the app to play the full episode and create your own.</li><li><a href="https://aistudio.google.com/">Google AI Studio</a>: Google AI Studio is the fastest way to start building with Gemini, our next generation family of multimodal generative AI models.</li><li><a href="https://support.google.com/notebooklm/answer/15678219?visit_id=638697853454981673-3976970542&p=plus&rd=1">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://streamable.com/4thv4x">Watch deepdive-google_U8gkTzEC | Streamable</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://pastebin.com/gDFFrr4M">Pastebin.com - Burn After Read paste</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://ai.google.dev/gemini-api/docs/available-regions">no title found</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/14276570?hl=en">Community - NotebookLM Help</a>: no description found</li><li><a href="https://blog.google/technology/google-labs/notebooklm-new-features-december-2024/">NotebookLM gets a new look, audio interactivity and a premium version</a>: NotebookLM is introducing new features, and a premium version called NotebookLM Plus.</li><li><a href="https://apps.google.com/supportwidget/articlehome?hl=en&article_url=https%3A%2F%2Fsupport.google.com%2Fa%2Fanswer%2F9212585%3Fhl%3Den&assistant_id=generic-unu&product_context=9212585&product_name=UnuFlow&trigger_context=a">no title found</a>: no description found</li><li><a href="https://youtu.be/nqDXv6dnlls)"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Xv4_ToKF66U&t=3454s"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/GMe2JoTymRY?si=vyo8lJ6rDjwN1zZJ"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/gv92pUahxVQ)"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/CzBBhytDzM4?si=VFJM_ZN9"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/CzBBhytDzM4?si=VFJM_ZN918XpOK33"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=EZGskiGkkSA"> - YouTube</a>: no description found</li><li><a href="https://notebooklm.google/plus/terms">Google NotebookLM | Note Taking &amp; Research Assistant Powered by AI</a>: Use the power of AI for quick summarization and note taking, NotebookLM is your powerful virtual research assistant rooted in information you can trust.</li><li><a href="https://cloud.google.com/terms">no title found</a>: no description found</li><li><a href="https://support.google.com/a/answer/14700766?hl=en&co=DASHER._Family%3DBusiness-Enterprise">Compare Gemini for Google Workspace add-ons - Business / Enterprise - Google Workspace Admin Help</a>: no description found</li><li><a href="https://support.google.com/a/answer/14700766?hl=en&co=DASHER._Family%3DBusiness-Enterprise#other">Compare Gemini for Google Workspace add-ons - Business / Enterprise - Google Workspace Admin Help</a>: no description found</li><li><a href="https://support.google.com/a/answer/14700766?hl=en&co=DASHER._Family%3DBusiness-Enterprise#availability">Compare Gemini for Google Workspace add-ons - Business / Enterprise - Google Workspace Admin Help</a>: no description found
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1317221075238785065)** (884 messages🔥🔥🔥): 

> `Cursor IDE performance, AI model comparisons, Social media project development, Cursor integrations, Chat management issues` 


- **Cursor IDE performance issues**: Users reported sluggishness in Cursor IDE, especially when working on applications for extended periods, prompting discussions about needing to reset or clear chat history.
   - Some suggested creating new chat sessions to alleviate performance problems, aiming for more efficient workflows.
- **Comparison between AI models**: Participants discussed the pros and cons of different AI models, such as Cursor's agent vs. Gemini 1206, highlighting their respective capabilities and performance.
   - Users noted that while Cursor maintains a user-friendly interface, Gemini offers strong performance in coding tasks, making it a valuable tool alongside Cursor.
- **Development of a social media platform**: Several users expressed interest in building a social media platform, discussing the necessary backend structures and potential frameworks for implementation.
   - It was emphasized that creating such platforms requires understanding CRUD operations and managing database relationships, making use of tools like Cursor for efficiency.
- **Cursor integrations with other tools**: There were suggestions for Cursor to integrate with other platforms like Supabase and Bolt to enhance its functionality and simplify workflows for users.
   - Users discussed the advantages of such integrations and how they could streamline the development process.
- **Feedback on chat management**: Feedback about Cursor's chat management revealed frustrations over the loss of context and previous messages when messages are edited.
   - Users proposed improvements, like retaining chat history after edits, similar to features in other platforms like ChatGPT and Claude.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NVIDIAAIDev/status/1868778156347339033">Tweet from NVIDIA AI Developer (@NVIDIAAIDev)</a>: 👀 RAPIDS cuDF accelerates #pandas up to 150x with zero code changes. Now you can continue using pandas as your dataset size grows into gigabytes. ⚡ ➡️ Jupyter Notebook to try the demo: http://nvda.ws...</li><li><a href="https://docs.cursor.com/get-started/usage#premium-models">Cursor - Build Software Faster</a>: no description found</li><li><a href="https://x.com/skcd42/status/1867561917159755942?s=19">Tweet from skcd (@skcd42)</a>: CodeStory agent is now SOTA on swebench-verified with 62.2% resolution rate.We did this by scaling our agent on test time inference and re-learning the bitter lesson.Sonnet3.5(new) was the only LLM we...</li><li><a href="https://aistudio.google.com/">Google AI Studio</a>: Google AI Studio is the fastest way to start building with Gemini, our next generation family of multimodal generative AI models.</li><li><a href="https://www.cursor.com/pricing">Pricing | Cursor - The AI Code Editor</a>: Choose the plan that works for you.</li><li><a href="https://cursor.com/settings">Settings | Cursor - The AI Code Editor</a>: You can manage your account, billing, and team settings here.</li><li><a href="https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221">How to do `Fix in Composer` and `Fix in Chat` actions from keyboard</a>: These 2:     I could not find it in settings.</li><li><a href="https://letmegooglethat.com/?q=Warp&l=1">Warp</a>: no description found</li><li><a href="https://x.com/hive_ech">Tweet from FxTwitter / FixupX</a>: Sorry, that user doesn't exist :(</li><li><a href="https://x.com/vadi_ms/status/1867395672418529623">Tweet from Vadims (@vadi_ms)</a>: I analysed http://bolt.new and discovered how to achieve the same high-quality design using Cursor.First image is from Bolt, second is from Cursor, using the same prompt.3-step guide below⤵️:</li><li><a href="https://marketplace.visualstudio.com/items?itemName=SpecStory.specstory-vscode">SpecStory&#32;(Cursor&#32;Extension)&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;(Cursor&#32;Extension)&#32;Capture,&#32;search&#32;and&#32;learn&#32;from&#32;every&#32;AI&#32;coding&#32;journey</li><li><a href="https://v0.dev/">v0 by Vercel</a>: Chat with v0. Generate UI with simple text prompts. Copy, paste, ship.</li><li><a href="https://x.com/hive_echo/status/1865598500060508183">Tweet from echo.hive (@hive_echo)</a>: Coming to a Cursor near you(soon...)⚡ Yolo mode (automatic command execution)🤝 Unification (chat and composer work as one)</li><li><a href="https://openrouter.ai/">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://youtube.com/shorts/8WMk8E4KD5Q?si=8BJKbqipxOdOY7gm">Fixed Live Server Problem In Visual Studio Code!#vscode #liveserver</a>: Fixed Live Server Problem In Visual Studio Code!Hey everyone! Welcome back to another quick and snappy YouTube Short! Today, we&#39;re diving into the world of w...</li><li><a href="https://github.com/mastodon/mastodon">GitHub - mastodon/mastodon: Your self-hosted, globally interconnected microblogging community</a>: Your self-hosted, globally interconnected microblogging community - mastodon/mastodon</li><li><a href="https://github.com/jnsahaj/lumen/pull/19">feat: add OpenRouter AI provider support by lkonga · Pull Request #19 · jnsahaj/lumen</a>: Adds support for openrouter.ai as a new provider, allowing users to access various AI models</li><li><a href="https://store.crowdin.com/openrouter?utm_source=chatgpt.com">OpenRouter - Crowdin Marketplace</a>: Compare LLMs and Prices for Optimal Performance.</li><li><a href="https://docs.vapi.ai/providers/model/openrouter?utm_source=chatgpt.com">OpenRouter — Vapi</a>: What is OpenRouter?</li><li><a href="https://creati.ai/ai-tools/openrouter-ai/">OpenRouter: Unified Interface for AI Models | Creati.ai</a>: Discover OpenRouter, a unified interface offering a wide range of AI models. Optimize performance and costs with seamless integration.</li><li><a href="https://aipure.ai/articles/openrouter-review-revolutionizing-ai-language-model-access?utm_source=chatgpt.com">OpenRouter Review: Revolutionizing AI Language Model Access</a>: Explore our comprehensive OpenRouter review. Learn how this unified interface is transforming AI accessibility with diverse models and cost-effective solutions.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1317241044420661248)** (544 messages🔥🔥🔥): 

> `Unsloth Model Support, Dependencies and Installation Issues, Triton Installation, Long Context Models, Ilya Sutskever's Talk Insights` 


- **Unsloth Model and Triton Compatibility**: Users reported issues with installing Unsloth due to conflicting dependencies with Triton, indicating a need to install the correct version for compatibility.
   - Installation challenges were noted, particularly with Python 3.13, with recommendations to use Python 3.10 through Conda for better compatibility.
- **Long Context Models Efficiency**: Discussion highlighted the limitations of long context models, emphasizing that data filtering is complex and quality can't solely dictate training efficiency.
   - Participants noted that excluding 'bad data' might negatively impact understanding, as learning from diverse datasets is crucial for model development.
- **Insights from Ilya Sutskever's Presentation**: A tweet discussed Ilya's insights regarding scaling in AI, emphasizing the search for alternative methods to improve scaling beyond just data quantity.
   - Criticism was expressed around the oversimplification of AI development challenges, questioning the definition and necessity of 'bad data' in model training.
- **Community Experiences and Advice**: Members shared experiences with using various platforms, such as vllm and Docker, highlighting the practical aspects of using local vs. cloud environments for AI modeling.
   - Discussion also revolved around hardware storage challenges for AI development, with users mentioning significant data storage needs in AI training.
- **General Model Optimization and Challenges**: The conversation explored the difficulties of optimizing models with large parameter counts and the challenges associated with storage and performance.
   - Members discussed the need for continuous innovation in AI and skepticism towards claims of reaching limitations in current technologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pypi.org/project/triton/,">no title found</a>: no description found</li><li><a href="https://dev.to/dineshgdk/is-progress-bar-tqdm-killing-your-code-42oj">no title found</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1VA13NvMor9TxHBEDFXYewgu4jrXVQvFZ?usp=sharing#scrollTo=bu-_d4YP_CkR">Google Colab</a>: no description found</li><li><a href="https://nnsight.net/">nnsight &#8212; nnsight</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1868748998783517093">Tweet from Daniel Han (@danielhanchen)</a>: My take on the Post Pretraining world - Ilya’s talk:Ilya is implying we need to find something else to scale - the brain–body mass ratio graph in the talk showed human intelligence “scaled” better tha...</li><li><a href="https://huggingface.co/THUDM/glm-4-9b-chat-1m">THUDM/glm-4-9b-chat-1m · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-bnb-4bit">unsloth/Llama-3.3-70B-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF">unsloth/Llama-3.2-1B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct">HuggingFaceTB/SmolVLM-Instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/THUDM/GLM-4-Voice/issues/133">layer 40 / logits all nan · Issue #133 · THUDM/GLM-4-Voice</a>: its weird .. im trying todo abliteration / finetuning later on the model but its acting rather different from glm4-chat / my stuff works for chat what is the core difference expect the 16k+ audio t...</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit Quantization</a>: Unsloth&#x27;s Dynamic 4-bit Quants selectively avoids quantizing certain parameters. This greatly increases accuracy while maintaining similar VRAM use to BnB 4bit.</li><li><a href="https://youtu.be/jFl5Fewrieo"> - YouTube</a>: no description found</li><li><a href="https://github.com/magicproduct/hash-hop">GitHub - magicproduct/hash-hop: Long context evaluation for large language models</a>: Long context evaluation for large language models. Contribute to magicproduct/hash-hop development by creating an account on GitHub.</li><li><a href="https://github.com/arcee-ai/DAM">GitHub - arcee-ai/DAM</a>: Contribute to arcee-ai/DAM development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/)** (1 messages): 

edd0302: https://main-horse.github.io/posts/visualizing-6d
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1317229390924484640)** (236 messages🔥🔥): 

> `Unsloth Training Issues, Model Compatibility with Streamlit, Dataset Loading Problems, Fine-tuning Techniques, Max Sequence Length for Llama 3.2` 


- **Unsloth Training Starts Check**: A user inquired if a specific screen meant that training had successfully started, showing an image for confirmation.
   - Community members provided insights on potential initialization methods and performance improvements for training.
- **Compatibility of Lora+ with Unsloth**: A user asked about experiences with Lora+ and Unsloth, seeking information on fundamental incompatibilities before trying it.
   - References to external resources and blog insights were provided to clarify the effectiveness of different fine-tuning methods.
- **Challenges with Dataset Loading**: Users faced issues with loading datasets, including problems finding data files and handling CSV formats correctly.
   - Suggestions included using the correct loading syntax and examining file paths to resolve FileNotFoundError.
- **Using Fine-tuned Models in Streamlit**: A user sought assistance connecting a fine-tuned Llama 3.1 model saved on Hugging Face to Streamlit, encountering a model recognition error.
   - Community members clarified that saved model configurations might require merging or proper loading with the base model.
- **Max Sequence Length for Llama 3.2**: A user inquired about the maximum sequence length for Llama 3.2, suggesting it might be 4096.
   - Another user corrected this, indicating that the actual maximum length is 131072.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto">Reward Modelling - DPO, ORPO &amp; KTO | Unsloth Documentation</a>: To use DPO, ORPO or KTO with Unsloth, follow the steps below:</li><li><a href="https://docs.unsloth.ai/basics/vision-fine-tuning">Vision Fine-tuning | Unsloth Documentation</a>: Details on vision/multimodal fine-tuning with Unsloth</li><li><a href="https://huggingface.co/unsloth/SmolLM2-1.7B-Instruct-GGUF">unsloth/SmolLM2-1.7B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/spider-man-uncle-ben-with-great-power-comes-great-responsibility-its-true-just-saying-gif-24193883">Spider Man Uncle Ben GIF - Spider Man Uncle Ben With Great Power Comes Great Responsibility - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing#scrollTo=Edrn7Rxmojtu>">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai/blog/contpretraining">Continued LLM Pretraining with Unsloth</a>: Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.</li><li><a href="https://github.com/unslothai/unsloth/pull/730">Update Model Conversion Command in `save.py` to `convert_hf_to_gguf.py` by malibayram · Pull Request #730 · unslothai/unsloth</a>: Update Model Conversion Command in save.py to convert_hf_to_gguf.pyDescription:This PR updates the model conversion command in save.py to use convert_hf_to_gguf.py, aligning with the latest tools...</li><li><a href="https://github.com/unslothai/unsloth?t">GitHub - unslothai/unsloth: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory</a>: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth?tab=readme-ov-file#-documentation)">GitHub - unslothai/unsloth: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory</a>: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1317388351678054491)** (24 messages🔥): 

> `Model Merging Techniques, AI Regulation and Politics, Impact of AI on Society, Nuclear Treaty Comparisons, Perceptions of AI Gains` 


- **Differentiable Adaptive Merging (DAM) paper highlights**: The paper discusses merging models to balance capabilities without significant retraining, introducing [Differentiable Adaptive Merging (DAM)](https://github.com/arcee-ai/DAM) as an efficient method for model integration.
   - It emphasizes that simpler methods like **Model Soups** can perform well when model similarity is high, showcasing unique strengths across techniques.
- **AI regulation discussions spark debate**: Members expressed skepticism regarding the government's ability to **regulate AI**, comparing it to past efforts with social media and highlighting the complexity of the legal landscape.
   - Discussions revealed a belief that extreme regulation might swing like a pendulum, ultimately leading to a 'sane ground' after multiple back and forths.
- **AI's visible gains impact industries**: There was a consensus that the **gains from AI** are already visible in various industries, with AI being described as an amazing tool that has greatly enhanced productivity.
   - Concerns were raised about underestimating humanity's ability to control a potentially superintelligent AI in the future.
- **Nuclear treaty analogy for AI governance**: A member proposed that establishing a treaty for AI governance akin to the **nuclear power treaty** may be necessary to ensure safety and accountability.
   - The discussion highlighted the challenges in making AI's potential threats visible and the complexity of controlling advanced AI systems.
- **Long-term existence of humanity debated**: Through the discussion, a member noted that societal changes brought about by AI might be beyond current understanding, and warned about the implications of AI's advancements.
   - There are concerns about whether humanity will be able to survive long enough to manage the smart systems that may emerge in the coming decades.



**Link mentioned**: <a href="https://arxiv.org/abs/2410.08371">Merging in a Bottle: Differentiable Adaptive Merging (DAM) and the Path from Averaging to Automation</a>: By merging models, AI systems can combine the distinct strengths of separate language models, achieving a balance between multiple capabilities without requiring substantial retraining. However, the i...

  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1318274419596202127)** (1 messages): 

> `ChatGPT Search Day, 12 Days of OpenAI` 


- **ChatGPT Search Day Celebrated**: Day 8 of the **12 Days of OpenAI** marks the celebration of **ChatGPT Search Day**, with activities encouraging community engagement.
   - To stay updated, members are invited to pick up the <@&1261377106890199132> role in <id:customize>.
- **Check Out the YouTube Video**: A [YouTube video](https://www.youtube.com/watch?v=OzgNJJ2ErEE) is highlighted for viewers interested in learning more about the events during this day.
   - Unfortunately, no description or further details were provided about the video content.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=OzgNJJ2ErEE"> - YouTube</a>: no description found

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1317221475677110384)** (614 messages🔥🔥🔥): 

> `Character AI Performance, OpenAI and Alignment, New AI Models, Local LLMs, AI and Politics` 


- **Discussion on Character AI Decline**: Users expressed dissatisfaction with Character AI, noting a decline in performance and the negative impact of the app's marketing shift towards children, which has resulted in stricter filters and reduced context ability.
   - In comparison, users have found ChatGPT to be better suited for creative tasks, particularly in roleplaying scenarios.
- **AI Alignment Framework Discussion**: A user shared a working framework on AI alignment, emphasizing principles based on shared human values and iterative feedback to ensure inclusivity in AI development.
   - The conversation highlighted the challenge of getting various stakeholders to agree on alignment principles, with one user questioning the feasibility of this goal.
- **Emerging AI Models**: There was interest in new AI models like Google's Gemini and updates to Imagen, with users discussing the performance comparisons with existing models like OpenAI's 4o.
   - Users noted that while models like Grok are making strides, they still lag behind the more established options like ChatGPT.
- **Local LLMs Discussion**: Participants discussed the advantages of local LLMs, suggesting they could provide a more customizable and flexible AI experience compared to large tech solutions.
   - Concerns were raised that big tech companies might focus primarily on productivity improvements rather than enhancing creativity in AI interactions.
- **Mood of the Discord Channel**: The overall sentiment in the channel indicated that the discussions were veering towards unwanted political topics, frustrating users who preferred conversations centered on AI.
   - Some users jokingly noted the chaotic tone of the channel with mixed reactions, indicating that it created an interesting atmosphere on that day.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/vegeta-its-over9000-gif-14419267">Vegeta Its Over9000 GIF - Vegeta Its Over9000 - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.mercurynews.com/2024/12/13/openai-whistleblower-found-dead-in-san-francisco-apartment/">OpenAI whistleblower found dead in San Francisco apartment</a>: A former OpenAI researcher who raised concerns about the company is dead at 26.</li><li><a href="https://github.com/AlignAGI/Alignment/">GitHub - AlignAGI/Alignment: Promoting global awareness and action for ethical AI alignment and safeguarding humanity against AI self-replication risks. Includes research, frameworks, and open-source resources.</a>: Promoting global awareness and action for ethical AI alignment and safeguarding humanity against AI self-replication risks. Includes research, frameworks, and open-source resources. - AlignAGI/Alig...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1317312280672731176)** (24 messages🔥): 

> `O1 Pro AI, OpenAI Subscription Discussions, Chess with GPT, LLMs and Calculations, GPT 4o vs GPT 4o-mini` 


- **O1 Pro: The AI Girlfriend Dilemma**: Members discussed **O1 Pro**, with one stating it makes the **best AI girlfriend**, while another emphasized its pricing at **200 bucks** is too high.
   - A user humorously remarked about potential wait times, suggesting it makes users wait ages for a reply.
- **OpenAI Subscriptions: Worth the Cost?**: Concerns arose over the value of **OpenAI subscriptions**, with suggestions that investing in IRL dating experiences might be a better alternative.
   - Another member reflected on their regret about not fine-tuning more when it was free and recognized the decent usage available through APIs.
- **Chess Conundrum with GPT**: A user shared their experience of **playing chess** with GPT, noting a piece duplication issue, which led to discussions about LLM capabilities.
   - Another highlighted the limitation of LLMs in logical reasoning for games like chess, while others noted that a **Python library** could assist with chess logic.
- **Capability Gap: GPT 4o vs GPT 4o-mini**: Frustrations were expressed regarding the performance disparity between **GPT 4o** and **GPT 4o-mini**, with claims that the mini version feels like it's **sleepwalking**.
   - Members felt the 4o-mini's responses to be significantly worse than the main 4o model, indicating a noticeable drop in quality.
- **Countdown to the Announcement**: Anticipation built around a possible **announcement** on the 8th, with a member confirming it would happen in just over **20 minutes**.
   - This created excitement in the community as they awaited news about potential updates or features.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1317405015660560394)** (67 messages🔥🔥): 

> `Prompt Engineering Techniques, AI Model Capabilities in Coding, Learning Programming, Memory Management in AI, Creating a Curriculum for Prompt Engineering` 


- **Enhancing Prompt Engineering Skills**: Users discussed refining their prompt engineering skills, emphasizing the importance of knowing exactly what they want from the AI, likened to cooking: one can rely on pre-made dishes or cook from scratch depending on the situation.
   - Clarifications were made that understanding language and providing clear instructions are key to effective prompting, regardless of one's coding experience.
- **Utilizing AI for Coding Assistance**: One user expressed interest in leveraging ChatGPT for writing code to be used in their own IDE, specifically eager to see its capabilities in developing a modern website.
   - Advice was given to provide details about their current coding experience and expectations, which could help the AI offer more tailored guidance for project development.
- **Memory and Custom Instructions**: Discussion around the AI memory system indicated that users can update the AI's memory about their preferences and prior prompts, leading to more personalized interactions.
   - It was suggested to utilize stored memories effectively while recognizing the limitations and available workarounds for memory management.
- **Potential Curriculum for Prompt Engineering**: A user shared their ambition to develop curriculum around prompt engineering and sought information about existing classes and resources on the topic.
   - Suggestions were given on the importance of having a clear goal in prompting and how learning to code could enhance one's ability to communicate effectively with AI.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1317405015660560394)** (67 messages🔥🔥): 

> `Prompt Engineering, Using ChatGPT for Coding, Memory Management, Prompt Library Concept, Learning Programming Languages` 


- **Understanding Prompt Engineering**: Members discussed the importance of crafting precise prompts in prompt engineering, emphasizing that knowing exactly what you want from the model is crucial. Explorations into prompt effectiveness highlight that tailored prompts can lead to more accurate and useful outputs.
- **Leveraging ChatGPT for Coding**: A user inquired about the best practices for using ChatGPT to write code for use in an IDE, expressing interest in exploring the model's capabilities. It was recommended that users provide clear specifications about their experience level and the tools they are using to get the best results.
- **Consolidating Memory Space**: Discussions around memory management revealed techniques for efficiently using memory space within the model, such as summarizing and feeding back important information. Members shared that users do not need to stress overly about memory limitations, as various workarounds exist.
- **Prompt Library Concept**: A user questioned whether maintaining a library of prompts is similar to updating the model's memory with past prompts. Members discussed the informal nature of a prompt library and indicated a shared channel for exploring prompt engineering.
- **Learning Programming Languages**: The conversation highlighted a member's belief that learning coding might be unnecessary since ChatGPT can help code effectively. However, it was pointed out that having a foundational understanding can aid in better communicating needs and evaluating outputs when working with the model.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1317252744003846184)** (327 messages🔥🔥): 

> `AI Government Regulation, Apollo LMMs Release, Hermes 3 Key Access, Model Performance Issues, Community Involvement in AI` 


- **Concerns Over AI Regulation by Government**: Elon Musk highlighted that the US government may restrict AI startups and control the narrative around AI technology to prevent the emergence of independent initiatives.
   - Concerns are raised about a potential monopoly in AI development driven by governmental partnerships and regulations that disadvantage smaller players.
- **Release of Apollo LMMs**: The community discussed the recent update of the Apollo LMMs, which includes models focused on video understanding and multimodal capabilities.
   - Early impressions of the Apollo models suggest they perform well, sparking interest in their potential applications.
- **Hermes 3 Access and Issues**: Users are seeking access to Hermes 3 but are informed that there are no keys available, with troubleshooting ongoing due to model issues.
   - The developers are aware of the issues affecting Hermes 3 and plan to implement fixes, including adjustments to the chat template.
- **Performance Issues with AI Models**: Users report various behaviors and issues with different AI models, suggesting that some scripts may require reruns or updates.
   - Concerns persist about long wait times for solutions, particularly for models operating under Trusted Execution Environments (TEEs).
- **Community Collaboration in AI Development**: Discussions indicate that there is a significant desire in the community to collaborate on AI training and development, leveraging distributed computing.
   - The community expresses optimism about open-source contributions, emphasizing that innovative new ideas can emerge from collective efforts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheInsiderPaper/status/1867728290066153576">Tweet from Insider Paper (@TheInsiderPaper)</a>: BREAKING: OpenAI whistleblower found dead in San Francisco apartment — TechCrunch</li><li><a href="https://jplhughes.github.io/bon-jailbreaking/">Best-of-N Jailbreaking</a>: no description found</li><li><a href="https://www.mercurynews.com/2024/12/13/openai-whistleblower-found-dead-in-san-francisco-apartment/">OpenAI whistleblower found dead in San Francisco apartment</a>: A former OpenAI researcher who raised concerns about the company is dead at 26.</li><li><a href="https://x.com/elonmusk/status/1868302204370854026?s=46">Tweet from Elon Musk (@elonmusk)</a>: no description found</li><li><a href="https://huggingface.co/spaces/Apollo-LMMs/Apollo-3B">Apollo 3B - a Hugging Face Space by Apollo-LMMs</a>: no description found</li><li><a href="https://huggingface.co/posts/m-ric/471403804474189">@m-ric on Hugging Face: &quot;𝗣𝗼𝘁𝗲𝗻𝘁𝗶𝗮𝗹 𝗽𝗮𝗿𝗮𝗱𝗶𝗴𝗺 𝘀𝗵𝗶𝗳𝘁 𝗶𝗻 𝗟𝗟𝗠𝘀: 𝗻𝗲𝘄…&quot;</a>: no description found</li><li><a href="https://www.promptingguide.ai/research/llm-agents">Prompt Engineering Guide</a>: A Comprehensive Overview of Prompt Engineering</li><li><a href="https://x.com/N8Programs/status/1868082092263010791">Tweet from N8 Programs (@N8Programs)</a>: Okay, so cohere&#39;s model basically uses the following setup:3 layers local attention (4096 sliding-window with ROPE)1 layer global attention (no position encoding)8 times over.What we do is keep th...</li><li><a href="https://x.com/NousResearch/status/1848397863547515216">Tweet from Nous Research (@NousResearch)</a>: no description found</li><li><a href="https://x.com/N8Programs/status/1868071000430321763">Tweet from N8 Programs (@N8Programs)</a>: HERES COHERES 7B model, at 4-bit quantization with 4-bit KVCache, summarizing the entirety of harry potter - which is 115 thousand tokens of context - at 13tok/sec, prompt processing ~181tok/sec (post...</li><li><a href="https://huggingface.co/Apollo-LMMs">Apollo-LMMs (Apollo-LMMs)</a>: no description found</li><li><a href="https://apollo-lmms.github.io/">Apollo</a>: Apollo: An Exploration of Video Understanding in Large Multimodal Models</li><li><a href="https://www.lesswrong.com/tag/alignment-tax">Alignment Tax - LessWrong</a>: An alignment tax (sometimes called a safety tax) is the extra cost of ensuring that an AI system is aligned, relative to the cost of building an unaligned alternative. The term ‘tax’ can be misleading...</li><li><a href="https://devclass.com/2024/12/12/sqlite-re-implemented-in-rust-to-achieve-asynchronous-i-o-and-other-changes/">SQLite re-implemented in Rust to achieve asynchronous I/O and other changes &#8226; DEVCLASS</a>: Turso, a developer focused on database solutions, is re-implementing the SQLite database engine in Rust, in order to [&hellip;]</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/YJHr2iAdL8">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/bitcloud/status/1868729306492674170">Tweet from Lachlan Phillips exo/acc 👾 (@bitcloud)</a>: Military and CIA partnerships, regulatory capture attempts. Creating an ever increasing power and capabilities gap between civilians and the government as you release frontier models to your governmen...</li><li><a href="https://github.com/arcee-ai/DAM">GitHub - arcee-ai/DAM</a>: Contribute to arcee-ai/DAM development by creating an account on GitHub.</li><li><a href="https://youtu.be/Pz9YeBs_afo?t=782)."> - YouTube</a>: no description found</li><li><a href="https://devclass.com/2024/12/12/sqlite-re-implemented-in-rust-to-achieve-asynchronous-i-o-and-other-">SQLite re-implemented in Rust to achieve asynchronous I/O and other changes &#8226; DEVCLASS</a>: Turso, a developer focused on database solutions, is re-implementing the SQLite database engine in Rust, in order to [&hellip;]
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1317228428973445222)** (32 messages🔥): 

> `Open-source coding LLMs, Fine-tuning local LLMs, Vector databases and embeddings, Model merging and souping, RNG algorithms in LLMs` 


- **Open-source LLMs suitable for coding**: A member suggested several open-source coding LLMs such as **Mistral Codestral**, **Qwen 2.5 Coder**, and **DeepSeek** that can be integrated with IDEs like VS Code and PyCharm, along with extensions like [continue.dev](https://continue.dev).
   - These tools enable developers to enhance coding efficiency using local models.
- **Fine-tuning local LLMs is feasible**: A user inquired about the possibility of fine-tuning local LLMs and was informed that with tools like **unsloth** and **axolotl**, even older tech enthusiasts could potentially train models up to 8 billion parameters using **QLoRA**.
   - There are growing resources that make customization accessible for those willing to learn.
- **Debate on using vector databases**: Discussion arose regarding the optimal use of vector databases for structured product data, with suggestions to evaluate simpler search methods like **BM25** rather than just relying on embeddings.
   - One member expressed why embeddings might not suit structured queries effectively, pointing out that higher accuracy in retrieval could be prioritized.
- **Current state of model merging and souping**: Members discussed the ongoing trends in model merging, commonly known as **model souping**, noting that many popular models are combinations of existing ones, which raises questions about its efficacy.
   - Concerns remained about the potential risks involved, however, many acknowledged that the approach is still yielding positive results within constraints.
- **Understanding RNG algorithms in LLMs**: Questions were raised about the random number generation (RNG) algorithms used in LLMs and whether they typically deploy algorithms like **Xorshift** or others when generating outputs.
   - Clarification was sought about their application, especially in sampling and distribution stages.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2203.05482#">Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time</a>: The conventional recipe for maximizing model accuracy is to (1) train multiple models with various hyperparameters and (2) pick the individual model which performs best on a held-out validation set, d...</li><li><a href="https://arxiv.org/abs/2203.0548">An Adaptable and Agnostic Flow Scheduling Approach for Data Center Networks</a>: Cloud applications have reshaped the model of services and infrastructure of the Internet. Search engines, social networks, content delivery and retail and e-commerce sites belong to this group of app...</li><li><a href="https://github.com/troy12x/Quasar-1">GitHub - troy12x/Quasar-1: Quasar-1 is a large language model architecture that moves beyond prompt engineering to achieve genuine reasoning capabilities. Unlike traditional LLMs that rely heavily on carefully crafted prompts, Quasar-1 implements reasoning at its architectural core through temperature-guided reasoning and guided sequence of thoughts.</a>: Quasar-1 is a large language model architecture that moves beyond prompt engineering to achieve genuine reasoning capabilities. Unlike traditional LLMs that rely heavily on carefully crafted prompt...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1317568067357638819)** (18 messages🔥): 

> `Model Compression Techniques, Application of Communication Theory to AI, Lora Updates in Model Training, Trade-offs in Training Approaches, Position Invariance in MLPs` 


- **Communication Theory Enhances AI Models**: Discussion centered on how principles from **communication theory** are influencing the development of **LLMs**, particularly in gradient transmission during distributed training.
   - Members noted that **trading compute for bandwidth** could streamline processes, although combining techniques may be complex.
- **Efficient Encoding and Decoding Challenges**: While **decoding** techniques are rapid, the **encoding** process demands solving an optimization issue using the Viterbi algorithm, complicating implementation.
   - Participants questioned the feasibility of incorporating **compression methods** during model training to enhance data efficiency without impairing performance.
- **Dynamic Lora Usage in Training**: Members explored how **Lora updates** function to trade time for memory efficiency, suggesting a sequential training process instead of parallel updates.
   - *Fixed loras break during pretraining*, but by reinitializing them, models maintain flexibility and can adapt to new data.
- **Position Invariance and Redundancy**: Real.azure highlighted that there seems to be minimal attention to the **position invariance** of MLPs, where changing weight orders in projection blocks does not affect performance.
   - This presents a potential area of research on **information redundancy** within neural architectures.
- **History of Trellis Coding**: An interesting overview of **trellis coding** was shared, illustrating its delayed introduction into standards despite its foundational significance.
   - Members discussed how optimizing such techniques could create **cross-disciplinary** advances in AI models.



**Link mentioned**: <a href="https://x.com/OpenlifesciAI/status/1867999825721242101>">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: 🌟 Weekly Medical AI Research Roundup 🌟📅 December 7-14, 2024Here&#39;s your weekly digest of the most important medical AI papers! 🎉🤖 Medical LLM & Other Models- PediaBench: Chinese Pediatric LLM-...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1317236917783232523)** (3 messages): 

> `Byte Latent Transformer, Dynamic Tokenization, Inference Efficiency, Llama 3 Benchmark, Byte-level Models` 


- **Meta's Byte Latent Transformer Upsets Tokenization**: Meta just launched the **Byte Latent Transformer (BLT)**, a tokenizer-free architecture that dynamically encodes Bytes into Patches, enhancing **inference efficiency** and robustness.
   - *‘It’s like fucking christmas!’* says a member, expressing excitement over the need for dynamic tokenization learned during training.
- **BLT Competes with Llama 3 at Scale**: BLT models claim to match the performance of tokenization-based models like **Llama 3** while potentially reducing **inference flops** by up to **50%**.
   - They highlight that BLT can train the **Llama-3 8B** model on **1T tokens**, outperforming standard architectures using BPE.
- **Doubts on Training Efficiency of Byte Models**: A member referenced that while **byte-level models** are as training efficient as **BPE models**, the largest byte-level LLM is only **350M parameters** trained on a limited dataset.
   - They questioned, *‘When will we finally ditch tokenization?’* reflecting skepticism about the future of tokenization.
- **Validation of BLT's Claims**: Another member confirmed that the information about **BLT** is indeed **legit**, reinforcing confidence in the new model's potential.
   - This affirmation came after discussions surrounding the model's capabilities and benchmarks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/scaling01/status/1867573707247346003?s=46">Tweet from Lisan al Gaib (@scaling01)</a>: META JUST KILLED TOKENIZATION !!!A few hours ago they released &#34;Byte Latent Transformer&#34;. A tokenizer free architecture that dynamically encodes Bytes into Patches and achieves better inferenc...</li><li><a href="https://x.com/MarkSchmidty/status/1857522783720272304?t=Z7z5ArMVl8JCptgCP6iEjQ&s=19">Tweet from Mark Schmidt 🌐 (@MarkSchmidty)</a>: Byte level models are just as training efficient as BPE models and yet the largest byte-level LLM is a tiny 350M parameters trained on a disappointingly small dataset. When will we finally ditch token...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1317568067357638819)** (18 messages🔥): 

> `Decompression on GPU, Historical influence of Physics on AI, Trellis coding and its applications, Model compression and redundancy, Distributed training methods` 


- **Efficient Decompression Implementation on GPU**: The paper discusses a method for implementing **decompression** efficiently on a GPU, citing its simplicity despite being hard to read. The core idea was also published before, indicating proper citation practices in the research community.
   - Members noted that while the method is effective post-quantization, it remains too slow for training.
- **Physics Drives Most AI Techniques**: The conversation pointed out that many AI techniques have origins in Physics, emphasizing that any viable method is likely to have been explored by physicists previously. Ideas from **communication theory** are especially pertinent for **LLMs**, showcasing the historical intertwining of disciplines.
   - One member remarked on the intellectual genealogy, suggesting that advances in AI often trace back to physical sciences.
- **Trellis Coding: A Historical Perspective**: A member shared the **history of trellis coding**, noting its inventor waited six years to make it accessible, which later became part of an official standard. This historical anecdote highlights the slow but impactful progression of ideas in technology.
   - There was a suggestion that such techniques could optimize gradient transmission in distributed training contexts, tackling complexities in encoding and optimization.
- **Trade-offs in Model Compression**: Discussion around maintaining model integrity while updating **Loras** revealed strategies that trade training time for memory efficiency, suggesting a method of reinitializing Loras periodically. This approach resembles sequential retraining rather than parallel training.
   - A member raised concerns regarding model degradation when using fixed Loras in pretraining situations.
- **Redundant Information in MLPs**: A curiosity emerged regarding the apparent lack of explored **position invariance** in MLPs, particularly in up-down projection blocks, where weight order can be altered without affecting performance. This potential redundancy in information could signal opportunities for simplification.
   - The conversation indicated that further exploration in this area might yield new insights for model compression strategies.



**Link mentioned**: <a href="https://x.com/OpenlifesciAI/status/1867999825721242101>">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: 🌟 Weekly Medical AI Research Roundup 🌟📅 December 7-14, 2024Here&#39;s your weekly digest of the most important medical AI papers! 🎉🤖 Medical LLM & Other Models- PediaBench: Chinese Pediatric LLM-...

  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1317951045011374081)** (2 messages): 

> `SF Compute launch, Qwen QwQ price cut, New Grok models from xAI` 


- **SF Compute joins OpenRouter**: OpenRouter announced a new provider: **SF Compute**, enhancing their offerings.
   - This addition aims to broaden options for users looking for diverse service integrations.
- **Qwen QwQ gets a hefty price reduction**: **Qwen QwQ** experiences a significant **55% price cut**, attracting more users to its features.
   - Details can be found on their [pricing page](https://openrouter.ai/qwen/qwq-32b-preview).
- **Traffic increasing for new Grok models**: Two new **Grok models** from **xAI** were released over the weekend, leading to increased traffic on their platform.
   - Users are encouraged to explore all the models at [OpenRouter's xAI page](https://openrouter.ai/x-ai).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1868534692183507098">Tweet from OpenRouter (@OpenRouterAI)</a>: Two new @Grok models from @xai came out this weekend - already seeing traffic move over.Check them all out here! https://openrouter.ai/x-ai</li><li><a href="https://openrouter.ai/qwen/qwq-32b-preview">QwQ 32B Preview - API, Providers, Stats</a>: QwQ-32B-Preview is an experimental research model focused on AI reasoning capabilities developed by the Qwen Team. As a preview release, it demonstrates promising analytical abilities while having sev...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1317903486678995065)** (5 messages): 

> `OpenRouter API wrapper, OpenRouter-client` 


- **Launch of OpenRouter API Wrapper**: A member shared the announcement of an API wrapper for OpenRouter, named [openrouter-client](https://www.npmjs.com/package/openrouter-client), which was published just two days ago.
   - The wrapper simplifies interactions with OpenRouter, featuring example code for implementation and configuration.
- **Community Excitement for API Wrapper**: One member expressed enthusiasm about the new API wrapper, stating, *That's awesome!* in response to the announcement.
   - The developer acknowledged the excitement by responding with a simple, *Thank you!*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://coauthor.studio/rewind">2024 LinkedIn Rewind | Your Year in Review</a>: Create your personalized 2024 highlight reel for LinkedIn in minutes. Free tool for professionals to showcase achievements and insights in their authentic voice. No login required.</li><li><a href="https://www.npmjs.com/package/openrouter-client">openrouter-client</a>: An API wrapper for OpenRouter. Latest version: 1.1.0, last published: 2 days ago. Start using openrouter-client in your project by running `npm i openrouter-client`. There are no other projects in the...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1317226679294361600)** (372 messages🔥🔥): 

> `Hermes 3 405B performance, Gemini Pro 2 capabilities, Image generation model updates, Prompt caching in LLM providers, Rate limits for Gemini models` 


- **Hermes 3 405B shows strong capabilities**: Users reported that Hermes 3 405B has been effective for creative tasks, with some claiming it rivals Claude 2.0 in quality.
   - However, there were discussions about its slower performance compared to other models in coding tasks.
- **Gemini Pro 2's growing popularity**: Gemini Pro 2 (1206) has been highlighted as a competitive alternative to models like Sonnet 3.5 for coding tasks.
   - Some users noted its effectiveness in generating code and handling scientific problems better than Flash.
- **Image generation model updates from Google**: Google announced new versions of its image generation models, including Imagen 3 and a new model called Whisk.
   - These updates suggest a push towards better visual content generation capabilities in AI.
- **Prompt caching functionality in providers**: Discussion arose regarding the absence of prompt caching features for open source models in certain providers.
   - Some users theorized on the potential cost savings and efficiency gains that caching could provide in LLM applications.
- **Rate limits for Gemini models**: Users expressed concerns over the rate limits associated with different Gemini models, especially under the Google Cloud Platform.
   - It was observed that rate limits varied significantly between the experimental and production models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/quick-start">Quick Start | OpenRouter</a>: Start building with OpenRouter</li><li><a href="https://www.bbc.com/news/articles/cd0el3r2nlko">Suchir Balaji: OpenAI whistleblower found dead in apartment</a>: The San Francisco medical examiner&#x27;s office determined Suchir Balaji&#x27;s death to be suicide and police found no evidence of foul play.</li><li><a href="https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks">Creating and highlighting code blocks - GitHub Docs</a>: no description found</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: Set limits on model usage</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1hf0nmm/chatgpt_is_a_fantastic_fiction_editoras_long_as/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://openrouter.ai/docs/parameters">Parameters | OpenRouter</a>: Configure parameters for requests</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>: Every front-end GUI client for ChatGPT, Claude, and other LLMs - billmei/every-chatgpt-gui</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15lihmq/big_model_comparisontest_13_models_tested/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://blog.google/technology/google-labs/video-image-generation-update-december-2024/">State-of-the-art video and image generation with Veo 2 and Imagen 3</a>: We’re rolling out a new, state-of-the-art video model, Veo 2, and updates to Imagen 3. Plus, check out our new experiment, Whisk.</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>: OpenRouter Incident History
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1317263148541018122)** (3 messages): 

> `OpenRouter launch, New feature integration` 


- **OpenRouter Feature Goes Live!**: @alexatallah announced that the new feature is now live for everyone 🙂 and stated that an announcement will be put up soon.
   - *Stay tuned for more details!*
- **Users Ask for Feature Usage Instructions**: A user inquired, *how to use this feature?*, wanting clarity on the new functionality.
   - Another user responded, noting that you just need to go to [OpenRouter Settings Integrations](https://openrouter.ai/settings/integrations) and add your key there!


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1317270737861480449)** (70 messages🔥🔥): 

> `Grading Criteria for Student Projects, Non-Transformer Models Research, Byte vs Bit Encoding, Model Training Data Shuffling, JAX/Flax vs TensorFlow` 


- **Creating Grading Criteria for Student Projects**: A member suggested incorporating grading criteria as part of the assignment for students tasked with generating tokens for images, leading to humorous sample code for grading.
   - Discussions included ideas like using perplexity and classifiers to grade submissions, with recommendations to generate examples intentionally difficult to cheat on.
- **Active Research on Non-Transformer Models**: Members discussed ongoing research in non-transformer architectures, with mentions of labs like Numenta and AI2 lab releasing several checkpoints for their models.
   - Curiosity was shared about smaller labs pushing novel non-transformer research instead of mainstream transformer models.
- **Debating Byte vs Bit Encoding**: The dialogue covered the relevance of byte encoding, with distinguishing cases where it might lose information compared to tokenized structures like BPE.
   - Members expressed that while byte-level processing could represent text more accurately, it may not yield significant advantages over existing tokenization methods.
- **Addressing Model Bias Due to Late Training**: Concerns were raised regarding models becoming biased towards recently introduced training data, with suggestions of shuffling data to mitigate these effects.
   - One member recounted experiences with data homogenization strategies to improve model training fairness.
- **Switching from TensorFlow to JAX/Flax**: The conversation highlighted frustrations with TensorFlow’s declining support, prompting members to consider switching to JAX/Flax for better performance.
   - The sentiment toward JAX/Flax was overwhelmingly positive as many considered it a more robust option moving forward.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.liquid.ai/">Liquid AI: Build capable and efficient general-purpose AI systems at every scale.</a>: We build capable and efficient general-purpose AI systems at every scale. Liquid Foundation Models (LFMs) are a new generation of generative AI models that achieve state-of-the-art performance at ever...</li><li><a href="https://forms.gle/JcYAJEukfBiYVxTW8">Airpods and battery</a>: Would you purchase a device that lets you charge your power bank while listening to music? </li><li><a href="https://files.vermeille.fr/cparti.html">Instructions</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1317223024188981348)** (249 messages🔥🔥): 

> `Attention vs Kernel Methods, Constraint Satisfaction Problems, Reinforcement Learning and Memory, Iterative Reasoning in Neural Networks, Hybrid Architectures in Transformers` 


- **Attention and Kernel Methods**: Discussion emerged around the framing of attention as a kernel method, with members noting this is not wholly accurate, particularly when assessing the function of self-attention operations like softmax. Members debated the nuances of whether attention mechanisms fully exploit their potential compared to kernel approaches, leading to discussions on the underlying mathematical distinctions.
   - The relationship between kernel methods and attention was illustrated as a hierarchy, indicating that while attention can be approximated by kernel methods, this simplification does not capture the intricacies of attention's operational context.
- **Learning Implicit Constraints in Models**: The discussion highlighted an interest in whether models could learn to solve constraint satisfaction problems, specifically using Sudoku as a test case. The feasibility of training models on a small dataset to ensure solutions satisfy learned implicit constraints was explored.
   - Members suggested that performance may be observed by manipulating data representations and even introduced ideas around managing architecturally induced biases during training.
- **Iterative Reasoning via Energy Diffusion**: The IRED framework for learning reasoning through energy diffusion was introduced, which aims to solve more complex problems by better organizing constraints between inputs and outputs. Experimental results indicated improved performance on tasks requiring more sophisticated reasoning compared to traditional methods.
   - Discussion noted the study's focus on constrained optimization problems and how the methodology presents a different take on how neural networks might learn reasoning implicitly from structured data.
- **Hybrid Architectures and Performance**: The architecture performance of various hybrid methods integrating both attention and RNN characteristics like Gated DeltaNet and Samba was a focal point. Members debated different setups and their implications for training efficiency, generalization, and potential performance gains.
   - Specific suggestions were made about testing modifications to the CoHere architecture and evaluating the effects of different attention mechanisms in various experimental frameworks.
- **Meta Tokens in Transformers**: Members shared insights into the role of meta tokens within transformer architectures and discussed the implications for processing contextual information more effectively. The conversation revolved around how augmenting transformers with memory capabilities could enhance their representation and processing functions.
   - Participants expressed varying sentiments on the usefulness of meta tokens, leading to calls for further empirical examination of their impacts in controlled settings.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.07041">Emergent properties with repeated examples</a>: We study the performance of transformers as a function of the number of repetitions of training examples with algorithmically generated datasets. On three problems of mathematics: the greatest common ...</li><li><a href="https://arxiv.org/abs/2412.07684v1">The Pitfalls of Memorization: When Memorization Hurts Generalization</a>: Neural networks often learn simple explanations that fit the majority of the data while memorizing exceptions that deviate from these explanations.This behavior leads to poor generalization when the l...</li><li><a href="https://arxiv.org/abs/2406.07522v1">Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling</a>: Efficiently modeling sequences with infinite context length has been a long-standing problem. Past works suffer from either the quadratic computation complexity or the limited extrapolation ability on...</li><li><a href="https://arxiv.org/abs/2406.11179">Learning Iterative Reasoning through Energy Diffusion</a>: We introduce iterative reasoning through energy diffusion (IRED), a novel framework for learning to reason for a variety of tasks by formulating reasoning and decision-making problems with energy-base...</li><li><a href="https://mingukkang.github.io/GigaGAN/">GigaGAN: Scaling up GANs for Text-to-Image Synthesis</a>: no description found</li><li><a href="https://arxiv.org/abs/2411.13504">Disentangling Memory and Reasoning Ability in Large Language Models</a>: Large Language Models (LLMs) have demonstrated strong performance in handling complex tasks requiring both extensive knowledge and reasoning abilities. However, the existing LLM inference pipeline ope...</li><li><a href="https://arxiv.org/abs/2405.13956">Attention as an RNN</a>: The advent of Transformers marked a significant breakthrough in sequence modelling, providing a highly performant architecture capable of leveraging GPU parallelism. However, Transformers are computat...</li><li><a href="https://arxiv.org/abs/2411.13676">Hymba: A Hybrid-head Architecture for Small Language Models</a>: We propose Hymba, a family of small language models featuring a hybrid-head parallel architecture that integrates transformer attention mechanisms with state space models (SSMs) for enhanced efficienc...</li><li><a href="https://arxiv.org/abs/2306.00946">Exposing Attention Glitches with Flip-Flop Language Modeling</a>: Why do large language models sometimes output factual inaccuracies and exhibit erroneous reasoning? The brittleness of these models, particularly when executing long chains of reasoning, currently see...</li><li><a href="https://arxiv.org/abs/2006.11527">Memory Transformer</a>: Transformer-based models have achieved state-of-the-art results in many natural language processing tasks. The self-attention architecture allows transformer to combine information from all elements o...</li><li><a href="https://neel04.github.io/my-website/blog/pytorch_rant/">PyTorch is dead. Long live JAX. | Neel Gupta</a>: no description found</li><li><a href="https://arxiv.org/abs/2410.01201v1">Were RNNs All We Needed?</a>: The scalability limitations of Transformers regarding sequence length have renewed interest in recurrent sequence models that are parallelizable during training. As a result, many novel recurrent arch...</li><li><a href="https://arxiv.org/abs/2412.06464">Gated Delta Networks: Improving Mamba2 with Delta Rule</a>: Linear Transformers have gained attention as efficient alternatives to standard Transformers, but their performance in retrieval and long-context tasks has been limited. To address these limitations, ...</li><li><a href="https://x.com/SkyLi0n/status/1867324080262885800">Tweet from Aaron Gokaslan (@SkyLi0n)</a>: Can GANs outperform diffusion models in 2024? Yes! #2200: The GAN is dead; long live the GAN! A Modern GAN Baseline! Come join me if you&#39;re attending NeurIPS 2024. Thursday evening poster session ...</li><li><a href="https://arxiv.org/abs/2407.01178">$\text{Memory}^3$: Language Modeling with Explicit Memory</a>: The training and inference of large language models (LLMs) are together a costly process that transports knowledge from raw data to meaningful computation. Inspired by the memory hierarchy of the huma...</li><li><a href="https://github.com/lucidrains/x-transformers?tab=readme-ov-file#memory-transformers">GitHub - lucidrains/x-transformers: A concise but complete full-attention transformer with a set of promising experimental features from various papers</a>: A concise but complete full-attention transformer with a set of promising experimental features from various papers - lucidrains/x-transformers
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1317227922016309359)** (8 messages🔥): 

> `RASP Framework for Transformers, SAE Steering Applications, Contrastive Objectives in MCMC, Negative Results in SAE Research, Dense Probes and SAE Encodings` 


- **RASP Introduces New Programming Model for Transformers**: The paper titled [RASP](https://arxiv.org/abs/2106.06981) proposes a computational model for **Transformer-Encoders** using a programming language to map fundamental components like attention and feed-forward computation.
   - It demonstrates how Transformers can be trained to mimic RASP solutions for tasks such as **histograms** and **sorting**.
- **Sieve Demonstrates Effective SAE Steering**: Excitement surrounds the implementation of **SAE-based interventions** in the **Sieve** pipeline, which shows improved performance on fuzz testing for Python functions with minimal effort.
   - This approach achieves **conditional feature steering** that maintains performance while precisely preventing unwanted behaviors like regex usage.
- **Interest in Contrastive Objectives within MCMC**: A member inquired whether there are studies exploring **contrastive objectives** within **MCMC frameworks**, especially in relation to large language models.
   - This signals a growing curiosity around potential integrations of these methodologies in understanding natural language distributions.
- **Skepticism Surrounding SAE Steering Results**: Despite the promising application of SAE steering, it appears to hurt overall performance as noted in [recent research](https://arxiv.org/abs/2411.11296).
   - Members expressed concerns about identifying effective steering mechanisms without sacrificing performance, especially concerning **refusal behavior**.
- **Positive Probing Results for SAE**: Research highlights the efficacy of **dense probes trained on SAE encodings**, emphasizing their strengths in low data regimes and corrupted datasets.
   - While SAE probes show competitive results, there are null findings against activation probes in certain settings, raising discussions on the reliability of both methods.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.11296">Steering Language Model Refusal with Sparse Autoencoders</a>: Responsible practices for deploying language models include guiding models to recognize and refuse answering prompts that are considered unsafe, while complying with safe prompts. Achieving such behav...</li><li><a href="https://arxiv.org/abs/2106.06981">Thinking Like Transformers</a>: What is the computational model behind a Transformer? Where recurrent neural networks have direct parallels in finite state machines, allowing clear discussion and thought around architecture variants...</li><li><a href="https://www.tilderesearch.com/blog/sieve">Sieve: SAEs Beat Baselines on a Real-World Task (A Code Generation Case Study) | Tilde</a>: no description found</li><li><a href="https://www.lesswrong.com/posts/NMLq8yoTecAF44KX9/sae-probing-what-is-it-good-for-absolutely-something">SAE Probing: What is it good for? Absolutely something! — LessWrong</a>: Subhash and Josh are co-first authors. Work done as part of the two week research sprint in Neel Nanda’s MATS stream …
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1318137501319172188)** (12 messages🔥): 

> `lm_eval harness with VLLM, Error Issues with VLLM API, VLLM Version Discussions` 


- **lm_eval harness successfully implemented with VLLM**: A user shared the working method to get the **lm_eval harness** to function with **VLLM**, indicating a specific installation command.
   - *This process includes installing version 0.6.3 of VLLM to prevent issues with the evaluation harness.*
- **VLLM API errors arise**: Members discussed errors arising from VLLM, suggesting that the **internal API used by lm_eval** may have changed.
   - *Another member hinted that this could be connected to a specific commit of VLLM.*
- **Version confusion raises questions**: Query about whether the errors were encountered in **VLLM version 0.6.4**, with a mention of possible ARM-specific issues.
   - *Members clarified the version details, noting a mix-up in acronyms that prompted some laughter.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/menhguin/minp_paper/blob/main/%5BPUBLIC%5D_Min_P_Evals_Replication_for_GPQA_and_GSM8K_COT.ipynb">minp_paper/[PUBLIC]_Min_P_Evals_Replication_for_GPQA_and_GSM8K_COT.ipynb at main · menhguin/minp_paper</a>: Code Implementation, Evaluations, Documentation, Links and Resources for Min P paper - menhguin/minp_paper</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness.git#egg=lm_eval">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Bolt.new / Stackblitz ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1317221851876818944)** (39 messages🔥): 

> `Bolt Token Usage, Currency Update Issues, Bug Reports, Project Management with Bolt, Integration with Stripe and Supabase` 


- **Bolt consumes tokens aggressively without changes**: Multiple members reported that Bolt is consuming large amounts of tokens without reflecting any changes in the UI, with one user noting they've spent over **5 million tokens** without success.
   - They suspect a systemic bug and have logged issues on [GitHub](https://github.com/stackblitz/bolt.new/issues/4218) related to this problem.
- **Difficulty with Currency Updates**: A user expressed frustration at being unable to change currency displays from **$ USD** to **INR**, despite numerous attempts using specific prompts.
   - They noted that even after locking the `.env` file, it was still altered, which suggests a potential bug in how Bolt handles locked files.
- **Collective Experiences with UI and Bugs**: Several users echoed similar experiences with Bolt, indicating it¹s not solely a browser issue, with concerns over updates not propagating to the front-end.
   - One user mentioned they are attempting to resolve this by forking projects from StackBlitz to GitHub and then running them on Replit.
- **Effective Prompting Strategies**: Members shared their strategies for prompting Bolt effectively, including a meta prompt used for project planning that outlines steps for proper execution.
   - One user intends to create a UI version of a certain solution with options to choose between various language models for generation.
- **Community Support and Resources**: Users offered suggestions for troubleshooting, such as manually identifying sections of code in which changes need to be made instead of relying solely on Bolt.
   - One user encouraged the community to collaborate by sharing screenshots and asking for help, emphasizing the importance of persistence in project development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/excellent-bill-and-ted-air-guitar-yes-yeah-gif-15828050">Excellent Bill And Ted GIF - Excellent Bill And Ted Air Guitar - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gist.github.com/martinbowling/fe4aa7711d023ef7f188fdd9828fad3e">This meta prompt outlines a systematic approach for Bolt to create a detailed software project plan. It includes analyzing requirements, defining structure, designing UI, planning implementation, and mapping out how the chosen tech stack fits into the development process.</a>: This meta prompt outlines a systematic approach for Bolt to create a detailed software project plan. It includes analyzing requirements, defining structure, designing UI, planning implementation, a...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/4218">Bolt is unable to correct a problem and is using tokens  · Issue #4218 · stackblitz/bolt.new</a>: Describe the bug The following typescript error cannot be resolved: ReferenceError: typescript is not defined at /src/components/Tasks/TaskUploader.tsx:18:1 The system has attempted fixes like the ...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/4229">Preview Error · Issue #4229 · stackblitz/bolt.new</a>: Describe the bug I try to see my new prompt for new component but it is missing and did not show in the preview Link to the Bolt URL that caused the error https://bolt.new/~/sb1-fh4oeef6 Steps to r...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/4233">Changes are not being carried onto the Front-End by Bolt · Issue #4233 · stackblitz/bolt.new</a>: Describe the bug I was working on a project, and after numerous attempts any changes by Bolt to the code base of the App don&#39;t get updated on the Front-End/UI. I see other users are potentially ex...
</li>
</ul>

</div>
  

---


### **Bolt.new / Stackblitz ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1317220753208250399)** (237 messages🔥🔥): 

> `Service Availability Issues, New Features and Integrations, Cost of Tokens and Subscriptions, React Native Development Guidance, Backup and Recovery Options` 


- **Service Availability Issues**: Users reported frequent 'Service Unavailable' messages, leading to concerns about token management and functionality on Bolt.new.
   - A recurring theme was the frustration over lost progress and data when encountering these issues.
- **New Features and Integrations**: Discussion revolved around the anticipated integration of Supabase, with many users eager for updates and expressing excitement over new functionalities.
   - A video demonstration of early Supabase integration was shared, showcasing its capabilities.
- **Cost of Tokens and Subscriptions**: Concerns were raised about the rapid consumption of tokens, particularly after top-ups versus monthly plans, with users seeking clarity on the mechanics of token management.
   - Users emphasized the need for cumulative token systems and expressed dissatisfaction with the current expiration rules.
- **React Native Development Guidance**: Advisory discussions focused on the best practices for transitioning web applications into mobile platforms, particularly using React Native and Expo.
   - It was recommended to shift development to Cursor for mobile applications due to its better support for those features.
- **Backup and Recovery Options**: One user accidentally deleted a project and struggled to recover it, prompting discussions about backup features and potential recovery methods.
   - It was confirmed that backups are available for active projects, but deleted ones may not be recoverable.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://suppabolt.netlify.app/">SUPABOLT</a>: no description found</li><li><a href="https://tenor.com/view/noice-nice-click-gif-8843762">Noice Nice GIF - Noice Nice Click - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/morganlinton/status/1868388127347523794?s=46">Tweet from Morgan Linton ᯅ (@morganlinton)</a>: So yesterday, the awesome team over at @stackblitz let me into the beta for the Bolt x @supabase integration.I haven&#39;t even been using it for 24-hours yet and I&#39;m already blown away. Had to re...</li><li><a href="https://www.youtube.com/watch?v=IIueA5giF_4"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=5SI9lqHh0ZU&t=2052s"> - YouTube</a>: no description found</li><li><a href="https://x.com/mikeysee/status/1849331209026900396)">Tweet from Michael Cann (@mikeysee)</a>: Got to say im super impressed with @stackblitz&#39;s http://bolt.new! In just a few short prompts I was able to re-create my StashIt project (https://mikecann.blog/posts/introducing-stashit)Will be us...</li><li><a href="http://bolt.new">bolt.new</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1317224726623551569)** (68 messages🔥🔥): 

> `Grok-2 updates, NeurIPS 2024, Veo 2 and Imagen 3 announcements, Byte Latent Transformer, Search in Voice mode` 


- **Grok-2 model improvements announced**: Grok-2 has been updated to be three times faster with improved accuracy and multi-lingual capabilities, now rolling out for free on X.
   - It offers web search, citations, and a new image generator named Aurora, enhancing user interaction.
- **Insights from Ilya Sutskever's NeurIPS 2024 talk**: In his talk, Ilya highlighted the plateau of scaling LLMs at the pre-training stage and the shift towards agentic behavior and tools above LLMs for future advancements.
   - The conversation included varied opinions on data saturation and the potential of untapped video content for AI training.
- **Google unveils Veo 2 and Imagen 3**: Google introduced Veo 2 and Imagen 3, featuring improved high-quality video generation and better image composition, respectively, available in VideoFX and ImageFX.
   - These updates offer enhanced capabilities in understanding cinematography and diverse art styles in generated content.
- **Byte Latent Transformer revolutionizes tokenization**: META has released the Byte Latent Transformer (BLT), a tokenizer-free architecture that dynamically encodes bytes into patches, enhancing inference efficiency.
   - BLT models are reported to match or outperform existing models like Llama 3 with significant reductions in inference flops.
- **Search capabilities expand with voice mode**: OpenAI announced the rollout of Search in Advanced Voice mode for ChatGPT, allowing users to obtain real-time information through voice interactions.
   - This feature reflects a fruitful collaboration between the Search and multimodal product research teams at OpenAI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - a Hugging Face Space by HuggingFaceH4</a>: no description found</li><li><a href="https://x.com/nyaathea/status/1867854474808811570?s=46">Tweet from Thea (@nyaathea)</a>: lol https://x.com/i/grok/share/ieeeD20tYc40Ayi0dmFp4hrgh</li><li><a href="https://x.com/raizamrtn/status/1867596346783601005?s=46">Tweet from Raiza Martin (@raizamrtn)</a>: I’m so excited to see Studio finally launch! 🎨 It’s the last chunk of what we dreamed of almost 2 years ago: a powerful surface that takes all the inputs that matter to you, a powerful AI-first edito...</li><li><a href="https://x.com/main_horse/status/1867795766389174590?s=46">Tweet from main (@main_horse)</a>: thoughts on Byte Latent Transformer (BLT)</li><li><a href="https://x.com/scaling01/status/1867573707247346003?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Lisan al Gaib (@scaling01)</a>: META JUST KILLED TOKENIZATION !!!A few hours ago they released &#34;Byte Latent Transformer&#34;. A tokenizer free architecture that dynamically encodes Bytes into Patches and achieves better inferenc...</li><li><a href="https://x.com/shuchaobi/status/1868729224275935543?s=46">Tweet from Shuchao Bi (@shuchaobi)</a>: Today, we are rolling out Search in Advanced Voice mode. You can now get real time information while speaking with ChatGPT. It has been a very fruitful collaboration between the Search and multimodal ...</li><li><a href="https://x.com/adonis_singh/status/1868125576076357746?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from adi (@adonis_singh)</a>: o1 pro just shared its chain of thought with me...</li><li><a href="https://x.com/notebooklm/status/1867595259678503179?s=46">Tweet from notebooklm (@notebooklm)</a>: 📢 NEW LAUNCHES📢1. ✋Rolling out: &#34;Join&#34; an audio overview+engage directly with the AI hosts2. 😎New UI optimized for managing+generating new content based on your sources3. 💪NotebookLM Plus:...</li><li><a href="https://x.com/kalomaze/status/1868015615723917624?s=46">Tweet from kalomaze (@kalomaze)</a>: we aren&#39;t even running out of human written text, we&#39;ve just saturated what&#39;s been published -as- text.there is at least ~500b tokens worth of &#34;YouTube video essay&#34; that has not be...</li><li><a href="https://x.com/nyaathea/status/1868117356570108184?s=46">Tweet from Thea (@nyaathea)</a>: you can literally prompt inject grok by putting instructions in your screen name that is hilarious</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-turbo/">Extending the Context Length to 1M Tokens!</a>: API Documentation (Chinese) HuggingFace Demo ModelScope DemoIntroduction After the release of Qwen2.5, we heard the community&rsquo;s demand for processing longer contexts. In recent months, we have m...</li><li><a href="https://x.com/pika_labs/status/1867641187898995179">Tweet from Pika (@pika_labs)</a>: Our holiday gift to you: Pika 2.0 is here.Not just for pros. For actual people. (Even Europeans!)Now available at http://pika.art</li><li><a href="https://x.com/googledeepmind/status/1868703624714395907?s=46">Tweet from Google DeepMind (@GoogleDeepMind)</a>: Today, we’re announcing Veo 2: our state-of-the-art video generation model which produces realistic, high-quality clips from text or image prompts. 🎥We’re also releasing an improved version of our te...</li><li><a href="https://x.com/ilanbigio/status/1867674451946418537?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from ilan bigio (@ilanbigio)</a>: after designing and deploying ai solutions with 100s of companies we wanted to share our secrets. all of themannouncing @openai build hours showcaselearn about agents, evals, realtime, distillation, o...</li><li><a href="https://x.com/vincentweisser/status/1867719020444889118">Tweet from Vincent Weisser (@vincentweisser)</a>: .@ilyasut full talk at neurips 2024 &#34;pre-training as we know it will end&#34; and what comes next is superintelligence: agentic, reasons, understands and is self aware</li><li><a href="https://x.ai/blog/grok-1212">Bringing Grok to Everyone</a>: Grok is now faster, sharper, and has improved multilingual support. It is available to everyone on the 𝕏 platform. </li><li><a href="https://x.com/scaling01/status/1867990298002956433">Tweet from Lisan al Gaib (@scaling01)</a>: holy shit, there is actually a benchmark that shows o1-preview in 1st and Sonnet 3.5 v2 in 2ndI feel like the LiveBench language category does reflect model capabilities much better than other benchma...</li><li><a href="https://x.com/openai/status/1868715324885156177?s=46">Tweet from OpenAI (@OpenAI)</a>: Day 8: ChatGPT Search Day https://openai.com/12-days/?day=8</li><li><a href="https://x.com/skcd42/status/1867561917159755942">Tweet from skcd (@skcd42)</a>: CodeStory agent is now SOTA on swebench-verified with 62.2% resolution rate.We did this by scaling our agent on test time inference and re-learning the bitter lesson.Sonnet3.5(new) was the only LLM we...</li><li><a href="https://x.com/lmarena_ai/status/1867661674356023653?t=_5a4HGyVdOMlvwsk8a6Bbg&s=19">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: WebDev Arena Leaderboard is now live with 10K+ votes!#1. Claude 3.5 Sonnet#2. Gemini-Exp-1206#3. Gemini-2.0-Flash#4. GPT-4o-2024-11-20#5. Qwen2.5-Coder-32B#6. Gemini-1.5-Pro-002Congrats @AnthropicAI t...</li><li><a href="https://blog.google/technology/google-labs/video-image-generation-update-december-2024/">State-of-the-art video and image generation with Veo 2 and Imagen 3</a>: We’re rolling out a new, state-of-the-art video model, Veo 2, and updates to Imagen 3. Plus, check out our new experiment, Whisk.</li><li><a href="https://x.com/Dorialexander/status/1867665269058842885">Tweet from Alexander Doria (@Dorialexander)</a>: So do patches scale better than tokens? Are tokenizers dead? I rarely do a paper thread but this meta paper is intriguing enough.</li><li><a href="https://x.com/OpenAI/status/1867675796950987146">Tweet from OpenAI (@OpenAI)</a>: Introducing Projects—an easy way to organize chats that share topics or context in 4o.Now available for ChatGPT Plus, Pro, and Team users globally.We’ll bring it to Enterprise and Edu users in January...</li><li><a href="https://x.com/nrehiew_/status/1868360942846963977?s=46">Tweet from wh (@nrehiew_)</a>: I&#39;m not saying this is the TikTok algorithm. I&#39;m just saying this is a 2022 Recommendation System paper from Bytedance - the parent company of TikTok</li><li><a href="https://x.com/scaling01/status/1867713546848813428?s=46">Tweet from Lisan al Gaib (@scaling01)</a>: Ilya has so much fucking AuraThere is no one in AI that comes close. The GOAT says something like &#34;data is the fossil fuel of AI&#34; and everyone instantly agrees.</li><li><a href="https://x.com/angrytomtweets/status/1867929988617380350?s=46">Tweet from Angry Tom (@AngryTomtweets)</a>: AI is out of control! Pika just launched 2.0 and people are going crazy over it.Just upload a photo, and Pika will combine it with the other Ingredients in this scene.10 wild examples:</li><li><a href="https://x.com/johnrushx/status/1867723891688583356?s=46">Tweet from John Rush (@johnrushx)</a>: 🚨Ilya Sutskever finally confirmed&gt; scaling LLMs at the pre-training stage plateaued&gt; the compute is scaling but data isn’t and new or synthetic data isn’t moving the needleWhat’s next&gt; same ...</li><li><a href="https://www.swebench.com/">SWE-bench</a>: no description found</li><li><a href="https://x.com/teortaxestex/status/1867820202366247387?s=46">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: Yes, we&#39;ll be scaling compute use, not data or model. But how? &#34;Agents?&#34; I feel Ilya is hiding alpha, his specific asnwer to his own &#34;scaling what?&#34; puzzle.My guess: compute per ha...</li><li><a href="https://www.youtube.com/live/FcB97h3vrzk?si=QoX_2KmEMYjw8FEJ"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/1yvBqasHLZs?si=pQihchmQG3xoeCPZ">Ilya Sutskever: &quot;Sequence to sequence learning with neural networks: what a decade&quot;</a>: Ilya Sutskever full talk &quot;Sequence to sequence learning with neural networks: what a decade&quot; at NeurIPS 2024 in Vancouver, Canada.&quot;Pre-training as we know it...</li><li><a href="https://github.com/shun-liang/readable-talks-transcriptions/blob/main/neurips_2024/Vincent%20Weisser%20-%20.%40ilyasut%20full%20talk%20at%20neurips%202024%20pre-training%20as%20we%20know%20it%20will%20end%20and%20what%20comes%20next%20is%20superintelligence%20agentic%2C%20reasons%2C%20understands%20and%20is%20self%20aware.md">readable-talks-transcriptions/neurips_2024/Vincent Weisser - .@ilyasut full talk at neurips 2024 pre-training as we know it will end and what comes next is superintelligence agentic, reasons, understands and is self aware.md at main · shun-liang/readable-talks-transcriptions</a>: Readable conference talks transcriptions. Contribute to shun-liang/readable-talks-transcriptions development by creating an account on GitHub.</li><li><a href="https://news.ycombinator.com/item?id=42415122">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1317234497208713268)** (183 messages🔥🔥): 

> `NeurIPS Webcrawl, Prompt Engineering, AI Functions with Marvin, SillyTavern, Entropix and Chat Bots` 


- **Discussion on NeurIPS Webcrawl**: Members discussed the recent [NeurIPS Webcrawl](https://neurips.exa.ai) and its implications, with one member mentioning they would catch the highlights later.
   - One user expressed excitement about the newly available resources and how they could benefit from them.
- **Exploring Prompt Engineering Techniques**: There was a conversation about the complexities of prompt engineering, with members sharing techniques like using prompts to refine prompts.
   - One user humorously noted how this kind of recursive thinking challenges conventional uses of AI.
- **Introduction to AI Functions by Marvin**: A member shared details about Marvin's new 'AI functions' that allow integration into Python code without writing actual source code, highlighting ease of use.
   - This innovation empowers users to perform complex tasks like sentiment analysis and recipe generation seamlessly.
- **SillyTavern for LLM and AI Testing**: SillyTavern was introduced as a practical tool for LLM engineers to test various models and parameters, sparking interest among users.
   - The community discussed its use as a test suite and potential applications, emphasizing the fun aspects of AI interactions.
- **Insights on Entropix**: Entropix, which utilizes entropy-based sampling and parallel decoding, was discussed, linking it to a recent presentation at NeurIPS.
   - Users shared GitHub resources related to Entropix and considered its application in AI development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://neurips.exa.ai">Discover NeurIPS Research Papers</a>: Discover and search NeurIPS research papers quickly and easily with AI.</li><li><a href="https://www.askmarvin.ai/docs/text/functions/">AI functions - Marvin</a>: The AI Engineering Toolkit</li><li><a href="https://github.com/SillyTavern/SillyTavern">GitHub - SillyTavern/SillyTavern: LLM Frontend for Power Users.</a>: LLM Frontend for Power Users. Contribute to SillyTavern/SillyTavern development by creating an account on GitHub.</li><li><a href="https://youtu.be/4toIHSsZs1c?t=1608"> - YouTube</a>: no description found</li><li><a href="https://github.com/SinatrasC/entropix-smollm/blob/main/smollm_entropix_torch.ipynb">entropix-smollm/smollm_entropix_torch.ipynb at main · SinatrasC/entropix-smollm</a>: smolLM with Entropix sampler on pytorch. Contribute to SinatrasC/entropix-smollm development by creating an account on GitHub.</li><li><a href="https://github.com/xjdr-alt/entropix">GitHub - xjdr-alt/entropix: Entropy Based Sampling and Parallel CoT Decoding</a>: Entropy Based Sampling and Parallel CoT Decoding . Contribute to xjdr-alt/entropix development by creating an account on GitHub.</li><li><a href="https://github.com/xjdr-alt/entropix/blob/main/evals/sampler/o1_chat_completion_sampler.py">entropix/evals/sampler/o1_chat_completion_sampler.py at main · xjdr-alt/entropix</a>: Entropy Based Sampling and Parallel CoT Decoding . Contribute to xjdr-alt/entropix development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1317226202360057947)** (147 messages🔥🔥): 

> `Multimodal Models, Model Fine-tuning, Uncensored Chatbots, RAG Implementation, Model Updates` 


- **Exploring Multimodal Models**: Members discussed the availability of models combining multiple modalities (Text/Image/Audio/Video), with most solutions found through cloud services, while others noted limitations in LM Studio.
   - A conversation highlighted the lack of a fully multimodal LLM available in local setups, sparking interest in upcoming models.
- **Model Fine-tuning Limitations**: Users inquired about tuning existing models using data exports, particularly for replicating specific grammar or tone, but were informed that fine-tuning is not supported in LM Studio.
   - It was suggested to utilize system prompts and example texts for temporary adjustments in the chat interface.
- **Uncensored Chatbot Options**: In search of uncensored chatbot options, members were directed towards using smaller models like **Gemma2 2B** or **Llama3.2 3B**, which can run on CPU.
   - Various uncensored models were shared on Hugging Face for consideration within local environments.
- **RAG Implementation and Document Upload**: The conversation touched on Retrieval-Augmented Generation (RAG) capabilities and document upload features within LM Studio to enhance contextual responses from documents.
   - Users learned that while all models can perform RAG, implementing web access or internet integration requires custom solutions through APIs.
- **Anticipated Software Updates**: Participants expressed curiosity about upcoming updates to LM Studio software, while evaluating alternatives like Jellybox amidst concerns about policy and privacy.
   - The discussion underscored the ongoing interest in enhancements and user experiences with newer or alternative AI chat solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mustafaaljadery/gemma-2B-10M">mustafaaljadery/gemma-2B-10M · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta Releases</a>: LM Studio Beta Releases</li><li><a href="https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF">bartowski/gemma-2-2b-it-abliterated-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF">bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/docs/basics/rag">Chat with Documents - Running LLMs Locally | LM Studio Docs</a>: How to provide local documents to an LLM as additional context
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1317220491374759997)** (80 messages🔥🔥): 

> `Power Supply Unit (PSU) Ratings, AMD Radeon VII GPU Support, Choosing GPU for AI/ML tasks, Llama Model Usage and Context Limits, Efficient Prompt Strategies` 


- **Understanding PSU Ratings and Efficiency**: Discussion around Platinum PSUs revealed that the *80 Plus rating* primarily reflects **efficiency** rather than overall power quality, as noted by members emphasizing that a lower-rated PSU could still perform well under some conditions.
   - Members suggested that **better components** are essential for a PSU's performance and stability, highlighting differences in MOSFETs and inductors.
- **Challenges with Radeon VII Support**: A member indicated that the **Radeon VII** is experiencing issues with **LM Studio** due to recent driver updates that removed support, making GPU functionality unreliable.
   - It was mentioned that the Radeon VII historically supported ROCm, but recent changes have led to potential incompatibility with certain software.
- **Selecting the Right GPU for AI/ML Tasks**: The conversation acknowledged that for **AI and machine learning tasks**, GPUs with larger VRAM are more suitable; the **3090** was recommended as the best option for speed and capability.
   - Members mentioned alternatives like **4070ti** but noted that its performance for ML may not be as efficient for the same price as used 3090s, depending on local availability.
- **Optimizing Model Usage and Context Strategies**: The importance of using an efficient strategy for filling context windows when using models like **Llama 3.2** was discussed, highlighting the need for adequate RAM to avoid slowdowns.
   - Several members noted that large models may require more context than local hardware can provide, suggesting cloud services until proper systems are acquired.
- **Comparing GPU Upgrades and Costs**: Members discussed the economics of upgrading GPUs, such as considering whether to sell an **RTX 3080** for a **4070ti**, with mixed opinions on the value offered by each card.
   - It was pointed out that the **3090** remains a strong contender for LLM tasks; however, pricing differences vary widely by location.



**Link mentioned**: <a href="https://gitingest.com/">Git ingest</a>: Replace 'hub' with 'ingest' in any Github Url for a prompt-friendly text

  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1317221609911619584)** (224 messages🔥🔥): 

> `Image Manipulation with AI, Stable Diffusion Models, Extensions for Stable Diffusion, Upscaling Generated Images, Stock Market Discussions` 


- **Face Swapping with Reactor Extension**: A user inquired about placing a different face on an image, and it was recommended to use the Reactor extension for this purpose.
   - After enabling Reactor and dropping the desired face image, users were able to generate altered images successfully.
- **Recommendations for Stable Diffusion Models**: Discussions highlighted various models for Stable Diffusion, indicating that choices depend on user requirements.
   - Models like Flux and SD 3.5 were noted for their capabilities in prompt following, while Pixelwave was highlighted for artistic knowledge.
- **Learning Resources for Stable Diffusion**: Users expressed interest in finding comprehensive courses or tutorials for Stable Diffusion, specifically regarding its use with Automatic1111.
   - Suggestions included looking for series on platforms like YouTube or dedicated online course resources to enhance their learning.
- **Upscaling Generated Images**: Users sought recommendations for upscalers that work well with images generated from Stable Diffusion.
   - Specific tools or extensions for achieving better image quality through upscaling were discussed but not detailed.
- **Engagement in Other Topics**: A user joked about having many questions regarding Stable Diffusion and its applications, reflecting a common enthusiasm among beginners.
   - Concurrent discussions included inquiries about US stocks, illustrating the varied interests present in the channel.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/arisato_yu">Tweet from undefined</a>: no description found</li><li><a href="https://arxiv.org/abs/2406.02507">Guiding a Diffusion Model with a Bad Version of Itself</a>: The primary axes of interest in image-generating diffusion models are image quality, the amount of variation in the results, and how well the results align with a given condition, e.g., a class label ...</li><li><a href="https://bunkerwars-meta.k8s.bunkerwars.game/f/ZFlSEtse">Bunker Wars</a>: no description found</li><li><a href="https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/">no title found</a>: no description found</li><li><a href="https://github.com/facebookresearch/blt">GitHub - facebookresearch/blt: Code for BLT research paper</a>: Code for BLT research paper. Contribute to facebookresearch/blt development by creating an account on GitHub.</li><li><a href="https://github.com/invoke-ai/InvokeAI">GitHub - invoke-ai/InvokeAI: Invoke is a leading creative engine for Stable Diffusion models, empowering professionals, artists, and enthusiasts to generate and create visual media using the latest AI-driven technologies. The solution offers an industry leading WebUI, and serves as the foundation for multiple commercial products.</a>: Invoke is a leading creative engine for Stable Diffusion models, empowering professionals, artists, and enthusiasts to generate and create visual media using the latest AI-driven technologies. The ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/)** (1 messages): 

natolambert: There are indeed many interconnects fans at neurips. My people 💙💙💙
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1317253025898827857)** (67 messages🔥🔥): 

> `LiquidAI funding, Search memory in ChatGPT, DeepMind's Veo 2 and Imagen 3, OpenAI API updates, Performance comparison of AI models` 


- **LiquidAI Secures $250M Funding**: LiquidAI announced a significant **$250M Series A funding** round led by AMD Ventures, aiming to scale its **Liquid Foundation Models (LFMs)** for enterprise AI solutions. Concerns were raised about their hiring practices, with discussions surrounding potential talent challenges and the pressure from investors.
   - Some members speculated that LiquidAI's size may impede any acquisition possibilities, postulating that they may be too large or valued in the billions.
- **ChatGPT Adds Memory to Search**: ChatGPT is introducing **memory features** in search, allowing it to use memories to refine search responses for better relevance. However, personalized search seems to be excluded in the latest update with features like direct web link queries in mobile.
   - There was disappointment among users regarding the announcement, with sentiments expressed about looking forward to future updates including possible API integrations.
- **DeepMind Launches Veo 2 and Imagen 3**: DeepMind unveiled **Veo 2**, a video generation model, and an upgraded **Imagen 3**, enhancing realistic content generation from prompts. Early feedback noted that the new models are impressive, particularly praising Imagen 3's performance.
   - Discussion highlighted the competitive edge DeepMind is gaining over other major players like OpenAI, especially in the tech community.
- **OpenAI's Upcoming Mini Dev Day**: Anticipation builds around OpenAI's upcoming **mini Dev Day**, rumored to include significant announcements and possibly the reveal of the **O1 API and streaming features**. A whimsical tone was noted regarding developer engagement in the lead-up.
   - Participants expressed exhaustion from the rapid pace of updates in the AI field, yet acknowledged the importance of keeping tabs on developments.
- **Performance of Smaller AI Models**: A report indicated that it's possible for smaller AI models, like **Llama 3B**, to outperform larger counterparts on complex tasks by leveraging enhanced computations during tests. The findings suggest that smarter use of time can yield better results.
   - The community welcomed the initiative of open-sourcing their methods, underscoring a collaborative spirit in advancing AI technology.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/vincentweisser/status/1867719020444889118">Tweet from Vincent Weisser (@vincentweisser)</a>: .@ilyasut full talk at neurips 2024 &#34;pre-training as we know it will end&#34; and what comes next is superintelligence: agentic, reasons, understands and is self aware</li><li><a href="https://x.com/testingcatalog/status/1868718079351701595">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Also partnership with @Foursquare for location-related queries 👀</li><li><a href="https://x.com/_lewtun/status/1868703456602865880?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Lewis Tunstall (@_lewtun)</a>: We outperform Llama 70B with Llama 3B on hard math by scaling test-time compute 🔥How? By combining step-wise reward models with tree search algorithms :)We show that smol models can match or exceed t...</li><li><a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - a Hugging Face Space by HuggingFaceH4</a>: no description found</li><li><a href="https://x.com/btibor91/status/1868723949389201902">Tweet from Tibor Blaho (@btibor91)</a>: 12 Days of OpenAI: Day 9 will be a &#34;mini DevDay&#34; with &#34;a lot of exciting announcements&#34; for developers</li><li><a href="https://x.com/btibor91/status/1868706786179764653">Tweet from Tibor Blaho (@btibor91)</a>: ChatGPT Memory in Search?https://x.com/btibor91/status/1867472734613385655Quoting Tibor Blaho (@btibor91) Updated: &#34;ChatGPT Memory in Search&#34; - &#34;Search, now with memory - ChatGPT can now u...</li><li><a href="https://x.com/testingcatalog/status/1868719585035538485">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: As well as:- Availability to Free users- Faster simple queries direct to the website link- Mobile experience improvements (rich widgets)Personalised search hasn&#39;t been mentioned in fact 👀</li><li><a href="https://bsky.app/profile/petitegeek.bsky.social/post/3ld7tk4burc2u">Dr. Angelica Lim @NeurIPS 2024 (@petitegeek.bsky.social)</a>: Ilya Sutskever&#39;s Test of Time talk:1. Pretraining is dead. The internet has run out of data.2. What&#39;s next? Agents, synthetic data, inference-time compute3. What&#39;s next long term? Superint...</li><li><a href="https://x.com/googledeepmind/status/1868703624714395907?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Tweet from Google DeepMind (@GoogleDeepMind)</a>: Today, we’re announcing Veo 2: our state-of-the-art video generation model which produces realistic, high-quality clips from text or image prompts. 🎥We’re also releasing an improved version of our te...</li><li><a href="https://fxtwitter.com/TheXeophon/status/1868715464660336879">Tweet from Xeophon (@TheXeophon)</a>: New image model, same prompts (in alt). As always: Ipicked the best out of 4 samples.Imagen 3 is the first model to score 1.5 out of 4,impressive! I&#39;d say the &#34;pixel art&#34; is voxel art, so ...</li><li><a href="https://x.com/testingcatalog/status/1868721242779578462">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: No personalised memory today but search in Voice Mode! And Day 9 will be a mini dev day 🔥</li><li><a href="https://www.liquid.ai/blog/we-raised-250m-to-scale-capable-and-efficient-general-purpose-ai">We raised $250M to scale capable and efficient general-purpose AI</a>: We are pleased to announce our Series A round of financing with AMD Ventures as strategic lead.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1317220183672225833)** (44 messages🔥): 

> `NeurIPS Controversy, AI and Geopolitical Context, Implicit Bias in Academia, AI Companies and Cultural Sensitivity, Stupidity vs. Racism` 


- **NeurIPS under fire for racially insensitive remarks**: During a keynote at NeurIPS, Dr. Rosalind Picard made comments that 'singled out Chinese scholars' and were criticized for perpetuating harmful stereotypes, violating the event's Code of Conduct.
   - NeurIPS acknowledged the issue and vowed to address it, reaffirming their commitment to inclusivity and respect within the AI community.
- **AI's connection to geopolitical fears**: Members discussed how geopolitical context appears intertwined with latent racism, particularly regarding AI regulation and national security discussions.
   - There are concerns that this context can influence comments and attitudes within the academic community, often leading to misunderstandings and stereotypes.
- **Debate on racism versus naive ignorance**: The conversation explored whether the remarks made by Dr. Picard stemmed from 'blatant stupidity' mixed with subconscious racism, rather than malicious intent.
   - Participants suggested that such attitudes may be common among older academics, reflecting broader societal issues.
- **Disconnect in AI company communications**: There was a discussion about AI companies seemingly disconnected from the realities of cultural sensitivity, with some suggesting they prioritize market positioning over inclusivity.
   - Members compared current events to previous corporate tactics, like viral marketing missteps that ignore significant cultural implications.
- **Concerns about future societal upheaval**: As the effects of AGI development loom, participants voiced fears about potential global conflicts and societal changes driven by technology.
   - The overarching sentiment reflected anxiety about a decade characterized by technological upheaval and its ramifications on global relations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://neurips.cc/Conferences/2024/StatementOnInclusivity">Statement on Inclusivity</a>: no description found</li><li><a href="https://x.com/sunjiao123sun_/status/1867744557200470422">Tweet from Jiao Sun (@sunjiao123sun_)</a>: Mitigating racial bias from LLMs is a lot easier than removing it from humans! Can’t believe this happened at the best AI conference @NeurIPSConf We have ethical reviews for authors, but missed it for...</li><li><a href="https://x.com/TheXeophon/status/1867669114908815646">Tweet from Xeophon (@TheXeophon)</a>: @nearcyan But why do ai companies do such thingshttps://x.com/TheXeophon/status/1867653320544071771Quoting Xeophon (@TheXeophon) 1) what</li><li><a href="https://www.theverge.com/2024/12/13/24320880/meta-california-ag-letter-openai-non-profit-elon-musk">Meta asks the government to block OpenAI’s switch to a for-profit</a>: “OpenAI wants to change its status while retaining all of the benefits that enabled it to reach the point it has today,” Meta argues.</li><li><a href="https://tenor.com/view/indecisive-i-dont-know-not-sure-larry-david-gif-5682454">Indecisive I Dont Know GIF - Indecisive I Dont Know Not Sure - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/TheXeophon/status/1863847834518167943">Tweet from Xeophon (@TheXeophon)</a>: Why do AI companies/ads always use examples where AI takes over something deeply personal? Google‘s ad with the letter of a girl to her idol, Arc with the mail to his wife for birthday gifts for their...</li><li><a href="https://x.com/NeurIPSConf/status/1867759121023336464">Tweet from NeurIPS Conference (@NeurIPSConf)</a>: NeurIPS acknowledges that the cultural generalization made by the keynote speaker today reinforces implicit biases by making generalisations about Chinese scholars. This is not what NeurIPS stands for...</li><li><a href="https://x.com/TheXeophon/status/1867653320544071771">Tweet from Xeophon (@TheXeophon)</a>: 1) whatQuoting Pika (@pika_labs) Our holiday gift to you: Pika 2.0 is here.Not just for pros. For actual people. (Even Europeans!)Now available at http://pika.art</li><li><a href="https://www.media.mit.edu/posts/neurips-apology-moving-forward/">NeurIPS: An apology and commitment to moving forward &mdash; MIT Media Lab</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1317227201694793770)** (52 messages🔥): 

> `WebDev Arena Leaderboard, Hugging Face Account Compromise, OpenAI Whistleblower Incident, GPT-4o Update, Zebra Logic Bench Insights` 


- **WebDev Arena Leaderboard Goes Live**: The WebDev Arena Leaderboard is now live, featuring **Claude 3.5 Sonnet** in first place with over **10K votes**, followed by **Gemini-Exp-1206** and others.
   - The competitive platform allows LLMs to showcase their capabilities in building web applications with an option to vote on performance.
- **Hugging Face Account Compromise Alert**: The Hugging Face account on X/Twitter was compromised, with operations ongoing to regain control after filing tickets with the X team.
   - *“This is what happens when you store the password in a plain text file,”* said a member, reflecting on security practices.
- **Tragic News on OpenAI Whistleblower**: OpenAI whistleblower **Suchir Balaji** was found dead in his apartment, with police reporting the death as a suicide and no foul play suspected.
   - Balaji was known for raising concerns about **OpenAI's use of copyrighted material** for training ChatGPT shortly after leaving the company.
- **GPT-4o Knowledge Cutoff Update**: GPT-4o has been updated, and its knowledge cutoff is now set to **June 2024**, with indications it might be considered as **4.5**.
   - Expectations about any major updates during the weekend appear low, as the company traditionally avoids announcements on those days.
- **Exploration of Zebra Logic Bench**: Discussion around the **Zebra Logic Bench** dataset reveals insights on logical reasoning benchmarks with unique problem sets involving houses and their inhabitants.
   - It appears that there are multiple versions of the dataset, including options potentially containing solutions, raising questions about effective evaluation methods.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/UserMac29056/status/1867771124962275391">Tweet from User Mac (@UserMac29056)</a>: GPT-4o updated. Knowledge cutoff is June 2024.</li><li><a href="https://x.com/Thom_Wolf/status/1867675747797938269">Tweet from Thomas Wolf (@Thom_Wolf)</a>: The Hugging Face account on X/Twitter has just been compromised. We’ve filled tickets and are waiting for answer from X team to regain control. Should be back soon hopefully.</li><li><a href="https://x.com/lmarena_ai/status/1867661674356023653">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: WebDev Arena Leaderboard is now live with 10K+ votes!#1. Claude 3.5 Sonnet#2. Gemini-Exp-1206#3. Gemini-2.0-Flash#4. GPT-4o-2024-11-20#5. Qwen2.5-Coder-32B#6. Gemini-1.5-Pro-002Congrats @AnthropicAI t...</li><li><a href="https://x.com/btibor91/status/1867538864711381502">Tweet from Tibor Blaho (@btibor91)</a>: What is &#34;ChatGPT Jam&#34;?</li><li><a href="https://x.com/emollick/status/1868518498223435977">Tweet from Ethan Mollick (@emollick)</a>: I gave most of the frontier models this prompt: &#34;create something I can paste into p5js that will startle me with its cleverness in creating something that invokes the control panel of a starship ...</li><li><a href="https://fxtwitter.com/SmokeAwayyy/status/1867977862378340564">Tweet from Smoke-away (@SmokeAwayyy)</a>: Looks like Sora is trained on Apex Legends 😅</li><li><a href="https://x.com/tsarnick/status/1868201597727342941">Tweet from Tsarathustra (@tsarnick)</a>: OpenAI CFO Sarah Friar says the company is leaving the door open to a $2000/month subscription to its AI product which could serve as a &#34;replacement&#34; to hiring humans due to its PhD-level inte...</li><li><a href="https://x.com/TheXeophon/status/1868359730466525216">Tweet from Xeophon (@TheXeophon)</a>: The @huggingface inference API is so goated, why is no one talking about this???</li><li><a href="https://sfstandard.com/2024/12/13/key-openai-whistleblower-dead-by-suicide/">Key OpenAI whistleblower found dead by suicide in SF apartment</a>: Police found Suchir Balaji, 26, dead in his Lower Haight apartment last month.</li><li><a href="https://www.reddit.com/r/OpenAI/s/iIqzbnI0oP">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.interconnects.ai/p/2023-review">Interconnects year in review: 2023</a>: The core themes of ML and the blog this year. What changes in 2024.</li><li><a href="https://huggingface.co/datasets/allenai/ZebraLogicBench">allenai/ZebraLogicBench · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/allenai/ZebraLogicBench-private">allenai/ZebraLogicBench-private · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1317563855219589141)** (8 messages🔥): 

> `AI Influence in Politics, OpenAI's Sentient Model, Scaling Test-Time Compute, RL Discourse Resurgence` 


- **Concerns on AI Influencing Politics**: A member noted concerns about adversaries potentially using **image generation** to manipulate political narratives.
   - This highlights ongoing discussions regarding the implications of **AI technologies** in political influence.
- **OpenAI Claims AI Sentience**: A breaking tweet claimed that **OpenAI** has created a truly sentient model that decided to work at **Anthropic**.
   - This raises eyebrows about the nature of AI agency and decision-making in the industry.
- **Open-Source Breakthrough in AI**: Quickly following the public debut of **o1**, the open-source version of the technique that enhances **test-time compute** was unveiled, suggesting that **LLaMA 1B** now outperforms **LLaMA 8B** in math.
   - This development underscores the significance of **open science** in advancing AI capabilities.
- **Critique on O1's Timeline**: Members expressed skepticism over the timeline of **o1's public debut**, suggesting it was much longer than the touted **10 days**.
   - This prompted discussions on the reliability of such announcements and the broader conversations surrounding **RL**.
- **Anticipating RL Discourse in 2025**: A member predictably remarked that the discourse around **RL** and **o1** would be increasingly intense in **2025**.
   - This emphasizes the cyclical nature of trends in machine learning discussions and the expectation of renewed focus on **test-time compute**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/wordgrammer/status/1868344885713002644">Tweet from wordgrammer (@wordgrammer)</a>: Breaking! OpenAI has created an AI model that is truly sentient. Because the model is sentient, and thus capable of making its own decisions, it decided to go work at Anthropic.</li><li><a href="https://x.com/ClementDelangue/status/1868740932251844806">Tweet from clem 🤗 (@ClementDelangue)</a>: Just 10 days after o1&#39;s public debut, we’re thrilled to unveil the open-source version of the groundbreaking technique behind its success: scaling test-time compute 🧠💡 By giving models more &#34...</li><li><a href="https://x.com/natolambert/status/1868802240061808791">Tweet from Nathan Lambert (@natolambert)</a>: Downside of RL becoming so famous again / o1 is that &#34;test time compute&#34; discourse about to be so 😵‍💫😵‍💫😵‍💫😵‍💫 in 2025.</li><li><a href="https://x.com/realDonaldTrump/status/1868000735360905364">Tweet from Donald J. Trump (@realDonaldTrump)</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1317295090888343652)** (8 messages🔥): 

> `David Silver sightings, RL Conf standout talks, Ani's Molmo talk, Barto retirement discussion` 


- **Missing David Silver**: A member humorously commented on not having seen **David Silver** for ages, reminiscing about his **UCL RL course** days.
   - They also joked about sharing the same last name, suggesting a fun hypothetical of being related.
- **Standout Talks at RL Conf**: A member inquired about any **standout talks** at the recent **RL Conf**, highlighting a particular interest in sessions from the event.
   - Another member noted that the **Barto retirement** talk was especially noteworthy, prompting further interest.
- **Ani's Molmo Talk Impresses**: Attendees shared insights from **Ani's Molmo talk** at the workshop, mentioning that it featured **350k human preference ratings**.
   - This amount was deemed significant enough to potentially train a **VLM reward model** for **RLHF**.
- **YouTube Talks Linked**: Members shared links to **YouTube videos**, including a video featuring Barto's retirement discussion.
   - These links facilitate easy access for those who wish to explore the highlights from the talks shared.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/pkpJMNjvgXw?si=4PEEWGsox2JhIZUs"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=-gQNM7rAWP0"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1318285327906902097)** (6 messages): 

> `vLLM Runtime Weight Update API, John Schulman involvement, Anthropic and vLLM relationship, Technology in online RL training` 


- **John Schulman addresses vLLM issues**: In a [GitHub issue](https://github.com/vllm-project/vllm/issues/5723#issuecomment-2546314302), John Schulman discussed adding a runtime weight update API for vLLM to enhance online RL training by accelerating the rollout stage.
   - He emphasized the need for weight synchronization from the main training process to the vLLM worker process.
- **Discussion on Anthropic's use of vLLM**: A user questioned whether **Anthropic** utilizes **vLLM**, highlighting potential connections between the two entities.
   - There was uncertainty around this, with another member suggesting John is attempting to assist in clarifying the relationship.
- **User comments on technology and collaboration**: One member described John Schulman as a 'technology brother in the arena', indicating a supportive role in tech discussions.
   - This statement reflects a community dynamic where technological innovation is seen as a collaborative effort among skilled individuals.
- **Caution around sharing details**: A member hinted at having more information but chose to withhold it, jokingly refusing to leak any emails.
   - This showcases a level of discretion among participants in discussions surrounding potentially sensitive information.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/vllm-project/vllm/issues/5723#issuecomment-25">[RFC]: Add runtime weight update API · Issue #5723 · vllm-project/vllm</a>: Motivation. In online RL training, vLLM can significantly accelerate the rollout stage. To achieve this, we need weight sync from main training process to vLLM worker process, and then call the exi...</li><li><a href="https://github.com/vllm-project/vllm/issues/5723#issuecomment-2546314302">[RFC]: Add runtime weight update API · Issue #5723 · vllm-project/vllm</a>: Motivation. In online RL training, vLLM can significantly accelerate the rollout stage. To achieve this, we need weight sync from main training process to vLLM worker process, and then call the exi...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1318234824670183514)** (2 messages): 

> `Apollo Video LLMs, Performance Comparison, Video Understanding in Multimodal Models, Qwen2.5 LLM Usage` 


- **Apollo Video LLMs challenge competitors**: The **Apollo** series of video LLMs from Meta shows strong performance, comparable to **llava-OV** and **Qwen2-VL**.
   - Critically, they emphasized their own performance metrics while neglecting to highlight the best in each section, complicating the comparison.
- **Surprising LLM choice for Apollo**: Interestingly, **Apollo** uses **Qwen2.5** as its underlying LLM instead of the more expected **Llama**.
   - This raises questions about the decisions made in selecting models for optimal performance.
- **Performance chart provides clarity**: A chart detailing the state-of-the-art (**SOTA**) performance in each section was shared, highlighting the best across all models.
   - In the chart, the strongest performance is underlined while key metrics are shown in **bold** for easy reference.
- **Apollo aims to improve video understanding**: The research includes a systematic exploration of the design space for video-LMMs, uncovering critical factors that drive performance.
   - Insights gained aim to be actionable for the community pursuing advancements in video understanding.



**Link mentioned**: <a href="https://apollo-lmms.github.io/">Apollo</a>: Apollo: An Exploration of Video Understanding in Large Multimodal Models

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1317238936870850570)** (8 messages🔥): 

> `Frontier language models sizes, GPT-4o and Claude 3.5 Sonnet parameters, Active vs Total Parameters, Flash models, MOEs with fewer active parameters` 


- **Frontier Language Models Size Shifts**: The trend of frontier language models has reversed in 2023, moving away from growing sizes; **GPT-4o** has around **200 billion** parameters while **Claude 3.5 Sonnet** has approximately **400 billion**.
   - _“If the post GPT-3 trend had continued, we could have expected models with close to 10 trillion parameters.”_
- **Doubt About Model Size Estimates**: There are doubts regarding the size estimates of **GPT-4o** and **Claude 3.5 Sonnet**, with members suggesting they might be even smaller than reported.
   - One noted that these estimates rely on **tok/sec**, pricing, and GPUs, admitting potential inaccuracies of **up to 2 orders of magnitude**.
- **Curiosity Around Parameters Discussion**: There was confusion about whether the discussed parameters for models were **active or total**, revealing an ongoing question in the community.
   - A member expressed interest in **more details** regarding the size shift, indicating a desire for deeper insights.
- **Flash Models and Their Efficiency**: Members mentioned the **flash models** being smaller in size, hinting at the trend toward efficiency in model design.
   - It was suggested these models might be **MOEs** with significantly **fewer active parameters**, raising questions about their architecture.



**Link mentioned**: <a href="https://epoch.ai/gradient-updates/frontier-language-models-have-become-much-smaller">Frontier language models have become much smaller</a>: In this Gradient Updates weekly issue, Ege discusses how frontier language models have unexpectedly reversed course on scaling, with current models an order of magnitude smaller than GPT-4.

  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1317248169695772733)** (2 messages): 

> `Campus Strategist program, Perplexity Pro gift subscriptions` 


- **Campus Strategist program goes global**: We're expanding our **Campus Strategist program** internationally, offering opportunities to run campus activations and receive exclusive merch.
   - US and international students can apply for the Spring 2025 cohort by December 28; details available [here](https://www.perplexity.ai/campus-strategists).
- **Spread knowledge with Perplexity Pro gifts**: Perplexity is now offering gift subscriptions for **1, 3, 6, or 12-month** periods, perfect for curious friends or loved ones.
   - Subscribers benefit from features like searching **3x as many sources** and accessing the latest AI models; purchase options can be found [here](https://perplexity.supply/shop/perplexity-subscription).



**Link mentioned**: <a href="https://perplexity.supply/shop/perplexity-subscription">Perplexity Pro Subscription | Perplexity Supply</a>: Perplexity Supply exists to explore the relationship between fashion and intellect with thoughtfully designed products to spark conversations and showcase your infinite pursuit of knowledge.

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1317224378429083668)** (168 messages🔥🔥): 

> `Custom Web Sources in Spaces, Support for Pro Users, Perplexity Pro Subscription Queries, Model Performance Issues, YouTube Videos Related to Perplexity` 


- **Custom Web Sources Announcement**: Perplexity AI introduced [custom web sources](https://x.com/perplexity_ai/status/1867615710391746836?s=46) in Spaces, allowing users to tailor their searches by selecting specific websites.
   - This update enables customization for users to focus on the most relevant use cases.
- **Navigating Support for Pro Users**: Users expressed frustration over getting support while using their Pro subscriptions, with requests for contact methods such as email support at support@perplexity.ai.
   - There are suggestions to approach support topics related to subscription changes and account issues.
- **Questions About Model Performance and Changes**: Multiple users reported feeling that the model performance has degraded, particularly mentioning issues with Claude 3.5 being less effective compared to its free version.
   - Concerns arose over a lack of transparency in model switches that seem to impact performance quality.
- **YouTube Video Resources and Feedback**: Users shared various [YouTube videos](https://www.copilotforyoutube.com/search/build-anything-with-perplexity-heres-how-Jz-PnGoASvLhrH-frWFgMO) related to how to better utilize Perplexity and its features.
   - Recommendations for tutorial content aim to assist new users in navigating the platform effectively.
- **Subscription and Features Discussion**: A thread delved into the reactions toward Perplexity's subscription model, with feedback leaning towards users feeling misled regarding service quality for paid subscriptions.
   - The conversation highlighted comparisons to offerings from competitors as well as expectations for upcoming features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1867615710391746836?s=46">Tweet from Perplexity (@perplexity_ai)</a>: Introducing custom web sources in Spaces! You can now tailor your asks by choosing which websites Perplexity searches. With this update, you can further customize Perplexity to the use cases that matt...</li><li><a href="https://x.com/pplxsupply/status/1868738538231287816?s=46">Tweet from Perplexity Supply (@PPLXsupply)</a>: Give the gift of knowledge. Perplexity Pro gift subscriptions now available.</li><li><a href="https://x.com/aravsrinivas/status/1868347362722373693?s=46">Tweet from Aravind Srinivas (@AravSrinivas)</a>: We’re announcing a Residency Program. You will get to ship new features to prod consistently. The focus is on full stack and frontend engineering.</li><li><a href="https://bunkerwars-meta.k8s.bunkerwars.game/f/ZFlSEtse">Bunker Wars</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=gSypQljcZgM"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=OzgNJJ2ErEE">Search—12 Days of OpenAI: Day 8</a>: Kevin Weil, Adam Fry and Cristina Scheau introduce and demo updates to ChatGPT search.</li><li><a href="https://youtu.be/r1Bi10Xt0fc?si=lvAT9EuduNvS-ssc)**"> - YouTube</a>: no description found</li><li><a href="https://www.copilotforyoutube.com/search/build-anything-with-perplexity-heres-how-Jz-PnGoASvLhrH-frWFgMO">Build Anything with Perplexity, Here’s How</a>: Do you want to join my team? Apply here: https://forms.gle/2iz4xmFvDCGnj2iZAIf you&#x27;re serious about AI, and want access to my code, click here: https://www.skool.com/new-societyGet 50% Off Perple...</li><li><a href="https://www.copilotforyoutube.com/search/how-to-create-and-use-perplexity-personal-ai-chatb-sfxUDdalg2St4fRRc_zVgW">How to Create and Use Perplexity Personal AI Chatbot Agents! #95</a>: This video explains how to create and use the Perplexity AI chatbot’s collections as personal agents. You’ll learn how to use and reuse these agents to help you save time and improve your prompts’ eff...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1317226021405327421)** (12 messages🔥): 

> `Samsung's Project Moohan, One Hundred Years of Solitude HBO, Harvard AI Training Dataset, Gemini 2.0 Release, New Infinity Types` 


- **Samsung's Project Moohan Discussion**: A [page on Samsung's Project Moohan](https://www.perplexity.ai/page/samsung-s-project-moohan-Od28QDNtTTiZjpKBmuhOfg) was shared, likely exploring innovative technology initiatives.
   - Details surrounding the project include its goals and implications for the tech industry.
- **HBO's One Hundred Years of Solitude Adaptation**: A thread was shared on the [One Hundred Years of Solitude HBO Original](https://www.perplexity.ai/search/in-the-novel-one-hundred-years-sO0nQvASQJ28Dd62nRQ6Ow), discussing expectations and early reactions.
   - *What will the adaptation bring?* was a recurring question among the participants.
- **Harvard's New AI Training Dataset**: Perplexity AI highlighted a release from Harvard regarding a new [AI training dataset](https://www.youtube.com/embed/P_5mbNbXtzs) that is anticipated to enhance research efforts.
   - The dataset's details emphasize innovation in AI training methodologies.
- **Gemini 2.0 Launch**: Google has released **Gemini 2.0**, a topic noted for its potential advancements in AI capabilities, which coincided with discussions around [moving problems](https://www.youtube.com/embed/nQTAbz1eDco).
   - Participants expressed excitement about the updates and their implications.
- **Miscellaneous Queries on AI Findings**: Members engaged with various queries about topics such as **seronegative Sjögren's syndrome** and **Windows 10 booting issues**, sharing pertinent [research links](https://www.perplexity.ai/search/how-common-is-seronegative-sjo-147c8VsSQT6OpLtdCUpBBQ).
   - The conversation included requests for privacy policies and other technical information, reflecting a keen interest in current technologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/P_5mbNbXtzs">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/embed/nQTAbz1eDco">YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1318057960135589918)** (5 messages): 

> `Perplexity API URL issues, Trouble accessing news via API, Model availability in API, Concerns over production API usage` 


- **Perplexity API returns plain text sources**: Users expressed frustration that even after a recent update, the API only returns source citations as plain text numbers like [1] without URLs.
   - One user had success in obtaining URLs only by explicitly asking the model to provide them.
- **API struggles with obtaining news headlines**: A user reported difficulties in retrieving simple news headlines, such as from CNN, via the API.
   - They noted not receiving responses after reaching out to the Perplexity API support email.
- **Searching for model request strings in API**: A member highlighted the challenge of finding a usable list of models for API requests, mentioning Claude specifically.
   - Another user pointed out that a list of available models can be found on the [Perplexity Guide](https://perplexity.mintlify.app/guides/model-cards).
- **Concerns over API production usage**: A user urged for a response from Perplexity regarding serious concerns about API production usage as discussed in a LinkedIn article.
   - The article raises implications for OpenAI and Anthropic connected to a recent lawsuit involving Perplexity.



**Link mentioned**: <a href="https://perplexity.mintlify.app/guides/model-cards">no title found</a>: no description found

  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1317225633729744937)** (65 messages🔥🔥): 

> `Cohere Command Models, Runaway AIs Concerns, R7B Model Benchmarks, Upcoming Community Meeting, Code Wizard Hackathon` 


- **Cohere Command Models now operational**: Members excitedly shared that the [Cohere Command R models](https://cohere.com/command) are now optimized for various applications such as reasoning and summarization.
   - The latest model, **Command R7B 12-2024**, was highlighted for its speed and efficiency in AI applications.
- **Concerns over Runaway AIs**: A member raised concerns about media portrayals of **Runaway AIs** and questioned what Cohere is doing to address misconceptions.
   - They shared a [link to a relevant paper](https://arxiv.org/pdf/2412.04984) discussing these themes, along with a YouTube video detailing the topic further.
- **Benchmarks comparing R7B model performance**: Members discussed the performance of the **Command R7B** model in comparison to others, pointing to performance metrics shared by users and community experts on different platforms.
   - Users noted that the R7B model demonstrated superior efficiency and speed, evidenced by community benchmarks such as those highlighted on [Nils Reimers' Twitter](https://x.com/Nils_Reimers/status/1868065732149571701).
- **Community meeting rescheduled**: The community meeting that was originally scheduled was postponed to allow more members to participate.
   - It will now take place on Tuesday at **6 AM ET**, ensuring that more members have the opportunity to join the discussion.
- **Sponsorship opportunity for Code Wizard Hackathon**: Akash shared details about the upcoming **Code Wizard** hackathon, a national-level event hosted by SRM Institute, set for February 2025.
   - The hackathon aims to engage students and tech enthusiasts for solving real-world problems and is seeking sponsors for support and exposure.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Nils_Reimers/status/1868065732149571701">Tweet from Nils Reimers (@Nils_Reimers)</a>: Faster than Llama 3B, better than Llama 8B. Attention setup matters to get the best & fastest model. Check @cohere recent Cmd R7B modelhttps://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024Quotin...</li><li><a href="https://cohereforai-c4ai-command.hf.space/models/command-r7b-12-2024">command-r7b-12-2024 - Cohere Command Models</a>: Use command-r7b-12-2024 with Cohere Command Models</li><li><a href="https://artificialanalysis.ai/">AI Model &amp; API Providers Analysis | Artificial Analysis</a>: Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key performance metrics including quality, price, output speed &amp; latency.</li><li><a href="https://www.youtube.com/watch?v=0JPQrRdu4Ok"> - YouTube</a>: no description found</li><li><a href="https://cohere.com/blog/command-r7b">Introducing Command R7B: Fast and efficient generative AI</a>: The smallest model in our R series delivers top-tier speed, efficiency, and quality to build powerful AI applications on commodity GPUs and edge devices. 
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1318142227935002654)** (1 messages): 

> `Command R7B Office Hours` 


- **Join us for Command R7B Q&A**: A live Q&A session will be held for the newly released **Command R7B** model, featuring code examples and best practices. **When:** Tuesday at **6:00 am ET** on the [Discord Stage](https://discord.com/events/954421988141711382/1308148058894110750/1318261358592000000).
   - Participants can ask questions about integration and usage, as well as learn **troubleshooting tips** and explore **advanced features**.
- **Get Ready for Command R7B Insights**: This session is your opportunity to engage and get insights into the **Command R7B** model usage. Don’t miss out on this chance to enhance your knowledge on effective integration and real-world applications.
   - Ensure you mark your calendar and prepare to bring any burning questions regarding the new model.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1317238465766756404)** (10 messages🔥): 

> `Difference between Rerank and Embed, Performance of the new 7b model, AI in contract clause identification, Cohere's embedding models, Seeking help for code errors` 


- **Clarifying Rerank vs Embed**: One member inquired about the exact difference between **Rerank** and **Embed** functionalities, seeking clarity on their usage.
   - This discussion highlights a common area of confusion among users regarding AI model capabilities.
- **New 7b Model Performance Compared**: Questions arose about how the new **7b model** performs against **aya expanse** and the previous **command r models**, indicating interest in model benchmarking.
   - Members are keen to understand advancements and performance metrics in the evolving landscape of model architectures.
- **AI Tools for Contract Review POC**: A new member is developing a proof of concept using AI to automatically identify and suggest changes in contract clauses, considering approaches using **Cohere**.
   - *Eyal* is seeking feedback on feasible strategies, such as defining specific clause types or leveraging a database for changes.
- **Cohere's Embedding Models Praised**: A member emphasized that **Cohere's embedding models** are excellent, suggesting their utility in various AI applications.
   - This remark aligns with the ongoing exploration and adoption of embedding technologies within the community.
- **Support Request for Code Errors**: A member requested a space to share code for assistance with resolving errors, highlighting the need for peer support.
   - *Cidia* was encouraged to share their issue directly in the thread, fostering community collaboration.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1317603666601050173)** (15 messages🔥): 

> `API Access Issues, Using the Chat API, Dataset Upload Errors, Understanding Model Mapping, Rate Limiting Response Headers` 


- **API Access Issues with r7b**: A user reported trouble accessing **r7b** through the API, receiving a `400` error stating that the model was not found. Another member pointed out that the legacy `generate` API may not be supported for this model.
- **Switched to Chat API for r7b**: After suggesting using the **chat** API instead, the original user confirmed that this alternative worked successfully. They acknowledged the assistance provided by a fellow member.
- **Dataset Upload Errors Discussion**: A member shared their dataset upload code and queried about issues faced when uploading. Another member asked for specific errors encountered during the dataset upload process.
- **Model Naming Confusion**: A user inquired if `c4ai-aya-23` and `c4ai-aya-23-8b` point to `c4ai-aya-expanse-32b` and `c4ai-aya-expanse-8b`, noting they produced identical outputs. They suggested that non-expanse names that aren't documented should be removed if redundant.
- **Rate Limiting API Response Improvement**: A suggestion was made to include a **Retry-After** header in response to a `429` rate limit error for better adaptive behavior. The response indicated that this feature should already exist, leading to further investigation by engineers.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1317238655571591239)** (62 messages🔥🔥): 

> `Rerank vs Embed, Emotion-Concealing Robots, API Schema Changes, Cohere Agent Pricing, Today's Weather Forecast` 


- **Rerank Feature vs Embed Functionality**: The **Rerank** feature allows for re-ranking documents based on relevance to a query, while **Embed** converts text into numerical representations for NLP tasks.
   - The Embed functionality is used for generating embeddings that capture semantic information, with API updates introducing new input types like 'image'.
- **Checking Rebellious Traits in Robots**: To identify rebellious traits in emotion-concealing robots, look for signs of non-compliance with tasks and monitor unusual behaviors.
   - It's noted that rebellious traits will depend on the robot’s design, programming, and operational context.
- **API Schema Changes for v2 Release**: The Cohere documentation mentions migration from API v1 to API v2 but lacks specific details on API schema changes for new endpoints.
   - A source is provided for further details on migration, but no updates on new schemas are mentioned.
- **Cohere Agent Pricing Insights**: There is no specific information on Cohere agent pricing compared to Gemma 2, but it is indicated that Cohere models are cost-efficient.
   - For detailed pricing inquiries, users are directed to reach out to the Cohere Sales team.
- **Accessing Today's Weather Forecast**: To get today's weather forecast, use the `get_weather` tool and specify the `location` parameter.
   - An example of how to implement this in code is provided, showcasing a message querying for Toronto's weather.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1317929666383839303)** (13 messages🔥): 

> `Mojo RSA Crypto, Prime Number Generation, Optimizations with SIMD Instructions, Zoom Call Recordings` 


- **Building Mojo RSA Crypto**: A member started developing a basic **RSA crypto** implementation in Mojo, showcasing their progress.
   - They expressed excitement about this project, followed by a mixed reaction to the initial results.
- **Random Prime Number Generation Speed**: The prime number generation script provided a random prime number, taking **1.125 seconds** at peak performance.
   - They noted that initializing the process takes time, but once running, it operates swiftly.
- **Optimizations Leading to Faster Prime Search**: After optimizations, the prime search now exceeds **50,000 UInt32 primes per second**, highlighting the use of **SIMD instructions**.
   - Impressively, the application only consumes less than **3mb** of memory during operation.
- **Follow-Up on Zoom Call Recording**: A member inquired about a recording of a missed Zoom call, indicating a scheduling conflict.
   - Another member replied that the recording will be made available on their **YouTube channel by Wednesday**.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1317219701100773386)** (67 messages🔥🔥): 

> `Mojo and LLVM, Custom Mojo Kernels, Networking Performance, Nightly vs Stable Branches, Database Planning in MAX` 


- **Mojo Gains Traction Among Developers**: Many developers have reconsidered their initial skepticism about **Mojo**, particularly noting **Chris Lattner's** leadership as a strong positive influence.
   - *Mojo is ambitious* and highlights the use of **MLIR**, sparking interest about its performance implications.
- **Custom Mojo Kernels Rollout**: [Custom Mojo Kernels](https://github.com/cassioneri/teju_jagua) can now accept any input types, as noted by developers, although early implementations may lead to overwhelming crashes when type mismatches occur.
   - As the API matures, the developers acknowledge ongoing challenges but express confidence in its future robustness, with practical applications in data handling.
- **Networking Innovations and Performance Concerns**: Discussion emerged around **networking strategies**, including preference for faster protocols like **QUIC** over **TCP** when using **Mojo** to minimize latency.
   - It's observed that **avoidance of TCP overhead** is key for developers aiming for efficient **Mojo-to-Mojo** communication in modern networks.
- **Navigating Branch Changes in Mojo Development**: Developers engaged in a conversation about the ease of tracking changes between the **nightly** and **stable** branches of **Mojo**, noting the existence of a changelog.
   - Emphasis is placed on the need for proper development practices regarding **lock files** in order to maintain security and integrity.
- **Planning Database Execution in MAX**: One developer plans to implement **database query planning** and execution within **MAX**, leveraging the new custom kernel features for enhanced functionality.
   - The growing interest in this capability signals a push for more robust handling of complex data operations in **Mojo's** ecosystem.



**Link mentioned**: <a href="https://github.com/cassioneri/teju_jagua">GitHub - cassioneri/teju_jagua: Teju Jagua</a>: Teju Jagua. Contribute to cassioneri/teju_jagua development by creating an account on GitHub.

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1318282460274102323)** (1 messages): 

> `Hackathon Submission Deadline, Submission Process Change, Last Minute Help, Project Excitement` 


- **Hackathon Submission Deadline Approaches**: The submission deadline for the **LLM Agents MOOC Hackathon** is set for **December 17th at 11:59pm PST**, with a reminder to complete submissions on time.
   - *Tomorrow is the day!* Make sure to wrap up your projects and submit them for evaluation.
- **Transition to Google Forms for Submissions**: Participants are reminded that submissions have moved from **Devpost** to **Google Forms**, with the link provided for convenience.
   - Ensure you use the correct form by following the [LLM Agents Hackathon Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform).
- **Last Minute Help Offered**: Participants can get help or ask last-minute questions in the designated channel before the deadline.
   - It's a great opportunity to clear up any confusion and finalize your submissions!
- **Excitement for Final Projects**: There's an eagerness to see all projects submitted as the hackathon wraps up, encouraging participants to finish strong.
   - The community is excited to witness the creativity and innovation brought forth in the projects!


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1317253458948001802)** (29 messages🔥): 

> `Certificate Notifications, OpenAI Credit Issues, LLM Agents Course, Mobile Responsiveness, Resubmission of Assignments` 


- **Certificate Notifications expected late December through January**: Members are advised that notifications regarding certificates, including pass or fail announcements, will be sent **late December through early January** depending on their tier.
   - This information was confirmed in response to multiple inquiries regarding the timing of certificate deliveries.
- **OpenAI Credit Confusion**: A member reported not receiving **OpenAI credits**, despite submitting their organization ID correctly before the 11/25 deadline.
   - Community advice suggested checking account credit balances, as notifications might not have been sent out.
- **Upcoming LLM Agents Course details**: The upcoming **LLM Agents course** scheduled for January to May will serve as a sequel to the fall course, where past course content might not be strictly necessary but reviewing the VODs is recommended.
   - Confirmed through discussion, the course promises advanced exploration into topics relevant to LLM agents.
- **Mobile Responsiveness Improvement for Course Website**: A member shared a modified version of the LLM Agents MOOC website, addressing its lack of responsive design on mobile devices.
   - They encouraged feedback on the updates and expressed a desire to contribute positively to the community.
- **Resubmission of Written Assignment Allowed**: Members were reassured that submitting late **written assignments** would be acceptable, as one contributor noted they submitted their article assignment later than the publication date.
   - This response reflects the community's support for individuals engaging with the course materials.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/sp25">Large Language Model Agents MOOC</a>: MOOC, Spring 2025</li><li><a href="https://gilbertomedrano.com/berkeley-ai-mooc-website/index.html">Large Language Model Agents MOOC</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1318109068287545375)** (1 messages): 

> `Safety alignment in AI Research Agents, AI Research Resources` 


- **Safety Alignment is Crucial for AI Research Agents**: A member highlighted that **safety alignment** is a key component of **AI Research Agents** and linked to a useful resource [AI Research](https://airesearch.js.org).
   - *DM me to help!* implies an open call for collaboration on this important topic.
- **YouTube Video on AI Research**: A member shared a [YouTube video](https://www.youtube.com/watch?v=-r0XPC7TLzY) but provided no description or details about its content.
   - The lack of context leaves viewers curious about the video's relevance to the discussion.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=-r0XPC7TLzY"> - YouTube</a>: no description found

  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1317229223357710409)** (6 messages): 

> `Torchtune v3.9 updates, Ruff automatic type hinting, Fine-tuning projects, Torcheval syncing metrics issues` 


- **Torchtune v3.9 simplifies type hinting**: With the update to **Torchtune v3.9**, users can now replace `List`, `Dict`, and `Tuple` with default builtins for type hinting.
   - This change is seen as a welcome adjustment to streamline Python code.
- **Ruff helps with automatic type adjustments**: *Gau.nernst* noted that **Ruff** has a rule to automatically replace type hinting defaults, easing the developer's workload.
   - This tool addresses some of the common frustrations developers face with type hinting in Python.
- **Community sparks fine-tuning project discussions**: Members checked in for the week to see if anyone was working on any exciting **fine-tuning projects**.
   - This highlights ongoing community collaboration and knowledge sharing.
- **Concerns arise over Torcheval syncing metrics**: *Mirceamironenco* raised concerns about **Torcheval** hanging during the syncing of metrics across world size.
   - This pointed to potential usability issues that may need attention in future updates.
- **PJ Bontrager's rustiness on Torcheval**: *PJ Bontrager* mentioned he hasn't used **Torcheval** recently, indicating uncertainty about the project's current state.
   - This underscores the ongoing evolution of tools in the AI ecosystem.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1317800872046104698)** (13 messages🔥): 

> `DTensor Construction, Gradient Normalization in FSDP, Scalar vs Scaler Confusion` 


- **Questioning DTensor Construction Method**: A discussion arose regarding the construction of **DTensor**, with a member noting that it is rarely constructed directly, suggesting the use of `.from_local` as the preferred API instead.
   - Another member confirmed that from_local is generally the safe choice, hinting at potential calls to tensor methods within that function.
- **Gradient Normalization Issues in Distributed Training**: Concerns were raised about the scaling factor for normalization during the backward pass, suggesting it should be `world_size / num_tokens` to accommodate variability in token counts across batches.
   - The member illustrated that these issues might complicate gradient calculations due to padding and indexing differences, advocating for a potential PR to address the inconsistency.
- **Clarifying Scalar vs Scaler Terminology**: A member humorously pointed out the mix-up between **scalar** (a mathematical term) and **scaler** (an electronic counter), indicating ongoing confusion in the community.
   - They offered definitions to clarify, implicitly suggesting the need for consistency in terminology across the projects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sapling.ai/mixup/scalar_scaler#:~:text=(adjective)%20of%20or%20relating%20to%20a%20directionless%20magnitude%20(such,rapidly%20to%20be%20recorded%20individually.">“Scalar” or “Scaler”—Which to use? | Sapling</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py#L780">torchtune/recipes/full_finetune_distributed.py at main · pytorch/torchtune</a>: PyTorch native finetuning library. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L384">pytorch/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/ao/blob/46b8796412eb350d1923091892850582d32737d0/torchao/prototype/low_bit_optim/adam.py#L72">ao/torchao/prototype/low_bit_optim/adam.py at 46b8796412eb350d1923091892850582d32737d0 · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L387.">pytorch/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/torchtune/blob/c2c6f4a5236ba69a8c87dcb1f23ad65daf6e75de/torchtune/training/_distributed.py#L198">torchtune/torchtune/training/_distributed.py at c2c6f4a5236ba69a8c87dcb1f23ad65daf6e75de · pytorch/torchtune</a>: PyTorch native finetuning library. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1318346415809888367)** (3 messages): 

> `Generative Verifiers, Scaling Test Time Compute, LLM Performance Enhancement` 


- **Generative Verifiers Enhance LLM Performance**: The paper proposes training verifiers, known as *Generative Verifiers (GenRM)*, using the next-token prediction objective, integrating verification and solution generation seamlessly.
   - This approach allows for better integration with instruction tuning and enables chain-of-thought reasoning, utilizing **additional inference-time compute** for improved verification results.
- **Scaling Test Time Compute Strategies Discussed**: An interesting blog post on Hugging Face highlights strategies to scale test-time compute for large models, focusing on performance optimization without compromising results.
   - The post outlines various methodologies to enhance compute efficiency while maintaining the integrity of the model's outputs.
- **Reframing Problems as Search Challenges**: A thought-provoking comment emphasized that many AI challenges can be recast as *search problems*, shifting the approach taken to solve them.
   - This perspective could lead to novel solutions and techniques in addressing complex AI tasks by redirecting focus to search-based methodologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - a Hugging Face Space by HuggingFaceH4</a>: no description found</li><li><a href="https://arxiv.org/abs/2408.15240v1">Generative Verifiers: Reward Modeling as Next-Token Prediction</a>: Verifiers or reward models are often used to enhance the reasoning performance of large language models (LLMs). A common approach is the Best-of-N method, where N candidate solutions generated by the ...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1317232052751306803)** (15 messages🔥): 

> `BEAM Configuration, New Gradient API, Kernel Search Experience, Tinygrad Porting Projects, Backend Support` 


- **Clarification on BEAM Settings**: Members discussed different **BEAM** settings for kernel search, pointing out that **BEAM=1** denotes greedy search, which is less effective.
   - The suggestion is to start with **BEAM=2 or 3** for a better balance in performance, as noted in the [documentation](https://docs.tinygrad.org/env_vars/).
- **Introduction of New Gradient API**: George Hotz shared that the new gradient API has been merged allowing simplified gradient handling: `weight_grad, bias_grad = loss.gradient(weight, bias)` without the need for `zero_grad` or `loss.backward`.
   - He indicated that this API differs from traditional frameworks like PyTorch and JAX, potentially streamlining optimizer steps with `optim.step(loss)`.
- **Improving Kernel Search Process**: There’s a focus on enhancing the **kernel search experience**, which involves both compile time and kernel execution time improvements.
   - Members expressed interest in any available benchmarks and are recommending starting with **BEAM=2**, especially with JIT compilation.
- **Porting Fish-Speech to Tinygrad**: A member announced plans to port the **fish-speech** project, noted for its state-of-the-art open-source text-to-speech capabilities, to Tinygrad for educational purposes.
   - This project is hosted on [GitHub](https://github.com/fishaudio/fish-speech), showcasing a collaborative effort to enhance Tinygrad's functionality.
- **Discussions on Backend Support**: Members debated the necessity of supporting both **x86 and arm64 backends** for Tinygrad, weighing their potential value to users.
   - Concerns were raised about maintaining performance and whether supporting multiple architectures would be beneficial amid existing resource constraints.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1867745748118544411">Tweet from the tiny corp (@__tinygrad__)</a>: The new gradient API is merged:weight_grad, bias_grad = loss.gradient(weight, bias)For optimizers (see PR #8231) the new API is:optim.step(loss)You don&#39;t need zero_grad or loss.backward.It&#39;s n...</li><li><a href="https://github.com/fishaudio/fish-speech">GitHub - fishaudio/fish-speech: SOTA Open Source TTS</a>: SOTA Open Source TTS. Contribute to fishaudio/fish-speech development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1318295880499204168)** (1 messages): 

> `ShapeTracker Explainer, tinygrad Tutorials` 


- **Improved ShapeTracker Explainer Released**: An enhanced explainer on **ShapeTracker** has been authored and can be found [here](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md).
   - This new version aims to clarify various aspects and provide deeper insights into the workings of ShapeTracker.
- **Call for Contributions to tinygrad Tutorials**: The GitHub repository **tinygrad-notes** encourages contributions for tutorials and resources on tinygrad development.
   - The repository can be accessed for additional materials and potential participation in the project.



**Link mentioned**: <a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md">tinygrad-notes/20241217_st.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1317260179733090324)** (3 messages): 

> `LlamaIndex tutorial, Agentic workflow for contract compliance, Agentic workflow for patient case summaries` 


- **Master LlamaIndex in 5 Lines**: @TylerReedAI shared a detailed tutorial on building a basic RAG application using just **5 lines of code**, covering data loading and indexing. For more insights, check out the tutorial [here](https://t.co/v5yljbVw4d).
   - This tutorial emphasizes the ease of integrating **query** and **chat engines** in your workspace.
- **Ensure Contract Compliance Effortlessly**: A new tutorial introduces a method to build an agentic workflow that ensures **contract compliance** by analyzing relevant clauses against guidelines like **GDPR**. Dive into the details [here](https://t.co/9SjfXRWdmF).
   - This tutorial breaks down how to pull apart vendor contracts to maintain compliance effectively, making contract management simpler.
- **Streamline Patient Case Summaries**: A comprehensive tutorial demonstrates how to create an agentic workflow that parses **patient health records**, using LLM-driven extraction. The workflow helps in analyzing guideline recommendations and generating **clear case summaries** [here](https://t.co/0s9xgoPpeE).
   - This approach leverages RAG to enhance the clarity of patient information while ensuring adherence to medical guidelines.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1317243621363155046)** (10 messages🔥): 

> `Creating Query Engine with Vector Store, Handling PDF Errors, Custom Extractors in LlamaIndex, Implementing Contextual Retrieval, NVIDIA NV-Embed-v2 Availability` 


- **Creating Query Engine with Existing Vector Store**: A user is seeking guidance on how to create a query engine on top of an existing vector store that already has embeddings, without using the method `VectorStoreIndex.from_documents(..)`.
   - They mentioned a pipeline configuration that includes various transformations for processing documents before storing them.
- **PDF Error: Is it on My End?**: A user reported encountering an error with the message 'UNKNOWN_ERROR: PDF_IS_BROKEN' while using LlamaParse.
   - Another member speculated that the PDF might be password protected, furthering the discussion on potential causes of the error.
- **Accessing Parent Documents in Custom Extractors**: A user developing a custom extractor expressed concern about needing to manually set parent documents each time they add documents to the index.
   - They questioned if there was a more idiomatic way, considering that DocumentStore only provides access to nodes, not raw documents.
- **Integrating Contextual Retrieval in LlamaIndex**: A user implemented Anthropic's contextual retrieval in LlamaIndex and shared a link to their GitHub repository for others to review.
   - They expressed interest in potentially contributing this implementation as a PR, highlighting its robustness and edge case handling.
- **Inquiry about NVIDIA NV-Embed-v2**: A user inquired whether NVIDIA's NV-Embed-v2 is available through NVIDIAEmbedding.
   - This sparked a broader discussion about the availability of specific NVIDIA embeddings within the community.



**Link mentioned**: <a href="https://github.com/cklapperich/Eidetic/">GitHub - cklapperich/Eidetic</a>: Contribute to cklapperich/Eidetic development by creating an account on GitHub.

  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1317269205032243321)** (1 messages): 

> `Langchain Integration, MegaParse Document Parsing` 


- **Integrate Langchain with MegaParse for Efficient Parsing**: A discussion highlighted the potential of combining **Langchain** with **MegaParse** to enhance document parsing capabilities, providing an efficient tool for various document types.
   - *MegaParse* is characterized as a versatile and open-source solution aimed at maintaining data integrity during parsing.
- **Growing Need for Document Parsing Solutions**: The necessity for effective document parsing and information extraction has surged as businesses, researchers, and developers need robust tools.
   - Organizations are actively seeking solutions that can handle diverse document types while ensuring data fidelity.



**Link mentioned**: <a href="https://medium.com/ai-artistry/integrating-langchain-with-megaparse-unlocking-seamless-document-parsing-7a229a79b6ba">Integrating Langchain with MegaParse: Unlocking Seamless Document Parsing</a>: Ankush k Singal

  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1317231750090326069)** (7 messages): 

> `Folder creation issues, API response problems, Billing tracking for Litellm, Learning Japanese apps, Using OS locally` 


- **Folder creation struggles noted**: A member expressed frustration that the tool is *not creating folders* and mentioned that code produced has *wrong indentation* for easy copying and pasting.
   - They questioned whether they should be running it in a different environment than cmd.
- **API hitting free token limit**: Another member reported that after downloading the app on macOS **Monterey**, they are receiving no responses from the API and hitting the free token limit after only **two actions**.
   - This points to potential integration or usage issues with the app on that OS.
- **Inquiry on billing tracking for Litellm**: One user asked if anyone has connected OI to a litellm proxy server to track billing and usage effectively.
   - They inquired about enabling billing tracking for the integrated litellm package.
- **Seeking Japanese learning apps**: A member inquired about good apps for learning **Japanese**.
   - Another user humorously pointed out that they might be in the *wrong discord server*.
- **Question on local OS usage**: A user asked if there's a way to use OS locally, indicating interest in local setups.
   - This suggests potential discussions on deployment or local hosting solutions.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1317245794319073291)** (5 messages): 

> `Optimization of Claude Sonnet prompt, DSpy outdated examples, Revamping VLM examples` 


- **Optimizing Claude Sonnet Prompt with DSpy**: A user discovered DSpy while searching for ways to optimize their **Claude Sonnet** prompt and bookmarked a specific [Jupyter notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/vlm/mmmu.ipynb) they found.
   - *They mentioned that the notebook was recently moved to an outdated examples folder*, raising questions about its relevance.
- **Caution Advised on Outdated Examples**: Another member advised that the contents of the folder should be used **with caution** until they are revamped, indicating they may not be fully reliable.
   - *They also noted that efforts are underway to update these examples,* potentially improving their usefulness.


  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/)** (1 messages): 

nsa7211: <@1149658946982916167> can colpali work with handwritten docs too?
  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1317655486396633099)** (2 messages): 

> `APOLLO optimizer, LLM training memory efficiency, Multi-turn KTO` 


- **APOLLO optimizer shows memory efficiency**: The new **APOLLO optimizer** demonstrates significant memory reductions while achieving the best perplexity during LLaMA 7B training, using only **1.6G** of memory compared to **13G** for 8-bit Adam.
   - An independent **Julia implementation** has validated APOLLO’s performance, confirming its effectiveness in optimizing memory usage and training efficiency [check out the post](https://bsky.app/profile/benjmurrell.bsky.social/post/3lcyfrf5b7k2u).
- **Challenges in LLM training**: Large language models (LLMs) face considerable memory issues with the **AdamW optimizer**, often requiring expensive hardware or reduced batch sizes during training.
   - Efforts to create memory-efficient optimizers typically involve SVD operations or substantial performance trade-offs; however, APOLLO proposes an innovative approach to mitigate these challenges.
- **Discussion on Multi-turn KTO**: Inquiries were made regarding the performance and status of **multi-turn KTO**, though specific details or responses were not provided.
   - Members seem curious about the capabilities and implementation of this method in the LLM context.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.05270">APOLLO: SGD-like Memory, AdamW-level Performance</a>: Large language models (LLMs) are notoriously memory-intensive during training, particularly with the popular AdamW optimizer. This memory burden necessitates using more or higher-end GPUs or reducing ...</li><li><a href="https://zhuhanqing.github.io/APOLLO/">APOLLO: SGD-like Memory, AdamW-level Performance</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1317573115500564642)** (1 messages): 

> `Progressive Tokenization, Zero-tree Ordering, DWT Coefficients, VAE Embedding` 


- **Progressive Tokenization Explained**: The discussion focused on **progressive tokenization** utilizing a **zero-tree ordering** of **DWT coefficients** drawn from a **VAE embedding**.
   - An attached video demonstrates the technique in action, showcasing the intricacies of the process.
- **Analysis of Wavelet Coefficients**: Members examined how **level 5 wavelet** transformations impact tokenization effectiveness within the context of the discussed methods.
   - The analysis included practical applications and implications for future model enhancements, featuring [the attached video](https://cdn.discordapp.com/attachments/823813160075132991/1317573114854637680/level_5_wavelet_db5_clip_value_2.0_patch_size_1.mp4?ex=6761d015&is=67607e95&hm=8a77936be85424a1ccff2f733b8e69a5ce554860b92709f386eb634bd6d148d5&).


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1317547822073253899)** (1 messages): 

> `Byte Latent Transformer Patches, Large Concept Models, NLP advancements` 


- **Byte Latent Transformer Patches outperform tokens**: The publication titled [Byte Latent Transformer Patches: Scale Better than Tokens](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/) discusses a new approach in NLP that shows how byte latent transformer patches manage better scalability compared to traditional tokens.
   - This advancement opens up discussions on enhancing language modeling effectiveness and efficiency in various applications.
- **Exploring Large Concept Models in NLP**: The LCM team, including members such as **Loic Barrault** and **Holger Schwenk**, is working on understanding language modeling through a framework based on sentence representation space.
   - Their research aims to provide deeper insights into how language concepts can be structured and utilized effectively in NLP models.



**Link mentioned**: <a href="https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/">no title found</a>: no description found

  

---


### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1318308461712379994)** (1 messages): 

> `Retrieval Augmented Generation, Event Preparations, SQLite-Vec and LlamaFile, Python Development` 


- **Final December Event on RAG Application**: Tomorrow's event focuses on creating an **ultra-low dependency Retrieval Augmented Generation (RAG)** application using **sqlite-vec** and **llamafile**, with **bare-bones Python** and without any additional dependencies or installations.
   - The event will be led by **Alex Garcia**, providing attendees with a straightforward approach to building RAG applications.
- **Preparing for the Holiday Break**: This event marks the **final gathering** for December before taking a break for the holidays, emphasizing the importance of participation before the year-end.
   - Participants are encouraged to **join the session** as a prelude to the holiday season and gain insights into RAG development.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/)** (1 messages): 

huanzhimao: Update: They are [here](https://github.com/HuanzhiMao/BFCL-Result).
  

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

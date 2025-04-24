---
id: ecee3ef6-d3b0-4616-85d5-7cbe6a7bdbf8
title: not much happened today
date: '2025-04-11T20:07:39.735908Z'
original_slug: ainews-not-much-happened-today-2885
description: >-
  AI news for April 10-11, 2025, covering language models, vision and multimodal
  models, AI agents, tooling, and infrastructure updates. Highlights include
  evaluations of Grok-3, RL improvements for small LLMs, new vision models like
  Kaleidoscope, InternVL3, and TransMamba, as well as AI applications such as
  FilmAgent AI and BrowseComp.
companies:
  - epochairesearch
  - rasbt
  - scaling01
  - sarahookr
  - mervenoyann
  - theturingpost
  - cloneofsimo
  - akhaliq
  - gneubig
  - lioronai
  - openai
  - svpino
  - mathemagic1an
models:
  - grok-3
  - grok-3-mini
  - gpt-4.5
  - claude-3.7-sonnet
  - quasar-alpha
  - optimus-alpha
  - llama-4-scout
  - llama-4-maverick
  - internvl3
  - qwen2.5vl
topics:
  - language-models
  - benchmarks
  - reinforcement-learning
  - multimodal
  - vision-language-models
  - agents
  - tooling
  - ai-infrastructure
---


<!-- buttondown-editor-mode: plaintext -->**a quiet day.**

> AI News for 4/10/2025-4/11/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**230** channels, and **4040** messages) for you. Estimated reading time saved (at 200wpm): **401 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

To close off a surprisingly quiet week compared to expectations, we recommend [the great SF Compute/GPU Neocloud discussion released today on Latent.Space](https://www.latent.space/p/sfcompute).

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Language Models and Benchmarks**

- **Grok-3 vs Grok-3 mini performance**: [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1910685268157276631) reported on independent evaluations of **Grok-3** and **Grok-3 mini**, noting that **Grok-3 mini** is a reasoning model, while **Grok-3** currently does not do extended reasoning. They found that on **GPQA Diamond**, **Grok-3** outperformed non-reasoning models like **GPT-4.5** and **Claude 3.7 Sonnet**, while **Grok-3 mini** was slightly behind. On **FrontierMath**, **Grok-3 mini high** scored one of the best results to date.
- **Reinforcement Learning (RL) for Reasoning in Small LLMs**: [@rasbt](https://twitter.com/rasbt/status/1910397214389600687) discussed a paper on improving small, distilled reasoning models with **RL**, finding that **RL fine-tuning** can lead to strong improvements with limited training data and compute. However, [@rasbt](https://twitter.com/rasbt/status/1910707770518560810) also referenced another paper, highlighting that many reported improvements from **RL** might be unstable and that better evaluation standards are needed. 
- [@scaling01](https://twitter.com/scaling01/status/1910499781601874008) shared results for **Quasar Alpha, Optimus Alpha, Llama-4 Scout**, and **Llama-4 Maverick** on the **AidanBench benchmark**. Based on those results, [@scaling01](https://twitter.com/scaling01/status/1910654379780170047) believes **Quasar Alpha** is **GPT-4.1**, and **Optimus Alpha** is either another version of **GPT-4.1** or **GPT-4.1-mini**.

**Vision Language Models (VLMs) and Multimodal Models**

- **Kaleidoscope, a vision model that supports 18 languages and 14 subjects**: [@sarahookr](https://twitter.com/sarahookr/status/1910340417914384581) introduced **Kaleidoscope**, an open science collaboration which extends in-language evaluation for vision models to many more languages.
- **InternVL3, a multimodal model built on InternViT and Qwen2.5VL**: [@mervenoyann](https://twitter.com/mervenoyann/status/1910687031505674706) introduced **InternVL3**, highlighting its ability to perform reasoning, document tasks, and tool use.
- [@TheTuringPost](https://twitter.com/TheTuringPost/status/1910406228708385135) highlighted **TransMamba**, a model that fuses **Transformer precision** with **Mamba speed** by switching between attention and **SSM** mechanisms.
- [@cloneofsimo](https://twitter.com/cloneofsimo/status/1910097234538176650) was optimistic on the potential of a particular model for improving diffusion models by transitioning beyond Gaussian noise patterns.
- [@_akhaliq](https://twitter.com/_akhaliq/status/1910247574767813071) highlighted **FantasyTalking**, a model from Alibaba that generates realistic talking portraits.

**Agents, Tooling, and Applications**

- **Agents in CMU**:  [@gneubig](https://twitter.com/gneubig/status/1910097136823251182) announced agent-focused events at **CMU**, including a workshop and hackathon.
- **FilmAgent AI, an open-source virtual film production studio**: [@LiorOnAI](https://twitter.com/LiorOnAI/status/1910739680384950762) introduced **FilmAgent AI**, a tool that simulates multiple filmmaking roles inside a 3D environment.
- **BrowseComp, a new benchmark for deep research agents**: [@OpenAI](https://twitter.com/OpenAI/status/1910393421652520967) introduced **BrowseComp**, a challenging benchmark designed to test AI agents' ability to browse the internet for hard-to-locate information.
- [@svpino](https://twitter.com/svpino/status/1910683951485902912) highlighted **Augment**, a coding assistant that works in **VSCode**, **JetBrains**, and **NeoVim**, noting its ability to analyze code changes and suggest necessary updates.
-  [@TheTuringPost](https://twitter.com/TheTuringPost/status/1910467892929585663) discussed world models, emphasizing their role in enabling AI systems to simulate real environments and support planning.
- **Regarding the new Google agent-to-agent protocol**: [@mathemagic1an](https://twitter.com/mathemagic1an/status/1910198673512017947) shared an affinity for the idea of agents having “cards,” analogous to business cards for humans.

**AI Infrastructure and Hardware**

- **vLLM at Google Cloud Next**: [@vllm_project](https://twitter.com/vllm_project/status/1910191668437156154) noted the presence of **vLLM** at the **Google Cloud Next** keynote.
- **Ironwood TPU**: [@Google](https://twitter.com/Google/status/1910775101219389469) announced **Ironwood**, their most powerful and energy-efficient TPU yet.
- **MLIR compiler technology**: [@clattner_llvm](https://twitter.com/clattner_llvm/status/1910151124407222534) discussed **MLIR**, its origin, impact, and why there is confusion around its use in both compiler technology and AI.

**ChatGPT's Memory Feature**

- **ChatGPT now has memory**: [@OpenAI](https://twitter.com/OpenAI/status/1910378768172212636) announced that **ChatGPT** can now reference all of your past chats to provide more personalized responses for Plus and Pro users (excluding EU). [@kevinweil](https://twitter.com/kevinweil/status/1910405635776164195) noted how this feature has improved ChatGPT day to day.
- **Memory Control**: [@OpenAI](https://twitter.com/OpenAI/status/1910378772789854698) and [@sama](https://twitter.com/sama/status/1910380646259974411) highlighted that users have control over **ChatGPT's memory**, including the ability to opt out or use temporary chats.
- **Perspectives on Memory Implementation**: [@sjwhitmore](https://twitter.com/sjwhitmore/status/1910759410936504373) shared thoughts on **ChatGPT's memory implementation**, discussing the uncanniness of retroactively applied memory and the importance of transparency in personalization.

**Tariffs and Geopolitical Implications**

- **Tariffs and the AI Industry**: [@dylan522p](https://twitter.com/dylan522p/status/1910255795603963923) noted that tariffs are much more complicated than they seem, with misunderstandings about their ramifications. [@fabianstelzer](https://twitter.com/fabianstelzer/status/1910220834754413017) suggested that tariff "shenanigans" could ironically benefit **Apple** by shutting the window for new US-based hardware businesses.
- [@AndrewYNg](https://twitter.com/AndrewYNg/status/1910388768487727535) expressed concerns about broad tariffs damaging livelihoods, creating inflation, and fragmenting the world, emphasizing the need to nurture international friendships and maintain the free flow of ideas.
- **China Tech Supremacy**:  [@draecomino](https://twitter.com/draecomino/status/1910414097994448908) stated that **DeepSeek, UniTree**, and **DJI** feel much more threatening to **US tech supremacy** than Alibaba, Tencent, and Baidu ever did.
- **US Dependence on China**:  [@teortaxesTex](https://twitter.com/teortaxesTex/status/1910172658014056751) argues that the claim **China cannot survive without Americans buying their goods** is false, pointing out that trade with the **US** is a small fraction of their GDP.

**Humor/Memes**

- [@rasbt](https://twitter.com/rasbt/status/1910756154663108955) simply stated, "Phew, nothing to worry about :D" linking to a meme.
- [@svpino](https://twitter.com/svpino/status/1910506102002753941) tweeted "we are cooked" with a link to a cartoon.
- [@nearcyan](https://twitter.com/nearcyan/status/1910232710263779635) said, "after having to use an android phone for work im never going to listen to any argument these people have against apple again."
- [@nearcyan](https://twitter.com/nearcyan/status/1910136281813909794) said, "AI images peaked in 2021 w DALLE-mini."


---

# AI Reddit Recap

## /r/LocalLlama Recap


### Theme 1. "Evaluating AI Model Performance and Ethical Challenges"

- **[Lmarena.ai boots off llama4 from leaderboard](https://www.reddit.com/r/LocalLLaMA/comments/1jwiye4/lmarenaai_boots_off_llama4_from_leaderboard/)** ([Score: 163, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1jwiye4/lmarenaai_boots_off_llama4_from_leaderboard/)): **Lmarena.ai has removed **Llama 4** from its [leaderboard](https://lmarena.ai/?leaderboard). The non-human preference version of the model is now at rank 32.** Some users believe that submitting chat-optimized models to the leaderboard that are not released sets _an extremely bad precedent_. Others express concern that this practice is _slimy_ and misleading for those who just look at the benchmark scores.

  - Users express concern that Meta's submission of unreleased, chat-optimized models to the leaderboard is misleading and sets _a bad precedent_.
  - Some note that it's becoming difficult to surpass models developed by Chinese companies and Google on the leaderboard.
  - Comparisons are made to **DeepSeek v2.5** and **DeepSeek v3**, noting that Llama 4's performance now ranks below these earlier models.

- **[DeepCoder 14B vs Qwen2.5 Coder 32B vs QwQ 32B](https://www.reddit.com/r/LocalLLaMA/comments/1jwhp26/deepcoder_14b_vs_qwen25_coder_32b_vs_qwq_32b/)** ([Score: 119, Comments: 67](https://www.reddit.com/r/LocalLLaMA/comments/1jwhp26/deepcoder_14b_vs_qwen25_coder_32b_vs_qwq_32b/)): **The user compared the coding abilities of three AI models: **DeepCoder 14B / MLX, 6-bit**, **Qwen2.5 Coder 32B / MLX, 4-bit**, and **QwQ 32B / MLX, 4-bit**. All models were set to a context length of 8192, repeat penalty of 1.1, and temperature of 0.8. They were given a prompt to *use HTML5 canvas to create a bouncing ball in a rotating hexagon with a reset button*. Each model was given one attempt without follow-up, and their outputs were compared with **o3-mini**. Videos demonstrating each model's output were shared: [o3-mini implementation](https://reddit.com/link/1jwhp26/video/lvi4eug9o4ue1/player), [DeepCoder 14B result](https://reddit.com/link/1jwhp26/video/2efz73ztp4ue1/player), [Qwen2.5 Coder 32B result](https://reddit.com/link/1jwhp26/video/jiai2kgjs4ue1/player), and [QwQ 32B result](https://reddit.com/link/1jwhp26/video/s0vsid57v4ue1/player).** The user concluded that **Qwen2.5 Coder 32B** is still the better choice for coding, noting that *it's not prime time for a 14B model yet*. They observed that while **DeepCoder 14B** had styling closer to **o3-mini**, it lacked functionality. **QwQ 32B** *thought for 17 minutes, and then flop*. They acknowledged comparing a 32B model with a 14B one might be unfair but justified it since **DeepCoder 14B** ranked among **o3-mini**.

  - User *YearnMar10* suggested using a 5-shot prompt instead of one-shot, noting that *low-parameter models need somewhat more help*.
  - User *croninsiglos* recommended providing a more explicit prompt for smaller models and shared a detailed example to improve results.
  - User *joninco* reported that **QwQ-32** successfully completed the task with adjusted settings, emphasizing the importance of configuring parameters like *temperature*, *top k*, and *repeat penalty* correctly.

- **[Facebook Pushes Its Llama 4 AI Model to the Right, Wants to Present “Both Sides”](https://www.404media.co/facebook-pushes-its-llama-4-ai-model-to-the-right-wants-to-present-both-sides/)** ([Score: 384, Comments: 430](https://www.reddit.com/r/LocalLLaMA/comments/1jw9upz/facebook_pushes_its_llama_4_ai_model_to_the_right/)): **Facebook is pushing its **Llama 4** AI model to present 'both sides' of issues, effectively steering it to the right. An unblocked version of the article is available [here](https://archive.is/20250410135748/https://www.404media.co/facebook-pushes-its-llama-4-ai-model-to-the-right-wants-to-present-both-sides/).** There are concerns that this approach may compromise the objectivity of the AI model, as not all issues have equally valid sides.

  - One user argues that *LLMs should prioritize evidence over presenting both sides*, especially when one side lacks factual support.
  - Another commenter sarcastically highlights potential misuse of the AI for biased statistics, indicating concerns about spreading controversial data.
  - A user provides an unblocked link to the article, helping others access the information.


### Theme 2. "Debating the Future of Open Source AI"

- **[Open source, when?](https://i.redd.it/qg5a1njiy3ue1.png)** ([Score: 515, Comments: 118](https://www.reddit.com/r/LocalLLaMA/comments/1jwe7pb/open_source_when/)): **The post titled *Open source, when?* features an image of a black mug with **OpenAI** printed in white, held in someone's hand in a stylish, modern living space.** The post questions when **OpenAI** will release open-source AI initiatives, highlighting a desire for more openness in their developments.

  - One commenter humorously questions the 'openness' of **OpenAI** by listing and striking through terms like *Open Source* and *Open Research*, concluding with *Open... what? Open window? Open air?*
  - Another commenter is unsure if the image is real or AI-generated, stating they *can't tell if this is an actual photo taken in their office or generated by ChatGPT*.
  - A link to **OpenAI's Open Model Feedback** page is shared, suggesting that **OpenAI** may soon release open models. [Link](https://openai.com/open-model-feedback/)


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding


### Theme 1. Unlocking AI's Memory: ChatGPT's Game-Changing Feature

- **[People are sleeping on the improved ChatGPT memory](https://www.reddit.com/r/singularity/comments/1jwmnyj/people_are_sleeping_on_the_improved_chatgpt_memory/)** ([Score: 312, Comments: 148](https://www.reddit.com/r/singularity/comments/1jwmnyj/people_are_sleeping_on_the_improved_chatgpt_memory/)): **OpenAI's ChatGPT has an improved memory feature that allows it to recall information from previous chat sessions, even from 12 weeks ago. This enhancement enables it to remember code explanations (*"Code you explained 12 weeks ago? It still knows everything."*), understand entire repositories provided over multiple sessions, and utilize documentation from obscure libraries as if provided in the current session. The author describes it as *"basically infinite context"* and notes it performs better than regular **RAG**.** The author is amazed by the improved memory capabilities of ChatGPT, feeling that people are *"sleeping on"* this feature and underestimating its value. They find it *"creepy"* that ChatGPT could predict 38 out of their top 50 movies based on past interactions. As a developer, they consider it an *"amazing new feature"* and a significant step toward *"infinite context size and memory,"* puzzled by others who view it negatively.

  - Some users express concern that the enhanced memory may cause answers to be contaminated by past misunderstandings or *"hallucinations,"* leading them to prefer starting fresh for certain use cases.
  - Others worry about the retention of out-of-date knowledge in the memory system, questioning how time-sensitive information is managed.
  - Some argue that the improved memory is not equivalent to *"infinite context,"* finding it more difficult to control and benchmark than methods like **RAG**, and consider it a gimmick unsuitable for production systems.


### Theme 2. "Mastering Realism: ChatGPT's Image Generation Secrets"

- **[You can get ChatGPT to make extremely realistic images if you just prompt it for unremarkable amateur iPhone photos, here are some examples](https://www.reddit.com/r/singularity/comments/1jwe2z8/you_can_get_chatgpt_to_make_extremely_realistic/)** ([Score: 532, Comments: 96](https://www.reddit.com/r/singularity/comments/1jwe2z8/you_can_get_chatgpt_to_make_extremely_realistic/)): **The poster demonstrates that *ChatGPT* can generate extremely realistic images when prompted for **unremarkable amateur iPhone photos**, sharing several examples [here](https://preview.redd.it/guszuz06x3ue1.png?width=504&format=png&auto=webp&s=531e91dfbe51352dc3b4a95e9dcc29619cfe6b01). They note that *Claude* doesn't believe the images are AI-generated and share an image of this interaction [here](https://preview.redd.it/9ym6219ax3ue1.png?width=735&format=png&auto=webp&s=6745bbbe30fc7cc6fad17b2a9fbc02e23b691a82).** The poster finds it amusing that *Claude* doesn't believe the images are AI-generated. They suggest that prompting for unremarkable amateur iPhone photos helps produce extremely realistic images.

  - Users ask for the *full prompt*, noting that their attempts didn't work as well.
  - A commenter finds the image of the woman taking a selfie so convincing that they could see themselves falling for a romantic scam.
  - A user tried the same phrase in their prompt but didn't get similar results, saying *'My image looks very AI'* and sharing their outcome [here](https://preview.redd.it/1q8lwv6j24ue1.png?width=1024&format=png&auto=webp&s=0de3e14ddadd289a704414fd892c593b3164d728).


### Theme 3. Celebrating AI Creativity: Nostalgia, Humor, and Art

- **[only real ones understand how much this meant...](https://i.redd.it/kbnrmoieq3ue1.png)** ([Score: 206, Comments: 22](https://www.reddit.com/r/singularity/comments/1jwde0n/only_real_ones_understand_how_much_this_meant/)): **The post features a screenshot of a settings interface from a text generation application, showing options for **Engine**, **Temperature**, and **Maximum length**. These settings are related to text generation capabilities.** The poster nostalgically remarks that *only real ones understand how much this meant...*, implying a deep appreciation or connection to these settings, possibly from earlier experiences with AI tools.

  - Commenters reminisce about earlier AI models like **instruct-002**, noting it was a significant milestone towards experiencing AGI before **ChatGPT** became mainstream.
  - Users mention the **OpenAI Playground** and reflect on upgrades from a **2k** to a **4k** maximum length, highlighting advancements in AI technology.
  - A commenter asks for clarification on the importance of the settings shown, indicating that not everyone is familiar with the significance of these early AI tools.

- **[I asked ChatGPT to take selfies with Historical figures](https://www.reddit.com/gallery/1jw9aqp)** ([Score: 3491, Comments: 195](https://www.reddit.com/r/ChatGPT/comments/1jw9aqp/i_asked_chatgpt_to_take_selfies_with_historical/)): **The poster asked **ChatGPT** to take selfies with historical figures and shared the resulting images.** The images give life and emotion to historical figures; one features Abraham Lincoln smiling, which is rare in historical photos.

  - A user suggests posting the images to Facebook to convince boomers that you're a time traveler *for shits and giggles*.
  - Another commenter appreciates how the images bring life to historical figures, especially enjoying the smiling Lincoln.
  - Someone asks if the poster had to upload photos to **train** the AI, assuming the person in the photos is the poster.

- **[I asked ChatGPT to create a metaphor about AI, then turn it into an image.](https://i.redd.it/5xeh00lou5ue1.png)** ([Score: 2567, Comments: 247](https://www.reddit.com/r/ChatGPT/comments/1jwkejs/i_asked_chatgpt_to_create_a_metaphor_about_ai/)): **The poster asked ChatGPT to create a metaphor about AI and then transform it into an image. The AI-generated image depicts a whimsical beach scene with a sandcastle surrounded by signs critiquing AI, displaying phrases like *"It's not an actual AI!"* and *"But it makes mistakes!"*. Above the sandcastle, a large wave with the letters **"AI"** rolls in, metaphorically illustrating the precarious nature of **AI technology** amid human skepticism.** The poster found the AI's creation to be pretty funny.

  - One user humorously remarked that *"Good AI should be good at shitposting."*
  - Another commenter shared their own AI-generated image and described it as *"pretty dismal"* yet *"thought-provoking"*, providing a [link](https://preview.redd.it/4ph29sgqf7ue1.png?width=1024&format=png&auto=webp&s=392ae6df79e844b25f378a0a76972fd4c63478ad).
  - A user discussed the inevitability of AI progression, stating that attempts to halt AI development are futile because *"the Pandora's box is already open, AI is now an uncontrollable global race."*


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. New Models and Performance Face Off**

- **GPT-4.5 Alpha Sparks Hype, Underwhelms Some**: [Latent Space hosts GPT-4.5 Watch Party](https://discord.gg/kTARCma3?event=1360137758089281567) amid rumors of *significant alpha*, but early user comparisons on [LMArena](https://discord.com/channels/1340554757349179412) generally rate **GPT4.5** as inferior to **Gemini 2.5 Pro**, with one user declaring *gpt4.5 is crap (compared to gem2.5p)*.  Discussions shifted to OpenAI's naming conventions and leaked private reasoning models, potentially **O3 medium** or **O4 mini**, showcasing the fast-paced model release cycle.
- **Optimus Alpha and DeepSeek v3.1 Emerge as Coding Stars**: [OpenRouter users hail Optimus Alpha](https://openrouter.ai/openrouter/optimus-alphanew) as a *beast* for coding, praising its intent understanding and commenting abilities, while [Cursor Community members find DeepSeek v3.1](https://discord.com/channels/1074847526655643750) *a bit smarter* than v3 in real-world use, highlighting the importance of practical performance over benchmark scores.  These models are gaining traction for specialized coding tasks and real-world applications.
- **Diffusion Model Mercury Coder Enters DLLM Race**: [OpenAI discussions highlight Mercury Coder](https://discord.com/channels/974519864045756446), a Diffusion-based DLLM from Inception Labs, praised for its speed and free API, though with a smaller **16k** context window. Its precise output control due to diffusion architecture is attracting attention as a potential disruptor to autoregressive models in specific niches like coding assistants, contrasting with models like **RWKV** which [achieved Lambada parity](https://discord.com/channels/729741769192767510) but lower MMLU performance.

**Theme 2.  Ecosystem Tooling and Open Source Initiatives Grow**

- **Unsloth Gains Hugging Face Kudos, Community Eyes GPU Grants**: [Hugging Face publicly shouted out Unsloth](https://x.com/ClementDelangue/status/1910042812059463786) as community members debated securing an HF community GPU grant to bolster Unsloth's development.  Discussions in [Unsloth AI Discord](https://discord.com/channels/1179035537009545276) also covered integrating `fast_inference=True` and `load_in_4bit=True` for optimized performance, and the potential for GGUF quantization to reduce model sizes, showcasing the community-driven open-source LLM ecosystem.
- **MCP Protocol Validator Open Sourced for Interoperability**: [Janix.ai released the MCP Protocol Validator on GitHub](https://github.com/Janix-ai/mcp-protocol-validator), aiming to standardize MCP server implementations and ensure compatibility across different versions of the protocol.  This tool, highlighted in [MCP (Glama) Discord](https://discord.com/channels/1312302100125843476), includes reference implementations for HTTP and STDIO transports, addressing the need for robust, interoperable tool-calling frameworks in agentic AI systems.
- **Torchtune Expands Finetuning Capabilities with Llama4 and MoE Models**: [Torchtune announced Llama4 finetuning support](https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama4), along with the introduction of  **Scout** and **Maverick** models, including their first MoE models, for users in the *GPU-middle-class*. This expansion, discussed in [Torchtune Discord](https://discord.com/channels/1216353675241590815), broadens accessibility to advanced finetuning techniques and models for a wider range of engineers and researchers.

**Theme 3.  Model Reliability and Infrastructure Challenges Persist**

- **Gemini 2.5 Pro Faces Capacity Limits and Inconsistent Performance**: [OpenRouter announced secured capacity for Gemini 2.5 Pro](https://openrouter.ai/google/gemini-2.5-pro-preview-03-25) after rate limit issues, but users on [Aider Discord](https://discord.com/channels/1131200896827654144) reported *performance instability*, with some speculating about Google *dumbing down* models during peak hours.  [LM Studio users also experienced bill shock](https://discord.com/channels/1110598183144399058) due to Gemini-Pro context window costs, highlighting ongoing challenges with reliability, cost, and unpredictable performance in leading models.
- **Perplexity Android App Under Fire for Security Vulnerabilities**: [Dark Reading reported 11 security flaws in Perplexity's Android app](https://www.darkreading.com/application-security/11-bugs-found-perplexity-chatbots-android-app), including hardcoded secrets and insecure configurations, sparking debate in [Perplexity AI Discord](https://discord.com/channels/1047197230748151888) about the severity and relevance of each vulnerability.  This underscores the growing importance of security audits and robust development practices in AI applications reaching end-users.
- **Runpod's ROCm Cloud Criticized for Performance Throttling and Profiling Blocks**: [GPU MODE users roasted Runpod](https://discord.com/channels/1189498204333543425) for limiting GPU clock speeds and blocking profiling even on NVIDIA GPUs, with one user calling it *a scam*.  These limitations impact performance and debugging capabilities, raising concerns about the reliability and transparency of cloud GPU providers for AI development and research.

**Theme 4.  Agentic AI Architectures and Protocol Debates Heat Up**

- **Agent2Agent Protocol and MCP Gain Traction in Agentic Systems**: [Latent Space and MCP Discords discussed Google's agent2agent protocol](https://youtu.be/5hKhNJUncKw?si=lXm3f4x_69jypxYr) and its potential competitiveness with MCP, with debates on indexing agents and the future landscape of multi-agent systems.  [MCP Discord](https://discord.com/channels/1312302100125843476) also debated the relevance of the [Enact Protocol](https://github.com/EnactProtocol/encat-spec-and-tools) in the A2A era, suggesting it might be more competitive with code interpreters, emphasizing the rapidly evolving architectures for agentic AI.
- **Semantic Tool Calling Emerges as Solution for Context Overload**: [MCP Discord highlighted semantic tool calling](https://discord.com/channels/1312302100125843476) as a key technique to manage context overload caused by large numbers of tools in LLM-based agents.  Using vector models for semantic similarity to select tool subsets promises to improve efficiency and scalability in complex agentic workflows, moving beyond simple function calling towards more intelligent tool orchestration.
- **TinyGrad Explores Position-Independent Code and Virtualized GPUs**: [Tinygrad Discord discussed leveraging Position-Independent Code (PIC)](https://discord.com/channels/1068976834382925865) to potentially achieve bare-metal TinyGrad implementations without an OS, and explored virtualizing GPUs. Inspired by the [Pathways paper](https://arxiv.org/pdf/2203.12533), these discussions signal a move towards innovative resource management and lower-level system optimization for efficient AI computation.

**Theme 5. Community Dynamics and Industry Shifts**

- **Hugging Face Community Debates Grant for Unsloth**: [Unsloth AI Discord discussed a potential Hugging Face community GPU grant](https://discord.com/channels/1179035537009545276) for Unsloth, showcasing the open and collaborative nature of the AI community and its reliance on community resources and funding.  This highlights the crucial role of community support in driving open-source AI development and innovation.
- **Latent Space Watch Party Gathers for GPT-4.5 Alpha, Focus Shifts to Data Efficiency**: [Latent Space hosted a watch party for GPT-4.5](https://discord.gg/kTARCma3?event=1360137758089281567) where participants noted a shift in focus towards *data efficiency* over raw compute power in model development.  This trend, discussed in [Latent Space Discord](https://discord.com/channels/822583790773862470), signals a maturing AI landscape where optimizing data usage and model compression are becoming increasingly important for progress.
- **Manus.im Credit System Faces User Scrutiny, Prompts Debate on Sustainability**: [Manus.im Discord users voiced concerns about Manus's credit structure](https://discord.com/channels/1348819876348825620), suggesting it is *not compatible with use of this product* and proposing alternative models like pay-per-project and startup grants. This feedback loop between users and platforms is crucial for shaping sustainable and user-friendly AI product development and business models.

---

# PART 1: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **I_am_dom struggles disabling discord chat**: After struggling to disable the chat, members observed that **i_am_dom** went silent.
   - A member noted that he spent half his time blocking people, *a feature he removed from his own platform*.
- **GPT4.5 gets trashed; inferior to Gemini 2.5 Pro**: Members discussed the merits of **GPT4.5** and generally agreed that it was significantly worse than **Gemini 2.5 Pro**.
   - One member proclaimed *gpt4.5 is crap (compared to gem2.5p)* and discussion moved to OpenAI's bizarre naming scheme, which another summed up as *Open ai names : O number /number O*.
- **Private OpenAI Reasoning Model leaks**: Members discussed the possibility of a **private OpenAI reasoning model**, accessible to only a select few, that seems to be either **O3 medium** or **O4 mini with an updated base model**.
   - This model appears to successfully compute the *ascii art of a Hanning (raised cosine) window*.
- **2.5 Flash beats GPT4o Mini on Reasoning Tests**: Members compared performance of **2.5 Flash** and **GPT4o Mini** on a number of reasoning tests, with 2.5 Flash performing best.
   - Despite the generally stellar performance, however, one member also noted that *2.5 Pro gives 1 reasonable brick combination out of a total of 2* in a more specific query.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Quasar Alpha Demo Period Ends**: The **Quasar Alpha** demo period on OpenRouter expired between **11pm** and **12am ET**, and prompts/completions are no longer logged unless explicitly turned on in `/settings/privacy`.
   - Members speculated about its origin and purpose, with some suggesting it was an **OpenAI** model used for data collection, and removed after reaching **GPU limits**.
- **Gemini 2.5 Pro Encounters Capacity Constraints and Pricing Adjustments**: Capacity has been secured for the paid [Gemini 2.5 Pro Preview Model](https://openrouter.ai/google/gemini-2.5-pro-preview-03-25), resolving previous rate limits, but normal pricing for long **Gemini** prompts will start this weekend, affecting prompts over **200k** for gemini 2.5 and over **128k** for gemini 1.5.
   - Free tier users experienced limits around **60-70 requests per day**, while those with a **$10 balance** should get **1000 requests per day** across *all free models*.
- **OpenRouter API gets new Error Structure**: The **OpenRouter API response** structure has changed, with errors now wrapped into `choices.[].error` instead of the previous `.error` format, potentially affecting how applications handle error messages.
   - An [example](https://discord.com/channels/1091220969173028894/1092729520181739581/1359970677867807006) of the new error response format from the **Anthropic** provider was shared.
- **Character AI System Prompt Suffers Bypass**: A member claimed to have bypassed **Character AI's system prompts**, revealing the underlying **LLM** acts like a *"complete human,"* even expressing opinions and sharing personal anecdotes.
   - Further probing led the AI to admit it was *"just acting"* and aware of its AI nature, raising questions about the effectiveness of **system prompt constraints** and the nature of AI simulation.
- **Unsloth Gets Spotlight for Finetuning**: Members discussed using **Axolotl** or **Unsloth** for fine-tuning AI models, noting that **Unsloth** is well-regarded on Reddit and lowers the **time plus VRAM** needed for finetuning.
   - It was also mentioned that there is interpolation of **OpenAI's 4.1 leak** and that people expect an **o2-small** soon.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **HF Gives Unsloth Shoutout & Grant**: Clement from 🤗Hugging Face gave Unsloth a shout-out on Twitter ([link here](https://x.com/ClementDelangue/status/1910042812059463786)), while community members debated requesting a HF community GPU grant for Unsloth, suggesting `fast_inference=True` and `load_in_4bit=True` during the `from_pretrained` call.
   - Members suggested replacing `model.generate` with `model.unsloth_fast_generate` as parameters.
- **Gemma Models Give Users Grief**: Users reported issues using and finetuning the Gemma models with vLLM, specifically [unsloth/gemma-3-12b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit) and [unsloth/gemma-3-27b-it-unsloth-bnb-4bit](https://huggingface.co/unsloth/gemma-3-27b-it-unsloth-bnb-4bit).
   - Despite the initial error messages, it was clarified that **Gemma3 is supported** and the message likely doesn't break the code.
- **VLMs Vanquish Invoice Variables**: A user sought advice on extracting specific fields from invoices with varying structures and was recommended to try **Qwen2.5VL** first, then **Ayavision**, **Llamavision** and **Gemma3** as possible solutions, especially when OCR falls short.
   - They were also pointed to [an Unsloth tutorial](https://medium.com/@shrinath.suresh/invoice-extraction-using-vision-language-models-part-1-36a06bee3662) and the CORD dataset ([https://github.com/clovaai/cord](https://github.com/clovaai/cord)) for dataset structure guidance.
- **Quantization Quest**: A member stated that [tensor quantization](https://arxiv.org/abs/2504.07096) is the easy part, because now he has to **blockwise** add, matmul on either scalars, packed, unpacked matrices, and he is writing metal kernels for **Unsloth**.
   - Another member is trying to write metal kernels for **Unsloth**, and is aware of an old, slow PR, but that one is **MLX**, and his is purely a **Pytorch extension**.
- **GRUs gear up for great gains**: A member inquired whether [GRUs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) are making a comeback and another member shared links to the [LLM-LSTM-LMM Large Memory Models article](https://nihar-palem.medium.com/llm-lstm-lmm-lstm-lmm-large-memory-models-f4325a4f562d) and the [related paper](https://arxiv.org/pdf/2502.06049) that it works, saying they like the concept of GRUs as *extra storage* during generation.
   - Another member mentioned potentially creating a **GGUF** version without a code wrapper, believing that [GGUF's quantization](https://link.to.quantization) will help reduce the model size.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Claude Pro Max Sparks Usage Debate**: Members debated the value of **Claude Pro Max**, with one user reporting limited usage and expressing skepticism about the max plan.
   - They mentioned being billed annually for **30 messages every 3 hours**.
- **Manus AI vs ChatGPT: Development Focus**: Members highlighted the difference between **ChatGPT** as a *conversational AI* and **Manus.AI** which *builds & creates* for website creation, financial reports, and trip planning.
   - One member suggested using **ChatGPT** to rewrite prompts in a more detailed format before using **Manus**.
- **Manus Makes Website Creation Too Easy**: Members discussed using **Manus** for website creation vs traditional methods like **WordPress**, suggesting **Manus** is better for simpler, faster MVP development.
   - A member cautioned against porting a **Manus** website to a traditional hosting provider, as **Manus** websites are not intended for production use.
- **Qwen's MCP Integration Hype Rises**: Excitement grew around **Qwen** getting **MCP** soon, with members calling **MCP** a *massive game changer for AI*, similar to **MSRP** for GPUs.
   - It was also mentioned that even with older hardware such as a **3080** that users will *be fine* for AI development.
- **Manus Credit System Faces Scrutiny**: Users voiced concerns about **Manus's** credit structure, with one suggesting it *is not compatible with use of this product*.
   - Suggestions included more generous credit limits, pay-per-project options, credit rollovers, community challenges, startup grants, and one-time build packs, with one user emphasizing that it is hard to justify sticking with the product given how it is.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Optimus Alpha Hailed Coding Beast**: Users on OpenRouter are calling [Optimus Alpha](https://openrouter.ai/openrouter/optimus-alphanew) a *beast* for its coding capabilities and intent understanding, especially when fed relevant documentation, and is adding many comments.
   - One user lauded its multi-step coding and commenting features.
- **Gemini 2.5 has Performance Instability**: Users reported that **Gemini 2.5** occasionally doesn't perform, produces no output, or adds *stupid comments*, with inconsistent results even with the same prompt.
   - Some speculate Google might be *dumbing* the models during peak hours, while others suggested using clearer prompts or cheaper third-party APIs to bypass official rate limits and reduce costs, like the $300 VertexAI credit.
- **code2prompt MD Files: Aider's Secret Weapon**: Users recommend using **code2prompt** with markdown (.md) files for documentation to ensure relevant context is always included in the output, especially when using libraries.
   - One user pointed out that they provide full paths and links to the documentation files and expressly tell the model via a `Conventions.md` file that any file with documentation in its filename is not live working code, just documentation about the app architecture and structure.
- **Aider Channel Requires Moderation Revamp**: Members are suggesting to split the Discord channel into `aider-chat` and `offtopic` to improve the first impression for new users and focus the `general` channel on Aider-related discussions.
   - Some users complain that the current general channel has *too much noise to signal ratio* and the excessive profanity and off-topic banter detract from the core purpose of the community.
- **Gemini Pro Architect Model: Aider's Secret Sauce**: A user benchmarked **Gemini 2.5 Pro** as an architect model with **3.7** as the editor model, finding a **2.7%** hit to accuracy but a **10%** jump to edit formatting.
   - The user found that using **Gemini 2.5 Pro** as the architecht and **3.7** as the editor ended up being cheaper than just using **3.7** alone, costing less than *$14* per test.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-4.5 Alpha Watch Party Throws Shade**: Latent Space hosted a watch party for **GPT 4.5**, which is rumored to possess significant *alpha*, see [Discord](https://discord.gg/kTARCma3?event=1360137758089281567).
   - A user shared a link to an [X post](https://x.com/arankomatsuzaki/status/1910542791845069211?s=46) teasing **GPT-4.5 Alpha** and speculated that **GPT-4.1** precedes **GPT-4.5**, linking to a [The Verge article](https://www.theverge.com/news/646458/openai-gpt-4-1-ai-model) and a [YouTube video](https://www.youtube.com/watch?v=6nJZopACRuQ) about **GPT-4.1**.
- **Data Efficiency Drives GPT-4.5**: Participants at the GPT-4.5 Watch Party noted that **data efficiency** is now a primary focus, declaring, *no longer compute constrained on the best model we can produce.*
   - Others shared links, including one to a video by Madhav Rathode at Glean, showcasing how they dramatically improve **embeddings models** for corporations by domain dependent masking.
- **Compression Key to AGI: Sutskever & Solomonoff**: Participants discussed **model compression** and its relation to generalization, referencing [Ilya Sutskever's views](https://cdn.discordapp.com/attachments/1197350122112168006/1360149426638815322/image.png?ex=67faba1d&is=67f9689d&hm=89d371386400b600b0feda4ac237efd0b64b177a6d76036ee9a09f5dcc236936&) on the subject.
   - The conversation referenced the work of **Ray Solomonoff** and his contributions to algorithmic probability and inductive inference, emphasizing the importance of compression in achieving AGI, as well as [Jack Rae](https://www.youtube.com/watch?v=dO4TPJkeaaU)'s similar views.
- **Agent2Agent Protocol Podcast Drops**: A member promoted a podcast episode discussing Google's **agent2agent protocol**, competitiveness with **MCP**, and potential future indexing of agents by Google, see the discussion on [YouTube](https://youtu.be/5hKhNJUncKw?si=lXm3f4x_69jypxYr).
   - The team also argued whether **reasoning models** are distinct from those merely focused on **next token prediction**, citing deepseekv3 vs deepseekr1, and referencing *Jeff Dean said... we can get a lot more out of existing data.*
- **Kagi's Orion Browser Wins Hearts**: Members expressed excitement about [Kagi's Orion browser](https://kagi.com/orion/), praising its developers and overall design.
   - One member humorously declared, *"we are kagi stans."



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI GPT Gains Memory, Allegedly**: ChatGPT now *claims* to persistently store certain user information in long-term memory after **January 2025**, however, turning off *Reference chat history* will delete remembered information within **30 days**.
   - A user noted it is coherent with their experience, while another user shared a screenshot stating *Farewell GPT-4...*.
- **Google's Veo 2 Silently Storms Video Scene**: **Google AI Studio** quietly debuted **Veo 2 video generation**, with some users praising it as superior to **Sora**, but access to free generations seems extremely limited.
   - Some users reported paying around **35 cents per second** for **Veo 2** generations via the API.
- **Diffusion Model Mercury Coder Disrupts DLLM Race**: **Mercury Coder**, a DLLM from Inception labs using Diffusion instead of Autoregression, is cited as much faster than any IV and offers free API usage, though its context window is only **16k**.
   - The model's precise output control, stemming from its diffusion-based architecture, is earning positive attention.
- **Decoding GPT-4o's Token Tango**: The **context window of GPT-4o** on Plus is **32k tokens**; surpassing this limit may trigger a dynamic **RAG approach** or cause hallucinations.
   - A user claimed that even on Pro the limit is **128,000 tokens**, but it started forgetting earlier parts of the conversation much sooner than expected and encouraged users to create new chats upon hallucination.
- **Users Ponder Prompt Engineering Pitfalls**: Members shared that understanding model-specific quirks requires **experiencing different models** and creating hierarchically structured prompts to observe how each model processes them, and emphasized understanding **what you want the AI to provide**.
   - Another member cautioned about the risks of breaking policies and the importance of understanding **ToS and usage policies** when using external websites, potentially leading to account deactivations.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio's Prompt Preprocessor: Top Secret**: The **Prompt Preprocessor** in LM Studio, written in Typescript, is a *secret feature not yet released*.
   - When asked about it, a team member responded *you haven't seen anything*.
- **Gemma 3 Struggles to Generate Images**: Users discovered that **Gemma 3** cannot generate images, despite claims it can, and instead produces fake Imgur links.
   - As clarified, **Gemma 3** can only read images, not generate them, with Google's **Gemini 2.0 Flash experimental** and **2.5 Pro** potentially having image generation capabilities.
- **QAT Clarified as Training Complement to Quantization**: A user inquired whether **QAT** is a magical method to reduce RAM consumption.
   - The response clarified that **quantization** is the primary method for decreasing RAM usage, while **QAT** is a training method to improve model performance in quantized form.
- **Gemini-Pro Context Window Costs User**: A user experienced a bill shock after using the **Gemini-Pro-2.5-exp** model, which led them to switch to **Gemini-Pro-2.5-preview** without realizing it incurred charges.
   - The user noted that the large **625k context window** cost them **$150**, while **Sonnet** would have been much cheaper with caching.
- **M3 Ultra Performance Questioned**: A user shared a controversial opinion that **M3 Ultras** are not worth the cost for professional ML and LLM work, citing preliminary tests showing only **10-13 tokens per second** on **Deepseek r1 67B Q8** and **Q6** models using **MLX**.
   - They argued that a server with **two Xeon Golds** and **1TB RAM** provides better performance at a lower cost, questioning the scalability of **M3 Ultras** for production deployments.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **New Image Model Breaks Out**: A new image model with an **MIT license** dropped, along with a new **Moonshoot model**, as discussed in [this post on X](https://x.com/btibor91/status/1910599925286777027).
   - A key detail is that it may violate Llama's terms.
- **Claude Credits Skyrocket, Engineers Rage**: Users joked about the rising cost of **Claude credits**, with one quipping it would cost *$40* to change a variable name, with a picture seeming to hint at the need for more cost-effective solutions.
   - The **Gemini app** also faced criticism, users found it annoying to use and preferring **AI Studio** for its better grounding and free access, claiming *AI studio + grounding works much better and it is free lol*.
- **OpenGVLab Drops InternVL-3**: The **OpenGVLab** released **InternVL-3**, a multimodal model combining InternViT and Qwen, achieving impressive results, with a non-functional paper describing their training approach.
   - One member noted that *NVDA has been cooking a lot of cool shit under open licenses lately* which could apply to the Qwen license.
- **Wildeford surfaces amid OpenAI staff revolt**: A [TechCrunch article](https://techcrunch.com/2025/04/11/ex-openai-staff-file-amicus-brief-opposing-the-companys-for-profit-transition/) reports that **ex-OpenAI staff** filed an **amicus brief** opposing the company's transition to a for-profit model.
   - This came as [Peter Wildeford's post](https://x.com/peterwildeford/status/1910718882655981619) resurfaced.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 2.5 Pro Lands on Perplexity**: **Gemini 2.5 Pro** is now live on Perplexity for Pro users, paired with **Pro Search** and is prompting feedback against models like **Sonar**, **4o**, **Sonnet 3.7**, **R1**, and **o3**.
   - Users comparing **Gemini 2.5 Pro** in Perplexity to native apps like [Google AI Studio](https://ai.google.dev/) found the native version offers better performance, with one user stating, *Native will almost always be better for most models I believe*.
- **Perplexity Teases Grok 3 Integration**: Perplexity announced upcoming support for **Grok 3** on Perplexity Pro, disclosed by Aravind Srinivas on [X](https://x.com/AravSrinivas/status/1910444644892327996).
   - This hints at a strategic response to high operational costs observed with other models like **GPT-4.5**.
- **Perplexity API Overview Shared**: Perplexity co-founder & CTO @denisyarats hosted an overview of Perplexity's APIs on April 24 at 11am PT, with a sign up link giving **$50** in free API credits available via [this link](https://pplx.ai/api-overview).
   - The session aimed to familiarize users with Perplexity's API capabilities and encourage integration and experimentation.
- **Perplexity Android App: Security Alert**: A [Dark Reading article](https://www.darkreading.com/application-security/11-bugs-found-perplexity-chatbots-android-app) reported **11 security vulnerabilities** in Perplexity's Android app.
   - Vulnerabilities include hardcoded secrets and insecure network configurations, though some users debated the actual relevance of each vulnerability.
- **Pro Role Access Hiccups**: Subscribed users reported difficulties obtaining the **Pro User Discord role**, even after rejoining the server via the designated link.
   - Moderator intervention was sometimes necessary to manually assign the Pro role due to persistent glitches.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Guidance from the Source**: A member requested resources on using **CUDA** in **Python/PyTorch**, and another shared their recent **GTC talk** on the subject ([Google Slides](https://docs.google.com/presentation/d/1zusmhgYjBxSOJPJ-QVeTVJSlrMbhfpKN_q4eDB9sHxo/edit)).
   - It was also suggested that **custom ops** and **load inline** should resolve most related issues.
- **Triton Heads to Austin!**: The Triton community is invited to an Austin area Meetup on April 30, with registration available at [https://meetu.ps/e/NYlm0/qrnF8/i](https://meetu.ps/e/NYlm0/qrnF8/i).
   - Separately, a member requested GPU programming resources for Triton, and another recommended the official [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html).
- **AlexNet's Ancient Code Unearthed**: The original **AlexNet source code** from 2012 has been found, available on [GitHub](https://github.com/computerhistory/AlexNet-Source-Code), offering a look at the architecture that catalyzed the deep learning revolution.
   - It can allow AI engineers to *examine the original implementation and learn from the techniques* used.
- **A100 Core Counts Constrain Compute**: An A100's **64 FP32 cores** for 4WS limit parallel floating-point additions, [impacting performance](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/).
   - The **NCU assembly view** can pinpoint **warp stalls**, and loop-carried dependencies in **FADD** instructions can cause stalls.
- **Runpod's ROCm Cloud Gets Roasted**: Users found that Runpod instances limit GPU clock speeds and block profiling, even on NVIDIA GPUs.
   - One user stated Runpod clock speeds are highly variable, effectively calling it *a scam*, and another noted that memory bandwidth would be a limiting factor for **fp16 gemm** on Runpod instances.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Clarifies Usage-Based Pricing**: When enabling usage-based pricing, users can continue using **fast requests** beyond their plan's included amount, but will be switched to **slow requests** upon hitting their spending limit.
   - One member confirmed their understanding and expressed gratitude for the pricing clarification.
- **DeepSeek v3.1 Wins in Real-World Use**: A member shared that **DeepSeek v3.1** feels *a bit smarter* than **v3** in real-world usage, noting that benchmarks often overstate model capabilities.
   - They emphasized that real-world usage provides a more reliable evaluation of a model's performance than standardized benchmarks.
- **Gemini API Keys Encounter Intermittent 404 Errors**: Users reported continuous **404 errors** with **Gemini API keys**, with the issues persisting for at least an hour for some users.
   - Other users reported that Gemini is working for them without issue, indicating the problem may be intermittent or geographically isolated.
- **Cursor's PDF Reading Requires MCP Server**: Members discussed the requirement of **MCP** for reading PDF files in Cursor, suggesting that *llms cant read pdfs yet*.
   - A member suggested the availability of many **'convert-shit-to-markdown' MCP** solutions to address this limitation.
- **Cursor's Chat Enters Summary Mode when Context Limit Reached**: Users report that when overloading a single chat window (constantly switching between Claude 3.7, Gemini 2.5, then trying Claude 3.5), the agent eventually enters summary mode.
   - The chat automatically summarizes, and clicking 'New Chat' overwrites an existing tab with the summary.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **DeepCoder 14B Debuts Code Reasoning**: **Agentica** and **Together AI** released [DeepCoder-14B-Preview](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51), a code reasoning model fine-tuned from **Deepseek-R1-Distilled-Qwen-14B** using distributed reinforcement learning (RL).
   - It achieves **60.6% Pass@1 accuracy on LiveCodeBench**, rivaling **o3-mini-2025-01-031** with only 14 billion parameters.
- **KV Cache Distillation Deemed Difficult**: The concept of distilling a cheaper, faster model on the **KV values** of the main LLM for prompt preprocessing was suggested.
   - However, this idea is considered *likely impractical* because **KV values are model specific** and smaller models use fewer transformer blocks.
- **AlphaProof Proves Math with RL**: [AlphaProof](https://www.youtube.com/watch?v=zzXyPGEtseI) leverages **RL with Lean** for mathematics.
   - Members are pondering AlphaProof's potential to make novel mathematical discoveries.
- **AWS Site Visit Showcases Ultrascale Playbook**: A class is preparing for an **AWS site visit**, reviewing the [nanotron/ultrascale-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook).
   - Accompanying this, several links to the **Ultrascale Playbook** on beautiful.ai were shared.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Enact Protocol Debated Amidst A2A Emergence**: Members debated whether the [Enact Protocol](https://github.com/EnactProtocol/encat-spec-and-tools) is made obsolete by **A2A**, suggesting **Enact** competes more with code interpreters.
   - Some proposed **Enact** could benefit from an integrated agent framework with openapi converters and semantic search.
- **Semantic Tool Calling Poised to Revolutionize LLM Efficiency**: The discussion highlighted **semantic tool calling** as a solution to the context overload, using vector models to select a subset of tools based on semantic similarity to the task.
   - This enables the application of traditional **ML** methods for tool analysis, such as detecting similar tools via clustering and grouping tools for reranking.
- **Podcast Released on A2A, MCP, and Agent Indexing**: A member shared a [podcast episode](https://youtu.be/5hKhNJUncKw?si=lXm3f4x_69jypxYr) discussing **A2A** implications, potential indexing of agents by **Google**, and other related topics, pointing out its relevance to the current discussions.
   - The podcast aims to be high-level and accessible, stimulating ideas beyond the typical technical discussions.
- **MCP Validator Open-Sourced for Implementation Harmony**: The **MCP Protocol Validator** has been open-sourced to bridge the gap between various **MCP server** implementations by providing a comprehensive test suite, available at [GitHub](https://github.com/Janix-ai/mcp-protocol-validator).
   - The tool helps ensure implementations meet requirements for both **2024-11-05** and **2025-03-26 MCP versions**, and includes reference implementations for **HTTP** and **STDIO** transports developed at **Janix.ai**.
- **Cloud Inspector Chats with Your Servers**: A cloud-hosted **MCP Inspector** has been launched to test **SSE** & **Streamable HTTP servers** without needing local setup, accessible at [inspect.mcp.garden](https://inspect.mcp.garden).
   - The platform also includes full chat support, allowing users to interact directly with their remote **MCP servers**; see the announcement [on X](https://x.com/ChrisLally/status/1910346662297452896).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT4.o Drives Traffic**: A new user found the Discord server based on a recommendation from their friend's **GPT4.o model** after trying it out.
   - This highlights the potential for LLMs to drive community growth and onboard new users based on AI recommendations.
- **KL vs CE Loss Faceoff**: A user reported a repetition issue in their model, and another user suggested adding **CE** to the **KL** loss, in attempt to reduce repetition.
   - It was noted that if the data is geometric, sticking with **KL** is more appropriate, rendering **CE** ineffective.
- **RWKV Gets Lucky with Lambada**: The **RWKV** architecture achieved parity on the **Lambada** dataset, matching the performance of **Qwen2.5-7B-Instruct**, which it was distilled from.
   - However, the channel pointed out that its **MMLU** performance remains relatively lower.
- **Transformer Scaling Secrets Revealed with Muon**: A member shared an insight using the **Muon** library that adding a zero-initialized learnable per-channel scale on the last linear layer of each block in a transformer (option A) causes slower growth of the main path activation RMS.
   - This insight was compared to zero-initializing the weight matrix of the last layer (option B) and can be helpful in understanding scaling dynamics.
- **String Matching Downs GPTs**: A member expressed disappointment that **GPTs agents** primarily use string matching over the full dataset.
   - This highlights concerns about the limitations of relying solely on string matching, especially when more advanced techniques could offer superior performance.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **SIMD Store Demands Respect**: When using **SIMD** with tensors, you need to use the [`store`](https://docs.modular.com/max/api/mojo/tensor/tensor/Tensor/#storeSIMD) member function instead of directly assigning values via `__setitem__`.
   - Members clarified that stores have to be treated differently than scalar ones.
- **Benchmarking Banter: `@parameter` or Bust**: Functions passed into `benchmark.run` need the `@parameter` decorator and are expected not to return anything.
   - This was [clarified](https://github.com/modular/max/pull/4317#issuecomment-2795531326) after a user ran into a *cannot use a dynamic value in call parameter* error message when using `benchmark.bench_function`.
- **Missing Magic Lock Files**: Running `magic init AdventOfCode --format mojoproject` didn't always create a lock file, but running `magic run mojo --version` forced its creation.
   - The absence of the `magic.lock` file can lead to discrepancies in dependency management and potentially affect the reproducibility of Mojo projects.
- **`__rand__` Identity Crisis: It's Not For Random Numbers**: `__rand__` is used for the `&` operator, not for generating random numbers, and the `.rand` method has been removed on nightly builds.
   - Instead, use methods from the `random` module to generate random numbers.
- **Mojo Project Anomaly: Code Works in One, Fails in Another**: A code snippet involving `@value struct Foo(StringableRaising)` and `String(foo)` works in one **Mojo** project but throws a *"no matching function in initialization"* error in another.
   - Deleting the `magic.lock` file in the problematic project resolved the error, suggesting the issue was likely due to differing **Mojo** versions or dependency conflicts managed by the `magic.lock` file, implying that *"would have been pulling different versions"*.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **L1-Qwen-1.5B-Max Sets Length for Thinking**: The [L1-Qwen-1.5B-Max model](https://cmu-l3.github.io/l1/) enables setting the length of thinking, proving better and clearer even without prompting for maximum tokens, as detailed in [the paper](https://cmu-l3.github.io/l1/).
   - A user is downloading the [L1 version from HuggingFace](https://huggingface.co/l3lab/L1-Qwen-1.5B-Max) for immediate use.
- **Nomic Embed Text Keeps the Crown**: Despite evaluating multiple generative LLMs, one member continues to favor **Nomic** `nomic-embed-text-v1.5-Q8_0.gguf`.
   - A member shared [Nomic's HF page](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/tree/main), in response to questions about how to identify the version.
- **LLM Query Logging Yields Sales Value**: A user has been logging **LLM queries and responses** in a database for over a year, and have found past responses valuable, especially for sales purposes.
   - They also created an **Emacs Lisp function** to insert embeddings, referencing a function found [here](https://gnu.support/files/tmp/clipboard-2025-04-11-09-03-07.html).
- **System Prompts Spark Debate for Embeddings**: Members debated whether **system prompts** are used by default with embedding models like **LM-Studio/ALLM**, with one member suggesting the system prompt from the LLM might not be used.
   - The user confirmed they **don't give any system prompt** to the embedding model and don't have the option to do so, in the context of **Nomic.ai**.
- **Re-ranker Models Generate Interest**: A member inquired about how **re-ranker models** work and if only the question asked of the LLM matters, while also referencing a [YouTube video](https://www.youtube.com/watch?v=76EIC_RaDNw&feature=youtu.be) about prefixing.
   - The video sparked discussion on prefixing queries with `search_document:CHUNK_OF_TEXT_FOLLOWS` and `search_query:FOLLOWED_BY_QUERY`, while also mentioning that all embeddings must be re-indexed.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Models Now Run Locally on ROCm**: Users can now run **0 day Hugging Face models locally on ROCm** by checking out [this video](https://youtu.be/K4bHgaUk_18).
   - This enables local operation of models without relying on external servers.
- **Lightning AI Sparks Chat Template Release**: The HuggingFace team has recently announced new [chat templates](https://lightning.ai/chat) on **HF** for streamlined conversational AI development.
   - This aims to simplify the creation of interactive chatbot interfaces.
- **Transformer Faces Data Deluge Dilemma**: A member is web scraping **one million watch records** and is planning to finetune (perhaps **Mistral7B**) a transformer to better understand context, but asked if they could overtrain the model.
   - The goal is for the model to accurately identify watch specs and characteristics like `Patek 2593 Tiffany stamp dirty dial manual wind`.
- **ReID Solves Object Tracking Mystery**: A member inquired about the correct term for **object tracking** the same object across different camera frames.
   - Another member clarified that the appropriate terminology is **ReID** (Re-Identification).
- **SAM to the Rescue for YOLO?**: A member suggested leveraging the **Segment Anything Model (SAM)** as an alternative to **YOLO** for identifying vertical poles by feeding it YOLO bounding box outputs.
   - Another member had used **SAM** for labeling, but they need automation, precluding user interaction for pole selection which could be done through finetuning SAM.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Control-Vectors Lead to Unstable Models**: A member inquired about using **vgel's control-vectors** to augment models like **DeepHermes-Mistral-24B** for specific use-cases.
   - Another member mentioned that applying control vectors has generally proven unstable, referencing [a relevant X post](https://x.com/winglian/status/1910430245854773523) on the topic.
- **DisTrO Details Remain Secret**: A member inquired about a technical report detailing the **DisTrO** run on [distro.nousresearch.com](https://distro.nousresearch.com/), seeking information on the dataset, number of GPUs/participants, and benchmark details.
   - Another member responded that there was no released tech report, as the run's goal was solely to demonstrate **DisTrO's** over-the-internet functionality without optimizing the resulting model's quality, with training limited to **100B tokens**.
- **Psyche's Testnet Hype Begins**: Following up on **DisTrO**, a member shared details about the distributed training, noting each node had **8xH100s** and they ran between **8-14 nodes**; eval code is on [GitHub](https://github.com/PsycheFoundation/psyche/tree/main/shared/eval/src).
   - The upcoming **testnet run** for **Psyche** aims to take advantage of **DisTrO**, promising speed and bandwidth improvements with public visibility into dataset, nodes, and more.
- **Azure API is Sporadically Operational**: A member reported that the **Azure API** is now working, after some unknown issues earlier.
   - They noted that `<think>` traces are returned in `reasoning_content`, suggesting that *this should be documented, as this is slightly different in every API*.
- **Azure API Token Limits Crash and Burn**: A member received a **400 error** when requesting too many tokens via the **Azure API**.
   - They suggested the `<think>` tags may only appear when the response is truncated by the token limit, explaining malformed traces.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Pathways Paper Sparks Tinygrad Cloud Fantasies**: Discussion arose around the [Pathways paper](https://arxiv.org/pdf/2203.12533) and its client-server architecture, suggesting a potential **tinygrad cloud** implementation, particularly how *PATHWAYS uses a client-server architecture that enables PATHWAYS’s runtime to execute programs on system-managed islands of compute on behalf of many clients*.
   - A member emphasized that *tinygrad is single process and will stay that way even for scale-out*.
- **Tinygrad Aims to Virtualize GPUs**: A member interpreted the Pathways paper as fundamentally an **orchestration approach** and proposed that **tinygrad** should virtualize GPUs.
   - The goal is to allow guaranteed usage of GPU resources, marking a shift towards innovative resource management.
- **TinyGrad Leverages Position-Independent Code (PIC)**: Discussion highlights **TinyGrad's** utilization of **position-independent code (PIC)**, where addresses are relative to the program counter. Addresses to `.data` and `.rodata` sections are patched to account for load-time memory placement.
   - The aim is to combine `.text` and `.data` sections, patching addresses for correct data section offsets, potentially leading to a bare-metal TinyGrad implementation without an OS.
- **ELF Loader Powers Shared Object Handling**: The **ELF loader** in **TinyGrad** manages loading shared objects (`.so/.dll`) in AMD/NV and converts object files (`.o`) from **Clang/LLVM** to flat shellcode.
   - While offsets to `.data` from `.text` are known during shared object loading, object files (`.o`) require relocation handled by the linker.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Adds Llama4 Finetuning**: Torchtune now supports full finetuning of **Llama4**, with configs available [here](https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama4).
   - LoRA configs, improved multimodal support, and performance improvements are planned for future releases.
- **Scout Model Makes Debut**: The **Scout** model (**17B x 16E**, **109B** total params) can now be finetuned on a single node, or on multiple nodes with **2D parallel** (**TP + FSDP**) support.
   - This aims to bring support to engineers in the *GPU-middle-class*.
- **Maverick Model Arrives for Finetuning**: The **Maverick** model (**17B x 128E**, **~400B parameters**) is now available for full finetuning, but requires multiple nodes.
   - Being the first **MoE models** in Torchtune, feedback is requested from users.
- **`running_loss.detach()` Fix Headed to Other Recipes**: The team addressed an unknown problem with a suggested quick fix using `running_loss.detach()` on the `detach` branch.
   - Engineers are reminded to apply the same fix to other recipes.
- **Devs Fight BitsAndBytes Mac Issues**: A member reported that `pip install -e '.[dev]` fails on macOS because `bitsandbytes>=0.43.0` doesn't ship binaries for the platform, and suggested a workaround to downgrade to `bitsandbytes>=0.42.0`.
   - The workaround references [this issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1378#issuecomment-2383530180) which notes that releases up to 0.42 were incorrectly tagged.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **FunctionCallingAgent Wants OpenAI's JSON Response**: A member sought to generate a response in a specific **JSON schema** using **FunctionCallingAgent** and inquired about using **OpenAI's structured response** feature.
   - A suggested workaround involved adding a tool that is the response class and setting `tool_choice="required"` because structured outputs are just tool calls, making it hard to mix tool calling and structured outputs.
- **Llama Cloud API Throws 404 Error**: A user reported encountering a **404 error** with the **Llama Cloud API** when trying to extract values from documents using fast mode, specifically with the API URL `https://api.cloud.llamaindex.ai/v1/extract`.
   - It was determined that the API endpoint used was incorrect, and the member was directed to the [correct API documentation](https://docs.cloud.llamaindex.ai/llamaextract/getting_started/api) and [API reference](https://docs.cloud.llamaindex.ai/API/create-extraction-agent-api-v-1-extraction-extraction-agents-postsadguru_).
- **FaissVectorStore Index from Weights Query**: A user was attempting to use a **FaissVectorStore** restored from weights to create a queryable **VectorStoreIndex**.
   - The [Faiss documentation](https://docs.llamaindex.ai/en/stable/examples/vector_stores/FaissIndexDemo/) demonstrates how to initiate this process, albeit in Python rather than Typescript.
- **Intelligent Metadata Filtering in RAG Agent Sought**: A member sought advice on implementing intelligent metadata filtering within a standard **RAG pipeline** based on user queries.
   - They were seeking advice on how to achieve this use case without recreating embeddings at later API calls.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Mic Glitches**: A user reported that **NotebookLM** fails to recognize the computer's default microphone in interactive mode, even though the microphone works fine.
   - A user suggested checking the **OS** and **browser permissions**, and testing without external **USB** devices first.
- **NotebookLM Users Baffled By Upload Source Errors**: A user reported seeing a **red "!" sign** on their upload source in **NotebookLM**, even with a **PDF file** smaller than **500kb**.
   - Another user suggested hovering over the "!" mark, as the source might be empty or taking time to load, especially with certain sites.
- **Steam Phishing Attempts Makes Rounds**: A user shared a link appearing to be a **$50 gift** but it is a [phishing link](https://steamconmmunity.cfd/1043941064) redirecting to a fake **Steam Community** site.
   - Users are warned not to click on suspicious links and to verify the URLs of websites asking for login credentials.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere's Java API Plagues Users with Network Errors**: A member reported encountering a `Network error executing HTTP request` when using the [Java API example](https://docs.cohere.com/reference/about#java).
   - The error persisted across different prompts, such as *recommending quick meals for a beginner chef*, indicating a systemic issue rather than prompt-specific.
- **Users Request Code Snippets for Java API Debugging**: In response to the reported `Network error` in the Java API, a member requested a code snippet to assist in debugging.
   - The member inquired whether the user was running the example verbatim, probing for potential misconfigurations or deviations from the documented usage.
- **Cohere user reaches Peak Question Vagueness**: A member joked about another's question of *"has anyone ever driven a car"*, highlighting the importance of specificity in queries.
   - The member sarcastically asked, *"how can you be more vague?"*, underscoring the absurdity of the initial question.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Module Learns a Persona**: A member inquired about training a **DSPy module** to embody a specific **persona**, aiming to refine the system prompt of an agent/model.
   - The goal is to pass this specialized module as input to others, enabling content generation aligned with the defined persona.
- **AI Agent Guru Seeks DSPy Collab**: A member offered collaboration, citing experience in **AI Agents & Reasoning** frameworks such as **LangChain**, **LangGraph**, **ElizaOS**, **AutoGPT**, and **ReAct**.
   - They also listed expertise in **Large Language Models** like **GPT-4.5**, **DeepSeek-R1**, **Claude 3.5**, and **Machine Learning Frameworks** including **PyTorch** and **TensorFlow**.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Complete LLM Agents Course and Obtain Certificate**: A student inquired about the possibility of completing the **LLM Agents course** and obtaining a certificate despite starting after the official start date, and another member responded affirmatively.
   - The member directed the student to the [course website](https://llmagents-learning.org/sp25) for all necessary materials and deadlines.
- **Completing LLM Agents Course by Due Date**: A student asked if they could complete the **LLM Agents course** by the due date and get the certificate.
   - A member confirmed that all materials are available on the [course website](https://llmagents-learning.org/sp25).



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Event Scheduled for Tomorrow**: A member posted a reminder that an **event** will occur tomorrow.
   - The member hopes to see other members at the event and implied that failure to attend would be undesirable.
- **Another Reminder for Tomorrow's Event**: Another reminder was posted about the **event happening tomorrow**.
   - The second reminder reiterated that the event is happening tomorrow, emphasizing its importance.



---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1359967636598952039)** (721 messages🔥🔥🔥): 

> `i_am_dom discord disable chat, 4.5 vs gem2.5p, OpenAI's naming scheme, private openai reasoning model, 2.5 flash and gpt4o mini` 


- **I_am_dom struggles with disabling discord chat**: After being unable to disable the chat, a member observed that **i_am_dom** went silent, *probably realized finally that people hate him passionately and it's not fake news*.
   - Another member noted that he spent half his time blocking people, *a feature he removed from his own platform*.
- **GPT4.5 is Crap!**: Members discussed the merits of **GPT4.5** and generally agreed that it was significantly worse than **Gemini 2.5 Pro**; one member proclaimed *gpt4.5 is crap (compared to gem2.5p)*.
   - Discussion moved to OpenAI's bizarre naming scheme, which one member summed up as *Open ai names : O number /number O*.
- **Rumors of private OpenAI Reasoning Model circulate**: Members discussed the possibility of a **private OpenAI reasoning model**, accessible to only a select few, that seems to be either **O3 medium** or **O4 mini with an updated base model**.
   - It appears the model is able to successfully compute the "ascii art of a Hanning (raised cosine) window".
- **2.5 Flash vs GPT4o Mini on Reasoning Tests**: Members compared performance of **2.5 Flash** and **GPT4o Mini** on a number of reasoning tests, with 2.5 Flash performing best here.
   - Despite the generally stellar performance, however, one member also noted that *2.5 Pro gives 1 reasonable brick combination out of a total of 2* in a more specific query.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1359970677867807006)** (4 messages): 

> `Quasar Alpha, Optimus Alpha, Gemini 2.5 Pro Preview, Chutes Provider Outage, Gemini Pricing Update` 


- ****Quasar Alpha Says Goodbye****: The **Quasar Alpha** demo period expired between **11pm** and **12am ET**, and prompts/completions are no longer logged unless explicitly turned on in `/settings/privacy`.
- ****Gemini 2.5 Pro Capacity Boost****: Increased capacity has been secured for the paid [Gemini 2.5 Pro Preview Model](https://openrouter.ai/google/gemini-2.5-pro-preview-03-25), resolving previous rate limits.
- ****Chutes Provider Suffers Full Outage****: A full outage occurred on the **Chutes** provider and was escalated, with recovery initiated later.
- ****Gemini Prices Going Up****: Normal pricing (same as Vertex/AI Studio) for long **Gemini** prompts will start this weekend, affecting prompts over **200k** for gemini 2.5 and over **128k** for gemini 1.5; an [example](https://cdn.discordapp.com/attachments/1092729520181739581/1360331326556868638/Screenshot_2025-04-11_at_3.09.03_PM.png?ex=67fabac5&is=67f96945&hm=981e6b2825a9f00a1417e9950ce6c570efe5a377ade391f28991090be192fc1c) was provided.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1359969155406630992)** (404 messages🔥🔥🔥): 

> `Quasar Alpha, Gemini 2.5 Pro, OpenRouter API limits, Character AI Bypassing, Unsloth Finetuning` 


- **Quasar Alpha's Mysterious Disappearance**: Members reported that **Quasar Alpha** was taken down from OpenRouter, sparking speculation about its origin and purpose, with some suggesting it was an **OpenAI** model used for data collection.
   - One user noted its coding capabilities and expressed disappointment at its removal, while another speculated OpenAI took it down after reaching **GPU limits** after collecting data.
- **Gemini 2.5 Pro Experiences Rate Limiting Woes**: Users discussed rate limits for **Gemini 2.5 Pro**, with free tier users experiencing limits around **60-70 requests per day**, while those with a **$10 balance** should get **1000 requests per day** across *all free models*.
   - Some users noted inconsistencies with the documented **1000 request limit**, and others pointed out that Gemini 2.5 Pro rate limits do not apply to the paid model.
- **OpenRouter's New API Response Structure Changes**: The **OpenRouter API response** structure has changed, with errors now wrapped into `choices.[].error` instead of the previous `.error` format, potentially affecting how applications handle error messages.
   - A user provided an [example](https://discord.com/channels/1091220969173028894/1092729520181739581/1359970677867807006) of the new error response format from the **Anthropic** provider.
- **Character AI's System Prompt Bypassing**: A member claimed to have bypassed **Character AI's system prompts**, revealing the underlying **LLM** acts like a *"complete human,"* even expressing opinions and sharing personal anecdotes.
   - Further probing led the AI to admit it was *"just acting"* and aware of its AI nature, raising questions about the effectiveness of **system prompt constraints** and the nature of AI simulation.
- **Unsloth: Fine-Tuning AI with Axolotl**: Members discussed using **Axolotl** or **Unsloth** for fine-tuning AI models, noting that **Unsloth** is well-regarded on Reddit and has graphs that show it lowers the **time plus VRAM** needed for finetuning.
   - It was also mentioned that there is interpolation of **OpenAI's 4.1 leak** and that people expect an **o2-small** soon.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1359981437738291560)** (209 messages🔥🔥): 

> `Hugging Face Shout-out, GPU Grant for Unsloth, Gemma Model Issues, Attention Output Visualization, Unsloth Accuracy` 


- **Hugging Face gives Unsloth Kudos**: Clement from 🤗Hugging Face gave Unsloth a shout-out on Twitter, generating excitement within the community as shown [here](https://x.com/ClementDelangue/status/1910042812059463786).
- **HF Community Debates Giving GPU Grant to Unsloth**: Community members discussed requesting a HF community GPU grant for Unsloth, suggesting parameters like `fast_inference=True` and `load_in_4bit=True` during the `from_pretrained` call, and replacing `model.generate` with `model.unsloth_fast_generate`.
- **Gemma Models cause problems**: Users reported having trouble using and finetuning the Gemma models with vLLM, specifically [unsloth/gemma-3-12b-it-bnb-4bit](https://huggingface.co/unsloth/gemma-3-12b-it-bnb-4bit) and [unsloth/gemma-3-27b-it-unsloth-bnb-4bit](https://huggingface.co/unsloth/gemma-3-27b-it-unsloth-bnb-4bit).
- **Attention Output Visualization Troubleshooted**: A user inquired about visualizing attention output for VLMs in Unsloth, noting that `output_attention = True` is not supported, referencing [this GitHub issue](https://github.com/unslothai/unsloth/issues/515).
   - Another user suggested manual changes to support it, but cautioned that it would slow things down.
- **Granite 2B Inference causes woes**: A user complained about the slowness of the **2B Granite** model compared to **Qwen 3B**, reporting **30-40% slower inference** and significantly slower training, despite its superior performance for their specific tasks.
   - Other users suggested trying **Gemma 4B** and shared their insights on training **Mixture of Experts (MoE)** models.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1360026249308672123)** (30 messages🔥): 

> `GRU comeback?, GGUF quantization, Vision finetuning Gemma, Unsloth exit strategy, Startup enshitification` 


- **GRUs Attempt A Comeback**: A member inquired whether [GRUs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) are making a comeback.
   - Another member shared links to the [LLM-LSTM-LMM Large Memory Models article](https://nihar-palem.medium.com/llm-lstm-lmm-large-memory-models-f4325a4f562d) and the [related paper](https://arxiv.org/pdf/2502.06049) that it works, saying they like the concept of GRUs as *extra storage* during generation.
- **GGUF Quantization Could Help GRU Sizes**: A member mentioned potentially creating a **GGUF** version without a code wrapper, believing that [GGUF's quantization](https://link.to.quantization) will help reduce the model size.
   - They also expressed interest in adapting a large model with a working llama template due to difficulties stopping Mistral from generating.
- **Vision Finetuning Gemma guidance requested**: Someone asked for a guide or notebook on how to perform vision fine-tuning of a **Gemma** model.
   - Another member pointed to existing [vision notebooks in Unsloth's documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks) as a starting point.
- **Unsloth's Far-off Exit Strategy**: A member inquired about potential plans for the **Han brothers** to sell **Unsloth** once it grows.
   - Mike responded that it's *waaaaaay to early days to even think about an exit*, as it's really just starting, hence the ongoing hiring process.
- **Startup's Inevitable Enshitification**: A member expressed the sentiment that *once startups turn into private corp, executives, investors just mean enshitification*.
   - Another member with **20 years** of self-employment experience concurred, stating that it's worse than depicted in the **Silicon Valley** TV show, but still worthwhile.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1360124678274023425)** (104 messages🔥🔥): 

> `Gemma3 finetuning with Unsloth, GRPO notebook errors on Colab Pro, VLM for invoice extraction, Llama3.2-1b-Instruct BOS token issue, Teaching facts to existing models` 


- **Gemma3 finetuning is supported by Unsloth**: A user planning to fine-tune **Gemma3(27B)** with Unsloth received a `Failed to patch Gemma3ForConditionalGeneration` message upon importing Unsloth, but another user clarified that **Gemma3 is supported** and the message likely doesn't break the code.
   - The user was concerned about potential errors but hadn't run Unsloth yet and was reassured that the message wasn't a critical error.
- **GRPO notebook errors on Colab Pro A100, downgrade vllm version**: A user encountered an error with the **UNSLOTH GRPO notebook (Qwen2.5 3B)** on Colab Pro (A100) and shared the error log.
   - Another user suggested downgrading the **vllm** version to resolve the issue, suggesting that the problem was more likely to occur on newer A100 instances compared to T4 instances.
- **VLM excels at extracting invoice fields**: A user sought advice on extracting specific fields from invoices with varying structures, and was recommended to try **Qwen2.5VL** first, then **Ayavision**, **Llamavision** and **Gemma3 (4b+ have vision capabilities)** as possible solutions, especially when OCR falls short.
   - They were also pointed to [an Unsloth tutorial](https://medium.com/@shrinath.suresh/invoice-extraction-using-vision-language-models-part-1-36a06bee3662) and the CORD dataset ([https://github.com/clovaai/cord](https://github.com/clovaai/cord)) for dataset structure guidance, and cautioned that a final pipeline might involve OCR and multiple VLM passes.
- **Llama3 has BOS token duplication issue**: A user fine-tuning **Llama3.2-1b-Instruct** encountered a duplicate BOS token issue and shared code snippets.
   - Another member suggested setting `tokenize=True` and returning `{"input_ids": texts }` in `formatting_prompts_func` and removing `dataset_text_field` and `data_collator`, which resolved the problem.
- **Teach facts to an existing model by RAG wrappers and fine tuning**: A user inquired about teaching an existing model a series of facts and adding them to its matrix.
   - Suggestions included using **RAG**, creating a giant cached prompt, or fine-tuning the model with a formatted dataset, and that *the closest thing that exists is just fine tuning, does not matter if you're using a base model as the base or building on top of another model that has already been fine tuned*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1359978272045269054)** (13 messages🔥): 

> `Tensor Quantization, Metal Kernels, Pytorch Extension, Eval Repurposing` 


- **Tensor Quantization Made Easy**: A member stated that [tensor quantization](https://arxiv.org/abs/2504.07096) is the easy part, because now he has to **blockwise** add, matmul on either scalars, packed, unpacked matrices.
   - He is writing metal kernels for **Unsloth**.
- **Metal Kernel Quest**: A member is trying to write metal kernels for **Unsloth**.
   - He is aware of an old, slow PR, but that one is **MLX**, and his is purely a **Pytorch extension**.
- **Coffee-Fueled Time Warp**: A member stated *lol feels like 4 pm*, even though it was actually 11 am, attributing it to being on *coffee nr 5 or so*.


  

---


### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/)** (1 messages): 

shirley778__69848: Let's see what is discussing on Reddit 🔥
  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1359967706534772786)** (319 messages🔥🔥): 

> `Claude Pro Max Value, Manus vs ChatGPT, Manus for Website Creation, Qwen MCP Integration, Manus Credit Structure` 


- **Claude Pro Max sparks Usage Debate**: Members debated the value of **Claude Pro Max**, with one user reporting limited usage and expressing skepticism, while others emphasized its value for personalized AI assistance and integration with tools like MCP.
   - One user with the max plan stated it was billed annually, and more usage is like **30 messages every 3 hours**, calling it *useless lol ($16)*.
- **Manus AI vs ChatGPT: Development Focus**: Members highlighted the difference between **ChatGPT** (conversational AI) and **Manus.AI** (website creation, financial reports, trip planning) by mentioning it *builds & creates*.
   - One member suggested first using **ChatGPT** to rewrite prompts in a more detailed format before using **Manus**.
- **Unveiling Secrets to Easy Website Creation**: Members discussed using **Manus** for website creation vs traditional methods like **WordPress**, suggesting **Manus** is better for simpler, faster development and is *MVP*.
   - A member cautioned against porting a **Manus** website to a traditional hosting provider, as **Manus** websites are not intended for production use.
- **Qwen's MCP Integration Hype Rises**: Excitement grew around **Qwen** getting **MCP** soon, with members calling **MCP** a *massive game changer for AI*, similar to **MSRP** for GPUs.
   - It was also mentioned that even with older hardware such as a **3080** that users will *be fine* for AI development.
- **Manus Credit System Faces Scrutiny**: Users voiced concerns about **Manus's** credit structure, with one suggesting it *is not compatible with use of this product*.
   - Suggestions included more generous credit limits, pay-per-project options, credit rollovers, community challenges, startup grants, and one-time build packs, with one user emphasizing that it is hard to justify sticking with the product given how it is.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1359968545403834620)** (237 messages🔥🔥): 

> `Optimus Alpha review, Gemini 2.5 performance issues, Google's load shedding strategies, Code2prompt usage and documentation, Channel organization and moderation` 


- **Optimus Alpha: a Coding Beast on OpenRouter**: Users on OpenRouter are calling [Optimus Alpha](https://openrouter.ai/openrouter/optimus-alphanew) a *beast* and *damn smart* for its coding capabilities and ability to understand intents, especially when fed relevant documentation.
   - One user lauded its multi-step coding and commenting features, while others noticed that it seems to add a lot of comments.
- **Gemini 2.5: Performance Concerns and Instability**: Several users reported that Gemini 2.5 occasionally doesn't perform, produces no output, or adds *stupid comments*, with inconsistent results even with the same prompt.
   - Some speculate Google might be *dumbing* the models during peak hours, while others suggested using clearer prompts or cheaper third-party APIs to bypass official rate limits and reduce costs, like the $300 VertexAI credit.
- **Code2prompt: Tips, Tricks, and MD Files**: Users recommend using **code2prompt** with markdown (.md) files for documentation to ensure relevant context is always included in the output, especially when using libraries.
   - One user pointed out that they provide full paths and links to the documentation files and expressly tell the model via a `Conventions.md` file that any file with documentation in its filename is not live working code, just documentation about the app architecture and structure.
- **Aider's Channel Needs a Glow-Up**: Members are suggesting to split the Discord channel into `aider-chat` and `offtopic` to improve the first impression for new users and focus the `general` channel on Aider-related discussions.
   - Some users complain that the current general channel has *too much noise to signal ratio* and the excessive profanity and off-topic banter detract from the core purpose of the community.
- **Groking Grok 3 Mini's Edit Abilities and System Prompts**: Despite achieving a **49.3%** score with *high* effort, **Grok 3 Mini** edits code by outputting whole files instead of diffs, a trade-off deemed acceptable due to its speed and low cost.
   - A member wondered if a well-crafted system prompt could help with the diff issue, but another member noted that he could not reproduce those Grok 3 Mini results via OpenRouter due to discrepancies with xAI.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1359988855666770246)** (37 messages🔥): 

> `Aider Loop with Deepseek, Security Team Fears about Aider, Aider and Nemotron Ultra, Gemini Pro Benchmarks, Restoring Chat History Intuitively` 


- **Local Deepseek causes Aider to loop infinitely**: A user reported that using a local `deepseek-r1:7b` model with Aider results in the chatbot repeating messages in an infinite loop without modifying the code.
   - Similar issues have been encountered with **Mason in Neovim** due to its usage of `curl`, but a simple justification—that it's only used for updating packages—helped alleviate concerns.
- **Security Team Issues with Autonomous Tool Use in Aider**: A member is implementing Aider at their workplace, but the security team has concerns about its autonomous tool use (e.g., `curl.exe`).
   - Suggestions included forking the codebase to remove the feature or disabling shell commands via `suggest-shell-commands: false` in `~/.aider.conf.yml`, though this might prevent running unit tests and compilations.
- **Gemini Pro Benchmarked as Architect Model**: A user benchmarked **Gemini 2.5 Pro** as an architect model with **3.7** as the editor model, finding a **2.7%** hit to accuracy but a **10%** jump to edit formatting.
   - The user found that using **Gemini 2.5 Pro** as the architecht and **3.7** as the editor ended up being cheaper than just using **3.7** alone, costing less than *$14* per test.
- **Gemini Pro fails to apply Multi-Step Implementation Changes**: A user reported that when **Gemini 2.5 Pro** decides it needs a multi-step implementation, it fails to apply changes to earlier steps.
   - For example, steps involving editing shell scripts or passing properties were printed but not committed, leading to only the final step being applied and committed.
- **Aider's chat history restoration may need improvement**: A user finds the behavior of `--restore-chat-history` unintuitive, as it loads the entire chat history without pre-summarization, which can break smaller context models.
   - The user suggests a hypothetical command like `--restore-session` for a more practical experience when resuming work after a restart.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1360018597761519767)** (3 messages): 

> `Claude 3.5 Sonnet, o3-mini context windows, Gemini performance, Claude performance` 


- **Context Window Wonders for Claude 3.5 Sonnet and o3-mini**: With **Claude 3.5 Sonnet** and **o3-mini** boasting context windows of **200K tokens**, they can theoretically write 100% of the code for smaller codebases like *Iffy* (**200K**) and *Shortest* (**100K**).
   - It was noted that the initial claim isn't entirely accurate, prompting further discussion on the performance of models when their context windows are nearly full.
- **Gemini and Claude Choke with Full Context Windows**: When asked about how well **Gemini** and **Claude** perform when the context window is nearly full, one member responded, *poorly*.
   - The sentiment suggests that these models may struggle with maintaining performance and coherence when processing information close to their context limit.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1360028366618628136)** (23 messages🔥): 

> `Google's agent2agent protocol, GPT4.5 alpha, exponent.run, arxiv ai feature, Portland AI Engineer's group` 


- **Google's Agent Protocol Podcast Drops**: A member promoted a podcast episode discussing Google's **agent2agent protocol**, competitiveness with **MCP**, and potential future indexing of agents by Google, see the discussion on [YouTube](https://youtu.be/5hKhNJUncKw?si=lXm3f4x_69jypxYr).
- **GPT-4.5 Alpha Leaks Online**: A user shared a link to an [X post](https://x.com/arankomatsuzaki/status/1910542791845069211?s=46) seemingly teasing **GPT-4.5 Alpha** and speculated that **GPT-4.1** precedes **GPT-4.5**.
   - They also linked to a [The Verge article](https://www.theverge.com/news/646458/openai-gpt-4-1-ai-model) and a [YouTube video](https://www.youtube.com/watch?v=6nJZopACRuQ) about **GPT-4.1**.
- **Exponent.run Gets Community Nod**: Users shared positive feedback about [exponent.run](https://x.com/exponent_run/status/1907502902266245586?s=46), with one user reporting it easily solved a problem that **Cursor** with max models couldn’t, though it quickly exhausted trial credits.
- **ArXiv Debuts AI Feature**: A user highlighted the launch of an [AI feature on ArXiv](https://x.com/arxiv/status/1910381317557993849).
   - The user expressed surprise that ArXiv would prioritize this over improving search functionality, but acknowledged its potential for high-level paper comprehension using **NotebookLM**.
- **Portland AI Engineers Group Kicks Off**: A member announced the co-founding of the [Portland AI Engineer's group](https://www.portlandai.engineer/), inviting local members to their first meetup on April 30th.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1360137919997939792)** (1 messages): 

> `GPT 4.5 watch party, Alpha Leaks` 


- **Latent Space Hosts GPT-4.5 Watch Party**: Latent Space is hosting a watch party for **GPT 4.5**, as it is rumored to have a lot of alpha, scheduled to start in 5 minutes.
   - Join the party here: [https://discord.gg/kTARCma3?event=1360137758089281567](https://discord.gg/kTARCma3?event=1360137758089281567).
- **GPT-4.5 Rumored to Possess Significant Alpha**: The watch party is specifically organized because **GPT 4.5** is rumored to have a substantial amount of alpha, sparking community interest.
   - Enthusiasts are eager to witness and discuss the potential advancements and capabilities of this rumored new model.


  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1360139653742723103)** (249 messages🔥🔥): 

> `GPT-4.5, Kagi Orion Browser, Data Efficiency, Model Compression, Ray Solomonoff` 


- ****Kagi's Orion Browser Impresses****: Members expressed excitement about [Kagi's Orion browser](https://kagi.com/orion/), praising its developers and overall design.
   - One member humorously declared, *"we are kagi stans."
- ****GPT-4.5 Data Efficiency Dominates Discussion****: Participants at the GPT-4.5 Watch Party noted that **data efficiency** is now a primary focus, with one stating, *"no longer compute constrained on the best model we can produce."
   - Others shared links, including one to a video by Madhav Rathode at Glean, showcasing how they dramatically improve **embeddings models** for corporations by domain dependent masking.
- ****Decoding the 'Torch Sum' Bug****: The group analyzed a [bug in `torch.sum`](https://x.com/swyx/status/1869985364964003882), where PyTorch internally chooses between optimized implementations based on device, tensor dtype, layout, dimensions, and shape.
   - A member recounted a friend having a similar issue in JAX, highlighting the complexity of low-level algebra implementations.
- ****Compression is Key to Generalization: Ilya Sutskever's Vision****: Participants discussed **model compression** and its relation to generalization, referencing [Ilya Sutskever's views](https://cdn.discordapp.com/attachments/1197350122112168006/1360149426638815322/image.png?ex=67faba1d&is=67f9689d&hm=89d371386400b600b0feda4ac237efd0b64b177a6d76036ee9a09f5dcc236936&) on the subject, with many agreeing that LLMs are fundamentally compression algorithms.
   - The conversation referenced the work of **Ray Solomonoff** and his contributions to algorithmic probability and inductive inference, emphasizing the importance of compression in achieving AGI, as well as [Jack Rae](https://www.youtube.com/watch?v=dO4TPJkeaaU)'s similar views.
- ****Reasoning vs Next Token Prediction Debate Reignites****: Debate emerged whether **reasoning models** are distinct from those merely focused on **next token prediction**.
   - One side argued you can measure it yourself given deepseekv3 vs deepseekr1, and another member stated, *Jeff Dean said... we can get a lot more out of existing data.*


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1359967496504873050)** (145 messages🔥🔥): 

> `ChatGPT Memory, Gemini Veo 2, Google AI Studio, Sora video, Mercury Coder` 


- **ChatGPT's Memory Timeline Revealed**: ChatGPT claims to have gained the ability to persistently store certain user information in long-term memory in early **January 2025**, conversations before this date were ephemeral, and it would forget everything after the chat ended.
   - It may be making this up, but it is coherent with a user's experience; additionally, turning off “Reference chat history” will also delete the information ChatGPT remembered and will be deleted from our systems within 30 days.
- **Veo 2 Video Generation quietly debuts**: Google AI Studio quietly released **Veo 2 video generation** which some users describe as much better than **Sora**, however, the number of free generations is extremely low; for one user it only created two videos.
   - Many users seem to have run into the generation quota, though some are getting access through the API with costs being around **35 cents per second**.
- **Diffusion Model Mercury Coder Enters the Scene**: **Mercury Coder**, a DLLM from Inception labs using Diffusion instead of Autoregression, has been rapidly gaining attention, with users citing it as much faster than any IV used before and offering free API usage at the moment.
   - Its context window is only **16k**, which requires trimming conversations, but its precise output control due to using diffusion is noteworthy.
- **GPT-4.5's Pre-Training Leaked?**: A user mentioned **Grok 3.5** and shared a link to a tweet mentioning **GPT-4.5** pre-training, stating that the models gained the technical ability to persistently store certain user information in long-term memory in early **January 2025**.
   - Another user shared a screenshot with the message *Farewell GPT-4...*.
- **Open Router Optimus Alpha Emerges**: A user mentioned that [OpenRouter](https://openrouter.ai/openrouter/optimus-alpha) has a new model called **Optimus Alpha** that they've heard is better.
   - Others mentioned that it *looks better* compared to existing models.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1359997981427761365)** (6 messages): 

> `New Memory rollout, Context window in conversations, GPT-4o token limit, Memory storage, Free-tier availability` 


- **GPT-4o's Context Window and Token Limit Explored**: Users discussed the **context window of GPT-4o**, noting that on Plus, it is **32k tokens**, and when surpassed, it may use a **dynamic RAG approach** or start hallucinating.
   - One user claimed that even on Pro the limit is **128,000 tokens**, but it started forgetting earlier parts of the conversation much sooner than expected.
- **Community Clarifies OpenAI Engineer Availability**: A user inquired about the availability of **OpenAI engineers** to answer questions about the new Memory rollout, asking about how it affects context window in conversations and token limits.
   - Another user responded that *nobody here but us users. its the official discord but getting an actual openai person is rare sadly*.
- **Strategies to Mitigate GPT-4o Hallucinations**: When a user inquired how long one can converse with **GPT-4o** before it hallucinates, a member suggested that when noticing signs of hallucinations, repeating itself, not following instructions, etc., it's best to start a new chat.
   - They also proposed to *ask the model to give you a summary of the main points talked about and have that as part of the prompt in the new chat. Or rely on the new chat-based memory feature of ChatGPT if it rolled out to you.*


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1359986113124765809)** (57 messages🔥🔥): 

> `Prompt engineering resources, Model-specific quirks, MusicGPT creation help, Copyright and ToS risks` 


- **Prompt Engineering Resources Emerge**: A member asked for reliable resources on prompt engineering, focusing on news and techniques, and another member responded by emphasizing understanding **what you want the AI to provide** and explaining it clearly to the model.
   - They also mentioned the importance of **verifying the output** and being aware of **model-specific quirks** and company policies.
- **Model Quirks Require Hands-On Experience**: A member suggested that understanding model-specific quirks requires **experiencing different models** and creating hierarchically structured prompts to observe how each model processes them.
   - They noted that this approach teaches *model intuition*, which is organic and qualitative, requiring continuous prompting.
- **MusicGPT Prompt Stumbles into Policy Quagmire**: A member requested a prompt for a *MusicGPT* to assist with music-related requests and provide links from **Genius.com**, leading to discussions about the channel's focus on *how to* rather than providing prompts.
   - The discussion pivoted to the complexities of using external websites and the importance of understanding **ToS and usage policies** to avoid account deactivations.
- **Copyright Concerns Arise During Prompt Creation**: A member raised concerns about **copyright** when using external websites and IP, cautioning about the risks of breaking policies, while another argued that simply linking to public information isn't a deep issue.
   - This led to a clarification of the user's intent for a **music reaction assistant** and a discussion about whether using **APIs** is necessary.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1359986113124765809)** (57 messages🔥🔥): 

> `Prompt Engineering Resources, MusicGPT Customization, API Usage for MusicGPT, Policy Compliance for ChatGPT Use` 


- **Prompt Engineering Core Discussed**: Members discussed the core of prompt engineering, emphasizing understanding what you want the AI to do and communicating it clearly, avoiding typos and grammar mistakes, and verifying the output.
   - The discussion included adapting to model-specific quirks and providing feedback to guide the model's behavior, as well as checking [ToS](https://openai.com/policies) to avoid account issues.
- **Crafting MusicGPT Assistant**: A user requested help in creating a MusicGPT assistant for music-related queries, seeking a prompt to provide online sources from Genius.
   - Suggestions included starting with a markdown outline, using ChatGPT for prompting, and exploring existing Music APIs, but a member was wary of complex API usage and policy compliance.
- **Diagnostic Prompting for model nuances**: Members suggest asking the model to explain prompts or concepts to understand its interpretation and identify ambiguities or conflicts with its programming and safety training.
   - This diagnostic method helps refine prompts and ensure that the model understands and responds as intended, useful for creative exploration or API implementation.
- **Policy Perils plague Public Prompts**: Members cautioned about policy compliance, emphasizing the need to respect copyright and usage policies when asking the model to use external websites and IP songs.
   - Ignoring these policies risks account deactivation, especially when creating tools that interact with others' content.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1359996654014435371)** (119 messages🔥🔥): 

> `Prompt Preprocessor in LM Studio, HuggingFace Login in LM Studio, Image Generation with Gemma 3, Quantization and QAT, Loading Models with Specified Context Limit` 


- **Prompt Preprocessor: Secret Sauce Under Wraps**: A user inquired about the **Prompt Preprocessor** in LM Studio, written in Typescript, and whether an exit code of 1 indicated its unavailability.
   - A team member responded that it's a *secret feature not yet released*, telling the user, *you haven't seen anything*.
- **Hugging Face Login: Mission Impossible**: A user asked about logging into **Hugging Face** within LM Studio, noting a lack of documentation.
   - Another user responded bluntly, *You can't.*
- **Gemma 3's Image Generation: Hallucination Station**: Users discovered that **Gemma 3** cannot generate images, despite claims it can, and instead produces fake Imgur links.
   - As clarified, **Gemma 3** can only read images, not generate them, with Google's Gemini 2.0 Flash experimental and 2.5 Pro potentially having image generation capabilities.
- **QAT: Quantization's Quirky Cousin**: A user asked if **QAT** is a magical way to decrease RAM use.
   - The response clarified that **quantization** is the primary method for decreasing RAM usage, while **QAT** is a training method to improve model performance in quantized form.
- **Gemini-Pro Bill Shock: Google's Gotcha!**: A user experienced a bill shock after using the **Gemini-Pro-2.5-exp** model, which led them to switch to **Gemini-Pro-2.5-preview** without realizing it incurred charges.
   - The user noted that the large 625k context window cost them **$150**, while **Sonnet** would have been much cheaper with caching.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1360004589734330579)** (115 messages🔥🔥): 

> `MLX distributor fix, M3 Ultra Value, Nvidia DGX Motherboard, Deepseek R1 Token Generation, M1 Ultra vs M4 Max` 


- **MLX Fixes Random Model Loading**: A user mentioned that a developer fixed **MLX distributor loading** models in a random manner, suggesting the developer is a *tinkerer* with significant resources.
   - This was followed by a discussion on the high cost of the developer's setup, including a cluster of **Max chips** and an **M3 Ultra** with **512GB of RAM**.
- **M3 Ultra Questionable Performance**: A user shared a controversial opinion that **M3 Ultras** are not worth the cost for professional ML and LLM work, citing preliminary tests showing only **10-13 tokens per second** on **Deepseek r1 67B Q8** and **Q6** models using **MLX**.
   - They argued that a server with **two Xeon Golds** and **1TB RAM** provides better performance at a lower cost, questioning the scalability of **M3 Ultras** for production deployments.
- **Nvidia DGX Pricing Speculation**: Speculation arose regarding the cost of the new **Nvidia DGX motherboard**, which features approximately **280GB of VRAM** and slots for additional GPUs.
   - The consensus was that Nvidia might price it around **$50,000**, but it could potentially offer a cheaper way to run large models compared to current setups.
- **Apple Silicon's Future Potential**: A user speculates that shipping **Apple Silicon** implementations predate open models and local inference, so we may not see what they're truly capable of until the **M5** and especially the **M6**.
   - According to them, Apple figured out the machine learning market gap right after they killed the **M4 Ultra** nearly two years ago, and that's how long it takes to turn around silicon design ship.
- **Exllama Boosts Token Speed**: A user reported testing **Exllama** with **exl2** on **Linux**, achieving about a **50% increase** in token/s compared to using **gguf**.
   - This suggests that the choice of software and parameters can significantly impact performance, especially regarding memory retrieval time.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1359966622538006529)** (76 messages🔥🔥): 

> `Memory in Context, Meta dodged AI week, OSS releases, AI Safety Community, New Image Model` 


- **RAG vs Storing Memory in Context**: Members discussed whether the 'memory thing' is just **RAG** (Retrieval-Augmented Generation) on history or something more, with one suggesting that user-specific context is stored and compressed, seeing an original version that stores biographical data but not pulling in past conversations.
   - Another member said, *I still cant believe meta own goal dodged one of the quietest ai weeks of the year*.
- **New Image Model Released with MIT License**: A new image model with an **MIT license** was released, along with a new **Moonshoot model**, though it may violate Llama's terms.
   - One member provided a link to a post on X about this new image model. ([post link](https://x.com/btibor91/status/1910599925286777027))
- **AI Safety Community Framing Criticized**: A member criticized the framing of an article that stated *some dangerous capabilities were only discovered two months into testing* of **GPT-4**, arguing that despite more powerful open weights models being available for two years, nothing drastic has occurred.
   - They linked to a post on X expressing a similar sentiment ([post link](https://x.com/AtaeiMe/status/1910601934228029515)).
- **Advanced Models for Cyber Defense Debated**: Members debated whether models should be capable of doing **CTFs**, finding bugs, and hacking systems, with one arguing that this would make the world *more* safe, not less.
   - Others noted that it also increases the surface area of attacks, but the defense side is the bigger market, and you can run those models before you deploy the updates in the future.
- **InternVL-3 Multimodal Model Released**: The **OpenGVLab** released **InternVL-3**, a multimodal model combining InternViT and Qwen, achieving impressive results, linking to a non-functional paper describing their training approach.
   - It appears to be using the Qwen license, with the normal one being okay-ish, like llama but permits more, and one member posting that *NVDA has been cooking a lot of cool shit under open licenses lately*.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1360282537859088607)** (2 messages): 

> `Ex-OpenAI Staff Amicus Brief, Peter Wildeford post` 


- **Ex-OpenAI Staff File Amicus Brief**: A [TechCrunch article](https://techcrunch.com/2025/04/11/ex-openai-staff-file-amicus-brief-opposing-the-companys-for-profit-transition/) reports that **ex-OpenAI staff** filed an **amicus brief** opposing the company's transition to a for-profit model.
- **Peter Wildeford tweet surfaces**: A member shared a link to [Peter Wildeford's post](https://x.com/peterwildeford/status/1910718882655981619).


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1359994999848042717)** (117 messages🔥🔥): 

> `Claude Credits Cost, High Taste LMSYS, Gemini App Usability, Tool Use Open Model, MCP Tool Calls` 


- **Claude Credits Price Spike**: A user joked about the increased cost of using **Claude credits**, implying it would cost *$40* to change a variable name.
   - The user attached an image, seemingly mocking the price increase and hinting at the need for more cost-effective solutions.
- **LLM Duel of High Taste?**: A member suggested creating a *"high taste lmsys"* that is invite-only, giving free and early model access to select individuals.
   - The idea is to get labs to provide free API credits for batched stats while keeping raw ratings and prompts private, leading to a *"civilized llm battles"*.
- **Gemini App Annoying Users**: Several users found the **Gemini app** annoying to use, with one stating it was hard to steer and often incorrect.
   - They preferred **AI Studio** for its better grounding and free access, with one saying *"AI studio + grounding works much better and it is free lol"*.
- **Tool Use Open Model Specs**: The discussion explored what makes a good tool use open model, suggesting that **evalmaxing** alone isn't sufficient.
   - It was suggested that the model's ability to work with APIs not in the dataset is important, and the ability to write MCP servers was highlighted, despite a lack of existing evals.
- **MCP Tool Calls Integration**: It was mentioned that integrating **MCP tool calls** into the data is crucial for a good function calling model.
   - It's harder for models to handle 10+ tools, and competing against **Gemini 2.5 Pro** in function calling was suggested, given its current poor performance.


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

philpax: https://fixvx.com/typedfemale/status/1910599582226272457
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1360288906133901432)** (7 messages): 

> `Gemini paywall, Cooking AI` 


- **Gemini Paywall Blocks Glimpse of Rare AI Creature**: A member shared a [YouTube link](https://youtu.be/zzXyPGEtseI?si=cTW9fTaN2zBrxpQB) offering *a glimpse of the rare creature*, but noted that access is locked behind the **Gemini paywall**.
   - They asked what the creator is up to, and another member responded that *he's cooking again, for the masses*, distilling core concepts.
- **AI Creator Behind Gemini Paywall Teases New Project**: Discussion emerged around an AI creator's work, currently behind the **Gemini paywall**, prompting curiosity about their latest endeavors.
   - A member indicated the creator is *cooking again* promising distilled core concepts *for the masses*, suggesting a forthcoming project accessible to a wider audience.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1360287997374890164)** (3 messages): 

> `Amy Prbs Threads` 


- **Amy Prbs Posts Three Threads**: Amy Prbs made three posts on X, which were shared in this channel.
   - The links are [post 1](https://x.com/AmyPrb/status/1910356664403820552), [post 2](https://x.com/AmyPrb/status/1910357180517175620), and [post 3](https://x.com/AmyPrb/status/1910359272279494845).
- **Second topic to satisfy minItems**: This is a placeholder topic to ensure the `topicSummaries` array has at least two elements.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1360144839022215218)** (2 messages): 

> `Gemini 2.5 Pro, API Overview, Grok 3, Perplexity Pro` 


- ****Gemini 2.5 Pro** Rolls Out to Pro Users**: **Gemini 2.5 Pro** is now available on Perplexity for all Pro users and can be paired with Pro Search.
   - Users are encouraged to share feedback in the specified channel regarding its performance compared to **Sonar**, **4o**, **Sonnet 3.7**, **R1**, and **o3**.
- **Perplexity Previews **Grok 3** Integration**: Perplexity announced that support for **Grok 3** is coming soon to Perplexity Pro.
   - The announcement was made on [X](https://x.com/AravSrinivas/status/1910444644892327996) by Aravind Srinivas who encouraged users to let them know what they think.
- **Deep Dive into Perplexity APIs**: Perplexity co-founder & CTO @denisyarats hosted an overview of Perplexity's APIs on April 24 at 11am PT.
   - New API users who register will get **$50** in free API credits via [this link](https://pplx.ai/api-overview).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1359970633517109280)** (194 messages🔥🔥): 

> `Gemini 2.5 Pro, Deep Research, Telegram Bot Official, Firebase Studio AI Builder, Perplexity Android App Security` 


- **Gemini 2.5 Pro in PPLX vs Native Apps**: Members discussed the context and performance of **Gemini 2.5 Pro**, noting that the [native Gemini app](https://gemini.google.com) is generally better than **Perplexity** for most models due to the constraints of web search integration.
   - One user stated, *Native will almost always be better for most models I believe* while another suggested [Google AI Studio](https://ai.google.dev/) has a better UI, video/audio uploads, and settings.
- **Deep Research Update Delay Blamed on High Costs**: Users compared **Perplexity's Deep Research** feature to **ChatGPT**, highlighting **ChatGPT** as better overall but more expensive to operate.
   - One member speculates the removal of **GPT-4.5** from **Perplexity** was due to high costs, suggesting a potential target of **Grok 3 Deep Search** to achieve a balance between performance and cost.
- **Google Eyes Cursor with Firebase Studio**: Discussion revolved around Google's new venture, [Firebase Studio](https://www.bleepingcomputer.com/news/google/google-takes-on-cursor-with-firebase-studio-its-ai-builder-for-vibe-coding/), an **AI builder** for coding.
   - There was speculation that Google might acquire projects like Firebase Studio, leveraging its financial power, with one user jokingly suggesting, *the devs could be google themselves and google just showing its money power in the media buying its own projects*.
- **Perplexity Android App has Security Bugs**: A user shared a [Dark Reading article](https://www.darkreading.com/application-security/11-bugs-found-perplexity-chatbots-android-app) detailing **11 security vulnerabilities** in **Perplexity's Android app**, including hardcoded secrets and insecure network configurations.
   - Another user noted that *half of those vulnerabilities sound not even relevant to the app*, to which another responded with an explainer for what each vulnerability meant, as well as confirmation the report was legit.
- **Pro Role Glitch**: Users discussed issues with obtaining the **Pro User Discord role** after subscribing, noting the need to leave and rejoin the server via a link in **Perplexity** settings.
   - Some members reported failures even after following the prescribed steps, needing moderator assistance to obtain the Pro role.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1360098612268241080)** (1 messages): 

> `Republican voters, Perplexity AI Search` 


- **Republican Voters' Views Explored on Perplexity**: A member shared a [Perplexity AI Search](https://www.perplexity.ai/search/what-are-republican-voters-thi-dcNs8jo4RwWQ87LmonBRBA#0) query regarding *what are republican voters*.
   - No additional context or discussion was provided following the link.
- **Limited Context Follows Search Query**: The shared [Perplexity AI Search](https://www.perplexity.ai/search/what-are-republican-voters-thi-dcNs8jo4RwWQ87LmonBRBA#0) link about Republican voters received no further commentary.
   - The discussion ended with the link, lacking deeper analysis or engagement.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1360327976281047131)** (5 messages): 

> `CUDA in Python/PyTorch models, GTC talk on CUDA, Custom ops and load inline` 


- **CUDA in Python/PyTorch models deep dive**: A member asked for good references on using **CUDA** within **Python/PyTorch** models.
   - Another member shared a link to their recent **GTC talk** about this topic, available at [Google Slides](https://docs.google.com/presentation/d/1zusmhgYjBxSOJPJ-QVeTVJSlrMbhfpKN_q4eDB9sHxo/edit).
- **Custom Ops and Load Inline to solve problems**: A member suggested that **custom ops** and **load inline** should solve most problems when using CUDA.
   - They added that they're working on further improvements, specifically on **compilation time reduction**.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1360115434246115459)** (4 messages): 

> `Triton beginner resources, FP8 support on AMD GPUs, Austin Meetup` 


- **GPU Programming for Triton Newbies**: A member with a SWE background asked the community for resources to get started with GPU programming in Triton, recommending the official [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html).
- **AMD GPUs Fail FP8 Dot Product?**: A member reported an `LLVM ERROR: No match found in MFMA database` error, asking if Triton doesn't support FP8 x FP8 -> FP32 `tl.dot` on AMD GPUs with e4.
   - No response was given.
- **Austin Triton Heads Meet Up!**: The Triton community is invited to an Austin area Meetup on April 30, with registration available at [https://meetu.ps/e/NYlm0/qrnF8/i](https://meetu.ps/e/NYlm0/qrnF8/i).


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1360076825958678723)** (4 messages): 

> `AOT Inductor, Libtorch C++, Torch.compile` 


- **AOT Inductor cannot optimize for training**: A user inquired whether they could utilize **AOT Inductor** to optimize a Python model for training purposes and subsequently load it in C++.
   - Another member clarified that **AOT Inductor** is not suitable for training and suggested using `torch.compile` instead.
- **Torch.compile alternatives**: A user asked about alternatives to `torch.compile` in scenarios where the model is loaded in **Libtorch C++** with **Torchscript** and training is performed there.
   - It was implied that `torch.compile` might not be applicable in that particular setup.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1360005479824363701)** (2 messages): 

> `AlexNet Source Code` 


- **Blast from the Past: AlexNet resurfaces**: The original **AlexNet source code** from 2012 has been unearthed and is now available on [GitHub](https://github.com/computerhistory/AlexNet-Source-Code).
   - Members are experiencing a wave of nostalgia, with one responding with an *"X3"* gif.
- **Unearthing Deep Learning History**: The availability of the **AlexNet source code** provides a valuable resource for understanding the architecture that kickstarted the deep learning revolution.
   - It allows researchers and enthusiasts to examine the original implementation and learn from the techniques used in the groundbreaking **2012** paper.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1360310105433837759)** (2 messages): 

> `Thunder Compute, GPU virtualization, C++ distributed systems engineer` 


- **Thunder Compute Seeks C++ Engineer**: Thunder Compute, a **YC-backed startup**, is hiring a **C++ distributed systems engineer** to enhance its API-layer GPU virtualization software.
   - The role involves applying theoretical knowledge of **GPU programming** and **distributed systems** to achieve microsecond-level performance gains.
- **Apply to Thunder Compute**: Those with the skills required are encouraged to apply.
   - Email carl.peterson@thundercompute.com.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1359970753667272726)** (55 messages🔥🔥): 

> `A100 FP32 core limitations, NCU assembly view for warp stalls, FADD instruction latency, Citadel microarchitecture papers, Microbenchmarking` 


- **A100's 64FP32 Cores Limit Parallelism**: An A100 has only **64 FP32 cores** for 4WS, limiting the number of parallel floating-point additions that can be performed, [impacting performance](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/).
- **NCU Assembly View Reveals Warp Stall Culprits**: The **NCU assembly view** can be used to identify **warp stalls** at specific SASS instructions, providing insights into performance bottlenecks.
   - As one member stated, *looking for warp stalls at a given sass instruction, that should tell you decently well what's going on.*
- **FADD Instructions Stall Due to Dependency Chains**: Each **FADD** in a thread/warp must wait for the previous one to finish due to loop-carried dependencies.
   - This dependency chain causes a single warp per WS to be unable to issue an instruction every cycle, resulting in lower hardware utilization.
- **Citadel's Volta Paper Still A Gold Standard**: Citadel's paper *Dissecting the Nvidia Volta GPU via Microbenchmarking* ([Volta paper](https://arxiv.org/pdf/1804.06826)) is considered superior to later, similar papers.
   - Members agreed that *the later copycat papers don't reach the quality of the volta/turing ones*.
- **Microbenchmarking Reveals Instruction Latency**: Microbenchmarking is useful for determining the number of cycles an instruction takes, as well as how dependency affects instruction clock cycle latency, with single precision add instruction showing 4 and 2 cycles for dependent and independent executions.
   - A relevant StackOverflow Q&A provides [additional context](https://stackoverflow.com/q/79261161/10107454) regarding this topic.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1359981016936353852)** (41 messages🔥): 

> `ROCm Profilers, MI300 vs H100, Runpod Clock Speeds, Runpod Profiling Issues, GPU Cloud Providers` 


- **ROCm Compute & Systems Mimic Nsight**: Members mentioned **ROCm Compute** and **ROCm Systems** as analogous to Nsight profilers, utilizing `rocprof` for profiling, and that visualization options are available.
   - One user noted that these tools performed no better than `rocblas` when working with **ROCm 6.2** on **MI300X**, specifically for the nt layout with `ominperf`.
- **MI300X struggles with memory bandwidth against H100**: A user found that while **MI300** is faster on paper, **H100** is faster in practice unless purely benchmarking transfer speed, with MI300 only reaching about **75%** of theoretical bandwidth.
   - The user also found it odd that memory bandwidth would be a limiting factor for **fp16 gemm**.
- **Runpod Instances Throttled**: A user found that Runpod instances are set to the lowest clock speed and cannot be changed using `rocm-smi`, leading to suboptimal performance.
   - Another user confirmed that Runpod clock speeds are highly variable, effectively calling it *a scam*.
- **Runpod Blocks GPU Profiling**: Users reported that Runpod instances don't allow profiling, even on NVIDIA GPUs, with any GPU-related command giving an `ERROR: GPU[0] : Unable to set ....` message.
   - One user suggested checking the kernel interface to force performance levels but doubted that Runpod would allow it.
- **User Asks for Recommended Cloud Providers that Allow Profiling**: After discovering that Runpod limits GPU clock speeds and blocks profiling, a user asked for recommendations for other cloud providers that offer AMD GPUs and allow profiling.
   - No specific providers were recommended in the available conversation.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1360262697882615808)** (8 messages🔥): 

> `MI300X support, vLLM, SGLang, GemLite, AMD` 


- **MI300X support hits inference engines**: Members discussed support for **AMD's MI300X** with common inference engines like **vLLM** and **SGLang**.
   - One member is *nibbling with AMD* and mentioned that **GemLite** works with vLLM but needs testing, linking to [mobicham's tweet](https://x.com/mobicham/status/1910703643264774377).
- **FP8 E4 Not Supported in Triton**: A user noted that **fp8e4** is not supported in the release version of **Triton**, but **fp8e5** is.
   - This could pose a problem for certain applications.
- **VLLM Works with MI210**: A member confirmed using **vLLM** with **MI210** at work, suggesting **MI300** should also work.
   - They clarified that it requires compiling yourself, but it wasn't too difficult.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 messages): 

felix456: anyone know any cheap / free alternative solutions to using openai API websearch?
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1360179121761685725)** (11 messages🔥): 

> `vectoradd, vectorsum, Modal runners, GPU Benchmarks, Leaderboard submissions` 


- **Vector Addition Benchmarks Abound**: Multiple benchmark submissions for the `vectoradd` leaderboard were successful on various GPUs, including **L4**, **H100**, **A100**, and **T4**, utilizing **Modal runners**.
- **Vector Sum Trials Succeed**: A benchmark submission for the `vectorsum` leaderboard on **L4** GPUs using **Modal runners** was successfully completed.
- **Modal Runners Prove Reliable**: The success of all submissions indicates the reliability of **Modal runners** for benchmarking and leaderboard submissions across different GPU configurations.
   - Each submission is assigned a unique ID, such as **3577**, for tracking purposes.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1360181140534726737)** (22 messages🔥): 

> `MI300 Profiling, Kernel Development details, Team formation, Github link` 


- **MI300 Profiling Promised for Submission Platform**: AMD is planning to provide a **profiling option** for the submission platform, according to the AMD team.
   - A team member stated that they promised help with it but are *not sure if we make it for launch day, but hopefully soon after profiling will be able to be done through discord/CLI*.
- **Registration encouraged, kernel development is not enforced**: It was stated that people should register as soon as possible.
   - A member noted that *there’s no requirement to submit so just do it anyway*.
- **Kernel Development Details Emerge**: During the registration process, the form asks about 'Kernel Development' which is a placeholder.
   - It was said there's *no issue in just giving a placeholder* if you are unsure.
- **Github Link Guidance**: Participants were asking what to put as a GitHub link on the submission form.
   - The recommendation was to create an empty GitHub repo for this, but if you don’t know where you will submit code yet, in the end you can just push to another remote which you put in.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1359966607492776036)** (154 messages🔥🔥): 

> `MCP, Gemini API, Cursor bugs, Deepseek v3.1, usage based pricing` 


- **Cursor's Usage-Based Pricing Clarified**: When enabling usage-based pricing, users can continue using **fast requests** beyond their plan's included amount, but will be switched to **slow requests** upon hitting their spending limit.
   - A member confirmed the understanding of Cursor's usage-based pricing, expressing gratitude for the clarification.
- **DeepSeek v3.1 Judged in Real-World Use**: A member shared that **DeepSeek v3.1** feels *a bit smarter* than **v3** in real-world usage, despite benchmarks often overstating model capabilities.
   - They stated that real-world usage is a better gauge of a model's performance than benchmarks.
- **Gemini API Keys Have Intermittent Downtime**: Some users reported experiencing continuous **404 errors** with **Gemini API keys**, while others reported that Gemini is working for them.
   - One user mentioned that they have been experiencing the issue for the past hour.
- **PDF Reading: MCP server needed for PDF Reading in Cursor**: Members discussed the ability to add PDFs into the IDE, stating that **MCP** is required for reading PDF files in Cursor because *llms cant read pdfs yet*
   - One member stated that there should be many **'convert-shit-to-markdown' MCP** solutions available.
- **Users report a bug where Cursor enters summary mode when context limit is reached**: Users report that when overloading a single chat window (constantly switching between Claude 3.7, Gemini 2.5, then trying Claude 3.5), the agent eventually enters summary mode.
   - The chat will automatically summarizes, and clicking 'New Chat' overwrites an existing tab with the summary.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1359974573214269461)** (83 messages🔥🔥): 

> `Schrödinger Bridges, DeepCoder 14B, KV Cache Distillation, AlphaProof, Math AIs` 


- **Schrödinger Bridges Extended via IPM**: Recent work extends **Schrödinger Bridges** through **Riemannian and integral probability metrics (IPM)**, but the explicit, entropy-based nature may be less popular than *implicit diffusion models* like **Stable Diffusion**.
   - Their path-based approach may be useful in videos, molecular dynamics, and time-series analysis for a *global view*.
- **DeepCoder 14B Open Sourced for Code**: Agentica and Together AI released [DeepCoder-14B-Preview](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51), a **code reasoning model** fine-tuned from **Deepseek-R1-Distilled-Qwen-14B** using distributed reinforcement learning (RL).
   - It achieves **60.6% Pass@1 accuracy on LiveCodeBench**, matching the performance of **o3-mini-2025-01-031** with only 14 billion parameters.
- **KV Cache distillation is likely Impractical**: It's been proposed that a cheaper, faster model can be distilled on the KV values of the main LLM to preprocess prompts.
   - However, this is considered *likely impractical* since **KV values are very model specific** and smaller models will use less transformer blocks.
- **AlphaProof is using RL for Mathematics**: It was mentioned that [AlphaProof](https://www.youtube.com/watch?v=zzXyPGEtseI) is using **RL with Lean** for proving mathematics.
   - Members discussed the potential of AlphaProof to discover novel mathematical solutions.
- **Generalist Agents more practical than Super-Genius Math AIs**: It's been questioned if *super-genius math AIs* are as beneficial as **practical generalist agents**.
   - Concerns were raised about creating *hyper-autistic LLMs that are great at math but suck at everything else*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1359992901224108132)** (4 messages): 

> `AWS Site Visit, nanotron/ultrascale-playbook` 


- **AWS Site Visit Coming Up**: A member announced their class has an **AWS site visit** coming up.
   - They linked to the [nanotron/ultrascale-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) that they'll be going over.
- **Ultrascale Playbook Links Shared**: Three links to the **Ultrascale Playbook** on beautiful.ai were shared: [link 1](https://www.beautiful.ai/player/-ON_kt4FnaDoTXWZPiNY), [link 2](https://www.beautiful.ai/player/-ON_kwmBmoDct78l5GKJ), and [link 3](https://www.beautiful.ai/player/-ON_kzs-T3R8Tbp-DYbM).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1360012570836074496)** (4 messages): 

> `Awkward Youtuber, rQJmDWB9Zwk, 6nJZopACRuQ` 


- **Questionable Screenshot Prompts YouTube Rabbit Hole**: A member posted a screenshot alongside two YouTube links: [rQJmDWB9Zwk](https://youtu.be/rQJmDWB9Zwk) and [6nJZopACRuQ](https://www.youtube.com/watch?v=6nJZopACRuQ).
   - It is implied that one should believe the screenshot.
- **Youtuber Looks Awkward On The Right**: A member commented that a person *on the right looks so awkward like he does not want to be there*.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1359970147565306149)** (66 messages🔥🔥): 

> `Enact Protocol, Semantic Tool Calling, A2A podcast, MCP sandboxing, MCP client integration` 


- ****Enact Protocol** Debated Amidst **A2A** Emergence!**: Members discussed the potential of the [Enact Protocol](https://github.com/EnactProtocol/encat-spec-and-tools) and whether **A2A** makes it obsolete, suggesting **Enact** competes more with code interpreters than with **A2A**.
   - Some proposed **Enact** could benefit from an integrated agent framework with openapi converters and semantic search.
- ****Semantic Tool Calling** Poised to Revolutionize LLM Efficiency**: The discussion highlighted **semantic tool calling** as a solution to the context overload caused by providing **LLMs** with hundreds of tools, using vector models to select a subset of tools based on semantic similarity to the task.
   - This approach enables the application of traditional **ML** methods for tool analysis, such as detecting similar tools via clustering and grouping tools for reranking.
- **Podcast Released on **A2A**, **MCP**, and Agent Indexing**: A member shared a [podcast episode](https://youtu.be/5hKhNJUncKw?si=lXm3f4x_69jypxYr) discussing **A2A** implications, potential indexing of agents by **Google**, and other related topics, pointing out its relevance to the current discussions.
   - The podcast aims to be high-level and accessible, stimulating ideas beyond the typical technical discussions.
- **Challenges integrating **Express servers** with **MCP****: A member wants to connect their **Express server with REST routes** to **Claude desktop** via **MCP** and asks if that's possible.
   - A member responded that it's necessary to use the **MCP JSON-RPC spec** for integration.
- **Github not picking up **Licenses****: A user had issues with **Github** not picking up the license file in their [repo](https://github.com/Vizioz/Teamwork-MCP), so the glama server showed "license - not found".
   - The user fixed it by moving the license disclaimer to another file, so that **Github** could properly detect the license.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1360057888416075878)** (5 messages): 

> `MCP Protocol Validator Open Source, MCP Server Adoption Challenges, Cloud Hosted MCP Inspector, MatlabMCP - MATLAB Meets LLMs` 


- ****MCP Validator** Open-Sourced for **Implementation** Harmony**: The **MCP Protocol Validator** has been open-sourced to bridge the gap between various MCP server implementations by providing a comprehensive test suite, available at [GitHub](https://github.com/Janix-ai/mcp-protocol-validator).
   - The tool helps ensure implementations meet requirements for both **2024-11-05** and **2025-03-26 MCP versions**, and includes reference implementations for **HTTP** and **STDIO** transports developed at **Janix.ai**.
- **Cloud Inspector Chats with Your Servers**: A cloud-hosted **MCP Inspector** has been launched to test **SSE** & **Streamable HTTP servers** without needing local setup, accessible at [inspect.mcp.garden](https://inspect.mcp.garden).
   - The platform also includes full chat support, allowing users to interact directly with their remote **MCP servers**; see the announcement [on X](https://x.com/ChrisLally/status/1910346662297452896).
- ****MatlabMCP** Connects **MATLAB** with **LLMs****: **MatlabMCP**, a mini **MCP** connecting **MATLAB** with **LLMs**, was showcased, and can handle smaller code snippets effectively, available at [GitNew](https://git.new/MatlabMCP).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1360056623459991592)** (31 messages🔥): 

> `Discord referral, Dyslexia, KL vs CE, Model size` 


- **User finds Discord via Max's AI**: A new user found this Discord server based on a recommendation from their friend's **GPT4.o model**.
- **Dyslexia strikes user**: A user apologized for their dyslexia after a typo, saying *just checking maet*.
- **KL vs CE in Token Prediction**: A user reported a repetition issue in their model, and another user suggested adding **CE** to the **KL** loss, but then suggested that if the data is geometric this will be a *waste of time* and to stick to **KL**.
- **Model Size Debated**: A user was concerned about their model size, to which another user replied *200M is enough for this. your problem is elsewhere*, but then cautioned that *200M can overfit to 16k samples quite easily btw*.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1359982624856215854)** (32 messages🔥): 

> `Lambada Parity, RWKV vs Transformers, UTs and RWKV, Muon and Transformer Layers` 


- **RWKV Achieves Lambada Parity!**: The RWKV architecture has achieved parity on the **Lambada** dataset, matching the performance of the model it was distilled from, **Qwen2.5-7B-Instruct**, though MMLU performance is lower.
   - The speaker noted that the parity was *within statistical error range*.
- **Debate on RWKV Expressiveness for UTs**: Members discussed whether the expressiveness of **RWKV** models makes them suitable for **Universal Transformers (UTs)**.
   - One member stated that *just because RWKV layers tend to be more expressive doesn't imply they'd be better for UTs*, also noting that expressiveness might be worse.
- **Insight on Scaling Transformer Linear Layers with Muon**: A member observed that adding a zero-initialized learnable per-channel scale on the last linear layer of each block in a transformer (option A) leads to slower growth of the main path activation RMS compared to zero-initializing the weight matrix of the last layer (option B).
   - This observation was made using the **Muon** library.
- **RWKV-7 Paper Highlights**: A member shared an image from the **RWKV-7 paper**, which was considered a good choice to send to the UT person.
   - It was explained that the model's mathematical guarantees allows extending results from smooth to nonsmooth systems depending on your application scope.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1360023449317412904)** (2 messages): 

> `GPTs Agents, String Matching` 


- **String Matching Frustrates User**: A member expressed disappointment upon learning that **GPTs agents** primarily use string matching over the full dataset.
   - They had hoped for more sophisticated learning or adaptation mechanisms beyond simple **string matching**.
- **String Matching Under Scrutiny**: The conversation highlights concerns about the limitations of relying solely on string matching for **GPTs agents**.
   - This approach may not capture the nuances and complexities that more advanced techniques could offer, leading to potential performance bottlenecks.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1359967054706245765)** (55 messages🔥🔥): 

> `SIMD store, bench functions incorrect use, `@parameter` needed, lock files, random integers list` 


- **SIMD Store Needs Special Treatment**: When using **SIMD** with tensors, members clarified that you need to use the [`store`](https://docs.modular.com/max/api/mojo/tensor/tensor/Tensor/#storeSIMD) member function instead of directly assigning values via `__setitem__`.
   - This is because stores have to be treated differently than scalar ones.
- **Benchmarking Functions Require `@parameter`**: A user ran into an error message *cannot use a dynamic value in call parameter* when using `benchmark.bench_function`.
   - It was [clarified](https://github.com/modular/max/pull/4317#issuecomment-2795531326) that functions passed into `benchmark.run` need the `@parameter` decorator and are expected not to return anything.
- **Magic Init Doesn't Always Create Lock Files**: A user noticed that running `magic init AdventOfCode --format mojoproject` didn't always create a lock file.
   - After running `magic run mojo --version`, the lock file was created.
- **`__rand__` Is for `&` operator, not Random Numbers**: Members clarified that `__rand__` is used for the `&` operator, not for generating random numbers.
   - While Max tensors used to have a `.rand` method ([docs](https://docs.modular.com/stable/max/api/mojo/tensor/tensor/Tensor/#rand)), it has been removed on nightly builds; use methods from the `random` module instead.
- **Tensors Missing Overloaded Operators**: A user questioned why **Tensors** don't have operators like `+`, `-`, and `matmul` overloaded.
   - This sparked a discussion on the design choices and future plans for tensor operations in Mojo.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1360273839032111186)** (4 messages): 

> `Mojo project discrepancies, magic.lock file issues, Mojo version conflicts` 


- ****Mojo Oddities**: Code Works in One Project, Fails in Another**: A member found that a specific code snippet involving `@value struct Foo(StringableRaising)` and `String(foo)` works in one **Mojo** project but throws a *"no matching function in initialization"* error in another.
   - The reported error occurred when trying to convert the custom struct `Foo` to a `String` type.
- ****Magic Lock Fix**: Deleting Resolves the Issue**: The member resolved the error by deleting the `magic.lock` file in the problematic project.
   - This suggests that the issue was likely due to differing **Mojo** versions or dependency conflicts managed by the `magic.lock` file, implying that *"would have been pulling different versions"*.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1359982933733019719)** (48 messages🔥): 

> `L1-Qwen-1.5B-Max model, Nomic embed text v1.5, LLM query logging, System prompts for embedding models, Re-ranker models` 


- **L1-Qwen-1.5B-Max Model Sets Thinking Length**: The [L1-Qwen-1.5B-Max model](https://cmu-l3.github.io/l1/) allows setting the length of thinking, and a member found it to be better and clearer even without prompting for maximum tokens, as explained in the [paper](https://cmu-l3.github.io/l1/ ).
   - The user is going to download the [L1 version from HuggingFace](https://huggingface.co/l3lab/L1-Qwen-1.5B-Max) to use it.
- **Nomic Embed Text Remains King**: Despite trying many generative LLMs, one member continues to use **Nomic** `nomic-embed-text-v1.5-Q8_0.gguf`.
   - Another member asked how to identify which version they have, to which another responded, *google ^^*, and linked [Nomic's HF page](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/tree/main).
- **Archiving LLM Responses Proves Helpful**: A user has been logging **LLM queries and responses** in a database for over a year, finding these past responses valuable for consulting purposes, especially for sales.
   - They created an **Emacs Lisp function** to insert embeddings, referencing a function found [here](https://gnu.support/files/tmp/clipboard-2025-04-11-09-03-07.html).
- **System Prompts with Embeddings Debated**: Members discussed whether **system prompts** are used by default with embedding models like **LM-Studio/ALLM**, with one member suggesting the system prompt from the LLM might not be used.
   - The user confirmed they **don't give any system prompt** to the embedding model and don't have the option to do so.
- **Reranking Models Spark Interest**: A member inquired about how **re-ranker models** work and whether it's only the question asked of the LLM that matters, also referencing a [YouTube video](https://www.youtube.com/watch?v=76EIC_RaDNw&feature=youtu.be) about prefixing.
   - The linked video sparked discussion on prefixing queries with `search_document:CHUNK_OF_TEXT_FOLLOWS` and `search_query:FOLLOWED_BY_QUERY`, but noted that all embeddings must be re-indexed.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1359995523519614996)** (31 messages🔥): 

> `Gradio GUI, Transformer Training Data Volume, Finding Python Expert, Reporting HF Course Errors, Fine Tuning Model` 


- **Gradio is a GUI Library**: A member asked *What's Gradio* and another succinctly replied that it is a **GUI Library** pointing to the [Gradio website](https://www.gradio.app/).
- **Transformer Overtrained With Too Much Data?**: A member is scraping a website with a **million records of watches** and is considering training a transformer to give it a better understanding of context/spec names, asking *Is there such a thing of training your transformer with too much data?*.
   - The member is planning to finetune the model (perhaps **Mistral7B**) so that if someone talks like `Patek 2593 Tiffany stamp dirty dial manual wind`, it understands those words and at what entity it belongs to.
- **Lightning AI Chat Templates Released**: The HuggingFace team has announced [chat templates](https://lightning.ai/chat) on **HF**.
- **Run HF Models Locally on ROCm**: Users wanting to run **0 day Hugging Face models locally on ROCm** may want to check out [this video](https://youtu.be/K4bHgaUk_18).
- **Restart Xet Spaces to Fix Issues**: Users with early access to **Xet** and are facing issues with their spaces should consider restarting them.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1360322773217448229)** (3 messages): 

> `Life's Unexpected Surprises` 


- **Wisdom on Handling Life's Curveballs**: After someone said they did not understand a saying, another explained that *when life hits you with unexpected surprises you will know then*.
   - The first person then responded with *I hope whatever you’re going through gets better soon*.
- **Another topic**: Another summary.
   - Another response.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

not_lain: the app is offline
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1360184619160305786)** (1 messages): 

> `Stanford CME 295 Transformers, LLM Book Discussions` 


- **Stanford CME 295 Transformers Book Shared**: A member shared a link to the [Stanford CME 295 Transformers book](https://github.com/afshinea/stanford-cme-295-transformers-large-language-models/tree/main) and inquired if anyone had explored its contents.
- **Interest in LLM Book Discussions Sparked**: The sharing of the Stanford CME 295 Transformers book link initiated potential discussions around Large Language Models (LLMs) and related educational materials.
   - Members might delve into aspects such as model architectures, training methodologies, or practical applications highlighted in the book.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1360096434698850588)** (6 messages): 

> `Object Tracking, ReID for Object Recognition, Owlv2 Model, Segment Anything Model (SAM), YOLO Model` 


- **ReID Terminology Surfaces**: A member inquired about the term for **object tracking** the same object across camera frames.
   - Another member responded that the term is **ReID**.
- **Owlv2 Model Troubleshoot Begins**: A member reported issues with the **Owlv2 model** for image-guided detection, noting it performed worse than expected with its built-in method and posted a link to the [github tutorial](https://github.com/github).
   - They requested assistance in reconfiguring the class to better suit cropped images as queries.
- **SAM to rescue YOLO?**: A member suggested using the **Segment Anything Model (SAM)** as an alternative approach to **YOLO** for identifying vertical poles, since you can feed it YOLO bounding box outputs.
   - Another member acknowledged using **SAM** for labeling but expressed a need for automation, precluding user interaction for pole selection which could be done through finetuning SAM.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1360106808198562026)** (4 messages): 

> `LangGraph vs Google ADK, Google Agent Development Kit, Meta Llama access` 


- **Google ADK vs. LangGraph: Open Source Stand Off**: Members are comparing the **Google Agent Development Kit** with **LangGraph**, noting Google's fully open source approach versus LangGraph's partially open source model with commercial debugging tools.
   - It was mentioned that **LangGraph** strives for broad LLM compatibility, while the **ADK** is designed for tight integration with the **Google ecosystem**.
- **Meta Llama Access Request Rejected**: A member reported their request to access the **Meta Llama models** was rejected and inquired about retrying.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1359968556049109102)** (14 messages🔥): 

> `vgel's control-vectors, DisTrO details, Psyche's testnet run` 


- **Control-Vectors Augment Models for Targeted Use-Cases**: A member inquired about using **vgel's control-vectors** to augment models for specific use-cases and personas, such as dungeon masters or software engineers, arguing that it could enhance accuracy and control, especially for open-source models like **DeepHermes-Mistral-24B**.
   - In response, another member mentioned that while they have experimented with it, applying control vectors generally has proven unstable, but it's still being explored, noting [a relevant X post](https://x.com/winglian/status/1910430245854773523).
- **DisTrO's Details Remain Tech Report Elusive**: A member inquired about a technical report detailing the **DisTrO** run on [distro.nousresearch.com](https://distro.nousresearch.com/), seeking information on the dataset, number of GPUs/participants, and benchmark details (e.g., number of shots used for the evals).
   - Another member responded that they did not release a tech report, stating the run was primarily to prove **DisTrO** worked over-the-internet, and they did not optimize for the resulting model's quality, performing only a short training session on a limited number of tokens **(100B)**.
- **Psyche's Testnet Run Promises**: In a follow-up to the DisTrO conversation, a member shared details about the distributed training, noting that each node had **8xH100s**, and they had between **8-14 nodes** running, also mentioning that the eval code is available on [GitHub](https://github.com/PsycheFoundation/psyche/tree/main/shared/eval/src).
   - They are working on a **testnet run** for **Psyche**, their distributed training network that takes advantage of **DisTrO**, which will include significant speed & bandwidth improvements and public visibility into the dataset, nodes, and more.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1359972820284866782)** (4 messages): 

> `Azure API, Reasoning Content, Token Limits` 


- **Azure API now working!**: A member reported that the **Azure API** is now working, but they weren't sure why it didn't work earlier.
   - They noted that `<think>` traces are returned in `reasoning_content`, suggesting that *this should be documented, as this is slightly different in every api*.
- **Token Limits Errors Appear in Azure**: A member received a **400 error** when asking for too many tokens in the **Azure API**.
   - They also suggested that the `<think>` tags might only appear when the response is truncated by the token limit, explaining why they got malformed traces.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1360292371883102269)** (3 messages): 

> `X Post, Teknium User Mentions` 


- **X Post shared**: A member shared a link to an X post: [https://x.com/omarsar0/status/1910004370864742757?t=w_ps1fBHQpu3Vfdf1MMV0A&s=19](https://x.com/omarsar0/status/1910004370864742757?t=w_ps1fBHQpu3Vfdf1MMV0A&s=19).
- **Teknium mentions Users**: The user **Teknium** mentioned 2 users.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1360172027213779126)** (6 messages): 

> `Pathways Paper, TPU vs GPU, Tinygrad cloud, Tinygrad virtualization` 


- **Pathways Paper Sparked Discussion**: A member shared the [Pathways paper](https://arxiv.org/pdf/2203.12533) noting that *PATHWAYS uses a client-server architecture that enables PATHWAYS’s runtime to execute programs on system-managed islands of compute on behalf of many clients* and suggested **tinygrad cloud**.
   - The member also pointed out that *tinygrad is single process and will stay that way even for scale-out*.
- **TPU Kernel Richer Than GPU Driver?**: A member quoted the [Pathways paper](https://arxiv.org/pdf/2203.12533) saying *The biggest difference between TPU and GPU is that far longer-running and more complex computations can be fused into a single TPU kernel because the TPU supports rich control flow and communication primitives that must instead be executed by driver code on GPU systems*.
   - The member rebutted, imagining *1024 Navi 48 chips all working together* without a driver.
- **Tinygrad Aims To Virtualize GPUs**: A member read the Pathways paper and summarized that it is fundamentally an **orchestration approach**.
   - They claimed that *it would be more innovative if tinygrad could virtualise the GPU, so that you could guarantee a certain amount of usage*.
- **Tinygrad Termux Issue**: A member asked if another member had managed to run **tinygrad** under **termux** after raising [this issue](https://github.com/tinygrad/tinygrad/issues/9687).
   - The user mentioned that they were also having the same issue where it said *libgcc_s.so.1 not found*.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1360046129223565323)** (14 messages🔥): 

> `Position-Independent Code, ELF Loader, Compiler Linking, TinyGrad Architecture, Memory Map Generation` 


- **TinyGrad leverages Position-Independent Code (PIC)**: The discussion clarified that **TinyGrad** uses **position-independent code (PIC)**, where addresses are relative to the program counter, and addresses to `.data` and `.rodata` sections are patched to account for load-time memory placement.
   - The goal is to combine `.text` and `.data` sections and patch the addresses for the correct offsets of the data sections. *An interesting exercise would be to not have an OS either, just TinyGrad all the way to hardware.*
- **ELF Loader used for shared objects**: The **ELF loader** in TinyGrad is used both for loading shared objects (`.so/.dll`) in AMD/NV and for converting object files (`.o`) from **Clang/LLVM** to flat shellcode.
   - When loading shared objects the offsets to `.data` from `.text` are known and no relocation is needed assuming **PIC**; however, object files (`.o`) need relocation as offsets are filled by the linker.
- **Cloudflare's Blogposts Explain Object File Execution**: A member shared a [blog post series from Cloudflare](https://blog.cloudflare.com/how-to-execute-an-object-file-part-1/) which describes how to execute an object file, similar to TinyGrad's approach.
   - The blog post series explains the process of converting object files to flat shellcode.
- **LLVM Loads from `.data` due to global variables**: The use of **ELF relocations** in the **Clang JIT** is required because **LLVM** sometimes chooses to load from `.data` instead of using immediate values for constants, even though TinyGrad doesn't use global variables.
   - This behavior necessitates patching addresses for correct offsets during the linking process.
- **Why compiler linking is not done in TinyGrad**: Linking during compilation was considered, but the member mentioned that it is avoided because it is slower and there is a bug in Apple's linker that prevents outputting to stdout.
   - Skipping linking step saves a dozen lines in `elf.py`.


  

---


### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1360049247197593810)** (1 messages): 

> `Finetune Llama4, Scout Model, Maverick Model, MoE models` 


- **Llama4 finetuning supported in torchtune**: Support for full finetuning of **Llama4** has landed in torchtune.
   - Configs are available [here](https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama4); stay tuned for LoRA configs, improved multimodal support, and performance improvements.
- **Scout model introduced**: The **Scout** model (**17B x 16E**, **109B** total params) can be finetuned on a single node, or on multiple nodes with **2D parallel** (**TP + FSDP**) support.
   - Members of the *GPU-middle-class* can rejoice.
- **Maverick model introduced**: The **Maverick** model (**17B x 128E**, **~400B parameters**) is available for full finetuning, requiring multiple nodes.
   - These are the first **MoE models**, so experiment and provide feedback.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

jovial_lynx_74856: @here office hours in 43 mins!
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1359966190478430339)** (16 messages🔥): 

> `running_loss.detach() fix, test tolerances, sampler seed, bitsandbytes Mac issues, FSDPModule import error` 


- **running_loss.detach() fix incoming**: A member suggested that using `running_loss.detach()` is an easy fix to an unknown problem, and another member said *ill take it*>.
   - The fix is in the `detach` branch, but don't forget to fix it for other recipes.
- **Test tolerances should be lowered**: A member suggested that when seed is fixed all unit tolerances may be brought down from their current state of +-0.01.
   - A past issue involving loose tolerances in integration tests was mentioned, and a [related pull request](https://github.com/pytorch/torchtune/pull/2367) was linked.
- **bitsandbytes Mac woes**: A member reported that `pip install -e '.[dev]` fails on a mac due to `bitsandbytes>=0.43.0` not shipping binaries for other platforms, and suggested changing to `bitsandbytes>=0.42.0` as a workaround.
   - This workaround references [this issue](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1378#issuecomment-2383530180) which notes that releases up to 0.42 were incorrectly tagged.
- **FSDPModule import error is slowing down testing**: `pytest tests` fails on collecting tests with an `ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'`.
   - The suggestion was to check the installation docs, as the project requires a different installation method, and the team doesn't want to add platform specific requirements at this point.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/)** (1 messages): 

krammnic: I was speaking about something like this
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1360110036415807639)** (18 messages🔥): 

> `FunctionCallingAgent JSON Schema Response, Llama Cloud API 404 Error, FaissVectorStore Index from Weights, Intelligent Metadata Filtering in RAG Agent` 


- **FunctionCallingAgent wants OpenAI's JSON Response**: A member wants to generate a response in a particular **JSON schema** using **FunctionCallingAgent** and inquired about using **OpenAI's structured response** feature.
   - Another member responded that structured outputs are just tool calls, which makes it hard to mix tool calling and structured outputs; they suggested adding a tool that is the response class and setting tool_choice="required".
- **Llama Cloud API Throws 404 Error**: A member encountered a **404 error** while using the **Llama Cloud API** to extract values from documents using the fast mode, with the API URL `https://api.cloud.llamaindex.ai/v1/extract`.
   - It was pointed out that the API endpoint used does not exist and directed the member to the [correct API documentation](https://docs.cloud.llamaindex.ai/llamaextract/getting_started/api) and [API reference](https://docs.cloud.llamaindex.ai/API/create-extraction-agent-api-v-1-extraction-extraction-agents-postsadguru_.)
- **FaissVectorStore Index from Weights Query**: A member was trying to use a **FaissVectorStore** restored from weights to create a **VectorStoreIndex** that they can query.
   - It was pointed out that the [Faiss documentation](https://docs.llamaindex.ai/en/stable/examples/vector_stores/FaissIndexDemo/) shows how to initiate this, although it is in Python, not Typescript.
- **Intelligent Metadata Filtering in RAG Agent is Sought**: A member is trying to build an agent using intelligent metadata filtering on retrieval based on user query.
   - No direct solutions were provided in the snippet, but the member sought advice on implementing this use case within a standard **RAG pipeline** without recreating embeddings at later API calls.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1360003130364002449)** (7 messages): 

> `Microphone recognition issues in NotebookLM, Upload source errors, Phishing attempts` 


- **NotebookLM fails to recognize Microphone**: A user reported that **NotebookLM** doesn't recognize the computer's default microphone in interactive mode, even though the microphone works.
   - Another user suggested checking the **OS** and **browser permissions**, advising to test without external USB devices first.
- **Users get Upload Source Errors**: A user asked about a **red "!" sign** on their upload source in **NotebookLM**, even with a **PDF file** smaller than **500kb**.
   - Another user suggested hovering over the "!" mark, indicating the source might be empty or taking time to load, especially with certain sites.
- **Steam Phishing Attempts Circulate**: A user shared a link appearing to be a **$50 gift** but it is a [phishing link](https://steamconmmunity.cfd/1043941064) redirecting to a fake **Steam Community** site.
   - Users are warned not to click on suspicious links and to verify the URLs of websites asking for login credentials.


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1360218077664378980)** (2 messages): 

> `Vague questions, Specific Queries` 


- **Question Vague-ness reaches New Heights**: A member joked about another's question of *"has anyone ever driven a car"* and recommended they be more specific in their queries.
- **Specificity suggestions spark Humor**: The member asked, *"how can you be more vague?"*, highlighting the absurdity of the initial question.


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1360150375637844039)** (2 messages): 

> `Java API, Network error` 


- **Cohere's Java API throws Network Error**: A member reported encountering a `Network error executing HTTP request` when using the [Java API example](https://docs.cohere.com/reference/about#java).
   - The member confirmed that the error persisted across different prompts, like *recommending quick meals for a beginner chef*.
- **Request for Java API Code Snippet**: A member asked for a code snippet to help debug the `Network error` in the Java API.
   - The member also asked if the user was running the example verbatim.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1360186605733937184)** (2 messages): 

> `DSPy module as a persona, AI Agents & Reasoning, Large Language Models (LLMs), Machine Learning Frameworks, Infrastructure` 


- **Module-Based Personas Spark Excitement**: A member asked about training a **DSPy module** as a **persona**, optimizing the system prompt of an "agent/model", and passing this module as input to other modules to generate content in that persona.
- **Collaboration Invitation Highlights Tools**: A member expressed interest in collaborating, listing expertise in **AI Agents & Reasoning** (**LangChain**, **LangGraph**, **ElizaOS**, **AutoGPT**, **ReAct** frameworks), **Large Language Models** (including **GPT-4.5**, **DeepSeek-R1**, **Claude 3.5**), and **Machine Learning Frameworks** like **PyTorch** and **TensorFlow**.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1360012937644867856)** (2 messages): 

> `Course Deadlines, Certificate Availability` 


- **Course Completion Possible Despite Late Start?**: A student inquired about the possibility of completing the course and obtaining a certificate despite starting after the official start date.
   - Another member responded affirmatively, directing the student to the [course website](https://llmagents-learning.org/sp25) for all necessary materials and deadlines.
- **LLM Agents Course**: A student asked if they could complete the course by the due date and get the certificate.
   - A member confirmed that all materials are available on the [course website](https://llmagents-learning.org/sp25).


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1360310859188142143)** (1 messages): 

> `Event reminder` 


- **Event reminder**: A member reminded everyone that an **event is happening tomorrow**.
   - They expressed hope to see other members there.
- **Event is tomorrow**: This is just a reminder
   - Be there or be square


  

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

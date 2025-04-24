---
id: 080524db-34b6-4c7d-b8e3-fa03f4e2d105
title: not much happened today
date: '2025-03-14T22:57:23.512875Z'
original_slug: ainews-not-much-happened-today-7693
description: >-
  Google DeepMind released Gemini 2.0 with enhanced reasoning, deeper app
  integration, and native image generation capabilities, marking significant
  progress in their AI models. Additionally, Cohere launched Command A, a 111B
  parameter dense model with a 256K context window and competitive performance
  close to OpenAI’s GPT-4o, now available on Hugging Face.
tags:
  - google
  - deepmind
  - gemini-2
  - image-generation
  - cohere
  - command-a
  - large-language-models
---


<!-- buttondown-editor-mode: plaintext -->**A quiet Friday**

> AI News for 3/14/2025-3/15/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **28** Discords (**222** channels, and **2399** messages) for you. Estimated reading time saved (at 200wpm): **240 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Happy 2nd birthday to [GPT4](https://en.wikipedia.org/wiki/GPT-4) and [Claude 1](https://www.anthropic.com/news/introducing-claude). Few would have guessed the [tremendous market share shifts](https://www.latent.space/p/2024-startups) that have happened in the past year.

![image.png](https://assets.buttondown.email/images/f5e3c589-7d40-4495-ad42-3b519c21606b.png?w=960&fit=max)

---

**SPECIAL NOTE**: We are launching [the 2025 State of AI Engineering Survey](https://www.surveymonkey.com/summary/NU9euNHK_2FMmqZLGjDImPimHFO_2FbIYG7s_2Bme46v_2BeQSA_3D?ut_source=lihp) today in preparation for the AI Eng World's Fair in Jun 3-5. **Please [fill it out](https://www.surveymonkey.com/summary/NU9euNHK_2FMmqZLGjDImPimHFO_2FbIYG7s_2Bme46v_2BeQSA_3D?ut_source=lihp) to have your voice heard!**

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Language Models and Model Updates**

- **Google's Gemini 2.0 Updates and Features**: [@jack_w_rae](https://twitter.com/jack_w_rae/status/1900401734046126274) announced improved **Google Deep Research** due to product development and underlying model updating from **1.5 Pro to 2.0 Flash Thinking**. The **Gemini app** is launching improvements, including an upgraded **Flash Thinking model with stronger reasoning**, deeper app integration, **Deep Research**, and **personalization** [@jack_w_rae](https://twitter.com/jack_w_rae/status/1900325293447061877). Additionally, [@jack_w_rae](https://twitter.com/jack_w_rae/status/1900334465945395242) noted the team's progress in creating **native image generation for Gemini 2**, highlighting its difference from text-to-image models.
- **Cohere's Command A Model**: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1900606602501341518) reported that **Cohere** launched **Command A**, a **111B** parameter dense model with an **Artificial Analysis Intelligence Index of 40**, close to **OpenAI’s latest GPT-4o**.  The model has a **256K context window**, a speed of **185 tokens/s**, and is priced at **$2.5/$10 per million input/output tokens**. It is available on **Hugging Face** for research and commercially with a license from **Cohere**.
- **Meta's Dynamic Tanh (DyT)**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1900528108140372411) reported that **Meta AI** proposed **Dynamic Tanh (DyT)** as a replacement for normalization layers in Transformers, which works just as well or better without needing extra calculations or tuning and works for images, language, supervised learning, and self-supervised learning.  **Yann LeCun** also announced the same thing on [Twitter](https://twitter.com/ylecun/status/1900610590315249833).
- **Alibaba's QwQ-32B**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1900351166086537659) highlighted **Alibaba's QwQ-32B**, a **32.5-billion-parameter** language model excelling in math, coding, and problem-solving. Fine-tuned with reinforcement learning, it rivals larger models like **DeepSeek-R1** and outperforms **OpenAI’s o1-mini** on benchmarks.  The model is freely available under the **Apache 2.0 license**.
- **Google's Gemma 3 Models**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1900549631647367268) announced the release of **Gemma 3**, available in sizes from **1B to 27B**, featuring a **128K token context window** and supporting over **140 languages** [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1900549635267014878). It also announced the **ShieldGemma 2**, a **4B image safety checker** built on the **Gemma 3 foundation** [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1900549638802813312). [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1900579291404046696) benchmarked **Gemma 3 27B** with an **Artificial Analysis Intelligence Index of 38**, noting its strengths include a permissive commercial license, vision capability, and memory efficiency, while not being competitive with larger models like **Llama 3.3 70B** or **DeepSeek V3 (671B)**. [@sirbayes](https://twitter.com/sirbayes/status/1900520172059815986) noted that **Gemma 3** is best in class for a **VLM that runs on 1 GPU**.

**Model Performance and Benchmarking**

- **Leaderboard Lore and History**: @_lewtun shared the origin story of the **Hugging Face LLM leaderboard**, highlighting contributions from [@edwardbeeching](https://twitter.com/_lewtun/status/1900557190722687440), [@AiEleuther](https://twitter.com/_lewtun/status/1900557190722687440), [@Thom_Wolf](https://twitter.com/_lewtun/status/1900557190722687440), [@ThomasSimonini](https://twitter.com/_lewtun/status/1900557190722687440), [@natolambert](https://twitter.com/_lewtun/status/1900557190722687440), [@abidlabs](https://twitter.com/_lewtun/status/1900557190722687440), and [@clefourrier](https://twitter.com/_lewtun/status/1900557190722687440). The post emphasizes the impact of **small teams**, **early releases**, and **community involvement**. [@clefourrier](https://twitter.com/clefourrier/status/1900572125238378939) added to this, noting that [@nathanhabib1011](https://twitter.com/clefourrier/status/1900572125238378939) and they were working on an internal evaluation suite when the leaderboard went public, leading to industrializing the code.
- **GPU Benchmarks and CPU Overhead**:  [@dylan522p](https://twitter.com/dylan522p/status/1900379633662779781) expressed their appreciation for **GPU benchmarks** that measure **CPU overhead**, such as **vLLM** and **KernelBench**.
- **Tic-Tac-Toe as a Benchmark**: [@scaling01](https://twitter.com/scaling01/status/1900333236565221400) stated they are a LLM bear until GPT-5 is released, citing that **GPT-4.5** and **o1** can't even play **tic-tac-toe** consistently and [@scaling01](https://twitter.com/scaling01/status/1900352006641848695) argued that if LLMs can't play **tic-tac-toe** despite seeing millions of games, they shouldn't be trusted for research or business tasks.
- **Evaluation Scripts for Reasoning Models**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1900595120053047452) announced a **GitHub repo** providing evaluation scripts for testing the benchmark performance of reasoning models and reproducing reported results for **QwQ**.

**AI Applications and Tools**

- **AI-Assisted Coding and Prototyping**: [@NandoDF](https://twitter.com/NandoDF/status/1900548832733069638) supports the idea that it's a great time to learn to code, as coding is more accessible due to **AI copilots**, potentially leading to a wave of entrepreneurship. This sentiment was echoed by [@AndrewYNg](https://twitter.com/DeepLearningAI/status/1900593192497520842), noting that AI and AI-assisted coding have reduced the cost of prototyping.
- **Agentic AI in IDEs**:  [@TheTuringPost](https://twitter.com/TheTuringPost/status/1900321016385359958) introduced **Qodo Gen 1.0**, an **IDE plugin by @QodoAI** that embeds agentic AI into **JetBrains** and **VS Code**, using **LangGraph by LangChain** and **MCP by Anthropic**.
- **Integration of Gemini 2.0 with OpenAI Agents SDK**: [@_philschmid](https://twitter.com/_philschmid/status/1900589029961109514) announced a one-line code change to use **Gemini 2.0** with the **OpenAI Agents SDK**.
- **LangChain's Long-Term Agentic Memory Course**: [@LangChainAI](https://twitter.com/LangChainAI/status/1900588929772122383) and [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1900562773303554110) announced a new **DeepLearningAI course** on **Long-term Agentic Memory with LangGraph**, taught by [@hwchase17](https://twitter.com/LangChainAI/status/1900588929772122383) and [@AndrewYNg](https://twitter.com/LangChainAI/status/1900588929772122383), focusing on building agents with semantic, episodic, and procedural memory to create a personal email assistant.
- **UnslothAI Updates**: [@danielhanchen](https://twitter.com/danielhanchen/status/1900592202621087944) shared updates for **UnslothAI**, including support for full fine-tuning + 8bit, nearly any model like **Mixtral**, **Cohere**, **Granite**, **Gemma 3**, no more OOMs for vision finetuning, further VRAM usage reduction, speedup boost for 4-bit, Windows support, and more.
- **Perplexity AI on Windows**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1900371155753853427) announced the **Perplexity App** is now available in the **Windows** and **Microsoft App Store**, with voice-to-voice mode coming soon.
- **HuggingSnap on TestFlight**: [@mervenoyann](https://twitter.com/mervenoyann/status/1900492593810546774) announced that **HuggingSnap**, an offline vision LM for phones built by [@pcuenq](https://twitter.com/mervenoyann/status/1900492593810546774) and [@cyrilzakka](https://twitter.com/mervenoyann/status/1900492593810546774), is available on **TestFlight**, seeking feedback for further development.
- **New Trends in Machine Translation**: [@_akhaliq](https://twitter.com/_akhaliq/status/1900402426886115362) highlighted a paper on **New Trends for Modern Machine Translation with Large Reasoning Models**.
- **Microsoft and Shopify**: [@MParakhin](https://twitter.com/MParakhin/status/1900614024116740309) announced **Shopify** has acquired **VantageAI**.

**AI and Hardware**

- **AMD's Radeon GPUs Support on Windows**: [@dylan522p](https://twitter.com/dylan522p/status/1900352609271300572) reported on **AMD's @AnushElangovan** discussing making **Radeon GPUs** first-class citizens on **Windows** at the **RoCm User meetup**, with support for multiple **GPU architectures** and a focus on **CI and constant shipping**.
- **MLX LM New Home**: [@awnihannun](https://twitter.com/awnihannun/status/1900311865026372032) announced that **MLX LM** has a new home.

**AI Conferences and Events**

- **AI Dev 25 Conference**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1900594063516254299) kicked off **AI Dev 25** in San Francisco, noting that agents are the most exciting topic for AI developers. The conference included talks from **Google's Bill Jia** [@AndrewYNg](https://twitter.com/AndrewYNg/status/1900596396140671194), **Meta’s Chaya Nayak** [@AndrewYNg](https://twitter.com/AndrewYNg/status/1900599467822510154), and a panel discussion on building AI applications in 2025 [@AndrewYNg](https://twitter.com/AndrewYNg/status/1900610468747899142). [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1900636957740323302) shared a takeaway from **Nebius' Roman Chernin** emphasizing solving real-world problems, and [@AndrewYNg](https://twitter.com/AndrewYNg/status/1900617330906067136) highlighted a tip from **Replit’s @mattppal** on debugging by understanding the LLM's context.
- **GTC Fireside Chat**: [@ylecun](https://twitter.com/ylecun/status/1900298938764202154) announced they would be doing a fireside chat at **GTC** with **Nvidia chief scientist Bill Dally** on Tuesday next week.
- **Interrupt Conference**: [@LangChainAI](https://twitter.com/LangChainAI/status/1900621522475381145) promoted the **Interrupt conference**, listing its sponsors, including **CiscoCX**, **TryArcade**, **Box**, and others [@LangChainAI](https://twitter.com/LangChainAI/status/1900621520508219473).
- **Khipu AI in Santiago, Chile**: [@sirbayes](https://twitter.com/sirbayes/status/1900294930599121068) shared their talk on **Sequential decision making** using online variational bayes at **@Khipu_AI** in Santiago, Chile.  [@sarahookr](https://twitter.com/sarahookr/status/1900390025293942829) mentioned that the museum was really curious why their top item to see was the **khipu**.

**Other**

- **The value of open-source models**: [@Teknium1](https://twitter.com/Teknium1/status/1900514887413227654) expressed concern that banning Chinese models from Americans won't slow down their progress, and that not having access to the full range of models will make the US fall off.
- **AI and Film Making**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1900610357602750619) discussed the divergent qualities of **AI video generation**, allowing for creative impulses and exploration of unexpected moments, unconstrained by physical limitations.
- **The future of software**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1900540465545372102) speculates on the future of major public software companies, suggesting that companies focused on features and complex interfaces are at risk because the new software stack is intention-driven.
- **Team Size**: [@scottastevenson](https://twitter.com/scottastevenson/status/1900357184191390184) made the argument that small teams are winning, and that clinging to old big team culture may be damaging for your career.

**Humor/Memes**

- **"Everything is Transformer"**:  [@AravSrinivas](https://twitter.com/AravSrinivas/status/1900376022639665640) simply stated, “everything is transformer” with a picture of a transformer.
- **"Our top technology at Midjourney is Domain Not Resolving"**:  [@DavidSHolz](https://twitter.com/DavidSHolz/status/1900618951627075710) joked that **Midjourney’s** top technology is "Domain Not Resolving", seeking someone with at least 6 years of experience in the domain.
- **"One million startups must perish"**:  [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1900625829396443447) said "one million startups must perish".
- **"I will vibe edit human genome on a PlayStation 2"**:  [@fabianstelzer](https://twitter.com/fabianstelzer/status/1900625627302023532) posted "I will vibe edit human genome on a PlayStation 2".

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Gemma 3 Fine-Tuning Revolution: Performance and Efficiency in Unsloth**

- **Gemma 3 Fine-tuning now in Unsloth - 1.6x faster with 60% less VRAM** ([Score: 172, Comments: 36](https://reddit.com/r/LocalLLaMA/comments/1jba8c1/gemma_3_finetuning_now_in_unsloth_16x_faster_with/)): **Unsloth** now allows fine-tuning of **Gemma 3 (12B)** with **1.6x faster performance** and **60% less VRAM** usage compared to Hugging Face + FA2, fitting models like the **27B** in a **24GB GPU**. The platform fixes issues such as **infinite exploding gradients** on older GPUs and **double BOS tokens**, and supports a broad range of models and algorithms, including **full fine-tuning** and **Dynamic 4-bit quantization**. For more details, visit their [blog](https://unsloth.ai/blog/gemma3) and access their [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb) for free fine-tuning.
  - Users express enthusiasm for **Unsloth's** advancements, particularly the support for **full fine-tuning** and the potential for **8-bit fine-tuning**. **Danielhanchen** confirms that all methods, including 4-bit, 8-bit, and full fine-tuning, will be prioritized, and mentions the possibility of adding **torchao** for **float8** support.
  - There is interest in a more user-friendly interface, with requests for a **webUI** for local running to simplify usage. **Few_Painter_5588** predicts that Unsloth will become the primary toolset for **LLM fine-tuning**.
  - **FullDeer9001** shares positive feedback on running **Gemma3** on **Radeon XTX** with **8k context**, highlighting VRAM usage and prompt statistics, and compares it favorably to **Deepseek R1**. Users discuss the idea of optimizing the **12B model** for **16GB RAM** to enhance performance.


**Theme 2. Sesame CSM 1B Voice Cloning: Expectations vs. Reality**

- **[Sesame CSM 1B Voice Cloning](https://github.com/isaiahbjork/csm-voice-cloning)** ([Score: 216, Comments: 30](https://reddit.com/r/LocalLLaMA/comments/1jaxec3/sesame_csm_1b_voice_cloning/)): **Sesame CSM 1B** is a newly released **voice cloning model**. No additional details were provided in the post.
  - **Voice Cloning Model Licensing and Usage**: There is a discussion about the licensing differences between **Sesame** (Apache licensed) and **F5** (Creative Commons Attribution Non Commercial 4.0), highlighting that **Sesame** can be used for commercial purposes. Users also mention the integration of voice cloning into a conversational speech model (CSM) as a potential advancement.
  - **Performance and Compatibility Issues**: Users report slow performance of the voice cloning model, taking up to **50 seconds** for a full paragraph on a GPU, and note that it may not be optimized for Windows. There are suggestions that it might work better on Linux and that running it on a mini PC without a dedicated GPU could be challenging due to the "experimental" triton backend for CPU.
  - **Technical Adjustments and API Access**: **Chromix_** shares steps to get the model working on Windows by upgrading to **torch 2.6** and other packages, and mentions bypassing the need for a **Hugging Face account** by downloading files from a mirror repo. They also provide a link to the [API endpoint for voice cloning](https://github.com/SesameAILabs/csm/issues/61#issuecomment-2724204772).


- **Conclusion: Sesame has shown us a CSM. Then Sesame announced that it would publish... something. Sesame then released a TTS, which they obviously misleadingly and falsely called a CSM. Do I see that correctly?** ([Score: 154, Comments: 51](https://reddit.com/r/LocalLLaMA/comments/1jb1sgv/conclusion_sesame_has_shown_us_a_csm_then_sesame/)): **Sesame's Controversy** revolves around their misleading marketing strategy, where they announced a **CSM** but released a **TTS** instead, falsely labeling it a CSM. The issue could have been mitigated if Sesame had clearly communicated that it wouldn't be open source.
  - **Misleading Marketing Strategy**: Many users express disappointment with Sesame's marketing tactics, noting that the company created significant hype by suggesting an open-source release, only to deliver a less impressive product. **VC-backed** companies often use such strategies to gauge product-market fit and generate investor interest, as seen with **Sesame's lead investor a16z**.
  - **Technical Challenges and Model Performance**: There's a consensus that the released **1B model** is underwhelming in performance, particularly in real-time applications. Users discuss technical aspects, such as the **Mimi tokenizer** and the model's architecture, which contribute to its slow speed, and suggest optimizations like using **CUDA graphs** or alternative models like **exllamav2** for better performance.
  - **Incomplete Product Release**: Discussions highlight that Sesame's release lacks crucial components of the demo pipeline, such as the **LLM, STT, and VAD**, forcing users to build these themselves. The demo's impressive performance contrasts with the actual release, raising questions about the demo's setup possibly using larger models or more powerful hardware like **8xH100 nodes**.


**Theme 3. QwQ's Rise: Dominating Benchmarks and Surpassing Expectations**

- **[QwQ on LiveBench (update) - is better than DeepSeek R1!](https://i.redd.it/sb78tt607joe1.png)** ([Score: 256, Comments: 117](https://reddit.com/r/LocalLLaMA/comments/1jaoc8n/qwq_on_livebench_update_is_better_than_deepseek_r1/)): **QwQ-32b** from Alibaba surpasses **DeepSeek R1** on **LiveBench**, achieving a global average score of **71.96**, compared to **DeepSeek R1's** **71.57**. **QwQ-32b** consistently outperforms in subcategories like Reasoning, Coding, Mathematics, Data Analysis, Language, and IF Average, as highlighted in the comparison table.
  - There is skepticism about the **QwQ-32b**'s performance compared to **DeepSeek R1**, with some users noting that **Alibaba** tends to optimize models for benchmarks rather than real-world scenarios. **QwQ-32b** is highlighted as a strong model, but there are doubts about its stability and real-world knowledge compared to **R1**.
  - **Coding performance** is a contentious point, with users questioning how **QwQ-32b** approaches **Claude 3.7** in coding capabilities. Discussions mention that **LiveBench** primarily tests in **Python** and **JavaScript**, while **Aider** tests over 30 languages, suggesting potential discrepancies in testing environments.
  - Some users express excitement about the potential of **QwQ-max**, with anticipation that it might surpass **R1** in both size and performance. There are also discussions on the impact of settings changes on the model's performance, with links provided for further insights ([Bindu Reddy's tweet](https://x.com/bindureddy/status/1900331870371635510)).


- **Qwq-32b just got updated Livebench.** ([Score: 130, Comments: 73](https://reddit.com/r/LocalLLaMA/comments/1jao3fg/qwq32b_just_got_updated_livebench/)): **QwQ 32B** has been updated on **LiveBench**, providing new insights into its performance. The full results can be accessed through the [Livebench](https://livebench.ai/#/) link.
  - The **QwQ 32B** model is praised for its local coding capabilities, with some users noting it surpasses larger models like **R1** in certain tasks. Users have discussed adjusting the model's thinking time by tweaking settings such as the **logit bias** for the ending tag, and some have experimented with recent updates to resolve issues like infinite looping.
  - Discussions highlight the evolving power of smaller models like **QwQ 32B**, with users noting their increasing potential for local applications compared to larger flagship models. Some users express surprise at the model's creative capabilities and its ability to perform well on benchmarks, leading to decisions like dropping **OpenAI** subscriptions.
  - There is a debate on the implications of open-sourcing models, with some users suggesting that **China's** strategy of open-sourcing models accelerates development, contrasting with the U.S. approach focused on corporate profit. Concerns are raised about the future of open-source availability, especially if competitive advantages shift.


- **[Meme i made](https://v.redd.it/vzku6n1lbjoe1)** ([Score: 982, Comments: 55](https://reddit.com/r/LocalLLaMA/comments/1jaoy9g/meme_i_made/)): The post titled "Meme i made" lacks detailed content as it only mentions a meme creation related to the **QwQ Model's Thinking Process**. No additional information or context about the video or the meme is provided, making it difficult to extract further technical insights.
  - Discussions highlight the **QwQ Model's** tendency to doubt itself, leading to inefficient token usage and prolonged response times. This behavior is likened to "fact-checking" itself excessively, which some users find inefficient compared to traditional LLMs.
  - There's a consensus that current reasoning models, like QwQ, are in their early stages, akin to **GPT-3's** initial release, with expectations for significant improvements in their reasoning capabilities over the next year. Users anticipate a shift towards reasoning in latent space, which could enhance efficiency by a factor of **10x**.
  - Humorous and critical commentary highlights the model's repetitive questioning and self-doubt, drawing parallels to outdated technology and sparking discussions about the potential for these models to improve in handling complex reasoning tasks without excessive self-questioning.


**Theme 4. Decentralized LLM Deployment: Akash, IPFS & Pocket Network Challenges**

- **[HowTo: Decentralized LLM on Akash, IPFS & Pocket Network, could this run LLaMA?](https://pocket.network/case-study-building-a-decentralized-deepseek-combining-open-data-compute-and-reasoning-with-pocket-network/)** ([Score: 229, Comments: 20](https://reddit.com/r/LocalLLaMA/comments/1jb1tum/howto_decentralized_llm_on_akash_ipfs_pocket/)): The post titled **"HowTo: Decentralized LLM on Akash, IPFS & Pocket Network, could this run LLaMA?"** suggests deploying a decentralized Large Language Model (LLM) using **Akash**, **IPFS**, and **Pocket Network**. It questions the feasibility of running **LLaMA**, a specific LLM, on this decentralized infrastructure, implying a focus on leveraging decentralized technologies for AI model deployment.
  - **Concerns about Security and Privacy**: Users question the cryptographic verification process of **Pocket Network**, expressing doubts about ensuring the correct model is served and the privacy of prompts. There are concerns about whether user data, such as IP addresses, might be logged and how the network handles latency for anonymity.
  - **Challenges of Decentralized Infrastructure**: Commenters highlight the technical challenges of running LLMs in a decentralized manner, especially the need for high bandwidth and low latency between nodes, which currently limits the feasibility of distributed LLM deployment compared to single-machine setups.
  - **Decentralization vs. Centralization**: The discussion contrasts **Pocket Network**'s API relay role with centralized AI hosting, noting that while **Pocket** does not run the model itself, the use of **Akash** for model hosting offers benefits like resilience and potential cost savings, despite adding complexity with a crypto layer.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. Advanced AI Video Generation with SDXL, Wan2.1, and Long Context Tuning**

- **[Another video aiming for cinematic realism, this time with a much more difficult character. SDXL + Wan 2.1 I2V](https://v.redd.it/t88g56krqnoe1)** ([Score: 1018, Comments: 123](https://reddit.com/r/StableDiffusion/comments/1jb47bs/another_video_aiming_for_cinematic_realism_this/)): This post discusses the creation of a video aimed at achieving **cinematic realism** using **SDXL** and **Wan 2.1 I2V**. It highlights the challenge of working with a more difficult character in this context.
  - **Technical Challenges and Techniques**: **Parallax911** shares the complexity of achieving **cinematic realism** with **SDXL** and **Wan 2.1 I2V**, highlighting the use of **Photopea** for inpainting and compositing in **Davinci Resolve**. They mention the difficulty in achieving consistency and realism, especially with complex character designs, and the use of **Blender** for animating segments like the door opening.
  - **Project Costs and Workflow**: The project incurred a cost of approximately **$70** using **RunPod's L40S** at **$0.84/hr**, taking about **80 hours** of GPU time. **Parallax911** utilized a workflow involving **RealVisXL 5.0**, **Wan 2.1**, and **Topaz Starlight** for upscaling, with scenes generated at **61 frames, 960x544** resolution, and **25 steps**.
  - **Community Feedback and Suggestions**: The community praised the atmospheric storytelling and sound design, with specific feedback on elements like water droplet size and the need for a tutorial. Some users suggested improvements, such as better integration of AI and traditional techniques, and expressed interest in more action-oriented scenes with characters like **Samus Aran** from **Metroid**.


- **[Video extension in Wan2.1 - Create 10+ seconds upscaled videos entirely in ComfyUI](https://v.redd.it/xi58u5d3qmoe1)** ([Score: 123, Comments: 23](https://reddit.com/r/StableDiffusion/comments/1jb0h7i/video_extension_in_wan21_create_10_seconds/)): The post discusses a **highly experimental workflow** in **Wan2.1** using **ComfyUI** for creating upscaled videos, achieving approximately **25% success**. The process involves generating a video from the last frame of an initial video, merging, upscaling, and frame interpolation, with specific parameters like **Sampler: UniPC**, **Steps: 18**, **CFG: 4**, and **Shift: 11**. More details can be found in the [workflow link](https://civitai.com/models/1297230?modelVersionId=1531202).
  - Users are inquiring about the **aspect ratio** handling in the workflow, questioning if it's automatically set or needs manual adjustment for input images.
  - There is **positive feedback** from users interested in the workflow, indicating anticipation for such a solution.
  - Concerns about **blurriness** in the second half of clips were raised, with suggestions that it might be related to the input frame quality.


- **[Animated some of my AI pix with WAN 2.1 and LTX](https://v.redd.it/z5r0kyf1smoe1)** ([Score: 115, Comments: 10](https://reddit.com/r/StableDiffusion/comments/1jb0n50/animated_some_of_my_ai_pix_with_wan_21_and_ltx/)): The post discusses the creation of **animated AI videos** using **WAN 2.1** and **LTX**. Without further context or additional details, the focus remains on the tools used for animation.
  - **Model Usage**: **LTX** was used for the first clip, the jumping woman, and the fighter jet, while **WAN** was used for the running astronaut, the horror furby, and the dragon.
  - **Hardware Details**: The videos were generated using a rented cloud computer from **Paperspace** with an **RTX5000** instance.


**Theme 2. OpenAI's Sora: Transforming Cityscapes into Dystopias**

- **[OpenAI's Sora Turns iPhone Photos of San Francisco into a Dystopian Nightmare](https://v.redd.it/y67d5ph47loe1)** ([Score: 931, Comments: 107](https://reddit.com/r/ChatGPT/comments/1jawa6c/openais_sora_turns_iphone_photos_of_san_francisco/)): **OpenAI's Sora** is a tool that transforms **iPhone photos** of **San Francisco** into images with a **dystopian** aesthetic. The post likely discusses the implications and visual results of using AI to alter real-world imagery, although specific details are not available due to the lack of text content.
  - Several commenters express skepticism about the impact of **AI-generated dystopian imagery**, with some suggesting that actual locations in **San Francisco** or other cities already resemble these dystopian visuals, questioning the need for AI alteration.
  - **iPhone** as the device used for capturing the original images is a point of contention, with some questioning its relevance to the discussion, while others emphasize its importance in understanding the image source.
  - The conversation includes a mix of admiration and concern for the **AI's capabilities**, with users expressing both astonishment at the technology and anxiety about distinguishing between AI-generated and real-world images in the future.


- **[Open AI's Sora transformed Iphone pics of San Francisco into dystopian hellscape...](https://v.redd.it/ukxvzsatzkoe1)** ([Score: 535, Comments: 58](https://reddit.com/r/OpenAI/comments/1javmkq/open_ais_sora_transformed_iphone_pics_of_san/)): **OpenAI's Sora** has transformed **iPhone photos of San Francisco** into a dystopian hellscape, showcasing its capabilities in altering digital images to create a futuristic, grim aesthetic. The post lacks additional context or details beyond this transformation.
  - Commenters draw parallels between the **dystopian images** and real-world locations, with references to **Delhi**, **Detroit**, and **Indian streets**, highlighting the AI's perceived biases in interpreting urban environments.
  - There are concerns about **AI's text generation capabilities**, with one commenter noting that **sign text** in the images serves as a tell-tale sign of AI manipulation.
  - Users express interest in the **process of creating such images**, with a request for **step-by-step instructions** to replicate the transformation on their own photos.


**Theme 3. OpenAI and DeepSeek: The Open Source Showdown**

- **[I Think Too much insecurity](https://i.redd.it/9xpl7abaoooe1.jpeg)** ([Score: 137, Comments: 58](https://reddit.com/r/ClaudeAI/comments/1jb8aj5/i_think_too_much_insecurity/)): **OpenAI** accuses **DeepSeek** of being "state-controlled" and advocates for bans on Chinese AI models, highlighting concerns over state influence in AI development. The image suggests a geopolitical context, with American and Chinese flags symbolizing the broader debate over state control and security in AI technologies.
  - The discussion highlights skepticism over **OpenAI's** claims against **DeepSeek**, with users challenging the notion of state control by pointing out that **DeepSeek's** model is open source. Users question the validity of the accusation, with calls for proof and references to **Sam Altman's** past statements about the lack of a competitive moat for **LLMs**.
  - **DeepSeek** is perceived as a significant competitor, managing to operate with lower expenses and potentially impacting **OpenAI's** profits. Some comments suggest that **DeepSeek**'s actions are seen as a form of economic aggression, equating it to a declaration of war on American interests.
  - There is a strong undercurrent of criticism towards **OpenAI** and **Sam Altman**, with users expressing distrust and dissatisfaction with their actions and statements. The conversation includes personal attacks and skepticism towards **Altman's** credibility, with references to his promises of open-source models that have not materialized.


- **Built an AI Agent to find and apply to jobs automatically** ([Score: 123, Comments: 22](https://reddit.com/r/OpenAI/comments/1jb49lo/built_an_ai_agent_to_find_and_apply_to_jobs/)): An AI agent called **SimpleApply** automates job searching and application processes by matching users' skills and experiences with relevant job roles, offering three usage modes: manual application with job scoring, selective auto-application, and full auto-application for jobs with over a **60% match** score. The tool aims to streamline job applications without overwhelming employers and is praised for finding numerous remote job opportunities that users might not discover otherwise.
  - Concerns about **data privacy and compliance** were raised, with questions on how **SimpleApply** handles **PII** and its adherence to **GDPR** and **CCPA**. The developer clarified that they store data securely with compliant third parties and are working on explicit user agreements for full compliance.
  - **Application spam risks** were discussed, with suggestions to avoid reapplying to the same roles to prevent being flagged by **ATS** systems. The developer assured that the tool only applies to jobs with a high likelihood of landing an interview to minimize spam.
  - Alternative **pricing strategies** were suggested, such as charging users only when they receive callbacks via email or call forwarding. This approach could potentially be more attractive to unemployed users who are hesitant to spend money upfront.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1. Google's Gemma 3 Takes Center Stage Across Tools**

- [**Unsloth Supercharges Gemma 3 Finetuning, Vision Too**](https://unsloth.ai/blog/gemma3):  **Unsloth AI** now boasts full support for **Gemma 3**, enhancing finetuning speeds by **1.6x**, slashing **VRAM usage by 60%**, and expanding context length **6x** compared to standard Flash Attention 2 setups on 48GB GPUs. Optimized versions for full finetuning, 8-bit, and pretraining are available on [Hugging Face](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b), and initial support for **Gemma 3 vision** is also implemented, though **Ollama** users might face compatibility issues *for now*.
- [**Gemma 3 12B Outsmarts Qwen, Needs GPT4All Update**](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b):  Users reported **Gemma 3 12B** outperforming **Qwen 14B** and **32B** in personal tests and excelling in multilingual question answering, yet **GPT4All** requires updates for full **Gemma 3 12B** support due to architectural shifts and the need for an `mmproj` file. In a basic physics test, **Gemma-3-12b** correctly predicted jar shattering when water freezes, unlike **DeepSeek-R1**.
- [**vLLM and LigerKernel Gear Up for Gemma 3 Integration**](https://github.com/linkedin/Liger-Kernel/pull/606):  **vLLM** is actively working on **Gemma 3** support, tracked in [this GitHub issue](https://github.com/vllm-project/vllm/issues/14696), while a draft implementation of **Gemma 3** into **LigerKernel** is underway, noting high architectural similarity to **Gemma 2** with minor **RMSNorm** call differences; however, some users are reporting context window size issues with **Gemma3** and TGI.


**Theme 2. New Models Emerge: OLMo 2, Command A, Jamba 1.6, PaliGemma 2 Mix**

- [**AI2's OLMo 2 32B Shines as Open-Source GPT-3.5 Killer**](https://allenai.org/blog/olmo2-32B):  **AI2** launched **OLMo 2 32B**, a fully open-source model trained on **6T tokens** using Tulu 3.1, which it claims outperforms **GPT3.5-Turbo** and **GPT-4o mini** on academic benchmarks, while costing only one-third of **Qwen 2.5 32B** training; available in **7B**, **13B**, and **32B** sizes, it is now available on **OpenRouter** and sparking discussion in **Yannick Kilcher's** community about its open nature and performance.
- [**Cohere's Command A and AI21's Jamba 1.6 Models Arrive with Massive Context**](https://openrouter.ai/cohere/command-a): **Cohere** unveiled **Command A**, a **111B parameter open-weights model** with a **256k context window**, designed for agentic, multilingual, and coding tasks, while **AI21** released **Jamba 1.6 Large** (**94B active parameters**, **256K context**) and **Jamba 1.6 Mini** (**12B active parameters**), both now featuring structured JSON output and tool-use, all models available on **OpenRouter**. However, **Command A** is exhibiting a peculiar bug with prime number queries, and local API performance is reportedly suboptimal without specific patches.
- [**Google's PaliGemma 2 Mix Family Unleashes Vision-Language Versatility**](https://huggingface.co/blog/paligemma2mix):  **Google** released **PaliGemma 2 Mix**, a vision language model family in **3B**, **10B**, and **28B** sizes, with **224** and **448** resolutions, capable of open-ended vision language tasks and document understanding, while **Sebastian Raschka** reviewed multimodal models including **Meta AI's Llama 3.2** in [a blog post](https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html); users in **HuggingFace** are also seeking open-source alternatives to **Gemini 2.0 Flash** with similar image editing capabilities.


**Theme 3. Coding Tools and IDEs Evolve with AI Integration**

- [**Cursor IDE Users Cry Foul Over Performance and Claude 3.7 Downgrade**](https://downloads.cursor.com/production/8f8a2000673d2c48f6cac5eea2f3f9f2ed5e4ec2/win32/x64/user-setup/CursorUserSetup-x64-0.47.4.exe): **Cursor IDE** faces user backlash for **lag and freezing** on **Linux** and **Windows** after updates like **0.47.4**, with **Claude 3.7** deemed *dumb as bricks* and rule-ignoring, costing double credits, and the **Cursor agent** criticized for spawning excessive terminals; despite issues, **v0** remains praised for rapid UI prototyping, contrasting with **Cursor's** credit system and limited creative freedom compared to **v0**.
- [**Aider and Claude Team Up, Users Debate Rust Port and MCP Server Setup**](https://github.com/sengokudaikon/aider-mcp-server):  Users laud the powerful combination of **Claude** with **Aider**, augmented with web search and bash scripting, while discussions on porting **Aider** to **Rust** for speedier file processing are met with skepticism, citing LLM API bottlenecks; a user-improved readme for the **Aider MCP Server** emerged, yet setup complexities persist, and **Linux** users are finding workarounds to run **Claude Desktop**.
- [**'Vibe Coding' Gains Momentum, Showcased in Game Dev and Resource Lists**](https://github.com/filipecalegario/awesome-vibe-coding):  The concept of "vibe coding"—AI-assisted collaborative coding—is gaining traction, exemplified by a developer creating a multiplayer 3D game **100% with AI** in 20 hours for 20 euros using **Cursor**, and **Awesome Vibe Coding**, a curated list of AI coding tools and resources, has been released on [GitHub](https://github.com/filipecalegario/awesome-vibe-coding), and a [GitDoc VS Code extension](https://github.com/lostintangent/gitdoc) for auto-committing changes is gaining popularity, sparking UI design ideas for "vibe coding" IDEs with visualized change trees.


**Theme 4. Training and Optimization Techniques Advance**

- [**Unsloth Pioneers GRPO for Reasoning Models, Dynamic Quantization for Speed**](https://unsloth.ai/blog/grpo):  **Unsloth** introduces **GRPO** (Guiding Preference Optimization), enabling **10x longer context** with **90% less VRAM** for reasoning models, and highlights dynamic quantization outperforming GGUF in quality, especially for **Phi-4**, showcased on the [Hugging Face leaderboard](https://unsloth.ai/blog/dynamic-4bit), while **Triton bitpacking** achieves massive speedups up to **98x** over Pytorch, reducing **Llama3-8B** repacking time from 49 sec to 1.6 sec.
- [**DeepSeek's Search-R1 Leverages RL for Autonomous Query Generation, IMM Promises Faster Sampling**](https://arxiv.org/abs/2503.09516):  **DeepSeek's Search-R1** extends **DeepSeek-R1** with reinforcement learning (**RL**) to generate search queries during reasoning, using retrieved token masking for stable training and enhanced **LLM** rollouts, while **Inductive Moment Matching (IMM)** emerges as a novel generative model class promising faster inference via one- or few-step sampling, surpassing diffusion models without pre-training or dual-network optimization.
- [**Reasoning-Gym Explores GRPO, veRL, and Composite Datasets for Enhanced Reasoning**](https://www.oxen.ai/blog/grpo-vram-requirements-for-the-gpu-poor):  **Group Relative Policy Optimization (GRPO)** gains popularity for RL in LLMs, with **reasoning-gym** confirming **veRL training** success for **chain_sum** and exploring composite datasets for improved reasoning capabilities, moving towards a refactor for enhanced *"all-around"* model performance, and the project nears 500 stars, with version *0.1.16* uploaded to pypi.


**Theme 5. Infrastructure and Access: H100s, VRAM, and API Pricing**

- [**SF Compute Disrupts H100 Market with Low Prices, Vultr Enters Inference API Space**](https://sfcompute.com/):  **SF Compute** offers surprisingly low **H100** rental prices, especially for short-term use, advertising **128 H100s** available hourly and launching an additional **2,000 H100s** soon, while **Vultr** announces inference API pricing at **$10 for 50 million output tokens** initially, then **2 cents per million**, accessible via an OpenAI-compatible endpoint, stemming from a large GH200 purchase.
- [**LM Studio Users Dive into Runtime Retrieval and Snapdragon Compatibility**](https://extensions.lmstudio.ai/backends-master-list-stable.json): **LM Studio** users are reverse-engineering the application to find download URLs for offline runtimes, after claims it runs offline, discovering CDN 'APIs' like [Runtime Vendor](https://extensions.lmstudio.ai/vendor-win-llama-cuda-vendor-v1.tar.gz), while **Snapdragon X Plus** GPU support in **LM Studio** requires direct *llama.cpp* execution, and users report **Gemini Vision** limitations potentially due to geo-restrictions in Germany/EU.
- [**VRAM Consumption Concerns Rise: Gemma 3 and SFT Discussed**](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm):  Users report increased **VRAM usage for Gemma 3** post-vision update, speculating **CLIP** integration as a cause, and **Gemma's SFT** VRAM needs are debated, suggesting potentially higher requirements than **Qwen 2.5** in similar conditions, while resources for estimating memory usage for LLMs are shared, like [Substratus AI blog](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm) and [Hugging Face space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage).


---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's Gemma 3 Support Gains Steam**: Unsloth now supports **Gemma 3**, including full fine-tuning and 8-bit, and optimizes **Gemma 3 (12B) finetuning by 1.6x**, reduces **VRAM usage by 60%**, and extends context length by **6x** compared to environments using Flash Attention 2 on a 48GB GPU.
   - All Gemma 3 model uploads are available on [Hugging Face](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b), including versions optimized for full finetuning, 8-bit, and pretraining.
- **Dynamic Quants Face Off GGUF Quality**: Discussion compares **dynamic quantization** with **GGUF** models, especially regarding the trade-offs between size and quality, with Unsloth's dynamic quants for **Phi-4** on the [Hugging Face leaderboard](https://unsloth.ai/blog/dynamic-4bit).
   - A direct comparison with GGUF benchmarks is anticipated to clarify performance at different bit widths, with a *likely* holdup being **llama-server** lacking vision support *yet*.
- **GRPO to Grant Reasoning Greatness**: **GRPO** (Guiding Preference Optimization) is *coming next week* along with new notebooks, and now supports **10x longer context with 90% less VRAM**, detailed in a [blog post](https://unsloth.ai/blog/grpo).
   - The team stated, *only if you let it reason about the rules first a la GRPO*, which is specifically designed for reasoning models, offering significant memory savings and expanded context windows.
- **Vision Models Get Unsloth's Vision**: Unsloth has implemented the *train on completions* feature and also resizing of images for Vision Language Models, and the models now *auto resize images which stops OOMs and also allows truncating sequence lengths*.
   - A [Qwen2_VL Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_%25287B%2529-Vision.ipynb) was also shared for images.
- **QwQ-32B Bugfixes Bolster Model**: Bugfixes have been implemented for the **QwQ-32B model**, as highlighted in a [blog post](https://unsloth.ai/blog/qwq-32b) with corresponding [model uploads](https://huggingface.co/collections/unsloth/qwen-qwq-32b-collection-676b3b29c20c09a8c71a6235).
   - These fixes improve the model's stability and performance, ensuring a smoother user experience.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Experiences Performance Hiccups on Linux and Windows**: Users have reported **Cursor** experiencing **lag** and **freezing** on **Linux** and **Windows**, particularly after updates such as **0.47.4** ([download link](https://downloads.cursor.com/production/8f8a2000673d2c48f6cac5eea2f3f9f2ed5e4ec2/win32/x64/user-setup/CursorUserSetup-x64-0.47.4.exe)).
   - One user detailed that the UI freezes for seconds after just **20-30 messages** on Linux; another noted constant lags on Windows even with a high performance laptop running version **3.7**.
- **Claude 3.7 Judged as Underperforming and Disobedient**: Users find **Claude 3.7** *dumb as bricks* following the update to **0.47.4**, and using it now costs double the credits.
   - Members mentioned that **Sonnet 3.7** ignores global rules, even when prompted to output them, with one user jokingly suggesting to *put 'be a good boy' in your prompt and it will fix anything*, according to [a tweet](https://x.com/kregenrek/status/1899941361908146430).
- **Cursor Agent Uncorks Terminal Barrage**: Multiple users are finding the Cursor agent is spawning an excessive amount of terminals, causing frustration, especially when it restarts servers that are already running.
   - One member suggested that this functionality should either be built-in or users should just write the terminal commands themselves.
- **V0 Praised for Prototyping Speed**: Some users advocate using **v0** for front-end prototyping due to its UI design capabilities with subframes, which is similar to Figma, before transferring designs to Cursor.
   - One user stated *it's much better to build prototype and layout (better front end) imo then import locally to cursor*, although others favor Cursor because of v0's credit system and limited creative autonomy.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LM Studio Users Seek Support**: A member suggested users facing issues with **LM Studio** seek assistance in the dedicated **LM Studio Discord**.
   - This aims to provide more focused help for **LM Studio** related problems.
- **SMILES String Encoders Sought for Stereoisomers**: A member inquired about models or architectures that can encode a **SMILES string** into various **stereoisomers** or a **ChemDraw** input.
   - The goal is to enable chemical descriptor extraction from these encodings.
- **Diffusion Models Excel at Generative Tasks**: A [Nature article](https://www.nature.com/articles/s41467-024-54281-3) was shared, highlighting the proficiency of **diffusion models (DMs)** in modeling complex data distributions and generating realistic samples for diverse media.
   - These models are now state-of-the-art for generating images, videos, audio, and 3D scenes.
- **Search-R1** Autonomously Searches with RL**: The **Search-R1** paper was introduced, detailing an extension of the **DeepSeek-R1** model that employs reinforcement learning (**RL**) to generate search queries during reasoning (see [paper](https://arxiv.org/abs/2503.09516)).
   - The model uses retrieved token masking for stable **RL** training, enhancing **LLM** rollouts through multi-turn search interactions.
- **IMM** Claims Faster Sampling Times**: A [paper](https://arxiv.org/abs/2503.07565) on **Inductive Moment Matching (IMM)** was shared, noting it as a novel class of generative models that promise faster inference through one- or few-step sampling, surpassing diffusion models.
   - Notably, **IMM** does not require pre-training initialization or the optimization of two networks, unlike distillation methods.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LLM Faceoff: Ministral 8B vs. Exaone 8B**: Members suggested using **Ministral 8B** or **Exaone 8B** at 4-bit quantization for LLM tasks.
   - A user running an M4 Mac mini with 24 GB RAM, was trying to figure out tokens per second.
- **SmolAgents Has Troubles With Gemma3**: A user reported errors running **Gemma3** with **SmolAgents**, stemming from code parsing and regex issues, pointing to [a potential fix on GitHub](https://github.com/huggingface/smolagents/pull/883).
   - The user resolved the problem by increasing the **Ollama context length**.
- **Awesome Vibe Coding Curates Resources**: A curated list of tools, editors, and resources for **AI-assisted coding** has been released, called [Awesome Vibe Coding](https://github.com/filipecalegario/awesome-vibe-coding).
   - The list includes **AI-powered IDEs**, **browser-based tools**, **plugins**, **command line tools**, and the **latest news** on *vibe coding*.
- **PaliGemma 2 Models Drop**: Google released **PaliGemma 2 Mix**, a family of vision language models with three sizes (**3B**, **10B**, and **28B**) and resolutions of **224** and **448** that *can do vision language tasks with open-ended prompts*.
   - Check out the [blog post](https://huggingface.co/blog/paligemma2mix) for more.
- **Chess Championship Models Make Illegal Moves?**: A user shared a [YouTube playlist](https://youtube.com/playlist?list=PLBRObSmbZluRddpWxbM_r-vOQjVegIQJC) titled *Chatbot Chess Championship 2025*, showcasing language models or chess engines playing chess.
   - Participants speculated whether the models were true language models or merely calling chess engines, and one person noted a language model made illegal moves.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Complexity Extension Goes Full Maintenance**: The **Complexity extension** for Perplexity AI is now in full maintenance mode due to a layout update breaking the [extension](https://github.com/danielcardeenas/perplexity-complexity).
   - The developer thanked users for their patience during this maintenance period.
- **Locking Kernel a Pipe Dream for Security**: Users debated whether **locking down the kernel** would improve security, in the general channel.
   - Others argued that this is not feasible due to the **open-source nature of Linux**, with one user joking about using Windows instead.
- **Perplexity Users Beg for More Context**: Users are requesting **larger context windows** in Perplexity AI and are willing to pay extra to avoid using ChatGPT.
   - A user cited Perplexity's features like *unlimited research on 50 files at a time*, **spaces for custom instructions**, and the **ability to choose reasoning models** as reasons to stay.
- **Grok 3 Riddled with Bugs Upon Release**: The newly released **Grok AI** is reportedly buggy.
   - Users reported that *suddenly the chat stops working or breaks in middle*.
- **Gemini Deep Research Not So Deep**: Users testing the new **Gemini Deep Research** feature found it weaker than **OpenAI's** offerings.
   - One user found it retained less context than the regular Gemini, even with search disabled.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude + Aider = Coding SuperPower**: Members discussed using **Claude** with **Aider**, which *augments it with web search/URL scrapping and running bash script calling*, resulting in more powerful prompting capabilities.
   - One user highlighted that *each unique tool added to Claude unlocks a lot more than the sum of its parts*, especially when the model searches the internet for bugs.
- **Does Rust Rocket Aider Speed?**: One user inquired about porting **Aider** to **C++** or **Rust** for faster file processing, particularly when loading large context files for **Gemini** models.
   - Others expressed skepticism, suggesting that the bottleneck remains with the **LLM API** and any improvements might not be *quantifiable*.
- **Linux Lovers Launch Claude Desktop**: Users shared instructions for getting the **Claude Desktop** app to work on **Linux**, as there isn't an official version.
   - One user referenced a [GitHub repo](https://github.com/aaddrick/claude-desktop-debian) providing **Debian-based** installation steps while another shared their edits to an **Arch Linux PKGBUILD**.
- **Aider MCP Server Readme Rescued**: Users discussed the **Aider MCP Server**, with one mentioning that another user's readme was *100x better*, referring to [this repo](https://github.com/sengokudaikon/aider-mcp-server).
   - However, another user humorously stated that they still *can't setup ur mcp* despite the readme's existence.
- **DeepSeek Models Speak Too Much**: A user reported that **DeepSeek models** are generating excessive output, around **20-30 lines** of phrases, and inquired about setting a `thinking-tokens` value in the configuration.
   - It was noted that **20 lines** is pretty standard for the **R1 model**, and one user shared that they once waited **2 minutes** for the model to think on a **5 word prompt**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OLMo 2 32B Dominates GPT 3.5**: AI2 released [**OLMo 2 32B**](https://allenai.org/blog/olmo2-32B), a fully open-source model trained up to **6T tokens** using Tulu 3.1, outperforming **GPT3.5-Turbo** and **GPT-4o mini** on academic benchmarks.
   - It is claimed to require only one third of the cost of training **Qwen 2.5 32B** while reaching similar performance and is available in **7B**, **13B**, and **32B** parameter sizes.
- **Vibe Coding Creates Games with AI**: A developer created a multiplayer 3D game **100% with AI**, spending 20 hours and 20 euros, calling the concept *vibe coding*, and sharing the [guide](https://x.com/nicolaszu/status/1899931187398979890?s=46).
   - The game features realistic elements like hit impacts, smoke when damaged, and explosions on death, all generated via prompting in **Cursor** with no manual code edits.
- **Levels.io's AI Flight Sim Soars to $1M ARR**: A member referenced the success of [Levels.io's flight simulator](https://x.com/levelsio/status/1893350391158292550), built with **Cursor**, which quickly reached **$1 million ARR** by selling ads in the game.
   - Levelsio noted, *AI really is a creativity and speed maximizer for me, making me just way more creative and more fast*.
- **GitDoc Extension Auto-Commits Changes**: Members shared the [GitDoc VS Code extension](https://github.com/lostintangent/gitdoc) that allows you to edit a Git repo and auto commit on every change.
   - One user suggested branching, restarting and other features and said *storage is cheap, like auto commit on every change and visualize the tree of changes*.
- **Latent Space Podcast Dives into Snipd AI App**: The Latent Space podcast released a new [Snipd podcast](https://x.com/latentspacepod/status/1900666708270215383) with Kevin Ben Smith about the **AI Podcast App** for Learning and released their first ever **OUTDOOR podcast** on [Youtube](https://youtu.be/FNRO_SYx68Q).
   - The podcast features a discussion about @aidotengineer NYC, switching from **Finance to Tech**, how AI can help us get a lot more out of our podcast time, and dish the details on the **tech stack of Snipd app**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio's Runtimes Retrieved via Reverse Engineering**: A user decompiled **LM Studio** to locate download URLs for offline use, discovering the [backends master list](https://extensions.lmstudio.ai/backends-master-list-stable.json) and CDN 'APIs' like the [Runtime Vendor](https://extensions.lmstudio.ai/vendor-win-llama-cuda-vendor-v1.tar.gz).
   - This was after another user claimed *LM Studio doesn't need an internet connection to run*, showing a demand for offline runtime access.
- **Snapdragon Support Requires Direct llama.cpp Execution**: A user reported that **LM Studio** did not detect their **Snapdragon X Plus** GPU, and another member replied that GPU support requires running *llama.cpp directly*.
   - They directed the user to this [github.com/ggml-org/llama.cpp/pull/10693](https://github.com/ggml-org/llama.cpp/pull/10693) pull request for more information.
- **Gemini Vision Hampered by Geo-Restrictions**: Users reported issues testing **Gemini 2.0 Flash Experimental's** image processing abilities, potentially due to regional restrictions in Germany/EU.
   - One user in Germany suspected that the limitations were due to local laws while a user in the US reported that Gemini in AI Studio failed to perform the image manipulation.
- **AI Chess Tournament Highlights Model Accuracy**: An AI chess tournament featuring **15 models** was held, with results available at [dubesor.de/chess/tournament](https://dubesor.de/chess/tournament), where results are impacted by game length and opponent moves.
   - Although **DeepSeek-R1** achieved a **92%** accuracy, the organizer clarified that accuracy varies based on game length and opponent moves, and normal O1 was too expensive to run in the tournament.
- **VRAM Consumption for Gemma 3 Jumps After Vision Update**: Following a vision speed increase update, a user reported a significant increase in **VRAM usage for Gemma 3**.
   - Speculation arose that the download size increase may be due to **CLIP** being used for vision, potentially being called from a separate file, increasing the overall memory footprint.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepHermes 3 Converts to MLX**: The model [mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit](https://huggingface.co/mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit) was converted to MLX format from [NousResearch/DeepHermes-3-Mistral-24B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview) using mlx-lm version **0.21.1**.
   - This conversion allows for efficient use on Apple Silicon and other MLX-compatible devices.
- **Deep Dive into VLLM Arguments for Hermes 3**: Members are sharing different configurations to get **vllm** working correctly with **Hermes-3-Llama-3.1-70B-FP8**, including suggestions like adding `--enable-auto-tool-choice` and `--tool-call-parser` for Hermes 3 70B.
   - One member noted the need for ``<tool_call>`` and ``</tool_call>`` tags in the tokenizer, which are present in **Hermes 3 models** but not necessarily in **DeepHermes**.
- **Vultr Announces Inference Pricing**: A member from Vultr shared the official pricing for their inference API, which includes **$10 for 50 million output tokens** initially, then **2 cents per million output tokens** after, accessible via an OpenAI-compatible endpoint at [https://api.vultrinference.com/](https://api.vultrinference.com/)
   - This pricing is a result of purchasing *an absurd amount of gh200s* and needing to do something with them, according to a member.
- **Dynamic LoRAs Docking into VLLM**: Members discussed the possibility of hosting dynamic **LoRAs** with **vllm** for various use cases, like up-to-date coding styles, referencing the [vLLM documentation](https://docs.vllm.ai/en/stable/features/lora.html#serving-lora-adapters).
   - It was suggested to let users pass in their huggingface repo IDs for the **LoRAs** and supply them into the **VLLM serve command cli args**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Astro Clients gear up for MCP Integration**: A member plans to use **MCP** for their **Astro** clients, using **AWS API Gateway** with each **MCP** server as a **Lambda** function, leveraging the MCP bridge with the **SSE gateway**.
   - The goal is to enable MCP usage specifically for customers and explore adding MCP servers to a single project for client visibility.
- **Decoding MCP Server Architecture**: A member inquired how clients like **Cursor** and **Cline**, which keep **MCP** servers on the client side, communicate with the backend.
   - The discussion involved the architecture and communication methods used by these clients but was redirected to a more specific channel for detailed information.
- **Smart Proxy Server Converts to Agentic MCP**: A smart proxy **MCP** server converts standard **MCP** servers with many tools into one with a single tool via its own **LLM**, effectively a sub-agent approach using *vector tool calling*.
   - The **OpenAI Swarm framework** follows a similar process of assigning a subset of tools to individual agents, now rebranded as **openai-agents** by OpenAI.
- **Debugger Uses MCP Server to Debug Webpages**: A member shared a debugger project, **chrome-debug-mcp** ([https://github.com/robertheadley/chrome-debug-mcp](https://github.com/robertheadley/chrome-debug-mcp)), that uses **MCP** to debug webpages with **LLMs**, originally built with **Puppeteer**.
   - The project has been ported to **Playwright**, with the updated GitHub repository pending after further testing.
- **MCP Hub Concept Simplifies Server Management**: To enhance enterprise adoption of **MCP**, a member created an **MCP Hub** concept featuring a dashboard for simplified server connections, access control, and visibility across **MCP** servers, as demoed in [this video](https://drive.google.com/file/d/1wkWSSGGbqVQavop26svmrryee4sx3NKz/view?usp=sharing).
   - The hub aims to address concerns about managing multiple MCP servers and permissions in enterprise settings.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek Confiscates Employee Passports**: The owner of **DeepSeek** reportedly asked R&D staff to surrender their passports to prevent foreign travel, according to [Amir on Twitter](https://fxtwitter.com/amir/status/1900583042659541477).
   - Members debated whether this would lead to more open source work from **DeepSeek**, or if the US might adopt similar measures.
- **SF Compute H100 Prices Shock the Market**: A member pointed out that [SF Compute](https://sfcompute.com/) offers surprisingly low prices for **H100s**, especially for short-term rentals, advertising **128 H100s** available for hourly use.
   - San Francisco Compute Company is [launching soon an additional **2,000 H100s**](https://x.com/evanjconrad/status/1884361612766896510) and runs a market for large-scale, vetted **H100 clusters**, while also sporting a [simple but powerful CLI](https://docs.sfcompute.com).
- **Gemma 3 License Raises Red Flags**: A recent TechCrunch article highlighted concerns over model licenses, particularly [Google's Gemma 3](https://techcrunch.com/2025/03/14/open-model-licenses-often-carry-concerning-restrictions/).
   - The article noted that while **Gemma 3's license** is efficient, its restrictive and inconsistent terms could pose risks for commercial applications.
- **User Data Privacy is Under Siege**: A member reported their frustration with individuals discovering their phone number online and making unsolicited requests, such as *"hey nato, can you post-train my llama2 model? ty"*.
   - They speculate that extensions or paid services are the source and are seeking methods to remove their data from sites like [Xeophon](https://x.com/alexalbert__/status/1900592059364634973).
- **Math-500 Sampling Validated**: In response to a question about seemingly random sampling in Qwen's [github repo](https://github.com/QwenLM/QwQ) evaluation scripts, it was confirmed that *apparently* sampling is random.
   - Members cited [Lightman et al 2023](https://cdn.discordapp.com/attachments/1179128538679488533/1350170687460868186/image.png?ex=67d5c3f0&is=67d47270&hm=6c771f09d27b7bad57e711e55ed2b111ac29af6a6485feb2c89103757f0771de&) and that long context evals and answer extraction is a headache and that **Math 500 is very well correlated**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Cohere Commands Attention with 111B Model**: **Cohere** launched [Command A](https://openrouter.ai/cohere/command-a), a new **open-weights 111B parameter model** boasting a **256k context window**, with a focus on agentic, multilingual, and coding applications.
   - This model is designed to deliver high performance in various use cases.
- **AI21 Jamba Jams with New Models**: **AI21** released [Jamba 1.6 Large](https://openrouter.ai/ai21/jamba-1.6-large) with **94 billion active parameters** and a **256K token context window**, alongside [Jamba 1.6 Mini](https://openrouter.ai/ai21/jamba-1.6-mini), featuring **12 billion active parameters**.
   - Both models now support structured JSON output and tool-use.
- **Gemma Gems Gleam for Free**: All variations of **Gemma 3** are available for free: [Gemma 3 12B](https://openrouter.ai/google/gemma-3-12b-it:free) which introduces multimodality, supporting vision-language input and text outputs and handles context windows up to **128k tokens**.
   - The model understands over **140 languages**, and also features [Gemma 3 4B](https://openrouter.ai/google/gemma-3-4b-it:free) and [Gemma 3 1B](https://openrouter.ai/google/gemma-3-1b-it:free) models.
- **Anthropic API Anomaly Averted**: **Anthropic** reported an incident of elevated errors for requests to **Claude 3.7 Sonnet**, with updates posted on their [status page](https://status.anthropic.com/incidents/qtxnlg9yrwqv).
   - The incident has been resolved.
- **Chess Tournament Pits AI Models Against Each Other**: An AI chess tournament, accessible [here](https://dubesor.de/chess/tournament), pits **15 models** against each other using standard chess notations for board state, game history, and legal moves.
   - The models are fed information about the board state, the game history, and a list of legal moves.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Go Wins at Practical Porting**: Members debated the utility of **Go** vs **Rust** for porting, explaining that porting to **Go** function-by-function allows for exact behavior parity, avoiding rewriting code for years.
   - While *Rust is faster and more efficient*, a member pointed out that *Golang is really ergonomic to develop in*, particularly for distributed, async, or networked applications.
- **DeepSeek Hype Suspicions Sparked**: Some members argued that the hype around **DeepSeek** is engineered and that their models are simplified, likening the comparison to **frontier AI models** *comparing Ananas with Apple*.
   - Others defended **DeepSeek**, claiming their *crazy engineers* developed a filesystem *faster than life*.
- **OLMo 2 32B Fully Open**: **OLMo 2 32B** launched as the [first fully-open model](https://allenai.org/blog/olmo2) to outperform **GPT3.5-Turbo** and **GPT-4o mini** on academic benchmarks.
   - It is claimed to be comparable to leading open-weight models while only costing one third of the training cost of **Qwen 2.5 32B**.
- **ChatGPT is Overrated, Claude Preferred**: One member expressed that **ChatGPT** is overrated because *it actually doesn't solve problems that I need solved*, preferring **Mistral Small 24B**, **QwQ 32B**, and **Claude 3.7 Sonnet**.
   - Another user shared, *I've had better luck getting what I want from Claude*, and that it *seems better at understanding intention and motivation for whatever reason*.
- **Grok 3 Writes Professional Code**: Members debated code generation qualities, highlighting that **OpenAI** models often generate legacy code, while **Mistral** can refactor it into more modern code.
   - It was also noted that **Grok 3** generates code that *looks like a professional programmer wrote it*, while in **VSCode**, one member prefers using **Amazon Q** over **Copilot**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Speech-to-Speech Models Spark Quest**: A member is actively seeking **speech-to-speech generation** models that focus on conversational speech, distinguishing them from multimodal models like **OpenAI Realtime API** or **Sesame AI**.
   - Two potential models were identified: [Moshi](https://github.com/kyutai-labs/moshi) from **Kyutai Labs** and [Hertz-dev](https://github.com/Standard-Intelligence/hertz-dev) from **Standard-Intelligence**.
- **Block Diffusion Bridges Autoregressive and Diffusion**: The **Block Diffusion** model, detailed in an **ICLR 2025 Oral** presentation, combines autoregressive and diffusion language model benefits, offering **high quality**, **arbitrary-length generation**, **KV caching**, and **parallelizable** processing.
   - Code can be found on [GitHub](https://github.com/kuleshov-group/BD3-LMs) and [HuggingFace](https://huggingface.co/collections/kuleshov-group/BD3-LMs-67be95f81b96b15fec50d53f).
- **Triton bitpacking gets Huge Boost**: **Bitpacking** in **Triton** achieved significant speed-ups versus the Pytorch implementation on the 4090, achieving **98x** speedup for **32-bit packing** and **26x** for **8-bit packing**.
   - Re-packing a **Llama3-8B** model time was reduced from **49 sec -> 1.6 sec** using the new bitpacking implementation, with code available on [GitHub](https://github.com/mobiusml/gemlite/blob/master/gemlite/bitpack.py#L59-L133).
- **Gemma3 Gains Traction in vLLM and LigerKernel**: Members discussed adding **Gemma 3** support to **vLLM**, referencing [this GitHub issue](https://github.com/vllm-project/vllm/issues/14696), while a member has started drafting an implementation of **Gemma3** into **LigerKernel**, and shared a [link to the pull request](https://github.com/linkedin/Liger-Kernel/pull/606).
   - According to the pull request, **Gemma3** has high similarities to **Gemma2** with some differences in **RMSNorm Calls**.
- **GRPO Gains Popularity for LLM Training**: Members discussed how **Group Relative Policy Optimization (GRPO)** has become popular for Reinforcement Learning in Large Language Models, referencing [the DeepSeek-R1 paper]().
   - A blog post from oxen.ai on [GRPO VRAM requirements](https://www.oxen.ai/blog/grpo-vram-requirements-for-the-gpu-poor) was shared, noting its effectiveness in training.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Intelligence Declines Spark Debate**: Discussion sparked from a [Financial Times article](https://www.ft.com/content/link-to-ft-article) that average intelligence is dropping in developed countries, citing increased reports of **cognitive challenges** and declining performance in **reasoning and problem-solving**.
   - One member theorized this could be due to technology, especially **smartphones** and **social media**, leading to outsourcing of thinking, however the graphics only showed the years really before **ChatGPT** became a thing.
- **Is Tech the Culprit for Cognitive Decline?**: Members debated potential causes for cognitive decline, including **technology's influence**, **immigration**, and **fluoridated water**.
   - One member pointed out that the rates of cognitive challenges were steadily increasing since the 1990s, and a sudden acceleration from around 2012.
- **DeepSeek V3 Distilled from OpenAI Models**: The discussion covers that **Deepseek V3 (the instruct version)** was likely distilled from **OpenAI models**.
   - One member notes that *even OpenAI unofficially supports distilling their models, they just don't seem to like it when Deepseek does it*.
- **Claude Sonnet 3.7 Dominates in Coding Tasks**: A member now uses **Claude Sonnet 3.7** exclusively for coding, finding **ChatGPT** lagging behind.
   - In related news, a member stated that the **o3-mini-high** model is better than **o1**.
- **Food Additives Fuel Mental Slowdown**: Members discuss that the availability and consumption of **ultra-processed foods (UPFs)** has increased worldwide and represents nowadays **50–60%** of the daily energy intake in some high-income countries, and is linked to cognitive decline
   - Another member mentions Multinational corporations such as **Nestlé** that operate in many countries produce and distribute worldwide, it seems understandable how different additives or changes made to these products in one of these corporations can have a worldwide impact.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Gemini 2.0 Deep Research Joins NotebookLM?**: Members are exploring pairing **Gemini 2.0 Deep Research** with **NotebookLM** for enhanced documentation handling.
   - The community questioned if **Deep Research** might eventually supersede **NotebookLM** in functionality.
- **NotebookLM Inspires African Project Ecokham**: A member from Africa reported using **NotebookLM** to connect thoughts, edit roadmaps, and generate audio for his project, **Ecokham**.
   - He expressed gratitude for **NotebookLM**'s contribution to inspiring his team.
- **NotebookLM Prototyping PhytoIntelligence Framework**: A member is leveraging **NotebookLM** to organize notes and prototype the **PhytoIntelligence framework** for autonomous nutraceutical design, with the aim of mitigating diagnostic challenges.
   - The user acknowledged Google for the tool's capabilities.
- **Users Demand Image & Table Savvy in NotebookLM**: Users are requesting **image and table recognition** in NotebookLM, complaining that the current state feels incomplete because of the need to constantly reopen source files and dig through Google Sheets; one user even shared [a relevant cat GIF](https://tenor.com/view/cat-wait-im-goated-pineapple-gif-1866711197257641569).
   - The community emphasized that images are worth a *"thousand words"* and the clearest data is often found in tables.
- **Mobile App still not here for NotebookLM**: Users are actively requesting a **mobile app version** of NotebookLM for improved accessibility.
   - The community feels a mobile version is *"still not yet coming up"*.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Google Gemini and Vertex AI Merge in LlamaIndex!**: The `@googleai` integration unifies **Google Gemini** and **Google Vertex AI** with streaming, async, multi-modal, and structured prediction support, even supporting images, detailed in [this Tweet](https://twitter.com/llama_index/status/1900590246070476929).
   - This integration simplifies building applications leveraging Google's latest models.
- **LlamaIndex Perks Debated**: A member sought clarity on **LlamaIndex**'s advantages over **Langchain** for building applications.
   - The inquiry did not lead to a conclusive discussion within the provided context.
- **OpenAI's Delta Event Absence Probed**: A member questioned why **OpenAI** models do not emit delta events for tool calling, observing the events are emitted but empty.
   - The consensus was that tool calling cannot be a stream because the LLM needs the full tool response to generate the subsequent response, advocating for a DIY approach.
- **API for Agentic RAG Apps Questioned**: There was a question on whether any **API exists specifically for building Agentic RAG applications** to streamline development and management.
   - The conversation mentioned that multiple constructs are available in LlamaIndex but lacked a clear, definitive guide.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Gemma 3 12B Edges Out Qwen in IQ Test**: A user reported that the **Gemma 3 12B model** outperformed **Qwen 14B** and **Qwen 32B** in terms of intelligence on their personal computer.
   - This was tested by asking questions in multiple languages; **Gemma 3** and **DeepSeek R1** consistently provided correct answers in the same language as the question.
- **Gemma 3 Needs New GPT4All Support**: Users noted that **GPT4All** may require updates to fully support **Gemma 3 12B** because its architecture differs from **Gemma 2**.
   - Specifically, **Gemma 3** needs an *mmproj* file to work with **GPT4All**, highlighting the challenges of quickly adapting to new AI model developments.
- **Freezing Water Tests AI Knowledge**: When queried about freezing water, **DeepSeek-R1** incorrectly predicted that jars would break, while **Gemma-3-12b** accurately described the shattering effect due to water expansion.
   - This demonstrates the models' varying levels of understanding of basic physics, indicating the diverse reasoning capabilities across different architectures.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Explicit Feedback Flows into Refine**: A member requested the reintroduction of **explicit feedback** into `dspy.Refine`, similar to `dspy.Suggest` to enhance debugging and understanding.
   - The member emphasized the value of **explicit feedback** for identifying areas needing improvement.
- **Manual Feedback Makes its Mark on Refine**: The team announced the addition of **manual feedback** into `Refine`.
   - The implementation involves including feedback in the **return value of the reward function** as a `dspy.Prediction` object, containing both a score and feedback.
- **Reward Function Returns Feedback**: A team member inquired about the feasibility of integrating feedback as part of the **return value of the reward_fn**.
   - The user responded *affirmatively*, expressing their gratitude.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command A Debuts on OpenRouter**: **Cohere's Command A**, a **111B** parameter open-weights model boasting a **256k** context window, is now accessible on [OpenRouter](https://openrouter.ai/cohere/command-a).
   - The model aims to deliver high performance across agentic, multilingual, and coding applications, setting a new standard for open-weight models.
- **Command A Flunks Prime Number Test**: Users discovered a peculiar bug in **Command A**: when queried about prime numbers whose digits sum to 15, the model either provides an incorrect response or gets stuck in an infinite loop.
   - This unexpected behavior highlights potential gaps in the model's mathematical reasoning capabilities.
- **Local API struggles with Command A**: Users are encountering performance bottlenecks when running **Command A** locally, reporting that even with sufficient VRAM, the model doesn't achieve acceptable speeds without patching the modeling in **ITOr** or using the **APIs**.
   - This suggests that optimizing **Command A** for local deployment may require further work to enhance its efficiency.
- **Cohere unveils Compatibility base_url**: A member suggests to use the [Cohere Compatibility API](https://api.cohere.com/compatibility/v1/chat/completions).
   - They recommend utilizing base_url for integration.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Discord Account Impersonation Alert**: A member reported being messaged by a **scam account** impersonating another user, and the impersonated user confirmed that *"That is not me. This is my only account"*.
   - The **fake account** *caroline_frascaa* was reported to Discord and banned from the server after a user posted [a screenshot of the fake account](https://cdn.discordapp.com/attachments/1098765954302873621/1350124411906293851/Impersonating_Account.png?ex=67d598d7&is=67d44757&hm=bd212e9e154251a202378828ccf61282fd69df840ade2eb535738fc7d7e248cb&).
- **Mojo stdlib Uses Discussed**: The use of some feature in the **Mojo stdlib** was mentioned by *soracc* in #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/).
   - The user mentioned it is used in `base64`.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Self-Reflection Needs External Evaluation**: A member inquired about statements on **self-evaluation** from the first lecture in contrast to the second, suggesting a contradiction regarding the role of external feedback.
   - The first lecture emphasized that **self-reflection** and **self-refinement** benefit from good external evaluation, while **self-correction** without oracle feedback can degrade reasoning performance.
- **Clarification on Self-Evaluation in Lectures 1 & 2 Sought**: A user is seeking clarification on the apparent contradiction between the lectures regarding **self-evaluation**.
   - They noted the emphasis on **self-evaluation and improvement** in the second lecture, while the first lecture highlighted the importance of external evaluation and the potential harm of self-correction without oracle feedback.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Vertex Preps for Version 1.6**: **Version 1.6** is not yet available on **Vertex**, but is slated to arrive in the near future.
   - It will also be available on other platforms like **AWS** for broader access.
- **AWS Soon to Host 1.6**: Version **1.6** will be available on platforms like **AWS** in the near future, expanding its reach.
   - This development aims to allow **AWS** customers access to the new features.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1350033025223622667)** (301 messages🔥🔥): 

> `Gemma 3 Support in Unsloth, Multi-GPU Training, Dynamic Quantization vs GGUF, GRPO and Reasoning, Vision Models` 


- **Unsloth Unleashes Gemma 3 Support**: Unsloth now supports **Gemma 3**, including full fine-tuning and 8-bit, with almost any model like **Mixtral, Cohere, Granite**.
   - According to a tweet, optimizations in Unsloth led to **10% less VRAM usage** and a **10% speedup boost for 4-bit**, plus fixes and improvements for **Windows support** and **GGUF conversions**.
- **Multi-GPU Support Still on the Horizon**: Despite user interest, Unsloth currently **doesn't natively support multi-GPU training** in the free version.
   - However, there is speculation that it is possible by deconstructing the components to your training code (FastLanguageModel), and there is the imminent release of the **AGPL3 multi-GPU** and the **Enterprise** version.
- **Dynamic Quants Duel GGUF in Quality**: There's ongoing discussion about comparing **dynamic quantization** with **GGUF** models, especially regarding the trade-offs between size and quality.
   - Unsloth's dynamic quants for **Phi-4** are on the [Hugging Face leaderboard](https://unsloth.ai/blog/dynamic-4bit), but a direct comparison with GGUF benchmarks is anticipated to clarify performance at different bit widths.
- **GRPO Powers Reasoning Prowess**: The team mentioned that **GRPO** (Guiding Preference Optimization) is *coming next week* along with new notebooks.
   - They will have a GRPO notebook; and they stated, *only if you let it reason about the rules first a la GRPO*.
- **Vision Models Get the Unsloth Treatment**: Unsloth has implemented the *train on completions* feature and also resizing of images for Vision Language Models, a highly demanded feature, to reduce OOM.
   - The models now *auto resize images which stops OOMs and also allows truncating sequence lengths.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-gemma-3-effectively">Tutorial: How to Run Gemma 3 effectively | Unsloth Documentation</a>: How to run Gemma 3 effectively with our GGUFs on llama.cpp, Ollama, Open WebUI, LM Studio.</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks">Unsloth Benchmarks | Unsloth Documentation</a>: Want to know how fast Unsloth is?</li><li><a href="https://unsloth.ai/blog/reintroducing">Re-introducing Unsloth</a>: In celebration of us being the #1 Trending GitHub repo of the day, we reflect on our journey and contributions to the open-source community.</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit Quantization</a>: Unsloth&#x27;s Dynamic 4-bit Quants selectively avoids quantizing certain parameters. This greatly increases accuracy while maintaining similar VRAM use to BnB 4bit.</li><li><a href="https://huggingface.co/docs/bitsandbytes/main/en/explanations/optimizers#paged-optimizers">8-bit optimizers</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslot">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth update: Mistral support + more</a>: We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...</li><li><a href="https://x.com/danielhanchen/status/1900592202621087944">Tweet from Daniel Han (@danielhanchen)</a>: Excited to share that @UnslothAI now supports:• Full fine-tuning + 8bit• Nearly any model like Mixtral, Cohere, Granite, Gemma 3• No more OOMs for vision finetuning!Blogpost with details: https://unsl...</li><li><a href="https://github.com/unslothai/unsloth/issues/2009)">unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/issues/36683">AttributeError: &#39;Gemma3Config&#39; object has no attribute &#39;vocab_size&#39; · Issue #36683 · huggingface/transformers</a>: System Info v4.50.0.dev0 Who can help? @ArthurZucker @LysandreJik @xenova Information The official example scripts My own modified scripts Tasks An officially supported task in the examples folder ...</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>: no description found</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/12343#issuecomment-2718131134">llama : Add Gemma 3 support (+ experimental vision capability) by ngxson · Pull Request #12343 · ggml-org/llama.cpp</a>: Model infoOfficial model page: https://ai.google.dev/gemma/docs/corePre-quantized GGUF: https://huggingface.co/collections/ggml-org/gemma-3-67d126315ac810df1ad9e913Available sizes: 1B, 4B, 12B,...</li><li><a href="https://github.com/vllm-project/vllm/pull/14660">[Model] Add support for Gemma 3 by WoosukKwon · Pull Request #14660 · vllm-project/vllm</a>: This PR adds the support for Gemma 3, an open-source vision-language model from Google.NOTE:The PR doesn&amp;#39;t implement the pan-and-scan pre-processing algorithm. It will be implemented by a fo.....
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1350168485648662579)** (1 messages): 

> `Gemma 3 models, Unsloth support for models, GRPO for reasoning models, QwQ-32B bugfixes, New model uploads` 


- **Google's Gemma 3 Integrated with Unsloth**: Google's new **Gemma 3** model is now supported in Unsloth with a [blog post](https://unsloth.ai/blog/gemma3) and a [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb) provided.
   - All Gemma 3 model uploads are available on [Hugging Face](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b), including versions optimized for full finetuning, 8-bit, and pretraining.
- **Unsloth Boosts Gemma 3 Finetuning Speed**: Unsloth accelerates **Gemma 3 (12B) finetuning by 1.6x**, reduces **VRAM usage by 60%**, and extends context length by **6x** compared to environments using Flash Attention 2 on a 48GB GPU.
   - The team fixed issues with training **Gemma 3** and uploaded all versions including 2-8 bit GGUFs, dynamic 4-bit, and 16-bit versions.
- **GRPO Enables Extended Context with Reduced VRAM**: Unsloth now supports **10x longer context with 90% less VRAM** using GRPO (Generalized Rank Position Optimization), detailed in a [blog post](https://unsloth.ai/blog/grpo) and [tutorial](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo).
   - This enhancement is designed for reasoning models, offering significant memory savings and expanded context windows.
- **QwQ-32B Model Gets a Makeover**: Bugfixes have been implemented for the **QwQ-32B model**, as highlighted in a [blog post](https://unsloth.ai/blog/qwq-32b) with corresponding [model uploads](https://huggingface.co/collections/unsloth/qwen-qwq-32b-collection-676b3b29c20c09a8c71a6235).
   - These fixes improve the model's stability and performance, ensuring a smoother user experience.
- **Fresh Model Uploads Hit Hugging Face**: New model uploads include **Gemma 3 GGUF** variants (1B, 4B, 12B, 27B), **Gemma 3 Dynamic 4-bit** versions, **QwQ-32B** variants, and **Phi-4-mini** versions, all available on [Hugging Face](https://huggingface.co/collections).
   - These models cater to various hardware configurations and performance needs, expanding the accessibility of state-of-the-art models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb)">Google Colab</a>: no description found</li><li><a href="https://unsloth.ai/blog/gemma3#everything)">Fine-tune Gemma 3 with Unsloth</a>: Gemma 3, Google&#x27;s new multimodal models.Fine-tune &amp; Run them with Unsloth! Gemma 3 comes in 1B, 4B, 12B and 27B sizes.</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo)">Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models)">Unsloth Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1350054985873887252)** (5 messages): 

> `Gemma 3, Ollama, Phi Vision, GGUFs vision` 


- **Gemma 3 Image Compatibility with Ollama Questioned**: A member inquired whether **Gemma 3** works with images via **Ollama**, similar to **Phi Vision**.
   - Another user clarified that their **Gemma 3 GGUFs vision** component functions on all engines except **Ollama**, including **LM Studio** and **llama.cpp**; this is *likely* due to **llama-server** lacking vision support *yet*.
- **Ollama lacks vision support**: The **Gemma 3 GGUFs vision** component functions on all engines except **Ollama**, including **LM Studio** and **llama.cpp**.
   - This is *likely* due to **llama-server** lacking vision support *yet*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1350042947114111016)** (51 messages🔥): 

> `Gemma-3 GGUF and Ollama, Llama 3.2 inference cancellation, Phi-4-mini support, Gemma finetuning error, TurboML Continual Pre-Training` 


- ****Gemma-3 GGUF vision fails with Ollama****: Support for **Gemma-3 GGUF** models with vision components in **Ollama** is currently non-functional due to a widespread issue affecting all uploaders.
   - The recommendation is to use **Ollama's original** Gemma model for text, until the issue can be debugged.
- ****Llama 3.2 inference needs cancellation method****: A user inquired about the possibility of cancelling long-running inferences with **Llama 3.2** models in **Unsloth** without unloading the model from memory.
   - The user asked if they could stop it after a certain amount of time in a timeout loop.
- ****Phi-4-mini receives new Unsloth update****: After upgrading to the latest **Unsloth** version, users should now be able to use **Phi-4-mini** models, which was previously causing a RuntimeError.
   - The user reported that **Phi-4-mini** works fine, but `unsloth/Phi-4-mini-instruct` gives a `rope_scaling` error.
- ****Gemma finetuning error requires unsloth update****: Users reported encountering an error (`only Tensors of floating point dtype can require gradients`) when fine-tuning **Gemma models** after adding new tokens and training new embeddings, particularly during the evaluation phase.
   - Upgrading to the latest version of **Unsloth** is recommended, though some users have reported that the issue persists even after the update.
- ****TurboML seeks guidance on Continual Pre-Training dataset format****: A member with a new framework - **TurboML** - sought advice on the correct dataset format for **Continual Pre-Training (CPT)**, specifically for Sentence Completion and SFT tasks.
   - They referenced a [Unsloth notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu) and inquired about the placement of the EOS token.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/m">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://github.com/huggi">huggi - Overview</a>: huggi has 2 repositories available. Follow their code on GitHub.</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">Finetuning from Last Checkpoint | Unsloth Documentation</a>: Checkpointing allows you to save your finetuning progress so you can pause it and then continue.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1350123121059037237)** (9 messages🔥): 

> `Gemma SFT, Maximum Context Length, Memory Usage Calculation` 


- **Gemma's SFT VRAM Needs Debated**: Members discussed **Gemma's** VRAM usage in SFT, suggesting it may require more VRAM than **Qwen 2.5** under similar training conditions, though specific numbers were not given.
   - One shared a [Qwen2_VL Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_%25287B%2529-Vision.ipynb) for images.
- **Max Context Length is a Limitation, Not a Hyperparameter**: Members clarified that the maximum context length is not a hyperparameter of a model but rather a **limitation** based on available memory.
   - The longer the context, the more memory is needed to process it, but exact calculation is not solely based on model size.
- **Estimating Memory Usage for LLMs**: It was mentioned that the amount of memory needed to process context depends on the model architecture, with different layers requiring varying amounts of memory.
   - Links were shared to approximate memory requirements ([Substratus AI blog](https://www.substratus.ai/blog/calculating-gpu-memory-for-llm) and [Hugging Face space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/m">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7B)-Vision.ipynb#scrollTo=idAEIeSQ3xdS">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1350033127891931137)** (263 messages🔥🔥): 

> `Cursor performance issues on Linux and Windows, Issues with Claude 3.7, Custom modes in Cursor, Gemini API key issues, Cursor agent spawning terminals` 


- ****Cursor Lags on Linux and Windows****: Users report **Cursor lagging** and **freezing** on both **Linux** and **Windows**, even with powerful hardware, especially after recent updates like **0.47.4** ([download link](https://downloads.cursor.com/production/8f8a2000673d2c48f6cac5eea2f3f9f2ed5e4ec2/win32/x64/user-setup/CursorUserSetup-x64-0.47.4.exe)).
   - One user mentioned that on Linux, after **20-30 messages**, the UI freezes for seconds, while another cited that they have "lags all the time" on Windows with a beefed up laptop and version **3.7**.
- ****Claude 3.7 Underperforms and Ignores Global Rules****: Users are experiencing issues with **Claude 3.7**, reporting it as *dumb as bricks* after upgrading to **0.47.4** and noting that it costs double the credits to use.
   - Some members have reported that **Sonnet 3.7** ignores global rules, even when explicitly asked to output the rules being used, with one suggesting to *put 'be a good boy' in your prompt and it will fix anything*.
- ****Cursor Agent Triggers Terminal Tsunami****: Multiple users find the Cursor agent excessively spawning terminals, which is seen as annoying, especially when it restarts already running servers.
   - A member suggested that this behavior should be built-in or that users should just write the terminal commands themselves if they don't like it.
- ****V0 Sparks Joy for Quick UI Prototyping****: Some users find **v0** better for front-end focused prototyping, allowing for UI design with subframes, similar to Figma, before importing to Cursor.
   - One user noted, *it's much better to build prototype and layout (better front end) imo then import locally to cursor*, however some prefer Cursor due to v0's credit-based system and less creative control.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cursor.com/context/ignore-files">Cursor – Ignore Files</a>: no description found</li><li><a href="https://github.com/kcolemangt/llm-router">GitHub - kcolemangt/llm-router: Access models from OpenAI, Groq, local Ollama, and others by setting llm-router as Cursor&#39;s Base URL</a>: Access models from OpenAI, Groq, local Ollama, and others by setting llm-router as Cursor&#39;s Base URL - kcolemangt/llm-router</li><li><a href="https://x.com/kregenrek/status/1899941361908146430">Tweet from Kevin Kern (@kregenrek)</a>: Sonnet 3.7 and Cursor:I got it somewhat under control by following these rules.And I still recommend - Edit Mode - Sonnet 3.5Quoting Jame (@Inveeest) @kregenrek Just noticed that you&#39;re the creato...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1350119670551679039)** (2 messages): 

> `LM Studio, SMILES string encoding, ChemDraw` 


- **Advocating for LM Studio Support**: A member suggested that users with **LM Studio** related problems might find more targeted assistance in the **LM Studio Discord**.
- **Seeking SMILES String Encoders for Stereoisomers**: A member inquired about existing models or architectures capable of encoding a **SMILES string** into various **stereoisomers** or encoding a **ChemDraw** input, with the aim of enabling chemical descriptor extraction.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1350055159018815500)** (255 messages🔥🔥): 

> `Diffusion Models for Generative Tasks, Search-R1: RL for Autonomous Search Query Generation, Spectral Analysis of Latent Spaces, Noise Sensitivity in Diffusion Models, Inductive Moment Matching (IMM) for Fast Sampling` 


- ****Diffusion Models** Emerge for Generative Tasks**: A member shared a [Nature article](https://www.nature.com/articles/s41467-024-54281-3) highlighting the advancements in **diffusion models (DMs)**, noting their capability in modeling complex data distributions and generating realistic samples for images, videos, audio, and 3D scenes.
   - The article describes how **diffusion models** have become state-of-the-art in generative tasks.
- ****Search-R1** Learns Autonomous Search via RL**: A member shared a [paper](https://arxiv.org/abs/2503.09516) introducing **Search-R1**, an extension of the **DeepSeek-R1** model that learns to generate search queries during reasoning using reinforcement learning (**RL**).
   - The model optimizes LLM rollouts with multi-turn search interactions, using retrieved token masking for stable **RL** training.
- **Latent Space's Spectral Analysis Probed**: Members discussed spectral analysis on latent spaces of diffusion models, with one member noting that the model is resilient to perturbations in the initial noise, even at **t=0** (max noise).
   - Another member noted that there was nothing actionable in comparing the radially averaged power spectral density of images encoded with **flux VAE** vs **SDXL VAE**.
- **Noise Sensitivity Analysis Reveals Variance**: Members discussed how small changes in initial noise affect the output of diffusion models, with one member sharing a plot showing points where *small noise turns into a big difference*.
   - They observed brighter pixels indicating sensitive areas in the initial noise that cause significant changes in the output.
- ****Inductive Moment Matching (IMM)** Promises Speedy Sampling**: A member shared a [paper](https://arxiv.org/abs/2503.07565) on **Inductive Moment Matching (IMM)**, a new class of generative models for one- or few-step sampling with a single-stage training procedure, which is faster than diffusion models at inference.
   - Unlike distillation, **IMM** does not require pre-training initialization and optimization of two networks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.09516">Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning</a>: Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Retrieval augmentation and tool-use traini...</li><li><a href="https://m-arriola.com/bd3lms/">SOCIAL MEDIA TITLE TAG</a>: SOCIAL MEDIA DESCRIPTION TAG TAG</li><li><a href="https://arxiv.org/abs/2503.07565">Inductive Moment Matching</a>: Diffusion models and Flow Matching generate high-quality samples but are slow at inference, and distilling them into few-step models often leads to instability and extensive tuning. To resolve these t...</li><li><a href="https://www.nature.com/articles/s41467-024-54281-3">Dynamical regimes of diffusion models - Nature Communications</a>: Diffusion methods are widely used for generating data in AI applications. Here, authors show that optimally trained diffusion models exhibit three dynamical regimes: starting from pure noise, they rea...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1350041171766022178)** (195 messages🔥🔥): 

> `Ministral 8B, Exaone 8B, Jungle Chess AI, Stable Diffusion, Gemini 2.0` 


- **Mistral 8B and Exaone 8B Recommended**: For users seeking LLMs, members recommended using **Ministral 8B** or **Exaone 8B** at 4-bit quantization.
   - A user with an M4 Mac mini with 24 GB RAM inquired about expected tokens per second, but the exact performance remains speculative depending on hardware specs.
- **User Tries to Train Jungle Chess AI**: A user attempted to create an AI for Jungle Chess using o3-mini, but the AI failed to understand or comply with a bug report related to its alpha-beta algorithm as discussed in [this thread](https://chatgpt.com/share/67d436c6-e2bc-8006-a450-41edf4acfac9).
   - The user noted it could reach depth 6 but couldn't avoid a simple opening trap, whereas a depth of 3 or 4 should suffice, suggesting the [Chinese Engine](https://gitee.com/WZ403809264/animalcraftAI/releases) was better at that.
- **Chatbot Chess Championship Emerges!**: A user shared a [YouTube playlist](https://youtube.com/playlist?list=PLBRObSmbZluRddpWxbM_r-vOQjVegIQJC) titled *Chatbot Chess Championship 2025*, showcasing language models or chess engines playing chess.
   - Participants speculated whether the models were true language models or merely calling chess engines, and one person noted a language model made illegal moves.
- **User Searches for Stable Diffusion Model**: A user requested help finding a **Stable Diffusion v1.x model** fine-tuned on datasets other than LAION.
- **Looking for Gemini 2.0 Flash Open Source Model**: A member inquired about the existence of an open-source model similar to **Gemini 2.0 Flash** with text-plus-image-to-image capabilities for image editing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/playlist?list=PLBRObSmbZluRddpWxbM_r-vOQjVegIQJC">Chatbot Chess Championship 2025</a>: Hi</li><li><a href="https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.ScopeParamShapeError)">flax.errors package</a>: no description found</li><li><a href="https://gitee.com/WZ403809264/animalcraftAI/releases">animalcraftAI 发行版 - Gitee.com</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

ilyachua: Hi all. I am starting on the CV course from hugging face
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1350200457578020924)** (2 messages): 

> `Awesome Vibe Coding, mahimairaja/awesome-csm-1b` 


- **Awesome Vibe Coding List Arrives**: A curated list of tools, editors, and resources for **AI-assisted coding** has been released, called [Awesome Vibe Coding](https://github.com/filipecalegario/awesome-vibe-coding).
   - The list includes **AI-powered IDEs**, **browser-based tools**, **plugins**, **command line tools**, and the **latest news** on *vibe coding*.
- **CSM 1B Use Cases Curated**: A curated list of use cases built using **Sesame's CSM 1B** has been released, called [awesome-csm-1b](https://github.com/mahimairaja/awesome-csm-1b).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/filipecalegario/awesome-vibe-coding">GitHub - filipecalegario/awesome-vibe-coding: A curated list of vibe coding references, collaborating with AI to write code.</a>: A curated list of vibe coding references, collaborating with AI to write code. - filipecalegario/awesome-vibe-coding</li><li><a href="https://github.com/mahimairaja/awesome-csm-1b">GitHub - mahimairaja/awesome-csm-1b: List of curated use cases built using Sesame&#39;s CSM 1B</a>: List of curated use cases built using Sesame&#39;s CSM 1B - mahimairaja/awesome-csm-1b
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1350133225573453866)** (1 messages): 

> `generate_without_kv_cache function` 


- **Typo Spotted in Function Name**: A user pointed out a discrepancy where the function name in the article is **generate_without_kv_cache** but the function call used is **generate_with_kv_cache**.
   - No further discussion was provided.
- **Function Call Discrepancy**: The article mentions a function named **generate_without_kv_cache**, but the actual code uses **generate_with_kv_cache**, indicating a possible error.
   - This discrepancy could lead to confusion or incorrect usage if users copy the function call directly from the article.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1350141638709743667)** (2 messages): 

> `PaliGemma 2 Mix, smolVLM2, QwenVL, Llama 3.2 Multimodal` 


- **Google Drops PaliGemma 2 Mix Models**: Google released **PaliGemma 2 Mix**, a new family of versatile vision language models with three sizes (**3B**, **10B**, and **28B**) and resolutions of **224** and **448** that *can do vision language tasks with open-ended prompts* and understand documents ([blog post](https://huggingface.co/blog/paligemma2mix)).
- **Raschka Roots into Multimodal LLMs**: Sebastian Raschka explains how multimodal LLMs function, and reviews recent multimodal papers and models, including **Meta AI's Llama 3.2** ([blog post](https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html)).
- **SmolVLM2 Architecture Deep Dive**: Users interested in understanding **SOTA architectures** from the root should start with **CLIP** and **BLIP2** before moving to **smolVLM2** and **QwenVL**.
   - Alternatively, one can learn about the recent ones first and traverse the same tree backward.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/merve">merve (Merve Noyan)</a>: no description found</li><li><a href="https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html">Understanding Multimodal LLMs</a>: There has been a lot of new research on the multimodal LLM front, including the latest Llama 3.2 vision models, which employ diverse architectural strategies to integrate various data types like text ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1350037931112398900)** (25 messages🔥): 

> `SerpAPI Key Errors, Deep RL Course, Interactive IDEs for Agent Code, Image to Video Loops, Gemma3 Issues with SmolAgents` 


- **SerpAPI Keys Causing Headaches**: A member reported a potential error with their **SerpAPI key** and wondered if others were experiencing the same issue.
   - Another member clarified that users are required to supply their own **SerpAPI key**, which wasn't immediately clear from the course example.
- **Deep RL Course Sparks Questions**: Several members inquired about the Discord server's dedication to the **Deep RL course** and reported that the [Deep Reinforcement Learning Leaderboard](https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard) was not working.
   - One member mentioned they were also facing the same problem and had recently started the Deep RL course.
- **Interactive IDE Recommendations Sought**: A member asked for recommendations for an **interactive IDE** that provides suggestions for writing agents code.
   - Another member recommended **VS Code** as a capable and free option.
- **Image Looping Sparks Model Search**: A member requested recommendations for a model capable of turning a **1920x1080 image into a 3-5 second video loop**, running on an **H100**.
   - They noted difficulty finding options beyond **720p**.
- **Gemma3's Troubles with SmolAgents**: A member encountered errors while running **Gemma3** with **SmolAgents**, specifically related to code parsing and regex patterns, and linked to a potential fix on [GitHub](https://github.com/huggingface/smolagents/pull/883).
   - They solved the issue by increasing the **Ollama context length**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fspaces%2Fhuggingface-projects%2FDeep-Reinforcement-Learning-Leaderboard">Weiterleitungshinweis</a>: no description found</li><li><a href="https://open.spotify.com/playlist/4J61XoHr2CINqRA1DV0ga7">Party on, Wayne!</a>: Playlist · Music League · 17 items · 4 saves</li><li><a href="https://github.com/huggingface/smolagents/pull/883">Update code_agent.yaml to fix persistent SyntaxErrors by toadlyBroodle · Pull Request #883 · huggingface/smolagents</a>: fix the perpetual SyntaxErrors and Error in code parsing: The code blob is invalid, caused by CodeAgents adding ``` before the py code blocks
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1350042931012177941)** (213 messages🔥🔥): 

> `Complexity Extension issues, Kernel locking, Perplexity context window sizes, Grok 3 bugs, Gemini's deep research` 


- **Complexity Extension in Full Maintenance Mode**: Due to a new Perplexity layout breaking the [Complexity extension](https://github.com/danielcardeenas/perplexity-complexity), it is now in full maintenance mode.
   - The developer thanked users for their patience.
- **Debate over Kernel Locking for Security**: Some users suggested **locking down the kernel** for security, but others argued that this is impossible due to the **open-source nature** of Linux.
   - One user sarcastically quipped that *if the anticheat decides to make and enforce a custom Linux kernel build... at that point you might as well just use windows lol*.
- **Context Window Size still a Pain Point for Perplexity**: Users are pleading to **increase the context window size**, so they stop paying for ChatGPT.
   - One user stated they were willing to pay extra for a larger context window, due to the other features that Perplexity has that others don't: *ability to do unlimited research on 50 files at a time*, *best thing is spaces, where we can give custom instructions along with 50 knowledge uploadable files*, and *option to choose reasoning models*.
- **Grok 3 Plagued by Bugs on Launch**: Users reported that the newly launched **Grok AI** has many bugs.
   - Reported bugs are that *suddenly the chat stops working or breaks in middle*.
- **Gemini's new deep research is lacking**: Some users have tested the new **Gemini Deep Research** and found it weak compared to what **OpenAI** offers.
   - One user found it retained less context than the regular Gemini, even with search off.



**Link mentioned**: <a href="https://xkcd.com/1053/">Ten Thousand</a>: no description found

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1350051629554077772)** (3 messages): 

> `OpenAI custom agent, Airpods Live Translation, Anthropic CEO AI quit-button` 


- **OpenAI debuts Custom Agents**: Perplexity links to [OpenAI releases new custom agent](https://www.perplexity.ai/page/openai-releases-new-custom-age-0SxjY06OTReWo9OK_gjQCA).
   - No discussion, but a potentially interesting link for someone to check out.
- **Airpods to introduce Live Translation**: Perplexity links to [Airpods to introduce live translation](https://www.perplexity.ai/page/airpods-to-introduce-live-tran-UFQuA8yaRY..k0Qwm3MOZw).
   - No discussion, but a potentially interesting link for someone to check out.
- **Anthropic CEO Floats AI Quit-Button**: Perplexity links to [Anthropic CEO floats AI quit-button](https://www.perplexity.ai/page/anthropic-ceo-floats-ai-quit-b-BotCYKfST6GePBfE_Psp6w).
   - No discussion, but a potentially interesting link for someone to check out.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1350035269071540235)** (133 messages🔥🔥): 

> `Claude with Aider, Rust for Aider, Claude Desktop on Linux, Aider MCP Server, Anthropic Status` 


- ****Claude** and **Aider** Combine Forces for Coding!**: Members discussed using **Claude** with **Aider**, which *augments it with web search/URL scrapping and running bash script calling*, resulting in more powerful prompting capabilities.
   - One user highlighted that *each unique tool added to Claude unlocks a lot more than the sum of its parts*, especially when the model searches the internet for bugs.
- **Does **Rust** make **Aider** run faster?**: One user inquired about porting **Aider** to **C++** or **Rust** for faster file processing, particularly when loading large context files for **Gemini** models.
   - Others expressed skepticism, suggesting that the bottleneck remains with the **LLM API** and any improvements might not be *quantifiable*.
- ****Linux** users turn to github for **Claude Desktop** app**: Users shared instructions for getting the **Claude Desktop** app to work on **Linux**, as there isn't an official version.
   - One user referenced a [GitHub repo](https://github.com/aaddrick/claude-desktop-debian) providing **Debian-based** installation steps while another shared their edits to an **Arch Linux PKGBUILD**.
- **Aider MCP Server Readme Needs Help!**: Users discussed the **Aider MCP Server**, with one mentioning that another user's readme was *100x better*, referring to [this repo](https://github.com/sengokudaikon/aider-mcp-server).
   - However, another user humorously stated that they still *can't setup ur mcp* despite the readme's existence.
- **Anthropic 3.7 Sonnet Encounters hiccups**: Users reported *empty responses* from the **Claude 3.7 Sonnet** model, prompting checks of their **Anthropic** accounts.
   - The [Anthropic Status page](https://status.anthropic.com/) confirmed *elevated errors*, indicating an issue was under investigation and a fix was being implemented.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/recordings/tree-sitter-language-pack.html">Add language support via tree-sitter-language-pack</a>: aider is AI pair programming in your terminal</li><li><a href="https://asciinema.org/">Record and share your terminal sessions, the simple way - asciinema.org</a>: no description found</li><li><a href="https://aider.chat/docs/recordings/">Screen recordings</a>: Screen recordings of aider building aider.</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://github.com/sigoden/aichat">GitHub - sigoden/aichat: All-in-one LLM CLI tool featuring Shell Assistant, Chat-REPL, RAG, AI Tools &amp; Agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more.</a>: All-in-one LLM CLI tool featuring Shell Assistant, Chat-REPL, RAG, AI Tools &amp; Agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more. - sigoden/aichat</li><li><a href="https://github.com/sengokudaikon/aider-mcp-server">GitHub - sengokudaikon/aider-mcp-server</a>: Contribute to sengokudaikon/aider-mcp-server development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=NTh0hYbfpis">Claude using aider mcp</a>: No hype, just open source tools: github.com/robert-at-pretension-io/mcp</li><li><a href="https://github.com/aaddrick/claude-desktop-debian.git">GitHub - aaddrick/claude-desktop-debian: Claude Desktop for Debian-based Linux distributions</a>: Claude Desktop for Debian-based Linux distributions - aaddrick/claude-desktop-debian</li><li><a href="https://tenor.com/view/fedora-tipshat-mlady-melady-athiest-gif-7191305">Fedora Tipshat GIF - Fedora Tipshat Mlady - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1350039436615749694)** (44 messages🔥): 

> `DeepSeek models configuration, Aider's architect mode behavior, Modifying Aider's completion endpoint, Aider configuration files` 


- **DeepSeek Models Speak Too Much**: A user reported that **DeepSeek models** are generating excessive output, around **20-30 lines** of phrases, and inquired about setting a `thinking-tokens` value in the configuration.
   - It was noted that **20 lines** is pretty standard for the **R1 model**, and one user shared that they once waited **2 minutes** for the model to think on a **5 word prompt**.
- **Architect Mode Plans Infinitely**: A user experienced issues with **Aider's architect mode**, where the model would continuously plan without passing code changes to the editor, even after being prompted to *make the code change*.
   - It was suggested that the user may need to explicitly add files beforehand and/or to *list down what files and functions may be affected*.
- **Modify Aider API Calls For OpenWebUI**: A user inquired about modifying how **Aider** calls the completions endpoint to integrate with **OpenWebUI's** knowledge collections, which requires a `files` parameter with a collection ID, referencing [OpenWebUI API documentation](https://docs.openwebui.com/getting-started/api-endpoints/#using-a-knowledge-collection-in-chat-completions).
   - It was suggested to use the `extra_params` or `extra_body` configuration options to add the necessary parameters.
- **Global vs Local `ai-instructions.md` Files**: A user asked whether `ai-instructions.md` files should be placed in each project or if a single global file can be configured.
   - The response clarified that these files, like `conventions.md`, can be handled as per user preference, with a recommendation to use a global file for personal use and local files for project-specific conventions.
- **Configure OpenRouter API Key**: A user needed help to configure **Aider** with the **OpenRouter API key**.
   - A member showed the correct config format `api-key: - openrouter=sk-or-v1-...`


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>: How to configure aider with a yaml config file.</li><li><a href="https://docs.openwebui.com/getting-started/api-endpoints/#using-a-knowledge-collection-in-chat-completions">🔗 API Endpoints | Open WebUI</a>: This guide provides essential information on how to interact with the API endpoints effectively to achieve seamless integration and automation using our models. Please note that this is an experimenta...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1350034619554070560)** (17 messages🔥): 

> `OLMo 2 32B, AI Engineer Singapore 2025, AI Game Generation, Gemini DeepResearch with 2.0, Claude's Birthday` 


- ****OLMo 2 32B** beats **GPT 3.5** and **GPT 4o mini**!**: AI2 released [**OLMo 2 32B**](https://allenai.org/blog/olmo2-32B), a fully open-source model trained up to **6T tokens** using Tulu 3.1, outperforming **GPT3.5-Turbo** and **GPT-4o mini** on academic benchmarks.
   - It is claimed to require only one third of the cost of training **Qwen 2.5 32B** while reaching similar performance and is available in **7B**, **13B**, and **32B** parameter sizes.
- ****AI Engineer Singapore 2025** event announced**: The [AI Engineer Singapore 2025](https://lu.ma/aiesg?tk=fHwK70) event was announced, aiming to bridge the gap between cutting-edge **AI research** and practical engineering applications.
   - The event is organized by the team behind **AI Eng Summit**, **World's Fair**, **JSConf Asia**, and **GovTech Singapore**.
- **Vibe Coding: Creating Games Entirely with AI**: A developer created a multiplayer 3D game **100% with AI**, spending 20 hours and 20 euros, calling the concept *vibe coding*, and sharing the [guide](https://x.com/nicolaszu/status/1899931187398979890?s=46).
   - The game features realistic elements like hit impacts, smoke when damaged, and explosions on death, all generated via prompting in **Cursor** with no manual code edits.
- ****Gemini DeepResearch 2.0** is getting love**: Members are reporting very good results with the new **Gemini DeepResearch with 2.0** model.
   - One member noted that *it is great for company OSINT compared to chatGPT deep research because it refuses to answer questions much less*.
- **Happy Birthday, Claude!**: Members celebrated the [second birthday of **Claude**](https://x.com/alexalbert__/status/1900592059364634973?s=46).
   - This coincides with the birthday of **GPT-4** as well.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/joshwoodward/status/1900201110717214914?s=46">Tweet from Josh Woodward (@joshwoodward)</a>: Next batch of @NotebookLM updates rolling out:* Even smarter answers, powered by Gemini 2.0 Thinking* See citations in your notes, not just in the Q&A (top request)* Customize the sources used for mak...</li><li><a href="https://x.com/alexalbert__/status/1900592059364634973?s=46">Tweet from Alex Albert (@alexalbert__)</a>: Two years ago today we announced Claude to the world.Happy second birthday, Claude!</li><li><a href="https://x.com/natolambert/status/1900249099343192573?s=46">Tweet from Nathan Lambert (@natolambert)</a>: A very exciting day for open-source AI! We&#39;re releasing our biggest open source model yet -- OLMo 2 32B -- and it beats the latest GPT 3.5, GPT 4o mini, and leading open weight models like Qwen an...</li><li><a href="https://x.com/levelsio/status/1900235423667909041?s=46">Tweet from @levelsio (@levelsio)</a>: Portals from games to other games!Quoting Josua Sievers (@SieversJosua) I&#39;ve added the first portal!You can now beam into @levelsio game!Would be great to have a way to actually immediatly spawn a...</li><li><a href="https://x.com/nicolaszu/status/1899931187398979890?s=46">Tweet from Nicolas Zullo (@NicolasZu)</a>: It&#39;s out!! The ULTIMATE guide to vibe coding games.How did I do it? 20 hours. 500 prompts. 20 euros. That&#39;s all it took to make a multiplayer 3D game 100% with AI, 0 human code, not even a sma...</li><li><a href="https://allenai.org/blog/olmo2-32B">OLMo 2 32B: First fully open model to outperform GPT 3.5 and GPT 4o mini  | Ai2</a>: Introducing OLMo 2 32B, the most capable and largest model in the OLMo 2 family.</li><li><a href="https://lu.ma/aiesg?tk=fHwK70).">AI Engineer Singapore 2025 · Luma</a>: Join us for the AI Engineer Singapore 2025, the definitive industry-focused event designed to complement the research-focused International Conference on…
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1350226688595853403)** (3 messages): 

> `Snipd Podcast, AI Podcast App, Latent Space Podcast` 


- ****Snipd Podcast** Launch Outdoor!**: The Latent Space podcast released a new [Snipd podcast](https://x.com/latentspacepod/status/1900666708270215383) with Kevin Ben Smith about the **AI Podcast App** for Learning.
   - The podcast features a discussion about aidotengineer NYC, switching from **Finance to Tech**, how AI can help us get a lot more out of our podcast time, and dish the details on the **tech stack of Snipd app**.
- **Latent Space Podcast releases a Youtube Video**: Latent Space Podcast released their first ever **OUTDOOR podcast** on [Youtube](https://youtu.be/FNRO_SYx68Q).
   - The podcast features a discussion about @aidotengineer NYC, switching from **Finance to Tech**, how AI can help us get a lot more out of our podcast time, and dish the details on the **tech stack of @snipd_app**.



**Link mentioned**: <a href="https://x.com/latentspacepod/status/1900666708270215383">Tweet from Latent.Space (@latentspacepod)</a>: 🆕 Snipd: The AI Podcast App for Learninghttps://youtu.be/FNRO_SYx68QOur first ever OUTDOOR podcast! @swyx and @KevinBenSmith chat about @aidotengineer NYC, switching from Finance to Tech, how AI can ...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1350196967078367254)** (120 messages🔥🔥): 

> `Cursor vs Claude, Levelsio flight sim, GitDoc VS Code extension, Vibe Coding IDE UI, Auto-git commit` 


- **Claude 3.5 or 3.7: Which Vibe Wins?**: Members discussed the differences between **Claude 3.5** and **3.7**, with some finding **3.7** *too eager* and prone to doing *20 more things I didn't ask for*.
   - Others intentionally use the two different models and workflows for coding and debugging - with one finding *vibe debugging* difficult.
- **Levels.io Flight Simulator Goes Viral**: A member referenced the success of [Levels.io's flight simulator](https://x.com/levelsio/status/1893350391158292550), built with **Cursor**, which quickly reached **$1 million ARR** by selling ads in the game.
   - Levelsio noted, *AI really is a creativity and speed maximizer for me, making me just way more creative and more fast*.
- **GitDoc extension commits on save**: Members shared the [GitDoc VS Code extension](https://github.com/lostintangent/gitdoc) that allows you to edit a Git repo and auto commit on every change.
   - One user said *storage is cheap, like auto commit on every change and visualize the tree of changes* and suggested branching, restarting and other features.
- **UI Innovation needed for Vibe Coding IDEs**: Members discussed that traditional IDEs may not be the right UI for vibe coding, suggesting the need for a **visualization of the tree of changes** as prompted by different chats.
   - This would allow users to easily revert to previous states and experiment with branching.
- **Enterprise AI Dev Team Enablement**: A member offered to discuss **enterprise AI dev team enablement**, focusing on the *hurdles* and red tape involved with adopting tools like Cursor in large organizations.
   - Some expressed interest in learning about the challenges of integrating AI into corporate development workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/lostintangent/gitdoc">GitHub - lostintangent/gitdoc: VS Code extension that allows you to edit a Git repo, like it&#39;s a multi-file, versioned document.</a>: VS Code extension that allows you to edit a Git repo, like it&#39;s a multi-file, versioned document. - lostintangent/gitdoc</li><li><a href="https://x.com/levelsio/status/1893350391158292550">Tweet from @levelsio (@levelsio)</a>: ✨ Today I thought what if I ask Cursor to build a flight simulatorSo I asked  &#34;make a 3d flying game in browser with skyscrapers&#34; And after many questions and comments from me I now have the o...</li><li><a href="https://x.com/levelsio/status/1899596115210891751">Tweet from @levelsio (@levelsio)</a>: ✨ http://fly.pieter.com has now gone from $0 to $1 million ARR in just 17 days!💸 Revenue update: $87,000 MRR (which is $1M ARR)My first project ever to go up this fast 🤯Only 3 ads left now: https://...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1350042362629591050)** (92 messages🔥🔥): 

> `Download LM Studio runtimes, Snapdragon X Plus support, Gemini Vision Capabilities, AI Chess Tournament, VRAM usage for Gemma 3` 


- ****Rummaging for Runtimes**: User Decompiles LM Studio to Find Download URLs**: A user sought to download **LM Studio runtimes** for offline use, asking where the program downloads them from, after initially being told by another user that *LM Studio doesn't need an internet connection to run*.
   - The user decompiled the app and located the runtime URLs, including the [backends master list](https://extensions.lmstudio.ai/backends-master-list-stable.json) and CDN "APIs" like the [Runtime Vendor](https://extensions.lmstudio.ai/vendor-win-llama-cuda-vendor-v1.tar.gz).
- ****Snapdragon Snags**: LM Studio's Compatibility Conundrums**: A user inquired about **LM Studio's support for Snapdragon X Plus**, reporting that LM Studio did not detect their GPU and another member replied that, for GPU support, *you need to run llama.cpp directly* with this [github.com/ggml-org/llama.cpp/pull/10693](https://github.com/ggml-org/llama.cpp/pull/10693).
- ****Gemini's Geography**: Location Locks Limit Vision Functionality**: A user requested assistance in testing **Gemini 2.0 Flash Experimental's** ability to process images, noting potential regional restrictions in Germany/EU, because *it doesnt seem to work in germany (maybe EU, because of our laws here i guess?)*.
   - A user in the US tested Gemini in AI Studio and the Gemini app, finding that it failed to perform the requested image manipulation.
- ****Checkmate Craze**: AI Chess Tournament Showcases Model Accuracy**: An AI chess tournament was conducted, featuring **15 models** competing against each other, with results and details available at [dubesor.de/chess/tournament](https://dubesor.de/chess/tournament).
   - One user noted that **DeepSeek-R1** had a **92%** accuracy, but the organizer clarified that accuracy varies based on game length and opponent moves, and normal O1 was too expensive to run in the tournament.
- ****VRAM Vampires**: Gemma 3's Appetite Increases Post-Update**: A user reported that their **VRAM usage for Gemma 3** had significantly increased following the vision speed increase update.
   - Another user speculated the download size increase may be due to **CLIP** used for vision being in a separate file, and they thought it might not be embedded in the downloads but called from a separate file when uploaded to LM Studio.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dubesor.de/chess/tournament">Dubesor LLM Chess tournament</a>: no description found</li><li><a href="https://extensions.lmstudio.ai/backend-llama.cpp-win-x86_64-nvidia-cuda-avx2-1.21.0.tar.gz">no title found</a>: no description found</li><li><a href="https://extensions.lmstudio.ai/vendor-win-llama-cuda-vendor-v1.tar.gz">no title found</a>: no description found</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/10693">Introducing experimental OpenCL backend with support for Qualcomm Adreno GPUs by lhez · Pull Request #10693 · ggml-org/llama.cpp</a>: This PR introduces a new experimental OpenCL backend for Adreno GPUs. Through OpenCL, we can tap into the computational power of Adreno GPUs, which are widely used in many mobile devices, allowing ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1350049149827022920)** (44 messages🔥): 

> `memtest_vulkan, H100 rental t/s, Corsair product quality, 4090 vs A6000, RTX8000` 


- ****Memtest Vulkan** for VRAM stability**: A member suggested using [memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan), a **Vulkan compute tool**, for testing video memory stability.
- **Renting **H100** for token speed tests**: A member rented an **H100** to measure the **tokens per second** achieved with models like **gemma3 27B**.
- ****Corsair's** diminishing product quality reported**: Users reported that **Corsair's** product quality has declined in recent years.
   - One user reported that *a Corsair RAM kit was DOA* (Dead On Arrival) when switching to a **9800X3D**.
- ****A6000** over modded **4090** for reliability**: Members debated buying a local used **A6000** for $3500 versus a Chinese-modded **4090 48GB** for $4100 on eBay.
   - The consensus favored the **A6000** for its manufacturer guarantee and *known reliability, calling the 4090 a gamble*.
- ****RTX8000** a viable alternative**: A member noted that **two RTX8000 48GB** could be acquired at the same price as the **A6000**, if just memory capacity is needed.
   - However, another warned that the **RTX8000** uses an older **Turing architecture**, potentially causing issues with newer image generation models and training, but likely ok for *pure LLM inference*.



**Link mentioned**: <a href="https://github.com/GpuZelenograd/memtest_vulkan">GitHub - GpuZelenograd/memtest_vulkan: Vulkan compute tool for testing video memory stability</a>: Vulkan compute tool for testing video memory stability - GpuZelenograd/memtest_vulkan

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1350040582109532181)** (120 messages🔥🔥): 

> `ElizaOs API framework, Helius API key pricing, Quicknode API key pricing, DeepHermes-3-Mistral-24B-Preview-4bit MLX, Hermes-3-Llama-3.1-70B-FP8 vllm args` 


- **DeepHermes 3 Gets MLX Conversion**: The model [mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit](https://huggingface.co/mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit) was converted to MLX format from [NousResearch/DeepHermes-3-Mistral-24B-Preview](https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview) using mlx-lm version **0.21.1**.
- **Deep Dive into VLLM Arguments for Hermes 3**: Members are sharing different configurations to get **vllm** working correctly with **Hermes-3-Llama-3.1-70B-FP8**, including suggestions like adding `--enable-auto-tool-choice` and `--tool-call-parser` for Hermes 3 70B.
   - One member noted the need for  ``<tool_call>`` and ``</tool_call>`` tags in the tokenizer, which are present in **Hermes 3 models** but not necessarily in **DeepHermes**.
- **Vultr's Very Alpha Inference Pricing**: A member from Vultr shared the official pricing for their inference API, which includes **$10 for 50 million output tokens** initially, then **2 cents per million output tokens** after.
   - It was further explained that this stems from purchasing *an absurd amount of gh200s* and needing to do something with them, offering an OpenAI-compatible endpoint at [https://api.vultrinference.com/](https://api.vultrinference.com/).
- **Dynamic LoRAs Docking into VLLM**: Members discussed the possibility of hosting dynamic **LoRAs** with **vllm** for various use cases, like up-to-date coding styles.
   - It was suggested to let users pass in their huggingface repo IDs for the **LoRAs** and supply them into the **VLLM serve command cli args**, and there is a link to the [vLLM documentation](https://docs.vllm.ai/en/stable/features/lora.html#serving-lora-adapters).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit">mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit · Hugging Face</a>: no description found</li><li><a href="https://docs.vllm.ai/en/stable/features/lora.html#serving-lora-adapters">LoRA Adapters &#8212; vLLM</a>: no description found</li><li><a href="https://api.vultrinference.com/">Vultr Inference API</a>: no description found</li><li><a href="https://github.com/google/minja">GitHub - google/minja: A minimalistic C++ Jinja templating engine for LLM chat templates</a>: A minimalistic C++ Jinja templating engine for LLM chat templates - google/minja</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jbbwc2/this_week_did_not_go_how_i_expected_at_all/">Reddit - Heart of the internet</a>: no description found</li><li><a href="https://hub.docker.com/r/drikster80/vllm-gh200-openai">no title found</a>: no description found</li><li><a href="https://github.com/substratusai/kubeai">GitHub - substratusai/kubeai: AI Inference Operator for Kubernetes. The easiest way to serve ML models in production. Supports VLMs, LLMs, embeddings, and speech-to-text.</a>: AI Inference Operator for Kubernetes. The easiest way to serve ML models in production. Supports VLMs, LLMs, embeddings, and speech-to-text. - substratusai/kubeai
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1350038946767179899)** (90 messages🔥🔥): 

> `MCP for Astro clients, MCP Servers & Architecture, Gitlab MCP server on Windows 11, Agentic Coder Conversion to MCP, Multi-Agent Systems (Swarm vs Mesh vs Sequence)` 


- **Astro Clients gear up for MCP**: A member is planning to use MCP for their **Astro** clients, allowing MCP usage specifically for customers and exploring the possibility of adding MCP servers to a single project for client visibility.
   - They are considering using **AWS API Gateway** with each MCP server as a **Lambda** function, leveraging the MCP bridge with the **SSE gateway**.
- **MCP Server Architecture Questioned**: A member inquired about MCP server architecture, noting that clients like **Cursor** and **Cline** keep the MCP servers on the client side and asking how these communicate with the backend.
   - The member was directed to a specific channel for more information.
- **Smart Proxy Server: MCP's Agentic Sub-Routing**: Members discussed creating a "smart proxy" MCP server that simplifies standard MCP servers with many tools into one with a single tool using natural language, converting it into specific tool calls via its own LLM.
   - It's a sub-agent approach that uses *vector tool calling* to make individual agents have a subset of tools and the **OpenAI Swarm framework** follows a similar process.
- **Debugging Webpages via MCP Server**: A member shared their project, a debugger that uses MCP to debug webpages with LLMs, which was originally built with **Puppeteer** and later ported to **Playwright**: [chrome-debug-mcp](https://github.com/robertheadley/chrome-debug-mcp).
   - The member is still testing the **Playwright** version and plans to update the GitHub repository after.
- **Swarm vs Mesh Multi-Agent System**: The discussion included alternative methods to the hierarchical agent system, and the user was pointed to resources like **Swarm** vs **Mesh** vs **Sequence** architectures, emphasizing how the **Swarm framework** hands off the single thread of execution between agents.
   - It was noted that OpenAI now supports and maintains the *swarm* concept, rebranding it as **openai-agents**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/robertheadley/chrome-debug-mcp">GitHub - robertheadley/chrome-debug-mcp: An MCP server to allow you to debug webpages using LLMs</a>: An MCP server to allow you to debug webpages using LLMs - robertheadley/chrome-debug-mcp</li><li><a href="https://news.ycombinator.com/item?id=43177117">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1350039030573699103)** (3 messages): 

> `MCP server management, Awesome Vibe Coding` 


- ****MCP Hub** Concept Born!**: To address concerns about enterprise adoption of **MCP**, a member built an **MCP Hub** concept with a dashboard to simplify server connections, control access permissions, and provide visibility across **MCP** servers, as shown in [this video](https://drive.google.com/file/d/1wkWSSGGbqVQavop26svmrryee4sx3NKz/view?usp=sharing).
- **Awesome Vibe Coding List Launched!**: A member announced **Awesome Vibe Coding**, a curated list of tools, editors, and resources that enhance AI-assisted coding, available on [GitHub](https://github.com/filipecalegario/awesome-vibe-coding).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/filipecalegario/awesome-vibe-coding">GitHub - filipecalegario/awesome-vibe-coding: A curated list of vibe coding references, collaborating with AI to write code.</a>: A curated list of vibe coding references, collaborating with AI to write code. - filipecalegario/awesome-vibe-coding</li><li><a href="https://drive.google.com/file/d/1wkWSSGGbqVQavop26svmrryee4sx3NKz/view?usp=sharing).">Discord Demo  - MCP Hub - 2025.03.13.mov</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1350031278505852929)** (57 messages🔥🔥): 

> `ZIRP Era Regret, AI Startup Valuations, DeepSeek Passport Confiscation, Long Context Evaluation Challenges, Xet Data Chunking Technology` 


- **OpenAI and Anthropic Shed Tears for ZIRP Era**: Members joked that **OpenAI and Anthropic** are regretting not being in the **ZIRP era**, referencing a [Gigachad meme](https://tenor.com/view/gigachad-chad-gif-20773266).
   - Another member quipped that all **AI startups raise at valuations that make ZIRP feel like kindergarten**.
- **NYT peers into AGI Future**: A [New York Times article](https://www.nytimes.com/2025/03/14/technology/why-im-feeling-the-agi.html) from the future (March 14, 2025) suggests that **AI systems** have started surpassing humans in domains like **math**, **coding**, and **medical diagnosis**.
   - The article anticipates that one or more **AI companies** will achieve general superhuman intelligence in **2026 or 2027**, or possibly as soon as this year.
- **DeepSeek Asks Employees To Fork Over Passports!**: It was reported that **DeepSeek's** owner asked R&D staff to hand in passports so they can't travel abroad, according to [Amir on Twitter](https://fxtwitter.com/amir/status/1900583042659541477).
   - Some members speculated whether this would lead to **DeepSeek** work remaining open source, or whether the US would ever take similar measures with frontier company employees.
- **Xet Uses Content-Defined Chunking (CDC)**: **Xet** uses [Content-Defined Chunking (CDC)](https://huggingface.co/blog/from-chunks-to-blocks) technology to intelligently break files into unique chunks, as mentioned in their [HuggingFace Join page](https://huggingface.co/join/xet).
   - A member asked how it was different from fast transfer, another member replied that they're different technology, and that fast transfer still uses Git LFS.
- **Math-500 Sampling Questioned and Validated**: A member asked why the math-500 sampling was random in Qwen's [github repo](https://github.com/QwenLM/QwQ) for evaluation scripts.
   - Another member replied that it was *apparently random* and quoted [Lightman et al 2023](https://cdn.discordapp.com/attachments/1179128538679488533/1350170687460868186/image.png?ex=67d5c3f0&is=67d47270&hm=6c771f09d27b7bad57e711e55ed2b111ac29af6a6485feb2c89103757f0771de&), also they cited that long context evals and answer extraction is a headache and that **Math 500 is very well correlated**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/join/xet">Join the Xet waitlist · Hugging Face</a>: no description found</li><li><a href="https://fxtwitter.com/amir/status/1900583042659541477">Tweet from Amir Efrati (@amir)</a>: new: DeepSeek&#39;s owner asked R&D staff to hand in passports so they can&#39;t travel abroad.Ladies & gentlemen... China:</li><li><a href="https://x.com/Alibaba_Qwen/status/1900595120053047452">Tweet from Qwen (@Alibaba_Qwen)</a>: Folks, we have set up a github repo for QwQ, specifically providing evaluation scripts for you to easily test the benchmark performance of reasoning models, and also reproduce our reported results. We...</li><li><a href="https://x.com/arcprize/status/1900627173280804941">Tweet from ARC Prize (@arcprize)</a>: 3/24/2025</li><li><a href="https://tenor.com/view/gigachad-chad-gif-20773266">Gigachad GIF - Gigachad Chad - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://archive.is/jQZln">Why I&#x2019;m Feeling the A.G.I. - The New York Times</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1350141493670576128)** (16 messages🔥): 

> `Invasion of Privacy, Claude's Birthday, Claude Code Vim mode, Gemma 3 licensing issues` 


- **Phone Number Fiasco Frustrates**: A member expressed frustration about people finding their phone number online and requesting favors such as *"hey nato, can you post-train my llama2 model? ty"*.
   - They attribute this to extensions or paid services and are seeking ways to remove their information from sites like [Xeophon](https://x.com/alexalbert__/status/1900592059364634973).
- **Claude Celebrates Another Year**: Members celebrated the second birthday of **Claude**, referencing the original announcement [two years ago](https://x.com/alexalbert__/status/1900592059364634973).
   - Another member highlighted new features for **Claude Code**, including [Vim mode](https://x.com/_catwu/status/1900593728664035590) activated by typing the slash command `/vim`.
- **Gemma 3's License Provokes Concerns**: A TechCrunch article mentioned a member's work regarding model licenses, specifically in relation to [Google's Gemma 3](https://techcrunch.com/2025/03/14/open-model-licenses-often-carry-concerning-restrictions/).
   - The article discusses how **Gemma 3's license**, while praised for efficiency, poses commercial use risks due to restrictive and inconsistent terms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_catwu/status/1900593728664035590">Tweet from cat (@_catwu)</a>: Another batch of features for Claude Code!Up first: Vim mode. This gives you the familiar insert/command modes for editing your prompts in Claude Code. Turn it on by typing the slash command /vim.But ...</li><li><a href="https://x.com/alexalbert__/status/1900592059364634973">Tweet from Alex Albert (@alexalbert__)</a>: Two years ago today we announced Claude to the world.Happy second birthday, Claude!</li><li><a href="https://techcrunch.com/2025/03/14/open-model-licenses-often-carry-concerning-restrictions/">&#039;Open&#039; model licenses often carry concerning restrictions | TechCrunch</a>: &#039;Open&#039; model releases from Google, Meta, and others have onerous terms that make some companies wary of using them.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1350148438523773020)** (5 messages): 

> `Mid-Training Analysis, SF Compute H100s, SF Compute CLI` 


- **Deep Dive into Mid-Training Analysis**: A member shared a [link on fxtwitter](https://fxtwitter.com/Yuchenj_UW/status/1900589508590268485) prompting discussion about the nuances of **mid-training analysis**.
   - The discussion was brief, with a single follow-up question.
- **SF Compute's Surprisingly Low H100 Prices**: A member highlighted that [SF Compute](https://sfcompute.com/) offers surprisingly low prices for **H100s**, particularly for short-term rentals, noting the availability of **128 H100s** for just an hour.
   - They had previously encountered a confusing placeholder page, before the domain was properly configured.
- **SF Compute Launches 2,000 Additional H100s Soon**: San Francisco Compute Company is [launching soon an additional **2,000 H100s**](https://x.com/evanjconrad/status/1884361612766896510) and runs a market for large-scale, vetted **H100 clusters**.
   - SF Compute supports users like a traditional cloud and also has a [simple but powerful CLI](https://docs.sfcompute.com) available.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/LocoMocoBassy/status/1900191262822551986">Tweet from LocoMocosBasilisk (@LocoMocoBassy)</a>: hmmm………</li><li><a href="https://sfcompute.com/">SF Compute | H100s with 3.2Tb/s InfiniBand</a>: The San Francisco Compute Company: Large, low-cost GPU clusters you can rent by the hour, for pre-training, inference, and more. Get H100s with 3.2Tb/s InfiniBand, parallel storage, fast networking, a...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1350190512946151576)** (11 messages🔥): 

> `GRPO implementation, KL penalty, RLHF algorithms` 


- **GRPO Implementation Trick: KL Penalty Applied in Loss**: A member discussed a **GRPO implementation trick** where the **KL penalty** is applied directly in the loss rather than when the reward is computed, noting that its impact is not well-understood and linking to the [RLHF Book](https://rlhfbook.com/c/11-policy-gradients.html#group-relative-policy-optimization-1).
   - The member also shared a [link to their question on X/Twitter](https://x.com/natolambert/status/1900639281791615387) asking for intuitions or ablations on this approach.
- **RLHF Algorithms Popularity Over Time**: The most popular algorithms used for **RLHF** has evolved over time; initially, a variant of **PPO** was used with **ChatGPT**, but research has shown promise in **REINFORCE** style algorithms, such as [Ahmadian et al. 2024](https://rlhfbook.com/c/11-policy-gradients.html#ref-ahmadian2024back) and [Wang et al. 2024](https://rlhfbook.com/c/11-policy-gradients.html#ref-wang2024helpsteer2p).
- **Focusing on Reward Signals via KL Penalty**: A member suggested applying the **KL penalty** in the loss might help the model focus on reward signals, but the member also noted that it *should end up being equivalent* to applying it when the reward is computed.
   - Another member guessed the **PG term** will maximize the reward, so the loss minimum should be the same for both versions, but the dynamics could still be different.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rlhfbook.com/c/11-policy-gradients.html#group-relative-policy-optimization-1">Policy Gradient
Algorithms | RLHF Book by Nathan Lambert</a>: The Reinforcement Learning from Human Feedback Book</li><li><a href="https://x.com/natolambert/status/1900639281791615387">Tweet from Nathan Lambert (@natolambert)</a>: Does anyone have an intuition or ablation on applying the KL penalty in the loss directly rather than when the reward is computed? How is this changing learning.normalrewards = rewards - self.beta * p...</li><li><a href="https://bsky.app/profile/natolambert.bsky.social/post/3lkeftspdzo2x">Nathan Lambert (@natolambert.bsky.social)</a>: Does anyone have an intuition or ablation on applying the KL penalty in the loss directly rather than when the reward is computed? How is this changing learning.normalrewards = rewards - self.beta * p...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1350132490500837458)** (2 messages): 

> `Cohere Command A, Jamba 1.6 Large, Jamba 1.6 Mini, Gemma 3 models, Anthropic incident` 


- ****Cohere Commands Attention****: A new **open-weights 111B parameter model** with a **256k context window** focused on delivering great performance across agentic, multilingual, and coding use cases, [Cohere Command A](https://openrouter.ai/cohere/command-a) is now available.
- ****AI21 Jamba Jams with New Models****: **AI21** released [Jamba 1.6 Large](https://openrouter.ai/ai21/jamba-1.6-large) featuring **94 billion active parameters** and a **256K token context window**, while also launching [Jamba 1.6 Mini](https://openrouter.ai/ai21/jamba-1.6-mini), with **12 billion active parameters**, both supporting structured JSON output and tool-use capabilities.
- ****Gemma Gems Gleam for Free****: All variations of **Gemma 3** are now available for free: [Gemma 3 12B](https://openrouter.ai/google/gemma-3-12b-it:free) introduces multimodality, supporting vision-language input and text outputs, handling context windows up to **128k tokens** and understands over **140 languages**, as well as [Gemma 3 4B](https://openrouter.ai/google/gemma-3-4b-it:free) and [Gemma 3 1B](https://openrouter.ai/google/gemma-3-1b-it:free).
- ****Anthropic API Anomaly Averted****: **Anthropic** declared an incident of elevated errors for requests to **Claude 3.7 Sonnet**, with updates posted on their [status page](https://status.anthropic.com/incidents/qtxnlg9yrwqv).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://status.anthropic.com/incidents/qtxnlg9yrwqv">Elevated errors for requests to Claude 3.7 Sonnet</a>: no description found</li><li><a href="https://openrouter.ai/cohere/command-a):">Discord</a>: no description found</li><li><a href="https://openrouter.ai/ai21/jamba-1.6-large):">Discord</a>: no description found</li><li><a href="https://openrouter.ai/ai21/jamba-1.6-mini):">Discord</a>: no description found</li><li><a href="https://openrouter.ai/google/gemma-3-12b-it:free)">Gemma 3 12B - API, Providers, Stats</a>: Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, ...</li><li><a href="https://openrouter.ai/google/gemma-3-4b-it:free)">Gemma 3 4B - API, Providers, Stats</a>: Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles context windows up to 128k tokens, understands over 140 languages, and offers improved math, reasoning, ...</li><li><a href="https://openrouter.ai/google/gemma-3-1b-it:free)">Gemma 3 1B - API, Providers, Stats</a>: Gemma 3 1B is the smallest of the new Gemma 3 family. It handles context windows up to 32k tokens, understands over 140 languages, and offers improved math, reasoning, and chat capabilities, including...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1350039038618243092)** (67 messages🔥🔥): 

> `OR ChatGPT model, OpenRouter model icons, Deepseek v3 issues, OLMO-2, Cohere repetition penalties` 


- **ChatGPT-4o-latest Price Higher Than Expected**: The **chatgpt-4o-latest** model is up to date but is slightly more expensive than the normal **4o** model.
- **OpenRouter Model Icons Not Available via API**: The icons for the models are not available in the `/api/v1/models` response, instead using website favicons.
- **Deepseek v3 Model Gives Zero Token Issues**: Sometimes the inference stack just returns **zero completion tokens**, and OpenRouter still gets charged by the upstream provider.
- **OLMO-2 Model Hosted on OpenRouter**: OLMo-2 is coming online through Parasail; someone will spin it up and notify OpenRouter to add it.
- **AI Chess Tournament Hosted with OpenRouter**: An AI chess tournament featuring **15 models** was created, models are fed information in standard chess notations about the board state, the game history, and a list of legal moves to compete against each other [here](https://dubesor.de/chess/tournament).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dubesor.de/chess/tournament">Dubesor LLM Chess tournament</a>: no description found</li><li><a href="https://openrouter.ai/openai/gpt-3.5-turbo-">Discord</a>: no description found</li><li><a href="https://openrouter.ai/openai/gpt-3.5-turbo-instruct">GPT-3.5 Turbo Instruct - API, Providers, Stats</a>: This model is a variant of GPT-3.5 Turbo tuned for instructional prompts and omitting chat-related optimizations. Run GPT-3.5 Turbo Instruct with API</li><li><a href="https://parasail.canny.io/model-request">Model Request | Parasail</a>: Request Models - Please Put in the Hugging Face Model and any other information!
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1350031425977581588)** (59 messages🔥🔥): 

> `Rust vs. Go for porting, DeepSeek Hype, OLMo 2 32B, ChatGPT Overrated, Code generation quality: Grok 3 vs Mistral vs OpenAI` 


- **Go beats Rust for practical porting**: Members discussed why a port wasn't being done in **Rust**, and one member explained it's because a function-by-function port to **Go** allows for exact behavior parity, avoiding years of rewriting and dealing with Rust's no-GC and lifecycle annotations.
   - It was added that *Rust is faster and more efficient, but it's really not as big a difference as people make it out to be in practice, and Golang is really ergonomic to develop in*, particularly for distributed, async, or networked applications.
- **DeepSeek hype is engineered**: Some members suggested that the hype around **DeepSeek** is engineered, arguing that their models are simplified and not on par with frontier AI models, comparing them to *comparing Ananas with Apple*.
   - Another member countered that the hype was driven by the *crazy engineers* at **DeepSeek** who developed a filesystem faster than life.
- **OLMo 2 32B is a fully open model**: **OLMo 2 32B** was released, and described as the [first fully-open model](https://allenai.org/blog/olmo2) to outperform **GPT3.5-Turbo** and **GPT-4o mini** on academic benchmarks.
   - It is claimed to be comparable to leading open-weight models while requiring only a fraction of training compute, costing only one third of the training cost of **Qwen 2.5 32B**.
- **ChatGPT is overrated, use Claude**: One member expressed that **ChatGPT** is overrated because *it actually doesn't solve problems that I need solved*, preferring **Mistral Small 24B**, **QwQ 32B**, and **Claude 3.7 Sonnet**.
   - Another user shared, *I've had better luck getting what I want from Claude*, and *seems better at understanding intention and motivation for whatever reason*.
- **Grok 3 for code generation**: Members debated code generation qualities, mentioning that **OpenAI** models often generate legacy code, while **Mistral** can refactor it into more modern code, and **Grok 3** generates code that *looks like a professional programmer wrote it*.
   - In **VSCode**, one member prefers using **Amazon Q** over **Copilot**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/mgostIH/status/1900577603930444239">Tweet from mgostIH (@mgostIH)</a>: &gt; yeah but software is easily replicable you can just copy code Someone who never touched CMake</li><li><a href="https://allenai.org/blog/olmo2-32B">OLMo 2 32B: First fully open model to outperform GPT 3.5 and GPT 4o mini  | Ai2</a>: Introducing OLMo 2 32B, the most capable and largest model in the OLMo 2 family.
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/)** (1 messages): 

> @erkinalp: 

.ogeneral: I would say neither
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1350049237353627649)** (3 messages): 

> `Speech-to-Speech Generation, Moshi by Kyutai Labs, Hertz-dev by Standard-Intelligence` 


- ****Speech-to-Speech** Model Quest Begins**: A member is looking for **speech-to-speech generation** models, focusing on conversational speech without multimodal input, distinguishing it from models like **OpenAI Realtime API** or **Sesame AI**.
   - The member seeks a standalone model rather than a multimodal one that accepts both text and audio.
- ****Moshi** Model Sounds off.**: [Moshi](https://github.com/kyutai-labs/moshi) from **Kyutai Labs** is a speech-text foundation model and full-duplex spoken dialogue framework.
   - It utilizes **Mimi**, a state-of-the-art streaming neural audio codec.
- ****Hertz-dev** Model is developed.**: [Hertz-dev](https://github.com/Standard-Intelligence/hertz-dev) from **Standard-Intelligence** is the first base model for full-duplex conversational audio.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/kyutai-labs/moshi">GitHub - kyutai-labs/moshi: Moshi is a speech-text foundation model and full-duplex spoken dialogue framework. It uses Mimi, a state-of-the-art streaming neural audio codec.</a>: Moshi is a speech-text foundation model and full-duplex spoken dialogue framework. It uses Mimi, a state-of-the-art streaming neural audio codec. - kyutai-labs/moshi</li><li><a href="https://github.com/Standard-Intelligence/hertz-dev">GitHub - Standard-Intelligence/hertz-dev: first base model for full-duplex conversational audio</a>: first base model for full-duplex conversational audio - Standard-Intelligence/hertz-dev
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1350043285439254618)** (3 messages): 

> `tl.int1 masks in Triton, tl.advance negative offsets, Triton Windows upgrade to 3.2` 


- **Optimize Triton masks with `tl.int1`?**: A member asked if there's any performance or functional benefit to explicitly converting masks to `tl.int1` when using `tl.load` in Triton.
   - No response was provided.
- **`tl.advance` accepts negative offsets?**: A member inquired whether `tl.advance` in Triton accepts negative offsets for pointer arithmetic.
   - No response was provided.
- **Windows Triton Upgrade Woes?**: A member seeks validation on steps to upgrade **Triton** from **3.1** to **3.2** on Windows, particularly concerning **PyTorch** and cache clearing. They linked to [this repo](https://github.com/woct0rdho/triton-windows).
   - They were using **Python 3.10 + CUDA 12.5** and **ComfyUI’s python_embedded: Python 3.12 + PyTorch 2.5.1+cu124 + Triton 3.1**



**Link mentioned**: <a href="https://download.pytorch.org/whl/cu124">no title found</a>: no description found

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1350161350214553650)** (6 messages): 

> `cuda::memcpy_async, A100, global vs shared memory` 


- **`cuda::memcpy_async` Experiments on A100**: A member experimented with [`cuda::memcpy_async`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#with-memcpy-async) on **A100s** using examples from the CUDA documentation.
   - They observed that the kernel with `memcpy_async` took slightly longer to run and inquired about the reason for this unexpected behavior.
- **Async Copies: Global vs Shared Memory**: A member clarified that async copies can only transfer data between **global and shared memory**.
   - They explained that the advantage of async copies lies in overlapping memory loading with other computations, requiring values to be loaded from shared memory to be utilized.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1350180800368218142)** (1 messages): 

> `Block Diffusion, Autoregressive Models, Diffusion Models, ICLR 2025` 


- **Block Diffusion Model Interpolates Autoregressive and Diffusion LMs**: The new **Block Diffusion** model aims to combine the benefits of both autoregressive and diffusion language models, detailed in a [paper](https://openreview.net/forum?id=tyEyYT267x) accepted as **ICLR 2025 Oral** presentation.
   - It achieves **high quality**, **arbitrary-length generation**, **KV caching**, and **parallelizable** processing, addressing the limitations of existing models; code available on [GitHub](https://github.com/kuleshov-group/BD3-LMs) and [HuggingFace](https://huggingface.co/collections/kuleshov-group/BD3-LMs-67be95f81b96b15fec50d53f).
- **Autoregressive Models**: Autoregressive models boast **high quality** output and **arbitrary-length** generation along with Key-Value (KV) caching.
   - However, autoregressive models suffer from being **non-parallelizable**.



**Link mentioned**: <a href="https://m-arriola.com/bd3lms/">SOCIAL MEDIA TITLE TAG</a>: SOCIAL MEDIA DESCRIPTION TAG TAG

  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1350130082085077042)** (2 messages): 

> `Dynamic Shapes, Segmentation Fault` 


- **Dynamic Shapes spark Segmentation Fault**: A member opened an issue about a [segmentation fault with dynamic shapes](https://github.com/tile-ai/tilelang/issues/215).
   - They added, *maybe my understanding is wrong*.
- **Tilelang faces Dynamic Shape challenge**: An issue was raised concerning segmentation faults encountered when working with dynamic shapes in Tilelang.
   - The reporter of the issue expressed uncertainty, stating that their understanding of the problem might be incorrect.



**Link mentioned**: <a href="https://github.com/tile-ai/tilelang/issues/215">segmentation fault with dynamic shapes · Issue #215 · tile-ai/tilelang</a>: # Copyright (c) Microsoft Corporation. # Licensed under the MIT License. from tilelang import tvm as tvm import tilelang.language as T import tilelang.testing import tilelang import torch def matmu...

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1350051208903262242)** (1 messages): 

> `Gemma3, LigerKernel, RMSNorm` 


- **Gemma3 Implementation Drafted into LigerKernel**: A member has started a new challenge for themself, drafting an implementation of **Gemma3** into **LigerKernel**, and shared a [link to the pull request](https://github.com/linkedin/Liger-Kernel/pull/606).
   - The member being fairly new to programming is asking for help and feedback on the draft.
- **Gemma3 and Gemma2 share similarities**: According to the pull request, **Gemma3** has high similarities to **Gemma2** with some differences in **RMSNorm Calls**.
   - The changes enable patching the Text Parts of **Gemma3** with **Liger kernels**.



**Link mentioned**: <a href="https://github.com/linkedin/Liger-Kernel/pull/606">Adding Support for Gemma3 by DRXD1000 · Pull Request #606 · linkedin/Liger-Kernel</a>: SummaryGemma3 has high similarities to Gemma2 with some differences in RMSNorm CallsThis change enables patching the Text Parts of Gemma3 with Liger kernels.Testing DoneHardware Type: AMD ...

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1350086681461784738)** (2 messages): 

> `Triton bitpacking, Gemlite, GTC CUDA content` 


- **Triton bitpacking goes Vroom Vroom**: **Bitpacking** in **Triton** achieved significant speed-ups versus the Pytorch implementation on the 4090, achieving **98x** speedup for **32-bit packing** and **26x** for **8-bit packing**.
   - Re-packing a **Llama3-8B** model time was reduced from **49 sec -> 1.6 sec** using the new bitpacking implementation.
- **Gemlite's bitpack Implementation Unleashes Performance**: A member spotlighted their work on optimizing bitpacking using **Triton** within the **Gemlite** project, with a link to the [relevant code on GitHub](https://github.com/mobiusml/gemlite/blob/master/gemlite/bitpack.py#L59-L133).
   - The optimization allows for *fast low-bit matmul kernels*.
- **GTC CUDA content Incoming**: A member shared information regarding [CUDA content](https://www.nvidia.com/gtc/sessions/cuda-developer/) at **GTC**, to create high-performance, GPU-accelerated applications with NVIDIA CUDA.
   - They shared some images of Plask and CERN which were featured at the conference.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.nvidia.com/gtc/sessions/cuda-developer/">NVIDIA GTC AI Conference 2025</a>: March 17–21, 2025. San Jose. Register Now.</li><li><a href="https://github.com/mobiusml/gemlite/blob/master/gemlite/bitpack.py#L59-L133">gemlite/gemlite/bitpack.py at master · mobiusml/gemlite</a>: Fast low-bit matmul kernels in Triton. Contribute to mobiusml/gemlite development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1350078770396266557)** (31 messages🔥): 

> `Gemma 3 support in vLLM, Group Relative Policy Optimization (GRPO), veRL Training for reasoning-gym, composite configurations in reasoning-gym, curriculum training` 


- **vLLM Gets Go-Ahead for Gemma 3**: Members discussed adding **Gemma 3** support to **vLLM**, referencing [this GitHub issue](https://github.com/vllm-project/vllm/issues/14696).
   - One member reported experiencing issues with context window size while using **Gemma3** with TGI, suspecting a problem in the underlying transformers implementation.
- **Group Relative Policy Optimization Grows Rapidly**: Members discussed how **Group Relative Policy Optimization (GRPO)** has become popular for Reinforcement Learning in Large Language Models.
   - A blog post from oxen.ai on [GRPO VRAM requirements](https://www.oxen.ai/blog/grpo-vram-requirements-for-the-gpu-poor) was shared, noting its effectiveness in training, with a link to the **DeepSeek-R1 paper** included.
- **veRL Voyages Victorious**: A member confirmed that **veRL training** is working for **chain_sum** with the newest veRL using the changes in [this branch](https://github.com/open-thought/reasoning-gym/tree/ollie/verl-experiments).
   - The change was merged into the main branch.
- **reasoning-gym Readies Refactor for Reasoning**: Members discussed training models using a single **RG dataset generator** versus a composite of multiple, leaning towards the latter for improving *"all-around"* reasoning capabilities.
   - They plan to test with a small composite of 3-5 datasets, referencing [the composite dataset code](https://github.com/open-thought/reasoning-gym/blob/main/tests/test_composite.py) and the curriculum status for datasets in [GALLERY.md](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md).
- **reasoning-gym Rockets Past 500 Stars**: It was mentioned that *reasoning-gym v.0.1.16* was [uploaded to pypi](https://pypi.org/project/reasoning-gym/) and the project is nearing 500 stars.
   - A member posted a picture celebrating.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pypi.org/project/reasoning-gym/)">Client Challenge</a>: no description found</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/tests/test_composite.py">reasoning-gym/tests/test_composite.py at main · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/tree/ollie/verl-experiments">GitHub - open-thought/reasoning-gym at ollie/verl-experiments</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/vllm-project/vllm/issues/14696">[Feature]: Support gemma3 architecture · Issue #14696 · vllm-project/vllm</a>: 🚀 The feature, motivation and pitch I am using vLLM for hosting of LLMs/SLMs and with the recent release of Gemma 3, I would love to have it supported in vLLM. Google has stated Gemma 3 has day 1 s.....</li><li><a href="https://www.oxen.ai/blog/grpo-vram-requirements-for-the-gpu-poor">🧠 GRPO VRAM Requirements For the GPU Poor | Oxen.ai</a>: Since the release of DeepSeek-R1, Group Relative Policy Optimization (GRPO) has become the talk of the town for Reinforcement Learning in Large Language Models due to its effectiveness and ease of tra...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1350062138890129418)** (7 messages): 

> `verl session, tilelang submission, pip install tilelang` 


- **Demand for **Verl** Session**: A member inquired about the timeline for a session on **Verl**, along with the possibility of submitting **tilelang**.
- **Installing **tilelang** via **pip****: A member stated that users can install any package via **pip** from a script, providing an example script to install **tilelang**.
   - They cautioned that the installation is *fairly long* and might lead to timeouts and unnecessary work for the machines.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1350175035817594880)** (3 messages): 

> `Leaderboard Submissions, Grayscale Leaderboard, Conv2d Leaderboard, H100 GPUs, Modal Runners` 


- **Grayscale Leaderboard sees New Submissions**: Three new leaderboard submissions were made to the **grayscale** leaderboard.
   - The submissions with IDs **2013** and **2015** used **H100 GPUs** and **Modal runners**.
- **Conv2d Leaderboard Populated**: A new leaderboard submission was made to the **conv2d** leaderboard.
   - The submission, with ID **2014**, also utilized **H100 GPUs** and **Modal runners**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1350036900383035442)** (46 messages🔥): 

> `Declining intelligence, impact of technology and smartphones, food additives and cognitive decline, Deepseek models distillation, ADHD diagnosis rates` 


- **Intelligence Declines Stir Debate!**: Discussion sparked from a [Financial Times article](https://www.ft.com/content/link-to-ft-article) that average intelligence is dropping in developed countries, citing increased reports of **cognitive challenges** and declining performance in **reasoning and problem-solving**.
   - One member theorized this could be due to technology, especially **smartphones** and **social media**, leading to outsourcing of thinking, however the graphics only showed the years really before **ChatGPT** became a thing.
- **Is Tech to Blame for Brain Drain?**: Members debated potential causes for cognitive decline, including **technology's influence**, **immigration**, and **fluoridated water**.
   - One member pointed out that the rates of cognitive challenges were steadily increasing since the 1990s, and a sudden acceleration from around 2012.
- **Food Additives Linked to Mental Slowdown?**: Members discuss that the availability and consumption of **ultra-processed foods (UPFs)** has increased worldwide and represents nowadays **50–60%** of the daily energy intake in some high-income countries, and is linked to cognitive decline
   - Another member mentions Multinational corporations such as **Nestlé** that operate in many countries produce and distribute worldwide, it seems understandable how different additives or changes made to these products in one of these corporations can have a worldwide impact.
- **DeepSeek's Model Origin Mystery**: The discussion covers that **Deepseek V3 (the instruct version)** was likely distilled from **OpenAI models**.
   - One member notes that *even OpenAI unofficially supports distilling their models, they just don't seem to like it when Deepseek does it*.
- **TikTok Brain Shortens Attention Spans**: Members believe platforms like **TikTok** and **Instagram** affect our brains by delivering a constant emotional impressions in an extremely short time.
   - The result is a *kind of addiction where we continuously seek more stimulation.*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1350105918485823509)** (1 messages): 

> `Claude Sonnet 3.7, o3-mini-high vs o1` 


- **Claude Sonnet 3.7 New Fav for Coding**: A member now uses **Claude Sonnet 3.7** exclusively for coding, finding **ChatGPT** lagging behind.
- **o3-mini-high Model Beats o1**: A member stated that the **o3-mini-high** model is better than **o1**.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1350058306588119093)** (4 messages): 

> `Gemini 2.0 Deep Research, NotebookLM, PhytoIntelligence framework` 


- **Gemini 2.0 Deep Research Joins NotebookLM**: Members considered pairing **Gemini 2.0 Deep Research** with **NotebookLM** to create a strong tool for documentation fetching and processing.
   - The aim is to containerize material without exceeding provided boundaries, questioning whether **Deep Research** could eventually replace **NotebookLM**.
- **NotebookLM Inspires African Project Ecokham**: A member from Africa is using **NotebookLM** to connect thoughts, edit and customize roadmaps, and generate inspiring audio for his project, **Ecokham**.
   - He expressed gratitude for **NotebookLM**'s assistance in inspiring him and his team.
- **NotebookLM Prototyping PhytoIntelligence Framework**: A member is using **NotebookLM** to organize notes and prototype the **PhytoIntelligence framework** for autonomous nutraceutical design.
   - This framework aims to mitigate diagnostic challenges, and the user thanked Google for **NotebookLM**'s capabilities.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1350045583200747602)** (41 messages🔥): 

> `Image and table recognition in Notebook LM, Notebook LM Mobile App, Notebook LM Language Settings, Public Notebook Sharing, Google Sheets Integration` 


- **NotebookLM Craves Image & Table Savvy**: Users are clamoring for **image and table recognition** in Notebook LM, emphasizing that the current state feels *"half-baked"* due to the need to constantly reopen source files; one user even shared [a relevant cat GIF](https://tenor.com/view/cat-wait-im-goated-pineapple-gif-1866711197257641569).
   - They believe image recognition is worth *"a thousand words"* and the clearest data comes from source tables and google sheets.
- **NotebookLM Mobile App**: Many users are requesting a **mobile app version** of NotebookLM.
   - Currently the community feels a mobile version is *"still not yet coming up"*.
- **User reports recurrent system errors in Notebook LM**: A user reported a recurring *"The system was unable to answer"* error, happening for the second time this week.
   - Other users tested the issue and couldn't reproduce it.
- **URL tricks to change NotebookLM language**: A user asked how to change the language of NotebookLM, another user shared the tip to add **?hl=LANGUAGE_CODE** at the end of the URL (e.g., `hl=es` for Spanish).
   - One user confirmed they are in France.
- **Notebook Public Sharing**: A user inquired about plans for **public sharing** of Notebooks.
   - A member responded that fully open access is unlikely but sharing with restrictions is currently possible with corp or Gmail accounts.



**Link mentioned**: <a href="https://tenor.com/view/cat-wait-im-goated-pineapple-gif-1866711197257641569">Cat Wait GIF - Cat Wait Im - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1350149344375996416)** (1 messages): 

> `Google Gemini, Google Vertex AI, Unified @googleai integration, Streaming, Async` 


- **Google Gemini and Vertex AI Merge!**: A unified `@googleai` integration now supports both **Google Gemini** and **Google Vertex AI**, according to [this Tweet](https://twitter.com/llama_index/status/1900590246070476929).
   - The integration supports **streaming**, **async**, **multi-modal**, and **structured prediction**, even supporting images.
- **Even More Google Gemini and Vertex AI Benefits!**: A unified `@googleai` integration now supports both **Google Gemini** and **Google Vertex AI**, with even more benefits!
   - The integration supports **streaming**, **async**, **multi-modal**, and **structured prediction**, even supporting images.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1350105702131306546)** (8 messages🔥): 

> `LlamaIndex vs Langchain, OpenAI delta events for tool calling, Agentic RAG applications` 


- **Debate on LlamaIndex Perks Over Langchain Erupts**: A member inquired about the advantages of **LlamaIndex** over **Langchain**.
   - However, there was no discussion or answer provided in the given context.
- **OpenAI's Delta Event Absence for Tool Calling Probed**: A member asked why **OpenAI** models don't emit delta events for tool calling, noting that they are emitted but empty.
   - Another member explained that tool calling cannot be a stream because the LLM requires the complete tool response to generate the next response, with the recommendation to build your own stream.
- **Inquiry on Agentic RAG App APIs Surfaces**: A member asked if there is any **API focused on building Agentic RAG applications** to simplify the process of creating and managing the application.
   - The discussion explores the preferred way for building agentic apps in LlamaIndex, pointing out the evolution and multiple constructs available, but lacks a definitive guide.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1350124926916366387)** (7 messages): 

> `Gemma 3 12B, Qwen 2.5 Coder, LM Studio, Multimodal Models, Water Freezing Experiment` 


- **Gemma 3 12B Surpasses Qwen in Intelligence**: A user finds that the **Gemma 3 12B model** surpasses **Qwen 14B** and even **Qwen 32B** in terms of intelligence on their computer.
   - The user noted that *Gemma 3 12b has calls for another additional gguf file, maybe that's why it won't work* with **GPT4All**.
- **Gemma 3 Requires GPT4All Updates**: A user notes that **GPT4All** may need updates to support **Gemma 3 12B** due to its different architecture from **Gemma 2** and the need for an *mmproj* file.
   - Another user jokes that *all use models that are 1 day old an expect that all worls fine*, highlighting the rapid pace of AI model development.
- **Language Comprehension Test: Gemma 3 Excels**: A user tested various models, including **Gemma 3 12B**, **DeepSeek R1**, and **Qwen 2.5 Coder**, by asking a question in multiple languages.
   - The user found that **Gemma 3** and **DeepSeek R1** consistently provided correct and comprehensive answers in the same language as the question.
- **Water Freezing Experiment Yields Varied Responses**: When asked about the outcome of freezing a jar of water, **DeepSeek-R1** indicated the jars would break.
   - **Gemma-3-12b** correctly responded that a *full jar of water outside in sub-freezing temperatures overnight will almost certainly result in the jar shattering or cracking due to the expansion of water as it freezes*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1350066494473633883)** (6 messages): 

> `Explicit Feedback in dspy.Refine, Manual Feedback Implementation in Refine, Reward Function Return Value for Feedback` 


- **Explicit Feedback beckons Refine's return**: A member inquired about integrating **explicit feedback** into `dspy.Refine`, akin to `dspy.Suggest` from earlier versions, to clearly indicate areas for improvement beyond a reward function threshold.
   - The member noted that **explicit feedback** was very helpful for debugging and understanding mistakes.
- **Manual Feedback makes marvelous move into Refine**: A team member confirmed that **manual feedback** is being added into `Refine`.
   - The proposed implementation involves including feedback in the **return value of the reward function** as a `dspy.Prediction` object, containing both a score and feedback.
- **Reward Function becomes Feedback Fountain**: The team member asked if it would be acceptable for feedback to be part of the **return value of the reward_fn**.
   - The user responded that it was *perfect*, thanking the team member.


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1350117120519704606)** (5 messages): 

> `Command A, OpenRouter Integration, Prime Number Bug, Local API Performance` 


- **Command A goes Live on OpenRouter**: **Cohere's Command A**, an open-weights **111B** parameter model with a **256k** context window, is now live on [OpenRouter](https://openrouter.ai/cohere/command-a).
- **Prime Number Puzzle Plagues Command A**: Users found that **Command A** either returns the wrong answer or enters an infinite loop when asked about prime numbers with digits summing to 15.
- **Local API struggles with Command A**: Users reported patching the modeling in **ITOr** or using the **APIs** is needed because local setups do not reach proper speeds even with sufficient VRAM.



**Link mentioned**: <a href="https://openrouter.ai/cohere/command-a">Command A - API, Providers, Stats</a>: Command A is an open-weights 111B parameter model with a 256k context window focused on delivering great performance across agentic, multilingual, and coding use cases.Compared to other leading propri...

  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/)** (1 messages): 

michael: it does, use the `https://api.cohere.com/compatibility/v1/chat/completions` base_url
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1350113696415744111)** (3 messages): 

> `Discord Scam Account, Account Impersonation` 


- **Discord Account Impersonation!**: A member reported being messaged by a **scam account** impersonating another user.
   - The impersonated user confirmed that *"That is not me. This is my only account"* and that the **fake account** has been reported to Discord and banned from the server.
- **User Warns of Impersonation Scam**: A user alerted the community to a **scam account** messaging them, impersonating another user on Discord.
   - The user clarified that their only legitimate account is "caroline_frasca" and that the imposter account had already been reported to Discord and banned from the server.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1350124411876937990)** (1 messages): 

> `Discord impersonation, Discord account security` 


- **Caroline Frasca imposter spamming DMs**: A spam account named *caroline_frascaa* with the nickname *Caroline* has been DM'ing folks while impersonating a real user.
   - The user posted a [screenshot of the fake account](https://cdn.discordapp.com/attachments/1098765954302873621/1350124411906293851/Impersonating_Account.png?ex=67d598d7&is=67d44757&hm=bd212e9e154251a202378828ccf61282fd69df840ade2eb535738fc7d7e248cb&) and updated their profile to help others easily identify the real account.
- **Discord server bans impersonating account**: The impersonating account *caroline_frascaa* has been reported to Discord and banned from this server.
   - Discord admins encourage reporting of any impersonating accounts.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/)** (1 messages): 

soracc: Yea, we use it in the stdlib (e.g. in `base64`) as well.
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1350148608418254940)** (1 messages): 

> `Self-Evaluation, Self-Reflection, Self-Refinement, Oracle Feedback` 


- **Self-Reflection Needs External Evaluation**: A member asked how statements about **self-evaluation** from the first lecture correspond to the second, as they seemed to contradict each other.
   - They pointed out that the first lecture mentioned that **self-reflection** and **self-refinement** work with good external evaluation, and that **self-correction** without oracle feedback hurts reasoning performance.
- **Clarification on Self-Evaluation in Lectures 1 & 2**: The user sought clarification on the apparent contradiction between the lectures regarding **self-evaluation**.
   - Specifically, they noted the emphasis on **self-evaluation and improvement** in the second lecture, while the first lecture highlighted the importance of external evaluation and the potential harm of self-correction without oracle feedback.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1350135872153976942)** (1 messages): 

> `Vertex, AWS` 


- **Vertex Gets Ready for 1.6**: **Version 1.6** is not yet available on **Vertex**, but is coming soon.
   - It will also be available on other platforms like **AWS**.
- **AWS Access to 1.6**: Version **1.6** will be available on other platforms like **AWS** in the near future.
   - This should allow customers on **AWS** to access the new features.


  

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

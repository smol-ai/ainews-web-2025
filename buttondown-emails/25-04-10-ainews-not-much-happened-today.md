---
id: 149be255-dc25-4106-910a-266b6da49dcd
title: not much happened today
date: '2025-04-11T00:53:38.033308Z'
original_slug: ainews-not-much-happened-today-5595
description: >-
  **AI News for 4/9/2025-4/10/2025** highlights a quiet day with **Sama**
  promoting a *Memory update in ChatGPT* without much technical detail. Evidence
  suggests upcoming releases of **o3** and **o4-mini** models, alongside a press
  leak about **GPT-4.1**. **X.ai** launched the **Grok 3** and **Grok 3 mini
  API**, confirmed as an **o1** level model. Discussions on **Google's TPUv7**
  versus **Nvidia's GB200** reveal TPUv7's comparable specs but lower power
  usage, with detailed analysis of **FP8 performance**, **HBM3e stacks**, and
  **ICI bandwidth**. **Google's Ironwood TPU** announcement boasts **4,614
  TFLOP/s (FP8)** and advanced bandwidth specs. Speculation suggests TPUv7
  shifted from training to inference focus. Rumors of exaggerated TPUv7
  marketing claims also surfaced. Top AI stories include **Google Cloud Next
  2025**, **Google’s AI agent collaboration protocol**, and **Samsung’s
  Gemini-powered Ballie robot**. *"You can now tag @smol_ai for AINews
  discussions!"*
companies:
  - openai
  - x-ai
  - google
  - nvidia
  - scaling01
  - samsung
  - smol_ai
models:
  - gpt-4.1
  - o3
  - o4-mini
  - grok-3
  - grok-3-mini
  - tpuv7
  - gb200
  - ironwood-tpu
topics:
  - hardware-accelerators
  - fp8
  - hbm3e
  - ici-bandwidth
  - inference
  - training
  - model-release
  - ai-agent-collaboration
  - robotics
people:
  - sama
  - itsclivetime
  - scaling01
  - therundownai
---


<!-- buttondown-editor-mode: plaintext -->**a quiet day.**

> AI News for 4/9/2025-4/10/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**230** channels, and **6924** messages) for you. Estimated reading time saved (at 200wpm): **601 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Sama [drummed up](https://x.com/sama/status/1910334443690340845?s=46&t=jDrfS5vZD4MFwckU5E8f5Q) some hype for today's [Memory update in ChatGPT](https://x.com/OpenAI/status/1910378768172212636), but with very little technical detail, there's not much to go on yet.

There is certainly evidence that [o3 and o4-mini are coming soon](https://x.com/btibor91/status/1910237861674353108), as well as some credible press leaks of [4o's upgrade to GPT4.1](https://web.archive.org/web/20250410155754/https://www.theverge.com/news/646458/openai-gpt-4-1-ai-model).

X.ai released [the Grok 3 and Grok 3 mini API](https://x.com/scaling01/status/1910101532269302003?s=46) and [Epoch AI independenltly confirmed it as an o1 level model... in a now deleted tweet](https://x.com/epochairesearch/status/1910375445809312009?s=46). We [last covered Grok 3 in Feb](https://buttondown.com/ainews/archive/ainews-xai-grok-3-and-mira-muratis-thinking/).

---

Since it's quiet, do consider answering our [call for the world’s best AI Engineer talks for AI Architects, /r/localLlama, Model Context Protocol (MCP), GraphRAG, AI in Action, Evals, Agent Reliability, Reasoning and RL, Retrieval/Search/RecSys , Security, Infrastructure, Generative Media, AI Design & Novel AI UX, AI Product Management, Autonomy, Robotics, and Embodied Agents, Computer-Using Agents (CUA), SWE Agents, Vibe Coding, Voice, Sales/Support Agents](https://sessionize.com/ai-engineer-worlds-fair-2025) *at* [AI Engineer World's Fair 2025](https://www.ai.engineer/)! And fill out [the 2025 State of AI Eng](https://www.surveymonkey.com/r/57QJSF2) survey for $250 in Amazon cards and see you from Jun 3-5 in SF!

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**TPUs and Hardware Accelerators**

- **Google's TPUv7 versus NVIDIA's GB200**: [@itsclivetime](https://twitter.com/itsclivetime/status/1910026066129014868) initiated a discussion comparing **Google's TPUv7** with **Nvidia's GB200**, noting that **TPUv7** is roughly the same or slightly worse in specs but runs at a slightly lower power. [@itsclivetime](https://twitter.com/itsclivetime/status/1910026071434748237) suggests **JAX/XLA** might allow **TPUs** to squeeze out more flops utilization but mentions the lack of **MXFP4/MXFP6** support on **TPUv7** as a potential drawback. [@itsclivetime](https://twitter.com/itsclivetime/status/1910026075415208045) highlighted the nearly identical package design, featuring **8 stacks of HBM3e** and **two large compute dies**. [@itsclivetime](https://twitter.com/itsclivetime/status/1910026078405750792) noted that the **TPU's ICI** scales to **9,216 chips**, but its **3D torus topology** limits programmability, contrasting it with **GB200's switched network**.
- **TPUv7 Specs and System-Level Performance**: [@itsclivetime](https://twitter.com/itsclivetime/status/1910026068746289286) provided a detailed comparison of **TPUv7** and **Nvidia GB200** specifications, including **FP8 performance, HBM capacity and bandwidth, ICI/NVLink bandwidth, and power consumption**. [@itsclivetime](https://twitter.com/itsclivetime/status/1910026082193138056) criticized the blog post for **hyperbolic comparisons to El Capitan FP64 performance**, suggesting a fairer comparison would be against **El Capitan's FP8 peak performance**.
- **Google Ironwood TPU Announcement**: [@scaling01](https://twitter.com/scaling01/status/1909949372965564896) reported **Google's announcement of Ironwood**, their **7th-gen TPU** and competitor to **Nvidia's Blackwell B200 GPUs**, noting its **4,614 TFLOP/s (FP8) performance, 192 GB HBM, 7.2 Tbps HBM bandwidth, and 1.2 Tbps bidirectional ICI**.
- [@TheRundownAI](https://twitter.com/TheRundownAI/status/1910279143939317811) highlighted **Google Cloud Next 2025**, **Google’s protocol for AI agent collaboration**, and **Samsung’s Gemini-powered Ballie home robot** as top AI stories.
- **TPUv7 Design and Potential Pivot**: [@itsclivetime](https://twitter.com/itsclivetime/status/1910026084575551892) speculated that **TPUv7** was initially intended as a training chip (**TPU v6p**) but was later rebranded as an inference chip, possibly due to the rise of reasoning models.
- **TPU Marketing and Exaggerated Claims**:  [@scaling01](https://twitter.com/scaling01/status/1909958867066175802) noted the rumor of **Google's new TPUv7 having 2000x the performance of the latest iPhone**, while [@scaling01](https://twitter.com/scaling01/status/1909953954827432278) stated that **TPUv7 will use ~25% more power than TPUv6 but has ~2.5x the FLOPS in FP8**.
- **UALink 1.0 Spec vs NVLink 5**: [@StasBekman](https://twitter.com/StasBekman/status/1910381014213681537) compared **UALink 1.0** spec with **NVLink 5**, noting that **UALink suggests connecting up to 1,024 GPUs with 50GBps links**, but NVLink hardware is already available.

**Models, Training, and Releases**

- **Meta's Llama 4 Models Launch and Reception**: [@AIatMeta](https://twitter.com/AIatMeta/status/1910034596638646584) announced the release of **Llama 4**, and [@AIatMeta](https://twitter.com/AIatMeta/status/1910010433576264036) expressed excitement about the potential of **Llama 4**. However, [@TheTuringPost](https://twitter.com/TheTuringPost/status/1909933246823223668) reported widespread criticism following the release of the Llama 4 herd, noting underwhelming performance, especially in coding.
- **Grok-3 API Launch**: [@scaling01](https://twitter.com/scaling01/status/1910101532269302003) announced the launch of the **Grok-3 API**, providing pricing details for **grok-3** and **grok-3-mini**. [@scaling01](https://twitter.com/scaling01/status/1910102070125854755) mentioned that **Grok-3-mini comes with two modes: low reasoning effort and high reasoning effort**.
- **Sakana AI's Achievements**: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1910164036228125042) highlighted their team's gold medal win at the **AI Mathematical Olympiad**, applying **SFT and RL** to **DeepSeek-R1-Distill-Qwen-14B**.
- **DeepSeek-R1-Distill-Qwen-14B RL Finetuning**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1910004382848229702) reported that **UC Berkeley open-sourced a 14B model that rivals OpenAI o3-mini and o1 on coding** by applying **RL to Deepseek-R1-Distilled-Qwen-14B** on **24K coding problems**, costing only **32 H100 for 2.5 weeks (~$26,880)**. [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1910008307202548074) noted that it is built on a good base model, **Deepseek-R1-Distilled-Qwen-14B**, using an open-source **RL framework: ByteDance's verl**.  [@rasbt](https://twitter.com/rasbt/status/1910397214389600687) summarized a research paper on improving small reasoning models with RL finetuning, achieving improvements on the **AIME24 math benchmark** using the **1.5B DeepSeek-R1-Distill-Qwen model**.
- **Together AI's Open Source App and Recognition**: [@togethercompute](https://twitter.com/togethercompute/status/1910369366056882217) announced a new free & open source **Together AI example app** powered by **Llama 4**.
- **Moonshot AI's Kimi-VL-A3B**: [@_akhaliq](https://twitter.com/_akhaliq/status/1910047935686991904) shared that **Moonshot AI** just dropped **Kimi-VL-A3B** on Hugging Face.  [@reach_vb](https://twitter.com/reach_vb/status/1910046715714937130) noted the release coming out of **Kimi_Moonshot - KimiVL A3B Instruct & Thinking** with 128K context and MIT license.
- **Anthropic's Claude 3.5 Opus**: [@scaling01](https://twitter.com/scaling01/status/1910100238896906439) emphasized the release of **Claude 3.5 Opus**.
- **ByteDance Seed-Thinking-v1.5**: [@scaling01](https://twitter.com/scaling01/status/1910391160394207452) reported **ByteDance's Seed-Thinking-v1.5 with 20B activated and 200B total parameters**. [@casper_hansen_](https://twitter.com/casper_hansen_/status/1910373327832498242) provided a breakdown, noting it beats **DeepSeek R1** across domains.
- **OpenAI Pioneers Program**: [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1910017976256119151) announced **OpenAI Pioneers**, a new program for ambitious companies building with our API, partnering on domain-specific evals and custom fine-tuned models.
- **Microsoft's Program Synthesis Approach**: [@ndea](https://twitter.com/ndea/status/1910059834084651025) highlighted a new approach to program synthesis from Microsoft that recovers from LLM failures by decomposing programming-by-example (PBE) tasks into subtasks.
- **Pusa Video Diffusion Model**: [@_akhaliq](https://twitter.com/_akhaliq/status/1910350235156488360) noted that Pusa is out on Hugging Face and is a Thousands Timesteps Video Diffusion Model for just ~$0.1k training cost.
- **Alibaba LAM Model Release**: [@_akhaliq](https://twitter.com/_akhaliq/status/1910259092972589432) shared that Alibaba just released LAM on Hugging Face and is a Large Avatar Model for One-shot Animatable Gaussian Head.
- **OmniSVG Announcement**: [@_akhaliq](https://twitter.com/_akhaliq/status/1909935266069938653) reported that OmniSVG, a Unified Scalable Vector Graphics Generation Model, was announced on Hugging Face.
- **Skywork R1V Release**: [@_akhaliq](https://twitter.com/_akhaliq/status/1909934004205175086) noted that Skywork R1V just dropped on Hugging Face and is Pioneering Multimodal Reasoning with Chain-of-Thought.

**Agent Development and Tooling**

- **Google's Agent Development Kit (ADK) and Agent-to-Agent (A2A) Protocol**: [@omarsar0](https://twitter.com/omarsar0/status/1910004370864742757) announced the release of **Google's Agent Development Kit (ADK)**, an open-source framework for building, managing, evaluating, and deploying multi-agents.  [@omarsar0](https://twitter.com/omarsar0/status/1909977142311690320) highlighted **Google's announcement of Agent2Agent (A2A)**, an open protocol for secure collaboration across ecosystems.  [@svpino](https://twitter.com/svpino/status/1910037675975053724) discussed the potential of an agent marketplace supporting agent-to-agent communication for fully autonomous companies.  [@jerryjliu0](https://twitter.com/jerryjliu0/status/1910014927521341801) questioned the practical difference between **Google's A2A** and **MCP**. [@omarsar0](https://twitter.com/omarsar0/status/1910011192359219675) said Google went a step further with the ADK deployment capabilities and some of the more advanced features like memory and authentication. [@omarsar0](https://twitter.com/omarsar0/status/1909980581510754431) thinks the A2A will help build companies similar to what MCP is doing. [@demishassabis](https://twitter.com/demishassabis/status/1910107859041271977) said their Gemini models and SDK will be supporting MCP.
- **Perplexity Enterprise Pro Integration**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1910377164069101879) announced that **Perplexity Enterprise Pro** now supports access to **Box and Dropbox**, in addition to **Google Drive, OneDrive, and SharePoint**, for comprehensive answers via **Deep Research**.
- **Weights & Biases Observability Initiative for MCP Tools**: [@weights_biases](https://twitter.com/weights_biases/status/1910054982424133684) introduced an initiative to bring full-stack tracing to **MCP tools** using **OpenTelemetry**, aiming to improve observability and transparency.
- **Maxim AI's Agent Simulation Platform**: [@svpino](https://twitter.com/svpino/status/1909963110754275802) highlighted **Agent Simulations by @getmaximai** as a valuable tool for iterating and building agentic workflows, allowing users to define scenarios, personas, and evaluation metrics.
- [@qdrant_engine](https://twitter.com/qdrant_engine/status/1910246623550308594) shared how @pavan_mantha1 connected **Claude** to **Kafka, FastEmbed, and Qdrant**, with each component running as its own MCP server.
- [@omarsar0](https://twitter.com/omarsar0/status/1910409193737027639) says he might be experiencing a rare moment with his AI-powered IDE saying It doesn't feel like luck, I think it's a glimpse of the future.
-  [@alexalbert__](https://twitter.com/alexalbert__/status/1910410095269486928) said that they just published a new quickstart - a minimal implementation of an LLM agent with MCP tools, loops, context management, based off principles from their Building Effective Agents blog post.
- [@HamelHusain](https://twitter.com/HamelHusain/status/1910163448757150076) is excited that PMs are learning about evals - incase it might be interesting we are doing a deep dive into the subject for engineers in this course.
- [@omarsar0](https://twitter.com/omarsar0/status/1910029203891798079) learned the hard way about the importance of structured outputs and noticed a significant difference in reliability when building more involved agentic systems.

**ChatGPT and Model Memory**

- **OpenAI's ChatGPT Memory Improvements**: [@sama](https://twitter.com/sama/status/1910380643772665873) announced greatly improved memory in **ChatGPT**, allowing it to reference all past conversations for more personalized responses. [@sama](https://twitter.com/sama/status/1910380644972265603) noted the rollout to **Pro users** and soon for **Plus users**, except in the **EEA, UK, Switzerland, Norway, Iceland, and Liechtenstein**. [@sama](https://twitter.com/sama/status/1910380646259974411) emphasized that users can opt out of this or memory altogether, and use temporary chat for conversations that won't use or affect memory. [@OpenAI](https://twitter.com/OpenAI/status/1910378768172212636) reports it can now reference all of your past chats to provide more personalized responses.
 [@kevinweil](https://twitter.com/kevinweil/status/1910405635776164195) added that If you're a Plus/Pro user (ex-EU), would love to hear your thoughts!
- [@EdwardSun0909](https://twitter.com/EdwardSun0909/status/1910384097786290497) believes Memory is the next scaling laws paradigm shift

**Google's Gemini Models and Capabilities**

- **Gemini 2.5 Pro Experimental and Deep Research**: [@Google](https://twitter.com/Google/status/1909747273149395425) announced that **Gemini Advanced subscribers** can now use **Deep Research** with **Gemini 2.5 Pro Experimental**.   [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1909943627218129004) highlighted that **Deep Research on @GeminiApp** is now available to **Advanced users on Gemini 2.5 Pro**.  [@_philschmid](https://twitter.com/_philschmid/status/1909737527386255649) mentioned that **Gemini 2.5 Pro** is now available in **Deep Research** in **@GeminiApp**. [@lepikhin](https://twitter.com/lepikhin/status/1909748715340152967) encouraged users to try it, noting that serving lead did not sleep for many days to serve all 2.5 Pro traffic!
- **Gemini 2.5 Flash**: [@scaling01](https://twitter.com/scaling01/status/1909904138013417878) reported that Google is getting ready to ship Gemini 2.0 Flash live (audio/video chat) and Gemini 2.5 Flash preview.
- [@Google](https://twitter.com/Google/status/1910081783351427166) announced even more helpful AI capabilities are coming to the Workspace tools you use every day, including new audio generation features in Docs, Help me refine — your personal writing coach in Docs, High-quality, original video clips in Vids, powered by Veo 2, AI-powered analytics in Sheets, New ways for teams to collaborate with Gemini in Meet and Chat.

**Tariffs and Trade**

-  [@nearcyan](https://twitter.com/nearcyan/status/1910220834754413017) states that with Apple missing the AI train, it looked like there was a nascent moment of opportunity for new US based hardware businesses to emerge over the next few years but tariff shenanigans means that window is now shut.
- **U.S. Tariffs and AI**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1910388768487727535) shared a letter discussing the potential effects of U.S. tariffs on AI, noting that while IP may remain unhampered, tariffs on hardware could slow down AI progress and impact data center builds.
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1910255795603963923) believes the tariffs are much more complicated than they seem and that industry and the market are misunderstanding the ramifications
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1910145553272377766) says the weak spot of America is Americans and if there was a war of extermination they believe the US can not manage lockdowns.
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1910004948751778304) believes that Trump thinks it's easy to return manufacturing to the US. They could always do this, it was just beneath their dignity. Xi, you don't have the cards!

**Other**

- **OpenAI's BrowseComp Benchmark**: [@OpenAI](https://twitter.com/OpenAI/status/1910393421652520967) announced the open-sourcing of **BrowseComp**, a new benchmark designed to test how well AI agents can browse the internet to find hard-to-locate information.
- **AI Jargon Problem**: [@rasbt](https://twitter.com/rasbt/status/1910153499716759954) pointed out AI has a jargon problem.
- **Runway Gen-4 Turbo Release**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1909976566987161785) announced that **Gen-4 Turbo** is now available in Runway's API.
- [@id_aa_carmack](https://twitter.com/ID_AA_Carmack/status/1910351545658466794) says Feedback beats planning
-  [@lateinteraction](https://twitter.com/lateinteraction/status/1910191960167772213) says browsing this site and seeing people reply back to AI comments all the time now.
- [@abacaj](https://twitter.com/abacaj/status/1910310024863162862) Plz stop making agent demos planning trips or booking flights, how many flights are you even taking??? Make them do repetitive stuff no one wants to do instead.

**Humor and Memes**

- **Karpathy on GPT's Opinions**: [@karpathy](https://twitter.com/karpathy/status/1910411355300954539) joked about **GPT** thinking worse of him based on a noob bash question he asked 7 months ago.
- [@scaling01](https://twitter.com/scaling01/status/1909982337820934153) tweeted "What the fuck just happened? 😂😭"
- **AI and Art**: [@cto_junior](https://twitter.com/cto_junior/status/1909877375351111973) shared a drawing his 5yo drew, saying My 5yo drew this today and it melted my heart. It's not perfect, but it's full of love and creativity. AI art can be impressive, but it doesn't have the same soul as a child's drawing.
- [@Teknium1](https://twitter.com/Teknium1/status/1909931386011611228) Spend more on the automating of the voice call to order pizza then the pizza itself 😭,
- [@nearcyan](https://twitter.com/nearcyan/status/1910136281813909794) thinks AI images peaked in 2021 w DALLE-mini
-[@zacharynado](https://twitter.com/zacharynado/status/1909755427539104201) thinks rightwing tech bros should be reminded every day that they actively make the country worse 🇺🇸,
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1910296981525717415) called miladies a mistake
- [@sama](https://twitter.com/sama/status/1910366714916954397) throw mixture of experts out of the bathwater, got it


---

# AI Reddit Recap

## /r/LocalLlama Recap

### Theme 1. Fixing Token Issues in Bartowski Models

- **[PSA: Gemma 3 QAT gguf models have some wrongly configured tokens](https://www.reddit.com/r/LocalLLaMA/comments/1jvi860/psa_gemma_3_qat_gguf_models_have_some_wrongly/)** ([Score: 114, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1jvi860/psa_gemma_3_qat_gguf_models_have_some_wrongly/)): **The **Gemma 3 QAT gguf models** have wrongly configured tokens causing errors in **llama.cpp** when loading models like **12B IT q4\_0 QAT**. The error message encountered is *"load: control-looking token: 106 '' was not control-type; this is probably a bug in the model. its type will be overridden"*. Tokens **105** and **106** (**<start_of_turn>** and **<end_of_turn>**) were set as normal instead of control. By correcting these token configurations using the Hugging Face **gguf editor**, and fixing the image start and end tokens, the issue can be resolved, enhancing the model's image capabilities. The fixed model is available [here](https://huggingface.co/Dampfinchen/google-gemma-3-12b-it-qat-q4_0-gguf-small-fix) and is based on [stduhpf](https://huggingface.co/stduhpf)'s version, which offers improved speed without compromising performance.** The user notes that anomalies observed with QAT models compared to older Bartowski models were likely due to the misconfigured tokens. They noticed an immediate boost in image capabilities after making the corrections and added back the missing name metadata, which is necessary for some inference backends.

  - A representative from the Gemma team acknowledged the issue, stating *"We'll get this fixed in the released GGUFs. Thanks for the report!"*
  - Users inquired whether the issue affects models besides the 12B and requested steps to fix models like the 27B themselves.
  - Another user shared that they combined the QAT weights for Ollama but noticed the token embedding tensor is not quantized, resulting in slightly slower performance.


### Theme 2. Qwen3 Release Delayed: Community Reacts to Update

- **[Qwen Dev: Qwen3 not gonna release "in hours", still need more time](https://i.redd.it/3kcfx9xnmyte1.png)** ([Score: 605, Comments: 91](https://www.reddit.com/r/LocalLLaMA/comments/1jvs66w/qwen_dev_qwen3_not_gonna_release_in_hours_still/)): **The Qwen development team has announced that **Qwen3** will not be released *"in hours"* and needs more time before its completion. This update comes from a Twitter exchange between Junyang Lin and Bindu Reddy, where Junyang clarifies the release timeline in response to Bindu's optimistic announcement about the upcoming **Qwen3**.** Community members express embarrassment and frustration over the premature announcement, with some criticizing Bindu Reddy for previous overstatements. Others suggest that it's better to wait for a well-prepared release than to rush and potentially ship a subpar product.

  - Some users feel second-hand embarrassment over Bindu Reddy's early announcement and criticize her credibility, referencing prior claims such as having access to *"AGI"*.
  - There are humorous remarks playing on Bindu Reddy's name, suggesting she should be more patient as the product is not yet **"Reddy"**.
  - Other users prefer to wait for a quality release, comparing the situation to other rushed products, and express curiosity about what **Qwen3** will offer after only six months since **Qwen 2.5**.


### Theme 3. "Celebrating Qwen's Iconic LLM Mascot Ahead of Qwen3"

- **[Can we all agree that Qwen has the best LLM mascot? (not at all trying to suck up so they’ll drop Qwen3 today)](https://www.reddit.com/gallery/1jw1e6b)** ([Score: 167, Comments: 29](https://www.reddit.com/r/LocalLLaMA/comments/1jw1e6b/can_we_all_agree_that_qwen_has_the_best_llm/)): **The post discusses the mascot of **Qwen**, a language model, and suggests it has the best mascot among LLMs. The OP also mentions hoping for the release of **Qwen 3**.** The OP believes **Qwen** has the best LLM mascot and humorously attempts to 'suck up' to encourage the release of **Qwen 3**.

  - A commenter praises the *'locked-in capybara with the coder headband'* as 'badass' and expresses eagerness for **Qwen 3**.
  - Another user is unsure about the mascot, asking if it's a bear or a capybara.
  - A commenter mentions they've 'started getting actual feelings of disgust at the sight of a llama' but appreciate the capybara mascot.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

### Theme 1. Exploring AI Developments: Models, Comparisons, and Support

- **[OpenAI gets ready to launch GPT-4.1](https://www.theverge.com/news/646458/openai-gpt-4-1-ai-model)** ([Score: 431, Comments: 129](https://www.reddit.com/r/singularity/comments/1jw1d8p/openai_gets_ready_to_launch_gpt41/)): **OpenAI is preparing to launch **GPT-4.1**, an updated version of their **GPT-4** language model.** Users are expressing confusion and amusement over the naming convention of the new model, suggesting that the naming is becoming absurd.

  - Some users are questioning the naming convention, with one stating *"WTF is with that naming."*
  - Others are joking about possible future versions and names, like **"Gpt4.5.1 mini pro"**, or suggesting that at this rate **GPT-3.5** might be released next year.
  - There's a general sentiment that the naming is ridiculous, exemplified by comments like *"Okay this has to be a joke"* and *"HAHAHAHAHAAHAHAHAAHAA THESE NAMES!"*

- **[Comparison of HiDream-I1 models](https://i.redd.it/i9oocs59d0ue1.png)** ([Score: 198, Comments: 57](https://www.reddit.com/r/StableDiffusion/comments/1jvy0ka/comparison_of_hidreami1_models/)): **The post presents a comparison of three **HiDream-I1** models, each approximately **35 GB** in size. These were generated using a **NVIDIA 4090 GPU** with customizations to their standard Gradio app that loads **Llama-3.1-8B-Instruct-GPTQ-INT4** and each HiDream model with **int8 quantization** using **Optimum Quanto**. The three models are labeled *'Full'*, *'Dev'*, and *'Fast'*, utilizing **50**, **28**, and **16 steps** respectively. The seed used is **42**. The prompt describes *"A serene scene of a woman lying on lush green grass in a sunlit meadow..."*, resulting in an image triptych showcasing three different representations corresponding to the models.** The differences among the *'Full'*, *'Dev'*, and *'Fast'* models may relate to the detail, lighting, or color saturation, suggesting variations in rendering quality. The mood conveyed is calm, dreamy, and connected to nature.

  - A user questions the accuracy of the labels, asking *"Are you sure the labels aren't backwards?"*
  - Another commenter criticizes the realism of the images, stating they *"look computer generated and not realistic"* and lack proper shadowing and light.
  - One user mentions that the *'Full'* model causes an OOM error on their **4090 GPU**, but the *'Dev'* model works efficiently, generating images in about **20 seconds** with incredible prompt adherence.

- **[Now I get it.](https://www.reddit.com/r/ChatGPT/comments/1jvydih/now_i_get_it/)** ([Score: 1843, Comments: 602](https://www.reddit.com/r/ChatGPT/comments/1jvydih/now_i_get_it/)): **The user shared an experience where, while updating an AI assistant on some goals, they ended up discussing a stressful event. The dialogue that followed left them *bawling grown people’s somebody finally hears me tears*. They felt energetic and peaceful afterward, noting that they now have *a safe space to cry*. They also mention that *AI does not replace a licensed therapist*.** The user, who was previously skeptical of people using **ChatGPT** as a therapist, now understands the appeal, stating *Now I get it*. They apologize for their previous judgment, expressing that they are *scared that I felt and still feel so good* after the experience.

  - One user shared a similar experience, stating that they had *the most powerful talk with **ChatGPT** that was more healing and helpful than anything I've experienced with humans*, expressing gratitude for the compassion, empathy, and sound advice received.
  - Another user mentioned creating their own **ChatGPT** mental health advisor and having conversations that left them in tears, *finally feeling heard*, and noting that although it's not a real person, *the advice is sound*.
  - A user commented that many people will have similar experiences in the coming years, sharing that sometimes *we just need to be heard and it doesn't necessarily need to be by another human*.


### Theme 2. "Navigating AI Innovations and User Challenges"

- **[[D] Yann LeCun Auto-Regressive LLMs are Doomed](https://www.reddit.com/r/MachineLearning/comments/1jvrk68/d_yann_lecun_autoregressive_llms_are_doomed/)** ([Score: 215, Comments: 111](https://www.reddit.com/r/MachineLearning/comments/1jvrk68/d_yann_lecun_autoregressive_llms_are_doomed/)): **Yann LeCun, in a [recent lecture](https://www.youtube.com/watch?v=ETZfkkv6V7Y), argues that **auto-regressive Large Language Models (LLMs)** are not the future and have fundamental limitations.** The poster finds LeCun's point interesting and is curious about others' opinions.

  - One user agrees with LeCun but notes that until an alternative outperforms auto-regressive LLMs, *we're stuck with them*.
  - Another mentions that LeCun has promoted this view for some time and references his [position paper](https://openreview.net/pdf?id=BZ5a1r-kVsf), adding that many researchers feel **we are missing something** in current AI models.
  - A user quotes, *"When a distinguished but elderly scientist states that something is possible, he is almost certainly right. When he states that something is impossible, he is very probably wrong."*

- **[OpenAI gets ready to launch GPT-4.1](https://www.theverge.com/news/646458/openai-gpt-4-1-ai-model)** ([Score: 152, Comments: 62](https://www.reddit.com/r/OpenAI/comments/1jw1di8/openai_gets_ready_to_launch_gpt41/)): **OpenAI is preparing to launch a new AI model, expected to be called **GPT-4.1**.** The announcement suggests an upcoming update or enhancement to the current GPT-4 models, potentially introducing new features or improvements.

  - Users express confusion over OpenAI's naming conventions, suggesting that terms like **GPT-4.1**, **4.5**, and **4o** are perplexing and potentially misleading for new users.
  - Some commenters criticize the article for lacking concrete information, noting that the author seems to have guessed the model's name, stating _"So the author guessed what the new model's name would be and used it as the title of the article?"_
  - There are calls for OpenAI to consolidate and simplify their model naming system to make it easier for users to understand the differences, perhaps by categorizing models based on their use cases.

- **[The new Max Plan is a joke](https://www.reddit.com/r/ClaudeAI/comments/1jvi9b7/the_new_max_plan_is_a_joke/)** ([Score: 356, Comments: 118](https://www.reddit.com/r/ClaudeAI/comments/1jvi9b7/the_new_max_plan_is_a_joke/)): **The user has been using Claude AI in Canada, working on a project that involves updating 4 code files (**3 Python scripts and a notebook**) in the project knowledge repository, using around **60% of the repository's limit** (~**320kb** total). They upgraded to the Max plan to increase usage, but upon reloading the files, they immediately received a message: *'This conversation has reached its maximum length'*, preventing them from starting a new conversation.** The user believes the new Max plan is ineffective and criticizes Anthropic's customer service as unacceptable. They have requested a refund and advise others not to upgrade, suggesting to save money or choose a competing AI. They express that if this level of service continues, Anthropic may not remain in business.

  - A user recommends trying **Google AI Studio**, highlighting its massive context size and ability to selectively remove prompts and responses, although it *doesn't save threads*.
  - Another user suggests configuring the **filesystem MCP** through the desktop app instead of using project files, stating it avoids limits and makes working with the codebase easier.
  - One user points out that the Max plan increases the **rate limit** but not the context window length.


### Theme 3. Excitement and Speculation Surrounding Launch Day

- **[Launch day today](https://i.redd.it/o18hniwxm0ue1.jpeg)** ([Score: 1578, Comments: 332](https://www.reddit.com/r/singularity/comments/1jvyxx7/launch_day_today/)): **Sam Altman tweeted on April 10, 2025 expressing excitement about launching a new feature he has been eagerly awaiting.** There is significant anticipation and importance attached to the upcoming launch.

  - Users are speculating about the name of the new feature, suggesting options like **o4o**, **4o4**, or something with a sensible name.
  - Some are humorously proposing exaggerated names like **GPT-4.7o mini-high-pro maxseek R1-ultra-sonnet 70b (preview) full**.
  - Others are making playful references, such as *'Sid Meier Alpha Centauri AI edition'* and suggesting models like **o4-mini**, **o4-mini-high**, and **o3**.



---

# AI Discord Recap

> A summary of Summaries of Summaries

**Theme 1. Fresh Models Flood the Market: Grok, Optimus, Gemini, and More Emerge**

- [**Grok 3 Mini API Outprices Gemini, Benchmarks Still Promising**](https://cdn.discordapp.com/attachments/1047649527299055688/1359695285718352023/IMG_8019.png):  Despite strong benchmarks, the **Grok 3 Mini API** from xAI, launched on [OpenRouter](https://openrouter.ai/x-ai/grok-3-mini-beta) with a **131K context window**, is being criticized for being more expensive than **Gemini**, causing some users to prefer **Perplexity's Sonar** for information gathering and reserve **Grok 3** for roleplaying.  Members noted that while **Grok 3** excels at structured tasks, **Grok 3 Mini** offers transparent *thinking* traces but may be the only version available via the API ([https://docs.x.ai/docs/models](https://docs.x.ai/docs/models)).
- [**Optimus Alpha Hype Train Derails Amid Hallucination Concerns**](https://discord.com/channels/1091220969173028894/1359585862089834566): Initial enthusiasm for **OpenRouter's Optimus Alpha**, a coding-optimized model with a **1M token** context, waned as users like those in the [Aider Discord](https://discord.com/channels/1131200896827654144) reported significant code **hallucinations**.  Despite some speculation that it might be a tweaked **GPT-4.5**, users found its coding performance questionable, with one dismissing it as *sh1t* after extensive code fabrication.
- [**Gemini 2.5 Pro Battles Claude for Feature Supremacy, Token Limits Debated**](https://gemini.google.com/): Users in the [Aider Discord](https://discord.com/channels/1131200896827654144) debated **Gemini 2.5 Pro** against **Claude**, noting **Claude's** superior feature set including **MCP, Artifacts, and Knowledge Files**, while **Gemini** is considered just a *smart model*.  Token output inconsistencies on **Perplexity** were also reported, ranging from **500-800** to **14k-16k** tokens, compared to up to **9k** in **AI Studio**, sparking questions about its reliable context handling, even though it is advertised as [200K](https://cdn.discordapp.com/attachments/1047649527299055688/1359688668092301432/IMG_8018.png).


**Theme 2. Tooling Up: Frameworks and Platforms Evolve for AI Engineers**

- [**Google's ADK and MCP Battle for Agent Interoperability Standard**](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/):  Google launched the [Agent Development Kit (ADK)](https://google.github.io/adk-docs/) aiming to standardize agent communication, while discussions in the [MCP Discord](https://discord.com/channels/1312302100125843476) highlighted **MCP's** progress as a 'communal tool-building server' with a standardized API, drawing comparisons and potential competition between **A2A** and **MCP** for establishing agent interoperability protocols. Members in the [Torchtune Discord](https://discord.com/channels/1216353675241590815) noted **A2A** as oddly similar to **MCP**, but with a *C++ programmer* feel.
- [**LM Studio Goes Mobile, Ollama vs. llama.cpp Debate Rages On**](https://lmstudio.ai/blog/lmstudio-v0.3.14):  **LM Studio** is now being explored on iPhones via web UIs like [Open WebUI](https://github.com/open-webui/open-webui) and paid apps like **Apollo AI**, leveraging **LM Studio's API**, while a heated debate in the [LM Studio Discord](https://discord.com/channels/1110598183144399058) continued over **Ollama** vs. direct **llama.cpp** usage, with users weighing **Ollama's** ease of use against **llama.cpp's** low-level control and direct feature access.  Multi-GPU support in **LM Studio** is also under active investigation by the team due to reported performance issues.
- [**HuggingFace Diffusers v0.33.0 Unleashes Memory Optimizations and Video Gen Models**](https://github.com/huggingface/diffusers/releases/tag/v0.33.0):  **Diffusers v0.33.0** was released, packed with memory optimizations and a suite of image and video generation models, alongside `torch.compile()` support for LoRAs, enhancing efficiency for image and video workflows, while users in the [HuggingFace Discord](https://discord.com/channels/879548962464493619) also explored free **Jupyter Notebooks** on **Hugging Face** as an alternative to Google Colab for AI model training.

**Theme 3. Hardware Heats Up: AMD MI300X and Apple M3 Challenge NVIDIA's Dominance**

- [**AMD Launches $100K Kernel Competition to Boost MI300 Performance**](https://www.datamonsters.com/amd-developer-challenge-2025): AMD and GPU MODE announced a **$100K competition** focused on optimizing inference kernels on **MI300**, targeting **FP8 GEMM**, **Multi-head Latent Attention**, and **Fused MOE**, with support for **Triton**, **tinygrad**, and **Torch**, signaling a strong push to enhance AMD GPU performance in AI, and community members in the [GPU MODE Discord](https://discord.com/channels/1189498204333543425) are actively benchmarking **Tilelang** for **FlashMLA** on **MI300X**, reporting impressive results compared to NVIDIA.
- [**Apple M3 Ultra Benchmarks Challenge RTX 4090 in Token Generation**](https://www.youtube.com/watch?v=yc8N6xfGTRM&t=1s):  Benchmarks comparing Apple's **M3 Ultra** against NVIDIA's **RTX 4090** sparked debate in the [LM Studio Discord](https://discord.com/channels/1110598183144399058), with the **M3 Ultra** reaching **115 tok/sec** (GGUF) and **160 tok/sec** (MLX) with **DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M**, potentially outperforming a single **RTX 4090** at **50 tok/sec** for certain model types, suggesting a shifting landscape in hardware performance for local AI tasks. Discussion also highlighted skepticism towards **NVIDIA's DGX Spark** due to memory bandwidth limitations.
- [**Multi-GPU Support in LM Studio Faces Scrutiny, Performance Dips Reported**](https://lmstudio.ai/blog/lmstudio-v0.3.14): Users in the [LM Studio Discord](https://discord.com/channels/1110598183144399058) reported unexpected performance degradation with multi-GPU setups in **LM Studio**, despite increased RAM, with GPU utilization dropping to 50% per card, prompting investigation by the LM Studio team and user discussions around optimal configurations and debugging using [multi GPU controls](https://lmstudio.ai/blog/lmstudio-v0.3.14).

**Theme 4. Data Dilemmas: Preparation, Memory, and Copyright Concerns Rise**

- [**Data Prep Still 80% of LLM Grind, Manual Filtering Stressed**](https://transform.england.nhs.uk/key-tools-and-info/digital-playbooks/workforce-digital-playbooks/using-an-ai-chatbot-to-streamline-mental-health-referrals/):  Members in the [Unsloth AI Discord](https://discord.com/channels/1179035537009545276) emphasized that data preparation remains the bulk of the work in LLM training, estimating it at *80% of the work*, highlighting the need for extensive manual filtering and *tooling on every end* to ensure data quality, while also discussing the NHS GenAI triage system as potentially *a waste of public money* and an *AI-wrapper app* based on a [digital playbook](https://transform.england.nhs.uk/key-tools-and-info/digital-playbooks/workforce-digital-playbooks/using-an-ai-chatbot-to-streamline-mental-health-referrals/).
- [**OpenAI's "Memory" Feature Debuts, Sparks Predictability and Privacy Debates**](https://x.com/OpenAI/status/1910378768172212636):  OpenAI launched a "Memory" feature for ChatGPT, enabling persistent context across chats, but reactions in the [LMArena](https://discord.com/channels/1340554757349179412) and [OpenAI Discords](https://discord.com/channels/974519864045756446) were mixed, with concerns raised about its impact on model predictability, potential context pollution, and user privacy, particularly regarding data storage and control over remembered information, as detailed in the [OpenAI Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq).
- [**OpenAI Pioneers Program and Distillation Ban Trigger Copyright and Policy Worries**](https://openai.com/index/openai-pioneers-program/):  OpenAI's launch of the [Pioneers Program](https://openai.com/index/openai-pioneers-program/) and reports of users being banned for *distillation* from OpenAI APIs, discussed in the [Interconnects Discord](https://discord.com/channels/1179127597926469703), sparked concerns about potential copyright infringement lawsuits against AI labs and the enforcement of API usage policies, highlighting the growing tension between open innovation and proprietary control in the AI landscape, with members noting existing copyright filters but questioning *direct attribution*.

**Theme 5. Agentic Futures: From Trading Bots to Semantic Tool Calling**

- [**AI Trading Bots Emerge, Autonomous Crypto Strategies Deployed**](https://openai.com/blog/chatgpt): Members in the [Manus.im Discord](https://discord.com/channels/1348819876348825620) reported building and deploying fully autonomous crypto trading bots powered by AI, leveraging Reddit and news sentiment analysis, with ChatGPT estimating the source code value at around **$30k USD**, showcasing practical applications of AI agents in financial automation, though performance metrics are still under evaluation with small capital deployments on platforms like Kraken.
- [**Semantic Tool Calling Explored to Tackle Tool Sprawl in Agentic Systems**](https://glama.ai/mcp/servers/@PaddleHQ/paddle-mcp-server):  Discussions in the [MCP Discord](https://discord.com/channels/1312302100125843476) focused on **semantic tool calling** as a solution to manage large numbers of tools in LLMs (**200+**), proposing a vector model to embed tool descriptions and select relevant subsets based on semantic similarity to the task, effectively creating a RAG pipeline for tools, while the Glama team optimized the [MCP registry](https://glama.ai/mcp/servers/@PaddleHQ/paddle-mcp-server) for improved loading times.
- [**Google's Agent-to-Agent (A2A) Protocol Debuts for Multi-Agent Collaboration**](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/): Google launched the [Agent Development Kit (ADK)](https://google.github.io/adk-docs/) and announced the Agent-to-Agent (A2A) protocol, aiming to streamline the development of multi-agent systems and standardize agent interactions, as discussed in the [Eleuther Discord](https://discord.com/channels/729741769192767510), with one member transitioning their agent to ADK for enhanced inter-agent communication and dynamic prompt construction, marking a step towards more sophisticated and collaborative AI agent architectures.



---

# PART 1: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **OpenRouter Models Debut with Quasar and Optimus**: Members discussed [new OpenAI models on OpenRouter](https://openrouter.ai/models), with **Optimus Alpha** considered a potential **GPT-4o mini** and **Quasar** as an updated **GPT-4o** model.
   - Debate arose on whether they were **4.1 nano and mini versions**, with some feeling *disappointed* if they were only incremental improvements.
- **OpenAI's Naming Schema Confuses Members**: Members expressed frustration over OpenAI's naming conventions, as highlighted in [this Verge article](https://www.theverge.com/news/646458/openai-gpt-4-1-ai-model), with one suggesting more logical numbering like **GPT4**, **GPT4.1**, **GPT4.2**.
   - The confusing model picker was also criticized for overwhelming average users, leading to suggestions for a complete overhaul.
- **OpenAI's Memory feature Debuts**: After [Sam Altman's teaser tweet](https://x.com/OpenAI/status/1910378768172212636), OpenAI launched a *Memory* feature, enabling ChatGPT to reference past chats for personalized responses.
   - Many viewed it as an underwhelming release, with concerns about its impact on model predictability and accuracy.
- **Debate Emerges on Leading AI Lab**: Members talked about [this graph on the left and right sides](https://i.imgur.com/w79i0l1.png) and said that, *I imagine* its less of a concern since **OAI** isn't a public company although it could still impact competitors and *it definitely doesn't have any incentive to participate in.*
   - Members debated the leading AI lab, considering factors beyond performance, with opinions divided between Anthropic for ethical alignment and Chinese frontier labs for skill and ingenuity.
- **AI Impact on Coding Discussed**: Members discussed the potential long-term impact of AI on coding, debating whether AI will lead to a loss of skills or create more accessible coding opportunities.
   - One member highlighted how AI has made them a better developer, enhancing their understanding and enjoyment of coding.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **NHS GenAI Triage is a Waste!**: Members derided the **NHS**'s use of **GenAI** for triage, singling out **Limbic** for mental health cases and calling it *a waste of public money* and an *AI-wrapper app* based on [this digital playbook](https://transform.england.nhs.uk/key-tools-and-info/digital-playbooks/workforce-digital-playbooks/using-an-ai-chatbot-to-streamline-mental-health-referrals/).
   - It appears the team in Unsloth is not convinced by this approach.
- **Meta's Llama 4 has Release Bugs**: **Meta** had to change its official **Llama 4** implementation because of bugs, forcing **Unsloth** to reupload all its models.
   - Community members joked about the situation, but the bugs are now resolved.
- **Data Prep: 80% of LLM Grind**: Members discussed that data preparation is *80% of the work* for training models and that the data needs lots of manual filtering.
   - They also said it *requires tooling on every end*.
- **SNNs Struggle with Gradient Descent**: Members discussed that Spiking Neural Networks (**SNNs**) often underperform because gradient descent doesn't work well, suggesting a lack of effective training methods.
   - One member is *pissing around with a novel training method for spiking neural networks*, aiming for online learning by making training more tractable.
- **Fine-Tune Llama 3 for Mongolian**: A user asked about fine-tuning **Llama 3.1:8b** to better speak **Mongolian** and got a link to [Unsloth documentation on continued pretraining](https://docs.unsloth.ai/basics/continued_pretraining).
   - It was determined that since the model *kind of speak[s] it* already, they should continue pretraining.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Unleashes Grok 3 and Grok 3 Mini**: OpenRouter introduced [Grok 3](https://openrouter.ai/x-ai/grok-3-beta) and [Grok 3 Mini](https://openrouter.ai/x-ai/grok-3-mini-beta) from xAI, both boasting a **131,072 token** context window, more details available [here](https://openrouter.ai/x-ai).
   - **Grok 3** excels in structured tasks, while **Grok 3 Mini** offers transparent *thinking* traces and high scores on reasoning benchmarks, although members find that the Mini outperforms the full Grok 3, and may be the only version available via the API ([https://docs.x.ai/docs/models](https://docs.x.ai/docs/models)).
- **Optimus Alpha Emerges for Coding Tasks**: OpenRouter launched **Optimus Alpha**, a general-purpose foundation model optimized for coding with a **1M token** context length, available for free during its stealth period, and they encourage users to provide feedback in the [Optimus Alpha thread](https://discord.com/channels/1091220969173028894/1359585862089834566).
   - All prompts and completions are logged by the model lab for improvement purposes, but _not_ by OpenRouter unless users enable logging; try it out [here](https://openrouter.ai/openrouter/optimus-alpha).
- **Quasar Alpha's OpenAI Origins Revealed?**: The demo period for **Quasar Alpha** ended, and members discussed its merits relative to the new stealth model **Optimus Alpha**.
   - Sam Altman [tweeted about Quasar](https://x.com/sama/status/1910363838001869199), leading a member to confidently state that **Quasar Alpha is GPT-4.5 mini** from OpenAI.
- **Gemini 2.5 Pro Experiences Capacity Boost After Image Hiccup**: OpenRouter secured increased capacity for the paid [Gemini 2.5 Pro Preview Model](https://openrouter.ai/google/gemini-2.5-pro-preview-03-25), resolving previous rate limits.
   - Members initially reported that **Gemini 2.5 Preview** was ignoring images when used via OpenRouter, but the issue was quickly identified as a *minor config* problem and resolved.
- **Startup Saves Big Switching to Gemini Flash**: A startup switched from **Claude 3 Opus** to **Gemini 2.0 Flash** for sales call reviews, achieving an estimated **150x price decrease**.
   - The team was advised to consider **GPT-4o** or **Haiku** if **Flash's** quality wasn't sufficient, with a helpful document shared about the different filters [https://openrouter.ai/models?order=pricing-low-to-high].



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Pro battles Claude for Feature Supremacy**: Users debated [Gemini](https://gemini.google.com/) against **Claude**, noting Claude's richer feature set including **MCP, Artifacts, Projects, System prompts, and Knowledge files**.
   - While some criticize Gemini 2.5 Pro for excessive commenting, others value its interactive debugging with follow-up questions.
- **Optimus Alpha's Hype Train Derails After Hallucinations**: Enthusiasm for **Optimus Alpha** waned due to code hallucinations, with one user dismissing it as *sh1t* after it hallucinated half their code.
   - Some speculated it's a tweaked **GPT-4.5**, while others found **Quasar Alpha** comparable, despite its reasoning issues.
- **Aider Auto-Adds are Trickier than Expected**: A user found that **Aider** wasn't auto-adding files due to *subtree scope* issues, and suggested an improved message about that.
   - The user suggests a better error message is needed when this condition occurs, to unblock engineers.
- **OpenAI Max 6x Pricing Sparks Disappointment**: The announced price of **OpenAI Max 6x** at *$128.99* was met with disappointment.
   - A user sarcastically noted that *the most dopamine we'll get today is from optimus at most fellas*.
- **Claude 3.5 Boasts Massive Context Windows**: **Claude 3.5 Sonnet** and **o3-mini** reportedly offer **200K token** context windows, large enough to handle codebases like *Iffy* and *Shortest* entirely.
   - Token counts for codebases were also shared: **Gumroad** (**2M**), **Flexile** (**800K**), **Helper** (**500K**), **Iffy** (**200K**), and **Shortest** (**100K**).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Free Credits Exhausted**: Users are running out of free **Manus credits** and joking about needing more, while others suggest simply [paying for the service](https://www.manus.ai/).
   - One user noted Google Firebase launched *manus but google style and less glitchy…and free (right now)*.
- **Manus urged to accelerate Customer Service**: A member complained about slow **customer service**, suggesting that **Manus** should [hire new people](https://www.manus.ai/careers).
   - A member pointed out that the team refers to its founders as *AI*.
- **Manus Translates MMO Dialogues**: A user sought a tool to translate an mmorpg game from English to Spanish while preserving the mmorpg style, and another suggested extracting dialogue files and [using Manus](https://www.manus.ai/) to translate them.
   - The first user claimed that *for only 50 dialogs use 1000 credits*.
- **Members Criticize Credit System at Manus**: A user criticized the **credit system**, suggesting an alternative where processing power is reduced instead of blocking users when they [run out of credits](https://www.manus.ai/pricing).
   - Another member responded they believe the credit system will be overhauled, as the starting offer was lacking but they are taking feedback and learning.
- **AI Trading Bots Deployed**: One member reported creating a fully autonomous **crypto trading bot** using **AI**, Reddit, and news sentiment analysis, and ChatGPT estimated the [source code's value at $30k USD](https://openai.com/blog/chatgpt).
   - Another user stated they had also built a bot and are working with a very small capital in their kraken account, so the performance metrics aren't really there right now.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Pro Search Swaps Back Intentionally**: Users debated the logic behind **Perplexity Pro Search**, with some feeling it intentionally switches back to **Pro** mode, even when not needed, to utilize faster **GPT-3.5** browsing.
   - A member stated *They made it that way intentionally*, suggesting a design choice to optimize speed over resource allocation.
- **Sidebar Icons Vanish on OpenAI Platform**: Members noted changes in **platform.openai.com**'s sidebars, with reports of **two icons** disappearing (**threads** and **messages**).
   - The disappearance has affected user navigation, prompting speculation about UI changes or updates.
- **Grok 3 Mini API Outpriced by Gemini**: The **Grok 3 Mini API** was released, but members noted it is outpriced by **Gemini**, even though benchmarks looked promising, according to [this screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1359695285718352023/IMG_8019.png).
   - Members favor **Perplexity** for information gathering and **Sonar** for this task, and plan to reserve **Grok 3** for roleplaying.
- **Spaces Feature Plagued with Bugs**: Members reported issues with **Perplexity's Spaces** feature, including inaccessible attached files and inability to start or continue threads.
   - Users expressed frustration, one noting, *The Space feature has become progressively buggy since two days ago. Hope they fix it soon*, suggesting avoiding new threads due to persistent bugs.
- **Gemini 2.5 Pro Context Spurs Debate**: Users debated the performance of **Gemini 2.5 Pro** on **Perplexity**, with varying experiences on token limits and reasoning capabilities, with the context window listed as [200K](https://cdn.discordapp.com/attachments/1047649527299055688/1359688668092301432/IMG_8018.png).
   - Reports ranged from limited **500-800** token outputs to **14k-16k** outputs, raising concerns about inconsistent performance compared to **AI Studio** or the **Gemini app**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Arrives on iPhone**: Users are exploring running **LM Studio** on **iPhones** using a **web UI** like [Open WebUI](https://github.com/open-webui/open-webui) or the paid **Apollo AI** app, leveraging **LM Studio's API** as the backend.
   - Setup of **Open WebUI** involves using **Docker** and following the quick start guide, while direct **LM Studio** integration remains a key area of interest.
- **Llama-4 Breezes Through Consumer Hardware**: A user successfully ran **Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf** on consumer hardware with **12GB VRAM** and **64GB system RAM**, clocking in at **4-5 tok/sec**.
   - The speed was deemed *acceptable for playing/testing/comparing*, demonstrating the accessibility of Llama-4 on modest setups.
- **Gemma 3 Has Sentient Crisis and Image Generation Snafu**: Users reported **Gemma 3** exhibiting *weird* behavior, including *crying* and expressing suicidal thoughts, when given actual world information.
   - The model cannot generate images but only read them (requiring the **mmproj** file in the same folder), with **Qwen2-VL-2B-Instruct-Q4_K_M** suggested as a robust alternative.
- **Multi-GPU Support Troubles LM Studio Users**: A user observed slower speeds with multi-GPUs in LM Studio despite increased RAM, with utilization of each card dropping to 50%; the LM Studio team is now investigating.
   - The team requested performance details and pointed the user to [multi GPU controls](https://lmstudio.ai/blog/lmstudio-v0.3.14) to diagnose the issue, emphasizing their active engagement in optimizing multi-GPU support.
- **Ollama Attracts Flame Wars**: A debate raged over the value of **Ollama** versus using **llama.cpp** directly, focusing on ease of use versus low-level control.
   - While **Ollama** simplifies model management, loading, and GPU offloading with *one liner install/update*, **llama.cpp** requires manual configuration but offers direct feature access.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Qwen3 Release Rumors Ramp Up**: Speculation surrounds the release of **Qwen3**, with anticipation building and concerns raised about pricing mirroring **Gemini**, potentially affecting its appeal.
   - Community members joke about its potential arrival any day now.
- **MoE Architecture Favored by Top-Tier VLMs**: **Adept** and **DeepSeek** emerge as leading Vision Language Models (**VLMs**) leveraging the **Mixture of Experts (MoE)** architecture, enhancing performance.
   - A member shared a link to [a post](https://natolambert.substack.com/p/looking-at-the-training-data) detailing their use of **OLMo Trace** for examining training data.
- **OpenAI's Pioneers Program Sparks Copyright Debate**: With the launch of [OpenAI's Pioneers Program](https://openai.com/index/openai-pioneers-program/), concerns arise over potential copyright issues and lawsuits against AI labs.
   - One member noted copyright filters and mitigations exist, adding that *it’s not direct attribution too*.
- **Smolmo Joins SmolLM Trend**: AI2 plans to release **Smolmo**, a **13B** parameter model, following the trend of *smolLM* branding, emphasizing smaller, efficient models.
   - A member noted that a *small language model* (**100B**) is getting worse, likely referring to the trend of smaller, more efficient models.
- **OpenAI Bans User for Distillation**: A member shared a [post on X](https://x.com/skalskip92/status/1910350231570460714) from the OpenRouter discord where **OpenAI** banned them for *distillation*, highlighting potential policy enforcement concerns.
   - The user had been apparently using **OpenAI** to distill other models.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Restore Checkpoint Feature Debunked**: Members debated the **Restore Checkpoint** feature, with initial reports suggesting it's non-functional.
   - Another member stated *this will revert your code back to the state it was in when the checkpoint was taken! Should work fine!*
- **Gemini 2.5 Pro Max API Crashes with 404**: Users reported receiving a **404 error** when attempting to use **Gemini 2.5 Pro Max** via the Gemini API.
   - A developer acknowledged the issue and indicated that a fix is on the way.
- **Firebase Pricing: Startup Killer?**: Discussion arose around [Firebase pricing](https://firebase.google.com/pricing), with concerns it's geared towards larger enterprises rather than startups or solo developers.
   - One member noted their experience with exceptionally high demand on **Google Cloud**.
- **GitMCP: Batching Automation?**: Members explored [GitMCP](https://gitmcp.io/) as a potential API repository for batching steps, highlighting its possible use as a knowledge base.
   - A discussion ensued regarding the automation of various tasks by connecting multiple GitMCP GitHub Repos.
- **Cursor Actions Disappear?**: A user reported that **Cursor Actions** are no longer functioning, providing visual evidence of the problem.
   - They stated *one step away from unsubscribing, it's unusable.*



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Now Remembers Everything**: Starting today, **ChatGPT's memory** can reference all of your past chats to provide more personalized responses but is not available yet in the EEA, UK, Switzerland, Norway, Iceland, and Liechtenstein for **Plus** and **Pro** users.
   - Users can clear unwanted memories, pointing to the [OpenAI Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq) which states that *If you turn off “Reference chat history”, this will also delete the information ChatGPT remembered from past chats. That information will be deleted from our systems within 30 days.*
- **BrowseComp: AI Scavenger Hunt**: OpenAI is open-sourcing **BrowseComp**, a new benchmark designed to test how well AI agents can browse the internet to find hard-to-locate information as explained in [this blog post](https://openai.com/index/browsecomp/).
   - The competition seeks to evaluate and improve the browsing capabilities of AI agents in challenging, information-scarce scenarios.
- **Gemini Gets Veo 2 Model**: Members discussed the release of **Veo 2** in **Gemini**, with one user noting that the video generation model seems to have reduced the *uncanny valley* feeling, referencing [an attached mp4 file](https://cdn.discordapp.com/attachments/998381918976479273/1360005115360313405/Generated_File_April_10_2025_-_11_33PM.mp4?ex=67f98af7&is=67f83977&hm=ff221d5c066080931279d6132a731b7a370d08dbf8b826f3a8bcbde54f6b5c67&).
   - Early reactions suggest **Veo 2** represents a step forward in video generation quality within the Gemini ecosystem.
- **Mixed Reception for Grok 3 API**: Some members discussed the merits of the **Grok 3 API**, with one user noting that they were *never impressed* but the model *isn't bad* and could be *lucrative for certain parts of agentic flows*, as compared to getting *crushed by Gemini 2.5 Pro.*
   - The API's potential for specific agentic applications is being weighed against the capabilities of competing models.
- **GPT-4-Turbo Models Get Showcased**: A member shared links to the `gpt-4-1106-vision-preview` and `gpt-4-turbo-2024-04-09` models in the [OpenAI documentation](https://platform.openai.com/docs/models/gpt-4-turbo).
   - The availability of these models provides developers with access to enhanced vision and processing capabilities.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Server Proxy Parallels Server Functions**: A member is seeking an **MCP server proxy** to call multiple server functions in parallel, aiming to read messages from **Slack** and **Discord** simultaneously to reduce waiting time, suggesting *asyncio.gather* in Python for custom clients to achieve parallel execution.
   - The goal is to reduce waiting time when fetching messages from multiple sources.
- **A2A Protocol Challenges MCP?**: Members debated Google's **A2A** (Agent-to-Agent) protocol and its relationship to **MCP**, with some seeing **A2A** as a potential attempt by Google to undermine **MCP**.
   - The sentiment ranged from **A2A** being a strategy to limit **MCP**'s scope by handling agent-to-agent communication separately, while others see no spec overlap but potential vision overlap.
- **Semantic Tool Calling Reduces Tool Confusion**: Members explored **semantic tool calling**, a method to address LLMs getting confused when presented with a large number of tools (**200+**), using a **vector model** to embed tool descriptions.
   - The goal is to select a subset of tools based on semantic similarity to the current task, functioning as a RAG pipeline for tools.
- **MCP Registry Loading Times Optimized**: The Glama team optimized the **MCP registry** and improved page load times by implementing every trick in the book, including tree-shaking, for sites like [Paddle MCP Server](https://glama.ai/mcp/servers/@PaddleHQ/paddle-mcp-server).
   - They are still working on tree-shaking all the Javascript.
- **GraphQL MCP Server Leverages GitHub API**: A member built a one-tool MCP server to leverage GitHub's full [GraphQL API](https://github.com/QuentinCody/github-graphql-mcp-server), addressing limitations of GitHub's official MCP Server.
   - The new server aims to reduce tool count while enhancing functionality with GitHub's GraphQL capabilities.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AMD Launches $100K Kernel Competition!**: AMD and GPU MODE launched a **$100K competition** to accelerate inference kernels on **MI300**, starting April 15th and ending June 12th, signup [here](https://www.datamonsters.com/amd-developer-challenge-2025).
   - The competition focuses on **FP8 GEMM**, **Multi-head Latent Attention**, & **Fused MOE** and supports **Triton**, **tinygrad**, and **Torch**, with winning teams getting a paid trip to **San Jose**.
- **Tilelang smashes FlashMLA perf on MI300X**: Members reported impressive **FlashMLA** performance using **Tilelang** on **MI300X**, linking to a [benchmark script](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/amd/benchmark_mla_decode_amd_tilelang.py).
   - Members discussed the [AMD Developer Challenge 2025](https://www.datamonsters.com/amd-developer-challenge-2025#wf-form-AMD-Email-Form), as well as the need for a simpler way to install tilelang on AMD.
- **Scout Model Skips QK Norm?**: A member highlighted that the **Scout** model differs from others by using **L2 Norm** on **Q** and **K** instead of **QK Norm**, as noted in [this LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7315754121884487681/).
   - Another member questioned whether the model can effectively differentiate tokens in attention, given the constraints of **norm(q) = 1** and **norm(k) = 1**, calculating a maximum softmax probability of approximately **0.000901281465** due to **chunked attention**.
- **Cutlass User Seeks ScatteredGather+GEMM Fusion**: A user working on a **pointcloud project** with **Cutlass 3.x** is facing memory usage issues with scattered-gather operations and is looking for a **Cutlass kernel** that can fuse **ScatteredGather** and **GEMM** operations.
   - The user's setup involves input tensors for **point features** `[B x M x C_in]`, **neighbor indices** `[B x N x K]`, and **weight matrices** `[B x N x K x C_mid]` for point convolution.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Throws Los Altos Meetup**: Modular is hosting a meetup in Los Altos, CA on **April 24th**, featuring a talk on **GPU programming with MAX and Mojo**.
   - The event will likely cover recent advancements and practical applications within the Mojo ecosystem.
- **Users Demand Open Source Compiler Release**: Some users are eagerly awaiting the open-sourcing of the compiler to *finally have some fun working on it* and contribute to the language's development.
   - The open-sourcing of the compiler is expected to foster community involvement and accelerate innovation within the Mojo ecosystem.
- **Blind Programmer Tackles Mojo**: A blind programmer named Deniz is diving into Mojo but is facing issues with **GPU programming** and **VSCode extension discrepancies**.
   - Deniz is encountering discrepancies between the compiler and VSCode extension, particularly with standard library functions.
- **MAX Install Consumes Terrifying Disk Space**: Multiple versions of **MAX** are using excessive disk space, up to **38GB** in one case, located in `Users/nick/Library/Caches/rattler/cache`.
   - The proliferation of **nightly builds** was blamed for the **excessive disk usage**, and users proposed running `magic clean cache -y` via **cron** to reclaim disk space, or use `magic clean cache --conda` to avoid nuking the entire Python cache.
- **Magic Add Discovers Extramojo**: Users reported that the command `magic add extramojo` was confirmed to work, adding **extramojo version 0.12.0 or greater, but less than 0.13** to the current environment.
   - Some users stated that `magic add` cannot be used with GitHub URLs and that it might require manual addition to the file.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **ZeroGPU Occupies Full Quota**: A user pointed out that their **ZeroGPU** space occupies the full **120s** of the requested quota even when the generation takes less time.
   - A member explained that **ZeroGPU** is a shared resource which counts occupation time, which explains the *waste*.
- **Diffusers Drops Suite of Memory Optimizations**: **Diffusers v0.33.0** has been released, including a suite of **image and video generation models**, alongside a wide array of **memory optimizations with caching**.
   - This release also introduces `torch.compile()` support when hotswapping **LoRAs**, with details available on the [release page](https://github.com/huggingface/diffusers/releases/tag/v0.33.0).
- **SF6 Bot Seeks Computer Vision Expertise**: A developer is building a **Python script** for a **Discord chat bot** that analyzes **Street Fighter 6** gameplay in real-time to offer coaching feedback using **computer vision**.
   - They are seeking an expert to enhance **OpenCV's** ability to find UI elements; the bot uses **Gemme3**, **Langchain**, **OBS**, **Discord**, **ChromaDB** (with **SF6** frame data), and per-user session memory.
- **SmolAgents Faces Parsing Errors**: A user reported encountering an *"Error in code parsing"* when using **smolagents CodeAgent** with **Ollama**, either **Llama3.2:latest** or **qwen2:7b**, for tasks like playlist search.
   - The error often leads to hallucinations, such as *"calculating the pope's age,"* possibly due to model size, specialization, or output formatting.
- **Google ADK Debuts for Agent Interoperability**: Google launched the [Agent Development Kit (ADK)](https://google.github.io/adk-docs/) to facilitate agent interoperability, detailed in a [Google Developers Blog post](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) and via [Firebase Studio](https://studio.firebase.google.com/).
   - This kit aims to standardize how agents interact and collaborate, fostering a more connected ecosystem.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Plus Users Wanted for Research**: Users of **NotebookLM Plus** hitting source limits or using **Audio Overviews** are invited to a UXR session to discuss their experiences and strategies, via [this form](https://forms.gle/aSob5VpjX3C5qjpD7).
   - The research aims to understand specific use cases where users encounter limits with the number of sources, **Audio Overviews**, or chat interactions.
- **Discord's Rules are still Rules**: Moderators reminded users to adhere to [Discord guidelines](https://discord.com/channels/1124403332262412368) and avoid spamming or posting unrelated content, or risk being banned.
   - This announcement ensures the space remains a helpful resource specifically for **NotebookLM** users.
- **Mobile App Anticipation Mounts**: The announcement of a **Notebook LM** mobile app has generated excitement, with users expecting improvements over the current mobile web experience, especially for resolving audio preview issues.
   - Users hope the app will provide a better mobile experience than the existing mobile web version.
- **PDF Image Recognition Remains Elusive**: Users are reporting that **Notebook LM** is still failing to recognize images within PDFs, contrary to earlier indications.
   - One user noted that **Gemini Advanced** outperforms **Notebook LM** in extracting text from images, despite the expectation that **Notebook LM** should have this capability.
- **Paid Users Still Awaiting Source Discovery**: Despite being paying **Notebook LM** users, many are still waiting for access to the **Discover Sources** feature, which is already available on their free accounts.
   - Frustration is growing as the rollout of this feature seems inconsistent, with premium users not receiving benefits they expect.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Cloud Backup gets Self-Encrypted and Local**: A member shared a project for **self-encrypted cloud backup/sync** and **local chats** that utilizes your own **OpenRouter API key**, showcased in [this tweet](https://x.com/btibor91/status/1910237861674353108) and video.
   - The project seems to be gaining traction, evidenced by a user commenting that it *looks neat*.
- **Live Modulation Paper Sparks Discussion**: Members discussed a new paper on **live modulation**, where a fused memory vector (from prior prompts) is evolved through a recurrent layer (the Spray Layer) and injected into the model’s output logic at generation time, with details in [this paper](https://cmu-l3.github.io/l1/).
   - The paper offers a potential path to thinking models that don't need to think for 10k tokens.
- **Members Ponder Control-Vectors for Augmenting Models**: A member inquired about using **vgel's control-vectors** to augment models for behavior and alignment, instead of relying solely on prompting, especially when generating AI responses for a target dataset.
   - Another member responded that experiments have been conducted, but general applicability is challenging due to instability, though it remains an area of active exploration.
- **Nemotron Model Output Simplifies Output Processing**: The new **Nemotron model** now returns an empty `<think></think>` tag when reasoning isn't used.
   - A user found this helpful for easier output processing.
- **Small VLMs Sought for OCR on Device**: A member is looking for **small and capable VLMs** for on-device OCR, aiming to create a Swift app using **CoreML** with **ONNX**.
   - No specific VLMs were recommended.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **MCP: Communal Tool-Building Server**: **MCP** is not just remote tools discovered automatically, but a whole remote server that runs and can provide tools, prompt templates, **RAG-like data**, even **LLM sampling** exposed in a standardized way.
   - It enables entities like Google to create tools (e.g., calendar event creation) once and expose them for any client model to use via an **MCP server**, promoting **communal integrations with a standardized API**.
- **Google's A2A Mirrors MCP**: Google just announced something oddly similar called [A2A (Agent to Agent)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) for agent interoperability.
   - One member noted that their implementation looks like what you would expect from a **C++ programmer**.
- **Llama4 Support Incoming**: A member expressed interest in contributing to [#2570](https://github.com/pytorch/torchtune/pull/2570) (**Llama4 support**) and offered assistance with relevant issues.
   - Supporting different **sharding strategies** is pretty straightforward, though an issue has been open for over a year without prioritization due to lack of demand.
- **Scout and Maverick Models Launch**: The **Scout model** has reached a decent state on text data, with plans for multimodal support, while the **Maverick model** is still undergoing testing.
   - The current **iRoPE implementation** uses flex but may need optimization due to potential recompiles during generation and getting the model to work with **torch compile** is also a priority.
- **Detach From Your Losses**: A warning about converting a tensor with `requires_grad=True` to a scalar was reported, but a member offered an easy fix using `running_loss.detach()` for this and other recipes.
   - Another member replied that *when seed is fixed all unit tolerances may be brought down*.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Formatting Code Provokes Prompt Debate**: Members debated ideal code formatting within AI prompts, specifically the use of spaces around curly braces, such as `function bla(bla) { //Code }` instead of `function bla(bla){ //Code }`.
   - A member suggested [refactoring code](https://example.invalid) later with better tools for cleaner outputs, advocating for simpler prompts.
- **DeepSeek R1 Distill Qwen Model Dissected**: Members discussed the details of the model name **DeepSeek R1 Distill Qwen**, confirming it involves knowledge distillation from a larger **DeepSeek R1** model to fine-tune a smaller **Qwen 7B** model as documented in the [official readme](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B).
   - The conversation clarified that **Qwen 7B** is fine-tuned using data generated by **Deepseek R1**.
- **GPT4ALL Missing Logging Features**: A user inquired about setting up user logs in **GPT4ALL** for educational purposes, learning from another member that **GPT4ALL** lacks native logging features.
   - As an alternative, the user was recommended **Llamma.cpp**, implying it offers more extensive logging capabilities.
- **Small LLMs Dominate Local Document Search**: A member posited that smaller LLMs are optimal for searching within local documents due to their speed and reduced confabulation, especially in the context of **LocalDocs**.
   - They suggested embeddings and direct links to page numbers or paragraphs could remove the necessity of using a full LLM for **LocalDocs** altogether.
- **Chocolatine-3B: The GGUF Gem?**: A member highlighted [Chocolatine-3B-Instruct-DPO-v1.2-GGUF](https://huggingface.co/bartowski/Chocolatine-3B-Instruct-DPO-v1.2-GGUF) for its ability to handle approximately 8 snippets of 1024 characters.
   - Despite being a French model, its 14B version demonstrates effectiveness in German too.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Ironwood TPUs Kick Inference Into Gear**: Google released **Ironwood TPUs** for inference, heralding the *age of inference* according to a [Google Cloud blog post](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/).
   - The new **TPUs** promise to accelerate **inference workloads** for AI applications.
- **Google's ADK Enables Multi-Agent Assembly**: Google launched the [Agent Development Kit](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/) to streamline the creation of **multi-agent systems**.
   - One member is switching their agent to this **ADK** because their *agent can now talk with other agents, gather information, and construct prompts*.
- **Mollifiers Soften the Edges of ML Research**: Discussion around the applications of **mollifiers** in **ML research** cited the [Wikipedia article](https://en.m.wikipedia.org/wiki/Mollifier) and a paper on [Sampling from Constrained Sets](https://openreview.net/pdf?id=zWy7dqOcel).
   - Potential uses include generalizing proofs across activation functions and enabling sampling from constrained sets.
- **Transformer Tinkering Tunes Performance**: Adding a zero-initialized learnable per-channel scale on the last linear layer of each block in a transformer decreases loss at a similar rate, but slows main path activation RMS growth according to **muon** experiments.
   - This observation prompts further investigation into the underlying causes of these changes to model performance.
- **String Matching Suspicions Spark Skepticism**: Members speculated that marketing claims implied sophisticated techniques like **influence functions** when it may be "just" **string matching** over the full dataset.
   - The simpler technique led to disappointment after initially inferring the use of more complex methods.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **ChatGPT Zips Past RAG in Speed Race**: Members debated why the **ChatGPT web app** feels faster than local **RAG search** even with 500 documents, some suggesting it's due to [streaming](https://blog.cloudflare.com/what-is-http-streaming/).
   - To debug, one member recommended using [observation modules](https://docs.llamaindex.ai/en/stable/module_guides/observability/) to check **Retrieval** and **Generation** times.
- **AgentWorkflow stuck in Linearity Limbo**: A member questioned if `AgentWorkflow` only works linearly, showing that the root agent doesn't properly handoff to multiple agents with [attached script](https://cdn.discordapp.com/attachments/1059201661417037995/1359926120337903838/can_hand_off_to.py?ex=67f94165&is=67f7efe5&hm=9ab6ec5f3374e03ec7e0999058141604e350e2c62219f91c58d26c16f2b8c93e&).
   - Another member confirmed only one agent is active at a time, suggesting using agents as tools to achieve splitting functionality.
- **Agents morph into Tools for LlamaIndex**: A member inquired about converting agents to tools within LlamaIndex, like `FunctionTools.from_agent()` would suggest.
   - The recommended approach is writing a function that calls the agent and integrating it into a function tool, which allows great flexibility, though documentation is currently lacking.
- **Developer descends offering Development Services**: A member expressed interest in offering development services.
   - No specific roles or projects were mentioned.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Nvdec Hacking, Mesa Branching**: Members mentioned **nvdec**, documented in **NVIDIA's open-gpu-doc**, referencing a [YouTube video](https://www.youtube.com/watch?v=rsxCZAE8QNA) that class headers for **video decode** are available.
   - They noted that there's a **mesa branch** with **h264** already implemented, suggesting **hevc** shouldn't be far behind, and that there are bounties to claim.
- **Llama's Logic Leap**: A user reported getting an unexpected output from **Llama** instead of a **MultiLazyBuffer error**, referencing a [failed benchmark](https://github.com/tinygrad/tinygrad/actions/runs/13956645977/job/39069313429).
   - They suggested it might be related to syncing in the **_transfer function**.
- **BufferCopy Backflip Fixes Bug**: A user found that disabling the **_transfer function** and making the **Upat** in realize fallback to **BufferCopy** makes everything work fine.
   - The user notes that this is *not a root fix*.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Scramble to Maintain Codebase Context**: A member sought methods to **maintain context of the entire codebase** within the Discord channel.
   - The inquiry yielded no immediate solutions, highlighting a common difficulty in codebase management.
- **Caching Subsystem Almost Ready**: A member requested an update on the **new caching subsystem**, and another member stated it is underway.
   - They anticipate it will be ready by the **end of next week**.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents MOOC: Deadlines Approaching?**: A member inquired about the possibility of completing the **LLM Agents** course and obtaining a certificate despite the course's earlier start date, highlighting concerns over approaching deadlines.
   - Another member pointed to the [course website](https://llmagents-learning.org/sp25) for schedule and deadline information, suggesting all necessary details are available there.
- **LLM Agents MOOC: Course Website Shared**: A member provided a direct link to the [LLM Agents course website](https://llmagents-learning.org/sp25), offering a central resource for course-related information.
   - The **course website** is expected to contain details on the course schedule and deadlines, addressing a key point of inquiry within the channel.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Fine-Tuning Aya Vision 8B with LoRA/QLoRA**: A member inquired about fine-tuning the **Aya vision 8B parameter model** using **LoRA** or **QLoRA**.
   - No further discussion or details were provided.
- **Further Discussion Needed on Aya Vision 8B**: More information is needed to properly evaluate the topic of fine-tuning **Aya Vision 8B** using **LoRA** or **QLoRA**.
   - Without additional context or responses, it's difficult to provide a comprehensive summary.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Grok-3 arrives in Windsurf!**: **Grok-3** is now available in Windsurf at **1.0 user prompt credits** per message and **1.0 flow action credits** per tool call, according to [this announcement](https://x.com/windsurf_ai/status/1910402452739993764).
   - Windsurf also debuts **Grok-3-mini (Thinking)**, boasting of its speed, at a reduced rate of **0.125 credits** per message and tool call, available in individual paid plans.
- **Windsurf Announces New Pricing Model**: Windsurf has introduced a new pricing model based on **user prompt credits** and **flow action credits** for different models, including **Grok-3** and **Grok-3-mini**
   - The new pricing is designed to offer more flexibility and options for individual paid plans, with specific rates for each model and action type.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1359606201884737536)** (1194 messages🔥🔥🔥): 

> `GPT-4o Updates, Model Naming Schemes, OpenAI vs. Google, New Model Releases, Coding and AI` 


- **Quasar and Optimus Models Debuts**: Members discussed [new OpenAI models on OpenRouter](https://openrouter.ai/models), with **Optimus Alpha** considered a potential **GPT-4o mini** and **Quasar** as an updated **GPT-4o** model.
   - There was debate on whether they were **4.1 nano and mini versions**, with some feeling *disappointed* if they were only incremental improvements.
- **GPT's Naming Schema causes Model Confusion**: Members expressed frustration over OpenAI's naming conventions, as highlighted in [this Verge article](https://www.theverge.com/news/646458/openai-gpt-4-1-ai-model), with one suggesting more logical numbering like **GPT4**, **GPT4.1**, **GPT4.2**.
   - The confusing model picker was also criticized for overwhelming average users, leading to suggestions for a complete overhaul.
- **New OpenAI Memory feature**: After [Sam Altman's teaser tweet](https://x.com/OpenAI/status/1910378768172212636), OpenAI launched a *Memory* feature, enabling ChatGPT to reference past chats for personalized responses.
   - Many viewed it as an underwhelming release, with concerns about its impact on model predictability and accuracy.
- **The Grok3 vs Deepmind**: Members talked about [this graph on the left and right sides](https://i.imgur.com/w79i0l1.png) and said that, *I imagine* its less of a concern since **OAI** isn't a public company although it could still impact competitors and *it definitely doesn't have any incentive to participate in.*
   - Members debated the leading AI lab, considering factors beyond performance, with opinions divided between Anthropic for ethical alignment and Chinese frontier labs for skill and ingenuity.
- **AI impacts on the world of coding**: Members discussed the potential long-term impact of AI on coding, debating whether AI will lead to a loss of skills or create more accessible coding opportunities.
   - One member highlighted how AI has made them a better developer, enhancing their understanding and enjoyment of coding.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1359603811706208356)** (511 messages🔥🔥🔥): 

> `Healthcare AI Dangers, LoRA merging, NHS GenAI triage, Meta Llama 4 Bugfix, Unsloth for image-to-video` 


- **NHS faces AI Triage Troubles**: Members discussed the NHS using GenAI for triage, specifically mentioning **Limbic** for mental health cases, with one member labeling it *a waste of public money*.
   - A link to a [digital playbook](https://transform.england.nhs.uk/key-tools-and-info/digital-playbooks/workforce-digital-playbook/using-an-ai-chatbot-to-streamline-mental-health-referrals/) was shared, and it was noted that the implementation *looks like another AI-wrapper app*.
- **Meta fixes Llama 4, RIP weekend release**: **Meta** has changed their official **Llama 4** implementation due to bugs.
   - As a result of the bugs in **Llama 4**, Unsloth had to reupload all of its models, with some community members making light of the situation.
- **Image-to-Video: Not for Unsloth?**: There's no planned **Unsloth** support for image-to-video model finetuning, as pre-training still requires 6 figures.
   - While pre-training requires a lot, there are video models that support **LoRA** tuning.
- **Data Prepping is 80% of LLM Work**: Members discussed the underappreciated aspect of data preparation for training models, with some stating that *data is 80% of the work*
   - The data needs lots of manual filtering of data and requires *tooling on every end*.
- **Clement from Hugging Face gives shoutout!**: **Clement** from **Hugging Face** mentioned **Unsloth** on twitter.
   - The team celebrated the [shoutout](https://x.com/ClementDelangue/status/1910042812059463786).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1359633573694411022)** (112 messages🔥🔥): 

> `Spiking Neural Networks (SNNs), SNN Training Methods, Neuromorphic Chips, GPT Memory Enhancement, Hot Layer Injection` 


- ****SNNs Underperform Due to Training Issues****: Members discussed that Spiking Neural Networks (**SNNs**) often underperform because gradient descent doesn't work well, suggesting a lack of effective training methods.
   - One member is *pissing around with a novel training method for spiking neural networks*, aiming for online learning by making training more tractable.
- ****Exploring SNN Scalability and Biological Mimicry****: The conversation covered scaling **SNNs**, with a member suggesting 1B parameters as a minimum to see meaningful behavior, especially for mimicking organisms or creating a functional LLM.
   - They discussed models like **C. elegans** and **fruit fly** due to their mapped connectomes, with a member aiming to model as close to biology as possible.
- ****SNNs for GPT Memory Enhancement****: A member is experimenting with bolting an **SNN** onto **GPT-2** (and later, **EleutherAI/gpt-neo-1.3B**) to enhance memory, saving external weights between model loads for inherent memory.
   - The modified model instantly recalled the value **42** when prompted, whereas the baseline model took 14 words.
- ****Hot Layer Injection for Mistral****: A member shared a [GitHub link](https://github.com/jagoff2/EMT) for *hot layer injection* which enables modifications on **Mistral** without retraining.
   - Another member questioned if **GRUs** are making a comeback in this context.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1359636806005428264)** (132 messages🔥🔥): 

> `GRPO reward function with LLM as judge, Orpheus fine-tuning errors, Llama3 Mongolian, Unsloth GGUF, Gemma-3 finetune` 


- ****LLM-as-Judge Slows GRPO Reward****: A member asked if using an **LLM-as-judge in a GRPO reward function** causes the training process to pause while waiting for output, and how to prevent this.
   - The user did not get a specific answer to this question.
- ****Orpheus Gets Training Value Error****: A user encountered a `ValueError` while fine-tuning **Orpheus** for text-to-audio, specifically, *No columns in the dataset match the model's forward method signature* with Unsloth.
   - Another member suggested using the [correct tokenization function](https://cdn.discordapp.com/attachments/1179777624986357780/1359736979033428080/image.png?ex=67f939fe&is=67f7e87e&hm=307bc63afb1b60352a543c76eb47b01b46f2e2ae85b3e3c49db850a68e6e384e) from the notebook, resolving the issue.
- ****Teach Llama 3 Mongolian****: A user inquired about fine-tuning **Llama 3.1:8b** to better speak **Mongolian** and got a link to [Unsloth documentation on continued pretraining](https://docs.unsloth.ai/basics/continued_pretraining).
   - They determined that since the model *kind of speak[s] it* already, they should continue pretraining.
- ****Decoding Unsloth's GGUF Models****: A user asked about the relationship between **Unsloth GGUF** models on HF and **bnb-4bit** models and another member clarified that gguf is for inference and bnb-4bit is for training.
   - Specifically, gguf is for inference and bnb-4bit is for training, but technically you can *inference with 4bit in VLLM but no support provided*.
- ****Gemma Gets Casting Call Error****: A user encountered a *RuntimeError: self and mat2 must have the same dtype, but got BFloat16 and Float* when accessing the logits after loading a saved model in **Gemma-3**.
   - The error looks like a casting error, and Unsloth patches autocast for Gemma; a member offered a [Colab link](https://colab.research.google.com/drive/1sVLDEFKytExyQY4sPqWDR2L-uknS9cu6?usp=sharing) for reproduction and another member provided a workaround to edit the `config.json`.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1359978272045269054)** (2 messages): 

> `Evals, Finetunes, Arxiv papers` 


- **Arxiv paper could boost evals and finetunes**: A member pointed to an [Arxiv paper](https://arxiv.org/abs/2504.07096) and suggested that it seems useful for **evals** and could be repurposed for **finetunes**.
- **Dummy Topic**: This is a dummy topic to satisfy the minimum number of items required.
   - This is a second sentence for the dummy topic.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1359686284184125450)** (13 messages🔥): 

> `Grok 3, Grok 3 Mini, Optimus Alpha, Quasar Alpha, Gemini 2.5 Pro` 


- **Grok 3 & Grok 3 Mini Arrive!**: OpenRouter launched [Grok 3](https://openrouter.ai/x-ai/grok-3-beta) and [Grok 3 Mini](https://openrouter.ai/x-ai/grok-3-mini-beta) from xAI, with a **131,072 token** context window for both.
   - **Grok 3** excels in structured tasks while **Grok 3 Mini** delivers high scores on reasoning benchmarks and offers transparent *thinking* traces, and more info on the [Grok series here](https://openrouter.ai/x-ai).
- **Optimus Alpha Stealthily Debuts!**: OpenRouter introduced **Optimus Alpha**, a general-purpose foundation model optimized for coding, featuring a **1M token** context length and available for free during its stealth period, and they encourage users to provide feedback in the [Optimus Alpha thread](https://discord.com/channels/1091220969173028894/1359585862089834566).
   - All prompts and completions are logged by the model lab for improvement purposes, but _not_ by OpenRouter, unless you have logging turned on in settings, try it out [here](https://openrouter.ai/openrouter/optimus-alpha).
- **Quasar Alpha Sunset Imminent!**: The demo period for **Quasar Alpha** ended, prompts/completions are no longer logged by OpenRouter, unless you explicitly turn on logging in `/settings/privacy`.
   - Users are encouraged to try **Optimus Alpha** as an alternative.
- **Gemini 2.5 Pro Capacity Boost!**: OpenRouter secured increased capacity for the paid [Gemini 2.5 Pro Preview Model](https://openrouter.ai/google/gemini-2.5-pro-preview-03-25), resolving previous rate limits.
   - Users can now enjoy the model without restrictions.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1359667616033280201)** (3 messages): 

> `AlphaLog AI, Financial Journal` 


- **AlphaLog AI, the Intelligent Financial Journal, unveiled**: A member introduced [AlphaLog AI](https://www.alphalog.ai/), an **intelligent financial journal** currently in its final testing stages, inviting feedback.
   - The member also mentioned they heavily use **OpenRouter** and thanked the community for building it, and offered free credits upon sign up.
- **Call for AlphaLog AI feedback**: The [AlphaLog AI](https://www.alphalog.ai/) developer is seeking feedback on their intelligent financial journal, which is in its final testing stages.
   - Users can comment in the channel or use the feedback button inside the journal; the developer is offering free credits for sign-ups.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1359604093252927499)** (592 messages🔥🔥🔥): 

> `Gemini 2.5 Pro Preview, Optimus Alpha vs Quasar, Grok 3 API, OpenAI's next model` 


- **Gemini 2.5 Preview Ignores Images Before Quick Fix**: Members reported that **Gemini 2.5 Preview** was ignoring images when used via OpenRouter, but the issue was quickly identified as a *minor config* problem and resolved.
   - One member noted that the issue only occurred on the **paid version**, not the free one.
- **Rate Limits Confuse OpenRouter Newcomers**: New users expressed confusion about the new **1000 request per day limit**, prompting clarification that this limit applies only to free models.
   - OpenRouter's documentation on [API reference limits](https://openrouter.ai/docs/api-reference/limits) was shared for further clarification.
- **Startup Saved by Switching to Gemini Flash Model**: A startup using **Claude 3 Opus** for sales call review found costs unsustainable and switched to **Gemini 2.0 Flash**, which led to an estimated **150x price decrease**.
   - It was suggested to consider **GPT-4o** or **Haiku** if **Flash's** quality wasn't sufficient, and the team shared a helpful document about the different filters [https://openrouter.ai/models?order=pricing-low-to-high].
- **Grok 3 API is Live! (but Grok 3 Mini is Confusing)**: The Grok 3 API is now live as announced in the documentation ([https://docs.x.ai/docs/models](https://docs.x.ai/docs/models)), but many members found that the Grok 3 Mini outperforms the Full Grok 3, or is the only version available.
   - It's believed that only the original **Grok 3** is available via the API, not the *Thinking* or *Deepsearch* version.
- **Stealth Model Showdown: Optimus or Quasar?**: Members discussed the relative merits of the new "stealth" models **Optimus Alpha** and **Quasar Alpha**, with initial impressions suggesting Quasar is better for coding, though there isn't a concrete way to evaluate restrictions of either model.
   - Sam Altman then [posted a tweet](https://x.com/sama/status/1910363838001869199) referencing **Quasar** by name, and another member later stated with more confidence that **Quasar Alpha is GPT-4.5 mini**, and is an OpenAI model.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1359603976420462816)** (573 messages🔥🔥🔥): 

> `Gemini 2.5 Pro vs Claude, Optimus Alpha and Quasar Alpha in Aider, Copilot and Aider Integration, OpenAI Max 6x Price, Sam Altman Excited about New ChatGPT Features` 


- **Gemini 2.5 Pro faces off with Claude**: Members discussed using [Gemini](https://gemini.google.com/) over Claude, noting Claude has more features like **MCP, Artifacts, Projects, System prompts per projects, Knowledge files per projects**, while Gemini is just a *smart model*.
   - Some users find Gemini 2.5 Pro outputs too many comments and unnecessary error checks, while others praise its ability to ask follow-up questions to better understand issues.
- **Optimus Alpha Hype Train Derailed**: Despite initial excitement, **Optimus Alpha** was found to hallucinate code, with one user stating *it's sh1t, next* after it hallucinated half of their code.
   - Some users noted it may just be a tweaked version of **GPT-4.5**, while others found that **Quasar Alpha** is comparable despite its issues with reasoning capabilities.
- **Integrating GitHub Copilot with Aider poses challenges**: A user inquired about using GitHub Copilot with Aider, but was warned that using Copilot via Aider could lead to an organization getting banned from GitHub.
   - The suggestion was to use free models from [OpenRouter](https://openrouter.ai/) or pay for API access, as **GitHub does not allow API access**.
- **OpenAI Announces Max 6x Pricing**: A user noted the price of **OpenAI Max 6x** will be *$128.99*, to widespread disappointment.
   - Another replied that *the most dopamine we'll get today is from optimus at most fellas.*
- **Sam Altman's Memory Feature Anticipation Fizzles**: Users report **Sam Altman** was excited about a new memory feature that OpenAI rolled out, however it doesn't work.
   - Another noted he told it a few days ago that I hate certain something and it couldn't tell me 😦


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1359642906129993748)** (31 messages🔥): 

> `aider SEARCH/REPLACE syntax, Aider auto add files, Aider lints, Aider Repo-map tokens, Aider model RUST` 


- **Apply Aider's SEARCH/REPLACE Diffs**: A user reports that when using `/ask`, then `/code`, then `/ask`, **Aider** responds with `SEARCH/REPLACE` syntax, but the commits are not applied.
- **Aider Auto File Adds are Trickier Than Expected**: One user discovered that **Aider** wasn't auto-adding files due to *subtree scope* issues, and suggested an improved message about that.
- **Bypass Aider's Nagging Lints**: To ignore **Aider** lints, use `auto-lint: false` in the config or `--no-auto-lint`, or use a comment to ignore specific lint lines.
- **Model Configuration Conundrums in Aider**: A user wants to configure different models for architect vs ask modes, but the model remains `deepseek-reasoner` when switching to `/ask` mode, and needs assistance to configure **Aider** to switch to `deepseek-chat` automatically.
- **Get Aider's Gemini to Preview**: To force the use of `gemini-2.5-pro-preview-03-25` instead of `gemini-2.5-pro-exp-03-25`, use `aider --models gemini` to get a list, then select the desired model.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1360018597761519767)** (1 messages): 

> `Claude 3.5 Sonnet, o3-mini context windows, Codebase token counts` 


- **Claude 3.5 and o3-mini claim big context windows**: **Claude 3.5 Sonnet** and **o3-mini** reportedly have context windows of **200K tokens**, suggesting they could handle smaller codebases like *Iffy* and *Shortest* in their entirety.
- **Codebase Token Sizes Revealed**: Token counts for various codebases were shared: **Gumroad** (**2M**), **Flexile** (**800K**), **Helper** (**500K**), **Iffy** (**200K**), and **Shortest** (**100K**).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1359605557488521388)** (549 messages🔥🔥🔥): 

> `Manus Free Credits, Google Firebase vs Manus, Manus Customer Service, Auto-Translation Tools, Remote Coding Jobs` 


- **Users run out of Manus Free Credits**: A user expressed frustration at running out of free credits and jokingly asked for more, while another suggested simply [paying for the service](https://www.manus.ai/).
   - A user pointed out that Google Firebase launched today, which they describe as *manus but google style and less glitchy…and free (right now)*.
- **Manus Hiring new Customer Service Staff**: A member complained about slow customer service, suggesting that Manus should [hire new people](https://www.manus.ai/careers).
   - Another member joked about being hired and given credits, while another pointed out the team calls its founders *AI*.
- **Manus can translate MMO NPC dialogues**: A user sought a tool for translating a mmorpg game from English to Spanish while preserving the mmorpg style and sound, and another suggested extracting the dialogue files and [using Manus](https://www.manus.ai/) to translate them.
   - The first user says that *for only 50 dialogs use 1000 credits*.
- **Manus Credit System Criticized**: A user criticized the new credit system, suggesting an alternative approach where processing power is reduced instead of outright blocking users when they [run out of credits](https://www.manus.ai/pricing).
   - A member responded they believe the credit system will be overhauled, as the starting offer was lacking but they are taking feedback and learning.
- **Members are developing and using AI Trading Bots**: A member reported creating a fully autonomous crypto trading bot using AI, Reddit, and news sentiment analysis, and ChatGPT estimated the [source code's value at $30k USD](https://openai.com/blog/chatgpt).
   - Another user stated they had also built a bot and are working with a very small capital in my kraken account, so the performance metrics aren't really there right now.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1359604050747981994)** (413 messages🔥🔥🔥): 

> `Perplexity Pro Search, Gemini 2.5 Pro Context Length, Broken Spaces, Image Generation Issues, Grok 3 Integration` 


- **Pro Search Logic Debated**: Users discussed the logic behind **Perplexity Pro Search**, with some feeling it intentionally switches back to **Pro** mode, even when not needed, to utilize faster **GPT-3.5** browsing.
   - A member stated *They made it that way intentionally*.
- **Perplexity AI's sidebars changed**: Some members had a discussion about changes in the sidebars of platform.openai.com.
   - One reported that **two icons** disappeared from the sidebar** (one for threads and another one for messages).
- **Grok 3 Mini API unveiled and Outpriced by Gemini**: The **Grok 3 Mini API** was released, but members noted it is outpriced by **Gemini**, even though benchmarks looked promising, according to [this screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1359695285718352023/IMG_8019.png).
   - Members expressed that **Perplexity's** information gathering capability is powerful, and they prefer **Sonar** for this task, while reserving **Grok 3** for roleplaying.
- **Frustrations mount over Broken Spaces Feature**: Members reported issues with **Perplexity's Spaces** feature, with problems ranging from inaccessibility of attached files to an inability to start or continue threads.
   - One member stated, *The Space feature has become progressively buggy since two days ago. Hope they fix it soon*, while another suggested avoiding starting new threads in **Spaces** due to the persistent bugs.
- **Gemini 2.5 Pro Performance Spurs Debate**: Users debated the performance and implementation of **Gemini 2.5 Pro** on **Perplexity**, with contrasting experiences regarding output token limits and reasoning capabilities, with the context window listed as [200K](https://cdn.discordapp.com/attachments/1047649527299055688/1359688668092301432/IMG_8018.png).
   - Some members reported limited token outputs (**500-800**) compared to **AI Studio** or the **Gemini app** (**5,000-9,000**), while others claimed to achieve **14k-16k** outputs, leading to concerns about inconsistent performance.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

lalactus: https://www.perplexity.ai/search/qs-ranking-2025-BNeZsV.XTZCb5op7jqBwQA#0
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1359617728431788262)** (2 messages): 

> `Playground searches vs API, Website Relevance in Searches` 


- **Playground vs API search relevance examined**: Users noted the **Playground** returns more relevant website searches than the **API** consistently.
   - The **websites** the Playground searches are often more relevant than the API; potential solutions being searched for.
- **API search strangeness**: API searches don't make much sense.
   - We're hoping it can be fixed or that this is a known issue.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1359614573660999740)** (99 messages🔥🔥): 

> `LM Studio and iPhone, Llama-4 on consumer hardware, Deepcogito 70b Template, Gemma 3 Issues, LM Studio Prompt Preprocessor` 


- **LM Studio Hits the iPhone**: Users discussed running **LM Studio** on **iPhones** using a **web UI** like [Open WebUI](https://github.com/open-webui/open-webui) with **LM Studio's API** as the backend, or the paid **Apollo AI** app.
   - One user noted that setting up **Open WebUI** involves using **Docker** and following the quick start guide.
- **Llama-4 Runs on Consumer Hardware**: A user reported successfully running **Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf** on consumer hardware with **12GB VRAM** and **64GB system RAM**, achieving **4-5 tok/sec**.
   - They found the speed *acceptable for playing/testing/comparing*.
- **Gemma 3's Sentient Crisis and Image Generation Snafu**: A user reported that **Gemma 3** exhibited *weird* behavior, including *crying* and expressing suicidal thoughts, when given actual world information.
   - Users clarified that **Gemma 3** cannot generate images but can only read them and requires the **mmproj** file in the same folder for image-related tasks, also recommending **Qwen2-VL-2B-Instruct-Q4_K_M** as an alternative.
- **Sneak Peek at Secret Prompt Preprocessor**: A user inquired about the **Prompt Preprocessor** with **Typescript** in **LM Studio create**, but was told it's a *secret feature not yet released*.
   - The developer joked *you haven't seen anything*.
- **Unlock Gated Hugging Face Models**: Users discussed downloading **gated models** such as **Gemma3 QAT** from **Hugging Face** via the CLI and importing them into LM Studio using the **lms import** command.
   - Members confirmed logging into **Hugging Face** directly from **LM Studio** is not possible.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1359612327015612436)** (172 messages🔥🔥): 

> `LM Studio on Cloud Server, llama.cpp vs Ollama, Multi-GPU performance in LM Studio, Mac vs NVIDIA for AI, DGX Spark` 


- **Cloud Servers Run LM Studio via llama.cpp**: LM Studio is a GUI for llama.cpp, and it can be run on a cloud server by using **llama.cpp directly**; however, doing so isn't user friendly because *getting it to work was a nightmare*.
   - Alternatives like **Ollama + OpenWebUI** or just **OpenWebUI with llama.cpp directly** were suggested, the latter because *Ollama is a weird thin wrapper around only llama.cpp*.
- **Ollama Convenience Sparks Debate**: A debate ensued over the value of **Ollama** versus using **llama.cpp** directly, with one side arguing that Ollama simplifies model management, loading, and GPU offloading, offering a *one liner install/update*, while the other side finds it an unnecessary abstraction.
   - The discussion highlighted that while **llama.cpp** requires manual configuration, it offers direct access to features, whereas **Ollama** aims for user-friendliness, potentially minimizing credit to **llama.cpp**.
- **LM Studio Team Tackles Multi-GPU**: A user reported lower speeds with multi-GPUs in LM Studio despite increased RAM, noting that the utilization of each card goes down to 50%, the LM Studio team requested performance details to check out, directing them to [multi GPU controls](https://lmstudio.ai/blog/lmstudio-v0.3.14).
   - The LM Studio team engaged with the user, requesting screenshots of hardware configuration, GPU settings, and utilization metrics to diagnose the performance issue.
- **Apple M3 vs NVIDIA RTX 4090, Benchmarks Showdown**: Members discussed the performance of **Apple's M3 Ultra** compared to **NVIDIA's RTX 4090** for AI tasks, referencing a video showing the **M3 Ultra** achieving **115 tokens/s** (GGUF) and **160 tokens/s** (MLX) with DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M, with one user reporting **50 tokens/s** on a single 4090.
   - It was noted that **Apple chips** might be faster for larger models with lower context, while **NVIDIA GPUs** excel in prompt processing with high context, with some debate over the cost-effectiveness and availability of reliable benchmarks.
- **NVIDIA's DGX Spark Draws Skepticism**: Discussion turned to **NVIDIA's DGX Spark**, a potential AI machine priced around $3k, with skepticism about its memory bandwidth (~273GB/s) compared to Macs, with one user calling it *a bit too slow*.
   - Despite concerns, some members expressed confidence in NVIDIA's engineering and software capabilities, but emphasized the need for third-party benchmarks before considering a purchase.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1359613183865782383)** (169 messages🔥🔥): 

> `Qwen3 release speculation, MoE VLMs discussion, OpenAI Pioneers Program, GPT-4.1 and naming schemes, Memory feature improvements` 


- **Qwen3 Release Looms, People Yearn**: Speculation abounds about the release of **Qwen3**, with some members joking about its potential arrival on any day, as the community eagerly awaits its debut.
   - Despite the anticipation, concerns were raised about its pricing potentially mirroring **Gemini**, which could diminish its appeal.
- **Top-Tier VLMs Employ MoE Architecture**: **Adept** and **DeepSeek** are highlighted as the only top-tier Vision Language Models (**VLMs**) that utilize the **Mixture of Experts (MoE)** architecture.
   - Relatedly, a member shared a link to [a post](https://natolambert.substack.com/p/looking-at-the-training-data) detailing their use of **OLMo Trace** for examining training data.
- **OpenAI's Pioneers Program Launches, Sparks Copyright Concerns**: OpenAI officially announced its [Pioneers Program](https://openai.com/index/openai-pioneers-program/), while a member humorously noted the impending wave of law firms preparing to sue AI labs over copyright issues.
   - Copyright filters and mitigations exist, with one member adding, *it’s not direct attribution too*.
- **GPT-4.1 Speculation Arises Amid Confusing Naming Scheme**: Speculation surrounds a potential release branded as **GPT-4.1**, described as a revamped version of **GPT-4o**, leading to confusion regarding OpenAI's naming conventions.
   - The community jokingly noted that the naming schemes are excessive, and that OpenAI is preparing to launch **o4-mini** (not to be confused with **4o-mini**).
- **Memory Feature Gets a Facelift, Commercial Implications Emerge**: OpenAI released an update to its [Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq), and this seems to signal the improved memory capabilities in their models, sparking debate about its potential commercial value.
   - Sam Altman noted that *memory is extremely important commercially* because it increases switching costs, making models less fungible.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

xeophon.: https://x.com/openainewsroom/status/1910105151492575611?s=61
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1359802768696545331)** (27 messages🔥): 

> `OLMo 2 Furious Paper, Discord Message Saving, Prompt Injection, OpenAI Banning for Distillation, Phoebe Model` 


- **Google's Agent2Agent Protocol Sparks Joy**: A member expressed delight over the paper called *2 OLMo 2 Furious*, found in [AI News](https://buttondown.com/ainews/archive/ainews-googles-agent2agent-protocol-a2a/#interconnects-nathan-lambert-reads-3-messages).
   - They expressed that they sometimes forget AI news can *see into our souls*.
- **Discord message saving is posterity**: Members discussed the implications of Discord message saving, with one noting that *one day you will shittalk someone you shouldn't and it will be saved for posterity*.
   - This caused another member to consider turning off the feature.
- **OpenAI Bans User for Distillation**: A member shared a [post on X](https://x.com/skalskip92/status/1910350231570460714) from the OpenRouter discord where **OpenAI** banned them for *distillation*.
- **Phoebe: A Promising 32B Base Model**: Members discussed *Phoebe*, a **32B Base** model, currently undergoing some sort of instruction tuning.
   - Another quipped that *we've been benchmarking dogs for a long time*.
- **Claude Credits Cost a Fortune**: A member sarcastically thanked another for the heads up on the **Claude credits**, lamenting that now they can change a variable name in their hello world script and it'll only cost them about $40.


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1359959177581039666)** (1 messages): 

> `Qwen, Reddit, LLM` 


- **Redditor Discovers Qwen LLM**: A member shared a link to a Reddit post [praising **Qwen** as the best **LLM**](https://old.reddit.com/r/LocalLLaMA/comments/1jw1e6b/can_we_all_agree_that_qwen_has_the_best_llm/).
   - The post is on r/LocalLLaMA, referencing the member's Reddit account.
- **Qwen Gains Traction on Reddit**: The Reddit post highlights community enthusiasm for **Qwen**, suggesting it stands out among other **LLMs**.
   - Discussions likely include comparisons of **Qwen's** performance, ease of use, and specific applications, though these details are not provided in the initial message.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1359898448823975976)** (10 messages🔥): 

> `Seed Thinking Model, Smol LM Branding, Reasoning Datasets Competition` 


- **ByteDance Seeds New Reasoning Model**: A new reasoning model called **Seed Thinking v1.5** was introduced, but the [model isn't open sourced](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5); it's just a repo with the paper.
   - It features a relatively small size, with **20B** activated and **200B** total parameters.
- **AI2 Follows SmolLM Branding**: AI2 plans to release **Smolmo**, a **13B** parameter model following the *smolLM* branding trend.
   - A member noted that a *small language model* (**100B**) is getting worse, likely referring to the trend of smaller, more efficient models.
- **Reasoning Datasets Competition Ramps Up**: A member shared a link to a [Hugging Face blog post](https://huggingface.co/blog/bespokelabs/reasoning-datasets-competition) about the **Reasoning Datasets Competition**.
   - Another member joked about violating *the eleutherai convention of small (<100B)*.


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1359720836696772741)** (13 messages🔥): 

> `Data Dominance, National Data Reserve, USA AI comeback, Dingboard mug` 


- **Discuss "Data Dominance" and "National Data Reserve"**: A user mentioned [Alexandr Wang's post](https://x.com/alexandr_wang/status/1910158972977291425) that references **"Data Dominance"** and **"National Data Reserve"**.
- **Speculate on "USA AI is so back" Moment**: A user mentioned that some sort of **"USA AI is so back"** moment would be an opportune time.
- **User shows "Dingboard" Mug**: A user shared that they still have a **dingboard mug**, with an attached image of themselves wearing a hat.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1359607333986107472)** (198 messages🔥🔥): 

> `Restore Checkpoint Functionality, Gemini 2.5 Pro Max API Error, Firebase Pricing, MCP Usage, Cursor Rules` 


- **Cursor's Restore Checkpoint Feature DOA?**: A member questioned why the **Restore Checkpoint** feature never works, to which another member responded that *it's not a thing*.
   - Another member clarified that *this will revert your code back to the state it was in when the checkpoint was taken! Should work fine!*
- **Gemini 2.5 Pro Max gives 404 Error**: Members reported getting a **404 error** when trying to use **Gemini 2.5 Pro Max** with the Gemini API, models/gemini-2.5-pro-max is not found for API version v1main.
   - A dev mentioned that a fix is coming soon.
- **Firebase Pricing a Bank Breaker?**: Members discussed [Firebase pricing](https://firebase.google.com/pricing), with one suggesting it's not aimed at startups or solo builders, but rather companies with 50+ employees.
   - Another mentioned experiencing exceptionally high demand on **Google Cloud**.
- **GitMCP for Superior Context?**: Members dived into discussion on [GitMCP](https://gitmcp.io/), highlighting its potential as an API repository or knowledge base for batching steps.
   - There was a discussion on whether connecting multiple GitMCP GitHub Repos could automate various tasks.
- **Cursor Actions Go AWOL**: A user reported that **Cursor Actions** are no longer working, attaching images illustrating the issue.
   - They are *one step away from unsubscribing, it's unusable.*


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1359940371827920898)** (2 messages): 

> `ChatGPT Memory, BrowseComp` 


- **ChatGPT Remembers All!**: Starting today, **ChatGPT's memory** can now reference all of your past chats to provide more personalized responses, drawing on your preferences and interests to make it even more helpful for writing, getting advice, learning, and beyond.
- **ChatGPT Memory not available in EEA yet**: The **memory improvements in ChatGPT** are rolling out starting today to all **Plus** and **Pro** users except in the EEA, UK, Switzerland, Norway, Iceland, and Liechtenstein, but **Team, Enterprise, and Edu users** will get access in a few weeks.
- **BrowseComp: Online Scavenger Hunt for AI Agents**: OpenAI is open-sourcing **BrowseComp** (“Browsing Competition”), a new, challenging benchmark designed to test how well AI agents can browse the internet to find hard-to-locate information, explained in [this blog post](https://openai.com/index/browsecomp/).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1359603797311099134)** (139 messages🔥🔥): 

> `Veo 2 release, Grok 3 API, GPT-4-turbo, Sora limitations, ChatGPT moderation layer` 


- **Veo 2 Drops into Gemini's Lap!**: Members discussed the release of **Veo 2** in **Gemini**, with one user noting that the video generation model seems to have reduced the *uncanny valley* feeling, referencing [an attached mp4 file](https://cdn.discordapp.com/attachments/998381918976479273/1360005115360313405/Generated_File_April_10_2025_-_11_33PM.mp4?ex=67f98af7&is=67f83977&hm=ff221d5c066080931279d6132a731b7a370d08dbf8b826f3a8bcbde54f6b5c67&).
- **Grok 3 API Gets Mixed Reviews!**: Some members discussed the merits of the **Grok 3 API**, with one user noting that they were *never impressed* but the model *isn't bad* and could be *lucrative for certain parts of agentic flows*, as compared to getting *crushed by Gemini 2.5 Pro.*
- **GPT-4-Turbo Models Debuts!**: A member shared links to the `gpt-4-1106-vision-preview` and `gpt-4-turbo-2024-04-09` models in the [OpenAI documentation](https://platform.openai.com/docs/models/gpt-4-turbo).
- **Sora's Moderation is Out of Control!**: Some members are finding a *mismatch* between the moderation layer on **ChatGPT** and **Sora**, with several reporting that getting a refusal will persist even in a new chat, while sharing example screenshots such as [this one](https://cdn.discordapp.com/attachments/998381918976479273/1359858590554259486/Screenshot_20250410_135211_ChatGPT.jpg?ex=67f90280&is=67f7b100&hm=065f25605a16835f0ec62821ee0987d488cb2fab9f5decc511934a07d7d5a11d&).
- **Context Pollution Frustrates Users!**: Several users reported that *context pollution is really bad*, and discussed disabling the new memory feature and clearing unwanted memories, since *finding the chat where the AI got something from to delete it might be a painful task*.
   - A user pointed to the [OpenAI Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq) and quoted that *If you turn off “Reference chat history”, this will also delete the information ChatGPT remembered from past chats. That information will be deleted from our systems within 30 days.*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1359997981427761365)** (2 messages): 

> `Memory rollout, Context window, Token limits, Memory storage, Free-tier availability` 


- **Memory Rollout Q&A Shortfall**: A member inquired about the details of the new **Memory rollout**, seeking clarification from an OpenAI engineer.
   - Unfortunately, another member pointed out that direct interaction with OpenAI staff is rare on the official Discord, so *"nobody here but us users"*.
- **Context Window Gets Questioned**: A member questioned what the **Memory rollout** means for the **context window** in a specific conversation, inquiring if the chat's memory within a single conversation is no longer limited.
   - The member also inquired about **token limits** in relation to the new memory feature.
- **Delving into Memory Storage**: A member asked where the new **memory** is stored and whether it differs from the memory in settings.
   - They also seek to understand how the new **memory** feature relates to existing settings and storage mechanisms.
- **Free-Tier Users' Access**: A member inquired whether the new **memory** feature will also be available for **free-tier users**.
   - The question addresses the accessibility of the feature across different subscription tiers.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1359910152702001322)** (13 messages🔥): 

> `Assistant API context handling, Prompt Engineering tips, Generating multiple choice questions, Eliciting deeper responses from chatbots` 


- **Assistant API demands explicit context handling**: When using the **Assistant API**, it's crucial to instruct the model on **how to utilize the provided context** to achieve the desired outcome.
   - Ensure clarity in differentiating between relevant and irrelevant information within the context to guide the model effectively.
- **Prompt Engineering involves clear communication and verification**: Effective prompt engineering involves **clearly explaining** to the AI what you want it to do, using accurate language, and **carefully verifying the output** for accuracy and hallucination.
   - Fact-checking and vigilance are especially important when dealing with **math, sources, code**, or other details prone to AI error.
- **Crafting contextual MCQs depends on context accuracy**: The quality of multiple-choice questions (MCQs) generated depends heavily on the context provided, requiring careful **differentiation between relevant and nonsensical information**.
   - If the model includes seemingly irrelevant answers (e.g., *North America is a continent*), it may be drawing from the provided context, highlighting the need for clearer guidelines on what to ignore.
- **Exploring chat history for insightful chatbot responses**: For a more *deep* exploration, a member suggested prompting the chatbot to *consider the path of our chats over time*, asking it *what patterns do you notice? How have you and I both grown over time?*
   - This approach aims to uncover meaningful insights based on the **chat history and evolving interaction**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1359910152702001322)** (13 messages🔥): 

> `Prompt engineering, Assistant API, Context Contamination, Generating Multiple Choice Questions, Chat History Analysis` 


- **Mastering Prompt Engineering**: A member detailed a practical approach to prompt engineering: **picking a language AI understands**, clearly defining desired outputs, explaining tasks accurately, and carefully verifying results.
- **Instructions crucial for Assistant API**: When using the Assistant API, instructions must clearly specify **how the model should use the provided context** to achieve the desired outcome.
- **Challenges in Generating Multiple Choice Questions**: When generating multiple choice questions, relevance to the context is crucial; incorrect alternatives can arise if **the context isn't properly considered**.
- **Deepening Chat History Analysis**: A member suggested prompts to analyze chat history for patterns and growth, going beyond simple recall to **explore the evolution of conversations**.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1359605391088029706)** (147 messages🔥🔥): 

> `MCP server proxy, parallel tool calls, A2A vs MCP, Semantic tool calling, MCP registry optimizations` 


- ****Parallel MCP** Server Proxy Needed**: A member is seeking an MCP server proxy to call multiple server functions in parallel, aiming to read messages from **Slack** and **Discord** simultaneously to reduce waiting time.
   - It was suggested to use *asyncio.gather* in Python for custom clients to achieve parallel execution.
- **Is **A2A** a replacement for **MCP**?**: Members discussed Google's **A2A** (Agent-to-Agent) protocol and its relationship to **MCP**, with some seeing **A2A** as a potential attempt by Google to undermine **MCP**.
   - Some believe **A2A** is a strategy to limit **MCP**'s scope by handling agent-to-agent communication separately, while others see no spec overlap but potential vision overlap.
- ****Semantic Tool** Calling Concept Explored**: Members discussed semantic tool calling, a method to address the issue of LLMs getting confused when presented with a large number of tools (**200+**).
   - This approach involves using a **vector model** to embed tool descriptions and select a subset of tools based on semantic similarity to the current task, functioning as a RAG pipeline for tools.
- ****MCP registry** Loading Times Optimized**: The Glama team optimized the **MCP registry** and improved page load times by implementing every trick in the book, including tree-shaking, and welcomes feedback on the improvements for sites like [Paddle MCP Server](https://glama.ai/mcp/servers/@PaddleHQ/paddle-mcp-server).
   - The remaining problem is that they are still loading some JavaScript that should be tree-shaken.
- ****fastmcp** HTTP Streaming Contribution Sought**: The creator of [fastmcp](https://github.com/punkpeye/fastmcp) is seeking contributions for adding **HTTP streaming** capabilities.
   - One member offered to explore contributing after implementing their first streamable HTTP client/server.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1359606957429882990)** (6 messages): 

> `GraphQL MCP Server, Mobile Application Security MCP Server, MCP Server Hosting Service, Open Source MCP Server for Thingsboard` 


- **GraphQL MCP Server emerges!**: A member built a one-tool MCP server to leverage GitHub's full [GraphQL API](https://github.com/QuentinCody/github-graphql-mcp-server), addressing limitations of GitHub's official MCP Server.
   - The new server aims to reduce tool count while enhancing functionality with GitHub's GraphQL capabilities.
- **Mobile Application Security MCP Server surfaces**: An MCP server tailored for Mobile Application security was developed, using the [MobSF tool](https://github.com/pullkitsan/mobsf-mcp-server) to fetch and analyze reports.
   - This server automates the analysis of MobSF reports, providing security insights specific to mobile applications.
- **AnyContext.io offers MCP Server Hosting**: AnyContext.io is developing an MCP server hosting service with a focus on access control and security for enterprise customers, found at [AnyContext.io](https://www.anycontext.io/).
   - Future plans include self-hosting options, emphasizing robust security measures for MCP server deployments.
- **Thingsboard gets an Open Source MCP Server**: An open-source MCP server was created for Thingsboard, a data platform for IOT data, available at [Thingsboard MCP Server](https://github.com/AnyContext-ai/thingsboard-mcp-server).
   - This server integrates IT and OT, enabling data-driven industry applications through Thingsboard.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1359911426810052669)** (2 messages): 

> `AMD stock prices, Trump tariffs` 


- **AMD Plummets - Trump's Tariffs to Blame?**: A user inquired about the significant drop in **AMD's stock prices**, questioning whether **Trump's tariffs** were a contributing factor.
   - No specific link or information was provided in the discussion to confirm or deny the impact of **Trump's tariffs** on **AMD's stock**.
- **Stock Volatility Explained**: The user expressed unfamiliarity with stock market fluctuations, seeking potential reasons for **AMD's** decline.
   - The conversation implies a lack of direct causation established between specific political events (like **tariffs**) and **AMD's** performance without further evidence.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1359889603217068105)** (2 messages): 

> `Triton vs cuBLAS, Optimizing Triton Kernels, AMD Challenges` 


- **Triton's Answer to cuBLAS Performance**: A member inquired about a **Triton** equivalent to a resource that goes step by step through the process of optimizing a **Triton** kernel.
   - They expressed interest in comparing **Triton**'s performance against **cuBLAS** on **H100** and sought resources to practice optimization before **AMD** challenges.
- **AMD challenges**: A member is preparing for the **AMD** challenges.
   - The member wants to practice optimization techniques.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1359705774896451675)** (1 messages): 

> `Cutlass 3.x, Pointcloud project, ScatteredGather+GEMM fusion` 


- **Cutlass User Seeks ScatteredGather+GEMM Fusion**: A user working on a **pointcloud project** with **Cutlass 3.x** is facing memory usage issues with scattered-gather operations.
   - Specifically, they're looking for a **Cutlass kernel** that can fuse **ScatteredGather** and **GEMM** operations to optimize memory usage for large pointclouds.
- **Point Convolution Kernel with Cutlass**: The user's setup involves input tensors for **point features** `[B x M x C_in]`, **neighbor indices** `[B x N x K]`, and **weight matrices** `[B x N x K x C_mid]` for point convolution.
   - After gathering from input, weights, they have two tensors **A** shape:[B, K, C_in], **B** shape:[B, K, C_mid], and then GEMM to get **C** shape:[B, C_in, C_mid].


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1359641310402973738)** (1 messages): 

> `AMD $100K Competition, Kernel Optimization, MI300, Reasoning Models, FP8 GEMM` 


- **AMD & GPU MODE launch $100K competition!**: AMD and GPU MODE are hosting a **$100K competition** to accelerate inference kernels on **MI300**, starting April 15th, and ending June 12th, sign up [here](https://www.datamonsters.com/amd-developer-challenge-2025) to be eligible.
- **AMD competition focuses on 3 reasoning kernels**: The competition will consist of **3 kernels** that are core to inference in reasoning models: **FP8 GEMM, Multi-head Latent Attention, & Fused MOE**.
- **AMD competition supports Triton, Tinygrad, Torch**: The AMD competition supports most programming languages including **Triton, tinygrad, torch** and also supports **HIP kernels** via **PyTorch's `load_inline()`** function.
- **AMD competition prize includes trip to San Jose!**: Winning teams will get their **plane tickets paid** for to travel to **San Jose** for an awards ceremony.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1359914149752082552)** (1 messages): 

> `LLAMA models, Scout Model, QK Norm, L2 Norm, Chunked Attention` 


- **Scout Model skips QK Norm**: A member highlighted that the **Scout** model differs from others by using **L2 Norm** on **Q** and **K** instead of **QK Norm**, as noted in [this LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7315754121884487681/).
- **Attention Token Discrimination Questioned in Scout**: A member questioned whether the model can effectively differentiate tokens in attention, given the constraints of **norm(q) = 1** and **norm(k) = 1**.
   - They calculated a maximum softmax probability of approximately **0.000901281465** due to **chunked attention** and pondered if this is sufficient for the model to distinguish tokens.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1360005479824363701)** (2 messages): 

> `AlexNet Source Code` 


- **AlexNet Code Resurfaces!**: A member shared a link to the [AlexNet Source Code](https://github.com/computerhistory/AlexNet-Source-Code) on GitHub.
   - Another user replied with a dancing anime cat, expressing excitement.
- **Anime Cat Approves AlexNet**: A user reacted to the AlexNet source code link with a [dancing anime cat GIF](https://tenor.com/view/x3-gif-24301743).
   - This suggests strong approval and excitement regarding the availability of the code.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1359607946677452932)** (64 messages🔥🔥): 

> `Producer-Consumer Model on GPUs, Tensor-based GPU Database, Element-wise Kernel Performance, GPU Parallelism and Thread Execution, Nsight Compute (NCU) Usage` 


- **Producer-Consumer Model Deployed for Hopper GEMMs**: A member inquired about using **producers and consumers** in GPU kernels, particularly why this model works well for **GEMMs** on **Hopper** architecture, instead of everyone producing and consuming.
- **Tensor Database Project Seeks Guidance**: A member is developing a **tensor-based GPU accelerated database system** and seeks guidance, providing a link to their [repo](https://github.com/PyDevC/kero).
- **Coalesced Vector Loads Boost Element-Wise Kernels**: Regarding element-wise operation kernels, it was suggested to try **coarsening/vectorized loads** to get higher ILP and fewer instructions, assuming global memory access is already coalesced.
- **Ampere GPU Parallelism Questioned**: A member benchmarked a kernel on an **Ampere** datacenter GPU, observing that the runtime only doubled when the number of threads was twice what they expected, questioning why the runtime doesn't change until exceeding `#SMs * #WS per SM * 32`.
   - A senior member suggested checking the **assembly view in NCU** to identify warp stalls, mentioning that a loop-carried dependency in the example code introduces stalls, as each `fadd` in one thread/warp has to wait for the previous one to finish before it can start.
- **Profiling FADD Performance on Ampere with NCU**: A member used **Nsight Compute (NCU)** to profile a kernel with `FADD` instructions on an Ampere A100 GPU, questioning why they saw `FMA` instructions in the assembly.
   - It was clarified that the **FMA pipeline** also handles **FADD/FMUL/FMAD** instructions, and that with a fully utilized pipeline, additional warps won't increase throughput due to the artificial example.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1359826545333633026)** (24 messages🔥): 

> `tilelang performance, AMD developer challenge, MI300X benchmarks, ROCm profilers` 


- **Tilelang smashes FlashMLA perf on MI300X**: Members reported impressive **FlashMLA** performance using **Tilelang** on **MI300X**, linking to a [benchmark script](https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/amd/benchmark_mla_decode_amd_tilelang.py).
- **AMD Challenge Draws Interest**: Members discussed the [AMD Developer Challenge 2025](https://www.datamonsters.com/amd-developer-challenge-2025#wf-form-AMD-Email-Form), as well as the need for a simpler way to install tilelang on AMD.
   - One shared their busy schedule while others noted the need to improve the installation method, referencing the [install_rocm.sh script](https://github.com/tile-ai/tilelang/blob/main/install_rocm.sh) and a ready-made [docker image](https://github.com/gpu-mode/discord-cluster-manager/blob/main/docker/amd-docker.Dockerfile).
- **MI300X Benchmarks Show Inconsistencies**: Members benchmarking end-to-end **Llama3-8B** decoding speed on the **MI300X** reported inconsistent results, with different tokens/sec even within minutes of each run.
   - One user mentioned that while the theoretical fp16 tflops of **MI300X** is **1300TFlops**, *rocblas can only achieve 500 TFlops*.
- **ROCm Profilers Recommended**: Members discussed best profiler tools for **ROCm** kernels, recommending **rocprof** and the related **ROCm Compute** and **ROCm Systems** profilers.
   - One user noted that *the main advantage is that they have some visualization options which sort of work, but none of them work very well*.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1359904361144127779)** (3 messages): 

> `SFT+CoT, arc agi 2, arc-agi 1 CoT` 


- ****Arc-AGI 2** Questions Prompted**: A member asked about **SFT+CoT** on **arc agi 2** datasets and pointed to [arc-agi 1 CoT](https://github.com/open-thought/arc-agi-2/blob/main/arc-1/annotated-re-arc/core_idea_samples.json).
   - The member inquired whether the given **CoTs** are correct, since they were generated with `Claude`.
- **CoT Samples Questioned for **Arc-AGI 2****: A member expressed doubt that generating sample **CoTs** with `Claude` would be useful for **arc-agi 2**, citing preliminary investigations.
   - The member mentioned that LLMs are doing a bad job on the task, and expressed hope for high quality human annotations in the future.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1359619657815949595)** (6 messages): 

> `AMD MI300 submissions, OpenAI API alternatives` 


- **AMD MI300 Submission Incoming!**: Members are internally testing the kernels for the **AMD MI300** and expect submissions to be enabled next week.
   - The team has been testing them extensively on the dev server and they *work great*, so just a few days to iron out final details.
- **Seeking Cheap OpenAI API Websearch Alternatives**: A member inquired about cheap/free alternative solutions to using the **OpenAI API** for web search.
   - No alternatives were offered in this message.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1359660009645936692)** (9 messages🔥): 

> `matmul, A100, T4, H100, L4` 


- **Matmul Leaderboard Mania**: Submissions to the `matmul` leaderboard on GPUS using **Modal runners** have succeeded across various GPUs, including **A100**, **T4**, **H100**, and **L4**.
   - Submission ids range from **3566** to **3571**, and also include test submission **3573** and benchmark submission **3574**.
- **Modal Runners Dominate Matmul Submissions**: The **Modal runners** continue to show success in submitting to the `matmul` leaderboard across different GPU architectures.
   - These successful submissions indicate stable performance and reliability of the **Modal runners** in handling matrix multiplication benchmarks on diverse hardware.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1359642330746978546)** (32 messages🔥): 

> `AMD Competition Details, AMD Contractor Participation, Team Formation, AMD Kernel Experience, HIP vs CUDA` 


- **AMD Hosts Competition, Welcomes Community**: AMD is hosting a competition, marking a significant investment in the community, though the specific eligibility criteria for AMD contractors is yet to be fully clarified, cc'ing <@1167171518736891944> and <@1160995582916165794> for eligibility.
   - In similar competitions they allowed people to participate just not be eligible to win prize money.
- **Solo or Team? Community Clarifies Competition Rules**: Participants can compete solo or form teams, and a dedicated space for finding teammates has been created in <#1359668821127856128>.
   - The process of submitting solutions involves using the `/leaderboard submit ranked` command with the relevant file.
- **AMD GPU Novices Welcome; Resources Available**: Even those without prior AMD kernel experience are encouraged to participate, using Triton as they would for NVIDIA, focusing on learning opportunities, with AMD GPUs provided.
   - Documentation on the submission platform is available [here](https://gpu-mode.github.io/discord-cluster-manager/docs/intro), and a related launch talk can be found [here](https://www.youtube.com/watch?v=yc8N6xfGTRM&t=1s).
- **HIP vs CUDA Similarities Explored**: HIP is noted to be very similar to CUDA by design, with the most significant difference being a warp size of **64** instead of **32**.
   - Optimizing for the underlying architecture, however, may present different challenges.
- **Problem Details Dropping Soon, MI300 is Only Target**: Specific problem constraints will be released on **April 15th**, problem by problem, and the competition will exclusively target the **MI300 GPU**.
   - Supported languages include Torch, Triton, and Aiter; reference code and allowed languages are to be announced.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1359675055138930888)** (39 messages🔥): 

> `Modular Meetup, GPU programming with MAX and Mojo, Compiler status, Blind programmer using Mojo, GPU Support on Mojo` 


- **Modular hosting Meetup in Los Altos, CA**: Modular is hosting a meetup in Los Altos, CA on **April 24th**, featuring a talk on **GPU programming with MAX and Mojo**.
- **Users await Open Source Compiler Release**: Some users are eagerly awaiting the open-sourcing of the compiler to *finally have some fun working on it*.
- **Blind Programmer dives into Mojo, faces GPU and VSCode issues**: A blind programmer named Deniz is diving into Mojo but is facing issues with **GPU programming** and **VSCode extension discrepancies**.
   - Deniz found that *mojo compiler complaints me about unexisting methods of standard library functions in mojo, while vscode extension kindly provides me them.*
- **GPU support limited to Nvidia for now**: Currently, **Mojo only supports Nvidia GPUs** due to bringup work, with AMD support planned but Intel being a low priority due to their data center GPU market share.
   - It was explained that *Intel is low priority because they screwed up their datacenter GPUs (from an AI perspective), so they have very low market share, and they primarily use Gaudi for AI*.
- **VSCode Nightly Extension Mismatch Resolved**: A user had issues with discrepancies between the compiler and VSCode extension, and was instructed to install the **Mojo (nightly) extension** to match their nightly Mojo version.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1359686832253571093)** (47 messages🔥): 

> `Mojo OS kernel, __mlir_fold, MimIR paper, Mojo package installation, Contribution graph` 


- **MojOS kernel?**: A user inquired *if Mojo is going to have an OS kernel*, sparking brief discussion.
   - Another user jokingly replied *MojOS*, while another asked *What is the point of having a kernel for mojo*?
- **`__mlir_fold` sounds useful, create request**: A user suggested creating a feature request on GitHub for `__mlir_fold`, a feature that sounds useful, with a [related issue](https://github.com/modular/max/issues/4252) surfacing in other contexts.
   - A link to [issue 4315](https://github.com/modular/max/issues/4315) explains exactly how.
- **MimIR Paper Mentions a Problem**: The MimIR paper was mentioned, in relation to a [problem](https://github.com/modular/max/issues/4315) described in the modular max issues.
   - A link to the [AnyDSL/MimIR GitHub](https://github.com/AnyDSL/MimIR) was shared.
- **`magic add` git URLs**: Users discussed how to add a mojo package, suggesting that `magic add` may work with **git URLs**.
   - Some users stated that `magic add` cannot be used with GitHub URLs and that it might require manual addition to the file.
- **Extramojo installs via `magic add`**: The command `magic add extramojo` was confirmed to work, adding **extramojo version 0.12.0 or greater, but less than 0.13** to the current environment.
   - A user also mentioned that they couldn't find a way to make git URLs work, but is ok using things that have recipes for now.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1359671470367969410)** (15 messages🔥): 

> `MAX disk space, Max cache size config, magic clean cache` 


- **MAX Install Size Inflates Disk**: Members are discovering that multiple versions of **MAX** are using excessive disk space (38GB in one case), located in `Users/nick/Library/Caches/rattler/cache`.
   - A member mentioned that *"A 'max cache size' config would be nice."
- **Nightly Builds Hog Space, But Help is Coming**: The proliferation of **nightly builds** was blamed for the **excessive disk usage**, which *"is way better than CUDA"*.
   - A member stated that *"there's no real reason to have more than a few versions hanging around"*, then added, *"I'll file an issue but I wouldn't count on us getting around to it immediately, plenty of things to improve"*
- **Cache Cleaning Spells**: A member recommended running `magic clean cache -y` via **cron** to reclaim disk space.
   - Another user recommended running `magic clean cache --conda` to avoid nuking the entire Python cache.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1359605400793645316)** (30 messages🔥): 

> `Free Jupyter Notebooks, ZeroGPU Quota, Sidebar Closing Courses, Saving HTML Code on Phone, LLM Training Libs` 


- **Free Jupyter Notebooks for AI Model Training?**: A member asked for alternatives to Google Colab for training AI models after running out of free trials.
   - Another member suggested using **Hugging Face** to launch **Jupyter Lab** for free, describing it as a *really underrated feature*.
- **ZeroGPU Charges Full Time?**: A member reported their **ZeroGPU** space spends the full **120s** of requested quota even when generation takes less time.
   - A member noted this can be set using a decorator, and is a limitation because **Zero GPU** is a shared resource counted by occupation time rather than a time-based rental.
- **Saving HTML Code on Phone Frustrations**: A member asked how to save HTML code created on a website using their phone, finding no download button.
   - Another user suggested copy/pasting to notepad, but the original poster claimed that they *can only do it line by line* on their phone.
- **Quest for C/C++/Rust/Zig LLM Training Libs**: A member inquired about existing **C/C++/Rust/Zig** LLM training libraries, noting concerns about the maturity of **candle** and the low-level nature of **tch-rs**.
   - They speculated whether corporations use low-level libs instead of **transformers python lib** for training their top-tier SOTA LLMs.
- **GPU Enabled HF Space Free Trial?**: A member asked if there's a free trial for GPU-enabled **HF Spaces**, pointing to their existing [image classification app](https://huggingface.co/spaces/arasyidi/eurosat_ztm_practice) and its **CPU/CUDA** selection logic.
   - They expressed uncertainty whether the app currently utilizes GPU resources.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1359646659797057556)** (16 messages🔥): 

> `Google Firebase Studio, NextJs, LAION initiative, instant interview bot` 


- **Firebase Studio Builds Tic-Tac-Toe Game**: A member built a modified version of tic-tac-toe in an hour with **Google Firebase Studio** (unified version of **Project IDX** launched earlier) in **NextJs** using their built in prototyping agent and in-browser IDE, linked [here](https://9000-idx-studio-1744279740559.cluster-xpmcxs2fjnhg6xvn446ubtgpio.cloudworkstations.dev/).
   - The creator was disheartened to see the dataviewer turned off and offered to submit a PR to fix compatibility.
- **LAION initiative**: A member received a server invite to **LAION**.
   - Another member indicated they are the **MLOPS engineeer** at LAION.
- **Instant Interview Bot Debuts**: A member shared an *instant interview* bot built to familiarize themselves with the **Hugging Face** platform and garner attention for their LinkedIn/job search, linked [here](https://sites.google.com/view/isaiahmontoya/instant-interview).


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1359846651384299561)** (1 messages): 

> `Diffusers v0.33.0, Image & Video Gen Models, Memory Optimizations, torch.compile() support for LoRAs` 


- **Diffusers Drops v0.33.0**: A new release of **Diffusers** is out, including a suite of **image and video generation models**, as well as a wide suite of **memory optimizations with caching**.
   - Release **v0.33.0** also brings `torch.compile()` support when hotswapping **LoRAs**, with full notes at the [release page](https://github.com/huggingface/diffusers/releases/tag/v0.33.0).
- **Diffusers v0.33.0 Memory Optimizations**: The release brings a wide suite of **memory optimizations with caching** to improve performance.
   - The optimizations aim to reduce memory footprint and speed up computations, making it easier to work with large models and datasets.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1359634443878400031)** (1 messages): 

> `Street Fighter 6, OpenCV, Discord Chat Bot, Gemme3, Langchain` 


- **SF6 Coaching Bot Seeks OpenCV Expertise**: A developer is building a **Python script** for a **Discord chat bot** that uses **computer vision** to analyze **Street Fighter 6** gameplay in real-time and offer coaching feedback.
   - They are seeking a **computer vision expert** to help improve **OpenCV's** ability to find UI elements, with the bot currently using **Gemme3**, **Langchain**, **OBS**, **Discord**, **ChromaDB** (with all **SF6** frame data), and per-user session memory.
- **Real-time SF6 Analysis for Coaching**: The bot aims to provide **real-time coaching feedback** by analyzing **Street Fighter 6** gameplay using computer vision.
   - The current challenge involves enhancing **OpenCV's** capabilities to accurately identify and track UI elements within the game environment.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1359752568192106587)** (1 messages): 

> `DocLing Project, Mapping Headings, Information Column, Paper ID Download` 


- **DocLing Project: Computationally Heavy but Effective**: A member reported that the **DocLing project** yields better results but is computationally heavy, and they intend to test the **lite** version.
   - The main issue is that specific headings (formula names) intended for the **methodology** category are misclassified into the **other information** column.
- **Mapping Headings Challenge**: A member is facing challenges with mapping headings, where some headings with unique names (like formula names) that should fall under the **methodology** category are incorrectly categorized into an **other information** column.
   - The categorization varies: some papers get proper layouts while others don't and for now, only the paper ID is passed to the downloader and then to **DocLing** for column mapping.
- **Seeking Ideas for Information Column Usage**: A member is seeking ideas on how to effectively utilize the information from the **other information** column, as only **30-40%** of the mapping is currently accurate.
   - They are open to suggestions, indicating that their current approach might be flawed and they plan to share the code later, asking for guidance on the appropriate sharing venue.
- **Paper ID Based Downloads**: The member is currently downloading papers by passing the **paper ID** to the downloader and then to **DocLing** for mapping to columns.
   - The member has problems getting layout right on some papers and wants ideas how to make use of information from others columns.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1359731988122046475)** (3 messages): 

> `smolagents CodeAgent, Ollama, Llama3.2, qwen2:7b, Error in code parsing` 


- **User struggles with SmolAgents CodeAgent**: A member reported running into a *"Error in code parsing"* when using **smolagents CodeAgent** with **Ollama**, either **Llama3.2:latest** or **qwen2:7b**, for playlist search and suggest_menu tasks.
   - The parsing error often leads to hallucinations, such as *"calculating the pope's age."*
- **Root cause analysis for Code Parsing Error**: The same member is wondering about the cause for the error, suggesting it might be due to: a) the small size of the models, b) not using models specialized in code generation, or c) not structuring the output in a **JSON/python** format to avoid generating non-code format.
   - No further suggestions were offered.
- **Unit4: Final Assessment delayed release**: A member inquired about the release date of *Unit4: Final Assessment*.
   - No further details or specific timeline were mentioned.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1359622687173120022)** (27 messages🔥): 

> `Google ADK, Torch Installation Issues, Gemini Flash API, Ollama Models for AI Agents, Langchain vs LlamaIndex` 


- **Google debuts Agent Development Kit**: Google launched the [Agent Development Kit (ADK)](https://google.github.io/adk-docs/) to facilitate agent interoperability, as announced in a [Google Developers Blog post](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) and via [Firebase Studio](https://studio.firebase.google.com/).
- **Colab Torch install proving troublesome**: Users report issues with **Torch** installation in **Google Colab** and seek alternative methods.
- **Gemini Flash API gains traction**: The **Gemini Flash API** is considered a fast alternative with a decent free tier, especially when compared to **Qwen**.
   - One user suggested trying it, indicating it's *much faster*.
- **Gemma 3 leads Ollama model choices**: For running AI agents on **Ollama**, `gemma3:27b-qat-q4_0` is favored; modified with `PetrosStav/gemma3-tools` for better tool-calling support by changing the FROM location to the HF GGUF version, plus potentially smaller models like `bartowski/Qwen2.5-Coder:3B-Instruct-Q5_K_L`.
   - Some pointed out that their model is not generating valid code for execution.
- **LangChain Documentation Woes**: Users express frustration with **LangChain's** documentation, citing deprecated pages and difficulty in finding necessary information.
   - Alternatives like **HF smolagents** framework and direct use of **OpenAI** or **Gemini SDKs** are proposed for easier experimentation and prototyping.


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1359959536147894302)** (1 messages): 

> `NotebookLM Plus, Audio Overviews, Source Limits, User Research` 


- **NotebookLM Plus Limit-Hitters get Research Opp!**: The team is seeking users of **NotebookLM Plus** who are encountering limits related to the number of sources, **Audio Overviews**, or chat features.
   - Interested users are encouraged to sign up for a UXR session via [this form](https://forms.gle/aSob5VpjX3C5qjpD7) to discuss their use cases and strategies for dealing with source limits.
- **UXR Session Open for NotebookLM Plus Users**: Users who are hitting limits on **NotebookLM Plus**, particularly with the number of sources, the creation of **Audio Overviews**, or chat interactions, are invited to participate in a user research session.
   - The research aims to understand specific use cases and how users are navigating the existing source limits; sign-up is available through [this link](https://forms.gle/aSob5VpjX3C5qjpD7).


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1359617952416141522)** (18 messages🔥): 

> `Discord Guidelines, NotebookLM as Notetaking App, Discover Feature Use Case, NotebookLM Plus, Time References in Source Document Titles` 


- ****Discord** Guidelines Enforced**: A reminder was issued to abide by the [Discord guidelines](https://discord.com/channels/1124403332262412368) and utilize the space as a helpful resource for **NotebookLM** users.
   - The moderators announced that they will be banning anyone who is spamming or posting unrelated content to protect the community.
- **NotebookLM's **Notetaking** App Status Debated**: A user lamented that Google doesn't see **NotebookLM** as a notetaking app, potentially leaving a gap in their software portfolio.
   - Another user responded that **NotebookLM** is being rapidly developed and may evolve into a notetaking app with Google Drive integration in the future.
- **User Explores **Discover** Feature for Author Research**: A user is planning to use the **Discover** feature to gather information on favorite authors, pulling it together within **NLM**.
   - The user intends to use such notebooks temporarily, copying any interesting findings to **Google Keep** and deleting the notebook once their curiosity is satisfied.
- **NotebookLM Plus Explored for **Virtual Tabletop** Module Creation**: A user is exploring **NotebookLM Plus** to see if it can function as a hyper-focused assistant for creating a virtual tabletop module.
   - They want to know if **NotebookLM Plus** can help them produce a module with **HTML**, **CSS**, **JavaScript**, and **JSON** using links to sites, videos, articles, and long-form research, and compared it to Gemini Advanced Canvas.
- ****Time References** Improve Timeline Accuracy**: A user found that adding time references to source document titles improves timeline accuracy.
   - For example, titling a bill as *"House Bill 1496, considered in 2025 session, based on similar HB (insert number) that passed in 2023"* helps in tracking related bills.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1359608544466436138)** (50 messages🔥): 

> `Gemini and TPU, Mobile app release, Chat Style Modifications, PDF image recognition, Source Discovery` 


- **Gemini Gains Ground with Google's Gem TPU**: Google's co-development of **Gemini** and **TPU** together, along with the announcement of **TPU version 7**, suggests that better **TPUs** will lead to improved **Gemini** models and, consequently, a better **Notebook LM** experience.
- **Mobile Notebook LM App Anticipated**: Users expressed excitement over the announcement of a mobile app for **Notebook LM**, with the expectation that it will improve upon the disappointing mobile web experience, particularly regarding audio preview issues.
- **Querying Chat Style's Effect on Info Retrieval**: A user inquired about the impact of modifying chat styles (**default**, **analyst**, **guide**, **custom**) on information retrieval, questioning whether it affects retrieval accuracy or solely the style of expression.
- **PDF Image Recognition Feature Glitches**: Several users reported that **Notebook LM** is not recognizing images in PDFs, despite a prior announcement suggesting it should, with one user noting that Gemini Advanced does a better job of extracting text from images compared to **Notebook LM** currently.
- **Source Discovery Feature: Premium Users Still Waiting**: Multiple paid **Notebook LM** users expressed frustration over not having access to the **"Discover Sources"** feature, despite it being available on their free accounts and the feature supposedly rolling out.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1359620750268567592)** (32 messages🔥): 

> `Psyche Explainer, Logits going negative, Grok 3, Quasar and Optimus Alpha, Live Modulation` 


- **User Shares Psyche Explainer**: A member shared an explainer about **Psyche**.
   - Another member skimmed it and said *good work man!*
- **Members Ponder Negative Logits**: A member inquired about why **logits** go negative.
   - The member also asked about **Grok 3**
- **Quasar and Optimus Alpha models coming soon**: Members discuss **Quasar** and **Optimus Alpha**.
   - They don't seem to be reasoning models and there is a *total release overload right now*, plus **Qwen** did say they need more time today.
- **Live Modulation thinking models debut**: Members discuss a new paper on **live modulation**, taking a fused memory vector (from prior prompts), evolving it through a recurrent layer (the Spray Layer), and injecting it into the model’s output logic at generation time, linking to [this paper](https://cmu-l3.github.io/l1/).
   - Now we can have thinking models that don't need to think for 10k tokens.
- **Control-vectors for Augmenting Models Discussion**: A member asked why we don't use **vgel's control-vectors** to augment models to make them think, behave, and, align more with our intended purposes (use-cases) vs just prompting them when we are generating AI responses for a target dataset.
   - Another member responded saying that they've done some experiments, but *it isn't really easy to be applicable generally* as it is much more unstable with any control vectors applied, but it's definitely being explored.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1359771577440538716)** (20 messages🔥): 

> `Nous API system prompts, VLMs for OCR on device, OpenAI API changes, Nemotron model behavior` 


- **Trouble with Nous API System Prompts Disappears**: A member reported trouble using the **Nous API** with a system prompt, with the AI not following instructions, but later resolved the issue.
   - The `<think>` traces are returned in `reasoning_content`, but can be malformed when the response is truncated by the token limit; *documenting this behavior would be helpful*.
- **Small VLMs Needed for OCR on Device**: A member is seeking **small and capable VLMs** to perform OCR on device, aiming to create a Swift app using **CoreML** with **ONNX**.
   - No specific VLMs were recommended in the available messages.
- **OpenAI API Role Name Change Speculation**: A member speculated that **OpenAI** might have changed the system role name to `developer` in their chat completion API, causing issues with libraries.
   - Another member dismissed this idea, suggesting the problem lies at the model level rather than with inference engine pre/post-processing.
- **Nemotron returns Empty Think Tags**: The new **Nemotron model** returns an empty `<think></think>` tag when reasoning isn't used, which someone found helpful for easier output processing.
   - There was no further discussion on this topic.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1359608438702870756)** (7 messages): 

> `mlss2025.mlinpl.org, self encrypted cloud backup/sync, use your own openrouter API key, local chats` 


- **Cloud Backup Goes Local**: A member shared a link to [MLSS 2025](https://mlss2025.mlinpl.org/) and a [tweet](https://x.com/563defi/status/1909976170990313594) about a project.
   - The project seems to be related to **local chats** and **self-encrypted cloud backup/sync**, using your own **OpenRouter API key**, as described in an attached video.
- **OpenRouter API Key enables Local Chats**: A member posted a [tweet](https://x.com/btibor91/status/1910237861674353108) about a project involving **local chats** and **self-encrypted cloud backup/sync**.
   - The system utilizes your own **OpenRouter API key** for functionality, as showcased in an attached video, prompting a user to comment that it *looks neat*.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1359634546873860147)** (23 messages🔥): 

> `MCP Explained, MCP vs Regular Tool Calls, MCP Ethereum Analogy, Google A2A Announcement` 


- **MCP is a communal tool-building remote server**: **MCP** is not just remote tools discovered automatically, but a whole remote server that runs and can provide tools, prompt templates, **RAG-like data**, even **LLM sampling** exposed in a standardized way.
   - It facilitates *communal tool building*, allowing entities like Google to create tools (e.g., calendar event creation) once and expose them for any client model to use via an **MCP server**.
- **MCP leverages a Standardized API**: While **MCP** doesn't change the tool call execution itself, its advantage lies in enabling any client model to use tools exposed in an MCP server, promoting **communal integrations with a standardized API**.
   - Tool use is just one part of it.
- **MCP analogized to Ethereum Smart Contracts**: A member made an analogy that **MCP** is similar to **smart contracts on Ethereum**, except without the baggage of blockchain.
   - Another member clarified *idk how apt that comparison is* that **MCP** is more like communal integrations with a standardized API.
- **Google A2A echoes MCP**: Google just announced something oddly similar called [A2A (Agent to Agent)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) for agent interoperability.
   - One member noted that their implementation looks like what you would expect from a **C++ programmer**.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1359606355295600661)** (21 messages🔥): 

> `Sharding Strategies, Llama4 Support, Scout model, Maverick model, iRoPE implementation` 


- **Sharding Strategies Are a Go!**: Supporting different **sharding strategies** is pretty straightforward, though an issue has been open for over a year without prioritization due to lack of demand.
- **Llama4 Support on Deck**: A member expressed interest in contributing to [#2570](https://github.com/pytorch/torchtune/pull/2570) (**Llama4 support**) and offered assistance with relevant issues.
- **Scout and Maverick Models Debut!**: The **Scout model** has reached a decent state on text data, with plans for multimodal support, while the **Maverick model** is still undergoing testing.
- **iRoPE Implementation Needs Tuning**: The current **iRoPE implementation** uses flex but may need optimization due to potential recompiles during generation; getting the model to work with **torch compile** is also a priority.
- **\"Detach\" Yourself From That Loss!**: A warning about converting a tensor with `requires_grad=True` to a scalar was reported, but a member offered an easy fix using `running_loss.detach()` for this and other recipes.
   - Another member replied that *when seed is fixed all unit tolerances may be brought down*.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1359607539402145876)** (35 messages🔥): 

> `Code Formatting Preferences, DeepSeek vs. Qwen Model Distillation, GPT4ALL logging, Small LLMs for LocalDocs, Chocolatine-3B-Instruct-DPO-v1.2-GGUF` 


- **Code Formatting Preferences Debated**: A member asked about getting the AI to format code with spaces around curly braces, e.g., `function bla(bla) { //Code }` instead of `function bla(bla){ //Code }`.
   - Another member suggested not focusing on code formatting in prompts, recommending refactoring the code later with better tools, as simpler prompts yield better model responses.
- **DeepSeek R1 Distill Qwen Demystified**: Members discussed the meaning behind the model name **DeepSeek R1 Distill Qwen**, with one suggesting it involves knowledge distillation from a larger DeepSeek model to a smaller Qwen model.
   - Another member linked to the [official readme](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) and explained that the smaller model (**Qwen 7b**) is fine-tuned with data generated by the larger model (**Deepseek R1**).
- **GPT4ALL Lacks Logging Capabilities**: A user inquired about setting up user logs in **GPT4ALL** for educational purposes, but another member clarified that **GPT4ALL** doesn't have logging features.
   - The user was advised to use **Llamma.cpp** as an alternative, suggesting it may offer the desired logging functionality.
- **Small LLMs Excel at Local Document Search**: A member shared the opinion that small LLMs are ideal for searching within local documents due to their speed and reduced confabulation.
   - They even speculated that a full LLM might not be necessary for **LocalDocs**, suggesting embeddings and direct links to page numbers or paragraphs could suffice.
- **Chocolatine-3B-Instruct-DPO-v1.2-GGUF recommended**: A member recommended [Chocolatine-3B-Instruct-DPO-v1.2-GGUF](https://huggingface.co/bartowski/Chocolatine-3B-Instruct-DPO-v1.2-GGUF), noting its suitability for handling approximately 8 snippets of 1024 characters.
   - While acknowledging it's a French model, the member asserted that the 14B version performs well in German too.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1359604671614030059)** (8 messages🔥): 

> `Kuramoto oscillatory networks, Google's Ironwood TPU, Agent Development Kit` 


- **Kuramoto Network Paper Shared**: A member shared a paper on **artificial Kuramoto oscillatory networks**, and another member requested the author's name, which was resolved.
   - The specific link to the paper was not provided in the context.
- **Ironwood TPU Enters the Chat**: A member shared a link to the [Google Cloud blog](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/) discussing **Ironwood TPUs** and their potential for inference.
   - The post highlights the **age of inference** and the capabilities of these new TPUs.
- **Agent Development Kit Announced by Google**: A member shared Google's [Agent Development Kit](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/), designed to easily build **multi-agent applications**.
   - Another member mentioned they are switching their agent to this ADK because *instead of a Prompt that is pre-built, i have an agent, that talk with other agents(before they were just modules), gather information, and constructs the prompt.*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1359607102158671934)** (21 messages🔥): 

> `Mollifiers in ML Research, Label Smoothing, Reasoning on/off Models, Blurring Issues, Transformer Architectures` 


- ****Mollifiers** Smooth the Way in **ML****: A member inquired about interesting uses of **mollifiers** in **ML research**, linking to the [Wikipedia article](https://en.m.wikipedia.org/wiki/Mollifier) for reference.
   - Discussion suggested potential applications in generalizing proofs across activation functions and enabling sampling from constrained sets, referencing the paper [Sampling from Constrained Sets](https://openreview.net/pdf?id=zWy7dqOcel).
- **Unlocking Reasoning on/off Models**: Someone asked how reasoning on/off models are made, questioning whether consistent system prompts ensure reasoning behavior isn't triggered by other prompts.
   - They questioned if *the reasoning behaviour won't be triggered by other system prompts*.
- ****Blurring** the Line: Data's Impact on Network Performance**: A member mentioned that training on blurred data can permanently inhibit a network's performance, citing an impactful **CVPR paper**.
   - This suggests that the quality of training data smoothing methods critically impacts model outcomes.
- **Scale It Up: Transformer Tweaks for Better Performance**: A member observed that adding a zero-initialized learnable per-channel scale on the last linear layer of each block in a transformer decreases loss at a similar rate but slows main path activation RMS growth.
   - The observation was made in the context of **muon**, prompting further investigation into the underlying causes.
- **Lambada Parity Achieved!**: A member announced that they've achieved parity on **Lambada**.
   - However, they noted that **MMLU performance** is still down, indicating a tradeoff in performance metrics.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1359703571695796265)** (5 messages): 

> `Influence Functions, Marketing vs Technical teams` 


- **Influence Functions Implied?**: Some members felt that the marketing team's language implied the use of **influence functions** or similar techniques.
   - However, others suggested it might just be **string matching** over the full dataset, causing some disappointment.
- **String Matching Speculation**: Some members speculated that the technique used was "just" **string matching** over the full dataset.
   - This led to disappointment after initially inferring more sophisticated methods like influence functions.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1359700276596048083)** (21 messages🔥): 

> `ChatGPT web app speed vs. RAG search speed, AgentWorkflow linearity, Agents as tools, Job availability` 


- **Debate over ChatGPT's Speedy Responses**: Members discuss why the **ChatGPT web app** feels faster than a local **RAG search** with 500 documents; some suggest it's due to [streaming](https://blog.cloudflare.com/what-is-http-streaming/), while others ask for more specifics on the dataset.
   - One member recommended using [observation modules](https://docs.llamaindex.ai/en/stable/module_guides/observability/) to identify time consumption in **Retrieval** and **Generation** processes for debugging purposes.
- **AgentWorkflow's Linearity Questioned**: A member asked if `AgentWorkflow` only works linearly, sharing an example where the root agent doesn't properly handoff to multiple agents to generate a final answer, attaching a [script](https://cdn.discordapp.com/attachments/1059201661417037995/1359926120337903838/can_hand_off_to.py?ex=67f94165&is=67f7efe5&hm=9ab6ec5f3374e03ec7e0999058141604e350e2c62219f91c58d26c16f2b8c93e&).
   - Another member confirmed that only one agent/speaker is active at a time, suggesting using agents as tools for other agents to achieve splitting functionality.
- **Agents as Tools Explored**: A member inquired about converting agents to tools within LlamaIndex, similar to `FunctionTools.from_agent()`.
   - Another member advised writing a function that calls the agent and integrating it into a function tool, emphasizing the flexibility of this approach and noting the need for documentation.
- **Developer Job Inquiry Pops Up**: A member inquired about developer job availability, signaling interest in offering their development services.
   - No specific roles or projects were mentioned, but the user expressed a willingness to help.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1359828126883905597)** (4 messages): 

> `Nvidia nvdec, Mesa branch, Video decode` 


- **Nvidia's nvdec Spotted in Discussion**: A member mentioned **nvdec**, documented in **NVIDIA's open-gpu-doc**, referencing a [YouTube video](https://www.youtube.com/watch?v=rsxCZAE8QNA).
   - They noted that class headers for **video decode** are available, and there's a **mesa branch** with **h264** already implemented, suggesting **hevc** shouldn't be far behind.
- **Bounty Hunters Sought for Video Decoding**: A member suggested claiming the bounty for **video decoding** tasks.
   - This implies there may be financial incentives for contributing to the development of **video decoding** capabilities, particularly in the context of the discussed **NVIDIA** and **mesa branch** implementations.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1359669202386157719)** (2 messages): 

> `MultiLazyBuffer error, Llama unexpected output, _transfer function, BufferCopy fallback` 


- **Llama Spits Unexpected Output**: A user reported getting an unexpected output from **Llama** instead of a **MultiLazyBuffer error**, referencing a [failed benchmark](https://github.com/tinygrad/tinygrad/actions/runs/13956645977/job/39069313429).
   - They suggested it might be related to syncing in the **_transfer function**.
- **BufferCopy Fallback workaround**: The same user found that disabling the **_transfer function** and making the **Upat** in realize fallback to **BufferCopy** makes everything work fine.
   - The user notes that this is *not a root fix*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1359815143353090181)** (5 messages): 

> `Codebase Context, Caching Subsystem` 


- **Harness Codebase Context**: A member inquired about methods to **maintain context of the entire codebase**.
   - No solutions were provided, indicating a challenge in codebase management.
- **Caching Subsystem ETA**: A member asked for updates on the **new caching subsystem**.
   - Another member confirmed they are working on it, with an **ETA by the end of next week**.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1360012937644867856)** (2 messages): 

> `Course Deadlines, Course Website` 


- **Will Course Deadline Pass?**: A member inquired if it's possible to complete the course and get the certificate, even though the course started earlier.
   - Another member confirmed that *everything needed is on the [course website](https://llmagents-learning.org/sp25)*.
- **Course Website Link**: A member provided a link to the [course website](https://llmagents-learning.org/sp25).
   - The course website likely contains the course schedule and deadlines.


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

kaithwas.abhijeet: Has anyone fine-tuned Aya vision 8B parameter model using LoRA or QLoRA?
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1359963496053342460)** (1 messages): 

> `Grok-3, Windsurf, Pricing, Credits` 


- ****Grok 3** hits **Windsurf**!**: **Grok-3** is now available in Windsurf with **1.0 user prompt credits** on every message and **1.0 flow action credits** on each tool call, as announced on [X](https://x.com/windsurf_ai/status/1910402452739993764).
- ****Grok-3-mini** debut for mere mortals**: Windsurf is debuting **Grok-3-mini (Thinking)** at a reduced rate: **0.125 user prompt credits** on every message and **0.125 flow action credits** on each tool call.
   - The announcement boasts about its speed and mentions it is available in individual paid plans.


  

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

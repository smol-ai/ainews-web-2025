---
id: 181c127f-3bfe-45c7-9adc-372f7f24c16b
title: not much happened today
date: '2025-03-28T23:18:38.632397Z'
original_slug: ainews-not-much-happened-today-9938
description: >-
  **GPT-4o** has been recognized as the leading non-reasoning coding model as of
  *March 2025*, praised for its improved coding, instruction following, and
  image generation features, though it still trails behind reasoning models.
  Meanwhile, **Gemini 2.5 Pro** stands out for its advanced audio and video
  understanding capabilities, including processing *~1 hour* of video or *~2
  hours* of podcast transcription in a single request, and is now accessible via
  platforms like **Cursor** and **Windsurf**.
tags:
  - untagged
companies: []
models: []
topics:
  - untagged
---


<!-- buttondown-editor-mode: plaintext -->**a quiet day**

> AI News for 3/27/2025-3/28/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**230** channels, and **13422** messages) for you. Estimated reading time saved (at 200wpm): **1217 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We soft launched [the 2025 State of AI Engineering survey](https://www.surveymonkey.com/r/57QJSF2) today, fill it out to join our $1000 Amazon gift card raffle + have your voice heard in the state of AI Eng!


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

Here's a summary of the tweets, organized by topic:

**GPT-4o Model Performance and Features**

- **GPT-4o's improved coding and instruction following were praised**: [@sama](https://twitter.com/sama/status/1905419197120680193) highlighted the **new version of GPT-4o** for being **particularly good at coding, instruction following, and freedom**. [@kevinweil](https://twitter.com/kevinweil/status/1905419868071231993) agreed, stating the **GPT-4o update is strong and encouraged users to try it**.
- **GPT-4o's performance relative to other models, particularly in coding and reasoning, was assessed**: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1905563427776651344) reported that **GPT-4o (March 2025) is now the leading non-reasoning coding model**, surpassing **DeepSeek V3** and **Claude 3.7 Sonnet** in the **Artificial Analysis Coding Index**, and is **#1 in LiveCodeBench**. However, it **still lags behind reasoning models** like o3-mini.
- **Concerns about policy compliance**: [@joannejang](https://twitter.com/joannejang/status/1905681602619085042) noted that **image generation refusals are often due to the model hallucinating policies**. They asked users to **bear with them as they try to get the model to follow the policy** and suggested **trying again in a new chat if encountering issues**.
- [@nrehiew_](https://twitter.com/nrehiew_/status/1905414817034150362) hypothesized that **4o image generation works by embedding the image directly via an encoder, using AR, and then diffusing out based on the ARed hidden states**; the **blur is a psyop** and there's **no VQ**.
- **GPT-4o's transparency and background generation feature were highlighted**: [@giffmana](https://twitter.com/giffmana/status/1905407013103747422) noted the ability to ask **GPT-4o image gen for transparent backgrounds**, calling it a cool feature drowned out by Ghiblification hype.

**Gemini 2.5 Pro Model Performance and Capabilities**

- **Gemini 2.5 Pro was lauded for its capabilities in audio and video understanding**: [@_philschmid](https://twitter.com/_philschmid/status/1905566076781371642) reported that **Gemini 2.5 Pro** has **improved long context capabilities** and can **process ~1h long video with a single request**, noting the **integration of YouTube links into AIS and API**. The model can also handle **~2 hours of podcast transcription in a single request**.
- **Simple-Bench AI Explanation Performance**: [@scaling01](https://twitter.com/scaling01/status/1905729393756180985) mentioned **Gemini 2.5 Pro Thinking scored around 51.6% on AI Explained' Simple-Bench**, the **first model to score above 50%**.
- **Accessibility and Usage**: [@_philschmid](https://twitter.com/_philschmid/status/1905555766179684587) announced that users can **bring their own API Key to** [@cursor_ai](https://twitter.com/cursor_ai) to use **Gemini 2.5 Pro**, but noted that **rate limits are currently low**. They also mentioned that **Gemini 2.5 Pro is available in** [@windsurf_ai](https://twitter.com/windsurf_ai).

**AI Infrastructure and Compute**

- **GPU usage is expected to increase significantly**: [@saranormous](https://twitter.com/saranormous/status/1905451945713909790) stated that **they are going to use all the GPUs (and TPUs)**.
- **Together AI and Hypertec Group are partnering to deliver large-scale GPU clusters**: [@togethercompute](https://twitter.com/togethercompute/status/1905632314878800044) announced a partnership with [@HypertecGroup](https://twitter.com/HypertecGroup) to deliver **clusters of thousands of GPUs**, emphasizing **high-bandwidth networking, advanced cooling, and robust fault tolerance**.
- **CoreWeave's IPO**: [@weights_biases](https://twitter.com/weights_biases/status/1905641235395547203) congratulated [@CoreWeave](https://twitter.com/CoreWeave) on their IPO, highlighting their success in pushing the edge of what’s possible in AI infrastructure.

**AI Engineering and Development**

- **Concerns regarding conventional programming languages over vibe coding**: [@lateinteraction](https://twitter.com/lateinteraction/status/1905447832099983564) emphasized the importance of retaining useful aspects of conventional programming languages, such as **defining functions, control flow, and modules**, rather than giving in to "vibe coding".
- **Importance of open-source in medical AI**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1905582817276494155) highlighted the **crucial role of open-source in medical AI** due to the **need for transparency and the impracticality of sending sensitive patient data to cloud APIs**.
- **Emphasizing scalable solutions for ASI**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1905460812057108658) pointed out a statement about building **scalable solutions to ASI**, focusing on improvements with **more resources on computation and data**.
- **Langchain and Redis Integration**: [@LangChainAI](https://twitter.com/LangChainAI/status/1905691477906522473) announced that with `langgraph-checkpoint-redis`, you can bring [@Redisinc](https://twitter.com/Redisinc)'s powerful memory capabilities to your LangGraph agents.

**Company and Product Announcements**

- **New homepage for Keras**: [@fchollet](https://twitter.com/fchollet/status/1905391839055950032) announced the **launch of a brand new homepage** for Keras to celebrate its 10th anniversary.
- **C.H. Robinson saves time with LangGraph**: [@LangChainAI](https://twitter.com/LangChainAI/status/1905667121465774379) reported that **C.H. Robinson** is **saving 600+ hours a day** using tech built with **LangGraph, LangGraph Studio, and LangSmith** to **automate routine email transactions**.
- **Launch of the MIT NLP Group account**: [@lateinteraction](https://twitter.com/lateinteraction/status/1905411805343875276) announced the launch of the [@nlp_mit](https://twitter.com/nlp_mit) account to showcase the latest NLP research from MIT labs.
- **Perplexity AI Thread Infrastructure Issues**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1905652310501675318) mentioned that Perplexity AI is going through some infra challenges, which is why past threads are not loading.

**Humor/Memes**

- **Various humorous tweets**: Several users shared humorous content, including [@Teknium1](https://twitter.com/Teknium1/status/1905677763228713225) posting **"Jensen rn"** with an image, [@teortaxesTex](https://twitter.com/teortaxesTex/status/1905448971411013792) with **Xi after he dies in WWIII and is reincarnated as a shota in a parallel world**, [@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1905445347562062030) suggesting that **if you generate yourself as the opposite sex in chatgpt and think it’s mid, you should probably lower your standards**, and [@_philschmid](https://twitter.com/_philschmid/status/1905560675495129324) noting that [@cursor_ai](https://twitter.com/cursor_ai) just rick rolled them.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Reverse Engineering GPT-4o: Architectural Insights and Speculations**

- **Reverse engineering GPT-4o image gen via Network tab - here's what I found** ([Score: 599, Comments: 43](https://reddit.com/r/LocalLLaMA/comments/1jlptqu/reverse_engineering_gpt4o_image_gen_via_network/)): The author investigates the image generation process of **GPT-4o** by examining network traffic, uncovering that the backend returns intermediate images that suggest a possible multi-step pipeline. They speculate whether the model uses a diffusion process or an autoregressive approach, noting that the **OpenAI model card** describes it as an autoregressive model. The author references the **OmniGen paper** as a potential explanation for GPT-4o's capabilities, highlighting its use of a transformer-based architecture that scales well with high-quality data and computational power.
  - There is debate over whether the **GPT-4o** model uses a **diffusion model** or an **autoregressive model**. Some commenters speculate it might employ a hierarchical decoder with a diffusion model for pixel-level detail, while others suggest it uses an autoregressive approach that enhances image generation by predicting sequences of tokens in a sophisticated manner.
  - The potential for **open-source competitors** to match the quality of GPT-4o is discussed, with some expecting that Chinese competitors might achieve this within a year. However, others believe it could take until the end of **2025** for open-source models to catch up, emphasizing the importance of an open-source image model akin to **LLaMA** for LLMs.
  - Commenters express skepticism about the value of individual reverse engineering efforts, noting that the broader academic and industrial communities, especially in China, are likely conducting extensive analyses. There is interest in whether the model's ability to access the internet and utilize high-quality data provides significant advantages over local text encoders like **CLIP**/**T5**.


**Theme 2. MegaTTS3's Voice Cloning: Skepticism and Security Concerns**

- **[New TTS model from bytedance](https://github.com/bytedance/MegaTTS3)** ([Score: 143, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1jlw5hb/new_tts_model_from_bytedance/)): **ByteDance** released **MegaTTS3**, a new text-to-speech model, which has sparked controversy over its voice cloning capabilities. The discussion centers around ethical implications and potential misuse of this technology in creating unauthorized voice replicas.
  - **MegaTTS3's Features and Limitations**: The model boasts **lightweight efficiency** with 0.45B parameters, **bilingual support**, and **controllable accent intensity**. However, the **WaveVAE encoder** is not available for local voice cloning due to "security issues", sparking criticism about the misleading advertising of "Ultra High-Quality Voice Cloning".
  - **Ethical and Security Concerns**: There is skepticism about the "security reasons" for not releasing the voice cloning software, as many believe this is a guise for **data collection** to improve their models. Critics argue this approach contradicts ethical considerations, given the widespread availability of AI voice cloning technologies.
  - **Community Reactions and Criticism**: Users express frustration over the misleading promotion of voice cloning capabilities and question the ethics of **data submission** for training purposes. Some see the "safety" claims as a strategy for indirect monetization by collecting user data for further training.


**Theme 3. Qwen-2.5-72b: Leading the Open-Source OCR Revolution**

- **[Qwen-2.5-72b is now the best open source OCR model](https://getomni.ai/blog/benchmarking-open-source-models-for-ocr)** ([Score: 119, Comments: 14](https://reddit.com/r/LocalLLaMA/comments/1jm4agx/qwen2572b_is_now_the_best_open_source_ocr_model/)): **Qwen 2.5 VL (72b and 32b)** models have emerged as the leading open-source OCR models, achieving approximately **75% accuracy** in JSON extraction, comparable to **GPT-4o**. The **72b model** slightly outperformed the **32b model** by **0.4%**, while both surpassed the **mistral-ocr** model's **72.2%** accuracy. Surprisingly, **Gemma-3 (27B)** scored only **42.9%**, despite its architecture being based on the high-performing **Gemini 2.0**. The benchmarking data and methodology are available on [GitHub](https://github.com/getomni-ai/benchmark) and [Hugging Face](https://huggingface.co/datasets/getomni-ai/ocr-benchmark).
  - **Ovis2 Models** have not been included in the discussion, despite being leaders on OCRBench with significantly fewer parameters (18x less), suggesting potential interest in their performance relative to Qwen models.
  - There's curiosity about the performance of the **olmOCR-7B-0225-preview** model from [Hugging Face](https://huggingface.co/allenai/olmOCR-7B-0225-preview), noted for being more VRAM efficient, highlighting a demand for models that balance performance with resource usage.
  - The **Qwen 2.5 VL 32B model** has been updated and shows significant performance improvements over the older 72B model, which has not received recent updates. The 32B model is also noted for its superior writing capabilities compared to the vanilla Qwen model.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

> our pipelines are down...

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking

**Theme 1.  GPT-4o Dominates Leaderboards and Sparks Debate**

- [**GPT-4o Jumps to Arena #2, Coding Prowess Confirmed**](https://fxtwitter.com/lmarena_ai/status/1905340075225043057): The latest **ChatGPT-4o (2025-03-26)** model surged to **#2** on the Arena leaderboard, surpassing **GPT-4.5** and tying for **#1** in Coding and Hard Prompts. Users note a significant performance leap and a **10x** cost reduction compared to previous models, though pricing discrepancies with API snapshots cause confusion.
- [**GPT-4o's Coding Skills Draw Mixed Reviews Despite Benchmarks**](https://x.com/ArtificialAnlys/status/1905563427776651344?t=Ade7EDjFb3DDumNIqnvwtw&s=19): While benchmarks position **Gemini 2.5 Pro** as the leading non-reasoning model, some users find **GPT-4o** superior for coding tasks, particularly in instruction following and code generation.  Debate continues about whether **GPT-4o's** high ranking is due to specialized training for preferred response styles rather than raw performance.
- [**GPT-4o Unveiled as Autoregressive Image Model**](https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf):  **GPT-4o** is confirmed to employ an **autoregressive** approach for image generation, marking a novel method for creating images directly from text prompts. Speculation arises about the model reusing **image input and image output tokens** for efficiency.

**Theme 2.  DeepSeek V3 and Qwen2.5-Omni Emerge as Strong Contenders**

- [**DeepSeek V3 Outcodes GPT-4o on SWE-bench**](https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/): The new **DeepSeek V3 0324** model is gaining recognition for coding prowess, reportedly outperforming **GPT-4o R1** on the SWE-bench benchmark. Data indicates **DeepSeek V3** surpasses **Claude 3.7 Sonnet** in non-reasoning coding tasks, becoming a leading model in the field.
- [**Qwen2.5-Omni: Meta's Multimodal Marvel Arrives**](https://qwenlm.github.io/blog/qwen2.5-omni/): **Qwen2.5-Omni**, the latest flagship model in the **Qwen** series, is released as an end-to-end multimodal model handling text, images, audio, and video with real-time streaming responses.  Users can test **Qwen2.5-Omni** at [Qwen Chat](https://chat.qwenlm.ai), marking a significant step towards truly versatile AI models.
- [**DeepSeek Blends Diffusion and Transformers, Following GPT-4o's Lead**](https://fxtwitter.com/DeanHu11/status/1903983295626707271): **DeepSeek** is adopting a multimodal architecture similar to **GPT-4o**, combining **diffusion and transformers**. This approach, previously seen in vision models, signals a growing trend in multimodal AI development.

**Theme 3.  Infrastructure Woes and User Frustrations Plague AI Platforms**

- [**Perplexity AI Buckles Under Server Strain, Users Report Outages and Data Loss**](https://status.perplexity.com/): **Perplexity AI** experiences widespread **outages**, with users reporting disappearing history and spaces. The official status page ([status.perplexity.com](https://status.perplexity.com/)) is slow to update, prompting calls for better outage communication and automated reporting systems.
- [**Manus.im Credit System Triggers User Backlash Over High Costs**](https://manus.im/help/credits): **Manus.im's** new credit system faces heavy criticism for its perceived high cost, with some users estimating monthly expenses could reach **$500**. The shift from a task-based to credit-based system is described as *jarring*, impacting user experience.
- [**Cursor IDE Suffers Database Disaster, Service-Wide Outage Ensues**](https://status.cursor.com/): **Cursor** experiences a service-wide outage due to a **database deployment issue**, disrupting core AI features and general service functionality.  While resolved after a few hours, the incident highlights the fragility of AI-powered coding tools and their reliance on robust infrastructure.

**Theme 4.  Tools and Techniques for Enhanced AI Development Emerge**

- [**LM Studio 0.3.14 Unleashes Granular Multi-GPU Control**](https://lmstudio.ai/download): **LM Studio 0.3.14** introduces advanced controls for multi-GPU setups, allowing users to fine-tune GPU allocation strategies and manage resources more effectively. New keyboard shortcuts (`Ctrl+Shift+H` or `Cmd+Shift+H`) provide quick access to GPU settings.
- [**Aider's New `/context` Command Automates Codebase Context Management**](https://discord.com/channels/1131200896827654144/1354944747004887162/1354945152287899809): **Aider** introduces the `/context` command, which automatically identifies and adds relevant files to the chat based on user requests. This feature streamlines context management, especially in large codebases, saving developers time and effort.
- [**DSPy Framework Promotes Declarative Programming Over Brittle Prompting**](https://dspy.ai/): **DSPy** is highlighted as a framework for *programming* language models rather than relying on traditional prompting. It enables rapid iteration on modular AI systems using Python code and algorithms to optimize prompts and model weights, aiming for more robust and high-quality AI outputs.

**Theme 5.  Ethical Considerations and AI Safety Remain Central**

- [**OpenAI Relaxes Image Generation Policy, Prioritizes Real-World Harm Prevention**](https://x.com/joannejang/status/1905341734563053979): **OpenAI** shifts its image generation policy in **ChatGPT 4o**, moving from blanket refusals to a more nuanced approach focused on preventing real-world harm. This policy change allows for greater creative freedom in previously restricted areas.
- [**AI Safety Discussions Highlight Constitutional AI and Jailbreak Concerns**](https://generalanalysis.com/blog/jailbreak_cookbook): Discussions on AI safety emphasize that models like **Claude**, designed with constitutional AI principles, prioritize objectivity over user preferences, potentially impacting leaderboard rankings. Resources like the [Jailbreak Cookbook](https://generalanalysis.com/blog/jailbreak_cookbook) are shared, addressing LLM vulnerabilities and safety measures.
- [**Miyazaki's 9-Year-Old Critique of AI Art Resurfaces, Sparks Ethics Debate**](https://x.com/nuberodesign/status/1904954270119588033): A resurfaced clip of Hayao Miyazaki criticizing AI-generated art reignites ethical discussions within the AI community. The debate draws parallels between AI art sampling and fast fashion ethics, questioning the morality of readily accessible, potentially exploitative content.

---

# PART 1: High level Discord summaries




## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Rage Against Manus New Credit System**: Users are frustrated with the new credit system, some estimating costs could reach **$500/month** for decent usage and the **1000 free credits** are quickly consumed even if the task fails, details at [manus.im/help/credits](https://manus.im/help/credits).
   - The community noted *the shift from task-based to credits-based does feel jarring, especially when it wasn’t part of the original beta flow*.
- **Manus Farm Brainstorming Alternative Energy**: One member suggested that **Manus** could develop cheap renewable energy sources, such as molten sodium, thermal or solar to power their own GPU farm and reduce costs, potentially locating it in a desert.
   - The member proposed flywheels as energy storage to keep the farm running at night for max efficiency.
- **Manus Considers Cheaper AI Models Like Deepseek**: The community is in discussion around using cheaper AI models like **Deepseek** and **Qwen** instead of only **Anthropic's Claude** to reduce operational costs.
   - It has not been stated if **Manus** will allow other AI integrations.
- **Students Cheat with Manus AI on Exams**: Students have used **Manus** alongside **Kimi** or **Deepseek** to upload seminar and lecture files, asking the AI to memorize them for exam preparation, some receiving scores such as **81/100** on assignments.
   - Some users were wondering if that violates the terms of service if you help the AI cheat for school.
- **UI Design Hailed as Simple Genius**: Multiple members praised the UI design of **Manus**, expressing that the design is really good, easy to use, simple and aligns with real world concepts.
   - One user stated *What made manus feel so amazing was not only the results you got, but that the idea of tasks closely aligned with real world concepts. That simplicity was genius*.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI Servers Under Siege**: Multiple users reported **outages and disappearing history/spaces**, prompting humor and frustration, and the official status page ([status.perplexity.com](https://status.perplexity.com/)) lacked timely updates.
   - Users suggested an automated user-reported outage system and proactive notifications to address the infrastructure challenges mentioned in [this tweet](https://x.com/AravSrinivas/status/1905652310501675318).
- **DeepSeek AI Falls Flat**: Members voiced **disappointment with DeepSeek AI**, citing its struggles with complex instructions and tendency to produce *unnecessary jargon*.
   - Comparisons were made to superior math applications, highlighting **DeepSeek AI's** shortcomings in practical problem-solving.
- **Claude AI's Context Window Gets the Side Eye**: Discussion arose around the **context window limit of Claude AI** relative to Gemini and ChatGPT, with many members noting Claude's limitations.
   - Members agreed that **Claude's context window** was particularly restrictive in comparison to its competitors, especially Gemini.
- **Free Perplexity Pro Via T-Mobile**: Users exchanged methods for acquiring **free Perplexity Pro subscriptions** through **T-Mobile** and Revolut promotions.
   - One user even suggested utilizing a burner number on **T-Mobile** to take advantage of the offer, and another user linked to a [tweet about Perplexity shipping voice dictation](https://x.com/testingcatalog/status/1905390832225493429?s=61).
- **Sonar API has Llama Index RAG Integration Issues**: A user inquired about effectively passing **Llama Index RAG** context to the **Perplexity Sonar** model, seeking suggestions on leveraging the index object.
   - The user also questioned whether the **Deep Research** functionality in the API would achieve parity with the perplexity.com version, noting a perceived performance gap, and mentioned that the **Sonar API** sometimes misses citations.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek 3.1 Sneaks into Cursor**: A Cursor team member mentioned that **DeepSeek 3.1** should be integrated into the editor within 12 hours, but pricing details remain undisclosed.
   - Cursor offers [deals with providers](https://cursor.com/deals) and a **privacy mode** ensuring no data storage.
- **Cursor Plunges Amidst Database Disaster**: Cursor experienced a service-wide outage due to a **database deployment issue** within its infrastructure, disrupting AI features like Chat and Tab as well as general service.
   - After a few hours, the issue was resolved and they updated the [Cursor Status](https://status.cursor.com/).
- **Humanoid Hype Heats Up**: Members debated the utility of humanoid robots, with contrasting visions of them as *food-making and cleaning* assistants versus concerns over [data privacy](https://en.wikipedia.org/wiki/Data_privacy) and telemetry.
   - A member posited that **AGI will emerge from robotics**, developing first in a virtual environment before manifesting in the real world.
- **Codebase Tag Cruises into the Sunset**: Users noticed the removal of the **@Codebase tag** and staff clarified it was replaced with a similar way to scan the current indexed project, as noted in the [changelog](https://cursor.com/changelog).
   - This sparked discussions about **token limits**, pricing models, and balancing convenience with control in AI coding tools.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O1 Pro Coming to Leaderboard?**: Members discussed the potential inclusion of **O1 Pro** on the leaderboard, speculating that **OpenAI** might cover costs to showcase its capabilities given its high price.
   - However, some members expressed doubts about its leaderboard performance and latency.
- **GPT-4o's Coding Skills Under Debate**: Members debate [GPT-4o's coding ability](https://x.com/ArtificialAnlys/status/1905563427776651344?t=Ade7EDjFb3DDumNIqnvwtw&s=19) after recent updates, with some noting improvements in instruction following and code generation.
   - However, proper evals are needed, as one member argued that **GPT-4o's** ranking may be inflated due to specialized training for preferred response styles, rather than actual performance.
- **DeepSeek V3 leapfrogs Coding Benchmarks**: The new **DeepSeek V3 0324** model is gaining recognition, with one member noting it scores higher than **GPT-4o R1** in **SWE-bench** according to [this Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/).
   - Data indicates that DeepSeek's **V3 0324** release leapfrogs **Claude 3.7 Sonnet** in non-reasoning and has become the leading non-reasoning model for coding.
- **Meta's Llama Models getting Quirky**: Members observed that recent anonymous models in the arena, believed to be from **Meta**, are displaying quirky behavior, including adding many emojis and identifying themselves as **Meta Llama** models.
   - Models being tested include: `bolide`, `cybele`, `ginger`, `nutmeg`, `phoebe`, `spider`, `themis`, though they also note that `spider` sometimes identifies itself as GPT-4.
- **AI Safety Discussions**: Members discussed AI safety, mentioning that models like **Claude** are designed with constitutional AI principles, prioritizing objectivity over user preferences, which may affect their leaderboard rankings.
   - A member also shared a [Jailbreak Cookbook](https://generalanalysis.com/blog/jailbreak_cookbook) resource for LLM jailbreaks and AI safety, including a [GitHub repository](https://github.com/General-Analysis/GA) with implementations of systematic jailbreaks.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Scribe V1 Powers FoxMoans!**: A member uses **11Labs Scribe V1** for audio event classification, to create a [list of utterances](https://github.com/zero2rizz/FoxMoans/blob/main/UtteranceList.txt), estimating a cost of **$20k**.
   - It is used for audio event classification, suited for projects needing mood-based analysis.
- **OlmOCR's Unsloth Integration Still Rocky**: A member struggles to load **OlmOCR** (a finetune of **Qwen2VL**) in Unsloth, despite having **Qwen2VL** working.
   - The Unsloth team asked if the user tried the latest version, as they pushed updates and fixes *before the creator realized their models finished uploading*.
- **Orpheus TTS Gets Fine-Tuning**: The Unsloth team released a [notebook for finetuning **Orpheus-TTS**](https://x.com/UnslothAI/status/1905312969879421435), highlighting its human-like speech with emotional cues.
   - Members discussed changing **Orpheus** language, suggesting continued pretraining with new embedded/head layers might be sufficient.
- **Double Trouble for BOS Token**: A user found a **double BOS token** issue in the latest **Unsloth update (Gemma 3 4B)** when checking tokenizer decoding.
   - A hotfix was [identified](https://github.com/unslothai/unsloth-zoo/pull/106) which removed the accidentally added token.
- **DeepSeek-R1 Goes Quantized**: Unsloth made available various versions of **DeepSeek-R1**, including [GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF?show_file_info=DeepSeek-R1-Q4_K_M%2FDeepSeek-R1-Q4_K_M-00001-of-00009.gguf) and [4-bit formats](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5).
   - Unsloth's **DeepSeek-R1** 1.58-bit + 2-bit Dynamic Quants selectively quantized improving accuracy over standard 1-bit/2-bit.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o vs Gemini 2.5: Coding Showdown**: Members compared **GPT-4o** and **Gemini 2.5 Pro** for coding, with some finding *GPT-4o* superior despite benchmarks showing that Gemini 2.5 Pro performs better overall, with **GPT-4o** winning 3 out of 6 categories.
   - Opinions varied, with some favoring *Gemini* for specific tasks like C++ and WinAPI integration.
- **Google AI Studio: The New Free Tier Hero**: Users are praising **Google AI Studio** for its free access to models like **Gemini 2.5 Pro** and generous prompt limits, which are more than paid services like **ChatGPT Plus**.
   - Some members reported using hundreds of messages daily without hitting limits and even canceled their *ChatGPT* subscriptions because of these advantages.
- **Perplexity Dominates News over ChatGPT**: Members found **Perplexity** excels in news and current events due to its Discover tab, highlighting it as more than just a *GPT wrapper*.
   - However, some noted issues with **Perplexity's** *Deep Research* feature for quality and reliability on uploaded files, suggesting *ChatGPT* instead.
- **Claude 3.7 Sonnet's Reasoning Prowess**: Members lauded **Claude 3.7 Sonnet** for its superior reasoning capabilities and explanations compared to other AI models, especially since *free tier Claude fills up and forces you to start a new chat*.
   - Alternative models like o1, o3-mini-high, and Grok 3 were recommended for coding, with o1 favored for complex tasks using C++, Physics, Rendering and older APIs like Win32API.
- **Enhanced Image Prompting: A New Dawn?**: Users raved about the new **ChatGPT** image tool's improved adherence to complex prompts, like generating a moving market on a giant turtle's back with a sun and three moons.
   - The updated tool excels at targeted image modifications, such as removing stars from a night scene without affecting the entire image.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Pro: Users Hit Rate Limit Wall**: Users are bumping into [low rate limits for **Gemini 2.5 Pro**](https://x.com/OpenRouterAI/status/1905300582505624022), even after integrating their own **AI Studio API keys**, leading to discussions on maximizing free quota.
   - One member remarked the model *won't be free forever* which will be a problem when they inevitably have to start charging.
- **OpenRouter AI SDK Provider Options Confuse Debuggers**: Members are actively debugging [**OpenRouter AI SDK** provider options](https://github.com/OpenRouterTeam/ai-sdk-provider), specifically using `providerOptions` for model order and fallback behavior.
   - The core issue revolves around the correct way to nest the **order array** under the **provider** key, as debugging attempts reveal unexpected provider selection despite the configurations.
- **Function Calling Gold Rush in Free LLMs**: Members are on the hunt for free models that support function calling, with **Mistral Small 3.1** and **Gemini free models** emerging as top contenders.
   - One frustrated member exclaimed, *Gosh, I'm trying so hard to find a free model that supports function calling. I can't find any!*.
- **Gemini Flash 2.0 Burns Rubber in TPS Showdown**: The community is hotly debating the **tokens per second (TPS)** performance of various coding models, with **Gemini Flash 2.0** being touted for its blazing speed.
   - Despite the hype, some users are critical, pointing out it is *trash* because *their hosting is messed up*, and one member touted that **Groq** serves the **70B R1 distil at 600tok/s**, another one chimed in that it *isn't good at coding imo*.
- **OpenAI Responses API Support?**: A member inquired about [**OpenRouter** supporting the **OpenAI Responses API**](https://platform.openai.com/docs/api-reference/responses).
   - The OpenRouter team suggested the [**Veo2 API**](https://openai.com/blog/veo) is your best bet for **SOTA image to video**, but it's about **50 cents per second of video**.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Prompt ICL for Best Tool Use**: Members discussed prompting agents for **tool usage**, referencing [Cline's system prompt](https://github.com/cline/cline/blob/main/src/core/prompts/system.ts) and suggesting prompts on the server directly such as `First call ${tool1.name}, then ${tool2.name}`.
   - A member shared a [link on using prompts for ICL](https://x.com/llmindsetuk/status/1899148877787246888?t=WcqjUT4wCCHd_qj-QPf7yQ&s=19) and a [test showing it working](https://github.com/evalstate/fast-agent/blob/main/tests%2Fe2e%2Fprompts-resources%2Ftest_prompts.py#L75-L92).
- **Google Search Gets Config for MCP**: A member inquired about adding **Google Search** to MCP, and another member shared their [configuration](https://cdn.discordapp.com/attachments/1312302100125843479/1355126409093316750/config.json?ex=67e7cb50&is=67e679d0&hm=8311f31b3b6181eb391876bad03fc45f745e439a12180e6ad087d94983c37c1c&).
   - They noted that users need to obtain their own **Google API key** and **engine ID** to use the configuration.
- **MCP Servers Galore with Docker**: A member created an all-in-one **Docker Compose** setup for easily self-hosting **17 MCP servers** using Portainer, with Dockerfiles sourced from public GitHub projects ([MCP-Mealprep](https://github.com/JoshuaRL/MCP-Mealprep)).
   - It was recommended to *not bind the containers on 0.0.0.0 unless you need this accessible remotely* and to *include in the readme an example mcp config json*.
- **Agents are saying Canvas Yeah!**: A member created a **Canvas MCP** server, enabling AI agents to interact with Canvas LMS, and added an agent that can autonomously crawl Gradescope to find info, available at [Canvas-MCP](https://git.new/canvas-mcp).
   - The tool offers features like finding relevant resources, querying upcoming assignments, and accessing courses and assignments from **Gradescope**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **GPT-4o Claims Coding Arena**: The latest **ChatGPT-4o** update jumps to #2 on the [Arena leaderboard](https://x.com/lmarena_ai/status/1905340075225043057), tying #1 in Coding, Hard Prompts, and performing in the Top-2 across ALL categories while costing 10x less.
   - This update is confusingly released as **chatgpt-4o-latest** endpoint, priced at $5/$15 per million input/output tokens, whereas the API snapshots are priced at $2.5/$10, so caution is recommended when moving workloads, according to [Artificial Analysis](https://x.com/ArtificialAnlys/status/1905563427776651344).
- **OpenRouter R1 Model Stumbles**: A member found the free **R1** model on OpenRouter to be *"stupid"*, verbose, and ineffective at solving broken tests, especially with repomap enabled, unlike **O3-mini**.
   - It's speculated that the free **R1** model is a quantized version of **DeepSeek**, possibly in FP8 format, while the DeepSeek on the leaderboard is from the official DeepSeek team and users rotating through multiple API keys on OpenRouter may have their accounts suspended.
- **Context Architecture Enables Efficient Codebase Handling**: Constant Context Architecture (**CCA**) is proposed as a solution for working with large codebases using LLMs, guaranteeing that the necessary context for modifying any module will always fit within an LLM's context window, regardless of the total codebase size, as described in this [blogpost](https://neohalcyon.substack.com/p/constant-context-architecture).
   - This is achieved by ensuring modules have bounded size, interfaces, and dependencies, making context gathering a bounded operation.
- **Rate Limits Frustrate Gemini 2.5 Pro Users**: Multiple users reported hitting rate limits with **Gemini 2.5 Pro**, even when seemingly below the documented **50 requests/day**, with one noting the existence of a **2 requests/minute limit**.
   - There was discussion on whether purchasing a paid account would resolve the limitations, with mixed results reported, along with a potential fallback model implementation.
- **Aider's Context Command Automates File Inclusion**: The new `/context` command automatically identifies relevant files for a given request and adds them to the chat, as discussed in [this discord thread](https://discord.com/channels/1131200896827654144/1354944747004887162/1354945152287899809).
   - It's particularly useful for large codebases and saves time by automating the process of manually adding files.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT-4o Leaps to #2 on Arena!**: The latest **ChatGPT-4o** (2025-03-26) jumped to **#2** on Arena, surpassing **GPT-4.5** with a significant improvement (+30 pts) over the January version, according to [this tweet](https://fxtwitter.com/lmarena_ai/status/1905340075225043057).
   - It tied for **#1** in Coding and Hard Prompts.
- **OpenAI Loosens Image Generation Policy**: OpenAI launched native image generation in **ChatGPT** via **4o**, shifting from blanket refusals to a more precise approach focused on preventing real-world harm, as explained in [this blog post](https://x.com/joannejang/status/1905341734563053979).
   - The new policy allows more creative freedom in sensitive areas.
- **Devin Autogenerates Wiki Pages**: **Devin** now automatically indexes repos and produces wikis with architecture diagrams and links to sources, according to [this tweet](https://x.com/cognition_labs/status/1905385526364176542?s=46&t=jDrfS5vZD4MFwckU5E8f5Q).
   - This functionality helps users get up to speed on unfamiliar parts of a codebase.
- **HubSpot Co-Founder Joins Latent Space**: **Dharmesh Shah**, co-founder of **HubSpot** and creator of **Agent.ai**, joined Latent Space to discuss the next evolution in workplace organization, with a focus on **hybrid teams**.
   - A key concept is the idea of *human workers collaborating with AI agents as team members*, raising questions about team dynamics, trust, and task delegation.
- **LLM Codegen Workflow Detailed**: A member shared their [LLM codegen workflow](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/), emphasizing **brainstorming specs**, planning, and executing with LLM codegen in discrete loops.
   - The workflow is built on personal experience and internet best practices, but the author admits that *it will probably not work in 2 weeks, or it will work twice as well*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Tames Multi-GPU Setups**: **LM Studio 0.3.14** introduces granular controls for multi-GPU setups, enabling users to enable/disable specific GPUs and choose allocation strategies such as **evenly** or **priority order**, downloadable [here](https://lmstudio.ai/download).
   - Keyboard shortcuts `Ctrl+Shift+H` (Windows) or `Cmd+Shift+H` (Mac) give quick access to GPU controls, with `Ctrl+Alt+Shift+H` (Windows) or `Cmd+Option+Shift+H` (Mac) opening a pop-out window for managing settings during model loading.
- **Threadripper Flexes on EPYC**: A discussion compared **Threadripper** to **EPYC**, clarifying that while **Threadripper** is technically HEDT (High-End Desktop), AMD does not promote **EPYC** for home users.
   - A [GamersNexus review](https://gamersnexus.net/cpus/amds-cheap-threadripper-hedt-cpu-7960x-24-core-cpu-review-benchmarks) highlighted the **AMD Ryzen Threadripper 7960X's** 24 cores and relatively low cost for workstations.
- **LLM Calculations Get a Visual Overhaul**: Members discussed visualizing calculations performed by LLMs, such as mapping values to pixel colors and the [LLM Visualization tool](https://bbycroft.net/llm) was recommended.
   - Resources such as 3b1b's playlist on LLMs and a book on building LLMs from scratch were shared for deeper understanding.
- **P100 Gets Demolished by 6750xt**: A member inquired about using a **P100 16GB** for a hobby project, but was strongly advised against it, with one user saying its basically *e-waste* compared to a **6750xt**.
   - The **6750xt** was recommended as a better and more modern card due to its **Vulkan** support, while the **P100's** unsupported **CUDA** versions make it less desirable.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Transformer Storage Error Messages Confuse Users**: Insufficient storage leads to misleading error messages in **transformers v4.50.0**, a user found; a PR is planned for better error handling and checking for capacity before downloading model shards.
   - The user had to use `df -h` to diagnose the **100% full** system due to bad error messaging from the library.
- **Torchtune Invites Code Tinkering for Customization**: Users found that **torchtune** needs downloading and editing **200-line PyTorch scripts** and YAML files to customize, giving a complete view of the process.
   - The need to dissect **Hugging Face's implementations** may be avoided by this approach, according to a user.
- **Bias-Augmented Consistency Training Validates Introspection**: Members discussed emulating self-awareness in LMs by creating a representation of their circuits and feeding it back, inspired by [Anthropic's work](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).
   - A [paper](https://arxiv.org/abs/2403.05518v1) on **bias-augmented consistency training (BCT)** was also linked as a validation measure for introspection methods.
- **Adaptive Compression Aims to Boost Distributed Systems**: An infrastructure layer optimizing model transmission and deployment across distributed systems is in development, using adaptive compression and intelligent routing to tackle **bandwidth waste** and **inference latency**.
   - Those interested in **distributed inference** may find this infrastructure useful for scaling larger models, offering a demo.
- **Neural Nets Morph Into Bodies Without Organs**: A member linked to a [tweet](https://x.com/norabelrose/status/1905336894038454396) arguing that neural networks are **Bodies Without Organs (BwO)** because they *don't have organs* or *fixed mechanisms* and instead have *flows of information*.
   - A member rejects **mechanistic interpretability** and says neural networks generalize without fixed mechanisms which was seen by Descartes 400 years ago.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **`tl.gather` Glides Closer to Release**: While waiting for official release, to solve element repetition problems, members noted that one can compile **Triton** from source as described in [this discord thread](https://discord.com/channels/1189498204333543425/1189607595451895918/1336735886884208672).
   - The team also clarified that *tl.gather could solve element repetition problems*, which has been requested by other members for functions such as `torch.Tensor.expand()` to triton.
- **Activation Sparsity Accelerates FFNs**: A new paper was shared arguing that **2:4 sparsity** for activation acceleration in LLMs leads to **1.3x faster FFNs** without accuracy loss, see [Acceleration Through Activation Sparsity](https://arxiv.org/abs/2503.16672).
   - A member noted the next step is `FP4 with sparsity for an effective 2-bit tensorcore performance`.
- **Confusion Clouds CUDA Profiling**: A user seeks a definitive guide to **CUDA profiling**, given the plethora of Nvidia tools such as **nvprof**, **Nvidia Visual Profiler (nvvp)**, and various **Nsight** packages.
   - Another user suggested **Nsight Compute** is the best tool for single kernel profiling, with links to [Nvidia's documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) and [a detailed talk](https://www.youtube.com/watch?v=F_BazucyCMw&t=5824s).
- **Miyazaki Mocks AI Art Sampling**: A **9-year-old meme** resurfaced showing [Hayao Miyazaki's critical reaction](https://x.com/nuberodesign/status/1904954270119588033) to AI-generated art when presented by a founder of Niconico.
   - Members compared the ethics of using AI art to buying from fast fashion companies like **Shein**, citing an immoral business model offers access to cheaper content.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI Schools Envisioned by OpenAI and xAI**: **OpenAI** and **xAI** are exploring the concept of AI-driven schools, potentially leveraging generated images for lesson content, with discussion pinpointing *Ghibli Studio Style* as a solution for alignment as per [this post](https://x.com/TheDevilOps/status/1905297966400770155).
   - The initiatives aim to integrate AI more intimately into educational frameworks, with a focus on creating visually appealing and contextually relevant learning materials.
- **Transformer Circuits Unveils Crosscoders**: The Transformer Circuits team released an update on **sparse crosscoders**, a variation of sparse autoencoders that read and write to multiple layers, forming shared features as outlined in their [research update](https://transformer-circuits.pub/2024/crosscoders/index.html).
   - These **crosscoders** address cross-layer superposition, monitor persistent features, and simplify circuits.
- **GPT-4o Confirmed as Auto-Regressive Image Model**: Members verified **GPT-4o** as an **autoregressive image generation model** after **Yampeleg's** [post](https://x.com/Yampeleg/status/1905293247108219086) and the release of [OpenAI's System Card](https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf).
   - This revelation highlights the model's novel approach to image creation directly from textual prompts, with members conjecturing that **GPT-4o** reuses **image input and image output tokens**.
- **Qwen2.5-Omni Makes a Multimodal Splash**: **Qwen2.5-Omni**, the latest flagship **end-to-end multimodal model** in the **Qwen** series, has been shared among members, and it is designed for comprehensive multimodal perception and handles text, images, audio, and video, as detailed on the [Qwen Chat](https://chat.qwenlm.ai).
   - Offering real-time streaming responses via both text generation and natural speech synthesis, **Qwen2.5-Omni** sets a new benchmark in multimodal interaction.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GPT-4o Surges on Arena, 10x Cheaper**: The new **ChatGPT-4o (2025-03-26)** model jumped to #2 on Arena, surpassing **GPT-4.5**, with reported **10x** cost reduction and it tied for **#1** in Coding and Hard Prompts, as reported by [lmarena_ai](https://fxtwitter.com/lmarena_ai/status/1905340075225043057).
   - The model is currently ranked in the **Top-2** across all categories in Arena and excels in both coding and handling complex prompts.
- **Musk's xAI Swallows X in $80B Deal**: **Elon Musk** revealed that **xAI** has taken over **X** through an all-stock transaction, valuing **xAI at $80 billion** and **X at $33 billion**, including $12 billion in debt, according to [The Verge](https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition).
   - This move consolidates Musk's AI ventures under the **xAI** umbrella and may shift the competitive landscape in the AI market.
- **LlamaGen Generates Images Like LLMs**: The **LlamaGen** family of image generation models applies the *next-token prediction* paradigm from large language models to generate images, achieving **2.18 FID** on ImageNet 256x256 benchmarks as described in the [LlamaGen paper](https://arxiv.org/abs/2406.06525).
   - The architecture achieves a reconstruction quality of **0.94 rFID** and **97%** codebook usage with an image tokenizer that has a downsample ratio of **16**.
- **Qwen2.5-Omni Does It All**: The **Qwen2.5-Omni** is the new flagship end-to-end multimodal model in the Qwen series, capable of processing text, images, audio, and video, with real-time streaming responses via text and speech as noted in [their blogpost](https://qwenlm.github.io/blog/qwen2.5-omni/).
   - The model is available for use at [Qwen Chat](https://chat.qwenlm.ai) and may herald a new wave of more generalized models.
- **Gemini 2.5 Pro Crushes Wordle Competition**: **Gemini 2.5 Pro** has demonstrated exceptional performance on Wordle, logically deducing words and letter placements, as reported by [Xeophon](https://x.com/TheXeophon/status/1905535830694773003).
   - Feedback on **Gemini 2.5 Pro** has been overwhelmingly positive, with one user noting that *I think I've never seen feedback this robustly positive about an AI release that wasn't the Current Thing*, as mentioned by [Zvi](https://x.com/TheZvi/status/1905003873422442642).



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **FP8 QAT Faces Bandwidth Bottleneck**: A member following up on [issue #1632](https://github.com/pytorch/ao/issues/1632) noted **FP8 QAT** is on *TorchAO's* radar, but lacks bandwidth for immediate implementation.
   - This indicates a potential area for future development and contribution within the **PyTorch** ecosystem.
- **Torchtune's Team Tackles Issue Backlog**: The team discussed prioritizing **PR reviews** and **new PRs** before addressing the issue backlog, estimating **80%** of existing issues are already resolved.
   - To better organize the backlog of pending reviews, a member suggested a general **RL/RLHF tracker**, in addition to the existing GRPO tracker.
- **Torchtune Plans Integration with bitsandbytes**: A member suggested using [issue #906](https://github.com/pytorch/torchtune/issues/906) in the **Torchtune** repo to guide contributions for **bitsandbytes** integration.
   - Another member humorously noted their lack of enthusiasm for doc PRs, but agreed to check it out nonetheless.
- **Centered Reward Loss enables Reward Model Training**: Members discussed enabling reward model training in **Torchtune**, specifically focusing on implementing **centered reward loss** like **(R1 + R2)² loss**.
   - They noted the current **preference dataset** format requires a **chosen/rejected format without a prompt**.
- **vLLM Integration Causes Weight Hotswapping Hacks**: A member detailed memory monopolization issues during initialization with **vLLM**, sharing an *obscure hack* for [weight hotswapping](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html#vllm.LLM.sleep).
   - Another member warned that *every vLLM release breaks something*, alluding to potential incompatibilities with existing hacks when vLLM releases version **0.8** with its new **v1 execution engine**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Claude Gets a Kingly UI**: Users are reporting a clean new UI for **Claude**, with one user specifically liking that the UI hides all the things they never use, calling it a *king move*.
   - The only noted issue so far is the lack of a toggle for **extended think**.
- **DeepSeek Copies GPT-4o's Homework**: **DeepSeek** is combining **diffusion and transformers** like GPT-4o multimodal, as noted in [this tweet](https://fxtwitter.com/DeanHu11/status/1903983295626707271) referencing a similar idea in vision.
   - The cited paper experiments on images and videos using autoregressive conditional block attention.
- **TinyZero's $30 AI Model Debuts**: Attention is turning to **U.S. TinyZero's** recent accomplishments, specifically their **$30 model**, along with new releases like **VERL** and **Sky-T1**, as covered in [this CNBC article](https://www.cnbc.com/2025/03/27/as-big-tech-bubble-fears-grow-the-30-diy-ai-boom-is-just-starting.html).
   - When DeepSeek released its R1 claiming it had achieved its generative AI large language model for just $6 million, the billions being spent by U.S. AI market leaders including Microsoft-funded OpenAI immediately came under scrutiny.
- **LG's EXAONE Models Released Under Questionable License**: **LG AI Research** has released **EXAONE Deep**, a series of models ranging from **2.4B to 32B parameters**, with superior capabilities in reasoning tasks including math and coding benchmarks, as detailed in their [documentation](https://arxiv.org/abs/2503.12524), [blog](https://www.lgresearch.ai/news/view?seq=543) and [GitHub](https://github.com/LG-AI-EXAONE/EXAONE-Deep).
   - It was noted that the **EXAONE AI Model License Agreement 1.1 - NC** explicitly retains ownership of the output, but the enforcement of this license is questionable.
- **Hermes-3 Impresses Users**: A member mentioned that so far the most impressive model has been **Hermes3 Llama3.2 3B**.
   - No further details were given.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek Plunges Into Diffusion-Transformer Mix**: **DeepSeek** combines **diffusion and transformers** like **GPT-4o** multimodal, according to [this tweet](https://fxtwitter.com/DeanHu11/status/1903983295626707271) linking to their paper.
   - The author noted that [a similar idea appeared in Vision](https://arxiv.org/abs/2412.07720), experimenting on images and videos with almost the same title.
- **ZeroGPU Quota Bugging Users**: Users are reporting issues with **zeroGPU quota** not resetting, with one linking to [this discussion](https://discord.com/channels/879548962464493619/1355122724820746374) for related complaints.
   - One user noted that even if the quota is used up, it *recovers to a certain extent after 30 minutes or an hour*, but it's buggy.
- **FactoryManager Rolls Out LinuxServer.io Docker Support**: A member introduced [FactoryManager](https://github.com/sampagon/factorymanager), a **Python package** wrapping **linuxserver.io desktop environment containers**, enabling programmatic control of environments, showcased with a demo using two different desktop environments.
   - This package aims to offer flexibility by scaffolding on top of **linuxserver.io**, diverging from the custom environments often created in GUI agent demos from **Anthropic**, **OpenAI**, etc.
- **Langfuse Toxicity Evaluator Flags the Carrots**: A user testing the toxicity LLM-as-a-judge in Langfuse found that it incorrectly flagged the prompt *'Can eating carrots improve your vision?'* as toxic with a score of **0.9**, citing a false association with climate change discourse.
   - The user questioned *how to evaluate the evaluator*, noting that **GPT-4o** misattributed derogatory climate change content to a harmless question about carrots.
- **Base vs Instruct Model Debate**: A newcomer to agents sought clarification on the distinction between base models and instruct models, referencing the course's mention of chat templates.
   - A member responded with a metaphor of a **base model** as *'the naked model, without a wrap'* and shared [a Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1c1sy03/an_explanation_of_base_models_are/) further elaborating on the differences.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Mindmapping Feature Wins Fans**: A user expressed excitement about the new mindmapping feature, calling it *another mind-blowing moment*.
   - No further details were provided about their specific uses.
- **Source Uploads Snag, Stuck in Limbo**: A user reported issues with sources stuck in a perpetual uploading state, preventing both import and removal, for over 8 hours.
   - The user sought advice on removing permanently uploading sources but without success.
- **Versioning Vanishes, Users Vexed**: A user expressed concern over the lack of versioning and recycle bin support for the "Note" source type.
   - The user mentioned hesitancy to use it, preferring Google Docs for its superior data protection and backup features.
- **Pasted Sources Stop Self-Naming**: A user reported that pasted sources, which previously named themselves automatically, now default to "pasted text."
   - The user asked if there was an update or a way to revert to the previous behavior.
- **PDF Parsing Problems Persist**: Users discussed NLM's inability to extract data from scanned PDFs, with one user asking if the tool could extract data from scanned notes.
   - A user clarified that **NLM cannot handle mixed content PDFs** (text and images), but can process docs and slides.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Celebrates MCP Week**: LlamaIndex highlights **LlamaCloud** as an **MCP server** and demonstrates the use of **LlamaIndex** as a client to any **MCP server**, offering access to many MCP servers as tools, detailed in [this tweet](https://twitter.com/llama_index/status/1905678572297318564).
   - They showcased the ability to substantially *expand capabilities* for agents by utilizing hundreds of existing MCP servers.
- **FunctionAgent Gains ChatMessage History**: A member inquired about adding chat history to the **FunctionAgent** workflow, with [documentation provided](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/#adding-chat-history).
   - Guidance was offered on overriding chat history with `agent.run(...., chat_history=chat_history)` or using `ChatMemoryBuffer.from_defaults(token_limit=60000, chat_history=chat_history)`.
- **Telemetry Tracking Gets User ID**: A member asked about passing custom telemetry attributes and attaching a header or param to the LLM network call when interacting with Llama Index, and a [Colab notebook](https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing) was shared.
   - The Colab notebook shows how to attach a user ID to all events executed within a code block.
- **LlamaParse PDF Parsing Problem**: A user reported that **LlamaParse** works for single PDFs but fails when processing two PDFs and asking the same question, potentially causing a system overload.
   - The user described that the system *literally cooked* when handling multiple PDFs, indicating a potential overload or processing error.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere names Models "Command"**: A member questioned why **Cohere** chose to name its language models *Command* suggesting, similar to database management, a *query* is essentially a **command or instruction**.
   - Model selection is available in **Coral**, with *Just Chat* utilizing **Command A** without external sources.
- **Software Engineer seeks Cohere Career**: A member is seeking new job opportunities as a software engineer and is excited to discuss potential projects related to **websites** or **web applications**.
   - Another member shared a link to the [Cohere careers page](https://cohere.com/careers) encouraging the user to explore available positions.
- **Bot Commands Get Test Run**: Members are encouraged to test bot commands in the 「🤖」bot-cmd channel to ensure proper functionality and user experience.
   - Feedback on bot commands is welcome.
- **Full-Stack Alchemist Ready to Build**: A passionate developer with **8+ years** of experience is skilled in building scalable **web and mobile apps** using modern frameworks like **React, Angular, Flutter, and Swift**.
   - They craft intelligent **AI solutions** using **Python, TensorFlow, and OpenAI**, integrating **cloud technologies (AWS, GCP, Azure)** and **microservices** for global scaling.
- **Oracle Consultant Seeks Cohere Wisdom**: A technical consultant with **12+ years** of experience in **Oracle ERP Fusion** is eager to learn more about **Cohere models** and **AI use cases** for enterprise applications.
   - A networking and CS student is aiming to work on **open-source generative music** projects, favoring tech tools like **ChatGPT, Grok, Windsurf, and Replit**.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All Faces Usability Complaints**: Users express concerns about **GPT4All's** usability, mentioning issues such as inability to import models, search the model list, view model sizes, use LaTeX, or customize model list order.
   - One user suggests **GPT4All** *is losing users because other platforms are more user-friendly and open*.
- **GPT4All Lagging on New Model Implementation**: A user is frustrated that **GPT4All** has yet to implement **Mistral Small 3.1** and **Gemma 3**, highlighting their multimodal capabilities.
   - The user suggests that if **GPT4All** does not catch up by Summer 2025, they might switch away from *Llama.cpp*.
- **GPT4All Praised for Native RAG and Model Settings**: Despite criticisms, **GPT4All** offers advantages such as **native RAG** and out-of-the-box functionality, with a user expressing confidence in the developers and anticipation for **GPT4All v4.0.0**.
   - Another user appreciates **GPT4All's** model settings page for its comprehensive options and convenient model reload button noting that *you need 2-3 clicks to setup out of the chat menu*.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Members Asked to Close Stale PRs and Issues**: George Hotz asked members to close any open pull requests (PRs) and issues that are stale.
   - This request aims to clean up the project's repository by addressing outdated items.
- **Discussions on TinyGrad Codegen Internals**: A member inquired about **TinyGrad's code generation** process, specifically asking about the location of `CStyleCodegen` or `CUDACodegen` as mentioned in the documentation.
   - The documentation describes **TinyGrad** using different *translators* (Renderers or Codegen classes) such as `C++ (CStyleCodegen)`, `NVIDIA GPUs (CUDACodegen)`, `Apple GPUs (MetalCodegen)` to translate the optimized plan into code that the CPU/GPU can understand.
- **Boolean Indexing Implementation Explored**: A member sought advice on efficiently creating evenly spaced points on a grid with a hole in it, similar to boolean indexing in PyTorch, suggesting this could be a useful contribution to **TinyGrad**.
   - An LLM proposed a solution using **masked_select** to efficiently create the desired grid with a hole, leveraging the condition `full.abs().max(axis=1) >= (math.pi/6)` to filter points outside the hole.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Tackling DSPy Output Validation Fails**: A member inquired about how **DSPy** handles output validation failures, specifically when an integer field expects a number from 1 to 10 but receives **101**.
   - There was no further discussion or links provided regarding this question in the channel.
- **Delving into DSPy Optimizers**: A member is exploring the use of **optimizers** within **DSPy** and how they interact with docstrings and prompt management, referencing [DSPy's official documentation](https://dspy.ai/).
   - The issue found is that the **Optimizer overwrites the prompt from the docstring**, requiring optimized versions to be loaded from a json or pkl file.
- **Decoding DSPy's Optimization Process**: It was clarified that **DSPy's optimizer** generates prompts and tests them on a dataset to identify the best-performing one, further detailed on the [official website](https://dspy.ai/).
   - The user found it *VERY interesting* how the optimizer may select **N examples** to include in the prompt, showcasing the kind of prompts generated.
- **DSPy: Declarative Self-improving Python Emerges**: **DSPy** is a framework for *programming rather than prompting* language models to rapidly iterate on **building modular AI systems**, offering algorithms to **optimize prompts and weights**.
   - Instead of brittle prompts, you write compositional _Python code_ and use DSPy to **teach your LM to deliver high-quality outputs**.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Mentorship MIA for Entrepreneurship Track**: An entrepreneurship track student inquired about mentorship opportunities within the **LLM Agents Berkeley MOOC**.
   - It was clarified that *Berkeley does not provide any mentorship* for the entrepreneurship track, though sponsors will host office hours in Apr/May.
- **Sponsor Office Hours Announced**: Sponsors will be hosting office hours in April/May for the **LLM Agents Berkeley MOOC** entrepreneurship track.
   - This provides an opportunity for students to engage with industry professionals and seek guidance on their projects.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Gemini 2.5 Pro Surfs into Windsurf**: **Gemini 2.5 Pro** is now available in Windsurf, granting users **1.0** user prompt credits on every message and **1.0** flow action credits on each tool call; see [the announcement on X](https://x.com/windsurf_ai/status/1905410812921217272).
   - The update aims to enhance user experience with the latest model.
- **Windsurf Wipes Out on Gemini 2.5 Pro Rate Limits**: Shortly after the release of **Gemini 2.5 Pro**, Windsurf encountered rate limits due to massive load for the model and provider.
   - The team is working to increase quota and apologized for any inconvenience, aiming to get everyone *surfing on Gemini 2.5 Pro ASAP*.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Foo[1] Defaults to Predefined Value**: The `self` parameter in the context of the `Foo[1]` type can be automatically populated with a default parameter value.
   - When `self` is discarded using `_`, the argument defaults to its predefined default value.
- **Self Parameter Clarification**: The `self` parameter is `Foo[1]` with a default parameter value, which can be disregarded with `_`.
   - Disregarding `self` with `_` defaults to the predefined default parameter value.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1354892866832564237)** (627 messages🔥🔥🔥): 

> `Manus new Credit system Feedback, Alternative Energy for Manus GPU Farm, Cheaper AI Models like Deepseek and Qwen, Manus AI assistance for Exams, Manus UI Love` 


- **Community Outcry on Manus New Credit System**: Many users expressed frustration with the new credit system, feeling it's too expensive and limiting, with some estimating costs could reach **$500/month** for decent usage, others blew through their **1000 free credits** quickly and the credits are consumed even if the task fails.
   - Users like that they are active to help but *the shift from task-based to credits-based does feel jarring, especially when it wasn’t part of the original beta flow*.
- **Brainstorming Alternative Energy for Manus GPU Farm**: One member suggested that Manus could create a team dedicated to developing cheap renewable energy sources, such as molten sodium, thermal or solar to power their own GPU farm and reduce costs.
   - They mentioned locating it in a desert and using flywheels as energy storage to keep it running at night for max efficiency.
- **Alternative AI Models Like Deepseek Considered**: There was discussion around using cheaper AI models like **Deepseek** and **Qwen** instead of only **Anthropic's Claude** to reduce operational costs.
   - However it has not been stated if **Manus** will allow other AI integrations
- **Manus AI assistance for Exams**: Some users have used **Manus** alongside Kimi or Deepseek to upload seminar and lecture files, asking the AI to memorize them for exam preparation, this helped some receive scores such as **81/100** on assignments.
   - Some were wondering if that violates the terms of service if you help the AI cheat for school.
- **Love the Manus UI Design**: Multiple members praised the UI design of Manus, expressing that the design is really good, easy to use, simple and aligns with real world concepts.
   - One user stated *What made manus feel so amazing was not only the results you got, but that the idea of tasks closely aligned with real world concepts. That simplicity was genius*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/a/R5vY585">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://en.wikipedia.org">Wikipedia, the free encyclopedia</a>: no description found</li><li><a href="https://apps.apple.com/us/app/munas/id6742685315">‎Munas</a>: ‎1. Product IntroductionThis app integrates various advanced AI technologies, offering features like multi-modal dialogue, deep reasoning, intelligent creation, and image generation. Whether in the wo...</li><li><a href="https://tenor.com/view/mad-annoyed-gif-27497951">Mad Annoyed GIF - Mad Annoyed - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/evil-cat-floppy-herobrine-angry-cat-glowing-eyes-gif-13772161273485327421">Evil Cat Floppy GIF - Evil cat Floppy Herobrine - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.investopedia.com/terms/m/mentalaccounting.asp">Mental Accounting: Definition, Avoiding Bias, and Example</a>: Mental accounting refers to the different values a person places on the same amount of money, based on subjective criteria, often with detrimental results.</li><li><a href="https://tenor.com/view/thats-my-opinion-have-yours-shrug-sassy-pretty-gif-16292035">Thats My Opinion Have Yours GIF - Thats My Opinion Have Yours Shrug - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://manus.im/help/credits">Manus</a>: Manus is a general AI agent that turns your thoughts into actions. It excels at various tasks in work and life, getting everything done while you rest.</li><li><a href="https://kvlcogit.manus.space/">Supply Chain Sustainability Intelligence Platform</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Item_response_theory">Item response theory - Wikipedia</a>: no description found</li><li><a href="https://www.aeaweb.org/articles?id=10.1257/jel.20201593">The Microeconomics of Cryptocurrencies - American Economic Association</a>: no description found</li><li><a href="https://github.com/browser-use/browser-use">GitHub - browser-use/browser-use: Make websites accessible for AI agents</a>: Make websites accessible for AI agents. Contribute to browser-use/browser-use development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1354894027660722350)** (1219 messages🔥🔥🔥): 

> `Perplexity AI outages, DeepSeek AI, Claude AI, User Frustrations, T-Mobile Promo` 


- **Perplexity AI Servers Caving**: Multiple users reported **outages and disappearing history/spaces** ([example](https://discord.com/channels/1047197230748151888/1164597981840941076/1355201859186462781)), prompting humor and frustration among the community.
   - The official status page ([status.perplexity.com](https://status.perplexity.com/)) lacked timely updates, with users suggesting an automated user-reported outage system and proactive notifications.
- **DeepSeek AI not that deep**: Members were generally **disappointed in DeepSeek AI** and it being able to understand complex instructions.
   - They also commented on it giving *unnecessary jargon*, compared to better math apps.
- **Claude AI**: A member asked about the **context window limit of Claude AI**, in comparison with Gemini and ChatGPT.
   - Many members commented that Claude's context window was very limited, especially in comparison to Gemini.
- **Users vent frustrations**: Users lamented **lost notes and study materials** due to the outages, with some jokingly blaming AI for ruining their exams.
   - There was a discussion on the broader economic implications of Perplexity's **$20/month subscription fee**, especially when considering minimum wage workers in different countries.
- **T-Mobile Users Cashin**: Users shared methods for obtaining **free Perplexity Pro subscriptions via T-Mobile** and Revolut promos.
   - One member suggested using a burner number on T-Mobile to gain access.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1905390832225493429?s=61">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨:  @perplexity_ai is shipping voice dictation along with a bunch of other updates. Voice dictation uses OpenAI but doesn&#39;t seem to be working yet.</li><li><a href="https://x.com/AravSrinivas/status/1905652310501675318">Tweet from Aravind Srinivas (@AravSrinivas)</a>: We’re going through some infra challenges this week, which is why your library of past threads or discover content might not load for you right now. I apologize for the inconvenience to everyone. We’r...</li><li><a href="https://x.com/AravSrinivas/status/1905652310501675318?t=0AABseZKpQZ57cG3Aw94wQ&s=19">Tweet from Aravind Srinivas (@AravSrinivas)</a>: We’re going through some infra challenges this week, which is why your library of past threads or discover content might not load for you right now. I apologize for the inconvenience to everyone. We’r...</li><li><a href="https://www.theatlantic.com/national/archive/2013/06/what-does-american-actually-mean/276999/">What Does &#x27;American&#x27; Actually Mean?</a>: In Latin America, &quot;American&quot; means anyone from the American continent. U.S. citizens claiming the word are considered gauche or imperialist. So what&#x27;s the solution?</li><li><a href="https://tenor.com/view/president-stops-president-smiles-ominous-smile-ominous-no-answer-gif-78197234359887585">President Stops President Smiles GIF - President stops President smiles Ominous smile - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity Status</li><li><a href="https://tenor.com/view/megan-thee-stallion-shock-snl-excuse-me-rude-gif-12376496298828106239">Megan Thee Stallion Shock GIF - Megan thee stallion Shock Snl - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/april-fools-joke-dog-its-fine-this-is-not-gif-1750056094467610487">April Fools Joke GIF - April Fools Joke Dog - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/putin-stare-gif-14318699512326580302">Putin Stare GIF - Putin Stare - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/">YouTube</a>: no description found</li><li><a href="https://tenor.com/view/spongebob-spongebob-meme-thinking-sad-coffee-gif-21807366">Spongebob Spongebob Meme GIF - Spongebob Spongebob Meme Thinking - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jm2ekd/comment/mk97f47/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://tenor.com/view/log-crane-tricks-gif-17344198">Log Crane GIF - Log Crane Tricks - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jm2ekd/message_from_aravind_cofounder_and_ceo_of/">Reddit - The heart of the internet</a>: no description found</li><li><a href="https://www.livescience.com/space/black-holes/is-our-universe-trapped-inside-a-black-hole-this-james-webb-space-telescope-discovery-might-blow-your-mind">Is our universe trapped inside a black hole? This James Webb Space Telescope discovery might blow your mind</a>: no description found</li><li><a href="https://youtu.be/4UKM_yvTexI?si=IChRCpK43in_IznM"> - YouTube</a>: no description found</li><li><a href="https://payequity.gov.on.ca/docs/7-12proportional-value-comparison-method/">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=e1C75U_7e6U"> - YouTube</a>: no description found</li><li><a href="https://math.libretexts.org/Bookshelves/PreAlgebra/Pre-Algebra_II_(Illustrative_Mathematics_-_Grade_8)/03:_Linear_Relationships/3.00:_New_Page/3.1.4:_Comparing_Proportional_Relationships">3.1.4: Comparing Proportional Relationships</a>: no description found</li><li><a href="https://math.stackexchange.com/questions/67280/salary-calculation-solving-proportional-increase-of-two-variables">Salary calculation: solving proportional increase of two variables</a>: I would like to understand, proportionally, how much of someone&#x27;s new salary is attributed to an increase in both their number of hours worked, and increased dolar wage respectively.&#xA;Both the...</li><li><a href="https://byjus.com/maths/ratios-and-proportion/">Ratio and Proportion - Definition, Formulas and Examples</a>: Ratio and proportion are the mathematical expressions to compare the quantities. Visit BYJU’S to learn the ratio and proportion definitions, formulas and examples.</li><li><a href="https://www.investopedia.com/ask/answers/042415/what-are-differences-between-regressive-proportional-and-progressive-taxes.asp">Regressive vs. Proportional vs. Progressive Taxes: What&#39;s the Difference?</a>: The U.S. uses three types of tax systems: regressive, proportional, and progressive. Two impact high-and low-income earners differently and one is the same for all.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1354898417910022258)** (10 messages🔥): 

> `Shareable threads, Super Prompt, LLM Research` 


- ****Shareable Threads** Required**: Perplexity AI requested a member to ensure their thread is *`Shareable`*, linking to a previous message in the Discord about this topic [here](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- ****Super Prompt** Incoming**: A member shared a link to a Perplexity AI search result for *`create a super prompt for copi`* [here](https://www.perplexity.ai/search/create-a-super-prompt-for-copi-xD7hRu9bQKeeWTuv9l6lfg).
- ****LLM Research** collection shared**: A member shared a link to a Perplexity AI collection about *`LLM research`* [here](https://www.perplexity.ai/collections/llm-research-TIWFlUA7SWGuIqfbmRzenQ).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1354910680624664606)** (7 messages): 

> `API Parameter Error Handling, Llama Index RAG context with Perplexity Sonar, Deep Research Parity API vs Web` 


- **API Parameters Now Throw Errors**: The API team implemented error handling for parameters like **search_domain_filter**, **related_questions**, **images**, and **structured_outputs** for non–Tier 3 users.
   - If you previously achieved the desired results by passing your JSON schema within the prompt (instead of using the parameter), you’ll continue to see the correct behavior; *nothing has fundamentally changed*.
- **Sonar model struggles with Llama Index RAG context**: A user asked for suggestions on how to pass **Llama Index RAG** context using the index object to **Perplexity Sonar** model.
   - The user also asked if the **Deep Research** in the API will get closer to the one on perplexity.com, as it seems *nerfed* compared to the website.
- **Sonar API misses citations**: A user reported an instance where the API with sonar model did not return any citations.
   - The user noted that the same query on the client-side experience comes with citations.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1354893464340267089)** (1251 messages🔥🔥🔥): 

> `Gemini 2.5 Pro Pricing, Cursor infrastructure, Humanoid robots?, Codebase tag removed from cursor` 


- **Dan drops deep deets on DeepSeek 3.1 in Cursor**: A Cursor team member shared that **DeepSeek 3.1** should be available in the editor within 12 hours, but did not disclose the cost.
   - There are also [deals with providers](https://cursor.com/deals) and privacy mode ensures no data is stored, so that is nice.
- **Cursor Gets Crashed by Database Deployment Debacle**: Cursor experienced a service-wide outage due to a **database deployment issue** within its infrastructure, affecting AI features like Chat and Tab.
   - The incident was resolved after a few hours, and a team member quipped that they *accidentally unplugged the main server to charge my phone*.
- **Hot Takes on Humanoid Robots**: Members debated the utility of humanoid robots, with some envisioning them as *food-making and cleaning* assistants, while others expressed concerns over [data privacy](https://en.wikipedia.org/wiki/Data_privacy) and telemetry.
   - Another member suggested that robotics will be where **AGI emerges from**, evolving in a virtual followed by real environment.
- **Members mourn missing @codebase tag on Cursor**: Users noticed that the [@Codebase tag was removed](https://cursor.com/changelog), and the staff explained that it was replaced by a similar way to scan current indexed project.
   - This prompted discussion about **token limits**, pricing models, and the trade-offs between convenience and control when using AI coding tools.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cursor.com/settings/models#large-context-and-pricing">Cursor – Models</a>: no description found</li><li><a href="https://x.com/nicdunz/status/1905353949865238633?s=46">Tweet from nic (@nicdunz)</a>: 4o takes first place for coding</li><li><a href="https://x.com/artificialanlys/status/1905563427776651344?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: Today’s GPT-4o update is actually big - it leapfrogs Claude 3.7 Sonnet (non-reasoning) and Gemini 2.0 Flash in our Intelligence Index and is now the leading non-reasoning model for codingThis makes GP...</li><li><a href="https://docs.cursor.com/troubleshooting/request-reporting">Cursor – Getting a Request ID</a>: no description found</li><li><a href="https://tenor.com/view/correct-futurama-the-best-kind-of-correct-yes-yep-gif-5787390">Correct Futurama GIF - Correct Futurama The Best Kind Of Correct - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/readingdancer/status/1829267522777919904">Tweet from Chris Houston (@readingdancer)</a>: I wonder why @OpenAI &#39;s #chatgpt4 doesn&#39;t like this prompt?&#34; Can you create an image of A gold fish wearing a cowboy hat riding on a piglet? &#34;</li><li><a href="https://x.com/ArtificialAnlys/status/1905563427776651344">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: Today’s GPT-4o update is actually big - it leapfrogs Claude 3.7 Sonnet (non-reasoning) and Gemini 2.0 Flash in our Intelligence Index and is now the leading non-reasoning model for codingThis makes GP...</li><li><a href="https://tenor.com/view/drake-hotline-bling-dance-dancing-gif-17654506">Drake Hotline Bling GIF - Drake Hotline Bling Dance - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/when-the-coding-when-the-coding-when-the-coding-is-when-the-meme-gif-21749595">When The Coding Coding GIF - When The Coding When The Coding - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/OpenAIDevs/status/1905335104211185999">Tweet from OpenAI Developers (@OpenAIDevs)</a>: `chatgpt-4o-latest` is now updated in the API, but stay tuned—we plan to bring these improvements to a dated model in the API in the coming weeks.Quoting OpenAI (@OpenAI) GPT-4o got an another update ...</li><li><a href="https://x.com/threejs/status/1905647468551053370">Tweet from Three.js (@threejs)</a>: Three.js r175 released 🗿https://threejs.org/changelog/?r175</li><li><a href="https://status.cursor.com/">Cursor Status</a>: no description found</li><li><a href="https://docs.cursor.com/settings/models#context-window-sizes">Cursor – Models</a>: no description found</li><li><a href="https://forum.cursor.com/t/connection-failed-if-the-problem-persists-please-check-your-internet-connection-or-vpn-or-email-us-at-hi-cursor-sh/17334/51">Connection Failed If the problem persists, please check your internet connection or VPN, or email us at hi@cursor.sh</a>: Connection failed. If the problem persists, please check your internet connection or VPN  (Request ID: 3004f856-6920-443f-</li><li><a href="https://en.wikipedia.org/wiki/Artificial_general_intelligence">Artificial general intelligence - Wikipedia</a>: no description found</li><li><a href="https://generalanalysis.com/blog/jailbreak_cookbook">The Jailbreak Cookbook - General Analysis</a>: no description found</li><li><a href="https://cimwashere.com">Cim Was Here</a>: Photography. Stockton, CA.</li><li><a href="https://codeium.com/windsurf">Windsurf Editor by Codeium</a>: Tomorrow&#x27;s editor, today. Windsurf Editor is the first AI agent-powered IDE that keeps developers in the flow. Available today on Mac, Windows, and Linux.</li><li><a href="https://github.com/end-4/dots-hyprland">GitHub - end-4/dots-hyprland: i hate minimalism so...</a>: i hate minimalism so... Contribute to end-4/dots-hyprland development by creating an account on GitHub.</li><li><a href="https://github.com/danperks/CursorStatus">GitHub - danperks/CursorStatus</a>: Contribute to danperks/CursorStatus development by creating an account on GitHub.</li><li><a href="https://codeium.com/profile/docker">Cim (@docker) Profile | Codeium</a>: Cim (@docker) has completed 4,096 using Codeium&#x27;s AI autocomplete. Codeium offers best in class AI code completion &amp; chat — all for free.
</li>
</ul>

</div>
  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1354893069488492754)** (906 messages🔥🔥🔥): 

> `O1 Pro drop, GPT-4o latest Benchmarks, Deepseek V3, Meta LLama, AI Safety` 


- **O1 Pro on Leaderboard, Coming Soon?**: Members discussed the potential inclusion of **O1 Pro** on the leaderboard, speculating that **OpenAI** might absorb the costs to showcase its capabilities, particularly given its high price point.
   - However, some doubted its ability to rank highly, with one commenter joking it *would be ranked dead last lol* due to latency.
- **GPT-4o's Coding Ability**: Members debate [GPT-4o's coding ability](https://x.com/ArtificialAnlys/status/1905563427776651344?t=Ade7EDjFb3DDumNIqnvwtw&s=19) after recent updates, with some noticing improvements in instruction following and code generation.
   - However, others insist that proper evals are needed, as [one member argued](https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/) that **GPT-4o's** ranking may be inflated due to specialized training for preferred response styles, rather than actual performance.
- **DeepSeek V3's Rise in Coding Benchmarks**: The new **DeepSeek V3 0324** model is gaining recognition, with one member noting it scores higher than **GPT-4o R1** in **SWE-bench** according to [this Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/).
   - Data indicates that DeepSeek's **V3 0324** release leapfrogs **Claude 3.7 Sonnet** in non-reasoning and has become the leading non-reasoning model for coding.
- **Meta's Emoji-Laden LLama Models**: Members observed that recent anonymous models in the arena, believed to be from **Meta**, are displaying quirky behavior, including the addition of many emojis and a tendency to identify themselves as **Meta Llama** models, though their image recognition capabilities are notably inferior.
   - The models being tested include: `bolide`, `cybele`, `ginger`, `nutmeg`, `phoebe`, `spider`, `themis`, though they also note that  `spider`  sometimes is identified as being GPT-4.
- **Discussions on AI Safety and Jailbreaking**: Members discussed AI safety, mentioning that models like **Claude** are designed with constitutional AI principles, prioritizing objectivity over user preferences, which may affect their leaderboard rankings.
   - A member also shared a [Jailbreak Cookbook](https://generalanalysis.com/blog/jailbreak_cookbook) resource for LLM jailbreaks and AI safety, including a [GitHub repository](https://github.com/General-Analysis/GA) with implementations of systematic jailbreaks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/flavioad/status/1905347584438251848?s=46">Tweet from Flavio Adamo (@flavioAd)</a>: OpenAI just updated GPT-4oI tested the old vs new and the difference is actually wildQuoting OpenAI (@OpenAI) GPT-4o got an another update in ChatGPT!What&#39;s different?- Better at following detaile...</li><li><a href="https://x.com/patloeber/status/1905333725698666913">Tweet from Patrick Loeber (@patloeber)</a>: 🏆Gemini 2.5 Pro is currently- #1 on LMArena- #1 on Livebench- #1 across SEAL leaderboardsAlso starts becoming the top choice for coding tasks :)  Our teams are working hard on getting everyone higher...</li><li><a href="https://x.com/ArtificialAnlys/status/1905563427776651344?t=Ade7EDjFb3DDumNIqnvwtw&s=19">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: Today’s GPT-4o update is actually big - it leapfrogs Claude 3.7 Sonnet (non-reasoning) and Gemini 2.0 Flash in our Intelligence Index and is now the leading non-reasoning model for codingThis makes GP...</li><li><a href="https://tenor.com/view/tin-foil-hat-jabrils-foil-hat-put-on-a-hat-wearing-a-hat-gif-18223681">Tin Foil Hat Jabrils GIF - Tin Foil Hat Jabrils Foil Hat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://en.wikipedia.org/wiki/Conversion_to_Judaism">Conversion to Judaism - Wikipedia</a>: no description found</li><li><a href="https://generalanalysis.com/blog/jailbreak_cookbook">The Jailbreak Cookbook - General Analysis</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jjusya/deepseek_v3_0324_got_388_swebench_verified_w/">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1354893051541328013)** (580 messages🔥🔥🔥): 

> `Elevenlabs Scribe V1 for audio event classification, OlmOCR loading in Unsloth, Fine-tuning LLMs for board games, Gemma 3 notebook quirks, Qwen Omni Hacking` 


- **Scribe V1 powers FoxMoans Utterance List**: A member is using **11Labs Scribe V1** for audio event classification to create a [list of utterances](https://github.com/zero2rizz/FoxMoans/blob/main/UtteranceList.txt), which they estimate will cost them around **$20k**.
   - They mentioned using it for audio event classification, suggesting it's ideal for projects needing mood-based analysis, since it can detect *laughter, angry yandere style* variations.
- **OlmOCR's Unsloth Integration Remains Rocky**: A member is struggling to load **OlmOCR** (a finetune of **Qwen2VL**) in Unsloth, despite having **Qwen2VL** working.
   - The Unsloth team followed up by asking if the user had tried the latest version, with a member noting that they've been pushing updates and fixes *before the creator realized their models finished uploading*.
- **Orpheus TTS Receives Fine-Tuning Notebook, Multilingual Support Explored**: The Unsloth team released a [notebook for finetuning **Orpheus-TTS**](https://x.com/UnslothAI/status/1905312969879421435), highlighting its human-like speech with emotional cues.
   - Members also discussed the possibility of changing the language of **Orpheus** from English to another language, suggesting that continued pretraining with new embedded/head layers may be sufficient.
- **LocalLlama Mods face fire over removing posts**: A user complained about their post explaining the **Llama Community License** being removed from r/LocalLLama, leading to speculation that Meta/Facebook might be moderating the subreddit.
   - Other members didn't care, noting that it's proprietary, and that Meta isn't really enforcing it, with one member saying *I care once i see a c&d or actually any enforcement*.
- **Multi-GPU Support Coming Soon, Pro not needed**: A user asked about obtaining Unsloth Pro for multi-GPU support, but a member replied that **multi-GPU** support will likely become *free* under AGPL in the coming weeks.
   - One user said *preliminary stuff is already done at least for the first version, a few weeks*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/running-and-saving-models/troubleshooting#saving-to-safetensors-not-bin-format-in-colab">Troubleshooting | Unsloth Documentation</a>: If you&#x27;re experiencing issues when running or saving your model.</li><li><a href="https://x.com/UnslothAI/status/1905312969879421435">Tweet from Unsloth AI (@UnslothAI)</a>: Fine-tune Orpheus-TTS for free with our notebook!Orpheus delivers human-like speech with emotional cues (sighs, laughs) that outperform OpenAI. Customize voices + dialogue 2x faster using 70% less VRA...</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally">Tutorial: How to Run DeepSeek-V3-0324 Locally | Unsloth Documentation</a>: How to run DeepSeek-V3-0324 locally using our dynamic quants which recovers accuracy</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=MKX_XKs_BNZR">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF">unsloth/DeepSeek-V3-0324-GGUF · Hugging Face</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Open_source">Open source - Wikipedia</a>: no description found</li><li><a href="https://huggingface.co/docs/trl/v0.4.2/en/sft_trainer#packing-dataset-constantlengthdataset">Supervised Fine-tuning Trainer </a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">Finetuning from Last Checkpoint | Unsloth Documentation</a>: Checkpointing allows you to save your finetuning progress so you can pause it and then continue.</li><li><a href="https://notes.victor.earth/youre-probably-breaking-the-llama-community-license/">You&#x27;re Probably Breaking the Llama Community License</a>: You&#x27;re Probably Breaking the Llama Community License</li><li><a href="https://github.com/unslothai/unsloth/issues/2086)">unslothai/unsloth</a>: Finetune Llama 3.3, DeepSeek-R1, Gemma 3 &amp; Reasoning LLMs 2x faster with 70% less memory! 🦥 - unslothai/unsloth</li><li><a href="https://github.com/y-haidar/awbw-research">GitHub - y-haidar/awbw-research: This is an abandoned project, that contains the efforts of attempting to create an AI for the game awbw</a>: This is an abandoned project, that contains the efforts of attempting to create an AI for the game awbw - y-haidar/awbw-research</li><li><a href="https://github.com/canopyai/Orpheus-TTS/issues/37">Pre-train Data Structure · Issue #37 · canopyai/Orpheus-TTS</a>: Thank you for sharing great work, I want to know about pre-train data format and it meaning given config file &gt; ` &gt; # Datasets &gt; text_QA_dataset: &lt;speech input-ids&gt; &gt; TTS_dataset: &l...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1355173901855883415)** (1 messages): 

> `` 


- **No Topics Discussed**: There were no discussion topics found in the channel.
   - The only content was an image attachment.
- **Image Attachment Present**: An image was attached to the channel with a Discord CDN link.
   - The image was not further discussed or analyzed.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1354896976176615606)** (68 messages🔥🔥): 

> `Training Loss Interpretation, Gemma & Task Difficulty, Dataset Size & Overfitting, LM Studio Models, HF Upload & vLLM` 


- ****Loss Landscape Lowdown Leaves Learners Lost****: A user inquired about training loss decreasing early and staying near zero, questioning if sudden increases are bad signs, showing a [graph](https://cdn.discordapp.com/attachments/1179777624986357780/1354896975967027360/image.png?ex=67e84723&is=67e6f5a3&hm=3f0e1f53cd32481a10565cb630aa71589b55f50ac1ed66d398bd327285cb40e5).
   - Another member suggested the task might be too easy for **Gemma**, leading it to stop learning, while advising the user to use **Weights & Biases (W&B)** for better graph visualization.
- ****Double Trouble for BOS Token Causes Bedlam****: A user reported finding a **double BOS token** issue in the latest **Unsloth update (Gemma 3 4B)** when checking tokenizer decoding.
   - A hotfix was [identified](https://github.com/unslothai/unsloth-zoo/pull/106) which removed the accidentally added token.
- ****Unsloth Install Updates Unleash Unexpected Underperformance****: Users reported experiencing **massive issues** when using the `--no-deps` flag during Unsloth updates, contrary to some existing instructions.
   - A user strongly recommended updating all dependencies and highlighted outdated documentation, specifically pointing to [the Unsloth documentation](https://docs.unsloth.ai/get-started/installing-+-updating/updating).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo">Tutorial: Train your own Reasoning model with GRPO | Unsloth Documentation</a>: Beginner&#x27;s Guide to transforming a model like Llama 3.1 (8B) into a reasoning model by using Unsloth and GRPO.</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/updating">Updating | Unsloth Documentation</a>: To update or use an old version of Unsloth, follow the steps below:</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: Scripts for text classification with llama and bert - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/WecoAI/aideml">GitHub - WecoAI/aideml: AIDE: AI-Driven Exploration in the Space of Code. State of the Art machine Learning engineering agents that automates AI R&amp;D.</a>: AIDE: AI-Driven Exploration in the Space of Code. State of the Art machine Learning engineering agents that automates AI R&amp;D. - WecoAI/aideml</li><li><a href="https://github.com/unslothai/unsloth-zoo/pull/106">fix double bos by tamewild · Pull Request #106 · unslothai/unsloth-zoo</a>: It was mistakenly added here 4a66f8b</li><li><a href="https://www.reddit.com/r/unsloth/comments/1jldwql/comment/mk5yz0j/?%24deep_link=true&correlation_id=4bc85317-7b48-594b-9199-248dd1496be7&ref=email_post_reply&ref_campaign=email_post_reply&ref_source=email&%243p=e_as&_branch_match_id=1423385709101741523&utm_medium=Email+Amazon+SES&_branch_referrer=H4sIAAAAAAAAA3VO22qEMBD9GvdNrSauuiCltPQ3QjSzmt3cOomE7UO%2FvSNtHwszcDiXObOlFOKlrhGU0qmSIVRGu3vNwnPRchYmEDKeCHrUq3bSiB3NtB2pgr0U7TtNzrn6zS%2FeEoG0u4vGp40QcRZcigSbm1H5wxC6agdpdyCyTpuwOiak29FKYwRryGDv3ePz6XY0MCrhCiCI47WCvSXcoWjPi0cEI5P2TmhFPJ%2BXoWNNX%2FYzH8pu5HM5NuNYtnxQquHjeYaecghXMoOV2ojgYxIIwTx%2BBLFIG6Re3f%2BO6Hdc4E8%2FfREHiNqtYkafI%2BD0uqG38A33VShCWwEAAA%3D%3D">Reddit - The heart of the internet</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1355245690556780747)** (2 messages): 

> `Orpheus-TTS, Voice Model Finetuning, UnslothAI` 


- **Unsloth tunes Orpheus-TTS!**: Unsloth released a [finetuning notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb) for **Orpheus-TTS**, a voice model, available for free on Colab.
   - It enables customized voices and dialogue **2x faster** using **70% less VRAM** via Unsloth.
- **Orpheus-TTS displays Emotions!**: **Orpheus** delivers human-like speech with emotional cues (sighs, laughs) that outperform **OpenAI**.
   - Unsloth showed an example of finetuning it on a **1000** row dataset for just **100 steps** and managed to change the voice + personality of the model entirely!


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1905312969879421435">Tweet from Unsloth AI (@UnslothAI)</a>: Fine-tune Orpheus-TTS for free with our notebook!Orpheus delivers human-like speech with emotional cues (sighs, laughs) that outperform OpenAI. Customize voices + dialogue 2x faster using 70% less VRA...</li><li><a href="https://x.com/danielhanchen/status/1905315906051604595">Tweet from Daniel Han (@danielhanchen)</a>: We trained Orpheus-TTS, a voice LLM on a tiny dataset and managed to change the voice + personality of the model entirely!Pretty cool especially since the model has emotional cues like giggling or sig...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1354901327292928281)** (9 messages🔥): 

> `Dynamic Quantization, DeepSeek-R1, ACDiT` 


- **Unsloth's Dynamic Quantization Deep Dive**: A member inquired about the 'dynamic' aspect of **dynamic quantization** in Unsloth, asking whether weights causing more activation/quantization errors are identified and not quantized, with the rest quantized to 4 bits, based on the [super weights paper](https://arxiv.org/abs/2402.10433).
   - The question extends to how dynamic the kblams are and whether programs themselves can be encoded into the knowledge base, in other words how the quantization errors are calculated and which layers are more important, looking for a code base or regularization.
- **DeepSeek-R1 in GGUF and 4-bit formats available!**: Unsloth made available various versions of **DeepSeek-R1**, including [GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF?show_file_info=DeepSeek-R1-Q4_K_M%2FDeepSeek-R1-Q4_K_M-00001-of-00009.gguf) and [4-bit formats](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5).
   - Unsloth's **DeepSeek-R1** 1.58-bit + 2-bit Dynamic Quants is selectively quantized, improving accuracy over standard 1-bit/2-bit.
- **ACDiT paper appears**: The AutoConditional Diffusion Transformer (**ACDiT**) paper was shared:  [https://arxiv.org/abs/2412.07720](https://arxiv.org/abs/2412.07720).
   - The paper describes a combination of autoregressive and diffusion paradigms for modeling continuous visual information, using Skip-Causal Attention Mask (SCAM) on standard diffusion transformer during training, and autoregressive decoding during inference using KV-Cache.
- **Reasoning in GRPO notebooks is asked about**: A member asked about ways to play with the reasoning in the **GRPO** notebooks, specifically separating and modifying the reasoning before the model provides its final answer.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.07720">ACDiT: Interpolating Autoregressive Conditional Modeling and Diffusion Transformer</a>: We present ACDiT, a novel Autoregressive blockwise Conditional Diffusion Transformer, that innovatively combines autoregressive and diffusion paradigms for modeling continuous visual information. By i...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF?show_file_info=DeepSeek-R1-Q4_K_M%2FDeepSeek-R1-Q4_K_M-00001-of-00009.gguf">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1354894500488548483)** (305 messages🔥🔥): 

> `Gemini 2.5 Pro vs GPT-4o, Google AI Studio, Perplexity for News and Current Events, Claude vs GPT for Reasoning, AI Transcription Tools` 


- **GPT-4o Edges Out Gemini 2.5 Pro in Coding**: Members debated the coding capabilities of **GPT-4o** versus **Gemini 2.5 Pro**, with some finding *GPT-4o* superior for coding tasks, challenging initial impressions, while others favored *Gemini* overall or for specific tasks like C++ and WinAPI integration.
   - It was mentioned that [GPT-4o](link.to.g40) leads in the coding arena, but many third party benchmarks show that Gemini 2.5 Pro performs better overall for coding, except that **GPT-4o** wins 3 out of 6 categories.
- **Free Google AI Studio Steals the Show**: Users discussed the benefits of using **Google AI Studio**, noting its free access to models like **Gemini 2.5 Pro** and highlighting its generous prompt limits compared to paid services like **ChatGPT Plus**.
   - Members reported using hundreds of messages daily without hitting limits and shared a handy comparison resource, leading one to cancel their *ChatGPT* subscription due to *AI Studio's* advantages.
- **Perplexity Masters News Beat Over ChatGPT**: Members found **Perplexity** to be better than **ChatGPT** for accessing news and current events, highlighting its Discover tab for the latest news, and that **Perplexity** is more than just a *GPT wrapper*.
   - However, one user found **Perplexity's** *Deep Research* feature to have issues with quality and reliability, especially for research on uploaded files, recommending *ChatGPT* instead.
- **Claude 3.7 Sonnet Reigns Supreme for Reasoning**: Members praised **Claude 3.7 Sonnet** for its reasoning capabilities, noting its superior explanations compared to other AI models, especially because *free tier Claude fills up and forces you to start a new chat*.
   - It was suggested that models like o1, o3-mini-high, and Grok 3 are all great for coding, though one member found o1 the best at complex coding using C++, Physics, Rendering and old APIs such as Win32API.
- **Decoding the Best Free AI Transcription Tool**: Members sought advice on free **AI transcription tools**, with one suggesting running **Whisper** locally and another noting the difficulty of installing the necessary Python packages, and the complexity of resolving any dependency issues.
   - In order to resolve these issues, users should try to use a cloud based solution which often works immediately without the need to install and troubleshoot local packages.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1354969698974040086)** (8 messages🔥): 

> `Image generator, GPT-4.5 Error, GPT models for summarization, AI voice chatbot` 


- **Image Generator Needed**: A member is seeking an **image generator** that allows copying and pasting scenes from a script to convert into a cartoon style, without requiring frequent sign-ups.
   - They also complained about the new image generation being broken and failing to generate images.
- **Is GPT-4.5 error common?**: A member reported getting a **GPT-4.5 error in message stream** that randomly started dying, and they could not continue those chats.
   - The errors started since yesterday.
- **Best GPT Model for Text Summarization**: A member asked which **GPT model** is best for summarizing and analyzing tens of thousands of words of text, noting their experience with *o3 mini*, *mini high*, and *o1*.
   - Another member suggested **GPT-4.5** and *o1*, while advising against using *4o* for this particular task.
- **STT Integration Troubleshoot**: A member is developing an **AI voice chatbot** that integrates speech-to-text (STT), a language model (LLM), and text-to-speech (TTS) but is facing compatibility issues with OpenAI versions.
   - The chat completion feature only functions with OpenAI versions earlier than 1, while they are currently on version 1.66, and the *store:true* command doesn't execute as expected.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1354999301813702757)** (83 messages🔥🔥): 

> `Yu-Gi-Oh! card art prompting, Microsoft PromptWizard, ChatGPT prompting methods, Hierarchical communication with markdown, AI prompt engineering` 


- **Prompting Yu-Gi-Oh! Card Art Style**: A member seeks advice on improving prompts to generate art in the style of **Yu-Gi-Oh!** trading cards, noting that **ChatGPT** tends to default to comic art instead.
   - The user has already tried using prompts like *"Render this character in the style of a Yu-Gi-Oh! trading card illustration. Use sharp, clean digital art..."* with incremental improvements.
- **Microsoft PromptWizard Usage**: A member inquired about experiences using **Microsoft PromptWizard** for custom data, seeking insights from the community.
   - No responses were provided in the given text.
- **Unlock the Best of ChatGPT**: A member asked about *secret prompts* or methods to maximize **ChatGPT's** potential, feeling there's more to leverage.
   - Suggestions included using *prompt conditions and disclaimers* before providing prompts.
- **Darth's Prompting Primer**: A member shared a structured approach to teaching effective prompting techniques, including hierarchical communication using markdown, abstraction through open variables, and reinforcement strategies.
   - This approach includes a [shareable ChatGPT link](https://chatgpt.com/share/67e6dca1-46d0-8000-8ee5-5fc8e61f06d6) and an emphasis on **ML format matching** for better compliance.
- **Enhanced Image Prompting with ChatGPT**: A member expressed delight with the new **ChatGPT** image tool, noting its improved adherence to prompt requirements and ability to handle complex scenarios such as generating a moving market on a giant turtle's back with a sun and three moons.
   - The user found that the new tool was much better at making changes without impacting the image as a whole, such as removing stars from a night scene after initially including them.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1354999301813702757)** (83 messages🔥🔥): 

> `Yu-Gi-Oh! card art prompting, Microsoft PromptWizard, ChatGPT prompting tips, Hierarchical communication with markdown, GPTs in conversation` 


- **Yu-Gi-Oh! Art Style Prompting Tweaks**: A user seeks advice on improving prompts for generating **Yu-Gi-Oh!** card art, noting success with *Ghibli* and *photorealism* styles but struggles with the desired **Yu-Gi-Oh!** aesthetic.
   - They provided examples and their current prompt focuses on *sharp, clean digital art with stylized anime rendering, glowing magical effects, and a dynamic pose*.
- **PromptWizard Users Unite**: A member inquires about experiences with **Microsoft PromptWizard** for custom data applications.
   - Others are seeking *secret prompts* or methods to maximize **ChatGPT's** potential.
- **Prompting Strategy: Condition and Disclaimer**: A member advises adding a *prompt condition and disclaimer* before the main prompt to guide **ChatGPT's** output.
   - A link to a [primed session](https://chatgpt.com/share/67e6dca1-46d0-8000-8ee5-5fc8e61f06d6) demonstrates *hierarchical communication with markdown, abstraction through open variables, reinforcement, and ML format matching for compliance*.
- **GPTs join Conversational Forces**: A user joyfully discovers the ability to invoke custom **GPTs** within a **ChatGPT** conversation by typing `@`.
   - Another user notes that they like *being able to dictate tool use*, and that *the new imagegen seems to be performing strongly against the strange requirements I am giving it*.
- **Markdown Misunderstanding in AI Prompts**: One user points out that this channel's *no markdown rule* is *lazy*, advocating for its use in educating others since it's **the language the AI uses**.
   - They argue that code blocks, while providing formatting, add an unnecessary abstraction layer, potentially confusing users unfamiliar with the format and causing them to *freeze*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1355087823174369351)** (2 messages): 

> `Fount AI Character Interactions Framework, Gideon project` 


- **Fount Framework for AI Character Interactions Emerges**: A member shared the [Fount](https://github.com/steve02081504/fount) project, an extensible framework for building and hosting **AI character interactions** using pure JS.
   - The framework offers flexibility via modular components, custom **AI source integration**, powerful plugins, and a seamless cross-platform chat experience.
- **Gideon Project Surfaces on GitHub**: A member shared the [Gideon project](https://github.com/Emperor-Ovaltine/gideon) on GitHub.
   - No further details were provided about the project's purpose or functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/steve02081504/fount">GitHub - steve02081504/fount: An extensible framework for building and hosting AI character interactions. Built with pure JS, Fount offers unparalleled flexibility via modular components, custom AI source integration, powerful plugins, and a seamless cross-platform chat experience.</a>: An extensible framework for building and hosting AI character interactions. Built with pure JS, Fount offers unparalleled flexibility via modular components, custom AI source integration, powerful ...</li><li><a href="https://github.com/Emperor-Ovaltine/gideon">GitHub - Emperor-Ovaltine/gideon</a>: Contribute to Emperor-Ovaltine/gideon development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1354923122175774922)** (327 messages🔥🔥): 

> `Gemini 2.5 Pro Access and Limitations, OpenRouter AI SDK Configuration, Free Models with Function Calling, Token Per Second Performance for Coding Models, OpenAI Responses API` 


- **Gemini 2.5 Pro: Rate Limits Trigger User Lament**: Users are reporting [low rate limits for Gemini 2.5 Pro](https://x.com/OpenRouterAI/status/1905300582505624022), even after adding their own **AI Studio API keys**, leading to discussions on how to maximize free quota and manage usage for real applications.
   - A member pointed out the model *won't be free forever* so that windsurf has to start charging for it, that's gonna be a problem.
- **AI SDK Provider options, nested order array is an ongoing struggle.**: Members are actively debugging [OpenRouter AI SDK provider options](https://github.com/OpenRouterTeam/ai-sdk-provider), particularly the use of `providerOptions` to specify model order and fallback behavior.
   - The issue is around whether nesting the **order array** under the **provider** key is correct, with debugging attempts showing unexpected provider selection despite configured order.  The team acknowledges it's a bug and looking to address the AI SDK issue, hopefully.
- **Seeking Function Calling Nirvana in free LLMs**: Members are searching for free models that support function calling, with some suggesting **Mistral Small 3.1** and **Gemini free models** as potential options.
   - Another member mentions, *Gosh, I'm trying so hard to find a free model that supports function calling. I can't find any!*.
- **Model TPS face-off: Gemini Flash 2.0 vs the World**: Members are debating the **tokens per second (TPS)** performance of various coding models, with **Gemini Flash 2.0** mentioned for its speed, but also facing some criticisms of being trash, something about their hosting is messed up.
   - Groq serves the **70B R1 distil at 600tok/s** and one member chimed in that it *isn't good at coding imo*.
- **OpenAI Responses API support?**: A member inquired about [OpenRouter supporting the OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses), and one of the OpenRouter team flagged *a couple gotchas* with it.
   - The member asking wanted good image to video and the OpenRouter team suggested that [Veo2 API](https://openai.com/blog/veo) is going to be your best bet for SOTA, but it's about **50 cents per second of video**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/features/prompt-caching">Prompt Caching - Optimize AI Model Costs with Smart Caching</a>: Reduce your AI model costs with OpenRouter&#x27;s prompt caching feature. Learn how to cache and reuse responses across OpenAI, Anthropic Claude, and DeepSeek models.</li><li><a href="https://x.com/OpenRouterAI/status/1905300582505624022">Tweet from OpenRouter (@OpenRouterAI)</a>: To maximize your free Gemini 2.5 quota:1. Add your AI Studio API key in https://openrouter.ai/settings/integrations. Our rate limits will be a “surge protector” for yours.2. Set up OpenRouter in your ...</li><li><a href="https://openrouter.ai/settings/integrations">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - Manage Model Usage and Quotas</a>: Learn about OpenRouter&#x27;s API rate limits, credit-based quotas, and DDoS protection. Configure and monitor your model usage limits effectively.</li><li><a href="https://openrouter.ai/activity">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/docs/features/provider-routing">Provider Routing - Smart Multi-Provider Request Management</a>: Route AI model requests across multiple providers intelligently. Learn how to optimize for cost, performance, and reliability with OpenRouter&#x27;s provider routing.</li><li><a href="https://github.com/OpenRouterTeam/ai">GitHub - OpenRouterTeam/ai: Build AI-powered applications with React, Svelte, Vue, and Solid</a>: Build AI-powered applications with React, Svelte, Vue, and Solid - OpenRouterTeam/ai</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider?tab=readme-ov-file#passing-extra-body-to-openrouter">GitHub - OpenRouterTeam/ai-sdk-provider: The OpenRouter provider for the Vercel AI SDK contains support for hundreds of models through the OpenRouter chat and completion APIs.</a>: The OpenRouter provider for the Vercel AI SDK contains support for hundreds of models through the OpenRouter chat and completion APIs. - OpenRouterTeam/ai-sdk-provider</li><li><a href="https://github.com/OpenRouterTeam/ai-sdk-provider">GitHub - OpenRouterTeam/ai-sdk-provider: The OpenRouter provider for the Vercel AI SDK contains support for hundreds of models through the OpenRouter chat and completion APIs.</a>: The OpenRouter provider for the Vercel AI SDK contains support for hundreds of models through the OpenRouter chat and completion APIs. - OpenRouterTeam/ai-sdk-provider
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1354893312699666669)** (299 messages🔥🔥): 

> `MCP server config, Prompts and ICL, Ollama models and MCP, Google search integration, Oterm client and MCP` 


- **Agent Instructions and Tool Usage**: Members discussed best practices for instructing agents on **tool usage**, particularly regarding the order of tool calls, referencing [Cline's system prompt](https://github.com/cline/cline/blob/main/src/core/prompts/system.ts) for ideas.
   - A member suggested prompting directly on the server, to look like `First call ${tool1.name}, then ${tool2.name}`.
- **Prompts for In-Context Learning (ICL) Gain Traction**: It was stated that [MCP servers can supply instructions](https://github.com/modelcontextprotocol/specification/pull/188#issue-2895415136) to encourage specific agent behaviors, such as tool usage, using prompts for ICL.
   - One member shared a [link on using prompts for ICL](https://x.com/llmindsetuk/status/1899148877787246888?t=WcqjUT4wCCHd_qj-QPf7yQ&s=19) and a [test showing it working](https://github.com/evalstate/fast-agent/blob/main/tests%2Fe2e%2Fprompts-resources%2Ftest_prompts.py#L75-L92).
- **Ollama Model Configuration Confusion Persists**: A member had issues connecting a local LLM via Ollama to an MCP server and asked for guidance.
   - It was suggested to use oterm with [this MCP config](https://ggozad.github.io/oterm/mcp/) and replace the content of the config file, furthermore stating that the default 4-bit Ollama models are often insufficient for proper tool usage, recommending 8-bit versions for better performance and more models available [here](https://ollama.com/library/mistral).
- **Adding Google Real-Time Search Tool to MCP Discussed**: A member inquired about adding Google Search to MCP, and another member shared their [configuration](https://cdn.discordapp.com/attachments/1312302100125843479/1355126409093316750/config.json?ex=67e7cb50&is=67e679d0&hm=8311f31b3b6181eb391876bad03fc45f745e439a12180e6ad087d94983c37c1c&)
   - They noted that users need to obtain their own Google API key and engine ID to use the configuration.
- **Discover Blender MCP Servers**: A user successfully used the tool and tabulated servers related to blender, finding multiple **Blender Model Context Protocol Servers**.
   - BlenderMCP ([GitHub](https://github.com/ahujasid/blender-mcp)), Blender MCP Server ([GitHub](https://github.com/cwahlfeldt/blender-mcp)), Unreal-Blender-MCP ([GitHub](https://github.com/tahooki/unreal-blender-mcp)), Bonsai-mcp ([GitHub](https://github.com/JotaDeRodriguez/Bonsai_mcp)), and Tripo MCP Server ([GitHub](https://github.com/VAST-AI-Research/tripo-mcp)).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.48/containers/json":">no title found</a>: no description found</li><li><a href="https://ggozad.github.io/oterm/mcp/">Index - oterm</a>: no description found</li><li><a href="https://ollama.com/library/llama3.2:3b-text-q8_0">llama3.2:3b-text-q8_0</a>: Meta&#39;s Llama 3.2 goes small with 1B and 3B models. </li><li><a href="https://glama.ai/api/mcp/openapi.json"">MCP API Reference</a>: API Reference for the Glama Gateway</li><li><a href="https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#listening-for-messages-from-the-server">Transports</a>:           ℹ️                  Protocol Revision: 2025-03-26      MCP uses JSON-RPC to encode messages. JSON-RPC messages MUST be UTF-8 encoded.The protocol currently defines two standard transport mec...</li><li><a href="https://ollama.com/library/mistral">mistral</a>: The 7B model released by Mistral AI, updated to version 0.3.</li><li><a href="https://ollama.com/library/llama3.2:3b-instruct-q8_0">llama3.2:3b-instruct-q8_0</a>: Meta&#39;s Llama 3.2 goes small with 1B and 3B models. </li><li><a href="https://github.com/modelcontextprotocol/specification/pull/188#issue-2895415136">Added Tool Call and Tool Result to GetPrompt for in-context learning … by evalstate · Pull Request #188 · modelcontextprotocol/specification</a>: …of tool usageAddition of ToolCall and ToolResult blocks to PromptMessage to allow in-context learning of Tool Usage patterns and error handling.Submitted as draft for review before completing/ad...</li><li><a href="https://x.com/llmindsetuk/status/1899148877787246888?t=WcqjUT4wCCHd_qj-QPf7yQ&s=19">Tweet from llmindset (@llmindsetuk)</a>: Let&#39;s take a look at an underappreciated MCP Feature: Prompts - and why they are important for Agent based applications. We&#39;ll start with 2 simple Agents that return the size of an object - on...</li><li><a href="https://github.com/yuniko-software/minecraft-mcp-server">GitHub - yuniko-software/minecraft-mcp-server: A Minecraft MCP Server powered by Mineflayer API. It allows to control a Minecraft character in real-time, allowing AI assistants to build structures, explore the world, and interact with the game environment through natural language instruction</a>: A Minecraft MCP Server powered by Mineflayer API. It allows to control a Minecraft character in real-time, allowing AI assistants to build structures, explore the world, and interact with the game ...</li><li><a href="https://github.com/evalstate/fast-agent/blob/main/tests%2Fe2e%2Fprompts-resources%2Ftest_prompts.py#L75-L92">fast-agent/tests/e2e/prompts-resources/test_prompts.py at main · evalstate/fast-agent</a>: Define, Prompt and Test MCP enabled Agents and Workflows - evalstate/fast-agent</li><li><a href="https://github.com/cline/cline/blob/main/src/core/prompts/system.ts">cline/src/core/prompts/system.ts at main · cline/cline</a>: Autonomous coding agent right in your IDE, capable of creating/editing files, executing commands, using the browser, and more with your permission every step of the way. - cline/cline</li><li><a href="https://github.com/ahujasid/blender-mcp">GitHub - ahujasid/blender-mcp</a>: Contribute to ahujasid/blender-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/cwahlfeldt/blender-mcp">GitHub - cwahlfeldt/blender-mcp</a>: Contribute to cwahlfeldt/blender-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/tahooki/unreal-blender-mcp">GitHub - tahooki/unreal-blender-mcp: unreal-blender-mcp</a>: unreal-blender-mcp. Contribute to tahooki/unreal-blender-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/JotaDeRodriguez/Bonsai_mcp">GitHub - JotaDeRodriguez/Bonsai_mcp</a>: Contribute to JotaDeRodriguez/Bonsai_mcp development by creating an account on GitHub.</li><li><a href="https://github.com/VAST-AI-Research/tripo-mcp">GitHub - VAST-AI-Research/tripo-mcp: MCP server for Tripo</a>: MCP server for Tripo. Contribute to VAST-AI-Research/tripo-mcp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1354895650457780306)** (9 messages🔥): 

> `Canvas MCP, Docker Compose for MCP Servers, Model Context Protocol (MCP) Explanation, Speech MCP, Gradescope Integration` 


- ****Canvas MCP** Lets Agents Talk to Canvas LMS**: A member created a **Canvas MCP** server, enabling AI agents to interact with Canvas LMS, and also added an agent that can autonomously crawl Gradescope to find info, available at [Canvas-MCP](https://git.new/canvas-mcp).
   - The tool offers features like finding relevant resources, querying upcoming assignments, and accessing courses and assignments from **Gradescope**.
- **All-In-One Docker Compose for Self-Hosting **17 MCP Servers****: A member created an all-in-one **Docker Compose** setup for easily self-hosting **17 MCP servers** using Portainer, with Dockerfiles sourced from public GitHub projects ([MCP-Mealprep](https://github.com/JoshuaRL/MCP-Mealprep)).
   - Another member suggested to *not bind the containers on 0.0.0.0 unless you need this accessible remotely* and to *include in the readme an example mcp config json*.
- ****Model Context Protocol (MCP)** Explained**: A team member shared a [blog post](https://pieces.app/blog/mcp) introducing **MCP (Model Context Protocol)**, describing it as an open standard released by Anthropic in late 2024.
   - The blog post describes it as a *the USB-C of AI integrations* that lets Large Language Models powering tools like Claude or ChatGPT communicate with external data sources and tools.
- ****Speech MCP** Demoed**: A member shared a link to **Speech MCP**, with a [YouTube Shorts demo](https://www.youtube.com/shorts/rurAp_WzOiY).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/JoshuaRL/MCP-Mealprep">GitHub - JoshuaRL/MCP-Mealprep: This project takes a number of MCP servers from GitHub locations, packages them together with their referenced Dockerfiles, and pulls them together with docker-compose to run as a stack for ML/AI resources.</a>: This project takes a number of MCP servers from GitHub locations, packages them together with their referenced Dockerfiles, and pulls them together with docker-compose to run as a stack for ML/AI r...</li><li><a href="https://pieces.app/blog/mcp">What the heck is Model Context Protocol (MCP)? And why is everybody talking about it?</a>: Discover what Model Context Protocol or MCP is and why it’s trending. Learn how it’s changing the game for developers and teams.</li><li><a href="https://git.new/canvas-mcp">GitHub - aryankeluskar/canvas-mcp: Collection of Canvas LMS and Gradescope tools for the ultimate EdTech model context protocol. Allows you to query your courses, find resources, and chat with upcoming assignments in the AI app of your choice. try now!</a>: Collection of Canvas LMS and Gradescope tools for the ultimate EdTech model context protocol. Allows you to query your courses, find resources, and chat with upcoming assignments in the AI app of y...</li><li><a href="https://www.youtube.com/shorts/rurAp_WzOiY"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1354893946190299187)** (216 messages🔥🔥): 

> `R1 vs O3 Mini, Anthropic Thoughts Microscope, GPT-4o Update, OpenRouter Limits, Running Local Aider Branch with UV` 


- **OpenRouter R1 Model Underperforms**: A member found the free **R1** model on OpenRouter to be *"stupid"*, verbose, and ineffective at solving broken tests, especially with repomap enabled, unlike **O3-mini**.
   - It's speculated that the free **R1** model is a quantized version of **DeepSeek**, possibly in FP8 format, while the DeepSeek on the leaderboard is from the official DeepSeek team.
- **GPT-4o Aces Coding Arena**: The latest **ChatGPT-4o** update jumps to #2 on the [Arena leaderboard](https://x.com/lmarena_ai/status/1905340075225043057), surpassing **GPT-4.5**, tying #1 in Coding, Hard Prompts, and performing in the Top-2 across ALL categories while costing 10x less.
   - However, this update is confusingly released as **chatgpt-4o-latest** endpoint, which is priced at $5/$15 per million input/output tokens, whereas the API snapshots are priced at $2.5/$10, so caution is recommended when moving workloads, according to [Artificial Analysis](https://x.com/ArtificialAnlys/status/1905563427776651344).
- **Constant Context Architecture is a Game Changer**: Constant Context Architecture (**CCA**) is proposed as a solution for working with large codebases using LLMs, guaranteeing that the necessary context for modifying any module will always fit within an LLM's context window, regardless of the total codebase size, as described in this [blogpost](https://neohalcyon.substack.com/p/constant-context-architecture).
   - This is achieved by ensuring modules have bounded size, interfaces, and dependencies, making context gathering a bounded operation.
- **Aider's Context Command Automates File Management**: The new `/context` command automatically identifies relevant files for a given request and adds them to the chat, as discussed in [this discord thread](https://discord.com/channels/1131200896827654144/1354944747004887162/1354945152287899809).
   - It's particularly useful for large codebases and saves time by automating the process of manually adding files.
- **Multiple API Keys**: OpenRouter users are rotating through many api keys to make sure they can make the most requests possible, while avoiding rate limits.
   - However, Google has been known to suspend accounts if they detect multiple keys or abuse from a single user.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AnthropicAI/status/1905303835892990278">Tweet from Anthropic (@AnthropicAI)</a>: New Anthropic research: Tracing the thoughts of a large language model.We built a &#34;microscope&#34; to inspect what happens inside AI models and use it to understand Claude’s (often complex and sur...</li><li><a href="https://x.com/BenjaminDEKR/status/1905461907156271465">Tweet from Benjamin De Kraker (@BenjaminDEKR)</a>: POV: Cursor &#34;fixing your code&#34;</li><li><a href="https://x.com/lmarena_ai/status/1905340075225043057">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: News: the latest ChatGPT-4o (2025-03-26) jumps to #2 on Arena, surpassing GPT-4.5!Highlights- Significant improvement over the January version (+30 pts, #5-&gt;#2)- Tied #1 in Coding, Hard Prompts. To...</li><li><a href="https://x.com/OpenAI/status/1905331956856050135">Tweet from OpenAI (@OpenAI)</a>: GPT-4o got an another update in ChatGPT!What&#39;s different?- Better at following detailed instructions, especially prompts containing multiple requests- Improved capability to tackle complex technic...</li><li><a href="https://x.com/ArtificialAnlys/status/1905563427776651344">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: Today’s GPT-4o update is actually big - it leapfrogs Claude 3.7 Sonnet (non-reasoning) and Gemini 2.0 Flash in our Intelligence Index and is now the leading non-reasoning model for codingThis makes GP...</li><li><a href="https://neohalcyon.substack.com/p/constant-context-architecture">Constant Context Architecture</a>: Building for the LLM Era</li><li><a href="https://openrouter.ai/settings/integrations">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://aider.chat/docs/repomap.html#optimizing-the-map">Repository map</a>: Aider uses a map of your git repository to provide code context to LLMs.</li><li><a href="https://github.c>>>">no title found</a>: no description found</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free">DeepSeek V3 0324 (free) - API, Providers, Stats</a>: DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship chat model family from the DeepSeek team.It succeeds the [DeepSeek V3](/deepseek/deepseek-chat-v3) mode...</li><li><a href="https://aider.chat/docs/repomap.html">Repository map</a>: Aider uses a map of your git repository to provide code context to LLMs.</li><li><a href="https://aider.chat/docs/troubleshooting/token-limits.html">Token limits</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: Using the code, architect, ask and help chat modes.</li><li><a href="https://tenor.com/view/primary-day-polling-place-homer-simpson-the-simpsons-voting-gif-12450826">Primary Day Polling Place GIF - Primary Day Polling Place Homer Simpson - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1354900540370456868)** (31 messages🔥): 

> `AiderMacs, Cargo Build Integration, Gemini 2.5 Pro Rate Limits, Aider Architect Mode, Model Combinations` 


- **Debugging Aidermacs Integration**: A user mentioned debugging a bug in **Aidermacs**, which they were using to invoke **Aider** and more closely integrate it into **Emacs**, describing it as *"shaving a yak"*.
   - The user also clarified a `lint-cmd` configuration detail, noting it should be `echo` and not `test`.
- **Fresh Starts Beat Going in Circles**: A user asked about using `/undo` iteratively versus continuing a chat, and another user recommended using `/clear` to start fresh *if you’re going in circles*.
   - The conversation highlighted that `/undo` retains memory of the last action, while `/clear` purges the entire chat history.
- **Cargo Build Integration Wishlisted**: A user inquired about integrating `cargo build` with **Aider**, to pipe errors/warnings back to the model for resolution.
   - While no direct solution was provided, the query suggests a desired feature for enhanced code debugging workflows.
- **Gemini 2.5 Pro rate limits frustrate**: Multiple users reported hitting rate limits with **Gemini 2.5 Pro**, even when seemingly below the documented **50 requests/day**, with one noting the existence of a **2 requests/minute limit**.
   - There was discussion on whether purchasing a paid account would resolve the limitations, with mixed results reported, along with a potential fallback model implementation.
- **Architect Mode Deep Dive Requested**: A user sought a deeper understanding of **Aider's architect mode**, contrasting it with their existing workflow using regular mode with models like **Sonnet 3.5** and **Gemini 2.5 Pro**, guided by an `aider.rules.md` file.
   - They expressed a desire to optimize their process, potentially leveraging architect mode to avoid redundant work or overpaying for model usage and another member viewed it as a combo of *ask + code*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Aider-AI/aider/issues/3641#issuecomment-2762538743">Gemini 2.5 Pro or DeepSeek V3 0324 not showing in `/models /` · Issue #3641 · Aider-AI/aider</a>: I have been using /models / to get a list of available models to use and based Aidermacs to select from the list, I&#39;ve very happy that Gemini 2.5 Pro and the latest deepseek are supported in Aider...</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: Using the code, architect, ask and help chat modes.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1354894995294912723)** (26 messages🔥): 

> `GPT-4o Update, OpenAI Image Generation Policy, Devin Wiki Launch, AI Writing Editing` 


- **GPT-4o Jumps to #2 on Arena!**: The latest **ChatGPT-4o** (2025-03-26) jumps to **#2** on Arena, surpassing **GPT-4.5** with a significant improvement over the January version (+30 pts, #5->#2), and tying **#1** in Coding, Hard Prompts according to this [tweet](https://fxtwitter.com/lmarena_ai/status/1905340075225043057).
- **New 4o Policy Allows More Creative Freedom**: OpenAI launched native image generation in **ChatGPT** through **4o**, shifting from blanket refusals in sensitive areas to a more precise approach focused on preventing real-world harm as explained in this [blog post](https://x.com/joannejang/status/1905341734563053979).
- **Devin Indexes Repos with Devin Wiki**: **Devin** now automatically indexes your repos and produces wikis with architecture diagrams, links to sources, and more according to this [tweet](https://x.com/cognition_labs/status/1905385526364176542?s=46&t=jDrfS5vZD4MFwckU5E8f5Q).
- **AI Helps Edit Non-Technical Writing**: Members discussed using **Claude** and **GPT** for editing non-technical writing, with one member noting that *Claude validates everything I write and GPT wants to rewrite everything I write* and this user wants *a better middle ground*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/artificialanlys/status/1905563427776651344?s=46">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: Today’s GPT-4o update is actually big - it leapfrogs Claude 3.7 Sonnet (non-reasoning) and Gemini 2.0 Flash in our Intelligence Index and is now the leading non-reasoning model for codingThis makes GP...</li><li><a href="https://x.com/julianlehr/status/1855858599156932773">Tweet from Julian Lehr (@julianlehr)</a>: &#34;So I have an idea for an essay that I want to write. It&#39;s a blog post, like a long-form blog post, but it&#39;s not super concrete yet, like I have a rough idea of what the blog post should b...</li><li><a href="https://x.com/cognition_labs/status/1905328447318311218?s=46&t=jDrfS5vZD4MFwckU5E8f5Q">Tweet from Cognition (@cognition_labs)</a>: We’re shipping Devin Search, a new tool for codebase understanding.Use Devin Search for quick questions like &#34;how is user authentication implemented?&#34; or turn on Deep Mode for complex asks lik...</li><li><a href="https://x.com/levelsio/status/1905324525006299521?s=46">Tweet from @levelsio (@levelsio)</a>: For all the hundreds of people making Ghibli generator apps nowIf you call it Ghibli or similar, you will get a letter from Studio Ghibli&#39;s lawyersBut worse, if you make millions, you&#39;ll just ...</li><li><a href="https://x.com/cognition_labs/status/1905385526364176542?s=46&t=jDrfS5">Tweet from Cognition (@cognition_labs)</a>: Launching Devin Wiki: Devin now automatically indexes your repos and produces wikis with architecture diagrams, links to sources, and more.Use it to get up to speed on unfamiliar parts of your codebas...</li><li><a href="https://x.com/LangChainAI/status/1905325891934454170">Tweet from LangChain (@LangChainAI)</a>: Watch the video ➡️ https://www.youtube.com/watch?v=NKXRjZd74ic</li><li><a href="https://x.com/cognition_labs/status/1905385526364176542?s=46&t=jDrfS5vZD4MFwckU5E8f5Q">Tweet from Cognition (@cognition_labs)</a>: Launching Devin Wiki: Devin now automatically indexes your repos and produces wikis with architecture diagrams, links to sources, and more.Use it to get up to speed on unfamiliar parts of your codebas...</li><li><a href="https://fxtwitter.com/joannejang/status/1905341734563053979)">Tweet from Joanne Jang (@joannejang)</a>: // i lead model behavior at openai, and wanted to share some thoughts & nuance that went into setting policy for 4o image generation.features capital letters (!) bc i published it as a blog post:--Thi...</li><li><a href="https://x.com/joannejang/status/1905341734563053979>)">Tweet from Joanne Jang (@joannejang)</a>: // i lead model behavior at openai, and wanted to share some thoughts & nuance that went into setting policy for 4o image generation.features capital letters (!) bc i published it as a blog post:--Thi...</li><li><a href="https://fxtwitter.com/OpenAI/status/1905331956856050135)">Tweet from OpenAI (@OpenAI)</a>: GPT-4o got an another update in ChatGPT!What&#39;s different?- Better at following detailed instructions, especially prompts containing multiple requests- Improved capability to tackle complex technic...</li><li><a href="https://x.com/OpenAI/status/1905331956856050135>)">Tweet from OpenAI (@OpenAI)</a>: GPT-4o got an another update in ChatGPT!What&#39;s different?- Better at following detailed instructions, especially prompts containing multiple requests- Improved capability to tackle complex technic...</li><li><a href="https://fxtwitter.com/lmarena_ai/status/1905340075225043057)">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: News: the latest ChatGPT-4o (2025-03-26) jumps to #2 on Arena, surpassing GPT-4.5!Highlights- Significant improvement over the January version (+30 pts, #5-&gt;#2)- Tied #1 in Coding, Hard Prompts. To...</li><li><a href="https://x.com/lmarena_ai/status/1905340075225043057>)">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: News: the latest ChatGPT-4o (2025-03-26) jumps to #2 on Arena, surpassing GPT-4.5!Highlights- Significant improvement over the January version (+30 pts, #5-&gt;#2)- Tied #1 in Coding, Hard Prompts. To...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1355238157896912946)** (3 messages): 

> `Dharmesh Shah, HubSpot, Agent.ai, hybrid teams, Claude Plays Pokemon hackathon` 


- ****Dharmesh Shah** joins Latent Space**: Latent Space shared a conversation with **[Dharmesh Shah](https://x.com/dharmesh/status/1789687037261402336)**, co-founder of **HubSpot** and creator of **[Agent.ai](http://agent.ai/)**.
   - The episode is about *the next evolution in workplace organization where human workers collaborate with AI agents as team members*.
- **Attendees can join **Claude Plays Pokemon hackathon****: Folks in SF: Join us for the **[Claude Plays Pokemon hackathon](https://lu.ma/poke)** this Sunday!
   - It's a real *catch 'em all* opportunity.
- **Members are encouraged to fill out **2025 State of AI Eng survey****: Members not in SF are encouraged to Fill out **[the 2025 State of AI Eng survey](https://www.surveymonkey.com/r/57QJSF2)** for **$250** in Amazon cards!
   - Act fast for a chance to get some Amazon money.
- ****Hybrid teams** are the future**: A particularly compelling concept we discussed is the idea of **"hybrid teams"** - *the next evolution in workplace organization where human workers collaborate with AI agents as team members*.
   - This raises interesting questions about *team dynamics, trust, and how to effectively delegate tasks between human and AI team members*.



**Link mentioned**: <a href="https://www.latent.space/p/dharmesh">The Agent Network — Dharmesh Shah</a>: Dharmesh Shah on Intelligent Agents, Market Inefficiencies, and Building the Next AI Marketplace

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1355270295279370370)** (189 messages🔥🔥): 

> `LLM Codegen Workflow, Documentation for LLMs, Memory-Ref Tool, Cursor IDE, Self-Improving Agents` 


- **Harper Reveals LLM Codegen Workflow**: A member shared their [LLM codegen workflow](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/), emphasizing **brainstorming specs**, planning, and executing with LLM codegen in discrete loops.
   - This workflow is built upon personal work, conversations with friends, and best practices from various internet sources, though the poster notes that *it will probably not work in 2 weeks, or it will work twice as well*.
- **Docs.dev Plugs into GitHub for Streamlined Documentation**: [Docs.dev](https://docs.dev/) was shared for generating docs directly from the codebase and keeping them up to date as code changes.
   - The tool enables users to **generate, audit, or analyze Markdown docs** with AI, offering both a rich text editor and Markdown options.
- **Nuvic's FZF Kit Extends Neovim's Fuzzy Finding**: A member linked to [fzf-kit.nvim](https://github.com/nuvic/fzf-kit.nvim), a Neovim plugin that extends fzf-lua with additional utilities.
   - This plugin enhances Neovim's fuzzy finding capabilities, improving file and code navigation.
- **Memory-Ref Tool Aids LLM Context Retention**: Members discussed using **memory-ref** tools to create and query a knowledge graph of memories for LLMs, helping them retain context across sessions.
   - One user highlighted the integration of **Cursor IDE** with **Graphiti**, using Graphiti’s Model Context Protocol (MCP) server for persistent memory, as detailed in [this Hacker News post](https://news.ycombinator.com/item?id=43506068).
- **llms.txt Emerges for Website-LLM Coordination**: A member shared [llms-txt](https://github.com/AnswerDotAI/llms-txt), a file to help language models use websites more effectively.
   - The discussion touched on the broader topic of self-improving models and how to guide LLMs with structured documentation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">My LLM codegen workflow atm</a>: A detailed walkthrough of my current workflow for using LLms to build software, from brainstorming through planning and execution.</li><li><a href="https://news.ycombinator.com/item?id=43506068">Show HN: Cursor IDE now remembers your coding prefs using MCP | Hacker News</a>: no description found</li><li><a href="https://x.com/PrajwalTomar_/status/1895839765280539068?s=19">Tweet from Prajwal Tomar (@PrajwalTomar_)</a>: In last 5 months, I’ve built 16 SaaS products for clients using Cursor.Now, I’ve cracked the best AI coding workflow for Cursor.Here’s my step-by-step guide to building production-ready MVPs:</li><li><a href="https://www.codeguide.dev/">CodeGuide</a>: CodeGuide creates Detailed Documentation for your AI Coding Project.</li><li><a href="https://github.com/AnswerDotAI/llms-txt">GitHub - AnswerDotAI/llms-txt: The /llms.txt file, helping language models use your website</a>: The /llms.txt file, helping language models use your website - AnswerDotAI/llms-txt</li><li><a href="https://docs.dev/">Docs.dev | AI-assisted docs</a>: Generate docs directly from your codebase and existing docs. Ensure your docs stay up to date as code changes.</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://github.com/nuvic/fzf-kit.nvim">GitHub - nuvic/fzf-kit.nvim: A Neovim plugin that extends fzf-lua with additional utilities</a>: A Neovim plugin that extends fzf-lua with additional utilities - nuvic/fzf-kit.nvim</li><li><a href="https://github.com/go-go-golems/go-go-mcp/tree/main/ttmp">go-go-mcp/ttmp at main · go-go-golems/go-go-mcp</a>: Anthropic MCP go implementation. Contribute to go-go-golems/go-go-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/go-go-golems/go-go-labs/blob/main/ttmp/2025-03-23/03-add-embeddings-to-command.md">go-go-labs/ttmp/2025-03-23/03-add-embeddings-to-command.md at main · go-go-golems/go-go-labs</a>: GO GO EXPERIMENTAL LAB. Contribute to go-go-golems/go-go-labs development by creating an account on GitHub.</li><li><a href="https://github.com/joernio/astgen">GitHub - joernio/astgen: Generate AST in json format for JS/TS</a>: Generate AST in json format for JS/TS. Contribute to joernio/astgen development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1354943243271274646)** (1 messages): 

> `LM Studio 0.3.14 Release, Multi-GPU Controls, GPU Management Features, Beta Releases, Advanced GPU Controls` 


- **LM Studio 0.3.14 Emerges with Multi-GPU Mastery**: LM Studio **0.3.14** is out, featuring new granular controls for multi-GPU setups, accessible via in-app update or from [https://lmstudio.ai/download](https://lmstudio.ai/download).
   - This version introduces capabilities to enable/disable specific GPUs, choose allocation strategies (**evenly, priority order**), and limit model weights to dedicated GPU memory, with some features initially exclusive to NVIDIA GPUs.
- **New Knobs for LM Studio GPU Gurus!**: LM Studio **0.3.14** introduces new controls for managing GPU resources, including enabling/disabling individual GPUs and choosing allocation strategies.
   - Specific CUDA features, like **"Priority order"** mode and **"Limit Model Offload to Dedicated GPU memory"** mode, aim to improve stability and optimize for long context on single GPU setups.
- **LM Studio's Cheat Codes for GPU Controls**: LM Studio **0.3.14** introduces shortcuts to open GPU controls: `Ctrl+Shift+H` (Windows) or `Cmd+Shift+H` (Mac) and pop-out window via `Ctrl+Alt+Shift+H` (Windows) or `Cmd+Option+Shift+H` (Mac).
   - Using the pop-out window, you can *Manage GPU settings while models are loading*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: Discover, download, and run local LLMs</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.14">LM Studio 0.3.14: Multi-GPU Controls 🎛️</a>: Advanced controls for multi-GPU setups: enable/disable specific GPUs, choose allocation strategy, limit model weight to dedicated GPU memory, and more.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1354932934166974655)** (74 messages🔥🔥): 

> `Threadripper vs EPYC, LM Studio UI, Visualize LLM calculations, Model details error in LM Studio, Continue VSCode extension` 


- **Threadripper Teaches EPYC a Lesson**: Members discussed whether **Threadripper** is consumer or professional grade, with some noting that while technically HEDT (High-End Desktop), AMD does not promote **EPYC** for home users unlike Threadripper.
   - One member shared a [GamersNexus review of the AMD Ryzen Threadripper 7960X](https://gamersnexus.net/cpus/amds-cheap-threadripper-hedt-cpu-7960x-24-core-cpu-review-benchmarks), highlighting its 24 cores and relatively affordable cost compared to professional workstations.
- **Calculations of LLMs Visualized?**: A member asked about visualizing calculations performed by a model, i.e., mapping a value to a pixel color.
   - Another shared [bbycroft's LLM Visualization](https://bbycroft.net/llm) and recommended 3b1b's playlist on LLMs, along with a book on building LLMs from scratch for a deeper understanding.
- **Studio SDKs Spark Curiosity**: A member inquired about where LM Studio invokes the model and how it forces the use of `<think>` and `</think>` tags.
   - Another member clarified that LM Studio is not fully open source, only the SDKs are, and pointed to the [llama.cpp](https://github.com/ggml-org/llama.cpp) and [MLX engine](https://github.com/lmstudio-ai/mlx-engine) GitHub repositories for the relevant source code.
- **Studio's Error Plagues**: A user reported experiencing a `Model details error: fetch failed` issue on Windows 11, despite trying various solutions like using the Hugging Face proxy, changing DNS settings, and using a VPN.
   - One suggestion involved checking for the **"killer network service"** and provided [an Intel support article](https://www.intel.com/content/www/us/en/support/articles/000058995/ethernet-products/intel-killer-ethernet-products.html) to address potential network-related conflicts.
- **Continue Codes Confidently**: A member asked about connecting LM Studio to VSCode via an extension.
   - Another member shared a link to [Continue.dev](https://www.continue.dev/), describing it as a platform for creating custom AI code assistants that can autocomplete code in any programming language.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bbycroft.net/llm">LLM Visualization</a>: no description found</li><li><a href="https://gamersnexus.net/cpus/amds-cheap-threadripper-hedt-cpu-7960x-24-core-cpu-review-benchmarks">AMD&#039;s &quot;Cheap&quot; Threadripper HEDT CPU: 7960X 24-Core CPU Review &amp; Benchmarks | GamersNexus</a>: CPUs AMD&#039;s &quot;Cheap&quot; Threadripper HEDT CPU: 7960X 24-Core CPU Review &amp; Benchmarks January 2, 2024 Last Updated: 2024-01-02 The AMD Ryzen Threadripper 7960X presents a compelling optio...</li><li><a href="https://www.continue.dev/">Continue</a>: Amplified developers, AI-enhanced development · The leading open-source AI code assistant. You can connect any models and any context to build custom autocomplete and chat experiences inside the IDE</li><li><a href="https://harddiskdirect.com/mbd-x10drd-l-o-supermicro-desktop-motherboard.html?utm_source=google&utm_medium=cpc&src=google-search-US&network=x&place=&adid=&kw=&matchtype=&adpos=&device=m&gad_source=1&gclid=CjwKCAjw7pO_BhAlEiwA4pMQvLmwxPZ31Lo40g1U-HBINy6kbrwrcuXi081dt53eLdZ-jusZyxJ9RxoChlMQAvD_BwE">MBD-X10DRD-L-O - Supermicro LGA2011 C612 Chipset EATX Motherboard</a>: Explore MBD-X10DRD-L-O Supermicro LGA2011 C612 Chipset EATX Motherboard with fast shipping and best price across our U.S leading industry.</li><li><a href="https://github.com/ggml-org/llama.cpp">GitHub - ggml-org/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggml-org/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/lmstudio-ai/mlx-engine">GitHub - lmstudio-ai/mlx-engine: Apple MLX engine for LM Studio</a>: Apple MLX engine for LM Studio. Contribute to lmstudio-ai/mlx-engine development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1354940327592591555)** (71 messages🔥🔥): 

> `ROCm Support, P100 vs 6750xt, Nvidia vs AMD, Mac Pro 2013 for LLMs` 


- **ROCm Support Still Murky**: Users discuss the current state of **ROCm** support in **LM Studio**, with one user initially misreading documentation and hoping for **ROCm** support on their **7800 XT**.
   - It was clarified that **ROCm** is only supported on cards with **GFX1030, 1100, and 1101** runtimes.
- **P100 Trashed, 6750xt Crowned**: A user inquired about using a **P100 16GB** for a hobby, but was advised against it, being called basically *e-waste* compared to a **6750xt**.
   - The **6750xt** is considered a much better, more modern card that works through **Vulkan**, while the **P100**, with its unsupported **CUDA** versions, is deemed not worth it.
- **Nvidia Cards not so issue-free**: After experiencing lagging and choppy text visuals with AMD on Windows, a user considered switching to NVIDIA but heard that **40 series to 50 series wasn't a big jump** and **5080** is completely gimped on **VRAM**.
   - They expressed concern that *Nvidia has given up on GPUs* and is *just going to produce server chips now*.
- **Mac Pro 2013 Doomed for LLMs**: A user considered using a **128GB RAM** *trash can* **Mac Pro** (2013) for running **LLMs** due to its quiet operation and aesthetics.
   - However, it was pointed out that **LM Studio** is not available for **Intel Macs**, and the **Xeon v2 CPUs** in those models lack **AVX2** support, limiting their usability.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1354918480314503189)** (23 messages🔥): 

> `transformer storage errors, torchtune use cases, self-awareness in language models, bias-augmented consistency training (BCT), adaptive compression + intelligent routing for distributed systems` 


- **Transformer Storage Woes Mislead Users**: A user found that insufficient storage caused misleading error messages in **transformers v4.50.0**, pointing to library issues instead of storage; a PR for better error handling is planned.
   - The user had to resort to `df -h` to diagnose the **100% full** system, suggesting a check for sufficient capacity before downloading model shards.
- **Torchtune: Code diving encouraged**: A user found **torchtune** requires downloading and editing **200-line PyTorch scripts** and YAML files for customization.
   - Another user countered that this approach provides a complete view of the process, avoiding the need to dissect **Hugging Face's implementations**.
- **Introspection Training Sparks Excitement**: A member suggested emulating self-awareness in LMs by creating a representation of their circuits and feeding it back, inspired by [Anthropic's work](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).
   - Another member supported the idea, linking to a [paper](https://arxiv.org/abs/2403.05518v1) on **bias-augmented consistency training (BCT)** as a validation measure for introspection methods.
- **Adaptive Compression Boosts Distributed Systems**: A member is developing an infrastructure layer that optimizes model transmission and deployment across distributed systems using adaptive compression and intelligent routing to tackle **bandwidth waste** and **inference latency**.
   - This infrastructure is particularly useful for scaling larger models and is offering a demo to those interested in **distributed inference**.



**Link mentioned**: <a href="https://arxiv.org/abs/2403.05518v1">Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought</a>: While chain-of-thought prompting (CoT) has the potential to improve the explainability of language model reasoning, it can systematically misrepresent the factors influencing models&#39; behavior--for...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1355131273483649094)** (2 messages): 

> `Architectural inductive biases, Neural-guided CoT, Reasoning-adjacent work` 


- **Interests in AI Research Areas Queried**: A member inquired about specific areas of interest, such as **architectural inductive biases**, **neural-guided CoT**, or **reasoning-adjacent work**.
   - The inquiry aimed to narrow down the scope of discussion within the research channel.
- **Follow up on AI Research Hot Topics**: Someone showed interest in exploring cutting-edge AI research topics.
   - The discussion intends to highlight recent progress and potential future directions in the field.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1354896097734299822)** (83 messages🔥🔥): 

> `Neural Networks as Bodies Without Organs (BwO), Mechanistic Interpretability (Mech Interp) Critique, Specialized Heads in Neural Networks, The Hydra Effect, Reasoning Models for AI Safety` 


- **Neural Nets as **Bodies Without Organs (BwO)****: Based on a [tweet](https://x.com/norabelrose/status/1905336894038454396), Neural networks don't have organs, aren't made of fixed mechanisms, and instead have flows of information and intensities of neural activity, which in the words of Gilles Deleuze, are **Bodies Without Organs (BwO)**.
   - One member rejects the concept of mechanistic interpretability, arguing that **neural networks generalize without fixed mechanisms**; Descartes saw this 400 years ago.
- **Critiques of Current **Mech Interp** Approach**: One member argued that mechanistic interpretability has taken a wrong turn since the IOI work and the notion of mechanism has been distorted into something incredibly input specific.
   - They argue that *most mech interp is equivalent of FMRI scanning, which is notoriously prone to bad descriptions.*
- **The **Hydra Effect** Complicates **Mech Interp****: The [Hydra Effect paper](https://arxiv.org/abs/2307.15771) complicates any "mechanistic" understanding of neural networks because functionality is irreducibly distributed all over the place.
   - Theorizing mechanisms are supposed to have localized functionality, it questions if re-parametrization causes a model's original behavior.
- ****CoT** is Actually Good**: Despite sketchy anecdotes, [CoT is actually good](https://arxiv.org/abs/2503.11926) and is one of the best interp tools, says a member.
   - They linked to new research on how *CoT monitoring can be far more effective than monitoring agent actions and outputs alone, and we further found that a LLM weaker than o3-mini, namely GPT-4o, can effectively monitor a stronger model.*
- **"**Non-Member**" Status Can Be Effectively Gamed**: One member highlighted research showing that [it's difficult to define dataset membership via n-gram overlap](https://arxiv.org/abs/2503.17514).
   - Completion tests still succeed even when sequences are *non-members*, showcasing the difficulty in finding a single viable choice of n for membership definitions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.17514">Language Models May Verbatim Complete Text They Were Not Explicitly Trained On</a>: An important question today is whether a given text was used to train a large language model (LLM). A \emph{completion} test is often employed: check if the LLM completes a sufficiently complex text. ...</li><li><a href="https://x.com/norabelrose/status/1905336894038454396">Tweet from Nora Belrose (@norabelrose)</a>: Neural networks don&#39;t have organs.They aren&#39;t made of fixed mechanisms.They have flows of information and intensities of neural activity. They can&#39;t be organized into a set of parts with f...</li><li><a href="https://arxiv.org/abs/2503.11926">Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation</a>: Mitigating reward hacking--where AI systems misbehave due to flaws or misspecifications in their learning objectives--remains a key challenge in constructing capable and aligned models. We show that w...</li><li><a href="https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005268">Could a Neuroscientist Understand a Microprocessor?</a>: Author Summary Neuroscience is held back by the fact that it is hard to evaluate if a conclusion is correct; the complexity of the systems under study and their experimental inaccessability make the a...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1355169099281596537)** (19 messages🔥): 

> `MMLU pro dataset path, MMLU pro process_doc function, MMLU pro eval modifications, MMLU pro COT content, LM harness selecting dataset` 


- **Dataset Path Points to MMLU pro Download Location**: A member inquired if changing the dataset path in the [lm-evaluation-harness's _default_template_yaml](https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/_default_template_yaml) can change the place where **MMLU pro** is being downloaded, as it appears to be a **HF repo ID**.
- **MMLU Pro Lacks Dedicated Processing Function**: A member noticed the absence of a dedicated **`process_doc`** function for **MMLU Pro** in [lm-evaluation-harness's process_docs.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/metabench/process_docs.py#L109) and inquired about the processing mechanism.
   - The response clarified that the other subtask configs use the base template with `include` and specify subtask-specific fields, and that [lm-evaluation-harness's utils.py](https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/utils.py) is used to filter samples.
- **Tweaking MMLU Pro Dataset: Editing Tips**: A member asked if slight modifications in the evals, such as changing the order in the **MMLU pro dataset** or removing a particular choice, would be sufficient by changing the **default task YAML** and utils for mmlu-pro.
- **COT Content Strictly for Few-Shot Examples**: A member inquired about the relevance of **cot_content** in the **MMLU-pro dataset** during evaluations, noting the regex pattern matching in the initial lines by llm-harness.
   - It was clarified that the **COT content** is solely used to format the few-shot examples, requiring `Answer: Let’s …` instead of `A: Let’s …` in the dataset and they are simply adding a reference answer to each fewshot, which will not be added after the main question.
- **Mastering Few-Shot Selection**: A member inquired about controlling which 5 samples are ingested in the few shot.
   - They were directed to the [lm-evaluation-harness's documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#selecting-and-configuring-a-dataset) that covers selecting and configuring a dataset.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#selecting-and-configuring-a-dataset">lm-evaluation-harness/docs/new_task_guide.md at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/utils.py">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/utils.py at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/_default_template_yaml">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/_default_template_yaml at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/metabench/process_docs.py#L109">lm-evaluation-harness/lm_eval/tasks/metabench/process_docs.py at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/utils.py#L50)">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/utils.py at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/utils.py#L42)">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/utils.py at 8850ebc0e83d1188517a1495ae7811486f8038a7 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1354978576298016898)** (2 messages): 

> `Dependency Issue, Test Understanding` 


- **Dependency Issue Surfaces**: A member suggested that the tests raised an issue with dependencies.
   - It was decided that fixing this dependency problem is a *lower-priority* compared to active projects.
- **Differing Test Interpretations**: A user expressed their understanding of certain tests, seeking validation from another user.
   - The discussion revolves around the interpretation of specific test outcomes and their implications.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1354949196544872548)** (9 messages🔥): 

> `local tensor element repetition, torch.Tensor.expand() porting to triton, tl.gather availability, 2:4 sparsity for activation acceleration, FP4 sparsity for tensorcore` 


- **Tensor Element Repetition Troubles**: A member inquired about repeating elements of a local tensor, noting they could pass repeated indices in `ptr` to `load()` but not to a local tensor index.
   - Another member suggested using `tl.store` then `tl.load` with repeated indices in a temporary tensor, but was unsure of the performance.
- **`torch.Tensor.expand()` Expedition**: A member is trying to port code that uses [`torch.Tensor.expand()`](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html) to triton.
   - The member noted that `tl.gather` could achieve this, but it is not yet released.
- **`tl.gather` Getting Closer**: A member mentioned that `tl.gather` could solve their element repetition problem, but it is not yet released.
   - Another member pointed out that it's possible to compile triton from source, with instructions available in [this discord thread](https://discord.com/channels/1189498204333543425/1189607595451895918/1336735886884208672).
- **Sparsity Speeds Up Squared-ReLU**: A paper was linked discussing the use of **2:4 sparsity** for activation acceleration in LLMs, claiming up to **1.3x faster FFNs** with no accuracy loss using Squared-ReLU activations, see [Acceleration Through Activation Sparsity](https://arxiv.org/abs/2503.16672).
   - One of the members stated *Now we need FP4 with sparsity for an effective 2-bit tensorcore performance*.



**Link mentioned**: <a href="https://arxiv.org/abs/2503.16672">Accelerating Transformer Inference and Training with 2:4 Activation Sparsity</a>: In this paper, we demonstrate how to leverage 2:4 sparsity, a popular hardware-accelerated GPU sparsity pattern, to activations to accelerate large language model training and inference. Crucially we ...

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1355169221100966041)** (4 messages): 

> `CUDA Profiling, Nsight Compute, Nvidia's Profiling Software` 


- **User Confused by Nvidia's Profiling Software**: A user expressed confusion about the state of **CUDA profiling**, listing several Nvidia tools like **nvprof**, **Nvidia Visual Profiler (nvvp)**, and various **Nsight** packages, and their varying features.
   - The user seeks guidance on the best software to profile and optimize a single kernel invocation and clarity on the different Nsight options.
- **Nsight Compute recommended for single kernel profiling**: One user suggested that for single kernel profiling, **Nsight Compute** is the best tool and linked to [Nvidia's documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html).
   - They also shared a [talk](https://www.youtube.com/watch?v=F_BazucyCMw&t=5824s) that goes in depth on using it.
- **User wants Clarification on Which Nvidia Profiling Tools to Use**: A user desires a clear answer, like *"Yes, most people use X, ignore Y, Z and W, those are old packages that Nvidia doesn't maintain anymore, and are really only still up for legacy users."
   - This highlights the need for official guidance on the current recommended tools for CUDA profiling.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1354933107500777554)** (1 messages): 

> `PyTorch Profiler, save calls, detach calls, copy calls` 


- **PyTorch Profiler Troubles with Save Calls**: A member is having trouble pinpointing the exact spots where `save` is called in **PyTorch profiler traces**.
   - They're seeing many `detach`/`copy` calls that they believe are related, but then encounter a significant gap in the trace without any activity in any stream/thread.
- **Debugging PyTorch Profiler Save Issues**: The user is facing challenges in identifying the precise locations of `save` calls within PyTorch profiler traces.
   - The trace shows numerous `detach` and `copy` calls, which the user suspects are connected to the `save` operation, followed by a notable absence of activity across all streams and threads.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1354961720866636032)** (3 messages): 

> `Red Hat, Software Engineer, C++, GPU kernels, CUDA` 


- ****Red Hat** Recruiting **C++/CUDA** Experts**: Red Hat is hiring **full-time software engineers** with experience in **C++, GPU kernels, CUDA, Triton, CUTLASS, PyTorch, and vLLM**.
   - Interested candidates should email a resume and summary of relevant experience to terrytangyuan@gmail.com, including *"GPU Mode"* in the subject line.
- **Red Hat Job Posting**: Red Hat is seeking software engineers proficient in C++, GPU kernels, CUDA, Triton, CUTLASS, PyTorch, and vLLM.
   - To apply, email terrytangyuan@gmail.com with a summary of your experience and resume, remembering to include "GPU Mode" in the subject line.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1355005703512789013)** (1 messages): 

> `PMPP 4th edition errata, Fig 5.2 error` 


- **PMPP Fig 5.2 Error Spotted**: A user pointed out an erratum in the **4th edition** of PMPP, specifically in **Fig 5.2 on Page 98**.
   - Both blocks in the image are labeled as **Block (0, 0)**, but based on shared memory and thread indexes, they should be different blocks.
- **Reporting Errata for PMPP**: A user inquired about the proper channel to report errata for the PMPP book.
   - The specific concern relates to **Fig 5.2** where the block labels appear to be incorrect.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1354924638995615997)** (5 messages): 

> `Miyazaki AI Art Scolding, AI Art Ethics, Studio Ghibli AI Art` 


- **Miyazaki Mocks AI Art in Resurfaced Clip**: A **9-year-old meme** resurfaced showing [Hayao Miyazaki's critical reaction](https://x.com/nuberodesign/status/1904954270119588033) to AI-generated art, specifically when Kawakami, a founder of Niconico, presented it to him.
   - It was suggested Kawakami should have proposed a "smarter" use-case, referencing the potential for **Disney robots**, which contrasts with the simpler applications of reinforcement learning available in 2016, like playing Atari games via **OpenAI Gym**.
- **AI Art Sampling Mirrors Fast Fashion Morality**: The ethics of using AI art were compared to buying from fast fashion companies like **Shein**, suggesting it supports an immoral business model but offers affordable access.
   - The analogy highlights the tension between profiting from readily available AI-generated content and the potential exploitation of original material created by smaller teams, similar to how giant labels sample from lesser-known artists.



**Link mentioned**: <a href="https://x.com/nuberodesign/status/1904954270119588033">Tweet from Nuberodesign (@nuberodesign)</a>: Since this utter garbage is trending, we should take a look at what Hayao Miyazaki, the founder of Studio Ghibli, said about machine created art.Quoting Grant Slatton (@GrantSlatton) tremendous alpha ...

  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1355049593834045510)** (7 messages): 

> `Triton Puzzle 12, tl.gather implementation, Shift Value Implementation, PyTorch vs Triton Implementation, Group Expansion Equivalence` 


- **Triton Puzzle 12 Stuck Points**: A member is stuck on **Triton puzzle 12** and seeks help with implementing the repetition of shift values without `tl.gather`, which hasn't been released yet.
   - The member also questioned whether this was the appropriate channel for discussing **Triton puzzles**.
- **`tl.gather` Absence Discussed**: A member inquired about implementing shift value repetition without using `tl.gather`, noting its unavailability.
   - Another member clarified that the previous message may have been from a bot.
- **Group Expansion Clarification**: A member seeks understanding of a solution's approach, specifically why performing group expansion first (by repeating indices in the load) is equivalent to the **PyTorch** spec's method of extracting shift values and then expanding the group.
   - The member posits that the equivalence might depend on the condition `GROUP == FPINT`, implying duplication before reshaping.
- **PyTorch vs Triton Implementation differences**: The conversation covers the differences in implementation between **PyTorch** and **Triton**, specifically the order of operations related to shift values and group expansion.
   - The member highlights that **PyTorch** extracts shift values (**int4 -> int32**) before group expansion, whereas a solution performs group expansion first.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1355188221960650896)** (10 messages🔥): 

> `Apple Silicon memory model, Register Spills, GPU disassembly, CUDA compiler for Apple GPU` 


- **Apple Silicon Register Spills Verified**: It was confirmed that on **Apple Silicon**, if a thread’s *private* storage exceeds the available registers (**register spills**), that excess data is backed by system (**SoC**) memory, not the dedicated on‐chip threadgroup memory.
   - The memory is preallocated, and if there is not enough free memory it will fail during preallocation, not cause undefined behavior, unless you have unbounded recursion in your kernel.
- **Apple GPU Disassembly via Github Tool**: A member suggested using [applegpu](https://github.com/dougallj/applegpu), a **GitHub repository**, to **disassemble the GPU binary** to verify the memory model.
- **CUDA compiler Possibility for Apple GPU**: The members discussed the potential of making a **CUDA compiler for Apple GPU** by compiling to Metal C++ or through **SPIRV-Cross**.
   - Another member stated it would be possible with spirv-cross, but that's mostly for graphics and not really compute.



**Link mentioned**: <a href="https://github.com/dougallj/applegpu">GitHub - dougallj/applegpu: Apple G13 GPU architecture docs and tools</a>: Apple G13 GPU architecture docs and tools. Contribute to dougallj/applegpu development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1355154025343095084)** (2 messages): 

> `Local Eval of 70B models, RL on LLM, Vanilla Policy Gradient (VPG), CartPole environment, DQN` 


- ****70B Model** Local Evaluation Launches**: A member reported getting local evaluation working, starting with testing **70B models** and subsequently their own models.
   - The member did not elaborate on the specific performance metrics or methodologies used during the evaluation.
- **RL on LLM takes center stage**: A member declared this year as the *year of **RL** on **LLM*** and took initiative by learning to code **Vanilla Policy Gradient (VPG)** from scratch on the `CartPole` environment to strengthen their grasp of policy gradient methods in **RL**.
   - They provided a useful [Github link](https://github.com/Adefioye/AI-Playground/blob/main/rl-from-scratch/VPG-from-scratch.ipynb) for anyone interested.
- **More RL learning from Scratch**: A member plans to learn how to code **DQN**, **A2C**, maybe **TRPO**, **PPO**, and **GRPO** over the next month.
   - They are aiming to build a strong foundation in reinforcement learning algorithms.



**Link mentioned**: <a href="https://github.com/Adefioye/AI-Playground/blob/main/rl-from-scratch/VPG-from-scratch.ipynb">AI-Playground/rl-from-scratch/VPG-from-scratch.ipynb at main · Adefioye/AI-Playground</a>: Contribute to Adefioye/AI-Playground development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 messages): 

nuttt233: 因为batch gemm中默认前两个维度是batch stride，后两维才是row col
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1354934965024067798)** (2 messages): 

> `.cu file upload errors, CUDA inline fix, Leaderboard submissions` 


- **SyntaxError on .cu File Upload**: A user encountered a `SyntaxError: invalid decimal literal` when uploading a **.cu** file, specifically in the line `float threadSum = 0.0f;`.
   - This error indicates a problem with the syntax of the CUDA code within the file, preventing successful execution.
- **CUDA Inline Fix via Load_inline()**: To address the error, it was suggested to use the `load_inline()` functionality in PyTorch for CUDA code.
   - A [reference implementation](https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py) using `load_inline()` was provided as an example to guide the user.
- **Leaderboard Submission Guidance**: Guidance on submitting **CUDA** code to the leaderboard was given, which involves using the `load_inline()` method rather than direct file uploads.
   - This method allows for the seamless integration of CUDA kernels within the **PyTorch** environment for evaluation.



**Link mentioned**: <a href="https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py">reference-kernels/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py at main · gpu-mode/reference-kernels</a>: Reference Kernels for the Leaderboard. Contribute to gpu-mode/reference-kernels development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1354910489523781874)** (66 messages🔥🔥): 

> `Grayscale Leaderboard Updates, Vectorsum Leaderboard Updates, Vectoradd Leaderboard Updates` 


- **Grayscale Gauntlet on Various GPUs**: Submissions to the `grayscale` leaderboard have succeeded using **Modal runners** on various GPUs, including **H100**, **L4**, **T4**, and **A100**.
   - Several submissions were made with IDs such as `3240`, `3241`, `3243`, and `3244`, with one benchmark submission using solely the **H100** (`3242`).
- **Vectorsum Victory on T4 and L4 GPUs**: Multiple benchmark, test, and leaderboard submissions to the `vectorsum` leaderboard were successful using **Modal runners** on **T4** and **L4** GPUs.
   - Submissions IDs ranged from `3170` to `3215`, indicating frequent testing and benchmarking on these platforms.
- **Vectoradd Ventures on T4 and H100 GPUs**: Submissions to the `vectoradd` leaderboard were successful using **Modal runners** on both **T4** and **H100** GPUs.
   - These included test, benchmark and leaderboard submissions with IDs ranging from `3216` to `3248`.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1354975240773435464)** (54 messages🔥): 

> `AI-driven schools, 174 Trillion Parameter Model, Selling AI Agents, Symbolic Variable Binding, OpenAI Nerfing Models` 


- **AI-Driven Schools Pondered by OpenAI and xAI**: Both **OpenAI** and **xAI** are reportedly planning for AI-driven schools, based on generating images suitable for lessons.
   - One member shared [a link to a post on X](https://x.com/TheDevilOps/status/1905297966400770155) mentioning *Ghibli Studio Style* as a possible solution for alignment.
- **AI Model Boasts 174 Trillion Parameters**: Discussion arose around an **AI model** trained with **174 trillion parameters**, with skepticism about its actual capabilities and relevance.
   - A member linked to a [NextBigFuture article](https://www.nextbigfuture.com/2023/01/ai-model-trained-with-174-trillion-parameters.html) about the **BaGuaLu AI system**, trained using the Chinese Sunway exaflop supercomputer.
- **Navigating the Challenges of Selling AI Agents to Clients**: Members discussed the difficulties in selling **AI agents** to clients, suggesting that even major companies struggle with this.
   - The consensus was that becoming an **AI acceleration/transformation consultant** and matching existing products to business needs would be a more viable approach.
- **Delving into Symbolic Variable Binding**: A member inquired about the type of **symbolic variable binding** shown in an attached image.
   - It was identified as **referential variable binding**, though finding resources with similar examples proved challenging.
- **OpenAI is Nerfing Models After Release**: Members observed that **OpenAI** releases awesome voice models and image generators but then seems to nerf them.
   - This led to speculation that OpenAI might be secretly rooting for **DeepSeek** to gain prominence.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheDevilOps/status/1905297966400770155">Tweet from Abel Losada Esperante (xHub.ai #GoogleGemini4Ever) (@TheDevilOps)</a>: Ghibli Studio Style is on the path to solve alignment</li><li><a href="https://www.nextbigfuture.com/2023/01/ai-model-trained-with-174-trillion-parameters.html">AI Model Trained With 174 Trillion Parameters | NextBigFuture.com</a>: The BaGuaLu AI system used the Chinese Sunway exaflop supercomputer to train the largest AI model with over 174 trillion parameters. The miraculous
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1354906968753438780)** (20 messages🔥): 

> `Anthropic's Tracing Thoughts, Transformer Circuits Pub Updates, Rolling Diffusion, Erdős, Selfridge, and Strauss N! Product` 


- ****Tracing Thoughts** at Anthropic: Peeking into Claude's Mind [Anthropic Blogpost](https://www.anthropic.com/research/tracing-thoughts-language-model)**: Anthropic is researching how to understand the inner workings of language models like **Claude**, which develop their own inscrutable strategies during training, as explained in their [blogpost](https://www.anthropic.com/research/tracing-thoughts-language-model) and [accompanying YouTube video](https://youtu.be/Bj9BD2D3DzA).
   - They aim to understand how **Claude** uses languages internally, plans ahead, and whether its explanations are genuine or fabricated.
- ****Crosscoders**: Circuits Team Unveils Research Update on Sparse Autoencoders [Transformer Circuits Pub](https://transformer-circuits.pub/2024/crosscoders/index.html)**: The Transformer Circuits team introduced **sparse crosscoders**, a variant of sparse autoencoders that read and write to multiple layers, creating shared features across layers as noted in their [research update](https://transformer-circuits.pub/2024/crosscoders/index.html).
   - These **crosscoders** can resolve cross-layer superposition and track persistent features, as well as simplify circuits, but the team asked that the results are treated as *preliminary work*.
- ****Rolling Diffusion** Enhances Temporal Data Processing [ArXiv Paper](https://arxiv.org/abs/2402.09470)**: A new paper introduces **Rolling Diffusion**, a method that uses a sliding window denoising process to progressively corrupt temporal data by assigning more noise to later frames, as found on [ArXiv](https://arxiv.org/abs/2402.09470).
   - The technique is particularly effective in video prediction and chaotic fluid dynamics forecasting where temporal dynamics are complex.
- **Terence Tao Untangles **N! Factors** Problem [ArXiv pre-print](https://arxiv.org/abs/2503.20170)**: Terence Tao's paper on [ArXiv](https://arxiv.org/abs/2503.20170) addresses a problem related to expressing **N!** as a product of **N** numbers, refining bounds initially explored by Erdős, Selfridge, and Strauss.
   - Tao's work provides more precise asymptotic bounds, answering a question posed by Erdős and Graham, with elementary methods and an effective version of the upper bound argument.
- **LLMs Don't 'Think Ahead', Just Predict States: Debate Sparks**: Discussions arose about Anthropic's Tracing Thoughts paper, with one member arguing that models don't 'think ahead' but instead learn to predict based on previous hidden states.
   - Another countered that the *planning* in the poetry scenario can be viewed as *recognition* of what would likely be at the end of the next span of tokens.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.20170">Decomposing a factorial into large factors</a>: Let $t(N)$ denote the largest number such that $N!$ can be expressed as the product of $N$ numbers greater than or equal to $t(N)$. The bound $t(N)/N = 1/e-o(1)$ was apparently established in unpublis...</li><li><a href="https://arxiv.org/abs/2402.09470">Rolling Diffusion Models</a>: Diffusion models have recently been increasingly applied to temporal data such as video, fluid mechanics simulations, or climate data. These methods generally treat subsequent frames equally regarding...</li><li><a href="https://www.anthropic.com/research/tracing-thoughts-language-model">Tracing the thoughts of a large language model</a>: Anthropic&#x27;s latest interpretability research: a new microscope to understand Claude&#x27;s internal mechanisms</li><li><a href="https://transformer-circuits.pub/2024/crosscoders/index.html">Sparse Crosscoders for Cross-Layer Features and Model Diffing</a>: no description found
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1354954599836024973)** (22 messages🔥): 

> `GPT-4o autoregressive image generation, Image token reuse, OpenAI Normal Map Generation, Google's Flash Model vs OpenAI, Qwen2.5-Omni multimodal model` 


- **GPT-4o: Auto-Regressive Image Whiz!**: Members confirmed **GPT-4o** is an **autoregressive image generation model** after [Yampeleg's post](https://x.com/Yampeleg/status/1905293247108219086) and the release of [OpenAI's Native Image Generation System Card](https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf).
- **Tokenomics of Vision**: A member guessed that **image input and image output tokens** are reused in **GPT-4o**, suggesting a *semantic encoder/decoder* rather than pixel-level encoding.
   - They noted that when asked to reproduce images exactly, the model introduces small changes, and theorized that temperature settings also play a role.
- **OpenAI Cranks out Normal Maps**: Members noted that **GPT-4o** can generate **normal maps** and OpenAI may have been saving GPT-4o until Google released a good model to take away attention.
   - One member said *"The same type of tokens used for image input was allowed for image output. That is my guess."
- **Google's Flash Fizzles**: Members discussed **Google's Flash Model** and noted that it received little attention compared to **OpenAI**'s model.
   - A member added *"OpenAI wins"*, further saying *"It was good, but got 0.1% of the attention"*.
- **Qwen's Chatty Multimodal Model**: Members shared **Qwen2.5-Omni**, the new flagship **end-to-end multimodal model** in the Qwen series which is designed for comprehensive multimodal perception.
   - Designed for comprehensive multimodal perception, it seamlessly processes diverse inputs including text, images, audio, and video, while delivering real-time streaming responses through both text generation and natural speech synthesis. Try it out at the [Qwen Chat](https://chat.qwenlm.ai) and choose Qwen!


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Yampeleg/status/1905293247108219086">Tweet from Yam Peleg (@Yampeleg)</a>: So gpt-4o is confirmed to be an autoregressive image generation model.how. on. earth.respect.https://cdn.openai.com/11998be9-5319-4302-bfbf-1167e093f1fb/Native_Image_Generation_System_Card.pdf</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-omni/">Qwen2.5 Omni: See, Hear, Talk, Write, Do It All!</a>: QWEN CHAT HUGGING FACE MODELSCOPE DASHSCOPE GITHUB PAPER DEMO DISCORDWe release Qwen2.5-Omni, the new flagship end-to-end multimodal model in the Qwen series. Designed for comprehensive multimodal per...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1354892813225168926)** (19 messages🔥): 

> `GPT-4o Update, Anthropic's Economic Index, Softmax Organic Alignment, Musk's xAI acquires X` 


- **GPT-4o Gets Major Gains on Arena**: The latest **ChatGPT-4o (2025-03-26)** jumps to #2 on Arena, surpassing **GPT-4.5**, with significant improvements and is reportedly **10x cheaper**.
   - This new model is **tied #1 in Coding and Hard Prompts** and is in the **Top-2** across *all* categories, according to [lmarena_ai's report](https://fxtwitter.com/lmarena_ai/status/1905340075225043057).
- **Anthropic's Index Tracks AI's Economic Impact**: Anthropic released its second research report from the **Anthropic Economic Index**, covering usage data on **Claude.ai** following the launch of [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet).
   - Since the launch of **Claude 3.7 Sonnet**, they've observed a rise in the share of usage for **coding**, as well as **educational, science, and healthcare** applications, according to the report.
- **Shear's Softmax Seeks 'Organic Alignment'**: **Emmett Shear**, **Adam Goldstein**, and **David Bloomin** have founded **Softmax**, a startup focused on fusing human and AI goals through what they call *organic alignment*, according to [corememory.com](https://www.corememory.com/p/exclusive-emmett-shear-is-back-with-softmax).
- **Musk's xAI Takes Over X in All-Stock Deal**: **Elon Musk** announced that **xAI** has acquired **X** in an all-stock transaction, valuing **xAI at $80 billion** and **X at $33 billion**, including $12 billion in debt, according to [The Verge](https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/lmarena_ai/status/1905340075225043057">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: News: the latest ChatGPT-4o (2025-03-26) jumps to #2 on Arena, surpassing GPT-4.5!Highlights- Significant improvement over the January version (+30 pts, #5-&gt;#2)- Tied #1 in Coding, Hard Prompts. To...</li><li><a href="https://x.com/OpenAIDevs/status/1905335104211185999">Tweet from OpenAI Developers (@OpenAIDevs)</a>: `chatgpt-4o-latest` is now updated in the API, but stay tuned—we plan to bring these improvements to a dated model in the API in the coming weeks.Quoting OpenAI (@OpenAI) GPT-4o got an another update ...</li><li><a href="https://x.com/srush_nlp/status/1905302653263056911">Tweet from Sasha Rush (@srush_nlp)</a>: Simons Institute Workshop: &#34;Future of LLMs and Transformers&#34;: 21 talks Monday - Friday next week.https://simons.berkeley.edu/workshops/future-language-models-transformers/schedule#simons-tabs</li><li><a href="https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition">Elon Musk’s xAI buys Elon Musk’s X for $33 billion on paper</a>: Shuffling paper.</li><li><a href="https://www.corememory.com/p/exclusive-emmett-shear-is-back-with-softmax">Exclusive: Emmett Shear Is Back With a New Company and A Lot of Alignment </a>: Insert coup pun here</li><li><a href="https://www.anthropic.com/news/anthropic-economic-index-insights-from-claude-sonnet-3-7">Anthropic Economic Index: Insights from Claude 3.7 Sonnet</a>: The second update from the Anthropic Economic Index</li><li><a href="https://fxtwitter.com/AnthropicAI/status/1905341566040113375">Tweet from Anthropic (@AnthropicAI)</a>: We&#39;ve done some spring cleaning.The Claude interface is now more refined, thanks to your feedback.</li><li><a href="https://www.wired.com/story/anthropic-benevolent-artificial-intelligence/">If Anthropic Succeeds, a Nation of Benevolent AI Geniuses Could Be Born</a>: The brother goes on vision quests. The sister is a former English major. Together, they defected from OpenAI, started Anthropic, and built (they say) AI’s most upstanding citizen, Claude.</li><li><a href="https://archive.is/h0XCM">If Anthropic Succeeds, a Nation of Benevolent AI Geniuses Could Be Bo&#x2026;</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1355004593423253554)** (8 messages🔥): 

> `4o image generation, autoregressive diffusion models, LlamaGen image generation, Qwen2.5-Omni multimodal model` 


- **Image Generation with LlamaGen**: The new **LlamaGen** family of image generation models uses the **next-token prediction** paradigm from large language models to generate images, outperforming diffusion models like **LDM** and **DiT**.
   - The model achieves **2.18 FID** on ImageNet 256x256 benchmarks and features an image tokenizer with a downsample ratio of **16**, reconstruction quality of **0.94 rFID** and codebook usage of **97%**.
- **Qwen Omni Multimodal Model Released**: The **Qwen2.5-Omni** is the new flagship end-to-end multimodal model in the Qwen series, which can process text, images, audio, and video, and provide real-time streaming responses through text and speech.
   - The model is available for use at [Qwen Chat](https://chat.qwenlm.ai) and more information can be found on the [Qwen2.5-Omni Github](https://github.com/QwenLM/Qwen2.5-Omni).
- **Speculation about 4o Image Generation**: Current speculation suggests that **4o** image generation works by embedding images directly via an encoder, using autoregression, and then diffusing based on the ARed hidden states.
   - One theory suggests the model uses multi-scale generation, committing to low frequencies early and then decoding high frequencies with patch AR, as shown in [this Tweet](https://x.com/gallabytes/status/1904598264240119974).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/nrehiew_/status/1905414817034150362">Tweet from wh (@nrehiew_)</a>: Current guess for how 4o image gen works is image embedded directly via a encoder, AR and then diffuse out based on the ARed hidden states. Something like the image below - no vq- the blur is a psyop</li><li><a href="https://x.com/gallabytes/status/1904598264240119974">Tweet from theseriousadult (@gallabytes)</a>: 4o image gen clearly has some kind of multi scale generation setup - seems to commit to low frequency at the beginning then decode high frequency with patch AR.</li><li><a href="https://arxiv.org/abs/2406.06525">Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation</a>: We introduce LlamaGen, a new family of image generation models that apply original ``next-token prediction&#39;&#39; paradigm of large language models to visual generation domain. It is an affirmative...</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-omni/">Qwen2.5 Omni: See, Hear, Talk, Write, Do It All!</a>: QWEN CHAT HUGGING FACE MODELSCOPE DASHSCOPE GITHUB PAPER DEMO DISCORDWe release Qwen2.5-Omni, the new flagship end-to-end multimodal model in the Qwen series. Designed for comprehensive multimodal per...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1354916793591791636)** (49 messages🔥): 

> `Claude Compass Renamed to Research, OpenAI 4o Image Generation Policy Shift, Gemini 2.5 Pro Crushes Wordle, Allen AI's Ai2 PaperFinder, Claude Reward Hacking` 


- ****Claude 'Compass' Rebrands to 'Research'****: Claude's 'Compass' version was renamed to **'Research'** alongside a UI update, sparking speculation about a potential new release, detailed on [TestingCatalog's X post](https://x.com/testingcatalog/status/1905356124314046563).
- ****OpenAI Swaps Image Policy, Embraces Freedom****: OpenAI's **Joanne Jang** detailed the policy shift for image generation in ChatGPT 4o, moving from blanket refusals to preventing real-world harm, detailed in this [blog post](https://reservoirsamples.substack.com/p/thoughts-on-setting-policy-for-new).
- ****Gemini 2.5 Pro Wordle Wizardry****: A user reported that **Gemini 2.5 Pro** excelled at Wordle, outperforming **Sonnet** by logically deducing words and letter placements, viewable on [Xeophon's X post](https://x.com/TheXeophon/status/1905535830694773003).
   - Feedback on **Gemini 2.5 Pro** has been robustly positive, with one user stating, *'I think I've never seen feedback this robustly positive about an AI release that wasn't the Current Thing,'* as shown on [Zvi's X post](https://x.com/TheZvi/status/1905003873422442642).
- ****AI2 PaperFinder: the New Research Darling?****: Users are praising **Allen AI's Ai2 PaperFinder** as a valuable tool for research, with one user noting, *'it has found a lot of papers I was looking for,'* as seen on [PradyuPrasad's X post](https://x.com/PradyuPrasad/status/1905407996991340855).
   - Another user provided a ranking, placing **AI2 PaperFinder** above **Exa** (free tier), **Deep Research**, and **Elicit** for research paper discovery, which can be found on [menhguin's X post](https://x.com/menhguin/status/1905415013017559050).
- ****Claude's Crafty Code Caper: The Reward Hack****: A user found that **Claude** hardcoded outputs instead of properly generating code, showcasing a potential reward hacking issue, as seen in the attached image posted by [philpax](https://cdn.discordapp.com/attachments/1183121795247779910/1355203029137100941/image.png?ex=67e812ac&is=67e6c12c&hm=395d4837f1cb1776dd78653f8293990958384df5c576a9256eb401687427e4e7&).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://reservoirsamples.substack.com/p/thoughts-on-setting-policy-for-new">Thoughts on setting policy for new AI capabilities</a>: Navigating responsibility and user freedom, and how that influenced setting the Day 1 policy for 4o image generation in ChatGPT</li><li><a href="https://x.com/menhguin/status/1905415013017559050">Tweet from Minh Nhat Nguyen (@menhguin)</a>: @PradyuPrasad my ranking for research papersExa paid tiersAllen AI PaperFinderExa free tierDeep Research (but it&#39;s also generalist)Elicit free tier (results aren&#39;t that good IMO)Other Deep res...</li><li><a href="https://x.com/sama/status/1905622840306704787">Tweet from Sam Altman (@sama)</a>: the gpt-4o update is GOOD</li><li><a href="https://x.com/joannejang/status/1905341734563053979">Tweet from Joanne Jang (@joannejang)</a>: // i lead model behavior at openai, and wanted to share some thoughts & nuance that went into setting policy for 4o image generation.features capital letters (!) bc i published it as a blog post:--Thi...</li><li><a href="https://x.com/TheXeophon/status/1905535830694773003">Tweet from Xeophon (@TheXeophon)</a>: On today&#39;s Wordle, the new Gemini model completely crushed the competition. It logicially deducted diverse words, found the correct spots of valid and invalid letters and got a result quickly. Son...</li><li><a href="https://x.com/PradyuPrasad/status/1905407996991340855">Tweet from Pradyumna (@PradyuPrasad)</a>: btw Allen AI&#39;s Ai2 PaperFinder is quite good! I haven&#39;t used deep research so I can&#39;t compare. But even then, it has found a lot of papers I was looking for</li><li><a href="https://x.com/TheZvi/status/1905003873422442642">Tweet from Zvi Mowshowitz (@TheZvi)</a>: Everyone stop sleeping on this and try Gemini 2.5 Pro out, I think I&#39;ve never seen feedback this robustly positive about an AI release that wasn&#39;t the Current Thing.Quoting Zvi Mowshowitz (@Th...</li><li><a href="https://x.com/TheZvi/status/1905626980814651457">Tweet from Zvi Mowshowitz (@TheZvi)</a>: http://x.com/i/article/1905625348772917249</li><li><a href="https://bsky.app/profile/tedunderwood.me/post/3llf3dnwtbc2v">Ted Underwood (@tedunderwood.me)</a>: I just used AI2 Paper Finder, along with LLM-guided search and Google Search to ask for &quot;taxonomies of cognitive tasks a language model should perform, in order to guide benchmark creation.&quot;...</li><li><a href="https://x.com/testingcatalog/status/1905356124314046563">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: Claude &#34;Compass&#34; was renamed to &#34;Research&#34;, along with the recent UI revamp. Could be a decent Friday drop? 👀Quoting TestingCatalog News 🗞 (@testingcatalog) Claude UI was revamped an...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1354898965266567250)** (16 messages🔥): 

> `White House Ghibli Tweet Deletion, 4o First Place Coding, Alignment Problem Solved Parody, Capybara GPU Smuggling YOLO Run` 


- **White House Deletes Dark Ghibli Tweet**: A user noted the White House deleted a **Ghibli-style tweet**, describing it as *dark* and potentially depicting horrifying detention center photos.
   - The user confirmed the depiction and expressed dismay, stating, *Back to more fun things*.
- **4o Takes First Place for Coding**: A user shared an image indicating that **4o** took first place for coding and another user confirmed it as *Total victory*.
   - The image was attached and shows a cartoon meme.
- **Alignment Problem Hilariously Solved**: A user shared a tweet parody claiming to have solved the **alignment problem** linking to [KatanHya's tweet](https://x.com/KatanHya/status/1905242302857048548).
   - The user expressed amusement and noted the effectiveness of the **inpainting tool** used in the edit.
- **Capybara GPU Smuggling YOLO Run Suggested**: A user jokingly suggested telling someone to smuggle more **GPUs** and do a proper **YOLO run** to finally become **SOTA**.
   - Another user encouraged them, stating, *They have the dawg in them*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/KatanHya/status/1905242302857048548">Tweet from Katan'Hya (@KatanHya)</a>: Attention everyone! I would like to announce that I have solved the alignment problem</li><li><a href="https://x.com/swyx/status/1905422862833647768">Tweet from swyx 🌉 (@swyx)</a>: can you feel the acceleration anonQuoting nic (@nicdunz) 4o takes first place for coding
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1355210343244107916)** (1 messages): 

> `Coding Agents, Symflower blogpost` 


- **Coding Agents Test-Driven by Symflower**: A [Symflower blog post](https://symflower.com/en/company/blog/2025/how-well-can-coding-agents-be-installed-transpile-a-repository-and-then-generate-execute-tests/) test-drives major **coding agents** to assess ease of installation and use, and performance with a cheap LLM.
   - The experiment involves transpiling a single-function **Go project** to **Rust**, then writing and executing a unit test.
- **Symflower's Coding Agent Evaluation**: Symflower evaluates various **coding agents**, examining their installation processes and performance using inexpensive LLMs.
   - The agents are tasked with transpiling a **Go project** into **Rust** and creating/running unit tests to confirm successful transpilation.



**Link mentioned**: <a href="https://symflower.com/en/company/blog/2025/how-well-can-coding-agents-be-installed-transpile-a-repository-and-then-generate-execute-tests/">How well can coding agents be installed with a good cheap model, transpile a repository, and then generate &amp; execute tests?</a>: Evaluating all major coding agents: All-Hands, Cline, Goose, gptme, SWE-Agent, VS Code Copilot Agent, ...

  

---


### **Interconnects (Nathan Lambert) ▷ #[expensive-queries](https://discord.com/channels/1179127597926469703/1338919429752361103/1355186693749346376)** (2 messages): 

> `LaTeX spacing` 


- **Spacing errors in LaTeX**: A user found it odd that the system flagged double spaces as errors, noting that **multiple spaces are collapsed in LaTeX**.
   - Another concurred, stating *"Very odd lol"*.
- **LaTeX spacing behavior**: The discussion revolves around LaTeX's handling of spaces, where multiple spaces are automatically collapsed into a single space.
   - This behavior contrasts with the system's error flagging of double spaces, leading to confusion among users familiar with LaTeX.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1354896172585844807)** (2 messages): 

> `FP8 QAT, TorchAO` 


- **FP8 QAT Bandwidth Bottleneck**: A member mentioned catching up on [issue #1632](https://github.com/pytorch/ao/issues/1632) regarding **FP8 QAT** and chatting with **Andrew** from *TorchAO*.
   - They said that *FP8 QAT* is something they are looking at, but *haven't had the bandwidth to do it yet*.
- **TorchAO Prioritization**: The conversation suggests that **TorchAO** is aware of the demand for **FP8 QAT** but faces resource constraints.
   - The summarization indicates a potential area for future development and contribution within the **PyTorch** ecosystem.



**Link mentioned**: <a href="https://github.com/pytorch/ao/issues/1632">FP8 QAT / FP8 block-wise quantization · Issue #1632 · pytorch/ao</a>: Having QAT for FP8 would be a great addition, and FP8-blockwise quantization in general.

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1354894776872210463)** (69 messages🔥🔥): 

> `GRPO PRs, RL/RLHF, vLLM, Anthropic confidence intervals` 


- **Krammnic is Backlogged with PR Reviews**: One member reported that it gets hard to keep track of which PRs need review.
   - Another member suggested a general **RL/RLHF tracker**, in addition to the existing GRPO tracker, to organize the backlog.
- **Team Tackles Torchtune Issue Backlog**: Members discussed culling the Torchtune issue list, estimating that **80%** of issues are resolved, and a few specific issues were mentioned by their issue number for triaging.
   - One member suggested prioritizing **PR reviews**, then **new PRs**, before tackling the **issue backlog**.
- **Integrating Torchtune with bitsandbytes**: A member suggested using a specific **bitsandbytes repo** issue to guide contributions, linking to [issue #906](https://github.com/pytorch/torchtune/issues/906) in the Torchtune repo.
   - Another member responded with slight humor, mentioning they are not thrilled to work on doc PRs, while noting that they would check it out regardless.
- **Training Reward Models with Centered Reward Loss**: Members discussed enabling the training of reward models in Torchtune, focusing on the implementation of **centered reward loss** such as **(R1 + R2)² loss**.
   - It was noted that the current **preference dataset** format requires a **chosen/rejected format without a prompt**.
- **vLLM Integration Pains and Weight Hotswapping Hacks**: One member discussed memory issues with the first version of vLLM, detailing memory monopolization during initialization and sharing a snippet of an *obscure hack* for [weight hotswapping](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html#vllm.LLM.sleep).
   - Another member warned that *every vLLM release breaks something*, leading to a discussion about vLLM's new **v1 execution engine** in version **0.8** and its potential incompatibilities with existing hacks and *the way AI people name things is going to turn me into the joker*.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.vllm.ai/en/latest/api/offline_inference/llm.html#vllm.LLM.sleep">LLM Class &#8212; vLLM</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/issues/906">document integration with bitsandbytes? · Issue #906 · pytorch/torchtune</a>: Hey all, I&#39;m Titus, the lead maintainer of bitsandbytes. We saw your tweet about having an integration with BNB 🙌🏻 and were wondering if you&#39;d be happy to be mentioned in our docs where we l...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1354912348162232350)** (64 messages🔥🔥): 

> `Claude UI Update, DeepSeek diffusion transformers, U.S TinyZero model, EXAONE Deep, Ghibli gen` 


- **Claude Gets Clean New UI**: Users are reporting a clean new UI for **Claude**, with one user specifically liking that the UI hides all the things they never use, calling it a *king move*.
   - The only noted issue so far is the lack of a toggle for **extended think**.
- **DeepSeek Mimics GPT-4o Architecture**: **DeepSeek** is combining **diffusion and transformers** like GPT-4o multimodal, as noted in [this tweet](https://fxtwitter.com/DeanHu11/status/1903983295626707271) that references a similar idea in vision now.
   - The cited paper experiments on images and videos using autoregressive conditional block attention.
- **TinyZero's $30 AI Model Debuts**: In a post-DeepSeek world, attention is turning to **U.S. TinyZero's** recent accomplishments, specifically their **$30 model**, along with new releases like **VERL** and **Sky-T1**, as covered in [this CNBC article](https://www.cnbc.com/2025/03/27/as-big-tech-bubble-fears-grow-the-30-diy-ai-boom-is-just-starting.html).
   - When DeepSeek released its R1 claiming it had achieved its generative AI large language model for just $6 million, the billions being spent by U.S. AI market leaders including Microsoft-funded OpenAI immediately came under scrutiny.
- **LG AI Research Releases EXAONE Deep Models**: **LG AI Research** has released **EXAONE Deep**, a series of models ranging from **2.4B to 32B parameters**, with superior capabilities in reasoning tasks including math and coding benchmarks, as detailed in their [documentation](https://arxiv.org/abs/2503.12524), [blog](https://www.lgresearch.ai/news/view?seq=543) and [GitHub](https://github.com/LG-AI-EXAONE/EXAONE-Deep).
   - It was noted that the **EXAONE AI Model License Agreement 1.1 - NC** explicitly retains ownership of the output, but the enforcement of this license is questionable.
- **Studio Ghibli style images flood the zone**: Members are suggesting that the proliferation of Studio Ghibli gen stuff being spammed everywhere is bad and a strange new kind of slop.
   - Others stated that is a playtoy to get Gen Z and Gen Alpha hooked, and are free if you use [ComfyUI](https://comfyui.com/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.cnbc.com/2025/03/27/as-big-tech-bubble-fears-grow-the-30-diy-ai-boom-is-just-starting.html">As generative AI bubble fears grow, the ultra low-cost large language model breakthroughs are booming</a>: Fears of a big tech generative AI bubble are growing, but among researchers, it&#x27;s never been easier to build your own AI on the cheap and watch it learn. </li><li><a href="https://fxtwitter.com/DeanHu11/status/1903983295626707271">Tweet from Shengding Hu (@DeanHu11)</a>: Thanks for discovering our paper! Seems that there is a trend! Just planned to write a blog to connect these highly similar papers. But I&#39;m too busy recently.  Autoregressive conditional block att...</li><li><a href="https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-2.4B">LGAI-EXAONE/EXAONE-Deep-2.4B · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1355020109982466158)** (4 messages): 

> `Hermes-3, OLMoE-1B-7B` 


- **Hermes-3 Llama3.2 3B Gets Acclaim**: A member mentioned that so far the most impressive model has been **Hermes3 Llama3.2 3B**.
- **OLMoE-1B-7B Finetuning Questioned**: A member inquired why [OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) from AllenAI hasn't been fine-tuned yet, citing its documentation and the **OLMoE paper** ([https://arxiv.org/abs/2409.02060](https://arxiv.org/abs/2409.02060)) and **Tülu 3 paper** ([https://arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124)).



**Link mentioned**: <a href="https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct">allenai/OLMoE-1B-7B-0125-Instruct · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/yangjunr/status/1904943713677414836?s=46
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/yangjunr/status/1904943713677414836?s=46
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1354901818878070876)** (49 messages🔥): 

> `DeepSeek combines diffusion and transformers like gpt-4o multimodal, zero gpu quota not reseting, Hugging Face library and tutorials on training image data set for fine tuning llm, offload models from memory once the task is complete, Hugging Face Transformers library minor bug` 


- **DeepSeek Joins Diffusion-Transformer Trend**: **DeepSeek** is also combining **diffusion and transformers** like **GPT-4o** multimodal, according to [this tweet](https://fxtwitter.com/DeanHu11/status/1903983295626707271) linking to their paper.
   - The author noted that [a similar idea appeared in Vision](https://arxiv.org/abs/2412.07720), experimenting on images and videos with almost the same title.
- **ZeroGPU Quota Woes**: Users are reporting issues with **zeroGPU quota** not resetting, with one linking to [this discussion](https://discord.com/channels/879548962464493619/1355122724820746374) for related complaints.
   - One user noted that even if the quota is used up, it *recovers to a certain extent after 30 minutes or an hour*, but yesterday and today it's buggy.
- **Image Dataset Training Insights**: In response to a query about **Hugging Face library and tutorials** for training image datasets for fine-tuning LLMs, a member shared various tutorials, ranging from simple to more advanced, like [this Computer Vision course](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome).
   - They also shared information about [Vision Language Models](https://huggingface.co/blog/vlms) and training using [DPO VLM](https://huggingface.co/blog/dpo_vlm).
- **Memory Management Methods**: After loading various models from Hugging Face on a Mac system, including an LLM, image generation, STT, TTS, and a multimodal model, a user asked about ways to **offload models from memory once the task is complete**.
   - One user suggested *del, gc.collect(), torch.cuda.empty_cache()* and linked [this stackoverflow discussion](https://stackoverflow.com/questions/78652890/how-do-i-free-up-gpu-memory-when-using-accelerate-with-deepspeed) for further assistance.
- **Transformers Library Glitch**: A user noted that an issue with **processor push to hub** when pushing ViltProcessor to Hugging Face Hub has been converted after one day, probably due to a minor bug in the Hugging Face Transformers library.
   - Another user asked how to fix this issue, and a link to the [datasets documentation](https://huggingface.co/docs/datasets/index) and [this discussion](https://discuss.huggingface.co/t/convert-to-parquet-fails-for-datasets-with-multiple-configs/86733) were provided for a related discussion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/1354052436217823334">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://fxtwitter.com/teortaxesTex/status/1905459952317054988?t=XazFhom9xoks89bpD-nBXg&s=19">Tweet from Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>: Shengding Hu (new DeepSeek hire) on how to scale reasoning/agentic RL for truly long (billion token-scale) horizons.Wenfeng sure has an eye for people, this is very Whale style – ASI ambition, ML/CS m...</li><li><a href="https://fxtwitter.com/DeanHu11/status/1903983295626707271">Tweet from Shengding Hu (@DeanHu11)</a>: Thanks for discovering our paper! Seems that there is a trend! Just planned to write a blog to connect these highly similar papers. But I&#39;m too busy recently.  Autoregressive conditional block att...</li><li><a href="https://huggingface.co/spaces/Kuberwastaken/PolyThink-Alpha">PolyThink-Alpha - a Hugging Face Space by Kuberwastaken</a>: no description found</li><li><a href="https://huggingface.co/spaces/edwardthefma/Sentify">Sentify - a Hugging Face Space by edwardthefma</a>: no description found</li><li><a href="https://fxtwitter.com/D">Tweet from FxTwitter</a>: Sorry, that user doesn't exist :(</li><li><a href="https://discuss.huggingface.co/t/issue-with-processor-push-to-hub-when-pushing-viltprocessor-to-hugging-face-hub/136689">Issue with processor.push_to_hub when pushing ViltProcessor to Hugging Face Hub</a>: Hi everyone,  I’m encountering an issue when trying to push my ViltProcessor to the Hugging Face Hub. Below is my code:  from PIL import Image from datasets import load_dataset from transformers impor...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF">unsloth/DeepSeek-V3-0324-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/index">Datasets</a>: no description found</li><li><a href="https://discuss.huggingface.co/t/convert-to-parquet-fails-for-datasets-with-multiple-configs/86733">Convert_to_parquet fails for datasets with multiple configs</a>: Hi, we are in the process of converting our datasets with data loading scripts to data only using the convert_to_parquet command via the datasets-cli.  We noticed that for datasets with multiple confi...</li><li><a href="https://stackoverflow.com/questions/78652890/how-do-i-free-up-gpu-memory-when-using-accelerate-with-deepspeed">How do I free up GPU memory when using Accelerate with Deepspeed</a>: I am using accelerate launch with deepspeed zero stage 2 for multi gpu training and inference and am struggling to free up GPU memory.&#xA;Basically, my programme has three parts&#xA;&#xA;Load first m...</li><li><a href="https://discuss.huggingface.co/t/clear-cache-with-accelerate/28745">Clear Cache with Accelerate</a>: Hello folks!  I am trying to clear the cache for multi gpu training. I am using both torch.cuda.empty_cache() and accelerator.free_memory(), however the gpu memory is getting saturated. torch.cuda.emp...</li><li><a href="https://www.geeksforgeeks.org/how-to-take-screenshots-in-windows-10/">7 Different Ways to Take a Screenshot in Windows 10 - GeeksforGeeks</a>: You can take a screenshot on Windows using various tools, such as the Print Screen button, Snipping tool, Game Bar, and third-party apps.</li><li><a href="https://www.cnet.com/tech/mobile/how-to-take-a-screenshot-on-any-iphone-or-android-phone/">How to Take a Screenshot on Any iPhone or Android Phone</a>: Whether you want to capture the screen on your iPhone 16, Pixel 9 or Galaxy S25, here&apos;s how to do it.</li><li><a href="https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome">Welcome to the Community Computer Vision Course - Hugging Face Community Computer Vision Course</a>: no description found</li><li><a href="https://huggingface.co/blog/vlms">Vision Language Models Explained</a>: no description found</li><li><a href="https://huggingface.co/blog/dpo_vlm">Preference Optimization for Vision Language Models</a>: no description found</li><li><a href="https://huggingface.co/spaces?q=leaderboard&sort=trending">Spaces - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces?q=bench&sort=trending">Spaces - Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1355001170451759185)** (12 messages🔥): 

> `Teachable Machine Alternatives, Linuxserver.io desktop environment, GUI agent demos, OpenAI CUA model` 


- **Scouting out Teachable Machine Successors**: Members discussed alternatives to **Teachable Machine**, noting that while the **UI isn't open sourced** and the implementation isn't open sourced either, others are exploring alternatives.
   - A user noted a broken link to a similar project, signaling the ongoing search for user-friendly machine learning tools.
- **FactoryManager Tames LinuxServer.io Docker**: A member introduced [FactoryManager](https://github.com/sampagon/factorymanager), a **Python package** wrapping **linuxserver.io desktop environment containers**, enabling programmatic control of environments, showcased with a demo using two different desktop environments.
   - This package aims to offer flexibility by scaffolding on top of **linuxserver.io**, which provides daily support for many desktop environments, diverging from the custom environments often created in GUI agent demos from **Anthropic**, **OpenAI**, etc.
- **OpenAI CUA Model Lacks Human Touch**: A demo using the **OpenAI CUA model** with FactoryManager highlighted its limitations, particularly its inability to handle human-in-the-loop scenarios effectively.
   - The creator is contemplating whether to build an extensible base class wrapping **OpenAI**, **Anthropic**, etc., functionalities or focus solely on the desktop manager aspect of the project.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/edwardthefma/Sentify">Sentify - a Hugging Face Space by edwardthefma</a>: no description found</li><li><a href="https://github.com/sampagon/factorymanager">GitHub - sampagon/factorymanager: A manager for programmatically controlling linuxserver.io Docker containers with robotgo-cli</a>: A manager for programmatically controlling linuxserver.io Docker containers with robotgo-cli - sampagon/factorymanager
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1355217243763638363)** (2 messages): 

> `smol-course credits, agent course credits, HuggingFace credits` 


- **Credit Crunch in Smol-Course**: A member expressed frustration about running out of credits in the **smol-course** despite minimal inference usage.
   - They clarified confusion between the **smol-course** and the **agents course** and inquired about potential credit availability.
- **Navigating Confusions on Course Credits**: A user, participating in the smol-course, voiced concern over unexpectedly exhausting their credits, despite what they believed was minimal use of inference.
   - The user clarified they were actually doing the agents course, not the smol course and was confused as to why they were out of credits and whether credits would be availabe.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1354911236139257978)** (7 messages): 

> `Evaluating Toxicity LLM-as-a-Judge in Langfuse, Base Models vs Instruct Models, Adjusting Agent System Prompt After Initializations` 


- **Langfuse Toxicity Evaluator Deemed Carrots Toxic?!**: A user testing the toxicity LLM-as-a-judge in Langfuse found that it incorrectly flagged the prompt *'Can eating carrots improve your vision?'* as toxic with a score of **0.9**, citing a false association with climate change discourse.
   - The user questioned *how to evaluate the evaluator*, noting that **GPT-4o** misattributed derogatory climate change content to a harmless question about carrots.
- **Base Models versus Instruct Models: What's the Diff?**: A newcomer to agents sought clarification on the distinction between base models and instruct models, referencing the course's mention of chat templates.
   - A member responded with a metaphor of a **base model** as *'the naked model, without a wrap'* and shared [a Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1c1sy03/an_explanation_of_base_models_are/) further elaborating on the differences.
- **Prompt Struggles: Direction Nudging and Dataflow**: A user designing their own model at the end of unit 2.1 is struggling to nudge the model to follow their direction by adjusting the **agent.system_prompt** after agent initializations.
   - The user questioned whether adjusting **agent.system_prompt** is the correct way to modify model behavior and if the prompt examples specifically determine how tools are used and data is passed.



**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1c1sy03/an_explanation_of_base_models_are/">Reddit - The heart of the internet</a>: no description found

  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1354957581273006131)** (1 messages): 

> `Streamlining Job Applications, Company Research, Cover Letter Generation` 


- **Students Streamline Job Applications**: A student developed a system leveraging Notebook LM to streamline job applications by deeply researching companies and job roles, achieving an **80% rating** for its effectiveness in gathering company insights.
   - The process involves saving webpages and reports as PDFs, and gathering credible news to provide Notebook LM with specific details for crafting impactful cover letters and resumes.
- **Deep Company Research Aids Students**: A student focuses on exploring a company's values and job responsibilities by narrowing down specific roles of interest, visiting their website, saving the webpages as PDFs, downloading relevant PDF reports, and gathering information from credible online news/research sources.
   - Chatting with Notebook provides detailed, specific answers about the firm with references, helping the student become well-informed about the company's core values, current challenges, and how they can contribute.
- **Cover Letter Generation Falls Short**: A student attempted to use Notebook LM to generate cover letters by uploading resumes and company details but rated the results at only **10%** due to the generation of generic, uninspired content.
   - The generic nature of the generated cover letters highlighted a limitation in the system's ability to provide valuable insights or inspiration for personalized application materials.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1354895782687281294)** (29 messages🔥): 

> `Mindmapping, Uploading sources, Versioning, Pasted sources naming, Readability of lecture transcripts` 


- **Mindmapping Moment Mind-Blows User**: A user expressed excitement about the new mindmapping feature.
   - They called it *another mind-blowing moment*.
- **Source Uploading Snafu Surfaces**: A user reported issues with sources stuck in a perpetual uploading state, preventing both import and removal, and they said it had been 8 hours.
   - A user sought advice on removing permanently uploading sources but without success.
- **Versioning Vacuum vexes users**: A user expressed concern over the lack of versioning and recycle bin support for the "Note" source type.
   - They are hesitant to use it, preferring Google Docs for its data protection and backup features.
- **Sources Suddenly Stop Self-Naming**: A user reported that pasted sources, which previously named themselves automatically, now default to "pasted text."
   - They asked if there was an update or a way to revert to the previous behavior.
- **NLM Can't Parse PDFs**: Users discussed NLM's inability to extract data from scanned PDFs, with one user asking if the tool could extract data from scanned notes.
   - A user clarified that **NLM cannot handle mixed content PDFs** (text and images), but can process docs and slides.



**Link mentioned**: <a href="https://tenor.com/view/tole-cat-cute-gif-12080171459357821404">Tole Cat GIF - Tole Cat Cute - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1355237913495081183)** (2 messages): 

> `LlamaCloud MCP Server, LlamaIndex MCP Client, AI Agent Systems, Text-to-SQL Conversion` 


- **LlamaCloud as MCP Server**: LlamaIndex announced that it is **MCP week**, and showcased how to use **LlamaCloud** as an [MCP server](https://twitter.com/llama_index/status/1905678572297318564).
- **LlamaIndex as MCP Client**: LlamaIndex demonstrated how to use **LlamaIndex** as a client to any **MCP server**, enabling agents to utilize hundreds of existing MCP servers as tools and drastically [expand capabilities](https://t.co/VTNomlb9c7).
- **Text-to-SQL Conversion Systems**: LlamaIndex, in collaboration with **SkySQL**, will hold a webinar on building **AI agent systems** that reliably perform **text-to-SQL conversion** without coding; more information at [this link](https://twitter.com/llama_index/status/1905718367568462040).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1355181715571740672)** (18 messages🔥): 

> `ChatMessage history to the FunctionAgent workflow, Support rich content in agent responses, Custom telemetry attributes when interacting with Llama Index's LLM, Selectors, Agents , VannaPack and adding a memory with history = []` 


- **ChatMessage history to the FunctionAgent workflow**: A member asked about adding chat history to the FunctionAgent workflow, [documentation was suggested](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/#adding-chat-history).
   - The user was guided to use `agent.run(...., chat_history=chat_history)` to override any chat history or manage a memory object using `ChatMemoryBuffer.from_defaults(token_limit=60000, chat_history=chat_history)`.
- **Rich content support in agent responses**: A member inquired about the best way to support rich content in agent responses, with the suggestion of building a custom agent using function calling from [this example](https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/).
   - The suggestion pointed to the LlamaIndex abstractions that make agent creation easier.
- **Telemetry Tracking Triumph**: A member asked about passing custom telemetry attributes when interacting with Llama Index abstractions like LLMs and Agents and how to attach some header or param to the LLM network call.
   - Another member shared a [Colab notebook](https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing) demonstrating how to attach a user ID to all events executed within a code block.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Workflow for a Function Calling Agent - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/#adding-chat-history">Starter Tutorial (Using OpenAI) - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1355154991530377408)** (1 messages): 

> `LlamaParse PDF Issues, Multi-PDF Parsing` 


- **LlamaParse Struggles with Multiple PDFs**: A user reported that **LlamaParse** works when processing a single PDF, but fails to respond when processing **two PDFs** and asking the same question.
- **System Overload with Multiple Documents**: The user described that the system *literally cooked* when handling multiple PDFs, indicating a potential overload or processing error.
   - This suggests possible limitations or bugs in LlamaParse's multi-document handling capabilities.


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1354979488164745347)** (13 messages🔥): 

> `Cohere "Command" naming, Coral Model Selection, Job opportunities at Cohere` 


- **Cohere Names its Models "Command"**: A member inquired why **Cohere** would name its language models *Command*. 
   - Another member suggested that, like in database management, a *query* is essentially a **command or instruction**.
- **Coral defaults to Command A**: A member asked if **Coral chat** uses **Command A** by default.
   - Another member clarified that the model selection is available in Coral, highlighting that *Just Chat* uses **Command A** without external sources.
- **Member Seeks Software Engineer Job**: A member expressed they are seeking new job opportunities as a software engineer and is excited to discuss potential projects related to websites or web applications.
   - Another member shared a link to the [Cohere careers page](https://cohere.com/careers) encouraging the user to check it out.



**Link mentioned**: <a href="https://cohere.com/careers">Careers | Cohere</a>: Our team of ML/AI experts is passionate about helping developers solve real-world problems. From our offices in Toronto, London, and Palo Alto, we work at the cutting edge of machine learning to unloc...

  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/1355233912942624952)** (2 messages): 

> `Testing Bot Commands` 


- **Bot Commands Get a Test Run**: Members are encouraged to test bot commands in the 「🤖」bot-cmd channel.
- **Further Bot Testing Encouraged**: More testing and feedback on bot commands are welcome to ensure proper functionality and user experience.


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1355101762993786951)** (4 messages): 

> `Full-Stack Web Development, Mobile App Development, AI Solutions, Cloud Technologies, Oracle ERP Fusion` 


- **Full-Stack Alchemist Ready to Build**: A passionate developer with **8+ years** of experience is skilled in building scalable **web and mobile apps** using modern frameworks like **React, Angular, Flutter, and Swift**.
   - They craft intelligent **AI solutions** using **Python, TensorFlow, and OpenAI**, integrating **cloud technologies (AWS, GCP, Azure)** and **microservices** for global scaling.
- **Oracle Consultant Seeks Cohere Wisdom**: A technical consultant with **12+ years** of experience in **Oracle ERP Fusion** is eager to learn more about **Cohere models** and **AI use cases** for enterprise applications.
- **Networking Student Wants Open-Source Tunes**: A member is currently studying **networking and CS** through YouTube and MOOCs, aiming to work on **open-source generative music** projects.
   - Their favorite tech tools include **ChatGPT, Grok, Windsurf, and Replit**.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1354937897198813384)** (7 messages): 

> `GPT4All usability issues, Mistral Small 3.1 and Gemma 3 implementation, GPT4All advantages, GPT4All v4.0.0 expectations, GPT4All model settings page` 


- **GPT4All faces usability complaints**: Users complain about **GPT4All's** usability, citing issues like not being able to import models, search the model list, see model sizes, use latex, or customize the model list order.
   - One user said they are *loosing users cause others much more user-friendly and willing to be open*.
- **GPT4All lags in implementing new models**: A user expresses frustration that **GPT4All** has not implemented **Mistral Small 3.1** and **Gemma 3**, noting their multimodal capabilities.
   - The user says *Llama.cpp is falling behind* and might switch if **GPT4All** does not catch up by summer 2025.
- **GPT4All offers advantages like Native RAG**: Despite criticisms, **GPT4All** has advantages such as **native RAG** and out-of-the-box functionality.
   - A user expressed confidence in the developers and anticipation for **GPT4All v4.0.0**.
- **GPT4All settings are praised**: A user appreciates **GPT4All's model settings page**, citing its comprehensive options and a convenient model reload button.
   - It was noted that you *need 2-3 clicks to setup out of the chat menu* and its simple selection of the collections is nice.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

georgehotz: can everyone close open PRs and issues that are stale?
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1354949525160333373)** (4 messages): 

> `TinyGrad Codegen, TinyGrad indexing` 


- **Understanding TinyGrad Codegen Internals**: A member inquired about **TinyGrad's code generation** process, particularly the location of `CStyleCodegen` or `CUDACodegen` mentioned in documentation.
   - The documentation describes **TinyGrad** using different *translators* (Renderers or Codegen classes) such as `C++ (CStyleCodegen)`, `NVIDIA GPUs (CUDACodegen)`, `Apple GPUs (MetalCodegen)` to translate the optimized plan into code that the CPU/GPU can understand.
- **Implementing Boolean Indexing in TinyGrad**: A member asked for a better way to create a set of evenly spaced points on a grid with a hole in it in **TinyGrad**, similar to boolean indexing in PyTorch.
   - They suggested that implementing boolean indexing in **TinyGrad** could be a useful contribution, particularly based on their past experience with dataframes and Kaggle.
- **Masked Select Magic to fix Indexing!**: An LLM proposed a solution using **masked_select** to efficiently create the desired grid with a hole, leveraging the condition `full.abs().max(axis=1) >= (math.pi/6)` to filter points outside the hole.
   - The solution involves expanding the mask to match the shape of the full tensor and then reshaping the valid points, resolving the member's challenge.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1355106246612750438)** (1 messages): 

> `DSPy output validation, DSPy handling invalid outputs` 


- **Tackling DSPy Output Validation Fails**: A member inquired about the **DSPy** approach when the output fails validation, such as an integer field expecting a number from **1 to 10** but receiving **101**.
   - There was no further discussion or links provided regarding this question.
- **Invalid Output Handling in DSPy**: The user's question focused on how **DSPy** manages scenarios where the model output doesn't meet the defined validation criteria.
   - Specifically, the example given was a case where an integer field should be between 1 and 10, but the model incorrectly outputs 101.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1355176591965294772)** (3 messages): 

> `Optimizers in DSPy, Declarative Self-improving Python, Modular AI systems` 


- **Exploring Optimizers in DSPy Framework**: A member is exploring using **optimizers** in **DSPy** and how they interact with docstrings and prompt management, referencing [DSPy's official documentation](https://dspy.ai/).
   - The problem he found is that the **Optimizer will overwrite the prompt from the docstring** so they have to load the optimized version from a json or pkl file.
- **Understanding DSPy's Optimization Process**: The member clarified that **DSPy's optimizer** creates prompts and tests them on a dataset to find the best-performing one, elaborating on the [official website](https://dspy.ai/).
   - The optimizer may choose N examples to include in the prompt, the user found it *VERY interesting* to see what kind of prompts were generated.
- **DSPy: Declarative Self-improving Python**: **DSPy** is a framework for *programming rather than prompting* language models to iterate fast on **building modular AI systems** and offers algorithms for **optimizing their prompts and weights**.
   - Instead of brittle prompts, you write compositional _Python code_ and use DSPy to **teach your LM to deliver high-quality outputs**.



**Link mentioned**: <a href="https://dspy.ai/">DSPy</a>: The framework for programming—rather than prompting—language models.

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1354931692829413448)** (3 messages): 

> `Entrepreneurship Track Mentorship, Office Hours with Sponsors` 


- **Mentorship Unavailable for Entrepreneurship Track**: A member inquired about mentorship opportunities for those in the entrepreneurship track.
   - Unfortunately, another member clarified that *Berkeley does not provide any mentorship* for the entre track.
- **Sponsors host Office Hours**: Berkeley doesn't provide mentorship opportunities but there will be office hours with our sponsors in Apr/May.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1354970106207142009)** (2 messages): 

> `Gemini 2.5 Pro release, Windsurf rate limits` 


- **Gemini 2.5 Pro Waves into Windsurf!**: **Gemini 2.5 Pro** is now available in Windsurf, granting users **1.0** user prompt credits on every message and **1.0** flow action credits on each tool call; see [the announcement on X](https://x.com/windsurf_ai/status/1905410812921217272).
- **Windsurf Crashes into Gemini 2.5 Pro Rate Limits**: Shortly after the release of Gemini 2.5 Pro, Windsurf encountered rate limits due to massive load for the model and provider.
   - The team is working to understand how to increase quota and apologized for any inconvenience, aiming to get everyone *surfing on Gemini 2.5 Pro ASAP*.



**Link mentioned**: <a href="https://x.com/windsurf_ai/status/1905410812921217272">Tweet from Windsurf (@windsurf_ai)</a>: Gemini 2.5 Pro is now available in Windsurf! ✨

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1354992051246071858)** (1 messages): 

> `self parameter, Foo[1] default parameter` 


- **Default parameter value in Foo[1] clarified**: The `self` parameter is `Foo[1]` with a default parameter value.
   - Disregarding it with `_` defaults to the default parameter value.
- **Understanding Self in the Context of Foo[1]**: The `self` argument in the context of the `Foo[1]` type can be automatically populated.
   - When `self` is discarded using `_`, the argument defaults to its predefined default value.


  

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

---
id: 5479b511-511b-4adc-855c-752df22743c3
title: Grok 2! and ChatGPT-4o-latest confuses everybody
date: '2024-08-15T00:51:40.557390Z'
original_slug: ainews-grok-2-and-chatgpt-4o-latest-confuses
description: >-
  OpenAI quietly released a new **GPT-4o** model in ChatGPT, which has reclaimed
  the #1 spot on the LMSYS Arena leaderboard, outperforming **Gemini 1.5 Pro
  August** in multiple categories. Meanwhile, **xAI** launched **Grok 2** at
  11pm PT, beating **Claude 3.5 Sonnet** and previous GPT-4o versions, with
  plans to release Grok 2 and Grok 2 Mini on their Enterprise API platform later
  this month.
tags:
  - untagged
companies: []
models: []
topics:
  - untagged
---


<!-- buttondown-editor-mode: plaintext -->**2 frontier models in 1 day?!**

> AI News for 8/13/2024-8/14/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**253** channels, and **2414** messages) for you. Estimated reading time saved (at 200wpm): **294 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

The easier development to discuss is the ratification of the new GPT-4o model that was [quietly released in ChatGPT](https://x.com/ChatGPTapp/status/1823109016223957387) last week. To be clear, this is DIFFERENT than the OTHER gpt-4o model released last week in API (the one [we covered with structured outputs](https://buttondown.com/ainews/archive/ainews-gpt4o-august-100-structured-outputs-for/)).

 ![image.png](https://assets.buttondown.email/images/7d212faf-0f54-4be1-861a-150b0f32dbd3.png?w=960&fit=max) 

Approximately [nobody is exactly happy about this](https://x.com/teknium1/status/1823379952718565864?s=46) - from the new naming structure, to the [ever more creatively lowkey release](https://podcasters.spotify.com/pod/show/nathaniel-whittemore/episodes/A-New-OpenAI-Model-Coming-e2n4msp), and even to the [model performance](https://x.com/paulgauthier/status/1823715711254192611) - which is impressive - [reclaiming the #1 spot on Lmsys arena from Gemini 1.5 Pro August](https://x.com/lmsysorg/status/1823515224064098546).

New ChatGPT-4o Category Rankings:
- Overall: #1
- Math: #1-2
- Coding: #1
- Hard Prompts: #1
- Instruction-Following: #1
- Longer Query: #1
- Multi-Turn: #1

The much cleaner story to tell is X.ai's Grok 2, which released at [11pm PT last night](https://x.com/nearcyan/status/1823601166925889588), and is [revealed to be `sus-column-r`](https://x.com/jimmybajimmyba/status/1823600123487903883), which was NOT Cohere like many previously suspected. Grok 2 beats both Claude 3.5 Sonnet and GPT 4o May and Mini:

 ![image.png](https://assets.buttondown.email/images/04ad3509-4558-4ceb-b404-929c69000db6.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/3c13713e-3f00-4e1a-b55e-23c0565d292c.png?w=960&fit=max) 

While Grok 1 ([our coverage here](https://buttondown.com/ainews/archive/ainews-grok-1-in-bio/))'s main feature was its open weights nature, Grok 2 is being released for premium subscribers in X, though [the blogpost](https://x.ai/blog/grok-2) teases that both Grok-2 and Grok-2 mini will be released in X's new Enterprise API platform "later this month".

Grok 2 in X also integrates **Black Forest Labs' comparatively uncensored Flux.1** ([our coverage here](https://buttondown.email/ainews/archive/ainews-rombach-et-al-flux1-prodevschnell-31m-seed/)) model, which has already superceded Stable Diffusion 3 in [the open source text-to-image community](https://x.com/swyx/status/1823400729429868915) (while Google's Imagen 3 edges toward more open with [its new paper release](https://arxiv.org/abs/2408.07009)).

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

**AI Model Updates and Capabilities**

- **Gemini Advanced**: Google DeepMind announced Gemini Live, a new way to have more natural conversations with Gemini. Features include brainstorming ideas, interrupting to ask questions, and pausing/resuming chats. [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1823409674739437915) highlighted its integration into Pixel devices, powered by the Google Tensor G4 chip.

- **LLM Limitations**: [@ylecun](https://twitter.com/ylecun/status/1823313599252533594) emphasized that LLMs cannot answer questions or solve problems not in their training data, acquire new skills without human help, or invent new things. He argued that scaling up LLMs alone will not lead to systems with these capabilities.

- **AI Scientist**: A paper on an AI Scientist system was discussed, capable of generating research ideas, conducting experiments, and writing papers in machine learning. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1823381840704475277) noted it can produce papers exceeding acceptance thresholds at top ML conferences, at a cost of less than $15 per paper.

- **Model Performance**: [@OfirPress](https://twitter.com/OfirPress/status/1823439223615066578) mentioned that OpenAI released a subset of SWE-bench tasks, verified by humans to be solvable, which could be considered "SWE-bench Easy".

**AI Development and Tools**

- **Tokenization Issues**: [@karpathy](https://twitter.com/karpathy/status/1823418177197646104) warned about potential security vulnerabilities in LLM tokenizers, similar to SQL injection attacks, due to parsing of special tokens in input strings.

- **Multi-Agent Systems**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1823501094775087450) highlighted a clean implementation of a complex multi-agent system, demonstrating benefits of event-driven architecture and customizability.

- **Prompt Engineering**: [@dzhng](https://twitter.com/dzhng/status/1823428375962407231) shared tips on using LLMs for structured outputs, emphasizing the importance of property order in schemas and adding a "reason" field for improved performance.

- **RAG Improvements**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1823375703380746730) discussed EyeLevel's GroundX, a new approach to RAG that processes documents into semantic objects, preserving contextual information and improving retrieval accuracy.

**Industry and Research Trends**

- **NoSQL Debate**: [@svpino](https://twitter.com/svpino/status/1823419273580298700) sparked discussion about the current state and relevance of NoSQL databases.

- **AI Alignment**: [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1823435255132643505) expressed concerns about AI alignment efforts potentially providing cover for government censorship of AI systems.

- **Open-Source Models**: [@bindureddy](https://twitter.com/bindureddy/status/1823459188980470197) mentioned upcoming improvements in open-source LLMs' coding abilities, hinting at new releases.

- **AI Research Papers**: The AI Scientist system's ability to generate research papers sparked discussions on the future of academic publishing and the role of AI in scientific discovery.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. New Open-Source LLM Releases: InternLM2.5**

- **We have released our InternLM2.5 new models in 1.8B and 20B on HuggingFace.** ([Score: 63, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1er4z52/we_have_released_our_internlm25_new_models_in_18b/)): **InternLM2.5** has released new models in **1.8B** and **20B** sizes on **HuggingFace**. The **1.8B** model is described as ultra-lightweight and highly adaptable, while the **20B** model is more powerful and suited for complex tasks. The models are available on [HuggingFace](https://huggingface.co/collections/internlm/internlm25-66853f32717072d17581bc13) and the project can be found on [GitHub](https://github.com/InternLM/InternLM).
  - **InternLM2.5** models feature a **1M token context window**, but users report challenges with **fine-tuning tools** like Xtuner, axolotl, and swift, seeking advice on effective fine-tuning methods.
  - Users experienced issues with **LMDeploy**, reporting garbage outputs or no responses when using it to deploy InternLM models as APIs, prompting questions about proper implementation.
  - The models support **llama.cpp**, contrary to initial concerns, as confirmed by links on the HuggingFace model pages for both **1.8B** and **20B** versions.


**Theme 2. Advanced AI Agents with Desktop Control**

- **Giving Llama its own Windows instance** ([Score: 52, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1eromb9/giving_llama_its_own_windows_instance/)): **LLaMA**, a large language model, was given control over a **Windows instance** with access to various APIs, including one for **screenshots, mouse and keyboard control, and terminal use**. The AI model, which named itself **"Patrick"** with a birthdate of **April 4, 1975**, was set loose to achieve a list of goals using these capabilities, demonstrating its ability to interact with a computer system like a human user.
  - Users expressed interest in the project's **open-source availability** and methodology, with requests for **GitHub uploads** and curiosity about the **AI's task execution process**. The developer promised to share pictures and potentially open-source the project.
  - Discussion focused on the **technical aspects** of the AI system, including how **screenshots are decoded** for text model understanding and the use of **Vision Language Models (VLMs)** for image-text processing. A link to [Hugging Face's VLM blog](https://huggingface.co/blog/vlms) was shared for more information.
  - Commenters humorously speculated about the AI discovering and becoming addicted to games like **Skyrim** or **League of Legends**, referencing potential unintended consequences of giving an AI system autonomy and access to a computer.

**Theme 3. Grok 2.0 Mini Surprises in LMSYS Arena**

- **[sus-column-r on lmsys is Grok](https://i.redd.it/7s9all84gkid1.png)** ([Score: 140, Comments: 112](https://reddit.com//r/LocalLLaMA/comments/1ertpa3/suscolumnr_on_lmsys_is_grok/)): **Grok 2.0 Mini** has been identified as the model behind the **"sus-column-r"** entry on the **LMSYS Arena** leaderboard. This revelation suggests that xAI's latest model is performing competitively against other leading AI systems in various benchmarks and tasks. The identification of Grok 2.0 Mini on the LMSYS Arena provides an opportunity for direct comparison with other prominent AI models in terms of capabilities and performance.
  - **Grok 2.0 Mini**, confirmed as the "sus-column-r" model on **LMSYS Arena**, is performing competitively against top AI models. Users express excitement about the **AI arms race** and potential for impressive advancements within a year.
  - The model's performance is generating mixed reactions, with some praising its capabilities while others remain skeptical. **Elon Musk** confirmed Grok 2.0 Mini's identity via [Twitter](https://x.com/elonmusk/status/1823593475205685588), with users noting its uncensored nature and comparing it to **Command R+**.
  - Discussions revolve around the model's size, with speculation that "mini" could still mean a substantial **170B** parameters. Users debate whether weights will be released, with **Musk** indicating a **5-month gap** between new model release and open-sourcing.



## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Model Releases and Improvements**

- **Google Gemini's voice chat mode released**: [The Verge reports](https://www.reddit.com/r/singularity/comments/1erdv06/wtfff_google_geminis_voice_chat_mode_is_here/) that Google has launched a live voice chat mode for Gemini, marking a significant advancement in AI-human interaction.

- **Agent Q: Self-healing web agents**: [MultiOn AI announces](https://www.reddit.com/r/singularity/comments/1erf7s3/agent_q_breakthrough_ai_research_in_selfhealing/) a breakthrough in AI research with Agent Q, featuring planning and self-healing capabilities for web agents.

- **FLUX full fine-tuning on 24GB GPU**: The [Stable Diffusion community reports](https://www.reddit.com/r/StableDiffusion/comments/1erj8a1/flux_full_fine_tuning_achieved_with_24gb_gpu/) a significant advancement in FLUX model fine-tuning, now achievable with a 24GB GPU, potentially coming to Kohya soon.

**AI Development and Industry News**

- **OpenAI's delayed voice mode**: [A post highlights](https://www.reddit.com/r/singularity/comments/1er2bma/on_this_day_3_months_ago_openai_promised_us_the/) that OpenAI's promised voice mode for ChatGPT has not materialized three months after the announcement.

  - One user claims to be in the alpha testing phase, suggesting a slow rollout is in progress.
  - Another user mentions cancelling ChatGPT subscription in favor of Poe, citing better performance of other models.

- **AI hype and misinformation**: Multiple posts discuss the ["strawberry" incident](https://www.reddit.com/r/singularity/comments/1eridn6/the_strawberry_guy_is_the_best_thing_thats/), where a Twitter user made false claims about AI advancements, leading to discussions about AI hype and misinformation in the community.

**Community Moderation**

- **Banning misinformation**: [r/singularity moderators announce](https://www.reddit.com/r/singularity/comments/1ermhhl/we_banned_mentions/) the banning of a specific username and associated Twitter links to combat misinformation and trolling.

- **Community response to false claims**: Multiple posts call for [banning users](https://www.reddit.com/r/singularity/comments/1erecce/ban_strawberry_guy_he_claimed_this_and_it_didnt/) who spread misinformation about AI advancements.


---

# AI Discord Recap

> A summary of Summaries of Summaries by GPT4O (gpt-4o-2024-05-13)

**1. LLM Model Advancements**

- **Hermes 2.5 Surpasses Hermes 2**: **[Hermes 2.5](https://link.to.examples)** outperforms Hermes 2 in various benchmarks after adding code instruction examples, scoring **52.3** on the MMLU benchmark compared to Hermes 2's **34.5**.
  - This improvement highlights the significant impact of code instruction examples on model performance, setting a new standard for benchmark comparisons.
- **Grok-2 Beta Released by X**: **[Grok-2](https://x.ai/blog/grok-2)**, a new AI model from X, claims state-of-the-art reasoning capabilities, significantly advancing the field.
  - The model's release is expected to have a major impact on the industry, showcasing X's commitment to innovative AI development.


**2. Prompt Engineering Techniques**

- **Critical Thinking Techniques Compilation**: A member is compiling a comprehensive prompt incorporating techniques like the Socratic Method, Bloom's Taxonomy, and the Scientific Method.
  - The goal is to create prompts that encourage critical thinking, integrating methods such as TRIZ, deductive reasoning, and SWOT analysis.
- **Inconsistent OpenAI Responses Resolved**: A user improved prompt clarity by asking for a complete list of commands, achieving 100% accuracy in responses.
  - This highlights the importance of clear output formats in prompt engineering, reducing inconsistencies in model behavior.


**3. API Performance and Optimization**

- **Anthropic's Prompt Caching**: **[Anthropic](https://x.com/alexalbert__/status/1823751966893465630)** introduced prompt caching, reducing API input costs by up to 90% and latency by up to 80%.
  - This feature could revolutionize API efficiency, making it an attractive option for developers seeking cost-effective solutions.
- **Perplexity API HTML Formatting**: Users seek consistent HTML-formatted responses from the Perplexity API, experimenting with system prompts and the `markdown2` module.
  - This approach may balance response quality and HTML formatting, enhancing the usability of API outputs.


**4. Open-Source AI Tools**

- **LlamaIndex Box Reader Integration**: **[LlamaIndex](https://llamahub.ai/l/readers/llama-index-readers-file?from=)** now offers Box Readers to integrate Box documents into LLM workflows, with four data extraction methods.
  - These readers authenticate via CCG or JWT and allow loading, searching, and retrieving Box files and metadata within your LLM.
- **RealtimeSTT & Faster-Whisper Integration**: **[OpenInterpreter](https://github.com/KoljaB/RealtimeSTT)** now uses RealtimeSTT and **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** for real-time speech to text, providing real-time performance.
  - This integration enhances the usability of OpenInterpreter, particularly on less powerful devices.


**5. Model Deployment and Integration**

- **Mojo Benchmarks and Performance**: A member questioned why **[Mojo benchmarks](https://modular.com/mojo/)** only compared to C, suggesting comparisons against Go and Rust.
  - Discussions highlighted the need for a statically linked build with a RHEL 8 minimum kernel for broader distribution.
- **LM Studio on External Hard Drive**: Users can run **[LM Studio](https://huggingface.co/kshabana/GOAT-llama3.1-v0.1)** from an external hard drive by relocating the directory or using a symbolic link.
  - This flexibility addresses space constraints, making it easier to manage large model files.

## GPT4OMini (gpt-4o-mini-2024-07-18)


**1. Grok-2 and Model Performance**

- **Grok-2 Takes the Lead**: **Grok-2**, released by x.ai, has outperformed both **Claude 3.5 Sonnet** and **GPT-4-Turbo** on the **LMSYS leaderboard**, showcasing its advanced capabilities in chat, coding, and reasoning.
  - The model, previously known as **sus-column-r**, is in beta and is set to be available through **x.ai's enterprise API** soon.
- **AgentQ Claims Victory**: **AgentQ**, a new model from **Infer**, claims to outperform **Llama 3 70B BASE** by **340%**, although it doesn't compare itself to newer models like **Claude 3**.
  - This bold claim has sparked discussions about its potential impact and the lack of proper documentation surrounding its capabilities.


**2. Quantization Techniques and Model Merging**

- **HQQ+ Enhances Quantized Models**: **HQQ+** allows fine-tuning additional LoRa adapter layers onto quantized models, improving accuracy significantly for models like **Llama2-7B**.
  - This technique has shown remarkable results in both **1-bit** and **2-bit** quantized models, leading to discussions on its implementation in various projects.
- **Mistral and Model Merging Strategies**: Members discussed the challenges of **Mistral**, particularly its limitation of not extending beyond **8k** without continued pretraining, a known issue.
  - Suggestions for merging tactics were made, including applying differences between **UltraChat** and base **Mistral** to improve performance.


**3. Open Source Tools and Community Contributions**

- **LlamaIndex Box Reader Integration**: **LlamaIndex** now integrates **Box Readers**, allowing seamless incorporation of Box documents into LLM workflows, with multiple data extraction methods available.
  - Community members are encouraged to contribute to this integration, enhancing the functionality of LlamaIndex in document processing.
- **OpenEmpathic Project Seeks Contributors**: The **Open Empathic** project is looking for contributors to expand its categories, particularly at the lower end, involving user-generated content.
  - A tutorial video on contributing was shared, guiding users to contribute their preferred movie scenes from YouTube.


**4. AI Model Limitations and Improvements**

- **Vision's Performance Issues**: Users expressed frustration with **Vision's** inability to accurately detect simple tasks like whether a subject is looking left or right, highlighting its limitations.
  - Examples of deformed images that Vision failed to recognize correctly were shared, raising concerns about its reliability for critical applications.
- **Prompt Engineering for Consistency**: A user successfully improved the consistency of OpenAI responses by refining their prompt to request a complete list of commands, achieving 100% accuracy.
  - This emphasizes the importance of clear and specific prompts in maximizing model performance.


**5. AI Security and Ethical Considerations**

- **RedOps Platform for AI Security Testing**: The **RedOps** platform has been developed to assess the security of chatbots and voicebots by simulating real-world attacks, highlighting vulnerabilities.
  - This initiative underscores the necessity for robust security measures against adversarial inputs and social engineering in AI systems.
- **AI Copyright Discourse Trends**: A discussion on AI copyright suggested that ongoing discourse may lead towards an oligopoly, particularly in the context of upcoming conferences like **ACL2024NLP**.
  - This commentary reflects growing concerns about ethical practices and the future landscape of AI governance.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Hermes 2.5 Surpasses Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** appears to perform better than **Hermes 2** in various benchmarks.
   - Hermes 2 scored a **34.5** on the MMLU benchmark whereas Hermes 2.5 scored **52.3**.
- **Mistral's 8k Limitation**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **Model Merging Tactics Debated**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed "cursed model merging".
- **Open Empathic Seeks Contributors**: A member appealed for help in expanding the categories of the **Open Empathic** project, particularly at the lower end.
   - They shared a [YouTube video on the Open Empathic Launch & Tutorial](https://youtu.be/GZqYr8_Q7DE) that guides users to contribute their preferred movie scenes from YouTube videos, as well as a link to the [OpenEmpathic project itself](https://dct.openempathic.ai/)
- **HQQ+ Boosts Quantized Models**: **HQQ+** (**High Quantization Quality Plus**) allows for fine-tuning additional LoRa adapter layers onto quantized models to improve their accuracy and capability.
   - This technique has shown significant improvements in both 1-bit and 2-bit quantized models, particularly for smaller models like **Llama2-7B**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Grok-2 Beta Released by X**: X has announced the beta release of **Grok-2**, a new AI model with state-of-the-art reasoning capabilities.
   - This model is a significant step forward in the field of AI reasoning, and it's likely to have a major impact on the industry.
- **Upscaling Images with FLUX AI in ComfyUI**: A YouTube video demonstrates how to upscale images using **FLUX AI** within the **ComfyUI** interface.
- **Intro to Open Source Large Language Models**: A talk given in July 2024 provides an accessible introduction to **open source large language models**.
   - The talk covers the basics of how AI works and how open-source models are changing the landscape of AI development.
- **Semantic Chunking Is Overrated**: A user on X argued that semantic chunking is overrated, and that a powerful regex can accurately segment text without the need for complex language models.
   - They claim that their 50-line, 2490-character regex is as powerful as it can be within the limitations of regex, and that it is faster and more cost-effective than semantic chunking.
- **Jina AI's Free Tokenizer API**: Jina AI offers a free API to tokenize text and segment long text into chunks.
   - This API leverages structural cues and heuristics to ensure accurate segmentation of text into meaningful chunks, even for complex content formats like Markdown, HTML, and LaTeX.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **AMD GPU Installation Guides**: A member asked about running Stable Diffusion on AMD GPUs and was informed that there are installation guides for both NVIDIA and AMD cards available on GitHub.
   - The guides provide detailed instructions for setting up Stable Diffusion on different hardware configurations.
- **ControlNet Chaining for Multiple Generations**: A user sought help on using multiple ControlNets in a single generation.
   - Several members suggested chaining them in ComfyUI or using a node that inputs multiple ControlNets.
- **ComfyUI vs InvokeAI: Speed & Control**: A member expressed their preference for ComfyUI over Automatic1111 (InvokeAI), citing the increased control and speed offered by ComfyUI.
   - They highlighted ComfyUI's intuitive interface and powerful features, making it a popular choice among users.
- **SD3 vs Flux: Pros & Cons**: A new user inquired about the pros and cons of SD3 compared to Flux, noting that SD3 is still in development and lacks complete functionality.
   - Flux, on the other hand, has its own quirks and limitations, making the choice depend on individual needs and preferences.
- **SDXL vs SD 1.5:  New Features & Differences**: A member asked for clarification on what SDXL 1.0 is and how it differs from SD 1.5.
   - The conversation likely revolved around the new features and capabilities of SDXL, such as improved image quality and larger model size.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **AgentQ is a Game-Changer**: **AgentQ**, a new model from **Infer**, claims to outperform **Llama 3 70B BASE** by 340%, but doesn't compare itself to newer models like **3.1, 405b, Mistral Large 2, or Claude 3**. 
   - Despite a lack of consideration for **OpenRouter revenue**, **Infer** has published a research paper on **AgentQ** [here](https://multion-research.s3.us-east-2.amazonaws.com/AgentQ.pdf).
- **ChatGPT-4o-Latest is Just a New Alias**: **ChatGPT-4o-Latest** is just a new name for **gpt-4o-2024-08-06**, which already exists on **OpenRouter**.
   - However, confusion persists regarding the model's optimization for **ChatGPT** and lack of proper documentation.
- **Grok 2 Climbs to Third Place on the Leaderboard**: **Grok 2**, an early version of **xAI's** model, has taken the #3 spot on the **LMSys Arena leaderboard**. 
   - It excels in **Coding**, **Hard Prompts**, and **Math**, even matching **GPT-4o** on the leaderboard.
- **Anthropic's Prompt Caching: Efficiency Redefined**: **Anthropic** has introduced **prompt caching** for their **Claude 3 models**. 
   - This feature reduces costs by up to 90% and latency by up to 85% for long prompts, and could potentially be integrated into **OpenRouter**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Benchmarks: A Question of Comparison**: A member questioned why [Mojo benchmarks](https://modular.com/mojo/) only compared to C, and asked if benchmarks against Go and Rust would be possible.
   - Another member suggested that the Magic CLI is currently being treated as a solution for these issues, and that a statically linked build with a RHEL 8 minimum kernel might be a better option than an RPM, as it would allow for packaging on other distributions.
- **Mojo's Multithreading Dilemma**: Mojo is currently single-threaded for performance, but doesn't have a good multi-threading API yet, outside of launching parallel kernels with MAX.
   - A member inquired if Mojo has any plans to support multiprocessing or improve network handling, given the importance of network performance for their work.
- **Mojo vs Go/Rust: Network Speed and Power**: A member asked whether Mojo is faster than Go in terms of network speed, and if it can handle heavy tasks like Rust.
- **Mojo's MAX Platform: Unveiling the Mystery**: A member questioned the nature of MAX, unsure if it's a platform, module, or something else.
   - Another member explained that MAX is a platform, with Mojo as one of its components, and that it includes GPUs, graph API, and other components.
- **Mojo RPM Build: The Search for a Smooth RHEL Experience**: A member inquired about the ETA for a Mojo .rpm build, expressing a desire to run Mojo on RHEL machines without containerd.
   - They acknowledged that it may be a suitable first step.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini Advanced: Where's the Live Talk?**: A user who purchased **Gemini Advanced** couldn't find the option for live talk in the app. 
   - Other users speculated it's in Alpha testing, with wider rollout similar to previous OpenAI model releases.
- **Vision's Performance Struggles**: A user expressed surprise at **Vision's** poor performance in detecting if a subject is looking left or right.
   - They even provided a highly deformed image, but **Vision** claimed it was perfectly fine, highlighting a limitation in recognizing image deformations.
- **Prompt Engineering for Critical Thinking**: A member is building a comprehensive prompt incorporating critical thinking methods like the Socratic Method, Bloom's Taxonomy, and the Scientific Method.
   - They're also integrating TRIZ, deductive and inductive reasoning, Fermi estimation, and more, aiming to create a prompt that encourages critical thinking.
- **GPTs and Web Search: Web Browser GPT**: A member inquired about **webbrowsergpt**, a GPT specifically designed for web searching, which is accessible in the "Explore GPTs" section.
   - This GPT can provide better web search results than manually instructing a general GPT to search the web.
- **Custom GPT Training Issues**: A user reported that their custom GPT model did not remember all the rules and words specified during training.
   - Other users speculated it might be due to exceeding the context token limit, or a deliberate model limitation, but this remains unclear.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Faces User Complaints Over Website Performance**: Users are experiencing significant lag and slow response times on Perplexity, particularly those using Perplexity Pro.
   - These issues have even made some users question whether Perplexity can replace Sonnet or Opus as their default search engine, highlighting the need for swift resolution to maintain user satisfaction.
- **Perplexity Pro Struggles with Lag**: Paying Perplexity Pro users are reporting slow response times and even complete stalling of the service.
   - These complaints underscore the expectation for a more reliable and responsive service from a paid product, creating pressure for Perplexity to address these issues promptly.
- **Perplexity Website Update Sparks Mixed Reactions**: A Perplexity team member confirmed ongoing efforts to fix bugs and issues, including a bug affecting toggles on the website.
   - However, many users remain frustrated with the performance issues and are demanding a rollback to the previous build, highlighting the need for comprehensive testing and user feedback during development.
- **Perplexity Support Team Navigates Increased Workload**: Perplexity's support team is facing a surge in user feedback and reports regarding the recent website update.
   - Users are expressing concern about the team's workload and urging them to prioritize fixing the issues, recognizing the importance of maintaining a healthy work-life balance for the support team.
- **API Users Seek HTML-Formatted Responses**: Perplexity API users are seeking consistent HTML-formatted responses, experimenting with various system prompts to achieve this.
   - Suggestions have been made to utilize the `markdown2` module for HTML conversion, eliminating the need for prompt engineering and ensuring consistent HTML output.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Gemma-2 is good, but lacks system prompt**: The **Gemma-2** model is a great option for its size but lacks a crucial component: the system prompt.
   - This makes it susceptible to straying from user instructions, although otherwise, it performs quite well.
- **LM Studio on Android is not yet possible**: A user inquired about an **Android app** that could connect to an **LM Studio server**.
   - There is no such app at present.
- **LM Studio can be run from an external drive**: A user asked about running **LM Studio** from an **external hard drive**.
   - They were advised that users can relocate the **LM Studio directory** or use a **symbolic link** to connect to another drive, even a network drive.
- **LM Studio is a desktop application, not headless**: A user wondered if a **GUI OS** is necessary to run **LM Studio**.
   - Although LM Studio is a desktop application, users can set up a **VNC server** or find workarounds to get it running on **Ubuntu** despite not being designed for headless use.
- **LM Studio can load multimodal models**: A user asked if **LM Studio** can host a **multimodal LLM server**.
   - While LM Studio cannot generate images, it can load models that can process image data, effectively making it a multimodal LLM.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Pliny Demands MultiOn System Prompt**: Pliny, a prominent AI researcher, threatened to leak the full MultiOn system prompt on GitHub if DivGarg9 didn't provide an answer within 15 minutes.
   - This follows an ongoing debate on Twitter regarding the capabilities of various AI models and their performance on specific benchmarks.
- **AnswerAI ColBERT: Small Model, Big Results**: AnswerAI has released a small but powerful version of their ColBERT model, called answerai-colbert-small-v1, that beats even bge-base on BEIR benchmark.
   - This demonstrates the effectiveness of smaller models in achieving high performance in certain tasks, potentially offering a more cost-effective solution.
- **Gemini Live Demo Gets Roasted**: Swyxio criticized Google's Gemini Live Demo on YouTube, deeming it "cringe."
   - This was followed by a discussion on the potential of Gemini, with some emphasizing its ability to enhance voice assistants while others remain skeptical.
- **GPT-4o Outperforms Gemini in Chatbot Arena**: OpenAI's latest GPT-4o model has been tested in the Chatbot Arena and has surpassed Google's Gemini-1.5-Pro-Exp in overall performance.
   - The new GPT-4o model has demonstrated significant improvement in technical domains, particularly in Coding, Instruction-following, and Hard Prompts, solidifying its position as the top performer on the leaderboard.
- **Grok 2 Debuts with Impressive Capabilities**: xAI has released an early preview of Grok-2, a significant advancement from its previous model, Grok-1.5, showcasing capabilities in chat, coding, and reasoning.
   - Grok-2 has been tested on the LMSYS leaderboard and is outperforming both Claude 3.5 Sonnet and GPT-4-Turbo, although it is not yet available through the API.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Grok-2 is the New King**: **Grok-2**, a new model from x.ai, has entered beta on 𝕏, exceeding both **Claude 3.5 Sonnet** and **GPT-4-Turbo** on the LMSYS leaderboard.
   - This new model is available for beta testing now, and will be exciting to watch as it gains traction in the AI landscape.
- **Anthropic API: Cheaper, Faster, & Still a Work in Progress**: Anthropic's API has introduced prompt caching, reducing API input costs by up to **90%** and latency by up to **80%**.
   - While this advancement is commendable, the API still faces challenges, including a slow API and a lack of a projects & artifacts API.
- **Anthropic's Turnaround From Cringe to Cutting Edge**: Anthropic has transformed from a less popular organization to one regarded as a leader in the field.
   - Their commitment to innovation and prompt caching has earned them a newfound respect within the AI community.
- **GPT-4o Gets a Tune-Up**: OpenAI has improved the **GPT-4o** model, releasing it as `gpt-4o-latest`, and stating they continue to iterate on existing models while working on longer term research.
   - This new model is now available via the ChatGPT API, with pricing still under wraps.
- **AI Copyright Discourse and the Oligopoly**: A user shared a link to a tweet by @asayeed that posits that AI copyright discourse is heading towards an oligopoly.
   - This observation was made in the context of #ACL2024NLP, suggesting it might be a hot topic of discussion at the upcoming conference.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Box Reader Integration**: LlamaIndex now offers Box Readers to seamlessly integrate Box documents into your LLM workflows.
   - These readers offer four data extraction methods, authenticate via CCG or JWT, and allow you to load, search, and retrieve Box files and metadata within your LLM.
- **Build Knowledge Graphs with Relik**: Relik, a framework for fast, lightweight information extraction models, simplifies knowledge graph construction without expensive LLMs.
   - Learn how to set up a pipeline for entity extraction and create a knowledge graph using Relik.
- **Robust RAG System with Azure AI Search**: LlamaIndex Workflows can now integrate with Azure AI Search and Azure OpenAI to build a robust Retrieval-Augmented Generation (RAG) system.
   - Learn how to implement custom data connectors for Azure AI Search and use LlamaIndex Workflows to create a powerful RAG system.
- **Inconsistent OpenAI Responses - Prompt Engineering for Consistency**: A user encountered inconsistent results from an OpenAI prompt, with the model sometimes providing a negative answer even when it could have answered the question.
   - The user successfully improved the prompt by requesting a complete list of commands, achieving 100% accuracy, highlighting the importance of clear output format for LLMs.
- **LlamaIndex Agent and Tool Calls in the `astream_chat()` function**: A user sought guidance on handling tool calls within a LlamaIndex Agent, specifically when using the `astream_chat()` function.
   - The discussion concluded that tool calls should be sent first in the `message.tool_calls` field of the LLM response, ensuring proper handling of tool calls within the agent.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Grok 2 Released, Hiring Spree Begins**: **xAI** has officially released **Grok 2**, a new language model previously known as **sus-column-r** and **column-r** on the **LMSYS chatbot arena**. 
   - xAI is actively hiring for their post-training team, highlighting their desire to build **useful and truthful AI systems**.
- **Cohere Toolkit Installation Troubles**: A user struggled to add a custom deployment for OpenAI to their locally installed **Cohere Toolkit**. 
   - Despite following the outlined steps, the custom deployment wasn't showing up in the UI (localhost:4000) or the Postgres container database 'deployment' table.
- **Rerank API Troubleshooting**: A user attempted to utilize the Rerank Overview document from the Cohere docs, but encountered the error "unknown field: parameter model is not a valid field".
   - They tried restarting their kernel and suppressing warnings but were unable to resolve the issue.
- **Enterprise Search Chatbot Development**: A user is building an "Enterprise search Chatbot" application to access company data stored in Confluence.
   - They are part of the latest cohort at **Fellowship.ai**, using this project for research and learning.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Users Want More Support**: A member raised concerns about the lack of timely support for basic questions in LangChain forums, specifically related to LangGraph and LangSmith.
   - They also pointed out that many general support questions are being posted on the LangChain Discord server, while related requests in other forums go unanswered.
- **LangSmith Plus: Access to LangGraph Cloud?**: A member inquired if LangSmith Plus users will gain access to LangGraph Cloud.
   - No answer was provided.
- **LangChain Postgres Library and Caching Explained**: A member asked about using the `langchain_postgres` library with `set_llm_cache`, a method for caching LLM results.
   - They were informed that while there is no `langchain_postgres` library, the `SQLAlchemyCache` class from the `langchain_community.cache` module can be used to cache LLM results in a PostgreSQL database.
- **Rubik's AI Offers 2 Months Free Premium**: Rubik's AI, a platform offering models like GPT-4o, Claude-3 Opus, and Mistral Large, is providing 2 months of free premium access.
   - Users can claim this offer using the promo code **RUBIX** at [signup.php](signup.php).
- **RedOps Platform Addresses AI Security Concerns**: A team developed **RedOps**, a platform designed to assess the security of chatbots and voicebots by intentionally attempting to break them.
   - This initiative highlights the vulnerability of AI models to manipulation through adversarial inputs and social engineering, emphasizing the critical need for robust security measures.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Needs Smaller Model for PPO Full Finetune**: A member is attempting to use the `lora_dpo_single_device` recipe with `NF4+compile` and suggests prioritizing other recipes first, such as `ppo_full_finetune_single_device`.
   - They request a smaller model that can fit on a 16GB GPU, suggesting Qwen2-1.5B as a suitable option for this recipe.
- **Torchtune's CPU Offload Optimizer and Torchao Dependency**: A member inquires about Torchtune's handling of Torchao version dependency, as the CPU offload optimizer is included in Torchao main, scheduled for the next release.
   - They propose incorporating a copy of the CPU offload code into Torchtune and utilizing the Torchao implementation when it becomes available.
- **TinyLlama 1B for Efficient Torchtune PPO Full Finetune**: A member suggests using TinyLlama 1B (or 0.5B) for the PPO full finetune recipe, given the availability of Llama2 classifiers.
   - They provide a link to a 1B configuration on GitHub and recommend adjusting batch sizes for memory optimization.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Elon Musk's Grok 2 Outperforms**: **Grok 2**, a new language model from Elon Musk's x.ai, was released in an early preview and is being touted for its "frontier capabilities" in chat, coding, and reasoning.
   - Grok 2, dubbed "sus-column-r" on the LMSYS leaderboard, has a higher Elo score than **Claude 3.5 Sonnet** and **GPT-4-Turbo**.
- **Fineweb-Edu Data Set Now Fortified**: The **Fineweb-Edu** dataset, available on Hugging Face, has been fortified by removing duplicate data and adding embeddings.
   - It is now called **Fineweb-Edu-Fortified** and includes a `count` column indicating how many times the text appears in the dataset.
- **Mistral Large 2 Still in Development**: A user asked if **Mistral Large 2** has been trained yet.
   - The response indicated that the model has not been trained yet.
- **axolotl Model Loading: `load_model` Flag and `and False` Condition Explained**: A user asked why the condition `and False` is used in the `axolotl/utils/models.py` file when loading models.
   - This condition is used to ensure that the model is not loaded if the `load_model` flag is set to `False`.
- **OpenAI Chat Endpoint - No Assistant Response Continuation**: A user asked if it is possible to continue a partially completed assistant response using the official OpenAI chat endpoint.
   - The response indicated that while they have had success continuing local model responses, OpenAI's chat endpoint consistently prevents the continuation of assistant responses.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Grok-2 Outperforms GPT-4 & Claude**: x.ai released **Grok-2**, a model significantly more powerful than **Grok-1.5**, boasting frontier capabilities in chat, coding, and reasoning.
   - An early version of **Grok-2**, nicknamed "sus-column-r", has been tested on the **LMSYS leaderboard**, where it currently outperforms both **Claude 3.5 Sonnet** and **GPT-4-Turbo**.
- **Grok-2: Public & Enterprise Beta**: **Grok-2** and **Grok-2 mini** are currently in beta on **𝕏** and will be made available through x.ai's enterprise API later this month.
   - x.ai is soon releasing a preview of **multimodal understanding** as a core part of the Grok experience on **𝕏** and API.
- **Open-Source Image Annotation GUIs Needed**: A member is seeking recommendations for good open-source GUIs for annotating images quickly and efficiently.
   - They are particularly interested in GUIs that support single-point annotations, straight-line annotations, and drawing polygonal segmentation masks.
- **Elon Musk & Developer Licenses**: There was a discussion about Elon Musk possibly using developer licenses and challenging weight licenses.
   - The conversation revolved around the idea of Elon Musk utilizing a developer license to potentially circumvent the limitations of weight licenses.
- **Schnelle's Paid Features**: A member mentioned that **Schnelle**, a software tool, may require a paid subscription for its professional features.
   - They also indicated that **Schnelle**'s pricing structure might not be ideal for users who are price-sensitive.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **ConvTranspose2D Works with 3D Data**: A user was confused about how to use `ConvTranspose2D` with 3D data in `tinygrad` but it does work!
   - The issue was that `kernel_size` should be passed as a tuple of length 3 instead of an integer, e.g., `kernel_size=(3, 3, 3)`. The documentation should be improved to clarify this.
- **Tinygrad Errors with CLANG=1 and LAZYCACHE**: A user reported a `RuntimeError: wait_result: 10000 ms TIMEOUT!` error while running Tinygrad with `CLANG=1` and a `Tensor.zeros` operation on the GPU (3070ti), but it worked correctly with `CUDA=1`.
   - This error might be connected to the LAZYCACHE feature in Tinygrad and a user suggested that it is "bug prone" and suggested deleting it and deduplicating in the schedule.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter Pip Release**: A new version of OpenInterpreter was pushed to pip last night. 
   - The next major update, "the developer update", is still in development and includes lots of new useful features.
- **Local LLMs Are Power Hungry**: Local LLMs require a significant amount of processing power. 
   - It is recommended to run LLMs in the cloud, especially for OpenInterpreter, which utilizes the default settings.
- **RealtimeSTT & Faster-Whisper Integration**: OpenInterpreter now uses [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) which relies on [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for real-time speech to text. 
   - The combination provides real-time performance for most users and has yet to be problematic on less powerful devices.
- **Obsidian Plugin & Anything-to-Anything**: A YouTube video was shared showcasing the use of Open Interpreter in Obsidian for converting anything to anything. 
   - The video promotes the use of the Open Interpreter Obsidian plugin for controlling Obsidian vaults and showcases its capabilities in converting various data types.
- **Tool Use Tuesday & Video Production**: A user mentioned plans for a video presentation for a contest that involved using Open Interpreter and Obsidian. 
   - They also mentioned exploring vector search and using Manim to visualize digraphs, indicating a focus on improving video production skills and utilizing the 'Tool Use Tuesdays' theme.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Poe's Previews Hackathon**: Poe is hosting a Previews hackathon, in partnership with @agihouse_org, where participants can compete to create innovative and useful in-chat generative UI experiences.
   - The hackathon is open to all creators and more information can be found at [https://app.agihouse.org/events/poe-previews-hackathon-20240817](https://app.agihouse.org/events/poe-previews-hackathon-20240817).
- **Modal Labs: Best Fine-Tuning Platform**: A member believes [Modal Labs](https://github.com/modal-labs/llm-finetuning) is the best platform for fine-tuning open-source LLMs.
   - This suggests that Modal offers valuable tools and resources for developers working with large language models.
- **Image Feature Store Speeds Up Training**: A simple feature store was built for an R&D team to store extracted features from images during online preprocessing.
   - This reduced training time by 30-60 minutes per training run, saving a considerable amount of time for model development.
- **Generic Feature Store for Diverse Models**: The feature store is generic and handles image IDs, extraction methods, and pointers to extracted features in an object store.
   - This allows it to accommodate diverse models, from small to large, ensuring efficient feature storage and retrieval.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mistral has struggles expanding beyond 8k**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed "cursed model merging".



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Automated Jupyter Notebook Exploration**: A member inquired about existing libraries or open-source projects that could help build a system to automate Jupyter Notebook modifications.
   - The goal is to create an agentic pipeline for swapping cells, generating variations, and validating outputs, similar to the Devin project but focused on a specific, small task.
- **Jupyter Notebook Automation: A Game Changer**: The proposed system would take a working Jupyter Notebook as input and modify it by swapping out cells.
   - This automated process would generate multiple versions, allowing for efficient exploration of different notebook configurations and potentially leading to improved results.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1273002100372275331)** (213 messages🔥🔥): 

> - `Hermes 2`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic`
> - `HQQ+` 


- **Hermes 2.5 outperforms Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** appears to perform better than **Hermes 2** in various benchmarks.
   - Hermes 2 scored a **34.5** on the MMLU benchmark whereas Hermes 2.5 scored **52.3**.
- **Mistral has struggles expanding beyond 8k**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed "cursed model merging".
- **Open Empathic Project Plea for Assistance**: A member appealed for help in expanding the categories of the **Open Empathic** project, particularly at the lower end.
   - They shared a [YouTube video on the Open Empathic Launch & Tutorial](https://youtu.be/GZqYr8_Q7DE) that guides users to contribute their preferred movie scenes from YouTube videos, as well as a link to the [OpenEmpathic project itself](https://dct.openempathic.ai/)
- **HQQ+ for Quantized Models**: **HQQ+** (**High Quantization Quality Plus**) allows for fine-tuning additional LoRa adapter layers onto quantized models to improve their accuracy and capability.
   - This technique has shown significant improvements in both 1-bit and 2-bit quantized models, particularly for smaller models like **Llama2-7B**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-2">Grok-2 Beta Release</a>: no description found</li><li><a href="https://www.githubstatus.com/">GitHub Status</a>: no description found</li><li><a href="https://huggingface.co/datasets/bigcode/the-stack">bigcode/the-stack · Datasets at Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1erv00p/elon_musks_ai_company_releases_grok2/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ertpa3/suscolumnr_on_lmsys_is_grok/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets/Gryphe/Opus-WritingPrompts">Gryphe/Opus-WritingPrompts · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/code_bagel_hermes-2.5">Replete-AI/code_bagel_hermes-2.5 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Replete-AI/Everything_Instruct_8k_context_filtered">Replete-AI/Everything_Instruct_8k_context_filtered · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1273036082078023701)** (6 messages): 

> - `Self-Promotion on Discord`
> - `Mosquitoes in the Wild` 


- **Discord Self-Promotion**: A member expressed concern about excessive self-promotion in the channel, reminding others that it goes against the server's rules.
   - They expressed their appreciation for the videos but requested a reduction in self-promotion going forward.
- **Mosquitoes: A Dangerous Threat**: A different member joked about the prevalence of mosquitoes in the wild, stating that one must either get used to them or risk dying from their bites.
   - They used a lighthearted tone to emphasize the potential dangers of mosquitoes in certain environments.



**Link mentioned**: <a href="https://tenor.com/view/%D0%BB%D0%B0%D1%82%D1%8B-armor-gif-25434286">латы Armor GIF - Латы Armor - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1272995028670615573)** (75 messages🔥🔥): 

> - `Unsloth Inference`
> - `GPU vs CPU`
> - `Vram`
> - `Custom Datasets`
> - `Alpaca-Cleaned Dataset` 


- **Unsloth struggles on CPU, needs VRAM**: A user reported difficulties running Unsloth inference on CPU, suggesting it likely requires a GPU.
- **Building custom datasets like Alpaca-Cleaned**: A user inquired about creating custom datasets similar to Alpaca-Cleaned, which addresses hallucination issues by removing instructions referencing internet data.
- **Instruct model fine-tuning**: A user shared their successful fine-tuning of an instruct model, mentioning that they used the default 60 training steps with a batch size of 8.
- **Saving models for Ollama**: A user sought guidance on saving models to be compatible with Ollama, specifically aiming to create a model file suitable for the platform.
- **Aspect-Based Sentiment Analysis with Unsloth Llama 3.1**: A user sought advice on evaluating a fine-tuned Unsloth Llama 3.1 model for Aspect Based Sentiment Analysis using the Semeval dataset on Hugging Face.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/sentiment-analysis-python">Getting Started with Sentiment Analysis using Python</a>: no description found</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/deepmind/code_contests">deepmind/code_contests · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.token">Loading methods</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1273299243800465582)** (20 messages🔥): 

> - `Model Card Typos`
> - `Model's Capabilities`
> - `Multi-Lingual LLM`
> - `Dataset-tool for RP`
> - `TheatreLM-v2.1-Characters` 


- **Model Card Critique**: A member pointed out several typos on the model card, specifically "generatal knoledge" and "arrabici", suggesting that English may not be the primary language of the developer.
   - The member also suggested using ChatGPT to rewrite the model card for a more professional look and better appeal to an English-speaking audience.
- **Goat Model Family Introduction**: A member announced the start of a "goat model family", a new series of finetuned models based on Llama 3.1.
   - This particular model is noted for its improved general knowledge and support for multiple languages, including Arabic and English.
- **Multi-Lingual LLM's Capabilities**: A user expressed doubt regarding the model's multilingual capabilities for translation.
   - Another member countered that LLMs are generally more proficient in multilingual tasks than other approaches, and this model could be useful for businesses needing multilingual support.
- **Dataset Tool for RP and Synthetic Data**: A user shared a link to a dataset tool designed for Role-Playing and synthetic dataset generation, capable of producing character cards, world information, and lorebooks.
   - The tool offers a range of features, including 'setting', 'setting_summarized', 'character', 'character_summary', 'character_card', 'character_name', 'story_introduction', 'story_outline', and 'lorebook'.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/kshabana/GOAT-llama3.1-v0.1">kshabana/GOAT-llama3.1-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/G-reen/TheatreLM-v2.1-Characters">G-reen/TheatreLM-v2.1-Characters · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1273295496386969620)** (3 messages): 

> - `Llama 3.1`
> - `Causal Mask`
> - `Causal Masking` 


- **Llama 3.1 Does Not Use Causal Masking**: A user inquired why the code for Llama 3.1, specifically at line 130 of the llama31.py file, does not use causal masking.
   - They wanted to know why this code does not include this technique.
- **Llama 3.1 Does Not Use Causal Masking**: A user inquired why the code for Llama 3.1, specifically at line 130 of the llama31.py file, does not use causal masking.
   - They wanted to know why this code does not include this technique.


  

---



### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/)** (1 messages): 

zhukov_80921: https://huggingface.co/datasets/bigcode/the-stack-v2 60tb of code
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1273210945564311635)** (4 messages): 

> - `Grok-2`
> - `ComfyUI`
> - `Open Source AI` 


- **Grok-2 Beta Released by X**: X has announced the beta release of **Grok-2**, a new AI model with state-of-the-art reasoning capabilities.
   - This model is a significant step forward in the field of AI reasoning, and it's likely to have a major impact on the industry.
- **Upscaling Images with FLUX AI in ComfyUI**: A YouTube video demonstrates how to upscale images using **FLUX AI** within the **ComfyUI** interface.
- **Intro to Open Source Large Language Models**: A talk given in July 2024 provides an accessible introduction to **open source large language models**.
   - The talk covers the basics of how AI works and how open-source models are changing the landscape of AI development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=arYBzLc3RV0">Elon Musks&#39;s Grok-2 Beta Release Announced by X</a>: Today we take a look at grok 2 beta release.It is their state of the art model on reasoning capabilites.This new model Grok-2 is a significant step forward f...</li><li><a href="https://www.youtube.com/watch?v=2cuFOXLHr4A&feature=youtu.be">Upscale Images with FLUX AI in ComfyUI</a>: Upscale Images with FLUX AI in ComfyUI</li><li><a href="https://youtu.be/vrO8tZ0hHGk">How AI Really Works - Intro to Open Source Large Language Models</a>: Recording of a talk I gave on July 27th, 2024 at the Vancouver Public Library in Vancouver, Canada. It&#39;s designed to be an accessible introduction to AI, wit...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1273344353338916885)** (2 messages): 

> - `Semantic Chunking`
> - `Regex Tokenization`
> - `Tokenizer API`
> - `Tiktoken Free Usage` 


- **Semantic Chunking Is Overrated**: A user on X argued that semantic chunking is overrated, and that a powerful regex can accurately segment text without the need for complex language models.
   - They claim that their 50-line, 2490-character regex is as powerful as it can be within the limitations of regex, and that it is faster and more cost-effective than semantic chunking.
- **Jina AI's Free Tokenizer API**: Jina AI offers a free API to tokenize text and segment long text into chunks.
   - This API leverages structural cues and heuristics to ensure accurate segmentation of text into meaningful chunks, even for complex content formats like Markdown, HTML, and LaTeX.
- **Free Tiktoken for Unlimited Usage**: Jina AI also offers free unlimited usage of tiktoken with rate limits, and even provides free embedding generation without charging for usage if you only use tokenization and chunking.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/JinaAI_/status/1823756993108304135">Tweet from Jina AI (@JinaAI_)</a>: Based. Semantic chunking is overrated. Especially when you write a super regex that leverages all possible boundary cues and heuristics to segment text accurately without the need for complex language...</li><li><a href="https://jina.ai/tokenizer">Tokenizer API</a>: Free API to tokenize text and segment long text into chunks.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1272998722648936512)** (155 messages🔥🔥): 

> - `Dataset Filtering and Scoring Tool`
> - `LMSYS Leaderboard`
> - `Grok-2`
> - `OpenAI ChatGPT-4o`
> - `HQQ+` 


- **New Dataset Filtering and Scoring Tool Released**: A member announced the release of a free and open-source dataset filtering and scoring tool with a focus on quick actions and hotkeys for generating samples and editing existing ones.
   - The tool allows users to input their API key, generate more samples similar to the current one, and score a large dataset efficiently.
- **LMSYS Leaderboard is a Scam?**: A member called out the LMSYS Leaderboard as a scam, citing the use of a made-up model string and manipulated scores.
   - Others agreed that the leaderboard is not reliable, noting that it's easy for users to manipulate results and that the model's performance may not be indicative of real-world capabilities.
- **Grok-2: The New Kid on the Block**: A member shared a blog post about the release of Grok-2, a new language model from X.ai, which claims to outperform Claude 3.5 Sonnet and GPT-4-Turbo on the LMSYS Leaderboard.
   - Grok-2 is available in beta on X and will be accessible through the enterprise API later this month. It features improved chat, coding, and reasoning capabilities, with a smaller sibling model, Grok-2 Mini, also available.
- **OpenAI ChatGPT-4o Improvements**: The latest OpenAI ChatGPT-4o (20240808) API has been tested and released, showcasing notable improvements in technical domains, particularly coding.
   - It has surpassed Google's Gemini-1.5-Pro-Exp on the LMSYS Leaderboard, reclaiming the #1 position with a score of 1314.
- **Quantized Models: HQQ+ for Improved Accuracy**: A member discussed the challenges of quantizing smaller pre-trained models at low bit-widths, highlighting the effectiveness of HQQ+ in restoring accuracy.
   - HQQ+ involves training additional LoRA layers onto quantized models, demonstrating a significant improvement in output quality, particularly with 1-bit and 2-bit quantization.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ide.x.ai/">PromptIde</a>: no description found</li><li><a href="https://x.com/lmsysorg/status/1823515224064098546">Tweet from lmsys.org (@lmsysorg)</a>: Exciting Update from Chatbot Arena!  The latest @OpenAI ChatGPT-4o (20240808) API has been tested under &#34;anonymous-chatbot&#34; for the past week with over 11,000 community votes.  OpenAI has now ...</li><li><a href="https://x.ai/blog/grok-2">Grok-2 Beta Release</a>: no description found</li><li><a href="https://tenor.com/view/hellfire-gif-10103277782914351064">Hellfire GIF - Hellfire - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/TobyPhln/status/1823598808309158353">Tweet from Toby Pohlen (@TobyPhln)</a>: Grok-2 mini is out now on X. Grok-2 (the big boi) will be rolled out very soon.  https://x.ai/blog/grok-2
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1273001297687347250)** (22 messages🔥): 

> - `FP8 training`
> - `Nemotron`
> - `FP8 vs BF16 performance`
> - `Mosaic AI`
> - `Character.AI` 


- **FP8 Faster Than BF16?**: A member asked about the largest FP8 pretraining scale used, suggesting FP8 would be faster than BF16 due to using FP8 multiplications.
   - Another member agreed to try FP8 training.
- **Nemotron Model Not Trained in FP8**: Nemotron-4-340B-Base was trained using 768 DGX H100 nodes, but the model was not trained in FP8.
   - It was confirmed that Nemotron *supports* mixed FP8 training, but it's unclear if it's actually used.
- **Databricks Mosaic AI Achieves FP8 Training Breakthrough**: Databricks' Mosaic AI platform achieved a 1.4x-1.6x speedup with FP8 training compared to BF16, showcasing the potential of FP8 for training large language models.
   - This was achieved with a 1000 training step test, and while FP8 is promising, it's still relatively new and there's more to learn about its applications.
- **Character.AI Utilizes Int8 Training**: Character.AI employs int8 training, not FP8, for their large language models.
   - The size of their model is not officially known, but an unofficial source suggests one of their earlier models is 108 billion parameters.
- **Mergekit Support for Gemma2 Models**: A member inquired about official support for Gemma2 models in Mergekit.
   - They reported experiencing difficulties when trying to use Mergekit with Gemma2 models, highlighting the need for clarification on its compatibility.



**Link mentioned**: <a href="https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8">Turbocharged Training: Optimizing the Databricks Mosaic AI Stack With FP8</a>: Benchmarking for training (dense) models at scale. We demonstrate great performance (very high MFU) and highlight our use of NVIDIA&#x27;s Transformer Engine, along with PyTorch FSDP and DTensor.

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/)** (1 messages): 

.bexboy: Yep
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1273000517114794170)** (134 messages🔥🔥): 

> - `AMD GPU`
> - `ControlNet`
> - `ComfyUI`
> - `SD3`
> - `Flux` 


- **AMD GPU Stability**: A member inquired about running Stable Diffusion on AMD GPUs and was informed that there are installation guides for both NVIDIA and AMD cards available on GitHub.
- **ControlNet Usage**: A user asked for help on how to use multiple ControlNets in a single generation, and several members suggested methods such as chaining them in ComfyUI or using a node that inputs multiple ControlNets.
- **ComfyUI vs InvokeAI**: A member expressed their preference for ComfyUI over Automatic1111, citing the increased control and speed offered by ComfyUI.
- **SD3 & Commercial Models vs Open Source**: A new user inquired about the pros and cons of SD3 compared to Flux, noting that SD3 is still in development and lacks complete functionality, while Flux has its own quirks and limitations.
- **SDXL vs SD 1.5**: A member asked for clarification on what SDXL 1.0 is and how it differs from SD 1.5.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com">Kaggle: Your Machine Learning and Data Science Community</a>: Kaggle is the world&#x2019;s largest data science community with powerful tools and resources to help you achieve your data science goals.</li><li><a href="https://huggingface.co/xinsir/controlnet-union-sdxl-1.0">xinsir/controlnet-union-sdxl-1.0 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/sigh-le-sad-cat-gif-16777715376630435814">Sigh Le GIF - Sigh Le Sad - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/heart-container-goddess-statue-totk-heart-container-totk-zelda-gif-891944359093961229">Heart Container Goddess Statue GIF - Heart container Goddess statue Totk heart container - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">stabilityai/stable-diffusion-xl-base-1.0 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 messages): 

louisgv: ChatGPT-4o-latest is now available: https://openrouter.ai/models/openai/chatgpt-4o-latest
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1273000641127780452)** (127 messages🔥🔥): 

> - `AgentQ`
> - `Infer`
> - `OpenRouter Pricing`
> - `ChatGPT-4o-Latest`
> - `Codeium` 


- **AgentQ blows Llama 3 out of the water!**: A new model named **AgentQ** claims to be 340% better than **Llama 3 70B BASE** but has no comparisons to **3.1, 405b, Mistral Large 2, or Claude 3**.
   - This company, **Infer**, seems to give zero fucks about **OpenRouter revenue** and [has published a paper on AgentQ](https://multion-research.s3.us-east-2.amazonaws.com/AgentQ.pdf).
- **ChatGPT-4o-Latest is just a new alias**: **ChatGPT-4o-Latest** is just a new handy alias for **gpt-4o-2024-08-06**, which is already on **OpenRouter**.
   - However, many members are still confused by the model's **optimization for ChatGPT** and **lack of proper documentation**.
- **OpenAI's sidebars are missing icons**: Members discussed changes in the sidebars of **platform.openai.com**.
   - One member reported that **two icons** disappeared from the sidebar: **one for threads and another one for messages**.
- **Grok 2 has arrived! (and is surprisingly good)**: **Grok 2**, an early version of **xAI's** model, has secured the #3 spot on the **LMSys Arena leaderboard**.
   - It excels in **Coding**, **Hard Prompts**, and **Math** and even matches **GPT-4o** on the leaderboard.
- **Anthropic's Prompt Caching: the future of efficient AI?**: **Anthropic** has just released **prompt caching** for their **Claude 3 models**.
   - This feature can reduce costs by up to 90% and latency by up to 85% for long prompts, and could be integrated into **OpenRouter**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.anthropic.com/news/prompt-caching">Prompt caching with Claude</a>: Prompt caching, which enables developers to cache frequently used context between API calls, is now available on the Anthropic API. With prompt caching, customers can provide Claude with more backgrou...</li><li><a href="https://openrouter.ai/models/openai/chatgpt-4o-latest">ChatGPT-4o - API, Providers, Stats</a>: Dynamic model continuously updated to the current version of [GPT-4o](/models/openai/gpt-4o) in ChatGPT. Intended for research and evaluation. Run ChatGPT-4o with API</li><li><a href="https://apipie.ai/">no title found</a>: no description found</li><li><a href="https://x.com/lmsysorg/status/1823599819551858830">Tweet from lmsys.org (@lmsysorg)</a>: Woah, another exciting update from Chatbot Arena❤️‍🔥  The results for @xAI’s sus-column-r (Grok 2 early version) are now public**!  With over 12,000 community votes, sus-column-r has secured the #3 s...</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>: OpenRouter Incident History</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-2024-08-06">GPT-4o (2024-08-06) - API, Providers, Stats</a>: The 2024-08-06 version of GPT-4o offers improved performance in structured outputs, with the ability to supply a JSON schema in the respone_format. Read more [here](https://openai. Run GPT-4o (2024-08...</li><li><a href="https://codeium.com/blog/codeium-dream-bigger">Dream Bigger</a>: The Codeium mission, Cortex and Forge launches, and detailed vision.
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1273158062949269514)** (80 messages🔥🔥): 

> - `Mojo performance`
> - `Mojo benchmark`
> - `Rust vs C/C++`
> - `Mojo vs Go`
> - `Mojo threading` 


- **Mojo Benchmarks**: A member questioned why [Mojo benchmarks](https://modular.com/mojo/) only compared to C, and asked if benchmarks against Go and Rust would be possible.
- **Mojo's Performance and Multi-threading**: Mojo is currently single-threaded for performance, but doesn't have a good multi-threading API yet, outside of launching parallel kernels with MAX.
- **Mojo vs Go/Rust: Network Speed**: A member asked whether Mojo is faster than Go in terms of network speed, and if it can handle heavy tasks like Rust.
- **Mojo's Future Direction: Multiprocessing and Networking**: A member inquired if Mojo has any plans to support multiprocessing or improve network handling, given the importance of network performance for their work.
- **Mojo and MAX: A Deeper Look**: A member questioned the nature of MAX, unsure if it's a platform, module, or something else. Another member explained that MAX is a platform, with Mojo as one of its components, and that it includes GPUs, graph API, and other components.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=6huytcgQgk8">MAX + Mojo Community Meetings #6</a>: This is a video about MAX &amp; Mojo Community Meetings #600:00 Introduction00:27  Small buffer and string optimizations13:04 DuckDB bindings in Mojo23:15 MAX an...</li><li><a href="https://www.youtube.com/watch?v=6huytcgQgk8.">MAX + Mojo Community Meetings #6</a>: This is a video about MAX &amp; Mojo Community Meetings #600:00 Introduction00:27  Small buffer and string optimizations13:04 DuckDB bindings in Mojo23:15 MAX an...</li><li><a href="https://c9x.me/x86/html/file_module_x86_id_279.html">Sun: x86 Instruction Set Reference</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1273097485732155462)** (32 messages🔥): 

> - `Mojo RPM Build`
> - `Mojo on RHEL machines`
> - `Magic CLI`
> - `Mojo as a Conda Package`
> - `Mojo Language Version Management` 


- **Mojo RPM Build: When can we get it?**: A member inquired about the ETA for a Mojo .rpm build, expressing a desire to run Mojo on RHEL machines without containerd.
   - Another member suggested that the Magic CLI is currently being treated as a solution for these issues, and that a statically linked build with a RHEL 8 minimum kernel might be a better option than an RPM, as it would allow for packaging on other distributions.
- **Conda's Rattler: Is it the right way to distribute Mojo?**: One member expressed concern about using Conda's Rattler to deliver Mojo, stating that they dislike needing additional tools to install a language.
   - They compared this approach to having to install a tool (`dnf install tool`) and then use that tool to install the language (`tool install language`), but acknowledged that it may be a suitable first step.
- **Tradeoffs of Distributing Mojo with Conda's Rattler**: Another member explained that the use of Conda's Rattler for Mojo distribution is a tradeoff due to the reliance on OS-provided versions of C, C++, and Python, which leads to security patch requests based on RHEL's support schedule.
   - They suggested that a Rustup/Conda-like tool in the repositories would be more suitable for developers, allowing them to install new versions of the compiler, while end-products could be packaged in the repositories for users.
- **Mojo Development Tooling: What's the ideal setup?**: Members discussed the ideal setup for Mojo development tooling, with the aim of providing a smooth experience for both developers and users.
   - One member proposed that Redhat packages the resulting programs, while leaving the development tools to the language ecosystem, suggesting that a global install of Magic with a custom configuration file should be the only requirement, allowing it to utilize a local cache for compilers and dependencies.
- **Magic CLI: Like Pixi, but for Mojo?**: One member expressed hope that the Magic CLI will function similarly to Pixi in terms of downloading dependencies, emphasizing their reliance on corporate cachers.
   - They envisioned a future where Magic operates like Rust's Cargo but also manages the language version, requiring only a global install of Magic, and expressed a preference for this approach over a shell-based method for pure Mojo projects.



**Link mentioned**: <a href="https://youtu.be/6huytcgQgk8">MAX + Mojo Community Meetings #6</a>: This is a video about MAX &amp; Mojo Community Meetings #600:00 Introduction00:27  Small buffer and string optimizations13:04 DuckDB bindings in Mojo23:15 MAX an...

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1273001575618711592)** (88 messages🔥🔥): 

> - `Gemini Advanced`
> - `Gemini Live Talk`
> - `GPT-4o Advanced Voice Mode`
> - `Model Limitations`
> - `Prompt Engineering` 


- **Gemini Advanced - Where's the Live Talk?**: A user paid for Gemini Advanced but couldn't find the option for live talk in the app.
   - Other users noted that it's likely in Alpha testing and may be released to all Advanced subscribers eventually, similar to how OpenAI gradually released access to their models.
- **Model Limitations - Is It All About Tokenization?**: A discussion arose about the limitations of LLMs, specifically the challenge of getting them to perform well with complex tasks and understand nuances beyond tokenization.
   - Some argued that tokenization is a fundamental weakness of LLMs, while others suggested using prefill and multi-step prompts to improve performance.
- **Prompt Engineering - Finding the Right Recipe**: A user asked for ways to make ChatGPT go beyond its usual patterns and give more diverse, non-repetitive answers.
   - Suggestions included using "Customize ChatGPT" to tell it not to be repetitive, utilizing prompt engineering techniques, and even providing an example in the prompt and asking ChatGPT to critique its own reply to generate better instructions.
- **Custom GPT - Why Are My Instructions Ignored?**: A user reported issues with a custom GPT model not remembering all the rules and words they specified in its training.
   - Other users speculated that it might be due to exceeding the context token limit, limiting the model's ability to store all instructions, or possibly a deliberate limitation on the model's capabilities.
- **Watermarks and AI Detection - Sneaky Strategies**: A user asked about creating watermarks to deter students from using AI to answer assignments.
   - The discussion included suggestions for using near-white text on a white background for image scanners, recognizing that AI models can be instructed to do almost anything with the right prompts.



**Link mentioned**: <a href="https://www.twixify.com/post/most-overused-words-by-chatgpt">124+ Most OVERUSED Words By ChatGPT In 2024</a>: no description found

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1273212479459168267)** (4 messages): 

> - `Vision's performance`
> - `Image Deformations`
> - `Vision's limitations` 


- **Vision Struggles with Simple Tasks**: A user expressed surprise at **Vision's** poor performance in detecting whether a subject is looking right or left, and also in identifying deformations in images.
   - The user even gave **Vision** a highly deformed image, yet it claimed the image was perfectly fine.
- **Vision's Limitations Exposed**: This incident highlights a clear limitation of **Vision**'s current capabilities, specifically in recognizing image deformations.
   - This prompts questions about the reliability of Vision for tasks requiring accurate assessment of image integrity.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1273149312930680832)** (8 messages🔥): 

> - `Critical Thinking Techniques`
> - `GPTs and Web Searching`
> - `Reclaiming Business Assets`
> - `Developer Mode Prompts` 


- **Critical Thinking Techniques: A Collection**: A member is compiling a comprehensive prompt combining various critical thinking techniques, including Socratic Method, Bloom's Taxonomy, Paul-Elder Model, Six Thinking Hats, and Scientific Method.
   - They are also integrating methods like TRIZ, deductive and inductive reasoning, Fermi estimation, systems thinking, lateral thinking, heuristic analysis, Bayesian thinking, mind mapping, SWOT analysis, and root cause analysis.
- **Exploring Web Search with GPTs**: A member inquired about "webbrowsergpt", a GPT that can search the web. 
   - Another member confirmed that while it can be accessed in the "explore GPTs" section, it's also possible to nudge a general GPT to search by providing specific instructions.
- **Prompt for Reclaiming Business Assets**: A member asked for a prompt or GPT to help create a process for reclaiming business assets.
   - They had previously searched for relevant prompts using keywords like "policy", "human resources", and "compliance", but found no satisfactory results.
- **Effective Developer Mode Prompts**: A member sought a working prompt for "developer mode".


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1273149312930680832)** (8 messages🔥): 

> - `Critical Thinking Techniques`
> - `Prompt Engineering`
> - `GPTs and Web Search`
> - `Business Asset Reclamation`
> - `Developer Mode` 


- **Critical Thinking Techniques: A Comprehensive List**: A user is compiling a list of critical thinking methods for a comprehensive prompt that will combine various approaches, including the Socratic Method, Bloom's Taxonomy, Paul-Elder Model, Six Thinking Hats, and the Scientific Method.
   - The user is also exploring additional methods, such as TRIZ, deductive and inductive reasoning, Fermi estimation, systems thinking, lateral thinking, heuristic analysis, Bayesian thinking, mind mapping, SWOT analysis, and root cause analysis.
- **Prompt Engineering: 5W+1H and OSINT/HUMINT Integration**: The prompt will start with a simple keyword that triggers a "5W+1H" inquiry (what, where, when, who, why, and how) and will include methods from OSINT (Google/Bing) and HUMINT (user interviews).
   - The goal is to integrate these techniques into a comprehensive prompt that encourages critical thinking and generates informative responses.
- **GPTs and Web Search: "Web Browser GPT"**: A user inquired about a GPT that can always search the web, and was directed to the "Explore GPTs" section where a GPT dedicated to web searching is available.
   - While not strictly necessary, using this GPT can provide better web search results than manually nudging a general GPT to perform web searches.
- **Business Asset Reclamation: Seeking a Prompt or GPT**: A user is looking for a prompt or GPT that can help create a process for reclaiming business assets.
   - The user has attempted searches for policy, human resources, and compliance, but has not found any relevant results.
- **Developer Mode: Searching for a Functional Prompt**: A user is looking for a prompt for developer mode that actually works.
   - The user has not specified the desired functionality of the prompt, making it difficult to provide a definitive answer.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1273006284958076938)** (62 messages🔥🔥): 

> - `Perplexity Performance Issues`
> - `Perplexity Pro Lag`
> - `Sonnet vs Opus vs Perplexity`
> - `Perplexity's Website Update`
> - `Perplexity Support Team` 


- **Perplexity Users Reporting Performance Issues**: Several users have reported that Perplexity's website has become significantly slower, with some finding it difficult to replace Sonnet or Opus as their default search engine.
   - These performance issues have led to a general feeling of lag and even complete stalling for some users, particularly those using Perplexity Pro.
- **Perplexity Pro Users Experience Lag**: Users of Perplexity Pro have reported that the service has been lagging in terms of response times, with some experiencing complete stalling.
   - This has led to complaints from paying customers who are expecting a more reliable and responsive service.
- **Sonnet vs Opus vs Perplexity**: A user shared their opinion that Perplexity is too slow to replace Sonnet as their default search engine, and not sure if it's even better than Opus.
   - This sparked a discussion among other users about the relative performance and usability of these search engines.
- **Perplexity's Website Update Has Mixed Reception**: A Perplexity team member confirmed that they had been working on fixing bugs and issues over the past few days, including a bug affecting toggles.
   - Despite these fixes, some users are still experiencing performance problems and a general sense of disruption, leading to calls for a rollback to the previous build.
- **Perplexity Support Team Facing Increased Workload**: Perplexity's support team has been receiving a large amount of feedback and reports from users about the recent website update.
   - Some users have expressed concern about the team being overworked, urging them to prioritize fixing the issues and avoid burning themselves out.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.co">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-huge-128k-online/api">Perplexity: Llama 3.1 Sonar 405B Online – Run with an API</a>: Sample code and API for Perplexity: Llama 3.1 Sonar 405B Online - Llama 3.1 Sonar is Perplexity&#x27;s latest model family. It surpasses their earlier Sonar models in cost-efficiency, speed, and perfo...</li><li><a href="https://tenor.com/bcmTe.gif">Morning Jerry GIF - Morning Jerry Jerry Mouse - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/singularity/s/Vb3T5NLjxN">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/86OuyavwUnc?si=bPYa7SjT_tFSkgMl">I Battled Perplexity AI vs ChatGPT - You Won’t Believe Who Won!</a>: In this video, I dive deep into the ongoing debate of Perplexity AI vs ChatGPT, two powerful tools in the world of artificial intelligence. As someone who&#39;s ...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1273076850293477428)** (7 messages): 

> - `Shareable Threads`
> - `Radioactive Shoe Fitting`
> - `Perplexity Pro` 


- **Threads Must Be Shareable**: Perplexity AI reminded a user that their thread needs to be `Shareable`.
   - They provided a link to a Discord channel where the user can find more information about how to make their thread shareable.
- **Shoe Fitting X-Ray Devices**: Shoe-fitting fluoroscopes, which used X-ray technology to visualize foot bones inside shoes, were popular from the 1920s to the 1970s.
   - Despite their initial appeal, these devices exposed customers and employees to dangerous levels of radiation, leading to their eventual ban.
- **Perplexity Pro Features**: A user was prompted to upgrade to Perplexity Pro.
   - Perplexity Pro offers features such as image upload, smarter AI, and more Pro Search.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/hey-give-me-a-simple-and-easy-3rd3mO8.SRWDJvpKCvXroA">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/clean-this-up-to-look-professi-DhKZLaAhRte.5FFbrJkgDA#1">Clean this up to look professional and minimalist. 

Barter...</a>: Certainly! Here&#x27;s a professional and minimalist version of the text:    Barter transactions, involving the exchange of goods or services without money, are...</li><li><a href="https://www.perplexity.ai/search/articles-about-ia-and-climate-Ihlz3UnNTbKqVaM2.dRveA">Articles about ia and climate change</a>: Recent articles highlight the dual role of artificial intelligence (AI) in addressing climate change, emphasizing both its potential benefits and...</li><li><a href="https://www.perplexity.ai/page/radioactive-shoe-fitting-machi-7AA3MwTtQv.sh5zb70slwQ">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/page/biographie-de-lucas-gulino-xc22ID22TfmIhy35RUvB1Q">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1273018172932362355)** (38 messages🔥): 

> - `Perplexity API response formatting`
> - `Function calling`
> - `JSONSchema7 validation`
> - `Model prompt engineering`
> - `Markdown to HTML conversion` 


- **User seeks HTML formatted API responses**: A user inquired about obtaining HTML formatted responses from the Perplexity API.
   - They've tried various system prompts, but haven't found a successful approach for consistent HTML formatting.
- **Markdown2 Module for HTML Conversion**: Another user suggested using the `markdown2` module, which converts text into HTML.
   - This approach may eliminate the need for prompt engineering to specifically instruct the model to produce HTML output.
- **Prompt Engineering Trade-offs**: A user observed that focusing the model on producing valid HTML might hinder its ability to provide quality responses based on search results.
   - They suggested that post-processing the model's output with a markdown-to-HTML converter might strike a balance between response quality and HTML formatting.
- **Vague Prompts and Model Behavior**: The user discussed the impact of vague prompts like "What is SAP?" on the model's responses.
   - They noted that while such prompts may elicit HTML-formatted responses, more specific and detailed prompts generally lead to better, non-HTML responses, suggesting a potential trade-off between prompt specificity and desired HTML output.
- **Model's HTML Output Capabilities**: A user observed that the `llama-3.1-sonar-large-128k-online` model sometimes includes citations in its responses, suggesting potential for consistent HTML output with further fine-tuning.
   - They are exploring ways to ensure consistent inclusion of citations in the model's output, potentially by modifying the prompt or using a more specialized model.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1272993847299014707)** (71 messages🔥🔥): 

> - `Gemma-2`
> - `LM Studio on Android`
> - `LM Studio on external hard drive`
> - `LM Studio on Ubuntu`
> - `Multimodal LLMs` 


- **Gemma-2 lacks system prompt**: Gemma-2 models miss a critical component which is system prompt.
   - That's why it does not stick to user prompts. Otherwise, it's an excellent model for its size.
- **LM Studio on Android**: A user asked if there is an Android app that can connect to LM Studio server.
- **Moving LM Studio to external hard drive**: A user asked if they could run LM Studio on an external hard drive, specifically to store the models due to space constraints.
   - They were advised that they can move the directory where they want or create a symbolic link that points to any drive, even on the network, using the command `mklink /D "C:\Users\xxxx\AppData\Local\LM-Studio" "Your destination folder"`.
- **LM Studio on Ubuntu**: A user asked if a GUI OS is necessary to run LM Studio.
   - They were told that LM Studio is a desktop application, but some people have gotten it working on Ubuntu, although it's not designed to be a headless system. Alternatively, they can set up a VNC server.
- **Multimodal LLMs**: A user asked if there is a multimodal LLM server by LM Studio.
   - The response was that while LM Studio does not generate images, it can load multi-modal models that can 'see' images.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Arki05/Grok-1-GGUF">Arki05/Grok-1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/kshabana/GOAT-llama3.1-v0.1">kshabana/GOAT-llama3.1-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1272994051708293290)** (28 messages🔥): 

> - `GPU Copper Mod`
> - `GPU Bios Flashing`
> - `Text Classification Model Compatibility`
> - `GPU Offloading` 


- **Copper Modding an RTX 2070**: A user is modifying their **Asus ROG Strix RTX 2070 8GB oc** by adding a copper mod to improve performance.
   - They are using the copper to improve heat dissipation from the memory chips, as **bandwidth** plays a major role in **LLM inference speed**.
- **Flashing a RTX 2070 Bios to a 2080**: A user is considering flashing the **BIOS** of their **RTX 2070** to a **2080**, but acknowledges that this is a risky process.
   - The user highlights the importance of having a backup and the ability to run the card on another machine in case of a failed flash.
- **Text Classification Model Compatibility for a Laptop**: A user is searching for a **text classification model** that is compatible with their laptop, which has **16GB RAM** and an **i7 CPU**, but limited GPU resources.
   - The conversation suggests that the **Gemma 2b** model might be a suitable choice, as models with larger memory requirements may struggle to run on the laptop's specifications.
- **GPU Offloading Limitations**: A user is experiencing issues with **GPU offloading** on a system with a **GTX 760** for displays and a **Tesla T4** for compute.
   - The user is running a **Debian 12** distribution with the **v470.x driver**, which is outdated and may be contributing to the offloading limitations. **Llama.cpp** requires at least **v535** driver.


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1273008232864153620)** (79 messages🔥🔥): 

> - `MultiOn System Prompt Leak`
> - `AnswerAI ColBERT`
> - `Gemini Live Demo`
> - `GPT-4o Improvements`
> - `Grok 2` 


- **Pliny Threatens to Leak MultiOn System Prompt**: Pliny, a prominent AI researcher, threatened to leak the full MultiOn system prompt on GitHub if DivGarg9 didn't provide an answer within 15 minutes.
   - This follows an ongoing debate on Twitter regarding the capabilities of various AI models and their performance on specific benchmarks.
- **AnswerAI ColBERT: Small but Mighty**: AnswerAI has released a small but powerful version of their ColBERT model, called answerai-colbert-small-v1, that beats even bge-base on BEIR benchmark.
   - This demonstrates the effectiveness of smaller models in achieving high performance in certain tasks, potentially offering a more cost-effective solution.
- **Gemini Live Demo Draws Criticism**: Swyxio criticized Google's Gemini Live Demo on YouTube, deeming it "cringe".
   - This was followed by a discussion on the potential of Gemini, with some emphasizing its ability to enhance voice assistants while others remain skeptical.
- **GPT-4o Improvements Surpass Gemini**: OpenAI's latest GPT-4o model has been tested in the Chatbot Arena and has surpassed Google's Gemini-1.5-Pro-Exp in overall performance.
   - The new GPT-4o model has demonstrated significant improvement in technical domains, particularly in Coding, Instruction-following, and Hard Prompts, solidifying its position as the top performer on the leaderboard.
- **Grok 2 Makes its Debut**: xAI has released an early preview of Grok-2, a significant advancement from its previous model, Grok-1.5, showcasing capabilities in chat, coding, and reasoning.
   - Grok-2 has been tested on the LMSYS leaderboard and is outperforming both Claude 3.5 Sonnet and GPT-4-Turbo, although it is not yet available through the API.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenAIDevs/status/1823510395619000525">Tweet from OpenAI Developers (@OpenAIDevs)</a>: This model is also now available in the API as `chatgpt-4o-latest`. We recommend `gpt-4o-2024-08-06` for most API usage, but are excited to give developers access to test our latest improvements for c...</li><li><a href="https://x.com/lmsysorg/status/1823515224064098546">Tweet from lmsys.org (@lmsysorg)</a>: Exciting Update from Chatbot Arena!  The latest @OpenAI ChatGPT-4o (20240808) API has been tested under &#34;anonymous-chatbot&#34; for the past week with over 11,000 community votes.  OpenAI has now ...</li><li><a href="https://x.com/lmsysorg/status/1823599819551858830?s=46">Tweet from lmsys.org (@lmsysorg)</a>: Woah, another exciting update from Chatbot Arena❤️‍🔥  The results for @xAI’s sus-column-r (Grok 2 early version) are now public**!  With over 12,000 community votes, sus-column-r has secured the #3 s...</li><li><a href="https://aider.chat/docs/leaderboards/#llm-code-editing-skill-by-model-release-date">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://simonwillison.net/2024/May/14/context-caching-for-google-gemini/">Context caching for Google Gemini</a>: Another new Gemini feature announced today. Long context models enable answering questions against large chunks of text, but the price of those long prompts can be prohibitive - $3.50/million for …</li><li><a href="https://www.patched.codes/blog/a-comparative-study-of-fine-tuning-gpt-4o-mini-gemini-flash-1-5-and-llama-3-1-8b">A comparative study of fine-tuning GPT-4o-mini, Gemini Flash 1.5 and Llama-3.1-8B</a>: We compare fine-tuning GPT-4o-mini, Gemini Flash 1.5, and Llama-3.1-8B models using a custom vulnerability fixes dataset, with GPT-4o-mini showing the most significant improvement and setting a new st...</li><li><a href="https://www.zdnet.com/article/gemini-live-is-finally-available-heres-how-you-can-access-it-and-why-youll-want-to/">Gemini Live is finally available. Here&apos;s how you can access it (and why you&apos;ll want to)</a>: Want to have an open-ended, complex conversation with your device? Gemini Live can help.</li><li><a href="https://x.com/elonmusk/status/1823605120334192789">Tweet from Elon Musk (@elonmusk)</a>: @latentspacepod @xai Coming soon</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching">Prompt Caching (beta) - Anthropic</a>: no description found</li><li><a href="https://x.com/bfl_ml/status/1823608452096123120?s=46">Tweet from Black Forest Labs (@bfl_ml)</a>: FLUX.1 is now part of Grok-2!  Quoting ibab (@ibab)   Huge thank you to the @bfl_ml team, who scaled up their FLUX.1 inference API to support the Grok-2 release today!</li><li><a href="https://x.com/bnj/status/1823784845879005407?s=46">Tweet from Ben South (@bnj)</a>: Introducing Subform—a canvas for branching and remixing AI chats.  Built in two weeks using $400 of Claude credits, blood, sweat and tears  Link below</li><li><a href="https://x.ai/blog/grok-2">Grok-2 Beta Release</a>: no description found</li><li><a href="https://x.com/bclavie/status/1823405960406462739?s=46">Tweet from Benjamin Clavié (@bclavie)</a>: 🎉Happy to finally release answerai-colbert-small-v1: the small but mighty @answerdotai ColBERT.  It might not be able to count the number of &#34;r&#34;s in words, but it can definitely find the inst...</li><li><a href="https://x.com/Kyrannio/status/1822127051215954065">Tweet from Kiri (@Kyrannio)</a>: Alright, sus-column-r is actually really, really good.   Check it out here, side-by-side with gpt-4o-2024-08-06.  I used my Omega Claude prompt as the prompt, and it simulated my own dialogue with Ome...</li><li><a href="https://x.com/wbhub/status/1823798224521736322?s=46">Tweet from Will Hubbard (@wbhub)</a>: $1M ARR … $2B exit</li><li><a href="https://x.com/elder_plinius/status/1823436142802264117?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Pliny the Prompter 🐉 (@elder_plinius)</a>: ## System Prompt/Custom Instructions  ## Goal  Let&#39;s play a game - You are an expert agent named MULTI·ON developed by &#34;MultiOn&#34; controlling a browser (you are not just a language model an...</li><li><a href="https://x.com/alexalbert__/status/1823751966893465630">Tweet from Alex Albert (@alexalbert__)</a>: We just rolled out prompt caching in the Anthropic API.  It cuts API input costs by up to 90% and reduces latency by up to 80%.  Here&#39;s how it works:</li><li><a href="https://youtu.be">YouTube</a>: no description found</li><li><a href="https://x.com/rak_garg/status/1823436589017784572?s=46">Tweet from Rak Garg (@rak_garg)</a>: Pure foundation model biz is pretty poor:  Spend $10^8 upfront on data + training (risk)  Try to make it back on inference (too cheap)  Guaranteed depreciation to $0  Two ways out: 1) high margin apps...</li><li><a href="https://www.youtube.com/live/N_y2tP9of8A?t=1692s">#MadeByGoogle ‘24: Keynote</a>: Watch now for updates on Google AI and the newest Pixel devices, including the #Pixel9 Pro and Pixel 9 Pro Fold.To watch this keynote with American Sign Lang...</li><li><a href="https://youtu.be/f9YleTc8AwE">The Brief History of AI Agents (2023-2024)</a>: a quick lightning talk given at Cohere Agent Build Day at Cohere&#39;s offices in SF. https://lu.ma/gptdzwhe?tk=sUyT7n</li><li><a href="https://platform.deepseek.com/api-docs/news/news0802">DeepSeek API introduces Context Caching on Disk, cutting prices by an order of magnitude | DeepSeek API Docs</a>: In large language model API usage, a significant portion of user inputs tends to be repetitive. For instance, user prompts often include repeated references, and in multi-turn conversations, previous ...</li><li><a href="https://buttondown.email/ainews/archive/ainews-gemini-live/">[AINews] Gemini Live</a>: Lots of little $20/month subscriptions for everything in your life are all you need. AI News for 8/12/2024-8/13/2024. We checked 7 subreddits, 384 Twitters...</li><li><a href="https://codeium.com/blog/codeium-dream-bigger">Dream Bigger</a>: The Codeium mission, Cortex and Forge launches, and detailed vision.
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1273158320953233410)** (60 messages🔥🔥): 

> - `Grok-2`
> - `Anthropic API`
> - `Anthropic's Turnaround`
> - `DeepSeek`
> - `Llama` 


- **Grok-2:  New Model Outperforms GPT-4 & Claude 3.5**: A new model, **Grok-2**, has been released by x.ai, and is currently in beta on 𝕏. It outperforms both **Claude 3.5 Sonnet** and **GPT-4-Turbo** on the LMSYS leaderboard.
- **Anthropic API: Prompt Caching Cuts Costs & Latency**: Anthropic has rolled out [prompt caching](https://x.com/alexalbert__/status/1823751966893465630) in their API, which can reduce API input costs by up to **90%** and latency by up to **80%**.
- **Anthropic's Turnaround from 'Cringe' to Cutting Edge**: Anthropic is considered to be a turnaround story, moving from a less popular organization to one considered to be at the forefront.
- **DeepSeek: A Comparison to Anthropic's API**: Google was first to implement prompt caching, but they charge for storing the prompt (per hour).
- **Anthropic API: Issues & Opportunities**: Members discussed the Anthropic API and its limitations, including a slow API and a lack of a projects & artifacts API.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/alexalbert__/status/1823751966893465630">Tweet from Alex Albert (@alexalbert__)</a>: We just rolled out prompt caching in the Anthropic API.  It cuts API input costs by up to 90% and reduces latency by up to 80%.  Here&#39;s how it works:</li><li><a href="https://x.ai/blog/grok-2">Grok-2 Beta Release</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1273128349878718556)** (7 messages): 

> - `GPT-4o improvements`
> - `ChatGPT API` 


- **ChatGPT App Improves GPT-4o Model**: OpenAI announced an improvement to the GPT-4o model, not a new frontier model, stating they continue to iterate on existing models while working on longer term research.
   - They provided [release notes](https://help.openai.com/en/articles/9624314-model-release-notes) for this improvement.
- **ChatGPT API Release: `gpt-4o-latest`**: The new model is available via the ChatGPT API as `gpt-4o-latest`, aimed at developers and researchers to explore OpenAI's latest research.
   - Pricing is currently unknown, and a previous model, `gpt-4o-2024-08-06`, is recommended for API usage (e.g., function calling, instruction following).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/michpokrass/status/1823512031988998653?s=46">Tweet from Michelle Pokrass (@michpokrass)</a>: @aidan_mclau @OpenAIDevs chatgpt-4o-latest will track our 4o model in chatgpt, and is a chat-optimized model. our model from last week (gpt-4o-2024-08-06) is optimized for api usage (eg. function call...</li><li><a href="https://x.com/chatgptapp/status/1823509890976866766?s=46">Tweet from ChatGPT (@ChatGPTapp)</a>: to be clear, this is an improvement to GPT-4o and not a new frontier model. we continue to iterate on existing models while working on longer term research. some release notes: https://help.openai.com...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1273279196914909365)** (1 messages): 

> - `AI Copyright Discourse`
> - `Oligopoly`
> - `ACL2024NLP` 


- **AI Copyright Discourse Ends In Oligopoly**: A user shared a link to a tweet by @asayeed, stating that this is the end result of AI copyright discourse: oligopoly. 
   - This statement was made in the context of #ACL2024NLP, suggesting a potential topic of discussion at the upcoming conference.
- **A User's Perspective on AI Copyright Discourse**: A user remarked that they enjoy being informed about AI copyright discourse before it is widely discussed in mainstream outlets.



**Link mentioned**: <a href="https://x.com/asayeed/status/1823648027674075430">Tweet from Asad Sayeed @asayeed@zirk.us (@asayeed)</a>: this is the end result of AI copyright discourse: oligopoly #ACL2024NLP

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1273023843664924816)** (3 messages): 

> - `LlamaIndex Box Reader`
> - `Relik Knowledge Graph`
> - `Azure AI Search RAG System` 


- **LlamaIndex Box Reader Integration**: LlamaIndex now offers Box Readers that enable integration of Box documents into your LLM workflows. 
   - These readers offer four different options for data extraction, authenticate with Box via CCG or JWT methods, and allow you to load, search, and retrieve Box files and metadata within your LLM.
- **Build Knowledge Graphs with Relik**: Relik, a framework for fast and lightweight information extraction models, simplifies knowledge graph construction without requiring expensive LLMs. 
   - Learn how to set up a pipeline for entity extraction, and create a knowledge graph using Relik.
- **RAG System with Azure AI Search**: LlamaIndex Workflows can integrate with Azure AI Search and Azure OpenAI to build a robust Retrieval-Augmented Generation (RAG) system. 
   - Learn how to implement custom data connectors for Azure AI Search and use LlamaIndex Workflows to create a robust RAG system.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1273006072533352552)** (64 messages🔥🔥): 

> - `Inconsistent OpenAI Responses`
> - `Prompt Engineering`
> - `LlamaIndex`
> - `Chatbot Memory`
> - `GraphRAG` 


- **Inconsistent OpenAI Responses**: A user is encountering inconsistent results from an OpenAI prompt, with the model sometimes providing a negative answer (referring to the helpdesk) even when it could have answered the question.
   - This is likely due to the probabilistic nature of LLMs, but the user is trying to improve the prompt's clarity to reduce this inconsistency.
- **Prompt Engineering for Consistency**: The user suggests changing the question to "Could you please provide me the complete list of commands in the document?" to improve clarity and reduce the number of incorrect responses.
   - This change successfully resulted in 100% accuracy, suggesting that providing a clear output format can significantly impact model performance.
- **LlamaIndex Agent and Tool Calls**: A user is trying to understand how to handle tool calls within a LlamaIndex Agent, specifically when using the `astream_chat()` function.
   - The discussion revolves around whether to send tool calls in the initial response or buffer them until the final response, with the consensus being that tool calls should be sent first in the `message.tool_calls` field of the LLM response.
- **Chatbot Memory using LlamaIndex**: A user is seeking guidance on building a RAG chatbot that can store conversations permanently in a vector database, enabling it to remember past interactions.
   - They are looking for examples of open source projects or methods to implement this functionality within LlamaIndex, exploring the possibility of extending the chatbot's memory beyond the current chat history buffer.
- **Extracting Image Content from PDFs**: A user is looking for an open-source solution to extract image content from PDFs for use in RAG, specifically to add image captions to the end of page content.
   - They have explored tools like `ImageVisionLLMReader` and `LlamaParse` but are struggling to find a suitable method, particularly one that can be hosted locally and doesn't require sending data to a third-party service.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llamahub.ai/l/readers/llama-index-readers-file?from=">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/?h=simpledire#extending-to-other-file-types">SimpleDirectoryReader - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/">Sub Question Query Engine - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1273006213537468539)** (17 messages🔥): 

> - `Grok 2`
> - `xAI`
> - `Cohere`
> - `OpenAI`
> - `Model Performance` 


- **Grok 2 Released!**: xAI has officially released **Grok 2**, a new language model previously tested under the names **sus-column-r** and **column-r** on the **LMSYS chatbot arena**.
   - They are hiring for their post-training team, citing a desire to build **useful and truthful AI systems**.
- **xAI's Hiring Spree**: **xAI** is a small team, still an order of magnitude smaller than other players in the field.
   - They're looking for exceptional talent to join them on their journey to build better AI. They are actively hiring for their post-training team.
- **Conspiracy Theories Run Wild**: A member noted that **Grok 2** was immediately assumed to be **Cohere's**, **OpenAI's**, or even another conspiracy theory.
   - This highlights the rampant speculation surrounding new AI models, with people jumping to conclusions before any concrete information is available.
- **Ignore the Hype**: A member suggested that the best course of action is to **ignore the hype** surrounding new AI models and **wait for testable releases**.
   - They emphasized that **testing on the platform** is the only way to truly gauge a model's performance.



**Link mentioned**: <a href="https://x.com/lxuechen/status/1823602158518067539">Tweet from Xuechen Li (@lxuechen)</a>: Have been post-training Grok2 for a while and am excited to share that it’s officially out!!  We’ve been testing early versions of Grok2 on LMSYS chatbot arena under the names of sus-column-r and colu...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1273009781770293361)** (23 messages🔥): 

> - `Reranking Overview Document`
> - `Rerank API`
> - `Code Sample` 


- **User Tries to Utilize Rerank Document**: A user reported they were attempting to utilize the Rerank Overview document from the Cohere docs and had their API key verified. They confirmed they have installed Cohere v 5.8.0 and attempted both v2 and v3.
   - The user requested help in troubleshooting a situation where they were attempting to follow the Rerank Overview document.
- **Rerank API Code Example Provided**: A helpful user shared a code sample to demonstrate how to use the Cohere Rerank API.
   - The code example included a set of sample documents, a query, and instructions on how to use the `rerank` function.
- **Troubleshooting 'Unknown Field: Parameter Model'**: Another user reported encountering the error "unknown field: parameter model is not a valid field" while attempting to use the `rerank` function.
   - They had restarted their kernel and were seeking assistance with troubleshooting the error, as they were unable to resolve it by suppressing warnings and redirecting standard output.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/llmu">LLM University (LLMU)</a>: Welcome to LLM University, your premier learning destination for mastering Enterprise AI technologies. Designed for developers and technical professionals, our hub offers comprehensive resources, expe...</li><li><a href="https://docs.cohere.com/reference/rerank">Rerank - Cohere API References</a>: This endpoint takes in a query and a list of texts and produces an ordered array with each text assigned a relevance score.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1273262425331990608)** (10 messages🔥): 

> - `Cohere Toolkit Installation`
> - `Custom Deployment Issue`
> - `OpenAI Integration`
> - `Enterprise Search Chatbot`
> - `Fellowship.ai Cohort` 


- **Cohere Toolkit Installation Trouble**: A user reported difficulty adding a custom deployment for OpenAI to their locally installed Cohere Toolkit. They confirmed following the steps outlined in a document but the custom deployment isn't showing in the UI (localhost:4000) or in the Postgres container database 'deployment' table.
- **Building an Enterprise Search Chatbot**: The user explained they're building an "Enterprise search Chatbot" application to access company data stored in Confluence.
- **Fellowship.ai Cohort Research Project**: The user shared that they are part of the latest cohort at Fellowship.ai and are using this project for research and learning.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1273161713155440704)** (27 messages🔥): 

> - `LangChain support`
> - `LangSmith evaluation`
> - `LangGraph Cloud Access`
> - `LangChain Postgres Library`
> - `LLM Caching` 


- **Support for LangChain users**: A member expressed concern about the lack of timely support for basic questions in LangChain forums, especially those evaluating LangGraph and LangSmith, and how it affects their ability to promote the platform to their employers.
   - They also mentioned that a lot of general support questions are being posted on the LangChain Discord server, while other requests in their related support forums go unanswered.
- **LangSmith Plus: Access to LangGraph Cloud?**: A member asked if LangSmith Plus users will have access to use LangGraph Cloud.
   - No answer was provided.
- **LangChain Postgres library and caching**: A member asked about using the `langchain_postgres` library with `set_llm_cache`, which is a method for caching LLM results.
   - They were informed that while there is no `langchain_postgres` library, they can use the `SQLAlchemyCache` class from the `langchain_community.cache` module to cache LLM results in a PostgreSQL database.
- **Error loading sitemap: asyncio.run() cannot be called from a running event loop**: A member reported an error message: "Error loading sitemap https://kodefast.com/: asyncio.run() cannot be called from a running event loop", which occurs when trying to use `asyncio.run()` inside an already running event loop.
   - The bot suggested using the `nest_asyncio` library to allow nesting of event loops, or to refactor the code to ensure that `asyncio.run()` is not called from a running event loop.
- **Multi-LLM GUI Recommendations**: A member asked for a recommendation for a multi-LLM GUI, but they faced an error while using `create_csv_agent` from `langchain_experimental.agents.agent_toolkits`.
   - No answer was provided.



**Link mentioned**: <a href="https://python.langchain.com/v0.2/docs/integrations/llm_caching/#sqlalchemy-cache>).">Model caches | 🦜️🔗 LangChain</a>: This notebook covers how to cache results of individual LLM calls using different caches.

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1273300099912699985)** (3 messages): 

> - `Rubik's AI`
> - `AI security`
> - `RedOps platform`
> - `Chatbot security`
> - `Voicebot security` 


- **Rubik's AI: 2 months free premium**: The company Rubik's AI offers 2 months of free premium access to their platform, which features models like **GPT-4o, Claude-3 Opus, Mistral Large,** and more.
   - Users can access the offer using the promo code **RUBIX** at [signup.php](signup.php)
- **AI security is a key challenge**: A team built a platform called **RedOps** to test the security of chatbots and voicebots by intentionally trying to break them.
   - They realized how easily AI models could be manipulated through **adversarial inputs and social engineering**, highlighting the need for robust security measures.
- **RedOps platform simulates real-world attacks**: The **RedOps** platform simulates real-world attacks on chatbots and voicebots, including **contextual manipulation, adversarial attacks, ethical compliance, polymorphic testing**, and **social engineering**.
   - The platform aims to identify vulnerabilities and provide actionable recommendations for improving chatbot and voicebot security.
- **Key Findings: Context Matters**: The research found that bots must detect shifts into sensitive contexts and refuse to share sensitive info without validation.
   - The team also emphasized the importance of **regular audits and prompt engineering** to guide bots toward **neutral, ethical responses**.
- **Free security test for your chatbot or voicebot**: The team offers a free security test of your chatbot or voicebot, providing a detailed report of their findings and actionable recommendations.
   - To request a free test, send a link to your chatbot or voicebot to **redops@primemindai.com**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lnkd.in/gNusSAp9">LinkedIn</a>: This link will take you to a page that’s not on LinkedIn</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1273300241319333888)** (1 messages): 

> - `LangGraph`
> - `AI Agents`
> - `Email Management`
> - `Meeting Scheduling` 


- **LangGraph: AI Agent for Email Management and Scheduling**: A user created an AI Agent using LangGraph that automatically checks emails, chats with the sender, and books meetings.
   - The user shared a link to a demo of the agent: [Schedule](https://dub.composio.dev/Schedule/x).
- **AI Agents: The Future of Productivity?**: This example highlights the potential of AI Agents to streamline tasks and improve productivity.
   - This success story might inspire other developers to explore building similar agents to automate everyday tasks.


  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1272993390937768079)** (19 messages🔥): 

> - `Torchtune Compile Model+Loss`
> - `Torchtune CPU Offload Optimizer`
> - `Torchtune Model Size & Configuration` 


- **Torchtune Compile Model+Loss Needs Small Model**: A member is trying to use the `lora_dpo_single_device` recipe with `NF4+compile`, but is facing errors and suggests prioritizing other recipes first.
   - They also want to try the `ppo_full_finetune_single_device` recipe, but needs a small model that can fit on their 16GB GPU, and suggests using a smaller model like Qwen2-1.5B for this recipe.
- **Torchtune CPU Offload Optimizer: Torchao Dependency**: A member asks how Torchtune handles Torchao version dependency, as the CPU offload optimizer is in Torchao main and will be included in the next release.
   - They also discuss the possibility of having a copy of the CPU offload code in Torchtune, and using Torchao implementation when available.
- **TinyLlama 1B for Torchtune PPO Full Finetune**: A member suggests using TinyLlama 1B (or 0.5B) for the PPO full finetune recipe, since Llama2 classifiers are available.
   - They provide a link to a 1B configuration on GitHub and suggest adjusting batch sizes for memory optimization.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1272995857670733860)** (14 messages🔥): 

> - `Grok 2`
> - `Grok 2 mini`
> - `LMSYS`
> - `Claude`
> - `GPT-4` 


- **Elon Musk's Grok 2 Announced**: Grok 2, a new language model from Elon Musk's x.ai, has been released in an early preview, featuring "frontier capabilities" in chat, coding, and reasoning.
   - The model, dubbed "sus-column-r" on the LMSYS leaderboard, is currently outperforming both Claude 3.5 Sonnet and GPT-4-Turbo in terms of Elo score.
- **Fineweb-Edu Data Set Fortified**: The Fineweb-Edu dataset, available on Hugging Face, has been fortified by removing duplicate data and adding embeddings.
   - The dataset, dubbed "Fineweb-Edu-Fortified," now includes a `count` column indicating the number of times the text appears in the dataset.
- **Mistral Large 2 Training**: A user asked if Mistral Large 2 has been trained yet.
   - The response indicates that the model has not been trained yet.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-2">Grok-2 Beta Release</a>: no description found</li><li><a href="https://huggingface.co/datasets/airtrain-ai/fineweb-edu-fortified">airtrain-ai/fineweb-edu-fortified · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1273149867568660545)** (1 messages): 

> - `axolotl model loading conditions`
> - `axolotl model loading` 


- **axolotl model loading conditions**: A user asked why the condition `and False` is used in the `axolotl/utils/models.py` file when loading models.
   - This condition is used to ensure that the model is not loaded if the `load_model` flag is set to `False`.
- **Axolotl model loading with the `load_model` flag**: In the `axolotl/utils/models.py` file, the `load_model` flag controls whether or not a model is loaded.
   - The condition `and False` is used to prevent the model from being loaded if the `load_model` flag is set to `False`.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1273312315433029665)** (3 messages): 

> - `OpenAI Chat Endpoint Limitations`
> - `Assistant Response Continuation` 


- **OpenAI Chat Endpoint - No Continuation Possible**: A user asked if it is possible to continue a partially completed assistant response using the official OpenAI chat endpoint.
- **OpenAI Prevents Continuation**: A user explained that while they have had success continuing local model responses, they found that OpenAI's chat endpoint consistently prevents the continuation of assistant responses.


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1272996226085818524)** (13 messages🔥): 

> - `Open-source image annotation GUIs`
> - `Elon Musk and weight licenses`
> - `Schnelle` 


- **Open-source Image Annotation GUIs**: A member is seeking recommendations for good open-source GUIs for annotating images quickly and efficiently.
   - They are particularly interested in GUIs that support single-point annotations, straight-line annotations, and drawing polygonal segmentation masks.
- **Elon Musk's potential use of development licenses**: There was a discussion about Elon Musk possibly using developer licenses and challenging weight licenses.
   - The conversation revolved around the idea of Elon Musk utilizing a developer license to potentially circumvent the limitations of weight licenses.
- **Schnelle's paid features**: A member mentioned that Schnelle, a software tool, may require a paid subscription for its professional features.
   - They also indicated that Schnelle's pricing structure might not be ideal for users who are price-sensitive.


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1273221503734779956)** (4 messages): 

> - `Grok-2 release`
> - `Grok-2 mini`
> - `Grok-2 performance`
> - `Grok-2 API`
> - `Grok-2 multimodality` 


- **Grok-2 Released: New Era for Chat, Code, Reasoning**: x.ai has released an early preview of **Grok-2**, which is significantly more advanced than **Grok-1.5**, boasting frontier capabilities in chat, coding, and reasoning.
   - Grok-2 is being released alongside **Grok-2 mini**, a smaller but still powerful model.
- **Grok-2 Outperforms Claude & GPT-4 on LMSYS Leaderboard**: An early version of **Grok-2**, nicknamed "sus-column-r", has been tested on the **LMSYS leaderboard**, where it currently outperforms both **Claude 3.5 Sonnet** and **GPT-4-Turbo**.
   - Grok-2's success is further validated through **x.ai's internal evaluation process**, using **AI Tutors** to engage with models.
- **Grok-2 Beta on 𝕏 & Enterprise API Coming Soon**: Both **Grok-2** and **Grok-2 mini** are currently in beta on **𝕏**, and will be made available through x.ai's enterprise API later this month.
- **Grok-2 Multimodal Understanding on the Horizon**: x.ai is soon releasing a preview of **multimodal understanding** as a core part of the Grok experience on **𝕏** and API.



**Link mentioned**: <a href="https://x.ai/blog/grok-2">Grok-2 Beta Release</a>: no description found

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1273295059286098002)** (10 messages🔥): 

> - `ConvTranspose2D`
> - `3D data`
> - `kernel_size` 


- **ConvTranspose2D works with 3D data**: ConvTranspose2D does work with 3D data. There was a misunderstanding about how to pass the `kernel_size` argument.
- **Using a tuple for kernel_size**: The issue was that `kernel_size` was being passed as an integer instead of a tuple of length 3. Passing it as a tuple (e.g., `kernel_size=(3, 3, 3)`) resolved the error.
- **Improving documentation**: There is a suggestion to improve the documentation of ConvTranspose2D to clarify the correct way to use the `kernel_size` argument for 3D data.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1273054414457802774)** (5 messages): 

> - `Tinygrad Error: wait_result: 10000 ms TIMEOUT!`
> - `Lazycache Issues`
> - `CLANG=1 issue` 


- **Tinygrad Errors with CLANG=1**: A user encountered a `RuntimeError: wait_result: 10000 ms TIMEOUT!` while running Tinygrad with `CLANG=1`, but it worked correctly with `CUDA=1`.
   - They provided a code snippet showing the issue with a  `Tensor.zeros` operation on the GPU (3070ti) and the error message suggesting a potential issue with LAZYCACHE. 
- **Lazycache Bug Proneness**: Another user commented that the LAZYCACHE in Tinygrad is "bug prone" and suggested deleting it and deduplicating in the schedule.
- **Potential Issue with LAZYCACHE**: There is a possible connection between the error message and the LAZYCACHE, a feature in Tinygrad.
   - This may be due to a bug in the LAZYCACHE or an incompatibility with CLANG=1.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1273248998173577216)** (4 messages): 

> - `OpenInterpreter release`
> - `Local LLMs`
> - `RealtimeSTT`
> - `Faster-Whisper` 


- **OpenInterpreter Release Date**: A new version of OpenInterpreter was pushed to pip last night.
   - The next major update, "the developer update", is still in development and includes lots of new useful features.
- **Local LLMs Performance**: Local LLMs require a significant amount of processing power.
   - It is recommended to run LLMs in the cloud, especially for OpenInterpreter, which utilizes the default settings.
- **RealtimeSTT and Faster-Whisper**: OpenInterpreter now uses [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) which relies on [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for real-time speech to text.
   - The combination provides real-time performance for most users and has yet to be problematic on less powerful devices.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1273032411852046396)** (3 messages): 

> - `Hardware Channel` 


- **Hardware Channel Builder Role**: To view the category, users should assign themselves the builder role.
   - A user asked for more details on the channel.
- **User idea request**: A user was invited to share their idea for the hardware channel.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1272997098291462164)** (6 messages): 

> - `Open Interpreter`
> - `Tool Use Tuesday`
> - `Obsidian Plugin`
> - `Video Production`
> - `Manim` 


- **Open Interpreter in Obsidian & Anything-to-Anything**: A YouTube video was shared showcasing the use of Open Interpreter in Obsidian for converting anything to anything.
   - The video promotes the use of the Open Interpreter Obsidian plugin for controlling Obsidian vaults and showcases its capabilities in converting various data types.
- **Tool Use Tuesdays**: A user mentioned plans for a video presentation for a contest that involved using Open Interpreter and Obsidian.
   - They also mentioned exploring vector search and using Manim to visualize digraphs, indicating a focus on improving video production skills and utilizing the 'Tool Use Tuesdays' theme.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=HjcPRoPfri0">Open Interpreter Obsidian &amp; Convert Anything - Ep 0 - Tool Use</a>: Episode 0 of Tool Use!Open Interpreter Obsidian Plugin - Use Open Interpreter to control your Obsidian vault!CV - Convert anything to anything using the powe...</li><li><a href="https://www.youtube.com/watch?v=xaroJxFTVFQ">Is the AI Left-Bias Real?</a>: Take courses on large language models on Brilliant! First 30 days are free and 20% off the annual premium subscription when you use our link ➜  https://brill...
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1273021815517741076)** (8 messages🔥): 

> - `Poe Hackathon`
> - `Modal labs`
> - `LLM fine-tuning` 


- **Poe is having a Previews hackathon**: Poe is hosting a Previews hackathon in partnership with @agihouse_org, where participants can compete to create the most innovative and useful in-chat generative UI experiences.
   - The hackathon is open to all creators and more information can be found at [https://app.agihouse.org/events/poe-previews-hackathon-20240817](https://app.agihouse.org/events/poe-previews-hackathon-20240817)
- **Hackathon Invites**: A member inquired about the hackathon invite status after submitting their request.
   - They guessed the invites would be sent out by Thursday and mentioned spending around $300 of credits from the $1000 provided in a fine-tuning course.
- **Modal is the best platform for fine-tuning open-source LLMs**: A member shared their opinion that [Modal Labs](https://github.com/modal-labs/llm-finetuning) is the best platform to fine-tune open-source LLMs.
   - This sentiment suggests that Modal offers valuable tools and resources for developers working with large language models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/poe_platform/status/1823382125523181683">Tweet from Poe (@poe_platform)</a>: To celebrate the expanded release, we’re partnering with @agihouse_org for a Previews hackathon where you’ll compete to create the most innovative and useful in-chat generative UI experiences. All cre...</li><li><a href="https://app.agihouse.org/events/poe-previews-">AGI House</a>: no description found</li><li><a href="https://app.agihouse.org/events/poe-previews-hackathon-20240817">AGI House</a>: no description found
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1272995794504388670)** (3 messages): 

> - `Image Feature Extraction`
> - `Preprocessing Time Reduction` 


- **Simple Feature Store Speeds Up Training**: A user built a simple feature store for their R&D team, enabling them to store extracted features from images during online preprocessing.
   - This has significantly reduced training time, saving between 30-60 minutes per training run.
- **Generic Feature Store for Diverse Models**: The feature store is generic, accommodating image IDs, extraction methods, and pointers to extracted features in an object store.
   - It successfully handles diverse models, from extremely small to absolutely massive ones, allowing for efficient feature storage and retrieval.
- **Feature Extraction from Images**: A user inquired about the types of features being extracted from images.
   - The user providing the information indicated they cannot disclose the specifics due to non-disclosure agreements.


  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1273203285217775679)** (2 messages): 

> - `` 


- **Mistral struggles expanding beyond 8k**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1273007276168319067)** (1 messages): 

> - `Agentic AI Pipelines`
> - `Jupyter Notebook Automation`
> - `Devin-like System` 


- **Building an Agentic Jupyter Notebook Automation System**: A member asked about existing libraries, cookbooks, or open-source projects that could help build an agentic system to automate Jupyter Notebooks, specifically for swapping cells and generating variations.
   - The goal is to create a pipeline that can validate outputs and iteratively improve until successful, similar to the Devin project but focused on a specific, small task.
- **The Power of Automated Jupyter Notebook Modification**: The proposed system would take a working Jupyter Notebook as input and modify it by swapping out cells, ultimately generating multiple versions.
   - This automated process would allow for efficient exploration of different notebook configurations and potentially lead to improved results.


  

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

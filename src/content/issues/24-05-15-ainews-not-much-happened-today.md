---
id: 9d073ccd-a2aa-4c56-95e2-6966c0386805
title: Not much happened today
date: '2024-05-15T21:20:08.374758Z'
original_slug: ainews-to-be-named-3669
description: >-
  Ilya Sutskever is stepping down as OpenAI's Chief Scientist after nearly a
  decade, with Jakub Pachocki named as his successor to lead the company’s AGI
  efforts. Meanwhile, Google unveiled several AI advancements at I/O, including
  Gemini 1.5 Pro with a 2 million token context window, the Project Astra AI
  assistant prototype, and the new Imagen 3 text-to-image model.
tags:
  - openai
  - ilya-sutskever
  - jakub-pachocki
  - google
  - gemini-1.5-pro
  - project-astra
  - imagen-3
---


<!-- buttondown-editor-mode: plaintext -->*The GPT4o and Gemini aftermath.*

> AI News for 5/14/2024-5/15/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**427** channels, and **6455** messages) for you. 
Estimated reading time saved (at 200wpm): **686 minutes**.

```
'Twas the night after I/O, when all through AI
Not a startup was posting, not even on LI
The UBI research was studied by e/accs with care
In hopes that AGI soon would be there
```

You can wish [Ilya](https://twitter.com/ilyasut/status/1790517455628198322) and [Jan](https://news.ycombinator.com/item?id=40363273) and [Evan](https://x.com/E0M/status/1790814866695143696) well (is there something to the [departure timeline](https://twitter.com/liron/status/1790773952811545051)?), read about GPT4o's [incredible multi-Needlestack performance](https://news.ycombinator.com/item?id=40348947), or [watch John Schulman](https://www.youtube.com/watch?v=Wo95ob_s_NI) or [Sama](https://www.youtube.com/watch?v=fMtbrKhXMWc)'s latest interviews, if you're team OpenAI, or you can congratulate [Mike Krieger on joining Anthropic](https://twitter.com/i/trending/1790766332885299320), or you can read [all the Google I/O roundups](https://twitter.com/i/trending/1790833517636764082) that came after us (it seems we underrated [PaliGemma](https://news.ycombinator.com/item?id=40371237) initially).

---

**Table of Contents**

[TOC] 



---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**Ilya Sutskever Leaving OpenAI**

- **Ilya Sutskever stepping down as Chief Scientist after nearly a decade**: [@sama](https://twitter.com/sama/status/1790518031640347056) praised Ilya as "one of the greatest minds of our generation" who was foundational to OpenAI's success. [@ilyasut](https://twitter.com/ilyasut/status/1790517455628198322) expressed it was an "honor and privilege" to work together and that he will miss everyone as he pursues a personally meaningful project.
- **Jakub Pachocki named new Chief Scientist**: [@sama](https://twitter.com/sama/status/1790518031640347056) expressed confidence that Jakub, another "one of the greatest minds of our generation", will lead OpenAI to make rapid and safe progress towards AGI in his new role. 
- **Ilya's pivotal early role shaping OpenAI's mission and strategy**: [@gdb](https://twitter.com/gdb/status/1790519014562898012) reflected on countless hours he and Ilya spent in the early non-profit days aligning on culture, technical direction and strategy to build OpenAI, even when others doubted AGI was achievable in the near-term.

**Google I/O AI Announcements**

- **Gemini 1.5 Pro and Flash language models**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790432978126139411) unveiled Gemini 1.5 Pro with 2 million token context and improved code, reasoning and multimodal capabilities, along with Gemini 1.5 Flash optimized for low latency and cost. Both are now available in Google AI Studio and Vertex AI.
- **Project Astra AI assistant prototype demoed**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790433540548558853) shared videos of Project Astra, a future AI assistant that can interact with the world, remember context, and assist in everyday life. Many compared its capabilities to GPT-4.
- **Imagen 3 text-to-image model released**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790434750592643331) introduced Imagen 3, their most advanced text-to-image model yet with enhanced detail and realism. 
- **Veo generative video model previewed**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790435824598716704) offered a glimpse of Veo, capable of generating 1080p 60+ second video clips across diverse styles. It's now available via a Labs waitlist.
- **Music AI Sandbox tools for creators**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790435413682975043) developed a collection of AI tools in collaboration with musicians to transform the music creation process, showcased through new demo recordings.


**AI Model Releases and Benchmarks**

- **GPT-4o Tops Leaderboards**: [@percyliang](https://twitter.com/percyliang/status/1790622792347701432) noted **GPT-4o tops the VHELM leaderboard**. [@maximelabonne](https://twitter.com/maximelabonne/status/1790519226677026831) shared that GPT-4o **significantly outperforms competitors on the LMSYS Chatbot Arena** based on data from [@LiamFedus](https://twitter.com/LiamFedus).
- **Reka Core and Qwen Models**: [@maximelabonne](https://twitter.com/maximelabonne/status/1790519226677026831) mentioned @RekaAILabs released a solid multimodal model **Reka Core with 128K context**, and @Alibaba_Qwen released the open-source **Qwen1.5-110B** and closed-source **Qwen Max**.
- **Salesforce Online RLHF Recipe**: [@_philschmid](https://twitter.com/_philschmid/status/1790747448807215428) shared Salesforce's **reproducible recipe for online iterative RLHF**, showing online methods like iterative DPO outperform offline methods. The code, models, dataset and training details are open source.

**Multimodal AI and Video Models**

- **Imagen 3 and Veo**: Google introduced **Imagen 3**, their highest quality text-to-image model yet with incredible detail and realistic lighting, per [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790434750592643331). They also revealed **Veo**, a powerful video model that can create 1080p 60s+ clips in various styles, with a waitlist to try features in VideoFX Labs, according to [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790435824598716704).
- **Music AI Sandbox**: In collaboration with @YouTube, [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1790435413682975043) has been building **Music AI Sandbox**, a suite of AI tools to transform music creation, working closely with musicians and producers. New demo recordings are available on YouTube.

**Memes and Humor**

- **Scarlett Johansson AI**: [@karpathy](https://twitter.com/karpathy/status/1790373216537502106) joked that the killer app of LLMs is Scarlett Johansson, not math or something. People thought it was math but it's ScarJo.
- **Gemini Flash Naming**: [@agihippo](https://twitter.com/agihippo/status/1790435129577599188) noted they're still contributing to Google names after being an ex-Googler, in reference to the Gemini Flash name.
- **Gemini Watching I/O**: [@zacharynado](https://twitter.com/zacharynado/status/1790474150345081123) joked that Gemini watched Google I/O, a reference to the Project Astra demo of the AI watching the keynote.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**OpenAI Leadership Changes and Internal Dynamics**

- **Jan Leike, co-head of OpenAI's Superalignment team, resigned and tweeted concerns about "whatever is going on behind the scenes"**: In /r/singularity, a [screenshot of Jan Leike's tweet](https://i.redd.it/ztfqaypt0j0d1.png) announcing his resignation from OpenAI and expressing unease with internal dynamics was shared. This hints at potential disagreements within OpenAI about the company's direction and approach to AI development.

- **Ilya Sutskever, OpenAI co-founder and Chief Scientist, announced he is leaving the company after almost a decade**: OpenAI CEO Sam Altman [tweeted](https://twitter.com/sama/status/1790518031640347056?t=0fsBJjGOiJzFcDK1_oqdPQ&s=19) that Sutskever is departing, fueling speculation in /r/singularity about misalignment between leadership regarding OpenAI's trajectory and AI safety priorities.

- **Former OpenAI employee Logan Kilpatrick hinted at further internal drama in response to Leike's resignation**: Kilpatrick [replied](https://twitter.com/OfficialLoganK/status/1790604996641472987) to Leike's tweet saying "Keep fighting the good fight 🫡", suggesting more behind-the-scenes tensions at OpenAI that may come to light.

**GPT-4o Capabilities and Limitations**

- **GPT-4o introduced as OpenAI's new flagship model with efficiency improvements over GPT-4**: In /r/singularity, an [OpenAI email announcement](https://www.reddit.com/r/singularity/comments/1crv0ri/new_openai_email/) was shared detailing GPT-4o's capabilities - claimed to match GPT-4 performance with 50% lower pricing, 2x faster latency, and 5x higher rate limits.

- **Some users found GPT-4o still fails basic reasoning tests and underperforms on programming compared to alternatives**: Posts in /r/OpenAI demonstrate GPT-4o [struggling with elementary logic puzzles](https://www.reddit.com/r/OpenAI/comments/1cs11b1/chat_gpt4o_still_fails_my_very_basic_intelligence/) and exhibiting [disappointing code generation abilities](https://www.reddit.com/r/OpenAI/comments/1cs210q/gpt4o_disappointing_performance_for_programming/) versus competitors like Anthropic's Claude model.

- **GPT-4o's image generation capabilities were "bizarrely under-presented" in the announcement**: A /r/singularity [thread](https://www.reddit.com/r/singularity/comments/1crto0m/gpt4o_was_bizarrely_underpresented/) argues GPT-4o's visual skills, including generating standalone objects and images for 3D reconstruction, deserved more emphasis and demonstration.

**Google I/O AI Announcements**

- **Google announced several new AI initiatives at I/O, but reception was mixed compared to GPT-4o**: Some /r/singularity commenters felt Google's AI presentations and demos were [underwhelming](https://www.reddit.com/r/singularity/comments/1cs22sj/the_contrast_in_openai_versus_googles_approach/) next to the GPT-4o reveal.

- **New Google AI products include Gemini 1.5 Flash, Imagen 3, and Project Astra**: [Gemini 1.5 Flash](https://www.reddit.com/r/singularity/comments/1cs2v7h/gemini_15_flash_is_very_price_effective_relative/) is an efficient language model, [Imagen 3](https://deepmind.google/technologies/imagen-3/) improves image generation, and [Project Astra](https://www.youtube.com/watch?v=nXVvvRhiGjI) focuses on AI assistants.

**Open Source Alternatives and Concerns** 

- **Some advocate for open source AI as an important alternative to closed models from OpenAI and Google**: An [opinion piece](https://open.substack.com/pub/molbal94/p/opinion-the-case-for-open-source?utm_source=share&utm_medium=android&r=1wdxxq) argues open source AI can better prioritize user privacy, customization, and accessibility as the big tech AI race accelerates.

- **Meta's Llama-3 model shows promise but its restrictive license raises derivative work concerns**: A /r/LocalLLaMA [post](https://www.reddit.com/r/LocalLLaMA/comments/1csctvt/we_need_to_have_a_serious_conversation_about_the/) calls for discussion on the legal implications of Llama-3's license for derivative models like Salesforce's recent release.

**Implications and Societal Impact**

- **Increasingly capable AI like GPT-4o is predicted to disrupt education, creative fields, and coding jobs**: /r/singularity threads speculate about major changes to [schooling](https://www.reddit.com/r/singularity/comments/1crqogx/with_the_recent_gpt4o_release_how_will_the_future/), [entertainment](https://www.reddit.com/r/singularity/comments/1cs8r9q/im_excited_for_ai_generated_movies_made_by_great/), and software engineering as AI rapidly advances.

- **Poll indicates majority of Americans support regulations to prevent development of superintelligent AI**: A survey shared in /r/singularity found 63% in favor of measures to restrict the creation of AI systems surpassing human intelligence levels.

**Memes and Humor**

- **Memes and jokes react to the rapid pace of AI progress**: Posts in /r/OpenAI humorously [predict people attempting to marry ChatGPT](https://www.reddit.com/r/OpenAI/comments/1crxc89/i_bet_5_that_someone_will_really_try_marry_gpt/) and [AI enabling "tortured artists"](https://www.reddit.com/gallery/1cs2cwa) to cope with the unsettling speed of AI advancements.

---

# AI Discord Recap

> A summary of Summaries of Summaries. We are concluding that Claude still remains the best summarizer model so we are dropping the GPT4T and 4o comparisons.

1. **Unveiling of New AI Models and Capabilities**:
   - Google introduced several new AI models at Google I/O, including [**Veo**](https://deepmind.google/technologies/veo/) for high-quality video generation, [**Imagen 3**](https://deepmind.google/technologies/imagen-3/) for improved text-to-image capabilities, and [**Gemma 2**](https://techcrunch.com/2024/05/14/google-announces-gemma-2-a-27b-parameter-version-of-its-open-model-launching-in-june/), a 27B parameter model. [Source](https://discord.com/channels/1053877538025386074/1149866623109439599/1239835061893857371)
   - OpenAI's [**GPT-4o**](https://arstechnica.com/information-technology/2024/05/before-launching-gpt-4o-broke-records-on-chatbot-leaderboard-under-a-secret-name/) was revealed as the top model on LMSYS's Chatbot Arena leaderboard under a secret name before its launch. [Source](https://discord.com/channels/879548962464493619/879548962464493622/1239846551526965259)
   - Nous Research released [**Hermes 2 Θ**](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B), an experimental model merging Hermes 2 Pro and Llama-3 Instruct, outperforming previous models on benchmarks while retaining function calling capabilities. [Source](https://discord.com/channels/1053877538025386074/1145143867818119272/1240351762863493150)

2. **Advances in Multimodal AI and Unified Models**:
   - Discussions centered around the challenges and potential of **multimodal models**, with members exploring unified models like [**ImageBind**](https://ai.meta.com/blog/imagebind-six-modalities-binding-ai/) that bind information across multiple modalities using joint embeddings. [Source](https://discord.com/channels/1053877538025386074/1154120232051408927/1239836871887159306)
   - Google's [**Gemini 1.5 Flash**](https://openrouter.ai/models/google/gemini-flash-1.5) and [**Gemini 1.5 Pro**](https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/) were introduced, offering multimodal capabilities for visual understanding, classification, summarization, and content creation from various inputs. [Source](https://discord.com/channels/1091220969173028894/1092729520181739581/1239890486387281920)
   - Members discussed the potential of integrating **multimodal models directly into smartphones and edge devices** for low latency and enhanced multimodal functionalities. [Source](https://discord.com/channels/1053877538025386074/1149866623109439599/1239835061893857371)

3. **Optimization and Efficiency Efforts for LLMs**:
   - Techniques like [**Gemini's context caching**](https://ai.google.dev/gemini-api/docs/caching) and [**llama.cpp's prompt caching**](https://github.com/ggerganov/llama.cpp/blob/e1b40ac3b94824d761b5e26ea1bc5692706029d9/examples/main/main.cpp#L225-L245) were discussed as ways to make LLM workflows more efficient and cost-effective by reducing token usage for long prompts. [Source](https://discord.com/channels/823971286308356157/1097032579812687943/1240332490154053725)
   - Members explored strategies to improve the **L2 cache hit rate** for better performance, referencing resources like the [**Triton Matrix Multiplication tutorial**](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) on block-level multiplication and pointer arithmetic. [Source](https://discord.com/channels/1189498204333543425/1189607726595194971/1239868744906575992)
   - Discussions revolved around optimizing **tensor allocations and caching** when using `torch.compile`, with recommendations to replace dynamic allocations with pre-allocated tensors and leverage static caching to reduce overhead. [Source](https://discord.com/channels/1189498204333543425/1189607750876008468/1239840938080342057)

4. **Debates on LLM Evaluation and Industry Dynamics**:
   - A [blog post](https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation) highlighted the closed nature of current LLM evaluation practices, dominated by academic benchmarks and private A/B testing, calling for broader accessibility in evaluations. [Source](https://discord.com/channels/1179127597926469703/1183121795247779910/1239839347138756748)
   - Members discussed Anthropic's transition towards becoming a product company, OpenAI's potential foray into search with a key hire, and the strategic need for AI companies to offer end-user products rather than just APIs or services. [Source](https://discord.com/channels/1179127597926469703/1183121795247779910/1239839347138756748)
   - The departure of Ilya Sutskever from OpenAI sparked discussions about potential reshuffles within the company, with [Sam Altman](https://x.com/ilyasut/status/1790517455628198322) and others commenting on the transition. [Source](https://discord.com/channels/822583790773862470/1075282825051385876/1239868524688576552)

---

# PART 1: High level Discord summaries


## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**GPT-4o Faces Creative Block**: **GPT-4o**'s faster response time compared to **GPT-4** comes with a trade-off in creative writing tasks, where it often regurgitates rough drafts instead of enhancing them with intelligent revising. This has caused some discontent among users trying to leverage the AI for writing enhancements.

**Models Go Musical with Auditory Descriptions**: Community members have creatively used prompts asking **GPT-4** and **GPT-4o** to describe instrumental songs, like *"The XX Intro"* and *"Tears in Rain"*, to gauge the models' descriptive abilities when it comes to auditory perceptions. Results from these prompts may provide insights into each model's interpretive skills.

**An Imaginative Challenge in Imagery Generation**: A user encountered difficulties with both **GPT-4** and **GPT-4o** when requesting specific side views for platform game development – the AI tended to supply unwanted isometric perspectives and extraneous details, indicating a gap in perspective understanding and context adherence.

**Access and Features on a Staggered Path**: The rollout of **GPT-4o** features has been sequential, with functionality like Voice Mode and multimodal capabilities first reaching API partners, then Plus users. This gradual deployment has led to confusion and access issues for some members.

**Custom Models Spark Collaboration**: The discussion regarding the integration of custom GPTs with **GPT-4o** and feedback sessions on custom models like **OptimusGPT** highlight the community's eagerness for improvement and collaboration. Users have been advised to migrate their custom GPTs to GPT-4o within the forthcoming weeks for enhanced performance. 

**Simplicity in AI Voice Assistance**: A member highlighted the **Plug & Play AI Voice Assistant**, which can be set up quickly and invited the community to try it out. Despite the repeated announcements, the focus was on the assistant's ease of setup and the value of user feedback on its efficacy.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**Machine Learns Mythos**: An AI geared towards **Cthulhu** worship was developed by a guild member using Unsloth to create and fine-tune data, resulting in the creation of TinyLlama and Mistral 7B Cthulhu models with resources available on [Huggingface](https://rasmusrasmussen.com/2024/05/14/artificial-intelligence-in-the-name-of-cthulhu/).

**Sailing the Quantization Seas**: Conversations tackled challenges in **quantization** and **model merging**, with members sharing tips like manually upcasting to 16-bit before merging and using notebooks to facilitate conversion processes, illustrating the complex terrain of optimizing AI models for better performance.

**Global Model Outreach**: Unsloth was recognized in an [AI News](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/#unsloth-ai-daniel-han-discord) feature for its strides in open-source AI development, and community members rallied in support of a proposal to showcase Unsloth at an upcoming New York City open-source data pipeline meetup.

**GPT-4 Lifeguard On Duty**: Assistance was offered to a guild member struggling with **Trigonometry** problems, evidencing the community's rapid response in providing resources like ChatGPT and Claude for academic aid.

**AI Summarization Scrutinized**: Potential conflicts with European data privacy laws were flagged concerning the use of AI to summarize Discord interactions, which signals the ongoing vigilance required to balance technological innovation with legal compliance.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Transcribing Woes and Speedy GPT-4o**: Engineers discussed Perplexity apps' ability to transcribe hour-long meetings and praised GPT-4o for outpacing Turbo in speed. However, concerns were raised about the current feature limitations in audio recording, particularly full transcription functionality not being rolled out.

- **Parsing Problems Persist**: Users highlighted issues with Perplexity parsing URL content inaccurately, implying it makes guesses instead of analyzing the actual web content. This indicates potential areas for enhancement in content parsing algorithms.

- **LLaMA-3 Diversifies Its Skill Set**: The *LLaMA-3-sonar-large-32k-chat* model is optimized for conversational nuances, while *LLaMA-3-8b-instruct* is designed for a more comprehensive instructional scope. Additionally, there's interest in the web search capabilities of the *LLaMA-3-sonar-large-32k-online* model, akin to RAG models on platforms like Perplexity.com.

- **API Access and Latency Issues Come to Light**: Requests for beta access to the citations API and observations of increased latency in Perplexity's API calls reflect the active development and utilization of API features among the community. There is active consideration of API timeout efficiencies, particularly for longer 3000-word inputs, which currently face timeouts at a 10000ms setting.

- **A Trove of Perplexity Search Discoveries**: Members shared a variety of [Perplexity.ai](https://www.perplexity.ai) searches ranging from detailed analyses of market sizes, profound insights into mindfulness practices, to comprehensive resources on the finetuning of models. Such searches serve as a testament to the platform's rich informational ecosystem for AI exploration and model tuning.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Bold Breakthrough in LLM Performance**: The newly launched **Hermes 2 Θ**, outshining Hermes 2 Pro and Llama-3 Instruct, boasts superior performance in benchmarks while maintaining the ability to call functions, as announced in the **[announcements](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B)**.

**Discord Meets Innovation**: A tool exploiting a bug in Discord allows embedding **AV1 videos** larger than 500MB, which can also be shared on platforms like Twitter, as discussed in **[off-topic](https://autocompressor.net/av1?s=sznVX9AV)**.

**GPT-4's Mixed Reviews**: Despite GPT-4's prowess in data science tasks, **[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1239651072503709797)** channel discussions reveal its underperformance in complex tasks and a tendency to lose context, hinting at trade-offs between speed and accuracy.

**Nordic AI Language Model Unleashed**: **[interesting-links](https://www.silo.ai//blog/viking-7b-the-first-open-llm-for-the-nordic-languages)** showcases Viking 7B, a leading-edge multilingual LLM designed for Nordic languages by Silo AI and University of Turku's TurkuNLP, enhancing language AI accessibility.

**AI Skepticism and Enthusiasm Intertwined**: General sentiment across various channels such as **[general](https://github.com/NousResearch/Hermes-Function-Calling/tree/main)** and **[ask-about-llms](https://arxiv.org/abs/2305.05665)** remains mixed with enthusiasm over new models like Hermes 2 Θ, yet sceptical on multimodal capabilities and the barriers faced when building LLMs from scratch.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Getting Vocal with LLMs**: Talk of integrating voice interaction with local large language models (LLMs) highlighted the use of tools like *AnythingLLM*. The community discussed resource-intensive solutions involving Whisper.cpp and Coqui TTS, albeit with complexities and suboptimal experiences.

**Beefing Up the Hardware Arms**: Debates swung around hardware preferences for AI models, pitching a 3060Ti GPU against dual 16-core Xeon V4 CPUs. Enthusiasts mooted over VRAM's pivotal role, with a bias towards Nvidia cards for top-tier AI performance. The mention of a 4060 sparked interest for its prospective gains.

**PrivateGPT vs. AnythingLLM - A Document Query Duel**: The competition between **PrivateGPT** and **AnythingLLM** for querying documents with LLMs incited a technical analysis. Discussions underlined setup intricacies and user-friendly aspects of each platform.

**MacOS First Strikes A Sour Note**: A Mac-tier debate surfaced with grievances regarding app release priorities, primarily the MacOS-first strategy from OpenAI. This spun into a dialogue on the complexities and divergences in MacOS versus Windows app development.

**Battle of the Giants in the Model Arena**: From uncensored local LLM recommendations, notably [Dolphin 2.8 Mistral 7B v0.2](https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02), to the nuances of quantization and model performance, the community dissected various AI paradigms. Aforementioned also was Command R models' comparison and GPU-related enigmas.

**Hacking the Hardware Frontier**: ROGUE RX6600, not typically supported by AMD for the ROCM build, gamely runs in Koboldcpp, while official llama.cpp binaries restrict usage due to GPU ID verification processes. Users flagged user-interface (UI) complexities within LM Studio settings.

**Gleaning GPU Gems**: Tips on GPU resource optimization with Windows Task Manager sallied forth, with quirky recommendations like disabling hardware acceleration to enhance resource visibility. However, struggles continue with configuring CUDA on select laptops, leading to persisting model loading errors in LM Studio.

**Old Guard vs. New Recruits in GPU Tussle**: Tesla M40's disappointing showdown with GeForce 1060 on LLM tasks and the touted VRAM speed's importance got limelight. Financial constraints loomed over users, with low-end PCs finding refuge in modest local models and APUs revealing no performance perks over CPUs in llama.cpp.

**Beta Build Blues**: In beta territory, ruminations on multimodal feature parity shared space with reports of LM Studio's launch issues due to lacking AVX2 support. A user's exasperation with a non-launching LM Studio was quelled by identifying that the AGX instruction set was paramount for operation.

**The Developer's Digest**: Intel’s overture for Intel GPU support using SYCL for llama.cpp broadened the horizon for LM Studio. Conversations flourished around DL model adaptation, the quest for AGI, and community calls to keep dev chatter tethered to LM Studio's APIs and software construction.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**GPT-4o Stealthy Champion**: OpenAI's [GPT-4o](https://arstechnica.com/information-technology/2024/05/before-launching-gpt-4o-broke-records-on-chatbot-leaderboard-under-a-secret-name/) was confirmed to be the top model under a secret name in the LMSYS's Chatbot Arena, boasting undisclosed performance feats.

**Datasets and Models Leverage Enhanced**: A team released a [700,000-sample Vietnamese dataset](https://huggingface.co/datasets/Vi-VLM/Vista) for open-source language modeling, while AutoTrain extended its toolkit with Object Detection functionality, and [Diarizers](https://x.com/kamilakesbi/status/1790025563631132940) emerged as a new library for fine-tuning speaker diarization systems with multilingual support on Hugging Face's Hub.

**AI-Powered Story Crafters**: A reading group engaged in a comprehensive review of [AI story generation](https://arxiv.org/abs/2310.05388), with discussion pivoting towards refining the [GROVE framework paper](https://arxiv.org/abs/2310.05388) and community members sharing endeavors and learnings via [Medium](https://isamu-website.medium.com/understanding-ai-for-stories-d0c1cd7b7bdc).

**Visual Data to Revenue Insights**: Inquiry in the [#computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1239859086409863271) channel sparked a discussion about the feasibility of training a model that converts images to sales data outputs; the original poster provided a related dataset [link](https://huggingface.co/datasets/tonyassi/sales1) for reference.

**Enhancing Chatbots with LangChain**: In the [#NLP](https://discord.com/channels/879548962464493619/922424173916196955/1240103986791452758) channel, a member sought to improve chatbot conversations using LangChain, with suggestions directing to an initial [starter example](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/) for using local LLM and embedding models.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **LoRA Training Achieves Lift-Off**: An engineer has successfully completed their first training session with LoRA, taking just 90 minutes, and plans to share the result on [Civitai](https://civit.ai/).
- **Zooming in with Powerpaint**: Technical discussions among users centered on the use of inpainting and Powerpaint for enhancing fine details in images, specifically with version 1.5 capable of improving detailed features like eyes.
- **Workflow Wizards**: For those curious about outpainting techniques using ComfyUI, a helpful engineer linked to a [GitHub workflow guide](https://github.com/cubiq/ComfyUI_Workflows/blob/main/in-out_painting/README.md), aiding fellow users in mastering inpainting and outpainting.
- **Google's Imagen vs. The People's Choice**: A comparison of Google's Imagen 3 with Stable Diffusion reflected community preference for the latter, citing better accessibility and usability against the tech giant's offering.
- **GPU Gossip**: Engineers discussed GPU preferences for AI-related tasks, stressing VRAM's importance for long-term utility. A consensus suggested awaiting the 50xx series GPUs slated for November might yield better performance-to-price ratios.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Unleashes a Model Avalanche**: OpenRouter released new AI models like [DeepSeek-v2 Chat](https://openrouter.ai/models/deepseek/deepseek-chat) and [Llama 3 70B Base](https://openrouter.ai/models/meta-llama/llama-3-70b), with [Gemini Flash 1.5](https://openrouter.ai/models/google/gemini-flash-1.5) from Google and several [models by Perplexity](https://docs.perplexity.ai/changelog/new-models-llama-3-sonar-family) to expand its arsenal, emphasizing innovations despite requiring activation of logging to utilize DeepSeek models.

- **Performance Demands End WizardLM-2 8x22B Nitro's Run**: OpenRouter axed the WizardLM-2 8x22B Nitro variant due to a provider's inability to sustain expected throughput of 100 tokens per second, indicating a rigorous performance standard.
  
- **Curiosity in Crypto Confirmations Quenched**: Crypto balance delays were attributed to network confirmation requirements by platforms like Coinbase, mandating 128 block confirmations on Polygon and similar criteria on other networks.

- **API Tool for Model Mastery**: One user contributed an API-based tool for tracking OpenRouter model updates, providing an hourly-refreshed list accessible through a [GitHub repository](https://github.com/fry69/orw), signalling a community-driven approach to technology monitoring.
  
- **Mixed Reactions to Google’s Gemini Gathering**: Google's Gemini event stirred varied responses, introducing new models like Gemini 1.5 Flash, but didn't seem to impress some against the backdrop of OpenAI's more buzzed-about events, showcasing contrasting community expectations.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo Integrates MLIR**: Engineers discussed **Mojo's** ability to execute **MLIR** code with minor syntax adjustments, contributing to Mojo's versatility and access to lower-level features.

**Strategies for Mojo Mastery**: A variety of resources for learning **Mojo** were recommended, including the [Mojo SDK manual](https://docs.modular.com/mojo/manual/get-started/) and the [Mandelbrot notebook](https://docs.modular.com/mojo/notebooks/Mandelbrot), with the community highlighting the language's advantages like cross-vendor GPU code portability.

**Python Convenience Without Python**: The community is exploring alternatives to Python dependencies within the **Mojo** toolchain, indicating a drive for a more language-agnostic ecosystem. Follow the progress on the [feature request on GitHub](https://github.com/modularml/mojo/issues/935).

**C/C++ and Python Interop with Mojo Abuzz**: There's active discussion on calling C/C++ libraries using ffi and dealing with Python interoperability issues, reflecting a keen interest in **Mojo's interlanguage capabilities**. Engineers are sharing insights on the mechanics, evidenced by the shared [tweetorial](https://github.com/modularml/devrel-extras/tree/main/tweetorials/ffi) and problem-solving threads.

**Modular’s Multimedia Mojo Hustle**: Modular provided updates and tutorials via new videos on Mojo nightly builds and MAX Graph API, as well as via a MAX Graph API [blog tutorial](https://www.modular.com/blog/max-graph-api-tutorial). Additionally, two tweets teasing updates and a community meeting were noted, although details remained unspecified.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Mimetic Initialization Shows Promise**: Introducing **mimetic initialization** to Transformers yields significant accuracy improvements on datasets like CIFAR-10 and ImageNet, per a shared [paper](https://proceedings.mlr.press/v202/trockman23a/trockman23a.pdf). This technique mimics weight patterns from pre-trained models, signaling potential for more efficient training.

**Dataset Diversification with Sakuga-42M**: The new **Sakuga-42M dataset** was unveiled, containing 42 million keyframes of cartoon animations and aiming to reduce biases of models trained on natural images. An [arXiv link](https://arxiv.org/abs/2405.07425) to the dataset provides the gateway for further exploration.

**Hypernetworks Pique Interest for Initialization**: Discussions emerged around employing hypernetworks for weight initialization, suggesting the possibility of symbolic regression for crafting innovative initialization techniques.

**Leveraging Dot Products in Neural Networks**: A lively discussion endorsed the effectiveness of dot products in neural networks, with a member linking to an [article](https://archive.is/GfliU) that examines their connection with Fourier transforms and implications for cognitive processing.

**Enhancing Multiple Choice Analysis**: Debates flared around optimizing the processing of multiple-choice questions in models, highlighting the **lm-evaluation-harness**' approach to manage requests per answer and considering an output export feature for accuracy analysis, referencing [GitHub code](https://github.com/EleutherAI/lm-evaluation-harness/blob/a9eaaf46f1e246e5ce090e37f2f99fe1cfe5a919/lm_eval/models/utils.py#L485).



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA curiosities and integrations in LLVM**: Participants grappled with a synchronization anomaly in CUDA streams, suspecting it to affect gradient accumulation and GPU activities. *Stream misordering* was observed to potentially introduce race conditions. Suggestions pushed for a more explicit stream handling and rethinking gradient logic ([PR #417](https://github.com/karpathy/llm.c/pull/417)). Tolerance levels in gradient checking also saw a rigorous debate with advocates for practical thresholds relative to magnitude.

- **Triton Tutorial Entices Matrix Multipliers**: A *Triton tutorial* was highlighted for its insight into improving L2 cache hit rates through block-level matrix multiplication and optimized pointer arithmetic ([Link to tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)). It dovetailed with discussions illuminating problems in CUDA such as a *naive dot product implementation* error due to FP32 precision constraints ([Dot Product Puzzle Issue](https://gist.github.com/chetandhembre/6e93a652026f0bb669c981513e2cc5b8)).

- **PyTorch torch.compile quirks revealed**: Users uncovered torch.compile's facets when dealing with dynamic tensor allocation and its performance costs, comparing against static allocation. Advice went as far as suggesting `torch._dynamo` decorators for debugging during compilation. DeepSpeed's recent release raised questions about its compatibility with `torch.compile`, directing attention towards a [GitHub PR suggesting a compile flag](https://github.com/microsoft/DeepSpeed/pull/4878).

- **Lectures and Guides Light the Way**: A newcomer seeking guidance on CUDA kernels was steered toward a useful [YouTube lecture for Python programmers on CUDA](https://youtu.be/4sgKnKbR-WE), while on another front, a member surfaced the [NVIDIA GPU Programming Guide](https://download.nvidia.com/developer/GPU_Programming_Guide/GPU_Programming_Guide.pdf) for GPU programming evangelists.

- **Cloudy with a Chance of Tweets**: Amidst the technical discourse, a member whimsically recommended checking out cloud's Twitter without giving context ([link to tweet](https://twitter.com/cloud11665/status/1790776040681271583)), demonstrating the community's occasional drift towards light-hearted interactions.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Run Your Own LLM with Mozilla's llamefile**: Mozilla's new **llamefile** makes it easy for engineers to set up a local, private research assistant. A simple download and execution allows direct use of a local LLM and embedding model from LlamaIndex, enhancing data privacy and control. [Find out more here](https://t.co/qFIA6j1OWe).

- **Navarasa Recognition and LlamaIndex's New Partnerships**: **Navarasa**, a model supporting 15 Indic languages, has earned the spotlight at Google I/O. Additionally, LlamaIndex's collaboration with **Vertex AI** for a RAG API signals a movement towards simplifying complex AI system integrations. [Navarasa at Google I/O](https://t.co/zc00GjOmc4) | [LlamaIndex on Vertex AI](https://t.co/ekAQ24hNWr).

- **Chatbot Creation Made Easy with GPT-4o**: The introduction of create-llama empowers even the less technically inclined to build a chatbot using **GPT-4o** through a streamlined question-and-answer setup process. This is a big step towards democratizing AI-powered conversational agents. [Discover how](https://t.co/wtcaWdrB7H).

- **Various Technical Debates and Clarifications**: Members of the guild engaged in technical discussions around the efficiency of "small to big retrieval," the process of updating the **sec-insights repo**, discrepancies in model performance, specifically between **Meta-Llama** and quantized **Ollama**, and the integration of **GPT-4o** with **LlamaIndex**.

- **Security Protocols for LlamaParse Questioned**: Concerns regarding **LlamaParse**'s security led to clarifications on data retention policies, such as the 48-hour caching policy and an on-premise option for those prioritizing stringent data security measures.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Nathan Lambert Stirs AI Talks**: Nathan Lambert critiqued OpenAI's user-centric approach, expressing this view in a [tweet](https://twitter.com/natolambert/status/1790393805633712246) and addressed Google's generative video advances at Google I/O as impressive, but noted some announcements like Gemini 1.5 Ultra were overlooked.

**Google Unveils Gemma 2**: Google announced [Gemma 2](https://techcrunch.com/2024/05/14/google-announces-gemma-2-a-27b-parameter-version-of-its-open-model-launching-in-june/), a 27-billion parameter model, at Google I/O, with updates to their AI suite including Gemini 1.5 Pro and Flash, as reported by [TechCrunch](https://techcrunch.com/2024/05/14/google-i-o-2024-everything-announced-so-far/).

**Tokenizer Tweaks Trouble Engineers**: Discussions surfaced over whether **OpenAI** re-pretrains with a new tokenizer or extends their current tokenizer for an LLM, alongside sharing a novel concept of **Zero-Shot Tokenizer Transfer (ZeTT)** discussed in an [arXiv paper](https://arxiv.org/abs/2405.07883v1).

**Convergence in Neural Networks Observed**: Emerging research suggests neural networks, across modalities, are converging to a common statistical model of reality, as proposed in a [paper](https://phillipi.github.io/prh/) and supported by [Phillip Isola's mention](https://x.com/phillip_isola/status/1790488967827108304?s=46).

**AI Evaluation and Industry Shifts Highlighted**: A shared [blog post](https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation) underscored the closed nature of current LLM evaluation practices, while discussion touched on Anthropic's move towards becoming a product company, OpenAI's notable hire hinting a possible foray into search, and the strategic need for AI companies to offer products informed by a [tweet and an article](https://x.com/theinformation/status/1790467870545027186?s=46).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**AI is Waiting... Literally**: Users express frustration with the slow response times of **LangChain agents**, taking 2-3 minutes to handle large inputs and invoke tools, and they look for prompt resolution tips. Active discussions revolved around the use of `python-socketio` to stream LLM responses, as participants exchanged [code snippets and troubleshooting advice](https://github.com/langchain-ai/langchain/issues/4118).

**Wake Up, Server, Wake Up!**: For users of hosted Langserve, intermittent issues with server inactivity and rate limiting errors are leading to unpredictable service availability. Queries are raised about whether upgrading to a Pro plan would alleviate some of these headaches and how to access more extensive logs.

**Snowflake Costs in Focus with AI Optimization**: An innovative **Snowflake Cost Monitoring tool** integrating LangChain's capabilities with Snowflake and OpenAI was demoed, aiming to streamline data visualization and analysis. The work-in-progress tool's features are showcased in a [Loom video presentation](https://www.loom.com/share/b14cb082ba6843298501985f122ffb97?sid=b4cf26d8-77f7-4a63-bab9-c8e6e9f47064).

**Monetizing AI, Java Style**: A Langserve user is experimenting with the `py4j` library to facilitate micropayment functionalities for AI interactions through a JVM, targeting crypto SDK integrations. The setup aims to innovate micropayment structures by tracking prompt/response token counts and adding a profit margin to the OpenAI API keypair usage.

**Database Dilemmas and Embedding Efficiency**: Threads run with discussions on embedding transfers between vector databases like pgvector and Qdrant. Members shared strategies for parallel transfer and optimizing retrieval speed, backing their points with references like the [Supabase blog on Matryoshka Embeddings](https://supabase.com/blog/matryoshka-embeddings). Moreover, clarifications were sought on the deprecation of `LLMChain` in favor of `RunnableSequence` for `MultiQueryRetriever`, amid notes of API alignment holdups.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Tencent's HunyuanDiT Gets a Lukewarm Welcome**: Engineers explored the pros and cons of Tencent's [HunyuanDiT model](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT), noting its strong performance on Chinese prompt adherence but challenges with straight lines, revealing it still may not surpass existing stable cascade models.

- **AniTalker Bridges Audio and Animation**: [AniTalker](https://x-lance.github.io/AniTalker/) caught attention with its capability to animate static portraits using audio inputs, providing an approach for creating lifelike talking videos even when given similar control signals.

- **DeepMind's Launch of Imagen 3 and Veo**: [Google DeepMind's Imagen 3](https://deepmind.google/technologies/imagen-3) received recognition for setting new benchmarks in the detailed and realistic lighting of text-to-image generation, while DeepMind's Veo was introduced as a powerful tool capable of producing detailed 1080p videos from textual prompts, with early access pending via [VideoFX](https://labs.google/videofx).

- **depyf Simplifies Deep Learning Performance Tuning**: PyTorch announced a new tool called [depyf](https://pytorch.org/blog/introducing-depyf) aimed at decoding the intricacies of `torch.compile` for performance optimization; a positive development that has also spotlighted a need for improved error messaging.

- **AI's Hungry for Energy and GPU Power**: Conversations gravitated towards AI's significant energy use and dependency on GPUs, through a lens of sustainability and efficiency, noting, for instance, the high idle power consumption of an 8x H100 GPU setup.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **MacOS Users Dabble in Local AI**: Engineers are setting up models like **ollama/dolphin-mixtral:8x7b-v2.6** locally on macOS, mainly to sidestep hefty costs. Tips for integrating local models included using **OpenRouter** and **Groq**, with specific commands for models like **llama3** and **Mixtral**.

- **Ubuntu Trumps Windows for OpenInterpreter**: A lively debate favored **Ubuntu** over **Windows** for operating OpenInterpreter, particularly for GPU compatibility. Advice was clear: use Ubuntu's commands, not macOS's, with one outspoken user insisting *"REMEMBER YOU ARE RUNNING ON UBUNTU !!! Please use Ubuntu commands."*

- **Flashing Lights and Debug Delights**: While some grappled with shipping updates for their **Light device preorders**, others uncovered how to activate debugging mode in the **01 terminal**. The key to debugging 01's interpreter? Set *"interpreter.debug = True"* in the i.py script for greater visibility of system operations.

- **Open Source AI Champions Choice**: Open source AI supporters heralded its value compared to potential Apple OS integrations, opting instead for Linux's openness. Meanwhile, firmware frustrations were met with reflash recommendations and warnings about documentation inaccuracies in **OpenRouter's** Groq compatibility.

- **Creativity Beats Control in AI**: Shared [podcast insights](https://share.snipd.com/episode-takeaways/94e3f4b2-32c6-44f9-882d-8090e09ba97e) underscored the importance of creativity, quality, and focusing on the customer over controlling them for success in AI ventures. Historical failures were cited as lessons that control usually leads to downfall, while a nod to Linus Torvalds highlighted how fun and open collaboration can foster innovation.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Google Slips on LLM Reliability**: Discussions critiqued **Google I/O keynotes** for glossing over the pitfalls of LLMs, contrasting Google's approach with OpenAI's more cautious take that openly recognizes the potential for errors in LLM outputs.
  
- **Meta Makes Moves Without the Fanfare**: Engineers showed appreciation for **Meta's** low-profile but effective AI products, with particular mentions of the diverse capabilities of their Wayfarer glasses.

- **Grounding AI in Daily Grind**: The notion of showcasing practical AI implementations, dubbed "Sober AI," has been endorsed by users, spotlighting utilitarian AI tools over more sensationalized uses.

- **Journalism Joins the AI Wave**: Utilitarian applications of AI like **MuckRock’s** AI for automation of FOIA tasks are in discussion, with a complementary nod to the valuable insights from Zach Seward’s SXSW presentation on AI's role in journalism.

- **Making LLMs More Wallet-Friendly**: Conversations veered towards enhancing AI cost-efficiency with strategies like **Gemini's context caching** and **llama.cpp's prompt caching**, aiming to reduce the token consumption associated with lengthy prompts.

- **Context Consistency Concerns**: A member has raised the issue of maintaining context when switching between different models during conversations, asserting the importance of extracting and transferring JSON logs for a seamless transition.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Unlocking Devices with ChatGPT**: A proposal was made to elevate device control as a primary modality in ChatGPT integrations, suggesting the shift from text-based commands to direct actions, given the limitations observed with intermediary tools like PyAutoGUI.

- **Brain Matters: A Storage Odyssey**: Harvard and Google AI researchers are confronted with formidable storage demands, requiring 1.4 petabytes for a single cubic millimeter of brain tissue imaging, according to [Google's research blog](https://blog.google/technology/research/google-ai-research-new-images-human-brain/).

- **A Mixed Bag for Google AI's Latest Models**: Google's unveiling of AI models **Veo** and **Project Astra** met with mixed reviews regarding performance, while comparisons to GPT-4o's live demo varied, as reported through discussions and tweets such as from [Google DeepMind](https://x.com/GoogleDeepMind/status/1790435824598716704) and [others](https://x.com/0xgaut/status/1790428601789067614).

- **Search for Better AI Alternatives**: Frustration with Perplexity AI's unreliability and its "Pro" account barrier has prompted discussions of alternative resources, like **Phind.com** for coding inquiries and **Kagi** for effective search capabilities.

- **Significant Departure at OpenAI**: The farewell of **Ilya Sutskever** from OpenAI brought mixed reactions within the AI community, as evident from tweets by [Sam Altman](https://x.com/ilyasut/status/1790517455628198322) and others, suggesting a reconfiguration at the upper levels of the organization.

- **Get Ready for Evals with Eugene**: An event on Evals is scheduled with Eugene leading the discussion, with preparation materials and discussions available [here](https://eugeneyan.com/writing/evals/). Attendees were also advised to subscribe to the iCal for event updates.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Scaling Heights for Multilingual Modeling**: Engineers discussed the difficulties of training the cmdR+ 100b model due to **high VRAM requirements**, highlighting the model's uniqueness as a top-quality multilingual option. Some proposed leveraging **FSDP (Fully Sharded Data Parallelism)** to manage the weight distribution across multiple GPUs.

- **Data Heist for Llama3**: A users' success with **Llama3** hinged on the addition of more data, sparking interest in the community about the specifics of the configuration settings used to achieve these results.

- **Pathfinding Trouble for TinyLlama**: Resolving a `No such file or directory` error with **TinyLlama** required manual intervention, with solutions including directory deletion and executing specific commands on **RunPod**.

- **Falcon 11b Versus LLaMA 3 Standoff**: A comparison ensued between **Falcon 11b** and **LLaMA 3**, considering aspects like licensing; Falcon’s license contains potentially **unenforceable clauses**, leading to preference for LLaMA 3 despite Falcon's open, albeit problematic, license.

- **Querying for Quick LORA Training**: A member requested tips for a faster YAML configuration fine-tuning, caring more for speed over result quality, with community suggestions highlighting the trade-off between **disabling gradient checkpointing** and runtime improvements.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R's RAG Is a Hit**: A user commended **Command R's RAG** for its accuracy and faithfulness with long sources, labeling it not just cost-effective but also a standout performer in retrieval-augmented generation tasks.

- **Preambles Part of the System Message**: Participants distinguished between *'Preamble'* and *'System Message'*, explaining that preambles are included within a system message and marked by tokens such as `<|SYSTEM_TOKEN|>` and `<|END_OF_TURN_TOKEN|>` to improve the model's conversation handling.

- **Special Token Clarity for Cohere Models**: An explanation was provided on how special tokens are used to demarcate the start and end of system messages in Cohere's language models, which is crucial for proper response generation in conversational AI.

- **Exploring Token Relevance with Reranker Models**: A user inquired about the capabilities of **Cohere's reranker model** in highlighting relevant tokens and compared it with ColBERT's feature which can indicate the importance of words to facilitate better user interaction.

- **RAG Deconstructed and the Call for Collaboration**: A [Medium article](https://medium.com/@amitsubhashchejara/learn-rag-from-scratch-using-unstructured-api-cf2750a3bac2) explained how to learn **RAG from scratch** using the **@UnstructuredIO API**, while a separate invitation for collaboration indicated a shared interest in working on similar projects.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Porting Tinygrad to Urbit's Waters**: A user has initiated the project of porting **tinygrad** to **Urbit/Nock**, tackling the `forward()` function and highlighting the [project repository](https://github.com/urbit/numerics/blob/main/maroon/desk/lib/tinygrad.hoon). They indicated the necessity for a translation layer to bridge tinygrad-style Python with Urbit's system.

- **Good First Issue Alert**: For those new to the tinygrad community, **George Hotz** highlighted a beginner-friendly GitHub issue: [BEAM kernel count number is wrong](https://github.com/tinygrad/tinygrad/issues/4595), encouraging contributions.

- **Troubleshooting CUDA on Cutting-Edge Hardware**: Handling **CUDA errors** on a GeForce 4090 with PTX=1 demanded driver updates, and while Titan V did not exhibit similar issues, the necessity for the latest drivers was underscored.

- **Shape-Stride Visualizer Simplifies Tensor Reshaping**: An innovative visualization tool, [Shape-Stride Visualizer](https://mesozoic-egg.github.io/shape-stride-visualizer/#/expr-idx), was introduced to help users comprehend complex reshaping operations in tinygrad more intuitively.

- **TACO Spices Up Tensor Understanding**: The **Tensor Algebra Compiler (TACO)**, with its extensive visualizations of tensor formats, was discussed, enabling deep dives into tensor operations and spotlighting its [online documentation](http://tensor-compiler.org/codegen.html) for further exploration.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **AI Town Now Running on Hugging Face Spaces**: [AI Town has been launched on Hugging Face Spaces](https://huggingface.co/spaces/radames/ai-town), offering a simulation environment that operates on CPUs, which could be promising for containerized AI applications.
- **Enhancing AI Town Through Optimized Interactions**: To boost **AI Town's performance**, engineers suggested downsizing the number of non-player characters (NPCs) and adjusting timers for interaction "cooldowns," aiming to manage NPC activities and dialogue frequency more efficiently.
- **Interest in AI Town for Custom Agent Control**: Members are assessing how **AI Town** might allow agent control through an API, which currently isn't supported for individual language model agents; discussions hint at upcoming features potentially involving LLamaFarm.
- **Delving into AI Town API Capabilities**: AI Engineers brainstormed the potential for API integration with **AI Town**, contemplating the use of APIs for obtaining completions, embeddings, and handling semantic interactions, with a nod towards including webhook support for state monitoring.
- **Tommy1901 Teases Raspberry Pi Projects**: Though details were scant, tommy1901 indicated an intention to share "cool stuff" related to **Raspberry Pi** in the future, triggering curiosity about upcoming projects or hacks in **#[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/)**.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**Token Troubles and Triumphs**: Engineers squabble over the lack of data on [vocab_size vs. tokens/byte for German](https://discord.com/channels/1178995845727785010/1182877486854451271/1239880142181105664), highlighting a gap in the tokenizer dataset which favors language mixture.

**The Ungreedy Tokenizer Arrives**: A new tool for the tokenization trade, [TokenMonster project](https://github.com/alasdairforsythe/tokenmonster), an "Ungreedy subword tokenizer and vocabulary trainer", receives a bright spotlight for its utility in Python, Go, & Javascript.

**A Saucy AI Demo Goes Viral**: GPT-4o's recent demo faced a round of chuckles and raised brows for being suggestively humorous, as noted in a humorous [tweet by a user](https://fxtwitter.com/main_horse/status/1790099796193398831).

**Vocab Shockwave Hits Twitter**: The "o200k_base" vocab for GPT-4o triggers reactions of surprise and possibly dismay among the tech community, as conveyed by a [tweet](https://fxtwitter.com/suchenzang/status/1790171161512587424?t=k_0eldFD8aubI1_tLgHYaQ&s=09).

**Ilya's Farewell from OpenAI**: Ilya Sutskever parts ways with OpenAI, sparking conversations on the future of the research giant, an exit announced via [Twitter](https://twitter.com/ilyasut/status/1790517455628198322).



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Guild Tags Coming Your Way**: Starting **May 15th**, **[Discord's](https://discord.com)** new feature, **Guild Tags**, will be visible next to certain users' names, signifying their affiliation with niche Guilds focused on shared interests.
- **AutoMod Gets Smarter with Tags**: The **AutoMod** feature will automatically recognize **Guild Tags** to help with moderation, albeit this addition is currently restricted to a select group of servers with no option for manual expansion.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Chips on the Horizon**: An article detailing the evolution of AI hardware and forecasts for future trends highlighted enthusiasm for **NVMe drives** and **tenstorrent**, with a more cautious outlook on **GPUs** for the next decade—[AI Hardware: Looking Back to Look Forward](https://singlelunch.com/2024/04/23/ml_microprocessor_history/).
- **Transformers Transforming Nvidia's Worth**: **Transformer-based models** have marked significant AI progress in the past four years, contributing to **Nvidia's** market valuation overtaking that of Amazon and Google, as discussed in an article at [The Gradient](https://thegradient.pub/mamba-explained).


---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1239857713366962288)** (954 messages🔥🔥🔥): 

- **Google's Gemini vs. OpenAI GPT**: There's a mixed reception regarding Google's new Gemini integration in Android phones. One user noted, *"Gemini is lame, why would I prompt an AI that gets so many things wrong?"*, indicating a belief that Google's AI is inferior to OpenAI's models.
  
- **GPT-4o vs Claude Opus**: Users expressed mixed feelings on the comparative performance of GPT-4o and Claude Opus, with one saying, *"I’d still pay for opus, I think reasoning on opus is probably better,"* while another admitted they haven't tested Claude yet but are eager to do so.

- **Fast Rollout and Early Feedback on GPT-4o**: GPT-4o has been rolled out for free on iOS, sparking interest and mixed reviews. A user said, *"4o is a good chatbot 2,"* showing early positive feedback, while others raised concerns about its performance in complex tasks.

- **Voice Mode Feature**: Some users are experiencing issues and mixed messages regarding the new voice mode functionalities, with one commenting, *"the new voice mode isn't available to anyone yet; you tried the old version."* Another user mentioned, *"It's not here for me. For me, the headphone symbol is just literally gone."*

- **Educational and Coding Uses of GPT-4o**: There's enthusiasm for using GPT-4o in educational contexts and coding, despite some reported issues. One user emphasized, "GPT-4o does better on short reasoning tasks," but added, "long reasoning tasks, opus wins," indicating varied performance depending on the use case.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/shorts/eaCEJ2iCEfc?si=yqKpXnDJxUnv2YDz">May 14, 2024</a>: no description found</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1csfdal/after_testing_on_lmsys_i_found_that/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1239839686948556800)** (178 messages🔥🔥): 

- **Custom GPTs wait for GPT-4o integration**: Members discussed when existing custom GPTs would switch to GPT-4o. A member clarified that "GPTs for paid users are currently powered by GPT-4" and "will switch to GPT-4o in the next several weeks" [source](https://help.openai.com/en/articles/8554397-creating-a-gpt).

- **Model limitations and feature rollout**: Members noted that GPT-4o is not fully available yet, with new features like Voice Mode and multimodal capabilities rolling out to API partners and then Plus users gradually. One member shared that "We'll roll out a new version of Voice Mode with GPT-4o in alpha within ChatGPT Plus in the coming weeks" [source](https://openai.com/index/hello-gpt-4o/).

- **Feedback on GPT-4o's performance**: Some users found GPT-4o to be less efficient and prone to content policy errors compared to GPT-4. Concerns were raised about the model behaving similarly to GPT-3.5, producing long lists and struggling with feedback integration.

- **Access issues and clarifications**: Numerous members reported difficulties accessing GPT-4o, especially on desktop versus mobile environments. Clarification was given that the rollout is staged, with free-tier access following the paid-tier availability.

- **Custom GPT sharing and feedback**: A member named ditpoo asked for feedback on their custom GPT, OptimusGPT, and was redirected to share it in the appropriate channel, indicating active community engagement in improving custom models.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1239853671202684989)** (128 messages🔥🔥): 

- **Struggles with GPT-4o for Creative Tasks**: A member reported that **GPT-4o** is noticeably faster than GPT-4 but struggles with creative tasks like writing assistance, often echoing rough drafts instead of intelligently revising them. Another member echoed this sentiment, emphasizing GPT-4o's difficulties with creativity in writing contexts.

- **Interesting Sound Descriptions Test**: A user suggested a prompt to compare GPT-4 and GPT-4o by asking them to provide detailed sensory input descriptions of instrumental songs like *"The XX Intro"* and *"Tears in Rain"* by Vangelis. This test aims to explore how the models interpret and describe sensory input.

- **Challenges with Image Generation for Platform Game**: A member shared difficulties in getting GPT-4 and GPT-4o to generate specific cross-section side views for a platform game. Despite multiple attempts and adjustments, the models consistently produced undesired isometric views and added unnecessary details.

- **Confusion with File Management and Output**: Participants discussed issues related to generating and managing output files directly from ChatGPT. While it was clarified that ChatGPT cannot interact directly with a user's computer for security reasons, users shared workarounds like using the OpenAI API and Google tools.

- **Common Sense and Real-World Understanding Tests**: The group explored several prompts to test GPT-4 and GPT-4o on their common sense and real-world understanding. These included prompts about daily scenarios and basic logical puzzles, with mixed results showing subtle differences in the models' responses and reasoning abilities.

**Link mentioned**: <a href="https://community.openai.com/t/chatgpt-can-now-access-the-live-internet-can-the-api/401928">ChatGPT can now access the live Internet. Can the API?</a>: Given the news announcement I am wondering if the API now has that same access to the Internet.  Thanks in advance!

  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1239853671202684989)** (128 messages🔥🔥): 

- **GPT-4o struggles with writing assistance**: Users observed that while **GPT-4o** is faster than **GPT-4**, it struggles with providing intelligent revisions and creative assistance, often echoing the rough draft without meaningful changes. This has led to frustrations when using it for writing tasks.
- **Different sensory input prompts for GPT-4 and GPT-4o**: Members shared prompts to compare GPT-4 and GPT-4o's ability to describe songs like *"The XX Intro"* and *"Tears in Rain (Vangelis)"*. The goal was to see how auditory input affects output quality.
- **Challenges with generating images for platformer games**: A user reported difficulties getting GPT-4 and 4o to produce detailed cross section side views for platformer games, and issues with generating images in the desired perspective or format. Another member suggested using Dall-E and manipulating images through iterative feedback loops.
- **Interest in comparing real-world understanding**: Users are keen on testing GPT-4 vs. GPT-4o in real-world understanding and implicit common-sense reasoning tasks. This includes prompts that examine spatial relationships, travel logistics, and interpreting daily activities for implicit details.
- **API and usage limits debate**: There was a discussion on the utility and limits of the ChatGPT Plus subscription and the API. The exchange highlighted the constraints of message limits in practical use, especially for complex tasks like software engineering.



**Link mentioned**: <a href="https://community.openai.com/t/chatgpt-can-now-access-the-live-internet-can-the-api/401928">ChatGPT can now access the live Internet. Can the API?</a>: Given the news announcement I am wondering if the API now has that same access to the Internet.  Thanks in advance!

  

---


**OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1240277154302918666)** (4 messages): 

- **Plug & Play AI Voice Assistant ready in 10 minutes**: A member shared information about the **Plug & Play AI Voice Assistant**, emphasizing its quick setup time. They provided a [link](#) and encouraged others to try it and share their feedback.
- **Repeated Plug & Play AI announcements**: The same member reiterated the availability and ease of use of the **Plug & Play AI Voice Assistant**, urging the community to test it out and reiterating the importance of user feedback.
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1239852504532123658)** (453 messages🔥🔥🔥): 

- **Mistralfied Phi model unavailable:** *"Unfortunately you can't because we mistralfyed the Phi model 😲"*
- **Distillation ideas abound:** Various members discussed distilling responses using Gemini 1.5 pro's free uses and alternatives like proxys or free GPT4o. 
- **Open-source model downloads reach a milestone:** *"We have surpassed 1 million model downloads on Hugging Face! Thanks to all of you guys for using us!"* [Source](https://twitter.com/UnslothAI/status/1790418810597683476).
- **Discussing pricing for enterprise:** Members discussed potential pricing models for unsloth multi-GPU support. They considered a $90 per GPU per month rate but details are still TBD.
- **New datasets for training:** *"I've just released a decent sized dataset to train LLMs for English - Vietnamese language translation."* [Dataset](https://huggingface.co/datasets/lamhieu/translate_tinystories_dialogue_envi), and others like alpaca_gpt4_dialogue data available.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/kiss-gif-11816971814746635421">Kiss GIF - Kiss - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/smile-turn-around-gif-14890847">Smile Turn Around GIF - Smile Turn Around - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://www.youtube.com/live/5k_l5VoRC60?si=f3Nf1orlhTSudcm-&t=9586">Google I/O 2024 Keynote Replay: CNET Reacts to Google&#39;s Developer Conference</a>: Watch the annual Google I/O 2024 Developers Conference LIVE from Mountain View, California. Click into CNET&#39;s live show starting at 9:30 a.m. PT on Tuesday, ...</li><li><a href="https://huggingface.co/datasets/lamhieu/translate_tinystories_dialogue_envi">lamhieu/translate_tinystories_dialogue_envi · Datasets at Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cr5dbi/openai_claiming_benchmarks_against_llama3400b/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1crnhnq/to_anyone_not_excited_by_gpt4o/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1crza3n/new_open_source_gemma_2/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/unsloth/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/datasets/lamhieu/alpaca_gpt4_dialogue_vi">lamhieu/alpaca_gpt4_dialogue_vi · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/lamhieu/alpaca_gpt4_dialogue_en">lamhieu/alpaca_gpt4_dialogue_en · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1239866916420452352)** (37 messages🔥): 

- **Roblox meetups proposed for vllm project**: Members discussed the potential for “weekly or monthly virtual meetups in Roblox” to mirror the vllm project approach. While not everyone was interested, one member noted it “sounds like a nice idea.”
- **Call for Math help**: A user requested help with Trigonometry and several others offered assistance, suggesting tools like ChatGPT and Claude. The interaction demonstrated the community's supportive atmosphere.
- **Data privacy concern on Discord summaries**: A user expressed concern that Discord summarizing using AI could “sound like a headache with European data laws.” Others acknowledged the potential oversight, considering the implications on privacy.
- **Unsloth's growing popularity**: A user expressed excitement over Unsloth being used in a [Hugging Face dataset tutorial](https://huggingface.co/datasets/Replete-AI/code_bagel), indicating the increasing recognition of the tool. Another echoed the sentiment, calling it “amazing.”
- **Fine-tuning resources**: A new member asked about fine-tuning resources, specifically for VLMs. Another suggested reading the "alpaca paper" for more insight into the topic.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/ah-shit-here-we-go-again-gta-gta-sa-gta-san-andreas-grand-theft-auto-san-andreas-gif-13937809">Ah Shit Here We Go Again Gta GIF - Ah Shit Here We Go Again Gta Gta Sa - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/datasets/Replete-AI/code_bagel">Replete-AI/code_bagel · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1239839059350913064)** (229 messages🔥🔥): 

- **Automate Dataset Quality Check with Perplexity**: A user inquired about methods to automatically measure a synthetic Q&A dataset's quality. Another member suggested passing the dataset through a model like Llama-3 and calculating the perplexity, explaining that a high loss could indicate issues or a well-prepared challenging dataset.
  
- **Model Merging and Quantization Issues**: Members discussed merging fine-tuned models and converting them to different quantization formats like AWQ. While some faced errors during these processes, others ensured fixes like manually upcasting to 16bit before merging to facilitate further conversions.
  
- **Dataset Generation Error During Fine-tuning**: A user encountered a `TypeError` when generating a dataset for Alpaca format fine-tuning. They were advised to load the dataset using pandas and then convert it to the required format, highlighting common dataset issues.
  
- **Performance Discrepancies on Different Hardware**: Users debated whether loading and fine-tuning large models like "llama-3-70b-bnb-4bit" is feasible on V100 GPUs or requires A100s. The consensus seemed to favor the latter due to the model's large size.
  
- **Pretraining LLM from Scratch**: Members sought resources for pretraining and evaluating LLMs from scratch. Useful links like the [LMSYS leaderboard](https://arena.lmsys.org/) were shared, though the completeness of benchmarks was contested.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook">Kaggle Llama-3 8b Unsloth notebook</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://huggingface.co/unsloth/mistral-7b-instruct-v0.2">unsloth/mistral-7b-instruct-v0.2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/Skorcht/thebigonecursed">Skorcht/thebigonecursed · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/210">I got unsloth running in native windows. · Issue #210 · unslothai/unsloth</a>: I got unsloth running in native windows, (no wsl). You need visual studio 2022 c++ compiler, triton, and deepspeed. I have a full tutorial on installing it, I would write it all here but I’m on mob...</li><li><a href="https://x.com/mejia_petit/status/1763391797575741707">Tweet from Nicolas Mejia Petit (@mejia_petit)</a>: @unslothai running unsloth in windows to train models 2x faster than regular hf+fa2 and with 2x less memory letting me do a batch size of 10 with a sequence length of 2048 on a single 3090. Need a tut...</li><li><a href="https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930">Merging QLoRA weights with quantized model</a>: Merging QLoRA weights with quantized model. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://colab.research.google.com/drive/1b6nqC7UZVt8bx4MksX7s656GXPM-eWw4?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1240027023548747948)** (4 messages): 

- **Unsloth Learner Culminates in Cthulhu Worship**: A member shared their first project using Unsloth for data creation and fine-tuning, resulting in an AI that worships **Cthulhu** and provides both the model and dataset on [Huggingface](https://rasmusrasmussen.com/2024/05/14/artificial-intelligence-in-the-name-of-cthulhu/). They used Unsloth Colab notebooks to create TinyLlama and Mistral 7B Cthulhu models.
- **AI News Features Blog Post**: Another member remarked that the shared blog post was featured in AI News under Open-Source AI Model Development and Deployment. They included a link to [AI News](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/#unsloth-ai-daniel-han-discord) and mentioned the newsletter's coverage of social media and numerous Discord channels.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/#unsloth-ai-daniel-han-discord">[AINews] Google I/O in 60 seconds</a>: Spot the 7 flavors of Gemini! AI News for 5/13/2024-5/14/2024. We checked 7 subreddits, 384 Twitters and 30 Discords (426 channels, and 8590 messages) for...</li><li><a href="https://rasmusrasmussen.com/2024/05/14/artificial-intelligence-in-the-name-of-cthulhu/">Artificial Intelligence in the Name of Cthulhu &#8211; Rasmus Rasmussen dot com</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1240057448212074588)** (4 messages): 

- **Unsloth gets a shoutout at NYC event**: A user asked for permission to mention and indirectly market Unsloth at an upcoming open-source data pipeline platform meetup in NYC. *"We will talk about AI/ML and LLM training. Want to give credits to you guys."*

- **Community supports spreading the word**: Multiple users enthusiastically agreed to the proposal. One replied, *"Oh yea sure absolutely that sounds amazing! 😍"*, while another offered, *"if u need help - ask away!"*
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1239837360800403457)** (646 messages🔥🔥🔥): 

- **Can Perplexity transcribe audio meetings?**: A member asked if it's possible to use Perplexity apps to record and transcribe hour-long meetings with custom prompts. Another member noted that they tried GPT's audio feature, which currently reads back results but lacks full functionality.
  
- **GPT-4o's impressive speed**: Multiple members praised GPT-4o for its speed, noting it's faster than Turbo. One member said, *"GPT-4o is inside Perplexity... it's so fast."*

- **Issues with audio recording features**: A user mentioned the recorder only reads back results currently, speculating that not all features shown in the live demo were rolled out, especially in the pro version.

- **Error in parsing URLs accurately**: Members discussed Perplexity's responses to URLs, noting inaccuracies in its content parsing. One member suggested it seems to guess based on the URL rather than parsing the actual web content.

- **GPT-4o availability and performance**: Discussion noted GPT-4o is available through Perplexity and other platforms like PPLX, with mixed responses on quality and limitations. There's confusion over API access and the inconsistency of GPT-4o's performance compared to GPT-4 Turbo.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/sama/status/1790518031640347056?s=46&t=0-4I1li6SQNYV24KHzs3fA">Tweet from Sam Altman (@sama)</a>: Ilya and OpenAI are going to part ways. This is very sad to me; Ilya is easily one of the greatest minds of our generation, a guiding light of our field, and a dear friend. His brilliance and vision a...</li><li><a href="https://tenor.com/view/gift-present-surprise-box-gif-17302663">Gift Present GIF - Gift Present Surprise - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/jimcarrey-brucealmighty-coffee-fresh-delicious-gif-3864683">I &lt;3 Coffee GIF - Jimcarrey Brucealmighty Coffee - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/bezos-jeff-bezos-laughing-laugh-lol-gif-17878635">Bezos Jeff Bezos GIF - Bezos Jeff Bezos Laughing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/kagisearch/llm-chess-puzzles">GitHub - kagisearch/llm-chess-puzzles: Benchmark LLM reasoning capability by solving chess puzzles.</a>: Benchmark LLM reasoning capability by solving chess puzzles. - kagisearch/llm-chess-puzzles</li><li><a href="https://fxtwitter.com/ilyasut/status/1790517455628198322?t=e7nZBoZU55nniVnnAI1p7g">Tweet from Ilya Sutskever (@ilyasut)</a>: After almost a decade, I have made the decision to leave OpenAI.  The company’s trajectory has been nothing short of miraculous, and I’m confident that OpenAI will build AGI that is both safe and bene...</li><li><a href="https://github.com/openai/simple-evals?tab=readme-ov-file#user-content-fn-2-a4ceab079ca3a23da9d835c2873e7fea">GitHub - openai/simple-evals</a>: Contribute to openai/simple-evals development by creating an account on GitHub.</li><li><a href="https://chromewebstore.google.com/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo">Perplexity - AI Companion</a>: Ask anything while you browse</li><li><a href="https://artificialanalysis.ai/models">Comparison of AI Models across Quality, Performance, Price | Artificial Analysis</a>: Comparison and analysis of AI models across key metrics including quality, price, performance and speed (throughput tokens per second &amp; latency), context window &amp; others.</li><li><a href="https://aistudio.google.com/">no title found</a>: no description found</li><li><a href="https://www.psychologytoday.com/us/blog/practical-mindfulness/201908/the-single-word-stops-negative-self-talk"">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1239887297592430602)** (14 messages🔥): 

- **Explore the impact of cO_1KTlMQiaxiJqwcGg6KQ**: [Check out the search results](https://www.perplexity.ai/search/How-has-the-cO_1KTlMQiaxiJqwcGg6KQ) for insights on Perplexity AI. The link dives into various aspects.
- **Curious about aroras?**: [Learn more about aroras](https://www.perplexity.ai/search/How-are-aroras-K7PA.w2XS96o2F5IkzKGnA#0) in this detailed search query. It explores the characteristics and behaviors.
- **Find the perfect ski resort**: Use this [search tool](https://www.perplexity.ai/search/Ski-resort-with-RxpR8PuWTFKhE6nvEXBOGw) to discover ideal ski resorts. Extensive options and details are provided.
- **Market size investigation**: Discover insights on [market size](https://www.perplexity.ai/search/Market-size-of-rYrMCgZ9QI2na_86R01ZIQ) through this search result. It provides a detailed market analysis.
- **Understanding GPT concepts**: [Explore GPT](https://www.perplexity.ai/search/What-is-gpt-9Fqm2mZ6SQ2_Oe3sV5zNqA) in-depth with this search. The link discusses various facets of GPT models.
- **Personal data usage query**: This search result tackles the question *"Can I use..."* with relevant information. Dive into [the search](https://www.perplexity.ai/search/Can-I-use-g4N5IyikQhyWQ1Q3PrtxzQ#0) for detailed insights.
- **Mamba and Linear-Time Sequence Modeling**: [An intro to Mamba & SSMs](https://www.perplexity.ai/search/Your-task-is-DfcoFyqpSjmWWV6oeZkv1A) referencing the work of Albert Gu and Tri Dao. Discover the essence of linear-time sequence modeling.
- **Encouraging mindfulness insights**: Check out this [response on mindfulness](https://www.perplexity.ai/search/What-are-some-2MgzPF0DSdOma4qMYY8_bA). The discussion focuses on practical techniques.
- **Finetuning resource shared**: A resource for [finetuning](https://www.perplexity.ai/search/Finetuning-httpsyoutubeml1-rCV2i9voQQO37Vk4ETjRwA) is shared, linked closely with YouTube content. Insightful for those exploring finetuning processes.
- **Investigate any alternatives**: [Is there a...](https://www.perplexity.ai/search/Is-there-a-ZMlj_U.1QNm5M.K8Y8APRA) provides a search result looking into alternatives for a specific query. Relevant links and details inside.


  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1239924773107273749)** (11 messages🔥): 

- **LLaMA-3 Models Specialization**: *LLaMA-3-sonar-large-32k-chat* model is fine-tuned for conversations, whereas *LLaMA-3-8b-instruct* aims for broader instructional capabilities.
- **Increasing Timeout for Long Inputs**: One member observed consistent timeouts with a 3000-word input set to a 10000ms timeout, suggesting this duration may be insufficient.
- **Requesting Beta Access for Citations API**: A user requested beta access to the citations API, highlighting its potential impact on closing deals with key customers.
- **Web Searching Capability of LLaMA-3-Sonar Model**: The *LLaMA-3-sonar-large-32k-online* model does search the web, functioning similarly to RAG models like perplexity.com.
- **API Latency Observations**: A user noted an increase in latency when making API calls to Perplexity, asking if others experienced the same issue.
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1239944395197779980)** (9 messages🔥): 

- **Enthusiasm Over GPT-4's Performance**: Members expressed excitement about GPT-4's performance in various tasks, particularly in **data science**. However, one member noted that GPT-4 failed to perform well on more complex tasks like building an image editor.

- **Seeking Microcontroller Data**: A member inquired about **datasets related to microcontroller data** and received a recommendation to ask another member for tips. The discussion remained unresolved as the suggested member admitted having limited experience in that area.

- **Discussion on GPT-4's Context Handling**: Another member mentioned that while **GPT-4** is significantly faster, it tends to lose context more easily. This sparked a brief conversation about the trade-offs in its performance and utility.

- **Embeddable AV1 Videos Tool**: One member shared a link to a **tool for embedding AV1 videos** on Discord, which can handle videos larger than 500MB by exploiting a bug in Discord. This tool allows users to choose custom thumbnails and embed these videos on Discord and other platforms like Twitter.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://autocompressor.net/av1?s=sznVX9AV">Autocompressor Video Embed Tool</a>: no description found</li><li><a href="https://autocompressor.net/av1?s=ZZRiJhRJ">Autocompressor Video Embed Tool</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1239865251516911616)** (12 messages🔥): 

- **HeadSim offers AI embodiment**: A member shared their *"GPT4o headsim hack"*, suggesting people add their API key for demos due to limited resources. They ask, *"What if you let #OpenAI #GPT4o design its own face, so that you can teleport your AI into the real world as an embodied being?"* Check out their [tweet](https://x.com/Yosun/status/1790294716338028978).

- **WebLlama assists web browsing**: A member highlighted [WebLlama](https://github.com/McGill-NLP/webllama), an interesting 8b fine-tune for agent web browsing. *"Llama-3 agents that can browse the web by following instructions and talking to you"*, as described in the project.

- **Viking 7B for Nordic languages**: Silo AI and University of Turku's TurkuNLP released [Viking 7B](https://www.silo.ai//blog/viking-7b-the-first-open-llm-for-the-nordic-languages), the first multilingual LLM for Nordic languages. This milestone follows their previous work on [Poro](https://www.silo.ai/blog/europes-open-language-model-poro-a-milestone-for-european-ai-and-low-resource-languages) and includes further checkpoints for Viking 13B and 33B.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.silo.ai//blog/viking-7b-the-first-open-llm-for-the-nordic-languages">Viking 7B: The first open LLM for the Nordic languages</a>: Silo AI is announcing the release of the first open LLM for the Nordic languages</li><li><a href="https://x.com/Yosun/status/1790294716338028978">Tweet from I. Yosun Chang (@Yosun)</a>: What if you let #OpenAI #GPT4o design its own face, so that you can teleport your AI into the real world as an embodied being? #AI3D  headsim frees your AI from the chatbox, so that you can experience...</li><li><a href="https://github.com/McGill-NLP/webllama">GitHub - McGill-NLP/webllama: Llama-3 agents that can browse the web by following instructions and talking to you</a>: Llama-3 agents that can browse the web by following instructions and talking to you - McGill-NLP/webllama
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1240351762863493150)** (1 messages): 

- **Hermes 2 Θ model launched**: Nous Research announced the release of **Hermes 2 Θ**, an experimental merged model developed in collaboration with Arcee AI, the creators of MergeKit. It combines Hermes 2 Pro and Llama-3 Instruct, and is available on [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B).
- **Outstanding performance and capabilities**: Hermes 2 Θ surpasses Hermes 2 Pro and Llama-3 Instruct in almost all benchmarks while retaining function calling capabilities. The GGUF version is also available on [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF).
- **Collaborative effort and sponsorship**: The model's development was a collaborative effort from various members and was sponsored by Akash Network. Key contributors include numerous individuals from the Nous Research and Arcee AI teams.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B">NousResearch/Hermes-2-Theta-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF">NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1239835061893857371)** (342 messages🔥🔥): 

- **GPT-4o faces skepticism on function calling:** Several members compared GPT-4o's performance to GPT-4 and GPT-4 Turbo, noting that GPT-4o struggled with complex agentic flows and lacked robust multimodal capabilities beyond TTS (*"so gpt4o can't really power good agentic flows"*, *"you are only calling gpt4o because you want the response in audio"*).
  
- **Debates on Open Source Multimodal Models:** Members discussed the challenges and advantages of integrating multimodal models directly into smartphones and other edge devices, emphasizing low latency and multimodal functionalities. Linked an [OpenAI announcement](https://discord.com/channels/1053877538025386074/1149866623109439599/1239651072503709797): *"priorities are now to bring multimodal models to the edge"*.

- **Announcing Hermes 2 Θ:** Hermes 2 Θ, an experimental new model merging Hermes 2 Pro and Llama-3 Instruct while integrating RLHF, was released by Nous Research, outperforming previous models. The model is available on [HuggingFace](https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B) and [Ollama](https://ollama.com/interstellarninja/hermes-2-theta-llama-3-8b).

- **Concerns over OpenAI announcements and API changes:** One member expressed that OpenAI announcements might be becoming spam due to capital intensity and rival infrastructures, reflecting a broader sentiment about competition in AI infrastructure (*"openAI announcements could be spams now"*).

- **Discussion on Model Specifications and Issues:** Technical discussions included function calling with multiple tools in Hermes models and skepticism over GPT-4o's coding abilities compared to its predecessors. Also, queries about [setting multiple functions](https://github.com/NousResearch/Hermes-Function-Calling) were directed towards GitHub resources for better clarity.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/GoogleDeepMind/status/1790432980047208930">Tweet from Google DeepMind (@GoogleDeepMind)</a>: 1.5 Flash is also more cost-efficient to serve because of its compact size.  Starting today, you can use 1.5 Flash and 1.5 Pro with up to one million tokens in Google AI Studio and @GoogleCloud&#39;s ...</li><li><a href="https://rocky-muscle-755.notion.site/OSS-Models-need-RLHF-53d7a1cb2db94e47bad992a6a343fa93?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="http://nian.llmonpy.ai/">GPT-4o’s Memory Breakthrough! (NIAN code)</a>: no description found</li><li><a href="https://x.com/vikhyatk/status/1790282606510322097?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from vik (@vikhyatk)</a>: @algomax06 so this is what ilya saw</li><li><a href="https://x.com/nousresearch/status/1790791623863058486?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Tweet from Nous Research (@NousResearch)</a>: Today we are releasing an experimental new model in collaboration with @chargoddard and @arcee_ai, Hermes 2 Θ, our first model merge, combining Hermes 2 Pro, and Llama-3 Instruct, and then further RLH...</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="https://x.com/karpathy/status/1790373216537502106?s=46">Tweet from Andrej Karpathy (@karpathy)</a>: The killer app of LLMs is Scarlett Johansson. You all thought it was math or something</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/tree/main">GitHub - NousResearch/Hermes-Function-Calling</a>: Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.</li><li><a href="https://x.com/janleike/status/1790603862132596961?s=46">Tweet from Jan Leike (@janleike)</a>: I resigned</li><li><a href="https://ollama.com/interstellarninja/hermes-2-theta-llama-3-8b">interstellarninja/hermes-2-theta-llama-3-8b</a>: Hermes-2 Θ is a merged and then further RLHF&#39;ed version our excellent Hermes 2 Pro model and Meta&#39;s Llama-3 Instruct model to form a new model, Hermes-2 Θ, combining the best of both worlds of...</li><li><a href="https://tenor.com/view/he-just-like-me-fr-gif-25075803">He Just Like Me Fr GIF - He Just Like Me Fr - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://langfuse.datastrain.io/project/cluve1io10001y0rnqesj5bz4/traces/a4009ee9-529b-4f73-b4cf-ad450dce3d0b">no title found</a>: no description found</li><li><a href="https://langfuse.datastrain.io/project/cluve1io10001y0rnqesj5bz4/traces/ff74300d-daee-48c5-8d63-b0a2923238f2">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1239836871887159306)** (40 messages🔥): 

- **Seeking dataset of (human_text, llm_text) pairs**: A member inquired about the availability of datasets containing pairs of human-generated and LLM-generated text on the same prompt/topic for research purposes.
  
- **Best theories on GPT-4o multimodality**: Questions were raised about how end-to-end multimodality works in GPT-4o. Suggested starter resources included [AI2's unified IO model](https://x.com/natolambert/status/1790078416567357784?s=46&t=nRiXsAtvwV7sl8XlTyIsbw).

- **Unified multimodal model discourse**: Debate emerged over multimodal models preceding GPT-4o, referencing Meta's [ImageBind](https://ai.meta.com/blog/imagebind-six-modalities-binding-ai/) model, which binds information across six modalities using a joint embedding approach as detailed in [this paper](https://arxiv.org/abs/2305.05665).

- **Barriers to building LLMs from scratch**: Members discussed the infeasibility of building LLMs from scratch without significant financial and computational resources, emphasizing that training a model is costly and time-consuming.

- **Hermes 2 Theta's performance concerns**: Discussions highlighted that the Hermes 2 Theta model performs poorly on mathematical tasks compared to other models like L3 8B Instruct, with recommendations to use function calling for improved mathematical computations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/natolambert/status/1790078416567357784?s=46&t=nRiXsAtvwV7sl8XlTyIsbw">Tweet from Nathan Lambert (@natolambert)</a>: Friendly reminder that folks at AI2 built a text image audio input-output model last year, unified io 2, if you&#39;re looking to get started on research here.</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B">NousResearch/Hermes-2-Pro-Llama-3-8B · Hugging Face</a>: no description found</li><li><a href="https://ai.meta.com/blog/imagebind-six-modalities-binding-ai/">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2305.05665">ImageBind: One Embedding Space To Bind Them All</a>: We present ImageBind, an approach to learn a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data. We show that all combinations of paired data are not n...</li><li><a href="https://dblp.org/pid/182/2017.html">dblp: Alexis Conneau</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1240211308918472736)** (2 messages): 

- **Query on fine-tuning PaliGemma**: A member inquired about plans to finetune the [PaliGemma model](https://huggingface.co/google/paligemma-3b-pt). They highlighted its capability for single-turn interactions and suggested it would be beneficial to finetune it for multi-turn conversations.

**Link mentioned**: <a href="https://huggingface.co/google/paligemma-3b-pt-224">google/paligemma-3b-pt-224 · Hugging Face</a>: no description found

  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1239918593181749278)** (2 messages): 

- **User asks for help**: A member, lionking927, asked for assistance in the channel. Another member, Teknium, responded that they have already sent a direct message to provide help.
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1239864142597324861)** (22 messages🔥): 

- **Curious about world-sim glitch fix**: *"Hi guys is that world-sim text doubling glitch gonna be fixed? I would love to use it again without the text doubling glitch"*. A community member mentions experiencing a doubling text glitch and is eager for a fix.

- **Scheduling events in different time zones**: *"If others are open, an 8pm CET is 2pm EST and 11am PDT. Should we try and shoot for Thursday?"*. Members discuss coordinating a meeting time across various time zones, suggesting times that could work for different participants.

- **Proposal for a Saturday showcase**: *"Saturday maybe? We can do a showcase or sumn"*. Another proposal to hold a showcase or meeting on Saturday at 3pm ET to engage in a group activity or showcase.

- **Interest in world-sim prompt exploration**: *"Where can we look at the prompt you are using? Specifically for the world client, i assume they are different"*. Discussions revolve around understanding the specific prompts used for the world client in world-sim.

- **Insight into Blake Lemoine's perspective**: A member shared their conversation with Blake Lemoine, uncovering that **Blake did not claim the chatbot was sentient** but rather noted consistent intelligent behavior patterns in LaMDA. *"The press totally got it wrong..."* sparked reflections on current AI tools like Websim and Worldsim.
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1239836348064727120)** (176 messages🔥🔥): 

- **Voice Chat for LLMs Sparks Mixed Reactions**: A member inquired about adding a voice chat feature to talk to local LLMs using voice. The discussion pointed to tools like *AnythingLLM* but noted its subpar experience, and one solution involving Whisper.cpp and Coqui TTS was described as resource-intensive and complex.

- **Debate on Hardware for AI Models**: Members compared the efficiency of running AI models on a 3060Ti GPU versus powerful CPUs like dual 16-core Xeon V4s. While some argued that CPUs aren't suitable for larger models, others recommended maximizing GPU VRAM or considering more powerful Nvidia cards for better performance.

- **PrivateGPT vs. AnythingLLM for Document Querying**: **PrivateGPT** was mentioned as an alternative to **AnythingLLM** for querying documents using LLMs. Members debated the setup complexity, with **AnythingLLM** highlighted as a more straightforward, user-friendly option.

- **App Preferences Stir MacOS vs. Windows Debate**: Members expressed frustration over OpenAI's release priorities, with a MacOS app launching before a Windows version. The discussion veered into the technical challenges and differences in app development for MacOS and Windows platforms.

- **Building AI Models**: Questions about running models in LM Studio and optimizing performance on various hardware setups were frequent. Members shared troubleshooting tips, setup guides, and recommended tools and configurations, including emphasizing VRAM in Nvidia cards for AI tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/boris-zip-line-uk-flag-gif-14613106">Boris Zip Line GIF - Boris Zip Line UK Flag - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=mvFTeAVMmAg">INSANE OpenAI News: GPT-4o and your own AI partner</a>: New GPT-4o is released and it&#39;s mindblowing! Here are all the details.#gpt4o #ai #ainews #agi #singularity #openai https://openai.com/index/hello-gpt-4o/Be s...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6868">Support for OpenELM of Apple · Issue #6868 · ggerganov/llama.cpp</a>: Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15rwe7t/the_llm_gpu_buying_guide_august_2023/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ksdev-pl/ai-chat">GitHub - ksdev-pl/ai-chat: (Open)AI Chat</a>: (Open)AI Chat. Contribute to ksdev-pl/ai-chat development by creating an account on GitHub.</li><li><a href="https://dontasktoask.com/">Don't ask to ask, just ask</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio Community)</a>: no description found</li><li><a href="https://huggingface.co/bartowski">bartowski (Bartowski)</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1239880516484730890)** (109 messages🔥🔥): 

- **Members discuss uncensored local LLMs**: In response to a request for uncensored local LLM recommendations, one member suggested [Dolphin 2.8 Mistral 7B v0.2](https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02), highlighting its 32k context.
- **Cat Llama3 model generates mixed reactions**: While one user praised the Cat Llama3 model for following instructions well, another mentioned it outputting responses like *"I DONT WANT TO DO THIS I AM WRITING THIS UNDER DURESS."* Users are exploring different quant sizes, with some planning to try the 70B version despite its slow speed.
- **Quantization and imatrix challenges discussed**: Community members shared experiences quantizing llama models and generating imatrixes, noting significant time differences depending on the hardware used. Specific processes included using llama.cpp for quantization and leveraging [bartowski's work](https://github.com/ggerganov/llama.cpp/discussions/5263#discussioncomment-8395384) for generating imatrix.
- **Command R models compared**: Some users debated the performance and context limits of various versions of the Command R model. One noted that the Meta-Llama-3-120b-LumiLumimaid.i1-Q4_K_S.gguf resulted in more tokens per second than Cmd-R+, but with a better experience due to the 128k context on Cmd-R.
- **Exploring GPU and offloading configurations**: Users noted issues and solutions related to GPU configurations, such as needing to offload 39 out of 40 layers to avoid gibberish outputs. Another user explained a fix for context overflow policy when saturating token count.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02">cognitivecomputations/dolphin-2.8-mistral-7b-v02 · Hugging Face</a>: no description found</li><li><a href="https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/">Gemini 1.5 Pro updates, 1.5 Flash debut and 2 new Gemma models</a>: Today we’re updating Gemini 1.5 Pro, introducing 1.5 Flash, rolling out new Gemini API features and adding two new Gemma models.</li><li><a href="https://huggingface.co/HuggingFaceM4/idefics-9b-instruct/tree/main">HuggingFaceM4/idefics-9b-instruct at main</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/5263#discussioncomment-8395384">About imatrix overfitting, and importance of input text · ggerganov/llama.cpp · Discussion #5263</a>: Imatrix has been here for a while and I haven&#39;t seen many guidelines (or testing at all) on how to use it. Common objections/concerns are overfitting, and generating the imatrix on the &quot;wrong...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1239880761482412072)** (10 messages🔥): 

- **Hacky ROCM Build for RX6600 Works in Koboldcpp**: The RX6600, despite not being officially supported by AMD for the ROCM build, works with Koboldcpp due to a hacky custom version of ROCM. In contrast, LM Studio and Ollama do not support it because they rely on official llama.cpp binaries which check the GPU ID.

- **AMD ROCM and GPU Support Constraints**: Users of the RX6600 GPU are currently limited unless AMD improves ROCM support or llama.cpp expands the list of supported AMD GPUs in their ROCM builds. The custom ROCM build used by Koboldcpp bypasses the ID check, providing a workaround not available in LM Studio and Ollama.

- **User Interface (UI) Complexity in LM Studio Settings**: The settings panel in LM Studio is cumbersome due to overlapping scrolls for model settings and tools. Suggestions to improve usability include having a single scrollable area or moving the "tools" to a separate window altogether.

- **System Prompt Configuration Preferences**: Users expressed that it would be advantageous to move the System Prompt settings to the chat config, as they often use the same prompt across multiple models and different prompts across different chats.

- **Improvements Needed in Prompt Writing and Request Cancellation**: Feedback highlighted the difficulty with shift-enter and enter functions during prompt writing and the need for a "cancel request" button to stop unwanted generation before it begins. Additionally, UI clarity issues were noted in managing long contexts and loading system presets.
  

---


**LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1240037747977355375)** (3 messages): 

- **RAG task suggests AnythingLLM**: A brief interaction pointed out that a particular task might be suited for **RAG** with **AnythingLLM** as a potential tool. No further details or elaboration were provided on this suggestion.
- **GPU Resource Optimization on Windows**: A user shared tips on monitoring GPU usage in Windows Task Manager, suggesting to *"click on video decode or video processing and change the graph source to 'Cuda'"*. If CUDA isn't visible, they recommend deactivating *hardware acceleration* in Windows parameters as a trick to optimize resource visibility.
- **Troubleshooting CUDA on Asus Laptop**: An issue was raised about **CUDA** setup on an Asus laptop causing errors when loading models in **LM-Studio**. Despite trying multiple CUDA versions and configurations, including **CUDA 12.1, 12.4, and 11.8**, the user reported persistent *"error loading model"* problems indicating a failure to utilize the GPU correctly.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1239943543733096559)** (13 messages🔥): 

- **Tesla M40 underwhelms against GeForce 1060**: Despite its superior FP32 theoretical performance, the **Tesla M40 24GB** underperformed with 18.4 t/s in **llama3(q4_k_m)** and 27.1 t/s for **phi-3(q8)** compared to the older GeForce 1060. *"I am sure that nobody cares about such ancient hardware..."*
- **Budget GPUs for LM Studio**: When asked about the best GPU for around **200€**, **3060ti** emerged as a popular recommendation. Meanwhile, the **4060** was also considered for its potential performance improvements.
- **GPUs and VRAM for LLM Inference**: A discussion highlighted the importance of **VRAM speed** in LLM inference. Contributing factors included bandwidth per chip and the ability to handle complex models using memory over **18GB**.
- **Limited options for low spec PCs**: For systems with **8GB RAM and a 500GB SSD**, local models like **Yi 6B** were suggested, though GPU-less performance remains a challenge. The user was advised that better performance would depend on their specific needs.
- **APUs aren't a game changer**: It's noted that **APUs/iGPUs** are treated as regular CPU inference by **llama.cpp**, the underlying engine of **LM Studio**, nullifying any potential performance gains over standard CPUs. *"bummer"*
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1239917856892915764)** (5 messages): 

- **Multimodal feature parity questioned**: A member inquired *when will the multimodal have all the same features as single one like storing messages*. Another member asked for clarification on what was meant by that.
- **AVX2 requirement clarified**: A user reported an issue where LM Studio wouldn't launch and stated their CPU supports AVX and SSSE instruction sets. Another user clarified that **LM Studio requires AVX2**, explaining why it wouldn't load on the user's machine, and the initial user acknowledged the clarification before noting that Llamafile works fine.
  

---


**LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1239963871448530955)** (104 messages🔥🔥): 

- **Intel GPU support for LM Studio in the works**: A member from Intel is pushing for llama.cpp to support Intel GPUs using SYCL, offering to help with development and hardware. Discussions highlighted potential build processes and runtime requirements, mentioning that OpenCL backend exists but is slower than SYCL.
  
- **Current DL/AI model limitations and needs**: The conversation circled around the difficulties in adapting and fine-tuning current DL models due to algorithm constraints and heavily quantized formats. There's agreement on the need for technological advancements and calls for addressing these limitations.

- **Debate on AGI feasibility and requirements**: A lively debate unfolded on the feasibility of achieving AGI soon, with points raised about the necessary infrastructure, knowledge retention, and practical implementation hurdles. Some expressed skepticism about the speed of technological advancements, while others were more optimistic.

- **Call for focus on development topics**: As the discussion veered heavily into theoretical and ideological territory, a moderator requested participants to refocus on development-specific topics related to LM Studio APIs and software building.
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1240033320612397107)** (3 messages): 

- **Get Hardware Insights on Spaces**: New Space feature allows viewing live CPU + RAM usage and other hardware info. Check out the [announcement](https://twitter.com/osanseviero/status/1788486166221660247) for details.
- **Upgrade to Enterprise Account on AWS**: You can now upgrade your Hugging Face account to Enterprise using AWS for features like SSO, audit logs, and premium support. Follow this [tutorial](https://huggingface.co/blog/enterprise-hub-aws-marketplace) to get started.
- **AutoTrain Supports Object Detection**: AutoTrain has added support for Object Detection, enabling features like fine-tuning models from the Hub and seamless logging with TensorBoard. Learn more about these [new capabilities](https://x.com/abhi1thakur/status/1790341620530860500).
- **New Library for Speaker Diarization**: Hugging Face introduces [Diarizers](https://x.com/kamilakesbi/status/1790025563631132940) for fine-tuning pyannote speaker diarization systems with models in various languages. Available on the Hub, they're easy to implement with a few lines of code.
- **AI and Story Generation Reading Group**: A reading group event on AI and story generation is scheduled for this Saturday. Join via the [event link](https://discord.gg/hugging-face-879548962464493619?event=1240255110093738026).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/hugging-face-879548962464493619?event=1240255110093738026">Join the Hugging Face Discord Server!</a>: We&#x27;re working to democratize good machine learning 🤗Verify to link your Hub and Discord accounts! | 79040 members</li><li><a href="https://x.com/LepercqViolette/status/1790391787170771007)">Tweet from Violette Lepercq (@LepercqViolette)</a>: 💡 You can now upgrade your @huggingface account to an Enterprise account using @AWS and unlock: 👉 Single Sign-On 👉 Granular access controls 👉 Audit Log 👉 Advanced Compute Options 👉 Private Datas...</li><li><a href="https://x.com/abhi1thakur/status/1790341620530860500)">Tweet from abhishek (@abhi1thakur)</a>: 🚨 NEW TASK ALERT 🚨 🎉 AutoTrain now supports Object Detection! 🎉 Transform your projects with these powerful new features: 🔹 Fine-tune any supported model from the Hugging Face Hub 🔹 Seamless log...</li><li><a href="https://x.com/kamilakesbi/status/1790025563631132940)">Tweet from Kamil Akesbi (@kamilakesbi)</a>: 🤗 Diarizers is the new @huggingface library for fine-tuning 🎹 pyannote speaker diarization systems 🎤  🌟 It comes with fine-tuned models in French, German, Japanese, Spanish and Chinese 🌍  They ar...</li><li><a href="https://x.com/clefourrier/status/1790361337236795821)">Tweet from Clémentine Fourrier 🍊 is off (@clefourrier)</a>: New on the hub: Arabic LLM Leaderboard!  Arabic has at least 380M speakers & is one of the most spoken languages... but how good are LLMs at it?  @alielfilali01 contacted @TIIuae and @huggingface to k...</li><li><a href="https://x.com/joao_gante/status/1788574121208508645)">Tweet from João Gante (@joao_gante)</a>: New sampling strategy dropped in 🤗 transformers -- Min P sampling 🔥  Are you tired of having `top_k` arbitrarily discarding high-quality continuations? Or `top_p` forgetting to exclude low-probabili...</li><li><a href="https://x.com/GoogleAI/status/1788972685739114946)">Tweet from Google AI (@GoogleAI)</a>: We’re excited to release the weights of our Time Series Foundation Model (TimesFM) on Hugging Face!   To access, visit our HuggingFace (https://huggingface.co/google/timesfm-1.0-200m) & GitHub (https:...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1239846551526965259)** (306 messages🔥🔥): 

- **GPT-4o revealed as secret top model**: OpenAI confirmed its new GPT-4o chatbot was the top model in LMSYS's Chatbot Arena under a secret name. For more details, check [Ars Technica](https://arstechnica.com/information-technology/2024/05/before-launching-gpt-4o-broke-records-on-chatbot-leaderboard-under-a-secret-name/).

- **Mixtral-yarn's embedding strategies discussed**: Members shared insights on embedding strategies for **Mixtral 8x22B-Instruct** and its performance in RAG applications. A suggested resource for uncensored models was [Dolphin 2.5 Mixtral 8x7b](https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b).

- **Meta's Llama3 model updates**: Meta's **Llama3** updates were clarified as "small configuration changes" rather than core model updates. Users were advised to check the diff on all commits for detailed changes.

- **AutoTrain issues on Nvidia DGX Cloud**: A user using **AutoTrain Nvidia DGX Cloud** encountered a 500 Server Error, with advice to email logs to autotrain@hf.co for troubleshooting.

- **New benchmarks and dissatisfaction**: Members criticized existing coding benchmarks such as HumanEval as insufficient, and discussed newer benchmarks like **SWE Bench** and **MBPP+** for better evaluating LLM capabilities.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2401.15963">Paper page - NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional
  Correctness</a>: no description found</li><li><a href="https://osanseviero.github.io/hackerllama/blog/posts/llm_evals/#what-about-code">hackerllama - LLM Evals and Benchmarking</a>: Omar Sanseviero Personal Website</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.5-mixtral-8x7b">cognitivecomputations/dolphin-2.5-mixtral-8x7b · Hugging Face</a>: no description found</li><li><a href="https://x.com/noaroggendorff/status/1790485047306244415?s=46&t=m7jfctWh0zl_3Oj2DZJA9A">Tweet from Noa Roggendorff (@noaroggendorff)</a>: wait you guys are getting money? i owe like 500$ to @huggingface, 52 a month to @Adobe, 21 a month to @Google and my only income is a 5/week allowance I havent gotten in a year  Quoting Adrian Batista...</li><li><a href="https://huggingface.co/inference-endpoints/dedicated">Inference Endpoints - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">stabilityai/stable-diffusion-xl-base-1.0 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/agents">License to Call: Introducing Transformers Agents 2.0</a>: no description found</li><li><a href="https://huggingface.co/blog/mcpotato/hub-incident-post-mortem-20240422">2024-04-22 - Hub Incident Post Mortem</a>: no description found</li><li><a href="https://github.com/huggingface/diffusers/discussions/7953">Batched multilingual caption generation using PaliGemma 3B! · huggingface/diffusers · Discussion #7953</a>: Multilingual captioning with PaliGemma 3B Motivation The default code examples for the PaliGemma series I think are very fast, but limited. I wanted to see what these models were capable of, so I d...</li><li><a href="https://arstechnica.com/information-technology/2024/05/before-launching-gpt-4o-broke-records-on-chatbot-leaderboard-under-a-secret-name/">Before launching, GPT-4o broke records on chatbot leaderboard under a secret name</a>: Anonymous chatbot that mystified and frustrated experts was OpenAI&#39;s latest model.</li><li><a href="https://huggingface.co/collections/google/paligemma-release-6643a9ffbf57de2ae0448dda">PaliGemma Release - a google Collection</a>: no description found</li><li><a href="https://huggingface.co/collections/google/paligemma-ft-models-6643b03efb769dad650d2dda">PaliGemma FT Models - a google Collection</a>: no description found</li><li><a href="https://www.lamini.ai?">Lamini - Enterprises LLM Platform</a>: Lamini is the enterprise LLM platform for existing software teams to quickly develop and control their own LLMs. Lamini has built-in best practices for specializing LLMs on billions of proprietary doc...</li><li><a href="https://github.com/huggingface/transformers">GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1239908991354536018)** (3 messages): 

- **Prompt models with clear examples**: One member suggested prompting a model by providing clear examples of input and output in the system prompt. This approach was emphasized to improve model performance.

- **Learn about Exploration/Exploitation trade-off**: A user shared their learning about the **Exploration/Exploitation trade-off**, a fundamental concept in various decision-making algorithms.

- **Game of Life fascination**: Another member expressed their fascination with the **Game of Life**, encouraging others to share demos or videos. The enthusiasm suggests the community values practical showcases of this cellular automaton.
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1239882820311191623)** (9 messages🔥): 

- **New in Video Generation: Veo by DeepMind**: Veo is DeepMind's latest video generation model, producing high-quality, 1080p resolution videos that exceed a minute in length. It offers "unprecedented level of creative control" and will soon be available through Google's [VideoFX](https://labs.google/VideoFX) tool.
- **Hugging Face Daily Papers Resurrected**: Hugging Face has revived their Daily Papers, offering trending AI and ML papers delivered to your inbox. Users can subscribe to the service [here](https://huggingface.co/papers).
- **Rajesh's AI Journey Begins**: A fundamental AI article by Rajesh P. Kanaka is shared on LinkedIn, marking the start of his journey in the field. He is encouraged to republish it on HuggingFace's new [Blog Explorers](https://huggingface.co/blog-explorers) platform.
- **Authentic Hand Avatar on GitHub**: The official Pytorch implementation of the "Authentic Hand Avatar from a Phone Scan via Universal Hand Model" is now available on [GitHub](https://github.com/facebookresearch/UHM). This project is slated for presentation at CVPR 2024.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2310.08715">Toward Joint Language Modeling for Speech Units and Text</a>: Speech and text are two major forms of human language. The research community has been focusing on mapping speech to text or vice versa for many years. However, in the field of language modeling, very...</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>: no description found</li><li><a href="https://deepmind.google/technologies/veo/">Veo</a>: Veo is our most capable video generation model to date. It generates high-quality, 1080p resolution videos that can go beyond a minute, in a wide range of cinematic and visual styles.</li><li><a href="https://github.com/facebookresearch/UHM">GitHub - facebookresearch/UHM: Official PyTorch implementation of &quot;Authentic Hand Avatar from a Phone Scan via Universal Hand Model&quot;, CVPR 2024.</a>: Official PyTorch implementation of &quot;Authentic Hand Avatar from a Phone Scan via Universal Hand Model&quot;, CVPR 2024. - facebookresearch/UHM</li><li><a href="https://huggingface.co/blog-explorers">blog-explorers (Blog-explorers)</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1239933437951868988)** (12 messages🔥): 

- **700K Vietnamese dataset goes open-source**: A team announced the release of a 700,000 samples open-source dataset for Vietnamese language modeling. Check out the full dataset on [Hugging Face](https://huggingface.co/datasets/Vi-VLM/Vista).


- **New AI model OpenGPT-4o launched**: Features include text, text + image, and audio input with a variety of outputs, and it’s 100% free and super-fast. Accessible on [Hugging Face Spaces](https://huggingface.co/spaces/KingNish/GPT-4o) with future enhancements like video generation and better UI customization.

- **Filtering data to improve quality**: One member tested a new filtering method on datasets, noting it doesn’t work as a standalone but catches bad examples that other methods miss. *“It caught in a OCR’d books dataset that I hadn’t caught/cleaned yet.”*

- **AI mentor-mentee platform launch**: A new AI mentorship platform was launched to solve problems in connecting mentors and mentees in AI. Check it out and support on [Product Hunt](https://www.producthunt.com/posts/semis-from-reispar).

- **On-prem GPU cluster management simplified**: Introducing dstack for managing on-prem GPU clusters efficiently, integrating seamlessly with tools like Slurm. Full documentation and demos available at [dstack.ai](https://dstack.ai/docs).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/KingNish/GPT-4o">OpenGPT 4o - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://www.producthunt.com/posts/semis-from-reispar"> Semis from Reispar - Improving AI &amp; big tech knowledge gap | Product Hunt</a>: Semis from Reispar is a platform connecting aspiring and existing big tech and AI professionals with experienced mentors reducing the knowledge gap in the AI tech space across the world.</li><li><a href="https://github.com/bghira/SimpleTuner/blob/main/documentation/DREAMBOOTH.md#refiner-tuning)">SimpleTuner/documentation/DREAMBOOTH.md at main · bghira/SimpleTuner</a>: A general fine-tuning kit geared toward Stable Diffusion 2.1, DeepFloyd, and SDXL. - bghira/SimpleTuner</li><li><a href="https://huggingface.co/datasets/Vi-VLM/Vista?fbclid=IwZXh0bgNhZW0CMTEAAR2BXlXiqe6SjTjol1ViKCmI7HgogMPvrQU2pIBACQyZyI0av_ey8okihDA_aem_AdV1HiWxI6SngeQmTHG6XLs6v440zT5XTtTpW0yXlGkBFSQkIFrfY7nZyyMJXTF51eFvNHIwuPyArt-XQaSrGf0R">Vi-VLM/Vista · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1239869353894219877)** (11 messages🔥): 

- **AI Story Generation Papers Discussed**: A member shared plans to do a literature review on AI story generation, referencing the [Awesome Story Generation GitHub repo](https://github.com/yingpengma/Awesome-Story-Generation) and several papers including [this one](https://arxiv.org/abs/2212.04634). They later decided to focus on the [GROVE framework paper](https://arxiv.org/abs/2310.05388) for a comprehensive review.
- **Medium Write-Up Shared**: Upon completion, a presentation on AI for story generation was shared via a [Medium article](https://isamu-website.medium.com/understanding-ai-for-stories-d0c1cd7b7bdc).
- **Event Scheduled for Presentation**: There was a discussion about scheduling the presentation, leading to a decision to do it this Saturday. A member appreciated the scheduling and the placeholder event link was shared: [Discord Event](https://discord.gg/hugging-face-879548962464493619?event=1240255110093738026).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discord.gg/hugging-face-879548962464493619?event=1240255110093738026">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.</li><li><a href="https://arxiv.org/abs/2310.05388">GROVE: A Retrieval-augmented Complex Story Generation Framework with A Forest of Evidence</a>: Conditional story generation is significant in human-machine interaction, particularly in producing stories with complex plots. While Large language models (LLMs) perform well on multiple NLP tasks, i...</li><li><a href="https://github.com/yingpengma/Awesome-Story-Generation?tab=readme-ov-file.">GitHub - yingpengma/Awesome-Story-Generation: This repository collects an extensive list of awesome papers about Story Generation / Storytelling, primarily focusing on the era of Large Language Models (LLMs).</a>: This repository collects an extensive list of awesome papers about Story Generation / Storytelling, primarily focusing on the era of Large Language Models (LLMs). - yingpengma/Awesome-Story-Generation</li><li><a href="https://arxiv.org/abs/2212.04634">Open-world Story Generation with Structured Knowledge Enhancement: A Comprehensive Survey</a>: Storytelling and narrative are fundamental to human experience, intertwined with our social and cultural engagement. As such, researchers have long attempted to create systems that can generate storie...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1239859086409863271)** (8 messages🔥): 

- **Inviting users to Discord is prohibited**: A member reminded another that *"discord invites are against the <#895532661383254098>"*.
- **Image to Sales Model Challenges**: A member sought resources for training a model where an image serves as input and sales data as the output. Another member highlighted the complexity, emphasizing that the model depends heavily on the availability of relevant training data.
- **Training Data Availability**: A continuation of the previous discussion had a member asking if such training data is available. The original poster provided a dataset link from [HuggingFace](https://huggingface.co/datasets/tonyassi/sales1) noting previous work on sales prediction using image similarity.

**Link mentioned**: <a href="https://huggingface.co/datasets/tonyassi/sales1">tonyassi/sales1 · Datasets at Hugging Face</a>: no description found

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1240103986791452758)** (10 messages🔥): 

- **Starting with LangChain for a Job Assessment**: A member, chhabii, expressed needing help creating a chatbot using langchain after successfully creating a vector store. Another member, hitoriarchie, provided a [starter example](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/) for setting up local LLM and embedding models with Ollama.
- **Possible College Assignment Inquiry**: hitoriarchie asked chhabii if the task was a college assignment, to which chhabii clarified it was for a job assessment and requested further assistance.
- **Fine-Tuning Llama2 Locally**: Member uwaix. inquired about the process of fine-tuning Llama2 locally. There was no follow-up or response provided within the message history.

**Link mentioned**: <a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>: no description found

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1239984722181099650)** (11 messages🔥): 

- **Members seek examples for transformer agents**: One member asked for specific project examples using transformer agents. Another member pointed them to a [blog post](https://huggingface.co/blog/agents), but the inquirer had already read it and sought user experiences.

- **Connecting agents to diffusion models?**: Another noted the possibility of connecting transformer agents with diffusion models. This spurred brief interest, indicating some crossover potential between the technologies.

- **Error faced with load_image function**: A member encountered an "UnidentifiedImageError" while trying to load an image from a URL. They later found success using the `Image` module from PIL to load images from a local directory instead.

- **Request for chatbot to generate PowerPoints**: A member inquired about a chatbot capable of generating PowerPoint presentations with the OpenAI Assistant API. They wanted it to learn from previous presentations to modify slide content without altering the structure, also asking for recommendations of any suitable RAG or LLM models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/blog/agents">License to Call: Introducing Transformers Agents 2.0</a>: no description found
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1239834659052060723)** (282 messages🔥🔥): 

- **First LoRA Training Success**: A newbie shared excitement about their first successful LoRA training that took about 90 minutes. They promised to upload the final version to [Civitai](https://civit.ai/).
- **Creating Detailed Inpaint with Powerpaint**: Users discussed improving specific details in images, particularly eyes, using inpaint and reference photos. Powerpaint combined with brush commands significantly enhances fine details, but currently only works with version 1.5.
- **ComfyUI Workflows for Outpainting**: A member asked about expanding images with ComfyUI; another provided a [GitHub link](https://github.com/cubiq/ComfyUI_Workflows/blob/main/in-out_painting/README.md) to an easy-to-follow workflow for inpainting and outpainting within ComfyUI.
- **Google's Imagen 3 vs. Stable Diffusion**: Users expressed skepticism about Google's Imagen 3, highlighting concerns about accessibility and comparing it to Sora and GPT-4o. The discussion concluded that SD3 and its finetunes offer better usability.
- **GPU Recommendations for AI Tasks**: Frequent discussions on GPU choices for AI tasks, with an emphasis on the importance of VRAM for future-proofing. Recommendations to wait for the 50xx series GPUs in November for better pricing.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://muse-model.github.io/">Muse: Text-To-Image Generation via Masked Generative Transformers</a>: no description found</li><li><a href="https://huggingface.co/spaces/multimodalart/HunyuanDiT">HunyuanDiT - a Hugging Face Space by multimodalart</a>: no description found</li><li><a href="https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html">SD WebUI Benchmark Data</a>: no description found</li><li><a href="https://github.com/cubiq/ComfyUI_Workflows/blob/main/in-out_painting/README.md">ComfyUI_Workflows/in-out_painting/README.md at main · cubiq/ComfyUI_Workflows</a>: A repository of well documented easy to follow workflows for ComfyUI - cubiq/ComfyUI_Workflows</li><li><a href="https://altacc21294.wixsite.com/hightechcitysmp">Home | Hightechsmp</a>: no description found</li><li><a href="https://universebox.pages.dev/">Universe Box</a>: no description found
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1239890486387281920)** (5 messages): 

- **OpenRouter drops a slew of new models**: The OpenRouter platform announced multiple new models including [DeepSeek-v2 Chat](https://openrouter.ai/models/deepseek/deepseek-chat) and [DeepSeek Coder](https://openrouter.ai/models/deepseek/deepseek-coder). Other models include [Llama Guard 2 8B](https://openrouter.ai/models/meta-llama/llama-guard-2-8b) and [Llama 3 70B Base](https://openrouter.ai/models/meta-llama/llama-3-70b).
- **Google releases Gemini Flash 1.5**: A new multimodal model, [Gemini Flash 1.5](https://openrouter.ai/models/google/gemini-flash-1.5), has been added to OpenRouter's offerings.
- **Perplexity introduces Llama3-based Sonar models**: [New models from Perplexity](https://docs.perplexity.ai/changelog/new-models-llama-3-sonar-family) include [Llama3 Sonar 8B](https://openrouter.ai/models/perplexity/llama-3-sonar-small-32k-chat) and their 70B online counterparts. Older models have been deprecated and redirected to these new variants.
- **DeepSeek requires logging**: Users must enable logging in [Settings](https://openrouter.ai/settings#analytics) to use models from DeepSeek as the platform logs and trains on user data.
- **WizardLM-2 8x22B Nitro removed**: The model has been discontinued because no provider could maintain throughput above 100tps with quality standards.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-flash-1.5)">Google: Gemini Flash 1.5 (preview) by google | OpenRouter</a>: Gemini 1.5 Flash is a foundation model that performs well at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3-sonar-small-32k-chat>)">Perplexity: Llama3 Sonar 8B by perplexity | OpenRouter</a>: Llama3 Sonar is Perplexity&#x27;s latest model family. It surpasses their earlier Sonar models in cost-efficiency, speed, and performance.  This is a normal offline LLM, but the [online version](/mode...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3-sonar-small-32k-online>)">Perplexity: Llama3 Sonar 8B Online by perplexity | OpenRouter</a>: Llama3 Sonar is Perplexity&#x27;s latest model family. It surpasses their earlier Sonar models in cost-efficiency, speed, and performance.  This is the online version of the [offline chat model](/mode...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3-sonar-large-32k-chat>)">Perplexity: Llama3 Sonar 70B by perplexity | OpenRouter</a>: Llama3 Sonar is Perplexity&#x27;s latest model family. It surpasses their earlier Sonar models in cost-efficiency, speed, and performance.  This is a normal offline LLM, but the [online version](/mode...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3-sonar-large-32k-online)">Perplexity: Llama3 Sonar 70B Online by perplexity | OpenRouter</a>: Llama3 Sonar is Perplexity&#x27;s latest model family. It surpasses their earlier Sonar models in cost-efficiency, speed, and performance.  This is the online version of the [offline chat model](/mode...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat>)">DeepSeek-V2 Chat by deepseek | OpenRouter</a>: DeepSeek-V2 Chat is a conversational finetune of DeepSeek-V2, a Mixture-of-Experts (MoE) language model. It comprises 236B total parameters, of which 21B are activated for each token.  Compared with D...</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-coder>)">Deepseek Coder by deepseek | OpenRouter</a>: Deepseek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese.  The model ...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-guard-2-8b>)">Meta: LlamaGuard 2 8B by meta-llama | OpenRouter</a>: This safeguard model has 8B parameters and is based on the Llama 3 family. Just like is predecessor, [LlamaGuard 1](https://huggingface.co/meta-llama/LlamaGuard-7b), it can do both prompt and response...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-70b>)">Meta: Llama 3 70B by meta-llama | OpenRouter</a>: Meta&#x27;s latest class of model (Llama 3) launched with a variety of sizes &amp; flavors. This is the base 70B pre-trained version.  It has demonstrated strong performance compared to leading closed...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3-8b>)">Meta: Llama 3 8B by meta-llama | OpenRouter</a>: Meta&#x27;s latest class of model (Llama 3) launched with a variety of sizes &amp; flavors. This is the base 8B pre-trained version.  It has demonstrated strong performance compared to leading closed-...</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-2024-05-13>)">OpenAI: GPT-4o by openai | OpenRouter</a>: GPT-4o (&quot;o&quot; for &quot;omni&quot;) is OpenAI&#x27;s latest AI model, supporting both text and image inputs with text outputs. It maintains the intelligence level of [GPT-4 Turbo](/models/open...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

obiefernandez: I signed up but it's not clear what the unique value proposition is
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1239861093585915934)** (200 messages🔥🔥): 

- **Crypto Balance Delay Clarified**: A member inquired about delays in crypto balance confirmation, with clarification provided that it's due to required network confirmations by Coinbase, such as 128 blocks for Polygon and 85 for Ethereum.

- **Tool for Exploring OpenRouter Models**: A user shared their tool for exploring and sorting the OpenRouter model list via API, which updates the list hourly. They provided a [GitHub link](https://github.com/fry69/orw) for those interested in contributing.

- **GPT-4o Versions Explained**: Members discussed the differences between GPT-4o versions, with clarifications that there is no difference currently, but the options exist for future version control.

- **WizardLM 8x22B Nitro Removed**: WizardLM 8x22B Nitro was removed due to providers falling below the 100 tokens/sec threshold, with requests redirected to the standard variant. Some users expressed frustration about constant model changes.

- **Google's Gemini Event**: Reactions to Google's event unveiling the Gemini 1.5 models were mixed, with some users finding it less exciting compared to OpenAI's recent event. The new models include Gemini 1.5 Flash and TPUv6 announcements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.litellm.ai/">LiteLLM</a>: LiteLLM handles loadbalancing, fallbacks and spend tracking across 100+ LLMs. all in the OpenAI format</li><li><a href="https://huggingface.co/Salesforce/SFR-Iterative-DPO-LLaMA-3-8B-R">Salesforce/SFR-Iterative-DPO-LLaMA-3-8B-R · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mlabonne/Meta-Llama-3-120B-Instruct">mlabonne/Meta-Llama-3-120B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/fry69/orw">GitHub - fry69/orw: Watch for changes in OpenRouter models API and store changes in a SQLite database. Includes a simple web interface.</a>: Watch for changes in OpenRouter models API and store changes in a SQLite database. Includes a simple web interface. - fry69/orw</li><li><a href="https://orw.karleo.net/list">OpenRouter API Watcher</a>: OpenRouter API Watcher monitors changes in OpenRouter models and stores those changes in a SQLite database. It queries the model list via the API every hour.</li><li><a href="https://orw.karleo.net/removed">OpenRouter API Watcher</a>: OpenRouter API Watcher monitors changes in OpenRouter models and stores those changes in a SQLite database. It queries the model list via the API every hour.
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1239895427336372284)** (41 messages🔥): 

- **Mojo Runs MLIR Natively with Minor Syntax Adjustments**: Members discussed how **Mojo** can run **MLIR** natively with just a bit of extra syntax. One user shared a [link](https://docs.modular.com/mojo/notebooks/BoolMLIR) explaining the advantage of Mojo's access to low-level MLIR features.
- **Mojo to Have Python Dependency Alternatives**: Conversations in the thread proposed scenarios where the whole Mojo toolchain could work without Python. A related [GitHub issue](https://github.com/modularml/mojo/issues/935) was cited for tracking this feature request.
- **Strategies for Learning Mojo**: New member seeking advice on learning Mojo was directed to the [Mojo SDK manual](https://docs.modular.com/mojo/manual/get-started/) and other helpful resources like the [Mandelbrot notebook](https://docs.modular.com/mojo/notebooks/Mandelbrot).
- **Advocacy for Mojo in GPU Market**: Users debated the portability advantages of **Mojo** over **CUDA**, highlighting that Mojo’s GPU code portability could create a more competitive hardware market. It was noted that CUDA's vendor lock-in currently benefits Nvidia, but Mojo's cross-vendor potential is promising.
- **Community Spirit Encouraged**: Members discussed the importance of community discussion and advocacy to promote Mojo. They also noted the long adoption timescale for new programming languages, suggesting that early discussions can help refine and popularize Mojo.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/get-started/">Get started with Mojo🔥 | Modular Docs</a>: Get the Mojo SDK or try coding in the Mojo Playground.</li><li><a href="https://docs.modular.com/mojo/notebooks/BoolMLIR">Low-level IR in Mojo | Modular Docs</a>: Learn how to use low-level primitives to define your own boolean type in Mojo.</li><li><a href="https://docs.modular.com/mojo/notebooks/Mandelbrot">Mandelbrot in Mojo with Python plots | Modular Docs</a>: Learn how to write high-performance Mojo code and import Python packages.</li><li><a href="https://docs.modular.com/mojo/manual/basics">Introduction to Mojo | Modular Docs</a>: Introduction to Mojo&#x27;s basic language features.</li><li><a href="https://github.com/modularml/mojo/issues/935">[Feature Request] binary build via `mojo build` could not run directly on other os · Issue #935 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? Hi, I tried to build a simple mojo app which use numpy...</li><li><a href="https://modul.ar/community-meeting">Google Calendar - Sign in to Access &amp; Edit Your Schedule</a>: no description found
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1240005922076758159)** (2 messages): 

- **Modular shares an update on Twitter**: [Modular tweeted](https://twitter.com/Modular/status/1790442405273161922) an update, presumably about their ongoing projects or announcements related to their platform. No further details were provided in the shared link.
- **Another Modular tweet shared**: [Modular posted](https://twitter.com/Modular/status/1790774045581152561) another tweet, likely discussing recent developments or community news. Specific content of the tweet was not disclosed in the shared link.
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1239985859370160190)** (3 messages): 

- **Mojo🔥 Nightly Build Video Released**: [Modular's new video](https://www.youtube.com/watch?v=arZS5-plt2Q) discusses the **Mojo🔥 nightly build** and nightly Visual Studio Code extension. Modular engineer Brian Gesiak covers the new branch called Nightly, in sync with the Mojo nightly build.
- **Open-Source Mojo🔥 Standard Library Contributions**: [A video by Modular](https://www.youtube.com/watch?v=TJpFSSIts5Q) explains how to contribute to the **open-source Mojo🔥 standard library**. Modular engineer Joe Loser provides guidance on getting started with contributing using Mojo.
- **Introduction to MAX Graph API and Custom Operators**: [Modular's latest video](https://www.youtube.com/watch?v=nkWhnFNlguQ) explores the **MAX Graph API** for building AI inference pipelines in Mojo. Ehsan Kermani explains how to begin with MAX Graph and custom operators in Mojo.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=arZS5-plt2Q">Using Mojo🔥 nightly build and nightly Visual Studio Code extension</a>: Mojo🔥 public repo now has a new branch called Nightly, which is in sync with the Mojo nightly build. In this video, Modular engineer Brian Gesiak discusses ...</li><li><a href="https://www.youtube.com/watch?v=TJpFSSIts5Q">Contributing to Open-Source Mojo🔥 Standard Library</a>: Mojo🔥 standard library is now open-source. In this video Modular engineer Joe Loser discusses how you can get started with contributing to Mojo🔥 using Mojo...</li><li><a href="https://www.youtube.com/watch?v=nkWhnFNlguQ">Introduction to MAX Graph API and custom operators</a>: The MAX Graph API allows you to build your entire AI inference pipeline in Mojo. In this video Ehsan Kermani discusses how you can get started with MAX Graph...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/)** (1 messages): 

Zapier: Modular: MAX Graph API Tutorial
https://www.modular.com/blog/max-graph-api-tutorial
  

---


**Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1240007747890708540)** (1 messages): 

- **Join the Mojo Community Meeting!**: The Mojo team is hosting a community meeting for developers, contributors, and users on Monday, May 20, 10-11 am via [Zoom](https://modul.ar/community-meeting-zoom). The meeting will share future plans for Mojo and discuss upcoming events; details can also be added to your calendar through this [link](https://modul.ar/community-meeting).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modul.ar/community-meeting-zoom.">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://modul.ar/community-meeting.">Google Calendar - Sign in to Access &amp; Edit Your Schedule</a>: no description found</li><li><a href="https://modul.ar/community-meeting-doc">[Public] Mojo Community Meeting</a>: no description found
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1239838353806069790)** (120 messages🔥🔥): 

- **Calling C/C++ Libraries from Mojo**: Members discussed the possibility of calling C/C++ libraries from Mojo. It was confirmed that it is possible by using the ffi and external_call [tweetorial](https://github.com/modularml/devrel-extras/tree/main/tweetorials/ffi).

- **Convert String to Float Function Added**: A member shared that they created a PR to add a method `atof()` to `String` for converting strings to floats in Mojo. The PR can be viewed [here](https://github.com/modularml/mojo/pull/2649).

- **Mojo Compatibility with CLion**: There was an inquiry about using Mojo with CLion, and a link to the required plugin was shared [here](https://plugins.jetbrains.com/plugin/23371-mojo).

- **Creating HTTP Clients in Mojo**: For creating HTTP clients, it was suggested to use the [lightbug_http](https://github.com/saviorand/lightbug_http) framework as the Mojo docs are currently missing an HTTP module.

- **Python Interoperability Issue**: A member shared a Python interoperability problem that was resolved after updating to the latest version of Mojo nightly. The initial issue description and subsequent resolution discussion can be accessed [here](https://discord.com/channels/1087530497313357884/1151418579913277450/1239755238924095579).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/values/value-semantics#">Value semantics | Modular Docs</a>: An explanation of Mojo&#x27;s value-semantic defaults.</li><li><a href="https://docs.modular.com/mojo/manual/values/value-semantics#python-style-reference-semantics">Value semantics | Modular Docs</a>: An explanation of Mojo&#x27;s value-semantic defaults.</li><li><a href="https://plugins.jetbrains.com/plugin/23371-mojo">Mojo - IntelliJ IDEs Plugin | Marketplace</a>: Provides basic editing for Mojo programming language: syntax checks and highlighting, commenting and formatting. New features will be added in the future, please feel...</li><li><a href="https://github.com/modularml/devrel-extras/tree/main/tweetorials/ffi">devrel-extras/tweetorials/ffi at main · modularml/devrel-extras</a>: Contains supporting materials for developer relations blog posts, videos, and workshops - modularml/devrel-extras</li><li><a href="https://github.com/modularml/mojo/issues/2653">[BUG] NumPy array in-place operation over a copied object reference does not modify the original Python object in-place · Issue #2653 · modularml/mojo</a>: Bug description from python import Python def do_numpy_stuff(ar: PythonObject) -&gt; PythonObject: ar.__iadd__(3) print(&quot;inside function:\n&quot;, ar) return ar fn main() raises: var np = Python....</li><li><a href="https://github.com/saviorand/lightbug_http">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/pull/2649">[stdlib] Add method `atof()` to `String`  by fknfilewalker · Pull Request #2649 · modularml/mojo</a>: This PR adds a function that can convert a String to a Float64.  Right now it is implemented just for Float64 but maybe we should add other precisions? This supports the following notations: &quot;-12...</li><li><a href="https://a.co/d/6dK6Xzl">no title found</a>: no description found</li><li><a href="https://ivellapillil.github.io/mojo">Learn Mojo Programming Language</a>: no description found</li><li><a href="https://github.com/modularml/mojo/blob/bf73717d79fbb79b4b2bf586b3a40072308b6184/stdlib/src/builtin/tuple.mojo#L100>">mojo/stdlib/src/builtin/tuple.mojo at bf73717d79fbb79b4b2bf586b3a40072308b6184 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/carlca/ca_mojo.git">GitHub - carlca/ca_mojo</a>: Contribute to carlca/ca_mojo development by creating an account on GitHub.</li><li><a href="https://github.com/dimitrilw/toybox/issues/9>">Issues · dimitrilw/toybox</a>: Various data-structures and other toys implemented in Mojo🔥. - Issues · dimitrilw/toybox
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1239996921435918576)** (17 messages🔥): 

- **Modular Bot Notifications in the Works**: Members discussed the idea of having the Modular bot notify them when nightly builds drop. A developer hinted this might be organized using GitHub Actions.

- **Nightly Builds and Weekly Updates**: A suggestion was made to have nightly builds include only non-compiler changes and reserve compiler-changing updates for a weekly build. No further consensus reached on this yet.

- **New Nightly Mojo Compiler Released**: The latest nightly release `2024.5.1515` is available and can be updated via `modular update nightly/mojo`. [Link to changes since the last release](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

- **Potential Self-test Failures on macOS**: Today’s nightly release may have non-deterministic self-test failures on mac due to an LLDB initialization issue, which is currently being investigated. Users are informed about the ongoing issue.

- **Exciting Commits Highlighted**: Two noteworthy commits were shared: one on making `Tuple`'s constructor move its input elements and another changing `Reference.is_mutable` to `Bool`. [Tuple constructor change](https://github.com/modularml/mojo/commit/f05749db22548f61c17e6cb0db938e22f092341f) and [Reference.is_mutable change](https://github.com/modularml/mojo/commit/09db8f3bcb0d783078a63241cda57047a850e05e).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/commit/f05749db22548f61c17e6cb0db938e22f092341f">[mojo-stdlib] Make `Tuple`&#39;s constructor move its input elements. (#3… · modularml/mojo@f05749d</a>: …9904)  This changes `Tuple` to take its input pack as &amp;#39;owned&amp;#39; and then move from the pack into it storage.  This unearthed some bugs handling owned packs which were causing multiple d...</li><li><a href="https://github.com/modularml/mojo/commit/09db8f3bcb0d783078a63241cda57047a850e05e">[stdlib] Change `Reference.is_mutable` to `Bool` (from `i1`) · modularml/mojo@09db8f3</a>: With the recent change to `Bool` to use an `i1` as its representation, many of the errors holding up moving `Reference.is_mutable` to `Bool` were resolved.  Co-authored-by: Chris Lattner &amp;lt;clatt...
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1239968856722636962)** (23 messages🔥): 

- **Austin introduces himself from AWS GenAI Innovation Center**: Austin joined the chat to explore open-source research opportunities and received a warm welcome, with advice to check out [other channels](https://discord.com/channels/747850033994662000) for more resources.
- **Comparison between epinet and JARVIS character**: Members debated why the "Her" character is more compelling for AGI development compared to "JARVIS." One member pointed out, *"The mass market wants to f* 'HER' more than they want to f* JARVIS."*
- **Discussion on epinet's effectiveness and limitations**: The group explored the residual effects and practicality of epinets as a method to simulate an ensemble model to estimate epistemic uncertainty. Despite liking epinets, one pointed out, *"it's not as concretely grounded as other Bayesian methods of determining uncertainty."*
- **Concerns over neural network stability with epinets**: A detailed explanation was provided regarding how neural networks can be tricky when adding an epinet to the original model, potentially leading to inaccurate uncertainty predictions. *"You could have the predicted variance be low because the model is sure what it's doing is correct... high despite the model being unsure something is correct."*
- **Confusion about residuals and their impact on epinets**: The conversation included speculation on the necessity of residuals for the epinet to adjust the base model outputs more effectively. One member suggested, *"the residual just makes it easier to train in that sense."*
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1239940002998255666)** (79 messages🔥🔥): 

- **Discussion on Activation Functions for Model Convergence**: Members debated the requirements for an activation function to ensure good convergence, emphasizing the need for it to be nonlinear. One member shared a concern about parametrizing between two functions in a pre-trained model without making mistakes.

- **Fine-Tuning Conversations around FlashInfer and FA2**: Detailed discussions ensued about different methodologies in AI models such as FlashInfer and FA2. Clarifications were provided regarding the splitting across K sequence lengths and the inclusion of a reduction step in FA2.

- **Insight on the Dot Product in Neural Networks**: Members shared thoughts on the simplifications and robustness of dot products in neural networks, with one linking to an [article](https://archive.is/GfliU) to expand on Fourier transforms and their cognitive dissonance implications.

- **Introduction to the Sakuga-42M Dataset**: A new large-scale cartoon animation dataset named Sakuga-42M was introduced, comprising 42 million keyframes aimed to tackle biases present in models trained on natural videos. A corresponding [arXiv link](https://arxiv.org/abs/2405.07425) was shared for further reading.

- **Visual Question Answering and Specialized Models**: Members discussed the limitations and applications of visual question answering (VQA) models and multimodal models like BLIP3, referring to resources on [Hugging Face](https://huggingface.co/tasks/visual-question-answering) for specific use-cases like aiding the visually impaired and image retrieval.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.07518">SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts</a>: Monolithic large language models (LLMs) like GPT-4 have paved the way for modern generative AI applications. Training, serving, and maintaining monolithic LLMs at scale, however, remains prohibitively...</li><li><a href="https://arxiv.org/abs/2405.08707">Beyond Scaling Laws: Understanding Transformer Performance with Associative Memory</a>: Increasing the size of a Transformer model does not always lead to enhanced performance. This phenomenon cannot be explained by the empirical scaling laws. Furthermore, improved generalization ability...</li><li><a href="https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1">Salesforce/xgen-mm-phi3-mini-instruct-r-v1 · Hugging Face</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.07425">Sakuga-42M Dataset: Scaling Up Cartoon Research</a>: Hand-drawn cartoon animation employs sketches and flat-color segments to create the illusion of motion. While recent advancements like CLIP, SVD, and Sora show impressive results in understanding and ...</li><li><a href="https://arxiv.org/abs/2405.08644">Thinking Tokens for Language Modeling</a>: How much is 56 times 37? Language models often make mistakes in these types of difficult calculations. This is usually explained by their inability to perform complex reasoning. Since language models ...</li><li><a href="https://huggingface.co/tasks/visual-question-answering">What is Visual Question Answering? - Hugging Face</a>: no description found</li><li><a href="http://arxiv.org/abs/2405.08553">Improving Transformers with Dynamically Composable Multi-Head Attention</a>: Multi-Head Attention (MHA) is a key component of Transformer. In MHA, attention heads work independently, causing problems such as low-rank bottleneck of attention score matrices and head redundancy. ...</li><li><a href="https://github.com/caiyun-ai/dcformer">GitHub - Caiyun-AI/DCFormer</a>: Contribute to Caiyun-AI/DCFormer development by creating an account on GitHub.</li><li><a href="http://arxiv.org/abs/2112.00114">Show Your Work: Scratchpads for Intermediate Computation with Language Models</a>: Large pre-trained language models perform remarkably well on tasks that can be done &#34;in one pass&#34;, such as generating realistic text or synthesizing computer programs. However, they struggle w...</li><li><a href="http://arxiv.org/abs/2305.00833">Learning to Reason and Memorize with Self-Notes</a>: Large language models have been shown to struggle with multi-step reasoning, and do not retain previous reasoning steps for future use. We propose a simple method for solving both of these problems by...</li><li><a href="http://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is imp...</li><li><a href="https://archive.is/GfliU">The Power of the Dot Product in Artificial Intelligence | by Manuel B&#x2026;</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1239948993925222444)** (45 messages🔥): 

- **Mimetic initialization could boost Transformer training**: A member shared a [paper](https://proceedings.mlr.press/v202/trockman23a/trockman23a.pdf) showing that mimicking weight patterns from pre-trained Transformers (coined "mimetic initialization") can significantly improve training efficiency and accuracy on small datasets like CIFAR-10 and ImageNet, achieving over 5% and 4% gains in accuracy, respectively. This involves initializing query and key weights to approximate the identity matrix, and value and projection weights to approximate the negative identity matrix.
- **Hypernetworks for initialization discussed**: A dialogue on utilizing hypernetworks for weight initialization concluded that while it’s essentially meta-learning, finding a low-dimensional, simple symbolic initialization could be beneficial. One member pondered symbolic regression to derive and reverse engineer new initialization principles.
- **Reflections on Minsky's impact on neural networks**: The conversation highlighted a sentiment that Marvin Minsky receives excessive credit for derailing interest in neural networks during their early failures, despite his own background in neural nets. There’s an argument that Minsky's influence wouldn't be as notable if neural networks had succeeded from the start.
- **The challenges of small dataset training**: Real-world struggles with training Transformers on small datasets were shared, with some members expressing ignorance of this being an issue until now, hinting at a mental unawareness potentially lowering barriers to problem-solving.
- **Idea-sharing and community projects proposed**: Members discussed the abundance of ideas but lack of time to pursue them. There’s a call to create an "idea-dump" channel for community-driven projects, reflecting a common bottleneck of limited manpower but much enthusiasm.

**Link mentioned**: <a href="https://arxiv.org/abs/2210.03651">Understanding the Covariance Structure of Convolutional Filters</a>: Neural network weights are typically initialized at random from univariate distributions, controlling just the variance of individual weights even in highly-structured operations like convolutions. Re...

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

ocg6377: I might also be interested in helping, depending on what's needed
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1240087798841606187)** (5 messages): 

- **Harness Processes Multiple Choice per Token Call**: Members discussed how a single multiple-choice question in MMLU results in one request per answer (A, B, C, D), even though they're just one token. One member confirmed that the harness indeed processes each answer option through a single call as shown in [the harness code](https://github.com/EleutherAI/lm-evaluation-harness/blob/a9eaaf46f1e246e5ce090e37f2f99fe1cfe5a919/lm_eval/models/utils.py#L485).
- **Export Multiple Choice Answers for Accuracy Analysis**: A user inquired about exporting individual answers from multiple-choice questions to discern if the model got them correct or incorrect. This would facilitate a comparison between distributions of correct/incorrect responses.

**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/a9eaaf46f1e246e5ce090e37f2f99fe1cfe5a919/lm_eval/models/utils.py#L485)">lm-evaluation-harness/lm_eval/models/utils.py at a9eaaf46f1e246e5ce090e37f2f99fe1cfe5a919 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1240176018799464478)** (3 messages): 

- **Thunder Kittens creators' meeting request**: A member inquired about the possibility of having the **Thunder Kittens creators** present in **CUDA MODE**. Another member responded positively, saying, *“I can ask.”*
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1239868744906575992)** (19 messages🔥): 

- **CuSPARSE Vector Reuse Unclear**: A member questioned the cost and reusability of `cusparseCreateDnVec` calls, expressing confusion over the documentation's lack of detail on memory allocation. They inquired if only the descriptor memory is affected or if vector data is cached elsewhere.

- **Trouble with clangd and CUDA Files**: A member struggled with making clangd parse `.cu` files correctly despite having a `compile_commands.json` file and using both VSCode and Neovim. They noted it worked previously with the CUDA Toolkit but faced issues after switching to NVHPC.

- **Torch Tensor Accessor Discussion**: A member sought advice on whether to use accessors for Torch tensors in C++ as documented or pass `tensor.data_ptr` directly to the kernel, raising a question about using an unsigned char pointer for these tensors. They requested more documentation on this topic.

- **Dot Product Puzzle Floating Point Issue**: A member encountered a floating point overflow error in the naive implementation of a dot product for large arrays, which differed in results when using reduction methods. Another explained that the issue ties back to FP32 precision limitations, and suggested merging their kernel code for better accuracy.

- **Improving L2 Cache Hit Rate**: A suggestion was made regarding improving L2 cache hit rates, with a reference to a Triton lecture and a [Triton Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). The tutorial covers block-level matrix multiplications, pointer arithmetic, and program reordering for better cache performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html">Matrix Multiplication &mdash; Triton  documentation</a>: no description found</li><li><a href="https://github.com/NVIDIA/cccl/blob/main/.clangd">cccl/.clangd at main · NVIDIA/cccl</a>: CUDA C++ Core Libraries. Contribute to NVIDIA/cccl development by creating an account on GitHub.</li><li><a href="https://github.com/srush/GPU-Puzzles/tree/main?tab=readme-ov-file#puzzle-10---dot-product).">GitHub - srush/GPU-Puzzles: Solve puzzles. Learn CUDA.</a>: Solve puzzles. Learn CUDA. Contribute to srush/GPU-Puzzles development by creating an account on GitHub.</li><li><a href="https://gist.github.com/chetandhembre/6e93a652026f0bb669c981513e2cc5b8">puzzle10_dotproduct floating point overflow error</a>: puzzle10_dotproduct floating point overflow error. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1239840938080342057)** (39 messages🔥): 

- **Check your tensor allocations when using `torch.compile`**: A member suggested replacing dynamically allocated tensors, as seen in `torch.cat` implementations, with pre-allocated tensors to improve performance ([example here](https://github.com/openai/whisper/blob/main/whisper/model.py#L301)).
  
- **Using `torch.compile` with Triton kernels**: A member asked for advice on creating a network graph with triton kernels. It was suggested to "create a custom op and wrap those with torch.compile," linking a detailed discussion for further assistance.

- **Dealing with DeepSpeed compatibility issues**: A user questioned whether DeepSpeed's latest release is compatible with `torch.compile`. A member referenced a [PR on GitHub](https://github.com/microsoft/DeepSpeed/pull/4878) emphasizing that a compile flag should be added to the DeepSpeed config.

- **Optimize custom operations with `torch.compile`**: A detailed discussion unfolded on the benefits and challenges of using `torch.compile` with custom operations and external libraries. Recommendations included using `torch._dynamo` decorators to properly trace and debug tensor issues during compilation ([example project](https://version.aalto.fi/gitlab/AaltoRSE/xmc-sparse-pytorch/-/blob/v043/src/python/bindings.py)).

- **Static vs dynamic tensor allocation in `torch.compile`**: A member explained why dynamically allocated tensors can hurt performance, particularly with `torch.cat`, as reallocating and copying the cache is expensive. The importance of static cache and techniques to reduce overhead were highlighted ([blog link here](https://pytorch.org/blog/accelerating-generative-ai-2/)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/accelerating-generative-ai-2/">Accelerating Generative AI with PyTorch II: GPT, Fast</a>: This post is the second part of a multi-series blog focused on how to accelerate generative AI models with pure, native PyTorch. We are excited to share a breadth of newly released PyTorch performance...</li><li><a href="https://pytorch.org/blog/introducing-depyf">Introducing depyf: mastering torch.compile with ease</a>:   </li><li><a href="https://github.com/zchee/cuda-sample/blob/master/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp#L218">cuda-sample/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp at master · zchee/cuda-sample</a>: CUDA official sample codes. Contribute to zchee/cuda-sample development by creating an account on GitHub.</li><li><a href="https://version.aalto.fi/gitlab/AaltoRSE/xmc-sparse-pytorch/-/blob/v043/src/python/bindings.py?ref_type=heads#L50">src/python/bindings.py · v043 · AaltoRSE / XMC Sparse PyTorch · GitLab</a>: Aalto Version Control System</li><li><a href="https://github.com/openai/whisper/blob/main/whisper/model.py#L301)">whisper/whisper/model.py at main · openai/whisper</a>: Robust Speech Recognition via Large-Scale Weak Supervision - openai/whisper</li><li><a href="https://github.com/microsoft/DeepSpeed/pull/4878">Enable torch.compile with ZeRO (Experimental) by tohtana · Pull Request #4878 · microsoft/DeepSpeed</a>: This PR enables torch.compile with ZeRO stages 1/2/3. You need to add compile section in your DeepSpeed config. The fields in the section are passed to torch.compile.   &quot;compile&quot;: {     &quo...</li><li><a href="https://pastebin.com/XHwFwDLx">compile problem - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://version.aalto.fi/gitlab/AaltoRSE/xmc-sparse-pytorch/-/jobs/86868">manylinux-cu121: [cp310, 2.3] (#86868) · Jobs · AaltoRSE / XMC Sparse PyTorch · GitLab</a>: Aalto Version Control System</li><li><a href="https://version.aalto.fi/gitlab/AaltoRSE/xmc-sparse-pytorch/-/blob/v043/src/python/ops.py?ref_type=heads#L41">src/python/ops.py · v043 · AaltoRSE / XMC Sparse PyTorch · GitLab</a>: Aalto Version Control System</li><li><a href="https://github.com/pytorch/ao/pull/184/files#diff-3444226e1dc5947e486c918c8d57b8742bbcd9af6b4f5a599e0443b08bd7164aR222">[wip] fast semi-sparse sparse training  by jcaip · Pull Request #184 · pytorch/ao</a>: So was testing this on HuggingFace BERT, wasn&amp;#39;t seeing speedups - it&amp;#39;s because i was bottlenecked by a bunch of other stuff. (bf16, compile, adamw, dataloader, batchsize) bf16 + compil...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1239840680256213022)** (3 messages): 

- **Beginner seeks advice on custom CUDA kernels in PyTorch**: A beginner requested resources for learning how to use custom **CUDA kernels** in **PyTorch**.
- **Helpful response provides lecture link**: A member shared a link to a [YouTube lecture by Jeremy](https://youtu.be/4sgKnKbR-WE) titled *"Lecture 3: Getting Started With CUDA for Python Programmers"*, which demonstrates how to write custom CUDA kernels and use them in PyTorch.

**Link mentioned**: <a href="https://youtu.be/4sgKnKbR-WE?si=00-k8KV5ESxqks3h">Lecture 3: Getting Started With CUDA for Python Programmers</a>: Recording on Jeremy&#39;s YouTube https://www.youtube.com/watch?v=nOxKexn3iBoSupplementary Content: https://github.com/cuda-mode/lecture2/tree/main/lecture3Speak...

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1239966431060295712)** (4 messages): 

- **Mobile issue causes user confusion**: A user mentioned experiencing a "weird issue" on mobile, but stated that it appears "fine now." Another user reported that the issue also occurs on PC and mobile, confirming it was not resolved.
- **Event link troubleshooting continues**: Users discussed a hyperlink to a Discord event and whether accessing the event via the "events tab" resolves the problem. Upon checking, a user confirmed that accessing the link through the events tab works.
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1240353611956097074)** (3 messages): 

- **Check out the cloud on Twitter**: A member shared a [link to cloud (@cloud11665) on X](https://twitter.com/cloud11665/status/1790776040681271583). This link was provided in a message without much context.
- **NVIDIA GPU Programming Guide**: Another member linked to the [NVIDIA GPU Programming Guide](https://download.nvidia.com/developer/GPU_Programming_Guide/GPU_Programming_Guide.pdf), offering a resource for those interested in GPU programming.
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1240324433667686453)** (1 messages): 

- **CUDA Puzzle 10 Solves and Issues**: A member shared their solution to [CUDA Puzzle 10 - Dot Product](https://github.com/srush/GPU-Puzzles/tree/main?tab=readme-ov-file#puzzle-10---dot-product) using both naive and reduction methods. They encountered a floating-point overflow error in the naive implementation with the output `16777216` instead of `20480000` for arrays of size **20480000**, while the reduction method worked correctly. They are seeking an explanation for this behavior. You can review their code [here](https://gist.github.com/chetandhembre/6e93a652026f0bb669c981513e2cc5b8).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/srush/GPU-Puzzles/tree/main?tab=readme-ov-file#puzzle-10---dot-product).">GitHub - srush/GPU-Puzzles: Solve puzzles. Learn CUDA.</a>: Solve puzzles. Learn CUDA. Contribute to srush/GPU-Puzzles development by creating an account on GitHub.</li><li><a href="https://gist.github.com/chetandhembre/6e93a652026f0bb669c981513e2cc5b8">puzzle10_dotproduct floating point overflow error</a>: puzzle10_dotproduct floating point overflow error. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1239877561010618419)** (65 messages🔥🔥): 

- **Gradient Accumulation and `layernorm_backward_kernel8` Fix**: Discussions about a fresh look at gradient accumulation and verifying kernel calculations identified issues with the `layernorm_backward_kernel8` not accumulating biases and weights. This was addressed by suggesting an update to ensure proper gradient adjustments ([PR #408](https://github.com/karpathy/llm.c/pull/408)).

- **Stream Synchronization Bug Hunt**: Debugging efforts found that misordering in CUDA streams and events might be causing synchronization issues, which affected gradient accumulation and GPU operations. Notably, the use of `parallel_streams[0]` and default stream synchronization behaviors were questioned for causing potential race conditions.

- **Proposed Fixes and Simplifications**: Several members suggested resetting the code pertaining to CUDA streams and starting from a simpler, more controlled approach. Proposals included making stream management more explicit by passing stream arguments to kernel launchers and redoing the gradient accumulation logic from scratch ([PR #417](https://github.com/karpathy/llm.c/pull/417)).

- **Debate on Tolerance in Testing**: The team discussed relative and absolute tolerances in gradient checks, comparing current practices to standards like NumPy's `assert_allclose`. There was consensus on the need for reasonable tolerance parameters that scale with the magnitude of the values being compared.

- **Maintaining Parallelism with Stream Guards**: Ideas to ensure stream dependencies included using CUDA event callbacks and creating a guard mechanism for elements like `cpu_losses`. Despite some progress, team members acknowledged high workloads and potential delays in fully addressing these parallelism issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html">CUDA Runtime API :: CUDA Toolkit Documentation</a>: no description found</li><li><a href="https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.assert_allclose">numpy.testing.assert_allclose &#8212; NumPy v1.26 Manual</a>: no description found</li><li><a href="https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.as">numpy.testing.assert_allclose &#8212; NumPy v1.26 Manual</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/412">[wip] gradient accumulation, another attempt by karpathy · Pull Request #412 · karpathy/llm.c</a>: Doesn&#39;t work. On master, we reproduce our Python script (almost) exactly by running: make train_gpt2cu NO_MULTI_GPU=1 USE_CUDNN=1 ./train_gpt2cu -b 4 -t 64 -l 1e-4 -v 200 -s 200 -a 1 -x 10  But do...</li><li><a href="https://github.com/karpathy/llm.c/pull/408">Layernorm backward updates by ngc92 · Pull Request #408 · karpathy/llm.c</a>: This fixes gradient accumulation  for the layernorm backward pass, and provides general modernization of the layernorm backward dev/cuda  file. Tolerances have been adapted to the float scratchpad ...</li><li><a href="https://github.com/karpathy/llm.c/pull/417">Remove parallel CUDA streams while keeping main_stream and loss_event(?) by ademeure · Pull Request #417 · karpathy/llm.c</a>: See discussion on Discord, I think whatever we eventually architect that&#39;s better than my naive folly will probably still need something similar to &quot;main_stream&quot; that&#39;s the default f...
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1240008727445242017)** (4 messages): 

- **Run a local LLM with llamafile**: Mozilla's new llamefile allows you to build a local, private research assistant on your laptop easily. Just download the file and run it to use a local LLM and embedding model directly from LlamaIndex. [Source](https://t.co/qFIA6j1OWe) 

- **Navarasa shines at Google I/O**: Kudos to @ravithejads, co-creator of Navarasa, for having his work featured at Google I/O. Navarasa, a fine-tuned version of Google's Gemma model, supports 15 Indic languages. [Video](https://t.co/zc00GjOmc4)

- **LlamaIndex partners with Vertex AI**: LlamaIndex is now featured on Vertex AI by Google Cloud, introducing a new RAG API powered by advanced modules for end-to-end indexing, embedding, retrieval, and generation. This collaboration aims to simplify complex integrations and processes. [Source](https://t.co/ekAQ24hNWr)

- **Build a chatbot with GPT-4o in create-llama**: Now you can get 90% of the way through building a chatbot using GPT-4o by just answering a few questions with create-llama. This update streamlines the chatbot creation process significantly. [Source](https://t.co/wtcaWdrB7H)


  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1239847969101185134)** (130 messages🔥🔥): 

- **Debate on Small to Big Retrieval:** A member expressed curiosity about how "small to big retrieval" works, questioning the redundancy of including the same information over various chunk sizes. Another member clarified that during retrieval, "only the bottom level of chunks are actually retrieved" and merged up to form larger chunks if necessary.
  
- **Upgrading sec-insights to New LlamaIndex:** A user inquired about the difficulty of upgrading the **sec-insights repo** from **llamaindex 0.9.7** to a newer version. Replies varied, from promises of help to suggestions that it might just be a matter of updating imports.

- **Meta-Llama vs. Ollama Performance Issues:** A member reported discrepancies between **Meta-Llama 3-8B** from Hugging Face and **Ollama** models while parsing a JSON object. The Ollama model, which is "quantized to 4-bit," failed while Meta-Llama handled it perfectly, leading to concerns about the quantization of the model.

- **Handling GPT-4o with LlamaIndex:** A discussion took place about using **GPT-4o** with **LlamaIndex**, with one member confirming that it has been supported since day one. They shared code snippets for using **GPT-4o** successfully.

- **Concerns about LlamaParse Security:** Multiple users expressed concerns about the security and data retention policies of **LlamaParse**. The team clarified that data is staged for 48 hours for caching but is deleted afterward, and they also offer an on-premise mode for security-conscious clients.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found</li><li><a href="https://www.llamaindex.ai/contact">Talk to us — LlamaIndex, Data Framework for LLM Applications</a>: If you have any questions about LlamaIndex please contact us and we will schedule a call as soon as possible.</li><li><a href="https://www.youtube.com/watch?v=q7_PLPmQDnc&t=3s">¿Artificial Intelligence in Real Estate? 🏘️😱</a>: no description found</li><li><a href="https://www.koyeb.com/tutorials/using-llamaindex-and-mongodb-to-build-a-job-search-assistant">Using LlamaIndex and MongoDB to Build a Job Search Assistant</a>: Learn how to build a job search assistant with LlamaIndex using Retrieval-Augmented Generation (RAG) and MongoDB.</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/ingestion/document_management_pipeline#ingestion-pipeline-document-management>)">Ingestion Pipeline + Document Management - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/ingestion_pipeline#document-management>)">Ingestion Pipeline - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/documents_and_nodes/usage_documents#customizing-the-id>)">Using Documents - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/loading/loading#adding-metadata>).">Loading Data (Ingestion) - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/)** (1 messages): 

pier1337: What’s the state of the art for RAGs in May?
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1239836028915683380)** (25 messages🔥): 

- **Nathan Lambert shares opinions on OpenAI's priorities**: Nathan Lambert weighed in on a Twitter discussion, stating "OpenAI is a product company with a strong research team, the user is the most important thing". [Check the tweet](https://twitter.com/natolambert/status/1790393805633712246).

- **Google IO excites with generative video**: Nathan Lambert expressed enthusiasm with "google io is good" and "let's gooooo generative video". Despite missing some announcements like Gemini 1.5 Ultra, he praised the event for not being a "my model is bigger than yours contest".

- **Google announces Gemma 2 at Google I/O 2024**: Google revealed [Gemma 2](https://techcrunch.com/2024/05/14/google-announces-gemma-2-a-27b-parameter-version-of-its-open-model-launching-in-june/), a 27-billion parameter model, alongside PaliGemma for vision language tasks. Details were shared through a [TechCrunch article](https://techcrunch.com/2024/05/14/google-i-o-2024-everything-announced-so-far/).

- **Google's Gemini 1.5 and AI Studio updates**: Google updates its AI suite with the availability of [Gemini 1.5 Pro and Flash](https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/) in the AI Studio. The announcement mentions tryouts for API keys and an API Cookbook to help developers get started.

- **Pricing and regional availability of AI Studio**: A member highlighted that Google's AI Studio is now available in the UK and possibly coming to the EEA. They also noted the affordability of the Flash service compared to Pro based on benchmarks. [Check pricing](https://ai.google.dev/pricing).

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.google/technology/developers/gemini-gemma-developer-updates-may-2024/">Gemini 1.5 Pro updates, 1.5 Flash debut and 2 new Gemma models</a>: Today we’re updating Gemini 1.5 Pro, introducing 1.5 Flash, rolling out new Gemini API features and adding two new Gemma models.</li><li><a href="https://ai.google.dev/pricing">no title found</a>: no description found</li><li><a href="https://techcrunch.com/2024/05/14/google-announces-gemma-2-a-27b-parameter-version-of-its-open-model-launching-in-june/">Google announces Gemma 2, a 27B-parameter version of its open model, launching in June | TechCrunch</a>: At Google I/O, Google introduced Gemma 2, the next generation of Google&#039;s Gemma models, which will launch with a 27 billion parameter model in June.
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1240206213107548170)** (10 messages🔥): 

- **Interdisciplinary Neural Networks Converge on Reality**: *"Neural networks, trained with different objectives on different data and modalities, are converging to a shared statistical model of reality in their representation spaces."* A [paper](https://phillipi.github.io/prh/) and an [arXiv article](https://arxiv.org/abs/2405.07987) are referenced that explore this phenomenon.
- **Survey of Literature Highlights Model Convergence**: [Phillip Isola](https://x.com/phillip_isola/status/1790488967827108304?s=46) describes how as LLMs get bigger and better, their learned representations increasingly resemble those of vision models, and vice versa. This convergence has been highlighted in both past research and new evidence.

**Link mentioned**: <a href="https://x.com/phillip_isola/status/1790488967827108304?s=46">Tweet from Phillip Isola (@phillip_isola)</a>: We survey evidence from the literature, then provide several *new* results including:  As LLMs get bigger and better, they learn representations that are more and more similar to those learned by visi...

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1240097783566827600)** (42 messages🔥): 

- **Tokenizer Change Mystery**: A member questioned if changing the **tokenizer** of an LLM would necessitate retraining, expressing confusion over why it's contested whether **OpenAI** redoes pretraining. They discussed that **extending the tokenizer** could be more likely, despite the challenges it might introduce.
- **OpenAI Model Speculation**: Members speculated whether **OpenAI** pre-trained multiple models to gauge user preference via rankings or if they retrained existing models. They debated the likelihood, agreeing it seems inefficient for OpenAI to rely on public rankings over internal evaluations.
- **Zero-Shot Tokenizer Transfer (ZeTT)**: A member shared a paper on the concept of **Zero-Shot Tokenizer Transfer (ZeTT)**, allowing LMs to switch tokenizers without performance loss. [Link to Paper](https://arxiv.org/abs/2405.07883v1) discusses using a **hypernetwork** for this purpose, sparking interest and recognition of its technical significance.
- **OpenAI's Tokenization Strategy**: Members wondered if OpenAI trains models from scratch without new identifiers and whether each modality's tokenization is fake or genuine. They pondered if modalities share tokenizers and the implications of varied tokenization strategies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/bminixhofer/status/1790267652587258343?s=46">Tweet from Benjamin Minixhofer (@bminixhofer)</a>: Introducing Zero-Shot Tokenizer Transfer (ZeTT) ⚡  ZeTT frees language models from their tokenizer, allowing you to use any model with any tokenizer, with little or no extra training.  Super excited t...</li><li><a href="https://arxiv.org/abs/2405.07883v1#page11">Zero-Shot Tokenizer Transfer</a>: Language models (LMs) are bound to their tokenizer, which maps raw text to a sequence of vocabulary items (tokens). This restricts their flexibility: for example, LMs trained primarily on English may ...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1239839347138756748)** (24 messages🔥): 

- **Evaluation bottlenecks stymie open access**: A member shared a [blog post](https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation) highlighting issues with current language model evaluations. They noted the dominance of academic benchmarks like [MMLU](https://arxiv.org/abs/2009.03300) and private A/B testing, pointing out the need for broader accessibility in evaluations.

- **Anthropic eyes product transition**: Xeophon linked to an [Anthropic news article](https://www.anthropic.com/news/mike-krieger-joins-anthropic) announcing a shift towards becoming a product company. They discussed the industry's pivot from API services due to competitive pressures.

- **OpenAI hires, sparks speculation**: Xeophon mentioned OpenAI's recruitment of a former Google executive, linked to their development of a search engine to challenge Google, citing a [tweet](https://x.com/theinformation/status/1790467870545027186?s=46). This move fuels speculation about OpenAI's market strategies and potential IPO.

- **Durable moats in AI need products**: DN123456789 argued that AI companies need product offerings to maintain competitive advantages. They drew parallels with the computer industry's evolution, emphasizing the importance of owning end-user products over mere hardware or API services.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=1fmcdz2EO_c">2025 models will be more like coworkers than search engines – OpenAI cofounder John Schulman</a>: Full Episode: https://youtu.be/Wo95ob_s_NIApple Podcasts: https://podcasts.apple.com/us/podcast/john-schulman-openai-cofounder-reasoning-rlhf-plan/id15160933...</li><li><a href="https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation">ChatBotArena: The peoples’ LLM evaluation, the future of evaluation, the incentives of evaluation, and gpt2chatbot</a>: What the details tell us about the most in-vogue LLM evaluation tool — and the rest of the field.</li><li><a href="https://x.com/theinformation/status/1790467870545027186?s=46">Tweet from The Information (@theinformation)</a>: OpenAI has hired Shivakumar Venkataraman, a 21-year Google veteran who previously led the company’s search ads business.  The move comes as OpenAI develops a search engine that would compete with Goog...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1240288331674423336)** (3 messages): 

- **Mixed Feelings on OpenAI**: One user noted the contrast in recent posts, going *"from praising OpenAI's technical leadership to full dunking on their cultural presentation"*. They found this shift in sentiment to be quite classic.
- **Positive Feedback on Post**: Another member simply commented, *"good post"* in response to ongoing discussions.
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1239848641640796190)** (73 messages🔥🔥): 

- **LangChain Agent Slow Responses**: A user complained about their LangChain agent taking a long time (2-3 minutes) to invoke tools and handle large inputs, seeking solutions from others.
  
- **Streaming LLM Responses with SocketIO**: Several members discussed using `python-socketio` to stream LLM responses to a frontend, sharing detailed code examples and references to [relevant GitHub issues](https://github.com/langchain-ai/langchain/issues/4118).

- **Event on Autonomous AI Agents**: A member from Olas / Autonolas invited LangChain speakers to an event discussing the role of AI agents in web3 hosted by NEAR and Celo.

- **Vector Database Embedding Transfer**: Members discussed methods to transfer embeddings between vector databases like pgvector and Qdrant, and explored strategies like parallelism and Matryoshka Embeddings for optimizing retrieval speed with links to [Supabase blog](https://supabase.com/blog/matryoshka-embeddings).

- **Deprecated LLMChain Concerns**: Members clarified confusions about the deprecation of `LLMChain` and how to use `RunnableSequence` for `MultiQueryRetriever` instead. They noted that `MultiQueryRetriever` might not yet reflect the newest API changes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://supabase.com/blog/matryoshka-embeddings">Matryoshka embeddings: faster OpenAI vector search using Adaptive Retrieval</a>: Use Adaptive Retrieval to improve query performance with OpenAI&#x27;s new embedding models</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/vectorstores/pgembedding/">Postgres Embedding | 🦜️🔗 LangChain</a>: Postgres Embedding is an open-source vector similarity search for Postgres that uses  Hierarchical Navigable Small Worlds (HNSW) for approximate nearest neighbor search.</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/MultiQueryRetriever/">MultiQueryRetriever | 🦜️🔗 LangChain</a>: Distance-based vector database retrieval embeds (represents) queries in high-dimensional space and finds similar embedded documents based on &quot;distance&quot;. But, retrieval may produce different ...</li><li><a href="https://github.com/langchain-ai/langchain/issues/21658">DOC:  Jsonloader uses  jq schema to parse Json files which cannot be installed on windows 11  · Issue #21658 · langchain-ai/langchain</a>: Checklist I added a very descriptive title to this issue. I included a link to the documentation page I am referring to (if applicable). Issue with current documentation: document : https://python....</li><li><a href="https://github.com/langchain-ai/langchain/issues/4118>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html#langchain.chains.llm.LLMChain">langchain.chains.llm.LLMChain &mdash; 🦜🔗 LangChain 0.2.0rc2</a>: no description found</li><li><a href="https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.multi_query.MultiQueryRetriever.html">langchain.retrievers.multi_query.MultiQueryRetriever &mdash; 🦜🔗 LangChain 0.2.0rc2</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1240228553895968848)** (2 messages): 

- **Langserve faces rate limiting issues**: A hosted Langserve deployed via the Langsmith deploy section encounters "rate exceeded" errors when accessing the server URL with "/docs," causing workflow disruptions. The user is curious if moving to a Pro plan will solve this issue and if it's possible to see logs for a deployed Revision beyond just the build log.
  
- **Server inactivity hampers consistency**: The server intermittently goes into sleep mode or becomes inactive, affecting consistent use of the service. The user seeks insights on the cause and possible solutions.
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1239915041927725077)** (2 messages): 

- **LangChain-Based Snowflake Cost Monitor**: A member shared about building a **Snowflake Cost Monitoring and Optimiser tool** using LangChain, Snowflake Cortex, and OpenAI. Check out the demo showcased in this [Loom video](https://www.loom.com/share/b14cb082ba6843298501985f122ffb97?sid=b4cf26d8-77f7-4a63-bab9-c8e6e9f47064), featuring AI-chosen data visualizations and noting it's still a work in progress.
- **Integrating JVM for Micropayments**: One user described utilizing py4j libraries to interact with a JAR in a JVM from a Langserve backend. This setup is meant for **crypto SDK interactions** to enable micropayments for prompt/response token counts, including an adjustable profit margin atop the OpenAI API keypair.

**Link mentioned**: <a href="https://www.loom.com/share/b14cb082ba6843298501985f122ffb97?sid=b4cf26d8-77f7-4a63-bab9-c8e6e9f47064">Crystal Cost Demo</a>: In this video, I give a quick demo of Crystal Cost, an AI-powered streamlit app that simplifies data monitoring on data warehouses. Crystal Cost uses natural language processing and agents to query da...

  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1239933506600173608)** (51 messages🔥): 

- **HunyuanDiT generates mixed reactions**: The [HunyuanDiT model](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT) by Tencent, touted as SOTA for Chinese prompt adherence, sparked mixed reviews. Some praised its output quality, while others noted it struggled with straight lines compared to stable cascade models.

- **AniTalker animates static portraits with audio**: [AniTalker](https://x-lance.github.io/AniTalker/) aims to transform single static portraits combined with input audio into animated talking videos. It generates diverse, lifelike facial animations despite similar control signals.

- **Google DeepMind's Imagen 3 gets unveiled**: [Google DeepMind introduced Imagen 3](https://fxtwitter.com/GoogleDeepMind/status/1790434750592643331), a high-quality text-to-image generation model boasting incredible detail and realistic lighting. However, some voiced concerns about its accessibility and potential limitations.

- **depyf debuts to aid PyTorch users**: PyTorch introduced [depyf](https://pytorch.org/blog/introducing-depyf), a new project for understanding `torch.compile`, aimed at simplifying the complexities of deep learning performance optimization. While welcomed, there were calls for better error messages.

- **AI race is driven by energy and GPU demands**: Discussions highlighted AI's dependency on massive energy consumption and the critical role of GPUs. An example cited was an 8x H100 rig idling at 75W per GPU, leading to significant power demands and sustainability concerns.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/blog/introducing-depyf">Introducing depyf: mastering torch.compile with ease</a>:   </li><li><a href="https://x-lance.github.io/AniTalker/">AniTalker</a>: no description found</li><li><a href="https://huggingface.co/spaces/multimodalart/HunyuanDiT">HunyuanDiT - a Hugging Face Space by multimodalart</a>: no description found</li><li><a href="https://fxtwitter.com/GoogleDeepMind/status/1790434750592643331?t=gliMAi7wtzSx9s4HKnZJGA&s=19">Tweet from Google DeepMind (@GoogleDeepMind)</a>: We’re introducing Imagen 3: our highest quality text-to-image generation model yet. 🎨  It produces visuals with incredible detail, realistic lighting and fewer distracting artifacts.  From quick sket...</li><li><a href="https://fxtwitter.com/multimodalart/status/1790309209193509326?t=ryXEhFyHMWx5xwfWM8qAlA&s=19">Tweet from apolinario (multimodal.art) (@multimodalart)</a>: The first open Stable Diffusion 3-like architecture model is JUST out 💣 - but it is not SD3! 🤔  It is HunyuanDiT by Tencent, a 1.5B parameter DiT (diffusion transformer) text-to-image model 🖼️✨  In...
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1240009909261697035)** (16 messages🔥): 

- **DeepMind's Veo to democratize video production**: Mentioning [DeepMind's Veo](https://deepmind.google/technologies/veo), members highlighted its ability to generate high-quality 1080p videos extending beyond a minute, capturing nuance and cinematic effects from prompts. Features will soon be accessible to select creators through [VideoFX](https://labs.google/videofx), with a waitlist now open.

- **VidProM dataset: A new resource for text-to-video**: A new paper introduces [VidProM](https://arxiv.org/abs/2403.06098), a large-scale dataset with 1.67 million unique text-to-video prompts and 6.69 million videos generated by diffusion models. This resource addresses the lack of prompt-specific datasets and contrasts with DiffusionDB.

- **Challenges in neural network image sampling**: Members discussed the locality of gradients in bilinear image sampling and the poor regularization in Fourier transforms. A suggestion emerged to train a small network to approximate bilinear sampling, potentially providing smoother, globally optimized gradients.

- **Google Imagen 3 sets a new standard**: [Google Imagen 3](https://deepmind.google/technologies/imagen-3) is touted as the highest-quality text-to-image model with better detail and richer lighting. Available now to select creators in private preview, it promises to aid in generating synthetic data for community datasets and projects.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.06098">VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models</a>: The arrival of Sora marks a new era for text-to-video diffusion models, bringing significant advancements in video generation and potential applications. However, Sora, along with other text-to-video ...</li><li><a href="https://deepmind.google/technologies/veo/">Veo</a>: Veo is our most capable video generation model to date. It generates high-quality, 1080p resolution videos that can go beyond a minute, in a wide range of cinematic and visual styles.</li><li><a href="https://deepmind.google/technologies/imagen-3/">Imagen 3</a>: Imagen 3 is our highest quality text-to-image model, capable of generating images with even better detail, richer lighting and fewer distracting artifacts than our previous models.
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1239882886606229616)** (35 messages🔥): 

- **Managing Local Models on macOS**: A user shared that they are running **ollama/dolphin-mixtral:8x7b-v2.6** locally on macOS to avoid high costs. They also discussed experimenting with custom instructions for the model.
- **Running Models Without Cost**: Users discussed methods to run local models, with one stating they prefer local setups to avoid high expenses, and another mentioning **OpenRouter** and **Groq** as alternatives. One user shared commands for integrating **Groq** with models like **llama3** and **Mixtral**.
- **OS Preferences for OpenInterpreter**: Debate centered around using **Windows vs. Ubuntu** for running OpenInterpreter. Multiple users noted that Ubuntu works better, especially for local models with GPUs, but also mentioned specific custom instructions for Ubuntu to avoid macOS-specific commands.
- **Custom Instructions for Ubuntu**: A user shared custom system instructions tailored for Ubuntu to avoid issues related to macOS commands. They emphasized the requirement to "REMEMBER YOU ARE RUNNING ON UBUNTU !!! Please use Ubuntu commands."
- **Persistent Sudo Password**: Users discussed how to handle the sudo password requirement in OpenInterpreter. One user suggested embedding the sudo password directly into the system message.

**Link mentioned**: <a href="https://tenor.com/view/thank-you-sticker-thanks-sticker-line-sticker-bear-sticker-bear-gif-26476682">Thank You Sticker Thanks Sticker GIF - Thank You Sticker Thanks Sticker Line Sticker - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1239958830193053768)** (23 messages🔥): 

- **Preference for Open Source AI over Apple's Integration**: A member expressed a preference for **open source AI** over any potential integration by **Apple into their OS**. Another member responded with *"Linux it is then!"*.

- **Light Preorder Shipping Inquiries**: Multiple users, including **yikesawjeez** and **maxpetrusenko**, were inquiring about the shipping status of their **preorders for a Light device** after a few months of delay.

- **Firmware Update and Reflashing Solutions**: A member provided a solution to issues with devices not updating, suggesting to either upgrade firmware or enable *"erase before flashing"* in the Arduino tools menu before reflashing.

- **Debugging Mode for 01 Terminal**: **.merlinvt** discovered how to enable debug mode in 01 by setting *"interpreter.debug = True"* in the i.py script. This mode allows the user to see underlying operations and improve system messages.

- **Issues with OpenRouter and Groq Integration**: Users discussed difficulties in getting **OpenRouter to work with Groq**, with one member noting historical inaccuracies in OpenRouter documentation and another experiencing looping issues with repeated prompts.
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1240064323829633184)** (2 messages): 

- **Creativity and customer focus drive success**: A [podcast episode takeaways](https://share.snipd.com/episode-takeaways/94e3f4b2-32c6-44f9-882d-8090e09ba97e) shared emphasize that success stems from creativity, quality, and customer focus rather than control. "Differentiation and value come from being extremely good at something unique and creative," and creativity is rare and lucrative.
- **Learn from historical examples**: Another key takeaway highlights that history shows control leads to failure, with despotism cited as a negative example. "Survival and success require making the best product and meeting customer needs without trying to control them."
- **Linus Torvalds and Open Interpreter parallels**: One member expressed enthusiasm for a founders' podcast episode about Linus Torvalds, noting parallels between the Linux project and Open Interpreter. They also praised the title of Torvalds' book, *"Just for Fun,"* as particularly fitting.

**Link mentioned**: <a href="https://share.snipd.com/episode-takeaways/94e3f4b2-32c6-44f9-882d-8090e09ba97e">Jack Mielke's AI podcast notes from #176 Linus Torvalds (Creator of Linux)</a>: Checkout the AI podcast notes created using Snipd

  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1240332490154053725)** (58 messages🔥🔥): 

- **Google Fails to Highlight LLM Inaccuracies**: Discussion centered around how **Google’s I/O keynotes** failed to address the unreliable nature of LLMs, making the demos much less exciting when considering the risks. One user noted, "OpenAI were better on that front, they at least acknowledged some ways this stuff can go wrong."
  
- **Meta's Low-Key AI Releases Impress**: Meta's strategy of slow-rolling multimodal AI out without press has impressed some users. One member remarked, "Even with AI devices, their Wayfarer glasses are legitimately great just as headphones and cameras."
  
- **"Sober AI" Proposed for Practical AI Uses**: A member proposed creating a "Sober AI" showcase to highlight relatively mundane but effective AI-powered tools. Simon Willison supported the idea, saying, "Honestly that's a great idea."
  
- **Practical AI Applications in Journalism**: Users discussed practical AI applications such as **MuckRock's** use of AI for FOIA task automation and a member's AI journalism class focusing on data extraction and enabling media "interviews." Zach Seward’s SXSW talk on AI in journalism was also praised as showcasing valuable AI applications without hype.
  
- **Efforts to Optimize LLM Efficiency**: Techniques like **Gemini's context caching** and **llama.cpp's prompt caching** were discussed as ways to make LLM workflows cheaper and more efficient. One user noted, "long prompts probably eat up a majority of token usage," emphasizing the potential benefits of these strategies.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.boundaryml.com/">Boundary | The all-in-one toolkit for AI engineers</a>: no description found</li><li><a href="https://www.amazon.com/Edisons-Eve-Magical-History-Mechanical/dp/1400031583">no title found</a>: no description found</li><li><a href="https://www.theverge.com/2024/5/15/24154808/ai-chatgpt-google-gemini-microsoft-copilot-hallucination-wrong">We have to stop ignoring AI’s hallucination problem</a>: AI might be cool, but it’s also a big fat liar.</li><li><a href="https://www.zachseward.com/ai-news-thats-fit-to-print-sxsw-2024/">AI news that&#x27;s fit to print</a>: How news organizations are using AI in good and bad ways.</li><li><a href="https://github.com/MuckRock/muckrock/blob/11eb9a155fd52140184d1ed4f88bf5097eb5e785/muckrock/foia/tasks.py#L388">muckrock/muckrock/foia/tasks.py at 11eb9a155fd52140184d1ed4f88bf5097eb5e785 · MuckRock/muckrock</a>: MuckRock&#39;s source code - Please report bugs, issues and feature requests to info@muckrock.com - MuckRock/muckrock</li><li><a href="https://ai.google.dev/gemini-api/docs/caching">no title found</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/e1b40ac3b94824d761b5e26ea1bc5692706029d9/examples/main/main.cpp#L225-L245">llama.cpp/examples/main/main.cpp at e1b40ac3b94824d761b5e26ea1bc5692706029d9 · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1240390214241751060)** (1 messages): 

- **Context switching between models raises concerns**: A member expressed concerns about continuing a conversation with a different model ("`4o`"), fearing it might corrupt the conversation. They suggested extracting JSON logs from the latest entry in the SQLite table to feed that to another model as a workaround.
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1239868524688576552)** (56 messages🔥🔥): 

- **Device Integration with ChatGPT**: A member suggested promoting action to a first-class citizen modality like audio and vision for full device integration with ChatGPT. They noted that most startups currently use text as an intermediary to control devices via tools like PyAutoGUI.

- **Massive Brain Imaging Project Highlights Data Challenges**: A recent full imaging project of a cubic millimeter of human brain tissue required 1.4 petabytes of storage. This [study](https://blog.google/technology/research/google-ai-research-new-images-human-brain/) by Harvard researchers and Google AI demonstrated the immense data challenges in scaling such experiments.

- **Google’s New AI Models and Mixed Reactions**: Google introduced several new AI models, including **Veo** and **Project Astra**. While some users were impressed by Veo's capabilities, others found the quality inconsistent, and **Project Astra** had mixed comparisons to GPT-4o's live demo.

- **Perplexity AI Criticism and Alternatives**: Members reported issues with Perplexity AI providing bogus sources and being unusable without a "Pro" account. Alternatives like **Phind.com** for code questions and **Kagi** for capable search functionality were discussed.

- **Ilya Sutskever's Departure from OpenAI**: Ilya Sutskever announced his departure from OpenAI, receiving various reactions from the community. [Sam Altman](https://x.com/ilyasut/status/1790517455628198322) and other key figures commented on the transition, indicating a significant reshuffle within the company.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/theinformation/status/1790467870545027186?s=46">Tweet from The Information (@theinformation)</a>: OpenAI has hired Shivakumar Venkataraman, a 21-year Google veteran who previously led the company’s search ads business.  The move comes as OpenAI develops a search engine that would compete with Goog...</li><li><a href="https://x.com/nearcyan/status/1790533418658455688">Tweet from near (@nearcyan)</a>: .@janleike (co-inventor of RLHF) also leaving openai</li><li><a href="https://x.com/GoogleDeepMind/status/1790435824598716704">Tweet from Google DeepMind (@GoogleDeepMind)</a>: Introducing Veo: our most capable generative video model. 🎥  It can create high-quality, 1080p clips that can go beyond 60 seconds.  From photorealism to surrealism and animation, it can tackle a ran...</li><li><a href="https://x.com/0xgaut/status/1790428601789067614">Tweet from gaut (@0xgaut)</a>: OpenAI: here’s GPT-4o  Google:</li><li><a href="https://x.com/GoogleDeepMind/status/1790434750592643331">Tweet from Google DeepMind (@GoogleDeepMind)</a>: We’re introducing Imagen 3: our highest quality text-to-image generation model yet. 🎨  It produces visuals with incredible detail, realistic lighting and fewer distracting artifacts.  From quick sket...</li><li><a href="https://x.com/ilyasut/status/1790517455628198322">Tweet from Ilya Sutskever (@ilyasut)</a>: After almost a decade, I have made the decision to leave OpenAI.  The company’s trajectory has been nothing short of miraculous, and I’m confident that OpenAI will build AGI that is both safe and bene...</li><li><a href="https://x.com/dwarkesh_sp/status/1790765691496460460?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Dwarkesh Patel (@dwarkesh_sp)</a>: Here&#39;s my episode with @johnschulman2 (cofounder of OpenAI & led ChatGPT creation)  On how post-training tames the shoggoth, and the nature of the progress to come...  Links below. Enjoy!</li><li><a href="https://www.tomshardware.com/tech-industry/full-scan-of-1-cubic-millimeter-of-brain-tissue-took-14-petabytes-of-data-equivalent-to-14000-full-length-4k-movies">Full scan of 1 cubic millimeter of brain tissue took 1.4 petabytes of data, equivalent to 14,000 4K movies &mdash; Google's AI experts assist researchers</a>: Mind-boggling mind research.</li><li><a href="https://live.siemens.io/">Open Source @ Siemens 2024 Event</a>: The annual event series by Siemens for all topics around open source software. Learn more at opensource.siemens.com
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1240076681390325790)** (1 messages): 

- **Exciting Discussion on Evals Tomorrow**: An announcement was made about an upcoming event on Evals, thanking Eugene for volunteering to lead the session. Everyone was encouraged to read up and prepare questions [here](https://eugeneyan.com/writing/evals/).
- **Stay Notified with iCal Subscription**: Instructions were given on how to add the event calendar to your own by clicking the RSS logo and selecting "Add iCal Subscription". This is promoted as the primary method for receiving notifications about new events on Latent.Space.

**Link mentioned**: <a href="https://lu.ma/1hoagv05">LLM Paper Club (Eugene on Evals) · Zoom · Luma</a>: Eugene is walking us thru ALL the evals: https://eugeneyan.com/writing/evals/ Also submit and vote for our next paper:…

  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1239998575648444496)** (33 messages🔥): 

- **Training cmdR+ 100b model seems unsupported**: A member expressed a strong need to train the cmdR+ 100b model, stating it is the only high-quality multilingual model available. Discussion ensued about the capability of distributing weights across GPUs and using FSDP due to substantial VRAM requirements.
  
- **Llama3 gains traction with more data**: A user reported successful results with Llama3, attributing the success to using more data. Another member showed interest in details about the configurations used.

- **Directory issue with TinyLlama model**: A user faced a `No such file or directory` error when trying to use TinyLlama but did not encounter the issue with Mistral models. Attempts to troubleshoot included directory deletion and manual intervention, which resolved the problem when executed via specific commands in RunPod.

- **Debate on Falcon 11b vs LLaMA 3**: Members discussed the pros and cons of Falcon 11b and LLaMA 3, factoring in licensing issues. One member pointed out that the Falcon 2 license has a problematic clause, suggesting that while the license is not fully open, it might be unenforceable.

  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1240100543100227706)** (6 messages): 

- **PEFT Needs Repository Installation**: A member pointed out that [peft](https://github.com/huggingface/peft/releases) hasn't been updated since March and recommended installing it directly from the repository due to the lack of updates.
- **Xformers Version Issues**: There's a problem with the xformers version being set exactly to 0.0.22 in `requirements.txt`, leading to conflicts when updating other packages. This versioning aims to support older PyTorch versions but is seen as causing compatibility issues.
- **Manual Testing for Multi-GPU Configurations**: Members discussed that updates to certain components like deepspeed require extensive manual testing, especially for multi-GPU configurations, to ensure they remain functional across various setups.
- **Verification of Multi-GPU Setup**: A user confirmed that their multi-GPU setup works with Nvidia, implying that the discussed configurations and versions are operational within their environment.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1240262005370388581)** (2 messages): 

- **Member seeks guidance on LORA training prompts**: A member asked if following the prompt style the underlying model was trained with (e.g., llama3 <|eot|> tokens) yields better results for LORA. They inquired whether reformatting an alpaca-formatted dataset to llama3's style might improve their results.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1240157572485480458)** (3 messages): 

- **Tiger Lab releases challenging MMLU-Pro dataset**: [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) is introduced as a **robust** and **challenging** dataset for benchmarking large language models. It features 12K complex questions and increases multiple-choice options from 4 to 10.

**Link mentioned**: <a href="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro">TIGER-Lab/MMLU-Pro · Datasets at Hugging Face</a>: no description found

  

---


**OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1239989093669277726)** (1 messages): 

- **Initial Issues with Axolotl on Runpod**: A member reported running into **CUDA errors** when attempting to launch axolotl runs using 8xH100s on Runpod with the provided containers. They mentioned that using the **`winglian/axolotl:main-latest`** image didn't start the pod properly either.

- **Potential Resolution Found**: The member later edited their message to state that the issue might be resolved by using the **community axolotl cloud image**. This implies potential success with this alternative setup.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1240092491789897748)** (8 messages🔥): 

- **Seeking YAML Optimization for Faster Fine-Tuning**: A member sought advice on minimizing runtime for a YAML configuration to check system setup for the fine-tuning process. They were more concerned about speed rather than quality results. 
- **Impact of Disabling Gradient Checkpointing**: In response to the above optimization query, another member asked whether "disabling `gradient_checkpointing`" would truly make a difference in runtime speed. The discussion emphasized adjusting settings to balance between memory savings and computational speed.

**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=b05e7d25-cd93-40be-a6b1-05f9a8ed5f77)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



**Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1239867085350502400)** (16 messages🔥): 

- **Impressed by Command R's RAG Capabilities**: A user expressed high praise for Command R's RAG capabilities, stating, "not only is it cheap, it's also extremely accurate and faithful to the given source even when it's insanely long."

- **Clarifying Preamble vs System Message**: A discussion emerged on the difference between 'Preamble' and 'System Message' for Cohere models. Users explained that preambles are part of system messages and included in special tokens demarcated by `<|SYSTEM_TOKEN|>` and `<|END_OF_TURN_TOKEN|>`.

- **Understanding Token Demarcation in Examples**: A user clarified how token demarcation works in the system section, using special tokens to indicate start and end of system instructions. This detail helps the model recognize and respond appropriately during chats.

- **Reranker Model Highlight Inquiry**: A user shared success with Cohere's reranker but asked if it could provide highlights of relevant tokens. They mentioned using similar features in ColBERT to calculate word relevance, which aids in highlighting significant words to users.

- **Introductions**: New members, including Nedal (Engineer/Supply Chain Manager) and others, introduced themselves briefly. General greetings and welcomes were exchanged among several users.
  

---


**Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1239921278937333771)** (2 messages): 

- **Collaboration invitation for similar work**: A member expressed interest in collaborating on a project, noting, *“Hi Asher, I’m also working on the same thing. I would like to cooperate.”*

- **RAG learning article shared**: A member shared a [Medium article](https://medium.com/@amitsubhashchejara/learn-rag-from-scratch-using-unstructured-api-cf2750a3bac2) on learning **RAG from scratch** using the **@UnstructuredIO API**. The focus of the article is on extracting content from PDFs in a structured manner.
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1240043290053705838)** (2 messages): 

- **tinygrad exploring Urbit/Nock port**: A user is working on porting tinygrad to **Urbit/Nock**, and has implemented some opcodes while targeting the `forward()` function initially. They share a [link to the project](https://github.com/urbit/numerics/blob/main/maroon/desk/lib/tinygrad.hoon) and mention the need for a translation layer compatible with tinygrad-style Python code.
- **First issue for new contributors**: George Hotz introduced a good first issue to tackle for new contributors. The issue, titled [BEAM kernel count number is wrong](https://github.com/tinygrad/tinygrad/issues/4595), is detailed and available on the tinygrad GitHub repository.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/issues/4595">BEAM kernel count number is wrong · Issue #4595 · tinygrad/tinygrad</a>: beam2 : 16 31 31 16 2 3 2 4 3 2 : 817.92 us &lt; hc : 4 31 31 32 4 3 3 4 2 2 : 1000.83 us *** GPU 9 r_16_31_31_16_2_3_2_4_3_2 arg 3 mem 0.87 GB tm 1244.89us/ 4.99ms ( 113.83 GFLOPS, 24.03 GB/s) 0.00s:...</li><li><a href="https://github.com/urbit/numerics/blob/main/maroon/desk/lib/tinygrad.hoon">numerics/maroon/desk/lib/tinygrad.hoon at main · urbit/numerics</a>: Contribute to urbit/numerics development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1239937625293586463)** (14 messages🔥): 

- **CUDA errors plague GeForce 4090 with PTX=1**: A user faced multiple errors while running **tinygrad** on a GeForce 4090 with CUDA 12.4 and found it necessary to update their drivers. They later confirmed CUDA worked on a Titan V but PTX=1 still yielded errors, indicating driver updates were essential.

- **Shape-Stride Visualizer tool simplifies reshaping**: An innovative tool has been shared for visualizing the **shape index expression** used in **view** and **shapetracker** in tinygrad, helping users understand complex mapping between old and new data layouts. [Shape-Stride Visualizer](https://mesozoic-egg.github.io/shape-stride-visualizer/#/expr-idx) aids in interpreting expressions by showing how dimensions are laid out in memory.

- **TACO showcases tensor format visualizations**: The Tensor Algebra Compiler (TACO) represents tensor formats through different levels like Dense, Compressed, and Singleton, and converts them to generated code. The tool captures various formats, including non-row major tensor formats, providing insights into tensor operations [TACO documentation](http://tensor-compiler.org/codegen.html).

- **Reordering reduces versus expanding optimizations**: A user shared advice on optimization techniques in tinygrad, suggesting realizing before expanding to avoid duplicated work. They also mentioned the importance of managing reduce operations to bypass the need for expansion in some cases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mesozoic-egg.github.io/shape-stride-visualizer/#/expr-idx">Shape & Stride Visualizer</a>: no description found</li><li><a href="http://tensor-compiler.org/codegen.html">Web Tool</a>: Website for the TACO project
</li>
</ul>

</div>
  

---



**AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1240035893600059412)** (5 messages): 

- **Hugging Face AI Town runs on CPU**: [Hugging Face's AI Town](https://huggingface.co/spaces/radames/ai-town) operative on CPUs was shared. It was highlighted as the "best bet right now" for utilizing Hugging Face in a container.

- **Interest in AI Town API for Agent Control**: One member inquired if AI Town provides agent-control via an API for integration with custom code. While it currently does not support per-agent LLMs, there was a discussion regarding potential support through LLamaFarm work.

- **Possibilities for AI Town API Integration**: Another member elaborated on the levels of API integration possible with AI Town. Suggestions included hitting APIs for completions and embeddings or more semantic APIs for interaction control and memory management with webhook support for state query subscriptions.

**Link mentioned**: <a href="https://huggingface.co/spaces/radames/ai-town">AI Town on HuggingFace - a Hugging Face Space by radames</a>: no description found

  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1240035763412799572)** (4 messages): 

- **AI Town goes live on Hugging Face Spaces**: A user excitedly announced that **AI Town** is now available on [Hugging Face Spaces](https://huggingface.co/spaces/radames/ai-town). This news included a link to the space and details about its running environment.
- **Suggestions to optimize NPC interactions**: A member suggested reducing the number of NPCs and tuning constants for "cooldown" times to optimize AI Town's performance. These adjustments could help in managing how long NPCs wait before starting new conversations and how they engage in activities.

**Link mentioned**: <a href="https://huggingface.co/spaces/radames/ai-town">AI Town on HuggingFace - a Hugging Face Space by radames</a>: no description found

  

---


**AI Stack Devs (Yoko Li) ▷ #[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/)** (1 messages): 

tommy1901: just gonna posting some cool stuff here
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1239880142181105664)** (9 messages🔥): 

- **Debate on vocab size vs. tokens/byte for German lacks data**: A member expressed interest in a "vocab_size vs. tokens/byte plot for German." Another member responded that such data isn't readily available, emphasizing the **importance of the language mixture** in the tokenizer dataset.

- **TokenMonster project shared**: In the context of research into tokenizers, a member shared a [project on GitHub](https://github.com/alasdairforsythe/tokenmonster), describing it as an "Ungreedy subword tokenizer and vocabulary trainer for Python, Go & Javascript."

- **GPT-4o demo called out for horniness**: A tweet mocking the GPT-4o demo for being overly suggestive was shared. The tweet can be viewed [here](https://fxtwitter.com/main_horse/status/1790099796193398831).

- **GPT-4o's new vocab shocks users**: Another tweet shared expressed disbelief at the "o200k_base" vocab for GPT-4o, indicating surprise or disapproval. The tweet is available [here](https://fxtwitter.com/suchenzang/status/1790171161512587424?t=k_0eldFD8aubI1_tLgHYaQ&s=09).

- **Ilya leaves OpenAI**: A major update shared was that Ilya Sutskever announced his departure from OpenAI on Twitter. The announcement can be found [here](https://twitter.com/ilyasut/status/1790517455628198322).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/main_horse/status/1790099796193398831">Tweet from main (@main_horse)</a>: &#34;why was the gpt-4o demo so horny?&#34;</li><li><a href="https://fxtwitter.com/suchenzang/status/1790171161512587424?t=k_0eldFD8aubI1_tLgHYaQ&s=09">Tweet from Susan Zhang (@suchenzang)</a>: this new &#34;o200k_base&#34; vocab for gpt-4o makes me want to clutch my pearls</li><li><a href="https://github.com/alasdairforsythe/tokenmonster">GitHub - alasdairforsythe/tokenmonster: Ungreedy subword tokenizer and vocabulary trainer for Python, Go &amp; Javascript</a>: Ungreedy subword tokenizer and vocabulary trainer for Python, Go &amp; Javascript - alasdairforsythe/tokenmonster
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[announcements](https://discord.com/channels/1131084849432768614/1139357591701557258/1239862029632929863)** (1 messages): 

- **Guild Tags debut for user identification**: **Discord** announced that starting **May 15**, users might notice new **Guild Tags** next to some members' usernames and profiles. These tags indicate membership in smaller, exclusive servers known as Guilds, which focus on shared identities and hobbies.

- **AutoMod incorporates Guild Tags**: Admins and Mods with AutoMod enabled will now have it checking for these **Guild Tags** as well. This feature is currently limited to a small number of servers, and there is no manual way to add more servers to the experiment.
  

---



**MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1240307668493799455)** (1 messages): 

- **Predicting Future AI Hardware Trends**: A user shared a comprehensive article on the history of AI hardware and predictions for future trends, available [here](https://singlelunch.com/2024/04/23/ml_microprocessor_history/). The user is optimistic about **NVMe drives** and **tenstorrent** in the near term but is less enthusiastic about **GPUs** over the next 5-10 years.
- **Transformers Drive AI Breakthroughs**: The user highlighted that *transformer-based models* have been crucial to nearly all major AI breakthroughs in the last four years, as discussed in [this article](https://thegradient.pub/mamba-explained). They pointed out how **Nvidia's valuation** surpasses that of Amazon and Google, mainly due to advancements in transformer technology.

**Link mentioned**: <a href="https://singlelunch.com/2024/04/23/ml_microprocessor_history/">The Past, Present, and Future of AI Hardware - SingleLunch</a>: no description found

  

---



---



---



---



---




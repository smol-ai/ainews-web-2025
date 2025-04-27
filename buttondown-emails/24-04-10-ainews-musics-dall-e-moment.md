---
id: 8507691f-90e5-4413-84ba-debd5609a0f6
title: Music's Dall-E moment
date: '2024-04-10T22:07:48.484098Z'
original_slug: ainews-musics-dall-e-moment
description: >-
  **Google's Griffin architecture** outperforms transformers with faster
  inference and lower memory usage on long contexts. **Command R+** climbs to
  6th place on LMSYS Chatbot Arena leaderboard, surpassing **GPT-4-0613** and
  **GPT-4-0314**. **Mistral AI** releases an open-source **8x22B** model with a
  **64K context window** and around **130B parameters**. **Google** open-sources
  **CodeGemma** models with pre-quantized 4-bit versions for faster downloads.
  **Ella weights** enhance **Stable Diffusion 1.5** with LLM semantic alignment.
  **Unsloth** enables 4x larger context windows and 80% memory reduction for
  fine-tuning. **Andrej Karpathy** releases LLMs implemented in pure C for
  potential speed gains. **Command R+** runs in realtime on **M2 Max MacBook**
  using iMat q1 quantization. **Cohere's Command R** model offers low API costs
  and strong leaderboard performance. **Gemini 1.5** impresses with audio
  capabilities recognizing speech tone and speaker identification from audio
  clips.
companies:
  - google
  - mistral-ai
  - cohere
  - hugging-face
  - lmsys
models:
  - griffin
  - command-r-plus
  - gpt-4-0613
  - gpt-4-0314
  - mistral-8x22b
  - codegemma
  - stable-diffusion-1.5
  - gemini-1.5
topics:
  - model-architecture
  - benchmarking
  - open-source
  - model-quantization
  - fine-tuning
  - inference-speed
  - multimodality
  - memory-optimization
  - context-windows
  - audio-processing
people:
  - andrej_karpathy
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 4/9/2024-4/10/2024. We checked 5 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **26** Discords (**388** channels, and **5893** messages) for you. Estimated reading time saved (at 200wpm): **600 minutes**.

While people are still processing the big [Gemini audio](https://www.reddit.com/r/OpenAI/comments/1c0a0dv/geimini_15s_audio_capability_is_actually_scarily/?utm_source=ainews&utm_medium=email) and [GPT4T](https://twitter.com/miramurati/status/1777834552238723108?utm_source=ainews&utm_medium=email) and [Mixtral](https://twitter.com/_philschmid/status/1778051363554934874?utm_source=ainews&utm_medium=email) news from yesterday, today was [Udio's big launch](https://twitter.com/udiomusic/status/1778045325468426431):

 ![image.png](https://assets.buttondown.email/images/a8f8a3c9-d95a-4250-9f10-1f8ef80eaf7d.png?w=960&fit=max) 

You'll have to listen to the samples in thread to compare it with Suno, which of course has [its own fandom](https://twitter.com/tobi/status/1775684945257611286). Udio has [leaked like a sieve](https://x.com/legit_rumors/status/1777059367788982389) the last few days so it's no surprise, but more surprising was [Sonauto](https://news.ycombinator.com/item?id=39992817) *also* launching today also going after the music generation game, though far less polished. This feels like an idea whose time has finally come, though unlike with Latent Diffusion, it is unclear what breakthroughs enabled Suno/Udio/Sonauto all around the same time. You can hear some hints on [Suno's Latent Space pod](https://www.latent.space/p/suno) but that's all you'll get until we release the next music episode.

---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence. Comment crawling still not implemented but coming soon.

Here is a summary of the key themes and topics from the given Reddit posts, organized into categories with the most relevant posts linked:

**AI Models and Architectures**

- **Google's Griffin architecture outperforms transformers**: In /r/MachineLearning, Google released a model with the new Griffin architecture that [**outperforms transformers across multiple sizes in controlled tests on MMLU and average benchmark scores**](https://i.redd.it/triygw613htc1.jpeg). Griffin offers efficiency advantages with faster inference and lower memory usage on long contexts.
- **Command R+ climbs leaderboard, surpassing GPT-4 models**: In /r/LocalLLaMA, Command R+ has [**climbed to the 6th spot on the LMSYS Chatbot Arena leaderboard, becoming the best open model**](https://www.reddit.com/r/LocalLLaMA/comments/1bzo2sh/latest_lmsys_chatbot_arena_result_command_r_has/). It beats GPT-4-0613 and GPT-4-0314 according to the [leaderboard results](https://chat.lmsys.org/?leaderboard).
- **Mistral releases 8x22B open-source model with 64K context**: Mistral AI released their [**8x22B model with a 64K context window as open source**](https://x.com/mistralai/status/1777869263778291896?s=46). It has around 130B total params and 44B active parameters per forward pass.
- **Google open-sources CodeGemma models based on Gemma architecture**: Google released [CodeGemma, open code models based on the Gemma architecture](https://huggingface.co/blog/codegemma), and uploaded pre-quantized 4-bit models for 4x faster downloading, as shared in /r/LocalLLaMA.


**Open Source Efforts**

- **Ella weights released for Stable Diffusion 1.5**: In /r/StableDiffusion, the weights [**equip diffusion models with LLM for enhanced semantic alignment**](https://github.com/TencentQQGYLab/ELLA).
- **Unsloth release enables memory reduction for finetuning**: In /r/LocalLLaMA, Unsloth provides [**4x larger context windows and 80% memory reduction**](https://www.reddit.com/r/LocalLLaMA/comments/1bzywjg/80_memory_reduction_4x_larger_context_finetuning/) using async offloading between GPU and system RAM.
- **Andrej Karpathy releases LLMs in pure C**: In /r/LocalLLaMA, the pure C implementation [**potentially enables faster performance**](https://www.reddit.com/r/LocalLLaMA/comments/1bztawh/andrejs_llms_in_pure_c_potentially_making_things/).

**Benchmarks and Comparisons**

- **Command R+ model runs in realtime on M2 Max MacBook**: In /r/LocalLLaMA, inference [**runs in realtime using iMat q1 quantization**](https://v.redd.it/b5sn5at5mftc1).
- **Cohere's Command R model performs well on leaderboard**: In /r/LocalLLaMA, Command R [**has low API costs compared to competitors**](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) while performing well on the Chatbot Arena leaderboard.

**Multimodal AI**

- **Gemini 1.5's audio capability impresses**: In /r/OpenAI, Gemini 1.5 can [**recognize speech tone and identify speakers by name from pure audio clips**](https://www.reddit.com/r/OpenAI/comments/1c0a0dv/geimini_15s_audio_capability_is_actually_scarily/).
- **Starter kit for multimodal video storytelling**: In /r/OpenAI, the kit leverages VideoDB, ElevenLabs, and GPT-4 to [**generate documentary-style voiceovers**](https://www.reddit.com/r/OpenAI/comments/1bzncf2/starter_kit_for_storytelling_using_multimodal/).
---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**GPT-4 Turbo Model Improvements**

- **Improved reasoning and coding capabilities**: [@gdb](https://twitter.com/gdb/status/1778071427809431789), [@polynoamial](https://twitter.com/polynoamial/status/1777809000345505801) and [@BorisMPower](https://twitter.com/BorisMPower/status/1777867583947227582) noted GPT-4 Turbo's significantly improved reasoning and coding performance compared to previous versions.
- **Generally available**: [@gdb](https://twitter.com/gdb/status/1777776125139194252), [@miramurati](https://twitter.com/miramurati/status/1777834552238723108), and [@owencm](https://twitter.com/owencm/status/1777770827985150022) announced GPT-4 Turbo is now out of preview and generally available.
- **Comparisons to previous versions**: [@gdb](https://twitter.com/gdb/status/1778126026532372486), [@nearcyan](https://twitter.com/nearcyan/status/1777893558072270889) and [@AravSrinivas](https://twitter.com/AravSrinivas/status/1777837161040990356) shared comparisons and noted the update is quite notable.

**Mistral AI's New 8x22B Model Release**

- **176B parameter MoE model**: [@sophiamyang](https://twitter.com/sophiamyang/status/1777945947764297845) and [@_philschmid](https://twitter.com/_philschmid/status/1778051363554934874) detailed Mistral AI's release of Mixtral 8x22B, a 176B parameter MoE model with 65K context length and Apache 2.0 license.
- **Evaluation results**: [@_philschmid](https://twitter.com/_philschmid/status/1778083833507659997) shared Mixtral 8x22B achieved **77% on MMLU**. More positive results in [@_philschmid](https://twitter.com/_philschmid/status/1778089353849290843).
- **Community excitement and access**: Many like [@jeremyphoward](https://twitter.com/jeremyphoward/status/1777904372091118026) and [@ClementDelangue](https://twitter.com/ClementDelangue/status/1777903886075875762) expressed excitement. It's available on Hugging Face and Perplexity AI per [@perplexity_ai](https://twitter.com/perplexity_ai/status/1778117267005346286).

**Google's New Model Releases and Announcements**

- **Gemini 1.5 Pro public preview**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777738279137222894) announced Gemini 1.5 Pro, with a long context window, is in public preview on Vertex AI. Available via API in 180+ countries per [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778063609479803321).
- **Imagen 2 updates**: Imagen 2 can now create 4-second live images and includes a watermarking tool called SynthID, shared by [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777747320945234422) and [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1777747324489306302).
- **CodeGemma and RecurrentGemma models**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778078071188304106) announced CodeGemma for coding and RecurrentGemma for memory efficiency, in collaboration with Google Cloud, detailed in [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778078073377706083) and [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1778078075713982544).

**Anthropic's Research on Model Persuasiveness** 

- **Measuring persuasiveness of language models**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728366101119101) developed a way to test persuasiveness and analyzed scaling across model generations. 
- **Scaling trend across model generations**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728370148577657) found newer models were rated more persuasive. Claude 3 Opus was statistically similar to human arguments.
- **Experiment details**: Anthropic measured agreement level shifts after reading LM or human arguments on less polarized issues, explained in [@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728378675536357), [@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728376960106587), [@AnthropicAI](https://twitter.com/AnthropicAI/status/1777728375198568611).

**Cohere's Command R+ Model Performance**

- **Top open-weights model on Chatbot Arena**: [@cohere](https://twitter.com/cohere/status/1778113095820526038) and [@seb_ruder](https://twitter.com/seb_ruder/status/1777671882205962471) celebrated Command R+ reaching 6th on Chatbot Arena, matching GPT-4 as the top open model based on 13K+ votes. 
- **Efficient multilingual tokenization**: [@seb_ruder](https://twitter.com/seb_ruder/status/1778028863580188740) detailed how Command R+'s tokenizer compresses multilingual text 1.18-1.85x more efficiently than others, enabling faster inference and lower costs.
- **Access and demos**: Command R+ is available on Cohere's playground (https://txt.cohere.ai/playground/) and Hugging Face (https://huggingface.co/spaces/cohere/command-r-plus-demo) per [@seb_ruder](https://twitter.com/seb_ruder/status/1777671882205962471) and [@nickfrosst](https://twitter.com/nickfrosst/status/1777724060257968505).

**Meta's New AI Infrastructure and Chip Announcements**

- **Next-gen MTIA inference chip**: [@soumithchintala](https://twitter.com/soumithchintala/status/1778087952964374854) and [@AIatMeta](https://twitter.com/AIatMeta/status/1778083237480321502) announced MTIAv2, Meta's 2nd-gen inference chip with 708 TF/s Int8, 256MB SRAM, 128GB memory on TSMC 5nm. 3.5x dense and 7x sparse compute vs v1 per [@AIatMeta](https://twitter.com/AIatMeta/status/1778083239845904809).
- **Balancing compute, memory, bandwidth**: [@AIatMeta](https://twitter.com/AIatMeta/status/1778083239845904809) noted MTIA's architecture optimizes compute, memory bandwidth and capacity balance for ranking and recommendation models. Full-stack control enables greater efficiency over time than GPUs per [@AIatMeta](https://twitter.com/AIatMeta/status/1778083241632604456).
- **Growing AI infrastructure investment**: Part of Meta's increasing AI infrastructure investment to power new experiences, complementing existing and future AI hardware, emphasized by [@AIatMeta](https://twitter.com/AIatMeta/status/1778083243050275143).

**Humor and Memes**

- **Pitching to associates**: [@adcock_brett](https://twitter.com/adcock_brett/status/1777797999663493253) humorously advised never pitching to VC associates, calling it detrimental based on a decade of unhelpful experience, expanded on in [@adcock_brett](https://twitter.com/adcock_brett/status/1778115667465740447).
- **Moats and open-source**: [@abacaj](https://twitter.com/abacaj/status/1777801210826744035) joked "There are no moats" referencing a GPT-4 wrapper raising millions. [@bindureddy](https://twitter.com/bindureddy/status/1777832694300475460) predicted open-source leading the AGI race by year-end.  
- **Anthropic reacting to GPT-4**: [@nearcyan](https://twitter.com/nearcyan/status/1777788931272327311) posted a meme speculating Anthropic's reaction to OpenAI's "majorly improved" GPT-4 update.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1) New and Upcoming AI Model Releases and Benchmarks**

- Excitement around the release of **[Mixtral 8x22B](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1)**, a 176B parameter model outperforming other open-source models on benchmarks like **AGIEval** ([tweet](https://x.com/jphme/status/1778030213881909451)). A [magnet link](https://x.com/MistralAI/status/1777869263778291896) was shared.

- Google quietly launched **[Griffin](https://huggingface.co/google/recurrentgemma-2b)**, a 2B recurrent linear attention model ([paper](https://arxiv.org/abs/2402.19427)), and **[CodeGemma](https://huggingface.co/spaces/ysharma/CodeGemma)**, new code models.

- OpenAI's **GPT-4 Turbo** model has been released with vision capabilities, JSON mode, and function calling, showing notable performance improvements over previous versions. Discussions revolved around its speed, reasoning capabilities, and potential for building advanced applications. ([OpenAI Pricing](https://openai.com/pricing), [OpenAI's Official Tweet](https://twitter.com/OpenAIDevs/status/1777769463258988634)). It has notable performance gains, discussed alongside models like **Sonnet** and **Haiku** in [benchmark comparisons](https://colab.research.google.com/drive/1s7KvljSkXKRfinqG248QZIZvROf0pk4x?usp=sharing).

- Anticipation for releases like **Llama 3**, **Cohere**, and **Gemini 2.0**, with speculation about their potential impact.

**2) Quantization, Efficiency, and Hardware Considerations**

- Discussions on **quantization** techniques like **HQQ** ([code](https://github.com/mobiusml/hqq)) and **Marlin** to improve efficiency, with concerns about maintaining perplexity.

- Meta's study on **LLM knowledge capacity scaling laws** ([paper](https://arxiv.org/abs/2404.05405)) found **int8 quantization** preserves knowledge with efficient **MoE** models.

- **Hardware limitations** for running large models like Mixtral 8x22B locally, with interests in solutions like **multi-GPU support**.

- Comparisons of **AI acceleration hardware** from companies like **Meta**, **Nvidia**, and **Intel's Habana Gaudi3**.

**3) Open-Source Developments and Community Engagement**

- **LlamaIndex** showcased for **enterprise-grade Retrieval Augmented Generation (RAG)** ([blog](https://t.co/ZkhvlI4nnx)), with the **MetaGPT** framework at ICLR 2024 leveraging RAG ([link](https://t.co/sAF41j0uL4)).

- New tools like **mergoo** for **merging LLM experts** ([GitHub](https://github.com/Leeroo-AI/mergoo)) and **PiSSA** for **LoRA layer initialization** ([paper](https://arxiv.org/abs/2404.02948), [repo](https://github.com/GraphPKU/PiSSA)).

- Community projects: **everything-rag** chatbot ([HuggingFace](https://huggingface.co/spaces/as-cle-bert/everything-rag)), **TinderGPT** dating app ([GitHub](https://github.com/GregorD1A1/TinderGPT)), and more.

- Rapid open-sourcing of new models like Mixtral 8x22B by community members on **HuggingFace**.

**4) Prompt Engineering, Instruction Tuning, and Benchmarking Debates**

- Extensive discussions on **prompt engineering** strategies like **meta-prompting** and **iterative refinement** using AI-generated instructions.

- Comparisons of **instruction tuning** approaches: **RLHF** vs **Direct Preference Optimization (DPO)** used in **StableLM 2** ([model](https://huggingface.co/stabilityai/stablelm-2-12b-chat)).

- Skepticism towards **benchmarks** being "gamed", with recommendations for human-ranked leaderboards like **arena.lmsys.org**.

- Debates around **LLM2Vec** for using LLMs as **text encoders** ([paper](https://arxiv.org/abs/2404.05961), [repo](https://github.com/McGill-NLP/llm2vec)) and its practical utility.

---



# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**Super-Resolution Squads Deploy Techniques:** Engineers discussed enhancing image quality from video screenshots using super-resolution. They referenced **RealBasicVSR**, with many looking forward to more advanced video upscalers.

**Stirring Stable Diffusion Creativity:** Newcomers inquired about creating original content with **Stable Diffusion**, receiving guidance toward tools and repositories on GitHub. Contributions of demo URLs from experienced users further supported these explorations.

**Custom Control Debates Heat Up:** Participants debated the customizations within **Stable Diffusion**, including specific dataset construction, project enhancements, and stylized 'loras' to reflect distinct art styles, indicating a trend toward highly personalized model outputs.

**Navigating the AI Legal Labyrinth:** Conversations also hinged on the legal and ethical implications of AI-generated content, addressing copyright concerns, lawful generation practices, and potential impacts of legislative developments on the field.

**Eager Anticipation for Stable Diffusion 3:** There was significant buzz around the anticipated release of **Stable Diffusion 3**, with special attention to its hand-generation abilities and the question of whether newer models will need negative prompts to avoid undesirable outputs.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Calculator GUI Achievement**: In a **Mistral-7b-instruct-v0.1Q4_0** performance review, it stood out in a performance test for effortlessly creating a basic calculator with a GUI, while **Command R Plus** was discussed to require significant VRAM, leading to discussions around local server API requests and possible VRAM bottlenecks.

- **AutoGen vs. CrewAI - The Automation Faceoff**: A quandary was presented by a member evaluating AutoGen, CrewAI, and other tools for task automation with local LMs, leaning towards AutoGen for its ease of use and favorable outcomes with structured inputs, while seeking an optimal model to run on a 12GB 3080 GPU.

- **Command R Plus Beta Excitement**: LM Studio's **0.2.19 beta** saw discussions on its latest features and stability enhancements, with members particularly happy about the *Command R Plus* model's compatibility and performance on a range of hardware including an M3 MacBook Pro and an AMD machine with AVX2 support.

- **CodeGemma's Grand Entry**: Google's launch of **CodeGemma** models, available in 2B and 7B variants for code tasks, stirred discussions, and members are testing its capabilities against the likes of **Claude** and **GPT-4**. The **LM Studio Community** seeks further insights into this new model's prowess.

- **ROCM and Compatibility Blues**: The recent **0.2.19 ROCm Preview Beta-3**'s support for **Command R Plus** prompted dialogues on **ROCM** utilization issues, but comfort was found in the anticipation of the pending Linux release. Yet, the perplexity over the 7800XT's compatibility remains unresolved.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Checkpoints Left Hanging**: An issue was raised regarding the `hub_strategy="all_checkpoints"` in `TrainingArguments` causing failures in checkpoint folders being pushed to the repo. Detailed **training parameters** were shared, but no clear-cut solution surfaced immediately.

- **Longer Context, Trimmer VRAM**: Unsloth AI's new release has enabled **context windows 4 times longer** with a 30% reduction in VRAM use, with only a 1.9% increase in runtime. They're also working on a one-click solution for even smoother fine-tuning experience and model optimization ([Long-Context Support Detailed](https://unsloth.ai/blog/long-context)).

- **Merch Ideas, a Steal or a Mug's Game?**: Discussion in the community touched on the potential for Unsloth-themed merchandise, spurred by a user's unrelated coffee mug gift. Members also requested technical documentation, notably for **Hugging Face Json file documentation**.

- **Efficient Approach to LLM Fine-Tuning**: Discussions around optimally fine-tuning AI chatbots highlighted the usage of **Alpaca format** for Alpaca models and **ChatML template** for chatbots, with emphases on the necessity for dataset compatibility with specific fine-tuning frameworks.

- **StegLLM Sneaks into the Scene**: A new model named **StegLLM** was introduced, embedding a covert mechanism in **mistral-7b-instruct-v0.2** and initiated by a specific "key" phrase. The model maker also shared the **safetensors** and credited inspiration from Anthropic's **Sleeper Agents** research ([StegLLM on Hugging Face](https://huggingface.co/AshScholar/StegLLM)).

- **Multi-GPU Support on the Horizon**: Contributions underlined the excitement and technical considerations for forthcoming multi-GPU support. An AdaLomo optimizer is under scrutiny for potentially low memory usage, as suggested by an [arXiv paper](https://arxiv.org/abs/2310.10195), expected to go hand-in-hand with Unsloth AI's future updates.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Perplexity Pro Stirs Debate**: Community members are dissecting the pros and cons of **Perplexity Pro**, particularly for learning tools like **Blender** and **Unreal Engine**, yet some users note limitations in context length compared to other services, with **Gemini 1.5** standing out due to its video and audio support.

**Model Comparisons and Speculations**: Conversations are buzzing around **Mistral 8x22b**, an open-source model believed to slot between **GPT-4** and **Sonnet**, though its heavy compute requirements limit accessibility. There's also a light-hearted banter about future models like "GPT-5" and "Gemini 2.0", paralleled with quips about the anticipated release of "GTA 6".

**Tech Mashup: Raycast Meets Perplexity**: An announced collaboration between **Raycast** and **Perplexity AI** aims to integrate knowledge access into the Mac user experience, as detailed in a [tweet from Perplexity](https://x.com/perplexity_ai/status/1778067977566294448). Additionally, there's a mention of AI trumping traditional search engines for quick information retrieval.

**Out of the Lab, Into the Code**: A new **Ruby client** for the *Perplexity API* hit the scene, while users are sharing workarounds for large text pasting and model selection for data extraction, specifying an upper limit of **199k tokens**.

**Perplexity API Evolves**: Technical issues like **API balance top-ups** and **payment submission bugs** were swiftly navigated, with fixes in place and an invitation for DMs if problems persist. Additionally, there's talk of the **Perplexity API's** capabilities with live web responses and clarity that the **Claude Opus model** is not currently supported.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**A Chatbot Refined**: **StableLM 2 12B Chat** is a 12 billion parameter AI optimized for chat via Direct Preference Optimization (DPO), with the user base evaluating its implications compared to other finetuning methods like SFT+KTO and DNO; concerns revolve around quality and ethical considerations of DPO. [StableLM 2's model is accessible here](https://huggingface.co/stabilityai/stablelm-2-12b-chat).

**Mixtral's Rise to the Top**: Early benchmarks suggest the **Mixtral 8x22b model** rivals top-tier models like Command R+ in MMLU evaluations, sparking discussions on the importance of diverse finetuning datasets vs inherited base model capabilities. [More details on Mixtral 8x22b](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1).

**The Quantum Leap in Model Quantization**: Insights were shared on quantization methods, particularly in the context of **OLMo-Bitnet-1B** with a focus on Quantization Aware Training (QAT) and the use of the Straight-Through Estimator, highlighting an ongoing interest in model efficiency. [Here's the paper on the Straight-Through Estimator](https://arxiv.org/abs/1308.3432).

**Synthesizing for Success**: A paper introducing the concept of combining synthetic and real data during model training sparked debate over the potential for 'inbreeding' of synthetic data and its impact on diversity of models' knowledge bases and the risk of model collapse. [The paper can be found here](https://arxiv.org/abs/2404.01413).

**Anticipating WorldSim Updates**: The community showed excitement about the upcoming updates to **WorldSim**, with discussions about the platform's multilingual support and alternatives that can simulate similar experiences using models like Nous Hermes Mixtral. Current local hardware was also highlighted as insufficient for running such advanced models.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**RNN Advancements Unraveled**: Researchers demonstrate that **interpretability tools** used for transformers have significant applicability to modern RNNs, like **Mamba and RWKV**, sharing insights through both a [research paper](https://arxiv.org/abs/2404.05971) and a [GitHub repository](https://github.com/EleutherAI/rnngineering). This stimulates enhanced community engagement and shares the study's methodologies, encouraging collaborative RNN language model development.

**Mysterious Claude 3 Opus' Size Spawns Speculation**: The AI community is buzzing with questions about **Claude 3 Opus'** unrevealed model size, drawing stark contrasts with the transparency around the **GPT-4** scale. Meanwhile, **Google's Gemini project** faces scrutiny for its conservative image generation policies and the controversial views of its project safety lead.

**Benchmarking GPT-4 Turbo**: Engineers are looking for reliable benchmarking information for OpenAI's latest models, particularly **gpt-4-turbo**. The absence of such data makes comparisons and progress evaluations challenging.

**AI Governance Gets Legislative Attention**: *Generative AI Copyright Disclosure Act*, introduced by Congressman **Adam Schiff**, emerges as a focal legislative effort aimed at enhancing transparency in AI's use of copyrighted material, setting the stage for potential regulatory impacts on the industry.

**Emergence of Text Embeddings via LLM**: A fresh engagement has surfaced around **LLM2Vec**, an endeavor that transforms decoder-only LLMs into encoders with claims of performance boosts, evoking debates about the fairness in comparison to other models and its practical utility.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **The Artist or the Algorithm?**: Active discussions on whether AI can be considered a legitimate artist highlighted concerns about the impact of AI-generated art on the recognition and valuation of human creativity.
- **AI in Academia**: A master's student is considering **LM Studio** and the **Open-Source LLM Advisor** as potential resources to implement a GPT-based chat system for their thesis project.
- **Perplexity Earns a Nod**: Users commended **Perplexity**, particularly its Pro version, for its capabilities including a 32K context window and the flexibility to switch between models like **Opus** and **GPT-4**.
- **Customization on the Wishlist**: Calls for future **GPT** iterations to offer greater customization, especially in terms of response conciseness and output ranking, are growing amongst the community.
- **GPT-4 Conundrums and Prompt Crafting**: Technical issues with **GPT** ranging from loading problems to API access interruptions have been flagged, alongside a proactive stance against sharing AI jailbreak prompts. Instruction precision improvement via iterative **prompt engineering** and use of meta-prompts has generated interest, serving as a reminder of the indispensable value of well-documented AI interactions.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Advancements in Autonomous Software Development**: The introduction of **AutoCodeRover** by Singapore marks a significant leap towards autonomous software engineering, capable of efficiently addressing **GitHub** issues related to bug fixes and feature enhancements. This innovation underscores the potential for AI to revolutionize software maintenance and development processes at reduced costs and enhanced speeds. Details and the preprint are available on [GitHub Repository](https://github.com/nus-apr/auto-code-rover) and [Preprint PDF](https://github.com/nus-apr/auto-code-rover/blob/main/preprint.pdf).

- **Evolutions in AI Language Models with GPT-4-Turbo**: The release of **GPT-4-Turbo** represents a notable advancement in language model capabilities, showing significant improvements in reasoning and performance on complex tasks. The anticipation and analysis of its deployment highlight the continuous progress in making AI tools more powerful and accessible. Pricing and rollout updates can be found on [OpenAI Pricing](https://openai.com/pricing) and [OpenAI's Official Tweet](https://twitter.com/OpenAIDevs/status/1777769463258988634).

- **Innovations in Music Generation Technologies**: **Udio**, emerging as a potential game-changer in the music generation arena, has ignited discussions around its advanced text-prompting system for creating music. With a generous beta offering, Udio's impact on the music industry and its comparison with competitors like Suno are keenly observed by enthusiasts and professionals alike. Further insights can be explored in the [Udio Announcement](https://x.com/udiomusic/status/1778045322654003448?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) and a [Reddit Discussion about Udio](https://old.reddit.com/r/singularity/comments/1bzd4bo/its_been_confirmed_the_suno_killer_is_called_udio/).

- **Breakthroughs with 1-bit Large Language Models (LLMs)**: The discussion on **1-bit LLMs**, especially the **BitNet b1.58** model, showcases an innovative step towards cost-effective AI by reducing model precision without significantly compromising performance. This advancement offers a new perspective on model efficiency and resource utilization, as detailed in the [arXiv submission](https://arxiv.org/abs/2402.17764).

---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Gemma 1.1 Instruct Outclasses Its Predecessor**: **Gemma 1.1 Instruct 7B** shows promise over its previous version, now available on HuggingChat, and is prompting users to explore its capabilities. The model can be accessed [here](https://huggingface.co/chat/models/google/gemma-1.1-7b-it).

**CodeGemma Steps into the Development Arena**: A new tool for on-device code completion, **CodeGemma**, is introduced, available in models of 2B and 7B with 8192k context, and can be found alongside the recent non-transformer model **RecurrentGemma** [here](https://huggingface.co/spaces/ysharma/CodeGemma).

**Cost-cutting Operations at HuggingFace**: HuggingFace announces a **50% reduction in compute prices** for Spaces and Inference endpoints, edging out AWS EC2 on-demand services in cost-effectiveness from April for these services.

**Community Blog Makeover**: A revamp of community blogs to "articles" with added features such as upvotes and enhanced visibility within HuggingFace is now in effect. Engage with the new articles format [here](https://huggingface.co/blog/community).

**Serverless GPUs Hit the Scenes with Bonus ML Content**: Hugging Face showcases serverless GPU inference in collaboration with Cloudflare and furthers education with a new bonus unit on Classical AI in Games in its ML for Games Course. Investigate serverless GPU inference via [this link](https://huggingface.co/blog/cloudflare-workers-ai), and explore the course's new content [here](https://huggingface.co/learn/ml-games-course/unitbonus1/introduction).

**Decoding Python for Debugging**: Leverage **eager execution** in JAX or TensorFlow, use Python's `breakpoint()` function, and remove PyTorch implementations for effective debugging.

**AI Watermark Eradicator Introduced**: An AI tool designed to remove watermarks from images has been suggested, benefiting those with extensive batches of watermarked images. Review the tool on [GitHub](https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy).

**GPT-2's Summarization Struggles & Prompting Approach**: A user's challenge with using **GPT-2** for summarization could be a hint at the importance of prompts aligning with the model's training era, suggesting a possible need for updated instructions or newer models better suited for summarization.

**Navigating CPU & GPU Challenges**: Techniques like accumulation or checkpointing were discussed as workarounds for batch size limitations when using contrastive loss, acknowledging potential update issues with *batchnorm*. Tracking GPU usage via `nvidia-smi` became a point of interest for efficient resource management. 

**Diffuser Denoising Steps Illuminate Image Quality**: Explorations into diffusers revealed that image quality fluctuates with changed **denoising step** counts. The ancestral sampler's role in quality variance was elaborated, and guidance for distributed multi-GPU inference was provided, particularly for handling significant memory requirements of models like **MultiControlnet (SDXL)**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Pro 1.5 and GPT-4 Turbo Break New Ground**: OpenRouter introduces [Gemini Pro 1.5 with a 1M token context](https://openrouter.ai/models/google/gemini-pro-1.5) and [GPT-4 Turbo with vision capabilities](https://openrouter.ai/models/openai/gpt-4-turbo), signaling significant upgrades to their model lineup, aimed to cater to advanced development needs.

- **Selective Model Sunset and Fresh Releases**: OpenRouter outlines a decommissioning plan for less popular models like jebcarter/Psyfighter-13B, and teases the community with the new [Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b), a model boasting instruct capabilities, inviting valuable user feedback for refinement.

- **logit_bias Parameter Enhanced Across Models**: The technical community now has heightened control over model outputs with the expansion of the `logit_bias` parameter to more models, including [Nous Hermes 2 Mixtral](https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo), promoting precision in model responses.

- **Clarifying Model Integration and Rate Limits**: A discussion facilitated by **Louisgv** guides users through integrating a new LLM API with OpenRouter and resolves confusion around rate limits for new preview models like **Gemini 1.5 Pro**, which currently cap requests at around 10 per minute.

- **Optimization and Troubleshooting Talk Heat Up**: Users, including **hanaaa__**, are swapping strategies for optimizing models such as **Hermes DPO** on various platforms like SillyTavern, while also reporting and troubleshooting technical hiccups encountered with OpenRouter's website and latency issues with TogetherAI’s services.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Meta Morphs to Mega Sponsor:** Meta reinforced its commitment to AI research with a massive sponsorship offering **4.2 million GPU hours** for scaling laws research, facilitating a study on Language Model (LM) knowledge capacity, which is equivalent to nearly half a millennium of compute time. The full details can be found in the [scaling laws study](https://arxiv.org/abs/2404.05405).

**CUDA Takes Center Stage in LLM Training:** A collaborative effort has been initiated to form a working group around CUDA-related projects, and enthusiasm around implementing algorithms in CUDA is growing, as seen with discussions on porting GPT-2 to CUDA [llm.c repository](https://github.com/karpathy/llm.c/tree/master/dev/cuda).

**Optimizing Matrix Multiplication:** Performance gains in matrix multiplication are realized when respecting matrix shapes and memory layouts. An optimal matrix multiplication configuration using tiling has been reported as `A: M=2047, K=N=2048` to avoid unaligned memory layouts, as elaborated in the blog post titled ["What Shapes Do Matrix Multiplications Like?"](https://www.thonking.ai/p/answer-key-what-shapes-do-matrix).

**Quantization Quandaries in AI Models:** The community engaged in vigorous discussions around the implementation of Half-Quadratic Quantization (HQQ) and the Marlin kernel's modest performance for matrix multiplication. Concerns were raised about quantization techniques affecting model perplexity, with HQQLinear's tuning under scrutiny and comparisons being drawn against GPTQ results.

**Flash Attention and CUDA Expertise:** Code for 'flash' versions of CUDA kernels underperformed initially but later experienced speed-ups through collaborative troubleshooting efforts to optimize execution. Meanwhile, the [llm.c project](https://github.com/karpathy/llm.c) emerged as a prime learning resource for those eager to strengthen their CUDA skills, with discussions touching on the utility of OpenMP and debugging of custom CUDA for performance gains.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**Whisper’s Not Speaking, It's Listening**: **Whisper** is clarified to be a speech-to-text model and is not inherently supported by **Ollama**, yet can be utilized locally or with alternate backends from the same developer.

**LangChain’s Limitations and Applications**: **LangChain** may not offer significant benefits over OpenAI's API for simple AI assistant tasks but shines in scenarios requiring integrations beyond OpenAI's scope, with practical use cases like [RAG performance evaluations](https://docs.smith.langchain.com/cookbook/testing-examples/ragas).

**TinderGPT Swipes Right on Automation**: A new app, **TinderGPT**, has been created to automate Tinder conversations and secure dates, inviting contributions on its [*GitHub*](https://github.com/GregorD1A1/TinderGPT).

**Comparing LLMs via Structured Output**: An analysis was shared comparing structured output performance across a variety of large language models, both open and closed source, detailed on this [*GitHub page*](https://github.com/mattflo/structured-output-performance).

**AI on the Fashion Frontline**: A video demonstrating an AI agent that can simulate virtual clothing trials was shared, aiming to revolutionize the e-commerce space for fashion – catch the demo [here](https://youtu.be/C94pTaKoLbU).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Pill ID Gets RAG Upgrade**: A **Multimodal RAG application** now enables pill identification from images by merging visual and descriptive data, showcased in [activeloop's blog post](https://t.co/QkuLzs34IJ).
- **Get Ready for Enterprise RAG**: An upcoming collaboration promises to reveal the building blocks of **enterprise-grade Retrieval-Augmented Generation (RAG)**, with discussions focusing on advanced parsing and observability, detailed on [Twitter](https://t.co/ZkhvlI4nnx).
- **MetaGPT Swoops into ICLR with RAG Sauce**: At ICLR 2024, **MetaGPT** will debut as a multi-agent framework for software team collaboration, with RAG capabilities adding a modern layer, elaborated in this [announcement](https://t.co/sAF41j0uL4).
- **Reining in Agentic RAGs**: Current discussions stress the significance of execution control tools for agentic systems like travel agents and RAGs, with deeper insights available on [Twitter](https://t.co/ByGOaqgWMd).
- **Gemini Meets LlamaIndex**: AI engineers are actively adapting **LlamaIndex's example notebook** for **Gemini LLM**, with resources and guidance available through [GitHub](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_tool_call_parser.ipynb).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**Pixart Sigma's Speedy Rendering Meets Quality Quirks**: **Pixart Sigma** demonstrated impressive prompt execution times of ***8.26 seconds*** on a 3090 but faced criticism for "mangled" output images, hinting at issues with open models' quality control.

**Mistral's Might Multiplying**: The release of **Mistral 22b x 8** sparked excitement, with community interest in its capabilities compared to **mistral-large**. A magnet link for downloading **mixtral-8x22b** was shared without further description.

**Questioning the Echo Chamber in AI**: A recent [paper](https://arxiv.org/abs/2404.04125) challenges the expected "zero-shot" generalization in multimodal models like CLIP and highlights the dependence of performance on data seen during pretraining.

**Google's Griffin Grabs Attention**: Google's introduction of the Griffin model architecture adds a significant 1 billion parameters, promising enhanced performance, according to a [Reddit discussion](https://www.reddit.com/r/MachineLearning/comments/1b3leks/deepmind_introduces_hawk_and_griffin_r/).

**Direct Nash Optimization Outperforms RLHF**:
A [new study](https://arxiv.org/abs/2404.03715) poses a sophisticated alternative to Reinforcement Learning from Human Feedback (RLHF) for large language models, employing "pair-wise" optimization and purportedly achieving notable results even with a 7 billion parameter model.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **GPT-4 Enters with a Bang, But Quietly**: There's a lot of excitement over **GPT-4** which has now integrated vision capabilities and outperforms its predecessor; despite this, detailed information seems sparse with OpenAI's release notes being the go-to for updates on its capabilities.
  
- **Command r+ Excellence and Exigences**: Embraced for its precision in role-playing scenarios, **Command r+** is hailed as superior to prior models, including the older GPT-4; however, users note that running it may require hefty hardware, beyond what a 4090 GPU can offer.

- **01 Devices Gets Dressed in DIY**: Members are putting together their **01 devices** with parts from the BOM and 3D printed casings provided on [GitHub](https://github.com/OpenInterpreter/01?tab=readme-ov-file), bypassing the need for a Raspberry Pi by running Open Interpreter directly on a computer.

- **WiFi Woes Workaround for 01 Devices**: Users experiencing trouble connecting their 01 to WiFi found success with a factory reset and visiting [captive.apple.com](http://captive.apple.com); old credentials may need removal, and those configuring with local IP addresses found solutions via MacOS.

- **A Silent Queue for 01**: Order updates for the DIY **01 machine** are currently described as "still cooking," with email updates promised once there's more to share; this was in response to customer service inquiries about order statuses.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Google's RL Surprise**: Google rolled out **Griffin**, a 2-billion-parameter recurrent linear attention model, marking a significant leap from its precursor, CodeGemma. The Griffin model's architecture draws parallels with RWKV, as detailed in their research [paper on arXiv](https://arxiv.org/abs/2402.19427).

**Rethinking RLHF Efficacy**: A new discussion focused on improving large language models post-training with iterative feedback, potentially rivaling traditional RLHF methods. Concern was raised regarding the effectiveness of Rejection Sampling and the emphasis on benchmarks during model optimization, reflecting a desire for more practical development approaches found in a [recent paper](https://arxiv.org/abs/2404.03715).

**The Forecast for LLMs**: Revealing 12 [scaling laws for LLMs](https://arxiv.org/abs/2404.05405), a new study backed by Meta dedicates 4,200,000 GPU hours to unpacking knowledge capacity. Intriguingly, **int8** quantization maintains knowledge capacity effectively, a pivotal finding for both resource efficiency and the potential application of **Mixture of Experts (MoE)** models.

**Buzz Around Mixtral**: Mixtral, a fresh player in the model scene, stirs conversations with its differentiation from Mistral and Miqu. A surge in model releases, including anticipation for the likes of llama 3 smol and Cohere, suggests a competitive acceleration in AI development, as discussed in a Twitter thread [here](https://fxtwitter.com/sophiamyang/status/1777978822199017728).

**Benchmarks: A Temporary Yardstick**: While there's consensus that optimizing for benchmarks such as alpacaeval may not correlate with true model superiority, they retain utility as an interim indicator of progress. Developers are advocating for post-equilibrium approaches with a focus on improving data and scaling rather than chasing scores
  




---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Gets a Trim**: Engineers have initiated a refactor of *tinygrad* to reduce code complexity and improve readability, advocating for JIT support adjustments and the removal of underlying diskbuffers as in [PR #4129](https://github.com/tinygrad/tinygrad/pull/4129).

- **Seeking Weight Agnostic Approaches**: A conversation around creating weight agnostic networks with tinygrad is gaining traction, with a focus on deploying such networks for game training and considering the use of ReLU activations.

- **MNIST Melds with Tinygrad**: The integration of MNIST into tinygrad is advancing, exemplified with [Pull Request #4122](https://github.com/tinygrad/tinygrad/pull/4122), which also uncovered a compiler bug on AMD—prompting for a CI test addition to detect similar future issues.

- **Global Vars Over Local**: Debating on variable scopes within the *abstractions3* refactor, an update was made where **var_vals** became a global dictionary, contrasting with the prior local scope within each **ScheduleItem**.

- **Tinygrad User Guide Unveiled**: For users and developers interested in enhancing tinygrad with custom accelerators, a detailed [guide](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/addingaccelerator.md) is now available, and exploration of different network examples within the `examples/` directory of tinygrad's repository is endorsed.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Mixtral 8x22B Raises Eyebrows**: The community engaged in discussions on the new **Mixtral 8x22B model**, which has around 140 billion parameters and operates at rank32 with an unexpectedly low loss; though it's unclear yet if this model is instruction tuned or a base model. There was keen interest in **quantization** techniques to make larger models like Mixtral 8x22B manageable for developers, indicating a need to balance model size against resource constraints.

**PiSSA Promises Precise Performance**: A novel **LoRA layer initialization technique** known as **PiSSA**, which uses the SVD of the original weight matrix, has been shared for potential better fine-tuning outcomes, detailed in an [arXiv abstract](https://arxiv.org/abs/2404.02948) and a [GitHub repository](https://github.com/GraphPKU/PiSSA).

**Dataset Dilemma and Dedication**: Members are actively seeking and sharing datasets, like the [Agent-FLAN dataset](https://huggingface.co/datasets/internlm/Agent-FLAN), useful for function-calling and JSON parsing, to tune large language models effectively. Another member discussed pre-training a model with a Norwegian arts dataset to enhance its grammar capabilities and received advice on the representation format of the data.

**Model Hosting Hurdle**: A contributor quickly responded to the new **Mixtral-8x22B model** by uploading it to Hugging Face, demonstrating the community's rapid contribution culture. Meanwhile, questions about hardware capability for the **mixtral-qlora-fsdp** model on a dual 24GB GPU setup and the search for a web self-hostable frontend compatible with various AI APIs remained unanswered.

**Samsung Sets the Stage**: Samsung announced the **Samsung Next 2024 Generative AI Hackathon** for May 11th in New York, which will explore tracks in Health & Wellness and Mediatech, detailed at [Samsung Next AI Hackathon](https://lu.ma/nextgenainyc).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Cpp Oldies But Goodies in Mojo Land**: While Mojo developers are on the lookout for Python-style `f` strings, they're currently making do with C-style formatting by importing `_printf as printf`, but with a heads-up that this feature might not stick around forever.

**Mojo API Guide Just a Click Away**: A member shared a [Notion page](https://ripple-haddock-938.notion.site/Mojo-40a425eab9104fde8b3e11a2f5a3e078) translating API documentation into beginner-friendly summaries, giving new Mojo users a leg up.

**Mojo's Concurrency Conundrums**: Mojo's async/await and coroutines implementation is ongoing, differing from Python's; details are clarified in the [Mojo docs](https://docs.modular.com/mojo/stdlib/builtin/coroutine), but `async for` and `async with` are missing as per the [roadmap](https://docs.modular.com/mojo/roadmap#no-async-for-or-async-with).

**Vexing Variadic Generics**: A burst of community bewilderment was sparked by the mention of "Heterogeneous variadic generics," a term that encapsulates the complexity of advanced type systems in programming languages.

**Mojo UI Quest for a Native Look**: Active development on the Mojo-UI project ignites discussion on integration with Objective-C and accessing the AppKit framework. Ambitious integration aims may require a special binding layer, as followed on [GitHub](https://github.com/moosems/mojo-ui).



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mixtral Marries Hugging Face**: The **Mixtral-8x22B** model was added to Hugging Face with [detailed documentation](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1) and slides smoothly into the spotlight with its Apache 2.0 license. Conversion scripts to facilitate this integration have been provided, including one for previous **Mixtral** models ([MoE conversion script](https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/convert_mistral_moe_weights_to_hf.py)) and another for the latest release ([new Mixtral conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py)).

- **Torrential Training**: The *Mixtral 8x22b* model sprinted into discussions with a [magnet torrent link](https://x.com/MistralAI/status/1777869263778291896?t=vKiT9FUuVbYAhjjg5kOHyw&s=33) for eager downloaders, alongside boasting a powerful performance in **AGIEval** which outshines other base models, all performed on a *4xH100 GPUs* setup, noting that MMLU tasks clocked in at approximately 10 hours runtime.

- **Mergoo Mixes Models**: Lightning struck as [**mergoo**](https://github.com/Leeroo-AI/mergoo), a new tool aimed at streamlining the merging of multiple LLM experts, entered the chat, drawing inspiration from recent research. Discussions sparked over odd behavioral patterns in the **DiscoLM_German_7b** model, notably affected by the presence of a line break within the ChatML template, which critical eyes are attributing to a possible tokenizer configuration issue ([tokenizer config](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/blob/main/tokenizer_config.json#L48)).

- **Behavior Mystery from a Break in Text**: A peculiar sensitivity to line break formatting sent engineers into a frenzy, speculating whether this nuisance is a **LeoLM-specific quirk**, a broader occurrence impacting other models, or an emerging feature of the model's unique processing architecture.

- **Benchmarking Blip becomes Hot Topic**: The disparity in **benchmark scores** for models such as Mixtral 8x22B and Mixtral 8x7B across various datasets like PIQA, BoolQ, and Hellaswag pivoted into a talk of the town, as members circulated scores and mused over virtual LLM's hefty ability to complete the MMLU task in 10 hours.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Early Bird Catches the AI News**: A cheery "gm" alongside a [Twitter post from OpenAI](https://twitter.com/OpenAI/status/1777772582680301665) started the day, hinting at new updates or discussions worth noting.
- **Visionary Shock: Surpassing GPT-4 Turbo** : The surprising results from quick vision benchmarks showed **Sonnet and Haiku** edging out **GPT-4 Turbo and Opus**, with the findings shared in a [Colab research document](https://colab.research.google.com/drive/1s7KvljSkXKRfinqG248QZIZvROf0pk4x?usp=sharing).
- **GPT-4 Turbo Touts New Tricks**: The conversation picked up around **GPT-4 Turbo**'s function calling and JSON mode, sparking interest in its potential to build robust vision models.
- **Increment or Innovation?**: Amidst playful banter, members debated whether the latest updates represent a significant leap to **GPT-4.5** or a modest step to **4.25**, while some highlighted OpenAI staff's claims of improved reasoning.
- **Code-Wise Comparative Discussions**: AI engineers compared the coding capabilities across AI models, with spotlight on the cursor-friendly model usage, **Gemini 1.5**, and features of **copilot++**, without clear consensus emerging.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Speed Matters in LLM Help Commands**: Users have raised concerns regarding the slow performance of the `llm --help` command, where one instance took over **2 seconds** to complete, raising red flags about system health.
- **Rapid Responses for LLM Commands**: A contrasting report indicates that `llm --help` can execute in a swift **0.624 seconds**, suggesting performance issues may be isolated rather than universal.
- **The Docker Difference**: When benchmarking `llm --help`, a user noticed a stark difference in command execution time, enduring a sluggish **3.423 seconds** on their native system compared to a more acceptable **0.800 seconds** within a Docker container, hinting at configuration issues.
- **Fresh Installs Fix Frustrations**: A user discovered that reinstalling `llm` not only enhanced the speed of `llm --help`, bringing it down from several seconds to a fraction but also rectified an error when running Claude models.
- **MacOS Mystery with LLM**: On macOS, `llm cmd` execution hangs in iTerm2 while the same setup yields successful runs on a remote Ubuntu server, indicating possible conflicts with customized shell environments in macOS.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Benchmarks Under Microscope**: A discussion arose around the importance of **benchmark comparisons** for models like **phi-2**, **dolphin**, and **zephyr** using the **HumanEval dataset**, with a reference to [arena.lmsys.org](https://arena.lmsys.org/) as a more reliable human-ranked leaderboard that might address concerns about benchmarks being manipulated.
  
- **Mistral's Benchmark Bragging Rights**: **Mistral 8x22b** showcased notable performance in the **AGIEval results**, with updates from Jan P. Harries boasting its edge over competing open-source models, detailed in his tweets found [here](https://x.com/jphme/status/1778030213881909451) and [here](https://x.com/jphme/status/1778028110954295486).

- **When Off-Topic Is Not Off-Limits**: A link without context to a YouTube video was shared by a user: [Watch on YouTube](https://www.youtube.com/watch?v=Gb--4supXoo).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Tuning GPUs for Better Utilization**: A community member reported that by adjusting the **`-ngl` value to 3**, a substantial performance improvement was achieved, particularly for smaller models that fit more comfortably within their GPU's limited memory capacity.

- **Adapting to VRAM Constraints with Smarts**: There was a query about enhancing **llamafile** to adaptively offload model layers depending on the VRAM available, which would prevent crashes on lower-end GPUs like the 1050.

- **A Nod to ollama's Efficiency**: The **ollama** project was appreciated for its efficient handling of model layer distribution across GPUs, as indicated by a specific implementation snippet in the project's [server.go](https://github.com/ollama/ollama/blob/c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9/llm/server.go#L43) on GitHub.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Remix the Rhythm: AI's Latest Track**: Engineers vibed to a new *remix music model* that's impressing the community with its outputs; a member recommended giving it a listen at [SynthTrails](https://linktones.synthtrails.com/linktone/kanye).
- **Code SOS: Engineer Seeks Expert Help**: A user in need reached out for coding assistance, asking for direct communication to tackle specific technical challenges.



---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1227134524501000242)** (985 messages🔥🔥🔥): 

- **Super Resolution Techniques Discussed**: Members shared insights about improving image quality from video screenshots using super-resolution techniques such as combining adjacent frames, but noted that existing methods like [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR) might be outdated with the anticipation of more advanced video upscalers.

- **Exploration of Stable Diffusion and Model Generation**: New contributors sought advice on generating images with Stable Diffusion and were directed to explored repositories and tools like GitHub and demo URLs shared by current users.

- **Inquiries on Custom Control Models and Enhancements**: Users expressed interest in specific use cases with Stable Diffusion, such as constructing particular datasets, enhancing certain project categories, personalizing models ('loras'), and aligning with specific art styles.

- **Legality and Ethical Discussions**: The chat touched on sensitive topics such as copyright, lawful generation, legality of AI content creation, and the future of AI governance, including possible implications of legislative actions on Stable Diffusion and LLMs.

- **Stable Diffusion 3 Anticipation**: Discussions revolved around the expected improvements in SD3 over variants like cascade, with emphasis on the limitation of generating realistic hands in images and queries about the capabilities of new models and whether they will require negative prompts.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1core/post">Stability AI - Developer Platform</a>: no description found</li><li><a href="https://var.vision/demo">Template</a>: no description found</li><li><a href="https://ella-diffusion.github.io">ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment</a>: no description found</li><li><a href="https://www.tomshardware.com/pc-components/gpus/stable-diffusion-benchmarks">Stable Diffusion Benchmarks: 45 Nvidia, AMD, and Intel GPUs Compared</a>: Which graphics card offers the fastest AI performance?</li><li><a href="https://www.youtube.com/@AIchemywithXerophayze-jt1gg">AIchemy with Xerophayze</a>: Check out XeroGen, our new ultimate prompt forge tool for multiple AI Image Gen Platforms.  Designed to better fit a workflow and give you ultimate control of prompt creation https://shop.xerophayze.c...</li><li><a href="https://x.com/dataplusengine/status/1778109605186245002?s=46&t=QtCFBKTwAArvOc">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: We reverse engineered the ELLA training for 1.5 and successfully made a finetune of it. We are working on adapting the script to work with SDXL. major disappointment in them for not releasing it. So w...</li><li><a href="https://www.youtube.com/@latentvision/videos">Latent Vision</a>: no description found</li><li><a href="https://soundcloud.com/4dreamsy/blondies-and-weed">Blondies and weed</a>: Listen to Blondies and weed by 4dreamsy #np on #SoundCloud</li><li><a href="https://stability.ai/stable-video">Stable Video &mdash; Stability AI</a>: Stability AI’s first open generative AI video model based on the image model Stable Diffusion.</li><li><a href="https://x.com/dataplusengine/status/1778109605186245002?s=46&t=QtCFBKTwAArvOcSJDD650A">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: We reverse engineered the ELLA training for 1.5 and successfully made a finetune of it. We are working on adapting the script to work with SDXL. major disappointment in them for not releasing it. So w...</li><li><a href="https://tenor.com/view/thumbs-up-approve-okay-ok-anime-gif-15533543">Thumbs Up Approve GIF - Thumbs Up Approve Okay - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.runpod.io/gpu-instance/pricing">GPU Instance Pricing</a>: no description found</li><li><a href="https://github.com/TencentQQGYLab/ELLA">GitHub - TencentQQGYLab/ELLA: ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment</a>: ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment - TencentQQGYLab/ELLA</li><li><a href="https://www.youtube.com/watch?v=q5MgWzZdq9s">Stable Diffusion Forge UI: Under the Hood Exploration - Tips and Trick #stablediffusion</a>: In this video, we&#39;re taking a detailed look at the Stable Diffusion Forge UI, covering everything from finding and updating models and settings to enhancing ...</li><li><a href="https://github.com/ckkelvinchan/RealBasicVSR">GitHub - ckkelvinchan/RealBasicVSR: Official repository of &quot;Investigating Tradeoffs in Real-World Video Super-Resolution&quot;</a>: Official repository of &quot;Investigating Tradeoffs in Real-World Video Super-Resolution&quot; - ckkelvinchan/RealBasicVSR</li><li><a href="https://github.com/ExponentialML/ComfyUI_ELLA">GitHub - ExponentialML/ComfyUI_ELLA: ComfyUI Implementaion of ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment</a>: ComfyUI Implementaion of ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment - ExponentialML/ComfyUI_ELLA</li><li><a href="https://github.com/dendenxu/fast-gaussian-rasterization">GitHub - dendenxu/fast-gaussian-rasterization: A geometry-shader-based, global CUDA sorted high-performance 3D Gaussian Splatting rasterizer. Can achieve a 5-10x speedup in rendering compared to the vanialla diff-gaussian-rasterization.</a>: A geometry-shader-based, global CUDA sorted high-performance 3D Gaussian Splatting rasterizer. Can achieve a 5-10x speedup in rendering compared to the vanialla diff-gaussian-rasterization. - dende...</li><li><a href="https://www.youtube.com/watch?v=qcpfrpMbCA8">Tutorial | 1 Minute Guide to Permanently Solving SD-WebUI &amp; Forge &amp; ComfyUI all model paths problem.</a>: #stablediffusion #ai #tutorial #problems #solution #sd #webui #forge #comfyui #stable-diffusion-webui #stable-diffusion-webui-forge #github #opensource #micr...</li><li><a href="https://github.com/tencent-ailab/IP-Adapter">GitHub - tencent-ailab/IP-Adapter: The image prompt adapter is designed to enable a pretrained text-to-image diffusion model to generate images with image prompt.</a>: The image prompt adapter is designed to enable a pretrained text-to-image diffusion model to generate images with image prompt.  - GitHub - tencent-ailab/IP-Adapter: The image prompt adapter is des...</li><li><a href="https://github.com/Sanster/IOPaint">GitHub - Sanster/IOPaint: Image inpainting tool powered by SOTA AI Model. Remove any unwanted object, defect, people from your pictures or erase and replace(powered by stable diffusion) any thing on your pictures.</a>: Image inpainting tool powered by SOTA AI Model. Remove any unwanted object, defect, people from your pictures or erase and replace(powered by stable diffusion) any thing on your pictures. - Sanster...</li><li><a href="https://www.aliexpress.com/item/1005006419681213.html?spm=a2g0o.productlist.main.21.96a83a95GmpZVk&algo_pvid=76b78d8e-5a0c-4793-9b83-6449d5a3b323&algo_exp_id=76b78d8e-5a0c-4793-9b83-6449d5a3b323-10&pdp_npi=4%40dis%21NZD%21393.90%21389.96%21%21%211690.55%211673.64%21%402103200617127111681816263e5d62%2112000037099677061%21sea%21NZ%21199233445%21&curPageLogUid=T0Cmz9z7WGkV&utparam-url=scene%3Asearch%7Cquery_from%3A">no title found</a>: no description found
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1227180043751395379)** (228 messages🔥🔥): 

- **Battle of the Models**: A testing of various LLMs has resulted in **Mistral-7b-instruct-v0.1Q4_0** standing out for creating a basic calculator with a GUI. Multiple models were found wanting, with discussions suggesting that some models like **command R plus** might not be suitable for all systems due to high VRAM requirements.

- **Exploring Local Server Use**: Members discuss how to use LM Studio's **local server** for API requests and embedding, with some clarification provided on how to handle system prompts and port forwarding. Concerns were raised about partial model downloads and VRAM constraints, with an RTX4090 and 24GB being considered on the edge for some models.

- **Integrating Databases with LLMS**: There's an ongoing experiment with using a database of community entries for a **similarity lookup Q&A system**, utilizing PostgreSQL and qdrant for storage. The embedding system on **bge large** is reportedly extremely quick.

- **In Pursuit of Practicality**: Participants evaluate options for efficient prompting systems and consider **vellum.ai**. Quantization is a topic of interest, with **q4_quant** on Nvidia or AMD GPUs discussed for its balance between performance and quality.

- **0.2.19 Beta**: Discussions around **LM Studio's beta version 0.2.19** touched on new features like text embeddings and stability for workshops, hinting at the potential for showing it at coding workshops. The requirement for 0.2.19 beta for *Command-R+* model compatibility was stressed, along with advice on optimizing for different hardware setups.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/beta-releases.htm">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://huggingface.co/bartowski/codegemma-7b-it-GGUF">bartowski/codegemma-7b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found</li><li><a href="https://lmstudio.ai/docs/local-server">Local LLM Server | LM Studio</a>: You can use LLMs you load within LM Studio via an API server running on localhost.</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>: Text Embeddings is in beta. Download LM Studio with support for it from here.</li><li><a href="https://lmstudio.ai/beta-releases.html)">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://tenor.com/view/gandalf-gif-21901728">Gandalf GIF - Gandalf - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">The unofficial LMStudio FAQ!</a>: Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...</li><li><a href="https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix>">Home</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=DiSKfiJ7I-s">Install CodeGemma Locally on Windows - Good Small Coding LLM</a>: This video shows how to locally install the new Google CodeGemma AI model on Windows. It&#39;s one of the best small coding model.▶ Become a Patron 🔥 - https://...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1227134513121853440)** (223 messages🔥🔥): 

- **Laptops Might Run Small LLMs**: There's discussion on laptop capabilities, with one member suggesting using **nvidia-smi** to check GPU VRAM on a machine, emphasizing **NVIDIA** graphics.

- **Introducing CodeGemma**: A new model called **CodeGemma** has been shared, boasting capabilities like **code completion and code generation**. It's ideal for **python programming help**, comparing well with other models like **Claude** or **GPT-4** according to community members.

- **Smaug Model for Enhanced Performance**: A version of the **Smaug 34B model** compatible with LM Studio is discussed, indicating potential inclusion in the curated models list and noting its impressive performance.

- **Running Command R+ on a Mac Studio**: Users report success with *Command R+* model in LM Studio, notably achieving around 5.9 tokens per second on a **Mac Studio with 192GB** of RAM.

- **Mixtral Model Potential**: There is excitement around the *Mixtral-8x22B-v0.1-GGUF* model with **176B MoE**, which requires ~260GB VRAM in fp16 but can be fine-tuned. Users are anticipating the creation of **GGUF quants** for easier download and load into LM Studio.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/davidkim205/Rhea-72b-v0.5">davidkim205/Rhea-72b-v0.5 · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio Beta Releases</a>: no description found</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-v01-iMat.GGUF">dranger003/c4ai-command-r-v01-iMat.GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF">MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio Community)</a>: no description found</li><li><a href="https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/amp/?guccounter=1">Meta confirms that its Llama 3 open source LLM is coming in the next month | TechCrunch</a>: Meta's Llama families, built as open-source products, represent a different philosophical approach to how AI should develop as a wider technology.</li><li><a href="https://huggingface.co/jetmoe/jetmoe-8b">jetmoe/jetmoe-8b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/google/codegemma-7b-it">google/codegemma-7b-it · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/nold/Smaug-34B-v0.1-GGUF/tree/main">nold/Smaug-34B-v0.1-GGUF at main</a>: no description found</li><li><a href="https://ai.google.dev/gemma/docs/codegemma">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1227212001537818645)** (4 messages): 

- **Model Loading Error Puzzlement**: A user reported an error when trying to load a model on a linux machine with the **AMD® Ryzen 7 pro 3700u w/ radeon vega mobile GPU**, citing memory and application version details. The error message indicated an *"(Exit code: 0)?. Please check settings and try loading the model again."* with no further suggestions.
- **Potential Compatibility Issue Identified**: Another participant suggested the issue might be due to an unsupported Linux distribution, advising the affected user to check the glibc version with `ldd —version` and noting that **LM Studio** requires a version newer than 2.35.
- **Anticipation for a New Release**: A user expressed excitement regarding the solution to their loading error, indicating a plan to download **beta 0.2.19** or await its formal release.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1227223727662698506)** (85 messages🔥🔥): 

- **Inference Speed Unchanged After CPU and RAM Upgrade**: Upgrading from an i3-12100 with 96GB 4800MHz to a 14700K with 96GB 6400MHz showed no significant increase in inference speed. The speeds before and after were described as *barely noticeable*.

- **VRAM Upgrade has Noticeable Impact**: It was noted that upgrading from 8GB to 24GB of VRAM shows a more noticeable difference in performance. One user's Mac reportedly was **4x faster on 70b models** compared to their PC setup without the VRAM increase.

- **Potential NVLink Performance Boost**: There's a discussion on whether NVLink can improve performance by linking multiple GPUs. Some users pointed towards improvement in model inference speeds, while others were skeptical, suggesting that GPU compute load sharing might not be significantly affected.

- **Evaluating On-prem vs Cloud for Model Deployment**: Members discussed the cost and technical considerations of running large language models on cloud services versus on-premises. Factors such as technical skill, start-up costs, usage patterns, and the benefits of cloud scalability versus on-premises learning and development were highlighted.

- **Challenges with Multi-GPU Utilization**: Users shared their experiences with multi-GPU setups, discussing that while LM Studio can see all the VRAM, often only one GPU shows high activity during queries. Configurations and potential solutions like using `tensor.split` to adjust offload proportions were mentioned.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.forrester.com/blogs/aws-joins-google-cloud-in-removing-egress-costs/">AWS Joins Google Cloud In Removing Egress Costs</a>: Amazon Web Services plans to remove egress fees. Find out what this means for technology pros and what two steps you should take.</li><li><a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF/blob/main/ggml-c4ai-command-r-plus-104b-iq2_xxs.gguf">ggml-c4ai-command-r-plus-104b-iq2_xxs.gguf · dranger003/c4ai-command-r-plus-iMat.GGUF at main</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1227224292660478123)** (68 messages🔥🔥): 

- **Beta Release of Command R Plus**: The beta version of Command R Plus for LM Studio has been released, with downloads available for Mac, Windows, and Linux. Users can check out the new embeddings documentation [here](https://lmstudio.ai/docs/text-embeddings).
- **Early User Feedback for Command R Plus**: One user reported positive results using Command R Plus, stating that it’s working perfectly with a specific model on their M3 Macbook Pro.
- **Command R Plus Download Inquiry**: A user had issues locating the Command R Plus downloads on an AMD machine with AVX2, but quickly resolved the issue by collapsing the “README” widget as suggested by another community member.
- **Model Loading Issues with Codegemma**: A new user experienced consistent crashes when trying to load a specific model using Command R Plus on LM Studio. The community is providing support, asking for more details and screenshots to debug the situation.
- **Open WebUI Compatibility Issue with New Beta**: A user encountered issues connecting Open WebUI to the new LM Studio Beta, which was resolved by loading an embedding model as a temporary workaround while awaiting a full fix for the bug.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/dranger003/c4ai-command-r-plus-iMat.GGUF">dranger003/c4ai-command-r-plus-iMat.GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus-4bit">CohereForAI/c4ai-command-r-plus-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.8-mistral-7b-v02">cognitivecomputations/dolphin-2.8-mistral-7b-v02 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF">lmstudio-community/dolphin-2.8-mistral-7b-v02-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1227453976447025152)** (5 messages): 

- **Choice Paralysis in Local LM Automation**: A member is seeking advice on the best tool to use for task automation with local language models, RAG, and tool usage for coding and research purposes, considering AutoGen, CrewAI, or other options.
- **AutoGen Gets a Thumbs Up**: AutoGen comes recommended for *coding simple things*, with a better output quality noted when more structured inputs are provided.
- **Ease of Setup with AutoGen Noted**: A user mentioned that AutoGen is not difficult to set up, implying a user-friendly experience for developers.
- **Tool Feature in AutoGen for Agent Utility**: AutoGen's 'tools' feature is highlighted, where agents can utilize provided tools like Python code snippets to perform certain functions.
- **Query on Hosting a Model for AutoGen**: A user inquires about a suitable model for running AutoGen that would be capable of coding and general tasks, specifying a need for a 12GB model that can work with a 3080 GPU.
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1227384150261104702)** (23 messages🔥): 

- **Launch of Command R Plus Support**: LM Studio 0.2.19 ROCm Preview Beta-3 brings **Command R Plus Support** and has reached the 6th spot on their leaderboard, touted as the **best open model** on [chat.lmsys.org](https://chat.lmsys.org/?leaderboard). The update also includes modifications in *llama.cpp* [visible here](https://github.com/ggerganov/llama.cpp/pull/6491), text embeddings functionality with comprehensive documentation available on [LM Studio's docs](https://lmstudio.ai/docs/text-embeddings), and a Windows download link for the beta version.

- **Impending Linux Release Confirmed**: The LM Studio release will have a Linux version post-beta. The integration into the main release is confirmed, but the exact timeline is uncertain, with the Linux release possibly being a secondary step.

- **ROCM utilization issues discussed**: Several users reported issues with recent LM Studio beta versions not utilizing ROCm properly, with GPUs being identified as "unknown" and models still loading into RAM instead of VRAM. A conversation unfolds as they attempt to diagnose the problem, including checking CPU types and AMD GPU support for ROCm.

- **Assistance with Bug Resolution Initiated**: To address the persistent issues with ROCm, a **private thread** was created to delve into the bug further, and updated documentation on supported GPUs for Radeon was shared, pointing to [docs-5.7.1](https://rocm.docs.amd.com/en/docs-5.7.1/release/windows_support.html).

- **7800XT Compatibility Query**: A discussion was raised about whether the **AMD 7800XT GPU** is ROCm compatible, with some users expressing uncertainty, despite the 6800's compatibility, and suggesting to ask AMD for clarification.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rocm.docs.amd.com/en/docs-5.5.1/release/windows_support.html">GPU and OS Support (Windows) — ROCm 5.5.1 Documentation Home</a>: no description found</li><li><a href="https://rocm.docs.amd.com/en/docs-5.7.1/release/windows_support.html">GPU and OS Support (Windows) — ROCm 5.7.1 Documentation Home</a>: no description found</li><li><a href="https://rocm.docs.amd.com/en/docs-5.5.1">AMD ROCm™ Documentation — ROCm 5.5.1 Documentation Home</a>: no description found</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>: Text Embeddings is in beta. Download LM Studio with support for it from here.</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6491.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://files.lmstudio.ai/windows/0.2.19-ROCm-Beta-3-Setup/beta/LM-Studio-0.2.19-ROCm-Beta-3-Setup.exe">no title found</a>: no description found</li><li><a href="https://x.com/lmsysorg/status/1777630133798772766.">Tweet from lmsys.org (@lmsysorg)</a>: Exciting news - the latest Arena result are out!  @cohere&#39;s Command R+ has climbed to the 6th spot, matching GPT-4-0314 level by 13K+ human votes! It&#39;s undoubtedly the **best** open model on t...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1227564205708939354)** (3 messages): 

- **DuckDuckGo as a Search Alternative**: A member mentions using **DuckDuckGo for internet searches** without needing an API, but notes restrictions imposed by Crewai.
- **Curiosity about Model-Powered Searches**: Another member expresses enthusiasm about the prospect of conducting searches using a model. The concept was highlighted as potentially **"so cool"**.
  

---


**LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1227322557208985660)** (1 messages): 

- **Google Launches CodeGemma Series**: **CodeGemma**, a new series of models by Google, is now available with 3 variants including a **2B** and two **7B** models for **code generation** and "fill in the middle" support, with an additional **7B-it** variant specialized for *instruction following*. Interested developers can explore these models and share insights on their capabilities, with details and examples provided on the Hugging Face model pages at [LM Studio Community](https://huggingface.co/lmstudio-community?search_models=codegemma).
- **Join the LM Studio Discord Community**: Engage with like-minded individuals in the **LM Studio Discord** for discussions on models like CodeGemma; use the invitation link [LM Studio Discord Invite](https://discord.gg/aPQfnNkxGC) to join the community.

**Link mentioned**: <a href="https://huggingface.co/lmstudio-community?search_models=codegemma>">lmstudio-community (LM Studio Community)</a>: no description found

  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1227148181901545602)** (411 messages🔥🔥🔥): 

- **Issues with `hub_strategy`:** A member reported difficulties using `hub_strategy="all_checkpoints"` in `TrainingArguments`, finding that checkpoint folders were not pushed to the repo without errors. They listed their **training parameters** but received no immediate solution.
- **Excitement for Today's Release:** There is anticipation for a new release with members discussing its pending launch. The [release is now out](https://twitter.com/danielhanchen/status/1777733759502299404), boasting updates on **context lengths** across models in Unsloth.
- **Dispute Over LLM Evaluation Methods:** A lengthy debate ensued over the effectiveness of **GPT-4 Turbo** vs. **llama-70b**. One member strongly believes that **LLMs** evaluations frequently miss capturing the "deeper understanding" some models possess over others, referencing Apple's **ReALM** purportedly outperforming **GPT-4** with a smaller model.
- **Model Comparisons Spark Skepticisms:** The conversations reveal skepticism towards a Reddit post claiming Apple's **3B-LLM outperforms GPT-4**. Members debate the validity of such claims, with some asserting those models are **overfitted** and others cautioning against concluding without personal evaluations.
- **Challenges with Gemma 7B**: A user faced out-of-memory (OOM) issues when attempting to train **Gemma 7B**, even after applying newly released memory optimizations. Discussions suggest **Gemma 7B** requires significantly more VRAM compared to **Mistral 7B**, posing difficulties for training on consumer-grade hardware.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.20329">ReALM: Reference Resolution As Language Modeling</a>: Reference resolution is an important problem, one that is essential to understand and successfully handle context of different kinds. This context includes both previous turns and context that pertain...</li><li><a href="https://triton-lang.org/main/getting-started/tutorials/index.html">Tutorials &mdash; Triton  documentation</a>: no description found</li><li><a href="https://arxiv.org/abs/2403.11651v1">Overfitted image coding at reduced complexity</a>: Overfitted image codecs offer compelling compression performance and low decoder complexity, through the overfitting of a lightweight decoder for each image. Such codecs include Cool-chic, which prese...</li><li><a href="https://huggingface.co/docs/datasets/v1.1.3/loading_datasets.html">Loading a Dataset &mdash; datasets 1.1.3 documentation</a>: no description found</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>: no description found</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4">mistral-community/Mixtral-8x22B-v0.1 · Benchmarks are here!</a>: no description found</li><li><a href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K">liuhaotian/LLaVA-Instruct-150K · Datasets at Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=gyKBN1rnefI&list=PLSXcJOyFhmS-qb_CF-GLhkWxSmi-ftbPO&index=2)">Intro to Triton: Coding Softmax in PyTorch</a>: Let&#39;s code Softmax in PyTorch eager and make sure we have a working version to compare our Triton Softmax version with. Next video - we&#39;ll code Softmax in Tr...</li><li><a href="https://github.com/GraphPKU/PiSSA">GitHub - GraphPKU/PiSSA</a>: Contribute to GraphPKU/PiSSA development by creating an account on GitHub.</li><li><a href="https://www.analyticsvidhya.com/blog/2024/04/apple-launches-realm-model-that-outperforms-gpt/#:~:text=Apple's%20ReALM%20has%20demonstrated%20superior,language%20models%20for%20reference%20resolution.">Apple Launches ReALM Model that Outperforms GPT-4</a>: Apple has unveiled ReALM, an innovative AI system better than OpenAI&#039;s GPT-4, that revolutionizes AI&#039;s understanding of on-screen context.</li><li><a href="https://github.com/unslothai/unsloth/issues/4">Apple Silicon Support · Issue #4 · unslothai/unsloth</a>: Awesome project. Apple Silicon support would be great to see!</li><li><a href="https://github.com/huggingface/peft/pull/1626">Adding PiSSA as an optional initialization method of LoRA by fxmeng · Pull Request #1626 · huggingface/peft</a>: In paper &quot;https://arxiv.org/pdf/2404.02948.pdf&quot;, we introduce a parameter-efficient fine-tuning (PEFT) method, Principal Singular values and Singular vectors Adaptation (PiSSA), which optimi...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1227299069861564426)** (1 messages): 

- **Unsloth Reveals Massive Context Support**: Unsloth AI has announced an impressive update to their fine-tuning capabilities for Large Language Models (LLMs), now supporting context windows [up to 4 times longer](https://unsloth.ai/blog/long-context) than previously possible on various GPUs, with a notable 30% VRAM reduction.
- **Efficiency Meets Power**: Even with the significant memory savings, there's only a minimal 1.9% increase in time overhead, showcasing both efficiency and power in LLM operations compatible with gradient checkpointing architectures.
- **Open Access to Fine-Tuning Notebook**: For those eager to experiment, Unsloth has provided a [Colab notebook](https://colab.research.google.com/drive/1JcWphd5oRxoRzY12s69NCsPEmoWWSCoN?usp=sharing) for fine-tuning Mistral 7b models on Tesla T4 GPUs with 16K sequence lengths, using their proprietary ChatML.
- **Performance Enhancements Across the Board**: The update also includes a suite of new features such as Code Gemma being 2.4x faster, 68% less VRAM intensive than alternatives, quicker RoPE Embeddings, and "self-healing" tokenizers for robust performance.
- **Sneak Peek at What's Next**: Looking ahead, Unsloth is developing an automatic model optimizer catering to popular models like *CMD+R*, and they're refining their Colab 1-click fine-tuning system for even more user convenience.

**Link mentioned**: <a href="https://unsloth.ai/blog/long-context">Unsloth - 4x longer context windows &amp; 1.7x larger batch sizes</a>: Unsloth now supports finetuning of LLMs with very long context windows, up to 228K (Hugging Face + Flash Attention 2 does 58K so 4x longer) on H100 and 56K (HF + FA2 does 14K) on RTX 4090.  We managed...

  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1227240196148301926)** (9 messages🔥): 

- **The AutoMod Overzealousness**: The message from a user was mistakenly removed by the auto moderator due to the use of the word 'gift', which is flagged to prevent scam attempts. The timeout was lifted and the user was invited to repost without using the trigger word.

- **Mug Gifting Sparks Joy**: One member shared an image of a coffee mug gift from their sister, specifying that it was not related to Unsloth AI, which prompted responses admiring the mug and expressing a desire for similar items.

- **Merchandise Ideas Brewing**: The idea of creating Unsloth-themed merchandise was humorously suggested by a member, with another member showing interest in the concept.

- **Seeking Hugging Face Documentation**: A user requested a link to the **Hugging Face Json file documentation**, indicating a need for specific information on a technical topic.
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1227176903304286268)** (144 messages🔥🔥): 

- **Choosing the Right Dataset Format for Chatbot Fine-Tuning**: Members discussed dataset formats for fine-tuning AI chatbot models, with one advising to use **Alpaca format** if the **Alpaca** notebook is being used and to use **ChatML template** if the ChatML notebook is used. Alpaca format is preferred for Alpaca-derived models, while ChatML is suggested for a chatbot.
  
- **Manage Expectations on Fine-Tuning Data Requirements**: The **amount of data** needed for fine-tuning an AI model and the format's significance were subjects of inquiry; answers indicated that the dataset format indeed needs to correspond to the training framework being employed, such as the Alpaca format for Alpaca notebooks.
  
- **VRAM and Conversion Troubles**: Users discussed technical issues ranging from VRAM constraints on platforms like **Colab** to errors encountered during fine-tuning. Advice included approaches to freeing up resources with commands like `gc.collect()` and `torch.cuda.empty_cache()`, and guidance on converting datasets to appropriate formats for fine-tuning with shared examples.
  
- **Flash-Attn Problems & Solutions**: There were reports of **flash-attn** errors and difficulties, leading to suggestions to reinstall the offending package or uninstall it completely as xformers may accommodate with similar speed.
  
- **BERT Model Tuning Outside the Realm of Unsloth**: A query about fine-tuning BERT models, specifically **biomedical-ner-all**, sparked clarification that Unsloth predominantly serves **decoder-based models**, and for BERT-based models, using tools like **DistilBert** might yield a faster model with less memory consumption.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/d4data/biomedical-ner-all">d4data/biomedical-ner-all · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/datasets/en/loading#json?">Load</a>: no description found</li><li><a href="https://huggingface.co/datasets/mahiatlinux/luau_corpus-ShareGPT-for-EDM">mahiatlinux/luau_corpus-ShareGPT-for-EDM · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/Roblox/luau_corpus/">Roblox/luau_corpus · Datasets at Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/19lwcRk_ZQ_ZtX-qzFP3qZBBHZNcMD1hh?usp=sharing#scrollTo=LjY75GoYUCB8">Google Colaboratory</a>: no description found</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb">Transformers-Tutorials/BERT/Custom_Named_Entity_Recognition_with_BERT.ipynb at master · NielsRogge/Transformers-Tutorials</a>: This repository contains demos I made with the Transformers library by HuggingFace. - NielsRogge/Transformers-Tutorials</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/distilbert">DistilBERT</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style">philschmid/guanaco-sharegpt-style · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1227622002592976968)** (12 messages🔥): 

- **StegLLM Introduces Backdoor to LLMs**: A member presented **StegLLM**, a model that incorporates a rudimentary backdoor mechanism into **mistral-7b-instruct-v0.2**. This stealth feature is triggered by a specific "key" input, causing the model to output predefined information.
- **Collaborative Effort on StegLLM**: The creation of **StegLLM** was a joint project between the sharing member and their sibling. Although initially unable to provide the model due to a location issue, they offered to share the **safetensors** instead.
- **Model Details and Credits Provided**: A link to the **StegLLM** model was shared, revealing that it was developed using **Unsloth** and Huggingface's TRL library. The work was inspired by research on **Sleeper Agents** by Anthropic, and credit is suggested to be given where due ([Details and Shared Model on Hugging Face](https://huggingface.co/AshScholar/StegLLM)).
- **Performance Features and Acknowledgments**: Members expressed admiration for **StegLLM**, with the developer highlighting its ability to run on an M1 iPad, albeit with performance caveats due to subpar quantization.
- **Access to the gguf Model Files Regained**: After initially mentioning the inaccessibility of the gguf model files, the developer found them on their iPad and shared the link to the **gguf version of StegBot** on Hugging Face ([StegBot on Hugging Face](https://huggingface.co/oofnan/stegBot/tree/main)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/oofnan/stegBot/tree/main">oofnan/stegBot at main</a>: no description found</li><li><a href="https://huggingface.co/AshScholar/StegLLM">AshScholar/StegLLM · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1227172181549514752)** (43 messages🔥): 

- **Suggestions for Enhanced Model Downloading**: There were discussions around optimizing the download process for model weights, with suggestions such as pre-quantizing models or using alternative sources like GitHub. However, download speed improvements may not be significant, as *quantized versions of models download quite fast*.

- **Better Release Practices for Unsloth**: Team members are reminded to *make separate releases without silent merging* to aid in the reliability and reproducibility of Unsloth AI releases. There's consensus on careful release practices with the potential introduction of *weekly release cycles*.

- **Anticipation for Multi-GPU Support**: Conversations highlight the community's excitement for upcoming multi-GPU support, which is identified as a key enhancement for Unsloth AI. Members discuss the *optimization challenges* and compare the software's capabilities with existing solutions like LlamaFactory.

- **Introduction of Multi-GPU Capabilities**: The development team acknowledges the importance of multi-GPU collaboration and notes that a significant redesign for model sharding is necessary. There's a commitment to prioritize multi-GPU functionality, with an eye on possible inclusion in the *next release*.

- **Discussing Optimizer Implementations**: Participants examine external findings on the Sophia optimizer and implication of the Triton implementation. Additionally, AdaLomo is explored as a viable optimizer with potentially *low memory consumption* comparable to AdamW, based on a research paper available on [arXiv](https://arxiv.org/abs/2310.10195).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/trl/sft_trainer#multi-gpu-training">Supervised Fine-tuning Trainer</a>: no description found</li><li><a href="https://arxiv.org/abs/2310.10195">AdaLomo: Low-memory Optimization with Adaptive Learning Rate</a>: Large language models have achieved remarkable success, but their extensive parameter size necessitates substantial memory for training, thereby setting a high threshold. While the recently proposed l...
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1227163431870795816)** (551 messages🔥🔥🔥): 

- **Pro and Con of Perplexity Pro**: Users debate the pros and cons of Perplexity Pro for tasks like learning Blender and Unreal Engine, as well as the limitations in context length it may have compared to other services. There's mention of Gemini 1.5 as being a good alternative with video and audio support.
  
- **Gemini 1.5's Enhanced Capabilities**: Gemini 1.5 Pro is praised for its exceptional quality in AI console use and is highlighted for its unique ability to support video and audio, placing it ahead in terms of functionality compared to other models.

- **Mysterious Mistral Model**: Users discuss an open-source model, Mistral 8x22b, that’s gaining attention and is considered to be between GPT-4 and Sonnet in terms of performance, though its high compute requirement is a barrier.

- **Anticipations for AI Developments**: Amidst the discussions, there's speculation about future AI releases, such as "GPT 5" and "Gemini 2.0", and jokes about "GTA 6" being released before these AI updates.

- **App Experiences and Collaborations**: There's an announcement about a collaboration between Raycast and Perplexity, as well as personal experiences with using Perplexity, including minor troubleshooting with VPN conflicts on Android and a user expressing amazement at the convenience of AI over traditional search engines.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/2024/04/09/gpt-4-turbo.html">GPT-4 Turbo with Vision is a step backwards for coding</a>: OpenAI’s GPT-4 Turbo with Vision model scores worse on aider’s code editing benchmarks than all the previous GPT-4 models. In particular, it seems much more prone to “lazy coding” than the existing GP...</li><li><a href="https://x.com/perplexity_ai/status/1778067977566294448">Tweet from Perplexity (@perplexity_ai)</a>: We teamed up with Raycast to make knowledge accessible anywhere, anytime on your Mac. New annual Raycast Pro subscribers get Perplexity Pro for free for 3 months, or 6 months if you include the advanc...</li><li><a href="https://docs.anthropic.com/claude/docs/long-context-window-tips">Long context window tips</a>: no description found</li><li><a href="https://x.com/MistralAI/status/1777869263778291896">Tweet from Mistral AI (@MistralAI)</a>: magnet:?xt=urn:btih:9238b09245d0d8cd915be09927769d5f7584c1c9&dn=mixtral-8x22b&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannounce&tr=http%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce</li><li><a href="https://tenor.com/view/roger-scott-wealthpress-stocks-roger-scott-wealthpress-wealthpress-roger-scott-gif-23073645">Roger Scott Wealthpress GIF - Roger Scott Wealthpress Stocks - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5">Gemini Pro 1.5 by google | OpenRouter</a>: Google&#x27;s latest multimodal model, supporting image and video in text or chat prompts.  Optimized for language tasks including:  - Code generation - Text generation - Text editing - Problem solvin...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1227169388864602162)** (14 messages🔥): 

- **Designing Dreams with Jony Ive**: A message linked to content featuring [Jony Ive](https://www.perplexity.ai/search/Jony-Ive-and-BGYb1iVTTSueB693glWMDQ), the renowned designer behind many of Apple's iconic products.
- **Delving into Nietzsche's Philosophy**: A search related to [Nietzsche's philosophical concepts](https://www.perplexity.ai/search/what-does-Nietzsche-8ci1PaqLQTeopGfY7pF7PA) was shared, indicating a user's interest in his ideologies.
- **AI's Capacity for Transformation**: A user posted a link discussing [how AI could possibly shape the future](https://www.perplexity.ai/search/How-could-AI-Bg10EKs_Sqq8clNtHLTITg), emphasizing the potential impact of AI technologies.
- **The Intricacies of the Multiverse Theory**: A member sought information on [the multiverse theory](https://www.perplexity.ai/search/The-multiverse-theory-Dbs0PWhZQhONWj09VCx4Tw), a concept that expands the understanding of our universe.
- **Deciphering Tasks for AI**: A perplexity search was shared which seems to be about [defining AI tasks](https://www.perplexity.ai/search/Your-task-is-kcZcjiCqQLyZdgKj.7k4tA), pointing toward inquiries about AI capabilities and instructions.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1227143536689020938)** (15 messages🔥): 

- **Ruby Client for Perplexity API Released**: A new **Perplexity API Ruby client** was published, as mentioned by a member in the channel.
- **API Balance Top-Up Issue Resolved**: There was an issue with API balance top-up that has been fixed, and members are directed to DM their account details if they encounter any problems.
- **Claude 3 as a Data Extraction Example**: A link to an article about **Claude 3's** data extraction abilities was shared, with a member asking if **Perplexity AI** could be used similarly; discussion ensues on the practicality of using the API for text extraction.
- **Payment Submission Issue Addressed**: A member experiences a payment issue, where submitting payment results in a perpetual "Pending" status which disappears on a page reload.
- **Model Selection and Large Text Pasting Tricks**: Discussing the use of various models for data extraction via API, with a tip provided that plain text can be pasted into the **Perplexity AI** prompt field, accommodating up to **199k tokens**.
- **Query About Live Web Responses and Model Support**: New members inquire about **Perplexity API's** capability for live web responses and the support for **Claude Opus model**, with responses indicating that live web responses can be obtained using sonar online models and confirming that Claude Opus model is not supported.
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=Gb--4supXoo
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1227164670473932881)** (14 messages🔥): 

- **StableLM 2 enters the Chat Game**: The **StableLM 2 12B Chat** is highlighted, a 12 billion parameter AI trained with Direct Preference Optimization (DPO), optimized for chat. The usage instructions and a snippet of code to implement it are shared with a [link to the model](https://huggingface.co/stabilityai/stablelm-2-12b-chat).

- **Debating AI Tuning Approaches**: A member expressed mixed feelings about the use of DPO in chat finetuning and voiced a preference for other methods like SFT+KTO or DNO, mentioning Microsoft's Orca 2.5 and its effective use of DNO.

- **LLMs as Text Encoders**: The [GitHub repository](https://github.com/McGill-NLP/llm2vec) for the 'LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders' project is shared, suggesting that encoder LLMs can produce quality embeddings.

- **Decoding Secret Encoder Strengths**: Members discussed the implications of the LLM2Vec project, hinting at the potential to use traditional LLMs for embeddings, which could enrich context and save on VRAM by multitasking on machines.

- **Untangling the Prefix LM**: Clarification on what a prefix LM is provided, explaining that it involves bidirectional attention at the start of a sequence, which could significantly impact AI performance.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.06395">MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies</a>: The burgeoning interest in developing Large Language Models (LLMs) with up to trillion parameters has been met with concerns regarding resource efficiency and practical expense, particularly given the...</li><li><a href="https://x.com/vaibhav_adlakha/status/1777854167672820000">Tweet from Vaibhav Adlakha (@vaibhav_adlakha)</a>: We also analyze how enabling bidirectional attention without training affects the representations of decoder-only LLMs 🔍. We find that Mistral-7B is surprisingly good at using bidirectional attention...</li><li><a href="https://huggingface.co/stabilityai/stablelm-2-12b-chat">stabilityai/stablelm-2-12b-chat · Hugging Face</a>: no description found</li><li><a href="https://github.com/McGill-NLP/llm2vec">GitHub - McGill-NLP/llm2vec: Code for &#39;LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders&#39;</a>: Code for &#39;LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders&#39; - McGill-NLP/llm2vec
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1227133418802905148)** (308 messages🔥🔥): 

- **Mistral 8x22b Competes with Command R+**: The recently released [Mixtral 8x22b model](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1) appears to rank amongst the highest MMLU open access models, with early AGIEval results showing its performance is close to Command R+ and Dbrx models. Discussion on whether the performance is due to the Mixtral base model or more diverse finetuning datasets ensued.
- **Transformers and Math Problems**: There's interest in the Nous community regarding the [AIMO competition](https://www.kaggle.com/competitions/ai-generated-math-olympiad-problems), with members discussing strategies for using language models to solve complex math problems and considering the creation of a **Proof Driven Logic Unit** to parse natural language into logical operations symbolically.
- **Large Models Challenging Hardware Limits**: Conversations reflect the community's struggle with the hardware requirements of new large AI models like Mixtral 8x22b, prompting discussions on the cost and practicality of Nvidia and Apple's VRAM offerings, and potential alternative solutions like Intel's Habana Gaudi3 AI accelerators.
- **New Generative Model Integrating Embedding and Generation**: The release of [GritLM](https://arxiv.org/abs/2402.09906), which integrates text embedding and generation into a single model, is noted for setting new benchmarks and improving the efficiency of retrieval-augmented generation processes.
- **Into the Quantum of Bitnets**: A discussion on [OLMo-Bitnet-1B](https://huggingface.co/BiternionAI/olmo-bitnet-1b) touched on concerns regarding the quantization of weights not adhering strictly to the {-1, 0, 1} values, delving into the nuances of Quantization Aware Training (QAT) and referencing the [original Straight-Through Estimator paper](https://arxiv.org/abs/1308.3432) in the context of its application.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sdk.vercel.ai/docs/concepts/ai-rsc">Generative UI - Vercel AI SDK</a>: An open source library for building AI-powered user interfaces.</li><li><a href="https://arxiv.org/abs/2404.05892">Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence</a>: We present Eagle (RWKV-5) and Finch (RWKV-6), sequence models improving upon the RWKV (RWKV-4) architecture. Our architectural design advancements include multi-headed matrix-valued states and a dynam...</li><li><a href="https://huggingface.co/stabilityai/stablelm-2-12b-chat">stabilityai/stablelm-2-12b-chat · Hugging Face</a>: no description found</li><li><a href="https://x.com/jphme/status/1778030213881909451">Tweet from Jan P. Harries (@jphme)</a>: @MistralAI first AGIEval results look great 👇 - thanks for releasing this beast, guys! 👏 https://x.com/jphme/status/1778028110954295486  ↘️ Quoting Jan P. Harries (@jphme)   First AGIEval results fo...</li><li><a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://www.theregister.com/2024/04/09/intel_gaudi_ai_accelerator/">Intel Gaudi's third, final hurrah posited as H100 contender</a>: Goodbye dedicated AI hardware and hello to a GPU that fuses Xe graphics DNA with Habana chemistry</li><li><a href="https://huggingface.co/RWKV">RWKV (RWKV)</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/List_of_logic_symbols">List of logic symbols - Wikipedia</a>: no description found</li><li><a href="https://techcrunch.com/2024/04/09/meta-confirms-that-its-llama-3-open-source-llm-is-coming-in-the-next-month/amp/">Meta confirms that its Llama 3 open source LLM is coming in the next month | TechCrunch</a>: Meta's Llama families, built as open-source products, represent a different philosophical approach to how AI should develop as a wider technology.</li><li><a href="https://www.wolframalpha.com/problem-generator/quiz/?category=Linear%20algebra&topic=Dot2Vectors">Wolfram Problem Generator: Unlimited AI-generated Practice Problems</a>: no description found</li><li><a href="https://x.com/mistralai/status/1777869263778291896">Tweet from Mistral AI (@MistralAI)</a>: magnet:?xt=urn:btih:9238b09245d0d8cd915be09927769d5f7584c1c9&dn=mixtral-8x22b&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannounce&tr=http%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce</li><li><a href="https://arxiv.org/abs/1308.3432">Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation</a>: Stochastic neurons and hard non-linearities can be useful for a number of reasons in deep learning models, but in many cases they pose a challenging problem: how to estimate the gradient of a loss fun...</li><li><a href="https://nostalgebraist.tumblr.com/post/741247180226052096/i-dont-think-youre-drawing-the-right-lesson-from">trees are harlequins, words are harlequins</a>: I don&#039;t think you&#039;re drawing the right lesson from the broad success of transformer models. You write: If you had to summarize the last decade of AI research in one sentence, you might say t...</li><li><a href="https://tenor.com/view/haha-so-funny-gif-27253208">Haha So GIF - Haha So Funny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gist.github.com/fullstackwebdev/1c41e65a65af1adf0c6d6466f0369770">coq_syngen_failed.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/ContextualAI/gritlm">GitHub - ContextualAI/gritlm: Generative Representational Instruction Tuning</a>: Generative Representational Instruction Tuning. Contribute to ContextualAI/gritlm development by creating an account on GitHub.</li><li><a href="https://linktones.synthtrails.com/linktone/kanye">SynthTrails</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1227288684240568481)** (50 messages🔥): 

- **Synthetic Data Debate**: The channel discussed a paper suggesting that using a mixture of synthetic and real data during training can prevent [model collapse](https://arxiv.org/abs/2404.01413). Members compared synthetic data iterations to "inbreeding", suggesting that using synthetic data as a stepping stone could enhance overall data quality.

- **Anticipation for Hermes-3**: A member appreciated the current **Hermes-2-Pro-Mistral-7B** but inquired about **Hermes-2-Pro-Mixtral-8x7B-DPO**, learning its release is on hold for the **Hermes 3** preview. The general consensus is that the current flagship model will likely stay until **Hermes-3-Pro-Mixtral-8x7b-DPO** is released.

- **Optimizer Confusion**: A member requested resources for optimizers, schedulers, and learning rates for transformers, expressing that the original formula from "Attention Is All You Need" had issues with converging too rapidly.

- **Understanding Function Calling in AI**: The discussion explained that function calling in AI involves providing function signatures for the AI to use in applications. This is designed to be generalizable to various tools, and users are responsible for how outputs are utilized.

- **Model Modification and Rollback**: There was a clarification that **DPO** (Domain/Developer Personality Overlay) modifies the actual model in discussion. Users can revert to previous stages (e.g., SFT before DPO). Despite some confusion, it was clarified that the **gguf** file is not modified post-download.

**Link mentioned**: <a href="https://arxiv.org/abs/2404.01413">Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data</a>: The proliferation of generative models, combined with pretraining on web-scale data, raises a timely question: what happens when these models are trained on their own generated outputs? Recent investi...

  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/)** (1 messages): 

4biddden: Is there a runpod template available for the bittensor fine-tune?
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1227150742406828032)** (93 messages🔥🔥): 

- **DDOS Basic Defense Tactics**: One member highlighted that IP-rotation is a fundamental aspect of DDOS attacks and blocking a single IP is a common defense method, with another member jokingly responding about their "white hat" hacker status.

- **WorldSim Anticipation Builds**: Several members expressed excitement about the possible return of **WorldSim**, speculating that it might come back sometime this week with predictions for a Thursday reopening.

- **Language Flexibility in WorldSim**: Discussions indicate that **WorldSim** is capable of functioning in multiple languages, including Japanese and French, by setting the interface language or if the user can interact with the underlying AI (like Claude) in that language.

- **Alternatives to WorldSim Usage**: Members provided alternative ways to engage with world simulation experiences using publicly available prompts or by building agents with Nous Hermes Mixtral for free, while others mentioned platforms like AI Dungeon and openrouter.ai as temporary options.

- **Local vs. Datacenter Capabilities for AI Sims**: There was a consensus that running powerful AI models like the ones used in **WorldSim** locally on personal devices would offer substantially degraded performance compared to datacenter capabilities, and that it is unlikely to be a viable option in the near future.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://play.aidungeon.com/scenario/9D9o0X3tA8Vb/world-sim">AI Dungeon</a>: no description found</li><li><a href="https://hf.co/chat/assistant/65ffac7250c6fddecfd20bc8">HuggingChat</a>: no description found</li><li><a href="https://openrouter.ai/models?q=opus>">OpenRouter</a>: Browse models on OpenRouter
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1227538082124009512)** (1 messages): 

- **RNNs Under the Interpretabily Microscope**: A new study suggests that **interpretability tools** designed for transformers are largely applicable to modern RNNs like **Mamba and RWKV**. The research demonstrated that techniques such as vector arithmetic, eliciting early next-token predictions, and revealing true answers despite false fine-tuning are effective. View the paper [here](https://arxiv.org/abs/2404.05971).

- **Open-Sourcing RNN Insights**: The study's methodologies and experiments with RNN language models have been made openly available on GitHub, fostering community engagement in engineering the state of these models. Check out the repository [here](https://github.com/EleutherAI/rnngineering).

- **RNN Developments Take to Twitter**: A summary and discussion about the versatility of interpretability tools between transformers and RNNs were shared by the author in a **Twitter thread**, extending the conversation to the broader AI community. Join the thread [here](https://x.com/norabelrose/status/1777975663590531533).

- **Collaborative Efforts Acknowledged**: Special gratitude was extended to several collaborators and the broader community channel for their contributions to this interpretability research on RNN language models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05971">Does Transformer Interpretability Transfer to RNNs?</a>: Recent advances in recurrent neural network architectures, such as Mamba and RWKV, have enabled RNNs to match or exceed the performance of equal-size transformers in terms of language modeling perplex...</li><li><a href="https://github.com/EleutherAI/rnngineering">GitHub - EleutherAI/rnngineering: Engineering the state of RNN language models (Mamba, RWKV, etc.)</a>: Engineering the state of RNN language models (Mamba, RWKV, etc.) - EleutherAI/rnngineering</li><li><a href="https://x.com/norabelrose/status/1777975663590531533">Tweet from Nora Belrose (@norabelrose)</a>: RNN language models are making a comeback recently, with new architectures like Mamba and RWKV.  But do interpretability tools designed for transformers transfer to the new RNNs? We tested 3 popular i...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1227258395598524497)** (250 messages🔥🔥): 

- **Speculation about Claude 3 Opus' Model Size**: Amidst discussions on the undisclosed model size of **Claude 3 Opus**, several participants expressed surprise at the lack of reliable information, drawing contrasts to previous models like **GPT-4** where early predictions about scale were available. It was mentioned that leaks about model sizes at **Anthropic** may bear serious consequences.

- **Debating Daniel Han's Claims**: A member questioned Daniel Han's credibility, referencing a history of making optimistic claims with errors. Further discussion included asking for specific instances of errors and examining Han's approval by prominent figures in the AI community, such as **Karpathy** and **Hugging Face**, with links to previous discussions provided for context.

- **Google's Gemini Faces Backlash**: The conversation turned to the backlash against **Google's Gemini**, focusing on its restrictive image generation policies and later finding out that the project safety lead held controversial views. Despite the discussion of its repercussions, it was suggested that the backlash may have contributed to an increase in Gemini's popularity as people were curious to test it themselves.

- **Mistral and Unsloth in the Spotlight**: Discussions arose around **Mistral** and a new optimization library called **Unsloth**, with one member advocating for the performance enhancements it supposedly offers over **Hugging Face** combined with **Flash Attention 2 (FA2)**. A complex and technical conversation ensued about the veracity of performance claims and the importance of proper baselines for legitimate benchmarking.

- **AI Governance and Regulation**: A bill titled *Generative AI Copyright Disclosure Act* was introduced by Representative **Adam Schiff**, aiming to increase transparency about the use of copyrighted material in AI training datasets. The community shared the [link to the bill](https://schiff.house.gov/imo/media/doc/the_generative_ai_copyright_disclosure_act.pdf) and discussed potential impacts on the industry.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jphme/status/1778030213881909451">Tweet from Jan P. Harries (@jphme)</a>: @MistralAI first AGIEval results look great 👇 - thanks for releasing this beast, guys! 👏 https://x.com/jphme/status/1778028110954295486  ↘️ Quoting Jan P. Harries (@jphme)   First AGIEval results fo...</li><li><a href="https://schiff.house.gov/news/press-releases/rep-schiff-introduces-groundbreaking-bill-to-create-ai-transparency-between-creators-and-companies">Rep. Schiff Introduces Groundbreaking Bill to Create AI Transparency Between Creators and Companies</a>: The Official U.S. House website of Congressman Adam Schiff of California District 30</li><li><a href="https://theaidigest.org/timeline">Timeline of AI forecasts - AI Digest</a>: What to expect in AI capabilities, potential harms, and society&#x27;s response</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth update: Mistral support + more</a>: We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/4">mistral-community/Mixtral-8x22B-v0.1 · Benchmarks are here!</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/cross_entropy_loss.py#L76">unsloth/unsloth/kernels/cross_entropy_loss.py at main · unslothai/unsloth</a>: 2-5X faster 80% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/issues/26498">Mistral loss instability · Issue #26498 · huggingface/transformers</a>: System Info Hello, I&#39;ve been working with dhokas who finetuned Mistral&#39;s official instruct model. I have been trying to finetune mistral with several datasets over dozens of ablations. There i...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1227145452001034261)** (203 messages🔥🔥): 

- **Discussion on Knowledge Storage Capacity of LMs**: A paper estimating language models' knowledge storage capacity revealed they can store a **maximum of 2 bits per parameter**, suggesting a 7B parameter model could store enough factual knowledge to exceed English Wikipedia ([Scaling Laws for Neural Language Models' Capacity to Store and Manipulate Information](https://arxiv.org/abs/2404.05405)). Critics raised concerns that hyperparameter re-tuning was neglected which could affect MLP ablation results, impacting the accuracy of the findings.

- **Rapid Releases in Diffusion Model Finetuning**: Three new papers were released exploring various aspects of **diffusion finetuning**. The first explores a new approach for aligning text-to-image diffusion models ([Diffusion-KTO: Knowledge-enhanced Text-to-Image Diffusion Models without Training Pairwise Comparisons](https://arxiv.org/abs/2404.05961)), another addresses optimizing information storage in MoE language models ([DS-MoE: Towards IO Efficiency for MoE Language Model Inference via Dense-Sparse Mixture-of-Expert Training](https://arxiv.org/abs/2404.05567)), and the last paper investigates fine-tuning diffusion models at scale ([Batch Size Invariant Adam](https://arxiv.org/abs/2404.04860)).

- **LoRA-Based Innovations and Comparisons**: Discussions abounded around a paper utilizing **singular value decomposition (SVD)** and **LoRA (Low-Rank Adaptation)** to decompose pretrained values, drawing comparisons to the LoRD technique but highlighting significant differences in approach and goal ([No references provided]).

- **Encoder vs. Decoder Performance and Potential**: A study introduced **LLM2Vec**, converting a decoder-only LLM into an encoder for text embeddings and claimed vastly improved performance ([LLM2Vec: Unsupervised Contrastive Learning of Large Decoder-only Language Models](https://arxiv.org/abs/2404.05961)). Commenters debated fairness in comparisons and the practicality of the approach, recalling similar past efforts like CARP for controlled story generation and evaluation.

- **Exploring the Untapped Powers of Encoder-Decoder Models**: There was a notable interest in discussing the untapped potential of encoder-decoder models for embedding research, suggesting these architectures could be configured to enforce specific representation characteristics or hierarchies.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.05961">LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders</a>: Large decoder-only language models (LLMs) are the state-of-the-art models on most of today&#39;s NLP tasks and benchmarks. Yet, the community is only slowly adopting these models for text embedding ta...</li><li><a href="https://arxiv.org/abs/2404.05892">Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence</a>: We present Eagle (RWKV-5) and Finch (RWKV-6), sequence models improving upon the RWKV (RWKV-4) architecture. Our architectural design advancements include multi-headed matrix-valued states and a dynam...</li><li><a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model&#39;s capability via loss or benchmarks, we estimate the n...</li><li><a href="https://arxiv.org/abs/2404.05567">Dense Training, Sparse Inference: Rethinking Training of Mixture-of-Experts Language Models</a>: Mixture-of-Experts (MoE) language models can reduce computational costs by 2-4$\times$ compared to dense models without sacrificing performance, making them more efficient in computation-bounded scena...</li><li><a href="https://openreview.net/forum?id=Kloou2uk_Rz">A Large Batch Optimizer Reality Check: Traditional, Generic...</a>: We retune the Nesterov/Adam optimizers on pipelines where LARS/LAMB are commonly used and achieve similar or better performance, providing competitive baselines for the large batch training setting.</li><li><a href="https://openreview.net/forum?id=xIHi5nxu9P">Subtractive Mixture Models via Squaring: Representation and Learning</a>: Mixture models are traditionally represented and learned by adding several distributions as components. Allowing mixtures to subtract probability mass or density can drastically reduce the number...</li><li><a href="https://fxtwitter.com/JiaChenyan/status/1732898372359799159">Tweet from Chenyan Jia (@JiaChenyan)</a>: Can we design AI systems to consider democratic values as their objective functions? Our new #CSCW24 paper w/ @michelle123lam, Minh Chau Mai, @jeffhancock, @msbernst introduces a method for translatin...</li><li><a href="https://arxiv.org/abs/2402.18824">Batch size invariant Adam</a>: We propose a batch size invariant version of Adam, for use in large-scale, distributed settings, in which the mini-batch is divided into micro-batches which are distributed among worker nodes. For the...</li><li><a href="https://arxiv.org/abs/2402.00691">Comparative Study of Large Language Model Architectures on Frontier</a>: Large language models (LLMs) have garnered significant attention in both the AI community and beyond. Among these, the Generative Pre-trained Transformer (GPT) has emerged as the dominant architecture...</li><li><a href="https://arxiv.org/abs/2403.00871">Teach LLMs to Phish: Stealing Private Information from Language Models</a>: When large language models are trained on private data, it can be a significant privacy risk for them to memorize and regurgitate sensitive information. In this work, we propose a new practical data e...</li><li><a href="https://arxiv.org/abs/2205.05862">AdaVAE: Exploring Adaptive GPT-2s in Variational Auto-Encoders for Language Modeling</a>: Variational Auto-Encoder (VAE) has become the de-facto learning paradigm in achieving representation learning and generation for natural language at the same time. Nevertheless, existing VAE-based lan...</li><li><a href="https://arxiv.org/abs/2307.13912">Embedding Democratic Values into Social Media AIs via Societal Objective Functions</a>: Can we design artificial intelligence (AI) systems that rank our social media feeds to consider democratic values such as mitigating partisan animosity as part of their objective functions? We introdu...</li><li><a href="https://tenor.com/view/avocado-bacon-salad-lunch-salad-gif-12338945">Avocado Bacon Salad Lunch GIF - Avocado Bacon Salad Lunch Salad - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2110.03111">Cut the CARP: Fishing for zero-shot story evaluation</a>: Recent advances in large-scale language models (Raffel et al., 2019; Brown et al., 2020) have brought significant qualitative and quantitative improvements in machine-driven text generation. Despite t...</li><li><a href="https://arxiv.org/abs/2210.07792">Robust Preference Learning for Storytelling via Contrastive Reinforcement Learning</a>: Controlled automated story generation seeks to generate natural language stories satisfying constraints from natural language critiques or preferences. Existing methods to control for story preference...</li><li><a href="https://arxiv.org/abs/2404.05595">UniFL: Improve Stable Diffusion via Unified Feedback Learning</a>: Diffusion models have revolutionized the field of image generation, leading to the proliferation of high-quality models and diverse downstream applications. However, despite these significant advancem...</li><li><a href="https://arxiv.org/abs/2404.04860">ByteEdit: Boost, Comply and Accelerate Generative Image Editing</a>: Recent advancements in diffusion-based generative image editing have sparked a profound revolution, reshaping the landscape of image outpainting and inpainting tasks. Despite these strides, the field ...</li><li><a href="https://arxiv.org/abs/2404.04465">Aligning Diffusion Models by Optimizing Human Utility</a>: We present Diffusion-KTO, a novel approach for aligning text-to-image diffusion models by formulating the alignment objective as the maximization of expected human utility. Since this objective applie...</li><li><a href="https://github.com/andreaspapac/CwComp">GitHub - andreaspapac/CwComp: Convolutional Channel-wise Competitive Learning for the Forward-Forward Algorithm. AAAI 2024</a>: Convolutional Channel-wise Competitive Learning for the Forward-Forward Algorithm. AAAI 2024 - andreaspapac/CwComp</li><li><a href="https://pubmed.ncbi.nlm.nih.gov/35662458/">GWYRE: A Resource for Mapping Variants onto Experimental and Modeled Structures of Human Protein Complexes - PubMed</a>: Rapid progress in structural modeling of proteins and their interactions is powered by advances in knowledge-based methodologies along with better understanding of physical principles of protein struc...</li><li><a href="https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/">How to Scale Hyperparameters as Batch Size Increases</a>: no description found
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1227556417502969917)** (4 messages): 

- **Scaling Laws Explored for Knowledge Storage**: A new [paper on arXiv](https://arxiv.org/abs/2404.05405) introduces an approach to estimate the number of knowledge bits that language models can store. It suggests that models can store **2 bits of knowledge per parameter**, meaning a **7B model could store 14B bits of knowledge**, potentially exceeding the knowledge contained in English Wikipedia and textbooks combined.
- **Eleuther Community Ponders Over New Paper**: In the Eleuther community, there was a mention of positive opinions regarding the **knowledge storage paper** discussed, yet it was also noted that the paper is *hard to parse* and there might be a need for a discussion on its relevant results.
- **Seeking Benchmarks for OpenAI's New Model**: There's an inquiry about benchmarks for the latest **OpenAI model versions** like **gpt-4-turbo**; the question is where to find these benchmark results when they are released via API.

**Link mentioned**: <a href="https://arxiv.org/abs/2404.05405">Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws</a>: Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model&#39;s capability via loss or benchmarks, we estimate the n...

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

norabelrose: https://arxiv.org/abs/2404.05971
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1227160807347851306)** (8 messages🔥): 

- **Call for Collaboration on `apply_chat_template`**: A member mentions their eagerness to assist with the integration of `apply_chat_template` for model evaluation. Another member acknowledges the ongoing work and invites assistance upon their return, while a different participant also volunteers to help.

- **Inference Speed Boost via `big-refactor`**: A query regarding whether the `big-refactor` branch offers faster inference when compared to the `main` branch is confirmed positively by another member.

- **Torrenting ThePile v1**: A member shares a magnet link for downloading EleutherAI's ThePile v1 dataset.

- **Chat Templating Pull Requests**: `stellaathena` provides links to two pull requests that contribute to chat templating features for Hugging Face models; you can find the first PR [here](https://github.com/EleutherAI/lm-evaluation-harness/pull/1287#issuecomment-1967469808) and there's another one linked [here](https://github.com/EleutherAI/lm-evaluation-harness/pull/1578). They note that adding batchwise operations for `apply_chat_template` in the transformers library would be highly beneficial for the project and others.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1287#issuecomment-1967469808)">[WIP] Add chat templating for HF models by haileyschoelkopf · Pull Request #1287 · EleutherAI/lm-evaluation-harness</a>: This is a WIP PR , carrying on the draft @daniel-furman in #1209 started of adding the specified oft-requested chat templating feature. Current TODOs are:   Check performance using e.g. OpenHermes ...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1578).">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1227144864055951400)** (68 messages🔥🔥): 

- **AI Becomes the New Picasso?**: Debates emerged about whether AI should be accepted as a new form of artist. While some appreciate the effort and hard work of human artists, concerns were raised about AI-generated art and its implications for artistic credit and effort.
- **Master Student Seeks AI Chat System**: A Master’s student seeks an open-source GPT chat system template for their thesis. Recommended tools include **LM Studio** and the **Open-Source LLM Advisor**.
- **Perplexity Garners Praise**: Users discussed **Perplexity**, an AI tool with a 32K context window, capable of switching between models like **Opus** and **GPT-4**. Some users have upgraded to Pro and report a positive experience.
- **Customization a Key Ask for Future GPT Versions**: A user expressed desire for better customizability such as ranking system output and conciseness in GPT's responses. The idea of introducing "custom instructions" for more finely-tailored outputs was floated.
- **GPT-4 Access Capped, Users Confused**: Members reported messages indicating they have reached a usage cap for **GPT-4** even though they were set to use version 3.5. A link to OpenAI's status updates was shared, which documents ongoing investigations into ChatGPT errors.

**Link mentioned**: <a href="https://status.openai.com/">OpenAI Status</a>: no description found

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1227303668299206727)** (33 messages🔥): 

```html
<ul>
  <li><strong>Domain Verification Troubles</strong>: A user encountered an error when trying to publish a GPT, asking for suggestions on how to verify a domain even after setting up the TXT records.</li>
  <li><strong>GPT to SaaS Transformation Inquiry</strong>: One member was seeking advice on services available to convert GPT into a single-purpose SaaS application, aiming to create a proof of concept for future endeavors.</li>
  <li><strong>Technical Difficulties with GPT</strong>: Several members reported experiencing issues ranging from inability to load GPT, mentions not functioning, to suspended API access due to billing problems despite sufficient funds.</li>
  <li><strong>Chatbot Outage Reports</strong>: Users were facing outages with GPT, signalizing errors such as "GPT inaccessible or not found" and having trouble retrieving existing conversations.</li>
  <li><strong>Service Status Updates and Confirmation</strong>: A link to <a href="https://status.openai.com/">OpenAI's service status page</a> was shared, confirming the ongoing investigation into elevated errors and intermittent outages affecting ChatGPT services.</li>
</ul>
```

**Link mentioned**: <a href="https://status.openai.com/">OpenAI Status</a>: no description found

  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1227265760783503390)** (179 messages🔥🔥): 

- **AI Maximum Security**: A member cautioned against sharing or promoting **jailbreak-related prompts** as it violates AI stewardship and OpenAI policies. They also referenced a [Google search article](https://www.google.com/search?q=Don%27t+worry+about+AI+breaking+out%2C+worry+about+us+breaking+in) for in-depth understanding.
- **Prompt Engineering 101**: The conversation turned into a workshop on **prompt engineering**, using Pokemon Showdown and AI dialogue as examples. One user suggested *meta-prompting* - iteratively refining prompts by asking the AI itself to generate instructions for desired outputs.
- **Fine-Tuning for Excellence**: The same user also revealed that asking ChatGPT for *specific dialogue examples* and then for *instructions* based on those, can help mimic those patterns in future outputs, emphasizing the significance of letting the AI structure instructions. 
- **Guarding Against AI Missteps**: There was a mention of a technique to prevent a custom GPT from sharing its instructions, involving a phrase added to the “Instructions” that mitigates some basic prompt injection threats if **Code Interpreter** is enabled.
- **ChatGPT Writes Instructions**: Multiple participants engaged in refining the way to generate compelling **Pokemon battle dialogue** by leveraging ChatGPT's ability to produce instructions for generating better dialogue, highlighting the AI's potential to surpass even the engineers' own capabilities in task specificity.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1227265760783503390)** (179 messages🔥🔥): 

- **AI Jailbreak Prompt Awareness**: A member emphasized caution against sharing AI jailbreak prompts. They highlighted the ethical considerations and referred to a Google search term, *“Don't worry about AI breaking out, worry about us breaking in”*, to explain the risks and issues with promoting AI jailbreaking techniques.
- **Custom Instructions Against AI Misuse**: There's an exchange about creating custom instructions for AI models to prevent them from revealing sensitive information when Code Interpreter is enabled. A member shared a prompt to encourage the Custom GPT to ***graciously decline*** revealing details about its system.
- **The Documentary Nature of AI**: One participant expressed that ***“In the era of large language models, the documentation is the source code.”*** This statement underscored the importance of AI documentation in understanding and replicating model behavior.
- **Enhancing AI-Generated Pokémon Battle Dialogues**: A lengthy discussion was had about generating better Pokémon battle dialogues using ChatGPT. A member suggested using **meta-prompting**—letting the AI suggest how to construct prompts to improve the writing of dialogues—accompanied by iterative refinement and testing with the AI.
- **Meta-Prompting as a Powerful Tool**: A member illustrated the concept of meta-prompting for another user, demonstrating how to refine the AI's output to improve its battle dialogue writing for a Pokémon game. Through this process, the user learned to ask ChatGPT for specific instructional prompts until the results met expectations.
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1227139444684095520)** (141 messages🔥🔥): 

- **Introducing AutoCodeRover**: Singapore presents **AutoCodeRover**, an autonomous software engineer capable of resolving GitHub issues tied to bug fixes or feature additions with minimal costs and quick turnaround times. Links to the project on GitHub and a preprint paper were shared. [GitHub Repository](https://github.com/nus-apr/auto-code-rover), [Preprint PDF](https://github.com/nus-apr/auto-code-rover/blob/main/preprint.pdf).
  
- **GPT-4-Turbo Models Hit the Scene**: The latest **GPT-4-Turbo** model with a training cutoff date of December 2023 is out, offering vast improvements over previous iterations. The community's reaction includes observations of improved reasoning on complex tasks and anticipation of its rollout for ChatGPT Plus subscribers. [OpenAI Pricing](https://openai.com/pricing), [OpenAI's Official Tweet](https://twitter.com/OpenAIDevs/status/1777769463258988634).

- **Music Generation Enters New Era with Udio**: A hot topic discussion about Udio, a new music generation app, sparked interest for its potential to rival Suno with its intuitive text-prompting system for music creation and generous allowance of 1200 songs per user each month during its beta phase. There's excitement and speculation about how this new player will impact the music industry. [Udio Announcement](https://x.com/udiomusic/status/1778045322654003448?s=46&t=6FDPaNxZcbSsELal6Sv7Ug), [Reddit Discussion about Udio](https://old.reddit.com/r/singularity/comments/1bzd4bo/its_been_confirmed_the_suno_killer_is_called_udio/).

- **Mixtral 8x22b Model Released**: The release of Mixtral's **8x22b model** drew attention for its substantial parameter count and notable comparison to the performance of GPT-4 and Claude Sonnet. The conversation highlighted the model's technical specs and its capacity to run on heavy hardware, with further evaluations awaited from the AI community. [Teknium Tweet](https://x.com/Teknium1/status/1777875926807929157).

- **Nvidia's Blackwell Performance Analysis**: Nvidia's Blackwell chips were a big talking point, especially after an analysis was shared comparing their total cost of ownership and performance relative to older models like the H100 and A100, with the focus on their applicability for GPT-4's inference and training needs. The discussion pointed out the importance of marketing realism regarding performance claims. [SemiAnalysis Article](https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/2024/04/09/gpt-4-turbo.html">GPT-4 Turbo with Vision is a step backwards for coding</a>: OpenAI’s GPT-4 Turbo with Vision model scores worse on aider’s code editing benchmarks than all the previous GPT-4 models. In particular, it seems much more prone to “lazy coding” than the existing GP...</li><li><a href="https://x.com/cursor_ai/status/1777886886884986944?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Cursor (@cursor_ai)</a>: Cursor users can now use the new gpt-4-turbo model. We’ve observed improved reasoning on complex tasks.  Example comparison between gpt-4-1106 and the new gpt-4-turbo below:</li><li><a href="https://turbopuffer.com/">turbopuffer</a>: turbopuffer is a vector database built on top of object storage, which means 10x-100x cheaper, usage-based pricing, and massive scalability</li><li><a href="https://x.com/AbhikRoychoudh1/status/1777494000611852515">Tweet from Abhik Roychoudhury (@AbhikRoychoudh1)</a>: Introducing  AutoCodeRover Presenting our autonomous software engineer from Singapore ! Takes in a Github issue (bug fixing or feature addition), resolves in few minutes, with minimal LLM cost ~$0.5 !...</li><li><a href="https://x.com/7oponaut/status/1777971159478194256?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from 7oponaut (@7oponaut)</a>: New GPT-4 passes the magic elevator test</li><li><a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>: no description found</li><li><a href="https://x.com/kwindla/status/1777712299215901062">Tweet from kwindla (@kwindla)</a>: @latentspacepod Here&#39;s a video from @chadbailey59 showing the possibilities of fast voice response + tool calling.</li><li><a href="https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis">Nvidia Blackwell Perf TCO Analysis - B100 vs B200 vs GB200NVL72</a>: GPT-4 Profitability, Cost, Inference Simulator, Parallelism Explained, Performance TCO Modeling In Large &amp; Small Model Inference and Training</li><li><a href="https://supabase.com/docs/guides/database/extensions/pgvector">pgvector: Embeddings and vector similarity | Supabase Docs</a>: pgvector: a PostgreSQL extension for storing embeddings and performing vector similarity search.</li><li><a href="https://x.com/polynoamial/status/1777809000345505801?s">Tweet from Noam Brown (@polynoamial)</a>: GPT-4 reasoning has been further improved  ↘️ Quoting OpenAI (@OpenAI)   Majorly improved GPT-4 Turbo model available now in the API and rolling out in ChatGPT.</li><li><a href="https://x.com/liambolling/status/1777758743637483562?s=46&t=90xQ8sGy63D2Otia">Tweet from Liam Bolling (@liambolling)</a>: 🎉 It’s a big day for @Google Gemini.   Gemini 1.5 Pro now understands audio, uses unlimited files, acts on your commands, and lets devs build incredible things with JSON mode! It’s all 🆓. Here’s why...</li><li><a href="https://openai.com/pricing">Pricing</a>: Simple and flexible. Only pay for what you use.</li><li><a href="https://x.com/gdb/status/1778126026532372486?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Greg Brockman (@gdb)</a>: Example of new GPT-4 Turbo compared to old:  ↘️ Quoting Pietro Schirano (@skirano)   Side-by-side comparison between the latest version of gpt-4-turbo and the previous one, 0125-preview.  Not only is ...</li><li><a href="https://x.com/Teknium1/status/1777875926807929157">Tweet from Teknium (e/λ) (@Teknium1)</a>: Mistral releases an 8x22b model  ↘️ Quoting Mistral AI (@MistralAI)   magnet:?xt=urn:btih:9238b09245d0d8cd915be09927769d5f7584c1c9&dn=mixtral-8x22b&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fanno...</li><li><a href="https://x.com/moultano/status/1777727219097342287">Tweet from Ryan Moulton (@moultano)</a>: The way Nigerian twitter is blowing up at this makes me think a lot of ChatGPTisms are just colloquial language for the workforce they hired to write fine tuning data.  ↘️ Quoting Paul Graham (@paulg)...</li><li><a href="https://x.com/getdelve/status/1777814330207297721?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Delve (YC W24) (@getdelve)</a>: We 100% agree. Imagine founding a YC company called Delve.  ↘️ Quoting Paul Graham (@paulg)   My point here is not that I dislike &#34;delve,&#34; though I do, but that it&#39;s a sign that text was w...</li><li><a href="https://x.com/liambolling/status/1777758743637483562?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Liam Bolling (@liambolling)</a>: 🎉 It’s a big day for @Google Gemini.   Gemini 1.5 Pro now understands audio, uses unlimited files, acts on your commands, and lets devs build incredible things with JSON mode! It’s all 🆓. Here’s why...</li><li><a href="https://x.com/AlpayAriyak/status/1777852771514904719">Tweet from Alpay Ariyak (@AlpayAriyak)</a>: I ran humaneval (base and plus) on the new GPT-4-Turbo-2024-04-09, and it ranks #1 on both</li><li><a href="https://x.com/farbood/status/1777775047543054525?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from farbood — e/acc (@farbood)</a>: Today we are open-sourcing and sharing a longevity assistant called Sequel  - locally stored: we don’t get or see your data - chat with your complete health picture: blood labs, Whoop, DEXA, MRI, ther...</li><li><a href="https://x.com/stevenheidel/status/1777789577438318625?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Steven Heidel (@stevenheidel)</a>: delve into the latest gpt-4-turbo model: - major improvements across the board in our evals (especially math) - dec 2023 knowledge cutoff  ↘️ Quoting OpenAI (@OpenAI)   Majorly improved GPT-4 Turbo mo...</li><li><a href="https://x.com/rohanpaul_ai/status/1777747790564589844?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Rohan Paul (@rohanpaul_ai)</a>: BREAKING 🔥🤯  Google releases model with new Griffin architecture that outperforms transformers.  Across multiple sizes, Griffin out performs the benchmark scores of transformers baseline in controll...</li><li><a href="https://x.com/phill__1/status/1777816655386538021?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Phil (@phill__1)</a>: The new GPT-4 Turbo model is the only one that could solve this math question: &#34;Determine the sum of the y-coordinates of the four points of intersection of y = x^4 - 5x^2 - x + 4 and y = x^2 - 3x...</li><li><a href="https://x.com/rohanpaul_ai/status/1777747790564589844?s=46&t=90">Tweet from Rohan Paul (@rohanpaul_ai)</a>: BREAKING 🔥🤯  Google releases model with new Griffin architecture that outperforms transformers.  Across multiple sizes, Griffin out performs the benchmark scores of transformers baseline in controll...</li><li><a href="https://x.com/polynoamial/status/1777809000345505801?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Noam Brown (@polynoamial)</a>: GPT-4 reasoning has been further improved  ↘️ Quoting OpenAI (@OpenAI)   Majorly improved GPT-4 Turbo model available now in the API and rolling out in ChatGPT.</li><li><a href="https://x.com/dylan522p/status/1777954675012305176?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Dylan Patel (@dylan522p)</a>: Nvidia Blackwell Perf TCO Analysis B100 vs B200 vs GB200NVL72 GPT-4 Profitability, Cost Inference Simulator Parallelism Explained Performance TCO Modeling In Large & Small Model Inference and Training...</li><li><a href="https://x.com/udiomusic/status/1778045322654003448?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from udio (@udiomusic)</a>: Introducing Udio, an app for music creation and sharing that allows you to generate amazing music in your favorite styles with intuitive and powerful text-prompting.  1/11</li><li><a href="https://www.youtube.com/live/qhOwhoi8XUU?si=F_SyTNdHwCijw437&t=1083">Gen AI Office Hours: Jason, Hamel, Eugene</a>: no description found</li><li><a href="https://www.youtube.com/live/qhOwhoi8XUU?s">Gen AI Office Hours: Jason, Hamel, Eugene</a>: no description found</li><li><a href="https://x.com/BorisMPower/status/1777867583947227582">Tweet from Boris Power (@BorisMPower)</a>: “Majorly improved” 😉  ↘️ Quoting OpenAI (@OpenAI)   Majorly improved GPT-4 Turbo model available now in the API and rolling out in ChatGPT.</li><li><a href="https://x.com/]">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://x.com/teortaxestex/status/1778090743816442202?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: So, ~Medium v2. I guess this means they will retire the current Meduim soon enough.  ↘️ Quoting Waseem AlShikh (@waseem_s)   @Get_Writer  team had the opportunity to run an eval  for Mixtral-8x22b, re...</li><li><a href="https://x.com/bindureddy/status/1778090437448024231?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Bindu Reddy (@bindureddy)</a>: This is an EXCELLENT TABLE of the various benchmarks for all the models.  The new Mixtral has the top MMLU score of 77.3, just a little ahead of Qwen 72B, which was yesterday&#39;s best open-source mo...</li><li><a href="https://x.com/danielhanchen/status/1777912653580771674?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Daniel Han (@danielhanchen)</a>: Can&#39;t download @MistralAI&#39;s new 8x22B MoE, but managed to check some files!  1. Tokenizer identical to Mistral 7b 2. Mixtral (4096,14336) New (6144,16K), so larger base model used. 3. 16bit ne...</li><li><a href="https://x.com/awnihannun/status/1778054275152937130?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Awni Hannun (@awnihannun)</a>: New Mixtral 8x22B runs nicely in MLX on an M2 Ultra.  4-bit quantized model in the 🤗 MLX Community: https://huggingface.co/mlx-community/Mixtral-8x22B-4bit  h/t @Prince_Canuma for MLX version and v2r...</li><li><a href="https://x.com/reach_vb/status/1777946948617605384?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: mixtral 8x22B - things we know so far 🫡  &gt; 176B parameters &gt; performance in between gpt4 and claude sonnet (according to their discord) &gt; same/ similar tokeniser used as mistral 7b &gt; 6553...</li><li><a href="https://x.com/togethercompute/status/1778052158501667128?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Together AI (@togethercompute)</a>: New model now available on Together AI!   @MistralAI&#39;s latest base model, Mixtral-8x22B! 🚀  https://api.together.xyz/playground/language/mistralai/Mixtral-8x22B</li><li><a href="https://old.reddit.com/r/singularity/comments/1bzd4bo/its_been_confirmed_the_suno_killer_is_called_udio/">It's been confirmed - the &quot;Suno Killer&quot; is called Udio</a>: I've been investigating what some people have been calling the &quot;Suno killer&quot; - a music generation AI model that's supposedly 2 to 10 times better...</li><li><a href="https://x.com/reach_vb/status/1778020589225091453?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: IT WORKS! Running Mixtral 8x22B with Transformers! 🔥  Running on a DGX (4x A100 - 80GB) with CPU offloading 🤯  ↘️ Quoting Vaibhav (VB) Srivastav (@reach_vb)   mixtral 8x22B - things we know so far �...</li><li><a href="https://x.com/TheSeaMouse/status/1777870962882441596">Tweet from Hassan Hayat 🔥 (@TheSeaMouse)</a>: mixtral 8x22b config
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1227692289808269392)** (3 messages): 

- **Upcoming Paper Club on 1-bit LLMs**: A presentation on the **1-bit Large Language Models (LLMs)** paper is scheduled in **10 minutes** in the **LLM Paper Club** channel. For more details and to join the event, [register here](https://lu.ma/jcxntjox).

- **Delving into 1-bit LLMs**: The featured paper titled "BitNet b1.58" discusses a **ternary {-1, 0, 1}** 1-bit LLM that achieves performance comparable to full-precision models while being more **cost-effective**. To read the paper, check [arXiv submission](https://arxiv.org/abs/2402.17764).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://lu.ma/jcxntjox">LLM Paper Club (1-bit LLMs paper) · Luma</a>: This week @rj45 will be covering https://arxiv.org/abs/2402.17764 The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits Also submit and vote for our next paper:...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1227694310418546738)** (268 messages🔥🔥): 

- **Troubles with Visual Aid**: There were multiple reports of difficulties viewing the screen share during the session, with some members offering alternative platforms such as [x-ware.online](https://spaces.x-ware.online/r/5825ccea-4101-4718-9b67-a07932b81cdc) and [matrix.org](https://matrix.to/#/#temporarylatentspace:matrix.org).
- **Deep Learning Papers and Experience Sharing**: The channel discussed the pre-print paper titled "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," and additional resources such as a [blogpost](https://learning-exhaust.hashnode.dev/preview/6609ec4565bff73f1db1b51b) and the paper's PDF on [arXiv.org](https://arxiv.org/abs/2402.17764) were shared.
- **Audio and Video Issues**: Alongside the screen sharing problem, there were issues with project members unable to hear or speak during the meeting, eventually leading to the discussion about moving back to Zoom.
- **1-bit LLMs Insights and Discussion**: Members discussed the concept of 1-bit Large Language Models (LLMs), focusing on how regularization and quantization during training could be key to their success. A related Huggingface repository [BitNet-Transformers](https://github.com/Beomi/BitNet-Transformers) was also shared.
- **Paper Club Coordination and Future Topics**: Towards the end of the chat, the group coordinated on the next paper to discuss, and papers related to time series, such as TimeGPT, were suggested as potential topics of interest. There was also a reference to another LLM, BloombergGPT, which led to sharing of related content like a podcast [YouTube video](https://www.youtube.com/watch?v=byCe7-c84d4) for further exploration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://matrix.to/#/#temporarylatentspace:matrix.org">You&apos;re invited to talk on Matrix</a>: no description found</li><li><a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions">Join Slido: Enter #code to vote and ask questions</a>: Participate in a live poll, quiz or Q&A. No login required.</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://arxiv.org/abs/2402.18041">Datasets for Large Language Models: A Comprehensive Survey</a>: This paper embarks on an exploration into the Large Language Model (LLM) datasets, which play a crucial role in the remarkable advancements of LLMs. The datasets serve as the foundational infrastructu...</li><li><a href="https://arxiv.org/abs/2310.04793">FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets</a>: In the swiftly expanding domain of Natural Language Processing (NLP), the potential of GPT-based models for the financial sector is increasingly evident. However, the integration of these models with ...</li><li><a href="https://shapes.inc">Shapes, Inc.</a>: Shapes are AI friends that can talk to you on Discord</li><li><a href="https://spaces.x-ware.online/r/5825ccea-4101-4718-9b67-a07932b81cdc">Openhouse</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=byCe7-c84d4">BloombergGPT - an LLM for Finance with David Rosenberg - 639</a>: Today we’re joined by David Rosenberg, head of the machine learning strategy team in the Office of the CTO at Bloomberg. In our conversation with David, we d...</li><li><a href="https://spaces.x-ware.online/r/5825ccea-4101-4718-9b67-a">Openhouse</a>: no description found</li><li><a href="https://github.com/Beomi/BitNet-Transformers">GitHub - Beomi/BitNet-Transformers: 0️⃣1️⃣🤗 BitNet-Transformers: Huggingface Transformers Implementation of &quot;BitNet: Scaling 1-bit Transformers for Large Language Models&quot; in pytorch with Llama(2) Architecture</a>: 0️⃣1️⃣🤗 BitNet-Transformers: Huggingface Transformers Implementation of &amp;quot;BitNet: Scaling 1-bit Transformers for Large Language Models&amp;quot; in pytorch with Llama(2) Architecture - Beomi/...</li><li><a href="https://learning-exhaust.hashnode.dev/preview/6609ec4565bff73f1db1b51b">[Draft] 1.58 bits?</a>: no description found</li><li><a href="https://github.com/AI4Finance-Foundation/FinGPT">GitHub - AI4Finance-Foundation/FinGPT: FinGPT: Open-Source Financial Large Language Models!  Revolutionize 🔥    We release the trained model on HuggingFace.</a>: FinGPT: Open-Source Financial Large Language Models!  Revolutionize 🔥    We release the trained model on HuggingFace. - AI4Finance-Foundation/FinGPT</li><li><a href="https://spaces.x-ware.online">Openhouse</a>: no description found
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1227354692867199087)** (4 messages): 

- **Gemma 1.1 Instruct 7B Takes Center Stage**: **Gemma 1.1 Instruct 7B**, a newer and improved version, is now available on HuggingChat. The update is expected to be a net improvement from 1.0 and users are encouraged to try it [here](https://huggingface.co/chat/models/google/gemma-1.1-7b-it).

- **CodeGemma Unveiled**: *[CodeGemma](https://huggingface.co/spaces/ysharma/CodeGemma)* has landed, featuring models optimized for on-device code completion in sizes **2B and 7B** with 8192k context, and is available on HuggingFace. Google's [RecurrentGemma](https://twitter.com/jeethu/status/1777703476195196982), a non-transformer model that boasts solid results and scalability, was also released.

- **A More Economical Hugging Face**: Compute prices have been slashed by **up to 50%** for Spaces and Inference endpoints on HuggingFace, making them now more cost-effective than AWS EC2 on-demand services. Users will benefit from this price reduction starting April for Spaces or Inference Endpoints usage.

- **Community Insights Get Revamped**: HuggingFace's community blogs have been upgraded to "articles" with new features like upvotes and activity feed presence, plus access for paper authors. Visit the updated articles and explore user-generated content [here](https://huggingface.co/blog/community).

- **Serverless GPUs and Bonus ML Content**: Hugging Face introduces serverless GPU inference with Cloudflare and adds a new bonus unit focusing on Classical AI in Games to its ML for Games Course, enriching learning resources for interested parties. To delve into serverless GPUs, check out [Deploy on Cloudflare Workers AI](https://huggingface.co/blog/cloudflare-workers-ai), and for the bonus ML content, visit [Classical AI in Games](https://huggingface.co/learn/ml-games-course/unitbonus1/introduction).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/chat/models/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus - HuggingChat</a>: Use CohereForAI/c4ai-command-r-plus with HuggingChat</li><li><a href="https://x.com/NSarrazin_/status/1777634083197124995">Tweet from Nathan Sarrazin (@NSarrazin_)</a>: We just added support for Gemma 1.1 Instruct 7B on HuggingChat! It should be a net improvement from 1.0, curious to see how folks use it.  Try it out here: https://huggingface.co/chat/models/google/ge...</li><li><a href="https://x.com/_philschmid/status/1777673558874829090">Tweet from Philipp Schmid (@_philschmid)</a>: Gemma can now code!🤯 🔔 @GoogleDeepMind  just released Code Gemma, a collection of specialized open code models. Code Gemma comes in two different sizes 2B & 7B, excellent for on-device code completi...</li><li><a href="https://huggingface.co/spaces/ysharma/CodeGemma">CodeGemma - a Hugging Face Space by ysharma</a>: no description found</li><li><a href="https://x.com/_philschmid/status/1775885996435087449">Tweet from Philipp Schmid (@_philschmid)</a>: We are lowering the prices for Compute on Hugging Face by up to 50%!🤯 Yes, you heard it right @huggingface Spaces & Inference Endpoints are now, on average, 20% cheaper than AWS EC2 on-demand! 🤑  We...</li><li><a href="https://x.com/mervenoyann/status/1777630974693539849">Tweet from merve (@mervenoyann)</a>: recently we have shipped bunch of changes to community blogs (now called articles) 🆙 we now have upvotes, and upvoted articles appear in activity feed 🤝 we have given access to paper authors  📝 use...</li><li><a href="https://x.com/julien_c/status/1777328456709062848">Tweet from Julien Chaumond (@julien_c)</a>: We have decided to update text-generation-inference (TGI)&#39;s license.  We switch the license from HFOIL (our custom license) back to Apache 2, hence making the library fully open-source.  Read belo...</li><li><a href="https://x.com/freddy_alfonso_/status/1777390461704953934">Tweet from Freddy A Boulton (@freddy_alfonso_)</a>: Very sleek demo with a new custom @Gradio component by @Wauplin 👀  ↘️ Quoting Arcee.ai (@arcee_ai)   In collab /w @huggingface, Arcee is thrilled to release our MergeKit Hugging Face Space.   🙌 You ...</li><li><a href="https://x.com/m_olbap/status/1775201738397765775">Tweet from Pablo Montalvo (@m_olbap)</a>: It was hard to find quality OCR data... until today! Super excited to announce the release of the 2 largest public OCR datasets ever 📜 📜  OCR is critical for document AI: here, 26M+ pages, 18b text ...</li><li><a href="https://x.com/fleetwood___/status/1776281292109234626">Tweet from Fleetwood (@fleetwood___)</a>: A week of absolute struggle but Phi2 officially runs on Ratchet 🎺  Pretty sluggish right now 🐌 but lots of optimisation to come.</li><li><a href="https://github.com/huggingface/accelerate/releases/tag/v0.29.0">Release v0.29.0: NUMA affinity control, MLU Support, and DeepSpeed Improvements · huggingface/accelerate</a>: Core  Accelerate can now optimize NUMA affinity, which can help increase throughput on NVIDIA multi-GPU systems. To enable it either follow the prompt during accelerate config, set the ACCELERATE_C...</li><li><a href="https://huggingface.co/learn/ml-games-course/unitbonus1/introduction">Classical AI in Games - Hugging Face ML for Games Course</a>: no description found</li><li><a href="https://x.com/clefourrier/status/1777319187913875893">Tweet from Clémentine Fourrier 🍊 (@clefourrier)</a>: Follow up &#34;eval is fun&#34; tweet: how much do scores change depending on prompt format choice?  The score range for a given model is of 10 points! :D  Prompt format on the x axis, all these evals...</li><li><a href="https://x.com/abidlabs/status/1775787643324051582">Tweet from Abubakar Abid (@abidlabs)</a>: Introducing the Gradio API Recorder 🪄  Every Gradio app now includes an API recorder that lets you reconstruct your interaction in a Gradio app as code using the Python or JS clients!</li><li><a href="https://huggingface.co/blog/OzzyGT/outpainting-differential-diffusion">Outpainting II - Differential Diffusion</a>: no description found</li><li><a href="https://huggingface.co/blog/cloudflare-workers-ai">Bringing serverless GPU inference to Hugging Face users</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1227132805855838290)** (105 messages🔥🔥): 

- **Checkpoints Saving Woes**: A member had issues with a model not saving checkpoints to the specified directory using `TrainingArguments`. After confirming [[no errors in the training loop]](https://discord.com/channels/879548962464493619) and trying different paths, they resolved the problem by using `trainer.save_model("")` to save the model weights explicitly.

- **Gradio Questions Go Here**: When asked about the right place for Gradio-related inquiries, a link to the appropriate Discord channels was provided, including [general Gradio questions](https://discord.com/channels/879548962464493619/1025174734427656283), [Gradio in Spaces](https://discord.com/channels/879548962464493619/1019296127847239751), and [Gradio Feature Requests](https://discord.com/channels/879548962464493619/1014577787039924226).

- **Call for SEO Prompts**: A member sought assistance with prompts for SEO blog articles. Although the initial call didn't get a direct response, it indicates an interest in content creation guidance.

- **Learning Journey for AI Novices**: A new member to machine learning, proficient in Python, requested advice on starting with LLMs or image generator AI. This highlights a common entry point question for those new to the field.

- **Model Error Queries and Troubleshooting**: Several members discussed issues with model errors. Solutions ranged from checking parameters like `max_seq_len` to more granular advice such as taking a photo of the end errors, which are usually telling, and varying from assistance with code to actual deployment scenarios.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nroggendorff/cascade/blob/main/app.py">app.py · nroggendorff/cascade at main</a>: no description found</li><li><a href="https://huggingface.co/spaces/huggingface-projects/LevelBot">LevelBot - a Hugging Face Space by huggingface-projects</a>: no description found</li><li><a href="https://huggingface.co/settings/token">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/BAAI/bge-m3">BAAI/bge-m3 · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/Qz0KTGYJtUk?si=dq_Ptn1lpmwdNrt5">Coding Adventure: Ray Tracing</a>: I tried creating a custom ray/path tracing renderer. Featuring: maths, shaders, and cats!This project was written in C# and HLSL, and uses the Unity game eng...</li><li><a href="https://youtu.be/C94pTaKoLbU">Future of E-commerce?! Virtual clothing try-on agent</a>: I built an agent system which will autonomously iterate &amp; generate img of AI model wearing certain cloth and produce millions+ social postsFree access to run...</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/BrutPitt/glChAoS.P">GitHub - BrutPitt/glChAoS.P: 3D GPUs Strange Attractors and Hypercomplex Fractals explorer - up to 256 Million particles in RealTime</a>: 3D GPUs Strange Attractors and Hypercomplex Fractals explorer - up to 256 Million particles in RealTime - BrutPitt/glChAoS.P
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1227131692637229087)** (2 messages): 

- **Learn NLP in a Day**: A repository has been shared that offers a solution for sentiment classification using the IMDB movie 50k review dataset. The guide is easy to follow, comprehensive, and could be a generic approach for most NLP tasks. [Sentiment Classifier on GitHub](https://github.com/ManoBharathi93/Sentiment_Classifier/tree/main).

- **Navigating the Maze of Package Management**: A video was shared discussing various package management tools including Conda, Pip, and Libmamba, as well as tackling hard resets for Linux distributions. This content might help those struggling with package management complexities. [Watch on YouTube](https://youtu.be/7x4-zgCXz4M).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/7x4-zgCXz4M">Package Management On Conda, Pip, Libmamba, and hard resets</a>: Apologies for the departure from the daily videos. Got sick and also and also had to reset/update my linux distro. Making lemonade out of lemons, taking this...</li><li><a href="https://github.com/ManoBharathi93/Sentiment_Classifier/tree/main">GitHub - ManoBharathi93/Sentiment_Classifier: Sentiment classifier on IMDB movie dataset</a>: Sentiment classifier on IMDB movie dataset. Contribute to ManoBharathi93/Sentiment_Classifier development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1227194354209259592)** (7 messages): 

- **SimA: AI Trained Across Many Worlds**: [DeepMind presents SimA](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-generalist-ai-agent-for-3d-virtual-environments/Scaling%20Instructable%20Agents%20Across%20Many%20Simulated%20Worlds.pdf), a generalist AI agent for 3D virtual environments. This AI is designed to scale across numerous simulated worlds and perform various complex tasks.

- **Qdrant Meets DSPy for Enhanced Search**: A new [Medium post details the integration](https://medium.com/ai-advances/unlocking-advanced-capabilities-integrating-qdrant-with-dspy-72e570857f23) of **Qdrant** with DSPy to advance search capabilities. Combining these tools offers enhanced vector search and could potentially unlock new AI functionalities.

- **Karpathy's Tweet Sparks Curiosity**: The latest tweet from [Andrej Karpathy](https://twitter.com/karpathy/status/1777427944971083809) has stirred conversations among enthusiasts. The contents are unspecified in this context, requiring a direct visit to the link for details.

- **Explore HuggingFace Models with Marimo Labs**: The [Marimo Labs team developed an interface](https://github.com/marimo-team/marimo-labs) for interactively experimenting with any HuggingFace model. Marimo provides a user-friendly environment for testing and tuning various AI models.

- **Multilingual Information Extraction on HuggingFace**: Discover a [powerful and multilingual information extraction model](https://huggingface.co/spaces/urchade/gliner_multiv2.1) on HuggingFace Spaces. This tiny model can be used for robust information extraction tasks and is open-sourced under the Apache 2.0 license.

- **Quantum Leap for Transformers with Quanto**: A [new GitHub notebook](https://github.com/andysingal/llm-course/tree/main/Quantization) showcases how to employ Quanto for quantizing Transformers. This could enable more efficient deployment of these models on constrained hardware.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/urchade/gliner_multiv2.1">GLiNER-Multiv2.1 - a Hugging Face Space by urchade</a>: no description found</li><li><a href="https://github.com/andysingal/llm-course/tree/main/Quantization">llm-course/Quantization at main · andysingal/llm-course</a>: Contribute to andysingal/llm-course development by creating an account on GitHub.</li><li><a href="https://github.com/marimo-team/marimo-labs">GitHub - marimo-team/marimo-labs</a>: Contribute to marimo-team/marimo-labs development by creating an account on GitHub.</li><li><a href="https://github.co">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://marimo.app/l/tmk0k2">marimo | a next-generation Python notebook</a>: Explore data and build apps seamlessly with marimo, a next-generation Python notebook.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1227194612834369576)** (12 messages🔥): 

- **Deep Dive into Deep Q-Learning**: Shared a link to a collection of Deep Q-Learning projects on [GitHub](https://github.com/SuleymanEmreErdem/deep-q-learning-applications), promising a wealth of Deep Q-Learning applications with a variety of use cases.
- **Tracing Data Science Evolution**: Introduction of **RicercaMente**, a collaborative project aiming to map the evolution of data science through significant scientific papers. The project encourages community participation and can be found on [GitHub](https://github.com/EdoPedrocchi/RicercaMente).
- **Local LLMs Unleashed with everything-rag**: **everything-rag**, a fully customizable, local chatbot assistant that boasts support for any Long Large Model (LLM) and data, including the use of personal pdf files, was announced. It highlights the open-source and local nature of the tool, with the GitHub repo available [here](https://github.com/AstraBert/everything-rag) and a live demo provided on the HuggingFace [space](https://huggingface.co/spaces/as-cle-bert/everything-rag).
- **Fashion Forward with Virtual Try-On**: A virtual try-on system using IP-Adapter Inpainting has been created, showcased on the HuggingFace [space](https://huggingface.co/spaces/tonyassi/fashion-try-on), where users can visualize clothing items on models with impressive results, despite occasional color inversion issues.
- **Insights on Model Layer Behavior**: In an exchange about model layers, it was observed that the connection between layers varied depending on the input type—be it code, math, QA, or chat—with consistency in lower connection layers. The discussion also touched upon targeted dataset use for specific cases and the potential for pruning in models like **Mixtral 8x22B**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/tonyassi/fashion-try-on">Fashion Try On - a Hugging Face Space by tonyassi</a>: no description found</li><li><a href="https://huggingface.co/spaces/as-cle-bert/everything-rag">everything-rag - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://github.com/SuleymanEmreErdem/deep-q-learning-applications">GitHub - SuleymanEmreErdem/deep-q-learning-applications: My Deep Q-Learning projects</a>: My Deep Q-Learning projects. Contribute to SuleymanEmreErdem/deep-q-learning-applications development by creating an account on GitHub.</li><li><a href="https://github.com/EdoPedrocchi/RicercaMente">GitHub - EdoPedrocchi/RicercaMente: Open source project that aims to trace the history of data science through scientific research published over the years</a>: Open source project that aims to trace the history of data science through scientific research published over the years - EdoPedrocchi/RicercaMente
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1227149914048561206)** (8 messages🔥): 

- **Python Debugging Advice**: A member recommended understanding **Python classes, functions, decorators, imports**, and objects for better code implementation. They suggested removing PyTorch implementations for testing and enabling **eager execution** on JAX or TensorFlow, as well as utilizing Python's `breakpoint()` for tracking variable changes during line-by-line code execution.

- **Navigating Colab's Features**: To assist with coding on Google Colab, tips were shared such as using `function_name` for documentation lookup, `object_name.__class__` to find an object's class, and `inspect.getsource` to print a class's source code efficiently.

- **Gratitude Expression**: A member acknowledged the community help with a simple **"🙏"** emoji.

- **Link to Prior Inquiry**: A member referenced a past question asked in the **ask-for-help** section by providing a Discord channel link, noting their improved understanding of PyTorch since the initial query.

- **Request for Dialogue System Paper**: A request was made for research papers or work related to building a multi-turn dialogue system for intelligent customer service, indicating an interest in instructional problem-solving capabilities within the chat system.

- **Mathematical Breakdown of Samplers Needed**: In search of mathematical insights, a member requested recommendations for papers on sampling methods that followed after **ddpm** and **ddim**, seeking to focus only on the foundational samplers in the field.
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1227135169396477962)** (15 messages🔥): 

- **Diving into Computer Vision with TensorFlow**: A member inquired about resources or a roadmap for starting deep learning with computer vision using **TensorFlow**.

- **Contrastive Loss Requires Large Batch Size**: It was discussed that contrastive loss benefits from large batch sizes, and techniques like accumulation or checkpointing could be a workaround for limited compute resources. However, concerns about *batchnorm* not updating correctly with accumulated large batches were raised.

- **Efficient Watermark Removal from Millions of Images**: A member asked for tools to automatically remove watermarks from a large number of images. The [repository](https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy) for an AI watermark removal tool and its associated YouTube video were suggested.

- **Monitoring GPU Usage**: For those without access to a task manager, it was pointed out that the command `nvidia-smi` can be used to monitor GPU usage, and `nvidia-smi -l` allows for continuous monitoring over time. Another member mentioned seeking a way to log metrics in real time during model training.

**Link mentioned**: <a href="https://github.com/Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy">GitHub - Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy: Aladdin-Persson-AI-Watermark-Destroy Public</a>: Aladdin-Persson-AI-Watermark-Destroy Public. Contribute to Firdavs-coder/Aladdin-Persson-AI-Watermark-Destroy development by creating an account on GitHub.

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1227199066346098739)** (7 messages): 

- **GPT-2 and Summarization Issues**: A member reported issues using **GPT-2** for text summarization, even when following instructions from a [HuggingFace course](https://huggingface.co/learn/nlp-course/chapter7/5) suggesting its potential application for that purpose. The difficulty arises despite the dataset and task being described as straightforward.

- **Mistral Meets RAG With Underwhelming Results**: One participant indicated **disappointing outcomes** when combining **Mistral 7B** with **RAG (Retrieval-Augmented Generation)**, experiencing significantly subpar results.

- **Pinning Down the `TL;DR:`**: In response to the above GPT-2 issue, another user suggested the problem might be related to era-specific prompting, particularly "TL;DR" used for summarization instructions, implying a potential temporal mismatch in prompting strategies.

- **Sculpting Discord Bot Personality with Llama**cpp**: A user queried about methods to craft a **Discord bot character using llamacpp**, seeking a way to steer the bot's behavior beyond simple prompting. They also expressed an interest in tracking the conversation history to maintain context.

- **Multi-Model Evaluation Using Cosine Similarity**: A complex evaluation strategy for language models was discussed, involving the use of **cosine similarity** between embedding vectors to assess whether models incorporate specific knowledge points and tutoring principles in their outputs. This prompted another member to suggest a **weighted approach** to pooling embeddings, to better tailor the evaluation to the context's demands.
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1227297185712963636)** (18 messages🔥): 

- **Save Custom Modules in Diffusers**: A user encountered an error when trying to save a custom `nn.Module`; after adding mixins to the module, they were able to **resolve the issue**.

- **Schedulers/Samplers Behavior Explained**: In a discussion about **diffusers**, a user was clarified on why the quality of images varies with different numbers of **denoising steps**. Ancestral sampler was specifically mentioned and explanations were provided on how the scheduler interpolates between noised and denoised images.

- **Understanding Schedulers/Samplers Maths**: A user asked for recommendations on **papers to read** for understanding the mathematics behind basic schedulers/samplers beyond ddpm and ddim.

- **Multi-GPU Inference with SDXL**: A user enquired about performing inference with **MultiControlnet (SDXL)** across multiple GPUs. Guidance to use **🤗 Accelerate** and **PyTorch Distributed** for distributed inference was provided, but challenges of the pipeline requiring more than 10GBs were noted.

- **Layer Decomposer Search**: A member requested **information or tools** related to a Layer Decomposer that separates and complements images, much like a tool found on [cre8tiveai.com](https://cre8tiveai.com/ld).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Noqkb8Z9xzD782BjfA6oRsGtV35N_XhB?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://huggingface.co/docs/diffusers/main/en/training/distributed_inference">Distributed inference with multiple GPUs</a>: no description found</li><li><a href="https://cre8tiveai.com/ld"> Layer Decomposer（Layer separation AI）｜Image and video editing AI tool: cre8tiveAI</a>: An AI-based SaaS that solves a variety of photo and illustration editing tasks in under 10 seconds, such as automatic painting and increasing the resolution of images and videos, as well as clipping, ...</li><li><a href="https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement">Distributed inference with multiple GPUs</a>: no description found
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1227353404842709126)** (4 messages): 

- **Gemini Pro 1.5 & GPT-4 Turbo Expand Horizons**: Meet the [Gemini Pro 1.5](https://openrouter.ai/models/google/gemini-pro-1.5) with a 1M token context and GPT-4 Turbo with vision capabilities now at [openai/gpt-4-turbo](https://openrouter.ai/models/openai/gpt-4-turbo), bringing new advancements to the OpenRouter model lineup.
- **Enhanced `logit_bias` Support Rolls Out**: The `logit_bias` parameter, enabling users to influence model output more granularly, has been extended to additional models including [Nous Hermes 2 Mixtral](https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo) and various Llama and Mistral models.
- **Farewell to Lesser Used Models**: Models like jebcarter/Psyfighter-13B and jondurbin/bagel-34b-v0.2 are set for discontinuation, with a 2-week grace period before returning a 404 error, and migtissera/synthia-70b will redirect to xwin-lm/xwin-lm-70b from April 15th.
- **New Mixtral 8x22B Unveiled**: The [Mixtral 8x22B](https://openrouter.ai/models/mistralai/mixtral-8x22b), a base model with instruct capabilities, has been launched; feedback and discussions are encouraged in the designated Discord channel.
- **Updates & Price Reductions Announced**: The Gemma 7B model has been updated, and reduced pricing is now offered for models including [LZLV 70B](https://openrouter.ai/models/lizpreciatior/lzlv-70b-fp16-hf), [Databricks DBRX 132B Instruct](https://openrouter.ai/models/databricks/dbrx-instruct), and [Nous Hermes 2 Mixtral 8x7B DPO](https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mixtral-8x22b>)">Mixtral 8x22B by mistralai | OpenRouter</a>: Mixtral 8x22B is a large-scale language model from Mistral AI. It consists of 8 experts, each 22 billion parameters, with each token using 2 experts at a time.  It was released via [X](https://twitter...</li><li><a href="https://openrouter.ai/models/google/gemma-7b-it>)">Gemma 7B by google | OpenRouter</a>: Gemma by Google is an advanced, open-source language model family, leveraging the latest in decoder-only, text-to-text technology. It offers English language capabilities across text generation tasks ...</li><li><a href="https://openrouter.ai/models/lizpreciatior/lzlv-70b-fp16-hf>)">lzlv 70B by lizpreciatior | OpenRouter</a>: A Mythomax/MLewd_13B-style merge of selected 70B models. A multi-model merge of several LLaMA2 70B finetunes for roleplaying and creative work. The goal was to create a model that combines creativity ...</li><li><a href="https://openrouter.ai/models/databricks/dbrx-instruct>)">DBRX 132B Instruct by databricks | OpenRouter</a>: DBRX is a new open source large language model developed by Databricks. At 132B, it outperforms existing open source LLMs like Llama 2 70B and Mixtral-8x7B on standard industry benchmarks for language...</li><li><a href="https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo>)">Hermes 2 Mixtral 8x7B DPO by nousresearch | OpenRouter</a>: Nous Hermes 2 Mixtral 8x7B DPO is the new flagship Nous Research model trained over the [Mixtral 8x7B MoE LLM](/models/mistralai/mixtral-8x7b).  The model was trained on over 1,000,000 entries of prim...</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5)">Gemini Pro 1.0 by google | OpenRouter</a>: Google&#x27;s flagship text generation model. Designed to handle natural language tasks, multiturn text and code chat, and code generation.  See the benchmarks and prompting guidelines from [Deepmind]...</li><li><a href="https://openrouter.ai/models/openai/gpt-4-turbo)">GPT-4 Turbo by openai | OpenRouter</a>: The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling. Training data: up to Dec 2023.  This model is updated by OpenAI to point to the lates...</li><li><a href="https://openrouter.ai/models/nousresearch/nous-hermes-2-mixtral-8x7b-dpo)">Hermes 2 Mixtral 8x7B DPO by nousresearch | OpenRouter</a>: Nous Hermes 2 Mixtral 8x7B DPO is the new flagship Nous Research model trained over the [Mixtral 8x7B MoE LLM](/models/mistralai/mixtral-8x7b).  The model was trained on over 1,000,000 entries of prim...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct)">Mistral 7B Instruct by mistralai | OpenRouter</a>: A 7.3B parameter model that outperforms Llama 2 13B on all benchmarks, with optimizations for speed and context length.  This is v0.1 of Mistral 7B Instruct. For v0.2, use [this model](/models/mistral...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-2-13b-chat)">Llama v2 13B Chat by meta-llama | OpenRouter</a>: A 13 billion parameter language model from Meta, fine tuned for chat completions</li><li><a href="https://openrouter.ai/models/meta-llama/llama-2-70b-chat)">Llama v2 70B Chat by meta-llama | OpenRouter</a>: The flagship, 70 billion parameter language model from Meta, fine tuned for chat completions. Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned ve...</li><li><a href="https://openrouter.ai/models/mistralai/mixtral-8x7b-instruct)">Mixtral 8x7B by mistralai | OpenRouter</a>: A pretrained generative Sparse Mixture of Experts, by Mistral AI. Incorporates 8 experts (feed-forward networks) for a total of 47B parameters. Base model (not fine-tuned for instructions) - see [Mixt...
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/)** (1 messages): 

stonedjesusape: Fuck
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1227152405221609512)** (166 messages🔥🔥): 

- **Discussing Model Integrations**: A user inquired about integrating a new LLM API into OpenRouter; they were directed to DM for setting up a chat between both companies. **Louisgv** is handling the integration discussions.
  
- **Rate Limit Confusion**: Users expressed uncertainty regarding **OpenRouter's rate limits** on new models like **Gemini 1.5 Pro**, with clarifications on the heavy rate limitations for preview models that typically allow around 10 requests per minute.

- **Pricing and Token Estimates on OR**: There was a detailed conversation around the **pricing of Gemini models**, with **louisgv** explaining that tokens are counted as individual characters for billing purposes. This led to discussions about the **potential impacts** on token pricing, especially with languages like Chinese.

- **Imminent Updates Teased**: **Alexatallah** hinted at possible news forthcoming, following observations of a large amount of model updates on a single day, including **Mixtral 8x22b** being added to providers.

- **Technical Adaptations for Hermes DPO**: A user **hanaaa__** mentioned needing to patch SillyTavern for better performance with **Hermes DPO** providers, indicating issues with **TogetherAI’s latency**. They also noted that the **OpenRouter website experiences crashes** on certain pages when accessed via an iPhone.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cloud.google.com/blog/products/ai-machine-learning/google-cloud-gemini-image-2-and-mlops-updates">Google Cloud Gemini, Image 2, and MLOps updates | Google Cloud Blog</a>: Vertex AI adds expanded Gemini 1.5 access, the new CodeGemma model, enhancements to Imagen, and new MLOps features.</li><li><a href="https://deepinfra.com/databricks/dbrx-instruct">databricks/dbrx-instruct - Demo - DeepInfra</a>: DBRX is an open source LLM created by Databricks. It uses mixture-of-experts (MoE) architecture with 132B total parameters of which 36B parameters are active on any input. It outperforms existing open...</li><li><a href="https://cloud.google.com/blog/topics/google-cloud-next/welcome-to-google-cloud-next24">Welcome to Google Cloud Next ‘24 | Google Cloud Blog</a>: Google Cloud CEO Thomas Kurian provides an overview of all the news and customer momentum from Google Cloud Next ‘24.</li><li><a href="https://openrouter.ai/models/google/gemma-7b-it:free">Gemma 7B by google | OpenRouter</a>: Gemma by Google is an advanced, open-source language model family, leveraging the latest in decoder-only, text-to-text technology. It offers English language capabilities across text generation tasks ...</li><li><a href="https://openrouter.ai/models?o=pricing-high-to-low">OpenRouter</a>: Browse models on OpenRouter
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1227142494446293012)** (5 messages): 

- **Meta's Generosity in GPU Hours**: Meta backed a significant study on [LLM knowledge capacity](https://arxiv.org/abs/2404.05405), providing 4.2 million GPU hours, which comically translates to nearly **half a millennium** of compute time.
- **Scaling Laws Courtesy of Meta**: A notable scaling laws research has been sponsored by Meta, which required a jaw-dropping 4.2 million GPU hours reflecting Meta's commitment to propelling AI research forward.
- **Porting GPT-2 to CUDA**: An enthusiast mentioned their current project of porting GPT-2 training code to CUDA, which could become a remarkable benchmark for the AI community, and shared the [llm.c repository](https://github.com/karpathy/llm.c/tree/master/dev/cuda).
- **Formation of a Working Group for CUDA Development**: In response to expressed interests, a working group is set to be formed to foster collaboration on CUDA-related projects, indicating a healthy, community-driven approach to AI development.
- **Meta's Impressive AI Hardware Specs Revealed**: Details of Meta's next-generation AI hardware, boasting **354 TFLOPS/s** at only 90 watts, were discussed, with an accompanying blog post outlining Meta's robust investment in AI infrastructure.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ZeyuanAllenZhu/status/1777513016592040248">Tweet from Zeyuan Allen-Zhu (@ZeyuanAllenZhu)</a>: Our 12 scaling laws (for LLM knowledge capacity) are out: https://arxiv.org/abs/2404.05405. Took me 4mos to submit 50,000 jobs; took Meta 1mo for legal review; FAIR sponsored 4,200,000 GPU hrs. Hope t...</li><li><a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/">no title found</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/tree/master/dev/cuda">llm.c/dev/cuda at master · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1227157865085534278)** (1 messages): 

- **Excitement for C code in CUDA**: A member expressed enthusiasm about integrating **C code implementations of algorithms** into fast CUDA. They mentioned the possibility of adding this to their library and inquired about the compatibility between **MIT license** and **Apache 2.0 license**, seeking advice from anyone knowledgeable about licenses.
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1227359912347308092)** (1 messages): 

- **Matrix Multiplication Performance Explained**: A post highlights the importance of matrix shapes in performance with a focus on tiling and memory layouts. A matrix multiplication example (`[M x K] @ [K x N]`) is provided with the optimal configuration being **A: M=2047, K=N=2048** because it avoids **unaligned memory layouts** which impact performance negatively.

- **Offer for Answer Key**: The linked [blog post](https://www.thonking.ai/p/answer-key-what-shapes-do-matrix) discusses the performance of matrix multiplication shapes, offering the first answer publicly and providing further answers in exchange for readers' solutions to questions. This encourages engagement and helps deepen understanding of the material copresented.

**Link mentioned**: <a href="https://www.thonking.ai/p/answer-key-what-shapes-do-matrix">Answer Key: What Shapes Do Matrix Multiplications Like?</a>: Companion to https://www.thonking.ai/p/what-shapes-do-matrix-multiplications

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1227551275869143041)** (1 messages): 

- **pmpp book lecture viewing party proposal**: A member is organizing a viewing party for the **University of Illinois lectures** on the *pmpp book*. They offered to share a Zoom link to go through the lectures, which are 1 hour to 1 hour and 15 minutes long, with pauses for discussion, proposing early CET time or a later slot on weekdays.
  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1227192278859386920)** (7 messages): 

- **Ring-Flash Attention Query**: A member questioned the `if step <= comm.rank:` condition in the [ring-flash-attention code](https://github.com/zhuzilin/ring-flash-attention/blob/55ff66fd35f329dfcc24ce7a448bfdd532865966/ring_flash_attn/ring_flash_attn.py#L32), asking why it doesn't iterate over all key-value pairs for all hosts.
- **Clarifying Causal Self-Attention**: Additional context was given explaining that in *causal self-attention* each query does not need to attend to all key-values but only those prior to it.
- **Experimentation with State Space Models**: One member expressed interest in testing how well state space models can do 'no-in-head' attention (NiH) and **specifically asked if the process could be run for a mamba model**.
- **Collaborative Work on Flash Attention**: There's ongoing collaborative work, as stated by a member, on creating **educational flash attention examples**, which is in progress and available on [GitHub](https://github.com/cuda-mode/ring-attention/tree/naive_flash_attn_examples/naive_flash_attn).
- **Commitment to Model Testing**: It was mentioned that a member will attempt to run the ring-flash-attention code on a **mamba model** to see its effectiveness.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/cuda-mode/ring-attention/tree/naive_flash_attn_examples/naive_flash_attn">ring-attention/naive_flash_attn at naive_flash_attn_examples · cuda-mode/ring-attention</a>: ring-attention experiments. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.</li><li><a href="https://github.com/zhuzilin/ring-flash-attention/blob/55ff66fd35f329dfcc24ce7a448bfdd532865966/ring_flash_attn/ring_flash_attn.py#L32">ring-flash-attention/ring_flash_attn/ring_flash_attn.py at 55ff66fd35f329dfcc24ce7a448bfdd532865966 · zhuzilin/ring-flash-attention</a>: Ring attention implementation with flash attention - zhuzilin/ring-flash-attention
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1227462918367612939)** (4 messages): 

- **Celebrating Personal Milestones**: A member shares that they will turn 28 in May and expresses a potentially unpopular opinion: **Staying abreast with the entire AI field is futile and perhaps counterproductive**. Instead, they advocate for a more selective approach to information, filtering out the noise and focusing on what's truly important to them.
- **A Nod to Pop Culture**: The server's picture is recognized as **Goku**, a character from the anime series *Dragon Ball*.
- **Milestone for the Community**: The server celebration as it just surpassed **5000 members**.
- **Quality Over Quantity in Learning**: A member responds to a sentiment about information overload, recommending a **once-a-week reading routine** and emphasizing problem-driven learning rather than consumption-focused habits for better intellectual engagement.
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1227229143003824190)** (5 messages): 

- **Correction in Puzzle Logic**: A message pointed out a small error in **puzzle 11**, stating that the summation should be over the shared index \( l \). 
- **Pull Request for Puzzle 11 Fix**: A member agreed with the summation correction in **puzzle 11** and mentioned the need for `B_MID` to represent the block size on the MID dimension, subsequently creating a pull request to address the issue. Here is the [GitHub Pull Request](https://github.com/srush/Triton-Puzzles/pull/10).

**Link mentioned**: <a href="https://github.com/srush/Triton-Puzzles/pull/10">minor on puzzle 11 by ZhaoyueCheng · Pull Request #10 · srush/Triton-Puzzles</a>: fix formula on puzzle 11 to sum over dimension L    add B_MID on puzzle 11 for the parameter on block size to loop over the MID dimension

  

---


**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1227138905740935231)** (96 messages🔥🔥): 

- **Progress with Half-Quadratic Quantization (HQQ) Implementation**: Some basic placeholder code for inference using **HQQ+** was shared ([code example](https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L808)), indicating that **HQQ** alone only looks at the weights but **HQQ+**, which includes calibration, may be required for better performance.

- **Mixed Results with Marlin Kernel**: One member discussed their experience with the [Marlin kernel](https://github.com/IST-DASLab/marlin), noting that while Marlin reports up to 4x speed-up for fp16 x int4 matrix multiplication, the results on their A6000 Ada were underwhelming, also mentioning slight errors introduced by the kernels.

- **Quantization Techniques Discussion**: There was an exchange around the use of Marlin and HQQ quantization techniques with the suggestion of [perplexity evaluation scripts](https://discord.com/channels/1189498204333543425/1225499037516693574/1226798701293342793) to measure effective perplexity, aiming to achieve results similar to GPTQ.

- **Benchmark Concerns and Perplexity in Quantized Models**: Members compared perplexity scores of different models with modifications, noting discrepancies and seeking consistency with expected performance, identified by a perplexity of around 5.3 on wikitext with group-size=64.

- **Tuning and Testing HQQ Quantization**: Technical discussion ensued about the quantization settings for **HQQLinear**, particularly the importance of using `quant_scale=False, quant_zero=False` in the quantization settings. A detailed chat about execution speed raises concerns as to why **AOInt4** kernels are slower on some hardware compared to `torch.matmul` with **HQQLinear** and the potential causes ([issue demonstration](https://gist.github.com/mobicham/4b08fb0bdf4c3872e5bbf68ec9803137)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L137-L139">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L808">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/main/scripts/convert_hf_checkpoint.py#L89">gpt-fast/scripts/convert_hf_checkpoint.py at main · pytorch-labs/gpt-fast</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - pytorch-labs/gpt-fast</li><li><a href="https://github.com/IST-DASLab/marlin">GitHub - IST-DASLab/marlin: FP16xINT4 LLM inference kernel that can achieve near-ideal ~4x speedups up to medium batchsizes of 16-32 tokens.</a>: FP16xINT4 LLM inference kernel that can achieve near-ideal ~4x speedups up to medium batchsizes of 16-32 tokens. - IST-DASLab/marlin</li><li><a href="https://github.com/zhxchen17/gpt-fast">GitHub - zhxchen17/gpt-fast: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python.</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - zhxchen17/gpt-fast</li><li><a href="https://github.com/mobiusml/hqq/blob/master/examples/llama2_benchmark/quant_llama2_hqq_demo.py">hqq/examples/llama2_benchmark/quant_llama2_hqq_demo.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/mobiusml/hqq/blob/ao_int4_mm/hqq/core/torch_lowbit.py#L135">hqq/hqq/core/torch_lowbit.py at ao_int4_mm · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch/pytorch/blob/8aa08b8b9d1fab2a13dc5fbda74c553cb2a08729/aten/src/ATen/native/cuda/int4mm.cu#L805-L860">pytorch/aten/src/ATen/native/cuda/int4mm.cu at 8aa08b8b9d1fab2a13dc5fbda74c553cb2a08729 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch-labs/gpt-fast/pull/155">testing HQQ [not for land] by HDCharles · Pull Request #155 · pytorch-labs/gpt-fast</a>: Stack from ghstack (oldest at bottom):  -&gt; #155  Summary: hqq wikitext: {&#39;word_perplexity,none&#39;: 12.698986130023261, &#39;word_perplexity_stderr,none&#39;: &#39;N/A&#39;, &#39;byte_perplexi...</li><li><a href="https://gist.github.com/mobicham/4b08fb0bdf4c3872e5bbf68ec9803137">hqq_eval_int4mm.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/zhxchen17/gpt-fast/commit/f7c8151e749ec1d8c3f6d3361dcfce4feec5b3b0">HQQ 4 bit llama 2 7b · zhxchen17/gpt-fast@f7c8151</a>: export MODEL_REPO=meta-llama/Llama-2-7b-hf scripts/prepare.sh $MODEL_REPO python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4-hqq --groupsize 64 python generate.py --...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/)** (1 messages): 

kerenzhou: I like the corresonding code on the figure
  

---


**CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1227359195519778858)** (42 messages🔥): 

- **Early CUDA Forward Pass Strides**: An update reports all forward layers have been implemented for a project, with **efficient attention** being the last hurdle. The first round of optimizations led to a status termed as "mostly in good shape," indicating a **potential for performance gains** over the initial code.

- **LLM.C Repository Highlighted as a Learning Resource**: A GitHub repository named [llm.c](https://github.com/karpathy/llm.c) has been shared and praised as a valuable resource for learning and honing CUDA skills. It involves LLM training using simple, raw C/CUDA.

- **OpenMP Usage in LLM.C Discussed**: Members have noted that OpenMP is employed in the llm.c codebase, with one suggesting that *OMP offloading* could replace direct CUDA usage for simplicity and cross GPU vendor compatibility, though there is uncertainty about support on Windows.

- **Debugging Performance Issues in Custom CUDA Code**: There has been a **performance comparison** between different versions of CUDA kernels. The 'flash' version was initially 3x slower than expected; however, after further testing by various members on different hardware, it showed a speed-up and workings towards resolving the slowdown were underway.

- **Gap to Close in Pure CUDA Forward Pass Performance**: The recently pushed pure CUDA forward pass shows an execution time of 111ms/iter compared to PyTorch's 180ms/iter, but there is still a gap compared to PyTorch optimized with compilation and tensor cores, which runs at 26ms/iter. The push included a comparison of performance metrics and an aim to close this **performance gap**. The code can be found on the GitHub repository [karpathy/llm.c](https://github.com/karpathy/llm.c).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/msaroufim/5defcd59aed4364846d034ac01eb6cfd">gist:5defcd59aed4364846d034ac01eb6cfd</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/karpathy/llm.c/blob/8386e5393c61ec2faf706f3040e68127c2f08398/dev/cuda/attention_forward.cu#L14">llm.c/dev/cuda/attention_forward.cu at 8386e5393c61ec2faf706f3040e68127c2f08398 · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpa">karpa - Overview</a>: karpa has 13 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/karpathy/llm.c">GitHub - karpathy/llm.c: LLM training in simple, raw C/CUDA</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/blob/8386e5393c61ec2faf706f3040e68127c2f08398/dev/cuda/gelu_forward.cu#L170">llm.c/dev/cuda/gelu_forward.cu at 8386e5393c61ec2faf706f3040e68127c2f08398 · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/blob/8386e5393c61ec2faf706f3040e68127c2f08398/dev/cuda/gelu_forward.cu#L53-L60">llm.c/dev/cuda/gelu_forward.cu at 8386e5393c61ec2faf706f3040e68127c2f08398 · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/blob/8386e5393c61ec2faf706f3040e68127c2f08398/dev/cuda/residual_forward.cu#L42-L48">llm.c/dev/cuda/residual_forward.cu at 8386e5393c61ec2faf706f3040e68127c2f08398 · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/karpathy/llm.c/blob/8386e5393c61ec2faf706f3040e68127c2f08398/dev/cuda/gelu_forward.cu#L149">llm.c/dev/cuda/gelu_forward.cu at 8386e5393c61ec2faf706f3040e68127c2f08398 · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/cuda-mode/lectures/blob/main/lecture8/occupancy.cu#L31">lectures/lecture8/occupancy.cu at main · cuda-mode/lectures</a>: Material for cuda-mode lectures. Contribute to cuda-mode/lectures development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1227185465212538961)** (108 messages🔥🔥): 

- **Misconceptions about Whisper Capabilities**: There's discussion clarifying that **Whisper** is a speech-to-text (STT) model, not text-to-speech (TTS), and while **Ollama** doesn't inherently support Whisper, it's possible to use Whisper locally or with a different backend provided by the same developer.
- **LangChain Use Cases Explored**: Members shared insights into the practical applications of LangChain, such as evaluating retrieval systems, with one member pointing to [an example involving RAG metrics](https://docs.smith.langchain.com/cookbook/testing-examples/ragas) to assess retrieval-augmented generation performance.
- **Comparing LangChain with OpenAI's API**: A member inquired about the benefits of using LangChain over OpenAI's API for building AI assistants. The consensus suggests that if integrations beyond OpenAI's offerings are not needed, LangChain might not add significant value.
- **LangChain Functionality Debates**: Users discussed the capability of **Very Large Language Models (VLLM)** to support function calling, with the suggestion to use **Outlines**, which provides structured text generation capabilities.
- **Beginner's Query on Starting AI/ML Career**: A new member requested guidance on where to begin their career in AI/ML after learning basic Python and MySQL.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/get_started/introduction#api-reference>).">Introduction | 🦜️🔗 LangChain</a>: LangChain is a framework for developing applications powered by large language models (LLMs).</li><li><a href="https://python.langchain.com/docs/guides/structured_output#openai>).">[beta] Structured Output | 🦜️🔗 LangChain</a>: It is often crucial to have LLMs return structured output. This is</li><li><a href="https://python.langchain.com/docs/guides/structured_output#openai>)">[beta] Structured Output | 🦜️🔗 LangChain</a>: It is often crucial to have LLMs return structured output. This is</li><li><a href="https://python.langchain.com/docs/modules/model_io/output_parsers/quick_start#get-started>)">Quickstart | 🦜️🔗 LangChain</a>: Language models output text. But many times you may want to get more</li><li><a href="https://docs.smith.langchain.com/cookbook/testing-examples/ragas">RAG evaluation with RAGAS | 🦜️🛠️ LangSmith</a>: Ragas is a popular framework that helps you evaluate your Retrieval Augmented Generation (RAG) pipelines.</li><li><a href="https://js.langchain.com/docs/integrations/chat/openai#withstructuredoutput-->).">ChatOpenAI | 🦜️🔗 Langchain</a>: You can use OpenAI&#x27;s chat models as follows:</li><li><a href="https://python.langchain.com/docs/use_cases/data_generation#extraction-from-generated-examples>)">Synthetic data generation | 🦜️🔗 LangChain</a>: Open In Colab</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: Structured Text Generation</a>: Structured Text Generation. Contribute to outlines-dev/outlines development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/1497>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/ggerganov/whisper.cpp">GitHub - ggerganov/whisper.cpp: Port of OpenAI&#39;s Whisper model in C/C++</a>: Port of OpenAI&#39;s Whisper model in C/C++. Contribute to ggerganov/whisper.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/)** (1 messages): 

lhc1921: https://python.langchain.com/docs/integrations/llms/azure_openai/
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1227283037461418055)** (3 messages): 

- **Swipe Right for Automation**: Introducing **TinderGPT**, an app designed to automate messaging on Tinder, promising to save users time and secure dates automatically. Find the code and contribute to the future of digital dating on [*GitHub*](https://github.com/GregorD1A1/TinderGPT).

- **Chat with Retrieval Augmented Generation Locally**: **everything-rag** offers a fully customizable, local chatbot assistant with free, 100% local functionality, using Langchain and ChromaDB vectorized databases. Explore the HuggingFace space [here](https://huggingface.co/spaces/as-cle-bert/everything-rag), star the [*GitHub* repo](https://github.com/AstraBert/everything-rag), and read about the significance of open-source LLMs in the associated [blog post](https://astrabert.github.io/hophop-science/Attention-and-open-source-is-all-you-need/).

- **Analyzing Structured Output Across LLMs**: A performance analysis of structured output is presented, comparing popular open and closed source large language models. Check the findings and methodology on the [*GitHub page*](https://github.com/mattflo/structured-output-performance).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/as-cle-bert/everything-rag">everything-rag - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://github.com/GregorD1A1/TinderGPT">GitHub - GregorD1A1/TinderGPT</a>: Contribute to GregorD1A1/TinderGPT development by creating an account on GitHub.</li><li><a href="https://github.com/mattflo/structured-output-performance">GitHub - mattflo/structured-output-performance: A comparison of structured output performance among popular open and closed source large language models.</a>: A comparison of structured output performance among popular open and closed source large language models. - mattflo/structured-output-performance
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1227236485326045257)** (3 messages): 

- **AI Agent for Virtual Fashion Trials**: A member shared a YouTube video titled ["Future of E-commerce?! Virtual clothing try-on agent"](https://youtu.be/C94pTaKoLbU), showcasing an AI agent they built that can generate images of a model wearing various clothes and create social media posts.
- **Guidance on Publishing an AI Agent**: A member inquired about how to publish and create a UI for an AI agent they've developed, seeking tutorials for guidance. Another member recommended learning web development as a necessary step for accomplishing this task.

**Link mentioned**: <a href="https://youtu.be/C94pTaKoLbU">Future of E-commerce?! Virtual clothing try-on agent</a>: I built an agent system which will autonomously iterate &amp; generate img of AI model wearing certain cloth and produce millions+ social postsFree access to run...

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1227282302783197216)** (4 messages): 

- **Multimodal RAG Revolutionizes Pill Identification**: A new **Multimodal RAG application** has been highlighted that is capable of identifying pills from images, integrating visual data with a descriptive database in the medical domain. The blog post by @activeloop demonstrates its usefulness in recognizing pharmacy products and can be found at [this Twitter link](https://t.co/QkuLzs34IJ).
- **Event Alert: Building Enterprise-Grade RAG**: @llama_index announced a collaboration with @traceloop and @getreflex to show the essential components for constructing enterprise-level **Retrieval-Augmented Generation (RAG)**. Advanced parsing and observability features are among the core tools to be discussed at the event, with more details available on [Twitter](https://t.co/ZkhvlI4nnx).
- **MetaGPT Steps into ICLR 2024 with RAG**: Introducing MetaGPT by Hong et al., a multi-agent framework premiering at ICLR 2024 that treats agents as a software company's diverse roles, from PMs to engineers, solving tasks through collaboration. RAG-enhanced MetaGPT adds a cutting-edge twist to this framework, with more details shared at [this link](https://t.co/sAF41j0uL4).
- **Controllability in Agent Execution via Execution Stopping Tools**: Highlighting the importance of execution control tools in agent systems, @llama_index shared insights into how these tools are integral to a travel agent's booking confirmation process and an **agentic RAG** system's search and reply function. Interested readers can follow the conversation on [Twitter](https://t.co/ByGOaqgWMd).
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1227176829790457857)** (104 messages🔥🔥): 

- **OpenAI Agent vs Gemini LLM Adaptation**: Users discussed adapting an [openaiagent example notebook](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_tool_call_parser.ipynb) found in LlamaIndex's documentation to work with the *Gemini LLM*, suggesting code modifications like replacing `OpenAI llm` with `gemini LLM` and using `ReActAgent` instead of `OpenAIAgent`.

- **RAG Optimization Quest**: A user sought advice for optimizing *Retrieval Augmented Generation (RAG)* with short documents, leading to a recommendation to review *RAG 101* on [Gradient AI](https://gradient.ai/blog/rag-101-for-enterprise) and referring to the *MTEB leaderboard* for embeddings.

- **Tool Creation within LlamaIndex**: A conversation unfolded around how to create new tools within LlamaIndex and dynamically add them to `OpenAIAgent`. After a detailed exchange, a member successfully managed to create tools by using `FunctionTool`, despite various challenges.

- **Debugging LLM Prompting Issues**: A member asked how to see the exact prompt sent to an LLM for debugging purposes. They were directed to a [logging guide](https://discord.com/channels/1059199217496772688/1227269649440313357/1227271613234282637), eventually discovering they needed a particular type of chat mode that conditionally uses RAG to reduce unnecessary LLM calls.

- **Integration Woes and Example Requests**: Users queried about project setups, integration instructions with Open Source tools, and example use cases. References included an end-to-end guide video for the SEC Insights project on [YouTube](https://youtu.be/2O52Tfj79T4?si=CYUcaBkc9P9g_m0P) and source code on [GitHub](https://github.com/run-llama/sec-insights/tree/main/backend/app/chat).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.useinstructor.com/">Welcome To Instructor - Instructor</a>: no description found</li><li><a href="https://tenor.com/view/im-a-sad-panda-peetie-south-park-crying-disappointed-gif-21544015">Im A Sad Panda Peetie GIF - Im A Sad Panda Peetie South Park - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/disco-dance-party-happy-zebra-gif-16162722">Disco Dance GIF - Disco Dance Party - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/mindblown-omg-triggered-gif-19814900">Mindblown Omg GIF - Mindblown Omg Triggered - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/storage/docstore/postgres/">Postgres - LlamaIndex</a>: no description found</li><li><a href="https://gradient.ai/blog/rag-101-for-enterprise">Gradient Blog: RAG 101 for Enterprise </a>: RAG 101 for Enterprise Gradient Team</li><li><a href="https://youtu.be/C94pTaKoLbU">Future of E-commerce?! Virtual clothing try-on agent</a>: I built an agent system which will autonomously iterate &amp; generate img of AI model wearing certain cloth and produce millions+ social postsFree access to run...</li><li><a href="https://github.com/microsoft/autogen/blob/main/notebook/agentchat_inception_function.ipynb">autogen/notebook/agentchat_inception_function.ipynb at main · microsoft/autogen</a>: A programming framework for agentic AI. Discord: https://aka.ms/autogen-dc. Roadmap: https://aka.ms/autogen-roadmap - microsoft/autogen</li><li><a href="https://github.com/run-llama/sec-insights">GitHub - run-llama/sec-insights: A real world full-stack application using LlamaIndex</a>: A real world full-stack application using LlamaIndex - run-llama/sec-insights</li><li><a href="https://github.com/run-llama/llama_index/tree/main/llama-index-core/llama_index/core/chat_engine">llama_index/llama-index-core/llama_index/core/chat_engine at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/sec-insights/tree/main/backend/app/chat">sec-insights/backend/app/chat at main · run-llama/sec-insights</a>: A real world full-stack application using LlamaIndex - run-llama/sec-insights</li><li><a href="https://youtu.be/2O52Tfj79T4?si=CYUcaBkc9P9g_m0P">Discover LlamaIndex: SEC Insights, End-to-End Guide</a>: secinsights.ai is a full-stack app that uses the Retrieval Augmented Generation (RAG) capabilities of LlamaIndex to answer questions about SEC 10-K &amp; 10-Q do...</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/">Vector Stores - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/AzureAISearchIndexDemo/?h=azure">Azure AI Search - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/c01beee1fab7c0de22869ce74f34ebd1f1d54722/llama-index-core/llama_index/core/tools/function_tool.py#L31">llama_index/llama-index-core/llama_index/core/tools/function_tool.py at c01beee1fab7c0de22869ce74f34ebd1f1d54722 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/c01beee1fab7c0de22869ce74f34ebd1f1d54722/llama-index-core/llama_index/core/tools/types.py#L97">llama_index/llama-index-core/llama_index/core/tools/types.py at c01beee1fab7c0de22869ce74f34ebd1f1d54722 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1227176894680666152)** (2 messages): 

- **Cookbook Gains a Fan**: A member expressed **appreciation** for the **openaiagent example** provided in the Llama Index GitHub cookbook, finding it "quite useful," and inquired about the possibility of **similar resources for Gemini LLM**. The relevant resource can be found at this [GitHub notebook](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/openai_agent_tool_call_parser.ipynb).

- **New Member Seeks Clarification on API Key**: A new participant in the discussion expressed **confusion** about the operation of services, inquiring whether an **API Key for OpenAI** is required to make things work, referencing guidance from documentation.
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1227410537776943104)** (87 messages🔥🔥): 

- **Pixart Sigma Performance Review**: Members discussed the performance of **Pixart Sigma** on a 3090, indicating fast and promising results with a **prompt execution time of 8.26 seconds**. However, the outputs were described as "mangled" with users noting the presence of "bleed" in the results from current open models.

- **Mistral 22b x 8 Release Chat**: There was a mention of **Mistral 22b x 8** being out. A magnet link for **mixtral-8x22b** was shared, followed by responses indicating excitement and queries about potential relation to **mistral-large**.

- **Skepticism Surrounding Ella SDXL and SD3**: Discussion turned to the unlikelihood of Ella SDXL becoming available and skepticism about the benefits of **Stable Diffusion V3 (SD3)**, comparing it unfavorably to **Terminus** and **Pixart Sigma**. Members also weighed in on industry responses to the Sora announcement affecting competitors such as Stability, Pika labs, Runway, and Midjourney.

- **New Audio Generation Solutions Emerging**: There was a buzz around **Udio**, an app backed by artists for intuitive music creation via text-prompts, and a New TTS engine by the **Huggingface team** that allows voice prompting. 

- **AI Acceleration Hardware Buzz**: Members discussed the new Meta Training and Inference Accelerator (**AI-MTIA**) with its impressive specs, reflecting on the trend of major tech companies developing their own AI acceleration hardware solutions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://2Fopen.demonii.com%3A1337%2Fannounce&tr=http%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce`">no title found</a>: no description found</li><li><a href="https://www.udio.com/songs/renWwtB7Zqk2mqZamEHHgJ>">no title found</a>: no description found</li><li><a href="https://x.com/udiomusic/status/1778045337833193720">Tweet from udio (@udiomusic)</a>: Our goal is to make Udio a game-changing tool for both musicians and non-musicians alike, and we are excited to be backed by leading artists @iamwill and @common.   8/11</li><li><a href="https://news.ycombinator.com/item?id=39992817">Show HN: Sonauto – a more controllable AI music creator | Hacker News</a>: no description found</li><li><a href="https://x.com/udiomusic/status/1778045322654003448">Tweet from udio (@udiomusic)</a>: Introducing Udio, an app for music creation and sharing that allows you to generate amazing music in your favorite styles with intuitive and powerful text-prompting.  1/11</li><li><a href="https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1227327910432739621)** (9 messages🔥): 

- **Reevaluating "Zero-Shot" Generalization in Multimodal Models**: [A recent paper](https://arxiv.org/abs/2404.04125) questions the degree to which "zero-shot" generalization truly exists in multimodal models like CLIP and Stable-Diffusion. Analysis across various models and datasets suggests that performance heavily depends on the prominence of concepts within the pretraining data.

- **Data Quality Over Quantity for CLIP Models**: When testing CLIP models on less common concepts, improving data filtering and selection for quality and diversity is crucial, possibly more so than simply increasing the quantity of data.

- **Google Advances with Larger Griffin Model**: Google reportedly releases a model with a new Griffin architecture, featuring an additional 1 billion parameters, boasting improved performance and throughput over long contexts. The details can be found on their [subreddit post](https://www.reddit.com/r/MachineLearning/comments/1b3leks/deepmind_introduces_hawk_and_griffin_r/).

- **New Study Challenges Traditional LLM Training Methods**: A [groundbreaking paper](https://arxiv.org/abs/2404.03715) presents an alternative to Reinforcement Learning from Human Feedback (RLHF) by optimizing directly over "pair-wise" or general preferences, showing significant performance improvements even with a 7 billion parameter model.

- **Performance Boost in Large Language Models**: The aforementioned method provides a significant performance leap compared to other leading models, indicating the potential advantages of pair-wise optimization strategies over traditional point-wise reward methods.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.03715">Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences</a>: This paper studies post-training large language models (LLMs) using preference feedback from a powerful oracle to help a model iteratively improve over itself. The typical approach for post-training L...</li><li><a href="https://arxiv.org/abs/2404.04125">No &#34;Zero-Shot&#34; Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance</a>: Web-crawled pretraining datasets underlie the impressive &#34;zero-shot&#34; evaluation performance of multimodal models, such as CLIP for classification/retrieval and Stable-Diffusion for image gener...</li><li><a href="https://www.reddit.com/r/singularity/comments/1bzzreq/google_releases_model_with_new_griffin/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1227147549325135914)** (51 messages🔥): 

- **GPT-4 Makes a Grand Entrance**: Excitement buzzes in the **OpenInterpreter** Discord about the newly released **GPT-4** model, which boasts notable improvements over its predecessor, including **integrated vision capabilities** and enhanced performance. The chatter also includes observations that GPT-4 is *actually 3 times faster* and some users report their firsthand experience with its speed, acknowledging its prompt response times and swift operation.

- **GPT-4 Turbocharged and Under the Radar**: Amidst the release frenzy, a member notes that there's a lack of widespread notice or detailed information on **GPT-4**, with no substantial chatter outside the community and only OpenAI's release page serving as a primary info source on the [continuous model upgrades](https://platform.openai.com/docs/models/continuous-model-upgrades).

- **Mixtral and OI Compatibility Queries**: Some discussions have arisen about the potential match-up of **Mixtral 8x22b** with **OpenInterpreter (OI)**, as users compare it against past models like the 8x7b and consider implications for performance within OI's framework.

- **Enthusiasm for Command r+**: A member raves about a model called **Command r+**, praising it as *the best model ever used* for role-playing (RP) and following instructions precisely, indicating it feels like a better version of GPT-3.5 and outperforms the old GPT-4 in benchmarks, especially with the right prompts.

- **Compute Conundrums for Command r+**: A conversation surfaces regarding the compute power required for running **Command r+** locally, with members discussing their setups, and one reporting that *even a 4090 isn't enough* for optimal performance, indicating that significant hardware might be needed.
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1227145348942921738)** (38 messages🔥): 

- **Troubleshooting 01 Hotspot Reconnection Issues**: A member resolves an issue reconnecting to the WiFi and server settings page for 01 by suggesting a factory reset and navigating to [captive.apple.com](http://captive.apple.com) to trigger the portal. Mention of removing old WiFi credentials was also advised.

- **Installation Hurdles with 01 an Windows 11**: Members report issues where talking to the installed 01 yields no response, despite the microphone functioning correctly. Suggestions included checking the Python script and ensuring [sounddevice](https://pypi.org/project/sounddevice/) is installed.

- **Constructing the 01 from GitHub Repository**: An individual shared their experience of buying parts from the Bill of Materials (BOM) and 3D printing the body from files available in the 01 GitHub [repository](https://github.com/OpenInterpreter/01?tab=readme-ov-file).

- **Clarification on Raspberry Pi Requirements for 01**: A discussion clarified that a Raspberry Pi is not required for 01 and that running Open Interpreter or 01OS on any computer suffices. For those interested in adding Raspberry Pi to their setup, the conversation suggested initiating a broader discussion in a dedicated forum.

- **Local IP Use for 01 Server Configuration**: A new 01 user successfully connects their device to the server using their MacBook's local IP address, after facing issues and confusion with configuring and understanding ngrok domains.

- **Order Updates and Customer Service Inquiries**: In response to a customer order status inquiry, it was mentioned that emails would be sent out once there are updates. All current order statuses are humorously referred to as "still cooking".
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dashboard.ngrok.com/cloud-edge/domains/new">ngrok - Online in One Line</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/01?tab=readme-ov-file#01-server">GitHub - OpenInterpreter/01: The open-source language model computer</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/01?tab=readme-ov-file">GitHub - OpenInterpreter/01: The open-source language model computer</a>: The open-source language model computer. Contribute to OpenInterpreter/01 development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1227274131423367280)** (45 messages🔥): 

- **Google Drops a Non-Transformer Surprise**: Google quietly launched a 2 billion parameter recurrent linear attention model named [Griffin](https://huggingface.co/google/recurrentgemma-2b), which is a significant development beyond the recent CodeGemma release, drawing comparisons to the RWKV architecture. The related research paper available on [arXiv](https://arxiv.org/abs/2402.19427).
  
- **Rumblings of Rapid Model Releases**: The conversation touches on rapid and somewhat unexpected model releases, such as **Mixtral**, which may be a result of competitive pressure from other anticipated model releases like llama 3 smol and Cohere.

- **OpenAI Drops Their Own News**: OpenAI's tweet alluded to an intriguing development, but specifics were not discussed in the messages—only a [link to a tweet from OpenAI](https://vxtwitter.com/OpenAI/status/1777772582680301665) was provided without further context.

- **New Model Excitement with Mixtral**: Mixtral, a new model, is stirring up excitement, and differences from previous models like Mistral and Miqu are highlighted in a [Twitter conversation](https://fxtwitter.com/sophiamyang/status/1777978822199017728).

- **Public Human Eval Blog Proposal**: A member discusses the possibility of starting a blog dedicated to unbiased human evaluations of new model releases, expressing frustration over the current focus on benchmark scores rather than practical utility for developers. There's a call for contributions and participation in this endeavor.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/sophiamyang/status/1777978822199017728">Tweet from Sophia Yang, Ph.D. (@sophiamyang)</a>: @LiHongtu12138 neither. it&#39;s a completely new model.</li><li><a href="https://x.com/jeethu/status/1777703476195196982?s=46">Tweet from Jeethu Rao (@jeethu)</a>: Looks like Google has just silently released a 2B recurrent linear attention based model (non-transformer based, aka the Griffin architecture). This is a bigger deal than CodeGemma, IMO. AFAIK, the cl...</li><li><a href="https://x.com/realmrfakename/status/1777882147707322479?s=46">Tweet from mrfakename (@realmrfakename)</a>: UPDATE: A mod on Mistral Discord server confirmed that the model is not any previous model, it&#39;s a completely new model  ↘️ Quoting mrfakename (@realmrfakename)   The new Mixtral model is...  (per...</li><li><a href="https://fxtwitter.com/jphme/status/1778028110954295486">Tweet from Jan P. Harries (@jphme)</a>: First AGIEval results for @MistralAI s new 8x22b model are in, destroying all other open source (base) models  - 🤯
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1227367608282779699)** (14 messages🔥): 

- **Decoding RLHF in Modern LLM Training**: Sebastian Raschka published a [breakdown on RLHF](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) as a crucial part of LLM training, impacting model helpfulness and safety, comparing ChatGPT's and Llama 2's use of RLHF, with regular updates on alternatives.
- **Rejection Sampling Raises Questions**: A user reading the article was confused by **Rejection Sampling**, a concept implying the use of the best model generations for PPO updates, and sought insights on why this might be superior to learning from average or worse generations.
- **Exploring PPO Through Online Resources**: Another user aimed to clarify their understanding of PPO by consulting [Cameron Wolfe's newsletter](https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo), acknowledging a lack of prior knowledge in RL.
- **Rejection Sampling's Role Clarified**: **Nathan Lambert** clarified that Rejection Sampling is applied on the entire instruction dataset before continued training, acknowledging that this process isn't well documented in relevant papers, and required direct correspondence with authors.
- **Considerations on Rejection Sampling's Efficacy**: Lambert further explained the probable rationale for Rejection Sampling: **most of the data is likely low quality**, implying that filtering for higher-quality examples before PPO could lead to more stable training outcomes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives">LLM Training: RLHF and Its Alternatives</a>: I frequently reference a process called Reinforcement Learning with Human Feedback (RLHF) when discussing LLMs, whether in the research news or tutorials. RLHF is an integral part of the modern LLM tr...</li><li><a href="https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo">Proximal Policy Optimization (PPO): The Key to LLM Alignment</a>: Modern policy gradient algorithms and their application to language models...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1227442004267368479)** (7 messages): 

- **Chasing Eclipses in Texas**: Members shared personal experiences regarding travel to Texas for an **optimal viewing experience** of an eclipse or celestial event.
- **Brief Encounter with the Skies**: Despite cloudy conditions, one member expressed joy at catching a **glimpse** of the event, considering themselves to have **great luck**.
- **A Cosmic Resemblance**: A member noted that the celestial sight resembled **the eyeball in the sky** from Netflix's series "3-Body," evoking an image from pop culture.
  

---


**Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1227341028877406209)** (2 messages): 

- **Universal Quirk in ML Discords**: A member humorously noted the widespread use of the <:berk:750111476483752166> emoji across various machine learning Discord communities.
- **Humor Shared in the Community**: A user found something amusing and declared it "Too funny to not share" in the channel.
  

---


**Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1227435523916693568)** (10 messages🔥): 

- **Debating the Merits of RLHF Post-Training Improvements**: A member highlighted a [new paper](https://arxiv.org/abs/2404.03715) discussing the improvement of large language models (LLMs) through iterative feedback using an oracle, potentially challenging typical Reinforcement Learning from Human Feedback (RLHF) methods which rely on reward maximization.

- **Size Matters in Model Efficacy**: A question was raised comparing a 7B model to GPT-4, suggesting the smaller model might outperform the larger.

- **Skepticism About Benchmark Optimization**: Members expressed skepticism toward LLM-evaluated benchmarks, pointing out that while benchmarks can be optimized, it doesn't necessarily reflect better fundamental model performance.

- **Practical Model Improvement Philosophy**: A member has disclosed a preference for tangible improvements in models through **better data** and **better scaling** rather than what they deemed as "bullshit" new papers on the topic.

- **Benchmarks as Imperfect Proxies**: There was acknowledgment that while benchmarks like alpacaeval may be broken once optimizing starts, they can still be useful as a temporary measure of a model's capabilities.

**Link mentioned**: <a href="https://arxiv.org/abs/2404.03715">Direct Nash Optimization: Teaching Language Models to Self-Improve with General Preferences</a>: This paper studies post-training large language models (LLMs) using preference feedback from a powerful oracle to help a model iteratively improve over itself. The typical approach for post-training L...

  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1227136719510306866)** (3 messages): 

- **Scaling Laws for LLMs Unveiled**: A new paper introduces **12 scaling laws for Language Model (LLM) knowledge capacity**, which could be pivotal in the era of large language models. The research required a massive amount of resources, with Meta's FAIR team sponsoring **4,200,000 GPU hours** for this study. [Read the paper here](https://arxiv.org/abs/2404.05405).
- **Exploration of Quantization and MoE**: The paper also explores **inference and quantization**, revealing that quantizing model weights to **int8** doesn't harm the knowledge capacity of even maximally-capable models, and that **Mixture of Experts (MoE)** models with 32 experts preserve knowledge capacity efficiently. [See the detailed results](https://fxtwitter.com/zeyuanallenzhu/status/1777513026243174543?s=46).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/zeyuanallenzhu/status/1777513016592040248?s=46">Tweet from Zeyuan Allen-Zhu (@ZeyuanAllenZhu)</a>: Our 12 scaling laws (for LLM knowledge capacity) are out: https://arxiv.org/abs/2404.05405. Took me 4mos to submit 50,000 jobs; took Meta 1mo for legal review; FAIR sponsored 4,200,000 GPU hrs. Hope t...</li><li><a href="https://fxtwitter.com/zeyuanallenzhu/status/1777513026243174543?s=46">Tweet from Zeyuan Allen-Zhu (@ZeyuanAllenZhu)</a>: Results 8/9: scaling laws for quantization and MoE.  // Quantization to int8 does not hurt knowledge capacity even for models at max capacity =&gt; 2bit of knowledge can be stored to int8  // MoEs wit...
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1227323705118560288)** (52 messages🔥): 

- **Refactoring Tinygrad for Efficiency**: A discussion on streamlining tinygrad's codebase is underway, with a particular focus on reducing line count and enhancing the code for review readiness, as well as addressing backend peculiarities that necessitate JIT support for non-disk backends.
- **The Quest for a Weight Agnostic Network**: One member expresses interest in creating a weight agnostic network using tinygrad to train a game, intending to experiment with ReLU activations.
- **Merging MNIST into Tinygrad**: Efforts to integrate MNIST more closely into tinygrad are highlighted, with [Pull Request #4122](https://github.com/tinygrad/tinygrad/pull/4122) showcasing the move and revealing a compiler bug on AMD, calling action to add a test in CI to catch such issues.
- **Variable Naming in Abstractions3**: There's a debate about the necessity of variable names in the context of abstractions3, with the suggestion being made that variables should be defined by their IDs. This led to a change where **var_vals** will be a global dict instead of being in each **ScheduleItem**.
- **CI Performance and Test Discussion**: Concerns are raised regarding CI performance regression and missing tests, particularly for the functionality `copy_from_fd`, something to be addressed in a subsequent [pull request](https://github.com/tinygrad/tinygrad/pull/4125/files).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/actions/runs/8633930065/job/23668153464">no more underlying diskbuffer, that&#39;s just the device (#4129) · tinygrad/tinygrad@ee457a4</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - no more underlying diskbuffer, that&#39;s just the device (#4129) · tinygrad/tinygrad@ee457a4</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4124">abstractions3 is currently wishful thinking by geohot · Pull Request #4124 · tinygrad/tinygrad</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4122">spend 5 lines to bring mnist into the repo by geohot · Pull Request #4122 · tinygrad/tinygrad</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/actions/runs/8625081714/job/23641105062">spend 5 lines to bring mnist into the repo (#4122) · tinygrad/tinygrad@fea774f</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - spend 5 lines to bring mnist into the repo (#4122) · tinygrad/tinygrad@fea774f</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4125/files">create schedule has global vars by geohot · Pull Request #4125 · tinygrad/tinygrad</a>: no description found
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1227199232448921600)** (13 messages🔥): 

- **Step-by-Step Guide to Custom Accelerators**: A user shared a [step-by-step guide](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/addingaccelerator.md) on adding custom accelerators to tinygrad, pointing to a GitHub repository for detailed instructions and illustrations.
- **Seeking Network Examples**: One member was in search of neat network examples using tinygrad and was directed to review the `examples/` directory within the tinygrad repository.
- **Discussing 'Double Reducc'**: Users were discussing an issue named 'double reducc' and there seemed to be a consensus and recognition of the problem, indicating a collaborative effort towards a resolution.
- **Converting Tensor to Array in Tinygrad**: A query was raised about turning tensors into arrays within the tinygrad environment. Another user responded by recommending the use of `.numpy()` on the tensor to accomplish this conversion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/mesozoic">mesozoic - Overview</a>: mesozoic has 39 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/addingaccelerator.md">tinygrad-notes/addingaccelerator.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1227434265914900501)** (40 messages🔥): 

- **Mixtral Model Evolution**: The new **Mixtral 8x22B model** was discussed, presumably having around 140 billion parameters, and a dataset of 1.5GB operating at rank32 with unexpectedly low loss. Members are curious if this version is instruction tuned or a base model.
- **Quantization and Model Size Limits**: Community members are looking into **quantization** for practical use and expressing concerns about the feasibility of running larger models like **Mixtral 8x22B** with resources available to typical developers. There's interest in finding a balance between model size and utility.
- **Rapid Community Contributions**: A contributor has already started uploading the new big model, **Mixtral-8x22B**, to Hugging Face, demonstrating the community's quick response to developments. The link to the repository was shared: [Hugging Face - Mixtral-8x22B](https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF/tree/main).
- **Seeking Compatible Frontends**: A question was raised about a web self-hostable frontend that is compatible with various APIs, including OpenAI's and Google's. No specific solutions were mentioned in the responses.
- **Generative AI Hackathon Announcement**: Samsung Next 2024 Generative AI Hackathon is announced for May 11th in New York, focusing on tracks in **Health & Wellness** and **Mediatech**. The link for details and applications was provided: [Samsung Next AI Hackathon](https://lu.ma/nextgenainyc).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF/tree/main">MaziyarPanahi/Mixtral-8x22B-v0.1-GGUF at main</a>: no description found</li><li><a href="https://lu.ma/nextgenainyc">Samsung Next 2024 Generative AI Hackathon · Luma</a>: 🚀 What&#x27;s Happening Apply to join the Samsung Next 2024 Generative AI Hackathon! We&#x27;ll explore two tracks: Health &amp; Wellness: Harness the power of AI in improving healthcare outcomes,...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1227253703359074384)** (4 messages): 

- **Axolotl Dataset Versioning Feature on the Horizon**: A member expressed interest in adding dataset versioning to Axolotl, noting its absence. A response indicated that **dataset versioning** had not been previously requested and encouraged the member to proceed with a contribution.

- **LoRA Layer Initialization Technique Sparks Interest**: Sharing a tip from CFGeek's tweet, the group discussed a novel initialization method for **LoRA** layers that involves using the SVD of the original weight matrix for better fine-tuning results. The technique, termed **PiSSA** (Principal Singular Values and Singular Vectors Adaptation), which reportedly improves finetuned performance, is detailed in an [arXiv abstract](https://arxiv.org/abs/2404.02948) and a corresponding [GitHub repository](https://github.com/GraphPKU/PiSSA).

**Link mentioned**: <a href="https://x.com/cfgeek/status/1777556286047166673?s=46&t=hIokEbug9Pr72tQFuXVULA">Tweet from Charles Foster (@CFGeek)</a>: YES! If you initialize a LoRA layer based on the SVD of the original weight matrix (with its top singular values & vectors), you get significantly better fine-tuning results.  This is a straight-up fr...

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1227168958973607937)** (9 messages🔥): 

- **Pre-training with Norwegian Articles**: A member is preparing to pre-train a large language model with a dataset of arts in Norwegian to enhance grammar capabilities. They inquired about the best way to split the articles and received advice to consider using one row per article, possibly in a `.jsonl` format.

- **Seeking Function Calling Fine-tuning DataSet**: A request was made for a good dataset suitable for JSON mode or function calling, specifically to fine-tune **LoRAs for function calling** with axolotl; however, no recommendations were provided within the current message history.

- **Hardware Capability Query for the mixtral-qlora-fsdp Model**: A member questioned whether the **mixtral-qlora-fsdp** model would fit on a dual 24GB GPU setup, but no follow-up information or answers were given.

- **Fixing Empty Queue Error**: A user experiencing an empty queue error was advised to check for an empty condition before iterating, presented with refactored code as a potential solution.

- **Code Refactoring for Simplicity**: An example of refactoring code was given, simplifying a function that checks for a stop token from several lines to just one, enhancing code efficiency.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1227241338903199844)** (2 messages): 

- **Seeking Function-Calling & JSON Datasets**: A member inquired about datasets for function-calling or JSON parsing.
- **Agent-FLAN Dataset Shared**: Another member responded with a dataset suggestion, providing a [link to the Agent-FLAN dataset](https://huggingface.co/datasets/internlm/Agent-FLAN) on HuggingFace. This dataset includes AgentInstruct, Toolbench, and custom negative agent samples, designed for effective agent tuning in large language models.

**Link mentioned**: <a href="https://huggingface.co/datasets/internlm/Agent-FLAN">internlm/Agent-FLAN · Datasets at Hugging Face</a>: no description found

  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1227206147388342343)** (8 messages🔥): 

- **Hints at C Formatting in Mojo**: It has been hinted that while Mojo developers wait for Python-style `f` strings, they can use old C-style formatting by importing `_printf as printf` with the caveat that this feature *“may not be around for ever”*.
- **API Documentation Summarized for Beginners**: A member shared [a link to a Notion page](https://ripple-haddock-938.notion.site/Mojo-40a425eab9104fde8b3e11a2f5a3e078) that provides translated API documentation in a summarized format, aiming to help beginners.
- **Exploring Contributions Beyond Mojo stdlib**: A discussion for potential contributors looking to get involved with Mojo or MAX projects, with suggestions for web development on *lightbug*, AI on *basalt*, or starting a new project.
- **Curated List of Mojo Resources**: Contribution opportunities and resources for Mojo can also be found on the curated list maintained on GitHub, known as [awesome-mojo](https://github.com/mojicians/awesome-mojo).
- **Call for Community Feedback on Mojo Traits**: A new discussion has been initiated regarding the use of traits in Mojo, and feedback from the broader community has been requested in a [GitHub discussion](https://github.com/modularml/mojo/discussions/2259).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ripple-haddock-938.notion.site/Mojo-40a425eab9104fde8b3e11a2f5a3e078?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://github.com/modularml/mojo/discussions/2259">[Proposal]: Deprecate Triple Dots (...) for Unimplemented Methods · modularml/mojo · Discussion #2259</a>: Motivation: Mojo aspires to be a seamless successor to Python++, adhering closely to Pythonic principles and fostering a positive experience for the Python community. The current practice of using ...</li><li><a href="https://github.com/mojicians/awesome-mojo">GitHub - mojicians/awesome-mojo: A curated list of awesome Mojo 🔥 frameworks, libraries, software and resources</a>: A curated list of awesome Mojo 🔥 frameworks, libraries, software and resources - mojicians/awesome-mojo
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1227304021505867826)** (2 messages): 

- **Modular Shares Update**: Modular tweeted an update which can be viewed on their official Twitter page. The specific content of the tweet was not shared in the message.

- **Another Modular Announcement**: A second tweet from Modular was posted, the details of which can be explored through the provided link. The exact nature or topic of the announcement was not mentioned in the chat.
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1227199221531148320)** (32 messages🔥): 

- **Simd Support Coming to Mojo**: Discussions indicate excitement around the addition of simd support to Mojo, with expectations of fascinating benchmark results following the update.
- **Concurrency Features are a Work in Progress**: Mojo currently supports async/await and coroutines; however, these features are unfinished. The coroutine API in Mojo differs from Python, and details can be found in the [Mojo documentation](https://docs.modular.com/mojo/stdlib/builtin/coroutine).
- **Mojo's Roadmap for Async Constructs**: The language currently lacks `async for` and `async with` constructs, and discussions link to the roadmap indicating a focus on essential core system programming features of Mojo, available [here](https://docs.modular.com/mojo/roadmap#no-async-for-or-async-with).
- **Running Mojo Natively on Intel Macs**: A user expresses a limitation with running Mojo natively on Intel Macs, relying on a VM for larger projects, although small tests are done within the playground.
- **Mojo-UI Efforts and Objective-C Integration**: A new project for a UI library specifically for Mojo called Mojo-UI is underway, with efforts focusing on Mac as the primary platform, raising questions about the future potential for integrating Objective-C or accessing the AppKit framework with Mojo. This integration could possibly require designing a binding layer between Mojo and Swift via C or C++, as suggested in recent discussions. The project is tracked on [GitHub](https://github.com/Moosems/Mojo-UI).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/coroutine">coroutine | Modular Docs</a>: Implements classes and methods for coroutines.</li><li><a href="https://docs.modular.com/mojo/roadmap#no-async-for-or-async-with">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://docs.modular.com/mojo/roadmap#lifetime-tracking-inside-collections">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://github.com/Moosems/Mojo-UI">GitHub - Moosems/Mojo-UI: A cross-platform GUI library for Mojo</a>: A cross-platform GUI library for Mojo. Contribute to Moosems/Mojo-UI development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo">GitHub - modularml/mojo: The Mojo Programming Language</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2252">[BUG] Compiler bug when typing async function pointer call return type · Issue #2252 · modularml/mojo</a>: Bug description The mojo compiler incorrectly types the result of calling an async function pointer. Expected behavior async fn() -&gt; Int functions return a Coroutine[Int] type when called. This Cor...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1227425197355700265)** (4 messages): 

- **MojoGeek Unveils Mojo GPT**: A member introduced a platform called **Mojo GPT** tailored for answering Mojo programming queries and sought community feedback. The platform can be tested and feedback provided at [Mojo GPT](https://chat.openai.com/g/g-RPORxvimH-mojogpt).

- **Serving Up Iterators for String Characters**: A helpful crosspost was shared for those needing an iterator over string characters, with a link directing to the relevant message on Discord.

- **mojo-ui-html Gets Exciting Updates**: A new update to **mojo-ui-html** includes *keyboard events* for creating video games or custom widgets, a new minimize window feature, and **CSS kwags** for additional per-element styling. Details and demonstrations are available on [GitHub](https://github.com/rd4com/mojo-ui-html/blob/main/demo_keyboard_and_css.mojo).

- **Lightbug Framework Gaining Momentum**: Contributions to the **Lightbug HTTP framework** were highlighted, including performance boosts, a pure Mojo-based client implementation, and comparisons showing Lightbug serving more requests per second than Python's Flask. The advancements and contributions can be explored further on [GitHub](https://github.com/saviorand/lightbug_http).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/rd4com/mojo-ui-html/blob/main/demo_keyboard_and_css.mojo">mojo-ui-html/demo_keyboard_and_css.mojo at main · rd4com/mojo-ui-html</a>: Immediate mode GUI, HTML, CSS, Work in progress, Mojo language - rd4com/mojo-ui-html</li><li><a href="https://github.com/saviorand/lightbug_http/issues/6).">Issues · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1227402169809113120)** (1 messages): 

- **Scrumtuous Achieves Top Google Ranking**: A member humorously announced their rank as #1 on Google for a high-value **Python keyword**, attributing the success to **Mojo**. There are no further details or links provided regarding the specific keyword or the content that achieved the ranking.
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1227316880541356122)** (2 messages): 

- **Seeking SYRK Implementation in Mojo**: A member inquired about an implementation of **SYRK** (symmetric rank-k update) in **Mojo** for the purpose of conducting some performance tests.
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - Issue 29
https://www.modular.com/newsletters/modverse-weekly-29
  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1227207975488454676)** (2 messages): 

- **Tuning into Prince**: A member whimsically suggests that the phrase "Purple flame" could inspire a song reminiscent of a famous Prince hit, humorously adapting the lyrics to "*Purple flame, purple flame...*".

- **Generics Shock**: Another member expresses astonishment at the mention of "Heterogeneous variadic generics," conveying a mixture of surprise and confusion at the complex programming concept.
  

---



**DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1227544935398506586)** (5 messages): 

- **Mixtral Model Conversion Scripts Shared**: A member shared the MoE Weights conversion script for a previous **Mixtral** model ([convert_mistral_moe_weights_to_hf.py](https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/convert_mistral_moe_weights_to_hf.py)) and the official conversion script for the new **Mixtral** release found on the Hugging Face GitHub repository ([convert_mixtral_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py)).

- **New Mixtral Model on Hugging Face**: An updated **Mixtral-8x22B** model has been uploaded to Hugging Face, with a [model card](https://huggingface.co/v2ray/Mixtral-8x22B-v0.1) and conversion scripts provided by the uploader, later cloned to an official community repo.

- **Misinterpretation on Model Performance Corrected**: There was a correction about a **performance comparison** between *GPT-4*, *Claude Sonnet*, and the *Mixtral model*; the original statement mistakenly referred to a different model named *command-r+*, not **Mixtral**.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/DiscoResearch/mixtral-7b-8expert/blob/main/convert_mistral_moe_weights_to_hf.py">convert_mistral_moe_weights_to_hf.py · DiscoResearch/mixtral-7b-8expert at main</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py">transformers/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1227492138422370344)** (18 messages🔥): 

- **Mixtral's Magnet Link Shared**: A link to the **Mixtral 8x22b** model torrent was posted, providing a way to download this new AI model.
- **License Confirmation for Mixtral**: The Mixtral model is confirmed to be released under the **Apache 2.0 license**, with an instruct version anticipated to follow.
- **First AGIEval Results Are Promising**: A member highlighted the **Mixtral 8x22b** model's impressive performance in the *First AGIEval Results*, suggesting it outperforms other base models.
- **Benchmark Scores Released**: Benchmark scores for various datasets such as PIQA, BoolQ, and Hellaswag were shared, comparing the performance of **Mixtral 8x22B** and **Mixtral 8x7B** models.
- **Model Runs on Virtual Large Language Model (vLLM)**: It's noted that the benchmark scores are generated using a virtual Large Language Model setup with **4xH100 GPUs**, and there's a mention of the **MMLU task taking around 10 hours** on this configuration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/MistralAI/status/1777869263778291896?t=vKiT9FUuVbYAhjjg5kOHyw&s=33">Tweet from Mistral AI (@MistralAI)</a>: magnet:?xt=urn:btih:9238b09245d0d8cd915be09927769d5f7584c1c9&dn=mixtral-8x22b&tr=udp%3A%2F%http://2Fopen.demonii.com%3A1337%2Fannounce&tr=http%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce</li><li><a href="https://huggingface.co/v2ray/Mixtral-8x22B-v0.1">v2ray/Mixtral-8x22B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/discussions/3#6616af73203bf9d751696a84">mistral-community/Mixtral-8x22B-v0.1 · MMLU - 77</a>: no description found
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1227194075162214486)** (24 messages🔥): 

- **New LLM Merging Tool Unveiled**: A new library for merging multiple Large Language Model (LLM) experts named [**mergoo**](https://github.com/Leeroo-AI/mergoo) has been shared, which claims to simplify and improve the efficiency of the merging process. This tool is noted to be inspired by the branch train mix paper from March.

- **RAG Benchmarking Reveals Odd Behavior**: [DiscoResearch/DiscoLM_German_7b_v1 model](https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval) shows disparate performance outcomes dependent on the placement of a line break in the ChatML template; without it, accuracy drops significantly on some tasks within a newly created RAG benchmark.

- **Line Break Impact Investigated**: The discovery of a line break affecting benchmarks triggered discussions about a potential data loading/formatting script issue, and whether this could relate to broader erratic benchmark results. It prompted a plan to review training data application, with a mention of updating data for an upcoming 8x22 model.

- **Model Formatting Issues Explored**: Conversation around the [tokenizer configuration](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/blob/main/tokenizer_config.json#L48) for **DiscoLM_German_7b_v1**, speculating about whether modifying the tokenizer config might address performance anomalies.

- **Generalizability of Line Break Issue in Question**: The unique sensitivity to line break formatting has raised questions about whether this could be an issue specific to **DiscoResearch/LeoLM** models, or a more general phenomenon affecting other models as well. The topic remains open for further testing and investigation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1/blob/main/tokenizer_config.json#L48">tokenizer_config.json · DiscoResearch/DiscoLM_German_7b_v1 at main</a>: no description found</li><li><a href="https://github.com/Leeroo-AI/mergoo">GitHub - Leeroo-AI/mergoo: A library for easily merging multiple LLM experts, and efficiently train the merged LLM.</a>: A library for easily merging multiple LLM experts, and efficiently train the merged LLM. - Leeroo-AI/mergoo</li><li><a href="https://github.com/Crystalcareai/BTX">GitHub - Crystalcareai/BTX</a>: Contribute to Crystalcareai/BTX development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/deutsche-telekom/Ger-RAG-eval">deutsche-telekom/Ger-RAG-eval · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/lighteval/blob/main/community_tasks/german_rag_evals.py">lighteval/community_tasks/german_rag_evals.py at main · huggingface/lighteval</a>: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron. - hug...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/5ed29393e34cf57b24a20ac1bafa3a94272ac3f5/src/axolotl/prompt_strategies/dpo/chatml.py#L86">axolotl/src/axolotl/prompt_strategies/dpo/chatml.py at 5ed29393e34cf57b24a20ac1bafa3a94272ac3f5 · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



**LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/1227332519016530031)** (16 messages🔥): 

- **Good Morning with a Tweet**: A member greeted the channel with a "gm" and shared a [Twitter link](https://twitter.com/OpenAI/status/1777772582680301665) potentially related to new updates or discussions from OpenAI.
- **Surprising Benchmark Results**: Wenquai reported unexpected findings where **Sonnet and Haiku** performed better than **GPT-4 Turbo and Opus** in a quick vision benchmark, linking to a [Colab research document](https://colab.research.google.com/drive/1s7KvljSkXKRfinqG248QZIZvROf0pk4x?usp=sharing) for review.
- **Exploration of GPT-4 Turbo Features**: The **GPT-4 Turbo**'s function calling and JSON mode were highlighted as promising for building with vision models, sparking interest in further benchmarking these features.
- **Is It GPT-4.5 or not?**: Members joked about the incremental nature of the latest model improvements, with one stating it feels more like a **4.25** update, while others cited OpenAI employees' claims of enhanced reasoning capabilities.
- **Comparison of AI Coding Abilities**: There was a brief exchange discussing the coding capabilities of the latest models, where potrock mentioned no coding issues using the model in cursor while others brought up comparisons with **Gemini 1.5** and discussed the benefits of **copilot++**.

**Link mentioned**: <a href="https://colab.research.google.com/drive/1s7KvljSkXKRfinqG248QZIZvROf0pk4x?usp=sharing">Google Colaboratory</a>: no description found

  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1227189540457021541)** (15 messages🔥): 

- **LLM Help Command Performance**: A user reported that the `llm --help` command was slow, taking more than 2 seconds to run. Their concern was whether this indicated a potential security issue like being hacked.
- **Benchmarking LLM Help**: In response to concerns about `llm --help` performance, a different user shared a fast benchmark result: `0,50s user 0,10s system 94% cpu 0,624 total`.
- **Timing LLM on Different Setups**: A follow-up by the original user indicated that `llm --help` took 3.423 seconds on their setup, but only 0.800 seconds in a fresh docker container, suggesting that the slowdown might be related to system configuration rather than the `llm` tool itself.
- **Reinstallation Resolves Issues**: The user facing performance issues with `llm --help` found that reinstalling `llm` resolved both the speed problem and an error encountered when running Claude models, suggesting that a fresh install could alleviate certain operational problems.
- **LLM Command Hiccups on MacOS**: Another user experienced the `llm cmd` command hanging when run locally on macOS with iTerm2, while it worked fine on a remote Ubuntu server. They noted a customized shell environment, which they suspected might contribute to the issue, though the same configuration was working on Ubuntu.
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1227237932872630275)** (3 messages): 

- **Seeking Benchmark Comparisons**: A member inquired about a paper citing performance benchmarks for models like **phi-2**, **dolphin**, and **zephyr** on the **HumanEval dataset**.

- **Skepticism on Benchmarks**: A member expressed skepticism towards benchmarks, suggesting they can be gamed. However, they recommended a human-ranked leaderboard for trustworthy results, available at [arena.lmsys.org](https://arena.lmsys.org/).

- **First AGIEval Results for Mistral 8x22b**: The **Mistral 8x22b** model's first **AGIEval results** have been shared, indicating superior performance over other open source (base) models. The updates can be found in two tweets by Jan P. Harries, detailed [here](https://x.com/jphme/status/1778030213881909451) and [here](https://x.com/jphme/status/1778028110954295486).

**Link mentioned**: <a href="https://x.com/jphme/status/1778030213881909451">Tweet from Jan P. Harries (@jphme)</a>: @MistralAI first AGIEval results look great 👇 - thanks for releasing this beast, guys! 👏 https://x.com/jphme/status/1778028110954295486  ↘️ Quoting Jan P. Harries (@jphme)   First AGIEval results fo...

  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=Gb--4supXoo
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1227430928259743774)** (4 messages): 

- **Fine-Tuning GPU Usage**: A member discovered that using a **lower `-ngl` value** (number of GPU layers to use) resolved their issue, and they settled on `-ngl 3`. They noted that performance was significantly better with smaller models due to their GPU's limited memory.

- **Adaptive Layer Offloading in Question**: In the context of VRAM limitations, a member inquired if **llamafile** could potentially offload layers to fit a user's available VRAM instead of crashing, linking to their own configuration with a 1050 GPU.

- **ollama Offers LLM Flexibility**: A member praised **ollama** for its method of handling model layer distribution, sharing a specific GitHub link discussing the implementation details: [ollama server.go](https://github.com/ollama/ollama/blob/c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9/llm/server.go#L43).

**Link mentioned**: <a href="https://github.com/ollama/ollama/blob/c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9/llm/server.go#L43>">ollama/llm/server.go at c5c451ca3bde83e75a2a98ed9fd4e63a56bb02a9 · ollama/ollama</a>: Get up and running with Llama 2, Mistral, Gemma, and other large language models. - ollama/ollama

  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1227303320171974676)** (2 messages): 

- **Tuning into Remix Music AI**: A member shared excitement about a remix music model, describing it as "pretty fucking amazing" with a link to listen: [Loading Song...](https://linktones.synthtrails.com/linktone/kanye).
- **Call for Coding Support**: A user requested direct messaging for assistance with their code, reaching out to a specific member for help.

**Link mentioned**: <a href="https://linktones.synthtrails.com/linktone/kanye">SynthTrails</a>: no description found

  

---



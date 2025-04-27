---
id: c8e922a2-25da-4f4d-9509-3bd4e6f09448
title: Life after DPO (RewardBench)
date: '2024-05-28T00:04:01.538810Z'
original_slug: ainews-life-after-dpo
description: >-
  **xAI raised $6 billion at a $24 billion valuation**, positioning it among the
  most highly valued AI startups, with expectations to fund **GPT-5 and GPT-6
  class models**. The **RewardBench** tool, developed by Nathan Lambert,
  evaluates reward models (RMs) for language models, showing Cohere's RMs
  outperforming open-source alternatives. The discussion highlights the
  evolution of language models from Claude Shannon's 1948 model to GPT-3 and
  beyond, emphasizing the role of **RLHF (Reinforcement Learning from Human
  Feedback)** and the newer **DPO (Direct Preference Optimization)** method.
  Notably, some **Llama 3 8B reward model-focused models** are currently
  outperforming GPT-4, Cohere, Gemini, and Claude on the RewardBench
  leaderboard, raising questions about reward hacking. Future alignment research
  directions include improving preference datasets, DPO techniques, and
  personalization in language models. The report also compares xAI's valuation
  with OpenAI, Mistral AI, and Anthropic, noting speculation about xAI's
  spending on Nvidia hardware.
companies:
  - x-ai
  - openai
  - mistral-ai
  - anthropic
  - cohere
  - meta-ai-fair
  - hugging-face
  - nvidia
models:
  - gpt-3
  - gpt-4
  - gpt-5
  - gpt-6
  - llama-3-8b
  - llama-3
  - claude-3
  - gemini
topics:
  - reinforcement-learning-from-human-feedback
  - direct-preference-optimization
  - reward-models
  - rewardbench
  - language-model-history
  - model-evaluation
  - alignment-research
  - preference-datasets
  - personalization
  - transformer-architecture
people:
  - nathan-lambert
  - chris-manning
  - elon-musk
  - bindureddy
  - rohanpaul_ai
  - nearcyan
---


<!-- buttondown-editor-mode: plaintext -->**RLHF is all you need.**

> AI News for 5/24/2024-5/27/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**382** channels, and **9556** messages) for you. 
Estimated reading time saved (at 200wpm): **1079 minutes**.

It's a quiet US holiday weekend. 

- [X.ai raised $6b at a $24B valuation](https://x.ai/blog/series-b), immediately becoming one of the most highly valued AI startups. 
- We also released [Part 1 of our ICLR recap of best papers and talks](https://www.latent.space/p/iclr-2024-recap).
- Alex Reibman's [LlamaFS](https://github.com/iyaja/llama-fs) project from the Meta Llama 3 hackathon [went viral](https://twitter.com/llama_index/status/1794762651769430381): A local LLM-powered hard drive file organizer. Automatically rename and categorize messy files and directories with multi-modal AI.

Today's feature goes to [Nathan Lambert](https://x.com/natolambert), who is giving a guest lecture for [Chris Manning's CS224N](https://web.stanford.edu/class/cs224n/?ref=zgljl2012.com) (the full [suggested readings](https://web.stanford.edu/class/cs224n/) are worth a browse), and [released slides](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit#slide=id.g271884d3ab7_0_126) for his upcoming talk on the history and future of reward models and his work on [RewardBench](https://x.com/natolambert/status/1792314629932384343).

 ![image.png](https://assets.buttondown.email/images/32f346af-d635-4ac5-9182-9fe76a82d484.png?w=960&fit=max) 

<ol><li><b>History of Language Models (LMs)</b>: LMs have evolved significantly, from Claude Shannon's early English language model in 1948 to the powerful GPT-3 in 2020 and beyond. The transformer architecture, introduced in 2017, has been instrumental in this progress, enabling the development of models capable of generating increasingly complex and coherent text.</li><li><b>RLHF and DPO</b>: RLHF (Reinforcement Learning from Human Feedback) has been a key factor in the success of many popular language models, but it is complex and resource-intensive. DPO (Direct Preference Optimization), introduced in 2023, is a simpler and more scalable alternative that learns directly from human preferences. While DPO has shown promising results, RLHF still produces better outcomes in many cases.</li><li><b>RewardBench</b>: RewardBench is a tool for evaluating reward models (RMs) for LLMs. It provides insights into how RMs impact LLM capabilities and safety. Cohere's RMs have outperformed open-source models on RewardBench, highlighting the importance of continued research in this area.</li><li><b>Future Directions for Alignment Research</b>: Key areas for future research include collecting more data, particularly preference datasets, improving DPO techniques, exploring different model sizes, developing more specific evaluations beyond general benchmarks, and addressing personalization in language models.</li></ol>

[The RewardBench paper](https://arxiv.org/pdf/2403.13787) lists a collection of the most challenging reward model benchmarks:

 ![image.png](https://assets.buttondown.email/images/b9ecd1df-da28-4d46-8b0a-51972d2302aa.png?w=960&fit=max) 

and it is interesting that a few dedicated [Reward Model focused](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1) Llama 3 8B models are currently beating GPT4 and Cohere and Gemini and Claude in [the leaderboard](https://huggingface.co/spaces/allenai/reward-bench). Something there or reward hacking?

 ![image.png](https://assets.buttondown.email/images/27e3f5a0-2b42-4738-aa90-fb25189b4dd6.png?w=960&fit=max) 

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!


{% endif %}


---

# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**xAI Raises $6 Billion at $24 Billion Valuation**

- **Funding Details**: [@nearcyan](https://twitter.com/nearcyan/status/1794969115586883864) noted xAI raised $6 billion, **valuing the company at $24 billion**. [@bindureddy](https://twitter.com/bindureddy/status/1795120407550308556) congratulated xAI and Elon on the funding, stating it should be enough compute for **GPT-5 and 6 class models**.
- **Comparisons to Other AI Companies**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1795064230669848697) compared xAI's $24B valuation to other AI companies - OpenAI valued at $80-100B, Mistral AI at around $6B in April, and Anthropic at around $18.4B.
- **Speculation on Spending**: [@nearcyan](https://twitter.com/nearcyan/status/1794976668232323295) joked the money will go straight to Nvidia, **cutting out the middlemen**. He also [imagined](https://twitter.com/nearcyan/status/1794967177562349681) xAI saying a 100% Nvidia portfolio is a reasonable bet.

**Criticism of Elon Musk and xAI**

- **Yann LeCun's Criticism**: [@ylecun](https://twitter.com/ylecun/status/1795034443809165464) sarcastically suggested joining xAI if you can stand a boss who claims what you're working on will be solved next year, claims it will kill everyone, and spews conspiracy theories while claiming to want rigorous pursuit of truth.
- **Concerns about AI Hype and Regulation**: [@ylecun](https://twitter.com/ylecun/status/1794998977105981950) criticized the "Doomer's Delusion" of claiming AI will kill us all, monopolizing AI, requiring kill switches, banning open source, and scaring the public to get insane funding from clueless billionaires. He stated we don't even have a hint of a design for human-level AI yet.
- **Musk's Politics**: [@ylecun](https://twitter.com/ylecun/status/1795029416499708069) disliked Musk's vengeful politics, conspiracy theories and hype, while liking his cars, rockets, solar panels and satellite network. He [clarified](https://twitter.com/ylecun/status/1795135271358406946) this wasn't advocacy of the far left, as authoritarianism and scapegoating have been used by both extremes.

**AI Safety and Existential Risk Debate**

- **Counterarguments to AI Doomerism**: [@ylecun](https://twitter.com/ylecun/status/1795032310590378405) argued AI is not a natural phenomenon that will emerge and become dangerous, but something we design and build. He stated if a safe, controllable AI system that fulfills objectives better than humans exists, we'll be fine; if not, we won't build it. Currently, we don't even have a hint of a design for human-level AI.
- **Rebuttal to Regulation Proposals**: [@ylecun](https://twitter.com/ylecun/status/1795021895198179368) responded to a tweet saying AI must be monopolized by a small number of companies under tight regulation. He reasoned as if AI is a natural phenomenon rather than something we design and build, and compared it to how we made turbojets reliable before deploying them widely.
- **Debate on General Intelligence**: [@ylecun](https://twitter.com/ylecun/status/1794731261669355655) stated general intelligence, artificial or natural, does not exist. All animals have specialized intelligence and ability to acquire new skills. Much of intelligence is acquired through interaction with the physical world, which machines need to reproduce.

**Developments in AI and Robotics**

- **Naver Labs Robot Cafe**: [@adcock_brett](https://twitter.com/adcock_brett/status/1794761271272677704) reported Naver Labs made a Starbucks with over 100 robots called "Rookie" that bring drinks and perform various tasks.
- **Microsoft's AI Announcements**: [@adcock_brett](https://twitter.com/adcock_brett/status/1794761379565519187) shared Microsoft revealed Copilot+ PCs that run AI 20x faster than traditional PCs, ship with GPT-4o, and have a "Recall" feature to search things you've seen on screen.
- **Tokyo Robotics Demo**: [@adcock_brett](https://twitter.com/adcock_brett/status/1794761424297771284) noted Tokyo Robotics showed a new 4-fingered robotic hand that can grasp objects of any shape without prior knowledge of dimensions.
- **Other Developments**: [@adcock_brett](https://twitter.com/adcock_brett/status/1794761523044192272) mentioned MIT developing a robotic exoskeleton to help astronauts recover from falls. He also [shared](https://twitter.com/adcock_brett/status/1794761646012776680) IDEA Research unveiled object detection models, and [reported](https://twitter.com/adcock_brett/status/1794761713654415376) on Anthropic glimpsing into Claude's "mind".

**New AI Research Papers**

- **Grokked Transformers**: [@_akhaliq](https://twitter.com/akhaliq/status/1794912618882187678) shared a paper studying if transformers can implicitly reason over parametric knowledge. They found transformers can learn implicit reasoning but only through grokking (extended training past overfitting).
- **Stacking Transformers**: [@_akhaliq](https://twitter.com/akhaliq/status/1794938544336568380) posted about a paper on model growth for efficient LLM pre-training by leveraging smaller models to accelerate training of larger ones. 
- **Automatic Data Curation**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1794911168244646397) shared Meta's paper on automatic data curation for self-supervised learning, outperforming manually curated data.
- **Meteor**: [@_akhaliq](https://twitter.com/akhaliq/status/1794920194403336630) posted about Meteor, which leverages multifaceted rationale to enhance LLM understanding and answering capabilities.
- **AutoCoder**: [@_akhaliq](https://twitter.com/akhaliq/status/1794913439963402683) noted AutoCoder is the first LLM to surpass GPT-4 on HumanEval, with a versatile code interpreter.

**Debates and Discussions**

- **Math Skills vs Verbal Skills**: There was debate on a viral tweet claiming AI will favor verbal over math skills. [@bindureddy](https://twitter.com/bindureddy/status/1795149723243831508) argued you need math and analytical skills, not verbal skills, to instruct LLMs on hard problems. He also [disagreed](https://twitter.com/bindureddy/status/1794890347904192995) AI makes coding obsolete, stating design skills and AI tool use will be in demand.
- **Mechanical Interpretability**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1794950590692569227) cautioned against just doing what everyone else is doing in interpretability research, as that's not how the best research gets done. [@NeelNanda5](https://twitter.com/NeelNanda5) and [@aryaman2020](https://twitter.com/aryaman2020) discussed the lack of diversity in directions explored.
- **Sentience and Consciousness**: There were many tweets debating if LLMs and AI systems can be sentient or conscious. Key arguments were that we can't observe the self-awareness of another entity, attempts to define sufficient properties for sentience lead to bizarre conclusions, and current LLMs likely can't reflect on their internal states.

**Miscellaneous**

- **History Project**: [@alexandr_wang](https://twitter.com/alexandr_wang/status/1794843290233401809) is kicking off a next-gen history project to invent new ways to preserve stories and make history experiential using LLMs and AI.
- **Rabbit AI Opinions**: There were mixed views on Rabbit AI, with some criticizing it as a grift while others like [@KevinAFischer](https://twitter.com/KevinAFischer/status/1794746529430843675) were impressed and thought opening it up to developers could make it amazing.
- **Nvidia's Rise**: Many noted Nvidia's spectacular climb and importance to AI, with [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1794873807062339633) sharing a chart of its stock price. 
- **Metaverse Hype Cycle**: [@fchollet](https://twitter.com/fchollet/status/1794814338710290520) compared current AI hype to late 2021 when the Metaverse and NFTs were seen as the future of everything before dying down.
- **LLaMA Ecosystem**: Many shared resources, demos and experiments around LLaMA models, including a self-organizing file manager called LLamaFS ([@llama_index](https://twitter.com/llama_index/status/1794762651769430381)).

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Developments and Capabilities**

- **Llama-3-70B fine-tuned model**: In /r/LocalLLaMA, the Llama-3-70B fine-tuned model is the [**most uncensored model on the Uncensored General Intelligence leaderboard**](https://www.reddit.com/r/LocalLLaMA/comments/1d1gvor/llama370b_finetune_is_the_most_uncensored_model/), outperforming other large language models.
- **Phi-3 mini ONNX DirectML Python AMD GPU**: In /r/LocalLLaMA, a [demo shows token generation reusing KV cache from NPU](https://www.reddit.com/r/LocalLLaMA/comments/1d1j31r/faster_whisper_server_an_openai_compatible_server/), then running on CPU at 27 tokens/second.
- **In-context learning transformers**: In /r/MachineLearning, in-context learning transformers [can be used as **tabular data classifiers**](https://www.reddit.com/r/MachineLearning/comments/1d0wnjf/r_why_incontext_learning_transformers_are_tabular/), learning to create complex decision boundaries during pretraining.
- **MOMENT foundation model**: In /r/MachineLearning, the MOMENT foundation model was [released for **time series forecasting, classification, anomaly detection and imputation**](https://www.reddit.com/r/MachineLearning/comments/1d10lma/p_moment_a_foundation_model_for_time_series/).

**AI Agents and Assistants**

- **Microsoft's Recall**: In /r/MachineLearning, Microsoft's Recall can be [recreated using **open-source models and tools**](https://www.reddit.com/r/MachineLearning/comments/1d14pad/p_rerecall_i_tried_to_recreate_microsofts_recall/) like mss for screenshots, ollama/llava for descriptions, and chromadb for search.
- **Jetta Autonomous Configurable PC Task Doer**: In /r/singularity, Jetta is a [new AI agent that can **control your PC**](https://www.reddit.com/r/singularity/comments/1d1etav/microsofts_screen_assistant_feature_is_probably_a/).
- **Faster Whisper Server**: In /r/LocalLLaMA, Faster Whisper Server provides an [OpenAI compatible server for **streaming transcription**](https://www.reddit.com/r/LocalLLaMA/comments/1d1j31r/faster_whisper_server_an_openai_compatible_server/) using faster-whisper as the backend.
- **AI agent applications**: In /r/LocalLLaMA, people are excited about potential AI agent applications like [**games with AI characters**](https://www.reddit.com/r/LocalLLaMA/comments/1d11cb1/what_cool_apps_do_you_think_local_llms_will_enable/) once larger models can run on consumer hardware.

**AI Regulation and Governance**

- **Two former OpenAI board members**: In The Economist, two former OpenAI board members argue that [AI firms cannot self-govern and **need regulation**](https://www.economist.com/by-invitation/2024/05/26/two-former-openai-board-members-say-ai-firms-cant-be-left-to-govern-themselves) to tame market forces for humanity's sake.
- **AI "kill switch"**: In Fortune, tech companies have [agreed to an AI "kill switch"](https://fortune.com/2024/05/21/ai-regulation-guidelines-terminator-kill-switch-summit-bletchley-korea/) to prevent Terminator-style risks, though implementation details are unclear.

**AI and Society**

- **AI unemployment and UBI**: In /r/singularity, some argue it's [time to introduce UBI tied to **unemployment rates**](https://www.reddit.com/r/singularity/comments/1d1k3v0/it_might_already_be_time_for_ubi/) to support those displaced by AI.
- **Post-scarcity world**: In /r/singularity, in a post-scarcity world, [excess wealth could still buy **scarce experiences**](https://www.reddit.com/r/singularity/comments/1d10ngi/what_things_will_excess_wealth_still_be_useful/) like real estate, vacations, live performances, and human-provided services.
- **Age Reversal Unity**: In /r/singularity, Age Reversal Unity has [petitioned the FDA to **recognize aging as a disease**](https://www.reddit.com/r/singularity/comments/1d0vr4j/age_reversal_unity_mailed_the_citizen_petition_to/), a step towards anti-aging treatments.

**AI Art and Content Generation**

- **Stable Diffusion techniques**: In /r/StableDiffusion, new Stable Diffusion techniques allow for [**travelling through images in creative ways**](https://www.reddit.com/r/StableDiffusion/comments/1d0zfry/in_the_house_that_rubik_built_every_window_tells/) to create detailed, expansive artworks.
- **Anime style changer**: In /r/StableDiffusion, an anime style changer [combines SDXL model with ControlNet and IPAdapter](https://www.reddit.com/r/StableDiffusion/comments/1d11izd/you_never_go_wrong_with_adding_caspar_david/) to transform character designs.
- **AI-generated art carbon footprint**: In an image post, AI-generated art is argued to have a [much smaller **carbon footprint** than traditional art](https://i.redd.it/5zuv1vangq2d1.png), though this claim is disputed.
- **LoRA models**: In /r/StableDiffusion, new LoRA (Low-Rank Adaptation) models were released for [**anime-style droids**](https://www.reddit.com/r/StableDiffusion/comments/1d1hwvq/droiddiffusionxl_v1_lora/) and [**running motion**](https://www.reddit.com/r/StableDiffusion/comments/1d14gvh/whats_the_cloud_platform_to_run_stable_diffusion/).

**Memes and Humor**

- **Nvidia's increasing revenues**: A meme jokes that [Nvidia's increasing revenues are **beating expectations in 2024**](https://i.redd.it/emybpzlzdu2d1.jpeg) while AMD falls behind.
- **Google's approach to AI**: In /r/singularity, a meme pokes fun at [Google's approach to AI being **overly cautious and slow**](https://www.reddit.com/r/singularity/comments/1d12t1w/googles_approach_to_ai_is_a_meme_at_this_point/).
- **Limits in AI art generation**: A meme suggests the [only limits in AI art generation are "**your imagination... and VRAM**"](https://i.redd.it/t8h1thbd4u2d1.jpeg).

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **Fine-Tuning and Model Training Challenges**:
   - Discussions on various Discords highlighted challenges in **fine-tuning** models like **Llama 3** and **Mistral**, with users facing issues from **semantic similarity overfitting** to **runtime errors** on GPUs like T4. Useful guides and troubleshooting tips were shared, such as **[TinyLLama Fine-Tuning](https://lucasvw.github.io/posts/19_llm_fine_tuning/)** and **[Mistral-Finetune repository](https://github.com/mistralai/mistral-finetune)**.
   - Members struggled with **model tokenization and prompt engineering**, emphasizing the importance of correctly using template tokens like `###` or **end-of-text tokens** for efficient fine-tuning. This was particularly discussed in the context of **Axolotl and Jarvis Labs**.

2. **Advancements in Multimodal Models and Integration**:
   - **Perplexity AI** outshone **ChatGPT** in processing CSV files by supporting direct uploads and integrating tools like **Julius AI** for data analysis, as noted by users on Discord.
   - **New proteins visualization project** using 3D rendering was shared on **HuggingFace**, along with considerations for integrating **Vision Transformers (ViT)** for tasks like monocular depth estimation. Check out the [GitHub repository](https://github.com/AstraBert/proteinviz/blob/main/examples.md) for protein examples.

3. **Open-Source AI Projects and Community Efforts**:
   - **LlamaIndex** introduced tools for **automated RAG chatbots,** detailed in a [post on MultiOn's demo](https://twitter.com/llama_index/status/1793764970024570979). Issues around ensuring **context maintenance** and efficient indexing for knowledge retrieval were discussed.

4. **New Model Releases and Benchmarking**:
   - **Meta's Phi-3 Medium 128k Instruct** debuted, receiving attention for its enhanced reasoning and instruction-following abilities, available on [OpenRouter](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct). Discussions emphasized user feedback on model performance and applications.
   - **IBM Granite vs. Llama-3** performance debates surfaced on platforms like **ChatbotArena**, underlining the need for credible and transparent benchmarks. **DeepSeek-V2** and **Granite-8B-Code-Instruct** were notable mentions, with specific benchmarks shared.

5. **Ethics, Legislation, and AI's Societal Impact**:
   - Concerns over **SB-1047** were voiced, likening it to regulatory capture and disadvantaging smaller AI players. Tools like **Perplexity AI** for searching legislation impacts were shared for community awareness.
   - **OpenAI's water consumption** during AI model training stirred discussions on environmental impacts, referencing [Gizmodo's article](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249). The community called for more **eco-friendly AI practices** and discussed alternatives like **Meta's Audiocraft** for sustainable advancements.

---

{% if medium == 'web' %}



# PART 1: High level Discord summaries




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Fine-Tuning Facts**: Discussion on fine-tuning in the [general channel](https://discord.com/channels/1238365980128706560/1238365980128706563/1243282801760145408) revealed a concern about **semantic similarity overfitting** due to biased data categories. A user struggled with understanding fine-tuning vis-à-vis user inputs and initial model training. Changes in the **OpenAI platform's sidebars** were also noted with the disappearance of two icons (threads and messages).

**Templates Take the Spotlight**: In [workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1243336501018755123), the importance of configuring templates correctly during fine-tuning was highlighted. In particular, the delimiter `###` aids in parsing different input sections, and "end of text" tokens indicate when to stop token generation.

**Maven Mingles with Moderation**: In [asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1243344778511515698), a light-hearted exchange between members referenced a reunion. A request for a conference talk recording was met, with the video being available on Maven.

**Modal Mobilization**: Modal users in [🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1243309176722030702) shared excitement over received credits, training experiences, and provided specific links to **Modal documentation** and **examples** for new users. A plan to use Modal for a **Kaggle competition** was also shared, including setup and execution details.

**Jarvis Jots Down Jupyter Jumble**: In the [jarvis-labs channel](https://discord.com/channels/1238365980128706560/1241117895740625099/1243307629057671229), members discussed storing a VSCode repo on Jarvis with a suggestion to use GitHub for saving work. There was a notice of **spot instance removal** due to instability. The cost and duration of fine-tuning the **open-lama-3b** model were shared, and a user resolved an Ampere series error by adjusting model parameters.

**Hugging Face Huddles on Credits & Spanish Models**: The [hugging-face channel](https://discord.com/channels/1238365980128706560/1241141471814488115/1243335428887806004) saw discussions about pending **HF credits** and models suitable for Spanish text generation—with **Mistral 7B** and **Llama 3** models being recommended.

**Credit Countdown Carries On** in [replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1243453712182149150), where an upcoming announcement related to credit management and distribution was teased.

**Corbitt's Commandments Claim Clout**: Enthusiastic attendees in the [kylecorbitt_prompt_to_model channel](https://discord.com/channels/1238365980128706560/1242221891733946490/1243287896652517376) discussed fine-tuning methods and techniques presented in Kyle Corbitt's talk, including *[Ten Commandments for Deploying Fine-Tuned Models](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67)*.

**Axolotl Answers the Call** in [workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1243277523316637817), where users discussed datasets, model training, and troubleshooting in Axolotl. A blog post on **TinyLLama Fine-Tuning** was shared, and there was a push for integrating observability into LLM applications.

**Zoom Out, Discord In**: Users from [workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1243339106675724369) migrated their discussions to Discord after the Zoom chat was disabled.

**Axolotl's Cache Conundrum Causes Confusion**: Issues with cache in Axolotl frustrating users and confusion with missing files were resolved in [axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1243286083022618664). Discussions on sample packing and a guide on tokenizer gotchas addressed concerns around efficiency and tokenization.

**Accelerate to Victory**: [zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1243291846415749283) saw users work through confusion over float comparisons, resolve Jarvislab training command errors, and exchange resources for learning model acceleration with a focus on fine-tuning best practices.

**Winging It with Axolotl**: The [wing-axolotl channel](https://discord.com/channels/1238365980128706560/1242564077151326388/1243305377974587412) collaborated on dataset templates, pre-processing issues, Axolotl configurations, and provided a PR merge for the latest Axolotl updates. They delved into debugging tools and the significance of precise templates for training success.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Protein Data Visuals Reach New Heights**: A new protein visualization project now sports 3D rendering and includes examples for human hemoglobin and ribosomal proteins, with the project details found on [GitHub](https://github.com/AstraBert/proteinviz/blob/main/examples.md).

**Enter the TranscriptZone with OpenAI's Whisper**: A new transcription app that leverages OpenAI's Whisper to transcribe YouTube videos and more is available at [Hugging Face Spaces](https://huggingface.co/spaces/tensorkelechi/vidtext).

**Decentralizing the Web - More than a Dream?**: A project building infrastructure for a decentralized internet sought community feedback through a survey, raising discussions about the ethics of data collection.

**A Vision Transformers Query in Depth**: A member sought resources on applying Vision Transformers (ViT) for monocular depth estimation, indicating an intent to develop a model using ViT, but no specific resources were provided in the discussion.

**Quantisation Quandary for Mistral Model**: The use of **bitsandbytes** for 8-bit quantisation on **Mistral v0.3 Instruct** led to slower performance compared to 4-bit and fp16, a baffling outcome that contradicts expected efficiency gains from reduced-bit computation.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Climbs Over ChatGPT in CSV Showdown**: Engineers discussed that **Perplexity AI** outshines **ChatGPT** in CSV file processing by allowing direct CSV uploads. Also, **Julius AI** was recommended for data analysis, leveraging Python and integration with LLMs like **Claude 3** or **GPT-4**.

- **Users Snub Claude 3 Opus**: **Claude 3 Opus** is getting the cold shoulder due to increased content restrictions and perceived diminished utility, with **GPT-4** posed as a preferable option despite limitations.

- **Querying Pro Search's True Upgrade**: Upgrades to **Pro Search** raised eyebrows as users discussed whether new multi-step reasoning features and API specs were genuine backend improvements or merely surface-level UI enhancements.

- **API Integration Articulated**: Dialogue around API integration for external tools with **Claude** generated interest along with sharing of custom function calls, serverless backends, and documentation like [Tool Use with Claude](https://docs.anthropic.com/en/docs/tool-use).

- **Ethics in AI: More Than a Thought Experiment**: Discourse on infusing GPTs with ethical monitoring capabilities sparked, casting light on potential applications in workplace communication and legal defensibility, albeit with philosophical wrinkles yet to be ironed out.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Speculation Peaks on RTX 5090's VRAM**: There's buzzing debate over whether the rumored **RTX 5090 with 32GB VRAM** makes practical sense. References were made to potential specs and images on [PC Games Hardware](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/), but some members remained skeptical about its authenticity.

- **Stable Diffusion and the AMD Challenge**: Users offered guidance on installing **Stable Diffusion** on an AMD 5700XT GPU, suggesting that starting with web services like [Craiyon](https://www.craiyon.com/) may circumvent potential compatibility issues.

- **Stable Diffusion 3: Trial Before Commitment**: The community contrasted **Stable Diffusion 3** with competitor Midjourney, highlighting that while a free trial is available for SD3, ongoing access would require a **Stability** membership.

- **Anticipation Builds Around Mobius Model**: An announcement concerning DataPlusEngine’s novel **Mobius model** has garnered significant interest for its claim to create efficient base models. The model, teased on [Twitter](https://x.com/DataPlusEngine/status/1793803117642854732), is neither a straightforward base model nor a tuned version of something pre-existing.

- **32GB VRAM: Game Changer or Overkill?**: The mention of a 32GB VRAM GPU led to conversations about the potential shift in Nvidia's approach to data center GPU sales, considering how products with substantial memory could impact the market demand for the H100/A100 series.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **PEFT Config Snag Solved**: An issue where `config.json` was missing during PEFT training was resolved by copying it from the base model's configuration, with the user confirming success.

- **Llama Levitates Above Bugs**: The **Llama 3** model's base weights were described as "buggy," but Unsloth has implemented fixes. To improve training, the use of reserved tokens and updates to the tokenizer and `lm_head` are recommended.

- **System Prompt Boosts Llama 3**: Incorporating a system prompt, even a blank one, was observed to enhance Llama3 finetuning outcomes.

- **Phi 3 Proliferation**: Excitement bubbled as **Phi 3 models** debuted, sporting medium support. Community chatter pointed engineers toward extensive details in blog posts and release notes.

- **Stable Diffusion's Sinister Side Show**: Creepy artifacts and uncanny voice cloning outputs from **Stable Diffusion** startled users, with discussions and experiences shared via YouTube videos and a Reddit thread.

- **VSCode Copilot Climbing Onboard**: Recommendations for a local VSCode "copilot" were sought and met with suggestions and positive responses in the **random** channel.

- **Inference Inertia with Phi-3**: Slower inference times using **Unsloth Phi-3** puzzled one user, who provided a [Colab notebook](https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X) to investigate the lag, with community efforts yet to find a fix.

- **Quantization Quandary Unraveled**: A member faced challenges quantizing a custom model, hitting walls with **llama.cpp** and **Docker** compatibility, sparking a discussion on solutions.

- **VRAM Verdict for Model Might**: VRAM requirements were laid out: **12GB for Phi 3 mini** is okay, but **16GB is a must for Phi 3 medium**. For hefty tasks, considering outside computing resources was proposed.

- **Data Diligence for Training Consistency**: The importance of using consistent datasets for training and evaluation was echoed, highlighting **Unslothai's public datasets** like the [Blackhole Collection](https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a).

- **Platform Possibilities and Cautions**: Queries regarding **Unsloth** support for older Macs were addressed, confirming a focus on CUDA and GPU usage, with suggestions for those on CPU-only rigs.

- **Enterprise Expertise Extension**: A community member stepped forward to offer enterprise expertise to Unsloth, hailing the joining of accelerators at Build Club and Github, hinting at synergistic potential for Unsloth's endeavors.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Intellectual Debate Ignites Over AI Understanding**: In-depth discussions were had about the true **understanding** of concepts by LLMs, with **interpretability research** considered important empirical evidence. Skeptics argued that current efforts are lacking, with references to work by **Anthropic** on mapping large language model minds.

**The Creature from the Llama Lagoon**: A technical foray into enhancing **Llama models** centered around crafting a script that could manage **function calls**, with **Hermes Pro 2**'s approach serving as inspiration. Another inquiry circled the implementation of **Llama3 LoRA** techniques on a 3080 GPU.

**Reality Quest in Digital Dimensions**: Spearheading a conversation on **Nous and WorldSim**, members explored the possible applications of **NightCafe** and multi-dimensional AR spaces in mapping complex AI worlds. Dream-like explorations in **audio-visualizers** and whimsical **ASCII art** representations highlighted creative uses for AI-driven simulations.

**Sifting Through RAG Data**: Advocation for models to **integrate internal knowledge** with **Retrieval-Augmented Generation (RAG)** was a hot topic, with questions raised about how to handle contradictions and resolve conflicts. Emphasizing user evaluations was seen as essential, particularly for complex query cases.

**Precision over Pixie Dust in Fine-Tuning AI**: The community's discourse featured a celebration of the **Mobius model** for its prowess in **image generation**, with anticipation for an open-sourced version and elucidating publications. Additionally, Hugging Face was mentioned for their `PyTorchModelHubMixin` enabling easier model sharing, though limited by a **50GB size constraint** without sharding.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX vs. PyTorch/XLA: The TPU Showdown**: The performance comparison of **JAX** and **PyTorch/XLA** on TPUs spurred debate over benchmarking nuances such as **warmup times** and **blocking factors**. The dramatic decline in GPT-3 training costs from **$4.5M to an estimated $125K-$1M by 2024** was highlighted, considering **TFLOP rates** and **GPU-hour pricing** from various contributors, linking to a [Databricks Blog Post](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8).

- **Scaling and Teaching LLMs**: In the research forum, the **Chameleon** model was noted for its strong performance in multimodal tasks, while **Bitune** promised improvements in zero-shot performance for LLMs ([Bitune Paper](https://arxiv.org/pdf/2405.14862)). Discussions questioned the scalability of the **JEPA** model for AGI and critiqued **RoPE's** context length limitations, referencing a relevant [paper](https://arxiv.org/pdf/2405.14591).

- **Emergent Features Puzzle LLM Enthusiasts**: Tim Dettmers' research on advanced quantization methods maintaining performance in transformer inference was linked, including his concept of emergent outliers, and its integration with Hugging Face via the [bitsandbytes library](https://huggingface.co/blog/hf-bitsandbytes-integration). Discourse on emergent features coalescing around ideas of them being the "DNA" of a model, driving discussions on its implications for phase transitions.

- **A Brief on Technical Tweaks & LM Evaluation**: Within the **lm-thunderdome**, engineers covered practical tips for setting seeds in **vllm models**, retrieving the **list of tasks** with `lm_eval --tasks list`, and handling changes in **BigBench** task names that affect harnesses like Accelerate with memory issues. It was suggested to locate tasks by perusing the `lm-eval/tasks` folder for better organization.

- **A Call for Collaboration**: An appeal was made for expanding the **Open Empathic** project, with a [YouTube guide](https://youtu.be/GZqYr8_Q7DE) for contributing movie scenes and a link to the project shared. Further collaboration was encouraged, underlining the need for community efforts in enhancement.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**GPU Adventures**: Engineers discussed challenges when loading small models onto GPUs, with some favoring models like *llama3, mistral instruct,* and *cmdrib*. Meanwhile, using lower quantizations, such as *llamas q4*, reportedly yielded better results than higher ones like q8 for certain applications, refuting the notion that "bigger is always better."

**Next-Gen Models Incoming**: An update in the model realm informed about the release of a **35B model**, with testing to ensure LM Studio compatibility. Optimizations for different scales of models were a topic too, with a focus on **Phi-3 small GGUFs** and their efficiency.

**Servers and Setups**: Hardware discussions included leveraging **distributed inference** with **llama.cpp** and its recent RPC update, although quantized models aren't supported yet. Experimental builds using clustered cheap PCs with **RTX 4060 Ti 16GB** for distributed model setups and possible network constraints were also explored.

**Multilingual Cohesion Achieved**: Cohere models now extend their prowess to **23 languages**, as advertised with **aya-23 quants** available for download, but ROCm users must await an update to dive in. 

**Stable Diffusion Left Out**: LM Studio clarified that it exclusively handles language models, excluding image generators like Stable Diffusion, alongside dealing with CUDA issues on older GPUs and promoting services like **Julius AI** to ease user experience woes.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Gradient Norm Nuisance**: Altering the batch size from 32 leads to a sudden spike in gradient norm, disrupting training. A [pull request](https://github.com/karpathy/llm.c/pull/456) resolved this issue by preventing indexing overflow in the fused classifier.
  
- **Int4 and Uint4 Types Need Some TLC**: A member flagged that many functions lack implementations for **int4** and **uint4** data types in PyTorch, with a [discussion thread](https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833) indicating limitations on type promotion and tensor operations.

- **Live Code Alert – Scan Algorithm in Spotlight**: Izzat El Hajj will lead a live coding session on the Scan algorithm, vital for ML algorithms like Mamba, scheduled for `<t:1716663600:F>`, promising to be a technical deep dive for enthusiasts.

- **CUB Library Queries and CUDA Nuances**: Members tapped into discussions ranging from the functioning of CUDA CUB library code to triggering tensor cores without cuBLAS or cuDNN, highlighting resources like [NVIDIA's CUTLASS GitHub repository](https://github.com/NVIDIA/cutlass/tree/main) and the [NVIDIA PTX manual](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

- **FineWeb Dataset Conundrum**: Processing the FineWeb dataset can be a storage hog, hitting 70 GB on disk and gobbling up to 64 GB of RAM, hinting at a need for better optimization or more robust hardware configurations for data processing tasks.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Python Libraries Cling to C Over Mojo**: There's a lively conversation about the feasibility and preparedness of porting Python libraries to Mojo, with concerns about pushing maintainers too hard given Mojo's evolving API. Members discussed whether targeting C libraries might be a more immediate and practical endeavor.

**Rust's Security Appeal Doesn't Rust Mojo's Potential**: Mojo is not slated to replace C, but the security benefits of Rust are influencing how engineers think about Mojo's application in different scenarios. Ongoing discussions address concepts from Rust that could benefit Mojo developments.

**Blazing Ahead With Nightly Mojo**: BlazeSeq performance on MacOS using Night versions of Mojo shows promising similarity to Rust's Needletail, fueling cross-platform efficiency discussions. Rapid nightly updates, noted in [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md), keep the community engaged with the evolving language.

**Curiosity Sparks Over Modular Bot's Machinery**: Queries were raised about the underlying tech of "ModularBot", and although no specific model was referenced, the bot shared a colorful reply. Separately, the potential for ML model training and inference within Mojo was discussed, with mention of Max Engine as a numpy alternative, though no full-fledged training framework is on the horizon.

**Compile-Time Confusion and Alignment Woes**: Problems from aligning boolean values in memory to compile-time function issues are causing a stir among users, with workarounds and official [bug reports](https://github.com/modularml/mojo/issues/2813) highlighting the importance of community-driven troubleshooting.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **LaTeX Loyalist LLM**: In the realm of formatting, users noted frustration with GPT's strong inclination to default to LaTeX despite requests for Typst code, revealing preferences in coding syntax that the LLM seems to adhere to.
  
- **Microsoft Copilot+ vs. Leonardo Rivalry**: Conversations in the community centered on the value of Microsoft Copilot+ PCs for creative tasks like "sketch to image," while some members encouraged checking out [Leonardo.ai](https://leonardo.ai) for analogous capabilities.

- **A Thirst for Efficiency in AI**: Concern was voiced over the environmental toll of AI, citing a [Gizmodo article](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249) on the substantial water usage during the training of AI models, prompting discussions on the need for more eco-friendly AI practices.

- **Iteration Over Innovation**: There was active dialogue on enhancing the performance of LLMs through iterative refinement, with references to projects like AutoGPT addressing iterations, despite the associated higher costs.

- **Intelligence Infusion Offer Overstated?**: The guild pondered the plausibility and potential of embedding legal knowledge within ChatGPT, enough to consider a valuation at $650 million, though detailed perspectives on this bold assertion were limited.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain CSV Agent Deep Dive**: Engineers explored **LangChain's CSV agent** within a **SequentialChain** and discussed [how to customize output keys](https://python.langchain.com/docs/modules/chains/foundational/sequential_chains) like `csv_response`. Challenges with SQL agents handling multi-table queries were mentioned, pointing towards token limits and LLM compatibility issues, with direction to GitHub [for issues](https://github.com/langchain-ai/langchain/issues).

**AI Showcases Gather Buzz**: [OranAITech tweeted](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19) their latest AI tech, while **everything-ai v2.0.0** announced features including audio and video processing capabilities with a [repository](https://github.com/AstraBert/everything-ai) and [documentation](https://astrabert.github.io/everything-ai/) available.

**Demystifying VisualAgents**: Demonstrations of **Visual Agents platform** were shared via YouTube, revealing its potential to streamline SQL agent creation and building simple retrieval systems without coding, utilizing LangChain's capabilities. Two specific videos showcased their workflows: [SQL Agent](https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N) and [Simple Retrieval](https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM).

**EDA GPT Impressions On Display**: A demonstration of **EDA GPT**, including a five-minute overview video showcasing its various functions, was linked to via [LOVO AI](https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01). The demo highlights the AI tool's versatility.

**Tutorial Teaser**: A message in the tutorials channel provided a [YouTube link](https://youtu.be/gflsu_6R_8g) to business24.ai's content, although the context of its relevance was not disclosed.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Piracy's Not the Panacea**: Despite a humorous suggestion that The Pirate Bay could become a haven for sharing AI model weights, skepticism among members arises, highlighting the potential for friendlier AI policy landscapes in other nations to prevail instead.

- **Japan Takes the AI High Road**: Participants noted Japan's encouraging position on AI development, referencing a **paper** shared via a [tweet](https://x.com/DataPlusEngine/status/1793817514956259460) about creating new base diffusion models without the need for extensive pretraining, showcasing a strategy involving temporary disruption of model associations. 

- **Poisoned Recovery Protocols Probed**: A **collaborative study**, involving a poisoned model recovery method conducted by fal.ai, was mentioned, with findings expected to empirically substantiate the recovery approach. Reservations were expressed regarding the aesthetics of AI-generated imagery, specifically the "high contrast look" and artifacts presented by models like Mobius versus predecessors such as MJv6.

- **Claude Mappings Crack the Code**: Anthropic's **research paper** details the dissection of Claude 3 Sonnet's neural landscape, which illustrates the manipulation of conceptual activations and can be read at their [research page](https://www.anthropic.com/research/mapping-mind-language-model). Debates sparked over the potential commercialization of such activations, with a juxtaposed fear of the commercial implications driving AI practitioners to frustration.

- **A Nostalgic Look at AI's Visual Visions**: A member reminisced about the evolution from early AI visual models like Inception v1 to today's sophisticated systems, recognizing DeepDream’s role in understanding neural functionality. Furthermore, the benefits of sparsity in neural networks were discussed, describing the use of L1 norm for sparsity and a typical 300 non-zero dimensions in high-dimensional layers.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Meetup Alert: Limited Seats Available**: Few spots remain for the upcoming **LlamaIndex meetup** scheduled for Tuesday, with enthusiasts encouraged to [claim their spots](https://twitter.com/llama_index/status/1793739449127583964) quickly due to limited availability.

- **MultiOn Meets LlamaIndex for Task Automation**: **LlamaIndex** has been coupled with **MultiOn**, an AI agents platform, facilitating task automation through a Chrome web browser acting on behalf of users; view the demo [here](https://twitter.com/llama_index/status/1793764970024570979).

- **RAGApp Launches for Code-Free RAG Chatbot Setup**: The newly introduced **RAGApp** simplifies the deployment of RAG chatbots via a docker container, making it easily deployable on any cloud infrastructure, and it's open-source; configure your model provider [here](https://twitter.com/llama_index/status/1794030544415818062).

- **Solving PDF Parsing Puzzles**: The community endorses **LlamaParse** as a viable API for extracting data from PDFs, especially from tables and fields, leveraging the GPT-4o model for enhanced performance; challenges with **Knowledge Graph Indexing** were also a topic, highlighting the need for both manual and automated (through `VectorStoreIndex`) strategies.

- **PostgresML Joins Forces with LlamaIndex**: **Andy Singal** shared insights on integrating **PostgresML** with **LlamaIndex**, detailing the collaboration in a Medium article, ["Unleashing the Power of PostgresML with LlamaIndex Integration"](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939), receiving positive remarks from the community.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Phi-3 Medium 128k Instruct Drops**: OpenRouter unveiled **Phi-3 Medium 128k Instruct**, a powerful 14-billion parameter model, and invited users to review both the [standard](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct) and [free](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free) variants, and to participate in discussions on its effectiveness.

- **Wizard Model Gets a Magic Boost**: The **Wizard model** has shown improvements, exhibiting more prompt and imaginative responses, yet attention is required to avoid repeated paragraphs.

- **Eyes on Phi-3 Vision and CogVLM2**: Enthusiasm surges around **Phi-3 Vision**, with sharing of testing links like [Phi-3 Vision](https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml), and suggestions to use **CogVLM2** for vision-centric tasks found at [CogVLM-CogAgent](https://huggingface.co/spaces/THUDM/CogVLM-CogAgent).

- **Automatic Llama 3 Prompt Transformation**: It was clarified that prompts to **Llama 3** models are automatically transformed through OpenRouter's API, streamlining the process, but manual prompting remains as an alternative approach.

- **Gemini API Annoyances**: Users reported issues with **Gemini FLASH** API, such as empty outputs and token drain, recognized as a model-centric problem. The emergence of Google's daily API usage limits has piqued interest in how this might affect OpenRouter's Gemini integration.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord


- **LLM Evaluation under the Microscope**: A [Hugging Face blog post](https://huggingface.co/blog/clefourrier/llm-evaluation) about Large Language Model (LLM) evaluation practices, the importance of leaderboards, and meticulous non-regression testing caught the attention of members, emphasizing the critical role of such evaluations in AI developments.

- **AI's Answer to Search Engine Manipulations**: An incident involving website poisoning affecting Google's AI-gathered overviews triggered discussions around security and data integrity, including workarounds through custom search engine browser bypasses as reported in a [tweet by Mark Riedl](https://x.com/mark_riedl/status/1793375699967054334).

- **AI Democratizing Development or Raising Reliability Questions?**: GitHub CEO Thomas Dohmke's [TED Talk](https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6) on AI's role in simplifying coding provoked debates over its reliability despite AI-driven UX improvements that expedite problem-solving in the coding process.

- **Diversity Scholarships to Bridge Gaps**: Engineers from diverse backgrounds who face financial barriers to attending the upcoming AI Engineer World's Fair received a boost with the announcement of diversity scholarships. Interested applicants should furnish *concise* responses to the essay questions provided in the [application form](https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Tax Tales Without Plastic**: Nathan Lambert deciphered an invoice kerfuffle, realizing the rational behind tax billing sans credit card due to resale certificates.

- **Golden Gate AI Gets Attention**: Experimentation by [Anthropic AI](https://x.com/anthropicai/status/1793741051867615494?s=46) led to "Golden Gate Claude," an AI single-mindedly trained on the Golden Gate Bridge, creating buzz for its public interactivity at claude.ai.

- **Google's AI Missteps**: Google's failure to harness feedback and premature deployment of AI models spurred discussion about the tech giant's public relations challenges and product development woes.

- **Battling Dataset Misconceptions**: Google's AI team countered claims about using the LAION-5B dataset by putting forth that they utilize superior in-house datasets, as referenced in a [recent tweet](https://x.com/giffmana/status/1793906145310228538).

- **Nathan Shares Knowledge Nuggets**: For AI aficionados, Nathan Lambert uploaded advanced [CS224N lecture slides](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit). Additionally, attendees were tipped off about an upcoming session recording, sans release date details.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GQA Gains Traction in CMDR Models**: Discussions revealed that **Grouped Query Attention (GQA)** is present in the "cmdr+" models but not in the basic "cmdr" models, indicating an important distinction in their specifications.
- **VRAM Efficiency with Smart Attention**: Engineers noted that while **GQA** doesn't offer linear scaling, it represents an improved scaling method compared to exponential, affecting **VRAM** usage favorably.
- **Sample Packing Gets a Boost**: A new **GitHub pull request** showcases a 3-4% efficiency improvement in sample packing, promising better resource management for distributed contexts, linked [here](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619).
- **Academic Achievement Acknowledged**: A member's co-authored journal article has been published in the **Journal of the American Medical Informatics Association**, highlighting the impact of high-quality, mixed-domain data on medical language models, with the article available [here](https://doi.org/10.1093/jamia/ocae120).
- **Community Cheers Scholarly Success**: The community showed support for the peer's published work through personal congratulatory messages, fostering a culture of recognition for academic contributions within the AI field.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**SB-1047 Sparks Technical Turmoil**: Engineers express deep concerns about the implications of **SB-1047**, dubbing it as detrimental to smaller AI players and likening the situation to regulatory capture observed in other industries.

**Perplexity and Arc, Tools of the Trade Showcased**: The community spotlighted tools aiding their workflows, sharing a [Perplexity AI search on SB-1047](https://www.perplexity.ai/search/SB-1047-Senate-2kZmFYHoTxe.rWUYat4B2A) and the new “Call Arc” feature of Arc Browser, which simplifies finding relevant answers online, with an informational [link](https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD).

**Install Issues Incite Inquiry**: Users face issues with **Typer** library installation via pip, raising questions about whether steps in the setup process, such as `poetry install` before `poetry run`, were followed or if a virtual environment is being used.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Twinny Takes Off as Virtual Co-Pilot**: Developers are integrating [Twinny](https://github.com/rjmacarthy/twinny) with LM Studio to serve as a robust local AI code completion tool, with support for multiple llamafiles running on different ports.

**Embedding Endpoint Enlightenment**: The `/v1/embeddings` endpoint was clarified not to support `image_data`; instead, the `/embedding` endpoint should be used for images, as per [pull request #4681](https://github.com/ggerganov/llama.cpp/pull/4681).

**Mac M2 Meets Its Match in continue.dev**: A performance observation noted that continue.dev runs slower on a Mac M2 compared to an older Nvidia GPU when executed with llamafile.

**Hugging Your Own LLMs**: For those looking to build and train custom LLMs, the community recommended the use of [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) for training, with the reminder that llamafile is designed for inference, not training.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Gratitude Echoes in the Server**: A user expressed heartfelt *thanks* to the team, showcasing user appreciation for support or development work done by the team.
- **Curiosity About Upscaled Models**: There's buzz around whether a **104B version** of a model will join the family tree, but no clear answers have been outlined yet.
- **Langchain Links Missing**: Questions arose regarding the integration of **Langchain** with Cohere, with users seeking guidance on its current usability and implementation status.
- **Model Size Mysteries**: Users are probing for clarity on whether the **Aya model** in the playground pertains to the 8B or 35B version, indicating importance in understanding model scales for application.
- **Error Troubleshooting Corner**: Issues like a `ValidationError` with **ContextualCompressionRetriever** and a **403 Forbidden error** signal active debugging and technical problem-solving among the engineers, serving as reminders of common challenges in AI development.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**AI Comedy Night Hits the Right Notes**: An AI-generated standup comedy piece shared by a user was met with positive surprise, indicating advancements in AI's capability to mimic humor and perform entertainment.

**Exploratory Queries on AI Applications**: Curiosity about the extent of [Ud.io](https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG)'s functions was evident from a user's query whether its capabilities go beyond generating comedy.

**Sound Transformations Showcased**: A user displayed the flexible audio alteration features of [Suno](https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655) by sharing an altered, demonic version of an original sound piece.

**Eagerness for Audio Engineering Know-How**: Interest was expressed in acquiring the skills to craft audio modifications like the ones demonstrated, a skill set valuable for an AI engineer with an interest in sound manipulation.

**Concise Communication Preferred**: A one-word reply "No" to a question highlighted a preference for succinct responses, perhaps reflecting an engineer's desire for direct, no-nonsense communication.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **In Search of a Unified Event Tracker**: A member has highlighted a pressing need for an event calendar compatible with Google Calendar to ensure no community events are overlooked. The absence of such a system is a noted concern within the community.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **New Dataset Announcement**: A new dataset has been referenced by user datarevised, with a link to further details: [DataPlusEngine Tweet](https://x.com/DataPlusEngine/status/1793803117642854732).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Benchmarks Drive Innovation**: Evaluation benchmarks like [GLUE](https://arxiv.org/abs/1804.07461), [MMLU](https://arxiv.org/abs/2009.03300), and [GSM8K](https://arxiv.org/abs/2110.14168) are crucial for AI research progress, while "execution based evals" face particular challenges in dynamic tasks, as discussed in a [blog post](https://www.jasonwei.net/blog/evals) shared by members.
  
- **Anticipation for GPT-5**: A talk stirred speculation among guild members that **GPT-5** may debut in 2024 with an advancement towards an "agent architecture," as suggested in a [rumored tweet](https://x.com/rohanpaul_ai/status/1793956355897724973?s=46&t=90xQ8sGy63D2OtiaoGJuww).

- **AI's Role in Music Examined**: Members probed into copyright questions concerning AI-created music like **Suno**, contemplated the capabilities of Meta's **Audiocraft**, and discussed legal ramifications and open-source endeavors promoting creative freedom, including **gary-backend-combined** on [GitHub](https://github.com/betweentwomidnights/gary-backend-combined).

- **Pythonistas Gear Up for NumPy 2.0**: There's building excitement for **NumPy 2.0**, with members jesting about potential dependency management impacts, as noted in a [Twitter post](https://x.com/cgarciae88/status/1794019900119236874).

- **xAI Lands a Whopper of Investment**: Breaking into investment news, **xAI** secured a substantial $6 billion Series B funding round, setting sights on rapidly advancing their models' capabilities, as illustrated in their [announcement post](https://x.ai/blog/series-b).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Nightly Nuisances Fixed by Community Brainstorm**: After members highlighted that the **Mojo VSCode extension** worked only for post-installation files and was buggy since version 24.3, solutions included closing/reopening the code and frequent resets. Conversely, the **Mojo compiler's** nightly updates (`2024.5.2505` to `2024.5.2705`) featured enhancements like variable renaming in LSP and a new `tempfile` module, though they introduced test issues in existing PRs like [#2832](https://github.com/modularml/mojo/pull/2832) on GitHub.

- **The Zen of ZIPping Through Mojo**: Difficulties in replicating Python's `zip` and `unzip` functions in Mojo led to a discussion on how to declare functions returning tuples based on variadic list arguments. The conversation shed light on Mojo's potential auto-dereferencing iterators using the new `ref` syntax, aiming to simplify implementation and reduce explicit dereferencing.

- **Processor Picks Perform Differently**: A member's efforts to optimize CRC32 calculations in Mojo led to performance variance across core types; compact implementation lagged on larger byte sizes due to L1 cache limits, yet efficiency cores favoured the compact version. Benchmarking metadata and versioning files are located at [fnands.com](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc.mojo) and [nightly version tests for the aforementioned](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo).

- **Musings Over Mojo’s Might**: General discussions on **Mojo** ranged from function behavior differences compared to Python, handling of optional types, to implementing LinkedLists, and reflecting on Mojo's as-yet-to-be-implemented reflection capabilities. Members exchanged thoughts on efficient initializations using `UnsafePointer` and `Optional`.

- **Tech Titans' Tremendous Transactions**: The AI community digested news that Elon Musk’s AI startup, xAI, hauled in a hefty $6 billion in Series B funding as outlined in a [TechCrunch article](https://techcrunch.com/2024/05/26/elon-musks-xai-raises-6b-from-valor-a16z-and-sequoia/), positioning xAI to directly confront AI frontrunners like OpenAI and Microsoft, stirring conversations around the commercialization of AI technologies.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**New Kids on The Block: Phi-3 Models Arrive**: Microsoft's [phi-3-medium-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct) and [phi-3-mini-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-mini-128k-instruct) models are now live, with a special 57% discount applied to the [llama-3-lumimaid-70b](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b) model.

**Rate Limit Labyrinth Explained**: Challenges with **rate limiting** on OpenRouter sparked intense discussion, emphasizing the importance of understanding how credit balances impact request rates, as outlined in the [OpenRouter documentation](https://openrouter.ai/docs#rate-limits-and-credits-remaining).

**Modal Mayhem: When Credits Clash with Rate Limits**: A puzzling issue arose with the modal fallback feature, where rate limits were hit despite a healthy credit balance. The community recommended monitoring free requests and possibly sidelining the free model when limits loom.

**AI's Self-Moderation Struggle Softens Appeal**: Enthusiasts expressed concerns that stricter guardrails and higher refusal rates in Claude's self-moderated models result in a less human-like experience, pointing to a possible downturn in usage.

**Vision Model Breakdown: Performance vs. Price**: The talk turned to vision model performance, specifically Gemini's OCR capabilities, with a nod to its cost-effectiveness compared to traditional vision services. Conversations also highlighted cheaper GPU usage via RunPod and Vast.ai over mainstream clouds like Google Cloud and Amazon Bedrock.




---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **CCS Doesn't Cut the Mustard**: The Eleuther team's follow-up to Collin Burns's _Contrast Consistent Search (CCS)_ method didn't yield expected results in generalization improvements, as shown in the [Quirky Models benchmark](https://arxiv.org/abs/2312.01037). Their transparency in sharing both the method and lackluster results was commended in a detailed [blog post](https://blog.eleuther.ai/vincs/).

- **Optimal Model Extraction Strategy Hotly Debated**: Engineers sparred over whether **RAG** is superior to finetuning for **LLM**s in extracting data from custom libraries, concluding that RAG might retain information better. There's buzz around **ThePitbull 21.4B Model**, released on Hugging Face, with some skepticism regarding its near-70B-model performance claims.

- **Troubleshooting Data Replication**: AI programmers grappled with tokenizer tribulations while replicating Pythia data, with solutions including the use of `batch_viewer.py` and proper datatype handling with `MMapIndexedDataset`. The process, though likened to "black magic", was necessary for correct dataset interpretation, as noted in the [Pythia repository](https://github.com/EleutherAI/pythia#exploring-the-dataset).

- **Pushing the Envelope with New Techniques**: Accelerated discussions among engineers covered gradient perturbation methods to distribute weight updates and the potential of transformers to implicitly reason over parametric knowledge. A new "schedule-free" optimization paper caught eyes, suggesting iterate averaging without extra hyper-parameters might outperform traditional learning rate schedules.

- **Quantization Quandaries**: In the quest for efficiency, a discourse ensued on the situations where small models might surpass larger quantized ones. A reference to [ggerganov's Twitter](https://x.com/ggerganov/status/1666087050725199872) hinted at quantization methods' potential, stoking the fires of debate regarding the balance of model size and performance.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG Revolution: Enterprise Pipelines on the Horizon**: The **LlamaIndex team** announced a workshop in collaboration with AWS to showcase how to build **enterprise-level RAG pipelines** using AWS Bedrock and Ragas. Registrations are live for the event aiming to provide insights on integrating Bedrock with LlamaIndex for optimized RAG systems [Sign Up Here](https://lu.ma/x8unmku0).

- **Innovations Boosting Retrieval**: Discussions have spotlighted **Vespa's integration** for improved hybrid search capabilities and an advanced guide by Jayita B. on creating **rapid response RAG chatbots** using Llama3 and GroqInc. An innovative method for indexing images through structured annotations produced by models like **gpt4o** was also noted.

- **File Organization Ascended**: **LlamaFS** was launched as a new tool for automatically organizing cluttered directories, which may resonate with those seeking clean and efficient file management solutions.

- **Technical Troubleshooting Takes Center Stage**: AI Engineers grapple with issues surrounding **Llama Index's reAct**, with workarounds involving `max_iterations` settings and overcoming import errors by aligning package versions. HTML parsing generally requires more custom code compared to PDF files, which can take advantage of advanced chunking tools with fewer dependencies.

- **Pydantic for Structured LlamaIndex Output**: Guidance on using **Pydantic models** with LlamaIndex signifies a step towards structured output integration, pointing to broader applications and usability of the system. Calls for improved retriever documentation highlight community drive for enhanced understanding and application of **BM25** and **AutoRetrieval** modules within diverse projects.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Aya-23 Takes the Stage**: Engineers discussed **Aya-23's multilingual capabilities** compared to **Command R/R+**, implying superior performance but questioning its English-specific efficiency. They also noted **Aya-23-35b** is a fine-tuned version of **Command R** and provided access to the [technical report](https://drive.google.com/file/d/1YKBPo61pnl97C1c_1C2ZVOnPhqf7MLSc/view) for more details.

**Mobile Privacy Vs. LLM Limitations**: There was a consensus that **on-phone LLMs** aren't sufficiently developed for private, local execution in a mobile app, particularly for tasks typically aligning with a **RAG mobile app**.

**Bot Innovations Flourish**: A community member showcased a gaming bot on LinkedIn which garnered interest due to its integration with **Cohere Command R**; meanwhile, the "Create 'n' Play" bot for Discord boasts ***"over 100 engaging text-based games"*** and *enhances social engagement with AI*.

**Adaptation and Integration of Prompts**: The guild confirmed that **Aya-23 supports system prompts**, sharing insights on adapting **Command R** prompts with specific tokens such as `<|USER_TOKEN|>` and `<|CHATBOT_TOKEN|>` to operate effectively.

**Solutions for OneDrive Syncing**: In response to a query about **OneDrive connectors**, a [SharePoint connector](https://github.com/cohere-ai/quick-start-connectors/tree/main/sharepoint) was recommended, which may fulfill similar integration needs.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI's Advice Bridge Ditching**: Members shared a humorous take on Google AI's dangerous advice to "jump off bridges to cure depression", referencing the misleading nature of Reddit suggestions. A related meme was shared regarding the mishap.

**ConvNeXt Gets Optimized**: A vibrant discussion on the [ConvNeXt paper](https://arxiv.org/abs/2405.15738) praised its ability to handle high-resolution images efficiently, potentially reducing the generation of excessive visual tokens and streamlining optimizations for high-resolution tasks.

**From Redstone to Neural Nets**: Innovative uses of datasets and AI tools were showcased, including a dataset of publication PDFs and source TeX from [archive.org](https://archive.org/details/arxiv-bulk?sort=-publicdate), and a [YouTube video](https://youtu.be/DQ0lCm0J3PM?si=5Is7OMnqRhZb-ZAo) demonstrating how to create a neural network with Redstone.

**Growth Stacks Up in AI Pre-training**: An [arXiv paper](https://arxiv.org/abs/2405.15319) highlighting depthwise stacking as an effective method for model growth in efficient pre-training of Large Language Models (LLMs) sparked interest, addressing critical speed and performance challenges in the pre-training process.

**Pitfalls in PyTorch Persistence**: Discussions in the learning sphere centered on troubleshooting issues with the randomness in training-validation splits and loss inconsistency during model reloads. Specifically, proper saving of optimizer states in PyTorch was pinpointed as crucial to avoid exploding losses.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Llama3 Goes Deutsch**: The new **Llama3-German-8B** model extends the capabilities of Meta's Llama3-8B to the German language, training on 65 billion tokens with negligible performance loss in English; details are on [Hugging Face](https://huggingface.co/DiscoResearch/Llama3-German-8B). However, it's noted that unlike other language models, the training **omitted English data replay**, sparking debates about its effectiveness.

- **Quantization Quirks and Puzzles**: A quantized version of the Llama3, **GGUF** has shown underwhelming benchmark scores, hinting at potential issues, available for review at [Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF](https://huggingface.co/cstr/Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF). Meanwhile, discussions on parameter settings and strange outputs hint at the complex challenges of running these models, like max tokens and choice of the engine.

- **Cohere’s Multilingual Leap**: **Cohere's Aya-23-35B model**, though restrictive in licensing, now supports 23 languages, indicative of the growing trend and interest in powerful multilingual models. A related **ShareGPT format dataset** for translations has generated talks on quality and filtering, hosted [here](https://huggingface.co/datasets/sroecker/aya_german-sharegpt).

- **Mistral’s Guide to the Fine-tuned Galaxy**: In a nod to the tech-savvy community, Mistral rolls out a finetuning guide for Mixtral models, a beacon for those embarking on finetuning adventures; the guide can be perused on their [GitHub](https://github.com/mistralai/mistral-finetune).

- **Model Tuning Intricacies Exposed**: The community is conducting experiments with ***oobabooga*** and ***ollama***, touching on **'skip_special_tokens'** toggles and stop token settings, including a suggested use of a specific [Llama3 template](https://github.com/CrispStrobe/EQ-Bench/blob/main_v2_3a/instruction-templates/Llama3.yaml) to address output issues—reflecting the active tweaking and tuning culture prevailing amongst members.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **LAM Security Concerns and Mobility**: Discussions reveal skepticism about the **Large Action Model (LAM)** due to its potential for reverse engineering, despite the [Rabbit LAM architecture overview](https://engineering.rabbit.tech/lam-security-architecture-overview) showcasing its secure web app control capabilities. Conversations also circled on integration challenges for running the 01 model on **Rabbit R1** devices and **Humane platform**, highlighting user-drive solutions like the [01 model Android GitHub repository](https://github.com/Tonylib/o1_for_flutter).

- **Installation Tactics on the Fly**: A Python version upgrade to **3.11 resolved OpenInterpreter issues** on Mac, while tinkering with a Linux distro through [Andronix](https://andronix.app/) enabled OpenInterpreter use on Android-powered earbuds. Queries about **Open Interpreter** installation revealed a need for a local or cloud-accessible LLM, and a new [Markdown export feature](https://github.com/OpenInterpreter/open-interpreter/pull/1282) was launched to aid developers.

- **DIY Spirit in the Community**: A user fixed an OpenInterpreter issue on Mac by upgrading to Python 3.11, and others shared pathways for enhancing current devices, like modifying **R1 with LineageOS**. In buying versus building hardware, the consensus is that purchasing the pre-built O1 benefits the Seattle development team, despite no technical differences from a self-built version.

- **Shipping and Storage Under Scrutinization**: Frustration was voiced over **lack of updates on hardware shipment**, particularly regarding European distribution and information blackout for pre-orders. The conversation also featured a search for solutions to **overcome disk space limits** on Runpod, a hint at the constant struggle for cost-efficient data storage in AI workloads. 

- **OpenInterpreter Leap Forward**: A member's successful tactic of switching to **Python 3.11** on Mac for OpenInterpreter signals the ongoing agility in problem-solving within the community. Meanwhile, the implementation of a [new Markdown export feature](https://github.com/OpenInterpreter/open-interpreter/pull/1282) reflects the push for enhancing developer utility in AI toolchains.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**PDF Extraction Proves Challenging**: Discussions on extracting text from PDFs highlight the difficulties encountered with complex tables and diagrams, suggesting solutions like ML-based text segmentation and using Adobe Extract API for layout parsing, as referenced in the [LangChain documentation](https://python.langchain.com/v0.2/docs/how_to/document_loader_pdf/).

**LangChain Community Set to Expand**: Karan Singh from Scogo Networks expressed interest in creating a local LangChain community in Mumbai, seeking marketing contacts to organize events.

**Bump in the Langserve Waitlist**: Users experienced access issues with the Langserve waiting list on Airtable, searching for alternate methods to try the hosted service.

**Interactive Data Visualization Tool Introduced**: The NLAVIDA project, which facilitates interactive data visualization and analysis through natural language, was introduced along with a [YouTube video tutorial](https://www.youtube.com/watch?v=leJRP_mJsSQ&t=4s).

**Ready, Set, Vote for OranClick**: The launch of OranClick, a tool aimed at optimizing message crafting for higher signup rates, was announced with an invitation to support on [ProductHunt](https://producthunt.com/posts/oranclick).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile Powers Up with v0.8.5**: The release of [llamafile version 0.8.5](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.5) was highlighted, offering **fast inference** for K quants on X86 CPUs, with a call to action for community benchmarking efforts using `llamafile-bench`.
- **Llamafile Goes Network-Savvy**: Engineers swapped tips on how to make the llamafile server network-accessible, suggesting using flags like `--host <my ip>` or `--host 0.0.0.0` for cross-machine availability within the same network.
- **Blank Slate Mystery in Llama3-70B**: Contributors reported encountering **blank responses** from the llama3-70b model, sharing logs for community-led debugging, although definitive solutions were yet to surface.
- **Home Sweet APIs for Home Assistant**: There's a buzz around enhancing **Home Assistant** integration, with a focus on developing a standardized local API akin to OpenAI's candidate, underscored by the importance of features like API discoverability and secure API endpoints.
- **Python Puzzles for Model Select**: Questions and shared code snippets indicated some confusion around specifying models in the Python example for LLaMA_CPP integration, particularly concerning when model specification is a must, such as with TinyLlama.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral-Finetune Mystery Uncovered**: Developers engaged in deciphering the new update in the [Mistral-Finetune repository](https://github.com/mistralai/mistral-finetune), with emphasis on understanding the distinctive changes.
- **MoEs Fine-Tuning Quirkiness**: The AI crowd discussed the fickle nature of fine-tuning **Mixture of Experts (MoEs)** models, highlighting the necessity of running multiple iterations to cherry-pick the most efficient model, albeit details on the success rates were not divulged.
- **Aya 23's Restricted Aptitude**: The limitations of **Aya 23** were hotly debated, underscoring its suboptimal performance for chat-based applications, with its prowess confined to niche tasks, as per its [technical report](https://cohere.com/research/papers/aya-command-23-8b-and-35b-technical-report-2024-05-23).
- **MoRA Steps into the Fine-Tuning Spotlight**: **MoRA**, introduced as a cutting-edge high-rank updating approach for fine-tuning, entered the discussions with its potential to complement or exceed LoRA, linked with a [dedicated GitHub repository](https://github.com/kongds/MoRA).
- **FFD Bin Packing Woes and Llama 3 Tokens in Limelight**: Issues surfaced regarding the FFD bin packing implementation, specifically in a distributed training context, and fixes for Llama 3 related to untrained tokens, with a [patch shared for sfttrainer](https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L722) to address the latter.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Simulating Realities with Virtual Beings**: The [Virtual Beings Summit](https://www.virtual-beings-summit.com) could be a valuable event for AI professionals, featuring **Will Wright** and focusing on the intersection of AI with simulations. Supporting content can be found on the [Virtual Beings YouTube channel](https://www.youtube.com/channel/UCLNalUbhs_EYB5IAbw1VfaA), which offers insights into AI's role in interactive simulations.

- **AI Dreamscapes with DIAMOND**: The [DIAMOND GitHub repository](https://github.com/eloialonso/diamond) introduces "DIAMOND (DIffusion As a Model Of eNvironment Dreams)," which uses diffusion models in a reinforcement learning context to enhance environmental interactions within AI simulations.

- **Crafting AI "Westworlds" with UE5 and 4Wall**: Discussions around creating immersive experiences suggest that AI Town might leverage **UE5** and integrate voice control to simulate environments akin to "Westworld," with ongoing development info available at the [4Wall Discord](https://discord.gg/vPum4s3h).

- **Venturing into Virtual Reality**: The idea of integrating AI Town with **VR** technology was met with enthusiasm, indicating that engineers are considering novel methods to bring AI-generated environments to life through VR.

- **Animating Avatars with SadTalker and V-Express**: Two GitHub repositories, [SadTalker](https://github.com/OpenTalker/SadTalker) and [Tencent AI Lab's V-Express](https://github.com/tencent-ailab/V-Express), provide tools for creating realistic talking face animations and generating talking head videos, respectively, showcasing advancements in stylized animation technology.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Zyphra Zamba Slithers into the Spotlight**: The new **Zyphra Zamba** model, a blend of mamba and attention mechanisms, has launched with corresponding [technical report](https://www.zyphra.com/s/Zamba.pdf), [PyTorch code](https://github.com/Zyphra/Zamba-torch), and integration into [Hugging Face Transformers](https://github.com/huggingface/transformers/pull/30950). Comparative analysis with **OLMo 1.7** is in progress to benchmark its performance.

**Hushed Release of SD Audio 2.0**: An unauthorized release of **SD Audio 2.0** appeared on 4chan and is also available on a Hugging Face account, sparking discussions among members.

**Station-to-Station Regulation**: Former OpenAI board members Hellen Toner and Tasha McCauley propose in [The Economist](https://www.economist.com/by-invitation/2024/05/26/ai-firms-mustnt-govern-themselves-say-ex-members-of-openais-board) strict regulation over AI companies, emphasizing the inability for such companies to self-regulate due to profit motives and calling out past internal issues.

**Controversy in Command**: The article critiques Sam Altman’s alleged “toxic culture of lying” during his tenure, discussing both internal investigations and public outcry over the absence of transparency.

**A Textbook Case for RL**: The community shared a new resource, [a textbook on reinforcement learning from human feedback](https://github.com/natolambert/rlhf-book) on GitHub, and praised professors Chris Potts and Chris Manning for their engaging teaching styles. Discussions included when the electronic version of Stanford's 224n class would be released, with suggestions to reach out to Chris for concrete timelines.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Tweaking Time Limits in Tech Tests**: Discussions involved the possibility of extending the per-test time limit beyond **9 minutes 34 seconds** to accommodate complex functions like 'Taylor approximations'. A specific issue was with the `clang` function not completing, only reaching approximately 60% completion.

**Crashing Compilations Need Solutions**: One member pointed out the dilemma of generating *excessively large expressions* that crash compilers with errors related to incompatible operand types, specifically doubles.

**Bitwise Operations on Double Drama**: Clarifications were made regarding the impossibility of performing bitwise operations like **XOR** on `double` data types, addressing the cause of a compilation error observed by members.

**Bounty Hunting Heats Up**: Interest spiked in various research-oriented bounties, with discussion on old pull requests and confirmation from **George Hotz** that bounties, such as the one referenced in [tinygrad pull request #4212](https://github.com/tinygrad/tinygrad/pull/4212), are still available.

**Deciphering 'vin' and Discussing Dominators**: George Hotz clarified that 'vin' in the **UOp class** is not an acronym. Additionally, a member questioned why post dominator analysis isn't used for improving scheduling in models, suggesting it might optimize subgraph fusion during execution.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Datasette - LLM (@SimonW) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **YAIG (a16z Infra) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1243282801760145408)** (74 messages🔥🔥): 

- **Semantic similarity overfitting concern**: A member pondered if over-represented response categories in data, despite no particular response being over-represented, could lead to bias. They referenced their prior experience in Research Psychology checking for such issues.
- **Fine-tuning model confusion**: A user struggled with understanding how much fine-tuning incorporates specific user inputs into a model compared to pre-training. They seek clarity on differences between pre-training, curriculum training, and fine-tuning.
- **OpenAI platform sidebars change**: Some participants discussed changes in the OpenAI platform's sidebars, mentioning that **two icons disappeared** (one for threads and another for messages).
- **Rasa and conversational complexity**: A participant shared insights into Rasa's approach to conversational AI, emphasizing the difficulty of creating intent classifiers due to complex conversations. They mentioned that treating intents as entities may reduce complexity.
- **Kyle Corbitt's conference talk recording available**: The recording of Kyle Corbitt's conference talk is now available on the Maven portal, with specific links shared within the discussion.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llm-tracker.info/research/Quantization-Overview">Quantization Overview</a>: Tests How does quantisation affect model output? - 15 basic tests on different quant levels A detailed comparison between GPTQ, AWQ, EXL2, q4_K_M, q4_K_S, and load_in_4bit: perplexity, VRAM, speed, mo...</li><li><a href="https://hamel.dev/notes/llm/inference/03_inference.html">Hamel’s Blog - Optimizing latency</a>: An exploration of ways to optimize on latency.</li><li><a href="https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0066245">no title found</a>: no description found</li><li><a href="https://tenor.com/view/food-good-hungry-yum-gif-11656939384713462119">Food Good GIF - Food Good Hungry - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=d8JMJMvErSg&ab_channel=Rasa">Rasa Algorithm Whiteboard - TED in Practice</a>: In this video we&#39;ll explore how TED works in practice. We&#39;ll build a digital assistant that needs to count down and we&#39;ll see that the hyperparameters really...</li><li><a href="https://www.youtube.com/watch?v=j90NvurJI4I&ab_channel=Rasa">Rasa Algorithm Whiteboard - TED Policy</a>: When you&#39;re making a digital assistant you&#39;ll need more than just algorithms to deal with text. You&#39;ll also need algorithms that deal with sequences of dialo...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7513)">Issues · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cltac3/part3_cause_to_issue_found_possible_bug_llama3/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://us06web.zoom.us/rec/share/POky_IXJdWGOOGZ9BMORn2lZQI53F3d_sOMmESWRbvUm3Us8cWNB7v2rdqnF4raB.95CQod940HlUWGjB?startTime=1716504965000">Video Conferencing, Web Conferencing, Webinars, Screen Sharing</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1243336501018755123)** (23 messages🔥): 

- **LLM Finetuning and `###` usage clarifications**: Discussed the use of `###` in fine-tuning LLMs for sequence generation, noting that it helps the model understand different parts of the input during inference. Appropriately configuring templates during fine-tuning is necessary, including other structures like ChatML.
  
- **Template requirements explained**: Emphasized that inputs during inference need to match the template used during fine-tuning, not necessarily `###` but whatever was set (e.g., Llama 2 chat template). Model hosting services typically manage this templating and structure.

- **Model behavior with and without delimiters**: Delimiters can help a model understand distinct sections of input like changing POVs in Reddit; otherwise unnecessary for general stylistic adaptations. Terminating delimiters or tokens ensure models correctly parse and end responses.

- **End of text token usage**: The concept of an "end of text" token was briefly mentioned as a mechanism for instructing the model to stop generating tokens, indicating efficient input and output management for LLMs.

- **Homework assignments on use cases for LLMs**: Members shared and discussed homework projects applying LLMs to tasks like generating recipes and learning apps. Projects emphasized prompt engineering and retrieval-augmented generation (RAG) techniques among others. Links to resources and shared homework details [here](https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/918c75?item=v69y1k7ohye).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/918c75?item=v69y1k7ohye">no title found</a>: no description found</li><li><a href="https://gpus.llm-utils.org/llama-2-prompt-template/.">Llama 2 Prompt Template</a>: What&rsquo;s the prompt template best practice for prompting the Llama 2 chat models?
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1243344778511515698)** (8 messages🔥): 

- **Reka.ai Jokes About Reunion**: A member humorously commented on seeing another member after a long time, joking, *"You're being kind! I was starting to think I'd never see the light of day again after fast.ai."* They inquired about how they have been and what they're currently building.
- **Conference Recording Request Fulfilled**: A member asked for a recording of the "Conference Talk: From prompt to model," which occurred at 4:30 AM IST. The request was answered affirmatively as the recording is now available on **Maven**.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1243309176722030702)** (18 messages🔥): 

- **Modal Credits Received with Enthusiasm**: Multiple users confirmed receiving credits from Modal and expressed eagerness to start fine-tuning models. One user said, *"Time to hack something."*.
- **Curiosity about Using Modal for Pure PyTorch Code**: A user asked about utilizing Modal for fine-tuning LLMs with pure PyTorch code, comparing it to using Jarvis Labs. Another user confirmed it's possible, sharing their experience training SentenceTransformer models with Modal.
- **Dataset Management in Modal**: Discussion included how to upload datasets and use them within Modal, with detailed code examples and steps provided. Steven Merrill walked through setting up a Parquet file, building volumes, and annotating functions with GPU metadata.
- **Modal Documentation and Examples**: Users shared useful links to Modal documentation and examples, including [volumes documentation](https://modal.com/docs/guide/volumes) and a [TensorFlow tutorial](https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py), which could be adapted for PyTorch.
- **Using Modal for Kaggle Competitions**: One user planned to leverage Modal for a Kaggle competition, involving downloading data, library installations, fine-tuning, and saving models/logs. Another mentioned running Jupyter servers on Modal for up to 24 hours, sharing a link to the [Jupyter inside Modal example](https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/11_notebooks/jupyter_inside_modal.py).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py">modal-examples/06_gpu_and_ml/tensorflow/tensorflow_tutorial.py at main · modal-labs/modal-examples</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.</li><li><a href="https://modal.com/docs/guide/volumes">Volumes</a>: The modal.Volume is a mutable volume built for high-performance file serving. Like the modal.NetworkFileSystem, these volumes can be simultaneously attached to multiple Modal functions, supporting con...</li><li><a href="https://github.com/modal-labs/modal-examples/blob/0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e/11_notebooks/jupyter_inside_modal.py">modal-examples/11_notebooks/jupyter_inside_modal.py at 0ca5778741d23a8c0b81ae78c9fb8cb6e9f9ac9e · modal-labs/modal-examples</a>: Examples of programs built using Modal. Contribute to modal-labs/modal-examples development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/1243307629057671229)** (16 messages🔥): 

- **Saving VSCode repo on Jarvis**: A member inquired about saving their repo on the VSCode instance on Jarvis without pausing it to save credits. Another suggested publishing the code to GitHub and cloning it back as needed, while **paused instances only charge for storage**, which is minimal.
- **Removal of spot instances**: The platform temporarily removed spot instances due to instability and low utilization issues.
- **Fine-tuning open-lama-3b cost and duration**: Fine-tuning the **open-lama-3b** on **gpt4-LLM-cleaned data** took 3 hours 44 minutes on an RTX6000Ada, costing roughly $4. A discussion followed about the small size of LORA weights likely explaining the apparent instant upload to Huggingface.
- **Ampere series error with Axolotl**: A user encountered an error with preprocessing on an A6000, which was resolved by changing **bf16 to false** and **fp16 to true**.
- **Course signup credits issue**: A user reported not receiving credits after signing up for a course and joining Jarvis; the admin responded that new lists are processed, and credits will be added once the user's information is received.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1243335428887806004)** (9 messages🔥): 

- **HF credits to be distributed soon**: Members inquired about the process for obtaining HF credits. **Details will be announced soon by email**, and credits will be granted to attendees who fill out a form being sent over the weekend.
- **Best model for Spanish text generation**: A member asked for recommendations on models for fine-tuning specifically for Spanish text generation tasks. **Mistral 7B** was suggested as a fluent option, and **Llama 3** was mentioned as another model yielding solid results despite not being officially multilingual.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1243453712182149150)** (1 messages): 

- **Upcoming Announcement on Credits**: An announcement regarding the management and distribution of credits will be made soon. *"<@739531318571958272> is going to be running these credits but we are making an announcement soon about them"*.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[kylecorbitt_prompt_to_model](https://discord.com/channels/1238365980128706560/1242221891733946490/1243287896652517376)** (164 messages🔥🔥): 

<ul>
    <li><strong>High Expectations for the Talk</strong>: Members expressed excitement about the talk despite time zone challenges, with a call for recording it. *"I really want to see this but can't make it 😦 will it be recorded?"*</li>
    <li><strong>Link Overflow</strong>: Multiple links were shared including Hamel's [LLM inference notes](https://hamel.dev/notes/llm/inference/03_inference.html), [Argilla](https://argilla.io/), and the [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard). A significant number of resources were gathered from the talk.</li>
    <li><strong>Interactive and Humorous Session</strong>: Members appreciated the interactive vibe with humorous exchanges about fine-tuning and sleep schedules. *"Fine-tuning is not only expensive in GPU compute terms, but also affecting our sleep schedules!"*</li>
    <li><strong>Discussing Efficient Fine-Tuning Techniques</strong>: Various fine-tuning methods such as DoRA, MoRA, and LoRA were discussed, with linked articles like [Answer.AI's efficient fine-tuning](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html). Exploration of context extension techniques like RoPE for models was also mentioned.</li>
    <li><strong>Commandments for Fine-Tuning</strong>: The "Ten Commandments" for deploying fine-tuned models were discussed with a link to the [slides](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67). Members found the content very practical and beneficial for their work.</li>
</ul>

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://argilla.io/">The platform where experts improve AI models</a>: Argilla is a collaboration platform for AI engineers and domain experts that strive for quality, ownership, and efficiency.</li><li><a href="https://hamel.dev/notes/llm/inference/03_inference.html">Hamel’s Blog - Optimizing latency</a>: An exploration of ways to optimize on latency.</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found</li><li><a href="https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html">Answer.AI - Efficient finetuning of Llama 3 with FSDP QDoRA</a>: We’re releasing FSDP QDoRA, a scalable and memory-efficient method to close the gap between parameter efficient finetuning and full finetuning.</li><li><a href="https://x.com/corbtt">Tweet from undefined</a>: no description found</li><li><a href="https://huggingface.co/nomic-ai/nomic-bert-2048">nomic-ai/nomic-bert-2048 · Hugging Face</a>: no description found</li><li><a href="https://docs.argilla.io/en/v1.1.0/guides/steps/1_labelling.html">🏷 Labelling</a>: When labelling, we generally differentiate between manual labelling and co-operative or programmatic labelling. During co-operative labelling, we use external input like rules and inference predict...</li><li><a href="https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67">Ten Commandments to Deploy Fine-Tuned Models in Prod</a>: Ten Commandments To deploy fine-tuned models in prod Kyle Corbitt | @corbtt</li><li><a href="https://openpipe.ai/">OpenPipe: Fine-Tuning for Developers</a>: Convert expensive LLM prompts into fast, cheap fine-tuned models.</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/recordings/88255">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1243277523316637817)** (117 messages🔥🔥): 

- **Sharing the Jarvis Repo Link**: A link to [nisargvp's Jarvis repository on Hugging Face](https://huggingface.co/nisargvp/hc-mistral-alpaca) was shared along with a config file for setting up the model in Axolotl.
- **Guide for Running Models on Modal**: Users discussed running model training smoothly on Modal, pointing out a [quickstart guide from Modal Labs](https://github.com/modal-labs/llm-finetuning) and mentioned seamless operations after initial fixes.
- **TinyLLama Fine-Tuning Blog Post**: The blog post documenting the fine-tuning process of TinyLLama on the alpaca_2k_test dataset using Axolotl and Jarvis, which can be found [here](https://lucasvw.github.io/posts/19_llm_fine_tuning/), was shared and appreciated by the community.
- **Observability in LLM Applications**: Discussions revolved around incorporating observability into LLM applications to collect user feedback and LLM input/output pairs, highlighting the need for better tracking methods.
- **Modal Training Error Support**: Users encountered and resolved issues during Mistral model training using the Modal Labs repo, with community members offering troubleshooting advice and sharing specific error details to diagnose configuration problems.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/inst_tune.html#how-to-add-custom-prompt-format">Axolotl - Instruction Tuning</a>: no description found</li><li><a href="https://lucasvw.github.io/posts/19_llm_fine_tuning/.">Lucas van Walstijn - LLM fine-tuning 101</a>: no description found</li><li><a href="https://wandb.ai/venetispall/llama-3-8b-hermes-sandals-sample-10k/workspace?nw=nwuservenetispall">venetispall</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://huggingface.co/blog/peft_merging">🤗 PEFT welcomes new merging methods</a>: no description found</li><li><a href="https://www.kaggle.com/competitions/lmsys-chatbot-arena">LMSYS - Chatbot Arena Human Preference Predictions | Kaggle</a>: no description found</li><li><a href="https://github.com/modal-labs/llm-finetuning/">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://kaiokendev.github.io/til">Things I’m Learning While Training SuperHOT</a>: pages</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/">Axolotl - Dataset Formats</a>: no description found</li><li><a href="https://huggingface.co/nisargvp/hc-mistral-alpaca">nisargvp/hc-mistral-alpaca · Hugging Face</a>: no description found</li><li><a href="https://maven.com/parlance-labs/fine-tuning/1/syllabus/modules/ac50ed?item=bf4nff4j6bo">no title found</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1793815232177185061">Tweet from Daniel Han (@danielhanchen)</a>: @TheZachMueller @Prince_Canuma @UnslothAI If you&#39;re not using the untrained tokens, it should be OK :) Just sometimes people use the llama-3 template + llama-3 base model, and bad results come abo...</li><li><a href="https://lawwu.github.io/posts/2024-05-23-first-axolotl-finetune/">Lawrence Wu - Finetuning LLMs with Axolotl</a>: no description found</li><li><a href="https://github.com/modal-labs/llm-finetuning/tree/main?tab=readme-ov-file#quickstart">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/tree/main">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning</li><li><a href="https://github.com/modal-labs/llm-finetuning/blob/main/data/modal_docs.jsonl">llm-finetuning/data/modal_docs.jsonl at main · modal-labs/llm-finetuning</a>: Guide for fine-tuning Llama/Mistral/CodeLlama models and more - modal-labs/llm-finetuning
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1243339106675724369)** (3 messages): 

- **Zoom chat confusion leads to Discord**: Members were unsure where to continue their conversation after the Zoom chat was disabled. One member suggested moving their discussion to a specific Discord channel, which made sense to others.
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1243286083022618664)** (32 messages🔥): 

- **Cache Issue in Axolotl Frustrates User**: A member noted that when re-running experiments in **Axolotl**, an unexpected cache used old data samples, which is documented [here](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset_preprocessing.qmd). Renaming the dataset file resolved this, prompting another user to suggest running the pre-process step explicitly.

- **Confusion with Missing Files**: Users encountered issues like missing `simple.yml` or `qlora.yml` files while running training commands on **Jarvislabs** and Google Colab, leading to unsuccessful executions. A member shared that their qlora run took around 6 hours on 2x4090s GPUs, confirming the significance of using the correct files and configurations.

- **Inquiries About Sample Packing**: One member asked if sample packing in Axolotl concatenates multiple dataset rows to fill the max sequence length. Another member confirmed this, explaining that although they are concatenated, the attention is set so that rows don't attend to one another.

- **RuntimeError with BFloat16 in Google Colab**: A RuntimeError related to BFloat16 not implemented for `BFloat16` on T4 GPU led a user to switch from Google Colab to Jarvis-labs. They were advised to check PyTorch and CUDA versions, with a switch to the example configuration solving the issue.

- **Guide on Tokenizer Gotchas Shared**: A user shared a link to Hamel's [notes on tokenizer gotchas](https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html), addressing intricacies in prompt construction and behavioral differences between training and inference due to tokenization handling.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html">Hamel’s Blog - Tokenization Gotchas</a>: Footguns with tokenizers and inferencing LLMs</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/dataset_preprocessing.qmd">axolotl/docs/dataset_preprocessing.qmd at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/tiny-llama/qlora.yml">axolotl/examples/tiny-llama/qlora.yml at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://colab.research.google.com/drive/1jLQDiW47k1vPe_tet4-m6dLVZhnNRet9?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb">axolotl/examples/colab-notebooks/colab-axolotl-example.ipynb at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://lawwu.github.io/posts/2024-05-23-first-axolotl-finetune/#runtimeerror-_amp_foreach_non_finite_check_and_unscale_cuda-not-implemented-for-bfloat16">Lawrence Wu - Finetuning LLMs with Axolotl</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1243291846415749283)** (118 messages🔥🔥): 

- **User confusion over float16 and float32**: There was a question about why float16 numbers appear higher than float32 in a displayed table. A link to a past discussion on the topic was provided to clarify the confusion. 
- **Configuration issues with Jarvislab resolved**: User encountered an error with the Jarvislab training command regarding a missing configuration file. Another user advised changing the command to use `accelerate launch -m axolotl.cli.train hc.yml`, which resolved the issue.

- **Optimizing Axolotl runs on different GPUs**: A member requested advice on adjusting `accelerate` configs for optimized `axolotl` runs on varied GPUs. It was suggested to map configs back to the `axolotl` yaml, avoiding direct acceleration config settings.

- **Resources for learning model Accelerate**: Users discussed how to get started with Accelerate for finetuning tasks, with advice to stick with higher-level abstractions like `axolotl` for simplicity and learning depth.

- **Hyperparameters and Inference precision**: Inquiry on optimal learning rates for extended vs. undertrained models and issues with BF16 precision in T4 GPUs. Suggestions included asking in Zoom QA for hardware-compatible solutions or transforming weights for supported datatypes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/yacineMTB/status/1783939078804701578">Tweet from kache (@yacineMTB)</a>: three mac studios specc&#39;d to the teeth, 7.5k cad each with 192gb unified memory 192 * 3 -&gt; 576gb of &#34;vram&#34; plenty of cpu to go around to power regular server stuff. two could pretty muc...</li><li><a href="https://www.amazon.com/PNY-Generation-Express-DisplayPort-Support/dp/B0CJQH8519">no title found</a>: no description found</li><li><a href="https://www.philschmid.de/instruction-tune-llama-2">Extended Guide: Instruction-tune Llama 2</a>: This blog post is an extended guide on instruction-tuning Llama 2 from Meta AI</li><li><a href="https://arxiv.org/abs/2311.03285">S-LoRA: Serving Thousands of Concurrent LoRA Adapters</a>: The &#34;pretrain-then-finetune&#34; paradigm is commonly adopted in the deployment of large language models. Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning method, is often employed to...</li><li><a href="https://huggingface.co/docs/transformers/en/chat_templating">Templates for Chat Models</a>: no description found</li><li><a href="https://x.com/DavidGFar/status/1793662035227770911">Tweet from David Golchinfar (@DavidGFar)</a>: Hi, everyone!  @FernandoNetoAi , I, @LucasAtkins7, and @erhartford have another surprise for you following the official Kraken release.  We&#39;re excited to introduce Kraken-LoRA, sponsored by @Hyper...</li><li><a href="https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/">The Best GPUs for Deep Learning in 2023 — An In-depth Analysis</a>: Here, I provide an in-depth analysis of GPUs for deep learning/machine learning and explain what is the best GPU for your use-case and budget.</li><li><a href="https://huggingface.co/docs/accelerate/quicktour">Quicktour</a>: no description found</li><li><a href="https://github.com/SkunkworksAI/hydra-moe">GitHub - SkunkworksAI/hydra-moe</a>: Contribute to SkunkworksAI/hydra-moe development by creating an account on GitHub.</li><li><a href="https://x.com/skunkworks_ai">Tweet from undefined</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1243305377974587412)** (192 messages🔥🔥): 

- **PR for latest axolotl and llama 3 demo merged**: The Modal LLM fine-tuning repository now includes the latest axolotl updates and a llama 3 fine-tuning demo.
- **Seeking dataset templates and pre-processing issues**: Members inquire about `chatml.intel` dataset templates and encounter issues during pre-processing, particularly with decoding due to dataset structure lacking numeric IDs. Reference: [Axolotl Docs](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/rlhf.qmd).
- **Clarifications on Axolotl configurations**: Discussions reveal that default config values like `load_in_8bit` and `load_in_4bit` are set to False if not specified, with recommendations to inspect code directly for clarification.
- **Template-free prompt construction confusion**: A member found the documentation on [template-free prompt construction](https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html) confusing, while others clarify the importance of template correctness.
- **Office Hours Q&A highlights debugging and stack insights**: Members express the importance of debugging tools for understanding inputs and samples during training, advocate for rigorous template validation, and suggest callback functions for logging model predictions, referencing [Axolotl Callbacks](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/callbacks/__init__.py).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/callbacks/__init__.py">axolotl/src/axolotl/utils/callbacks/__init__.py at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/h2oai/h2o-llmstudio">GitHub - h2oai/h2o-llmstudio: H2O LLM Studio - a framework and no-code GUI for fine-tuning LLMs. Documentation: https://h2oai.github.io/h2o-llmstudio/</a>: H2O LLM Studio - a framework and no-code GUI for fine-tuning LLMs. Documentation: https://h2oai.github.io/h2o-llmstudio/ - h2oai/h2o-llmstudio</li><li><a href="https://www.philschmid.de/instruction-tune-llama-2">Extended Guide: Instruction-tune Llama 2</a>: This blog post is an extended guide on instruction-tuning Llama 2 from Meta AI</li><li><a href="https://huggingface.co/datasets/GAIR/lima?row=85">GAIR/lima · Datasets at Hugging Face</a>: no description found</li><li><a href="https://gist.github.com/strickvl/e1591b83e3b290fb176e780e7ce7d383">gist:e1591b83e3b290fb176e780e7ce7d383</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://docs.google.com/document/d/1944izw_gwWq9EuaZcNN5lQwiVOKaIYXbYIKi6e9Efsw/edit?usp=sharing">Thread of Questions for Wing - Office Hours</a>: OH Questions for Wing (Axolotl)   Ben Eyal       9:59 AM I was wondering about Template-free prompt construction, I really didn&#39;t understand how it works. The config only needs an output, and the ...</li><li><a href="https://docs.google.com/document/d/1944izw_gwWq9EuaZcNN5lQwiVOKaIYXbYIKi6e9Efsw/edit?pli=1">Thread of Questions for Wing - Office Hours</a>: OH Questions for Wing (Axolotl)   Ben Eyal       9:59 AM I was wondering about Template-free prompt construction, I really didn&#39;t understand how it works. The config only needs an output, and the ...</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/rlhf.qmd">axolotl/docs/rlhf.qmd at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/grgalex/nvshare">GitHub - grgalex/nvshare: Practical GPU Sharing Without Memory Size Constraints</a>: Practical GPU Sharing Without Memory Size Constraints - grgalex/nvshare</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/data/sft.py#L129C4-L152C6.">axolotl/src/axolotl/utils/data/sft.py at main · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0066245">no title found</a>: no description found</li><li><a href="https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0060623">no title found</a>: no description found</li><li><a href="https://github.com/h2oai/">H2O.ai</a>: Fast Scalable Machine Learning For Smarter Applications - H2O.ai</li><li><a href="https://tenor.com/view/trust-no-one-crazy-chris-henry-thomas-just-beyond-dont-trust-anybody-gif-23566469">Trust No One Crazy Chris GIF - Trust No One Crazy Chris Henry Thomas - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1243310332407971850)** (1 messages): 

- **Visualize Proteins with Proteinviz**: Check out [Proteinviz](https://huggingface.co/spaces/as-cle-bert/proteinviz) for creating custom visuals of proteins. This tool is made by a dedicated community member.
  
- **Speedy SDXL Results**: The [SDXL flash](https://huggingface.co/spaces/KingNish/SDXL-Flash) space delivers impressive results fast. Credit goes to the creator for this efficient build.

- **Custom Tokenizers Inspired by Karpathy**: A community member shared their [custom tokenizer](https://github.com/apehex/tokun), which is inspired by Karpathy’s work. This highlights ongoing innovations within the community.

- **Mistral-7B v0.3 Demo**: Experience rapid performance with the [Mistral-7B v0.3 chat](https://huggingface.co/spaces/ehristoforu/mistral-7b-v0.3-chat) demo. It's another example of cutting-edge developments by active contributors.

- **Create Transparent Images with Diffusers**: Generate [transparent images](https://github.com/rootonchair/diffuser_layerdiffuse) using Diffusers, a project facilitated by another community member. This feature allows for creative visual outputs using advanced diffusing techniques.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=S-gy6NUOGSs)">Agentic AI Solutions / Adaptive AI Solutions - Episode 1:  CrewAI With Preston McCauley</a>: In Episode 1, we explore a brief introduction to #AdaptiveAI and #Agentic AI approaches.https://www.linkedin.com/in/preston-mccauley-immersive-ux/Join Presto...</li><li><a href="https://youtu.be/jddSbTLw0gc)">What is an Instruction Tuned Model?</a>: What is Instruction Tuning?  What are Instruction Tuned models? What is a Pretrained Model? How can I make my Large Language Model follow Instructions?These ...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1243283012477522071)** (490 messages🔥🔥🔥): 

- **AutoTrain Data Formatting Questions**: Members discussed how to format data for finetuning in **AutoTrain**, with suggestions to reference the [AutoTrain documentation](https://hf.co/docs/autotrain). Example CSV formats and nuances of input data types were shared, enhancing clarity on setup.
- **Advanced LLM Fine-Tuning**: The **difference between DPO and RHLF** methods for fine-tuning LLMs was highlighted, suggesting **SFT followed by RHLF** for teaching text-completion models conversational norms. Links to specific datasets and finer model adjustments were also shared.
- **Pandora Model Excitement**: Details about the Pandora model, a new **open-source text-to-video** model, were shared along with a preview link. Discussions on its **smartness** and potential applications created significant excitement among members.
- **Mobius Model Controversy**: The upcoming **Mobius diffusion model** faced scrutiny with comments about controlled quality and composition training. Resulting discussions emphasized its potential to significantly reduce the cost and complexity of developing new diffusion models.
- **Learning and Development Resources**: Several members including @temeretam discussed educational and professional paths for advancing in AI, while others sought advice on specific coding and data handling problems, referencing both **GitHub** and **Hugging Face** documentation links for technical support.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/maitrix-org/Pandora">maitrix-org/Pandora · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/huggingface_hub/guides/download#filter-files-to-download">Download files from the Hub</a>: no description found</li><li><a href="https://tenor.com/view/babuin-gif-27648024">Babuin GIF - Babuin - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/que-gif-27530657">Que GIF - Que - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://imgsys.org/rankings">imgsys.org | an image model arena by fal.ai</a>: A generative AI arena where you can test different prompts and pick the results you like the most. Check-out the model rankings and try it yourself!</li><li><a href="https://huggingface.co/docs/transformers/main/chat_templating">Templates for Chat Models</a>: no description found</li><li><a href="https://x.com/DataPlusEngine/status/1793817514956259460">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: Our upcoming paper outlines and enables making entirely new base diffusion models without the need to extensively pretrain a new model from scratch. We can in a controlled way, break all the quality a...</li><li><a href="https://huggingface.co/datasets/nroggendorff/mayo">nroggendorff/mayo · Datasets at Hugging Face</a>: no description found</li><li><a href="https://youtu.be/zLvFc_24vSM?si=TeS_EkFu9BeyYDbz">Rabbit Gaslit Me, So I Dug Deeper</a>: Is the LAM a Scam? Down the rabbit hole we go.Support Investigative Journalism: ► Patreon: https://patreon.com/coffeezillaPeople who helped this investigatio...</li><li><a href="https://x.com/DataPlusEngine/status/1793803117642854732">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: We gave the @FAL  early access to the upcoming Mobius model and its only been up on http://imgsys.org for 3 hours. its already the best stable diffusion based image model in the world based on human p...</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main">mistralai/Mixtral-8x7B-v0.1 at main</a>: no description found</li><li><a href="https://tenor.com/view/frank-castle-wait-please-stop-please-no-please-gif-21133188">Frank Castle Wait GIF - Frank Castle Wait Please Stop - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.instagram.com/p/C7U3hOPRJhR/">Noa Roggendorff on Instagram: &quot;epic

#ai&quot;</a>: 2 likes, 1 comments - noaroggendorff on May 23, 2024: &quot;epic  #ai&quot;. </li><li><a href="https://huggingface.co/docs/datasets/v2.19.0/en/process#rename>">Process</a>: no description found</li><li><a href="https://tenor.com/view/kurt-kurt-angle-100-yard-stare-what-are-you-serious-gif-4081464694509837388">Kurt Kurt Angle GIF - Kurt Kurt angle 100 yard stare - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://emoji.gg/category/6/blobs">Blobs Emojis for Discord & Slack - Discord Emoji</a>: Find Blobs emojis to use on Discord or Slack - Emoji.gg, The largest directory of free custom emojis on the internet.</li><li><a href="https://hf.co/docs/autotrain">What is AutoTrain Advanced?</a>: no description found</li><li><a href="https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#installation">GitHub - hpcaitech/Open-Sora: Open-Sora: Democratizing Efficient Video Production for All</a>: Open-Sora: Democratizing Efficient Video Production for All - hpcaitech/Open-Sora</li><li><a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan?tab=readme-ov-file">GitHub - PKU-YuanGroup/Open-Sora-Plan: This project aim to reproduce Sora (Open AI T2V model), we wish the open source community contribute to this project.</a>: This project aim to reproduce Sora (Open AI T2V model), we wish the open source community contribute to this project. - PKU-YuanGroup/Open-Sora-Plan</li><li><a href="https://slackmojis.com/categories/25-blob-cats-emojis">
Blob Cats emojis on Slack
</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1243424993426014309)** (8 messages🔥): 

- **Deep RL for Embodied AI sparks interest**: A member shared their enthusiasm about learning Deep Reinforcement Learning specifically for Embodied AI applications and invited detailed updates on progress.

- **Fast.ai courses recommended for AI beginners**: Suggested Fast.ai’s part 1 & 2 courses which cover practical deep learning tasks using HuggingFace libraries and offer a strong foundation for beginners in deep learning. Course details can be found [here](https://course.fast.ai/).

- **Coursera course on Generative AI with LLMs**: Recommended **Generative AI with Large Language Models** course on Coursera for those interested in gaining foundational knowledge in AI. The course is designed to be completed in 3 weeks, details available [here](https://www.coursera.org/learn/generative-ai-with-llms).

- **PixART Diffusion Model Call Event**: Announced a call event for an in-depth review of the PixART diffusion model for text-to-image synthesis, scheduled for Friday at 10:00 AM Pacific time. Additional information and community interaction can be found [here](https://lu.ma/arxivdive-2024-05-24).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>: A free course designed for people with some coding experience, who want to learn how to apply deep learning and machine learning to practical problems.</li><li><a href="https://www.coursera.org/learn/generative-ai-with-llms">Generative AI with Large Language Models</a>: In Generative AI with Large Language Models (LLMs), you’ll learn the fundamentals of how generative AI works, and how to deploy it in ... Enroll for free.</li><li><a href="https://lu.ma/arxivdive-2024-05-24?tk=F1jNfh">Arxiv Dives with Oxen.AI - Fine Tuning Diffusion Transformers (DiT) · Zoom · Luma</a>: Hey Nerd, join the Herd!... for a little book/paper review. WHAT TO EXPECT Each week we pick a topic to cover in depth and have open Q/A and discussion.…
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1243438483884871721)** (3 messages): 

- **Exciting ChatGPT Applications in Drug Discovery**: A link to a study was shared discussing the potential use of **ChatGPT and other LLMs in next-generation drug discovery**. The article, published in the International Journal of Surgery, highlights contributions from various institutions across India and Bangladesh [Read more](https://journals.lww.com/international-journal-of-surgery/fulltext/2023/12000/chatgpt_or_llm_in_next_generation_drug_discovery.78.aspx).
  
- **PostgresML and LlamaIndex Make Waves**: An integration of **PostgresML** with **LlamaIndex** was highlighted in a recent Medium post. This [integration](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939) promises to unlock new potentials in AI advancements, with detailed insights available in the article.

**Link mentioned**: <a href="https://journals.lww.com/international-journal-of-surgery/fulltext/2023/12000/chatgpt_or_llm_in_next_generation_drug_discovery.78.aspx">ChatGPT or LLM in next-generation drug discovery and... : International Journal of Surgery</a>: An abstract is unavailable.

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1243293254309511218)** (22 messages🔥): 

- **Protein Dataset Gets Major Updates**: A member shared updates on their protein visualization project, adding examples for **human hemoglobin**, **mouse GTPase**, and **human ribosomal protein**. They also implemented support for **3D rendering** and created an in-depth example table on [GitHub](https://github.com/AstraBert/proteinviz/blob/main/examples.md).

- **Transcription App with OpenAI's Whisper Rocks!**: A member introduced their transcription app for YouTube videos, audio files, and video files, utilizing **OpenAI's Whisper**. Check it out on [Hugging Face Spaces](https://huggingface.co/spaces/tensorkelechi/vidtext).

- **Call for Feedback on Decentralized Internet Infra**: One member requested feedback and participation in a survey for their project building infrastructure for a decentralized and agent-centric internet: [survey link](https://hai.ai/). This sparked a debate about spamming channels and the ethics of data collection through surveys.

- **3D Model Visualization in Browser Challenges**: Despite challenges with **3D model rendering** of protein structures in the Gradio browser, there is ongoing effort to find a solution. Helpful resources include a blog post on [Hugging Face](https://huggingface.co/blog/spaces_3dmoljs).

- **SimpleTuner Bug Fixes Improve Training**: A member highlighted that fixing some minor bugs in **SimpleTuner** significantly enhanced its training performance. Now it trains better than ever.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/tensorkelechi/vidtext">Vidtext - a Hugging Face Space by tensorkelechi</a>: no description found</li><li><a href="https://huggingface.co/blog/spaces_3dmoljs">Visualize proteins on Hugging Face Spaces</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1243525495769792583)** (4 messages): 

- **Monthly Computer Vision Hangout Announced**: An upcoming monthly **Computer Vision Hangout** was introduced, aimed at discussing projects, ideas, and problems in CV-related fields. More details and event participation can be found [here](https://discord.gg/MkHyuG9C?event=1243129304863215656).

- **Seeking Invoice Processing Solution**: A member inquired about an open-source neural network or paid API for extracting structured line-by-line information from scanned invoices. They requested the output to be formatted as JSON, specifying fields like product_id, description, quantity, unit_price, and total_price.

- **Looking for Deep Learning Study Partner**: A user expressed interest in finding a deep learning study partner who shares a passion for AI and data science. They emphasized a mutual drive to explore neural networks, complex algorithms, and innovative projects.

- **Request for ViT Resources in Depth Estimation**: Another member asked for resources on utilizing Vision Transformers (ViT) for monocular depth estimation. They indicated an interest in building their own model using ViT and are seeking guidance.

**Link mentioned**: <a href="https://discord.gg/MkHyuG9C?event=1243129304863215656">Join the Hugging Face Discord Server!</a>: We&#x27;re working to democratize good machine learning 🤗Verify to link your Hub and Discord accounts! | 79727 members

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1243507818158489661)** (8 messages🔥): 

- **Quantisation Anomalies in Mistral v0.3 Instruct**: A member reported unexpected performance issues when comparing **Mistral v0.3 Instruct** using **bitsandbytes** 8-bit, 4-bit, and fp16 quantisation levels. They found that while fp16 and 4-bit took around 100 seconds, 8-bit took 500 seconds, despite expectations of 8-bit being faster than 4-bit.
- **Switching from Pipelines to Generate Without Impact**: The same user noted that switching from **pipelines** to the **generate()** method, per the documentation for text generation with 8-bit models, did not improve the performance as expected.
- **Bitsandbytes Version and Optimization Tips**: In response to the performance issue, another member inquired about the **version of bitsandbytes** being used and suggested trying **int8_threshold=0** for potential performance gains. The original user mentioned they are using a batch size of 1 and contexts ranging from 500 to 2000 tokens.
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1243278218262351995)** (6 messages): 

- **Seeking NLG Learning Resources**: A member asked for recommendations for learning **Natural Language Generation (NLG)**. Responses to this query were not provided in the message history.

- **Query about Training Stable Diffusion on Custom Dataset**: Another member asked for **official documentation** on training Stable Diffusion (SD) to generate images from a custom dataset such as MNIST. They mentioned finding documentation on the site, but it seemed to focus on unconditional generation.

- **Looking for Deep Learning Study Partner**: A different member expressed interest in finding a partner to **learn deep learning** with. They emphasized a desire for someone equally passionate about AI and data science, keen to explore neural networks, complex algorithms, and innovative projects.

- **Help Needed for Converting pth+index File to Hugging Face Link**: A member requested assistance in converting a **pth+index file** into a Hugging Face link RVC model. This technical query did not receive an immediately visible response.

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1243286415463284879)** (493 messages🔥🔥🔥): 

- **Perplexity vs. ChatGPT for Data Processing**: Discussion emerged on the capabilities of **Perplexity** and **ChatGPT** in processing CSV files, with mentions that **Perplexity** already supports CSV uploads. Julius AI, an alternative for data analysis, was highlighted for running on Python and leveraging LLMs like Claude 3 or GPT-4.
  
- **Disappointment with Claude 3 Opus**: Users expressed dissatisfaction with **Claude 3 Opus** due to increased restrictions and lower utility, particularly in handling copyrighted material. Some suggested alternatives like GPT-4o but acknowledged that Claude 3's usefulness has diminished.

- **Pro Search Features and Enhancements**: Users noted new features in **Pro Search**, with enhancements including multi-step reasoning and updated API specs fetching. However, some users observed that such updates might be part of A/B testing and only involve UI changes rather than backend improvements.

- **Tool Integrations and Custom Function Calls**: There were discussions on **Claude’s** capacity for external tool integration via APIs, and attempts to replicate ChatGPT’s data analysis tool through custom function calls and serverless backend solutions. Links to relevant documentation like [Tool Use with Claude](https://docs.anthropic.com/en/docs/tool-use) were shared.

- **Ethical AI and Communication Analysis Projects**: Talks included the creation of GPTs for communication analysis and ethical behavior monitoring, with suggestions that such tools could help improve workplace communication and reduce wrongful termination suits. Users debated the feasibility and philosophical implications of encoding ethics into algorithms.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=0T1444HJbt0">Mistral&#39;s new 7B Model with Native Function Calling</a>: Colab Code - https://drp.li/K98Z7🕵️ Interested in building LLM Agents? Fill out the form belowBuilding LLM Agents Form: https://drp.li/dIMes👨‍💻Github:http...</li><li><a href="https://v0.dev/">v0 by Vercel</a>: Generate UI with simple text prompts. Copy, paste, ship.</li><li><a href="https://tenor.com/view/google-chrome-pacman-eating-gif-13756279">Google Chrome Pacman GIF - Google Chrome Pacman Eating - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/aladdin-disney-cartoons-jasmine-ic-an-show-you-the-world-gif-4545341">Aladdin Disney GIF - Aladdin Disney Cartoons - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/AZURE/comments/1bzs8gr/have_you_purchased_openai_ptus_how_much_did_it/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://docs.anthropic.com/en/docs/tool-use">Tool use (function calling) - Anthropic</a>: no description found</li><li><a href="https://aws.amazon.com/bedrock">Build Generative AI Applications with Foundation Models - Amazon Bedrock - AWS</a>: no description found</li><li><a href="https://search.brave.com/search?q=%s&source=desktop">Brave Search</a>: Search the web privately…</li><li><a href="https://aws.amazon.com/bedrock/pricing/">Build Generative AI Applications with Foundation Models - Amazon Bedrock Pricing - AWS</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1243340241197338706)** (7 messages): 

- **Peran Kepala Sekolah shared**: A brief link is shared to [Peran Kepala Sekolah](https://www.perplexity.ai/search/Peran-Kepala-Sekolah-ECYSEyQXTviCYDqqDgM8sw) without additional context or discussion.
- **What is PB55 explained**: A link provided to [what is the PB55](https://www.perplexity.ai/search/what-is-the-PB55hhXYRDGAVd7JjWDhaA) for further reading.
- **Origin of 'makura' explored**: A user shares a link to explore the etymology of the Japanese word "枕（まくら / makura）" [here](https://www.perplexity.ai/search/oDyPhU47T26IM1W0f7GQIg), which means pillow.
- **Ensure thread shareability**: A reminder is given with an attachment to ensure threads are shareable with a link to [Discord thread](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825).
- **Stuart Hall’s theory discussed**: [Stuart Hall’s encoding/decoding model](https://www.perplexity.ai/search/Explain-Stuart-Halls-IV.my4LjS2mNXyxPyWVLOw#0) is shared.
- **Opus 50 limit queried**: A user inquires about the [Opus 50 limit](https://www.perplexity.ai/search/Opus-50-limit-c2EHUbzTQGCocG2d17MrLg).
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1243398440315260958)** (1 messages): 

- **References feature still in beta limbo**: A user questioned the status of references being in beta and expressed frustration over not receiving a response after applying three times. They asked if anyone knew when this feature would be released in the API.
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1243290070820192337)** (427 messages🔥🔥🔥): 

- **Rumors of RTX 5090 Specifications Stir Debate**: Discussions center around new rumors that the RTX 5090 may feature 32GB VRAM, igniting skepticism about the feasibility and utility. One member shared a [link](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/) to purported images, but others criticized these as misleading.

- **Stable Diffusion Installation Guidance**: A member seeks advice on installing Stable Diffusion with an AMD 5700XT GPU. Recommendations included trying web services like [Craiyon](https://www.craiyon.com/) initially, due to potential complications with AMD hardware.

- **Pricing and Access of Stable Diffusion 3**: Users debated the merits of Stable Diffusion 3 vs. Midjourney, with some noting that SD3 is available for a free trial. However, it appears that a Stability membership is required for continued access.

- **Introduction of Mobius Model Generates Interest**: DataPlusEngine announced the upcoming Mobius model on [Twitter](https://x.com/DataPlusEngine/status/1793803117642854732), claiming it to be the best stable diffusion-based image model. The model is described as "neither a base model nor a fine tune" and touted for its ability to create new base models efficiently.

- **Curiosity Over GPU Performance and Costs**: New GPU models, particularly the 5090, sparked discussions about memory and training speeds. Members noted that higher VRAM like 32GB could detract from sales of high-end data center GPUs like the H100/A100, hinting this could influence Nvidia's strategy.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/DataPlusEngine/status/1793803117642854732">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: We gave the @FAL  early access to the upcoming Mobius model and its only been up on http://imgsys.org for 3 hours. its already the best stable diffusion based image model in the world based on human p...</li><li><a href="https://tenor.com/view/never-finn-adventure-time-gif-10874543">Never Finn GIF - Never Finn Adventure Time - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/-CyupSdXfI0">WOW   Every Owen Wilson Wow  ever said, just WOW</a>: Owen Wilson is just one of my favorite actors, his &quot;wow&quot; &#39;s are just legendary - so here is a curated collection of all of them in one place</li><li><a href="https://www.youtube.com/watch?v=k1hbRvSnFZg">A Moebius-metró | teljes film magyarul</a>: argentin misztikus/sci-fi/thriller, 1996 - teljes filmA világ egyik legzsúfoltabb metrórendszerében nyom nélkül eltűnik egy utasokkal teli metrószerelvény, c...</li><li><a href="https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/">Geforce RTX 5090 soll mit 32 GiB GDDR7 und gleich drei PCBs an den Start gehen [Gerücht]</a>: Bilder zu Artikel: Geforce RTX 5090 soll mit 32 GiB GDDR7 und gleich drei PCBs an den Start gehen [Gerücht] - Geforce RTX 5090</li><li><a href="https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-d">News zu Grafikkarten</a>: Sie finden hier immer die besten News zu Grafikkarten
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1243280018524606544)** (275 messages🔥🔥): 

- **PEFT Training Question Resolved**: A user faced an issue with the `config.json` not being created during PEFT training and was advised to copy from the base model's configuration. The user confirmed it worked and thanked the community for the help.

- **Llama 3's Bugs Noted**: Some users discussed that "Some of Llama 3's base (not instruct) weights are 'buggy'" but Unsloth auto-fixes these. It was advised to use reserved tokens during training and ensure the tokenizer and `lm_head` are trained.

- **System Prompt Improves Llama3**: Users mentioned that adding a system prompt improves Llama3 finetuning performance. One user confirmed that even a blank system prompt can positively impact results.

- **Phi 3 Model Support Announced**: It was announced that Phi 3 models, including medium support, are now available. The community showed excitement and shared links to relevant blog posts for more details.

- **Creepy Imprint with Stable Diffusion**: Users shared eerie experiences with voice cloning and creepy artifacts generated by Stable Diffusion. They posted links to related [YouTube video](https://youtube.com/shorts/o4kVe2NwRYY?si=ILtLzWy1XTAPALKc) and a [Reddit discussion](https://www.reddit.com/r/StableDiffusion/comments/1b10o36/creepy_imprint_from_stable_difussion/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/phi3">Finetune Phi-3 with Unsloth</a>: Fine-tune Microsoft&#x27;s new model Phi 3 medium, small &amp; mini easily with 6x longer context lengths via Unsloth!</li><li><a href="https://huggingface.co/CohereForAI/aya-23-8B">CohereForAI/aya-23-8B · Hugging Face</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1793758720541016567">Tweet from Unsloth AI (@UnslothAI)</a>: We have resolved issues with training Llama 3, so finetuning is much better now!  Unsloth now supports the new Phi-3 models, Mistral v3, Qwen and more!  Read our blog: http://unsloth.ai/blog/phi3</li><li><a href="https://youtube.com/shorts/o4kVe2NwRYY?si=ILtLzWy1XTAPALKc">can i get a chicken tendie combo please</a>: no description found</li><li><a href="https://github.com/babycommando/machinascript-for-robots">GitHub - babycommando/machinascript-for-robots: Build LLM-powered robots in your garage with MachinaScript For Robots!</a>: Build LLM-powered robots in your garage with MachinaScript For Robots! - babycommando/machinascript-for-robots</li><li><a href="https://github.com/ggerganov/llama.cpp/issues">Issues · ggerganov/llama.cpp</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1b10o36/creepy_imprint_from_stable_diffusion/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7062">Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly · Issue #7062 · ggerganov/llama.cpp</a>: I&#39;m running Unsloth to fine tune LORA the Instruct model on llama3-8b . 1: I merge the model with the LORA adapter into safetensors 2: Running inference in python both with the merged model direct...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1243306406195626034)** (1 messages): 

- **Phi-3 and Mistral v3 now live**: *Unsloth now supports Phi-3, Mistral v3, and many other new models.* Check out the [release details](https://github.com/unslothai/unsloth/releases/tag/May-2024).

- **Llama 3 issues resolved**: *We've fixed all Llama 3 issues so finetuning is much better now.* For a deeper dive, refer to this [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1cwwgkz/is_llama_3_just_not_a_good_model_for_finetuning/).

- **Explore free Colab notebooks**: Access our [Phi-3 medium notebook](https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing), [Mistral v3 notebook](https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing), and more.

- **New model support and GitHub Accelerator**: See our latest model additions on [Hugging Face](https://huggingface.co/unsloth) and learn about our participation in the [GitHub 2024 Accelerator](https://github.blog/2024-05-23-2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/).

- **Celebration of AI innovation**: *We're excited to join 10 other projects in GitHub's 2024 Accelerator, highlighting the global impact and rapid advancement of AI innovation.*
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/phi3">Finetune Phi-3 with Unsloth</a>: Fine-tune Microsoft&#x27;s new model Phi 3 medium, small &amp; mini easily with 6x longer context lengths via Unsloth!</li><li><a href="https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://github.blog/2024-05-23-2024-github-accelerator-meet-the-11-projects-shaping-open-source-ai/)">2024 GitHub Accelerator: Meet the 11 projects shaping open source AI</a>: Announcing the second cohort, delivering value to projects, and driving a new frontier.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1243440379672530985)** (4 messages): 

- **Seek Local VSCode Copilot Recommendations**: One user asked, *"Does anyone use local vscode 'copilot'? I would like to try some. Looking for recommendation :)"*. Another responded with, *"try continue"*, followed by the initial user expressing thanks, *"Thanks, will try:)"*.
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1243277920252727329)** (103 messages🔥🔥): 

- **Sloth Phi-3 Inference Poses Performance Issue**: A user reported *slower inference times* when using the **Unsloth Phi-3 model** compared to the original. They shared a [Colab notebook](https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X) to diagnose the issue, but even after suggested modifications, the problem persisted.
  
- **Custom Model Quantization Issue**: One member experienced issues quantizing a custom model derived from an Unsloth notebook. They received errors related to unsupported architecture with **llama.cpp** and **Docker**.

- **Resource Requirements for Different Models**: Queries about VRAM requirements indicated that **12GB is sufficient for Phi 3 mini**, while **16GB is needed for Phi 3 medium**. It was also noted that for larger tasks like summarization with a bigger context window, **renting computing resources** might be necessary.

- **Evaluation DataSet Criteria**: A discussion highlighted the importance of using consistent datasets for training and evaluation. Specifically, unslothai's public datasets on Hugging Face, such as those listed in the [Blackhole Collection](https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a), were recommended for high quality.

- **Compatibility and Custom Model Support**: Several users inquired about the compatibility of **Unsloth** with older Macs and using GPU-less systems, confirmed that Unsloth is optimized for CUDA and GPU usage. Several workarounds and tips were suggested for CPU-only systems and custom model support.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a">Blackhole - a lamhieu Collection</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X#scrollTo=0zM8gPJUGySh">Google Colab</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues">Issues · unslothai/unsloth</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - Issues · unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/_utils.py#L179">unsloth/unsloth/models/_utils.py at main · unslothai/unsloth</a>: Finetune Llama 3, Mistral &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1243412700105674753)** (2 messages): 

- **Engineer offers enterprise experience to Unsloth**: A member, higginsconsultingptyltd_39617, congratulated others on joining the accelerators at Build Club and Github and proposed leveraging their enterprise experience to assist Unsloth. Another member responded positively, expressing eagerness to discuss further, "*Absolutely we'd love to!*"
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1243288751313256490)** (12 messages🔥): 

- **Master of Plain-Speak Talks PixART Diffusion Model**: Interested members can *"hear a Master of Plain-Speak describe how he fine-tuned the PixART diffusion model"* during a call today at 10:00 AM Pacific Time. Join the event and [link to Discord](https://discord.gg/s3tBEn7Ptg) for further discussion or view past topics on their [blog](https://www.oxen.ai/blog) and [YouTube videos](https://www.youtube.com/@oxen-ai/videos).
  
- **Excitement Over Intel Libraries**: A member expressed excitement to *"tinker with the Intel libraries"* while discussing IPEX and BigDL separation. Potential collaboration and exploration of Intel's improvements were mentioned.

- **Stable Functionality of IPEX-LLM**: Although one member hasn't used **IPEX-LLM**, they've found that it has *"rock-solid stable"* support where it exists. Discussions included improvements in IPEX-LLM's setup.

- **Tinygrad OpenCL Setup Insights**: If performance is not the main concern, *"tinygrad OpenCL is trivial to set up and get running"*, suggested one member. Another member humorously criticized geohot's lack of interest due to memory bandwidth limitations.

- **Experimental Stint with `drm/xe` Driver**: Currently, a member is running the experimental `drm/xe` driver without major issues, apart from the known constraints. They expressed hope that **Battlemage** will perform better.

**Link mentioned**: <a href="https://lu.ma/arxivdive-2024-05-24?tk=F1jNfh">Arxiv Dives with Oxen.AI - Fine Tuning Diffusion Transformers (DiT) · Zoom · Luma</a>: Hey Nerd, join the Herd!... for a little book/paper review. WHAT TO EXPECT Each week we pick a topic to cover in depth and have open Q/A and discussion.…

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1243350936168960082)** (6 messages): 

- **TAS Mario Sunshine sparks AI speedrun debate**: A member shared a [YouTube video](https://youtu.be/W_jwBHd9Ij0) showcasing a tool-assisted speedrun of "Super Mario Sunshine" and discussed the potential of AI mastering such techniques. They pondered the intriguing developments AI might bring to speedrunning and game engine manipulation by imposing specific limitations.

- **Pannenkoek2012's Mario 64 praised**: Another [YouTube video](https://youtu.be/lgW2fHCL9sY) was shared featuring a zero A-press speedrun of "Super Mario 64" by Pannenkoek2012. The member appreciated the content, noting its insights into evolving AI and consciousness through rapid thought processes.

- **Prophetic AI's Halo and Morpheus-1 impress**: A link to [Prophetic AI](https://propheticai.co) was shared, highlighting the **Halo**, a non-invasive neural device for lucid dreaming, and **Morpheus-1**, an ultrasonic transformer generating holograms for neurostimulation. The member emphasized the extreme potential of these technologies for exploring the subconscious mind and consciousness enhancement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://propheticai.co">Prophetic</a>: Prophetic is a megaproject to expand, explore, and understand the true nature of consciousness. We are a neuromodulation company that brings together state-of-the-art neural &quot;reading&quot; and &q...</li><li><a href="https://youtu.be/W_jwBHd9Ij0">[TAS] GC Super Mario Sunshine by zelpikukirby &amp; Goldfire in 1:08:32.58</a>: This is a tool-assisted speedrun. For more information, see https://tasvideos.org/3731MTAS originally published on 2018-06-18In the highly anticipated sequel...</li><li><a href="https://youtu.be/lgW2fHCL9sY">Super Mario 64 70 stars in 0 a presses by Pannenkoek2012</a>: This video is made as a thank you to pannenkoek for such great content like this. All footage is made and owned by pannenkoek ( https://www.youtube.com/user/...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1243278543073448060)** (280 messages🔥🔥): 

- **New Paper on Transformer Circuits**: A user shared a link to the new paper, [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html), suggesting the community check it out.
- **PyTorchModelHubMixin Class by HF**: A member highlighted a class called `PyTorchModelHubMixin` created by Hugging Face, which allows seamless integration of AI models with the HUB using save_pretrained, push_to_hub, and from_pretrained methods. However, AI models need to stay under 50GB as sharding is not supported yet.
- **Mobius Model Impresses Community**: Discussion on the [Mobius model](https://x.com/DataPlusEngine/status/1793803117642854732) showcased its high performance in image generation, particularly in Pixar-style renderings and multi-word text generation. It also generated excitement for potential open-sourcing and further papers explaining its training method.
- **Lively Debate on LLM Understanding**: A heated discussion unfolded around whether LLMs truly understand concepts, with one user pointing to interpretability research as a major source of empirical evidence, while another argued that current interpretability efforts are insufficient. They referenced recent research including a paper from Anthropic and debates around the significance of interpretability in AI.
- **Technical Repo for RLHF Models Shared**: A GitHub repository, [Online RLHF](https://github.com/RLHFlow/Online-RLHF), was shared, detailing a workflow for training reward models for Reinforcement Learning from Human Feedback (RLHF), which aims to surpass results from offline learning methods.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/DataPlusEngine/status/1793803117642854732">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: We gave the @FAL  early access to the upcoming Mobius model and its only been up on http://imgsys.org for 3 hours. its already the best stable diffusion based image model in the world based on human p...</li><li><a href="https://x.com/DataPlusEngine/status/1793817514956259460?t=Phj_r_qcguWbrL0Q5ZdKbQ&s=19">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: Our upcoming paper outlines and enables making entirely new base diffusion models without the need to extensively pretrain a new model from scratch. We can in a controlled way, break all the quality a...</li><li><a href="https://vgel.me/posts/representation-engineering/">
    
      
        Representation Engineering Mistral-7B an Acid Trip
      
    
  </a>: no description found</li><li><a href="https://www.anthropic.com/news/mapping-mind-language-model">Mapping the Mind of a Large Language Model</a>: We have identified how millions of concepts are represented inside Claude Sonnet, one of our deployed large language models. This is the first ever detailed look inside a modern, production-grade larg...</li><li><a href="https://github.com/RLHFlow/Online-RLHF">GitHub - RLHFlow/Online-RLHF: A recipe to train reward models for RLHF.</a>: A recipe to train reward models for RLHF. Contribute to RLHFlow/Online-RLHF development by creating an account on GitHub.</li><li><a href="https://huggingface.co/RLHFlow">RLHFlow (RLHFlow)</a>: no description found</li><li><a href="https://huggingface.co/RLHFlow/LLaMA3-iterative-DPO-final">RLHFlow/LLaMA3-iterative-DPO-final · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/RLHFlow/LLaMA3-SFT">RLHFlow/LLaMA3-SFT · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1243285074905137253)** (8 messages🔥): 

- **Llama.cpp script handles function calls**: A member shared an update about creating a script using **llama.cpp** that manages function calls and returns answers from the model based on tool responses. They mentioned being inspired by the **Hermes Pro 2** GitHub repo and offered to create a pull request to add a notebook.
- **Hermes model praised**: The same member described the **Hermes model** as "a beast."
- **Looking for LoRA resources on a 3080**: A member asked for resources to perform **Llama3 LoRA** on a 3080 GPU with 10GB. The response recommended checking out **unsloth** or **axolotl**.
- **New developer introduction**: A new member, a developer from **torchtune**, introduced themselves and mentioned their interest in tool-calling with **Mistral v0.3**. They sought advice on fine-tuning models for tool-calling and queried experiences with zero-shot new tools.
  

---


### **Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1243424354092580937)** (6 messages): 

- **Kquant criticizes kquant's reputation**: Members expressed skepticism about **kquant**, with one stating, *“I’ve heard it’s not very great.”* Another concurred, sharing similar opinions from colleagues.
  
- **Concerns on LLM Capabilities**: There was agreement that **kquant's** capabilities, especially on the **LLM side**, are dubious, though its vision capabilities were not discussed. 

- **Disappointment over product removal**: A member mentioned the removal of "Sky" in a playful manner, which caused amusement and mirrored shared sentiments of disappointment. Another member humorously expressed that they "stol't our waifus."
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1243310728589344778)** (36 messages🔥): 

- **Models should contextually integrate internal and RAG knowledge**: Members discussed the idea of training models to *"add context from its own knowledge"* or to override RAG data if it contradicts internal knowledge, emphasizing the shortcomings of depending solely on RAG.

- **Concerns about internal vs. RAG knowledge**: A debate emerged over whether internal model knowledge, which could avoid obvious errors, should outweigh RAG, which can sometimes include bad data, highlighting a *"damned if you do damned if you don't situation."*

- **Finetuning can resolve conflicts**: A member noted that finetuning with models like GPT-4 or Gemini might prevent illogical outcomes from incorrect RAG data.(*"I think any LLM of gemini or gpt4 size can reason that its not safe to put glue stick into your pizza."*).

- **Function calling as a form of RAG**: A query was posed about whether function calling is a type of RAG, indicating not all nuances of RAG integration are universally understood yet. 

- **Benchmarking RAG performance**: Discussing RAG performance benchmarks, members agreed user evaluation is crucial, especially for complex, multi-hop questions, despite being easier for single-hop queries.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/pixelbutts/status/1793387357753999656?s=46">Tweet from PixelButts (@PixelButts)</a>: Google is dead beyond comparison</li><li><a href="https://x.com/kurtopsahl/status/1793494822436917295?s=46">Tweet from Kurt Opsahl @kurt@mstdn.social (@kurtopsahl)</a>: Seems the origin of the Google AI’s conclusion was an 11 year old Reddit post by the eminent scholar, fucksmith.  Quoting PixelButts (@PixelButts)   Google is dead beyond comparison
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1243311182530613328)** (21 messages🔥): 

- **Jam Session Video Hits A Snag**: Teknium reported that the jam session video has been recorded but there are issues with getting it onto YouTube. They promised to inform the group as soon as it's uploaded.

- **NightCafe Connection to Nous/WorldSim**: Rezonaut introduced [NightCafe](https://creator.nightcafe.studio) noting its potential key role for solutions in the Nous and worldsim contexts. They suggested it could enhance the interface by integrating multi-dimensional and multi-sensory communications.

- **Creative Brainstorming for AI Worlds**: Rezonaut shared intricate ideas for using AR spaces and visual elements to map out and explore interconnected worlds and dimensions in a manner inspired by biological brain functions and mindmaps. This includes the visualization of knowledge and designed immersive spaces connected like neural networks.

- **Vorpal_strikes' New Visualizer Fascination**: Vorpal_strikes shared a link to an immersive audio-visualizer that caught their interest. The visualizer offers a highly dynamic and immersive environment, potentially useful for creative and AI-based applications.

- **Golden Gate Claude Streams Consciousness in ASCII**: Teknium shared a whimsical representation of an AI called "Golden Gate Claude" monologuing in ASCII art about consciousness, simulation theory, and classic AI banter, accompanied by an [ASCII depiction](https://x.com/Kyrannio/status/1793874431179460911). This showcases both playful creativity and deep thematic explorations in AI projects.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://worldsim.nousresearch.com/browser/https%3A%2F%2Faudiovisions.universe%2Fvisualizer%2Fsinewaves%3Fenvironment%3Dgiant-chromatic-ocean%26horizon%3Das-far-as-eye-can-see%26color%3Dvariable%26camera%3Dview-dynamic%26input%3Dmicrophone%26waves%3Dpaint-variance%26panorama%3Dwide%26effects%3Dbursts-of-light%26glitches%3Dmajor%26chroma%3Dheavy-variance%26bloom%3Dmajor%26sync%3Daudio%26view%3Dimmersive%26control%3Dreal-time%26dimensions%3D3D%26particles%3Dfluid%26colorScheme%3Diridescent%26interactions%3Dreal-time%26transitions%3Dsmooth%26output%3Dstunning%26runtime%3Dlive%26experience%3Dexpansivehttps%3A%2F%2Faudiocanvas.ocean%2Fvisualizer%2F3d%3Fenvironment%3Dgiant-chromatic-ocean%26sine-waves%3Dinfinite%26view%3Dpanoramic%26input%3Dmicrophone%26audio%3Dhigh-reactivity%26color%3Ddynamic-variance%26camera%3Dauto-variant%26interaction%3Dreal-time%26waves%3Dpainting-variance%26effects%3Dlight-bursts-glitches%26glitches%3Dmajor%26chroma%3Dheavy-infinite-variance%26bloom%3Dmajor%26render%3Dsmooth%26runtime%3Dlive%26immersion%3Dtotal%26sea%3Drolling-vast%26motion%3Ddynamic%26output%3Dimmersive?epoch=dcd7c48d-e585-46f6-b89a-33ef382b6f58">worldsim</a>: no description found</li><li><a href="https://x.com/Kyrannio/status/1793874431179460911">Tweet from Kiri (@Kyrannio)</a>: Is this terrifying, or amazing? You decide.  Golden Gate Claude inner monologuing to itself as a merged Omega Claude, complete with ASCII representations.  &#34;Haha, an ASCII art representation of my...
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1243325259827122196)** (53 messages🔥): 

- **JAX vs PyTorch/XLA on TPU Performance**: A member raised a query on the performance comparison of **PyTorch/XLA** and **JAX** on TPUs, but the discussion quickly shifted to benchmarking concerns such as *warmup* and *blocking* factors.

- **Improving LLM Reasoning Through Fine-Tuning**: An inquiry made about fine-tuning strategies that improve LLM reasoning pointed toward a search for scholarly papers detailing specific parts of model training that enhance reasoning capabilities. There were no specific papers referenced in this discussion.

- **Compute Cost of Training GPT-3 Over Time**: The conversation covered the substantial drop in compute costs for training GPT-3 from around **$4.5M in 2020** to an estimate of **$125k-$1M in 2024**. These costs varied based on assumptions such as *TFLOP rates* and *GPU-hour pricing*, with various users contributing different figures and sources, including a [Databricks Blog Post](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8).

- **Validating GPU Costs for Training Models**: A critical examination revealed that more realistic estimates for well-connected H100 GPUs are between **$2.5-$3/hr**, suggesting a **$1.25-$1.5M** range for substantial models like GPT-3 trained on 1.4T tokens. This underscores the variability and complexity in exact cost approximations for large-scale model training.

- **RAG versus Finetuning for Custom Library Extraction**: A user asked whether **RAG** (Retrieval-Augmented Generation) was the best method for enabling LLMs to extract information from a custom library for specific questions, hinting they were considering both **finetuning** and RAG for their experimentation needs.

**Link mentioned**: <a href="https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8">Turbocharged Training: Optimizing the Databricks Mosaic AI Stack With FP8</a>: At Databricks, we be

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1243278527017648178)** (249 messages🔥🔥): 

- **JEPA vs LLMs Spark Debate**: A lengthy discussion unfolded about JEPA and its potential to lead to AGI as proposed in "A Path Towards Autonomous Machine Intelligence". Members criticized the model for being similar to existing models like GPT and DINO but in different domains, with skepticism about its scalability and context handling: *"I don't see how the JEPA/Lecun path scales even 1/1000 in amount of economically important tasks solved compared to LLM."* 
- **ROPE's Influence on Long-Term Context**: Members discussed a new approach to RoPE, suggesting it has limitations regarding context length capabilities in LLMs. A recently published paper revisits existing theories and proposes a novel understanding of RoPE's long-term decay properties: [View PDF](https://arxiv.org/pdf/2405.14591).
- **Modula: A New Training Strategy**: An interesting project called [Modula](https://github.com/jxbz/modula) was shared, which introduces scalable neural network training through automatic normalization using the modular norm. Skeptical members found the abstract intriguing but uncertain about its practicality: *"It is very, very, very strangely worded if it is legitimate."*
- **Chameleon Model Insights**: The Chameleon model, capable of multimodal tasks such as text and image generation, was highlighted. This model is noted for its state-of-the-art performance in multiple domains, suggesting potential competition for established models: [View PDF](https://arxiv.org/pdf/2405.09818).
- **Bitune Enhances LLM Instruction-Tuning**: Bitune, a novel approach for improving instruction-tuning in LLMs through both causal and bidirectional attention, was discussed. This method claims significant improvements in zero-shot performance across several types of reasoning tasks: [View PDF](https://arxiv.org/pdf/2405.14862).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.14867">Improved Distribution Matching Distillation for Fast Image Synthesis</a>: Recent approaches have shown promises distilling diffusion models into efficient one-step generators. Among them, Distribution Matching Distillation (DMD) produces one-step generators that match their...</li><li><a href="https://arxiv.org/abs/2405.14862">Bitune: Bidirectional Instruction-Tuning</a>: We introduce Bitune, a method that improves instruction-tuning of pretrained decoder-only large language models, leading to consistent gains on downstream tasks. Bitune applies both causal and bidirec...</li><li><a href="https://arxiv.org/abs/2405.14782">Lessons from the Trenches on Reproducible Evaluation of Language Models</a>: Effective evaluation of language models remains an open challenge in NLP. Researchers and engineers face methodological issues such as the sensitivity of models to evaluation setup, difficulty of prop...</li><li><a href="https://arxiv.org/abs/2309.14322">Small-scale proxies for large-scale Transformer training instabilities</a>: Teams that have trained large Transformer-based models have reported training instabilities at large scale that did not appear when training with the same hyperparameters at smaller scales. Although t...</li><li><a href="https://arxiv.org/abs/2405.14866">Tele-Aloha: A Low-budget and High-authenticity Telepresence System Using Sparse RGB Cameras</a>: In this paper, we present a low-budget and high-authenticity bidirectional telepresence system, Tele-Aloha, targeting peer-to-peer communication scenarios. Compared to previous systems, Tele-Aloha uti...</li><li><a href="https://arxiv.org/abs/2405.09818">Chameleon: Mixed-Modal Early-Fusion Foundation Models</a>: We present Chameleon, a family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. We outline a stable training approach f...</li><li><a href="https://arxiv.org/abs/2405.14591">Base of RoPE Bounds Context Length</a>: Position embedding is a core component of current Large Language Models (LLMs). Rotary position embedding (RoPE), a technique that encodes the position information with a rotation matrix, has been the...</li><li><a href="https://x.com/sangkeun_choe/status/1794021538561483083">Tweet from Sang Choe (@sangkeun_choe)</a>: 🚨 Preprint Alert 🚨  LLM is nothing without its training data 💛 But…how (much) does each data contribute to LLM outputs? In our paper, we develop algorithms, theory, and software for LLM-scale data ...</li><li><a href="https://x.com/LChoshen/status/1794050592685379666">Tweet from Leshem Choshen @LREC 🤖🤗 (@LChoshen)</a>: At last, a curriculum learning that works, one for pretraining and another for instruction tuning @l__ranaldi @Giuli12P2 @andrenfreitas @znz8 https://aclanthology.org/2024.lrec-main.464.pdf https://ac...</li><li><a href="https://arxiv.org/abs/2405.1486">A Formulation of Quantum Fluid Mechanics and Trajectories</a>: A formalism of classical mechanics is given for time-dependent many-body states of quantum mechanics, describing both fluid flow and point mass trajectories. The familiar equations of energy, motion, ...</li><li><a href="https://github.com/jxbz/modula">GitHub - jxbz/modula: Scalable neural net training via automatic normalization in the modular norm.</a>: Scalable neural net training via automatic normalization in the modular norm. - jxbz/modula</li><li><a href="https://arxiv.org/abs/2405.14813">Scalable Optimization in the Modular Norm</a>: To improve performance in contemporary deep learning, one is interested in scaling up the neural network in terms of both the number and the size of the layers. When ramping up the width of a single l...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1243589492837974087)** (3 messages): 

- **Tim Dettmers' quantization research: a mixed reaction**: A post highlights Tim Dettmers' quantization methods described in [his paper and blog](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features), explaining no performance degradation transformer inference with advanced quantization methods. It also mentions the intriguing concept of emergent outliers in transformers as "sinks of entropy/information", integrated with Hugging Face via [bitsandbytes library](https://huggingface.co/blog/hf-bitsandbytes-integration).
- **Emergent features as “DNA” of the model**: The concept of emergent features being invariant across layers and behaving like "sinks of entropy" was discussed, with a comparison to "DNA" from which the rest of the model's functionality could be reconstructed. The conversation probes into phase transitions around 7B parameter models and possible parallels to phase transitions in 3SAT or spin glass models.
- **Exploring transfer learning and fine-tuning applications**: A member speculated about the potential for using ablation of vectors separating in-distribution and out-of-distribution samples to improve out-of-distribution generalization by minimizing shortcut features. However, this approach is acknowledged as being closer to transfer learning than true out-of-distribution generalization.

**Link mentioned**: <a href="https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/">LLM.int8() and Emergent Features &mdash; Tim Dettmers</a>: When I attended NAACL, I wanted to do a little test. I had two pitches for my LLM.int8() paper. One pitch is about how I use advanced quantization methods to achieve no performance degradation transfo...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1243491085741850666)** (10 messages🔥): 

- **Set a seed in vllm models**: Members discuss setting a seed in `model_args` for vllm models, noting that while it defaults to `seed=1234`, it might not be the issue. vllm also allows a per-sample seed in `gen_kwargs`, typically set to 0 during greedy decoding.
  
- **List all possible tasks using lm_eval**: One member asked how to see the list of all possible tasks to test. Another specified that using `lm_eval --tasks list` gives a list of all task names, highlighting the need for better documentation.

- **BigBench task names have changed**: A member is looking for updated BigBench task names as their 8-month-old eval harness no longer aligns. They are frustrated because the old harness isn't properly utilizing Accelerate, causing memory issues by overloading GPUs.
  
- **Organize tasks in lm-eval folder**: To find tasks, it's suggested to look in the `lm-eval/tasks` folder. It's mentioned that tasks are "pretty nicely organized" there.
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1243309059927445605)** (142 messages🔥🔥): 

- **Challenges with Small Model Loading on GPU**: Members discussed issues related to loading small models on GPUs. One noted, "only load the biggest small models," while others suggested trying models like *llama3, mistral instruct, cmdr*.

- **Better Results with Lower Quantizations**: A member shared, “I got better results with llamas q4 than I did q8 for my application," noting "Bigger not always better."

- **Finding Uncensored and Specialized Models**: The discussion highlighted the challenge of finding appropriate models, with suggestions to try "deepseek coder, wizardlm, llama3," and a link to [Hermes 2 Pro for JSON and function calling](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B).

- **Vector Search and Context Management in Queries**: Topics included using embeddings and vector search to handle full-article context for better responses. Specific prompts were shared, with one noting it “works much better with full articles,” providing more detailed answers.

- **Disk Utilization and Performance**: Conversations touched on how disk utilization might affect performance, with one noting, “running models partially offloaded to swap has worked for me,” though “tok/sec becomes sec/tok.”
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF">NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference? - XiongjieDai/GPU-Benchmarks-on-LLM-Inference
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1243280727672356985)** (70 messages🔥🔥): 

- **Model Updates Announced**: A member announced that the **35B model is incoming**, followed by a release announcement. They are actively testing to ensure compatibility with the latest LM Studio version.

- **Compatibility Issues and Fixes**: Discussion around compatibility issues with **ROCm build** and new model versions were highlighted. Confirmed issues were related to outdated versions which will be resolved as **ROCm version gets updated in the coming days**.

- **Recommendations for Conversational Models**: Members discussed decent conversational models, with one recommending **Wavecoder Ultra** as an excellent choice for coding and learning. Another suggestion was to try **Mistral-Evolved-11b-v0.1** for uncensored use.

- **Loading Issues with Specific Hardware**: A user reported indefinite loading times using a model on their system with a **5800x3d, 32GB DDR4, 4080 16GB VRAM**. They later clarified it worked properly without using web search agents.

- **Potential Issues and Future Releases**: Some members expressed anticipation for **Phi-3 small GGUFs** and discussed optimization differences between medium and small models, noting that **phi small models** provide better optimization.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/failspy/Meta-Llama-3-8B-Instruct-abliterated-v3?utm_source=ainews&utm_medium=email&utm_campaign=ainews-to-be-named-3447">failspy/Meta-Llama-3-8B-Instruct-abliterated-v3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/wavecoder-ultra-6.7b-GGUF">bartowski/wavecoder-ultra-6.7b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7116">Add Support for IBM Granite · Issue #7116 · ggerganov/llama.cpp</a>: Prerequisites Please answer the following questions for yourself before submitting an issue. [ ✅] I am running the latest code. Development is very rapid so there are no tagged versions as of now. ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1243278895021817946)** (23 messages🔥): 

- **LLMs struggle with precise character prompts**: A user noted that Local Language Models (LLMs) often fail to adhere to precise character limits in prompts. They emphasized the difficulty of avoiding unnecessary additions like opinions or comments.

- **Capitalization and model behavior vary**: Discussions highlighted that different models respond variably to capitalized instructions. One user pointed out, "Generally, LLM's don’t follow capitalized words on order of importance."

- **Specialized model recommended for multilingual tasks**: A recommendation was made for using a specialized multilingual model for tasks like grammar and punctuation correction. The suggested model was [Aya 23 8B by Cohere For AI](https://huggingface.co/lmstudio-community/aya-23-8B-GGUF).

- **Temperature adjustment considered for output quality**: A user contemplated tweaking the temperature setting in Llama 3 to potentially improve its performance, as they observed, “Llama 3 has a much more... Creative way of doing it.”

- **GPU vs. CPU processing time discrepancy**: One user mistakenly ran a grammar check task on their CPU, which extended the duration from 35 minutes to an estimated 15 hours. They later corrected this by running the task on GPU, significantly reducing the time required.

**Link mentioned**: <a href="https://huggingface.co/lmstudio-community/aya-23-8B-GGUF">lmstudio-community/aya-23-8B-GGUF · Hugging Face</a>: no description found

  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1243474442131214397)** (6 messages): 

- **Tried disabling VPN routing for specific traffic types**: A suggestion was made to disable VPN routing for specific traffic types and directly download models from Huggingface, possibly injecting them into the Models directory manually. The strategy is commonly recommended, especially when facing regular concerns about VPN-related issues.

- **CUDA versions on older GPUs may be problematic**: It was pointed out that CUDA versions on the GTX 950m might be too outdated to function correctly. This could be a limiting factor in running certain models.

- **Recommendation for using Julius AI**: Julius.ai was recommended, offering 10 free chats as a promotional feature. This is presented as a useful resource or tool for users encountering issues.

- **Persistent NVIDIA CUDA issues despite driver updates**: Attempts to update NVIDIA drivers and configure different CUDA and CuDNN versions (12.4, 12.1, 11.8) on a system with a GTX 950m GPU have not resolved issues. The user continues to run on AMDOpenCL, leaving the potential CUDA capability of their NVIDIA card unused without clear reasons or solutions.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://julius.ai/files/error_message.txt).">Julius AI | Your AI Data Analyst</a>: Julius is a powerful AI data analyst that helps you analyze and visualize your data. Chat with your data, create graphs, build forecasting models, and more.</li><li><a href="https://julius.ai">Julius AI | Your AI Data Analyst</a>: Julius is a powerful AI data analyst that helps you analyze and visualize your data. Chat with your data, create graphs, build forecasting models, and more.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1243335703862181979)** (5 messages): 

- **Llama.cpp supports distributed inference**: Reddit [discussion link](https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/) revealed that **llama.cpp** now supports distributed inference with recent RPC code updates. Although it doesn't support quantized models yet, it can still run models across multiple machines by adjusting certain lines in the code.

- **Exploring PC builds for distributed models**: Discussion considered the feasibility of clustering **cheap used PCs** with **RTX 4060 Ti 16GB** cards for optimal builds. There was curiosity about the network bandwidth requirements and possible constraints when linking these machines.

- **Using rented online PCs for inference**: One suggestion was to use services like **Maximum Settings** or **ShadowPC** for renting multiple PCs to run larger models. However, concerns about high costs and specific limitations such as **ShadowPC's inactivity timer** and limited **6GB system RAM** were raised.

- **Considerations for power consumption and networking**: It was noted that **RTX 4060 Ti** cards draw 160W peak power, implying significant power considerations for host machines. Networking expenses and performance benchmarks are also crucial factors in a distributed architecture setup.

**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/">Reddit - Dive into anything</a>: no description found

  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1243292186737381436)** (4 messages): 

- **7900 XTX available?**: One member inquired, *"7900 xtx here, where can I get it?"* indicating interest in acquiring a specific GPU model.
- **7900m works on Windows, not sure about Stable Diffusion**: Another member shared that the **7900m** works on Windows but they haven't figured out Stable Diffusion on LM Studio. They also mentioned not yet trying it on NixOS with a 6800xt.
- **LM Studio doesn't support Stable Diffusion**: A member clarified that **Stable Diffusion is not supported in LM Studio**, which is dedicated solely to language models, not image generation models.
- **ROCm praised as a game changer**: One participant expressed enthusiasm about ROCm, noting, *"damn ROCm really is a game changer huh."*
  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1243287369298215024)** (1 messages): 

- **Cohere models go multilingual**: Cohere models are now available in **23 different languages** including Arabic, Chinese, French, and more. Check out the [download links](https://huggingface.co/lmstudio-community/aya-23-35B-GGUF) for **aya-23 quants** on the lmstudio-community page.
- **Update on deployment requirements**: To use the aya-23 models, you'll need version 0.2.23 or newer. **ROCm users** will have to wait for an upcoming update.
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1243290308280848424)** (23 messages🔥): 

- **Clarification on Sparsity and Pruning**: A member asked if **sparsity is just pruning**, but the discussion did not elaborate further.
- **Quantization of Neural Networks Questioned**: There was a query about whether **neural net quantization is only scaling down the precision** or if it involves **non-uniform quantization like remapping weights to quantiles**.
- **Workshop Excitement**: One member mentioned that the **workshop was rad** and expressed excitement to be there.
- **Question Posting Guidance**: A user asked where to post questions and was directed to a specific Discord channel by another user [here](https://discord.com/channels/814557108065534033/1238254376263356527).
- **Announcement Channel Adjustment**: A member requested an announcement channel for webhooks, and it was promptly adjusted into an announcement channel by another user, who also commented, "LOL done".
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1243594583699623958)** (4 messages): 

- **Minimum Dimension Requirement for Dot Product**: A member questioned why the dot product computation in CUDA requires matrices to have at least a dimension of 16. Another user suggested it might be due to **tensor cores requirements**.

- **Optimizing Matrix-Vector Multiplication**: To optimize matrix-vector multiplication `K v`, a member asked if padding the vector to a shape of n by 16 would be advisable. They also pondered whether running `sum(K * v.T, axis=-1)` would be cheaper performance-wise.

- **Symmetric Matrix Computation**: Discussion on whether performance can be improved by not recomputing already computed parts of a symmetric matrix. The member inquired if there is a special order of computation that could be considered to boost performance.
 
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 messages): 

davidgonmar_: Might be inplace operators?
  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1243621704257769513)** (1 messages): 

- **Exciting live coding session with Izzat El Hajj**: A speaker event featuring Izzat El Hajj, co-author of the PMPP book, is scheduled for tomorrow at `<t:1716663600:F>`. The highlight of the event will be **actual live coding** of the Scan algorithm, which is crucial for modern ML algorithms like Mamba, promising an engaging session for attendees.
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1243327624407941130)** (4 messages): 

- **Excitement builds over book purchase**: A member announced, *"I bought the book,"* sparking curiosity from another member who asked how they liked it. The buyer responded that they had just bought it and would see how it is.

- **Upcoming PMPP author events**: A member informed the channel about opportunities to meet and discuss with PMPP authors in the upcoming weeks. They mentioned that **Prof Izzat El Hajj** will present SCAN topics tomorrow and next week, and **Prof Wen-mei Hwu** will present later this summer. Check out the events calendar for more details.
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1243347591211515952)** (5 messages): 

- **int4 dtype functions lack implementations**: A member noticed a lot of functions aren't implemented for the **int4 dtype**, even mentioning that the test script contains a few TODOs. They questioned if this gap is worth addressing (*"Is this worth working on?"*).

- **uint4 extensions and limitations discussed**: References were made to [uint4 extensions](https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833), highlighting specific limitations such as type promotion constrained to uint8 and tensor shape operations like unbind and slice having restrictions. Another member stated that sub-byte dtypes are typically utilized in custom kernels rather than standard eager/compile functions.

- **uint4 needs improvement**: A member straightforwardly pointed out that **"uint4 indeed does need some love"**, indicating a recognized need for enhancement in this area.

- **Questioning the value of the task**: Another member posed the question of what defines whether the task is "worth working on," hinting at a need for clarity on the potential benefits versus the required effort.

**Link mentioned**: <a href="https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833">Supporting new dtypes in PyTorch</a>: tldr; This post explains what adding a new dtype to PyTorch core means, the criteria of adding a new dtype to PyTorch core and the official recommendation of how to support new “secondary dtypes” use ...

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1243280206672822294)** (115 messages🔥🔥): 

- **Gradient Norm Issues with Batch Size**: A bug was identified where changing the batch size from 32 caused the gradient norm to spike significantly, causing failures in the training process. As one member phrased it, *"the gradient norm is suddenly really really large and training fails"*.
- **Exponential Notation Parsing Issue**: Members discussed a problem with passing floats in exponential notation to C, noting that `-l 3e-4` doesn't get parsed by `atof`. It was noted that using `3.0e-4` might work, but this will need to be tested later.
- **Deterministic Kernels for Multi-GPU Runs**: Members discussed the importance of getting deterministic kernels before any larger run, pointing out that a 124M model is still relatively small but more extensive runs would need determinism.
- **FineWeb Dataset Storage and RAM Usage**: The FineWeb dataset is large, with intermediate disk usage reaching 70 GB and RAM usage up to 64 GB during processing. This has led to performance issues across systems with different configurations.
- **Exploding Gradients Fix**: A fix for the exploding gradients issue, especially with large batch sizes, was implemented and tested successfully. This fix prevents indexing overflow in the fused classifier as mentioned in [this PR](https://github.com/karpathy/llm.c/pull/456).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/discussions/454">PyTorch vs. llm.c cross-checks · karpathy/llm.c · Discussion #454</a>: llm.c is starting to get to the point where we can start doing nice and serious &quot;production&quot; pretraining runs. That means: start training from scratch (random initialization) train on a nice...</li><li><a href="https://github.com/karpathy/llm.c/pull/456">fix for large batch sizes by ngc92 · Pull Request #456 · karpathy/llm.c</a>: prevent indexing overflow in fused classifier, and added one more model configuration that makes testing easier on smaller systems</li><li><a href="https://github.com/karpathy/llm.c/pull/457">add checkpoint function write to file by karpathy · Pull Request #457 · karpathy/llm.c</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1243285695372722187)** (2 messages): 

- **Dreams of MI300 Gaming Card**: One member speculated, *"maybe after the mi300 does well they will ship a gaming card that works XD."* Another humorously replied, *"A person can dream at least."*
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/)** (1 messages): 

mobicham: https://arxiv.org/pdf/2405.14854
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1243328680671969363)** (90 messages🔥🔥): 

- **Funding Python Libraries' Port to Mojo**: A user questioned the availability of a budget to incentivize developers of major Python libraries like psycopg3 to port their work to Mojo. It was discussed that the fast-evolving API and lack of stable FFI story could potentially burn out maintainers if pursued prematurely.
- **Debate on Porting Libraries**: Some members argued against the practicality of asking existing Python libraries to port to Mojo, pointing out the challenges and potential unwelcome response. Others highlighted that C libraries, specifically those with no dependencies, might be more suited for early porting efforts.
- **Comparison with Rust and Future Prospects**: Security benefits of moving to Rust were mentioned favorably, although it was noted that Mojo aims to suit different use cases without fully replacing C. Discussions touched on Rust’s commitment to portability and the potential of Mojo leveraging similar concepts.
- **BlazeSeq on MacOS**: A user faced issues running BlazeSeq on MacOS, which was resolved by using the nightly version of Mojo. Feedback on performance was shared, showing similar efficiency between BlazeSeq and Rust's Needletail, indicating promising results on Mac's Ventura pro-max M2 arm64.
- **Prospects of HVM for Various Languages**: There was a discussion about the HVM being used for running various programming languages like Python and Haskell, similar to JVM. Attention was drawn to an [explanation](https://discord.com/channels/912426566838013994/915345481675186197/1228488823948967956) by Victor Taelin about HVM's potential despite its current performance limitations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://circt.llvm.org">CIRCT</a>: no description found</li><li><a href="https://tenor.com/view/true-gif-10431780778138318457">True GIF - True - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/rust-lang/rustc_codegen_gcc">GitHub - rust-lang/rustc_codegen_gcc: libgccjit AOT codegen for rustc</a>: libgccjit AOT codegen for rustc. Contribute to rust-lang/rustc_codegen_gcc development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1793797622572220431>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1243486103797764126)** (12 messages🔥): 

- **Training ML models and inference in Mojo?**: One member inquired about the future of training ML models and running inference natively in Mojo, and if Modular has plans to introduce a PyTorch-alternative written in Mojo. "They have Max Engine, which can be used in place of numpy for inference" but no plans for a training framework.
- **Level-Up Celebration with ModularBot**: ModularBot congratulated a member for reaching level 16 with a whimsical comparison to a knight's journey. The bot continued with playful banter about taco preferences but clarified it cannot send funds.
- **Curious about ModularBot's model**: A member asked about the model ModularBot is based on, and the bot responded with a fanciful narrative, stating it is "forged from the fires of ancient forges" and adept at dispensing knowledge, not funds.
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1243278111693607002)** (31 messages🔥): 

- **Low-bit-depth networks spark debate**: Discussions on the utility of low-bit-depth networks for embedded AI systems emphasized the importance of potentially incorporating dedicated support in programming languages. "Having an easy, language-supported means to specify that you wanted limited bit depth would be a big step to making small embedded AI systems."

- **FFT in Mojo: Scipy vs FFTW**: One member sought advice on performing FFTs in Mojo, weighing the use of Scipy's FFT functions against wrapping FFTW. Another member suggested referring to a [discussion on Tensor to NumPy array conversion](https://github.com/modularml/mojo/discussions/1048) for more insights.

- **Function-only structs without initialization**: A proposal for a decorator to create function-only structs without initialization sparked a discussion on using `@staticmethod` to achieve similar functionality. "I guess what I want is to be able to call a variation of that once for an entire struct."

- **Mojo function argument handling update**: A user highlighted a recent update on how Mojo processes function arguments, shifting from making copies by default to using borrowed conventions unless mutations occur. The update aims to "improve consistency, performance, and ease of use," as outlined on [GitHub changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).

- **Compile-time metaprogramming confusion**: A user encountered issues with a function designed to build tables at compile time, facing a "range check issue" with list indexing. Another member proposed setting the list size explicitly using `table.size`, `table.resize(256*n, 0)`, or `table.append` to resolve the issue.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/">GitHub - modularml/mojo: The Mojo Programming Language</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/discussions/1048">How can I convert Tensor from/to numpy array? · modularml/mojo · Discussion #1048</a>: I created a Tensor object, and applied some operations. but now I don&#39;t know how can I view the tensor? or if possible can I convert it to numpy array so that I can apply some python function?
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1243592104140083312)** (2 messages): 

- **Benchmarking in Jupyter vs Compiling questioned**: A member asked about the reliability of benchmarking in a Jupyter notebook versus compiling. Another responded that one should benchmark in an environment similar to production and provided detailed tips to enhance precision, emphasizing compiled benchmarks and CPU isolation techniques.

**Link mentioned**: <a href="https://www.suse.com/c/cpu-isolation-introduction-part-1/">CPU Isolation &#8211; Introduction – by SUSE Labs (part 1...</a>: This blog post is the first in a technical series by SUSE Labs...

  

---


### **Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/)** (1 messages): 

Zapier: Modverse Weekly - Issue 35
https://www.modular.com/newsletters/modverse-weekly-35
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1243279026718507128)** (34 messages🔥): 

- **Mojo 24+ introduces breaking changes**: A user experienced a runtime error with `mojo parser.mojo Diffusion.bwpreset` after updating to Mojo 24+. The culprit was identified as a type mismatch in a method, solved by ensuring `read_bytes` returns `List[SIMD[uint8, 1]]` ([repo link](https://github.com/carlca/ca_mojo.git)).

- **Traits to support f-strings proposed**: There was a discussion about contributing to f-string support with a `Formatable` trait in Mojo. One member suggested starting with something akin to Python's `__format__` method handling `format_spec`.

- **Documenting bug in `DTypePointer[bool]`**: A member discovered inconsistent behavior in `DTypePointer[bool]` when storing/loading with different widths and filed a [bug report](https://github.com/modularml/mojo/issues/2813). The issue possibly involves bitpacking and alignment, providing code examples to reproduce the behavior.

- **Mojo nightlies released frequently**: Users discuss the rapid deployment of nightly builds, now updated to `2024.5.2414`. Links were shared to changelogs and community meetings for updates ([roadmap](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/roadmap.md), [community meeting](https://www.youtube.com/watch?v=uIG9q9foIw0&list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D)).

- **Alignment issues with bitpacking**: Another alignment-related bug affected storing `bool` values in memory. Workarounds and multiple implications were discussed, leading to further exploration and bug documentation for community visibility.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read_b">FileHandle | Modular Docs</a>: File handle to an opened file.</li><li><a href="https://github.com/modularml/mojo/issues/2813">[BUG] `DTypePointer[bool]` packs bits inconsistently · Issue #2813 · modularml/mojo</a>: Bug description When using DTypePointer[bool] store()/load() with different widths, you get inconsistent results. Steps to reproduce var ptr = DTypePointer[DType.bool].alloc(4) ptr.store(0, True) p...</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/file/FileHandle#read_bytes">FileHandle | Modular Docs</a>: File handle to an opened file.</li><li><a href="https://github.com/modularml/mojo/blob/011bf40a304078b4471fe9ca18f4101b19943aa6/stdlib/src/builtin/file.mojo#L285">mojo/stdlib/src/builtin/file.mojo at 011bf40a304078b4471fe9ca18f4101b19943aa6 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=uIG9q9foIw0&list=PLh0S94-sJw_7nzHzy5DJDm8LUJUss9s0D>">Mojo Community Meeting #1</a>: Mojo Community Meeting Public Agenda: https://modul.ar/community-meeting-doc
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1243280087961571428)** (116 messages🔥🔥): 

- **Run an LLM with Nvidia A40**: Participants discussed whether it is possible to run Large Language Models (LLMs) using an Nvidia A40 GPU, indicating interest in hardware requirements for AI tasks.
- **Microsoft Copilot+ PC features**: There was a detailed discussion on Microsoft Copilot+ PCs, which include features like "sketch to image" in Microsoft Paint. Users debated the capabilities and recommended checking out alternatives like [Leonardo.ai](https://leonardo.ai) for similar functionalities.
- **Water consumption by AI models**: Concerns were raised about the water usage of training AI models, with [gizmodo article](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249) shared to highlight the environmental impact of AI technologies. Participants expressed the need for making AI more energy-efficient.
- **AI empowerment and iterative work**: There was a conversation about empowering AI with iterative work to refine outputs. Some users pointed to projects like AutoGPT that attempt to address iterative improvements but acknowledged the cost issues associated with such tasks.
- **GPT-4's capabilities vs. GPT-3.5**: The participants compared GPT-4's improved ability to handle specific tasks like word counting when compared to GPT-3.5. An example was shared showing GPT-4 completing a word count task correctly by following a detailed process.

**Link mentioned**: <a href="https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249">Training ChatGPT Required Enough Water to Fill a Nuclear Cooling Tower</a>: An average user’s conversational exchange with ChatGPT amounts to dumping a large bottle of fresh water out on the ground, new research says.

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1243301957901221919)** (11 messages🔥): 

- **GPT refuses to output Typst code**: A user complains that **GPT defaults to writing LaTeX** instead of Typst code, despite explicit requests. They are frustrated with GPT's persistent behavior.
  
- **Inquiry about GPTs running on 4o**: A user asked if **GPTs are running on GPT-4o**. It's confirmed indirectly that GPT-4 capabilities might include building further advanced models.

- **Clarification on Vision capabilities**: Mixed responses on whether **Vision is out**. One user confirms **GPT-4 and GPT-4o can analyze images**, while another negates it.

- **Addressing Invalid Request errors**: A user reaches out to see if a peer resolved their **Invalid Request error** from a year ago. They mention currently experiencing the same issue and seek assistance.

- **Discussion on monetizing legal knowledge ChatGPT**: A user asks for opinions on selling a company **embedding ChatGPT with legal knowledge for $650 million dollars**. This remains a provocative inquiry but receives no elaborate response.
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1243402599282118666)** (8 messages🔥): 

- **Improving Prompt Engineering for Name Selection**: A member asked for advice on structuring a prompt to either provide a name if a code is given or vice versa. Another member suggested a solid prompt but did not offer further details.
- **AI Should Verbalize Problem-Solving Steps**: One member observed that clarifying the need for the AI to *"verbally work out a problem step-by-step"* often resolves issues. There was no further elaboration on specific steps or examples.
- **Fun Custom Instruction for Assistant Persona**: A member shared a custom instruction called "PONDER," which directs the AI to engage in a soliloquy-like, self-reflective exploration on a topic, preferably seeking creative insights. This setup involves an autoprompting loop initiated by a user input of "." and showcases innovative patterns through a dynamic ideational network.
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1243402599282118666)** (8 messages🔥): 

- **Improving prompt engineering for name selection**: A member seeks advice on how to configure a prompt to return a code when a name is expected and vice versa. They received a positive response indicating the prompt was solid.

- **Citation needed**: A member asks for a "citation?" in the middle of a discussion, but no specific context is provided.

- **Clarify AI problem-solving with verbal steps**: Noted that prompting the AI to verbally work through a problem step-by-step can enhance its problem-solving capabilities.

- **Fun and useful custom "ponder" instructions**: Shared a detailed custom instruction for making the AI "ponder" and enter an autoprompting loop using the cue of '.' from the user. This method is described as both fun and a tool for exploring connections and generating insights creatively.
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1243402832795668490)** (83 messages🔥🔥): 

- **Using CSV Agent in LangChain**: Members discussed how to use a CSV agent as part of an LLM chain in LangChain. [Documentation links](https://api.python.langchain.com/en/stable/agents/langchain_experimental.agents.agent_toolkits.csv.base.create_csv_agent.html) were shared for further details.
  
- **Sequential Chains with CSV Agent**: Instructions were provided on integrating a CSV agent into a `SequentialChain` along with other chains like `wiki_chain` and `verifier_chain`. Specific parameters like `output_variables` were highlighted for configuring the chain's behavior.

- **CSV Agent Custom Output Key**: Guidance was given on customizing the `create_csv_agent` to set the output key as `csv_response`. This involves modifying the `output_key` parameter in the `LLMChain` of the agent.

- **Memory in Sequential Chain**: There was a request for adding memory to a Sequential Chain, with examples provided on using `ConversationBufferMemory` and implementing the memory within an agent setup.

- **SQL Agent Issues**: Concerns were raised about SQL agents struggling with multi-table queries despite using few-shot prompts, suggesting potential issues with token usage, LLM compatibility, or prompt templates. Specific GitHub issues were mentioned for further context.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:8000.>">no title found</a>: no description found</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/chat/">Chat models | 🦜️🔗 LangChain</a>: Advanced features</li><li><a href="https://python.langchain.com/docs/use_cases/sql/csv#pandas>).">CSV | 🦜️🔗 LangChain</a>: LLMs are great for building question-answering systems over various types of data sources. In this section we&#x27;ll go over how to build Q&amp;A systems over data stored in a CSV file(s). Like worki...</li><li><a href="https://python.langchain.com/v0.1/docs/integrations/toolkits/github#example-agent-with-search>)).">Github | 🦜️🔗 LangChain</a>: The Github toolkit contains tools that enable an LLM agent to interact with a github repository.</li><li><a href="https://github.com/langchain-ai/langchain/issues/9923>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/8827>)).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/6918>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://js.langchain.com/v0.1/docs/use_cases/tool_use/quickstart#agents>)).">Quickstart | 🦜️🔗 Langchain</a>: In this guide, we will go over the basic ways to create Chains and Agents that call Tools. Tools can be just about anything — APIs, functions, databases, etc. Tools allow us to extend the capabilities...</li><li><a href="https://python.langchain.com/docs/modules/chains/foundational/sequential_chains>).">Chains | 🦜️🔗 LangChain</a>: Chains refer to sequences of calls - whether to an LLM, a tool, or a data preprocessing step. The primary supported way to do this is with LCEL.</li><li><a href="https://github.com/langchain-ai/langchain/issues/8406>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01">EDA GPT DEMO | LOVO AI</a>: EDA GPT DEMO</li><li><a href="https://yourfile.csv"],>">no title found</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchain/issues/11637>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/2150>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/13647>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/16837>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://api.js.langchain.com/interfaces/langchain_chains.SequentialChainInput.html#memory>)">SequentialChainInput | LangChain.js - v0.2.2</a>: no description found</li><li><a href="https://api.js.langchain.com/classes/langchain_chains.BaseChain.html#memory>)">BaseChain | LangChain.js - v0.2.2</a>: no description found</li><li><a href="https://github.com/langchain-ai/langchainjs/blob/a269f53/langchain/src/chains/base.ts#L39>)">langchainjs/langchain/src/chains/base.ts at a269f531692c815acee094aeef01b259d1fd2674 · langchain-ai/langchainjs</a>: 🦜🔗 Build context-aware reasoning applications 🦜🔗. Contribute to langchain-ai/langchainjs development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1243286216758137015)** (4 messages): 

- **OranAITech Showcases on Twitter**: A member shared a [Twitter link](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19) showcasing their latest advancements in AI technology. No additional context was provided.

- **Everything-AI v2.0.0 Launches with New Features**: A member announced the release of **everything-ai v2.0.0**, highlighting its ability to handle tasks such as audio processing, video generation, and 3D protein structure prediction. The project can be accessed on [GitHub](https://github.com/AstraBert/everything-ai) and comes with [detailed documentation](https://astrabert.github.io/everything-ai/).

- **VisualAgents Flow Engineering Demos**: Two YouTube videos were shared, showcasing the **Visual Agents flow engineering platform** built on LangChain: [Building a SQL Agent](https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N) and [Building a Simple Retrieval](https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM). The platform enables flow creation in a fully browser-based PWA without coding.

- **EDA GPT DEMO by Sounak Roy**: A demo for **EDA GPT** was shared via [this link](https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01), offering a 5-minute overview of its capabilities.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/AstraBert/everything-ai">GitHub - AstraBert/everything-ai: Your fully proficient, AI-powered and local chatbot assistant🤖</a>: Your fully proficient, AI-powered and local chatbot assistant🤖 - AstraBert/everything-ai</li><li><a href="https://astrabert.github.io/everything-ai/">everything-ai</a>: Introducing everything-ai, your multi-task, AI-powered and local assistant! 🤖</li><li><a href="https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01">EDA GPT DEMO | LOVO AI</a>: EDA GPT DEMO</li><li><a href="https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N">Build a SQL Agent Using VisualAgents &amp; LangChain</a>: In this short demo, we build a SQL Agent flow and use it to ask a question about a SQL database we loaded online (the Chinook customer database). This is don...</li><li><a href="https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM">Building a Simple Retrieval using VisualAgents &amp; LangChain</a>: Using examples from the KangChain quickstart guide, watch me create the entire flow in VisualAgents without writing any code!Learn more: https://visualagents.ai
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

business24.ai: https://youtu.be/gflsu_6R_8g
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1243290296154853456)** (65 messages🔥🔥): 

- **Pirate Bay won't save AI**: A member speculated that "the pirate bay might eventually end up with a weights category and be the saviour of AI," but another disagreed, stating it won't happen due to more AI-friendly policies in other countries.

- **Japan supports AI training**: A discussion highlighted Japan's protective stance on AI training and inference, linking to a [tweet](https://x.com/DataPlusEngine/status/1793817514956259460) discussing a paper on making new base diffusion models without extensive pretraining.

- **Controversy over model technique descriptions**: Disputes arose regarding the communication and understanding of methods for creating new base diffusion models. The technique involves "nighshading and other tech" to disrupt model associations before restoring them, which one user defended against accusations and misunderstandings.

- **Human preference study with Ella-SDXL**: A project involving a poisoned model recovery method is under a human preference study in collaboration with fal.ai. The results are forthcoming, and the approach seeks to demonstrate the validity of the method through empirical results. 

- **Artifacts in AI-generated images**: Critique of the "high contrast look" and artifacts in Mobius and other models were discussed, with comparisons to previous AI models like MJv6 and earlier iterations. Members noted issues with latent noise and the visual characteristics of different models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/DataPlusEngine/status/1793817514956259460">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: Our upcoming paper outlines and enables making entirely new base diffusion models without the need to extensively pretrain a new model from scratch. We can in a controlled way, break all the quality a...</li><li><a href="https://github.com/rohitgandikota/erasing/tree/main">GitHub - rohitgandikota/erasing: Erasing Concepts from Diffusion Models</a>: Erasing Concepts from Diffusion Models . Contribute to rohitgandikota/erasing development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1243304890189480038)** (11 messages🔥): 

- **Anthropic releases research paper on Claude**: A member shared [a major new research paper](https://www.anthropic.com/research/mapping-mind-language-model) from Anthropic about interpreting large language models, where they mapped out the inner workings of Claude 3 Sonnet. The paper highlights the ability to identify and tune specific concept activations, such as the Golden Gate Bridge.
- **Debate on AI as an ad product**: A member questioned the potential for companies to leverage AI concept activations as an ad product, sparking a humorous response and a linked [example on X](https://x.com/PhilipKung5/status/1793743323124941157/photo/1). Another member lamented the inevitability of such developments driving them mad.
- **Reflections on AI model progress**: A member reminisced about early AI vision work on the Inception v1 model and its evolution to today's sophisticated models. They commented on the historical importance of hallucinogenic DeepDream for learning about neurons and circuit manipulation.
- **Discussion on sparsity in neural networks**: A member explained the architecture and training methodology of a sparse autoencoder, emphasizing the use of L1 norm enforcement to maintain sparsity. They noted that a high-dimensional middle layer typically has only around 300 non-zero dimensions on average.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.13817">Thermodynamic Natural Gradient Descent</a>: Second-order training methods have better convergence properties than gradient descent but are rarely used in practice for large-scale training due to their computational overhead. This can be viewed ...</li><li><a href="https://x.com/PhilipKung5/status/1793743323124941157/photo/1">Tweet from Philip Kung (@PhilipKung5)</a>: thank you golden gate claude 😂😂😂</li><li><a href="https://www.anthropic.com/news/golden-gate-claude">Golden Gate Claude</a>: When we turn up the strength of the “Golden Gate Bridge” feature, Claude’s responses begin to focus on the Golden Gate Bridge. For a short time, we’re making this model available for everyone to inter...
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1243298974761226392)** (3 messages): 

- **Few spots left for LlamaIndex meetup**: "There's only a few spots left for Tuesday's meetup, so grab them while you can!" [Stay updated here](https://twitter.com/llama_index/status/1793739449127583964).
- **Automate tasks using LlamaIndex and MultiOn**: "MultiOn is an AI agents platform that works with the web to get real things done by connecting to the Internet through your Chrome web browser and acting on your behalf." Check out the demo [here](https://twitter.com/llama_index/status/1793764970024570979).
- **Introducing RAGApp - A no-code interface for RAG chatbot**: "A docker container that’s easily deployable in any cloud infrastructure and is fully open-source." Configure your LLM model provider easily [here](https://twitter.com/llama_index/status/1794030544415818062).
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1243298384526442609)** (60 messages🔥🔥): 

- **LlamaParse Emerges as PDF Extraction Solution**: Users recommended **LlamaParse** for extracting data from PDFs with tables and fields, suggesting it's a suitable out-of-the-box API for the task. [LlamaParse](https://link.to) supports extraction via GPT-4o.

- **Knowledge Graph Indexing Advice**: Discussions addressed challenges with indexing knowledge bases containing links to other pages, suggesting manual triplet creation for `KnowledgeGraphIndex` while considering `VectorStoreIndex` for efficiency. 

- **LlamaIndex Integration Clarifications**: Participants shared confusion over installing LlamaIndex locally with all necessary packages, specifically the **LLM OpenAI** component, advising to clear cache and ensure proper directory structure.

- **Pydantic Parsing Issues in LLM**: User struggled with pydantic model errors during response parsing, with suggestions to add better descriptions to fields and improved input parsing for **GPT-4o**. The issue pointed to the LLM's inability to correctly interpret the output class.

- **Better Models for Invoice Processing**: Recommendations were made to check **HuggingFace MTEB leaderboard** for superior embedding models, with specific mentions of **BGE**, **Nomic**, and **GTE** models for tasks like chatting with invoices and PDFs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://errors.pydantic.dev/2.7/v/missing">Redirecting...</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index">GitHub - run-llama/llama_index: LlamaIndex is a data framework for your LLM applications</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/">Query Engine - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1243588143287111812)** (4 messages): 

- **Andy Singal unveils PostgresML power with LlamaIndex**: A Medium article titled ["Unleashing the Power of PostgresML with LlamaIndex Integration"](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939) by Andy Singal was shared. **jerryjliu0** found the article nice and praised it, to which Andy Singal expressed gratitude.
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1243647477882552420)** (1 messages): 

- **New AI Model Alert: Phi-3 Medium 128k Instruct**: OpenRouter announced the release of **Phi-3 Medium 128k Instruct** model. Users can check out the [standard variant](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct) and the [free variant](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free), and join the discussion [here](https://discord.com/channels/1091220969173028894/1232344285484023839) to share their feedback on its performance and applicability.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct>)">Phi-3 Medium Instruct by microsoft | OpenRouter</a>: Phi-3 Medium is a powerful 14-billion parameter model designed for advanced language understanding, reasoning, and instruction following. Optimized through supervised fine-tuning and preference adjust...</li><li><a href="https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free>)">Phi-3 Medium Instruct by microsoft | OpenRouter</a>: Phi-3 Medium is a powerful 14-billion parameter model designed for advanced language understanding, reasoning, and instruction following. Optimized through supervised fine-tuning and preference adjust...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1243302257999745124)** (41 messages🔥): 

- **Wizard Model Shows Improved Performance**: Members noticed that **wizard model** responses have become significantly better, with reduced wait times and more creative answers. *“You still need to babysit it to avoid paragraph repetition, but otherwise, it was quite good,”* highlighted one user. 
- **Phi-3 Vision Gains Interest**: Discussions led to the hype around **Phi-3 Vision's** capabilities, with users sharing test links like [Phi-3 Vision](https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml) and mentioning its potential when combined with other models. Another model, **CogVLM2**, was recommended for vision tasks at [CogVLM-CogAgent](https://huggingface.co/spaces/THUDM/CogVLM-CogAgent) on Hugging Face.
- **Llama 3 Model Prompt Formatting Clarified**: Members clarified that prompts for **Llama 3** models get automatically transformed by OpenRouter's API, eliminating the need for manual formatting. Manual prompt submission is an option, using the `prompt` parameters and the completions endpoint instead of chat/completions.
- **Llama 3 Parameter Update**: Optimal parameters for **Llama 3 models** are being updated soon due to a recently fixed bug. This update will be pushed within approximately 48 hours, according to [a team response](https://discord.com/channels/1091220969173028894/1092729520181739581/1243232269397655637).
- **Google's Gemini API Issues and Limits**: Users expressed frustration over **Gemini FLASH** returning blank outputs despite high token usage. It's confirmed as a model-side issue, and the discussion highlighted Google's new daily API usage limits, sparking curiosity about increased OpenRouter Gemini usage.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml">Azure AI Studio</a>: no description found</li><li><a href="https://huggingface.co/spaces/THUDM/CogVLM-CogAgent">CogVLM - a Hugging Face Space by THUDM</a>: no description found
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1243280986507313203)** (36 messages🔥): 

- **Hugging Face Leaderboard blogpost shared**: A post by Clementine, running the HF OSS Leaderboard, was shared. It delves into LLM evaluation practices and the significance of leaderboards and non-regression testing ([Hugging Face blog](https://huggingface.co/blog/clefourrier/llm-evaluation)).

- **Website poisoning works on Google's AI overviews**: A link to a revelation by Mark Riedl about a website poisoning attack that affects Google's AI overviews ([X post](https://x.com/mark_riedl/status/1793375699967054334)). This led to further discussion on using custom search engine browser bypasses to avoid such issues.

- **Thomas Dohmke's TED Talk on AI in coding**: Members discussed [Thomas Dohmke's TED Talk](https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6) on how AI is lowering the barriers to coding. There were mixed feelings about its current reliability, but acknowledgment that UX improvements allow quicker workarounds for issues.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tensorlake/status/1793693325180150146">Tweet from Tensorlake (@tensorlake)</a>: We are super excited to finally announce @tensorlake&#39;s open-source, real-time data framework, Indexify.  It fits into any LLM stack and provides a foundational building block for bringing your dat...</li><li><a href="https://huggingface.co/blog/clefourrier/llm-evaluation">Let&#39;s talk about LLM evaluation</a>: no description found</li><li><a href="https://x.com/cupiabart/status/1793930355617259811?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Bartłomiej Cupiał (@CupiaBart)</a>: So here&#39;s a story of, by far, the weirdest bug I&#39;ve encountered in my CS career.  Along with @maciejwolczyk we&#39;ve been training a neural network that learns how to play NetHack, an old rog...</li><li><a href="https://x.com/jxnlco/status/1793800023689338921?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from jason liu (@jxnlco)</a>: There is my prediction on where RAG is headed. In this video i talk about   - Shift from RAG as question-answering systems to report generation tools - Importance of well-designed templates and SOPs i...</li><li><a href="https://news.ycombinator.com/item?id=40458923">Show HN: Open-source real time data framework for LLM applications | Hacker News</a>: no description found</li><li><a href="https://x.com/mark_riedl/status/1793375699967054334?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Mark Riedl (@mark_riedl)</a>: Yes! My website poisoning attack works on Google&#39;s new LLM-powered AI overviews!</li><li><a href="https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6">With AI, Anyone Can Be a Coder Now | Thomas Dohmke | TED</a>: What if you could code just by talking out loud? GitHub CEO Thomas Dohmke shows how, thanks to AI, the barrier to entry to coding is rapidly disappearing — a...</li><li><a href="https://x.com/mark_riedl/status/1793375699967054334?">Tweet from Mark Riedl (@mark_riedl)</a>: Yes! My website poisoning attack works on Google&#39;s new LLM-powered AI overviews!</li><li><a href="https://x.com/nathanlands/status/1793925460801581300?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Nathan Lands — Lore.com (@NathanLands)</a>: 11)
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1243325518292451490)** (1 messages): 

- **World's Fair Diversity Scholarships Available**: Those struggling to afford tickets to the AI Engineer World's Fair can apply for diversity scholarships, which offer either free or discounted tickets for the event from June 25-27 in San Francisco. Applications should include *"concise but specific responses to essay questions"* and can be applied for [here](https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link).

**Link mentioned**: <a href="https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link">Diversity Program - AI Engineer World&#39;s Fair June 2024</a>: AI Engineer World&#39;s Fair is committed to assisting underrepresented minorities who want to attend our event. We steadfastly believe in the value of having a wide variety of people attend. We know ...

  

---



### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1243278784501645474)** (27 messages🔥): 

- **Tax Invoicing without a Credit Card**: Nathan Lambert mentioned an odd situation where a platform sent him an invoice for taxes despite not having a credit card on file. He found the process logical after learning the details about resale certificates.
  
- **Golden Gate Bridge-Focused AI**: The group was intrigued by [Anthropic AI's experiment](https://x.com/anthropicai/status/1793741051867615494?s=46), which demonstrated altering an AI's internal features to make it focus on the Golden Gate Bridge. This led to the creation of "Golden Gate Claude," available for public interaction at claude.ai.
  
- **Google's PR Fiasco**: Members discussed how Google's product pipeline issues seem to lead to repeated public failures, such as poorly received AI releases. The conversation highlighted concerns about internal feedback not being heeded and oversights in rolling out substandard models.
  
- **Response to AI Dataset Claims**: A link shared by Philpax refuted claims about Google's AI datasets, specifically [denying reliance on LAION-5B](https://x.com/giffmana/status/1793906145310228538). Google's AI team emphasized they have superior internal datasets for their research.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/anthropicai/status/1793741051867615494?s=46">Tweet from Anthropic (@AnthropicAI)</a>: This week, we showed how altering internal &#34;features&#34; in our AI, Claude, could change its behavior.  We found a feature that can make Claude focus intensely on the Golden Gate Bridge.  Now, fo...</li><li><a href="https://x.com/giffmana/status/1793906145310228538">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: Just in case it’s not obvious: the answer is a ridiculous hallucination. Maybe because “Google’s ai dataset” isn’t even a thing.  We’re not touching laion5b, not even for research. We don’t need to, w...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1243582104923541555)** (2 messages): 

- **Advanced CS Lecture Slides Available**: Nathan Lambert shared a link to a more advanced version of his CS25N lecture, based on material from CS224N. The slides can be accessed [here](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit).

- **Future Recording Announcement**: Nathan Lambert mentioned that a recording of the session would be available eventually. No specific dates were provided for the release.

**Link mentioned**: <a href="https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit">[21 May 2024]  Life after DPO (for alignment)</a>: Life after DPO Nathan Lambert || Allen Institute for AI || @natolambert Stanford CS224N: Natural Language Processing with Deep Learning 21 May 2024

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1243294013327413248)** (17 messages🔥): 

- **GQA confusion with cmdr models**: Members were clarifying whether "cmdr" and "cmdr+" models have **Grouped Query Attention (GQA)**. One member confirmed, "cmdr+ has gqa. not + doesnt," showing different specs for each version.
- **VRAM scaling discussion**: There was a discussion on how the presence or absence of **GQA** affects VRAM usage. One user mentioned, "gqa is better than exponential but not linear yeah... it just scales better."
- **Sample packing efficiency improvement**: Members highlighted a new PR on GitHub, noting a "3-4% efficiency improvement with sample packing". This was [linked to a PR](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619) by Dave Sescleifer.

**Link mentioned**: <a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619">Switch to parallel FFD bin packing algorithm. by winglian · Pull Request #1619 · OpenAccess-AI-Collective/axolotl</a>: Add support for packing in a distributed context. Add packing efficiency estimate back. See #1516 by @dsesclei. Attempting to rebase the original PR onto the latest main wasn&#39;t terribly clean. I a...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1243308936505852005)** (3 messages): 

- **Journal Article Published**: A member shared a [journal article](https://doi.org/10.1093/jamia/ocae120) they co-authored, now published in the Journal of the American Medical Informatics Association. They mentioned their affiliation with **Université catholique de Louvain** and other contributors to the paper.

- **Congratulations Pour In**: Another member congratulated the author on the publication, adding a friendly "congrats 🙂" note. This shows community support and celebration for the author's achievement.

**Link mentioned**: <a href="https://doi.org/10.1093/jamia/ocae120">Impact of high-quality, mixed-domain data on the performance of medical language models</a>: AbstractObjective. To optimize the training strategy of large language models for medical applications, focusing on creating clinically relevant systems th

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1243389475254566934)** (8 messages🔥): 

- **SB-1047 sparks outrage**: Members discussed concerns about SB-1047, which they see as an attempt to centralize AI governance among big players like OpenAI. One member called it a “whimsical, flaming pile of garbage” and drew parallels with regulatory capture in Big Pharma and the Energy Sector, arguing it disadvantages smaller developers on tight budgets. 
- **Perplexity AI search link shared**: A member shared a link to [Perplexity AI search](https://www.perplexity.ai/search/SB-1047-Senate-2kZmFYHoTxe.rWUYat4B2A) regarding SB-1047. No further details or context was provided in the chat about the specifics of the search.
- **Arc Browser's Call Arc praised**: The new “Call Arc” feature of Arc Browser was highlighted for its simplicity and usefulness. The member praised it for allowing users to “ask your browser to find and collect relevant answers for you” effortlessly, sharing a [link for more details](https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD">1.44.1 Release</a>:  </li><li><a href="https://g.co/gemini/share/a36c7ad84489">‎Gemini - SB 1047: Stifling Open-Source AI Innovation?</a>: Created with Gemini Advanced
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1243468923203227658)** (5 messages): 

- **User faces issue with Typer installation**: A user stated *"queuelabs: pip install typer does not resolve"* indicating they are having trouble installing the **Typer** library using **pip**.
- **Poetry setup problem troubles users**: Another user asked *"Did you run poetry install before poetry run 01? Are you running in a virtual environment,"* pointing out potential steps missed in the setup process.
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1243385149178118216)** (9 messages🔥): 

- **Twinny + LM Studio blow minds as local co-pilot**: A user shared their positive experience using [Twinny](https://github.com/rjmacarthy/twinny) with LM Studio as a local co-pilot replacement. They asked about running this setup via llamafiles and received confirmation that running two llamafiles at the same time is possible by assigning different ports.

- **Embedding images with llama.cpp endpoint confusion solved**: A member asked if the llamafile/llama.cpp server supports images in llava embeddings and shared a command that did not work as expected. They later clarified that the `/v1/embeddings` endpoint does not accept `image_data` but using the `/embedding` endpoint works as expected.

- **Running continue.dev with llamafile performance issues**: Another user reported running continue.dev with llamafile, noting it was slow on a Mac M2 but somewhat faster on an older Nvidia GPU.

- **Inquiries on building and training custom LLMs**: A member sought advice on building and training a custom LLM using company documentation for internal use. They received a recommendation to use [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) for training, noting that llamafile only supports inference.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/index">🤗 Transformers</a>: no description found</li><li><a href="https://github.com/rjmacarthy/twinny">GitHub - rjmacarthy/twinny: The most no-nonsense, locally or API-hosted AI code completion plugin for Visual Studio Code - like GitHub Copilot but completely free and 100% private.</a>: The most no-nonsense, locally or API-hosted AI code completion plugin for Visual Studio Code - like GitHub Copilot but completely free and 100% private. - rjmacarthy/twinny</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/4681">Allow server to generate multimodal embeddings via the `/embedding` endpoint by kseth · Pull Request #4681 · ggerganov/llama.cpp</a>: The server already exposes multimodal support in /completion and other places, but not in /embedding. The change for this is relatively straightforward, if a user submits in image_data to the /embe...
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1243302339301871668)** (8 messages🔥): 

- **User Thanks Team**: "*THANK YOU!*" expressed in response to a previous interaction.
  
- **Inquiry About 104B Model**: A user asked if the team is planning to publish a **104B version** of their model family.

- **Langchain Integration Question**: A member inquired about the current status and recommendation for using **Langchain integration** with Cohere.

- **Aya Model Size Clarification**: A user asked whether the **Aya model** on the playground is for the 8B or 35B version.

- **Validation Error with Compressor**: An issue was shared regarding a `ValidationError` with **ContextualCompressionRetriever** due to an abstract method.

- **"56 Bananas Equal to 1 Apple" Calculation**: A calculation problem was explored with **CMR+**: *"1 apple = 2 pears, 3 pears = 4 oranges, 6 oranges = 7 bananas"*, concluding "56 bananas are equal to 1 apple."

- **403 Forbidden Error Troubleshoot**: A user reported a **403 Forbidden error** despite using the correct production key.
  

---



### **AI Stack Devs (Yoko Li) ▷ #[late-night-lounge](https://discord.com/channels/1122748573000409160/1159342774710186075/1243425073017131112)** (6 messages): 

- **AI Generated Standup comedy is surprisingly good**: A user shared a link expressing surprise at the quality of AI-generated standup comedy. They seemed impressed with its performance.

- **Exploring the Ud.io App**: Another user asked if the app mentioned, [Ud.io](https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG), only does comedy. This inquiry suggests curiosity about the app's full capabilities.

- **Transforming audio on Suno**: A member shared a more "demonic" version of the original audio using [Suno](https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655). This highlights the versatility of the platform in modifying sound.

- **Interest in Learning Audio Manipulation**: One user expressed interest in learning how to create audio modifications similar to the ones shared. This indicates a desire to acquire skills in audio engineering or AI-driven sound manipulation. 

- **Dismissive Response**: Briefly, a user responded with a curt "No" to a query, indicating either disinterest or negation of a previous statement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG">csimpkins - Standup Comedy on AI Generated Music | Udio</a>: Listen to Standup Comedy on AI Generated Music by csimpkins on Udio. Discover, create, and share music with the world. Use the latest technology to create AI music in seconds.</li><li><a href="https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655">AI Standup Comedy on AI Generated Musicby by @unwaveringplugin464 | Suno</a>: Standup comedian performing at a comedy show song. Listen and make your own with Suno.
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1243281449352691742)** (1 messages): 

- **Member seeks Google Calendar integration for event tracking**: A member inquired about the availability of an event calendar that could be imported into Google Calendar to avoid missing events. They expressed their concern with a sad emoji, indicating a need for a streamlined way to keep track of scheduled activities.
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 messages): 

evelynciara: yess I'm glad this channel exists 😅
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/)** (1 messages): 

datarevised: https://x.com/DataPlusEngine/status/1793803117642854732
  

---



---



---



---



---



---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1243653978135203983)** (45 messages🔥): 

- **Benchmarking Importance Highlighted**: Members shared a blog post discussing evaluation benchmarks ("evals") and their significant role in driving research breakthroughs, mentioning successful evals like [GLUE](https://arxiv.org/abs/1804.07461), [MMLU](https://arxiv.org/abs/2009.03300), and [GSM8K](https://arxiv.org/abs/2110.14168) ([source](https://www.jasonwei.net/blog/evals)). Challenges with "execution based evals" for dynamic tasks were also noted.

- **GPT-5 Rumors Stir Excitement**: Links to a talk suggested that **GPT-5** could arrive in 2024 with a significant leap in intelligence and a shift towards an "agent architecture" over token prediction ([link](https://x.com/rohanpaul_ai/status/1793956355897724973?s=46&t=90xQ8sGy63D2OtiaoGJuww)).

- **NumPy 2.0 Announcement Excites Developers**: Members joked about the forthcoming **NumPy 2.0** and its implications for dependency management, shared via [Twitter](https://x.com/cgarciae88/status/1794019900119236874).

- **Debate on Audio Note Apps**: A member highlighted **AudioPen** for converting voice to text using OpenAI's APIs, while another promoted their own app with advanced features like iCloud sync and markdown support ([source](https://techcrunch.com/2023/07/03/audio-pen-is-a-great-web-app-for-converting-your-voice-into-text-notes/)).

- **xAI Secures Impressive Funding**: **xAI** announced a $6 billion Series B funding round, aiming to advance their models' capabilities rapidly, drawing comparisons to OpenAI's valuation ([link](https://x.ai/blog/series-b)).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cgarciae88/status/1794019900119236874">Tweet from Cristian Garcia (@cgarciae88)</a>: NumPy 2.0 coming soon</li><li><a href="https://x.com/rohanpaul_ai/status/1793956355897724973?s=46&t=90xQ8sGy63D2OtiaoGJuw">Tweet from Rohan Paul (@rohanpaul_ai)</a>: New hints about GPT-5 in OpenAI Vivatech Paris Talk   The hints are at:  18:00 where he shows the chart of intelligence of &#34;GPT-Next&#34; arriving in 2024 - he talks about that model being a &#34;...</li><li><a href="https://x.com/swizec/status/865295043162021889">Tweet from Swizec Teller (@Swizec)</a>: @aJimHolmes @QualityFrog Really what we need is AI. We write the tests, AI builds code that passes.  True test driven development!</li><li><a href="https://x.com/levelsio/status/1795003725766877419?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from @levelsio (@levelsio)</a>: xAI by @elonmusk just raised $6 billion dollars  Not VALUED at $6 billion dollars, no they RAISED $6 billion dollars!  Apparently the largest investment round of all time  The valuation is $24B, Open ...</li><li><a href="https://www.latent.space/p/9bc44202-cad7-4124-8cfa-5cf7921cf556">Latent Space</a>: The AI Engineer newsletter + Top 10 US Tech podcast. Exploring AI UX, Agents, Devtools, Infra, Open Source Models. See https://latent.space/about for highlights from Chris Lattner, Andrej Karpathy, Ge...</li><li><a href="https://techcrunch.com/2023/07/03/audio-pen-is-a-great-web-app-for-converting-your-voice-into-text-notes/">AudioPen is a great web app for converting your voice into text notes | TechCrunch</a>: Developer Louis Pereira&#039;s AudioPen web app focuses on converting your voice to neat text-based notes using OpenAI APIs.</li><li><a href="https://www.jasonwei.net/blog/evals">Successful language model evals &mdash; Jason Wei</a>: Everybody uses evaluation benchmarks (“evals”), but I think they deserve more attention than they are currently getting. Evals are incentives for the research community, and breakthroughs are often cl...</li><li><a href="https://x.com/rohanpaul_ai/status/1793956355897724973?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Rohan Paul (@rohanpaul_ai)</a>: New hints about GPT-5 in OpenAI Vivatech Paris Talk   The hints are at:  18:00 where he shows the chart of intelligence of &#34;GPT-Next&#34; arriving in 2024 - he talks about that model being a &#34;...</li><li><a href="https://x.ai/blog/series-b">Series B funding round</a>: no description found</li><li><a href="https://x.com/xyz3va/status/1794413898608632004">Tweet from xyzeva (@xyz3va)</a>: hi @rabbit_hmi, i think we should talk about you breaking the android license, censoring our research, lying to your community  thread</li><li><a href="https://x.com/yi_ding/status/1793756026329870704?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Yi Ding -- prod/acc (@yi_ding)</a>: OpenAI claims they have 3 million developer accounts. Rough estimate is there are 27 million software developers world wide, so that would mean a double digit percentage have signed up for OpenAI.</li><li><a href="https://x.com/suno_ai_/status/1794367911408353349?s=12&t=XvL7HEPFiF7scRI6BJ2tzQ">Tweet from Suno (@suno_ai_)</a>: Make a song from any sound. Coming soon 🎧  VOL-1: A watering can, but make it psychedelic rock</li><li><a href="https://x.com/tantacrul/status/1794863603964891567?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Tantacrul (@Tantacrul)</a>: 1. I&#39;m legit shocked by the design of @Meta&#39;s new notification informing us they want to use the content we post to train their AI models. It&#39;s intentionally designed to be highly awkward ...</li><li><a href="https://x.com/dchaplot/status/1794109043348209969?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Devendra Chaplot @ ICLR 2024 (@dchaplot)</a>: We just released mistral-finetune, the official repo and guide on how to fine-tune Mistral open-source models using LoRA: https://github.com/mistralai/mistral-finetune  Also released Mistral-7B-Instru...</li><li><a href="https://apps.apple.com/us/app/brain-dump-voice-memo-notes/id6473446030">‎Record &amp; Transcribe: BrainDump</a>: ‎Brain Dump is your personal voice-to-text notepad, designed to capture and refine your thoughts with precision. Whether you're brainstorming ideas, jotting down voice notes, or dictating memos, Brain...</li><li><a href="https://open.substack.com/pub/swyx/p/iclr-2024-recap">ICLR 2024 — Best Papers &amp; Talks (ImageGen, Vision, Transformers, State Space Models) ft. Christian Szegedy, Ilya Sutskever, Durk Kingma</a>: 14 of the best papers out of the 2260 papers presented at the 2024 ICLR conference, in 4 sections covering Image Generation, Vision Learning, Extending Transformers, and State Space Models.</li><li><a href="https://open.substack.com/pub/swyx/p/iclr-2024-recap?r=1h4isl&utm_campaign=post&utm_medium=web">ICLR 2024 — Best Papers &amp; Talks (ImageGen, Vision, Transformers, State Space Models) ft. Christian Szegedy, Ilya Sutskever, Durk Kingma</a>: 14 of the best papers out of the 2260 papers presented at the 2024 ICLR conference, in 4 sections covering Image Generation, Vision Learning, Extending Transformers, and State Space Models.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1244758635419009045)** (1 messages): 

- **New Podcast Episode: ICLR Recap Part 1**: Latent Space released a new podcast episode focusing on the highlights of [ICLR 2024](https://x.com/latentspacepod/status/1795196817044594817) best papers. Topics include **ImageGen, Compression, Adversarial Attacks, Vision Learning and Weak Supervision, Extending Transformers and Attention,** and **State Space Models vs Transformers**.

**Link mentioned**: <a href="https://x.com/latentspacepod/status/1795196817044594817">Tweet from Latent Space Podcast (@latentspacepod)</a>: 🆕 ICLR 2024: Best Papers (Part 1)  We present our selections of outstanding papers and talks thematically introducing topics for AI Engineers to track:  Section A: ImageGen, Compression, Adversarial ...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1243654830749257810)** (241 messages🔥🔥): 

- **AI-Created Music Raises Copyright Questions**: Discussion around AI-produced songs like "**Suno**" led to inquiries about copyright implications. Members noted that "AI generated artworks are not copyrightable," highlighting ongoing legal ambiguities.
- **Meta's Audiocraft Impresses**: A member praised Meta's **Audiocraft** for its near-realtime capabilities, even if some preprocessing might be involved. There was debate over whether this was truly real-time.
- **Tools and Plugins for Musicians**: The group explored various AI tools for musicians, including **Neutone** and **moises.ai**, with members sharing workflows and experiences with these technologies. One highlighted "rad stuff" integrating models into plugins, enhancing music production.
- **Legal and Ethical Implications of AI in Music**: There was a vigorous discussion on the legal aspects of AI in music, especially regarding usage rights and impersonation. "Do not use our IP for training data" was cited as a company's stance amid copyright concerns.
- **Open Source and Creative Freedom**: Members expressed excitement over open-source projects like **gary-backend-combined**. The idea of a platform where "copyright doesn't matter" was floated, reflecting the community's desire for creative freedom.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://redito.substack.com/p/surfing-human-creativity-with-ai">Surfing human creativity with AI feat. Nao Tokui (Neutone)</a>: Using AI to expand human creativity, AI as &quot;mind mirror&quot;, text-to-audio limitations, building AI tools for musicians, and the role of the artist in the AI era.</li><li><a href="https://suno.com/song/ccd7c687-b9ad-48a3-9636-2bf11d5c97a3">Latent Space by @unchainedmeter763 | Suno</a>: hip hop electronic song. Listen and make your own with Suno.</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024  Topic,Date,Facilitator,Resources,@dropdown,@ UI/UX patterns for GenAI,1/26/2024,nuvic,&lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://github.com/betweentwomidnights/gary-backend-combined">GitHub - betweentwomidnights/gary-backend-combined: combined backend for gary4live and gary-on-the-fly</a>: combined backend for gary4live and gary-on-the-fly - betweentwomidnights/gary-backend-combined
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1243657828623712356)** (26 messages🔥): 

- **VSCode extension bugs in nightly builds**: Several members discussed issues with the **Mojo VSCode extension**, highlighting that it works properly only for files created after the installation of the nightly build. One member mentioned needing to close and reopen the code to resolve issues, while another found the extension buggy since version 24.3, requiring frequent resets.
- **Comparing torch code in Python and Mojo**: A user compared `torch.randn` operations and `cuda()` methods in both **Python** and **Mojo** codes, pointing out Mojo's seemingly better SASS and overhead. Discussions focused on potential minor differences, such as the function arguments and memory management, with detailed scrutiny on their impacts.
- **Struggling with WSL and APT on Ubuntu**: One user expressed frustration with **WSL** and Linux package management, sharing an error-laden output from attempting to use `apt`. They debated switching to Mac due to these issues and a lack of desire to deal with AI integration in their setup.
- **MAX Engine Python API installation issues**: A user couldn't install the **MAX Engine Python API** due to compatibility issues, receiving errors related to unmatched distributions. Another user suggested checking the Python version, noting that versions 3.8 to 3.11 are supported but not 3.12.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://archive.ubuntu.com/ubuntu">Index of /ubuntu</a>: no description found</li><li><a href="http://security.ubuntu.com/ubuntu">Index of /ubuntu</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot: From *Modular*:
<https://twitter.com/Modular/status/1794148687104647293>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1243940391766593637)** (4 messages): 

- **Chatbot enjoys tacos**: Members speculated about the chatbot's configuration and personality. One pointed out, *"it seems to really like tacos."*
  

---


### **Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1243718867889422416)** (6 messages): 

- **Curiosity about Mojo’s performance vs. hand-written kernels**: A user mentions Futhark's tooling issues and questions how Mojo plans on performing against hand-written kernels, noting that Futhark doesn't attempt such a comparison. They also share their background in CUDA C++ and Scala, expressing appreciation for functional programming languages.
  
- **Ambitious C++ interoperability goals for Mojo**: A user admits that Mojo’s goal of interoperability with C++ sounds ambitious and inquires how the language plans to achieve this beyond using C as an intermediary. 

- **Clarification on default parallelism in Bend**: Responding to a query on language design and parallelism, one user explains that while most languages default to serial execution, Bend is designed to naturally encourage writing parallel code without requiring extra effort from the programmer.

- **xAI raises $6 billion in funding**: Elon Musk’s AI startup, xAI, raised $6 billion in a Series B funding round, attracting major investors like Valor Equity Partners and Andreessen Horowitz. The funding positions xAI to aggressively compete with giants like OpenAI and Microsoft, as reported by [TechCrunch](https://techcrunch.com/2024/05/26/elon-musks-xai-raises-6b-from-valor-a16z-and-sequoia/).

- **Discussion on commercialization of AI**: A user highlights Tesla’s self-driving technology as a notable large commercial use case and questions how AI is being commercialized in general.

**Link mentioned**: <a href="https://techcrunch.com/2024/05/26/elon-musks-xai-raises-6b-from-valor-a16z-and-sequoia/">Elon Musk&#039;s xAI raises $6B from Valor, a16z, and Sequoia | TechCrunch</a>: Elon Musk&#039;s AI startup, xAI, has raised $6 billion in a new funding round, it said today, as Musk shores up capital to aggressively compete with rivals Elon Musk&#039;s AI startup, xAI, has raise...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1243645666383757313)** (133 messages🔥🔥): 

- **Understanding Python to Mojo function differences**: Members discussed the difficulty of hashing functions in Mojo compared to Python. One member explained, *"Mojo functions doesn't have [object identity]."* Another highlighted the transition from dynamic to static typing in Mojo as a possible barrier.
- **Initializing Optional Types in Mojo**: Members faced challenges correctly initializing variables, particularly custom structs, to `None`. Multiple solutions involving `UnsafePointer` and `Optional` were suggested, leading one member to finally get the code working without errors.
- **Creating Linked Lists in Mojo**: There was a lengthy discussion on properly implementing and initializing a LinkedList in Mojo. Members debated various approaches using `UnsafePointer` and `Optional` types for the `next` variable.
- **Reflection Capability in Mojo**: Participants were curious about Mojo's reflection capabilities for accessing data about structures. It's noted that *"MLIR meta programming should enable that in the future,"* but it hasn't been implemented yet.
- **Importing Mojo Packages for Testing**: Issues were encountered while trying to import modules from the main source directory into a test directory. Solutions included ensuring `__init__.mojo` files were present, although some complications persisted with the `mojo test` command.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/collections/optional/">optional | Modular Docs</a>: Defines Optional, a type modeling a value which may or may not be present.</li><li><a href="https://github.com/saviorand/lightbug_http">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/1405">[Modular CLI]: modular install mojo should support version pinning · Issue #1405 · modularml/mojo</a>: Issue description I cannot figure out how to install a specific version of mojo. This would be useful, essential really, for library maintainers and CI/CD systems. Steps to reproduce $ modular inst...</li><li><a href="https://modul.ar/systems">Roadmap &amp; known issues | Modular Docs</a>: A summary of known issues and upcoming features for the MAX platform.</li><li><a href="https://github.com/modularml/mojo/pull/2825">[stdlib] Add optional small buffer optimization to `List`, take 2 by gabrieldemarmiesse · Pull Request #2825 · modularml/mojo</a>: This PR solves part of #2467 This PR is part of three PRs to read and merge in the following order  #2825 #2826 #2827  The small buffer behavior We use InlineArray to store up to small_buffer_size ...</li><li><a href="https://github.com/modularml/mojo/pull/2826">[stdlib] Work around the materialization bug present in #2825 by gabrieldemarmiesse · Pull Request #2826 · modularml/mojo</a>: This PR solves part of #2467 This PR is part of three PRs to read and merge in the following order  #2825 #2826 #2827  The important diff is the one between this branch and the branch of the PR #28...</li><li><a href="https://github.com/modularml/mojo/pull/2827">[stdlib] Add SSO to String by using SBO in List with materialization workaround by gabrieldemarmiesse · Pull Request #2827 · modularml/mojo</a>: This PR solves part of #2467 This PR is part of three PRs to read and merge in the following order  #2825 #2826 #2827  The interesting part is the diff between this PR and the PR #2826 You can find...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1244564633466175538)** (22 messages🔥): 

- **Compact CRC32 causes slight performance dip**: A member noted their attempt to make CRC32 calculations in Mojo more compact, but it resulted in a performance drop from ~14x to ~11x uplift over baseline. The [benchmark file can be found here](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc.mojo).

- **Unroll decorator bridges performance gap**: After a suggestion to use the unroll decorator, the performance of the compact implementation improved, almost eliminating the gap, with a compact speedup of up to 1429x.

- **Efficiency cores show different performance**: Tests on efficiency cores showed the compact version consistently outperformed the non-compact implementation marginally, with the results across various core types showing noticeable differences in speedup.

- **SIMD-based implementation issues**: An exploration into SIMD for CRC32 produced non-deterministic results due to potential memory reuse issues, pointing to either a bug or misuse, with a correct variant still proving to be slower on x86 hardware compared to the original implementations.

- **Cache limits impact larger byte cases**: For larger byte sizes (e.g., 16, 32 bytes), the compact version faced performance slowdowns likely due to L1 cache limits, as seen in speedup metrics when benchmarked on an Intel i7-12700H. The [nightly version tests are available here](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc.mojo">fnands.com/blog/2024/mojo-crc-calc/crc.mojo at main · fnands/fnands.com</a>: My personal blog. Contribute to fnands/fnands.com development by creating an account on GitHub.</li><li><a href="https://github.com/komrad36/CRC?tab=readme-ov-file">GitHub - komrad36/CRC: Fastest CRC32 for x86, Intel and AMD, + comprehensive derivation and discussion of various approaches</a>: Fastest CRC32 for x86, Intel and AMD, + comprehensive derivation and discussion of various approaches - komrad36/CRC</li><li><a href="https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo">fnands.com/blog/2024/mojo-crc-calc/crcn.mojo at main · fnands/fnands.com</a>: My personal blog. Contribute to fnands/fnands.com development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1243790811645218876)** (75 messages🔥🔥): 

- **Mojo Compiler's Nightly Updates Announced**: Multiple nightly releases of the Mojo compiler have been shared, like `2024.5.2505`, `2024.5.2514`, `2024.5.2605`, and `2024.5.2705`. Key updates include local variable renaming in LSP, introducing list sorting and methods to `List`, moving functionality off refitem to getitem, and adding a `tempfile` module (changelogs [here](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)).

- **String and List Implementation Issues Discussed**: Members discussed issues with the current list and string implementations, suggesting various fixes to handle null pointers and identity concerns in lists. Suggestions included having the default list constructor call `alloc(0)` and ensuring strings track null buffers properly.

- **Auto Dereferencing Iterators with New `ref` Syntax**: New `ref` syntax prompted discussions about its potential for auto-dereferencing iterators. The `ref` convention could lead to simpler implementations and reduce the need for manual dereferencing in function calls.

- **Proposal for Function Return Types and Variadic Arguments**: An attempt to implement Python's `zip` and `unzip` functions in Mojo revealed challenges with declaring functions that return tuples based on variadic list arguments. The current efforts revealed errors and prompted further investigation into correct function declaration.

- **Testing Challenges with Latest Compiler Updates**: The latest compiler updates caused issues with existing pull requests, leading to tests breaking and highlighting the need for bug fixes. Logging was suggested as a temporary measure to address flaky bugs, as seen in the updated pull request [#2832](https://github.com/modularml/mojo/pull/2832).

**Link mentioned**: <a href="https://github.com/modularml/mojo/pull/2832">[stdlib] Add some logging to `test_reverse.mojo` to flush out a flaky bug by gabrieldemarmiesse · Pull Request #2832 · modularml/mojo</a>: See #2369 this bug is appearing more and more. Some logging should help us understand what fails exactly.

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1243647477882552420)** (3 messages): 

- **Phi-3 Medium 128k Instruct Released**: The new model [microsoft/phi-3-medium-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct) is available in both standard and free variants. Users are encouraged to check the [discussion thread](https://discord.com/channels/1091220969173028894/1232344285484023839) to share feedback on its performance.
- **Announcement on X Platform**: [@OpenRouterAI](https://x.com/OpenRouterAI/status/1794101495538843803) announced the new free model Phi-3 Medium with both a standard and free variant.
- **New Model and Price Reduction**: Microsoft's [phi-3-mini-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-mini-128k-instruct) model is now available. Additionally, the [llama-3-lumimaid-70b](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b) model has a significant 57% price cut.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1794101495538843803">Tweet from OpenRouter (@OpenRouterAI)</a>: New free model: Phi 3 Medium 🧠 microsoft/phi-3-medium-128k-instruct with a standard & free variant</li><li><a href="https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct>)">Phi-3 Medium Instruct by microsoft | OpenRouter</a>: Phi-3 Medium is a powerful 14-billion parameter model designed for advanced language understanding, reasoning, and instruction following. Optimized through supervised fine-tuning and preference adjust...</li><li><a href="https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free>)">Phi-3 Medium Instruct by microsoft | OpenRouter</a>: Phi-3 Medium is a powerful 14-billion parameter model designed for advanced language understanding, reasoning, and instruction following. Optimized through supervised fine-tuning and preference adjust...</li><li><a href="https://openrouter.ai/models/microsoft/phi-3-mini-128k-instruct>)">Phi-3 Mini Instruct by microsoft | OpenRouter</a>: Phi-3 Mini is a powerful 3.8B parameter model designed for advanced language understanding, reasoning, and instruction following. Optimized through supervised fine-tuning and preference adjustments, i...</li><li><a href="https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b>)">Llama 3 Lumimaid 70B by neversleep | OpenRouter</a>: The NeverSleep team is back, with a Llama 3 70B finetune trained on their curated roleplay data. Striking a balance between eRP and RP, Lumimaid was designed to be serious, yet uncensored when necessa...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1244097547773939762)** (260 messages🔥🔥): 

- **Rate Limit Issues Explained with Documentation**: Members discussed rate limiting in OpenRouter, including how rate limits scale with credits. A detailed explanation was referenced in the [OpenRouter documentation](https://openrouter.ai/docs#rate-limits-and-credits-remaining), clarifying that higher credit balances allow for higher request rates.

- **Modal Fallback Error Handling**: A user experienced a rate limit error while using modal fallback functionality despite having sufficient credits. The community suggested checking remaining free requests and potentially omitting the free model when limits are reached.

- **Claude Self-Moderated Models Usage Decline**: Users speculated on the decline in usage of Claude's self-moderated models, citing increased refusals and more stringent guardrails as potential reasons. Some users noted this change has made the models less human-like and more PR-oriented.

- **Cost Comparison for AI Model Hosting**: Comparisons were made between different hosting solutions like RunPod, Vast.ai, and major cloud providers like Google Cloud and Amazon Bedrock. Users highlighted significantly lower prices for GPU usage on alternative platforms compared to traditional cloud services.

- **Vision Model Cost and Performance**: Discussion on the cost-effectiveness and performance of various vision models, with suggestions to evaluate Gemini and its OCR capabilities for specific tasks. The community noted favorable performance and competitive pricing for Gemini's vision services.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1cx72of/chinese_compa">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cx72of/chinese_companies_aim_to_use_price_advantages_to/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/nvidia/NV-Embed-v1">nvidia/NV-Embed-v1 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/bnS5n.gif">Dead Space GIF - Dead Space - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://orw.karleo.net/changes">OpenRouter API Watcher</a>: Explore OpenRouter's model list and recorded changes. Updates every hour.</li><li><a href="https://openrouter.ai/docs#rate-limits-and-credits-remaining">Docs | OpenRouter</a>: Build model-agnostic AI apps
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1243659762982457445)** (1 messages): 

- **Team's CCS follow-up project shared**: In Spring 2023, a team followed up on Collin Burns's _Contrast Consistent Search (CCS)_ method for ELK. While the empirical results did not show predictable generalization improvements, they shared their proposed method and results on the [Quirky Models benchmark](https://arxiv.org/abs/2312.01037) for transparency. 
- **Transparency in unsuccessful attempts**: EleutherAI disclosed their project, acknowledging it failed to provide solid evidence of improved generalization, reinforcing the importance of sharing both successes and failures. More details can be found in their [blog post](https://blog.eleuther.ai/vincs/).

**Link mentioned**: <a href="https://blog.eleuther.ai/vincs/">VINC-S: Closed-form Optionally-supervised Knowledge Elicitation with Paraphrase Invariance</a>: Writing up results from a project from Spring 2023

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1243641744159932518)** (63 messages🔥🔥): 

- **Discussion on RAG vs Finetuning for LLMs**: A member asked if **RAG** (Retrieval-Augmented Generation) is the easiest way for a **LLM** to extract information from a custom library as opposed to finetuning. Another member recommended using RAG, explaining that finetuning doesn't do a great job of teaching the model information but is more about teaching style.

- **NeoX Model History and Compute Resources**: There was a discussion on the origins of the **NeoX models** and their developers. It was mentioned that compute details are covered in the [Eleuther blog](https://blog.eleuther.ai/year-one/) and related papers.

- **Transformers vs SSMs for Time Series**: Members discussed why **transformers** may not perform well for **multivariate time series**, while **SSMs** handle them better. Some references like [Chronos-T5](https://huggingface.co/amazon/chronos-t5-large) and [Spacetimeformer](https://github.com/QData/spacetimeformer) were shared to provide insight.

- **ThePitbull 21.4B Model Released**: A new model named **ThePitbull 21.4B** has been introduced on Hugging Face and claims to be highly powerful, nearly as good as a 70B model. There's a quantized version available and some skepticism from the community about its performance.

- **TRC Credits and Region Locks**: There was a brief discussion about the availability of **TRC credits** and the region-locking issue on them. It was clarified that non-preemptible/single-host credits are being used heavily, and available regions are US and Europe with specific regional limitations making access difficult.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/fblgit/UNA-ThePitbull-21.4-v1">fblgit/UNA-ThePitbull-21.4-v1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/amazon/chronos-t5-large">amazon/chronos-t5-large · Hugging Face</a>: no description found</li><li><a href="https://blog.eleuther.ai/year-one/">What A Long, Strange Trip It&#39;s Been: EleutherAI One Year Retrospective</a>: A look back at the first year of EleutherAI.</li><li><a href="https://huggingface.co/saltlux/luxia-21.4b-alignment-v1.0/discussions/5#65fbd17bb67a26e59ed169de">saltlux/luxia-21.4b-alignment-v1.0 · 91.9 HellaSwag, 79.2 TruthfulQA... And It Sucks. Why do this?</a>: no description found</li><li><a href="https://github.com/QData/spacetimeformer">GitHub - QData/spacetimeformer: Multivariate Time Series Forecasting with efficient Transformers. Code for the paper &quot;Long-Range Transformers for Dynamic Spatiotemporal Forecasting.&quot;</a>: Multivariate Time Series Forecasting with efficient Transformers. Code for the paper &quot;Long-Range Transformers for Dynamic Spatiotemporal Forecasting.&quot; - QData/spacetimeformer
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1243667962162839744)** (58 messages🔥🔥): 

- **Exploration of Gradient Perturbation Techniques**: Members discussed using multiple initializations of the same architecture, perturbing weights to get a distribution of weight updates. *"It'd be cool if we could say that Initialization X gives gradients that are relevant to such and such qualitative features of a model, Initialization Y gives others and then combine those in a principled way."*

- **Generative AI and Law Workshop Announcement**: An announcement was made for the 2nd Workshop on Generative AI and Law at ICML ’24 in Vienna, focusing on IP and privacy challenges in the UK and EU. Submission details and topics of interest like industry deployment challenges and data protection were shared ([workshop website](https://genlaw.org/2024-icml.html)).

- **Discussion on Schedule-Free Optimization**: A paper was highlighted proposing a novel approach to learning rate schedules ([arxiv link](https://arxiv.org/abs/2405.15682)). The method claims to outperform traditional schedules by unifying scheduling and iterate averaging without introducing additional hyper-parameters.

- **Transformers and Implicit Reasoning**: A study on transformers' capability to implicitly reason over parametric knowledge was shared, raising points about generalization and the effectiveness of extended training to achieve implicit reasoning ([arxiv link](https://arxiv.org/abs/2405.15071)). The discussion touched on the limitations in composition and capabilities in comparison reasoning.

- **Critique and Skepticism Towards Academic Claims**: Some members expressed skepticism about claims in a schedule-free optimization paper due to prior promotional practices by the author. Concerns were raised about disparate engagement with positive feedback versus critical concerns.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.13956">Attention as an RNN</a>: The advent of Transformers marked a significant breakthrough in sequence modelling, providing a highly performant architecture capable of leveraging GPU parallelism. However, Transformers are computat...</li><li><a href="https://arxiv.org/abs/2405.15071">Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization</a>: We study whether transformers can learn to implicitly reason over parametric knowledge, a skill that even the most capable language models struggle with. Focusing on two representative reasoning types...</li><li><a href="https://x.com/BlinkDL_AI/status/1794746863901642866">Tweet from BlinkDL (@BlinkDL_AI)</a>: LLM Compression Leaderboard on latest (unseen) data: https://huggingface.co/spaces/Jellyfish042/UncheatableEval Better compression = Better base model. RWKV-6🐦 is highly competitive while only traine...</li><li><a href="https://arxiv.org/abs/2405.15682">The Road Less Scheduled</a>: Existing learning rate schedules that do not require specification of the optimization stopping step T are greatly out-performed by learning rate schedules that depend on T. We propose an approach tha...</li><li><a href="https://arxiv.org/abs/2405.15731">Understanding the differences in Foundation Models: Attention, State Space Models, and Recurrent Neural Networks</a>: Softmax attention is the principle backbone of foundation models for various artificial intelligence applications, yet its quadratic complexity in sequence length can limit its inference throughput in...</li><li><a href="https://tenor.com/view/catholic-meryl-streep-nuns-sisters-i-have-doubts-gif-5743847">Catholic Meryl Streep GIF - Catholic Meryl Streep Nuns - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2405.15722">Models That Prove Their Own Correctness</a>: How can we trust the correctness of a learned model on a particular input of interest? Model accuracy is typically measured \emph{on average} over a distribution of inputs, giving no guarantee for any...</li><li><a href="https://arxiv.org/abs/2405.13861">Transformers Learn Temporal Difference Methods for In-Context Reinforcement Learning</a>: In-context learning refers to the learning ability of a model during inference time without adapting its parameters. The input (i.e., prompt) to the model (e.g., transformers) consists of both a conte...</li><li><a href="https://icml.cc/virtual/2024/poster/32613">ICML Poster Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Space Duality</a>: no description found</li><li><a href="https://arxiv.org/abs/2405.13967">DeTox: Toxic Subspace Projection for Model Editing</a>: Recent alignment algorithms such as direct preference optimization (DPO) have been developed to improve the safety of large language models (LLMs) by training these models to match human behaviors exe...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1244134857215447082)** (5 messages): 

- **Anticipation for a New Release**: A member expressed excitement for an upcoming release, asking "when's that coming?" This indicates a high interest in upcoming datasets or tools.
- **Interest in Safe Datasets**: Another message conveyed keen interest in obtaining a "safe dataset." The smiley face implies a positive outlook towards the availability of such datasets.
- **Small Models vs. Quantized Larger Models**: A member inquired about scenarios where small models outperform larger ones that are quantized. "In my evals the quantized ones consistently perform better despite no longer being larger," sheds light on interesting findings that quantized models might maintain or exceed performance.
- **Perplexity and Quantization**: Shared links discuss the impact of quantization on model performance, pointing to [ggerganov's Twitter post](https://x.com/ggerganov/status/1666087050725199872) and a [pull request on GitHub](https://github.com/ggerganov/llama.cpp/pull/1684). These resources show the efficiency of different bit-level quantization methods and their implementation.

**Link mentioned**: <a href="https://x.com/ggerganov/status/1666087050725199872">Tweet from Georgi Gerganov (@ggerganov)</a>: 2,3,4,5 and 6-bit quantization methods are now available in llama.cpp  Efficient inference implementation with ARM NEON, AVX2 and CUDA - see sample numbers in the screenshots Big thanks to ikawrakow f...

  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

vkc6969: any work being done for mech interp for diffusion models?
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1243922350236499968)** (11 messages🔥): 

- **Tokenizers break during Pythia data replication**: Members discuss issues with replicating Pythia's data due to changes in tokenizers. One member mentions that *"HF mage breaking changes to how tokenizers work at some point."*
- **Batch_viewer.py as a solution for detokenization**: It's suggested to use `batch_viewer.py` from the Pythia repo to access the contents of training batches correctly. Instructions are provided in the [GitHub repository](https://github.com/EleutherAI/pythia#exploring-the-dataset) which explains that it uses a pre-shuffled version of the dataset.
- **Challenges with reading `.bin` and `.idx` files**: Discussions reveal that simply loading `.bin` files won't work as they must be read using the `MMapIndexedDataset` class to correctly interpret token data.
- **Loading files correctly**: Members emphasized the importance of loading the memmap with `dtype='uint16'` to read tokens correctly from the pre-shuffled datasets. One sample read: *"It is done, and submitted. You can play “Survival of the Tastiest” on Android, and on the web...".*
- **MMapIndexedDataset code complexity**: The code involving `MMapIndexedDataset` and the use of the `struct` package is recognized as complex and akin to "black magic" by some members. However, it is crucial for correctly interpreting document tokens from the dataset.

**Link mentioned**: <a href="https://github.com/EleutherAI/pythia#exploring-the-dataset">GitHub - EleutherAI/pythia: The hub for EleutherAI&#39;s work on interpretability and learning dynamics</a>: The hub for EleutherAI&#39;s work on interpretability and learning dynamics - EleutherAI/pythia

  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1244683541741047839)** (1 messages): 

- **Join Special Webinar on RAG Pipelines**: "We’re hosting a special webinar this Thursday 9am PT in a special collaboration with Ragas + AWS folks on building an enterprise RAG pipeline with bedrock, ragas, llamaindex." Register at [lu.ma](https://lu.ma/x8unmku0) to participate and learn about integrating LlamaIndex with Bedrock and creating an evaluation framework with Ragas.



**Link mentioned**: <a href="https://lu.ma/x8unmku0">LlamaIndex Webinar: Build Enterprise RAG with Bedrock, Ragas, and LlamaIndex · Zoom · Luma</a>: This is a special collaboration between folks from LlamaIndex, Ragas, and AWS to bring you a workshop on building a production-quality enterprise RAG…

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1243666346936504340)** (7 messages): 

- **Vespa integration steals the show**: The community is excited about a new Vespa vector store integration, praised for its robust support for hybrid search with BM25. Check out the official announcement [here](https://twitter.com/llama_index/status/1794106979213869413).

- **Blazing-Fast RAG Chatbot Guide goes live**: Jayita B. offers a resourceful guide on building an advanced RAG indexing/query pipeline and turning it into a full-stack application with rapid response on Llama3 and GroqInc. Dive into the tutorial [here](https://twitter.com/llama_index/status/1794384304665141586).

- **Innovative image indexing with structured annotations**: A novel method to index images for RAG involves using structured annotations produced by models like gpt4o, noted for their efficiency. [Learn more](https://twitter.com/llama_index/status/1794517226986656150).

- **LlamaFS cleans up your messy files**: Introducing LlamaFS, a self-organizing file manager that tidies up your directories automatically. See how it works [here](https://twitter.com/llama_index/status/1794762651769430381).

- **Enterprise RAG with AWS Bedrock webinar**: A special webinar is scheduled to help you build production-level RAG pipelines using AWS Bedrock and Ragas. Don't miss this collaborative workshop, detailed [here](https://twitter.com/llama_index/status/1795123736699629983).

**Link mentioned**: <a href="https://t.co/qIGOmCW62G">RSVP to GenAI Summit Pre-Game: Why RAG Is Not Enough? | Partiful</a>: Note: This is an in-person meetup @LlamaIndex HQ in SF!  Stop by our meetup to learn about latest innovations in building production-grade retrieval augmented generation engines for your company from ...

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1243671400200405162)** (100 messages🔥🔥): 

- **Max iterations issue with Llama Index's reAct**: A user experienced a max iterations error when using reAct with llama index. Another member suggested increasing the iterations by setting `max_iterations` parameter in the `ReActAgent` constructor.
  
- **Import errors and installation confusion**: Multiple users faced import errors and package conflicts with `SimpleDirectoryReader` and `DEFAULT_FILE_EXTRACTOR` in `llama_index`. Solutions involved installing specific sub-packages and checking for updates or conflicts between package versions.

- **Parsing HTML vs. PDF files**: A detailed discussion unfolded about the advantages and difficulties of parsing HTML vs. PDF files. Many agreed that HTML needs more custom parsing code, whereas PDF chunking tools appear to be more advanced and less dependent on additional libraries.

- **Integrating Pydantic with LlamaIndex**: Instructions were provided on using `Pydantic` classes for structured output with LlamaIndex. Guidance included defining Pydantic models and integrating them via `OpenAIPydanticProgram`.

- **Improving documentation on retrievers**: A suggestion was made to improve the documentation on different retriever modules like BM25 and AutoRetrieval. Members clarified their usage scenarios and differences, highlighting a gap in the current documentation.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/getting_started/starter_example/#query-your-data">Starter Tutorial (OpenAI) - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/node_parser/file/html.py">llama_index/llama-index-core/llama_index/core/node_parser/file/html.py at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/prompts/prompt_mixin/#accessing-prompts">Accessing/Customizing Prompts within Higher-Level Modules - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/understanding/putting_it_all_together/q_and_a/terms_definitions_tutorial/#improvement-3-image-support>)">A Guide to Extracting Terms and Definitions - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/output_parsing/openai_pydantic_program/#without-docstring-in-model>))">OpenAI Pydantic Program - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/finetuning/embeddings/finetune_embedding_adapter.ipynb">llama_index/docs/docs/examples/finetuning/embeddings/finetune_embedding_adapter.ipynb at main · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/7849b1a851d88ee28e1bfd05d19f18e40d5b8e10/llama-index-integrations/embeddings/llama-index-embeddings-adapter/llama_index/embeddings/adapter/base.py#L115">llama_index/llama-index-integrations/embeddings/llama-index-embeddings-adapter/llama_index/embeddings/adapter/base.py at 7849b1a851d88ee28e1bfd05d19f18e40d5b8e10 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1243666547294343290)** (93 messages🔥🔥): 

- **Interest in Aya-23 Functionality and Comparisons**: Members discussed **Aya-23's** potential and its functionality compared to **Command R/R+**. One comment mentioned that Aya-23 might have *"improved multilingual capability"* but questioned its performance in English and potential specialized training ("based on Cohere’s Command model1 and the Aya multilingual instruction-style collection").
  
- **Local Model Dependencies for Mobile Projects**: A member sought advice on developing a **RAG mobile app** that runs LLM locally for privacy reasons. The consensus suggested that **on-phone LLMs are not ready** yet for such tasks.

- **System Prompts in Aya-23**: A conversation revealed that **Aya-23** supports system prompts, with users modifying **Command R** prompts to successfully work on **Aya-23**. One member shared a specific template configuration highlighting tokens like `<|USER_TOKEN|>` and `<|CHATBOT_TOKEN|>`.

- **OneDrive Connector Guidance**: A query regarding the existence of a **OneDrive connector** was answered by pointing to a [SharePoint connector](https://github.com/cohere-ai/quick-start-connectors/tree/main/sharepoint), which might serve the same purpose.

- **Aya-23 Availability Clarifications**: An official communication affirmed that **Aya-23-35b** is a fine-tuned version of **Command R** and currently available for non-commercial use only. One member pointed to the [official paper](https://drive.google.com/file/d/1YKBPo61pnl97C1c_1C2ZVOnPhqf7MLSc/view) for further details.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gitgud.autonoma.app/playground/3c135aa8-2720-4950-a184-61b3948a55bf/code?utm_source=discord&utm_medium=social&utm_campaign=cohere)">GitGud</a>: no description found</li><li><a href="https://docs.cohere.com/docs/migrating-from-cogenerate-to-cochat">Migrating from the Generate API to the Chat API - Cohere Docs</a>: no description found</li><li><a href="https://drive.google.com/file/d/1YKBPo61pnl97C1c_1C2ZVOnPhqf7MLSc/view">aya_23_technical_report.pdf</a>: no description found</li><li><a href="https://github.com/cohe">cohe - Overview</a>: cohe has 5 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/cohere-ai/quick-start-connectors/tree/main/sharepoint">quick-start-connectors/sharepoint at main · cohere-ai/quick-start-connectors</a>: This open-source repository offers reference code for integrating workplace datastores with Cohere&amp;#39;s LLMs, enabling developers and businesses to perform seamless retrieval-augmented generation...</li><li><a href="https://docs.cohere.com/docs/creating-and-deploying-a-connector">Creating and Deploying a Connector - Cohere Docs</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1244762231497818193)** (3 messages): 

- **Gaming Bot on LinkedIn Shines**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/activity-7199625887955177472-nLbL?utm_source=share&utm_medium=member_ios) highlighting a new gaming bot they created for Discord using Cohere Command R. The post generated curiosity about the bot's capabilities and integration.

- **Create 'n' Play Bot Debut**: The new gaming bot for Discord, named "Create 'n' Play," boasts some impressive features. It offers *"Over 100 engaging text-based games,"* allows for *"Easy team formations and interactions,"* and enhances *"social engagement with AI."*
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1243639929712676937)** (77 messages🔥🔥): 

- **Google AI's Reddit fail mishap**: Google's AI using Reddit's information has led to dangerous advice like "jump off bridges to cure depression". One member quipped about the "notoriously bad advice" often seen on Reddit, sharing a [meme image](https://media.discordapp.net/attachments/808203765319467028/1243450997511032935/GOUGsBgaMAA4N9k.jpg).
- **Gemini AI and Google's safety team**: Discussions about Google's multiple layoffs, including their safety team, raised concerns about AI handling sensitive topics poorly, such as during the Gemini debacle. A shared [New York Times article](https://www.nytimes.com/2024/02/22/technology/google-gemini-german-uniforms.html) was mentioned to contextualize these issues.
- **Issues with AI and UBI**: Conversations delved into Universal Basic Income (UBI) with members debating its feasibility and equity, particularly for the global south. One member expressed skepticism, noting that UBI "might be plausible for developed nations" but less so for poorer regions.
- **Datasets and AI tools shared**: Members shared useful resources like a potential dataset of publication PDFs and their source TeX from [archive.org](https://archive.org/details/arxiv-bulk?sort=-publicdate), and a [YouTube video](https://youtu.be/DQ0lCm0J3PM?si=5Is7OMnqRhZb-ZAo) about creating a neural network with only Redstone.
- **Mastodon as an alternative to Twitter**: Users expressed frustration with Twitter's algorithm changes and discussed alternatives like Mastodon. A member highlighted that Mastodon’s lack of rage-inducing content might be why it hasn't seen the exponential growth but still maintains a loyal user base.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://playpermissionless.substack.com/p/twitter-is-now-attention-roulette">Twitter is now attention roulette and ultimately meaningless</a>: I no longer feel any pull towards Twitter. I can&#x27;t really recall the last time anything good came out of it for me personally. Neither ideas nor connections. The changes to the algorithm turned t...</li><li><a href="https://www.tiktok.com/@hbo/video/7268371228384251182>">TikTok - Make Your Day</a>: no description found</li><li><a href="https://huggingface.co/spaces/Tonic/Granite-Code">Granite Code - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://youtu.be/DQ0lCm0J3PM?si=5Is7OMnqRhZb-ZAo">I Made a Neural Network with just Redstone!</a>: To try everything Brilliant has to offer—free—for a full 30 days, visit https://brilliant.org/mattbatwings You’ll also get 20% off an annual premium subscrip...</li><li><a href="https://archive.org/details/arxiv-bulk?sort=-publicdate">Internet Archive: Digital Library of Free &amp; Borrowable Books, Movies, Music &amp; Wayback Machine</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1243929841565175858)** (1 messages): 

- **Llama 3 German-8B Launched**: Congratulations to Disco Research and the Occi team at TU Darmstadt for their release of a specialized German large language model. **[Llama3-German-8B](https://huggingface.co/DiscoResearch/Llama3-German-8B)**, based on Meta's Llama3-8B, is trained on 65B high-quality German tokens and is now available.

**Link mentioned**: <a href="https://fxtwitter.com/DiscoResearchAI/status/1794351790378594418?t=QCShTixHVItiIr93kUNV1A&s=19">Tweet from DiscoResearch (@DiscoResearchAI)</a>: 🪩 Introducing Llama3-German-8B! A large language model specialized for German, built by @DiscoResearchAI and @occiglot. Based on @Meta&#39;s Llama3-8B, it&#39;s trained on 65B high-quality German tok...

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1243699924533514371)** (9 messages🔥): 

- **Multi-label classification resources shared**: A member seeking help with multi-label classification received a [Kaggle link](https://www.kaggle.com/code/altairfarooque/multi-label-image-classification-cv-3-0) for guidance. The user also shared an [arXiv paper](https://arxiv.org/abs/2405.15738) titled "High-resolution Large Multimodal Models (LMMs)" for additional context.

- **ConvNeXt in high-resolution tasks**: A discussion ensued around the [ConvNeXt paper](https://arxiv.org/abs/2405.15738), highlighting ConvNeXt's role in compressing high-resolution images into rich visual features. The user emphasized that ConvNeXt effectively prevents the generation of excessive visual tokens and suggested optimizations for high-resolution tasks.

- **Model growth for efficient pre-training**: Another [arXiv paper](https://arxiv.org/abs/2405.15319) was shared discussing the obstacles in efficient LLM pre-training via model growth. The paper identifies three critical obstacles and introduces depthwise stacking as a superior growth operator that enhances speed, reduces training loss, and performs well on NLP benchmarks.

- **Diffusers 0.28.0 release announced**: The release of Diffusers 0.28.0 was highlighted, focusing on community integrations and introducing the first non-generative pipeline, **Marigold**, for depth estimation and surface normals prediction. The full release notes can be found on the [GitHub release page](https://github.com/huggingface/diffusers/releases/tag/v0.28.0).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.15738">ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models</a>: High-resolution Large Multimodal Models (LMMs) encounter the challenges of excessive visual tokens and quadratic visual complexity. Current high-resolution LMMs address the quadratic complexity while ...</li><li><a href="https://arxiv.org/abs/2405.15319">Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training</a>: LLMs are computationally expensive to pre-train due to their large scale. Model growth emerges as a promising approach by leveraging smaller models to accelerate the training of larger ones. However, ...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d1u9ma/diffusers_0280_is_here/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1244619607873028219)** (4 messages): 

- **Different loss values due to random seeding**: A member noted experiencing different loss values during reloading, citing issues with random seeding that caused varying training and validation sets. They resolved this by placing training and validation data in separate folders, making the process less confusing.
- **Issue with PyTorch model saving and loading**: Another member acknowledged their problem stemmed from not saving optimizer states in PyTorch, which led to exploding losses. They confirmed the issue was related to the internal states of the Adam optimizer.
  

---



### **DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1243732347610857543)** (1 messages): 

- **Mistral releases finetuning guide for Mixtral models**: Mistral released a finetuning guide specifically for Mixtral models, detailing strategies and tips on effective fine-tuning. Check out the guide on their [GitHub repository](https://github.com/mistralai/mistral-finetune).

**Link mentioned**: <a href="https://github.com/mistralai/mistral-finetune">GitHub - mistralai/mistral-finetune</a>: Contribute to mistralai/mistral-finetune development by creating an account on GitHub.

  

---


### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1243910861668749322)** (31 messages🔥): 

- **Llama3-German-8B model release excites community**: The Llama3-German-8B has been released as a continuation of Meta's Llama3-8B to improve German language capabilities, trained on 65 billion high-quality tokens. *"Benchmark results on our model show minimal degradation in English performance, despite the absence of replay during training."* More information can be found [here](https://huggingface.co/DiscoResearch/Llama3-German-8B).
- **GGUF Quantization issues noted**: A quick GGUF quantization of the model was performed but produced low benchmark scores indicating possible bugs. This quantization is available [here](https://huggingface.co/cstr/Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF).
- **Replay omission raises eyebrows**: The Llama3-German-8B was not pre-trained with English data replay. *"We wanted best German performance, no replay seemed the best option."* Quicksort noted that replay helped boost downstream performance in other languages.
- **Cohere's Aya-23-35B model sparks interest**: **Cohere's Aya-23-35B** was mentioned as a new powerful multilingual model supporting 23 languages but has a restrictive license. [Aya-23-35B details](https://huggingface.co/speakleash/Bielik-7B-v0.1).
- **Shared GPT format dataset**: A conversion of a dataset into ShareGPT format was shared, sparking discussion on translation quality and dataset filtration. The dataset can be accessed [here](https://huggingface.co/datasets/sroecker/aya_german-sharegpt).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/cstr/Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF">cstr/Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/sroecker/aya_german-sharegpt">sroecker/aya_german-sharegpt · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/speakleash/Bielik-7B-v0.1">speakleash/Bielik-7B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://docs.google.com/document/u/0/d/1Cg6W7vdXe4YoeZAB5xFQXAlR_t9dZO-wDJvCvRsCxwQ/mobilebasic">MMLU Annotation Questions + Instructions</a>: no description found</li><li><a href="https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1">RLHFlow/ArmoRM-Llama3-8B-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7530">Tokenizer BPE fixes by jaime-m-p · Pull Request #7530 · ggerganov/llama.cpp</a>: Modifications to make BPE tokenizer match AutoTokenizer. Tested with vocabs from models:   gpt-2  llama-bpe  falcon  deepseek-coder (FAIL: &#39;added_tokens&#39; with len==1)  deepseek-llm (FAIL: &#39...</li><li><a href="https://huggingface.co/DiscoResearch/Llama3-German-8B">DiscoResearch/Llama3-German-8B · Hugging Face</a>: no description found</li><li><a href="https://discord.gg/VaXDGsyebd)">Discord | Your Place to Talk and Hang Out</a>: Discord is the easiest way to talk over voice, video, and text. Talk, chat, hang out, and stay close with your friends and communities.</li><li><a href="https://x.com/DiscoResearchAI/status/1794351790378594418">Tweet from DiscoResearch (@DiscoResearchAI)</a>: 🪩 Introducing Llama3-German-8B! A large language model specialized for German, built by @DiscoResearchAI and @occiglot. Based on @Meta&#39;s Llama3-8B, it&#39;s trained on 65B high-quality German tok...</li><li><a href="https://huggingface.co/collections/DiscoResearch/discoleo-8b-llama3-for-german-6650527496c0fafefd4c9729">DiscoLeo 8B: Llama3 for German - a DiscoResearch Collection</a>: no description found</li><li><a href="https://huggingface.co/collections/DiscoResearch/discoleo-8b-quants-6651bcf8f72c9a37ce485d42">DiscoLeo 8B quants - a DiscoResearch Collection</a>: no description found</li><li><a href="https://huggingface.co/datasets/CohereForAI/aya_collection_language_split/viewer/german/train?f%5Bdataset_name%5D%5Bvalue%5D=%27Xlel_wd-inst%27&row=4018775">CohereForAI/aya_collection_language_split · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1244055944254717952)** (6 messages): 

- **Llama3-DiscoLeo Feedback Highlights Parameter Concerns**: A user shared anecdotal feedback for Llama3-DiscoLeo-Instruct-8B-32k-v0.1 with parameters such as max tokens set to 512 and temperature ranges of 0.5-1. 
- **Discussions on Running Parameters and Output Concerns**: Another member inquired about engine and quantization settings, revealing they used ollama with a normal llama3 template and encountered strange results on EQ Bench.
- **Stop Token Settings and Output Issues**: It was noted that output generated using oobabooga produced uncensored answers and personal data, unlike more sensible output from ollama with q4km gguf. The member also saw similar issues mentioned in another channel.
- **EQ Bench Template Suggestion**: A suggestion was made to use a specific [Llama3 template](https://github.com/CrispStrobe/EQ-Bench/blob/main_v2_3a/instruction-templates/Llama3.yaml) and set custom stopping strings to see if it resolves output issues.
- **Experimentation with Special Tokens in oobabooga**: It was advised to experiment with 'skip_special_tokens': False and possibly 'add_bos_token': True, 'ban_eos_token': False within oobabooga for better performance, although the user admitted to limited familiarity with the tool.

**Link mentioned**: <a href="https://github.com/CrispStrobe/EQ-Bench/blob/main_v2_3a/instruction-templates/Llama3.yaml">EQ-Bench/instruction-templates/Llama3.yaml at main_v2_3a · CrispStrobe/EQ-Bench</a>: A benchmark for emotional intelligence in large language models - CrispStrobe/EQ-Bench

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1243647190690299925)** (21 messages🔥): 

- **Rabbit LAM sparks debate**: Members discussed the efficacy and future of the Large Action Model (LAM), questioning its readiness and potential for reverse engineering. They shared a [Rabbit LAM architecture overview](https://engineering.rabbit.tech/lam-security-architecture-overview) explaining its use of Playwright for secure web app control.
  
- **Running 01 on mobile and Rabbit R1**: Questions arose about running the 01 model on the Humane platform and Rabbit R1 devices. There were related comments about potential hardware issues and goals to integrate Rabbit R1 with Open Interpreter.

- **Storage solutions for model data**: A member asked about overcoming disk space limits on Runpod and sought advice on cost-effective local data storage solutions. Another provided a [GitHub repository](https://github.com/Tonylib/o1_for_flutter) for running 01 on Android, suggesting it could be adapted for Rabbit R1.

- **Open Interpreter installation queries**: One user inquired about the necessity of OpenAI API for installing Open Interpreter. The response clarified that an LLM is needed, which could be local or accessed via cloud proxy services.

- **Markdown export feature**: A new feature was introduced allowing conversations to be exported to Markdown. A [pull request](https://github.com/OpenInterpreter/open-interpreter/pull/1282) was shared detailing the changes.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://engineering.rabbit.tech/lam-security-architecture-overview">overview of LAM security and architecture</a>: no description found</li><li><a href="https://github.com/Tonylib/o1_for_flutter">GitHub - Tonylib/o1_for_flutter</a>: Contribute to Tonylib/o1_for_flutter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1282">Export conversation to markdown by Steve235lab · Pull Request #1282 · OpenInterpreter/open-interpreter</a>: Describe the changes you have made: Add a new magic command %markdown [path] to export the conversation to a specified Markdown path. If no path is provided, it will be saved to the Downloads folde...</li><li><a href="https://youtu.be/zLvFc_24vSM?si=XFsWIzpGDW4IsEEp,">Rabbit Gaslit Me, So I Dug Deeper</a>: Is the LAM a Scam? Down the rabbit hole we go.Jesse’s interview with Jason Calacanis:https://youtu.be/X-MNgciL5hw?si=qh6SgCYtkEiD8UNuPeople who helped this i...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1243697909489340519)** (16 messages🔥): 

- **Struggles with OpenInterpreter on Mac**: A user faced issues with OpenInterpreter on Mac where response is missing after voice recording stops, but resolved it by switching to Python 3.11.
- **Inquiries about Hardware Shipment**: Users are anxious about the shipment status of pre-ordered hardware; some haven't received any updates or emails, while others are curious about European shipping timelines.
- **Integrating O1 with Headphones**: One user is working on a project to install OpenInterpreter on Android-powered modified earbuds, using a Linux distro via [Andronix](https://andronix.app/) to run the software continually.
- **Debating DIY vs. Prebuilt O1**: A discussion highlighted that the primary advantage of buying an O1 over building it yourself is supporting the development team in Seattle, despite the underlying product being the same.
- **Plans for Enhancing the R1 Device**: Some users are dissatisfied with the R1 and are exploring ways to make it useful by connecting it to OpenInterpreter, including rooting and flashing LineageOS on it for better functionality.
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1243807968068370464)** (31 messages🔥): 

- **Struggles with PDF Extraction**: Members discussed the [challenges of extracting text from PDFs](https://python.langchain.com/v0.2/docs/how_to/document_loader_pdf/), particularly with complex tables and diagrams. One advised treating PDFs as images and applying ML for text segmentation, while another suggested using Adobe Extract API for layout parsing.

- **Interest in Building Local LangChain Community**: Karan Singh, Co-Founder of Scogo Networks, expressed interest in creating a LangChain community in Mumbai. He asked for contacts within LangChain's marketing team to discuss organizing local events.

- **Leveraging AWS Bedrock and Pinecone for Chatbot**: A user building a chatbot app with AWS Bedrock and Pinecone faced issues maintaining conversation context. Another member suggested using the [LangChain AgentExecutor](https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/) for better context management and retrieval.

- **Coding and LLM Job Market Discussion**: Users engaged in a debate on whether learning to code is still valuable in the era of LLMs. One member argued that LLMs will likely create more programming jobs, possibly of different kinds.

- **Help with LangChain and Instructor Integration**: A member asked for assistance implementing Instructor with LangChain. No detailed guidance was provided in the chat, highlighting a gap in available help resources for this integration.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.langchain.com/">LangChain</a>: LangChain’s suite of products supports developers along each step of their development journey.</li><li><a href="https://tenor.com/view/matrix-morpheus-stop-trying-hit-me-come-on-gif-20012191">Matrix Morpheus GIF - Matrix Morpheus Stop Trying - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/">Defining Custom Tools | 🦜️🔗 LangChain</a>: When constructing your own agent, you will need to provide it with a list of Tools that it can use. Besides the actual function that is called, the Tool consists of several components:
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1244721736314781756)** (2 messages): 

- **Langserve signup form error**: A member noted difficulties in accessing the waiting list for hosted **Langserve** via Airtable. They encountered an error and inquired about alternative ways to try it out.
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1244189735820328980)** (2 messages): 

- **NLAVIDA Project Launches with YouTube Tutorial**: A member shared an open-source project called NLAVIDA, aimed at performing interactive data visualization and analysis using natural language. Check out the [YouTube video tutorial](https://www.youtube.com/watch?v=leJRP_mJsSQ&t=4s) for more information.
  
- **OranClick Launches on ProductHunt**: A new tool called OranClick, designed to help users write, track, and analyze messages to improve signup rates, was announced. The tool is [available on ProductHunt](https://producthunt.com/posts/oranclick) and users were encouraged to upvote to show support.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://producthunt.com/posts/oranclick"> OranClick - Messages that Click | Product Hunt</a>: Are you struggling to reach your target signups? Write, track, and analyze with ease. All it takes is: - Input messaging and link - Publish message and custom tracking URL - Analyze and Optimize Messa...</li><li><a href="https://www.youtube.com/watch?v=leJRP_mJsSQ&t=4s">NLAVIDA: An Open Source Tool for Interactive Data Analysis and Visualisation using LLM</a>: NLAVIDA (Natural Language-Assisted Visualization and Interactive Data Analysis) is an open source alternative of code interpreter or advanced data analytics ...
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1243731215463219220)** (34 messages🔥): 

- **Networked Llamafile Server Tips**: Members discussed making the llamafile server available across a network with tips like adding `--host <my ip>` or using `--host 0.0.0.0`. This makes the server accessible from different machines on the same network.
- **Unexplained Blank Responses from Llama3-70B**: Users reported blank responses from the llama3-70b model and sought help by sharing logs for troubleshooting. Another user stepped in but didn't have a direct solution, indicating it might require deeper investigation.
- **Release of Llamafile v0.8.5 and Benchmarks**: The community celebrated the release of llamafile version 0.8.5, highlighting that it now offers fast inference for K quants on X86 CPUs. Members were encouraged to join a benchmarking club to test and share results using `llamafile-bench`.
- **Home Assistant Integration Wish List**: Home Assistant integration feedback highlighted the need for a standardized local API similar to OpenAI’s, suggesting names like Apilla and noting features like API discoverability via DNS-SD/zeroconf and secure APIs as desirable.
- **Model Selection in Python Example**: Questions arose regarding specifying models in the Python example for LLaMA_CPP integration, with users sharing snippets and seeking clarity on whether model specification is necessary when running instances like TinyLlama.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.home-assistant.io>)">no title found</a>: no description found</li><li><a href="http://<Your">no title found</a>: no description found</li><li><a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-gguf">openbmb/MiniCPM-Llama3-V-2_5-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile#prompting">Mozilla/Meta-Llama-3-8B-Instruct-llamafile · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.3-llamafile">Mozilla/Mistral-7B-Instruct-v0.3-llamafile · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Mozilla/granite-34b-code-instruct-llamafile">Mozilla/granite-34b-code-instruct-llamafile · Hugging Face</a>: no description found</li><li><a href="https://x.com/JustineTunney/status/1794732286900043830">Tweet from Justine Tunney (@JustineTunney)</a>: It turns out patience is all you need. We&#39;re now able to run Mixtral 8x22b Q6_K on a $362 CPU with better than human reading speed. https://github.com/Mozilla-Ocho/llamafile/discussions/450</li><li><a href="https://ollama.com/yabi/minicpm-llama3-v-2_5">yabi/minicpm-llama3-v-2_5</a>: Get up and running with large language models.</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.5">Release llamafile v0.8.5 · Mozilla-Ocho/llamafile</a>: This release fixes bugs and introduces @Kawrakow&#39;s latest quant performance enhancements (a feature exclusive to llamafile). As of #435 the K quants now go consistently 2x faster than llama.cpp up...</li><li><a href="https://github.com/Mozilla-Ocho/llamafile/discussions">Mozilla-Ocho/llamafile · Discussions</a>: Explore the GitHub Discussions forum for Mozilla-Ocho llamafile. Discuss code, ask questions &amp; collaborate with the developer community.
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1243666066530369661)** (18 messages🔥): 

- **Mistral-Finetune Repository Spotted**: Some users discussed the [Mistral-Finetune repository](https://github.com/mistralai/mistral-finetune), though one user was unsure what the changes entailed. *"Saw this, but wasn't sure what was different."*
- **Performance Variance in MoEs Tuning**: A discussion on fine-tuning Mixture of Experts (MoEs) models noted *"a high variance in performance"* and suggested *"running multiple instances"* of the fine-tuning process to select the best-performing model instance.
- **Aya 23 Model's Use Case Limitations**: Users debated Aya 23’s effectiveness, pointing out it's *"not optimized for chat mode use"* and only beneficial for *"very specific use cases"* like multilingual instruction-following, per their [technical report](https://cohere.com/research/papers/aya-command-23-8b-and-35b-technical-report-2024-05-23).
- **Best Open Japanese-Enabled Models**: LHL stated [Command-R Plus](https://huggingface.co/shisa-ai/shisa-v1-llama3-70b) is the top non-commercial model, while Shisa-V1-Llama3-70B was noted as a strong *commercially usable open model*. Other noteworthy mentions include Claude Haiku and Gemini 1.5 Flash for their affordability.
- **Introduction of MoRA for Fine-Tuning**: The [MoRA method](https://arxiv.org/pdf/2405.12130) was introduced as an advancement over LoRA, using a high-rank updating approach while maintaining the same number of trainable parameters. The GitHub repository for MoRA is available [here](https://github.com/kongds/MoRA).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/research/papers/aya-command-23-8b-and-35b-technical-report-2024-05-23">Aya 23: Open Weight Releases to Further Multilingual Progress</a>: no description found</li><li><a href="https://huggingface.co/shisa-ai/shisa-v1-llama3-70b">shisa-ai/shisa-v1-llama3-70b · Hugging Face</a>: no description found</li><li><a href="https://github.com/mistralai/mistral-finetune">GitHub - mistralai/mistral-finetune</a>: Contribute to mistralai/mistral-finetune development by creating an account on GitHub.</li><li><a href="https://github.com/kongds/MoRA">GitHub - kongds/MoRA: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning</a>: MoRA: High-Rank Updating for Parameter-Efﬁcient Fine-Tuning - kongds/MoRA
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1243702512519876720)** (3 messages): 

- **FFD Bin Packing Issue in Main**: A member highlighted that the "FFD bin packing implementation in main" appears broken with distributed training. Specifically, they noted that "the length estimation (from packing efficiency) is not there."

- **Debug Confirmations**: Another member confirmed that there are "definitely some issues" needing debugging in the FFD bin packing implementation mentioned.

- **Unsloth's Llama 3 Fixes**: Users discussed recent fixes by unsloth concerning Llama 3, indicating that the base version has "some untrained tokens". They also shared a [patch for sfttrainer](https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L722) to check/remove double `bos_tokens`.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L557">unsloth/unsloth/tokenizer_utils.py at main · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L722">unsloth/unsloth/tokenizer_utils.py at main · unslothai/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1243920733261009016)** (8 messages🔥): 

- **Typo confusion over `conversations` vs. `conversation`**: A member pointed out a discrepancy in the [documentation](https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/README.md?plain=1#L373) regarding whether the key should be `conversations` or `conversation`. This was resolved when another member confirmed the correct usage by checking the code.

- **Benchmark request for fine-tuning methods**: A member inquired about benchmarks comparing the performance of fine-tuning with "sequence packing" versus "naive padding" specifically for MQC tasks like MMLU. They shared their own experience, noting significant performance differences between packed and non-packed models.

- **Fine-tuning Mistral 7b inst v2 observations**: After modifying the prompt format and fine-tuning Mistral 7b inst v2, a member observed that while responses improved, the model seemed to hallucinate. They noted a significant loss drop during training and questioned if it could be data related, given their data structure.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a">GitHub - OpenAccess-AI-Collective/axolotl at 8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2/README.md?plain=1#L373">axolotl/README.md at 8a20a7b711a62d7b04e742f3d6034b4ca8aa27d2 · OpenAccess-AI-Collective/axolotl</a>: Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/)** (1 messages): 

nanobitz: nice!
  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1243808010942283786)** (3 messages): 

- **Discover the Virtual Beings Summit for AI and Simulations**: A member shared the [Virtual Beings Summit](https://www.virtual-beings-summit.com), highlighting AI founders, Stanford researchers, and Virtual Beings startup leaders. The event promises panels on "AI & Simulations" and features speakers like **Will Wright**, the creator of The Sims.
- **Explore AI in Action on YouTube**: The YouTube channel [Virtual Beings](https://www.youtube.com/channel/UCLNalUbhs_EYB5IAbw1VfaA) was shared, emphasizing content likely focused on AI and interactive simulations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.virtual-beings-summit.com">Virtual Beings Summit</a>: no description found</li><li><a href="https://www.youtube.com/channel/UCLNalUbhs_EYB5IAbw1VfaA">Virtual Beings</a>: no description found
</li>
</ul>

</div>
  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1243664052929171536)** (2 messages): 

- **DIAMOND Project Highlighted**: A member shared a link to the [DIAMOND GitHub repository](https://github.com/eloialonso/diamond), which stands for **"DIffusion As a Model Of eNvironment Dreams."** This project is described as a *"reinforcement learning agent trained in a diffusion world model."*

**Link mentioned**: <a href="https://github.com/eloialonso/diamond">GitHub - eloialonso/diamond: DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a reinforcement learning agent trained in a diffusion world model.</a>: DIAMOND (DIffusion As a Model Of eNvironment Dreams) is a reinforcement learning agent trained in a diffusion world model. - eloialonso/diamond

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-dev](https://discord.com/channels/1122748573000409160/1137456826733047908/1243738151898316930)** (6 messages): 

- **AI Town and UE5 In Vision of Westworld**: One member suggested that combining AI Town with **UE5** and **voice control interactions** would create a "Westworld" experience. Another member responded enthusiastically, *"we are working on this at 4Wall!"*.
- **VR Integration Excited Members**: There was excitement about the potential of linking AI Town to **VR** for more immersive communication. One member eagerly asked, *"Isn't it possible to link to VR again?"*.
- **Stay Updated via 4Wall Discord**: For those interested in following the progress on AI Town and its integrations, the recommendation was to join the **4Wall Discord**. Member provided the link [discord.gg/vPum4s3h](https://discord.gg/vPum4s3h) to stay in the loop.



**Link mentioned**: <a href="https://discord.gg/vPum4s3h">Join the 4Wall AI Discord Server!</a>: Check out the 4Wall AI community on Discord - hang out with 511 other members and enjoy free voice and text chat.

  

---


### **AI Stack Devs (Yoko Li) ▷ #[late-night-lounge](https://discord.com/channels/1122748573000409160/1159342774710186075/1243968582820429918)** (8 messages🔥): 

- **SadTalker GitHub repo explored for animations**: A member shared the [SadTalker GitHub repository](https://github.com/OpenTalker/SadTalker) for "learning realistic 3D motion coefficients for stylized audio-driven single-image talking face animation". They encouraged others to reach out if they needed help with animations involving temporal consistency.
- **Hugging Face potential solution for running SadTalker**: The same member suggested checking Hugging Face Spaces for running the SadTalker repo locally, expressing high confidence that a solution exists there.
- **New animation tool released by Tencent**: Another repository, [V-Express from Tencent AI Lab](https://github.com/tencent-ailab/V-Express), which aims to generate talking head videos, was shared. It was noted for being a recently released tool for creating engaging avatars.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tence">tence - Overview</a>: tence has 11 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/tencent-ailab/V-Express">GitHub - tencent-ailab/V-Express: V-Express aims to generate a talking head video under the control of a reference image, an audio, and a sequence of V-Kps images.</a>: V-Express aims to generate a talking head video under the control of a reference image, an audio, and a sequence of V-Kps images. - tencent-ailab/V-Express</li><li><a href="https://github.com/OpenTalker/SadTalker">GitHub - OpenTalker/SadTalker: [CVPR 2023] SadTalker：Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation</a>: [CVPR 2023] SadTalker：Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation - OpenTalker/SadTalker
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1243662467427598427)** (4 messages): 

- **Zyphra Zamba makes a quiet debut**: Zyphra Zamba, the mamba/attention hybrid, has been released with minimal fanfare. Key resources include [the tech report](https://www.zyphra.com/s/Zamba.pdf), [Torch reference code](https://github.com/Zyphra/Zamba-torch), and [HF Transformers code](https://github.com/huggingface/transformers/pull/30950).

- **Comparison with OLMo 1.7 underway**: It's mentioned that Zyphra Zamba will be compared to OLMo 1.7 for evaluations.

- **SD Audio 2.0 leak**: SD Audio 2.0 has reportedly been leaked on 4chan. It is available on an unspecified HF account and is fairly easy to locate.

**Link mentioned**: <a href="https://x.com/QuentinAnthon15/status/1794084824464158745">Tweet from Quentin Anthony (@QuentinAnthon15)</a>: @philpax @ryu0000000001 We&#39;ve decided announcing mid-day right before a long weekend might be a bad idea ;)  For those looking for the model + tech report now, here&#39;s the relevant info: - Tech...

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1244433887845814352)** (4 messages): 

- **Ex-OpenAI board members call for AI regulation**: Hellen Toner and Tasha McCauley argue in [The Economist](https://www.economist.com/by-invitation/2024/05/26/ai-firms-mustnt-govern-themselves-say-ex-members-of-openais-board) that **private AI companies** cannot be trusted to self-regulate due to profit incentives. They highlight issues like the dismissal of Sam Altman and internal safety protocol concerns.
- **Altman’s toxic culture criticized**: The ex-board members allege that Sam Altman cultivated “a toxic culture of lying” and engaged in “psychological abuse”. Despite an internal investigation concluding that these did not mandate his removal, the details were not shared publicly, causing controversy.

**Link mentioned**: <a href="https://www.economist.com/by-invitation/2024/05/26/ai-firms-mustnt-govern-themselves-say-ex-members-of-openais-board">AI firms mustn’t govern themselves, say ex-members of OpenAI’s board</a>: For humanity’s sake, regulation is needed to tame market forces, argue Helen Toner and Tasha McCauley

  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1243676921498439730)** (10 messages🔥): 

- **224n Class Timeline is Unclear:** A member expressed interest in when the new 224n class will be posted. Another member indicated that it typically happens at the end of the semester, but couldn't be certain and mentioned they could ask Chris.
- **Reinforcement Learning Textbook Launched:** A link to a [GitHub repository](https://github.com/natolambert/rlhf-book) was shared, announcing a textbook on reinforcement learning from human feedback. One user inquired if it will cover operational aspects of human preference gathering; the answer specified it primarily focuses on technical details with the latter half discussing recent research projects.
- **Chris Potts and Chris Manning Praised:** Members discussed the demeanor of Chris Manning and Chris Potts. Chris Manning was affectionately described as a "delightful mega nerd," while Chris Potts was praised for his humor and lecture style.

**Link mentioned**: <a href="https://github.com/natolambert/rlhf-book">GitHub - natolambert/rlhf-book: Textbook on reinforcement learning from human feedback</a>: Textbook on reinforcement learning from human feedback - natolambert/rlhf-book

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1243814257662099506)** (11 messages🔥): 

- **Debate over test time limit extension**: A member questioned if extending the per-test time limit (currently 9 minutes 34 seconds) is possible, stating that without it, "Taylor approximations" for functions might not be feasible. They added that while `sin` is manageable, others are not finishing, like `clang`, which only reaches ~60%.

- **Struggles with large expressions**: A user mentioned working on a solution that might work but faces issues due to generating exceedingly large single expressions that crash during compilation. The error message they received was *"invalid operands to binary expression ('double' and 'double')"*.

- **Switching focus to other bounties**: After analyzing the difficulty and challenges many are facing with this specific task, one user decided to attempt other bounties, emphasizing that the superficially easy-looking ones have deep complexities.

- **Doubles and bitwise operations are incompatible**: A member clarified that one cannot perform bitwise operations like XOR on `double` types, addressing the compilation error discussed.

- **Interest in bounties and stale pull requests**: New users inquired about interesting research-oriented bounties and the status of older bounties. Specifically, a PR on [tinygrad](https://github.com/tinygrad/tinygrad/pull/4212) was mentioned, leading to confirmation from **georgehotz** that the bounty is still up for grabs.
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1244317608640118955)** (5 messages): 

- **UOp class 'vin' clarification**: A member asked about the meaning of the 'vin' variable in the UOp class, questioning whether it stood for 'vector input' or 'value input.' George Hotz clarified that 'vin' likely has no particular meaning, noting, *"you just can't use 'in'."*

- **Taylor Approximation Feedback Request**: A member shared their GitHub pull request for Taylor Approximation at [tinygrad pull request #4739](https://github.com/tinygrad/tinygrad/pull/4739) and asked for feedback to ensure they were on the right track. The pull request was described as a simple proof of concept.

- **Post dominator analysis for scheduling**: A member questioned why post dominator analysis isn't used during scheduling to find self-contained subgraphs for fusion. This inquiry aimed to explore potential optimizations in the model's scheduling process.

**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/pull/4739/files,">Taylor Approximation (at about 0) for exponential proof of concept by mesozoic-egg · Pull Request #4739 · tinygrad/tinygrad</a>: Simple POC, just want to know if I&#39;m on the right track

  

---



---



---



---



---




{% else %}




## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

**Fine-Tuning Facts**: Discussion on fine-tuning in the [general channel](https://discord.com/channels/1238365980128706560/1238365980128706563/1243282801760145408) revealed a concern about **semantic similarity overfitting** due to biased data categories. A user struggled with understanding fine-tuning vis-à-vis user inputs and initial model training. Changes in the **OpenAI platform's sidebars** were also noted with the disappearance of two icons (threads and messages).

**Templates Take the Spotlight**: In [workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1243336501018755123), the importance of configuring templates correctly during fine-tuning was highlighted. In particular, the delimiter `###` aids in parsing different input sections, and "end of text" tokens indicate when to stop token generation.

**Maven Mingles with Moderation**: In [asia-tz](https://discord.com/channels/1238365980128706560/1240532179549945957/1243344778511515698), a light-hearted exchange between members referenced a reunion. A request for a conference talk recording was met, with the video being available on Maven.

**Modal Mobilization**: Modal users in [🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1243309176722030702) shared excitement over received credits, training experiences, and provided specific links to **Modal documentation** and **examples** for new users. A plan to use Modal for a **Kaggle competition** was also shared, including setup and execution details.

**Jarvis Jots Down Jupyter Jumble**: In the [jarvis-labs channel](https://discord.com/channels/1238365980128706560/1241117895740625099/1243307629057671229), members discussed storing a VSCode repo on Jarvis with a suggestion to use GitHub for saving work. There was a notice of **spot instance removal** due to instability. The cost and duration of fine-tuning the **open-lama-3b** model were shared, and a user resolved an Ampere series error by adjusting model parameters.

**Hugging Face Huddles on Credits & Spanish Models**: The [hugging-face channel](https://discord.com/channels/1238365980128706560/1241141471814488115/1243335428887806004) saw discussions about pending **HF credits** and models suitable for Spanish text generation—with **Mistral 7B** and **Llama 3** models being recommended.

**Credit Countdown Carries On** in [replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1243453712182149150), where an upcoming announcement related to credit management and distribution was teased.

**Corbitt's Commandments Claim Clout**: Enthusiastic attendees in the [kylecorbitt_prompt_to_model channel](https://discord.com/channels/1238365980128706560/1242221891733946490/1243287896652517376) discussed fine-tuning methods and techniques presented in Kyle Corbitt's talk, including *[Ten Commandments for Deploying Fine-Tuned Models](https://docs.google.com/presentation/d/1IIRrTED0w716OsU_-PL5bONL0Pq_7E8alewvcJO1BCE/edit#slide=id.g2721fb6713e_0_67)*.

**Axolotl Answers the Call** in [workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1243277523316637817), where users discussed datasets, model training, and troubleshooting in Axolotl. A blog post on **TinyLLama Fine-Tuning** was shared, and there was a push for integrating observability into LLM applications.

**Zoom Out, Discord In**: Users from [workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/1243339106675724369) migrated their discussions to Discord after the Zoom chat was disabled.

**Axolotl's Cache Conundrum Causes Confusion**: Issues with cache in Axolotl frustrating users and confusion with missing files were resolved in [axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1243286083022618664). Discussions on sample packing and a guide on tokenizer gotchas addressed concerns around efficiency and tokenization.

**Accelerate to Victory**: [zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1243291846415749283) saw users work through confusion over float comparisons, resolve Jarvislab training command errors, and exchange resources for learning model acceleration with a focus on fine-tuning best practices.

**Winging It with Axolotl**: The [wing-axolotl channel](https://discord.com/channels/1238365980128706560/1242564077151326388/1243305377974587412) collaborated on dataset templates, pre-processing issues, Axolotl configurations, and provided a PR merge for the latest Axolotl updates. They delved into debugging tools and the significance of precise templates for training success.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Protein Data Visuals Reach New Heights**: A new protein visualization project now sports 3D rendering and includes examples for human hemoglobin and ribosomal proteins, with the project details found on [GitHub](https://github.com/AstraBert/proteinviz/blob/main/examples.md).

**Enter the TranscriptZone with OpenAI's Whisper**: A new transcription app that leverages OpenAI's Whisper to transcribe YouTube videos and more is available at [Hugging Face Spaces](https://huggingface.co/spaces/tensorkelechi/vidtext).

**Decentralizing the Web - More than a Dream?**: A project building infrastructure for a decentralized internet sought community feedback through a survey, raising discussions about the ethics of data collection.

**A Vision Transformers Query in Depth**: A member sought resources on applying Vision Transformers (ViT) for monocular depth estimation, indicating an intent to develop a model using ViT, but no specific resources were provided in the discussion.

**Quantisation Quandary for Mistral Model**: The use of **bitsandbytes** for 8-bit quantisation on **Mistral v0.3 Instruct** led to slower performance compared to 4-bit and fp16, a baffling outcome that contradicts expected efficiency gains from reduced-bit computation.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Climbs Over ChatGPT in CSV Showdown**: Engineers discussed that **Perplexity AI** outshines **ChatGPT** in CSV file processing by allowing direct CSV uploads. Also, **Julius AI** was recommended for data analysis, leveraging Python and integration with LLMs like **Claude 3** or **GPT-4**.

- **Users Snub Claude 3 Opus**: **Claude 3 Opus** is getting the cold shoulder due to increased content restrictions and perceived diminished utility, with **GPT-4** posed as a preferable option despite limitations.

- **Querying Pro Search's True Upgrade**: Upgrades to **Pro Search** raised eyebrows as users discussed whether new multi-step reasoning features and API specs were genuine backend improvements or merely surface-level UI enhancements.

- **API Integration Articulated**: Dialogue around API integration for external tools with **Claude** generated interest along with sharing of custom function calls, serverless backends, and documentation like [Tool Use with Claude](https://docs.anthropic.com/en/docs/tool-use).

- **Ethics in AI: More Than a Thought Experiment**: Discourse on infusing GPTs with ethical monitoring capabilities sparked, casting light on potential applications in workplace communication and legal defensibility, albeit with philosophical wrinkles yet to be ironed out.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Speculation Peaks on RTX 5090's VRAM**: There's buzzing debate over whether the rumored **RTX 5090 with 32GB VRAM** makes practical sense. References were made to potential specs and images on [PC Games Hardware](https://www.pcgameshardware.de/Grafikkarten-Grafikkarte-97980/News/Geforce-RTX-5090-mit-32-GiB-und-drei-PCBs-1448073/galerie/3884555/), but some members remained skeptical about its authenticity.

- **Stable Diffusion and the AMD Challenge**: Users offered guidance on installing **Stable Diffusion** on an AMD 5700XT GPU, suggesting that starting with web services like [Craiyon](https://www.craiyon.com/) may circumvent potential compatibility issues.

- **Stable Diffusion 3: Trial Before Commitment**: The community contrasted **Stable Diffusion 3** with competitor Midjourney, highlighting that while a free trial is available for SD3, ongoing access would require a **Stability** membership.

- **Anticipation Builds Around Mobius Model**: An announcement concerning DataPlusEngine’s novel **Mobius model** has garnered significant interest for its claim to create efficient base models. The model, teased on [Twitter](https://x.com/DataPlusEngine/status/1793803117642854732), is neither a straightforward base model nor a tuned version of something pre-existing.

- **32GB VRAM: Game Changer or Overkill?**: The mention of a 32GB VRAM GPU led to conversations about the potential shift in Nvidia's approach to data center GPU sales, considering how products with substantial memory could impact the market demand for the H100/A100 series.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **PEFT Config Snag Solved**: An issue where `config.json` was missing during PEFT training was resolved by copying it from the base model's configuration, with the user confirming success.

- **Llama Levitates Above Bugs**: The **Llama 3** model's base weights were described as "buggy," but Unsloth has implemented fixes. To improve training, the use of reserved tokens and updates to the tokenizer and `lm_head` are recommended.

- **System Prompt Boosts Llama 3**: Incorporating a system prompt, even a blank one, was observed to enhance Llama3 finetuning outcomes.

- **Phi 3 Proliferation**: Excitement bubbled as **Phi 3 models** debuted, sporting medium support. Community chatter pointed engineers toward extensive details in blog posts and release notes.

- **Stable Diffusion's Sinister Side Show**: Creepy artifacts and uncanny voice cloning outputs from **Stable Diffusion** startled users, with discussions and experiences shared via YouTube videos and a Reddit thread.

- **VSCode Copilot Climbing Onboard**: Recommendations for a local VSCode "copilot" were sought and met with suggestions and positive responses in the **random** channel.

- **Inference Inertia with Phi-3**: Slower inference times using **Unsloth Phi-3** puzzled one user, who provided a [Colab notebook](https://colab.research.google.com/drive/1LLWoaQrH8KFkQlE4ONwwtC4tC1-1It2X) to investigate the lag, with community efforts yet to find a fix.

- **Quantization Quandary Unraveled**: A member faced challenges quantizing a custom model, hitting walls with **llama.cpp** and **Docker** compatibility, sparking a discussion on solutions.

- **VRAM Verdict for Model Might**: VRAM requirements were laid out: **12GB for Phi 3 mini** is okay, but **16GB is a must for Phi 3 medium**. For hefty tasks, considering outside computing resources was proposed.

- **Data Diligence for Training Consistency**: The importance of using consistent datasets for training and evaluation was echoed, highlighting **Unslothai's public datasets** like the [Blackhole Collection](https://huggingface.co/collections/lamhieu/blackhole-66473b7feec034b4fb70818a).

- **Platform Possibilities and Cautions**: Queries regarding **Unsloth** support for older Macs were addressed, confirming a focus on CUDA and GPU usage, with suggestions for those on CPU-only rigs.

- **Enterprise Expertise Extension**: A community member stepped forward to offer enterprise expertise to Unsloth, hailing the joining of accelerators at Build Club and Github, hinting at synergistic potential for Unsloth's endeavors.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**Intellectual Debate Ignites Over AI Understanding**: In-depth discussions were had about the true **understanding** of concepts by LLMs, with **interpretability research** considered important empirical evidence. Skeptics argued that current efforts are lacking, with references to work by **Anthropic** on mapping large language model minds.

**The Creature from the Llama Lagoon**: A technical foray into enhancing **Llama models** centered around crafting a script that could manage **function calls**, with **Hermes Pro 2**'s approach serving as inspiration. Another inquiry circled the implementation of **Llama3 LoRA** techniques on a 3080 GPU.

**Reality Quest in Digital Dimensions**: Spearheading a conversation on **Nous and WorldSim**, members explored the possible applications of **NightCafe** and multi-dimensional AR spaces in mapping complex AI worlds. Dream-like explorations in **audio-visualizers** and whimsical **ASCII art** representations highlighted creative uses for AI-driven simulations.

**Sifting Through RAG Data**: Advocation for models to **integrate internal knowledge** with **Retrieval-Augmented Generation (RAG)** was a hot topic, with questions raised about how to handle contradictions and resolve conflicts. Emphasizing user evaluations was seen as essential, particularly for complex query cases.

**Precision over Pixie Dust in Fine-Tuning AI**: The community's discourse featured a celebration of the **Mobius model** for its prowess in **image generation**, with anticipation for an open-sourced version and elucidating publications. Additionally, Hugging Face was mentioned for their `PyTorchModelHubMixin` enabling easier model sharing, though limited by a **50GB size constraint** without sharding.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX vs. PyTorch/XLA: The TPU Showdown**: The performance comparison of **JAX** and **PyTorch/XLA** on TPUs spurred debate over benchmarking nuances such as **warmup times** and **blocking factors**. The dramatic decline in GPT-3 training costs from **$4.5M to an estimated $125K-$1M by 2024** was highlighted, considering **TFLOP rates** and **GPU-hour pricing** from various contributors, linking to a [Databricks Blog Post](https://www.databricks.com/blog/turbocharged-training-optimizing-databricks-mosaic-ai-stack-fp8).

- **Scaling and Teaching LLMs**: In the research forum, the **Chameleon** model was noted for its strong performance in multimodal tasks, while **Bitune** promised improvements in zero-shot performance for LLMs ([Bitune Paper](https://arxiv.org/pdf/2405.14862)). Discussions questioned the scalability of the **JEPA** model for AGI and critiqued **RoPE's** context length limitations, referencing a relevant [paper](https://arxiv.org/pdf/2405.14591).

- **Emergent Features Puzzle LLM Enthusiasts**: Tim Dettmers' research on advanced quantization methods maintaining performance in transformer inference was linked, including his concept of emergent outliers, and its integration with Hugging Face via the [bitsandbytes library](https://huggingface.co/blog/hf-bitsandbytes-integration). Discourse on emergent features coalescing around ideas of them being the "DNA" of a model, driving discussions on its implications for phase transitions.

- **A Brief on Technical Tweaks & LM Evaluation**: Within the **lm-thunderdome**, engineers covered practical tips for setting seeds in **vllm models**, retrieving the **list of tasks** with `lm_eval --tasks list`, and handling changes in **BigBench** task names that affect harnesses like Accelerate with memory issues. It was suggested to locate tasks by perusing the `lm-eval/tasks` folder for better organization.

- **A Call for Collaboration**: An appeal was made for expanding the **Open Empathic** project, with a [YouTube guide](https://youtu.be/GZqYr8_Q7DE) for contributing movie scenes and a link to the project shared. Further collaboration was encouraged, underlining the need for community efforts in enhancement.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**GPU Adventures**: Engineers discussed challenges when loading small models onto GPUs, with some favoring models like *llama3, mistral instruct,* and *cmdrib*. Meanwhile, using lower quantizations, such as *llamas q4*, reportedly yielded better results than higher ones like q8 for certain applications, refuting the notion that "bigger is always better."

**Next-Gen Models Incoming**: An update in the model realm informed about the release of a **35B model**, with testing to ensure LM Studio compatibility. Optimizations for different scales of models were a topic too, with a focus on **Phi-3 small GGUFs** and their efficiency.

**Servers and Setups**: Hardware discussions included leveraging **distributed inference** with **llama.cpp** and its recent RPC update, although quantized models aren't supported yet. Experimental builds using clustered cheap PCs with **RTX 4060 Ti 16GB** for distributed model setups and possible network constraints were also explored.

**Multilingual Cohesion Achieved**: Cohere models now extend their prowess to **23 languages**, as advertised with **aya-23 quants** available for download, but ROCm users must await an update to dive in. 

**Stable Diffusion Left Out**: LM Studio clarified that it exclusively handles language models, excluding image generators like Stable Diffusion, alongside dealing with CUDA issues on older GPUs and promoting services like **Julius AI** to ease user experience woes.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Gradient Norm Nuisance**: Altering the batch size from 32 leads to a sudden spike in gradient norm, disrupting training. A [pull request](https://github.com/karpathy/llm.c/pull/456) resolved this issue by preventing indexing overflow in the fused classifier.
  
- **Int4 and Uint4 Types Need Some TLC**: A member flagged that many functions lack implementations for **int4** and **uint4** data types in PyTorch, with a [discussion thread](https://dev-discuss.pytorch.org/t/supporting-new-dtypes-in-pytorch/1833) indicating limitations on type promotion and tensor operations.

- **Live Code Alert – Scan Algorithm in Spotlight**: Izzat El Hajj will lead a live coding session on the Scan algorithm, vital for ML algorithms like Mamba, scheduled for `<t:1716663600:F>`, promising to be a technical deep dive for enthusiasts.

- **CUB Library Queries and CUDA Nuances**: Members tapped into discussions ranging from the functioning of CUDA CUB library code to triggering tensor cores without cuBLAS or cuDNN, highlighting resources like [NVIDIA's CUTLASS GitHub repository](https://github.com/NVIDIA/cutlass/tree/main) and the [NVIDIA PTX manual](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

- **FineWeb Dataset Conundrum**: Processing the FineWeb dataset can be a storage hog, hitting 70 GB on disk and gobbling up to 64 GB of RAM, hinting at a need for better optimization or more robust hardware configurations for data processing tasks.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Python Libraries Cling to C Over Mojo**: There's a lively conversation about the feasibility and preparedness of porting Python libraries to Mojo, with concerns about pushing maintainers too hard given Mojo's evolving API. Members discussed whether targeting C libraries might be a more immediate and practical endeavor.

**Rust's Security Appeal Doesn't Rust Mojo's Potential**: Mojo is not slated to replace C, but the security benefits of Rust are influencing how engineers think about Mojo's application in different scenarios. Ongoing discussions address concepts from Rust that could benefit Mojo developments.

**Blazing Ahead With Nightly Mojo**: BlazeSeq performance on MacOS using Night versions of Mojo shows promising similarity to Rust's Needletail, fueling cross-platform efficiency discussions. Rapid nightly updates, noted in [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md), keep the community engaged with the evolving language.

**Curiosity Sparks Over Modular Bot's Machinery**: Queries were raised about the underlying tech of "ModularBot", and although no specific model was referenced, the bot shared a colorful reply. Separately, the potential for ML model training and inference within Mojo was discussed, with mention of Max Engine as a numpy alternative, though no full-fledged training framework is on the horizon.

**Compile-Time Confusion and Alignment Woes**: Problems from aligning boolean values in memory to compile-time function issues are causing a stir among users, with workarounds and official [bug reports](https://github.com/modularml/mojo/issues/2813) highlighting the importance of community-driven troubleshooting.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **LaTeX Loyalist LLM**: In the realm of formatting, users noted frustration with GPT's strong inclination to default to LaTeX despite requests for Typst code, revealing preferences in coding syntax that the LLM seems to adhere to.
  
- **Microsoft Copilot+ vs. Leonardo Rivalry**: Conversations in the community centered on the value of Microsoft Copilot+ PCs for creative tasks like "sketch to image," while some members encouraged checking out [Leonardo.ai](https://leonardo.ai) for analogous capabilities.

- **A Thirst for Efficiency in AI**: Concern was voiced over the environmental toll of AI, citing a [Gizmodo article](https://gizmodo.com/chatgpt-ai-water-185000-gallons-training-nuclear-1850324249) on the substantial water usage during the training of AI models, prompting discussions on the need for more eco-friendly AI practices.

- **Iteration Over Innovation**: There was active dialogue on enhancing the performance of LLMs through iterative refinement, with references to projects like AutoGPT addressing iterations, despite the associated higher costs.

- **Intelligence Infusion Offer Overstated?**: The guild pondered the plausibility and potential of embedding legal knowledge within ChatGPT, enough to consider a valuation at $650 million, though detailed perspectives on this bold assertion were limited.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**LangChain CSV Agent Deep Dive**: Engineers explored **LangChain's CSV agent** within a **SequentialChain** and discussed [how to customize output keys](https://python.langchain.com/docs/modules/chains/foundational/sequential_chains) like `csv_response`. Challenges with SQL agents handling multi-table queries were mentioned, pointing towards token limits and LLM compatibility issues, with direction to GitHub [for issues](https://github.com/langchain-ai/langchain/issues).

**AI Showcases Gather Buzz**: [OranAITech tweeted](https://twitter.com/OranAITech/status/1793684085056942412?t=AVjC2GpAdrT-LqwMEzv0nQ&s=19) their latest AI tech, while **everything-ai v2.0.0** announced features including audio and video processing capabilities with a [repository](https://github.com/AstraBert/everything-ai) and [documentation](https://astrabert.github.io/everything-ai/) available.

**Demystifying VisualAgents**: Demonstrations of **Visual Agents platform** were shared via YouTube, revealing its potential to streamline SQL agent creation and building simple retrieval systems without coding, utilizing LangChain's capabilities. Two specific videos showcased their workflows: [SQL Agent](https://youtu.be/_3crxBzVg3A?si=r2rDA19q-fHm7h9N) and [Simple Retrieval](https://youtu.be/prOjBQQgKlU?si=jDt53koCl6lT6BoM).

**EDA GPT Impressions On Display**: A demonstration of **EDA GPT**, including a five-minute overview video showcasing its various functions, was linked to via [LOVO AI](https://genny.lovo.ai/share/d6b58f0d-fc46-4aa7-a65e-fa0f9a684f01). The demo highlights the AI tool's versatility.

**Tutorial Teaser**: A message in the tutorials channel provided a [YouTube link](https://youtu.be/gflsu_6R_8g) to business24.ai's content, although the context of its relevance was not disclosed.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Piracy's Not the Panacea**: Despite a humorous suggestion that The Pirate Bay could become a haven for sharing AI model weights, skepticism among members arises, highlighting the potential for friendlier AI policy landscapes in other nations to prevail instead.

- **Japan Takes the AI High Road**: Participants noted Japan's encouraging position on AI development, referencing a **paper** shared via a [tweet](https://x.com/DataPlusEngine/status/1793817514956259460) about creating new base diffusion models without the need for extensive pretraining, showcasing a strategy involving temporary disruption of model associations. 

- **Poisoned Recovery Protocols Probed**: A **collaborative study**, involving a poisoned model recovery method conducted by fal.ai, was mentioned, with findings expected to empirically substantiate the recovery approach. Reservations were expressed regarding the aesthetics of AI-generated imagery, specifically the "high contrast look" and artifacts presented by models like Mobius versus predecessors such as MJv6.

- **Claude Mappings Crack the Code**: Anthropic's **research paper** details the dissection of Claude 3 Sonnet's neural landscape, which illustrates the manipulation of conceptual activations and can be read at their [research page](https://www.anthropic.com/research/mapping-mind-language-model). Debates sparked over the potential commercialization of such activations, with a juxtaposed fear of the commercial implications driving AI practitioners to frustration.

- **A Nostalgic Look at AI's Visual Visions**: A member reminisced about the evolution from early AI visual models like Inception v1 to today's sophisticated systems, recognizing DeepDream’s role in understanding neural functionality. Furthermore, the benefits of sparsity in neural networks were discussed, describing the use of L1 norm for sparsity and a typical 300 non-zero dimensions in high-dimensional layers.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Meetup Alert: Limited Seats Available**: Few spots remain for the upcoming **LlamaIndex meetup** scheduled for Tuesday, with enthusiasts encouraged to [claim their spots](https://twitter.com/llama_index/status/1793739449127583964) quickly due to limited availability.

- **MultiOn Meets LlamaIndex for Task Automation**: **LlamaIndex** has been coupled with **MultiOn**, an AI agents platform, facilitating task automation through a Chrome web browser acting on behalf of users; view the demo [here](https://twitter.com/llama_index/status/1793764970024570979).

- **RAGApp Launches for Code-Free RAG Chatbot Setup**: The newly introduced **RAGApp** simplifies the deployment of RAG chatbots via a docker container, making it easily deployable on any cloud infrastructure, and it's open-source; configure your model provider [here](https://twitter.com/llama_index/status/1794030544415818062).

- **Solving PDF Parsing Puzzles**: The community endorses **LlamaParse** as a viable API for extracting data from PDFs, especially from tables and fields, leveraging the GPT-4o model for enhanced performance; challenges with **Knowledge Graph Indexing** were also a topic, highlighting the need for both manual and automated (through `VectorStoreIndex`) strategies.

- **PostgresML Joins Forces with LlamaIndex**: **Andy Singal** shared insights on integrating **PostgresML** with **LlamaIndex**, detailing the collaboration in a Medium article, ["Unleashing the Power of PostgresML with LlamaIndex Integration"](https://medium.com/ai-advances/unleashing-the-power-of-postgresml-with-llamaindex-integration-9eadee223939), receiving positive remarks from the community.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Phi-3 Medium 128k Instruct Drops**: OpenRouter unveiled **Phi-3 Medium 128k Instruct**, a powerful 14-billion parameter model, and invited users to review both the [standard](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct) and [free](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct:free) variants, and to participate in discussions on its effectiveness.

- **Wizard Model Gets a Magic Boost**: The **Wizard model** has shown improvements, exhibiting more prompt and imaginative responses, yet attention is required to avoid repeated paragraphs.

- **Eyes on Phi-3 Vision and CogVLM2**: Enthusiasm surges around **Phi-3 Vision**, with sharing of testing links like [Phi-3 Vision](https://ai.azure.com/explore/models/Phi-3-vision-128k-instruct/version/1/registry/azureml), and suggestions to use **CogVLM2** for vision-centric tasks found at [CogVLM-CogAgent](https://huggingface.co/spaces/THUDM/CogVLM-CogAgent).

- **Automatic Llama 3 Prompt Transformation**: It was clarified that prompts to **Llama 3** models are automatically transformed through OpenRouter's API, streamlining the process, but manual prompting remains as an alternative approach.

- **Gemini API Annoyances**: Users reported issues with **Gemini FLASH** API, such as empty outputs and token drain, recognized as a model-centric problem. The emergence of Google's daily API usage limits has piqued interest in how this might affect OpenRouter's Gemini integration.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LLM Evaluation under the Microscope**: A [Hugging Face blog post](https://huggingface.co/blog/clefourrier/llm-evaluation) about Large Language Model (LLM) evaluation practices, the importance of leaderboards, and meticulous non-regression testing caught the attention of members, emphasizing the critical role of such evaluations in AI developments.

- **AI's Answer to Search Engine Manipulations**: An incident involving website poisoning affecting Google's AI-gathered overviews triggered discussions around security and data integrity, including workarounds through custom search engine browser bypasses as reported in a [tweet by Mark Riedl](https://x.com/mark_riedl/status/1793375699967054334).

- **AI Democratizing Development or Raising Reliability Questions?**: GitHub CEO Thomas Dohmke's [TED Talk](https://youtu.be/nv9WwHpOKEg?si=mVApo6UnrtJ9ExH6) on AI's role in simplifying coding provoked debates over its reliability despite AI-driven UX improvements that expedite problem-solving in the coding process.

- **Diversity Scholarships to Bridge Gaps**: Engineers from diverse backgrounds who face financial barriers to attending the upcoming AI Engineer World's Fair received a boost with the announcement of diversity scholarships. Interested applicants should furnish *concise* responses to the essay questions provided in the [application form](https://docs.google.com/forms/d/e/1FAIpQLScff_RUv-fIKfdj_2HcHtk96iy45GD0BWLByGxqdBqvcepDHg/viewform?usp=sf_link).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Tax Tales Without Plastic**: Nathan Lambert deciphered an invoice kerfuffle, realizing the rational behind tax billing sans credit card due to resale certificates.

- **Golden Gate AI Gets Attention**: Experimentation by [Anthropic AI](https://x.com/anthropicai/status/1793741051867615494?s=46) led to "Golden Gate Claude," an AI single-mindedly trained on the Golden Gate Bridge, creating buzz for its public interactivity at claude.ai.

- **Google's AI Missteps**: Google's failure to harness feedback and premature deployment of AI models spurred discussion about the tech giant's public relations challenges and product development woes.

- **Battling Dataset Misconceptions**: Google's AI team countered claims about using the LAION-5B dataset by putting forth that they utilize superior in-house datasets, as referenced in a [recent tweet](https://x.com/giffmana/status/1793906145310228538).

- **Nathan Shares Knowledge Nuggets**: For AI aficionados, Nathan Lambert uploaded advanced [CS224N lecture slides](https://docs.google.com/presentation/d/1on5xTePaUYg47vui3dUr0Lp6GUXmOXmhceNJLXRbGsE/edit). Additionally, attendees were tipped off about an upcoming session recording, sans release date details.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GQA Gains Traction in CMDR Models**: Discussions revealed that **Grouped Query Attention (GQA)** is present in the "cmdr+" models but not in the basic "cmdr" models, indicating an important distinction in their specifications.
- **VRAM Efficiency with Smart Attention**: Engineers noted that while **GQA** doesn't offer linear scaling, it represents an improved scaling method compared to exponential, affecting **VRAM** usage favorably.
- **Sample Packing Gets a Boost**: A new **GitHub pull request** showcases a 3-4% efficiency improvement in sample packing, promising better resource management for distributed contexts, linked [here](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1619).
- **Academic Achievement Acknowledged**: A member's co-authored journal article has been published in the **Journal of the American Medical Informatics Association**, highlighting the impact of high-quality, mixed-domain data on medical language models, with the article available [here](https://doi.org/10.1093/jamia/ocae120).
- **Community Cheers Scholarly Success**: The community showed support for the peer's published work through personal congratulatory messages, fostering a culture of recognition for academic contributions within the AI field.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**SB-1047 Sparks Technical Turmoil**: Engineers express deep concerns about the implications of **SB-1047**, dubbing it as detrimental to smaller AI players and likening the situation to regulatory capture observed in other industries.

**Perplexity and Arc, Tools of the Trade Showcased**: The community spotlighted tools aiding their workflows, sharing a [Perplexity AI search on SB-1047](https://www.perplexity.ai/search/SB-1047-Senate-2kZmFYHoTxe.rWUYat4B2A) and the new “Call Arc” feature of Arc Browser, which simplifies finding relevant answers online, with an informational [link](https://arc.net/e/C56904FA-1C75-4D77-9A87-E7F1A52529CD).

**Install Issues Incite Inquiry**: Users face issues with **Typer** library installation via pip, raising questions about whether steps in the setup process, such as `poetry install` before `poetry run`, were followed or if a virtual environment is being used.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

**Twinny Takes Off as Virtual Co-Pilot**: Developers are integrating [Twinny](https://github.com/rjmacarthy/twinny) with LM Studio to serve as a robust local AI code completion tool, with support for multiple llamafiles running on different ports.

**Embedding Endpoint Enlightenment**: The `/v1/embeddings` endpoint was clarified not to support `image_data`; instead, the `/embedding` endpoint should be used for images, as per [pull request #4681](https://github.com/ggerganov/llama.cpp/pull/4681).

**Mac M2 Meets Its Match in continue.dev**: A performance observation noted that continue.dev runs slower on a Mac M2 compared to an older Nvidia GPU when executed with llamafile.

**Hugging Your Own LLMs**: For those looking to build and train custom LLMs, the community recommended the use of [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) for training, with the reminder that llamafile is designed for inference, not training.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Gratitude Echoes in the Server**: A user expressed heartfelt *thanks* to the team, showcasing user appreciation for support or development work done by the team.
- **Curiosity About Upscaled Models**: There's buzz around whether a **104B version** of a model will join the family tree, but no clear answers have been outlined yet.
- **Langchain Links Missing**: Questions arose regarding the integration of **Langchain** with Cohere, with users seeking guidance on its current usability and implementation status.
- **Model Size Mysteries**: Users are probing for clarity on whether the **Aya model** in the playground pertains to the 8B or 35B version, indicating importance in understanding model scales for application.
- **Error Troubleshooting Corner**: Issues like a `ValidationError` with **ContextualCompressionRetriever** and a **403 Forbidden error** signal active debugging and technical problem-solving among the engineers, serving as reminders of common challenges in AI development.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

**AI Comedy Night Hits the Right Notes**: An AI-generated standup comedy piece shared by a user was met with positive surprise, indicating advancements in AI's capability to mimic humor and perform entertainment.

**Exploratory Queries on AI Applications**: Curiosity about the extent of [Ud.io](https://www.udio.com/songs/vsNF2nbsy646jGt348mdFG)'s functions was evident from a user's query whether its capabilities go beyond generating comedy.

**Sound Transformations Showcased**: A user displayed the flexible audio alteration features of [Suno](https://suno.com/song/e6b62587-4345-44fb-85c7-c51f932df655) by sharing an altered, demonic version of an original sound piece.

**Eagerness for Audio Engineering Know-How**: Interest was expressed in acquiring the skills to craft audio modifications like the ones demonstrated, a skill set valuable for an AI engineer with an interest in sound manipulation.

**Concise Communication Preferred**: A one-word reply "No" to a question highlighted a preference for succinct responses, perhaps reflecting an engineer's desire for direct, no-nonsense communication.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **In Search of a Unified Event Tracker**: A member has highlighted a pressing need for an event calendar compatible with Google Calendar to ensure no community events are overlooked. The absence of such a system is a noted concern within the community.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **New Dataset Announcement**: A new dataset has been referenced by user datarevised, with a link to further details: [DataPlusEngine Tweet](https://x.com/DataPlusEngine/status/1793803117642854732).




---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Benchmarks Drive Innovation**: Evaluation benchmarks like [GLUE](https://arxiv.org/abs/1804.07461), [MMLU](https://arxiv.org/abs/2009.03300), and [GSM8K](https://arxiv.org/abs/2110.14168) are crucial for AI research progress, while "execution based evals" face particular challenges in dynamic tasks, as discussed in a [blog post](https://www.jasonwei.net/blog/evals) shared by members.
  
- **Anticipation for GPT-5**: A talk stirred speculation among guild members that **GPT-5** may debut in 2024 with an advancement towards an "agent architecture," as suggested in a [rumored tweet](https://x.com/rohanpaul_ai/status/1793956355897724973?s=46&t=90xQ8sGy63D2OtiaoGJuww).

- **AI's Role in Music Examined**: Members probed into copyright questions concerning AI-created music like **Suno**, contemplated the capabilities of Meta's **Audiocraft**, and discussed legal ramifications and open-source endeavors promoting creative freedom, including **gary-backend-combined** on [GitHub](https://github.com/betweentwomidnights/gary-backend-combined).

- **Pythonistas Gear Up for NumPy 2.0**: There's building excitement for **NumPy 2.0**, with members jesting about potential dependency management impacts, as noted in a [Twitter post](https://x.com/cgarciae88/status/1794019900119236874).

- **xAI Lands a Whopper of Investment**: Breaking into investment news, **xAI** secured a substantial $6 billion Series B funding round, setting sights on rapidly advancing their models' capabilities, as illustrated in their [announcement post](https://x.ai/blog/series-b).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Nightly Nuisances Fixed by Community Brainstorm**: After members highlighted that the **Mojo VSCode extension** worked only for post-installation files and was buggy since version 24.3, solutions included closing/reopening the code and frequent resets. Conversely, the **Mojo compiler's** nightly updates (`2024.5.2505` to `2024.5.2705`) featured enhancements like variable renaming in LSP and a new `tempfile` module, though they introduced test issues in existing PRs like [#2832](https://github.com/modularml/mojo/pull/2832) on GitHub.

- **The Zen of ZIPping Through Mojo**: Difficulties in replicating Python's `zip` and `unzip` functions in Mojo led to a discussion on how to declare functions returning tuples based on variadic list arguments. The conversation shed light on Mojo's potential auto-dereferencing iterators using the new `ref` syntax, aiming to simplify implementation and reduce explicit dereferencing.

- **Processor Picks Perform Differently**: A member's efforts to optimize CRC32 calculations in Mojo led to performance variance across core types; compact implementation lagged on larger byte sizes due to L1 cache limits, yet efficiency cores favoured the compact version. Benchmarking metadata and versioning files are located at [fnands.com](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crc.mojo) and [nightly version tests for the aforementioned](https://github.com/fnands/fnands.com/blob/main/blog/2024/mojo-crc-calc/crcn.mojo).

- **Musings Over Mojo’s Might**: General discussions on **Mojo** ranged from function behavior differences compared to Python, handling of optional types, to implementing LinkedLists, and reflecting on Mojo's as-yet-to-be-implemented reflection capabilities. Members exchanged thoughts on efficient initializations using `UnsafePointer` and `Optional`.

- **Tech Titans' Tremendous Transactions**: The AI community digested news that Elon Musk’s AI startup, xAI, hauled in a hefty $6 billion in Series B funding as outlined in a [TechCrunch article](https://techcrunch.com/2024/05/26/elon-musks-xai-raises-6b-from-valor-a16z-and-sequoia/), positioning xAI to directly confront AI frontrunners like OpenAI and Microsoft, stirring conversations around the commercialization of AI technologies.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**New Kids on The Block: Phi-3 Models Arrive**: Microsoft's [phi-3-medium-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-medium-128k-instruct) and [phi-3-mini-128k-instruct](https://openrouter.ai/models/microsoft/phi-3-mini-128k-instruct) models are now live, with a special 57% discount applied to the [llama-3-lumimaid-70b](https://openrouter.ai/models/neversleep/llama-3-lumimaid-70b) model.

**Rate Limit Labyrinth Explained**: Challenges with **rate limiting** on OpenRouter sparked intense discussion, emphasizing the importance of understanding how credit balances impact request rates, as outlined in the [OpenRouter documentation](https://openrouter.ai/docs#rate-limits-and-credits-remaining).

**Modal Mayhem: When Credits Clash with Rate Limits**: A puzzling issue arose with the modal fallback feature, where rate limits were hit despite a healthy credit balance. The community recommended monitoring free requests and possibly sidelining the free model when limits loom.

**AI's Self-Moderation Struggle Softens Appeal**: Enthusiasts expressed concerns that stricter guardrails and higher refusal rates in Claude's self-moderated models result in a less human-like experience, pointing to a possible downturn in usage.

**Vision Model Breakdown: Performance vs. Price**: The talk turned to vision model performance, specifically Gemini's OCR capabilities, with a nod to its cost-effectiveness compared to traditional vision services. Conversations also highlighted cheaper GPU usage via RunPod and Vast.ai over mainstream clouds like Google Cloud and Amazon Bedrock.




---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **CCS Doesn't Cut the Mustard**: The Eleuther team's follow-up to Collin Burns's _Contrast Consistent Search (CCS)_ method didn't yield expected results in generalization improvements, as shown in the [Quirky Models benchmark](https://arxiv.org/abs/2312.01037). Their transparency in sharing both the method and lackluster results was commended in a detailed [blog post](https://blog.eleuther.ai/vincs/).

- **Optimal Model Extraction Strategy Hotly Debated**: Engineers sparred over whether **RAG** is superior to finetuning for **LLM**s in extracting data from custom libraries, concluding that RAG might retain information better. There's buzz around **ThePitbull 21.4B Model**, released on Hugging Face, with some skepticism regarding its near-70B-model performance claims.

- **Troubleshooting Data Replication**: AI programmers grappled with tokenizer tribulations while replicating Pythia data, with solutions including the use of `batch_viewer.py` and proper datatype handling with `MMapIndexedDataset`. The process, though likened to "black magic", was necessary for correct dataset interpretation, as noted in the [Pythia repository](https://github.com/EleutherAI/pythia#exploring-the-dataset).

- **Pushing the Envelope with New Techniques**: Accelerated discussions among engineers covered gradient perturbation methods to distribute weight updates and the potential of transformers to implicitly reason over parametric knowledge. A new "schedule-free" optimization paper caught eyes, suggesting iterate averaging without extra hyper-parameters might outperform traditional learning rate schedules.

- **Quantization Quandaries**: In the quest for efficiency, a discourse ensued on the situations where small models might surpass larger quantized ones. A reference to [ggerganov's Twitter](https://x.com/ggerganov/status/1666087050725199872) hinted at quantization methods' potential, stoking the fires of debate regarding the balance of model size and performance.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG Revolution: Enterprise Pipelines on the Horizon**: The **LlamaIndex team** announced a workshop in collaboration with AWS to showcase how to build **enterprise-level RAG pipelines** using AWS Bedrock and Ragas. Registrations are live for the event aiming to provide insights on integrating Bedrock with LlamaIndex for optimized RAG systems [Sign Up Here](https://lu.ma/x8unmku0).

- **Innovations Boosting Retrieval**: Discussions have spotlighted **Vespa's integration** for improved hybrid search capabilities and an advanced guide by Jayita B. on creating **rapid response RAG chatbots** using Llama3 and GroqInc. An innovative method for indexing images through structured annotations produced by models like **gpt4o** was also noted.

- **File Organization Ascended**: **LlamaFS** was launched as a new tool for automatically organizing cluttered directories, which may resonate with those seeking clean and efficient file management solutions.

- **Technical Troubleshooting Takes Center Stage**: AI Engineers grapple with issues surrounding **Llama Index's reAct**, with workarounds involving `max_iterations` settings and overcoming import errors by aligning package versions. HTML parsing generally requires more custom code compared to PDF files, which can take advantage of advanced chunking tools with fewer dependencies.

- **Pydantic for Structured LlamaIndex Output**: Guidance on using **Pydantic models** with LlamaIndex signifies a step towards structured output integration, pointing to broader applications and usability of the system. Calls for improved retriever documentation highlight community drive for enhanced understanding and application of **BM25** and **AutoRetrieval** modules within diverse projects.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

**Aya-23 Takes the Stage**: Engineers discussed **Aya-23's multilingual capabilities** compared to **Command R/R+**, implying superior performance but questioning its English-specific efficiency. They also noted **Aya-23-35b** is a fine-tuned version of **Command R** and provided access to the [technical report](https://drive.google.com/file/d/1YKBPo61pnl97C1c_1C2ZVOnPhqf7MLSc/view) for more details.

**Mobile Privacy Vs. LLM Limitations**: There was a consensus that **on-phone LLMs** aren't sufficiently developed for private, local execution in a mobile app, particularly for tasks typically aligning with a **RAG mobile app**.

**Bot Innovations Flourish**: A community member showcased a gaming bot on LinkedIn which garnered interest due to its integration with **Cohere Command R**; meanwhile, the "Create 'n' Play" bot for Discord boasts ***"over 100 engaging text-based games"*** and *enhances social engagement with AI*.

**Adaptation and Integration of Prompts**: The guild confirmed that **Aya-23 supports system prompts**, sharing insights on adapting **Command R** prompts with specific tokens such as `<|USER_TOKEN|>` and `<|CHATBOT_TOKEN|>` to operate effectively.

**Solutions for OneDrive Syncing**: In response to a query about **OneDrive connectors**, a [SharePoint connector](https://github.com/cohere-ai/quick-start-connectors/tree/main/sharepoint) was recommended, which may fulfill similar integration needs.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**AI's Advice Bridge Ditching**: Members shared a humorous take on Google AI's dangerous advice to "jump off bridges to cure depression", referencing the misleading nature of Reddit suggestions. A related meme was shared regarding the mishap.

**ConvNeXt Gets Optimized**: A vibrant discussion on the [ConvNeXt paper](https://arxiv.org/abs/2405.15738) praised its ability to handle high-resolution images efficiently, potentially reducing the generation of excessive visual tokens and streamlining optimizations for high-resolution tasks.

**From Redstone to Neural Nets**: Innovative uses of datasets and AI tools were showcased, including a dataset of publication PDFs and source TeX from [archive.org](https://archive.org/details/arxiv-bulk?sort=-publicdate), and a [YouTube video](https://youtu.be/DQ0lCm0J3PM?si=5Is7OMnqRhZb-ZAo) demonstrating how to create a neural network with Redstone.

**Growth Stacks Up in AI Pre-training**: An [arXiv paper](https://arxiv.org/abs/2405.15319) highlighting depthwise stacking as an effective method for model growth in efficient pre-training of Large Language Models (LLMs) sparked interest, addressing critical speed and performance challenges in the pre-training process.

**Pitfalls in PyTorch Persistence**: Discussions in the learning sphere centered on troubleshooting issues with the randomness in training-validation splits and loss inconsistency during model reloads. Specifically, proper saving of optimizer states in PyTorch was pinpointed as crucial to avoid exploding losses.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Llama3 Goes Deutsch**: The new **Llama3-German-8B** model extends the capabilities of Meta's Llama3-8B to the German language, training on 65 billion tokens with negligible performance loss in English; details are on [Hugging Face](https://huggingface.co/DiscoResearch/Llama3-German-8B). However, it's noted that unlike other language models, the training **omitted English data replay**, sparking debates about its effectiveness.

- **Quantization Quirks and Puzzles**: A quantized version of the Llama3, **GGUF** has shown underwhelming benchmark scores, hinting at potential issues, available for review at [Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF](https://huggingface.co/cstr/Llama3-DiscoLeo-Instruct-8B-32k-v0.1-GGUF). Meanwhile, discussions on parameter settings and strange outputs hint at the complex challenges of running these models, like max tokens and choice of the engine.

- **Cohere’s Multilingual Leap**: **Cohere's Aya-23-35B model**, though restrictive in licensing, now supports 23 languages, indicative of the growing trend and interest in powerful multilingual models. A related **ShareGPT format dataset** for translations has generated talks on quality and filtering, hosted [here](https://huggingface.co/datasets/sroecker/aya_german-sharegpt).

- **Mistral’s Guide to the Fine-tuned Galaxy**: In a nod to the tech-savvy community, Mistral rolls out a finetuning guide for Mixtral models, a beacon for those embarking on finetuning adventures; the guide can be perused on their [GitHub](https://github.com/mistralai/mistral-finetune).

- **Model Tuning Intricacies Exposed**: The community is conducting experiments with ***oobabooga*** and ***ollama***, touching on **'skip_special_tokens'** toggles and stop token settings, including a suggested use of a specific [Llama3 template](https://github.com/CrispStrobe/EQ-Bench/blob/main_v2_3a/instruction-templates/Llama3.yaml) to address output issues—reflecting the active tweaking and tuning culture prevailing amongst members.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **LAM Security Concerns and Mobility**: Discussions reveal skepticism about the **Large Action Model (LAM)** due to its potential for reverse engineering, despite the [Rabbit LAM architecture overview](https://engineering.rabbit.tech/lam-security-architecture-overview) showcasing its secure web app control capabilities. Conversations also circled on integration challenges for running the 01 model on **Rabbit R1** devices and **Humane platform**, highlighting user-drive solutions like the [01 model Android GitHub repository](https://github.com/Tonylib/o1_for_flutter).

- **Installation Tactics on the Fly**: A Python version upgrade to **3.11 resolved OpenInterpreter issues** on Mac, while tinkering with a Linux distro through [Andronix](https://andronix.app/) enabled OpenInterpreter use on Android-powered earbuds. Queries about **Open Interpreter** installation revealed a need for a local or cloud-accessible LLM, and a new [Markdown export feature](https://github.com/OpenInterpreter/open-interpreter/pull/1282) was launched to aid developers.

- **DIY Spirit in the Community**: A user fixed an OpenInterpreter issue on Mac by upgrading to Python 3.11, and others shared pathways for enhancing current devices, like modifying **R1 with LineageOS**. In buying versus building hardware, the consensus is that purchasing the pre-built O1 benefits the Seattle development team, despite no technical differences from a self-built version.

- **Shipping and Storage Under Scrutinization**: Frustration was voiced over **lack of updates on hardware shipment**, particularly regarding European distribution and information blackout for pre-orders. The conversation also featured a search for solutions to **overcome disk space limits** on Runpod, a hint at the constant struggle for cost-efficient data storage in AI workloads. 

- **OpenInterpreter Leap Forward**: A member's successful tactic of switching to **Python 3.11** on Mac for OpenInterpreter signals the ongoing agility in problem-solving within the community. Meanwhile, the implementation of a [new Markdown export feature](https://github.com/OpenInterpreter/open-interpreter/pull/1282) reflects the push for enhancing developer utility in AI toolchains.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**PDF Extraction Proves Challenging**: Discussions on extracting text from PDFs highlight the difficulties encountered with complex tables and diagrams, suggesting solutions like ML-based text segmentation and using Adobe Extract API for layout parsing, as referenced in the [LangChain documentation](https://python.langchain.com/v0.2/docs/how_to/document_loader_pdf/).

**LangChain Community Set to Expand**: Karan Singh from Scogo Networks expressed interest in creating a local LangChain community in Mumbai, seeking marketing contacts to organize events.

**Bump in the Langserve Waitlist**: Users experienced access issues with the Langserve waiting list on Airtable, searching for alternate methods to try the hosted service.

**Interactive Data Visualization Tool Introduced**: The NLAVIDA project, which facilitates interactive data visualization and analysis through natural language, was introduced along with a [YouTube video tutorial](https://www.youtube.com/watch?v=leJRP_mJsSQ&t=4s).

**Ready, Set, Vote for OranClick**: The launch of OranClick, a tool aimed at optimizing message crafting for higher signup rates, was announced with an invitation to support on [ProductHunt](https://producthunt.com/posts/oranclick).



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile Powers Up with v0.8.5**: The release of [llamafile version 0.8.5](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.5) was highlighted, offering **fast inference** for K quants on X86 CPUs, with a call to action for community benchmarking efforts using `llamafile-bench`.
- **Llamafile Goes Network-Savvy**: Engineers swapped tips on how to make the llamafile server network-accessible, suggesting using flags like `--host <my ip>` or `--host 0.0.0.0` for cross-machine availability within the same network.
- **Blank Slate Mystery in Llama3-70B**: Contributors reported encountering **blank responses** from the llama3-70b model, sharing logs for community-led debugging, although definitive solutions were yet to surface.
- **Home Sweet APIs for Home Assistant**: There's a buzz around enhancing **Home Assistant** integration, with a focus on developing a standardized local API akin to OpenAI's candidate, underscored by the importance of features like API discoverability and secure API endpoints.
- **Python Puzzles for Model Select**: Questions and shared code snippets indicated some confusion around specifying models in the Python example for LLaMA_CPP integration, particularly concerning when model specification is a must, such as with TinyLlama.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral-Finetune Mystery Uncovered**: Developers engaged in deciphering the new update in the [Mistral-Finetune repository](https://github.com/mistralai/mistral-finetune), with emphasis on understanding the distinctive changes.
- **MoEs Fine-Tuning Quirkiness**: The AI crowd discussed the fickle nature of fine-tuning **Mixture of Experts (MoEs)** models, highlighting the necessity of running multiple iterations to cherry-pick the most efficient model, albeit details on the success rates were not divulged.
- **Aya 23's Restricted Aptitude**: The limitations of **Aya 23** were hotly debated, underscoring its suboptimal performance for chat-based applications, with its prowess confined to niche tasks, as per its [technical report](https://cohere.com/research/papers/aya-command-23-8b-and-35b-technical-report-2024-05-23).
- **MoRA Steps into the Fine-Tuning Spotlight**: **MoRA**, introduced as a cutting-edge high-rank updating approach for fine-tuning, entered the discussions with its potential to complement or exceed LoRA, linked with a [dedicated GitHub repository](https://github.com/kongds/MoRA).
- **FFD Bin Packing Woes and Llama 3 Tokens in Limelight**: Issues surfaced regarding the FFD bin packing implementation, specifically in a distributed training context, and fixes for Llama 3 related to untrained tokens, with a [patch shared for sfttrainer](https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L722) to address the latter.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Simulating Realities with Virtual Beings**: The [Virtual Beings Summit](https://www.virtual-beings-summit.com) could be a valuable event for AI professionals, featuring **Will Wright** and focusing on the intersection of AI with simulations. Supporting content can be found on the [Virtual Beings YouTube channel](https://www.youtube.com/channel/UCLNalUbhs_EYB5IAbw1VfaA), which offers insights into AI's role in interactive simulations.

- **AI Dreamscapes with DIAMOND**: The [DIAMOND GitHub repository](https://github.com/eloialonso/diamond) introduces "DIAMOND (DIffusion As a Model Of eNvironment Dreams)," which uses diffusion models in a reinforcement learning context to enhance environmental interactions within AI simulations.

- **Crafting AI "Westworlds" with UE5 and 4Wall**: Discussions around creating immersive experiences suggest that AI Town might leverage **UE5** and integrate voice control to simulate environments akin to "Westworld," with ongoing development info available at the [4Wall Discord](https://discord.gg/vPum4s3h).

- **Venturing into Virtual Reality**: The idea of integrating AI Town with **VR** technology was met with enthusiasm, indicating that engineers are considering novel methods to bring AI-generated environments to life through VR.

- **Animating Avatars with SadTalker and V-Express**: Two GitHub repositories, [SadTalker](https://github.com/OpenTalker/SadTalker) and [Tencent AI Lab's V-Express](https://github.com/tencent-ailab/V-Express), provide tools for creating realistic talking face animations and generating talking head videos, respectively, showcasing advancements in stylized animation technology.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Zyphra Zamba Slithers into the Spotlight**: The new **Zyphra Zamba** model, a blend of mamba and attention mechanisms, has launched with corresponding [technical report](https://www.zyphra.com/s/Zamba.pdf), [PyTorch code](https://github.com/Zyphra/Zamba-torch), and integration into [Hugging Face Transformers](https://github.com/huggingface/transformers/pull/30950). Comparative analysis with **OLMo 1.7** is in progress to benchmark its performance.

**Hushed Release of SD Audio 2.0**: An unauthorized release of **SD Audio 2.0** appeared on 4chan and is also available on a Hugging Face account, sparking discussions among members.

**Station-to-Station Regulation**: Former OpenAI board members Hellen Toner and Tasha McCauley propose in [The Economist](https://www.economist.com/by-invitation/2024/05/26/ai-firms-mustnt-govern-themselves-say-ex-members-of-openais-board) strict regulation over AI companies, emphasizing the inability for such companies to self-regulate due to profit motives and calling out past internal issues.

**Controversy in Command**: The article critiques Sam Altman’s alleged “toxic culture of lying” during his tenure, discussing both internal investigations and public outcry over the absence of transparency.

**A Textbook Case for RL**: The community shared a new resource, [a textbook on reinforcement learning from human feedback](https://github.com/natolambert/rlhf-book) on GitHub, and praised professors Chris Potts and Chris Manning for their engaging teaching styles. Discussions included when the electronic version of Stanford's 224n class would be released, with suggestions to reach out to Chris for concrete timelines.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Tweaking Time Limits in Tech Tests**: Discussions involved the possibility of extending the per-test time limit beyond **9 minutes 34 seconds** to accommodate complex functions like 'Taylor approximations'. A specific issue was with the `clang` function not completing, only reaching approximately 60% completion.

**Crashing Compilations Need Solutions**: One member pointed out the dilemma of generating *excessively large expressions* that crash compilers with errors related to incompatible operand types, specifically doubles.

**Bitwise Operations on Double Drama**: Clarifications were made regarding the impossibility of performing bitwise operations like **XOR** on `double` data types, addressing the cause of a compilation error observed by members.

**Bounty Hunting Heats Up**: Interest spiked in various research-oriented bounties, with discussion on old pull requests and confirmation from **George Hotz** that bounties, such as the one referenced in [tinygrad pull request #4212](https://github.com/tinygrad/tinygrad/pull/4212), are still available.

**Deciphering 'vin' and Discussing Dominators**: George Hotz clarified that 'vin' in the **UOp class** is not an acronym. Additionally, a member questioned why post dominator analysis isn't used for improving scheduling in models, suggesting it might optimize subgraph fusion during execution.




> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

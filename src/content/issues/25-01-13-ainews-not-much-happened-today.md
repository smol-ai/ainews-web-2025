---
id: 3b3c903f-bd16-42fd-97e8-2a91264faf5e
title: not much happened today
date: '2025-01-14T06:08:22.078500Z'
original_slug: ainews-not-much-happened-today-9477
description: >-
  Kyutai Labs previewed Helium-1, a 2B-parameter multilingual LLM optimized for
  edge devices that outperforms Qwen 2.5, while Mistral AI released Codestral
  25.01, a new state-of-the-art coding model supporting over 80 programming
  languages with double the speed of previous versions. Additionally,
  innovations in retrieval-augmented generation (RAG) frameworks like AutoRAG
  and Agentic RAG are enhancing retrieval accuracy and autonomy in AI systems.
tags:
  - kyutai-labs
  - helium-1
  - mistral-ai
  - codestral-25-01
  - coding
  - rag
  - autoreg
---


<!-- buttondown-editor-mode: plaintext -->**a quiet day is all you need.**

> AI News for 1/10/2025-1/13/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **32** Discords (**219** channels, and **2928** messages) for you. Estimated reading time saved (at 200wpm): **312 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!


Welcome to [Codestral](https://x.com/lmarena_ai/status/1878872916596806069), but for the frontier model labs, releases happen closer to the 15th of every month. Not long now.

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

**AI Model Releases & Benchmarks**

- **Helium-1 Preview by @kyutai_labs**: [@reach_vb](https://twitter.com/reach_vb/status/1878860650560025011) announced **Helium-1 Preview**, a **2B-parameter multilingual base LLM** targeting edge and mobile devices. It **outperforms Qwen 2.5**, trained on **2.5T tokens** with a **4096 context size** and utilizes **token-level distillation from a 7B model**.
  
- **Phi-4 in @lmstudio**: [@awnihannun](https://twitter.com/awnihannun/status/1878564132125085794) released **Phi-4 (4-bit)** model in **@lmstudio** on an **M4 max**, noted for its **speed and performance**.
  
- **Sky-T1-32B-Preview by @LiorOnAI**: [@LiorOnAI](https://twitter.com/LiorOnAI/status/1878876546066506157) introduced **Sky-T1-32B-Preview**, a **$450 open-source reasoning model** matching **o1's performance** with **82.4% on Math500** and **86.3% on LiveCodeBench-Easy**.
  
- **Codestral 25.01 by @MistralAI**: [@sophiamyang](https://twitter.com/sophiamyang/status/1878902888434479204) released **Codestral 25.01**, a **new SOTA coding model**, **#1 on LMSYS**, offering **80+ programming languages** and **2x speed** compared to previous versions.

**AI Research & Innovations**

- **AutoRAG Framework**: [@llama_index](https://twitter.com/llama_index/status/1878881368186454161) unveiled **AutoRAG**, a framework for **optimizing RAG pipelines**, highlighting that **hybrid retrieval** often outperforms pure **vector or BM25 approaches**.
  
- **Agentic RAG by @huggingface**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1878590804325011727) explored **Agentic RAG**, which **reformulates user queries**, **critiques retrievals**, and **repeats** the process to **enhance system accuracy and autonomy**.
  
- **Multiagent Finetuning**: [@omarsar0](https://twitter.com/omarsar0/status/1878816276312989821) introduced **Multiagent Finetuning**, using a **society of models** for **self-improvement**, showing **performance gains across reasoning tasks** with models like **Phi-3, Mistral, LLaMA-3, and GPT-3.5**.
  
- **VideoRAG Framework**: [@omarsar0](https://twitter.com/omarsar0/status/1878827350315659421) presented **VideoRAG**, enhancing **RAG** by incorporating **video content** using **Large Video Language Models (LVLMs)**, achieving strong results in tasks requiring **procedural knowledge**.

**AI Applications & Tools**

- **Dynamic UI AI Chat App**: [@skirano](https://twitter.com/skirano/status/1878865450702139824) developed an **AI chat app** that **transforms its UI** based on dialogue, supporting **themes like dark mode** and **Windows 98**, available on **@Replit**.
  
- **LangChain AI Tools**:
  - **DocTalk**: [@LangChainAI](https://twitter.com/LangChainAI/status/1878864591230234941) introduced **DocTalk**, enabling **natural conversations with PDF documents** through voice interactions.
  - **AI Travel Agent Tutorial**: Demonstrates building an **AI travel agent** using **LangChain's Plan and Execute architecture**.
  - **Intelligent News Agent**: Facilitates **AI-powered news summarization** using **LangGraph**.
  
- **GPU Rentals by Hyperbolic Labs**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1878850466626576696) offers **GPU rentals** with competitive pricing, featuring GPUs like **H100 ($0.99/hr)**, **A100 ($1.2/hr)**, and **RTX 4090 ($0.5/hr)**, supporting **compute accessibility**.
  
- **LLMQuoter**: [@omarsar0](https://twitter.com/omarsar0/status/1878820053933855147) presented **LLMQuoter**, which **enhances RAG** by **identifying key quotes** before generating answers, achieving **over 20-point accuracy gains**.

**AI Infrastructure & Hardware**

- **MLX Export for C++**: [@fchollet](https://twitter.com/fchollet/status/1878880859077714382) shared the capability to **export LLM inference** from **Python** to a **self-contained C++ binary** using **MLX**.
  
- **SemHash by @philschmid**: [@_philschmid](https://twitter.com/_philschmid/status/1878743789155516565) introduced **SemHash**, a **semantic text deduplication library** that **deduplicates millions of records** in minutes, crucial for **data leakage prevention**.
  
- **Local LLM Apps for Apple Devices**: [@awnihannun](https://twitter.com/awnihannun/status/1878843809460875593) launched an **open-source LLM app** supporting **iPhone, iPad, Mac**, built with **MLX Swift**, under **MIT license**.
  
- **Torch Compatibility Guides**: [@StasBekman](https://twitter.com/StasBekman/status/1878609223963246979) provided a **backward compatibility guide** for **torch._scaled_mm** across **PyTorch versions**.

**AI Safety, Ethics & Policies**

- **ICLR 2025 Workshop on Trust in LLMs**: [@micahgoldblum](https://twitter.com/micahgoldblum/status/1878834198620119443) announced the **ICLR 2025 Workshop** focusing on **building trust in LLMs** and their **applications**, featuring **paper awards** and a **lineup of speakers**.
  
- **Anthropic Fellows Program**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1878844587491643721) called for **applications** to the **inaugural cohort** of the **Anthropic Fellows Program** for **AI safety research**.
  
- **UK AI Policy Strategy**: [@jackclarkSF](https://twitter.com/jackclarkSF/status/1878821057370681466) praised the **UK government's strategy** for **AI adoption and development**, highlighting initiatives like **AI growth zones**, **unlocking national data**, **20X public compute**, and **funding technical regulators**.
  
- **AI Agent Productivity**: [@bindureddy](https://twitter.com/bindureddy/status/1878606861433463240) discussed **AI agents** that can **perform autonomous tasks** in systems like **Salesforce, PayPal, and Confluence**, potentially **increasing productivity by 50%** and reducing work weeks.

- **@RichardMCNgo on AI Self-Coercion**: [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1878561834724123120) addressed **self-coercion in AI agents**, emphasizing the importance of **model discipline** to prevent **high illegibility** and ensure **ethical behavior**.

**Memes/Humor**

- **Humorous Rants by @reach_vb**: [@reach_vb](https://twitter.com/reach_vb/status/1878898525830050265) tweeted, **"hahaha, what the actual fuck? how do you reconcile the two?"**
  
- **@agihippo's Meme Inquiry**: [@agihippo](https://twitter.com/agihippo/status/1878800703109710287) asked, **"Is this a meme. Am I doing it right?"**
  
- **@teortaxesTex's Rants**: Various humorous and ranting tweets, such as **"Sonnet is more CCP-censored than DeepSeek btw"** and **"God King Claude sounds based"**.
  
- **Personal Humor from @saranormous**: [@saranormous](https://twitter.com/saranormous/status/1878585361632485748) shared, **"also I’ve been a shitty sleeper since kid 1 😮‍💨"**.
  
- **Meme Engagement by @yrhesiaj**: [@yrhesiaj](https://twitter.com/yrhesiaj/status/1878718780974760226) enjoyed a meme format, stating, **"I like this meme format, we need more of it"**.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Criticism of 'Gotcha' tests to determine LLM intelligence**

- **[Llama goes off the rails if you ask it for 5 odd numbers that don’t have the letter E in them](https://i.redd.it/w5j543q9pnce1.jpeg)** ([Score: 465, Comments: 198](https://reddit.com/r/LocalLLaMA/comments/1i01k4s/llama_goes_off_the_rails_if_you_ask_it_for_5_odd/)): The post humorously highlights the challenges faced by **Llama**, an AI model, when tasked with identifying five odd numbers that lack the letter 'E' in their spelling. The AI's response includes incorrect and nonsensical terms like "Sand," "One," "Tud," and "Dug," illustrating the model's difficulty in accurately processing and reasoning through the request.
  - Commenters discuss the inherent difficulty for AI models to find **odd numbers without the letter "E"** in their spelling, noting that most odd numbers in English include "E". Despite various attempts, models like **Deepseek R1** and **O1-Mini** confirmed the impossibility of the task, with some models trying to circumvent the problem using numerals or Roman numerals, as seen with **Gemini 1.5 pro**.
  - The discussion highlights the **failure modes** of AI models with this challenge, with models like **Groq 2** humorously altering spellings to fit the criteria. This issue is compared to the **"strawberry test"**, emphasizing that the task involves both a spelling and logical challenge, requiring models to recognize the absence of a valid solution.
  - The conversation includes references to various AI models and platforms, such as **Meta's 70B and 405B models**, **Qwen2.5-Plus**, and **Pal Chat iOS app**, with **Deepseek v3** notably evaluating numbers from 1-100 and concluding that none fit the criteria. This underscores the complexity of the task and the models' varied approaches to problem-solving.


**Theme 2. Kokoro TTS Achieves High Performance with Limited Parameters**

- **Speaches v0.6.0 - Kokoro-82M and PiperTTS API endpoints** ([Score: 90, Comments: 15](https://reddit.com/r/LocalLLaMA/comments/1i02hpf/speaches_v060_kokoro82m_and_pipertts_api_endpoints/)): **Speaches v0.6.0** introduces support for **Piper** and **Kokoro** Text-to-Speech models with features like GPU/CPU support, Docker deployment, and OpenAI API compatibility. It also offers streaming and live transcription via SSE and WebSocket, dynamic model handling, and upcoming features like audio generation, sentiment analysis, and a Realtime API. [Project link](https://github.com/speaches-ai/speaches) and [documentation](https://speaches-ai.github.io/speaches/) are available for further details.
  - **Docker Image Access Issue**: Users report a **401 Unauthorized** error when trying to pull the Docker image from **ghcr.io**, suggesting that the image repository might be set to private or there is an issue with authorization tokens.


- **How is Kokoro TTS so good with so few parameters?** ([Score: 100, Comments: 46](https://reddit.com/r/LocalLLaMA/comments/1i06mew/how_is_kokoro_tts_so_good_with_so_few_parameters/)): **Kokoro TTS** achieves impressive results with only **82M parameters** by incorporating modifications to the **StyleTTS 2** model architecture and training primarily on synthetic data from **OpenAI** and **ElevenLabs**. The effectiveness may stem from either the quality of the synthetic data or undisclosed architectural changes. [Kokoro TTS on Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M).
  - Discussions highlight skepticism about the quality of **open source audio datasets** and suggest that **Kokoro TTS** could achieve similar results with fewer parameters. Users express interest in seeing the modified training code to explore pretraining models on consumer hardware, emphasizing the potential to achieve more with less.
  - The **voice cloning feature** of Kokoro TTS is debated, with some users noting its absence due to limited training time, while others point out successful voice restoration with minimal audio samples. The restoration of **Sky's voice**, which was removed by OpenAI, exemplifies this capability using only 3 minutes of audio.
  - **Quantization techniques** in TTS models are discussed, with users noting the potential for Kokoro TTS to maintain performance with reduced parameters through methods like **FP16 and Int8 quantization**. The trade-off between model size and performance is considered, with some suggesting further compression could compromise utility.


**Theme 3. Sky-T1: Open-Source AI Model Training for $450**

- **[Researchers open source Sky-T1, a 'reasoning' AI model that can be trained for less than $450](https://techcrunch.com/2025/01/11/researchers-open-source-sky-t1-a-reasoning-ai-model-that-can-be-trained-for-less-than-450/)** ([Score: 52, Comments: 12](https://reddit.com/r/LocalLLaMA/comments/1i0hecs/researchers_open_source_skyt1_a_reasoning_ai/)): **Researchers** have released **Sky-T1**, an open-source AI model focused on **reasoning** capabilities, which can be trained for under **$450**. This development highlights the trend towards more accessible and cost-effective AI training solutions.
  - **Sky-T1's Training Process**: Discussion highlights that **Sky-T1** was fine-tuned on **QWEN-32B-Instruct** using distilled data from **QwQ**, rather than being trained from scratch for **$450**. This clarification indicates a misunderstanding in the article regarding the training cost.
  - **Dataset and Reasoning**: **17k tasks** were used as a dataset, which some find surprisingly small given the potential to easily gather more data from math textbooks. This raises questions about the novelty and effectiveness of the dataset used for training.
  - **Distillation and Thinking Steps**: The model's ability to perform reasoning tasks through completion-based distillation is notable, sparking curiosity about why **OpenAI** doesn't provide explicit thinking steps in their models. There's a mention that even **Gemini** thinking models don't offer these steps, except for an experimental version.

**Theme 3. Hugging Face Unveils Agent Course for AI Developers**

- **Hugging Face released a free course on agents.** ([Score: 289, Comments: 18](https://reddit.com/r/LocalLLaMA/comments/1i0b289/hugging_face_released_a_free_course_on_agents/)): **Hugging Face** has released a new chapter in its **Smolagents course**, focusing on three types of agents: code agents, retrieval agents, and custom functional agents. The course is available for free and aims to assist developers in building agent applications, accessible [here](https://github.com/huggingface/smol-course/tree/main/8_agents).
  - **Smolagents and Model Compatibility**: Users report issues with the **Hugging Face demo code** when using **qwen2.5-coder 32B** with **ollama**, suggesting potential problems with the default ollama system prompt or endpoint configuration. There is also a discussion about the flexibility of loading different models, including **HfApiModel** and the possibility of using **gguf** for VRAM-limited scenarios.
  - **Guidelines on LLM Calls**: The guideline to "reduce LLM calls whenever possible" is debated, with some users arguing that in complex agentic workflows involving tasks like search and classification, frequent short LLM calls can be more effective. This approach, while potentially costly, may be necessary for achieving higher precision in professional use cases.
  - **Course Prerequisites and Code Usability**: The course is deemed accessible with basic **Python** knowledge and understanding of **LLMs via APIs**. There is feedback on the course materials, with a specific note that some code snippets were initially not runnable, which has been addressed in updates to the documentation.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. UC Berkeley's Sky-T1 Outperforms OpenAI-o1 with Budget Training**

- **[berkeley labs launches sky-t1, an open source reasoning ai that can be trained for $450, and beats early o1 on key benchmarks!!!](https://techcrunch.com/2025/01/11/researchers-open-source-sky-t1-a-reasoning-ai-model-that-can-be-trained-for-less-than-450/)** ([Score: 217, Comments: 32](https://reddit.com/r/OpenAI/comments/1i0cy09/berkeley_labs_launches_skyt1_an_open_source/)): Berkeley Labs has released **Sky-T1**, an open-source reasoning AI model that significantly reduces training costs to **$450**, outperforming the early **O1** model on key benchmarks. This development follows the recent launch of DeepSeek's v3 model, which costs **$5,500** to train, highlighting Sky-T1's cost efficiency and performance advantage. [Read more](https://techcrunch.com/2025/01/11/researchers-open-source-sky-t1-a-reasoning-ai-model-that-can-be-trained-for-less-than-450/).
  - **Cost and Performance**: There is a correction regarding the training cost of DeepSeek's v3 model, which is **$5.5 million**, not **$5,500**, emphasizing Sky-T1's cost efficiency.
  - **Open Source Transparency**: The open-source nature of **Sky-T1** is highlighted, allowing for transparency in design and data, eliminating the need for speculation about its capabilities.
  - **Innovation and Overfitting Concerns**: Some commenters question the true innovation behind Sky-T1, suspecting reliance on well-curated synthetic data and potential overfitting to benchmarks.


- **Sky-T1-32B: Open-sourced reasoning model outperforms OpenAI-o1 on coding and maths benchmarks** ([Score: 103, Comments: 9](https://reddit.com/r/OpenAI/comments/1i0cyip/skyt132b_opensourced_reasoning_model_outperforms/)): **UC Berkeley** has released **Sky-T1-32B**, an open-source reasoning model, which outperforms **OpenAI-o1** on benchmarks such as **Math500**, **AIME**, and **Livebench medium & hard**. The model was trained for under **$450**, and further details can be found [here](https://youtu.be/uzuhjeXdgSY).
  - Users expressed frustration over the **YouTube video** as a source of information, preferring direct links to benchmarks and model downloads. **R4_Unit** criticized the lack of useful info in the video description, leading to downvotes.
  - **LocoMod** provided a direct link to the model on **Hugging Face**: [Sky-T1-32B-Preview-GGUF](https://huggingface.co/bartowski/Sky-T1-32B-Preview-GGUF), emphasizing the importance of saving time.
  - **Formal-Narwhal-1610** pointed out that the title was misleading, clarifying that **Sky-T1-32B** outperformed **O1 Preview** rather than the full **O1** model.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-2024-12-17

**Theme 1. New Models and Surprising Stats**  
- [**Codestral 25.01 Crushes Speed Charts**](https://mistral.ai/news/codestral-2501/): It hit #1 on a copilot arena leaderboard, yet managed only 11% on the Aider polyglot benchmark. Members are excited about its 256k context window, with many eyeing production readiness.  
- [**Sky-T1 Speeds Past $450 Budget**](https://novasky-ai.github.io/posts/sky-t1/): This 32B model competes with o1-preview on popular reasoning tasks without big money. Its open codebase, [SkyThought](https://github.com/NovaSky-AI/SkyThought), openly courts more community-driven breakthroughs.  
- [**Helium-1 Goes Mobile**](https://kyutai.org/2025/01/13/helium.html): Kyutai’s 2B-parameter model aims for low-latency privacy on edge devices, supporting 6 languages. Users cheer for small-scale solutions that don’t sacrifice performance.  

**Theme 2. HPC Tuning and Memory Moves**  
- [**Triton Puzzles Push GPUs to the Limit**](https://github.com/gauravjain14/mlcompilers_and_kernels/tree/main/triton_kernels): Devs autotune kernels on A100 vs A30, watching shared memory constraints for big wins. They also reference [Liger Kernel cross entropy code](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py) to squeeze more speed out of small data chunks.  
- [**Slurm Solutions Save the Day**](https://slurm.schedmd.com/sbatch.html): Setting --mem=0 or --exclusive resolves CPU-based OOM issues on multi-GPU clusters. Proper resource flags transform HPC heartbreak into a smooth run.  
- [**Patchy Profiling in PyTorch**](https://github.com/pytorch/pytorch/issues/64345): UTF-8 decode bugs hamper advanced pipeline analysis. Users keep meta devices and stream activations with NNSight to dodge OOM fiascos.  

**Theme 3. Building Agents and Custom Bots**  
- [**Friday Agents Party in JS**](https://github.com/amirrezasalimi/friday-agents): This multi-agent framework helps devs parallelize tasks, easily hooking into [OpenRouter](https://openrouter.ai/). People praise concurrency for making agent experiments feel unstoppable.  
- [**DeVries AI Chuckles with 200+ LLMs**](https://devriesai.com/): For $24.99/month, Telegram fans quickly swap among 200+ models in one chat feed. The free trial lures early adopters to test labyrinthine AI combos.  
- [**Aider Adds Chat Modes**](https://aider.chat/HISTORY.html): v0.71.0 improves toggles between “/ask” and “/code,” streaming pretty outputs with triple-backtick fences. Users love code and question modes flipping on a dime.  

**Theme 4. Fine-Tuning, LoRA, and Data Delights**  
- [**Unsloth’s 30x Speedup Claims**](https://unsloth.ai/introducing): Custom Triton kernels promise big leaps in LLM training, with examples like Llama 3.3 and long context expansions. Users watch memory footprints drop while chat templates keep model outputs stable.  
- [**LoRA Magic Wins Author Styles**](https://docs.unsloth.ai/get-started/unsloth-notebooks): Provided enough curated text, LoRA replicates writing nuances at scale. Iterative fine-tuning fosters consistent voices, wowing creative and medical tasks alike.  
- [**Quality Trumps Quantity**](https://arxiv.org/abs/2402.12847): Forum dwellers stress rigorous data prep outruns massive raw dumps. They propose using other LLMs to filter docs before burning precious GPU hours.  

**Theme 5. Privacy, Caching, and Extended Context**  
- [**Privacy Mode Sparks Concerns**](https://forum.cursor.com/t/concerns-about-privacy-mode-and-data-storage/5418): Users question data embeddings stored on servers and potential NDA breaches. They call for deeper transparency on how code is handled.  
- [**Prompt Caching for Speedy RAG**](https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts/): Devs rely on proper file sets for consistent hits in caching. Differences across Anthropic, OpenAI, and local setups keep them inventing new strategies.  
- [**128k Context Dreams**](https://lmstudio.ai/model/phi-3.1-mini-128k-instruct): Adventurous testers push bigger windows with Phi 3.1 Mini 128k. They see moderate demands in VRAM but love the extra breathing room for monstrous prompts.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth & Llama 3.3 Race Ahead**: Users reported that **Llama 3.3** fine-tuned with **Unsloth** yields stable training with chat templates, scoring better on performance metrics, and requiring less VRAM.
   - **Unsloth** includes custom Triton kernels and claims a **30x** training speedup, prompting community interest in [Unsloth's blog](https://unsloth.ai/introducing).
- **LoRA Trick for Author Style**: Members used **LoRA** to replicate writing styles, emphasizing that substantial data preparation is critical for success.
   - They noted that **iterative fine-tuning** fosters consistent voice replication, addressing nuance in [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks).
- **Cyber Ops with a Deceptive LLM**: A cybersecurity researcher built a specialized LLM for **cyber deception**, generating over **1k** simulated adversary connections.
   - Participants appreciated how these persona-based tactics can spot **scams** more effectively, fueling interest in advanced methods.
- **Maya's Multilingual V-L Leap**: **Maya** was introduced as a **Multilingual Vision-Language Model**, outlined in a preprint shared on [Twitter](https://twitter.com/nahidalam/status/1866667770114609217).
   - Members praised **Maya**'s potential cross-lingual capabilities, calling it an exciting direction for combined text and image tasks.
- **TTS Chatbots from Transcribed Videos**: Developers sought to streamline **video transcripts** for real-time TTS chatbots, referencing [Whisper](https://huggingface.co/whisper) and other speech-to-text tools.
   - They explored **Fish Agent** and **Kokouro** for spoken output, underscoring the need for **10,000 hours** of audio for advanced language coverage.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SmolLM Sizzles with a 315GiB Release**: The **SmolLM-Corpus** launched with **315GiB** of data, split into 23698 `jsonl.zst` shards, including subsets from **cosmopedia-v2** and **fineweb-edu-dedup**, as shown on [Hugging Face](https://huggingface.co/datasets/Avelina/smollm-corpus).
   - Community members noted strong interest in large-scale dataset usage, referencing **Grouped-Query Attention** and expanded **VLM** capabilities in the same discussions.
- **Latro Gains Ground with PRMs and VinePPO**: The **Latro** model aims to improve reasoning via RL plus **Chain-of-Thought**, potentially outperforming RLVR in dense reward settings, with references to [Entropy-Regularized Process Reward Model](https://arxiv.org/abs/2412.11006) and related research.
   - **VinePPO** was cited as a way to provide refined credit assignment step-by-step, though worries remain that soft reward signals may encourage memorization rather than deeper reasoning.
- **Goodfire API Sparks Collaboration**: A member integrated a **Goodfire API** build matching **Llama 8B** with **VLLM** on the `gsm8k_cot_llama` task, inviting further development in the [lm-eval-harness repo](https://github.com/menhguin/lm-evaluation-harness/blob/main/lm_eval/models/goodfire.py).
   - The **MATH-Hard** dataset removal from Hugging Face caused leaderboard evaluation issues, with a [GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/2618#issuecomment-2583172531) suggesting a temporary fix.
- **Neel Nanda’s Mechanistic Tales**: Audio from **mechanistic interpretability** reading groups remains partially untranscribed, despite attempts with **Whisper** tools.
   - Listeners applauded a **Neel Nanda** podcast on SAEs, shared via [Spotify](https://open.spotify.com/episode/5XjHhNQxIb16eJZXGmbaCk?si=Z8LTnSo7QHGJkBxgGZbIJA), focusing on clearer internal model understanding.
- **Slurm Memory Moves**: **Slurm** flagged CPU-based OOM instead of GPU memory, resolved by using `--mem=0` or `--exclusive`, per [Slurm sbatch docs](https://slurm.schedmd.com/sbatch.html).
   - A user asked about estimating CPU RAM and cores needed per GPU for pretraining, prompting suggestions to track usage more systematically.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Cascade's Confounding Code**: Users voiced that **Cascade** is generating random outputs and mislabeling files, producing errors that hamper development even with [prompt engineering guidelines](https://docs.codeium.com/best-practices/prompt-engineering). They also complained about unpredictability, referencing *the 70% problem* as an example of how code may still stray from expected results.
   - Some participants suggested more rigorous testing to reduce mistakes, but they remain hopeful that **Cascade** can improve soon.
- **Custom Model Fever: Gemini Flash vs Current Options**: An enthusiastic crowd requested compatibility with **Gemini Flash**, lamenting that only pre-approved models can be used in **Windsurf** and pointing to [Codeium's feature requests](https://codeium.canny.io/feature-requests) for broader model support. They want the freedom to swap in new AI models without restrictions.
   - Despite multiple pleas, there's no formal timeline to add this feature, so some folks keep searching for alternative editors that accommodate wider AI usage.
- **Cursor Clash: Autocomplete Face-Off**: Users compared **Cursor** to **Windsurf**, applauding **Cursor** for sharper autocomplete suggestions while criticizing its reliability under stress, while *Windsurf's agentic features* draw praise for advanced workflows ([support docs](https://codeium.com/support)).
   - They concluded both require more stability, with some pushing for a different subscription structure instead of the current flow-credit model.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE Gains and Grumbles**: Some devs report brisker coding flows in **Cursor IDE** while others still encounter slowdowns and conflicting AI suggestions, especially during larger projects.
   - Community members have proposed restoring code states with checkpoints, pointing to [bug reports on the forum](https://forum.cursor.com/t/error-unauthorized-request/39861/28), with a clear call for more stable extension setups.
- **Codestral's Colossal Context**: The new **Mistral** release, [Codestral 25.01](https://mistral.ai/news/codestral-2501/), offers a massive 256k context window, promising dramatic improvements in code comprehension.
   - It's already supported in **Continue.dev**, and participants speculated that merging it with Cursor could streamline advanced code-generation features.
- **Collaborative Creations in Cursor**: Enthusiasts suggested joint efforts on AI-based apps, like a **Test Manager AI** agent, to sharpen both junior and senior dev skills.
   - They cheered the potential synergy, emphasizing hands-on learning and how it could spotlight **Cursor**’s capabilities for next-level coding collaborations.
- **Privacy Puzzle: Embedded Data Woes**: Concerns arose about **Cursor** storing chat embeddings, referencing [privacy-mode details](https://forum.cursor.com/t/concerns-about-privacy-mode-and-data-storage/5418) and NDAs in corporate settings.
   - Forums indicated that switching on 'Privacy Mode' prevents code uploads, but many requested deeper transparency on data management and server-side storage.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.6 Rolls Out Beta Tools**: LM Studio released version **0.3.6** with a new **Tool Calling API in beta** and an updated installer system, announced in [their blog](https://lmstudio.ai/blog/lmstudio-v0.3.6).
   - Users tested **Qwen2VL** and **QVQ** in local runs, logging issues and successes in the [official bug tracker](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues), with some praising the performance jump on **M4 Ultra** hardware.
- **Bartowski’s Sky T1 Teases 32B Performance**: Community members examined the [Bartowski/Sky-T1-32B-Preview-GGUF](https://huggingface.co/bartowski/Sky-T1-32B-Preview-GGUF) model for local coding tasks via LM Studio.
   - They reported stronger performance with **Q4_K** or **Q5_K** quantization but noted memory overhead on older rigs in user-submitted [feedback posts](https://gist.github.com/shermanhuman/2b9a82df1bab242a8edffe504bb1867c).
- **PowerMac G3 Gets an AI Makeover**: A user showcased a repurposed **PowerMac G3** running LM Studio, sparking hardware nostalgia and discussions about bridging classic cases with modern internals.
   - Others compared this build to [NVIDIA's Project DIGITS](https://www.nvidia.com/en-us/project-digits/) in terms of resource usage, with some advocating for dedicated GPUs instead.
- **Phi 3.1 Mini 128k Extends Context Boundaries**: Adventurous testers tried the [Phi 3.1 Mini 128k model](https://lmstudio.ai/model/phi-3.1-mini-128k-instruct) in LM Studio for larger context requirements.
   - They discovered moderate system demands and recommended carefully managing VRAM for stable outputs, with tips posted on [LM Studio docs](https://lmstudio.ai/docs/basics/chat).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Claude Takes an "Angry" Turn**: Some users noticed the **Claude** model adopting a more unapologetic style, with repeated usage of words like 'direct' and 'helpful' in responses, fueling jokes about an 'angry AI' persona.
   - One comedic tweet claimed a new Claude model had launched, drawing skepticism but sparking laughter about possible "secret updates" ([Tweet from Jacques](https://x.com/jacquesthibs/status/1878851967981887736)).
- **Hyperparameter Tuning Services Ignite Curiosity**: A question about automated solutions for hyperparameter search got traction, highlighting **Bayesian optimization** and the complexity of debugging training issues.
   - Some stressed the need for rigorous tests to catch hidden pitfalls, with speculation about eventual 'Hyperparam-as-a-Service' offerings.
- **Qwen 0.5B Stumbles on Math**: The smaller **Qwen 0.5B** model excelled at certain tasks yet often produced nonsensical answers or fell into endless loops ([kz919/QwQ-0.5B-Distilled](https://huggingface.co/kz919/QwQ-0.5B-Distilled)).
   - People wondered whether **Generative Knowledge Distillation (GKD)** introduced unintended quirks, noting confusion over how it differs from regular distillation.
- **MobileLLM Shakes Up Smaller Models**: **MobileLLM**'s paper suggested label-based training outperformed standard distillation for compact on-device language models ([MobileLLM on arXiv](https://arxiv.org/abs/2402.14905)).
   - This triggered deeper questions about whether synthetic data or advanced distillation methods will remain important for low-parameter models.
- **Element-wise Attention Sparks Discussion**: A paper titled **Element-wise Attention Is All You Need** proposed a new approach that promises lower training complexity while preserving quality ([arxiv.org/abs/2501.05730](https://arxiv.org/abs/2501.05730)).
   - Several engineers weighed the possibility that such a mechanism could reshape standard attention-based architectures for more efficient inference, fueling hopes for next-level improvements.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **StackBlitz Sparks with a Teaser Tweet**: We saw a [tweet from StackBlitz](https://x.com/stackblitz/status/1878818461905739994) referencing progress with **Bolt.new** announcements, generating curiosity among devs.
   - Some participants speculated about upcoming improvements but no detailed info was confirmed, leaving watchers energized for official news.
- **Stripe Strides into Bolt**: Reports indicated **Stripe integration** is on the way, with some folks already achieving success and calling it a **major plus** for their setups.
   - Others faced hiccups with code merges, referencing YouTube tutorials for fixes and even switching to **PayPal** as a backup option.
- **Prompting Pain and Gains**: Multiple users lamented lost code whenever new features were added, highlighting solutions like enabling **diffs** for stable expansions.
   - They referred to [The Ultimate Guide to Prompting with Bolt](https://docs.google.com/document/d/1SwlpZH1SotqPg2KbZqzWPdpBbs6aKIqMDspSCBCD1iQ/edit) for best practices, sharing comedic remarks like *'I keep pushing my products forward past a certain point.'*
- **Token Crunch Woes**: Excessive token usage hit a nerve, with one user burning **1.5 million tokens** on a single overlay, prompting calls for leaner prompts.
   - Demands for cheaper reloads and promo codes grew louder, with [a YouTube tutorial on saving tokens](https://youtu.be/ayagXgAShSk) circulating as a money-saving approach.
- **Webinar Whirlwind**: A free live training on **AI LLM Apps with Bolt** was announced for Tuesday at 10 AM EST, guiding devs in building structured, dynamic apps.
   - Organizers pointed to environment setup tips, referencing [How to Build Next-Level AI Apps with No Code](https://www.reinventing.ai/next-level-ai-apps-no-code) for further support.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **UK's Big Bill: Doubling Productivity**: The UK government invests £14 billion into **AI** to double **productivity** within three years, stirring debates over budget allocation and potential workforce displacement.
   - Critics question whether the funds could be more effectively directed elsewhere and warn against **AI** replacing human roles.
- **Claude & Gemini Conquer ChatGPT in Minecraft**: **Claude** and **Gemini** outperformed **ChatGPT** in a Minecraft contest, highlighting stronger reasoning and planning skills when handling complex tasks.
   - Observers voiced concern about **ChatGPT**'s performance gap and its implications for GPT-based models in competitive scenarios.
- **Codestral Debuts with 256k Context**: A new **codestral** model launched on the **Mistral** API, claiming a 256k context capacity and sparking curiosity about comparisons to **GPT-4**.
   - Members wait to see if its features synergize with upcoming canvas enhancements, leaving its practical impact under discussion.
- **Table Turmoil: GPT vs OCR**: Users reported **GPT** repeatedly misaligning wide table data, averaging around 60% accuracy, while pointing to tools like **Amazon Textract** for more consistent results.
   - They noted the model’s erratic performance in parsing complex layouts, prompting talk of better data formats or 'trickery' to improve outcomes.
- **Custom AI Agents at Work**: Participants explored **embedded AI** solutions for client-facing support, suggesting **n8n** and **flowise** while considering integration with Slack and WhatsApp.
   - They discussed challenges related to service costs and provider reliability, emphasizing practicality in deploying robust **AI** agents.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Mobile Magic & $50 Perk**: The team invites participants for a remote interview about the **NotebookLM mobile experience** on **January 14–15**, with sign-ups at [this screener form](https://forms.gle/75gYapqbgCmxXiJL6) and a **$50** or Google merch voucher for completion.
   - Community members look forward to sharing usage insights, aiming to shape **NotebookLM**'s mobile features through direct feedback.
- **Audio Overviews & Gift Codes**: A quick [~5 minute screener](https://forms.gle/NBzjgKfGC24QraWMA) is gathering feedback on **Audio Overviews**, rewarding a **$10** gift code for completing the follow-up survey.
   - Participants want to refine clarity and style of these AI-generated summaries, hoping to match user expectations for reliable audio content.
- **Easy Podcasting with Akas**: Users explored [Akas](https://akashq.com) for uploading **AI-generated podcasts**, bypassing strict login requirements on **NotebookLM**.
   - They enjoyed simpler distribution models, letting them share conversation-based content more freely with others.
- **Multiple Sources & Citation Confusion**: Some discovered **NotebookLM** struggles with referencing multiple files, causing frustration around citation links and repeated details.
   - Workarounds include careful doc naming and prompts, though results remain mixed for complex notebooks.
- **Embedding NotebookLM & Broader Uses**: A user proposed placing **NotebookLM** on websites like Google Sites to extend functionality beyond personal note-taking.
   - Others saw potential for broader adoption in educational or group settings, highlighting more open collaboration.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Gallop Over Pony Models for Illustrious Imagery**: While **Pony XL** claims strong tag cohesion, it disappoints in final outputs, prompting creators to prefer **Illustrious** and also mention **JuggernautXL** plus [RealVisXL v5](https://civitai.com/models/139562/realvisxl-v50) for more realistic images.
   - Participants suggested more refined datasets to fix the subpar performance, highlighting the significance of thorough testing before adopting new models.
- **Dreambooth Falls as Koyha_ss & OneTrainer Rise**: Creators are abandoning **Dreambooth** due to outdated methods and leaning on **Koyha_ss** plus **OneTrainer**, referencing a [FLUX training tutorial](https://www.youtube.com/watch?v=FvpWy1x5etM) for advanced steps.
   - Some recommended using 50–150 images for enhanced **character-specific Loras**, finding these newer tools more reliable than older tutorials.
- **High-Res Magic with Hires Fix**: Teams found that generating at lower resolutions and then applying **hires fix** at 1024x1024 yields superior clarity, supported by [Reddit discussions](https://www.reddit.com/r/StableDiffusion/comments/14x6o2c/finally_figured_out_how_to_create_realistic).
   - They observed direct high-resolution generation often duplicates image elements, reinforcing the use of incremental upscale to maintain image coherence.
- **Extensions Expand with sd-webui-regional-prompter**: Various tools like **sd-webui-regional-prompter** and [Forge Webui's sd-forge-couple](https://github.com/Haoming02/sd-forge-couple) advanced image slicing and attention control in **Stable Diffusion**.
   - Users stressed correct installation procedures, typically via git cloning into the right folders, to dodge scamming links floating around.
- **Stable Point Aware 3D Sparks Swift Edits**: **Stable Point Aware 3D (SPAR3D)** from [Stability AI](https://stability.ai/news/stable-point-aware-3d) promises real-time object editing and full structure creation from a single image in under a second.
   - Many were enthusiastic about rapid prototyping capabilities, seeing it as an important step for integrating **3D generation** with 2D diffusion workflows.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI Models: Cost vs. Elo explained**: The newly shared [LLM elo vs pricing chart](https://docs.google.com/spreadsheets/d/1x9bQVlm7YJ33HVb3AGb9qlDNkvTy9CyOFZoah0kr3wo/edit#gid=0) compares **o1-preview**, **GPT-4o**, and others in terms of cost and performance, detailing advanced Elo scores and monthly subscription pricing. It underscores that paying more doesn't always guarantee better results, especially at higher usage scales.
   - Community members celebrated the chart's clarity, with one stating *'it’s notable how predictive the Lmsys Elo vs $ curve is,'* referencing correlations found in **MMLU** benchmarks.
- **Copilot’s Waitlist Wiped**: Satya Nadella announced there is no more waitlist for **GitHub Copilot Workspace** on [X](https://x.com/satyanadella/status/1878578314115473577), enabling immediate agentic coding. It highlights the push for broader AI adoption by dropping sign-up barriers.
   - This move resonates with the community’s call for deeper integration, as some see it as a leap toward **autonomous development flows**. Others anticipate cost shifts, referencing **$20/month** plans vs. premium tiers.
- **Lightning-Fast Llama 3 Benchmarks**: New speed tests for **Llama 3.3** 70B hit **652 tokens/s** on SambaNova's custom **SN40L** hardware, surpassing conventional GPU setups. Observers view this as a major win for AI performance in 2025, potentially reshaping HPC.
   - A tweet from [Santiago](https://x.com/svpino/status/1878797424590012907) called this *'the fastest I've seen Llama 3.3 running anywhere,'* fueling excitement about multi-model concurrency. Meanwhile, user anecdotes highlight faster fine-tuning with reduced GPU hours.
- **Raspberry AI’s Retail Round**: Bryan Kim from **a16z** announced a new investment in **Raspberry AI**, an end-to-end generative design platform designed for retail. The vision focuses on automating product ideation, with key emphasis on speed and customization.
   - He explained the motivation in a [tweet](https://x.com/kirbyman01/status/1878844418972885077), highlighting the venture's potential for scaling. The news spurred conversation about funding momentum, with some praising how specialized solutions can thrive in the retail sector.
- **O1 Shifts from Chat to Reports**: Recent discourse frames **O1** as more than just a chat model, encouraging usage akin to a *report generator*. **Ben Hylak** underscored how rethinking prompts reveals deeper outputs, referencing Sam Altman’s stance on alternative usage.
   - A [guest post](https://www.latent.space/p/o1-skill-issue) on O1 reached the Hacker News front page, illustrating widespread interest in this perspective. Participants applauded the pivot, with one noting *'it really is mind-blowing when you know how to use it.'*



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.71.0 Zooms Forward**: Aider v0.71.0 shipped new commands for chat mode switching and improved streaming output, boosting user engagement, as described in [release history](https://aider.chat/HISTORY.html).
   - Users praised simpler toggles between question and code modes, celebrating the persistent pretty output for triple-backtick edits.
- **DeepSeek's Funky Fails**: Multiple users reported that **DeepSeek** drifted into unresponsiveness, causing missed deadlines and frustration.
   - They demanded stable API performance, suggesting quick fixes to ensure reliability.
- **Configuration Curiosities & Prompt Caching Quirks**: A user discovered that `.aider.conf.yml` requires a dash instead of an underscore for `editor-model`, raising bigger questions about ignoring config files in repos.
   - Others shared that prompt caching only works if the exact same set of files is included, prompting talk of possible enhancements.
- **Quantization & Polyglot Talk**: Members highlighted **quantization** for neural networks, urging robust knowledge for coding tasks, and flagged certain C++ tests in the polyglot suite needing special compiler flags.
   - Participants compared performance of **O1** with **Sonnet**, fueling speculation about which model outperforms the other in coding scenarios.
- **New Tools: CodeGate & Always-On Assistants**: Secure code generation sparked conversation with [CodeGate](https://github.com/stacklok/codegate), aimed at privacy and security in CodeGen workflows.
   - Projects like [Deepseek AI Assistant](https://www.youtube.com/watch?v=zoBwIi4ZiTA) and [always-on-ai-assistant](https://github.com/disler/always-on-ai-assistant/) showcased continuous, background help for engineers.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Phi 4 Fanfare from Microsoft**: On **OpenRouter**, the new [Phi 4](https://openrouter.ai/microsoft/phi-4) from **Microsoft** appeared this week, boasting upgraded text generation, lower latency, and partial code-handling for AI applications.
   - Users note gains in general performance and discuss possible integration paths, pointing to **OpenRouter** as a hub for expanded experimentation.
- **Friday Agents Flex Framework**: The **Friday Agents** multi-agent **JavaScript** stack at [GitHub - amirrezasalimi/friday-agents](https://github.com/amirrezasalimi/friday-agents) rolled out, offering two core parts that simplify AI app development with built-in concurrency.
   - Developers praise its capacity for parallel tasks, suggesting **OpenRouter** model endpoints might bring even broader functionality to this structure.
- **Telegram Taps 200+ LLMs via DeVries**: The **DeVries AI Chatbot** at [devriesai.com](https://devriesai.com/) grants direct Telegram access to **200+ large language models** for $24.99/month, with a free trial to entice early adopters.
   - Community members highlight its ability to streamline multi-model usage, emphasizing the convenience of switching among various providers in a single chat feed.
- **Mistral’s Codestral Climbs Context Counts**: The new **Codestral** model from **Mistral**—unveiled at [mistral.ai/news/codestral-2501/](https://mistral.ai/news/codestral-2501/)—features a **262K** context and accelerated coding speeds but has been retracted from general release.
   - Participants mention it was briefly accessible before removal, spurring debate on whether it’s production-ready despite strong coding benchmarks.
- **LLM Cost Chat & Deepseek V3 Feedback**: Discussants compare different platform plans for large language model hosting and view **Deepseek V3** as a strong option with steady speed and fair pricing.
   - They also weigh performance quirks across various providers, noting the path to become a model host on **OpenRouter** as a key point of interest.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Anthropic’s $60B Ascent**: Recently, **Anthropic** soared to a **$60B** valuation, generating buzz about the future of **language model startups**, with speculation on upcoming product expansions and interest from **major investors**.
   - In community chatter, participants described it as *“massive hype”* for the entire AI sector, hinting that more high valuations could spark intense competition among potential contenders.
- **Sonar 3.3 Surfaces, But API Is MIA**: Members discovered **Sonar 3.3** in Perplexity’s web UI but not in the **public API**, raising questions on release timelines and official announcements.
   - Multiple users indicated interest in further **llama-3.1-sonar** variants, while *guessing about a 70B version* despite no formal Perplexity statement.
- **Perplexity vs Claude: The Model Muddle**: Enthusiasts argued over whether **Perplexity** outperforms **Claude** in real tasks, referencing anecdotal speed tests and user experiences with no definitive winner.
   - Some insisted **Claude** excelled in certain areas, while *Perplexity fans* lauded its overall interface and features like **citations** in **llama-3.1-sonar**, fueling continuing debates around reliability and performance.
- **Chips & Stacks: The 3D AI Craze**: Community members spotlighted emerging **AI chips**, including **MIT’s 3D-stacked designs**, emphasizing sharper data processing gains.
   - They expressed optimism that expanded memory in these upcoming chips will *enable more demanding local model hosting*, especially for **LLM** workloads.
- **Perplexity’s Pricey Predicament**: Users aired frustrations with **Perplexity**’s subscription tiers, comparing a **$200/month** plan to **ChatGPT**, while calling for more appealing pro-level costs.
   - Many reported slow performance and restricted **API** use, *suggesting* that Perplexity refine its pricing approach and boost stability to remain competitive.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Codestral 25.01 Climbs the Charts**: The newly upgraded **Codestral 25.01** soared to #1 on the LMsys copilot arena leaderboard, demonstrating higher efficiency and performance ([official news](https://mistral.ai/news/codestral-2501/)).
   - It scored **11%** on the Aider polyglot benchmark ([tweet reference](https://x.com/paulgauthier/status/1878886495609815054)), sparking concerns from members about how it stacks up against leading models.
- **Helium-1 Targets Mobile Scale**: **Kyutai’s Helium-1** emerged as a 2B-parameter backbone language model, focused on edge devices and supporting 6 languages ([announcement](https://kyutai.org/2025/01/13/helium.html)).
   - Contributors emphasized **privacy** and speed as main goals, noting Helium-1’s potential in personal AI systems with minimal latency.
- **Qwen 2.5-Math Models Multiply Accuracy**: The **Qwen 2.5-Math-PRM-72B** line introduced Process Reward Models to reduce errors in mathematical reasoning ([Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B)).
   - Members reported improvement in step-by-step logic, underscoring **less intermediate slip-ups** and consistently strong performance across math evaluations.
- **Sky-T1-32B-Preview Soars on a Budget**: The [Sky-T1-32B-Preview](https://novasky-ai.github.io/posts/sky-t1/) was trained for under **$450**, demonstrating reasoning on par with bigger proprietary models.
   - Its open codebase ([SkyThought GitHub](https://github.com/NovaSky-AI/SkyThought)) points toward more community-driven, **low-cost** development of advanced LLMs.
- **LoRa Fine-Tuning Boosts Qwen Instruct**: A member employed **LoRa** to fine-tune Qwen Instruct models on an out-of-distribution dataset, aiming to retain performance for domain-specific tasks.
   - They reported some training setbacks yet maintained optimism about LoRa’s capacity to adapt robustly in specialized use cases.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R+ Gains Momentum**: Cohere introduced new performance details for **Command R+**, referencing multiple blog posts like [Command R: RAG at Production Scale](https://cohere.com/blog/command-r) and [Introducing Command R7B](https://cohere.com/blog/command-r7b). The updates cover advanced features for enterprise-grade LLM tasks, with highlights on speed, context length, and easier fine-tuning.
   - Community discussions showcased **Command R+** usage in **Rust** and Python, praising efficiency for code generation, while linking to [the official docs](https://docs.cohere.com/v2/docs/command-r-plus) for deeper insights. One user said *“Command R+ makes complex queries more approachable”*, echoing broader excitement about improved workflows.
- **Large Datasets Approach at Cohere**: Some users tested uploading JSONL files up to **800MB** with more than **180,000 lines**, exploring feasible large-scale data flows. They discovered challenges in the dataset environment with hints that **enterprise-level** usage can require specialized solutions.
   - Members are curious about scaling data ingestion for training and fine-tuning, referencing expansions with **Command R+**. There's an active conversation about optimizing processes for **big data** ingestion, hoping official docs clarify best practices.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Claude & O1: Cohesive Collaboration**: Members shared that the only **O1 workflow** to succeed involves using **Claude** to clarify project goals, create directives, and define **interfaces** between functions. They emphasized that O1 handles **algorithms** effectively once properly prompted.
   - A participant mentioned doubts on whether this group is best suited for such in-depth O1 discussions, hinting at a mismatch of interests. This reflects a desire for more specialized focus on **O1** within the community.
- **Triton Tuning Tactics**: Efforts to optimize **Triton Puzzles** on real GPUs (citing [this repo](https://github.com/gauravjain14/mlcompilers_and_kernels/tree/main/triton_kernels)) included autotuning on **A100** vs **A30**, plus discussing memory constraints for large `num_stages`. Another user investigated kernel occupancy, raising concerns that multiple programs per CUDA block could affect performance for small data chunks.
   - They also explored improving **cross entropy** kernels to reduce overhead, referencing [Liger Kernel code](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py). Feedback on profiling and hyper-parameters reaffirmed Triton's flexibility, though consumer GPUs demanded careful attention to shared memory usage.
- **Cranking CUDA & HPC**: Members discussed installing **CUDA** on **Ubuntu**, referencing the [official guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu) and using the [Nsight Visual Studio Code edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition) plugin. The group noted curiosity about **Blackwell** thread block clustering and detailed query on **FA3** performance when comparing **H100** to **H200**.
   - They highlighted **GPU** intricacies such as block assignments, linking these learnings to HPC tasks across different compute architectures. Concerns around driver setup, plugin usage, and HPC scaling remained core topics of interest for participants.
- **Torch Trials & Triumphs**: A **UTF-8 decode issue** in PyTorch Profiler with Hugging Face transformer's trainer.py was noted, referencing [issue #64345](https://github.com/pytorch/pytorch/issues/64345). Discussion also focused on integrating **Flash Attention** with MultiheadAttention, plus the impact of **DDP** and **FSDP** on module usage outside the forward pass.
   - Members building a **large-model inference pipeline** used meta devices and cached intermediate states to manage memory, though accessing all layers per request posed a challenge. **NNSight** was highlighted as a method to stream activations on-demand, reducing out-of-memory pitfalls during advanced analysis.
- **Events & LLM Evolution**: Upcoming presentations cover **Flash Infer** on Jan 24, **Mosaic GPU** on Jan 25, **int8 matmul for Turing** on Feb 8, and **NVIDIA profiling** on Feb 14, among others, while a new **Maya** Multilingual Vision-Language Model was shared ([link](https://twitter.com/nahidalam/status/1866667770114609217)). Meanwhile, **Qwen2-VL** clashed with the **liger kernel**, prompting a transformers downgrade per [this issue](https://github.com/linkedin/Liger-Kernel/issues/515).
   - Meta posted GPU-centric job openings for **GenAI inference**, directing interested candidates to [their careers site](https://www.metacareers.com/jobs/1517576482367228/). Additional off-topic updates included **Sonoma AI speaker series**, creative fundraising ideas, and more candid GPU interests across the community.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Community Tackles MAX GPU & MAX-CV**: The first 2025 community meeting spotlighted **MAX GPU benchmarking** and **MAX-CV** during a lively Q&A, with a recording promised [here](https://discord.com/events/1087530497313357884/1300880439673884712).
   - Scheduling conflicts hindered some attendees, and **Chris Lattner** responded to queries while **Caroline Frasca** pledged a follow-up video update.
- **macOS Mojo Testing Ramps Up**: Volunteers ran **Mojo** code on macOS for cross-platform checks, stepping up collaboration through DMs.
   - They discovered **nightly docs** by switching version numbers at [the docs site](https://docs.modular.com), satisfying curious developers.
- **Async Proposals Stir Mojo Enthusiasm**: Two plans, [Structured Async for Mojo](https://github.com/modularml/mojo/pull/3945) and [Provided Effect Handlers](https://github.com/modularml/mojo/pull/3946), aim to integrate asynchronous features without sacrificing performance.
   - Contributors compared **Rust-inspired** async methods, fueling further conversation on concurrency for **Mojo**.
- **Mojo Compiler Crash Zapped**: A crash occurred while defining a list of structs implementing a shared trait, documented in [Issue #3944](https://github.com/modularml/mojo/issues/3944).
   - Dev feedback linked it to tricky initialization, prompting an official bug report and suggested code fixes.
- **Int8 to String Conversion Quirk**: A [Mojodojo guide](https://github.com/modularml/mojo/issues/3947) highlighted trouble converting **Int8** to string, surprising testers.
   - Conversations covered compile vs runtime type details, steering folks to [Modular docs](https://docs.modular.com/mojo/manual/types/) for clarity.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **A Substack Peeks at Agentic AI**: Thanks to [this Substack post](https://kanesimms.substack.com/p/what-agentic-ai-actually-is-a-deeply), readers can investigate how **agentic AI** is conceptualized and the complexities behind it.
   - Discussion was concise, but it sets the stage for more nuanced viewpoints on **AI's capacity** for decision-making and autonomy.
- **AzureOpenAI Integration Example Shines**: A code snippet revealed how to set up **AzureOpenAI** with explicit API credentials and parameters, referencing the [Azure OpenAI documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview).
   - The example illustrated direct usage patterns, showing how quickly engineers can get started with **Azure**'s service.
- **dspy.react and phi-4: Surprising Function Calls**: A user noted that **dspy.react** let **phi-4** run function calling, even though the model had minimal training on that capability.
   - Though not flawless, the demonstration suggested that basic function calling can be slotted into **phi-4** for flexible usage.
- **Voice AI Ambitions Circulate in DSPy**: A newcomer asked about using **DSPy** for voice AI, but learned there's currently no direct audio support.
   - They were pointed to [GitHub Issue #2037](https://github.com/stanfordnlp/dspy/issues/2037), which documents requests and potential future expansions for **voice** capabilities.
- **Prompt Performance Variations Spark Debate**: Some users compared **gemini-8b** prompts with those for **deepseekv3**, suspecting model-specific prompts might yield different outcomes.
   - Others noted that the same prompt design may not address core errors across distinct architectures, reinforcing the idea of **prompt specialization**.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Phi-4 File Frenzy**: A user requested a 'dummy' file for **Phi-4** finetuning and shared [this Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb), noting an upcoming **Phi-4 PR** that could make it unnecessary.
   - They expect the PR to be merged soon, suggesting the workflow might transition smoothly without the standalone file.
- **Adaptive Batching Buzz**: A contributor presented an [RFC for adaptive batching](https://github.com/pytorch/torchtune/pull/2199) in **Torchtune**, aiming to refine batch size dynamically.
   - They plan to incorporate feedback before moving forward with further alterations in the next iteration.
- **Instruct vs. Non-Instruct for Medical Gains**: A discussion arose about using an **instruct** or **non-instruct LLaMA model** for training with a 50B-token medical dataset, citing the 10B instruct version as a possible candidate.
   - They emphasized that extensive dataset curation and effective post-processing could be critical to achieving robust medical capabilities.
- **Data Quality Triumphs**: One member underlined that **data quality > data quantity**, suggesting well-processed datasets trump massive raw collections.
   - They proposed using other LLMs to gauge document relevance before dedicating large resources to training.
- **Mistral 7B Shown Effective**: A user shared research where **Mistral 7B** performed well for pretraining tasks on medical society guidelines.
   - They attributed these positive outcomes to curated datasets, highlighting the importance of well-chosen training material.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Instant MOOC Enrollment & No Fees**: Filling out the **SP 25 signup form** grants automatic enrollment in the LLM Agents MOOC at zero cost, letting everyone join without extra steps.
   - Organizers confirmed that it’s *completely free*, which energized prospective learners eager to jump in.
- **Anticipating Final Project Results**: The final project outcomes are expected later this month, possibly within a week, as indicated by course leads.
   - The community is on edge, eagerly awaiting official announcements on grading specifics and future awards.
- **January 27th Lectures: The Learning Begins**: The **weekly lectures** for the Spring 2025 LLM Agents MOOC will ignite on **January 27th**, setting a firm schedule for participants.
   - Instructors reminded everyone to mark calendars and come prepared for a high-octane learning experience.
- **Separate Google Forms Fuel Assignment Submission**: Each assignment in the MOOC requires its own Google Form, enabling accurate progress tracking via email.
   - Students must consistently use the *same email address* to streamline the grading process and avoid confusion.
- **Gauge Crash Course Difficulty with Fall 2024 Lectures**: The **Fall 2024 MOOC** materials at [this link](https://llmagents-learning.org/f24) offer a sense of the base-level content for newcomers.
   - Leads noted the *slightly harder* Spring session, but recommended reviewing archived lectures and the [Quizzes Archive - LLM Agents MOOC](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit) to feel fully equipped.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AI Builders Summit Showcases 40+ Speakers**: The AI Builders Summit announces over **40 speakers** in a 4-week online training, highlighting the use of small language models for enterprise work. Additional info from [@_odsc](https://twitter.com/_odsc) confirms **RAG-focused sessions** with experts like [@seldo](https://twitter.com/seldo).
   - Attendees plan to learn **scaling** strategies for retrieval-augmented generation (RAG) without sacrificing performance, gaining direct guidance from seasoned presenters.
- **AutoRAG Steps Up RAG Pipelines**: The newly introduced **AutoRAG** framework helps developers choose effective configurations for retrieval-augmented generation by systematically testing multiple methods. According to the paper, it provides a structured path for LlamaIndex users who want more precision in RAG setups.
   - Community members view **AutoRAG** as a notable enhancement, praising its potential to streamline pipeline decisions and refine performance.
- **LlamaIndex Engineer Needed for Bot Project**: A user seeks an engineer proficient with **LlamaIndex** to assist in designing a bot solution, offering paid consultation. Interested professionals were asked to share credentials via direct message.
   - Others emphasized that proven experience in **structured data retrieval** and prompt engineering could be critical for this role.
- **GraphRAG Graphs Only Nodes**: Some users found **GraphRAG** notebooks displaying only nodes and no connecting edges, even with default OpenAI models. This issue was linked to potential gaps in data or missed **fine-tuning** steps.
   - Suggestions included reviewing examples like the [property_graph_neo4j notebook](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_neo4j.ipynb) to confirm proper relationships and configurations.
- **Prompt Caching and Variable Tricks**: Multiple users discussed **prompt caching** for OpenAI models, noting it works in a built-in manner unlike the Anthropic example. They cited limited official references but suggested that caching occurs automatically for many calls.
   - Others explored adding dynamic variables to the `QuestionsAnsweredExtractor`, recommending **function mappings** within LlamaIndex to feed custom context with ease.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **EPUB Expeditions in GPT4All**: A user asked if GPT4All can read **.epub** files, and the group confirmed basic support but flagged issues with certain languages like **Chinese**.
   - They suggested referencing the [GPT4All docs](https://docs.gpt4all.io/gpt4all_desktop/settings.html#sampling-settings) for potential workarounds, emphasizing consistent language handling.
- **Jinja Prompt Puzzle for Llama**: A user struggled with creating a **Jinja prompt template** for a fine-tuned **Llama** model when `get_chat_template()` didn't work as expected.
   - They sought guidance on customizing prompt design in GPT4All, highlighting complexities in prompt engineering.
- **Context Length Constraints Raise Eyebrows**: Contributors confirmed **GPT4All** enforces about **2048 tokens** for conversation recall, truncating text if it exceeds that limit.
   - They noted this affects both chat input and file-based references, triggering careful planning for longer sessions.
- **Full-Chat Export Sorely Missed**: A user wanted a **full-chat exporting** feature to retrieve past conversation logs without manual copying.
   - The GPT4All team does not yet offer it and encouraged opening a request at the [GitHub issues page](https://github.com/nomic-ai/gpt4all/issues).
- **Remote GPT4All from Weak Laptops**: One user aimed to run GPT4All remotely by linking a less powerful laptop via **VPN** or a reverse proxy on a stronger desktop.
   - This approach leverages the main machine’s hardware, letting the user offload processing while preserving local convenience.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad's Tidy Tensor Compiler**: Participants explained how **Tinygrad** uses a minimal instruction set and **kernel fusion** for GPU optimization, referencing [toonygrad/PLAN.md](https://github.com/tinygrad/toonygrad/blob/master/PLAN.md).
   - They noted that these fused kernels execute on diverse hardware and likened the design to *LLVM approaches* for simplifying ML operations.
- **Monday’s #53 Meeting Moves**: Team members scheduled **Meeting #53** for **9:30 AM** in San Diego, addressing **DSP contracts**, **Python speed**, and **MLPerf BERT** assessments.
   - They mentioned future bounties on **Tensor cores** and **RetinaNet**, cautioning about driver quirks and **ONNX** integration.
- **Stale PRs & the FSDP Bounty Lock**: A call went out to close outdated pull requests, alongside a bounty discussion on **FSDP** in [PR #8571](https://github.com/tinygrad/tinygrad/pull/8571).
   - Bounty conditions highlighted *multi-GPU training* requirements, prompting analysis of scaling beyond a single GPU.
- **Checkpointing & Memory Management Magic**: A user asked about **activation checkpointing** methods to curb memory overhead in **Tinygrad** while preserving training efficiency.
   - They also sought ways to *free memory* for return tensors without fracturing the **gradient context**, highlighting a prominent need for resource handling tips.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Installation Triumph**: One user encountered **tiktoken** errors and missing Rust requirements while installing **Open Interpreter** via [Homebrew](https://brew.sh) and [pipx](https://github.com/pypa/pipx), eventually achieving a stable setup.
   - They offered a brief command list for a clean environment, reinforcing **pipx** as a straightforward way to isolate Python applications.
- **Command Blitz: Open Interpreter's Hidden Screen Feature**: After installation, a user confirmed **Open Interpreter** can run arbitrary commands, including video editing steps.
   - A lesser-known **screen control** function generated excitement about potential expansions, prompting curiosity around usage scenarios.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Stable Audio 3 Speeds to Open Source**: Developers announced **Stable Audio 3** will be [open source](https://vxtwitter.com/dadabots/status/1878505508711157791), trained on music, and geared toward creative audio projects.
   - Enthusiasts noted that this approach could strengthen community-driven collaboration, especially with a focus on reusing and remixing **music-based datasets**.
- **Seeking Hypertension Audio Dataset**: A member asked for a **dataset** to identify hypertension through audio recordings, requesting help in data collection for health-focused research.
   - They stressed the importance of **collaboration** to compile audio samples, hoping to address a gap in specialized health data.
- **Megatron Checkpoint Conversion Quest**: A user ran Megatron training and wants a script to convert **torch format** to **HF format** without relying on Nemo, saving them from manual hacking.
   - They labeled this as *“saving a lot of work”* and asked the community to share any existing checkpoint conversion code or references.
- **MegaTron-LM Clone for Reference**: A user cloned the official **NVIDIA MegaTron-LM** repo at commit `31a29b87` and mentioned training logs stored [here](https://fz-juelich.sciebo.de/s/Yh8Q8RRTxliupLh).
   - They noted that **permissions** block direct file uploads, prompting calls for alternate file-sharing methods to boost community input.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Axolotl AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1327369162376482877)** (1120 messages🔥🔥🔥): 

> `Fine-tuning Llama 3.3, Using Unsloth for AI Models, Performance Metrics, GPUs and Cloud Solutions, Chat Templates and Tokenization` 


- **Fine-tuning Llama 3.3 Model**: Users are seeking assistance in fine-tuning the Llama 3.3 70B model, with discussions on utilizing different methods such as Unsloth and AutoTrain.
   - It was noted that proper configuration of templates and handling of datasets are crucial for obtaining good results during training.
- **Utilizing Unsloth for AI Models**: Unsloth users highlighted its efficiency in model fine-tuning, especially with functionality for long context and reduced VRAM requirements.
   - There were recommendations for using specific notebooks for methods like ORPO to enhance model performance.
- **Performance Metrics of AI Models**: Participants discussed expected performance outcomes based on GPU configurations, with mentions of benchmark speeds for LLMs.
   - Concerns were raised regarding unusual outputs and potential issues with model prompts and configuration.
- **Choosing the Right GPU for AI Tasks**: A user inquired about alternatives to AMD GPUs for their needs and was advised that Unsloth doesn't currently support them.
   - Suggestions were made to focus on NVIDIA options for better compatibility and performance, as well as cloud-based solutions.
- **Importance of Chat Templates**: The necessity of proper chat templates for interaction with models like Phi-4 was emphasized to improve response quality.
   - Users were informed about the importance of these templates in generating coherent and relevant outputs during inference.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/ibm-granite/granite-3.1-8b-base">ibm-granite/granite-3.1-8b-base · Hugging Face</a>: no description found</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-1-8b-conversational-unsloth/notebook"> Kaggle Llama 3.1 8b Conversational Unsloth</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://www.gigabyte.com/Motherboard/TRX40-DESIGNARE-rev-10#kf">TRX40 DESIGNARE (rev. 1.0) Key Features | Motherboard - GIGABYTE Global</a>: no description found</li><li><a href="https://huggingface.co/datasets/Yahir/test">Yahir/test · Datasets at Hugging Face</a>: no description found</li><li><a href="https://unsloth.ai/blog/llama3-3">Fine-tune Llama 3.3 with Unsloth</a>: Fine-tune Meta&#x27;s Llama 3.3 (70B) model which has better performance than GPT 4o, open-source 2x faster via Unsloth! Beginner friendly.Now with Apple&#x27;s Cut Cross Entropy algorithm.</li><li><a href="https://x.com/UnslothAI/status/1877779176473944212">Tweet from Unsloth AI (@UnslothAI)</a>: You can finetune Phi-4 for free on Colab now!Unsloth finetunes LLMs 2x faster, with 70% less VRAM, 12x longer context - with no accuracy loss.GitHub repo: https://github.com/unslothai/unslothDocumenta...</li><li><a href="https://x.com/abacaj/status/1876315285428609240">Tweet from anton (@abacaj)</a>: xml is extremely effective and basically the only way I found to prompt LLM as agents properly</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/updating">Updating | Unsloth Documentation</a>: To update Unsloth, follow the steps below:</li><li><a href="https://huggingface.co/microsoft/phi-4/discussions/7">microsoft/phi-4 · Function Call support</a>: no description found</li><li><a href="https://huggingface.co/collections/unsloth/unsloth-4-bit-dynamic-quants-67503bb873f89e15276c44e7?">Unsloth 4-bit Dynamic Quants - a unsloth Collection</a>: no description found</li><li><a href="https://huggingface.co/unsloth/phi-4-GGUF">unsloth/phi-4-GGUF · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/releases/tag/2025-01">Release Phi-4 &amp; Bug Fixes · unslothai/unsloth</a>: Please update Unsloth if you&#39;re seeing significant or unusual loss results—the latest update fixes an issue caused by the new transformers version. See our new updating instructions herePhi-4 is ....</li><li><a href="https://huggingface.co/unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit">unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements">Unsloth Requirements | Unsloth Documentation</a>: Here are Unsloth&#x27;s requirements including system and GPU VRAM requirements.</li><li><a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto">Reward Modelling - DPO, ORPO &amp; KTO | Unsloth Documentation</a>: To use DPO, ORPO or KTO with Unsloth, follow the steps below:</li><li><a href="https://huggingface.co/microsoft/phi-4/blob/main/tokenizer_config.json#L774>">tokenizer_config.json · microsoft/phi-4 at main</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/pull/1533">Give GGUF the same filename as project name by sebaxakerhtc · Pull Request #1533 · unslothai/unsloth</a>: Why?Default filename is &amp;quot;unsloth&amp;quot; - and that&amp;#39;s nice, but when you have multiple models on HF and try to download them to OpenWebUI - they all have the same name like &amp;quo...</li><li><a href="https://gist.github.com/darkacorn/71658f280ea0fc0ad4b97d2a616f4ce8">100k test . exllama2(testbranch) + fa  1 - 100k in 128t steps</a>: 100k test . exllama2(testbranch) + fa  1 - 100k in 128t steps - gist:71658f280ea0fc0ad4b97d2a616f4ce8</li><li><a href="https://github.com/unslothai/unsloth/issues/698">protobuf version · Issue #698 · unslothai/unsloth</a>: Is there any reason why protobuf is pinned to less than 4? Some of the other packages I use require protobuf &gt;=4, so I cannot install unsloth together with the other packages. Just trying to unders...</li><li><a href="https://analyticsindiamag.com/ai-trends/6-open-source-llms-that-can-run-on-smartphones/">6 Open-Source LLMs That Can Run on Smartphones </a>: Maximise privacy and control by leveraging the power of LLMs on your smartphone without using the internet.</li><li><a href="https://github.com/huggingface/transformers/pull/34858">🧹 Remove deprecated RotaryEmbedding parts in the Attention layers by Cyrilvallez · Pull Request #34858 · huggingface/transformers</a>: What does this PR do?This cleans-up the (expired) deprecated cycle for the rotary embeddings and fully move them to the Model instead of the Attention. Also removes deprecated EmbeddingClasses, an...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1327465866161291326)** (23 messages🔥): 

> `AI Chatbot Creation, Transcribing Videos, AI for Language Learning, Llama Model Usage, Voice Modes in AI` 


- **Creating an AI Chatbot to Mimic a Streamer**: To develop an AI chatbot that acts as a streamer, you will need to transcribe your videos first for fine-tuning the conversational model and ensure real-time TTS functionality.
   - Utilizing models like [Google Voice Recognition](https://link.to.google-voice) or [Whisper](https://huggingface.co/whisper) can help in speech-to-text conversion, while TTS options include free models like **Fish Agent** and **Kokouro** or paid services like **Eleven Labs**.
- **Starting AI/ML as a Flutter Developer**: New developers in AI/ML, especially those using Flutter, are encouraged to try platforms like [DuckDuckGo](https://duck.ai) for easy access to LLMs without setup barriers.
   - Creating a structured learning pathway or using prototypes to explore LLM capabilities can help to clarify goals and desired outcomes within AI/ML.
- **Using Llama Models for Language Learning**: For learning new languages, using models like **Llama 3.1-70B** can be effective due to their ability to follow instructions well and generate creative content.
   - Experimentation with different personas and dynamic user-defined filters can also enhance the learning experience by making the interaction feel more coherent and engaging.
- **Insights on Transcribing for TTS**: For training a TTS system in languages outside of the mainstream, having over **10,000 hours** of correctly transcribed audio is crucial for cohesive output.
   - Utilizing models such as **Cosy** for languages like Chinese, Japanese, or Korean could streamline the language learning process.
- **Creative Scenarios with Llama**: Users enjoy experimenting with Llama's dynamic filters to create imaginative scenarios, which can yield humorous and engaging narratives.
   - Playing with various character prompts can lead to interesting storytelling, like Mrs. Whiskers chasing her cats while her husband performs a trombone solo when the mailman arrives.



**Link mentioned**: <a href="https://duck.ai">DuckDuckGo AI Chat at DuckDuckGo</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1327366553888362527)** (236 messages🔥🔥): 

> `Fine-tuning LLMs, Data Preparation and Augmentation, LoRA for Style Transfer, Challenges in AI and NLP, Using Pre-trained Models` 


- **Fine-tuning LLMs for Specific Styles**: Users discussed the possibility of fine-tuning LLMs to mimic specific authors' styles, with the process involving substantial data preparation and iteration.
   - Using tools like LoRA can simplify style transfer, but understanding the underlying model mechanics is crucial for effective fine-tuning.
- **Data Preparation Challenges**: Preparing text datasets involves curation and cleaning, which can be time-consuming for deploying fine-tuned models effectively.
   - Proper data preparation is essential, as it constitutes a significant portion of the effort in training LLMs to achieve desired outcomes.
- **Time Investment for AI Projects**: Users noted that starting from scratch to achieve usable results with LLMs requires approximately 6 weeks of learning and experimentation.
   - The complexity of AI projects necessitates a solid understanding of foundational concepts beyond surface-level knowledge.
- **Exploration of Options in AI Development**: Individuals considered potential avenues for hiring expertise for AI development, particularly around respectful use of copyrighted material.
   - While outsourcing can be an option, the costs may outweigh the recreational nature of personal projects.
- **Maintaining Energy for Projects**: Discussion highlighted the importance of energy and motivation in pursuing personal projects, especially in the context of chronic illnesses.
   - Participants stressed that maintaining health is critical, as fatigue can significantly impact one's ability to engage in time-consuming learning processes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/TinyLlama_(1.1B)-Alpaca.ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://ollama.com/vanilj/phi-4-unsloth">vanilj/phi-4-unsloth</a>: The Phi 4 model with fixed tokenizer from Unsloth</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit">unsloth/Llama-3.2-3B-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://cohere.com/llmu/what-is-semantic-search">What is Semantic Search?</a>: In this LLM University chapter, you’ll learn how to use embeddings and similarity in order to build a semantic search model.</li><li><a href="https://discuss.huggingface.co/t/perhaps-your-features-output-in-this-case-have-excessive-nesting-inputs-type-list-where-type-int-is-expected/135553">Perhaps your features (`output` in this case) have excessive nesting (inputs type `list` where type `int` is expected)</a>: I am also getting similar issue here.  ValueError: Unable to create tensor, you should probably activate truncation and/or padding with  &#39;padding=True&#39; &#39;truncation=True&#39; to have batche...</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-Math-7B-Instruct">unsloth/Qwen2.5-Math-7B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1qN1CEalC70EO1wGKhNxs1go1W9So61R5?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/1528">transformers 4.48 breaks Mistral FastLanguageModel.from_pretrained with NameError: name &#39;MistralConfig&#39; is not defined · Issue #1528 · unslothai/unsloth</a>: Greetings, I&#39;ve been trying to run the Mistral_v0.3_(7B)-Conversational notebook on my own server. I followed the notebook cells and was hit with NameError: name &#39;MistralConfig&#39; is not def...</li><li><a href="https://github.com/unslothai/unsloth/issues/787">How to load the fine-tuned lora adapter and the downloaded model from the local directory for inference? · Issue #787 · unslothai/unsloth</a>: Hello, Thank you so much for such great work! I need help with loading the lora adapter which had finished fine-tuning process. Here is my code: model, tokenizer = FastLanguageModel.from_pretrained...</li><li><a href="https://github.com/unslothai/unsloth/issues/934">Unable to load unsloth trained model saved to a local directory. · Issue #934 · unslothai/unsloth</a>: I created a tar file out of a unsloth fine-tuned model(base-model: unsloth/gemma-2b-bnb-4bit) using PEFT and pushed it to gcsBucket. I am downloading the artifacts from gcs bucket, extracting the f...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1327372236054855682)** (2 messages): 

> `Maya: Multilingual Vision-Language Model` 


- **Maya's Preprint Release Excites**: A member announced the release of the preprint for **Maya: Multilingual Vision-Language Model** on [Twitter](https://twitter.com/nahidalam/status/1866667770114609217).
   - Another member expressed enthusiasm, stating, *'this is pretty cool thanks for sharing*.
- **Community Reaction to Maya**: Community members showed positive engagement with the announcement of **Maya**, expressing interest in the model’s implications.
   - One commented, *'this is pretty cool thanks for sharing*, indicating excitement among peers.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1327402901647392818)** (11 messages🔥): 

> `DataCollatorForPromptCompletion, Unsloth Training Speed, Cybersecurity LLM for Deception, Fine-tuning in Unsloth, Research Submission Inquiry` 


- **DataCollatorForPromptCompletion constructs minimal input-output interaction**: The provided Python code for `DataCollatorForPromptCompletion` masks input tokens while preserving output, aiming for focused language model training.
   - This implementation mimics the `DataCollatorForCompletionOnlyLM` but emphasizes handling inputs without a designated separator.
- **Unsloth claims radical speed improvements for LLM training**: According to a blog post, **Unsloth** enables **30x faster** LLM training and reduces memory usage by **60%**, allowing for larger batch sizes without accuracy loss.
   - The introduction highlighted significant optimizations including custom Triton kernels that enhance performance across various GPU architectures.
- **Cybersecurity researcher proposes specialized LLM**: A cybersecurity researcher discussed building a specialized LLM for cyber deception operations, focusing on fine-tuning based on distinct personas.
   - This approach has already generated over **1k unique connections** and helped identify scams through immersive interaction with simulated adversaries.
- **Fine-tuning techniques in Unsloth revealed**: Discussion highlighted that fine-tuning in **Unsloth** is faster due to custom triton kernels and advanced mathematical algorithms, bolstered by linked resources.
   - Users pointed to specific blog posts detailing fine-tuning speed enhancements, with models now finetuned **14x faster** and significantly reduced VRAM usage.
- **Research submission query for cybersecurity talk**: A member solicited opinions on the desirability of attending their talk at a conference about automated cybersecurity deception, along with venues for submission.
   - Responses indicated interest in the talk, while suggestions on submission venues remained open-ended, highlighting the relevance of the topic.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/introducing">Introducing Unsloth</a>: no description found</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth update: Mistral support + more</a>: We’re excited to release QLoRA support for Mistral 7B, CodeLlama 34B, and all other models based on the Llama architecture! We added sliding window attention, preliminary Windows and DPO support, and ...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1327385193887305880)** (68 messages🔥🔥): 

> `SmolLM-Corpus release, Grouped-Query Attention analysis, VLMs with multiple image input, Context Attribution research, User introductions and expertise` 


- **SmolLM-Corpus is officially launched!**: The **SmolLM-Corpus** has now been shuffled and sharded into **23698** `jsonl.zst` files, which allows for easy streaming and in-memory decompression, totaling only **315GiB**.
   - The dataset includes subsets from **cosmopedia-v2** and **fineweb-edu-dedup**, and users are encouraged to check out the [Hugging Face link](https://huggingface.co/datasets/Avelina/smollm-corpus) for access.
- **Exploration of Grouped-Query Attention**: A discussion arose on whether low-rank approximation of Q.K^T is possible given almost orthogonal weight matrices in grouped-query attention, suggesting it allows for efficient calculations.
   - Users debated the feasibility of relating different query weight matrices, pointing out challenges in maintaining computational cost while attempting to preserve transformations.
- **Multiple Image Input in VLMs**: Members shared insights about various **VLMs** capable of handling multiple images in context, specifically mentioning **Pixtral** and the **Qwen** models.
   - It was confirmed that **Qwen VLs** can support this functionality, contributing to broader discussions about model capabilities.
- **Context Attribution Research Gaining Attention**: The concept of **context attribution** was discussed, referencing a method called **ContextCite** proposed in a recent paper that identifies which parts of context lead to model outputs.
   - Participants noted ongoing debates in the field, including the potential applications and implications of such research.
- **New Community Members Introduced**: Several new members introduced themselves, sharing backgrounds in MLOps, computational linguistics, and research interests, including alignment and statistical learning theory.
   - The community expressed excitement in welcoming newcomers and fostering collaborative discussions around AI developments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/iScienceLuvr/status/1831220742626939248">Tweet from Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr)</a>: ContextCite: Attributing Model Generation to Contextabs: https://arxiv.org/abs/2409.00729&#34;we introduce the problem of context attribution: pinpointing the parts of the context (if any) that led a ...</li><li><a href="https://huggingface.co/datasets/Avelina/smollm-corpus">Avelina/smollm-corpus · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1327370221165482035)** (738 messages🔥🔥🔥): 

> `Latro Model, Process Reward Models (PRMs), Chain-of-Thought (CoT) Reasoning, VinePPO Algorithm, Reinforcement Learning (RL)` 


- **Latro Model Offers Potential Improvements**: The Latro model aims to enhance reasoning in various domains, where its use of reinforcement learning combined with CoT generation could lead to better outcomes compared to traditional methods like RLVR.
   - Discussion highlighted the importance of evaluating its performance against existing frameworks like RLOO to measure improvements specifically in dense reward settings.
- **Challenges of Process Reward Models (PRMs)**: PRMs are explored for their ability to improve reasoning by mitigating intermediate errors, but face challenges in data annotation and evaluation methodologies.
   - Research indicates that methods like Monte Carlo estimation may yield inferior performance compared to LLM-as-a-judge techniques for ensuring correctness in reasoning steps.
- **Concerns Over Reward Signal Effectiveness**: Concerns were raised that using soft rewards based on log probabilities might not sufficiently guide the model to develop effective reasoning chains, potentially leading to memorization rather than understanding.
   - This highlights the need for more robust methods to ensure that reasoning quality is retained and improved during training.
- **Impact of KL Regularization in RL Training**: KL regularization is discussed as a means to stabilize learning in Latro models, making it necessary to carefully consider how it interacts with the overall training objective.
   - Through progressively adjusting the action space, the training dynamics could potentially produce denser and more informative rewards.
- **Evaluation of New Algorithms Like VinePPO**: VinePPO is introduced as a novel approach that enhances credit assignment in RL, allowing for better performance on reasoning-heavy tasks compared to traditional value networks.
   - The method focuses on sampling from each reasoning step to provide a richer reward signal, emphasizing the importance of detailed evaluation across varying datasets.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.05441">The GAN is dead; long live the GAN! A Modern GAN Baseline</a>: There is a widely-spread claim that GANs are difficult to train, and GAN architectures in the literature are littered with empirical tricks. We provide evidence against this claim and build a modern G...</li><li><a href="https://x.com/lifan__yuan/status/1875020673476944337?s=46">Tweet from Lifan Yuan (@lifan__yuan)</a>: @rm_rafailov The process reward here is defined as “advantage”, namely the difference of Qs. So intuitively, even if there is a constant baseline, it should be canceled out and have no effect on the r...</li><li><a href="https://arxiv.org/abs/2412.11006">Entropy-Regularized Process Reward Model</a>: Large language models (LLMs) have shown promise in performing complex multi-step reasoning, yet they continue to struggle with mathematical reasoning, often making systematic errors. A promising solut...</li><li><a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>: Process Reward Models (PRMs) emerge as a promising approach for process supervision in mathematical reasoning of Large Language Models (LLMs), which aim to identify and mitigate intermediate errors in...</li><li><a href="https://arxiv.org/abs/2106.06431">Offline Reinforcement Learning as Anti-Exploration</a>: Offline Reinforcement Learning (RL) aims at learning an optimal control from a fixed dataset, without interactions with the system. An agent in this setting should avoid selecting actions whose conseq...</li><li><a href="https://arxiv.org/abs/2501.06282">MinMo: A Multimodal Large Language Model for Seamless Voice Interaction</a>: Recent advancements in large language models (LLMs) and multimodal speech-text models have laid the groundwork for seamless voice interactions, enabling real-time, natural, and human-like conversation...</li><li><a href="https://arxiv.org/abs/2501.07542">Imagine while Reasoning in Space: Multimodal Visualization-of-Thought</a>: Chain-of-Thought (CoT) prompting has proven highly effective for enhancing complex reasoning in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Yet, it struggles in complex ...</li><li><a href="https://arxiv.org/abs/2310.04363">Amortizing intractable inference in large language models</a>: Autoregressive large language models (LLMs) compress knowledge from their training data through next-token conditional distributions. This limits tractable querying of this knowledge to start-to-end a...</li><li><a href="https://arxiv.org/abs/2410.01679">VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment</a>: Large language models (LLMs) are increasingly applied to complex reasoning tasks that require executing several complex steps before receiving any reward. Properly assigning credit to these steps is e...</li><li><a href="https://arxiv.org/abs/2402.05808">Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning</a>: In this paper, we propose R$^3$: Learning Reasoning through Reverse Curriculum Reinforcement Learning (RL), a novel method that employs only outcome supervision to achieve the benefits of process supe...</li><li><a href="https://arxiv.org/abs/2408.16737v2">Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling</a>: Training on high-quality synthetic data from strong language models (LMs) is a common strategy to improve the reasoning performance of LMs. In this work, we revisit whether this strategy is compute-op...</li><li><a href="https://arxiv.org/abs/2203.11171">Self-Consistency Improves Chain of Thought Reasoning in Language Models</a>: Chain-of-thought prompting combined with pre-trained large language models has achieved encouraging results on complex reasoning tasks. In this paper, we propose a new decoding strategy, self-consiste...</li><li><a href="https://arxiv.org/abs/2412.01981">Free Process Rewards without Process Labels</a>: Different from its counterpart outcome reward models (ORMs), which evaluate the entire responses, a process reward model (PRM) scores a reasoning trajectory step by step, providing denser and more fin...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1328229388457611294)** (5 messages): 

> `Mechanistic Interpretability Audio Content, Neel Nanda Podcast on SAEs, Weekly Mechanistic Interpretability Reading Groups` 


- **Audio Recordings from Weekly Reading Groups**: There are audio-only recordings of about a year of **weekly mechanistic interpretability reading groups** that some members were involved with.
   - One member was supposed to transcribe them using Whisper but only managed a subset due to implementation difficulties.
- **Neel Nanda's Insightful Podcast on SAEs**: A member recommended a podcast episode featuring **Neel Nanda**, who leads the mechanistic interpretability team at Google DeepMind, discussing his work on SAEs [here](https://open.spotify.com/episode/5XjHhNQxIb16eJZXGmbaCk?si=Z8LTnSo7QHGJkBxgGZbIJA).
   - Nanda emphasizes the need for **mechanistic interpretability**, as machine learning models can perform complex tasks without clear internal understanding.
- **Positive Reception of the Podcast**: One member acknowledged enjoying Neel Nanda's podcast on the first day it was released, indicating it resonated well with listeners.
   - This suggests a growing interest in the insights provided by leading figures in mechanistic interpretability.



**Link mentioned**: <a href="https://open.spotify.com/episode/5XjHhNQxIb16eJZXGmbaCk?si=Z8LTnSo7QHGJkBxgGZbIJA">Neel Nanda - Mechanistic Interpretability (Sparse Autoencoders)</a>: Machine Learning Street Talk (MLST) · Episode

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1327740240198238218)** (19 messages🔥): 

> `Goodfire API Implementation, MLQA Benchmark Clarity, Dataset Issues, GPT-4o Usage, Pre-commit Line Ending Issues` 


- **Goodfire API seeks contributors**: A member shared that they implemented a basic version of the **Goodfire API** within the Eval Harness and successfully matched **Llama 8B** with **VLLM** on the **gsm8k_cot_llama** task.
   - They invited others to collaborate, troubleshoot, or discuss the next steps in enhancing the API's functionality.
- **Clarification on doc_to_text field**: A member inquired about the **doc_to_text field** in YAML, specifically if it is meant to construct prompts based on certain inputs.
   - Others confirmed that it's indeed intended for prompt construction, ensuring clarity on its function.
- **MATH-Hard Dataset Disappearance**: Concerns about the **MATH-Hard dataset** being removed from the **lighteval account** on Hugging Face were raised, causing issues with leaderboard evaluations.
   - A potential workaround was shared via a GitHub [issue discussion](https://github.com/EleutherAI/lm-evaluation-harness/issues/2618#issuecomment-2583172531).
- **Guidance on GPT-4o Usage**: A member requested an example using the framing **with gpt-4o**, specifically for implementing a task using the `generate_until` function.
   - Advice was given to reference the **gsm8k** as a template for creating the task.
- **Fixing Mixed Line Ending Issues**: **Mixed line ending** failures in pre-commit checks were discussed, leading to concerns about commits being blocked.
   - Suggestions included running `pre-commit run --all-files` to auto-fix issues related to line ending characters specific to different operating systems.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/bbrabbasi">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/maziyarpanahi/status/1878491088199032913?s=46">Tweet from Maziyar PANAHI (@MaziyarPanahi)</a>: @ailozovskaya was MATH-Hard dataset removed from lighteval account on @huggingface? I can&#39;t use lm-eval-harness to run leaderboard eval anymore! :(</li><li><a href="https://colab.research.google.com/drive/14-KkodIIVdq5fB-rDBMKoAK3KiUOsrmB#scrollTo=qsKt8d6TVnC_">Google Colab</a>: no description found</li><li><a href="https://github.com/menhguin/lm-evaluation-harness/blob/main/lm_eval/models/goodfire.py">lm-evaluation-harness/lm_eval/models/goodfire.py at main · menhguin/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - menhguin/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/2618#issuecomment-2583172531">Can&#39;t find the dataset lighteval/MATH-Hard in the huggingface · Issue #2618 · EleutherAI/lm-evaluation-harness</a>: when I run the code to get the results of the tasks in the open llm leaderboard v2, I can&#39;t find lighteval/MATH-Hard in the huggingface, how should I solve this problem? Thanks！
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1327723143208505355)** (5 messages): 

> `Slurm CPU memory issues, Pretraining resource recommendations` 


- **Slurm reports OOM for CPU memory**: Members discussed that Slurm is indicating an **Out of Memory (OOM)** issue related to CPU memory, not GPU memory, prompting a recommendation to check CPU RAM availability.
   - Suggestions included using the `--mem=0` flag to request all available memory or allocating the node exclusively with the `--exclusive` flag to access all resources, related to [Slurm sbatch options](https://slurm.schedmd.com/sbatch.html).
- **Successful resolution using Slurm flags**: One member confirmed that using the suggested flags resolved their issue when trying the **6.7B config** again, indicating the importance of correct resource allocation.
   - This experience highlights the effectiveness of properly setting resource flags in Slurm to prevent memory-related errors.
- **Request for pretraining resource estimates**: A member sought recommendations for the amount of **CPU RAM and CPU Cores** needed during pretraining per GPU, questioning if there are methods to estimate these requirements.
   - This indicates a growing interest in optimizing resource allocation for training large models effectively.


  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1327371427829125243)** (141 messages🔥🔥): 

> `Windsurf Login Issues, Codeium Pricing and Subscription, Feature Requests and Feedback, Technical Errors and Troubleshooting, User Experience and Support Concerns` 


- **Windsurf Users Face Recurrent Login Issues**: Several users, including @coyote775799 and @junyuh, reported trouble logging into the Windsurf application, often resulting in manual login failures and unresponsive features.
   - Solutions suggested include using AppCleaner for complete uninstallation and removing residual files, which seem to resolve many of these connectivity problems.
- **Clarification on Codeium Pro Subscription Usage**: @osplus6235 expressed frustration with not being able to access their Pro subscription on multiple devices, despite being logged in with the same account.
   - After ten days without a response from support, they sought escalation from moderators to resolve the matter.
- **Disable Users Highlight Problems with Codeium's Service**: Users like @johnreel_ reported high credit consumption due to issues with Windsurf that disrupt workflow, suggesting dissatisfaction with the pricing model.
   - As a result, some are considering exploring alternatives to Codeium to avoid ongoing frustrations.
- **Feature Requests for Windsurf Enhancements**: Users like @shivamkumar inquired about the potential for custom model support in Codeium, specifically mentioning the desire for compatibility with Gemini Flash.
   - The response indicated that only available models are supported, though updates may bring new features in the future.
- **Technical Errors Prompt Diagnostics Requests**: Issues were raised regarding 'Abnormal connection close by server' for Codeium, prompting users to gather diagnostics for submission.
   - This underscores ongoing connectivity concerns that have impacted multiple users' experiences with the platform.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://freemacsoft.net/appcleaner/">AppCleaner</a>: no description found</li><li><a href="https://bit.ly/8HWebAI3">8 Highlights From Our Webinar “The Intersection Of AI Agents &amp; Web3”</a>: On December 17th, we hosted the webinar, “The Intersection of AI Agents &amp; Web3”, focused on how AI Agents leverage web3 infrastructure to…</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>: Contact the Codeium team for support and to learn more about our enterprise offering.</li><li><a href="https://codeium.com/faq">FAQ | Windsurf Editor and Codeium extensions</a>: Find answers to common questions.</li><li><a href="https://codeium.com/terms-of-service-individual">Terms of Service: Individual &amp; Pro | Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://codeium.com/blog/copilot-trains-on-gpl-codeium-does-not">GitHub Copilot Emits GPL. Codeium Does Not.</a>: Demonstrating that GitHub Copilot trains on non-permissive licenses and is unable to filter out suggestions properly, while Codeium does not expose users to legal risk.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1327378707714801724)** (592 messages🔥🔥🔥): 

> `Windsurf functionality issues, Cascade performance, Subscription model concerns, User experiences with AI tools, Feature requests for Windurf` 


- **Windsurf encounters functionality issues**: Users reported experiencing persistent errors, such as Cascade editing files incorrectly and encountering 'internal errors' during operations.
   - There have been complaints about Windsurf generating incorrect outputs and failing to recognize certain files, leading to frustration among users.
- **Cascade's performance raising concerns**: Some users expressed dissatisfaction with Cascade's recent performance, stating it consistently hallucinates and produces errors despite strict rules being in place.
   - Users noted that while Cascade was previously reliable, it has become less predictable and more error-prone, impacting their workflow significantly.
- **Concerns over subscription model**: Users discussed dissatisfaction with the subscription pricing structure, feeling it does not reflect their actual usage patterns, especially concerning flow credits.
   - Many advocate for a more flexible model, suggesting alternatives that would prevent wasted credits from casual inquiries.
- **Mixed experiences with AI tools like Cursor**: Several users compared Windsurf to Cursor, noting Cursor's autocomplete functionality as superior in some respects but still expressing concerns over its reliability.
   - Users suggested that while Cursor may perform better under certain conditions, they find Windsurf beneficial for specific tasks, highlighting the mixed preferences.
- **Feature requests and enhancements**: There was a call for improvements in features such as the ability to set custom AI rules, optimal usage of credits, and more robust context management.
   - Users expressed interest in enhancing the integration of external APIs and allowing for better rule definitions to improve the development experience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://addyo.substack.com/p/the-70-problem-hard-truths-about">The 70% problem: Hard truths about AI-assisted coding</a>: A field guide and why we need to rethink our expectations</li><li><a href="https://tenor.com/view/frustrated-angry-upset-bullshit-table-bang-gif-8128901">Frustrated Angry GIF - Frustrated Angry Upset - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.codeium.com/best-practices/prompt-engineering">Prompt Engineering - Codeium Docs</a>: no description found</li><li><a href="https://pastebin.com/Lk422FgE">Critical Documentation and WorkflowDocumentation ManagementMaintain a &#039;winds - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://tenor.com/view/wink-eye-turn-around-chewing-gif-23703707">Wink Eye GIF - Wink Eye Turn - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/indian-phahaha-gif-27058287">Indian Phahaha GIF - Indian Phahaha - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://codeium.com/">Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://codeium.canny.io/feature-requests">Feature Requests | Codeium</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://youtu.be/-qa7_oe5uWQ"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1327369983638114407)** (553 messages🔥🔥🔥): 

> `Cursor IDE Performance, New AI and Tools, Collaboration Projects, AI Rules and Guidelines, Cursor Extension Issues` 


- **Mixed Experiences with Cursor IDE**: Users have shared varied experiences with Cursor IDE's performance, with some noting improvements in AI operations while others report persistent issues like slowdowns and errors during coding sessions.
   - For instance, one user mentioned repeatedly encountering problems with Claude's recommendations not aligning with the current code, leading to significant debugging work.
- **New AI Tools and Models**: Several users discussed the emergence of new models in the AI landscape, such as Mistral's Codestral with a 256k context length, and Claude's ongoing updates that enhance code generation capabilities.
   - There's significant interest in how these advancements could aid developers, especially when integrated with tools like Cursor.
- **Collaboration on Projects**: Participants expressed enthusiasm about collaborating on projects, with some suggesting it could help improve skills and promote learning among junior and senior developers.
   - Ideas ranged from creating a Test Manager AI agent to various applications that could showcase Cursor's capabilities.
- **AI Rules and Optimization**: Users discussed customizing AI rules to improve output quality and response accuracy in Cursor, emphasizing the importance of detailed prompting.
   - One user shared an extensive set of guidelines aimed at enhancing Claude's reasoning and analysis processes.
- **Issues with Extensions and Setup**: Some users have reported challenges with Cursor extensions and settings, such as installation errors and situations where the IDE does not behave as expected.
   - Suggestions included using checkpoints to restore previous states and troubleshooting common issues like slow performance or unresponsive features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youactuallysuck.com/message/161A8yb1aw2t">You Actually Suck - Anonymous Email Feedback</a>: no description found</li><li><a href="https://youactuallysuck.com/">You Actually Suck - Anonymous Email Feedback</a>: no description found</li><li><a href="https://autoblogpilot.com/">AutoBlogPilot</a>: no description found</li><li><a href="https://21st.dev/serafimcloud/splite">Spline Scene | 21st.dev - The NPM for Design Engineers</a>: A React component that integrates Spline 3D scenes.Demo component combines interactive 3D visualization with a spotlight effect and responsive text content.Features:	•	Lazy-loaded Spline integration	•...</li><li><a href="https://www.latent.space/p/o1-skill-issue">o1 isn’t a chat model (and that’s the point)</a>: How Ben Hylak turned from ol pro skeptic to fan by overcoming his skill issue.</li><li><a href="https://mistral.ai/news/codestral-2501/">Codestral 25.01</a>: Code at the speed of Tab. Available today in Continue.dev and soon on other leading AI code assistants.</li><li><a href="https://21st.dev/s/background">Backgrounds Components | 21st.dev - The NPM for Design Engineers</a>: Ready-to-use backgrounds React Tailwind components inspired by shadcn/ui.</li><li><a href="https://x.com/ryandavogel/status/1878240606289338759?s=46">Tweet from Ryan Vogel (@ryandavogel)</a>: We are a year away from AGI and OpenAI is hiring people for $385k base to write React</li><li><a href="https://x.com/kregenrek/status/1878487131099898269?s=46">Tweet from Kevin Kern (@kregenrek)</a>: Introducing Codefetch for DevelopersTurn code into Markdown for LLMs with one simple terminal command.Use it in bolt .new, cursor and many more AI coding tools.→ Chat with your codebase→ Save tokens→ ...</li><li><a href="https://forum.cursor.com/t/which-model-is-the-best-for-claude-sonnet/35148">Which model is the best for claude sonnet?</a>: Hi Everyone.  I just upgraded my plan to pro.  I can see that models     I used claude-sonnet-20241022.  But, my question is which model is the latest between claude sonnet 3.5 and that sonnet-2024102...</li><li><a href="https://forum.cursor.com/t/error-unauthorized-request/39861/28">ERROR: Unauthorized request</a>: Everyone, sorry for the delay in a post on this.  Unfortunately, over the last couple days, we’ve seen a high volume of abuse coming from temporary email addresses that got to the point where it was a...</li><li><a href="https://github.com/lvllvlTlvl/cursor-conversation-manager">GitHub - lvllvlTlvl/cursor-conversation-manager: A Python library for managing and reconstructing conversation histories from the Cursor IDE. The files are separated into directories by context.</a>: A Python library for managing and reconstructing conversation histories from the Cursor IDE. The files are separated into directories by context. - lvllvlTlvl/cursor-conversation-manager</li><li><a href="https://www.youtube.com/watch?v=TuO21CLrluU"> - YouTube</a>: no description found</li><li><a href="https://github.com/sksarvesh007">sksarvesh007 - Overview</a>: sksarvesh007 has 62 repositories available. Follow their code on GitHub.</li><li><a href="https://www.nico.fyi/blog/tailwind-css-group-modifier-to-prevent-react-rerender?ref=dailydev">How to prevent re-render in React with Tailwind CSS</a>: Learn how to use Tailwind CSS&#x27;s group modifier and data attributes to create dynamic UI elements without React re-renders. Improve performance and simplify your code.</li><li><a href="https://www.youtube.com/watch?v=nxss50uZgE0"> - YouTube</a>: no description found</li><li><a href="https://elthos.com">no title found</a>: no description found</li><li><a href="https://forum.cursor.com/t/where-is-the-data-generated-by-codebase-indexing-stored-locally/22517">Where is the data generated by Codebase indexing stored locally?</a>: Where is the data generated by Codebase indexing stored locally? If I connect to a remote server via ssh, is the data stored on the remote server? Or is it stored locally?  I would like to know the pa...</li><li><a href="https://daily.dev/blog/cursor-ai-everything-you-should-know-about-the-new-ai-code-editor-in-one-place">Cursor AI: The AI-powered code editor changing the game</a>: Explore how Cursor AI transforms coding with advanced AI features, enhancing productivity and code quality for developers of all levels.</li><li><a href="https://blog.cloudflare.com/sqlite-in-durable-objects/">Zero-latency SQLite storage in every Durable Object</a>: Traditional cloud storage is inherently slow because it is accessed over a network and must synchronize many clients. But what if we could instead put your application code deep into the storage layer...</li><li><a href="https://forum.cursor.com/t/concerns-about-privacy-mode-and-data-storage/5418/15">Concerns about Privacy Mode and Data Storage</a>: The computed embeddings from your code is stored on Cursor servers and contain a lot of information bits about your code. They currently don’t seem to address the concerns and think these embeddings a...</li><li><a href="https://www.cursor.com/privacy">Privacy Policy | Cursor - The AI Code Editor</a>: If you have any questions or feedback, please email us at hi@cursor.com.</li><li><a href="https://stackoverflow.com/questions/9455774/is-it-a-bad-idea-to-store-sqlite-cursor-in-android">Is it a bad idea to store SQLite Cursor in Android?</a>: I am trying to implement a dictionary application on Android. As user enters a letter into the EditText (or deletes a letter), application queries the database and shows all entries begining with the </li><li><a href="https://www.youtube.com/watch?v=KenChh4p0nI">Build an app with Storage using AI in 10 min (Cursor AI, Claude AI, Firebase Storage)</a>: Let&#39;s learn how to set up Firebase Storage in your application to allow users to upload profile pictures.SUBSCRIBE for more! 👉 http://bit.ly/3zlUmiS 👈Let&#39;s...</li><li><a href="https://forum.cursor.com/t/chat-history-folder/7653">Chat history folder</a>: Hi all, where is the chat history stored?  I love Cursor but I just changed PC, while working on a same project, same folder but chat history is gone.  Thanks!</li><li><a href="https://www.reddit.com/r/androiddev/comments/bwxh4r/sqlite_cursor_with_a_large_amount_of_data/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://forum.cursor.com/t/concerns-about-privacy-mode-and-data-storage/5418">Concerns about Privacy Mode and Data Storage</a>: Hi everyone,  I’m currently using Cursor.sh to assist with my development work on a client project. My client has raised concerns about potential NDA violations and data protection, even though I’ve b...</li><li><a href="https://forum.cursor.com/t/how-do-i-export-chat-with-ai/144">How do I export chat with AI?</a>: How do I export or share a chat with AI?  I want to be able to share my chat with AI same as ChatGPT does in order to share it with my coworkers.  Or at least be able to export it. Currently I have to...</li><li><a href="https://forum.cursor.com/t/concerns-about-privacy-mode-and-data-storage/5418/6">Concerns about Privacy Mode and Data Storage</a>: Just adding the recent tl;dr version of the privacy policy at:  https://www.cursor.com/privacy   TLDR    If you enable “Privacy Mode” in Cursor’s settings, none of your code will ever be stored by us ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1327368022532554995)** (317 messages🔥🔥): 

> `LM Studio capabilities, Model performance comparisons, AI hardware discussions, Quantization effects on models, User experiences with coding models` 


- **Testing Local AI Models Effectively**: Users have shared insights on testing various locally hosted coding models, emphasizing the importance of model quantization for performance, with some quantizations turning good models ineffective.
   - A user created a comprehensive QA document with recommendations for running coding models effectively based on their testing experience.
- **Hardware Discussions and Recommendations**: Participants discussed the feasibility of using high-spec hardware, like the M4 Ultra, versus dedicated AI rigs, pondering the balance of GPU performance and cost-effectiveness for AI tasks.
   - Some users expressed that while dedicated hardware may outperform, the simplicity and versatility of high-spec Apple devices could provide a suitable alternative for various tasks.
- **Insights on Large Models**: It's noted that larger models like the 72B variants may exceed available VRAM on single GPUs but can be effectively managed across multiple devices.
   - Discussions highlighted the challenge of running such models while suggesting specific quantization strategies for optimal performance.
- **Quantization and Model Efficiency**: User feedback pointed to the critical nature of selecting appropriate quantization levels for model efficiency, with recommendations to favor Q4_K and Q5_K for better performance.
   - Participants shared anecdotes of the discrepancies in model performance based on the quantization method used.
- **Community Engagement and Feature Requests**: Users have expressed interest in features such as TTS (text-to-speech) options in LM Studio, along with suggestions for enhancing the usability of various coding models.
   - Some community members have shared links to useful resources and models that are compatible with LM Studio, promoting collaboration and knowledge sharing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.exolabs.net/day-2/">12 Days of EXO</a>: 12 Days of Truly Open Innovation</li><li><a href="https://markdownpastebin.com/?id=9912b825d602429d87c11a80e8d8f543">MarkdownPastebin</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Sky-T1-32B-Preview-GGUF">bartowski/Sky-T1-32B-Preview-GGUF · Hugging Face</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Voodoo2">Voodoo2 - Wikipedia</a>: no description found</li><li><a href="https://youtu.be/P8qlE5XBopw?si=Ew359LbJzBS90nsd"> - YouTube</a>: no description found</li><li><a href="https://gist.github.com/shermanhuman/2b9a82df1bab242a8edffe504bb1867c">Coding local LLM recommendations that meet some minimum useful standard</a>: Coding local LLM recommendations that meet some minimum useful standard - MiniumStandardsLLM.md</li><li><a href="https://youtu.be/XYBI_Ow7F_4?t=817">Please Don&#39;t Download HackerGPT...</a>: Hello guys and gals, it&#39;s me Mutahar again! This time we take a look at a pretty wild set of ads that have been popping up all over social media in regards t...</li><li><a href="https://youtu.be/JWfNLF_g_V0?si=avXvc4VzdJ2LZbdM">Turn ANY Website into LLM Knowledge in SECONDS</a>: One of the biggest challenges we face with LLMs is their knowledge is too general and limited for anything new. That’s why RAG is such a huge topic when it c...</li><li><a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: Discover, download, and run local LLMs</li><li><a href="https://www.humblebundle.com/books/machine-learning-engineer-masterclass-packt-books">Humble Tech Book Bundle: Machine Learning Engineer Masterclass by Packt</a>: Learn the basics and advanced techniques of machine learning with this amazing bundle of coding and programming courses. Pay what you want & support charity! </li><li><a href="https://youtu.be/VkzO2w6EqK4?si=dONXQA4qc6VCdUvk">The 3Dfx Voodoo Difference: This is why we love them</a>: In this video we will learn why the 3DFX Voodoo is such a special graphics card! 💙 Consider supporting me 💙Patreon: Get exclusive early access, behind the ...</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/300">0.3.6 Developer Log Truncation regardless of settings · Issue #300 · lmstudio-ai/lmstudio-bug-tracker</a>: Upon updating to 0.3.6 Build 8 from 0.3.5, I&#39;m unable to obtain full, un-truncated dev. logs. This is regardless of my settings used, such as Verbose Logging, and Log Prompts and Responses. I had ...</li><li><a href="https://lmstudio.ai/model/phi-3.1-mini-128k-instruct">Phi 3.1 Mini 128k</a>: phi3 • Microsoft • 3.8B</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/294#issuecomment-2581576638">Not able to install LM Runtime after upgrading to version LM Studio 0.3.6 (Build8) under MacOS 15.2 (M4 Apple Silicon) · Issue #294 · lmstudio-ai/lmstudio-bug-tracker</a>: After upgrading to version 0.3.6 (from 0.3.5) it is not possible to install any LM Runtime. I can download Metal llama.cpp v1.7.1 as well as LM Studio MLX v0.1.3, but they are not installed and the...</li><li><a href="https://github.com/microsoft/ML-For-Beginners/tree/main">GitHub - microsoft/ML-For-Beginners: 12 weeks, 26 lessons, 52 quizzes, classic Machine Learning for all</a>: 12 weeks, 26 lessons, 52 quizzes, classic Machine Learning for all - microsoft/ML-For-Beginners</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.6">LM Studio 0.3.6</a>: Tool Calling API in beta, new installer / updater system, and support for `Qwen2VL` and `QVQ` (both GGUF and MLX)</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues">Issues · lmstudio-ai/lmstudio-bug-tracker</a>: Bug tracking for the LM Studio desktop application - Issues · lmstudio-ai/lmstudio-bug-tracker</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/285">(Exit code 133) Error when loading large LLM models · Issue #285 · lmstudio-ai/lmstudio-bug-tracker</a>: When loading large LLMs (for example, Meta-Llama-3.1-70B-Instruct-IQ2_S with context window 32768), I would encounter the error (Exit code: 133). Please check settings and try loading the model aga...</li><li><a href="https://lmstudio.ai/docs/basics/chat">Manage chats - Running LLMs Locally | LM Studio Docs</a>: Manage conversation threads with LLMs</li><li><a href="https://lmstudio.ai/docs/configuration/presets">Config Presets - Configuration | LM Studio Docs</a>: Save your system prompts and other parameters as Presets for easy reuse across chats.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1327462261576695918)** (186 messages🔥🔥): 

> `PowerMac G3 Build, Llama Model Loading Issues, RTX Graphics Cards, NVIDIA DIGITS Launch, Dual GPU Setup Considerations` 


- **PowerMac G3 Repurposed for AI Models**: A member showcased their custom-built **PowerMac G3** used for running models on LM Studio, highlighting its unique design.
   - Discussion ensued on technical specifications and comparisons with modern hardware.
- **Challenges Loading Llama Models**: Several members discussed difficulties encountered when attempting to load the **Llama 3.3 70b Instruct Q3_K_L model** on a MacBook Pro, citing insufficient system resources despite having adequate RAM.
   - Suggestions included adjusting GPU memory allocation and evaluating system settings.
- **Insights on RTX Graphics Cards for AI Workload**: A conversation emerged regarding the **RTX 4090 vs A6000** graphics cards, focusing on their performance and value for large AI models.
   - Users expressed opinions on price-to-performance ratios while discussing potential future upgrades.
- **Anticipation for NVIDIA DIGITS**: The NVIDIA **DIGITS** launch was debated, with users noting its potential for AI tasks versus existing options like Apple machines.
   - Opinions were mixed on whether to adopt early or wait for further developments.
- **Selecting Dual GPU Setup Options**: A member sought recommendations for purchasing a **second 4060** for dual-card setup, comparing options from PNY and MSI.
   - They also discussed considerations regarding PSU limitations and the potential upcoming **5060** variant.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vladmandic.github.io/sd-extension-system-info/pages/benchmark.html">SD WebUI Benchmark Data</a>: no description found</li><li><a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: Reserve yours today.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1327387005318987776)** (390 messages🔥🔥): 

> `Claude Model Discussions, Hyperparameter Search as a Service, Twitter Experience and Features, Models and Quantization, GitHub Downtime Concerns` 


- **Claude Model Reactions**: The community discussed the latest Claude model updates, noting a shift in its language patterns, such as increased use of words like 'direct' and 'helpful'. Some expressed humor over a fake announcement regarding a new Claude model, reflecting mixed sentiments around its performance and style.
   - The model's responses have been described as more assertive, causing some users to perceive it as 'angry'.
- **Interest in Hyperparameter Optimization Services**: A user inquired about services that provide hyperparameter search as a service, hinting at the complexities involved in tuning models effectively. There is a growing interest in automated solutions for optimizing model parameters with techniques like Bayesian optimization.
   - Discussion included the challenges of debugging issues during model training that could potentially be mitigated by unit testing.
- **Challenges with Twitter's User Experience**: Users expressed frustrations with Twitter's current timeline experience, stating that recent changes have led to a less personalized feed. The removal of features like 'simclusters' was noted as detrimental to user experience, especially for those who appreciated themed content.
   - Concerns were raised about handling spam and irrelevant content amid the ongoing changes on the platform.
- **Model Availability and Quantization**: Participants discussed the availability of large models, specifically mentioning SambaNova and Together AI as key providers of the 405 billion parameter model. The importance of quantization methods and the deployment of models on edge devices were highlighted as significant topics.
   - There’s a call for more models like Helium-1, designed for lightweight applications, showcasing an interest in practical A.I. deployments.
- **Concerns Over GitHub Downtime**: A user reported issues with GitHub being down, impacting their ability to push updates to their projects. This downtime was seen as a sign for some to take a break from their coding tasks, leading to a discussion about the implications of such interruptions.
   - Others humorously acknowledged the situation, suggesting it was a good opportunity for a moment of reflection and untested improvisation in their development processes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://novasky-ai.github.io/posts/sky-t1/">Sky-T1: Train your own O1 preview model within $450</a>: We introduce Sky-T1-32B-Preview, our reasoning model that performs on par with o1-preview on popular reasoning and coding benchmarks.</li><li><a href="https://githubnext.com/projects/copilot-workspace">GitHub Next | Copilot Workspace</a>: GitHub Next Project: A Copilot-native dev environment, designed for everyday tasks.</li><li><a href="https://x.com/jacquesthibs/status/1878851967981887736?s=46">Tweet from Jacques (@JacquesThibs)</a>: New Claude model just dropped</li><li><a href="https://kyutai.org/2025/01/13/helium.html">Announcing Helium-1 Preview</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF">Qwen/Qwen2.5-72B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF">bartowski/Qwen2.5-72B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset">HumanLLMs/Human-Like-DPO-Dataset · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/not-lain/tensor-dims">Mastering Tensor Dimensions in Transformers</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1327618109749727283)** (15 messages🔥): 

> `LLM for medical advice, Privacy concerns with AI, Audiobook text extraction` 


- **Inquiring About LLM for Medical Advice**: One user asked for recommendations on a good LLM for medical advice, but another cautioned against using any AI for reliable medical information, recommending a human doctor instead.
   - Despite the inquiry, the suggestion was clear: AI can provide insights but should always be double-checked by a real doctor to avoid potential misinformation.
- **Privacy Issues of Using AI for Medical Queries**: Concerns were raised about privacy when using AI for medical advice, with emphasis on the risk of personal information being accessed by moderators.
   - A member pointed out that users must trust the companies behind these AI tools, as their privacy practices can vary significantly.
- **Seeking Reliable PDF Text Extraction for Audiobooks**: A member inquired about reliable tools for extracting text from PDFs for audiobook creation, specifically asking about removing headers and footnotes.
   - The suggestion of using **Gemini Flash** was mentioned as a cost-effective option for this purpose.
- **Users Engaged in Progress Tracking**: A member noted lack of progress in a certain context, spurring inquiries about the tool or site being used for aggregation.
   - This indicates a potential interest in the methods of tracking or collaborating within the community.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/prajdabre1/status/1877720543933370418?s=46
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1327721839283142716)** (59 messages🔥🔥): 

> `Qwen 0.5B Model Performance, Generative Knowledge Distillation (GKD), Synthetic Data Usage in AI, MobileLLM Research Insights, Improvements in Attention Mechanisms` 


- **Qwen 0.5B shows mixed performance**: The Qwen 0.5B model demonstrates proficiency in mathematical tasks but struggles with coherent responses in general contexts, often generating nonsensical content.
   - Users expressed concerns about its capability, noting that it frequently fails with math questions and can enter infinite loops during computations.
- **Confusion around GKD in model training**: There is confusion amongst users regarding the Generative Knowledge Distillation (GKD) term used in the model card, as they are unsure how it contrasts with traditional distillation techniques.
   - Some speculate that GKD might refer to training synthetic data from another model rather than distilling logits from the original.
- **Synthetic Data discussed by Hugging Face**: A talk by Loubna Ben Allal emphasized the importance of synthetic data in training Smol Language Models, illustrated through the SmolLM model's design.
   - YouTube resources and discussions referenced highlight the significance of understanding how synthetic data contributes to model performance.
- **MobileLLM paper reveals insights**: The MobileLLM paper indicates that distillation methods were found to be less effective than label-based training, raising questions about current practices in model training.
   - This reference underlines the ongoing debate regarding the effective methodologies for training smaller models in AI.
- **New approaches to attention mechanisms**: Recent research explores advancements in attention mechanisms that aim to retain performance while lowering complexity during training and inference.
   - A proposed novel element-wise attention mechanism suggests an alternative approach to computing similarity, potentially leading to efficiency gains.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.14905">MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases</a>: This paper addresses the growing need for efficient large language models (LLMs) on mobile devices, driven by increasing cloud costs and latency concerns. We focus on designing top-quality LLMs with f...</li><li><a href="https://arxiv.org/abs/2501.05730">Element-wise Attention Is All You Need</a>: The self-attention (SA) mechanism has demonstrated superior performance across various domains, yet it suffers from substantial complexity during both training and inference. The next-generation archi...</li><li><a href="https://x.com/novaskyai/status/1877793041957933347?s=46">Tweet from NovaSky (@NovaSkyAI)</a>: 1/6 🚀 Introducing Sky-T1-32B-Preview, our fully open-source reasoning model that matches o1-preview on popular reasoning and coding benchmarks — trained under $450! 📊Blog: https://novasky-ai.github....</li><li><a href="https://huggingface.co/kz919/QwQ-0.5B-Distilled">kz919/QwQ-0.5B-Distilled · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/kz919/QwQ-0.5B-Distilled-SFT">kz919/QwQ-0.5B-Distilled-SFT · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=AjmdDy7Rzx0">Best of 2024: Synthetic Data / Smol Models, Loubna Ben Allal, HuggingFace [LS Live! @ NeurIPS 2024]</a>: https://latent.space/2024-syndata-smolmodelsLoubna Ben Allal, who works on synthetic data and Smol Language Models at Huggingface, dropped by to drop knowled...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

teknium: https://x.com/prajdabre1/status/1877720543933370418?s=46
  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/)** (1 messages): 

katetra: https://x.com/stackblitz/status/1878818461905739994
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1327719064189276262)** (27 messages🔥): 

> `Stripe Integration, Prompting Techniques, Building AI Apps, User Experiences with Bolt, Webinar Announcement` 


- **Stripe Integration Coming Soon**: A member shared that **Stripe integration** is on the way, with existing users already having success using it, and recommended searching for relevant **YouTube tutorials**.
   - Another user expressed excitement about the integration, describing it as a **HUGE plus** for their program.
- **Mastering Prompting to Avoid Code Loss**: Several users reported frustrations about the system removing existing components every time they added new functionality, with one member humorously saying, *'I keep pushing my products forward past a certain point.'*
   - They discussed solutions like enabling **diffs** in settings to help retain existing code while adding new features.
- **Webinar on AI LLM Apps**: A member announced a **free live training webinar** on building AI applications with structured outputs using Bolt, scheduled for **Tuesday at 10 AM EST**.
   - Participants can learn valuable steps to create dynamic apps, emphasizing the potential of using AI coding platforms.
- **User Frustrations with Workflow Efficiency**: Users expressed their dissatisfaction with the repetitive cycle of code loss, with one stating, *'I'm here to see if someone else has been able to get past this point.'*
   - There was a consensus on the need for methods to prevent unwanted changes, particularly when striving for efficient workflow.
- **Navigating Support and Reporting Issues**: Members provided tips on how to report issues directly in the editor next to the undo button, highlighting the **feedback mechanism** for users.
   - They encouraged each other to share their experiences and thoughts on whether recent YouTube resources accurately reflect the potential solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reinventing.ai/next-level-ai-apps-no-code">How to Build Next-Level AI Apps with No Code</a>: no description found</li><li><a href="https://tangerine-kleicha-76b338.netlify.app/">Vite + React + TS</a>: no description found</li><li><a href="https://docs.google.com/document/d/1SwlpZH1SotqPg2KbZqzWPdpBbs6aKIqMDspSCBCD1iQ/edit?usp=sharing">The Ultimate Guide to Prompting with Bolt</a>: The Ultimate Guide to Prompting with Bolt.new  I am sharing this guide with everyone interested as I think it&#39;s helpful. I asked Bolt AI agent itself how I should talk with “it” and the below are ...
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1327367991104507914)** (386 messages🔥🔥): 

> `Token Management on Bolt, Integrating with Supabase and Netlify, Usage Issues with AI Prompts, CORS Handling in API Requests, Using Stripe with Bolt` 


- **Frustrations with Token Consumption**: Many users reported excessive token consumption while trying to implement features, with one user mentioning a single prompt costing **1.5 million tokens** for a simple overlay.
   - Others shared similar experiences, emphasizing the need for more efficient prompt management and specific requests to reduce token waste.
- **Challenges with API Integrations**: Users expressed difficulties in integrating various services, with issues relating to CORS and erroneous inclusion of code across files during fixes.
   - One user noted errors with Stripe integration, while another mentioned relying more on PayPal buttons due to complications with Stripe.
- **Exporting and Reusing Code**: Users discussed exporting projects as zip files and finding ways to upload them back into Bolt via StackBlitz for continued development.
   - The process highlighted the importance of effective use of StackBlitz in managing larger codebases developed in Bolt.
- **User Feedback on Bolt's Functionality**: Feedback was shared on various functionalities of Bolt, including requests for features like better handling of environmental variables and API keys.
   - Discussions included the possibility of separating Bolt discussions into their own categories due to its popularity and specific issues encountered.
- **Concerns Over Subscription and Token Pricing**: Users raised questions about token pricing, with some feeling that the cost to reload tokens was higher compared to subscription plans.
   - Overall, there was interest in any potential promo codes or alternative options to manage costs associated with using Bolt.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://boltsync.mystify.tech/">BoltSync - GitHub Repository Management with Bolt</a>: Modify your GitHub repositories with Bolt Prompts &amp; sync changes back to GitHub with BoltSync. Streamline your development workflow with AI-powered repository management.</li><li><a href="https://support.bolt.new/Prompting-Effectively-How-to-talk-to-Bolt-13fd971055d6801b9af4e965b9ed26e2">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team</li><li><a href="https://stayyoung.app">Stay Young - Your Daily Dose of Wellness</a>: no description found</li><li><a href="https://coolify.io/)">Coolify</a>: An open-source & self-hostable Heroku / Netlify / Vercel alternative.</li><li><a href="https://docs.google.com/document/d/1SwlpZH1SotqPg2KbZqzWPdpBbs6aKIqMDspSCBCD1iQ/edit?usp=sharing">The Ultimate Guide to Prompting with Bolt</a>: The Ultimate Guide to Prompting with Bolt.new  I am sharing this guide with everyone interested as I think it&#39;s helpful. I asked Bolt AI agent itself how I should talk with “it” and the below are ...</li><li><a href="https://youtu.be/ayagXgAShSk">Bolt.new - Fixing Common Errors While Saving Tokens</a>: In this video, I share a simple method to resolve a common coding error without using any tokens. I demonstrate how to take a screenshot of the error and upl...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1327372414580948993)** (264 messages🔥🔥): 

> `AI productivity in the UK, Embedded AI agents for customer service, Comparison of AI models, New AI model releases, The future of coding with AI` 


- **UK Government's AI Investment**: The UK government plans to invest £14 billion into AI technology with the aim of doubling productivity within three years, sparking debate over the effectiveness of such initiatives.
   - Critics argue that this funds could be better allocated and suggest that AI should not replace human productivity.
- **Building Custom AI Agents for Clients**: A user seeks recommendations for creating embedded AI customer service agents that can integrate with popular APIs like WhatsApp and Slack.
   - Others suggested checking tutorials on n8n or flowise for integration and ease of use, while cautioning about various providers and costs.
- **Comparative Performance of AI Models**: In a challenge involving Minecraft, Claude and Gemini reportedly outperformed ChatGPT in various tasks, leading to discussions about the capabilities of different AI models.
   - Users expressed concerns regarding the performance gap, especially if GPT continues to lag behind in competitive scenarios.
- **New AI Model Release**: A new model called 'codestral' has been released on the Mistral API, offering a 256k context capacity and promising performance.
   - Questions remain about the differences between this model and existing ones like GPT-4 after the canvas feature integration.
- **AI's Impact on Coding**: A user reflects on the potential of AI to transform coding and programming roles, suggesting that as AI evolves, traditional coding could diminish.
   - The conversation points to the growing integration of AI in software development, which could streamline workflows and reduce the necessity for manual coding.



**Link mentioned**: <a href="https://status.openai.com/">OpenAI Status</a>: no description found

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1327375437956907048)** (25 messages🔥): 

> `Canvas Issues, Code Output Concerns, Location Usage, Team Account Problems, Custom GPT Functionality` 


- **Canvas frequently fails to open**: Members discussed ongoing struggles with getting Canvas to open, often receiving code blocks instead. One suggested that specific prompts led to Canvas being used, indicating potential workarounds needed.
   - Another user mentioned that an effective prompt for Canvas was requesting home plans with specific bedroom and bathroom counts.
- **Users frustrated with incomplete code responses**: Multiple members expressed frustration over being provided code comments instead of full code. One user insisted that the model often fails to deliver the complete code despite multiple requests for it.
   - Another user highlighted a bug report to address this issue and encouraged others to upvote it for visibility.
- **ChatGPT providing user location unexpectedly**: A user revealed that ChatGPT provided their location when recommending a YouTube video, leading to confusion and concern. They clarified that the location wasn't directly shared, hinting at potential data usage through IP addresses.
   - A discussion arose regarding whether this information was stored in ChatGPT's memory from prior chats.
- **Problems with team account projects visibility**: A user reported issues with their team account where they could not see any projects despite their teammate being logged in. They mentioned contacting OpenAI but receiving no clear solution or assistance.
   - This raised questions about project visibility and potential technical issues within the workspace.
- **Questions on Custom GPT capabilities**: Inquiries were made about whether custom GPTs utilize memory and custom instructions, with one user asserting they do not. This sparked a conversation about how much context these custom models truly have during their interactions.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1327366579884916877)** (48 messages🔥): 

> `GPT table interpretation issues, OCR vs AI for table reading, Improving table accuracy, Lateral thinking for data format, Reliability of AI models` 


- **GPT struggles with wide tables**: Users reported consistent problems with GPT misinterpreting tables, specifically in wide formats where it struggles with alignment and row shifts.
   - This issue was frequently experienced across various tables, prompting calls for more reliable solutions.
- **Debate on OCR effectiveness**: A user mentioned they primarily use Amazon Textract for table recognition, which usually aligns text correctly, but it faces challenges with complex structures.
   - Another participant countered that traditional OCR still outperforms AI models in table reading accuracy, which remains a concern.
- **Inconsistencies in table interpretation**: Participants discussed how different tables could be broken in unique ways, leading to unreliable AI performance despite occasional fixes.
   - Vitojanko emphasized the importance of AI's advertised capability in reading tables, highlighting its current unreliability.
- **Call for innovative solutions**: Users expressed openness to finding 'trickery' or innovative approaches to enhance the performance of AI in interpreting tables.
   - This included suggestions to possibly request data in a format more compatible with AI capabilities.
- **Acknowledgment of limitations**: There was a consensus acknowledging that while AI offers advancements in table processing, its performance rarely exceeds 60% accuracy.
   - Users noted the inherent challenges of relying on beta software while highlighting the need for improved reliability.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1327366579884916877)** (48 messages🔥): 

> `GPT interpreting tables, Real OCR performance, Complex table structures, Improving AI accuracy, Lateral thinking for data formats` 


- **GPT struggles with wide table formats**: A user reported issues with GPT misaligning data in 'wide' table formats, leading to misinterpretations of prices and other details.
   - This problem has been consistent across various tables and prompts, raising concerns about the model's reliability when handling complex layouts.
- **Real OCR proves more reliable than AI**: Another user suggested using real Optical Character Recognition (OCR) tools, emphasizing that AI models have low accuracy with vision tasks, typically around **60%**.
   - While tools like Amazon Textract can often get text right, they can struggle with complex tables, prompting some users to consider alternatives.
- **Unreliable AI for table reading**: There was a discussion on the unreliability of AI in reading tables despite it being marketed as a use-case, causing frustrations among users.
   - Some argue that while AI may sometimes fix table issues, its inconsistency remains a significant concern for users.
- **Seeking tricks for better data parsing**: A user expressed openness to finding ways to improve the performance of table reading, hinting at potential 'trickery' to enhance outcomes.
   - The community is encouraged to explore creative solutions as they continue to face challenges with table data.
- **Adoption of better data formats**: One suggestion put forth was to ask data providers for better formats, which could help alleviate parsing issues.
   - This lateral thinking approach reflects an effort to actively enhance data usability and reduce reliance on AI’s current limitations.


  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1327438062023606354)** (2 messages): 

> `NotebookLM Mobile Experience Study, Feedback on Audio Overviews, Participant Incentives, User Experience Research` 


- **NotebookLM Welcomes Mobile Experience Input**: The team invites participants for a remote interview regarding the upcoming **NotebookLM mobile experience** scheduled for **January 14th to 15th**. Interested individuals can sign up via this [screener form](https://forms.gle/75gYapqbgCmxXiJL6) for a chance to share their perspectives.
   - Participants will receive the equivalent of **$50** for their time or can opt for a Google merchandise voucher as a thank you gift.
- **Seeking Feedback on Audio Overviews**: A quick **~5 minute screener** has been launched to gather feedback on **Audio Overviews** for NotebookLM. Eligible participants who complete the subsequent survey will receive a **$10** gift code via email as appreciation.
   - Prospective participants must be at least **18 years old**, and can express interest through the provided [form](https://forms.gle/NBzjgKfGC24QraWMA), noting that gift codes are awarded only for completing the survey.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forms.gle/NBzjgKfGC24QraWMA">Register your interest: Google feedback survey</a>: Hello,We are looking for feedback on NotebookLM via a short survey. This will help the Google team better understand your needs in order to incorporate them into future product enhancements. To regist...</li><li><a href="https://forms.gle/75gYapqbgCmxXiJL6">We want to know what you think!</a>: Thank you for your interest in speaking with us! We&#39;ve heard a lot of interest in a mobile experience for NotebookLM, and would love to hear what you think.We’re currently scheduling participants ...
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1327368658057564182)** (46 messages🔥): 

> `Notebook LM capabilities, Podcast sharing platforms, D&D resources with AI, Audio overview feedback, AI in education` 


- **Notebook LM facilitates in-depth research summaries**: A user tested Notebook LM by requesting an audio overview of their original research paper, resulting in over **28 minutes** of summary capturing key aspects effectively.
   - Despite being biased as the author, they sought feedback from others about the clarity of the overview.
- **Ease of Podcast Sharing with Akas**: Users discussed the limitations of sharing AI-generated podcasts through Notebook LM, particularly regarding required login permissions for access.
   - A user introduced [Akas](https://akashq.com), a platform that allows easy uploading and sharing of AI-generated podcasts without login constraints.
- **Innovative AI Use in Education**: A conversation highlighted the use of AI to summarize lectures and enhance learning, especially noted by a nursing student who turns lecture notes into podcasts.
   - Members shared personal experiences and recommendations on using AI tools for creating engaging educational content.
- **Multilingual AI Discussions**: A user proposed instructions for creating a multi-language panel discussion led by a witty host, incorporating various characters for entertainment.
   - The idea suggests exploring diverse dialogues and discussions in a humorous and engaging format.
- **Engaging Readers through Audio**: An author demonstrated the potential of using AI audio summaries to provide readers with a preview of their novel, enhancing engagement.
   - They likened it to a 'test drive' for books, showcasing how AI can bring stories to life through original dialogue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://akashq.com">Akas: home to AI podcasts</a>: no description found</li><li><a href="https://notebooklm.google.com/notebook/6dd1946b-561b-446c-818a-e9e17e332aac/audio">no title found</a>: no description found</li><li><a href="https://notebooklm.google.com/notebook/a6acd033-3af7-41e6-a258-ed7ac973f184/audio">no title found</a>: no description found</li><li><a href="https://notebooklm.google.com/notebook/2119508c-a81e-4a23-8f61-2f7363af4ea3/audio">no title found</a>: no description found</li><li><a href="https://youtu.be/lqaqLZ9ha2I">AI Sonnets: an Experiment with Gemini, VideoFX and Notebook LM</a>: This is an experiment where I asked Gemini to create some sonnets on a theme of my choice, refining them by hand, and with alternating advice from Perplexity...</li><li><a href="https://youtu.be/-C6k5IGBDbY?si=V24b-gkszFj5xZcV),">【AI Podcast】I let AI introduce myself ! Experimental Live2D Podcast</a>: This experimental video is the result of combining artificial intelligence, Live2D animation, and coding to explore how AI summarizes a person—me in this cas...</li><li><a href="https://www.akashq.com/post/b966107e-ef54-41a0-a664-21dc27f841e6">What happened on Jan 10?</a>: What happened on Jan 10? by This Day in History</li><li><a href="https://www.akashq.com/post/2e2231bf-907b-4805-84ae-71f8a7a45c19">What happened on Jan 13?</a>: What happened on Jan 13? by This Day in History</li><li><a href="https://www.akashq.com/post/db284bc7-a4bb-4144-a4d4-05d496f71dd0">What happened on Jan 12?</a>: What happened on Jan 12? by This Day in History
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1327397405590814822)** (289 messages🔥🔥): 

> `NotebookLM Features and Limitations, Using NotebookLM for Research, Podcast Customization, Embedding NotebookLM, User Onboarding and Support` 


- **Understanding Features and Limits of NotebookLM**: Users discussed the limitations of NotebookLM, including issues with adding multiple sources, understanding citation links in outputs, and the absence of features like revised table of contents.
   - Many expressed confusion regarding how many sources could be processed and accessed within notebooks, leading to frustration when outputs weren't as expected.
- **Using NotebookLM Effectively for Research**: Users shared strategies for utilizing NotebookLM in their studies, including how to create summaries, audio overviews, and manage sources effectively, while others highlighted the importance of properly naming and formatting documents for better model access.
   - Some users also suggested creating prompts to enhance the clarity and length of podcast outputs, although success with customization was inconsistent.
- **Podcast Functionality and Customization**: There were discussions on ways to customize podcasts generated by NotebookLM, with users trying various prompts to ensure specific hosts participated and to control the length of episodes.
   - Some users noted challenges with audio overview generation and requested tips for effectively managing podcast content.
- **Interest in Embedding NotebookLM**: A user inquired about embedding NotebookLM into websites, specifically for Google Sites integration, indicating a desire to expand its functionality beyond just personal use.
   - This interest highlights the potential for NotebookLM to be adapted for broader collaborative or educational environments.
- **User Experience and Onboarding**: New users expressed their experiences and challenges as they navigated in NotebookLM, with recurring mentions about the clarity of instructions and the effectiveness of customer support.
   - Some community members offered assistance and shared insights on tools, helping each other to better utilize NotebookLM's features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://illuminate.google.com/home">Illuminate | Learn Your Way</a>: Transform research papers into AI-generated audio summaries with Illuminate, your Gen AI tool for understanding complex content faster.</li><li><a href="https://chromewebstore.google.com/detail/notebooklm-web-importer/ijdefdijdmghafocfmmdojfghnpelnfn))">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.</li><li><a href="https://thedrive.ai?">The Drive AI: Revolutionizing File Management &amp; Knowledge Bases</a>: Discover The Drive AI&#x27;s breakthrough in smart file organization. Our platform transforms your files into a dynamic knowledge base with the help of cutting-edge AI. Elevate your business operation...</li><li><a href="https://youtu.be/spj0n-bFKJo?si=IKMq04zZW7KZMHeZ&t=453"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/spj0n-bFKJo?t=659&si=vrheRfu7QcBE4S2K">NotebookLM: 10 Exclusive Tips Not Found on the Web! (2025)</a>: Free and Best AI Tool to level up your research and Content Creation - NotebookLM! From instantly digesting 100s of documents, videos, websites to multi-lang...</li><li><a href="https://youtu.be/pEC3-5oeIQU?si=3DlU22lAWEAycdzM">Drunk AI Discusses The History of Drinking Alcohol | Funny | Podcast</a>: What happens when AI has a few too many... facts about alcohol? In this fun and light-hearted episode, we take a tipsy dive into the fascinating history of d...</li><li><a href="https://form.typeform.com/to/bOA6l2qF)]">Discover Typeform, where forms = fun</a>: Create a beautiful, interactive form in minutes with no code. Get started for free.
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1327372933437587556)** (313 messages🔥🔥): 

> `Pony Models vs Illustrious, Dreambooth and Training Loras, High-Resolution Generation Techniques, Extensions and Tools for Stable Diffusion, Generating Images with AI` 


- **Pony Models show poor quality compared to Illustrious**: Many users reported that while **Pony XL** has high tag cohesion, it is poorly trained, leading to undesirable results. In contrast, **Illustrious** is preferred for its better handling of realistic images and character generation.
   - It was noted that **JuggernautXL** and **RealVision v5** are also solid alternatives for realism.
- **Dreambooth Training has shifted focus**: Users are moving away from **Dreambooth** due to outdated methods and are now using tools like **Koyha_ss** and **OneTrainer** for training models. Some participants found past Dreambooth tutorials to be outdated and ineffective, seeking more current resources.
   - For **character-specific Loras**, it's suggested to use 50 to 150 images for effective training.
- **High-Resolution Generation Techniques**: The technique of using **hires fix** in Stable Diffusion allows users to generate images at 1024x1024 and then upscale them, leading to better quality outputs. Generating directly at higher resolutions often leads to image duplication and incoherence.
   - Many participants recommended starting at lower resolutions and enabling hires fix to achieve pleasing results.
- **Extensions and Tools for Stable Diffusion**: Users discussed various extensions and tools such as **sd-webui-regional-prompter** for better image control in Stable Diffusion. The importance of the installation method, such as git cloning into the correct directory, was highlighted.
   - There were also warnings about potential scams in Discords and third-party support links.
- **Image Generation with AI**: For casual users wanting quick image generation without high quality, **turbo models** like **SDXL-Turbo** are recommended for their speed. Depending on the model and settings, some users report generating images quickly, but proper training and dataset compilation remain crucial for high-quality output.
   - Feedback on various image generation models indicated that the rule of thumb is to choose ones that require fewer steps for faster results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/mmazco/status/1876336631080419593)">Tweet from maz (@mmazco)</a>: .@doji_com figured out a pretty sticky way to beta test their app - get users to upload selfies and try on a bunch of fits. the curation and elevation in the ecom experience through product discovery ...</li><li><a href="https://huggingface.co/stabilityai/sdxl-turbo">stabilityai/sdxl-turbo · Hugging Face</a>: no description found</li><li><a href="https://stability.ai/news/stable-point-aware-3d">Introducing Stable Point Aware 3D: Real-Time Editing and Complete Object Structure Generation  &mdash; Stability AI</a>: Stable Point Aware 3D (SPAR3D) introduces real-time editing and complete structure generation of a 3D object from a single image in less than a second.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/14x6o2c/finally_figured_out_how_to_create_realistic/?rdt=54950">Reddit - Dive into anything</a>: no description found</li><li><a href="https://safebooru.org/index.php?page=post&s=list&tags=mount_fuji">Safebooru  / mount_fuji</a>: no description found</li><li><a href="https://civitai.com/models/139562/realvisxl-v50)">RealVisXL V5.0 - V5.0 Lightning (BakedVAE) | Stable Diffusion XL Checkpoint | Civitai</a>: H A P P Y N E W Y E A R Check my exclusive models on Mage: ParagonXL / NovaXL / NovaXL Lightning / NovaXL V2 / NovaXL Pony / NovaXL Pony Lightning ...</li><li><a href="https://youtu.be/8eHYYFgzNW0">glossy workshop scan</a>: no description found</li><li><a href="https://github.com/hako-mikan/sd-webui-regional-prompter">GitHub - hako-mikan/sd-webui-regional-prompter: set prompt to divided region</a>: set prompt to divided region. Contribute to hako-mikan/sd-webui-regional-prompter development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=FvpWy1x5etM">FLUX Full Fine-Tuning / DreamBooth Training Master Tutorial for Windows, RunPod &amp; Massed Compute</a>: If you want to train FLUX with maximum possible quality, this is the tutorial looking for. In this comprehensive tutorial, you will learn how to install Kohy...</li><li><a href="https://youtu.be/MQz58wPvT3I?t=4887">ALL THINGS PONY! ft. AstraliteHeart // Creator of Pony Diffusion XL V6 // Civitai Guest Creator</a>: In this video Ally interviews the creator of the incredible Pony Diffusion XL V6 model, AstraliteHeart!  Together, they discuss and dive into everything Pony...</li><li><a href="https://github.com/Haoming02/sd-forge-couple">GitHub - Haoming02/sd-forge-couple: An Extension for Forge Webui that implements Attention Couple</a>: An Extension for Forge Webui that implements Attention Couple - Haoming02/sd-forge-couple
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1327379923672891412)** (138 messages🔥🔥): 

> `AI Model Cost and Performance, Ciortation and Training of Models, AI Services and Tools, Generative AI in Retail, New AI Research and Technologies` 


- **AI Models Cost and Performance Chart**: A detailed chart comparing the cost and performance of various AI models has been shared, indicating how models like o1-preview and GPT-4o stack against each other in pricing and Elo scores.
   - The chart helps to visualize the competitive landscape in the AI model market, highlighting trends such as the performance-to-cost ratio.
- **New Features in GitHub Copilot**: Satya Nadella announced the removal of the waitlist for GitHub Copilot Workspace, advertising it as an advanced agentic editor ready for use.
   - This change aims to facilitate users in building with AI agents more readily than before.
- **Investment in Raspberry AI**: Bryan Kim from a16z announced their investment in Raspberry AI, a generative AI design platform tailored for retail product development.
   - The investment reflects a commitment to innovation in transforming retail design through AI-driven tools.
- **Speed Records for Llama 3 Models**: Users shared impressive speed metrics for Llama 3 models, reporting speeds surpassing conventional setups, highlighting the efficiency of the SambaNova cloud technology.
   - The use of custom chips (SN40L) allows multiple models to run simultaneously, significantly improving deployment for AI applications.
- **STORM: LLM-Powered Article Generation**: Aashutosh.dev introduced STORM, an LLM-powered system designed to write Wikipedia-like articles using web search for research.
   - STORM generates comprehensive reports complete with citations, showcasing a practical application of LLMs in content creation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/whitepaper-agents">Agents</a>: Authors: Julia Wiesinger, Patrick Marlow and Vladimir Vuskovic</li><li><a href="https://blog.gumloop.com/gumloops-17m-series-a/">Gumloop&#x27;s $17m Series A</a>: We&#x27;re excited to start the next phase of Gumloop&#x27;s growth with a $17m Series A led by Nexus Venture Partners with participation from First Round Capital, Y Combinator, angels like Max Mullen...</li><li><a href="https://simonwillison.net/2025/Jan/10/ai-predictions/">My AI/LLM predictions for the next 1, 3 and 6 years, for Oxide and Friends</a>: The Oxide and Friends podcast has an annual tradition of asking guests to share their predictions for the next 1, 3 and 6 years. Here’s 2022, 2023 and 2024. This …</li><li><a href="https://x.com/Sebasti54919704/status/1877948459103515020">Tweet from Sebastian Sosa (@Sebasti54919704)</a>: @huggingface What is HuggingFace&#39;s solution/strategy towards structured outputs (constrained decoding)??Been asking everyone, none seems to know. My best lead so far is that it is being offloaded ...</li><li><a href="https://x.com/AymericRoucher/status/1878456854856048746">Tweet from Aymeric (m-ric) (@AymericRoucher)</a>: -&gt; OS-Genesis: why not to generate the GUI agent trajectories from exploration rather than from high-level tasks?(spoilers: it works reaally well🔥)The main bottleneck in building GUI agents it to ...</li><li><a href="https://arxiv.org/abs/2412.19048">Jasper and Stella: distillation of SOTA embedding models</a>: A crucial component of many deep learning applications (such as FAQ and RAG) is dense retrieval, in which embedding models are used to convert raw text to numerical vectors and then get the most simil...</li><li><a href="https://www.anthropic.com/research/building-effective-agents">Building effective agents</a>: A post for developers with advice and workflows for building effective AI agents</li><li><a href="https://www.all-hands.dev/blog/dont-sleep-on-single-agent-systems">All Hands AI</a>: no description found</li><li><a href="https://arxiv.org/abs/2305.15717">The False Promise of Imitating Proprietary LLMs</a>: An emerging method to cheaply improve a weaker language model is to finetune it on outputs from a stronger model, such as a proprietary system like ChatGPT (e.g., Alpaca, Self-Instruct, and others). T...</li><li><a href="https://x.com/altryne/status/1877220144725758414?s=46">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: Ugh guys... Microsoft just made Qwen 7B solve AIME at the level of o1 😵‍💫 They also showed that with their MCTS driver process, there was self-reflection capability like with reasoning models. Will ...</li><li><a href="https://x.com/ilanbigio/status/1878940258349510764?s=46">Tweet from ilan bigio (@ilanbigio)</a>: announcing our brand new function calling guide @openai!we heard your feedback and made some key changes:- 50% shorter & clearer- new best practices (more on this below 👇)- in-doc function generation...</li><li><a href="https://simonwillison.net/2024/Dec/20/building-effective-agents/">Building effective agents</a>: My principal complaint about the term &quot;agents&quot; is that while it has many different potential definitions most of the people who use it seem to assume that everyone else shares …</li><li><a href="https://x.com/dottxtai/status/1877760709246824919">Tweet from .txt (@dottxtai)</a>: You may have heard the term &#34;agent&#34; quite a bit. But what does that actually mean?A few themes of agents:- Autonomy (operate without human interaction)- Perception (receives information throug...</li><li><a href="https://x.com/hwchase17/status/1867683506861838635?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Harrison Chase (@hwchase17)</a>: Talking to more companies getting serious about putting agents in production, a common trend I&#39;m seeing is:&#34;single agent&#34; -&gt; &#34;multi-agent high level (crewai, autogen)&#34; -&gt; &#3...</li><li><a href="https://youtube.com/playlist?list=PLLAfEmC9OS7WgFe4Te5sFa3l9J_K7qDrq&si=kOQtY9_u5qysiOht">Good AI</a>: no description found</li><li><a href="https://x.com/maxbrodeururbas/status/1877778718208446567?s=46">Tweet from Max Brodeur-Urbas (@MaxBrodeurUrbas)</a>: Gumloop will be a 10-person $1b companyWe have 6 spots leftQuoting Gumloop (@gumloop_ai) We&#39;re excited to announce Gumloop&#39;s $17m series A led by @NexusVP with participation from @firstround, ...</li><li><a href="https://huggingface.co/docs/smolagents/conceptual_guides/intro_agents">Introduction to Agents</a>: no description found</li><li><a href="https://x.com/hcompany_ai/status/1877403314091852169">Tweet from H (@hcompany_ai)</a>: Heading to #CES2025  in Vegas? Put Runner H to secure for you the must-see events. Perfect lineup in minutes!  #RunnerH</li><li><a href="https://x.com/swyx/status/1878392101815099728?s=46">Tweet from swyx.io (@swyx)</a>: wait youre fking kidding me how do these profs just casually book @denny_zhou, @lmthang, @hanjundai, 2 strawberry researchoors, and other top llm people</li><li><a href="https://x.com/backus/status/1878484938003034391?s=46">Tweet from John Backus (@backus)</a>: Zuck approved torrenting and training on LibGen for Llama.Founder mode.</li><li><a href="https://x.com/wayne_hamadi/status/1868742755402621103?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Wayne Hamadi 🖇️ (@wayne_hamadi)</a>: http://x.com/i/article/1867032485768597505</li><li><a href="https://yenchenlin.github.io/blog/2025/01/08/video-generation-models-explosion-2024/">Video Generation Models Explosion 2024 - Yen-Chen Lin</a>: no description found</li><li><a href="https://x.com/svpino/status/1878797424590012907">Tweet from Santiago (@svpino)</a>: This is the fastest I&#39;ve seen Llama 3.3 running anywhere!Llama 3.3 70B running at 652 t/s is lightning fast.And if you want Llama 3.1, here are the speeds I was able to get:• Llama 3.1 8B: 1006 t/...</li><li><a href="https://x.com/slow_developer/status/1877798620692422835?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Haider. (@slow_developer)</a>: 🚨 Mark Zuckerberg on the Joe Rogan podcastin 2025, AI systems at Meta and other companies will be capable of writing code like mid-level engineers.at first, it&#39;s costly, but the systems will beco...</li><li><a href="https://x.com/kalinowski007/status/1877809579154948223">Tweet from Caitlin Kalinowski 🇺🇸 (@kalinowski007)</a>: Really excited to be posting our FIRST Robotics hardware roles for @OpenAI, including two very senior tech lead engineering (IC) roles and a TPM Manager.The first role is for an **EE Sensing Engineer*...</li><li><a href="https://www.reforge.com/blog/ai-native-product-teams">Reforge</a>: no description found</li><li><a href="https://x.com/nrehiew_/status/1877956822318862768?s=46">Tweet from wh (@nrehiew_)</a>: &gt; Use QwQ to generate completions &gt; Use GPT 4o mini to format the outputs &gt; Remove samples that get the incorrect answer &gt; Standard SFT on 17k samples&gt; 19 hours on 8xH100 ($450)Big reas...</li><li><a href="https://x.com/satyanadella/status/1878578314115473577?s=46">Tweet from Satya Nadella (@satyanadella)</a>: There is no more waitlist for GitHub Copilot Workspace—the most advanced agentic editor. Start building with agents today.</li><li><a href="https://x.com/BlackHC/status/1878883222911877375">Tweet from Andreas Kirsch 🇺🇦 (@BlackHC)</a>: NeurIPS 2024 PCs being a bunch of clowns 🤡 the state of ML 🙄All you get back a month after raising a concern:</li><li><a href="https://centml.ai/">Home - CentML</a>: Reduce LLM Serving Costs by up to 65% Elevate your AI efficiency to accelerate deployment and inference while optimizing GPU [&hellip;]</li><li><a href="https://x.com/kirbyman01/status/1878844418972885077">Tweet from Bryan Kim (@kirbyman01)</a>: Thrilled to announce we&#39;re leading the Series A for Raspberry AI, an end-to-end generative AI, design platform built specifically for retail product design.Why our team @a16z invested (cc: @zachco...</li><li><a href="https://x.com/teortaxesTex/status/1877958319127597452">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: imo this isn&#39;t going anywhere, it&#39;s alpaca era &#34;OpenAI has no moat&#34; all over again. yes narrow parity with o&#39;s on benchmarks, but as we scale up and  try to generalize to harder pr...</li><li><a href="https://x.com/swyx/status/1838663794320642328">Tweet from swyx.io (@swyx)</a>: updated for sept 2024 https://x.com/Smol_AI/status/1838663719536201790Quoting AI News by Smol AI (@Smol_AI) it&#39;s notable how predictive the Lmsys Elo vs $ pricing curve is, and how the strategy is...</li><li><a href="https://x.com/bryantchou/status/1877790833371697169?s=46">Tweet from brryant (@bryantchou)</a>: I haven&#39;t been this excited about a software product since... I guess Webflow. 😅past 6mo of gumloop I have:— automated competitive intelligence research (Reddit)— automated competitor ad strategi...</li><li><a href="https://youtu.be/c_9bxtyOd1o?si=31RJlBdZ0E_PLfCH">Hyung Won Chung: Shaping the Future of AI from the History of Transformer</a>: Guest lecture by Hyung Won Chung, Research Scientist, OpenAI, in Prof. Naik&#39;s course CIS 7000: Large Language Models (Fall 2024) on October 14, 2024.</li><li><a href="https://youtu.be/yhpjpNXJDco?si=7sfPgTlyCTi3lNLP">Jason Wei: Scaling Paradigms for Large Language Models</a>: Guest lecture by Jason Wei, Member of the Technical Staff, OpenAI, in Prof. Naik&#39;s course CIS 7000: Large Language Models (Fall 2024) on November 20, 2024.Li...</li><li><a href="https://youtu.be/SN4Z95pvg0Y?si=wyrwJ1VeV2BFElLG"> - YouTube</a>: no description found</li><li><a href="https://docs.google.com/spreadsheets/d/1x9bQVlm7YJ33HVb3AGb9qlDNkvTy9CyOFZoah0kr3wo/edit?gid=0#gid=0">LLM elo vs pricing chart</a>: no description found</li><li><a href="https://x.com/bclavie/status/1878349981570187311">Tweet from Benjamin Clavié (@bclavie)</a>: 🧵 Stella Embeddings: What&#39;s the big deal? (a mini explanation thread)If you enjoy RAG Twitter, or compulsively check the MTEB leaderboard, you might&#39;ve come across the &#34;Stella&#34; (and n...</li><li><a href="https://docs.google.com/document/d/10fnHaH5uEAh-xmc79D7jGB7gJAt-7wQKhZBM2cr6xAc/edit">Scaling Paradigms for Large Language Models</a>: The following is a technical article based on the YouTube video transcript: https://youtu.be/yhpjpNXJDco?si=7sfPgTlyCTi3lNLP Scaling Paradigms for Large Language Models Introduction The field of artif...</li><li><a href="https://cs329a.stanford.edu/">Stanford CS329A | Self-Improving AI Agents</a>: no description found</li><li><a href="https://youtu.be/YdqJSjfi4iw?si=WdLU6j-V_LW_H9jZ)">Aakanksha Chowdhery: Multimodal Reasoning and its Applications to Computer Use and Robotics</a>: Guest lecture by Aakanksha Chowdhery, Senior Staff Research Scientist, Meta, in Prof. Naik&#39;s course CIS 7000: Large Language Models (Fall 2024) on November 2...</li><li><a href="https://youtu.be/kOdl-ncrYDk?si=wDiEgrbW1iAPaUGK)">Hanjun Dai: Preference Optimization for Large Language Models</a>: Guest lecture by Hanjun Dai, Staff Research Scientist &amp; Research Manager, Google Brain, in Prof. Naik&#39;s course CIS 7000: Large Language Models (Fall 2024) on...</li><li><a href="https://youtu.be/T1SeqBapMBo?si=VIkFfcGoROxH7JMu)">LTI Special Seminar by Yi Wu</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1327379067447546028)** (25 messages🔥): 

> `New Podcast Episode, O1 Guest Post Discussion, User Experiences with O1 Pro, Article Featured on HN, Dynamic Use of O1` 


- **New Podcast Episode Drops**: A new podcast episode featuring a discussion on the limitations of MMLU knowledge was released, emphasizing our desire for intelligence over memorized trivia. The episode includes insights from William Bryk about the impressive setups at ExaAILabs.
   - *Prepare yourselves for what’s coming* as they detail the impressive technical specifications of their Exacluster, including **144 H200s** and **20TB GPU RAM**.
- **User Experiences with O1 Pro**: Users discussed their experiences with O1 Pro, highlighting its remarkable performance when engaging with structured contexts compared to the standard version. One user shared insights about working with a **20k** token React/TS codebase, asserting that O1 Pro has been notably reliable.
   - Despite enjoying the benefits, users expressed caution, weighing the cost against alternatives like the **$20/month** plan and reminding themselves of budget constraints.
- **Article Featured on HN**: The recent guest post on O1 saw a spike in visibility and was featured on the front page of Hacker News, capturing community interest. This achievement was celebrated enthusiastically in the chat, highlighting the article's impact.
   - Many expressed curiosity about the implications of the findings for using O1 Preview, confirming the relevance of the discussions beyond just the main version.
- **Dynamic Use of O1**: A member emphasized the need for a different approach to using O1, likening it more to a *report generator* rather than a typical chat model. This notion spurred further discussion on how prompting could enhance performance and usability.
   - Quotations from influential figures like *Sam Altman* and *Ben Hylak* were shared to illustrate evolving perspectives on effectively leveraging the technology.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/swyx/status/1877818998060175508">Tweet from swyx.io (@swyx)</a>: It&#39;s legitimately bizarre that we spend billions of params on storing MMLU/GPQA knowledge that 99% of us don&#39;t really need and can simply look up/learn on demand.We wanted intelligence; we got...</li><li><a href="https://www.latent.space/p/o1-skill-issue">o1 isn’t a chat model (and that’s the point)</a>: How Ben Hylak turned from ol pro skeptic to fan by overcoming his skill issue.</li><li><a href="https://x.com/benhylak/status/1878237490194366744?s=46">Tweet from ben (@benhylak)</a>: o1 is mind-blowing when you know how to use it. it&#39;s really not a chat model -- you have to think of it more like a &#34;report generator&#34;(link to article below)Quoting Sam Altman (@sama) it&#...</li><li><a href="https://x.com/gdb/status/1878489681702310392">Tweet from Greg Brockman (@gdb)</a>: o1 is a different kind of model. great performance requires using it in a new way relative to standard chat models.Quoting Dan Mac (@daniel_mac8) this is an amazing way to think about prompting o1from...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1327381316387340313)** (116 messages🔥🔥): 

> `Claude Projects, AI Tools in Development, Ruby for Prototyping, Mob Coding, AI Applications in Interior Design` 


- **Claude Projects become personal assistants**: Users are expressing how they have integrated **Claude Projects** into their workflows, getting hooked and reporting significant productivity gains.
   - *'Claude Projects are basically my personal assistant at this point, I'm hooked'* highlights the affection for the tool.
- **Transitioning from Gatsby to Astro**: One member shared their experience moving from **Gatsby** to **Astro**, utilizing **bolt.new** for prototyping and **Cursor** for complex features, cutting development time by **60%**.
   - They implied that gaining familiarity with AI tools was essential in handling gnarlier elements and best practices.
- **Ruby's role in rapid prototyping**: Participants discussed their mixed feelings about using **Ruby** for LLM-generated code, appreciating it for human prototyping but feeling it lacks in LLM output quality.
   - One member stated they enjoy Ruby, stating, *'I've spent a LOT of hours in ruby land, and mostly enjoy it'*, indicating a balance of pros and cons.
- **Exploring Mob Coding**: Participants showed interest in **mob coding**, considering it a unique approach to collaborative development, with some expressing personal enjoyment exploring the concept.
   - Comments like *'Mob coding sounds cool'* generated further discussions about its effectiveness in team settings.
- **AI applications in interior design**: A user noted the successful use of AI tools in **interior design** to generate matching décor items based on client preferences, showcasing practical applications of AI.
   - Examples included generating items like pillows to match specific colorways, demonstrating AI's versatility in creative fields.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://claude.site/artifacts/0565699b-deab-419d-9634-ae60ece764a5">Claude Artifact</a>: Try out Artifacts created by Claude users</li><li><a href="https://changelog.com/jsparty/338">Undirected hyper arrows with Chris Shank (JS Party #338)</a>: Chris Shank has been on sabbatical since January, so he&#39;s had a lot of time to think deeply about the web platform. On this episode, Jerod &amp; KBall pick Chris&#39; brain to answer questions lik...</li><li><a href="https://github.com/Little-Languages/quiver">GitHub - Little-Languages/quiver: Your quiver of declarative arrows for the web. ⤵</a>: Your quiver of declarative arrows for the web. ⤵. Contribute to Little-Languages/quiver development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1327411769626329190)** (5 messages): 

> `Aider v0.71.0, Chat mode switching, DeepSeek prompts, Pretty output in editing, Release history insights` 


- **Aider v0.71.0 notable features**: Aider v0.71.0 introduces prompts to help **DeepSeek** work better when alternating between `/ask` and `/code` commands.
   - Streaming pretty **LLM responses** is now smoother and faster, improving user interaction significantly.
- **Commands now switch chat modes**: The bare `/ask`, `/code`, and `/architect` commands now allow users to switch the chat mode, making communication more intuitive.
   - As noted, using `/ask` means all subsequent messages are treated as questions, which users find very nice.
- **Enduring pretty output in editing**: Users expressed excitement about the feature that keeps pretty output enabled even when editing files with **triple-backtick fences**.
   - One user described this change as 'massive', emphasizing its practical benefits.



**Link mentioned**: <a href="https://aider.chat/HISTORY.html">Release history</a>: Release notes and stats on aider writing its own code.

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1327395246580961411)** (182 messages🔥🔥): 

> `DeepSeek Model Performance, Model Configuration in Aider, Quantization for Neural Networks, AI Coding Tools Improvement, Polyglot Benchmark Issues` 


- **DeepSeek Model's Reliability Issues**: Users reported that **DeepSeek** has been unreliable and unresponsive, leading to missed deadlines and frustrations.
   - There's a consensus that the API instability needs to be addressed to improve overall user experience.
- **Aider Configuration Challenges**: A user encountered errors when trying to set the `editor-model` in **.aider.conf.yml**, discovering that it should use a dash instead of an underscore.
   - This raised a discussion about whether the configuration file should be included in the repository's gitignore.
- **Quantization for Neural Networks Discussion**: There was a discussion regarding **quantization** for neural networks, with one user mentioning the importance of understanding the concept for effective coding.
   - Users expressed the need for LLMs to have a better grasp of fundamental concepts to prevent issues in coding tasks.
- **Potential Improvements in AI Coding Tools**: The group is interested in the potential for **AI coding tools** to improve, but some believe advancements are dependent on LLM capabilities.
   - Participants debated the effectiveness of using models like **Sonnet** and **O1** for various coding tasks.
- **Polyglot Benchmark Testing**: A user shared observations that some C++ exercises in the **polyglot benchmark** may require specific flags to run all tests properly.
   - Another user expressed interest in comparing the performance of **O1** against **Sonnet** on specific benchmark tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral-2501/">Codestral 25.01</a>: Code at the speed of Tab. Available today in Continue.dev and soon on other leading AI code assistants.</li><li><a href="https://x.com/OpenRouterAI/status/1878876208877953235">Tweet from OpenRouter (@OpenRouterAI)</a>: 23% growth in inference last week 👀@AnthropicAI&#39;s Claude 3.5 Sonnet self-moderated was the biggest source</li><li><a href="https://x.com/hive_echo/status/1878400401164140890">Tweet from echo.hive (@hive_echo)</a>: Get free full o1 API usage 🥳also free o1-mini. gpt-4o 1) make sure you have access to o1 via API2) go to your dashboard&gt;data controls&gt;sharing tab3) see if you have this notification</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://github.com/Aider-AI/aider/blob/main/CONTRIBUTING.md">aider/CONTRIBUTING.md at main · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://github.com/stacklok/codegate">GitHub - stacklok/codegate: CodeGate: CodeGen Privacy and Security</a>: CodeGate: CodeGen Privacy and Security . Contribute to stacklok/codegate development by creating an account on GitHub.</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1hys13h/new_model_from_httpsnovaskyaigithubio/">New Model from https://novasky-ai.github.io/ Sky-T1-32B-Preview, open-source reasoning model that matches o1-preview on popular reasoning and coding benchmarks — trained under $450! </a>: Posted in r/LocalLLaMA by u/appakaradi • 501 points and 120 comments</li><li><a href="https://aide.dev/">Aide - Your AI Programming Assistant</a>: Code with the speed and knowledge of the best programmer you know. Aide is by your side.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1327369508041785415)** (84 messages🔥🔥): 

> `Aider configuration, Prompt caching in Aider, Editing files in Aider, Using models from Hyperbolic, Handling suggestions in Aider` 


- **Configuration Issues with Aider on Mac**: A user reported trouble using the /help command in Aider after installing it through Homebrew on Mac, facing installation issues with tokenizers.
   - They seek guidance on how to set the ask mode as the default chat mode within the .env file.
- **Prompt Caching Challenges**: Discussions on prompt caching suggest that the caching behavior depends on including the exact same set of files for cache hits, leading to frustration when files are added dynamically.
   - Users consider whether to submit issues about caching inefficiencies while discussing potential optimizations in how Aider manages read-only files.
- **Editing and File Management in Aider**: Users shared insights on editing files with Aider, including using the /add command to quickly add directories, often mentioning that some files like 'python' are created erroneously.
   - A suggestion was made to adjust user conventions to avoid excessive suggestions when adding new files.
- **Using Models from Hyperbolic**: A user inquired about utilizing models from Hyperbolic, particularly how to configure Aider to call specific models like DeepSeek-V3 using OpenAI's API structure.
   - The community clarified that users can select LLMs from different providers, and shared specific model names to ensure proper configuration in Aider.
- **Multi-Line Commands in Aider**: A question was raised about performing multi-line asks in Aider, leading to a tip that using Shift + Alt + Enter serves this purpose.
   - This functionality allows for more complex commands that extend beyond simple one-liners, enhancing user interaction with the tool.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting and testing</a>: Automatically fix linting and testing errors.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1327379527369887804)** (4 messages): 

> `browser-use, CodeGate, Deepseek AI Assistant, Always On AI Assistant` 


- **Make websites AI-friendly with browser-use**: The [browser-use project](https://github.com/browser-use/browser-use) aims to make websites accessible for AI agents by enhancing their interaction capabilities.
   - This development is crucial for improving how AI agents process and navigate web content.
- **Focus on CodeGen Privacy with CodeGate**: [CodeGate](https://github.com/stacklok/codegate) provides insights into CodeGen's privacy and security measures, targeting more secure code generation practices.
   - Engaging in this project can help enhance security protocols in AI-driven coding environments.
- **Deepseek AI Assistant always on duty**: The [Deepseek AI Assistant](https://www.youtube.com/watch?v=zoBwIi4ZiTA) presentation highlights a new Python AI agent, Ada, which is designed to function continuously for engineers.
   - This innovative approach is set to revolutionize how engineers deploy and manage code effectively.
- **Pattern for a Real-Time AI Assistant**: [The always-on AI Assistant](https://github.com/disler/always-on-ai-assistant/) utilizes Deepseek-V3, RealtimeSTT, and Typer to create an efficient engineering assistant.
   - This pattern provides a framework for developing assistants that are responsive and continually available for engineering tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=zoBwIi4ZiTA">Deepseek AI Assistant: ALWAYS ON Python AI Agent for Engineers that SHIP</a>: 🔥 Is your Personal AI Assistant truly ALWAYS ON? Discover how Ada, powered by DeepSeek V3, is revolutionizing the way engineers ship code! 🚀🎥 Resources fo...</li><li><a href="https://github.com/stacklok/codegate">GitHub - stacklok/codegate: CodeGate: CodeGen Privacy and Security</a>: CodeGate: CodeGen Privacy and Security . Contribute to stacklok/codegate development by creating an account on GitHub.</li><li><a href="https://github.com/disler/always-on-ai-assistant/">GitHub - disler/always-on-ai-assistant: A pattern for an always on AI Assistant powered by Deepseek-V3, RealtimeSTT, and Typer for engineering</a>: A pattern for an always on AI Assistant powered by Deepseek-V3, RealtimeSTT, and Typer for engineering - disler/always-on-ai-assistant</li><li><a href="https://github.com/browser-use/browser-use">GitHub - browser-use/browser-use: Make websites accessible for AI agents</a>: Make websites accessible for AI agents. Contribute to browser-use/browser-use development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 messages): 

louisgv: Phi 4 is now available: https://openrouter.ai/microsoft/phi-4
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1327736854635745416)** (2 messages): 

> `Friday Agents, Telegram LLM Interface, DeVries AI Chatbot` 


- **Friday Agents Framework Launch**: The GitHub repository for **Friday Agents** introduces a powerful JavaScript framework for building **AI-powered applications** using a multi-agent architecture, available at [GitHub - amirrezasalimi/friday-agents](https://github.com/amirrezasalimi/friday-agents).
   - This framework consists of two main components to streamline the development of AI applications.
- **Unlock 200+ AI Models via Telegram**: The DeVries AI Chatbot allows users to converse with **200+ large language models** directly in Telegram for a low-cost subscription, with a free trial available at [devriesai.com](https://devriesai.com/).
   - For just **$24.99/month**, users gain access to all current and upcoming AI models, simplifying interactions through a familiar platform.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://devriesai.com/">devriesai</a>: Your Telegram AI Agent</li><li><a href="https://github.com/amirrezasalimi/friday-agents">GitHub - amirrezasalimi/friday-agents: Friday Agents. App: https://chat.toolstack.run/</a>: Friday Agents. App: https://chat.toolstack.run/. Contribute to amirrezasalimi/friday-agents development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1327366626181644309)** (212 messages🔥🔥): 

> `OpenRouter usage, Deepseek model performance, Launching of Mistral's Codestral model, Comparison of different LLMs, Provider deployment on OpenRouter` 


- **OpenRouter Offers Flexible LLM Options**: Users express satisfaction with OpenRouter's Deepseek V3 model due to its performance and pricing, while others explore Android apps with features similar to OpenRouter's chat interface.
   - Concerns are raised about the limitations of specific models, particularly regarding handling images and performance inconsistencies.
- **Mistral's Codestral Model Release**: Mistral has announced the launch of their new Codestral model featuring a 262K context and improvements over previous versions, although it is no longer available for general release.
   - The model is noted for its efficient architecture and increased speed for coding tasks, but users express disappointment at the lack of open access.
- **Discussions on LLM Pricing and Cost Comparisons**: Participants discuss the costs associated with various cloud services for LLM implementation, with some expressing an interest in comparing expenses across platforms.
   - Questions arise about ideal providers among users, particularly when considering different models' performance and relevant features.
- **Insights on OpenRouter Model Providers**: Some users inquire about becoming model providers on OpenRouter, seeking guidance on the process to deploy models within the OpenRouter ecosystem.
   - Support is mentioned as a crucial contact point for individuals interested in offering their own models through the platform.
- **Exploring Model Selection and Usage Strategies**: Discussion on the effectiveness of different LLMs indicates a preference for models that generate comprehensive reports over conversational responses.
   - Users share their strategies for selecting models based on specific use cases, noting the importance of context and processing times in their choices.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral-2501/">Codestral 25.01</a>: Code at the speed of Tab. Available today in Continue.dev and soon on other leading AI code assistants.</li><li><a href="https://openrouter.ai/docs/crypto-api">Crypto Payments API | OpenRouter</a>: APIs related to purchasing credits without a UI</li><li><a href="https://openrouter.ai/api/v1",">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/docs/models">Models | OpenRouter</a>: A table of all available models</li><li><a href="https://x.com/OpenRouterAI/status/1878876208877953235">Tweet from OpenRouter (@OpenRouterAI)</a>: 23% growth in inference last week 👀@AnthropicAI&#39;s Claude 3.5 Sonnet self-moderated was the biggest source</li><li><a href="https://github.com/openai/openai-node#undocumented-request-params">GitHub - openai/openai-node: Official JavaScript / TypeScript library for the OpenAI API</a>: Official JavaScript / TypeScript library for the OpenAI API - openai/openai-node</li><li><a href="https://developers.cloudflare.com/ai-gateway/providers/openrouter/">OpenRouter · Cloudflare AI Gateway docs</a>: OpenRouter ↗ is a platform that provides a unified interface for accessing and using large language models (LLMs).</li><li><a href="https://x.com/CloudflareDev/status/1861861672358654107">Tweet from Cloudflare Developers (@CloudflareDev)</a>: We now support @OpenRouterAI on Cloudflare&#39;s AI Gateway. You can now add them as a provider to monitor, log and control your OpenRouter LLM requests. Read more on how to add them here 👇</li><li><a href="https://openrouter.ai/api/v1">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://edition.cnn.com/2025/01/07/tech/meta-hateful-conduct-policy-update-fact-check/index.html">Calling women ‘household objects’ now permitted on Facebook after Meta updated its guidelines | CNN Business</a>: no description found</li><li><a href="https://github.com/AaronWard/generative-ai-workbook/discussions/36">Weights &amp; Biases - Evaluating and testing LLM applications · AaronWard/generative-ai-workbook · Discussion #36</a>: Article W&amp;B Sweeps - used for iterating of configurations and evaluating metrics such as tokens used, cost, response quality results, different templates, additional configurations etc. 1. Underst...</li><li><a href="https://www.404media.co/its-total-chaos-internally-at-meta-right-now-employees-protest-zuckerbergs-anti-lgbtq-changes/">‘It’s Total Chaos Internally at Meta Right Now’: Employees Protest Zuckerberg’s Anti LGBTQ Changes</a>: Meta&#x27;s decision to specifically allow users to call LGBTQ+ people &quot;mentally ill&quot; has sparked widespread backlash at the company.</li><li><a href="https://www.latestly.com/socially/world/mark-zuckerberg-orders-removal-of-tampons-from-mens-bathrooms-at-meta-offices-report-6556071.html#google_vignette">Mark Zuckerberg Orders Removal of Tampons From Men's Bathrooms at Meta Offices: Report | 🌎 LatestLY</a>: It is also reported that business managers were instructed to remove tampons from men&amp;#039;s bathrooms, which Meta provided to non-binary and transgender employees using the men&amp;#039;s bathroo...</li><li><a href="https://www.404media.co/meta-deletes-trans-and-nonbinary-messenger-themes/">Meta Deletes Trans and Nonbinary Messenger Themes</a>: Amid a series of changes that allows users to target LGBTQ+ people, Meta has deleted product features it initially championed.</li><li><a href="https://github.com/OpenRouterTeam/openrouter-runner">GitHub - OpenRouterTeam/openrouter-runner: Inference engine powering open source models on OpenRouter</a>: Inference engine powering open source models on OpenRouter - OpenRouterTeam/openrouter-runner
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1327375132133425263)** (190 messages🔥🔥): 

> `Perplexity Subscription Concerns, Comparison of AI Models, User Experience Issues, Image Generation Features, API Usage and Costs` 


- **Concerns about Perplexity Subscription Costs**: Users expressed concerns regarding the pricing of Perplexity's subscription services, especially noting the $200/month cost for ChatGPT.
   - Suggestions were made for Perplexity to introduce a more competitive pricing structure that would appeal to pro users.
- **Discussions on AI Model Effectiveness**: Members debated whether Perplexity's models are superior to alternatives such as ChatGPT, with opinions split on the efficiency of Claude models compared to newer offerings.
   - Users noted that while Perplexity has advantages, no specific model was unanimously agreed upon as the best.
- **User Experience Issues with Perplexity**: Several users reported performance issues with the Perplexity app, mentioning slow loading times and persistent 'pending' notifications.
   - The problems seemed consistent across devices, leading to frustration among pro users who expect higher performance.
- **Image Generation Features**: Conversations highlighted difficulties users faced in generating images on Perplexity, with suggestions to utilize external tools like Grok for better results.
   - Despite the challenges, many users expressed a desire for improved image generation capabilities within the platform itself.
- **Clarifications on API Costs and Usage**: Users inquired about the costs associated with API usage, with $5 worth of calls included in the pro subscription but additional charges for tokens.
   - There was confusion regarding token definitions, and users discussed the potential financial implications of frequent API use.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://belladoreai.github.io/llama3-tokenizer-js/example-demo/build/">llama-tokenizer-js playground</a>: no description found</li><li><a href="https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui>">no title found</a>: no description found</li><li><a href="https://docs.perplexity.ai/guides/pricing">no title found</a>: no description found</li><li><a href="https://x.com/pplxsports/status/1878550531603312947?s=61">Tweet from Perplexity Sports (@PPLXsports)</a>: All game. No noise.</li><li><a href="https://x.com/cb_doge/status/1877570367239209273?s=46">Tweet from DogeDesigner (@cb_doge)</a>: BREAKING: Grok is now the #3 app in Productivity category on the AppStore in just one day of release. 🇺🇸</li><li><a href="https://newsletter.moneylion.com/subscribe?ref=yJmsSyv2l7">MoneyLion Markets Daily Newsletter</a>: Your Daily Dose of Market News</li><li><a href="https://youtu.be/EXfFBEuCAr0?si=pJqgmK_4RVJiA8LO">you STILL need a website RIGHT NOW!! (yes, even in 2024)</a>: Build your website in 5 seconds: https://hostinger.com/networkchuck10 (use coupon code NETWORKCHUCK for an extra 10% off)🗳️🗳️VOTE!!: Who has the best websi...</li><li><a href="https://www.copilotforyoutube.com/search/joe-rogan-experience-2255-mark-zuckerberg-vAHjVHQqkgI07k7F3G4piE">Joe Rogan Experience #2255 - Mark Zuckerberg</a>: Mark Zuckerberg is the chief executive of Meta Platforms Inc., the company behind Facebook, Instagram, Threads, WhatsApp, Meta Quest, Ray-Ban Meta smart glasses, Orion augmented reality glasses, and o...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1327376696273403977)** (14 messages🔥): 

> `Anthropic valuation, Roman Empire lead poisoning, AI Chips, Bitcoin Recovery, Spotify CEO cashout` 


- **Anthropic Valuation Hits $60 Billion**: The **$60 billion valuation** of **Anthropic** has attracted significant attention, with discussions circulating around its implications in the AI industry.
   - This comes alongside other AI advancements, signaling strong investor interest in emerging **AI companies**.
- **Lead Poisoning's Impact on IQ Revealed**: Recent discussions highlighted how **lead poisoning** during the **Roman Empire** era contributed to declining **IQ rates**.
   - This historical analysis prompted further inquiry into how environmental factors impact cognitive functions today.
- **Bitcoin Recovery Efforts Stall**: Reports surfaced of a **$750M Bitcoin Recovery** effort being halted, leaving many to ponder the effectiveness of current asset recovery strategies.
   - Questions arose about the evolving landscape of blockchain security and recovery processes.
- **Spotify CEO's Huge Cashout**: The recent **massive cashout** by Spotify's CEO has raised eyebrows and sparked conversations about executive compensation.
   - Members analyzed the potential ramifications of such financial maneuvers amid ongoing debates on corporate governance.
- **AI Chips and Technological Advancements**: Several members discussed **AI Chips** and their growing significance within the tech ecosystem, alongside recent breakthroughs like **MIT's Stacked 3D Chips**.
   - The conversation emphasized the competitive edge these innovations bring to data processing capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/iCGhq5Og_Lg">YouTube</a>: no description found</li><li><a href="https://www.youtube.com/embed/ula7jilgJdY">YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1327652246854303755)** (11 messages🔥): 

> `Sonar 3.3 API Availability, Citations in Llama-3.1-Sonar, Future Model Releases, Changelog Confusion, Model Deprecation Notice` 


- **Sonar 3.3 is available but not in API**: Members expressed confusion about **Sonar 3.3** being available on the Perplexity web UI but not as an API model, causing inquiries about its future availability.
   - Another member also shared similar interest, adding to the discussion on whether more models could be released.
- **Citations working in Llama-3.1-Sonar**: A user noted that citations appear in the JSON response for **llama-3.1-sonar** models, yet it doesn't work with **claude-3.5-sonnet**.
   - This raised questions about the functionality across different model versions.
- **Requests for more models**: One member asked if there could be releases beyond the **llama-3.1-sonar-small/large/huge** variations.
   - A response highlighted that improved versions of existing models may be released at irregular intervals, urging users to monitor announcements.
- **Confusion regarding the changelog**: A user was unsure about the changelog contents as they couldn't find updates for **o1** and **Llama 3.3 70b** after November.
   - It was clarified that the changelog only pertains to the API updates, along with an email notification regarding model deprecation received by API users.
- **Assumption on upcoming updates**: Discussion revealed that there has been no formal announcement concerning **llama 3.3.70b**, and while something might be coming, it's merely assumption.
   - For **o1**, it was noted that it was never on the agenda for API release.



**Link mentioned**: <a href="https://perplexity.mintlify.app/changelog/changelog">no title found</a>: no description found

  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1328375033160138774)** (38 messages🔥): 

> `Codestral 25.01, Helium-1 Model Launch, CC-BY License Discussions, Qwen 2.5-Math Models, OpenAI and OSS Contributions` 


- **Codestral 25.01 Takes the Lead**: The newly upgraded **Codestral 25.01** has debuted at **#1** on the LMsys copilot arena leaderboard, showcasing enhanced efficiency and performance.
   - While it scored **11%** on the Aider polyglot benchmark, it was noted that some members express concerns about competing with leading models.
- **Kyutai Introduces Helium-1**: Kyutai announces the preview of their new backbone language model, **Helium-1**, featuring around **2B parameters** targeted for edge and mobile devices.
   - As a multi-lingual model, Helium-1 currently supports **6 languages**, emphasizing the importance of latency and privacy in personal AI systems.
- **Debate on CC-BY License for Models**: There was a healthy debate on the appropriateness of using **CC-BY licenses** for AI models, particularly around how copyright pertains to model weights.
   - Several members expressed that traditional licenses may not effectively cover the unique nature of AI model outputs, calling for the creation of more suitable licenses.
- **Qwen 2.5-Math Models Enhance Reasoning**: The release of **Qwen 2.5-Math-PRM-72B** introduces Process Reward Models to improve mathematical reasoning accuracy in Large Language Models.
   - These models aim to reduce intermediate errors in reasoning processes and showcase impressive performance in various evaluations.
- **OpenAI's OSS Contributions Under Scrutiny**: Members discussed the irony that OpenAI's most significant current contribution to the open-source ecosystem is their library/client offering to modify base URLs.
   - Comments noted the awkward position of this contribution contrasting with the original intent of advancing open-source development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/codestral-2501/">Codestral 25.01</a>: Code at the speed of Tab. Available today in Continue.dev and soon on other leading AI code assistants.</li><li><a href="https://x.com/deedydas/status/1877549539781128319?t=hFlLBI6S6s0xaB2ciDeztw&s=19">Tweet from Deedy (@deedydas)</a>: Pretty crazy that after OpenAI o3 hit 71.7% on SWE-Bench Verified, yesterday, Claude Sonnet 3.5 using CodeStory hit 62.2%.A &#34;last-gen&#34; non-reasoning model getting within 10% of the unreleased ...</li><li><a href="https://x.com/paulgauthier/status/1878886495609815054">Tweet from Paul Gauthier (@paulgauthier)</a>: Codestral 25.01 scored 11% on the aider polyglot benchmark. 62% o1 (high)48% DeepSeek V316% Qwen 2.5 Coder 32B Instruct11% Codestral 25.01 4% gpt-4o-mini https://aider.chat/docs/leaderboards/</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Math-PRM-72B">Qwen/Qwen2.5-Math-PRM-72B · Hugging Face</a>: no description found</li><li><a href="https://kyutai.org/2025/01/13/helium.html">Announcing Helium-1 Preview</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1327436519241420842)** (9 messages🔥): 

> `Qwen Instruct Model Training, LoRa Fine-Tuning, Generative Agents and Environment Navigation` 


- **Struggles with Qwen Instruct Model Training**: Many users reported **degradation** when attempting to train the **Qwen instruct model** on standard benchmark tasks.
   - One member mentioned that similar issues have been observed with **Llama** as well.
- **LoRa Used for Qwen Instruct Fine-Tuning**: A user shared their experience of using **LoRa** for fine-tuning a **Qwen-instruct model** for their specific tasks.
   - They indicated that their dataset could be reasonably out of **distribution** from what the model was originally trained on.
- **Interest in Generalized Agentic Data Studies**: A member inquired about studies on fine-tuning models with **generalized agentic data**, particularly for **ReAct** style agents interacting with external environments.
   - They noted the lack of broader studies on **SFT** and **RL** for these type of agent trajectories.
- **Comparing ReAct and Tool-Using Agents**: The discussion highlighted the distinctions between **ReAct agents** navigating real environments and tool-using agents, which lack an intrinsic sense of navigation.
   - The implications of effective environment navigation methods in AI were considered, emphasizing their potential relevance.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

420gunna: https://x.com/aidan_mclau/status/1878944278782890158
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1327753172013551687)** (38 messages🔥): 

> `Learning about AI, Local AI Models, Meta Ray-Bans, CIOs and AI Talks, VITURE Pro Neckband` 


- **What Everyone's Learning**: Members are sharing what they're currently learning, including **Emma Brunskill RL** and **David Silver RL** audio formats.
   - *“How to open a business bank account”* and *“How to speak with VCs”* were also mentioned as key learning topics.
- **Debate on Local AI Models**: Discussion revolves around local AI models not being ideal for general use, emphasizing they serve better in **privacy** and **internet connectivity** challenges.
   - One member noted local model enthusiasts are building ecosystems that provide value through optimization and task specialization.
- **Concerns over Voice Messaging Flexibility**: Concerns were expressed about issues with voice command systems, finding that commands like 'send message to X' often result in simplified messages.
   - This was linked to the perception that current iterations of the technology feel outdated.
- **CIO Talk Preparation for AI**: A member is preparing for a talk on AI with higher-ed CIOs and is considering covering basic **LLM functionality**, prompting, and use cases.
   - Another suggested addressing how employees inadvertently share sensitive data in chatbots, adding a layer of relevance.
- **Interest in VITURE Pro Neckband**: Discussion emerged around the **VITURE Pro Neckband** and its potential for enhancing productivity during multitasking.
   - Despite styling concerns, one member highlighted that these devices could improve their daily activities significantly.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1878340286423634184">Tweet from Xeophon (@TheXeophon)</a>: You don‘t use local models to save costs, these calculations never work in your favor. You do local models for latency* and/or privacy.Quoting Xeophon (@TheXeophon) @ashrafaddani @tomchapin @anushkmit...</li><li><a href="https://x.com/chesterzelaya/status/1873936772696334570)">Tweet from chester (@chesterzelaya)</a>: i can’t believe this really exists instantly added to my daily workflow</li><li><a href="https://x.com/AutismCapital/status/1878475791379603499?s=19">Tweet from Autism Capital 🧩 (@AutismCapital)</a>: Shirts that we need NOW</li><li><a href="https://www.viture.com/">VITURE: Next Gen XR Glasses</a>: Making the parallel possible...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1327639498753966131)** (58 messages🔥🔥): 

> `Sky-T1-32B-Preview, Reinforcement learning vs. Supervised fine-tuning, Generative AI for talks, Challenges in academic talks, Process Reward Models` 


- **Sky-T1-32B-Preview shows affordable reasoning capabilities**: The [Sky-T1-32B-Preview](https://novasky-ai.github.io/posts/sky-t1/) can perform on par with o1-preview on reasoning benchmarks while being trained for under **$450**.
   - Its open-source code is available on [GitHub](https://github.com/NovaSky-AI/SkyThought) and highlights the potential of effective open-weight models.
- **Debate on RL vs. SFT learning**: Discussants pondered whether self-tuning on reasoning traces could truly replicate RL-trained behaviors, citing it as a philosophical question.
   - Natolambert noted that while behaviors might be induced, the outcomes likely won't maintain the same robustness.
- **AI's role in enhancing presentations**: There is interest in employing AI to generate relevant imagery during talks, though some express skepticism about its efficiency in combatting laziness.
   - Participants agreed that crafting high-quality talks remains a challenging endeavor necessitating substantial effort.
- **Challenges in consuming academic papers**: Readers discuss the difficulties of reading full academic papers in the current information-rich environment, with many opting for selective reading.
   - Natolambert mentioned reading mostly relevant sections of the LLaMA 3 paper, indicating a strategical approach to digesting extensive material.
- **Insights into Process Reward Models**: A paper on Process Reward Models highlights their effectiveness in supervising mathematical reasoning in LLMs but underscores significant challenges in data annotation.
   - The findings stress that conventional data synthesis methods yield inferior performance compared to human evaluation techniques.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>: Process Reward Models (PRMs) emerge as a promising approach for process supervision in mathematical reasoning of Large Language Models (LLMs), which aim to identify and mitigate intermediate errors in...</li><li><a href="https://novasky-ai.github.io/posts/sky-t1/">Sky-T1: Train your own O1 preview model within $450</a>: We introduce Sky-T1-32B-Preview, our reasoning model that performs on par with o1-preview on popular reasoning and coding benchmarks.</li><li><a href="https://x.com/teortaxesTex/status/1877958319127597452">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: imo this isn&#39;t going anywhere, it&#39;s alpaca era &#34;OpenAI has no moat&#34; all over again. yes narrow parity with o&#39;s on benchmarks, but as we scale up and  try to generalize to harder pr...</li><li><a href="https://arxiv.org/abs/2305.15717">The False Promise of Imitating Proprietary LLMs</a>: An emerging method to cheaply improve a weaker language model is to finetune it on outputs from a stronger model, such as a proprietary system like ChatGPT (e.g., Alpaca, Self-Instruct, and others). T...</li><li><a href="https://youtu.be/kOdl-ncrYDk?si=41wy2nlWuv88_XFH"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=kOdl-ncrYDk&list=PLF3-CvSRq2SbrG9pUQZh9WkKE2OjgHXVT">Hanjun Dai: Preference Optimization for Large Language Models</a>: Guest lecture by Hanjun Dai, Staff Research Scientist &amp; Research Manager, Google Brain, in Prof. Naik&#39;s course CIS 7000: Large Language Models (Fall 2024) on...</li><li><a href="https://youtu.be/YR9EztOF0R8?si=-hCAEtMlXhgpRw3p&t=2527">Learning to Reason, Insights from Language Modeling</a>: Noah Goodman, Stanford University</li><li><a href="https://youtu.be/T1SeqBapMBo?si=VIkFfcGoROxH7JMu">LTI Special Seminar by Yi Wu</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1327465367555014717)** (1 messages): 

> `Dr. Huberman's Insights, Mental Health Support, Impact of Discussions` 


- **Listening to Smart People is Soothing**: A member expressed that listening to **smart people talk** helps to soothe their **manic depression**, reflecting a need for positive mental health support.
   - They commented on the calming influence that **intellectual discussions** can provide during tough times, indicating an interest in perspectives from **Dr. Huberman**.
- **Inquiry about Dr. Huberman's Advice**: The member's message prompted a curiosity about **Dr. Huberman's** insights on mental health, specifically related to feeling soothed through conversation.
   - This highlights the potential significance of expert advice in handling emotional challenges.


  

---


### **Interconnects (Nathan Lambert) ▷ #[retort-podcast](https://discord.com/channels/1179127597926469703/1238601424452059197/1327510839686594601)** (6 messages): 

> `Forum Channel Suggestion, Website for Podcast Listings` 


- **Suggestion for a Forum Channel**: A member suggested creating a **forum channel** for posting static topics like the latest articles, podcast episodes, and videos for easier access.
   - *Thank you for sharing your knowledge* was expressed as appreciation for the contributions of others.
- **Website as a Resource**: Another member mentioned that the **website natolambert.com** lists external podcast appearances, suggesting it could serve as a resource.
   - The suggestion to bookmark the website was made, emphasizing *learning is fun* during the discussion about navigating new information.


  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1328353022044016683)** (11 messages🔥): 

> `U.S. AI Economic Blueprint, AI Diffusion Controls, National Security & Economic Strength, Export Controls, AI Leadership` 


- **U.S. AI Economic Blueprint Released**: The [U.S. AI Economic Blueprint](https://openai.com/global-affairs/openais-economic-blueprint/) outlines strategies to enhance U.S. technology leadership and prevent adversary abuse of AI.
   - It emphasizes the need for American technology to support global AI use without compromising national security.
- **Concerns Over AI Diffusion Controls**: A member questioned the effectiveness of AI diffusion controls, noting shipments under **1700 GPUs** are not counted against national caps.
   - They criticized this loophole, highlighting that smuggling often occurs through smaller, shell company orders, suggesting the system is poorly designed.
- **AI's Role in National Security**: The White House's fact sheet stresses the importance of AI in maintaining U.S. **national security** and economic strength amidst rising global competition.
   - It warns that powerful AI systems could exacerbate risks such as **weapons of mass destruction** and mass surveillance if misused.
- **American Leadership in AI Technologies**: An article from NVIDIA highlights that U.S. leadership in computing has historically driven global influence and innovation in AI.
   - It notes that maintaining a competitive environment has allowed the U.S. to excel in areas such as **healthcare** and **manufacturing**.
- **General Sentiment on Policy Discussions**: Members expressed ambivalence about the AI policy discussions, with one remarking that the situation feels quite **unsettling**.
   - Another member noted the overwhelming coverage of export controls, implying they might not find the discussions necessary to follow closely.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blogs.nvidia.com/blog/ai-policy/">NVIDIA Statement on the Biden Administration’s Misguided &#039;AI Diffusion&#039; Rule</a>: For decades, leadership in computing and software ecosystems has been a cornerstone of American strength and influence worldwide. The federal government has wisely refrained from dictating the design,...</li><li><a href="https://x.com/angelusm0rt1s/status/1878776558644875295">Tweet from Zephyr (@angelusm0rt1s)</a>: Wait, what&#39;s the point of AI diffusion controls if shipments below 1700 GPUs are not counted against national capsMost of smuggling is happening thru multiple shell companies who place small order...</li><li><a href="https://www.whitehouse.gov/briefing-room/statements-releases/2025/01/13/fact-sheet-ensuring-u-s-security-and-economic-strength-in-the-age-of-artificial-intelligence/">FACT SHEET: Ensuring U.S. Security and Economic Strength in the Age of Artificial Intelligence | The White House</a>: Artificial intelligence is quickly becoming central to both security and economic strength. The United States must act decisively to lead this transition
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1327380364339183728)** (13 messages🔥): 

> `Community Chat Etiquette, Command R+ Capabilities, North Waitlist Interest` 


- **Community Keeps it Casual**: Members discussed maintaining a **casual chat environment** in the channel, with some expressing concerns over long message formats.
   - *Sorry* from one member was acknowledged, emphasizing a friendly atmosphere.
- **Exploring Command R+'s Coding Strengths**: A member inquired about the capabilities of **Command R+** specifically for programming in **Rust**.
   - Discussion hinted at ongoing assessments of the tool's effectiveness in coding tasks.
- **Interest in Joining North Waitlist**: A member mentioned their past experience with **Reka space** and expressed interest in joining the **North waitlist**.
   - They noted their active participation in the chat, making them eager for new opportunities.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1327390647749050398)** (53 messages🔥): 

> `Cohere Datasets Bug, Command R+ Benchmarks, Dataset Upload Issues, API Response Error, User Communication Concerns` 


- **Cohere Datasets Bug Discovered**: A user reported a **bug in the Cohere Datasets** when attempting to upload large datasets, which results in a **'TooManyRequestsError'** and causes the account to become non-functional.
   - Multiple attempts to resolve the issue have been made over the last two months, but there has been frustration over perceived lack of support.
- **Issues with Uploading Large Datasets**: The user revealed that trying to upload larger JSONL files, specifically around **800MB** with **180,000 lines**, leads to a **frozen dataset environment** in both web and API interfaces.
   - This bug has rendered the account unusable, leading to consequences for the user's business operations, affecting the ability to use models via the API.
- **Command R+ Benchmark Links Shared**: A user inquired about benchmarks for the **Command R+ model**, prompting a response that provided links to a blog detailing performance evaluations of various Command models.
   - The blog contains insights on functionality and comparisons to enhance understanding of the **Command R+** and its capabilities.
- **User Communication Concerns Raised**: The user expressed frustration regarding perceived neglect from support, stating they had sent **36 emails** regarding the problem but felt ignored until now.
   - Concerns were articulated about a perceived lack of attention to important bugs affecting multiple users and the general customer experience.
- **Efforts to Acknowledge Reported Issues**: Support indicated they were aware of the problems and emphasized ongoing escalation for review, reassuring that it was not just an individual user issue.
   - There were attempts to troubleshoot the issue by suggesting the user recreate API keys, with an acknowledgement of the universality of the reported bug.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/command-r">Command R: RAG at Production Scale</a>: Command R is a scalable generative model targeting RAG and Tool Use to enable production-scale AI for enterprise.</li><li><a href="https://cohere.com/blog/command-r7b">Introducing Command R7B: Fast and efficient generative AI</a>: The smallest model in our R series delivers top-tier speed, efficiency, and quality to build powerful AI applications on commodity GPUs and edge devices. </li><li><a href="https://cohere.com/blog/command-r-plus-microsoft-azure">Introducing Command R+: A Scalable LLM Built for Business</a>: Command R+ is a state-of-the-art RAG-optimized model designed to tackle enterprise-grade workloads, and is available first on Microsoft Azure. </li><li><a href="https://docs.cohere.com/v2/docs/command-r-plus">Cohere&#x27;s Command R+ Model (Details and Application) — Cohere</a>: Command R+ is Cohere&#x27;s model for conversational interaction and long-context tasks, best suited for complex RAG workflows and multi-step tool use.</li><li><a href="https://docs.cohere.com/v2/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>: We&#x27;re excited to announce updates to our Command R and R+ models, offering improved performance, new features, and more.</li><li><a href="https://cohere.com/blog/fine-tuning-command0824">Updates to Command R fine-tuning</a>: Fine-tune the updated Command R 08-2024 with support for newer options giving you more control and visibility including a seamless integration with Weights &amp; Biases.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1328234848921452607)** (4 messages): 

> `API Issues, Trial Account Problems` 


- **Users report delays in API responses**: A user reported waiting for **2 minutes** without receiving a response from the API while on a trial account.
   - Another member responded expressing willingness to help and requested details about the **model and endpoint** in question.
- **Assistance offered for troubleshooting**: One member jumped in to assist the user facing API issues, asking for specific information to understand the problem better.
   - The member engaged politely, indicating readiness to resolve the issue promptly.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1327398917025038336)** (46 messages🔥): 

> `Cohere's command functionalities, Theological programming, LLM code generation, Bot interaction guidelines` 


- **Cohere Command Bot can answer queries**: Users can interact with the Cohere bot by pinging it directly, which allows for a continuous thread of conversation.
   - This bot can execute Google searches and provide specific insights from Cohere's documentation.
- **Light Creation in DivineWill Class**: A user provided Java code demonstrating how a class named **DivineWill** creates light through a static method.
   - This class humorously suggests that *the divine command always succeeds*, exemplifying a playful take on programming.
- **No documentation found for theological code**: The Cohere bot struggled to find documentation related to generating code for **theologically based programming languages**.
   - It indicated there was insufficient information within its resources to support such requests.
- **Cohere's Code Generation for LLMs**: Instructions were shared on how to generate code for LLMs, including a sample in **Python** to explain their functionality.
   - This demonstrates the bot's capability to assist developers by providing code snippets relevant to their queries.
- **Bot's limitations in responding**: At times, the Cohere bot expressed its limitations, stating it couldn't provide responses or fail to find specific details.
   - Several inquiries did not yield results, highlighting areas where users may need to seek external resources.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1327455726859321417)** (3 messages): 

> `O1 Workflow, Claude's Role, Interface Directives` 


- **O1 Workflow Successfully Utilizes Claude**: A member shared that the only **O1 workflow** that has worked for them involves using **Claude** to understand project goals and set up directives.
   - They emphasized the importance of establishing **interfaces between functions** and using logical notation in prompts to optimize the workflow.
- **O1 Performs Well on Algorithms**: After outlining directives, the member noted that O1 tends to perform adequately at executing the **actual algorithms** when properly prompted.
   - This sounds promising, as it indicates that while O1 may have challenges, it shows potential in algorithm execution.
- **Discussion on Group Relevance**: A member expressed uncertainty regarding the appropriateness of this group for discussing their concerns.
   - This highlights a potential mismatch in focus and interests within the group.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1327548765477015572)** (13 messages🔥): 

> `Triton Puzzles Optimization, Autotuning GPU Models, CUDA Block Assignment, Num Stages Impact, Cross Entropy Kernel Improvement` 


- **Optimizing Triton Puzzles on Real GPUs**: A member shared their progress on the [Triton Puzzles](https://github.com/gauravjain14/mlcompilers_and_kernels/tree/main/triton_kernels) and inquired about profiling techniques for hyper-optimization on GPUs.
   - Feedback requested on their work, demonstrating engagement with the Triton community.
- **Results of Autotuning Different GPU Models vary**: A discussion highlighted the performance differences when autotuning inputs on GPUs, specifically between **A100 and A30** models.
   - Another member noted that settings with a large `num_stages` could lead to issues on consumer GPUs due to shared memory limitations.
- **CUDA Block Assignment in Triton**: One member questioned whether Triton assigns multiple programs to a single CUDA block and discussed how it affects kernel occupancy for small data chunks.
   - This raises concerns over achieving high occupancy, which might be simpler in CUDA C.
- **Understanding Num Stages in Operations**: A user sought clarification on the impact of `num_stages` during operations and requested related resources.
   - A member recommended a [YouTube video](https://www.youtube.com/watch?v=PAsL680eWUw) discussing pipelining in persistent kernels to enhance understanding.
- **Improving Cross Entropy Kernel**: A member asked for tips on enhancing their cross entropy kernel to reduce memory usage and increase speed.
   - Another shared a link to the [Liger Kernel implementation](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py) for reference and comparison.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/gauravjain14/mlcompilers_and_kernels/tree/main/triton_kernels">mlcompilers_and_kernels/triton_kernels at main · gauravjain14/mlcompilers_and_kernels</a>: Contribute to gauravjain14/mlcompilers_and_kernels development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=PAsL680eWUw">Pipelining Persistent Kernels</a>: Pawel describes how Triton supports pipelining in the context of persistent kernels. (This talk was voted the audience&#39;s favorite in an informal poll!)Slides...</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py at main · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1327394385150611466)** (7 messages): 

> `CUDA Installation on Ubuntu, CUDA in Visual Studio Code, Blackwell GeForce GPU support, FA3 Profiling on H200 vs H100` 


- **CUDA Installation on Ubuntu Guidance**: A member asked for instructions on how to install **CUDA** on **Ubuntu**, referencing the [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu) for details.
   - The guide provides comprehensive steps necessary for the installation of CUDA Toolkit on Linux systems.
- **Importing CUDA to Visual Studio Code Simplified**: Another inquiry focused on how to import **CUDA** into **Visual Studio Code**, leading to a mention of the [Nsight Visual Studio Code edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition) plugin.
   - This plugin enhances productivity with features like **Intellisense** and debugging, which streamline the development process.
- **Curiosity about Blackwell's Thread Block Cluster Support**: A question was raised about whether the upcoming **Blackwell** on **GeForce GPUs** will support **thread block clusters**.
   - Another member expressed interest in finding the whitepaper for **GeForce Blackwell**, indicating anticipation for further insights.
- **FA3 Performance Comparison on H200 vs H100**: A query was presented regarding the performance difference of **FA3** when run on **H200** compared to **H100**.
   - One contributor confirmed a significant difference exists between **FA3** and **FA2**, but the specific variance in performance between H200 and H100 remains unclear.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition">Nsight&#32;Visual&#32;Studio&#32;Code&#32;Edition&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;CUDA&#32;development&#32;and&#32;debugging&#32;support&#32;for&#32;VS&#32;Code</li><li><a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu">CUDA Installation Guide for Linux</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1327473003210215506)** (11 messages🔥): 

> `Profiler UTF-8 decode issue, Using Flash Attention with Transformers, Challenges with Data Parallelism Strategies, Inference Pipeline for Large Models, NNSight for Memory Efficiency` 


- **Profiler UTF-8 Decode Issue Identified**: A GitHub issue was raised regarding a **UTF-8 decode issue** encountered while using the PyTorch Profiler with a modified Hugging Face transformer's trainer.py. The issue details the failure when wrapping the train function with `profiler.profile` during model execution.
   - The GitHub issue can be viewed [here](https://github.com/pytorch/pytorch/issues/64345) for more context.
- **Implementing Flash Attention in Transformers**: Discussion arose around integrating **Flash Attention** methods with a simple **MultiheadAttention** structure in Torch to enhance performance. Questions were raised on whether a specific setup for using a flash-attn kernel is necessary or if Manual integration via flash-attn2 is required.
- **Sensitivity in Data Parallelism Approaches**: An inquiry was made about the sensitivity of **DDP** and **FSDP** strategies to the usage of modules/parameters outside the wrapped module's `forward` method. The discussion referenced a paper proposing cut cross-entropy loss methods, considering flexibility against strategy defaults like `find_unused_parameters=False`.
- **Building an Inference Pipeline for Large Models**: A user explored building an **inference pipeline** that conserves GPU and CPU memory while handling larger-than-memory models. They plan to utilize Accelerate's meta device wrapper to execute prompts while caching hidden states in CPU RAM to avoid excessive memory consumption.
   - Challenges were recognized in accessing intermediate outputs during inference, making it hard to avoid loading all layers with every request.
- **NNSight Boosts Memory Efficiency**: NNSight was suggested as a solution to optimize memory usage by forming a compute graph that lazily loads layers and activations only when necessary. This method has proven helpful for caching activations to avoid out-of-memory (OOM) issues while analyzing neural networks.
   - The tool creates proxies, allowing memory management in a more efficient manner, appealing for users needing mechanistic interpretability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.09009">Cut Your Losses in Large-Vocabulary Language Models</a>: As language models grow ever larger, so do their vocabularies. This has shifted the memory footprint of LLMs during training disproportionately to one single layer: the cross-entropy in the loss compu...</li><li><a href="https://github.com/pytorch/pytorch/issues/64345">Profiler UTF-8 decode issue · Issue #64345 · pytorch/pytorch</a>: 🐛 Bug To Reproduce Steps to reproduce the behavior: Modify huggingface transformer&#39;s trainer.py, wrap train function with profiler.profile Run a huggingface transformer&#39;s model single-node mu...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1327703410929303654)** (1 messages): 

> `Upcoming Talks, Flash Infer, Mosaic GPU, Turing int8 matmul, Profiling at NVIDIA` 


- **Upcoming talks schedule!**: Scheduled talks include **Zihao Ye on Flash Infer** on **Jan 24** and **Adam Paszke on Mosaic GPU** on **Jan 25**, both at **12:00 PM PST**.
   - Event information can be found in the events tab; suggestions for additional speakers are welcome.
- **Diving into NVIDIA's Profiling Techniques**: **Magnus Strengert** and others from NVIDIA will discuss **profiling** on **Feb 14** at **10:00 AM PST**.
   - This session is expected to provide insights into the efficiency of profiling practices in machine learning.
- **Exploring int8 Matmul Innovations**: **Erik Schultheis** will present on **int8 matmul for Turing** on **Feb 8** at **12:00 PM PST**.
   - This talk aims to shed light on the advancements in matrix multiplication for enhanced performance.
- **Optimizing with CUBLAS Alternatives**: On **Feb 15**, **pranjalssh** will discuss how to **outperform CUBLAS on H100** during his session at **12:00 PM PST**.
   - This promising talk focuses on leveraging the capabilities of the H100 architecture for optimization.
- **Scaling Laws for Low Precision**: **Tanishq Kumar** will present on **scaling laws for low precision** on **March 22** at **12:00 PM PST**.
   - This session is anticipated to explore important trends in low precision computations across architectures.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1327440452814635069)** (4 messages): 

> `GPU expertise hiring at Meta, GenAI inference acceleration, Depth of technical work at Meta` 


- **Meta seeks GPU expertise for GenAI**: Meta is hiring GPU experts to assist in accelerating **GenAI inference**, working on projects like [GPU Kernels](https://www.metacareers.com/jobs/1517576482367228/) and **Compilers**.
   - Interested candidates can reach out directly for more details and opportunities.
- **Team recognized for technical depth**: A member noted that this team publishes some of the **deepest and most technical work** at Meta, reflecting their high impact in the field.
   - Another confirmed this sentiment, stating, *“Can’t agree more!”* highlighting the value of their research contributions.



**Link mentioned**: <a href="https://www.metacareers.com/jobs/1517576482367228/">Software Engineer, Systems ML -  HPC Specialist</a>: Meta&#039;s mission is to build the future of human connection and the technology that makes it possible.

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1327393706009165868)** (7 messages): 

> `Importing CUDA to Visual Studio Code, CUDA Toolkit Installation, Building Copilot with Llama 3.2, CUDA Atomic Functions for Doubles, Using Integer Functions for Doubles` 


- **CUDA Integration with Visual Studio Code**: To import CUDA into Visual Studio Code, you can use the [Nsight Visual Studio Code edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition), which offers various support features.
   - However, you will still need to install the **CUDA Toolkit** alongside this plugin.
- **Building a Copilot with Llama 3.2**: A member sought help in configuring **NVIDIA CUDA** and **Docker** while building a copilot using **Llama 3.2**.
   - Another member suggested looking into **Ollama** as a potential resource for assistance.
- **CUDA Atomic Functions Inquiry**: Inquiring about the existence of a **CUDA** atomic function for finding the minimum of double type elements, a user pointed out that such functions seem only available for integers.
   - A member mentioned that if your doubles are positive, you could **use the integer version** of the function instead.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition">Nsight&#32;Visual&#32;Studio&#32;Code&#32;Edition&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;CUDA&#32;development&#32;and&#32;debugging&#32;support&#32;for&#32;VS&#32;Code</li><li><a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu">CUDA Installation Guide for Linux</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1327426318312149115)** (3 messages): 

> `DGX H100 concerns, Sonoma AI Speaker Series, Fundraising ideas for server` 


- **Concerns about DGX H100**: A member expressed feelings of distress regarding the **DGX H100** in a heartfelt message, referring to it with a tone of lament.
   - An image was shared that visually represents the sentiment conveyed.
- **Launch of Sonoma AI Speaker Series**: A new **AI speaker series** in Sonoma County is set to kick off on January 16th, featuring talks on AI platforms and unstructured data, with [registration required](https://lu.ma/o6br0dg3).
   - Speakers include **Christy Bergman**, **Paco Nathan**, and **Allison Ding**, discussing topics from AI tools to catching bad actors with AI applications.
- **Innovative Fundraising Ideas Proposed**: A member hinted at a potential fundraising concept for the server, indicating a desire to bolster community support.
   - An accompanying image was shared to illustrate the idea further.



**Link mentioned**: <a href="https://lu.ma/o6br0dg3">Sonoma AI with Wine · Luma</a>: This is an in-person event! Registration required in order to get in.Topic: Sonoma AI (and wine) Meetups for remote tech workersWhat we’ll do:Have some food…

  

---


### **GPU MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1328233634091499581)** (5 messages): 

> `Learning CUDA and GPU Programming, Completing Lecture Exercises, Forming Study Groups` 


- **New Learners Discuss CUDA**: New users are expressing interest in learning **CUDA and GPU programming** and have completed the first lecture.
   - One user mentioned focusing solely on the **book exercises** as the next step before advancing to the following lecture.
- **Lectures Offer Strong Foundation**: A responder confirms that the **lectures will provide substantial knowledge**, encouraging new learners to start there.
   - They recommend finding a **working group** to join and contribute to after completing the lecture series.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1327625637321510942)** (6 messages): 

> `Qwen2-VL issues, Error with Liger Kernel, Downgrading Transformers` 


- **Qwen2-VL experiencing kernel issues**: A user reported issues with **Qwen2-VL** when running a simple inference script, questioning whether the **liger kernel** was broken.
   - The error message suggested that the issue might stem from model compatibility conflicts.
- **Encountered error during inference**: An error was triggered when generating output, indicating `TypeError: lce_forward() got an unexpected keyword argument 'cache_position'`.
   - This type of error was not experienced when fine-tuning **Llama3.1** with **liger kernel**, hinting at a specific incompatibility.
- **Possible related issue found on GitHub**: A user linked to an existing GitHub issue indicating a related **IndexError** with the **Qwen2-VL** in context of using the **liger kernel**.
   - The issue referenced encountered similar problems and noted that it occurred during text generation attempts.
- **Workaround by downgrading Transformers**: A workaround suggested was to downgrade the **transformers** package, to resolve compatibility issues with **Qwen2-VL**.
   - An [image](https://cdn.discordapp.com/attachments/1275130785933951039/1327631470457782392/image.png) was shared showing the steps to facilitate this downgrading process.



**Link mentioned**: <a href="https://github.com/linkedin/Liger-Kernel/issues/515">IndexError: The shape of the mask [7387] at index 0 does not match the shape of the indexed tensor [1] at index 0 · Issue #515 · linkedin/Liger-Kernel</a>: 🐛 Describe the bug The error exists when I try to use the qwen2-vl with qwen2-vl liger kernel to generate text. The following code got the following error. But the same code if I change the liger k.....

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1327548266761420820)** (2 messages): 

> `ASRG Season 1 Pilot, Maya Multilingual Vision-Language Model` 


- **ASRG Season 1 Kicks Off**: The **Pilot of ASRG Season 1** is set for tomorrow, featuring a reading of [The Linux Kernel Module Programming Guide in C](https://x.com/asrg_gg/status/1877968239084687602). Participants are encouraged to prepare an x86-64 VM with Ubuntu 22.04 or to use multipass.
   - *Check it out and share some love!*
- **Maya Preprint Announcement**: A member announced their work on **Maya: Multilingual Vision-Language Model**, with the preprint now available at [this link](https://twitter.com/nahidalam/status/1866667770114609217).
   - They expressed excitement about sharing this development with the community.



**Link mentioned**: <a href="https://x.com/asrg_gg/status/1877968239084687602">Tweet from Systems Reading Group (@asrg_gg)</a>: EP0: Pilot of ASRG Season 1 is tomorrow!We’ll read The Linux Kernel Module Programming Guide in C. Make sure to have an x86-64 VM with Ubuntu 22.04 or just use multipass like @nanod1jkstra does.

  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1327813188057825362)** (5 messages): 

> `Vulkan on Raspberry Pi 5, Nvidia Cosmos on Jetson, Transformer Engine Porting, 3D Vision Stack Libraries` 


- **Vulkan Runs Smoothly on Raspberry Pi 5**: A [GitHub pull request](https://github.com/pytorch/executorch/pull/7615) suggests changing the Vulkan allocation to **SEQUENTIAL_WRITE** to enhance performance on the **Raspberry Pi 5**.
   - This adjustment aims to resolve issues with Vulkan functionality on the device, although Vulkan remains untested by the user.
- **Nvidia Cosmos Powers Jetson Devices**: Discussion highlights the implementation of **Nvidia Cosmos** on Jetson platforms, focusing on enhancing AI capabilities.
   - *JohnnyCano* shared [this LinkedIn post](https://www.linkedin.com/posts/johnnycano_nvidia-cosmos-nvidiacosmos-activity-7283774665943109632-VQDa?utm_source=share&utm_medium=member_ios) detailing the experience with the technology.
- **Extensive Transformer Engine Porting Efforts**: One member revealed that they have successfully ported the **Transformer Engine** along with more than **30 libraries**.
   - This broad effort demonstrates significant advancements in integrating transformer models across various platforms.
- **3D Vision Stack Libraries Introduced**: The user mentioned ports of libraries including **Mamba** and the **3D vision stack**, indicating growing development in visual AI solutions.
   - These contributions are part of the broader initiative to enhance GPU capabilities and library support.



**Link mentioned**: <a href="https://github.com/pytorch/executorch/pull/7615">[ET-VK] Request VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, not RANDOM by swolchok · Pull Request #7615 · pytorch/executorch</a>: SummaryIt looks like we are careful to use only copy_from and copy_to with StagingBuffer on CPU, in which case we only need SEQUENTIAL_WRITE.This matters on Raspberry Pi 5, where there appears (f...

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1328415568939716690)** (4 messages): 

> `2025 Community Meeting, MAX GPU Benchmarking, MAX-CV, Meeting Video Upload, Attendance Concerns` 


- **Kickoff of 2025 Community Meeting**: The first community meeting of **2025** is scheduled to discuss **MAX GPU benchmarking** and **MAX-CV**, with Q&A sessions planned for community members. Interested participants can RSVP and join the meeting via [this link](https://discord.com/events/1087530497313357884/1300880439673884712).
   - The meeting started just a few minutes after the announcement, highlighting the importance of community engagement.
- **Meeting Questions Addressed**: During the community meeting, Chris Lattner addressed a question posed by a member, although the video recording's availability remains uncertain. Members expressed anticipation for the meeting video, with updates promised by Caroline Frasca.
   - It's expected that the meeting video will be uploaded either today or tomorrow, with follow-up communication to be shared with the community.
- **Attendance Challenges**: One member expressed gratitude for the update regarding the meeting content, noting they could only attend the start due to class conflicts. This highlights ongoing accessibility concerns for community members with scheduling overlaps.
   - The conversation reflects the community's continued interest in participation despite personal scheduling challenges.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1327378583584247839)** (23 messages🔥): 

> `Testing Mojo Code on macOS, Nightly Documentation for Mojo/Max, Async Proposals for Mojo, Compiler Issues with Mojo, Int8 to String Conversion in Mojo` 


- **Testing Mojo Code on macOS**: A user requested assistance to test Mojo code on a macOS device to ensure cross-platform functionality.
   - Another member volunteered to help by direct messaging them.
- **Nightly Documentation for Mojo/Max**: Inquiries were made about the availability of a nightly documentation site for Mojo/Max, which was confirmed to be accessible.
   - Users were directed to change the version number in the documentation link to view the nightly version.
- **Async Proposals for Mojo**: Proposals for introducing structured asynchronous programming in Mojo were shared, aimed at avoiding performance compromises.
   - The discussion aimed at building a unified ecosystem around Mojo's async capabilities with interested members being pinged.
- **Compiler Issues with Mojo**: A member encountered a crash in the Mojo compiler while working on a list of structs implementing a common trait.
   - Feedback suggested that the issue originated from improper initialization and a recommendation to file a bug report was made, which the member did.
- **Int8 to String Conversion in Mojo**: An issue was reported concerning the conversion of Int8 to string in Mojo, referencing a specific example from Mojodojo.
   - Discussion highlighted the importance of understanding data types in Mojo, and directing the user to relevant documentation regarding compile vs runtime values.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=42659061">Flattening ASTs and other compiler data structures (2023) | Hacker News</a>: no description found</li><li><a href="https://docs.modular.com/mojo/manual/types/">Types | Modular</a>: Standard Mojo data types.</li><li><a href="https://github.com/modularml/mojo/issues/3944">[BUG] Compiler crash when defining a list of structs based on a common trait. · Issue #3944 · modularml/mojo</a>: Bug description The Mojo compiler crashes on the line, _list = List[Bar[TFoo]]() in the code example below. Discussed on Discord with @owenhilyard. https://discord.com/channels/1087530497313357884/...</li><li><a href="https://github.com/modularml/mojo/issues/3947">[mojo-examples] Mojodojo Int8 to string conversion example not working · Issue #3947 · modularml/mojo</a>: Where is the problem? https://mojodojo.dev/guides/intro-to-mojo/basic-types.html#strings What can we do better? This conversion var word = List[Int8]() word.append(78) word.append(79) word.append(0...</li><li><a href="https://docs.modular.com/nightly/">MAX Docs | Modular</a>: MAX is a unified set of APIs and tools that help you build and deploy high-performance AI pipelines.</li><li><a href="https://docs.modular.com">MAX Docs | Modular</a>: MAX is a unified set of APIs and tools that help you build and deploy high-performance AI pipelines.</li><li><a href="https://www.cs.cornell.edu/~asampson/blog/flattening.html">Flattening ASTs (and Other Compiler Data Structures)</a>: This is an introduction to data structure flattening, a special case of arena allocation that is a good fit for programming language implementations. We build a simple interpreter twice, the normal wa...</li><li><a href="https://docs.modular.com/mojo/manual/parameters/">Parameterization: compile-time metaprogramming | Modular</a>: An introduction to parameters and compile-time metaprogramming.</li><li><a href="https://github.com/modularml/mojo/pull/3945">[proposal] Structured Async for Mojo by owenhilyard · Pull Request #3945 · modularml/mojo</a>: Proposes to add structured async to Mojo, following in the the Rust tradition of async since Mojo has the ability to fix many of the issues with Rust&amp;#39;s async, some of which are ecosystem infli...</li><li><a href="https://github.com/modularml/mojo/pull/3946">[proposal] Provided Effect Handlers by owenhilyard · Pull Request #3946 · modularml/mojo</a>: This proposal contains an alternative to an effect system which I think is more suitable for abstracting async, raises, and similar function colors in a systems language where the context may not a...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

wiltonb: Happy reading!

https://kanesimms.substack.com/p/what-agentic-ai-actually-is-a-deeply
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1327713883145764875)** (19 messages🔥): 

> `AzureOpenAI Integration, dspy.react with phi-4 Functionality, Getting Started with DSPy, Optimizing LLMs, Prompt Performance Across Models` 


- **AzureOpenAI Client Setup Example**: A member shared a code example for initializing the **AzureOpenAI** client, demonstrating the use of API credentials and parameters.
   - They referenced sections of the [Azure OpenAI documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview) for additional context.
- **dspy.react Enables phi-4 Function Calling**: A member pointed out that **dspy.react** allowed **phi-4** to perform function calling, which was surprisingly effective despite initial doubts regarding the model's training.
   - They noted that although performance was not optimal, it showcased the flexibility of function calling within the architecture.
- **DSPy for Voice AI Projects**: A new member inquired about starting a voice AI project with **DSPy**, expressing interest in beginner-friendly resources.
   - Another member highlighted the lack of current voice support, directing them to a [GitHub issue](https://github.com/stanfordnlp/dspy/issues/2037) discussing future audio capabilities.
- **Navigating Optimization with LLMs**: A user shared their experience optimizing an LLM as a judge, emphasizing the seamless improvement in performance without manual adjustments.
   - Discussions emerged regarding the effectiveness of nesting optimizers and whether multiple rounds of optimization are beneficial.
- **Prompt Performance Variation Among Models**: A member queried the expected performance differences when using prompts optimized for a smaller model like **gemini-8b** compared to a larger one like **deepseekv3**.
   - They theorized that prompts might be model-specific and could not equally address errors across different architectures, which another member affirmed as a common challenge.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/stanfordnlp/dspy/issues/2037">Support for audio files · Issue #2037 · stanfordnlp/dspy</a>: Similar to dspy.Image, it would be useful to add dspy.Audio. We&#39;ve started using DSPy recently for our voice AI agent, but lack of support for audio is a blocker for many use-cases. We&#39;re happ...</li><li><a href="https://docs.litellm.ai/docs/providers/azure">Azure OpenAI | liteLLM</a>: Overview
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1327403528649703424)** (20 messages🔥): 

> `Phi-4 Models, Adaptive Batching, Using Instruct Models for Medical Training, Quality over Quantity in Training Data` 


- **Candidate file for Phi-4 finetuning**: A member requested a 'dummy' version for finetuning **Phi-4**, linking to a [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb).
   - Another user mentioned that their **Phi-4 PR** will likely be merged soon, thus the need for such a file may not arise.
- **Discussion on Adaptive Batching RFC**: A member expressed interest in getting feedback on their [RFC for adaptive batching](https://github.com/pytorch/torchtune/pull/2199) in **Torchtune**.
   - If the feedback is positive, they plan to implement changes and proceed with the next iteration.
- **Choosing the Right LLaMA Model for Medical Training**: A member is evaluating whether to use an **instruct** or **non-instruct LLaMA model** for enhancing medical capabilities with their dataset of 50b tokens.
   - They are considering experimenting with the **10B instruct dataset**, acknowledging the importance of model post-training.
- **Importance of Data Quality in Training**: Another member highlighted that **data quality > data quantity**, advocating for well-prepared diverse datasets over vast amounts of raw data.
   - They suggested evaluating documents using other LLMs to judge their effectiveness before extensive resource consumption.
- **Sharing Research on Smaller Models**: A member shared the publication of their results using smaller models based on **Mistral 7B**, which proved effective for pretraining.
   - They referenced the published paper on medical society guidelines, accentuating the significance of utilizing high-quality documents in training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://arxiv.org/abs/2402.12847">Instruction-tuned Language Models are Better Knowledge Learners</a>: In order for large language model (LLM)-based assistants to effectively adapt to evolving information needs, it must be possible to update their factual knowledge through continued training on new dat...</li><li><a href="https://github.com/pytorch/torchtune/pull/2199">[RFC] Online and offline adaptive batching in torchtune. by krammnic · Pull Request #2199 · pytorch/torchtune</a>: About: #2191Enable adaptive batching in torchtuneIntuitionIt is useful to set a maximum batch size that is not causing OOM for a given compute. Also, it might be interesting to increase batch si...
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1327366464272863324)** (16 messages🔥): 

> `MOOC Enrollment, Final Project Results, Weekly Lectures Start Date, Assignment Submission Process, Course Difficulty` 


- **MOOC Enrollment is Automatic**: Once you fill out the SP 25 signup form, you are automatically enrolled in the MOOC without any fees.
   - *It's free, guys!*
- **Final Project Results Coming Soon**: The final results for projects are expected to be released sometime later this month, hopefully within the next week.
   - Stay tuned for updates!
- **Weekly Lectures Kick Off January 27th**: Weekly lectures for the MOOC are scheduled to start on **January 27th**.
   - Prepare for an exciting learning experience!
- **Separate Forms for Assignments**: Assignments will require separate Google forms for submission, ensuring progress tracking via email addresses.
   - Make sure to use the same email for each assignment!
- **Gauge Difficulty by Reviewing Past Lectures**: For those questioning the beginner-friendliness of the MOOC, it's suggested to review the lectures from the [Fall 2024 MOOC](https://llmagents-learning.org/f24).
   - The Spring 2025 MOOC builds upon these concepts with slightly increased difficulty, but no prerequisites!



**Link mentioned**: <a href="https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing)">Quizzes Archive - LLM Agents MOOC</a>: NOTE: The correct answers are in the black boxes (black text on black background). Highlight the box with your cursor to reveal the correct answer (or copy the text into a new browser if it’s hard to ...

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1327386820094197921)** (2 messages): 

> `AI Builders Summit, AutoRAG Framework, RAG Techniques, Small Language Models` 


- **Join the AI Builders Summit with 40+ speakers!**: Catch our own [@seldo](https://twitter.com/seldo) and more than **40 speakers** at the AI Builders Summit, a **4-week virtual training course** hosted by [@_odsc](https://twitter.com/_odsc).
   - Participants will learn to **customize open-source small language models** for enterprise use and **scale RAG systems** without compromising performance.
- **Introducing AutoRAG for Optimal RAG Pipelines**: The **AutoRAG** framework offers a method for selecting the optimal configuration for your **RAG pipelines** and was introduced in a recent paper. This is especially relevant for LlamaIndex users as it systematically evaluates various techniques and components.
   - The approach facilitates tailored configurations, enhancing the effectiveness of RAG implementations.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1327794595035349058)** (12 messages🔥): 

> `LlamaIndex Engineer Search, GraphRAG Visualization Issue, OpenAI Model Prompt Caching, Dynamic Variables in Prompt Templates` 


- **Seeking LlamaIndex Engineer for Discussion**: A member is looking for an engineer experienced with **LlamaIndex** and bot implementations, offering compensation for their time to discuss an implementation.
   - They requested interested individuals to DM with information showcasing their knowledge.
- **GraphRAG Notebook Shows Only Nodes**: Discussion arose regarding the **GraphRAG Notebook** displaying only nodes without relationships when graphing a book, even with default OpenAI models.
   - Another member suggested that this behavior could be linked to **fine-tuning** processes.
- **Prompt Caching for OpenAI Models**: A member inquired about tutorials for prompt caching similar to the **Anthropic** example, voicing concerns over the lack of specific resources for OpenAI models.
   - Another member affirmed that OpenAI's prompt caching is **automatic**, referencing a documentation link.
- **Adding Dynamic Variables to Prompt Templates**: A member sought guidance on adding custom prompt templates and dynamic context variables in `QuestionsAnsweredExtractor` within LlamaIndex.
   - Another member recommended using **function mappings** to attach any variable dynamically to the prompt template.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://]">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/prompts/advanced_prompts/#3-prompt-function-mappings">Advanced Prompt Techniques (Variable Mappings, Functions) - LlamaIndex</a>: no description found</li><li><a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/property_graph_neo4j.ipynb).">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1327588257474674749)** (14 messages🔥): 

> `EPUB file support, LLama model prompt templates, AI context length limitations, Exporting chat history, Running GPT4All remotely` 


- **EPUB File Reading Capabilities**: A user inquired if GPT4All can read **.epub** files, to which another member confirmed that it should work, but cautioned about issues with certain languages like **Chinese**.
   - This highlights the importance of language compatibility in file support.
- **Creating Jinja Prompt Templates for Llama**: A user shared their challenge in developing a relevant **Jinja prompt template** for their fine-tuned **Llama model**, since `get_chat_template()` didn't work.
   - They are seeking advice to effectively use their model with GPT4All, showcasing the intricacies involved in prompt design.
- **Understanding AI Context Length**: Concerns were raised about how the **context length** in GPT4All limits conversation recall to approximately **2048 tokens**.
   - Members clarified that text gets truncated once the conversation length exceeds this limit, regardless of whether it’s from a chat or an exported file.
- **Desire for Full-Chat Exporting Feature**: A user expressed the need for **full-chat exporting** to easily read previous conversations without manual copying and pasting.
   - However, the team acknowledged that such a feature is currently unavailable and encouraged submitting a request on their GitHub page.
- **Remote Access to GPT4All via Reverse Proxy**: A user sought ways to use GPT4All remotely from a less powerful laptop, leading to advice about running a **VPN or reverse proxy** on their desktop.
   - This solution implies feasible strategies to access powerful processing remotely while leveraging the main desktop's capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.gpt4all.io/gpt4all_desktop/settings.html#sampling-settings">Settings - GPT4All</a>: GPT4All Docs - run LLMs efficiently on your hardware</li><li><a href="https://github.com/nomic-ai/gpt4all/issues">nomic-ai/gpt4all</a>: GPT4All: Run Local LLMs on Any Device. Open-source and available for commercial use. - nomic-ai/gpt4all
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1327642810719015073)** (11 messages🔥): 

> `Tinygrad Tensor Compiler, Meeting #53 Agenda, Stale PR Closure, FSDP Bounty Lock Discussion, Understanding Tinygrad` 


- **Tinygrad's Tensor Compiler Explained**: Tinygrad serves as a tensor compiler simplifying complex ML operations like convolutions and attention into basic blocks, similar to LLVM for Rust. It employs a minimal instruction set and kernel fusion for optimal GPU performance, allowing flexible compilation for various hardware [{GitHub link}](https://github.com/tinygrad/toonygrad/blob/master/PLAN.md).
   - These fused kernels are generated and executed on compatible hardware, optimizing performance through operation combination.
- **Key Topics for Meeting #53 on Monday**: Meeting #53 is scheduled for **9:30 AM** San Diego time, covering critical points like company updates and CES involvement. Other items include discussions on **DSP contracts**, **Python speed**, and **MLPerf BERT** evaluations.
   - The agenda also highlights updates on scheduling, driver issues, **ONNX**, and various bounties including **Tensor cores** and **RetinaNet**.
- **Reminder to Close Stale PRs**: A direct appeal was made for team members to close any stale or outdated pull requests (PRs). This call to action emphasizes the importance of maintaining a clean and functional codebase.
   - The reminder helps focus effort on current and relevant contributions to the project.
- **FSDP Bounty Lock Query and Conditions**: A developer inquired about the potential for a bounty lock while working on **FSDP** within Tinygrad, linked with a specific PR titled [FSDP in Tinygrad](https://github.com/tinygrad/tinygrad/pull/8571).
   - Conditions for securing the bounty included demonstrating functionality through model training that exceeds single GPU capacity.
- **Seeking Resources to Understand Tinygrad**: A discussion arose about finding comprehensive resources on Tinygrad and its purpose beyond the official documentation. The user expressed difficulty in connecting Tinygrad's concept with their existing knowledge of LLVM.
   - This inquiry reflects a broader interest in the foundational motivations behind Tinygrad's development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/toonygrad/blob/master/PLAN.md">toonygrad/PLAN.md at master · tinygrad/toonygrad</a>: Because tinygrad got out of hand with line count. Contribute to tinygrad/toonygrad development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/pull/8571">FSDP in Tinygrad [WIP] by KhanerX · Pull Request #8571 · tinygrad/tinygrad</a>: FSDP semantics can be fully captured by: Sharding model and optimizer parameters across GPUS (already supported by multi.py) Sharding gradients along the same axis as parametersThis is done...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1327615843563212822)** (2 messages): 

> `Activation Checkpointing, Memory Management in Tinygrad` 


- **Seeking Activation Checkpointing Methods**: A member inquired about methods for performing **activation checkpointing** in **tinygrad**, aiming to enhance memory efficiency during training.
   - *Activation checkpointing is essential for reducing memory costs while allowing backpropagation; thus, this topic is crucial for resource management.*
- **Freeing Memory Without Breaking Gradient Context**: The same member asked how to **free memory** used by return tensors of functions without disrupting the **gradient context graph** in tinygrad.
   - *Addressing memory management without breaking the gradient graph is a significant concern among users aiming for efficient training practices.*


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1327391944820920360)** (10 messages🔥): 

> `Installing Open Interpreter, Homebrew and pipx, Open Interpreter functionality` 


- **JanitorJesh struggles with Open Interpreter installation**: A user expressed difficulty installing **Open Interpreter** on their Mac, running into **error codes** related to the tiktoken package and needing a Rust compiler.
   - After a bit of troubleshooting advice, they successfully set everything up.
- **Deseculavalutent shares installation steps**: A user outlined the installation steps for Open Interpreter using **Homebrew** and **pipx**, emphasizing the benefits of isolating Python applications.
   - This included commands to install Homebrew, pipx, and then Open Interpreter itself.
- **Questions about Open Interpreter's capabilities**: After getting set up, a user inquired about the most useful functions of Open Interpreter, specifically about its ability to edit videos.
   - Another member reassured that it can run arbitrary commands, including video editing commands.
- **Discussion on Open Interpreter's screen control feature**: A user mentioned they have never used the feature of Open Interpreter that can control and look at the screen.
   - This sparked curiosity about the potential capabilities and limitations of this functionality.


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1327716773876731924)** (8 messages🔥): 

> `Stable Audio 3 Open Source Announcement, Hypertension Recognition Dataset Request` 


- **Stable Audio 3 Goes Open Source**: Great news for audio enthusiasts! **Stable Audio 3** will be [open source](https://vxtwitter.com/dadabots/status/1878505508711157791) and trained on music, offering exciting opportunities for developers and creators.
   - This development promises to enhance music-based projects and toolkits in the audio space.
- **Request for Hypertension Audio Dataset**: A member is seeking a **dataset for hypertension recognition**, specifically an audio dataset of individuals with hypertension.
   - They are requesting assistance, indicating a need for collaboration in data collection for this health-related project.



**Link mentioned**: <a href="https://vxtwitter.com/dadabots/status/1878505508711157791">Tweet from undefined</a>: no description found

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1328309778870698055)** (1 messages): 

> `Megatron Checkpoint Conversion, Evaluation Scripts, NVIDIA MegaTron-LM` 


- **Need for Megatron Checkpoint to HF Conversion**: A member has conducted reference training runs with **Megatron** and seeks a script to convert checkpoints from **torch format** to **HF format** without using Nemo.
   - They highlighted that having this script would **save a lot of work** and requested anyone with relevant code to share it.
- **Accessing Training Materials**: There was a mention of materials related to the **checkpoint example** and training logs available [here](https://fz-juelich.sciebo.de/s/Yh8Q8RRTxliupLh) but noted permissions issues prevent file uploads.
   - This appears to be a barrier for sharing important training resources and prompts a need for alternative methods of exchange.
- **Cloning NVIDIA MegaTron-LM**: The member confirmed that the **MegaTron-LM** repo used for runs is a clone of the official NVIDIA repository, specifically identified by the commit hash `31a29b87`.
   - This facilitates cloning from the official source for reference and ensures alignment with the latest updates.
- **Request for Successful Code Exchange**: The user encouraged team collaboration, inviting members to share any **working code** or hints regarding the checkpoint conversion process for Megatron.
   - They emphasized community support as key in solving the ongoing challenge of checkpoint conversion.



**Link mentioned**: <a href="https://fz-juelich.sciebo.de/s/Yh8Q8RRTxliupLh">sciebo - www.hochschulcloud.nrw</a>: C4_50B_cosine_bs-4M_lr-6e-3_warmup-1000 is publicly shared

  

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

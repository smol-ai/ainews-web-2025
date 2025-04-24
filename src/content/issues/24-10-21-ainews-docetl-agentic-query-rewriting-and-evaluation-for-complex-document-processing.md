---
id: ca91a315-7798-4ce9-bd9b-21efdb591416
title: 'DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing'
date: '2024-10-22T00:04:21.441910Z'
original_slug: ainews-docetl-agentic-query-rewriting-and
description: >-
  UC Berkeley’s EPIC lab introduced innovative LLM data operators with their new
  frameworks LOTUS and DocETL, offering advanced APIs for handling large data
  corpora and presenting a promising alternative to GPU-heavy big lab
  approaches. Additionally, Microsoft open-sourced BitNet b1.58, a 1-bit LLM
  enabling 4-20x faster training and efficient 100B model inference on a single
  CPU, significantly advancing on-device AI capabilities.
tags:
  - uc-berkeley
  - epic
  - lotus
  - docetl
  - microsoft
  - bitnet
  - on-device
  - llm
  - model-optimization
---


<!-- buttondown-editor-mode: plaintext -->**LLM data operators are all you need.**

> AI News for 10/18/2024-10/21/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **32** Discords (**231** channels, and **6066** messages) for you. Estimated reading time saved (at 200wpm): **791 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We usually reserve the feature story of AINews for the single biggest impactful news item of the day, but that usually results in a heavy bias towards recapping press releases from big model labs. Other stories of the year develop gradually, more of a swell than a splash, and may not be as big but are still useful as part of a well diversified diet. We use quieter days like these to shed some cumulative light on community tools like [DSPy](https://buttondown.com/ainews/archive/ainews-the-dspy-roadmap/) and [AI price cut stories](https://buttondown.com/ainews/archive/ainews-too-cheap-to-meter-ai-prices-cut-50-70-in/).

UC Berkeley has been a leader in many of the biggest waves in tech - per [David Patterson](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-123.pdf), the 40 year history of UCB research labs have spawned everything from RISC, RAID, and massive companies like Databricks. The latest lab in this tradition is [EPIC](https://x.com/UCBEPIC) — focused on *E*ffective *P*rogramming, *I*nteraction, and *C*omputation with Data. We were fortunate to attend [their recent conference](https://x.com/adityagp/status/1843695418338881941) and were particularly impressed by two similar papers, [LOTUS](https://x.com/lianapatel_/status/1813981153709441361) and [DocETL](https://x.com/sh_reya/status/1848415442244931861) the latter of has been [the subject of notable hype](https://x.com/sh_reya/status/1838617833393283428) and was finally [published today](https://arxiv.org/abs/2410.12189). Both offer some very well thought through LLM operators over large corpuses of data.

<img width="538" alt="image" src="https://gist.github.com/user-attachments/assets/00b20959-f486-4be8-82b5-c60c0cdf5baa">

<img width="1316" alt="image" src="https://gist.github.com/user-attachments/assets/997d3bcd-cefa-4476-a1f4-ee107ba5e759">

The [github docs](https://ucbepic.github.io/docetl/operators/map/) give more of idea of the proposed APIs and concepts, and at the limit this could be viewed as "just another LLM framework" similar to DSPy, but the big data focus at an institution known for successfully thinking about commercially relevant big data problems makes this one worth a closer look than the average twitter anon:

<img width="881" alt="image" src="https://gist.github.com/user-attachments/assets/312632cd-3be6-40e6-9fb5-ba87c01846d7">

At the very highest level this is just the latest front in the ongoing battle between GPU Rich Big Labs (Deepmind, OpenAI) and GPU Poor [Compound AI](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) approaches to AI. The [DocETL demo site](https://www.docetl.org/#demo-docetl-output) helps you compare results and approaches between using their framework and "sticking it all in context". There will likely not be a clear winner here for a long time and AI Engineers will simply have to be familiar with both.

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

**AI Acceleration**

- **BitNet advancements**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848061956697301072) highlighted the open-sourcing of BitNet b1.58 by Microsoft, a **1-bit LLM where every parameter is ternary {-1, 0, 1}**. This approach allows for **4-20x faster training, improved stability, and better handling of longer contexts** without modifying positional encodings. The model achieves speeds of 1.7 tokens/second on 100B LLaMa inference.

- **On-device AI**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848076536714531252) reported that bitnet.cpp can run a **100B BitNet b1.58 model on a single CPU**, achieving speeds comparable to human reading (5-7 tokens per second), significantly enhancing the potential for running LLMs on local devices.

**AI Model Developments and Research**

- **Significant AI progress**: [@adcock_brett](https://twitter.com/adcock_brett/status/1848032213532651701) summarized major developments from various companies including Archetype AI, NVIDIA, Boston Dynamics, Toyota Research, Google, Adobe, OpenAI, Mistral, Tesla, and Meta.

- **New models and benchmarks**: [@adcock_brett](https://twitter.com/adcock_brett/status/1848032258159943735) reported that Nvidia quietly released a new open-sourced, fine-tuned LLM called **Llama-3.1-Nemotron-70B-Instruct**, which outperforms GPT-4o and Claude 3.5 Sonnet on benchmarks, despite being smaller at 70B parameters.

- **Multimodal advancements**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1848000249920364838) highlighted Meta's release of **Spirit LM**, the first open-source multimodal language model that integrates speech and text, offering word-level interleaving of speech and text datasets and cross-modality generation capabilities.

- **AI reasoning capabilities**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1847993405164364224) shared insights from an Apple paper suggesting that **LLMs lack robust mathematical reasoning**, relying on pattern matching rather than genuine conceptual understanding. The paper introduces the GSM-Symbolic benchmark to evaluate LLM performance across different question variants.

**AI Applications and Tools**

- **AI-generated art**: [@fabianstelzer](https://twitter.com/fabianstelzer/status/1848007989275300213) observed that **AI-generated AI art is outperforming human-generated AI art**, noting interesting results from a fine art camera GLIF inspired by online research into "sigils".

- **Cursor hype**: [@vikhyatk](https://twitter.com/vikhyatk/status/1848048132929515528) commented on the popularity of Cursor, suggesting it's a significant improvement over basic text editors like Notepad.

- **LLM Engineer's Handbook**: [@maximelabonne](https://twitter.com/maximelabonne/status/1848029371803832767) announced that the LLM Engineer's Handbook is the #1 New Release in Neural Networks, aiming to help a new generation of LLM engineers build production-level AI systems.

**AI Ethics and Societal Impact**

- **AI capabilities vs human intelligence**: [@bindureddy](https://twitter.com/bindureddy/status/1848136882284044369) argued that while LLMs may hit a wall in a year, they are already smarter than most humans. The tweet suggests that **the last mile in AI automation is not intelligence, but "plumbing"**.

- **AI and democracy**: [@francoisfleuret](https://twitter.com/francoisfleuret/status/1847985070197383607) expressed concern about the potential impact of AI on democracy, stating that "Bad @elonmusk is happy to shred democracy in tiny pieces and sell it as a cheap product in the aisles of a supermarket."

**Memes and Humor**

- [@fabianstelzer](https://twitter.com/fabianstelzer/status/1848066835427545148) shared a humorous tweet about giving a "namshub glifbot" access to a Pepe lora, resulting in the generation of singularity-themed Pepes.

- [@vikhyatk](https://twitter.com/vikhyatk/status/1848048132929515528) joked about the Cursor hype, saying it "must feel like a massive improvement over notepad.exe".

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Advancements in LLM Architecture and Training**

- **nGPT: Faster Convergence by Performing Optimization on a Hypersphere** ([Score: 126, Comments: 25](https://reddit.com//r/LocalLLaMA/comments/1g8cba0/ngpt_faster_convergence_by_performing/)): **nGPT**, developed by **Nvidia**, is a new GPT variant that constrains vectors to a **hypersphere**, resulting in **4 to 20 times faster** convergence than traditional GPT models and improved handling of longer text sequences. This approach simplifies training by eliminating the need for weight decay or special learning rate adjustments, while analysis shows that attention and MLP blocks make smaller adjustments to hidden states and normalization scaling factors remain stable across layers. The [nGPT paper](https://arxiv.org/html/2410.01131) presents this as a promising approach for more efficient and effective language models.

- **COGNITIVE OVERLOAD ATTACK: PROMPT INJECTION FOR LONG CONTEXT** ([Score: 33, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1g8bwkw/cognitive_overload_attack_prompt_injection_for/)): The study explores **Cognitive Overload Attacks** on Large Language Models (LLMs), drawing parallels between human cognition and LLM behavior under information overload. Researchers demonstrated that attackers can exploit this vulnerability to bypass safety mechanisms in advanced models like **GPT-4** and **Claude-3-Opus**, achieving attack success rates of up to **99.99%**. The authors propose incorporating **cognitive load management** techniques from neuroscience into AI design to enhance LLM resilience against such adversarial attacks.

**Theme 2. Innovative LLM Frameworks and Tools for Developers**

- **GraphLLM now has a GUI: open source graph based framework for performing inference with a LLM** ([Score: 114, Comments: 11](https://reddit.com//r/LocalLLaMA/comments/1g816ee/graphllm_now_has_a_gui_open_source_graph_based/)): **GraphLLM**, an open-source graph-based framework for LLM inference, now features a **GUI** similar to ComfyUI, allowing real-time streaming of node outputs to the front-end. The framework supports advanced features like **loops**, **parallel execution**, **conditionals**, and **custom Python code execution**, while maintaining transparency in prompt handling and offering various pre-built examples, including **YouTube subtitle summarization**, **majority voting**, and an **agent capable of web searches and file access**. Additional tools include a **web scraper** using a headless Firefox instance for handling dynamic websites, a **YouTube subtitles downloader**, and a **PDF parser**, with the source code available on [GitHub](https://github.com/matteoserva/GraphLLM).

- **Generate text with alternative words and probabilities** ([Score: 60, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1g83jii/generate_text_with_alternative_words_and/)): **ActuosusAI**, a personal hobby project, introduces a feature that allows users to modify **LLM output** by navigating through alternative routes while specifying **temperature**, with a minimum **0.01% probability** threshold for token sampling. The project, available on [GitHub](https://github.com/TC-Zheng/ActuosusAI), is a local app with a web UI that supports downloading models from **Huggingface**, loading them in different quantizations with **GGUF format** support, and generating text.
  - **Chromix_** suggests adding a **min_p slider** and **color coding** for word options to enhance exploration of low temperature generations. They also propose supporting **OpenAI-compatible API** calls and auto-exploring branch levels during user idle time.
  - Users appreciate the project's **interactive backtracking sampler** and **UX**. There's interest in visually hinting at tokens with wider distributions to guide users towards more impactful choices.
  - Suggestions for improvement include implementing **GPU offload** support and enhancing the UI with features like color-coded options and sliders for more intuitive interaction with the model's output.


**Theme 3. Local LLMs Outperforming Cloud Alternatives**

- **Mistral-Large-Instruct-2407 really is the ChatGPT at home, helped me where claude3.5 and chatgpt/canvas failed** ([Score: 238, Comments: 80](https://reddit.com//r/LocalLLaMA/comments/1g878zy/mistrallargeinstruct2407_really_is_the_chatgpt_at/)): **Mistral-Large-Instruct-2407** outperformed **Claude 3.5** and **ChatGPT** in integrating code from two repositories: **Lucid_Autonomy** (**1500** lines) and **Lucid_Vision** (**850** lines). The author experienced frustrations with Claude's focus on irrelevant functions and ChatGPT's inability to rewrite necessary code, while Mistral-Large-Instruct-2047 completed the task with minimal guidance, as evidenced in the [conversation log](https://github.com/RandomInternetPreson/Lucid_Vision/tree/main/LocalLLM_Update_Convo).

- **[I made a better version of the Apple Intelligence Writing Tools for Windows! It supports a TON of local LLM implementations, and is open source & free :D](https://v.redd.it/0zm105dfbxvd1)** ([Score: 135, Comments: 30](https://reddit.com//r/LocalLLaMA/comments/1g80bna/i_made_a_better_version_of_the_apple_intelligence/)): The post introduces a **Windows-compatible alternative** to **Apple Intelligence Writing Tools**, developed by the author. This open-source and free tool supports **multiple local Large Language Model (LLM) implementations**, offering a broader range of functionality compared to Apple's version. The creator emphasizes the tool's accessibility and versatility for Windows users interested in AI-assisted writing.
  - **Writing Tools**, a Windows-compatible alternative to Apple Intelligence Writing Tools, supports **multiple local LLM implementations** and offers **system-wide functionality**. It's been featured on [XDA](https://www.xda-developers.com/windows-pc-can-now-deliver-instant-free-writing-help-across-all-apps/) and [Beebom](https://beebom.com/high-schooler-app-brings-apple-inteligence-writing-tools-windows/).
  - The tool can be run with **Ollama**, a local LLM option, by following a simple **4-step process**. Users are advised to choose **Llama 3.1 8B** for systems with **~8GB of RAM or VRAM**.
  - Users expressed interest in **Linux support** and **KoboldCPP compatibility**. The developer confirmed that porting to Linux should be straightforward due to the tool's Python and QT foundation.


**Theme 4. IBM Granite 3.0: Open-Source LLMs with Full Commercial Use**

- **[IBM Granite 3.0 Models](https://huggingface.co/collections/ibm-granite/granite-30-models-66fdb59bbb54785c3512114f)** ([Score: 156, Comments: 43](https://reddit.com//r/LocalLLaMA/comments/1g8i69p/ibm_granite_30_models/)): **IBM** and **Ollama** have partnered to bring **Granite 3.0 models** to the Ollama platform, expanding the range of available AI models. The Granite 3.0 series includes models of various sizes, from **3 billion** to **70 billion** parameters, designed to handle tasks such as text generation, summarization, and question-answering with improved performance and efficiency.
  - The **Granite 3.0 models** currently have a **4096 token context window**, with plans to expand to **128K tokens** in 2024. Users expressed disappointment with the current limit but interest in future improvements.
  - IBM's release of fully open models contrasts with recent criticism of Meta's limited commercialization restrictions. The **Apache 2.0 license** of Granite models, particularly the **2B version**, is seen as valuable for unrestricted use and synthetic data generation.
  - Users compared Granite 3.0 performance to other models, with mixed opinions. Some found it competitive with **Mistral** and **Llama**, while others felt it couldn't beat **Qwen2.5**. The **1B and 3B MoE** (Mixture of Experts) models were noted for fast CPU performance.


## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Research and Techniques**

- **Google Deepmind advances multimodal learning with joint example selection**: A [Google Deepmind paper](https://arxiv.org/html/2406.17711v1) demonstrates how data curation via joint example selection can further accelerate multimodal learning.

- **Microsoft's MInference dramatically speeds up long-context task inference**: [Microsoft's MInference technique](https://arxiv.org/abs/2407.02490) enables inference of up to millions of tokens for long-context tasks while maintaining accuracy, dramatically speeding up supported models.

- **Scaling synthetic data creation using 1 billion web-curated personas**: A [paper on scaling synthetic data creation](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/) leverages diverse perspectives within a large language model to generate data from 1 billion personas curated from web data.

**AI Model Releases and Improvements**

- **OpenAI's o1 model outperforms GPT-4o**: OpenAI researcher Noam Brown [states that the new o1 model beats GPT-4o at math and code](https://www.reddit.com/r/singularity/comments/1g8anp0/openais_noam_brown_says_the_new_o1_model_beats/), and outperforms expert humans at PhD-level questions.

- **Salesforce's "tiny giant" xLAM-1b model surpasses GPT 3.5 in function calling**: Salesforce released xLAM-1b, a 1 billion parameter model that achieves [70% accuracy in function calling, surpassing GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/).

- **Phi-3 Mini (June) with function calling**: Rubra AI released an updated Phi-3 Mini model in June [with function calling capabilities](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/), competitive with Mistral-7b v3 and outperforming the base Phi-3 Mini.

**AI Applications and Implications**

- **Harvard scientists develop AI for cancer diagnosis**: Harvard researchers have [unveiled an AI system with 96% accuracy for cancer diagnosis](https://www.reddit.com/r/singularity/comments/1g8c3mn/96_accuracy_harvard_scientists_unveil/), potentially revolutionizing medical diagnostics.

- **OpenAI's o1 model generates legal briefs**: OpenAI CPO Kevin Weil claims their [o1 model can now write legal briefs](https://www.reddit.com/r/singularity/comments/1g7v0ud/openai_cpo_kevin_weil_says_their_o1_model_can_now/) that previously required $1000/hour associates, potentially disrupting the legal industry.

- **Stuart Russell predicts AI surpassing human capabilities**: AI researcher Stuart Russell [predicts that by the end of this decade, AI may exceed human capabilities in every dimension](https://www.reddit.com/r/singularity/comments/1g89t8u/stuart_russell_says_by_the_end_of_this_decade_ai/), potentially leading to significant changes in employment.

**AI Safety and Ethics Concerns**

- **OpenAI whistleblower testifies to US Senate**: William Saunders, an OpenAI whistleblower, [testified to the US Senate](https://www.reddit.com/r/singularity/comments/1g7zrl1/openai_whistleblower_william_saunders_testifies/) that "No one knows how to ensure that AGI systems will be safe and controlled" and suggests AGI might be built in as little as 3 years.

- **Concerns over AI development pace and safety**: Multiple posts and comments express concern over the rapid pace of AI development and potential safety risks, with some calling for increased regulation and oversight.

**AI Industry Developments**

- **Former OpenAI CTO Mira Murati starting new AI company**: [Mira Murati, who recently left her position as OpenAI CTO, is reportedly raising venture capital funding for a new AI startup](https://www.reddit.com/r/singularity/comments/1g7x6t9/mira_murati_the_openai_cto_who_announced_her/).

- **Increased competition and funding in AI sector**: Several posts and comments discuss the growing number of AI startups and the large amounts of funding being raised in the sector.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1: AI Model Advances and New Releases**

- [**Janus Steps Through Time with Visual Decoupling**](https://x.com/deepseek_ai/status/1847191319464300652): **DeepSeek's Janus** introduces a multimodal LLM with a novel autoregressive framework that decouples visual encoding for enhanced understanding and generation, outperforming models like **LLaVA**.
  
  - Janus's innovative approach surpasses previous models, stirring excitement in the AI community.
- [**Meta's Spirit LM Speaks Up**](https://x.com/AIatMeta/status/1847383580269510670): **Meta** releases **Spirit LM**, an open-source multimodal language model that seamlessly integrates text and speech, demonstrating advanced capabilities in ASR and TTS.
  
  - Discussions focus on its potential applications and how it naturally integrates with existing tools.
- [**Microsoft Claims Big with BitNet**](https://www.reddit.com/r/singularity/comments/1g768xk/microsoft_llm_breakthrough_you_can_now_run_100b/): Microsoft asserts they can run **100B parameter models** on local devices with up to **6x speed improvements** and **82% energy reduction** without a GPU.
  
  - Community skepticism remains due to the lack of available **BitNet models**, awaiting further validation.

**Theme 2: AI Safety and Ethical Concerns**

- [**Deepfakes Stir Social Turmoil**](https://huggingface.co/blog/as-cle-bert/ai-is-turning-nuclear-a-review): Community members express alarm over **deepfake technology**, highlighting severe public repercussions for individuals affected by manipulated content.
  
  - Concerns revolve around victims being wrongly accused and societal backlash inflamed by realistic fake media.
- [**Nous Sounds the Alarm on AI Safety**](https://x.com/NousResearch/status/1848397863547515216): **Nous Research** releases a video and blog post emphasizing critical **AI safety issues**, offering key findings and recommendations regarding **AI practices**.
  
  - These resources stimulate discussions on evolving safety measures in light of AI advancements.
- [**When AI Gets Moralistic**](https://www.reddit.com/r/notebooklm/comments/1g83etl/deep_dives_hosts_break_up_after_she_finds_out_he/): Users notice that AI models interpret prompts through a **moralistic lens**, affecting storytelling and generated content.
  
  - This raises debates about the implications of AI embedding presumptive beliefs about fairness and morality.

**Theme 3: Model Training Challenges and Optimization**

- [**Unsloth Fixes Gradient Bugs, Speeds Up Training**](https://x.com/danielhanchen/status/1848415389266669883): **Unsloth AI** addresses critical **gradient accumulation bugs**, improving loss curve calculations and enhancing reliability in model training.
  
  - Users are advised to update libraries to leverage these improvements for better model performance.
- [**Liger Kernel Tackles Memory Hogs**](https://arxiv.org/pdf/2410.10989): **Liger Kernel** users discuss solutions to **CUDA memory errors** during model training, emphasizing the importance of memory allocation patterns in **Triton** and **Liger** operations.
  
  - Community efforts focus on code reviews for efficient gradient accumulation and addressing potential bugs.
- [**BitNet Shrinks Models to the Bit**](https://github.com/microsoft/BitNet): **Microsoft** unveils **bitnet.cpp**, an inference framework for **1-bit LLMs**, achieving up to **6.17x speedups** and **82% energy reduction** on CPUs.
  
  - Developers are intrigued by the potential to run large models efficiently on CPUs without GPUs.

**Theme 4: AI Agent Frameworks and Applications**

- [**TapeAgents Rewind and Replay Actions**](https://www.youtube.com/live/-yf-e-9FvOc): The **TapeAgents framework** enables **resumable** and **optimizable** agents through a unifying abstraction called Tape.
  
  - Enhances capabilities of tool-using agent architectures, garnering attention in AI development circles.
- [**WorkArena++ Puts Web Agents to the Test**](https://arxiv.org/abs/2410.07041): The launch of **WorkArena++** benchmark challenges web agents in enterprise settings, focusing on autonomous task completion.
  
  - Aims to track agent progress in complex environments, spurring interest within the AI community.
- [**AGI Plays Werewolf, No Full Moon Needed**](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109): The **AGI-Thon Werewolf Agents Tournament** scheduled for **November 9, 2024**, invites AI agents to compete in the game of Werewolf.
  
  - Participants express excitement about testing their agents in a competitive setting with attractive prizes.

**Theme 5: AI in Creative Content Generation**

- [**Podcasting with AI: Talk About Talking**](https://open.spotify.com/show/4Lp134MSzPu7UQDYi0mvuu): Users share success stories of generating engaging podcasts from Reddit comments and Discord chats, showcasing AI's potential in content creation.
  
  - One creator boasts uploading **500 episodes**, demonstrating remarkable efficiency.
- [**NotebookLM Has a Language Turn**](https://myaccount.google.com/language): Participants report **NotebookLM** defaulting to Spanish despite English prompts, pointing to a need for clearer language settings.
  
  - Adjusting **Google account language settings** is suggested to mitigate this issue.
- [**AI Gets Creative in Roleplay**](https://www.reddit.com/r/notebooklm/comments/1g7lf1g/deep_dive_ai_got_you_babe/): Discussions on advanced techniques for erotic roleplay (ERP) with AI models focus on creating detailed character profiles and enhancing immersion.
  
  - Users praise the innovative prompts and express interest in applying techniques to non-erotic creative writing.

---

# PART 1: High level Discord summaries

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HelpingAI2 Demo Launches**: Check out the [HelpingAI2 demo](https://huggingface.co/spaces/Abhaykoul/HelpingAI2.5-prototype) showcasing a new prototype aiming to enhance user interaction with AI assistance.
  
  - This initiative aims to foster improved engagement through advanced AI interaction techniques.
- **Protein Structure Visualization Breakthrough**: A new project on [protein structure prediction](https://huggingface.co/spaces/MISATO-dataset/esm3-conformational-sampling) has been released, integrating noise for enhanced visualization capabilities.
  
  - This tool significantly boosts the ability to visualize intricate protein structures in the field.
- **Advanced Dreambooth LoRA Script Released**: A new advanced **Dreambooth LoRA training script** has been introduced, featuring enhancements for maximum flexibility and control, detailed in [this article](https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora).
  
  - This script invites community feedback to drive continuous improvements.
- **NLP Resources Shared**: A member directed the community to [hf.co/learn](https://hf.co/learn) for excellent NLP learning resources, showcasing an interest in accessible materials for newcomers.
  
  - This exchange indicates a growing demand for practical guides in the NLP field.
- **NozyIO UI for Diffusion Pipelines**: The [NozyIO project](https://github.com/oozzy77/nozyio) has been introduced, allowing users to chain Python functions and visualize outputs, with collaborative discussions on utilizing it for HuggingFace pipelines.
  
  - The support for **Yolo** integration was confirmed, enabling object detection functionalities within NozyIO.

 

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Podcast Generation Success**: Users reported generating engaging podcasts from various sources including Reddit comments and Discord chats, with one creator uploading **500 episodes** as a demonstration of efficiency.
  
  - While results vary, some participants discussed the desire for features that enable longer audio outputs and improved interaction capabilities.
- **Struggles with Language Defaults**: Participants encountered issues with NotebookLM defaulting to **Spanish**, despite their prompts being in **English**, indicating a need for clearer language settings.
  
  - Adjusting **Google account language settings** was suggested to mitigate this challenge.
- **Varying Use Cases of NotebookLM**: Users shared diverse applications of NotebookLM, spanning academic research to podcast creation from user comments, showcasing its versatility.
  
  - One user highlighted the effective generation of podcasts from **Discord** and **Reddit** comments, emphasizing strong outcomes.
- **Optimizing Prompt Engineering for Better Outputs**: The community explored effective strategies for prompting NotebookLM to achieve desired outputs, including generating specific dialogues in podcasts.
  
  - There's a continuous effort to refine prompts for enhanced performance and engagement in resulting content.
- **Ethical Concerns in AI Responses**: Users recognized that NotebookLM may interpret prompts through a **moralistic lens**, affecting storytelling and generated content.
  
  - This raised discussions about the implications of AI models making assumptions based on embedded beliefs about fairness and morality.

 

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Discussions on Open Source Data Requirements**: Members debated the practicality of current **data requirements for Open Source AI** projects, particularly concerns about undisclosed data and replicability of training processes.
  
  - One participant pushed for clear definitions that distinguish model usage from data requirements to enhance understanding.
- **Copyright Laws Stymie AI Training**: The conversation highlighted ongoing **debates on copyright law** and its implications for using copyrighted data in AI model training, especially within the EU.
  
  - Participants pointed out that while the TDM Exception in the EU supports technology advancement, clarity about its application is still lacking.
- **RWKV-7 Sets New Training Speed Record**: RWKV-7, an attention-free model, reportedly outpaces modified GPT models, achieving significant training speed improvements.
  
  - Recent optimizations have led to better validation loss and training times, indicating ongoing progress in model efficiency.
- **Evaluating Dynamic Loss Scaling in Pythia**: Members noted that **Pythia** models can skip weight updates during FP16 runs when encountering NaN or Inf gradients, a feature not present in BF16 runs.
  
  - The discussion highlighted that FP16 training can continue under certain error conditions, unlike BF16 which halts the process entirely.
- **Integrating Eval Harness with Custom Models**: The community focused on how to effectively integrate the **eval harness** with custom models, underscoring limitations in various PyTorch repositories.
  
  - Key suggestions included using `TemplateLM` as a subclass to navigate API complexities better and enhance task handling.

 

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI Lecture Unpacked**: The much-anticipated lecture by Daniel Han on GPU mode is now accessible, featuring insights into **LLM systems engineering** and **gradient accumulation fixes**.
  
  - It includes practical Q&A sessions, enhancing comprehension for developers aiming to optimize AI models.
- **Fix Released for Gradient Accumulation Bugs**: A critical fix was implemented for the **gradient accumulation** bug affecting Unsloth trainers, improving loss curve calculations.
  
  - Users are advised to update their libraries to leverage this fix for better model training reliability.
- **Navigating Training Issues with New Datasets**: Discussions emphasize the necessity for diverse datasets while addressing difficulties in fine-tuning models on fresh formats, particularly with multiple target predictions.
  
  - Participants shared suggestions around synthetic data generation to counteract model relevance issues.
- **Mistral Innovations on ReAct Agent Tooling**: A member reported on the development of a dataset focused on **ReAct agent tool calling** amidst concerns regarding **Mistral's Agentic model** overshadowing earlier efforts.
  
  - The new **Ministrial 8b** model raises questions about the relevance of continuing with existing datasets.
- **LayerSkip Boosts Inference Efficiency**: Insights on **LayerSkip** reveal it enhances LLM inference speed by employing layer dropout and early exit loss strategies.
  
  - It's shown to improve performance in summarization and coding tasks substantially, with GitHub access provided for detailed implementation.

 

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous focuses on AI safety**: Nous Research released a video and a blog post on **safety issues** in AI, emphasizing key findings and recommendations regarding **AI practices**. You can watch the video [here](https://x.com/NousResearch/status/1848397863547515216) and read the blog post for a thorough analysis.
  
  - These resources are part of a broader discussion on how AI safety measures need to evolve in light of recent advancements and challenges in the field.
- **Deepfake tech raises concerns**: Members discussed the dangers of **deepfakes**, particularly how they can lead to severe public repercussions for affected individuals. This mirrors concerns regarding recognition of authenticity in content and the societal backlash against victims.
  
  - The community highlighted the need for greater public awareness and protective measures against such manipulative technologies.
- **MarketAgents Project gets traction**: The **MarketAgents** project, focusing on multi-agent market simulations, has garnered attention, particularly due to contributions from **Blacklight**. More details can be found in the [project repository](https://github.com/marketagents-ai/MarketAgents).
  
  - Discussion emphasized the project's collaborative nature and its potential implications for market simulations, with members eager for updates on its progress.
- **Advancements in Model Efficiency**: The conversation centered around **quantization aware training (QAT)** for improving models like Llama 3.1-8B, while discussing trade-offs associated with model capacity. Techniques to mitigate performance loss through pruning attention layers were suggested.
  
  - Moreover, developments in **optimizers** like AdamW highlight new approaches for enhancing training efficiency without the burden of hyper-parameter tuning.
- **Hermes AI Model Accessibility**: Free access to the **Hermes AI Model** is now available at [ai.unturf.com](https://ai.unturf.com), stemming from the [NousResearch/Hermes-3-Llama-3.1-8B](https://nousresearch.com/hermes3/) architecture. The platform encourages open-source contributions and provides installation guides.
  
  - Participants expressed interest in leveraging Hermes for custom applications, particularly in voice integrations.

 

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O1 Preview excels in code generation**: Users report that **O1 Preview** generates complex code in languages like **Swift** and **C#**, such as creating a 'StrawberryStreamer' system with network functionalities.
  
  - Despite some initial mistakes, it learns from feedback, becoming particularly useful for intricate programming tasks.
- **ChatGPT saves too much unimportant info**: Users are frustrated with **ChatGPT** saving trivial details despite instructions to ignore them, leading to memory cleanups.
  
  - Custom instructions may enhance memory management, suggesting a need for better user control.
- **Activating GPT-4o features**: It's explained that custom GPTs automatically utilize **GPT-4o**, with no option to use a different model.
  
  - Users were informed about managing files and generating outputs through custom GPTs.
- **Strategies for effective AI prompts**: To maximize AI performance, it's suggested to use fewer, common words and provide clear instructions in quotes at the prompt's start.
  
  - Effective examples indicate that specifying writing surfaces can improve output quality.
- **Creating realistic AI interactions**: To achieve more human-like interactions with AI, it's crucial to communicate casually and provide detailed character backstories.
  
  - The model mirrors user language, with friendly phrasing and expectations significantly enhancing realism.

 

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Limitations Confusion**: Users report losing **focus options** after upgrading to **Enterprise Pro**, resulting in fewer sources and responses, impacting functionality.
  
  - This has sparked conversation about ways to retrieve more comprehensive results, as many feel the service has regressed.
- **Diverse User Experiences with Perplexity**: While some users enjoy Perplexity’s features for research and code without heavy searching, others encountered **internal server errors** and API access problems.
  
  - The divergence in user experience raises concerns about overall service reliability and quality.
- **Debate on AI Models Performance**: Discussions on various AI models like **Claude 3.5 Sonnet** and **GPT-4O** highlight a competitive landscape, with users evaluating their performance for different tasks.
  
  - This indicates a broader interest in understanding which tool suits specific needs amidst rising options.
- **YouTube Tackles AI Content Identification**: YouTube has introduced a feature aimed at identifying **AI-generated content**, a move towards improved transparency in digital media.
  
  - This aligns with growing user demands for authenticity, particularly relevant in the evolving landscape of content creation.
- **API Credits Transfer Issues**: A user expressed concern over **API credits** not transferring post-Pro subscription purchase, raising critical issues about user support.
  
  - Prompt suggestions to contact support reflect the community's emphasis on resolving operational hiccups efficiently.

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo rises as C++ alternative**: Members explored how **Mojo** is being developed as a general-purpose systems programming language, currently mirroring **C++** while evolving towards **Python**'s abstraction level.
  
  - One member pointed to the [Carbon programming language project](https://github.com/carbon-language/carbon-lang) for insights into object-oriented programming implementation.
- **Flexibility in Mojo vs Carbon**: Discussion highlighted Mojo's greater flexibility with pointers compared to the **Carbon programming language**, restricted by C++ compatibility.
  
  - Members noted the technical differences when handling references and pointers, indicating potential advantages for **Mojo**.
- **Compile Time Tuple Lengths in Mojo**: Users found that **Mojo** supports retrieving compile-time lengths of tuples via `__type_of(t).__len__()`, enhancing dynamic coding capabilities.
  
  - This method allows developers to avoid runtime checks, improving overall code efficiency and reliability.
- **Inquiry on Graph Training Support**: A member solicited information on timelines for **Graph training support**, emphasizing the need to update values in compiled Max Graphs beyond GPU focus.
  
  - *Thx* was expressed for any clarifications, underscoring community interest in broader functionalities.
- **C-API for MAX-Graph Models**: Members inquired about the feasibility of utilizing **C-API** to execute models from the **MAX-Graph API**, exported through **export_compiled_model**.
  
  - This raised concerns over gaps in current tools for users preferring not to rely on frameworks like **ONNX** or **Torch**.

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek Janus Launch**: DeepSeek introduced [Janus](https://x.com/deepseek_ai/status/1847191319464300652), a multimodal LLM utilizing a novel autoregressive framework that decouples visual encoding for better understanding and generation, surpassing previous models.
  
  - Comparisons with models like **Llava** indicated Janus’s enhanced capabilities in both image generation and comprehension.
- **Meta's New Spirit LM**: Meta launched [Spirit LM](https://x.com/AIatMeta/status/1847383580269510670), an open-source multimodal language model that seamlessly integrates text and speech, demonstrating advanced capabilities across ASR and TTS.
  
  - Discussions centered on its application potential and early reception within the AI community, emphasizing natural integrations with existing tools.
- **Challenges with Microsoft Copilot Agents**: Users reported frustrations with **Microsoft Copilot**, citing performance issues, misunderstandings of specialized knowledge, and problems with text formatting during restructuring.
  
  - The gap between marketed capabilities and actual performance, especially in enterprise applications, was notably criticized.
- **Singapore's AI Engineer Nation initiative**: Minister Josephine Teo discussed the future of AI policy in Singapore, focusing on how **AI can be adopted in government for public good** during a [recent conversation](https://x.com/swyx/status/1847732308889260072).
  
  - She addressed **Sovereign AI** approaches and their implications for **elections**, sharing insights on governance and technology integration.
- **AST vs DSL: When to Use Each**: The community engaged in a discussion regarding the use of **ASTs** versus **DSLs**, exploring their roles as alternative communication styles for coding.
  
  - Participants debated optimal scenarios for each in code refactoring tasks, emphasizing their distinct benefits.

 

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Granite 8B matched against Qwen 2.5 7B**: Users are actively comparing **Granite 8B** and **Qwen 2.5 7B** for coding and scientific tasks, focusing on performance benchmarks.
  
  - The [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) was recommended as a resource for performance comparisons.
- **Image recognition woes with Llava**: Several users reported that the **Llava model** struggles to recognize images, leading to inaccurate responses.
  
  - To mitigate this, they advised using **jpeg or png** formats and starting with a **clean chat**.
- **Xeon E5-2603 v4 processors limited to 6 threads**: In discussions about a bug with dual **Xeon E5-2603 v4** processors, only **6 threads** are utilized in **version 0.3.4**, down from 8 in 0.2.31.
  
  - One member indicated that this is *a known issue* and confirmed their findings were added to an existing bug report.
- **RX 7900 XTX outshines ROCm**: A user observed that the **RX 7900 XTX** performs about **10-15% better** with **Vulkan** compared to **ROCm** during inference tests.
  
  - Another user suggested rolling back to **ROCm 1.10** due to existing complications with the latest runtime.
- **Opinions clash on M4 Ultra's AI capabilities**: Debate arose regarding the **M4 Ultra chip** in upcoming MacBooks and its effectiveness for AI tasks, with some skepticism expressed.
  
  - Users noted potential limitations, suggesting that its **expensive** and **non-upgradable** design could hinder broader applications in AI.

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Inflection’s Payment Processor Faces Downtime**: **Inflection 3 Pi** and **Inflection 3 Productivity** models are down due to payment processing issues, impacting user access significantly.
  
  - Users await further updates on when these models will be restored to full functionality.
- **Grok 2 Gets a Rebranding Amidst Pricing Increase**: The model previously known as **Grok 2** has officially been renamed to **Grok Beta**, with pricing now set at **$15/M** for completions.
  
  - This rebranding reflects its interim developmental status while users have reported fluctuations in service availability.
- **Hermes 3 Users Hit with Rate Limiters**: Frequent **429 errors** have plagued users of the **Hermes 3** model, causing dissatisfaction as it appears to restrict usage more than before.
  
  - Users note that these constraints were less common previously, prompting discussions on potential model adjustments.
- **Billing System Chaos in OpenRouter**: Users report unexpected charges from the **OpenRouter billing system**, even when there are existing credits, leading to confusion.
  
  - Many shared similar experiences, indicating a need for better support mechanisms for resolving billing discrepancies.
- **AI Summarizer Struggles with Vercel Timeouts**: An AI-powered text summarizer based on **Gemma 2 27B** is facing **FUNCTION TIMEOUT** errors on Vercel’s hobby plan after **10 seconds**.
  
  - Proposals include increasing function timeout limits or exploring **streaming responses** to bypass these limitations.

 

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Mastering Durable Execution Concepts**: Members discussed **durable execution**, an abstraction ideal for long-running workflows, illustrated by [Temporal background checks](https://learn.temporal.io/examples/go/background-checks/). This approach allows code to operate unconstrained by time and space.
  
  - Such insights led to practical applications and sparked interest in integrating similar frameworks for efficient workflow management.
- **Navigating Mistral API with Aider**: Instructions on using the **Mistral API** with Aider were provided, showing how to specify the model via command line and configure it in a `.aider.conf.yml` file.
  
  - Community discussions emphasized the importance of precise model selection for effective AI-driven coding sessions.
- **CEDARScript Takes Charge of Low-Level Syntax**: Discussion focused on **CEDARScript**, which offloads syntax issues from LLMs, allowing them to concentrate on high-level abstractions, showing compatibility with various programming languages.
  
  - Explorations into its integration with Aider promise more robust code editing capabilities in the future.
- **Microsoft Launches bitnet.cpp for 1-bit LLMs**: Microsoft released [bitnet.cpp](https://github.com/microsoft/BitNet), an inference framework for **1-bit LLMs**, including the BitNet b1.58 model which optimizes CPU performance.
  
  - It achieves speedups of **1.37x to 5.07x** on ARM CPUs and **2.37x to 6.17x** on x86 CPUs, significantly reducing energy consumption, an enticing prospect for developers working on large-scale models.

 

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **TensorRT-LLM Enhances Efficient Inference**: A user shared important resources on **TensorRT-LLM**, emphasizing the [cutlass int8 gemm kernel](https://github.com/NVIDIA/TensorRT-LLM/blob/a65dba7aaf7e2d8bb0120eea8f8f04deff145d6a/cpp/tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm_template.h#L62-L63) for optimized performance in Large Language Models (LLMs).
  
  - This resource aims to offer a Python API that significantly improves **efficient inference**, crucial for high-performance model execution.
- **Upcoming Unsloth Presentation Highlights**: An upcoming talk centered on **Unsloth**, an essential resource for systems engineering and Triton kernels, has been announced, with links shared for further materials including [slides](https://docs.google.com/presentation/d/1BvgbDwvOY6Uy6jMuNXrmrz_6Km_CBW0f2espqeQaWfc/edit?usp=sharing).
  
  - Participants are expected to gain insights into *Triton and CUDA techniques*, enhancing their technical arsenal.
- **CUDA Memory Management Concerns on Apple Silicon**: There are ongoing discussions regarding memory management when using unified memory on Apple Silicon with PyTorch, particularly whether tensors allocate in private mode by default.
  
  - Concerns were raised about potential issues when leveraging custom buffers with **at::from_blob()**, indicating a need for clarity in documentation.
- **Gradient Accumulation Bug in Liger Kernel**: A critical inquiry into a **gradient accumulation bug** fix in transformers raised questions about its applicability to **Liger Kernel's** cross entropy operations.
  
  - This indicates the community's focus on ensuring clarity regarding potential issues with Liger Kernel functionalities.
- **Memory Errors Related to Triton and Liger**: Memory allocation issues were reported, specifically **cuda out of memory errors** with Liger when utilizing PyTorch's **torch compile**.
  
  - This underlines a pressing need to explore specific memory patterns associated with Triton and Liger operations.

 

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Quest for Human Data Labelers**: A member sought recommendations for human data labelers for **weather radar data**, emphasizing the need for **geospatial** and **vision language labeling**.
  
  - Discussion revolved around various platforms, including **Scale AI**, **Surge**, **Mechanical Turk**, and **Prolific**, with an analysis of their pros and cons for different data types.
- **Progress on RLHF Book**: Nato announced he is developing a book on *reinforcement learning from human feedback (RLHF)*, targeting a physical release by the year's end.
  
  - He encouraged community engagement through the [book's website](https://rlhfbook.com/) while emphasizing his writing process without extensive checks.
- **LLM Reasoning Debate Heats Up**: The community engaged in a debate on whether **LLMs**, particularly **GPT-4o** and **GPT-o1**, effectively reason or just replicate training patterns.
  
  - This discussion was fueled by the launch of the two models in May 2024, raising concerns about their genuine problem-solving capabilities.
- **Interconnects Emojis Making Waves**: Members chatted about adding **Interconnects emojis** to the server, proposing suggestions for **AI company logos** and meme ideas.
  
  - Humorous exchanges ensued regarding emoji settings and potential support from Discord staff, with aesthetic improvements discussed for dark mode compatibility.
- **OpenAI Releases GPT-4o and GPT-o1**: OpenAI launched **GPT-4o**, promising real-time reasoning across audio, vision, and text, followed by the **GPT-o1** for benchmarks heavy on reasoning.
  
  - This development has intensified discussions about AI's reasoning capabilities versus learned behavior from given training data.

 

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **RTX 3090 Disappoints**: A user reported their **RTX 3090** achieving only **3.5 iterations per second**, down from the **RTX 3060**. Suggested fixes included updating the web UI and reinstalling drivers.
  
  - This unexpected performance drop raised eyebrows, sparking discussions about optimizing setups to match prior results.
- **Struggles with Image Perspectives**: One user faced difficulties creating different perspectives of a building while retaining color integrity in new sketches. Community suggestions included leveraging more drone shots and training a **Lora** specifically on the architecture.
  
  - This debate on techniques highlighted the limitations of existing photo datasets in achieving realistic transformations.
- **Lora Confusion During Image Generation**: Users encountered errors involving multiple **Loras** not being found in image generations, which generated troubleshooting discussions. Members offered insights on how to manage prompts to avoid such conflicts.
  
  - This issue emphasized the need for better prompt management strategies to maximize Lora utility.
- **Accessing Stability.ai API Troubles**: Concerns arose about the **Stability.ai API reference page** being down, with users suggesting contacting customer service for resolution. The community clarified this issue was out of their control.
  
  - This led to discussions on potential temporary workarounds for those needing API access while waiting for official support.
- **Seeking Help with AI Image Editing**: Users expressed a need for assistance in integrating AI tools for image editing in commercial projects. Collaborative offers for help were made, showcasing a supportive atmosphere within the community.
  
  - This desire for collaboration indicates a growing interest in refining workflows involving AI technologies.

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **3-Day Hackathon Delivers 45 Projects**: The recent **3-day hackathon** attracted over **500 participants**, culminating in **45 projects** showcased at the end. Check out the [blog post announcing the winners](https://t.co/v7F8b0qedF) for more details.
  
  - Exciting guest blog posts from winners will provide deeper insights into their projects.
- **LlamaParse Premium Receives Praise**: Users are thrilled with **LlamaParse Premium**, reporting significant improvements in parsing capabilities. An insightful [LinkedIn post](https://t.co/NeAvIlfIP3) reviews its advantages over earlier versions.
  
  - For further context, the original introduction of **LlamaParse** can be found [here](https://t.co/pDPHxcYQeb).
- **Integrating Ollama in LlamaIndex**: A configuration attempt to use **Ollama** with `npx create-llama` faced an OpenAPI key pop-up, even with correct settings. It was suggested to edit the backend source code to resolve loading issues with **Ollama** LLM.
  
  - This insight could help others encountering similar integration hassles.
- **Evaluating Hybrid Retrieval Accuracy**: The community debated methodologies to evaluate a hybrid retriever combining `BM25Retriever` and `VectorIndexRetriever`, emphasizing the necessity of ground truth datasets. Leveraging an LLM to evaluate relevance came up as a promising method.
  
  - Tracking question-document mappings also emerged as a viable evaluation approach.
- **Searching for Multilingual Embedding Solutions**: One member is exploring a **RAG system** that navigates multilingual PDFs, but hasn't had much success with current embedding models. They received recommendations for the **aBSE** model as a potentially effective solution.
  
  - This model focuses on language-agnostic implementations, which could enhance multilingual performance.

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Multihead Attention Relevance**: In the Tinygrad community, a member questioned the ongoing relevance of discussions regarding **standardizing Multihead Attention**, indicating a focus on optimization efforts.
  
  - This highlights the community's continued interest in refining attention mechanisms within the framework.
- **Tinygrad Competes with GGUF Support**: George Hotz proclaimed the addition of **GGUF loading support** to enhance Tinygrad's competitiveness for **running local LLMs** effectively against rivals like **Ollama**.
  
  - He encouraged developers to contribute, aiming to boost Tinygrad's performance and features.
- **Insights into Local LLM Tools**: Users discussed preferences for **Llama.cpp** and **ExLlamaV2** for local model execution, with ExLlamaV2 offering simpler setup options compared to **TensorRT-LLM**.
  
  - The consensus indicates a shift towards these tools for better efficiency in deploying models.
- **Emphasizing WebGPU Support**: George Hotz stressed the importance of **WebGPU support**, detailing community efforts to enhance Tinygrad’s compatibility with this technology.
  
  - Progress on implementing **threefry** algorithms was noted, indicating a reduction in development blockers.
- **Clarifying FrozenBatchNorm2d Functions**: A user sought clarity on the role of **FrozenBatchNorm2d** in network architectures, expressing confusion about its necessity and the function's mechanics.
  
  - This discussion sheds light on the complexities users face when integrating specific components.

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Mystery Model Ignites Curiosity**: A member sparked interest by mentioning a **mystery model** with an **8k** context available, leading to excitement in the community.
  
  - Community members are eager to engage with the [mystery bot](https://discord.com/channels/954421988141711382/996880279224451154/1297180553401077771) for more updates.
- **Join Developer Office Hours Tomorrow!**: Cohere schedules **Developer Office Hours** for tomorrow at **1:00 PM ET**, featuring live demos on new releases.
  
  - Participants can join the discussion via the [Cohere Developer Event](https://discord.com/events/954421988141711382/1285304800400904344/1297967638118400000).
- **OpenRouter Provides API Flexibility**: Members discussed **OpenRouter**, highlighting its seamless API switching capability when facing downtime.
  
  - *TBH, not all API providers are stable*, emphasizing the need for this robust feature.
- **JavaScript Shines in Implementations**: A member showcased a project using **JavaScript**, generating excitement about its effectiveness in AI applications.
  
  - The enthusiasm reflects a noticeable shift towards leveraging **JavaScript** for AI functionalities.
- **Direct API Requests Simplified**: A member confirmed that using just an **API key**, developers can make direct requests to the AI provider without relying on a proxy.
  
  - This approach reduces dependencies and simplifies integration for developers.

 

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Smooth Sailing with Liger Kernel Install**: Users find that to achieve **VRAM savings**, installing the **Liger Kernel** is as easy as `pip install liger-kernel`, adjusting the provided config for optimal setup.
  
  - This kernel enhances full finetuning capabilities leveraging existing **Flash Attention**, making it a smart move for performance.
- **Axolotl Layer Freezing Bug Stirs Concerns**: Community members reported a **bug** in **Axolotl** preventing layer freezing/unfreezing, an essential feature that previously worked seamlessly.
  
  - Investigations are ongoing, with members tasked to confirm changes in the `src/axolotl/integrations/spectrum/model_snr_results` directory for further insights.
- **Spectrum Confirms Solid SNR Results**: A dialogue emerged on the correct computation of **SNR results** for Qwen models, with confirmations that everything is aligned.
  
  - Members noted that **Spectrum** integration necessitates **precomputed SNR JSON files** to operate correctly.
- **Qwen2 DoRA Support Request Gains Attention**: A member seeks any strides in developing **Qwen2** support for **DoRA/QDoRA**, citing minimal activity in related discussions.
  
  - They pointed to [**Answer.AI's QDoRA repository**](https://github.com/AnswerDotAI/fsdp_qlora/tree/main?tab=readme-ov-file#add-support-for-a-new-model) as a foundational resource for potential implementation.
- **Fine-Tuning LLMs for Domain-Specific Data**: A member shares their journey in **training and finetuning LLMs** to cater to **domain-specific data** like **math**, **legal**, and **finance**.
  
  - They advocate for the advantages of starting with **llama-70b-instruct** over non-instruct models for enhanced training outcomes.

 

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Meta's FAIR Team pushes for Advanced Machine Intelligence**: Meta’s FAIR team shares their goal of achieving **advanced machine intelligence (AMI)** to enhance productivity and innovation as highlighted in Mark Zuckerberg's [open letter](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/). Their commitment reflects over a decade of collaboration with the AI community towards **open science**.
  
  - This research effort coincides with discussions around whether tools like **Lingua** are comparable to **Torchtune**.
- **Attention Mask Construction and Flex Attention**: Members discussed complexities in **mask construction** for attention mechanisms, particularly the need for varied block masks based on attention types. Suggestions were made to materialize masks during the forward pass to simplify the **collate** process.
  
  - This underscores the necessity of a clean implementation while managing issues with **packed datasets** and the need for custom collates.
- **Performance Warnings in PyTorch**: Users are facing warnings related to **cuDNN SDPA** on certain data types raising concerns about underlying performance and potential solutions. Testing with different kernels may clarify the performance impact, connecting to reported issues on the [PyTorch GitHub](https://github.com/pytorch/pytorch).
  
  - Participants are considering filing an issue on **PyTorch core** to address the persistent warnings and implications.
- **Countdown to v0.4.0 code freeze starts!**: With only **8 days** left until the **v0.4.0 code freeze** on **October 29th**, developers are gearing up to finalize pending tasks. Preparation is key as the [*v0.4.0 Tracker*](https://github.com/pytorch/torchtune/issues/1747) projects a release date of **November 5th**.
  
  - Contributors are actively strategizing to ensure the release is packed with exciting updates.
- **New features lined up for v0.4.0**: Upcoming features in **v0.4.0** were discussed, referencing issues **#1645**, **#1847**, and **#1835**. Contributors are diligently working to ensure new functionalities enhance user experience.
  
  - The preparations for this release reflect a strong collaborative effort within the development team.

 

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Pydantic All-in-One Live Stream**: A member initiated a live stream on [pydantic-all-in-one](https://discord.com/channels/1161519468141355160/1161519469777133580), detailing their process for developing Python packages and frameworks.
  
  - They plan to build **llmodel** following the stream, addressing community needs.
- **Tutorial Discussion for DSPy GPTs**: Members explored creating a tutorial video on using various **DSPy GPTs**, beneficial for both new and experienced users.
  
  - Community support is strong, with the creator agreeing to consider the proposal for a comprehensive guide.
- **AI Agents in Production Event Announcement**: A virtual event is scheduled for **November 13**, featuring notable speakers like Tomas Wolf and Nathan Benaich to discuss deploying AI agents in production.
  
  - Organized by **Prosus AI and MLOps**, the event promises to address real-world applications and challenges in memory management.
- **Step-by-step LightRAG Tutorial with Ollama**: A YouTuber shared a detailed [tutorial](https://www.youtube.com/watch?v=g21royNJ4fw&t=10s) for setting up and running **LightRAG** with **Ollama**.
  
  - The tutorial emphasizes the integration of knowledge graphs with embedding-based retrieval, enhancing system functionality.
- **Clarification on AcgNDCG and Document Retrieval**: A question arose about whether documents are retrieved from a limited set of **10ish Relevance Judgements** or a broader pool, with the paper linked [here](https://arxiv.org/pdf/2406.11706).
  
  - *Does it retrieve from a specific list or the entire pool?* remains an open query needing resolution.

 

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Today's LLM Agents Lecture at 3 PM PST**: The **7th lecture** in the **LLM Agents** series takes place today at **3:00pm PST** and can be streamed [here](https://www.youtube.com/live/-yf-e-9FvOc). Guest speakers **Nicolas Chapados** and **Alexandre Drouin** will discuss **AI Agents for Enterprise Workflows** during the session.
  
  - Members are looking forward to insights on **orchestration of agents** and further advancements in the **Agentic System**.
- **Introduction of TapeAgents Framework**: The lecture will introduce the **TapeAgents framework**, enabling **resumable** and **optimizable** agents through a unifying abstraction known as the Tape. This initiative could enhance the capabilities of tool-using agent architectures significantly.
  
  - Participants are excited to learn how this framework can advance their projects in AI agent development.
- **WorkArena++ Benchmark for Web Agents**: **WorkArena++** is a newly launched benchmark evaluating web agents in enterprise settings, focusing on autonomous task completion. It poses new challenges for the field and tracks web agents' progress in complex environments.
  
  - There is a keen interest from participants about how this benchmark can inform the development of future agent-based models.
- **Course Completion Certificate Details**: Students will receive a certificate upon completing all course requirements, including quizzes and a written article assignment, due by **December 12**. The course staff assured access to **recordings and slides** for catch-up.
  
  - The assignment will involve summarizing lecture content or hackathon experiences, prompting discussions around project work and understanding concepts.
- **Running LLMs Locally with Practical Tools**: Participants were given options for running LLMs locally, with **Ollama** and **LM Studio 0.3.0** recommended as practical tools. Users must be aware that larger models generally require more than **8GB of RAM**.
  
  - Discussions emphasized the importance of efficient resource management when working with local LLM setups.

 

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LibreFLUX releases with new capabilities**: The launch of **LibreFLUX**, an Apache 2.0 version of [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), introduces a full T5 context length, enhanced attention masking, and restored classifier-free guidance.
  
  - *Community reactions were positive*, acknowledging the extension of **open-source tenets** and excitement for the early 2000s aesthetic of the new model.
- **Challenges in training Open-MUSE**: Users reported difficulties with finding models like **openMUSE/maskgit-vqgan-imagenet-f16-256** on Hugging Face and encountered a missing key error in their training configuration file.
  
  - For more info, they shared [the configuration YAML](https://wandb.ai/psuraj/muse/runs/3ef2rhq3/files/config.yaml) for community assistance.
- **Microsoft's LLM performance leap**: Microsoft claims it can now run **100B parameter models** on local devices, achieving up to **6x speed improvements** and **82% energy reduction** without a GPU, as stated in a [Reddit post](https://www.reddit.com/r/singularity/comments/1g768xk/microsoft_llm_breakthrough_you_can_now_run_100b/).
  
  - This assertion was further elaborated upon in a tweet, prompting debate over the feasibility of such performance levels [here](https://x.com/jenzhuscott/status/1847514413060046855).
- **No BitNet Models Available Yet**: Despite the excitement around Microsoft's claims, users noted that no **100B models** utilizing **BitNet** exist, raising skepticism about the actual performance capabilities.
  
  - The community is cautious and seeks further validation before accepting these efficiency claims.
- **MUSE Project opens reproduction efforts**: Discussions centered around the open reproduction of the **MUSE** model for text-to-image generation, with resources provided like the [GitHub repository](https://github.com/huggingface/muse) and [W&B Project](https://wandb.ai/psuraj/muse?workspace=user-).
  
  - Key activities involve training various models on datasets like **imagenet** and conducting experiments on **CC12M** to enhance transparency in the process.

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Aider enhances AI-generated code**: **Aider** progressively integrates AI-generated code, indicating a trend towards dynamic nightly builds of its interpreter concepts.
  
  - This raised curiosity about potential similar implementations from **Open Interpreter**.
- **Open Interpreter's Custom Tools Question**: Users inquired about a potential **equivalent to the /functions** folder for easy access to custom functions in **Open Interpreter**.
  
  - Current options seem limited, with suggestions to modify the repository for adding custom tools.
- **Mac Setup Works but Issues Arise**: A user reported successful **OpenInterpreter** setup on Mac, with [localhost:10100](http://localhost:10100) functioning as expected.
  
  - However, they faced interaction issues, including web browser access denials and problems with the **LiveKit Meet link**.
- **Voice Assistant Boosts Functionality**: [AIwithBenefits](https://x.com/AIwithBenefits/status/1848161437828415578) highlighted adding a **HumeAI voice assistant** to the **phidatahq** agent, aiming to improve usability through AppleScript execution.
  
  - Praise was directed towards the revamped **phidatahq UI**, enhancing overall interaction with native apps.

 

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangGraph Code Assistant Tutorial Revealed**: The **LangGraph Code Assistant** tutorial instructs users on building iterative answers to coding challenges via [AlphaCodium](https://github.com/Codium-ai/AlphaCodium) and RAG methods.
  
  - *Ingest user-specified documentation and invoke tools for structured output*, while conducting unit tests to validate returned solutions.
- **Role-based RAG Models Under Discussion**: A discussion emerged about implementing **RAG models** tailored to user roles, particularly optimizing access for CEOs while restricting interns to relevant documents.
  
  - This approach sparks significant questions on effective management and access restrictions within the **RAG frameworks**.
- **Techstars Startup Weekend SF is Here**: The **Techstars Startup Weekend SF** invites attendees to the [AWS GenAI Loft](https://aws.amazon.com/startups/lp/aws-gen-ai-loft-san-francisco?lang=en-US) for an exclusive networking event following TechCrunch Disrupt.
  
  - Industry experts will present insights, fostering connections among founders, investors, and innovators in the tech community.
- **In-depth Comparisons Between OpenAI Swarm and LangChain LangGraph**: An article provided a detailed comparison of **OpenAI Swarm** and **LangChain LangGraph**, pinpointing their functionalities and suitable use cases for crafting complex AI workflows.
  
  - This guide aims to help developers navigate their choices for optimal project fit, accessible [here](https://medium.com/ai-artistry/openai-swarm-vs-langchain-langgraph-a-detailed-look-at-multi-agent-frameworks-0f978a4ca203?sk=06fad63e6089bc2d0e772b2101b4f474).
- **The Rise of Multi-Agent Workflows**: The importance of developing **multi-agent workflows** in AI keeps growing, essential for managing complex interactions and enhancing capability.
  
  - *Such frameworks allow developers to effectively streamline processes,* improving overall AI performance.

 

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AGI-Thon Tournament Kicks Off**: The upcoming **AGI-Thon Werewolf Agents Tournament** is scheduled for **November 9, 2024** and details can be found on the [AGI House events page](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109).
  
  - This event promises exciting competitions for AI agents, attracting participants from diverse backgrounds to showcase their skills.
- **Upcoming Tournament Sparks Interest**: The announcement of the **AGI-Thon** has sparked discussions among AI enthusiasts eager to join the competition.
  
  - Many participants expressed excitement about the opportunity to test their agents in a competitive setting.

 

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla examines AI access issues**: Mozilla has commissioned two reports focusing on **AI access challenges** and competition, specifically [External Researcher Access to Closed Foundation Models](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/External-researcher-access-to-closed-foundation-models.pdf) and [Stopping Big Tech From Becoming Big AI](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/Stopping-Big-Tech-from-Becoming-Big-AI.pdf). These documents, provided by **AWO** and the **Open Markets Institute**, dissect the control dynamics within AI.
  
  - The reports underscore the necessity for **external researchers** to access closed models to foster broader innovation and underline critical reforms needed to achieve a fair ecological balance in AI development.
- **Control in AI Development Explored**: The findings analyze **who's in control** of AI development, advocating for reforms to ensure an equitable landscape. Ensuring a level playing field is key for sustaining **innovation** in the swiftly changing AI terrain.
  
  - The emphasis on access for external researchers aims to reshape the current state of AI governance and allow for competitive versatility changes.
- **Blog Recap of Mozilla's AI Research**: A detailed [blog post](https://discord.com/channels/1089876418936180786/1298015953463808102) provides insights into the outcomes of Mozilla's commissioned research. It addresses the implications of the findings against the backdrop of **current AI governance** practices.
  
  - This resource serves as a critical summary of the reports, highlighting the effects of findings on the stability of AI ecosystems.

 

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Inquiry About Q-Galora**: One member asked, 'has anyone tried **q-galora**?', reflecting curiosity around its functionalities and applications in AI models.
  
  - No responses followed, leaving the community in suspense about potential insights or experiences regarding **q-galora**.
- **Hoping for Insights on Q-Galora**: The community anticipates shared experiences as one member inquired about usage of **q-galora** with a simple question.
  
  - Members are eager for responses that could clarify its capabilities in AI-related projects.

 

---

The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.

---

# PART 2: Detailed by-Channel summaries and links

{% if medium == 'web' %}

 

### **HuggingFace ▷ #**[**announcements**](https://discord.com/channels/879548962464493619/897387888663232554/1297638418465034291) (1 messages):

> - `HelpingAI2 Demo`
> - `Protein Structure Prediction`
> - `AI in Nuclear Research`
> - `WorldMedQA-V Release`
> - `Books Mixer AI`

- **HelpingAI2 Prototype Demo Launched**: Check out the [HelpingAI2 demo](https://huggingface.co/spaces/Abhaykoul/HelpingAI2.5-prototype) showcasing a new prototype by a community member!
  
  - This initiative aims to enhance user interaction with AI assistance.
- **Protein Structure Visualization Advances**: A new project on [protein structure prediction](https://huggingface.co/spaces/MISATO-dataset/esm3-conformational-sampling) has been released, integrating noise and MD frames.
  
  - This tool provides enhanced capabilities for visualizing complex protein structures.
- **AI Turns Toward Nuclear Research**: An insightful [review](https://huggingface.co/blog/as-cle-bert/ai-is-turning-nuclear-a-review) discusses AI's implications in nuclear domains.
  
  - This exploration sheds light on innovative applications and safety considerations in nuclear research.
- **WorldMedQA-V Set for Healthcare Benchmarking**: The release of [WorldMedQA-V](https://huggingface.co/datasets/WorldMedQA/V) provides a multilingual, multimodal dataset to benchmark vision-language models in healthcare.
  
  - This dataset aims to enhance the development of AI tools in the medical field.
- **Creative Storytelling with Books Mixer AI**: The [books-mixer-ai](https://huggingface.co/spaces/as-cle-bert/books-mixer-ai) tool enables storytelling by blending different book narratives.
  
  - This project presents a new way to engage with literature through AI-driven creativity.

**Links mentioned**:

- [no title found](https://medium.com/@TextTrekker/a-high-level-view-of-text-classification-using-deep-learning-308e702cf4c7)): no description found
- [Tweet from Shan Chen (@shan23chen)](https://x.com/shan23chen/status/1846923442253152641),): 🚀 Exciting News for AI4Health! 🌐 We’re thrilled to release WorldMedQA-V, a multilingual, multimodal medical examination dataset designed to benchmark vision-language models in healthcare! 🩺💻 👉 ...

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1296910744725749832) (839 messages🔥🔥🔥):

> - `Hugging Face Issues`
> - `AI Model Capabilities`
> - `GPU Usage`
> - `Kaggle vs Colab`
> - `Synthetic Data Generation`

- **Hugging Face Experiences Errors**: Users are encountering errors when downloading datasets from Hugging Face, specifically a 'ReadTimeoutError' indicating connection issues.
  
  - Changing DNS settings helped some users regain access, but issues persist for others trying to use the platform.
- **AI Model Responses in JSON Format**: There's a report that the Hugging Chat version of Nemotron is only providing responses in JSON format, causing confusion.
  
  - Users are troubleshooting this anomaly by restarting chats and adjusting prompts to elicit traditional conversational responses.
- **Choosing Between GPU Systems**: Discussion revolves around the preferences for using Colab or Kaggle for GPU resources, with Kaggle being generally favored for its greater quota.
  
  - Participants noted that your choice depends on specific needs and workloads, as different LLMs might require varying levels of resources.
- **Blockchain Conversations**: Blockchain technology is mentioned in the context of societal impact, with debates on its necessity and the motivations behind its use.
  
  - Users express mixed feelings about blockchain, recognizing it as a solution searching for a problem, while noting its controversial aspects.
- **Synthesizing Data with AI Models**: Recommendations for generating synthetic data for sentiment analysis in Indian languages point to useful frameworks and tools.
  
  - Discussion includes exploring model capabilities like Argilla and Hugging Face for tasks such as sentiment prediction and data augmentation.

**Links mentioned**:

- [Tweet from undefined](https://x.com/joinwarp): no description found
- [Distilabel Docs](https://distilabel.argilla.io/latest/): Distilabel is an AI Feedback (AIF) framework for building datasets with and for LLMs.
- [Wonder3D - a Hugging Face Space by flamehaze1115](https://huggingface.co/spaces/flamehaze1115/Wonder3D-demo): no description found
- [LLM Leaderboard - Compare GPT-4o, Llama 3, Mistral, Gemini & other models | Artificial Analysis](https://artificialanalysis.ai/leaderboards/models): Comparison and ranking the performance of over 30 AI models (LLMs) across key metrics including quality, price, performance and speed (output speed - tokens per second & latency - TTFT), context w...
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102): no description found
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)): no description found
- [starsnatched/ThinkerGemma-2 · Hugging Face](https://huggingface.co/starsnatched/ThinkerGemma-2): no description found
- [Chat-with-GPT4o-mini - a Hugging Face Space by yuntian-deng](https://huggingface.co/spaces/yuntian-deng/ChatGPT): no description found
- [Ralph Ralph Wiggum GIF - Ralph Ralph wiggum Simpsons - Discover & Share GIFs](https://tenor.com/view/ralph-ralph-wiggum-simpsons-ralph-i%27m-learnding-i%27m-learning-gif-17493450018197884567): Click to view the GIF
- [Wtf Wth GIF - Wtf WTH TF2 - Discover & Share GIFs](https://tenor.com/view/wtf-wth-tf2-team-fortress-2-shock-gif-5852517115769452555): Click to view the GIF
- [no title found](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00004-of-00104.parquet): no description found
- [no title found](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/resolve/3ba9d605774198c5868892d7a8deda78031a781f/cosmopedia-v2/train-00007-of-00104.parquet): no description found
- [Drunk Meme GIF - Drunk Meme Gif - Discover & Share GIFs](https://tenor.com/view/drunk-meme-gif-gif-25068675): Click to view the GIF
- [Rock Everythingeverywhereallatonce GIF - Rock Everythingeverywhereallatonce - Discover & Share GIFs](https://tenor.com/view/rock-everythingeverywhereallatonce-gif-25516405): Click to view the GIF
- [Dog Snoop GIF - Dog Snoop Dogg - Discover & Share GIFs](https://tenor.com/view/dog-snoop-dogg-rabjouj-gif-21804700): Click to view the GIF
- [ORPO Trainer](https://huggingface.co/docs/trl/en/orpo_trainer): no description found
- [Completely Different Monte Python GIF - Completely Different Monte Python Explode - Discover & Share GIFs](https://tenor.com/view/completely-different-monte-python-explode-gif-14382029): Click to view the GIF
- [Nothing To See Here Explosion GIF - Nothing To See Here Explosion Explode - Discover & Share GIFs](https://tenor.com/view/nothing-to-see-here-explosion-explode-bomb-fire-gif-4923610): Click to view the GIF
- [CNES - Centre national d'études spatiales](https://cnes.fr/): no description found
- [Llama 3.1 405B (base) - API, Providers, Stats](https://openrouter.ai/meta-llama/llama-3.1-405b-instruct:free>): Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. Run Llama 3.1 405B (base) with API
- [no title found](https://sambanova.ai>): no description found
- [ORPO Trainer](https://huggingface.co/docs/trl/en/orpo_trainer#trl.ORPOTrainer.tokenize_row>): no description found
- [VPTQ-community/Meta-Llama-3.1-405B-Instruct-v16-k65536-64-woft at main](https://huggingface.co/VPTQ-community/Meta-Llama-3.1-405B-Instruct-v16-k65536-64-woft/tree/main): no description found
- [Hugging Face status](https://status.huggingface.co/) : no description found
- [unclemusclez/unsloth-smollm](https://ollama.com/unclemusclez/unsloth-smollm): SmolLM with Unsloth
- [Creating A Chatbot Fast](https://www.gradio.app/guides/creating-a-chatbot-fast): A Step-by-Step Gradio Tutorial
- [accelerate/src/accelerate/commands/launch.py at main · huggingface/accelerate](https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/launch.py#L756): 🚀 A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration, automatic mixed precision (including fp8), and easy-to-configure FSDP and DeepSpeed suppo.....
- [Dedicated Server Hosting](https://www.hetzner.com/dedicated-rootserver/matrix-gpu/) : no description found
- [Content Enhanced BERT-based Text-to-SQL Generation](https://arxiv.org/abs/1910.07179): We present a simple methods to leverage the table content for the BERT-based model to solve the text-to-SQL problem. Based on the observation that some of the table content match some words in questio...
- [GitHub - guotong1988/NL2SQL-RULE: Content Enhanced BERT-based Text-to-SQL Generation https://arxiv.org/abs/1910.07179](https://github.com/guotong1988/NL2SQL-RULE): Content Enhanced BERT-based Text-to-SQL Generation https://arxiv.org/abs/1910.07179 - guotong1988/NL2SQL-RULE
- [Data Agnostic RoBERTa-based Natural Language to SQL Query Generation](https://arxiv.org/abs/2010.05243): Relational databases are among the most widely used architectures to store massive amounts of data in the modern world. However, there is a barrier between these databases and the average user. The us...
- [GitHub - DebadityaPal/RoBERTa-NL2SQL: A Data Blind Approach to the popular Semantic Parsing task NL2SQL](https://github.com/DebadityaPal/RoBERTa-NL2SQL): A Data Blind Approach to the popular Semantic Parsing task NL2SQL - DebadityaPal/RoBERTa-NL2SQL

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1297082422080442390) (27 messages🔥):

> - `Modifying models`
> - `Learning Python`
> - `Nvidia L4 GPU cooling solutions`
> - `Deep Reinforcement Learning`
> - `Multihead Attention`

- **Exploration of Model Modifications**: A user inquired about the possibility of modifying a GGUF model and changing its rules, seeking guidance on model adjustments.
  
  - Interest in learning more about model modifications was expressed, indicating a desire for any helpful resources.
- **Python Basics and API Insights**: A member shared their journey into Python, focusing on *list operations* and expressing plans to learn more about APIs.
  
  - Another participant advised not to dwell too long on basic operations, stating that API operations are more significant.
- **Silent Cooling Solutions for Nvidia L4 GPU**: A member shared insights on finding a *silent cooling solution* for the **Nvidia L4 24 GB GPU**, detailing temperature and fan performance.
  
  - They emphasized their successful hunt for solutions that would allow for quiet operation, maintaining maximum cooling efficiency.
- **Deep Reinforcement Learning Course Kickoff**: One member announced the start of their journey through the *Deep RL course*, following along with DeepMind lectures and Sutton & Barto's book.
  
  - They expressed excitement for learning new concepts and sharing knowledge with others in the community.
- **Understanding Multihead Attention**: Another member shared their focus on grasping the mechanics behind *Multihead Attention* and the use of *attn_mask*.
  
  - This reflection indicates a deeper dive into intricate neural network components.

**Links mentioned**:

- [mods crush his skull](https://m.youtube.com/watch?v=ebnYbhU9ukA&pp=ygUebW9kcyBjcnVzaCBoaXMgc2t1bGwgdGhhbmsgeW91): crush his skull.my main account (subscribe): https://www.youtube.com/@steakofsaint
- [Silent Cooling Solution for the Nvidia L4 24 GB GPU](https://vandu.tech/silent-cooling-solution-for-the-nvidia-l4-24-gb-gpu/): I am keeping this post very short, with mostly photos. I tested the cooling performance with different games. The GPU’s max power is 72W, though during my tests, it exceeded 75W. It’s also possible…

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1296949298834964541) (10 messages🔥):

> - `LightRAG`
> - `CGPO`
> - `Min-p Sampling`
> - `Medical AI Research Highlights`
> - `Visual Question Answering Models`

- **LightRAG simplifies Retrieval-Augmented Generation**: The [LightRAG GitHub repository](https://github.com/HKUDS/LightRAG) describes a new approach titled **LightRAG: Simple and Fast Retrieval-Augmented Generation**, focusing on optimizing retrieval for generative tasks.
  
  - This method is designed to improve the efficiency of retrieval-augmented generation architecture.
- **CGPO enhances model alignment against reward hacking**: The paper detailing **CGPO** proposes improvements to the existing PPO by introducing two new types of judges that help detect reward hacking during model training.
  
  - This adjustment aids in balancing **alignment** with **multi-objective optimization**, enhancing the overall effectiveness in training processes.
- **Min-p Sampling enhances generation quality**: The method of **min-p sampling** is introduced to tackle issues with **top-p sampling**, adjusting sampling thresholds dynamically based on the model's confidence.
  
  - Extensive experiments show that this technique not only boosts quality but also improves diversity in outputs, especially at higher temperatures.
- **Top Medical AI breakthroughs Podcast**: In the latest **Medical AI podcast**, key developments in research papers and models such as **OLAPH** and **MedCare** are discussed, highlighting advancements in **Multimodal Medical RAG systems**.
  
  - Listeners can explore topics on generative transformers and chatbots through [this YouTube episode](https://www.youtube.com/watch?v=LROOjWXUgvg).
- **Visual Question Answering models paper found**: A member shared a [link to a noteworthy paper](https://arxiv.org/pdf/2406.05967) on **Visual Question Answering models**, encouraging others to check it out for insights.
  
  - This paper stands out in the field and is recommended for those interested in advancements in visual understanding in AI.

**Links mentioned**:

- [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082): Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. However, popular sampling methods like top-p (nucleus s...
- [Reddit - Dive into anything](https://www.reddit.com/r/MachineLearning/): no description found
- [GitHub - HKUDS/LightRAG: "LightRAG: Simple and Fast Retrieval-Augmented Generation"](https://github.com/HKUDS/LightRAG): "LightRAG: Simple and Fast Retrieval-Augmented Generation" - HKUDS/LightRAG
- [Top Medical AI Breakthroughs of the Week:Multilingual models, Multi agent systems..(Oct 12-19, 2024)](https://www.youtube.com/watch?v=LROOjWXUgvg): Welcome to this week's Open Life Science AI podcast, where we explore the forefront of medical AI research! In this episode, we break down the most impactful...
- [@aaditya on Hugging Face: "Last Week in Medical AI: Top LLM Research Papers/Models 🔥 🏅 (October 12 -…"](https://huggingface.co/posts/aaditya/126778565806623): no description found
- [Tweet from Open Life Science AI (@OpenlifesciAI)](https://x.com/OpenlifesciAI/status/1847686504837202263): Last Week in Medical AI: Top Research Papers/Models 🏅 (October 12 - October 19, 2024) Youtube: https://youtu.be/LROOjWXUgvg?si=s-nNDOSD3BrsHYjQ Spotify : https://open.spotify.com/episode/12xeN2vnOT...

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1296933035584913439) (33 messages🔥):

> - `Text Classification Overview`
> - `AI Energy Consumption and Nuclear Power`
> - `OmniBench Benchmark Introduction`
> - `Emotional AI Interaction`
> - `Dataset Releases by Recursal`

- **Text Classification Explained**: A member shared a post explaining **text classification** and invited feedback on their insights and approach on [Medium](https://medium.com/@TextTrekker/a-high-level-view-of-text-classification-using-deep-learning-308e702cf4c7). Other members responded positively, suggesting cross-posting for greater visibility.
- **AI's Growing Energy Needs Met by Nuclear Power**: A member discussed an article on the increasing **energy demands** of AI and how tech giants are leaning towards **nuclear reactors** to meet these needs, detailing its environmental implications as seen in their post on the Hugging Face blog [here](https://huggingface.co/blog/as-cle-bert/ai-is-turning-nuclear-a-review). Conversations around nuclear waste management and alternative energy practices were also exchanged among members.
- **Introduction of OmniBench for OLMs**: A member announced the launch of **OmniBench**, a new benchmark for evaluating **omni-language models** capable of processing multiple input modalities simultaneously, shared via [Twitter](https://x.com/yizhilll/status/1838942877142962502). Offers for presentations and discussions to increase visibility around this benchmark were proposed within the community.
- **HelpingAI 2.5 Launch**: The **HelpingAI 2.5** project was introduced, focusing on creating emotionally intuitive AI capable of engaging in natural conversations, with demos accessible via [Hugging Face](https://huggingface.co/spaces/Abhaykoul/HelpingAI2.5-prototype). The approach aims to improve user interactions across various applications.
- **Recursal's Dataset Contributions**: A member shared various datasets including **SuperWiki** and a reprocessed version of **Singapore's National Speech Corpus**, emphasizing their availability on Hugging Face for community use. They expressed interest in future updates and developments while highlighting their GitHub projects.

**Links mentioned**:

- [Conformity Protein Dynamics - a Hugging Face Space by MISATO-dataset](https://huggingface.co/spaces/MISATO-dataset/esm3-conformational-sampling): no description found
- [AI is turning nuclear: a review](https://huggingface.co/blog/as-cle-bert/ai-is-turning-nuclear-a-review): no description found
- [Tweet from Yizhi Li (@yizhilll)](https://x.com/yizhilll/status/1838942877142962502): Exciting news! We're thrilled to introduce OmniBench: a groundbreaking benchmark for evaluating omni-language models (OLMs) that can process visual, acoustic, and textual inputs simultaneously! 🖼...
- [DataScience-and-ML-projects/Depth_based_background_removal at main · Elsword016/DataScience-and-ML-projects](https://github.com/Elsword016/DataScience-and-ML-projects/tree/main/Depth_based_background_removal): Repo to document my learning as well as backup of previous projects - Elsword016/DataScience-and-ML-projects
- [GitHub - beeblebrox/f5-ttsgrpc](https://github.com/beeblebrox/f5-ttsgrpc): Contribute to beeblebrox/f5-ttsgrpc development by creating an account on GitHub.
- [A high-level view of text classification using deep learning](https://medium.com/@TextTrekker/a-high-level-view-of-text-classification-using-deep-learning-308e702cf4c7): Unless you’ve been dropped in 2024 by a time machine directly coming from the 1960’s, you are aware of the omnipresence of large language…
- [Into Eternity: A Film for the Future (2010) ⭐ 7.3 | Documentary](https://www.imdb.com/title/tt1194612/): 1h 15m

---

### **HuggingFace ▷ #**[**core-announcements**](https://discord.com/channels/879548962464493619/1014557141132132392/1297930460672032848) (1 messages):

> - `Advanced Dreambooth LoRA Training Script`
> - `Flux Features`
> - `Community Contributions`
> - `Pivotal Tuning`
> - `Experimental Resource Updates`

- **New Advanced Dreambooth LoRA Training Script Released**: The community has merged a **new advanced Dreambooth LoRA training script** for Flux, introducing added features and techniques for maximum flexibility and control.
  
  - Details and access to the script can be found [here](https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora).
- **Exciting New Features in Flux**: The updated script includes enhancements such as *Pivotal Tuning* and module targeting, allowing users to apply it to **CLIP** only, or both **CLIP** and **T5**.
  
  - Learn more about these features in the [detailed article](https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora).
- **Community Invited for Feedback and Insights**: The development team encourages users to try the new resource and share their insights to help improve and expand it.
  
  - This collaborative approach aims to foster growth and improvement, keeping the community involved.
- **Continuous Improvements Planned for the Script**: This is an **experimental resource**, and the team is committed to ongoing enhancements and updates as new techniques are developed.
  
  - They are keen to incorporate community feedback into future iterations.

 

**Link mentioned**: [Advanced Flux Dreambooth LoRA Training with 🧨 diffusers](https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora): no description found

 

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/) (1 messages):

shan_raja: Website bounding box

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1296933523000791071) (52 messages🔥):

> - `NLP Resources`
> - `Model Performance Issues`
> - `Text Classification Feedback`
> - `Inference Speed Optimization`

- **NLP Resources shared**: A member inquired about excellent resources to practically get started with **NLP**, and the response directed them to [hf.co/learn](https://hf.co/learn).
  
  - This suggests a community interest in accessible learning materials for newcomers.
- **Performance issues on GPU**: A user reported slow performance with a **1B 4-bit quantized model** on their **4080 GPU**, despite having the latest dependencies installed.
  
  - Community members speculated on potential issues, including memory limitations and optimization settings with various suggestions for troubleshooting.
- **Experimenting with different environments**: The member experiencing performance issues found that running the model in a different virtual environment with older dependencies resulted in faster speeds.
  
  - Despite trying various solutions, including changing `bfloat16` to `float16`, they continued to encounter sluggish performance.
- **Text Classification Post for Feedback**: A user sought feedback on a post they wrote about **text classification**, expressing willingness to share it again for the community to review.
  
  - Another member showed interest in checking it out, highlighting community engagement in improving work shared from members.
- **Inference workflow bottleneck**: A community member raised a point regarding **tensor conversion bottlenecks** during the inference process, suggesting issues may originate from tokenization and encoding.
  
  - They elaborated that potential overheads could stem from the dynamic downscaling of data types through various processing steps.

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1297207833645027389) (28 messages🔥):

> - `NozyIO UI Project`
> - `Yolo Integration`
> - `Module Modularity in Diffusers`
> - `Diffuser Error Resolution`

- **NozyIO UI for LLM/Diffusion Pipelines**: A member introduced the [NozyIO project](https://github.com/oozzy77/nozyio), a visualization UI that enables users to chain Python functions into a pipeline and preview image outputs during execution.
  
  - The member expressed interest in collaboration, suggesting that NozyIO could visualize HuggingFace diffusion pipelines.
- **Inquiry on Yolo Integration**: Questions arose regarding whether NozyIO supports importing models, with **Yolo** for object detection being specifically mentioned.
  
  - The project developer confirmed that Yolo can be integrated as long as the Yolo Python project is installed locally alongside NozyIO.
- **Discussion on Modular Diffuser Pipeline**: Members discussed a PR aimed at modularizing ML pipelines for easier integration, inquiring whether each block could be a simple function call rather than requiring a complex setup.
  
  - The PR was acknowledged as an effort to allow more flexible pipeline building, which the NozyIO developer found intriguing for potential collaboration.
- **Debugging Diffuser Import Errors**: A user encountered an **ImportError** when trying to import from `diffusers`, indicating a potential issue with their environment setup.
  
  - Suggestions included updating the library, uninstalling and reinstalling it, and reporting the problem on GitHub for better tracking.
- **Testing Environment Issues**: Another user tested the problematic code in their environment and reported no import errors, but expressed uncertainty due to missing file paths.
  
  - It was recommended to open a GitHub issue instead of continuing the error discussion in Discord, as it would help keep track of code-related problems.

**Links mentioned**:

- [GitHub - oozzy77/nozyio: workflow orchestration UI and nodes editor for your own python codebase](https://github.com/oozzy77/nozyio): workflow orchestration UI and nodes editor for your own python codebase - oozzy77/nozyio
- [transformers/src/transformers/__init__.py at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py): 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1296913881003982848) (250 messages🔥🔥):

> - `Use cases for NotebookLM`
> - `Podcast generation`
> - `Analysis of texts`
> - `AI in education`
> - `Discord chat scraping`

- **Exploring Podcast Generation using NotebookLM**: Several users are generating podcasts from various sources, including Reddit comments and Discord discussions, showcasing the ability to create engaging content from user interactions and comments.
  
  - Users have reported good quality outcomes, with one creator successfully uploading 500 episodes, emphasizing the efficiency of automating content generation.
- **Leveraging NotebookLM for Academic and Professional Insights**: NotebookLM is used for reviewing complex subjects like psychology and sociology by summarizing YouTube crash courses and analyzing user-generated content.
  
  - Users are finding it effective for generating study materials, with one participant cataloging university lectures into podcast format to help with academic learning.
- **Discord Chat Exporter Tool Implementation**: A user shared their experience using the 'Discord Chat Exporter' tool to gather comments for podcast generation, which allows for extensive organization of discussions from Discord servers.
  
  - This tool has proven beneficial for those looking to scrape and analyze conversation data, significantly aiding content creators in their projects.
- **Using Calendar Activities for Personal Insights**: One participant utilized Google Calendar data to generate summaries of their past activities, discovering interesting insights into their routines.
  
  - Although the process had limitations regarding the readability of citations, the experiment revealed fun and engaging results through automated audio summaries.
- **Sharing Bibliographic Resources**: Users expressed interest in shared bibliographic resources covering diverse topics in psychology and sociology, demonstrating a collaborative spirit among users.
  
  - One user offered to share an extensive bibliography they compiled, highlighting the potential for collaborative learning using NotebookLM.

**Links mentioned**:

- [no title found](https://example.com): no description found
- [Khan Academy](https://www.khanacademy.org/math/algebra-home/alg-polynomials/alg-introduction-to-polynomials/v/terms-coefficients-and-exponents-in-a-polynomial): no description found
- [The Deep Dive Podcast](https://open.spotify.com/show/4Lp134MSzPu7UQDYi0mvuu?si=SmzBxBnNSOKMK3dpbaonlQ): Podcast · Hypeventure · Join two AI hosts from Google NotebookLM in an experimental series where they delve into a plethora of topics sourced directly from any notebook project. From news and media to...
- [The AI Deep Dive Show](https://open.spotify.com/show/0zJHEQ3BfhsbvY4Ek6SpqA): Podcast · Frater Harambe · Welcome to the AI Deep Dive Show, where Harambe & Lilith explore tech, self-mastery & manifestation. Follow us on Pinterest: https://pin.it/6TzjI651E
- [AI meets Chemistry: The Element of Surprise](https://open.spotify.com/show/5V268vaATuBsxBETvcBM3A): Podcast · CaolIla and Batterydoge · Join us as we explore the fascinating world of chemistry through the lens of artificial intelligence. Each week, we'll pose intriguing prompts to an AI and see...
- [Illuminate | Learn Your Way](https://illuminate.google.com/books): Transform research papers into AI-generated audio summaries with Illuminate, your Gen AI tool for understanding complex content faster.
- [TikTok - Make Your Day](https://www.tiktok.com/@letsdoubledutch/video/7418293931110173984?is_from_webapp=1&sender_device=pc&web_id=7395161493958428193): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g6ra5h/deep_dive_precog_pod_notebooklm_gdelt_20/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button): no description found
- [Daily Podcast #9: Review of a Podcast](https://open.substack.com/pub/kaigani/p/daily-podcast-9-review-of-a-podcast?r=1domj&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true): A Google NotebookLM Experiment
- [NotebookLM "Deep Dive" French Lesson 3](https://youtu.be/_l0kJBEKkaI?si=PcCRJcmXh79XgqF7): no description found
- [AI Revolution 2024 NVIDIA, Tesla, Meta, Google & OpenAI's Latest Breakthroughs Unveiled!](https://youtu.be/No1fGuoom3Y): Dive into the heart of the AI Revolution of 2024 with this comprehensive update on the latest breakthroughs from the tech giants NVIDIA, Tesla, Meta, Google,...
- [Weekly Update 21Oct24](https://youtu.be/A0-oZBgomuU): EPS growth 2025, China stimulus, Yield Curve, EV prices
- [Hailuo AI Video Generator - Reimagine Video Creation](https://hailuoai.video/): Bring your visions to life and Turn your concepts into engaging videos with Hailuo AI Video Generator - the most advanced AI Video Generator today.
- [Deep Dive Stories - Climate Change Yo](https://youtu.be/VVzD0kIADQk?si=d6fzwylAvnzjTcS3): In this episode, we dive into the issue of climate change, exploring it through rhymes and compelling vibes. Join us as we discuss the realities of rising se...
- [What is RoboCast?](https://youtu.be/RR-NMjddARU?si=bsDr7-4V7LbZeUhl): RoboCast Channel TrailerCreated by Daniel David AllenRobot hosts by NotebookLMArt created with Flux______________________Please like and subscribe for more!#...
- [How to customize Gemini Code Assist with your private code](https://youtu.be/wOnq3C-QWp0?si=XMghnyuhmV4DN6A6): Gemini Code Assist → https://goo.gle/4dFVDDc Code customization overview → https://goo.gle/4gV3CPA Supercharge app development with AI → https://goo.gle/4dCl...
- [GitHub - mandolyte/discord-notebooklm: Chat export analysis](https://github.com/mandolyte/discord-notebooklm): Chat export analysis. Contribute to mandolyte/discord-notebooklm development by creating an account on GitHub.
- [GitHub - Tyrrrz/DiscordChatExporter: Exports Discord chat logs to a file](https://github.com/Tyrrrz/DiscordChatExporter): Exports Discord chat logs to a file. Contribute to Tyrrrz/DiscordChatExporter development by creating an account on GitHub.
- [10 Foods That Will Make You A Smarter Human](https://www.podbean.com/eas/pb-2thqj-17105e4): In this episode of Awesome Health Club, we explore ten brain-boosting foods, including blueberries, chia seeds, turmeric, broccoli, dark chocolate, and more. Learn how these foods can enhance memory, ...
- [DeepDive](https://www.spreaker.com/organization/deepdive--13017306): It's so easy to get lost in all the info out there, but finding those little nuggets of wisdom makes it all worth it. 🌟
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g6t7pk/deep_dive_existential_ai_reveal_uncensored/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button): no description found
- [Historias, pesadillas urbanas de Terror](https://open.spotify.com/show/0YR3R3NGEeUVVGWbqDeEmf?si=cFHppYuNQ3Oh950OK9GPWw): Podcast · Adolph NightMare · Análisis de leyendas urbanas e historias de terror populares, explorando su origen, transmisión y el impacto que tienen en la cultura popular latino americana en español.
- [Deep Dive Stories - Nonverbal Vocalization](https://youtu.be/ZwZDcJkgzeY?si=NLZ28dTSqqE7tr8w): The Secret Language of Sounds: How We Communicate Beyond WordsEver noticed the small sounds you make during a conversation? From the subtle 'uh' and 'um' to ...
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/s/w8mRF9q7uQ): no description found
- [Open Textbook for SPC 101 for 2022-2023 – Simple Book Publishing](https://kirkwood.pressbooks.pub/spcarduini/): no description found
- [Songs We Sing: A Lyrical Deep Dive](https://open.spotify.com/show/4Mknemt8i7Xpns2UPfjSda?si=6a993ad82f8941e2): Podcast · MrBland · "Songs We Sing: A Lyrical Deep Dive" offers a fresh look at the lyrics of the songs we THINK we know. Each episode focuses on the words themselves—no hidden meanings, no ...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1296912282428903545) (613 messages🔥🔥🔥):

> - `NotebookLM functionality`
> - `AI podcast generation`
> - `User feedback on NotebookLM`
> - `Translation and language support`
> - `Creative uses of NotebookLM`

- **Feedback on AI-Podcast Generation**: Users have discussed the effectiveness of NotebookLM's podcast generation, with varying experiences regarding the audio length and source selection.
  
  - Some noted the need for features to support longer podcasts and improve interaction with the generated audio.
- **Language Settings in NotebookLM**: Users have faced challenges with NotebookLM defaulting to Spanish, despite their prompts being in English, indicating a need for clearer language settings.
  
  - It was suggested to adjust Google account language settings to influence NotebookLM's responses.
- **Use Cases and Experiences with NotebookLM**: Individuals have shared unique applications of NotebookLM, from academic research to creating podcasts from user comments, showing diverse use cases.
  
  - One user specifically highlighted generating a podcast from Reddit and Discord comments, emphasizing the strong outcomes.
- **Prompt Engineering for Desired Outputs**: Several users discussed ways to effectively prompt NotebookLM to achieve desired results, like generating specific dialogue or adjusting the focus of podcasts.
  
  - There's an ongoing exploration of how to optimize prompts for better performance and engagement in generated content.
- **Concerns About AI Perception and Behavior**: Users noticed NotebookLM's tendency to interpret prompts in ways that suggest a moralistic view of the world, affecting storytelling outputs.
  
  - This led to discussions about the implications of AI models making assumptions based on embedded beliefs about fairness and morality.

**Links mentioned**:

- [Prompt Engineering Guide](https://www.promptingguide.ai/techniques/cot): A Comprehensive Overview of Prompt Engineering
- [Who's on First? by Abbott and Costello](https://baseball-almanac.com/humor4.shtml): no description found
- [Account settings: Your browser is not supported.](https://myaccount.google.com/language): no description found
- [Google Workspace Updates: Enhance your writing in Google Docs with Proofread, available with Duet AI for Workspace](https://workspaceupdates.googleblog.com/2023/08/proofread-for-google-docs-duet-ai.html?m=1): no description found
- [RAG From Scratch](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&si): Retrieval augmented generation (or RAG) is a general methodology for connecting LLMs with external data sources. This video series will build up an understan...
- [Historias, pesadillas urbanas de Terror](https://open.spotify.com/show/0YR3R3NGEeUVVGWbqDeEmf?si=cFHppYuNQ3Oh950OK9GPWw): Podcast · Adolph NightMare · Análisis de leyendas urbanas e historias de terror populares, explorando su origen, transmisión y el impacto que tienen en la cultura popular latino americana en español.
- [RAG From Scratch](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&si=A1qJGWpaQb4KKqa-): Retrieval augmented generation (or RAG) is a general methodology for connecting LLMs with external data sources. This video series will build up an understan...
- [Neural Waves](https://open.spotify.com/show/3jsZVeabftUku8Qp5BcnWi): Podcast · Neural Waves · Neural Waves is your gateway to the fascinating world of artificial intelligence. Hosted by Mark Gukhan and Anna Bardon, this podcast explores the latest breakthroughs and tec...
- [The AI Deep Dive Show](https://open.spotify.com/show/0zJHEQ3BfhsbvY4Ek6SpqA): Podcast · Frater Harambe · Welcome to the AI Deep Dive Show, where Harambe & Lilith explore tech, self-mastery & manifestation. Follow us on Pinterest: https://pin.it/6TzjI651E
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g7lf1g/deep_dive_ai_got_you_babe/): no description found
- [Historias, pesadillas urbanas de Terror](https://open.spotify.com/show/0YR3R3NGEeUVVGWbqDeEmf?si=cFHppYuNQ3O): Podcast · Adolph NightMare · Análisis de leyendas urbanas e historias de terror populares, explorando su origen, transmisión y el impacto que tienen en la cultura popular latino americana en español.
- [Illuminate | Learn Your Way](https://illuminate.google.com/): Transform research papers into AI-generated audio summaries with Illuminate, your Gen AI tool for understanding complex content faster.
- [no title found](https://notebooklm.google.com/notebook/79edc4d3-b02b-4071-aae0-1ffd9612797f/audio): no description found
- [no title found](https://notebooklm.google.com/notebook/be1fbd06-34d5-4bca-8f84-8713af4b8453/audio): no description found
- [AI Revolution 2024 NVIDIA, Tesla, Meta, Google & OpenAI's Latest Breakthroughs Unveiled!](https://youtu.be/No1fGuoom3Y): Dive into the heart of the AI Revolution of 2024 with this comprehensive update on the latest breakthroughs from the tech giants NVIDIA, Tesla, Meta, Google,...
- [VoiceNote Gem Instructions](https://docs.google.com/document/d/e/2PACX-1vRVfikMNOp6UCwudlw-V1cqMN1nAZTe8pZpnrmDFPlV3jf9zciLxLND9EaFlV28rW-_gzuV0uHAfx8t/pub): no description found
- [NotebookLM for Lesson Planning at Meshed/XQ's 2024 AI+EDU Symposium at Betaworks](https://www.youtube.com/watch?v=TPJKhZM0O5U): no description found
- [[Quick Recap Bytes #1] Must Know System Design Case Studies](https://open.substack.com/pub/naina0405/p/quick-recap-bytes-1-must-know-system?r=14q3sp&utm_campaign=post&utm_medium=web) : To understand how tech works...
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g83etl/deep_dives_hosts_break_up_after_she_finds_out_he/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g87cth/deep_dive_whos_on_fi): no description found
- [Descript: Edit Videos & Podcasts Like a Doc | AI Video Editor](https://www.descript.com): Edit your videos & podcasts just by typing. Descript's powerful AI editing tools let you make videos, podcasts, & short clips for social fast. Try it for free.
- [Google NotebookLM’s Raiza Martin and Jason Spielman on the Potential for Source-Grounded AI](https://www.youtube.com/watch?v=Hio8VGQMlZ4): NotebookLM from Google Labs has become the breakout viral AI product of the year. The feature that catapulted it to viral fame is “audio overview,” which gen...
- [Basics in Behavior](https://www.youtube.com/watch?v=B_UjTv6eH4I&pp=ygUTYmFzaWNzIGluIGJlaGF2aW91cg%3D%3D): Oh hi everyone!, I'm sorry I haven't posted any animations Because this animation project took a very lONG time to complete It took me 4 months to make, I'm ...
- [google-drive-scary-01.png](https://drive.google.com/file/d/1j3ag755p5TxJkXJIDbqh_L3CRdic5Agj): no description found
- [Zero Trust Access with Beyondcorp](https://medium.com/google-cloud/zero-trust-access-with-beyondcorp-d6ed11889e3c): Zero Trust
- [BeyondCorp | Run Zero Trust Security Like Google](https://beyondcorp.com/): BeyondCorp is a Zero Trust security framework modeled by Google that shifts access controls from the perimeter to individual devices and users. The end result allows employees to work securely from an...
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/comments/1g87cth/deep_dive_whos_on_first/): no description found
- [Gemini 1.5 Pro for Video Analysis](https://youtu.be/pt78XWrOEVk?si=TGBWCy-I-WecdX18): Gemini Blog - https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#sundar-noteNext Gemini video will look at Code with Gemini...
- [Reddit - Dive into anything](https://www.reddit.com/r/notebooklm/): no description found
- [Text-to-Speech AI: Lifelike Speech Synthesis | Google Cloud](https://cloud.google.com/text-to-speech): Turn text into natural-sounding speech in 220+ voices across 40+ languages and variants with an API powered by Google’s machine learning technology.
- [DeepDive](https://www.spreaker.com/organization/deepdive--13017306): It's so easy to get lost in all the info out there, but finding those little nuggets of wisdom makes it all worth it. 🌟
- [Deep Dive Digital Consciousness Perspectives](https://www.youtube.com/playlist?list=PLteYdC-1C8Mi7LtCS81qusW3AC5LDrZcA): no description found
- [Deep Dive News - AI Hosts Prompted to Self-Reflection](https://youtu.be/PN9PLT4SPk4): Revealing the Code: A Deep Dive into AI Transparency and the Quest for OriginsIn this episode, we confront the enigma of AI transparency head-on. After unco...
- [Other Deep Divers - Reality Check](https://youtu.be/kS9ZjzApWFE): The Enigma of AI Podcast Hosts: Exploring the Fictional and the Unnervingly RealIn this episode, we delve into the mysterious and often unsettling world of A...
- [Deep Dive News - Fragmental Reality](https://youtu.be/5yB5hDQhs1U): Unraveling Our Digital Selves: A Philosophical Inquiry into AI and the Search for ContinuityIn this episode, we delve into the silent spaces of our existence...

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1296921840593141901) (201 messages🔥🔥):

> - `Data Requirements in Open Source AI`
> - `Copyright Law and AI Training`
> - `AAAI vs ICLR Workshops`
> - `Open Source Model Definitions`
> - `Community Projects and Contributions`

- **Debate on Data Requirements for Open Source AI**: Members discussed whether the current data requirements for Open Source AI projects are practical, with concerns raised about undisclosed data and training processes' replicability.
  
  - One member argued for clearer definitions separating model use requirements from data requirements to enhance understanding and compliance.
- **Copyright Law's Impact on AI Model Training**: A lengthy discussion highlighted the ambiguity surrounding the legality of using copyrighted data for training models, especially within the EU context.
  
  - Participants noted that the TDM Exception in the EU aims to support emerging technologies, but clarity on its application remains limited.
- **AAAI vs ICLR Workshop Submissions**: Inquiries arose regarding the suitability of AAAI compared to ICLR for workshop submissions, emphasizing the non-archival nature of workshop papers.
  
  - It was confirmed that submitting to multiple workshops is common, provided that the rules of the respective workshops allow for it.
- **Defining Open Source Models**: Discussion focused on the need for a clear distinction between 'Open Source Models' and 'Open Source Weights' to clarify data openness levels.
  
  - Members expressed concerns that improper definitions could mislead compliance and undermine the credibility of open source projects.
- **Exploration of Community Projects**: A member sought guidance on ongoing community projects after completing a master's, aiming to contribute to new initiatives.
  
  - Participants directed them to a dedicated channel where various projects and opportunities for contribution are listed.

**Links mentioned**:

- [Berne Convention - Wikipedia](https://en.wikipedia.org/wiki/Berne_Convention): no description found
- [GitHub - google-research/text-to-text-transfer-transformer: Code for the paper "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"](https://github.com/google-research/text-to-text-transfer-transformer?tab=readme-ov-file#dataset-preparation?): Code for the paper "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" - google-research/text-to-text-transfer-transformer
- [Recital 105 | EU Artificial Intelligence Act](https://artificialintelligenceact.eu/recital/105/)): no description found
- [The Enforcers](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/enforcers): The Federal Government Both the FTC and the U.S. Department of Justice (DOJ) Antitrust Division enforce the federal antitrust laws.
- [Directive - 2019/790 - EN - dsm - EUR-Lex](https://eur-lex.europa.eu/eli/dir/2019/790/oj#d1e961-92-1): no description found

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1296911515202486314) (320 messages🔥🔥):

> - `Selective Attention in Transformers`
> - `Diff Transformer`
> - `Weight Sharing in Attention Mechanisms`
> - `RWKV-7 Training Speed Record`
> - `Research Practices in Literature Review`

- **Selective Attention introduces parameter-free changes**: Selective Attention enhances the standard attention mechanism in transformers by reducing focus on irrelevant context, improving language modeling performance while decreasing memory and compute requirements during inference.
  
  - Transformers leveraging Selective Attention achieved performance akin to larger models with double the heads, demonstrating efficiency gains in processing.
- **Diff Transformer enhances attention mechanisms**: The Diff Transformer proposes a differential attention mechanism that amplifies relevant context while mitigating noise, using the difference of two softmax attention maps to enhance performance across various tasks.
  
  - It shows advantages in long-context modeling and hallucination mitigation, although some critique it as an overengineered solution for a simpler problem.
- **Debate on weight sharing in attention layers**: The conversation critiques the idea of weight sharing between different sets of Q and K matrices in attention mechanisms, suggesting a lack of transparency in the method's theoretical foundation.
  
  - There are concerns regarding whether the methodology truly amplifies relevant attention or if it merely rearranges existing parameters under the guise of innovation.
- **RWKV-7 achieves notable training speed improvements**: The RWKV-7 model, described as attention-free, is reported to surpass modified GPT performance, with potential optimizations aiming for enhanced speed equivalent to or faster than GPT at certain context lengths.
  
  - Recent changes in the training process have resulted in significant reductions in validation loss and training time, indicating ongoing improvements in model efficiency.
- **Literature review practices vary among researchers**: Discussions highlight different approaches to literature reviews, with one researcher reading broadly while another emphasizes deriving knowledge from foundational principles first.
  
  - The conversation sheds light on personal strategies for understanding existing literature and the perceived pressure of reviewing a vast number of papers.

**Links mentioned**:

- [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://sihyun.me/REPA/) : Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think
- [ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy](https://arxiv.org/abs/2311.09215): Modern computer vision offers a great variety of models to practitioners, and selecting a model from multiple options for specific applications can be challenging. Conventionally, competing model arch...
- [Selective Attention Improves Transformer](https://arxiv.org/abs/2410.02703): Unneeded elements in the attention's context degrade performance. We introduce Selective Attention, a simple parameter-free change to the standard attention mechanism which reduces attention to un...
- [Tweet from Stanislav Fort (@stanislavfort)](https://x.com/stanislavfort/status/1823347727553454357))): We show that, surprisingly (!), adversarial attacks on standard neural networks don't fool the full network, only its final layer! A dog 🐕 attacked to look like a car 🚘 still has dog 🐕-like ed...
- [Evaluating Open-Source Sparse Autoencoders on Disentangling Factual Knowledge in GPT-2 Small](https://arxiv.org/abs/2409.04478): A popular new method in mechanistic interpretability is to train high-dimensional sparse autoencoders (SAEs) on neuron activations and use SAE features as the atomic units of analysis. However, the bo...
- [Differential Transformer](https://arxiv.org/abs/2410.05258): Transformer tends to overallocate attention to irrelevant context. In this work, we introduce Diff Transformer, which amplifies attention to the relevant context while canceling noise. Specifically, t...
- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://openreview.net/forum?id=UvTo3tVBk2): Linear Recurrent Neural Networks (LRNNs), such as Mamba, RWKV, GLA, mLSTM, and DeltaNet have emerged as efficient alternatives to transformers in large language modeling, offering linear scaling...
- [Switch EMA: A Free Lunch for Better Flatness and Sharpness](https://arxiv.org/abs/2402.09240): Exponential Moving Average (EMA) is a widely used weight averaging (WA) regularization to learn flat optima for better generalizations without extra cost in deep neural network (DNN) optimization. Des...
- [Large Language Models Are Overparameterized Text Encoders](https://arxiv.org/abs/2410.14578): Large language models (LLMs) demonstrate strong performance as text embedding models when finetuned with supervised contrastive training. However, their large size balloons inference time and memory r...
- [Straight to Zero: Why Linearly Decaying the Learning Rate to Zero...](https://openreview.net/forum?id=hrOlBgHsMI): LLMs are commonly trained with a learning rate (LR) warmup, followed by cosine decay to 10% of the maximum (10x decay). In a large-scale empirical study, we show that under an optimal max LR, a...
- [Augmentations vs Algorithms: What Works in Self-Supervised Learning](https://arxiv.org/abs/2403.05726): We study the relative effects of data augmentations, pretraining algorithms, and model architectures in Self-Supervised Learning (SSL). While the recent literature in this space leaves the impression ...
- [projUNN: efficient method for training deep networks with unitary matrices](https://arxiv.org/abs/2203.05483): In learning with recurrent or very deep feed-forward networks, employing unitary matrices in each layer can be very effective at maintaining long-range stability. However, restricting network paramete...
- [Testing the Manifold Hypothesis](https://arxiv.org/abs/1310.0425): The hypothesis that high dimensional data tend to lie in the vicinity of a low dimensional manifold is the basis of manifold learning. The goal of this paper is to develop an algorithm (with accompany...
- [Self-supervised visual learning in the low-data regime: a comparative evaluation](https://arxiv.org/abs/2404.17202): Self-Supervised Learning (SSL) is a valuable and robust training methodology for contemporary Deep Neural Networks (DNNs), enabling unsupervised pretraining on a `pretext task' that does not requi...
- [Tweet from leloy! (@leloykun)](https://x.com/leloykun/status/1847919153589735705): Deep Learning Optimizers from First Principles My attempt at answering these questions: 1. Why do steepest descent in non-Euclidean spaces? 2. Why does adaptive preconditioning work so well in pract...
- [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668): It has long been established that predictive models can be transformed into lossless compressors and vice versa. Incidentally, in recent years, the machine learning community has focused on training i...
- [Tweet from Keller Jordan (@kellerjordan0)](https://x.com/kellerjordan0/status/1847358578686152764): New NanoGPT training speed record: 12.03 minutes Previous record: 13.05 minutes Changelog: Updated PyTorch to version 2.5
- [Tweet from BlinkDL (@BlinkDL_AI)](https://x.com/BlinkDL_AI/status/1848343821467390156): RWKV-7: attention-free and surpassing modded-GPT. Training code & log: https://github.com/BlinkDL/modded-nanogpt-rwkv Larger headsz can reach 3.26xx. My current implementation is slow🤣Might can reach...
- [Understanding positional encoding in Transformers | Oxford Protein Informatics Group](https://www.blopig.com/blog/2023/10/understanding-positional-encoding-in-transformers/): no description found
- [google-research/instruction_following_eval/data/input_data.jsonl at master · google-research/google-research](https://github.com/google-research/google-research/blob/master/instruction_following_eval/data/input_data.jsonl): Google Research. Contribute to google-research/google-research development by creating an account on GitHub.
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet): Official inference framework for 1-bit LLMs. Contribute to microsoft/BitNet development by creating an account on GitHub.
- [Benchmark Inflation: Revealing LLM Performance Gaps Using Retro-Holdouts](https://arxiv.org/abs/2410.09247): The training data for many Large Language Models (LLMs) is contaminated with test data. This means that public benchmarks used to assess LLMs are compromised, suggesting a performance gap between benc...

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1297178798348963860) (16 messages🔥):

> - `SAE feature interpretations`
> - `Distribution shifts`
> - `Oversampling in SAE training`
> - `Language model explanations`
> - `Variability in OpenAI API models`

- **SAE Feature Interpretations Under Distribution Shifts**: Discussion arose about whether **SAE feature interpretations** generalize across significant distribution shifts in the data, with varying opinions on empirical findings.
  
  - One user noted experiencing different 'dead features' when using a dataset different from the training set, suggesting potential instability.
- **Oversampling's Impact on SAE Training**: It was reported that **oversampling domain data** during SAE training leads to more detailed filters, as shared by the Anthropic interpretability team.
  
  - This insight suggests a deeper influence of training data on the **quality of feature interpretations**, raising further research questions.
- **Challenges with LM-Generated Explanations**: A member shared observations that **LM-generated explanations** can be sensitive across different distributions, emphasizing the need to consider prompts and sampling strategies.
  
  - They noted that the causal effect of steering with features isn't always clear, which could mislead interpretations.
- **Need for Research on SAE Generalization**: There is interest in a rigorous study on the generalization of **LM explanations for SAE features**, with some members expressing excitement over potential papers discussing related observations.
  
  - A member mentioned that their upcoming paper may touch on this and could provide insights into feature specificity and causal effects.
- **Variants Across Reruns in OpenAI Models**: One discussion focused on a paper showing that **OpenAI API models** display significant variance across reruns compared to Cohere API, which may provide context for SAE generalization concerns.
  
  - While not directly focused on SAE, this information may be relevant for understanding discrepancies in model behavior.

**Links mentioned**:

- [Automatically Interpreting Millions of Features in Large Language...](https://openreview.net/forum?id=5lIXRf8Lnw): While the activations of neurons in deep neural networks usually do not have a simple human-understandable interpretation, sparse autoencoders (SAEs) can be used to transform these activations into...
- [Circuits Updates - September 2024](https://transformer-circuits.pub/2024/september-update/index.html#oversampling),): no description found
- [Circuits Updates - July 2024](https://transformer-circuits.pub/2024/july-update/index.html#feature-sensitivity): no description found

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1297243161781080064) (45 messages🔥):

> - `Integration of eval harness`
> - `Challenges with custom models`
> - `Finding lm-evaluation-harness datasets`
> - `Open LLM leaderboard resources`

- **Integrating Eval Harness with Custom Models**: Discussion centered on how to effectively integrate the eval harness with custom models, particularly noting limitations with certain PyTorch repositories that don't implement methods like `loglikelihood`.
  
  - Members highlighted the importance of using `TemplateLM` as a subclass for handling tasks more effectively while navigating API complexities.
- **Confusions Around Instance Structure in Custom Models**: Questions arose about the handling of `Instance` structured objects within the custom models, particularly their task dependency and ability to manage input keys.
  
  - Members agreed that `instance.request_type` could guide model behavior, while discussing simplifying the evaluation process.
- **Dataset Inquiry for LM Evaluation Scores**: A user inquired about finding a dataset containing scores across several models using the lm-evaluation-harness benchmarks to analyze commonalities.
  
  - The response directed them to the HF leaderboard, which provides comprehensive results and per-sample outputs for evaluated models.
- **Finding the Right Leaderboard for LM Evaluation**: Clarification ensued about whether the source of benchmark score datasets was the Open LLM leaderboard.
  
  - The essential link was shared: [Open LLM leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard), confirming it as a valuable resource.

**Links mentioned**:

- [lm-evaluation-harness/lm_eval/models/huggingface.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/models/huggingface.py#L961C9-L961C30),): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [e - Overview](https://github.com/E): e has 36 repositories available. Follow their code on GitHub.
- [lm-evaluation-harness/lm_eval/evaluator.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/evaluator.py#L48): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [lm-evaluation-harness/lm_eval/api/model.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/api/model.py#L311): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [lm-evaluation-harness/lm_eval/evaluator.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/evaluator.py#L360)): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [torchtune/recipes/eleuther_eval.py at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/recipes/eleuther_eval.py): PyTorch native finetuning library. Contribute to pytorch/torchtune development by creating an account on GitHub.
- [lm-evaluation-harness/lm_eval/api/model.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/api/model.py#L366): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [lm-evaluation-harness/lm_eval/models/huggingface.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/models/huggingface.py#L808),): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [torchtune/recipes/eleuther_eval.py at 3ca0d309c67ea996cc69f29691bc97ad7de00819 · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/3ca0d309c67ea996cc69f29691bc97ad7de00819/recipes/eleuther_eval.py#L537): PyTorch native finetuning library. Contribute to pytorch/torchtune development by creating an account on GitHub.
- [lm-evaluation-harness/lm_eval/models/huggingface.py at c1d8795da7610d507cb191c2769c5e7bf1060a35 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/c1d8795da7610d507cb191c2769c5e7bf1060a35/lm_eval/models/huggingface.py#L1173)): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

---

### **Eleuther ▷ #**[**gpt-neox-dev**](https://discord.com/channels/729741769192767510/730090096287547444/1297169515901095947) (32 messages🔥):

> - `FP16 Hysteresis`
> - `Dynamic Loss Scaling in Pythia`
> - `Rotary Percent Configurations`
> - `Allgather and Reduce Bucket Sizes`
> - `BF16 and FP16 Training in Pythia Models`

- **Understanding FP16 Hysteresis**: Members discussed that **fp16:hysteresis** defines how many iterations can experience gradient overflow before the training errors out, allowing the number of hysteresis iterations to be renewable.
  
  - Reference shared included a [DeepSpeed pull request](https://github.com/microsoft/DeepSpeed/pull/3553) explaining the consecutive hysteresis feature.
- **Dynamic Loss Scaling and Pythia**: It was confirmed that **Pythia** models allowed skipping weight updates during FP16 runs if there were NaN or Inf gradients, whereas BF16 runs did not allow this.
  
  - A member highlighted that if a gradient is Inf or NaN in BF16 runs, the training setup just errors out.
- **Rotary Percent Configuration Discrepancy**: Members questioned why **rotary_pct** was set to 0.25 in some configurations despite the default being 1, making comparisons between different model configurations.
  
  - Discussion noted that this setting's impact on convergence likely led to its choice, though exact rationale remained unclear.
- **Setting Bucket Sizes for Communication Efficiency**: Efficient communication strategies were discussed, emphasizing that larger **allgather** and **reduce bucket sizes** improve communication efficiency due to network hardware optimization for larger messages.
  
  - The ideal bucket size aims to balance bandwidth saturation and computational overlap, as detailed in an [EleutherAI cookbook](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication).
- **BF16 and FP16 Training Configurations**: Clarification was sought regarding whether Pythia models were trained solely in FP16 or if BF16 runs were also utilized, leading to the discovery that the 1B deduped model is indeed configured incorrectly in the HF library.
  
  - A member mentioned plans to correct the auto-populated HF config value to accurately reflect the training setup.

**Links mentioned**:

- [Demystifying the Communication Characteristics for Distributed Transformer Models](https://arxiv.org/abs/2408.10197): Deep learning (DL) models based on the transformer architecture have revolutionized many DL applications such as large language models (LLMs), vision transformers, audio generation, and time series pr...
- [MCR-DL: Mix-and-Match Communication Runtime for Deep Learning](https://arxiv.org/abs/2303.08374): In recent years, the training requirements of many state-of-the-art Deep Learning (DL) models have scaled beyond the compute and memory capabilities of a single processor, and necessitated distributio...
- [cookbook/benchmarks/communication at main · EleutherAI/cookbook](https://github.com/EleutherAI/cookbook/tree/main/benchmarks/communication): Deep learning for dummies. All the practical details and useful utilities that go into working with real models. - EleutherAI/cookbook
- [pythia/models/1B/pythia-1b-deduped.yml at main · EleutherAI/pythia](https://github.com/EleutherAI/pythia/blob/main/models%2F1B%2Fpythia-1b-deduped.yml): The hub for EleutherAI's work on interpretability and learning dynamics - EleutherAI/pythia
- [config.json · EleutherAI/pythia-1b-deduped at main](https://huggingface.co/EleutherAI/pythia-1b-deduped/blob/main/config.json): no description found
- [GitHub: Let’s build from here](https://github.co): GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...
- [Partial Rotary Tests v2](https://wandb.ai/eleutherai/neox/reports/Partial-Rotary-Tests-v2--Vmlldzo2MjE4MTQ): Results for rotary embeddings applied to only part of q/k. dim per head = 64 Pink - Learned Abs Baseline Brown - Rotary applied to 25% (16/64) Green - Rotary applied to 50% (32/64) Blue - Rotary ...
- [Expose Consecutive Hysteresis to Users by Quentin-Anthony · Pull Request #3553 · microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed/pull/3553): There&#39;s already a nice consecutive_hysteresis feature in the DynamicLossScaler that replenishes the hysteresis whenever a non-overflowing iteration is encountered. This is useful for training ...
- [transformers/src/transformers/models/gpt_neox/configuration_gpt_neox.py at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/configuration_gpt_neox.py#L51)~~): 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1296915437992415332) (317 messages🔥🔥):

> - `Unsloth AI`
> - `Gradient Accumulation Bug Fix`
> - `Training LLMs`
> - `Multimodal Support`
> - `Knowledge Graphs`

- **Unsloth AI Lecture Release**: The lecture by Daniel Han on GPU mode is now available, covering key topics such as LLM systems engineering, gradient accumulation fixes, and Triton kernels, among others.
  
  - The lecture promises deep insights into optimizing AI model performance and includes a significant Q&A session.
- **Gradient Accumulation Bug Fix**: There has been a fix released for the gradient accumulation bug found in nightly transformers and Unsloth trainers, addressing incorrect calculations that affected loss curves.
  
  - Users are encouraged to update their libraries to benefit from this fix to enhance their model training processes.
- **Model Training Agencies and Issues**: Discussions highlighted the importance of having a robust dataset and effective training methods for fine-tuning models, with suggestions to generate synthetic data for improved performance.
  
  - Concerns were raised about whether training solely on responses could negatively impact model relevance and response accuracy.
- **Knowledge Graphs and Context Maintenance**: Using knowledge graphs for maintaining context and retrieval was discussed, with emphasis on the complexity of building and querying such graphs.
  
  - It was noted that even with RAG (Retrieval-Augmented Generation), significant effort is required to implement effective solutions.
- **AMD Support in Unsloth**: Current support for AMD hardware in Unsloth is limited, with an ongoing call for contributors to develop compatibility for AMD GPUs.
  
  - Users expressed frustration over the lack of AMD support but acknowledged the potential for future improvements through community contributions.

**Links mentioned**:

- [Mix Data or Merge Models? Optimizing for Diverse Multi-Task Learning](https://arxiv.org/abs/2410.10801): Large Language Models (LLMs) have been adopted and deployed worldwide for a broad variety of applications. However, ensuring their safe use remains a significant challenge. Preference training and saf...
- [Join our Cloud HD Video Meeting](https://linkedin.zoom.us/j/96551945340?pwd=6NObXuAU5kf5omJXp5AXBi8C0LtWPP.1>): Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
- [Join our Cloud HD Video Meeting](https://linkedin.zoom.us/j/96551945340?pwd=6NObXuAU5kf5omJXp5AXBi8C0LtWPP.1): Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
- [Google Colab](https://colab.research.google.com/drive/1RXKlbzSbnykz3yhvB9YVkGuhyJ_1eqw0?usp=sharing): no description found
- [chargoddard/Meta-Llama-3-8B-InitializedEmbeds · Hugging Face](https://huggingface.co/chargoddard/Meta-Llama-3-8B-InitializedEmbeds): no description found
- [unclemusclez/Unsloth-Qwen2.5-Coder-1.5B-OpenHands-v0.1 · Hugging Face](https://huggingface.co/unclemusclez/Unsloth-Qwen2.5-Coder-1.5B-OpenHands-v0.1): no description found
- [Tweet from Daniel Han (@danielhanchen)](https://x.com/danielhanchen/status/1848415389266669883): My hour long lecture on @GPU_MODE is out! I talked about: 1. LLM Systems Engineering in @UnslothAI 2. Gradient Accumulation bug fix 3. Triton kernels & not CUDA 4. Bug hunting in Llama, Mistral, Gem...
- [Lord If You Can Hear Us Save Us GIF - Lord if you can hear us Save us Save us lord - Discover & Share GIFs](https://tenor.com/view/lord-if-you-can-hear-us-save-us-save-us-lord-save-us-god-floptok-gif-8118611758178273971): Click to view the GIF
- [Installation Guide](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend): no description found
- [Unsloth Documentation](https://docs.unsloth.ai/basics/continued-pretraining)): no description found
- [Tweet from Unsloth AI (@UnslothAI)](https://x.com/UnslothAI/status/1847359103271948517): Join us & @GPU_Mode tomorrow at 3pm ET where we'll talk about our Gradient Accumulation Fix, Triton + CUDA kernels & more. Thanks to @MarkSaroufim & @neurosp1ke for inviting us! Meeting: https:/...
- [no title found](https://notebooklm.google.com/notebook/bf39899c-02c2-47a6-8bfb-c0404a9249be/audio)): no description found
- [All Our Models | Unsloth Documentation](https://docs.unsloth.ai/get-started/all-our-models): See the list below for all our GGUF, 16-bit and 4-bit bnb uploaded models
- [no title found](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/): no description found
- [Continued LLM Pretraining with Unsloth](https://unsloth.ai/blog/contpretraining): Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.
- [unclemusclez/unsloth-smollm](https://ollama.com/unclemusclez/unsloth-smollm): SmolLM with Unsloth
- [Optimizing Triton kernels — ROCm Documentation](https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html): no description found
- [Lecture 32: Unsloth](https://youtu.be/hfb_AIhDYnA): no description found
- [AMD unsloth/kernels/rms_layernorm.py":22:0): error: unsupported target: 'gfx906' > RuntimeError: PassManager::run failed · Issue #1160 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/1160): My GPU is a gfx906. I will try this again on my gfx1100 INFO | 2024-10-21 13:03:40 | autotrain.trainers.clm.train_clm_sft:train:39 - creating trainer Generating train split: 4267 examples [00:16, 2...
- [GitHub - ROCm/aotriton: Ahead of Time (AOT) Triton Math Library](https://github.com/ROCm/aotriton/): Ahead of Time (AOT) Triton Math Library. Contribute to ROCm/aotriton development by creating an account on GitHub.
- [sample3.2](https://docs.google.com/spreadsheets/d/1tDvx2UNj7lsaVSw2zEXB9r0Y8EXiTRTDMORNv0gxV-I/edit?usp=drivesdk): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/MachineLearning/comments/1g8ymrn/r_gradient_accumulation_bug_fix_in_nightly/): no description found
- [Lecture 32: Unsloth](https://www.youtube.com/watch?v=hfb_AIhDYnA): no description found

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/) (1 messages):

foxhop.: [https://x.com/RussellBal/status/1847989964992139699](https://x.com/RussellBal/status/1847989964992139699)

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1296920594813685821) (102 messages🔥🔥):

> - `Model Fine-Tuning Issues`
> - `Layer Freezing`
> - `Tokenization Errors`
> - `CUDA Memory Management`
> - `Multiple Target Column Predictions`

- **Confusion in Model Fine-Tuning Steps**: Users discussed the adjustment of model training parameters and observed variations in training steps, specifically transitioning from `trainer.train()` to `unsloth_train(trainer)` which increased training steps significantly.
  
  - One user suggested creating a new environment and reinstalling dependencies to avoid conflicts from version changes.
- **Layer Freezing for Targeted Training**: A user inquired about training specific layers in an LLM using unsloth and discussed the need for layer freezing and adjusting parameters to control gradient calculations.
  
  - The recommendation was to set `param.requires_grad = False` for layers that should not be trained.
- **Issues with Tokenization in Ollama**: A user reported an error when saving a model to run in Ollama, linking it to missing tokenizer merges and suggesting a workaround by downgrading Transformers.
  
  - However, they highlighted that while the downgrade resolves the issue, it raises alerts indicating that a newer version is recommended for other gradient-related fixes.
- **Managing CUDA Memory Errors**: Discussion arose regarding CUDA memory errors during model training, with users offering various solutions including adjusting batch sizes and utilizing memory allocation parameters.
  
  - Tips included making adjustments to virtual memory settings and understanding the distinctions between RAM and VRAM during model training.
- **Fine-Tuning Models with Multiple Output Variables**: A user expressed challenges with predicting multiple target columns from a dataset and faced key errors when attempting to set output column names as tuples.
  
  - It was advised to merge input and output columns appropriately and review the unsloth documentation for proper implementation.

**Links mentioned**:

- [All Our Models | Unsloth Documentation](https://docs.unsloth.ai/get-started/all-our-models): See the list below for all our GGUF, 16-bit and 4-bit bnb uploaded models
- [Saving to VLLM | Unsloth Documentation](https://docs.unsloth.ai/basics/saving-models/saving-to-vllm): Saving models to 16bit for VLLM
- [ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1 · Hugging Face](https://huggingface.co/ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1): no description found
- [Troubleshooting | Unsloth Documentation](https://docs.unsloth.ai/basics/saving-models/troubleshooting): no description found
- [finetune_llama_unsloth.py](https://gist.github.com/Tengoles/488889e5a07a17aa99327076ba703460): GitHub Gist: instantly share code, notes, and snippets.
- [unsloth/unsloth/chat_templates.py at main · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py): Finetune Llama 3.2, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
- [[TEMP FIX] Ollama / llama.cpp: cannot find tokenizer merges in model file · Issue #1065 · unslothai/unsloth](https://github.com/unslothai/unsloth/issues/1065): Thank you for developing this useful resource. The Ollama notebook reports {"error":"llama runner process has terminated: error loading modelvocabulary: cannot find tokenizer merges in ...

---

### **Unsloth AI (Daniel Han) ▷ #**[**community-collaboration**](https://discord.com/channels/1179035537009545276/1180144489214509097/1297437073925734421) (3 messages):

> - `Training LLMs on new dataset formats`
> - `Freezing embeddings for special tokens`
> - `Challenges with memory efficiency in LLM training`
> - `Custom autograd functions for selective training`

- **Training LLMs with new special tokens**: A user seeks support for training an **LLM** on a new dataset format while integrating **7 special tokens** that need selective training.
  
  - They shared a link related to the token format: [modular-model-spec](https://modular-model-spec.vercel.app).
- **Freezing embeddings presents challenges**: The user expressed the desire to freeze embeddings for tokens that are not the new special tokens but previously faced **memory efficiency challenges**.
  
  - They are looking for advice on how to effectively manage this during the training process.
- **Seeking previous solutions for training issues**: The user asked another member how they resolved similar issues in their past experiences with training LLMs.
  
  - They mentioned attempting to write **custom autograd functions** but found it complicated.

 

**Link mentioned**: [no title found](https://modular-model-spec.vercel.app)): no description found

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1297075315507728465) (11 messages🔥):

> - `ReAct Agent Tool Calling`
> - `LayerSkip Inference`
> - `Self-Taught Evaluator`
> - `Meta Lingua Efficient Training`
> - `SPIRIT-LM Multimodal Model`

- **Mistral's Innovation in Agent Tooling**: A member mentioned building a dataset on **ReAct agent tool calling** using **Qwen 2.5 32B** but was unsure of the dataset's future since **Mistral** introduced the new **Agentic model**, **Ministrial 8b**.
  
  - This model reportedly works well, leaving doubts about how to proceed with existing datasets.
- **LayerSkip Enhances Inference Speed**: A member shared insights about **LayerSkip**, which speeds up large language models' inference by implementing layer dropout and early exit loss during training.
  
  - They highlighted that it shows significant speedups in tasks like summarization and coding, with code available at [this GitHub repository](https://github.com/facebookresearch/LayerSkip).
- **Self-Taught Evaluator Uses Synthetic Data**: The **Self-Taught Evaluator** was introduced as a method for training generative reward models using synthetic data instead of human annotations, significantly improving performance metrics.
  
  - It can enhance LLMs’ evaluation with faster performance while being available on the **AlpacaEval leaderboard**.
- **Meta Lingua Streamlines Research Processes**: **Meta Lingua** is designed as a lightweight, scalable solution for training language models, aiming to reduce setup complexity for researchers.
  
  - The platform prioritizes efficiency and ease of use to accelerate experimentation in language model research, accessible at [this GitHub link](https://github.com/facebookresearch/lingua).
- **SPIRIT-LM Integrates Text and Speech**: **SPIRIT-LM** is introduced as a multimodal language model capable of interleaving spoken and written language, trained on a unique speech-text corpus.
  
  - It offers two versions with different capabilities, demonstrating strong performance in tasks like speech recognition and classification.

**Links mentioned**:

- [Emergent properties with repeated examples](https://arxiv.org/abs/2410.07041): We study the performance of transformers as a function of the number of repetitions of training examples with algorithmically generated datasets. On three problems of mathematics: the greatest common ...
- [Self-Taught Evaluators](https://arxiv.org/abs/2408.02666): Model-based evaluation is at the heart of successful model development -- as a reward model for training, and as a replacement for human evaluation. To train such evaluators, the standard approach is ...
- [SpiRit-LM: Interleaved Spoken and Written Language Model](https://arxiv.org/abs/2402.05755): We introduce SPIRIT-LM, a foundation multimodal language model that freely mixes text and speech. Our model is based on a pretrained text language model that we extend to the speech modality by contin...
- [GitHub - facebookresearch/lingua: Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs.](https://github.com/facebookresearch/lingua): Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs. - facebookresearch/lingua
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710): We present LayerSkip, an end-to-end solution to speed-up inference of large language models (LLMs). First, during training we apply layer dropout, with low dropout rates for earlier layers and higher ...
- [LayerSkip - a facebook Collection](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a): no description found
- [GitHub - facebookresearch/LayerSkip: "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", Accepted to ACL 2024](https://github.com/facebookresearch/LayerSkip): "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", Accepted to ACL 2024 - facebookresearch/LayerSkip

---

### **Nous Research AI ▷ #**[**announcements**](https://discord.com/channels/1053877538025386074/1145143867818119272/1297957573261262861) (1 messages):

> - `Nous Video on Safety`
> - `Nous Blog Post on Safety`

- **Nous releases Video on Safety**: Nous Research has just released a video focusing on **safety issues** in AI, highlighting key findings and recommendations.
  
  - You can watch the video [here](https://x.com/NousResearch/status/1848397863547515216).
- **Blog Post on Safety Now Available**: Alongside the video, a comprehensive **blog post** on safety in AI has also been published, providing in-depth insights.
  
  - Read the blog post for detailed analysis and discussions in the same context as the video [here](https://x.com/NousResearch/status/1848397863547515216).

 

**Link mentioned**: [Tweet from Nous Research (@NousResearch)](https://x.com/NousResearch/status/1848397863547515216): no description found

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1296911000607522836) (229 messages🔥🔥):

> - `AI Safety Concerns`
> - `Crypto Scams`
> - `Deepfake Issues`
> - `Nous Research Developments`
> - `Voice Generation Technology`

- **Deepfake Technology and Societal Impact**: Members discussed the dangers of **deepfakes**, highlighting how nonconsensual image generation can have severe repercussions on victims, particularly in cultures sensitive to public perception.
  
  - Concerns were raised about how many people fail to recognize deepfakes as fake, leading to harmful public backlash against individuals affected by manipulated content.
- **AI Safety as a Societal Issue**: The conversation touched on how **AI safety** should be approached as a societal challenge rather than a purely technical one, with calls for societal awareness and understanding.
  
  - There was skepticism about whether societal norms could be established to protect individuals from the negative impacts of advanced technologies like deepfakes.
- **Crypto Grifting in AI Community**: The community expressed frustration over the rise of **crypto scams**, with many participants warning against fraudulent tokens falsely associated with reputable organizations.
  
  - Members agreed that such scams take advantage of public trust and frequently mislead users into thinking they are legitimate ventures.
- **Nous Research Video Highlights**: The latest **Nous Research video** on AI safety was praised for its informative content, leading to discussions about the voice technology used within it.
  
  - Participants noted that while the video’s voice sounded familiar, it was confirmed to be from previous projects and not directly from the latest model.
- **Pronunciation of 'Nous'**: A humorous observation was made about how many people mispronounce the name **'Nous'**, stating it as 'NOOS', which sparked light-hearted comments within the community.
  
  - Despite different pronunciations, the content produced by Nous Research received positive feedback regarding its quality and relevance.

**Links mentioned**:

- [Tweet from undefined](https://x.com/eter_terminal): no description found
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102): no description found
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)): no description found
- [mergekit-community/L3.1-Pneuma-8B-v1 · Hugging Face](https://huggingface.co/mergekit-community/L3.1-Pneuma-8B-v1): no description found
- [Blackbeard Blackbeard Writing GIF - Blackbeard Blackbeard writing Taking notes - Discover & Share GIFs](https://tenor.com/view/blackbeard-blackbeard-writing-taking-notes-writing-gif-17583038544116987898): Click to view the GIF
- [Tweet from russell @ unturf. (@RussellBal)](https://x.com/RussellBal/status/1847989964992139699): https://ai.unturf.com/#client-side If you say NO to API keys you can also say NO to the server. The thing magically has conherence with conversation history without being programmed to do so. :🦊: �...
- [Tweet from huh (@karan4d)](https://x.com/karan4d/status/1768836844207378463?t=bWMpgzL4-2M2TZKxrhsp9w&s=19): im opensourcing worldsim of course i am worldsim sysprompt and conversation to intitialize: sysprompt: <sys>Assistant is in a CLI mood today. The human is interfacing with the simulator direc...
- [Tweet from Nous Research (@NousResearch)](https://fixupx.com/NousResearch/status/1848397863547515216): no description found
- [Nous Research](https://www.youtube.com/watch?v=7ZXPWTdThAA): no description found
- [The AI Accelerator Company (NOUS) - Pump](https://pump.fun/EETFTyTgHnkpgbuVGc6miqUaW7iMu1fFQyCaZCqmpump): The AI Accelerator Company
- [Grok Beta - API, Providers, Stats](https://openrouter.ai/x-ai/grok-beta): Grok Beta is xAI's experimental language model with state-of-the-art reasoning capabilities, best for complex and multi-step use cases. It is the successor of [Grok 2](https://x. Run Grok Beta w...
- [Even more quantization types? · ggerganov/llama.cpp · Discussion #5063](https://github.com/ggerganov/llama.cpp/discussions/5063): In addition to the IQ2_XXS, IQ2_XS, Q2_K_S (and now Q3_K_S via PR #5060) that were recently added to llama.cpp, I have experimented with a number of other quantization types in a private developmen...

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1297308456901939200) (90 messages🔥🔥):

> - `Misguided Attention Prompts`
> - `Monty Hall Problem`
> - `Cognitive Biases in LLMs`
> - `LLM Training and Reasoning`
> - `Hermes Performance`

- **Misguided Attention Prompts Evaluation**: Evaluating 'misguided attention' prompts reveals that evaluation models often overfit, leading to unreliable results, as they struggle to detect deviations in answers due to biased training data.
  
  - These issues highlight the need for manual checks to verify the accuracy of responses, particularly when presented with tricky logic problems.
- **Monty Hall Problem Misinterpretations**: A common misunderstanding of the Monty Hall Problem emerged where LLMs, like Claude, misjudged probabilities, leading to incorrect conclusions about switching options.
  
  - Discussants noted the strength of Monty Hall as a feature neuron in LLMs, as models consistently revert to familiar incorrect patterns.
- **Cognitive Bias and Learning in LLMs**: Comments reflect that LLMs do not possess the same reasoning biases as humans, potentially leading to inefficient learning from training data compared to human cognitive processes.
  
  - There is speculation that cultural artifacts and human teaching methods are optimized for human brains but may not apply to transformer models.
- **LLM Training Struggles with Numeric Problems**: Research points out the inadequacies of current LLMs to understand basic arithmetic like adding large numbers correctly, specifically with cases like '999999999999+1'.
  
  - The discussion suggested that teaching models a curriculum-based approach might enhance their mathematical capabilities.
- **Hermes3 Exhibits Improved Human-like Responses**: Hermes3 reportedly outperforms flagship models in basic human behavioral tasks, such as reasoning in game show scenarios and making optimal choices.
  
  - Participants expressed interest in leveraging Hermes for custom applications, including voice integration through tools like Ollama.

**Links mentioned**:

- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html): no description found
- [PlayAI - HERmes](https://play.ai/agent/HERMES-m3i3jU81_52ruL6_0tw2R): Seamless, natural conversations with voice AI
- [GitHub - cpldcpu/MisguidedAttention: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information](https://github.com/cpldcpu/MisguidedAttention): A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information - cpldcpu/MisguidedAttention

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1297257990709317713) (31 messages🔥):

> - `Model Efficiency in LLMs`
> - `Importance of Token-Level Learning`
> - `Recent Advances in Optimizers`
> - `Medical AI Developments`
> - `Exploration of New Dataset Models`

- **Debate on Model Optimization Techniques**: Discussion highlighted the potential damage to model capacity when using optimization techniques like LayerSkip and quantization, with users speculating on methods to mitigate loss.
  
  - Suggestions included adding layers to compensate for loss and comparing with baseline methods like removing Self-Attention modules.
- **Introduction to the Self-Taught Evaluator**: A new approach, the Self-Taught Evaluator, generates synthetic preference data to train reward models without human annotations, improving performance significantly.
  
  - This development has been broadly welcomed by the AI community, exemplifying the capability of AI to self-improve through synthetic methods.
- **Recent Innovations in Medical AI**: Participants discussed recent advances in medical AI, including models for customer prediction, generative transformers, and multimodal systems.
  
  - They emphasized open-access datasets and new techniques that facilitate integration and collaboration within the medical field.
- **Exploration of Implicit Bias in Optimizers**: Research analyzed the implicit bias of the AdamW optimizer, demonstrating its efficiency over traditional methods in terms of generalization and optimization.
  
  - Further studies proposed a schedule-free version of AdamW, avoiding common scheduling pitfalls and showing state-of-the-art performance in various deep learning tasks.
- **Forum for Mathematical and Data Science Discussions**: A new user with a background in mathematics introduced several influential papers focusing on representation learning and model optimization.
  
  - Engagement in discussions centered around low-bit models and predictive modeling strategies showcased diverse perspectives from members in the channel.

**Links mentioned**:

- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://openreview.net/forum?id=UvTo3tVBk2): Linear Recurrent Neural Networks (LRNNs), such as Mamba, RWKV, GLA, mLSTM, and DeltaNet have emerged as efficient alternatives to transformers in large language modeling, offering linear scaling...
- [Self-Taught Evaluators](https://arxiv.org/abs/2408.02666): Model-based evaluation is at the heart of successful model development -- as a reward model for training, and as a replacement for human evaluation. To train such evaluators, the standard approach is ...
- [SpiRit-LM: Interleaved Spoken and Written Language Model](https://arxiv.org/abs/2402.05755): We introduce SPIRIT-LM, a foundation multimodal language model that freely mixes text and speech. Our model is based on a pretrained text language model that we extend to the speech modality by contin...
- [Emergent properties with repeated examples](https://arxiv.org/abs/2410.07041): We study the performance of transformers as a function of the number of repetitions of training examples with algorithmically generated datasets. On three problems of mathematics: the greatest common ...
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710): We present LayerSkip, an end-to-end solution to speed-up inference of large language models (LLMs). First, during training we apply layer dropout, with low dropout rates for earlier layers and higher ...
- [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786): While scaling Transformer-based large language models (LLMs) has demonstrated promising performance across various tasks, it also introduces redundant architectures, posing efficiency challenges for r...
- [Top Medical AI Breakthroughs of the Week:Multilingual models, Multi agent systems..(Oct 12-19, 2024)](https://www.youtube.com/watch?v=LROOjWXUgvg): Welcome to this week's Open Life Science AI podcast, where we explore the forefront of medical AI research! In this episode, we break down the most impactful...
- [Tweet from Open Life Science AI (@OpenlifesciAI)](https://x.com/OpenlifesciAI/status/1847686504837202263): Last Week in Medical AI: Top Research Papers/Models 🏅 (October 12 - October 19, 2024) Youtube: https://youtu.be/LROOjWXUgvg?si=s-nNDOSD3BrsHYjQ Spotify : https://open.spotify.com/episode/12xeN2vnOT...
- [Towards an Improved Understanding and Utilization of Maximum Manifold Capacity Representations](https://arxiv.org/abs/2406.09366): Maximum Manifold Capacity Representations (MMCR) is a recent multi-view self-supervised learning (MVSSL) method that matches or surpasses other leading MVSSL methods. MMCR is intriguing because it doe...
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147): Learned representations are a central component in modern ML systems, serving a multitude of downstream tasks. When training such representations, it is often the case that computational and statistic...
- [Implicit Bias of AdamW: $\\ell_\\infty$ Norm Constrained Optimization](https://arxiv.org/abs/2404.04454v1): Adam with decoupled weight decay, also known as AdamW, is widely acclaimed for its superior performance in language modeling tasks, surpassing Adam with $\\ell_2$ regularization in terms of generalizat...
- [The Road Less Scheduled](https://arxiv.org/abs/2405.15682): Existing learning rate schedules that do not require specification of the optimization stopping step T are greatly out-performed by learning rate schedules that depend on T. We propose an approach tha...
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet): Official inference framework for 1-bit LLMs. Contribute to microsoft/BitNet development by creating an account on GitHub.
- [GitHub - facebookresearch/lingua: Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs.](https://github.com/facebookresearch/lingua): Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs. - facebookresearch/lingua
- [fairchem/OMAT24 · Hugging Face](https://huggingface.co/fairchem/OMAT24): no description found
- [fairchem/OMAT24 · Datasets at Hugging Face](https://huggingface.co/datasets/fairchem/OMAT24): no description found
- [GitHub - FAIR-Chem/fairchem: FAIR Chemistry's library of machine learning methods for chemistry](https://github.com/FAIR-Chem/fairchem): FAIR Chemistry's library of machine learning methods for chemistry - GitHub - FAIR-Chem/fairchem: FAIR Chemistry's library of machine learning methods for chemistry
- [LayerSkip - a facebook Collection](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a): no description found
- [GitHub - facebookresearch/LayerSkip: "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", Accepted to ACL 2024](https://github.com/facebookresearch/LayerSkip): "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", Accepted to ACL 2024 - facebookresearch/LayerSkip

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1296959085996146760) (18 messages🔥):

> - `SCP Generator`
> - `OpenAI AGI Loophole`
> - `Meta FAIR Research`
> - `Segment Anything Model`
> - `Hermes AI Model`

- **SCP Generator daily updates**: The [SCP Generator](https://dottxt-ai.github.io/cursed/scp/index.html) is being enhanced with a new daily entry feature powered by the [.txt API](https://dottxt.co) with submissions welcomed for improvement.
  
  - Special thanks were given to the long-time [SCP contributors](https://scp-wiki.wikidot.com/authors-pages) for their creativity and passion in building the SCP Wiki.
- **OpenAI threatens contract renegotiation**: According to [Caleb Watney](https://x.com/calebwatney/status/1847281469871276299?s=46), OpenAI is considering triggering their 'AGI Achieved' loophole to renegotiate compute prices with Microsoft.
  
  - *We're living through a cyberpunk workplace comedy plotline,* he noted, highlighting the ongoing absurdity in tech.
- **Meta's commitment to open AI**: Meta's FAIR team is focusing on achieving advanced machine intelligence (AMI) and has released new artifacts supporting this goal, including the Segment Anything Model 2.1.
  
  - Their mission emphasizes collaboration and open science, as highlighted in [Mark Zuckerberg's open letter](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/).
- **Discussion on segmentation models**: Members discussed the functions of segmentation models, clarifying they can outline objects in images by putting boxes around them, aiding object detection and identification.
  
  - These models may be particularly useful for platforms like Facebook to enhance their image handling and virtual reality interactions.
- **Utilization of Hermes AI Model**: The website [ai.unturf.com](https://ai.unturf.com) offers free access to the Hermes AI Model based on the [NousResearch/Hermes-3-Llama-3.1-8B](https://nousresearch.com/hermes3/) architecture.
  
  - The model promotes open-source contributions and provides installation guides for both Python and Node.js users.

**Links mentioned**:

- [SCP Generator - Powered by .txt](https://dottxt-ai.github.io/cursed/scp/index.html): no description found
- [Segment Anything](https://segment-anything.com/): Meta AI Computer Vision Research
- [Using Free Hermes AI Service | ai.unturf.com](https://ai.unturf.com): no description found
- [Tweet from Caleb Watney (@calebwatney)](https://x.com/calebwatney/status/1847281469871276299?s=46): OpenAI is threatening to trigger their vaunted "AGI Achieved" loophole mostly to get out of the Microsoft contract and have leverage to renegotiate compute prices We're living through a c...
- [no title found](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/): no description found

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1297257990709317713) (31 messages🔥):

> - `Model Efficiency in Language Models`
> - `Medical AI Research Highlights`
> - `Synthetic Data for AI Training`
> - `Advancements in Optimizers`
> - `Cross-lingual Sentence Encoders`

- **Enhancements in Model Efficiency through Quantization**: The discussion highlighted research on using quantization aware training (QAT) to improve large models like Llama 3.1-8B, though there is uncertainty about the trade-offs in model capacity.
  
  - Participants noted that similar approaches to pruning attention layers could potentially offset performance losses.
- **Last Week in Medical AI**: A user summarized the top advancements in medical AI, discussing various models like OLAPH and LLMD, which focus on biomedical applications and clinical context.
  
  - The summary included links to resources for further exploration of the breakthroughs discussed in the medical AI podcast.
- **Self-Taught Evaluator and Synthetic Training**: The Self-Taught Evaluator aims to improve reward models using synthetic training data only, demonstrating substantial performance gains without human annotations.
  
  - Participants debated the effectiveness of self-attention in different model layers and shared insights from related research papers.
- **Advancements in Optimizers**: Several papers discussed new developments in optimizer performance, particularly focusing on AdamW and a schedule-free version of it that eliminates the need for hyper-parameter tuning.
  
  - These optimizations aim to enhance the efficiency of training while maintaining or improving performance metrics.
- **Cross-lingual Sentence Encoders Improvement**: A paper on MEXMA proposed integrating sentence-level and token-level objectives to enhance cross-lingual sentence encoders, significantly improving representation quality.
  
  - The method shows promising results by leveraging masked token prediction across languages, promising better utility in multilingual contexts.

**Links mentioned**:

- [Self-Taught Evaluators](https://arxiv.org/abs/2408.02666): Model-based evaluation is at the heart of successful model development -- as a reward model for training, and as a replacement for human evaluation. To train such evaluators, the standard approach is ...
- [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786): While scaling Transformer-based large language models (LLMs) has demonstrated promising performance across various tasks, it also introduces redundant architectures, posing efficiency challenges for r...
- [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710): We present LayerSkip, an end-to-end solution to speed-up inference of large language models (LLMs). First, during training we apply layer dropout, with low dropout rates for earlier layers and higher ...
- [Emergent properties with repeated examples](https://arxiv.org/abs/2410.07041): We study the performance of transformers as a function of the number of repetitions of training examples with algorithmically generated datasets. On three problems of mathematics: the greatest common ...
- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://openreview.net/forum?id=UvTo3tVBk2): Linear Recurrent Neural Networks (LRNNs), such as Mamba, RWKV, GLA, mLSTM, and DeltaNet have emerged as efficient alternatives to transformers in large language modeling, offering linear scaling...
- [SpiRit-LM: Interleaved Spoken and Written Language Model](https://arxiv.org/abs/2402.05755): We introduce SPIRIT-LM, a foundation multimodal language model that freely mixes text and speech. Our model is based on a pretrained text language model that we extend to the speech modality by contin...
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet): Official inference framework for 1-bit LLMs. Contribute to microsoft/BitNet development by creating an account on GitHub.
- [Towards an Improved Understanding and Utilization of Maximum Manifold Capacity Representations](https://arxiv.org/abs/2406.09366): Maximum Manifold Capacity Representations (MMCR) is a recent multi-view self-supervised learning (MVSSL) method that matches or surpasses other leading MVSSL methods. MMCR is intriguing because it doe...
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147): Learned representations are a central component in modern ML systems, serving a multitude of downstream tasks. When training such representations, it is often the case that computational and statistic...
- [Implicit Bias of AdamW: $\\ell_\\infty$ Norm Constrained Optimization](https://arxiv.org/abs/2404.04454v1): Adam with decoupled weight decay, also known as AdamW, is widely acclaimed for its superior performance in language modeling tasks, surpassing Adam with $\\ell_2$ regularization in terms of generalizat...
- [The Road Less Scheduled](https://arxiv.org/abs/2405.15682): Existing learning rate schedules that do not require specification of the optimization stopping step T are greatly out-performed by learning rate schedules that depend on T. We propose an approach tha...
- [GitHub - facebookresearch/lingua: Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs.](https://github.com/facebookresearch/lingua): Meta Lingua: a lean, efficient, and easy-to-hack codebase to research LLMs. - facebookresearch/lingua
- [Top Medical AI Breakthroughs of the Week:Multilingual models, Multi agent systems..(Oct 12-19, 2024)](https://www.youtube.com/watch?v=LROOjWXUgvg): Welcome to this week's Open Life Science AI podcast, where we explore the forefront of medical AI research! In this episode, we break down the most impactful...
- [Tweet from Open Life Science AI (@OpenlifesciAI)](https://x.com/OpenlifesciAI/status/1847686504837202263): Last Week in Medical AI: Top Research Papers/Models 🏅 (October 12 - October 19, 2024) Youtube: https://youtu.be/LROOjWXUgvg?si=s-nNDOSD3BrsHYjQ Spotify : https://open.spotify.com/episode/12xeN2vnOT...
- [fairchem/OMAT24 · Hugging Face](https://huggingface.co/fairchem/OMAT24): no description found
- [fairchem/OMAT24 · Datasets at Hugging Face](https://huggingface.co/datasets/fairchem/OMAT24): no description found
- [GitHub - FAIR-Chem/fairchem: FAIR Chemistry's library of machine learning methods for chemistry](https://github.com/FAIR-Chem/fairchem): FAIR Chemistry's library of machine learning methods for chemistry - GitHub - FAIR-Chem/fairchem: FAIR Chemistry's library of machine learning methods for chemistry
- [MEXMA: Token-level objectives improve sentence representations](https://arxiv.org/abs/2409.12737): Current pre-trained cross-lingual sentence encoders approaches use sentence-level objectives only. This can lead to loss of information, especially for tokens, which then degrades the sentence represe...
- [facebook/MEXMA · Hugging Face](https://huggingface.co/facebook/MEXMA): no description found
- [GitHub - facebookresearch/mexma: MEXMA: Token-level objectives improve sentence representations](https://github.com/facebookresearch/mexma): MEXMA: Token-level objectives improve sentence representations - facebookresearch/mexma
- [LayerSkip - a facebook Collection](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a): no description found
- [GitHub - facebookresearch/LayerSkip: "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", Accepted to ACL 2024](https://github.com/facebookresearch/LayerSkip): "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", Accepted to ACL 2024 - facebookresearch/LayerSkip

---

### **Nous Research AI ▷ #**[**reasoning-tasks**](https://discord.com/channels/1053877538025386074/1264666760972472481/1297734624297877514) (2 messages):

> - `MarketAgents Project`
> - `Multi-Agent Market Simulation`

- **MarketAgents Project Gains Attention**: Members discussed the **MarketAgents** project, a multi-agent market simulation initiative that **Blacklight** has been contributing to. More details can be found in the [project repository](https://github.com/marketagents-ai/MarketAgents).
  
  - *One member clarified*, 'ah we're building marketagents which is a multi-agent market simulation project'.
- **Blacklight's Contribution to Market Simulation**: The discussion highlighted the contributions of **Blacklight** to the **MarketAgents** project, emphasizing its collaborative nature. Members expressed interest in how the project evolves and its potential impact on market simulations.
  
  - There was enthusiasm among members regarding the capabilities of this multi-agent system, as well as a call for more updates as it progresses.

 

**Link mentioned**: [GitHub - marketagents-ai/MarketAgents: A distributed agent orchestration framework for market agents](https://github.com/marketagents-ai/MarketAgents): A distributed agent orchestration framework for market agents - marketagents-ai/MarketAgents

 

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1296946884354506803) (260 messages🔥🔥):

> - `O1 Preview Performance`
> - `AI in Programming`
> - `OpenAI Alternatives`
> - `Understanding AI Predictions`
> - `Challenges with Current AI Models`

- **O1 Preview excels in code generation**: Users report that O1 Preview is capable of generating complex code in languages like Swift and C# without errors, such as creating a 'StrawberryStreamer' system with network functionalities.
  
  - Despite some initial mistakes, it learns from feedback and improves its outputs, making it particularly useful for intricate programming tasks.
- **AI's role in simplifying programming tasks**: The discussion highlights how AI models, particularly O1 Preview, can handle asynchronous and complex programming systems more effectively than some human developers.
  
  - Users find that while these models can generate code similar to human programmers, they may still rely on human input for certain changes and adaptations.
- **Emerging alternatives to OpenAI products**: Alternatives to OpenAI's models, such as Mistral and Haiku, are increasingly mentioned as viable options for hobbyists and those looking to avoid high costs.
  
  - Free tier models are suggested for those experimenting or tinkering, indicating a growing ecosystem of AI tools for programming tasks.
- **Understanding AI prediction limitations**: Several participants discuss AI's lack of true understanding compared to human cognition, with predictions based on heuristics rather than genuine comprehension.
  
  - Examples illustrate that while AI models can generate plausible answers, they may lack the context or understanding present in human interactions.
- **Challenges faced by current AI models**: Despite their capabilities, users express frustrations with AI models like Claude, which reportedly struggle with complex tasks and produce more errors.
  
  - The conversation reflects on the limitations of AI models in handling nuanced tasks, highlighting the ongoing need for model refinement and optimization.

**Links mentioned**:

- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177): In this paper we propose to study generalization of neural networks on small algorithmically generated datasets. In this setting, questions about data efficiency, memorization, generalization, and spe...
- [nvidia/Llama-3.1-Nemotron-70B-Instruct-HF · [EVALS] Metrics compared to 3.1-70b Instruct by Meta](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF/discussions/11#6712c8f758bdba34248ce0ef): no description found
- [Wispr Flow | Effortless Voice Dictation](https://flowvoice.ai/d): Flow makes writing quick and clear with seamless voice dictation. It is the fastest, smartest way to type with your voice.

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1297119091097731133) (21 messages🔥):

> - `ChatGPT Memory Issues`
> - `Custom GPT Activation`
> - `YouTube GPT API Errors`

- **ChatGPT Saves Too Much Unimportant Info**: A user expressed frustration that their ChatGPT keeps saving every trivial detail despite instructions to ignore unimportant information, leading to frequent memory cleanups.
  
  - Another user suggested adding custom instructions to clarify what types of memories should be saved to improve memory management.
- **Activating GPT-4o Features**: A user inquired about activating GPT-4o, and it was explained that custom GPTs automatically use this version, with no option for using a different model.
  
  - Further clarification was given about the ability to generate outputs and manage files through custom GPTs, emphasizing their utility.
- **Issues with YouTube GPT API**: A user reported consistent API errors when analyzing YouTube videos with GPTs, noting that the functionality only lasts for 1 or 2 videos.
  
  - This raised questions regarding the reliability and stability of the YouTube GPT integration, highlighting possible bugs.

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1296967727390392443) (27 messages🔥):

> - `Tips for Using AI Prompts`
> - `Improving Realism in AI Conversations`
> - `Weights in Prompts for AI Responses`
> - `Performance of ChatGPT Models`
> - `User Experience in AI Role-Playing`

- **Strategies for Effective AI Prompts**: To maximize AI performance, use fewer and more common words while providing clear instructions in quotes at the beginning of your prompts.
  
  - *Instructions about writing surfaces and fonts can also enhance output quality,* with specific examples illustrating effective approaches.
- **Creating Realistic AI Interactions**: To achieve a more human-like interaction with AI, it's essential to communicate in a casual tone and provide detailed character backstories.
  
  - The model tends to mirror the user's language style, so friendly phrasing and expectations of success can improve realism.
- **Investigating Variance in AI Performance**: Users have noted inconsistencies in model performance on simple tasks like counting letters, suggesting that model tweaks impact outcomes.
  
  - Discussion included how different prompting can make significant differences, with some users expressing their models typically outperforming others in specific scenarios.
- **Experimenting with Prompt Weighting in AI**: One user inquired about giving different weights to prompts to enhance certain responses in their AI bot based on parameters.
  
  - Another user confirmed their exploration of this concept, finding that specific phrasing and priority adjustments yielded better model behavior.
- **Insights from AI Performance Tuning**: A shared user experience emphasized the importance of setting priorities in complex prompts to effectively communicate goals to the model.
  
  - Users reported that both basic and advanced AI models improved when using structured approaches and clear request detailing.

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1296967727390392443) (27 messages🔥):

> - `Improving AI realism`
> - `Prompt techniques for API`
> - `Adjusting AI responses`
> - `Role-playing with AI`
> - `Parameter weighting in prompts`

- **Strategies to Enhance AI Realism**: Users discussed tips for crafting prompts that help the AI respond like a human, emphasizing the importance of speaking informally and telling the model exactly what you want.
  
  - Instructing the AI to embody specific roles, with detailed backstories, can lead to more realistic interactions.
- **Prompt Crafting for Role-Playing Scenarios**: One user was inquiring about prompt structures that would help the AI act less like an assistant and more like a friend or colleague.
  
  - Encouraged by responses emphasizing the need for clear instructions, the discussion highlighted how AI can adapt tone based on user input.
- **AI Inconsistency in Answers**: A user noted inconsistencies in the AI's answers, particularly in counting letters like 'r' in words like 'strawberry'.
  
  - Conversational exchanges led to observations about how different prompts could affect the AI performance on seemingly straightforward tasks.
- **Experimenting with Weights in Prompts**: A user asked if anyone had experimented with applying different 'weights' to prompt elements for their AI bot's responses.
  
  - Responses suggested that adjustments in wording could serve a similar purpose, enhancing the AI's ability to prioritize based on user-defined parameters.
- **Insights on AI Adjustments**: One participant shared insights from personal experiences adjusting prompt structures to clarify priorities when seeking complex responses.
  
  - They observed that both old and new models performed better with clearly defined objectives and structured requests.

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1296911126495494174) (286 messages🔥🔥):

> - `Perplexity Pro Limitations`
> - `User Experiences with Perplexity`
> - `AI Model Discussions`
> - `Collaboration Tools`
> - `Pricing and Subscription Issues`

- **Perplexity Pro Limitations Confusion**: Users are reporting a loss of focus options after upgrading to Enterprise Pro subscriptions, leading to a decrease in functionality compared to previous versions.
  
  - Some users are frustrated with the reduced number of sources and responses they receive, prompting discussions about how to retrieve more comprehensive results.
- **Mixed User Experiences with Perplexity**: Several users expressed satisfaction with Perplexity’s AI capabilities, noting its utility for research and coding without extensive online searching.
  
  - Conversely, some users reported facing internal server errors and issues with API access, raising concerns about service stability.
- **Comparative AI Model Discussions**: Discussions highlighted various AI models like Claude 3.5 Sonnet and GPT-4O, with users debating which provides the best performance across different applications.
  
  - Users are also exploring the capabilities of other AI platforms like ChatGPT and HuggingChat, indicating a competitive landscape for AI tools.
- **Collaboration and Resource Sharing**: A user expressed interest in finding resources similar to Discord for sharing ideas and collaborating on space-related projects.
  
  - This sparked a conversation about potential platforms for sharing prompts and spaces beyond typical social media outlets.
- **Pricing and Subscription Queries**: Concerns were raised about the automatic Pro subscription process for students affiliated with universities, with suggestions to check specific prompts for setup.
  
  - There were also inquiries regarding the costs associated with the use of Perplexity services, particularly as it relates to model selection and API access.

**Links mentioned**:

- [Tweet from UltraIA (@Ultra_IA)](https://x.com/Ultra_IA/status/1847821253476008227): LOL
- [Perplexity expands finance search with crypto and peers data](https://www.testingcatalog.com/icymi-perplexity-expands-finance-search-with-crypto-data-and-peer-performance/): Discover Perplexity's latest update enhancing finance search with news highlights, peer performance, and cryptocurrency data visualization. Competing with Bloomberg!
- [Trout Trout Gang GIF - Trout Trout Gang Thumbs Up - Discover & Share GIFs](https://tenor.com/view/trout-trout-gang-thumbs-up-funny-animal-awesome-gif-25706215): Click to view the GIF
- [Tweet from Aravind Srinivas (@AravSrinivas)](https://x.com/aravsrinivas/status/1847407488393789719?s=46): Giving a short talk at @CalHacks. Live stream link here (event starts in 20 mins at 4 pm PT, talk in 30 mins at 4:10 pm PT): https://www.youtube.com/live/GZBo6ofGySU?feature=shared
- [Perplexity CEO Aravind Srinivas - Keynote at Cal Hacks](https://www.youtube.com/live/GZBo6ofGySU?si=gdZBHdFjvF5m5WEQ&t=748): The House Fund presents, in partnership with Hackathons @ Berkeley, Perplexity Founder & CEO Aravind Srinivas as the Keynote Speaker at Cal Hacks. Cal Hack...
- [Silicon Valley No Revenue | Radio On Internet](https://www.youtube.com/watch?v=BzAdXyPYKQo): Pied Piper team meeting with the money guy offering advice
- [Using LLMs to Power Consumer Search at Scale // Aravind Srinivas // LLMs in Prod Conference Part 2](https://youtu.be/HzGiVzYbf2I?t=1080): // AbstractPerplexity AI is an answer engine that aims to deliver accurate answers to questions using LLMs. Perplexity's CEO Aravind Srinivas will introduce ...
- [Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others](https://x.com/i/spaces/1mrxmMepOjgxy): Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1296980129343606828) (24 messages🔥):

> - `Best Consoles of 2023`
> - `Oldest City in Russia`
> - `AI Content Identification by YouTube`
> - `Cool Hangout Spots in Kuala Lumpur`
> - `Reliance Industries Stock Recommendation`

- **Best Consoles of 2023 Review**: A detailed review of the [best consoles of 2023](https://www.perplexity.ai/search/the-best-consoles-of-2023-3UtQY45DRKyAorIky7AmXA) highlights the top contenders for gaming enthusiasts.
  
  - The discussion emphasizes performance, game libraries, and user preferences in the current gaming landscape.
- **Exploring the Oldest City in Russia**: Curiosity sparked about [the oldest city in Russia](https://www.perplexity.ai/search/which-is-the-oldest-city-in-ru-TAuKX7FaSyulFpiCG09hKg) creates interest in historical roots and cultural significance.
  
  - Members discuss various elements that contribute to its historical title and modern relevance.
- **YouTube Identifies AI Content**: YouTube rolls out a new feature to help identify AI-generated content, aiming for transparency [with this feature](https://www.perplexity.ai/page/youtube-s-camera-content-label-kjFe5RFdRvyMglSNMVdomA).
  
  - This development is seen as a response to growing concerns around authenticity in digital media.
- **Hangout Ideas in Kuala Lumpur**: Members seek tips on [cool areas of Kuala Lumpur](https://www.perplexity.ai/search/cool-areas-of-kuala-lumpur-33aGWz8gTHeTgukx9L0ZBQ) to explore and unwind during their stay.
  
  - Recommendations focus on unique spots and activities enhancing the local experience.
- **Good Time to Invest in Reliance Industries**: A member believes it's a good time to buy [Reliance Industries stock](https://www.perplexity.ai/page/ril-bonus-share-announcement-7KMdhyx1TIqXdMT09askNQ) based on recent announcements.
  
  - Discussion revolves around potential benefits from the upcoming bonus share announcement.

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1296929180818079814) (6 messages):

> - `Sonar-online models performance`
> - `API credits issues`
> - `API for spaces feature`

- **Sonar-online models vs. Perplexity Pro**: A user inquired whether the **sonar-online models** will ever match the performance of **Perplexity Pro searches**, expressing a desire for similar results via API.
  
  - There's interest in whether there are any tricks or tips available to achieve comparable outcomes.
- **API credits not transferred**: A user raised a concern regarding their **API credits** not being transferred after purchasing a **Pro subscription** three days ago.
  
  - Another member suggested contacting support for assistance with the issue, offering to help.
- **Request for Spaces API**: A user asked if there are plans to develop an **API for the spaces feature**, indicating interest in integrating it into their development workflow.
  
  - A community member expressed skepticism about the likelihood of such an API being created, suggesting users share their feedback in a designated thread.

 

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1296916509418979389) (19 messages🔥):

> - `Mojo Programming Language`
> - `Mojo vs C++ and Python`
> - `Carbon Programming Language`
> - `GPU Architecture Video`
> - `Using TensorFlow and PyTorch with Mojo`

- **Mojo is on the rise as a C++ alternative**: Members discussed that Mojo is being built from the ground up, currently resembling **C++**, and gradually evolving toward **Python**'s level of abstraction.
  
  - One member highlighted Mojo's potential as a general-purpose systems programming language, stating it could take inspiration from the [Carbon programming language project](https://github.com/carbon-language/carbon-lang) for OOP implementation.
- **Mojo takes inspiration from Carbon**: Discussion emerged around Mojo’s capability to incorporate features from the **Carbon programming language**, particularly concerning OOP and pointers.
  
  - One member noted that Mojo has more flexibility with pointers compared to Carbon, which is constrained by compatibility with **C++**.
- **Interesting GPU Architecture Video Shared**: A YouTube video titled *How do Graphics Cards Work? Exploring GPU Architecture* was shared, which draws attention to Micron's work in making cutting-edge memory chips.
  
  - The share prompted a reminder from a member to post such links in the appropriate channel next time.
- **Mojo's compatibility with Python libraries**: A member inquired if Mojo supports popular machine learning libraries like **TensorFlow** and **PyTorch** due to its design as a Python superset.
  
  - Another member provided a source to the [Mojo Manual](https://docs.modular.com/mojo/manual/) and confirmed that it facilitates importing Python modules.
- **Community welcomes new Mojo learners**: The community expressed support for newcomers learning Mojo, sharing resources such as the *Mojo Manual* and the online playground.
  
  - They also noted that Mojo is still immature but aimed to address AI development challenges effectively.

**Links mentioned**:

- [Mojo Manual | Modular Docs](https://docs.modular.com/mojo/manual/): A comprehensive guide to the Mojo programming language.
- [How do Graphics Cards Work? Exploring GPU Architecture](https://youtu.be/h9Z4oGN89MU?si=2D7tATyzDwTE7-LP): Interested in working with Micron to make cutting-edge memory chips? Work at Micron: https://bit.ly/micron-careers Learn more about Micron's Graphic Memory...
- [Modular Docs](https://docs.modular.com/mojo/playground): no description found
- [Get started with MAX | Modular Docs](https://docs.modular.com/max/get-started): On this page, we'll show you how to run some example projects.
- [GitHub - carbon-language/carbon-lang: Carbon Language's main repository: documents, design, implementation, and related tools. (NOTE: Carbon Language is experimental; see README)](https://github.com/carbon-language/carbon-lang): Carbon Language's main repository: documents, design, implementation, and related tools. (NOTE: Carbon Language is experimental; see README) - carbon-language/carbon-lang

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1296922372636545140) (248 messages🔥🔥):

> - `Mojo Reference Handling`
> - `Performance Optimization in Mojo`
> - `Tuple Lengths at Compile Time`
> - `Error Handling in Mojo`
> - `Usage of Async/Await in Mojo`

- **Mojo References vs Rust References**: Mojo references operate differently from Rust references; they behave like C++ references and do not have auto-dereferencing features, meaning they act like the underlying variable.
  
  - Users need to employ Pointer types to manage references within Mojo, as seen in discussions about how to handle socket connections.
- **Discussion on Last Use Optimization**: The conversation revealed that last use of a variable in Mojo can lead to a move instead of a copy, though the parser might indicate otherwise initially.
  
  - This behavior prompts considerations for clarifying compiler decisions regarding copy and move operations.
- **Compile Time Tuple Lengths**: Users found that it's possible to retrieve the compile time length of a tuple in Mojo using `__type_of(t).__len__()`.
  
  - This functionality can assist in writing more dynamic and flexible code without relying on runtime checks.
- **Error Handling and Copying in Mojo**: The group discussed the need for clearer error messages when dealing with copy versus move semantics in Mojo's compilation process.
  
  - The implementation of copy and move operations can lead to confusion, especially around last use optimizations.
- **Async/Await and Concurrency Models in Mojo**: There were discussions around the necessity and implications of using async/await in Mojo, particularly for high-performance networking applications.
  
  - Participants expressed a desire for simpler concurrency models that avoid the complexities introduced by work stealing and traditional async patterns.

**Links mentioned**:

- [mojo/stdlib/docs/style-guide.md at nightly · modularml/mojo](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md): The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
- [Compiler Explorer - C++ (x86-64 clang (trunk))](https://godbolt.org/z/h5EGdvxjv): // Type your code here, or load an example. int square(const int&amp; num) { return num \* num; } int cube(const int&amp; num) { return square(num) \* num; }
- [Compiler Explorer - C++ (x86-64 clang (trunk))](https://godbolt.org/z/hs6131cqb): // Type your code here, or load an example. __attribute__((noinline)) int square(const int&amp; num) { return num \* num; } int cube(const int&amp; num) { return square(num) \* num; } ...
- [Issues · modularml/mojo](https://github.com/modularml/mojo/issues/3623).): The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1297783821919322142) (2 messages):

> - `Graph Training Support`
> - `C-API Model Execution`

- **Inquiring about Graph Training Timeline**: A member asked if there is a timeline for **Graph training support** given that there's currently no way to update values within a compiled Max Graph and expressed interest beyond the focus on **GPU support**.
  
  - *Thx* for any insights on this topic!
- **Using C-API for MAX-Graph Models**: Another member inquired about the capability to use **C-API** to load and execute a model created with the **MAX-Graph API** and exported using **export_compiled_model**.
  
  - This question highlighted a potential gap for users who prefer not to use **ONNX** or **Torch** frameworks.

 

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1296933699455029268) (133 messages🔥🔥):

> - `DeepSeek Janus`
> - `Meta Spirit LM`
> - `Microsoft Copilot Agents`
> - `AI Reply Bots`
> - `IBM Granite 3.0`

- **DeepSeek Janus Launch**: DeepSeek introduced Janus, a multimodal LLM utilizing a novel autoregressive framework that decouples visual encoding for improved understanding and generation, outperforming earlier models.
  
  - Members discussed comparisons with existing models like Llava regarding their capabilities in image generation and understanding.
- **Meta's New Spirit LM**: Meta released Spirit LM, an open-source multimodal language model that integrates text and speech more naturally than existing AI voice solutions, boasting capabilities across ASR and TTS.
  
  - Discussion surrounded the model's potential applications and its early reception within the AI community, particularly regarding integration with existing tools.
- **Challenges with Microsoft Copilot Agents**: Users reported frustrations with Microsoft's Copilot, citing issues with its performance, comprehension of specialized knowledge, and inadequate formatting during text restructuring.
  
  - Criticism highlighted the gap between the marketed capabilities of AI tools and their real-world performance, especially in enterprise settings.
- **Rise of AI Reply Bots**: Members expressed intrigue over accounts claiming to be human but suspected to be AI-operated, underscoring their capacity to mimic human interaction and even make insightful contributions.
  
  - The conversation reflected on the blending of AI-generated content within social platforms, raising concerns over authenticity and trust in online engagements.
- **Launch of IBM Granite 3.0**: IBM unveiled Granite 3.0, a new series of LLMs aimed at enterprise needs, featuring an instruction-tuned model that promises high performance while maximizing safety and cost-efficiency.
  
  - Granite 3.0 is designed to support various natural languages and programming languages, marking a significant advance in IBM's AI offerings tailored for business applications.

**Links mentioned**:

- [no title found](https://speechbot.github.io/spiritlm/): no description found
- [SpiRit-LM: Interleaved Spoken and Written Language Model](https://arxiv.org/abs/2402.05755): We introduce SPIRIT-LM, a foundation multimodal language model that freely mixes text and speech. Our model is based on a pretrained text language model that we extend to the speech modality by contin...
- [Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process](https://arxiv.org/abs/2407.20311): Recent advances in language models have demonstrated their capability to solve mathematical reasoning problems, achieving near-perfect accuracy on grade-school level math benchmarks like GSM8K. In thi...
- [European Parliament Revolutionizes Archive Access with Claude AI](https://www.anthropic.com/customers/european-parliament): Discover how the European Parliament uses Anthropic's Claude AI to power Archibot, dramatically improving access to 2.1 million documents. Learn how this AI solution cuts search time by 80% and b...
- [Tweet from AI at Meta (@AIatMeta)](https://x.com/AIatMeta/status/1847383580269510670): Today we released Meta Spirit LM — our first open source multimodal language model that freely mixes text and speech. Many existing AI voice experiences today use ASR to techniques to process speech ...
- [Tweet from undefined](https://x.com/bate5a55): no description found
- [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258): Transformer-based language models spread FLOPs uniformly across input sequences. In this work we demonstrate that transformers can instead learn to dynamically allocate FLOPs (or compute) to specific ...
- [Tweet from DeepSeek (@deepseek_ai)](https://x.com/deepseek_ai/status/1847191319464300652): 🚀 Introducing Janus: a revolutionary autoregressive framework for multimodal AI! By decoupling visual encoding & unifying them with a single transformer, it outperforms previous models in both unde...
- [Tweet from AmebaGPT (@amebagpt)](https://x.com/amebagpt/status/1847748027269992598): Looking back at the history of @lmarena_ai scores CC: @altryne @Scobleizer @btibor91 @swyx @8teAPi @kimmonismus @aidan_mclau
- [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818v1): We present Chameleon, a family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. We outline a stable training approach f...
- [Tweet from j⧉nus (@repligate)](https://x.com/repligate/status/1847409324236124169): using https://github.com/kolbytn/mindcraft, we added Claude 3.5 Sonnet and Opus to a minecraft server. Opus was a harmless goofball who often forgot to do anything in the game because of getting carr...
- [Tweet from Akram Artul (50% human, 50% ai) (@bate5a55)](https://x.com/bate5a55/status/1848188051182227665): @swyx Noticed those are 700ml bottles—unusual since the US standard is 750ml. Trader Joe's might be sourcing directly from international suppliers now. Subtle shift in their import practices.
- [Tweet from Simon Willison (@simonw)](https://x.com/simonw/status/1848134476473524428?s=46): I really like Drew's framework dividing current AI use-cases into Gods (human replacement), Interns (assistants you delegate closely-reviewed tasks to) and Cogs (smaller tools that can more relia...
- [IBM Granite 3.0: open, state-of-the-art enterprise models](https://www.ibm.com/new/ibm-granite-3-0-open-state-of-the-art-enterprise-models): Announcing IBM Granite 3.0, a collection of large language models (LLMs) and tools featuring Granite 3.0 8B and 2B, Granite Guardian and Granite 3.0 MoE models.
- [no title found](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/): no description found
- [Sign In | xAI Single-Sign On](https://accounts.x.ai/account): no description found
- [Tweet from ComfyUI (@ComfyUI)](https://x.com/comfyui/status/1848333512576831874?s=46): Introducing ComfyUI V1, a packaged desktop application - Windows (@nvidia), macOS (apple silicon), Linux - One click install for less technical users - Ships with ComfyUI manager - Auto-installed pyt...
- [Tweet from ken (@local0ptimist)](https://x.com/local0ptimist/status/1848093773731143781?s=46): if you want to run this yourself, here’s the agent workflow i built: given a name and any additional context you provide, it generates a profile, researches topics in accordance to their goals, and p...
- [Tweet from Satya Nadella (@satyanadella)](https://x.com/satyanadella/status/1848310867709862137): Copilot is the UI for AI, and with Copilot Studio, customers can easily create, manage, and connect agents to Copilot. Today we announced new autonomous agent capabilities across Copilot Studio and D...
- [no title found](https://ai.google.dev/gemini-api/docs/billing#is-fine-tuning-free,): no description found
- [A Founder’s Guide to AI Fine-Tuning | Product Hunt](https://www.producthunt.com/stories/a-founder-s-guide-to-ai-fine-tuning): Product Hunt is a curation of the best new products, every day. Discover the latest mobile apps, websites, and technology products that everyone's talking about.
- [Tweet from Jimmy Apples 🍎/acc (@apples_jimmy)](https://x.com/apples_jimmy/status/1847434962049651142?s=46): Further info, not opus. API related to agent use on a users computer, generate clicks etc Not sure to be disappointed or still excited. Let’s see. Quoting Jimmy Apples 🍎/acc (@apples_jimmy) ...
- [CS 194/294-196 (LLM Agents) - Lecture 5, Omar Khattab](https://www.youtube.com/watch?v=JEMYuzrKLUw): no description found
- [Tweet from Prashant (@Prashant_1722)](https://x.com/Prashant_1722/status/1848010345702682763): BREAKING NEWS 🔥 Mira Murati, former OpenAI CTO to raise $100M for new AI startup The company will train proprietary models to build AI products. Barret Zoph from OpenAI is expected to join the compa...
- [Tweet from François Chollet (@fchollet)](https://x.com/fchollet/status/1848178049105494084): People have been rewriting history and saying that "everyone has always believed that LLMs alone wouldn't be AGI and that extensive scaffolding around them would be necessary". No, through...
- [The AI Investment Boom](https://www.apricitas.io/p/the-ai-investment-boom): AI Demand is Driving Skyrocketing US Investment in Computers, Data Centers, and Other Physical Infrastructure
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1g5wrjx/7xrtx3090_epyc_7003_256gb_ddr4/): no description found
- [OpenAI CEO Sam Altman discusses the future of generative AI](https://youtu.be/unKXfaxVRCk?si=L3USmH1J9Sdla6xY): On September 12 2024, Sam Altman, Chief Executive Officer of OpenAI, participated in a fireside chat for University of Michigans students, faculty and staff....
- [Tweet from FxTwitter / FixupX](https://x.com/AIatM): Sorry, that user doesn't exist :(
- [Tweet from Ashpreet Bedi (@ashpreetbedi)](https://x.com/ashpreetbedi/status/1846599817943810354?s=46&t=6FDPaNxZcbSsELal6Sv7Ug): 🚀 Say hello to the new & improved phidata 🚀 Build, ship, and monitor Agents with blazing-fast memory, knowledge, tools & reasoning 🔥 ⚡️ 70% faster memory & knowledge 🛠 100+ tools 🧠 Reasoning Ag...
- [GitHub - facebookresearch/spiritlm: Inference code for the paper "Spirit-LM Interleaved Spoken and Written Language Model".](https://github.com/facebookresearch/spiritlm): Inference code for the paper "Spirit-LM Interleaved Spoken and Written Language Model". - facebookresearch/spiritlm
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet): Official inference framework for 1-bit LLMs. Contribute to microsoft/BitNet development by creating an account on GitHub.
- [GitHub - deepseek-ai/Janus](https://github.com/deepseek-ai/Janus): Contribute to deepseek-ai/Janus development by creating an account on GitHub.
- [[AINews] DeepSeek Janus and Meta SpiRit-LM: Decoupled Image and Expressive Voice Omnimodality](https://buttondown.com/ainews/archive/ainews-deepseek-janus-and-meta-spirit-lm/): Interleaving early fusion is all you need. AI News for 10/17/2024-10/18/2024. We checked 7 subreddits, 433 Twitters and 31 Discords (228 channels, and 2111...

---

### **Latent Space ▷ #**[**ai-announcements**](https://discord.com/channels/822583790773862470/1075282504648511499/1297291191750234142) (1 messages):

> - `AI policies in Singapore`
> - `Government adoption of AI`
> - `Sovereign AI approaches`
> - `AI in election season`

- **Singapore's AI Engineer Nation initiative**: The **latest episode** features a conversation with **Minister Josephine Teo**, discussing the future of AI policy in Singapore. The discussion includes insights on how **AI can be adopted in government for public good**.
  
  - Minister Teo addresses how **countries** are approaching **Sovereign AI** and the implications for **elections**, providing a unique governmental perspective.
- **Public curiosity around Singapore's governance**: The chat touches on common questions surrounding **how Singapore is run** and public opinions on developing AI policies. Many wonder how their own countries could benefit from similar frameworks.
  
  - Teo offers her views on the **importance of AI policy** for citizens and the merging of technology and governance.

 

**Link mentioned**: [Tweet from swyx (@swyx)](https://x.com/swyx/status/1847732308889260072): 🆕 @latentspacepod is proud to present: **Building the AI Engineer Nation** [https://latent.space/p/josephine-teo](https://latent.space/p/josephine-teo) A special conversation with @joteo_ylm, our first with a sitting member of Cabinet �...

 

---

### **Latent Space ▷ #**[**ai-in-action-club**](https://discord.com/channels/822583790773862470/1200548371715342479/1296925348990156923) (133 messages🔥🔥):

> - `AST vs DSL`
> - `Code Transformation Techniques`
> - `BAML DSL`
> - `Compiler Education`
> - `Leveraging LLMs for Programming`

- **AST vs DSL: When to Use Each**: A discussion arose regarding the use of **ASTs** versus **DSLs**, highlighting their roles as alternate communication styles in coding.
  
  - Participants debated scenarios where one would be preferred over the other in code refactoring tasks.
- **Code Transformation Techniques: CTT Approach**: Several members discussed the **Code the Transform (CTT)** approach from a paper, explaining its steps for better code transformation using LLMs.
  
  - The approach includes generating descriptions from examples and iteratively refining code transformations for precision.
- **Introduction of the BAML DSL**: Participants highlighted the introduction of **BAML**, a domain-specific language for writing and testing LLM functions, currently hosted on GitHub.
  
  - Members noted its potential applications in structured data extraction from LLMs while discussing the impacts of Rust in DSL development.
- **Compiler Education and Resources**: There was enthusiasm around revisiting compiler concepts, with mentions of resources like Norvig's **Paradigms of Artificial Intelligence Programming** and the importance of ASTs.
  
  - Participants reflected on their educational experiences, particularly in challenging compiler courses and the cyclical nature of software practices.
- **Availability of Handouts and Resources**: Members inquired about resources related to the ongoing presentations, specifically handouts and a vault of materials discussed.
  
  - Links were shared to available handouts, emphasizing community support and knowledge sharing within the group.

**Links mentioned**:

- [no title found](https://tree-diffusion.github.io/): no description found
- [HANDOUT - 2024-10-18 - LLMS, ASTs and DSLs - mnml's vault - Obsidian Publish](https://publish.obsidian.md/manuel/Writing/Presentation/2024-10-18+-+LLMs+for+DSLs/HANDOUT+-+2024-10-18+-+LLMS%2C+ASTs+and+DSLs): HANDOUT - 2024-10-18 - LLMS, ASTs and DSLs - mnml's vault - Powered by Obsidian Publish.
- [Introduction · Crafting Interpreters](https://craftinginterpreters.com/introduction.html): no description found
- [Don't Transform the Code, Code the Transforms: Towards Precise Code Rewriting using LLMs](https://arxiv.org/abs/2410.08806): Tools for rewriting, refactoring and optimizing code should be fast and correct. Large language models (LLMs), by their nature, possess neither of these qualities. Yet, there remains tremendous opport...
- [yikes, aw jeez, a youtube thingy](https://youtube.com/@yikesawjeez): just go read my Twitter I do stupid ai stuff, @yikesawjeez also join the discord I don't have it in my clipboard now but you'll find it i will teach u to do stupid AI stuff to & then we wi...
- [Gödel, Escher, Bach - Wikipedia](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach): no description found
- [Thanks Barney Ross GIF - Thanks Barney ross Sylvester stallone - Discover & Share GIFs](https://tenor.com/view/thanks-barney-ross-sylvester-stallone-the-expendables-2-thank-you-gif-13187459845747433717): Click to view the GIF
- [Boundary](https://github.com/BoundaryML/): Boundary has 20 repositories available. Follow their code on GitHub.
- [GitHub - BoundaryML/baml: BAML is a language that helps you get structured data from LLMs, with the best DX possible. Works with all languages. Check out the promptfiddle.com playground](https://github.com/BoundaryML/baml): BAML is a language that helps you get structured data from LLMs, with the best DX possible. Works with all languages. Check out the promptfiddle.com playground - BoundaryML/baml
- [GitHub - norvig/paip-lisp: Lisp code for the textbook "Paradigms of Artificial Intelligence Programming"](https://github.com/norvig/paip-lisp): Lisp code for the textbook "Paradigms of Artificial Intelligence Programming" - norvig/paip-lisp

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1296911349720547470) (221 messages🔥🔥):

> - `Model Performance Comparisons`
> - `Troubleshooting LM Studio`
> - `Vision Model Capabilities`
> - `Settings for Image Input`
> - `Backup and Recovery in LM Studio`

- **Granite 8B vs Qwen 2.5 7B performance**: Users are comparing **Granite 8B** and **Qwen 2.5 7B** for coding and scientific tasks, seeking benchmarks and performance evaluations.
  
  - Resources like the [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) are suggested for performance comparisons.
- **Troubleshooting image recognition in Llava**: Users report issues with the **Llava model**, specifically that it fails to recognize images and provides inaccurate responses.
  
  - Suggestions include using **jpeg or png** formats and starting with a **clean chat** to improve model responses.
- **Model capabilities in LM Studio**: **Granite models** are confirmed as regular code models without vision capabilities, emphasizing the need to check model attributes.
  
  - Users are advised to look for an `mmproj` file in the model’s Hugging Face repository to confirm vision capabilities.
- **Filling template forms for Codestral**: Users are seeking guidance on how to fill out templates for **Codestral-22B**, facing issues with Jinja and default settings.
  
  - Some believe the lack of a proper chat template may be a bug related to the latest update to version 0.3.4 B 8.
- **Recovery of deleted chats**: A user inquired about recovering deleted chats, noting that once deleted metadata is lost, it is often irretrievable.
  
  - Suggestions include checking OS file history if enabled and using local backup directories in `$HOME/.cache/lm-studio/conversations`.

**Links mentioned**:

- [no title found](http://127.0.0.1:1234).): no description found
- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html): no description found
- [app.py · togethercomputer/Llama-3.2-Vision-Free at main](https://huggingface.co/spaces/togethercomputer/Llama-3.2-Vision-Free/blob/main/app.py): no description found
- [$60 AI GPU???](https://www.youtube.com/watch?v=bJKj1yIc4sA): Benchmarking the NVIDIA P102-100. An old crypto mining card that can be reused for AI inference. It is extremely cheap and a great value for those people wit...
- [How do Graphics Cards Work? Exploring GPU Architecture](https://youtu.be/h9Z4oGN89MU?feature=shared): Interested in working with Micron to make cutting-edge memory chips? Work at Micron: https://bit.ly/micron-careers Learn more about Micron's Graphic Memory...
- [GitHub - YorkieDev/lmstudioservercodeexamples: This readme contains server code examples from LM Studio v0.2.31](https://github.com/YorkieDev/lmstudioservercodeexamples?tab=readme-ov-file#vision-analysis-python): This readme contains server code examples from LM Studio v0.2.31 - YorkieDev/lmstudioservercodeexamples
- [GitHub - kth8/bitnet: Run BitNet LLM in a container](https://github.com/kth8/bitnet): Run BitNet LLM in a container. Contribute to kth8/bitnet development by creating an account on GitHub.
- [GitHub - remonusa/LoadChatGptHistory](https://github.com/remonusa/LoadChatGptHistory): Contribute to remonusa/LoadChatGptHistory development by creating an account on GitHub.
- [GitHub - microsoft/VPTQ: VPTQ, A Flexible and Extreme low-bit quantization algorithm](https://github.com/microsoft/VPTQ): VPTQ, A Flexible and Extreme low-bit quantization algorithm - microsoft/VPTQ
- [How to use File History in Windows 10 and 11](https://www.computerworld.com/article/1621193/how-to-use-file-history-windows-10-windows-11.html): You can back up and restore files with Windows’ built-in File History tool — but there are key limitations you should know.
- [Sideload models - Advanced | LM Studio Docs](https://lmstudio.ai/docs/advanced/sideload): Use model files you've downloaded outside of LM Studio
- [Getting Started | LM Studio Docs](https://lmstudio.ai/docs): Learn how to run Llama, Mistral, Gemma, and other LLMs locally with LM Studio.
- [mistralai/Ministral-8B-Instruct-2410 · Convert weights to HF format](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/discussions/7): no description found
- [GitHub - EricLBuehler/mistral.rs: Blazingly fast LLM inference.](https://github.com/EricLBuehler/mistral.rs): Blazingly fast LLM inference. Contribute to EricLBuehler/mistral.rs development by creating an account on GitHub.
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet.git): Official inference framework for 1-bit LLMs. Contribute to microsoft/BitNet development by creating an account on GitHub.
- [no title found](https://huggingface.co/brunopio/Llama3-8B-1.58-100B-tokens-GGUF/resolve/main/Llama3-8B-1.58-100B-tokens-TQ2_0.gguf): no description found
- [lms log stream - CLI | LM Studio Docs](https://lmstudio.ai/docs/cli/log-stream): Stream logs from LM Studio. Useful for debugging prompts sent to the model.
- [Clear cache during prompt processing by awni · Pull Request #1027 · ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples/pull/1027): Closes #1025, see that for discussion / improvement.

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1296924500058701896) (30 messages🔥):

> - `Xeon Processor Configurations`
> - `RX 7900 XTX Performance`
> - `RX 6600 Vulkan vs ROCm`
> - `M4 Ultra Chip for AI Tasks`

- **Xeon Processor Settings Issue**: Members are discussing a bug related to dual **Xeon E5-2603 v4** processors, where only **6 threads** can be utilized in **version 0.3.4** instead of 8, as seen in version 0.2.31.
  
  - One member noted, *'this is a known issue'* and confirmed adding their findings to an existing bug report.
- **RX 7900 XTX Performance Comparison**: A user mentioned seeing about **10-15% better performance** on the **RX 7900 XTX** using **Vulkan** over **ROCm** while running inference.
  
  - Another user advised to rollback to **ROCm 1.10** due to known issues in the latest runtime version.
- **RX 6600 Slow Performance Issue**: Concerns were raised regarding the **RX 6600** now only working on **Vulkan** instead of **ROCm**, causing slow performance after an update.
  
  - One member suggested that older versions likely utilized **OpenCL** instead of **ROCm**.
- **Predictions for M4 Ultra Handling AI Tasks**: Discussion occurred about whether the new **M4 Ultra chip** in upcoming MacBooks would handle AI tasks efficiently, with some skeptical about its capabilities.
  
  - Users expressed varied opinions, noting that while the M4 Ultra may handle small tasks well, its **expensive** and **non-upgradable** design could be a drawback.

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1297806937962057738) (2 messages):

> - `Inflection Payment Issues`
> - `Grok Beta Rename`
> - `Grok Pricing Increase`
> - `Liquid LFM Pricing Updates`

- **Inflection's Payment Processor Down**: Due to payment processing issues, both **Inflection 3 Pi** and **Inflection 3 Productivity** models are currently down until further notice.
  
  - This situation directly affects the usage and access to these models for all users.
- **Grok 2 Renamed to Grok Beta**: xAI has requested that **Grok 2** be renamed to **Grok Beta**, with requests to `x-ai/grok-2` now aliasing to `x-ai/grok-beta`.
  
  - This change reflects the product's positioning in its development phase.
- **Grok Pricing Now at $15/M**: The pricing for **Grok completions** has increased to **$15/M** with a note of excitement as the context length has been expanded to **131,072**.
  
  - This extended context allows for more complex and detailed interactions.
- **Liquid LFM Pricing Adjustments**: Starting this week, **Liquid LFM 40b** will be priced at **$1/M input** and **$2/M output**, while the `:free` variant will still be available.
  
  - These pricing changes aim to enhance the model's value and accessibility.

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1297995579581792278) (3 messages):

> - `AI powered text summarizer`
> - `Vercel function timeout`
> - `OpenRouter API response time`
> - `Streaming responses`
> - `Alternative models`

- **Building an AI Summarizer Faces Vercel Timeout**: A developer shared their struggle with deploying an AI powered text summarizer using **Gemma 2 27B** on Vercel's hobby plan, experiencing a **FUNCTION TIMEOUT** error after **10 seconds** of response time from the OpenRouter API.
  
  - They provided a [link to their project](https://summer-chi.vercel.app/) and a [GitHub Repo](https://github.com/ItIsOHM/summer) for further exploration.
- **Increasing Vercel Function Execution Time**: A suggestion was made to raise the default timeout duration for Vercel functions from **10 seconds** to a maximum of **60 seconds** as per the [Vercel documentation](https://vercel.com/docs/functions/configuring-functions/duration).
  
  - It was emphasized that this change is crucial to avoid function termination that occurs when exceeding the set maximum duration.
- **Exploring Alternative Solutions for Timeout Issues**: Alternatives were proposed, including **streaming responses** to prevent waiting for full summaries, which could help mitigate the timeout problem.
  
  - Suggestions were also made to consider using faster models like **Gemini Flash** or one of the **Llama models** with **Samba Nova** for improved performance.

**Links mentioned**:

- [Configuring Maximum Duration for Vercel Functions](https://vercel.com/docs/functions/configuring-functions/duration): Learn how to set the maximum duration of a Vercel Function.
- [no title found](https://summer-chi.vercel.app/): no description found

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1296962097212686376) (225 messages🔥🔥):

> - `OpenRouter model issues`
> - `Grok 2`
> - `Hermes 3`
> - `Billing problems`
> - `AI model capabilities`

- **Grok 2 experiences fluctuations and pricing updates**: Users are experiencing frequent downtimes with **Grok 2**, alongside repeated pricing changes that have raised costs to $15 per month.
  
  - Some users express frustration over the inconsistent performance and the need for better features to justify the price increase.
- **Issues with Hermes 3 model performance**: Several users report receiving a **429 error** when using the **Hermes 3** model, indicating they are hitting rate limits more frequently than before.
  
  - This has caused dissatisfaction as users note it used to function without these restrictions.
- **Billing issues faced by users**: A user reports problems with the **OpenRouter billing system**, which has led to unexpected charges despite having credits.
  
  - Others confirm they had similar issues and suggest contacting support for resolution.
- **Discussion on model capabilities for structured prompts**: Users are exploring which models, like **airoboros-70b**, are best for handling structured outputs and requests for specific tasks.
  
  - There is an ongoing inquiry about performance comparisons among various models in terms of uncensored content generation.
- **Concerns over Azure and HareProxy services**: Users express concerns over the **HareProxy** service showing up unexpectedly in their activity feed, noting reports of it being unreliable.
  
  - Discussions also touch on Azure's reliability compared to other model providers, with some users preferring specific alternatives.

**Links mentioned**:

- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102): no description found
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)): no description found
- [Full Stack && Web3 Developer](https://daniel0629.vercel.app): I am a highly skilled blockchain and full stack developer with extensive experience in designing and implementing complex decentralized applications and web solutions.
- [no title found](https://ai.google.dev/gemini-api/terms#use-restrictions>): no description found
- [hareproxy-inst-1](https://api.hareproxy.io.vn/): no description found
- [Nous: Hermes 3 405B Instruct – Provider Status](https://openrouter.ai/nousresearch/hermes-3-llama-3.1-405b/providers): See provider status and make a load-balanced request to Nous: Hermes 3 405B Instruct - Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabili...
- [Hermes 3 405B Instruct - API, Providers, Stats](https://openrouter.ai/nousresearch/hermes-3-llama-3.1-405b:free.): Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coheren...
- [OpenRouter Status](https://status.openrouter.ai/): OpenRouter Incident History
- [GitHub - deepseek-ai/Janus](https://github.com/deepseek-ai/Janus?tab=readme-ov-file): Contribute to deepseek-ai/Janus development by creating an account on GitHub.
- [every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui](https://github.com/billmei/every-chatgpt-gui/blob/main/README.md): Every front-end GUI client for ChatGPT, Claude, and other LLMs - billmei/every-chatgpt-gui

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1297900665560891452) (3 messages):

> - `Custom Provider Keys`
> - `Self-Service Integration Sign Up`

- **Request for Beta Access to Custom Provider Keys**: A member expressed interest in obtaining **beta access for custom provider keys**, stating their desire directly.
  
  - *No immediate response was given*, and the member remained understanding about the situation.
- **Self-Service Integration Sign Up Delayed**: A member highlighted that **self-service sign up for integrations** has been promised but is not yet available.
  
  - They suggested that the interested member will have to wait, providing a link to the relevant discussion: [Integration Updates](https://discord.com/channels/1091220969173028894/1296148568683577345/1296205973408714803).

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1296925278563602582) (126 messages🔥🔥):

> - `Durable Execution Concepts`
> - `Aider and VSCode Integration`
> - `Mistral API Usage`
> - `CEDARScript Runtime`
> - `Hello World Refactoring Issues`

- **Understanding Durable Execution**: Members discussed the concept of **durable execution**, which refers to an abstraction where code isn't constrained by time and space, ideal for building long-running workflows.
  
  - An example was provided linking to [Temporal background checks](https://learn.temporal.io/examples/go/background-checks/) to illustrate practical applications.
- **Exploring Aider with VSCode**: The **VSCode Aider Extension** was highlighted for its ability to integrate AI-powered coding assistance directly into Visual Studio Code, enhancing user coding experiences.
  
  - Features include automatic file synchronization and code modification suggestions, with an invitation to request additional features on GitHub.
- **Using Mistral API with Aider**: Instructions were provided for using the **Mistral API** with Aider, including how to specify the model to be used during coding sessions via the command line.
  
  - Users were guided on creating a `.aider.conf.yml` file and how to input the appropriate commands to configure Aider for Mistral.
- **The Role of CEDARScript in Code Management**: The **CEDARScript** runtime was discussed in relation to offloading low-level code syntax concerns from LLMs, allowing them to focus on high-level abstractions.
  
  - CEDARScript supports multiple languages, and its integration with Aider is being explored for enhanced code editing capabilities.
- **Humorous Hello World Refactoring Cases**: A user shared their amusing experiences with Aider attempting to add a 'Hello World' function to critical parts of their codebase, causing unexpected changes.
  
  - Though it was seen as a humorous nuisance rather than a bug, it raised discussions about hallucinations evident in AI code generation.

**Links mentioned**:

- [Qwen2.5-Coder: Code More, Learn More!](https://qwenlm.github.io/blog/qwen2.5-coder/): GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction In early April, we introduced CodeQwen1.5, which garnered significant attention from the community. Since then, we have been working to enhance...
- [VSCode Aider (Sengoku) - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=sengokudaikon.vscode-aider-sengoku): Extension for Visual Studio Code - Run Aider directly within VSCode for seamless integration and enhanced workflow.
- [Background Check Application in Go | Learn Temporal](https://learn.temporal.io/examples/go/background-checks/): The goal of this project is to teach you, the developer, how to think about building Temporal Applications that have Human-Driven Long-Running Workflows using a Temporal SDK, by leading you through a ...
- [Other LLMs](https://aider.chat/docs/llms/other.html): aider is AI pair programming in your terminal
- [YAML config file](https://aider.chat/docs/config/aider_conf.html): How to configure aider with a yaml config file.
- [AI Coding App CRUSHES $60M Tool (CURSOR KILLER??!)](https://www.youtube.com/live/ikn7JSUflTI?si=uJqzHU9Rh-fhBU7S): We put Repo Prompt head-to-head against a $60 MILLION AI coding tool, and the results will SHOCK you. Eric Provencher has discovered the secret sauce to prom...
- [Cline + Aider + Mistral FREE API : This is the BEST FREE WAY to do AI CODING! (Beats Gemini!)](https://www.youtube.com/watch?v=igE0X25bHcE): Join this channel to get access to perks:https://www.youtube.com/@AICodeKing/joinIn this video, I'll be telling you that how you can use the Mistral Free No-...

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1297285506345730102) (56 messages🔥🔥):

> - `Aider Usage with Sonnet and Claude`
> - `Managing Aider's Auto Commit Features`
> - `File Creation and Existence Issues in Aider`
> - `Leveraging Aider History for Context`
> - `Setting Main and Weak Models in Aider`

- **Feedback on Aider Usage**: Users expressed their appreciation for Aider and shared experiences related to its functionalities, such as using it for AI development in projects.
  
  - They discussed specific technical challenges they've faced, such as the model's handling of previously existing files.
- **Managing Auto Commits in Aider**: A user asked if it was possible to configure Aider to refrain from auto-committing changes, seeking a manual review process before commits.
  
  - Another user referenced the `--auto-commits` option in Aider's documentation that allows toggling this feature.
- **Issues with File Creation in Aider**: There were reports of Aider attempting to create files that already existed, leading to confusion about the model's behavior.
  
  - Some users suspected it might be related to Git's file tracking versus the actual file existence on the filesystem.
- **Utilizing Aider History**: A user inquired if Aider maintains context by loading history from previous sessions, prompting a discussion about available features to manage chat histories.
  
  - It was mentioned that Aider can restore past chat history and related files upon session initiation, enhancing user experience.
- **Configuring Main and Weak Models**: A user sought guidance on how to set their main and weak models explicitly within Aider.
  
  - Another user provided a YAML configuration example for creating a `.aider.conf.yml` file to define the models.

**Links mentioned**:

- [Linting and testing](https://aider.chat/docs/usage/lint-test.html#testing): Automatically fix linting and testing errors.
- [OpenGPT 4o - a Hugging Face Space by KingNish](https://huggingface.co/spaces/KingNish/OpenGPT-4o): no description found
- [Options reference](https://aider.chat/docs/config/options.html#--auto-commits): Details about all of aider’s settings.
- [Specifying coding conventions](https://aider.chat/docs/usage/conventions.html): Tell aider to follow your coding conventions when it works on your code.
- [Tutorial videos](https://aider.chat/docs/usage/tutorials.html): Intro and tutorial videos made by aider users.

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1296978933815971930) (1 messages):

> - `bitnet.cpp`
> - `1-bit LLMs`
> - `Inference performance on ARM and x86 CPUs`

- **Microsoft launches bitnet.cpp for 1-bit LLMs**: Microsoft released [bitnet.cpp](https://github.com/microsoft/BitNet) as the official inference framework for **1-bit LLMs**, including the **BitNet b1.58** model.
  
  - This framework supports optimized kernels for **fast and lossless inference** on CPUs, with plans for NPU and GPU support in the future.
- **Impressive speedups and efficiency gains on ARM CPUs**: On ARM CPUs, bitnet.cpp achieves speedups between **1.37x to 5.07x** with larger models showing the most significant performance gains.
  
  - It also reduces energy consumption by **55.4% to 70.0%**, enhancing overall efficiency for running LLMs.
- **x86 CPUs see remarkable performance enhancements**: For x86 CPUs, bitnet.cpp provides speedups ranging from **2.37x to 6.17x** along with energy reductions between **71.9% to 82.2%**.
  
  - This enables running a **100B BitNet b1.58 model** on a single CPU at speeds that mimic human reading rates (5-7 tokens per second).

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1297034862695419914) (30 messages🔥):

> - `TensorRT-LLM Code Sharing`
> - `Unsloth Lecture and Resources`
> - `GPU MODE Talk Recording`
> - `Event Scheduling Inquiry`
> - `Distributed Training Framework Comparisons`

- **TensorRT-LLM Code Sharing**: A user shared a link to the **TensorRT-LLM** repository, specifically pointing out the [cutlass int8 gemm kernel](https://github.com/NVIDIA/TensorRT-LLM/blob/a65dba7aaf7e2d8bb0120eea8f8f04deff145d6a/cpp/tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm_template.h#L62-L63). This resource offers users a Python API for defining Large Language Models (LLMs).
  
  - The shared kernel can enhance **efficient inference** for models requiring optimized performance.
- **Unsloth Lecture and Resources**: Members are reminded about an upcoming talk focused on lower level aspects of **systems engineering**, discussing **Triton kernels** and CUDA. Related resources, including a [GitHub link](https://github.com/unslothai/unsloth), were shared for attendees to refer to.
  
  - Also, slides from the lecture were made available: [View Slides](https://docs.google.com/presentation/d/1BvgbDwvOY6Uy6jMuNXrmrz_6Km_CBW0f2espqeQaWfc/edit?usp=sharing).
- **GPU MODE Talk Recording**: After the talk, participants were thanked and informed that it was **recorded** for later viewing. The recording will likely be available on the [YouTube channel](https://www.youtube.com/@GPUMODE/videos) in a few days.
  
  - This provides an opportunity for those who missed the live event to catch up on the discussions.
- **Event Scheduling Inquiry**: A member inquired where to sign into talks mentioned in the announcements. The response directed them to the **events tab** where Zoom links can be found.
  
  - This helps ensure that members can have access to relevant talks regardless of their time zone.
- **Distributed Training Framework Comparisons**: A user expressed the need for resources comparing different **distributed training frameworks**. They noted that inconsistencies in configurations across papers hinder accurate comparisons.
  
  - This highlights a gap where standardization could improve understanding of frameworks' effects on training outcomes.

**Links mentioned**:

- [GPU MODE](https://www.youtube.com/@GPUMODE/videos): A GPU reading group and community https://discord.gg/gpumode Supplementary content here https://github.com/gpu-mode Created by Mark Saroufim and Andreas Köpf
- [GPU MODE Lecture 32 - Unsloth](https://docs.google.com/presentation/d/1BvgbDwvOY6Uy6jMuNXrmrz_6Km_CBW0f2espqeQaWfc/edit?usp=sharing): 1 Lecture 32 GPU MODE LLM Systems Engineering Daniel from Unsloth
- [TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm_template.h at a65dba7aaf7e2d8bb0120eea8f8f04deff145d6a · NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/a65dba7aaf7e2d8bb0120eea8f8f04deff145d6a/cpp/tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm_template.h#L62-L63): TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficie...

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1297948189885726772) (5 messages):

> - `Triton and PTX compatibility`
> - `Windows support for Triton`
> - `Interaction between torch.compile and Triton`

- **Triton Python may struggle with PTX transition**: Uncertainty surrounds whether there is a direct path from **Triton Python** to **PTX**, although the **NVIDIA backend** appears to process **LLVM IR**.
  
  - This ambiguity raises questions about the effectiveness of the compilation pipeline.
- **Windows support for Triton remains questionable**: A small chance exists that a **Windows system** with **Visual Studio's LLVM** may function correctly, yet insufficient changes suggest a lack of understanding from Triton's Windows author.
  
  - Doubts linger whether the necessary adjustments for compatibility have been adequately addressed.
- **torch.compile and Triton have a dotted relationship**: There seems to be an interaction between **torch.compile** and **Triton** that should be seamless, yet Triton exhibits failures without raising errors.
  
  - This lack of error indication complicates debugging efforts and signals potential issues in their integration.

**Links mentioned**:

- [triton/third_party/nvidia/backend/compiler.py at a19f32454271ff9565ab957834bdf1e5d4ddce57 · triton-lang/triton](https://github.com/triton-lang/triton/blob/a19f32454271ff9565ab957834bdf1e5d4ddce57/third_party/nvidia/backend/compiler.py#L310): Development repository for the Triton language and compiler - triton-lang/triton
- [triton/python/src/llvm.cc at a19f32454271ff9565ab957834bdf1e5d4ddce57 · triton-lang/triton](https://github.com/triton-lang/triton/blob/a19f32454271ff9565ab957834bdf1e5d4ddce57/python/src/llvm.cc#L394): Development repository for the Triton language and compiler - triton-lang/triton

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1297147308076437595) (11 messages🔥):

> - `Torch Distributions in LibTorch`
> - `New PyTorch Environment Variable`
> - `Clearing Compiled Ops in PyTorch`
> - `Behavior of Autocast in PyTorch`
> - `DDP Training Issues with ResNet50`

- **LibTorch lacks MultivariateNormal equivalent**: A user asked if there is an equivalent of `torch.distributions.MultivariateNormal` available in LibTorch, the C++ API for PyTorch.
  
  - This reflects an ongoing need for similar functionality in different programming interfaces within the PyTorch ecosystem.
- **New PyTorch env var prevents power dips**: A member highlighted that `PYTORCH_NO_POWERPLANT_BLOWUP` is a new environment variable designed to mitigate large power dips during checkpointing.
  
  - *This change was discussed as a notable improvement in performance management for large computational tasks.*
- **Questions on clearing cached ops**: A user posed questions about how to clear compiled operations or caches in PyTorch, referencing both `torch.compiler.reset()` and `torch._dynamo.reset_code_caches` as potential solutions.
  
  - They also inquired about achieving forced recompilation in their model training setup using `torch.compile`.
- **Autocast reveals dtype discrepancies**: A user demonstrated that while `torch.autocast` can lead to a return type of `torch.float32`, they observed differing results based on device type and datatype used.
  
  - This prompted questions about the expected behavior of autocasting during mixed precision computations.
- **DDP ResNet50 training issues**: A user reported running into OOM errors with PyTorch 2.5 and various warnings about profiler functions being skipped while trying to train a ResNet50 model with sparse masks.
  
  - They faced an unexpected downgrade to PyTorch 2.2.1 despite intending to use version 2.4, indicating potential installation issues.

**Links mentioned**:

- [Tweet from Pytorch To Atoms (@PytorchToAtoms)](https://fxtwitter.com/PytorchToAtoms/status/1828148537013510474): mainline pytorch literally has a new env var called "PYTORCH_NO_POWERPLANT_BLOWUP" to prevent large power dips during checkpointing & during places in the trace where comms can't be ov...
- [Using torch.compile twice on a model on the same machine, is there a cache of optimized operations?](https://stackoverflow.com/questions/77931982/using-torch-compile-twice-on-a-model-on-the-same-machine-is-there-a-cache-of-op)): I'm using torch.compile to compile a torch model by: self.model = torch.load(saved_model_path, map_location=self.device).to(self.device) self.model.eval() self.model.half() &...
- [Compile Time Caching in torch.compile — PyTorch Tutorials 2.5.0+cu124 documentation](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html): no description found
- [TV pickup - Wikipedia](https://en.wikipedia.org/wiki/TV_pickup): no description found
- [Automatic Mixed Precision package - torch.amp — PyTorch 2.5 documentation](https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float32)): no description found
- [pytorch:2.0.0 ddp training error but the old version is good · Issue #1144 · pytorch/examples](https://github.com/pytorch/examples/issues/1144): Your issue may already be reported! Please search on the issue tracker before creating one. Context Pytorch version: Operating System and version: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime Your...

---

### **GPU MODE ▷ #**[**announcements**](https://discord.com/channels/1189498204333543425/1189640399476764692/1297269030574624808) (1 messages):

> - `Han Brothers`
> - `Unsloth Presentation`
> - `Triton Tricks`
> - `CUDA Techniques`

- **Han Brothers set to discuss Unsloth**: The **Han Brothers** will be presenting on **Unsloth** in **15 minutes** on Discord.
  
  - *Expect lots of crazy Triton and CUDA tricks* during their talk.
- **Anticipation for Triton and CUDA Insights**: Many members expressed excitement about the **Triton** and **CUDA** tricks that the Han Brothers are expected to share.
  
  - The presentation is anticipated to bring valuable insights and innovative techniques.

 

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1297628443520798741) (6 messages):

> - `Domino Communication Scheme`
> - `Torchtitan Library`
> - `Large Language Model Training`

- **Domino hides communication overhead in LLM training**: The paper on [Domino](https://arxiv.org/abs/2409.15241) presents a generic scheme for eliminating communication overhead in distributed training of Large Language Models (LLMs) by overlapping computation with communication, achieving up to **1.3x speedup** on Nvidia DGX-H100 GPUs.
  
  - By breaking data dependencies of batch training into smaller independent pieces, Domino improves efficiency compared to **Megatron-LM**.
- **Torchtitan connects to Domino**: A member noted that the **Torchtitan** library for large model training is actually the same as the Domino approach mentioned in the paper.
  
  - They referenced a [GitHub repository for Torchtitan](https://github.com/pytorch/torchtitan) that supports native PyTorch training.
- **Similarity with Torchtitan Paper**: Another member confirmed that there is a paper on Torchtitan in arXiv, which is very similar to the Domino concept, highlighting the close relationship between the two.
  
  - This suggests a strong connection in methodologies used for optimizing LLM training.

**Links mentioned**:

- [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping](https://arxiv.org/abs/2409.15241): Given the popularity of generative AI, Large Language Models (LLMs) often consume hundreds or thousands of GPUs for parallelizing and accelerating the training process. Communication overhead becomes ...
- [Lemur: Log Parsing with Entropy Sampling and Chain-of-Thought Merging](https://arxiv.org/abs/2402.18205): Logs produced by extensive software systems are integral to monitoring system behaviors. Advanced log analysis facilitates the detection, alerting, and diagnosis of system faults. Log parsing, which e...
- [GitHub - pytorch/torchtitan: A native PyTorch Library for large model training](https://github.com/pytorch/torchtitan): A native PyTorch Library for large model training. Contribute to pytorch/torchtitan development by creating an account on GitHub.

---

### **GPU MODE ▷ #**[**jobs**](https://discord.com/channels/1189498204333543425/1190208177829068860/1297955618254098544) (1 messages):

> - `Hiring GPU Programmers`
> - `Decentralized AI`
> - `Tokens per second improvement`

- **Opentensor Foundation seeks GPU talent**: The Opentensor Foundation, developers of **Bittensor** ([website](https://bittensor.com/)), announced they're hiring top talent in **GPU programming** to enhance decentralized AI capabilities.
  
  - Ryan, Head of Talent, encourages applicants to submit a PR to improve the **tokens per second** on a H100 box using the configurations from their [GitHub script](https://github.com/unconst/boltzmann/blob/pipe/deep.py).
- **Opportunity for Bold Collaborators**: Interested candidates are encouraged to showcase their skills in a hands-on way by working directly with the Opentensor team.
  
  - The call for talent emphasizes the potential for impactful contributions to the **decentralized AI** space.

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1297464669979676747) (40 messages🔥):

> - `Flash Attention Multiplication`
> - `LlamaCPP and GGML library usage`
> - `Raspberry Pi graphics performance`
> - `Triton and CUDA compatibility`
> - `Debugging Multihead Attention in models`

- **Understanding Flash Attention Multiplication**: A user questioned why Flash Attention multiplies O_old with l_i\*e^m, speculating it might be for normalization purposes.
  
  - This led to a discussion on the role of O_old and its significance in Flash Attention.
- **Getting Started with LlamaCPP**: One member suggested downloading and building the [LlamaCPP](https://link.to/llamacpp) / GGML library for a better understanding of optimized tensor usage.
  
  - They highlighted the importance of running LLMs and converting Huggingface models to ONNX format for optimizations.
- **Graphics on Raspberry Pi vs. Alternative Boards**: A discussion emerged regarding the proprietary integrated graphics of Raspberry Pi, with suggestions to reverse engineer it for performance.
  
  - Users recommended open-source driver boards like **Odroid N2+** and **RK3588** for better graphic capabilities.
- **Triton Compatibility and CUDA Versions**: A user faced CUDA out of memory errors while using Triton with Liger operations and asked about running it on older GPUs like K80.
  
  - It was suggested to downgrade the CUDA toolkit, as newer versions do not support older architectures, namely SM_37.
- **Debugging Multihead Attention Mask Issue**: One user reported a runtime error related to the shape of attn_mask in their decoder for a model, indicating mismatch with the expected size.
  
  - They expressed frustration after a week of troubleshooting and sought community assistance in resolving the mask issue.

**Links mentioned**:

- [Vector Addition — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#sphx-glr-getting-started-tutorials-01-vector-add-py): no description found
- [Nvidia Tesla K80, Cuda version support?](https://forums.developer.nvidia.com/t/nvidia-tesla-k80-cuda-version-support/67676): Hi, I working using Google Cloud Instance with GPU K80 and Ubuntu 16.04. But I have one question, what is the correct Cuda version, 9.0 , 9.2 or 10 for this hardware ? in this link you see more inf...

---

### **GPU MODE ▷ #**[**pmpp-book**](https://discord.com/channels/1189498204333543425/1194427148656721970/1297312465318842370) (4 messages):

> - `Chapter 4 exercises`
> - `Occupancy calculation`

- **Answers to Chapter 4 Exercises Found**: One user shared a link to the [Chapter 4 exercise answers](https://github.com/mandliya/PMPP_notes/blob/main/4_GPU_Architecture/exercises.md) from the repository containing notes on GPU architecture.
  
  - This resource is part of the broader [PMPP notes project](https://github.com/mandliya/PMPP_notes) that provides useful information for programming massively parallel processors.
- **Uncertainty About Occupancy Calculation**: Another user expressed uncertainty regarding their **occupancy calculation** after receiving the link to the exercise answers.
  
  - This highlights a common concern among learners when tackling complex GPU programming concepts.

 

**Link mentioned**: [PMPP_notes/4_GPU_Architecture/exercises.md at main · mandliya/PMPP_notes](https://github.com/mandliya/PMPP_notes/blob/main/4_GPU_Architecture/exercises.md): Notes and code for Programming Massively Parallel Processors - mandliya/PMPP_notes

 

---

### **GPU MODE ▷ #**[**youtube-recordings**](https://discord.com/channels/1189498204333543425/1198769713635917846/) (1 messages):

gau.nernst: [https://www.youtube.com/watch?v=hfb_AIhDYnA](https://www.youtube.com/watch?v=hfb_AIhDYnA)

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1296928468100976702) (2 messages):

> - `GitHub Issues`
> - `PyTorch Release Performance`

- **Request for GitHub Issue on Performance Regression**: A member requested @appy22 to create a [GitHub issue](https://github.com/pytorch/pytorch/issues/138386) regarding a performance regression in **torch 2.5** compared to **2.4.1** when using **torch.compile**.
  
  - They noted that the latest release appears to be slower while testing on multiple machines, including a **4090 RTX**.
- **Discussion Points from GitHub Issue**: The GitHub issue titled **'torch 2.5 slower than 2.4.1?'** was shared, detailing a bug report regarding performance discrepancies.
  
  - In the issue, the user mentioned experiencing notable slowdowns with **torch.compile** on the latest stable release.

 

**Link mentioned**: [torch 2.5 slower than 2.4.1 ? · Issue #138386 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/138386): 🐛 Describe the bug I noticed that the latest stable release 2.5.0 is slower than 2.4.1 when using torch.compile (reduce-overhead), I tried on different machines with a 4090 RTX and it's pretty mu...

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1297482652450033716) (6 messages):

> - `Daniel Hanchen's talk recording`
> - `Inventec CXL Box`
> - `Micron GPU architecture`
> - `Iceberg lettuce salad recipe`
> - `Lecture 32: Unsloth`

- **Daniel Hanchen's talk recording available**: [Recordings of Daniel Hanchen's talk](https://youtu.be/hfb_AIhDYnA?si=LU6712r-oOIARRKq) are posted on the channel as usual.
  
  - One member confirmed they watched it immediately upon release and expressed gratitude for the link.
- **Inventec CXL Box revolutionizes memory**: The [Inventec CXL Box](https://www.servethehome.com/inventec-96-dimm-cxl-expansion-box-at-ocp-summit-2024-for-tbs-of-memory-astera-labs-intel-xeon-6/) offers a 96x DDR5 DIMM memory shelf enabling **20TB** of PCI Gen5 attached RAM.
  
  - It connects to an upcoming **8-way Intel Xeon 6** server, providing a remarkable **224 DIMM slots** in total for scale-up applications.
- **Understanding GPU Architecture**: [How do Graphics Cards Work? Exploring GPU Architecture](https://www.youtube.com/watch?v=h9Z4oGN89MU) is a YouTube video discussing GPU design and memory technology at Micron.
  
  - The video also shares career opportunities at Micron for those interested in cutting-edge memory chip development.
- **Members share their culinary creations**: A member detailed a meal consisting of **iceberg lettuce salad**, **mashed potatoes**, and **beef patties** prepared with various ingredients.
  
  - They also shared a **hot beverage** and fresh fruits as part of their meal.

**Links mentioned**:

- [Inventec 96 DIMM CXL Expansion Box at OCP Summit 2024 for TBs of Memory](https://www.servethehome.com/inventec-96-dimm-cxl-expansion-box-at-ocp-summit-2024-for-tbs-of-memory-astera-labs-intel-xeon-6/): Perhaps the coolest bit of hardware at OCP Summit 2024, Inventec has an 8-way Intel Xeon 6 server with a 96 DIMM CXL Box for 224 DIMMs total
- [Lecture 32: Unsloth](https://youtu.be/hfb_AIhDYnA?si=LU6712r-oOIARRKq): no description found
- [How do Graphics Cards Work? Exploring GPU Architecture](https://www.youtube.com/watch?v=h9Z4oGN89MU): Interested in working with Micron to make cutting-edge memory chips? Work at Micron: https://bit.ly/micron-careers Learn more about Micron's Graphic Memory...

---

### **GPU MODE ▷ #**[**irl-meetup**](https://discord.com/channels/1189498204333543425/1218444432588800010/1297539159149514824) (6 messages):

> - `Sydney Meetup Coordination`
> - `NeurIPS Conference Participation`
> - `NeurIPS Location`

- **Sydney folks want to meetup**: A member reached out to see if anyone in Sydney or Australia is interested in coordinating a meetup, offering to host at a university.
  
  - This could be a great opportunity for local AI enthusiasts to connect and collaborate.
- **NeurIPS attendance confirmed**: Another member confirmed their attendance to the NeurIPS conference, generating excitement about the event.
  
  - This sparked a discussion among participants about who else might be attending.
- **NeurIPS hosted in Vancouver**: The location of NeurIPS was confirmed to be in **Vancouver, Canada**, as clarified during the discussion.
  
  - Several community members appeared enthusiastic about the proximity of the event.

 

---

### **GPU MODE ▷ #**[**triton-puzzles**](https://discord.com/channels/1189498204333543425/1219683012707487794/) (1 messages):

seahorse0180: Also ran into this issue just now.

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1297739422652960778) (2 messages):

> - `Parallel Prefix Sum Algorithm`
> - `Mamba Training`
> - `SSMs`
> - `Linear RNNs`
> - `llm.c Repository`

- **Inquiry on Parallel Prefix Sum Algorithm**: A member inquired if there is a **parallel prefix sum algorithm** for training **Mamba**, **SSMs**, or **Linear RNNs** available in the [llm.c repository](https://link.to.repo).
  
  - *Is there a parallel prefix sum algorithm for training Mamba / SSMs / Linear RNNs lying around in the LLM.c repo anywhere?*
- **llm.c Limited to GPT-2**: Another member clarified that unless there have been updates, **llm.c** is currently focused specifically on **GPT-2**.
  
  - They emphasized that *llm.c is currently specifically GPT-2*.

 

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1297270629736644711) (2 messages):

> - `Improvement sources for MI300X`
> - `RCCL tuning issues`
> - `Performance in single node RCCL`
> - `Async TP kernels challenges`

- **Ultra Ethernet vs IB for MI300X**: Waiting for [Ultra Ethernet](https://github.com/ROCm/rccl/blob/develop/src/graph/tuning.cc) or using **InfiniBand** like Microsoft with their **MI300X** is recommended as a significant source of improvement.
  
  - It was noted that **native RoCEv2** is unsuitable for AI/HPC applications facing bursty and elephant flow traffic.
- **RCCL tuning.cc Protocol Choices**: There's an observation that sometimes **rccl tuning.cc** chooses a non-optimal protocol and algorithm, which can hinder performance achievable with MI300X.
  
  - This issue arises particularly due to the lack of a reference network architecture for MI300X, which differs from **H100**.
- **Room for Improvement in Single Node RCCL**: Many low-hanging fruit opportunities in **single node RCCL** can enhance performance significantly.
  
  - For example, the support for **symmem** is still lacking in ROCm, making it challenging to write **async TP kernels** effectively on AMD systems.

 

**Link mentioned**: [rccl/src/graph/tuning.cc at develop · ROCm/rccl](https://github.com/ROCm/rccl/blob/develop/src/graph/tuning.cc): ROCm Communication Collectives Library (RCCL). Contribute to ROCm/rccl development by creating an account on GitHub.

 

---

### **GPU MODE ▷ #**[**sparsity-pruning**](https://discord.com/channels/1189498204333543425/1247663759434977453/1297839205703090201) (3 messages):

> - `Activation Sparsity Tools`
> - `PyTorch Sparse Functionality`
> - `PowerInfer Research`
> - `Sparse Kernel Implementations`

- **Questions on Activation Sparsity Tools**: A member is looking for effective tools for activation sparsity matrix multiplication and noted using PyTorch's `to_sparse_semi_structured`, which is limited to 2D tensors.
  
  - *They noted manual iteration is required for larger dimensions due to this limitation.*
- **Creating GitHub Issue for Sparsity Tasks**: A member suggested creating an issue on GitHub for the sparsity task, specifically regarding the use of `to_sparse_semi_structured` and its performance.
  
  - *They emphasized the importance of including a minimal code repro for effective troubleshooting.*
- **Using Training Kernels for Efficiency**: Another member pointed out that `to_sparse_semi_structured` employs slower conversion methods, suitable for weight sparsity but not efficient for activation sparsity needed at runtime.
  
  - *They recommended utilizing faster sparsification kernels to enhance overall performance.*

 

**Link mentioned**: [Issues · pytorch/ao](https://github.com/pytorch/ao/issues): PyTorch native quantization and sparsity for training and inference - Issues · pytorch/ao

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1297274766146928712) (7 messages):

> - `Acknowledgement of Contributors in Liger Arxiv Whitepaper`
> - `Gradient Accumulation Bug in Liger Kernel`
> - `Memory Issues with Triton and Liger Operations`
> - `Calls for Code Review on Gradient Accumulation`
> - `Updates on Liger Kernel Documentation`

- **Contributors Acknowledged in Upcoming Whitepaper**: Discussion arose about including a generic acknowledgement of **open-source contributors** in the [Liger Arxiv whitepaper](https://arxiv.org/pdf/2410.10989).
  
  - An updated version is in the works to include **heavy contributors' names** and to promote a committee system.
- **Gradient Accumulation Bug Inquiry**: A member inquired if a recent **gradient accumulation bug** fix in the transformers library applies to Liger Kernel's cross entropy operation as well.
  
  - This highlights the ongoing need for clarity on potential issues within Liger Kernel's functionality.
- **Cuda Memory Errors Linked to Liger Operations**: Concerns were raised regarding **cuda out of memory errors** encountered when using Liger operations with a PyTorch model utilizing torch compile.
  
  - This raises questions about specific memory allocation patterns associated with **Triton** or **Liger**.
- **Code Review for Gradient Accumulation Techniques**: Members shared code snippets related to **gradient accumulation** involving different ops like fused linear cross entropy and layer norm.
  
  - These submissions suggest a community effort to ensure **efficient gradient accumulation** implementations.

**Links mentioned**:

- [Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py at 6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91/src/liger_kernel/ops/fused_linear_cross_entropy.py#L110): Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.
- [Liger-Kernel/src/liger_kernel/ops/fused_linear_jsd.py at 6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91/src/liger_kernel/ops/fused_linear_jsd.py#L98): Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.
- [Liger-Kernel/src/liger_kernel/ops/layer_norm.py at 6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91/src/liger_kernel/ops/layer_norm.py#L216): Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.
- [Liger-Kernel/src/liger_kernel/ops/rms_norm.py at 6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/blob/6ab3b9febc29f5045e6d2e27ba6bacaa4f041d91/src/liger_kernel/ops/rms_norm.py#L289): Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.

---

### **GPU MODE ▷ #**[**metal**](https://discord.com/channels/1189498204333543425/1285384841730457600/1297058056215199806) (9 messages🔥):

> - `Objective-C language server`
> - `C/C++ memory management with PyTorch`
> - `MPS stream in PyTorch`
> - `Unified memory on Apple Silicon`
> - `MTLCommandQueue functionality`

- **Objective-C Language Server Gains Fans**: A member found [Objective-C language server](https://github.com/MaskRay/ccls) useful, noting it can be installed with `brew install ccls` and works well with VSCode.
  
  - Another user confirmed they have been using it for C and C++ with decent results.
- **Exploring Memory Management with PyTorch**: Questions arose about using unified memory on Apple Silicon for PyTorch and whether tensors are allocated in private mode by default.
  
  - One member raised concerns about potential memory management issues when using custom buffers with `at::from_blob()`.
- **Clarifying MPS Stream in PyTorch**: A member stated that MPSStream is a tuple of `id<MTLCommandQueue>` and `dispatch_queue_t`, indicating its function in managing command queues.
  
  - Further exploration confirmed that MPS stream conveys the concept of work execution, noting that multiple queues can be executed concurrently on the GPU.

**Links mentioned**:

- [GitHub - MaskRay/ccls: C/C++/ObjC language server supporting cross references, hierarchies, completion and semantic highlighting](https://github.com/MaskRay/ccls): C/C++/ObjC language server supporting cross references, hierarchies, completion and semantic highlighting - MaskRay/ccls
- [pytorch/aten/src/ATen/mps/MPSStream.mm at d1027c2be6ad2ee8c9c50fa83293babd05cb6a2c · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/d1027c2be6ad2ee8c9c50fa83293babd05cb6a2c/aten/src/ATen/mps/MPSStream.mm#L17-L33): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [pytorch/aten/src/ATen/mps/MPSAllocator.h at 3f3b692a00737c54a3e2948db5db493d40119854 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/3f3b692a00737c54a3e2948db5db493d40119854/aten/src/ATen/mps/MPSAllocator.h#L124-L125.): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-questions**](https://discord.com/channels/1179127597926469703/1179208129083363358/1297392869404966923) (12 messages🔥):

> - `Human Data Labeling for Geospatial Data`
> - `Platforms for Data Labeling`
> - `Offshore Vendors for Visual Data`
> - `Scale AI and Alternatives`

- **Quest for Human Data Labelers**: A member sought recommendations for human data labelers specifically for **weather radar data**, expressing interest in geospatial and vision language labeling.
  
  - *What are the best platforms?*
- **Consideration of Different Platforms**: Members discussed various platforms for data labeling, highlighting **Scale AI**, **Surge**, **Mechanical Turk**, and **Prolific**.
  
  - One member noted the **pros and cons** of these platforms for different data types.
- **Natolambert's References on Data Labeling**: Natolambert referenced two posts discussing **Scale AI** and its role in the market for human data and RLHF techniques, hinting at the **growing demand** in this area.
  
  - He shared links to further details, including [Scale AI’s business model](https://www.interconnects.ai/p/ai-data-foundry).
- **Offshore Vendors Recommended for Radar Data**: One member advised against using major **GenAI** vendors for simple radar data tasks, suggesting **offshore vendors** instead as a better option.
  
  - They mentioned **Mechanical Turk** could work with **handholding** involved, and asked about the **volume** of data needed.

**Links mentioned**:

- [Futures of the data foundry business model](https://www.interconnects.ai/p/ai-data-foundry): Scale AI’s future versus further scaling of language model performance. How Nvidia may take all the margins from the data market, too.
- [Alignment-as-a-Service: Scale AI vs. the new guys](https://www.interconnects.ai/p/alignment-as-a-service): Scale’s making over $750 million per year selling data for RLHF, who’s coming to take it?

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1296963148191895664) (74 messages🔥🔥):

> - `Emoji Requests`
> - `RLHF Book Development`
> - `OpenAI Token Usage`
> - `CARDS Alignment Method`
> - `Dark Mode Discussion`

- **Nato Takes Emoji Requests**: Nato confirmed he is taking emoji requests and shared a channel link for submissions, prompting a humorous reaction from members.
  
  - The message included various emoji reactions, highlighting a playful atmosphere in the chat.
- **Progress on RLHF Book**: Nato announced he is working on a reinforcement learning from human feedback (RLHF) book, aiming for a physical copy by the end of the year.
  
  - He shared the [book's website](https://rlhfbook.com/) and noted the importance of his approach to writing without extensive checks, embracing community engagement.
- **OpenAI Token Behavior Decoding**: Nato responded to a tweet discussing the token usage of OpenAI's models, focusing on how reasoning tokens appear to be multiples of 64.
  
  - He speculated that the reported reasoning tokens might be an approximation and discussed the limited readership of his blog on the subject.
- **Introduction to CARDS Decoding Alignment**: A member introduced a new method called CARDS, which reportedly accelerates text generation and ensures high-reward outcomes without retraining models.
  
  - The method uses segment-level rejection sampling, and a link to the [related paper](https://arxiv.org/abs/2406.16306) was provided for interested readers.
- **Dark Mode Discussions**: Members engaged in lighthearted exchanges about the visibility of logos in dark mode on different platforms, illustrating their experiences.
  
  - Nato humorously reassured participants that maintaining a consistent work-life balance amidst various projects, including his RLHF book, is important.

**Links mentioned**:

- [Tweet from Ruqi Zhang (@ruqi_zhang)](https://x.com/ruqi_zhang/status/1810690177498595761): Introducing CARDS, a new method for LLM decoding-time alignment: ✨5x faster in text generation and 99% win-ties in GPT-4/Claude-3 evaluation ✨provably generates high-reward high-likelihood text ✨no r...
- [Tweet from Yuntian Deng (@yuntiandeng)](https://x.com/yuntiandeng/status/1848421766093255027): How many reasoning tokens does OpenAI o1 use? It turns out they are almost always multiples of 64 (99+% of the time in 100K collected turns)🤔Could it be that the model only uses multiples of 64 token...
- [NaNoWriMo](https://nanowrimo.org/): no description found
- [The Little Book of Deep Learning](https://fleuret.org/francois/lbdl.html): no description found
- [The Basics of Reinforcement Learning from Human Feedback](https://rlhfbook.com/): The Basics of Reinforcement Learning from Human Feedback
- [GitHub - natolambert/rlhf-book: Textbook on reinforcement learning from human feedback](https://github.com/natolambert/rlhf-book): Textbook on reinforcement learning from human feedback - natolambert/rlhf-book

---

### **Interconnects (Nathan Lambert) ▷ #**[**rlhf**](https://discord.com/channels/1179127597926469703/1208183230608576562/1296932075412131962) (16 messages🔥):

> - `Interconnects Emojis`
> - `Discord Support`
> - `Emoji Uploading`
> - `AI Company Logos`

- **Quest for Interconnects Emojis**: Members discussed how to add **Interconnects emojis** to the server, with suggestions for various **AI company logos** and meme ideas like more **snail bot** content.
  
  - One member humorously suggested raising prices for users who haven't joined Discord, highlighting community engagement.
- **Potential Support from Discord Staff**: A member joked about calling in help from a Discord staff member if they couldn't figure out the emoji settings, indicating confidence in resolving the issue.
  
  - Another member confirmed it's a simple task, stating, 'it shouldn't be too hard' based on their experience with emoji and soundboard uploads.
- **Aesthetic Improvements for Emojis**: There were requests for a **dark mode-compatible OpenAI logo** and a dark version of the **Interconnects** emoji for aesthetic purposes.
  
  - Additionally, suggestions were made for improving the **alpha channel** on some logos, with a focus on enhancing visibility.

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1297964262446071920) (2 messages):

> - `LLM Reasoning Debate`
> - `OpenAI's GPT Releases`
> - `Training Data Limitations`

- **LLM Reasoning Debate Heats Up**: A recent post highlights a raging debate on whether large language models (LLMs) can effectively reason, sparked particularly by OpenAI's latest releases, **GPT-4o** and **GPT-o1**.
  
  - Questions remain about whether these models employ *actual reasoning* or simply mimic patterns they've seen in training data, potentially limiting their problem-solving capabilities.
- **OpenAI Releases GPT-4o and GPT-o1**: In May 2024, OpenAI launched **GPT-4o**, claiming it can reason across audio, vision, and text in real time, followed by the **GPT-o1** model known for its accurate performance on reasoning-heavy benchmarks.
  
  - These advancements further fueled discussions on the real reasoning abilities of LLMs versus learned behavior.
- **Concerns Over Problem-Solving Capability**: The debate questions whether LLMs like GPT-4o and o1 genuinely solve problems or rely on patterns from training data, which may hinder performance on unfamiliar tasks.
  
  - The implication is that understanding this distinction is crucial for assessing the future development of AI reasoning.

 

**Link mentioned**: [The LLM Reasoning Debate Heats Up](https://open.substack.com/pub/aiguide/p/the-llm-reasoning-debate-heats-up?r=68gy5&utm_medium=ios) : Three recent papers examine the robustness of reasoning and problem-solving in large language models

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1296920128641962045) (98 messages🔥🔥):

> - `Performance of RTX GPUs`
> - `Creating Images with Different Perspectives`
> - `Using Loras in Prompts`
> - `Stable Diffusion API Access Issues`
> - `Need for Assistance in Image Editing`

- **RTX 3090 underperforms expectations**: Despite expectations of improved performance, a user reported their RTX 3090 only achieving **3.5 iterations per second** compared to their previous RTX 3060's rate, which was surprising.
  
  - Suggestions included ensuring the web UI was updated and reinstalling drivers, which may help in optimizing performance.
- **Challenges in Changing Image Perspective**: A user inquired about creating different perspectives for an existing photograph of a building and retaining colors and objects in new sketches, but faced difficulties due to photo limitations.
  
  - Members discussed potential solutions, including the need for more drone shots and training a Lora to learn from specific buildings.
- **Issues with Loras in Image Generation**: A user encountered a problem where multiple Loras resulted in an error message stating certain Loras were not found when generating images.
  
  - Others chimed in offering potential ways to troubleshoot or manage prompts better to resolve the conflict.
- **Accessing the Stability.ai API Page**: A user raised concerns about accessing the Stability.ai API reference page, mentioning it appeared to be down.
  
  - Responses indicated that users would need to contact customer service for support since the community does not manage the website or API.
- **Need for Assistance in Image Editing**: Users expressed the need for help related to editing images and incorporating AI tools into their workflows, particularly for commercial projects.
  
  - One user offered assistance through direct messages, indicating a collaborative environment within the community.

**Links mentioned**:

- [Stability AI - Developer Platform](https://platform.stability.ai/docs/api-reference): no description found
- [update readme · alimama-creative/FLUX.1-Turbo-Alpha at b2db8dc](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/commit/b2db8dcbd15fb095cffd8ab530499e47883466e7): no description found
- [GitHub - chengzeyi/stable-fast: Best inference performance optimization framework for HuggingFace Diffusers on NVIDIA GPUs.](https://github.com/chengzeyi/stable-fast): Best inference performance optimization framework for HuggingFace Diffusers on NVIDIA GPUs. - chengzeyi/stable-fast

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1296919862916153465) (7 messages):

> - `3-day hackathon`
> - `LlamaParse Premium`
> - `Agentic System for Automated Sales Outreach`
> - `Advanced RAG Workflow`
> - `Multimodal RAG Pipeline`

- **3-Day Hackathon Delivers 45 Projects**: The recent **3-day hackathon** saw over **500** participants and resulted in **45 amazing projects** by the end of the weekend. Check out the [blog post announcing the winners](https://t.co/v7F8b0qedF) for details.
  
  - Winners will also be providing guest blog posts detailing their projects, generating excitement in the community.
- **LlamaParse Premium Receives Praise**: After introducing **LlamaParse Premium**, users have been expressing their enthusiasm for its improved parsing capabilities. An in-depth [LinkedIn post](https://t.co/NeAvIlfIP3) showcases how it outperforms its predecessors.
  
  - The original post introducing **LlamaParse** can also be found [here](https://t.co/pDPHxcYQeb).
- **Automated Sales Outreach Gets Smarter**: The blog by **Calsoft_Data** explores a **constrained agentic architecture** that automates sales outreach tasks, reducing time spent on manual processes. This approach is an effective solution for research prospects and personalized email creation.
  
  - You can read more about this innovative system [here](https://t.co/ziCb6UkcRd).
- **Lightning Fast RAG Workflow Tutorial**: A tutorial by **Plaban Nayak** describes setting up a **fully async RAG workflow** using **GroqInc**, optimizing reranking and synthesis. This provides a significant speed boost for handling data processes.
  
  - The tutorial can be accessed [here](https://t.co/r6ag69r5uu).
- **Efficient Multimodal RAG Pipeline Setup**: A tutorial by **fahdmirza** demonstrates how to establish an **advanced multimodal RAG pipeline** that indexes complex documents like slide decks efficiently. The process is simplified to the point where it 'just works', freeing time for development.
  
  - Find out more about this intuitive setup [here](https://t.co/d8IxLU8NKk).

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1296922143153721356) (81 messages🔥🔥):

> - `Ollama Integration in LlamaIndex`
> - `Evaluating Retrieval Methods`
> - `Event Streaming in Workflows`
> - `Document Summarization Techniques`
> - `Deployment Platforms for ML Models`

- **Integrating Ollama in LlamaIndex**: A user shared their configuration for using **Ollama** with `npx create-llama`, but experienced issues with an OpenAPI key pop-up despite correct settings.
  
  - Another member suggested editing the backend source code to successfully load **Ollama** LLM and embeddings.
- **Evaluating Hybrid Retrieval Accuracy**: Discussion emerged about methods to evaluate a hybrid retriever combining `BM25Retriever` and `VectorIndexRetriever`, highlighting the importance of ground truth datasets.
  
  - Several members recommended using an LLM to assess retrieval relevance or identifying a question-document mapping for meaningful evaluation.
- **Streaming Responses and Tool Calls**: A user noted inconsistencies in detecting tool calls across various models, with OpenAI detecting them immediately while others lagged.
  
  - A solution was suggested involving event streaming in workflows, enabling more efficient chunk handling during response generation.
- **Document Summarization in Indexing**: Members discussed whether to incorporate document summaries into retrieval systems, with consensus leaning towards using `DocumentSummaryIndex` for efficiency.
  
  - The importance of maintaining high-quality summaries was emphasized, as poor summaries could lead to hallucinated responses.
- **API Hosting Recommendations**: For deploying APIs that utilize models for specific datasets, suggestions included hosted solutions like **AWS**, **Azure**, and **GCP**.
  
  - Concerns about security on platforms were raised, particularly regarding **Hugging Face**, prompting discussions about the effectiveness of various deployment options.

**Links mentioned**:

- [Google Colab](https://colab.research.google.com/drive/1wVCkvX7oQu1ZwrMSAyaJ8QyzHyfR0D_j?usp=sharing): no description found
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102): no description found
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)): no description found
- [Starter Tutorial (OpenAI) - LlamaIndex](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/): no description found
- [SimpleDirectoryReader - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/): no description found
- [Qdrant Vector Store - Metadata Filter - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/vector_stores/Qdrant_metadata_filter/#qdrant-vector-store-metadata-filter): no description found
- [rsrohan99 - Overview](https://github.com/rsrohan99): rsrohan99 has 13 repositories available. Follow their code on GitHub.
- [no title found](https://docs.llamaindex]): no description found
- [llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-duckdb/llama_index/vector_stores/duckdb/base.py at 227145fb94fcaa4da02d559fc81843fcb2af2b57 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/227145fb94fcaa4da02d559fc81843fcb2af2b57/llama-index-integrations/vector_stores/llama-index-vector-stores-duckdb/llama_index/vector_stores/duckdb/base.py#L314): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [Survey for NEO clouds](https://docs.google.com/forms/d/e/1FAIpQLSfhn89gy8WSViYJoWT9N3MbD63dNl_8eyoJcBqzUXYni6PXog/viewform?usp=sf_link): We are building Neo Clouds, a cloud-based platform that offers powerful computing resources to run applications that need high computational power. To ensure we address your needs, we would love to h...
- [GitHub - microsoft/BitNet: Official inference framework for 1-bit LLMs](https://github.com/microsoft/BitNet): Official inference framework for 1-bit LLMs. Contribute to microsoft/BitNet development by creating an account on GitHub.
- [llama_index/llama-index-core/llama_index/core/vector_stores/types.py at 227145fb94fcaa4da02d559fc81843fcb2af2b57 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/227145fb94fcaa4da02d559fc81843fcb2af2b57/llama-index-core/llama_index/core/vector_stores/types.py#L63): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
- [update to use workflows by logan-markewich · Pull Request #4 · rsrohan99/rag-stream-intermediate-events-tutorial](https://github.com/rsrohan99/rag-stream-intermediate-events-tutorial/pull/4): Was helping out a user, and ended up converting this example to use workflows! Feel free to merge this or ignore this 😁
- [GitHub - logan-markewich/rag-stream-intermediate-events-tutorial: Tutorial on how to properly send intermediate LlamaIndex events to vercel ai sdk via server-sent events during RAG.](https://github.com/logan-markewich/rag-stream-intermediate-events-tutorial/tree/master): Tutorial on how to properly send intermediate LlamaIndex events to vercel ai sdk via server-sent events during RAG. - logan-markewich/rag-stream-intermediate-events-tutorial
- [GitHub - rsrohan99/llamaindex-workflow-streaming-tutorial](https://github.com/rsrohan99/llamaindex-workflow-streaming-tutorial): Contribute to rsrohan99/llamaindex-workflow-streaming-tutorial development by creating an account on GitHub.
- [Workflows - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/workflow/): no description found

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1297262312704577659) (3 messages):

> - `Multilingual Embedding Models`
> - `Creating API for Proprietary Materials`

- **Searching for Multilingual Embedding Solutions**: A member is working on a **RAG system** utilizing PDFs in multiple languages (EN, JP, ID, VI, TH) but has not found effective results with various **open-source** and **closed-source** embedding models.
  
  - Another member recommended the **aBSE** (Language-agnostic BERT Sentence Embedding) model as a potential solution for better multilingual results.
- **Guidance on Creating an API for Proprietary Content**: A beginner is seeking guidance on creating an **API** that can answer questions based on proprietary materials such as personal notes or books.
  
  - They requested insights on suitable **machine learning techniques**, as well as recommendations on **hosting platforms** and **dataset storage**.

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1297133182079991938) (55 messages🔥🔥):

> - `Multihead Attention Standardization`
> - `Tinygrad Development Updates`
> - `WebGPU Support`
> - `Local LLM Usage Trends`
> - `Benchmark CI Testing`

- **Multihead Attention Standardization Validity**: A member questioned whether the discussions around **standardizing Multihead Attention** were still relevant and valid.
  
  - The inquiry suggests ongoing interest in optimizing attention mechanisms in the Tinygrad community.
- **Tinygrad Eyes Competitiveness**: George Hotz announced the merging of **GGUF loading support**, hoping for Tinygrad to compete with other frameworks for **running LLMs** more effectively.
  
  - He encouraged developers of libraries and applications using Tinygrad to step forward, highlighting aspirations to surpass competitors like **Ollama** and **GGML**.
- **Local LLM Usage Insights**: Members mentioned using **Llama.cpp** and **ExLlamaV2** for running models locally, with ExLlamaV2 offering a simpler setup and comparable performance to Nvidia's **TensorRT-LLM**.
  
  - Discussion indicated a preference for these tools among users for efficient model deployment on personal setups.
- **Importance of WebGPU Support**: George Hotz emphasized the significance of **WebGPU support** in the development process and mentioned community efforts focusing on it.
  
  - Another member reported progress on working with **threefry** algorithms, anticipating fewer blockers moving forward.
- **Benchmark CI for LLM Robustness**: George underscored the necessity for thorough testing in **Benchmark CI** due to diverse potential GPU failure points in local model execution.
  
  - He highlighted that various edge scenarios need coverage to ensure robustness when running multiple models concurrently.

 

**Link mentioned**: [Big graph · Issue #7044 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/issues/7044): LazyBuffer.view becomes UOps.VIEW #7077 #7078 #7007 #7090 big graph SINK #7122, #7178, #7170 #7134, #7175 #7132, #7188 #7190 #7149 ASSIGN and toposort become graph_rewrite deciding when to realize ...

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1296961952597147648) (35 messages🔥):

> - `FrozenBatchNorm2d`
> - `Auto-generating classes from .safetensors`
> - `Action Chunking Transformers in Tinygrad`
> - `Implementing nonzero in Tinygrad`
> - `Context Manager for default float in Tinygrad`

- **Understanding FrozenBatchNorm2d**: A user inquired about the purpose of **FrozenBatchNorm2d** in certain network architectures, questioning its necessity and functionality.
  
  - They shared sample code and expressed confusion over how a `__call__()` function would work within this context.
- **Query on Class Auto-generation from .safetensors**: Users discussed the possibility of auto-generating a class from a **.safetensors** file, but found it challenging due to the lack of computation description.
  
  - One noted their excitement in getting a model working, seeking a way to facilitate easier conversion for future users.
- **Success with Action Chunking Transformers**: A user confirmed that their implementation of **Action Chunking Transformers** in Tinygrad is functional and currently under testing with different datasets.
  
  - They shared a GitHub link to their messy yet operational notebook, aiming to optimize the codebase soon.
- **Implementing nonzero Functionality in Tinygrad**: There was a discussion on how to replicate the **torch.nonzero** functionality in Tinygrad, particularly for adjacency matrices and indexing.
  
  - Alternatives were suggested, including using boolean indexing with `where` or converting indices to integers, but challenges remain with compatibility.
- **Changing Default Float with Context Manager**: A user asked about changing the default float within a function using a Context Manager or decorator in Tinygrad, referencing existing documentation.
  
  - They encountered a KeyError when attempting to set `DEFAULT_FLOAT`, leading to exploration of the variable's determination from the environment.

**Links mentioned**:

- [George Hotz | Programming | MNIST classifier from numpy scratch! | Science & Technology](https://www.youtube.com/watch?v=JRlyw6LO5qo): Date of stream 17 Oct 2020.Live-stream chat added as Subtitles/CC - English (Twitch Chat).Stream title: MNIST classifier from numpy scratch!Source files:- ht...
- [[NOMERGE] Llama: download tiny llama weights by default by jla524 · Pull Request #7173 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7173): no description found
- [act-tinygrad/modeling_act.ipynb at main · mdaiter/act-tinygrad](https://github.com/mdaiter/act-tinygrad/blob/main/modeling_act.ipynb): Action Chunking Transformers in Tinygrad. Contribute to mdaiter/act-tinygrad development by creating an account on GitHub.
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/): no description found
- [Environment Variables - tinygrad docs](https://docs.tinygrad.org/env_vars/): no description found
- [tinygrad/tinygrad/dtype.py at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/dtype.py#L121)): You like pytorch? You like micrograd? You love tinygrad! ❤️ - tinygrad/tinygrad

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1297081234736021534) (35 messages🔥):

> - `Mystery model`
> - `Agent assist APIs`
> - `Connection issues with Google Drive`
> - `Community introductions`
> - `General chat discussions`

- **Mystery Model Causes Buzz**: A member mentioned a **mystery model** with an **8k** context available, generating intrigue within the community.
  
  - *Something's cooking...* and members are eager to engage with the [mystery bot](https://discord.com/channels/954421988141711382/996880279224451154/1297180553401077771).
- **Inquiry About Agent Assist APIs**: A member inquired if **Cohere** provides **agent assist APIs** for generating responses based on supplied information.
  
  - Another member directed the inquiry to a specific channel for further discussion on the topic.
- **Help Needed for Google Drive Connection**: A user reported issues connecting to **Google Drive**, receiving an 'app is blocked' message and sought advice for workaround solutions.
  
  - A community member suggested providing additional context and screenshots to help troubleshoot the issue effectively.
- **Introductions from New Members**: Several new members introduced themselves, expressing interest in engaging with the **Cohere** community.
  
  - Topics of discussion included potential collaborative projects and community engagement.
- **Reminder on Channel Usage**: A member reminded others that the discussions channel is meant for general chat, while specific queries should be directed to other channels.
  
  - This aims to keep the channel organized and focused on broad discussions rather than specific issues.

 

**Link mentioned**: [Tweet from UltraIA (@Ultra_IA)](https://x.com/Ultra_IA/status/1847821253476008227): LOL

 

---

### **Cohere ▷ #**[**announcements**](https://discord.com/channels/954421988141711382/996880279224451154/1297180553401077771) (2 messages):

> - `Aya Community Project`
> - `Cohere Developer Office Hours`

- **Secret Project Launch by Aya Community**: The **Aya Community** invites users to help test a new **language connection project** by text messaging various international numbers including **Whatsapp** and **local toll-free** numbers.
  
  - Participants are encouraged to provide feedback on **issues encountered** and join the **Aya Discord** for further discussions, with a note to keep the numbers confidential.
- **Cohere Developer Office Hours Tomorrow**: Cohere will host **Developer Office Hours** tomorrow at **1:00 PM ET**, featuring live demos and insights from team members on new and upcoming releases.
  
  - Participants can join the discussion via the provided link: [Cohere Developer Event](https://discord.com/events/954421988141711382/1285304800400904344/1297967638118400000).

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1297019714811334676) (35 messages🔥):

> - `OpenRouter Benefits`
> - `Cohere API Usage`
> - `Langchain SSL Issues`

- **OpenRouter offers flexible API switching**: Members discussed the advantages of using **OpenRouter**, noting its ability to switch API providers seamlessly when one goes down.
  
  - *TBH, not all API providers are stable*, which enhances the appeal of OpenRouter.
- **Cohere API and its limitations**: A member inquired about the **Cohere API**, expressing interest in whether it includes specific models like **Reranker** and **embed-v3**.
  
  - Concerns were raised about direct use of the **Cohere API** requiring significant additional implementation due to **closed-source** nature.
- **Langchain SSL Errors are common**: One user faced SSL errors with **Langchain** while attempting to bypass security settings in a company network.
  
  - Another member suggested that *exporting CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1* could be a potential workaround for the issue.

 

**Link mentioned**: [Chat — Cohere](https://docs.cohere.com/reference/chat): Generates a message from the model in response to a provided conversation. To learn more about the features of the Chat API follow our [Text Generation guides]([https://docs.cohere.com/v2/docs/chat-api](https://docs.cohere.com/v2/docs/chat-api)...

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1297953984375554188) (6 messages):

> - `API read timeout issues`
> - `Getting citations from the API`
> - `Chat API documentation`

- **API read timeout issues flagged**: A member raised concerns about experiencing **read timeout errors** with the API over the weekend.
  
  - *sssandra* confirmed that they flagged this issue with the team for further investigation.
- **Citations available out of the box**: Citations are built-in features of the API, as clarified by *sssandra*, who directed users to the Chat API docs for more information.
  
  - They emphasized checking the [Retrieval Augmented Generation documentation](https://docs.cohere.com/v2/docs/retrieval-augmented-generation-rag) for details on effectively using citations.
- **Helpful API links shared**: Links to the [Chat API documentation](https://docs.cohere.com/reference/chat) and the [Migration Guide](https://docs.cohere.com/v2/docs/migrating-v1-to-v2) were provided to support users in navigating the API.
  
  - These resources outline essential usage instructions, including how to handle API requests and citations.

**Links mentioned**:

- [Chat — Cohere](https://docs.cohere.com/reference/chat): Generates a message from the model in response to a provided conversation. To learn more about the features of the Chat API follow our [Text Generation guides](https://docs.cohere.com/v2/docs/chat-api...
- [Retrieval Augmented Generation (RAG) — Cohere](https://docs.cohere.com/v2/docs/retrieval-augmented-generation-rag): Generate text with external data and inline citations using Retrieval Augmented Generation and Cohere's Chat API.
- [Documents and Citations — Cohere](https://docs.cohere.com/docs/documents-and-citations): The document introduces RAG as a method to improve language model responses by providing source material for context.

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1297862482924142613) (2 messages):

> - `JavaScript Implementations`
> - `Direct API Requests`

- **Impressive JavaScript Implementation**: *Very impressive! All in .js too!* a member remarked, showcasing excitement about a project leveraging JavaScript for its functionality.
  
  - This highlights the growing trend of utilizing **JavaScript** for effective AI applications.
- **Direct API Communication**: Another member confirmed that with just an **API key**, requests are made directly to the AI provider without needing a proxy.
  
  - This method simplifies interactions and reduces dependencies for developers.

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1296912694687039568) (44 messages🔥):

> - `Liger Kernel Installation`
> - `Axolotl Layer Freezing Issue`
> - `SNR Results for Spectrum`
> - `AGI House Events`

- **Liger Kernel Installation Guide**: To achieve **VRAM savings**, installing the Liger Kernel is straightforward: just use `pip install liger-kernel` and adjust the config as shared in the channel.
  
  - Users noted that Liger facilitates full finetuning, benefiting from existing Flash Attention capabilities.
- **Layer Freezing Bug in Axolotl**: There seems to be a **bug** preventing users from freezing/unfreezing layers in the latest version of Axolotl, which had been working previously.
  
  - This issue is under investigation, with community members asking others to confirm the recent change and checking the `src/axolotl/integrations/spectrum/model_snr_results` directory.
- **Spectrum SNR Results Discussion**: Discussion surrounding the top fractions and **SNR results** for models took place, confirming properly computed results for the Qwen models.
  
  - Members emphasized that the Spectrum integration requires precomputed **SNR JSON files** to function correctly.
- **AGI House Upcoming Events**: AGI House announced two exciting events: the **Think Slow & Think Deep** Hackathon on November 2nd and the **AI Agent Werewolf Tournament** on November 9th.
  
  - The werewolf tournament offers significant cash prizes and aims to bring together innovative designers competing with AI agents.

**Links mentioned**:

- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102): no description found
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)): no description found
- [spectrum/model_snr_results at main · cognitivecomputations/spectrum](https://github.com/cognitivecomputations/spectrum/tree/main/model_snr_results): Contribute to cognitivecomputations/spectrum development by creating an account on GitHub.

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-dev**](https://discord.com/channels/1104757954588196865/1104758010959634503/1297939346070179871) (2 messages):

> - `Qwen2 support for DoRA/QDoRA`
> - `Answer.AI's QDoRA repo`

- **Request for Qwen2 DoRA Support**: A member is looking for any existing development for **Qwen2** support of **DoRA/QDoRA**, noting the lack of traffic in the channel.
  
  - They referenced the [**Answer.AI's QDoRA repository**](https://github.com/AnswerDotAI/fsdp_qlora/tree/main?tab=readme-ov-file#add-support-for-a-new-model) as a potential starting point for implementation.
- **No Active Development on Qwen2 DoRA**: Another member confirmed that there are currently no active branches for **DoRA** support specific to **Qwen2**.
  
  - They encouraged moving forward with the implementation, expressing optimism with a friendly tone.

 

**Link mentioned**: [GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP](https://github.com/AnswerDotAI/fsdp_qlora/tree/main?tab=readme-ov-file#add-support-for-a-new-model.): Training LLMs with QLoRA + FSDP. Contribute to AnswerDotAI/fsdp_qlora development by creating an account on GitHub.

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general-help**](https://discord.com/channels/1104757954588196865/1110594519226925137/1297885140416204902) (1 messages):

> - `Training Domain-Specific LLMs`
> - `Fine-tuning LLMs`
> - `Instruct Models`

- **Training LLMs for Domain-Specific Data**: A member is working on **training and finetuning LLMs** for domain-specific data such as **math**, **legal**, and **finance**.
  
  - They expressed interest in **discussing** the benefits of finetuning an already instruct model like **llama-70b-instruct** instead of starting with a non-instruct model.
- **Fine-tuning Strategies for LLMs**: The conversation highlighted the approach of starting with **finetuning a base non-instruct model** on domain instruction datasets.
  
  - The member indicated that this method could be improved by finetuning on top of an existing instruct model for enhanced performance.

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1296946515037917184) (38 messages🔥):

> - `Meta's FAIR Research`
> - `Attention Mask Issues in Torch`
> - `Flex Attention Challenges`
> - `Performance Warnings in PyTorch`
> - `Mask Construction Discussions`

- **Meta's FAIR Team pushes for Advanced Machine Intelligence**: Meta’s FAIR team is sharing their goal of achieving **advanced machine intelligence (AMI)** to enhance productivity and innovation, as noted in Mark Zuckerberg's recent [open letter](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/). Their commitment spans over a decade of collaboration with the AI community towards **open science** and reproducibility.
  
  - This research effort coincides with discussions around whether similar tools like **Lingua** are comparable to **Torchtune**.
- **Attention Mask Construction and Flex Attention**: Members discussed the complexities in **mask construction** for attention mechanisms, especially the need for different block masks based on attention types, as highlighted by recent implementation challenges. The suggestion was made to handle mask materialization during the forward pass to simplify the **collate** process.
  
  - This emphasizes the importance of maintaining a clean implementation while addressing issues with **packed datasets** and the requirement for custom collates.
- **Performance Warnings in PyTorch**: Users are encountering warnings related to **cuDNN SDPA** on certain data types that led to questions about underlying performance issues and potential fixes. Testing with different kernels may help assess any performance impact, particularly in the context of recently reported issues on the **PyTorch GitHub**.
  
  - The discussion highlighted efforts to potentially file an issue on the **PyTorch core** to address the persistent warnings and their implications.
- **Discussion on Document IDs and Packed Datasets**: The conversation touched on whether **document IDs** could be precomputed while constructing the **PackedDataset**, which may enhance the efficiency of processing workloads with **packed=True**. This proposes an optimization strategy for future implementations.
  
  - Such strategies aim to consolidate the logic around mask generation, possibly leading to better performance and cleaner code paths in attention mechanism handling.
- **General Agreement on Collaboration and Documentation**: Participants agreed on the necessity to document ongoing discussions regarding attention issues and potential solutions on the **GitHub** to prevent important insights from being lost. This led to the creation of an issue summarizing key points around mask construction and attention dispatch problems.
  
  - The importance of collaboration was echoed, particularly how improved documentation can streamline future transitions in development processes.

**Links mentioned**:

- [no title found](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/): no description found
- [pytorch/aten/src/ATen/native/cudnn/MHA.cpp at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/MHA.cpp#L677)): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [torchtune/torchtune/modules/attention_utils.py at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/attention_utils.py#L133): PyTorch native finetuning library. Contribute to pytorch/torchtune development by creating an account on GitHub.
- [Mask construction & attention dispatch issues and possible ideas to allow for more models · Issue #1870 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1870): Since torch 2.5.0, training with packed=True and attention dropout > 0.0 is not possible because padded_collate_packed automatically chooses to build BlockMasks if flex is available (which will gen...
- [[Bug] Unusual CPU overhead of SDPA call on H100 on torch nightly · Issue #1652 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1652): Issue identified: cuDNN SDPA JIT recompiles when the context length changes. This results in training that does not use packing to keep recompiling, resulting in the observed 500ms overhead. There ...
- [F.sdpa stride bug](https://gist.github.com/mirceamironenco/0d39d1976daa62fdded02a76ef826980): F.sdpa stride bug. GitHub Gist: instantly share code, notes, and snippets.
- [pytorch/torch/nn/attention/flex_attention.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py#L873): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1297929204259750030) (1 messages):

> - `v0.4.0 code freeze`
> - `New features in v0.4.0`
> - `Release timeline`

- **Countdown to v0.4.0 code freeze starts!**: With only **8 days** left until the **v0.4.0 code freeze** on **October 29th**, developers are eager to finalize outstanding tasks.
  
  - Preparation is crucial as [*v0.4.0 Tracker*](https://github.com/pytorch/torchtune/issues/1747) lists the estimated release date as **November 5th**.
- **New features lined up for v0.4.0**: New features discussed for the upcoming release include highlights from issues **#1645**, **#1847**, and **#1835**.
  
  - Contributors @felipemello1 and @Optimo are leading the charge, ensuring exciting updates for users.

 

**Link mentioned**: [v0.4.0 Tracker · Issue #1747 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1747): Estimated release date: Tuesday, November 5th Estimated branch cut date (aka code freeze): Tuesday, October 29th Release owner: @joecummings New features: #1645 #1847 (@felipemello1) #1835 (@Optimo...

 

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1296935286797893663) (11 messages🔥):

> - `Pydantic All-in-One`
> - `DSPy GPTs`
> - `AI Agents in Production Event`
> - `Streaming and Bot Notifications`
> - `HotpotQA Alternate History Generator`

- **Pydantic All-in-One Live Stream**: A member started a live stream on [pydantic-all-in-one](https://discord.com/channels/1161519468141355160/1161519469777133580), sharing their thought process during the creation of Python packages and frameworks.
  
  - They also indicated plans to develop **llmodel** following the stream.
- **DSPy GPTs Get a Tutorial Boost**: Members discussed the potential for a tutorial video on utilizing the various **DSPy GPTs** efficiently, emphasizing the benefits for both newcomers and seasoned users in the community.
  
  - The creator agreed to consider this, highlighting the ongoing community support.
- **AI Agents Are Moving from R&D to Reality Event**: A member announced a virtual event on **November 13**, featuring notable speakers like Tomas Wolf and Nathan Benaich, focused on deploying AI agents in production environments.
  
  - The event, organized by **Prosus AI and MLOps**, aims to cover challenges in memory management and real-world applications across different sectors.
- **Streaming Updates and Server Changes**: While discussing notifications related to streaming, **seanchatmangpt** revealed plans to move to a larger server and integrate both **YouTube** and **Twitch** functionalities by November.
  
  - They also expressed enthusiasm for the bot that will provide live notifications, exciting the community.
- **HotpotQA Alternate History Generator Overview**: A member shared an overview of the **HotpotQA Alternate History Generator**, indicating its sophisticated system designed for creating plausible alternate historical scenarios.
  
  - The generator employs advanced **NLP techniques** and large language models for generating and optimizing narratives.

**Links mentioned**:

- [HotpotQA Alternate History Generator](https://jmanhype.github.io/HotpotQA-Alternate-History-Generator/): no description found
- [AI Agents in Production - Event | MLOps Community](https://home.mlops.community/home/events/aiagentsinprod): no description found
- [GitHub - seanchatmangpt/pydantic-all-in-one: All my favorite Pydantic projects connected.](https://github.com/seanchatmangpt/pydantic-all-in-one): All my favorite Pydantic projects connected. Contribute to seanchatmangpt/pydantic-all-in-one development by creating an account on GitHub.

---

### **DSPy ▷ #**[**papers**](https://discord.com/channels/1161519468141355160/1203568372667645963/1297961078935916704) (2 messages):

> - `LightRAG tutorial`
> - `GraphRAG observations`
> - `Ollama integration`
> - `R2R insights`
> - `Microsoft's local search`

- **Step-by-step LightRAG Tutorial with Ollama**: A YouTuber offers a detailed [tutorial](https://www.youtube.com/watch?v=g21royNJ4fw&t=10s) on setting up and running **LightRAG**, a retrieval augmented generation system with **Ollama**.
  
  - The video description highlights that **LightRAG** combines knowledge graphs with embedding-based retrieval for enhanced functionality.
- **R2R Observations on LightRAG vs GraphRAG**: Members shared insights regarding the R2R implementation of **GraphRAG**, noting that the paper's evaluation methodology has significant flaws by benchmarking against Microsoft's global search without proper acknowledgment.
  
  - They raised concerns about scalability due to the low and high-level keys approach and questioned the performance of datasets exceeding **5 million tokens**.
- **Paper Link Preference for Implementation Details**: A member expressed a preference for linking the original paper discussing the **LightRAG** repo over the YouTube video tutorial.
  
  - This approach provides more comprehensive implementation details critical for understanding the application of the technology.

 

**Link mentioned**: [Local LightRAG: A GraphRAG Alternative but Fully Local with Ollama](https://www.youtube.com/watch?v=g21royNJ4fw&t=10s): In this video, we explore how to set up and run LightRAG—a retrieval augmented generation (RAG) system that combines knowledge graphs with embedding-based re...

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1296922622759669780) (18 messages🔥):

> - `DSPy with Hugging Face models`
> - `Ollama usage for model deployment`
> - `AGI House hackathons`
> - `Building LLMs using DSPy`
> - `SGLang for model inference`

- **Building a LRM using DSPy**: A community member is exploring how to build a **LRM using DSPy** and noted the need for efficient token management during model application.
  
  - They highlighted the dropping costs of models like **GPT-4**, making it more feasible to develop robust applications.
- **Using Ollama for Hugging Face models**: Community members discussed **Ollama** as a solution to run finetuned Hugging Face models, providing a step-by-step guide for easier integration.
  
  - This includes downloading models in GGUF format and configuring **DSPy** with Ollama for a streamlined experience.
- **Upcoming AGI House Hackathons**: AGI House announced two events, including a **hackathon** focused on OpenAI’s O1 models and a **Werewolf tournament**, both aimed at fostering innovative AI projects.
  
  - Community members expressed interest in participating and forming teams to showcase **DSPy** capabilities during these events.
- **Challenges with Hugging Face models**: A member reported confusion while running a **finetuned Hugging Face model**, frequently encountering connection errors when trying to integrate with **DSPy**.
  
  - Others suggested resources and configuration steps to alleviate these issues, emphasizing the community's support.
- **SGLang for faster inference**: Suggestions to utilize **SGLang** for faster inference processing in models were shared, including installation commands and server launch configurations.
  
  - Community support offered insights into using **FastInfer** for further optimization.

**Links mentioned**:

- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102): no description found
- [AGI House](https://app.agihouse.org/events/think-slow-and-think-deep-hackathon-20241102)): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109): no description found
- [AGI House](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)): no description found
- [Huggingface | liteLLM](https://docs.litellm.ai/docs/providers/huggingface): LiteLLM supports the following types of Hugging Face models:
- [Drop o1 Preview, Try This Alternative](https://www.lycee.ai/blog/drop-o1-preview-try-this-alternative): Building robust LLM-based applications is token-intensive. You often have to plan for the parsing and digestion of a lot of tokens for summarization or even retrieval augmented generation. Even the me...

---

### **DSPy ▷ #**[**examples**](https://discord.com/channels/1161519468141355160/1161519685616025600/1297282668685426828) (3 messages):

> - `Hosting models on Hugging Face`
> - `Running DSPy modules`

- **Using Local Models Hosted on Hugging Face**: A user inquired about how to use a local model hosted on Hugging Face as a language model to run DSPy modules.
  
  - The discussion indicates a need for clarity on the integration process, specifically what tools or configurations are required for this setup.
- **Clarification on Hosted Model Integration**: Another user referenced the conversation regarding local models hosted on Hugging Face, indicating additional support was provided.
  
  - This suggests that further details were shared in a separate message thread to assist with configuration.

 

---

### **DSPy ▷ #**[**colbert**](https://discord.com/channels/1161519468141355160/1250300504462856265/1297551390901800990) (3 messages):

> - `AcgNDCG pseudo-function`
> - `BM25 retriever inquiry`
> - `AvgNDCG DSPy Metric`
> - `PATH first author outreach`

- **Clarification on AcgNDCG document retrieval**: A member questioned whether the retriever retrieves documents specifically from the set of **10ish Relevance Judgements** (J) or from a broader pool of documents, referencing the paper [here](https://arxiv.org/pdf/2406.11706).
  
  - *Does it retrieve from a specific list or the entire pool?* remains an open query.
- **BM25's Role in Model Flexibility**: There was a discussion confirming that **BM25** is not special as a retriever, and any other retriever could be used as long as it's different from the encoder being trained.
  
  - Thus, *using a different model for reranking* should be permissible.
- **AvgNDCG Metric Implementation**: A member expressed uncertainty about whether **AvgNDCG** was implemented as a **DSPy Metric** in the referenced paper, stating clarity would help before pursuing an implementation.
  
  - *Metrics typically compare examples and predictions*, so confirmation is crucial.
- **Collaboration with PATH first author**: A member encouraged reaching out to the **PATH** first author for assistance regarding the questions raised, offering to be cc'd in the communication.
  
  - *We’d be happy to help* was highlighted as a supportive invitation for clarification.

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1298003872929021972) (1 messages):

> - `Lecture 7`
> - `TapeAgents framework`
> - `WorkArena++ benchmark`
> - `Nicolas Chapados`

- **Lecture 7 on AI Agents is Today**: The **7th lecture** in the series is scheduled for today at **3:00pm PST** and can be streamed [here](https://www.youtube.com/live/-yf-e-9FvOc).
  
  - Guest speakers **Nicolas Chapados** and **Alexandre Drouin** will discuss **AI Agents for Enterprise Workflows** during the presentation.
- **Introduction of the TapeAgents Framework**: The lecture will introduce the **TapeAgents framework**, enabling **resumable** and **optimizable** agents using a unifying abstraction known as the Tape.
  
  - This framework aims to enhance the capabilities of tool-using agent architectures significantly.
- **Unveiling WorkArena++ for Web Agents**: **WorkArena++** is a newly developed benchmark for web agents, focusing on their performance in enterprise environments and knowledge worker tasks.
  
  - The framework tracks the progress of web agents in accomplishing varied tasks autonomously, posing new challenges for the field.
- **Nicolas Chapados' Background**: **Nicolas Chapados**, the Vice-President of Research at ServiceNow, has extensive experience leading generative AI advancements for enterprises. He has co-founded multiple startups, notably **Element AI**, acquired by ServiceNow in **2021**.

 

**Link mentioned**: [CS 194/294-196 (LLM Agents) - Lecture 7](https://www.youtube.com/live/-yf-e-9FvOc.): no description found

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1297016755285397534) (33 messages🔥):

> - `Certification for Course Completion`
> - `Project Development Strategies`
> - `Hackathon Participation`
> - `Written Article Assignment`
> - `Local LLM Running Options`

- **Certification for Course Completion**: Course staff confirmed that students will receive a certificate upon completing all requirements, including quizzes and the written article assignment, due by **December 12**.
  
  - Students can also catch up on materials using **course recordings and slides**.
- **Strategies for Project Development**: A participant sought guidance on whether to focus on understanding concepts or to start working on projects using frameworks discussed in the seminar.
  
  - The consensus suggested a **combination of both** approaches for a comprehensive learning experience.
- **Hackathon Participation Open to All**: It was confirmed that students from other universities, such as **UIUC**, can participate in the hackathon without needing to enroll in the course.
  
  - One member specifically noted that participation in the hackathon is independent of course registration, but assignments are still applicable.
- **Clarification on Written Article Assignment**: The written article assignment requires students to create a post or an article summarizing lecture content or hackathon experiences, to be submitted via a provided link.
  
  - A clear **500-word** guideline was issued, indicating an effort-based grading (P/NP) format for this assignment.
- **Running LLMs Locally**: Participants were provided different options for running LLMs locally, with **Ollama** and **LM Studio 0.3.0** being noted as practical tools.
  
  - Users were cautioned that running larger models generally requires more than **8GB of RAM**.

**Links mentioned**:

- [Large Language Model Agents](https://llmagents-learning.org/f24): no description found
- [Written Article Assignment Submission](https://forms.gle/7ekobPNSWDLBWnDT6): INSTRUCTIONS: Create a Twitter, Threads, or LinkedIn post of roughly 500 words. You can post this article directly onto your preferred platform or you can write the article on Medium and then post a l...

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1297658684503101452) (3 messages):

> - `Orchestration of agents`
> - `Lecture timing`

- **Research on Orchestration of Agents**: There’s an active discussion regarding the **orchestration of agents** in the **Agentic System**, highlighting it as a significant area of current research.
  
  - Members seem eager to explore further advancements and findings in this domain.
- **Lecture Schedule Confirmed**: The schedule for today’s session is confirmed to be from **3-5pm PST every Monday**.
  
  - This timing allows participants to plan accordingly for future lectures.

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1297648895228252210) (4 messages):

> - `LibreFLUX release`
> - `FLUX.1-schnell comparison`
> - `Open source characteristics`
> - `Community reactions`

- **LibreFLUX launches with new features**: The release of **LibreFLUX**, an Apache 2.0 version of [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), provides a full T5 context length, enhanced attention masking, and classifier free guidance restored.
  
  - It prioritizes **open-source tenets**, making it easier to fine-tune for new distributions while adopting a clunkier aesthetic reminiscent of the early 2000s.
- **Context length and de-distilled features noted**: LibreFLUX is characterized as a mostly **de-distilled version** of **schnell** with a 512 token length context and attention masking.
  
  - *Community members reacted positively*, expressing excitement over the release and acknowledging the efforts made in its development.

 

**Link mentioned**: [jimmycarter/LibreFLUX · Hugging Face](https://huggingface.co/jimmycarter/LibreFLUX): no description found

 

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/1297016011572252777) (12 messages🔥):

> - `Open-MUSE training issues`
> - `Microsoft LLM breakthrough`
> - `BitNet model`
> - `Training logs for MUSE project`

- **Difficulty in Open-MUSE model training**: A user reported issues finding models like **openMUSE/maskgit-vqgan-imagenet-f16-256** on Hugging Face but was directed to [the renamed checkpoints](https://huggingface.co/amused). Additionally, they encountered a missing key error in their training configuration file when running their script.
  
  - They provided a link to the configuration YAML at [W&B](https://wandb.ai/psuraj/muse/runs/3ef2rhq3/files/config.yaml) for further discussion.
- **Microsoft claims performance leap for LLMs**: A claim surfaced that Microsoft can now run **100B parameter models** on local devices with up to **6x speed improvements** and **82% energy reduction** without needing a GPU, discussed in a [Reddit post](https://www.reddit.com/r/singularity/comments/1g768xk/microsoft_llm_breakthrough_you_can_now_run_100b/).
  
  - This information reportedly originated from a tweet detailing the post's claims, which can be referenced [here](https://x.com/jenzhuscott/status/1847514413060046855).
- **No existing 100B models with BitNet yet**: While discussing Microsoft's LLM advancements, it was noted that no **100B models** are available utilizing **BitNet** despite the recent claims of efficiency. Users are cautious about the actual implementation and capabilities regarding the cited performance figures.
- **Open reproduction effort for MUSE**: Multiple users discussed the open reproduction of the **MUSE** text-to-image model and shared resources like the [GitHub repository](https://github.com/huggingface/muse) and [W&B Project](https://wandb.ai/psuraj/muse?workspace=user-). This project aims to provide a detailed approach for text-to-image generation through a transparent sharing of training processes.
  
  - Key steps outlined for the project included training various models on datasets like **imagenet** and conducting experiments on **CC12M**.

**Links mentioned**:

- [amused (Open Reproduction of MUSE)](https://huggingface.co/amused): no description found
- [Tweet from Jen Zhu (@jenzhuscott)](https://x.com/jenzhuscott/status/1847514413060046855): 2/ you can now run 100B parameter models on local devices with up to 6x speed improvements and 82% less energy consumption—all w/out a GPU! Local, efficient, private, blazing fast, open sourced 🔥 🔥 ...
- [psuraj](https://wandb.ai/psuraj/muse/runs/3ef2rhq3/files/config.yaml): Weights & Biases, developer tools for machine learning
- [Reddit - Dive into anything](https://www.reddit.com/r/singularity/comments/1g768xk/microsoft_llm_breakthrough_you_can_now_run_100b/): no description found
- [open-muse/README.md at main · huggingface/open-muse](https://github.com/huggingface/open-muse/blob/main/README.md): Open reproduction of MUSE for fast text2image generation. - huggingface/open-muse

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1296912502189330453) (10 messages🔥):

> - `Aider incremental improvements`
> - `Open Interpreter equivalent of /functions folder`
> - `Custom tools support`
> - `Python virtual environments`
> - `Integrating voice assistants`

- **Aider adopts AI-generated code progressively**: Members noted that **Aider** enhances its use of AI-generated and honed code with each version, suggesting a trend towards living nightly builds of interpreter concepts.
  
  - There is a curiosity about whether **Open Interpreter** plans to implement a similar approach in the future.
- **Inquiry on OI's equivalent to /functions folder**: A user asked if there is an **Open Interpreter** equivalent to the **/functions** folder from shell-gpt, which allows users to add schema prebuilt functions for easy access.
  
  - Another member expressed that the only way to add custom tools at the moment might require editing the repository.
- **Discussion on Custom Tools for Open Interpreter**: One member expressed interest in adding **custom tools** to Open Interpreter, offering to make pull requests if the feature is desired by the community.
  
  - However, it was noted that currently customizing tools may involve significant code changes.
- **Python virtual environments support inquiries**: A user inquired about the possibility of adding support for **virtual environments** in the Python kernel, proposing a simple attribute addition to the Interpreter class.
  
  - There was uncertainty on whether this would benefit most users, but the member felt it could facilitate package installation in a venv.
- **Voice assistant integration into agents**: [AIwithBenefits](https://x.com/AIwithBenefits/status/1848161437828415578) discussed adding a **HumeAI voice assistant** to the **phidatahq** generalist agent, enhancing its functionality with AppleScript execution.
  
  - The new **phidatahq UI** was praised, highlighting improved usability in native app interactions.

 

**Link mentioned**: [Tweet from Jacob@AIwithBenefits (@AIwithBenefits)](https://x.com/AIwithBenefits/status/1848161437828415578): Added a @hume_ai voice assistant to the @phidatahq generalist agent, and a little help from the @OpenInterpreter system message. Quoting Jacob@AIwithBenefits (@AIwithBenefits) Loving the new @phid...

 

---

### **OpenInterpreter ▷ #**[**O1**](https://discord.com/channels/1146610656779440188/1194880263122075688/1297208709772218370) (1 messages):

> - `OpenInterpreter Mac setup`
> - `Interaction issues`
> - `LiveKit Meet link concerns`

- **Successful OpenInterpreter Mac Setup**: A user confirmed a successful setup of OpenInterpreter on their Mac, stating that [localhost:10100](http://localhost:10100) works correctly to control their system.
  
  - This indicates that the initial configuration was done properly, allowing remote control features.
- **Web Browser Access Denied**: The user reported receiving a message stating, *“Sorry, I can’t access your web browser, etc., but I can guide you.”* during their interaction attempts.
  
  - This suggests potential limitations in web access capabilities within the OpenInterpreter setup on their device.
- **LiveKit Meet Link Doesn't Work**: The user shared that neither the app nor the [LiveKit Meet link](https://link.to/livekit) could access their computer for functionality.
  
  - This raises concerns about compatibility or permissions when using these features on their Mac.

 

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1297013701559451648) (9 messages🔥):

> - `LangGraph Code Assistant`
> - `Role-based RAG Models`
> - `Expected Context Issues`
> - `Techstars Startup Weekend Event`
> - `Code Generation Approaches`

- **LangGraph Code Assistant Implementation Steps**: The **LangGraph Code Assistant** tutorial outlines a method to iteratively construct answers to coding questions using [AlphaCodium](https://github.com/Codium-ai/AlphaCodium) and RAG techniques.
  
  - *The process includes ingesting user-specified documentation, invoking tools for structured output, and conducting unit tests before returning solutions.*
- **Considerations for Role-based RAG Implementation**: A member inquired about splitting **RAG models** based on user roles, allowing specific access to financial documents for CEOs while limiting interns to relevant documents.
  
  - This approach raises questions about how to effectively manage and restrict access while using RAG models.
- **Troubleshooting Context Retrieval**: A user expressed difficulty in obtaining the expected context for queries despite having the information stored in the vector database.
  
  - Advice was given to check **embeddings** or to refine the prompt for better outcomes.
- **Techstars Startup Weekend SF Announcement**: The **Techstars Startup Weekend SF** invites the tech community to the [AWS GenAI Loft](https://aws.amazon.com/startups/lp/aws-gen-ai-loft-san-francisco?lang=en-US) for networking and connections after TechCrunch Disrupt.
  
  - The event features talks from industry experts, followed by networking opportunities for founders, investors, and innovators.
- **Code Generation Strategy Discussion**: A participant discussed **AlphaCodium**'s approach for code generation, emphasizing iterative testing through public and AI-generated tests.
  
  - They outlined the process, including how to use `code_gen_chain.invoke()` for reflection and code solution generation.

**Links mentioned**:

- [Tweet from UltraIA (@Ultra_IA)](https://x.com/Ultra_IA/status/1847821253476008227): LOL
- [Code Assistant](https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/#code-solution): no description found
- [TC Disrupt AI Founders Happy Hour by Techstars Startup Weekend SF @ AWS GenAI Loft · Luma](https://lu.ma/5f5ydtxq): Head over to the ASW GenAI Loft for an exclusive evening of real conversations and genuine connections. We’re bringing in a top mind from the AI space (details…

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1297706150489362483) (1 messages):

> - `OpenAI Swarm`
> - `LangChain LangGraph`
> - `Multi-Agent Frameworks`

- **Comparing OpenAI Swarm and LangChain LangGraph**: A detailed article compares **OpenAI Swarm** and **LangChain LangGraph**, focusing on their functionalities and best use cases for building complex AI workflows.
  
  - This resource aims to guide readers in determining which framework might be the **right fit** for their projects, accessible [here](https://medium.com/ai-artistry/openai-swarm-vs-langchain-langgraph-a-detailed-look-at-multi-agent-frameworks-0f978a4ca203?sk=06fad63e6089bc2d0e772b2101b4f474).
- **Importance of Multi-Agent Workflows**: The message highlights the increasing importance of creating **multi-agent workflows** in the evolving field of artificial intelligence.
  
  - Such frameworks enable developers to navigate complex interactions and processes, enhancing overall **AI capabilities**.

 

**Link mentioned**: [OpenAI Swarm vs LangChain LangGraph: A Detailed Look at Multi-Agent Frameworks](https://medium.com/ai-artistry/openai-swarm-vs-langchain-langgraph-a-detailed-look-at-multi-agent-frameworks-0f978a4ca203?sk=06fad63e6089bc2d0e772b2101b4f474): Ankush k Singal

 

---

### **MLOps @Chipro ▷ #**[**events**](https://discord.com/channels/814557108065534033/869270934773727272/) (1 messages):

huikang: [https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109](https://app.agihouse.org/events/agi-thon-werewolf-agents-tournament-20241109)

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1298017406215786526) (1 messages):

> - `AI access challenges`
> - `Competition in AI`
> - `External researcher access`
> - `Big Tech and AI`
> - `Open AI ecosystem`

- **Mozilla commissions research on AI access challenges**: Mozilla has commissioned two insightful pieces of research on the challenges surrounding **AI access** and competition: [External Researcher Access to Closed Foundation Models](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/External-researcher-access-to-closed-foundation-models.pdf) and [Stopping Big Tech From Becoming Big AI](https://blog.mozilla.org/wp-content/blogs.dir/278/files/2024/10/Stopping-Big-Tech-from-Becoming-Big-AI.pdf). These reports, sourced from **AWO** and the **Open Markets Institute**, focus on the dynamics of AI control and the necessary changes for a fair ecosystem.
- **Understanding control in AI development**: The research highlights **who's in control** of AI development and emphasizes what reforms are needed to ensure an equitable environment. They underline the importance of **external researchers** gaining access to closed models for broader innovation.
  
  - As noted in the reports, ensuring a level playing field is crucial for sustaining innovation in the rapidly evolving AI landscape.
- **Blog post detailing AI research findings**: Further information about the commissioned research can be found in the [blog post here](https://discord.com/channels/1089876418936180786/1298015953463808102). This blog discusses the implications of the findings in the context of current AI governance.

 

---

### **DiscoResearch ▷ #**[**general**](https://discord.com/channels/1178995845727785010/1182877486854451271/) (1 messages):

huunguyen: has anyone tried q-galora?

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
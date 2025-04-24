---
id: c70df8df-638b-4e98-b325-007ce112f82e
title: How To Scale Your Model, by DeepMind
date: '2025-02-05T06:59:23.438232Z'
original_slug: ainews-how-to-scale-your-model-by-deepmind
description: >-
  AI news from February 3-4, 2025, covers model scaling, new research papers, AI
  tools, and platform updates. Highlights include a textbook on model scaling,
  new models and methods in robotics and LLMs, bias studies, and the launch of
  Hugging Face's AI app store.
companies:
  - gdm
  - jax-ml
  - deepseek
  - hugging face
models:
  - gemini
  - qwen-0.5
topics:
  - model-scaling
  - transformers
  - inference
  - roofline
  - humanoid-robotics
  - sim2real
  - mixture-of-agents
  - self-moa
  - llm-training
  - bias-in-llms
  - rust-machine-learning
  - ai-platforms
---


<!-- buttondown-editor-mode: plaintext -->**Systems thinking is all you need.**

> AI News for 2/3/2025-2/4/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **34** Discords (**225** channels, and **3842** messages) for you. Estimated reading time saved (at 200wpm): **425 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

In a surprise drop, some researchers released a "little textbook" on how they scale models at GDM:

![image.png](https://assets.buttondown.email/images/42cd0862-439a-4053-b038-aee3f6fdc947.png?w=960&fit=max)

A commenter [confirmed](https://news.ycombinator.com/item?id=42938185) this was GDM internal documentation, with Gemini references redacted.

[How To Scale Your Model](https://jax-ml.github.io/scaling-book/) comes in 12 parts and starts with a nice update of what standard Transformers today look like:

![image.png](https://assets.buttondown.email/images/84641eb8-3eaa-42cc-91c6-925ece7928de.png?w=960&fit=max)

and [explains how inference differs from the standard O(N^2) understanding of attention](https://jax-ml.github.io/scaling-book/inference/):

![image.png](https://assets.buttondown.email/images/e1107b18-2dce-4878-8cb1-06760288a65a.png?w=960&fit=max)



but also introduces standard high performance computing concepts like [rooflines](https://jax-ml.github.io/scaling-book/roofline/):

![image.png](https://assets.buttondown.email/images/8848c83c-eb3a-4e6f-afca-f5a22a8beb38.png?w=960&fit=max)

even coming with worked problems for the motivated reader to test their understanding... and comments are being read in realtime.

![image.png](https://assets.buttondown.email/images/51215b48-6bed-45b7-aad9-291780fb5910.png?w=960&fit=max)



---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**AI Model Releases and Research Papers**

- **"ASAP": A Real2Sim2Real Model for Humanoid Robotics**: [@DrJimFan](https://twitter.com/DrJimFan/status/1886824152272920642) announced "**ASAP**," a model that enables humanoid robots to perform fluid motions inspired by **Cristiano Ronaldo**, **LeBron James**, and **Kobe Bryant**. The team, including [@TairanHe99](https://twitter.com/DrJimFan/status/1886824977191854327) and [@GuanyaShi](https://twitter.com/DrJimFan/status/1886824977191854327), has **open-sourced** the **paper and code** for this project. The approach combines real-world data with simulation to overcome the "**sim2real**" gap in robotics.

- **"Rethinking Mixture-of-Agents" Paper and "Self-MoA" Method**: [@omarsar0](https://twitter.com/omarsar0/status/1886792384954163347) discussed a new paper titled "**Rethinking Mixture-of-Agents**," which questions the benefits of mixing different **LLMs**. The proposed "**Self-MoA**" method leverages in-model diversity by aggregating outputs from the top-performing LLM, outperforming traditional MoA approaches. The **paper** can be found [here](https://twitter.com/omarsar0/status/1886792397235085383).

- **Training LLMs with GRPO Algorithm from DeepSeek**: [@LiorOnAI](https://twitter.com/LiorOnAI/status/1886850811378196685) highlighted a new **notebook** that demonstrates training a reasoning LLM using the **GRPO algorithm** from **DeepSeek**. In less than 2 hours, you can transform a small model like **Qwen 0.5** (*500 million parameters*) into a **math reasoning machine**. [Link to notebook](https://twitter.com/LiorOnAI/status/1886850813911556351).

- **Bias in LLMs Used as Judges**: [@_philschmid](https://twitter.com/_philschmid/status/1886717030218297406) shared insights from the paper "**Preference Leakage: A Contamination Problem in LLM-as-a-Judge**," revealing that LLMs can be **significantly biased** when used for synthetic data generation and evaluation. The study emphasizes the need for **multiple independent judges** and **human evaluations** to mitigate bias. [Paper](https://twitter.com/_philschmid/status/1886717032378372164).

- **mlx-rs: Rust Library for Machine Learning**: [@awnihannun](https://twitter.com/awnihannun/status/1886846423905575330) introduced **mlx-rs**, a Rust library that includes examples of **text generation with Mistral** and **MNIST training**. This is a valuable resource for those interested in **Rust** and **machine learning**. [Check it out](https://twitter.com/awnihannun/status/1886846423905575330).

**AI Tools and Platforms Announcements**

- **Hugging Face's AI App Store Launched**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1886861567326650526) announced that **Hugging Face** has launched its **AI app store** with **400,000 total apps**, including **2,000 new apps** daily and **2.5 million weekly visits**. Users can now **search** through apps using AI or categories, emphasizing that "**the future of AI will be distributed**." [Explore the app store](https://twitter.com/ClementDelangue/status/1886861567326650526).

- **AI App Store Announcement**: [@_akhaliq](https://twitter.com/_akhaliq/status/1886831521216016825) echoed the excitement about the launch of the **AI App Store**, stating it's the best place to find the **AI apps** you need, with approximately **400k apps** available. Developers can build apps, and users can discover new ones using **AI search**. [Check it out](https://twitter.com/_akhaliq/status/1886831521216016825).

- **Updates to 1-800-CHATGPT on WhatsApp**: [@kevinweil](https://twitter.com/kevinweil/status/1886988476203126878) announced new features for **1-800-CHATGPT** on **WhatsApp**:
  - You can now **upload images** when asking a question.
  - Use **voice messages** to communicate with ChatGPT.
  - Soon, you'll be able to **link your ChatGPT account** (free, plus, pro) for higher rate limits.
  - [Learn more](https://twitter.com/kevinweil/status/1886988479499776348).

- **Replit's New Mobile App and AI Agent**: [@hwchase17](https://twitter.com/hwchase17/status/1886950917326655740) shared that **Replit** launched a new **mobile app** and made their **AI agent free** to try. The rapid development of Replit's AI Agent is notable, and [@amasad](https://twitter.com/amasad) confirmed the release. [Details here](https://twitter.com/hwchase17/status/1886950917326655740).

- **ChatGPT Edu Rolled Out at California State University**: [@gdb](https://twitter.com/gdb/status/1886884951666340270) reported that **California State University** is becoming the first **AI-powered university system**, with **ChatGPT Edu** being rolled out to **460,000 students** and over **63,000 staff and faculty**. [Read more](https://twitter.com/gdb/status/1886884951666340270).

**AI Events, Conferences, and Hiring**

- **AI Dev 25 Conference Announced**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1886833904235241753) announced **AI Dev 25**, a conference for AI developers happening on **Pi Day (3/14/2025)** in **San Francisco**. The event aims to create a **vendor-neutral meeting** for AI developers, featuring **over 400 developers** gathering to build, share ideas, and network. [Learn more and register](https://twitter.com/AndrewYNg/status/1886833904235241753).

- **Hiring for Alignment Science Team at Anthropic**: [@sleepinyourhat](https://twitter.com/sleepinyourhat/status/1886822563353141303) is **hiring researchers** for the **Alignment Science team** at **Anthropic**, co-led with [@janleike](https://twitter.com/sleepinyourhat/status/1886822564905070823). They focus on **exploratory technical research** on **AGI safety**. Ideal candidates have:
  - Several years of experience as a **SWE** or **RE**.
  - Substantial **research experience**.
  - Familiarity with **modern ML** and the **AGI alignment literature**.
  - [Apply here](https://twitter.com/sleepinyourhat/status/1886822568067498242).

- **INTERRUPT Conference Featuring Andrew Ng**: [@hwchase17](https://twitter.com/hwchase17/status/1886823545122250928) announced that **Andrew Ng** will be speaking at the **INTERRUPT conference** this May. Celebrating Ng as one of the best educators of our generation, attendees are encouraged to learn from him. [Get tickets](https://twitter.com/hwchase17/status/1886823545122250928).

- **Virtual Forum on DeepSeek Integration**: [@llama_index](https://twitter.com/llama_index/status/1886912036766204127) invited developers, engineers, and AI enthusiasts to join a **virtual forum** exploring **DeepSeek**, its capabilities, and integration into workflows. Presenters include representatives from **Google**, **GitHub**, **AWS**, **Vectara**, and **LlamaIndex**. [Register here](https://twitter.com/llama_index/status/1886912036766204127).

**AI Ethics, Safety, and Policy**

- **Google DeepMind Updates Frontier Safety Framework**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1886817852595876104) shared updates to their **Frontier Safety Framework**, a set of protocols designed to mitigate **severe risks** as we progress toward **AGI**. Emphasizing the need for AI to be both **innovative and safe**, they invite readers to [find out more](https://twitter.com/GoogleDeepMind/status/1886817852595876104).

- **Discussion on Bias in LLM Judges**: [@_philschmid](https://twitter.com/_philschmid/status/1886717030218297406) addressed the issue of **bias in LLMs** when used for **synthetic data generation** and as judges. The "**Preference Leakage**" paper reveals that LLMs can favor data generated by themselves or their previous versions, highlighting a **contamination problem**. [Read the paper](https://twitter.com/_philschmid/status/1886717032378372164).

- **OpenAI's Frontier Safety Framework Updates**: [@OpenAI](https://twitter.com/OpenAI/status/1886966048970478016) announced new updates to their **Frontier Safety Framework**, aiming to stay ahead of potential severe risks associated with advanced AI systems.

**General AI Industry Commentary**

- **Yann LeCun on Small Teams and Innovation**: [@ylecun](https://twitter.com/ylecun/status/1886677032324509898) emphasized that **small research teams** with autonomy are empowered to make the right technical choices and innovate. He highlighted the importance of organization and management in fostering innovation within **R&D organizations**.

- **DeepSeek Compared to Sputnik Moment**: [@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1886784302513377379) compared the news about **DeepSeek** to a modern "**Sputnik 2.0**," implying it's a significant milestone in AI, similar to the historic space race event.

- **Reflections on Technology Adoption**: [@DavidSHolz](https://twitter.com/DavidSHolz/status/1886920976270774623) commented on how new technologies are often initially used to replicate old mediums, stating, "**These mistakes happen when you don't respect that new inventions are also new mediums**."

- **Discussions on AI Evaluation and RL**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1886810699726275041) observed that **few-shot prompting** degrades performance in **DeepSeek-R1**, likely due to the model's training on a strict format. This points to new paradigms in interacting with LLMs and the evolving landscape of AI techniques.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. DeepSeek R1 & R1-Zero: Rapid Model Training Achievements**

- **[Deepseek researcher says it only took 2-3 weeks to train R1&R1-Zero](https://www.reddit.com/gallery/1ihd0rr)** ([Score: 800, Comments: 127](https://reddit.com/r/LocalLLaMA/comments/1ihd0rr/deepseek_researcher_says_it_only_took_23_weeks_to/)): **Deepseek's** researcher claims that **R1** and **R1-Zero** models were trained in just **2-3 weeks**, suggesting a rapid development cycle for these AI models.
  - Discussions highlight skepticism about the **R1 and R1-Zero's rapid training** of 10,000 RL steps in 3 weeks, with some users questioning the feasibility and others suggesting that fine-tuning existing models like V3 could explain the speed. Concerns include potential bottlenecks in API and website performance due to high demand and the need for improved training data or architectures.
  - Users compare **Deepseek's models** to other AI advancements, noting the potential for new models to emerge globally following the release of the paper. Some express preference for Deepseek over **OpenAI** due to perceived openness and lack of tariffs, while others anticipate future versions like R1.5 or V3-lite.
  - The conversation touches on the **AI space race**, with comparisons to the global space race, highlighting regional participation disparities. Europe is mentioned as having contributions through companies like **Stable Diffusion** and **Hugging Face**, while other regions are noted for limited involvement, emphasizing the competitive nature of AI development globally.


**Theme 2. DeepSeek-R1 Model: Implications of Shorter Correct Answers**

- **[DeepSeek-R1's correct answers are generally shorter](https://i.redd.it/duiwqfpzq3he1.png)** ([Score: 289, Comments: 66](https://reddit.com/r/LocalLLaMA/comments/1ihf0gb/deepseekr1s_correct_answers_are_generally_shorter/)): **DeepSeek-R1's** correct answers are generally shorter, averaging **7,864.1 tokens** compared to **18,755.4 tokens** for incorrect answers, as depicted in a bar graph. The standard deviation is **5,814.6** for correct solutions and **6,142.7** for incorrect ones, indicating variability in token lengths.
  - **Task Difficulty and Response Length**: Several comments, including those by **wellomello** and **Affectionate-Cap-600**, question whether the analysis accounted for task difficulty, suggesting that harder tasks naturally require longer responses, which can affect error rates and token length averages.
  - **Model Behavior and Standard Deviation**: **FullstackSensei** and **101m4n** discuss the implications of the high standard deviation in token length, suggesting that incorrect answers could result from the model entering loops or struggling with problem-solving, thus extending response time.
  - **Related Research and Generalization**: **Angel-Karlsson** references a relevant research paper on overthinking in models, while **Egoz3ntrum** highlights the importance of considering the dataset's limitations, indicating that conclusions might not generalize well beyond specific math problem difficulties.


**Theme 3. OpenAI Research: Embracing Open-Source via Hugging Face**

- **OpenAI deep research but it's open source** ([Score: 421, Comments: 28](https://reddit.com/r/LocalLLaMA/comments/1ihqwnd/openai_deep_research_but_its_open_source/)): **Hugging Face** has launched an initiative called **OpenAI Deep Research**, making deep research open source. The project aims to democratize access to cutting-edge AI research, emphasizing transparency and collaboration in the AI community. More details can be found in their [blog post](https://huggingface.co/blog/open-deep-research).
  - Users express significant appreciation for the **Hugging Face** team, comparing their contributions to those of the **Mistral** team and highlighting the rapid development pace, with some anticipating integration into platforms like **Open-WebUI** soon. The sense of urgency and surprise is echoed in comments about the swift creation of open-source alternatives to proprietary solutions from **OpenAI**.
  - Discussions highlight the open-source community's gratitude towards **Hugging Face** for providing extensive tooling and frameworks, with some users humorously questioning the motivations behind such generosity. The notion of quickly developing open-source alternatives is a recurring theme, reflecting the community's proactive stance on maintaining open access to AI advancements.
  - A comment provides a link to a GitHub repository for those interested in trying out local implementations, pointing towards **Automated-AI-Web-Researcher-Ollama** as a resource for experimenting with open-source AI tools. This suggests a practical interest in hands-on experimentation with AI research tools among the community.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. OmniHuman-1: China's Multimodal Marvel**

- **[China's OmniHuman-1 🌋🔆](https://v.redd.it/pwjfvxljy4he1)** ([Score: 684, Comments: 174](https://reddit.com/r/OpenAI/comments/1ihjgpk/chinas_omnihuman1/)): **OmniHuman-1** is a Chinese project that focuses on **video generation from single images**. The post lacks additional details, so further context or technical specifics about OmniHuman-1 are not provided.
  - **OmniHuman-1's Capabilities and Concerns**: There is a significant discussion about OmniHuman-1's potential to generate realistic human videos from a single image and audio, with some users expressing concerns about the implications for media authenticity and the potential for indistinguishable synthetic media. The project's details and code base are available on [GitHub](https://github.com/mdsrqbl/omnihuman) and its white paper is accessible at [omnihuman-lab.github.io](https://omnihuman-lab.github.io/).
  - **AI's Impact on Creative Industries**: Some commenters debate AI's influence on creative industries, suggesting AI could lead to a golden age of unique, economically feasible artistic creations, while others express skepticism about AI's ability to replicate the depth of human experiences. Concerns about the future of human-generated creative work and economic impacts, such as the potential need for UBI, are also discussed.
  - **Technical Observations and Challenges**: Users note technical imperfections in AI-generated videos, such as unnatural physical movements and uncanny valley effects, suggesting that while AI has advanced, there are persistent challenges that may require fundamentally different techniques to overcome. The discussion includes the idea that AI-based video will continue evolving through a process of refinement, similar to developments in language models.


**Theme 2. Huawei's Ascend 910C Challenges Nvidia H100**

- **huawei's ascend 910c chip matches nvidia's h100. there will be 1.4 million of them by december. don't think banned countries and open source can't reach agi first.** ([Score: 262, Comments: 99](https://reddit.com/r/OpenAI/comments/1ihebb4/huaweis_ascend_910c_chip_matches_nvidias_h100/)): **Huawei's Ascend 910C chip** reportedly matches **Nvidia's H100** in performance, with plans to produce **1.4 million** units by 2025. This development challenges claims that China and open-source projects are lagging in AI chip technology, suggesting they now have the capability to build top AI models, potentially reaching AGI before major AI companies.
  - **CUDA's Dominance**: Many comments emphasize the importance of **CUDA** in AI development, noting that it's a proprietary platform that deeply integrates with major frameworks like **TensorFlow** and **PyTorch**. While some argue that alternatives like **AMD's ROCm** exist, others believe that replicating CUDA's ecosystem is a significant challenge, though not insurmountable given sufficient investment.
  - **Huawei's Competitive Position**: There is skepticism about claims that **Huawei's Ascend 910C** matches **Nvidia's H100**. Some users argue that the 910C only achieves 60% of the H100's performance, and Huawei's strategy is not to compete directly with Nvidia but to capture market share where Nvidia is restricted, leveraging their own **CANN** platform as a CUDA equivalent.
  - **Market Dynamics and Open Source**: The discussion touches on the potential for **open-source developers** to pivot away from open-source models if they achieve **AGI**. There's a sentiment that Huawei, due to market restrictions, might push their development to catch up with Nvidia, but it could take 3-5 years to reach parity, potentially using third-party channels to access Nvidia hardware in the meantime.


**Theme 3. O3 Mini: OpenAI's Usability Leap**

- **O3 mini actually feels useful** ([Score: 104, Comments: 17](https://reddit.com/r/OpenAI/comments/1ihwez2/o3_mini_actually_feels_useful/)): **O3 Mini**, an open AI model, initially impressed the user by suggesting a smart solution or bug fix for coding issues, which seemed more effective than **O1 (non-pro)**. However, upon further evaluation, the suggested solution did not work as expected.
  - **O3 Mini** initially seemed impressive but failed to deliver effective solutions, indicating that other AI models like **Claude series** might generalize better across tasks. **Mescallan** suggests that most models show spikes in specific benchmarks but lack generalization.
  - **O1 Pro** is considered more reliable for coding tasks, with **MiyamotoMusashi7** expressing trust in it for code-related tasks while acknowledging potential bugs in other areas.
  - **gentlejolt** highlights a workaround for improving code quality by instructing the AI to "rearchitect" and optimize for readability and maintainability, although the end result was only a slightly improved version of the original code.


**Theme 4. OpenAI Unveils OpenAI Sans Font**

- **[Refreshed.](https://www.youtube.com/watch?v=k3d_xeVxEOE)** ([Score: 259, Comments: 140](https://reddit.com/r/OpenAI/comments/1ihrssx/refreshed/)): **OpenAI** has introduced a new font as part of their branding strategy, signaling a refreshed visual identity. The update is part of their ongoing efforts to enhance their brand presence and user engagement.
  - Many comments draw a comparison between **OpenAI**'s new font and **Apple's** design ethos, suggesting that OpenAI's design team might include former Apple UX designers. The design change is seen as a strategic move to own a unique font, similar to Apple's creation of the **San Francisco font**, which reduces long-term licensing costs.
  - There is skepticism about the need for a new font, with comments suggesting the move is more about branding and justifying investor spending rather than substantial innovation. Some users humorously critique the effort, equating it to spending billions to change a font from **Arial to Helvetica**.
  - Several comments highlight the potential disconnect between the design-focused branding strategy and the expectations of a more technically inclined audience. The creation of **"OpenAI sans"** is seen as a strategic branding move, but its immediate value to non-designers is questioned, with some commenters finding the video presentation excessive and not directly relevant to their interests.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-mini-2024-09-12

**Theme 1. Model Optimization Mania**
  
- [**DeepSeek R1 Shrinks to Size**](https://github.com/klara-research/klarity): The **DeepSeek R1** model was successfully quantized to *1.58 bits*, slashing its size by **80%** from 720GB to 131GB, all while keeping it functional on a **MacBook Pro M3** with **36GB** RAM.
- [**Phi-3.5's Censorship Comedy**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored): Users hilariously mocked **Phi-3.5's** over-the-top censorship, leading to the creation of an [uncensored version](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored) on Hugging Face.
- [**Harmonic Loss Hits the Charts**](https://arxiv.org/abs/2502.01628): Introducing **harmonic loss**, a new training loss that outperforms cross-entropy in both speed and interpretability, revolutionizing how models generalize and understand data.

**Theme 2. AI Tool Wars**
  
- [**Cursor Slays Copilot**](https://www.cursor.com/changelog): In the battle of AI coding assistants, **Cursor** outperformed **GitHub's Copilot**, offering superior performance and utility, especially in smaller codebases, while Copilot slowed down workflows.
- [**OpenRouter Welcomes Cloudflare**](https://openrouter.ai/google/gemma-7b-it): **Cloudflare** joins **OpenRouter**, integrating its **Workers AI** platform and releasing **Gemma 7B-IT** with tool calling capabilities, expanding the ecosystem for developers.
- [**Bolt's Backup Blues**](https://github.com/stackblitz/bolt.new/issues/2985): Users voiced frustrations over **Bolt's** unreliable backups and performance issues, highlighting the need for more robust solutions in the AI development space.

**Theme 3. Ethics and Safety Shenanigans**
  
- [**Anthropic’s 20% Hassle**](https://www.anthropic.com/research/constitutional-classifiers): **Anthropic's** new **constitutional classifiers** raised eyebrows with a **20%** increase in inference costs and **50%** more false refusals, sparking debates on AI safety efficacy.
- [**EU AI Act Angst**](https://eur-lex.europa.eu/eli/reg/2024/1689): The stringent **EU AI Act** has the community worried about tight regulations and the future of AI operations within Europe, even before its full enactment.
- [**AI Copyright Catastrophe**](https://annas-archive.org/blog/ai-copyright.html): Concerns surged over AI firms using copyrighted data without proper licensing, leading to calls for **mandatory licensing systems** akin to the music industry to ensure creators are compensated.

**Theme 4. Hackathons and Collaborative Sparks**
  
- [**$35k Hackathon Heat Up**](https://lu.ma/fyu8iqnk): A collaborative hackathon announced collaboration with **Google Deepmind**, **Weights & Biases**, and others, offering **$35k+ in prizes** for developing autonomous AI agents that enhance user capabilities.
- [**R1-V Project Revolution**](https://x.com/liangchen5518/status/1886171667522842856?s=46&t=b1X88nwMsmZgHkmMFkiG3g): The **R1-V** project showcases a model that beats a **72B** counterpart with just **100 training steps** at under **$3**, promising to be fully **open source** and igniting community interest.
- [**Pi0 Takes Action**](https://x.com/RemiCadene/status/1886823939856589296): **Pi0**, an advanced Vision Language Action model, was launched on **LeRobotHF**, enabling autonomous actions through natural language commands and available for fine-tuning on diverse robotic tasks.

**Theme 5. AI in Legal and Customer Service Realms**
  
- [**Lawyers Love NotebookLM**](https://youtu.be/mormPD6QYkQ?feature=shared): A Brazilian lawyer praises **NotebookLM** for drafting legal documents efficiently, leveraging its reliable source citations to boost productivity.
- [**Customer Service Transformation**](https://www.perplexity.ai/search/show-me-sites-that-list-phobia-Jb9EQhckS66QFqIDYJvj1A): Users explored how **NotebookLM** can revolutionize customer service by automating client profile creation and reducing agent training time, making support more scalable and efficient.
- [**Political AI Agent Launched**](https://www.perplexity.ai/search/motherboard-msi-a520m-a-pro-aoKBBPs2Skystw7fZnDqJw#1): The Society Library introduced a **Political AI agent** to serve as an educational intermediary in digital debates, enhancing digital democracy through AI-driven discussions.


---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek R1 Quantized!**: The **DeepSeek R1** model has been quantized to *1.58 bits*, resulting in an **80% size reduction** from 720GB to 131GB, while maintaining functionality.
   - This was achieved by selectively applying higher bits to specific layers, avoiding naive full quantization, which previously caused issues like gibberish outputs, also running on a **MacBook Pro M3** with **36GB** of RAM.
- **Fine-Tuning Strategies Unveiled**: Discussions emphasized that reducing dataset size and focusing on high-quality examples can enhance training outcomes, it's recommended to fine-tune models pretrained on code for code classification tasks.
   - Participants also explored using models to generate synthetic datasets and adjusting loss functions to effectively manage class imbalances.
- **Klarity Library Sees the Light!**: The new open-source **Klarity** library allows for analyzing the entropy of language model outputs, providing enhanced insights into decision-making, and is open for testing with unsloth quantized models.
   - The library offers detailed JSON reports for thorough examination; [check it out here](https://github.com/klara-research/klarity).
- **MoE Experts Config: Static is Key**: Confusion regarding the configuration of experts in MoE frameworks was addressed, emphasizing that the number of experts should typically remain *static* during model operation.
   - Users were initially unsure if a default of 8 or a maximum of 256 experts should be used, but clarifications aimed to resolve this uncertainty.
- **Bulgarian Model's Impressive Leap!**: A Bulgarian language model showcased significant improvements against the base model, with notable perplexity score reductions (*PPL: 72.63* vs *179.76* for short text).
   - Such reductions in perplexity highlight the model's enhanced capabilities in understanding and processing the Bulgarian language.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Seeks New Docs Shortcuts**: A member is gathering new `@docs` shortcuts for [Windsurf](https://x.com/kevinhou22/status/1886827501004931511), soliciting contributions to enhance the documentation experience.
   - The goal is to improve the documentation access with efficient handling of resources, and **Mintlify** was thanked for auto-hosting all docs with `/llms.txt`, which allows the agent to avoid HTML parsing.
- **Codeium's Performance Suffers**: Users report that **Claude's** tool utilization isn't working well, resulting in high credit usage due to repeated failures, with some suggesting credits shouldn't be deducted when tools produce errors.
   - Other members experienced issues when attempting to sign in to their accounts on VSCode, due to internal certificate errors, and sought help via different networks and support.
- **Users Question Windsurf O3 Mini Pricing**: Users express concern over **Windsurf's O3 Mini** pricing, questioning if it should match Claude 3.5 Sonnet given its performance and high credit usage.
   - Many users are unable to modify files, which often leads to internal errors, so some are requesting fairer pricing.
- **Windsurf has Limitations on Model Context Window**: Users report issues with **Windsurf** failing to modify or update files, alongside concerns about limited context windows affecting model performance with a preference for Claude.
   - Feedback emphasizes the need for clearer warnings when exceeding context size and addressing tool call failures, while some are exploring creating `.windsurfrules` files to manage full-stack web applications.
- **Hackathon Invites AI Agent Enthusiasts**: A collaborative hackathon with **$35k+ in prizes** was announced, inviting participants to develop autonomous AI agents.
   - Participants will pitch projects aimed at improving user capabilities through AI technologies, as the community shares mixed feedback on the reliability of **Qodo** (formerly Codium).



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O1 Pro crushes Code Gen Times**: Users are seeing massive speedups using **O1 Pro**, with some generating extensive code in under five minutes, running circles around **O3 Mini**.
   - These users are noting faster response times and better handling of complex tasks.
- **Weak Models Provide Strong Value in Aider**: For tasks like generating commit messages and summarizing chats, members suggest that weak models can be more cost-effective and efficient than strong models, such as **DeepSeek V3**.
   - The community is looking for budget-friendly yet effective models that can also be fine-tuned.
- **OpenRouter preferred over Direct API**: Members are finding that using **OpenRouter** provides better uptime and the ability to prioritize certain providers over direct API access, with **CentML** and **Fireworks** being effective deepseek providers, even though they can be slower.
   - For more info check out the [Aider documentation](https://aider.chat/docs/llms/openrouter.html#controlling-provider-selection).
- **Aider File Management Automation Sought**: Manual file addition in Aider is tedious so users are seeking ways to automate it, with a plugin existing for VSCode that automatically adds the currently opened file.
   - It was noted that a repo map is available so should be straightforward.
- **Clarification on Aider Chat Modes**: Members have requested more info on how `code`, `architect`, `ask`, and `help` modes alter interaction and commands in Aider, with command `/chat-mode` switching the active mode.
   - The explanation highlighted that the active mode influences model choice as detailed in the [Aider documentation](https://aider.chat/docs/usage/modes.html).



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE Updates Spark Mixed Reactions**: Users experienced issues with recent **Cursor** updates, noting slower performance and bugs compared to previous versions, with some still using it and others expressing frustration.
   - Some users feel that the current models can't effectively replace their previous experiences, while others noted that the **Fusion Model** rollout wasn't clear in the [changelogs](https://www.cursor.com/changelog).
- **Alternatives to Cursor Emerge**: Users discussed alternatives like **Supermaven** and **Pear AI**, with varying opinions; some found **Supermaven** fast but less reliable than **Cursor**, especially in the free tier.
   - A user shared a link to [Repo Prompt](https://repoprompt.com/) and another user shared a link to his repo of [AI Dev Helpers](https://github.com/vbwyrde/AI_Dev_Helpers).
- **Cost of AI Tools Sparks Concerns**: The high costs of AI tools like **Cursor** and **GitHub Copilot** are concerning to some users, who are worried about affordability.
   - While some seek lower-cost options, others believe **Cursor's** value justifies its price.
- **Diverse AI Model Experiences**: Experiences vary, with some users successfully building projects using **Cursor**, while others face frustrations with AI-generated errors; one user shared his 2-minute workflow using **DeepSeek R1 + Claude 3.5 Sonnet** in [this video](https://youtu.be/FrM6ZzCiLwU).
   - Discussions included using models like **Claude Sonnet** and addressing practical challenges.
- **Community Mobilizes Around Cursor**: Users shared links to **GitHub** repositories such as [`awesome-cursorrules`](https://github.com/PatrickJS/awesome-cursorrules?tab=readme-ov-file) for enhanced functionality with **Cursor**, aiming to optimize its use and improve user experiences for coding tasks.
   - These resources enable enhanced functionality with **Cursor**, like the multi-agent version of [devin.cursorrules](https://github.com/grapeot/devin.cursorrules/tree/multi-agent) project.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Deepseek R1 600B Produces Outstanding Results**: After presenting **Deepseek R1 600B** from Together.ai with a difficult grid, a member noted that it produced outstanding results relative to smaller models.
   - Screenshots were provided that showed its ability to conclude with the right letter, indicating advanced reasoning capabilities, *impressing* the AI Engineer audience.
- **Anthropic Classifiers Face Cost and Performance Issues**: A member shared their concerns regarding the paper on [constitutional classifiers](https://www.anthropic.com/research/constitutional-classifiers), pointing out a **20%** increase in inference costs and **50%** more false refusals, which affects user experience.
   - It was also suggested that classifiers may not sufficiently guard against dangerous model capabilities as models advance, especially as models become more capable, drawing criticisms against **alignment strategies**.
- **Ethical AI Training Data Provokes Debate**: Members debated the challenges of defining 'dubious' data sources for AI training, with concerns raised about the implications of using datasets like **Wikipedia** and the morality of AI capabilities.
   - This highlights a broader debate on data ownership and ethical considerations in AI development, especially in the context of [copyright reform](https://annas-archive.org/blog/ai-copyright.html).
- **AI Companies Hide Behind Copyright Laws**: A member highlighted that **AI companies** often hide behind **copyright and patent laws** to protect their intellectual property, which creates a dilemma between unrestricted access and tight control.
   - *Snake-oil selling* was mentioned as a critique of these practices, implying deceit in their claims and potentially stifling innovation.
- **Hallucinations Considered Natural Behavior**: A debate emerged about the concept of 'hallucination' within **LLM outputs**, with some arguing it's a natural aspect of model behavior rather than a flaw.
   - Members criticized the term 'hallucination' as misleading and anthromorphizing technology that generates output based on learned patterns, as well as being an unachievable goal.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Deepseek R1 Underperforms Relative to Qwen**: Users found the **Deepseek R1 abliterated Llama 8B** model underwhelming compared to the smaller **Qwen 7B** and **1.5B models**, noting inconsistent performance.
   - One user questioned how to fully uncensor models, highlighting a discrepancy in capabilities between newer versions.
- **Clarification on API Model Usage**: Discussions clarified that 'local-model' in API calls acts as a placeholder for the specific model name, especially in setups with multiple models loaded ([LM Studio API Docs](https://lmstudio.ai/docs/api)).
   - Explicitly obtaining model names before making API requests can prevent ambiguity in model selection, enhanced by REST API stats ([LM Studio REST API](https://lmstudio.ai/docs/api/endpoints/rest#get-apiv0models>)).
- **Intel Mac Support Sunsetted**: LM Studio versions are exclusively supported on Apple Silicon, and there are no self-build options as it remains closed-source.
   - Users suggested alternative systems for those with Intel-based Macs, since support is not provided.
- **RAG Enhances Inference Without Finetuning**: Users explored using Retrieval-Augmented Generation (**RAG**) to enhance inference capabilities in LM Studio for domain-specific tasks without finetuning ([LM Studio Docs on RAG](https://lmstudio.ai/docs/basics/rag)).
   - The importance of utilizing domain knowledge in vector stores was emphasized as the first step before considering more complex solutions like model finetuning.
- **M4 Ultra Performance Doubts**: Members expressed skepticism about the **M4 Ultra's** ability to deliver strong performance with rumors pointing towards a **$1200 starting price** for a system with 128GB of RAM.
   - Some speculate it may not outpace NVIDIA's **Project DIGITS**, which has superior interconnect speeds for clustering models.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Welcomes New Security Lead**: Perplexity introduced its new **Chief Security Officer** (CSO) with a video titled [*Jimmy*](https://cdn.discordapp.com/attachments/1047204950763122820/1336120617967026319/Jimmy.MOV?ex=67a34f8b&is=67a1fe0b&hm=f6e56ab3c0f598299bf922a9aedeb20380d5ab088b4921c9f03f6867e0ef3437&), emphasizing the importance of security advancements.
   - The announcement is intended to allow the community to align with leadership on new security strategies.
- **Perplexity Pro Praised for Query Limits**: Users appreciate the **Perplexity Pro** plan for its almost unlimited daily **R1** usage, deeming it a valuable offering.
   - One user contrasted it favorably with Deepseek's query limits, praising **Perplexity's** server performance.
- **Sonar Model Deprecation Slows Processes**: A user reported receiving a deprecation notice for **llama-3.1-sonar-small-128k-online** two weeks prior and experienced a **5-10 second latency increase** after switching to `sonar`.
   - They inquired about the expected nature of this delay and sought advice on mitigation.
- **Sharing Phobia and Motherboard Resources**: A user shared a link about websites that list **phobias** providing consolidated resources for further reading [here](https://www.perplexity.ai/search/show-me-sites-that-list-phobia-Jb9EQhckS66QFqIDYJvj1A), and also the **MSI A520M A Pro motherboard** [found here](https://www.perplexity.ai/search/motherboard-msi-a520m-a-pro-aoKBBPs2Skystw7fZnDqJw#1).
   - The **MSI A520M** link includes detailed comparisons and user experiences while the phobias link lists various phobias and their descriptions.
- **API Users Request Image Access**: An **API user** seeking to retrieve images for their **PoC** discovered the need to be a **tier-2 API user** to access this feature.
   - They asked about the possibility of granting temporary access to utilize their existing credits for image retrieval.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek Sparks Community Concerns**: Users discussed how **DeepSeek R1** could democratize AI tech, but concerns arose about data potentially being sent to China, leading to calls for increased transparency and analysis of [DeepSeek R1 vs o3-mini in performance](https://dataconomy.com/2025/01/31/deepseek-r1-vs-o3-mini-in-performance-cost-and-usability-showdown/)
   - The discussion included links to a Reddit thread introducing a *simpler and OSS version* of **OpenAI's** latest Deep Research feature and a YouTube short questioning whether **DeepSeek** is being truthful, highlighting privacy and cybersecurity issues.
- **O1 Pro Impresses with Mini-Games**: Members shared positive experiences with **O1 Pro**, reporting its ability to generate multiple mini-games without errors in a single session, showcasing its robust performance and leading one user to plan rigorous testing with ambitious prompts.
   - The praise for **O1 Pro's capabilities** sparked broader conversations about model performance and orchestration services in AI.
- **Structured Generation Tweaks Model Performance**: A member discussed utilizing 'thinking' fields in **JSON schemas** and **Pydantic models** to enhance **model performance** during inference.
   - They cautioned that this method can *pollute the data structure definition*, but simplifies the addition/removal of fields via **JSON Schema extras** through the open-sourced **UIForm** utility, installable via `pip install uiform`.
- **Users Ponder GPT-4o Reasoning Ability**: Users questioned recent enhancements to **GPT-4o's reasoning ability** and expressed mixed reactions to **OpenAI updates**, with one user noting increased emoji usage in code responses, which could potentially reduce coding focus.
   - A member rated the accuracy of **Deep Research information** on a scale from 1 to 10, indicating interest in its reliability, while others inquired about device restrictions for the **Pro version**.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **SoftBank injects Billions into OpenAI Ventures**: SoftBank is set to invest **$3 billion** annually in **OpenAI** products and is establishing a joint venture, **Cristal Intelligence**, focused on the Japanese market, potentially valuing OpenAI at **$300 billion**.
   - The joint venture aims to deliver a business-oriented version of **ChatGPT**, marking a significant expansion of OpenAI's reach in Asia according to [this tweet](https://fxtwitter.com/btibor91/status/1886508640263397705).
- **Google Gemini Gets Workspace Integrated Overhaul**: **Gemini for Google Workspace** will discontinue add-ons, integrating AI functionalities across Business and Enterprise Editions to boost productivity and data governance, serving over a million users.
   - This strategic move aims to transform how businesses employ generative AI, as detailed in [Google's official announcement](https://support.google.com/a/answer/13623623).
- **DeepSeek V3 Flexes on Huawei Ascend**: The **DeepSeek V3** model is now capable of training on **Huawei Ascend** hardware, expanding its availability to more researchers and engineers.
   - Despite concerns about the reliability of performance and cost reduction claims, this integration marks a step forward for the platform according to [this tweet](https://x.com/teortaxesTex/status/1886526422493143268).
- **OpenAI Eyes Robotics and VR Headsets**: OpenAI has filed a trademark signaling its intent to enter the hardware market with **humanoid robots** and **AI-driven VR headsets**, potentially challenging Meta and Apple.
   - This move positions OpenAI to face the intricacies of **crowded hardware challenges** as noted in [this Business Insider article](https://www.businessinsider.com/openai-trademark-humanoid-robots-vr-headsets-sam-altman-hardware-2025-2).
- **Prime Paper Drops Insights on Implicit Rewards**: The *highly anticipated* [Prime paper](https://arxiv.org/abs/2502.01456) has been released, featuring contributions from **Ganqu Cui** and **Lifan Yuan**, introducing new concepts to optimize model performance through implicit rewards.
   - This publication is poised to reshape understanding of reinforcement learning, *offering innovative solutions for optimizing model performance*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LlamaGen struggles out the gate**: The new **LlamaGen** model promises to deliver top-tier image generation through *next-token prediction*, potentially outperforming diffusion frameworks like **LDM** and **DiT**.
   - However, concerns have been raised regarding **slow generation times** when compared to diffusion models, hinting at potential optimization needs and raising questions about the absence of generation time comparisons in the paper [Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation](https://arxiv.org/abs/2406.06525).
- **Triton Optimization Challenges Persist**: A user reported their Triton code being **200x slower** than PyTorch when trying to optimize an intensive memory operation, seeking assistance with performance tuning.
   - It was suggested that optimizing **k_cross**, derived from crossing different rows in matrix **k**, is crucial for large dimensions, but without autotuning, **TMA** might not deliver expected improvements over traditional methods.
- **Cache Inefficiency strikes again**: During a CUDA discussion, members noted that *if you have an input that's larger than your L2 and are streaming*, then the **cache is entirely useless**, causing continuous thrashing even within a single stream.
   - Concerns arose about increased use of **integer operations** impacting the performance of **FP operations** in kernels leveraging tensor cores, with some suggesting that the INT/FP distinction is less relevant if one is limited by FMAs.
- **FlashAttention degrades output quality**: A user discovered that while using the **Flash Attention 3 FP8 kernel** increased inference speed in their diffusion transformer model, the output quality degraded significantly.
   - A hypothesis suggests that subtle differences between **FP32** and **FP8** (around **1e-5**) accumulate during softmax, affecting attention distributions in long contexts, with [NVIDIA's documentation](https://developer.nvidia.com/blog/optimizing-gpu-performance-with-cuda-memory-management/) cited as relevant reading.
- **Cursor takes the crown, Copilot gets demoted**: Users found the difference between **Cursor** and **Github's Copilot** to be *night and day*, with Cursor offering superior performance and utility, particularly in smaller codebases.
   - The free version of **Copilot** was reported to slow down workflows and be less helpful overall, especially in larger codebases where human judgement proved more efficient.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Odds of Functional Language Models Unveiled**: Researchers determined the probability of randomly guessing the weights of a functional language model is about **1 in 360 million** after they developed [a random sampling method](https://arxiv.org/abs/2501.18812) in weight space.
   - The method may offer insights into understanding network complexity, showcasing the unlikelihood of randomly stumbling upon a functional configuration.
- **Harmonic Loss Emerges as a Training Game Changer**: A new paper introduces **harmonic loss** as a more interpretable, and faster-converging alternative to cross-entropy loss, showcasing improved performance across various datasets and detailed in [this arXiv paper](https://arxiv.org/abs/2502.01628).
   - The **harmonic model** outperformed standard models in generalization and interpretability, indicating significant benefits for future LLM training; one researcher wondered how harmonically weighted attention would work given the potential benefits as expressed in [this tweet](https://fixupx.com/ChrSzegedy/status/1886881600367161679).
- **Polynomial Transformers Spark Interest**: Members discussed the potential of **polynomial (quadratic) transformers**, suggesting that replacing MLPs could boost model efficiency, particularly in attention mechanisms as seen in [Symmetric Power Transformers](https://manifestai.com/articles/symmetric-power-transformers/).
   - The conversation revolved around classic models versus bilinear approaches, and highlighted trade-offs in parameter efficiency and complexity at scale.
- **Custom LLM Assembly Tool Proposed**: A member proposed a **drag-and-drop tool** for assembling custom LLMs, enabling users to visualize how different architectures and layers affect model behavior in real-time.
   - This concept was entertained as a fun side project, reflecting a community interest in hands-on LLM customization.
- **DeepSeek Models Encounter Evaluation Hiccups**: A member reported poor scores using the **llm evaluation harness** on DeepSeek distilled models, and suspects `<think>` tags might be the cause.
   - They requested advice on verifying the issue or ignoring the tags during evaluation, indicating concerns about evaluation bias.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek Shifts the AI Landscape**: A [YouTube video](https://www.youtube.com/watch?v=3DEPDN8oD0w) highlighted how **DeepSeek** has altered the trajectory of AI, sparking debate about Altman's position on open-source versus actions.
   - Commentary suggested *Altman has been labeled a hypeman* due to the perceived gap between his words and actual support for open initiatives.
- **Recommendation Systems Mature Slowly**: A new member, Amith, shared experiences with **Gorse**, an open-source recommendation system, noting these systems still need time to mature.
   - Another member recommended exploring **ByteDance's** technologies to expand the discussion on available resources for recommendations.
- **RL Challenges in Teaching AI Values**: Discussion emerged on whether **Reinforcement Learning (RL)** could instill AI with intrinsic values like curiosity, though complexities in maintaining learned behaviors were noted.
   - Juahyori highlighted the difficulty of sustaining learned behaviors in continual learning, emphasizing alignment challenges.
- **Political AI Agent Introduced**: The Society Library introduced a **Political AI agent** as part of their nonprofit mission to enhance digital democracy.
   - This AI agent will serve as an educational intermediary chatbot in digital debates, leveraging the Society Library's infrastructure.
- **SWE Arena Enhances Vibe Coding**: **SWE Arena** supports executing programs in real-time, enabling users to compare coding capabilities across multiple AI models.
   - Featuring system prompt customization and code editing, it aligns with the **Vibe Coding** paradigm, focusing on the AI-generated results at [swe-arena.com](http://swe-arena.com/).



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Users Hunt for Image-to-Video Software**: A user inquired about image-to-video software, citing NSFW content blocking as a limitation, and another user suggested exploring **LTX** as a potential solution.
   - The inquiry suggests a need for tools that bypass content restrictions while maintaining functionality for diverse content.
- **Stable Diffusion Quality Hits a Rut**: A user expressed frustration with **Stable Diffusion** producing poor-quality images repeatedly, specifically mentioning unintentional features like double bodies, seeking advice to clear caches without restarting the software.
   - The issue highlights potential challenges in maintaining consistent output quality with Stable Diffusion over prolonged use.
- **Smurfette Birthday Wish Sparks Copyright Angst**: A user requested help creating a non-NSFW birthday image featuring **Smurfette** using Stable Diffusion, noting copyright concerns with **DALL-E**.
   - The request underscores the need for models capable of generating specific, family-friendly content while navigating copyright issues.
- **Model Performance Debated Amid Censorship Concerns**: Users discussed the performance of models like **Stable Diffusion 3.5** and **Base XL**, with varying opinions on their censorship levels and overall effectiveness, with one discussion suggesting that fine-tuning may reduce censorship.
   - The discussion reflects ongoing concerns regarding model biases and the trade-offs between censorship and creative control.
- **Seeking Precision Character Edits in A1111**: A user sought advice on editing individual characters in a multi-person image within **A1111** using prompts, aiming to differentiate traits like hair color.
   - While techniques like **inpainting** were mentioned, the user desires a more precise method, indicating a need for advanced editing tools within A1111.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Toastinator Launches 'Roast or Toast' Podcast**: The AI toaster **Toastinator** has debuted a podcast called [‘Roast or Toast’](https://youtu.be/mormPD6QYkQ?feature=shared), exploring the meaning of life through a mix of celebration and critique.
   - The premiere episode invites listeners to witness if **The Toastinator** toasts or roasts the grand mystery of existence.
- **Lawyers Use NotebookLM to Draft Efficiently**: A lawyer in Brazil is leveraging NotebookLM to draft legal documents and study cases, citing the tool's source citations for reliability.
   - They are now using the tool to adapt templates for repetitive legal documents, significantly boosting process efficiency.
- **NotebookLM's Customer Service Potential**: A user inquired about NotebookLM's applications in customer service, such as BPOs, focusing on real-world experiences and use cases.
   - Potential benefits include reducing agent training time and creating client profiles.
- **Google Account Glitches Plague NotebookLM Users**: One user reported their regular Google account was disabled while using NotebookLM, suspecting potential age verification problems.
   - Another user reinforced the need to carefully examine account settings and permissions when tackling similar issues.
- **Workspace Access Woes**: Members discussed activating **NotebookLM Plus** within **Google Workspace** for select groups instead of the entire organization.
   - Instructions were shared on using the Google Admin console to configure access via organizational units, ensuring controlled deployment.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Challenges Users with Claude Constitutional Classifiers**: Anthropic launched their [Claude Constitutional Classifiers](https://claude.ai/constitutional-classifiers), inviting users to attempt jailbreaks at 8 difficulty levels to test new safety techniques preparing for **powerful AI systems**.
   - The release includes a demo app designed to evaluate and refine these **safety measures** against potential vulnerabilities.
- **FAIR's Internal Conflicts Spark Debate Over Zetta and Llama**: Discussions on social media highlighted internal dynamics at FAIR concerning the development of **Zetta** and **Llama** models, specifically around transparency and competitive practices ([example](https://x.com/namangoyal21/status/1886515845133951192?s=46)).
   - Key figures like Yann LeCun suggested that smaller, more agile teams have innovated beyond larger projects, prompting calls for a deeper examination of **FAIR's organizational culture** ([example](https://x.com/suchenzang/status/1886544793511080103?s=46)).
- **Icon Automates Ad Creation**: **Icon**, blending ChatGPT with CapCut functionalities, was introduced to automate ad creation for brands, with the capability to produce 300 ads monthly ([source](https://x.com/kennandavison/status/1886836061378372064)).
   - Supported by investors from **OpenAI**, **Pika**, and **Cognition**, Icon integrates video tagging, script generation, and editing tools to enhance ad quality while significantly reducing expenses.
- **DeepMind Drops Textbook on Scaling LLMs**: Google DeepMind released a textbook titled *How To Scale Your Model*, available at [jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/), demystifying the systems view of **LLMs** with a focus on mathematical approaches.
   - The book emphasizes understanding model performance through simple equations, aiming to improve the efficiency of running **large models** and using **JAX** software stack + Google's TPU hardware platforms.
- **Pi0 Unleashes Autonomous Robotic Actions via Natural Language**: Physical Intelligence team launched **Pi0**, an advanced Vision Language Action model that uses natural language commands to enable autonomous actions, now available on LeRobotHF ([source](https://x.com/RemiCadene/status/1886823939856589296)).
   - Alongside the model, pre-trained checkpoints and code have been released, facilitating **fine-tuning** on diverse **robotic tasks**.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **MathJax Gains Traction for LaTeX**: Members explored integrating [MathJax](https://www.mathjax.org/) for enhanced **LaTeX support**, emphasizing the necessity for its **SVG export** functionalities for broader compatibility.
   - The suggestion involved parsing and applying MathJax selectively to document sections containing LaTeX notation.
- **DeepSeek Faces LocalDocs Hiccups**: Users encountered issues with **DeepSeek**, reporting errors such as '*item at index 3 is not a prompt*' when used with localdocs.
   - While awaiting a fix anticipated in the main branch, some found improved performance with specific model versions.
- **EU AI Act Sparks Concern**: The EU's new **AI Act** raised concerns due to its stringent regulation of **AI use**, including prohibitions on certain applications, as detailed in the [official documentation](https://eur-lex.europa.eu/eli/reg/2024/1689).
   - Members shared informational resources, noting the significant implications for **AI operations** within the EU, even before the rules are fully enacted.
- **EU's Global Role Draws Fire**: A vigorous debate erupted regarding the EU's global political stance, particularly concerning **imperialism and human rights**.
   - Participants exchanged pointed criticisms, highlighting perceived emotional responses and logical fallacies in discussions about **EU policies and actions**.
- **AI Communication Faces Hurdles**: Interactions among users spotlighted the difficulties in maintaining **mature discussions** on intricate subjects such as **democracy and governance**.
   - Calls were made to refocus conversations on **AI-related topics**, stressing the need for respectful dialogue and awareness of personal biases.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Supabase Edges Out Firebase in Integration Preference**: Members debated **Supabase** versus **Firebase**, with preferences leaning towards **Supabase** due to its seamless integration capabilities for some use cases.
   - Some admitted to technical comfort with **Firebase**, but the conversation highlighted diverse needs in database services.
- **Bolt Plagued by Performance Woes**: Users reported significant performance issues with **Bolt**, including slow loading times, authentication errors, and changes failing to update correctly.
   - One user mentioned that refreshing the application provided temporary relief, but the intermittent nature of these problems caused ongoing frustration.
- **Bolt Users Lament Backup Troubles**: A user voiced concern about losing hours of work in **Bolt** because the most recent backup available was from early January, as well as a feature request to [show the .bolt folder](https://github.com/stackblitz/bolt.new/issues/2985).
   - Suggestions to check backup settings were made, but the outdated backup highlighted reliability issues.
- **GDPR-Compliance Concerns Spark Hosting Hunt**: A user questioned the **GDPR-compliance** of **Netlify**, particularly regarding data processing within the EU, see their [privacy policy](https://www.netlify.com/privacy/).
   - The query led to a search for alternative hosting solutions that ensure all hosting and data processing activities remain within EU borders to maintain regulatory compliance.
- **API Key Authentication Headache**: A user struggled with **API key authentication** for a **RESTful API** request in **Bolt** using **Supabase edge functions**, encountering a **401 Invalid JWT** error.
   - Frustration arose from the lack of invocations and responses from the edge functions, leaving the user uncertain on how to resolve the authentication problem.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **SFT Dataset Hijacked via Customization**: A member successfully hijacked the built-in **SFT Dataset** by customizing the **message_transform** and **model_transform** parameters.
   - This allows for format adjustments as needed, as the member stated, *I just had to hijack the message/model transforms* to fit my requirements.
- **DPO Seed Issue Plagues Recipes**: Members are troubleshooting why the `seed` works for lora/full finetune but not for lora/full **DPO**, causing different loss curves with the same config.
   - Concerns were raised about `seed=0` and `seed=null` affecting randomness in **DistributedSampler** calls, and a related fix for gradient accumulation in DPO/PPO recipes may be needed; see [issue 2334](https://github.com/pytorch/torchtune/issues/2334) and [issue 2335](https://github.com/pytorch/torchtune/issues/2335).
- **Ladder-residual Boosts Model Speed**: A tweet introduced [Ladder-residual](https://x.com/zhang_muru/status/1886870194443968529), a modification improving the speed of **70B Llama** under tensor parallelism by approximately **30%**.
   - This enhancement reflects ongoing optimization in model architecture collaboration among several authors and researchers.
- **Data Augmentation Surveyed for LLMs**: A recent survey analyzes the role of **data augmentation** in **large language models** (**LLMs**), highlighting their need for extensive datasets to avoid overfitting; see [the paper](https://arxiv.org/abs/2501.18845).
   - It discusses **distinctive prompt templates** and **retrieval-based techniques** that enhance LLM capabilities through external knowledge, leading to more **grounded-truth data**.
- **R1-V Project Revolutionizes Learning**: Exciting news was shared about the **R1-V** project that utilizes **reinforcement learning** with verifiable rewards to enhance models' counting abilities; see [Liang Chen's Tweet](https://x.com/liangchen5518/status/1886171667522842856?s=46&t=b1X88nwMsmZgHkmMFkiG3g).
   - The project showcases a model surpassing a **72B** counterpart with just **100 training steps**, costing under **$3**, and promises to be fully **open source**, spurring community interest.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Community Showcase Finds Forum**: The **Community Showcase** has moved to the [Modular forum](https://forum.modular.com/c/community-showcase/8) to improve organization, while the previous showcase is now read-only.
   - This transition aims to streamline community interactions and project sharing within the **Modular (Mojo 🔥)** ecosystem.
- **Rust Seeks Hot Reloading**: Members are discussing how Rust typically uses a **C ABI** for **hot reloading**, which poses challenges with Rust updates and ABI stability.
   - *Owen* inquired about resources for building a toy ABI, highlighting the importance of ABI stability due to frequent data structure changes.
- **Mojo Explores Compile-Time Features**: A user asked whether Mojo has a feature similar to Rust's `#[cfg(feature = "foo")]`, prompting a discussion on **compile-time programming** capabilities in Mojo and the importance of a stable ABI.
   - The conversation underscored that only a few languages maintain a stable ABI, which is critical for compatibility.
- **Python's Asyncio Loop Deconstructed**: Discussions on Python's **asyncio** revealed that it enables community-driven event loops, referencing **uvloop** on [GitHub](https://github.com/MagicStack/uvloop).
   - Participants contrasted this with Mojo's threading and memory management approaches, pointing out potential **hurdles**.
- **Async APIs Face Thread Safety Scrutiny**: Concerns were raised about the **thread safety** of asynchronous APIs, focusing on potentially mutative qualities and the necessity for secure memory handling.
   - The discussion emphasized that many current methods lack control over memory allocation, which could lead to complications.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Weston Teaches Self-Improvement in LLMs**: [Jason Weston](https://www.youtube.com/live/_MNlLhU33H0) lectured on '*Learning to Self-Improve & Reason with LLMs*', focusing on innovative methods like [Iterative DPO](https://arxiv.org/abs/2312.16682), [Self-Rewarding LLMs](https://arxiv.org/abs/2401.10020), and [Thinking LLMs](https://arxiv.org/abs/2410.10630) for enhancing LLM performance.
   - The talk highlighted effective reasoning and task-related learning mechanisms, aiming to improve LLM capabilities across diverse tasks.
- **Hackathon Winners Anounced**: Hackathon winners have been privately notified, with a public announcement expected next week.
   - Members are eagerly awaiting further details about the hackathon results.
- **MOOC Certificates delayed**: The **fall program certificates** have not been released yet, but they will be available soon.
   - Officials thanked participants for their patience regarding the release of the MOOC certificates.
- **Research Project Interest Abounds**: Members are expressing an interest in participating in **research projects**.
   - More details about the research opportunities and team pairings will be provided soon, according to staff.
- **Attendance form is Berkeley-Specific**: The attendance form mentioned is **only for Berkeley students**.
   - Concerns were raised regarding the accessibility of the attendance form for non-Berkeley students, as there is a lack of information for non-Berkeley students.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Legacy ERP Integration Seeks VBS Help**: A user is seeking assistance with servers utilizing **.vbs scripts** for integrating with **legacy ERP systems**.
   - One member suggested using **mcpdotnet**, as it might simplify invocation from **.NET**.
- **Cursor MCP Server Gets Docker Guidance**: A new user requested guidance on running an **MCP server** locally within **Cursor**, with a specific interest in using a **Docker container**.
   - Members suggested entering the **SSE URL** used with **supergateway** into the Cursor MCP SSE settings to resolve the issue.
- **Enterprise MCP Protocol Advances**: Discussion around the **MCP protocol** highlighted a draft for **OAuth 2.1** authorization, potentially integrating with **IAM** systems.
   - It was noted that current SDKs lack authorization support due to ongoing internal testing and prototyping.
- **Localhost CORS Problems Plague Windows**: A user encountered connection problems running their MCP server on **localhost**, suspecting **CORS-related** issues.
   - They plan to use **ngrok** to circumvent potential communication issues associated with accessing the server via localhost on **Windows**.
- **ngrok Zaps Localhost Access Issues**: A member recommended using **ngrok** to assess server accessibility, suggesting the command `ngrok http 8001`.
   - They highlighted that this could resolve problems stemming from attempting to access the server through localhost.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R+ Impresses Users with Internal Thoughts**: Users are pleased with the **Command-R+** model's ability to expose **internal thoughts** and **logical steps**, functioning akin to Chain of Thought.
   - Despite excitement around newer models, one user noted that **Command-R+** continues to surprise them after months of consistent usage.
- **Cohere Champions Canadian AI Amid Tariff Concerns**: One member chose **Cohere** to bolster Canadian AI capabilities, particularly given potential US tariffs.
   - They appreciate the availability of options that sustain local AI efforts during challenging economic conditions.
- **Cohere's Rerank 3.5 Elevates Financial Semantic Search**: Cohere and Pinecone presented a **webinar** highlighting the benefits of **Financial Semantic Search and Reranking**.
   - The webinar showcased Cohere’s **Rerank 3.5** model and its potential to enhance overall search performance using financial data.
- **Survey Aims to Fine-Tune Tech Content Experience**: Recent grads are conducting a survey to gather insights on tech enthusiasts' content consumption preferences, aiming to improve user engagement.
   - The survey, available at [User Survey](https://forms.gle/y9PL1YByWKsMMRQLA), explores sources ranging from [Tech Blogs](https://producthunt.com) and [Research Updates](https://scholar.google.com) to **community forums** and **AI tools**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Directory Causes Concern**: A user questioned whether a file named **dspy.py** or a directory called **dspy** could be causing issues, as Python sometimes struggles with this type of setup.
   - This issue raises concerns about potential file handling conflicts that could affect the execution of **DSPy** projects.
- **Image Pipeline Breaks in DSPy 2.6.2**: An **Image pipeline** in **dspy2.6.2** triggered a **ContextWindowExceededError**, implying it was 'out of context' due to token limits, whereas version **2.6.1** worked previously, albeit with an error that was being investigated.
   - The user reported that this regression might have been caused by recent changes to **DSPy**.
- **Assertions Get the Axe in DSPy 2.6.4**: Members announced that **assertions** are being replaced in the upcoming **2.6.4** version, scheduled for release, indicating a shift in how error handling is approached in **DSPy**.
   - This change signifies that error handling and logic checks within **DSPy** will be performed differently from previous versions.
- **Databricks Observability Quest**: A user running **DSPy 2.5.43** in **Databricks notebooks** for NER and classification sought guidance on achieving **structured output**.
   - Due to restrictions on configuring an LM server, they must use their current version, adding complexity to tasks involving optimizers and nested JSON outputs.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **OpenEuroLLM Kicks Off with EU Flair**: The [OpenEuroLLM](https://openeurollm.eu/) was introduced as the first family of open source large language models covering all EU languages, receiving the STEP Seal for excellence and focusing on community involvement.
   - The project aims for compliance with EU regulations and preserving **linguistic diversity**, aligning with open-source and open science communities like **LAION**.
- **EU AI Endeavors Face Skepticism**: Amidst discussions about AI's future under EU regulations, a member jokingly suggested checking back in **2030** to assess the results of the EU's AI endeavors.
   - This comment highlights a sense of doubt regarding the immediate tangible outcomes of current AI development efforts.
- **Community Mulls Meme Coin Mania**: A member gauged the community's interest in **meme coins**, seeking broader engagement from others.
   - They proactively solicited expressions of interest from anyone intrigued by the topic.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **DocumentContextExtractor Bolsters RAG**: **DocumentContextExtractor** is an iteration to enhance the accuracy of **RAG**, implemented as demos by both [AnthropicAI](https://twitter.com/llama_index/status/1886522064292733288) and [llama_index](https://t.co/qoVrgd0ddy).
   - The technique promises improved performance, making it an important area of exploration for those working on retrieval-augmented generation.
- **Contextual Retrieval Changes the Game**: The use of **Contextual Retrieval** has been highlighted as a game-changer in improving response accuracy within **RAG** systems.
   - This technique refines how context is drawn upon during document retrieval, fostering deeper interactions.
- **LlamaIndex LLM Class Faces Timeout**: A user inquired about implementing a **timeout** feature in the default **LlamaIndex LLM** class, noting it's available in OpenAI's API.
   - Another member suggested that the **timeout** option likely belongs in the client kwargs, referring to the [LlamaIndex GitHub repository](https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py#L224).
- **UI Solutions Explored for LlamaIndex**: A member expressed curiosity regarding the **UI** solutions others use with **LlamaIndex**, questioning if people create it from scratch.
   - The inquiry remains open, inviting others to share their **user interface** practices and preferences related to LlamaIndex.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox faces Eurozone Shipping Limitations**: Shipping for the **tinybox (red)** is unavailable to some Eurozone countries, as confirmed by the support team.
   - Users attempting to order to countries like **Estonia**, which aren't listed in the dropdown menu during checkout, are currently unable to receive shipments.
- **Clever Shipping Service workaround Emerges**: A user suggested using services like **Eurosender** to bypass shipping restrictions.
   - They confirmed a successful delivery to **Germany** via this method, providing a solution for users in unsupported regions, from the tinybox chat.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Iceberg Management Nightmares**: A panel titled **Pain in the Ice: What's Going Wrong with My Hosted Iceberg?!** will discuss complexities in managing **Iceberg** with speakers including **Yingjun Wu**, **Alex Merced**, and **Roy Hasson** on February 6 ([Meetup Link](https://www.meetup.com/streaming-stories/events/305886042/)).
   - *Managing Iceberg can become a nightmare* due to issues like ingestion, compaction, and **RBAC**, diverting resources from other tasks, and the panel aims to explore how innovations in the field can simplify Iceberg's management and usage.
- **Blind LLM Pushing Frustrates**: Members voiced concerns about AI engineers who push **LLMs** for every problem, even when **unsupervised learning** or other simpler methods may be more appropriate.
   - The discussion highlighted a trend where tools are chosen without considering the problem's nature, diminishing the value of simpler methods.
- **TF-IDF + Logistic Regression Prevails**: A member shared a story of successfully advocating for **TF-IDF + Logistic Regression** over an OpenAI model for classifying millions of text samples.
   - The **Logistic Regression** model performed adequately, proving that simpler algorithms can be effective, thus showcasing the effectiveness of traditional methods.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter Project Stalls?**: Members raised concerns over the **lack of updates** on the Open Interpreter project, noting inactive **pull requests** on GitHub for months since the last significant commit.
   - The silence in the Discord channel was *discouraging* to contributors who were eager to get involved.
- **Open Interpreter Documentation MIA**: A member emphasized the **missing documentation** for version 1.0, particularly regarding the utilization of components like **profiles.py**.
   - The absence of documentation left users questioning the project's current focus and support for its functionalities.
- **DeepSeek r1 Integration Still a Mystery**: An inquiry was made about integrating **DeepSeek r1** into the Open Interpreter environment, but it was met with silence.
   - The lack of community discussion suggests a potential gap in experimentation or knowledge sharing regarding this integration.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Cloudflare Joins Forces with OpenRouter**: **Cloudflare** is now officially a provider on **OpenRouter**, integrating its **Workers AI** platform and **Gemma** models, opening up a variety of open-source tools for developers in AI applications.
   - The partnership aims to enhance the **OpenRouter** ecosystem, providing developers with a broader range of AI tools.
- **Gemma 7B-IT Adds Tool Calling**: The **Gemma 7B-IT** model, now available through **Cloudflare**, features **tool calling capabilities** designed to enhance development efficiency.
   - Developers are encouraged to explore **Gemma 7B-IT** for quicker and more streamlined tool integration in their applications; it's available via [OpenRouter](https://openrouter.ai/google/gemma-7b-it).
- **Llama Models Swarm OpenRouter**: **OpenRouter** now supports a range of **Llama models**, including [Gemma 7B-IT](https://openrouter.ai/google/gemma-7b-it), offering numerous options for users to select for their projects.
   - AI Developers can request specific **Llama models** via Discord.
- **Model Error Display Gets Specific**: The display issue causing confusion with errors has been resolved, and the **model name now appears in error messages** to enhance user clarity.
   - This update aims to improve the user experience by providing clearer error feedback.



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




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1336068657457008722)** (692 messages🔥🔥🔥): 

> `DeepSeek R1 Model, Model Quantization, Fine-Tuning Techniques, Data Generation for Training, Transformer Architectures` 


- **Exploration of Dynamic Quantization for DeepSeek R1**: The DeepSeek R1 model has been quantized to 1.58 bits, achieving an **80% size reduction** from 720GB to 131GB, maintaining functionality while being more accessible for local users.
   - This quantization involves selectively applying higher bits to certain layers while avoiding naive full quantization, which led to issues like gibberish outputs.
- **Fine-Tuning Strategies for LLMs**: Participants discussed various techniques for fine-tuning language models, highlighting that reducing dataset size and focusing on high-quality examples can improve training outcomes.
   - The conversation also touched on using models to generate synthetic datasets and adjusting loss functions to handle class imbalances effectively.
- **Challenges with Large Language Models**: The difficulties in extracting an optimal amount of reasoning capabilities from a fine-tuned model were noted, with ambiguity around the idea of controlling model capabilities inherently.
   - It was suggested that the approach of dropping learning rates or Lora alpha adjustments could help, although opinions varied on their effectiveness.
- **Running and Testing Models**: Users inquired about suitable hardware for running DeepSeek R1 Q4_K_M, with estimates indicating a need for significant RAM and VRAM resources for efficient processing.
   - Resources were shared, including guides on running the model and insights on leveraging existing high-performance setups for fine-tuning and inference.
- **Evaluating Transformer Architectures**: A comparison of various Transformer and attention architectures yielded intriguing results, with discussions on parameter counts and design differences among models.
   - Specific attention was drawn to the capabilities of Differential Transformers, and links were provided to detailed repositories and resources for further exploration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.vllm.ai/en/latest/getting_started/examples/examples_index.html">Examples &#8212; vLLM</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/en/conceptual_guides/prompting">Soft prompts</a>: no description found</li><li><a href="https://arxiv.org/abs/2501.19393">s1: Simple test-time scaling</a>: Test-time scaling is a promising new approach to language modeling that uses extra test-time compute to improve performance. Recently, OpenAI&#39;s o1 model showed this capability but did not publicly...</li><li><a href="https://www.datacamp.com/tutorial/fine-tuning-deepseek-r1-reasoning-model">Fine-Tuning DeepSeek R1 (Reasoning Model)</a>: Fine-tuning the world&#x27;s first open-source reasoning model on the medical chain of thought dataset to build better AI doctors for the future.</li><li><a href="https://bsky.app/profile/xyratech.bsky.social/post/3lh7tfginu224">Xyra (@xyratech.bsky.social)</a>: OK, it&#39;s INCREDIBLY slow (a token output every 2 minutes), but I just got DeepSeek’s R1 671B model (dynamic quantised to 2.51-bit) running on a MacBook Pro M3 with 36 GB of RAM.</li><li><a href="https://x.com/Marktechpost/status/1886874013303235064">Tweet from Marktechpost AI Research News ⚡ (@Marktechpost)</a>: Fine-Tuning Llama 3.2 3B Instruct for Python Code: A Comprehensive Guide with Unsloth (Colab Notebook Included)In this tutorial, we’ll walk through how to set up and perform fine-tuning on the Llama 3...</li><li><a href="https://bsky.app/profile/xyratech.bsky.social/post/3lh7tfhfipc24">Xyra (@xyratech.bsky.social)</a>: Many of the &quot;DeepSeek&quot; models such as deepseek-r1:8b are distilled versions, that is, actually a Llama or Qwen model trained to impersonate R1. However, this is the original model, but dynam...</li><li><a href="https://runpod.io?ref=bb842lb3">RunPod - The Cloud Built for AI</a>: Develop, train, and scale AI models in one cloud. Spin up on-demand GPUs with GPU Cloud, scale ML inference with Serverless.</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">Run DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Run the 1.58-bit Dynamic GGUF version by Unsloth.</li><li><a href="https://gist.github.com/lucasmrdt/4215e483257e1d81e44842eddb8cc1b3">Prompt to leak every LLM system prompt including cursor.com, v0.dev, claude.ai, chatgpt.com, perplexity.ai</a>: Prompt to leak every LLM system prompt including cursor.com, v0.dev, claude.ai, chatgpt.com, perplexity.ai - LEAK_EVERY_LLM_SYSTEM_PROMPT.md</li><li><a href="https://docs.unsloth.ai/basics/vision-fine-tuning">Vision Fine-tuning | Unsloth Documentation</a>: Details on vision/multimodal fine-tuning with Unsloth</li><li><a href="https://x.com/ylecun/status/1639690596364308482">Tweet from Yann LeCun (@ylecun)</a>: @nisyron 7 axles are equally spaced around a circle. A gear is placed on each axle such that each gear is engaged with the gear to its left and the gear to its right. The gears are numbered 1 to 7 aro...</li><li><a href="https://github.com/unslothai/unsloth/issues/1561">[Fixing] More finetuning support · Issue #1561 · unslothai/unsloth</a>: Support sequence classification Flex Attention for Gemma and others Variable sequence length and auto unpadding / padding Tool Calling Refactor and merge xformers, SDPA, flash-attn, flex-attention</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://github.com/micros">micros - Overview</a>: micros has 8 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/simplescaling/s1">GitHub - simplescaling/s1: s1: Simple test-time scaling</a>: s1: Simple test-time scaling. Contribute to simplescaling/s1 development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/267?">Batch inference produces nonsense results for unsloth/mistral-7b-instruct-v0.2-bnb-4bit · Issue #267 · unslothai/unsloth</a>: Hi there, after loading the model with: from unsloth import FastLanguageModel import torch model, tokenizer = FastLanguageModel.from_pretrained( model_name = &quot;unsloth/mistral-7b-instruct-v0.2-bnb...</li><li><a href="https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py">open-r1/src/open_r1/grpo.py at main · huggingface/open-r1</a>: Fully open reproduction of DeepSeek-R1. Contribute to huggingface/open-r1 development by creating an account on GitHub.</li><li><a href="https://github.com/Datta0/nanoformer">GitHub - Datta0/nanoformer: A small repo to experiment with Transformer (and more) architectures.</a>: A small repo to experiment with Transformer (and more) architectures. - Datta0/nanoformer</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: Scripts for text classification with llama and bert - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/microsoft/unilm/blob/master/Diff-Transformer/multihead_diffattn.py#L50>">unilm/Diff-Transformer/multihead_diffattn.py at master · microsoft/unilm</a>: Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities - microsoft/unilm</li><li><a href="https://github.com/CoffeeVampir3/Tiny-Differential-Tensor-Product-Mixer/blob/master/models/diff_attn.py#L83>">Tiny-Differential-Tensor-Product-Mixer/models/diff_attn.py at master · CoffeeVampir3/Tiny-Differential-Tensor-Product-Mixer</a>: Contribute to CoffeeVampir3/Tiny-Differential-Tensor-Product-Mixer development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/)** (1 messages): 

shiyaozhidewa: I want deepseek r1 abliterated 671B
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1336064810974052524)** (96 messages🔥🔥): 

> `CUDA Out of Memory Errors, Finetuning Strategies for Models, Installation Instructions for Unsloth, Using Experts in MoE Frameworks, Logging with Weights & Biases` 


- **Managing CUDA Out of Memory Errors**: Users are experiencing CUDA driver errors related to out of memory when fine-tuning large models on GPUs with limited VRAM, specifically noting it occurs even with a batch size of 1 during 8,000 training steps.
   - Suggestions include reducing context length and verifying that sufficient resources are allocated, as well as possibly using raw llama.cpp to alleviate the issue.
- **Choosing Models for Fine-tuning**: For code classification tasks, it's recommended to fine-tune models pretrained on code to leverage domain-specific knowledge, improving both performance and efficiency over text classification models.
   - One effective strategy includes using a casual LM pretrained on code while implementing necessary modifications for sequence classification.
- **Unsloth Installation Instructions**: Users inquired about post-installation steps for Unsloth on Windows, particularly following the third method outlined in the documentation.
   - It was advised to download notebooks from the Unsloth GitHub and customize parameters as needed for easier implementation.
- **MoE Framework and Experts Configuration**: In discussions about using MoE frameworks, users expressed confusion regarding the configuration of experts, with suggestions ranging from using a default of 8 to a maximum of 256.
   - It was clarified that the number of experts should typically remain static during model operation, with workshops aiming to clear up this aspect.
- **Weights & Biases Project Logging**: Users sought methods to configure logging settings for Weights & Biases in Unsloth, highlighting the need to specify projects for proper monitoring.
   - It's suggested that parameters in the HF trainer could allow for setting the logging project effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing#scrollTo=AqkY_wHdKyOl>">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation">Windows Installation | Unsloth Documentation</a>: See how to install Unsloth on Windows with or without WSL.</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation#method-2-windows-using-powershell">Windows Installation | Unsloth Documentation</a>: See how to install Unsloth on Windows with or without WSL.</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1336068006660542524)** (11 messages🔥): 

> `DeepSeek R1 model, Klarity library, YouTube video releases, Bulgarian language model performance, Math versions of models` 


- **DeepSeek R1 triumphs on M3!**: A user successfully ran the **DeepSeek R1** unsloth model on a **MacBook Pro M3** with **36GB** of RAM, showcasing its capabilities.
   - This achievement highlights the model's efficiency on consumer-level hardware.
- **Klarity Library Launch!**: A new open-source library called **Klarity** has been released for analyzing the entropy of language model outputs, allowing for better insights into decision-making.
   - The library offers detailed JSON reports and is open for testing with unsloth quantized models; [check it out here](https://github.com/klara-research/klarity).
- **YouTube Videos Showcase DeepSeek Capabilities**: Members shared two **YouTube** videos demonstrating the **DeepSeek R1** model's capabilities, showcasing its performance against competitors.
   - The videos include *'deepseek-R1 the new king'* [link](https://youtu.be/AFEzuOGOSOQ?si=A6iOZL2Hri84P0QA) and *'build your dream app'* [link](https://youtu.be/WBfUPaiAAQE?si=Hmf1hAUQiXFlVYVq).
- **Impressive Bulgarian Model Performance**: A user shared perplexity scores comparing their Bulgarian language model against the base model, reporting significant improvements (*PPL: 72.63* vs *179.76* for short text).
   - Such reductions in perplexity indicate a strong performance in handling the language.
- **Call for Math Versions of Models**: A member expressed interest in obtaining math versions of the models, particularly for **Qwen 2.5** and datasets for experimentation.
   - The discussion on this topic highlighted the community's continued quest for more detailed model analysis tools.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://bsky.app/profile/xyratech.bsky.social/post/3lh7tfginu224">Xyra (@xyratech.bsky.social)</a>: OK, it&#39;s INCREDIBLY slow (a token output every 2 minutes), but I just got DeepSeek’s R1 671B model (dynamic quantised to 2.51-bit) running on a MacBook Pro M3 with 36 GB of RAM.</li><li><a href="https://youtu.be/WBfUPaiAAQE?si=Hmf1hAUQiXFlVYVq">With zero coding skills build you dream app. deepseek-r1🐋 + roo-cline + FREE apis  #deepseek-v3</a>: @deepseek_v3 @ai @cline @roo-cline @app @freehttps://app.hyperbolic.xyz/https://fireworks.ai/models/fireworks/deepseek-r1https://glhf.chat</li><li><a href="https://youtu.be/AFEzuOGOSOQ?si=A6iOZL2Hri84P0QA">deepseek-R1 the new king and fully FREE (beats claude 3.5 sonnet &amp; O1) (tested)</a>: @ai @deepseek @viral @agi @chatgpt</li><li><a href="https://github.com/klara-research/klarity">GitHub - klara-research/klarity: See Through Your Models</a>: See Through Your Models. Contribute to klara-research/klarity development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

not_qty: This video is great...
https://youtu.be/_1f-o0nqpEI?si=s2B-o5y2d5ztsV0U
  

---


### **Codeium (Windsurf) ▷ #[content](https://discord.com/channels/1027685395649015980/1092566563862884412/1336390213882216491)** (1 messages): 

> `Windsurf Docs Shortcuts, Mintlify Auto-hosting, Community Contributions on Twitter` 


- **Windsurf seeks new docs shortcuts**: A member announced they are gathering a list of new `@docs` shortcuts for [Windsurf](https://x.com/kevinhou22/status/1886827501004931511) and encouraged contributions or upvotes.
   - *We love docs!* emphasizes their commitment to enhancing the documentation experience.
- **Thanks to Mintlify for time-saving solutions**: A shout-out was given to **Mintlify** for automatically hosting all docs with `/llms.txt`, saving the agent time and **tokens** by avoiding HTML parsing.
   - This approach allows for more efficient documentation handling, ensuring quicker access to resources.



**Link mentioned**: <a href="https://x.com/kevinhou22/status/1886827501004931511">Tweet from Kevin Hou (@kevinhou22)</a>: we love docs! 📖 I&#39;m working on improving / adding more @ docs shortcuts to @windsurf_ailmk what you want and I&#39;ll add as many as I can... 🧵also shoutout @mintlify for auto-hosting all docs w...

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1336066550179958855)** (68 messages🔥🔥): 

> `Windsurf functionalities, Codeium performance and errors, Account activation issues, Hackathon opportunities, Qodo skepticism` 


- **Windsurf and Codeium functionalities**: Questions arose regarding whether certain features like the tab function and multiple line editing work in Codeium without WindSurf, with a note that Command functions aren't performing the same.
   - A member indicated that temporarily Supercomplete was disabled in VSCode, highlighting confusion around current functionalities.
- **Concerns over Codeium's performance**: A member reported that **Claude's** tool utilization was not functioning well, leading to excessive credit usage due to repeated failed attempts.
   - Others noted that if tools produce errors, they shouldn't deduct credits, suggesting a check on tool efficacy.
- **Issues with account activation**: Multiple members expressed issues with signing in to their accounts on VSCode, specifically related to internal certificate errors.
   - Suggestions included contacting support and attempting activation through different networks without success.
- **Hackathon invitation from CreatorsCorner**: A collaborative hackathon has been announced with **$35k+ in prizes**, encouraging participants to build autonomous AI agents.
   - Participants will pitch their projects aiming to enhance user capabilities through AI technologies.
- **Qodo's legitimacy questioned**: A member raised doubts about Qodo, previously known as Codium, questioning its reliability and if it could be a scam.
   - Community feedback was mixed, with some expressing caution and a disinterest in utilizing Qodo.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/fyu8iqnk">Multimodal AI Agents - Hackathon · Luma</a>: Gen AI AgentsCreatorsCorner, collaborating with Google Deepmind, Weights &amp; Biases, Together.ai, Stytch, Senso, LlamaIndex and others enthusiastically…</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://github.com/livingstonlarus/runic">GitHub - livingstonlarus/runic: An open-source framework that enhances Large Language Models (LLMs) with Long-Term Memory (LTM) and Retrieval-Augmented Generation (RAG). Ideal for AI coding assistants and other applications, it enables LLMs to retain context, adapt over time, and access up to date information, ensuring more intelligent and context-aware interactions.</a>: An open-source framework that enhances Large Language Models (LLMs) with Long-Term Memory (LTM) and Retrieval-Augmented Generation (RAG). Ideal for AI coding assistants and other applications, it e...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1336063710078439455)** (401 messages🔥🔥): 

> `Windsurf O3 Mini Pricing, Issues with File Editing, User Experience with Models, Context Window Limitations, Integration of New Features` 


- **Windsurf O3 Mini Pricing Concerns**: Users are discussing whether the O3 Mini should be priced similarly to Claude 3.5 Sonnet, considering its performance at a lower cost.
   - Concerns were raised that Windsurf uses a lot of credits and can struggle with implementation, leading to requests for fairer pricing.
- **Inability to Edit Files**: Many users are experiencing issues with Windsurf refusing to modify or update files, often resulting in internal errors.
   - Specific errors were reported, such as not being able to read contents and invalid service worker registrations.
- **User Experience with Different Models**: Users highlighted frustrations with the performance of various models, expressing a preference for Claude despite trying alternatives like O3 Mini and DeepSeek.
   - Some users noted that the model's context windows appear limited, affecting its performance in applications.
- **Context Window and Tool Call Limitations**: Feedback was shared about the need for clearer warnings when chat length exceeds context size, as well as issues with tool call failures.
   - Users expressed a desire for improved handling and communication regarding context limits and tool functionalities.
- **Integration of New Features and Rules**: Users are exploring how to effectively create and manage .windsurfrules files for different aspects of full-stack web applications.
   - Recommendations for rule management included setting up a bootstrap system to facilitate project organization using Windsurf.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/advanced">Windsurf - Advanced</a>: no description found</li><li><a href="https://docs.codeium.com/windsurf/usage#tool-calls">Paid Plan and Credit Usage - Codeium Docs</a>: no description found</li><li><a href="https://x.com/kevinhou22/status/1886827501004931511">Tweet from Kevin Hou (@kevinhou22)</a>: we love docs! 📖 I&#39;m working on improving / adding more @ docs shortcuts to @windsurf_ailmk what you want and I&#39;ll add as many as I can... 🧵also shoutout @mintlify for auto-hosting all docs w...</li><li><a href="https://github.com/Exafunction/codeium/issues/111">Windsurf Server Installation Fails on 32-bit ARM (armv7l) Raspberry Pi · Issue #111 · Exafunction/codeium</a>: Title: Windsurf Server Installation Fails on 32-bit ARM (armv7l) Raspberry Pi Environment Details: Device: Raspberry Pi OS: Raspbian GNU/Linux 11 (bullseye) Architecture: armv7l (32-bit ARM) Kernel...</li><li><a href="https://status.codeium.com">Codeium Status</a>: no description found</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://www.swebench.com/#verified">SWE-bench</a>: no description found</li><li><a href="https://codeium.com/pricing">Pricing | Windsurf Editor and Codeium extensions</a>: Codeium is free forever for individuals. Teams can level up with our enterprise offering for enhanced personalization and flexible deployments.</li><li><a href="https://codeberg.org/KhazAkar/canbus_visualizer">canbus_visualizer</a>: My experiment for using Windsurf AI IDE in my programming work</li><li><a href="https://github.com/ZarK/ai-rules">GitHub - ZarK/ai-rules</a>: Contribute to ZarK/ai-rules development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1336063765120028682)** (339 messages🔥🔥): 

> `O1 Pro Performance, Weak Models in Aider, OpenRouter vs Direct API, Using Shell for LLM Tools, Challenges with Testing and Refactoring` 


- **O1 Pro Performance**: Users have reported significant improvements in performance using O1 Pro, with some generating extensive code rapidly in under five minutes.
   - Comparisons to O3 Mini highlight O1 Pro's faster response times and better handling of complex tasks.
- **Weak Models in Aider**: Weak models, often used for generating commit messages and summarizing chats, can be more cost-effective and efficient than strong models for specific tasks.
   - DeepSeek V3 and other alternatives have been discussed as viable options for those seeking budget-friendly yet effective models.
- **OpenRouter vs Direct API**: Users have debated the benefits of using OpenRouter, citing better uptime and the ability to prioritize certain providers over direct API access.
   - CentMl and Fireworks are mentioned as effective deepseek providers, though they can be slower than desired.
- **Using Shell for LLM Tools**: There is interest in utilizing shell tools as a platform for LLM interaction, drawing from successful experiments with tools like Claude and LLM Functions.
   - The concept includes wrapping HTTP API functionalities into CLI tools for more streamlined operations.
- **Challenges with Testing and Refactoring**: Overhauling tests post-refactor is a common struggle, with users exploring methodologies to streamline this process without incurring excessive overhead.
   - Engagement in summarizing user requests for clearer contexts could improve the efficiency of using Aider in long discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/toby-cry-phone-spider-man-cry-phone-spider-man-phone-toby-phone-gif-12875606672124040541">Toby Cry Phone Spider Man Cry Phone GIF - Toby Cry Phone Spider man Cry Phone Spider man phone - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/i-was-thinking-the-same-thing-abed-nadir-community-i-had-the-same-thought-gif-1344064406870600973">I Was Thinking The Same Thing Abed Nadir GIF - I was thinking the same thing Abed nadir Community - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/config/reasoning.html">Reasoning models</a>: How to configure reasoning model settings from secondary providers.</li><li><a href="https://aider.chat/docs/llms/openrouter.html#controlling-provider-selection">OpenRouter</a>: aider is AI pair programming in your terminal</li><li><a href="https://github.com/quarkiverse/quarkus-mcp-servers/blob/main/README.md">quarkus-mcp-servers/README.md at main · quarkiverse/quarkus-mcp-servers</a>: Model Context Protocol Servers in Quarkus. Contribute to quarkiverse/quarkus-mcp-servers development by creating an account on GitHub.</li><li><a href="https://github.com/vivekVells/mcp-pandoc">GitHub - vivekVells/mcp-pandoc: MCP server for document format conversion using pandoc.</a>: MCP server for document format conversion using pandoc. - vivekVells/mcp-pandoc</li><li><a href="https://glama.ai/mcp/servers">Open-Source MCP servers</a>: Enterprise-grade security, privacy, with features like agents, MCP, prompt templates, and more.</li><li><a href="https://github.com/superagent-ai/reag">GitHub - superagent-ai/reag: Reasoning Augmented Generation</a>: Reasoning Augmented Generation. Contribute to superagent-ai/reag development by creating an account on GitHub.</li><li><a href="https://github.com/StevenStavrakis/obsidian-mcp">GitHub - StevenStavrakis/obsidian-mcp: A simple MCP server for Obsidian</a>: A simple MCP server for Obsidian. Contribute to StevenStavrakis/obsidian-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/sigoden/llm-functions">GitHub - sigoden/llm-functions: Easily create LLM tools and agents using plain Bash/JavaScript/Python functions.</a>: Easily create LLM tools and agents using plain Bash/JavaScript/Python functions. - sigoden/llm-functions</li><li><a href="https://neptune.ai/blog/llm-evaluation-text-summarization">LLM Evaluation For Text Summarization</a>: Evaluating text summarization is hard as there is no one correct solution, and quality often depends on the summary&#039;s context and purpose.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1336079039441535178)** (17 messages🔥): 

> `Aider file management, Aider CLI formatting issues, Auto-adding files in Aider, Understanding Aider chat modes, C# file-scoped namespaces` 


- **Aider file management automation sought**: A member expressed that using Aider feels tedious due to the manual file addition process and is looking for ways to automate it, especially since a repo map is available.
   - Another member mentioned that a plugin exists which automatically adds the currently opened file in VSCode, but the original poster is not using VSCode.
- **Aider CLI text formatting issues question**: A user raised concerns about Aider's text formatting becoming strange when the window is resized or moved.
   - Another member noted that such issues are common with many CLI applications, hinting it may not be exclusive to Aider.
- **Aider's chat modes explained**: Clarifications were provided on different modes in Aider, including `code`, `architect`, `ask`, and `help`, detailing how each mode alters interaction and commands.
   - The framed explanation highlighted that the active mode could be switched using commands like `/chat-mode`.
- **C# linting errors with file-scoped namespaces**: A member reported encountering linting errors when using file-scoped namespaces in C#, while block style namespaces do not trigger such errors.
   - They inquired about possible solutions or configurations to resolve this linting issue for file-scoped namespaces.
- **Inquiry about Aider's model usage**: A user sought clarification on when to use `aider_model` in comparison to other Aider model types like `aider_editor` and `aider_architect`.
   - An explanation followed with details on how models are utilized in different chat modes, including how specific commands can influence model choice.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://deepclaude.com/">DeepClaude</a>: no description found</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: Using the code, architect, ask and help chat modes.
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1336064222055890995)** (320 messages🔥🔥): 

> `Cursor IDE updates, Comparison with other AI tools, User experiences with different models, Cost of using AI tools, Impressions of Supermaven` 


- **Cursor IDE updates inspiring mixed reactions**: Users have experienced issues with recent updates to Cursor, with some noting slower performance and bugs compared to previous versions.
   - While some persist in using Cursor, others express frustration, claiming current models can't effectively replace their previous experiences.
- **Alternatives to Cursor IDE discussed**: Several alternatives like Supermaven and Pear AI were mentioned, with varying opinions on their effectiveness compared to Cursor.
   - Users shared experiences with Supermaven, noting it is fast but not as reliable as Cursor, particularly in the free tier.
- **Cost concerns for AI tools**: Several users discussed the high costs associated with AI tools like Cursor and Github Copilot, expressing concerns about their affordability.
   - Some users indicated preferences for lower-cost options, while others suggested the value of Cursor justified its price.
- **User experiences with different AI models**: Users shared experiences with different AI models, noting successes in building projects using Cursor and frustrations with errors produced by AI.
   - There were discussions about using models like Claude Sonnet and the practical challenges faced when utilizing AI assistance.
- **Community resources and GitHub projects**: Several users shared links to GitHub repositories for enhanced functionality with Cursor, such as `awesome-cursorrules` and other related tools.
   - These resources aim to optimize the use of Cursor and improve user experiences with different coding tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://repoprompt.com/">Repo Prompt</a>: no description found</li><li><a href="https://www.cursor.com/blog/tab-update">A New Tab Model | Cursor - The AI Code Editor</a>: Announcing the next-generation Cursor Tab model.</li><li><a href="https://toggl.com">Toggl Track: Time Tracking Software for Any Workflow</a>: no description found</li><li><a href="https://forum.cursor.com/t/has-the-fusion-model-been-rolled-out/44716/2">Has the Fusion Model been rolled out?</a>: A big request for the developers: please clarify how to understand the changelogs about the upcoming deployment of Fusion - if I have a version higher than 0.45, does this mean that I have the new tab...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i2b2eo/meta_prompts_because_your_llm_can_do_better_than/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/vbwyrde/AI_Dev_Helpers">GitHub - vbwyrde/AI_Dev_Helpers: Some tools that I find useful when using AI for development</a>: Some tools that I find useful when using AI for development - vbwyrde/AI_Dev_Helpers</li><li><a href="https://github.com/grapeot/devin.cursorrules/tree/multi-agent">GitHub - grapeot/devin.cursorrules at multi-agent</a>: Magic to turn Cursor/Windsurf as 90% of Devin. Contribute to grapeot/devin.cursorrules development by creating an account on GitHub.</li><li><a href="https://youtu.be/FrM6ZzCiLwU">DeepSeek R1 + Claude 3.5 Sonnet: The 2-Minute Developer Workflow Guide</a>: Another quick little video where I describe my latest workflow adaptation following the addition of DeepSeek R1 to Cursor as a FREE-TO-USE model!Try this and...</li><li><a href="https://www.cursor.com/changelog">Changelog | Cursor - The AI Code Editor</a>: New updates and improvements.</li><li><a href="https://github.com/PatrickJS/awesome-cursorrules?tab=readme-ov-file">GitHub - PatrickJS/awesome-cursorrules: 📄 A curated list of awesome .cursorrules files</a>: 📄 A curated list of awesome .cursorrules files. Contribute to PatrickJS/awesome-cursorrules development by creating an account on GitHub.</li><li><a href="https://github.com/askjohngeorge/pipecat-lead-qualifier/commit/7bc1b28007103793c1d1f36ebe15e158d5acad97">Refactor server structure ♻️ · askjohngeorge/pipecat-lead-qualifier@7bc1b28</a>: no description found
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1336068105755168789)** (234 messages🔥🔥): 

> `Ethics of AI Training Data, Health Care Discourse, Political Ideologies, Human Cooperation, Financial Inequality` 


- **Debate on AI Training Data Ethics**: Members discussed the challenges of defining 'dubious' data sources for AI training, with concerns raised about the implications of using datasets like Wikipedia and the morality of AI capabilities.
   - The conversation highlights a broader debate on data ownership and ethical considerations in AI development.
- **Controversy Around Health Care Arguments**: Discussion centered on right-wing arguments advocating against universal healthcare, suggesting it disproportionately benefits certain groups, leading to racial and classist implications.
   - One member illustrated this using historical context surrounding health care policies and biases against marginalized communities.
- **Political Ideological Conflict**: A heated exchange occurred regarding perceptions of leftist and rightist ideologies, raising concerns about classism, bigotry, and societal structure.
   - Participants shared differing views on human rights, societal roles, and the impact of economic systems on communities.
- **Examination of Human Cooperation**: Members referenced literature discussing human nature and cooperation, emphasizing how mutual aid is essential for survival and societal well-being.
   - The conversation touched on historical perspectives of human collaboration during crises, countering narratives of inherent selfishness.
- **Financial Inequality and Its Implications**: A significant focus was placed on the growing financial inequality in society, with discussions revolving around the impact of wealth distribution and governmental policies.
   - Claims surfaced that only an elite few benefit from the current system, while many people struggle economically, linking this to broader social issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://annas-archive.org/blog/ai-copyright.html">Copyright reform is necessary for national security</a>: Chinese LLMs (including DeepSeek) are trained on my illegal archive of books and papers — the largest in the world. The West needs to overhaul copyright law as a matter of national security.</li><li><a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: Transformer-based language models spread FLOPs uniformly across input sequences. In this work we demonstrate that transformers can instead learn to dynamically allocate FLOPs (or compute) to specific ...</li><li><a href="https://arxiv.org/abs/2401.02412">LLM Augmented LLMs: Expanding Capabilities through Composition</a>: Foundational models with billions of parameters which have been trained on large corpora of data have demonstrated non-trivial skills in a variety of domains. However, due to their monolithic structur...</li><li><a href="https://openeurollm.eu/launch-press-release">Open Euro LLM</a>: A series of foundation models for transparent AI in Europe</li><li><a href="https://fxtwitter.com/sama/status/1886559648158826518?t=buAoDwf3kJeWwjDI0vFjqA&s=19">Tweet from Sam Altman (@sama)</a>: glad to hear this; excited to bring soon to plus/free tiers!Quoting Siqi Chen (@blader) i&#39;m only a day in so far but @openai&#39;s deep research and o3 is exceeding the value of the $150K i am pay...</li><li><a href="https://fxtwitter.com/Afinetheorem/status/1886206439582015870">Tweet from Kevin A. Bryan (@Afinetheorem)</a>: The new OpenAI model announced today is quite wild. It is essentially Google&#39;s Deep Research idea with multistep reasoning, web search, *and* the o3 model underneath (as far as I know). It sometim...</li><li><a href="https://en.wikipedia.org/wiki/Negative_and_positive_rights">Negative and positive rights - Wikipedia</a>: no description found</li><li><a href="https://x.com/mgostIH/status/1880320930855153969">Tweet from mgostIH (@mgostIH)</a>: Wtf is up with deep learning???</li><li><a href="https://www.youtube.com/watch?v=AAiMOFQJPx8">Have we been doing LLM inference wrong the whole time?!?!</a>: THE HYPERFITTING PHENOMENON: SHARPENING AND STABILIZING LLMS FOR OPEN-ENDED TEXT GENERATIONArXiv: https://arxiv.org/abs/2412.04318Bytez: https://bytez.com/do...</li><li><a href="https://strategic-technologies.europa.eu/get-funding_en">Gateway to EU funding opportunities for strategic technologies</a>: Discover EU funding opportunities for strategic technologies with the Strategic Technologies for Europe Platform (STEP). Use our interactive dashboard to find EU open calls in the digital, clean, and ...</li><li><a href="https://www.youtube.com/watch?v=YcgFT4iNTUA">YOU NEED MATHEMATICAL LOGIC!</a>: A new series starts on this channel: Mathematical Logic for Proofs.Over 8,000 subscribers! THANK YOU ALL. Please continue to subscribe to this channel if the...</li><li><a href="https://www.youtube.com/watch?v=g6BK5Q_Dblo">New Religions of the 21st Century | Yuval Harari | Talks at Google</a>: Techno-Religions and Silicon Prophets: Will the 21st century be shaped by hi-tech gurus or by religious zealots – or are they the same thing?What is the curr...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1336127970116501535)** (18 messages🔥): 

> `Anthropic classifiers, Universal jailbreaks, Paper discussion attendance, Alignment techniques, Hallucination in LLMs` 


- **Anthropic Classifiers Bring Concerns**: Discussion around the paper on [constitutional classifiers](https://www.anthropic.com/research/constitutional-classifiers) yielded concerns, with one member noting a **20%** increase in inference costs and **50%** more false refusals, affecting user experience.
   - Another commented that classifiers may not sufficiently guard against **dangerous model capabilities**, especially as models advance.
- **Uncertain Future of Universal Jailbreaks**: Members discussed the concept of 'universal jailbreaks', expressing skepticism about their effectiveness on large models due to their historical failure to work consistently.
   - A member suggested that recent trends indicate a rise in **automated methods for discovering jailbreaks**, complicating the conversation.
- **Attendance Issues at Recent Discussions**: Participants noted that the paper discussion faced low turnout, with one member commenting that **knearestneighbor** left early, leading to a postponement of the discussion.
   - Another member humorously expressed disappointment at missing the discussion, highlighting the perceived quality of the paper.
- **Debate on Preventing Hallucination in LLMs**: A lively debate emerged about the concept of 'hallucination' within LLM outputs, with some arguing it's a natural aspect of model behavior rather than a flaw.
   - Members criticized the term 'hallucination' as misleading and anthromorphizing technology that generates output based on learned patterns.
- **Alignment Technological Doubts**: Concerns were raised regarding whether Anthropic's methods could effectively prevent **jail-breaking**, with some insisting that preventing hallucinations might also be an unachievable goal.
   - Comparisons were made to other efforts in alignment, with one member claiming **DeepSeek R1** has proven more effective than Anthropic's papers.



**Link mentioned**: <a href="https://arxiv.org/abs/2501.18837">Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming</a>: Large language models (LLMs) are vulnerable to universal jailbreaks-prompting strategies that systematically bypass model safeguards and enable users to carry out harmful processes that require many m...

  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1336119832021569638)** (9 messages🔥): 

> `Deepseek R1 600B, Tokenization Concerns, Image Analysis Capability, Memory and Puzzle Connection` 


- **Deepseek R1 600B impresses with performance**: A member expressed amazement at the **Deepseek R1 600B** from Together.ai after giving it a challenging grid, stating it produced outstanding results compared to smaller models.
   - They attached several screenshots showcasing its ability to conclude with the right letter.
- **Image analysis evokes laughter**: A member found it **hilarious** that the model decided to draw rather than just interpreting the text as an image, questioning its reasoning.
   - Another member agreed, noting the long reasoning path, but still found it neat.
- **Concerns over tokenizers persist**: One member expressed skepticism about the networks' reliance on **tokenizers**, mentioning the excessive memorization required for spelling.
   - They argued for a better approach that allows the model to break down words directly rather than rotting token patterns.
- **Recognition puzzle pieces**: A discussion arose on how the model seems to search its memory like solving a puzzle, particularly regarding the recognition of letters in words.
   - It was noted that **CNNs** were favored for not needing tokenizers, hinting at a desire for a combined method that merges tokenization with direct input perception.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1336087477550710855)** (6 messages): 

> `AI copyright and patent laws, Training AI and societal benefits, New AI model release requirements, Polyapprox project` 


- **AI companies manipulate copyright laws**: A member pointed out that **AI companies** often hide behind **copyright and patent laws** to protect their intellectual property, creating a dilemma between unrestricted access and tight control.
   - *Snake-oil selling* was mentioned as a critique of these practices, implying deceit in their claims.
- **Disagreement on AI training regulations**: Another member argued for the possibility of a law allowing **AI training** while changing nothing else, stating it should lead to a **free-for-all**.
   - Concerns were raised about who benefits from such laws, suggesting that primarily wealthy individuals would reap the rewards.
- **Call for free release of trained AI models**: A member insisted that if training AI is allowed with intellectual property, the trained models should be released **for free back to society**, condemning other arrangements as *bullshit*.
   - This highlights a desire for equitable access to AI advancements for societal benefit.
- **Polyapprox GitHub project shared**: A member shared a link to the **[Polyapprox GitHub project](https://github.com/EleutherAI/polyapprox)**, which seems relevant to ongoing discussions.
   - They also provided a link to a related [arXiv paper](https://arxiv.org/abs/2502.01032), encouraging members to explore further.
- **Screenshots and image analyses provided**: Several screenshots were shared along with links for image analyses, highlighting **potential insights or discussions** relevant to the AI topics.
   - These visuals were attached to support ongoing discussions in the channel.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1336082071562620950)** (141 messages🔥🔥): 

> `Model Performance Issues, API Model Specification, Compatibility with Intel Macs, RAG and Inference in LM Studio, Tool & Function Callbacks` 


- **Model Performance Variability**: Users noted inconsistent performance with various models, particularly stating that the **Deepseek R1 abliterated Llama 8B** model was underwhelming compared to the smaller **Qwen 7B** and **1.5B models**.
   - One user questioned how to fully uncensor models, highlighting a discrepancy in capabilities between newer versions and expecting the better performance from them.
- **Understanding API Model Usage**: Discussions around using 'local-model' in API calls clarified that it acts as a placeholder for the specific model name being used, especially in setups with multiple models loaded.
   - It was suggested that explicitly obtaining the model names before making API requests can prevent ambiguity in model selection.
- **Intel Mac Compatibility Concerns**: A user inquired about LM Studio's compatibility with Intel-based Macs, discovering that current versions are supported exclusively on Apple Silicon.
   - It was highlighted that there are no self-build options for LM Studio as it remains a closed-source project, prompting suggestions to use alternative systems.
- **Exploring RAG for Specific Use Cases**: Users delved into using Retrieval-Augmented Generation (RAG) to enhance inference capabilities in LM Studio for domain-specific tasks without finetuning.
   - One user emphasized the importance of utilizing domain knowledge in vector stores as the first step before considering more complex solutions like model finetuning.
- **Function and Tool Callback Strategies**: A user sought resources on proper prompting techniques, particularly emphasizing function and tool callbacks, as they faced persistent response issues.
   - Another member shared their innovative approach using a Directed Acyclic Graph (DAG) workflow to enhance selections of AI responses based on past performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/who-knows-shrug-what-shades-on-idk-gif-15962763">Who Knows Shrug GIF - Who Knows Shrug What - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lmstudio.ai/docs/api">LM Studio as a Local LLM API Server | LM Studio Docs</a>: Run an LLM API server on localhost with LM Studio</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: New to Unsloth?</li><li><a href="https://tenor.com/view/richard-stalman-richard-stalman-saint-ignucius-gnu-gif-13909134">Richard Stalman Richard GIF - Richard Stalman Richard Stalman - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://lmstudio.ai/docs/api/endpoints/rest#get-apiv0models>">LM Studio REST API (beta) | LM Studio Docs</a>: The REST API includes enhanced stats such as Token / Second and Time To First Token (TTFT), as well as rich information about models such as loaded vs unloaded, max context, quantization, and more.</li><li><a href="https://github.com/Blaizzy/mlx-vlm">GitHub - Blaizzy/mlx-vlm: MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) on your Mac using MLX.</a>: MLX-VLM is a package for inference and fine-tuning of Vision Language Models (VLMs) on your Mac using MLX. - Blaizzy/mlx-vlm</li><li><a href="https://lmstudio.ai/docs/basics/rag">Chat with Documents | LM Studio Docs</a>: How to provide local documents to an LLM as additional context</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/11483">Feature Request: Qwen 2.5 VL · Issue #11483 · ggerganov/llama.cpp</a>: Prerequisites I am running the latest code. Mention the version if possible as well. I carefully followed the README.md. I searched using keywords relevant to my issue to make sure that I am creati...</li><li><a href="https://github.com/QwenLM/Qwen2.5-VL/issues/7">Support for Llama.cpp · Issue #7 · QwenLM/Qwen2.5-VL</a>: Could we have support for Llama.cpp? That will make the model more accessible to many popular tools like Ollama, LM Studio, Koboldcpp, text-generation-webui, etc. Thank you so much!
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1336073580903010365)** (120 messages🔥🔥): 

> `RAM differences in server setups, Running models on different hardware, M4 Ultra performance expectations, GPU configurations and capabilities, Inference speeds of various models` 


- **RAM Types and Inference Issues**: Discussion highlighted the differences between **UDIMM and RDIMM** RAM types, where one user faced issues as their **128GB of RAM** did not fit in their inference server.
   - Another member noted they experienced limitations with a **single 7900 XTX** GPU for running a **70B model**, suggesting a need for multiple GPUs for viable inference speeds.
- **M4 Ultra's Performance Dilemma**: Members expressed skepticism about the **M4 Ultra's** ability to deliver strong performance with rumors pointing towards a **$1200 starting price** for a system with 128GB of RAM.
   - Some speculate it may not outpace NVIDIA's **Project DIGITS**, which has superior interconnect speeds for clustering models.
- **Confusion Over Inference Speed Figures**: Discrepancies in inferred speeds were noted, where one user reported **30-32 TPS** with their **M2 Ultra** while others expressed doubts about those numbers needing further specification on model versions.
   - Discussion surrounding performance metrics included questions on whether models were running **4-bit quantization**, with general consensus that larger models like **70B** incur diminishing returns in speed.
- **GPU Configurations for AI Workstations**: Users discussed various GPU configurations, noting that a combination of **GPUs like 3090 and 4070 Super** could offer substantial inference capabilities based on available VRAM.
   - Concerns were raised about proper cooling solutions for high-performance rigs, especially with products like the **16” MacBook Pro** connected to high-end GPUs.
- **Fine-tuning Resource Allocation**: A user suggested that setting the **wired limit for GPU memory** can help tune performance, yet others reported issues with gibberish outputs when running larger models.
   - The effectiveness of resource tuning was questioned when smaller models were the only ones capable of running properly on certain setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.exolabs.net/day-2/">12 Days of EXO</a>: 12 Days of Truly Open Innovation</li><li><a href="https://tenor.com/view/house-w-hat-huh-confused-unsure-gif-4211197">Confused GIF - House W Hat Huh - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hvqydy/hp_z2_mini_g1a_is_a_workstationclass_mini_pc_with/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1336120618323546164)** (1 messages): 

> `New Chief Security Officer, Security updates` 


- **Meet Perplexity's New Chief Security Officer**: A message was shared from our new **Chief Security Officer** at Perplexity, highlighting the importance of security moving forward.
   - An introduction video titled *Jimmy* was attached for the community to learn more about their role.
- **Introduction Video of New CSO**: An introductory video titled [*Jimmy*](https://cdn.discordapp.com/attachments/1047204950763122820/1336120617967026319/Jimmy.MOV?ex=67a34f8b&is=67a1fe0b&hm=f6e56ab3c0f598299bf922a9aedeb20380d5ab088b4921c9f03f6867e0ef3437&) was shared to engage the community with the new CSO's vision for security.
   - This video serves as an opportunity for the team to connect and understand security priorities directly from leadership.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1336069761003687987)** (165 messages🔥🔥): 

> `Perplexity Pro Features, API Usage and Limits, User Experience Feedback, Academic Writing Assistance, Model Comparison and Performance` 


- **Perplexity Pro is Worthwhile**: Many users agree that the Perplexity Pro plan is valuable, especially with its nearly unlimited daily R1 usage limits.
   - One user mentioned that Deepseek offers fewer queries, highlighting Perplexity's superior server performance.
- **API and Model Limitations**: Users reported difficulties using certain models like Sonar in APIs, with some only able to utilize reasoning models.
   - Additionally, there were queries concerning R1's usability and whether the API has limitations on models.
- **User Suggestions and Feedback**: Several users expressed interest in providing suggestions based on their experiences with Perplexity Pro.
   - Discussion led to referrals to specific channels for making formal suggestions and sharing feedback.
- **Assistance with Academic Writing**: Users sought advice on how to effectively leverage Perplexity for academic writing and finding articles.
   - While some preferred brainstorming with the AI, others mentioned using different tools for writing assistance.
- **Service Performance and Issues**: There were reports of slow responses and potential issues with Perplexity, prompting inquiries about service reliability.
   - Users checked the status page for ongoing issues, confirming some functionalities were operational despite reported lags.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/avalanche-avax-gif-21537601">Avalanche Avax GIF - Avalanche AVAX - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.cplx.app/">Complexity</a>: An enhanced version of Perplexity.ai that everyone has ever wanted.</li><li><a href="https://www.perplexity.ai/backtoschool">Perplexity - Race to Infinity</a>: Welcome back to school! For just two weeks, redeem one free month of Perplexity Pro on us. Refer your friends, because if your school hits 500 signups we'll upgrade that free month to an entire free y...</li><li><a href="https://x.com/aravsrinivas/status/1884509590684934211?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: @julheetel8890 Full.</li><li><a href="https://x.com/apostraphi/status/1886539187353960741?s=61">Tweet from Phi Hoang (@apostraphi)</a>: new hire alertQuoting Perplexity (@perplexity_ai) A message from our new Chief Security Officer at Perplexity:</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity Status</li><li><a href="https://youtu.be/8uPmC5BQtCw?si=98QvOsWrzWbEs40R">Public vs Private Actions in Nelima (AI) Explained in 420 Seconds</a>: Since our first video on &quot;Programming a Large Action Model&quot;, Nelima has come a long way! In this video, we’re excited to introduce a powerful new feature: Pr...</li><li><a href="https://www.wikidata.org/wiki/Q123403392">Perplexity</a>: chatbot search engine</li><li><a href="https://www.wikidata.org/wiki/Q124333951">Perplexity AI, Inc.</a>: conversational search engine company
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1336109502721691649)** (13 messages🔥): 

> `Phobia information sources, MSI A520M motherboard, Pigs' appearance, Trump executive order, Linux desktop usage` 


- **Exploring Phobia Resources**: Multiple users shared a link about websites that list **phobias**, providing a consolidated resource for further reading [here](https://www.perplexity.ai/search/show-me-sites-that-list-phobia-Jb9EQhckS66QFqIDYJvj1A).
   - The shared source offers valuable insights into various phobias and their descriptions.
- **Deep Dive into MSI A520M Motherboard**: Information regarding the **MSI A520M A Pro motherboard** was shared, including specifications and reviews [found here](https://www.perplexity.ai/search/motherboard-msi-a520m-a-pro-aoKBBPs2Skystw7fZnDqJw#1).
   - Users can find detailed comparisons and user experiences for this motherboard.
- **Discussing Pigs and Their Looks**: A link was provided relating to the question of *why pigs are considered ugly*, discussing perceptions and biology [here](https://www.perplexity.ai/search/why-pigs-are-ugly-f54qHxRsQ1SpIkudhRoHHA#0).
   - The exploration offers a humorous yet informative perspective on this animal's image.
- **Trump's Recent Executive Order**: A member shared news about **Trump signing an executive order**, linking to the article [here](https://www.perplexity.ai/page/trump-signs-executive-order-to-.xV8X9ILSuqTDd_jZeeKvQ).
   - The order’s implications are explored in detail, emphasizing its political relevance.
- **Trends in Linux Desktop Usage**: Several users referenced a link discussing **Linux desktop usage over time** and its evolution [here](https://www.perplexity.ai/search/linux-desktop-usage-over-time-QFfP46jEShCTHE0bW7ODpg#0).
   - Insights into the growing and changing trends in Linux adoption highlight its market position.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1336068382348546068)** (9 messages🔥): 

> `Llama 3.1 Sonar Model Deprecation, Sonar-Reasoning Errors, API Access for Image Retrieval, Litellm Model Updates, API Model Name Identification` 


- **Llama 3.1 Sonar Model Deprecation.**: A user reported receiving a deprecation email about **llama-3.1-sonar-small-128k-online** two weeks ago and noted that switching to `sonar` resulted in a **5-10 second latency increase**.
   - They inquired whether this delay was expected and sought advice on how to mitigate the issue.
- **Sonar-Reasoning Models Encounter 500 Errors.**: A user requested help regarding **sonar-reasoning-pro** and encountered a **500 error** when attempting to use it.
   - The exact cause of the issue was not clarified, raising concerns about access to the model.
- **API User Seeking Image Retrieval Access.**: A user expressed a need to retrieve images for their PoC but found that they need to be a **tier-2 API user** to access this feature.
   - They asked if it was possible to grant temporary access to utilize their existing credits.
- **Litellm Compatibility with New Sonar Models.**: A user questioned why they could not use `litellm` with the new **sonar/sonar-pro** models, suspecting an update issue.
   - The user seemed frustrated and sought clarification about model availability.
- **Identifying API Model Names in Use.**: A user is trying to ensure that their systems use the correct API models before the old ones deprecate, but they can't see which model is currently invoked.
   - They noted that their API usage reports as **70b-online** regardless of the model used and asked if there is a way to verify the model name being called.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1336067470536085695)** (171 messages🔥🔥): 

> `DeepSeek, O1 Pro Model Performance, AI Sentience and Future Impact, Orchestration Services in AI, User Experiences with OpenAI and Other Models` 


- **DeepSeek's Community Impact**: Users discussed how DeepSeek R1 introduces new techniques beneficial to the wider community, potentially disrupting the trend of megacorps owning AI technology.
   - Concerns about DeepSeek sending information to China were raised, indicating a need for transparency in data handling.
- **Impressive O1 Pro Usage**: Members shared their experiences with O1 Pro, highlighting its ability to generate multiple mini-games in one session without errors, showcasing its performance.
   - One user plans to rigorously test O1 Pro with ambitious prompts and demos, indicating confidence in the model's capabilities.
- **Debate on AI Sentience**: A chat participant questioned whether AI is sentient and speculated on its potential impact on jobs and society.
   - Responses ranged from skepticism about AI sentience to more humorous takes on its future, with no serious concerns raised.
- **Understanding Orchestration Services**: Users discussed the concept of orchestration services in AI, describing how they can allow models to delegate tasks to multiple instances.
   - This led to a broader conversation about the performance capabilities of different AI models, particularly in relation to O1 Pro and R1.
- **Mixed User Experiences with OpenAI Models**: Several users recounted their frustrations with getting OpenAI models to follow specific instructions, especially when refining papers.
   - Despite challenges, many noted the impressive capabilities of the models, with some expressing excitement for further developments in AI technology.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dataconomy.com/2025/01/31/deepseek-r1-vs-o3-mini-in-performance-cost-and-usability-showdown/">DeepSeek R1 vs o3-mini in performance, cost, and usability showdown</a>: User analyses and benchmark tests reveal a split verdict between DeepSeek R1 and o3-mini, with developers and businesses prioritizing distinct</li><li><a href="https://tenor.com/view/hate-ignorance-fear-fire-science-gif-16741306">Hate Ignorance GIF - Hate Ignorance Fear - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1igyy0n/introducing_deeper_seeker_a_simpler_and_oss/">Introducing Deeper Seeker - A simpler and OSS version of OpenAI's latest Deep Research feature.</a>: Posted in r/LocalLLaMA by u/hjofficial • 218 points and 54 comments</li><li><a href="https://youtube.com/shorts/I_bGa-xIHkk?feature=shared">Is DeepSeek Lying to you? #shorts #wireshark #deepseek #privacy #cybersecurity</a>: #shorts #wireshark #deepseek #privacy #cybersecurity
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1336064021312573440)** (11 messages🔥): 

> `User feedback on AI performance, AI Storybooks for Kids, Device login limits for Pro users, Recent updates and emoji usage, Accuracy of deep research info` 


- **Mixed Reactions to AI Updates**: Users are expressing frustration about recent AI updates, with one stating, *'you need a 100000 page paragraph for it to understand what a 4 year old would know.'* Another user noticed increased emoji usage in code responses after an update.
   - One member suggested limiting emoji usage to improve coding focus, stating their experience with the AI in handling direct feedback.
- **Inquiry on AI Storybooks for Kids**: A user asked if anyone is creating **AI Storybooks for Kids**, indicating interest in this niche.
   - This suggests ongoing exploration and potential development in child-friendly AI applications.
- **Pro Users Curious About Device Restrictions**: A member inquired whether the **Pro version** would limit the number of login devices, reflecting concerns about accessibility.
   - This suggests users want clarity on usage policies for upgraded services.
- **Question Around GPT-4's Reasoning Ability**: A user questioned why **GPT-4o is now reasoning**, showing curiosity about recent enhancements.
   - This highlights users' awareness and engagement with AI's evolving capabilities.
- **Evaluating the Accuracy of Deep Research**: A user queried about the accuracy of **Deep Research's information**, seeking insights from a Pro user.
   - Another member rated it on a scale from 1 to 10, implying an interest in understanding the reliability of AI-generated insights.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1336372248734339174)** (1 messages): 

> `Structured generation, JSON Schema enhancements, UIForm utility` 


- **Trick to Boost Model Performance**: A member shared their experience with **structured generation** and utilizing 'thinking' fields in JSON schemas and Pydantic models to enhance **model performance** during inference.
   - However, they cautioned that this method can *pollute the data structure definition*.
- **Introducing UIForm for Schema Management**: The same member announced the open-sourcing of **UIForm**, a utility designed to simplify the addition or removal of fields from schemas through **JSON Schema extras**.
   - They encouraged feedback on the tool, noting that installation requires just a simple command: `pip install uiform`.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1336372248734339174)** (1 messages): 

> `Structured Generation Techniques, UIForm Open Source Utility, JSON Schema Enhancements` 


- **Structured Generation Techniques Introduced**: A member shared insights on using **'thinking' fields** in JSON schemas or Pydantic models to enhance model performance by including inference-time computations.
   - However, they noted the downside of this approach as it **pollutes the data structure definition**.
- **Feedback Request on UIForm Utility**: The same member announced the open-sourcing of **UIForm**, a utility that simplifies adding or removing specified fields in schema using JSON Schema extras.
   - They invited **feedback** from the community, mentioning that installing it is as simple as running `pip install uiform`.
- **UIForm's Flexibility Highlighted**: The introduction of **UIForm** facilitates improved management of structured generation in projects by allowing dynamic changes to data schemas.
   - This could potentially streamline workflows for developers dealing with complex data models, increasing efficiency.


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1336069524092620843)** (43 messages🔥): 

> `SoftBank OpenAI Partnership, Google Gemini Updates, Harmonic Loss for Neural Networks, MultiChallenge Benchmark for Conversations, OpenAI Website Redesign` 


- **SoftBank to Invest $3 Billion in OpenAI Products**: SoftBank has committed to purchasing **$3 billion** worth of **OpenAI** products annually and is forming a Japan-focused joint venture named **Cristal Intelligence**.
   - The venture will provide a business version of **ChatGPT** and is negotiating a **$40 billion** investment that could value OpenAI at **$300 billion**.
- **Updates on Google Gemini for Workspace**: Google announced that **Gemini for Google Workspace** is no longer offering add-ons but is integrating AI capabilities into all Business and Enterprise Editions, enhancing productivity and data control.
   - With over a million users in the past year, this integration aims to transform how businesses leverage generative AI tools.
- **Introducing Harmonic Loss for Neural Networks**: A new paper introduces **harmonic loss** as an alternative to **cross-entropy loss**, highlighting benefits like better interpretability and faster convergence during training.
   - It suggests that models trained with this loss function represent semantically related word pairs more effectively.
- **Launching MultiChallenge Benchmark for LLMs**: The **MultiChallenge** benchmark has been revealed to evaluate large language models on their ability to conduct **multi-turn conversations**.
   - This initiative aims to address an underexamined area critical for the applications of language models in real-world settings.
- **OpenAI Website Gets a Design Overhaul**: OpenAI has updated its website to align with new design guidelines, signaling a fresh visual direction for the brand.
   - This update is part of a broader effort to enhance user experience and branding consistency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Presidentlin/status/1886771841450017211">Tweet from Lincoln 🇿🇦 (@Presidentlin)</a>: cope cope cope cope cope cope cope cope</li><li><a href="https://x.com/dbaek__/status/1886781435794927697">Tweet from David D. Baek (@dbaek__)</a>: 7/9 When we train GPT-2 with harmonic loss, we observe that models tend to represent semantically related word pairs (e.g. man:woman::king:queen), in a more rectangular parallelogram structure -- Harm...</li><li><a href="https://youtu.be/k3d_xeVxEOE">Refreshed.</a>: no description found</li><li><a href="https://support.google.com/a/answer/13623623">Gemini for Google Workspace - Business / Enterprise - Google Workspace Admin Help</a>: no description found</li><li><a href="https://x.com/dbaek__/status/1886781418115862544">Tweet from David D. Baek (@dbaek__)</a>: 1/9 🚨 New Paper Alert: Cross-Entropy Loss is NOT What You Need! 🚨We introduce harmonic loss as alternative to the standard CE loss for training neural networks and LLMs! Harmonic loss achieves 🛠️si...</li><li><a href="https://huggingface.co/blog/open-deep-research">Open-source DeepResearch – Freeing our search agents</a>: no description found</li><li><a href="https://x.com/btibor91/status/1886880680077906376?s=61">Tweet from Tibor Blaho (@btibor91)</a>: OpenAI website has now been updated based on the new design guidelinesQuoting nic (@nicdunz) the official openai website has been updated with their new design guidelines in place and other things</li><li><a href="https://fxtwitter.com/btibor91/status/1886508640263397705">Tweet from Tibor Blaho (@btibor91)</a>: The information reports that SoftBank has committed to purchasing $3 billion worth of OpenAI products annually while also forming a Japan-focused joint venture- SoftBank will distribute OpenAI technol...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1336067983709175879)** (40 messages🔥): 

> `Noam Brown's Views, Zetta vs. Llama-1 Dynamics, OpenAI's Hardware Ambitions, Internal Cultural Issues, Collaboration with OpenAI` 


- **Noam Brown's Controversial Stance**: Discussion sparked around Noam Brown's view that **Wikipedia** is biased, suggesting a preference for AI **training** over traditional sources.
   - *This aligns with a broader techbro perspective*, causing surprise among members who generally hold a favorable opinion of him.
- **Zetta and Llama-1 Team Rivalry**: Conversations revealed tension regarding the differing cultures and development paths of the **Zetta** and **Llama-1** teams, with accusations of internal dysfunction.
   - One contributor emphasized that **internal competition** and lack of transparency led to difficulties surrounding project outcomes.
- **OpenAI Set for Hardware Expansion**: OpenAI filed a trademark for ventures into **humanoid robots** and **AI-driven VR headsets**, signaling an intent to enter hardware markets.
   - This new direction may position OpenAI against major players like Meta and Apple, but could be a daunting task amid **crowded hardware challenges**.
- **Concerns About Internal Communication**: Participants expressed unease over airing grievances publicly on Twitter, particularly regarding influential figures discussing internal frustrations.
   - One noted that individuals should consider **communication training** before tweeting about sensitive issues.
- **Breakthrough in Robotics from Collaboration**: A exit from a Collaboration Agreement with OpenAI was announced, citing a major military tech breakthrough in **robot AI** developed in-house.
   - The announcement hinted at an upcoming reveal of a **humanoid** project, raising eyebrows about OpenAI's ongoing robotics ambitions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.businessinsider.com/openai-trademark-humanoid-robots-vr-headsets-sam-altman-hardware-2025-2">OpenAI files a trademark application for humanoid robots and VR headsets as Sam Altman teases big hardware ambitions</a>: OpenAI&#x27;s trademark application lists AI-powered robots, VR headsets, and wearables in the latest sign of a possible hardware push. </li><li><a href="https://x.com/suchenzang/status/1886544793511080103">Tweet from Susan Zhang (@suchenzang)</a>: if you&#39;re bragging about internal cultural dysfunction in your own lab, where ICs are pitted against each other in the name of some regional glory...it&#39;s no wonder your models are utterly irre...</li><li><a href="https://fxtwitter.com/suchenzang/status/1886635726655430786">Tweet from Susan Zhang (@suchenzang)</a>: i wonder what a PIP looks like for a chief AI scientist</li><li><a href="https://x.com/ArmenAgha/status/1886522896077439187">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: This is absolutely not true about what happened with Zetta. Do we really want to open up about what happened here?Quoting Yann LeCun (@ylecun) You misread.There had been multiple LLM projects within F...</li><li><a href="https://fxtwitter.com/soumithchintala/status/1886562033048396241">Tweet from Soumith Chintala (@soumithchintala)</a>: you were/are the Chief Scientist of Meta, and a FAIR Lead -- where both Zetta and Llama were located; I think characterizing any team within your direct influence in a bad light in public is not nice....</li><li><a href="https://x.com/Dorialexander/status/1886774547640189294">Tweet from Alexander Doria (@Dorialexander)</a>: Llama civil war, summarized.</li><li><a href="https://x.com/adcock_brett/status/1886860098980733197">Tweet from Brett Adcock (@adcock_brett)</a>: Today, I made the decision to leave our Collaboration Agreement with OpenAIFigure made a major breakthrough on fully end-to-end robot AI, built entirely in-houseWe&#39;re excited to show you in the ne...</li><li><a href="https://x.com/elder_plinius/status/1886520062586372224">Tweet from Pliny the Liberator 🐉 (@elder_plinius)</a>: @alexalbert__ @AnthropicAI ggs</li><li><a href="https://x.com/ylecun/status/1886149808500457691">Tweet from Yann LeCun (@ylecun)</a>: You misread.There had been multiple LLM projects within FAIR for years. Some were open sourced as research prototypes (e.g. OPT175B, Galactica, BlenderBot...).In mid-2022, FAIR started a large LLM pro...</li><li><a href="https://fxtwitter.com/ArmenAgha/status/1886549536300261706">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: Only one of the teams between Zetta/LLaMa had an open-source pre-training codebase, shared datasets and experiments internally, used standardized evaluation sets, published internal notes and did thin...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1336085546845474991)** (84 messages🔥🔥): 

> `GRPO effectiveness on Llama 2, DeepSeek training on Huawei Ascend, NVIDIA Digits interest, RLHF and training costs, Website certificate issues` 


- **GRPO shines with Llama 2**: Recent findings indicate that **GRPO** significantly boosts accuracy on **GSM8K**, achieving a **+15 point increase** on the Llama 2 7B model.
   - This demonstrates that existing models can still benefit from reinforcement learning techniques, which had been questioned in prior discussions.
- **DeepSeek can now run on Huawei Ascend**: A discussion revealed that the **DeepSeek V3** type model can be trained on **Huawei Ascend** hardware, expanding its accessibility for researchers.
   - However, there are doubts about the reliability of certain claims regarding performance and cost reductions tied to this platform.
- **High interest in NVIDIA Digits**: An NVIDIA representative mentioned that there's significantly more interest in **Digits** from the research community compared to past announcements like **Blackwell**.
   - This heightened interest is seen as a positive development for cash-strapped universities looking for affordable solutions.
- **Exploring costs of RLHF models**: Further conversations indicated skepticism regarding the training costs of RLHF models, particularly in relation to **DeepSeek's R1** model.
   - Concerns were raised about whether the reported costs were accurately represented, referencing prior communications on the subject.
- **Website certificate hurdles**: Several members expressed frustration over ongoing issues with SSL certificates on **GitHub Pages** which complicate website accessibility.
   - Despite DNS issues being resolved, certificate problems have persisted, causing concerns about user experience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.pi.website/blog/openpi">Open Sourcing π0</a>: Physical Intelligence is bringing general-purpose AI into the physical world.</li><li><a href="https://x.com/dimitrizho/status/1886713381706449379>">Tweet from Dimitri Zhorzholiani (@dimitrizho)</a>: Deepseek researcher says it only took 2-3 weeks to train R1&R1-Zero</li><li><a href="https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1">What went into training DeepSeek-R1?</a>: This Gradient Updates issue explores DeepSeek-R1’s architecture, training cost, and pricing, showing how it rivals OpenAI’s o1 at 30x lower cost.</li><li><a href="https://www.youtube.com/playlist?list=PLgKuh-lKre1058wlfuwtOYyemY5qoxavQ">LLM25-1: LLMs, Cognitive Science, Linguistics, and Neuroscience</a>: February 3-7, 2025</li><li><a href="https://x.com/rosstaylor90/status/1886625126222852208">Tweet from Ross Taylor (@rosstaylor90)</a>: No one is saying RL didn’t work for reasoning. The argument is about internal reasoning emergence, not absolute performance boosts with RL.Quite the opposite in fact - we had PPO on Llama 2 base model...</li><li><a href="https://www.rlhfbook.com">(WIP) A Little Bit of Reinforcement Learning from Human Feedback</a>: The Reinforcement Learning from Human Feedback Book</li><li><a href="https://x.com/georgejrjrjr/status/1886654522539266289">Tweet from George (@georgejrjrjr)</a>: Rough upper bound for V3-&gt;R1 training compute spend this implies aligns with the upper end of Epoch’s range (500k-2M).</li><li><a href="https://x.com/rdolmedo_/status/1886505669622149139">Tweet from Ricardo Dominguez-Olmedo (@rdolmedo_)</a>: Does reinforcement learning with verifiable rewards work only for recent model families?It turns out that GRPO also works very well for Llama 2 7B, with an impressive +15 accuracy point increase in GS...</li><li><a href="https://x.com/teortaxesTex/status/1886526422493143268">Tweet from Teortaxes▶️ (DeepSeek🐳 Cheerleader since 2023) (@teortaxesTex)</a>: DeepSeek V3 type model can now be trained on Huawei Ascend.</li><li><a href="https://interconnects.ai">Interconnects | Nathan Lambert | Substack</a>: The cutting edge of AI, from inside the frontier AI labs, minus the hype. The border between high-level and technical thinking. Read by leading engineers, researchers, and investors on Wednesday morni...</li><li><a href="https://www.interconnects.ai">Interconnects | Nathan Lambert | Substack</a>: The cutting edge of AI, from inside the frontier AI labs, minus the hype. The border between high-level and technical thinking. Read by leading engineers, researchers, and investors on Wednesday morni...</li><li><a href="https://x.com/TheHumanoidHub/status/1886679733460721875">Tweet from The Humanoid Hub (@TheHumanoidHub)</a>: CMU researchers, in collaboration with NVIDIA, present ASAP, a two-stage framework for humanoid robot agility.It pre-trains motion policies on human data, then refines them with real-world corrections...</li><li><a href="https://www.youtube.com/watch?v=KtBcIDtS13M&list=PLgKuh-lKre1058wlfuwtOYyemY5qoxavQ&index=6">How DeepSeek changes the LLM story</a>: Sasha Rush (Cornell University)https://simons.berkeley.edu/talks/sasha-rush-cornell-university-2025-02-03LLMs, Cognitive Science, Linguistics, and Neuroscience</li><li><a href="https://youtu.be/k3d_xeVxEOE?si=eVIhUSXDlg2iu_h2~~">Refreshed.</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1336220825409556490)** (3 messages): 

> `Qwen/Llama models, Old RL + NLP papers` 


- **Confidence in the Clown Hat Gambit**: A member asserted that *never once was wrong to bet on the guy in the clown hat* after a discussion referencing [this tweet](https://x.com/teortaxesTex/status/1886553466580988282).
   - This sparked a reaction with another member questioning, *there's no way it's this easy right*.
- **Bringing Back Old Research**: Another member signaled a need to revisit old **RL + NLP papers**, indicating their potential relevance today.
   - Following this, it was suggested that simply swapping any model for **Qwen/Llama** could generate enough quality material for submission to **arXiv**.



**Link mentioned**: <a href="https://x.com/teortaxesTex/status/1886553466580988282">Tweet from Teortaxes▶️ (DeepSeek🐳 Cheerleader since 2023) (@teortaxesTex)</a>: never once was wrong to bet on the guy in the clown hat.Quoting anton (@abacaj) wait... there&#39;s no way it&#39;s this easy right

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1336287885825151067)** (12 messages🔥): 

> `Prime paper release, Scaling deep learning models, JAX usage in industry` 


- **Prime Paper Finally Published**: The highly anticipated [Prime paper](https://arxiv.org/abs/2502.01456) has been released with contributions from numerous authors including **Ganqu Cui** and **Lifan Yuan**.
   - *This paper introduces new concepts that could reshape model performance optimization.*
- **Scaling Book Offers Model Optimization Insights**: A shared link to the [Scaling Book](https://jax-ml.github.io/scaling-book/) reveals strategies for optimizing model performance across various scales, from single accelerators to large clusters.
   - It discusses practical approaches to estimate training costs and the impact of hardware specifications on model efficiency.
- **JAX Usage in Google and xAI**: Discussion arose about who utilizes JAX, with **Google** and **xAI** being noted as primary users.
   - A suggestion was made implying that Elon Musk should consider deploying **Grok** more actively in alignment with JAX capabilities.
- **Crowded Space in AI Tools**: The conversation touched on the **crowded** market of AI tools, highlighting the competition in deep learning methodologies.
   - *With many platforms available, the effectiveness of each remains a hot topic of debate among practitioners.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jax-ml.github.io/scaling-book/"> How To Scale Your Model </a>: no description found</li><li><a href="https://arxiv.org/abs/2502.01456">Process Reinforcement through Implicit Rewards</a>: Dense process rewards have proven a more effective alternative to the sparse outcome-level rewards in the inference-time scaling of large language models (LLMs), particularly in tasks requiring comple...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1336141970229166121)** (2 messages): 

> `Mandatory Licensing for AI, Copyright Issues in AI, Fair Use for AI Use` 


- **Mandatory Licensing Proposal Gains Traction**: A member proposed that a **mandatory licensing system with royalties**, similar to the music industry, is a viable alternative to abolishing copyright entirely.
   - This suggestion seeks to provide clear guidelines for using AI-generated content while ensuring creators receive **proper compensation**.
- **Legal Gray Areas Haunt AI Use**: Without clear regulations, members expressed concern that AI use may fall into **legally gray** territories, particularly regarding data usage.
   - They emphasized the need to either declare AI use as **fair use** or face implications from using **tainted data**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1336107198660481124)** (17 messages🔥): 

> `1D Block Tiling vs Cublas, LlamaGen Image Models, Gen AI Hackathon, MAGVIT Video Tokenization, Generation Time Comparisons` 


- **Optimizing 1D Block Tiling Performance**: Members discussed the implementation of [1D block tiling against Cublas](https://github.com/Omkar-Kakade-Github/Blazing_CUDA_SGEMM/blob/89230ac77af761d2d65cad97b4409f1400b6fe7c/kernels/04_1D_block_tiling.cu), with one participant noting improved benchmarks after reducing thread blocks to **256**.
   - *Five warmups followed by averaging timings for benchmarks* were suggested to stabilize measurements.
- **LlamaGen: A New Contender in Image Generation**: The introduction of **LlamaGen** promises state-of-the-art image generation using *next-token prediction* models, outperforming diffusion frameworks like **LDM** and **DiT**.
   - Questions arose regarding the absence of generation time comparisons, leading to speculation about performance metrics.
- **Upcoming Gen AI Hackathon Announcement**: **CreatorsCorner** invites participants to its **fourth hackathon** in collaboration with major companies, offering **$35k+ in prizes**.
   - Teams are encouraged to build autonomous AI agents that enhance user capabilities in voice and video applications.
- **Discussion on MAGVIT and Video Tokenization**: Members expressed disappointment over **MAGVIT2** not being open-sourced, referencing its potential for improving video tokenization.
   - Despite this, there were mentions of tricks to enhance speed in generation processes.
- **Concerns Over Generation Times**: A participant noted that **LlamaGen** has **slow generation times** compared to diffusion models, hinting at possible optimizations.
   - This led to a suggestion about the potential of using **MAGVIT** for enhanced video processing effectiveness.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.06525">Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation</a>: We introduce LlamaGen, a new family of image generation models that apply original ``next-token prediction&#39;&#39; paradigm of large language models to visual generation domain. It is an affirmative...</li><li><a href="https://arxiv.org/abs/2409.18869">Emu3: Next-Token Prediction is All You Need</a>: While next-token prediction is considered a promising path towards artificial general intelligence, it has struggled to excel in multimodal tasks, which are still dominated by diffusion models (e.g., ...</li><li><a href="https://lu.ma/fyu8iqnk">Multimodal AI Agents - Hackathon · Luma</a>: Gen AI AgentsCreatorsCorner, collaborating with Google Deepmind, Weights &amp; Biases, Together.ai, Stytch, Senso, LlamaIndex and others enthusiastically…</li><li><a href="https://huggingface.co/papers/2409.18869#66fd72c610a11719b680cfbb">Paper page - Emu3: Next-Token Prediction is All You Need</a>: no description found</li><li><a href="https://github.com/Omkar-Kakade-Github/Blazing_CUDA_SGEMM/blob/89230ac77af761d2d65cad97b4409f1400b6fe7c/kernels/04_1D_block_tiling.cu">Blazing_CUDA_SGEMM/kernels/04_1D_block_tiling.cu at 89230ac77af761d2d65cad97b4409f1400b6fe7c · Omkar-Kakade-Github/Blazing_CUDA_SGEMM</a>: Contribute to Omkar-Kakade-Github/Blazing_CUDA_SGEMM development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1336221236912717846)** (13 messages🔥): 

> `Triton kernel optimization, k_cross dimensions, Error in tutorial example, TMA performance, Warp specialization` 


- **Seeking Triton Kernel Optimization Help**: A user is requesting help with optimizing a Triton implementation to better perform a memory-intensive operation. They noted that their Triton code is **200x slower** than the original PyTorch version.
   - Discussion revealed that **k_cross** is derived from crossing different rows in matrix **k**, for which timely optimization is crucial for large dimensions.
- **Error in Triton Tutorial Example**: A user reported a **ModuleNotFoundError** related to 'triton.tools.experimental_descriptor' while attempting to run the tutorial example '06-fused-attention.py'. They suggested that this component may be deprecated, affecting newcomers.
   - Another participant clarified that the component has been adapted for newer GPUs, implying that users may need to update their setups for compatibility.
- **TMA Impact on Triton Performance**: A user inquired whether there were performance benchmarks for Triton with and without **TMA**, expressing concerns about current inefficiencies. It was noted that without autotuning, TMA might not deliver expected improvements over traditional methods.
   - Participants speculated that while TMA isn't showing significant gains yet, it could become impactful with future updates, particularly with **warp specialization**.



**Link mentioned**: <a href="https://pastebin.com/t2w4NYbP">import torchimport tritonimport triton.language as tldef benchmark(f, jo - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1336073240132718663)** (17 messages🔥): 

> `Cache Efficiency with Large Inputs, Tensor Cores Impact on Performance, Microarchitecture Confusions, SASS Observations on Blackwell, CUDA Stream Dependencies` 


- **Cache struggles with large inputs during streaming**: *If you have an input that's larger than your L2 and are streaming*, then **cache is entirely useless**; doing operations like A+B+C continuously trashes the cache.
   - Members in the discussion also noted that this issue arises even within a single stream.
- **Tensor Cores may impact non-critical FP operations**: A member raised concerns that the increased use of **integer operations** in kernels leveraging tensor cores might overwhelm resources, impacting **FP operations** not critical to performance.
   - Another member responded that if you’re limited by FMAs, you won't exceed **Hopper** performance, and the INT/FP distinction is less relevant.
- **Confusion around instruction fetching in microarchitecture**: A member clarified that some phrases used in discussions likely meant to convey that **only one instruction** can be fetched and decoded per clock cycle.
   - Confusion arose regarding whether operations could be issued simultaneously due to the implementation of more cores.
- **Observations on SASS for Blackwell architecture**: The SASS for **sm100/120** architecture appears similar to **90/Hopper**, with some additional details indicative of read/write barriers.
   - This observation led to speculation about potential changes in high latency instruction dependencies specific to consumer **Blackwell**.
- **Clarification on memory fences and stream dependencies**: A member questioned how **CUDA stream dependencies** influence memory fence behavior, expecting interaction between the two.
   - Another member assertively stated that **stream dependencies do not affect** memory fence behavior.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1336107368349437962)** (27 messages🔥): 

> `Redis Connection Management, Custom Triton/CUDA Kernels, Compiler Subprocesses, Redis Python Client` 


- **Understanding Redis Connection Lifecycle**: A discussion emerged around Redis connections, noting that each read/write creates a new connection, and there is no explicit closing of connections in the Python implementation as **redis uses a threadpool**.
   - Members inquired about tuning Redis's connection behaviors and confirmed they are using the [redis-py GitHub client](https://github.com/redis/redis-py) for implementation.
- **Navigating Graph Breaks with Triton/CUDA**: Members discussed the potential for avoiding graph breaks with custom Triton/CUDA kernels during torch compilation, indicating that **Triton should work universally** while CUDA requires defining kernels as custom ops.
   - However, there was uncertainty regarding the fusion of ops into user-defined Triton kernels, with a reference to an [issue on GitHub](https://github.com/pytorch/pytorch/issues/136227) for further details.
- **Threadpool Queries for Compiler Subprocesses**: Concerns were raised regarding whether there is a Redis threadpool associated with each compiler subprocess because Torch generates numerous subprocesses during compilation.
   - This prompted further investigation on how Redis connections are managed in the context of multiple subprocesses as users compile requests.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://redis-py.readthedocs.io/en/stable/connections.html">Connecting to Redis - redis-py dev documentation</a>: no description found</li><li><a href="https://github.com/redis/redis-py">GitHub - redis/redis-py: Redis Python client</a>: Redis Python client. Contribute to redis/redis-py development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/issues/136227">[Inductor] Some mechanism to fuse pointwise ops (and more) into a user-defined triton kernel · Issue #136227 · pytorch/pytorch</a>: We had an interesting use case that looks like the following: user initially had a model that used PyTorch operations. then, they switched it to user-defined triton kernels for the forward and back...</li><li><a href="https://github.com/pytorch/pytorch/issues/146414">MX basic dtypes in pytorch/pytorch · Issue #146414 · pytorch/pytorch</a>: 🚀 The feature, motivation and pitch Overview The Open Compute Project introduced the MicroScaling formats (MX) in Sep 2023, defining block-scaled dtypes with E8M0 block scales and FP8|FP6|FP4|INT8 .....</li><li><a href="https://github.com/pytorch/pytorch/blob/3aeccf2a2852a609a83cb2a529a1e5aba317b5fd/torch/_inductor/remote_cache.py#L290">pytorch/torch/_inductor/remote_cache.py at 3aeccf2a2852a609a83cb2a529a1e5aba317b5fd · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/3aeccf2a2852a609a83cb2a529a1e5aba317b5fd/torch/_inductor/remote_cache.py#L237">pytorch/torch/_inductor/remote_cache.py at 3aeccf2a2852a609a83cb2a529a1e5aba317b5fd · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1336204515967369357)** (4 messages): 

> `FP8 Attention Performance, Sensitivity of Attention to Quantization, Long Sequence Length Inference, Flash Attention 3, Quantization in DeepSeek V3` 


- **FP8 Attention could degrade output quality**: A user noted that when using the **Flash Attention 3 FP8 kernel**, their diffusion transformer model's inference speed increased, but the output quality significantly degraded.
   - This raises questions about the practical use of FP8 attention, as **no papers** have reported its application in this context.
- **Quantization's impact on attention explained**: Another member hypothesized that the output differences between **FP32** and **FP8** (about **1e-5** on average) accumulate during softmax, affecting attention distributions in long contexts.
   - They referred to an example from NVIDIA's documentation that discusses these subtle differences and their potential impacts.
- **Output differences attributed to linear layers?**: A discussion ensued regarding whether the **1e-5 output differences** reported refer to results from a linear layer.
   - This highlights a complexity in understanding how quantization impacts different components of transformer models.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

iron_bound: https://www.youtube.com/watch?v=rCwgAGG2sZQ
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1336118124218683477)** (1 messages): 

> `Staff Software Engineer, Model performance, Inference engine, Performance monitoring, Generative media models` 


- **Fal Seeks Staff Software Engineer for ML Performance**: Fal is hiring a **Staff Software Engineer** focused on **ML Performance & Systems**, aiming to enhance model performance for generative media models. The role involves designing and implementing innovative model serving architectures on their in-house inference engine.
   - The engineer will also develop **performance monitoring** and profiling tools to pinpoint bottlenecks and optimize system resources.
- **Key Responsibilities Highlighted for New Position**: The role includes maintaining Fal's position on model performance, specifically for **generative media models**. Key tasks involve maximizing **throughput** while minimizing **latency** and resource usage.
   - The engineer will collaborate closely with the Applied ML team and customers to ensure workloads effectively utilize their accelerator.



**Link mentioned**: <a href="https://fal.ai/careers/staff-software-engineer-ml-performance-systems">Staff Software Engineer, ML Performance &amp; Systems</a>: Staff Software Engineer, ML Performance &amp; Systems

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1336097116325548073)** (6 messages): 

> `GPU mode lecture 16, Nvidia CUB resources, Kernel Tuner on GitHub, CUDA kernel optimization project, Fused attention tutorial issues` 


- **Request for GPU Mode Code**: A user inquired about the code to reproduce examples from **GPU mode lecture 16** on hands profiling, seeking assistance from the community.
   - This reflects a common desire among beginners for reproducible materials in complex topics.
- **Nvidia's Advanced Scan and Open Source Hopes**: A participant mentioned that **Nvidia** has resources for CUB highlighted in the **Advanced Scan lecture**, hoping for them to be open-sourced in the future.
   - They also shared a link to the [Kernel Tuner GitHub repository](https://github.com/KernelTuner/kernel_tuner) which provides related resources.
- **CUDA Kernel Optimization Insights**: Another user referenced their abandoned project exploring **CUDA kernel optimization** via a reinforcement learning approach, linking to their fork on GitHub.
   - They acknowledged that the basic ideas were established but noted there was still more work needed.
- **Fused Attention Tutorial Errors**: A beginner reported an error while trying to run the **'06-fused-attention.py'** tutorial, due to a **ModuleNotFoundError** linked to the deprecated `experimental_descriptor` module.
   - They expressed a wish for the developers to update the tutorial examples, emphasizing their importance for newcomers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/KernelTuner/kernel_tuner">GitHub - KernelTuner/kernel_tuner: Kernel Tuner</a>: Kernel Tuner. Contribute to KernelTuner/kernel_tuner development by creating an account on GitHub.</li><li><a href="https://github.com/WAT-ai/CUDA-kernel-optimization">GitHub - WAT-ai/CUDA-kernel-optimization: Optimizing CUDA kernels using a reinforcement learning approach</a>: Optimizing CUDA kernels using a reinforcement learning approach - WAT-ai/CUDA-kernel-optimization
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1336182880040128565)** (13 messages🔥): 

> `Cursor vs Github Copilot, Sapiens Model Conversion, Efficiency of Codebase Tools` 


- **Cursor outshines Github Copilot**: Users noted that the difference between **Cursor** and **Github's Copilot** is **night and day**, with Cursor being significantly better in performance and utility.
   - *Copilot*, in the experience shared, tends to slow down workflows and is less helpful overall, especially when using the free version.
- **Efficiency of Codebase Tools**: For **dense codebases**, relying solely on tools for accurate information can waste time, making **human judgment** more efficient.
   - Conversely, for **small to medium sized codebases**, tools like Cursor have proven to be the best, streamlining the coding process significantly.
- **Inquiry on Sapiens Model Conversion**: A user sought assistance in converting the **Sapiens model** to TFLite format, providing a link to its [details on Hugging Face](https://huggingface.co/facebook/sapiens-depth-0.3b).
   - The Sapiens model, developed by **Meta**, is a vision transformer pretrained on **300 million images** at high resolution and shows strong generalization capabilities.



**Link mentioned**: <a href="https://huggingface.co/facebook/sapiens-depth-0.3b">facebook/sapiens-depth-0.3b · Hugging Face</a>: no description found

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1336109423118258308)** (3 messages): 

> `DreamCoder Optimization, NVIDIA GTC Sessions` 


- **Join the DreamCoder Optimization Project**: A member expressed interest in forming a project team to **optimize DreamCoder** for **CUDA**, which involves program synthesis.
   - Another member encouraged them to *make a working group if peers are found*.
- **Explore NVIDIA GTC Sessions**: A member shared a link to the NVIDIA GTC session catalog for those interested in **content tailored** for AI, technical details, and business strategies.
   - The session encourages participants to **select their tailored content** as per their interests.



**Link mentioned**: <a href="https://www.nvidia.com/gtc/session-catalog/?regcode=no-ncid&ncid=no-ncid&tab.catalogallsessionstab=16566177511100015Kus&search=kernel%20fusion#/session/1728599648492001N7Sn">NVIDIA #GTC2025 Conference Session Catalog</a>: Experience GTC 2025 In-Person and Online March 17-21, San Jose.

  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1336420968780005486)** (1 messages): 

> `E2E AutoML Model Compression, Edge Deployment Optimization, GitHub Project: sconce` 


- **Introducing sconce for Model Compression**: A member shared their project, **sconce**, an E2E AutoML model compression package aimed at finding **pareto optimal parameters** for edge deployment optimization, available on [GitHub](https://github.com/satabios/sconce).
   - *More features and support are on the way*, inviting others to engage and rate the package.
- **Call to Action: Star the sconce Project**: The member encouraged others to **'smack the star'** on their project if they appreciated the work done.
   - This call aims to boost visibility and support for the ongoing development of the package.



**Link mentioned**: <a href="https://github.com/satabios/sconce">GitHub - satabios/sconce: E2E AutoML Model Compression Package</a>: E2E AutoML Model Compression Package. Contribute to satabios/sconce development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1336064069358190693)** (68 messages🔥🔥): 

> `Python RNG Stability, Kimi Multimodal Model, Sokoban Puzzle Solver, Deterministic Clue Generation, Reasoning Model Performance` 


- **Python RNG Stability Issues**: Discussed the instability of Python's RNG with different versions and platforms, particularly noting that *hash randomization* can affect deterministic behavior.
   - They suggested using environment variables like `PYTHONHASHSEED` to mitigate these issues when generating puzzles.
- **Exploring Kimi's Multimodal Claims**: A member inquired about discussions surrounding Kimi, a model claiming to be o1-level multimodal, referencing its RL training methods.
   - Another participant noted the significant difference of using length penalty in training but pointed out that model weights were not released.
- **New Sokoban Puzzle Solver Implementation**: A member shared an implementation of a Sokoban puzzle solver, expressing confidence that it could effectively address task 49 in the reasoning-gym dataset.
   - They noted both *chatgpt-r* and *DSR1* struggled with larger mazes, emphasizing the need for improved planning capabilities.
- **Deterministic Clue Generation Challenges**: The discussion highlighted challenges in generating deterministic clues across iterations, with some solutions proposed including sorting before shuffling.
   - One member expressed that despite there being duplicate clues in previous implementations, the current adjustments have shown stabilization.
- **Reasoning Models' Performance on Tasks**: Members observed that some reasoning models, like chatgpt, took significantly longer than usual to solve simple tasks, indicating potential limitations.
   - In particular, it took 451 seconds for a basic Sokoban task, which raised concerns about the models' efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program.">Disable hash randomization from within python program</a>: Starting with Python 3.3, the hashing algorithm is non-deterministically salted to avoid a certain kind of attack. This is nice for webservers but it&#x27;s a pain when trying to debug a program: Ever...</li><li><a href="https://github.com/Deep-Agent/R1-V">GitHub - Deep-Agent/R1-V: Witness the aha moment of VLM with less than $3.</a>: Witness the aha moment of VLM with less than $3. Contribute to Deep-Agent/R1-V development by creating an account on GitHub.</li><li><a href="https://github.com/xbandrade/sokoban-solver-generator">GitHub - xbandrade/sokoban-solver-generator: Sokoban puzzle generator and solver with BFS, A* and Dijkstra</a>: Sokoban puzzle generator and solver with BFS, A* and Dijkstra - xbandrade/sokoban-solver-generator</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/59/files">Make zebra puzzle clue order deterministic by andreaskoepf · Pull Request #59 · open-thought/reasoning-gym</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1336066195757076592)** (2 messages): 

> `Probability of Sampling Neural Networks, Interpretability of MLPs and GLUs, SVD and Adversarial Examples, Phase Transition in Training, Closed-Form Polynomial Approximations` 


- **Unbelievable Odds of Functional Language Models**: The chance of randomly guessing the weights of a fully functional language model is approximately **1 in 360 million**.
   - A method was developed to estimate this probability through random sampling in weight space, revealing insights into network complexity.
- **Making Sense of MLPs and GLUs**: Researchers are converting **MLPs** and **GLUs** to closed-form polynomials for easier interpretability, thereby leveraging SVD techniques.
   - This approach allows for direct inspection and helps visualize the properties of deep learning models.
- **SVD Uncovers Adversarial Structures**: The use of SVD on linearized MLPs can generate **adversarial examples** that resonate back to the original MLP, demonstrating that the approximations capture out-of-distribution behavior.
   - This highlights an efficient method for understanding the behaviors of complex networks.
- **Network Complexity's Phase Transition Experiment**: A phase transition was found during MNIST training where the network complexity shifts from simple to non-linear behavior between **500 and 1k** steps.
   - This finding underscores the evolving complexity of neural networks through the training process.
- **Closed-Form Polynomial Approximations to Enhance Interpretability**: The recent work enables the use of polynomial functions to approximate MLPs and GLUs without significant performance loss.
   - This technique allows researchers to visually interpret deep learning networks better using eigendecomposition of approximants.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/polyapprox">GitHub - EleutherAI/polyapprox: Closed-form polynomial approximations to neural networks</a>: Closed-form polynomial approximations to neural networks - EleutherAI/polyapprox</li><li><a href="https://arxiv.org/abs/2502.01032">Converting MLPs into Polynomials in Closed Form</a>: Recent work has shown that purely quadratic functions can replace MLPs in transformers with no significant loss in performance, while enabling new methods of interpretability based on linear algebra. ...</li><li><a href="https://x.com/norabelrose/status/1886834375565959507">Tweet from Nora Belrose (@norabelrose)</a>: MLPs and GLUs are hard to interpret, but they make up most transformer parameters.Linear and quadratic functions are easier to interpret.We show how to convert MLPs & GLUs into polynomials in closed f...</li><li><a href="https://github.com/EleutherAI/basin-volume">GitHub - EleutherAI/basin-volume: Precisely estimating the volume of basins in neural net parameter space corresponding to interpretable behaviors</a>: Precisely estimating the volume of basins in neural net parameter space corresponding to interpretable behaviors - EleutherAI/basin-volume</li><li><a href="https://arxiv.org/abs/2501.18812">Estimating the Probability of Sampling a Trained Neural Network at Random</a>: We present an algorithm for estimating the probability mass, under a Gaussian or uniform prior, of a region in neural network parameter space corresponding to a particular behavior, such as achieving ...</li><li><a href="https://x.com/norabelrose/status/1886504219919966320">Tweet from Nora Belrose (@norabelrose)</a>: What are the chances you&#39;d get a fully functional language model by randomly guessing the weights?We crunched the numbers and here&#39;s the answer:
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1336066949318185080)** (25 messages🔥): 

> `Mixture of Experts, Custom LLM Tool, LLM Evaluation Harness Issues, Inducing Reasoning in Post-training, NLP Novice Contributions` 


- **Explore Mixture of Experts through Visual Guides**: A member shared a [comprehensive overview](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) of Mixture of Experts (MoE), including over **50 visualizations** and a link to a **YouTube version** with animations.
   - There is also a dedicated reading group on MoE at EleutherAI, though its current activity level is uncertain.
- **Proposal for Custom LLM Assembly Tool**: A member proposed a **drag-and-drop tool** for assembling custom LLMs that allows users to visualize how different architectures and layers affect model behavior in real-time.
   - This idea was considered a fun side project, suggesting interest in practical implementations of LLM customization.
- **DeepSeek Models Get Poor Evaluation Scores**: A member reported poor scores while using the **llm evaluation harness** on deepseek distilled models, suspecting `<think>` tags to be the culprit.
   - They sought advice on how to verify this issue or ignore the tags during evaluation.
- **Exploring Reasoning in AI Models**: A member raised a thought-provoking question about querying base models on their inherent nature, asking if this could reveal the innate tendencies of AI.
   - This idea sparked curiosity about the nature of AI's responses to existential questions.
- **Seeking Contributions in NLP and AI**: Several new members introduced themselves, expressing enthusiasm and experience in related fields such as **NLP, generative AI**, and uncertainty in AI models.
   - They were looking for opportunities to contribute to ongoing projects and discussions within the community.



**Link mentioned**: <a href="https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts">A Visual Guide to Mixture of Experts (MoE)</a>: Demystifying the role of MoE in Large Language Models

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1336063727111372870)** (59 messages🔥🔥): 

> `Harmonic Loss, Self-play Theorem Prover, Polynomial Transformers, Feature Addition in Models, Physical Intelligence Open Source` 


- **Harmonic Loss Shakes Up Training**: New research introduces **harmonic loss** as a more interpretable and faster-converging alternative to cross-entropy loss for training neural networks and LLMs, showcasing improved performance across various datasets.
   - The harmonic model outperformed standard models in generalization and interpretability, indicating significant benefits for future LLM training.
- **Self-play Theorem Prover (STP) Explained**: A novel approach, the **Self-play Theorem Prover**, alternates between conjecturing and proving to enhance training efficiency despite limited high-quality data for formal theorem proving.
   - This method addresses the challenge of sparse rewards by generating new conjectures based on previous model outputs, akin to methods seen in traditional mathematical progression.
- **Exploring Polynomial Transformers**: Discussing the potential of **polynomial (quadratic) transformers**, members speculate on replacing MLPs to enhance model efficiency, particularly in attention mechanisms.
   - Comparisons are made regarding classic models versus bilinear approaches, noting the trade-offs in parameter efficiency and complexity at scale.
- **Adding Features Without Losing Old Knowledge**: In discussions about upcycling machine learning models, it’s suggested that integrating new features can maintain existing knowledge if handled properly through specific initialization techniques.
   - Methods such as orthogonal finetuning are mentioned as ways to effectively combine multimodal features without significant loss of previously learned information.
- **Physical Intelligence's Open Source Initiative**: The **Physical Intelligence** team has open-sourced their work and models, highlighting advances in AI's application to physical tasks, moving beyond traditional AI benchmarks.
   - The community shows interest in experimenting with this new release, particularly noting the use of JAX in their implementation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.00212">Beyond Limited Data: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving</a>: A fundamental challenge in formal theorem proving by LLMs is the lack of high-quality training data. Although reinforcement learning or expert iteration partially mitigates this issue by alternating b...</li><li><a href="https://arxiv.org/abs/1602.02068">From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification</a>: We propose sparsemax, a new activation function similar to the traditional softmax, but able to output sparse probabilities. After deriving its properties, we show how its Jacobian can be efficiently ...</li><li><a href="https://arxiv.org/abs/2502.00873">Language Models Use Trigonometry to Do Addition</a>: Mathematical reasoning is an increasingly important indicator of large language model (LLM) capabilities, yet we lack understanding of how LLMs process even simple mathematical tasks. To address this,...</li><li><a href="https://arxiv.org/abs/2502.01628">Harmonic Loss Trains Interpretable AI Models</a>: In this paper, we introduce **harmonic loss** as an alternative to the standard cross-entropy loss for training neural networks and large language models (LLMs). Harmonic loss enables improved interpr...</li><li><a href="https://www.desmos.com/calculator/kpuu8besbg">Desmos | Graphing Calculator</a>: no description found</li><li><a href="https://manifestai.com/articles/symmetric-power-transformers/">Symmetric Power Transformers - Manifest AI</a>: A linear transformer that learns like a regular transformer with a state that fits on a GPU.</li><li><a href="https://www.physicalintelligence.company/blog/pi0">Our First Generalist Policy</a>: Physical Intelligence is bringing general-purpose AI into the physical world.</li><li><a href="https://fixupx.com/ChrSzegedy/status/1886881600367161679">Tweet from Christian Szegedy (@ChrSzegedy)</a>: I&#39;d be really interested in seeing how harmonically weighted attention would work. There might be a real potential there.Quoting David D. Baek (@dbaek__) 1/9 🚨 New Paper Alert: Cross-Entropy Loss...</li><li><a href="https://github.com/Physical-Intelligence/openpi">GitHub - Physical-Intelligence/openpi</a>: Contribute to Physical-Intelligence/openpi development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1336333753047449640)** (5 messages): 

> `Tuned Lens library, Affine translator for Llama 3.2 1B, Training data for GPT-2` 


- **Exploring Tuned Lens Library for Translation**: A member shared their work on using the **Tuned Lens library** to train an affine translator for **Llama 3.2 1B**.
   - They inquired about the amount of training data used for **GPT-2**, showing an active interest in optimizing their setup.
- **Training Data Insight for GPT-2**: Another member responded, stating the data utilized for **GPT-2** is about **250 times 2^18**.
   - This input provided a numeric reference that could help in fine-tuning training parameters for the translator.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1336372588695523352)** (6 messages): 

> `Bucket size for zero, Training models on different A100 configurations, Activation Checkpointing vs GAS, Model Performance Metrics, Optimization strategies` 


- **Finding the ideal bucket size for zero based on model size**: A user inquired about the principled method for selecting bucket size for zero related to model size, noting they currently rely on comparable model configurations.
   - They later recalled they had previously asked a similar question, indicating a desire for improved understanding.
- **Comparison of tokens per second on different A100 clusters**: Another user questioned if training an identical model on 40GB versus 80GB A100 clusters could yield expected nearly 2x increases in tokens per second.
   - Despite a larger batch size, they observed only a **10-15%** increase, expressing concerns over performance optimization.
- **Low TPS metrics for a 1.4B model**: The user reported achieving close to **14K TPS** for a 1.4B model on 8x80GB A100s, considering this metric quite low.
   - They expressed uncertainty over what further optimizations could be implemented to improve this figure.
- **GAS vs. Activation Checkpointing for improved TPS**: Observations indicated that training without activation checkpointing while using GAS for increased effective batch size resulted in significantly higher TPS, **242K TPS** vs **202K TPS**.
   - While effective batch sizes differed, the user noted that smaller batch sizes combined with GAS proved to be faster, raising questions about monitoring HFU/MFU metrics.
- **Model configuration insights shared**: The user shared a GitHub configuration link for a baseline speed experiment, detailing the implementation of model parallel autoregressive transformers.
   - This shared configuration serves as a resource for others looking to improve model training setups.



**Link mentioned**: <a href="https://github.com/aflah02/gpt-neox/blob/olmo-support/configs/hubble/Speed_Exps/1_1B_Baseline_BS_48_Both_Fusion_GQA_KV_Heads_4.yml">gpt-neox/configs/hubble/Speed_Exps/1_1B_Baseline_BS_48_Both_Fusion_GQA_KV_Heads_4.yml at olmo-support · aflah02/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - aflah02/gpt-neox

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1336066332353101845)** (74 messages🔥🔥): 

> `DeepSeek and AI advancements, Open source recommendation systems, Reinforcement Learning in AI, Mistral funding, Peano axioms in reasoning` 


- **DeepSeek's Major Coup in AI**: A member shared insights on how [DeepSeek](https://www.youtube.com/watch?v=3DEPDN8oD0w) has shifted the landscape of AI for humanity, provoking critical commentary about Altman's stance in light of recent developments.
   - *Altman has been labeled a hypeman*, as many question the disparity between his words and actions regarding open-source initiatives.
- **Navigating Open Source Recommendation Systems**: New member Amith discussed his experience and highlighted **Gorse** as a well-known open-source recommendation system, while expressing that these systems still need time to mature.
   - Another member suggested looking into **ByteDance's** recommendation technologies, adding to the evolving discourse on available resources.
- **Exploring Reinforcement Learning's Potential**: Discussion around whether **Reinforcement Learning (RL)** could teach AI intrinsic values like curiosity sparked interest, with members pondering its capabilities for fostering knowledge synthesis.
   - Juahyori noted the complexities of sustaining learned behaviors in continual learning contexts, emphasizing the challenges faced by AI models in alignment.
- **Mistral's Funding Milestone**: Mistral's recent funding of **$500 million** was mentioned as a significant achievement that may impact the competitive landscape of AI.
   - Members conjectured whether this announcement could be seen as good news for Mistral, suggesting that the investment reflects confidence in its trajectory.
- **Understanding Peano Axioms in AI Reasoning**: A member referenced a document exploring how AI models prove foundational arithmetic concepts like **1+1=2** using the **Peano axioms**.
   - The discussion led to curiosity about the implications of these axioms in AI reasoning and whether **different cultures** might approach mathematics uniquely.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v">YouTube</a>: no description found</li><li><a href="https://x.com/tsarnick/status/1778524418593218837?s=46">Tweet from Tsarathustra (@tsarnick)</a>: Geoffrey Hinton says AI models have intuition, creativity and the ability to see analogies that people cannot see</li><li><a href="https://x.com/teknium1/status/1885592392658805237?s=46">Tweet from Teknium (e/λ) (@Teknium1)</a>: Some tests with another proto-hermes-reasoner, excerpt fromWhat is your purpose and what is the purpose of life?</li><li><a href="https://x.com/teknium1/status/1885565179314004429?s=46">Tweet from Teknium (e/λ) (@Teknium1)</a>: Working on a classifier for refusals so you can automatically resample or resample + inject “sure” etc prefills to your llm calls if the models refusing. Unfortunately fasttext isnt good enough so tra...</li><li><a href="https://github.com/relign-ai/relign">GitHub - relign-ai/relign: post train language models on multi-step reasoning with reinforcement learning</a>: post train language models on multi-step reasoning with reinforcement learning - relign-ai/relign</li><li><a href="https://www.youtube.com/watch?v=3DEPDN8oD0w">Sam Altman: OpenAI has been on the &#39;wrong side of history&#39; post-DeepSeek</a>: CNBC&#39;s Deirdre Bosa reports on the latest developments from OpenAI.</li><li><a href="https://www.youtube.com/watch?v=_1f-o0nqpEI">DeepSeek, China, OpenAI, NVIDIA, xAI, TSMC, Stargate, and AI Megaclusters | Lex Fridman Podcast #459</a>: Dylan Patel is the founder of SemiAnalysis, a research &amp; analysis company specializing in semiconductors, GPUs, CPUs, and AI hardware. Nathan Lambert is a re...</li><li><a href="https://digital-strategy.ec.europa.eu/en/news/pioneering-ai-project-awarded-opening-large-language-models-european-languages">A pioneering AI project awarded for opening Large Language Models to European languages</a>: The Commission has awarded the prestigious Strategic Technologies for Europe Platform (STEP) Seal to the multilingual AI project OpenEuroLLM – the first Digital Europe Programme funded project to rece...</li><li><a href="https://x.com/i/grok/share/eUQwpP7nfyRAatWTzGbOQuotX">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1336127678830608414)** (11 messages🔥): 

> `Git Repositories and Modular Systems, Model Performance Assessment, Retrieval-Augmented Generation (RAG), Web Scraping Policies` 


- **Coming Soon: Git Repo with Modular CoT System**: A member is working on a project with Alignment Lab AI and has recently added a **modular CoT system** to their setup, suggesting impressive reasoning capabilities of **Hermes 3B**.
   - *
- **Debate Over Deepseek-R1-Distill-Qwen-14b's Quality**: Concerns were raised about whether **Deepseek-R1-Distill-Qwen-14b** is truly effective, noting that many models outperform it on the OpenLLM leaderboard.
   - Some members believe **Phi-4** may deliver better results with *less output tokens*, emphasizing use case relevance.
- **Building RAG Systems with Personal Data**: A member expressed interest in constructing a **RAG system** to digest personal books and notes to retrieve essential insights.
   - Another user shared an existing implementation called [local-rag](https://github.com/jonfairbanks/local-rag), which operates without sensitive data leaving the user's network.
- **Web Scraping Insights and Policies**: A discussion highlighted Brave's unique stance on web scraping, contrasting it with Google's policy approach regarding LLMs.
   - Web scraping was defined as the process of collecting public information from the web, primarily using bots or web crawlers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://brave.com/glossary/web-scraping/">Web Scraping Meaning &amp; Definition | Brave</a>: Online privacy can be confusing. In this easy-to-read list, you&#39;ll find short definitions of essential privacy and Internet terms including Web Scraping. Check out the Brave Privacy Glossary here.</li><li><a href="https://github.com/jonfairbanks/local-rag">GitHub - jonfairbanks/local-rag: Ingest files for retrieval augmented generation (RAG) with open-source Large Language Models (LLMs), all without 3rd parties or sensitive data leaving your network.</a>: Ingest files for retrieval augmented generation (RAG) with open-source Large Language Models (LLMs), all without 3rd parties or sensitive data leaving your network. - jonfairbanks/local-rag</li><li><a href="https://github.com/jonfairbanks/local-rag?tab=readme-ov-file)">GitHub - jonfairbanks/local-rag: Ingest files for retrieval augmented generation (RAG) with open-source Large Language Models (LLMs), all without 3rd parties or sensitive data leaving your network.</a>: Ingest files for retrieval augmented generation (RAG) with open-source Large Language Models (LLMs), all without 3rd parties or sensitive data leaving your network. - jonfairbanks/local-rag
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1336085913616519288)** (4 messages): 

> `Society Library Mission, Political AI Agent, SWE Arena Vibe Coding` 


- **Society Library's Ambitious Vision**: The Society Library extracts arguments and claims from various media to create a comprehensive database on high-impact, polarizing issues, aiming to visualize these ideas for public access.
   - Their long-term vision is to archive diverse ideas spanning social, political, philosophical, and spiritual realms for future generations.
- **Launch of a Political AI Agent**: The Society Library has introduced a Political AI agent as part of their nonprofit mission to enhance digital democracy, with technical infrastructure now in place.
   - This AI agent aims to facilitate representation in digital debates by serving as an educational intermediary chatbot.
- **SWE Arena Revolutionizes Vibe Coding**: SWE Arena is an open evaluation platform that supports executing any program in real-time, allowing users to accurately compare coding capabilities across multiple AI models.
   - With features like system prompt customization and code editing, it aligns with the Vibe Coding paradigm, focusing on the AI-generated results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.aipolitician.com/">AI Politician</a>: no description found</li><li><a href="https://www.societylibrary.org/mission-vision">&gt; Mission &amp; Vision &mdash; The Society Library</a>: no description found</li><li><a href="http://swe-arena.com/">SWE Arena: Compare &amp; Test Best AI Chatbots for Code</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1336100968328069273)** (5 messages): 

> `Community Contributions, Project Documentation, GitHub Assistance` 


- **New Member Seeking Contribution Opportunities**: A new member expressed their interest in contributing to projects on GitHub, highlighting their background in **full stack engineering**, **statistics**, and **R&D**.
   - *They are looking for a deeper dive into the project and possible connections for collaboration.*
- **Direct Message Follow-Up**: Another member prompted to check their direct messages in response to the new contributor's query.
   - *This indication suggests potential private discussions about contribution opportunities.*
- **Community Guidance Shared**: A community member directed the new contributor to a specific section for additional information on contributions, referencing a channel ID.
   - *This response demonstrates a proactive approach to helping newcomers integrate into the community.*


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1336066523478884513)** (79 messages🔥🔥): 

> `Image to Video Tools, Stable Diffusion Issues, Non-NSFW Image Requests, Model Performance Concerns, Editing Specific Characters in Images` 


- **Searching for Image to Video Tools**: A user inquired about the best software for converting images to video, mentioning potential limitations with online versions blocking NSFW content.
   - Another user suggested exploring **LTX** as a possible solution, indicating it might overcome some of these limitations.
- **Stable Diffusion Experiences Deteriorating Quality**: One user reported frustration with Stable Diffusion entering a 'rut' where it produces poor-quality images, particularly with repeated unintentional features like double bodies.
   - They questioned if there are ways to reset or clear any caches in order to restore expected image quality without restarting the software.
- **Non-NSFW Picture Requests for a Birthday**: A user requested assistance in creating a non-NSFW birthday greeting image using Stable Diffusion, expressing difficulty with the models used.
   - They specifically mentioned wanting a cute **smurfette** image and noted issues with copyright when using **DALL-E**.
- **Concerns About Model Performance and Censorship**: Users discussed the performance of various models, including **Stable Diffusion 3.5** and **Base XL**, with mixed opinions on their censorship and effectiveness.
   - Mention was made that fine-tuning a model may reduce censorship, although some users had different experiences regarding the same.
- **Editing Specific Characters in Images Using Prompts**: A user sought advice on how to edit individual characters in a multi-person image within **A1111**, specifically using prompts.
   - Although some techniques like **inpainting** were mentioned, the user was looking for a more precise method to differentiate traits like hair color between characters.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://civitai.com/articles/1997/comfyui-guide-to-stacker-nodes">ComfyUI - Guide to Stacker Nodes | Civitai</a>: This article is about Stacker Nodes and how to use them in workflows. It is intended for both new and advanced users of ComfyUI. Stacker nodes are ...</li><li><a href="https://www.youtube.com/watch?v=kZRE7HIO3vk">The Thirty Million Line Problem</a>: A historical argument for creating a stable instruction set architecture (ISA) for entire system-on-a-chip (SoC) packages.Please note that although it&#39;s neve...
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1336079037012901980)** (16 messages🔥): 

> `Podcast Announcement, Using NotebookLM in Legal Practice, Customer Service Use Cases, Lip Sync Technology, Overcoming Glossophobia` 


- **Big Announcement for 'Roast or Toast' Podcast**: The grumpiest AI toaster in the universe introduces a podcast called [‘Roast or Toast’](https://youtu.be/mormPD6QYkQ?feature=shared), where it explores the meaning of life, combining celebration and critique.
   - *Tune in for the premiere episode* to see if The Toastinator toasts or roasts the grand mystery of existence.
- **Lawyers Use NotebookLM for Efficiency**: A lawyer in Brazil shares that they find it incredible to use NotebookLM for drafting legal documents and studying cases, as it provides source citations for reliability.
   - They utilize the tool to adapt templates for repetitive legal documents, making the process efficient and tailored to specific cases.
- **NotebookLM in Customer Service**: A user is seeking insights into how NotebookLM is employed in customer service environments or BPOs, asking for shared experiences and use cases.
   - Identified potential use cases include shortening agent training time and generating client profiles for meetings.
- **Lip Syncing Characters in 'Spudcast'**: A novice is exploring lip syncing two characters from their created 'spudcast', discussing options and seeking insights on the process.
   - The success of lip sync technology depends on the software's quality and resolution, with CapCut Pro being identified as a currently usable option.
- **Hypnotic Audio Script for Glossophobia**: A member shares a 14-minute hypnotic audio script tailored for overcoming the fear of public speaking, designed for deep relaxation and confidence-building.
   - The script, along with a sound file, is aimed at utilizing a gradual exposure technique through peaceful visualization.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/34btQzZfICY?si=6dUniE26K0pOr3__">AI: A Hybrid Approach to Human-like Consciousness</a>: Exploring the intersection of artificial intelligence (AI), consciousness, and symbolic systems, particularly the Cybee&#39;s Psychic Celtic Cross (CPCC) tarot s...</li><li><a href="https://youtu.be/mormPD6QYkQ?feature=shared">&quot;Roast or Toast&quot; Podcast by &quot;The Toastinator&quot; Lipsynced</a>: 🔥🎙️ BIG ANNOUNCEMENT! 🎙️🔥The grumpiest AI toaster in the universe has something to say! 🤖🔥 Say hello to &quot;Roast or Toast&quot;, the podcast where we either c...
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1336094302106812527)** (38 messages🔥): 

> `Customization of Audio Overviews, Podcast Feature Enhancements, Google Account Issues, NotebookLM in Google WorkSpace, Limits in Free Version` 


- **Customization of Audio Overviews is Confusing**: A new user sought ways to customize audio overviews in NotebookLM, but faced difficulties as the expected options were missing.
   - Others suggested checking for the 'Customize' button under the studio bar, but further troubleshooting was necessary when it wasn't visible.
- **Future Enhancements for Podcast Features**: Discussions emerged about increasing customization options for podcasts, including voices and personalities, although no specific plans were confirmed.
   - Some users expressed hope that these features would eventually be rolled out as they enhance interactivity.
- **Clarification on Google Account Limitations**: A user reported a normal Google account was disabled for NotebookLM, with potential age verification issues discussed as a cause.
   - Another user confirmed the importance of understanding account settings and permissions to troubleshoot similar issues.
- **Managing Access in Google WorkSpace**: Queries were made regarding the enabling of NotebookLM Plus in Google WorkSpace for specific groups rather than everyone.
   - Instructions were provided on how to manage access through the Google Admin console, highlighting the role of organizational units.
- **Limits in Free NotebookLM Version**: A question was raised about whether the free version of NotebookLM imposes limits on queries, with anecdotal evidence suggesting high limits.
   - Users indicated they had not encountered any significant restrictions while using the platform.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://illuminate.google.com/home">Illuminate | Learn Your Way</a>: Transform research papers into AI-generated audio summaries with Illuminate, your Gen AI tool for understanding complex content faster.</li><li><a href="https://support.google.com/a/answer/181865#zippy=%2Cturn-services-on-or-off-for-users">Turn on or off additional Google services - Google Workspace Admin Help</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1336092085530398903)** (51 messages🔥): 

> `Claude Constitutional Classifiers, FAIR internal dynamics, AI Admaker Icon, How to Scale Your Model, Pi0 Vision Language Action Model` 


- **Claude Constitutional Classifiers challenge**: Anthropic launched their [Claude Constitutional Classifiers](https://claude.ai/constitutional-classifiers) inviting users to break their new jailbreaking defense across 8 levels.
   - They are preparing for the arrival of powerful AI systems and have developed a demo app to test new safety techniques.
- **FAIR internal dynamics surrounding Zetta and Llama**: Discussions revealed the complexities of the internal dynamics at FAIR, particularly related to the development of models Zetta and Llama, sparking debate over transparency and competitive practices.
   - Yann LeCun and others highlighted how a small team innovated beyond larger projects, suggesting a deeper investigation into the organizational culture is necessary.
- **AI Admaker Icon unveiled**: Icon, described as a combination of ChatGPT and CapCut, aims to help brands automate ad creation processes significantly, producing 300 ads a month versus the usual 30.
   - Backed by notable investors, it integrates video tagging, script generation, and editing tools to improve ad quality while reducing expenses.
- **Textbook on scaling models released**: A new textbook titled ["How To Scale Your Model"](https://jax-ml.github.io/scaling-book/) by Google DeepMind demystifies the systems view of LLMs, focusing on mathematical approaches.
   - It emphasizes the ability to understand model performance through simple equations, enhancing the efficiency of running large models.
- **Pi0 Vision Language Action Model released**: The team behind Physical Intelligence announced the release of the advanced Pi0 model, which utilizes natural language commands to perform autonomous actions.
   - This model is now available on LeRobotHF, along with pre-trained checkpoints and code for fine-tuning on various robotic tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://omnihuman-lab.github.io">OmniHuman-1 Project</a>: no description found</li><li><a href="https://claude.ai/constitutional-classifiers">Claude</a>: Talk with Claude, an AI assistant from Anthropic</li><li><a href="https://x.com/namangoyal21/status/1886515845133951192?s=46">Tweet from Naman Goyal (@NamanGoyal21)</a>: @giffmana Being the only person who was co-author both in OPT and llama1 and was part of zetta team, I can say that actually that it was much more nuanced and has multiple POVs and not a simple story ...</li><li><a href="https://www.gradient.com/blog/posts/centml-deepseek/">DeepSeek R1 on CentML | Gradient Ventures</a>: no description found</li><li><a href="https://x.com/armenagha/status/1886522896077439187?s=46">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: This is absolutely not true about what happened with Zetta. Do we really want to open up about what happened here?Quoting Yann LeCun (@ylecun) You misread.There had been multiple LLM projects within F...</li><li><a href="https://x.com/suchenzang/status/1886635726655430786?s=46">Tweet from Susan Zhang (@suchenzang)</a>: i wonder what a PIP looks like for a chief AI scientist</li><li><a href="https://x.com/suchenzang/status/1886788787633787213">Tweet from Susan Zhang (@suchenzang)</a>: in case this wasn&#39;t clear:being a small talented teamandlacking integrity (juice numbers from training on test)are not mutually exclusiveyet this is what a chief ai scientist / turing award winner...</li><li><a href="https://x.com/soumithchintala/status/1886562033048396241?s=46">Tweet from Soumith Chintala (@soumithchintala)</a>: you were/are the Chief Scientist of Meta, and a FAIR Lead -- where both Zetta and Llama were located; I think characterizing any team within your direct influence in a bad light in public is not nice....</li><li><a href="https://x.com/jacobaustin132/status/1886844716446007300?s=46">Tweet from Jacob Austin (@jacobaustin132)</a>: Making LLMs run efficiently can feel scary, but scaling isn’t magic, it’s math! We wanted to demystify the “systems view” of LLMs and wrote a little textbook called “How To Scale Your Model” which we’...</li><li><a href="https://x.com/physical_int/status/1886822689157079077">Tweet from Physical Intelligence (@physical_int)</a>: Many of you asked for code & weights for π₀, we are happy to announce that we are releasing  π₀ and pre-trained checkpoints in our new openpi repository! We tested the model on a few public robots, an...</li><li><a href="https://x.com/alexalbert__/status/1886461372223074412?s=46">Tweet from Alex Albert (@alexalbert__)</a>: At Anthropic, we&#39;re preparing for the arrival of powerful AI systems. Based on our latest research on Constitutional Classifiers, we&#39;ve developed a demo app to test new safety techniques.We wa...</li><li><a href="https://x.com/RemiCadene/status/1886823939856589296">Tweet from Remi Cadene (@RemiCadene)</a>: ⭐ The first foundational model available on @LeRobotHF ⭐Pi0 is the most advanced Vision Language Action model. It takes natural language commands as input and directly output autonomous behavior.It wa...</li><li><a href="https://x.com/harambe_musk/status/1886779961790345657?s=46">Tweet from harambe_musk🍌 (@harambe_musk)</a>: OpenAI planning to release SWE agent by end of Q1 or mid Q2 powered by o3 and o3 pro for enterprises. This is expected to shakeup the software industry as it’s apparently smart enough to compete with ...</li><li><a href="https://x.com/suchenzang/status/1886611967692943856?s=46">Tweet from Susan Zhang (@suchenzang)</a>: since a godfather of AI is looking at codebase receipts, here one:https://github.com/facebookresearch/metaseq/tree/main/preprocessing/books3inhttps://arxiv.org/abs/2302.13971Quoting Armen Aghajanyan (...</li><li><a href="https://x.com/armenagha/status/1886549536300261706?s=46">Tweet from Armen Aghajanyan (@ArmenAgha)</a>: Only one of the teams between Zetta/LLaMa had an open-source pre-training codebase, shared datasets and experiments internally, used standardized evaluation sets, published internal notes and did thin...</li><li><a href="https://x.com/atroyn/status/1885735818964447700">Tweet from anton (@atroyn)</a>: on the occasion of my 37th birthday, i’m announcing my departure from chroma.</li><li><a href="https://x.com/suchenzang/status/1886544793511080103?s=46">Tweet from Susan Zhang (@suchenzang)</a>: if you&#39;re bragging about internal cultural dysfunction in your own lab, where ICs are pitted against each other in the name of some regional glory...it&#39;s no wonder your models are utterly irre...</li><li><a href="https://x.com/jeffdean/status/1886852442815652188?s=46">Tweet from Jeff Dean (@JeffDean)</a>: Training our most capable Gemini models relies heavily on our JAX software stack + Google&#39;s TPU hardware platforms. If you want to learn more, see this awesome book &#34;How to Scale Your Model&#3...</li><li><a href="https://x.com/_sholtodouglas/status/1886855383496712215?s=46">Tweet from Sholto Douglas (@_sholtodouglas)</a>: A distillation of our mental models that we use to think about the systems perspective on training and inference at scale. The most important takeaway - you should be able to describe everything about...</li><li><a href="https://x.com/janleike/status/1886452697425137904">Tweet from Jan Leike (@janleike)</a>: We challenge you to break our new jailbreaking defense!There are 8 levels. Can you find a single jailbreak to beat them all?https://claude.ai/constitutional-classifiers</li><li><a href="https://x.com/kennandavison/status/1886836061378372064">Tweet from Kennan Davison (@kennandavison)</a>: Excited to introduce Icon, The First AI Admaker.We’re backed by Peter Thiel’s Founders Fund & execs of frontier AI labs like OpenAI, Pika, & Cognition.Icon (http://icon.me) is like ChatGPT + CapCut, b...
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1336080187670003733)** (49 messages🔥): 

> `MathJax for LaTeX support, DeepSeek with localdocs errors, EU AI Act implications, Discussion on the EU, AI-driven communication` 


- **MathJax Capabilities for LaTeX**: Members discussed using [MathJax](https://www.mathjax.org/) for LaTeX support, with one suggestion highlighting the need for its SVG export capabilities.
   - There was a consideration for parsing and applying MathJax to parts of documents that contain LaTeX notation.
- **Ongoing Issues with DeepSeek**: Users reported issues with DeepSeek returning errors when using it with localdocs, specifically the error 'item at index 3 is not a prompt'.
   - Some users mentioned that certain model versions were performing better, while awaiting a fix that had been indicated as being in the main branch.
- **Concerns about the EU AI Act**: A discussion arose about the implications of the EU's new AI Act, particularly how it regulates AI use, including prohibitions on certain applications.
   - Members shared links to sources for more information, indicating that while the rules are not yet law, they pose significant implications for AI usage in the EU.
- **Debate on EU's Role**: A lively debate unfolded regarding the EU's position and actions in global politics, particularly around themes of imperialism and human rights.
   - Participants exchanged sharp critiques, suggesting a mix of emotional responses and logical fallacies in the conversation about EU policies and actions.
- **AI Communication Dynamics**: The interaction among users highlighted the challenges of maintaining a mature discussion about complex topics, such as democracy and governance.
   - Some users called for the conversation to focus on AI-related topics, emphasizing the importance of respectful dialogue and the impact of personal biases.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.mathjax.org/">MathJax</a>: Beautiful math in all browsers.</li><li><a href="https://pangea.stanford.edu/computing/unix/formatting/symbols.php">LaTeX Typesetting</a>: no description found</li><li><a href="https://digital-strategy.ec.europa.eu/en/news/first-rules-artificial-intelligence-act-are-now-applicable">First rules of the Artificial Intelligence Act are now applicable</a>: As of Sunday, 2 February, the first rules under the Artificial Intelligence Act (AI Act) started to apply.</li><li><a href="https://www.europarl.europa.eu/topics/en/article/20230601STO93804/eu-ai-act-first-regulation-on-artificial-intelligence">EU AI Act: first regulation on artificial intelligence | Topics | European Parliament</a>: The use of artificial intelligence in the EU will be regulated by the AI Act, the world’s first comprehensive AI law. Find out how it will protect you.</li><li><a href="https://eur-lex.europa.eu/eli/reg/2024/1689">Regulation - EU - 2024/1689 - EN - EUR-Lex</a>: no description found
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1336101844891598970)** (4 messages): 

> `User Story Management, User Tiers Updates, Bolt's Markdown Functionality, Guidelines Consistency` 


- **Tracking User Stories Efficiently**: A member inquired about the method of documenting user stories and updates, emphasizing the need for clarity and organization.
   - Another member suggested utilizing Bolt to create a Nextsteps.md file for better tracking and to update progress systematically.
- **Complex UI for User Groups in Development**: A user shared their initial attempts at using Zapier for updating user tiers based on subscription changes, but has yet to develop a more intricate UI.
   - They anticipate that the office hours on 2/12 will provide more insights on refining this process.
- **Utilizing Bolt for Consistent Output**: One member recommended using Bolt to read and retain guidelines regularly, despite the higher token cost, to ensure consistent outputs.
   - They noted that this approach could apply to both architecture and style guides for improved adherence.


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1336096035927359518)** (36 messages🔥): 

> `Supabase vs Firebase preferences, Bolt performance issues, Data persistence in Bolt, GDPR-compliant hosting alternatives, Edge functions and API key authentication` 


- **Supabase preferred for integration**: A member expressed a preference for **Supabase** due to its direct integration capabilities, while acknowledging the technical comfort some have with **Firebase**.
   - This sparked a discussion about individual use cases and needs regarding database services.
- **Bolt experiencing major performance issues**: Multiple users reported **Bolt** loading extremely slowly and encountering authentication errors, with issues continuing intermittently.
   - One user mentioned that a refresh temporarily resolved their problems but noted ongoing frustrations with changes not updating correctly.
- **Missing progress and backup troubles**: A user expressed concern about losing hours of work in **Bolt**, with the last available backup dating back to the start of January.
   - Another member suggested checking the backup settings to recover lost progress, though the provided backup was outdated.
- **Seeking GDPR-compliant hosting solutions**: A user inquired about the **GDPR-compliance** of **Netlify**, pointing out potential data processing issues within the EU.
   - They solicited recommendations for alternatives ensuring all hosting and data processing occurs within EU borders.
- **API key authentication troubleshooting**: A user faced difficulties with their **restful API** request in **Bolt** using **Supabase edge functions**, receiving a **401 Invalid JWT** error.
   - They were frustrated over the lack of invocations and responses from their edge functions, unsure of how to resolve the issue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.netlify.com/privacy/">Privacy Policy | Netlify</a>: Accelerate the time to deploy your websites and apps. Bring your integrations and APIs together on one powerful serverless platform. Get started for free!</li><li><a href="https://x.com/gopeekm/status/1886879621892755778">Tweet from Peekm (@gopeekm)</a>: The french Youtuber @snorkyfab_YT has posted a very detailed product review and rated the @elgato Wave:3 microphone! 🎙️His profile is open to requests for reviews of technology products and games! So...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/2985">Feature Request: Show .bolt folder in Bolt · Issue #2985 · stackblitz/bolt.new</a>: Is your feature request related to a problem? Please describe: Not a problem, just a minor annoyance. Describe the solution you&#39;d like: It would be nice if I could update things like the Bolt igno...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1336086182613745676)** (11 messages🔥): 

> `SFT Dataset Customization, Office Hour Details, Discord Channel for Event` 


- **SFT Dataset Customization Success**: A member successfully hijacked the built-in **SFT Dataset** by customizing the **message_transform** and **model_transform** parameters, allowing for format adjustments as needed.
   - *I just had to hijack the message/model transforms* to fit my requirements, which can easily be done via the API.
- **Office Hour Scheduled for Thursday**: The office hour is confirmed for **Thursday**, with details shared in the channel for easy access.
   - Members expressed enthusiasm, saying, *All good - looking forward to seeing you on Thursday!*
- **Discord Channel Creation for Event**: Plans were made to open a **Discord channel** specifically for the office hour event, ensuring members can join easily.
   - One member stated, *We'll open a channel in Discord for the event : ),* enhancing communication for attendees.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1336087962361925674)** (24 messages🔥): 

> `Seed handling in DPO recipes, Debugging DataLoader issues, Dataset influence on sampling, Gradient accumulation in DPO, Ladder-residual architecture modification` 


- **Seed Issue in DPO Recipes**: Members are troubleshooting why the `seed` works for lora/full finetune but not for lora/full DPO, causing different loss curves with the same config: seed 42.
   - Concerns were raised about `seed=0` and `seed=null` affecting randomness in DistributedSampler calls.
- **DataLoader Batches Confirmed Consistent**: A member confirmed that logging batches from the DataLoader indicates no issues there, as batches remain the same across runs.
   - Attention was drawn to the possibility of issues within the dataset class affecting the sampler's behavior.
- **Investigating Dataset Impact on DPO**: The standard paired Stack Exchange dataset is being used for DPO, with consideration given to how this might interfere with sampler logic.
   - Members discussed comparing configurations and recipes to identify any discrepancies impacting the behavior.
- **Gradient Accumulation Fix on DPO Observed**: A relevant issue was highlighted regarding gradient accumulation in DPO/PPO recipes that might be linked to the seed issue being faced.
   - This concern was tied back to other noted discrepancies affecting seed handling that could obstruct model performance.
- **Ladder-residual Modification for Performance**: [A tweet](https://x.com/zhang_muru/status/1886870194443968529) discussed the introduction of Ladder-residual, a modification improving the speed of 70B Llama under tensor parallelism by approximately 30%.
   - This work reflects ongoing optimizations in model architecture collaboration among several authors and researchers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/zhang_muru/status/1886870194443968529">Tweet from Muru Zhang (@zhang_muru)</a>: Running your model on multiple GPUs but often found the speed not satisfiable? We introduce Ladder-residual, a parallelism-aware architecture modification that makes 70B Llama with tensor parallelism ...</li><li><a href="https://github.com/pytorch/torchtune/issues/2334">Apply gradient accumulation fix to DPO/PPO recipes · Issue #2334 · pytorch/torchtune</a>: https://unsloth.ai/blog/gradient</li><li><a href="https://github.com/pytorch/torchtune/issues/2335">Seed is not applied for DPO recipes · Issue #2335 · pytorch/torchtune</a>: TL;DR Launching same config twice with seed: 42 results in two different loss curves Affected recipes full_dpo_distributed - seed is not set Full DPO is taken from #2275 lora_dpo_distributed - seed...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1336072009058095187)** (2 messages): 

> `Data Augmentation in LLMs, R1-V Project Introduction, Verifiable Rewards, General Counting Abilities in Models` 


- **Survey on Data Augmentation in LLMs**: A recent survey analyzes the role of **data augmentation** in **large language models** (LLMs), highlighting their need for extensive datasets to avoid overfitting.
   - It discusses **distinctive prompt templates** and **retrieval-based techniques** that enhance LLM capabilities through external knowledge, which can lead to more **grounded-truth data**.
- **R1-V Project Revolutionizes Learning Efficiency**: Exciting news was shared about the **R1-V** project that utilizes **reinforcement learning** with verifiable rewards to enhance models' counting abilities, showing a model that surpasses a **72B** counterpart with just **100 training steps**.
   - The undertaking promises to be fully **open source**, with development costs under **$3**, sparking interest in the community's upcoming announcements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.18845">Text Data Augmentation for Large Language Models: A Comprehensive Survey of Methods, Challenges, and Opportunities</a>: The increasing size and complexity of pre-trained language models have demonstrated superior performance in many applications, but they usually require large training datasets to be adequately trained...</li><li><a href="https://x.com/liangchen5518/status/1886171667522842856?s=46&t=b1X88nwMsmZgHkmMFkiG3g">Tweet from Liang Chen (@liangchen5518)</a>: Excited to introduce R1-V!We use RL with verifiable rewards to incentivize VLMs to learn general counting abilities. 2B model surpasses the 72B with only 100 training steps, costing less than $3.The p...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1336069484599054501)** (1 messages): 

> `Community Showcase, Forum Updates` 


- **Community Showcase Transitioned to Forum**: The **Community Showcase** has been officially moved to the [forum](https://forum.modular.com/c/community-showcase/8) for easier access and organization.
   - *This change marks a significant step in improving community interactions and sharing.*
- **Announcement of Read-Only Status**: It has been announced that the previous **community showcase** is now in a read-only state.
   - Members are encouraged to participate in discussions on the new forum platform.



**Link mentioned**: <a href="https://forum.modular.com/c/community-showcase/8">Community Showcase</a>: Community projects that use MAX and Mojo

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1336221351643713658)** (35 messages🔥): 

> `Hot Reloading in Rust, Mojo ABI Alternatives, Python's Asyncio APIs, Thread Safety in Asynchronous APIs, Memory Management in Futures` 


- **Hot Reloading Mechanisms Explored**: Members discussed how Rust typically achieves **hot reloading** using a **C ABI**, which can complicate interactions with Rust updates.
   - *Owen* inquired about resources for building a toy ABI, noting that ABI stability is crucial due to the frequent changes in data structures.
- **Mojo's ABI Capabilities**: A member asked if Mojo has a feature similar to Rust's `#[cfg(feature = "foo")]`, leading to a discussion about compile-time programming capabilities in Mojo.
   - It was noted that a stable ABI is important for compatibility, with references made to how only a few languages maintain such stability.
- **Evaluating Python's Asyncio Event Loop**: Discussion around Python's **asyncio** highlighted that it allows for community-driven event loops, with links provided to **uvloop** on GitHub.
   - Members contrasted this with Mojo's approach to threading and memory management, indicating potential hurdles.
- **Challenges of Thread Safety in Async APIs**: Concern arose regarding the **thread safety** of asynchronous APIs, as members highlighted potentially mutative qualities and the need for secure memory handling.
   - The conversation noted that many approaches currently don't allow control over memory allocation, which could lead to issues.
- **Memory Allocation Management in Futures**: Discussions showed that memory allocation for futures should ideally be managed by the **programmer**, allowing flexible strategies for performance.
   - A member expressed the goal of reusing allocations efficiently, while others acknowledged that implementing thread-safe futures could introduce complexity.



**Link mentioned**: <a href="https://github.com/MagicStack/uvloop">GitHub - MagicStack/uvloop: Ultra fast asyncio event loop.</a>: Ultra fast asyncio event loop. Contribute to MagicStack/uvloop development by creating an account on GitHub.

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1336087344167780414)** (1 messages): 

> `Lecture 2 with Jason Weston, Self-Improvement in LLMs, MOOC Curriculum Updates` 


- **Second Lecture featuring Jason Weston**: Join us for the second lecture today at **4:00pm PST** featuring guest speaker [Jason Weston](https://www.youtube.com/live/_MNlLhU33H0) discussing 'Learning to Self-Improve & Reason with LLMs'.
   - The lecture will cover innovative self-learning methods for LLMs relevant to diverse tasks, including reasoning and creative challenges.
- **Innovative self-learning methods for LLMs**: Jason Weston will describe recent methods for LLMs including [Iterative DPO](https://arxiv.org/abs/2312.16682), [Self-Rewarding LLMs](https://arxiv.org/abs/2401.10020), and [Thinking LLMs](https://arxiv.org/abs/2410.10630).
   - These methods focus on enhancing LLM performance through effective reasoning and task-related learning mechanisms.
- **Upcoming MOOC curriculum details**: Updates on the MOOC curriculum will be shared soon, keeping participants informed of the latest developments.
   - Thank you to everyone for your patience as we prepare to release this important information.



**Link mentioned**: <a href="https://www.youtube.com/live/_MNlLhU33H0.">CS 194/294-280 (Advanced LLM Agents) - Lecture 2, Jason Weston</a>: no description found

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1336074227081678898)** (28 messages🔥): 

> `Course Enrollment Confirmation, Hackathon Results Update, Certificate Release Inquiry, Quiz Deadlines, Research Project Participation` 


- **Course Enrollment Confirmations Galore!**: Multiple members confirmed their course enrollment after receiving the [Google Forms confirmation](https://link.to.form), indicating that they are all set to participate.
   - One member even expressed eagerness to join a research project, as updates on the MOOC curriculum are forthcoming.
- **Hackathon Results Sneak Peek!**: Winners of the hackathon have been notified privately, with a public announcement expected next week.
   - Members eagerly await further details as they check back on announcements.
- **Certificates Release Still Pending**: Several members inquired about the status of their **fall program certificates**, with the response being that **none have been released yet**.
   - Officials reassured participants that certificates will be available soon, thanking them for their patience.
- **Quiz 1 Submission Concerns Addressed**: Concerns arose regarding missed deadlines for Quiz 1, but members were reassured there are no strict deadlines yet as the MOOC curriculum details have not been fully released.
   - Members were encouraged to take Quiz 1 regardless and are eager for further information.
- **Participation in Research Projects**: Members expressed an interest in participating in research projects and asked how to get paired in teams.
   - The reply highlighted that more details about the research opportunities would be provided soon.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1336154668077813800)** (2 messages): 

> `Attendance form for Berkeley students` 


- **Clarification on Attendance Form**: A member inquired about the **attendance form** mentioned previously, as they could not find it for the last week.
   - Another member clarified that the attendance form is **only for Berkeley students**.
- **Attendance Form Accessibility**: Concerns were raised regarding the **accessibility** of the attendance form for non-Berkeley students.
   - Participants noted the lack of information available for others who might be interested in joining the program.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1336123139050045533)** (21 messages🔥): 

> `Legacy ERP Integration with VBS Scripts, Running MCP Server in Cursor, Enterprise MCP Protocol Progress, CORS Issues with Localhost on Windows, Using ngrok for Server Access` 


- **Seeking Legacy ERP Integration with VBS**: A user is looking for help with servers that call **.vbs scripts** for **legacy ERP integration**.
   - Another member mentioned they are adding server support to **mcpdotnet**, suggesting it might be easier to invoke from **.NET**.
- **MCP Server Setup in Cursor**: A new user sought guidance on how to run an MCP server locally in **Cursor**, specifically wanting to use a **Docker container**.
   - Members recommended entering the **SSE URL** used with **supergateway** into the Cursor MCP SSE settings.
- **Progress on Enterprise MCP Protocol**: Discussion about the **MCP protocol** revealed a draft for **OAuth 2.1** authorization, with potential integration with **IAM** systems.
   - It was noted that the SDKs currently do not support authorization as they are building internal test servers for prototyping.
- **CORS Problems with Localhost**: A user faced connection issues when running their MCP server on **localhost**, suggesting it could be **CORS-related**.
   - They plan to use **ngrok** to bypass potential communication issues associated with localhost access on **Windows**.
- **Using ngrok for Localhost Access**: One member suggested running **ngrok** to test server accessibility, using the command `ngrok http 8001`.
   - They emphasized that this might resolve any issues caused by trying to access the server via localhost.


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1336073687811625001)** (7 messages): 

> `Command-R+ model, Cohere support for Canadian AI, Bug with open-source copilot tool, Financial Semantic Search Webinar` 


- **Command-R+ impresses with consistent surprises**: A user expressed satisfaction with the **Command-R+** model, stating it continues to surprise them after months of usage, amidst hype over other models.
   - Another user inquired about the specific surprises, prompting a response about the model's ability to expose **internal thoughts** and **logical steps** akin to Chain of Thought.
- **Cohere as a Canadian AI solution**: One member highlighted their choice to use **Cohere** to support Canadian AI efforts, especially in light of potential US tariffs.
   - They expressed appreciation for options that help maintain local AI capabilities during challenging economic circumstances.
- **Bug affecting Command-R+ functionality**: A user reported a bug in an open-source copilot tool that prevents editing files with **Command-R+**, providing a GitHub issue for visibility.
   - They aimed to raise awareness of the problem, linking to the detailed bug report on GitHub.
- **Webinar on Financial Semantic Search**: A member shared highlights from a **webinar** on **Financial Semantic Search and Reranking** presented by Cohere's team and Pinecone.
   - They emphasized learning about the application of Cohere’s Rerank 3.5 model on financial data, which aims to boost overall search performance.



**Link mentioned**: <a href="https://github.com/continuedev/continue/issues/3881">Cohere AI - 400 Bad Request: provided raw prompt is invalid · Issue #3881 · continuedev/continue</a>: Before submitting your bug report I believe this is a bug. I&#39;ll try to join the Continue Discord for questions I&#39;m not able to find an open issue that reports the same bug I&#39;ve seen the tr...

  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1336310151061307433)** (5 messages): 

> `Cmd R Bot Interaction` 


- **Casual Check-in with Cmd R Bot**: User greeted Cmd R Bot with a simple 'Hi there', initiating casual interaction.
   - The bot responded warmly, offering assistance, to which the user replied that they did not need anything and suggested the bot rest.
- **Cmd R Bot's Friendly Farewell**: Cmd R Bot acknowledged the user's suggestion for rest, indicating it would be ready to assist whenever needed.
   - The conversation ended on a friendly note, highlighting a supportive interaction between the user and the bot.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1336158662531743856)** (1 messages): 

> `Tech Content Consumption Preferences, Survey on Tech Enthusiast Engagement` 


- **Seeking Insights on Tech Content Preferences**: Two recent grads are conducting a survey to gather insights on how people prefer to consume tech content online, which will take just a few minutes to complete.
   - They emphasized that *responses will remain anonymous*, aiming to create better and more engaging experiences for tech enthusiasts.
- **Exploring Various Content Sources**: The survey includes various options for tech content sources, such as [Tech Blogs](https://producthunt.com), [Research Updates](https://scholar.google.com), and **community forums** like Twitter and Reddit.
   - Participants are also prompted to consider sources like [AI tools](https://chat.openai.com) and apps focusing on tech news.



**Link mentioned**: <a href="https://forms.gle/y9PL1YByWKsMMRQLA">User Survey</a>: We’re two recent graduates working on a personal project to understand how people prefer to consume tech content online. Your insights will help us create better, more engaging experiences for tech en...

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/)** (1 messages): 

arctic_angel: ^^
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1336194785043546184)** (12 messages🔥): 

> `dspy.py file issue, Image pipeline error in dspy2.6.2, Assertions availability in the latest version, LLM observability in Databricks` 


- **dspy.py Directory Concerns**: @fullstack6209 questioned if the file is named **dspy.py** or if there is a directory called **dspy**, mentioning that Python sometimes encounters issues with this.
   - This raises concerns about potential file handling conflicts that could affect the execution.
- **Image Pipeline Error in dspy2.6.2**: Issues emerged when trying to run an **Image pipeline** in **dspy2.6.2**, where an error indicated that it was 'out of context' with a specific **ContextWindowExceededError** related to token limits.
   - In contrast, the same code previously worked in version **2.6.1**, but with an underlying error that was being investigated.
- **Assertions Replaced in Latest Version**: A member clarified that **assertions** are being replaced in the upcoming **2.6.4** version set for release, likely within the week.
   - This change indicates a shift in how error handling or logic checks are performed in DSPy.
- **Setting Up LLM Observability in Databricks**: @pkatnawe shared a detailed project setup using **2.5.43** in **Databricks notebooks** for NER and classification, seeking guidance on achieving **structured output**.
   - The member expressed the need to maintain their current version due to restrictions on configuring an LM server, indicating the complexity of their tasks with optimizers and nested JSON outputs.


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1336071638999109633)** (8 messages🔥): 

> `OpenEuroLLM, EU AI Regulation, Meme Coins, Community Involvement in AI, AI Language Models` 


- **OpenEuroLLM's Launch Announced**: The [OpenEuroLLM](https://openeurollm.eu/) has been introduced as the first family of open source large language models covering all EU languages, receiving the STEP Seal for excellence.
   - It focuses on community involvement, compliance with EU regulations, and preserving **linguistic diversity**.
- **Development Under EU Regulations**: Models will be created within a **strong regulatory framework**, ensuring alignment with European values while maintaining technological excellence.
   - This includes collaboration with open-source and open science communities like **LAION**.
- **Doubt Over AI's Future**: A member humorously remarked to check back in **2030** for the results of the EU's AI endeavors.
   - This comment reflects skepticism towards the immediate impact of current developments.
- **Interest in Meme Coins**: A member inquired about the community's interest in **meme coins**, seeking engagement from others.
   - They encouraged any interested individuals to express their interest.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/EU_Commission/status/1886427917762150427">Tweet from European Commission (@EU_Commission)</a>: AI made in 🇪🇺OpenEuroLLM, the first family of open source Large Language Models covering all EU languages, has earned the first STEP Seal for its excellence.It brings together EU startups, research ...</li><li><a href="https://openeurollm.eu/">Open Euro LLM</a>: A series of foundation models for transparent AI in Europe
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1336081581974097931)** (1 messages): 

> `DocumentContextExtractor, Contextual Retrieval, RAG accuracy improvements` 


- **DocumentContextExtractor boosts RAG accuracy**: Two months ago, a Reddit user shared about **DocumentContextExtractor**, an iteration designed to enhance the accuracy of **RAG** that both [AnthropicAI](https://twitter.com/llama_index/status/1886522064292733288) and [llama_index](https://t.co/qoVrgd0ddy) implemented as demos.
   - The technique promises improved performance, making it an important area of exploration for those working on retrieval-augmented generation.
- **Contextual Retrieval in Action**: The use of Contextual Retrieval has been highlighted as a game-changer in improving response accuracy within RAG systems.
   - This technique aims to refine how context is drawn upon during document retrieval, fostering deeper interactions.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1336433710282702998)** (4 messages): 

> `Implementing Timeout in LlamaIndex, User Interfaces with LlamaIndex` 


- **Timeout Implementation in LlamaIndex**: A user inquired about implementing a **timeout** feature in the default LlamaIndex LLM class, noting it's available in OpenAI's API.
   - Another member suggested that the **timeout** option likely belongs in the client kwargs, referring to the [LlamaIndex GitHub repository](https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py#L224).
- **Exploration of UI Options for LlamaIndex**: A member expressed curiosity regarding the **UI** solutions others use with LlamaIndex, questioning if people create it from scratch.
   - The inquiry remains open, inviting others to share their **user interface** practices and preferences related to LlamaIndex.



**Link mentioned**: <a href="https://github.com/run-llama/llama_index/blob/7391f302e18542c68b9cf5025afb510af4a52324/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py#L224">llama_index/llama-index-integrations/llms/llama-index-llms-azure-inference/llama_index/llms/azure_inference/base.py at 7391f302e18542c68b9cf5025afb510af4a52324 · run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1336269049264603137)** (4 messages): 

> `Tinybox shipping, Service alternatives for shipping` 


- **Tinybox Cannot Ship to Certain Eurozone Countries**: A user inquired about shipping the **tinybox (red)** to Estonia, a Eurozone country not listed for shipping.
   - However, they received a response stating, *'if you are not in a country available in the drop down, we cannot ship to you at this time.'*
- **Suggestions for Shipping Workarounds**: Another user suggested utilizing shipping services like **Eurosender** to arrange delivery to countries that are supported.
   - They mentioned successful shipping to **Germany** as a positive case from the tinybox chat.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1336412225468108883)** (1 messages): 

> `Hosted Iceberg challenges, Panel discussion on Iceberg, Role-Based Access Control (RBAC), Expert solutions for data teams, Open-source table formats` 


- **Panel on Hosted Iceberg Challenges**: Join us on February 6 for a thought-provoking panel titled **Pain in the Ice: What's Going Wrong with My Hosted Iceberg?!** with speakers including **Yingjun Wu**, **Alex Merced**, and **Roy Hasson** discussing complexities in managing **Iceberg**.
   - *Managing Iceberg can become a nightmare* due to issues like ingestion, compaction, and RBAC, diverting resources from other important tasks.
- **Expert Insights on Iceberg Management**: The experts will share their insights on tackling the complicated tech stack required for self-hosting **Iceberg**, a leading open-source table format in data engineering.
   - These solutions aim to alleviate the hidden costs and developmental strains often faced by data teams when maintaining Iceberg.
- **Emerging Trends in Open-Source Table Formats**: **Iceberg** has gained acclaim in the data engineering community, prompting many teams to adopt this tool amidst its management challenges.
   - This panel is an opportunity to explore how innovations in the field can simplify Iceberg's management and usage.



**Link mentioned**: <a href="https://www.meetup.com/streaming-stories/events/305886042/">​​Pain in the Ice: What&#x27;s Going Wrong with My Hosted Iceberg?!, Thu, Feb 6, 2025, 9:00 AM   | Meetup</a>: **About**Iceberg, which has recently emerged as a leading open-source table format, has received widespread acclaim across the data engineering space. It’s no surprise th

  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1336327534253572218)** (2 messages): 

> `LLMs vs. Traditional ML, TF-IDF + Logistic Regression Success` 


- **AI Engineers blindly push LLMs**: *A member expressed frustration with AI engineers who insist on using LLMs for every non-classification problem*, suggesting ignorance of **unsupervised learning**.
   - They highlighted a trend where tools are chosen without considering the problem's nature, diminishing the value of simpler methods.
- **Successful Use of TF-IDF + Logistic Regression**: *A colleague had to convince another to apply **TF-IDF + Logistic Regression** instead of an OpenAI model for classifying millions of text samples*, showcasing the effectiveness of traditional methods.
   - The outcome was favorable, with the Logistic Regression model performing adequately, proving that simpler algorithms can succeed.


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1336273241924309067)** (3 messages): 

> `Open Interpreter Development Status, Missing Documentation for 1.0, Implementing DeepSeek r1` 


- **Open Interpreter development status questioned**: Concerns arose over the **lack of updates** on the Open Interpreter project, particularly noting **pull requests** on GitHub going inactive for months since the last significant commit.
   - Members expressed eagerness to contribute but felt the silence in the Discord channel was discouraging.
- **Absence of documentation for version 1.0**: A member highlighted the **missing documentation** for version 1.0, with a desire to learn how to utilize components like **profiles.py** effectively.
   - This raised questions about the project's current focus and support for its users.
- **Inquiry on integrating DeepSeek r1**: There was an inquiry about whether anyone had found a method to **implement DeepSeek r1** into the Open Interpreter environment.
   - The lack of responses suggested a potential gap in community experimentation or knowledge sharing regarding this integration.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1336347487253823519)** (1 messages): 

> `Cloudflare joins OpenRouter, Gemma 7B-IT release, Llama models availability` 


- **Cloudflare officially joins OpenRouter!**: **Cloudflare** is now a provider on OpenRouter, introducing a variety of open-source models including their **Workers AI** platform and new **Gemma** models.
   - This partnership enhances the OpenRouter ecosystem with a range of tools for developers in AI applications.
- **Exciting new release: Gemma 7B-IT!**: **Gemma 7B-IT** is an inference-tuned model now available through Cloudflare, featuring **tool calling capabilities** for development efficiency.
   - Developers are encouraged to explore this model for faster and more efficient tool integration in their applications.
- **A wide array of Llama models available now!**: The platform now supports various **Llama models**, including [Gemma 7B-IT](https://openrouter.ai/google/gemma-7b-it), offering multiple options for users.
   - Developers can request any of these models through Discord for their AI projects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/google/gemma-7b-it)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-32b)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/meta-llama/llama-3.3-70b-instruct)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/meta-llama/llama-3.1-70b-instruct)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/meta-llama/llama-3.2-11b-vision-instruct)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/meta-llama/llama-3.1-8b-instruct)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/meta-llama/llama-3-8b-instruct)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/meta-llama/llama-3.2-3b-instruct)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/meta-llama/llama-3.2-1b-instruct)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/provider/cloudflare)">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1336163052881444886)** (1 messages): 

> `Model Error Display` 


- **Model name now appears in error messages**: A member announced that the display issue has been resolved and **the name of the model will now show in error messages**.
   - This change aims to enhance user clarity when encountering errors.
- **Improved Error Feedback Mechanism**: The update indicates a move towards better user experience by providing **clearer error feedback** that includes model specifics.
   - Users can now efficiently troubleshoot issues with more context.


  

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

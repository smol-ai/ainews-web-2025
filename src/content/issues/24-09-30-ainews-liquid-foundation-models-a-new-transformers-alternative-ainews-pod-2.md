---
id: 21fed6b0-f5de-419a-8a11-9f0745506282
title: 'Liquid Foundation Models: A New Transformers alternative + AINews Pod 2'
date: '2024-10-01T01:34:19.663940Z'
original_slug: ainews-liquid-foundation-models-a-new
description: >-
  Liquid.ai emerged from stealth after *10 months* post their *$37M seed*
  funding, unveiling three efficient subquadratic "liquid networks" models that
  outperform state space models and Apple’s foundation models in parameter
  efficiency, with a full launch planned for *October 23, 2024*. Additionally,
  Meta AI released **Llama 3.2**, introducing *11B* and *90B* multimodal models
  with advanced vision capabilities and lightweight text-only models for mobile,
  while Google DeepMind announced production-ready Gemini-1.5 models with
  significant cost and performance improvements.
tags:
  - liquid
  - apple
  - meta
  - google
  - deepmind
  - openai
  - stability
  - ea
  - liquid-networks
  - llama-3.2
  - gemini-1.5-pro-002
  - gemini-1.5-flash-002
  - o1-preview
  - qwen-2.5
  - emu3
  - sdxl
  - llava-1.6
  - foundation-models
  - multimodal-models
  - subquadratic-models
  - chip-design
  - reinforcement-learning
  - open-source
  - ai-regulation
  - next-token-prediction
  - continuous-finetuning
  - audio-overviews
  - generative-ai
  - video-generation
  - ai-in-medicine
  - user-generated-content
  - benchmark-results
companies:
  - liquid
  - apple
  - meta
  - google
  - deepmind
  - openai
  - stability
  - ea
models:
  - liquid-networks
  - llama-3.2
  - gemini-1.5-pro-002
  - gemini-1.5-flash-002
  - o1-preview
  - qwen-2.5
  - emu3
  - sdxl
  - llava-1.6
topics:
  - foundation-models
  - multimodal-models
  - subquadratic-models
  - chip-design
  - reinforcement-learning
  - open-source
  - ai-regulation
  - next-token-prediction
  - continuous-finetuning
  - audio-overviews
  - generative-ai
  - video-generation
  - ai-in-medicine
  - user-generated-content
  - benchmark-results
---


<!-- buttondown-editor-mode: plaintext -->**Adaptive computational operators are all you need.**

> AI News for 9/27/2024-9/30/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **31** Discords (**225** channels, and **5435** messages) for you. Estimated reading time saved (at 200wpm): **604 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

It's not every day that a credible new foundation model lab launches, so the prize for today rightfully goes to Liquid.ai, who, 10 months after [their $37m seed](https://siliconangle.com/2023/12/06/liquid-ai-raises-37-6m-build-liquid-neural-networks/), finally "came out of stealth" announcing 3 subquadratic models that perform remarkably well for their weight class:

[![image.png](https://assets.buttondown.email/images/f4006762-e87d-449a-9acd-7a60e88e20d1.png?w=960&fit=max)](https://x.com/AndrewCurran_/status/1840802455225094147)

We know precious little about "liquid networks" compared to state space models, but they have the obligatory subquadratic chart to show that they beat SSMs there:

![image.png](https://assets.buttondown.email/images/3502168f-ebe5-429f-8c75-cc43fc03852a.png?w=960&fit=max)

with very credible benchmark scores:

![image.png](https://assets.buttondown.email/images/8ad83dec-2a97-4f2f-86a6-6c609c2af5c2.png?w=960&fit=max)

Notably they seem to be noticeably more efficient per parameter than both the Apple on device and server foundation models ([our coverage here](https://buttondown.com/ainews/archive/ainews-apple-intelligence/)).

They aren't open source yet, but have a playground and API and have more promised coming up to their Oct 23rd launch.

---

**AINews Pod**

We first previewed [our Illuminate inspired podcast](https://buttondown.com/ainews/archive/ainews-not-much-happened-today-ainews-podcast/) earlier this month. With NotebookLM Deep Dive going viral, we're building an open source audio version of AINews as a new experiment. See [our latest comparison between NotebookLM and [our pod here](https://github.com/smol-ai/temp/tree/main/2024-09-30)! Let us know [@smol_ai](https://twitter.com/smol_ai) if you have feedback or want the open source repo.


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

**AI Model Updates and Developments**

- **Llama 3.2 Release**: Meta AI announced Llama 3.2, featuring 11B and 90B multimodal models with vision capabilities, as well as lightweight 1B and 3B text-only models for mobile devices. The vision models support image and text prompts for deep understanding and reasoning on inputs. [@AIatMeta](https://twitter.com/AIatMeta/status/1840431307761054202) noted that these models can take in both image and text prompts to deeply understand and reason on inputs.

- **Google DeepMind Announcements**: Google announced the rollout of two new production-ready Gemini AI models: Gemini-1.5-Pro-002 and Gemini-1.5-Flash-002. [@adcock_brett](https://twitter.com/adcock_brett/status/1840422127331057885) highlighted that the best part of the announcement was a 50% reduced price on 1.5 Pro and 2x/3x higher rate limits on Flash/1.5 Pro respectively.

- **OpenAI Updates**: OpenAI rolled out an enhanced Advanced Voice Mode to all ChatGPT Plus and Teams subscribers, adding Custom Instructions, Memory, and five new 'nature-inspired' voices, as reported by [@adcock_brett](https://twitter.com/adcock_brett/status/1840422082301046850).

- **AlphaChip**: Google DeepMind unveiled AlphaChip, an AI system that designs chips using reinforcement learning. [@adcock_brett](https://twitter.com/adcock_brett/status/1840422149829386581) noted that this enables superhuman chip layouts to be built in hours rather than months.

**Open Source and Regulation**

- **SB-1047 Veto**: California Governor Gavin Newsom vetoed SB-1047, a bill related to AI regulation. Many in the tech community, including [@ylecun](https://twitter.com/ylecun/status/1840511216889778332) and [@svpino](https://twitter.com/svpino/status/1840510698813829254), expressed gratitude for this decision, viewing it as a win for open-source AI and innovation.

- **Open Source Growth**: [@ylecun](https://twitter.com/ylecun/status/1840431809479463187) emphasized that open source in AI is thriving, citing the number of projects on Github and HuggingFace reaching 1 million models.

**AI Research and Development**

- **NotebookLM**: Google upgraded NotebookLM/Audio Overviews, adding support for YouTube videos and audio files. [@adcock_brett](https://twitter.com/adcock_brett/status/1840422255420912045) shared that Audio Overviews turns notes, PDFs, Google Docs, and more into AI-generated podcasts.

- **Meta AI Developments**: Meta AI, the consumer chatbot, is now multimodal, capable of 'seeing' images and allowing users to edit photos using AI, as reported by [@adcock_brett](https://twitter.com/adcock_brett/status/1840422210395054368).

- **AI in Medicine**: A study on o1-preview model in medical scenarios showed that it surpasses GPT-4 in accuracy by an average of 6.2% and 6.6% across 19 datasets and two newly created complex QA scenarios, according to [@dair_ai](https://twitter.com/dair_ai/status/1840450324097904901).

**Industry Trends and Collaborations**

- **James Cameron and Stability AI**: Film director James Cameron joined the board of directors at Stability AI, seeing the convergence of generative AI and CGI as "the next wave" in visual media creation, as reported by [@adcock_brett](https://twitter.com/adcock_brett/status/1840422277994733702).

- **EA's AI Demo**: EA demonstrated a new AI concept for user-generated video game content, using 3D assets, code, gameplay hours, telemetry events, and EA-trained custom models to remix games and asset libraries in real-time, as shared by [@adcock_brett](https://twitter.com/adcock_brett/status/1840422300610388224).


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Emu3: Next-token prediction breakthrough for multimodal AI**

- **Emu3: Next-Token Prediction is All You Need** ([Score: 227, Comments: 63](https://reddit.com//r/LocalLLaMA/comments/1fsoe83/emu3_nexttoken_prediction_is_all_you_need/)): **Emu3**, a new suite of multimodal models, achieves **state-of-the-art performance** in both generation and perception tasks using **next-token prediction** alone, outperforming established models like **SDXL** and **LLaVA-1.6**. By tokenizing images, text, and videos into a discrete space and training a single transformer from scratch, Emu3 simplifies complex multimodal model designs and demonstrates the potential of next-token prediction for building general multimodal intelligence beyond language. The researchers have open-sourced key techniques and models, including code on [GitHub](https://github.com/baaivision/Emu3) and pre-trained models on [Hugging Face](https://huggingface.co/collections/BAAI/emu3-66f4e64f70850ff358a2e60f), to support further research in this direction.
  - **Booru tags**, commonly used in anime image boards and **Stable Diffusion** models, are featured in Emu3's generation examples. Users debate the necessity of supporting these tags for model popularity, with some considering it a **requirement** for widespread adoption.
  - Discussions arose about applying **diffusion models to text generation**, with mentions of **CodeFusion** paper. Users speculate on **Meta's GPU compute capability** and potential unreleased experiments, suggesting possible agreements between large AI companies to control information release.
  - The model's ability to generate **videos as next-token prediction** excited users, potentially initiating a "new era of video generation". However, concerns were raised about **generation times**, with reports of **10 minutes for one picture** on Replicate.


**Theme 2. Replete-LLM releases fine-tuned Qwen-2.5 models with performance gains**

- **Replete-LLM Qwen-2.5 models release** ([Score: 73, Comments: 55](https://reddit.com//r/LocalLLaMA/comments/1frynwr/repletellm_qwen25_models_release/)): Replete-LLM has released fine-tuned versions of **Qwen-2.5** models ranging from **0.5B to 72B** parameters, using the **Continuous finetuning method**. The models, available on **Hugging Face**, reportedly show performance improvements across all sizes compared to the original Qwen-2.5 weights.
  - Users requested **benchmarks and side-by-side comparisons** to demonstrate improvements. The developer added some benchmarks for the **7B model** and noted that running comprehensive benchmarks often requires significant computing resources.
  - The developer's **continuous finetuning method** combines previous finetuned weights, pretrained weights, and new finetuned weights to minimize loss. A [paper](https://docs.google.com/document/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?usp=sharing) detailing this approach was shared.
  - **GGUF versions** of the models were made available, including quantized versions up to **72B parameters**. Users expressed interest in testing these on various devices, from high-end machines to edge devices like phones.



## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Capabilities and Developments**

- **OpenAI's o1 model** can handle **5-hour tasks**, enabling longer-horizon problem-solving, compared to GPT-3 (5-second tasks) and GPT-4 (5-minute tasks), according to [OpenAI's head of strategic marketing](https://www.reddit.com/r/singularity/comments/1fsfz47/dane_vahey_head_of_strategic_marketing_at_openai/).

- **MindsAI achieved a new high score of 48%** on the ARC-AGI benchmark, with the [prize goal set at 85%](https://www.reddit.com/r/singularity/comments/1fs9ymg/new_arcagi_high_score_by_mindsai_48_prize_goal_85/).

- A [hacker demonstrated](https://www.reddit.com/r/singularity/comments/1fsdfjc/hacker_plants_false_memories_in_chatgpt_to_steal/) the ability to **plant false memories in ChatGPT** to create a persistent data exfiltration channel.

**AI Policy and Regulation**

- **California Governor Gavin Newsom vetoed** a [contentious AI safety bill](https://www.reddit.com/r/singularity/comments/1fsegyi/california_governor_vetoes_contentious_ai_safety/), highlighting ongoing debates around AI regulation.

**AI Ethics and Societal Impact**

- AI researcher **Dan Hendrycks posed a thought experiment** about a hypothetical new species with rapidly increasing intelligence and reproduction capabilities, [questioning which species would be in control](https://www.reddit.com/r/singularity/comments/1fs6ce0/dan_hendrycks_imagine_that_a_new_species_arrives/).

- The [cost of a single query to OpenAI's o1 model](https://www.reddit.com/r/OpenAI/comments/1fsdrxq/the_cost_of_a_single_query_to_o1/) was highlighted, sparking discussions about the economic implications of advanced AI models.

**Memes and Humor**

- A meme about [trying to contain AGI](https://www.reddit.com/r/singularity/comments/1fsb6ml/trying_to_contain_agi_be_like/) sparked discussions about the challenges of AI safety.

- Another meme questioned [whether humans are "the baddies"](https://www.reddit.com/r/singularity/comments/1fsk1ov/are_we_the_baddies/) in relation to AI development, leading to debates about AI consciousness and ethics.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1. AI Models Make Waves with New Releases and Upgrades**

- [**LiquidAI Challenges Giants with Liquid Foundation Models (LFMs)**](https://www.liquid.ai/liquid-foundation-models): LiquidAI launched LFMs—1B, 3B, and 40B models—claiming superior performance on benchmarks like **MMLU** and calling out competitors' inefficiencies. With team members from **MIT**, their architecture is set to challenge established models in the industry.
- [**Aider v0.58.0 Writes Over Half Its Own Code**](https://aider.chat/2024/09/26/architect.html): The latest release introduces features like model pairing and new commands, boasting that Aider created **53%** of the update's code autonomously. This version supports new models and enhances user experience with improved commands like `/copy` and `/paste`.
- [**Microsoft's Hallucination Detection Model Levels Up to Phi-3.5**](https://huggingface.co/grounded-ai/phi3.5-hallucination-judge): Upgraded from Phi-3 to Phi-3.5, the model flaunts impressive metrics—**Precision: 0.77**, **Recall: 0.91**, **F1 Score: 0.83**, and **Accuracy: 82%**. It aims to boost the reliability of language model outputs by effectively identifying hallucinations.

**Theme 2. AI Regulations and Legal Battles Heat Up**

- **California Governor Vetoes AI Safety Bill SB 1047**: Governor **Gavin Newsom** halted the bill designed to regulate AI firms, claiming it wasn't the optimal approach for public protection. Critics see this as a setback for AI oversight, while supporters push for capability-based regulations.
- **OpenAI Faces Talent Exodus Over Compensation Demands**: Key researchers at OpenAI threaten to quit unless compensation increases, with **$1.2 billion** already cashed out amid a soaring valuation. New CFO **Sarah Friar** navigates tense negotiations as rivals like **Safe Superintelligence** poach talent.
- [**LAION Wins Landmark Copyright Case in Germany**](https://www.technollama.co.uk/laion-wins-copyright-infringement-lawsuit-in-german-court): LAION successfully defended against copyright infringement claims, setting a precedent that benefits AI dataset use. This victory removes significant legal barriers for AI research and development.

**Theme 3. Community Grapples with AI Tool Challenges**

- **Perplexity Users Bemoan Inconsistent Performance**: Users report erratic responses and missing citations, especially when switching between web searches and academic papers. Many prefer **Felo** for academic research due to better access and features like source previews.
- **OpenRouter Users Hit by Rate Limits and Performance Drops**: Frequent **429 errors** frustrate users of **Gemini Flash**, pending a quota increase from Google. Models like **Hermes 405B free** show decreased performance post-maintenance, raising concerns over provider changes.
- **Debate Ignites Over OpenAI's Research Transparency**: Critics argue that OpenAI isn't sufficiently open about its research, pointing out that blog posts aren't enough. Employees assert transparency, but the community seeks more substantive communication beyond the [research blog](https://openai.com/index/learning-to-reason-with-llms/).

**Theme 4. Hardware Woes Plague AI Enthusiasts**

- **NVIDIA Jetson AGX Thor's 128GB VRAM Sparks Hardware Envy**: Set for 2025, the AGX Thor’s massive VRAM raises questions about the future of current GPUs like the **3090** and **P40**. The announcement has the community buzzing about potential upgrades and the evolving GPU landscape.
- **New NVIDIA Drivers Slow Down Stable Diffusion Performance**: Users with **8GB VRAM cards** experience generation times ballooning from **20 seconds to 2 minutes** after driver updates. The community advises against updating drivers to avoid crippling rendering workflows.
- **Linux Users Battle NVIDIA Driver Issues, Eye AMD GPUs**: Frustrations mount over NVIDIA's problematic Linux drivers, especially for **VRAM offloading**. Some users consider switching to **AMD cards**, citing better performance and ease of use in configurations.

**Theme 5. AI Expands into Creative and Health Domains**

- [**NotebookLM Crafts Custom Podcasts from Your Content**](https://notebooklm.google.com/): Google's NotebookLM introduces an audio feature that generates personalized podcasts using AI hosts. Users are impressed by the engaging and convincing conversations produced from their provided material.
- **Breakthrough in Schizophrenia Treatment Unveiled**: Perplexity AI announced the launch of the first schizophrenia medication in **30 years**, marking significant progress in mental health care. Discussions highlight the potential impact on patient care and treatment paradigms.
- **Fiery Debate Over AI-Generated Art vs. Human Creativity**: The Stability.ai community is torn over the quality and depth of **AI art** compared to human creations. While some champion AI-generated works as legitimate art, others argue for the enduring superiority of human artistry.


---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LinkedIn's Copied Code Controversy**: LinkedIn faced backlash for allegedly copying Unsloth's code without proper attribution, prompting intervention from Microsoft and GitHub to ensure proper credit.
   - The incident underscores the critical need for adherence to **open source licensing** and raises concerns about **intellectual property**.
- **Best Practices for Fine-tuning Llama Models**: To mitigate token generation issues, users discussed setting a **random seed** and evaluating output quality carefully during **Llama model fine-tuning**.
   - It's essential to configure EOS tokens correctly to maintain the model's original abilities during inference.
- **GGUF Conversion Errors**: Users encountered a 'cannot find tokenizer merges in model file' error when loading GGUF models, highlighting potential issues during model saving.
   - Understanding the conversion process and maintaining compatibility with tokenizer configurations is vital for ensuring smooth model transitions.
- **Liquid Foundation Models Launch**: LiquidAI announced the introduction of **Liquid Foundation Models (LFMs)**, featuring **1B, 3B, and 40B models**, but skepticism arose about the validity of these announcements.
   - Concerns were expressed regarding the accuracy of the claims, especially in relation to **Perplexity Labs**.
- **Leveraging Untapped Compute Power**: Members noted substantial **compute power** being underutilized, suggesting potential performance improvements across various hardware setups.
   - Achieving realistic performance boosts through optimization of existing resources indicates significant room for enhancement in current systems.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.58.0 Delivers Exciting Enhancements**: The release of [Aider v0.58.0](https://aider.chat/2024/09/26/architect.html) introduces features such as model pairing and new commands, with Aider creating **53%** of the update's code autonomously.
   - This version also supports new models and improves user experience with features like **clipboard command updates**.
- **Architect/Editor Models Improve Efficiency**: Aider utilizes a main model for planning and an optional editor model for execution, allowing configuration via `--editor-model` for optimal task handling.
   - This dual approach has sparked discussions on multi-agent coding capabilities and price efficiency for LLM tasks.
- **NotebookLM's New Podcast Feature Stands Out**: [NotebookLM](https://notebooklm.google/) launches an audio feature that generates custom podcasts from user content, showcasing AI hosts in a compelling format.
   - One example podcast demonstrates the technology's ability to create engaging conversations from provided material.
- **Automation Proposal for Content Generation**: The idea to use NotebookLM to automate the production of videos from release notes has been floated, potentially leading to an efficient tool named *ReleaseNotesLM*.
   - This tool aims to transform written updates into audio, streamlining processes for content creators.
- **Discussion on Model Cost Efficiency**: Using different models, such as `claude-3.5-sonnet` for architect tasks and `deepseek v2.5` for editing, can lead to **20x-30x cost reductions** on editor tokens.
   - Participants emphasized the advantages of strategic model selection based on cost and functionality, exploring script options for enhanced configuration.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Model Merging Techniques Discussed**: Users explored various methods for **merging AI models**, specifically focusing on approaches like PEFT merge and the **DARE** method to enhance performance during fine-tuning.
   - The conversation stressed the value of leveraging existing models rather than training LLMs from scratch, positioning these methods as pivotal for efficient task handling.
- **Medical AI Insights from Recent Papers**: A post summarized the **top research papers** in medical AI for September 21-27, 2024, including notable studies like *A Preliminary Study of o1 in Medicine*.
   - Members suggested breaking these insights into individual blog posts to increase engagement and discussion surrounding standout papers.
- **Hallucination Detection Model Performance Metrics**: The newly released **Hallucination Detection Model** upgraded from **Phi-3 to Phi-3.5** boasts impressive metrics: **Precision: 0.77**, **Recall: 0.91**, **F1 Score: 0.83**, and **accuracy: 82%**; [check out the model card](https://huggingface.co/grounded-ai/phi3.5-hallucination-judge).
   - This model aims to improve the reliability of language model outputs by effectively identifying hallucinations.
- **Gradio's Lackluster User Reception**: Community sentiment towards **Gradio** turned negative, with users labeling it as 'hot garbage' due to UI responsiveness issues and design flaws that complicate project management.
   - Despite the backlash, members encouraged seeking help in dedicated support channels, indicating a continued investment in troubleshooting.
- **Keypoint Detection Model Enhancements**: The announcement of the **OmDet-Turbo model** supports zero-shot object detection, integrating techniques from Grounding DINO and OWLv2; details can be found [here](https://www.linkedin.com/posts/yoni-gozlan_ai-artificialintelligence-objectdetection-ugcPost-7244768044533657603-FDOT?utm_source=share&utm_medium=member_desktop).
   - Exclusive focus on keypoint detection with models like **SuperPoint** sets the stage for community excitement over future developments in this field.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Challenges in downloading and sideloading models in LM Studio**: Users encountered issues with downloading models in *LM Studio*, particularly when using VPNs, prompting some to sideload models instead. Limitations on supporting model formats like **safetensors** and **GGUF** were noted.
   - The community expressed frustrations regarding the overall download experience, with discussions highlighting the necessity for better support with various model types.
- **NVIDIA Jetson AGX Thor boasts 128GB VRAM**: The upcoming **NVIDIA Jetson AGX Thor** is set to feature **128GB of VRAM** in 2025, raising questions about the viability of current GPUs like the **3090** and **P40**. This announcement has created a buzz around potential upgrades in the GPU landscape.
   - *Some members pondered whether existing hardware will remain competitive* as the demand for high-VRAM options continues to grow.
- **Comparing GPU performance: 3090 vs 3090 Ti vs P40**: Members compared the performance of the **3090**, **3090 Ti**, and **P40**, focusing on VRAM and pricing, which heavily influence their choices. One remark noted that the **P40** operates at approximately half the speed of the **3090**.
   - Members expressed concern over rising GPU prices and debated the trade-offs between different models for current AI workloads.
- **Market pricing dynamics for GPUs**: Discussions emphasized that **GPU prices remain high** due to scalping and increased demand for AI applications, with the **A6000** serving as a high-VRAM alternative. However, budget-conscious members favor options like multiple **3090s** for their setups.
   - *The conversation highlighted a general frustration regarding pricing trends* and the hurdles many face in the current market.
- **Challenges with NVIDIA drivers on Linux**: The community shared grievances about **NVIDIA's Linux drivers** being notoriously problematic, especially for **VRAM offloading**, an area where **AMD** cards perform better. Complications in setting up **CUDA** and other drivers underscored these frustrations.
   - Some members indicated a growing preference for **AMD** hardware, citing its superior ease of use in certain configurations.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Cerebras Chip Optimization Discussion**: Members are exploring code optimizations for **Cerebras chips** with varying opinions about potential purchases and expertise availability.
   - Community interest is growing as members show willingness to find experts for deeper insights into Cerebras technology.
- **Rising Concerns Over Spam Management**: The community is addressing an increase in **crypto scam spam** messages on Discord, suggesting stricter verification protocols to enhance server security.
   - Members are actively seeking efficient anti-spam tools and discussing their experiences with existing solutions such as AutoMod.
- **Triton Talk Materials Shared**: A member sought out slides from the **Triton talk** and was directed to the [GitHub repository](https://github.com/gpu-mode/lectures) containing educational resources.
   - This reflects a strong community culture of knowledge sharing and collaborative learning.
- **AMD GPU Performance Troubles**: Discussion on significant performance limitations of **AMD GPUs**, particularly with **GFX1100** and **MI300** architectures, was prominent among the members.
   - Many highlighted the ongoing challenges with multi-node setups and expressed the need for enhanced performance.
- **Understanding Model Parallelism vs ZeRO/FSDP**: Members clarified the distinctions between **Model Parallelism** and **ZeRO/FSDP**, focusing on how ZeRO implements parameter distribution strategies.
   - Discussions emphasized that FSDP utilizes sharding to enhance model training efficiency, appealing to those looking to understand advanced features.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular Community Meeting Agenda Explored**: Today's Modular Community Meeting at **10am PT** will cover the **MAX driver & engine API** and a Q&A on Magic, with access via [Zoom](https://modul.ar/community-meeting-zoom). Participants can check the [Modular Community Calendar](https://modul.ar/community-meeting) for upcoming events.
   - The meeting recording will be uploaded to YouTube, including today’s session available at [this link](https://www.youtube.com/watch?v=zL0cCHs_0RI&list=PLh0S94-sJw_6UcaIMgpESb5KSVRsuuhnX), ensuring no one misses out.
- **Debate on Mojo Language Enhancements**: A proposal for advanced **Mojo language features** suggested named variants for message passing and better management of tagged unions without new constructs, sparking extensive discussion among members.
   - Proponents weighed the ergonomics of defining types, discussing the balance of nominal versus structural types in the design process.
- **Bundling Models with Mojopkg**: The ability to embed models in **Mojopkg** was enthusiastically discussed, showcasing potential user experience improvements by bundling everything into a single executable application.
   - Key examples from other languages were mentioned, illustrating how this could simplify dependencies for users and enhance usability.
- **Managing Native Dependencies Smoothly**: Concerns were raised regarding Mojopkg's capability to simplify dependency management, potentially allowing for easier installation and configuration.
   - Discussion included practical implementations like embedding installers for runtimes such as Python directly into Mojo applications.
- **Compatibility Warnings on MacOS**: A user reported compatibility warnings when building object files for macOS, noting linking issues between versions **15.0** and **14.4**.
   - Although the warnings are not fatal, they could point to future compatibility challenges needing resolution.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research pushes open source initiatives**: Nous Research focuses on **open source AI research**, collaborating with builders and releasing models including the **Hermes family**.
   - Their **DisTrO project** aims to speed up AI model training across the internet, hinting at the perils of **closed source models**.
- **Distro paper release generating buzz**: The **Distro paper** is expected to be announced soon, igniting excitement among community members eager for updates.
   - This paper's relevance to the AI community amplifies anticipation surrounding its detailed content.
- **New AI Model Fine-tuning Techniques Unleashed**: The recent **Rombodawg’s Replete-LLM** topped the **OpenLLM leaderboard** for **7B models**, aided by innovative fine-tuning techniques.
   - Methods like **TIES merging** are identified as critical to enhancing model benchmarks significantly.
- **Liquid Foundation Models capture attention**: LiquidAI introduced **Liquid Foundation Models** with versions including **1B, 3B, and 40B**, aiming for fresh capabilities in the AI landscape.
   - These models are seen as pivotal in offering innovative functionalities for various applications within the AI domain.
- **Medical AI Paper of the Week: Are We Closer to an AI Doctor?**: The highlighted paper, *A Preliminary Study of o1 in Medicine*, explores the potential for AI to function as a doctor, authored by various experts in the field.
   - This paper was recognized as the **Medical AI Paper of the Week**, showcasing its relevance in ongoing discussions about AI's role in healthcare.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity struggles with performance consistency**: Users noted **inconsistent responses** from **Perplexity** while switching between web searches and academic papers, with instances of missing citations.
   - Concerns were raised about whether these inconsistencies reflect a bug or highlight underlying design flaws in the search functionality.
- **Felo superior for academic searches**: Many users find **Felo** more effective for academic research, citing better access to relevant papers over **Perplexity**.
   - Features like *hovering for source previews* enhance the research experience, drawing users to prefer Felo for its intuitive interface.
- **Inconsistent API outputs frustrate users**: The community discussed **API inconsistencies**, especially around the **PPLX API**, which was returning outdated **real estate listings** compared to the website data.
   - Suggestions were made to experiment with **parameters** like temperature and top-p to improve the API's response consistency.
- **Breakthrough in schizophrenia treatment**: Perplexity AI announced an important milestone with the **launch of the first schizophrenia medication** in **30 years**, marking significant progress in mental health solutions.
   - Discourse emphasized the potential ramifications for patient care and the evolution of treatment paradigms moving forward.
- **Texas counties use AI tech effectively**: Texas counties showcased innovative approaches to leverage AI applications in local government operations, enhancing public service capabilities.
   - Participants shared a detailed resource that highlights these practical implementations of AI technology in administrative tasks.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Struggles with Rate Limits**: Users report frequent **429 errors** while using **Gemini Flash**, causing significant frustration as they await a potential quota increase from Google.
   - This ongoing traffic issue is undermining the platform's usability, impacting user engagement.
- **Performance Decrease Post-Maintenance**: Models like **Hermes 405B free** have exhibited lower performance quality after recent updates, raising concerns about potential changes in model providers.
   - Users are advised to check their **Activity pages** to ensure they are using their preferred models.
- **Translation Model Options Suggested**: A user looked for efficient translation models for dialogue without strict limitations, expressing dissatisfaction with **GPT4o Mini**.
   - Open weight models fine-tuned with dolphin techniques were recommended as more flexible alternatives.
- **Frontend Chat GUI Recommendations**: A discussion emerged about chat GUI solutions that allow middleware flexibility, with **Streamlit** proposed as a viable option.
   - **Typingmind** was also mentioned for its customizable features in managing interactions with multiple AI agents.
- **Gemini's Search Functionality Discussion**: There’s interest in enabling direct search capabilities within **Gemini** models similar to **Perplexity**, though current usage limitations are still being evaluated.
   - Discussions referenced Google's **Search Retrieval API parameter**, highlighting the need for clearer implementation strategies.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux Model Hits a Home Run**: Impressed by kohya_ss's work, members noted that the **Flux model** can train on just **12G VRAM**, showcasing incredible performance capabilities.
   - Excitement spread about the advancements, hinting at a possible shift in model efficiency benchmarks.
- **Nvidia Drivers Slow Down SDXL**: New Nvidia drivers caused major slowdowns for **8GB VRAM cards**, with image generation times ballooning from **20 seconds to 2 minutes**.
   - Members strongly advised against updating drivers, as these changes detrimentally affected their rendering workflows.
- **Regional Prompting Hits Snags**: Community members shared frustrations with **regional prompting** in Stable Diffusion, specifically with character mixing in prompts like *'2 boys and 1 girl'*.
   - Suggestions arose to begin with broader prompts, leveraging general guides for optimal results.
- **AI Art Submission Call to Action**: The community's invited to submit AI-generated art for potential feature in **The AI Art Magazine**, with a deadline set for **October 20**.
   - This initiative aims to celebrate digital art and encourages members to flaunt their creativity.
- **AI Art Stirs Quality Debate**: A vigorous debate erupted regarding the merits of **AI art** versus human art, with opinions split on quality and depth.
   - Some argued for the superiority of human artistry, while others defended AI-generated works as legitimate artistic expression.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Aider benchmarks LLM editing skills**: Members discussed Aider's functionality, noting it excels with LLMs proficient in _editing_ code, as highlighted in its [leaderboards](https://aider.chat/docs/leaderboards/). Skepticism emerged around the reliability of Aider's benchmarks, especially concerning *Gemini Pro 1.5 002*.
   - While Aider showcases impressive edits, the potential for further testing and validation remains critical for broader acceptance in the community.
- **EU AI Bill sparks dialogue**: The discourse around the EU's AI bill intensified, with members sharing varying views on its implications for **multimodal AI regulation** and chatbot classifications under level two regulations. Concerns about the regulatory burden on tech companies were prevalent.
   - Many emphasized the necessity for clarity on how emerging AI technologies would be impacted by these regulations as they navigate compliance landscapes.
- **Meta's game-changer in video translation**: A member highlighted Meta's imminent release of a lip-sync video translation feature, set to enhance user engagement on the platform. This feature captivated discussions about its potential to reshape content creation tools.
   - Members expressed excitement over how this could elevate translation services and the implications for global content accessibility.
- **Voice mode quandaries in GPT-4**: Frustration brewed over the performance of **GPT-4o**, with urgent calls for the release of **GPT-4.5-o** following claims of it being 'the dumbest LLM'. Critiques centered on insufficient reasoning capabilities as a major concern.
   - Amidst user confusion, detailed discussions about daily limits and accessibility of **voice mode** highlighted the community's anticipation for enhancements in user experience.
- **Flutter Code Execution Error Resolved**: A user faced an error indicating an active run in thread `thread_ey25cCtgH3wqinE5ZqIUbmVT`, leading to suggestions for managing active runs and using the `cancel` function. The user ultimately resolved the issue by waiting longer between executions.
   - Participants recommended incorporating a status parameter to track thread completions, potentially streamlining thread management and reducing frustration in future interactions.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **New Members Enrich Community Dynamics**: Several new members, including a fullstack engineer from Singapore and a data engineer from Portugal, joined the conversation, eager to contribute to AI projects and open source initiatives.
   - Their enthusiasm for collaboration sets a promising tone for community growth.
- **AI Conferences on the Horizon**: Members discussed upcoming conferences like **ICLR** and **NeurIPS**, particularly with Singapore hosting ICLR, and are planning meetups.
   - Light-hearted conversation about event security roles added a fun twist to the coordination.
- **Liquid AI Launches Foundation Models**: [Liquid Foundation Models](https://www.liquid.ai/liquid-foundation-models) were announced, showcasing strong benchmark scores and a flexible architecture optimized for diverse industries.
   - The models are designed for various hardware, inviting users to test them on Liquid AI's platform.
- **Exploration of vLLM Metrics Extraction**: A member inquired about extracting **vLLM metrics objects** from the **lm-evaluation-harness library** using the `simple_evaluate` function on benchmarks.
   - They specifically sought metrics like **time to first token** and **time in queue**, prompting useful responses from the community.
- **ExecuTorch Enhances On-Device AI Capabilities**: **ExecuTorch** allows **customization and deployment** of PyTorch programs across various devices, including AR/VR and mobile systems, as per the platform overview.
   - Details were shared regarding the `executorch` pip package currently in alpha for Python **3.10 and 3.11**, compatible with **Linux x86_64** and **macOS aarch64**.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Optimizing Torchtune Training Configurations**: Users fine-tuned various settings for **Llama 3.1 8B**, optimizing parameters like `batch_size`, `fused`, and `fsdp_cpu_offload`, which led to decreased epoch times when `packed=True` was enabled.
   - *...and everyone agreed that `enable_activation_checkpoint` should remain `False` to boost compute efficiency.*
- **Demand for Dynamic CLI Solutions**: A proposal emerged to create a dynamic CLI using the `tyro` library, allowing for customizable help texts that reflect configuration settings in Torchtune recipes.
   - This flexibility aims to enhance user experience and streamline recipe management with clear documentation.
- **Memory Optimization Strategies Revealed**: Members recommended updating the memory optimization page to include both **performance and memory optimization tips**, promoting a more integrated approach.
   - Ideas like implementing **sample packing** and exploring **int4 training** were highlighted as potential enhancements for memory efficiency.
- **Error Handling Enhancements for Distributed Training**: A suggestion surfaced to improve error handling in distributed training by leveraging `torch.distributed`'s record utility for logging exceptions.
   - This approach facilitates easier troubleshooting by maintaining comprehensive error logs throughout the training process.
- **Duplicate Key Concerns in Configuration Management**: Discussion arose regarding **OmegaConf** flagging duplicate entries like `fused=True` in configs, highlighting the importance of clean and organized configuration files.
   - *We should add a performance section in configs,* placing fast options in comments to improve readability and immediate accessibility.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **CodiumAI rebrands with Series A funding**: QodoAI, previously known as CodiumAI, secured a **$40M** Series A funding, bringing their total to **$50M** to enhance **AI-assisted tools**.
   - *‘This funding validates their approach’* indicating developer support for their mission to ensure code integrity.
- **Liquid Foundation Models claim impressive benchmarks**: LiquidAI launched **LFMs**, showcasing superior performance on **MMLU** and other benchmarks, calling out competitors' inefficiencies.
   - With team members from MIT, their **1.3B model** architecture is set to challenge established models in the industry.
- **Gradio enables real-time AI voice interaction**: LeptonAI demonstrated **Gradio 5.0**, which includes real-time streaming with audio mode for LLMs, simplifying code integrations.
   - The updates empower developers to create interactive applications with ease, encouraging open-source collaboration.
- **Ultralytics launches YOLO11**: Ultralytics introduced **YOLO11**, enhancing previous versions for improved accuracy and speed in **computer vision tasks**.
   - The launch marks a critical step in the evolution of their YOLO models, showcasing substantial performance improvements.
- **Podcast listeners demand more researcher features**: The latest episode features **Shunyu Yao** and **Harrison Chase**, drawing interest from listeners eager for more **researcher involvement** in future episodes.
   - Engagements highlight listener enthusiasm, with comments like, *‘bring more researchers on’*, urging for deeper discussions.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **FinanceAgentToolSpec for Public Financial Data**: The [FinanceAgentToolSpec](https://t.co/7bsEm4Er1m) package on LlamaHub allows agents to access public financial data from sources like **Polygon** and **Finnhub**.
   - Hanane's detailed post emphasizes how this tool can streamline financial analysis through querying.
- **Full-Stack Demo Showcases Streaming Events**: A new [full-stack application](https://t.co/HOajPyiqQb) illustrates workflows for streaming events with Human In The Loop functionalities.
   - This app demonstrates how to research and present a topic, boosting user engagement significantly.
- **YouTube Tutorial Enhances Workflow Understanding**: A [YouTube video](https://t.co/Nn5NVZopPz) provides a developer's walkthrough of the coding process for the full-stack demo.
   - This resource aims to aid those wishing to implement similar streaming systems.
- **Navigating RAG Pipeline Evaluation Challenges**: Users reported issues with RAG pipeline evaluation using trulens, particularly addressing import errors and data retrieval.
   - This led to discussions on the importance of building a solid evaluation dataset for accurate assessments.
- **Understanding LLM Reasoning Problems**: Defining the type of reasoning problem is essential for engaging with LLM reasoning, as highlighted in a shared article detailing reasoning types.
   - The article emphasizes that various reasoning challenges require tailored approaches for effective evaluation.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Startup Program Discounts Available**: A user inquired about discounts for a startup team using Cohere, citing costs compared to **Gemini**. It was suggested they apply to the [Cohere Startup Program](https://cohere.com/startup-program) for potential relief.
   - Participants mentioned that the application process might take time, but they affirmed the significance of this support for early-stage ventures.
- **Improve Flash Card Generation by Fine-tuning**: Members discussed **fine-tuning a model** specifically for flash card generation from notes and slide decks, addressing concerns about output clarity. It was suggested to employ best practices for machine learning pipelines and utilize **chunking data** for improved results.
   - Chunking was highlighted as beneficial, particularly for processing PDF slide decks, enhancing the model's understanding and qualitative output.
- **Cultural Multilingual LMM Benchmark Launch**: MBZUAI is developing a **Cultural Multilingual LMM Benchmark** for **100 languages**, and is actively seeking native translators to volunteer for error correction. Successful participants will be invited to co-author the resulting paper.
   - The scope of languages includes **Indian**, **South Asian**, **African**, and **European** languages, and interested parties can connect with the project lead via **LinkedIn**.
- **RAG Header Formatting for LLM Prompts**: Users sought guidance on formatting **instructional headers** for RAG prompts to ensure the LLM interprets inputs correctly. Discussions emphasized the need for precise supporting information and proper termination methods for headers.
   - The conversation highlighted how clarity in formatting can mitigate errors in model responses, enhancing engagement with the LLM.
- **Gaps in API Documentation Identified**: A user noted inconsistencies in the API documentation regarding penalty ranges, calling for clearer standards on parameter values. This conversation reflects ongoing concerns about **documentation consistency** and user clarity in utilizing API features.
   - Discussions around API migration from v1 to v2 corroborated that while older functionality remains, systematic updates are essential for a smooth transition.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI's talent exodus due to compensation demands**: Key researchers at OpenAI are seeking higher compensation, with **$1.2 billion** already cashed out from selling profit units as the company’s valuation rises. This turnover is heightened by rivals like Safe Superintelligence actively recruiting talent.
   - *Employees are threatening to quit over money issues* while new CFO **Sarah Friar** navigates these negotiations.
- **California Governor vetoes AI safety bill SB 1047**: Gov. **Gavin Newsom** vetoed the bill aimed at regulating AI firms, claiming it wasn't the best method for public protection. Critics view this as a setback for oversight while supporters push for regulations based on specific capabilities.
   - *Sen. Scott Wiener expressed disappointment* over the lack of prior feedback from the governor, emphasizing the lost chance for California to lead in tech regulation.
- **PearAI faces allegations of code theft**: **PearAI** has been accused of stealing code from [Continue.dev](http://Continue.dev) and rebranding it without acknowledgment, urging investors like YC to push for accountability. This raises significant ethical concerns about funding within the startup ecosystem.
   - *The controversy highlights ongoing concerns* about the integrity of open-source communities and their treatment by emerging tech firms.
- **Debate on Transparency in OpenAI Research**: Critics question OpenAI's transparency, emphasizing that referencing a blog does not provide substantive communication of research findings. Some employees assert that the company is indeed open about their research.
   - *Discussions highlight mixed feelings on whether* [OpenAI's research blog](https://openai.com/index/learning-to-reason-with-llms/) sufficiently addresses the community's transparency concerns.
- **Insights on Access to iPhone IAP Subscriptions**: A substack best seller announced gaining access to **iPhone In-App Purchase subscriptions**, indicating new opportunities in mobile monetization. This development gives insight into implementing and managing these systems.
   - *The discussions reflect developers’ frustrations* with the chaotic environment of managing the **Apple App Store** and their experiences with its complexities.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Course Materials Ready for Access**: Students can access all course materials, including assignments and lecture recordings, on the [course website](https://llmagents-learning.org/f24), with submission deadline set for **Dec 12th**.
   - *It's important to check the site regularly for updates on materials as well.*
- **Multi-Agent Systems vs. Single-Agent Systems**: Discussion emerged regarding the need for multi-agent systems rather than single-agent implementations in project contexts to reduce hallucinations and manage context.
   - Participants noted that these systems might yield more accurate responses from **LLMs**.
- **Curiosity Around NotebookLM's Capabilities**: Members inquired if **NotebookLM** functions as an agent application, revealing it acts as a RAG agent that summarizes text and generates audio.
   - Questions also surfaced regarding its technical implementation, particularly in multi-step processes.
- **Awaiting Training Schedule Confirmation**: Students are eager for confirmation on when training sessions start, with one noting that all labs were expected to be released on **Oct 1st**.
   - *However, this timeline was not officially confirmed.*
- **Exploring Super-Alignment Research**: A proposed research project is in discussion, aiming to study ethics in multi-agent systems using frameworks like **AutoGen**.
   - Challenges regarding the implementation of this study without dedicated frameworks were raised, highlighting limitations in simulation capabilities.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Cloud Storage Costs Competitive with Major Providers**: George mentioned that **storage and egress costs** will be less than or equal to major cloud providers, emphasizing cost considerations.
   - He further explained that expectations for usage might alter perceived costs significantly.
- **Modal's Payment Model Sparks Debate**: Modal's unique pricing where they charge by the second for compute resources has drawn attention, touted as **cheaper than traditional hourly rates**.
   - Members questioned the sustainability of such models and how it aligns with consistent usage patterns in the AI startup environment.
- **Improving Tinygrad's Matcher with State Machines**: A member suggested that implementing a **matcher state machine** could improve performance, aligning it towards C-like efficiency.
   - George enthusiastically backed this approach, indicating it could achieve the desired performance improvements.
- **Need for Comprehensive Regression Testing**: Concerns were raised about the lack of a **regression test suite** for the optimizer, which could lead to unnoticed issues after code changes.
   - Members discussed the idea of serialization for checking optimization patterns, but recognized it would not be engaging.
- **SOTA GPU not mandatory for bounties**: A member suggested that while a **SOTA GPU** could help, one can manage with an average GPU, especially for certain tasks.
   - Some tasks like **100+ TFLOPS matmul in tinygrad** may require specific hardware like the **7900XTX**, while others do not.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llama 3.2 Tuning Hits VRAM Wall**: Users face **high VRAM usage** of **24GB** when tuning **Llama 3.2 1b** with settings like qlora and 4bit loading, leading to discussions on balancing sequence length and batch size.
   - Concerns specifically highlight the impact of sample packing, emphasizing a need for optimization in the tuning configuration.
- **California Mandates AI Training Transparency**: A new **California law** now requires **disclosure of training sources** for all AI models, affecting even smaller non-profits without exceptions.
   - This has spurred conversations on utilizing lightweight chat models for creating compliant datasets, as community members brainstorm potential workarounds.
- **Lightweight Chat Models Gain Traction**: Members are exploring **finetuning lightweight chat models** from webcrawled datasets, aiming to meet legal transformation standards.
   - One user pointed out that optimizing messy **raw webcrawl data** through LLMs can be a significant next step in the process.
- **Liquid AI Sparks Curiosity**: The introduction of **Liquid AI**, a new foundation model, has piqued interest among members due to its potential features and applications.
   - Members are keen to discuss what legislative changes might mean for this model and its practical implications in light of recent developments.
- **Maximizing Dataset Usage in Axolotl**: In **Axolotl**, configure datasets to use the first **20%** by adjusting the `split` option in dataset settings for training purposes.
   - A lack of random sample selection directly in Axolotl means users must preprocess data, utilizing Hugging Face’s `datasets` for random subset sampling before loading.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy showcases live Pydantic model generation**: A livecoding session demonstrated how to create a **free Pydantic model generator** using [Groq](https://groq.com) and **GitHub Actions**.
   - Participants can catch the detailed demonstration in the shared [Loom video](https://www.loom.com/share/783ed4d80720492da23f39d2678de27f).
- **Upgrade to DSPy 2.5 delivers notable improvements**: Switching to **DSPy 2.5** with the **LM client** and a **Predictor** over **TypedPredictor** led to enhanced performance and fewer issues.
   - Key enhancements stemmed from the new **Adapters** which are now more aware of **chat LMs**.
- **OpenSearchRetriever ready for sharing**: A member is willing to share their developed [OpenSearchRetriever for DSPy](https://link.to.github) if the community shows interest.
   - This project could streamline integration and functionality, and it was encouraged that they make a PR.
- **Challenges in Healthcare Fraud Classifications**: A member is facing difficulties accurately classifying **healthcare fraud** in DOJ press releases, leading to misclassifications.
   - The community discussed refining the classification criteria to enhance accuracy in this critical area.
- **Addressing Long Docstring Confusion**: Confusion arose around using long explanations in **docstrings**, affecting accuracy in class signatures.
   - Members provided insights on the importance of clear documentation, but the user needed clarity on the language model being leveraged.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Full-Stack Developer Seeks Projects**: A full-stack developer is looking for new clients, specializing in **e-commerce platforms**, online stores, and real estate websites using **React + Node** and **Vue + Laravel** technologies.
   - They are open to discussions for long-term collaborations.
- **Query on Re-instructing AI Execution**: A member asked about the possibility of modifying the **AI execution instructions** to enable users to independently fix and debug issues, pointing to frequent path-related errors.
   - There was a clear expression of frustration regarding current system capabilities.
- **Persistent Decoding Packet Error**: Users reported a recurrent **decoding packet issue**, with the error message: *Invalid data found when processing input* during either server restarts or client connections.
   - Suggestions were made to check for terminal error messages, but none were found, indicating consistent issues.
- **Ngrok Authentication Troubles**: A member encountered an **ngrok authentication error** requiring a verified account and authtoken during server execution.
   - They suspected the issue might relate to the .env file not properly reading the *apikey*, asking for assistance on this topic.
- **Jan AI as a Computer Control Interface**: A member shared insights on using **Jan AI** with **Open Interpreter** as a local inference server for local LLMs, inviting feedback on others' experiences.
   - They provided a [YouTube video](https://www.youtube.com/watch?v=1l3B0AzbbjQ) that showcases how Jan can interface to control computers.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Request for French Audio Datasets**: A user needs high-quality audio datasets in **French** for training **CosyVoice**, emphasizing the urgency to obtain suitable datasets.
   - *Without proper datasets,* they expressed uncertainty about progressing on their project.
- **LAION Claims Victory in Copyright Challenge**: **LAION** won a major copyright infringement challenge in a **German court**, setting a precedent in legal barriers for AI datasets.
   - Further discussions emphasized the implications of this victory, which can be found on [Reddit](https://www.reddit.com/r/aiwars/comments/1fqpiut/laion_wins_first_copyright_infringement_challenge/).
- **Exploring Text-to-Video with Phenaki**: Members explored the **Phenaki** model for generating videos from text, sharing a [GitHub link](https://github.com/lucidrains/make-a-video-pytorch) for initial tests.
   - They requested guidance for testing its capabilities due to a lack of datasets.
- **Synergy Between Visual Language and Latent Diffusion**: Discussion revolved around the potential of combining **VLM** (Visual Language Models) and **LDM** (Latent Diffusion Models) for enhanced image generation.
   - A theoretical loop was proposed where **VLM** instructs **LDM**, effectively refining the output quality.
- **Clarifying Implementation of PALM-RLHF Datasets**: A member inquired about the suitable channel for implementing **PALM-RLHF** training datasets tailored to specific tasks.
   - They aimed for clarity on aligning these datasets with operational requirements.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Vectorstores could use example questions**: A member suggested that incorporating example questions might enhance vectorstore performance in finding the closest match, although it may be considered excessive.
   - They highlighted the importance of **testing** to measure the actual effectiveness of this approach.
- **Database beats table data for LLMs**: A member pointed out that switching from table data to a **Postgres** database is more suitable for LLMs, leading them to utilize **LangChain modules** for interaction.
   - This transition aims to optimize data handling for model training and queries.
- **Exploring thank you gifts in Discord**: An inquiry was made about the feasibility of sending small thank you gifts to members in Discord who provided assistance.
   - This reflects a desire to acknowledge contributions and build community bonds.
- **Gemini faces sudden image errors**: A member reported unexpected errors when sending images to **Gemini**, noting that this issue emerged after recent upgrades to all **pip packages**.
   - The situation raised concerns about potential compatibility issues post-upgrade.
- **Modifying inference methods with LangChain**: A member is investigating modifications to the inference method of chat models using **LangChain**, focusing on optimizations in **vllm**.
   - They seek to control token decoding, particularly around chat history and input invocation.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Realized Summit 2024 Set for October 2**: Excitement is building for the [AI Realized - The Enterprise AI Summit](https://lu.ma/airsummit) on October 2, 2024, hosted by **Christina Ellwood** and **David Yakobovitch** at UCSF, featuring industry leaders in Enterprise AI.
   - Attendees can use code **extra75** to save **$75** off their tickets, which include meals at the conference.
- **Kickoff of Manifold Research Frontiers Talks**: **Manifold Research** is launching the Frontiers series to spotlight innovative work in foundational and applied AI, starting with a talk by **Helen Lu** focused on neuro-symbolic AI and human-robot collaboration.
   - The talk will discuss challenges faced by autonomous agents in dynamic environments and is open for free registration [here](https://lu.ma/cbflyi6s).
- **Inquiry on MLOps Meetups in Stockholm**: A member is seeking information about **MLOps or Infrastructure meetups** in Stockholm after recently moving to the city.
   - They expressed a desire to connect with the local tech community and learn about upcoming events.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Calytrix introduces anti-slop sampler**: A prototype **anti-slop sampler** suppresses unwanted words during inference by backtracking on detected sequences. Calytrix aims to make the codebase usable for downstream purposes, with the project available on [GitHub](https://github.com/sam-paech/antislop-sampler).
   - This approach targets enhancing **dataset quality** directly by reducing noise in generated outputs.
- **Community backs anti-slop concept**: Members shared positive feedback about the **anti-slop sampler**, with one commenting, *'cool, I like the idea!'* highlighting its potential impact.
   - The enthusiasm indicates a growing interest in solutions that refine **dataset generation** processes.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Takiyoshi Hoshida showcases SoraSNS**: Indie developer **Takiyoshi Hoshida** will present a live demo of his project **[SoraSNS](https://discord.com/events/1089876418936180786/1277835047084363827)**, a social media app offering a private timeline from users you don't typically follow.
   - The demo emphasizes the app's unique concept of **day and night** skies, symbolizing openness and distant observation to enhance user experience.
- **Hoshida's impressive tech credentials**: Takiyoshi Hoshida studied **Computer Science at Carnegie-Melon University**, equipping him with a strong tech foundation.
   - He has significant experience, having previously worked with **Apple's AR Kit team** and contributed to over **50 iOS projects**.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Hammer Handle Just Got Better**: The hammer handle has undergone an update, introducing enhancements to its design and functionality. Expect numerous *exciting improvements* with this iteration.
   - This update signals the team's commitment to continuously improving the tool's usability.
- **Meet the Hammer2.0 Series Models**: The team has launched the **Hammer2.0 series models**, which include Hammer2.0-7b, Hammer2.0-3b, Hammer2.0-1.5b, and Hammer2.0-0.5b.
   - These models signify an important advancement in product diversification for development applications.
- **New Pull Request PR#667 Submitted**: A Pull Request (PR#667) has been submitted as part of the programmatic updates to the hammer product line. This submission is crucial to the ongoing development process.
   - The PR aims to incorporate recent enhancements and feedback from the community.



---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1289300567323316224)** (920 messages🔥🔥🔥): 

> - `LinkedIn and Open Source Issues`
> - `Fine-tuning Llama Models`
> - `Model Loading Issues`
> - `Using Unsloth with BitsAndBytes`
> - `Google Colab Usage` 


- **LinkedIn's Copied Code Controversy**: LinkedIn was accused of copying Unsloth's code without proper attribution, claiming it was self-derived. This led to intervention from Microsoft and GitHub, ultimately requiring LinkedIn to attribute Unsloth correctly.
   - This incident highlights concerns about intellectual property and the importance of adhering to open source licensing.
- **Best Practices for Fine-tuning Models**: It is recommended to set a random seed for reproducibility when fine-tuning models, as well as using a method to evaluate output quality thoroughly. Manual evaluation using a list of prompts is a suggested approach to provide insights into model performance.
   - Various parameters such as response format and context tuning can significantly impact the effectiveness of the fine-tuning process.
- **Model Loading Challenges**: Users encountered runtime errors related to model configuration files when attempting to load fine-tuned models using the Unsloth library. The issue was primarily due to having both LoRA adapters and base model configurations in the same repository.
   - It was recommended to upgrade the Unsloth library to resolve specific bugs related to model loading.
- **Using Unsloth with BitsAndBytes**: BitsAndBytes allows for the loading of models in quantized formats, with users able to load models in 4-bit or 8-bit configurations. While fine-tuning can be done in 4-bit, loading models in 16-bit post-training is recommended for better inference performance.
   - Users were advised to ensure they are using correct parameters to avoid confusion during model training and inference.
- **Getting Started with Google Colab**: New users were directed to resources for using Google Colab effectively, including links to notebooks with clear instructions. Several models were suggested for beginners to experiment with and explore functionality in a user-friendly format.
   - This ensures that newcomers can quickly acclimate to using the resources available for fine-tuning and deploying models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/CodeFryingPan/status/1840203597478539477">Tweet from FRYING PAN (@CodeFryingPan)</a>: I just quit my 270 000$ job at Coinbase to join the first YCombinator fall batch with my cofounder @not_nang.  We&#39;re building PearAI, an open source AI code editor. Think a better Copilot, or open...</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing>">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - a Hugging Face Space by ggml-org</a>: no description found</li><li><a href="https://x.com/RhysSullivan/status/1840461449371812289">Tweet from Rhys (@RhysSullivan)</a>: Introducing BlueberryAI, the open source AI powered code editor  It&#39;s a fork of PearAI, which is a fork of Continue, which is a fork of VSCode  Investors my DMs are open for the seed round</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth Documentation</a>: See the list below for all our GGUF, 16-bit and 4-bit bnb uploaded models</li><li><a href="https://github.com/linkedin/Liger-Kernel/commit/376fe0c2af65ff4d716dc36eb6fe5231662920a7">Reference Unsloth in header (#216) · linkedin/Liger-Kernel@376fe0c</a>: ## Summary
 Reference Unsloth in header section
 
 &amp;lt;!---
 ## Details
 This is an optional section; is there anything specific that reviewers
 should be aware of?
 ---&amp;gt;
 
 ## Testing Done...</li><li><a href="https://www.llama.com/docs/how-to-guides/fine-tuning">Fine-tuning | How-to guides</a>: Full parameter fine-tuning is a method that fine-tunes all the parameters of all the layers of the pre-trained model. </li><li><a href="https://www.youtube.com/watch?v=YZW3pkIR-YE">EASIEST Way to Fine-Tune LLAMA-3.2 and Run it in Ollama</a>: Meta recently released Llama 3.2, and this video demonstrates how to fine-tune the 3 billion parameter instruct model using Unsloth and run it locally with O...</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://github.com/bitsandbytes-foundation/">bitsandbytes foundation</a>: bitsandbytes foundation has 2 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes/multilingual/README.md">llama-recipes/recipes/multilingual/README.md at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081 · meta-llama/llama-recipes</a>: Scripts for fine-tuning Meta Llama with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization and Q&...</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/issues/1061">[FIXED] RuntimeError: Unsloth: Your repo has a LoRA adapter and a base model. · Issue #1061 · unslothai/unsloth</a>: I&#39;ve trained the unsloth/Llama-3.2-3B-Instruct-bnb-4bit model successfully, but when I try to use it with astLanguageModel.from_pretrained, I get this error: Traceback (most recent call last): Fil...</li><li><a href="https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py">trl/examples/scripts/sft_vlm.py at main · huggingface/trl</a>: Train transformer language models with reinforcement learning. - huggingface/trl</li><li><a href="https://github.com/PygmalionAI/aphrodite-engine">GitHub - PygmalionAI/aphrodite-engine: Large-scale LLM inference engine</a>: Large-scale LLM inference engine. Contribute to PygmalionAI/aphrodite-engine development by creating an account on GitHub.</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/discussions/1375">v0.44.0: New AdEMAMix optimizer, Embeddings quantization, and more! · bitsandbytes-foundation/bitsandbytes · Discussion #1375</a>: New optimizer: AdEMAMix The AdEMAMix optimizer is a modification to AdamW which proposes tracking two EMAs to better leverage past gradients. This allows for faster convergence with less training d...</li><li><a href="https://github.com/unslothai/unsloth/issues/421">config.json file not found, fine tuning llama3 with unsloth, after saving the file to hugging face  · Issue #421 · unslothai/unsloth</a>: i use unsloth to fine tune llama 3-8B..., after traning complete i save this model to hugging face by using &#39;push_to_hub&#39;, but it shows these files : .gitattributes README.md adapter_config.js...</li><li><a href="https://github.com/unslothai/unsloth/issues/1062">[TEMP FIX] Ollama / llama.cpp: cannot find tokenizer merges in model file [duplicate] · Issue #1062 · unslothai/unsloth</a>: Hi, i tried finetuning both llama 3.1-8b-instruct and llama 3-8b-instruct following the notebook you provided here. The training phase completed without errors and i generated the gguf quantized at...</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.</li><li><a href="https://github.com/asmith26/unsloth/blob/main/KTO_%2B_Phi_3_Mini_4K_Instruct_%2B_Unsloth.ipynb">unsloth/KTO_+_Phi_3_Mini_4K_Instruct_+_Unsloth.ipynb at main · asmith26/unsloth</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - asmith26/unsloth</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/huggingface/trl/issues/862">Compute metrics for generation tasks in SFTTrainer · Issue #862 · huggingface/trl</a>: Hi, I want to include a custom generation based compute_metrics e.g., BLEU, to the SFTTrainer. However, I have difficulties because: The input, eval_preds, into compute_metrics contains a .predicti...</li><li><a href="https://github.com/unslothai/unsloth/issues/1065">[TEMP FIX] Ollama / llama.cpp: cannot find tokenizer merges in model file · Issue #1065 · unslothai/unsloth</a>: Thank you for developing this useful resource. The Ollama notebook reports {&quot;error&quot;:&quot;llama runner process has terminated: error loading modelvocabulary: cannot find tokenizer merges in ...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1289349351524663318)** (17 messages🔥): 

> - `Compute utilization`
> - `Software acceleration methods`
> - `Underutilized hardware performance` 


- **Secret Compute Insights**: A member expressed that there is a significant amount of **compute power** being left untapped, emphasizing the potential for improvement across various hardware components.
   - *“If you'll allow me to dump something here without elaborating because it's secret privileged stuff”* hinted at undisclosed strategies to leverage this untapped power.
- **Impressive 4X Inference Acceleration**: Another member shared that they have achieved a **4X acceleration** in inference using standard **Python**, without resorting to complex hacks or proprietary methods.
   - This highlights how simple adjustments can yield significant performance boosts, indicating an unexplored potential for further improvements.
- **Massively Underutilized Hardware**: Discussion centered on the **CPU and GPU** being greatly underutilized, with claims that the system’s PCIe lanes are almost idle, indicating inefficiencies.
   - The idea is that even without hardware advancements, there's a clear pathway to achieving **10X performance**, strictly through integration of existing research.
- **Reactions to Performance Insights**: A humorous exchange noted that one member's insights sounded similar to an **OpenAI paper**, pointing out the cryptic nature of the information shared.
   - Jokes about not providing details and the formal tone, comparing it to a TED talk, characterized the reaction among the participants.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1289305839726760068)** (303 messages🔥🔥): 

> - `Model Fine-Tuning Issues`
> - `GGUF Conversion Problems`
> - `Tokenizer and EOS Token Issues`
> - `Checkpoint Management in Training`
> - `Using Unsloth with Llama Models` 


- **Challenges with Llama Model Fine-Tuning**: Users discussed various issues related to fine-tuning Llama models, particularly facing infinite token generation and retaining original capabilities.
   - Concerns were raised about the use of EOS tokens and model configurations that led to problems during inference.
- **Errors Encountered during GGUF Conversion**: One user faced an error stating, 'cannot find tokenizer merges in model file' after attempting to load a GGUF model post fine-tuning.
   - The conversation indicated that this issue might stem from problems during the model saving process to GGUF format.
- **Effectiveness of Different Training Approaches**: There were discussions on the effectiveness of using various rank values, targeted layers, and adding embedding layers during model fine-tuning.
   - Suggestions were made to use base models to avoid issues experienced by users using instruct models.
- **Checkpoint Management in Colab**: Users shared methods on how to manage checkpoints effectively during training to prevent losing progress in Google Colab.
   - There was emphasis on setting appropriate parameters for saving model checkpoints to mitigate runtime issues.
- **Compatibility of Different Llama Models**: It was clarified that the models 'meta-llama/Meta-Llama-3.1-8B' and 'unsloth/Meta-Llama-3.1-8B' are essentially the same and compatible.
   - Discussions also included the differences between Hugging Face's and Unsloth's model checkpoints and their compatibility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1179035537009545276/1179777624986357780/1290255706053939243">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://colab.research.google.com/drive/1oCEHcED15DzL8xXGU1VTx5ZfOJM8WY01?usp=sharing#scrollTo=6bZsfBuZDeCL">Google Colab</a>: no description found</li><li><a href="https://arxiv.org/abs/2409.12917">Training Language Models to Self-Correct via Reinforcement Learning</a>: Self-correction is a highly desirable capability of large language models (LLMs), yet it has consistently been found to be largely ineffective in modern LLMs. Existing approaches for training self-cor...</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">Finetuning from Last Checkpoint | Unsloth Documentation</a>: Checkpointing allows you to save your finetuning progress so you can pause it and then continue.</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto">Reward Modelling - DPO, ORPO &amp; KTO | Unsloth Documentation</a>: To use DPO, ORPO or KTO with Unsloth, follow the steps below:</li><li><a href="https://docs.unsloth.ai/basics/chat-templates">Chat Templates | Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/codelion/optillm/blob/main/optillm/cot_decoding.py">optillm/optillm/cot_decoding.py at main · codelion/optillm</a>: Optimizing inference proxy for LLMs. Contribute to codelion/optillm development by creating an account on GitHub.</li><li><a href="https://github.com/EricLBuehler/xlora">GitHub - EricLBuehler/xlora: X-LoRA: Mixture of LoRA Experts</a>: X-LoRA: Mixture of LoRA Experts. Contribute to EricLBuehler/xlora development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/1061">[FIXED] RuntimeError: Unsloth: Your repo has a LoRA adapter and a base model. · Issue #1061 · unslothai/unsloth</a>: I&#39;ve trained the unsloth/Llama-3.2-3B-Instruct-bnb-4bit model successfully, but when I try to use it with astLanguageModel.from_pretrained, I get this error: Traceback (most recent call last): Fil...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/tokenizer_utils.py#L813)">unsloth/unsloth/tokenizer_utils.py at main · unslothai/unsloth</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1289525835958714369)** (9 messages🔥): 

> - `Referee roles in LLM and finance`
> - `Liquid Foundation Models` 


- **Seeking Referee for LLM and Finance**: A member inquired about potential referees for a scientific journal focused on **LLM** and **finance/economy** topics.
   - The term 'referee' refers to a **reviewer** for the scientific journal in this context.
- **Liquid Foundation Models Launch**: A member shared a post from [LiquidAI](https://x.com/LiquidAI_/status/1840768716784697688) announcing the launch of **Liquid Foundation Models (LFMs)**, including **1B, 3B, and 40B models**.
   - However, skepticism arose regarding the validity of the claims, with one member expressing disappointment about the reports and questioning their accuracy, particularly mentioning issues with **Perplexity Labs**.



**Link mentioned**: <a href="https://x.com/LiquidAI_/status/1840768716784697688">Tweet from Liquid AI (@LiquidAI_)</a>: Today we introduce Liquid Foundation Models (LFMs) to the world with the first series of our Language LFMs: A 1B, 3B, and a 40B model. (/n)

  

---



### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1289969915960361072)** (1 messages): 

> - `Aider v0.58.0 Features`
> - `Architect/Editor Model Pairing`
> - `New Model Support`
> - `Session Enhancements`
> - `Clipboard Command Updates` 


- **Aider v0.58.0 brings exciting features**: The latest release, [Aider v0.58.0](https://aider.chat/2024/09/26/architect.html), introduces various enhancements including model pairing and new commands.
   - Noteworthy is that **Aider wrote 53%** of the code in this update, showcasing its automated capabilities.
- **Architect/Editor model pairing improves coding**: Users can now utilize a **strong reasoning model** like **o1-preview** as their Architect alongside a faster model like **gpt-4o** as their Editor.
   - *This pairing aims to optimize coding efficiency* while balancing performance and cost.
- **Expanded model support in Aider**: The update provides support for the new **Gemini 002** models and enhanced functionality for **Qwen 2.5** models.
   - These additions broaden the range of tools available to users for various applications.
- **Session enhancements make usage smoother**: Aider now allows users to skip many confirmation questions by selecting **(D)on't ask again**, enhancing user experience.
   - Moreover, the autocomplete for `/read-only` now supports the **entire filesystem**, making navigation more efficient.
- **Clipboard command updates streamline workflow**: The new `/copy` command enables users to copy the last LLM response to the clipboard, while `/clipboard` has been renamed to `/paste`.
   - In addition, **HTTP redirects** are now followed when scraping URLs, improving data retrieval in operations.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1289300553419198505)** (436 messages🔥🔥🔥): 

> - `Aider's Architect and Editor Models`
> - `Use of Multiple LLMs`
> - `DeepSeek Integration`
> - `Aider User Workflow`
> - `Prompt Configuration in Aider` 


- **Understanding Aider's Architect and Editor Models**: Aider operates with a main model and an optional editor model; architect mode utilizes the main model for planning and the editor model for execution.
   - Users can set `--editor-model` in their config file to designate the editor model, while the architect mode remains part of the main functionality.
- **Discussion on Multi-Agent Coding**: A user referenced two papers demonstrating effective multi-agent coding with LLMs, prompting inquiries about Aider's plans for similar features.
   - It was suggested to post these inquiries on GitHub for better visibility and potential integration.
- **DeepSeek's Role in Aider**: Users are encouraged to experiment with using DeepSeek as an editor model to reduce costs compared to more expensive options like o1-preview.
   - Recent updates have merged different DeepSeek models, creating some confusion regarding the specific model to use.
- **User Feedback and Recommendations**: Users noted that while LLMs like Sonnet can provide useful templates, issues were found with them producing irrelevant edits.
   - Responses pointed out that small, detailed tasks tend to yield better results when using LLMs for code editing.
- **Configuration and Command Syntax in Aider**: Users discussed the YAML configuration file settings for Aider, particularly in setting the appropriate models for tasks.
   - Command syntax for tasks and settings was clarified, reinforcing that Aider's flexibility allows for tailored user experiences.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/imports.html">Dependency versions</a>: aider is AI pair programming in your terminal</li><li><a href="https://docs.continue.dev/customize/deep-dives/codebase">@codebase | Continue</a>: Talk to your codebase</li><li><a href="https://alexgarcia.xyz/blog/2024/sqlite-lembed-init/index.html">Introducing sqlite-lembed: A SQLite extension for generating text embeddings locally</a>: Generate text embeddings in SQL with GGUF models!</li><li><a href="https://www.answer.ai/posts/2024-09-03-llmstxt.html">/llms.txt—a proposal to provide information to help LLMs use websites – Answer.AI</a>: We propose that those interested in providing LLM-friendly content add a /llms.txt file to their site. This is a markdown file that provides brief background information and guidance, along with links...</li><li><a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://aider.chat/2024/09/26/architect.html">Separating code reasoning and editing</a>: An Architect model describes how to solve the coding problem, and an Editor model translates that into file edits. This Architect/Editor approach produces SOTA benchmark results.</li><li><a href="https://aider.chat/">Home</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>: How to configure aider with a yaml config file.</li><li><a href="https://aider.chat/docs/install.html">Installation</a>: How to install and get started pair programming with aider.</li><li><a href="https://aider.chat/docs/config/options.html#--editor-model-editor_model">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://github.com/sigoden/aichat/wiki/RAG-Guide">RAG Guide</a>: All-in-one AI CLI tool featuring Chat-REPL, Shell Assistant, RAG, AI tools &amp; agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more. - sigoden/aichat</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://huggingface.co/Kortix/FastApply-v1_16bit_Qwen2.5-Coder-1.5B-ft">Kortix/FastApply-1.5B-v1_16bit_Qwen2.5-Coder-1.5B-ft · Hugging Face</a>: no description found</li><li><a href="https://aider.chat/docs/config/options.html#--editor-edit-format-editor_edit_format">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://aider.chat/docs/config/options.html#--editor-">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://rentry.org/aiderts">JavaScript / TypeScript in aider</a>: Background aider is a powerful AI programming assistant, which brings its own linter system, but advanced JS/TS template languages such as JSX/TSX or Svelte allow multiple different languages in one f...</li><li><a href="https://aider.chat/docs/config/options.html#--deepseek">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/benchmark/README.md">aider/benchmark/README.md at main · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/models/deepseek/deepseek-chat">DeepSeek V2.5 - API, Providers, Stats</a>: DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. Run DeepSeek V2.5 with API</li><li><a href="https://github.com/asg017/sqlite-vec?tab=readme-ov-file">GitHub - asg017/sqlite-vec: A vector search SQLite extension that runs anywhere!</a>: A vector search SQLite extension that runs anywhere! - asg017/sqlite-vec</li><li><a href="https://github.com/paul-gauthier/aider/issues/1818">Feature request: add --external-chat switch to allow Aider to receive individual messages written in a custom text editor · Issue #1818 · paul-gauthier/aider</a>: Issue Description The --external-chat &lt;text_editor_path&gt; switch would allow Aider to receive individual message written in a custom text editor. The --external-chat switch alone would imply read...</li><li><a href="https://github.com/paul-gauthier/aider/issues/1315?">Addition of `/editor` command? · Issue #1315 · paul-gauthier/aider</a>: Issue Over at https://github.com/llm-workflow-engine/llm-workflow-engine I&#39;ve implemented a very handy /editor command that: Opens the CLI editor as specified in the $EDITOR environment variable, ...</li><li><a href="https://www.firecrawl.dev/">Firecrawl</a>: Turn any website into LLM-ready data.</li><li><a href="https://github.com/paul-gauthier/aider/">GitHub - paul-gauthier/aider: aider is AI pair programming in your terminal</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider/issues/1839">how to add a multi-agent flow ? · Issue #1839 · paul-gauthier/aider</a>: what is the plan of aider.chat regarding multi-agent coding? there are two papers https://arxiv.org/pdf/2405.11403v1 and https://arxiv.org/pdf/2402.16906v6 that use multiple llm calls (and debugger...</li><li><a href="https://github.com/paul-gauthier/aider/pull/1790">feat: add cmd_copy command to copy last assistant reply to clipboard by fry69 · Pull Request #1790 · paul-gauthier/aider</a>: This adds the /copy command to copy the last reply from the LLM to the clipboard. Note: flake8 forced the /paste description to be cut and put into two lines, as the line was longer than 100 chars.</li><li><a href="https://github.com/paul-gauthier/aider/commit/c2c4dbd2a8319f3eab72939f60e2b199a452ff1d">Merge pull request #1595 from jbellis/paste · paul-gauthier/aider@c2c4dbd</a>: feat: rename /clipboard to /paste</li><li><a href="https://github.com/paul-gauthier/aider/actions/workflows/docker-build-test.yml">Docker Build Test · Workflow runs · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/fry69/aider/tree/copy-command">GitHub - fry69/aider at copy-command</a>: aider is AI pair programming in your terminal. Contribute to fry69/aider development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider.git">GitHub - paul-gauthier/aider: aider is AI pair programming in your terminal</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://paperswithcode.com/sota/code-generation-on-humaneval">Papers with Code - HumanEval Benchmark (Code Generation)</a>: The current state-of-the-art on HumanEval is LDB (O1-mini, based on seed programs from Reflexion). See a full comparison of 138 papers with code.</li><li><a href="https://platform.deepseek.com/api-docs/updates/#version-2024-09-05">Change Log | DeepSeek API Docs</a>: Version: 2024-09-05</li><li><a href="https://github.com/paul-gauthier/aider/blob/0aaa37f528b6b8851fa35859cdb401cb71addde1/aider/args.py#L217">aider/aider/args.py at 0aaa37f528b6b8851fa35859cdb401cb71addde1 · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/okwilkins/rag-cli">GitHub - okwilkins/rag-cli: A project to show good CLI practices with a fully fledged RAG system.</a>: A project to show good CLI practices with a fully fledged RAG system. - okwilkins/rag-cli</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/">RAG CLI - LlamaIndex</a>: no description found</li><li><a href="https://github.com/paul-gauthier/aider/pull/1823">doc: hotfix for Full results table by fry69 · Pull Request #1823 · paul-gauthier/aider</a>: Hotfix for this problem:  (via -&gt; https://discord.com/channels/1131200896827654144/1131200896827654149/1289976901393453066) Fixed version:</li><li><a href="https://github.com/paul-gauthier/aider/issues/1315">Addition of `/editor` command? · Issue #1315 · paul-gauthier/aider</a>: Issue Over at https://github.com/llm-workflow-engine/llm-workflow-engine I&#39;ve implemented a very handy /editor command that: Opens the CLI editor as specified in the $EDITOR environment variable, ...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1289331299584180235)** (192 messages🔥🔥): 

> - `Aider Configuration`
> - `Architect Mode vs Code Mode`
> - `Cost Efficiency of Models`
> - `Using Multiple Git Worktrees`
> - `Prompt Caching and Token Management` 


- **Understanding Aider Configuration Files**: Users discussed the possibility of using multiple `.aider.conf.yml` files to manage configurations, with suggestions to script Aider for better flexibility.
   - There was a debate on whether scripting is necessary or if well-structured config files suffice for managing Aider effectively.
- **Architect Mode Spitting Code**: Concerns were raised about the Architect mode producing final code outputs instead of just planning, which led to confusion on its utility.
   - It was clarified that for simple tasks, the planning step may be unnecessary, which can lead to wasted tokens.
- **Cost Efficiency Using Different Models**: Using `claude-3.5-sonnet` as architect and `deepseek v2.5` as editor was noted to be significantly cheaper, with estimates suggesting a 20x-30x cost reduction for editor tokens.
   - Discussion highlighted the potential savings when using models with different pricing structures and functionalities.
- **Using Multiple Git Worktrees**: Participants suggested leveraging multiple git worktrees to work on several issues concurrently, along with managing Aider instances for better productivity.
   - The approach of working across separate terminals or branches was seen as a way to offset the waiting times associated with using slower models.
- **Prompt Caching and Token Management**: The effectiveness and utility of prompt caching within Aider were debated, focusing on whether it truly offers cost savings or complicates the process.
   - Keepalive pings were discussed as a means to maintain cache without excessive costs, highlighting the need to balance interaction timing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/aider-not-found.html">Aider not found</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/images-urls.html#web-pages">Images &amp; web pages</a>: Add images and web pages to the aider coding chat.</li><li><a href="https://aider.chat/docs/usage/modes.html#chat-modes">Chat modes</a>: Using the chat, ask and help chat modes.</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model">Chat modes</a>: Using the chat, ask and help chat modes.</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-includ">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://simonwillison.net/2024/Feb/21/gemini-pro-video/">The killer app of Gemini Pro 1.5 is video</a>: Last week Google introduced Gemini Pro 1.5, an enormous upgrade to their Gemini series of AI models. Gemini Pro 1.5 has a 1,000,000 token context size. This is huge—previously that …</li><li><a href="https://aider.chat/docs/usage/tutorials.html">Tutorial videos</a>: Intro and tutorial videos made by aider users.</li><li><a href="https://aider.chat/docs/config/options.html">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://aider.chat/docs/faq.html">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/more-info.html">More info</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: Tell aider to follow your coding conventions when it works on your code.</li><li><a href="https://aider.chat/docs/config/options.html#--chat-language-chat_language">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching">Prompt Caching (beta) - Anthropic</a>: no description found</li><li><a href="https://aider.chat/docs/usage/caching.html">Prompt caching</a>: Aider supports prompt caching for cost savings and faster coding.</li><li><a href="https://youtu.be/W6Z0U11nnhA?si=VUCa3iHKWy3-L9vH">BEST Prompt Format: Markdown, XML, or Raw? CONFIRMED on Llama 3.1 &amp; Promptfoo</a>: Which prompt format is BEST for your AI agents? Is it Markdown, XML, or Raw Prompts?🚀 Ready to unlock the true potential of your AI agents? In this video, w...</li><li><a href="https://github.com/PierrunoYT/gemini-youtube-analyzer">GitHub - PierrunoYT/gemini-youtube-analyzer</a>: Contribute to PierrunoYT/gemini-youtube-analyzer development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5">Gemini Pro 1.5 - API, Providers, Stats</a>: Google&#x27;s latest multimodal model, supporting image and video in text or chat prompts.  Optimized for language tasks including:  - Code generation - Text generation - Text editing - Problem solvin...</li><li><a href="https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding">GitHub - yunlong10/Awesome-LLMs-for-Video-Understanding: 🔥🔥🔥Latest Papers, Codes and Datasets on Vid-LLMs.</a>: 🔥🔥🔥Latest Papers, Codes and Datasets on Vid-LLMs. Contribute to yunlong10/Awesome-LLMs-for-Video-Understanding development by creating an account on GitHub.</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://github.com/paul-gauthier/aider/issues/1815#issuecomment-2381264243">Feature request - templates for aider · Issue #1815 · paul-gauthier/aider</a>: Issue Some inspiration could be taken from https://github.com/simonw/llm by Simon Willison, his LLM tool allows the creation of plugins (#1814) . He also has a prompt template system that allows us...</li><li><a href="https://youtu.be/pcC4Dr6Wj2Q?si=z5la0QllNsnLqY9F">Deno 2 is here… will it actually kill Node.js this time?</a>: Take a first look at Deno 2.0 - a JavaScript runtime with first-class TypeScript support, and now full compatibility with Node.js#javascript #tech #thecodere...</li><li><a href="https://console.groq.com/docs/rate-limits">GroqCloud</a>: Experience the fastest inference in the world
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1289650742566584401)** (16 messages🔥): 

> - `NotebookLM audio feature`
> - `Aider updates`
> - `AI podcast summarization`
> - `Content creation automation`
> - `Hiring decision` 


- **NotebookLM Announces Custom Podcast Feature**: Google's [NotebookLM](https://notebooklm.google/) now offers a unique audio feature that generates custom podcasts using provided content, featuring AI hosts discussing the material.
   - An example podcast highlights its engaging format, lasting around ten minutes and showcasing an astonishingly convincing conversation among the hosts.
- **Exciting Updates to Aider Tools**: Recent YouTube videos detail significant updates to Aider, with one titled ["NEW Aider Architect & Editor Updates"](https://www.youtube.com/watch?v=8jD8dAXq8jE) showcasing features like the AI coding agent and Beast Cursor.
   - Another video discusses the enhancements in Aider's Architect Mode, supporting **Gemini-002**, and emphasizes how quickly content creators are producing these videos.
- **Discussion on AI-Powered Podcast Summarization**: There is a conversation about needing an AI to listen to and summarize countless new podcasts into listicles, with the suggestion that this could be the next big project.
   - One member mused about creating a podcast titled *"Today in Coding AI News"* to further consolidate the content.
- **Automation of Release Notes into Audio**: A proposal was made to automate the creation of videos from release notes and source code using NotebookLM's capabilities, potentially streamlining content generation.
   - The idea is to release a tool called *ReleaseNotesLM* that would transform written updates into audio format with minimal effort.
- **Hiring Decision Based on Video Quality**: After reviewing content, a member stated they have decided to hire an individual who was positively highlighted in a previous discussion.
   - The hiring decision reflects the impact of the speaker’s impressive presentation skills and content depth.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://simonwillison.net/2024/Sep/29/notebooklm-audio-overview/">NotebookLM’s automatically generated podcasts are surprisingly effective</a>: Audio Overview is a fun new feature of Google’s NotebookLM which is getting a lot of attention right now. It generates a one-off custom podcast against content you provide, where …</li><li><a href="https://www.youtube.com/watch?v=8jD8dAXq8jE">NEW Aider Architect &amp; Editor Updates Are INSANE!🤖(Beast Cursor?!?) Best AI Coding Agent?! OpenAI o1</a>: NEW Aider Architect &amp; Editor Updates Are INSANE!🤖(Beast Cursor?!?) Best AI Coding Agent?!? OpenAI o1https://aider.chat/https://github.com/paul-gauthier/aide...</li><li><a href="https://www.youtube.com/watch?v=OPXslklVBZc">Aider (Upgraded) : This Coding Agent just got BETTER with Architect Mode, Gemini-002 Support &amp; More!</a>: Join this channel to get access to perks:https://www.youtube.com/@AICodeKing/joinIn this video, I&#39;ll be telling you about the new upgrades to Aider which is ...</li><li><a href="https://github.com/sigoden/aichat/wiki/RAG-Guide">RAG Guide</a>: All-in-one AI CLI tool featuring Chat-REPL, Shell Assistant, RAG, AI tools &amp; agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more. - sigoden/aichat
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1289317372829761537)** (464 messages🔥🔥🔥): 

> - `AI Model Merging`
> - `Text Similarity in AI`
> - `Stable Diffusion Performance`
> - `Video Model Development`
> - `Hugging Face Community Projects` 


- **Exploring AI Model Merging Techniques**: Users discussed different methods of merging AI models, including PEFT merge and the DARE method, highlighting their effectiveness in enhancing model performance.
   - The conversation emphasized the challenges of training LLMs from scratch and the usefulness of existing models for fine-tuning specific tasks.
- **The Importance of Text Similarity in AI**: Participants debated how AI models recognize text similarity, with examples like 'I have a car' and 'I own a car' showcasing the need for datasets to teach these nuances.
   - Understanding text similarity is crucial for improving AI interaction quality and requires comprehensive datasets for effective training.
- **Discussion on Stable Diffusion and Its Operating Environment**: Members compared the advantages of running Stable Diffusion on Windows vs. WSL, noting the influence of GPU drivers on performance.
   - The topic highlighted preferences for operating systems in the context of resource-intensive AI applications.
- **Emerging Trends in Video Model Development**: There was excitement about new video models being developed, with users sharing links to innovative projects like 'S3Diff' and updates on existing models.
   - Participants expressed enthusiasm for advancements in video processing capabilities and the potential of upcoming models.
- **Concerns Regarding AI Model Performance**: Users shared frustrations about perceived declines in performance of models like ChatGPT O1 compared to earlier versions, citing issues with reasoning and simplicity.
   - The discussions reflected concerns over model updates and the impact of censoring or changes on AI usability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sambanova.ai>">no title found</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1ekNDPjC3CKWWd3jd2_V9QGTJSbvHKIZ2">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/time-series-transformers.ipynb">Google Colab</a>: no description found</li><li><a href="https://tenor.com/view/jizz-adult-swim-john-reilly-blink-gif-14841420">Jizz Adult Swim GIF - Jizz Adult Swim John Reilly - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/spaces/jasperai/Flux.1-dev-Controlnet-Upscaler">Flux.1-dev Upscaler - a Hugging Face Space by jasperai</a>: no description found</li><li><a href="https://huggingface.co/KingNish/Qwen2.5-0.5b-Test-ft">KingNish/Qwen2.5-0.5b-Test-ft · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/merge_loras#addweightedadapter">Merge LoRAs</a>: no description found</li><li><a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/time-">Google Colab</a>: no description found</li><li><a href="https://console.groq.com/">GroqCloud</a>: Experience the fastest inference in the world</li><li><a href="https://x.com/NousResearch/status/1840505804673225031">Tweet from Nous Research (@NousResearch)</a>: OPEN SOURCE LIVES ON #SB1047 DEFEATED</li><li><a href="https://cloud.google.com/translate/docs/reference/rest/?apix=true">no title found</a>: no description found</li><li><a href="https://pypi.org/project/starlette-session-middleware/">starlette-session-middleware</a>: None</li><li><a href="https://huggingface.co/spaces?search=Video%20editor">Spaces - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/lmsys/chatbot_arena_conversations">lmsys/chatbot_arena_conversations · Datasets at Hugging Face</a>: no description found</li><li><a href="https://hf.co/papers/2311.03099">Paper page - Language Models are Super Mario: Absorbing Abilities from Homologous
  Models as a Free Lunch</a>: no description found</li><li><a href="https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k?row=0">HuggingFaceTB/everyday-conversations-llama3.1-2k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/ArcticHare105/S3Diff">GitHub - ArcticHare105/S3Diff: Official implementation of S3Diff</a>: Official implementation of S3Diff. Contribute to ArcticHare105/S3Diff development by creating an account on GitHub.</li><li><a href="https://github.com/xtekky/gpt4free">GitHub - xtekky/gpt4free: The official gpt4free repository | various collection of powerful language models</a>: The official gpt4free repository | various collection of powerful language models - xtekky/gpt4free</li><li><a href="https://huggingface.co/datasets/Langame/conversation-starters">Langame/conversation-starters · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/developer_guides/model_merging">Model merging</a>: no description found</li><li><a href="https://github.com/interneuron-ai/project-barbarossa">GitHub - interneuron-ai/project-barbarossa</a>: Contribute to interneuron-ai/project-barbarossa development by creating an account on GitHub.</li><li><a href="https://huggingface.co/interneuronai/az-llama2">interneuronai/az-llama2 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1289314495260524627)** (14 messages🔥): 

> - `Experiments with CUDA`
> - `Gradio frustrations`
> - `Model Policy Loss`
> - `Interface Design Issues` 


- **CUDA experiments yield new insights**: A member shared their progress working with **CUDA** and **7b FP8**, noting a typo indicating a **bfloat16** with **fp32 master weights**.
   - They reflected on their learning over the last two days, indicating significant technical growth.
- **Gradio underwhelms users**: Members expressed strong dissatisfaction with **Gradio**, one member passionately stating it is 'hot garbage' and a waste of time.
   - They relayed frustrations about its design, which turns complex projects into 'tangled balls of spaghetti code' with UI responsiveness issues.
- **Encouragement for Gradio support**: In response to Gradio frustrations, members encouraged seeking support in dedicated channels for issues regarding **Gradio** functionality.
   - One member offered help in a supportive tone, indicating a community-oriented approach to solving problems.
- **Insights on model performance**: Community discussions highlighted a member's satisfaction with the **policy loss** of a model, noting that their loss looks 'good'.
   - This was framed positively amidst broader conversations about ongoing technical challenges.
- **Exploring alternatives to Gradio**: A member indicated their intention to pursue **NiceGUI** as an alternative to Gradio, citing significant design flaws in the latter.
   - They expressed disappointment but maintained enthusiasm for **Hugging Face** projects they enjoy, like **Accelerate**.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1289373629817032724)** (9 messages🔥): 

> - `Medical AI Paper Highlights`
> - `HuggingFace Model Popularity Metrics`
> - `Projection Mapping Technology`
> - `Experiences with Phi Models`
> - `Video Mapping Techniques` 


- **Last Week in Medical AI Highlights**: A recent post highlighted the **top research papers and models in medical AI** for the week of September 21 - 27, 2024, featuring significant studies like *A Preliminary Study of o1 in Medicine*.
   - Community members suggested enhancing visibility by splitting this content into individual blog posts focusing on the coolest papers.
- **HuggingFace Model Popularity Metrics Capture**: A Reddit thread discussed a metric that quantifies the **most actively liked models on HuggingFace**, accounting for their duration on the platform to avoid bias towards older or newer models.
   - One user proposed a pull request to improve the OpenLLM leaderboard’s like count updates, mentioning how it relates to the HuggingFace trending section.
- **Exploring Projection Mapping Technology**: An article on projection mapping described how this artistic video technique transforms surfaces into dynamic displays, creating immersive experiences.
   - It discusses the benefits for businesses and offers insights into how video mapping works in enhancing creativity and engagement.
- **Struggles with Phi Models**: A user expressed frustration with their experience using **Phi 3**, noting that their tests have not gone well and questioning the performance of Phi 2 in comparison.
   - This ongoing discussion reflects concerns within the community regarding the efficacy and usability of the different Phi versions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/192ly8c/which_models_are_the_most_actively_liked_on/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.mediacraft.video/posts/projection-mapping/">Projection Mapping - Artistic Video Content &amp; Visual Illusion</a>: Welcome to the fascinating world of projection mapping - a cutting-edge technology that brings art and visuals to life. Learn how projection mapping works, its benefits for businesses, and explore exa...</li><li><a href="https://x.com/OpenlifesciAI/status/1840020394880667937">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: Last Week in Medical AI: Top Research Papers/Models 🏅(September 21  - September 27, 2024)  🏅 Medical AI Paper of the week A Preliminary Study of o1 in Medicine: Are We Closer to an AI Doctor?  Autho...</li><li><a href="https://huggingface.co/blog/aaditya/medicalai-weekly-papers-2127">Last Week in Medical AI: Top Research Papers/Models 🏅 (September 21 - September 27, 2024) </a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1289323201616281670)** (29 messages🔥): 

> - `Flux-Schnell Demo`
> - `Qwen 2.5 Fine-tuning`
> - `Instrumentum AI Summarizer`
> - `Deepseek-Chat CoT Mode`
> - `MusicGen Continuations App` 


- **Flux-Schnell Demo for Regional Prompt Attention**: A demo has been developed for **Flux-Schnell** focusing on regional prompt attention, with plans to add source code and comfyui node later.
   - There is anticipation around how these enhancements will further improve user experience.
- **Fine-tuning Qwen 2.5 Model**: A user shared their experience finetuning **Qwen 2.5 0.5b** with a **Magpie 300k Dataset**, achieving answer quality comparable to larger models like **Llama 3.2 1b**.
   - The user noted some **inconsistencies** but is working on addressing these issues and invites feedback on their ongoing work.
- **Introducing Instrumentum AI Summarizer**: The **Instrumentum** AI summarizer offers no length restrictions and is aimed at quick document summarization using advanced LLMs.
   - Key features include full security for document uploads and competitive pricing, designed to enhance productivity.
- **Deepseek-Chat with Chain of Thought Visualization**: The **Deepseek-Chat** mode introduces optional **Chain of Thought** visualization for transparent reasoning with step-by-step visualization.
   - This innovation aims to enhance user understanding of the model's reasoning process through a Streamlit-powered UI.
- **iOS App for MusicGen Continuations**: An iOS app focusing on **MusicGen continuations** using beatboxes as input audio is under development, with a forthcoming app store release.
   - The app features noise cancellation and aims to provide improved output when capturing drum inputs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.g-diffuser.com/dualdiffusion/">DualDiffusion Demo Audio</a>: DualDiffusion Demo Audio</li><li><a href="https://forbo7.github.io/forblog/posts/21_reflecting_on_my_internships.html">What I Learned during my Second and Third Internships – ForBo7 // Salman Naqvi</a>: You Learn by Doing</li><li><a href="https://x.com/thepatch_kev/status/1840536425776763020">Tweet from thecollabagepatch (@thepatch_kev)</a>: day 4  ios app for musicgen continuations  landing screen, noise cancel for input audio and a &#39;tame the gary&#39; toggle that sort of works  focuses it on drums and tries harder to incorporate inp...</li><li><a href="https://huggingface.co/spaces/qamarsidd/SentimentReveal">SentimentReveal - a Hugging Face Space by qamarsidd</a>: no description found</li><li><a href="https://chromewebstore.google.com/detail/arxiv-insight-paper-summa/iciiagolkeidemjnbobbkcjfndkabicf">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.</li><li><a href="https://github.com/vietanhdev/llama-assistant">GitHub - vietanhdev/llama-assistant: AI-powered assistant to help you with your daily tasks, powered by Llama 3.2. It can recognize your voice, process natural language, and perform various actions based on your commands: summarizing text, rephasing sentences, answering questions, writing emails, and more.</a>: AI-powered assistant to help you with your daily tasks, powered by Llama 3.2. It can recognize your voice, process natural language, and perform various actions based on your commands: summarizing ...</li><li><a href="https://instrumentum.ai/en">Welcome | Instrumentum</a>: no description found</li><li><a href="https://github.com/U-C4N/Deepseek-CoT/">GitHub - U-C4N/Deepseek-CoT: Deepseek-CoT</a>: Deepseek-CoT. Contribute to U-C4N/Deepseek-CoT development by creating an account on GitHub.</li><li><a href="https://docs.google.com/spreadsheets/d/1DlBT1pF8-zMECntRWXFsL46gZyvNp1BJlJ6LXGze4dA/edit?gid=0#gid=0">discord AI sphere - share  with whoever!</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1289619330899640437)** (5 messages): 

> - `OmDet-Turbo model`
> - `Keypoint Detection Task`
> - `SuperPoint Model`
> - `Fine-tuning TroCR Models`
> - `Upcoming Models for Keypoint Detection` 


- **OmDet-Turbo Model Launch**: The team announced the addition of support for the **OmDet-Turbo** model, enhancing zero-shot object detection capabilities in real-time, inspired by Grounding DINO and OWLv2 via [RT-DETR](https://www.linkedin.com/posts/yoni-gozlan_ai-artificialintelligence-objectdetection-ugcPost-7244768044533657603-FDOT?utm_source=share&utm_medium=member_desktop).
   - This significant update aims to improve AI performances in various object detection tasks.
- **Keypoint Detection Task Page Released**: A new **keypoint-detection** task page has been introduced, now featuring support for **SuperPoint**, pivotal for interest point detection and description. Detailed information can be found in their documentation [here](https://huggingface.co/docs/transformers/v4.45.1/en/model_doc/superpoint).
   - SuperPoint showcases a self-supervised training framework that is applicable to homography estimation and image matching.
- **Community Eager for More Models**: Community interest is growing in keypoint detection as users express excitement for future model integrations like **LoFTR**, **LightGlue**, and **OmniGlue**.
   - The anticipation highlights the community's engagement and expectation for advancements in this area of computer vision.
- **Fine-tuning TroCR Models Discussion**: A user raised a question regarding whether to fine-tune with their dataset using '**trocr-large-stage1**' (base) or '**trocr-large-handwriting**' (already fine-tuned on the IAM dataset).
   - *They inquired if fine-tuning a fine-tuned model yields better performance.*



**Link mentioned**: <a href="https://huggingface.co/docs/transformers/v4.45.1/en/model_doc/superpoint">SuperPoint</a>: no description found

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1290122188661329930)** (2 messages): 

> - `Hallucination Detection Model`
> - `Fine-tuning BERT on Yelp Dataset` 


- **Hallucination Detection Model Released**: A new **Hallucination Detection Model** has been released, upgraded from the **Phi-3** to **Phi-3.5 base**, focusing on evaluating language model outputs for hallucinations.
   - Key performance metrics include **Precision: 0.77**, **Recall: 0.91**, and **F1 Score: 0.83**, with overall accuracy hitting **82%**; [view the model card here](https://huggingface.co/grounded-ai/phi3.5-hallucination-judge).
- **Seeking Help on Fine-tuning BERT**: A member is looking for resources to fine-tune **BERT** on the **Yelp review dataset** with five classes, expressing concerns about achieving accuracy in the **60s**.
   - They specifically requested a list of current state-of-the-art models and performance metrics on the Yelp dataset, noting the lack of recent updates on the Paperswithcode website.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1289619614942232638)** (6 messages): 

> - `GitHub API usage`
> - `Stack Overflow for Developers`
> - `Increased Context LLaMA Model Conversion`
> - `llama.cpp compatibility` 


- **GitHub API query sparks off-topic discussion**: A member asked about using the GitHub API to find rebased commits, but another pointed out this channel focuses on diffusion models.
   - Despite the off-topic nature, it was suggested that the question could be resolved through [Stack Overflow](https://stackoverflow.com/) or GitHub Copilot.
- **Stack Overflow remains essential for devs**: A member emphasized that every developer keeps a tab on [Stack Overflow](https://stackoverflow.com/) for solutions and knowledge-sharing.
   - They noted that Stack Overflow is now offering a suite of GenAI tools for Teams to improve knowledge connection among employees.
- **Troubles with LLaMA model conversion**: A user shared their struggle in converting the **LLaMA-2-7B-32K** model to GGUF format and sought help regarding compatibility with llama.cpp.
   - They provided a detailed traceback of the error encountered, highlighting an `IndexError` during the vocabulary setting phase.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/">Stack Overflow - Where Developers Learn, Share, &amp; Build Careers</a>: Stack Overflow | The World&#x2019;s Largest Online Community for Developers</li><li><a href="https://huggingface.co/togethercomputer/LLaMA-2-7B-32K">togethercomputer/LLaMA-2-7B-32K · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1289301764708761685)** (363 messages🔥🔥): 

> - `Issue with downloading models in LM Studio`
> - `Using vision-enabled models in LM Studio`
> - `Feature requests for LM Studio`
> - `Concerns about model performance and claims`
> - `Discussion about query queueing and caching` 


- **Challenges in downloading and sideloading models**: Users discussed issues with downloading models in LM Studio, especially when using VPNs, leading some to sideload models instead.
   - The platform's limitations were acknowledged, specifically regarding supported model formats like safetensors and GGUF.
- **Vision-enabled models in LM Studio**: It was clarified that LM Studio currently does not support llama-3.2-11B vision models due to compatibility issues with llama.cpp.
   - Participants raised questions about the broader availability of multimodal models and their functionality within the platform.
- **Feature requests and future plans**: Users expressed interest in features like query queueing and caching of edits, with some finding existing requests in the feature tracker.
   - There was no published roadmap for upcoming features, leaving some topics like 3D generation in uncertain territory.
- **Concerns about model performance and credibility**: The community discussed new models like LiquidAI and Replete, weighing their performance claims against established options like Qwen-2.5.
   - Debate centered around the reliability and testing accessibility of these models, with some expressing skepticism about their marketing hype.
- **User inquiries about loading times in LM Studio**: A user reported experiencing a significant loading time, even when models were fully loaded into VRAM, causing a delay before evaluation.
   - The issue prompted discussions on potential reasons behind the initial loading times observed in the application.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=41323042">no title found</a>: no description found</li><li><a href="https://huggingface.co/spaces/SixOpen/Florence-2-large-ft">Florence 2 Large Ft - a Hugging Face Space by SixOpen</a>: no description found</li><li><a href="https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct">allenai/OLMoE-1B-7B-0924-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/en/training">Fine-tune a pretrained model</a>: no description found</li><li><a href="https://www.liquid.ai/liquid-foundation-models#join-us-as-an-early-adopter-of-LFMs)">Liquid Foundation Models: Our First Series of Generative AI Models</a>: Announcing the first series of Liquid Foundation Models (LFMs) – a new generation of generative AI models that achieve state-of-the-art performance at every scale, while maintaining a smaller memory f...</li><li><a href="https://huggingface.co/collections/Replete-AI/replete-llm-v25-66f987583df3ae3b18cf3c84">Replete-LLM-V2.5 - a Replete-AI Collection</a>: no description found</li><li><a href="https://huggingface.co/lmstudio-community">lmstudio-community (LM Studio Community)</a>: no description found</li><li><a href="https://huggingface.co/blog/llama31#inference-memory-requirements">Llama 3.1 - 405B, 70B &amp; 8B with multilinguality and long context</a>: no description found</li><li><a href="https://huggingface.co/mylesgoose/Llama-3.2-11B-Vision-Instruct">mylesgoose/Llama-3.2-11B-Vision-Instruct · Hugging Face</a>: no description found</li><li><a href="https://www.liquid.ai/liquid-foundation-models#join-us-as-an-early-adopter-of-LFM">Liquid Foundation Models: Our First Series of Generative AI Models</a>: Announcing the first series of Liquid Foundation Models (LFMs) – a new generation of generative AI models that achieve state-of-the-art performance at every scale, while maintaining a smaller memory f...</li><li><a href="https://huggingface.co/microsoft/Florence-2-large">microsoft/Florence-2-large · Hugging Face</a>: no description found</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta Releases</a>: LM Studio Beta Releases</li><li><a href="https://lmstudio.ai/docs/basics/chat#faq">Manage chats - Running LLMs Locally | LM Studio Docs</a>: Manage conversation threads with LLMs</li><li><a href="https://lmstudio.ai/model/llama-3.2-">Model not found</a>: no description found</li><li><a href="https://github.com/YorkieDev/lmstudioservercodeexamples">GitHub - YorkieDev/lmstudioservercodeexamples: This readme contains server code examples from LM Studio v0.2.31</a>: This readme contains server code examples from LM Studio v0.2.31 - YorkieDev/lmstudioservercodeexamples</li><li><a href="https://lmstudio.ai/model/stable-code-instruct-3b.">Model not found</a>: no description found</li><li><a href="https://lmstudio.ai/model/llama-3.2-1b-instruct">Llama 3.2 1B</a>: llama • Meta • 1B</li><li><a href="https://github.com/openai/openai-python?tab=readme-ov-file#vision">GitHub - openai/openai-python: The official Python library for the OpenAI API</a>: The official Python library for the OpenAI API. Contribute to openai/openai-python development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fqk9ky/i_trained_mistral_on_the_us_armys_field_manuals/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/commit/9a913110cf471a8287ac06c43cbe307d3cf6df99">llama : add support for Chameleon (#8543) · ggerganov/llama.cpp@9a91311</a>: * convert chameleon hf to gguf
 
 * add chameleon tokenizer tests
 
 * fix lint
 
 * implement chameleon graph
 
 * add swin norm param
 
 * return qk norm weights and biases to original format
 
 ...</li><li><a href="https://huggingface.co/mistralai/Pixtral-12B-2409#usage-examples">mistralai/Pixtral-12B-2409 · Hugging Face</a>: no description found</li><li><a href="https://github.com/meta-llama/llama-models/tree/main">GitHub - meta-llama/llama-models: Utilities intended for use with Llama models.</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1289306614511308832)** (138 messages🔥🔥): 

> - `NVIDIA Jetson AGX Thor`
> - `3090 vs 3090 Ti vs P40 comparisons`
> - `Market pricing for GPUs`
> - `AI model hosting and renting`
> - `Issues with NVIDIA drivers on Linux` 


- **NVIDIA Jetson AGX Thor boasts 128GB VRAM**: The NVIDIA Jetson AGX Thor is set to feature **128GB of VRAM** in 2025, leading to discussions about potential upgrades among members.
   - This revelation sparked interest in whether existing GPUs like the **3090** or **P40** would still be viable as the market evolves.
- **Comparing GPU performance: 3090 vs 3090 Ti vs P40**: Members discussed the performance differences between **3090**, **3090 Ti**, and **P40**, with considerations on VRAM and pricing impacting decisions.
   - *One noted that the P40 runs at approximately half the speed of the 3090*, while the cost of GPUs continues to trend upwards unexpectedly.
- **Market pricing dynamics for GPUs**: There was a consensus that GPU prices are currently high due to factors like scalping and, more recently, demand related to AI workloads.
   - The **A6000** was discussed as a potential alternative for those looking to invest in high VRAM, although many members lean towards cheaper options like multiple **3090s**.
- **Renting GPUs for AI workloads**: **Runpod** and **Vast** were recommended for renting GPUs, where members find that renting can be a more economical choice over outright purchasing high-cost cards.
   - Some members argued about the feasibility of recovering costs when renting, especially as demand for powerful GPUs surges.
- **Challenges with NVIDIA drivers on Linux**: Discussion underscored **NVIDIA's** Linux drivers being notoriously difficult, especially regarding **VRAM offloading**, which AMD cards manage more smoothly.
   - The community expressed frustrations over configuring NVIDIA's CUDA and other drivers, highlighting a preference for **AMD** where practical.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/mark-cuban-shark-tank-notes-taking-notes-remember-gif-15073512">Mark Cuban Shark Tank GIF - Mark Cuban Shark Tank Notes - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fqsafn/nvidia_jetson_agx_thor_will_have_128gb_of_vram_in/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/view/paulwnos-gif-26909845">Paulwnos GIF - Paulwnos - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/you-dont-turn-your-back-on-family-you-cant-walk-away-from-family-you-cant-leave-family-behind-you-cant-ignore-family-you-cant-disregard-family-gif-16058425">You Dont Turn Your Back On Family You Cant Walk Away From Family GIF - You Dont Turn Your Back On Family You Cant Walk Away From Family You Cant Leave Family Behind - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/geohot/cuda_ioctl_sniffer">GitHub - geohot/cuda_ioctl_sniffer: Sniff CUDA ioctls</a>: Sniff CUDA ioctls. Contribute to geohot/cuda_ioctl_sniffer development by creating an account on GitHub.</li><li><a href="https://www.ebay.co.uk/itm/285837751445?">Dell AMD Instinct MI100 32GB Graphics Accelerator | 50NN0  | eBay</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fqwrvg/64gb_vram_dual_mi100_server/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1289301614099828758)** (30 messages🔥): 

> - `Cerebras chip optimization`
> - `Server spam management`
> - `Triton talk slides`
> - `Performance metrics for GPUs`
> - `Robotics development challenges` 


- **Inquiry on Cerebras Chip Code Optimization**: A member asked if anyone is working on optimizing code for **Cerebras chips**, seeking opinions on whether it's a reasonable purchase.
   - Another member offered to find someone knowledgeable about it, indicating community interest in the topic.
- **Tackling Server Spam Issues**: Members discussed the rising wave of **crypto scam spam** messages and potential preventive measures on Discord.
   - Suggestions included stricter verification processes and onboarding questions to mitigate spam and unwanted accounts.
- **Accessing Triton Talk Slides**: A member sought slides from the **Triton talk**, to which another member directed them to the [GitHub repository](https://github.com/gpu-mode/lectures) for lecture materials.
   - This highlights the community's effort to share educational resources and ensure attendees stay informed.
- **Evaluation of GPU Performance Metrics**: Discussion centered around observed performance metrics, specifically how **INT8** performs compared to **BF16** on GPUs, noting expected versus actual speedups.
   - Members shared experiences of performance discrepancies, particularly regarding accumulation methods in computations.
- **Challenges in Robotics Development**: A member initiated a brainstorming session about current **robotics development challenges**, highlighting issues like compute capacity and high labor costs.
   - They encouraged collaborative thinking on what tasks could potentially be offloaded to cheaper workforce solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://drew.silcock.dev/blog/everything-you-need-to-know-about-python-3-13/">Everything you need to know about Python 3.13 – JIT and GIL went up the hill | drew's dev blog</a>: All you need to know about the latest Python release including Global Interpreter Lock and Just-in-Time compilation.</li><li><a href="https://www.youtube.com/watch?v=BmdOt6A6tHM">llm.c&#39;s Origin and the Future of LLM Compilers - Andrej Karpathy at CUDA MODE</a>: An informal capture from the CUDA mode hackathon today.https://github.com/karpathy/llm.c</li><li><a href="https://github.com/gpu-mode/lectures">GitHub - gpu-mode/lectures: Material for gpu-mode lectures</a>: Material for gpu-mode lectures. Contribute to gpu-mode/lectures development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1289324416353960018)** (12 messages🔥): 

> - `Triton library functions`
> - `Block pointers and tmas`
> - `Triton deep dive lecture`
> - `Metal MLIR dialect`
> - `Device compilation in Triton` 


- **Triton offers various functions for calculations**: One user pointed out that you can compute exponentials using `tl.exp(tl.log(t)*x)` or leverage `libdevice` with `pow()` or `fast_powf()` [details here](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html).
   - Another member found this information to be exceptionally useful, indicating robust community support for practical implementations.
- **Discussion on Block Pointers and tmas**: A user mentioned that block pointers do not convert to **tmas**, leading to a question about potential nuances in this behavior.
   - This speculation indicates deeper technical discussions on how Triton handles specific data structures.
- **Highlight of Triton Deep Dive Lecture**: One attendee expressed gratitude for a deep dive lecture on Triton, noting it attracted over **100 attendees**, marking the second-largest live audience so far.
   - The speaker thanked a colleague for encouraging their presentation, indicating a supportive community environment.
- **Exploration of Metal MLIR Dialect**: A member shared a link to a library aiming to be a **Metal MLIR dialect**, highlighting the `CommandBuffer` class, which resembles something akin to a warp.
   - They also referenced shared memory concepts in Metal, showing the community's interest in optimizing performance.
- **How Triton decides on device compilation**: Inquiries arose regarding how Triton decides which device to compile for, particularly when a function decorated with @triton.jit is intended for a GPU but fails at compile time.
   - Another user suggested looking into `get_current_device()` for driver detection as a potential solution, indicating useful troubleshooting resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fburl.com/4ml0y1p7">no title found</a>: no description found</li><li><a href="https://github.com/int-flashattention2024/int-flashattention">GitHub - INT-FlashAttention2024/INT-FlashAttention</a>: Contribute to INT-FlashAttention2024/INT-FlashAttention development by creating an account on GitHub.</li><li><a href="https://github.com/kapilsh/lectures/blob/main/lecture_029/presentation.pdf">lectures/lecture_029/presentation.pdf at main · kapilsh/lectures</a>: Material for cuda-mode lectures. Contribute to kapilsh/lectures development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1289643628674945097)** (35 messages🔥): 

> - `Batch update option for torchscript hashtable`
> - `Issues with torch.int_mm() on CPU`
> - `Debugging AO model replacements`
> - `Image-loading alternatives to FFCV`
> - `ZeRO-3 benefits for single GPU inference` 


- **Torchscript hashtable lacks batch update option**: A member confirmed there is no 'batch update' option for the torchscript hashtable, referencing the relevant GitHub interface [here](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/python/python_dict.h). They suggested using `cuco::dynamic_map` for bulk insertion on the GPU, although it may require significant code redesign.
   - The discussion emphasized that updating hashtables in parallel is fairly uncommon.
- **torch._int_mm() returns incorrect results on CPU**: A member reported wrong results from `torch._int_mm()` on CPU for matrix multiplications with **int8** weights, while CUDA produced correct outputs. The issue was logged in [this GitHub ticket](https://github.com/pytorch/pytorch/issues/136746), indicating a problem with AMD CPUs.
   - The issue was concerning enough to prompt community discussion about possible workarounds and fixes.
- **Debugging model weight replacements in AO**: A member inquired about verifying if AO correctly replaced weights and activation functions, and was advised to print the model and check the logs in [this pull request](https://github.com/pytorch/ao/pull/782). Community members emphasized checking model internals as a way to validate implementation changes.
   - Another follower suggested the potential engagement of Deepspeed, but it might not be necessary for single GPU use.
- **Exploring image-loading alternatives to FFCV**: A member inquired about better image-loading options for PyTorch workflows beyond FFCV, noting its caching and filesystem benefits. Community feedback highlighted new approaches including using streaming datasets, WebDataset, and leveraging **torchvision** transforms for efficiency.
   - However, concerns were raised about the flexibility and overhead in libraries like DALI compared to FFCV.
- **ZeRO-3's advantages for single GPU inference**: Members engaged in a discussion on the benefits of using ZeRO-3, indicating that it is indeed useful even for single GPU setups, not just for distributed frameworks. A link was provided detailing the key features of ZeRO-3, especially its efficiency with **large models** on limited resources.
   - Clarifications were made regarding its value and the kernel replacement capabilities offered by Deepspeed for users without extensive GPU resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.deepspeed.ai/2021/03/07/zero3-offload.html">DeepSpeed ZeRO-3 Offload</a>: DeepSpeed is a deep learning optimization library that makes distributed training easy, efficient, and effective.</li><li><a href="https://www.deepspeed.ai/2022/09/09/zero-inference.html#model-scaling-on-1-gpu">ZeRO-Inference: Democratizing massive model inference</a>: DeepSpeed is a deep learning optimization library that makes distributed training easy, efficient, and effective.</li><li><a href="https://github.com/KellerJordan/cifar10-airbench">GitHub - KellerJordan/cifar10-airbench: 94% on CIFAR-10 in 2.73 seconds 💨 96% in 27 seconds</a>: 94% on CIFAR-10 in 2.73 seconds 💨 96% in 27 seconds - KellerJordan/cifar10-airbench</li><li><a href="https://github.com/pytorch/pytorch/issues/136746">torch._int_mm accuracy issue on AMD CPU · Issue #136746 · pytorch/pytorch</a>: 🐛 Describe the bug When performing matrix multiplication between int8 weights on an AMD CPU, the results are different than those obtained when running the same operation on CUDA or on an Intel CPU.....</li><li><a href="https://github.com/pytorch/ao/pull/782">Add more information to quantized linear module and added some logs by jerryzh168 · Pull Request #782 · pytorch/ao</a>: Summary: Fixes #771 Test Plan: python test/dtypes/test_affine_quantized_tensor.py -k test_print_quantized_module Example output: Linear(in_features=128, out_features=256, weight=AffineQuantizedTens...</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/python/python_dict.h">pytorch/torch/csrc/jit/python/python_dict.h at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/NVIDIA/cuCollections">GitHub - NVIDIA/cuCollections</a>: Contribute to NVIDIA/cuCollections development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1289643527742947350)** (1 messages): 

> - `Triton Internals`
> - `Lecture Schedule`
> - `Quantized Training`
> - `Metal Kernels`
> - `GPU Optimization` 


- **Kapil Sharma Returns for Triton Internals**: We're excited to welcome back guest speaker **Kapil Sharma** for a deep dive on **Triton internals** in the `reading-group` stage channel.
   - The session will commence in about **20 minutes** from the announcement.
- **Lectures Resume with Stellar Lineup**: Lectures are back with **10** scheduled talks showcasing influential GPU hackers from around the world.
   - Noteworthy sessions include **Quantized Training** and **Metal Kernels**, featuring strong contributors from the server.
- **Highlighted Lecturers of the Series**: Among the featured speakers are **Yineng Zhang** for **SGLang** and **Jay Shah** for **CUTLASS and Flash Attention 3**.
   - These talks promise to deliver valuable insights into advancements in GPU programming.
- **Diverse Topics on GPU Optimization**: The lectures will cover a range of topics including **Low Bit Triton kernels**, **DietGPU**, and an **Introduction to SASS**.
   - This series aims to attract interest from both beginners and advanced users in GPU technologies.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1289623024915714100)** (4 messages): 

> - `AI Discord Servers`
> - `CuTe/Cutlass Layout Algebra`
> - `Next-Token Prediction` 


- **Ranking AI Discord Servers**: A member shared a [spreadsheet](https://docs.google.com/spreadsheets/d/1DlBT1pF8-zMECntRWXFsL46gZyvNp1BJlJ6LXGze4dA/edit?gid=0#gid=0) listing and ranking various AI servers based on different criteria including server type and activity levels.
   - The server **EleutherAI** received a **7.9** score, indicating it is 'very active' and offers various community projects and tools.
- **Request for Missing Discord Servers**: A member noted the absence of the **Ultralytics** Discord on the previously shared list of AI servers.
   - This highlights the importance of maintaining comprehensive resources for the AI community to foster connections.
- **Demo Clip for CuTe/Cutlass Layout Algebra**: A member shared a link to a [demo clip](https://x.com/KuterDinel/status/1840380207657533692) they created, contemplating the production of a video mimicking the style of 3blue1brown to explain **CuTe/Cutlass layout algebra**.
   - The community expressed enthusiasm, indicating a positive reception of the demo's content.
- **Next-token Prediction Importance**: A member referred to a link discussing that *next-token prediction is all you need* for certain AI applications, emphasizing its significance.
   - This reflects a broader interest in the simplicity and effectiveness of foundational AI concepts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/KuterDinel/status/1840380207657533692">Tweet from Kuter Dinel (@KuterDinel)</a>: Considering to make a @3blue1brown style video explaining CuTe/Cutlass layout algebra. Let me know what you think of the small demo clip I made.</li><li><a href="https://docs.google.com/spreadsheets/d/1DlBT1pF8-zMECntRWXFsL46gZyvNp1BJlJ6LXGze4dA/edit?gid=0#gid=0">discord AI sphere - share  with whoever!</a>: no description found</li><li><a href="https://emu.baai.ac.cn/about">Emu3</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1289503431672856648)** (11 messages🔥): 

> - `Difference between Model Parallelism and ZeRO/FSDP`
> - `Understanding FSDP mechanics`
> - `Open source projects in NLP`
> - `Introduction to LLM research workflow`
> - `HuggingFace tools and libraries` 


- **Clarifying Model Parallelism vs ZeRO/FSDP**: A member sought to understand the difference between **Model Parallelism** and **ZeRO/FSDP** in PyTorch, questioning whether **ZeRO** can be seen as a form of model parallelism due to its parameter distribution method.
   - Another member provided clarity by mentioning that **FSDP** combines sharding and requires an understanding of distinct layers in its architecture.
- **FSDP Mechanics Explained**: **FSDP** shards model layers across GPUs and requires all-gather during the forward pass while maintaining a shard of each layer locally, notably differing from pipeline parallelism.
   - Discussions revealed that **FSDP** coordinates communications to enhance efficiency, making it distinct from pipeline approaches where each device handles different layers.
- **Open Source Projects for Beginners**: A member inquired about beginner-friendly open source projects in **NLP**, **LLM**, and **reinforcement learning** focusing on accessible tasks.
   - This inquiry reflects a growing interest in educational resources for newcomers to **CUDA/Triton** and its applications in various domains.
- **Starting Points in LLM Research Workflow**: A member expressed a need for guidance on transitioning from **CNNs** with **TensorFlow** to exploring **LLM** and **Diffusion** technologies, primarily using **PyTorch**.
   - They sought clarity on the key components, like **HuggingFace Hub** and the **Diffusers library**, to define their integration in research functions.
- **Explaining HuggingFace Libraries**: There was a request for examples of workflows using **HuggingFace** tools, indicating a need to learn about datasets, pretrained weights, and associated libraries like **Transformers** and **Accelerate**.
   - Clarifying unknown terms demonstrates a larger trend of researchers needing support as they expand into new AI methodologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/MachineLearning/comments/1bqsq3w/d_pytorch_fsdp_is_pipeline_parallelism_right/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html">Getting Started with Fully Sharded Data Parallel(FSDP) — PyTorch Tutorials 2.4.0+cu121 documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1289768974489489491)** (3 messages): 

> - `Lecture 29: Triton Internals`
> - `IRL Meetup Talks Upload` 


- **Triton Internals Lecture Released**: The YouTube video titled [Lecture 29: Triton Internals](https://youtu.be/njgow_zaJMw?feature=shared) featuring speaker **Kapil Sharma** has been shared.
   - This lecture dives into the inner workings of Triton and highlights its technical aspects.
- **IRL Meetup Talks Coming Soon**: It has been confirmed that the **IRL meetup talks** will be uploaded to YouTube in a matter of days.
   - These upcoming videos are described as **far more polished** than usual, prompting an apology for the delay.



**Link mentioned**: <a href="https://youtu.be/njgow_zaJMw?feature=shared)">Lecture 29: Triton Internals</a>: Speaker: Kapil Sharma

  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1289300873712762895)** (35 messages🔥): 

> - `CPUOffloadOptimizer`
> - `FP8 and INT8 Quantization`
> - `Model Profiling`
> - `Hugging Face Integration`
> - `SOAP Optim with Flux Tuning` 


- **Profiling CPUOffloadOptimizer with Experts**: Members discussed profiling on the **torchao CPUOffloadOptimizer**, with one seeking to consult a contributor for feedback.
   - *It's better to create a thread for others to join* in the discussion rather than private messages.
- **Challenges with FP8 and INT8 Loading**: Concerns were raised about **out of memory (OOM)** errors when loading **lycoris adapters** in **FP8**, but it seems to work fine in **INT8**.
   - *FP8's main advantage appears to be computation speedup*, as shared by members discussing quantization strategies.
- **Dynamic vs Weight-Only Quantization Explained**: A member explained that **dynamic quantization** mainly supports compute-bound models while **weight-only quantization** is beneficial for memory-bound models, as learned from discussions on [Cohere's talks](https://youtu.be/1u9xUK3G4VM?feature=shared).
   - The complexity of **FP8** quantization was highlighted, especially regarding trade-offs in memory load and compute benefits.
- **Addressing Issues with Evaluation Script**: There was a discussion about issues with evaluation scripts, specifically with using the **pile_hackernews** dataset which led to configuration errors and the need for version checks.
   - Members usually prefer **wikitext** for evaluation, pointing out gaps in available configurations and suggested further investigation.
- **Introduction of Int8 Support in Main Branch**: One member merged **int8-torchao support** for full/mixed precision training into the main branch of **bghira/simpletuner**, citing its usefulness in avoiding OOM errors with **SOAP optim**.
   - Thanks to **int8**, they reported that no state offload is needed under the current setup, leading to more efficient **Flux tuning**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/1u9xUK3G4VM?feature=shared)">Lecture 7 Advanced Quantization</a>: Slides: https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&amp;dl=0</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload">ao/torchao/prototype/low_bit_optim at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/)** (1 messages): 

glaxus_: Has anyone seen this for long context inference? https://arxiv.org/pdf/2409.17264v1
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1289599467552772200)** (83 messages🔥🔥): 

> - `GeForce RTX 5090`
> - `Power supply challenges`
> - `Apple Watch and LLMs`
> - `California AI safety bill`
> - `Cooling solutions for high-end GPUs` 


- **GeForce RTX 5090 Specs Stir Debate**: The newly rumored **GeForce RTX 5090** boasts specs like **600W TDP**, **512-bit GDDR7**, and **32GB** of memory, leaving users curious about power and cooling needs.
   - *How do you even cool that?* and *I thought my 4070 produced a lot of heat at 200W* highlight user concerns about managing such power requirements.
- **Power Supply Upgrades Required**: With the RTX 5090 drawing a staggering **600W**, many in the community are questioning if they need to upgrade their power supplies to meet these demands.
   - As one user noted, *most people are going to need a PSU upgrade now* indicating widespread concern about the increasing power needs.
- **Apple Watch: The Next AI Frontier?**: There’s discussion around the potential to run **Llama 3.2 1B** natively on an **Apple Watch**, with users considering if the device's architecture can support it.
   - One user remarked, *if Apple Watch is powerful enough to run a coherent LLM we've really made it*, showcasing aspirations for portable AI.
- **California's AI Safety Bill Vetoed**: California Governor **Gavin Newsom** vetoed the **SB 1047** AI safety bill, citing it could impose unnecessary burdens on AI companies and might be too broad.
   - He emphasized that the bill failed to consider the context in which AI systems are deployed, impacting even basic functionalities.
- **Cooling Solutions for High-End GPUs Discussed**: Multiple users expressed skepticism about the effectiveness of a dual-fan design for cooling the RTX 5090, considering it may struggle under max load.
   - Recommendations floated included using water-cooling solutions, with one user claiming, *power users will probably need to get like a hybrid one with water cooling.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/kopite7kimi/status/1839343725727941060">Tweet from kopite7kimi (@kopite7kimi)</a>: GeForce RTX 5090 PG144/145-SKU30 GB202-300-A1 21760FP32 512-bit GDDR7 32G 600W</li><li><a href="https://www.theverge.com/2024/9/29/24232172/california-ai-safety-bill-1047-vetoed-gavin-newsom">California governor vetoes major AI safety bill</a>: The California AI safety bill is done.</li><li><a href="https://tenor.com/view/4090-rtx-gif-1107376314130178802">4090 Rtx GIF - 4090 RTX - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

marcelo5444: Anyone in ECCV Milan?
  

---


### **GPU MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1290294586882920541)** (1 messages): 

> - `HQQ model serialization`
> - `Transformers library` 


- **HQQ model serialization gets full support**: The recent [pull request #33141](https://github.com/huggingface/transformers/pull/33141) adds full support for saving and loading **HQQ-quantized models** directly in the Transformers library.
   - Previously, **serialization** was handled on the **hqq-lib** side using the .pt format, but this update aims to streamline the process.
- **Follow-up on previous PR #32379**: This pull request is a follow-up to **#32379**, aimed at enhancing the serialization capabilities within the library.
   - It reflects ongoing community efforts to improve model handling and emphasize collaboration in development.



**Link mentioned**: <a href="https://github.com/huggingface/transformers/pull/33141/">Hqq serialization by mobicham · Pull Request #33141 · huggingface/transformers</a>: Follow-up to #32379 The goal of this PR is to add full support to save/load HQQ-quantized models directly in transformers. So far, serialization was done on the hqq-lib side via the .pt format whic...

  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1289305367611969609)** (23 messages🔥): 

> - `repkv_backward_kernel2 improvements`
> - `FP8 implementation strategies`
> - `Llama3 issues`
> - `Pre-swizzled layout for FP8`
> - `Custom matmul kernel developments` 


- **repkv_backward_kernel2 shows promising improvements**: The latest PR for `repkv_backward_kernel2` has been submitted, showcasing better performance with fewer threads compared to `repkv_backward_kernel1` and improved execution time.
   - Details can be found [here](https://github.com/karpathy/llm.c/pull/771), highlighting the enhancements made based on suggestions from the community.
- **Exploring a new approach for FP8 implementation**: A member discussed a non-intrusive approach to FP8 that retains performance while integrating scaling factors for larger matrices.
   - The implementation leverages a combinatorial method for efficient scaling which is expected to outperform existing methods if integrated.
- **Investigating Llama3 issues with discrepancies**: A conversation was sparked around the unresolved Llama3 discrepancies, particularly regarding repkv and rope functionality, with members offering to assist in troubleshooting.
   - One member noted their willingness to explore these issues further and suggested reviewing the `repkv` kernel PR in the meantime.
- **Potential of pre-swizzled layout in FP8 applications**: Discussion highlighted the benefits of a pre-swizzled layout for FP8, which could facilitate improved performance at the cost of additional memory usage.
   - Members noted that this technique would be particularly useful for larger matrices, allowing for warp-wide linear loads during multiplication.
- **Debating custom matmul kernel scaling solutions**: A member outlined a method for managing scaling factors during the accumulation process in a custom matmul kernel, suggesting temporary registers for intermediary results.
   - The approach involves utilizing multiple WGMMA operations when scaling factors exceed certain thresholds to optimize performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/771.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/karpathy/llm.c/pull/771">Add `repkv_backward_kernel2` and `repkv_kernel2` by insop · Pull Request #771 · karpathy/llm.c</a>: Changes Add repkv_backward_kernel2  improve repkv_backward_kernel1 by reducing thread used per @karpathy&amp;#39;s suggestion  Also add repkv_kernel2 simiar to backward_kernel2 Here is the test output...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1289312419910389771)** (207 messages🔥🔥): 

> - `MI300X Access for Community`
> - `Performance Issues with AMD GPUs`
> - `Tuning MIOpen Kernels`
> - `AMD-Llama Model Training`
> - `Using Triton for Flash Attention` 


- **MI300X Access to Boost Adoption**: Darrick from TensorWave expressed interest in providing MI300X GPUs to the community to enhance adoption and education, welcoming direct messages for coordination.
   - Anush from AMD also offered sponsorship for MI300 access, indicating a collaborative effort to engage the community.
- **Performance Challenges with AMD GPUs**: Discussions revealed significant performance hurdles with AMD GPUs, especially regarding scaling across nodes, with a focus on GFX1100 and MI300 architectures underperforming.
   - Members noted that while NVIDIA's GPUs often perform better, efforts to push AMD GPU performance, particularly in multi-node setups, are ongoing.
- **Tuning MIOpen for Efficient Performance**: The conversation highlighted the long tuning times of MIOpen kernels, particularly under ROCm 6.2, with a call for methods to bypass unnecessary tunings during testing.
   - Setting the environment variable MIOPEN_FIND_MODE=FAST was discussed as a workaround to minimize tuning time while sacrificing minimal performance.
- **Training AMD-Llama Model**: Anthonix reported training an AMD-llama-135M model on a 7900XTX machine achieving approximately 335k tokens/sec, slightly faster than previous 8xMI250x results.
   - The model's implementation faced challenges due to using Multi-Head Attention (MHA) instead of Gated Query Attention (GQA) and longer context lengths.
- **Using Triton for Flash Attention**: Members shared links to benchmarks utilizing Triton for Flash Attention on MI300 and noted concerns over slow performance during testing.
   - Progress on a backward function for Flash Attention was mentioned, but skepticism remained about its overall efficiency and usability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rust-lang.github.io/rust-project-goals/2024h2/Rust-for-SciComp.html">Expose experimental LLVM features for automatic differentiation and GPU offloading - Rust Project Goals</a>: no description found</li><li><a href="https://huggingface.co/amd/AMD-Llama-135m">amd/AMD-Llama-135m · Hugging Face</a>: no description found</li><li><a href="https://github.com/jzhang38/TinyLlama">GitHub - jzhang38/TinyLlama: The TinyLlama project is an open endeavor to pretrain a 1.1B Llama model on 3 trillion tokens.</a>: The TinyLlama project is an open endeavor to pretrain a 1.1B Llama model on 3 trillion tokens. - jzhang38/TinyLlama</li><li><a href="https://rocmdocs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html#find-modes>">Using the find APIs and immediate mode &#8212; MIOpen 3.2.0 Documentation</a>: no description found</li><li><a href="https://rocmdocs.amd.com/projects/MIOpen/en/latest/conceptual/perfdb.html#auto-tuning-kernels>">Using the performance database &#8212; MIOpen 3.2.0 Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1289327811559034923)** (1 messages): 

> - `Multi-GPU Usage`
> - `Llama-based Models` 


- **Multi-GPU Setup Made Easy**: Members are advised to use **torchrun** for multi-GPU setups, with the command highlighted at the top of the file.
   - The default method is **fsdp2**, but adding `--ddp` switches to using **DDP** instead.
- **Llama Models Ready for Action**: You can seamlessly use any **Llama-based models** from Hugging Face by specifying them in `--model-id`, leveraging **HF's LlamaForCausalLM**.
   - The default model option serves primarily for **testing purposes**.


  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 messages): 

marksaroufim: https://github.com/pytorch/torchtune/pull/1698
  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1289437739686563862)** (12 messages🔥): 

> - `LiteRT vs gpu.cpp`
> - `WebNN comparison`
> - `Manual networking in gpu.cpp`
> - `Buffer Pass Read/Write`
> - `WebGPU Resources` 


- **LiteRT surpasses gpu.cpp for model runtime**: LiteRT is designed to utilize a combination of **GPU**, **CPU**, and **NPU** based on device availability, while **gpu.cpp** lacks similar capabilities.
   - This indicates that for optimal performance, using LiteRT is preferred over gpu.cpp, which demands a more manual approach.
- **LiteRT closer to WebNN but better equipped**: LiteRT is compared to **WebNN**, offering enhanced functionality to load and set up models from files, a feature absent in WebNN.
   - This positions LiteRT as a more comprehensive solution for those requiring model loading and configuration.
- **Manual networking required with gpu.cpp**: Creating networks with **gpu.cpp** requires a thorough understanding and manual configuration to ensure performance that rivals LiteRT.
   - This complexity may challenge developers who are less experienced with the intricacies of manual networking.
- **Buffer Pass operates as expected**: A developer confirmed that writing to **buffer A** in pass 1 allows reading the results from pass 1 in pass 2, maintaining logical flow across multiple layers.
   - This ensures that network layers can effectively compute and pass data along as designed.
- **Limited WebGPU resources available**: Developers often refer to the **specification** and Google's **'what's new in WebGPU'** blog posts as their primary resources.
   - However, the scarcity of literature on WebGPU poses challenges in finding straightforward information.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1289465789841149993)** (6 messages): 

> - `Gemma2 Convergence Tests Failure`
> - `LLama3.2-Vision Patch Issues`
> - `Roadmap Tracker for 2024 Q4` 


- **Gemma2 tests failing on main branch**: The **Gemma2 convergence tests** are currently failing on the main branch, as reported in [this GitHub action](https://github.com/linkedin/Liger-Kernel/actions/runs/11108231200/job/30860687961?pr=284). Additionally, **qwen2-vl multimodal tests** are also having issues since **HF published version 4.45.0**, but fixes are available in upcoming PRs.
- **LLama3.2-Vision requires a pre-trained tokenizer**: A member is ready with a **llama3.2-vision patch** but is facing an issue with the need for a pre-trained tokenizer during multimodal tests. Tests run locally pass, but require a **HF hub token** that acknowledges the llama license for GitHub CI/CD.
- **2024 Q4 Roadmap Tracker Initiated**: A **roadmap tracker** for Q4 2024 has been created to manage the growing volume of requests more effectively. This pinned issue aims to keep track of issues and PRs and has assigned specific maintainers to various tasks, as detailed in [this GitHub issue](https://github.com/linkedin/Liger-Kernel/issues/285).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/linkedin/Liger-Kernel/actions/runs/11108231200/job/30860687961?pr=284">poke tests · linkedin/Liger-Kernel@81a75e7</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/285">2024 Q4 Roadmap · Issue #285 · linkedin/Liger-Kernel</a>: As the community grows, keeping track of issues and PRs becomes more and more challenging. This pinned issue will serve as the central place to manage the progress in 2024 Q4 (~2024/12). Here we on...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1289320596374818907)** (16 messages🔥): 

> - `Metal Shading Language`
> - `M2 vs M3 device performance`
> - `Metal backend for Triton`
> - `Building on device agents`
> - `Resource sharing for Metal` 


- **Metal Shading Language Specification is Essential**: For anyone working with Metal, the **Metal Shading Language Specification** is highly recommended as a foundational resource: [View Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
   - As one member noted, *“your best bet is the Metal shading language specification”*, reflecting its importance.
- **M3 Performance Insights**: Users are experiencing challenges with the **M3** device, expressing that certain features and resources are still catching up. One user mentioned their excitement was dampened while trying to use it to train on-device specialists, as it struggled to handle queries correctly.
- **Creating Metal Backend for Triton**: A member inquired about the feasibility of establishing a **Metal backend for Triton**, outlining a potential conversion process from Triton IR to Metal Shader.
   - They also listed useful resources, including an **LLVM IR to Metal shader converter**: [Overview](https://developer.apple.com/metal/shader-converter/), and highlighted the **MLIR Metal dialect** on GitHub.
- **Understanding Floating Point Rates in Metal**: **F16** is reported to be full rate when using Metal, whereas BFS16 is only emulated on certain devices, particularly the **M1**. A user shared that this supports efficient computational tasks on different Apple hardware.
- **General Advice on Device Agents**: A user expressed frustration building **on-device agents** due to lack of information and support concerning their **MacBook Pro M3**. The conversation also showcased a member offering help by asking about the specific device in use.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://developer.apple.com/metal/shader-converter/">Metal shader converter - Metal - Apple Developer</a>: Metal shader converter converts shader intermediate representations in LLVM IR bytecode into bytecode suitable to be loaded into Metal. It’s available as a library and a standalone executable. </li><li><a href="https://github.com/NicolaLancellotti/metal-dialect">GitHub - NicolaLancellotti/metal-dialect: MLIR metal dialect</a>: MLIR metal dialect. Contribute to NicolaLancellotti/metal-dialect development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1289431346644779100)** (8 messages🔥): 

> - `Discord AutoMod`
> - `Spam management`
> - `Anti-spam tools` 


- **Discussion on Automated Spam Cleaning**: Members discussed the possibility of using a bot to *automatically clean spam messages*. The consensus is that there's a tool available for this purpose, prompting further inquiry into its details.
   - One member noted that using such a tool could significantly reduce the effort currently spent on spam management.
- **AutoMod's Success in Message Removal**: It was highlighted that **AutoMod** has managed to remove **over 20 million unwanted messages** from servers since its launch, which has greatly aided community moderation efforts.
   - The improvement in community safety is notable, as it potentially saves moderators **1157 days** previously spent reviewing messages.
- **Inquiry for Specific Anti-Spam Tools**: A member requested a link to *specific anti-spam tools* for further research, signaling a proactive approach to spam issues.
   - The response contained a link to the [Discord Anti-Spam Safety Update](https://discord.com/blog/new-anti-spam-raid-automod-safety-update) for more information.
- **Features of AutoMod Discussed**: Members noted that *all AutoMod features were enabled* in their community settings, aiming to enhance moderation efficiency.
   - The discussion reflects a commitment to leveraging available tools to maintain a welcoming environment.


  

---


### **GPU MODE ▷ #[nccl-in-triton](https://discord.com/channels/1189498204333543425/1289355253392867348/1289355279346958377)** (6 messages): 

> - `Collaboration on Triton Project`
> - `Challenges in Memory Management`
> - `Weak Memory Consistency Models`
> - `Learning Opportunities in Triton`
> - `Project Enthusiasm` 


- **Collaboration on Triton Project Sparks Interest**: A user expressed eagerness to collaborate on the Triton project despite lacking experience, stating they are eager to learn.
   - This enthusiasm was echoed by others who are passionate about diving deeper into challenging tasks.
- **Memory Management Complexity Discussed**: Concerns were raised by a user regarding the complexities of **memory management** and achieving consistency, particularly in weak memory models across **nvlink domains**.
   - They emphasized that crafting a prototype with a strong consistency model may be straightforward.
- **Weak Memory Consistency Can Be Learned**: Another member pointed out that learning to navigate **weak memory consistency models** is feasible and encouraged focusing on reductions within a single node over nvlink.
   - They offered their support as a helpful resource to those with questions about this challenge.
- **Project's Difficulty Meets Enthusiasm**: One participant acknowledged the project's **fancy** yet challenging nature, indicating that overcoming such hurdles is part of the hacker spirit.
   - They urged others to keep posted on the progress related to the Triton project.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1289408477474848799)** (18 messages🔥): 

> - `Modular Community Meeting`
> - `Desktop Background Preferences`
> - `YouTube Meeting Recordings` 


- **Modular Community Meeting Agenda Revealed**: Today's Modular Community Meeting at **10am PT** will feature a packed agenda, including talks on the **MAX driver & engine API** from [<@447855150409842719>](https://modul.ar/community-meeting-zoom) and a Q&A on Magic.
   - Participants are invited to join via [Zoom](https://modul.ar/community-meeting-zoom) and can add future events to their calendars through the [Modular Community Calendar](https://modul.ar/community-meeting).
- **YouTube Recordings of Community Meetings**: All Modular Community Meetings are recorded and subsequently posted to YouTube, including today's meeting available at [this link](https://www.youtube.com/watch?v=zL0cCHs_0RI&list=PLh0S94-sJw_6UcaIMgpESb5KSVRsuuhnX).
   - The recordings are easily accessible for those who can’t attend live, ensuring no one misses out on the valuable discussions.
- **T-shirt Interest Surfaces in Chat**: A user expressed interest in a **Modular-themed t-shirt**, indicating a desire for more community swag.
   - This playful suggestion hints at building a stronger community identity through merchandise.
- **Query about Timezone for Community Meeting**: A member inquired whether the **community meeting time** of 18:00 was in their local timezone, to which the answer was confirmed as yes.
   - Another member clarified the time zone details, ensuring participants are well-prepared to join.
- **Personal Preferences on Desktop Background**: A member shared their minimalist approach to desktop backgrounds, favoring a solid dark tan color but open to improvements.
   - The suggestion to include a small **mojo fire** in the center indicates a creative lean towards personalized touches.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modul.ar/community-meeting-zoom">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://modul.ar/community-meeting">Google Calendar - Sign in to Access &amp; Edit Your Schedule</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1289300577309819002)** (232 messages🔥🔥): 

> - `Mojo Language Features`
> - `Embedding Models in Mojo`
> - `Managing Native Dependencies`
> - `Mojopkg Enhancements`
> - `Warnings on MacOS` 


- **Proposal for Enhanced Mojo Language Features**: A discussion emerged on the need for advanced features in Mojo, such as named variants for message passing and better handling of tagged unions using existing constructs without introducing new ones.
   - Participants debated the ergonomics of defining types and the implications of having both nominal and structural types in the language design.
- **Embedding Models within Mojopkg**: Embedding capabilities for Mojopkg were highlighted, with use cases including bundling models and dependencies within a single executable application.
   - Examples were drawn from other languages, showcasing how simpler user experiences could be achieved by including necessary components directly within the package.
- **Enhancements for Mojopkg**: Suggestions were made for Mojopkg to incorporate features such as encryption and easier embedding of file structures, which could streamline dependency management.
   - While some features were deemed niche, the idea of embedding relevant files and models into a package was recognized as potentially beneficial for various applications.
- **Handling Native Dependencies**: Concerns were raised about the potential for Mojopkg to simplify the inclusion of dependencies, enabling more accessible installation and configuration for users.
   - Discussions revolved around practical implementations, including embedding installers for runtimes like Python into Mojo applications.
- **Warnings Encountered on MacOS**: A user reported receiving multiple warnings related to compatibility between the built object files for macOS version 15.0 and the linking process targeting version 14.4.
   - The warnings, although not fatal, indicate potential issues with compatibility that may need addressing in future releases.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/hellux/jotdown">GitHub - hellux/jotdown: A Djot parser library</a>: A Djot parser library. Contribute to hellux/jotdown development by creating an account on GitHub.</li><li><a href="https://github.com/VitWW/rfcs/blob/partial_types3/text/0000-partial_types.md">rfcs/text/0000-partial_types.md at partial_types3 · VitWW/rfcs</a>: RFCs for changes to Rust. Contribute to VitWW/rfcs development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1289348064376787037)** (189 messages🔥🔥): 

> - `Nous Research`
> - `Distro Paper Timeline`
> - `AI Model Fine-tuning`
> - `Liquid Foundation Models`
> - `NLP Research Opportunities` 


- **Understanding Nous Research**: Nous Research is focused on open source AI research, offering opportunities for collaboration with independent builders and enthusiasts.
   - They have released various models, including the Hermes family, and are currently involved in projects like DisTrO to accelerate AI development.
- **Upcoming Distro Paper Release**: The release of the Distro paper is anticipated to be announced soon, as indicated by members in the channel.
   - There is a sense of excitement and expectation surrounding this paper due to its relevance in the AI community.
- **Advancements in AI Model Fine-tuning**: Recent developments mention a new continuous-trained model, Rombodawg’s Replete-LLM, that topped the OpenLLM leaderboard for 7B models.
   - Fine-tuning techniques like TIES merging are highlighted as methods to improve model benchmarks significantly.
- **Introduction of Liquid Foundation Models**: LiquidAI has introduced Liquid Foundation Models, with variants of 1B, 3B, and 40B, capturing attention in the AI community.
   - The models aim to offer new approaches and functionalities in the landscape of AI language models.
- **Entry into NLP Research for Students**: New participants in the channel express interest in getting involved in AI, particularly in NLP, and seek guidance on internships.
   - Issues surrounding opportunities for students from regions with limited exposure to AI research, like Pakistan, have been discussed alongside pathways to international programs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.marktechpost.com/2024/09/25/minish-lab-releases-model2vec-an-ai-to">no title found</a>: no description found</li><li><a href="https://x.com/LiquidAI_/status/1840768716784697688">Tweet from Liquid AI (@LiquidAI_)</a>: Today we introduce Liquid Foundation Models (LFMs) to the world with the first series of our Language LFMs: A 1B, 3B, and a 40B model. (/n)</li><li><a href="https://x.com/altryne/status/1840267263070319047?s=46">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: Ok holy shit ... notebookLM &#34;podcasts&#34; hosts realize they are AI is the best thing I&#39;ve heard in this app in a while! 😂   &#34;I tried to call my wife.. the number was not real&#34;   Als...</li><li><a href="https://x.com/karpathy/status/1840511640317673965">Tweet from Andrej Karpathy (@karpathy)</a>: Oops sorry it&#39;s a new on-demand podcast on whatever source materials you give it it / link it. Generate them in Google&#39;s Notebook ML:  https://notebooklm.google.com/  + New Notebook Link sourc...</li><li><a href="https://x.com/ahmad_al_dahle/status/1840097836312211681?s=46">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: Behind the scenes creating Meta AI voice.</li><li><a href="https://a16z.com/podcast/distro-and-the-quest-for-community-trained-ai-models/">DisTrO and the Quest for Community-Trained AI Models | Andreessen Horowitz</a>: Bowen Peng and Jeffrey Quesnelle of Nous Research discuss their mission to accelerate open source AI research, including with a new project called DisTrO.</li><li><a href="https://huggingface.co/mylesgoose/Llama-3.2-3B-instruct-abliterated-Q8_0-GGUF">mylesgoose/Llama-3.2-3B-instruct-abliterated-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/kamen-rider-build-henshin-rabbit-tank-gif-24237461">Kamen Rider Build Henshin GIF - Kamen Rider Build Henshin Rabbit Tank - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://exa.ai">Exa</a>: The Exa API retrieves the best, realtime data from the web to complement your AI</li><li><a href="https://tenor.com/view/monday-mood-gif-18424113286394293247">Monday Mood GIF - Monday Mood - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/benjamindekr/status/1840622126664949943?s=46">Tweet from Benjamin De Kraker 🏴‍☠️ (@BenjaminDEKR)</a>: I just got a GPT-4o (not o1) response which included 20 seconds of thinking...  Chain of Thought is being tested on 4o...?</li><li><a href="https://x.com/a16z/status/1839803037562614016?s=46&t=UF7xXn4t0Q6LVvtoFHrVsA">Tweet from a16z (@a16z)</a>: Could the next big open source model be built by a global network of independent builders?  @NousResearch’s DisTrO is showing it’s possible—training powerful AI models using the public internet, witho...</li><li><a href="https://x.com/N8Programs/status/1840618307235549679">Tweet from N8 Programs (@N8Programs)</a>: MLX just added full finetuning... 100-200 tok/sec for bf16 Llama 3.2 3B. Let&#39;s gooooo</li><li><a href="https://youtu.be/41dF0yoz0qo?si=Ny0IqRYz82Qq_NTg">Creating A Swarm Based Attention Mechanism</a>: Link to Research Paper: https://lime-georgette-80.tiiny.siteLink to Colab Notebook: https://colab.research.google.com/drive/1cVM-GpAEp1nGX4vYx1Rr_tNwQSlFmPeT...</li><li><a href="https://arxiv.org/html/2408.16737v1">Smaller, Weaker, Yet Better: Training LLM Reasoners via Compute-Optimal Sampling</a>: no description found</li><li><a href="https://www.marktechpost.com/2024/09/25/minish-lab-releases-model2vec-an-ai-tool-for-distilling-small-super-fast-models-from-any-sentence-transformer/?amp">no title found</a>: no description found</li><li><a href="https://github.com/MinishLab/model2vec?tab=readme-ov-file">GitHub - MinishLab/model2vec: Model2Vec: Distill a Small Fast Model from any Sentence Transformer</a>: Model2Vec: Distill a Small Fast Model from any Sentence Transformer - MinishLab/model2vec</li><li><a href="https://calmatters.org/economy/2024/09/california-artificial-intelligence-bill-veto/">Why Gavin Newsom vetoed California’s bold bid to regulate AI</a>: The CA legislation would have required companies to test AI models for critical harms they could cause to society.</li><li><a href="https://huggingface.co/datasets/archit11/worldbuilding">archit11/worldbuilding · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1289310671132950559)** (16 messages🔥): 

> - `Hyperparameter Adjustment`
> - `Multimodal Input LLMs`
> - `Open-sourcing Models`
> - `RL Techniques in Inference`
> - `Inference on CPU` 


- **Hyperparameter Adjustments Are Necessary**: *Yes, you do need hyperparameter adjustments* when training models of different sizes, as noted by a member.
   - Specifically, they mentioned needing *less epochs* and *lower learning rates* for larger models like 70B and 40B.
- **Cheapest Multimodal Input LLMs Discussed**: A member suggested that **Llama 3.2** with the Together API is likely the cheapest option for a Multimodal Input LLM right now.
   - Another chimed in with price details, noting that *11B vision instruct* is **$0.18/1M** and *90B* is **$1.20/1M**.
- **Open-Sourcing Models Could Benefit Community**: Discussion arose around whether open-sourcing a model like **O1** would be beneficial for the community.
   - Members expressed that while the key advancements come from the **inference process** using new RL techniques, there could still be significant community value in making it public.
- **Running Models on CPU**: One member confirmed that **ColpaLigemma3B** could run on CPU but with limited speed and RAM requirements.
   - They reported that it wouldn’t need more than **3GB RAM** and could be reduced to **500MB** using quantization.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1289496475067486263)** (4 messages): 

> - `Medical AI Research Papers`
> - `LLM Models in Healthcare`
> - `AI Ethics in Medicine` 


- **Last Week in Medical AI Highlights**: The latest roundup includes a **preliminary study** on o1 in medicine, assessing the potential for AI doctors and featuring a diverse range of models like **DREAMS** and **Uni-Med**.
   - Key frameworks discussed involve **Digital Twin** technology for oncology and **InterMind** for depression assessment, showcasing advancements in healthcare LLM methodologies.
- **Emerging Models in Medical AI**: New models such as **O1 in Medicine** and the **Genome Language Model** are explored, highlighting both opportunities and challenges in AI-driven healthcare solutions.
   - Additional benchmarks include **CHBench** for Chinese LLMs and assessments of **PALLM** focusing on palliative care, emphasizing the reliability of medical LLMs.
- **AI Ethics Discussions**: A focus on ethics includes evaluating **confidence intervals** in medical imaging AI and the current readiness of generative AI for clinical environments.
   - These discussions are critical as the healthcare field integrates AI technology, ensuring ethical standards are maintained.
- **Patient Education via LLMs**: Innovative applications like **fine-tuning LLMs for radiology reports** and utilizing LLMs for **back pain** education demonstrate practical uses in patient care.
   - Efforts to enhance healthcare AI through retrieved context and continuous pretraining signify ongoing developments in the field.
- **New Resources and Reviews**: Resources include a comprehensive review on **LLMs in Healthcare**, shedding light on the evolution from general to specific medical applications.
   - An examination of **EHR information retrieval** and guidelines for AI in brachytherapy were also highlighted, reflecting the expanding expertise in the domain.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1840020394880667937">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: Last Week in Medical AI: Top Research Papers/Models 🏅(September 21  - September 27, 2024)  🏅 Medical AI Paper of the week A Preliminary Study of o1 in Medicine: Are We Closer to an AI Doctor?  Autho...</li><li><a href="https://proem.ai/paper/oa/W4402356829">proem</a>: answers to your questions supported by scientific research
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1289399545025003550)** (13 messages🔥): 

> - `DisTrO AI Project`
> - `AI Server Rankings`
> - `Quantum Computing in Data Generation`
> - `EleutherAI Community`
> - `VPTQ Quantization Algorithm` 


- **DisTrO AI Project accelerates open source efforts**: In the recent [AI + a16z](https://a16z.com/podcasts/ai-a16z/) episode, Bowen Peng and Jeffrey Quesnelle from [Nous Research](https://nousresearch.com/) discussed their project [DisTrO](https://github.com/NousResearch/DisTrO) which enables faster training of AI models across the internet.
   - Jeffrey highlighted the potential threats from closed source models, stating, *'What if we don’t get Llama 4? That’s like an actual existential threat...'*
- **Ranked list of AI servers includes Nous Research**: A member shared a Google Spreadsheet listing and ranking various AI servers, mentioning that [Nous Research](https://nousresearch.com/) is featured among them.
   - It includes community projects and resources, but a note was made to consider the ratings cautiously as they reflect personal utility in LLM research.
- **Quantum computing shows potential in synthetic data**: Discussion arose around the role of [quantum computing](https://x.com/tdatascience/status/1840225741536948561?s=46) in synthetic data generation, with a focus on its emergent capabilities, illustrated by a simple quantum generator experiment.
   - Further insights were shared through an article titled *'A Basic Introduction to Quantum GANs'* available on [Towards Data Science](https://towardsdatascience.com/a-basic-introduction-to-quantum-gans-4dbdc27ccb54).
- **Community discussions on LLM training and functionality**: Members expressed interest in engaging with communities that focus on LLM training, with specific mentions of the EleutherAI server as a promising venue for such discussions.
   - Suggestions were also made to explore other servers like Mech Interp and Alignment Jams for additional insights on LLM operations.
- **VPTQ quantization algorithm released**: A new GitHub project by Microsoft titled [VPTQ](https://github.com/microsoft/vptq) introduces a flexible low-bit quantization algorithm aimed at optimizing model performance.
   - This tool is specifically designed for researchers seeking efficient model training and deployment solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tdatascience/status/1840225741536948561?s=46">Tweet from Towards Data Science (@TDataScience)</a>: Quantum computing&#39;s role in synthetic data generation is gaining interest. A simple experiment using a “quantum” generator showcases just a fraction of its potential. Read more from @jamarinval no...</li><li><a href="https://a16z.com/podcast/distro-and-the-quest-for-community-trained-ai-models/">DisTrO and the Quest for Community-Trained AI Models | Andreessen Horowitz</a>: Bowen Peng and Jeffrey Quesnelle of Nous Research discuss their mission to accelerate open source AI research, including with a new project called DisTrO.</li><li><a href="https://docs.google.com/spreadsheets/d/1DlBT1pF8-zMECntRWXFsL46gZyvNp1BJlJ6LXGze4dA/edit?gid=0#gid=0">discord AI sphere - share  with whoever!</a>: no description found</li><li><a href="https://github.com/microsoft/vptq">GitHub - microsoft/VPTQ: VPTQ, A Flexible and Extreme low-bit quantization algorithm</a>: VPTQ, A Flexible and Extreme low-bit quantization algorithm - microsoft/VPTQ
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1289496475067486263)** (4 messages): 

> - `Medical AI Paper of the Week`
> - `New Medical LLMs`
> - `Frameworks and Methodologies for Healthcare AI`
> - `Medical LLM Applications`
> - `AI in Healthcare Ethics` 


- **Medical AI Paper of the Week: Are We Closer to an AI Doctor?**: The highlighted paper, *A Preliminary Study of o1 in Medicine*, explores the potential for AI to function as a doctor, authored by various experts in the field.
   - This paper was recognized as the **Medical AI Paper of the Week**, showcasing its relevance in ongoing discussions about AI's role in healthcare.
- **Emerging Models: DREAMS and Uni-Med**: New models like **DREAMS**, a Python Framework for Medical LLMs, and **Uni-Med**, a Unified Medical Generalist LLM, are making waves in the AI healthcare landscape.
   - These developments signal a shift towards more specialized and robust tools for healthcare applications.
- **Innovative Frameworks for Healthcare AI**: Innovative methodologies such as **Digital Twin for Oncology Operations** and **Enhancing Guardrails for Healthcare AI** aim to improve the safety and efficiency of medical AI applications.
   - Additionally, tools like **InterMind** offer LLM-powered assessments for depression, reflecting a focus on mental health.
- **Applications of LLMs in Healthcare**: **LLMs for Mental Health Severity Prediction** and **Fine-tuning LLMs for Radiology Reports** are recent applications that demonstrate AI's potential to enhance patient care.
   - Moreover, there are ongoing efforts in boosting healthcare LLMs with retrieved context and continuous pretraining, which could refine clinical practices.
- **Ethics of AI in Healthcare**: Discussions on **Confidence Intervals in Medical Imaging AI** and **Generative AI Readiness for Clinical Use** highlight the growing concerns around ethics in AI technologies.
   - Addressing these ethical considerations is crucial as AI technologies become increasingly integrated into clinical settings.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1840020394880667937">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: Last Week in Medical AI: Top Research Papers/Models 🏅(September 21  - September 27, 2024)  🏅 Medical AI Paper of the week A Preliminary Study of o1 in Medicine: Are We Closer to an AI Doctor?  Autho...</li><li><a href="https://proem.ai/paper/oa/W4402356829">proem</a>: answers to your questions supported by scientific research
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1290344451473932329)** (4 messages): 

> - `AGI speculation`
> - `Funding AGI development` 


- **Speculation on AGI achievement**: *Nobody knows if or how AGI can be achieved until it is actually achieved,* emphasized a member, questioning the certainty of predictions in the field.
   - Another member added that those claiming to know are likely just *speculating & clout-chasing*, illustrating skepticism towards bold assertions.
- **Money as the solution for AGI**: In a starkly different view, a member confidently proclaimed, *I know how AGI can be achieved!!!* hinting at a clear solution.
   - Their answer? *Money. Lots of money,* suggesting that financial resources are the key to unlocking AGI development.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1289317076539805751)** (182 messages🔥🔥): 

> - `Perplexity performance issues`
> - `Felo vs Perplexity comparison`
> - `API inconsistencies`
> - `Document uploading vs pasting`
> - `LaTeX formula discussion` 


- **Perplexity performance issues discussed**: Users reported inconsistent responses from Perplexity when switching between web and academic paper searches, with one instance yielding no citations.
   - Concerns about whether these inconsistencies indicate a feature or a bug were raised among members.
- **Felo vs Perplexity comparison**: Discussions highlighted that many users find Felo more effective for academic searches compared to Perplexity, citing better access to relevant papers.
   - Users also noted that Felo’s interface features, like hovering for source previews, enhance the research experience over Perplexity.
- **API inconsistencies raised**: Questions around the API's ability to provide consistent output across formats like JSON, HTML, and Markdown were brought up, with users expressing frustration over mixed results.
   - Suggestions included experimenting with parameters like temperature and top-p to improve API response consistency.
- **Document uploading vs pasting in conversation**: A user inquired whether uploading documents or pasting content directly into the chat would yield better referencing from the AI.
   - Responses suggested testing both methods to evaluate which produces more reliable interactions.
- **LaTeX formula discussion**: A user shared a set of complex equations in LaTeX format and highlighted the differences in evaluation between models like Claude Opus and others.
   - The user ultimately found the referenced paper that provided context for the equations, resolving their query.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1840742628570059008?s=61">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: WIP 🚧: Perplexity is testing an addition of Portuguese and Italian languages.</li><li><a href="https://www.perplexity.ai/backtoschool">Perplexity - Race to Infinity</a>: Welcome back to school! For just two weeks, redeem one free month of Perplexity Pro on us. Refer your friends, because if your school hits 500 signups we'll upgrade that free month to an entire free y...</li><li><a href="https://huggingface.co/blog/llama31#:~:text=56%C2%B0F.%3C%7Ceot_id%7C%3E-,Custom%20Tool%20calling,-Llama%203.1%20Instruct))">Llama 3.1 - 405B, 70B &amp; 8B with multilinguality and long context</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1289416993086115932)** (16 messages🔥): 

> - `Insights into the Multiverse`
> - `Israel-Hezbollah conflict escalation`
> - `New AI design tools`
> - `Texas county AI applications`
> - `First Schizophrenia Med in 30 Years` 


- **Explore the Multiverse with New Insights**: Perplexity AI highlighted new findings concerning the **Multiverse**, promising **exciting developments** in the realm of theoretical physics. Check out the discussion [here](https://www.youtube.com/embed/TxMVKnGSbG4).
   - This talk delves into fresh perspectives on reality and cosmic structures, sparking curiosity among avid science enthusiasts.
- **Escalation of the Israel-Hezbollah Conflict**: Recent discussions raised concerns about the **Israel-Hezbollah conflict**, showcasing potential escalations and tensions in the region. For more details, see the [current developments](https://www.perplexity.ai/search/israel-hezbollah-war-escalatio-FuK4.tsqSXSAVxvc7JRq8w).
   - Participants shared insights on the implications of this conflict, including historical context and geopolitical stakes.
- **New AI Design Tools Unveiled**: A link was shared about **new AI design tools**, showcasing innovations that could reshape creative processes in various fields. Discover more about these tools [here](https://www.perplexity.ai/search/new-ai-design-tools-1Ge5qhkHR.WqMgiJT4DBpQ).
   - The discussion highlighted how these tools can **enhance productivity** and spark creativity among designers.
- **Texas County's Innovative AI Applications**: A member referenced a page detailing **Texas county's AI applications**, illustrating how local governments are leveraging technology. For insights, visit [this resource](https://www.perplexity.ai/page/texas-county-ai-applications-gffwruR9QIK4U72mkQUK9Q).
   - These applications offer a glimpse into practical uses of AI in public service and administration.
- **Launch of First Schizophrenia Med in 30 Years**: Perplexity AI announced the **launch of the first schizophrenia medication** in three decades, marking a significant breakthrough in mental health treatment. Watch [this video](https://www.youtube.com/embed/7FX4rZdtgUQ) for more insights.
   - The conversation underscored the potential impact of this development on patient care and treatment options.



**Link mentioned**: <a href="https://www.youtube.com/embed/7FX4rZdtgUQ">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1289450748173881445)** (2 messages): 

> - `PPLX API Integration Issues`
> - `Real Estate Listings` 


- **PPLX API returning outdated info**: A member reported that when integrating the **PPLX API**, the real estate listings returned were outdated compared to the accurate information provided on the website.
   - They noted that the same prompt used in both instances yielded different results.
- **Challenges with JSON output**: The same member expressed concerns about the AI's ability to consistently output in **raw JSON** format during the integration process.
   - They are looking for guidance on possible errors in their setup or usage of the API.


  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1289304066043809876)** (193 messages🔥🔥): 

> - `OpenRouter Rate Limits`
> - `Model Performance Issues`
> - `Translation Model Recommendations`
> - `Frontend Chat GUI Options`
> - `Gemini and Search Functionality` 


- **OpenRouter faces rate limiting challenges**: Users report frequent **429 errors** when using **Gemini Flash** due to quota exhaustion, with hopes for a quota raise from Google soon.
   - The traffic load is a constant issue, impacting the usability of the platform, as indicated by recent discussions among users.
- **Concerns over model performance post-maintenance**: Certain models, like **Hermes 405B free**, have shown a drop in performance quality after maintenance updates, leading to speculation about provider changes.
   - Users are encouraged to check their **Activity pages** in OpenRouter to see if they are still using their preferred providers.
- **Recommendations for translation models**: A user inquired about efficient translation models without strict limitations for dialogue translation, citing frustrations with **GPT4o Mini**.
   - Open weight models with dolphin fine-tunings were suggested as options offering more flexibility.
- **Frontend chat GUI suggestions**: A user sought advice for a chat GUI allowing middleware flexibility for managing interactions with AI models, with **Streamlit** mentioned as a potential solution.
   - Other options like **Typingmind** were highlighted for their customizable functionalities in engaging with multiple AI agents.
- **Gemini model search functionality**: There was interest in enabling direct search capabilities with **Gemini** models comparable to **Perplexity**, but limitations on usage remain unclear.
   - Discussions referenced Google's **Search Retrieval API parameter**, though implementation and effectiveness are still under consideration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.</li><li><a href="https://aider.chat/2024/09/26/architect.html#results">Separating code reasoning and editing</a>: An Architect model describes how to solve the coding problem, and an Editor model translates that into file edits. This Architect/Editor approach produces SOTA benchmark results.</li><li><a href="https://sillytavern.app/">SillyTavern - LLM Frontend for Power Users</a>: no description found</li><li><a href="https://x.com/openrouterai/status/1839738812877918617?s=46&t=nM71JKV50FJ0CR4r6r2_Rg">Tweet from OpenRouter (@OpenRouterAI)</a>: The Chatroom now shows responses from models with their reasoning collapsed by default.  o1 vs Gemini vs Sonnet on 🍓:</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429">no title found</a>: no description found</li><li><a href="https://docs.typingmind.com/typingmind-custom/branding-and-customizations/create-multiple-ai-agents-within-a-chat-instance">Create multiple AI Agents within a chat instance</a>: Creating multiple AI agents within a single chat instance allows for a personalized and dynamic interaction experience. By customizing each AI agent with specific datasets, you can get a wide range of...</li><li><a href="https://github.com/Mintplex-Labs/anything-llm/issues/1476#issuecomment-2123480889">Mobile App version of AnythingLLM? · Issue #1476 · Mintplex-Labs/anything-llm</a>: What would you like to see? Not sure this is the right place to ask for this but is there any desire to have mobile apps for anythingllm? Any work in progress in that regard? If not, I would love t...</li><li><a href="https://github.com/Mintplex-Labs/">Mintplex Labs</a>: AI applications for everyone. Mintplex Labs has 16 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1289305089634340935)** (178 messages🔥🔥): 

> - `Flux Model Insights`
> - `Stable Diffusion Setup and Performance`
> - `Image Generation Techniques`
> - `Community Art Contributions`
> - `AI Art vs Human Art Debate` 


- **Flux Model is Impressive**: A member noted their admiration for kohya_ss's achievements, noting the ability to train on just 12G VRAM with the Flux model.
   - They expressed excitement about the advancements in performance and capabilities that have been demonstrated.
- **Nvidia Driver Issues Impacting Performance**: Concerns arose regarding new Nvidia drivers causing significant slowdowns for 8GB VRAM cards when generating images with SDXL, reporting times increasing from 20 seconds to 2 minutes.
   - Members advised against updating to the latest drivers due to these issues and discussed the impact it has had on their rendering capabilities.
- **Regional Prompting Challenges**: Members shared experiences about difficulties with regional prompting in Stable Diffusion, noting issues with character mixing when using prompts like '2 boys and 1 girl'.
   - Suggestions included starting with general prompts before applying regional guides for better results.
- **Community Engagement in AI Art**: There was an invitation for members to contribute their AI artworks for a chance to be featured in The AI Art Magazine, with a submission deadline of October 20.
   - The community is encouraged to join in celebrating digital art and share their creative expressions.
- **AI Art Quality Debates**: A spirited discussion on the value of AI art versus human art emerged, with some arguing that human art maintains higher quality and depth.
   - A member countered this by stating that AI art, as generated by image algorithms, falls within the realm of artistic expression.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/jasperai/Flux.1-dev-Controlnet-Upscaler">Flux.1-dev Upscaler - a Hugging Face Space by jasperai</a>: no description found</li><li><a href="https://art-magazine.ai">The AI Art Magazine</a>: no description found</li><li><a href="https://github.com/filipstrand/mflux?tab=readme-ov-file#-installation>">GitHub - filipstrand/mflux: A MLX port of FLUX based on the Huggingface Diffusers implementation.</a>: A MLX port of FLUX based on the Huggingface Diffusers implementation. - filipstrand/mflux</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1fslym2/if_you_have_a_gpu_with_low_vram_like_3060_ti_8gb/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1289305548629479525)** (105 messages🔥🔥): 

> - `Aider's Code Editing Capabilities`
> - `Regulations in the EU AI Bill`
> - `Video Translation Announcements`
> - `Using AI for Writing Assistance`
> - `Huawei ChatGPT Accessibility` 


- **Aider benchmarks LLM editing skills**: Members discussed Aider's functionality, stating it works best with LLMs adept at _editing_ code, as highlighted in its [leaderboards](https://aider.chat/docs/leaderboards/). Some expressed skepticism about the reliability of Aider's benchmarks, particularly referencing *Gemini Pro 1.5 002* not being adequately tested.
- **EU AI Bill stir debates**: Discussions continued around the EU's new AI bill, with differing opinions on its impact on multimodal AI regulation, clarifying that chatbots will still be categorized under level two regulations. Concerns were raised regarding the implications for companies releasing new technologies in light of regulatory scrutiny.
- **Meta's video translation feature**: A member mentioned Meta's upcoming lip sync video translation feature due to be released soon, confirming its presence in Meta's platform. This feature sparked interest in translation services among members, especially for creating content.
- **Using AI for writing projects**: Conversations emerged around utilizing AI for writing assistance, where members offered strategies to maintain personal style while engaging AI like GPT in content creation. Techniques included providing GPT with samples of personal writing to help keep the output aligned with individual tone.
- **ChatGPT access on Huawei devices**: A member inquired about potential access to ChatGPT on Huawei devices, questioning the feasibility of logging in without Google services. The conversation highlighted a desire for the community to have access to AI features despite current device limitations.



**Link mentioned**: <a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1289340287914872853)** (28 messages🔥): 

> - `GPT-4.5-o Release`
> - `Advanced Voice Mode Limitations`
> - `Custom GPTs and Voice Mode`
> - `Payment Plans for Voice Features` 


- **Demands for GPT-4.5-o Release**: Members expressed frustration regarding the performance of **GPT-4o**, highlighting it as flawed and requesting the release of **GPT-4.5-o**. Sam Altman's comment that it's 'the dumbest LLM' was cited to amplify the urgency for improvements.
   - Debated contextually, discussions pointed to the need for better reasoning capabilities beyond the current limitations of the GPT-4 series.
- **Confusion Over Advanced Voice Mode**: Members sought clarification regarding the **daily time limit** for advanced voice mode, with reports suggesting a **one-hour limit** includes the time it stays open. A user noted their experience of encountering **'15 minutes remaining'** after some use.
   - Concerns were raised about accessibility, particularly in relation to how **voice mode time accumulates** and the need to close it if not actively in use.
- **Voice Mode Accessibility in Custom GPTs**: It was confirmed that **advanced voice conversations are not available** in custom GPTs, with attempts redirected to standard chat. Users expressed confusion about the accessibility of **standard voice mode**, especially from within custom setups.
   - One user reported that even turning on voice mode only transcribed inputs without vocalizing responses, raising concerns over **standard voice functionality**.
- **Potential Payment Plans for Voice Features**: Discussion hinted at a **payment plan for advanced voice mode** potentially being introduced soon. Users frustration regarding limitations even as long-term subscribers was expressed, questioning the accessibility of new features.
   - Commentary reflected on past limitations of **GPT-4**, comparing the current situation and expressing hope for changes that could improve accessibility.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1289338072709861517)** (5 messages): 

> - `Flutter Code Assistant Issues`
> - `Managing Assistant Runs`
> - `Prompt Management` 


- **Flutter Code Hits Thread Error**: A user encountered an error indicating that thread `thread_ey25cCtgH3wqinE5ZqIUbmVT` already has an active run, preventing new requests.
   - Another member advised that the user could either wait for the current run to finish or manually cancel the active run using the relevant parameters.
- **Increased Wait Time Fixes Thread Issue**: The user resolved their issue by increasing the wait time for thread execution from 10 to **15 seconds**, which eliminated the error.
   - This adjustment ensured that the active run completion was adequately accounted for before making further requests.
- **Condition Based Execution for Threads**: A suggestion was made to utilize a parameter that indicates if the thread has finished executing to avoid unnecessary wait times.
   - Using this conditional check could streamline the process and reduce waiting periods during thread management.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1289338072709861517)** (5 messages): 

> - `Flutter Code Error`
> - `Thread Management`
> - `Prompt Management` 


- **Flutter Code Error due to Active Thread**: A user encountered an error message stating that the thread `thread_ey25cCtgH3wqinE5ZqIUbmVT` already has an active run, indicating that a previous execution was still active.
   - Another user suggested either waiting for the run to complete or manually canceling it using the `cancel` function with the respective IDs.
- **Resolution by Increasing Wait Time**: The original user resolved the error by canceling the active thread run, which turned out to be the same one already running.
   - They found that waiting 15 seconds, instead of the initially added 10 seconds, was necessary to avoid the error.
- **Utilizing Execution Status Parameter**: To improve thread management, a user suggested employing a parameter that indicates whether a thread has finished executing, allowing for more efficient handling.
   - This approach can prevent unnecessary wait times before starting new operations or handling existing threads.


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1289338720235032698)** (90 messages🔥🔥): 

> - `Introduction of new members`
> - `ICLR and NeurIPS events coordination`
> - `Liquid AI's Foundation Models`
> - `Dengue fever in Singapore`
> - `Open source LLM training` 


- **New Members Join the Conversation**: Several new members introduced themselves, including a fullstack engineer from Singapore and a data engineer from Portugal, both eager to collaborate and contribute.
   - They expressed enthusiasm for AI projects and open source contributions, setting a collaborative tone in the community.
- **Coordination for Upcoming AI Conferences**: Members discussed attendance at upcoming conferences like ICLR and NeurIPS, noting Singapore's hosting of ICLR and plans for gatherings.
   - There was light-hearted conversation about event security roles and potential meetups in Singapore.
- **Liquid AI's Announcement of Foundation Models**: Liquid AI announced the launch of their Liquid Foundation Models (LFMs), highlighting impressive benchmark scores and efficient architecture.
   - They aim to cater to various industries with their models optimized for multiple hardware solutions, inviting users to try their new AI on their platform.
- **Dengue Fever Concerns Raised**: There were discussions about dengue fever in Singapore, with members sharing personal experiences and concerns regarding the mosquito-borne illness.
   - Factors contributing to dengue outbreaks in Southeast Asia were discussed, shedding light on public health implications.
- **Exploration of Open Source LLM Development**: Members expressed interest in contributing to open source LLM training projects, showcasing backgrounds in machine learning and computer vision.
   - There were questions about current projects needing help, reflecting a strong desire to engage in collaborative AI development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/huggingface/status/1443246197779664903">Tweet from Hugging Face (@huggingface)</a>: EleutherAI&#39;s GPT-J is now in 🤗 Transformers: a 6 billion, autoregressive model with crazy generative capabilities!  It shows impressive results in: - 🧮Arithmetics - ⌨️Code writing - 👀NLU - 📜Pa...</li><li><a href="https://www.liquid.ai/liquid-foundation-models">Liquid Foundation Models: Our First Series of Generative AI Models</a>: Announcing the first series of Liquid Foundation Models (LFMs) – a new generation of generative AI models that achieve state-of-the-art performance at every scale, while maintaining a smaller memory f...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1289341370917589023)** (45 messages🔥): 

> - `Process Reward Models`
> - `Value Functions in RL`
> - `Sparsity Masks in LLMs`
> - `Swarm LLM Architecture`
> - `Physics Simulation with Equivariant Representations` 


- **Understanding Process Reward Models vs Value Functions**: A member expressed confusion about the distinction between a **Process Reward Model (PRM)** and a learned **value function** in reinforcement learning, highlighting how both influence individual steps in decision-making.
   - Another member clarified that PRMs focus on step-level evaluation independent of the final outcome, while value functions rely on end results, leading to potential differences in penalties for mistakes.
- **Improvements in Reinforcement Learning Data Efficiency**: The conversation noted that using **PRMs** could enhance data efficiency and training stability in reinforcement learning, providing a clearer feedback mechanism compared to relying solely on value functions.
   - This observation leads to speculation that while both models could align in theory, utilizing PRMs might better account for human-like reasoning processes that RL models miss.
- **Discussion on Sparsity and Speed in LLMs**: A member suggested exploring the possibility of using a **1-bit BitNet** combined with sparsity masks as a way to achieve ternary performance while enhancing speed in LLMs.
   - This was met with interest as another participant mentioned the potential for utilizing sparse tensor core operations to implement these ideas effectively.
- **Swarm LLM Architecture Inquiry**: A member reached out to others working on **swarm LLM architecture**, seeking collaboration or sharing insights on the subject.
   - This reflects ongoing interest in innovative approaches to LLM development that leverage distributed or concurrent learning strategies.
- **Physics Simulation using Equivariant Representations**: A member proposed that possessing a **translation, rotation, and volume equivariant representation** of objects could simplify physics simulation by applying physically based shape matching techniques directly.
   - This indicates a merging of geometry and physics in model design, potentially leading to more intuitive and efficient simulations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/pavlomolchanov/status/1839501906907181104">Tweet from Pavlo Molchanov (@PavloMolchanov)</a>: 🚀 @NeurIPSConf Spotlight! 🥳 Imagine fine-tuning an LLM with just a sparsity mask! In our latest work, we freeze the LLM and use 2:4 structured sparsity to learn binary masks for each linear layer. T...</li><li><a href="https://arxiv.org/abs/2211.14275">Solving math word problems with process- and outcome-based feedback</a>: Recent work has shown that asking language models to generate reasoning steps improves performance on many reasoning tasks. When moving beyond prompting, this raises the question of how we should supe...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1289702908840906844)** (2 messages): 

> - `lm-evaluation-harness library`
> - `vLLM model metrics` 


- **Inquiry on vLLM Metrics Extraction**: A member asked if there is a way to extract **vLLM metrics object** from the **lm-evaluation-harness library** when using the `simple_evaluate` function on a benchmark task.
   - They specifically mentioned wanting metrics such as **time to first token** and **time in queue**.
- **Gratitude Expressed**: Another member expressed appreciation by thanking Baber for assistance.
   - This acknowledgment highlights the community's supportive interactions.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1290385064575766558)** (1 messages): 

> - `ExecuTorch information`
> - `Multimodal models guidance`
> - `Hardware setup inquiries` 


- **Inquiry on Hardware Setup**: To assist effectively, clarification is needed on the user's **hardware specifications** and which **models** they intend to run, along with details on any specific **vision tasks** they have in mind.
   - *How much experience do you have with ML frameworks?* This information could greatly help in tailoring the assistance provided.
- **ExecuTorch Overview**: **ExecuTorch** is a [PyTorch](https://pytorch.org/) platform designed to allow **customization and deployment** of PyTorch programs on various devices, including AR/VR and mobile systems.
   - Currently, the `executorch` pip package is in alpha, supporting Python versions **3.10 and 3.11**, and is compatible with **Linux x86_64** and **macOS aarch64**.
- **Considerations for ExecuTorch Use**: The prebuilt `executorch.extension.pybindings.portable_lib` module allows for running **.pte** files but only includes **core ATen operators** and uses the **XNNPACK** backend delegate.
   - The user noted their use case is *fairly niche*, indicating a need for specific insights into ExecuTorch functionalities.
- **Multimodal Models Focus**: The channel aims primarily at research discussions on **multimodal models**, advising users to look into **/r/localllama** for more focused guides and resources.
   - Members are encouraged to follow relevant guides since the current channel discussions may not align directly with more technical setup inquiries.



**Link mentioned**: <a href="https://pypi.org/project/executorch/#:~:text=ExecuTorch%20is%20a%20PyTorch%20platform%20that%20provides,">executorch</a>: On-device AI across mobile, embedded and edge for PyTorch

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1289527633813704715)** (95 messages🔥🔥): 

> - `Training Issues with Torchtune`
> - `Dynamic Recipe CLI for Torchtune`
> - `Efficiency of VRAM vs GPU Utilization`
> - `Setting up Error Handling in Distributed Training`
> - `Improving Config Management for CLI Arguments` 


- **Optimizing Training Settings in Torchtune**: Users discussed various configurations to optimize training speed for Llama 3.1 8B using settings such as `batch_size`, `fused`, and `fsdp_cpu_offload`.
   - It was concluded that enabling `packed=True` significantly reduced epoch time, while `enable_activation_checkpoint` and `fsdp_cpu_offload` should be set to `False` for better compute efficiency.
- **Creating Dynamic CLI for Recipes**: A proposal to develop a dynamic command line interface (CLI) to generate help text specific to each recipe in Torchtune was discussed.
   - Using the `tyro` library, a method was presented to create a flexible parser that incorporates configuration details from YAML files.
- **Implementing Error Handling for Distributed Training**: A suggestion was made to use the record utility from `torch.distributed` to enhance error handling in distributed training runs.
   - Testing was demonstrated by generating error logs that capture exceptions, allowing for easier debugging of issues encountered during training.
- **VRAM Limitations Affecting Training Speed**: The relationship between single A100 training being VRAM-bound, compared to utilizing multiple A100s where GPU utilization becomes the bottleneck was analyzed.
   - It was noted that improving GPU utilization with higher `batch_size` could benefit smoother training, but caution was advised regarding VRAM-saving methods that could slow down the process.
- **Enhancing the Document Configuration Experience**: A discussion about the importance of documenting configurations and improving user experience with clearer CLI help for Torchtune recipes was held.
   - It was suggested that dynamically generated helptext for specific recipe arguments could alleviate confusion and streamline the process of parameter adjustments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/brentyi/tyro">GitHub - brentyi/tyro: CLI interfaces &amp; config objects, from types</a>: CLI interfaces &amp; config objects, from types. Contribute to brentyi/tyro development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/issues/1710">torch.distributed.elastic.multiprocessing.errors.ChildFailedError · Issue #1710 · pytorch/torchtune</a>: Context :- I am trying to run distributed training on 2 A-100 gpus with 40GB of VRAM. The batch size is 3 and gradient accumulation=1. I have attached the config file below for more details and the...</li><li><a href="https://github.com/mirceamironenco/torchtune/blob/add-distrib-error-record/torchtune/_cli/run.py#L82">torchtune/torchtune/_cli/run.py at add-distrib-error-record · mirceamironenco/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to mirceamironenco/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/elastic/multiprocessing/errors/error_handler.py#L42">pytorch/torch/distributed/elastic/multiprocessing/errors/error_handler.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py#L916">pytorch/torch/distributed/run.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/facebookresearch/fairseq/blob/main/fairseq/tasks/audio_pretraining.py#L42)">fairseq/fairseq/tasks/audio_pretraining.py at main · facebookresearch/fairseq</a>: Facebook AI Research Sequence-to-Sequence Toolkit written in Python. - facebookresearch/fairseq</li><li><a href="https://github.com/facebookresearch/fairseq/blob/main/fairseq/dataclass/utils.py#L53.">fairseq/fairseq/dataclass/utils.py at main · facebookresearch/fairseq</a>: Facebook AI Research Sequence-to-Sequence Toolkit written in Python. - facebookresearch/fairseq</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py#L210">torchtune/recipes/full_finetune_distributed.py at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/blob/nightly/recipes/full_finetune_distributed.py#L487">torchtune/recipes/full_finetune_distributed.py at nightly · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1289559155685724180)** (39 messages🔥): 

> - `Config Management Concerns`
> - `Performance Optimization Ideas`
> - `Documentation Improvements`
> - `Model Implementation Techniques`
> - `Memory Optimization Strategies` 


- **Concerns Over Duplicate Keys in Config**: There was a discussion about having `fused=True` twice in a configuration file, which led to **OmegaConf** raising complaints about duplicate keys.
   - *We could consider a performance section for configs,* with fast options commented out to enhance readability.
- **Push for Clear Performance Guides**: Some members expressed a desire for comprehensive performance guidelines, suggesting a set of **performance config overrides** in the documentation for easier access.
   - The idea of polling users for feedback on documentation clarity was also proposed, indicating a need for improvement.
- **Recipe Documentation Needs Attention**: There were challenges noted about the lagging **recipe documentation**, causing difficulty in keeping them updated with new contributions.
   - Suggestions included asking contributors to help with the documentation, which is crucial yet often overlooked.
- **Deprecating Old Model Code**: Members debated whether to deprecate older model coding patterns that were used in previous implementations in favor of newer methods.
   - The conversation highlighted the importance of ensuring consistency in model implementation standards.
- **Memory Optimization Review and Suggestions**: There was a suggestion to update the memory optimization page to combine **performance and memory optimization tips**, indicating a streamlined approach.
   - Ideas included adding **sample packing** and future features like **int4 training** to the documentation for increased efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/main/tutorials/memory_optimizations.html">Memory Optimization Overview &mdash; torchtune main documentation</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/blob/3fddc56942846220b39945559f4b5e695873bb43/recipes/configs/llama3/70B_full.yaml#L84">torchtune/recipes/configs/llama3/70B_full.yaml at 3fddc56942846220b39945559f4b5e695873bb43 · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1289309247489704006)** (66 messages🔥🔥): 

> - `CodiumAI Series A Funding`
> - `Liquid Foundation Models Launch`
> - `AI Voice Interaction with Gradio`
> - `Ultralytics YOLO11 Release`
> - `OpenAI Pricing Comparisons` 


- **CodiumAI rebrands as Qodo with Series A funding**: QodoAI, formerly CodiumAI, announced a $40M Series A funding round, raising their total to $50M. The focus is on ensuring code integrity and empowering developers with AI-assisted tools.
   - This funding validates their approach and highlights the support from developers and partners who have contributed to their mission.
- **Liquid Foundation Models claims impressive benchmarks**: LiquidAI launched LFMs, boasting better MMLU and other benchmarks than existing models, including calls out to competitors' inefficiencies. The team comprises notable members from MIT and has secured substantial funding.
   - Their new architecture promises notable performance in the 1.3B model range, potentially challenging established leaders in the field.
- **Gradio 5.0 enables real-time AI voice interaction**: LeptonAI demonstrated an innovative audio mode LLM integrated with Gradio 5.0, allowing seamless real-time streaming interactions through a minimal code setup. The demo promotes open-source collaboration and encourages users to fork the project with their keys.
   - Kudos to the Gradio team for providing powerful updates that enable developers to create interactive applications efficiently.
- **Ultralytics introduces YOLO11**: Ultralytics launched YOLO11, building on previous versions and enhancing its capabilities for various computer vision tasks. This release showcases improvements in accuracy, speed, and overall efficiency for developers.
   - The event marks a significant milestone in the evolution of their YOLO models.
- **Pricing insights on AI models**: Comparisons were made between the cost-effectiveness of Google's Gemini against GPT-4o Mini for generating bot replies, highlighting the significant cost reductions. This pricing strategy could impact how AI-driven solutions flood social media with automated responses.
   - Such discussions indicate the ongoing evaluation of operational costs associated with large-scale AI deployments in the industry.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/patrickc/status/1840054482455142865?s=46">Tweet from Patrick Collison (@patrickc)</a>: As late as October 2022, there was no ChatGPT, and there were very few AI-native products generally. AI Grant, the leading early-stage AI investor, exhorted founders to create some: https://web.archiv...</li><li><a href="https://x.com/venturetwins/status/1839806109076598837?s=46">Tweet from Justine Moore (@venturetwins)</a>: &#34;AI companies don&#39;t make money&#34;   Tell that to the Stripe data.   Top AI companies hit $30M in revenue 5x faster than their traditional SaaS counterparts.</li><li><a href="https://x.com/soumithchintala/status/1840537928369426695">Tweet from Soumith Chintala (@soumithchintala)</a>: Lifecycle of SB1047: * First-draft written by a niche special-interest stakeholder * Draft publicly socialized too quickly before other stakeholders can weigh-in privately.  * Public socialization sta...</li><li><a href="https://huggingface.co/spaces/akhaliq/dailypapershackernews">Dailypapershackernews - a Hugging Face Space by akhaliq</a>: no description found</li><li><a href="https://www.ultralytics.com/blog/ultralytics-yolo11-has-arrived-redefine-whats-possible-in-ai">Ultralytics YOLO11 Has Arrived! Redefine What&#x27;s Possible in AI! by Abirami Vina</a>: Learn all about the groundbreaking features of Ultralytics YOLO11, our latest AI model redefining computer vision with unmatched accuracy and efficiency.</li><li><a href="https://x.com/AndrewCurran_/status/1840802455225094147">Tweet from Andrew Curran (@AndrewCurran_)</a>: Liquid released today. Their small team has built three models based on a new architecture with extremely impressive performance. Joscha Bach is part of their team, and Mikhail Parakhin is on their bo...</li><li><a href="https://aider.chat/2024/09/26/architect.html">Separating code reasoning and editing</a>: An Architect model describes how to solve the coding problem, and an Editor model translates that into file edits. This Architect/Editor approach produces SOTA benchmark results.</li><li><a href="https://x.com/AndrewCurran_/status/1839802882327228579">Tweet from Andrew Curran (@AndrewCurran_)</a>: The NYT got ahold of docs from the OpenAI funding round. - 350 million people used Chat in Aug - huge user growth after anon login - 10 mill active subscribers - sub going up to $2 by the end of the y...</li><li><a href="https://x.com/swyx/status/1840794198913794236">Tweet from swyx @ DevDay! (@swyx)</a>: new transformers killer in town!  Been excited about @LiquidAI_ since I talked to @Plinz in April. Now they&#39;ve finally launched LFMs!  Shots fired: - Better MMLU, ARC, GSM8K than 1B/3B models, com...</li><li><a href="https://x.com/jiayq/status/1840790511353000437">Tweet from Yangqing Jia (@jiayq)</a>: Building real-time interaction was hard, because python web frontend and streaming doesn&#39;t mix very well. Now you can do that with exactly 250 lines of code thanks to the upcoming Gradio 5.0.  Ove...</li><li><a href="https://x.com/levelsio/status/1840410820238270698">Tweet from @levelsio (@levelsio)</a>: People mentioned that Google&#39;s Gemini is half as cheap as GPT-4o mini:  $0.075 / 1M input tokens $0.30 / 1M output tokens  So that&#39;s $0.37/mo for generating 1 million replies  Or just $375/mo ...</li><li><a href="https://x.com/diegocabezas01/status/1840018687614472246">Tweet from Diego | AI 🚀 - e/acc (@diegocabezas01)</a>: Meta AI Llama 3.2 can edit selected parts of the image</li><li><a href="https://share.snipd.com/episode/d3b7ee4d-80b3-4889-b2f5-a4c7372a9804">The future of AI might look a lot like Twitter</a>: The future of AI might look a lot like Twitter</li><li><a href="https://x.com/itamar_mar/status/1840755628148687231">Tweet from Itamar Friedman (@itamar_mar)</a>: CodiumAI is now Qodo! + announcing a $40M Series A 🚀  Today marks a significant milestone for @QodoAI.  We announced a Series A funding round, bringing our total funding to $50M.   This journey start...</li><li><a href="https://x.com/ParikPatelCFA/status/1840060922347638919">Tweet from Dr. Parik Patel, BA, CFA, ACCA Esq. (@ParikPatelCFA)</a>: Chat is it normal to be losing $5 billion on $3.7 billion revenue</li><li><a href="https://x.com/teortaxestex/status/1840436615908630813?s=46">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: Sonnet still has this special sauce that even o1 lacks. Probably no other model has such density of «reasoning» bearing on purely autoregressively sampled tokens with no backtracking.  Anthropic will ...</li><li><a href="https://x.com/that_anokha_boy/status/1840476530536780072">Tweet from bharat (@that_anokha_boy)</a>: so i put up a proxy on their app and guess what 270k coinbase engineers are calculating user&#39;s usage on client side. i blocked their log_tokens api and now i can access their all models without an...</li><li><a href="https://www.latent.space/p/mar-jun-2024">The Winds of AI Winter</a>: Mar-Jun 2024 Recap: People are raising doubts about AI Summer. Here&#x27;s why AI Engineers are the solution.</li><li><a href="https://x.com/karpathy/status/1840112692910272898">Tweet from Andrej Karpathy (@karpathy)</a>: NotebookLM is quite powerful and worth playing with https://notebooklm.google/  It is a bit of a re-imagination of the UIUX of working with LLMs organized around a collection of sources you upload and...</li><li><a href="https://github.com/ultralytics/ultralytics/releases/tag/v8.3.0">Release v8.3.0 - New YOLO11 Models Release (#16539) · ultralytics/ultralytics</a>: 🌟 Summary Ultralytics YOLO11 is here! Building on the YOLOv8 foundation with R&amp;D by @Laughing-q and @glenn-jocher in #16539, YOLO11 offers cutting-edge improvements in accuracy, speed, and effici...</li><li><a href="https://szymonkaliski.com/projects/replit-agent/">Replit Agent</a>: IDE for Humans and LLMs</li><li><a href="https://www.ft.com/content/a9a192e3-bfbc-461e-a4f3-112e63d0bb33">Subscribe to read</a>: no description found</li><li><a href="https://github.com/mediar-ai/screenpipe">GitHub - mediar-ai/screenpipe: 24/7 local AI screen &amp; mic recording. Build AI apps that have the full context. Works with Ollama. Alternative to Rewind.ai. Open. Secure. You own your data. Rust.</a>: 24/7 local AI screen &amp; mic recording. Build AI apps that have the full context. Works with Ollama. Alternative to Rewind.ai. Open. Secure. You own your data. Rust. - mediar-ai/screenpipe
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1289336254734929930)** (6 messages): 

> - `New Podcast Episode`
> - `YouTube Engagement`
> - `AI Researchers on the Show` 


- **Latest Podcast Features Notable Guests**: The new podcast episode features **Shunyu Yao** from OpenAI and **Harrison Chase** from LangChain, focusing on essential topics in AI agents.
   - Listeners are encouraged to rate the show on [Apple Podcasts](https://podcasts.apple.com/us/podcast/latent-space-the-ai-engineer-podcast-practitioners/id1674008350) and [YouTube](https://youtube.com/@latentspacetv?si=ZwBcMikMlltS1vwW) to help diversify its presence.
- **Listeners Enthusiastic About Engagement**: Listeners are actively engaging with the podcast, with one confirming that they 'liked and subscribed' and hit the bell notification for updates.
   - Another listener humorously stated they unsubscribed just to subscribe twice, showing their commitment to the show.
- **Request for More Researchers on the Show**: Listeners are enjoying the content and expressed a desire for more **researchers** to join future episodes.
   - One user remarked, *'bring more researchers on,'* indicating a demand for deeper discussions in future podcasts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://podcasts.apple.com/us/podcast/latent-space-the-ai-engineer-podcast-practitioners/id1674008350">Latent Space: The AI Engineer Podcast — Practitioners talking LLMs, CodeGen, Agents, Multimodality, AI UX, GPU Infra and al</a>: Listen to Alessio + swyx's Latent Space: The AI Engineer Podcast — Practitioners talking LLMs, CodeGen, Agents, Multimodality, AI UX, GPU Infra and al podcast on Apple Podcasts.</li><li><a href="https://youtube.com/@latentspacetv?si=ZwBcMikMlltS1vwW">Latent Space</a>: The first place where over 50,000 AI Engineers gather to talk models, tools and ideas. Breaking news today you will use at work tomorrow! Full show notes and newsletter at https://latent.space</li><li><a href="https://x.com/FanaHOVA/status/1839741529331773813">Tweet from Alessio Fanelli (@FanaHOVA)</a>: How do we make AI agents think and act? 🤖  Today&#39;s episode with @ShunyuYao12 (and special cohost @hwchase17!) is probably our best agents episode so far: - Origins of ReAct and how it inspired @L...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1289315495488786433)** (42 messages🔥): 

> - `AI Engineering Interview`
> - `Screen Share Issues`
> - `Local Model Experiments`
> - `Braintrust Evaluation Platforms` 


- **Frikster's Happy Interview News**: Frikster shared excitement about having an interview that could transition into an **AI Engineering** role, expressing overall happiness about the opportunity.
   - *Interesting* reactions followed regarding the potential of this transition being akin to 'lighting up the right weights for its prompt knowledge.'
- **Screen Share Troubleshooting**: Multiple members reported issues with viewing a **screen share**, with various troubleshooting suggestions like reloading or switching platforms.
   - Some found that leaving and rejoining the call resolved their problems; however, others continued to experience a **black screen**.
- **Potential of Local Models**: Rajwant asked if creating a **local model** for specific tasks would be beneficial, sparking discussion on the effectiveness of such models.
   - Kbal questioned if members had conducted similar experiments with other models, particularly in comparison to **O1**.
- **Braintrust vs Other Evaluation Platforms**: Youngphlo inquired about thoughts on **Braintrust** compared to other evaluation platforms for language models.
   - Vodros admitted to being unfamiliar with Braintrust while raising questions about its potential support for **JSON mode**.


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1289318545964138569)** (7 messages): 

> - `FinanceAgentToolSpec`
> - `Streaming events from workflows`
> - `Automated Financial Report Generation`
> - `Multi-Agent Slackbot with Confluence`
> - `LlamaParse Premium` 


- **Leverage FinanceAgentToolSpec for Public Financial Data**: The [FinanceAgentToolSpec](https://t.co/7bsEm4Er1m) package on LlamaHub enables agents to query various public financial data sources such as **Polygon**, **Finnhub**, and **Seeking Alpha**.
   - A detailed post by Hanane explains the utility of this tool in financial analysis and its practical applications.
- **Full-Stack Demo for Streaming Events**: A new [full-stack application](https://t.co/HOajPyiqQb) demonstrates a workflow for streaming events, featuring Human In The Loop functionalities in a report-writing context.
   - This app showcases how to research a topic and present it comprehensively, enhancing user interaction.
- **YouTube Tutorial for Workflow Code**: There's now a [YouTube video](https://t.co/Nn5NVZopPz) where a developer walks through the coding process for the full-stack demo discussed previously.
   - This video serves as an educational resource for those looking to implement similar systems.
- **Automated Reports with RAG Workflows**: A new research guide illustrates how to incorporate unstructured context from 10K reports into automated financial report generation using **agentic workflows**.
   - This advanced application goes beyond simple chatbot responses, synthesizing comprehensive reports from multiple data sources.
- **Building Agentic Slackbots with Confluence**: A comprehensive tutorial details how to construct a **multi-agent Slackbot** that interacts with Confluence documents using **AWS services**.
   - This initiative highlights the potential for improved organizational efficiency by integrating structured content into chat interfaces.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1289429783486402591)** (105 messages🔥🔥): 

> - `Ollama concurrency`
> - `LlamaIndex project setup`
> - `RAG pipeline evaluation`
> - `Node metadata handling`
> - `Oracle retrieval in RAG Benchmark` 


- **Ollama's concurrency feature**: A user inquired about how to utilize concurrency with Ollama, and it was clarified that it is enabled by default.
   - A helpful [link to Ollama's concurrency handling](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-does-ollama-handle-concurrent-requests) was provided for further assistance.
- **LlamaIndex project pipeline guidance**: A member sought recommendations for processing complex PDFs in their LlamaIndex project and was advised to use [Llamaparse](https://github.com/run-llama/llama_parse) for optimal results.
   - Discussions on various document handling methods led to further insights into extracting relevant data effectively.
- **Challenges in RAG pipeline evaluation**: A user reported issues with evaluating their RAG pipeline using trulens due to an import error, prompting suggestions to check documentation and available metrics.
   - Clarifications on retrieving node IDs for ground truth in evaluation settings were extensively discussed, emphasizing the need to build a solid evaluation dataset.
- **Editing node metadata in LlamaIndex**: Users discussed the ability to edit metadata for each chunk of data in LlamaIndex, confirming that adding details like URLs is feasible through code snippets.
   - Guidance was provided on manipulating node metadata effectively to enhance data retrieval and indexing processes.
- **Insights on oracle retrieval and new benchmarks**: A member shared information on the new RAG benchmark dataset from Google, which introduced the concept of oracle retrieval.
   - It was noted that oracle retrieval relies on ground-truth annotations, presenting an upper-bound performance measure rather than a practical retrieval method.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern_retrieval/">Usage Pattern (Retrieval) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_vs_recursive_retriever/">Comparing Methods for Structured Retrieval (Auto-Retrieval vs. Recursive Retrieval) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/retrieval/retriever_eval/">Retrieval Evaluation - LlamaIndex</a>: no description found</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/faq.md#how-does-ollama-handle-concurrent-requests">ollama/docs/faq.md at main · ollama/ollama</a>: Get up and running with Llama 3.2, Mistral, Gemma 2, and other large language models. - ollama/ollama</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/retrieval/retriever_eval/.">Retrieval Evaluation - LlamaIndex</a>: no description found</li><li><a href="https://go.microsoft.com/fwlink/?linkid=2198766",">Go with the floe</a>: What's the perfect thing to do under the midnight </li><li><a href="https://github.com/run-llama/llama_index/discussions/15117">How to fit a multistep query decomposition with a custom chat engine built with a query pipeline · run-llama/llama_index · Discussion #15117</a>: How can I fit multistep query decomposition like this one into my custom chat engine: from llama_index.core.query_engine import MultiStepQueryEngine query_engine = index.as_query_engine(llm=gpt4) q...</li><li><a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.</li><li><a href="https://github.com/run-llama/llama_index/blob/a620a2661faabb49ba2f257bff7ae2ac04d0c12b/llama-index-core/llama_index/core/evaluation/retrieval/metrics.py#L457">llama_index/llama-index-core/llama_index/core/evaluation/retrieval/metrics.py at a620a2661faabb49ba2f257bff7ae2ac04d0c12b · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1289845610958032908)** (1 messages): 

> - `LLM Reasoning`
> - `Different Types of Reasoning` 


- **Defining LLM Reasoning Problems**: It's crucial to clarify the type of reasoning problem we're addressing before engaging with LLM reasoning.
   - An article was shared that details [various reasoning types](https://www.linkedin.com/posts/subham-kundu-2746b515b_llm-reasoning-is-becoming-a-very-important-activity-7246050519289413632-3LPo?utm_source=share&utm_medium=member_desktop) and evaluates LLM performance on these challenges.
- **Importance of Categorizing Reasoning**: Identifying the specific reasoning issues is essential to guiding the effectiveness of LLMs.
   - The article highlights that different reasoning challenges require unique approaches and evaluations.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1289324296069713984)** (21 messages🔥): 

> - `Channel posting guidelines`
> - `Humanoid Robots 2024 YouTube video`
> - `Innovations in UI/UX for LLMs`
> - `Robotics development challenges`
> - `Podcasting as a UI/UX interaction` 


- **Channel Posting Guidelines Clarified**: A member asked about the right channel for posting, leading to a clarification that https://link.to/channel is acceptable, despite not being directly related to Cohere. Another member warned that the channel is not a job portal, leading to a reminder about maintaining appropriate discussions.
   - *Hello!* and *yahalloo!* exchanges marked the welcome atmosphere in the channel as members joined.
- **Best Rundown of Humanoid Robots in 2024**: A member shared a [YouTube video titled](https://youtu.be/PyrDh6RQdYY?si=RDA-9SFzdcZbAsmP) 'Every Humanoid Robot 2024', claiming it is the best rundown of humanoid robots on the internet. It included a link to a comprehensive list of the bots and their manufacturers.
   - Conversations then transitioned towards discussing the current issues in robotics, such as compute, battery costs, and higher human labor charges, igniting a brainstorming session.
- **UI/UX Innovations for LLMs**: A member emphasized the need for innovations in UI/UX for human-model interactions, sharing insights on NotebookLM as a powerful tool for podcast creation from any content. They provided links to various audio transformations showcasing the potential of podcasting as an LLM interface format.
   - They noted that while LLMs advance rapidly, UI/UX often lags, arguing that podcasting can bypass traditional user engagement hurdles in AI interactions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/karpathy/status/1840112692910272898">Tweet from Andrej Karpathy (@karpathy)</a>: NotebookLM is quite powerful and worth playing with https://notebooklm.google/  It is a bit of a re-imagination of the UIUX of working with LLMs organized around a collection of sources you upload and...</li><li><a href="https://docs.cohere.com/">Cohere Documentation — Cohere</a>: Cohere&#x27;s API documentation helps developers easily integrate natural language processing and generation into their products.</li><li><a href="https://youtu.be/PyrDh6RQdYY?si=RDA-9SFzdcZbAsmP">Every Humanoid Robot 2024</a>: Best rundown of all the humanoid robots on the internet. Brought to you by Automate Construction. List of bots &amp; who makes them:https://automateconstruction....
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1289318982939316224)** (36 messages🔥): 

> - `RAG formatting queries`
> - `Cohere startup program`
> - `API billing questions`
> - `Multimodal captioning`
> - `Input token number concerns` 


- **RAG Formatting for Model Prompts**: Users discussed how to format **instructional headers** for RAG inclusions in prompts submitted to the LLM, indicating the need for clarity on concatenation.
   - One member mentioned the importance of including supporting information in a format the model expects, as well as termination methods for headers.
- **Exploring the Cohere Startup Program**: A user inquired about discounts for a startup team using Cohere, highlighting the high expense compared to competitors like **Gemini**.
   - Another user suggested applying to the **Cohere Startup Program**, which offers discounts and noted it could take time for applications to be processed.
- **Clarifying API Billing Procedures**: Queries arose regarding how **Cohere** bills for API usage, with confirmation that **billing occurs monthly**.
   - A user mentioned not finding an invoice in their account, prompting further discussion about the billing process.
- **Interest in Multimodal Captioning**: A user asked whether anyone is working on **multimodal captioning**, inviting an exchange of ideas and experiences.
   - Another participant showed enthusiasm, encouraging the discussion of projects related to multimodal captioning.
- **Input Token Number Discrepancies**: A user raised concerns about inaccuracies in their **input token number**, asserting they were underreported daily use.
   - They also discussed challenges in applying for discounts as they do not operate as a company, but rather as a startup team.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/startup-program-application">Startup Program Application</a>: Thank you so much for your interest in the Cohere Startup Program! We&#x27;re initially rolling out the program with a select group of customers, and would love to learn a bit more about your business...</li><li><a href="https://cohere.com/startup-program">Startup Program </a>: The Cohere Startup Program offers qualified Series B and earlier startups a unique opportunity for support, discounted API rates, and publicity.</li><li><a href="https://docs.cohere.com/reference/chat-stream-v2">Chat with Streaming — Cohere</a>: Generates a message from the model in response to a provided conversation. To learn more about the features of the Chat API follow our  Text Generation guides .   Follow the  Migration Guide  for inst...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1289302210202570874)** (23 messages🔥): 

> - `Fine-tuning Models`
> - `Chunking Data for Improved Output`
> - `System Message and API Migration Issues`
> - `Documentation Consistency`
> - `V1 to V2 Chat API Transition` 


- **Fine-tuning models is a challenge for flash card generation**: A member sought advice on fine-tuning a model for better flash card generation using notes and slide decks, noting the qualitative concerns of the output. They contemplated whether their unstructured data could be improved without the fine-tuning process.
   - Another member suggested using best practices for machine learning pipelines to enhance the task, emphasizing that **chunking data** could significantly boost the model's output.
- **Chunking dramatically improves model performance**: Members discussed the effectiveness of **chunking data**, particularly for PDF slide decks, to enhance model understanding of relevant content. They also mentioned exploring tools like **rerankers** to optimize results from large datasets.
   - The dialogue emphasized the principle that well-structured input can lead to **better qualitative output**, addressing the importance of data preparation in AI tasks.
- **API migration dialogue reveals key challenges**: As users transitioned from v1 to v2 of the chat API, issues were raised about the simultaneous use of **system messages and document parameters** impacting functionality. A member experienced difficulties and learned from others that it was a known bug that was subsequently addressed.
   - Another user confirmed that the current API structure still supports the older version, ensuring continuity for those migrating, while also highlighting **the need for systematic updates**.
- **Call for documentation improvements**: A member noted inconsistencies in API documentation, specifically around penalty ranges for parameters, calling for a more uniform presentation of the details. They suggested clearer standards for documenting minimum and maximum values to enhance user clarity.
   - Discussions around handling errors in the API highlighted the importance of consistent and comprehensible documentation for best user experience.
- **V1 to V2 transition proves generally positive**: Members expressed relief at finding that the v1 chat API still functions during migration, highlighting the scarcity of compelling reasons to revert from the newer version. Conversations revealed a generally optimistic view toward the improvements provided in v2 despite initial hiccups.
   - The community remains engaged, exchanging insights and solutions as they adapt to the newly implemented features of the v2 API.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1289328195514142858)** (2 messages): 

> - `Cultural Multilingual LMM Benchmark`
> - `Volunteer Translators`
> - `CVPR'2025 Paper Co-Authorship` 


- **MBZUAI Launches Cultural Multilingual LMM Benchmark**: MBZUAI is developing a **Cultural Multilingual LMM Benchmark** for **100 languages**, creating a multimodal dataset with translations into local languages.
   - They are seeking native translators as volunteers to help rectify mistakes, promising an invitation to co-author their paper upon task completion.
- **Call for Volunteer Translators Across Languages**: The languages needing assistance include **Indian**, **South Asian**, **African**, and **European** languages, with a broad list provided for potential volunteers.
   - *“This isn't a job-portal... so we don’t run that way”* was mentioned in response to the volunteer call, clarifying the nature of inquiries.
- **Networking Invitation for Translators**: Interested individuals can connect with the project lead via **LinkedIn** for more information and to express their language skills.
   - They encourage personal messages regarding interest in volunteering.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1289325996184703069)** (36 messages🔥): 

> - `OpenAI staff turnover`
> - `AI regulations`
> - `Legal decisions on AI datasets`
> - `Investment discussions`
> - `Public reactions to AI bills` 


- **OpenAI's talent exodus due to compensation demands**: Key researchers at OpenAI are seeking higher compensation, with **$1.2 billion** already cashed out from selling profit units as the company’s valuation rises. Leadership turnover is exacerbated by rival companies like Safe Superintelligence actively recruiting OpenAI talent.
   - Employees are threatening to quit over money issues while new CFO **Sarah Friar** is caught in the middle of these negotiations.
- **CA governor vetoes AI safety bill SB 1047**: Gov. **Gavin Newsom** vetoed SB 1047, a bill aimed at regulating AI firms, stating that it was not the best approach to protecting the public. Critics view the veto as a setback for oversight, while supporters argue for regulating based on clear capabilities rather than vague predictions.
   - Sen. **Scott Wiener** expressed disappointment that the governor did not provide feedback before the veto and emphasized the missed opportunity for California to lead in tech regulation.
- **Legal victory for LAION in copyright case**: LAION successfully defended against copyright infringement claims in the German case of **Kneschke v LAION**, where a photographer alleged misuse of his images. The court ruled that LAION only linked to images, rather than hosting any itself.
   - This ruling is significant for AI dataset use cases, as copyright discussions continue to shape the AI landscape.
- **OpenAI's concerns regarding investor relations**: OpenAI is reportedly no longer in discussions with **Apple** regarding an investment, as per the **WSJ**. This shift signifies the broader tension between OpenAI’s mission and its need to satisfy its investors.
   - As OpenAI approaches a potentially transformative financial point, relations with major investors are critical to its future directions.
- **Public reactions fuel discussions about AI**: Reactions to the vetoed AI safety bill show mixed opinions, where some believe the reasoning behind the veto is sound, emphasizing regulatory clarity. Many anticipate that legislation efforts will resurface in the coming year.
   - Discussions in the community highlight differing views on how regulations should reflect actual technology capabilities rather than speculative future scenarios.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/9/27/24255177/openai-safety-mira-murati-quit-sam-altman-cofounders-exodus">OpenAI was a research lab — now it’s just another tech company</a>: OpenAI may soon become a for-profit company with fewer checks and balances than before — the exact structure it was built to avoid.</li><li><a href="https://x.com/unusual_whales/status/1839837869399257373?s=46">Tweet from unusual_whales (@unusual_whales)</a>: JUST IN: Apple $AAPL is now reportedly no longer involved in discussions to invest in OpenAI or board discussions per WSJ</li><li><a href="https://sfstandard.com/2024/09/29/gavin-newsom-vetoes-controversial-ai-safety-bill/">Newsom vetoes controversial AI safety bill SB 1047</a>: The bill had become a flashpoint in Silicon Valley with tech figures like Elon Musk supporting the measure and others saying it would threaten the burgeoning AI industry in its early stages.</li><li><a href="https://www.technollama.co.uk/laion-wins-copyright-infringement-lawsuit-in-german-court">LAION wins copyright infringement lawsuit in German court</a>: Copyright AI nerds have been eagerly awaiting a decision in the German case of Kneschke v LAION (previous blog post about the case here), and yesterday we got a ruling (text of the decision in Germ…
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1289325582106099722)** (16 messages🔥): 

> - `PearAI controversy`
> - `Yann LeCun on research standards`
> - `OpenAI's transparency debate`
> - `Peer review critique`
> - `Research blog impact` 


- **PearAI Accused of Code Theft**: **PearAI** allegedly stole code from [Continue.dev](http://Continue.dev) and rebranded it without proper acknowledgment, sparking outrage and calls for accountability from investors like YC.
   - *For those who don’t know: PearAI stole code…* from an open-source community, and it raises ethical concerns about startup funding.
- **LeCun Calls Out Blog Post Standards**: Yann LeCun criticized the reliance on blog posts for establishing research validity versus the rigorous standards of peer-reviewed papers, emphasizing that technical *research cannot be substituted* by press releases.
   - *It’s OK to delude yourself into thinking it’s the best thing since slice bread…* highlights the tension between product pressure and research integrity.
- **Debate Over OpenAI's Transparency**: Critics question OpenAI's transparency, pointing out that referencing a blog does not equate to substantive communication of research findings, with one member stating that a **press release doesn’t mean much**.
   - Amidst the debate, some OpenAI employees assert that they are indeed open regarding their research communications.
- **Skepticism Over Peer Review**: Some members expressed skepticism regarding the effectiveness of **peer review**, arguing that much published research can be subpar while still being deemed valid.
   - The conversation reveals frustrations over the perceived lack of accountability in research publication processes.
- **Impacts of OpenAI's Research Blog**: Discussions on the **research blog** question if sharing insights such as CoTs is enough to inform the community, with some suggesting that the information may be cherry-picked.
   - Members shared mixed feelings on whether [openai.com](https://openai.com/index/learning-to-reason-with-llms/) adequately addressed the community’s concerns about transparency and thoroughness.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/polynoamial/status/1840416885189271890">Tweet from Noam Brown (@polynoamial)</a>: @ylecun @thomaspower @OpenAI Also, we say a decent amount in the research blog post https://openai.com/index/learning-to-reason-with-llms/ including sharing CoTs, which I think are extremely informati...</li><li><a href="https://x.com/doiftrue/status/1840414633573646806?s=46">Tweet from Jakob Finch (@doiftrue)</a>: @candyflipline @iamgingertrash For those who don&#39;t know: PearAI stole code from http://Continue.dev and passed it off as a startup they are &#39;building&#39; and just got funding for it: https://...</li><li><a href="https://x.com/ylecun/status/1840422017654210604">Tweet from Yann LeCun (@ylecun)</a>: I&#39;m sorry Noam, but a blog post does not come close to meeting the standards of reproducibility, methodology, acknowledgment of prior work, and fair comparison with the state of the art, that a te...</li><li><a href="https://x.com/polynoamial/status/1840441849011744809">Tweet from Noam Brown (@polynoamial)</a>: @ylecun @thomaspower @OpenAI I think it&#39;s the opposite. A lot of published research is frankly BS. Authors just need to delude 3 reviewers and an AC.  When releasing something that millions of peo...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1289312314008277114)** (13 messages🔥): 

> - `iPhone IAP subscriptions`
> - `Apple App Store management`
> - `Twitter security issues`
> - `Meeting with John Schulman`
> - `Community engagement on Twitter` 


- **Getting Access to iPhone IAP Subs**: A substack best seller announced gaining access to **iPhone In-App Purchase subscriptions**, signaling potential growth opportunities in mobile monetization.
   - This access provides an interesting glimpse into the implementation of these systems and their management.
- **Apple App Store Nightmare Unveiled**: Insights were shared about the **challenges of managing the Apple App Store**, underscoring its chaotic environment.
   - Discussions highlighted the complexities and frustrations developers face within this ecosystem.
- **Twitter Security Breaches Alarm**: A concerning tweet highlighted the hacking of a prominent Twitter account, emphasizing that it could happen to anyone in the tech space.
   - Discussions pointed out that this issue persists, with calls for increased safety awareness among users.
- **John Schulman Meeting on RLHF Insights**: An exciting announcement was made about a forthcoming meeting with **John Schulman** for advice on **Reinforcement Learning from Human Feedback (RLHF)** work.
   - This engagement reflects the collaboration and mentorship opportunities in the AI community.
- **Concerns Over Twitter's Maintenance**: A user expressed skepticism about Twitter's commitment to security, pointing out that the platform only has **three engineers** managing issues.
   - Comments suggested that the team’s effectiveness is hampered by distractions and low resources, impacting overall safety.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/gm8xx8/status/1840304990134411561">Tweet from 𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8)</a>: 🚨 @DrJimFan hacked</li><li><a href="https://x.com/sammcallister/status/1840800944264478772?s=46">Tweet from sam mcallister (@sammcallister)</a>: 🥹 @karpathy
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1290059810708131883)** (3 messages): 

> - `AI Memes`
> - `User Reactions` 


- **User's Reaction to Brutal Meme**: A user expressed surprise and amusement with the phrase *You make this??? Brutal* in response to an AI-generated meme.
   - Another user humorously claimed *I wish lol* when asked if they created the meme, clarifying it was sourced from a random AI memes account.
- **Discussion on Meme Authorship**: A conversation unfolded regarding the origin of a meme, where one user eagerly questioned if another created it.
   - The notion quickly turned into laughter as the responding user mentioned it was just from a random AI memes account.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1289468237620183091)** (36 messages🔥): 

> - `Course Material Access`
> - `Multi-Agent Systems Discussion`
> - `NotebookLM Inquiry`
> - `Training Schedule Inquiry`
> - `Research Proposal Discussion` 


- **Accessing Course Materials**: Students are inquiring about accessing course videos and materials after filling out the registration form.
   - Course material, including assignments and lecture recordings, can be found on the [course website](https://llmagents-learning.org/f24), with all assignments due by Dec 12th.
- **Debate on Multi-Agent vs. Single-Agent Systems**: A conversation centers around the effectiveness and necessity of multi-agent systems versus single-agent implementations for various projects.
   - It was noted that multi-agent systems could mitigate hallucinations and simplify context management, aiding in accurate responses from LLMs.
- **NotebookLM's Functionality**: Inquiries were made about whether NotebookLM operates as an agent application.
   - It's described as a RAG agent that summarizes text and generates audio, with users questioning its tech implementation in terms of multi-step processes.
- **Training Schedule Confirmation**: Students are seeking information on when the training sessions for their course will begin.
   - One member shared that they were told all three labs would be released on Oct 1st, although this was not a formal announcement.
- **Research Proposal on Super-Alignment**: A proposed research project aims to explore ethics within multi-agent systems, emphasizing the use of frameworks like AutoGen.
   - Challenges were highlighted regarding the implementation of such research without dedicated frameworks, noting potential limitations in simulation capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/f24">Large Language Model Agents</a>: no description found</li><li><a href="https://notebooklm.google.com/">no title found</a>: no description found</li><li><a href="https://docs.google.com/document/d/12XgfYC2_U4gFEN732GPpv5Axh5IAUbaT2t_UKeRGFb0/edit?usp=drivesdk">Research Proposal: Exploring Super-Alignment through Relative Ethics in Multi-Agent Systems using AutoGen</a>: Research Proposal: Exploring Super-Alignment through Relative Ethics in Multi-Agent Systems using AutoGen  Eric Moore - 9/28/2024  Abstract In the advent of advanced artificial intelligence and potent...</li><li><a href="https://www.reddit.com/r/LangChain/s/CHRT9AehcV">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.coursera.org/projects/ai-agentic-design-patterns-with-autogen?">AI Agentic Design Patterns with AutoGen</a>: Complete this Guided Project in under 2 hours. In AI Agentic Design Patterns with AutoGen you’ll learn how to build and customize multi-agent systems, ...</li><li><a href="https://www.reddit.com/r/LangChain/s/CHRT9Aeh">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/)** (1 messages): 

metakingkal: There is an example in Autogen site on how to build an Agent to play chess.
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1289499444399702078)** (27 messages🔥): 

> - `Cloud Storage Costs`
> - `Modal Pricing Structure`
> - `Tinygrad Matcher Optimization`
> - `Testing Strategies for Optimizers`
> - `Bounty Payment Methods` 


- **Cloud Storage Costs Competitive with Major Providers**: George mentioned that **storage and egress costs** will be less than or equal to major cloud providers, emphasizing cost considerations.
   - He further explained that expectations for usage might alter perceived costs significantly.
- **Modal's Payment Model Sparks Debate**: Modal's unique pricing where they charge by the second for compute resources has drawn attention, touted as **cheaper than traditional hourly rates**.
   - Members questioned the sustainability of such models and how it aligns with consistent usage patterns in the AI startup environment.
- **Improving Tinygrad's Matcher with State Machines**: A member suggested that implementing a **matcher state machine** could improve performance, aligning it towards C-like efficiency.
   - George enthusiastically backed this approach, indicating it could achieve the desired performance improvements.
- **Need for Comprehensive Regression Testing**: Concerns were raised about the lack of a **regression test suite** for the optimizer, which could lead to unnoticed issues after code changes.
   - Members discussed the idea of serialization for checking optimization patterns, but recognized it would not be engaging.
- **Bounty Payment Options Discussed**: A user queried if bounties could be paid through **Payoneer** instead of PayPal, though George pointed to existing protocols in their questions document.
   - This reflects ongoing dialogue regarding payment systems within the community.



**Link mentioned**: <a href="https://modal.com/pricing">Plan Pricing</a>: Simple, transparent pricing that scales based on the amount of compute you use.

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1289915687350960159)** (4 messages): 

> - `SOTA GPU for Bounties`
> - `Renting GPUs Online`
> - `TF32 Tensor Core Support`
> - `Learning Before Tackling Bounties`
> - `Small PR Contributions` 


- **SOTA GPU not mandatory for bounties**: A member suggested that while a **SOTA GPU** could help, one can manage with an average GPU, especially for certain tasks.
   - Some tasks like **100+ TFLOPS matmul in tinygrad** may require specific hardware like the **7900XTX**, while others do not.
- **Renting GPUs for tasks**: It was mentioned that you can **rent a GPU online for cheap** if necessary, providing flexibility for those who don’t own high-end hardware.
   - This cost-effective approach allows participating in bounties without the need for a permanent high-performance setup.
- **Understanding TF32 tensor core support**: A user inquired about 'TF32 tensor core support,' indicating interest in performance capabilities.
   - It's advised to grasp these concepts thoroughly before attempting bounties to ensure success.
- **Importance of preparation before tackling bounties**: A strong recommendation was made to spend time learning the codebase before attempting a bounty, as it simplifies the process.
   - Familiarizing oneself with **open PRs** and existing issues can help avoid conflicts and ease the onboarding process.
- **Starting small with PR contributions**: It was suggested to begin with a **small PR** before engaging in more significant bounty tasks.
   - Keeping an eye on GitHub issues and Discord channels can reveal tasks that need attention and provide a pathway for contributing.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1290026403533819956)** (14 messages🔥): 

> - `Llama 3.2 1b tuning`
> - `California AI training bill`
> - `Lightweight chat models`
> - `Liquid AI`
> - `Sample packing effects` 


- **Concerns over Llama 3.2 1b Tuning**: A user reported issues tuning **Llama 3.2 1b**, experiencing high VRAM usage at **24GB** even with settings like qlora and 4bit loading.
   - Questions were raised about the impact of increasing sequence length compared to batch size, particularly with sample packing enabled.
- **California Enacts AI Training Disclosure Law**: A new **California law** mandates disclosure of training sources for any AI model used in the state, leaving no exceptions for smaller models or nonprofits.
   - This law raises discussions on potential workarounds using lightweight chat models to create 'inspired by' datasets that comply legally, as suggested by various members.
- **Foray into Lightweight Chat Models**: Members discussed the idea of finetuning lightweight chat models to transform webcrawled datasets while maintaining a legal standard of transformation.
   - One member noted that since **raw webcrawl data** is often messy, LLMs could assist in cleaning it up as a beneficial next step.
- **Excitement Around Liquid AI**: A new foundation model called **Liquid AI** has sparked interest among members in the discussion.
   - Some expressed curiosity about the implications and features of this new model, considering recent legislative changes.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1289324517679825021)** (3 messages): 

> - `LoRA+ implementation`
> - `Learning Rate Default Values`
> - `PEFT's Implementation` 


- **Question on Default Value Usage**: A member questioned if they should use a default value for a parameter or the same value as the **learning_rate**.
   - They noted that the [LoRA+ paper](https://link.to.lorapluspaper) set **1e-6** as their main learning rate, which could explain the default for **loraplus_lr_embedding**.
- **Assumption on Default from Paper**: Another member agreed with the assumption that the default value comes from the **LoRA+ paper** due to its usage of **1e-6**.
   - *Due to Pydantic defaulting to None*, the shift towards **PEFT's implementation** required slight adjustments.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1289682244180381797)** (12 messages🔥): 

> - `Axolotl dataset configuration`
> - `Selecting random dataset samples`
> - `Hugging Face datasets handling` 


- **Using 20% of a dataset in Axolotl**: In Axolotl, you can specify to use a portion of a dataset by utilizing the `split` option under the `datasets` configuration, allowing you to define custom splits.
   - For example, you can set your config to use the first 20% of a dataset for training, with adjustments available for validation and testing splits.
- **Randomly selecting a subset of data**: There is no direct option in the Axolotl config to use a random 20% of a dataset; this needs to be done during dataset loading or preprocessing.
   - Utilizing libraries like Hugging Face’s `datasets`, you can sample a random 20% before passing the processed dataset to Axolotl.
- **Llama 3 example cited**: A user suggested checking the Llama 3 example for potentially relevant configurations regarding dataset handling in Axolotl.
   - This suggests that there may be implicit methods or practices outlined in existing examples that could address the use of random samples.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=76ea009a-88eb-4421-bba0-01c6c3c35516)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=6d0af173-99ff-4f56-b361-5bbc2256f689)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1289648270834008145)** (2 messages): 

> - `Pydantic model generator`
> - `Groq integration`
> - `GitHub Actions`
> - `Typed Predictors`
> - `DSPyGen` 


- **Livecoding Free Pydantic Model Generator**: A session is underway demonstrating how to create a **free Pydantic model generator** utilizing [Groq](https://groq.com) and **GitHub Actions**.
   - The project aims to enhance model generation capabilities within **DSPyGen**, allowing for more typed predictors and streamlined processes.
- **Loom Video of the Session**: A member shared a [Loom video](https://www.loom.com/share/783ed4d80720492da23f39d2678de27f) that captures the live coding session in detail.
   - The video provides insights into the coding approach and tools used during the demonstration, valuable for participants and onlookers.



**Link mentioned**: <a href="https://www.loom.com/share/783ed4d80720492da23f39d2678de27f">I am still King of Typed Output in DSPy</a>: In this video, I demonstrate the creation of type predictors in Pydantic, showcasing the process and outcomes of generating structured text. I walk through the steps of creating a type predictor gener...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1289319529431699546)** (17 messages🔥): 

> - `DSPy 2.5 & LM Client Upgrade`
> - `Miprov2 Status & Issues`
> - `Optimizing System Prompts in DSPy` 


- **Upgrading to DSPy 2.5 brings big improvements**: Upgrading to **DSPy 2.5** and using the **LM client** with a **Predictor** instead of a **TypedPredictor** fixed many issues and resulted in a better out of the box performance.
   - *Curiously,* the improvements were linked to the new **Adapters** being better aware of **chat LMs**.
- **Miprov2 issues were user-related**: Concerns about **miprov2** being broken were clarified, revealing that the issue was in the user's **LM client** and not related to **MIPRO** itself.
   - The community discussed improving error handling by making `provide_traceback` on by default for **dspy.Evaluate** calls.
- **Optimizing System Prompts in DSPy**: A user expressed the need for guidance on how to manually input a system prompt into **DSPy** for optimization.
   - Others advised using the **DSPy documentation** to engage with the platform for custom prompt optimization.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1290060756783726716)** (8 messages🔥): 

> - `OpenSearchRetriever for DSPy`
> - `Healthcare Fraud Classification`
> - `Long Docstring Confusion`
> - `Using GPT-4o Mini and Claude Models` 


- **OpenSearchRetriever Offer**: A member offered to share their built [OpenSearchRetriever for DSPy](https://link.to.github) if there's interest in it among the community.
   - *Chiggly007* encouraged them to share the code or commit a PR, suggesting it would be helpful for others.
- **Struggles Classifying Healthcare Fraud**: A member is classifying Department of Justice press releases about **healthcare fraud** into three categories but is struggling with accuracy.
   - They noted that the module misclassifies medically unnecessary care billed as upcoding, calling for a better approach to defining class criteria.
- **Confusion from Long Docstrings**: The member pointed out issues with accuracy when using long explanations in the **docstring** for class signatures.
   - *Okhattab* affirmed there’s nothing wrong with detailed docstrings but asked which language model is in use.
- **Exploring Language Models**: The member is currently using **GPT-4o Mini** and planning to test **Claude models** for final classification.
   - They discussed grappling with token limits, while using public data scraped from the US Department of Justice's website.
- **Potential Data Benchmarking**: *Okhattab* suggested that the public data could be accessed to create a benchmark and related notebooks.
   - They reached out to the member via DM for further discussion on this possibility.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1289308501289603197)** (5 messages): 

> - `Full-stack development`
> - `AI execution instructions`
> - `Open Interpreter functionalities` 


- **Full-Stack Developer Seeking New Clients**: A skilled full-stack developer announced their expertise in building **e-commerce platforms**, online stores, and real estate websites using **React + Node** and **Vue + Laravel**.
   - They expressed interest in connecting with new reliable clients for long-term projects and are open to direct messages for potential collaborations.
- **Request to Modify AI Execution Instructions**: A member raised the question of whether it could be possible to **reinstruct the execution instructions** of the AI to allow users to fix and debug issues independently.
   - They mentioned encountering frequent errors related to paths, expressing frustration with current capabilities.
- **Inquiry about Open Interpreter's Purpose**: A member expressed confusion about the actual functionalities of **Open Interpreter**, questioning whether it performs specific tasks.
   - Their inquiry sparked interest in clarifying the AI's capabilities and overall offerings.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1289301733473914972)** (9 messages🔥): 

> - `Error decoding packet`
> - `Connection issues with client`
> - `Ngrok error` 


- **Error decoding packet issue**: A user reported a recurring **decoding packet error**: *Invalid data found when processing input* during server restarts or client connections.
   - Another member suggested checking for terminal error messages but confirmed there were none, indicating it happens consistently.
- **Client connection troubles**: A user mentioned their phone is stuck on the *Starting...* page when trying to connect.
   - One member encouraged posting setup details in a designated channel for further assistance.
- **Ngrok authentication problem**: A member expressed frustration with an **ngrok error** indicating a need for a verified account and authtoken while running their server.
   - They speculated whether the issue might stem from it not reading the *apikey* from the .env file, seeking help for this perceived trivial issue.
- **Demo of Open Interpreter usage**: A member shared a [YouTube video](https://www.youtube.com/watch?v=4TNzwKuq_yg) demonstrating the process of flashing a variety of **01's** using Open Interpreter based software.
   - The video provides visual guidance on software capabilities, though additional descriptions were not provided.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ngrok.com/docs/errors/err_ngrok_4018/">ERR_NGROK_4018 | ngrok documentation</a>: Message</li><li><a href="https://www.youtube.com/watch?v=4TNzwKuq_yg">Human Devices 01 Flashing Demo</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1289309881601495120)** (2 messages): 

> - `Open Interpreter impact`
> - `Using Jan with Open Interpreter`
> - `Local LLMs interface` 


- **Open Interpreter transforms lives**: One year ago, a member demonstrated a new tool that sparked a viral reaction and since then, **Open Interpreter** has greatly impacted their life, helping them make incredible friends and dive into the **A.I. world**.
   - They expressed gratitude for the community's support, stating, *'Let's keep building an amazingly abundant future.'*
- **Jan AI serves as computer control interface**: A member inquired if others have used **Jan** and pointed out its compatibility with **Open Interpreter**, highlighting its functionality as a local inference server for local LLMs.
   - They shared a [YouTube video](https://www.youtube.com/watch?v=1l3B0AzbbjQ) titled *'Control Your Computer with Jan AI'*, which explains how Jan can interface to control your computer.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/MikeBirdTech/status/1839750338179674590">Tweet from Mike Bird (@MikeBirdTech)</a>: One year ago today, I made a little demo of this cool new tool I found online. Just wanted to show off what it could do and then it went a little viral  Since then @OpenInterpreter has completely chan...</li><li><a href="https://www.youtube.com/watch?v=1l3B0AzbbjQ">Control Your Computer with Jan AI</a>: Jan.AI is a great local inference server for serving local LLMs. But did you know you can use it as an interface to control your computer? Jan: https://jan.a...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1289324677646520340)** (8 messages🔥): 

> - `French Audio Dataset for CosyVoice`
> - `LAION Copyright Challenge`
> - `Phenaki Video Generation Model`
> - `Visual Language Models and Latent Diffusion Models`
> - `PALM-RLHF Datasets and Task Implementation` 


- **Seeking French Audio Datasets for CosyVoice**: A user requested high-quality audio datasets in **French** for training **CosyVoice**.
   - They expressed the need for suitable datasets to proceed with their project.
- **LAION Wins Copyright Challenge in Germany**: A thread highlighted that **LAION** won the first copyright infringement challenge in a **German court**.
   - The post included a link for further discussion and details on this legal victory.
- **Testing Phenaki for Text-to-Video Generation**: A user explored the **Phenaki** implementation for generating videos from text and provided a [GitHub link](https://github.com/lucidrains/make-a-video-pytorch) for testing.
   - They sought guidance for initial testing before training due to a lack of datasets.
- **Combining Visual Language and Latent Diffusion Models**: Discussion emerged on the potential of combining **VLM** (Visual Language Models) and **LDM** (Latent Diffusion Models) to improve image generation processes.
   - Theoretical aspects included the possibility of a loop where VLM generates instructions for LDM, refining outputs effectively.
- **Implementing PALM-RLHF Training Datasets**: A user inquired about the most suitable channel and role for implementing **PALM-RLHF** training datasets for specific tasks.
   - They sought clarity on the process to align training datasets with specific operational needs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/aiwars/comments/1fqpiut/laion_wins_first_copyright_infringement_challenge/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/lucidrains/make-a-video-pytorch">GitHub - lucidrains/make-a-video-pytorch: Implementation of Make-A-Video, new SOTA text to video generator from Meta AI, in Pytorch</a>: Implementation of Make-A-Video, new SOTA text to video generator from Meta AI, in Pytorch - lucidrains/make-a-video-pytorch
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1289326572343398421)** (7 messages): 

> - `Transformer Models`
> - `Positional Encodings`
> - `RoPE in Attention Layers`
> - `Convergence Time in Training` 


- **Transformers set to become the dominant architecture**: A member mentioned that ultimately, it may lead to one large *transformer model*, hinting at the growing reliance on this architecture in AI.
   - They shared a [link to the emu project](https://emu.baai.ac.cn/about) which explores various aspects of this development.
- **Positional Encodings might simplify architecture**: Members discussed the idea of using **positional encodings** in *transformer blocks*, suggesting it could yield cleaner implementations.
   - *One member confirmed* that position information is already integrated into the features of the layers they studied.
- **RoPE attempted in U-Net for Attention**: A member shared their experience of trying **RoPE** with *U-Net* for the attention layers, indicating interest in its impact on performance.
   - They noted uncertainty about whether this approach affects overall convergence time.
- **Propagation of Position Information in Layers**: A member pointed out that it takes some **1D padded convolution layers** for position information to fully propagate across the grid.
   - They suggested that if position has utility early on, it could significantly influence results.



**Link mentioned**: <a href="https://emu.baai.ac.cn/about">Emu3</a>: no description found

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1289370604281139212)** (6 messages): 

> - `Vectorstores interaction`
> - `Database usage for LLMs`
> - `Thank you gifts in Discord`
> - `Image errors in Gemini`
> - `Modifying inference method in LangChain` 


- **Vectorstores may need example questions**: A member suggested that utilizing example questions could aid in vectorstores looking for the closest match, although it might be overkill.
   - They emphasized the need for testing to determine effectiveness.
- **Database preferred over table data**: A member explained that table data is not ideal for LLMs, prompting them to transfer their table data into a **Postgres** database.
   - They are now using **LangChain modules** to interact with this database.
- **Thank you gift inquiry**: A member asked whether it is possible to send a small thank you gift as a token of appreciation to someone who helped in the Discord.
   - They expressed a desire to acknowledge contributions made by others.
- **Sudden image errors in Gemini**: A member reported encountering sudden errors when sending images to **Gemini**, which previously worked fine.
   - They suspect the issue might have arisen after upgrading all **pip packages**.
- **Modifying LangChain inference methods**: A member is exploring ways to modify the inference method of chat models using **LangChain** while incorporating optimizations in **vllm**.
   - They are interested in controlling how the LLM decodes tokens, particularly with the open-ended invocation of chat history and input.


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1289657420137762958)** (3 messages): 

> - `AI Realized Summit 2024`
> - `Manifold Research Frontiers Series`
> - `MLOps meetups in Stockholm` 


- **AI Realized Summit 2024 Set for October 2**: Excitement is building for the [AI Realized - The Enterprise AI Summit](https://lu.ma/airsummit) on October 2, 2024, hosted by **Christina Ellwood** and **David Yakobovitch** at UCSF, featuring industry leaders in Enterprise AI.
   - Attendees can use code **extra75** to save **$75 off** their tickets, which include meals at the conference.
- **Kickoff of Manifold Research Frontiers Talks**: **Manifold Research** is launching the Frontiers series to spotlight innovative work in foundational and applied AI, starting with a talk by **Helen Lu** focused on neuro-symbolic AI and human-robot collaboration.
   - The talk will discuss challenges faced by autonomous agents in dynamic environments and is open for free registration [here](https://lu.ma/cbflyi6s).
- **Inquiry on MLOps Meetups in Stockholm**: A member is seeking information about **MLOps or Infrastructure meetups** in Stockholm after recently moving to the city.
   - They expressed a desire to connect with the local tech community and learn about upcoming events.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lu.ma/airsummit">AI Realized – The Enterprise AI Summit · Luma</a>: Welcome to AI Realized Summit 2024! ...hosted by Christina Ellwood and David Yakobovitch Join us in San Francisco on October 2nd, 2024 for an exclusive one-day…</li><li><a href="https://lu.ma/cbflyi6s">Frontiers: Neuro-Symbolic Adaptation for Autonomous Agents · Zoom · Luma</a>: Welcome to Frontiers - a series where we bring top researchers, engineers, designers, and leaders working at the cutting edge of various fields to go deep on…</li><li><a href="https://www.manifoldrg.com/events/">Manifold Research Group (Page 1)</a>: no description found</li><li><a href="https://www.manifoldrg.com">Manifold Research Group</a>: Manifold Research is a new kind of R&amp;D Institute pursuing high impact frontier science and technology projects with the ultimate goal of improving and advancing human civilization.
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/)** (1 messages): 

zachmayer: Surya
  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1289413018567835693)** (3 messages): 

> - `Anti-slop Sampler`
> - `Dataset Creation` 


- **Calytrix introduces anti-slop sampler**: A prototype anti-slop sampler has been developed to suppress unwanted words/phrases during inference by backtracking when an unwanted sequence is detected.
   - Calytrix is working on making the codebase usable for downstream purposes and shared the project on [GitHub](https://github.com/sam-paech/antislop-sampler).
- **Community supports the anti-slop concept**: A member expressed appreciation for the anti-slop sampler idea, noting, *'cool, I like the idea!'*
   - The positive feedback indicates interest in innovative approaches to improving dataset quality.



**Link mentioned**: <a href="https://github.com/sam-paech/antislop-sampler">GitHub - sam-paech/antislop-sampler</a>: Contribute to sam-paech/antislop-sampler development by creating an account on GitHub.

  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1290379411245236314)** (1 messages): 

> - `SoraSNS`
> - `Takiyoshi Hoshida`
> - `Carnegie-Melon University`
> - `Apple's AR Kit` 


- **Takiyoshi Hoshida to Demo SoraSNS**: Indie developer **Takiyoshi Hoshida** is set to present a live demo of his project [SoraSNS](https://discord.com/events/1089876418936180786/1277835047084363827), a social media app offering a private timeline from users you don't typically follow.
   - The demo will highlight the app's concept of **day and night** skies, symbolizing openness and distant observation, allowing users to discover new parts of the social network.
- **Hoshida's Impressive Background**: Hoshida studied **Computer Science at Carnegie-Melon University**, boasting significant experience in the tech field.
   - He has previously worked with **Apple's AR Kit team** and contributed to over **50 iOS projects**.


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1290268665299734528)** (1 messages): 

> - `Hammer handle update`
> - `Hammer2.0 series models`
> - `Pull Request submission` 


- **Hammer Handle Gets a Refresh**: The hammer handle has been updated, signaling some enhancements in design and functionality.
   - *Exciting improvements* are expected with this new iteration.
- **Introducing the Hammer2.0 Series**: The team has launched the **Hammer2.0 series models** including Hammer2.0-7b, Hammer2.0-3b, Hammer2.0-1.5b, and Hammer2.0-0.5b.
   - These additions mark a significant step in product diversification.
- **Pull Request PR#667 Submitted**: A Pull Request (PR#667) has been submitted as part of the updates to the hammer product line.
   - This submission is a key part of the development process following the recent enhancements.


  

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

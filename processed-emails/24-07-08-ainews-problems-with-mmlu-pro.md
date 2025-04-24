---
id: 4c58d595-fd90-448e-87dc-e3d95a3f7c54
title: Problems with MMLU-Pro
date: '2024-07-09T00:20:51.624419Z'
original_slug: ainews-et-tu-mmlu-pro
description: >-
  Recent AI news covers the launch of MMLU-Pro, its evaluation issues, prompt
  engineering impacts, and various AI developments including Meta's MobileLLM,
  Salesforce's APIGen, Runway Gen-3, Nomic AI GPT4All 3.0, and AI tools for
  vision, audio, and 3D generation, as well as deepfake video creation and
  transformer reasoning research.
companies:
  - meta
  - salesforce
  - runway
  - nomic-ai
  - pineapple
  - brivael-lp
models:
  - mmlu-pro
  - llama-3-8b-q8
  - gpt4all-3.0
  - claude-3-opus
topics:
  - benchmark
  - large-language-models
  - prompt-engineering
  - multimodal
  - ai-vision
  - ai-audio
  - ai-video
  - deepfake
  - transformers
  - reasoning
---


<!-- buttondown-editor-mode: plaintext -->**Reading benchmark code is all you need.**

> AI News for 7/5/2024-7/8/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**462** channels, and **4661** messages) for you. 
Estimated reading time saved (at 200wpm): **534 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

There's been a lot of excitement for [MMLU-Pro](https://x.com/_philschmid/status/1791137274337354166) replacing the saturated MMLU, and, ahead of [Dan Hendrycks making his own update](https://x.com/DanHendrycks/status/1804929811703591345), HuggingFace has already anointed MMLU-Pro the successor in the [Open LLM Leaderboard V2](https://huggingface.co/spaces/open-llm-leaderboard/blog) (more in an upcoming podcast with Clementine). It's got [a lot of improvements over MMLU](https://x.com/WenhuChen/status/1790597967319007564)... 

 ![image.png](https://assets.buttondown.email/images/6d389db1-b599-49fb-a88b-0209f7f8a29c.png?w=960&fit=max) 

but... the good folks at /r/LocalLlama have been digging into it and finding issues, first with [math heaviness](https://www.reddit.com/r/LocalLLaMA/comments/1du52gf/mmlupro_is_a_math_benchmark/), but today more damningly some alarming discrepancies in how models are evaluated by the MMLU-Pro team across sampling params, system prompts, and answer extraction regex:

 ![image.png](https://assets.buttondown.email/images/8aa9d3eb-510c-49cd-a9e1-78a646bb60e4.png?w=960&fit=max) 

For their part, the MMLU-Pro team acknowledge the discrepancies (both between models and between the published paper and what the code actually does) but [claim that their samples have minimal impact](https://github.com/TIGER-AI-Lab/MMLU-Pro/issues/5#issuecomment-2213291392), but the community is correctly pointing out that [the extra attention and customization paid to the closed models disadvantage open models](https://www.reddit.com/r/LocalLLaMA/comments/1dw8l3j/comment/lbu6efr/).

Experience does tell us that current models are still highly sensitive to prompt engineering, and **simple tweaks of the system prompt improved Llama-3-8b-q8's performance by 10 points (!!??!)**.

 ![image.png](https://assets.buttondown.email/images/e244cd9b-9f8b-45e7-a0bb-b99db1cbc59d.png?w=960&fit=max) 


Disappointing but fixable, and maintaining giant benchmarks are always a messy task, yet one would hope that these simple sources of variance would have been controlled better given the high importance we are increasingly placing on them.



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

**AI Developments**

- **Meta's MobileLLM**: [@ylecun](https://twitter.com/ylecun/status/1810035281472491665) shared a paper on running sub-billion LLMs on smartphones using techniques like **more depth, shared matrices, and shared weights between transformer blocks**.
- **APIGen from Salesforce**: [@adcock_brett](https://twitter.com/adcock_brett/status/1809981480052916275) highlighted new research on an **automated system for generating optimal datasets for AI training on function-calling tasks**, outperforming models 7x its size.
- **Runway Gen-3 Alpha**: [@adcock_brett](https://twitter.com/adcock_brett/status/1809981671606735253) announced the **AI video generator is now available to all paid users**, generating realistic 10-second clips from text and images.
- **Nomic AI GPT4All 3.0**: [@adcock_brett](https://twitter.com/adcock_brett/status/1809981693979201932) shared the new open-source LLM desktop app supporting **thousands of models that run locally and privately**.

**AI Agents and Assistants**

- **AI Assistant with Vision and Hearing**: [@svpino](https://twitter.com/svpino/status/1809921844297732268) built an AI assistant in Python that **sees and listens**, with step-by-step video instructions.
- **ChatLLM from Pineapple**: [@svpino](https://twitter.com/svpino/status/1810026351514321031) released an AI assistant providing access to **ChatGPT, Claude, Llama, Gemini and more for $10/month**.

**AI Art and Video**

- **Meta 3D Gen**: [@adcock_brett](https://twitter.com/adcock_brett/status/1809981569857114600) shared Meta's new AI system that **generates high-quality 3D assets from text prompts**.
- **Argil AI Deepfake Videos**: [@BrivaelLp](https://twitter.com/BrivaelLp/status/1809898328668209383) used Argil AI to **convert a Twitter thread into a deepfake video**.

**AI Research and Techniques**

- **Grokking and Reasoning in Transformers**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1809950057086530019) shared a paper on how transformers can **learn robust reasoning through extended 'grokking' training beyond overfitting**, succeeding at comparison tasks.
- **Searching for Best Practices in RAG**: [@_philschmid](https://twitter.com/dair_ai/status/1809878384526139782) summarized a paper identifying **best practices for Retrieval-Augmented Generation (RAG) systems** through experimentation.
- **Mamba-based Language Models**: [@slashML](https://twitter.com/slashML/status/1809881609316815175) shared an empirical study on **8B Mamba-2-Hybrid models trained on 3.5T tokens of data**.

**Robotics Developments**

- **Open-TeleVision for Tele-Op Robots**: [@adcock_brett](https://twitter.com/adcock_brett/status/1809981502702145951) shared an **open-source system from UCSD/MIT allowing web browser robot control from thousands of miles away**.
- **Figure-01 Autonomous Robots at BMW**: [@adcock_brett](https://twitter.com/adcock_brett/status/1809981547551817990) shared new footage of **Figure's robots working autonomously at BMW using AI vision**.
- **Clone Robotics Humanoid Hand**: [@adcock_brett](https://twitter.com/adcock_brett/status/1809981779194839193) highlighted a Polish startup building a **human-like musculoskeletal robot hand using hydraulic tendon muscles**.

**AI Culture and Society**

- **Concerns about AI Elections**: [@ylecun](https://twitter.com/ylecun/status/1810065581174931806) pushed back on claims that the **French far-right was "denied victory"**, noting they simply did not win a majority of votes.
- **Personality Basins as a Mental Model**: [@nearcyan](https://twitter.com/nearcyan/status/1810099024764289026) shared a post on using the concept of **"personality basins" as a mental model for understanding people's behavior over time**.
- **Increased LLM Usage**: [@fchollet](https://twitter.com/fchollet/status/1810025103054479459) polled followers on **how often they have used LLM assistants in the past 6 months compared to prior**.

**Memes and Humor**

- **Cracked Kids and Greatness**: [@teortaxesTex](https://twitter.com/teortaxesTex/status/1810066303165616326) joked that **those who are truly great do not care about the bitter lessons of "cracked" kids**.
- **Developers Trying to Make AI Work**: [@jxnlco](https://twitter.com/jxnlco/status/1809975279802003562) shared a meme about **the struggles of developers trying to get AI to work in production**.
- **AI Freaks and Digital Companionship**: [@bindureddy](https://twitter.com/bindureddy/status/1810042560271794456) joked about **"AI freaks" finding digital companionship and roleplaying**.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**Technology Advancements**

- **AI model training costs rapidly increasing**: In /r/singularity, Anthropic's CEO stated that AI models costing [$1 billion to train are underway, with $100 billion models coming soon](https://www.reddit.com/r/singularity/comments/1dy294l/ai_models_that_cost_1_billion_to_train_are/), up from the current largest models which take "only" $100 million to train. This points to the exponential pace of AI scaling.
- **Lifespan extension breakthrough in mice**: In /r/singularity, [Altos Labs extended the lifespan of mice by 25% and improved healthspan](https://www.reddit.com/r/singularity/comments/13uoq9o/altos_labs_extends_lifespan_of_mice_by_25_and/) using Yamanaka factor reprogramming, a significant achievement by a leading AI and biotech company in anti-aging research.
- **DeepMind AI generates audio from video**: In /r/singularity, [DeepMind's new AI found the "sound of pixels"](https://www.reddit.com/r/singularity/comments/13v1lz1/deepminds_new_ai_found_the_sound_of_pixels_by/) by learning to generate audio from video, demonstrating advanced multimodal AI capabilities linking visuals with associated sounds.

**Model Releases and Benchmarks**

- **Llama 3 finetunes underperform for story writing**: In /r/LocalLLaMA, one user found that [Llama 3 finetunes are terrible for story writing compared to Mixtral and Llama 2 finetunes](https://www.reddit.com/r/LocalLLaMA/comments/13v0yrm/llama_3_finetunes_are_terrible_for_story_writing/), as the Llama 3 models go off the rails and don't follow prompts well for long-form story generation.
- **Open-source InternLM2.5-7B-Chat model shows strong capabilities**: In /r/ProgrammerHumor, [InternLM2.5-7B-Chat, an open-source large language model, demonstrates unmatched reasoning, long-context handling, and enhanced tool use](https://www.reddit.com/r/ProgrammerHumor/comments/13v0yrk/internlm257bchat_an_opensource_large_language/), pushing the boundaries of open-source AI capabilities.
- **User benchmarks 28 AI models on various tasks**: In /r/singularity, [a user ran small-scale personal benchmarks on 28 different AI models](https://www.reddit.com/r/singularity/comments/13v0yrj/i_ran_smallscale_personal_benchmarks_on_28/), testing reasoning, STEM, utility, programming, and censorship. GPT-4 and Claude variants topped the rankings, while open models like Llama and GPT-J trailed behind, with detailed scoring data provided.
- **Default MMLU-Pro prompt suboptimal for benchmarking Llama 3**: In /r/LocalLLaMA, it was found that [the default MMLU-Pro system prompt is really bad for benchmarking Llama 3 models](https://www.reddit.com/r/LocalLLaMA/comments/1dxpns0/default_mmlupro_system_prompt_is_really_bad/), leading to inconsistent results, and modifying the prompt can dramatically improve model performance on this benchmark.

**Discussions and Opinions**

- **Concerns over LMSYS AI leaderboard validity**: In /r/singularity, it was argued that [LMSYS, a popular AI leaderboard, is inherently flawed and should not be used as a benchmark anymore](https://www.reddit.com/r/singularity/comments/1dxcyav/lmsys_is_inherently_flawed_and_should_not_be_used/) due to the potential for manipulation and inconsistent results, emphasizing the need for alternative evaluation methods.
- **Lessons learned in building AI applications**: In /r/ProgrammerHumor, [a user asked for the biggest lessons learned when building AI applications](https://www.reddit.com/r/ProgrammerHumor/comments/13v0yri/what_are_the_biggest_lessons_youve_learned_when/). Responses emphasized having a solid evaluation dataset, using hosted models to start, and avoiding time sinks like endlessly tweaking frameworks or datasets.
- **Potential for training larger models on supercomputers**: In /r/singularity, [a question was posed about whether modern supercomputers are capable of training much larger models than current ones](https://www.reddit.com/r/singularity/comments/13v0yre/are_modern_supercomputers_capable_of_training/). The computational capacity seems to be there, but it's unclear if any such large-scale training is happening in secret.

**Memes and Humor**

- **Humorous meme image**: In /r/singularity, [a meme image asks "Where Are Ü Now?"](https://www.reddit.com/r/singularity/comments/13v0yrl/where_are_%C3%BC_now/) in a humorous tone, with no further context provided.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Advancements in Model Architectures and Training**

- **Hermes 2's Benchmark Brilliance**: The **Hermes 2** model and its improved version **Hermes 2.5** have shown significant performance gains in benchmarks, outperforming many other models in the field.
   - Community discussions highlighted that while Hermes 2 excels, other models like **Mistral** struggle to extend beyond 8k context without further pretraining. This sparked debates on model scaling and the potential of merging tactics for performance improvements.
- **BitNet's Binary Breakthrough**: [BitNet](https://arxiv.org/abs/2310.11453) introduces a scalable 1-bit weight Transformer architecture, achieving competitive performance while significantly reducing memory footprint and energy consumption.
   - This innovation in 1-bit models opens up possibilities for deploying large language models in resource-constrained environments, potentially democratizing access to advanced AI capabilities.
- **T-FREE's Tokenizer Transformation**: Researchers introduced [T-FREE](https://arxiv.org/abs/2406.19223), a tokenizer embedding words through activation patterns over character triplets, significantly reducing embedding layer size by over 85% while maintaining competitive performance.
   - This novel approach to tokenization could lead to more efficient model architectures, potentially reducing the computational resources required for training and deploying large language models.

**2. Innovations in AI Efficiency and Deployment**

- **QuaRot's Quantization Quest**: [Recent research](https://arxiv.org/abs/2404.00456) demonstrated the effectiveness of QuaRot for 4-bit quantization on LLMs, achieving near full-precision performance with significantly reduced memory and computational costs.
   - This advancement in quantization techniques could dramatically improve the efficiency of LLM deployments, making it possible to run powerful models on more modest hardware configurations.
- **MInference's Speed Boost for Long-context LLMs**: Microsoft's [MInference project](https://github.com/microsoft/MInference) aims to accelerate Long-context LLMs' inference, **trimming latency** by up to 10x on an **A100** GPU.
   - MInference employs novel techniques for approximate and dynamic sparse calculations, balancing accuracy with **performance efficiency**. This tool could significantly improve the real-world applicability of large language models in scenarios requiring rapid responses.
- **Cloudflare's AI Scraping Shield**: **Cloudflare** introduced a feature allowing websites to block AI scraper bots, potentially impacting data collection for AI training and raising concerns in the AI community.
   - While some worry about the implications for AI development, others believe that only websites actively trying to block AI will use this feature. This development highlights the growing tension between data accessibility and privacy in the AI era.


---

# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Licensing Labyrinth at Stability AI**: The community is actively discussing the new [Stability AI model licensing](https://stability.ai/news/license-update) terms, focusing on the implications for businesses exceeding the $1M revenue mark.
   - **Concerns** persist around the SD3 model's use for commercial applications, particularly affecting smaller enterprises.
- **Pixel Perfection: The Upscaling Odyssey**: An **upscale workflow** was shared, combining tools like Photoshop, [SUPIR](https://old.reddit.com/r/StableDiffusion/comments/1b50sp0/ccsr_vs_supir_upscale_comparison_portrait/), and others to produce high-res images while balancing detail and consistency.
   - This multi-step strategy seeks to tackle **tiling issues**, a common bottleneck in image upscaling.
- **Model Quality Maze**: Some members were **disappointed** with the SD3 model's quality, eliciting comparisons to predecessors, and speculated about the potential consequences of rushed releases.
   - A future 8B version is highly anticipated, alongside discussions on ethical considerations and the perceived influences of agencies like the NSA.
- **Troubleshooting Text2img: VRAM Crunch**: User experiences highlighted slowdowns when combining controlnet with text2img, tying these to VRAM constraints and necessitating **memory management**.
   - Effective mitigation techniques like optimizing Windows pagefile settings and offloading have been recommended to counteract the slowdowns.
- **Cultivating Creative Prompts**: The guild has been swapping insights on how to better utilize prompts and external integrations, like [github.com/AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings), to enhance image generation outcomes.
   - Advice includes the strategic use of language in prompts and the application of multiple tools for optimal image results.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Inference Endurance Fails to Impress**: Reports of long initialization times for inference endpoints have surfaced, indicating challenges with GPU availability or specific configuration settings; One member suggested evaluating [AWS's Nvidia A10G](https://aws.amazon.com/ec2/instance-types/a10g/) on the eu-west-1 region as a remedy.
   - The topic of efficiency surfaced with a member’s concern regarding GPTs agents' inability to learn post initial training, fostering a discussion on the limits of current AI models' adaptability.
- **Glossary Unchants AI Lingo Confusion**: **LLM/GenAI Glossary** was unveiled as a comprehensive guide with the intent to make AI jargon accessible. Prashant Dixit shared a [link](https://github.com/freetoolsarebest/llm-glossary) to the community-created glossary, which is regularly updated to aid learning and contribution.
   - The initiative aims to simplify technical communication within the AI community, highlighting the significance of clarity in a field ripe with complex terminology.
- **AI Creatives Assemble in HuggingFace Space**: The **ZeroGPU HuggingFace Space** announced by a member caters to an array of Stable Diffusion Models comparison, including **SD3 Medium**, **SD2.1**, and **SDXL** [available for experimentation](https://huggingface.co/spaces/Nick088/stable-diffusion-arena).
   - In the spirit of DIY, **qdurllm** emerged as a combination of **Qdrant**, **URL scraping**, and **Large Language Models** for local search and chat, with its open-source format prompting collaborative exploration on [GitHub](https://github.com/AstraBert/qdurllm).
- **Visionary Metrics for Object Detection**: A nod was given to Torchmetrics for improving object detection metrics, with its utilization highlighted in the [Trainer API and Accelerate example scripts](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection).
   - The RT-DETR model made waves as a real-time object detection offering, blending the efficiency of convolutions with attention-centric transformers as shown in this [tweet](https://x.com/mervenoyann/status/1807790959884665029), licensed under Apache 2.0.
- **Artifact Enigma in sd-vae Reconstructions**: Members embarked on a discussion about the normalcy of blue and white pixel artifacting in **sd-vae** and what it signifies for reconstruction outcomes.
   - Exploration of parameter adjustments emerged as a shared strategy for community-based troubleshooting of this phenomenon, underscoring the collaborative approach to refining sd-vae models.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Under Scrutiny**: Users find Perplexity often returns **outdated information** and struggles with context retention, lagging behind **GPT-4o** and **Claude 3.5** in fluidity of follow-ups.
   - The Pro version's lack of a significant boost over the free service sparks debate with suggestions of alternative services such as **Merlin.ai** and **ChatLLM**.
- **Shining a Light on Hidden Features**: Perplexity's image generation capability takes some by surprise, with Pro users guiding others on maximizing the feature through *custom prompt* options.
   - Technical hiccup discussions include text overlaps and context loss, with the community leaning on **system prompts** for temporary remedies.
- **Niche Nuggets in Community Knowledge**: A deep-dive into Minecraft survival methods unearthed with a [guide to mastering the underground](https://www.perplexity.ai/page/minecraft-underground-survival-hj7PsuozQ32xoJKudQqm8g), sparking strategical exchanges.
   - Insights from a user's average cost **research raises eyebrows**, while another seeks solidarity in the frustrations of setting up a new Google account.
- **API Woes and Wins**: The updated Perplexity **API** shows promise with improved multi-part query handling, but frustrations grow over delayed Beta access and long processing times.
   - Clear as mud, the relationship between **API and search page results** confounds users, with some feeling left in the dark about multi-step search API capabilities.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **MacBook M3 Praised for Model Handling**: The new **M3 MacBook Pro** with **128GB RAM** garnered positive attention for its capability to manage large models like **WizardLM-2-8x22B**, distinguishing itself from older versions with memory limitations.
   - Despite the inability to load **WizardLM-2-8x22B** on an M2 MacBook, the M3's prowess reinforces **Apple's stronghold** in providing **robust solutions** for large model inference workloads.
- **Gemma 2 Models Await Bug Fixes**: Community discourse focused on **Gemma 2 models** suffering slow inference and calculation errors, with users anticipating future updates to iron out these issues.
   - Discussion threads pinpointed references to [Gemma model architectural bugs](https://github.com/ggerganov/llama.cpp/pull/8348), suggesting that forthcoming improvements might address their current constraints.
- **Advancements in Model Quantization Discussed**: Users exchanged insights on advanced quantization methods, debating the best balance between **model performance** and **output quality**.
   - Links to [quantized models](https://huggingface.co/Joseph717171/Models/tree/main) were shared, spurring conversations about leveraging formats like F32 and F16 for **enhanced results**.
- **LM Studio's x64bit Installer Query Clarified**: In LM Studio's discussion channel, a user's confusion about the absence of a 64-bit installer was clarified, explaining that the existing x86 designation also includes 64-bit compatibility.
   - The transparency resolved misconceptions and highlighted **LM Studio's attentive community interaction**.
- **Fedora 40 Kinoite and 7900XTX Synergy Proves Solid**: A notable uptick in **generation speed** within **LM Studio** was confirmed after deploying updates, serving as a testament to the synergy between **Fedora 40 Kinoite** and **7900XTX GPU** configurations.
   - This development reflects ongoing strides in optimization, underscoring **speed enhancements** as a key focus for current AI tools.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Hermes Heats Up, Mistral Misses Mark**: Debate heats up over performance of **Hermes 2** versus **Hermes 2.5**, contrasting the enhanced benchmarks against **Mistral's** difficulty scaling beyond 8k without further pretraining.
   - Discussions delve into the potential for **merging tactics** to improve AI models; meanwhile, Cloudflare's recent feature entices mixed reactions due to its capability to block AI data scraping bots.
- **Custom GPTs Grapple With Zapier**: Community members express their experiences with **custom GPTs**, discussing integration with **Zapier** to automate tasks despite encountering reliability issues.
   - **GPT-4o's** faster response time stirs a debate over its trade-off with quality compared to **GPT-4**, while repeated verification demands frustrate users.
- **Content Creation and Audience Engagement**: Members discuss strategies for content creators to generate engaging content, intensifying interest in platform-specific advice, content calendar structures, and key metrics that determine success.
   - AI engineers emphasize the important role of prompts for engaging content creation and customer acquisition, spotlighting members' ideas for innovative usage of current trends.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Hidden Talents of Qwen Revealed**: Community members highlighted the **Qwen Team**'s contribution with praises, emphasizing that the team's efforts are underappreciated despite creating excellent resources such as a new [training video](https://link.to.video).
   - The discussions around **Qwen** suggest a growing respect for teams that deliver practical AI tools and resources.
- **GPU Showdown: AMD vs NVIDIA**: A technical debate unfolded about the efficiency of **AMD GPUs** compared to **NVIDIA** for LLM training, noting NVIDIA's dominance due to superior software ecosystem and energy efficiency.
   - Despite AMD's advancements, community consensus leaned towards NVIDIA as the pragmatic choice for LLM tasks because of library support, with a point raised that '*Most libraries don't support AMD so you will be quite limited in what you can use*.'
- **Phi-3 Training Troubles with Alpaca**: AI engineers exchanged solutions for an error encountered during Phi-3 training with Alpaca dataset, pinpointing the lack of CUDA support in the `xformers` version being used and suggesting an update.
   - Inference speeds were compared for **Llama-3** versus **Phi 3.5 mini**, noting Parallel debates that included suggestions for boosting efficiency, like referencing Tensorrt-llm for state-of-the-art GPU inference speed.
- **Kaggle Constraints Provoke Innovation**: Discussion in the community revolved around overcoming the **Kaggle** platform's disk space constraints, which led to a session crash after surpassing **100GB**, but not before leveraging [Weights & Biases](https://wandb.ai) to save critical data.
   - This incident highlights continuous innovation by AI engineers even when faced with limited resources, as well as the importance of reliable checkpoints in data-intensive tasks.
- **Empowering Job Seekers in AI Space**: Members of the AI community proposed the creation of a dedicated job channel to streamline job seeking and posting, which reflects the dynamic growth and need for career-focused services in the industry.
   - This initiative shows an active effort to organize and direct community efforts towards career development within the ever-growing AI field.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Encapsulating Complexity with LLM APIs**: Rearchitecting coding structures utilizing **LLM-style APIs** streamlines complex tasks; a user emphasized the coder's pivotal role in systems integration.
   - Creative combinations of APIs through zeroshot LLM prompts transform exhaustive tasks into ones requiring minimal effort, promising significant time economization.
- **Exploring Governmental AI Scrutiny**: The **UK Government's Inspects AI framework** targets large language models, provoking curiosity for its potential exploration and implications.
   - Available on [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai), it's position in the public sector spotlights a growing trend towards scrutinizing and regulating AI technologies.
- **Podcast Episode Storms Hacker News**: A user shared a **podcast episode on Hacker News** ([Now on HN!](https://news.ycombinator.com/newest)) aiming to attract attention and drive engagement.
   - Supportive community members boosted visibility with upvotes, reflecting an active and participative online discourse on Hacker News.
- **Fortnite Revamps Fun Factor**: Fortnite aims to charm players anew by nixing crossovers, sparked by a [Polygon exposé](https://www.polygon.com/gaming/24185789/fortnite-reload-new-game-mode) discussing the game's dynamic.
   - **Immediate reaction** surfaced through upvotes, with user endorsements like those from PaulHoule adding flames to the promotional fire.
- **Merging AI Minds**: AI Engineer World Fair’s buzz reached fever pitch as deep dives into **model merging strategies** captured enthusiasts, bolstered by tools like [mergekit on GitHub](https://github.com/arcee-ai/mergekit).
   - Hints at automated merging strategy determination sparked debate, though its intellectual robustness was tagged as **questionable**.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Credentials Clash**: Debate ignited on the value of CUDA certification versus publicly available GitHub CUDA work when hiring, with **community consensus** leaning towards the tangible evidence of public repositories.
   - `proven work that is public is always more valuable than a paper` was a key point raised, highlighting the **merit of demonstrable skills** over certificates.
- **Compiling The Path Forward**: **Compiler enthusiasts** are sought by [Lightning AI](https://boards.greenhouse.io/lightningai/jobs/6045025003), promising opportunities to work alongside **Luca Antiga**.
   - [Thunder project's](https://github.com/Lightning-AI/lightning-thunder) source-to-source compiler aims to boost **PyTorch models by up to 40%**, potentially transforming optimization benchmarks.
- **PyTorch Profilers Peek at Performance**: Elevation of **torch.compile manual** as a **missing link** for optimization, with a shared guide addressing its roles and benefits.
   - Another member suggested `torch.utils.flop_counter.FlopCounterMode` as a robust alternative to `with_flops`, citing its ongoing maintenance and development.
- **The Quantum of Sparsity**: CUDA exploration took a turn towards the **2:4 sparsity pattern** with discussions around the comparison of **cusparseLT** and **CUTLASS** libraries for optimized sparse matrix multiplication (SpMM).
   - The debate continued around potential performance differences, with the general opinion skewing towards cusparseLT for its **optimization** and maintenance.
- **LLM Lessons Laid Out**: Ideation for **LLM101n**, a proposed course to guide users from the basics of **micrograd** and **minBPE**, towards more complex areas like **FP8 precision** and **multimodal training**.
   - Discussion emphasized a layered learning approach, grounding in essentials before escalating to **state-of-the-art model practices**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Critique Companions Boost AI Reward Models**: Exploring the utility of synthetic critiques from **large language models**, Daniella_yz's preprint reveals potential for improving preference learning during a Cohere internship, as detailed in [the study](https://arxiv.org/abs/2405.20850).
   - The research suggests **CriticGPT** could move beyond aiding human assessments, by directly enhancing reward models in active projects.
- **Test-Time-Training Layers Break RNN Constraints**: Karan Dalal introduced **TTT layers**, a new architecture supplanting an RNN's hidden state with ML models shown in [their preprint](https://arxiv.org/abs/2407.04620).
   - Such innovation leads to **linear complexity architectures**, letting LLMs train on massive token collections, with TTT-Linear and TTT-MLP outperforming top-notch Transformers.
- **Data Dialogue with Dataline**: The launch of [Dataline by RamiAwar](https://github.com/RamiAwar/dataline) delivers a platform where users query multiple databases like CSV, MySQL, and more via an AI interface.
   - A fresh study titled **The Geometrical Understanding of LLMs** investigates LLM reasoning capacities and their self-attention graph densities; read more in [the paper](https://arxiv.org/abs/2407.02678).
- **GPT-4 Benchmark Fever**: A noteworthy observation among a user circle is GPT-4's improved performance on benchmarks at higher temperatures, though reproducibility with local models seems challenging.
   - Excitement stirs as in-context examples boost model performance, while BitNet architecture's efficiency propels a surge in interest despite memory-saving training complexities.
- **RAG and Reality: Hallucinations Under the Lens**: A [new YouTube video](https://youtu.be/no7EQkOiHQM?si=b35yua7rZuaEVvKu) casts a spotlight on LegalTech tools' reliability, unearthing the frequency of hallucinations via RAG models.
   - Furthermore, helpful Wikipedia-style `ref` tags are proposed for citation consistency, and [AymericRoucher's RAG tutorials](https://x.com/mervenoyann/status/1810291157961781671) receive acclaim for optimizing efficiency.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **WSL Leap** - Windows Whimsy with Mojo**: Upgrading WSL for **Mojo** installation led to hiccups on older Windows 10 setups; the [Microsoft guide for WSL](https://learn.microsoft.com/en-us/windows/wsl/install) proved invaluable for navigating the upgrade path.
   - **Python's dependency woes** sparked conversation, with virtual environments being the go-to fix; a [GitHub thread](https://github.com/modularml/mojo/discussions/1401) also opened on the potential for Mojo to streamline these issues.
- **Round Robin Rumpus** - Mojo Math Muddles**: Rounding function bugs in Mojo drew collective groans; inconsistencies with **SIMD** highlighted in a community deep dive into **rounding quiristics**.
   - Amidst the int-float discourse, the **64-bit conundrum** took center stage with Mojo's classification of `Int64` and `Float64` leading to unanticipated behavior across operations.
- **Stacks on Stacks** - Masterful Matmul Moves**: Members marveled at Max's use of ***stack allocation*** within matmul to boost Mojo performance, citing cache optimization as a key enhancement factor.
   - Autotuning surfaced as a sought-after solution to streamline *simdwidth* adjustments and block sizing, yet the reality of its implementation remains a reflective discussion.
- **Libc Love** - Linking Legacy to Mojo**: A communal consensus emerged on incorporating **libc functions** into Mojo; lightbug_http demonstrated the **liberal linking** in action on [GitHub](https://github.com/saviorand/lightbug_http/blob/main/external/libc.mojo).
   - Cross compiling capability queries capped off with the current lack in Mojo, prompting members to propose possible future inclusions.
- **Tuple Tango** - Unpacking Mojo's Potential**: Mojo's lack of *tuple unpacking* for aliasing sparked syntax-driven speculations, as community members clamored for a conceptually clearer construct.
   - **Nightly compiler updates** kept the Mojo crowd on their codes with version `2024.7.705` introducing new modules and changes.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AI-Plans Platform Uncloaks for Alignment Strategies**: Discussion unveiled around **AI-Plans**, a platform aimed at facilitating peer review for alignment strategies, mainly focusing on **red teaming alignment plans**.
   - Details were sparse as the user did not provide further insight or direct links to the project at this time.
- **Rhea's Radiant 'Save to Project' Feature Lights Up HTML Applications**: Rhea has integrated a new 'Save to Project' feature, enabling users to directly stash interactive HTML applications from their **dashboards** as seen on [Rhea's platform](https://rhea.run).
   - This addition fosters a smoother workflow, poised to spark augmented user engagement and content management.
- **Rhea Signups Hit a Snag Over Case Sensitivity**: A snag surfaced in Rhea's signup process, where user emails must be input in lowercase to pass email verification, hinting at a potential oversight in **user-experience** considerations.
   - The discovery accentuates the importance of rigorous testing and feedback mechanisms in user interface design, specifically for case sensitivity handling.
- **Whispers of Cohere Community Bonds and Ventures**: Fresh faces in the Cohere community shared their enthusiasm, with interests converging on synergistic use of tools like **Aya** for collaborative workflows and documentation.
   - The introductions served as a launchpad for sharing experiences, enhancing Cohere's tool utilization and community cohesion.
- **Youth Meets Tech: Rhea Embarks on Child-Friendly AI Coding Club Adventure**: Members of a children's coding club are seeking new horizons by integrating Rhea's user-friendly platform into their **AI and HTML projects**, aiming to inspire the next generation of AI enthusiasts.
   - This initiation represents a step towards nurturing young minds in the field of AI, highlighting the malleability of educational tools like Rhea for varying age groups and technical backgrounds.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **T-FREE Shrinks Tokenizer Footprint**: The introduction of [T-FREE](https://arxiv.org/abs/2406.19223) tokenizer revolutionizes embedding with an 85% reduction in layer size, achieving comparable results to traditional models.
   - This tokenizer forgoes pretokenization, translating words through character triplet activation patterns, an excellent step toward model compactness.
- **SOLAR Shines Light on Model Expansion**: Discussions on [SOLAR](https://arxiv.org/abs/2310.07999), a model expansion technique, heated up, with queries about efficiency versus training models from the ground up.
   - While SOLAR shows performance advantages, better comparisons with from-scratch training models are needed for definitive conclusions.
- **BitNet's Leap with 1-bit Weight Transformers**: [BitNet](https://arxiv.org/abs/2310.11453) debuts a 1-bit weight Transformer architecture, balancing performance against resource usage with a memory and energy-friendly footprint.
   - Weight compression without compromising much on results enables BitNet's Transformers to broaden utility in resource-constrained scenarios.
- **QuaRot Proves Potent at 4-bit Quantization**: [QuaRot's research](https://arxiv.org/abs/2404.00456) displayed that 4-bit quantization maintains near-full precision in LLMs while efficiently dialing down memory and processing requirements.
   - The significant trimming of computational costs without severe performance drops makes QuaRot a practical choice for inference runtime optimization.
- **Seeking the Right Docker Deployment for GPT-Neox**: Queries about the effective use of Docker containers for deploying GPT-Neox prompted speculation on Kubernetes being potentially more suited for large-scale job management.
   - While Docker Compose has been handy, the scale leans towards Kubernetes for lower complexity and higher efficiency in deployment landscapes.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **JPEG XL Takes the Crown**: **JPEG XL** is now considered the leading image codec, recognized for its efficiency over other formats in the field.
   - Discussions highlighted its robustness against traditional formats, considering it for future standard usage.
- **Kolors Repository Gains Attention**: The [Kolors GitHub repository](https://github.com/Kwai-Kolors/Kolors) triggered a surge of interest due to its significant paper section.
   - Members expressed both excitement and a dose of humor regarding its technical depth, predicting a strong impact on the field.
- **Noise Scheduling Sparks Debate**: The effectiveness of adding 100 timesteps and transitioning to **v-prediction** for noise scheduling was a hot debate topic, notably to achieve zero terminal SNR.
   - **SDXL's paper** was referenced as a guide amid concerns of test-train mismatches in high-resolution sampling scenarios.
- **Meta's VLM Ads Face Scrutiny**: Meta's decision to advertise VLM rather than releasing **Llama3VLM** stirred discontent, with users showing skepticism towards Meta's commitment to API availability.
   - The community expressed concern over Meta prioritizing its own products over widespread API access.
- **VALL-E 2's Text-to-Speech Breakthrough**: **VALL-E 2** set a new benchmark in text-to-speech systems, with its zero-shot TTS capabilities distinguishing itself in naturalness and robustness.
   - Though it requires notable compute resources, its results on LibriSpeech and VCTK datasets led to anticipation of replication efforts within the community.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Parsing CSV through LangChain**: Users explored approaches for handling CSV files in LangChain, discussing the need for modern methods beyond previous constraints.
   - **LangChain's utility functions** came to the rescue with recommendations for converting model outputs into JSON, using tools like `Json RedactionParser` for enhanced parsing.
- **Async Configurations Unraveled**: Async configuration in LangChain, specifically the `ensure_config()` method within `ToolNode` using `astream_events`, was demystified through communal collaboration.
   - Crucial guidance was shared to include `config` in the `invoke` function, streamlining **async task management**.
- **Local LLM Experimentation Scales Up**: Discussions heated up around running smaller LLM models like `phi3` on personal rigs equipped with NVIDIA RTX 4090 GPUs.
   - Curiosity spiked over managing colossal models, such as 70B parameters, and the viability of such feats on multi-GPU setups, indicating a drive for **local LLM innovation**.
- **LangGraph Cloud Service Stirs Speculation**: Hints of **LangGraph Cloud**'s arrival led to questions on whether third-party providers would be needed for LangServe API deployments.
   - The community buzzed with the anticipation of new service offerings and potential shifts in deployment paradigms.
- **In-browser Video Analysis Tool Intrigues**: **'doesVideoContain'**, a tool for in-browser content scanning within videos, sparked interest with its use of **WebAI** tech.
   - A push for community engagement saw direct links to a YouTube demo and Codepen live example, promoting its application.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **RAG's Skills Library Sharpens Actions**: Elevating efficiency, a member pioneered the integration of a skills library with **RAG**, enhancing the consistency of specified actions.
   - This advancement was shared with the community, incentivizing further exploration of **RAG's** potential in diverse AI applications.
- **Securing the Perimeter with OI Team Vigilance**: The OI team's commitment to security was spotlighted at a recent video meeting, cementing it as a forefront priority for operational integrity.
   - Their proactive measures are setting a benchmark for collective security protocols.
- **GraphRAG Weaves Through Data Clusters Effectively**: A participant showcased **Microsoft's GraphRAG**, a sophisticated tool that clusters data into communities to optimize **RAG** use-cases.
   - Enthusiasm for implementing GraphRAG was ignited, paralleled by a resourceful [tweet from @tedx_ai](https://x.com/tedx_ai/status/1808561861589139690).
- **Festive Fundamentals at 4th of July Shindig**: The OI team's **4th of July** celebration generated camaraderie, showcasing new demos and fostering anticipation for future team gatherings.
   - The team's spirit was buoyed, with hopes to establish this celebratory event as a recurring monthly highlight.
- **O1 Units Gear Up for November Rollout**: Timelines indicate the inaugural 1000 O1 units are slated for a November delivery, reflecting high hopes for their on-schedule arrival.
   - Curiosity surrounds O1's conversational abilities, while community support shines with shared solutions to tackle a Linux 'typer' module hiccup.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Crypto Payments with Multiple Currencies**: Community discussions focused on **Coinbase Commerce**'s ability to handle payments in various cryptocurrencies, including **USDC** and **Matic** through **Polygon**.
   - One user confirmed seamless transactions using **Matic**, endorsing its effectiveness.
- **Perplexity API Underwhelms**: Users noted that **Perplexity API's** performance pales in comparison to its web counterpart, missing vital reference links in the payload.
   - Suggestions to circumvent this include using alternatives like **Phind** or directly scraping from **GitHub** and **StackOverflow**.
- **Predicting the Generative Video Trajectory**: A member queried about the anticipated trajectory of **generative video** regarding quality, execution speed, and cost within the next **18 months**.
   - No firm forecasts were made, emphasizing the inchoate nature of such generative mediums.
- **OpenRouter's Options for Customized AI**: OpenRouter's feature allowing users to deploy their own **fine-tuned models** was confirmed for those able to handle a substantial request volume.
   - This has been recognized as a boon for developers desiring to impart bespoke AI functionalities.
- **DeepInfra vs. Novita: A Price War**: OpenRouter bore witness to a price competition between **DeepInfra** and **NovitaAI**, as they jostled for leadership in serving models such as **Llama3** and **Mistral**.
   - A humorous battle of undercutting prices by **0.001** has led to ultra-competitive pricing for those models.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Trading on Autopilot**: LlamaIndex Drives AI Stock Assistant**: An AI trading assistant exploiting **Llama Index agent**, demonstrated in a [tutorial video](https://t.co/dcLG3orq0s), performs varied tasks for stock trading.
   - Its capabilities, powered by [Llama Index's RAG abstractions](https://t.co/ocPaeLphyG), include predictive analyses and trades, with practical uses showcased.
- **Crafting RAG Datasets**: Tools for Richer Questions**: Giskard AI's toolkit aids in producing robust datasets for RAG, generating diverse question types showcased in their [toolkit article](https://t.co/sewtQcb9b8).
   - The toolkit surpasses typical auto-generated sets, providing a [richer toolkit for dataset creation](https://t.co/rQ7WxplJpF).
- **Microservices, Maxi Potential**: Agile Agents at Scale**: Llama-agents now offer a setup for scalable, high-demand microservices addressed in [this insightful post](https://t.co/y9a3PdfW0M).
   - This agent-and-tools-as-services pattern enhances scalability and simplifies microservice interactions.
- **Analyzing Analysts**: LlamaIndex Powers 10K Dissection**: The **Multi-Document Financial Analyst Agent**, treating each document as a tool, tackles the analysis of finance reports like 10Ks, thanks to [Llama Index's capabilities](https://t.co/rOetN1zeNg).
   - Pavan Mantha demonstrates the efficiency of this analysis using [Llama Index's features](https://t.co/LJhV838EUM).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Red Hesitation: Instinct for Caution?**: A member raised concerns regarding **team red's drivers** for Instinct cards, creating hesitation around purchasing used Mi100s due to potential support issues.
   - The conversation included a note that currently only 7900xtx cards are under test, implying solo troubleshooting for Instinct card users.
- **API Evolution: Crafting Custom Gradients**: A user proposed a new **API for custom grads**, wishing for a functionality akin to **jax.customvjp**, enhancing tensor operations for tasks like quantization training.
   - The suggested improvement targets the replacement of current operations with **lazybuffers** in tinygrad.functions, advocating for direct tensor manipulation.
- **Amplifying Learning: Multi-GPU Guidance**: Users seeking knowledge on multi-GPU training with Tinygrad were directed to the [beautiful_mnist_multigpu.py example](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_mnist_multigpu.py), highlighting model and data sharding techniques.
   - Details on copying the model with `shard(axis=None)` and data splitting with `shard(axis=0)` were shared, aiding in efficient parallel training.
- **Equality Engagement: Torch-Like Tensor Wars**: Queries on tensor comparison methods analogous to `torch.all` were resolved by introducing the comparison through `(t1 == t2).min() == 1`, later culminating in the addition of **Tensor.all** to Tinygrad.
   - This feature parity progression was documented in [this Tinygrad commit](https://github.com/tinygrad/tinygrad/commit/6856f915d6f0e10d41e8e11c8976024989d90aa7), facilitating easier tensor operations for users.
- **Optimization Obstacle: Adam’s Nullifying Effect**: Concerns were voiced over the **Adam optimizer** in Tinygrad causing weights to turn into **NaNs** after its second iteration step, presenting a stark contrast to the stability of SGD.
   - This debugging dialogue remains active as engineers seek a solution to prevent the optimizer from deteriorating the learning process.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **MInference's Agile Acceleration**: A member highlighted Microsoft's [MInference project](https://github.com/microsoft/MInference), which purports to accelerate Long-context LLMs' inference, **trimming latency** by up to 10x on an **A100**.
   - **MInference** employs novel techniques for approximate and dynamic sparse calculations, aiming to balance accuracy with **performance efficiency**.
- **Yi-1.5-9B Batches Up with Hermes 2.5**: Updates on **Yi-1.5-9B-Chat** revealed it was fine-tuned using **OpenHermes 2.5**, with publicly shared [models and quantizations](https://huggingface.co/juvi21/Hermes-2.5-Yi-1.5-9B-Chat) that excelled on the **AGIEval Benchmark**.
   - The enhanced model trained on **4x NVIDIA A100 GPUs for over 48 hours** impresses with its 'awareness', and plans are in motion to push its context length to **32k tokens using POSE**.
- **Chat Template Conundrums for Mistral**: A discussion arose on the best **chat_template** to use for **Mistral finetuning** in Axolotl, with the answer depending on dataset structure.
   - Community consensus pointed towards utilizing the **"chatml"** template, with YAML configuration examples offered to guide members.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **MLOps Maneuvers and FP8 Puzzles**: Community members shared insights, with one referencing a [blog post](https://nik-hil.hashnode.dev/diving-deep-essential-questions-for-building-your-mlops-pipeline) focusing on **MLOps implementation**, and another discussing troubles with **FP8 quantization** in **distributed vllm inference**.
   - Solutions for **FP8's sensitivity issues** were identified, resulting in corrected outputs and a [GitHub thread](https://github.com/vllm-project/vllm/issues/6179) provides more context for those tackling similar issues.
- **Dissecting Model Integrations**: A member is [evaluating](https://discord.com/channels/1238365980128706560/1241163904927666287/1259183509965115392) the integration of traditional tools like **Transformers** & **Torch** against established models from **OpenAI** and **Anthropic**.
   - The conversation centers around finding an optimal approach that offers both effectiveness and seamless integration for project-specific needs.
- **Crunch-Time for Credit Claims**: Discussions in the #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1259445018905542707) channel made it clear: **credit claims are closed permanently**, signaling an end to that benefit.
   - It was highlighted that this termination of credit accumulation applies universally, sparing no one and shutting down avenues for any future claims.
- **Replicate Credits Countdown**: A conversation in the #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/) channel revealed a one-month availability of **first 25 Replicate credits**, a critical update for users.
   - This limited-time offer seems to be a pivotal point in usage strategies, especially for those counting on these initial credits for their projects.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Interconnects Bot: Room for Enhancement**: A user expressed that the **Interconnects bot** is performing well, but has not seen significant changes in recent summarization outputs.
   - The user advocated for notable updates or enhancements to boost the Interconnects bot's functionality.
- **RAG Use Cases and Enterprise Discussions**: Members discussed Retrieval Augmented Generation (**RAG**) models, highlighting their developing use cases within enterprises.
   - Some participants suggested **RAG** might enhance the use of internal knowledge bases, while others reminisced about the model's hype during the *early AI boom*.
- **Rummaging Through Early Reflections on RAG**: Conversations touched on the ancestral excitement around **RAG**, with shared sentiments about the initial exaggerated expectations.
   - The exchanges revealed a shared perspective that the early hype has not fully translated into extensive enterprise adoption.
- **Cost Efficiency and Knowledge Retrieval: An Enterprise View**: The talk revolved around how **RAG** could aid in cost efficiency within enterprise models.
   - A stance was put forward that such models, by tapping into vast internal knowledge repositories, could cultivate new technological avenues for businesses.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Buzz Gains Admirers & Teases Release**: Enthusiasm for **Buzz** was palpable in the group, with a member praising its capabilities and hinting at more to come.
   - Autometa teased an upcoming release, sparking curiosity within the community.
- **FPGA Focus: Autometa's Upcoming Meeting**: Autometa announced plans to convene and discuss novel applications in the **FPGA** sphere, indicating several key topics for the agenda.
   - Members were invited to engage and share their insights on the versatile uses of **FPGAs** in current projects.
- **Opening Doors: Calendly Scheduling for Collaboration**: To facilitate discussions on AI alignment, Autometa shared an [open Calendly link](https://calendly.com/alignmentlab/meeting) for the community.
   - The link serves as an open invitation for scheduling in-depth discussions, offering a platform for collaborative efforts.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Flash 1.5 Gaining Traction**: Member **jeffreyw128** expressed that **Flash 1.5** is performing exceptionally well.
   - No additional context or detailed discussions were provided on the topic.
- **Awaiting Further Insights**: Details are currently sparse regarding the technical performance and features of **Flash 1.5**.
   - Community discussions and more in-depth analysis are expected to follow as the tool gains more attention.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Sprite Quest: Google Image Galore**: A member mentioned **sprites** were sourced from **random Google image searches**, adhering to the quick and varied needs of asset collection.
   - The focus was on acquiring diverse **sprites** without purchase, while **tilesets** were the sole **paid assets**.
- **Tileset Trade: The Only Expense**: Conversations revealed that the only assets that were financially invested in were **tilesets**, highlighting a cost-conscious approach.
   - This distinction underscores the methodical selection of assets, with **money spent solely on tilesets** and **sprites obtained freely** via search engines.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **EuroPython Vectorization Talk**: A user expressed their participation in **EuroPython**, hinting at a forthcoming talk focused on **vectorization**.
   - Interested guild members might attend to gain insights into the role of vectorization in Python, an important aspect for **AI engineering**.
- **Community Engagement at Conferences**: The mention of **EuroPython** by a user highlights the community's outreach and active presence at Python conferences.
   - This encourages networking and knowledge sharing among Python practitioners in the **AI and Machine Learning** fields.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Google's Gem Sparkles in Size and Performance**: Google's [Gemma 2 9B](https://blog.google/technology/developers/google-gemma-2/) has entered the arena as an open-source language model, noted for its robust performance.
   - **Despite its smaller scale**, Gemma 2 9B challenges heavyweights like GPT-3.5, suitable for use in environments with limited resources.
- **Lambda Lift-Off: Gemma 2 Reaches Serverless Heights**: The community explored **serverless AI inference** by integrating Google's Gemma 2 with Mozilla's Llamafile on AWS Lambda, as demonstrated in [this tutorial](https://www.unremarkable.ai/serverless-ai-inference-with-gemma-2-using-mozillas-llamafile-on-aws-lambda).
   - This serverless methodology enables deploying Gemma 2 9B efficiently in low-resource settings, including mobile devices, personal computers, or localized cloud services.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Models Fusion Forge**: A member proposed using **Hermes-2-Theta-Llama-3-70B** as a foundation for crafting the **Llama3-DiscoLeo-Instruct-70B** model.
   - The ensuing conversation hinted at the advantage of merging capabilities from both models to amplify performance.
- **Enhancement Speculations**: Engineers considered the speculated benefits of model integration focused on **Hermes-2-Theta-Llama-3-70B** and **Llama3-DiscoLeo-Instruct**.
   - The dialogue revolved around potential strides in AI capabilities through strategic fusion of distinct model features.



---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1258859978572173475)** (804 messages🔥🔥🔥): 

> - `Model Licensing`
> - `Performance and Troubleshooting`
> - `Generation Techniques and Tools`
> - `Community and Ethical Concerns`
> - `Image Upscaling Techniques` 


- **Stability AI Model Licensing Confusion**: The community is grappling with understanding the new Stability AI model licensing terms, especially for businesses that make over $1M in revenue.
   - Clarifications were provided, but concerns remain about using SD3 for commercial purposes and the impact on small businesses.
- **Performance Issues with Image Generation**: Users report significant slowdowns when using controlnet with text2img, often due to VRAM limitations causing memory shuffling with system RAM.
   - Adjusting Windows pagefile settings and using offloading strategies can mitigate some of the slowdowns.
- **Advanced Image Upscaling Strategies**: A detailed workflow involving multiple upscaling steps and software like Photoshop, SUPIR, and transformer upscalers was shared for achieving high-resolution images.
   - This method avoids common issues like tiling and aims to maintain a balance between detail addition and image consistency.
- **Community's Reaction to Model Quality and Releases**: The community expressed disappointment over the quality of the SD3 model, comparing it unfavorably to previous versions and voicing concerns about its rushed release.
   - There is anticipation for improved models like the 8B version, and ongoing discussions about the potential impacts of NSA involvement and other ethical concerns.
- **Technical Support and Solutions**: Discussions included solving problems with specific prompts, integrating external tools for better results, and handling hardware limitations.
   - Advice was given on using terms effectively in prompts and leveraging multiple software tools to achieve desired image generation results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.invoke.com/">Invoke | AI Image Generator for Your Business</a>: Invoke is the only generative creation tool and custom AI model manager where you retain complete control &amp; ownership of your work, your models, and your IP.</li><li><a href="https://riteshrm.github.io/posts/Meta%20Learning%20By%20Radek%20Osmulski%20Chapter%20Wise%20Summary%20Points/">Ritesh Kumar Maurya - Meta Learning Book Chapter Wise Summary Points</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Ly6USRwTHe0">Generative AI for Krita - With ControlNet</a>: Generate images from within Krita with minimal fuss using Stable Diffusion.https://github.com/Acly/krita-ai-diffusionNow with ControlNet scribble &amp; line art....</li><li><a href="https://www.interstice.cloud/plugin">Interstice</a>: no description found</li><li><a href="https://huggingface.co/tianweiy/DMD2/tree/main">tianweiy/DMD2 at main</a>: no description found</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#pixart-sigma>">SwarmUI/docs/Model Support.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI, A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/SwarmUI</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Installation-Guides#nvidia-automatic1111-webui-stable-diffusion-webui">Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info</li><li><a href="https://www.runcomfy.com/comfyui-web">ComfyUI Online - Free ComfyUI Web</a>: Use ComfyUI online for free without installation required, easily build a Stable Diffusion workflow, and generate images in seconds.</li><li><a href="https://comfyuiweb.com/">Comfyui Web - Using comfyui free and online</a>: no description found</li><li><a href="https://www.reddit.com/r/krita/comments/r5nq1y/krita_wont_allow_me_to_change_its_settings/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings">Command Line Arguments and Settings</a>: Stable Diffusion web UI. Contribute to AUTOMATIC1111/stable-diffusion-webui development by creating an account on GitHub.</li><li><a href="https://research.nvidia.com/labs/dir/jedi/"> Joint-image Diffusion</a>: no description found</li><li><a href="https://stability.ai/news/license-update">Community License &mdash; Stability AI</a>: Our new Community License is now free for research, non-commercial, and commercial use. You only need a paid Enterprise license if your yearly revenues exceed USD$1M and you use Stability AI models in...</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI, A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>: SwarmUI, A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility. - mcmonkeyprojects/SwarmUI</li><li><a href="https://civitai.com/models/74776/moebius-jean-giraud-style">Moebius (Jean Giraud) Style - SD XL | Stable Diffusion LoRA | Civitai</a>: Moebius, also known as Jean Giraud, was a French comic artist and illustrator known for his influential and visionary work in the field of science ...</li><li><a href="https://www.sca.org/">Home - SCA.org</a>: The SCA is an international organization devoted to research/re-creation of pre-17th-century skills/arts/combat/culture/history through events and activities.</li><li><a href="https://github.com/Acly/krita-ai-diffusion?tab=readme-ov-file">GitHub - Acly/krita-ai-diffusion: Streamlined interface for generating images with AI in Krita. Inpaint and outpaint with optional text prompt, no tweaking required.</a>: Streamlined interface for generating images with AI in Krita. Inpaint and outpaint with optional text prompt, no tweaking required. - Acly/krita-ai-diffusion</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1b50sp0/ccsr_vs_supir_upscale_comparison_portrait/```">CCSR vs SUPIR upscale comparison (portrait photography)</a>: I did some simple comparison 8x upscaling 256x384 to 2048x3072. I use SD mostly for upscaling real portrait photography so facial fidelity...</li><li><a href="https://safebooru.org/index.php?page=post&s=list&tags=looking_at_viewer">Safebooru  / looking_at_viewer</a>: no description found</li><li><a href="https://github.com/civitai/civitai/blob/feb2337c202ab82661958481de9652a4a6b3417b/src/utils/metadata/lists/words-young.json#L4">civitai/src/utils/metadata/lists/words-young.json at feb2337c202ab82661958481de9652a4a6b3417b · civitai/civitai</a>: A repository of models, textual inversions, and more - civitai/civitai</li><li><a href="https://github.com/civitai/civitai/blob/feb2337c202ab82661958481de9652a4a6b3417b/src/utils/metadata/lists/words-nsfw.json">civitai/src/utils/metadata/lists/words-nsfw.json at feb2337c202ab82661958481de9652a4a6b3417b · civitai/civitai</a>: A repository of models, textual inversions, and more - civitai/civitai</li><li><a href="https://civitai.com/models/447902/mangled-merge-xl?modelVersionId=619849">Mangled Merge XL - v3.0 | Stable Diffusion Checkpoint | Civitai</a>: V3: It is my pleasure to introduce version 3.0, the next iteration of the Mangled Merge XL series. I&#x27;ve spent some time looking into the DARE/TIES ...</li><li><a href="https://civitai.com/models/101055?modelVersionId=128080">SD XL - v1.0 Refiner VAE fix | Stable Diffusion Checkpoint | Civitai</a>: Originally Posted to Hugging Face and shared here with permission from Stability AI. SDXL consists of a two-step pipeline for latent diffusion: Fir...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1258861335119270028)** (605 messages🔥🔥🔥): 

> - `Hermes 2`
> - `GPTs Agents`
> - `OpenAI's sidebars`
> - `Fundraising for AI projects`
> - `Inference API issues` 


- **Inference API faces stalling issues**: Several members reported long initialization times for inference endpoints, with potential causes being GPU availability issues or specific configuration settings. One member suggested using [AWS Nvidia A10G on eu-west-1](https://aws.amazon.com/ec2/instance-types/a10g/) as an alternative.
- **GPTs Agents cannot learn after initial training**: A member shared a concern about GPTs agents not learning from additional information provided after their initial training.
- **Request for Custom LLM Metrics**: A user inquired about custom metrics for LLMs such as response completeness, text similarity, and hallucination index. They mentioned evaluating metrics like leivenstein distance, surprisal/perplexity, and specific task-related metrics like BLEU score for machine translation.
- **Antispam Measures Considering Regex Patterns**: Discussions around improving antispam measures included implementing regex patterns to automatically filter and ban certain words or phrases.
- **Community Feedback on Summarization Feature**: Community discussed the utility of Discord's built-in summarization feature, which uses OpenAI's GPT-3.5, expressing concerns about privacy and effectiveness.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/learn/intro-to-machine-learning">Learn Intro to Machine Learning Tutorials | Kaggle</a>: no description found</li><li><a href="https://www.youtube.com/@matthew_berman">Matthew Berman</a>: Artificial Intelligence (AI), Open Source, Generative Art, AI Art, Futurism, ChatGPT, Large Language Models (LLM), Machine Learning, Technology, Coding, Tutorials, AI News, and more  ** Exclusive Pine...</li><li><a href="https://docs.continue.dev/how-to-use-continue#ask-questions-about-your-codebase">🧑‍🎓 How to use Continue | Continue</a>: Using LLMs as you code with Continue</li><li><a href="https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm">Metrics | DeepEval - The Open-Source LLM Evaluation Framework</a>: Quick Summary</li><li><a href="https://www.gradio.app/guides/getting-started-with-the-python-client">Getting Started With The Python Client</a>: A Step-by-Step Gradio Tutorial</li><li><a href="https://huggingface.co/spaces/InstantX/InstantStyle">InstantStyle - a Hugging Face Space by InstantX</a>: no description found</li><li><a href="https://huggingface.co/spaces/nroggendorff/llava">Llama - a Hugging Face Space by nroggendorff</a>: no description found</li><li><a href="https://riteshrm.github.io/posts/Meta%20Learning%20By%20Radek%20Osmulski%20Chapter%20Wise%20Summary%20Points/">Ritesh Kumar Maurya - Meta Learning Book Chapter Wise Summary Points</a>: no description found</li><li><a href="https://huggingface.co/spaces/qnguyen3/nanoLLaVA">nanoLLaVA-1.5 - a Hugging Face Space by qnguyen3</a>: no description found</li><li><a href="https://huggingface.co/spaces/LanguageBind/Video-LLaVA">Video LLaVA - a Hugging Face Space by LanguageBind</a>: no description found</li><li><a href="https://scale.com/leaderboard">SEAL leaderboards</a>: no description found</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers">stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/discord-community/HuggingMod/discussions/1">discord-community/HuggingMod · pls merge</a>: no description found</li><li><a href="https://huggingface.co/settings/billing">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://lu.ma/yzzespyu">4 weeks AI Study Group @ Block: Andrej Karpathy&#x27;s Zero to GPT Hero · Luma</a>: NOTE: This is a repeating event for 4 weeks in a row, starting July the 24th, ending August the 14th! ~ The GPT phenomenon is largely responsible for putting…</li><li><a href="https://www.youtube.com/watch?v=WhAMvOEOWJw">One Minute Gradio #1: Dynamic Rendering</a>: One Minute Gradio #1 - Learn Gradio tips and tricks quickly! Today, we&#39;ll discuss dynamic rendering (i.e. the @gr.render decorator) in Gradio and how it lets...</li><li><a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">Tweet from Prashant Dixit (@Prashant_Dixit0)</a>: ✨Open-sourcing comprehensive LLM Glossary✨  Explore, Learn, and Add terms about #LLMs and #GenAI. Let&#39;s make AI easy for everyone.  🚨Adding new terms on regular basis  Don&#39;t forget to give st...</li><li><a href="https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3">RAG chatbot using llama3</a>: no description found</li><li><a href="https://tenor.com/view/huh-cat-cat-huh-small-cat-huh-what-gif-2593177363967991691">Huh Cat GIF - Huh Cat Cat huh - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/rip-buff-spongebob-ripping-shirt-gif-14008353">Rip Buff GIF - Rip Buff Spongebob - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/iceage-possum-peaceout-gif-2272177960667492692">Iceage Possum GIF - IceAge Possum PeaceOut - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aclanthology.org/2023.eamt-1.19/">Large Language Models Are State-of-the-Art Evaluators of Translation Quality</a>: Tom Kocmi, Christian Federmann. Proceedings of the 24th Annual Conference of the European Association for Machine Translation. 2023.</li><li><a href="https://github.com/nroggendorff/diffusion/blob/main/zelda.ipynb">diffusion/zelda.ipynb at main · nroggendorff/diffusion</a>: Contribute to nroggendorff/diffusion development by creating an account on GitHub.</li><li><a href="https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/toxicity/template.py">deepeval/deepeval/metrics/toxicity/template.py at main · confident-ai/deepeval</a>: The LLM Evaluation Framework. Contribute to confident-ai/deepeval development by creating an account on GitHub.</li><li><a href="https://github.com/aymeric-roucher/agent_reasoning_benchmark/">GitHub - aymeric-roucher/agent_reasoning_benchmark: 🔧 Compare how Agent systems perform on several benchmarks. 📊🚀</a>: 🔧 Compare how Agent systems perform on several benchmarks. 📊🚀 - aymeric-roucher/agent_reasoning_benchmark</li><li><a href="https://github.com/hpcaitech/Open-Sora">GitHub - hpcaitech/Open-Sora: Open-Sora: Democratizing Efficient Video Production for All</a>: Open-Sora: Democratizing Efficient Video Production for All - hpcaitech/Open-Sora</li><li><a href="https://github.com/huggingface/lighteval">GitHub - huggingface/lighteval: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron.</a>: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron. - hug...</li><li><a href="https://tenor.com/view/red-kit-gif-11737462">Red Kit GIF - Red Kit - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceApi">Inference</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1259311320008753214)** (4 messages): 

> - `Boid AI`
> - `LLM/GenAI Glossary`
> - `GPA Predictor with Scikit-Learn`
> - `Generative Text Project` 


- **Introducing Boid AI Concept**: A member introduced the concept of **Boid AI**, where 'boid' stands for 'bird-oid', implying bird-like AI behavior.
- **Comprehensive LLM/GenAI Glossary Open-Sourced**: A member shared a [comprehensive LLM glossary](https://github.com/freetoolsarebest/llm-glossary) via GitHub, aimed at making AI terms more accessible.
   - *Explore, Learn, and Add terms about LLMs and GenAI.*
- **Building a GPA Predictor with Scikit-Learn**: A member shared about creating a rough **GPA predictor using Scikit-Learn** on Kaggle and reading 'Hands-On Machine Learning' by Geron Aurelion.
   - They also watched some of [3Blue1Brown’s series on neural networks](https://www.youtube.com/user/3blue1brown) for further learning.
- **Advice on Generative Text Project**: A member asked for advice on starting a generative text project, debating between using existing models or building one from scratch.
   - They mentioned a recommendation to use Hugging Face along with Langchain, seeking reasons for why Langchain should be used.



**Link mentioned**: <a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">Tweet from Prashant Dixit (@Prashant_Dixit0)</a>: ✨Open-sourcing comprehensive LLM Glossary✨  Explore, Learn, and Add terms about #LLMs and #GenAI. Let&#39;s make AI easy for everyone.  🚨Adding new terms on regular basis  Don&#39;t forget to give st...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1258883504171909271)** (16 messages🔥): 

> - `Claude Artifacts`
> - `PersonaHub Dataset`
> - `Pseudonymization Techniques`
> - `Admin Requests` 


- **Claude focuses on artifacts for impressive results**: A user speculated that Claude's impressive performance may be due to its focus on 'artifacts'.
- **Exploring the PersonaHub Dataset**: A user shared the [PersonaHub dataset](https://huggingface.co/datasets/proj-persona/PersonaHub) designed for understanding performing arts centers and urban planning.
   - The dataset includes scenarios like scheduling multi-show festivals and contrasting public services in different neighborhoods.
- **Pseudonymization Techniques Impact Model Quality**: A paper from [TrustNLP 2023](https://aclanthology.org/2023.trustnlp-1.20/) analyzed pseudonymization techniques for text classification and summarization.
   - *Replacing named entities with pseudonyms* preserved performance on some NLP tasks.
- **Frequent admin pings and spam issues**: Members frequently pinged admins and requested bans for repeated spam, specifically mentioning 'opensea'.
   - "Please ban the word opensea" and discussions on hacked users and potential bots occurred.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aclanthology.org/2023.trustnlp-1.20/">Privacy- and Utility-Preserving NLP with Anonymized data: A case study of Pseudonymization</a>: Oleksandr Yermilov, Vipul Raheja, Artem Chernodub. Proceedings of the 3rd Workshop on Trustworthy Natural Language Processing (TrustNLP 2023). 2023.</li><li><a href="https://huggingface.co/datasets/proj-persona/PersonaHub">proj-persona/PersonaHub · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1259013525809008680)** (24 messages🔥): 

> - `in10search Tabs Sidepanel AI`
> - `ZeroGPU HuggingFace Space`
> - `qdurllm`
> - `AI on-call developer: merlinn`
> - `DarkWebSight` 


- **Browse with in10search Tabs Sidepanel AI**: A new browser sidepanel extension called **in10search Tabs Sidepanel AI** integrates horizontal tabs and ChatGPT. More details can be found on [GitHub](https://github.com/vtempest/in10search-chrome).
- **ZeroGPU HuggingFace Space for Stable Diffusion Models**: A member introduced a **HuggingFace Space** that allows users to compare multiple **Stable Diffusion Models** like **SD3 Medium**, **SD2.1**, **SDXL**, and more. Check it out [here](https://huggingface.co/spaces/Nick088/stable-diffusion-arena).
- **qdurllm: Local Search Engine with Qdrant & LLMs**: The newly launched open-source product **qdurllm** combines **Qdrant**, **URL scraping**, and **Large Language Models** for local search and chat. Explore further on its [GitHub repository](https://github.com/AstraBert/qdurllm).
- **AI on-call developer: merlinn**: An AI on-call developer named **merlinn** helps investigate production incidents by providing contextual information. Check it out and provide feedback on [GitHub](https://github.com/merlinn-co/merlinn).
- **gary4live Ableton plug-in**: A fun plug-in called **gary4live** for Ableton was released on Gumroad. It's a max4live device that integrates playful workflows with AI, available for free [here](https://thecollabagepatch.com).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Nick088/stable-diffusion-arena">Stable Diffusion Arena - a Hugging Face Space by Nick088</a>: no description found</li><li><a href="https://huggingface.co/spaces/as-cle-bert/self-reviewing-coding-assistant">Self Reviewing Coding Assistant - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://huggingface.co/datasets/Csplk/DarkWebSight">Csplk/DarkWebSight · Datasets at Hugging Face</a>: no description found</li><li><a href="https://youtu.be/D4E9zAmrCQ8?si=YDTa8ZIOTNCSxG9H">ghost chords - the captain&#39;s chair, season two - episode 1</a>: 00:00 - intro01:28 - ghost chords explained02:25 - the riff03:40 - the robot joins in08:55 - the trackseason one on spotify:https://open.spotify.com/album/7h...</li><li><a href="https://wandb.ai/sauravmaheshkar/llamaindex-local-models-index/reports/Training-a-chatbot-on-personal-data-with-LlamaIndex-and-W-B--Vmlldzo4MzQzMDE3">Training a chatbot on personal data with LlamaIndex and W&B</a>: In this article, we&#39;ll go over about how we can create a chatbot on personal data using Llamaindex and local models with a Weights &amp; Biases integration.</li><li><a href="https://tenor.com/view/lost-lost-tv-show-desmond-desmond-hume-lost-desmond-gif-17240446">Lost Lost Tv Show GIF - Lost Lost Tv Show Desmond - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/NW42xY651cQ">Design ChatGPT like AI Assiatant | ML System Design | #machinelearning</a>: We explore the ML system design question to create a ChatGPT-like AI assistant. The purpose of the AI assistant illustrated in the video is to automatically ...</li><li><a href="https://github.com/AstraBert/qdurllm">GitHub - AstraBert/qdurllm: Search your favorite websites and chat with them, on your desktop🌐</a>: Search your favorite websites and chat with them, on your desktop🌐 - AstraBert/qdurllm</li><li><a href="https://github.com/vtempest/in10search-chrome">GitHub - vtempest/in10search-chrome: in10search Tabs Sidepanel AI   - Horizontal Tabs in Browser Sidepanel with ChatGPT</a>: in10search Tabs Sidepanel AI   - Horizontal Tabs in Browser Sidepanel with ChatGPT - vtempest/in10search-chrome</li><li><a href="https://github.com/merlinn-co/merlinn">GitHub - merlinn-co/merlinn: Open source AI on-call developer 🧙‍♂️ Get relevant context &amp; root cause analysis in seconds about production incidents and make on-call engineers 10x better 🏎️</a>: Open source AI on-call developer 🧙‍♂️ Get relevant context &amp; root cause analysis in seconds about production incidents and make on-call engineers 10x better 🏎️ - merlinn-co/merlinn</li><li><a href="https://github.com/U-C4N/H.I.BOT/">GitHub - U-C4N/H.I.BOT</a>: Contribute to U-C4N/H.I.BOT development by creating an account on GitHub.</li><li><a href="https://thecollabagepatch.com">no title found</a>: no description found</li><li><a href="https://x.com/thepatch_kev/status/1810063563823907172">Tweet from thecollabagepatch (@thepatch_kev)</a>: 13 legends just got an email for  gary4live  the ableton plugin that does this  dl on gumroad rn u guys  ⬇️link  @_buildspace @_nightsweekends</li><li><a href="https://portfolio-app-raj.streamlit.app/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1259056094878105600)** (22 messages🔥): 

> - `Torchmetrics for Object Detection`
> - `RT-DETR Model Release`
> - `CogVLM2 for Vision-Language Models`
> - `Zero-shot Object Detection Models`
> - `MaskFormer and Instance Segmentation` 


- **Torchmetrics recommended for Object Detection**: Torchmetrics is suggested for object detection metrics and utilized in [official example scripts](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection) with the Trainer API and Accelerate.
- **RT-DETR Model Release**: [RT-DETR](https://x.com/mervenoyann/status/1807790959884665029) is a YOLO-like model for real-time object detection combining convolutions and attention-based transformers.
   - It comes with an Apache 2.0 license, offering the best of both worlds.
- **CogVLM2 for Vision-Language Models**: The [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) is recommended for various tasks with large-scale vision language models, including impressive performance on benchmarks like TextVQA and DocVQA.
- **Zero-shot Object Detection Models**: The Transformers library supports zero-shot object detection models such as OWL-ViT, OWLv2, and Grounding DINO for textual description-based object detection.
   - These models can also perform image-guided object detection as demonstrated in this [demo](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OWLv2/Zero_and_one_shot_object_detection_with_OWLv2.ipynb).
- **MaskFormer and Instance Segmentation**: MaskFormer models trained on datasets like ADE20k for semantic segmentation can be extended for use in instance segmentation with official scripts newly added [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation).
   - It is suggested to start from pre-trained COCO models for fine-tuning on instance segmentation tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://segments.ai/,">Segments</a>: no description found</li><li><a href="https://huggingface.co/blog/finetune-florence2">Fine-tuning Florence-2 - Microsoft&#39;s Cutting-edge Vision Language Models</a>: no description found</li><li><a href="https://huggingface.co/facebook/maskformer-swin-large-ade">facebook/maskformer-swin-large-ade · Hugging Face</a>: no description found</li><li><a href="https://theadamcolton.github.io/image-ssl-on-a-shoestring">no title found</a>: no description found</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B">THUDM/cogvlm2-llama3-chat-19B · Hugging Face</a>: no description found</li><li><a href="https://x.com/mervenoyann/status/1807790959884665029">Tweet from merve (@mervenoyann)</a>: Real-time DEtection Transformer (RT-DETR) landed in @huggingface transformers 🤩 with Apache 2.0 license 😍  do DETRs Beat YOLOs on Real-time Object Detection?  keep reading 👀</li><li><a href="https://huggingface.co/spaces/andito/Florence-2-DocVQA/blob/main/app.py#L25">app.py · andito/Florence-2-DocVQA at main</a>: no description found</li><li><a href="https://x.com/skalskip92/status/1808874766515818840">Tweet from SkalskiP (@skalskip92)</a>: no more new VLMs?   I&#39;m finally working on a YouTube tutorial for my football AI project; the tutorial should be out next week.  stay tuned: https://www.youtube.com/roboflow</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OWLv2/Zero_and_one_shot_object_detection_with_OWLv2.ipynb">Transformers-Tutorials/OWLv2/Zero_and_one_shot_object_detection_with_OWLv2.ipynb at master · NielsRogge/Transformers-Tutorials</a>: This repository contains demos I made with the Transformers library by HuggingFace. - NielsRogge/Transformers-Tutorials</li><li><a href="https://huggingface.co/facebook/mask2former-swin-small-coco-instance">facebook/mask2former-swin-small-coco-instance · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection">transformers/examples/pytorch/object-detection at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1259012438276177941)** (7 messages): 

> - `Label Error in NLP Dataset`
> - `Extending deepseek-ai model context length`
> - `Byte Pair Encoding Implementation in C`
> - `Comprehensive LLM/GenAI Glossary` 


- **Label Error Frustrates User**: A user reported an error `ValueError: Invalid string class label ['B-COMPANY']` while working with an NLP dataset imported from a .txt file.
   - The issue causes frequent changes in error messages, complicating the troubleshooting process.
- **deepseek-ai Model Context Length Inquiry**: A user asked if it's possible to extend the context length of the `deepseek-ai/deepseek-math-7b-rl` model from 4k to 8k without tuning.
   - They explored options like vLLM or loading directly via HF to achieve this extension.
- **Byte Pair Encoding in C Released**: Ashpun announced the implementation of a minimal [Byte Pair Encoding mechanism](https://github.com/ash-01xor/bpe.c) in C.
   - A blog post is coming soon, and the code is now available on GitHub.
- **LLM/GenAI Glossary Open-Sourced**: Prashant Dixit promoted a [comprehensive LLM Glossary](https://x.com/Prashant_Dixit0/status/1809900514097979768) aimed at making AI easier for everyone.
   - The terms are regularly updated and the project is open-source, available [on GitHub](https://github.com/freetoolsarebest/llm-glossary).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">Tweet from Prashant Dixit (@Prashant_Dixit0)</a>: ✨Open-sourcing comprehensive LLM Glossary✨  Explore, Learn, and Add terms about #LLMs and #GenAI. Let&#39;s make AI easy for everyone.  🚨Adding new terms on regular basis  Don&#39;t forget to give st...</li><li><a href="https://github.com/ash-01xor/bpe.c">GitHub - ash-01xor/bpe.c: Simple Byte pair Encoding mechanism used for tokenization process . written purely in C</a>: Simple Byte pair Encoding mechanism used for tokenization process . written purely in C - ash-01xor/bpe.c
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1259958137360879778)** (1 messages): 

> - `Artifacting in sd-vae`
> - `Common issues in sd-vae reconstruction` 


- **Artifacting in sd-vae raises questions**: A member questioned if blue and white pixel artifacting is normal when using **sd-vae for reconstruction**.
   - This sparked a discussion about common issues and troubleshooting methods for pixel artifacting in **sd-vae**.
- **Identifying Common Issues in sd-vae**: Members delved into common issues encountered with sd-vae, focusing on pixel artifacting and reconstruction quality.
   - Suggestions for troubleshooting included experimenting with different parameter settings and sharing results for community feedback.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1259850068085964870)** (1 messages): 

> - `Enhanced Documentation Search on Gradio`
> - `Navigation of Gradio Documentation Pages` 


- **Gradio Enhances Documentation Search**: The Gradio community announced the release of a new [enhanced Search functionality](https://www.gradio.app/) within their documentation pages, making it easier to navigate and access information.
   - They invite users to try it out by visiting the documentation and emphasize their commitment to improving user experience.
- **Quickstart and Tutorials Now Easier to Access**: The improved search tool helps users find quickstart guides and in-depth tutorials more efficiently.
   - Gradio encourages users to keep sending feedback to enhance their experience further.



**Link mentioned**: <a href="https://www.gradio.app/">Gradio</a>: Build &amp; Share Delightful Machine Learning Apps

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1258875621149380689)** (502 messages🔥🔥🔥): 

> - `Issues with Perplexity`
> - `Pro Search and Limitations`
> - `Subscription Alternatives`
> - `Image Generation`
> - `Technical Problems and Bugs` 


- **Users face issues with Perplexity's performance**: Several users mentioned that Perplexity often fails to provide accurate or recent articles, returning outdated information despite precise prompts.
   - One user expressed frustration with context loss in follow-up questions, suggesting that GPT-4o maintains context better than Claude 3.5.
- **Pro Search disappoints some users in value**: A few users felt the Pro subscription is a waste of money, seeing no significant improvement in results compared to the free version.
   - Despite this, Perplexity Pro offers more advanced search capabilities and frequent updates, though some users believe alternative services provide better value for similar or lower costs.
- **Exploring alternative AI services**: Users discussed various alternatives like Merlin.ai, ChatLLM in Abacus.AI, and You.com, sharing mixed reviews on their performance and usability.
   - Monica.ai and OpenRouter with LibreChat were highlighted for their extensive features and user-friendly interfaces, making them strong competitors.
- **Image generation capabilities of Perplexity**: Some users were unaware that Perplexity can generate images, needing clarification on accessing this feature.
   - Perplexity Pro users have image generation access, and leveraging the custom prompt option in image generation can yield better results.
- **Bugs and technical issues**: Several users reported bugs in Perplexity, such as text overlap, context loss, and issues with generating scripts.
   - The community suggested workarounds like using system prompts and emphasized the need for more intuitive and straightforward features to improve user experience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://apps.apple.com/us/app/deepl-translate-write/id1552407475">‎DeepL: translate &amp; write</a>: ‎DeepL is your go-to AI translation and writing assistant for precise translations, powerful grammar fixes, and clear style enhancements. With the power of advanced Language AI, DeepL allows you to tr...</li><li><a href="https://msty.app">Msty - Using AI Models made Simple and Easy</a>: Chat with files, understand images, and access various AI models offline. Use models from Open AI, Claude, Perplexity, Ollama, and HuggingFace in a unified interface.</li><li><a href="https://console.groq.com/playground">GroqCloud</a>: Experience the fastest inference in the world</li><li><a href="https://x.com/baronitaigas/status/1809155575340544500?s=19">Tweet from Baron of the Taiga (@baronitaigas)</a>: ⚡️🇱🇻: The Latvian army will begin spelling Russia with a lower case &#39;r&#39; in official documents  - Sandra Brale, Public Affairs Officer for the Chief of Defense of Latvia.</li><li><a href="https://tenor.com/view/laughing-spongebob-patrick-blush-gif-4679526">Laughing Spongebob GIF - Laughing Spongebob Patrick - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://gitlab.com/monnef/ailin">monnef / AIlin · GitLab</a>: AIlin is a tool that connects AI services, such as Perplexity.ai, with your local computer.</li><li><a href="https://www.threads.net/@perplexity.ai/post/C7mU3LdC6Cj?xmt=AQGzr8iLqFKizU24JG74yUmtoD5g8xMIjIC5fZLt_7B_Iw">Perplexity AI (&#064;perplexity.ai) on Threads</a>: Well..speaking of upgrades! We&#039;re excited to roll out Perplexity Pages, a simple way to turn your research into visually appealing articles. With formatted images and sections, Pages lets you sha...</li><li><a href="https://chatllm.abacus.ai/">Abacus.AI - </a>: Abacus.AI is the world’s first AI platform where AI, not humans, build Applied AI agents and systems at scale. Using generative AI and other novel neural net techniques, AI can build LLM apps, gen AI ...</li><li><a href="https://chromewebstore.google.com/detail/deepl-translate/cofdbpoegempjloogbagkncekinflcnj?hl=fr).">DeepL Translate</a>: Translate while you read and write with DeepL Translate, the world’s most accurate translator.</li><li><a href="https://www.mlb.com/stats/?playerPool=ALL_CURRENT)">2024 MLB Player Hitting Stat Leaders</a>: The official source for player hitting stats, MLB home run leaders, batting average, OPS and stat leaders
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1258905764936814613)** (15 messages🔥): 

> - `Minecraft Underground Survival`
> - `Average Cost Research`
> - `Relational Table Considerations`
> - `Current Redemption Programs`
> - `Next iPad Mini Release` 


- **Minecraft Underground Survival Guide**: Several users discussed a detailed [guide to Minecraft Underground Survival](https://www.perplexity.ai/page/minecraft-underground-survival-hj7PsuozQ32xoJKudQqm8g), exploring strategies for thriving in the game's underground environment.
- **Average Cost Research Findings**: One member shared an incremental insight from their [research on average costs](https://www.perplexity.ai/search/what-is-the-average-cost-of-a-JM0.Us5FQO6c6cozzHTvOw) and mentioned it was 'jaw-dropping'.
- **Setting Up New Google Account Issues**: A user sought help about [setting up a new Google account](https://www.perplexity.ai/search/i-m-trying-to-set-up-a-new-goo-iFZlMi1vQ0qcmbkCH.heEQ), indicating they had difficulties during the process.
- **Exploring Neuromorphic Chips**: Members delved into the technicalities of [how neuromorphic chips work](https://www.perplexity.ai/page/how-neuromorphic-chips-work-jb7QR.G6TzGswMico3It5g), which emulate the human brain's architecture for efficient processing.
- **Craft CMS Upgrade Guidance**: One discussion focused on [upgrading Craft CMS from version 3.9.5 to 5](https://www.perplexity.ai/search/upgrade-craft-cms-3-9-5-to-5-w-D_kJzsmYTfOe3ISXp1GtFw), covering necessary steps and potential challenges.



**Link mentioned**: <a href="https://www.youtube.com/embed/CcOK72Jmlno">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1259136267703029770)** (9 messages🔥): 

> - `Online model performance`
> - `API request processing`
> - `API vs Perplexity search results`
> - `Beta access delay`
> - `Multi-step search in API` 


- **New online model shows improved performance**: **Online model** is reportedly performing better, particularly in handling multi-part queries, as shared by a user.
   - *Feels more robust and precise* in generating responses compared to previous versions.
- **Issues around API request processing**: Users are questioning the **processing time for API access requests**, and are curious about ways to expedite the process.
   - No clear answers were provided regarding **usual processing times** or **expedited requests**.
- **Disparity between API results and Perplexity search**: Concern raised about **API results** not matching the **Perplexity.ai search page** results.
   - A member clarified that API results are the same as the **non-pro search results**.
- **Long wait for Beta access**: A user expressed dissatisfaction with waiting nearly a **month for Beta access** with no response yet.
   - *No updates or timeframe* provided for resolving the delay in Beta access.
- **Multi-step search in Perplexity API**: A user inquired about the availability of the **multi-step search feature** in the Perplexity API.
   - No concrete information was available; the member was directed to a **Discord channel** link for potentially more details.


  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1258867467028271195)** (249 messages🔥🔥): 

> - `Hermes 2.5`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic`
> - `IPEX-LLM integration` 


- **IPEX-LLM integration works despite hassles**: After following the [IPEX-LLM quickstart guide](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_quickstart.md), users report varied success in integrating IPEX-LLM with llama.cpp.
   - Some members faced difficulties due to outdated guides, while others reported successful builds by following official instructions.
- **MacBook M3 handles large models**: Users discuss the performance of M2 and M3 MacBooks, particularly praising the M3 MacBook Pro with **128GB RAM** for handling large models like WizardLM-2-8x22B.
   - Despite some issues with memory limits on older models, the M3 is seen as a robust solution for large model inference.
- **WizardLM-2-8x22B performance tested**: A member sought help to test the performance of **WizardLM-2-8x22B-Q4_K_M** on an M2 MacBook with **32k context** due to previous claims of poor performance.
   - Due to memory constraints, the model failed to load, with a M3 MacBook scheduled for a retry.
- **InternLM models and vision capabilities**: Members inquired about using **InternLM models** for vision tasks, noting issues with compatibility in LM Studio.
   - While some models worked well in Python, users reported needing specific configurations and adapters for vision in LM Studio.
- **GLM4 model support in llama.cpp**: A user asked if LM Studio would support **GLM4 models** since **llama.cpp** recently added support for them, hoping to run CodeGeex models efficiently.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://host.docker.internal:PORT_NUMBER.">no title found</a>: no description found</li><li><a href="http://host.docker.internal:11434.">no title found</a>: no description found</li><li><a href="https://github.com">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://huggingface.co/internlm/internlm-xcomposer2d5-7b">internlm/internlm-xcomposer2d5-7b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/internlm/internlm2_5-7b-chat-gguf">internlm/internlm2_5-7b-chat-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mradermacher/internlm2_5-7b-GGUF">mradermacher/internlm2_5-7b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/THUDM/codegeex4-all-9b">THUDM/codegeex4-all-9b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/inter">inter (Xhark Zhang)</a>: no description found</li><li><a href="https://huggingface.co/QuantFactory/internlm2_5-7b-chat-1m-GGUF">QuantFactory/internlm2_5-7b-chat-1m-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/releases/tag/b3333">Release b3333 · ggerganov/llama.cpp</a>: no description found</li><li><a href="https://huggingface.co/bartowski/WizardLM-2-8x22B-GGUF/tree/main/WizardLM-2-8x22B-Q4_K_M.gguf">bartowski/WizardLM-2-8x22B-GGUF at main</a>: no description found</li><li><a href="https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_quickstart.md">ipex-llm/docs/mddocs/Quickstart/llama_cpp_quickstart.md at main · intel-analytics/ipex-llm</a>: Accelerate local LLM inference and finetuning (LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, Phi, etc.) on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and...</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio CLI. Written in TypeScript/Node</a>: LM Studio CLI. Written in TypeScript/Node. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=Y08Nn23o_mY">Intro to RAG for AI (Retrieval Augmented Generation)</a>: This is an intro video to retrieval-augmented generation (RAG). RAG is great for giving AI long-term memory and external knowledge, reducing costs, and much ...</li><li><a href="https://www.youtube.com/watch?v=pK8u4QfdLx0">&quot;okay, but I want Llama 3 for my specific use case&quot; - Here&#39;s how</a>: If you want a personalized AI strategy to future-proof yourself and your business, join my community: https://www.skool.com/new-societyFollow me on Twitter -...</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g">[1hr Talk] Intro to Large Language Models</a>: This is a 1 hour general-audience introduction to Large Language Models: the core technical component behind systems like ChatGPT, Claude, and Bard. What the...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1258862244566011955)** (163 messages🔥🔥): 

> - `Experiences with Different Model Versions`
> - `Model Performance Issues`
> - `Model Quantization Discussions`
> - `Fine-tuning and Customization`
> - `Categorizing Text Prompts` 


- **Diverse Model Experiences and Issues**: Users discussed their experiences with various models such as Hermes, Mistral, and Gemma, noting issues like performance discrepancies and infinite loops.
   - Some mentioned specific hardware setups and configurations to diagnose or improve performance, highlighting different quantization settings and their impacts.
- **Gemma 2 Models Face Performance Bugs**: Multiple users experienced performance issues with **Gemma 2** models, including slow inference and incorrect math calculations.
   - Community expects improvements in upcoming updates to resolve these bugs, with specific discussions around [Gemma model architectural issues](https://github.com/ggerganov/llama.cpp/pull/8348).
- **Quantization Techniques for Optimal Performance**: Conversations leaned towards advanced quantization techniques, like granularity in quantizing layers to improve model performance while maintaining output quality.
   - Users shared links to [quantized models](https://huggingface.co/Joseph717171/Models/tree/main) and discussed using formats like F32 and F16 for better results.
- **Challenges in Text Prompt Categorization**: A user asked about categorizing text prompts within LM Studio but was informed that LLMs aren't effective for such tasks.
   - Hints were given to explore BERT models for text classification, which aren't yet supported in LM Studio.
- **Custom Training and Fine-tuning Limitations**: A user inquired about training models with specific datasets in LM Studio but was corrected, as the platform supports only inference.
   - Alternatives like text embeddings and fine-tuning using platforms like Hugging Face were suggested.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/gokaygokay/Florence-2">Florence 2 - a Hugging Face Space by gokaygokay</a>: no description found</li><li><a href="https://huggingface.co/legraphista/glm-4-9b-chat-1m-GGUF">legraphista/glm-4-9b-chat-1m-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/TheDrummer/Smegmma-Deluxe-9B-v1-GGUF">TheDrummer/Smegmma-Deluxe-9B-v1-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/internlm/internlm2_5-7b-chat-gguf/tree/main">internlm/internlm2_5-7b-chat-gguf at main</a>: no description found</li><li><a href="https://huggingface.co/Joseph717171/Models/tree/main">Joseph717171/Models at main</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/tasks/sequence_classification">Text classification</a>: no description found</li><li><a href="https://github.com/THUDM/CodeGeeX4">GitHub - THUDM/CodeGeeX4: CodeGeeX4-ALL-9B, a versatile model for all AI software development scenarios, including code completion, code interpreter, web search, function calling, repository-level Q&amp;A and much more.</a>: CodeGeeX4-ALL-9B, a versatile model for all AI software development scenarios, including code completion, code interpreter, web search, function calling, repository-level Q&amp;amp;A and much more. - ...</li><li><a href="https://github.com/yfzhang114/SliME?tab=readme-ov-file">GitHub - yfzhang114/SliME: ✨✨Beyond LLaVA-HD: Diving into High-Resolution Large Multimodal Models</a>: ✨✨Beyond LLaVA-HD: Diving into High-Resolution Large Multimodal Models - yfzhang114/SliME</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8348">llama : fix n_rot default by ggerganov · Pull Request #8348 · ggerganov/llama.cpp</a>: fix #8246 #8251 The logic for determining default n_rot parameter did not take into account LLM_KV_ATTENTION_KEY_LENGTH overrides. This lead to invalid context shift for Gemma2 models: # gemma-2-27...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1259164948651704342)** (4 messages): 

> - `x64bit installer for LM Studio`
> - `Features of LM Studio`
> - `Community feedback on LM Studio`
> - `Vision-enabled models`
> - `Tool calling and model capabilities` 


- **LM Studio installer confusion with x64bit**: A member questioned the absence of a 64-bit installer for LM Studio, incorrectly assuming x86 was not 64-bit.
- **Community feedback on LM Studio**: A member shared their experience with LM Studio, praising its beginner-friendly nature but expressing a need for more advanced features.
- **Calls for advanced features in LM Studio**: The same member urged LM Studio to release beta features for tool calling, RAG for file uploads, and image generation capabilities to keep up with competitors.


  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1259508784854995015)** (1 messages): 

> - `RAG applications`
> - `Optimal placement of retrieved context`
> - `System message vs final user message` 


- **Optimal Context Placement in RAG Applications**: A discussion emerged about where to place the retrieved context from a vector database in **RAG applications**—either in the system message or the final user message.
   - Members are weighing the benefits of context placement strategies to enhance system response accuracy and relevance.
- **System vs Final User Message Debate**: The debate is focused on whether embedding the context in the **system message** or the **final user message** yields better performance.
   - Participants are considering various use cases and potential impacts on the user experience.


  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1259277897831682058)** (3 messages): 

> - `internllm2_5 config`
> - `models for understanding PDFs`
> - `using LMStudio with Shell GPT` 


- **Seeking config for internllm2_5**: A member asked if anyone can share a good configuration for **internllm2_5**.
- **Looking for models to understand PDFs**: Another member inquired about suitable models for understanding **PDFs**.
- **Help needed to use LMStudio with Shell GPT**: A member sought help on how to configure **LMStudio** instead of **Ollama** with [Shell GPT](https://github.com/ther1d/shell_gpt?tab=readme-ov-file) for command-line AI productivity.
   - They tried changing `API_BASE_URL` and `DEFAULT_MODEL`, but it didn't work, and they asked for further assistance.



**Link mentioned**: <a href="https://github.com/ther1d/shell_gpt?tab=readme-ov-file">GitHub - TheR1D/shell_gpt: A command-line productivity tool powered by AI large language models like GPT-4, will help you accomplish your tasks faster and more efficiently.</a>: A command-line productivity tool powered by AI large language models like GPT-4, will help you accomplish your tasks faster and more efficiently. - TheR1D/shell_gpt

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1258864698267406450)** (44 messages🔥): 

> - `Snapdragon Elite X Machines`
> - `RAM upgrades and costs`
> - `Unified Memory in Windows and Mac`
> - `External GPUs`
> - `Feasibility of using Quad Xeon Servers for AI` 


- **Waiting on NPU support for Snapdragon Elite X**: A user expressed concerns about the price difference between **16 GB and 32 GB** RAM in **Snapdragon Elite X machines** and is considering waiting for **NPU support** before making a purchase.
   - Another user suggested considering an **M3 Max MacBook Pro** instead, highlighting its suitability for development and LLM tasks.
- **Unified Memory Transition in Windows**: **Users discussed** the potential benefits of **Windows moving to unified memory**, with comparisons made to Apple's unified memory system.
   - They speculated on upcoming technologies, with mentions of **Lunar Lake** and current Qualcomm Snapdragon X laptops potentially supporting it.
- **External GPU for Inference**: A member asked whether an **external GPU** could be used for LLM inference on a laptop.
   - It was confirmed that it is possible with proper GPU configuration, but **bandwidth bottlenecks** might be a concern.
- **Feasibility of using Quad Xeon Servers for AI**: A user questioned the viability of running LLMs on a **quad Xeon X7560** server with **256 GB DDR3 RAM**.
   - Members noted that the absence of **AVX2** support and the limitations of DDR3 RAM would make it impractical for LLM tasks.


  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1259841910793568267)** (2 messages): 

> - `Suspicious Activity in Chat`
> - `Discord Update Delays` 


- **Suspicious User Handled Quickly**: A member pointed out that **<@302816205217988609> looks suspicious**.
   - Another member confirmed that it's been dealt with and is just awaiting Discord's update: *“ty dealt with, discord just taking it's time to update.”*
- **Discord Update Delays**: Discord is experiencing delays in updating changes related to suspicious users.
   - A member reassured that the issue has been addressed, but users might still see outdated information.


  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1259796930737999984)** (1 messages): 

> - `Cost Warning Suppression`
> - `LM-Studio Configuration`
> - `Messaging Bug` 


- ****Suppress Cost Warnings**: Logging Enhancements Implemented**: A user shared a code snippet to [suppress cost warnings](https://example.link) from the **autogen.oai.client** logger by adding a custom filter to eliminate specific messages.
- ****New LM-Studio Config**: Integrating `gemma-2b-it-GGUF` Model**: The new **LM-Studio configuration** was shared, featuring the `gemma-2b-it-GGUF` model with no caching enabled and a local server setup at **http://localhost:1234/v1**.
- ****Messaging Bug from January**: Known Issue with Message Order****: A user mentioned a prior bug from January about an issue with sending system, assistant, and user messages in a specific order.


  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1259667570621939752)** (2 messages): 

> - `LM Studio`
> - `Generation Speed`
> - `Fedora 40 Kinoite`
> - `7900XTX` 


- **Record-breaking Generation Speed in LM Studio**: A user confirmed that the latest update in **LM Studio** is functioning as expected and highlighted the **wild** increase in generation speed.
- **Fedora 40 Kinoite Testing with 7900XTX**: A user mentioned their configuration of **Fedora 40 Kinoite** running with a **7900XTX** GPU.


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1259668172928188497)** (3 messages): 

> - `Removing CPU requirement for app`
> - `Forcing the model to load into RAM`
> - `GPU offload configuration` 


- **Remove CPU requirement to open app**: A user inquired about how to remove the **minimum CPU requirement** to open the app.
- **Force model to load into RAM**: A user asked how to force the model to load into **RAM** instead of **VRAM** due to slowdown issues while running **Stable Diffusion** concurrently.
   - Another user suggested to *disable GPU offload* in the side config menu as a solution.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1258879556006449252)** (325 messages🔥🔥): 

> - `Hermes 2`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic`
> - `Cloudflare blocking AI bots` 


- **Discussion on the limitations and evolution of current AI**: Members are discussing the importance of **Hermes 2** and its improved version **Hermes 2.5** in benchmarks, yet expressing concerns about models like **Mistral** struggling to extend beyond 8k without further pretraining.
   - *Merging tactics* were suggested as potential improvements for AI models, while others noted safety and context limits in AI like **Claude 3.5**.
- **Cloudflare's AI scraper bot blocking feature**: A concern was raised about **Cloudflare** introducing a feature that allows websites to block AI scraper bots, which could impact data collection for AI.
   - However, some believe that only those actively trying to block AIs will use it, and most websites will not.
- **Debate on AGI and ASI potential**: The community is debating the potential and timeline for **Artificial General Intelligence (AGI)** and **Artificial Super Intelligence (ASI)**, with comparisons to **Nvidia’s Omniverse**.
   - Members are weighing the practicality and imminence of AGI, citing **Nvidia's advancements** and discussing whether companies like **Safe Superintelligence Inc.** can achieve ASI sooner than established players like **OpenAI** or **Google**.
- **Future of automation and AI's role in the workforce**: Participants discussed the impact of AI on automating factories, noting examples like an entirely automated **BMW factory** and **Tesla’s** plans for mass-producing bots.
   - There were concerns and opinions on how these advancements would affect human labor, the efficiency of creating a 'hard drive brain,' and the balance of human-AI collaboration.
- **Community and practical implementations of AI**: Suggestions were made for practical applications, like using **OpenAI's GPT-4o's vision capabilities** for real-time object detection, while alternatives like **Computer Vision models (YOLO)** were recommended for efficiency.
   - Members shared ideas for organizing community events and meetups to discuss these advancements, and engaging in forums like **OpenAI’s Community** for better coordinated efforts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forum.openai.com">OpenAI Forum</a>: no description found</li><li><a href="https://community.openai.com">OpenAI Developer Forum</a>: Ask questions and get help building with the OpenAI platform
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1258918233990500443)** (13 messages🔥): 

> - `GPT-4o vs GPT-4`
> - `Verification issues`
> - `Custom GPTs + Zapier integration` 


- **GPT-4o perceived as faster but not necessarily better**: Community members debated whether **GPT-4o** is a better replacement for **GPT-4** due to its faster responses, though some argued it sacrifices quality.
- **Recurring verification prompt issue**: Multiple users reported encountering a persistent *'Verify your Human'* pop-up when accessing ChatGPT, which caused significant frustration.
- **Challenges with Custom GPTs and Zapier integration**: A user inquired about experiences using **custom GPTs** with **Zapier** for automating tasks, noting that Zapier's unreliability is a challenge.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1259848003699871797)** (3 messages): 

> - `Content Creation Tips`
> - `Increasing Engagement`
> - `Platform Optimization`
> - `Content Calendar Structure`
> - `Tracking Metrics for Success` 


- **Best prompts for engaging content**: A member asked **which prompts work best** for a content creator looking to create engaging content and gain followers.
   - *Another user responded with a detailed request* to ChatGPT for content ideas, engagement tips, platform-specific advice, content calendar suggestions, and key metrics to track success.
- **Strategies for engaging content creation**: User provides a comprehensive request to ChatGPT inquiring about **5-10 fresh content ideas**, strategies to boost engagement, platform-specific advice, a content calendar structure, and metrics to monitor.
   - The detailed request outlines key areas such as optimizing content for **Instagram, YouTube, and TikTok** and tracking the success in terms of follower growth and engagement.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1259848003699871797)** (3 messages): 

> - `Content creation tips`
> - `Audience engagement strategies`
> - `Platform optimization advice`
> - `Content calendar structure`
> - `Key metrics for content success` 


- **Crafting Engaging Prompts for Content Creators**: A member asked for the best prompt for content creators to create engaging content and gain followers, leading to various suggestions and discussions.
   - One user provided a detailed prompt asking for **content ideas, engagement tips, platform-specific advice, a content calendar structure, and key metrics to track success**.
- **Detailed Prompt for Content Creation Strategy**: The detailed prompt suggested included requests for **5-10 fresh content ideas based on trending topics** in the niche, strategies for boosting engagement, and platform-specific optimization advice.
   - It also recommended asking for a simple content calendar structure and key metrics to monitor the success of the content and growth in followers.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1258872092280229888)** (167 messages🔥🔥): 

> - `Qwen Model underrating`
> - `Martin Shkreli presence`
> - `SLM finetuning practice`
> - `Unsloth Studio Beta UI`
> - `AMD vs NVIDIA for LLM training` 


- **Qwen Team underrated despite great work**: Multiple members praised the **Qwen Team**'s efforts, with sentiments like "*Qwen team is so underrated*."
   - A new [Qwen training video](https://link.to.video) was deemed excellent.
- **Martin Shkreli spotted in chat**: A member pointed out the appearance of Martin Shkreli in the chat, prompting laughter and acknowledgment that he participates in related Discords.
- **Finetuning practices debated**: Discussion around **finetuning** practices highlighted that a good dataset is crucial, with emphasis on quality over quantity: "*80-90% of the time and cost of a finetune is in the dataset*."
- **Unsloth Studio Beta UI**: Unsloth is 80% done with its [Studio Beta UI](https://docs.unsloth.ai) which simplifies **finetuning** on Colab to just 1-5 clicks.
   - Future possible integration with **Gradio UI** was discussed: "*this would be a FANTASTIC idea!!*"
- **AMD vs NVIDIA debate for LLM training**: **AMD GPUs** are catching up but **NVIDIA** remains superior for LLM training due to better software and efficiency.
   - "*Most libraries don't support AMD so you will be quite limited in what you can use*."


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/gemma-2-9b">unsloth/gemma-2-9b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Docs</a>: New to Unsloth? Start here!</li><li><a href="https://huggingface.co/Replete-AI/Llama3-8B-Instruct-Replete-Adapted">Replete-AI/Llama3-8B-Instruct-Replete-Adapted · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Llama3-8B-Instruct-Replete-Adapted-bpw6-exl2">blockblockblock/Llama3-8B-Instruct-Replete-Adapted-bpw6-exl2 · Hugging Face</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/notebooks/forms.ipynb)">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1259047540209225769)** (18 messages🔥): 

> - `Kaggle disk space limit`
> - `Anthropic's steering method`
> - `Model pruning`
> - `AI research community`
> - `LivePortrait` 


- **Kaggle disk space crash**: A member broke the **Kaggle** limit and the session crashed after surpassing **100GB**.
   - They managed to save a **juicy checkpoint** on [Weights & Biases](https://wandb.ai) before the crash.
- **Anthropic steering method inquiry**: There was a discussion about **Anthropic's steering method**, and a member requested a link to the Twitter post discussing it.
   - Another confirmed reading about Explainable AI being the future but couldn't provide the link as it was not saved.
- **Pruning model assistance**: A member sought help in **pruning 15-20B parameters** from an ⌘R 35b model for their own small model family project.
   - They reached out to another member for guidance on this task.
- **Community AI research focus**: A member is building a **community focused on AI research** and invited those interested in theoretical work to join.
   - The community aims to work on significant projects without requiring coding experience.
- **LivePortrait impresses**: A member expressed being impressed by **LivePortrait**.



**Link mentioned**: <a href="https://gate-app.com/research">Gate</a>: Gate Platform

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1258872426947940434)** (120 messages🔥🔥): 

> - `Training Phi-3 with Alpaca dataset`
> - `Inference speed and efficiency of Llama-3 vs Phi 3.5 mini`
> - `Issues with GGUF conversion post training`
> - `DPO with Gemma 2 27B`
> - `RAG approach with fine tuned models` 


- **Training Phi-3 with Alpaca dataset**: A user encountered an error `xFormers wasn't built with CUDA support` while training Phi-3 with Alpaca format and was advised to update the version of `xformers` package they were using.
- **Inference speed and efficiency of Llama-3 vs Phi 3.5 mini**: A user noted that Llama-3 8B was as fast as Phi 3.5 mini, both running at 280 tokens/second, using slightly less VRAM.
   - Another user mentioned Tensorrt-llm as the current state of the art for GPU inference speed.
- **Issues with GGUF conversion post training**: A user faced a `FileNotFoundError` when trying to convert a trained model to GGUF format, specifically missing `tokenizer.model` file.
   - It was suggested to re-download the model with `FastLanguageModel.from_pretrained(..., force_download = True)` due to an update where `tokenizer.model` might have been missing initially.
- **DPO with Gemma 2 27B**: Errors occurred while using DPO with Gemma 2 27B due to automatic differentiation issues during Llama model forward operations.
   - The issue was resolved after updating Unsloth, though it noted the process would now use significantly more memory.
- **RAG approach with fine tuned models**: A user inquired about using a fine-tuned model with RAG (Retrieval-Augmented Generation) and was affirmed that it's a viable approach.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/ZLbVdvOoTKM?si=6v4YyWtROCGZcTVX">How to Build an LLM from Scratch | An Overview</a>: 👉 Need help with AI? Reach out: https://shawhintalebi.com/This is the 6th video in a series on using large language models (LLMs) in practice. Here, I revie...</li><li><a href="https://github.com/Unstructured-IO/unstructured">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.  - GitHub - Unstructured-IO/unstructured: Open source librar...</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>: no description found</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sha">Google Colab</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1259592130817429574)** (13 messages🔥): 

> - `Asking for help in forums`
> - `Need for a job channel` 


- **Don't Ask to Ask - Just Ask!**: A user shared a [link](https://dontasktoask.com/) explaining why asking if experts are around before presenting a question is bad form and inefficient.
   - The underlying message is, *'Don't waste time; just ask your question directly,'* which resonated with some members.
- **Research channel misuse prompts job channel suggestion**: Members noted that turning the research channel into a job-hunting or job-posting forum is inappropriate, with one member explicitly requesting to keep the channel on-topic.
   - The suggestion to create a dedicated job channel was made in response to the off-topic posts about seeking AI jobs.



**Link mentioned**: <a href="https://dontasktoask.com/">Don't ask to ask, just ask</a>: no description found

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1258862345225113701)** (26 messages🔥): 

> - `LLM Coding Efficiency`
> - `Bug Fix Documentation`
> - `Inspect AI Framework`
> - `Dario Amodei's Insights`
> - `Schedule-Free Optimizers` 


- **Efficient Coding with LLMs**: A user discussed how rearchitecting code to use **LLM-style APIs** simplifies complex coding tasks, emphasizing the human role in communicating and integrating systems.
   - They contended that **gluing APIs together** can turn time-consuming tasks into straightforward, zeroshot LLM prompts, saving effort in the long run.
- **Deep Dive into Bug Fix Documentation**: One user shared a [detailed bug fix](https://github.com/go-go-golems/glazed/pull/418/files) for handling string alias and declaration types, adding extensive documentation and unit tests.
   - They highlighted that although the fix took 2 hours, the resulting documentation aids future enhancement and makes it easier for LLMs to generate solutions.
- **Inspect AI Framework by UK Government**: A user was excited about trying out the new [Inspect AI framework](https://github.com/UKGovernmentBEIS/inspect_ai), which evaluates large language models.
- **Dario Amodei's Economic Impact Insights**: Anthropic’s CEO, Dario Amodei, discussed compute costs (80% of expenses) and scalable models in a [recent podcast](https://youtu.be/xm6jNMSFT7g?si=BnYoL-E1QXGTw23P&t=3880).
   - He also mentioned his past and present experiences with **Final Fantasy**, adding a personal touch to the conversation.
- **Innovations in Schedule-Free Optimizers**: A researcher reported promising results with **schedule-free optimizers** that simplify hyperparameter tuning and perform well out of the box ([details](https://x.com/Yuchenj_UW/status/1809622351543484563?s=46)).
   - The approach allows continuous learning without predefined stopping points, showing potential for widespread adoption in AI model training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1810001066240413908">Tweet from OpenRouter (@OpenRouterAI)</a>: Announcing a brand-new model marketplace UI ✨  Explore 180 active language models processing 74 billion tokens/week 👇</li><li><a href="https://youtu.be/xm6jNMSFT7g?si=BnYoL-E1QXGTw23P&t=3880">Dario Amodei - CEO of Anthropic | Podcast | In Good Company | Norges Bank Investment Management</a>: Dario Amodei CEO of Anthropic: Claude, New models, AI safety and Economic impactHow much bigger and more powerful will the next AI models be? Anthropic’s CEO...</li><li><a href="https://x.com/yuchenj_uw/status/1809622351543484563?s=46">Tweet from Yuchen Jin (@Yuchenj_UW)</a>: I trained GPT-2 (124M) with @aaron_defazio&#39;s Schedule-Free optimizer on @karpathy&#39;s nanoGPT:  - Settings: AdamW with learning rate=0.0018 (same as https://x.com/Yuchenj_UW/status/1795850420503...</li><li><a href="https://x.com/firstadopter/status/1809633896436269347?s=46">Tweet from tae kim (@firstadopter)</a>: Anthropic’s CEO Dario Amodei says compute is more than 80% of their expenses on a podcast. Salaries of 600 employees are much smaller expense</li><li><a href="https://x.com/norabelrose/status/1810342367972495590?s=46">Tweet from Nora Belrose (@norabelrose)</a>: The @AiEleuther interpretability team is releasing a set of top-k sparse autoencoders for every layer of Llama 3 8B: https://huggingface.co/EleutherAI/sae-llama-3-8b-32x  We are working on an automate...</li><li><a href="https://x.com/alexalbert__/status/1810376544734556540">Tweet from Alex Albert (@alexalbert__)</a>: Two days left to participate in the contest!  Quoting Alex Albert (@alexalbert__)   Announcing the Build with Claude June 2024 contest.  We&#39;re giving out $30k in Anthropic API credits. All you nee...</li><li><a href="https://github.com/UKGovernmentBEIS/inspect_ai">GitHub - UKGovernmentBEIS/inspect_ai: Inspect: A framework for large language model evaluations</a>: Inspect: A framework for large language model evaluations - UKGovernmentBEIS/inspect_ai</li><li><a href="https://github.com/wesen/glazed/blob/e180e5d59031f20009c461466a2995ff28ee25a7/pkg/doc/topics/13-layers-and-parsed-layers.md">glazed/pkg/doc/topics/13-layers-and-parsed-layers.md at e180e5d59031f20009c461466a2995ff28ee25a7 · wesen/glazed</a>: a library to make it easy to output structured data in your command line tools. add the icing on top of your data - wesen/glazed</li><li><a href="https://aipapersoftheweek.substack.com/p/ai-papers-of-the-week-july-3rd-2024">AI papers of the week - July 3rd, 2024 - Kyutai Moshi, Meta 3D Gen, etc.</a>: Papers covered: AI Agents that Matter Kyutai Moshi Meta 3D Gen Open-TeleVision: Teleoperation with Immersive Active Visual Feedback PathAlign: A vision-language model for whole slide images in histopa...</li><li><a href="https://github.com/go-go-golems/glazed/pull/418/files">:ambulance: :umbrella: :books: Handle string alias and string declaration types for layers/parameters by wesen · Pull Request #418 · go-go-golems/glazed</a>: This adds code to handle string alias and string declaration handling in the parameters/layers/reflect modules of glazed. It uses the opportunity to add a lot of documentation and unit tests.</li><li><a href="https://www.nbim.no/en/publications/podcast/dario-amodei-ceo-of-anthropic-claude-new-models-ai-safety-and-economic-impact/">Dario Amodei CEO of Anthropic: Claude, new models, AI safety and economic impact | Norges Bank Investment Management</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1258895771055427635)** (5 messages): 

> - `HN post for podcast`
> - `Fortnite's new game mode`
> - `Communication problems at work`
> - `Upvotes and engagement on HN` 


- **Podcast episode shared on Hacker News**: [Now on HN!](https://news.ycombinator.com/newest) A user shared a link to a recent podcast episode on Hacker News, hoping to gain traction.
- **Engagement on Fortnite article**: A discussion emerged around a [Polygon article](https://www.polygon.com/gaming/24185789/fortnite-reload-new-game-mode) about **Fortnite removing crossovers** to regain its fun factor.
   - The article received initial engagement, with **1 upvote** and was shared by a user named PaulHoule.
- **Handling communication issues at work**: Another interesting topic on HN was about [dealing with a colleague's communication problem](https://www.nytimes.com/2024/07/07/business/work-friend-anna-holmes.html), *shared by jaredwiener*.
- **Community engagement on HN**: A user expressed support by upvoting the podcast episode shared on HN, encouraging ongoing participation.



**Link mentioned**: <a href="https://news.ycombinator.com/newest">New Links | Hacker News</a>: no description found

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1258874839163207785)** (243 messages🔥🔥): 

> - `AI in Action`
> - `AI Engineer World Fair`
> - `LlamaFile vs. Ollama`
> - `Model Merging`
> - `Wearables and Privacy` 


- **AI Engineer World Fair Insights**: The AI Engineer World Fair featured notable talks including Justine Tunney's keynote, a highly-praised AI leadership workshop, and interesting discussions on LLMs and model merging.
   - A member noted that despite some logistics issues, the conference was well-received with diverse, high-energy sessions on topics like **AI-generated music** and **Tool Use with Open-Source LLMs**.
- **LlamaFile vs. Ollama Debate**: Members discussed the differences between LlamaFile and Ollama, with LlamaFile focusing on **portability and optimization**, and Ollama on **compatibility with a large amount of models**.
   - Some members expressed the desire for an adapter to combine the strengths of both tools, suggesting that **Ollama might function as a Llama.cpp wrapper**.
- **Model Merging Techniques Explored**: Model merging was a hot topic, with members sharing resources like the [mergekit GitHub](https://github.com/arcee-ai/mergekit) and new updates on merging strategies.
   - The possibility of using deep learning models to converge on the best model merging strategy was discussed, though it was noted this approach might be **intellectually suspect**.
- **Wearables Privacy Concerns**: Concerns were raised about wearable devices and consent to record off-mic moments during events.
   - A solution involving desktop integration and notification features for wearables was proposed to ensure user awareness and consent.
- **Future Conference Planning**: Discussions on next year's AI Engineer World Fair included extending the event by an extra day or incorporating a break day with activities.
   - Ideas such as a dedicated track for AI girlfriend applications and gamification of the conference schedule were suggested to enhance attendee experience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.workshopsurvival.com">The Workshop Survival Guide</a>: Learn how to design and teach educational workshops that work every time. Now available on Amazon.</li><li><a href="https://www.ivanleo.com/blog/ai-conf">AI Engineering World Fair</a>: no description found</li><li><a href="https://aie.compasswearable.com/events">AI Engineers World Fair Recaps - Powered by Compass</a>: Experience the biggest technical AI conference with live transcriptions and AI-generated summaries.</li><li><a href="https://x.com/latentspacepod/status/1805836033445216644">Tweet from Latent.Space (@latentspacepod)</a>: @aiDotEngineer huge turnout!</li><li><a href="https://x.com/philip_kiely/status/1808589566921879702">Tweet from Philip Kiely (@philip_kiely)</a>: Here are 3 themes I picked up in 3 incredibly high-energy days at @aiDotEngineer World&#39;s Fair:  1. Open source is closing the gap 2. Inference everywhere 3. Evals are everything  Details:</li><li><a href="https://x.com/RickLamers/status/1808705188024439187">Tweet from Rick Lamers (@RickLamers)</a>: Model merging is nuts, check out this family tree :0</li><li><a href="https://github.com/arcee-ai/mergekit">GitHub - arcee-ai/mergekit: Tools for merging pretrained large language models.</a>: Tools for merging pretrained large language models. - arcee-ai/mergekit</li><li><a href="https://docs.google.com/document/d/1TLXkcaNX6cvpiqqyo952_K2a7XTF064R44v3WL9CSbE/edit?usp=sharing">AI Engineering Worlds Fair</a>: AI Engineering Worlds Fair   Thomas Dohmke Human centric approach - “co-pilot” Copilot helps devs be in the flow of software Democratizes access to information - onboarding Agent - ai dishwasher (side...</li><li><a href="https://docs.google.com/presentation/d/1A_yLcD6Sy1Nr_v2YesOzvtcg5yAmmrfPR2bU4dyxTzw/edit?usp=sharing">AI in action - 2024-07-05</a>: AI in action AI Engineers World Fair recap 2024-07-05</li><li><a href="https://codingwithintelligence.com/p/ai-engineer-world-fair-in-sf">AI Engineer World Fair in SF</a>: Week 26 of Coding with Intelligence</li><li><a href="https://x.com/intertwineai/status/1807060271828975632">Tweet from Bryan Young (@intertwineai)</a>: @aiDotEngineer Day 3 Recap and Wrap!  1/12: Day 3 of #AIEWF 2024 is over and it&#39;s clear we&#39;re just scratching the surface of AI&#39;s potential and defining what an @aiDotEngineer is.  Here&#3...</li><li><a href="https://x.com/intertwineai/status/1806270266965889289">Tweet from Bryan Young (@intertwineai)</a>: @aiDotEngineer 2nd Day Recap!  1/14. The second day started with a timely session on AI-generated music by @YoungPhlo_. We all made some sick beats together.   Although the fresh @RIAA lawsuits agains...</li><li><a href="https://x.com/intertwineai/status/1805867608593645916">Tweet from Bryan Young (@intertwineai)</a>: 1/5: Day 1 of @aiDotEngineer was just as exciting as I thought it would be! #AIEWF  Quick recap of the day:
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1258900955001262250)** (10 messages🔥): 

> - `CUDA Certification vs GitHub Repos`
> - `NVIDIA Deep Learning Institute`
> - `Peak FLOPS Comparison`
> - `Educational Expense Strategies` 


- **Public GitHub Repos Trump CUDA Certification**: A user raised a question about preferring a CUDA certification course versus GitHub links to CUDA kernels when hiring, sparking a debate on the value of public work over certificates.
   - *as_ai* stated, *"proven work that is public is always more valuable than a paper that doesn’t tell the full story."*
- **NVIDIA Deep Learning Institute Resources**: A user recommended the [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/) for various educational resources, citing personal experience from courses held at their university.
   - The institute offers self-paced and live training programs covering AI, accelerated computing, and more—ideal for using company learning budgets.
- **Mind the Gap: Comparing GPU Peak FLOPS**: A user shared surprising performance numbers, noting that the 4090 Ti has a peak of **93 TFLOPS** while the A100 only 19.5 TFLOPS for single precision.
   - *eriks.0595* explained that comparing Ampere and Ada architectures shows differences, with Ada having improved FP32 throughput as noted in the [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html).
- **Expense Strategies for Educational Purposes**: A user humorously suggested expensing a GPU and claiming it's for educational purposes.
   - The discussion centered around creative ways to use company learning budgets for personal gains and upskilling.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#improved-fp32-throughput">NVIDIA Ada GPU Architecture Tuning Guide</a>: no description found</li><li><a href="https://www.nvidia.com/en-us/training/.">NVIDIA Deep Learning Institute and Training Solutions</a>: We provide hands-on training in AI, accelerated computing, and accelerated data science.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1259172056692555837)** (10 messages🔥): 

> - `torch.compile the missing manual`
> - `PyTorch tensors and type erasure`
> - `Flexibility vs templates in graph creation`
> - `PyTorch profiler and FLOP estimation`
> - `FlopCounterMode vs with_flops` 


- **torch.compile manual clarifies usage**: A member shared a link to [torch.compile, the missing manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab), emphasizing its usefulness.
- **Discussion on PyTorch tensors using type erasure**: A member inquired about documentation on why **PyTorch tensors** use extensive type erasure and the benefits over using more templates.
   - *Type erasure simplifies handling across Python and C++ frontends,* cited an example of challenges with templates requiring complicated macros or if/else statements.
- **PyTorch profiler's FLOP estimation feature**: A member was intrigued by the `with_flops` argument in **PyTorch profiler** which estimates FLOPs taken by a model, though this isn't well-documented.
   - Another member suggested using `torch.utils.flop_counter.FlopCounterMode` for FLOP counting as `with_flops` is not actively developed.



**Link mentioned**: <a href="https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab">torch.compile, the missing manual</a>: torch.compile, the missing manual You are here because you want to use torch.compile to make your PyTorch model run faster. torch.compile is a complex and relatively new piece of software, and so you ...

  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1259879862991196253)** (1 messages): 

> - `Compiler enthusiasts job opening`
> - `Thunder compiler optimization project` 


- **Lightning AI seeks Compiler Enthusiasts**: A job opening at [Lightning AI](https://boards.greenhouse.io/lightningai/jobs/6045025003) is available for those who like compilers and working in a team with notable colleagues including Luca Antiga.
- **Thunder boosts PyTorch models performance**: The [Thunder project](https://github.com/Lightning-AI/lightning-thunder) by Lightning AI promises to make **PyTorch models up to 40% faster** through a source-to-source compiler.
   - Thunder enables the use of different hardware executors simultaneously, whether it's a single GPU or thousands.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://boards.greenhouse.io/lightningai/jobs/6045025003">Compiler Engineer</a>: London, England, United Kingdom</li><li><a href="https://github.com/Lightning-AI/lightning-thunder">GitHub - Lightning-AI/lightning-thunder: Make PyTorch models up to 40% faster! Thunder is a source to source compiler for PyTorch. It enables using different hardware executors at once; across one or thousands of GPUs.</a>: Make PyTorch models up to 40% faster! Thunder is a source to source compiler for PyTorch. It enables using different hardware executors at once; across one or thousands of GPUs. - Lightning-AI/ligh...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1258880966047432796)** (20 messages🔥): 

> - `CUDA beginner project`
> - `Using Python vs. C++ for CUDA`
> - `Starting with CUDA`
> - `Tinygrad and Teenygrad`
> - `SpMM with 2:4 sparsity pattern` 


- **CUDA beginner project ideas**: A member mentioned they want to start a CUDA project and wondered if implementing Flash attention is suitable; they're open to suggestions and collaboration.
   - Others volunteered ideas like looking at teenygrad or suggested more manageable projects due to the complexity.
- **Community recommends Python for CUDA learners**: A member debated using Python vs. C++ for writing a deep learning framework with CUDA, concerned about complexity and performance.
   - The community suggested starting with Python and CUDA Python, citing examples like llama.c or Karpathy's repos for easier understanding.
- **Study recommendation for deep learning beginners**: Several community members recommended top-down and bottom-up approaches to understand the mathematical fundamentals before diving into coding.
   - They stressed understanding a forward and backward pass of a simple neural network as essential groundwork.
- **Comparison between cusparseLT and CUTLASS for SpMM**: A community member asked if there's a performance difference between cusparseLT and CUTLASS for SpMM with a 2:4 sparsity pattern.
   - It's suggested that cusparseLT might be more rigorously optimized and maintained.
- **Resources for learning CUDA**: A beginner asked for resources to start learning GPU programming with CUDA.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1259004537142775848)** (19 messages🔥): 

> - `2:4 sparsity with int8 quantization`
> - `Hackable pure Python low-bit optimizers`
> - `Non-contiguous gradients issue`
> - `FP8 Adam optimization`
> - `Regression tests on CI machines` 


- **2:4 sparsity now composes with int8 quantization**: This new feature was quietly added, allowing **2:4 sparsity** to compose with **int8 quantization**, with a simple implementation in [Python code](https://github.com/pytorch/ao).
- **Pure Python low-bit optimizers available**: [TorchAO](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim) now has hackable pure Python implementations of **8-bit** and **4-bit** optimizers.
- **Non-contiguous gradients issue discussed**: The topic of using `.view()` versus `.reshape()` for handling non-contiguous gradients was debated in relation to **torchao** optimizers.
- **Experimentation with FP8 Adam optimizer**: An experiment to replace custom quant/dequant logic in **FP8 Adam** with hardware instructions (requiring **Ada** or **Hopper**) shows promising results.
- **Regression tests on CI machines**: Using multiple GPUs on CI machines, a specific **benchmark script** can replace the test suite to print results to the console.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/486">Enable `model.to(device)` for int8 weight only quantized model by jerryzh168 · Pull Request #486 · pytorch/ao</a>: Summary: Fix some implementation issue for int8_wo_quantized_model.to(device) Test Plan: python test/quantization/test_quant_api.py -k test_quantized_model_to_device Reviewers: Subscribers: Tasks: ...</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim">ao/torchao/prototype/low_bit_optim at main · pytorch/ao</a>: Create and integrate custom data types, layouts and kernels for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/pull/403/files#diff-c9eb698c2226c153d926c4709d378e3349d020f4a18bc59245d60320cc317e5fR585">add FSDP QLoRA test and revert failing PR by weifengpy · Pull Request #403 · pytorch/ao</a>: fix error when running torchtune QLoRA + FSDP2 #380 TypeError: nf4_detach() missing 1 required positional argument: &amp;#39;args&amp;#39; torchtune command tune download meta-llama/Llama-2-7b-hf --ou...</li><li><a href="https://github.com/pytorch/ao/pull/403/files">add FSDP QLoRA test and revert failing PR by weifengpy · Pull Request #403 · pytorch/ao</a>: fix error when running torchtune QLoRA + FSDP2 #380 TypeError: nf4_detach() missing 1 required positional argument: &amp;#39;args&amp;#39; torchtune command tune download meta-llama/Llama-2-7b-hf --ou...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1258943198395502684)** (13 messages🔥): 

> - `AliExpress Anniversary Promo`
> - `Creative Pixel Art Tool`
> - `Summer Vacation for Startup Founders`
> - `Techsupportgore Subreddit`
> - `Potential Online Scams` 


- **AliExpress Anniversary Promo Sparks Skepticism**: Members expressed doubt about an AliExpress promotion offering an RTX 4090 for $430 with bulk purchase incentives, calling it unbelievable.
   - One comment sarcastically suggested that buyers might receive a mere printed picture of the 4090 instead of the actual product.
- **Startup Founders Can't Relate to Vacations**: A user joked about not knowing what a summer vacation is while living in the US, highlighting the continuous grind in the startup world.
   - Another member humorously noted, 'Startup founders: what's a vacation?' emphasizing the constant work culture.
- **Techsupportgore Subreddit Protests Reddit's API Policy**: Discussion included [Techsupportgore subreddit](https://www.reddit.com/r/techsupportgore/comments/xorwdy/a_crypto_miner_rinsing_cards_off_with_a_pressure/) known for cringe-worthy tech support moments, currently protesting Reddit's API policies.
   - Users are warned that the subreddit isn't for seeking tech support but rather for viewing and posting photos of poor tech practices.
- **Pixel Mirror Turns Reality into Pixel Art**: A [new tool](https://www.yankodesign.com/2024/07/04/this-crystal-fragment-turns-everything-you-see-into-8-bit-pixel-art-and-its-fascinating/) called the Pixel Mirror by designer Hakusi Katei transforms real-world views into 8-bit pixel art, blending analog and digital experiences.
   - The product appeals to nostalgic fans of early computer graphics, creating pixelated images through a crystal with unique resolution reduction properties.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.aliexpress.com/item/1005006997501364.html?src=google">ORIGINAL NEW Fast selling NVIDIA GeForce RTX 4090 Founders Edition Graphics Card 24GB - AliExpress 1420</a>: Smarter Shopping, Better Living! Aliexpress.com</li><li><a href="https://www.reddit.com/r/techsupportgore/comments/xorwdy/a_crypto_miner_rinsing_cards_off_with_a_pressure/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.yankodesign.com/2024/07/04/this-crystal-fragment-turns-everything-you-see-into-8-bit-pixel-art-and-its-fascinating/">This Crystal Fragment turns everything you see into 8-bit Pixel Art, and it’s FASCINATING - Yanko Design</a>: https://www.youtube.com/watch?v=v4VN2ZZZT9c&amp;feature=youtu.be There is no denying that modern graphic resolutions have reached unachievable heights. Yet, there are many with an emotional connect to...</li><li><a href="https://www.reddit.com/r/techsupportgore/comments/xorwdy/a_crypto_min">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

fancytrevor: curious if anyone has sf meetup
recommendations
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1258911594310533120)** (179 messages🔥🔥): 

> - `muP Experiments`
> - `FP8 Precision`
> - `CUDA Checkpointing`
> - `Inference Optimizations`
> - `LLM101n Course Plan` 


- **muP Experiments yield mixed results**: The team's muP experiments didn't significantly surpass the baseline, with mixed results on hyperparameters like `attn_mult` needing further exploration.
- **FP8 precision exploration**: Discussions around the use of FP8 for certain matmuls, particularly its benefits for final layers, ongoing efforts to benchmark and optimize FP8 usage.
- **NVIDIA checkpointing utility interest**: NVIDIA's new [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) utility and its integration with CRIU for fine-grained checkpointing sparked interest among members.
- **Inference optimizations through reduced batch sizes**: [PR #671](https://github.com/karpathy/llm.c/pull/671) changes inference checks to use minimum B/T rather than maximum, aiming for faster performance without divergence.
- **LLM101n course and development plans**: Plans discussed for a stepwise LLM development course (LLM101n), covering foundational building blocks like micrograd, minBPE, and progressing to advanced topics like FP8 and multimodal training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/EleutherAI/pythia-1.4b-v0">EleutherAI/pythia-1.4b-v0 · Hugging Face</a>: no description found</li><li><a href="https://github.com/google-research/t5x/blob/0728d8429041d6c6e75077334e76eb2370c6057b/t5x/losses.py#L25-L57)">t5x/t5x/losses.py at 0728d8429041d6c6e75077334e76eb2370c6057b · google-research/t5x</a>: Contribute to google-research/t5x development by creating an account on GitHub.</li><li><a href="https://github.com/ash-01xor/bpe.c/tree/main">GitHub - ash-01xor/bpe.c: Simple Byte pair Encoding mechanism used for tokenization process . written purely in C</a>: Simple Byte pair Encoding mechanism used for tokenization process . written purely in C - ash-01xor/bpe.c</li><li><a href="https://github.com/karpathy/llm.c/pull/671">Faster inference by changing (B,T) to (1,t) by ademeure · Pull Request #671 · karpathy/llm.c</a>: The inference sanity checks currently process all (B,T) despite only needing (1,64) by default. This PR is bit-for-bit identical to previous versions while reducing this to (1,t) where t is rounded...</li><li><a href="https://github.com/NVIDIA/nccl/issues/1026">half precision reduction accumulation in fp32? · Issue #1026 · NVIDIA/nccl</a>: Are there plans to fix NCCL to perform reductions on BFLOAT16 operands with fp32 accumulation? Otherwise we can&#39;t reduce grads without a large loss and have to use fp32 comms which is both expensi...</li><li><a href="https://github.com/ademeure/llm.c/blob/fp8_phase1/dev/cuda/advanced_copy_transpose.cu">llm.c/dev/cuda/advanced_copy_transpose.cu at fp8_phase1 · ademeure/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to ademeure/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/discussions/1505">Flash attention · tinygrad/tinygrad · Discussion #1505</a>: Is something like Flash Attention (2) automatically computed by tinygrad with all the lazy (expression template ?) stuff?</li><li><a href="https://x.com/__tinygrad__/status/1802435228570616192">Tweet from the tiny corp (@__tinygrad__)</a>: The main area where we are behind is NVIDIA speed, especially for LLM training since we don&#39;t have flash attention and have an awful softmax.  The main area where we are ahead is portability. tiny...</li><li><a href="https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu/">Checkpointing CUDA Applications with CRIU | NVIDIA Technical Blog</a>: Checkpoint and restore functionality for CUDA is exposed through a command&#x2d;line utility called cuda&#x2d;checkpoint. This utility can be used to transparently checkpoint and restore CUDA state wi...
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1259448514463662100)** (3 messages): 

> - `Critiques in Preference Learning`
> - `Test-Time-Training Layers` 


- **Synthetic critiques enhance reward models**: @Daniella_yz explored using synthetic critiques from **large language models** to improve reward models during a @Cohere internship, as detailed in [their preprint](https://arxiv.org/abs/2405.20850).
   - *Beyond assisting human evaluation (e.g., CriticGPT), critiques can directly enhance preference learning*.
- **New architecture replaces RNN hidden state**: @karansdalal shared a new architecture, **Test-Time-Training layers (TTT layers)**, which replaces the hidden state of an RNN with a machine learning model and compresses context through gradient descent on input tokens, as discussed in [their preprint](https://arxiv.org/abs/2407.04620).
   - This innovation enables **linear complexity architectures** with expressive memory, allowing training of LLMs with millions or billions of tokens in context, with instantiations TTT-Linear and TTT-MLP both matching or beating the strongest Transformers and Mamba.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Daniella_yz/status/1809720946066092097">Tweet from Daniella Ye (@Daniella_yz)</a>: Beyond their use in assisting human evaluation (e.g. CriticGPT), can critiques directly enhance preference learning? During my @Cohere internship, we explored using synthetic critiques from large lang...</li><li><a href="https://x.com/karansdalal/status/1810338845659131940?s=46">Tweet from Karan Dalal (@karansdalal)</a>: I’m excited to share a project I’ve been working on for over a year, which I believe will fundamentally change our approach to language models.  We’ve designed a new architecture, which replaces the h...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1259264486523670663)** (2 messages): 

> - `Nous Magazine`
> - `Cryptoland`
> - `YouTube video`
> - `Fantasy Division` 


- **Upcoming Nous Magazine Sneak Peek**: John0galt has shared the first couple of pages from the upcoming **Nous Magazine**.
- **Cryptoland Explored in YouTube Video**: Iron_bound posted a [YouTube video titled 'Whatever Happened to Cryptoland?'](https://www.youtube.com/watch?v=W9ggP26yH7A) highlighting unforeseen events in the cryptocurrency world.
   - They also shared a link to [Fantasy Division](https://fantasydivision.online/References) and a related [Google Docs document](https://docs.google.com/docu).



**Link mentioned**: <a href="https://www.youtube.com/watch?v=W9ggP26yH7A">Whatever Happened to Cryptoland?</a>: there is no way anyone could&#39;ve seen this coming⚔️ You&#39;re gunna wanna check this out: https://fantasydivision.online/References: https://docs.google.com/docu...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1259435421683941388)** (2 messages): 

> - `Dataline by RamiAwar`
> - `LLM Reasoning Capabilities` 


- **Chat with Your Data Using Dataline**: A new GitHub project called [Dataline](https://github.com/RamiAwar/dataline) offers AI data analysis and visualization across multiple databases like **CSV**, **Postgres**, **MySQL**, **Snowflake**, and **SQLite**.
- **Exploring LLM Reasoning Capabilities via Geometry**: A new paper on arXiv, [The Geometrical Understanding of LLMs](https://arxiv.org/abs/2407.02678), explores the connection between LLMs' reasoning abilities and the density of their self-attention graphs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.02678">Reasoning in Large Language Models: A Geometric Perspective</a>: The advancement of large language models (LLMs) for real-world applications hinges critically on enhancing their reasoning capabilities. In this work, we explore the reasoning abilities of large langu...</li><li><a href="https://github.com/RamiAwar/dataline">GitHub - RamiAwar/dataline: Chat with your data - AI data analysis and visualization on CSV, Postgres, MySQL, Snowflake, SQLite...</a>: Chat with your data - AI data analysis and visualization on CSV, Postgres, MySQL, Snowflake, SQLite... - RamiAwar/dataline
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1258872555184324688)** (211 messages🔥🔥): 

> - `GPT4 Benchmark Scores`
> - `Temperature Effects`
> - `In-Context Learning Examples`
> - `Prompt Caching Costs`
> - `BitNet Training` 


- **GPT4 scores higher with increased temperature**: A member reported that GPT4 scores higher on benchmarks with higher temperatures, but another member couldn't reproduce these results with local models.
- **In-Context Learning (ICL) increases model performance**: Members discussed the impact of increasing the number of examples in In-Context Learning, agreeing that more examples enhance model performance.
- **BitNet garners interest but faces training challenges**: Members expressed interest in the BitNet architecture, with some wanting to train models using its 1.58-bit format to save memory.
- **Expect rapid advancements in generative video tech**: Members are optimistic that generative video technology will achieve real-time generation within the next 1-1.5 years, driven by strong incentives and current developmental speeds.
- **Access to fine-tuning resources and guidance**: Participants shared fine-tuning resources and discussed creating diverse, high-quality data from raw documents using models like Ada Instruct and Nous' Genstruct 7B.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/fellowship">Announcing the Hugging Face Fellowship Program</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/OLMo-Bitnet-1B">NousResearch/OLMo-Bitnet-1B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5">Experiments with Bitnet 1.5 (~ngmi~)</a>: no description found</li><li><a href="https://arxiv.org/html/2407.03040v1">Raw Text is All you Need: Knowledge-intensive Multi-turn Instruction Tuning for Large Language Model</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/go-green-cool-swag-dance-gif-12671410">Go Green GIF - Go Green Cool - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/hugg">Hugg (Yy)</a>: no description found</li><li><a href="https://x.com/karan4d/status/1768836844207378463">Tweet from mephisto ∃ (@karan4d)</a>: im opensourcing worldsim of course i am  worldsim sysprompt and conversation to intitialize:  sysprompt:  &lt;sys&gt;Assistant is in a CLI mood today. The human is interfacing with the simulator direc...</li><li><a href="https://x.com/norabelrose/status/1810342367972495590?s=46">Tweet from Nora Belrose (@norabelrose)</a>: The @AiEleuther interpretability team is releasing a set of top-k sparse autoencoders for every layer of Llama 3 8B: https://huggingface.co/EleutherAI/sae-llama-3-8b-32x  We are working on an automate...</li><li><a href="https://huggingface.co/Replete-AI/Llama3-8B-Instruct-Replete-Adapted">Replete-AI/Llama3-8B-Instruct-Replete-Adapted · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blockblockblock/Llama3-8B-Instruct-Replete-Adapted-bpw6-exl2">blockblockblock/Llama3-8B-Instruct-Replete-Adapted-bpw6-exl2 · Hugging Face</a>: no description found</li><li><a href="https://github.com/microsoft/MInference">GitHub - microsoft/MInference: To speed up Long-context LLMs&#39; inference, approximate and dynamic sparse calculate the attention, which reduces inference latency by up to 10x for pre-filling on an A100 while maintaining accuracy.</a>: To speed up Long-context LLMs&amp;#39; inference, approximate and dynamic sparse calculate the attention, which reduces inference latency by up to 10x for pre-filling on an A100 while maintaining accu...</li><li><a href="https://github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models">GitHub - ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models: This repository collects all relevant resources about interpretability in LLMs</a>: This repository collects all relevant resources about interpretability in LLMs - ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models</li><li><a href="https://huggingface.co/Weyaxi/Einstein-v7-Qwen2-7B">Weyaxi/Einstein-v7-Qwen2-7B · Hugging Face</a>: no description found</li><li><a href="https://fxtwitter.com/Weyaxi/status/1809644014515154961">Tweet from Weyaxi (@Weyaxi)</a>: 🚀 Introducing 𝐄𝐢𝐧𝐬𝐭𝐞𝐢𝐧 𝐯𝟕, based on the increcible 𝐐𝐰𝐞𝐧𝟐 𝟕𝐁 model, supervised fine-tuned using diverse, high-quality datasets!     📊 Version 7 adds SystemChat and a portion of the a...</li><li><a href="https://anyscale.com/blog/fine-tuning-is-for-form-not-facts)">Blog | Anyscale</a>: Anyscale is the leading AI application platform. With Anyscale, developers can build, run and scale AI applications instantly.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1259276528160739338)** (2 messages): 

> - `Deterministic Reports with LLMs`
> - `Integration of Traditional Programming` 


- **Seeking methods for deterministic reporting using LLMs**: **nav10** asked for methods to create deterministic reports using LLMs for identifying bottlenecks in business processes, aiming for an 80%+ consistency rate.
   - **nav10** is considering structured generation and ranking possibilities with an LLM judge.
- **Advice on combining traditional programming and LLMs**: A member, **deoxykev**, advises coding the deterministic parts in a conventional language and using LLMs for small, structured tasks where traditional programming isn't efficient.
   - *The trick is to use LLMs as little as possible, and when you do, only let them do constrained, simple tasks.*


  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1259557677701271744)** (4 messages): 

> - `RAG and Hallucinations`
> - `Wikipedia-Style Citations`
> - `RAG and Hugging Face Agents` 


- **RAG Hallucinations Examined**: A YouTube video ["Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools"](https://youtu.be/no7EQkOiHQM?si=b35yua7rZuaEVvKu) discusses a Stanford paper on the degree of hallucinations in various LegalTech tools.
   - *Examining how RAG models handle legal queries gives insight into their hallucination rates and reliability in critical applications.*
- **Wikipedia-Style Citations Proposed**: Members discussed using Wikipedia-style `<ref> </ref>` tags for citations, citing familiarity of base models with this format from pretraining.
   - One member shared an [example template](https://en.wikipedia.org/wiki/Template:Ref) to illustrate how to format these citations properly.
- **Criminally Underrated RAG Tutorials**: A tweet highlighted [@AymericRoucher's](https://x.com/mervenoyann/status/1810291157961781671) RAG and Agents tutorials in the Hugging Face Cookbook, noting that agentic RAG outperforms standard RAG.
   - *These tutorials provide invaluable insights and techniques for enhancing RAG performance, making them essential reading.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/no7EQkOiHQM?si=b35yua7rZuaEVvKu">Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools (Paper Explained)</a>: #rag #hallucinations #legaltech An in-depth look at a recent Stanford paper examining the degree of hallucinations in various LegalTech tools that incorporat...</li><li><a href="https://x.com/mervenoyann/status/1810291157961781671">Tweet from merve (@mervenoyann)</a>: Criminally underrated: RAG and Agents tutorials by @AymericRoucher at @huggingface Cookbook 📝  Latest one is on agentic RAG which outperforms standard RAG  find all below this one ⥥</li><li><a href="https://example.com">Example Domain</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Template:Ref">Template:Ref - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1258870858672242688)** (5 messages): 

> - `WorldSIM simulation success`
> - `Next era of simulation` 


- **WorldSIM Buddha sim achieves enlightenment swiftly**: A user shared their experience of creating a world rooted in **Buddhist principles** that evolved into a **single enlightened population** in under 30K steps, calling it 'almost too easy'.
   - They mentioned blowing through all their credits in a single lunch hour due to this simulation.
- **Anticipation builds for next era of simulation**: A member teased that resources are currently directed towards the **next era of simulation** they are working on.
   - This prompted excitement and curiosity among others in the channel.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1258872163432271964)** (68 messages🔥🔥): 

> - `Updating WSL to WSL2 for Mojo Installation`
> - `Dependency Hell in Python`
> - `Mojo Rounding Function Bugs`
> - `Getting 'Mojician' Badge`
> - `Mojo's Int64 vs Float64 Behavior` 


- **Updating WSL to WSL2 for Mojo Installation**: Users discussed problems related to updating WSL to WSL2 for installing Mojo, with issues arising particularly for those on older Windows 10 computers.
   - Links were shared to [Microsoft's guide on installing WSL](https://learn.microsoft.com/en-us/windows/wsl) which helped solve the problem for a user after several hours of trying.
- **Dependency Hell Nightmare in Python**: A user queried about handling conflicting dependency versions in Python projects, to which other users responded by discussing that the only known solution is side-by-side installations or using virtual environments.
   - An interesting discussion emerged on whether Mojo or other systems can handle this problem, pointing to a [GitHub discussion](https://github.com/modularml/mojo/discussions/1401) suggesting improvements.
- **Mojo's Struggles with Rounding Functions**: Several users uncovered multiple bugs regarding the `round` function in Mojo, particularly with int and float types not rounding as expected.
   - While discussing the inconsistencies, users identified that SIMD rounding in Mojo does not use the second parameter properly, resulting in unexpected outputs.
- **Steps to Get a 'Mojician' Badge**: Users inquired about how to get the 'Mojician' badge on the server, discovering that you need to create something cool in Mojo and post it to Community Posts.
- **Unexpected Behavior in Mojo's Int64 and Float64 Types**: Through the discussion, users noted that Mojo's handling of `Int64` and `Float64` types leads to unexpected behavior when using rounding functions.
   - The `Roundable` trait in Mojo currently has limitations, which causes rounding to always occur to zero decimal places despite specifying otherwise.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/windows/wsl/install">Install WSL</a>: Install Windows Subsystem for Linux with the command, wsl --install. Use a Bash terminal on your Windows machine run by your preferred Linux distribution - Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin,...</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/math/round">round | Modular Docs</a>: roundT: Roundable -&gt; $0</li><li><a href="https://github.com/modularml/mojo/discussions/1401">Auto patch 3rd party imports to mitigate dependency hell · modularml/mojo · Discussion #1401</a>: The Wikipedia article dependency hell provides a good definition and list of dependency hell forms. The proposed solution should solve all the forms of the dependency hell, but for the sake of clar...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1259055427094839306)** (10 messages🔥): 

> - `__del__ method in Mojo`
> - `Mojo 3D Graphics Examples`
> - `Common libc functions in Mojo`
> - `Cross compilation with Mojo`
> - `Using partition method in Mojo` 


- **Understanding __del__ method in Mojo**: Members discussed how Mojo uses **ASAP** to call destructors when an instance is last used and the lifetime can be manually extended.
- **Mojo and 3D Graphics: OpenGL, Vulkan, and WebGPU**: A member inquired about examples of using Mojo for 3D graphics like **OpenGL**, **Vulkan**, or **WebGPU**.
- **Common libc functions accessible in Mojo**: Members discussed the availability of common **libc functions** in Mojo outside its standard library.
- **Mojo does not currently support cross compilation**: Members asked whether cross compilation is possible with Mojo.



**Link mentioned**: <a href="https://github.com/saviorand/lightbug_http/blob/main/external/libc.mojo">lightbug_http/external/libc.mojo at main · saviorand/lightbug_http</a>: Simple and fast HTTP framework for Mojo! 🔥. Contribute to saviorand/lightbug_http development by creating an account on GitHub.

  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1259023164164542464)** (9 messages🔥): 

> - `Mojo compiler nightly releases`
> - `Changelog updates and PRs`
> - `C pointer semantics removal`
> - `Expression color changes`
> - `Tuple unpacking request` 


- **Mojo Releases Nightly Compiler Updates**: A new nightly Mojo compiler has been released, updating to `2024.7.605` and later to `2024.7.705`, with updates available via `modular update nightly/mojo`.
   - The updates include various changes such as a fallback for the home directory if `HOME` is not set, and the addition of a new `pwd` module following Python syntax.
- **Changelog Updates Crucial for PRs**: Members clarified that any changes, additions, or removals provided by a `PR` should be documented in the `changelog.md`.
   - This ensures clear documentation and tracking of project modifications.
- **Moving Away from C Pointer Semantics**: The removal of the ability to convert integers to pointers is part of moving away from C pointer semantics.
   - *Melodyogonna* found the reason for this change in the changelog.
- **Expression Failure Color: Red or Black?**: A user noted that the color for Expression failure appears black now instead of red, asking if this change was intentional.
   - Another user confirmed that it still appears red on their end.
- **Feature Request: Tuple Unpacking**: Benny.n inquired about the possibility of getting tuple unpacking for non-def functions and aliases.
   - *Alias a, b, c = (1, 2, 3)* would be a very useful feature, according to the user.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1258896353711493250)** (77 messages🔥🔥): 

> - `Optimization of Matmul Algorithm`
> - `SIMD and Cache Performance`
> - `Compile Time Issues with Mojo`
> - `Autotuning for Performance` 


- **Opt for Stack Allocation in Matmul**: A member discussed using stack allocation for temporary storage in a matmul algorithm to improve cache locality and performance, especially in the innermost loop.
   - Their tests showed substantial performance differences, emphasizing the importance of prefetching and cache optimization.
- **Alignment and SIMD Optimization on Graviton 3**: Members confirmed that Graviton 3 has a cache line size of 64 bytes and discussed the alignment requirements for SIMD instructions.
   - One suggested that simdwidth should ideally be a multiple of 256 bytes to avoid performance issues.
- **Handling Small Matrices in Matmul Algorithm**: Optimizations for small matrices were introduced, utilizing simple loops to minimize overhead and improve performance.
- **Compile Times with Mojo Specializations**: A user pointed out long compile times due to multiple specializations for different matrix sizes and data types in Mojo.
   - Suggestions were made to handle compile time values efficiently to avoid performance bottlenecks.
- **Autotuning Prospects for Performance Optimization**: Discussion highlighted the utility of autotuning for optimizing simdwidth and block sizes, which is currently very time-consuming and not portable.
   - Members expressed a wish for autotuning capabilities to return to ease the optimization process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/gabrieldemarmiesse/Mojo-Marathons/tree/output_csv">GitHub - gabrieldemarmiesse/Mojo-Marathons at output_csv</a>: Contribute to gabrieldemarmiesse/Mojo-Marathons development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/2053.">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1258862049170161715)** (67 messages🔥🔥): 

> - `AI-Plans platform`
> - `Using the rerank API`
> - `Cohere community introductions`
> - `Meta Learning by Radek Osmulski`
> - `Dark theme for Coral Chat Interface` 


- **AI-Plans platform for red teaming alignment**: A user mentioned working on **AI-Plans**, a peer review platform designed for red teaming alignment plans.
   - They did not provide additional details or links at this time.
- **Struggles with rerank API deployment**: A member experienced issues with the **rerank API** using a production key, encountering a `TypeError` during deployment despite it working locally.
   - Other users suggested checking the script, particularly the data encoding, and possibly updating the Cohere SDK to resolve discrepancies.
- **New members' introductions and discussions**: New users introduced themselves to the community, expressing excitement about **joining Cohere and exploring its tools**.
   - One user, for example, shared their interest in coworking and using **Aya** for document and ideation workflows.
- **Meta Learning by Radek Osmulski**: A user shared a summary of **Radek Osmulski's Meta Learning** and provided a link to more detailed notes on their blog [here](https://riteshrm.github.io/posts/Meta%20Learning%20By%20Radek%20Osmulski%20Chapter%20Wise%20Summary%20Points/).
   - They described key takeaways including the importance of Stack Overflow, effective use of a code editor, and the value of practical exercises while learning.
- **Suggestions for Coral Chat Interface improvements**: Users suggested multiple enhancements for the **Coral Chat Interface**, such as implementing a dark theme and adding an edit button for messages.
   - One user acknowledged that **Cohere is continuously evolving** and hinted at a forthcoming version 2 with more UI features.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://riteshrm.github.io/posts/Meta%20Learning%20By%20Radek%20Osmulski%20Chapter%20Wise%20Summary%20Points/">Ritesh Kumar Maurya - Meta Learning Book Chapter Wise Summary Points</a>: no description found</li><li><a href="https://youtu.be/gFTLmVsX3ZQ?feature=shared">The Dream Team | Crash Zone - Season 1 Episode 1</a>: A coded message on the Net becomes an irresistible puzzle for Mike, Pi, Bec, Marcello and Ram. Following the clues leads them to a meeting with Alexandra Dav...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1258888358910759006)** (78 messages🔥🔥): 

> - `Rhea Platform`
> - `AI Creation and Interaction`
> - `Organizational Accounts`
> - `User Experience Feedback`
> - `Coding Club Projects with Children` 


- **Rhea launches 'Save to Project' Feature**: The 'Save to Project' feature is now available on [Rhea's platform](https://rhea.run), enabling users to save interactive HTML applications directly from their dashboards.
- **Coding Club Explores AI with Rhea**: A user who runs a children's coding club is excited to integrate AI and HTML projects using Rhea with their students, noting its user-friendly and inspirational platform.
- **Bug in Rhea Signup Process Uncovered**: A user discovered that Rhea's signup process has an email verification issue where email addresses must be entered in lowercase.
- **Rhea Organizational Accounts In Progress**: Rhea is working on supporting organizational accounts, which will allow multiple accounts to share and manage project outputs within a common org, enhancing collaborative work.
- **Powerful AI Features and Tips from Rhea**: Users shared tips on utilizing different AIs like GPT-4 and Claude within Rhea to troubleshoot and enhance code; also discussed hidden commands and upcoming features for a richer user experience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/the-cohere-platform">The Cohere Platform - Cohere Docs</a>: no description found</li><li><a href="https://rhea.run">Rhea | Byte Breeze Studios</a>: no description found
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1259901451879059467)** (1 messages): 

> - `Top-k Sparse Autoencoders`
> - `Llama 3 8B`
> - `Automated Pipeline for SAE Features`
> - `Training SAEs for 70B Model` 


- **Top-k Sparse Autoencoders for Llama 3 8B Released**: The interpretability team released a set of top-k sparse autoencoders for every layer of **Llama 3 8B**, available at [Hugging Face](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x).
   - You can load them using the [sae library](https://github.com/EleutherAI/sae).
- **Automated Pipeline and New Training Efforts**: The team is working on an automated pipeline to explain the **SAE features** and will start training SAEs for the **70B model** shortly.
   - For those interested in helping out, please check out <#1153431135414669422>.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1258885906484559992)** (33 messages🔥): 

> - `PDF markup tools`
> - `Training variable resolution ViT using IJEPA`
> - `Evaluating LlaVa LLM`
> - `Randomness of Copilot`
> - `Determinism in LLM completions` 


- **Struggle to find a PDF markup tool with Search and Markup All function**: A user is searching for a PDF markup tool with a 'Search -> Markup All' function and reports having found only expensive options like Bluebeam and PDF Studio.
- **Training ViT with IJEPA shows promise**: A user is training a variable resolution ViT using IJEPA and achieving about 30% accuracy on ImageNet1k after 20 epochs, sharing their preliminary report [here](https://theadamcolton.github.io/image-ssl-on-a-shoestring).
   - They seek feedback and assistance to refine and speed up their setup.
- **Evaluating LlaVa LLM using lm-evaluation-harness faces issues**: A user reports an error while evaluating LlaVa LLM using lm-evaluation-harness regarding unrecognized configuration class.
   - They are seeking help to resolve this issue.
- **Randomness of Copilot in Name Selection Questioned**: A member raised concerns about Copilot's randomness when selecting 50 names from a list of 120 for a giveaway, questioning whether LLMs are good at being random.
   - Discussions highlighted that LLMs are statistical models and might show deterministic behavior, with some evidence suggesting a narrower set of name completions in finetuned models.
- **Determinism in Copilot Completions**: *philpax* notes that Copilot seems to produce deterministic completions, often generating the same inline suggestions like 'This is a hack, but it works' across projects.
   - Other members discuss that even with temperature settings allowing multiple completions, the inline completions appear consistent and possibly deterministic.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://theadamcolton.github.io/image-ssl-on-a-shoestring">no title found</a>: no description found</li><li><a href="https://x.com/ptrschmdtnlsn/status/1617019805793669125">Tweet from Peter Schmidt-Nielsen (@ptrschmdtnlsn)</a>: Copilot really *really* wants to write the comment &#34;This is a hack, but it works&#34;. It&#39;s sort of disconcerting.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1259063961458835489)** (67 messages🔥🔥): 

> - `T-FREE Tokenizer`
> - `Research on Model Expansion Efficiency`
> - `The BitNet Transformer`
> - `Gradient Conflicts in Diffusion Models`
> - `Quantization in Inference` 


- **T-FREE Tokenizer proposes parameter reduction**: Researchers introduced [T-FREE](https://arxiv.org/abs/2406.19223), a tokenizer embedding words through activation patterns over character triplets, significantly reducing embedding layer size by over 85% while maintaining competitive performance.
- **Debate on model expansion efficiency**: Members discussed the efficiency of model expansion techniques like SOLAR, citing [papers](https://arxiv.org/abs/2310.07999) that show performance gains but often lack comparisons to training models from scratch.
- **BitNet Transformer: A leap for 1-bit models**: [BitNet](https://arxiv.org/abs/2310.11453) introduces a scalable 1-bit weight Transformer architecture, achieving competitive performance while significantly reducing memory footprint and energy consumption.
- **Gradient conflicts slow convergence in diffusion models**: A paper on diffusion models, [Min-SNR-$\gamma$](https://arxiv.org/abs/2303.09556), reveals that slow convergence results from conflicting optimization directions and proposes adapting loss weights based on signal-to-noise ratios to address this, improving convergence speed by 3.4x.
- **Quantization in inference demonstrates practical benefits**: [Recent research](https://arxiv.org/abs/2404.00456) showed the effectiveness of QuaRot for 4-bit quantization on LLMs, achieving near full-precision performance with significantly reduced memory and computational costs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>: Self-attention performs well in long context but has quadratic complexity. Existing RNN layers have linear complexity, but their performance in long context is limited by the expressive power of their...</li><li><a href="https://arxiv.org/abs/2407.03502">AgentInstruct: Toward Generative Teaching with Agentic Flows</a>: Synthetic data is becoming increasingly important for accelerating the development of language models, both large and small. Despite several successful use cases, researchers also raised concerns arou...</li><li><a href="https://arxiv.org/html/2312.15166v2">SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling</a>: no description found</li><li><a href="https://arxiv.org/abs/2212.09720">The case for 4-bit precision: k-bit Inference Scaling Laws</a>: Quantization methods reduce the number of bits required to represent each parameter in a model, trading accuracy for smaller memory footprints and inference latencies. However, the final model size de...</li><li><a href="https://arxiv.org/abs/2407.02783">52B to 1T: Lessons Learned via Tele-FLM Series</a>: Large Language Models (LLMs) represent a significant stride toward Artificial General Intelligence. As scaling laws underscore the potential of increasing model sizes, the academic community has inten...</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: Recent research, such as BitNet, is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58, in which every single param...</li><li><a href="https://arxiv.org/abs/2406.19223">T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings</a>: Tokenizers are crucial for encoding information in Large Language Models, but their development has recently stagnated, and they contain inherent weaknesses. Major limitations include computational ov...</li><li><a href="https://arxiv.org/abs/2310.11453">BitNet: Scaling 1-bit Transformers for Large Language Models</a>: The increasing size of large language models has posed challenges for deployment and raised concerns about environmental impact due to high energy consumption. In this work, we introduce BitNet, a sca...</li><li><a href="https://arxiv.org/abs/2407.02423">On the Anatomy of Attention</a>: We introduce a category-theoretic diagrammatic formalism in order to systematically relate and reason about machine learning models. Our diagrams present architectures intuitively but without loss of ...</li><li><a href="https://arxiv.org/abs/2303.09556">Efficient Diffusion Training via Min-SNR Weighting Strategy</a>: Denoising diffusion models have been a mainstream approach for image generation, however, training these models often suffers from slow convergence. In this paper, we discovered that the slow converge...</li><li><a href="https://arxiv.org/abs/2404.00456">QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs</a>: We introduce QuaRot, a new Quantization scheme based on Rotations, which is able to quantize LLMs end-to-end, including all weights, activations, and KV cache in 4 bits. QuaRot rotates LLMs in a way t...</li><li><a href="https://arxiv.org/abs/1812.10783">Topological Constraints on Homeomorphic Auto-Encoding</a>: When doing representation learning on data that lives on a known non-trivial manifold embedded in high dimensional space, it is natural to desire the encoder to be homeomorphic when restricted to the ...</li><li><a href="https://arxiv.org/abs/2310.07999">LEMON: Lossless model expansion</a>: Scaling of deep neural networks, especially Transformers, is pivotal for their surging performance and has further led to the emergence of sophisticated reasoning capabilities in foundation models. Su...</li><li><a href="https://github.com/martius-lab/hitchhiking-rotations">GitHub - martius-lab/hitchhiking-rotations: Learning with 3D rotations, a hitchhiker’s guide to SO(3) - ICML 2024</a>: Learning with 3D rotations, a hitchhiker’s guide to SO(3) - ICML 2024 - martius-lab/hitchhiking-rotations</li><li><a href="https://github.com/Mooler0410/LLMsPracticalGuide">GitHub - Mooler0410/LLMsPracticalGuide: A curated list of practical guide resources of LLMs (LLMs Tree, Examples, Papers)</a>: A curated list of practical guide resources of LLMs (LLMs Tree, Examples, Papers) - Mooler0410/LLMsPracticalGuide
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1259864689849008199)** (3 messages): 

> - `Attention as Hypernetwork`
> - `Empirical results of Attention as Hypernetwork` 


- **Reformulating Attention as Hypernetwork**: A member shared a [paper](https://arxiv.org/abs/2406.05816) that reformulates Attention as a **Hypernetwork**.
   - *To me, it seems that W_key and W_value make up the hypernetwork.*
- **Dismissal of Attention as Hypernetwork paper**: One member suggested ignoring the [paper](https://arxiv.org/abs/2406.05816) and interpreted the hypernetwork part as the attention scores.
   - Another member agreed with this assessment.



**Link mentioned**: <a href="https://arxiv.org/abs/2406.05816">Attention as a Hypernetwork</a>: Transformers can under some circumstances generalize to novel problem instances whose constituent parts might have been encountered during training but whose compositions have not. What mechanisms und...

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1259571359990943824)** (2 messages): 

> - `Mech Interp Reading List v2`
> - `Opinionated List of Favourite Papers`
> - `Mechanistic Interpretability`
> - `Reading List`
> - `Literature Review` 


- **Highly Opinionated Mech Interp Reading List v2 Released!**: **Neelnanda** announced the release of [v2 of their mechanistic interpretability reading list](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite-1), updating the list with their favourite papers, key takeaways, and critiques.
   - *This is a massively updated version of [a similar list](https://www.alignmentforum.org/posts/SfPrNY45kQaBozwmu/an-extremely-opinionated-annotated-list-of-my-favourite) I made two years ago*.
- **Community Expresses Gratitude for Reading List**: A member thanked Neelnanda for the effort put into creating the new reading list.
   - The list aims to help newcomers to the field navigate the overwhelming amount of mechanistic interpretability papers available.



**Link mentioned**: <a href="https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite-1">An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers v2 — AI Alignment Forum</a>: This post represents my personal hot takes, not the opinions of my team or employer. This is a massively updated version of a similar list I made two…

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1258878175455481897)** (32 messages🔥): 

> - `LlaVa LLM evaluation`
> - `Feature requests in lm-evaluation-harness`
> - `AISI's Inspect vs. lm-eval harness`
> - `Long-context evaluation benchmarks` 


- **LlaVa LLM evaluation struggles**: A member faced a `ValueError` while trying to evaluate **LlaVa LLM** with **lm-evaluation-harness**, as it's a multimodal model not currently supported by the harness.
   - The community suggested using `HFLM._get_model` and pointed out that **lm-evaluation-harness** supports `AutoModelForSeq2SeqLM` and `AutoModelForCausalLM` classes.
- **lm-evaluation-harness feature requests**: A question about excluding default tasks in `lm-evaluation-harness` was raised, and a suggestion was made to add a CLI flag for this option.
   - Members discussed the possibility of using `include_default` flag and detailed fixes for an OOM issue ([GitHub Issue #1923](https://github.com/EleutherAI/lm-evaluation-harness/issues/1923)).
- **AISI's Inspect vs. lm-eval harness**: Inspect AI has a strong UI and well-designed library but lacks battle-tested support for local models compared to **lm-eval harness**.
   - Inspect provides robust support for multiple LM calls, prompt engineering, and frontier API models, whereas **lm-eval harness** focuses on standardization and built-in task logic.
- **Proposing long-context evaluation benchmarks**: A thread was created to discuss long-context evaluations like sliding window PPL and other new tasks, with a suggestion to follow `wikitext` for word_perplexity and byte_perplexity metrics.
   - Community members shared links to potential benchmarks and discussed using metrics like word_perplexity for long-context evaluations ([arXiv paper](https://arxiv.org/abs/2402.13718)).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.13718">$\infty$Bench: Extending Long Context Evaluation Beyond 100K Tokens</a>: Processing and reasoning over long contexts is crucial for many practical applications of Large Language Models (LLMs), such as document comprehension and agent construction. Despite recent strides in...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1923">OOM Issue · Issue #1923 · EleutherAI/lm-evaluation-harness</a>: Hi! I am running evaluations but keep getting OOM errors. Here is my script: TASKS=&quot;mmlu&quot; BATCH_SIZE=1 NUM_SHOTS=5 MODEL=Qwen/Qwen1.5-4B API=vllm lm_eval \ --model ${API} \ --model_args pret...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py#L1329).">lm-evaluation-harness/lm_eval/api/task.py at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py#L59">lm-evaluation-harness/lm_eval/api/model.py at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

wendlerc: Does anyone have a good SDXL latent downscaler? I’d like to go from 128x128x4 to 64x64x4.
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1259707602410016768)** (1 messages): 

> - `Docker container usage`
> - `GPT-Neox deployment`
> - `Kubernetes for large-scale jobs`
> - `Docker Compose vs. Kubernetes` 


- **Questions on GPT-Neox deployment using Docker**: A member asked about the practical usage of the Docker container for GPT-Neox, mentioning some success but questioning its efficacy for larger-scale jobs.
   - They speculated that Kubernetes could be more useful than Docker Compose for such jobs and sought insights on the actual deployment practices from others.
- **Considering Kubernetes over Docker Compose**: A member wondered if Kubernetes might be more beneficial than Docker Compose for running larger-scale jobs with GPT-Neox.
   - They asked if others were actually using Docker containers in practice and if Docker Compose was the preferred platform.


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1258890742009565244)** (41 messages🔥): 

> - `JPEG XL Image Codec`
> - `Kolors GitHub Repository`
> - `Noise Scheduling in Machine Learning`
> - `Meta VLM Ads`
> - `IJEPA Training with Variable Resolution ViT` 


- **JPEG XL Dominates Image Codecs**: Inquiring about the state-of-the-art image codec, a member declares **JPEG XL** as the superior choice currently.
- **Kolors GitHub Repository Highlighted**: A member shared the [Kolors GitHub repository](https://github.com/Kwai-Kolors/Kolors) which contains a paper section they found particularly noteworthy.
   - They mentioned it could cause an *instant stroke* because of its impactful content.
- **Debate over Noise Scheduling in Machine Learning**: Participants debated if adding 100 timesteps is viable, indicating **switching to v-prediction** doesn't require further hacks and can achieve zero terminal SNR for complete noise at the terminal timestep.
   - Citing **SDXL's paper** (Citation 20) for guidance, another noted this technique despite **test-train mismatches** at high-resolution sampling.
- **Meta VLM Ads Criticized**: A member questioned why Meta is running ads for their VLM instead of releasing **Llama3VLM**, suggesting frustration among users.
   - There is skepticism about the availability of an API, fearing it could remain tied to Meta's specific products.
- **IJEPA Training Experiment Shared**: A member shared [preliminary results](https://theadamcolton.github.io/image-ssl-on-a-shoestring) of training a variable resolution ViT using IJEPA, achieving **30% accuracy on Imagenet1k** after 20 epochs.
   - They invited feedback and collaboration to enhance this promising yet resource-efficient model training method.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://theadamcolton.github.io/image-ssl-on-a-shoestring">no title found</a>: no description found</li><li><a href="https://palette.fm/">Colorize Photo | Try Free | Realistic Colors</a>: Colorize your black and white images within seconds. Try our online AI colorize tool for free, no sign-up needed.</li><li><a href="https://github.com/Kwai-Kolors/Kolors">GitHub - Kwai-Kolors/Kolors: Kolors Team</a>: Kolors Team. Contribute to Kwai-Kolors/Kolors development by creating an account on GitHub.</li><li><a href="https://gokaygokay-kolors.hf.space/">Kolors</a>: no description found
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1258988195601711114)** (28 messages🔥): 

> - `VALL-E 2`
> - `Terminator model discussion`
> - `New caption model - CapPa`
> - `VisualKeras tool` 


- **VALL-E 2 achieves human parity in text-to-speech**: **VALL-E 2** is a milestone in zero-shot TTS, introducing Repetition Aware Sampling and Grouped Code Modeling to surpass previous models in robustness and naturalness on the LibriSpeech and VCTK datasets.
   - Despite needing substantial compute resources, it is reproducible with publicly available datasets, and there's hope someone like lucidrains might replicate the code.
- **Debate over Terminator model's validity**: Discussion highlighted concerns about many model studies claiming superiority without proper compute-scale comparisons; **Terminator** was critiqued heavily for high compute demands and lack of scaling law evidence.
   - A call was made for scientifically sound comparisons, checking models across a compute scale span instead of arbitrarily picked benchmarks.
- **CapPa caption model needs JAX**: A new caption model, **CapPa**, dropped and training it using JAX has been showcased [here](https://wandb.ai/craiyon/cappa-jax/reports/CapPa-Training-vision-models-as-captioners--Vmlldzo4NDUyNDUz).
   - The GitHub repository providing details is [visualkeras](https://github.com/borisdayma/clip-jax/blob/main/utils/demo_cappa.ipynb).
- **VisualKeras tool introduction**: A potentially helpful tool called **VisualKeras** was introduced to visualize Keras neural network architectures with customizable styling options.
   - [Check it out on GitHub](https://github.com/paulgavrikov/visualkeras) for both layered and graph-style visualizations suitable for different types of neural networks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/craiyon/cappa-jax/re">craiyon</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://github.com/paulgavrikov/visualkeras">GitHub - paulgavrikov/visualkeras: Visualkeras is a Python package to help visualize Keras (either standalone or included in TensorFlow) neural network architectures. It allows easy styling to fit most needs. This module supports layered style architecture generation which is great for CNNs (Convolutional Neural Networks), and a graph style architecture, which works great for most models including plain feed-forward networks.</a>: Visualkeras is a Python package to help visualize Keras (either standalone or included in TensorFlow) neural network architectures. It allows easy styling to fit most needs. This module supports la...</li><li><a href="https://wandb.ai/craiyon/cappa-jax/reports/CapPa-Training-vision-models-as-captioners--Vmlldzo4NDUyNDUz">CapPa: Training vision models as captioners</a>: Open-source reproduction of &quot;Image Captioners are Scalable Vision Learners Too&quot;. Made by Boris Dayma using Weights &amp; Biases</li><li><a href="https://github.com/borisdayma/clip-jax/blob/main/utils/demo_cappa.ipynb">clip-jax/utils/demo_cappa.ipynb at main · borisdayma/clip-jax</a>: Train vision models using JAX and 🤗 transformers. Contribute to borisdayma/clip-jax development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1258882445437308948)** (52 messages🔥): 

> - `Handling CSV files in LangChain`
> - `LangChain utility functions`
> - `LangGraph setup issues`
> - `Running LLMs locally`
> - `Async configuration in LangChain` 


- ****CSV File Handling in LangChain****: A user was seeking advice on handling CSV files with LangChain, asking for modern approaches to using multiple CSV files and improving from previous limitations.
- ****Async Configuration in LangChain****: A user asked how to use the `ensure_config()` method in an asynchronous environment within LangChain, seeking guidance on getting `thread_id` in a `ToolNode` using `astream_events`.
   - The user received advice to include the `config` parameter in the tool's `invoke` function to extract `thread_id`.
- ****LangGraph ToolNode Errors****: A user reported errors with the `ToolNode` in `create_react_agent` from `langgraph.prebuilt`, causing `NameError: name 'Type' is not defined` and requested help to troubleshoot.
   - The user shared a link to their notebook on GitHub for further investigation.
- ****Running LLMs on Local Machines****: Users discussed their experiences running smaller LLM models like `phi3`, `mistral`, and `llama3` on local PCs with high-end specifications, including NVIDIA RTX 4090 GPUs.
   - Questions were also raised about the feasibility and performance of running larger-scale models, such as 70B parameters, using multiple GPUs.
- ****LangChain Utility Functions****: A user sought help in converting model responses to JSON format within LangChain, and was directed to specific documentation on using `JsonOutputParser` and integrating with `Pydantic`.
   - The user thanked for the guidance and confirmed their issue was resolved.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/">JSON parser | 🦜️🔗 LangChain</a>: This output parser allows users to specify an arbitrary JSON schema and query LLMs for outputs that conform to that schema.</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/self_query/">Self-querying | 🦜️🔗 LangChain</a>: Head to Integrations for documentation on vector stores with built-in support for self-querying.</li><li><a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">Tweet from Prashant Dixit (@Prashant_Dixit0)</a>: ✨Open-sourcing comprehensive LLM Glossary✨  Explore, Learn, and Add terms about #LLMs and #GenAI. Let&#39;s make AI easy for everyone.  🚨Adding new terms on regular basis  Don&#39;t forget to give st...</li><li><a href="https://github.com/Adefioye/Alpha-Agent/blob/main/financial_annual_report/financial_annual_report.ipynb">Alpha-Agent/financial_annual_report/financial_annual_report.ipynb at main · Adefioye/Alpha-Agent</a>: Contribute to Adefioye/Alpha-Agent development by creating an account on GitHub.</li><li><a href="https://langchain-ai.github.io/langgraphjs/tutorials/multi_agent/agent_supervisor">Agent supervisor - LangGraph.js</a>: no description found</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/callbacks_async/#next-steps>).">How to use callbacks in async environments | 🦜️🔗 LangChain</a>: This guide assumes familiarity with the following concepts:</li><li><a href="https://github.com/langchain-ai/langchain/issues/16425>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1259264035187064885)** (1 messages): 

> - `LangServe Deployment Issues`
> - `LangGraph Cloud Announcement` 


- **Confusion on LangServe deployment**: A user expressed confusion about deploying LangServe from LangSmith, mentioning they only receive a message about **LangGraph Cloud** coming soon when attempting a deployment.
   - *Will I have to go with a third party cloud provider if I want to deploy my langserve API?* was a follow-up question.
- **LangGraph Cloud Coming Soon**: Members noticed a message about **LangGraph Cloud** coming soon when attempting to deploy LangServe via LangSmith.
   - This created uncertainty about whether third-party cloud providers would be needed for LangServe deployments.


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1259062994109993050)** (10 messages🔥): 

> - `doesVideoContain`
> - `qdurllm`
> - `OranScribe`
> - `LLM glossary`
> - `advanced research assistant` 


- **Innovative 'doesVideoContain' Tool Makes Waves**: Introducing a new tool, **'doesVideoContain'**, that allows videos to self-scan for specific content, using [WebAI](https://github.com/jasonmayes/doesVideoContain), running entirely in-browser in JS.
   - Check the [YouTube video demo](https://www.youtube.com/watch?v=3FrYr13RL1E) and [live demo on Codepen](https://codepen.io/jasonmayes/pen/eYaqZZo).
- **Launch of 'qdurllm' Blends Qdrant, URLs, and LLMs**: Introducing **qdurllm**: a local search engine that embeds and stores URL contents in a vector database using [LangChain and Sentence Transformers](https://github.com/AstraBert/qdurllm).
   - Allows users to run semantic searches and utilize LLMs like **gemma-2b-it** for enhanced query results, all locally with a Gradio interface.
- **Self-Correcting AI Coding Assistant Released**: Announcing a new self-correcting, self-reviewing python coding assistant combining [Langchain and GPT4-o](https://huggingface.co/spaces/as-cle-bert/self-reviewing-coding-assistant), inspired by **Codium-AI's AlphaCodium**.
   - This assistant is designed to enhance coding workflows by efficiently identifying and resolving issues automatically.
- **AI Agents for LangGraph Now in Beta**: A new tool called **Devin for LangGraph**, designed to turn interviews into AI agents in LangGraph, is looking for beta testers.
   - More details can be found on [Streamlit](https://definitive-ai.streamlit.app/) and [GitHub](https://github.com/Definitive-AI/Agent-Examples), with a private beta currently running.
- **Llamapp: Local RAG for Accurate Responses**: Introducing **Llamapp**, a locally operating Retrieval Augmented Generator that [combines document retrieval and LLM generation](https://github.com/rajatasusual/llamapp) for accurate responses.
   - This tool uses custom retrieval techniques and enforces the LLM to adhere to the source data.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">Tweet from Prashant Dixit (@Prashant_Dixit0)</a>: ✨Open-sourcing comprehensive LLM Glossary✨  Explore, Learn, and Add terms about #LLMs and #GenAI. Let&#39;s make AI easy for everyone.  🚨Adding new terms on regular basis  Don&#39;t forget to give st...</li><li><a href="https://huggingface.co/spaces/as-cle-bert/self-reviewing-coding-assistant">Self Reviewing Coding Assistant - a Hugging Face Space by as-cle-bert</a>: no description found</li><li><a href="https://github.com/rajatasusual/llamapp">GitHub - rajatasusual/llamapp: A Retrieval Augmented Generator (RAG) that operates entirely locally, combining document retrieval and language model generation to provide accurate and contextually relevant responses. Built with @Langchain-ai</a>: A Retrieval Augmented Generator (RAG) that operates entirely locally, combining document retrieval and language model generation to provide accurate and contextually relevant responses. Built with ...</li><li><a href="https://link.medium.com/DLY6pZRE3Kb">no title found</a>: no description found</li><li><a href="https://github.com/AstraBert/qdurllm">GitHub - AstraBert/qdurllm: Search your favorite websites and chat with them, on your desktop🌐</a>: Search your favorite websites and chat with them, on your desktop🌐 - AstraBert/qdurllm</li><li><a href="https://github.com/Haste171/rag-demo">GitHub - Haste171/rag-demo: Basic explanation &amp; walkthrough of RAG</a>: Basic explanation &amp; walkthrough of RAG. Contribute to Haste171/rag-demo development by creating an account on GitHub.</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: no description found</li><li><a href="https://github.com/jasonmayes/doesVideoContain">GitHub - jasonmayes/doesVideoContain</a>: Contribute to jasonmayes/doesVideoContain development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=3FrYr13RL1E">Web AI Demo: Does Video Contain - use AI to enable videos to watch themselves to perform useful work</a>: The initial release of this little utility library allows you to ask the most common question when working with video content - does the video contain someth...</li><li><a href="https://x.com/jason_mayes/status/1809497359812030801">Tweet from Jason Mayes (@jason_mayes)</a>: 💡What if you could answer the most common question when working with video content: Does it contain something you want? I made an MVP, with #WebAI of course, that watches with you to grab images of k...</li><li><a href="https://scribe.oranai.com/">OranScribe</a>: OranScribe is your ultimate AI Writing Flow library, designed to help your business create content using industry-best practices. Streamline your content creation and produce high-performance social m...</li><li><a href="https://definitive-ai.streamlit.app/">no title found</a>: no description found</li><li><a href="https://github.com/Definitive-AI/Agent-Examples">GitHub - Definitive-AI/Agent-Examples: Agent Generator Outputs</a>: Agent Generator Outputs. Contribute to Definitive-AI/Agent-Examples development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1259065615268839444)** (2 messages): 

> - `LangGraph state`
> - `Langchain + Graph RAG + GPT-4o` 


- **Exploring LangGraph state tutorial**: A YouTube video titled ["LangGraph state"](https://youtu.be/DBXdE_5Jces) explains how to use LangGraph with State, representing the current snapshot of the app.
   - *"In this Tutorial, we will use LangGraph with State."*
- **Integrating Langchain with Graph RAG and GPT-4o**: A YouTube video titled ["Langchain + Graph RAG + GPT-4o Python Project"](https://www.youtube.com/watch?v=HPmO1UZwfHc&t=1s) outlines a 4-step process to create an AI/chatbot for your website.
   - *"#coding #rag #llm #ai #graphrag #chatbot 💚 Link to Code: https://www.patreon.com/GaoDalie_AI."*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=HPmO1UZwfHc&t=1s">Langchain + Graph RAG + GPT-4o Python Project: Easy AI/Chat for your Website</a>: #coding #rag #llm #ai #graphrag #chatbot 💚 Link to Code: https://www.patreon.com/GaoDalie_AI in this video, I will walk you through 4 steps to give you a ro...</li><li><a href="https://youtu.be/DBXdE_5Jces">LangGraph state</a>: In this Tutorial, we will use LangGraph with State. The State is a shared data structure and represents the current snapshot of the app.00:45 Join The Skool ...
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1258860087091396708)** (32 messages🔥): 

> - `Skills library with RAG`
> - `Security prioritization by OI team`
> - `GraphRAG`
> - `4th of July house party`
> - `Langchain in RAG system` 


- **Expanding Skills with RAG Delivers Consistency**: A member successfully got a skills library with **RAG** working, which should make certain actions more consistent.
- **OI Team Prioritizes Security Measures**: A member commended the OI team for taking the time to meet on video and discuss security measures, highlighting the team's commitment to making security a significant priority.
- **GraphRAG Introduced for Enhanced Retrieval-Augmented Generation**: A user shared a detailed breakdown and tutorial of **Microsoft's GraphRAG**, which clusters data into communities for better **RAG** use-cases.
- **4th of July House Party Success**: The OI team celebrated their **4th of July** house party with new demos, faces, and a preview of updates, and plans to continue these events every first Thursday.
- **Implementing Langchain with RAG**: Discussions highlighted the use of **Langchain** within RAG systems for various projects, and members showed active interest in exploring its capabilities further.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tedx_ai/status/1808561861589139690">Tweet from Ted Werbel (@tedx_ai)</a>: Just released by @Microsoft is their implementation of GraphRAG - a graph-based retrieval-augmented generation system built in Python. Here&#39;s a simplified breakdown of how it works and why it&#39;...</li><li><a href="https://github.com/MTG/freesound-python">GitHub - MTG/freesound-python: python client for the freesound API</a>: python client for the freesound API. Contribute to MTG/freesound-python development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1258984807317110785)** (10 messages🔥): 

> - `Shipment Timeline`
> - `O1 Talking Capability`
> - `Text Display Options`
> - `Google I/O Demo Glasses`
> - `Linux Module Error` 


- **First 1000 Units Shipping by November**: As of April 30th, the estimated timeline for shipments/fulfillment of the first 1000 units was approximately November this year, though this may have changed since April.
- **O1's Speaking Ability in Question**: A member asked if O1 can talk, with a response indicating it should if configured correctly.
- **Use Glasses as Text Display**: One user suggested that glasses might display text output, potentially functioning like Google's I/O Demo glasses.
   - Another user mentioned the possibility of jailbreaking Meta's Rayban glasses for similar functionality.
- **Linux Module Error 'typer' Solution Sought**: A user running Linux sought help for a 'ModuleNotFoundError: No module named 'typer'' error and mentioned trying `pip install typer` without success.


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1259560590649851986)** (2 messages): 

> - `New model browser UI`
> - `Noromaid Mixtral deprecation` 


- **OpenRouter launches new model browser UI**: OpenRouter introduced a [brand-new model browser UI](https://x.com/OpenRouterAI/status/1810001066240413908) featuring **16 parameter filters**, **category filters**, context length, price, and more.
   - The /models page is now significantly faster, especially on mobile devices, making it easier to explore **180 active language models** processing 74 billion tokens per week.
- **Neversleep's Noromaid Mixtral model deprecated**: Due to decreased usage, the **Noromaid Mixtral model** will be deprecated and will continue to function over the API for the next two weeks before being removed.
   - *Say goodbye to Neversleep's Noromaid Mixtral*, as it will 404 after the set period.



**Link mentioned**: <a href="https://x.com/OpenRouterAI/status/1810001066240413908">Tweet from OpenRouter (@OpenRouterAI)</a>: Announcing a brand-new model marketplace UI ✨  Explore 180 active language models processing 74 billion tokens/week 👇

  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1258991361336545322)** (6 messages): 

> - `Viinyx AI launch`
> - `Text to image API services` 


- **Viinyx AI Launch Boosts Productivity**: **Viinyx AI**, a browser extension, launched to augment the browsing experience by integrating multiple generative AI models like **ChatGPT**, **Anthropic**, and **Gemini** to write and create images anywhere on the web. [Check it out on the Chrome Web Store](https://chromewebstore.google.com/detail/viinyx-ai-assistant-chatg/ochleehcckobncbecepoccjhpjfgepae) and the [official website](https://www.viinyx.com).
- **Seeking Text to Image API Services**: A user asked for recommendations on services providing text-to-image API with different models, similar to **OpenRouter**. **Replicate** was suggested as a possible option, and other mentions included **Novita** and **Fireworks**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chromewebstore.google.com/detail/viinyx-ai-assistant-chatg/ochleehcckobncbecepoccjhpjfgepae">Viinyx - AI Assistant (ChatGPT, GPT-4o, Claude, Gemini)</a>: Powerful all-in-one AI copilot to increase your productivity. Use generative AI (ChatGPT, Claude, Gemini) to write &amp; paint anywhere.</li><li><a href="https://www.viinyx.com">Tweet from Viinyx AI - The Best All-in-one AI browser assistant</a>: Viinyx AI browser extension - Use ChatGPT, Claude, Meta.ai, Microsoft Copilot on any web page. Summarize pages and videos to accelerate your learning. Viinyx AI is BYOK and use your own AI provider br...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1259002342649626736)** (27 messages🔥): 

> - `Crypto payments`
> - `Perplexity models`
> - `Generative video future`
> - `OpenRouter provider options`
> - `Model pricing competition` 


- **Explore multiple crypto options for payments**: Users discussed that **Coinbase Commerce** allows payments in USDC, Matic via Polygon, and other cryptocurrencies.
   - One noted that **Matic** payments worked well.
- **Perplexity models have API limitations**: The **Perplexity API** does not perform as well as its web interface, especially lacking reference links in responses.
   - Alternatives like **Phind** and **direct scraping of GitHub and StackOverflow** might be better for summarizing technical queries.
- **Generative video quality predictions**: A user inquired about the future of **generative video** in terms of **quality, speed, and price** over the next 1-1.5 years.
   - The discussion did not yield concrete predictions, highlighting the speculative nature of such advancements.
- **OpenRouter allows custom providers**: Members confirmed that **OpenRouter** allows users to serve their own finetuned models if they can handle a substantial number of requests.
   - This provides flexibility for developers seeking to integrate custom AI solutions.
- **Price war between DeepInfra and Novita on OpenRouter**: **DeepInfra** and **NovitaAI** are competing for the top slot on OpenRouter for models like **Llama3** and **Mistral** with minuscule price differences.
   - Users joked about them lowering prices by **0.001** to switch ranking spots until very competitive thresholds were reached.


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1258947144123945027)** (6 messages): 

> - `Agentic RAG for Stock Trading`
> - `Toolkits for RAG dataset generation`
> - `Agents as Microservices`
> - `Multi-Document Financial Analyst Agent`
> - `RAG Retrieval Evaluations` 


- **Agentic RAG for Stock Trading 📈🤖**: A tutorial video shows how to build an AI-enabled trading assistant powered by [Llama Index agent/tool/RAG abstractions](https://t.co/ocPaeLphyG).
   - The assistant can perform various tasks for stock trading as demonstrated in the [video tutorial](https://t.co/dcLG3orq0s).
- **Toolkits for RAG Dataset Generation**: Creating an evaluation dataset for RAG is challenging, but [Giskard AI offers a toolkit](https://t.co/rQ7WxplJpF) for generating diverse question sets.
   - This toolkit covers a broader range of questions compared to most automatic dataset generators, as discussed in their [article](https://t.co/sewtQcb9b8).
- **Agents as Microservices**: Llama-agents enable the setup of both agent services and tool services as microservices capable of handling large volumes of requests, as explained in [this post](https://t.co/y9a3PdfW0M).
   - The pattern simplifies the interaction between agents and tools, turning them into scalable microservices.
- **Multi-Document Financial Analyst Agent**: Treating each financial document as a tool, a [Multi-Document Financial Analyst Agent](https://t.co/LJhV838EUM) can be built for analyzing categorized documents, especially 10K reports.
   - Pavan Mantha demonstrates the usage of [Llama Index's features](https://t.co/rOetN1zeNg) to facilitate this agent's analysis.
- **Importance of RAG Retrieval Evaluations**: Retrieval evaluations in RAG may be more critical than LLM evaluations; necessary steps include identifying the right metrics and having a unified dataset representation, as detailed in [this article by Ross A.](https://t.co/7uSgwwWThM).
   - These evaluations can significantly impact the effectiveness and accuracy of the RAG systems, discussed further in [this post](https://t.co/xxj69nneDK).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1258866349866745969)** (21 messages🔥): 

> - `AI application mentorship`
> - `Claude 3 models in Bedrock`
> - `Knowledge graphs from GitHub code`
> - `Structured data queries with LlamaIndex`
> - `ReAct agent observations` 


- **Request for AI application mentorship**: A member requested a mentor or guide to help build an AI application, stating that they only needed guidance while they handle the execution.
   - *pwnosaurusrex* suggested starting with the [5 lines of code](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/) starter example from LlamaIndex's documentation.
- **Claude 3 models now supported in Bedrock**: A question about the support for Claude 3 models in Bedrock was raised.
   - *whitefang_jr* confirmed that Claude 3 models are supported and shared a [GitHub link](https://github.com/run-llama/llama_index/blob/65eb552b13664e713d3cdcf8f432e9696cabc50c/llama-index-integrations/llms/llama-index-llms-bedrock/llama_index/llms/bedrock/utils.py#L47) for reference.
- **Challenges in building knowledge graphs from GitHub code**: A member asked if anyone was building knowledge graphs from GitHub code repositories.
   - They mentioned using a property graph store index for entity extraction and embeddings creation but faced challenges with the results using a custom retriever.
- **Seeking better ways to query structured data with LlamaIndex**: A member expressed difficulty in querying structured data (SQL) across multiple tables and shared a link to LlamaIndex documentation.
   - They also mentioned looking into [Vanna](https://github.com/vanna-ai/vanna) for potential solutions.
- **Accessing ReAct agent's intermediate steps through response object**: Someone inquired about accessing the observations, thoughts, actions, and steps of the ReAct agent via the response object.
   - *cheesyfishes* replied that it's possible through the lower-level API and shared a [Google Colab link](https://t.co/YEGfTOkAkY).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">Tweet from Prashant Dixit (@Prashant_Dixit0)</a>: ✨Open-sourcing comprehensive LLM Glossary✨  Explore, Learn, and Add terms about #LLMs and #GenAI. Let&#39;s make AI easy for everyone.  🚨Adding new terms on regular basis  Don&#39;t forget to give st...</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial (OpenAI) - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/65eb552b13664e713d3cdcf8f432e9696cabc50c/llama-index-integrations/llms/llama-index-llms-bedrock/llama_index/llms/bedrock/utils.py#L47">llama_index/llama-index-integrations/llms/llama-index-llms-bedrock/llama_index/llms/bedrock/utils.py at 65eb552b13664e713d3cdcf8f432e9696cabc50c · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://t.co/YEGfTOkAkY">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1259197885099282633)** (7 messages): 

> - `team red's drivers`
> - `Instinct cards`
> - `custom grads API`
> - `tinygrad functions`
> - `Monday team meeting` 


- **Instinct cards confidence questioned**: A member questioned the **confidence level in team red's drivers** making Instinct cards worth buying, expressing hesitation in purchasing cheap used Mi100s until there's better support.
   - *Another member noted that only the 7900xtx cards are being tested and that going with instinct cards would mean being on one's own.*
- **Proposal for custom grads API**: A user suggested the implementation of a better **API for custom grads**, similar to **jax.customvjp**, to make operations with tensors easier, especially for quantization training.
   - They offered to work on this improvement and argued that the current syntax in tinygrad.functions is not ideal as it operates with **lazybuffers** instead of tensors.
- **Upcoming Monday team meeting agenda**: The Monday meeting at **9:40 a.m. PT** includes topics like tinybox update, feedback from tinybox owners, and discussions on [new memory scheduler](https://github.com/tinygrad/tinygrad/pull/5278), llvm nan fix, `UOps.VECTORIZE`, bug fixes, and new APIs.
   - Additional discussion points are **sharded llama**, **sin/exp/log approximation**, **mlperf**, and **other bounties** such as **std mean one kernel**, **Qualcomm runtime**, **Apple AMX**, and **clang mmap runtime**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1258916976466858127)** (20 messages🔥): 

> - `requires_grad behavior`
> - `multi-GPU training docs`
> - `tensor comparison method`
> - `Adam optimizer issue`
> - `new methods for Tinygrad tensors` 


- **Clarifying requires_grad default behavior**: Discussion on why `requires_grad` in `tensor.py` can be **None**, **False**, or **True**. None is the default and gets updated to **True** if the tensor is put in an optimizer.
- **Intro to multi-GPU training in Tinygrad**: For multi-GPU training, users can refer to the [beautiful_mnist_multigpu.py example](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_mnist_multigpu.py). The model can be copied using `shard(axis=None)`, and data can be split using `shard(axis=0)`.
- **Comparing tensors in Tinygrad made easy**: Users inquired about equivalent methods to `torch.all` for tensor comparison in Tinygrad. It was suggested to compare tensors using `(t1 == t2).min() == 1`, and **Tensor.all** was later added to match Torch methods in [this commit](https://github.com/tinygrad/tinygrad/commit/6856f915d6f0e10d41e8e11c8976024989d90aa7).
- **Adam optimizer causing NaNs**: A member reported that weights turn to **NaN** after the second step when using Adam optimizer, while it works fine with SGD.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1259389623155691531)** (8 messages🔥): 

> - `Model Merging`
> - `MInference`
> - `RAM Issue`
> - `Offload Config` 


- **Model Merging Troubles**: A member asked another if they are still trying to merge their model.
   - Another member inquired about the tools being used for the merge.
- **Introducing MInference by Microsoft**: A member shared a [GitHub link](https://github.com/microsoft/MInference) to Microsoft's **MInference** project, which speeds up Long-context LLMs' inference and reduces latency by up to 10x.
   - The tool employs approximate and dynamic sparse calculations to maintain accuracy while improving pre-filling performance on an **A100**.
- **RAM Issues During Model Merging**: Following an inquiry about running out of RAM, another user confirmed the issue.
   - The problem was resolved by **specifying CPU** for the process.



**Link mentioned**: <a href="https://github.com/microsoft/MInference">GitHub - microsoft/MInference: To speed up Long-context LLMs&#39; inference, approximate and dynamic sparse calculate the attention, which reduces inference latency by up to 10x for pre-filling on an A100 while maintaining accuracy.</a>: To speed up Long-context LLMs&amp;#39; inference, approximate and dynamic sparse calculate the attention, which reduces inference latency by up to 10x for pre-filling on an A100 while maintaining accu...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1259513609596309524)** (1 messages): 

> - `Yi-1.5-9B-Chat training`
> - `Hermes-2.5 integration`
> - `Benchmark results`
> - `Future plans for extended context length` 


- **Yi-1.5-9B-Chat fine-tuned on OpenHermes-2.5**: A member shared that they trained **Yi-1.5-9B-Chat** on **OpenHermes-2.5** and are pleased with the results, offering [GGUF versions and common quantizations](https://huggingface.co/juvi21/Hermes-2.5-Yi-1.5-9B-Chat) for trial.
   - The model now appears smarter and more 'aware' in specific situations, citing a notable improvement on the **AGIEval Benchmark** for its class.
- **Fine-tuning details of Hermes-2.5-Yi-1.5-9B-Chat**: The fine-tuned model is a version of [01-ai/Yi-1.5-9B-Chat](https://huggingface.co/01-ai/Yi-1.5-9B-Chat) using the [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) dataset, trained on 4 NVIDIA A100 40GB GPUs for 48:32:13 hours.
   - The model's **sequence length** is 8192 tokens, and it is trained with the **chat-template**: chatml.
- **Future improvements with POSE**: There are plans to extend the model's context length to **32k tokens** using **POSE**.
   - This enhancement aims to improve the model's performance in handling more extended context scenarios.



**Link mentioned**: <a href="https://huggingface.co/juvi21/Hermes-2.5-Yi-1.5-9B-Chat">juvi21/Hermes-2.5-Yi-1.5-9B-Chat · Hugging Face</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1259560193583218910)** (5 messages): 

> - `chat_template`
> - `mistral finetuning in axolotl` 


- **Query on Mistral Finetuning Chat Template**: A member asked which **chat_template** should be used for **Mistral finetuning** in axolotl.
   - Another member responded that it depends on the dataset structure.
- **Configuring Chat Template in YAML**: A suggestion was made to use the `"chatml"` chat template for Mistral finetuning in Axolotl.
   - An example configuration was provided using the `"chatml"` template in the YAML format.



**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=be14de0a-8a0e-4075-90ea-a6fac1a0008b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1259150359318888488)** (8 messages🔥): 

> - `MLOps implementation`
> - `Distributed VLLM inference`
> - `FP8 quantization issues`
> - `Chat template challenges` 


- **Strategic Insights for MLOps Implementation**: A user shared a [blog post](https://nik-hil.hashnode.dev/diving-deep-essential-questions-for-building-your-mlops-pipeline) exploring key questions in building an MLOps pipeline, emphasizing the importance of understanding MLOps fundamentals and high-quality data.
   - The post aims to guide companies through crucial considerations for successful MLOps deployment, to improve model accuracy and reduce operational costs.
- **Issues with Distributed VLLM Inference using FP8**: A user requested help with distributed vllm inference on an fp8 quantized Llama 3 70B model using 8xL40S GPUs, facing performance drops and incorrect outputs.
   - Following debugging, the issue was identified as related to the sensitivity of autofp8 to padding tokens and mishandling of chat templates, which was later resolved.
- **Neural Magic FP8 Quantization**: The user attempted fp8 quantization with code similar to an [example from Neural Magic](https://github.com/neuralmagic/AutoFP8/blob/147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a/example_dataset.py), and faced issues with the inference setup.
   - It was identified that the FlashAttention-2 backend doesn't support fp8 KV cache, likely contributing to the performance issues.
- **Resolution of FP8 Quantization and Chat Template Issues**: Upon further investigation, the user discovered that autofp8's sensitivity to the padding token and misapplication of chat templates were the root causes of the problem.
   - Adjustments and rewriting parts of the code eventually resolved the issues, leading to correct inference operations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://nik-hil.hashnode.dev/diving-deep-essential-questions-for-building-your-mlops-pipeline">Essential Questions for Your MLOps Pipeline</a>: Guide to building a robust MLOps pipeline by addressing key questions on data, model development, deployment, monitoring, tools, and governance</li><li><a href="https://github.com/neuralmagic/AutoFP8/blob/147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a/example_dataset.py">AutoFP8/example_dataset.py at 147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a · neuralmagic/AutoFP8</a>: Contribute to neuralmagic/AutoFP8 development by creating an account on GitHub.</li><li><a href="https://github.com/vllm-project/vllm/issues/6179">[Usage]: Struggling to get fp8 inference working correctly on 8xL40s · Issue #6179 · vllm-project/vllm</a>: Your current environment Collecting environment information... PyTorch version: 2.3.0+cu121 Is debug build: False CUDA used to build PyTorch: 12.1 ROCM used to build PyTorch: N/A OS: Ubuntu 22.04.4...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1259183509965115392)** (1 messages): 

> - `Replicate billing setup issues` 


- **Replicate credits not added after billing setup**: A member expressed concern that **Replicate credits** were not added after setting up billing.
   - *Sorry for too late*, they mentioned, suggesting a possible delay or misconfiguration.
- **Concerns over billing setup timing**: Another point raised was whether the timing of billing setup affects the allocation of credits.
   - The member did not see credits for replicate today, implying **timing issues** might be at play.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1259271877252092006)** (1 messages): 

> - `Transformers & Torch`
> - `Integrating with OpenAI/Anthropic models` 


- **Exploring Transformers & Torch Alternatives**: A member is currently experimenting with **Transformers** and **Torch** to evaluate their potential effectiveness for their project.
- **Integration Considerations: OpenAI/Anthropic**: Another alternative being considered is integrating with models from **OpenAI** and **Anthropic**.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1259445018905542707)** (1 messages): 

> - `Credit Claims Closed`
> - `Credit Eligibility` 


- **Credit Claims Closed Permanently**: A message clarified that all forms to claim credits are closed, and **no one is eligible for new credits** anymore.
- **Credit Eligibility Update**: The update indicates a permanent closure of credit claims, and this applies to all users without exceptions.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/)** (1 messages): 

4.8.15.16.23.42_: the first 25 credits are available for all but only for 1 month 🙂
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1258882468967088220)** (2 messages): 

> - `Interconnects Bot Feedback` 


- **Interconnects Bot: Minor Feedback**: A user noted that the Interconnects bot's performance was satisfactory but suggested that there hasn't been much change in its recent summaries.
- **Possible Improvements for the Interconnects Bot**: A follow-up message from the same user indicated a desire for more significant updates or improvements in the Interconnects bot's functionality.


  

---


### **Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1258909623163949197)** (8 messages🔥): 

> - `RAG discussions`
> - `Enterprises and RAG`
> - `RAG use cases`
> - `early AI boom`
> - `retrieval and cost efficiency` 


- ****Debate on RAG****: Members discussed **RAG** and its perceived utility for enterprises, with some suggesting it is often talked about by those not working with enterprises.
   - Another member noted that while **RAG** can help enterprises leverage their internal knowledge base, use cases are still evolving.
- ****Early AI Boom Hype****: There were remarks about the initial **hype** around **RAG** during the early AI boom.
   - *People were ridiculous about it back then* was a sentiment shared.
- ****Retrieval and Cost Efficiency in Enterprises****: A member highlighted that while not all enterprises might be using RAG, it could enable cost-efficient models and new use cases.
   - Another user noted that harnessing internal knowledge bases is a technology choice that enterprises understand and want.


  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1259210089537994833)** (6 messages): 

> - `Buzz excitement`
> - `FPGA meeting`
> - `Calendly scheduling` 


- **Buzz is awesome, says member**: A member expressed their enthusiasm for Buzz, followed by Autometa hinting at another interesting release coming soon.
- **Autometa schedules FPGA meeting**: Autometa requested to schedule a meeting to discuss FPGA topics and mentioned having several interesting points to cover.
- **Open Calendly scheduling for Alignment Lab**: Autometa shared an [open Calendly link](https://calendly.com/alignmentlab/meeting) for scheduling discussions, welcoming anyone interested to set up a meeting.



**Link mentioned**: <a href="https://calendly.com/alignmentlab/meeting">meeting - Auto Meta</a>: no description found

  

---



### **LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/)** (1 messages): 

jeffreyw128: wow flash 1.5 is actually so good
  

---



### **AI Stack Devs (Yoko Li) ▷ #[assets](https://discord.com/channels/1122748573000409160/1176906086368935966/1259654236917202995)** (1 messages): 

> - `Google image searches for sprites`
> - `Purchased assets for tilesets` 


- **Sprites sourced from Google image searches**: A member mentioned that all the sprites were obtained through random Google image searches.
- **Only tilesets are purchased assets**: The discussion emphasized that the only purchased assets were tilesets, not the sprites.


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/)** (1 messages): 

jonononono: Anyone going to europython? Doing a talk on vectorization 👀
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1259341165237502035)** (1 messages): 

> - `Gemma 2 9B`
> - `Small Language Models (SLMs)`
> - `Serverless AI inference` 


- **Google's Gemma 2 9B Impresses**: Google's [Gemma 2 9B](https://blog.google/technology/developers/google-gemma-2/?ref=unremarkable.ai) is a recently released open-source language model that has garnered significant attention for its performance and capabilities.
   - [Despite its small size](https://www.reddit.com/r/LocalLLaMA/comments/1drxhlh/gemma_2_9b_appreciation_post/?ref=unremarkable.ai), Gemma 2 9B is comparable or even superior to larger models like GPT-3.5, making it suitable for deployment in resource-constrained environments.
- **Serverless AI Inference with Gemma 2 on AWS Lambda**: A tutorial on [Serverless AI inference](https://www.unremarkable.ai/serverless-ai-inference-with-gemma-2-using-mozillas-llamafile-on-aws-lambda) using Gemma 2 and Mozilla's Llamafile on AWS Lambda has been shared.
   - This approach facilitates deploying Gemma 2 9B in low-resource environments like phones, PCs, or on-premises clouds.



**Link mentioned**: <a href="https://www.unremarkable.ai/serverless-ai-inference-with-gemma-2-using-mozillas-llamafile-on-aws-lambda/">Serverless AI Inference with Gemma 2 using Mozilla&#x27;s llamafile on AWS Lambda</a>: Google&#x27;s Gemma 2 9B is a recently released open-source language model that has garnered significant attention in our community. This lightweight model, is part of the Gemma family of models devel...

  

---



### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1259476099763798038)** (1 messages): 

> - `Experiment with base models`
> - `Hermes-2-Theta-Llama-3-70B`
> - `Llama3-DiscoLeo-Instruct-70B` 


- **Hermes-2-Theta-Llama-3-70B as a base for Llama3-DiscoLeo-Instruct**: A member suggested an interesting experiment to use **Hermes-2-Theta-Llama-3-70B** as the base model for creating **Llama3-DiscoLeo-Instruct-70B**.
- **Potential benefits of combined models**: The discussion implied potential **benefits** of combining models like **Hermes-2-Theta-Llama-3-70B** with **Llama3-DiscoLeo-Instruct** for enhanced performance and capabilities.


  

---



---



{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

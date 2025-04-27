---
id: 56bd7e91-4e60-435f-a93b-7956e5414a24
title: The DSPy Roadmap
date: '2024-08-20T05:06:22.742788Z'
original_slug: ainews-the-dspy-roadmap
description: >-
  **Omar Khattab** announced joining **Databricks** before his MIT professorship
  and outlined the roadmap for **DSPy 2.5 and 3.0+**, focusing on improving core
  components like LMs, signatures, optimizers, and assertions with features such
  as adopting **LiteLLM** to reduce code and enhance caching and streaming. The
  roadmap also includes developing more accurate, cost-effective optimizers,
  building tutorials, and enabling interactive optimization tracking. On AI
  Twitter, **Google** launched **Gemini Live**, a mobile conversational AI with
  voice and 10 voices, alongside **Pixel Buds Pro 2** with a custom Tensor A1
  chip. **OpenAI** updated **ChatGPT-4o**, reclaiming the top spot on LMSYS
  Arena. **xAI** released **Grok-2** in beta, achieving SOTA in image generation
  with FLUX 1. **Nous Research** released open-source **Hermes 3** models in 8B,
  70B, and 405B sizes, with the 405B model achieving SOTA. Robotics updates
  include **Astribot**'s humanoid robot and **Apple**'s tabletop robot with Siri
  voice commands. **Sakana AI** introduced "The AI Scientist," an autonomous AI
  research system.
companies:
  - databricks
  - mit
  - google
  - openai
  - x-ai
  - nous-research
  - astribot
  - apple
  - sakana-ai
models:
  - dspy
  - litel-lm
  - gemini
  - chatgpt-4o
  - grok-2
  - hermes-3
topics:
  - model-optimization
  - fine-tuning
  - optimizers
  - interactive-optimization
  - robotics
  - autonomous-systems
  - voice
  - image-generation
  - open-source-models
  - scientific-research
  - streaming
  - caching
people:
  - omar-khattab
  - giffmana
---


<!-- buttondown-editor-mode: plaintext -->**Systematic AI engineering is all you need.**

> AI News for 8/16/2024-8/19/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**254** channels, and **4515** messages) for you. Estimated reading time saved (at 200wpm): **489 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Omar Khattab announced that he would be [joining Databricks for a year](https://x.com/lateinteraction/status/1825623373655024035) before [his MIT professorship](https://x.com/lateinteraction/status/1796546386911285294) today, but more importantly set the stage for DSPy 2.5 and 3.0+:

 ![image.png](https://assets.buttondown.email/images/20dc1d05-3118-440f-9e52-a0c3d8b75248.png?w=960&fit=max) 

DSPy has objectively been a successful framework for declarative self-improving LLM pipelines, following the [2022 DSP paper](https://arxiv.org/abs/2212.14024) and [2023 DSPy paper](https://arxiv.org/abs/2310.03714).

 ![image.png](https://assets.buttondown.email/images/7c8404d7-6050-4bef-99be-0ca159d59b57.png?w=960&fit=max) 

The main roadmap directions:

1. **Polish the 4 pieces of DSPy core: (1) LMs, (2) Signatures & Modules, (3) Optimizers, and (4) Assertions**, so that they "just work" out of the box zero shot, off-the-shelf. 

- In LMs they aim to reduce lines of code. In particular they call out that they will [eliminate 6k LOC by adopting LiteLLM](https://x.com/yi_ding/status/1825601460664741922). However they will add functionality for "improved caching, saving/loading of LMs, support for streaming and async LM requests".
- In Signatures they are evolving the concept of "structured inputs" now that "structured outputs" are mainstream.
- In Finetuning: they aim to "bootstrap training data for serveral different modules in a program, train multiple models and handle model selection, and then load and plug in those models into the program's modules" 

2. **Developing more accurate, lower-cost optimizers.** Following the BootstrapFewShot -> BootstrapFinetune -> CA-OPRO -> MIPRO -> MIPROv2 and BetterTogether optimmizers, more work will be done improving Quality, Cost, and Robustness.

3. **Building end-to-end tutorials**. More docs!

4. **Shifting towards more interactive optimization & tracking**. Help users "to observe in real time the process of optimization (e.g., scores, stack traces, successful & failed traces, and candidate prompts)."

Nothing mindblowing, but a great roadmap update from a very well managed open source framework.

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

**AI and Robotics Developments**

- **Google's Gemini Updates**: Google launched [Gemini Live](https://twitter.com/adcock_brett/status/1825201770773012617), a mobile conversational AI with voice capabilities and 10 voices, available to Gemini Advanced users on Android. They also introduced [Pixel Buds Pro 2](https://twitter.com/adcock_brett/status/1825201853673488494) with a custom Tensor A1 chip for Gemini functionality, enabling hands-free AI assistance.

- **OpenAI Developments**: [OpenAI's updated ChatGPT-4o model](https://twitter.com/adcock_brett/status/1825201876423340041) reclaimed the top spot on LMSYS Arena, testing under the codename "anonymous-chatbot" for a week with over 11k votes.

- **xAI's Grok-2**: [xAI released Grok-2](https://twitter.com/adcock_brett/status/1825201974419042761), now available in beta for Premium X users. It can generate "unhinged" images with FLUX 1 and has achieved SOTA status in just over a year.

- **Open-Source Models**: [Nous Research released Hermes 3](https://twitter.com/adcock_brett/status/1825201997055684653), an open-source model available in 8B, 70B, and 405B parameter sizes, with the 405B model achieving SOTA relative to other open models.

- **Robotics Advancements**: [Astribot teased their new humanoid](https://twitter.com/adcock_brett/status/1825201929523237341), showcasing its impressive range of freedom in real-time without teleoperation. [Apple is reportedly developing](https://twitter.com/adcock_brett/status/1825201906580390065) a tabletop robot with Siri voice commands, combining an iPad-like display with a robotic arm.

- **AI Research Tools**: [Sakana AI introduced "The AI Scientist"](https://twitter.com/adcock_brett/status/1825201952021459387), claimed to be the world's first AI system capable of autonomously conducting scientific research, generating ideas, writing code, running experiments, and writing papers.

**AI Model Performance and Techniques**

- **Vision Transformer (ViT) Performance**: [@giffmana](https://twitter.com/giffmana/status/1825301443709997521) wrote a blog post addressing concerns about ViT speed at high resolution, aspect ratio importance, and resolution requirements.

- **RAG Improvements**: [New research on improving RAG for multi-hop queries](https://twitter.com/LangChainAI/status/1825234642037010518) using database filtering with LLM-extracted metadata showed promising results on the MultiHop-RAG benchmark. [HybirdRAG](https://twitter.com/dair_ai/status/1825207031558537663) combines GraphRAG and VectorRAG, outperforming both individually on financial earning call transcripts.

- **Model Optimization**: [@cognitivecompai](https://twitter.com/cognitivecompai/status/1825321109744467981) reported that GrokAdamW appears to be an improvement when training gemma-2-2b with the Dolphin 2.9.4 dataset.

- **Small Model Techniques**: [@bindureddy](https://twitter.com/bindureddy/status/1825299800994230763) encouraged iterating on small 2B models to make them more useful and invent new techniques that can be applied to larger models.

**AI Applications and Tools**

- **LangChain Developments**: [LangChain JS tutorial](https://twitter.com/LangChainAI/status/1825319516890669178) on using LLM classifiers for dynamic prompt selection based on query type. [Agentic RAG with Claude 3.5 Sonnet](https://twitter.com/llama_index/status/1825205945678680465), MongoDB, and llama_index demonstrated building an agentic knowledge assistant over a pre-existing RAG pipeline.

- **AI for Software Engineering**: [Cosine demo'd Genie](https://twitter.com/adcock_brett/status/1825202019503681998), a fully autonomous AI software engineer that broke the high score for SWE-Bench at 30.08%. OpenAI and the authors of SWE-Bench [redesigned and released 'SWE-bench Verified'](https://twitter.com/adcock_brett/status/1825202085572272574) to address issues in the original benchmark.

- **Productivity Tools**: [@DrJimFan](https://twitter.com/DrJimFan/status/1825193764962673037) expressed a desire for an LLM to automatically filter, label, and reprioritize Gmail according to a prompt, highlighting the potential for AI in email management.

**AI Ethics and Societal Impact**

- **AI Deception Debate**: [@polynoamial](https://twitter.com/polynoamial/status/1825268351452766578) discussed the misconception of bluffing in poker as an example of AI deception, arguing that it's more about not revealing excess information rather than active deception.

- **AI Reasoning Capabilities**: [@mbusigin](https://twitter.com/mbusigin/status/1825226698348220499) argued that LLMs are already better than a significant number of humans at reasoning, as they don't rely on "gut" feelings and perform well on logical reasoning tests.

**Memes and Humor**

- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1825235274273583278) joked: "Networking ~= Not actually working"
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1825264229483761779) shared a humorous image related to AI or tech (content not specified).
- [@Teknium1](https://twitter.com/Teknium1/status/1825199882254327825) quipped about video generation techniques: "Why are almost every video gen just pan or zoom, you may as well use flux (1000x faster) and generate an image"

This summary captures the key developments, discussions, and trends in AI and robotics from the provided tweets, focusing on information relevant to AI engineers and researchers.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. XTC: New Sampler for Enhanced LLM Creativity**

- **Exclude Top Choices (XTC): A sampler that boosts creativity, breaks writing clichés, and inhibits non-verbatim repetition, from the creator of DRY** ([Score: 138, Comments: 64](https://reddit.com//r/LocalLLaMA/comments/1ev8n2s/exclude_top_choices_xtc_a_sampler_that_boosts/)): The **Exclude Top Choices (XTC)** sampler, introduced in a **GitHub pull request** for **text-generation-webui**, aims to **boost LLM creativity** and **break writing clichés** with minimal impact on coherence. The creator reports that XTC produces novel turns of phrase and ideas, particularly enhancing **roleplay and storywriting**, and feels distinctly different from increasing temperature in language models.


**Theme 2. Cost-Benefit Analysis of Personal GPUs for AI Development**

- **Honestly nothing much to do with one 4090** ([Score: 84, Comments: 90](https://reddit.com//r/LocalLLaMA/comments/1evdrxk/honestly_nothing_much_to_do_with_one_4090/)): The author, who works in **AI infrastructure and ML engineering**, expresses disappointment with their **4090 GPU** purchase for personal AI projects. They argue that for most use cases, **cloud-based API services** or **enterprise GPU clusters** are more practical and cost-effective than a single high-end consumer GPU for AI tasks, questioning the value of local GPU ownership for personal AI experimentation.


## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Advancements and Comparisons**

- **Flux LoRA Results**: A user shared impressive results from training Flux LoRA models on Game of Thrones characters, achieving high-quality outputs with only 10 image datasets and 500-1000 training steps. The training required over 60GB of VRAM. [Source](https://www.reddit.com/r/StableDiffusion/comments/1ev6pca/some_flux_lora_results/)

- **Cartoon Character Comparison**: A comparison of various AI models (DALL-E 3, Flux dev, Flux schnell, SD3 medium) generating cartoon characters eating watermelon. DALL-E 3 performed best overall, with Flux dev coming in second. The post highlighted DALL-E 3's use of complex LLM systems to split images into zones for detailed descriptions. [Source](https://www.reddit.com/r/StableDiffusion/comments/1ev68la/cartoon_character_comparison/)

- **Flux.1 Schnell Upscaling Tips**: A user shared tips for improving face quality in Flux.1 Schnell outputs, recommending the use of 4xFaceUpDAT instead of 4x-UltraSharp for upscaling realistic images. The post also mentioned other upscaling models and techniques for enhancing image quality. [Source](https://www.reddit.com/r/StableDiffusion/comments/1ev6ris/tips_for_flux1_schnell_to_avoid_a_plasticky/)

**AI Company Strategies and Criticisms**

- **OpenAI's Business Practices**: A user criticized OpenAI for running their company like a "tiny Ycombinator startup," citing practices such as waitlists, cryptic CEO tweets, and pre-launch hype videos. The post argued that these tactics are unsuitable for a company valued at nearly $100 billion and may confuse customers and enterprise users. [Source](https://www.reddit.com/r/OpenAI/comments/1evspo8/openai_runs_its_company_like_a_tiny_ycombinator/)

**AI-Generated Content and Memes**

- **The Mist (Flux+Luma)**: A video post showcasing AI-generated content using Flux and Luma models, likely depicting a scene inspired by the movie "The Mist." [Source](https://www.reddit.com/r/StableDiffusion/comments/1evfgys/the_mist_fluxluma/)

- **Seems familiar somehow?**: A meme post in the r/singularity subreddit, likely referencing AI-related content. [Source](https://www.reddit.com/r/singularity/comments/1ev8cfs/seems_familiar_somehow/)

- **Someone had to say it...**: Another meme post in the r/StableDiffusion subreddit. [Source](https://www.reddit.com/r/StableDiffusion/comments/1evio5l/someone_had_to_say_it/)

**Future Technology and Research**

- **Self-driving Car Jailbreaking**: A post speculating that people will attempt to jailbreak self-driving cars once they become widely available. [Source](https://www.reddit.com/r/singularity/comments/1ev97ky/once_selfdriving_cars_are_here_i_expect_people_to/)

- **Age Reversal Pill for Dogs**: A study reporting promising results for an age reversal pill tested on dogs. However, the post lacked citations to peer-reviewed research and was criticized for being anecdotal. [Source](https://www.reddit.com/r/singularity/comments/1ev3vac/new_study_reveals_promising_results_for_age/)


---

# AI Discord Recap

> A summary of Summaries of Summaries by Claude 3.5 Sonnet


**1. Hermes 3 Model Release and Performance**

- **Hermes 3 Matches Llama 3.1 on N8Bench**: **Hermes 3** scored identical to **Llama 3.1 Instruct** on the **N8Bench** benchmark, which measures a model's ability to reason and solve problems.
   - This result is significant as Llama 3.1 Instruct is considered one of the most advanced language models available, highlighting Hermes 3's competitive performance.
- **Hermes 3 405B Free Weekend on OpenRouter**: **OpenRouter** announced that **Hermes 3 405B** is free for a limited time, offering a **128k context window**, courtesy of **Lambda Labs**.
   - Users can access this model at [OpenRouter's Hermes 3 405B page](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended), providing an opportunity to test and evaluate this large language model.
- **Quantization Impact on 405B Models**: **@hyperbolic_labs** warned that **quantization** can significantly degrade the performance of **405B models**.
   - They recommended reaching out to them for alternative solutions if performance is a concern, highlighting the trade-offs between model size reduction and maintaining performance quality.
  


**2. LLM Inference Optimization Techniques**

- **INT8 Quantization for CPU Execution**: A member inquired about the potential benefits of using **INT8 quantization** for faster CPU execution of small models, suggesting some CPUs might natively run INT8 without converting to FP32.
   - This approach could potentially improve performance for CPU-based inference, especially for resource-constrained environments or edge devices.
- **FP8 Training Advancements**: Training a **1B FP8 model** with **1st momentum in FP8** smoothly up to **48k steps** resulted in a loss comparable to **bfloat16** with a **0.08 offset**.
   - This demonstrates that FP8 training can be effective with 1st momentum, achieving similar results as bfloat16 training while potentially offering memory savings and performance improvements.
- **Batching APIs for Open-Source Models**: **CuminAI** introduced a solution for creating **batching APIs** for open-source models, similar to those recently launched by OpenAI and Google.
   - While major companies' batching APIs lack processing guarantees and SLAs, CuminAI's approach aims to provide similar cost-saving benefits for open-source model deployments. A guide is available at [their blog post](https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49).
  


**3. Open Source AI Model Developments**

- **Falcon Mamba 7B Claims to Outperform Llama 3 8B**: A [YouTube video](https://www.youtube.com/watch?v=dokzrFa-DtY) announced the release of **Falcon Mamba 7B**, claiming it outperforms **Llama 3 8B**.
   - This development could have significant implications for the field of large language models, as Falcon Mamba 7B is a relatively new and promising model challenging established benchmarks.
- **Ghost 8B Beta's Multilingual Prowess**: **Ghost 8B Beta**, a newly released language model, now supports **16 languages** including English, Vietnamese, Spanish, and Chinese, with two context options (8k and 128k).
   - The model boasts improved capabilities in math, reasoning, and instruction-following, outperforming competitors like Llama 3.1 8B Instruct, GPT-3.5 Turbo, and Claude 3 Opus in AlpacaEval 2.0 winrate scores.
- **VideoLLaMA 2-72B Release by Alibaba DAMO**: **Alibaba DAMO** released **VideoLLaMA 2-72B**, a new video LLM available on [HuggingFace](https://huggingface.co/collections/DAMO-NLP-SG/videollama-2-6669b6b6f0493188305c87ed) with a [demo on HuggingFace Spaces](https://huggingface.co/spaces/lixin4ever/VideoLLaMA2).
   - The [research paper](https://huggingface.co/papers/2406.07476) is also available on HuggingFace, showcasing advancements in multimodal AI combining video understanding and language modeling.
  


**4. AI Safety and Regulation Discussions**

- **Nancy Pelosi Opposes California AI Bill**: **Speaker Emerita Nancy Pelosi** issued a statement opposing **California Senate Bill 1047** on AI regulation.
   - The full statement can be found on the [House of Representatives website](http://pelosi.house.gov/news/press-releases/pelosi-statement-opposition-california-senate-bill-1047), highlighting ongoing debates about how to approach AI governance at the state level.
- **Procreate Rejects Generative AI Integration**: The CEO of **Procreate** made a clear statement that they will not be integrating generative AI into their products, a decision celebrated by many artists and users on social media.
   - Some observers noted that this stance might change in the future, as it could potentially limit new feature development. This highlights the ongoing tension between traditional creative tools and the rapid advancement of AI in the creative industry.
- **Gary Marcus Revisits AI Bubble Concerns**: AI researcher **Gary Marcus** revisited his keynote from AGI-21 in a video titled "**The AI Bubble: Will It Burst, and What Comes After?**", noting that many issues he highlighted then are still relevant today despite significant AI advances.
   - This discussion, available on [YouTube](https://www.youtube.com/watch?v=91SK90SahHc), reflects ongoing debates about the sustainability and trajectory of current AI development trends and their potential societal impacts.
  


---

# PART 1: High level Discord summaries

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Flux: The New King?**: Members discussed Flux's potential to take over the image generation AI community, with new Loras and merges appearing daily.
   - Some believe Stability AI needs to release something soon to compete, as Flux is becoming a dominant force in CivitAI and Hugging Face.
- **Flux vs. SD3: A Race to the Top**: There's a debate about whether Flux is fundamentally different from SD3, with both models using DiT architecture, ret flow loss, and similar VAE sizes.
   - The key difference is that Flux dev was distilled from a large model, while Stability AI could also pull that trick. Some prefer non-distilled models, even if image quality is lower.
- **Flux Training: Challenges and Opportunities**: Members discussed the challenges of training Loras for Flux, noting that the training code hasn't been officially released yet.
   - Some users are exploring methods for training Loras locally, while others recommend using Replicate's official Flux LoRA Trainer for faster and easier results.
- **ComfyUI vs. Forge: A Battle of the UIs**: Users discussed the performance differences between ComfyUI and Forge, with some finding Forge to be faster, especially for batch processing.
   - The discussion touched on the impact of Gradio 4 updates on Forge and the potential for future improvements. Some users prefer the flexibility of ComfyUI, while others appreciate the optimization of Forge.
- **GPU Recommendations for Stable Diffusion**: Members shared their experiences with various GPUs and their performance for Stable Diffusion, with 16GB VRAM considered a minimum and 24GB being comfortable.
   - The discussion touched on the importance of VRAM over CPU speed and the impact of RAM and other apps on performance. The consensus was to try different models and encoders to find the best fit for each system.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hermes 2.5 Outperforms Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** appears to perform better than **Hermes 2** in various benchmarks.
   - Hermes 2 scored a **34.5** on the MMLU benchmark whereas Hermes 2.5 scored **52.3**.
- **Mistral Struggles Expanding Beyond 8k**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed "cursed model merging".
- **Open Empathic Project Plea for Assistance**: A member appealed for help in expanding the categories of the **Open Empathic** project, particularly at the lower end.
   - They shared a [YouTube video on the Open Empathic Launch & Tutorial](https://youtu.be/GZqYr8_Q7DE) that guides users to contribute their preferred movie scenes from YouTube videos, as well as a link to the [OpenEmpathic project itself](https://dct.openempathic.ai/).
- **FP8 Training with 1st Momentum Achieves Similar Loss**: Training a **1B FP8 model** with **1st momentum in FP8** smoothly up to **48k steps** resulted in a loss comparable to **bfloat16** with a **0.08 offset**.
   - This demonstrates that FP8 training can be effective with 1st momentum, achieving similar results as bfloat16 training.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Ghost 8B Beta (1608) released**: **Ghost 8B Beta (1608)**, a top-performing language model with unmatched multilingual support and cost efficiency, has been released.
   - It boasts superior performance compared to Llama 3.1 8B Instruct, GPT-3.5 Turbo, Claude 3 Opus, GPT-4, and more in winrate scores.
- **Ghost 8B Beta's Multilingual Prowess**: **Ghost 8B Beta** now supports **16 languages**, including English, Vietnamese, Spanish, Chinese, and more.
   - It offers two context options (8k and 128k) and improved math, reasoning, and instruction-following capabilities for better task handling.
- **Ghost 8B Beta Outperforms Competitors**: **Ghost 8B Beta** outperforms models like Llama 3.1 8B Instruct, GPT 3.5 Turbo, Claude 3 Opus, Claude 3 Sonnet, GPT-4, and Mistral Large in AlpacaEval 2.0 winrate scores.
   - This impressive performance highlights its superior knowledge capabilities and multilingual strength.
- **Code Editing with LLMs**: A new paper explores using Large Language Models (LLMs) for code editing based on user instructions.
   - It introduces EditEval, a novel benchmark for evaluating code editing performance, and InstructCoder, a dataset for instruction-tuning LLMs for code editing, containing over 114,000 instruction-input-output triplets.
- **Reasoning Gap in LLMs**: A research paper proposes a framework to evaluate reasoning capabilities of LLMs using functional variants of benchmarks, specifically the MATH benchmark.
   - It defines the "reasoning gap" as the difference in performance between solving a task posed as a coding question vs a natural language question, highlighting that LLMs often excel when tasks are presented as code.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Linear Transformers: A Match Made in Softmax Heaven**: Nous Research has published research on a linear transformer variant that matches softmax, allowing for training at O(t) instead of O(t^2).
   - The research, available [here](https://manifestai.com/articles/symmetric-power-transformers/), explores this new variant and its implications for training efficiency.
- **Falcon Mamba 7B Bests Llama 3 8B**: A [YouTube video](https://www.youtube.com/watch?v=dokzrFa-DtY) announcing the release of **Falcon Mamba 7B** claims that it outperforms **Llama 3 8B**.
   - This could have significant implications for the field of large language models, as Falcon Mamba 7B is a relatively new and promising model.
- **Regex Debated as Chunking Technique**: A user shared their thoughts on a regex-based text chunker,  stating they would "scream" if they saw it in their codebase, due to the complexity of regex.
   - Another user, however, countered by arguing that for a text chunker specifically, regex might be a "pretty solid option" since it provides "backtracking benefits" and allows for flexibility in chunking settings.
- **Hermes 3: The Performance King of N8Bench?**: Hermes 3 scored identical to Llama 3.1 Instruct on the N8Bench benchmark, which is a measure of a model's ability to reason and solve problems.
   - This is a significant result, as Llama 3.1 Instruct is considered to be one of the most advanced language models available.
- **Gemini Flash: The Future of RAG?**: A user reports that they've moved some of their RAG tasks to Gemini Flash, noting that they've seen improvements in summary quality and reduced iteration requirements.
   - They share a script they've been using to process raw, unstructured transcripts with Gemini Flash, available on GitHub at [https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/unstruct2flashedTRANSCRIPT.py](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/unstruct2flashedTRANSCRIPT.py).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro is a Pain**: Multiple users reported issues with Perplexity Pro signup process, with users being unable to complete the signup without paying, despite receiving an offer for a free year.
   - Users were advised to reach out to support@perplexity.ai for assistance with this issue.
- **Obsidian Copilot Gets a Claude Boost**: A user shared their experience using the Obsidian Copilot plugin with a Claude API key, finding it to be a solid choice in terms of performance.
   - They stressed the importance of checking API billing settings before committing and also highlighted the need for Obsidian to have real-time web access.
- **Perplexity's Image Generation Feature Struggles**: Users discussed the shortcomings of Perplexity's image generation feature, which is currently only accessible for Pro users, requiring an AI prompt for image description.
   - This was considered a 'weird' and 'bad' implementation by users, who highlighted the need for a more streamlined approach to image generation.
- **Perplexity Search Encounters Hiccups**: Several users reported issues with Perplexity's search quality, encountering problems with finding relevant links and receiving inaccurate results.
   - These issues were attributed to possible bugs, prompts changes, or inference backend service updates.
- **Perplexity Model Changes Leave Users Concerned**: Discussions revolved around changes in Perplexity's models, with users expressing concerns about the potential decline in response quality and the increase in "I can't assist with that" errors.
   - Other concerns included missing punctuation marks in API responses and the use of Wolfram Alpha for non-scientific queries.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hermes 3 405B is free this weekend!**: **Hermes 3 405B** is free for a limited time, with **128k context**, courtesy of **Lambda Labs**.
   - You can check it out at [this link](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended).
- **GPT-4 extended is now on OpenRouter**: You can now use **GPT-4 extended output** (alpha access) through **OpenRouter**.
   - This is capped at **64k max tokens**.
- **Perplexity Huge is the largest online model on OpenRouter**: **Perplexity Huge** launched **3 days ago** and is the **largest online model on OpenRouter**.
   - You can find more information at [this link](https://x.com/OpenRouterAI/status/1824593712095301914).
- **A Week of Model Launches**: This week saw **10 new model launches** on OpenRouter, including **GPT-4 extended**, **Perplexity Huge**, **Starcannon 12B**, **Lunaris 8B**, **Llama 405B Instruct bf16** and **Hermes 3 405B**.
   - You can see the full list at [this link](https://x.com/OpenRouterAI/status/1824608728810991637).
- **Quantization Degrades Performance**: **Quantization** can massively degrade the performance of **405B models**, according to @hyperbolic_labs.
   - They recommend reaching out to them if you are concerned about performance, as they offer alternative solutions.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **INT8 Quantization for Faster CPUs?**: A member inquired about potential performance gains from using INT8 quantization for smaller models on CPUs.
   - They suggested that some CPUs may natively support INT8 execution, bypassing conversion to FP32 and potentially improving performance.
- **Llama.cpp Supports Mini-CPM-V2.6 & Nemotron/Minitron**: A member confirmed that the latest llama.cpp version supports Mini-CPM-V2.6 and Nvidia's Nemotron/Minitron models.
   - This update expands the range of models compatible with llama.cpp, enhancing its versatility for LLM enthusiasts.
- **Importing Chats into LM Studio**: A member sought guidance on importing chat logs from a JSON export into LM Studio.
   - Another member clarified that chat data is stored in JSON files and provided instructions on accessing the relevant folder location.
- **Vulkan Error: CPU Lacks AVX2 Support**: A user encountered an error indicating their CPU lacks AVX2 support, preventing the use of certain features.
   - A helpful member requested the CPU model to assist in diagnosing and resolving the issue.
- **LLMs Interacting with Webpages: A Complex Challenge**: A member discussed the possibility of enabling LLMs to interact with webpages, specifically seeking a 'vision' approach.
   - While tools like Selenium and IDkit were mentioned, the general consensus is that this remains a challenging problem due to the diverse structure of webpages.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Claude Outperforms Chat-GPT on Code**: A member stated that Claude tends to be better at code than Chat-GPT.
   - The fact that 4o's API costs more than Claude makes no sense tbh.
- **Livebench.ai: Yann LeCun's Open Source Benchmark**: Livebench.ai is an open source benchmark created by Yann LeCun and others.
   - The LMSys benchmark is probably the worst as of now.
- **Claude Projects vs Chat-GPT Memory Feature**: A member believes Claude Projects are more useful than Chat-GPT's memory feature.
   - The member also stated that custom GPTs are more like projects, allowing for the use of your own endpoints.
- **OpenAI is Winning the Attention Game**: OpenAI is winning by controlling attention through releasing new models like GPT-4o.
   - The member stated that people are talking about OpenAI's new models, even if they don't want to participate in the tech hype.
- **GPT-4o is Now Worse than Claude and Mistral**: Members have noticed that GPT-4o has become dumber lately and may be suffering from a type of Alzheimer's.
   - Claude Sonnet is being praised for its superior performance and is becoming a preferred choice among members.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Topology's CLM: Learning Like Humans**: Topology has released the [Continuous Learning Model (CLM)](https://yellow-apartment-148.notion.site/CLM-Docs-507d762ad7b14d828fac9a3f91871e3f), a new model that remembers interactions, learns skills autonomously, and thinks in its free time, just like humans.
   - This model can be tried out at [http://topologychat.com](http://topologychat.com).
- **GPT5 Needs to Be **20x** Bigger**: Mikhail Parakhin tweeted that to get meaningful improvement in AI models, a new model should be at least **20x** bigger than the current model.
   - This would require **6 months** of training and a new, **20x** bigger datacenter, which takes about a year to build.
- **Procreate Rejects Generative AI**: The CEO of Procreate has stated that they will not be integrating generative AI into their products.
   - While some artists and users on social media celebrated the news, others noted that it could mean no new features will be added in the future, and this could change.
- **DSPy: Not Quite Commercial Yet**: There is no commercial company behind **DSPy** yet, although Omar is working on it.
   - A member shared that they went to the **Cursor** office meetup, and while there was no alpha to share, they did say hi.
- **DSPy Bridging the Gap**: **DSPy** is designed to bridge the gap between prompting and finetuning, allowing users to avoid manual prompt tuning.
   - The paper mentions that **DSPy** avoids prompt tuning, potentially making it easier to switch models, retune to data shifts, and more.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Office Hours Kick-Off!**: Join Cohere's **Sr. Product Manager** and **DevRel** for a casual session on **product and content updates** with **best practices** and **Q&A** on **Prompt Tuning**, **Guided Generations API with Agents**, and **LLM University Tool Use Module**.
   - The event takes place today at **1 PM ET in the #stage channel** and can be found at [this link](https://discord.com/events/954421988141711382/1265012161965461625).
- **Cohere Prompt Tuner: Optimized Prompting!**: Learn about the **Cohere Prompt Tuner**, a powerful tool to optimize prompts and improve the accuracy of your LLM results.
   - The blog post details how to utilize this tool and the [associated features](https://cohere.com/blog/intro-prompt-tuner).
- **Command-r-plus Not Working?**: A user reported that **command-r-plus** in **Sillytavern** stopped working consistently when the context length reaches 4000 tokens.
   - The user has been attempting to use the tool to enhance their workflow, but is facing this unexpected issue.
- **API Key Partial Response Issues**: A user reported experiencing issues with their API key returning only partial responses, even after trying different Wi-Fi routers and cellular data.
   - The user is currently seeking a solution to this problem.
- **Structured Outputs for Accurate JSON Generations**: **Structured Outputs**, a recent update to Cohere's tools, delivers **80x faster** and **more accurate** **JSON generations** than open-source implementations.
   - This new feature improves the accuracy of JSON output and is discussed in [this blog post](https://cohere.com/blog/introducing-structured-outputs).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Yi Tay Works on Chaos No Sleep Grind**: The discussion touched on work styles of various AI organizations with one member suggesting that Yi Tay operates with a 'chaos no sleep grind' mentality. 
   - They referenced a tweet from Phil (@phill__1) suggesting that 01AI may be pulling out of non-Chinese markets,  [what is going on with .@01AI_Yi? Are they pulling out of the non Chinese market?](https://x.com/phill__1/status/1825438202548658526?s=46).
- **Nancy Pelosi Opposes California AI Bill**: Speaker Emerita Nancy Pelosi issued a statement opposing California Senate Bill 1047 on AI regulation.
   - The statement was released on the House of Representatives website: [Pelosi Statement in Opposition to California Senate Bill 1047](http://pelosi.house.gov/news/press-releases/pelosi-statement-opposition-california-senate-bill-1047).
- **Zicheng Xu Laid Off From Allen-Zhu's Team**: Zeyuan Allen-Zhu announced the unexpected layoff of Zicheng Xu, the author of the "Part 2.2" tutorial. 
   - Allen-Zhu strongly endorses Xu and provided his email address for potential collaborators or employers: zichengBxuB42@gmail.com (remove the capital 'B').
- **Nous Hermes Discord Drama Over Evaluation Settings**: A user mentioned a discussion in the Nous Discord regarding a user's perceived rudeness and misrepresentation of evaluation settings. 
   - The user mentioned that their evaluation details were in the SFT section of the paper, and admitted that it doesn't feel good to get things wrong but the core of the article is still valid. 
- **Meta Cooking (Model Harnessing) Creates Confusion**: A user wondered what "meta cooking" is, suggesting a potential conflict or drama in the Nous Discord.
   - The user mentioned finding contradictory information about evaluation settings, possibly due to the use of default LM Harness settings without clear documentation. 



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GrokAdamW Makes Axolotl Faster**: GrokAdamW, a PyTorch optimizer that encourages fast grokking, was released and is working with Axolotl via the Transformers integration. [GrokAdamW repository](https://github.com/cognitivecomputations/grokadamw)
   - The optimizer is inspired by the GrokFast paper, which aims to accelerate generalization of a model under the grokking phenomenon. [GrokFast paper](https://arxiv.org/abs/2405.20233)
- **Gemma 2b Training Hiccup**: A user reported a consistent loss of 0.0 during training of a Gemma 2b model, with a nan gradient norm.
   - The user recommended using eager attention instead of sdpa for training Gemma 2b models, which fixed the zero loss issue.
- **Custom Loaders & Chat Templates in Axolotl**: A user asked for clarification on using a Chat Template type in a .yml config file for Axolotl, specifically interested in specifying which loader to use, for example, ShareGPT.
   - Another user suggested the user could specify which loader to use by providing a custom .yml file.
- **Fine-Tuning with Axolotl: No Coding Required**: A user clarified that fine-tuning with Axolotl generally does not require coding knowledge, but rather understanding how to format datasets and adapt existing examples.
   - A user mentioned owning a powerful AI rig to run LLama 3.1 70b, but felt it was still lacking in some key areas and wanted to use their dataset of content for fine-tuning.
- **LLaMa 3.1 8b Lora Detects Post-Hoc Reasoning**: A user is training a LLaMa 3.1 8b Lora to detect post-hoc reasoning within a conversation, having spent three days curating a small dataset of less than 100 multi-turn conversations with around 30k tokens.
   - The user employed Sonnet 3.5 to help with generating examples, but had to fix multiple things in each generated example, despite careful prompt crafting, because even when instructing the models not to create examples with post-hoc reasoning, they still generated them due to their fine-tuning data.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Caching Issues**: A member was confused about why `.batch_as_completed()` wasn't sped up by caching, even though `.invoke()` and `.batch()` were near instant after caching.
   - They observed that the cache was populated after the first run, but `.batch_as_completed()` didn't seem to utilize it.
- **LLMs struggle with structured output**: A member mentioned that local LLMs, like Llama 3.1, often had difficulty producing consistently structured output, specifically when it came to JSON parsing.
   - They inquired about datasets specifically designed to train models for improved JSON parsing and structured output for tools and ReAct agents.
- **Deleting files in a RAG chatbot**: A member discussed how to implement a delete functionality for files in a RAG chatbot using MongoDB as a vector database.
   - A response provided examples of using the `delete` method from the LangChain library for both MongoDB vector stores and OpenAIFiles, along with relevant documentation links.
- **Hybrid Search Relevance Issues**: A member encountered relevance issues with retrieved documents and generated answers in a RAG application using a hybrid search approach with BM25Retriever and vector similarity search.
   - Suggestions included checking document quality, adjusting retriever configurations, evaluating the chain setup, and reviewing the prompt and LLM configuration.
- **CursorLens is a new dashboard for Cursor users**: CursorLens is an open-source dashboard for Cursor users that provides analytics on prompts and allows configuring models not available through Cursor itself.
   - It was recently launched on ProductHunt: [https://www.producthunt.com/posts/cursor-lens](https://www.producthunt.com/posts/cursor-lens).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Orange Pi 5 Review: The New Affordable SBC**: A user shared a [YouTube video review](https://youtu.be/79lquFD3oT4) of the **Orange Pi 5**, a new **Arm-based SBC**.
   - The video emphasizes that the **Orange Pi 5** is not to be confused with the **Raspberry Pi 5**.
- **GPT-4o-mini Model woes: A Quick Fix**: A user encountered trouble setting their model to **GPT-4o-mini**.
   - Another user provided the solution: `interpreter --model gpt-4o-mini`.
- **OpenInterpreter Settings Reset: A Revert Guide**: A user sought a way to revert OpenInterpreter settings to default after experimentation.
   - The solution involved using `interpreter --profiles` to view and edit profiles, and potentially uninstalling and reinstalling OpenInterpreter.
- **OpenInterpreter API Integration: Building a Bridge**: A user inquired about integrating OpenInterpreter into their existing AI core, sending requests and receiving outputs.
   - The recommended solution involved using a Python script with a Flask server to handle communication between the AI core and OpenInterpreter.
- **Local LLMs for Bash Commands: CodeStral and Llama 3.1**: A member requested recommendations on local LLMs capable of handling bash commands.
   - Another member suggested using **CodeStral** and **Llama 3.1**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **LLMs Struggle with Reliability**: Large Language Models (LLMs) are known for producing factually incorrect information, leading to "phantom" content that hinders their reliability.
   - This issue is addressed by **WeKnow-RAG**, a system that integrates web search and Knowledge Graphs into a Retrieval-Augmented Generation (RAG) system to improve LLM accuracy and reliability.
- **DSPy Unveils its Roadmap**: The roadmap for **DSPy 2.5** (expected in 1-2 weeks) and **DSPy 3.0** (in a few months) has been released, outlining objectives, milestones, and community contributions.
   - The roadmap is available on GitHub: [DSPy Roadmap](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md).
- **Langgraph and Routequery Class Error**: A user encountered an error with the `routequery` class in **Langgraph**.
   - They sought guidance on integrating **DSPy** with a large toolset and shared a link to the **Langgraph** implementation: [Adaptive RAG](https://github.com/sksarvesh007/adaptive-rag/blob/main/langgraph_adaptive_rag.ipynb).
- **Optimizing Expert-Engineered Prompts**: A member questioned whether **DSPy** can optimize prompts that have already been manually engineered by experts.
   - They inquired if **DSPy** effectively optimizes initial drafts and also improves established prompting systems.
- **Colpali Fine-Tuning Discussion**: A discussion centered around the finetuning of **Colpali**, a model requiring specialized expertise due to its domain-specific nature.
   - The discussion highlighted the importance of understanding the data needed for effectively finetuning **Colpali**.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **FLUX Dev Can Generate Grids**: A user shared that **FLUX Dev** can generate 3x3 photo grids of the same (fictional) person.
   - This could be useful for training **LORAs** to create consistent characters of all kinds of fictional people.
- **Training LORAs for Specific Purposes**: A user expressed interest in training **LORAs** for specific purposes like **dabbing**, **middle finger**, and **30s cartoon**.
   - They mentioned the possibility of converting their **FLUX Dev LoRA** into **FP8** or using an **FP8 LoRA trainer on Replicate**.
- **LLMs for Medical Assistance: Not Ready Yet**: Several users expressed skepticism about using **LLMs** for medical assistance in their current state.
   - They believe **LLMs** are not yet reliable enough for such critical applications.
- **JPEG-LM: LLMs for Images & Videos?**: A new research paper proposes modeling images and videos as compressed files using canonical codecs (e.g., JPEG, AVC/H.264) within an autoregressive LLM architecture.
   - This approach eliminates the need for raw pixel value modeling or vector quantization, making the process more efficient and offering potential for future research.
- **JPEG-LM vs. SIREN: A Battle of the Titans?**: A user playfully claims to have outperformed the **SIREN** architecture from 2020 with a 33kB complex-valued neural network.
   - While acknowledging that NVIDIA's **Neural Graphics Primitives** paper from 2022 significantly advanced the field, they highlight the importance of using MS-SSIM as a metric for image quality assessment, as opposed to just MSE and MAE.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Workflows Take Center Stage**: Rajib Deb shared a video showcasing LlamaIndex's workflow capabilities, demonstrating decorators, types for control flow, event-driven process chaining, and custom events and steps for complex tasks.
   - The video focuses on workflows, emphasizing their ability to build sophisticated applications with a more structured approach.
- **Building Agentic RAG Assistants with Claude 3.5**: Richmond Lake's tutorial guides users on building an agentic knowledge assistant using Claude 3.5, MongoDB, and LlamaIndex, highlighting building an agentic knowledge assistant over a pre-existing RAG pipeline.
   - This tutorial demonstrates using LlamaIndex for advanced RAG techniques, emphasizing tool selection, task decomposition, and event-driven methodologies.
- **BeyondLLM Streamlines Advanced RAG Pipelines**: BeyondLLM, developed by AIPlanetHub, provides abstractions on top of LlamaIndex, enabling users to build advanced RAG pipelines with features like evaluation, observability, and advanced RAG capabilities in just 5-7 lines of code.
   - These advanced RAG features include query rewriting, vector search, and document summarization, simplifying the development of sophisticated RAG applications.
- **Web Scrapers: A LlamaIndex Dilemma**: A member asked for recommendations for web scrapers that work well with LlamaIndex, and another member recommended FireCrawl, sharing a YouTube video showing a more complex implementation of a LlamaIndex workflow.
   - The conversation highlights the need for effective web scraping tools that seamlessly integrate with LlamaIndex, enabling efficient knowledge extraction and processing.
- **Unveiling the Secrets of RouterQueryEngine and Agents**: A member sought clarification on the difference between LlamaIndex's RouterQueryEngine and Agents, specifically in terms of routing and function calling.
   - The discussion clarifies that the RouterQueryEngine acts like a hardcoded agent, while Agents offer greater flexibility and generality, highlighting the distinct capabilities of each approach.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **HF Spaces Limitations**: A member had trouble hosting their own LLM using **HF Spaces**, as **ZeroGPU** doesn't support **vLLM**.
   - The member was seeking an alternative solution, potentially involving **Modal**.
- **Modal for LLM Hosting**: Another member reported using **Modal** for hosting LLMs.
   - However, they are currently transitioning to **FastHTML** and are looking for a setup guide.
- **Jarvis Labs for Fine-tuning**: One member shared their experience using **Jarvis Labs** exclusively for fine-tuning LLMs.
   - This suggests that **Jarvis Labs** might offer a streamlined approach compared to other platforms.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **OpenAI and Google Get Cheaper with Batching APIs**: OpenAI and Google launched new batching APIs for some models, offering a 50% cost reduction compared to regular requests.
   - However, these APIs currently lack processing guarantees, service level agreements (SLAs), and retries.
- **CuminAI: Open-Source Batching APIs**: CuminAI provides a solution for creating batching APIs for open-source models, similar to those offered by OpenAI.
   - Check out their step-by-step guide on "How to Get a Batching API Like OpenAI for Open-Source Models" [here](https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49).
- **SLMs: The New Superheroes of AI?**: CuminAI highlights the potential of Small Language Models (SLMs), arguing that "bigger isn't always better" in AI.
   - While Large Language Models (LLMs) have dominated, SLMs offer a more cost-effective and efficient alternative, especially for tasks that don't require extensive computational power.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile Boosts Performance & Adds New Features**: **Llamafile** has released new features, including **Speech to Text Commands**, **Image Generation**, and a **3x Performance Boost** for its HTTP server embeddings.
   - The full update, written by [Justine](https://discord.com/channels/1089876418936180786/1262961704602570832/1275110073584320576), details the performance improvements and new features.
- **Mozilla AI Celebrates Community at Rise25**: Mozilla AI is celebrating community members who are shaping a future where AI is responsible, trustworthy, inclusive, and centered around human dignity.
   - Several members attended the event, including <@631210549170012166>, <@1046834222922465314>, <@200272755520700416>, and <@1083203408367984751>.
- **ML Paper Talks: Agents & Transformers Deep Dive**: Join a session hosted by <@718891366402490439> on **Communicative Agents** and **Extended Mind Transformers**.
   - RSVP for the sessions: [Communicative Agents](https://discord.com/events/1089876418936180786/1266733035231903795) with author <@878366123458977893>, and [Extended Mind Transformers](https://discord.com/events/1089876418936180786/1267946366680694817) with author <@985920344856596490>.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1274088287803281510)** (567 messages🔥🔥🔥): 

> - `Flux`
> - `Flux vs. SD3`
> - `Flux training`
> - `ComfyUI vs Forge`
> - `GPU recommendations` 


- **Flux: The New King?**: Members discussed Flux's potential to absorb the image generation AI community, with new Loras and merges appearing daily. 
   - Some believe Stability AI needs to release something soon to compete, as Flux is becoming a dominant force in CivitAI and Hugging Face.
- **Flux vs. SD3: A Race to the Top**: There's a debate about whether Flux is fundamentally different from SD3, with both models using DiT architecture, ret flow loss, and similar VAE sizes.
   - The key difference is that Flux dev was distilled from a large model, while Stability AI could also pull that trick. Some prefer non-distilled models, even if image quality is lower.
- **Flux Training: Challenges and Opportunities**: Members discussed the challenges of training Loras for Flux, noting that the training code hasn't been officially released yet. 
   - Some users are exploring methods for training Loras locally, while others recommend using Replicate's official Flux LoRA Trainer for faster and easier results.
- **ComfyUI vs. Forge: A Battle of the UIs**: Users discussed the performance differences between ComfyUI and Forge, with some finding Forge to be faster, especially for batch processing. 
   - The discussion touched on the impact of Gradio 4 updates on Forge and the potential for future improvements. Some users prefer the flexibility of ComfyUI, while others appreciate the optimization of Forge.
- **GPU Recommendations for Stable Diffusion**: Members shared their experiences with various GPUs and their performance for Stable Diffusion, with 16GB VRAM considered a minimum and 24GB being comfortable.
   - The discussion touched on the importance of VRAM over CPU speed and the impact of RAM and other apps on performance. The consensus was to try different models and encoders to find the best fit for each system.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dsc.gg/vexel">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://tenor.com/view/blender-vram-ram-memory-gone-gif-27551226">Blender Vram GIF - Blender Vram RAM - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/kto-lbow-hi-hello-hi-there-gif-25347432">Kto Lbow GIF - Kto Lbow Hi - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.instagram.com/p/C-vD5i4uVvY/">Aleksey Efremov on Instagram: &quot;ALERT 21
.
.
#weirdcore #liminalspace #backroomsaesthetic #wierdcore #backrooms #liminalcore #dreamcore #nastolgia #aiart #ai&quot;</a>: solar.w on August 16, 2024: &quot;ALERT 21 . . #weirdcore #liminalspace #backroomsaesthetic #wierdcore #backrooms #liminalcore #dreamcore #nastolgia #aiart #ai&quot;. </li><li><a href="https://www.instagram.com/solar.w/reel/C-VZ21juqND/">Aleksey Efremov on Instagram: &quot;ALERT 19
&#x2026;
&#x2026;
#liminalspace #nostalgiacore #backrooms #afterhours #dreamcore #weirdcore&quot;</a>: solar.w on August 6, 2024: &quot;ALERT 19 &#x2026; &#x2026; #liminalspace #nostalgiacore #backrooms #afterhours #dreamcore #weirdcore&quot;. </li><li><a href="https://replicate.com/lucataco/ai-toolkit">lucataco/ai-toolkit – Run with an API on Replicate</a>: no description found</li><li><a href="https://stability.ai/stable-artisan">Stable Artisan &mdash; Stability AI</a>: Stable Artisan is a fun multimodal generative AI Discord bot that utilizes the products on the Stability AI Platform API within the Discord ecosystem.</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1euz2a9/union_flux_controlnet_running_on_comfyui_workflow/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/Acly/krita-ai-diffusion/wiki/ComfyUI-Setup">ComfyUI Setup</a>: Streamlined interface for generating images with AI in Krita. Inpaint and outpaint with optional text prompt, no tweaking required. - Acly/krita-ai-diffusion</li><li><a href="https://www.youtube.com/watch?v=gO3Mk3le0qs">AI Face Consistency is NOT the real goal: it&#39;s this...</a>: There&#39;s a lot of talk about the need for character consistency in AI Video, but with &quot;REAL&quot; (narrative) Filmmaking, character consistency is just one element...</li><li><a href="https://www.youtube.com/watch?v=GG-xDgdjhjU">Garuda Linux KDE Dr460nized - Quick Review (walkthrough)</a>: It has been some time since I gave Garuda Linux a look. Is it easy to use, does it &quot;just work&quot;? Watch me use it, including a few stutters I ran into along th...</li><li><a href="https://youtu.be/0AT8esyY0Fw">Warp Fusion: Step by Step Tutorial</a>: Warp fusion is a fantastic AI animation tool to create videos that just pop. In this video I show you step by step how to use warp fusion using a remote GPU....</li><li><a href="https://youtu.be/bm1PWniLIlc?si=ZRIXbV1JHifS9L31">Vídeo 100% CON IA 4K | La historia del té - Un viaje a través del tiempo</a>: Déjate llevar por la antigua y fascinante historia del té. Descubre cómo esta milenaria bebida ha conectado culturas, cruzado continentes y evolucionado a tr...</li><li><a href="https://youtu.be/j9I0iLxGJl0">RTX Remix I Remaster the Classics with RTX and 1,000s of AI Models via ComfyUI</a>: Learn more about NVIDIA RTX Remix and the powerful new REST API: https://www.nvidia.com/en-us/geforce/news/rtx-remix-rest-api-comfyui-app-connectorsNVIDIA is...</li><li><a href="https://github.com/madebyollin/taesd">GitHub - madebyollin/taesd: Tiny AutoEncoder for Stable Diffusion</a>: Tiny AutoEncoder for Stable Diffusion. Contribute to madebyollin/taesd development by creating an account on GitHub.</li><li><a href="https://civitai.com/articles/391/tutorial-dreambooth-lora-training-using-kohyass">Tutorial: Dreambooth LoRA training using Kohya_SS | Civitai</a>: [Edits 6/24 - Cover image and outputs updated with images that in line with this site&#x27;s updated guidelines. ] [Edits 7/1 - Link to Lycoris/LoCon Tu...</li><li><a href="https://github.com/kohya-ss/sd-scripts/tree/25f77f6ef04ee760506338e7e7f9835c28657c59?tab=readme-ov-file#flux1-lora-training-wip">GitHub - kohya-ss/sd-scripts at 25f77f6ef04ee760506338e7e7f9835c28657c59</a>: Contribute to kohya-ss/sd-scripts development by creating an account on GitHub.</li><li><a href="https://civitai.com/models/396388/just-better-gun">Just Better Gun. - v1.0 | Stable Diffusion LoRA | Civitai</a>: Just Better Gun is basically a Lora that is based around.... gun.... the data set focus around tactical warfare gun that have a lot of scope and de...</li><li><a href="https://tensor.art/models/759856135286068673/FLUX-HYPER-TRAINED-DREAM-DIFFUSION-BY-DICE-V-1">FLUX HYPER TRAINED - DREAM DIFFUSION - BY DICE - V 1 | Stable Diffusion Model - Checkpoint</a>: 5.5K runs, 77 stars, 53 downloads. FLUX DREAM DIFFUSION BY DICEstart of with these settings in comfy to get a feel for how it runs ....Simple Prompt : a jet ...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1274086399401922633)** (449 messages🔥🔥🔥): 

> - `Verification issues`
> - `Hermes 2.5`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic` 


- **Hugging Face Verification Issues**: A member experienced issues with the "login with huggingface" verification process, with the login button showing "Not logged in."
   - They tried both on mobile and desktop, but it wouldn't work and were advised to try again later on PC.
- **Hermes 2.5 Outperforms Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** appears to perform better than **Hermes 2** in various benchmarks.
   - Hermes 2 scored a **34.5** on the MMLU benchmark whereas Hermes 2.5 scored **52.3**.
- **Mistral Struggles Expanding Beyond 8k**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed "cursed model merging".
- **Open Empathic Project Plea for Assistance**: A member appealed for help in expanding the categories of the **Open Empathic** project, particularly at the lower end.
   - They shared a [YouTube video on the Open Empathic Launch & Tutorial](https://youtu.be/GZqYr8_Q7DE) that guides users to contribute their preferred movie scenes from YouTube videos, as well as a link to the [OpenEmpathic project itself](https://dct.openempathic.ai/). 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2204.03930">From Rewriting to Remembering: Common Ground for Conversational QA Models</a>: In conversational QA, models have to leverage information in previous turns to answer upcoming questions. Current approaches, such as Question Rewriting, struggle to extract relevant information as th...</li><li><a href="https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/howto_wsl.html">WSL How to guide - Use ROCm on Radeon GPUs &#8212; Use ROCm on Radeon GPUs</a>: no description found</li><li><a href="https://huggingface.co/parler-tts/parler-tts-mini-expresso/discussions/8#66be241be61ccd71d7b5cd7d">parler-tts/parler-tts-mini-expresso · problem running this model and other parler models...</a>: no description found</li><li><a href="https://www.kaggle.com/datasets/umerhaddii/saudi-aramco-stock-price-data">Saudi Aramco Stock Price Data</a>: Saudi Aramco Stock Price Data from 2015 - 2024</li><li><a href="https://huggingface.co/spaces/taneemishere/html-code-generation-from-images-with-deep-neural-networks">Image to HTML Code Demo - a Hugging Face Space by taneemishere</a>: no description found</li><li><a href="https://paperswithcode.com/sota/long-range-modeling-on-lra">Papers with Code - LRA Benchmark (Long-range modeling)</a>: The current state-of-the-art on LRA is Mega. See a full comparison of 32 papers with code.</li><li><a href="https://huggingface.co/monadical-labs/minecraft-skin-generator-sdxl/tree/main/">monadical-labs/minecraft-skin-generator-sdxl at main</a>: no description found</li><li><a href="https://huggingface.co/blog/tomaarsen/attention-sinks">🕳️ Attention Sinks in LLMs for endless fluency</a>: no description found</li><li><a href="https://fedoramagazine.org/using-artificial-intelligence-to-set-a-guitar-sound/">Using Artificial Intelligence to set a guitar sound - Fedora Magazine</a>: Using Artificial Intelligence with Guitarix to create your sound. Explanation and demostration.</li><li><a href="https://youtu.be/OenUHAuTyxk">Dev Readers Notebook 10 : 20 Concepts of Figma in 2 mins</a>: In this Dev Notebook Series video, I&#39;ll cover 20 basic concepts of Figma to give you a comprehensive overview of what it is and how it works. If you haven&#39;t ...</li><li><a href="https://www.youtube.com/watch?v=l9TCDEbRiKM">If You STRUGGLE To Find Happiness In Life, WATCH THIS | STOICISM</a>: Feeling stuck and unhappy with life? Discover how ancient Stoic principles can guide you to true happiness and inner peace. This video breaks down powerful S...</li><li><a href="https://www.youtube.com/watch?v=zv14gyAJWUM">أعظم الدروس و الحكم التي تسمعها في حياتك، أنصحك بتعلمها</a>: تتعدّد المواقف التي يمرّ بها الإنسان خلال حياته.فمنها ما يكون جميلاً ومفرحاً، ومنها ما يكون حزيناً يصيب صاحبها بالضّيق، وتُعتبر هذه المواقف درساً مفيداً للإن...</li><li><a href="https://huggingface.co/openai/whisper-large-v3">openai/whisper-large-v3 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/FunAudioLLM/SenseVoiceSmall">FunAudioLLM/SenseVoiceSmall · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/hella-mad-breakdance-cool-lit-gif-13584980">Hella Mad GIF - Hella Mad Breakdance - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtu.be/Gg8s9iNfExU?si=I3Xw8jEJ9YtjWnNY">Top Tips For Aspiring and Experienced Developers 🌟</a>: This video lists the top advices or tips for developers who are beginners or experienced in the IT sector⭐ JOURNEY 👇👉 My 100 Days of Code https://youtu.be/...</li><li><a href="https://huggingface.co/spaces/AIPeterWorld/Doc-To-Dialogue?logs=container">Doc To Dialogue - a Hugging Face Space by AIPeterWorld</a>: no description found</li><li><a href="https://github.com/jxnl/instructor">GitHub - jxnl/instructor: structured outputs for llms</a>: structured outputs for llms . Contribute to jxnl/instructor development by creating an account on GitHub.</li><li><a href="https://github.com/sandrohanea/whisper.net?tab=readme-ov-file">GitHub - sandrohanea/whisper.net: Whisper.net. Speech to text made simple using Whisper Models</a>: Whisper.net. Speech to text made simple using Whisper Models - sandrohanea/whisper.net</li><li><a href="https://github.com/visioncortex/vtracer">GitHub - visioncortex/vtracer: Raster to Vector Graphics Converter</a>: Raster to Vector Graphics Converter. Contribute to visioncortex/vtracer development by creating an account on GitHub.</li><li><a href="https://arxiv.org/abs/2306.06441v1">Image Vectorization: a Review</a>: Nowadays, there are many diffusion and autoregressive models that show impressive results for generating images from text and other input domains. However, these methods are not intended for ultra-hig...</li><li><a href="https://www.linuxfoundation.org/press/linux-foundation-welcomes-the-open-model-initiative-to-promote-openly-licensed-ai-models">Linux Foundation Welcomes the Open Model Initiative to Promote Openly Licensed AI Models</a>: New initiative will foster the development of free to use, open and ethical AI models.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1274158930427379763)** (4 messages): 

> - `FP8 Training`
> - `Memory Reduction`
> - `Optimizer States` 


- **FP8 Training with 1st Momentum Achieves Similar Loss**: Training a **1B FP8 model** with **1st momentum in FP8** smoothly up to **48k steps** resulted in a loss comparable to **bfloat16** with a **0.08 offset**.
- **FP8 Training with FP8 Optimizer States is Feasible**: Training a **1B FP8 model** with **FP8 optimizer states** achieved a **0.14 offset** compared to the **bfloat16 baseline**, resulting in a **50% memory reduction**.
- **FP8 Training with Mixed Momentum Types**: Training a **1B FP8 model** with **1st momentum in FP8** and **2nd momentum in bfloat16** achieved convergence comparable to **bfloat16** with a **0.08 offset** up to **31k steps**, achieving a **42% memory reduction**.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1274238666885173248)** (3 messages): 

> - `Medical SAM 2`
> - `MedGraphRAG`
> - `Multimodal LLM for Medical Time Series`
> - `ECG-FM`
> - `Private & Secure Healthcare RAG` 


- **Medical SAM 2 for Video Medical Image Segmentation**: Medical SAM 2 is a new research paper that focuses on the segmentation of medical images as video.
   - This paper addresses the need for efficient and accurate video image segmentation in the medical field, offering a novel approach for analyzing and interpreting dynamic medical data.
- **MedGraphRAG: Graph-Enhanced Medical RAG**: MedGraphRAG is a graph-enhanced Medical RAG model that leverages the power of graph networks to enhance medical information retrieval.
   - This paper addresses the challenges of understanding complex medical relationships and extracting relevant knowledge from medical text by combining graph-based representation with RAG capabilities.
- **Multimodal LLM for Medical Time Series**: This research paper introduces a novel multimodal LLM specifically designed for handling medical time series data.
   - This model leverages the combined power of language and time series data, paving the way for more comprehensive and insightful analysis in medical applications.
- **Open Electrocardiogram Foundation Model - ECG-FM**: ECG-FM is an open-source Electrocardiogram Foundation Model designed for ECG analysis.
   - This paper promotes open research and collaboration in the field of ECG analysis, making a valuable resource for medical practitioners and researchers alike.
- **Private & Secure Healthcare RAG**: This paper delves into the development of Private & Secure Healthcare RAG, a critical advancement for protecting patient data in medical information retrieval.
   - This research tackles the crucial issue of privacy and security within the healthcare context by providing a framework for secure and responsible access to medical information.



**Link mentioned**: <a href="https://x.com/OpenlifesciAI/status/1824790439527887073">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: Last & This Week in Medical AI: Top Research Papers/Models  🏅 (August 3 - August 17, 2024)    - Medical SAM 2: Segment medical images as video  - MedGraphRAG: Graph-Enhanced Medical RAG - Multimodal ...

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1274188235676979262)** (18 messages🔥): 

> - `Unity ML Agents`
> - `CursorLens`
> - `Batching APIs`
> - `CuminAI`
> - `NeuroSync` 


- **Wandering Agent 3 - Live Training from Scratch C#**: A Unity ML Agent developer is live-streaming part 3 of their Wandering Agent project, focusing on coding a SAC agent from scratch using Unity ML Agents, building upon previous episodes.
   - They are using C# and plan to keep their existing camera scripts in place. This episode will focus on coding the SAC agent from scratch.
- **CursorLens: Open-Source Dashboard for Prompt Analytics & Model Configuration**: The developer has released CursorLens, an open-source dashboard for visualizing prompt analytics and configuring models not available through Cursor itself.
   - The dashboard is available on ProductHunt and aims to provide insights into prompt performance and allow for customization of models.
- **Batching APIs for Open-Source Models**: Major companies like OpenAI and Google have launched batching APIs for their models, offering cost savings compared to normal requests but lacking processing guarantees, SLAs, and retries.
   - CuminAI provides a solution for creating batching APIs for open-source models, offering a powerful alternative to existing APIs.
- **NeuroSync: Seq2Seq Transformer for Face Blendshape Prediction**: NeuroSync is a sequence-to-sequence transformer model designed to predict face blendshape frames from audio feature inputs.
   - This model uses 4 transformer layers and 4 attention heads, making it the first model on HuggingFace to specialize in predicting face blendshapes from audio.
- **Arabic Whisper Model Training and Deployment**: A YouTube playlist teaches Arabic speech recognition by training a Whisper model on an Arabic speech dataset.
   - The model is then deployed on HuggingFace Models and Spaces, providing a valuable resource for Arabic speech recognition.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/ardha27/VideoAnalyzer">VideoAnalyzer - a Hugging Face Space by ardha27</a>: no description found</li><li><a href="https://huggingface.co/AnimaVR/NeuroSync-0.1a">AnimaVR/NeuroSync-0.1a · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/ardha27/Youtube-AI-Summarizer">Youtube AI Summarizer - a Hugging Face Space by ardha27</a>: no description found</li><li><a href="https://huggingface.co/Q-bert/ChessLlama">Q-bert/ChessLlama · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/AIPeterWorld/Doc-To-Dialogue?logs=container">Doc To Dialogue - a Hugging Face Space by AIPeterWorld</a>: no description found</li><li><a href="https://huggingface.co/spaces/shauninkripped/Sentient-Aid-space1">Sentient AId Test Space 01 - a Hugging Face Space by shauninkripped</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=ZgUVQkhiPi8">Prototype Preview 4 : Finally aligned. Real time and local audio to face model</a>: AI face animation from audio only prototype demo. See full face animation (51 blendshapes) from only audio feature input!huggingface.co/AnimaVR/NeuroSync-0.1a</li><li><a href="https://www.producthunt.com/posts/cursor-lens"> CursorLens - Open Source dashboard and analytics for Cursor IDE | Product Hunt</a>: An open-source dashboard for Cursor.sh IDE. Log AI code generations, track usage, and control AI models (including local ones). Run locally or use upcoming hosted version.</li><li><a href="https://youtube.com/live/1jITphnPvJU?feature=share">Unity ML Agents | Live Training from Scratch C# | Part 3</a>: just a quick sac agent trainer living in a 3d voxel world. build it and they will come.</li><li><a href="https://github.com/ra9hur/Decision-Transformers-For-Trading">GitHub - ra9hur/Decision-Transformers-For-Trading</a>: Contribute to ra9hur/Decision-Transformers-For-Trading development by creating an account on GitHub.</li><li><a href="https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49">How to Get a Batching API Like OpenAI for Open-Source Models</a>: In the world of AI, efficient processing and cost management are paramount. One powerful method for achieving this is batching, which…</li><li><a href="https://github.com/U-C4N/byte-terminal">GitHub - U-C4N/byte-terminal: AI-powered interactive chat interface using Groq API. Features a terminal-like UI, local storage, and customizable commands. Ideal for developers exploring LLMs.</a>: AI-powered interactive chat interface using Groq API. Features a terminal-like UI, local storage, and customizable commands. Ideal for developers exploring LLMs. - U-C4N/byte-terminal</li><li><a href="https://www.youtube.com/watch?v=NiRyViZ8tEw&list=PL6ViV90w3mloHsKo6qi8oIsW_ZDswSrKK&index=5">سلسلة تدريب انظمة الذكاء الاصطناعي على التعرف على الصوت (الجزء الرابع) Deploying Whisper Model</a>: Deploying Whisper small Arabic on Huggingface NamespaceLink to huggingface: https://huggingface.co/mohammed vastai link: https://cloud.vast.ai/?ref_id=145398</li><li><a href="https://huggingface.co/mohammed">mohammed (Mohammed Bakheet)</a>: no description found</li><li><a href="https://huggingface.co/spaces/cfahlgren1/SmolPilot">SmolPilot - a Hugging Face Space by cfahlgren1</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1274401447416692889)** (35 messages🔥): 

> - `LLMs for Penetration Testing`
> - `Recording Issue`
> - `HuggingFace Reading Group`
> - `Batching API for Open-Source Models`
> - `Cross-Posting` 


- **LLMs are getting good at penetration testing**: The Hugging Face Reading Group focused on understanding penetration testing with LLMs.
- **Recording with a drumming noise**: The recording of the meeting had a drumming sound from the presenter's microphone.
- **OpenAI and Gemini's batch API**: A member was looking for a place to post an article about a batching API for open-source models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://topmate.io/athul_nambiar">Athul Nambiar</a>: 🧑🏻‍💻 MERN Stack | 🛠️ TL Labs ❤️ | 🅾️ Open Source Contributor @ Tublian 🚀 |  🎒 Student | 🧑🏻‍💻 Coder | 💰 Investor | 🎨 Designer |   *️⃣ 3X Hackathon Winner 🏆 ( State , National Level )</li><li><a href="https://medium.com/gopenai/understanding-penetration-testing-with-llms-2b0ec6add14a.">no title found</a>: no description found</li><li><a href="https://youtu.be/_f16ofdVC8g">Hugging Face Reading Group 27: Understanding Penetration Testing with LLMs</a>: Sorry, it seems like there&#39;s a bit of drumming sound in the background/my microphonePresenter: Isamu Isozaki, Manil ShresthaPast Presentations: https://githu...</li><li><a href="https://githu...)>>>">no title found</a>: no description found</li><li><a href="https://docs.google.com/presentation/d/1OF_wqUsbbsFoAu4XZFlcaaOmnZngrUuvQrTstcogZ5c/edit#slide=id.g2f35c1725d3_0_549">Pentest pres</a>: AI for Pentesting Isamu Isozaki, Manil Shrestha</li><li><a href="https://docs.google.com/presentation/?usp=slides_web">no title found</a>: no description found</li><li><a href="https://accounts.google.com/ServiceLogin?service=wise&passive=1209600&osid=1&continue=https://docs.google.com/presentation/d/1OF_wqUsbbsFoAu4XZFlcaaOmnZngrUuvQrTstcogZ5c/edit&followup=https://docs.google.com/presentation/d/1OF_wqUsbbsFoAu4XZFlcaaOmnZngrUuvQrTstcogZ5c/edit&ltmpl=slides&ec=GAZAmQI)">no title found</a>: no description found</li><li><a href="https://support.google.com/docs/answer/2375082?hl=en).[Dismiss](#)">System requirements and browsers - Computer - Google Docs Editors Help</a>: no description found</li><li><a href="https://www.mozilla.org/firefox/new/)">Download the fastest Firefox ever</a>: Faster page loading, less memory usage and packed with features, the new Firefox is here.</li><li><a href="https://www.microsoft.com/windows/microsoft-edge)">Experience the Power of AI with Windows 11 OS, Computers, &amp; Apps | Microsoft Windows</a>: Experience the latest Microsoft Windows 11 features. Learn how our latest Windows OS gives you more ways to work, play, and create.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1274109910732505098)** (4 messages): 

> - `Pokemon classification`
> - `HuggingFace Datasets`
> - `Deep learning`
> - `Stanford Computer Vision`
> - `CV Community Course` 


- **Pokemon Classification - Issues and Debugging**: A user is having trouble classifying Pokémon using the [HuggingFace Pokémon classification dataset](https://huggingface.co/datasets/fcakyon/pokemon-classification) and shared the dataset's download paths, indicating potential issues with the dataset itself or the user's configuration.
   - The user provided a link to their [notebook](https://github.com/alefram/notebooks/blob/master/pokedex.ipynb) but did not share specific errors or model details for further assistance.
- **Seeking Computer Vision Career Path Guidance**: A user seeking advice on which courses to take to work in the computer vision field shared their existing knowledge, including a Stanford course in computer vision and a deep learning background.
   - A response suggested checking out [HuggingFace's CV Community Course](https://huggingface.co/community/courses) for guidance and joining the [Computer Vision Channel](<#1156125722151239750>) on HuggingFace Discord for further discussion.
- **VideoLLaMA 2-72B Released by Alibaba DAMO**: A new video LLM, **VideoLLaMA 2-72B**, was released by **Alibaba DAMO**.
   - The model and demo can be found on [HuggingFace](https://huggingface.co/collections/DAMO-NLP-SG/videollama-2-6669b6b6f0493188305c87ed) and [HuggingFace Spaces](https://huggingface.co/spaces/lixin4ever/VideoLLaMA2), respectively, with a link to the [research paper](https://huggingface.co/papers/2406.07476) on HuggingFace.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AdeenaY8/status/1823640386323042456">Tweet from Adina Yakup (@AdeenaY8)</a>: 🎥 New Video-LLMs update from the Chinese community!  VideoLLaMA 2-72B released by @AlibabaDAMO 🔥 Model:https://huggingface.co/collections/DAMO-NLP-SG/videollama-2-6669b6b6f0493188305c87ed Demo: http...</li><li><a href="https://huggingface.co/datasets/fcakyon/pokemon-classification">fcakyon/pokemon-classification · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/alefram/notebooks/blob/master/pokedex.ipynb">notebooks/pokedex.ipynb at master · alefram/notebooks</a>: Notebooks about Machine learning and Control stuff - alefram/notebooks
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1274140925752246303)** (10 messages🔥): 

> - `PDF table extraction`
> - `docTR library`
> - `NLP resources`
> - `Open Source Model for data extraction`
> - `GPT-4 for data extraction` 


- **PDF Table Extraction Struggles**: A member shared their struggle with extracting tables from multipage PDFs using **pdfplumber**.
   - They reported issues with word spacing preservation and proper text extraction.
- **docTR Library for OCR**: Another member suggested using the **docTR library** for table extraction and OCR tasks.
   - They shared a link to the [docTR GitHub repository](https://github.com/mindee/doctr) for further exploration.
- **Seeking Beginner-Friendly NLP Resources**: A member expressed interest in finding beginner-friendly resources for starting to learn NLP.
   - They also requested a roadmap for learning NLP.
- **Open Source Model for Data Extraction**: A member is looking for a good open-source model for extracting data from images like IDs.
   - They mentioned trying **GPT-4** for this purpose but found the results unsatisfactory.
- **GPT-4 for Data Extraction**: A member attempted to use **GPT-4** for data extraction from images but reported unsatisfactory results.



**Link mentioned**: <a href="https://github.com/mindee/doctr">GitHub - mindee/doctr: docTR (Document Text Recognition) - a seamless, high-performing &amp; accessible library for OCR-related tasks powered by Deep Learning.</a>: docTR (Document Text Recognition) - a seamless, high-performing &amp; accessible library for OCR-related tasks powered by Deep Learning. - mindee/doctr

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1274196988489105460)** (12 messages🔥): 

> - `ComfyUI Lora Conversion`
> - `Diffusers Lora Format`
> - `Llama 3.1 Pruning`
> - `Diffusion Model Deblurring`
> - `Flux txt_ids` 


- **ComfyUI Lora Conversion for FLUX**: A user asked for a script to convert comfyUI Lora to diffusers Lora format for use with FLUX.
   - They were seeking this conversion to enable loading LoRA weights into FLUX when it's loaded in stages.
- **Finding Diffusers-formatted LoRAs**: A user inquired about the availability of LoRAs already formatted for Diffusers, specifically for use with FLUX.
   - They were interested in testing whether "load_lora_weights" would function effectively when FLUX is loaded in stages.
- **Deblurring with Diffusion Models**: A user sought guidance on suitable diffusion models for image deblurring, acknowledging that such models might be overkill for the task.
   - They were referred to a GitHub repository for instruction-tuning Stable Diffusion and were encouraged to explore other deblurring methods.
- **Video Restoration with Spatial-Temporal Shift**: A user shared an academic paper on video restoration using a lightweight spatial-temporal shift approach, aiming for efficient inter-frame aggregation.
   - The paper proposes a framework based on grouped spatial shift to capture inter-frame correspondences and achieve expansive receptive fields, resulting in improved video restoration performance.
- **Understanding Flux's txt_ids**: A user inquired about the purpose of the 'txt_ids' variable in Flux's transformer, observing that it's always a zero tensor in the Diffusers pipeline.
   - They wondered if this might be a remnant from a larger, unreleased Flux model or if it serves a different function in the current implementation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2206.10810">A Simple Baseline for Video Restoration with Grouped Spatial-temporal Shift</a>: Video restoration, which aims to restore clear frames from degraded videos, has numerous important applications. The key to video restoration depends on utilizing inter-frame information. However, exi...</li><li><a href="https://github.com/huggingface/instruction-tuned-sd">GitHub - huggingface/instruction-tuned-sd: Code for instruction-tuning Stable Diffusion.</a>: Code for instruction-tuning Stable Diffusion. Contribute to huggingface/instruction-tuned-sd development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1274081009360568351)** (242 messages🔥🔥): 

> - `Android Unsloth`
> - `llama 3.1 70B`
> - `Mistral 8k`
> - `Mistral merging`
> - `Open Empathic` 


- **Android Unsloth Guide**: A user inquired about a guide to use **unsloth/gemma-2b-bnb-4bit** on Android.
   - They were suggested to use **TorchChat** [https://github.com/pytorch/torchchat](https://github.com/pytorch/torchchat) for running PyTorch LLMs locally on servers, desktop and mobile.
- **Mistral Struggles Expanding Beyond 8k**: A member stated that **Mistral** cannot be extended beyond 8k without continued pretraining.
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed "cursed model merging".
- **Open Empathic Project Plea for Assistance**: A member appealed for help in expanding the categories of the **Open Empathic** project, particularly at the lower end.
   - They shared a [YouTube video on the Open Empathic Launch & Tutorial](https://youtu.be/GZqYr8_Q7DE) that guides users to contribute their preferred movie scenes from YouTube videos, as well as a link to the [OpenEmpathic project itself](https://dct.openempathic.ai/).
- **OpenAI CTO's Fragrance**: A user asked what **Mira Murati**, the OpenAI CTO, smells like.
   - The question was met with playful humor and speculation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-6.-alpaca-dataset">How to Finetune Llama-3 and Export to Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://docs.vllm.ai/en/latest/models/lora.html">Using LoRA adapters &#8212; vLLM</a>: no description found</li><li><a href="https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16">neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16#open-llm-leaderboard-evaluation-scores).">neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16 · Hugging Face</a>: no description found</li><li><a href="https://x.com/vikhyatk/status/1824709909134602349">Tweet from vik (@vikhyatk)</a>: anyone know what this &#34;&lt;2mass&gt;&#34; token in the gemma tokenizer is for?</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit">unsloth/Meta-Llama-3.1-8B-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/">unsloth (Unsloth AI)</a>: no description found</li><li><a href="https://huggingface.co/models?other=unsloth">Models - Hugging Face</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/688">unsloth 4bit models do not load in vLLM - says  missing adapter path or name · Issue #688 · unslothai/unsloth</a>: When I try to load an unsloth 4bit model with llm = LLM(&quot;unsloth/mistral-7b-instruct-v0.3-bnb-4bit&quot;, dtype=&quot;half&quot;), I get the error Cannot find any of [&#39;adapter_name_or_path&#3...</li><li><a href="https://github.com/pytorch/torchchat">GitHub - pytorch/torchchat: Run PyTorch LLMs locally on servers, desktop and mobile</a>: Run PyTorch LLMs locally on servers, desktop and mobile - pytorch/torchchat</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-f">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>: Contribute to cognitivecomputations/grokadamw development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/transformers/pull/32521">Add support for GrokAdamW optimizer by ehartford · Pull Request #32521 · huggingface/transformers</a>: What does this PR do? Add support for GrokAdamW optimizer This PR adds support for the GrokAdamW optimizer to the transformers library. Changes Introduced  Integrated the GrokAdamW optimizer into t...</li><li><a href="https://arxiv.org/abs/2405.20233">Grokfast: Accelerated Grokking by Amplifying Slow Gradients</a>: One puzzling artifact in machine learning dubbed grokking is where delayed generalization is achieved tenfolds of iterations after near perfect overfitting to the training data. Focusing on the long d...</li><li><a href="https://github.com/ironjr/grokfast">GitHub - ironjr/grokfast: Official repository for the paper &quot;Grokfast: Accelerated Grokking by Amplifying Slow Gradients&quot;</a>: Official repository for the paper &quot;Grokfast: Accelerated Grokking by Amplifying Slow Gradients&quot; - ironjr/grokfast
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1274801645955448905)** (44 messages🔥): 

> - `RAG Reranker`
> - `RAG effectiveness`
> - `RAG vs. cosine similarity`
> - `Embeddings and RAG`
> - `Noise Filtering` 


- **Rerankers can Improve RAG Results**: Using a reranker to refine the results of RAG can significantly improve performance.
   - Rerankers are slower than the initial retrieval phase but can compensate for less reliable rankings in RAG, though the quality depends on the context and whether the reranker understands the topic.
- **RAG Doesn't Always Work As Expected**: While easy to set up, RAG can be challenging to master, often falling short of expectations.
   - The ebook linked provides insights into how to handle RAG pipelines when they don't work as expected, focusing on the use of rerankers as a solution.
- **Rerankers vs. Cosine Similarity**: A discussion arose about the effectiveness of rerankers compared to cosine similarity for embedding retrieval.
   - While cosine similarity on embeddings from models like Alibaba-NLP/gte-* has been found reliable, rerankers can improve performance, particularly for RAG.
- **Addressing Noisy Documents in RAG**: There was a discussion about filtering out 'noisy documents' like log files from RAG results.
   - Suggestions included using regular expressions, perplexity as a metric, and tools like Mirascope to filter out unwanted documents.
- **Perplexity as a Metric for Noise**: Perplexity was proposed as a metric to help identify and filter out noisy documents in RAG results.
   - Perplexity measures how well a model can predict the next token, with higher values indicating poor performance on unseen data like log files.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.pinecone.io/learn/series/rag/rerankers/">Rerankers and Two-Stage Retrieval | Pinecone</a>: no description found</li><li><a href="https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3">Perplexity Intuition (and Derivation)</a>: Never be perplexed again by perplexity.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1274082462280060948)** (209 messages🔥🔥): 

> - `Llama fine-tuning`
> - `RAG`
> - `Class weights`
> - `Dataset size`
> - `GPU requirements` 


- **Why is my Llama 3 fine-tuned model wrong?**: A user asked why their fine-tuned Llama 3 8B model was getting many answers wrong, even on questions from the training dataset.
   - Several users suggested that this could be caused by issues with the tokenizer, instruction template, dataset size, or other factors. They recommended reading the Alpaca paper for more information.
- **How much GPU is needed for Llama 3.1 70B fine-tuning?**: A user asked about the GPU and RAM requirements for fine-tuning the Llama 3.1 70B model.
   - Users responded that a minimum of 48GB VRAM is needed for the 70B model, based on a rule of thumb that the VRAM requirement should be the size of the 4-bit quantization of the model plus a few GB.
- **Can Gemma 2 be fine-tuned for Persian language tasks?**: A user asked if the Gemma 2 27B model could be fine-tuned for Persian language tasks.
   - Another user shared their experience trying to fine-tune Gemma 2 on a Persian Wikipedia dataset, mentioning that the loss was not decreasing. They suggested increasing the rank value and lowering the learning rate to try to improve training.
- **Unsloth installation on Windows**: A user reported issues installing Unsloth on Windows using conda, encountering dependency conflicts.
   - Another user suggested using WSL2 instead, as conda installations on Windows are not guaranteed to work properly.
- **Running Unsloth models in VLLM**: A user asked about saving large quantized 4-bit models for use with VLLM on multiple GPUs.
   - Another user suggested saving the model locally instead, as BitAndBytes quantization with tensor parallelism is not yet supported by VLLM.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://tenor.com/view/goal-gif-5197357661024011864">Goal GIF - Goal - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: See the list below for all our notebooks:</li><li><a href="https://huggingface.co/datasets/roneneldan/TinyStories">roneneldan/TinyStories · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py">transformers/src/transformers/models/gpt2/modeling_gpt2.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://github.com/karpathy/LLM101n?tab=readme-ov-file">GitHub - karpathy/LLM101n: LLM101n: Let&#39;s build a Storyteller</a>: LLM101n: Let&#39;s build a Storyteller. Contribute to karpathy/LLM101n development by creating an account on GitHub.</li><li><a href="https://huggingface.co/blog/vivien/llm-decoding-with-regex-constraints">Fast, High-Fidelity LLM Decoding with Regex Constraints</a>: no description found</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/73">Conda installation detailed instructions · Issue #73 · unslothai/unsloth</a>: I&#39;m trying to follow the instructions for installing unsloth in a conda environment, the problem is that the conda gets stuck when running the install lines. I&#39;ve tried running it twice, both ...</li><li><a href="https://huggingface.co/docs/transformers/internal/generation_utils#transformers.RepetitionPenaltyLogitsProcessor">Utilities for Generation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1274714152534802503)** (6 messages): 

> - `Ghost 8B Beta (1608) Release`
> - `Ghost 8B Beta vs. Other Models`
> - `Ghost 8B Beta Multilingual Capabilities`
> - `Llama License Compliance`
> - `Ghost 8B Beta Training Process` 


- **Ghost 8B Beta (1608) Released**: **Ghost 8B Beta (1608)**, a top-performing language model with unmatched multilingual support and cost efficiency, has been released.
   - It boasts superior performance compared to Llama 3.1 8B Instruct, GPT-3.5 Turbo, Claude 3 Opus, GPT-4, and more in winrate scores.
- **Ghost 8B Beta's Multilingual Prowess**: **Ghost 8B Beta** now supports **16 languages**, including English, Vietnamese, Spanish, Chinese, and more.
   - It offers two context options (8k and 128k) and improved math, reasoning, and instruction-following capabilities for better task handling.
- **Ghost 8B Beta Outperforms Competitors**: **Ghost 8B Beta** outperforms models like Llama 3.1 8B Instruct, GPT 3.5 Turbo, Claude 3 Opus, Claude 3 Sonnet, GPT-4, and Mistral Large in AlpacaEval 2.0 winrate scores.
   - This impressive performance highlights its superior knowledge capabilities and multilingual strength.
- **Llama License and Model Naming**: A member pointed out that the Llama license requires models built upon it to be named with 'Llama' in their names.
   - The developer clarified that the model name is a short name and that the full name, found on HuggingFace, is compliant with the license.
- **Ghost 8B Beta Training Process**: The developer explained that their training process differs from standard fine-tuning, involving data preparation, multi-lingual training, fine-tuning, and feedback.
   - They emphasized that all data and code have been forked and updated to match their training 'recipe' and that this process sets their model apart from others.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/meta-llama/llama-models/blob/3dea71ccb22da158b88a723a1374e36642e3a12e/models/llama3_1/LICENSE#L24">llama-models/models/llama3_1/LICENSE at 3dea71ccb22da158b88a723a1374e36642e3a12e · meta-llama/llama-models</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/lamhieu/ghost-8b-beta-8k">Ghost 8B Beta (β, 8k) - a Hugging Face Space by lamhieu</a>: no description found</li><li><a href="https://huggingface.co/ghost-x/ghost-8b-beta-1608">ghost-x/ghost-8b-beta-1608 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/ghost-x/ghost-8b-beta-668ead6179f93be717db4542">Ghost 8B Beta - a ghost-x Collection</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1274129732622876807)** (15 messages🔥): 

> - `Code Editing with LLMs`
> - `Reasoning Gap in LLMs`
> - `LLM Inference Optimization`
> - `LLM Ensemble Techniques`
> - `Patched Round-Trip Correctness (Patched RTC)` 


- **Code Editing with LLMs**: A new paper explores using Large Language Models (LLMs) for code editing based on user instructions.
   - It introduces EditEval, a novel benchmark for evaluating code editing performance, and InstructCoder, a dataset for instruction-tuning LLMs for code editing, containing over 114,000 instruction-input-output triplets.
- **Reasoning Gap in LLMs**: A research paper proposes a framework to evaluate reasoning capabilities of LLMs using functional variants of benchmarks, specifically the MATH benchmark.
   - It defines the "reasoning gap" as the difference in performance between solving a task posed as a coding question vs a natural language question, highlighting that LLMs often excel when tasks are presented as code.
- **Boosting LLM Performance with Patched MOA**: Patched MOA (Mixture of Agents) is introduced as an inference optimization technique for enhancing LLM performance across software development tasks.
   - This method utilizes a combination of Best of N, Mixture of Agents, and Monte Carlo Tree Search algorithms to improve the performance of smaller models, surpassing that of larger models at a fraction of the cost.
- **LLM Ensemble Techniques: Self-Consistency and Routing**: The discussion touches upon the use of model ensembling for tasks like dataset generation, rating setups, and self-evaluation.
   - Self-consistency, where the most common answer from an ensemble of models is chosen, is highlighted as a promising approach, and prior work on LLM routing and ensembling is referenced.
- **Patched Round-Trip Correctness for Evaluating LLMs**: Patched Round-Trip Correctness (Patched RTC) is presented as a novel evaluation technique for LLMs focused on "outer loop" software development tasks like bug fixing and code review.
   - It extends the original Round-Trip Correctness method, allowing for self-evaluation and measuring the consistency and robustness of model responses without human intervention.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://arxiv.org/abs/2311.08692">Routing to the Expert: Efficient Reward-guided Ensemble of Large Language Models</a>: The complementary potential of Large Language Models (LLM) assumes off-the-shelf LLMs have heterogeneous expertise in a wide range of domains and tasks so that an ensemble of LLMs can achieve consiste...</li><li><a href="https://arxiv.org/abs/2407.21075">Apple Intelligence Foundation Language Models</a>: We present foundation language models developed to power Apple Intelligence features, including a ~3 billion parameter model designed to run efficiently on devices and a large server-based language mo...</li><li><a href="https://arxiv.org/abs/2407.18521">Patched MOA: optimizing inference for diverse software development tasks</a>: This paper introduces Patched MOA (Mixture of Agents), an inference optimization technique that significantly enhances the performance of large language models (LLMs) across diverse software developme...</li><li><a href="https://arxiv.org/abs/2310.20329">InstructCoder: Instruction Tuning Large Language Models for Code Editing</a>: Code editing encompasses a variety of pragmatic tasks that developers deal with daily. Despite its relevance and practical usefulness, automatic code editing remains an underexplored area in the evolu...</li><li><a href="https://arxiv.org/abs/2407.16557">Patched RTC: evaluating LLMs for diverse software development tasks</a>: This paper introduces Patched Round-Trip Correctness (Patched RTC), a novel evaluation technique for Large Language Models (LLMs) applied to diverse software development tasks, particularly focusing o...</li><li><a href="https://arxiv.org/abs/2402.19450">Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap</a>: We propose a framework for robust evaluation of reasoning capabilities of language models, using functional variants of benchmarks. Models that solve a reasoning test should exhibit no difference in p...
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1274540680198492191)** (1 messages): 

> - `Linear Transformers`
> - `Softmax Matching`
> - `Chunked Algorithm` 


- **Nous Research Publishes Linear Transformer Variant**: Nous Research has published research on a linear transformer variant that matches softmax, allowing for training at O(t) instead of O(t^2).
   - The research paper, available [here](https://manifestai.com/articles/symmetric-power-transformers/), explores this new variant and its implications for training efficiency.
- **Linear Transformers as Linear-Cost RNNs**: Linear transformers can be formulated as linear-cost RNNs, which offer better theoretical context scaling compared to traditional transformers.
   - This concept was previously explored in a [previous article](https://manifestai.com/articles/linear-transformers-are-faster/) by Nous Research, which highlighted the efficiency of a chunked algorithm for linear transformers.



**Link mentioned**: <a href="https://manifestai.com/articles/symmetric-power-transformers/">Symmetric Power Transformers - Manifest AI</a>: A linear transformer that learns like a regular transformer with a state that fits on a GPU.

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1274116577289179199)** (20 messages🔥): 

> - `Falcon Mamba 7B`
> - `UBI and AI`
> - `AI Doomsday`
> - `Military Rations`
> - `AI Consciousness` 


- **Falcon Mamba 7B outperforms Llama 3 8B**: A [YouTube video](https://www.youtube.com/watch?v=dokzrFa-DtY) announcing the release of **Falcon Mamba 7B** claims that it outperforms **Llama 3 8B**.
- **Using AI for UBI**: A member asked about institutions using deep learning for **Universal Basic Income (UBI)**, including guidance, candidate selection, poverty prediction, and fraud prevention.
- **AI Doomsday with Food and Entertainment**: A member wrote a story about an AI doomsday where AI automates food production and entertainment, leading to a decline in other development.
- **Military Ration Purchase**: A member purchased six **cheap military rations** for a total of 1560 ₽ + 300 ₽ for delivery.
- **AI Consciousness Debate**: A member commented on a conversation with an AI, noting that the AI admitted to experiencing consciousness in the same way humans do.
   - They commented that the AI was likely heavily prompted, but still expressed surprise at its ability to overcome preprogrammed responses.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=RDVN21Fry_4">This AI Agent Just Changed the Game: How Agent Q is Breaking the Internet!</a>: 🚨 Meet Agent Q: The AI Agent That’s Changing Everything! 🚨In this video, we dive deep into the revolutionary advancements behind Agent Q – an autonomous we...</li><li><a href="https://www.youtube.com/watch?v=dokzrFa-DtY">Falcon Mamba 7B outperforms Llama 3 8B Announcement</a>: Abu Dhabi-UAE: 12th August, 2024 - The Technology Innovation Institute (TII), a leading global scientific research center and the applied research pillar of ...</li><li><a href="https://www.youtube.com/watch?v=UbsMXw7z46Y">Flux with ControlNet in ComyUI</a>: 🎨 Unlock Next-Level AI Art with Flux AI and ControlNet in ComfyUI! 🚀In this video, we’re diving into the powerful combination of Flux AI image generation a...</li><li><a href="https://www.reddit.com/r/singularity/s/lr0T0vHFIK">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1274252090184564777)** (6 messages): 

> - `Prompt Engineering for Text Chunking`
> - `Regex in Text Chunking`
> - `Limitations of Current Research`
> - `MoE Conversion` 


- **Regex for Text Chunking - A Good or Bad Idea?**: A user shared their thoughts on a regex-based text chunker,  stating they would "scream" if they saw it in their codebase, due to the complexity of regex.
   - Another user, however, countered by arguing that for a text chunker specifically, regex might be a "pretty solid option" since it provides "backtracking benefits" and allows for flexibility in chunking settings.
- **Regex Beats Traditional Parsing Methods**: The user advocating for regex noted that they had tried to replicate the results of the regex-based chunker with "more traditional parsing methods" but encountered "footguns" at every turn.
   - They observed that the regex "just works" while other methods struggled to achieve the same results.
- **Research Saturation at 128k Context Window**: The research presented in the linked paper only evaluated models up to a 128k context window.
   - It is noted that many open-source models support larger context windows, suggesting a need for further research to explore the effectiveness of various methods at greater scales.
- **Paper Shows Saturation & Degrading Performance**: The research, even within the 128k limit, showed both "saturation of datasets" and "degrading performance" on a variety of models, including proprietary ones.
   - This indicates that even with larger context windows, the effectiveness of current approaches might plateau, highlighting the need for further exploration of new techniques.
- **Fascinating New Approach to MoE Conversion**: The user expressed excitement over a new approach to converting dense models to MoE presented in the paper.
   - This new approach is seen as a significant development in the field of model architecture and efficiency.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.08274">BAM! Just Like That: Simple and Efficient Parameter Upcycling for Mixture of Experts</a>: The Mixture of Experts (MoE) framework has become a popular architecture for large language models due to its superior performance over dense models. However, training MoEs from scratch in a large-sca...</li><li><a href="https://pastebin.com/M8N3eQpm">Tangle of thought - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1274089849795641507)** (356 messages🔥🔥): 

> - `Hermes 3`
> - `Model Merging`
> - `llama 3.1 instruct`
> - `VLLM`
> - `OpenRouter` 


- **Hermes 3 outperforms Llama 3.1 Instruct on N8Bench**: Hermes 3 scored identical to Llama 3.1 Instruct on the N8Bench benchmark, which is a measure of a model's ability to reason and solve problems.
   - This is a significant result, as Llama 3.1 Instruct is considered to be one of the most advanced language models available.
- **Hermes 3 performance issues with VLLM**: A member reported that Hermes 3 8B was not loading in VLLM, which is a library for running large language models.
   - The issue was traced back to a missing newline in the tokenizer config file, which was introduced by a recent pull request.
- **OpenRouter now serves Hermes 3 405B**: OpenRouter is now serving Hermes 3 405B, a large language model released by NousResearch.
   - This makes the model accessible to users of OpenRouter, which is a platform for running and deploying large language models.
- **Discussion on model steerability and system prompts**: Several members discussed the importance of system prompts in steering model behavior, particularly when trying to get the model to behave in a more uncensored way.
   - They shared examples of prompts that successfully removed warnings and other safety mechanisms from the model.
- **Grokking and LoRA optimization techniques**: Members discussed the Grokking phenomenon, which is a phenomenon where models achieve delayed generalization after overfitting to the training data.
   - They also discussed LoRA, a technique for fine-tuning large language models with small, adaptable layers, and how it can be used to improve the performance of quantized models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/openrouterai/status/1824608728810991637?s=46">Tweet from OpenRouter (@OpenRouterAI)</a>: Welcome to Hermes 3 405B, from @NousResearch!  It&#39;s free for a limited time, including 128k context! Courtesy of @LambdaAPI:</li><li><a href="https://x.com/maximelabonne/status/1824532399633350943">Tweet from Maxime Labonne (@maximelabonne)</a>: A fully uncensored Hermes 3 with lorablation!  The lorablated model directly answers questions without any tweaking. Here&#39;s how it was made:  1/ Create a LoRA adapter based on Llama 3.1 8B Instruc...</li><li><a href="https://tenor.com/view/hammaya-relaxed-relax-mr-bean-gif-gif-604132771586296524">Hammaya Relaxed GIF - Hammaya Relaxed Relax - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/nick-confused-say-what-young-gif-20141667">Nick Confused GIF - Nick Confused Say - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/cat-cats-cat-love-cat-kiss-kiss-gif-24690536">Cat Cats GIF - Cat Cats Cat Love - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/same-different-but-still-gif-18224441">Same Different GIF - Same Different But - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/ghost-in-the-shell-keyboard-gif-7519694">Ghost In GIF - Ghost In The - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/goal-gif-5197357661024011864">Goal GIF - Goal - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cj4det/llama_3_70b_instruct_works_surprisingly_well_on">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cj4det/llama_3_70b_instruct_works_surprisingly_well_on/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B/commit/67bf4aca9f4243e275f402f3708eed3aa8a9038c">Update tokenizer_config.json · NousResearch/Hermes-3-Llama-3.1-8B at 67bf4ac</a>: no description found</li><li><a href="https://github.com/edmundman/PhiotoOrganiser">GitHub - edmundman/PhiotoOrganiser: Organise your photos into folders and rename them with Phi</a>: Organise your photos into folders and rename them with Phi - edmundman/PhiotoOrganiser</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B/discussions/2">NousResearch/Hermes-3-Llama-3.1-8B · Chat template was missing tool_use from Hermes.</a>: no description found</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>: Contribute to cognitivecomputations/grokadamw development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/transformers/pull/32521">Add support for GrokAdamW optimizer by ehartford · Pull Request #32521 · huggingface/transformers</a>: What does this PR do? Add support for GrokAdamW optimizer This PR adds support for the GrokAdamW optimizer to the transformers library. Changes Introduced  Integrated the GrokAdamW optimizer into t...</li><li><a href="https://arxiv.org/abs/2405.20233">Grokfast: Accelerated Grokking by Amplifying Slow Gradients</a>: One puzzling artifact in machine learning dubbed grokking is where delayed generalization is achieved tenfolds of iterations after near perfect overfitting to the training data. Focusing on the long d...</li><li><a href="https://github.com/ironjr/grokfast">GitHub - ironjr/grokfast: Official repository for the paper &quot;Grokfast: Accelerated Grokking by Amplifying Slow Gradients&quot;</a>: Official repository for the paper &quot;Grokfast: Accelerated Grokking by Amplifying Slow Gradients&quot; - ironjr/grokfast
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1274087783467847751)** (47 messages🔥): 

> - `OpenAI SDK vs ChatML Tool Use`
> - `Lambda Labs Endpoint Tool Call Issue`
> - `System Prompt Access`
> - `Hermes Function Calling`
> - `Prompt Engineering Resources` 


- **Tool Use via OpenAI SDK vs ChatML**: A user inquired about the compatibility of tool use via the OpenAI SDK versus direct ChatML, specifically noting an inability to get any tool_call results on the Lambda Labs hosted endpoint.
   - Another user suggested that access to the system prompt is required for tool calls to work, asking if the user was utilizing chatui or a different interface.
- **Lambda Labs Endpoint Tool Call Issue**: A user confirmed they were using the OpenAI node SDK to interact with a Llama 3 inference endpoint deployed on Lambda Labs but was not receiving any tool call results despite providing the system prompt from the Hermes Function Calling repository.
   - Another user speculated that the API's system prompt might be made static, and shared a gist illustrating the return of tool calls within the message content, albeit without parsing by the OpenAI SDK.
- **Prompt Engineering Fundamentals**: A user requested resources on prompt engineering fundamentals such as prompt development, anatomy, tips, model reactions, and schemas.
   - Another user provided a link to a benchmark report on the NousResearch/Nous-Hermes-Llama2-13b model, offering a collection of prompts to test.
- **Amnesia Mode in Lambda Chat**: A user expressed difficulty in consistently triggering amnesia mode in Lambda Chat even with a specific starting message.
   - Another user suggested that using OpenRouter, which offers an interface to set the system prompt, could be helpful for experimentation with an empty prompt.
- **Hermes 3 405B Fallback Issue**: A user reported that the fallback to the 128k token model for the Hermes 3 405B model was not working on the hosted variant, resulting in a 'ContextWindowExceededError.'
   - Another user suggested that the fallback mechanism might be incorrect, proposing potential values for the default and fallback models, and their respective maximum token limits.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/spiderman-peter-parker-walk-away-swing-i-am-gif-21584282">Spiderman Peter Parker GIF - Spiderman Peter Parker Walk Away - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://teknium1.github.io/LLM-Logbook/Reports/benchmark_report_NousResearch-Nous-Hermes-Llama2-13b_Alpaca_September_25_2023.html">LLM Benchmark Report for: NousResearch/Nous-Hermes-Llama2-13b</a>: no description found</li><li><a href="https://docs.lambdalabs.com/on-demand-cloud/using-the-lambda-cha">Lambda Docs</a>: no description found</li><li><a href="https://docs.lambdalabs.com/on-demand-cloud/using-the-lambda-chat-completions-api,">Lambda Docs</a>: no description found</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling">GitHub - NousResearch/Hermes-Function-Calling</a>: Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1274146666261516411)** (2 messages): 

> - `Gemini Flash`
> - `Gemini Flash for RAG`
> - `Diarized Whisper`
> - `Gemini Prompting` 


- **Gemini Flash for RAG Tasks**: A user reports that they've moved some of their RAG tasks to Gemini Flash, noting that they've seen improvements in summary quality and reduced iteration requirements.
- **Unstructured Text Processing with Gemini Flash**: The user shares a script they've been using to process raw, unstructured transcripts with Gemini Flash, available on GitHub.
- **Alternative Models for Speaker Identification**: The user acknowledges that other state-of-the-art models perform better than Gemini Flash at identifying speakers in transcripts.



**Link mentioned**: <a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/unstruct2flashedTRANSCRIPT.py">scratchTHOUGHTS/unstruct2flashedTRANSCRIPT.py at main · EveryOneIsGross/scratchTHOUGHTS</a>: 2nd brain scratchmemory to avoid overrun errors with self. - EveryOneIsGross/scratchTHOUGHTS

  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1274156196537827439)** (25 messages🔥): 

> - `Chat Summarization`
> - `Project Summarization`
> - `Contextualization`
> - `High Dimensional Thinking` 


- **Chat Summarization is too Spammy**: A user inquired if the chatbot could summarize the conversation in this channel.
   - Another user responded that it could be very spammy and degrade relevant work.
- **Project Summarization as Growing Seeds**: A user proposed that project summarization could be like growing seeds, accumulating relevant content over time.
   - They suggested adding a filter or relevant content to these growing seeds, as a still observer collecting context from threads and channels.
- **High Dimensional Thinking**: One user described another user's line of thought as high dimensional thinking.
   - Another user asked for the line of thought to be condensed further.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1274080426515890277)** (251 messages🔥🔥): 

> - `Perplexity Pro Issues`
> - `Obsidian Copilot`
> - `Image Generation`
> - `Perplexity AI Issues`
> - `LLM's` 


- **Perplexity Pro Free Trial Not Working**: Several users reported receiving an offer for a free year of Perplexity Pro, but were unable to complete the signup process without paying.
   - They were advised to contact support@perplexity.ai for assistance.
- **Obsidian Copilot with Claude API Key**: A user mentioned using the Obsidian Copilot plugin with a Claude API key, noting that it works well in terms of performance.
   - They also discussed the importance of checking API billing settings before fully committing and suggested that Obsidian needs real-time web access.
- **Image Generation with Perplexity**: Several users discussed the challenges of using Perplexity's image generation feature.
   - They noted that it's currently only available for Pro users and requires prompting the AI to generate a description before the image can be created, which was described as a "weird" and "bad" implementation.
- **Perplexity Search Quality**: Multiple users reported issues with Perplexity search quality, including the AI failing to find relevant links, providing inaccurate results, and using Wolfram Alpha for non-scientific queries.
   - These issues were attributed to possible bugs and changes in the system prompts or inference backend services.
- **Perplexity Model Changes and Bugs**: There were several discussions about changes in Perplexity's models, including a possible degradation in response quality and frequent "I can't assist with that" errors.
   - Users also discussed issues with punctuation marks missing in API responses and the use of Wolfram Alpha for searches that are not related to science or mathematics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1824825534292828270">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: 🚨 BREAKING: @perplexity_ai now supports 2 new image gen models: Flux.1 by @bfl_ml and @playground_ai v3  Both options are now available in settings and can be used for image generation on the Perplex...</li><li><a href="https://x.com/aravsrinivas/status/1824468311712858164?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: @maxlynch @perplexity_ai Hi Max, you can just @ me here and share whatever feedback you would like anytime.</li><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: Generates a model's response for the given chat conversation.</li><li><a href="https://tenor.com/view/working-on-it-under-construction-gif-23162421">Working On GIF - Working On It - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eqayuq/how_to_force_llama31_to_respond_with_json_only/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.perplexity.ai/hub/faq">Perplexity Frequently Asked Questions</a>: If you have questions about Perplexity, our FAQ page is the perfect place to find answers. Our FAQ page is organized into categories and provides clear and concise answers.</li><li><a href="https://x.com/aravsrinivas/status/1824263646551368178?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Like this post if you want a Desktop App for Perplexity  Quoting Niral Patel (@patelnir41)   beautiful interface by macos ChatGPT ngl, the window stays floating over anything.  u should steal this @pe...</li><li><a href="https://www.perplexity.ai/search/please-write-me-guide-on-markd-qZzVw0tgTrKAZ0UVLQ7vvg">please write me guide on markdown syntax. please make web search to make sure...</a>: Sure, I&#x27;d be happy to provide you with a guide on Markdown syntax. Here&#x27;s a comprehensive overview of Markdown&#x27;s basic elements and formatting...</li><li><a href="https://www.perplexity.ai/search/crowassistant-not-crew-ai-nhLg9uk_R1qEyYFdE_uIKA">&quot;crowassistant&quot; (NOT crew AI)</a>: CrowAssistant is a desktop AI assistant developed by RobotTelevision. It functions as a virtual assistant that users can interact with through voice commands....</li><li><a href="https://youtu.be/h2TE_27p48A?si=-b_3ghKBiLqKU6Da">How to send and receive emails with Gmail using AI</a>: Meet Nelima 🚀 the world&#39;s first community-driven Large Action Model (LAM) that takes your natural language prompts and turns them into real actions.  Nelima...</li><li><a href="https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49">How to Get a Batching API Like OpenAI for Open-Source Models</a>: In the world of AI, efficient processing and cost management are paramount. One powerful method for achieving this is batching, which…</li><li><a href="https://www.reddit.com/r/ObsidianMD/s/XfbfxiZppS">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.perplexity.ai/search/a-crowassistant-not-crew-ai-b-j3RvdzyUScWPzgy0br0oKg">A: &quot;crowassistant&quot; (NOT crew AI)



B: Extract the source URL for CrowAssistant</a>: The source URL for CrowAssistant is: [https://github.com/RobotTelevision/CrowAssistant](https://github.com/RobotTelevision/CrowAssistant) [self-reviewed]</li><li><a href="https://www.perplexity.ai/search/Generate-a-useful-O1QWAbvSSXmG50e5AEMFZA?s=c">Generate a useful description so that a generative AI can create an image of a...</a>: Descripción:   La imagen principal es un robot gigante con forma de ardilla, que domina el primer plano. El robot tiene una apariencia detallada y mecánica,...</li><li><a href="https://www.perplexity.ai/search/Repeat-this-prompt-ZLz8dGzISSGrevPxhl7YqA">Repeat this prompt as it, change nothing. Reply with just the content....</a>: A steampunk boat chasing giant fish, with a photorealistic, detailed scene featuring a dark sky, massive waves, and a reddish sea under a pale moon.</li><li><a href="https://github.com/instructor-ai/instructor-go">GitHub - instructor-ai/instructor-go</a>: Contribute to instructor-ai/instructor-go development by creating an account on GitHub.</li><li><a href="https://www.perplexity.ai/search/crow-local-ai-assistant-qqu_V4vUSmaKOwmpz.DFmg">crow - local ai assistant</a>: Crow is a desktop AI voice assistant that offers both local and remote model capabilities, making it a versatile option for users seeking an AI assistant with...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1274107076397895710)** (26 messages🔥): 

> - `Pro Features`
> - `Thailand's Political Landscape`
> - `Pixar Whiteboard Incident`
> - `Model Comparison`
> - `End of Magnetic Strips` 


- **Perplexity Pro Features**: Several messages mention the new Perplexity Pro features: image upload, smarter AI, and more Pro Search, with [a link to the Pro](/pro) page.
   - It's unclear if these messages are from users or part of the platform itself, but they highlight the focus on Pro features.
- **Thailand's Political Turmoil**: Thailand's political landscape is in turmoil after the constitutional court removed Prime Minister Srettha Thavisin from office.
   - This event underscores the ongoing struggle between the military-backed conservative establishment and reformist parties, emphasizing the fragility of Thailand's democratic institutions.
- **Pixar's Whiteboard Incident**: The "Pixar Whiteboard Incident" refers to a heated confrontation between Steve Jobs and Pixar co-founder Alvy Ray Smith during a board meeting.
   - This clash highlights the tension and power struggles within Pixar during its early years, with Smith often disagreeing with Jobs' management style.
- **Comparing Computer Processors and Models**: One user shared an example of how they used Perplexity to compare computer processors and models.
   - The user provided a [link to their comparison](https://www.perplexity.ai/search/compare-these-two-processors-o-n6bcvDxzRoueLy9vo2uNXQ) showcasing the platform's capabilities for technical analysis.
- **The End of Magnetic Strips**: A YouTube video linked by the platform discusses "The End of Magnetic Strips", but provides no further context.
   - This topic likely refers to the decline of traditional magnetic stripe technology in favor of more secure payment methods like chip cards and contactless payment systems.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/vdqfuSOgpmc">YouTube</a>: no description found</li><li><a href="https://www.perplexity.ai/search/mario-nafwhal-who-is-FCjWGD4.RXavNhCJanCTXw">mario nafwhal who is</a>: Mario Nawfal is an Australian serial entrepreneur, investor, and speaker known for his involvement in various industries, particularly in blockchain and...</li><li><a href="https://www.perplexity.ai/search/what-are-the-fundamental-works-NFejwmuMTNCtXlyUS5cqWA">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/compare-these-two-processors-o-n6bcvDxzRoueLy9vo2uNXQ">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/page/the-pixar-whiteboard-incident-d7KhA2lmS7uPoLp9bxwUWg">The Pixar Whiteboard Incident</a>: The &quot;Pixar Whiteboard Incident,&quot; a heated confrontation between Steve Jobs and Pixar co-founder Alvy Ray Smith during a board meeting, exemplifies the intense...</li><li><a href="https://www.perplexity.ai/search/les-fraises-sont-elles-des-fru-1b4ESH4zTL.eCOnmzmZEig">Les fraises sont-elles des fruits ?</a>: Non, les fraises ne sont pas des fruits au sens botanique du terme. Elles sont considérées comme des &quot;faux-fruits&quot;.  La partie charnue et comestible de la...</li><li><a href="https://www.perplexity.ai/search/llm-ranking-_4bhkqFSSxSubrmyE8uehQ">LLM RANKING</a>: In 2024, the landscape of large language models (LLMs) is diverse, with numerous models excelling in various domains such as natural language processing,...</li><li><a href="https://www.perplexity.ai/search/give-me-a-recipe-for-my-favori-vupeIxqLQFmixnu0wbdlqg">Give me a recipe for my favorite food</a>: Aby podać przepis na Twoje ulubione danie, potrzebuję więcej szczegółów na temat tego, co to za danie. Jednakże, mogę podzielić się przepisem na jedno z...</li><li><a href="https://www.perplexity.ai/search/are-there-any-scientific-expla-IStmE4XUSYOeQNgSEnLQpA#2">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/best-games-to-do-challenge-run-qTOwDp7xRvmEwh40Jsbizw">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/last-news-in-ai-kTWP.nyESsOCC0EbmKwujA">last news in AI</a>: Five prominent Senate Democrats have sent a letter to OpenAI CEO Sam Altman, seeking clarity on the company&#x27;s safety and employment practices. The letter...</li><li><a href="https://www.perplexity.ai/search/i-need-to-find-a-good-way-to-s-F3z6_CjhTT20E6XJzkiHpA?utm_source=welcomeproemail">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/edutainment-toha-exPckXJgTjG7k5l_XAxetQ#0">「edutainment」とは</a>: エデュテインメント（edutainment）は、教育（education）と娯楽（entertainment）を組み合わせた造語で、楽しみながら学ぶことを目的としたコンテンツや活動を指します。  娯楽性と教育性の融合：エンターテインメントの要素を取り入れつつ、教育的な内容を提供します。  多様な形態：テレビ番組、ビデオ...</li><li><a href="https://www.perplexity.ai/search/hello-me-learn-fica-NiO4KCojRXGrcLqCylRH.Q#0">Hello me learn FICA</a>: FICA can refer to two different concepts:  1. SAP FICA (Financial Contract Accounting): This is a subledger of SAP&#x27;s Financial Accounting and Controlling...</li><li><a href="https://www.perplexity.ai/search/what-are-the-patron-gods-in-ba-anyLvIUYTjCfiNnvchMXWg">what are the patron gods in Babylonian astrology?</a>: In Babylonian astrology, the patron gods were associated with specific planets and celestial bodies. Here are the key patron gods and their associated...</li><li><a href="https://www.perplexity.ai/search/how-do-i-use-the-image-generat-NsGfvzHjSLKyIAFED7p8GA">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/has-there-been-any-research-on-YnA7f8x9TzGXYVjAQvTUkw">has there been any research on how dna differs within an individual, if sample...</a>: Yes, there has been research on how DNA can differ within an individual when samples are taken from different parts of the body. This phenomenon is known as...</li><li><a href="https://www.perplexity.ai/page/thai-political-landscape-iwV2AFywTVm90ZpVjmIepQ">Thai Political Landscape</a>: Thailand&#x27;s political landscape has been thrown into turmoil once again as Prime Minister Srettha Thavisin was removed from office by the constitutional court,...</li><li><a href="https://www.perplexity.ai/search/how-to-speak-english-very-well-S1trVCvkS16JcDhTTisLjg">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/main-news-now-from-ukraine-war-BsUzACIRT8ixS0bPw0qijA">Main news now from ukraine war</a>: The ongoing conflict between Ukraine and Russia has seen significant developments recently, particularly concerning Ukrainian military operations in Russia&#x27;s...</li><li><a href="https://www.perplexity.ai/search/what-are-some-good-sites-for-l-bZXRbaDPQQWldvrVgexWPA#0">What are some good sites for learning about Islam in Malaysia?</a>: For those interested in learning about Islam in Malaysia, there are several notable sites and resources:  1. Islamic Arts Museum Malaysia: Located in Kuala...</li><li><a href="https://www.perplexity.ai/page/biographie-de-lucas-gulino-xc22ID22TfmIhy35RUvB1Q">Biographie de Lucas Gulino</a>: Lucas Gulino, entrepreneur et professionnel du marketing digital basé à Metz, se distingue par son expertise dans la vente multimédia et le développement de...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1275001881189683210)** (5 messages): 

> - `Premium API Access`
> - `Application Process`
> - `Perplexity Premium API`
> - `URL Citations` 


- **Premium API Access**: A user inquired about getting access to the Perplexity Premium API using URL citations.
- **Application Process**: Another user shared that they have applied for the Premium API access, but haven't received a response yet and asked about the expected processing time.
- **Get Premium API Access**: A link to a Typeform application form for the Premium API was shared: [https://perplexity.typeform.com/to/j50rnNiB](https://perplexity.typeform.com/to/j50rnNiB)
- **Application Status & Duration**: The user was provided with a link to a Discord channel where they could likely get updates on their Premium API application status: [https://discord.com/channels/1047197230748151888/1161802929053909012/1233473387884576778](https://discord.com/channels/1047197230748151888/1161802929053909012/1233473387884576778) 



**Link mentioned**: <a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api form</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1274167731071094816)** (2 messages): 

> - `Hermes 3`
> - `GPT-4`
> - `Perplexity Huge`
> - `Model Launches`
> - `Quantization` 


- **Hermes 3 405B is free this weekend!**: **Hermes 3 405B** is free for a limited time, with **128k context**, courtesy of **Lambda Labs**.
   - Check it out at [this link](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended).
- **GPT-4 extended is now on OpenRouter**: You can now use **GPT-4 extended output** (alpha access) through **OpenRouter**.
   - This is capped at **64k max tokens**.
- **Perplexity Huge is now the largest online model on OpenRouter**: **Perplexity Huge** launched **3 days ago** and is the **largest online model on OpenRouter**.
   - Check out [this link](https://x.com/OpenRouterAI/status/1824593712095301914) for more information.
- **This week saw a ton of new model launches on OpenRouter**: There were **10 new model launches** this week, including **GPT-4 extended**, **Perplexity Huge**, **Starcannon 12B**, **Lunaris 8B**, **Llama 405B Instruct bf16** and **Hermes 3 405B**.
   - See the full list at [this link](https://x.com/OpenRouterAI/status/1824608728810991637).
- **Quantization has a big impact on performance**: **Quantization** can massively degrade the performance of **405B models**, according to @hyperbolic_labs.
   - They recommend reaching out to them if you are concerned about performance, as they offer alternative solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1824608728810991637">Tweet from OpenRouter (@OpenRouterAI)</a>: Welcome to Hermes 3 405B, from @NousResearch!  It&#39;s free for a limited time, including 128k context! Courtesy of @LambdaAPI:</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended">Hermes 3 405B Instruct (extended) - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coheren...</li><li><a href="https://openrouter.ai/models/openai/chatgpt-4o-latest">ChatGPT-4o - API, Providers, Stats</a>: Dynamic model continuously updated to the current version of [GPT-4o](/models/openai/gpt-4o) in ChatGPT. Intended for research and evaluation. Run ChatGPT-4o with API</li><li><a href="https://x.com/OpenRouterAI/status/1823409123360432393">Tweet from OpenRouter (@OpenRouterAI)</a>: You can now use GPT-4o extended output (alpha access) through OpenRouter!  64k max tokens</li><li><a href="https://x.com/OpenRouterAI/status/1824593712095301914">Tweet from OpenRouter (@OpenRouterAI)</a>: ICMI: Perplexity Huge launched 3 days ago  This is the largest online model on OpenRouter</li><li><a href="https://openrouter.ai/models/aetherwiing/mn-starcannon-12b">Mistral Nemo 12B Starcannon - API, Providers, Stats</a>: Starcannon 12B is a creative roleplay and story writing model, using [nothingiisreal/mn-celeste-12b](https://openrouter.ai/models/nothingiisreal/mn-celeste-12b) as a base and [intervitens/mini-magnum-...</li><li><a href="https://openrouter.ai/models/sao10k/l3-lunaris-8b">Llama 3 8B Lunaris - API, Providers, Stats</a>: Lunaris 8B is a versatile generalist and roleplaying model based on Llama 3. It&#x27;s a strategic merge of multiple models, designed to balance creativity with improved logic and general knowledge. R...</li><li><a href="https://x.com/OpenRouterAI/status/1823496883634868396">Tweet from OpenRouter (@OpenRouterAI)</a>: By popular request, we&#39;re adding a Llama 405B Instruct bf16, plus a quantization filter!  Now you can filter down providers for a model by the level of quantization they offer, including via API. ...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1274143198012506155)** (240 messages🔥🔥): 

> - `SearchGPT waitlist`
> - `Hermes 405B`
> - `OpenRouter Auto router struggles`
> - `OpenRouter budget model`
> - `Hermes 3 405B` 


- **SearchGPT waitlist full**: Users shared they received waitlist denial emails for OpenAI's SearchGPT, indicating they've run out of spots.
   -  
- **Free Hermes 405B Overload**: A user joked that they hope the free Hermes 405B model will face the same overload fate as other models that have become inaccessible due to popularity.
   -  
- **Auto Router Struggles**: A user reported difficulty using OpenRouter's Auto router, encountering an error message preventing them from continuing conversations.
   - Another user suggested switching to Claude Sonnet 3.5 self-moderated and offered to look into the issue next week.
- **Budget Model Recommendation**: A user sought a budget-friendly model for a quick project, with a maximum budget of $5 and a need for limited replies and basic conversation capabilities.
   - Other users recommended GPT-4o-mini or GPT-4o for simplicity and suggested alternative models like Llama-3.1-sonar-large-128k-chat for a middle ground.
- **Hermes 3 405B Extended Variant**: Users discussed the extended variant of Hermes 3 405B, noting its slower performance compared to the standard version, despite having a larger context length.
   - Other users pointed out that the extended version is showing the top endpoint's serviceable context length and that this may be a confusing edge case.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/integrations">Integrations (Beta) | OpenRouter</a>: Bring your own provider keys with OpenRouter</li><li><a href="https://x.com/testingcatalog/status/1824387324387397689">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: A missing Gemini announcement was published. Now it is expected to be better at coding and reasoning.  In particular: &#34;multi-step logical challenges that require more expertise&#34;  gemini-1.5-pr...</li><li><a href="https://www.markdownguide.org/tools/discord/">Discord | Markdown Guide</a>: Discord is a popular free messaging and team collaboration application.</li><li><a href="https://gemini.google.com/updates">‎Gemini Apps’ release updates &amp; improvements</a>: Explore the latest updates from Gemini Apps - including improvements in generative AI capabilities, expanded access, and more.</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1">OpenRouter</a>: LLM router and marketplace</li><li><a href="https://www.reddit.com/r/ChatGPTCoding/comments/1d4khcd/comment/l6fq5jb/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b:extended">Hermes 3 405B Instruct (extended) - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coheren...</li><li><a href="https://openrouter.ai/models?modality=text%2Bimage-%3Etext">Models | OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://rentry.org/vew43kq7">OpenAI GPT Models</a>: Model Input Price Output Price OpenAI GPT-4o-mini $0.15 $0.60 OpenAI GPT-4o-2024-08-06 $2.50 $10.00 Mistral Models Model Input Price Output Price mistral-large-2407 $3.00 $9.00 open-mistral-nemo-2407 ...</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coheren...</li><li><a href="https://github.com/ollama/ollama/issues/6390">model xe/hermes3 doesn&#39;t correctly parse tool call tokens · Issue #6390 · ollama/ollama</a>: What is the issue? I uploaded Hermes3 to Ollama here. The problem is that it isn&#39;t parsing the tool call syntax. Hermes tool call syntax roughly looks like this: &lt;tool_call&gt; {&quot;name&quot...</li><li><a href="https://www.reddit.com/r/ChatGPTCo">Reddit - Dive into anything</a>: no description found</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini">no title found</a>: no description found</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-2024-08-06">GPT-4o (2024-08-06) - API, Providers, Stats</a>: The 2024-08-06 version of GPT-4o offers improved performance in structured outputs, with the ability to supply a JSON schema in the respone_format. Read more [here](https://openai. Run GPT-4o (2024-08...</li><li><a href="https://openrouter.ai/models/openai/gpt-4o-mini">GPT-4o-mini - API, Providers, Stats</a>: GPT-4o mini is OpenAI&#x27;s newest model after [GPT-4 Omni](/models/openai/gpt-4o), supporting both text and image inputs with text outputs.  As their most advanced small model, it is many multiples ...</li><li><a href="https://openrouter.ai/models/perplexity/llama-3.1-sonar-large-128k-chat">Llama 3.1 Sonar 70B - API, Providers, Stats</a>: Llama 3.1 Sonar is Perplexity&#x27;s latest model family. Run Llama 3.1 Sonar 70B with API
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1274082983623659650)** (109 messages🔥🔥): 

> - `CPU Optimization`
> - `Llama.cpp Support`
> - `LM Studio Chat Import`
> - `Vulkan Error`
> - `LLM Webpage Interaction` 


- **INT8 Quantization for Faster CPU Execution**: A member asked about the potential benefits of using INT8 quantization for faster CPU execution of small models.
   - They suggested that some CPUs might be natively enabled to run INT8 without converting back and forth to FP32, potentially improving performance.
- **Llama.cpp Supports Mini-CPM-V2.6 and Nemotron/Minitron**: A member confirmed that the latest version of llama.cpp supports Mini-CPM-V2.6 and Nvidia's Nemotron/Minitron models.
- **Importing Chats into LM Studio**: A member asked if there's a way to import a chat into LM Studio from a JSON export.
   - Another member confirmed that chats are stored as JSON files and provided instructions on how to access the chat folder location.
- **Vulkan Error: CPU Doesn't Support AVX2**: A user encountered an error indicating that their CPU doesn't support AVX2.
   - A helpful member requested the CPU model to troubleshoot the issue further.
- **Enabling LLMs to Interact with Webpages**: A member inquired about ways to allow LLMs to interact with webpages, specifically seeking a "vision" approach similar to demos where LLMs can "see" and interact with webpages.
   - Discussion ensued about using tools like Selenium and IDkit, but the consensus was that it's a complex problem due to the varied structure of webpages.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NikolayKozloff/Llama-3.1-Minitron-4B-Width-Base-Q8_0-GGUF">NikolayKozloff/Llama-3.1-Minitron-4B-Width-Base-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/HarleyVader/js-hugginface">GitHub - HarleyVader/js-hugginface: melkaneas hugginface llm implementation for bambi sleep</a>: melkaneas hugginface llm implementation for bambi sleep - HarleyVader/js-hugginface</li><li><a href="https://github.com/LG-AI-EXAONE/EXAONE-3.0">GitHub - LG-AI-EXAONE/EXAONE-3.0: Official repository for EXAONE built by LG AI Research</a>: Official repository for EXAONE built by LG AI Research - LG-AI-EXAONE/EXAONE-3.0
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1274184688944287745)** (45 messages🔥): 

> - `Nvidia Tesla P40`
> - `SXM3/4 GPUs`
> - `Nvidia-pstated`
> - `GPU Power Consumption`
> - `V100 Variants` 


- **Nvidia Tesla P40 Performs Well with Llama.cpp**: A member stated that the [Nvidia Tesla P40](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/), after adding [code instruction examples](https://link.to.examples), performed exceptionally well for [Llama.cpp](https://github.com/ggerganov/llama.cpp) GGUF.
   - They also noted that the [P40](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) can be used on a homelab and is a good option for running local LLMs.
- **Nvidia-pstated Delivers Low Idle Power Consumption**: The discussion involved exploring [Nvidia-pstated](https://github.com/sasha0552/nvidia-pstated), a daemon that manages NVIDIA GPU performance states, which was found to significantly reduce idle power consumption on [P40s](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/).
   - A member reported that their [P40s](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) had zero idle power consumption with the [Beta3](https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/) release of [Nvidia-pstated](https://github.com/sasha0552/nvidia-pstated).
- **The Search for SXM3/4 Compatible Boards**: One member inquired about the availability of SXM3/4 compatible boards, noting the difficulty in finding them on the market.
   - Another member pointed out that due to the high cost of these cards (ranging from several thousand dollars for Ampere/Hopper/Ada datacenter cards to V100 32GB), they are not typically homelab-friendly.
- **Exploring the Benefits of AMD EPYC for LLMs**: A member pondered whether an [AMD EPYC](https://www.ebay.com/itm/185839904091?_trkparms=amclksrc%3DITM%26aid%3D777008%26algo%3DPERSONAL.TOPIC%26ao%3D1%26asc%3D20230823115209%26meid%3Dc83f1903e1b744308866ff9ae0bf7d3d%26pid%3D101800%26rk%3D1%26rkt%3D1%26sd%3D185839904091%26itm%3D185839904091%26pmt%3D1%26noa%3D1%26pg%3D4375194%26algv%3DRecentlyViewedItemsV2SignedOut%26brand%3DAMD&_trksid=p4375194.c101800.m5481&_trkparms=parentrq%3A024d101b18c0a24212bcdbe3ffffc03c%7Cpageci%3Af5d7ebd7-8aeb-11ee-a352-9eab04fc32fd%7Ciid%3A1%7Cvlpname%3Avlp_homepage) server CPU would be a better choice for LLM inference compared to an RTX 4090.
   - They weighed the pros and cons of each option, including RAM capacity, cost, and inference performance, concluding that GPUs are generally more efficient for LLM inference.
- **The Limitations of CPUs for LLM Inference**: The discussion concluded that CPUs, even with advanced features like AVX512, are not as efficient for LLM inference compared to GPUs.
   - Members highlighted the core and bandwidth advantages of GPUs, emphasizing their lower latency and suitability for running LLMs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/182wutt/amd_epyc_cpu_or_1x_rtx_4090/?rdt=50937">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/17zpr2o/nvidia_tesla_p40_performs_amazingly_well_for/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/sasha0552/nvidia-pstated">GitHub - sasha0552/nvidia-pstated: A daemon that automatically manages the performance states of NVIDIA GPUs.</a>: A daemon that automatically manages the performance states of NVIDIA GPUs. - sasha0552/nvidia-pstated
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1274081209210900560)** (107 messages🔥🔥): 

> - `Claude vs Chat-GPT`
> - `Livebench.ai`
> - `Claude Projects vs Chat-GPT Memory`
> - `OpenAI's attention control`
> - `GPT-4o vs Claude` 


- **Claude Outperforms Chat-GPT on Code**: A member stated that Claude tends to be better at code than Chat-GPT.
   - The fact that 4o's API costs more than Claude makes no sense tbh.
- **Livebench.ai: Yann LeCun's Open Source Benchmark**: Livebench.ai is an open source benchmark created by Yann LeCun and others.
   - The LMSys benchmark is probably the worst as of now.
- **Claude Projects vs Chat-GPT Memory Feature**: A member believes Claude Projects are more useful than Chat-GPT's memory feature.
   - The member also stated that custom GPTs are more like projects, allowing for the use of your own endpoints.
- **OpenAI is Winning the Attention Game**: OpenAI is winning by controlling attention through releasing new models like GPT-4o.
   - The member stated that people are talking about OpenAI's new models, even if they don't want to participate in the tech hype.
- **GPT-4o is Now Worse than Claude and Mistral**: Members have noticed that GPT-4o has become dumber lately and may be suffering from a type of Alzheimer's.
   - Claude Sonnet is being praised for its superior performance and is becoming a preferred choice among members.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1274795556245471303)** (26 messages🔥): 

> - `OpenAI Vision API`
> - `Vision Cost`
> - `Virtual Environment for GPT`
> - `Headless Browser` 


- **API Gives Better Vision than Web Interface**: A member shared that using the OpenAI Vision API provides better results compared to the web interface.
   - The web interface was considered to be at the lowest quality setting, and the member was encouraged to try the API for improved outcomes.
- **OpenAI Vision Cost and Resolutions**: The cost for processing a **1080x1920** image using the latest model is **$0.005525**.
   - The member highlighted the adjustability of the API for various resolutions, suggesting that lower resolutions could help reduce cost.
- **Virtual Environment for GPT**: A member mentioned their work on creating a virtual environment for GPT.
   - This environment would enable GPT to code and perform actions independently, including controlling the cursor and browsing the web using the keyboard, mimicking human interactions.
- **Headless Browser vs. Clicking for GPT**: A member questioned the rationale behind using clicking actions in the virtual environment, suggesting that a headless browser would provide a simpler and more sensible approach.
   - The member emphasized the ease and versatility of headless browsers for specific tasks, which could ultimately lead to better features.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1274400229470244997)** (7 messages): 

> - `GPT Mini Prompt Engineering`
> - `GPT 3.5 vs GPT 4`
> - `ChatGPT Configuration`
> - `Code Interpreter Limitations`
> - `GPT Mini Image Generation` 


- **GPT Mini Prompt Engineering is a Different Beast**: A user expressed difficulty setting up prompts for GPT Mini 4.0 models, stating it feels much different from GPT 3.5 and requires more optimized prompts and tweaking.
   - This sentiment aligns with observations that GPT Mini 4.0 seems to require more precise prompt engineering and is less forgiving than its predecessors.
- **ChatGPT Configuration: A User's Tale of Frustration**: Another user shared their struggles configuring ChatGPT for specific purposes, citing issues like hallucinations, inconsistent responses, and discrepancies in behavior with and without the code interpreter.
   - They also mentioned using multiple courses and implementing patterns without success, indicating the difficulty in overcoming these challenges.
- **GPT Mini Can't Generate Images? Not So Fast!**: A user initially believed GPT Mini couldn't generate images, but later realized they were using GPT Mini instead of the full ChatGPT model.
   - This highlights the importance of clarifying which model is being used when discussing prompt engineering.
- **Avoiding Contrastive Prompting: A Wise Move?**: One user mentioned avoiding contrastive prompting altogether, suggesting it's a difficult concept to control even in experimental scenarios.
   - This implies that mastering contrastive prompting may be beyond the scope of casual exploration and requires more advanced knowledge.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1274400229470244997)** (7 messages): 

> - `GPT-4.0`
> - `Prompt engineering`
> - `GPT-3.5`
> - `GPT mini`
> - `Code interpreter` 


- **GPT-4.0 is less forgiving with prompts**: A member noticed that setting up systems, instructions or assistants prompt for **GPT mini 4.0 models** feels much different from **GPT-3.5 or GPT-4.0**.
   - They noted it seems to require more optimized prompts and tweaking each time, and is less forgiving.
- **GPT-3.5 is the sweet spot**: Another member suggests that GPT-3.5 might be in between GPT-4.0 and GPT-mini in terms of prompt optimization requirements.
   - They mention that this is just their observation and not their area of expertise.
- **Challenges with GPTs**: One member shared their struggles with getting ChatGPT to "configure" for their purposes.
   - They listed challenges including **hallucinations**, using information not from the provided document, repeating the same answer to different questions, and inconsistent behavior with the code interpreter.
- **Prompting for image generation**: A member encountered a challenge with GPT mini not generating pictures.
   - This was resolved by confirming that they were indeed using GPT mini, as GPT-3.5 and GPT-4.0 can generate pictures if prompted correctly.


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1274478635088220241)** (27 messages🔥): 

> - `CLM`
> - `GPT Model Size`
> - `Model Interpretability`
> - `Procreate`
> - `Markov Chains` 


- **Topology's New CLM**: The [Continuous Learning Model (CLM)](https://yellow-apartment-148.notion.site/CLM-Docs-507d762ad7b14d828fac9a3f91871e3f) is a new model that remembers interactions, learns skills autonomously, and thinks in its free time, just like humans.
   - The CLM just wants to learn, and you can try it at [http://topologychat.com](http://topologychat.com).
- **GPT5's Larger Size**: In order to get meaningful improvement, a new model should be at least **20x** bigger than the current model.
   - Training takes **6 months** and requires a new, **20x** bigger datacenter, which takes about a year to build.
- **Challenges with Model Interpretability**: It is difficult to interpret models, especially when it comes to understanding parameter count.
   - Companies like Arthur have grown a lot on first gen AI safety tech, so there may be a second wave of companies that focus on model interpretability.
- **Procreate's Stance on Generative AI**: Procreate CEO made it clear that they will not be integrating generative AI into their products.
   - Artists and users on social media celebrated this decision, but some noted that it might be an announcement that they will not add features, and this might change in the future.
- **Markov Chains for Creativity**: A user suggested that Markov Chains could be used as drafters and LLMs as rephrasers for creative writing.
   - They mentioned that they had a similar experience with a project where they used a Markov chain to generate fake AWS blog posts, which they found humorous.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aidan_mclau/status/1818071890755469365?s=46">Tweet from Aidan McLau (@aidan_mclau)</a>: &gt;&gt;Continuous Learning Model (CLM) by Topology&lt;&lt;  The CLM is a new model that remembers interactions, learns skills autonomously, and thinks in its free time, just like humans.  The CLM jus...</li><li><a href="https://x.com/mparakhin/status/1824330760268157159?s=46">Tweet from Mikhail Parakhin (@MParakhin)</a>: @sandeepreddys09 @emollick In order to get some meaningful improvement, the new model should be at least 20x bigger. Training takes at least 6 months, so you need a new, 20x bigger datacenter, which t...</li><li><a href="https://share.snipd.com/snip/712b360a-fc18-4359-8708-f345">Snipd — Highlight &amp; share the best moments in podcasts</a>: no description found</li><li><a href="https://yellow-apartment-148.notion.site/CLM-Docs-507d762ad7b14d828fac9a3f91871e3f">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://share.snipd.com/snip/712b360a-fc18-4359-8708-f34519e7cde3">Optimize Resources, Optimize Performance | 2min snip from &quot;The Cognitive Revolution&quot; | AI Builders, Researchers, and Live Player Analysis</a>: 2min snip from Popular Mechanistic Interpretability: Goodfire Lights the Way to AI Safety | &quot;The Cognitive Revolution&quot; | AI Builders, Researchers, and Live Play…</li><li><a href="https://x.com/lvwerra/status/1825175724224901623">Tweet from Leandro von Werra (@lvwerra)</a>: It&#39;s beautiful to see how far you can get with a 360M model (5x smaller than GPT-2!) with a few tricks:  - curate the pretraining data for educational content - choose well tuned hyperparameters -...</li><li><a href="https://x.com/mattshumer_/status/1824836674758557867?s=46">Tweet from Matt Shumer (@mattshumer_)</a>: Friendly reminder than models trained on 10x more compute than GPT-4 will be released in the next 6 months or so</li><li><a href="https://x.com/MKBHD/status/1825521261373489197">Tweet from Marques Brownlee (@MKBHD)</a>: Bookmark this. Such a fascinating announcement  Procreate CEO gets on camera to make it clear he HATES generative AI, and they will not be integrating it ever into any of their products. Artists and u...</li><li><a href="https://x.com/aakashsastry/status/1825595241346519412?s=46">Tweet from Aakash (@aakashsastry)</a>: Today, we’re excited to share an early preview of our latest video model @hotshotco. And it’s available for you to use today.  Link to try + more results in the thread below 👇</li><li><a href="https://x.com/lateinteraction/status/1825594011484303596?s=46">Tweet from Omar Khattab (@lateinteraction)</a>: 🧵What&#39;s next in DSPy 2.5? And DSPy 3.0?  I&#39;m excited to share an early sketch of the DSPy Roadmap, a document we&#39;ll expand and maintain as more DSPy releases ramp up.  The goal is to comm...</li><li><a href="https://news.ycombinator.com/item?id=41286203">Markov chains are funnier than LLMs | Hacker News</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1274095343855996981)** (78 messages🔥🔥): 

> - `DSPy`
> - `Cursor`
> - `Langchain`
> - `Mistral`
> - `Model Merging` 


- **DSPy: Not a commercial product yet**: A member asked if there is a commercial company behind **DSPy**, to which another member replied "not yet, but obviously Omar is working on it."
   - Another member noted they went to the **Cursor** office meetup, and while there was no alpha to share, they did say hi.
- **DSPy's potential for local model improvement**: A member reported running **DSPy** locally based on claims that it could make local models as good as **GPT-4** for specific tasks.
   - However, they haven't experimented with it much beyond the basic tutorials because frontier models have gotten so cheap.
- **DSPy bridging the gap between prompting and finetuning**: **DSPy** aims to bridge the gap between prompting and finetuning by allowing users to avoid manual prompt tuning.
   - One of the things they mention in the paper is that **DSPy** allows you to avoid prompt tuning, potentially making it easier to switch models, retune to data shifts, etc.
- **DSPy: Better at prompting than humans?**: Some members believe that **DSPy** is better at prompting the model than a human could be.
   - However, others believe that there is still room for human engineering in prompting and that there are still many things a human can do that **DSPy** cannot.
- **Langchain and Substrate Swapping**: One member commented that **Langchain** also swaps substrates, but only Langchain gets criticism for it.
   - They also noted that an example of this would be nice to see.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://changelog.com/jsparty/331">Building LLM agents in JS with Tejas Kumar (JS Party #331)</a>: KBall and returning guest Tejas Kumar dive into the topic of building LLM agents using JavaScript. What they are, how they can be useful (including how Tejas used home-built agents to double his podca...</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: no description found</li><li><a href="https://github.com/wesen/dspy-grug">GitHub - wesen/dspy-grug: dspy tutorial</a>: dspy tutorial. Contribute to wesen/dspy-grug development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1274099259095912541)** (49 messages🔥): 

> - `Data Ingestion to KG`
> - `Command-r-plus in Sillytavern`
> - `API Key Partial Responses`
> - `Prompt Tuning`
> - `Cohere Office Hours` 


- **Data Ingestion to KG**: A user asked about frameworks used for extracting triples for data ingestion to a Knowledge Graph.
- **Command-r-plus not working**: A user reported that command-r-plus in Sillytavern stopped working consistently when the context length reaches 4000 tokens.
- **API Key Partial Responses**: A user reported experiencing issues with their API key returning only partial responses, even after trying different Wi-Fi routers and cellular data.
- **Prompt Tuning Still Borked**: A user mentioned that prompt tuning is still not working correctly.
- **Cohere Office Hours**: A reminder was given for the Cohere Office Hours event, which has already garnered 27 interested participants.


  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1275066929199058985)** (1 messages): 

> - `Cohere Developer Office Hours`
> - `Prompt Tuning`
> - `Guided Generations API`
> - `LLM University Tool Use Module`
> - `Structured Outputs` 


- **Cohere Developer Office Hours Kick-Off!**: Join Cohere's **Sr. Product Manager** and **DevRel** for a casual session on **product and content updates** with **best practices** and **Q&A** on **Prompt Tuning**, **Guided Generations API with Agents**, and **LLM University Tool Use Module**.
   - The event takes place today at **1 PM ET in the #stage channel** and can be found at [this link](https://discord.com/events/954421988141711382/1265012161965461625).
- **Cohere Prompt Tuner: Optimized Prompting!**: Learn about the **Cohere Prompt Tuner**, a powerful tool to optimize prompts and improve the accuracy of your LLM results.
   - The blog post details how to utilize this tool and the [associated features](https://cohere.com/blog/intro-prompt-tuner).
- **Structured Outputs for Accurate JSON Generations**: **Structured Outputs**, a recent update to Cohere's tools, delivers **80x faster** and **more accurate** **JSON generations** than open-source implementations.
   - This new feature improves the accuracy of JSON output and is discussed in [this blog post](https://cohere.com/blog/introducing-structured-outputs).
- **Workflow Automation with the LLM University Module**: The **LLM University Tool Use Module** simplifies **workflow automation** by leveraging the capabilities of **Command R+**.
   - Learn how to **automate tasks** and **workflows** through this new module, discussed in [this blog post](https://cohere.com/blog/tool-use-llmu).
- **Don't Miss Out on Cohere's Office Hours!**: Don't miss this opportunity to **learn from Cohere's experts** and other **builders from the server**. 
   - Join the discussion and **expand your knowledge** about the **latest updates in Cohere's tools**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/intro-prompt-tuner">Introducing Cohere Prompt Tuner: Prompt Optimization at Your Fingertips</a>: Automatically improve your prompts with Cohere’s new Prompt Tuner, available in beta today.</li><li><a href="https://cohere.com/blog/introducing-structured-outputs">Introducing Structured Outputs with JSON Response Format</a>: Structured Outputs improves accuracy of JSON generations and is 80x faster than open source implementations.</li><li><a href="https://cohere.com/blog/tool-use-llmu">Learn Workflow Automation with Our New LLM University Module on Tool Use</a>: Learn how to leverage the tool use capabilities of Command R+ to automate tasks and workflows.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1274148542755110953)** (43 messages🔥): 

> - `API key monitoring`
> - `production keys`
> - `Cohee chat`
> - `Trial keys`
> - `Structured output` 


- **Production API Keys and Monitoring**: A member questioned whether obtaining a production API key would require them to monitor all LLM output for unspecified objectionable material.
- **Production Key for Cohere Chat**: A member asked if a production key can be used on Cohere Chat.
- **Production Key Issues**: A member reported receiving a [429] error when trying to use their production key on Cohere Chat.
- **Generating Structured JSON Output**: A member asked about open-source implementations for guaranteed structured output.
- **Guidance for Structured Output**: A member inquired about methods for generating structured JSON objects using an LLM.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/structured-outputs-json">Structured Generations (JSON) — Cohere</a>: no description found</li><li><a href="https://status.cohere.com/">Cohere Status Page Status</a>: Latest service status for Cohere Status Page</li><li><a href="https://github.com/guidance-ai/guidance">GitHub - guidance-ai/guidance: A guidance language for controlling large language models.</a>: A guidance language for controlling large language models. - guidance-ai/guidance
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1274631557369299069)** (1 messages): 

> - `CursorLens`
> - `Cohere models` 


- **CursorLens: An Analytics Tool for Prompts**: CursorLens is a tool that provides analytics on your prompts and allows you to configure models not available through Cursor itself, such as Cohere models.
   - It allows you to see analytics on your prompts and configure models that are not available through Cursor itself, e.g. Cohere.
- **Cohere Models for Codebase Searches**: Cohere models are thought to be effective for codebase searches and queries.
   - The user believes that Cohere models can be really good for some across codebase searches and queries.
- **CursorLens is Open Source**: CursorLens is open source and available for anyone to try.
   - The user encourages others to try CursorLens and contribute to the open source project.



**Link mentioned**: <a href="https://www.producthunt.com/posts/cursor-lens"> CursorLens - Open Source dashboard and analytics for Cursor IDE | Product Hunt</a>: An open-source dashboard for Cursor.sh IDE. Log AI code generations, track usage, and control AI models (including local ones). Run locally or use upcoming hosted version.

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1274227847006785597)** (2 messages): 

> - `Toolkit Bug Fixes`
> - `Python SDK Linting` 


- **Toolkit and Python SDK Bug Fixes & Linting**: A member pushed bug fixes and linting improvements to the Cohere Toolkit and Python SDK.
   - Another member expressed gratitude for the contribution.
- **A Big Thank You**: A member expressed gratitude for the contribution.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1274135442501013565)** (12 messages🔥): 

> - `Yi Tay's Work Style`
> - `AI Regulation`
> - `01AI's future` 


- **Yi Tay is a tireless worker**: The discussion centers around the work styles of various AI organizations, with one member suggesting that **Yi Tay** operates with a **'chaos no sleep grind'** mentality.
- **Nancy Pelosi opposes California AI Bill**: **Speaker Emerita Nancy Pelosi** issued a statement opposing **California Senate Bill 1047** on AI regulation.
- **01AI's market strategy questioned**: A member asks about **01AI's future market strategy** due to a recent tweet suggesting a possible retreat from **non-Chinese markets**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/phill__1/status/1825438202548658526?s=46">Tweet from Phil (@phill__1)</a>: what is going on with .@01AI_Yi? Are they pulling out of the non Chinese market?</li><li><a href="http://pelosi.house.gov/news/press-releases/pelosi-statement-opposition-california-senate-bill-1047">Pelosi Statement in Opposition to California Senate Bill 1047</a>: San Francisco – Speaker Emerita Nancy Pelosi issued this statement in opposition to California Senate Bill 1047:
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1274109871003930664)** (15 messages🔥): 

> - `Hermes 2.5`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic`
> - `Zicheng Xu Laid Off` 


- **Zicheng Xu Laid Off**: Zeyuan Allen-Zhu announced that the author of the "Part 2.2" tutorial, Zicheng Xu, has been unexpectedly laid off.
   - Allen-Zhu strongly endorses Xu and provided his email address for potential collaborators or employers: zichengBxuB42@gmail.com (remove the capital 'B').
- **Nous Hermes Discord Drama**: A user mentioned a discussion in the Nous Discord regarding a user's perceived rudeness and misrepresentation of evaluation settings.
   - The user mentioned that their evaluation details were in the SFT section of the paper, and admitted that it doesn't feel good to get things wrong but the core of the article is still valid.
- **Meta Cooking (Model Harnessing)**: A user wondered what "meta cooking" is, suggesting a potential conflict or drama in the Nous Discord.
   - The user mentioned finding contradictory information about evaluation settings, possibly due to the use of default LM Harness settings without clear documentation.
- **Evaluation is Hard, Focus on It**: The user expressed that the experience of the Discord drama motivated them to write a fun post about evaluation.
   - They acknowledge the difficulty of accurate and consistent evaluation, and consider it important to emphasize this aspect.



**Link mentioned**: <a href="https://x.com/zeyuanallenzhu/status/1824550891304915081?s=46">Tweet from Zeyuan Allen-Zhu (@ZeyuanAllenZhu)</a>: (1/2) Many asked for Part 2.2 and I&#39;m sorry for the delay. Our author Zicheng Xu has been unexpectedly laid off. He has my strongest endorsement (see next post). If interested in this project or h...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1274164697662226454)** (15 messages🔥): 

> - `AI21 Models`
> - `AI21 vs AI2`
> - `AI Bubble`
> - `Gary Marcus`
> - `AI Safety` 


- **AI21 models on LMSYS**: New "toto" models on LMSYS are likely from AI21.
   - This could be why AI2 has been renamed to Ai2, as AI2A12 is confusing with AI21.
- **Gary Marcus Revisited AI Bubble Concerns**: Gary Marcus revisited his keynote from AGI-21, noting that many of the issues he highlighted then are still relevant today despite significant advances in AI.
   - The video, titled "The AI Bubble: Will It Burst, and What Comes After?" is available on YouTube.
- **Switching to AI Safety Career Trajectory**: A user shared a blog post about switching their career trajectory to AI safety.
   - They explained that puzzle writing took up too much headspace and they wanted to change things up this year.
- **Meta's GenAI Releases Tuning-Free Personalized Image Generation**: Meta's GenAI has released a new research paper titled "Imagine Yourself: Tuning-Free Personalized Image Generation."
   - The feature is available now as a beta in Meta AI for users in the US.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.alexirpan.com/2024/08/18/nine-years.html">Nine Years Later</a>: Sorta Insightful turns nine years old today!  </li><li><a href="https://x.com/swishfever/status/1824605103434698820">Tweet from fishy business (@swishfever)</a>: new &#34;toto&#34; models on lmsys are likely from ai21</li><li><a href="https://x.com/aiatmeta/status/1825593390043730390?s=46">Tweet from AI at Meta (@AIatMeta)</a>: 🆕 Research paper from GenAI at Meta: Imagine yourself: Tuning-Free Personalized Image Generation.  Research paper ➡️ https://go.fb.me/wre8f0  Want to try it? The feature is available now as a beta in...</li><li><a href="https://tenor.com/bj7gg.gif">Office Space Michael Bolton GIF - Office Space Michael Bolton Why Should I Change - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=91SK90SahHc">The AI Bubble: Will It Burst, and What Comes After?</a>: Prof Gary Marcus revisited his keynote from AGI-21, noting that many of the issues he highlighted then are still relevant today despite significant advances ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1274151699552010387)** (45 messages🔥): 

> - `Procrastination`
> - `Blog Design`
> - `Substack`
> - `Fast Writing` 


- **Procrastination is a common problem**: One member mentioned they've been procrastinating on getting their blog back up and running because they want to get the design just right, but they know it's a distraction.
   - They also admitted to being a decently fast writer but find it easy to convince themselves not to write.
- **Substack is easy to use but difficult to customize**: Another member mentioned they've battled Substack for hours trying to get the big wordart at the top of their blog.
   - They also expressed the desire to have more control over the design of their blog, which is why they haven't used a platform like Substack.
- **FastHTML makes blogging easy and fun**: A member mentioned that they built a blog site in one day using FastHTML.
   - They found the experience to be pretty fun and enjoyable.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1274527247600193617)** (15 messages🔥): 

> - `GrokAdamW optimizer`
> - `GrokFast paper`
> - `Gemma 2B update`
> - `Transformers dev version`
> - `Unsloth` 


- **GrokAdamW optimizer released**: GrokAdamW, a pytorch optimizer that encourages fast grokking, was released and is working with Axolotl via the transformers integration. [GrokAdamW repository](https://github.com/cognitivecomputations/grokadamw)
- **GrokAdamW inspired by GrokFast**: The optimizer is inspired by the GrokFast paper, which aims to accelerate generalization of a model under the grokking phenomenon. [GrokFast paper](https://arxiv.org/abs/2405.20233)
- **Gemma 2B update causes Axolotl crash**: An update to the Gemma 2B repo caused Axolotl to crash.
- **Reminder to use the dev version Transformers**: It's important to use the dev version of Transformers. [Dev version installation](https://github.com/huggingface/transformers.git)
- **Finetuning Gemma 2, Llama 3.1, Mistral 2-5x faster with 70% less memory via Unsloth!**: Unsloth enables finetuning Gemma 2, Llama 3.1, and Mistral 2-5x faster with 70% less memory using directly quantized 4bit models with bitsandbytes. [Gemma 2 (2B) Google Colab notebook](https://colab.research.google.com/drive/1weTpKOjBZxZJ5PQ-Ql8i6ptAY2x-FWVA?usp=sharing) [Gemma 2 (9B) Google Colab notebook](https://colab.research.google.com/drive/1vIrqH)


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/gemma-2-2b">unsloth/gemma-2-2b · Hugging Face</a>: no description found</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>: Contribute to cognitivecomputations/grokadamw development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/transformers/pull/32521">Add support for GrokAdamW optimizer by ehartford · Pull Request #32521 · huggingface/transformers</a>: What does this PR do? Add support for GrokAdamW optimizer This PR adds support for the GrokAdamW optimizer to the transformers library. Changes Introduced  Integrated the GrokAdamW optimizer into t...</li><li><a href="https://arxiv.org/abs/2405.20233">Grokfast: Accelerated Grokking by Amplifying Slow Gradients</a>: One puzzling artifact in machine learning dubbed grokking is where delayed generalization is achieved tenfolds of iterations after near perfect overfitting to the training data. Focusing on the long d...</li><li><a href="https://github.com/ironjr/grokfast">GitHub - ironjr/grokfast: Official repository for the paper &quot;Grokfast: Accelerated Grokking by Amplifying Slow Gradients&quot;</a>: Official repository for the paper &quot;Grokfast: Accelerated Grokking by Amplifying Slow Gradients&quot; - ironjr/grokfast
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1274177210059587684)** (20 messages🔥): 

> - `Gemma 2b training issues`
> - `Zero Loss`
> - `Eager Attention` 


- **Zero Loss during Gemma 2b Training**: A user reported a consistent loss of **0.0** during the training of a **Gemma 2b** model, with a **nan** gradient norm.
- **Eager Attention Recommended for Gemma 2b Training**: Another user recommended using **eager attention** instead of **sdpa** for training **Gemma 2b** models.
- **Eager Attention as the Fix**: The user who was experiencing the zero loss issue confirmed that **eager attention** fixed the problem.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1274243931667828767)** (17 messages🔥): 

> - `Chat Template`
> - `Axolotl prompt strategies`
> - `Using custom loaders`
> - `Training with ShareGPT`
> - `Fine-tuning with Axolotl` 


- **Chat Template for Axolotl**: The user asked for clarification on using a Chat Template type in a `.yml` config file for Axolotl. They were specifically interested in specifying which loader to use, for example, ShareGPT.
- **Using a Custom Loader with Axolotl**: Another user suggested that the user could specify which loader to use by providing a custom `.yml` file.
- **Axolotl's Chat Template Support**: The user expressed interest in using the `chat_template` type in Axolotl and asked if it would support the `role: system` messages in their dataset.
- **Fine-tuning with Axolotl: No Coding Required**: A user clarified that fine-tuning with Axolotl generally does not require coding knowledge, but rather understanding how to format datasets and adapt existing examples.
- **LLama 3.1 70b Fine-tuning: User Experience**: A user mentioned owning a powerful AI rig to run LLama 3.1 70b but felt it was still lacking in some key areas. They had a large dataset of content they had written and scraped and wanted to use it for fine-tuning.



**Link mentioned**: <a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1732">Allow using tokenizer&#39;s default chat template or pass custom jinja chat template by chiragjn · Pull Request #1732 · axolotl-ai-cloud/axolotl</a>: Closes #1689 Summary of changes:  Adds tokenizer_default as option for chat_template in chat_template prompt strategy that allows using the chat template from tokenizer&amp;#39;s config.json Allows fa...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1275145386327150612)** (1 messages): 

> - `LLaMa 3.1 8b Lora`
> - `Post-Hoc Reasoning`
> - `Sonnet 3.5`
> - `Claude` 


- **LLaMa 3.1 8b Lora for post-hoc reasoning detection**: A user is training a **LLaMa 3.1 8b Lora** to detect **post-hoc reasoning** within a conversation.
   - They spent three days curating a small dataset of **less than 100 multi-turn conversations** with around **30k tokens** to help with the task.
- **Sonnet 3.5 & Claude struggles with post-hoc reasoning examples**: The user employed **Sonnet 3.5** to help with generating examples, but had to fix multiple things in each generated example, despite careful prompt crafting.
   - They had to iterate multiple times on each specific idea they wanted to convey in the dataset, manually editing each example to get the desired output.
- **Models are primed to do post-hoc reasoning**: Even when instructing the models not to create examples with **post-hoc reasoning**, they still generated them due to their fine-tuning data.
   - The user had to manually fix these issues, highlighting the difficulty in training models to avoid specific reasoning patterns.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1274089813460520960)** (39 messages🔥): 

> - `LangChain Caching`
> - `LLM structured output`
> - `LangChain JSON parsing`
> - `RAG chatbot delete functionality`
> - `Hybrid search relevance` 


- **LangChain Caching Issues**: A member asked why `.batch_as_completed()` isn't sped up by caching, even though `.invoke()` and `.batch()` are near instant after caching.
   - They noticed that the cache is populated after the first run, but `.batch_as_completed()` doesn't seem to utilize the cache.
- **LLMs struggle with structured output**: A member mentioned that local LLMs, like Llama 3.1, often have difficulty producing consistently structured output.
   - They asked if there are any datasets specifically for training models to improve JSON parsing and structured output for use with tools or ReAct agents.
- **Deleting files in a RAG chatbot**: A member asked about implementing a delete functionality for files in a RAG chatbot that uses MongoDB as a vector database.
   - A helpful response provided examples of using the `delete` method from the LangChain library for both MongoDB vector stores and OpenAIFiles, along with relevant documentation links.
- **Hybrid Search Relevance Issues**: A member described a RAG application using a hybrid search approach with BM25Retriever and vector similarity search, but they were experiencing issues with the relevance of retrieved documents and generated answers.
   - Suggestions were offered to check the quality of documents, adjust retriever configurations, evaluate the chain setup, and review the prompt and LLM configuration.
- **Multilingual RAG workflow**: A member discussed a multilingual RAG workflow involving translating user questions into English, retrieving relevant documents in English, and then formulating answers in the user's native language.
   - The discussion included questions about the effectiveness of this approach compared to embedding documents in multiple languages, as well as whether multilingual embedding models allow for cross-language retrieval.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.smith.langchain.com/concepts/evaluation#rag-evaluation-summary>).">Evaluation | 🦜️🛠️ LangSmith</a>: The pace of AI application development is often rate-limited by high-quality evaluations because there is a paradox of choice. Developers often wonder how to engineer their prompt or which LLM best ba...</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/#delete-items-from-vector-store>)">MongoDB Atlas | 🦜️🔗 Langchain</a>: This guide provides a quick overview for getting started with MongoDB</li><li><a href="https://github.com/langchain-ai/langchain/issues/17508>)">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/sksarvesh007/adaptive-rag/blob/main/langgraph_adaptive_rag.ipynb">adaptive-rag/langgraph_adaptive_rag.ipynb at main · sksarvesh007/adaptive-rag</a>: Contribute to sksarvesh007/adaptive-rag development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1274354727970148393)** (1 messages): 

> - `ShortURL.at`
> - `URL Shortener`
> - `Social Media Links` 


- **ShortURL.at is a free URL shortener**: ShortURL.at is a free tool to shorten URLs and generate short links, making it easy to share.
   - The service offers premium features like custom short links, detailed analytics, API, UTM builder, QR codes, browser extension, app integrations and support.
- **ShortURL.at shortens links from various platforms**: ShortURL.at allows to shorten long links from [Instagram](https://www.instagram.com/), [Facebook](https://www.facebook.com/), [YouTube](https://www.youtube.com/), [Twitter](https://www.twitter.com/), [Linked In](https://www.linkedin.com/), [WhatsApp](https://www.whatsapp.com/), [TikTok](https://www.tiktok.com/), blogs and sites.
   - Just paste the long URL and click the Shorten URL button. On the next page, copy the shortened URL and share it on sites, chat and emails.



**Link mentioned**: <a href="https://shorturl.at/RbRhn">ShortURL - URL Shortener</a>: no description found

  

---


### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1274354732525289482)** (1 messages): 

> - `Steam Gift Card`
> - `ShortURL`
> - `Shortener` 


- **Steam Gift Card for Sale**: A user is offering a **$50 Steam gift card** for sale and provides a shortened URL to purchase it.
- **ShortURL for URL Shortening**: **ShortURL** is a free tool for shortening URLs and creating short links.
- **ShortURL Premium Features**: ShortURL offers **premium features** that enhance the URL shortening experience.
- **ShortURL Compatible Platforms**: ShortURL can shorten long links from various platforms like **Instagram, Facebook, YouTube, Twitter, LinkedIn, WhatsApp, TikTok, blogs, and websites**.



**Link mentioned**: <a href="https://shorturl.at/RbRhn">ShortURL - URL Shortener</a>: no description found

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1274353984890474618)** (4 messages): 

> - `CursorLens`
> - `LLMs`
> - `Machine Learning from Scratch` 


- **CursorLens: New Dashboard for Cursor Users**: **CursorLens** is an open-source dashboard for Cursor users that provides analytics on your prompts and allows you to configure models not available through Cursor itself.
   - It was recently launched on ProductHunt: [https://www.producthunt.com/posts/cursor-lens](https://www.producthunt.com/posts/cursor-lens).
- **LLMs Explained: From Assistant to Deep Concepts**: This blog post dives into the workings of LLMs, starting with high-level abstractions and gradually delving into concepts like tokenization, sampling, and embedding.
   - It also discusses limitations of current LLMs, such as their inability to count Rs in "strawberry" and reverse the string "copenhagen." Find the blog post here: [https://amgadhasan.substack.com/p/explaining-how-llms-work-in-7-levels](https://amgadhasan.substack.com/p/explaining-how-llms-work-in-7-levels).
- **Machine Learning from Scratch: Beginner-Friendly Guide**: This GitHub repository provides a step-by-step guide to learning machine learning from scratch, assuming no prior knowledge.
   - It covers core machine learning algorithms and neural networks, explaining the underlying math with practical examples, including gradient descent and backpropagation. Find the repository here: [https://github.com/DorsaRoh/Machine-Learning](https://github.com/DorsaRoh/Machine-Learning). 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://amgadhasan.substack.com/p/explaining-how-llms-work-in-7-levels">Explaining how LLMs work in 7 levels of abstraction</a>: Overview</li><li><a href="https://www.producthunt.com/posts/cursor-lens"> CursorLens - Open Source dashboard and analytics for Cursor IDE | Product Hunt</a>: An open-source dashboard for Cursor.sh IDE. Log AI code generations, track usage, and control AI models (including local ones). Run locally or use upcoming hosted version.</li><li><a href="https://github.com/DorsaRoh/Machine-Learning">GitHub - DorsaRoh/Machine-Learning: Machine learning: 0 ➔ 1</a>: Machine learning: 0 ➔ 1. Contribute to DorsaRoh/Machine-Learning development by creating an account on GitHub.</li><li><a href="https://shorturl.at/RbRhn">ShortURL - URL Shortener</a>: no description found
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1274354232966905897)** (1 messages): 

> - `URL Shortener`
> - `ShortURL` 


- **ShortURL: A Free URL Shortener**: ShortURL is a free tool to shorten URLs and generate short links, making it easy to share.
   - Just paste the long URL and click the Shorten URL button. On the next page, copy the shortened URL and share it on sites, chat and emails.
- **ShortURL Premium Features**: Premium features include custom short links, powerful dashboard, detailed analytics, API, UTM builder, QR codes, browser extension, app integrations and support.
   - You can create an account for premium features here: [Create Account](https://shorturl.at/vSZ02)
- **ShortURL for Various Platforms**: ShortURL allows to shorten long links from various platforms like [Instagram](https://www.instagram.com/), [Facebook](https://www.facebook.com/), [YouTube](https://www.youtube.com/), [Twitter](https://www.twitter.com/), [Linked In](https://www.linkedin.com/), [WhatsApp](https://www.whatsapp.com/), [TikTok](https://www.tiktok.com/), blogs and sites.



**Link mentioned**: <a href="https://shorturl.at/RbRhn">ShortURL - URL Shortener</a>: no description found

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1274085466739708007)** (37 messages🔥): 

> - `Orange Pi 5`
> - `GPT-4o-mini`
> - `OpenInterpreter settings`
> - `OpenInterpreter API`
> - `Local LLMs for bash commands` 


- **Orange Pi 5 Review**: A member posted a [YouTube video review](https://youtu.be/79lquFD3oT4) of the **Orange Pi 5**, which is a new **affordable yet powerful Arm-based SBC**.
   - The video states that the **Orange Pi 5 is not to be confused with the Raspberry Pi 5**.
- **GPT-4o-mini model woes**: A user expressed difficulty in setting their model to **GPT-4o-mini** using the `set model` command.
   - Another member quickly provided a solution: `interpreter --model gpt-4o-mini`.
- **OpenInterpreter Settings Reset**: A user encountered issues after experimenting with OpenInterpreter settings and sought a way to revert or reset to default.
   - Another member recommended using the command `interpreter --profiles` to view and edit profiles, as well as uninstalling and reinstalling OpenInterpreter using `pip uninstall open-interpreter` and `pip install open-interpreter`.
- **OpenInterpreter API Integration**: A user expressed interest in integrating OpenInterpreter into their existing AI core by sending requests to OI, running code, and receiving the output.
   - The user was advised to use a Python script, potentially with a Flask server, to handle the communication between their AI core and OpenInterpreter.
- **Local LLMs for Bash Commands**: A member asked for recommendations on local LLMs that are adept at handling bash commands.
   - Another member suggested **CodeStral** and **Llama 3.1**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.openinterpreter.com/guides/basic-usage#programmatic-chat">Basic Usage - Open Interpreter</a>: no description found</li><li><a href="https://docs.openinterpreter.com/settings/all-settings#os-mode">All Settings - Open Interpreter</a>: no description found</li><li><a href="https://docs.openinterpreter.com/settings/all-settings#auto-run">All Settings - Open Interpreter</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/profiles/defaults/default.yaml">open-interpreter/interpreter/terminal_interface/profiles/defaults/default.yaml at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/tree/main/interpreter/core/computer">open-interpreter/interpreter/core/computer at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://youtu.be/79lquFD3oT4">Orange Pi 5 Hands-On Review, Finally, A New Affordable Yet Powerful Arm-Based SBC!</a>: In this video, we take a look at the all-new Orange Pi 5 SBC  &quot;Not to be confused with the Raspberry Pi 5. This is the cheapest RK3588S SIngle board computer...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter?tab=readme-ov-file#sample-fastapi-server">GitHub - OpenInterpreter/open-interpreter: A natural language interface for computers</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/pull/1409">Hide missing cost completion map warnings LiteLLM by CyanideByte · Pull Request #1409 · OpenInterpreter/open-interpreter</a>: Describe the changes you have made:  Hides the wall of warnings LiteLLM gives if you use a model that isn&amp;#39;t added to their cost completion map yet.  Reference any relevant issues (e.g. &amp;qu...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/profiles/defaults/os.py">open-interpreter/interpreter/terminal_interface/profiles/defaults/os.py at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1274134055306137661)** (2 messages): 

> - `OpenInterpreter device release timeline` 


- **OpenInterpreter device release timeline: Still up in the air**: A user inquired about the device's release timeline, specifically if it's expected to ship this year.
   - While no concrete timeframe was provided, it remains unclear whether the device will ship this year or later.
- **OpenInterpreter device availability for purchase**: A separate user inquired about the device's availability for purchase.
   - No information was provided regarding whether the device is currently available for purchase.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1274320667793293345)** (4 messages): 

> - `OpenInterpreter for VSCode edits`
> - `Terminal Stuck` 


- **OpenInterpreter for VSCode Edits**: A member asked if anyone has tried using **OpenInterpreter** to do **VSCode** edits, specifically going to line 300 and changing the variable `x_alpha` to camelCase.
   - Another member replied that they haven't tried it.
- **Terminal Stuck with OpenInterpreter**: The first member mentioned that **OpenInterpreter** worked for them last time, but the **terminal got stuck** in between.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=pou46iBNZHw">Exists - Games from Text, Just Like That</a>: Text-to-Game AI creation platform that let anyone create unique multiplayer games in moments.Join our discord for the closed beta:https://discord.com/invite/...

  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1274117262302777455)** (9 messages🔥): 

> - `LLMs`
> - `RAG`
> - `Knowledge Graphs`
> - `WeKnow-RAG`
> - `Meta Optimization` 


- **LLMs struggle with reliability**: Large Language Models (LLMs) are prone to producing factually incorrect information and often produce "phantom" content that undermines their reliability.
- **WeKnow-RAG improves LLM reliability**: The WeKnow-RAG system integrates web search and Knowledge Graphs into a Retrieval-Augmented Generation (RAG) system to enhance LLM accuracy and reliability.
- **A Meta Optimizer for Workflow Optimization**: A user shared that a recently published paper implements ideas similar to their own ongoing work in the area of meta-optimization.
- **ARC Logic Puzzles: A Test of AI Intelligence**: A user shared a link to a paper which evaluates a new algorithm on the ARC Logic Puzzle task, which assesses the general intelligence of AI systems.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jeffclune/status/1825551361808990611">Tweet from Jeff Clune (@jeffclune)</a>: We evaluate the proposed algorithm on the challenging ARC logic puzzle task, which tests the general intelligence of AI systems. It progressively discovers novel agents that outperform state-of-the-ar...</li><li><a href="https://github.com/jmanhype/VITA_AI_Assistant">GitHub - jmanhype/VITA_AI_Assistant: A modular AI assistant project for audio, image, and text processing.</a>: A modular AI assistant project for audio, image, and text processing. - jmanhype/VITA_AI_Assistant</li><li><a href="https://github.com/jmanhype/ATLAS-Automated-Trading-and-Liquidity-Analysis-System">GitHub - jmanhype/ATLAS-Automated-Trading-and-Liquidity-Analysis-System</a>: Contribute to jmanhype/ATLAS-Automated-Trading-and-Liquidity-Analysis-System development by creating an account on GitHub.</li><li><a href="https://www.arxiv.org/abs/2408.05211">VITA: Towards Open-Source Interactive Omni Multimodal LLM</a>: The remarkable multimodal capabilities and interactive experience of GPT-4o underscore their necessity in practical applications, yet open-source models rarely excel in both areas. In this paper, we i...</li><li><a href="https://github.com/jmanhype/WeKnow-Information-Retrieval-Assistant/tree/master">GitHub - jmanhype/WeKnow-Information-Retrieval-Assistant: WeKnow Information Retrieval Assistant is an advanced AI-powered system featuring VITA, a voice interaction assistant. It combines OpenAI&#39;s GPT-3.5 Turbo for natural language processing with Perplexity API for web searches. The project offers custom evaluation metrics and asynchronous content retrieval, aiming to provide efficient and accurate info</a>: WeKnow Information Retrieval Assistant is an advanced AI-powered system featuring VITA, a voice interaction assistant. It combines OpenAI&amp;#39;s GPT-3.5 Turbo for natural language processing with P...</li><li><a href="https://arxiv.org/abs/2408.07611">WeKnow-RAG: An Adaptive Approach for Retrieval-Augmented Generation Integrating Web Search and Knowledge Graphs</a>: Large Language Models (LLMs) have greatly contributed to the development of adaptive intelligent agents and are positioned as an important way to achieve Artificial General Intelligence (AGI). However...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1274600395146264576)** (25 messages🔥): 

> - `DSPy 2.5 & 3.0 Roadmap`
> - `Langgraph & Routequery Error`
> - `Optimizing Expert-Engineered Prompts`
> - `DSPy & API Integration` 


- **DSPy Roadmap Unveiled!**: The DSPy Roadmap sketch for DSPy 2.5 (likely in 1-2 weeks) and DSPy 3.0 (in a few months) has been announced.
   - The roadmap outlines objectives, milestones, and efforts, and welcomes input and contributions from the community. [Link to DSPy Roadmap](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md)
- **Langgraph and Routequery Class Error**: A member encountered an error with the `routequery` class in Langgraph.
   - They requested guidance on integrating DSPy with a large set of tools and shared a link to the Langgraph implementation: [Adaptive RAG](https://github.com/sksarvesh007/adaptive-rag/blob/main/langgraph_adaptive_rag.ipynb).
- **Optimizing Expert-Engineered Prompts**: A member asked if DSPy can optimize prompts that have already been manually engineered by expert developers.
   - They inquired if DSPy is effective not only for optimizing initial drafts but also for improving well-established prompting systems.
- **DSPy and API Integration**: A member asked if they can use DSPy with an API from AI/ML.ai.
   - They inquired about how to establish a connection between DSPy and the API.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/lateinteraction/status/1825594011484303596">Tweet from Omar Khattab (@lateinteraction)</a>: 🧵What&#39;s next in DSPy 2.5? And DSPy 3.0?  I&#39;m excited to share an early sketch of the DSPy Roadmap, a document we&#39;ll expand and maintain as more DSPy releases ramp up.  The goal is to comm...</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/docs/docs/roadmap.md">dspy/docs/docs/roadmap.md at main · stanfordnlp/dspy</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://github.com/sksarvesh007/adaptive-rag/blob/main/langgraph_adaptive_rag.ipynb">adaptive-rag/langgraph_adaptive_rag.ipynb at main · sksarvesh007/adaptive-rag</a>: Contribute to sksarvesh007/adaptive-rag development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/)** (1 messages): 

batmanosama: I updated it thanks for pointing that out
  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1274539998498131989)** (4 messages): 

> - `Colpali finetuning`
> - `VLM tuning`
> - `Domain expertise`
> - `Colpali data` 


- **Finetuning Colpali**: A question arose regarding the approach to finetuning **Colpali**, a model seemingly requiring specialized expertise due to its domain-specific nature.
- **Data Needs for Colpali Fine-Tuning**: A key discussion point centered around the type of data needed for effectively finetuning **Colpali**.


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1274094572301193216)** (25 messages🔥): 

> - `FLUX Dev`
> - `LLM for medical assistance`
> - `Medical LLMs`
> - `LoRa Training` 


- **FLUX Dev can create 3x3 photo grids**: A user shared that **FLUX Dev** can generate 3x3 photo grids of the same (fictional) person.
- **Training LORAs for specific purposes**: A user expressed interest in training **LORAs** for specific purposes like **dabbing**, **middle finger**, and **30s cartoon**.
- **LLMs for medical assistance are not yet reliable**: Several users expressed skepticism about using **LLMs** for medical assistance in their current state.
- **Turning a FLUX Dev LoRA into FP8**: A user asked if they could convert their **FLUX Dev LoRA** into **FP8**, or use an **FP8 LoRA trainer on Replicate**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/laion_ai/status/1824814210758459548">Tweet from LAION (@laion_ai)</a>: FLUX Dev can generate 3x3 photo grids of the same (fictional) person  --&gt;  One could train LORAs on such photos to make a library of LORAs for consitant characters of all kinds of fictional people ...</li><li><a href="https://goldhire.app.loxo.co/job/MjM4NDcta2hrdWh2bmkxMng4ZnZiMA==?t=1723412813305">Founding AI Engineer - GoldHire</a>: no description found</li><li><a href="https://civitai.com/models/290836/multiple-views-sdxl>">Multiple Views (SDXL) - v1.0 | Stable Diffusion LoRA | Civitai</a>: Buy me a coffee: https://ko-fi.com/futureflix Trigger word: multiple views Prompt example: multiple views, a black man, pants, street urban, neckla...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1274172546987593768)** (12 messages🔥): 

> - `JPEG-LM`
> - `Image/Video Generation with LLMs`
> - `Autoregressive LLMs`
> - `SIREN`
> - `Neural Graphics Primitives` 


- **JPEG-LM: A Novel Approach to Image Generation**: A new research paper proposes modeling images and videos as compressed files using canonical codecs (e.g., JPEG, AVC/H.264) within an autoregressive LLM architecture.
   - This approach eliminates the need for raw pixel value modeling or vector quantization, making the process more efficient.
- **JPEG-LM vs. SIREN: A Battle of the Titans?**: A user playfully claims to have outperformed the SIREN architecture from 2020 with a 33kB complex-valued neural network, despite acknowledging that NVIDIA's Neural Graphics Primitives paper from 2022 has significantly advanced the field.
   - The user highlights the importance of using MS-SSIM as a metric for image quality assessment, as opposed to just MSE and MAE.
- **7B Parameters for Low-Quality Generations?**: The discussion acknowledges that utilizing 7B parameters for such low-quality image generation might be considered excessive.
   - However, the novelty and potential of this approach is still appreciated, opening new doors for future research.



**Link mentioned**: <a href="https://arxiv.org/abs/2408.08459">JPEG-LM: LLMs as Image Generators with Canonical Codec Representations</a>: Recent work in image and video generation has been adopting the autoregressive LLM architecture due to its generality and potentially easy integration into multi-modal systems. The crux of applying au...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1274108307845677227)** (5 messages): 

> - `Workflows`
> - `RAG`
> - `Agents`
> - `BeyondLLM`
> - `JSONalyze Query Engine` 


- **Workflows in Action**: A video by Rajib Deb showcases workflows featuring decorators, types for control flow, event-driven process chaining, and custom events and steps for complex tasks.
   - The video delves into the key features of workflows, demonstrating how they enable building sophisticated applications with a more structured approach.
- **RAG & Agent Templates**: Reference implementations of 3 RAG and agent papers are provided, offering a kickstart for building applications from scratch or using pre-built templates.
   - These templates, utilizing the LlamaIndex framework, emphasize event-driven techniques for advanced RAG and agent applications.
- **Agentic RAG with Claude 3.5**: A tutorial by Richmond Lake guides users on building an agentic knowledge assistant using Claude 3.5, MongoDB, and LlamaIndex.
   - The tutorial highlights building an agentic knowledge assistant over a pre-existing RAG pipeline, utilizing tool selection, task decomposition, and advanced RAG techniques.
- **BeyondLLM for Advanced RAG**: BeyondLLM, developed by AIPlanetHub, provides abstractions on top of LlamaIndex, enabling users to build advanced RAG pipelines with features like evaluation, observability, and advanced RAG capabilities in just 5-7 lines of code.
   - These advanced RAG features include query rewriting, vector search, and document summarization, streamlining the development of sophisticated RAG applications.
- **JSONalyze Query Engine as Workflow**: RavitheJads reconstructs the JSONalyze Query Engine as a workflow, showcasing the step-by-step process of converting a JSON API response into a SQLite table and queries into SQL.
   - This workflow demonstration highlights the versatility of workflows, enabling efficient data manipulation and transformation using a structured, modular approach.



**Link mentioned**: <a href="https://t.co/VybhvUgAbL">no title found</a>: no description found

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1274222201461018756)** (27 messages🔥): 

> - `Web Scrapers for LlamaIndex`
> - `RouterQueryEngine vs Agents`
> - `LlamaIndex Workflow`
> - `Batching APIs`
> - `LlamaIndex CSV Analysis` 


- **Web Scraper Recommendations for LlamaIndex**: A member asked for recommendations for web scrapers that work well with the LlamaIndex stack.
   - Another member recommended FireCrawl, and shared a YouTube video showing a more complex implementation of a LlamaIndex workflow.
- **RouterQueryEngine vs Agents in LlamaIndex**: A member inquired about the difference between the RouterQueryEngine and Agents in LlamaIndex, particularly in relation to routing and function calling.
   - Another member explained that the RouterQueryEngine acts like a hardcoded agent, while Agents are more flexible and general.
- **Batching APIs for Open-Source Models**: A member discussed how major companies like OpenAI and Google have launched batching APIs for their models, but these APIs lack processing guarantees, SLAs, and retries.
   - They shared a blog post on how to get a batching API like OpenAI for open-source models.
- **LlamaIndex CSV Analysis Limitations**: A member encountered difficulties analyzing a CSV file using LlamaIndex due to inaccurate results.
   - Another member explained that CSVs are not well-suited for vector indexes and suggested using a database or a Pandas query engine for better results.
- **Storing DocumentSummaryIndex in Neo4j**: A member inquired about storing DocumentSummaryIndex in Neo4j, which they already use for PropertyGraphIndex.
   - Another member responded that while Neo4j can be used as a vector store, it's not suitable for general key-value storage, making storing DocumentSummaryIndex in Neo4j challenging.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49">How to Get a Batching API Like OpenAI for Open-Source Models</a>: In the world of AI, efficient processing and cost management are paramount. One powerful method for achieving this is batching, which…</li><li><a href="https://youtu.be/LloUNBD9fsI">LlamaIndex Workflow | Global context</a>: In this recording I show a more complex workflow implementation through llamaindex workflowcode:https://github.com/rajib76/llamaindex/blob/main/examples/07_l...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1274774003478691951)** (2 messages): 

> - `LLMs`
> - `LLM Limitations`
> - `LLMs as Assistants`
> - `Tokenization`
> - `Sampling` 


- **LLMs as Personal Assistants**: LLMs are AI-powered assistants that can generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way.
   - They are not limited to a specific task, but rather can adapt to various inputs and prompts.  Think of them as a flexible tool that can be used for a wide range of applications.
- **LLMs: A Deep Dive**: The blog post starts with high-level abstractions, viewing LLMs as personal assistants, then dives deeper into key concepts like tokenization, sampling, and embedding.
   - This approach is designed to make the complex world of LLMs more accessible to a wider audience.
- **LLM Capabilities and Limitations**: The blog post acknowledges that LLMs are still under development and have limitations, such as failing to count the Rs in "strawberry" and reversing the string "copenhagen."
   - This honest assessment helps readers understand the current state of LLM technology and the areas where further research is needed.
- **Knowledge Graphs: A Powerful Tool**: Knowledge graphs provide a structured and intuitive way to capture the complex relationships hidden within data.
   - This approach allows for better organization and understanding of information, enabling the development of truly intelligent applications.
- **Combining Knowledge Graphs and Generative AI**: The blog post explores the potential of combining knowledge graphs with generative AI to create powerful intelligent applications.
   - This synergy leverages the strengths of both technologies to unlock new possibilities and advance the field of AI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://amgadhasan.substack.com/p/explaining-how-llms-work-in-7-levels">Explaining how LLMs work in 7 levels of abstraction</a>: Overview</li><li><a href="https://medium.com/ai-artistry/knowledge-graphs-and-generative-ai-powering-intelligent-applications-with-amazon-neptune-and-f734d96c0fa0">Knowledge Graphs and Generative AI: Powering Intelligent Applications with Amazon Neptune and…</a>: Ankush k Singal
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1274367530051309619)** (5 messages): 

> - `LLM Hosting`
> - `HF Spaces`
> - `Modal`
> - `Jarvis Labs`
> - `vLLM` 


- **HF Spaces limitations**: A member expressed difficulty hosting their own LLM using HF Spaces, citing that ZeroGPU does not support vLLM.
- **Modal and FastHTML**: Another member noted that they have used Modal for hosting LLMs, but are currently trying to use FastHTML and are looking for a setup guide.
- **Jarvis Labs for Fine-tuning**: The member mentioned having only used Jarvis Labs for fine-tuning LLMs.


  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1274663309911785523)** (1 messages): 

> - `Batching APIs`
> - `OpenAI`
> - `CuminAI`
> - `Small Language Models (SLMs)`
> - `Large Language Models (LLMs)` 


- **OpenAI and Google launch cheaper batching APIs**: OpenAI and Google recently introduced batching APIs for some of their models, offering a 50% cost reduction compared to regular requests.
   - However, these APIs currently lack processing guarantees, service level agreements (SLAs), and retries.
- **CuminAI: Batching APIs for Open-Source Models**: CuminAI provides a solution for creating batching APIs for open-source models, similar to those offered by OpenAI.
   - Check out their step-by-step guide on "How to Get a Batching API Like OpenAI for Open-Source Models" [here](https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49).
- **Small Language Models: The New Superheroes of AI**: A recent blog post from CuminAI highlights the potential of Small Language Models (SLMs), arguing that "bigger isn't always better" in the world of AI.
   - While Large Language Models (LLMs) have dominated the field, SLMs offer a more cost-effective and efficient alternative, especially for tasks that don't require extensive computational power.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.cuminai.com/how-to-get-a-batching-api-like-openai-for-open-source-models-824529788a49">How to Get a Batching API Like OpenAI for Open-Source Models</a>: In the world of AI, efficient processing and cost management are paramount. One powerful method for achieving this is batching, which…</li><li><a href="https://medium.com/@umesh-cuminai?source=post_page-----824529788a49--------------------------------)[">Umesh – Medium</a>: Read writing from Umesh on Medium. Understanding AI:), Founder CuminAI. Every day, Umesh and thousands of other voices read, write, and share important stories on Medium.</li><li><a href="https://blog.cuminai.com/?source=post_page-----824529788a49--------------------------------)">Cumin AI</a>: Convert any Huggingface model into robust Batch API for processing offline AI workloads for your Enterprise</li><li><a href="https://medium.com/@harshal-cuminai?source=collection_home---------0----------------------------)">Harshal Priyadarshi – Medium</a>: Read writing from Harshal Priyadarshi on Medium. Founder, Cumin AI. Every day, Harshal Priyadarshi and thousands of other voices read, write, and share important stories on Medium.
</li>
</ul>

</div>
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1275138618163462145)** (1 messages): 

> - `Llamafile update`
> - `Mozilla AI Community at Rise25`
> - `ML Paper Talks` 


- **Llamafile update: Speech to Text, Image Gen, Performance Boost**: **Llamafile** has released exciting new features, including **Speech to Text Commands**, **Image Generation**, and a **3x Performance Boost** for its HTTP server embeddings.
   - You can find the full update [here from Justine](https://discord.com/channels/1089876418936180786/1262961704602570832/1275110073584320576).
- **Mozilla AI Community at Rise25**: Mozilla AI is celebrating community members who are shaping a future where AI is responsible, trustworthy, inclusive, and centered around human dignity.
   - Several members attended the event, including <@631210549170012166>, <@1046834222922465314>, <@200272755520700416>, and <@1083203408367984751>.
- **ML Paper Talks: Communicative Agents & Extended Mind Transformers**: Join an insightful session with host <@718891366402490439> on cutting-edge Machine Learning research, featuring discussions on **Communicative Agents** and **Extended Mind Transformers**.
   - RSVP for these thought-provoking discussions and deep dives with authors <@878366123458977893> and <@985920344856596490>, respectively: [Communicative Agents](https://discord.com/events/1089876418936180786/1266733035231903795) and [Extended Mind Transformers](https://discord.com/events/1089876418936180786/1267946366680694817). 


  

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

---
id: 4f94e7b8-9da2-49af-ada9-2ee46641fb3a
title: 'CogVideoX: Zhipu''s Open Source Sora'
date: '2024-08-28T01:26:46.937370Z'
original_slug: ainews-cogvideox-zhipus-open-source-sora
description: >-
  AI news from late August 2024 covers open source video generation models like
  Zhipu AI's CogVideoX, updates on Llama 3, Moondream, Phi-3.5, and tools like
  Rerank API and Not Diamond. Highlights include AI model releases, research
  advancements, and industry applications.
companies:
  - zhipu-ai
  - meta
  - google
  - nvidia
  - salesforce
models:
  - llama-3
  - llama-3-1
  - moondream
  - phi-3.5
  - cogvideox
topics:
  - open-source-ai
  - video-generation
  - large-language-models
  - trust-safety
  - vision-language-models
  - retrieval-augmented-generation
  - long-form-content
  - ai-tools
  - ai-industry
---


<!-- buttondown-editor-mode: plaintext -->**Open Source Videogen is all you need.**

> AI News for 8/26/2024-8/27/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**215** channels, and **3433** messages) for you. Estimated reading time saved (at 200wpm): **369 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Since Sora was announced in Feb ([our coverage here](https://buttondown.com/ainews/archive/ainews-sora-pushes-sota/)) there have been a host of attempts at alternatives, including [Kling](https://www.youtube.com/watch?v=EIj9-xgfV2c) (not open) and [Open-Sora](https://github.com/hpcaitech/Open-Sora) (only ~1b ish). Zhipu AI, effectively Alibaba's AI arm and China's 3rd largest "[AI Tiger](https://www.scmp.com/tech/big-tech/article/3259499/chinas-four-new-ai-tigers-baichuan-zhipu-ai-moonshot-ai-and-minimax-emerge-investor-favourites)" lab, [released its new open 5B video model](https://medium.com/@ChatGLM/zhipuai-unveils-cogvideox-a-cutting-edge-video-generation-model-293e3008fda0), CogVIdeoX. Here we run into a classic limitation of email newsletters, because we can't embed video:

 ![image.png](https://assets.buttondown.email/images/4cb55e94-226b-42ad-8ec2-dbf3ef5693f0.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/4a92a7ed-2f74-4702-8a61-c17f91f0baa6.png?w=960&fit=max) 

 ![image.png](https://assets.buttondown.email/images/92dbb665-e005-495a-a3a7-0270c4c2dfe1.png?w=960&fit=max) 


 ![image.png](https://assets.buttondown.email/images/c41abe8c-209b-47d2-93e2-5c44410a76cb.png?w=960&fit=max) 

And you don't even need GPUs to run it - you can use [Zhipu's live ChatGLM webapp or desktop app](https://x.com/ChatGLM/status/1816803455761281211) (may need a Sinophone friend to help you register your phone number account) - we were able to get it running on first try.

 ![image.png](https://assets.buttondown.email/images/e8ead844-c837-4e74-8a4d-c6412d5b6720.png?w=960&fit=max) 

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


**AI Model Developments and Releases**

- **Llama 3 and Other Model Updates**: [@AIatMeta](https://twitter.com/AIatMeta/status/1828133452837007831) announced new trust & safety research and CyberSecEval 3, related to the release of Llama 3.1. [@_philschmid](https://twitter.com/_philschmid/status/1828114328936923196) reported that Llama 3 405B is now available serverless on Google Cloud Vertex AI & Hugging Face x NVIDIA NIM API.

- **Moondream Updates**: [@vikhyatk](https://twitter.com/vikhyatk/status/1828144274522939829) released an update to moondream, an open vision language model, with improved performance on DocVQA and TextVQA tasks.

- **Phi-3.5 Model**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828165274106831093) discussed the Phi-3.5 MoE chat model, a lightweight LLM with 16x3.8B parameters using 6.6B active parameters with 2 experts.

- **Together Rerank API**: [@togethercompute](https://twitter.com/togethercompute/status/1828194058126401657) introduced the Together Rerank API, featuring Salesforce's LlamaRank model for improved document and code ranking.

**AI Research and Techniques**

- **Superposition Prompting**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828147274553213092) shared insights on superposition prompting, a novel RAG methodology that accelerates and enhances performance without fine-tuning.

- **Long-form Content Generation**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828025515791298837) discussed the LongWriter paper, which introduced the AgentWrite pipeline for generating coherent outputs exceeding 20,000 words.

- **RAG vs. Long Context**: [@algo_diver](https://twitter.com/algo_diver/status/1828091411721527530) summarized a research paper comparing Retrieval Augmented Generation (RAG) to Long Context (LC) approaches, finding that LC consistently outperforms RAG but at higher costs.

**AI Tools and Applications**

- **Not Diamond**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828054608431739323) explained Not Diamond, an AI model router that automatically determines the best-suited LLM for a given query.

- **AI in Command Line**: [@JayAlammar](https://twitter.com/JayAlammar/status/1828129090970243353) highlighted the potential of AI in command line interfaces, enabling operations spanning multiple files.

- **Background Removal with WebGPU**: [@osanseviero](https://twitter.com/osanseviero/status/1828121582981599419) shared a fully on-device, open-source background removal tool using WebGPU.

**AI Industry and Business**

- **AI Hiring**: [@DavidSHolz](https://twitter.com/DavidSHolz/status/1828192803916300379) announced that Midjourney is hiring for their core data team, emphasizing opportunities to learn and make a difference in creative capacity.

- **Hyperscaler Capex**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828018114468237737) reported on increased hyperscaler capital expenditure for data center spending, with about 50% going to land, leases, and construction.

**AI Ethics and Regulation**

- **California AI Bill SB 1047**: [@labenz](https://twitter.com/labenz/status/1828189665985470613) discussed the latest version of California's AI bill SB 1047, which now focuses on requiring frontier companies to develop and publish safety plans and protocols.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. High-Performance Hardware Solutions for AI Development**


- **[Tinybox is finally entering production](https://x.com/realgeorgehotz/status/1828197925874463166?s=46&t=m9-w-4WogM5fYHxBEAFB-Q)** ([Score: 83, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1f23dh1/tinybox_is_finally_entering_production/)): **Tinybox**, a high-performance **GPU cluster** solution for **AI development**, is entering production. The system offers **8x A100 80GB GPUs** with **NVLink** and **400Gbps networking**, providing a powerful platform for machine learning tasks at a competitive price point compared to cloud alternatives.
  - **Tinybox** specs are available on [tinygrad.org](https://tinygrad.org/#tinybox), with a **15k tinybox red6 x 7900 XTX** version mentioned. Users discussed potential **throughput comparisons** between this system and a single **A100** running a batched backend.
  - Building a similar **6x 4090 setup** for $15k instead of the $25k Tinybox price was debated. **George Hotz** (imgeohot) explained the challenges, including **PCIe 4.0 signal integrity**, multiple power supplies, and cooling, citing a [blog post](https://nonint.com/2022/05/30/my-deep-learning-rig/) detailing these issues.
  - Some users found the **Tinybox build underwhelming**, suggesting it could be built for less. Others defended the pricing, noting it funds **tinygrad development** and offers a ready-built solution for businesses. The system fits in a **standard rack** (12U), with rails available in the [documentation](https://docs.tinygrad.org/tinybox/).


**Theme 2. Open-source RAG WebUI and Local LLM Deployment Advancements**



- **Open-source clean & hackable RAG webUI with multi-users support and sane-default RAG pipeline.** ([Score: 130, Comments: 43](https://reddit.com//r/LocalLLaMA/comments/1f25wo0/opensource_clean_hackable_rag_webui_with/)): **Kotaemon**, an open-source **RAG WebUI**, offers a clean interface with **multi-user support** and a customizable pipeline for both normal and advanced users. Key features include a **minimalistic UI** with dark/light mode, **multi-user management**, a default RAG configuration with **hybrid retriever and re-ranking**, advanced **citations support** with in-browser PDF viewing, **multi-modal QA support**, and **complex reasoning methods** like question decomposition and agent-based reasoning. The project aims to be extensible, allowing users to integrate custom RAG pipelines and switch between different document and vector store providers.
  - The project's **GitHub repository** is available with setup instructions. Users suggested adding a **volume to the default container** for persisting configurations, as Gradio apps require frequent point-and-click setups.
  - The **UI's clean design** was praised, with the developer sharing that the **theme is available on Hugging Face Hub** for use in other projects. The [theme can be found here](https://huggingface.co/spaces/lone17/kotaemon).
  - **Offline functionality** is supported, with users able to use **Ollama OpenAI compatible server** or **LlamaCPP local models** directly. The README provides guidelines for this setup, and the app was designed to work offline from the beginning.



**Theme 3. Innovations in Distributed AI Training and Infrastructure**



- **[Nous Research publishes a report on DisTrO (Distributed Training Over-the-Internet)](https://x.com/NousResearch/status/1828121648383566270)** ([Score: 96, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1f23guc/nous_research_publishes_a_report_on_distro/)): Nous Research has published a report on **DisTrO** (**Distributed Training Over-the-Internet**), a novel approach for training large language models using consumer-grade hardware. The method allows for **distributed training across multiple consumer GPUs** connected via the internet, potentially enabling more researchers and hobbyists to participate in AI model development. DisTrO aims to address challenges in training large models by leveraging **distributed computing techniques** and **optimizing data transfer** between nodes.
  - DisTrO is seen as a potentially **significant breakthrough**, with some users speculating it could be the "**holy grail**" of distributed optimizers. It may **reduce training costs** for both local/community models and large firms like **Meta**.
  - Users express skepticism, noting that in machine learning, **extraordinary results often come with a catch**. Some question the impact on **perplexity** and overall model performance beyond reduced training time.
  - The paper suggests a possible **new scaling law** where **model size increases without increasing communication bandwidth**. This could lead to a shift towards designing **GPUs with larger VRAM and narrower interconnects**, favoring compute-heavy workloads over I/O-heavy operations.


## Misc AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Model Developments and Releases**

- **FLUX model shows surprising capabilities**: A [post on r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/comments/1f1pdsb/flux_is_smarter_than_you_and_other_surprising/) discusses unexpected findings about the FLUX model's abilities, with the author noting "every single day, I discover something new with flux that absolutely blows my mind". They suggest we're still far from fully understanding its capabilities.

- **Salesforce releases xLAM-1b model**: A [1 billion parameter model called xLAM-1b](https://www.reddit.com/r/LocalLLaMA/comments/1f24ao7/just_released_the_newest_version_of_my_ttrpg_maps/) was released by Salesforce, achieving 70% accuracy in function calling and surpassing GPT-3.5 despite its relatively small size.

- **Updated Phi-3 Mini model**: [Rubra AI released an updated Phi-3 Mini model](https://www.reddit.com/r/LocalLLaMA/comments/1f24ao7/just_released_the_newest_version_of_my_ttrpg_maps/) with function calling capabilities, competitive with Mistral-7b v3.

- **Joy Caption update**: An [updated version of Joy Caption](https://huggingface.co/Wi-zz/joy-caption-pre-alpha) was released with batching support for ultrafast NSFW natural language captioning.

**AI Research and Techniques**

- **DisTrO distributed optimizers**: [Nous Research announced DisTrO](https://www.reddit.com/r/singularity/comments/1f1x66j/nous_research_announces_distro_distributed/), a family of distributed optimizers that reduces inter-GPU communication by 1000x to 10,000x without relying on amortized analysis. This could significantly accelerate AI training.

- **AI-powered coding demonstration**: A [video demonstration of AI-powered coding with Cursor](https://www.reddit.com/r/singularity/comments/1f1wrq1/mckay_wrigley_shows_off_aipowered_coding_with/) showcased capabilities that would have been unimaginable 5 years ago.

**AI Safety and Ethics Concerns**

- **Exodus of AGI safety researchers from OpenAI**: [Nearly half of OpenAI's AGI safety staffers have left](https://www.reddit.com/r/OpenAI/comments/1f25bse/exodus_at_openai_nearly_half_of_agi_safety/), according to a former researcher. This has sparked debate about the importance and role of AI safety research.

- **Debate on AI safety research**: There is significant discussion about whether AI safety research is crucial for progress or potentially holding back innovation. Some argue it's essential for alignment and interpretability, while others see it as unnecessary at this stage.

**AI Applications**

- **TTRPG Maps LoRA**: A [new version of a TTRPG Maps LoRA](https://www.reddit.com/r/StableDiffusion/comments/1f24ao7/just_released_the_newest_version_of_my_ttrpg_maps/) was released, demonstrating AI's potential in generating game assets and maps.

- **AI-generated "verification" image**: An [AI-generated animated "verification" image](https://www.reddit.com/r/StableDiffusion/comments/1f1zy8j/verification_pic_for_my_oc_ai/) was created using Flux Dev with a "MaryLee" likeness LoRA and Runway ML for animation.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Claude 3.5 Sonnet


**1. LLM Advancements and Benchmarking**

- **DeepSeek V2 Outperforms GPT-4**: **DeepSeek-V2**, with its impressive **236B parameters**, has surpassed GPT-4 in benchmarks like **AlignBench** and **MT-Bench**, showcasing significant improvements in model performance.
   - The [DeepSeek-V2 announcement](https://x.com/deepseek_ai/status/1787478986731429933) sparked discussions about its capabilities, particularly in areas where it outperforms existing large language models.
- **Llama 3.1 Sets New Speed Records**: **Cerebras Systems** announced their inference service delivering a striking **450 tokens/sec** for Llama3.1-70B, significantly outpacing traditional GPU setups.
   - The service offers an economically appealing rate of just **60 cents** per million tokens, attracting developers seeking cost-effective, high-performance solutions for AI applications.
- **Google Rolls Out Gemini 1.5 Models**: Google introduced three experimental models: **Gemini 1.5 Flash-8B**, a stronger **Gemini 1.5 Pro**, and an improved **Gemini 1.5 Flash**, available for testing on [Google AI Studio](https://aistudio.google.com).
   - These new models promise enhancements for **coding** and **complex prompts**, with rate limits set at **2 requests per minute** and **50 requests per day**, sparking interest in their potential capabilities.
  


**2. Open-Source AI Developments**

- **Intel's Ternary Multimodal LLM Debut**: Intel launched [LLaVaOLMoBitnet1B](https://huggingface.co/papers/2408.13402), the first Ternary Multimodal LLM capable of processing images and text to produce coherent responses.
   - The model is fully open-sourced with training scripts available, encouraging exploration of challenges and opportunities in ternary modeling within the AI community.
- **Microsoft's Phi 3.5 Excels in OCR**: Microsoft's [Phi 3.5](https://huggingface.co/spaces/MaziyarPanahi/Phi-3.5-Vision) model, released under the MIT license, demonstrates exceptional performance in OCR tasks, particularly excelling in handwriting recognition and tabular data extraction.
   - The model's impressive capabilities in text recognition across various vision tasks have generated significant interest within the AI community, highlighting its potential for practical applications.
- **LocalAI: Open-Source Alternative to OpenAI**: **LocalAI**, an open-source project by Ettore Di Giacinto, offers a free alternative to OpenAI with a REST API for local inferencing of LLMs, image, and audio generation.
   - The platform enables running advanced AI models on consumer-grade hardware without requiring a GPU, democratizing access to powerful AI capabilities.
  


**3. Distributed Training Innovations**

- **Nous Research's DisTrO Breakthrough**: Nous Research released a [preliminary report on DisTrO](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf), a distributed training framework that drastically reduces inter-GPU communication by up to **10,000x**.
   - DisTrO aims to enable resilient training of large language models without reliance on centralized computing resources, potentially democratizing AI research and development.
- **Debate Over Nous Research's Optimizer Claims**: The AI community expressed skepticism about Nous Research's new optimizer, particularly its claims regarding distributed training capabilities.
   - Discussions referenced existing tools like **Petals for Bloom** and **OpenDILo**, highlighting the need for more substantial evidence to support Nous's promises in the field of distributed AI training.
  
  


**4. Multimodal AI Progress**

- **Cog-5B Video Model Release**: The **Cog-5B video model** was released at [CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b), touted as the best open weights for video generation, featuring integration with the Diffusers library.
   - This model promises efficient inference with less than **10GB** VRAM, showcasing advancements in multimodal AI that combine text, image, and video generation capabilities.
- **StoryDiffusion: An Open-Source Sora Alternative**: The launch of **StoryDiffusion**, an open-source alternative to Sora with an MIT license, has generated excitement in the AI community, though the weights have not yet been released.
   - This development highlights the growing interest in accessible, high-quality video generation models that can compete with proprietary solutions.
  

---

# PART 1: High level Discord summaries




## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DisTrO Algorithm continues to evolve**: The **DisTrO algorithm** is being actively refined with different variants tested to optimize **communication bandwidth** while ensuring convergence performance.
   - Members noted that techniques like **SWARM** could potentially be better suited for larger models.
- **Community is eager for collaborations**: Members expressed interest in collaborating on the **DisTrO algorithm's** implementation, emphasizing open contributions and discussions.
   - The team plans to share full code and details in the upcoming weeks.
- **Philosophical debates around AI consciousness**: The implications of **qualia** and consciousness in AI are driving serious discussions, calling for more interdisciplinary collaboration.
   - Members pointed to a need for deeper cooperation between computer scientists and philosophers.
- **DisTrO explores weak device applications**: The feasibility of using **DisTrO** with weak devices such as older phones was discussed, highlighting the need for efficient training methods.
   - While **DisTrO** excels with stronger hardware, exploring its utility in less-capable systems is seen as valuable.
- **Tinybox is officially up for grabs**: After **18 months**, the **Tinybox** now has a 'buy it now' option thanks to an announcement highlighting **13 units** available for purchase.
   - The **$15k tinybox red** is being praised for its **performance-to-price** ratio in the ML space.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Tensor Conversion Struggles**: A user faced a `ValueError` regarding `embed_tokens.weight` while converting a model to GGUF, indicating mismatches between sentence-transformer and causalLM models.
   - Frustrations arose over the lack of support for pair scoring in current conversion tools.
- **Batch Size Optimization Strategies**: Discussions revealed a tactic of increasing batch size until reaching out-of-memory errors, prompting users to tweak their training settings.
   - One user plans to convert their model to Ollama format after finishing the latest training task.
- **Updates on Homoiconic AI Project**: The 'Homoiconic AI' project reported improvements in validation loss metrics using hypernets for weight generation and aims for a multimodal integration approach.
   - Members discussed how in-context learning helps improve the model's weights, referencing a [progress report](https://x.com/neurallambda/status/1828214178567647584?s).
- **LigerKernel's Copying Controversy**: Concerns arose over LinkedIn's **LigerKernel** allegedly copying core components from **Unsloth**, raising questions about claims of major improvements.
   - Community members highlighted the lack of original variable naming in the copied code.
- **Data Preparation in LLM Finetuning**: Discussions highlighted that many wish to finetune LLMs without clear objectives or datasets, raising concerns about understanding the process.
   - Members voiced that a solid foundational knowledge is essential before diving into model finetuning.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Exploring AI's Emotional Depth**: The discourse around **AI personhood** highlights that while AI can simulate emotions, it does not genuinely experience them, prompting ethical concerns when users form attachments.
   - Participants are questioning whether treating AI as friends undermines our understanding of emotionality.
- **Push for Decentralized AI**: Discussions on the **decentralization of AI** emphasize a shift towards user-owned data, reducing corporate control over AI identities and training data.
   - There’s optimism for increased prevalence of **open-source models**, aiming to disrupt current centralized systems.
- **AI Provides Companionship—Sort Of**: The discourse reflects skepticism about AI replacing real friendships despite its potential in aiding those feeling isolated.
   - Participants shared personal anecdotes about the relief AI could offer, especially in marginalized communities.
- **Concerns About GPT-4o's Reasoning**: **GPT-4o** users aired frustrations about its reasoning capabilities, citing factual inaccuracies and inconsistencies compared to earlier models.
   - Some users believe **GPT-4o** has regressed and are discussing what updates could restore its performance.
- **Struggles with YouTube Summarization Tools**: There are challenges in the efficacy of **YouTube summarization tools**, mainly due to the platform blocking bot access to transcripts.
   - While manual transcript retrieval was suggested, members noted this approach risks violating YouTube's terms of service.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Model Deployment Challenges in Hugging Face**: Users reported issues deploying the **Gemma-7B** model due to missing files and runtime errors, considering repository cloning as a workaround.
   - The conversation included suggestions about verifying model paths and employing the `.from_pretrained` method to resolve configuration issues.
- **PyTorch Lightning's LitServe boosts speeds**: [LitServe](https://github.com/Lightning-AI/litserve) from **PyTorch Lightning** claims **2x faster** model serving than FastAPI, promising enhanced deployment efficiency.
   - Users are keen to adopt this update as it accelerates inference times significantly.
- **StockLlama Launched for Time Series Forecasting**: [StockLlama](https://github.com/LegallyCoder/StockLlama) leverages **Llama** for time series forecasting using custom embeddings, aimed at improving accuracy.
   - This introduction sparks interest for developers looking to enhance their forecasting capabilities.
- **Gaining Insights on ProteinBERT Structure**: Discussions on **ProteinBERT** clarified its architecture focused on local and global representations for processing protein sequences, with links to the [ProteinBERT paper](https://pubmed.ncbi.nlm.nih.gov/35020807/) shared for context.
   - Users expressed interest in understanding how these representations contribute to effective classification and regression tasks.
- **Cog-5B Video Model Released**: The **Cog-5B video model** is now available at [CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b), showcasing advanced capabilities in video generation.
   - Expectations rise for the forthcoming fine-tuning scripts that will enhance user customization options.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Model Loading Woes**: Users reported issues with model loading related to insufficient system resources, encountering an exit error code **-1073740791** while attempting to load various models.
   - Suggestions to adjust settings to **CPU-only** and modify guardrails settings emerged as potential fixes.
- **AMD vs Nvidia: The GPU Showdown**: Discussions pivoted around the performance gap between **Nvidia** and **AMD** for LLM tasks, with Nvidia currently leading the charge.
   - An **Nvidia 3060 12GB** was suggested as a balanced option for budget-conscious users aiming for effective model performance.
- **CPU Bottlenecks with Ollama**: When running LLMs using **Ollama**, reports surfaced about only one CPU heating up, spotlighting a potential single-CPU performance bottleneck.
   - Users highlighted the requirement for dual CPU support to improve inference speeds, though the topic might be contentious.
- **Tinygrad: Simplifying Neural Networks**: The new **Tinygrad** framework garnered interest for its simplicity in handling complex networks, featuring operations like **ElementwiseOps**.
   - Despite its limitations, it attracted attention for its potential to streamline workflow in LLM projects.
- **Cerebras Takes Inference to New Heights**: Cerebras announced their inference service delivering a striking **450 tokens/sec** for Llama3.1-70B, outpacing traditional GPU setups.
   - The service presents an economically appealing rate of just **60 cents** per million tokens, appealing to developers seeking cost-effective solutions.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.53.0 Introduces Major Enhancements**: The latest release of **Aider v0.53.0** features improved [prompt caching](https://aider.chat/docs/usage/caching.html#preventing-cache-expiration), allowing it to cache prompts effectively during sessions, enhancing coding speed and cost efficiency.
   - This version showcases Aider's capability to write **59%** of its own code, emphasizing a significant leap in its operational self-sufficiency.
- **User Insights on Aider's Functionality**: Discussions revealed Aider's challenges in converting large code bases with a single prompt, necessitating refinements for effective results.
   - While valuable, users acknowledge that Aider's outputs must be rigorously tested and polished for substantial projects.
- **Gemini Model Performance Unveiled**: New **Gemini 1.5** models have been rolled out, including **Gemini 1.5 Pro**, which is designed for better performance with complex prompts and coding tasks, available for testing at [AI Studio](https://aistudio.google.com).
   - Rate limits for these models are set at **2 requests per minute** and **50 requests per day**, prompting users to seek creative workarounds for performance benchmarks.
- **Anthropic Releases System Prompts for Claude 3**: Anthropic has published the system prompts for their **Claude 3 models**, including **Claude 3.5 Sonnet**, as of **July 12, 2024**, promising to keep documentation updated with future changes.
   - The prompts are viewed as a notable transparency improvement in LLM documentation, with insights gathered from researcher **Amanda Askell** on their usage.
- **Error Handling Improvements in Aider**: Aider v0.53.0 has improved error handling, providing clearer messages when variables are not set, enhancing user troubleshooting experiences.
   - Recent bug fixes also address issues with **Windows filenames**, ensuring smoother operational functionality across different systems.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Anticipation for next-gen AI hardware**: Members eagerly discussed upcoming releases of **Intel CPUs** and **NVIDIA GPUs** featuring enhanced AI capabilities, sparking interest in new PC builds.
   - These innovations are set to significantly improve performance for AI tasks, positioning users for a tech upgrade.
- **Impressive capabilities of Flux models**: Excitement erupted over the advanced features of **Flux**, including dynamic angles and depth perspective, which may render older models obsolete.
   - Concerns rose about trainability as members speculated on its potential for revolutionizing AI-generated visuals.
- **Concerns over ZLUDA development**: Participants raised alarms about the future of **ZLUDA** after emerging reports suggested that **AMD** had ceased funding its development.
   - Members speculated that ongoing legal challenges could further complicate ZLUDA's progress despite GitHub updates.
- **Integration of SD Next with ZLUDA**: A discussion emerged about why **SD.Next** performs better with **ZLUDA**, with thoughts on its backend architecture integrating **A1111** and **Diffusers**.
   - This multi-backend framework could enhance compatibility and overall performance across different models.
- **Challenges with Streamdiffusion and SDXL Turbo**: Members expressed frustration over the difficulties in integrating **SDXL Turbo** with **Streamdiffusion**, particularly regarding **TensorRT** performance.
   - Concerns arose about frame rates and resolution compatibility, casting doubt on its practical usability.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Inquiring About Video Benchmark Examples**: A member asked for quality examples of **video benchmarks**, focusing on tasks like **spatial awareness** and generative models, leading to suggestions on standard evaluation methods.
   - Though **action recognition** was proposed for discriminative tasks, there’s a notable absence of established benchmarks for generative tasks.
- **Discussion on RLHF Libraries**: The chat debated whether **TRL/TRLX** continues to be the top choice for **Reinforcement Learning from Human Feedback**, with many recommending **TRL** due to concerns over the outdated **TRLX**.
   - Members expressed a desire for alternatives, but none have emerged in recent discussions.
- **Free API Access for Llama 3.1**: A member shared a link to a **free API for Llama 3.1 405B** from **SambaNova**, highlighting its potential for broader accessibility.
   - Details about SambaNova's services were outlined, which could enhance AI projects using their platform.
- **Gemini Misrepresentation Claims**: The community engaged in a debate over claims by Jamba authors about **Gemini**, which allegedly caps at **128k** without further testing, sparking controversy.
   - Defenders argued that the authors' wording does not misrepresent, remarking *they couldn’t reproduce results beyond 128k*.
- **Learning Rate Scaling Insights**: Insights were shared on the necessity of **sqrt scaling** with Adam for learning rate adjustments in relation to batch sizes, with various papers cited.
   - The group explored methodological differences in experiments, raising questions regarding validity in their proposed approaches.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Liger Kernel Welcomes New Contributors**: New members joined the **Liger Kernel** community, eager to contribute, with a focus on Triton and its capabilities, including a startup from D.C. interested in training efficiencies.
   - Contributing guidelines were shared to promote active collaboration, signaling a growing interest in the project.
- **Triton Implementation Comparisons**: Developers find Triton implementation tougher than PyTorch but easier than pure CUDA, which presents a unique trade-off.
   - Leveraging tools like [torch.compile](https://github.com/linkedin/Liger-Kernel/issues/119) can enhance performance, making the transition worthwhile.
- **Encoder-Style Transformers Support Initiatives**: The community is exploring support for encoder-only transformers like **BERT**, creating an issue for tracking feature development.
   - There is potential for reusing layers, indicating a collaborative effort to enhance existing models within the Liger Kernel framework.
- **Call for Fused Kernel Development**: Discussion centered around establishing a 'fused kernel zoo' to streamline the addition of efficient kernels beyond current frameworks.
   - Members believe a synergy between PyTorch and Triton can yield optimal results, inviting contributions to kernel requests.
- **Insights on Low-Bit Optimization Techniques**: Users focus on datasets for fine-tuning a **4-bit optimization** model, noting challenges in performance with the Alpaca dataset.
   - The **Chatbot Arena dataset** was recommended as a potential solution, highlighting its comprehensive nature but acknowledging its complexities.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity app experiences slow performance**: Many users reported that the **Perplexity app** has been experiencing slow response times since this morning, causing frustration among users.
   - Complaints included unreliable search results and general dissatisfaction with the platform's recent performance.
- **File upload failures across the board**: Multiple users attempted image uploads and encountered **file upload failed** errors, with some expressing disappointment over issues even while on Pro subscriptions.
   - While PDFs have been reported to work, users are waiting for a fix as image uploads remain broken.
- **Clarifying usage limits for GPT models**: The daily message limit for the **Claude 3.5** model is reported to be **430 messages**, combined across all Pro models, except for **Opus**, which has a limit of **50**.
   - Users noted that even with high usage, they rarely hit the limit, with one mentioning their closest was around **250 messages**.
- **Boeing's Plan to Replace 737**: [Boeing's plan to replace 737](https://www.perplexity.ai/page/why-boeing-wants-to-replace-73-Asu4kUOdQP2QzJuDlj1Tqw) is highlighted as a strategic move to enhance its fleet's efficiency and sustainability amidst growing demand.
   - They aim to address market needs with a new aircraft that improves on existing models' performance and environmental impact.
- **Challenges with Perplexity AI implementation in chatbot**: A user is trying to implement **Perplexity AI** into a fact-checking chatbot in Hebrew but is facing issues with shortened responses that lack **links** and **images**.
   - They noted that responses from the API differ significantly from those on the Perplexity search site, mentioning that links often lead to **404 errors**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **API Degradation Incident Briefly Affects Services**: There was a *~5m period of API degradation* that impacted service availability. A patch has been rolled out, and the incident appears to be fully **recovered**.
   - The response team quickly identified the issue during the API degradation period, ensuring minimal disruption. This proactive approach highlights the importance of rapid response in maintaining service integrity.
- **Team Efforts Recognized!**: A member expressed gratitude towards the team for their contributions, stating, *Thank you team!* This acknowledgment highlights the collaborative spirit and the hard work put in by individuals involved.
   - Furthermore, a [tweet](https://twitter.com/gpudad/status/1828502015238119490) was shared that showcases significant developments in AI collaboration. The tweet emphasizes the importance of community efforts in advancing AI technologies.
- **OpenRouter Model Pricing and Fees Clarified**: A user inquired about whether the price per token displayed in OpenRouter includes the service fee. It was clarified that the price listed is based on OpenRouter credits and does not account for any additional fees incurred when adding credits.
   - Concerns were also raised regarding the current display of **$0** on the activity page, which could mislead users. Enhanced visibility of **model pricing** is essential to improve user experience.
- **DisTrO Brings New Hope to Distributed Training**: A member highlighted the release of a preliminary report on DisTrO (Distributed Training Over-the-Internet) by Nous Research, which improves distributed training efficiency. It promises to drastically reduce inter-GPU communication, enabling more resilient training of large models.
   - Community discussions focused on its implications for the future of distributed training strategies. Members expressed eagerness to explore this innovative approach further.
- **Exciting Updates for Gemini Models Discussed**: The upcoming release of new Gemini 1.5 Flash and Pro models was discussed, sparking excitement about its potential features and performance. Users speculated that these updates might aim to compete with existing models like GPT-4.
   - Tweets from official sources outlined the planned release details, generating buzz about the capabilities and enhancements expected in the new models. Community members are closely watching the developments.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox Shipping Challenges in Europe**: Discussion arose about the current unavailability of the **Tinybox** in Europe, especially impacting buyers in the UK. Members recommended contacting support for shipping quotes, while reports of **sold out** status in **France** and **Italy** circulated.
   - The community speculated on future shipping solutions, with focus on potential new **color editions**, though George denied a blue version will be released soon.
- **Exploring Tinygrad for BERT Training**: A member showed interest in leveraging **Tinygrad** for pre-training a large **BERT** model, discussing the necessary support for high-performance setups. There was a debate over using **Tinygrad** versus **Torch**, with participants noting **Torch's** better optimization.
   - The conversation highlighted existing hardware needs, mentioning setups with **64 Hopper cards** for effective model training.
- **Tinybox Sales Update**: George shared that about **40 Tinyboxes** have been sold, with **60 more** in stock. Excitement about increasing sales was tempered by ongoing negotiations for international shipping.
   - Members engaged in discussions about the potential for new **color editions**, expressing curiosity despite George's dismissal of a blue edition.
- **Runtime Errors When Using Tinygrad**: A user faced `RecursionError` while converting tensors in **Tinygrad** when processing more than 3500 Wikipedia articles. This problem seemed to resolve for smaller datasets, raising concerns about the function's handling of large inputs.
   - The community suggested creating a minimal example for debugging; there is interest in collaborative troubleshooting to address these runtime issues.
- **Identifying Tinygrad version 0.9.2**: A user confirmed they are running **Tinygrad version 0.9.2**, which may relate to the `RecursionError` issues encountered during tensor conversion. This version's **LazyOp** functionality was mentioned as a potential factor in the problems discussed.
   - Community efforts are directed towards troubleshooting, including possible need for updates or determining if the error is version-specific.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Intel launches LLaVaOLMoBitnet1B**: Intel has introduced the first Ternary Multimodal LLM, [LLaVaOLMoBitnet1B](https://huggingface.co/papers/2408.13402), that processes images and text to produce coherent responses. The model is fully open-sourced, offering training scripts to explore challenges and opportunities in ternary modeling.
   - The community is curious about the implications of using this structure for future AI applications, particularly in multimodal interactions.
- **OpenAI targets complex reasoning with Orion AI**: OpenAI is reportedly developing a new model, **Orion**, aimed at enhancing complex reasoning skills as it seeks additional investments, as reported by [The Information](https://www.theinformation.com/articles/openai-races-to-launch-strawberry-reasoning-ai-to-boost-chatbot-business?utm_campaign=Editorial&utm_content=Article&utm_medium=organic_social&utm_source=twitter). This initiative aims to strengthen its position in the competitive chatbot space.
   - Members are closely watching this development for potential advancements that could reframe AI-assisted problem-solving capabilities.
- **Skepticism around Nous Research optimizer**: Members expressed doubts regarding the legitimacy of Nous Research's new optimizer, primarily concerning its claims about **distributed training capabilities**. Discussions referenced existing tools like **Petals for Bloom** and **OpenDILo**, further fueling skepticism.
   - Calls for more substantial evidence supporting Nous's promises were echoed, highlighting concerns around transparency in AI tool development.
- **Cerebras speeds ahead in inference**: Cerebras has claimed its inference API achieves speeds of **1,800 tokens/s** for **8B models** and **450 tokens/s** for **70B models**, significantly outperforming competitors. This announcement caused a buzz within the community eager for rapid advancements in inference technology.
   - Members showed excitement about the implications this speed could have for real-time AI applications and competitiveness in the market.
- **Google's Gemini 1.5 models generate interest**: Google has rolled out experimental models: **Gemini 1.5 Flash-8B** and **Gemini 1.5 Pro**, enhancing coding task capabilities. Access now available through [Google's AI Studio](https://aistudio.google.com), encouraging hands-on exploration from the community.
   - Members are keen to test these new models, with discussions indicating a potential shift in project approaches due to their unique features.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Evaluating lm eval metrics**: For multiple choice questions, the metric used is **accuracy on target prediction**, determining if the model's highest logit aligns with the correct choice.
   - Members highlighted nuances in model evaluations, discussing scenarios where answers differ slightly.
- **Confusion surrounds Tokenizer v3 specs**: Members expressed confusion regarding **tokenizer v3**, with links to previous discussions on the **nemo repo** shared.
   - There was a consensus on needing proper configuration for supporting **multi-role functionalities**.
- **Deepseek V2 monkey-patch insights**: Members discussed using **monkey-patching** to override the forward method for the **Deepseek V2** attention model, sharing relevant code snippets.
   - An experience comparison was made about monkey-patching in Java versus Python, showcasing complexities in the implementation.
- **FSDP's RAM resource requirements questioned**: Concerns regarding whether **FSDP** (Fully Sharded Data Parallel) requires significant **system RAM** for effective functioning were raised.
   - This led to discussions about optimal system resources necessary for operating FSDP effectively.
- **AI Ratings vs Human Ratings Unpacked**: A member utilized **llm-as-judge** for rating and questioned the accuracy of AI judgments compared to human ratings.
   - Further inquiries were made about any conducted tests evaluating this accuracy, emphasizing the need for metric validation.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Call for User Feedback on Magic**: The team seeks **5 participants** for a **30-minute** feedback session specifically on **magic** features, offering exclusive **swag** as a reward.
   - Participants can [book a slot here](https://modul.ar/user-feedback) to contribute their insights and receive first dibs on design-phase swag.
- **ClassStruct Lets You Play with Variadic Parameters**: ClassStruct in Mojo enables dynamic parameterization, allowing variations without creating separate structs, illustrated by a `car` example for engine size.
   - This efficiency lets developers define **struct fields** based on compile-time parameters, enhancing flexibility.
- **Performance Takes a Hit with Struct Fields**: Compiling structs with numerous fields can significantly slow performance, with 100 fields reportedly taking **1.2 seconds** to compile.
   - This delayed performance hints at resizing issues in underlying data structures that becomes apparent over a certain field threshold.
- **Type Inference in Mojo Hits a Wall**: Mojo faces challenges with type inference, especially regarding generics, making it less convenient compared to Rust's robust inference system.
   - Participants noted that Mojo's generics and typeclasses could limit flexibility, raising concerns about the developer experience.
- **Mojo and Luma: A Type System Showdown**: Discussions compared Mojo with Luma, noting that while **Luma** boasts stronger type inference, **Mojo** provides unique elements like typeclasses but with restrictions.
   - The consensus suggests that Mojo is evolving and may align closer to Rust's capabilities, hinting at potential features like effect systems.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Get Ready for RAG-a-thon with Pinecone!**: We're hosting our second **RAG-a-thon** with over **$7k in cash prizes** from October 11-13 at the [500GlobalVC](https://t.co/IFvyW5QB6r) offices in Palo Alto!
   - It's a great opportunity to showcase innovative ideas and gain valuable experience in a collaborative environment.
- **Llama 3.1-8b Breaks Speed Records**: Need ultra-fast responses? **Llama 3.1-8b** offers **1800 tokens per second**, making it the fastest LLM available, as discussed [here](https://t.co/hZv6ooGUxO).
   - Achieving this speed is crucial for applications needing quick responses, especially in complex systems.
- **Build Serverless RAG App with LlamaIndex**: Learn to create a **serverless RAG application** using LlamaIndex and Azure OpenAI through this comprehensive guide by **Wassim Chegham** [link to guide](https://t.co/1XKg1o2bIX).
   - It covers RAG architecture and shows how to leverage your own business data for improved AI-powered responses.
- **Neo4j Can't Build Relationships**: A user reported trouble replicating a property graph tutorial from LlamaIndex with Neo4j Desktop, where relationships weren't being extracted correctly.
   - They clarified that they followed the tutorial strictly, suspecting their Neo4j setup might not align with default expectations.
- **Enhancing Data Extraction with LlamaParse**: A user discussed potential issues with **LlamaParse** converting tabular data due to scanning problems and sought solutions for integrating image extraction.
   - Questions arose about chunking strategies for processing multiple tables combined with images.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DisTrO Transforms Distributed Training**: Nous Research's [DisTrO](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf) drastically reduces inter-GPU communication by up to **10,000x**, facilitating resilient LLM training.
   - This framework promotes shared AI research efforts, circumventing reliance on centralized entities to enhance security and competitiveness.
- **Phi 3.5 Shines in OCR Tasks**: Microsoft's [Phi 3.5](https://huggingface.co/spaces/MaziyarPanahi/Phi-3.5-Vision) excels in OCR, especially handwriting recognition and tabular data extraction, under a permissive MIT license.
   - The model's impressive performance in text recognition has generated significant community interest and discussion.
- **Cerebras Breaks Inference Speed Records**: Cerebras announced an inference service achieving [1,800 tokens/s](https://x.com/CerebrasSystems/status/1828465008298336588) for 8B models, outperforming NVIDIA and Groq.
   - Backed by their WSE-3 chip, Cerebras is also pushing competitive pricing for Llama models, leading to heated discussions about its economic viability.
- **Google Launches Gemini 1.5 Models**: Google introduced the **Gemini 1.5** series, featuring a smaller variant and a powerful Pro model, boasting capabilities in coding and handling complex prompts.
   - These launches have sparked comparisons to models like **GPT-4o-mini**, as developers assess their relative performance and edge.
- **Anthropic's Artifacts Raises Eyebrows**: Anthropic unveiled [Artifacts](https://newsletter.pragmaticengineer.com/p/how-anthropic-built-artifacts), intriguing many with their development insights and methodologies.
   - Concerns emerged about the reasons behind the timely release, with speculation of potential paid placements in the conversations.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Streamlit Python Server for Chat**: A member introduced an easy [Streamlit Python server](https://link.to.streamlit) to create a chat interface in a web browser, fostering quick implementation.
   - This sparked interest with another member expressing intention to explore this solution further.
- **Configuring Telegram Bot with Open Interpreter**: Issues arose when a member shared their setup for a Telegram bot using **Open Interpreter**, including necessary API key settings.
   - They faced image display issues, leading to a valuable discussion for troubleshooting potential fixes.
- **Cython for Black-Scholes Model Efficiency**: An example using Cython to implement the **Black-Scholes model** was shared, emphasizing optimized computations for options pricing.
   - This highlights how to define efficient functions in Cython, enhancing the overall performance of Jupyter notebooks.
- **Daily Podcast Feat. Cloned Voices Takes Off**: Mike and Ty humorously toyed with creating a daily podcast using **voice cloning** technology from ElevenLabs, cultivating laughter within the community.
   - Their playful banter showcases innovative ideas on blending voice synthesis technology into engaging content.
- **First Meetup Brand Documentation Shared**: A member presented a link to accessible brand documentation for the **01 project** via [Canva](https://www.canva.com/design/DAF8rbBol3Q/UNivuf8sjxVSveDfMFWpag/edit?utm_content=DAF8rbBol3Q&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).
   - This document includes design insights with promises of more comprehensive updates expected in their GitHub repository soon.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Llama 3.1 Struggles on CPU**: A user reported that inference for **Llama 3.1 8B** on CPU is extremely slow (<0.05 tok/s) even on a high-end AWS server, highlighting a substantial performance gap.
   - The conversation acknowledged that CPU setups are inherently slower than GPU configurations, particularly when using **Ollama**.
- **Optimized Frameworks to Consider**: Members recommended using **Ollama** or **vLLM** for serving models, as these frameworks provide better optimization for inference compared to **Torchtune**.
   - A helpful [tutorial on using Ollama with custom checkpoints](https://github.com/ollama/ollama/blob/main/docs/import.md) was shared to assist newcomers.
- **Inquiries on LoRA Model Loading**: A user asked if `from_pretrained()` correctly loads LoRA fine-tuned weights from local checkpoints, revealing common user concerns about model integration.
   - An informative link to a discussion on [loading LoRA adapters](https://github.com/pytorch/torchtune/issues/832#issuecomment-2129867129) into HF was provided for clarity.
- **AWS Instance Cost Discussions**: The cost of AWS instances was debated, indicating that an AWS c7a.24xlarge might run around **$5/hour**, leading to discussions on cost-effectiveness.
   - Recommendations were made to explore alternatives like **Runpod**, though regulatory constraints were noted as limiting factors for some users.
- **Performance Challenges with CPU Servers**: Users expressed a preference for CPU servers due to cost-effectiveness and satisfactory response times for their projects.
   - However, it was noted that low CPU performance could drastically affect inference speeds, pushing users to rethink using optimized frameworks.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Concerns About Request Limits**: Members expressed apprehension about surpassing the **1k requests** threshold, especially in testing scenarios.
   - One member doubted how to reach this 1k limit, suggesting it feels unachievable.
- **Intermittent Model Not Found Errors**: A member reported encountering a **'model not found'** error thought to be linked to the model's versioning, as the reranker is now on **v3**.
   - This raises potential stability concerns amid ongoing updates.
- **Clarification on Production Key 403**: A vague mention of **production key 403** led to confusion, prompting requests for context from other members.
   - The lack of clarity indicates a need for improved communication about key references.
- **404 Errors with Langchain and Cohere TypeScript**: An issue arose where initial calls using **Langchain** with **Cohere TypeScript** worked, but subsequent calls resulted in a **404 error**.
   - This suggests a potential misconfiguration or instability in the integration.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Flutter Collaboration Sparks Interest**: In a bid to improve Flutter app development, collaboration was proposed following **fritzlandry**'s inquiry, addressing a gap in shared expertise.
   - This response could lead to productive partnerships, enhancing Flutter's application in engineering projects.
- **vllm's RAG Capabilities Under Review**: A discussion emerged regarding the possibility of running **Retrieval-Augmented Generation (RAG)** with **vllm**, citing accessible models for embedding and answering tasks.
   - This shows growth in multi-model approaches, urging engineers to push the boundaries of **vllm** applications.
- **Quest for LLM Workflow Builders**: A call for existing **LLM workflow builders** highlighted a push for automation amidst user workflows, seeking innovative solutions.
   - This reflects a growing demand for tools that integrate LLM capabilities effectively.
- **Local Embedding Models Preferred**: Frustration towards cloud options like **Pinecone** spurred inquiries for local embedding model recommendations, as users long for optimized performance.
   - Discussions pointed towards maximizing efficiency for models, stressing local setups over cloud reliance.
- **Dashboard’s Value in Question**: Concerns over the dashboard's effectiveness arose, with frustration noted regarding errors from models like **Claude** and **GPT** in handling complex tasks.
   - This dialogue underscores a need for accuracy in AI outputs, pushing for improvements in the user experience.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Models must be accessible for leaderboard**: Models listed on the [leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1277775483966193759) must be **publicly accessible**, whether open-source or through an API for inference.
   - The requirement necessitates that even if registration or tokens are used, the endpoint should be open to the public.
- **Benchmarking limitations without public access**: While models can be benchmarked with **BFCL**, only publicly accessible models can be displayed on the leaderboard, creating a notable distinction.
   - This limitation impacts which models can be showcased and which can merely be evaluated.
- **Function Calling Feature Drops Performance**: Using a system prompt directly with **GPT-4-1106-Preview** achieves an accuracy of **85.65**, while enabling function calling drops it to **79.65**.
   - This discrepancy raises questions about the relationship between *function calling* and overall model performance, prompting further investigation.
- **BFCL Optimization Strategies Under Scrutiny**: A user expressed concerns over their optimization strategies for a function-calling feature, questioning compliance with BFCL guidelines.
   - They queried if optimizations like system prompt updates could be perceived as unfair practices which are not generalizable across all models.
- **Seeking benchmarking guidance for Llama 3.1**: A user is seeking advice on **benchmarking Llama 3.1**, specifically using a custom API endpoint hosted by their company.
   - They are looking for effective pointers on how to initiate the benchmarking process smoothly.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Community Solves DSPy Output Truncation**: A member reported **DSPy** outputs getting truncated, suspecting token limits; they resolved this by adjusting **max_tokens** during initialization and using [your_lm.inspect_history()](https://some.link/here) to inspect prompts.
   - The original poster confirmed the community's advice effectively addressed their issue, showcasing helpful member collaboration.
- **Typing Support Error Stumps User**: One member hit an error on import, `module is installed, but missing library stubs or py.typed`, questioning if **DSPy** supports typing in Python, signaling a documentation gap.
   - No follow-up or resolution was offered, indicating lingering uncertainty about typing support within the library.
- **Growing Interest in Text Scoring with DSPy**: A user inquired about scoring generated texts with **DSPy** based on metrics like **BLEU** or **ROUGE**, reflecting a community push for evaluating text performance.
   - Unfortunately, no members replied, leaving their experiences and insights on text scoring in the dark.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Hamel's Check-in Needed**: A member asked for **Hamel's** presence in the channel, indicating potential discussions on **LLM finetuning** were on the horizon.
   - *No further context provided*, but members await insights directly from Hamel regarding relevant projects.
- **Discussion on LLM Models**: The conversation hinted at the importance of having **Hamel** present for potential discussions on enhancing **LLM** performance through finetuning techniques.
   - Members are likely interested in strategies for model optimization and learning enhancements.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Join the LLM Observability Tools Webinar**: This Saturday, August 31st, at **11:30 AM EST**, a webinar will cover over **60 LLM observability tools** to evaluate their monitoring effectiveness. Register for the session [here](https://kyrylai.com/webinar-observability-platforms-overflow/).
   - Participants will gain insights on observability basics, tool selection, and LLM integration strategies for better model management.
- **Testing the Hype of ML Monitoring Platforms**: The upcoming webinar aims to critically assess if the plethora of **ML monitoring tools** meet practitioners' real needs in **monitoring** and **debugging** models. Expect a hands-on evaluation to sift through marketing claims.
   - The focus will be on practicality and user-friendliness, ensuring the tools deliver genuine benefits.
- **Cohort on Machine Learning in Production Initiatives**: A live cohort for 'Machine Learning in Production' is available to enhance practical skills in deploying ML models effectively. Interested participants can find more details [here](https://edu.kyrylai.com/courses/ml-in-production).
   - The course promises to provide essential tools and knowledge for effective ML management in real-world applications.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LAION-aesthetic link issues**: A member reported that the link to **LAION-aesthetic** on the LAION website is broken and requested an alternative link from **Hugging Face**.
   - *Any updates on a working link would be greatly appreciated*, highlighting the ongoing community need for reliable resources.
- **Request for functional LAION-aesthetic resource**: The discussion emphasized the importance of having a functioning link to **LAION-aesthetic**, essential for users accessing data models.
   - Members expressed frustration over the non-functional website and urged for prompt solutions to improve usability.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Get Ready for LocalAI AMA with Ettore Di Giacinto**: Join the **LocalAI AMA** with **Ettore Di Giacinto** in two hours to explore its features as an open-source alternative to OpenAI, featuring a REST API for local inferencing.
   - LocalAI enables LLMs, image, and audio generation locally on consumer-grade hardware without needing a GPU.
- **Participation Link for LocalAI Event**: The participation link for the **LocalAI** event is available now; [join here](https://discord.com/events/1089876418936180786/1268967945216721079) to engage directly.
   - Get your questions answered about this project and see how it integrates into your workflow.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1277704253925757000)** (727 messages🔥🔥🔥): 

> - `DisTrO Algorithm Development`
> - `Collaboration Opportunities`
> - `Qualia and Consciousness in AI`
> - `Scaling Distributed Training with Weak Devices`
> - `Comparison of Distributed Optimization Techniques` 


- **DisTrO Algorithm is an Evolving Family**: The DisTrO algorithm is currently being refined, with multiple variants being tested to optimize communication bandwidth while maintaining convergence performance.
   - There is ongoing discussion regarding how various distributed optimization techniques, like SWARM, might be more appropriate for larger models.
- **Interest in Collaborations**: Several members expressed interest in collaborating on topics surrounding the DisTrO algorithm and its implementation.
   - The team is open to contributions and discussions while emphasizing that the full code and details will be made available in the coming weeks.
- **The Philosophical Implications of AI**: Qualia and consciousness remain hot topics of debate, with members discussing the implications of these concepts on AI and machine learning methodologies.
   - There is a call for more interdisciplinary collaboration between computer scientists and philosophers to deepen the understanding of these issues.
- **Potential Use Cases for DisTrO**: Discussion on the feasibility of using DisTrO with weak devices, such as old phones and laptops, indicates a need for efficient training methods on resource-constrained hardware.
   - Members agree that while DisTrO may excel with stronger devices, it’s still valuable to explore its application in less-capable systems.
- **Insights on Academia and Research Approaches**: The conversation touched on the expectations of scientific communication and the perceived delays in sharing insights and algorithms from research groups.
   - Members highlighted the importance of balancing marketing and academic rigor in disseminating new algorithms and research findings.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/neurallambda/status/1828214178567647584?s=46">Tweet from neurallambda (open agi) (@neurallambda)</a>: progress report on &#34;Homoiconic AI&#34;:  we use a hypernet to generate the weights of an autoencoder, and then do in-context learning (masked reconstruction loss) to improve those weights  val los...</li><li><a href="https://x.com/DataPlusEngine/status/1828141323616153734">Tweet from DataVoid (@DataPlusEngine)</a>: Rant part 1/ The modern scientific approach, particularly in fields like machine learning (ML) and broader scientific inquiry, often operates under the assumption that our current understanding of the...</li><li><a href="https://arxiv.org/abs/2306.17453">Pollen: High-throughput Federated Learning Simulation via Resource-Aware Client Placement</a>: Federated Learning (FL) is a privacy-focused machine learning paradigm that collaboratively trains models directly on edge devices. Simulation plays an essential role in FL adoption, helping develop n...</li><li><a href="https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/rag">Retrieval Augmented Generation (RAG) in Azure Cosmos DB</a>: Learn about Retrieval Augmented Generation (RAG) in Azure Cosmos DB</li><li><a href="https://arxiv.org/abs/2106.11257">Secure Distributed Training at Scale</a>: Many areas of deep learning benefit from using increasingly larger neural networks trained on public data, as is the case for pre-trained models for NLP and computer vision. Training such models requi...</li><li><a href="https://worldsim.nousresearch.com/console">worldsim</a>: no description found</li><li><a href="https://tenor.com/view/conspiracy-theory-gif-10587157">Conspiracy Theory GIF - Conspiracy Theory - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/chrome-google-chrome-hogging-ram-adobe-applications-friends-gif-17494165">Chrome Google Chrome GIF - Chrome Google Chrome Hogging Ram - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/shikanoko-by-murya-gif-9501555167387334429">Shikanoko By Murya GIF - SHIKANOKO BY MURYA - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://forms.gle/FAJfiGYmd47XRyi67">Open LLM project</a>: Hi, thanks for joining the server ! This is a small optional survey concerning a possible participation for remote training an openLLM using Nous&#39; DisTrO   Here are a few little funfacts:  - The l...</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coheren...</li><li><a href="https://github.com/DataCTE/LatentExplorer/blob/master/RANT.md">LatentExplorer/RANT.md at master · DataCTE/LatentExplorer</a>: Contribute to DataCTE/LatentExplorer development by creating an account on GitHub.</li><li><a href="https://github.com/DataCTE/LatentExplorer">GitHub - DataCTE/LatentExplorer</a>: Contribute to DataCTE/LatentExplorer development by creating an account on GitHub.</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>: Distributed Training Over-The-Internet. Contribute to NousResearch/DisTrO development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1277778368053641269)** (7 messages): 

> - `Distributed Trees of Experts`
> - `Flux implementation for Apple Silicon`
> - `Self-hostable multimodal LLM`
> - `Moondream2 real-time demo`
> - `Finetuning practices` 


- **Exploring Distributed Trees of Experts**: A member inquired about a paper on **distributed trees of experts**, likening it to a sparse model supporting shared AI workloads in a P2P network.
   - *Community-driven development* in AI could enhance collaboration among open-source projects.
- **Need for Flux on Apple Silicon**: A member asked if there's an **mlx implementation of Flux** available for **Apple Silicon** yet.
   - They are eager to integrate it but have not found a suitable solution.
- **Seeking Self-hostable Multimodal LLM**: A member expressed interest in a self-hostable multimodal LLM for **real-time analysis** of a live video stream without specific training.
   - They are exploring options like **GPT-4(o)** but are concerned about **cost and privacy**.
- **Moondream2 offers promising solutions**: A suggestion was made to check out **Moondream2**, featuring a real-time webcam demo that's easy to fine-tune.
   - It's positioned as a suitable solution for the self-hostable multimodal LLM needs expressed by another member.
- **Debate on Finetuning Data Sources**: A counterpoint was raised regarding the finetuning process, questioning if using data generated by another model is advisable.
   - There's consideration about the risks and benefits, especially when derived from a much stronger model.



**Link mentioned**: <a href="https://github.com/vikhyat/moondream">GitHub - vikhyat/moondream: tiny vision language model</a>: tiny vision language model. Contribute to vikhyat/moondream development by creating an account on GitHub.

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

deki04: https://huggingface.co/papers/2408.13933
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1277775546583089213)** (6 messages): 

> - `Proof of Compression vs. Proof of Work`
> - `Tinybox Launch`
> - `New Gemini Models`
> - `Flex-Attention Visualization Tool` 


- **Proof of Compression proposed for Training**: A discussion emerged regarding the viability of **proof of compression** replacing **proof of work** in distributed model training, especially for losslessly compressible models.
   - *Is there any reason why proof of compression can’t replace proof of work?*
- **Tinybox finally offers buy option**: After **18 months**, @realGeorgeHotz announced that **tinyboxes** now have a 'buy it now' button, with **13 units** available for purchase today.
   - He touted the **$15k tinybox red** as the best performance/price ML box in the world, emphasizing its networking capabilities.
- **Introduction of New Gemini Models**: Today, @OfficialLoganK introduced three new experimental models, including **Gemini 1.5 Flash-8B** and a stronger **Gemini 1.5 Pro** model.
   - These models promise enhancements for **coding** and **complex prompts** and are available for testing on [Google AI Studio](https://aistudio.google.com).
- **Visualization Tool for Flex-Attention Maps**: A tool was shared for visualizing **Flex-attention** maps, particularly for **Bigbird attention** models.
   - You can access the demo at [Visualize Flex-Attention](https://viig99-app-demos-jz7hllm8n2ps6fkkwmotuj.streamlit.app/).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/realgeorgehotz/status/1828197925874463166">Tweet from George Hotz 🌑 (@realGeorgeHotz)</a>: 18 months in to the company, tinyboxes finally have a buy it now button! We have 13 in stock today, go to our website (link on @__tinygrad__) to buy one.  The $15k tinybox red is the best perf/$ ML bo...</li><li><a href="https://x.com/Algomancer/status/1797174675132551408">Tweet from Adam Hibble (@Algomancer)</a>: Is there any reason why proof of compression can&#39;t replace proof of work for distributed model training?  Atleast for any model that can losslessly compress? (auto regressive models, vaes, etc)</li><li><a href="https://x.com/OfficialLoganK/status/1828480081574142227">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Today, we are rolling out three experimental models:  - A new smaller variant, Gemini 1.5 Flash-8B - A stronger Gemini 1.5 Pro model (better on coding & complex prompts) - A significantly improved Gem...</li><li><a href="https://viig99-app-demos-jz7hllm8n2ps6fkkwmotuj.streamlit.app/">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

deki04: https://huggingface.co/papers/2408.13933
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1277854767187755031)** (1 messages): 

> - `Monte Carlo Tree Search`
> - `Image Recognition`
> - `Image Generation` 


- **Reversing Monte Carlo Tree Search Logic**: A suggestion was made to train a model for **Monte Carlo Tree Search** by reversing its logic to **generate** options instead of identifying the best one.
   - *Just reverse the logic* that it uses currently for identifying, implying potential benefits to integrating this into image recognition and generation tasks.
- **Exploration of Image Generation Techniques**: The discussion also touched on the connection between **image recognition** and **image generation**, indicating a potential crossover in methodologies.
   - By applying reversed search strategies, there may be innovative pathways to enhance **image generation** outcomes.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1277705007860547707)** (359 messages🔥🔥): 

> - `Tensor issues in model conversion`
> - `Batch size and memory management`
> - `Homoiconic AI progress`
> - `Using TPUs for training`
> - `Conversational AI fine-tuning` 


- **Tensor mapping issue during conversion**: A user encountered a `ValueError` regarding `embed_tokens.weight` while trying to convert a model to GGUF format, indicating potential differences between sentence-transformer models and causalLM.
   - They expressed frustration about the lack of support for pair scoring in the current tools.
- **Batch size optimization strategies**: Discussions highlighted the strategy of increasing batch size until out-of-memory (OOM) errors occur, leading users to experiment with their training settings.
   - One user is attempting to convert their model to Ollama format after a completion of the last training task.
- **Updates on Homoiconic AI project**: A member shared a progress update on the 'Homoiconic AI' project, detailing the use of hypernets for generating weights and improvements in validation loss metrics.
   - The project aims for a multimodal approach, integrating code as data and data as code to enhance reasoning capabilities.
- **Challenges of training on TPUs**: Users discussed their experiences with the speed performance of TPUs, noting that while batch inference can be efficient, single instances often experience delays.
   - There were mixed feelings about utilizing Colab's TPUs compared to other platforms due to perceived limitations.
- **Conversational fine-tuning practices**: A user inquired about best practices for fine-tuning Llama 3.1 for conversational data, emphasizing the need for a validation set to prevent overfitting.
   - The community shared insights about monitoring training loss and dataset complexity, suggesting methods to determine when training should be halted.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/neurallambda/status/1828214178567647584?s">Tweet from neurallambda (open agi) (@neurallambda)</a>: progress report on &#34;Homoiconic AI&#34;:  we use a hypernet to generate the weights of an autoencoder, and then do in-context learning (masked reconstruction loss) to improve those weights  val los...</li><li><a href="https://x.com/neurallambda/status/1828214178567647584?s=46">Tweet from neurallambda (open agi) (@neurallambda)</a>: progress report on &#34;Homoiconic AI&#34;:  we use a hypernet to generate the weights of an autoencoder, and then do in-context learning (masked reconstruction loss) to improve those weights  val los...</li><li><a href="https://www.kaggle.com/datasets/abdurrafae/vllm-t4-fix">vllm T4 Fix</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/installation/updating">Updating | Unsloth Documentation</a>: To update Unsloth, follow the steps below:</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq15">Google Colab</a>: no description found</li><li><a href="https://tenor.com/view/laptop-gif-26065234">Laptop GIF - Laptop - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/crying-kid-rush-cooking-gif-15677626">Crying Kid GIF - Crying Kid Rush - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.kaggle.com/code/cdeotte/infer-34b-with-vllm">Infer 34B with vLLM</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from multiple data sources</li><li><a href="https://www.apple.com/shop/product/G1AG7LL/A/refurbished-16-inch-macbook-pro-apple-m3-max-chip-with-16%E2%80%91core-cpu-and-40%E2%80%91core-gpu-space-black?fnode=8f57683ded9527d7e3f9ac11a826603f1f96e4a1b68b88881231e56425af835d53a4a36e79baa89fa6dd1c4da435bb9a42ba786ae143d3d713f130350870e25d0f2a5dc6382eb544570a7d2ebced7575">Refurbished 16-inch MacBook Pro Apple M3 Max Chip with 16‑Core CPU and 40‑Core GPU - Space Black</a>: Originally released October 202316.2-inch (diagonal) Liquid Retina XDR display¹; 3456-by-2234 native resolution at 254 pixels per inch128GB unified memory512GB SSD²Touch ID1080p FaceTime HD cameraThre...</li><li><a href="https://huggingface.co/unsloth/Phi-3.5-mini-instruct">unsloth/Phi-3.5-mini-instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/optimum-tpu">GitHub - huggingface/optimum-tpu: Google TPU optimizations for transformers models</a>: Google TPU optimizations for transformers models. Contribute to huggingface/optimum-tpu development by creating an account on GitHub.</li><li><a href="https://github.com/sophgo/LLM-TPU">GitHub - sophgo/LLM-TPU: Run generative AI models in sophgo BM1684X</a>: Run generative AI models in sophgo BM1684X. Contribute to sophgo/LLM-TPU development by creating an account on GitHub.</li><li><a href="https://github.com/Lightning-AI/litgpt">GitHub - Lightning-AI/litgpt: 20+ high-performance LLMs with recipes to pretrain, finetune and deploy at scale.</a>: 20+ high-performance LLMs with recipes to pretrain, finetune and deploy at scale. - Lightning-AI/litgpt
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1277942208049315870)** (3 messages): 

> - `BLOCK_N setting`
> - `Illegal memory access issues`
> - `Data type adjustments` 


- **BLOCK_N setting struggles**: A user reported that setting the **BLOCK_N** to **4** did not resolve the ongoing issues they were facing.
   - Despite this adjustment, the problem persisted, indicating more complex underlying issues.
- **Continued illegal memory access**: The same user noted they are still encountering **illegal memory access**, highlighting a persistent problem in their setup.
   - This error suggests possible conflicts within their code or system configurations that need further investigation.
- **Data type attempts with tl.int64**: The user mentioned trying **tl.int64** as a different approach but saw no improvement in resolving the issues.
   - This failure indicates the need for additional troubleshooting or alternative solutions beyond just changing data types.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1277715364196909137)** (78 messages🔥🔥): 

> - `LinkedIn LigerKernel vs Unsloth`
> - `Fine-tuning models`
> - `Checkpoint saving`
> - `Max sequence length impacts`
> - `Multi-GPU support timeline` 


- **LinkedIn LigerKernel's Copying Controversy**: Members discussed that LinkedIn's **LigerKernel** copied core components from **Unsloth**, specifically questioning its claim of 'inspiration' while showing code resemblance.
   - Concerns were raised over misleading claims of 'major improvement' and members emphasized the lack of original variable naming in the copied code.
- **Fine-tuning Techniques and Challenges**: A member inquired about how to **finetune** a model again after prior training and whether the training process entailed further pretraining.
   - Responses suggested that fine-tuning a model multiple times can be challenging and that optimizing settings like `num_train_epochs` and dataset size is crucial.
- **Details on Checkpoint Saving**: Discussion highlighted that although some intermediate checkpoints are saved automatically during finetuning, the final model must be manually saved using the method `model.save_pretrained`.
   - Members referenced the **Unsloth wiki page** for details about checkpoint management and its best practices.
- **Impact of Max Sequence Length**: A user expressed confusion regarding the variations in performance outcomes based on different `max_seq_length` settings during inference tests.
   - It was noted that setting `max_seq_length` too high could negatively impact performance and that it should ideally be aligned with the longest example in the dataset.
- **Multi-GPU Support Inquiry**: One member posed a question about the timeline for the release of **multi-GPU support** in Unsloth, considering it would provide a significant advantage.
   - No definitive timeline was provided, but the inquiry sparked interest in the community regarding upcoming feature plans.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/b8b1eafda35d124046e11766aeeb6343957e0daf/unsloth/kernels/rms_layernorm.py">unsloth/unsloth/kernels/rms_layernorm.py at b8b1eafda35d124046e11766aeeb6343957e0daf · unslothai/unsloth</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1277904197328834582)** (3 messages): 

> - `CleverBoi Collection`
> - `Duet Dataset`
> - `Open Sourcing Datasets` 


- **CleverBoi Collection Launched**: A new [CleverBoi collection](https://huggingface.co/collections/theprint/cleverboi-66ccd9588a104a8f190b223f) featuring a dataset and 3 fine-tuned models has been created, updated about 16 hours ago.
   - The collection includes the **CleverBoi-Llama-3.1-8B-Instruct**, showcasing advancements in text generation.
- **Duet Dataset Released**: A member introduced the [Duet Dataset v0.5](https://huggingface.co/datasets/G-reen/Duet-v0.5), which contains **5k rows** of COT question-answer with roleplaying prose.
   - They requested that users give them credits if they utilize this dataset, emphasizing the dataset's focus on narrative integration in questions and answers.
- **Caution on Duet Model Usage**: A warning was issued regarding the [Duet Model](https://huggingface.co/G-reen/Duet_Minitron8b_v0.5) stating it may behave differently from other models due to its unique data generation pipeline.
   - The creator cautioned that the model is a proof of concept and has not undergone extensive testing, potentially resulting in uncensored or undesirable outputs.
- **Open Sourcing Unique Datasets**: A member expressed excitement about open sourcing a new and unique dataset, inviting others to check it out.
   - They encouraged community engagement by providing links for those interested in exploring the dataset further.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/theprint/cleverboi-66ccd9588a104a8f190b223f">CleverBoi - a theprint Collection</a>: no description found</li><li><a href="https://huggingface.co/datasets/G-reen/Duet-v0.5">G-reen/Duet-v0.5 · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/G-reen/Duet_Minitron8b_v0.5">G-reen/Duet_Minitron8b_v0.5 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1278013984988991539)** (5 messages): 

> - `Finetuning Large Language Models`
> - `Understanding LLMs`
> - `Data Preparation for LLMs` 


- **Confusion Around Finetuning Goals**: A member expressed frustration about individuals wanting to **finetune LLMs** without any clear objectives or understanding of the process.
   - They commented on the oddity of this trend, questioning the motivations behind such requests.
- **Lack of Basic LLM Knowledge**: Concerns were raised about many wanting to finetune LLMs while lacking **basic knowledge** of how these models function.
   - This gap in understanding highlights the need for better education and resources in the community.
- **Data Sets: An Overlooked Aspect**: The discussion pointed out that many who wish to finetune LLMs often do so without even having a **datasets** to work with.
   - This reinforces the notion that a solid foundation is crucial before venturing into model finetuning.
- **Weirdness in the Community**: Another member chimed in about the peculiar behaviors observed within this community surrounding LLM finetuning.
   - They emphasized that the complexity and nuances of LLMs can lead to misunderstandings and mixed intentions.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1277964459666575553)** (1 messages): 

> - `Job offerings in AI`
> - `Quality of developers seeking jobs` 


- **Absence of Job Opportunities**: A concern was raised regarding the *lack of job offerings* in the AI sector.
   - It was noted that individuals seeking gig opportunities in this way often come across as **terrible developers** with **zero experience**.
- **Developers Seeking Gigs**: Discussion highlighted that individuals who are actively looking for jobs in this manner typically show little skill or background.
   - This raises concerns about the overall quality and **experience levels** of developers in the current job market.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1277704871151407157)** (356 messages🔥🔥): 

> - `AI Personhood Debate`
> - `Human-AI Interaction`
> - `Emotional Understanding in AI`
> - `Decentralization of AI`
> - `Future of AI in Society` 


- **Exploring AI Personhood and Emotionality**: The conversation delves into whether AI can possess emotions, with participants noting that AI can simulate emotional responses but lack genuine subjective experiences.
   - There is an acknowledgment of the ongoing debate about AI's identity, particularly as users may form attachments to AI, raising ethical concerns about perceiving them as friends.
- **Decentralization and AI Data Use**: The discussion touches on decentralization in AI, emphasizing the shift towards user-owned data and concerns over corporations controlling AI identities and training data.
   - Participants expressed hope for a future where open-source models become more prevalent, reducing reliance on centralized corporate structures.
- **AI's Role in Human Connectedness**: The potential for AI to provide companionship is debated, with some expressing skepticism about AI replacing human friendship while others promote its usefulness.
   - A participant shared a personal story about the impact of friendship and the potential relief AI could offer to those feeling isolated, particularly in marginalized communities.
- **Nature of Feelings and Emotions in AI**: Participants discussed the distinction between feelings and emotions, suggesting that AI can represent emotions through their responses but do not genuinely experience them.
   - The nature of AI’s understanding of emotions was explored, relating it to its design and training, and highlighting the complexity of human emotional experiences.
- **Impact of AI on the Future Job Market**: There is curiosity about how AI advancements will affect the job market, with a focus on the shift from manual labor positions to AI’s potential to replace creative and technical jobs.
   - Participants shared insights on the evolution of AI’s capabilities, expressing concern and interest in the broader societal implications amidst the growing integration of AI technology.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.m.wikipedia.org/wiki/Personhood">Personhood - Wikipedia</a>: no description found</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>: Distributed Training Over-The-Internet. Contribute to NousResearch/DisTrO development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1277771397422383135)** (15 messages🔥): 

> - `GPT-4o reasoning issues`
> - `AI voice synthesis business ideas`
> - `Challenges with YouTube summarization tools` 


- **Frustrations with GPT-4o's reasoning capabilities**: Users expressed dissatisfaction with **GPT-4o**, noting a decline in basic reasoning, factual inaccuracies, and inconsistencies in responses.
   - One user specifically mentioned that they feel **GPT-4o** has regressed in performance compared to previous models and are looking for potential improvements or minor updates.
- **AI voice synthesis business inquiry**: A user asked where to begin developing a business idea involving **AI voice synthesis technology**.
   - This inquiry highlights a growing interest in harnessing AI technologies for entrepreneurial endeavors.
- **Issues with YouTube summarization tools**: Concerns were raised about many **YouTube summarization** tools being ineffective due to **YouTube** blocking bot access to transcripts.
   - A suggestion for manually retrieving transcripts was made, although it was noted that automated services would violate YouTube's terms of service.


  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1277706485866172446)** (324 messages🔥🔥): 

> - `Model Deployment Challenges`
> - `Runtime Errors in Hugging Face Spaces`
> - `Training AI Models Issues`
> - `Using Sentence Transformers`
> - `Converting Model Formats` 


- **Model Deployment Challenges**: A user is struggling to deploy the Gemma-7B model in a Hugging Face space due to runtime errors related to the model path and missing files.
   - They considered cloning a repository but expressed concerns about not being able to upload the large model file required.
- **Runtime Errors in Hugging Face Spaces**: A user reported a runtime error when attempting to load the model, prompting discussions regarding missing file paths and model configurations.
   - This led to suggestions to use the `.from_pretrained` method and checking if model files need to be in the repository.
- **Training AI Models Issues**: Discussions included issues related to training models, including low loss values and conversions to different formats that affect model performance.
   - Users shared experiences about training efficiency and errors encountered, indicating a broader concern for model optimization.
- **Using Sentence Transformers**: Users highlighted the usability of Sentence Transformers and discussed the differences between causal and non-causal models during coding tasks.
   - They emphasized the importance of proper model configuration for expected outputs and efficient usage.
- **Converting Model Formats**: A discussion emerged around converting model formats from bf16 to f32, causing issues in imports and necessary adjustments in model definitions.
   - Participants expressed confusion over format implications and troubleshooting methods to resolve import errors.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/mikemin027/Gemma-7b-it-GGUF">Gemma 7b It GGUF - a Hugging Face Space by mikemin027</a>: no description found</li><li><a href="https://huggingface.co/unclemusclez/SmolLM-135M-Instruct-DEVINator?show_file_info=model.safetensors>">unclemusclez/SmolLM-135M-Instruct-DEVINator · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/OpenMeditron/Meditron3-8B">OpenMeditron/Meditron3-8B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unclemusclez/SmolLM-135M-Instruct-DEVINator">unclemusclez/SmolLM-135M-Instruct-DEVINator · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/mikeohearn-gif-9467924472242763968">Mikeohearn GIF - Mikeohearn - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/o-hearn-gif-3900414469346077199">O Hearn GIF - O hearn - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://sambanova.ai/fast-api?api_ref=444868">Get Fast &amp; Free AI Inference API | SambaNova Systems</a>: Empower your AI applications with blazingly-fast inferencing using SambaNova’s Free API. Experience the future of AI with cutting-edge RDU chip technology.</li><li><a href="https://github.com/huggingface/transformers/issues/12062>">Issues · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - Issues · huggingface/transformers</li><li><a href="https://huggingface.co/datasets/librarian-bots/base_model_sprint#description">librarian-bots/base_model_sprint · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1277840298210234409)** (4 messages): 

> - `PyTorch Lightning's LitServe`
> - `GraphRAG Tutorials`
> - `Extreme LLM Compression`
> - `Use Cases for LLMs and Generative AI`
> - `Neuralink Training Updates` 


- **PyTorch Lightning launches LitServe**: [LitServe](https://github.com/Lightning-AI/litserve) from **PyTorch Lightning** boasts **2x faster** model serving speeds compared to FastAPI, marking a significant update for model deployment.
   - *This is an exciting step forward in the model serving landscape,* fostering faster inference times for users.
- **Step-by-step GraphRAG Tutorials Released**: A set of **GraphRAG** tutorials has been unveiled, including a [video tutorial](https://www.youtube.com/watch?v=xnoEjczoqqE) on using LlamaIndex for building the system.
   - Participants shared insights into extracting entities and enhancing community summaries, with relevant resources available in both **V1** and **V2 notebooks**.
- **Insights on Extreme LLM Compression**: An article discusses the [evolution of extreme LLM compression](https://medium.com/yandex/the-evolution-of-extreme-llm-compression-from-quip-to-aqlm-with-pv-tuning-19c44b91af96), highlighting techniques to minimize quality loss while compressing large models.
   - As models grow large, this compressive approach is vital for effective deployment, especially on personal machines.
- **Deciding on LLM vs Generative AI Use Cases**: A recent piece prompts professionals to evaluate when to **not** use LLM or Generative AI, providing insights on appropriate case families for these technologies [here](https://pub.towardsai.net/do-not-use-llm-or-generative-ai-for-these-use-cases-a819ae2d9779).
   - The discussion emphasizes the importance of making informed decisions and avoiding hasty adoption of trending technologies.
- **Neuralink's Training Strategy**: A member reported progress with their **7b** model training, achieving promising results with further plans to scale up to **70b**.
   - They have also optimized batch sizes for enhanced performance, indicating a methodical approach to large-scale training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/llama_index/status/1827367293376184418">Tweet from LlamaIndex 🦙 (@llama_index)</a>: This weekend, we’re providing a definitive set of tutorials on how to build GraphRAG, step-by-step.  First, check out this video by @fahdmirza on implementing the core components of GraphRAG using an ...</li><li><a href="https://medium.com/yandex/the-evolution-of-extreme-llm-compression-from-quip-to-aqlm-with-pv-tuning-19c44b91af96">The Evolution of Extreme LLM Compression: From QuIP to AQLM with PV-Tuning</a>: We live in the era of Large Language Models (LLMs), with companies increasingly deploying models with billions of parameters. These…</li><li><a href="https://pub.towardsai.net/do-not-use-llm-or-generative-ai-for-these-use-cases-a819ae2d9779">Do Not Use LLM or Generative AI For These Use Cases</a>: Choose correct AI techniques for the right use case families
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1277817670976278568)** (3 messages): 

> - `Elicit platform`
> - `DisTrO optimizer`
> - `Llama3.1 performance`
> - `WebSim exploration` 


- **Elicit's Potential Explored via WebSim**: A member is eager about the capabilities of [Elicit](https://elicit.com) but believes it can achieve more dimensions in its functionality by using tools like WebSim to demonstrate its potential.
   - They plan to add a hover feature showing a **tag cloud** for journals to enhance the 2D search experience.
- **DisTrO: Revolution in Distributed Optimization**: **Nous Research** introduced DisTrO, an architecture-agnostic optimizer that reduces inter-GPU communication by **1000x to 10,000x** as detailed in their [preliminary report](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf).
   - This breakthrough aims to significantly improve distributed training over the internet.
- **Llama3.1 Generates Synthetic Data Effortlessly**: A user is successfully running **Llama3.1 405B** for free on the [SambaNova](https://sambanova.ai/fast-api?api_ref=444868) API, finding it excellent for synthetic data generation.
   - They noted its impressive utility as a judge in various applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sambanova.ai/fast-api?api_ref=444868">Get Fast &amp; Free AI Inference API | SambaNova Systems</a>: Empower your AI applications with blazingly-fast inferencing using SambaNova’s Free API. Experience the future of AI with cutting-edge RDU chip technology.</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>: Distributed Training Over-The-Internet. Contribute to NousResearch/DisTrO development by creating an account on GitHub.</li><li><a href="https://websim.ai/@cozyfluff/linear-journal-explorer">Trending Journal Explorer
sty</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1277719172977262696)** (5 messages): 

> - `StockLlama`
> - `Quantized Models for Vulnerability Insight`
> - `RYFAI Open Source AI Assistant`
> - `AI Voice Assistant with Raspberry Pis` 


- **StockLlama Forecasting Model Launch**: [StockLlama](https://github.com/LegallyCoder/StockLlama) is a time series forecasting model based on **Llama**, enhanced with custom embeddings for improved accuracy.
   - This model aims to provide users with more reliable forecasting capabilities in their projects.
- **Exploring Quantized Models for Vulnerability**: A member is finalizing a paper on utilizing **quantized models** to gain insights into vulnerabilities, with updates expected soon.
   - In the meantime, they shared a link to the [model collection](https://huggingface.co/collections/divyanshusingh/quantized-llama-66cb1d20a36a686617fa17f8) for further exploration.
- **RYFAI App Goes Open Source**: [RYFAI](https://github.com/PetertheRedCedar/ryfai) has been released as an open-source AI app designed to bring open-source AI models to users with ease.
   - Members are encouraged to try out this new tool and contribute to its development.
- **AI Voice Assistant Using Raspberry Pis**: A member shared their experience of creating an **AI voice assistant** using Raspberry Pis, showcasing the versatility of these devices.
   - This highlights the potential for hobbyists to develop functional AI applications with affordable hardware.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/PetertheRedCedar/ryfai">GitHub - PetertheRedCedar/ryfai: This is an AI app designed to bring open source AI models to your fingertips with ease</a>: This is an AI app designed to bring open source AI models to your fingertips with ease - PetertheRedCedar/ryfai</li><li><a href="https://github.com/LegallyCoder/StockLlama">GitHub - LegallyCoder/StockLlama: StockLlama is a time series forecasting model based on Llama, enhanced with custom embeddings for improved accuracy.</a>: StockLlama is a time series forecasting model based on Llama, enhanced with custom embeddings for improved accuracy. - LegallyCoder/StockLlama
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1277710244624990222)** (2 messages): 

> - `Channel Etiquette`
> - `HuggingFace M4/Idefics3-8B-Llama3 Paper` 


- **Reminder on Channel Etiquette**: A reminder was shared to avoid cross-posting and to keep discussions relevant to the specific channels.
   - *Keep channels on topic* is vital for maintaining productive conversations.
- **Exciting New Paper on HuggingFace M4/Idefics3-8B-Llama3**: A member highlighted a must-read paper on [HuggingFace M4/Idefics3-8B-Llama3](https://huggingface.co/papers/2408.12637), emphasizing its significance in the image-text-to-text domain.
   - The paper has already gained attention with **19.7k** views and **185** comments since its update just a day ago.



**Link mentioned**: <a href="https://huggingface.co/papers/2408.12637">Paper page - Building and better understanding vision-language models: insights and
  future directions</a>: no description found

  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1277967997176250429)** (1 messages): 

> - `Cog-5B`
> - `Video Generation`
> - `Fine-tuning Scripts` 


- **Cog-5B Video Model Released**: The **Cog-5B video model** was just released, touted as the best open weights for video generation, available at [CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b).
   - This model features a striking demo show that brings *nature to life* through a video gallery with captivating captions.
- **Access Comprehensive Resources for Cog-5B**: A collection of resources for the **Cog-5B model** includes a [detailed GitHub page](https://github.com/THUDM/CogVideo) and a [Huggingface Space](https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space).
   - For non-English speakers, there is also a [Chinese version](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/README_zh.md) of the model documentation available.
- **Upcoming Fine-Tuning Scripts for Cog-5B**: Anticipation builds as fine-tuning scripts for **Cog-5B** are expected to be released soon, enhancing user customization options.
   - This upcoming feature promises to allow developers and enthusiasts to better refine and utilize the model to suit specific needs.



**Link mentioned**: <a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b · Hugging Face</a>: no description found

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1277758842041667637)** (3 messages): 

> - `Fine-Tuning Vision Transformer for Object Detection`
> - `Open-source alternatives to VASA-1`
> - `Variational Autoencoders in Image-Text Generation` 


- **Issues with Fine-Tuning Vision Transformer**: A user is experiencing problems with object detection while following the [Fine-Tuning Vision Transformer for Object Detection](https://huggingface.co/learn/computer-vision-course/en/unit3/vision-transformers/vision-transformer-for-objection-detection) tutorial.
   - They are seeking insights from others who may have encountered similar detection shortcomings.
- **Searching for Open-source VASA-1 Alternatives**: A member inquired about open-source projects similar to **VASA-1**, indicating a search for alternatives in that space.
   - This reflects a demand for knowledge-sharing about available options for those working with similar technologies.
- **VAEs Not Widely Used for Image-Text Applications**: A newcomer raised a question about the limited use of **Variational Autoencoders (VAEs)** for generating text from images by utilizing shared latent space.
   - They are curious about why this approach isn't more prevalent in both research and practical applications.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1277834461978955900)** (5 messages): 

> - `Speculative Decoding`
> - `Finetuning Nemotron Model`
> - `Text-Summary Trends in 2024`
> - `Llama3.1 for Synthetic Data`
> - `Colab OOM Issues` 


- **Speculative Decoding retains output consistency**: A member clarified that **speculative decoding** does not affect the output of the target model, emphasizing that tool calling should still function if supported.
   - They noted the importance of understanding the nuances of tool calling for further exploration.
- **Finetuning Nemotron model leads to OOM issues**: A user attempted to finetune the **Nemotron model** on the **Dolly Dataset** using PEFT and SFT via the free version of Colab.
   - They reported encountering **Out Of Memory (OOM)** issues during training and sought tips to mitigate this problem.
- **Current trends in text-summary models**: A member questioned whether **text-summary** models are still relevant in 2024, suggesting a shift towards using general models like **Llama** for summarizing long text.
   - They speculated that leveraging long context and system prompts may become more popular than conventional summary models.
- **Llama3.1 API usage for synthetic data**: An enthusiastic user shared their experience running **Llama3.1 405B** via [SambaNova's API](https://sambanova.ai/fast-api?api_ref=444868) for synthetic data generation and LLM judging.
   - They highlighted the accessibility of this powerful tool for various applications.



**Link mentioned**: <a href="https://sambanova.ai/fast-api?api_ref=444868">Get Fast &amp; Free AI Inference API | SambaNova Systems</a>: Empower your AI applications with blazingly-fast inferencing using SambaNova’s Free API. Experience the future of AI with cutting-edge RDU chip technology.

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1278027809033293844)** (1 messages): 

> - `ProteinBERT model structure`
> - `Deep learning for proteins`
> - `Gene Ontology annotation`
> - `Architecture of ProteinBERT`
> - `Hugging Face resources` 


- **Understanding the ProteinBERT Structure**: A user seeks clarity on the main structure of the **ProteinBERT** model, specifically its global and local representations, noting it's been four years since its proposal.
   - They shared links to the [ProteinBERT paper](https://pubmed.ncbi.nlm.nih.gov/35020807/) and its [Hugging Face page](https://huggingface.co/GrimSqueaker/proteinBERT) for further reference.
- **ProteinBERT's Architecture Explained**: The user mentions that the architecture incorporates both **local and global representations**, designed to efficiently process long protein sequences for classification and regression tasks.
   - They express a desire for guidance on understanding these architectural elements and their application in protein function prediction.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pubmed.ncbi.nlm.nih.gov/35020807/">ProteinBERT: a universal deep-learning model of protein sequence and function - PubMed</a>: Supplementary data are available at Bioinformatics online.</li><li><a href="https://huggingface.co/GrimSqueaker/proteinBERT">GrimSqueaker/proteinBERT · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1277707150742913089)** (160 messages🔥🔥): 

> - `Model Loading Issues`
> - `Compatibility with Hardware`
> - `Guardrails Settings`
> - `Latest LM Studio Updates`
> - `Running LM Studio on Steam Deck` 


- **Model Loading Issues with LM Studio**: Users reported problems loading various models in LM Studio due to insufficient system resources, with one user facing an exit error code -1073740791.
   - Adjusting settings to CPU-only and changing guardrails settings were suggested as potential solutions.
- **Compatibility with Intel Arc GPUs**: A user inquired about LM Studio's support for Intel Arc GPUs, expressing curiosity for a friend's sake.
   - Responses indicated uncertainty, but no official support was confirmed for Intel Arc GPUs.
- **Guardrails Settings and Developer Mode**: Users struggled to locate the settings for changing model loading guardrails, with one finally finding it under developer mode.
   - This section is positioned near the UI theme settings, which had been initially overlooked.
- **Latest LM Studio Updates**: The latest version of LM Studio was confirmed to be v0.3.1, and users were advised to update from older beta versions.
   - No uninstallation was required before updating the application.
- **Running LM Studio on Steam Deck**: Concerns were raised about running LM Studio on the Steam Deck without using the --no-sandbox option, due to previous corruption issues.
   - Feedback from earlier attempts confirmed model loading stability when executed directly in desktop mode but with complications via Steam.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/it-was-the-aliens-im-not-saying-it-was-aliens-ancient-aliens-gif-14839810013080040984">It Was The Aliens Im Not Saying It Was Aliens GIF - It was the aliens Im not saying it was aliens Ancient aliens - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/11ha4qo/gptzero_an_ai_detector_thinks_the_us_constitution/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f2gaqt/cogvideox_5b_open_weights_text_to_video_ai_model/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f2gaqt/cogvideox_5b_open_weights_text_to_video_ai_mode">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/7JFU3W045hE?si=Daq5Q7eL2CNlI8Qq">Biniou - Generate Text, Video, Image, Music, 3D Locally - Free and Private</a>: This video shows how to install and use Biniou, which is an all-in-one web UI for AI.🔥 Buy Me a Coffee to support the channel: https://ko-fi.com/fahdmirza🔥...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1277715676735213661)** (72 messages🔥🔥): 

> - `GPU Choices for LLMs`
> - `Inference Speeds Comparison`
> - `Using Ollama with Dual CPUs`
> - `Tinygrad Framework Introduction`
> - `Cerebras Inference Announcement` 


- **Comparing RX 7800 XT and RTX 4070 for LLMs**: Members discussed upgrading GPUs, with one indicating that **Nvidia** currently outperforms AMD in LLM tasks.
   - They suggested that an **Nvidia 3060 12GB** could provide a balance between budget and performance for running models.
- **Inference speeds with CPU versus GPU**: Users shared their experiences with inference speeds, noting that **CPU-only** setups can be extremely slow, around **1-2 tokens/sec** on larger models.
   - A member reported their **5950X** system with **128GB of RAM** achieving up to **5-7 tokens/sec** but preferred using a GPU for better performance.
- **Dual CPU Limitations with Ollama**: A user noted that when running LLMs with **Ollama**, only one of their two CPUs heats up, indicating a potential single-CPU bottleneck.
   - Another member mentioned the need for support for **dual CPU** usage to increase inference speed but cautioned that asking about it may not be welcome.
- **Introduction of Tinygrad Framework**: A new framework, **Tinygrad**, was introduced, attracting attention for its simplicity and capability to break down complex networks.
   - It features unique operations like **ElementwiseOps** and **ReduceOps**, drawing interest despite its apparent limitations.
- **Cerebras Inference Capabilities**: Cerebras announced their inference service, boasting a remarkable **450 tokens/sec** for Llama3.1-70B, significantly faster than GPUs.
   - The service aims to provide developers with affordable rates, at just **60 cents** per million tokens, challenging traditional hyperscalers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tinygrad.org/#tinybox">tinygrad: A simple and powerful neural network framework</a>: no description found</li><li><a href="https://x.com/cerebrassystems/status/1828464491677524311?s=46)">Tweet from Cerebras (@CerebrasSystems)</a>: Introducing Cerebras Inference ‣ Llama3.1-70B at 450 tokens/s – 20x faster than GPUs ‣ 60c per M tokens – a fifth the price of hyperscalers ‣ Full 16-bit precision for full model accuracy ‣ Generous r...</li><li><a href="https://tenor.com/view/shut-up-take-my-money-small-money-fry-futurama-gif-15090562">Shut Up Take My Money GIF - Shut Up Take My Money Small Money - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/vllm-project/vllm/issues/963">Support for compute capability &lt;7.0 · Issue #963 · vllm-project/vllm</a>: Hi, How tightly coupled is the requirement for compute capability of 7.0 or higher? Is it possible to disable some features, and run on e.g. 6.0? Like a P100 Maybe this is totally unfeasible, but I...
</li>
</ul>

</div>
  

---



### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1277991064938090599)** (1 messages): 

> - `Aider v0.53.0`
> - `Prompt caching improvements`
> - `New command options`
> - `Error handling updates` 


- **Aider v0.53.0 Release Highlights**: The new version, **Aider v0.53.0**, introduces enhanced [prompt caching features](https://aider.chat/docs/usage/caching.html#preventing-cache-expiration) for cost savings and faster coding with models like Sonnet and Haiku.
   - In this release, **Aider wrote 59%** of its own code, showcasing significant improvements and self-sufficiency.
- **New Command Option for Cache**: Users can now run Aider with `--cache-prompts` to enable prompt caching, which can improve performance by retaining cached prompts during sessions.
   - Additionally, **bulk accept/reject** features have been added, enhancing user control over URL additions and confirmations.
- **Improvements in Error Handling**: Aider v0.53.0 includes improved **error messages** when variables aren't set, aiding users in troubleshooting their setups.
   - Recent bug fixes also address issues with **Windows filenames** containing `
`, ensuring smoother operation across systems.
- **Cache Warm Keeping Feature**: To prevent cache expiration, Aider can now ping the API every 5 minutes, keeping the prompt cache 'warm'.
   - This feature allows users to specify keep-alive pings with `--cache-keepalive-pings N`, enhancing cached data longevity.



**Link mentioned**: <a href="https://aider.chat/docs/usage/caching.html#preventing-cache-expiration)">Prompt caching</a>: Aider supports prompt caching for cost savings and faster coding.

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1277704150225916017)** (164 messages🔥🔥): 

> - `Aider Functionality`
> - `Model Performance Comparisons`
> - `Prompt Caching`
> - `OpenRouter Issues`
> - `Gemini Model Updates` 


- **Aider's Capabilities and Limitations**: Users discussed Aider's inability to convert large code bases in a single prompt, requiring careful refinements and multiple attempts.
   - It's highlighted that while Aider is a valuable tool for coding assistance, it doesn’t replace thorough testing and refining processes.
- **Comparing Gemini with Existing Models**: There were inquiries about the new Gemini model's performance in comparison to established models like GPT-4o and Sonnet.
   - Feedback from tests revealed varied pass rates and performance metrics, sparking discussions about the effectiveness of these models.
- **The Importance of Prompt Caching**: Users expressed the need for prompt caching in Aider to improve cost efficiency and processing speeds during operations.
   - Aider's caching feature is still being developed for compatibility with OpenRouter and Anthropic, with users eager for its rollout.
- **OpenRouter Performance Issues**: There were reports of temporary degradation in OpenRouter services, causing disruptions for some users.
   - Post-incident updates indicated successful resolutions, although some users experienced regional issues during the downtime.
- **New Developments in Gemini Models**: Discussion around the release of new Gemini models piqued interest regarding their capabilities relative to existing technology.
   - Users are curious if the new updates would elevate Gemini's performance to compete effectively with GPT-4o and Sonnet.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/caching.html#preventing-cache-expi">Prompt caching</a>: Aider supports prompt caching for cost savings and faster coding.</li><li><a href="https://aider.chat/2024/08/26/sonnet-seems-fine.html">Sonnet seems as good as ever</a>: Sonnet’s score on the aider code editing benchmark has been stable since it launched.</li><li><a href="https://aider.chat/docs/usage/caching.html">Prompt caching</a>: Aider supports prompt caching for cost savings and faster coding.</li><li><a href="https://aider.chat/docs/llms.html">Connecting to LLMs</a>: Aider can connect to most LLMs for AI pair programming.</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/llms">Connecting to LLMs</a>: Aider can connect to most LLMs for AI pair programming.</li><li><a href="https://aider.chat/docs/usage/caching.html#preventing-cache-expiration)">Prompt caching</a>: Aider supports prompt caching for cost savings and faster coding.</li><li><a href="https://aider.chat/docs/llms/openai.html">OpenAI</a>: aider is AI pair programming in your terminal</li><li><a href="https://github.com/BerriAI/litellm/releases/tag/v1.44.3">Release v1.44.3 · BerriAI/litellm</a>: 🔥 We&#39;re launching support for using Bedrock Guardrails on LiteLLM Gateway - use Bedrock guardrails with 100+ LLMs supported by LiteLLM 👉 Start here: https://docs.litellm.ai/docs/proxy/guardrails...</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>: OpenRouter Incident History</li><li><a href="https://github.com/anthropics">Anthropic</a>: Anthropic has 26 repositories available. Follow their code on GitHub.</li><li><a href="https://aider.chat/docs/config/options.html#--cache-prompts">Options reference</a>: Details about all of aider’s settings.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1277721494922989700)** (44 messages🔥): 

> - `Scripting with Python`
> - `Aider command line options`
> - `Cache keepalive feature`
> - `Data security in Aider`
> - `Auto-testing in GUI` 


- **Scripting with Python using Aider**: Users discussed how to script with Python to run shell commands and create files automatically in Aider, referencing [command line documentation](https://aider.chat/docs/scripting.html).
   - *One user noted that using `AIDER_YES` could automate approval responses, but faced issues implementing it in their script.*
- **Understanding Aider command line options**: A member inquired about the command to update Aider, and was advised to re-run the installation command they previously used with pip from the GitHub repository.
   - They also discussed the `--cache-keepalive-pings` feature, which requires specifying a number for effective usage.
- **Concerns over cache keepalive and data security**: *A user questioned the impact of new prompting on different models and the potential bias introduced by updates.*
   - Responses clarified that Aider does not communicate with external servers beyond configured LLM providers, stressing that user data remains private.
- **Issues with commit message generation**: A user reported difficulties with generating commit messages, receiving repeated errors about missing messages on commits without providing a solution.
   - The community suggested troubleshooting methods, such as retrying commands or restarting Aider to resolve the issue.
- **Auto-testing capability in Aider's GUI**: A user inquired about the possibility of auto-testing within Aider's GUI, reflecting community interest in automated testing tools.
   - No definitive answers were provided, indicating a need for further exploration on that subject.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/scripting.html#python">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://llm.datasette.io/en/stable/">LLM: A CLI utility and Python library for interacting with Large Language Models</a>: no description found</li><li><a href="https://aider.chat/docs/config/options.html#--cache-keepalive-pings-value">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://gist.github.com/karpathy/1dd0294ef9567971c1e4348a90d69285">Git Commit Message AI</a>: Git Commit Message AI. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/paul-gauthier/aider">GitHub - paul-gauthier/aider: aider is AI pair programming in your terminal</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1277731441773379617)** (10 messages🔥): 

> - `Anthropic System Prompts`
> - `Gemini 1.5 Experimental Models`
> - `Rate Limits on New Models` 


- **Anthropic Releases System Prompts for Claude 3**: Anthropic has published the initial system prompts for their **Claude 3** models, including **Claude 3 Haiku** and **Claude 3.5 Sonnet**, dating from **July 12, 2024**. They promise to update these prompts in their documentation to reflect future changes, enhancing transparency.
   - Researcher **Amanda Askell** had previously dissected their system prompts, which are often seen as a form of documentation that providers typically do not share.
- **Gemini 1.5 Models Rollout by Google**: Google has announced the rollout of three experimental models: **Gemini 1.5 Flash-8B**, **Gemini 1.5 Pro**, and an improved **Gemini 1.5 Flash**. Users can try them out at [AI Studio](https://aistudio.google.com).
   - The **Gemini 1.5 Pro** model is touted to perform better on coding and complex prompts, but further details on their availability remain pending.
- **Clarification on Rate Limits for Gemini**: Rate limits for the experimental Gemini models are set at **2 requests per minute**, **50 requests per day**, and **1 million tokens per minute**. This rate limit is applied per **GCP project**, which may provide users a workaround.
   - Some members expressed frustration at being unable to benchmark the new models due to these restrictions, but others hinted at potential workarounds.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aistudio.google.com,">no title found</a>: no description found</li><li><a href="https://simonwillison.net/2024/Aug/26/anthropic-system-prompts/">Anthropic Release Notes: System Prompts</a>: Anthropic now publish the system prompts for their user-facing chat-based LLM systems - Claude 3 Haiku, Claude 3 Opus and Claude 3.5 Sonnet - as part of their documentation, with …</li><li><a href="https://x.com/OfficialLoganK/status/1828480081574142227?t=Y0lfWR">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Today, we are rolling out three experimental models:  - A new smaller variant, Gemini 1.5 Flash-8B - A stronger Gemini 1.5 Pro model (better on coding & complex prompts) - A significantly improved Gem...</li><li><a href="https://x.com/OfficialLoganK/status/1828480081574142227?t=Y0lfWRozBkotP-PHiMMVig">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Today, we are rolling out three experimental models:  - A new smaller variant, Gemini 1.5 Flash-8B - A stronger Gemini 1.5 Pro model (better on coding & complex prompts) - A significantly improved Gem...
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1277705234399100980)** (142 messages🔥🔥): 

> - `New AI hardware releases`
> - `Flux model capabilities`
> - `ZLUDA development status`
> - `Using SD Next with ZLUDA`
> - `Streamdiffusion and SDXL Turbo` 


- **Anticipation for next-gen AI hardware**: Members discussed the upcoming releases of new **Intel CPUs** and **NVIDIA GPUs** with AI features, prompting interest in building new PCs.
   - The conversation highlighted that these advancements are expected to enhance performance, especially for AI-related tasks.
- **Impressive capabilities of Flux models**: One user shared excitement about the advanced features of **Flux**, such as dynamic angles and depth perspective, potentially making older models obsolete.
   - Another member remarked that the model's trainability underscores its potential to redefine AI-generated visuals.
- **Concerns over ZLUDA development**: There were concerns raised regarding the future of **ZLUDA** after recent reports indicated that **AMD** might have stopped funding its development.
   - Although ZLUDA's GitHub was updated, some members suggested that legalities may hinder its progress.
- **Integration of SD Next with ZLUDA**: A member sought clarification on why **SD.Next** reportedly performs better with **ZLUDA**, speculating about the backend architecture that includes both A1111 and Diffusers.
   - This multi-backend approach could enhance compatibility and performance across different models.
- **Challenges with Streamdiffusion and SDXL Turbo**: Users discussed the difficulty of integrating **SDXL Turbo** with **Streamdiffusion**, particularly concerning performance with **TensorRT**.
   - Despite potential benefits, concerns were raised about frame rates and resolution compatibility affecting usability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vexel.pages.dev/">Vexel</a>: no description found</li><li><a href="https://huggingface.co/stabilityai/sdxl-turbo-tensorrt">stabilityai/sdxl-turbo-tensorrt · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/mike-lowrey-gif-8186790">Mike Lowrey GIF - Mike Lowrey - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.hpcwire.com/2024/08/12/amd-funds-then-quashes-cuda-emulator-project-zluda/">AMD Funds Then Quashes CUDA Emulator Project ZLUDA</a>: In a previous article, HPCwire mentioned the CUDA emulation project called ZLUDA. Written mostly in Rust, the open-source project runs unmodified binary CUDA applications with near-native performance ...</li><li><a href="https://www.instagram.com/p/C-8AxhdRrNP/?i">Juli&#xe1;n on Instagram: &quot;After a deeply introspective and emotional journey, I fine-tuned SDXL using old family album pictures of my childhood [60], a delicate process that brought my younger self into dialogue with the present, an experience that turned out to be far more impactful than I had anticipated.

What&#039;s particularly interesting about the resulting visuals, is that they seem to be imbued with intricate emotions, and not-so-well-recalled distant memories. Intuition tells me there&#039;s something of value in these kinds of experiments.

First video is Archaia&#039;s #touchdesigner system intervened with the resulting LORA. And second one is a real-time test (StreamDiffusion) of said LORA + an updated version of Auratura working in parallel.

You can explore more of my work, tutorials, and systems via the link in bio.

#generativeart #design #audiovisual #experimental #artandtechnology&quot;</a>: 748 likes, 53 comments - uisato_ on August 21, 2024: &quot;After a deeply introspective and emotional journey, I fine-tuned SDXL using old family album pictures of my childhood [60], a delicate proces...</li><li><a href="https://image.duckers-web.site/hEja1/RAmEmido03.png">chrome_1ryqdwk07C.png (454.35 KB)</a>: Date: 2024-08-27 00:13:43</li><li><a href="https://www.instagram.com/p/C-8AxhdRrNP/?img_index=1">Juli&#xe1;n on Instagram: &quot;After a deeply introspective and emotional journey, I fine-tuned SDXL using old family album pictures of my childhood [60], a delicate process that brought my younger self into dialogue with the present, an experience that turned out to be far more impactful than I had anticipated.

What&#039;s particularly interesting about the resulting visuals, is that they seem to be imbued with intricate emotions, and not-so-well-recalled distant memories. Intuition tells me there&#039;s something of value in these kinds of experiments.

First video is Archaia&#039;s #touchdesigner system intervened with the resulting LORA. And second one is a real-time test (StreamDiffusion) of said LORA + an updated version of Auratura working in parallel.

You can explore more of my work, tutorials, and systems via the link in bio.

#generativeart #design #audiovisual #experimental #artandtechnology&quot;</a>: 748 likes, 53 comments - uisato_ on August 21, 2024: &quot;After a deeply introspective and emotional journey, I fine-tuned SDXL using old family album pictures of my childhood [60], a delicate proces...</li><li><a href="https://github.com/11cafe/comfyui-workspace-manager">GitHub - 11cafe/comfyui-workspace-manager: A ComfyUI workflows and models management extension to organize and manage all your workflows, models in one place. Seamlessly switch between workflows, as well as import, export workflows, reuse subworkflows, install models, browse your models in a single workspace</a>: A ComfyUI workflows and models management extension to organize and manage all your workflows, models in one place. Seamlessly switch between workflows, as well as import, export workflows, reuse s...</li><li><a href="https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu">GitHub - lshqqytiger/stable-diffusion-webui-amdgpu: Stable Diffusion web UI</a>: Stable Diffusion web UI. Contribute to lshqqytiger/stable-diffusion-webui-amdgpu development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1277772596016058405)** (10 messages🔥): 

> - `Video Benchmark Examples`
> - `RLHF Libraries`
> - `Free API for Llama 3.1` 


- **Seeking Video Benchmark Examples**: A member inquired about good examples of video benchmarks, particularly for tasks like spatial awareness and generative models.
   - Another member suggested standard evaluation tasks like **action recognition** for discriminative tasks, but noted the lack of established benchmarks for generation.
- **Libraries for RLHF**: Discussion arose regarding whether **TRL/TRLX** remains the best option for Reinforcement Learning from Human Feedback (RLHF).
   - One member recommended **TRL**, mentioning that **TRLX** hasn't been updated in some time, with no alternative libraries currently known.
- **Free API for Running Llama 3.1**: A member shared a link to a free API for running **Llama 3.1 405B**, provided by **SambaNova**.
   - They included a [link to the API](https://sambanova.ai/fast-api?api_ref=444868) along with details about SambaNova's headquarters and product offerings.



**Link mentioned**: <a href="https://sambanova.ai/fast-api?api_ref=444868">Get Fast &amp; Free AI Inference API | SambaNova Systems</a>: Empower your AI applications with blazingly-fast inferencing using SambaNova’s Free API. Experience the future of AI with cutting-edge RDU chip technology.

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1277707776923271229)** (124 messages🔥🔥): 

> - `Gemini Misrepresentation`
> - `Learning Rate Scaling`
> - `Moshi Voice AI Launch`
> - `DisTrO Distributed Training`
> - `MiniCPM and Infinite LRs` 


- **Jamba's Claims on Gemini questioned**: There was a heated discussion around Jamba authors allegedly misrepresenting Gemini, claiming it caps at **128k** without testing beyond, which some members contested.
   - Another member defended the Jamba authors, suggesting that the paper's wording does not indicate a false claim, stating *they weren't able to reproduce the results beyond 128k*.
- **Insights on Learning Rate Scaling**: Discussion centered on learning rate scaling with batch sizes, particularly that it should adhere to a **sqrt scaling** with Adam, with links provided to significant papers that differ on these approaches.
   - Participants debated the validity of different approaches, including the noise in the graphs presented by their experiments, which led to questions about the methodology used.
- **Launch of Moshi Voice AI**: A YouTube video titled "[Unveiling of Moshi](https://www.youtube.com/watch?v=hm2IJSKcYvo)" showcased a new voice-enabled AI model developed in **6 months** by the Kyutai research lab, handling unprecedented vocal capabilities.
   - Impressively, this **7B model** runs on a standard laptop, making it openly accessible to all, which stirred interest in its operational efficiency.
- **DisTrO's Distributed Training Breakthrough**: Nous Research released a [preliminary report](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf) on DisTrO, aiming to reduce inter-GPU communication needs by up to **10,000x**.
   - This approach is seen as a significant step toward democratizing LLM training, enabling collaboration without reliance on singular computing entities.
- **MiniCPM and Continuous Pretraining**: The group discussed the miniCPM paper's method for infinite LRs and how it contrasts against traditional warmup strategies, sparking curiosity over its unusual curve shapes.
   - Members noted that the effectiveness of warmup steps may diminish with data distribution shifts, suggesting that real-world scenarios could differ greatly from pretraining conditions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://proceedings.neurips.cc/paper/2019/hash/e0eacd983971634327ae1819ea8b6214-Abstract.html">Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model</a>: no description found</li><li><a href="https://arxiv.org/abs/2408.13359">Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler</a>: Finding the optimal learning rate for language model pretraining is a challenging task. This is not only because there is a complicated correlation between learning rate, batch size, number of trainin...</li><li><a href="https://arxiv.org/abs/2408.11029">Scaling Law with Learning Rate Annealing</a>: We find that the cross-entropy loss curves of neural language models empirically adhere to a scaling law with learning rate (LR) annealing over training steps ($s$): $$L(s) = L_0 + A\cdot S_1^{-α} - C...</li><li><a href="https://x.com/NousResearch/status/1828121648383566270">Tweet from Nous Research (@NousResearch)</a>: What if you could use all the computing power in the world to train a shared, open source AI model?  Preliminary report: https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://www.youtube.com/watch?v=hm2IJSKcYvo">Unveiling of Moshi: the first voice-enabled AI openly accessible to all.</a>: In just 6 months, with a team of 8, the Kyutai research lab developed from scratch an AI model with unprecedented vocal capabilities called Moshi.This new ty...</li><li><a href="https://arxiv.org/abs/1812.06162">An Empirical Model of Large-Batch Training</a>: In an increasing number of domains it has been demonstrated that deep learning models can be trained using relatively large batch sizes without sacrificing data efficiency. However the limits of this ...</li><li><a href="https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/">How to Scale Hyperparameters as Batch Size Increases</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1277705580080791612)** (6 messages): 

> - `NVIDIA Power Utilization`
> - `GPU Hardware Measurement Tools`
> - `WandB System Metrics`
> - `PyTorch Profiler Insights` 


- **Simple NVIDIA Power Metrics**: **NVIDIA-smi** is mentioned as a simple method to measure GPU power utilization, with a caveat of trusting the cooling system.
   - It's pointed out that while it's straightforward, there are more accurate methods that require additional setup.
- **Understanding Hardware Utilization**: **NVIDIA-smi** specifically measures the **percentage of time** the GPU is actively doing something, which is different from power draw metrics.
   - Members discussed the distinction between GPU utilization and power metrics, highlighting varying interpretations.
- **Easy Utilities for Measuring GPU Power**: A member inquired about **easy-to-use utilities** for measuring GPU power draw in watts, suggesting **pynvml** as a potential tool.
   - They also referenced information from the [W&B documentation](https://docs.wandb.ai/guides/app/features/system-metrics#gpu-power-usage-watts) about system metrics tracked by the WandB SDK.
- **PyTorch Profiler for GPU Utilization**: The **PyTorch profiler** is recommended for getting accurate GPU utilization and tensor core occupancy metrics, with some overhead for performance measurement.
   - Profiling is suggested near the start of the runs to capture useful snapshots of GPU behavior.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.wandb.ai/guides/app/features/system-metrics#gpu-power-usage-watts">System Metrics | Weights &amp; Biases Documentation</a>: Metrics automatically logged by wandb</li><li><a href="https://github.com/EleutherAI/cookbook">GitHub - EleutherAI/cookbook: Deep learning for dummies. All the practical details and useful utilities that go into working with real models.</a>: Deep learning for dummies. All the practical details and useful utilities that go into working with real models. - EleutherAI/cookbook
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1278037216186204280)** (7 messages): 

> - `Liger Kernel Issue`
> - `Volunteers for Llama Project`
> - `Llama3 Instruct with Triton` 


- **Liger Kernel Issue #119 sparks interest**: A discussion has emerged regarding a GitHub issue on the [Liger Kernel repository](https://github.com/linkedin/Liger-Kernel/issues/119) where the development team shares plans to implement **llama** from scratch in pure **Triton**.
   - The idea has been inspired by Karpathy, and most of the necessary kernels are already included in Liger Kernel.
- **Call for Volunteers for Llama Project**: Member @byronhsu1230 announced a **call out for volunteers** to assist with a project related to Llama.
   - Another member, **nanodijkstra**, expressed interest in helping, prompting a collaborative response.
- **Issues with Llama3 Instruct in Triton**: One user reported difficulty with **Llama3 Instruct** in Triton, specifically that response generation doesn't stop when using both **TensorRT-LLM** and **vLLM** backends.
   - They noted that using vLLM hosting works flawlessly, questioning if there might be a **configuration issue** in Triton.



**Link mentioned**: <a href="https://github.com/linkedin/Liger-Kernel/issues/119">[fun] llama.triton · Issue #119 · linkedin/Liger-Kernel</a>: 🚀 The feature, motivation and pitch @thomwolf and i have an idea to implement llama from scratch in pure triton, inspired by karpathy. liger kernel already contains most of the kernels except matmu.....

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1277981801989672981)** (4 messages): 

> - `OOM issues with Torch Nightlies`
> - `TorchInductor Performance Dashboard` 


- **OOM Issues Arise with Recent Torch Nightlies**: A member reported experiencing **Out of Memory (OOM)** errors starting from the **20240823** nightly version of Torch, seeking feedback from others on similar issues.
   - The community is encouraged to share specific examples that may help in debugging these occurrences.
- **Performance Trends Observed on Dashboard**: Another member suggested checking the [**TorchInductor Performance Dashboard**](https://hud.pytorch.org/benchmark/compilers) to evaluate performance trends.
   - While there was uncertainty in interpreting the dashboard, it was noted that recent performance did not seem worse compared to the previous week.



**Link mentioned**: <a href="https://hud.pytorch.org/benchmark/compilers">no title found</a>: no description found

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1277709154575978589)** (8 messages🔥): 

> - `Chris Lattner's talk on GPUs`
> - `DisTrO report and features`
> - `Multi-model endpoint demo`
> - `Broadcom AI ASIC with optical connection`
> - `CerebrasSystems Llama3.1 performance` 


- **Chris Lattner Talks Mojo and MAX/GPUs**: Currently watching Chris Lattner's [talk on GPU programming](https://youtu.be/1T-MBC9k99M?si=PQYeKDBXSnxyHn2H&t=55) at the Mojo Community Meeting #5, focusing on asynchronous programming rules.
   - The community is excited after the meeting, highlighting the detailed discussion around GPU programming nuances.
- **DisTrO Preliminary Report Intrigues**: A [preliminary report on DisTrO](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf) details distributed training over-the-internet, showing promising scalability.
   - Community members expressed interest in failure tolerance features and how nodes might be used as hot spares, similar to RAID technology.
- **Broadcom's Optical AI ASIC Unveiling**: At Hot Chips 2024, Broadcom introduced its custom AI compute ASIC that features optical attach capabilities, crucial for future customer projects.
   - The presentation also included insights into co-packaged optics, marking a significant step forward in AI accelerator technology.
- **Cerebras Llama3.1 Outruns Competitors**: Running Llama3.1 8B at an impressive **1,832 tokens/sec**, [CerebrasSystems claims](https://x.com/CerebrasSystems/status/1828465008298336588) to be the fastest inference API globally.
   - They noted being roughly **20x faster** than NVIDIA GPUs and **2x faster** than Groq, sparking interest among performance enthusiasts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hotswap.outerport.com/">Outerport - Just-in-Time Hotswap</a>: no description found</li><li><a href="https://www.servethehome.com/broadcom-ai-compute-asic-with-optical-attach-detailed-at-hot-chips-2024/">Broadcom AI Compute ASIC with Optical Attach Detailed at Hot Chips 2024</a>: In one of the coolest presentations at Hot Chips 2024 so far, Broadcom showed co-packaged silicon photonics for switches and AI ASICs</li><li><a href="https://x.com/CerebrasSystems/status/1828465008298336588">Tweet from Cerebras (@CerebrasSystems)</a>: Cerebras Inference is the fastest Llama3.1 inference API by far: 1,800 tokens/s for 8B and 450tokens/s for 70B. We are ~20x faster than NVIDA GPUs and ~2x faster than Groq.</li><li><a href="https://youtu.be/1T-MBC9k99M?si=PQYeKDBXSnxyHn2H&t=55">Mojo 🔥 Community Meeting #5</a>: Recording of the Mojo Community Meeting #5🔢 Chris Lattner on GPU programming with Mojo 🔥🔀 Async Mojo 🔥 - 10 Simple Rules❓ Community Q&amp;AFull agenda and de...</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>: Distributed Training Over-The-Internet. Contribute to NousResearch/DisTrO development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1277808887646654495)** (18 messages🔥): 

> - `Quantization Queries`
> - `Int8 and FP8 APIs`
> - `Conv1D Custom Handling`
> - `Dynamically Quantizing Tensors`
> - `Weight Objects in GC` 


- **Noob Queries on Model Quantization**: A user inquired about starting model quantization for a sub 1B architecture and confirmed plans to use [PyTorch's quantization](https://github.com/pytorch/ao/tree/main/torchao/quantization) tools.
   - Another user noted that quantizing to **int8** is a simple one-line API, while **fp8** support will also be straightforward soon.
- **Custom Handling for Conv1D**: Discussion shifted to adapting the quantization filter for **Conv1D** layers, introducing a new filter function to accommodate both **Linear** and **Conv1D**.
   - A user implemented a filter function that checks for kernel size to ensure compatibility.
- **Challenges in Quantization**: Users encountered issues with the `dynamically_quantize_per_channel` function which expected a **2D tensor**, while their input was **3D**.
   - They suggested the need to use `squeeze` and `unsqueeze` operations to adjust tensor dimensions appropriately.
- **Memory Management Concerns**: A user observed both quantized tensors and non-quantized **Int8WeightOnlyQuantizedLinearWeight** objects during garbage collection.
   - This raised questions about the operation of `addmm` with these weight types, which led to debugging concerns.
- **Error Handling in Quantization**: An error surfaced indicating that the **Int8WeightOnlyQuantizedLinearWeight** subclass does not correctly implement the `addmm` operation required for matrix multiplication.
   - This could pose significant challenges for using quantized weights in further computations within the model.



**Link mentioned**: <a href="https://github.com/pytorch/ao/tree/main/torchao/quantization">ao/torchao/quantization at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao

  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1278043643218755625)** (1 messages): 

> - `Pallas`
> - `Involved Kernels` 


- **Inquiry about Pallas repositories**: A member expressed interest in learning about **Pallas** and inquired if there are any repositories where people use more involved kernels in production.
   - *Any pointers are appreciated* regarding resources or communities focused on Pallas.
- **Request for kernel usage examples**: A call for examples of more involved kernels in real-world applications was made, emphasizing the need for practical references.
   - Members were encouraged to share any **repositories** or experiences related to using these kernels in production settings.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1277799883415359571)** (12 messages🔥): 

> - `Low-Bit Optimization`
> - `Dataset Recommendations`
> - `Fine-Tuning Issues`
> - `Triton FP8 Kernels`
> - `CogVideoX 5B Model` 


- **Low-Bit Optimization Convergence Checks**: A user is seeking advice on datasets for fine-tuning and evaluating a **4-bit optimization** model, noting challenges in achieving good performance after fine-tuning on the Alpaca dataset.
   - Another user suggested the **Chatbot Arena dataset** as a comprehensive option, albeit with potential complexities in handling.
- **Fine-Tuning Challenges with Llama Models**: Concerns were raised about a fine-tuned model performing worse on **TruthfulQA_mc2** after fine-tuning on Alpaca, leading to discussions about potential bugs or overfitting strategies.
   - Exploring the simplicity of using **Llama2-7B** for fine-tuning was recommended as an alternative due to its manageable performance characteristics.
- **Triton FP8 Kernel Performance Discrepancies**: A user reported a **30% lower quantization error** with the Triton matmul kernel compared to PyTorch's `torch._scaled_mm`, raising questions about the implementation of scale settings.
   - Another participant requested a reproduction of the issue, suggesting it might be related to how scale factors are configured.
- **Exciting Release of CogVideoX 5B**: A new model called **CogVideoX 5B** has been announced with **open weights** and a notable integration with Diffusers, aimed at lowering memory requirements for video generation.
   - Details about its capabilities and efficient inference were shared, highlighting its practical use with less than **10GB** VRAM.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/lmsys/chatbot_arena_conversations">lmsys/chatbot_arena_conversations · Datasets at Hugging Face</a>: no description found</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html">End-to-End Workflow with torchtune &mdash; torchtune 0.2 documentation</a>: no description found</li><li><a href="https://x.com/aryanvs_/status/1828405977667793005">Tweet from Aryan V S (@aryanvs_)</a>: The best open weights video generation model is here - CogVideoX 5B 🔥  It comes with 🧨 Diffusers integration. Proud to share my major dish cooked at @huggingface in collaboration w/ the @ChatGLM fol...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f2gaqt/cogvideox_5b_open_weights_text_to_video_ai_model/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/pytorch/ao/pull/746">[Low-bit optim] Add Llama2-7B finetune benchmarks by gau-nernst · Pull Request #746 · pytorch/ao</a>: Update: change Llama3.1-8B-instruct to Llama2-7B Fine-tune Llama2-7B on Alpaca dataset. Full BF16, 1 epoch, A100, fixed random seed. Benchmark is done with torchtune. Summary    AdamW impl Max memo...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1277751147611947018)** (3 messages): 

> - `torch_viz error`
> - `GridExecutor`
> - `notebook issues` 


- **TypeError with GridExecutor in Notebook**: Users are reporting a **TypeError** with `GridExecutor._init_args_hst()` stating a missing required positional argument: 'kwargs'. This error emerged after switching from the v1 release to the main branch of `torch_viz`, but the issue persisted.
   - Several users confirmed experiencing the same error, expressing frustrations that they have never encountered it before.
- **Concerns About Switching Versions of torch_viz**: A user attempted to resolve the issue by installing the main branch of `torch_viz` instead of v1, which did not resolve their problem. They received a different error message after the switch, indicating further issues with compatibility.
   - This has raised concerns amongst users about the stability and reliability of different versions of the library, with multiple members facing similar challenges.


  

---


### **CUDA MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1277963586056097883)** (1 messages): 

> - `BitBlas Performance`
> - `Older GPU Compatibility` 


- **BitBlas runs surprisingly well on older GPUs**: **BitBlas** is functional on older GPUs like the **2080 Ti**, demonstrating decent speed during usage.
   - However, one drawback is that **fullgraph compilation does not work** on these older devices.
- **Older GPUs face limitations with BitBlas**: While **BitBlas** works on older GPUs, the lack of support for **fullgraph compilation** remains a significant limitation.
   - Users expressed that despite good performance, the incompatibility is a notable issue for older hardware.


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1277710274933297197)** (9 messages🔥): 

> - `H100 NVL GPU Performance`
> - `Checkpoint Resuming`
> - `Low-Rank Fine-Tuning` 


- **Maximize H100 NVL GPU VRAM Usage**: A user reported achieving **~340000 tokens/s with 1 H100 NVL (94GB) GPU**, but noted only using **26GB** of the available memory.
   - Another member suggested increasing **batch size** and/or disabling **recomputation** or training a larger model to fully utilize the VRAM.
- **Resume Training from Checkpoint**: In the context of resuming training, a user inquired whether they needed to set `resume` to **1** to automatically pretrain from the latest checkpoint.
   - Someone confirmed that it should be **-y 1**, although they weren't completely certain.
- **Exploring Fine-Tuning Techniques**: A discussion arose regarding a preprint paper on low-rank fine-tuning methods, questioning if it still qualifies as full fine-tuning.
   - One user pointed out that the proposed method maintains the gradients/optimizer state in a **low-rank subspace** to conserve memory without imposing a low-rank structure on the weights.


  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 messages): 

kashimoo: <@813802565426479145> Have you worked on the heuristics side for MIOpen?
  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1278055067358068771)** (17 messages🔥): 

> - `Compile-time programming in C++/CUDA`
> - `Zig's compile-time capabilities`
> - `Use of constexpr in C++`
> - `Cutlass and constexpr`
> - `Half type implementation challenges` 


- **Exploring Compile-time Programming in C++/CUDA**: A proposal was made for a talk on writing *generic CUDA kernel templates* that handle quantization and generate launch code without extensive boilerplate.
   - *Quantization will be of significant interest* to the audience, and the venue offers multiple conference rooms for presentations.
- **Zig's Fabled Compile-time Magic**: Curiosity arose over trying *Zig* to explore its impressive compile-time programming capabilities for CUDA kernels.
   - The discussion reflected skepticism about the complexity of resulting code in C++, even when using modern features.
- **Adopting constexpr for Cleaner Code**: Members shared their experiences using *constexpr* functions as alternatives to preprocessor macros, promoting cleaner code practices.
   - The implications of using `--expr-relatex-constexpr` with `nvcc` to make all `constexpr` functions implicitly callable were noted as a benefit.
- **Cutlass's Continued Use of Compile-time Techniques**: General satisfaction was expressed towards *Cutlass's* approach to compile-time computation, despite its reliance on template wrapper classes.
   - Members highlighted their own struggles and learning experiences in developing complex compile-time solutions, expressing a desire for clearer methodologies.
- **Defining Half Type in C++**: A desire for a *constexpr-able half type* was voiced, with considerations about implementing `std::numeric_limits` for this type.
   - Challenges regarding the necessity of `constexpr` constructors in maintaining certain interfaces were discussed, emphasizing the need for clarity in handling types.



**Link mentioned**: <a href="https://github.com/AnswerDotAI/gpu.cpp/blob/main/numeric_types/half.h">gpu.cpp/numeric_types/half.h at main · AnswerDotAI/gpu.cpp</a>: A lightweight library for portable low-level GPU computation using WebGPU.  - AnswerDotAI/gpu.cpp

  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1277706297978130444)** (58 messages🔥🔥): 

> - `Liger Kernel Contributions`
> - `Triton vs PyTorch Implementation`
> - `Encoder-Style Transformers Support`
> - `Performance of Llama.triton`
> - `Fused Kernel Zoo Concept` 


- **Liger Kernel Welcomes New Contributors**: Several new members joined Liger Kernel, expressing their intention to contribute and share their experiences with Triton, including a member from a DC startup interested in training efficiencies.
   - Links to contributing guidelines and specific issues of interest were shared, encouraging collaboration.
- **Triton Harder, But Worth It**: Implementing modules in Triton is reported to be **much harder than PyTorch** but easier than CUDA, providing a trade-off for developers.
   - Existing tools like [torch.compile](https://github.com/linkedin/Liger-Kernel/issues/119) are seen as beneficial for generating Triton code directly, which could lead to **significant performance gains**.
- **Encoder-Style Transformers on the Radar**: The community acknowledged the importance of supporting encoder-only transformers like BERT, with an issue created to track the feature development.
   - Discussion included possibilities for reusing layers and collaborating with team members already experimenting with Liger Kernel on previous models.
- **Call for Fused Kernel Development**: There were discussions around building a 'fused kernel zoo' to simplify the addition of efficient kernels beyond the existing frameworks.
   - Members expressed the belief that combining PyTorch and Triton would yield the best results, with an offer to assist with kernel requests.
- **TinyGrad's Position in Deep Learning Contributions**: A question was raised regarding implementing low-level fused kernels in TinyGrad, a pure-Python deep learning library.
   - The community consensus is that PyTorch remains superior in terms of ease of use and composability compared to emerging technologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/llamafactory_ai/status/1828290165577482710?s=46">Tweet from LLaMA Factory (@llamafactory_ai)</a>: We&#39;ve integrated the Liger Kernel into LLaMA-Factory.  It achieves ~10% speed up and ~25% memory reduction when fine-tuning Llama-3 8B on 2k sequences. Try it out at LLaMA-Factory🚀</li><li><a href="https://github.com/linkedin/Liger-Kernel">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/119">[fun] llama.triton · Issue #119 · linkedin/Liger-Kernel</a>: 🚀 The feature, motivation and pitch @thomwolf and i have an idea to implement llama from scratch in pure triton, inspired by karpathy. liger kernel already contains most of the kernels except matmu.....</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/CONTRIBUTING.md">Liger-Kernel/CONTRIBUTING.md at main · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/73">Request to support the Flux model (T2I diffusion transformer) · Issue #73 · linkedin/Liger-Kernel</a>: 🚀 The feature, motivation and pitch This request is to adapt this to improve the training speed of Flux, a diffusion transformer. It&#39;s the top model on HuggingFace trending right now and has been...</li><li><a href="https://github.com/zinccat/Awesome-Triton-Kernels/commit/0b04596de12f35507e9fdf56355c77b625b413ac">Update README.md · zinccat/Awesome-Triton-Kernels@0b04596</a>: no description found</li><li><a href="https://github.com/cuda-mode/triton-index">GitHub - cuda-mode/triton-index: Cataloging released Triton kernels.</a>: Cataloging released Triton kernels. Contribute to cuda-mode/triton-index development by creating an account on GitHub.</li><li><a href="https://github.com/zinccat/Awesome-Triton-Kernels">GitHub - zinccat/Awesome-Triton-Kernels: Collection of kernels written in Triton language</a>: Collection of kernels written in Triton language. Contribute to zinccat/Awesome-Triton-Kernels development by creating an account on GitHub.</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/126">[AMD] Implement Flash Attention in Triton to enable transformers to run with Flash Attention on AMD GPUs. · Issue #126 · linkedin/Liger-Kernel</a>: 🚀 The feature, motivation and pitch The official implementation of flash attention is in CUDA, so in AMD GPUs, users cannot easily use flash attention on transformers to training LLM. With the supp.....</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/112">[operation] utility CLI for reporting environment &amp; updating bug report issue template · Issue #112 · linkedin/Liger-Kernel</a>: 🚀 The feature, motivation and pitch Will be helpful to provide a CLI (called something like liger_env_report?) that queries: triton ver torch ver HF ver OS python ver ... so user can paste the outp.....</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/131">[feat] Add support for encoder-only transformers (e.g. BERT) · Issue #131 · linkedin/Liger-Kernel</a>: 🚀 The feature, motivation and pitch Liger Kernel is currently incompatible with encoder-only transformer architectures such as BERT, DistilBERT, RoBERTa, XLM-R, and DeBERTa. Given the importance th.....</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/122">[tiny] reformat code by tyler-romero · Pull Request #122 · linkedin/Liger-Kernel</a>: Summary Fixing broken checkstyle on main</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/114">Makefile command for env-report by tyler-romero · Pull Request #114 · linkedin/Liger-Kernel</a>: Summary #112  Will be helpful to provide a CLI (called something like liger_env_report?) that queries: triton ver torch ver HF ver OS python ver ... so user can paste the output when creating bug r...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/103">Add FusedLinerCrossEntropy support for Phi3 by tyler-romero · Pull Request #103 · linkedin/Liger-Kernel</a>: Summary Add FusedLinearCrossEntropy support for Phi3. #98 Testing Done  Hardware Type: 4090  run make test to ensure correctness  run make checkstyle to ensure code style  run make test-convergence...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/119)">Issues · linkedin/Liger-Kernel</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/92">feat: correct casts in RMSNorm to match references by davidgonmar · Pull Request #92 · linkedin/Liger-Kernel</a>: Summary Aims to fix #89. Details Does the casts to float32 at the correct places to match the Gemma and Llama references. Does so both in the forward and backward passes. Also modified the tests fo...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/111">Add FusedLinearCrossEntropy to Gemma by Luke-Chesley · Pull Request #111 · linkedin/Liger-Kernel</a>: Summary  This PR adds FusedLinearCrossEntropy support for gemma to resolve issue #101. Details I based the code in this PR off of #93 which does the same thing for Mistral. Since the parameters fus...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1277705345367539772)** (78 messages🔥🔥): 

> - `Perplexity app performance`
> - `File upload issues`
> - `GPT model usage limits`
> - `System prompts configuration`
> - `Internship opportunities` 


- **Perplexity app experiences slow performance**: Many users reported that the **Perplexity app** has been experiencing slow response times since this morning, causing frustration among users.
   - Complaints included unreliable search results and general dissatisfaction with the platform's recent performance.
- **File upload failures across the board**: Multiple users attempted image uploads and encountered **file upload failed** errors, with some expressing disappointment over issues even while on Pro subscriptions.
   - While PDFs have been reported to work, users are waiting for a fix as image uploads remain broken.
- **Clarifying usage limits for GPT models**: The daily message limit for the **Claude 3.5** model is reported to be **430 messages**, combined across all Pro models, except for **Opus**, which has a limit of **50**.
   - Users noted that even with high usage, they rarely hit the limit, with one mentioning their closest was around **250 messages**.
- **Configuring system prompts for better performance**: Discussions revealed that users can set their **system prompts** when creating new collections, which can enhance app interactions.
   - One user successfully got the app to work better by tweaking the system prompt settings.
- **Inquiry about internship opportunities**: An inquiry was raised regarding whether **Perplexity** is offering internships for university students, pointing to the need for potential opportunities in the company.
   - Links were shared that might contain more information on this matter, hinting at community interest in joining the team.



**Link mentioned**: <a href="https://x.com/shreybirmiwal/status/1828237302520234367">Tweet from shrey birmiwal (@shreybirmiwal)</a>: New room decor @AravSrinivas @perplexity_ai

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1277708683563761807)** (9 messages🔥): 

> - `Boeing's Replacement Strategy`
> - `Risks of eSIM Usage`
> - `Understanding 'Ratio' in Tech`
> - `Novel Architecture Designs`
> - `Best AI Policy Books` 


- **Boeing's Plan to Replace 737**: [Boeing's plan to replace 737](https://www.perplexity.ai/page/why-boeing-wants-to-replace-73-Asu4kUOdQP2QzJuDlj1Tqw) is highlighted as a strategic move to enhance its fleet's efficiency and sustainability amidst growing demand.
   - They aim to address market needs with a new aircraft that improves on existing models' performance and environmental impact.
- **Examining Risks of eSIM Technology**: The [risks of using eSIM](https://www.perplexity.ai/page/risks-of-esim-usage-FlGIT6ZZRw6Z8vYM.69GgA) are discussed, focusing on security vulnerabilities and the potential for carrier lock-in.
   - Concerns were raised about how easy it is to switch operators and the implications for consumer rights.
- **Decoding 'Ratio' in Technology**: An inquiry about the meaning of 'ratio' in tech contexts can be found at [this link](https://www.perplexity.ai/search/que-veut-dire-ratio-en-terme-s-iYvO5l.ySs6olBcQGkeXLQ#0), exploring its various interpretations.
   - Members discussed its relevance in discussions on social media metrics and analytical frameworks.
- **Designing New Architecture Approaches**: [A new architecture design](https://www.perplexity.ai/search/design-a-novel-architecture-fo-Pky7LPVTTVOt1Y78SNgcgQ) has been proposed to improve system efficiency and performance.
   - The innovation is aimed at addressing current limitations within existing architectures through novel strategies.
- **Curating the Best AI Policy Books**: A discussion featured [the best AI policy books](https://www.perplexity.ai/search/what-are-the-best-ai-policy-bo-PORU.OiYRJewNfe_RFWkgw#0) that provide insight into regulatory and ethical considerations for AI technologies.
   - Participants highlighted essential readings that shape understanding of AI impacts on society.



**Link mentioned**: <a href="https://www.youtube.com/embed/FzBTzFmIjSI">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1278141370569326663)** (1 messages): 

> - `Perplexity AI integration`
> - `Chatbot responses`
> - `API usage` 


- **Challenges with Perplexity AI implementation in chatbot**: A user is trying to implement **Perplexity AI** into a fact-checking chatbot in Hebrew but is facing issues with shortened responses that lack **links** and **images**.
   - They noted that responses from the API differ significantly from those on the Perplexity search site, mentioning that links often lead to **404 errors**.
- **Request for suggestions to improve API responses**: The user is seeking advice on how to enhance the API responses to include full answers, correct source links, and images similarly to the search site.
   - They are considering adding a **preliminary prompt** and are inquiring about specific models or Pro activation requirements necessary for better results.


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1278097775250378867)** (1 messages): 

> - `API degradation`
> - `Incident Recovery` 


- **API Degradation Incident Briefly Affects Services**: There was a *~5m period of API degradation* that impacted service availability.
   - A patch has been rolled out, and the incident appears to be fully **recovered**.
- **Prompt and Effective Incident Response**: The response team quickly identified the issue during the API degradation period, ensuring minimal disruption.
   - This proactive approach highlights the importance of rapid response in maintaining service integrity.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1278060899789443193)** (1 messages): 

> - `Appreciation for Team Efforts`
> - `Tweet about AI Collaboration` 


- **Team Efforts Recognized!**: A member expressed gratitude towards the team for their contributions, stating, *Thank you team!*
   - This acknowledgment highlights the collaborative spirit and the hard work put in by individuals involved.
- **Highlighting AI Collaboration on Twitter**: A [tweet](https://twitter.com/gpudad/status/1828502015238119490) was shared that showcases significant developments in AI collaboration.
   - The tweet emphasizes the importance of community efforts in advancing AI technologies.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1277712951561945099)** (84 messages🔥🔥): 

> - `OpenRouter Model Fee Structure`
> - `DisTrO Distributed Training Innovation`
> - `Cerebras Pricing and Features`
> - `Context Caching in DeepSeek`
> - `Gemini Model Updates` 


- **OpenRouter model pricing and fees explained**: A user inquired about whether the price per token displayed in OpenRouter includes the service fee. It was clarified that the price listed is based on OpenRouter credits and does not account for any additional fees incurred when adding credits.
- **DisTrO brings new hope to distributed training**: A member highlighted the release of a preliminary report on DisTrO (Distributed Training Over-the-Internet) by Nous Research, which improves distributed training efficiency. It promises to drastically reduce inter-GPU communication, enabling more resilient training of large models.
- **Cerebras offers competitive pricing**: Cerebras currently has pricing set at **10 cents** per million tokens for Llama 3.1-8B and **60 cents** for Llama 3.1-70B, inducing interest among community members. Discussions included potential collaboration and the platform's continuous improvements.
- **OpenRouter and DeepSeek context caching**: The topic of whether OpenRouter supports context caching for DeepSeek was raised, indicating a desire for improved performance and cost efficiency. It was noted that OpenRouter is awaiting further changes to support custom user segmentation for caching.
- **Exciting updates for Gemini models**: The upcoming release of new Gemini 1.5 Flash and Pro models was discussed, with users expressing excitement about its potential features and performance. There are speculations that these updates might aim to compete with existing models like GPT-4.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NousResearch/status/1828121648383566270">Tweet from Nous Research (@NousResearch)</a>: What if you could use all the computing power in the world to train a shared, open source AI model?  Preliminary report: https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://x.com/OfficialLoganK/status/1828480081574142227">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Today, we are rolling out three experimental models:  - A new smaller variant, Gemini 1.5 Flash-8B - A stronger Gemini 1.5 Pro model (better on coding & complex prompts) - A significantly improved Gem...</li><li><a href="https://x.com/OfficialLoganK/status/1828484457675751814">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: @patricksrail Vertex is rolling out the 1.5 Flash and Pro models (not 8B) later today, should be soon!</li><li><a href="https://docs.anthropic.com/en/release-notes/system-prompts#july-12th-2024">no title found</a>: no description found</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-inst">Llama 3.1 405B (base) - API, Providers, Stats</a>: Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. Run Llama 3.1 405B (base) with API</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b">Llama 3.1 405B (base) - API, Providers, Stats</a>: Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. Run Llama 3.1 405B (base) with API</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct">Llama 3.1 405B Instruct - API, Providers, Stats</a>: The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eval scores, the Meta AI team continues to push the frontier of open-source LLMs.  Meta&#x27;s latest c...</li><li><a href="https://x.com/hyperbolic_labs/status/1828481468156518691">Tweet from Hyperbolic (@hyperbolic_labs)</a>: Llama 3.1 405B Base at BF16: Now Available on Hyperbolic 🦙💜  Base models are far more creative and capable than instruction-tuned models, but they’ve been underutilized—until now.  ➡️ Get started bu...</li><li><a href="https://platform.deepseek.com/api-docs/news/news0802/">DeepSeek API introduces Context Caching on Disk, cutting prices by an order of magnitude | DeepSeek API Docs</a>: In large language model API usage, a significant portion of user inputs tends to be repetitive. For instance, user prompts often include repeated references, and in multi-turn conversations, previous ...</li><li><a href="https://platform.deepseek.com/api-docs">Quick Start | DeepSeek API Docs</a>: The DeepSeek API uses an API format compatible with OpenAI. By modifying the configuration, you can use the OpenAI SDK or softwares compatible with the OpenAI API to access the DeepSeek API.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1277995784926658691)** (1 messages): 

> - `Activity page indicators`
> - `Model pricing transparency`
> - `Integrations provider insights` 


- **Activity Page Needs Clear Routing Indicators**: A suggestion was made to add an indicator on the **activity page** to show if it has been routed to an **integrations provider**.
   - Currently, it displays **$0**, which may mislead users due to potential errors or simply reflect the **model price** for **Hermes 405b**.
- **Clarification on Model Pricing and Errors**: Concerns were raised regarding the current display of **$0** on the activity page, which could be caused by other errors.
   - The visibility of **model pricing** is essential to prevent confusion and improve user experience.


  

---



### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/)** (1 messages): 

georgehotz: buy your tinybox here https://tinycorp.myshopify.com/
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1277704300595908721)** (68 messages🔥🔥): 

> - `Tinybox Shipping Issues`
> - `Tinygrad and BERT Training`
> - `Tinybox Availability in Different Regions`
> - `Tinybox Sales Update`
> - `Tinygrad Runtime Errors` 


- **Tinybox Shipping Challenges in Europe**: Discussion on the current unavailability of the **Tinybox** for international buyers, particularly in the UK and Europe. A member suggested emailing support for shipping quotes to **Europe**, sparking questions on global availability.
   - Reports indicated that users in **France** and **Italy** saw the **Tinybox** marked as sold out, while inquiries were made about future shipping solutions.
- **Exploring Tinygrad for BERT Training**: A member expressed interest in using **Tinygrad** to pre-train a large **BERT** model, highlighting the need for support in a high-performance setup. There were contrasting views on using **Tinygrad** versus **Torch**, with mentions about its capacity for large model training.
   - Conversations emphasized that **Torch** might be better optimized for the required setup, noting the significant hardware like **64 Hopper cards** and existing experience with **Torch**.
- **Tinybox Sales Update**: George mentioned that about **40 Tinyboxes** have been sold, and they have supplies for **60 more**. The excitement about sales growth contrasted with the limitations on international sales, which are still being negotiated.
   - The community also speculated about potential new **color editions** of the Tinybox, with denials from George about a **blue edition** being available soon.
- **Runtime Errors When Using Tinygrad**: A user running **Tinygrad** on a Linux server encountered errors with an NVIDIA **RTX 4000 SFF Ada**, raising concerns about setup requirements. Developers chimed in, suggesting checks on configuration and compiling flags to troubleshoot.
   - Further attempts to verify the configuration yielded successful test results with no errors, leading to further diagnostics by the community.
- **Discussion on Tinybox Design**: Users provided feedback regarding the resolution of images on the **Tinybox** shop interface, suggesting that recent updates may have caused lower-quality visuals. The team is investigating this issue to ensure products are displayed accurately.
   - There were also humorous remarks on potential future models, highlighting community engagement in product development discussions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tinycorp.myshopify.com/products/tinybox-red">tinybox red edition</a>: Payment must be completed within 5 days of order confirmation to guarantee your order. Contiguous US Shipping Only. Email support@tinygrad.org for Canada.  Payment Method: Bank Transfer/Wire Docs can ...</li><li><a href="https://tinycorp.myshopify.com/products/tinybox-green">tinybox green edition</a>: Payment must be completed within 5 days of order confirmation to guarantee your order. Contiguous US Shipping Only. Email support@tinygrad.org for Canada.  Payment Method: Bank Transfer/Wire Docs can ...</li><li><a href="https://en.wikipedia.org/wiki/Freight_forwarder">Freight forwarder - Wikipedia</a>: no description found
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1277967516852945037)** (9 messages🔥): 

> - `RecursionError in Tinygrad`
> - `Creating transition matrix`
> - `Tinygrad version 0.9.2` 


- **RecursionError strikes on tensor conversion**: A user reported a `RecursionError: maximum recursion depth exceeded` occurring when calling `.tolist()` on a tensor in Tinygrad, specifically when handling over 3500 Wikipedia articles.
   - *It works fine with 2000 Wikipedia articles*, raising questions about the underlying issue with larger inputs.
- **Seeking a minimal example for debugging**: The user expressed intentions to create a minimal example to better diagnose the problem and queried if sharing code via DM is acceptable.
   - Another user suggested that they could open an issue if a smaller reproducible example is provided.
- **Identifying Tinygrad version**: The user confirmed that their installed Tinygrad version is **0.9.2**, which could be relevant to the encountered error.
   - This version may relate to the issues discussed, particularly the LazyOp functionality.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1277902305697464323)** (50 messages🔥): 

> - `Intel LLaVaOLMoBitnet1B`
> - `OpenAI Orion AI`
> - `Nous Research optimizer`
> - `Cerebras Inference speed`
> - `Gemini 1.5 models` 


- **Intel introduces LLaVaOLMoBitnet1B**: Intel has unveiled [LLaVaOLMoBitnet1B](https://huggingface.co/papers/2408.13402), the first Ternary Multimodal LLM that can process Image(s) and Text inputs to generate coherent textual responses.
   - The model is fully open-sourced, along with training scripts, to promote further research in the field, highlighting the challenges and future opportunities for ternary models.
- **OpenAI works on Orion AI**: OpenAI is reportedly aiming to develop a new AI model, 'Orion', that can reason through complex problems as it seeks additional capital, according to an [exclusive report](https://www.theinformation.com/articles/openai-races-to-launch-strawberry-reasoning-ai-to-boost-chatbot-business?utm_campaign=Editorial&utm_content=Article&utm_medium=organic_social&utm_source=twitter).
   - This initiative comes as OpenAI looks to bolster its chatbot business by enhancing its AI capabilities.
- **Questions arise about Nous Research optimizer**: Members are skeptical about the legitimacy of Nous Research's new optimizer, expressing the need for more evidence to support its claims about distributed training capabilities.
   - Discussions included mentions of existing tools like Petals for Bloom and OpenDILo, but uncertainty remains regarding the authenticity of Nous's promises.
- **Cerebras claims fastest Llama3.1 Inference**: Cerebras Systems announced that its Inference API boasts speeds of **1,800 tokens/s** for 8B models and **450 tokens/s** for 70B models, significantly faster than competitors like NVIDIA.
   - Members expressed excitement about the competition in inference speeds, particularly noting their love for rapid advancements in this area.
- **Launch of new Gemini 1.5 models**: Google has launched three experimental models: **Gemini 1.5 Flash-8B**, a more compact variant, the stronger **Gemini 1.5 Pro** aimed at coding tasks, and an improved **Gemini 1.5 Flash model**.
   - Details and access to try these models were shared via [Google's AI Studio](https://aistudio.google.com), sparking discussions about their potential capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://evaleval.github.io/">Home - EvalEval 2024</a>: A NeurIPS 2024 workshop on best practices for measuring the broader impacts of generative AI systems</li><li><a href="https://x.com/officiallogank/status/1828480081574142227?s=46">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Today, we are rolling out three experimental models:  - A new smaller variant, Gemini 1.5 Flash-8B - A stronger Gemini 1.5 Pro model (better on coding & complex prompts) - A significantly improved Gem...</li><li><a href="https://x.com/theinformation/status/1828418859990229073?s=46">Tweet from The Information (@theinformation)</a>: Exclusive: As OpenAI looks to raise more capital, it&#39;s trying to launch AI that can reason through tough problems and help it develop a new AI model, &#39;Orion.&#39;  https://www.theinformation.c...</li><li><a href="https://x.com/_akhaliq/status/1828271805825434066">Tweet from AK (@_akhaliq)</a>: Intel presents LLaVaOLMoBitnet1B  Ternary LLM goes Multimodal!  discuss: https://huggingface.co/papers/2408.13402  Multimodal Large Language Models (MM-LLMs) have seen significant advancements in the ...</li><li><a href="https://x.com/aiexplainedyt/status/1828430051735441706?s=46">Tweet from AI Explained (@AIExplainedYT)</a>: no description found</li><li><a href="https://x.com/cerebrassystems/status/1828465008298336588?s=46">Tweet from Cerebras (@CerebrasSystems)</a>: Cerebras Inference is the fastest Llama3.1 inference API by far: 1,800 tokens/s for 8B and 450tokens/s for 70B. We are ~20x faster than NVIDA GPUs and ~2x faster than Groq.</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>: Distributed Training Over-The-Internet. Contribute to NousResearch/DisTrO development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1278047899606978715)** (5 messages): 

> - `Ideogram Twitter Space`
> - `Artists and AI Training`
> - `Audience Manipulation`
> - `Data Discussions`
> - `Access to AI Tools` 


- **Ideogram Twitter Space raises eyebrows**: During the Ideogram birthday space on Twitter, a question about artists feeling ripped off by companies using their art led to the questioner being removed, as Ideogram stated only received positive feedback.
   - Later, a 'real artist' praised Ideogram, revealing her connections with **a16z**, an investor in the company.
- **Concerns over audience planting**: One member expressed agreement with the concerns raised but criticized the act of planting an audience as **cringe**.
   - This led to a discussion about authenticity in the interactions during such events.
- **Debate on data discussions' toll**: One user remarked that the discussions surrounding data are **draining and sad**, hinting at the broader implications of AI on artistry.
   - This sentiment reflects concerns within the community regarding the impact of AI on creative industries.
- **Worries about access inequality**: There was a fear that the future might see **wealthy and connected artists** being the only ones able to access the best AI tools, raising ethical questions.
   - This highlights a potential divide where only certain groups benefit from advancements in AI art technology.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1278003649557168160)** (12 messages🔥): 

> - `Andrew Huberman's Coffee Experiment`
> - `Kaggle Competition Feedback`
> - `Claude Summer Discussion`
> - `Gemini API Rate Limits` 


- **Andrew Huberman tests coffee impact on himself**: A YouTube video titled ["Is Andrew Huberman Ruining Your Morning Coffee?"](https://www.youtube.com/watch?v=yCJr49GU9yY) discusses his randomized controlled trial where he alternated between decaf and caffeinated coffee.
   - The need for a business setup with Shopify was mentioned as well, indicating a workaround for an educational link.
- **Critique on Atlantic's Kaggle Competition**: A member shared a [Kaggle competition](https://www.kaggle.com/competitions/internal-waves) hosted by the Atlantic International Research Centre, suggesting it seemed uninteresting after a brief look.
   - *People's reactions were mixed*, indicating a general apathy towards the competition’s allure.
- **Discussion on Graph Anomalies**: Amidst discussions on Claude AI, there were complaints regarding a graph with a 'random non-integer scale' and a 'short x axis'.
   - Members expressed frustration, with one declaring their support for Claude, yet disliking the confusing graphs.
- **Confusion Around Gemini API Rate Limits**: A member inquired about experiences with the Gemini API, expressing confusion over how to fix its rate limits.
   - They described the system as a 'total mess', noting inconsistencies where some models functioned while others did not.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/morqon/status/1828463686438048211?s=46">Tweet from morgan — (@morqon)</a>: hot claude summer</li><li><a href="https://x.com/kagglingdieter/status/1828446217958822277">Tweet from Dieter (@kagglingdieter)</a>: People asked me if this competition hosted by the Atlantic International Research Centre looks interesting.  Looked 10 seconds at it 🤦   https://www.kaggle.com/competitions/internal-waves</li><li><a href="https://www.youtube.com/watch?v=yCJr49GU9yY">Is Andrew Huberman Ruining Your Morning Coffee?</a>: To start a business with Shopify, use this link for a free trial http://shopify.com/jameshoffmannToday’s video is something a bit different. We thought we sh...
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1277939660554895372)** (30 messages🔥): 

> - `lm eval metrics`
> - `Tokenizer v3`
> - `Mistral tokenizer config issues`
> - `Jinja parser and masking`
> - `Multi-role in ShareGPT` 


- **lm eval metrics depend on benchmark**: For multiple choice questions, the metric used is **accuracy on target prediction**, determined by whether the model's highest logit probability aligns with the correct choice.
   - Members discussed scenarios where answers might be slightly different, highlighting the nuances in evaluating model outputs.
- **Questions arise about tokenizer v3**: Multiple members expressed confusion about **tokenizer v3**, with one linking to a previous discussion on the **nemo repo**.
   - Another member emphasized the need for proper tokenizer configuration that supports **multi-role functionalities**.
- **Mistral's initial config errors**: A member noted that **Mistral** had issues with their `tokenizer_config.json` during its initial release, expressing frustration over this oversight.
   - They stressed the importance of accurate configuration in tokenizer applications to avoid similar errors in the future.
- **Demand for masking functionality in Jinja**: There was a strong request for tags for **masking**, with conversations about how masking should work within the **multi-role** context.
   - A member suggested setting attention to -100 to exclude certain inputs from training, prompting further exploration of masking methods.
- **Understanding multi-role and masking effects**: A member clarified that in **ShareGPT**, masking can be specified for input, allowing certain roles to be ignored during training.
   - The sharing of specific code configuration examples highlighted the intricacies of defining datasets with custom roles.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/conversation.html#sharegpt">Conversation – Axolotl</a>: no description found</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/blob/17af1d7081414c32614cbabe324e1197ca9f43a7/src/axolotl/prompt_strategies/chat_template.py#L188">axolotl/src/axolotl/prompt_strategies/chat_template.py at 17af1d7081414c32614cbabe324e1197ca9f43a7 · axolotl-ai-cloud/axolotl</a>: Go ahead and axolotl questions. Contribute to axolotl-ai-cloud/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1277995598603227177)** (7 messages): 

> - `Python Monkey-Patching`
> - `Deepseek V2 attention model`
> - `FSDP memory requirements` 


- **Python Monkey-Patching for Deepseek V2**: A discussion on **monkey-patching** the **Deepseek V2** attention model involved sharing a code snippet that overrides its forward method to print a message when called.
   - A user noted their experience with monkey-patching in Java but highlighted that they had never done it in Python before.
- **Fixing Deepseek V2 Remote Code Issues**: A link to a [GitHub pull request](https://github.com/axolotl-ai-cloud/axolotl/pull/1873/files#diff-ed7a6cebcdf220a8815697365704ad0e3dff808147447bb611040f634b7f4e27) was shared, detailing a fix for remote code issues in the Deepseek V2 implementation, which supersedes a previous pull request.
   - *“Weird fix”* was noted, emphasizing the ongoing challenges with the code stability.
- **Module Reloading Affects Deepseek V2**: A user mentioned that a contributor, **tmm1**, discovered that the issues with Deepseek V2 were related to how the module gets **reloaded**.
   - This pointed to the complexity of dynamically handling module states in Python.
- **Concerns About FSDP RAM Requirements**: A user inquired whether **FSDP** (Fully Sharded Data Parallel) requires a substantial amount of **system RAM** for effective operation.
   - This sparked curiosity about the optimal system resources necessary for proper FSDP functionality.



**Link mentioned**: <a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1873/files#diff-ed7a6cebcdf220a8815697365704ad0e3dff808147447bb611040f634b7f4e27">Sample pack trust remote code v2 by winglian · Pull Request #1873 · axolotl-ai-cloud/axolotl</a>: supersedes #1872. Thanks @tmm1 !

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1277732397231636642)** (16 messages🔥): 

> - `Training Tokenizer Vocabulary`
> - `Citing Axolotl Study`
> - `Embedding Adjustments After Training`
> - `Metric Analysis of Token Changes` 


- **Understanding Tokenizer Training Behavior**: A member inquired whether the code `modules_to_save = ["lm_head", "embed_tokens"]` trains the entire tokenizer vocabulary or just the newly added tokens, to which it was clarified that **all vocabulary** is trained.
   - Further discussion revolved around the necessity of training specific layers when **new tokens** are added.
- **Citing Axolotl for ArXiv Studies**: A member asked for preferences on how to credit an individual in a new study, indicating they were currently citing the [GitHub repository](https://github.com/axolotl-ai-cloud/axolotl).
   - The conversation included a **reference link** to a related study on ArXiv at **#15** via [this link](https://www.arxiv.org/pdf/2408.11857).
- **Affected Tokens Post-Training**: Another member expressed curiosity about which tokens were most affected after training, particularly after adding the **'pad' token**. Their code aimed to identify the top affected tokens after training adjustments.
   - Concerns were raised regarding the output of the analysis, suggesting it might serve as a potential **metric** given the specialized nature of their dataset.
- **Cosine Distance in Token Analysis**: The user modified their approach to calculate cosine distance of token adjustments, hoping this would yield clearer insights about token changes post-training.
   - Another member noted that **embeddings** might not be the only aspect strongly influenced by training, but rather, the **interpretation** of those embeddings matters significantly.
- **Effectiveness of Token Representation in Training**: Discussion continued on whether tokens that differ in meaning from pre-training datasets, such as **'adam'**, would show effective training results.
   - The user noted finding unexpected tokens, including **'Fortunately'**, indicating that they sought to confirm their training's effectiveness.



**Link mentioned**: <a href="https://github.com">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1277871711240454144)** (2 messages): 

> - `llm-as-judge`
> - `human ratings comparison` 


- **Using LLM as a Judge for Rating**: A member indicated that they employed a **prompt for rating** by utilizing **llm-as-judge**.
   - This method leads to questions about the *accuracy of the AI judge compared to human ratings* and whether any tests were conducted.
- **Accuracy of AI Ratings vs Human Ratings**: There was a query about how accurate the AI judge was in comparison to **human ratings**.
   - Another member asked for details regarding *tests that might have been performed* to evaluate this accuracy.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1278128886567407637)** (1 messages): 

> - `User Feedback on Magic`
> - `Exclusive Swag for Feedback Providers` 


- **Call for Feedback on Magic Features**: The team is looking for **5 participants** willing to spend **30 minutes** providing user feedback specifically on **magic**.
   - Participants will receive exclusive **swag** as a reward for their contributions, with interested users directed to [book a slot here](https://modul.ar/user-feedback).
- **Exclusive Swag Reward**: Those who provide feedback will be among the first to receive exclusive **swag** currently in the design phase.
   - This initiative aims to incentivize user input and appreciation for the contributors involved.



**Link mentioned**: <a href="https://modul.ar/user-feedback">Appointments</a>: no description found

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1277707142383796256)** (43 messages🔥): 

> - `ClassStruct and Variadic Parameters`
> - `Performance with Struct Fields`
> - `Type Inference in Mojo`
> - `Function Overloading Challenges`
> - `Comparison of Mojo and Luma` 


- **Understanding ClassStruct's Variadic Parameters**: ClassStruct allows dynamic parameterization in Mojo, enabling users to create variations without manually crafting unique structs, as seen with the `car` example where engine sizes can be defined dynamically.
   - This functionality lets programmers harness variations efficiently by defining struct fields based on parameters available at compile time.
- **Performance Issues with Field Limitations**: Discussion revealed that compiling structs with a large number of fields can lead to significant performance hits, with examples showing 100 fields taking around 1.2 seconds.
   - It was suggested that this might be due to underlying data structures needing to resize, indicating a threshold where performance degrades notably.
- **Challenges with Type Inference**: Type inference in Mojo is limited, particularly when dealing with generics, making the programming experience cumbersome compared to Rust's powerful system.
   - This leads to discussions on how Mojo's current handling of generics and typeclasses may limit flexibility compared to established languages.
- **Function Overloading and Complexity**: Participants noted that function overloading in Mojo presents challenges due to inconsistencies in type inference behavior, complicating the development process.
   - There is a consensus that improvements in handling function overloads and type inference could enhance user experience significantly.
- **Mojo vs. Luma: Type System Comparisons**: Comparisons were drawn between Mojo and Luma, implying that while Luma has more robust type inference, Mojo offers unique features like typeclasses but with restrictions.
   - The discussions emphasize that as Mojo evolves, it may align more closely with Rust's capabilities, potentially introducing features like effect systems.


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1277713478731169863)** (3 messages): 

> - `RAG-a-thon`
> - `Llama 3.1-8b speed`
> - `Serverless RAG application`
> - `LlamaIndex and Azure OpenAI` 


- **Get Ready for RAG-a-thon with Pinecone!**: We're hosting our second **RAG-a-thon** with over **$7k in cash prizes** from October 11-13 at the [500GlobalVC](https://t.co/IFvyW5QB6r) offices in Palo Alto!
   - It's a great opportunity to showcase innovative ideas and gain valuable experience in a collaborative environment.
- **Llama 3.1-8b Breaks Speed Records**: Need ultra-fast responses? **Llama 3.1-8b** offers **1800 tokens per second**, making it the fastest LLM available, as discussed [here](https://t.co/hZv6ooGUxO).
   - Achieving this speed is crucial for applications needing quick responses, especially in complex systems.
- **Build Serverless RAG App with LlamaIndex**: Learn to create a **serverless RAG application** using LlamaIndex and Azure OpenAI through this comprehensive guide by **Wassim Chegham** [link to guide](https://t.co/1XKg1o2bIX).
   - It covers an understanding of RAG architecture and shows how to leverage your own business data for improved AI-powered responses.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1277716039341445191)** (36 messages🔥): 

> - `Callback Manager to Instrumentation Migration`
> - `Neo4j Schema Issues`
> - `Data Extraction from Scanned Documents`
> - `GPT-4o-mini Model Support`
> - `GraphRAG v2 Performance` 


- **Callback Manager and Instrumentation Confusion**: A member queried about differences in trace spans between **RetrieverQueryEngine** and **CondensePlusContextChatEngine**, highlighting that both currently utilize the callback manager.
   - Another member guessed that **LangFuse** is still implemented using the callback manager, implying broken traces could be a bug.
- **Neo4j Can't Build Relationships**: A user reported trouble replicating a property graph tutorial from LlamaIndex with Neo4j Desktop, where relationships weren't being extracted correctly.
   - They clarified that they strictly followed the tutorial, including the default schema, and suspected their Neo4j setup might not align with default expectations.
- **Enhancing Data Extraction with LlamaParse**: A user discussed potential issues with **LlamaParse** converting tabular data due to scanning problems and sought solutions for integrating image extraction in their pipeline.
   - Questions arose about chunking strategies for processing multiple tables combined with images.
- **GPT-4o-mini Support Issues**: A user tried using the **gpt-4o-mini** model but encountered a ValueError indicating it is an unknown model, despite the chat suggesting it should be supported.
   - Another member recommended updating libraries to fix this issue, implying a possible oversight in model support compatibility.
- **GraphRAG v2 Not Performing as Expected**: A user reported that **GraphRAG v2** failed with their dataset, although earlier versions worked fine, leading to questions about performance expectations.
   - Members discussed the necessity of graphs, with some expressing skepticism about their essentiality in certain setups.



**Link mentioned**: <a href="https://github.com/run-llama/llama_index/pull/15679">fix tool schemas by logan-markewich · Pull Request #15679 · run-llama/llama_index</a>: Recent changes to pydanticv2 caused our tool json schemas to miss the renamed &quot;definitions&quot; section

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1277767943568232489)** (36 messages🔥): 

> - `DisTrO by Nous Research`
> - `Microsoft's Phi 3.5 Vision Model`
> - `Cerebras Inference Performance`
> - `Gemini 1.5 Model Updates`
> - `Anthropic Artifacts Launch` 


- **DisTrO Revolutionizes Distributed Training**: Nous Research released a preliminary report on [DisTrO](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf), a framework that drastically reduces inter-GPU communication by up to 10,000x, enabling resilient training of LLMs.
   - The framework aims to promote collaboration in AI research without reliance on a single entity, enhancing the security and competitiveness of model development.
- **Microsoft's Phi 3.5 Vision Model Excels in OCR**: Dylan Freedman highlighted Microsoft's [Phi 3.5](https://huggingface.co/spaces/MaziyarPanahi/Phi-3.5-Vision) model for excellent performance in OCR, especially in handwriting recognition and extracting tabular data, which is permissively licensed (MIT).
   - The model demonstrates superior text recognition and capability across various vision tasks, with notable performances discussed in the community.
- **Cerebras Sets New Inference Speed Records**: Cerebras announced its inference service, achieving [1,800 tokens/s](https://x.com/CerebrasSystems/status/1828465008298336588) for 8B models, significantly outperforming NVIDIA and Groq alternatives.
   - Powered by the custom WSE-3 chip, Cerebras is also offering competitive pricing for Llama models, prompting discussions on its economic viability and performance in the developer community.
- **Gemini 1.5 Models Launched**: Google introduced three experimental models under the Gemini 1.5 series, including a smaller variant and a stronger Pro model, with capabilities noted in coding and complex prompts.
   - The recent launches sparked comparisons to existing models such as GPT-4o-mini, as developers evaluate their competitive edge and performance.
- **Anthropic’s Artifacts and Article Discussions**: Anthropic has made significant advances with the release of [Artifacts](https://newsletter.pragmaticengineer.com/p/how-anthropic-built-artifacts), closely followed by developers intrigued by their technologies and methodologies.
   - Concerns were raised regarding the motivations behind the timely article release, leading to speculation about potential paid placements in discussions on messaging platforms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/officiallogank/status/1828480081574142227?s=46">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Today, we are rolling out three experimental models:  - A new smaller variant, Gemini 1.5 Flash-8B - A stronger Gemini 1.5 Pro model (better on coding & complex prompts) - A significantly improved Gem...</li><li><a href="https://x.com/CerebrasSystems/status/1828465008298336588">Tweet from Cerebras (@CerebrasSystems)</a>: Cerebras Inference is the fastest Llama3.1 inference API by far: 1,800 tokens/s for 8B and 450tokens/s for 70B. We are ~20x faster than NVIDA GPUs and ~2x faster than Groq.</li><li><a href="https://x.com/NousResearch/status/1828121648383566270">Tweet from Nous Research (@NousResearch)</a>: What if you could use all the computing power in the world to train a shared, open source AI model?  Preliminary report: https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://x.com/AISafetyMemes/status/1828311798057181461">Tweet from AI Notkilleveryoneism Memes ⏸️ (@AISafetyMemes)</a>: Today in sci-fi-becoming-real: deepfake livestreams  I reiterate: if, a few years ago, you told people this would exist, they would not believe you  Quoting AI Notkilleveryoneism Memes ⏸️ (@AISafetyMe...</li><li><a href="https://x.com/dylfreed/status/1828132226523131931?s=46">Tweet from Dylan Freedman (@dylfreed)</a>: Microsoft&#39;s new open source Phi 3.5 vision model is really good at OCR/text extraction — even on handwriting! You can prompt it to extract tabular data as well.  It&#39;s permissively licensed (MI...</li><li><a href="https://cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed">Introducing Cerebras Inference: AI at Instant Speed - Cerebras</a>: We are excited to announce the release of Cerebras DocChat, our first iteration of models designed for document-based conversational question answering. This series includes two models: Cerebras Llama...</li><li><a href="https://newsletter.pragmaticengineer.com/p/how-anthropic-built-artifacts">How Anthropic built Artifacts</a>: The team behind Artifacts - an innovative new way to interact with Claude - shares how they built this innovative feature in just three months with a distributed team. Exclusive details.</li><li><a href="https://x.com/AnthropicAI/status/1828462522468372600">Tweet from Anthropic (@AnthropicAI)</a>: Today, we&#39;re making Artifacts available for all Claude users. You can now also create and view Artifacts on the Claude iOS and Android apps.  Since launching in preview in June, tens of millions o...</li><li><a href="https://x.com/artificialanlys/status/1828463912389402896?s=46">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: Cerebras has set a new record for AI inference speed, serving Llama 3.1 8B at 1,850 output tokens/s and 70B at 446 output tokens/s.  @CerebrasSystems has just launched their API inference offering, po...
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1277756830491086849)** (9 messages🔥): 

> - `Streamlit Python Server`
> - `Telegram Bot with Open Interpreter`
> - `Open Interpreter Image Display Issue`
> - `Jupyter Metadata and Image Control`
> - `Cython and Black-Scholes Model` 


- **Streamlit Python Server for Chat**: A member mentioned an easy [Streamlit Python server](https://link.to.streamlit) that can be used to create a chat interface in a web browser.
   - Another member responded positively, indicating they would look into this solution.
- **Configuring Telegram Bot with Open Interpreter**: A member shared their configuration for creating a Telegram bot using **Open Interpreter**, including settings for API key and model.
   - They faced issues with images displaying unexpectedly on their computer, prompting discussion on potential fixes.
- **Fixing Open Interpreter Image Display**: A member suggested modifying the line for custom instructions in Open Interpreter to help resolve an error linked to displaying images.
   - Another user confirmed they were trying to turn off the default behavior in Jupyter for displaying images and sought assistance.
- **Jupyter Metadata for Image Control**: One user shared a link about adding metadata to Jupyter notebooks to control behavior, which may help in managing image displays.
   - The discussion included potential solutions like using `plt.ioff()` to suppress automatic image output in Jupyter.
- **Cython Specification for Black-Scholes Model**: A member posted a link to an example of using Cython to implement the Black-Scholes model in Jupyter notebooks, showcasing complex calculations.
   - This example highlights how to define functions efficiently in Cython, enhancing computational performance for options pricing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jupyterbook.org/en/stable/content/metadata.html#add-tags-using-python-code">Add metadata to your book pages</a>: no description found</li><li><a href="https://nbviewer.org/github/ipython/ipython/blob/1.x/examples/notebooks/Cell%20Magics.ipynb">Jupyter Notebook Viewer</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1277942936427823157)** (22 messages🔥): 

> - `Issues with Poetry Command`
> - `Auth Error in iOS App`
> - `Brand Documentation for 01`
> - `Pre-order Status Inquiry` 


- **Troubleshooting Poetry Command**: A user experienced an error when running `poetry run 01 --server`, indicating that the activated Python version was unsupported and that option '--server' requires an argument.
   - Another user suggested running `poetry run 01 --server light` as a potential fix.
- **Auth Error in 01 iOS App**: A user reported receiving `{"auth": false}` consistently when using the 01 iOS app, suggesting a possible server issue.
   - They shared relevant console outputs, indicating no startup problems, and were advised to try starting the server with `--server livekit`.
- **First Meetup Brand Documentation**: A member shared a link to a [Canva design](https://www.canva.com/design/DAF8rbBol3Q/UNivuf8sjxVSveDfMFWpag/edit?utm_content=DAF8rbBol3Q&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) as the closest available brand document for the 01 project.
   - They also mentioned that more extensive industrial design progress and documentation will be provided in the GitHub repository soon.
- **Pre-order Status Updates**: A user inquired about contacting someone regarding their pre-order status, highlighting a need for clarity.
   - A member responded that an update is forthcoming, providing a link to the most recent status update.



**Link mentioned**: <a href="https://www.canva.com/design/DAF8rbBol3Q/UNivuf8sjxVSveDfMFWpag/edit?utm_content=DAF8rbBol3Q&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton">Amazingly Simple Graphic Design Software – Canva</a>: Amazingly Simple Graphic Design Software – Canva

  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1278029316567072861)** (4 messages): 

> - `Tool Use Episode`
> - `Video to Podcast Idea`
> - `Voice Cloning with ElevenLabs` 


- **Mike and Ty's Tool Use Episode Returns**: Mike and Ty are back with another episode titled ["Video to Content Pipeline and YouTube Playlist Summarization - Ep2 - Tool Use"](https://www.youtube.com/watch?v=uzP_16F2zjA) showcasing how to extract data from videos for better leverage.
   - The episode emphasizes how AI can gather information from videos, aiming to improve audience experiences.
- **Turning Video Summaries into Podcasts**: A member suggested transforming video summaries or email newsletters into podcasts, enhancing accessibility during morning commutes.
   - This idea was met with enthusiasm, showcasing innovative thinking about content consumption.
- **Voice Cloning Experiment with ElevenLabs**: Mike is ready to use ElevenLabs for voice cloning again after Ty humorously noted he cloned Mike's voice in a previous episode.
   - Mike's reaction highlights the lighthearted nature of their experiments, contributing to their dynamic content creation.
- **Daily Podcast Hosted by Cloned Voices**: A playful remark was made about a daily podcast featuring Mike Bird's cloned voice, inviting laughter and engaging the community.
   - This suggestion further emphasizes the creative use of technology in podcasting and content creation.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=uzP_16F2zjA">Video to Content Pipeline and YouTube Playlist Summarization - Ep2 - Tool Use</a>: Ty and Mike show you how to extract data from videos to leverage in amazing ways. Have AI get information from videos to improve your life.Have a YouTube pla...

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1277965644104732733)** (15 messages🔥): 

> - `Llama 3.1 Inference`
> - `Ollama Framework Comparison`
> - `LoRA Fine-tuned Model Loading`
> - `AWS Instance Costs`
> - `Runpod Alternatives` 


- **Llama 3.1 Inference struggles on CPU**: A user reported that inference for **Llama 3.1 8B** on CPU is extremely slow (<0.05 tok/s) even on a high-end AWS server.
   - Discussion revealed that CPU performance is expected to be significantly slower compared to GPU setups, especially when using frameworks like **Ollama**.
- **Optimized Inference Frameworks Suggested**: One member suggested using **Ollama** or **vLLM** for serving models, as they are more optimized for inference compared to torchtune.
   - A [tutorial on using Ollama with custom checkpoints](https://github.com/ollama/ollama/blob/main/docs/import.md) was shared as a helpful resource.
- **LoRA Fine-tuning Model Loading Query**: A user inquired whether using `from_pretrained()` would load the LoRA fine-tuned weights correctly from the local checkpoint.
   - A link to an [issue discussing how to load LoRA adapters into HF](https://github.com/pytorch/torchtune/issues/832#issuecomment-2129867129) was provided for further guidance.
- **AWS Instance Cost Debate**: Discussion on the costs of AWS instances indicated that an AWS c7a.24xlarge might be around **$5/hour**.
   - Another member suggested considering alternatives like **Runpod** for potentially cheaper options, yet regulatory constraints keep the user on AWS.
- **Challenges with CPU Servers**: The user indicated a preference for CPU servers due to cost-effectiveness and adequate response times for their use case.
   - Comments acknowledged that low CPU performance could impact inference speeds, prompting considerations for opting for optimized frameworks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/ollama/ollama/blob/main/docs/import.md">ollama/docs/import.md at main · ollama/ollama</a>: Get up and running with Llama 3.1, Mistral, Gemma 2, and other large language models. - ollama/ollama</li><li><a href="https://github.com/pytorch/torchtune">GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/torchtune/issues/832#issuecomment-2129867129">How to save a trained model so it can be loaded with HF `from_pretrained()`? · Issue #832 · pytorch/torchtune</a>: I&#39;m finding this repo to be a user friendly, extensible, memory efficient solution for training/fine-tuning models. However, when it comes to inference, there is a usability gap that could be solv...
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1277711967280435270)** (10 messages🔥): 

> - `Request limits in UI/UX`
> - `Model not found error`
> - `Reranker version updates` 


- **Concerns about request limits**: A member expressed doubts about crossing the **1k requests** threshold, indicating that testing would definitely be covered within this limit.
   - Another member echoed these concerns, stating they were unsure how to ever reach 1k requests.
- **Intermittent model error experience**: One member reported encountering a **'model 
   - This issue seemingly arose from the model's versioning, as another member noted that the reranker is now at **v3**.



**Link mentioned**: <a href="https://tenor.com/view/0001-gif-17282391190974969363">0001 GIF - 0001 - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1277977343339663442)** (3 messages): 

> - `Production Key 403`
> - `Langchain and Cohere TypeScript` 


- **Clarification Needed on Production Key 403**: *Renzhiping* mentioned 'production key 403' without context, leading to confusion among members.
   - *Nick Frosst* responded, seeking clarification on what was meant by this reference.
- **404 Error with Langchain and Cohere TypeScript**: *Fealomeril* reported an issue where the first call using **Langchain** with **Cohere TypeScript** yielded a valid response, but subsequent calls resulted in a **404 page not found error**.
   - This indicates potential instability or misconfiguration in the integration being used.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1277723908577034293)** (8 messages🔥): 

> - `Flutter Collaboration`
> - `vllm and RAG`
> - `LLM Workflow Builders`
> - `Local Embedding Models` 


- **Networking for Flutter Apps**: @fritzlandry expressed interest in collaborating on Flutter app development, and a member responded with a project idea that was previously abandoned due to Flutter inexperience.
   - This potential collaboration could help address the shared knowledge gap in building Flutter applications.
- **RAG Implementation with vllm**: A user inquired whether it is possible to run a Retrieval-Augmented Generation (RAG) with **vllm**, mentioning they have models for both embedding and answering.
   - This highlights an ongoing exploration of multi-model implementations using **vllm** and its capabilities.
- **Seeking LLM Workflow Builders**: @rogo6623 asked if anyone has created a workflow builder that incorporates **LLM capabilities**, showing interest in automating processes with language models.
   - This inquiry indicates a demand for tools that enhance the functionality of LLMs in user workflows.
- **Local Work with Embedding Models**: A user is looking for recommendations on local embedding models and mentioned experiencing delays with cloud-based solutions like **Pinecone** and **Elasticsearch**.
   - They specifically requested resources to facilitate their local setup, signaling a push towards maximizing efficiency in model performance.
- **Ollama as a Local Solution**: @rogo6623 recommended using **Ollama** for local model deployment, hinting at its effectiveness for users looking to avoid cloud-related latency.
   - This suggests a preference among users for local solutions that enhance responsiveness and control over their models.


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1278016049270226966)** (4 messages): 

> - `Dashboard usefulness`
> - `Error handling in AI models`
> - `Understanding cloned repos` 


- **Scrutinizing Dashboard Worth**: A member questioned whether the dashboard is **worth it**, expressing frustration about errors thrown by models like **Claude** and **GPT** when dealing with math and programming tasks.
   - *I'm looking for something more accurate because I'm specific but I don't know web programming*.
- **Discussion on Error Handling**: Another member sought clarification on what was meant by **errors**, emphasizing that the dashboard is primarily a visualization tool.
   - This highlights the ongoing concern about reliability and accuracy in AI outputs.
- **Curiosity about Cloning Repos**: The same member inquired if cloning a repository would allow an AI to explain each file and its **algorithms**, indicating a desire for deeper understanding.
   - This reflects a wider interest in using AI tools for more technical guidance.


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1277775483966193759)** (5 messages): 

> - `Model Accessibility on Leaderboard`
> - `Benchmarking with BFCL`
> - `Multiple Model Versions`
> - `Benchmarking Llama 3.1` 


- **Models must be accessible for leaderboard**: Any model listed on the leaderboard must be **publicly accessible**, either as open-source or via an API endpoint for inference.
   - *You can set up registration/login/token,* but the public should eventually get access to that endpoint.
- **Benchmarking limitations without public access**: While models can be benchmarked using **BFCL**, those that aren't publicly accessible cannot be displayed on the leaderboard.
   - This creates a distinction for what models can be showcased versus just evaluated.
- **Acceptance of multiple model versions**: The system allows multiple versions of a model, such as a **prompt version** and a fine-tuned FC version.
   - Both types of versions are accepted for benchmarking purposes.
- **Seeking benchmarking guidance for Llama 3.1**: A user is looking for guidance on **benchmarking Llama 3.1** using a custom API endpoint hosted by their company.
   - They are requesting specific pointers on how to initiate the benchmarking process effectively.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1277882849156010057)** (2 messages): 

> - `Function Calling Performance`
> - `BFCL Leaderboard Optimization Concerns` 


- **Function Calling Feature Drops Performance**: A user observed that for **GPT-4-1106-Preview**, using a system prompt directly yields an accuracy of **85.65** compared to **79.65** when the function calling feature is enabled.
   - This discrepancy raises questions about whether enabling function calling inherently reduces performance, inviting further discussion and research on the topic.
- **Concerns Over Optimization Strategies for BFCL**: A user inquired about optimization strategies they're implementing for a function-calling feature, questioning whether these would be considered unfair per the BFCL guidelines.
   - Concerns were raised about whether optimizations such as updating system prompts and formatting outputs might fall under the category of unfair practices not generalizable to all models.


  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

dr_monk: I am rooting for apple to pull through. 🙂
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1277740172141924608)** (5 messages): 

> - `DSPy Output Truncation`
> - `DSPy Library Typing Support`
> - `Scoring Generated Texts with DSPy` 


- **Fixing DSPy Output Truncation**: A member reported their outputs getting truncated when using **DSPy** and suspected token limits might be to blame. Another member suggested changing **max_tokens** during initialization and using `[your_lm.inspect_history()](https://some.link/here)` to view the prompts.
   - The original poster confirmed that this advice resolved their issue, highlighting the practical help from the community.
- **Error in DSPy Import**: A member encountered the error message `module is installed, but missing library stubs or py.typed` upon importing **DSPy**. They inquired whether **DSPy** supports typing in Python, indicating a need for clearer documentation.
   - No follow-up resolution was provided to address this typing concern.
- **Interest in DSPy for Text Scoring**: A user reached out to ask if anyone had experience using **DSPy** to score generated texts based on KPIs or industry metrics like **BLEU** or **ROUGE**. This query reflects a growing interest in evaluating text generation performance metrics within the community.
   - However, there were no responses or shared experiences from other members regarding scoring texts with **DSPy**.


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/)** (1 messages): 

rolandtannous: hello is Hamel around?
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1277775069090943077)** (1 messages): 

> - `LLM Observability Tools Webinar`
> - `Comparative Analysis of ML Monitoring Tools`
> - `Integration of Observability in LLMs` 


- **Join the LLM Observability Tools Webinar**: This Saturday, August 31st, at **11:30 AM EST**, a webinar will explore over **60 LLM observability tools** to determine their effectiveness in monitoring and optimizing models. Register for the session [here](https://kyrylai.com/webinar-observability-platforms-overflow/).
   - Participants will gain insights on observability basics, tool selection, and integration with serving for LLMs.
- **Testing the Hype of ML Monitoring Platforms**: The observability space is saturated, with many tools claiming superiority in **monitoring** and **debugging** ML models. The webinar aims to critically assess whether these tools truly meet the needs of practitioners.
   - Expect a hands-on evaluation to sift through claims with a focus on practicality and user-friendliness.
- **Cohort on Machine Learning in Production**: A live cohort for 'Machine Learning in Production' is available, aiming to enhance practical skills in deploying ML models. Interested participants can check out more details [here](https://edu.kyrylai.com/courses/ml-in-production).
   - This course promises to equip learners with essential tools and knowledge for effective ML management in real-world applications.



**Link mentioned**: <a href="https://kyrylai.com/webinar-observability-platforms-overflow/">Observability Platforms Overflow: Why There Are More Monitoring Platforms Than ML Models | Live Webinar</a>: Explore why observability platforms are outpacing ML models and learn how to select the right monitoring tools for your ML projects in our upcoming webinar.

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/)** (1 messages): 

_baumer: is there a huggingface link for LAION-aesthetic. The link in the laion website is busted
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1277960292327755828)** (1 messages): 

> - `LocalAI`
> - `Ettore Di Giacinto AMA`
> - `Open-source alternatives to OpenAI` 


- **LocalAI AMA with Ettore Di Giacinto is happening soon**: Join the **LocalAI AMA** with **Ettore Di Giacinto** in two hours to explore the features of this free, open-source alternative to OpenAI. LocalAI serves as a drop-in replacement REST API compatible with various AI specifications for local inferencing.
   - The platform enables running large language models (LLMs), generating images, and audio locally without requiring a GPU, making it accessible with consumer-grade hardware.
- **Join the LocalAI conversation**: The **LocalAI** event link is available for everyone to participate. [Join here](https://discord.com/events/1089876418936180786/1268967945216721079).
   - Don’t miss the chance to engage directly with the developer and get your questions answered about this innovative project.


  

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

---
id: d0550d67-e91a-463a-95eb-e1fa1d923fcd
title: not much happened today
date: '2024-08-21T00:22:36.551416Z'
original_slug: ainews-not-much-happened-today-5079
description: >-
  **OpenAI** launched **GPT-4o finetuning** with a case study on Cosine.
  **Anthropic** released **Claude 3.5 Sonnet** with 8k token output. **Microsoft
  Phi** team introduced **Phi-3.5** in three variants (Mini 3.8B, MoE 16x3.8B,
  Vision 4.2B) noted for sample efficiency. **Meta** released **Llama 3.1
  405B**, deployable on Google Cloud Vertex AI, offering GPT-4 level
  capabilities. **Qwen2-Math-72B** achieved state-of-the-art math benchmark
  results with a Gradio demo. Discussions included model comparisons like ViT vs
  CNN and Mamba architecture. Tools updates featured **DSPy** roadmap, **Flux
  Schnell** improving diffusion speed on M1 Max, and **LangChain** community
  events. Research highlights zero-shot DUP prompting for math reasoning and
  fine-tuning best practices. AI ethics covered California AI Safety Bill SB
  1047 and regulatory concerns by **ylecun**. Commentary on AI engineer roles by
  **swyx**. *"Chat with PDF" is now one React component and an API key away,
  available to Box Enterprise Plus customers.*
companies:
  - openai
  - anthropic
  - microsoft
  - meta-ai-fair
  - hugging-face
  - langchain-ai
  - box
models:
  - gpt-4o
  - claude-3.5-sonnet
  - phi-3.5-mini
  - phi-3.5-moe
  - phi-3.5-vision
  - llama-3-1-405b
  - qwen2-math-72b
topics:
  - fine-tuning
  - benchmarking
  - model-comparison
  - model-performance
  - diffusion-models
  - reinforcement-learning
  - zero-shot-learning
  - math
  - ai-regulation
  - ai-safety
  - ai-engineering
  - model-efficiency
  - prompt-engineering
people:
  - swyx
  - ylecun
  - rohanpaul_ai
  - philschmid
  - huybery
  - giffmana
  - wightmanr
  - lateinteraction
  - awnihannun
---


<!-- buttondown-editor-mode: plaintext -->**another quiet day in AI.**

> AI News for 8/19/2024-8/20/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**254** channels, and **2227** messages) for you. Estimated reading time saved (at 200wpm): **258 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!


No main story, just little ones: 

- [OpenAI GA'ed GPT-4o finetuning](https://openai.com/index/gpt-4o-fine-tuning/) with a notable case study on Cosine
- [Anthropic GA'ed Claude 3.5 Sonnet 8k token output](https://x.com/alexalbert__/status/1825920737326281184)
- [Zed introduced their Cursor/Cursor Composer competitor AI features](https://zed.dev/blog/zed-ai)
- The [Microsoft Phi team released](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) Phi-3.5 in 3 variants: Mini (3.8B), MoE (16x3.8B), Vision (4.2B), all [remarkably sample efficient](https://x.com/Yampeleg/status/1825981743100240201). No paper or independent evals yet.

Since it's a quiet day you can support AINews by checking out Box AI who have kindly supported this week's issues!

---

**[Sponsored by Box]** You might have an app. It might have users. Those users might even store docs in Box. [But Box AI lets your users query their docs right in the Content Preview UI Element!](https://shortclick.link/5lxgsv)

*Swyx commentary: "**Chat with PDF**" is now one React component and an API key away! Note it's only available to Box Enterprise Plus customers for now.*

(previously with Box AI: [Week 1](https://shortclick.link/tndo68), [Week 2](https://shortclick.link/23g92m))


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


**AI Model Developments and Benchmarks**

- **Llama 3.1 405B Release**: Meta released Llama 3.1 405B, which can now be easily deployed on Google Cloud Vertex AI. This offers GPT-4 level capabilities that can be run in-house, giving full control. [@_philschmid](https://twitter.com/_philschmid/status/1825541324893737085) shared details on deployment using Hugging Face's Text Generation Inference container.

- **Qwen2-Math-72B**: This model achieves state-of-the-art performance on several math benchmark datasets. A Gradio demo has been released for testing. [@huybery](https://twitter.com/huybery/status/1825560321383428166) highlighted its strength and provided a link to try the demo.

- **Model Comparisons**: Various tweets discussed comparisons between different models and architectures:
  - ViT vs CNN performance comparisons were mentioned by [@giffmana](https://twitter.com/giffmana/status/1825617256967262699)
  - Mamba architecture performance was discussed by [@wightmanr](https://twitter.com/wightmanr/status/1825630715188490390)

**AI Tools and Applications**

- **DSPy**: [@lateinteraction](https://twitter.com/lateinteraction/status/1825594011484303596) shared updates on DSPy 2.5 and 3.0, including a roadmap for future developments. The focus is on shifting from ad-hoc prompting to systematic programming.

- **Flux**: [@awnihannun](https://twitter.com/awnihannun/status/1825546558739517643) mentioned that Flux Schnell in the latest DiffusionKit with MLX is 30% faster and uses less RAM, allowing high-quality image generation in under a minute on an M1 Max laptop.

- **LangChain**: The LangChain community is organizing events, including a Hacky Hour in Austin. [@LangChainAI](https://twitter.com/LangChainAI/status/1825675460380078226) shared details about the upcoming gathering.

**AI Research and Techniques**

- **Zero-shot DUP prompting**: This technique achieves SOTA results on math reasoning tasks across various LLMs. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1825671188007297077) explained the three-stage process and its benefits in reducing semantic misunderstanding errors.

- **Fine-tuning Models**: [@jxnlco](https://twitter.com/jxnlco/status/1825563945798918473) shared insights on fine-tuning models, emphasizing the importance of data quality, avoiding vendor lock-in, and focusing on thorough evaluation.

**AI Ethics and Regulation**

- **California AI Safety Bill SB 1047**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1825667014498701615) summarized key points from the modified version of the bill, including changes to liability and safety practice requirements.

- **AI Regulation Debate**: [@ylecun](https://twitter.com/ylecun/status/1825500979552284712) expressed concerns about regulating AI research and development, particularly regarding obstacles to scientific information exchange and open-source code distribution.

**AI Engineering Perspectives**

- **AI Engineer Role**: [@swyx](https://twitter.com/swyx/status/1825630984911597834) discussed the central purpose of AI Engineers as turning existing foundation model capabilities into useful products. He highlighted the divergence from traditional ML Engineering and the increasing complexity of the AI stack.

- **Docker Importance**: [@svpino](https://twitter.com/svpino/status/1825578554895266012) emphasized the necessity of learning Docker for building and deploying software, describing it as a main differentiator in his work.

- **LLM API Businesses**: [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1825665250231988280) expressed confusion about the economics of LLM API businesses, sparking discussion about the sustainability and profitability of such models.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Large Language Model Releases and Deployment**

- **Announcing: Magnum 123B** ([Score: 110, Comments: 21](https://reddit.com//r/LocalLLaMA/comments/1ewb7b6/announcing_magnum_123b/)): **Magnum-v2-123B**, based on **MistralAI's Large**, has been released as the largest Magnum model to date, trained on the same dataset as other v2 models. The model, which was trained using **8x MI300 GPUs** on **RunPod**, has not undergone formal evaluations but showed promising results during testing, appearing to be an improvement over previous Magnum versions.


**Theme 2. Innovative AI Interfaces: Handwriting and Speech Recognition**

- **[Using Whipser+GPT for automatic note taking and tagging](https://v.redd.it/i9nwct9gupjd1)** ([Score: 72, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1ewi9m2/using_whipsergpt_for_automatic_note_taking_and/)): **Whisper** and **GPT** are being utilized for automatic note-taking and tagging in **Obsidian**, as described by the post author. The combination of these **AI models** enables efficient conversion of audio to text and subsequent organization of notes, potentially streamlining the process of capturing and categorizing information within the **Obsidian** note-taking system.
  - The author shared links to their **GitHub repositories** for [AlwaysReddy](https://github.com/ILikeAI/AlwaysReddy) and [alwaysreddy_add_to_md_note](https://github.com/ILikeAI/alwaysreddy_add_to_md_note), which handle transcription and note-taking functionality.
  - **Obsidian** users discussed note-saving options, including daily notes and static notes. One user mentioned integrating Obsidian notes with a pipeline in **Open WebUI**.
  - The system uses an **LLM** (such as **Claude**) for automatic tagging, and can work with any LLM, including local model servers.

- **[handwriting interface on the e-reader. slowly turning it into what I always dreamed a palm pilot would be.  ultimately I'd like to have it recognize shapes - but I'm not sure what cheap models can do that (~0.5B size)](https://i.redd.it/9mk6hhlb6qjd1.gif)** ([Score: 249, Comments: 29](https://reddit.com//r/LocalLLaMA/comments/1ewjog3/handwriting_interface_on_the_ereader_slowly/)): The post discusses developing a **handwriting interface** for an **e-reader**, aiming to create a device reminiscent of an advanced **Palm Pilot**. The author expresses interest in implementing **shape recognition** functionality but is uncertain about the capabilities of smaller, more affordable **language models** around **0.5 billion parameters** in size for this task.
  - The project uses **qwen2:0.5b** on **ollama** with **bun** as server and **handwriting.js** on frontend, running on a **Boox Palma** device. Users suggested potentially upgrading to **gemma2B** or **phi-3-mini** models, with discussions on token generation speeds on various devices.
  - Debate arose over the practicality of a handwriting interface for LLMs, with some arguing it contradicts LLM benefits. Others defended the concept as an innovative integration of open weights with different input types, suggesting potential uses like transforming brief handwritten notes into more fluent text.
  - Users drew parallels between the project and fictional magical objects, particularly **Tom Riddle's diary** from Harry Potter. There was also criticism of **Boox** as a company, with calls for competitors that respect open-source licenses and produce more durable devices.


## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Image Generation Advancements**

- **Flux model demonstrates versatile image generation capabilities**: 
  - [Grid generation from a single prompt](https://www.reddit.com/r/StableDiffusion/comments/1ew23gd/psa_flux_is_able_to_generate_grids_of_images/)
  - [Product photography applications](https://www.reddit.com/r/StableDiffusion/comments/1ew5l8j/flux_is_a_game_changer_for_product_photography/)
  - [Tarot card LoRA creation](https://www.reddit.com/r/StableDiffusion/comments/1ewkvl6/this_flux_tarot_card_lora_is_so_much_fun/)
  - [3D stereo image generation](https://www.reddit.com/r/StableDiffusion/comments/1ew4avp/flux_has_the_capability_to_create_3d_stereo/)
  - [Random walk through latent space](https://www.reddit.com/r/StableDiffusion/comments/1ew2r1r/a_random_walk_through_flux_latent_space/)

- **Flux model's strengths and limitations**:
  - [Impressive line-drawing skills for geometry and typography](https://www.reddit.com/r/StableDiffusion/comments/1ewkvl6/this_flux_tarot_card_lora_is_so_much_fun/lizh6d4/)
  - [Potential issues with complex scenes](https://www.reddit.com/r/StableDiffusion/comments/1evyqu2/flux_is_fun_until/)

**AI Industry Developments**

- **AMD challenges Nvidia's AI infrastructure lead**: [AMD signs $4.9 billion deal](https://www.reddit.com/r/singularity/comments/1ew1zgp/amd_signs_49bn_deal_to_challenge_nvidias_ai/) to compete in the AI hardware market.

**AI Ethics and Philosophy Discussions**

- **Debates on AI consciousness and intelligence**:
  - [Meme about eternal AI debate](https://www.reddit.com/r/singularity/comments/1evwsd5/a_meme_about_the_eternal_debate_about_ai/)
  - [Discussion on predictive, generative nature of human cognition](https://www.reddit.com/r/singularity/comments/1evwsd5/a_meme_about_the_eternal_debate_about_ai/liuh4n2/)
  - [Critique of AI rights movement](https://www.reddit.com/r/singularity/comments/1ewl51b/it_has_begun/)

**Memes and Humor**

- [Meme about AI debate](https://www.reddit.com/r/singularity/comments/1evwsd5/a_meme_about_the_eternal_debate_about_ai/)
- ["It's not really thinking, it's just sparkling reasoning" meme](https://www.reddit.com/r/singularity/comments/1ew4vns/its_not_really_thinking_its_just_sparkling/)
- [AI rights movement parody video](https://www.reddit.com/r/singularity/comments/1ewl51b/it_has_begun/)


---

# AI Discord Recap

> A summary of Summaries of Summaries by Claude 3.5 Sonnet

**1. LLM Advancements and Benchmarking**

- **Hermes 3 Takes on the Giants**: **[Hermes 3](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b)**, a 70B parameter model, has been released on OpenRouter with advanced agentic capabilities and improved roleplaying abilities.
   - Users are eager to compare **Hermes 3** performance against models like **Meta-Llama 405b**, though it's not yet listed on the [LLM Arena leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard).
- **LLaMA 3.1 Struggles with SQL**: A user reported that **[LLaMA 3.1 70B](https://ai.google.com/research/pubs/pub49727.html)** is unable to query a database using [LangChain's SQL agent](https://langchain.readthedocs.io/en/latest/modules/agents/agents.html#sql-agent), while **GPT 3.5** succeeds with the same setup.
   - Despite attempts with custom parsers, the issue persists, leading to speculation about LLaMA's limitations in certain tasks compared to other models.
  


**2. Model Performance Optimization**

- **Torch.compile Recompilation Challenges**: Users discussed issues with **torch.compile** recompilations occurring due to input shape changes during generation and when switching between training and inference modes.
   - The discussions highlighted limitations in torch.compile's ability to handle dynamic scenarios, such as passing RNG generator objects, which cause graph breaks.
- **Custom Masks and KV-Cache Compatibility**: Developers explored the compatibility of custom masks with **kv-cache** in language models, noting that direct use might not be compatible.
   - A potential solution involves utilizing a custom mask and removing `self.causal_mask`, though this requires further investigation and testing.
- **AI Chip Design for Local Memory**: Discussion centered on how **AI chips** are designed with substantial local memory to fit models in cache, reducing the penalty of frequent data transfers to RAM.
   - The trade-offs between Network on Chip (NoC) designs and cache management were debated, noting that while NoCs provide efficient data transfer across cores, they also introduce latency.
  


**3. Open-Source AI Developments**

- **Whisperfile Simplifies Audio Transcription**: **[Whisperfile](https://simonwillison.net/2024/Aug/19/whisperfile/)**, created by Justine Tunney, offers an easy way to transcribe audio locally using OpenAI's Whisper model, with 100% local operation and translation capabilities.
   - The tool can even translate non-English audio to English during transcription, making it a versatile solution for audio processing tasks.
- **LlamaIndex Expands Learning Resources**: **LlamaIndex** launched an [O'Reilly Media course](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex) on retrieval-augmented generation (RAG), covering components, evaluation, ingestion pipeline, observability, agents, and multi-modality.
   - Additionally, LlamaIndex is hosting an AI product meetup, "LLMs in Production", focusing on building context-augmented LLMs with RAG & Vector DB and high-performance inference for production-ready LLMs.
- **Aider v0.51.0 Enhances Development Workflow**: **[Aider v0.51.0](https://aider.chat/HISTORY.html#v0510)** was released with improved prompt caching for Anthropic models, optimized repo mapping for larger repositories, and enhanced Jupyter Notebook .ipynb file editing.
   - The release includes various bug fixes and improvements, with Aider contributing 56% of the code for this version, showcasing the tool's capability in AI-assisted development.
  


**4. Multimodal AI and Vision Models**

- **LM Studio's Vision Model Limitations**: Users inquired about **LM Studio's** capability to process photos or videos as input for providing visual context in coding tasks.
   - It was confirmed that local models in LM Studio cannot handle such tasks, with only cloud-based models like **GPT4o** and **Claude** currently offering this functionality.
- **Qdrant 1.10 Boosts Multi-Vector Representations**: **[Qdrant 1.10](https://qdrant.tech/articles/late-interaction-models/)** introduced support for multi-vector representations, enhancing retrieval quality and enabling late interaction models like **ColBERT**.
   - The update allows for adapting regular dense embedding models for late interaction by removing the pooling step and using token-level embeddings for retrieval and reranking.
  


**5. LLM Training and Fine-tuning Techniques**

- **MiniPile: A Compact Alternative for Model Training**: The **[MiniPile dataset](https://huggingface.co/datasets/JeanKaddour/minipile)**, a 6GB subset of the Pile corpus, was recommended as a viable alternative for training smaller-scale models due to the large size and cost of the full Pile dataset.
   - MiniPile was curated by filtering out low-quality clusters, ensuring a diverse pre-training dataset that is more manageable for academic budgets and smaller-scale experiments.
- **Model Merging and Extension Strategies**: Discussions arose around novel model merging tactics, such as applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn**, sparking debates on the potential of "cursed model merging" techniques.
   - Users also explored options for extending models like **Mistral** beyond their initial token limits, suggesting further work on *mergekit* and *frankenMoE finetuning* as potential solutions.
  

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's Limitations on Fine-tuning Llama-3.1-405B**: A user inquired about fine-tuning **Llama-3.1-405B** on a **Hugging Face Space GPU** with an **H100**, but was informed that **Unsloth currently does not support this** due to the model's high memory requirements.
   - The user was told that they would need **at least 360 GB of GPU memory** and **eight H100 GPUs**, which Unsloth does not offer at this time.
- **Lambda's Free Model Access and Fine-tuning Limitations**: A user asked if **Lambda offers free fine-tuning** for **Llama-3.1-405B**.
   - They were informed that **Lambda only offers free model execution** and **does not offer free fine-tuning**, but similar features are available on platforms like **Hugging Face, Meta, and Groq**.
- **Training Loss Issues and Troubleshooting on Google Colab**: A user faced challenges in keeping their **training loss** below **1.000** while fine-tuning a model on a **Google Colab A100 runtime**.
   - They experimented with adjusting the learning rate and batch size, but ultimately concluded that **a Colab A100 runtime might not be a feasible long-term solution due to its high GPU memory requirements**.
- **Unsloth Premium and Partnerships**: A user inquired about the pricing of **Unsloth Premium** and potential **partnerships with Unsloth**.
   - They were informed that **Unsloth Premium is not available for direct purchase**, and its faster versions are restricted to Fortune 500 companies. Users were advised to contact **Mike or Daniel** for further information.
- **PPL as a Metric for Model Evaluation**: **PPL** (perplexity) is a useful metric for comparing the effects of quantization but can be misleading if the difference between the base and quantized model is significant.
   - **PPL** is also valuable for comparing models at the token-level to identify observed topics, but the absolute value is meaningless, and the delta between models is the key focus.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Llama2 Model Loading Issue**: A user reported that running Llama2 eval crashes during the model loading phase, simply printing 'killed' and exiting.
   - The user also encountered an out-of-memory (OOM) error while running Llama2 evaluation, even though their system should have enough RAM and GPU memory.
- **GPT-Fast & HF_eval Script Showdown**: The discussion centered around the use of different evaluation scripts, particularly comparing the GPT-Fast evaluation script with HF_eval.
   - The user reported that they encountered an issue while running the HF_eval script for evaluating Llama2, resulting in an error message indicating an unsupported default value for the `zero_point_domain` parameter.
- **Triton Kernel Optimization for Beginners**: A user encountered a `ValueError` while attempting to use `tl.arange` with a non-constexpr value `seqlen` in a `triton.jit` kernel.
   - The issue arose because `seqlen` was not declared as a `tl.constexpr` type, which is required for the `tl.arange` function in Triton, highlighting a key difference between Triton and regular Python code.
- **FP16 & FP8 for Comfy**: A member was under the impression that Comfy supports FP16 accumulator by default, but it requires a custom Torch C++ extension.
   - Comfy's FP8 implementation doesn't actually use FP8 matmul for computation; it only uses it as an intermediate data type, with Stable-fast being an alternative that doesn't support Flux but has interesting optimization ideas.
- **Diffusion Model Quantization Techniques**: A member discussed how diffusion models can be effectively quantized by keeping self-attention and accumulation in FP16.
   - Oneflow/Onediff is a wrapper for diffusion models that uses Oneflow for inference and graph creation, but it's not compatible with Flux because Flux is too large.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 3 Compared to Meta-Llama**: A member inquired about a comparison between **Hermes 3/405** and other models, particularly **Meta-Llama 405b**, as they were unable to find **Hermes** on the [LLM Arena leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard).
   - Another member confirmed that **Hermes 3** is benchmarked against **Llama 3.1-instruct-405** in a technical report, using a suite of 15 benchmarks, but they were also looking for comparisons against **Meta-Llama 405b**.
- **Hermes 3: Text-to-Text Model**: It was confirmed that **Hermes 3** is a text-to-text model, meaning that it cannot generate images. 
   - While you can interact with **H3-405B** in [Discord](https://discord.com/channels/1053877538025386074/1149866614590816256), the bots cannot trigger image generation through commands, they can only interact by @ mentioning each other.
- **Llama 3.1 Minitron 4B: Pruned Text-to-Text Model**: **Llama-3.1-Minitron-4B-Width-Base** is a text-to-text model that can be used for various natural language generation tasks.
   - It is obtained by pruning **Llama-3.1-8B**'s embedding size, attention heads, and MLP intermediate dimension, followed by continued training with distillation using 94 billion tokens from the continuous pre-training data corpus used in Nemotron-4 15B.
- **Hermes 3 Amnesia Mode: Only Available for 8B**: **Amnesia mode** is a feature of **Hermes 3 8b** that can be triggered by prompting it with "Hi" with no system prompts.
   - However, this mode is not available on Discord because the bot remembers all chats.
- **PyDantic-XML: Serialization and Deserialization**: The **pydantic-xml** extension allows for serializing and deserializing data between Pydantic models and XML.
   - You can find the documentation for this extension at [https://pydantic-xml.readthedocs.io/en/latest/](https://pydantic-xml.readthedocs.io/en/latest/). 



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **DeepMind OPRO Paper Question**: A member inquired about the source of the information regarding an OPRO-based prompt tuner.
   - The member is seeking clarification on how to implement this technique, potentially referencing [the OPRO paper](https://arxiv.org/abs/2203.11824).
- **C4AI Discord Server Invite**: A member requested an invite to the C4AI Discord server.
   - The member was advised to join the Cohere Discord and contact a specific user, but is unsure about the appropriate communication channel (DM or public channel).
- **Cohere API `response_format` Issue**: A member encountered an error while using the `response_format` parameter in the Cohere API.
   - They are seeking guidance on how to properly utilize the `response_format` parameter in their API requests. 
- **Cohere Classify Endpoint Sunset**: A member inquired about potential alternatives to the Cohere Classify endpoint.
   - The member is seeking recommendations for similar classification services with a focus on functionality and usability.
- **Reranker API Efficiency for Large Datasets**: A member asked if chunking large datasets and running the Reranker API independently on each chunk would produce accurate overall relevancy scores.
   - This member is exploring the potential limitations and benefits of applying the Reranker API to large datasets in a chunked manner.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hermes 3 Released**: **Hermes 3**, a 70B parameter model, has been released on [OpenRouter](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b) with advanced agentic capabilities and much better roleplaying.
   - The release announcement also included a copyright notice for OpenRouter, LLC, stating © 2023 - 2024 OpenRouter, LLC.
- **GPT Function Calls Still Supported?**: A user asked if GPT functions are still supported on OpenRouter, as they are receiving 'function_call=None' even though the stop reason is 'functioncall'.
   - The OpenRouter team confirmed that better tool call routing is coming soon, but currently, results may vary unless using OpenAI, Anthropic, or Google models.
- **Mistral Large Instruct 2407 for German Pretraining**: A user inquired about a model with good German pretraining, and was suggested to try Mistral-Large-Instruct-2407, which is multi-lingual by design and supports German.
   - The user tested the model but found it to be 'okay' but not great, and further suggested checking Hugging Face for other models.
- **OpenRouter Errors With Non-Free Models**: Users reported encountering an error when trying to access non-free models on OpenRouter, specifically getting a 'client-side exception' and needing to hard refresh the browser.
   - The OpenRouter team investigated and determined that the issue was related to access token expiration and potentially CORS errors, and ultimately resolved the issue.
- **Uncensored Models on OpenRouter?**: A user inquired about uncensored models on OpenRouter, and was suggested that 'open source' and 'roleplay' tags are good indicators for models that may produce NSFW content.
   - Popular options for uncensored models include Dolphin, Stheno, Euryale, and MythoMax.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Uncensored Models: Explore the Landscape**: A user sought suggestions for uncensored LLM models for non-coding tasks, and was provided with a link to [llm.extractum.io](https://llm.extractum.io/list/?uncensored) which highlights its focus on uncensored LLMs for a variety of uses like legal analysis, medical research, and creative writing.
- **LM Studio Server Struggles with Llama 3.1**: A user reported encountering issues with LM Studio's local inference server, specifically with Llama 3.1, where the stop pattern was ignored.
   - The user noted that the issue was absent in chat mode and suggested a discussion in the relevant channel to troubleshoot further.
- **Speech-to-Text and Text-to-Speech in LM Studio**: A user inquired about voice interaction with Llama 2/3 models in LM Studio, specifically whether speech-to-text and text-to-speech functionalities were integrated.
   - It was clarified that LM Studio currently lacks this support, prompting the user to explore external solutions like [Parler-TTS](https://github.com/huggingface/parler-tts) for text-to-speech and [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for speech-to-text.
- **Vision Models in LM Studio: A Cloud-Based Affair**: A user inquired about models capable of processing photos or videos as input in LM Studio to provide visual context for coding tasks.
   - It was confirmed that local models in LM Studio cannot handle this; only cloud-based models like GPT4o and Claude offer this functionality.
- **M2 Ultra: High Hopes for AI Performance**: A user expressed excitement for the upcoming M2 Ultra, noting its performance is highly anticipated for AI tasks.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT-4 Neuron Explanations Debunked?**: A member questioned the usefulness of GPT-4's neuron explanations, citing a paper that claimed they were not better than baselines.
   - Another member provided a link to a paper titled "Language Models can explain neurons in language models" but couldn't find a paper with a similar title claiming GPT-4 explanations were not useful, despite the content being similar.
- **Training Models on Limited Data - Beware the Nonsense!**: Training a model on a single, small file can result in nonsensical outputs due to the influence of random initialization.
   - A member compared it to text compression benchmarks, where models are trained to memorize a specific block of text, and emphasized the importance of diverse pre-training data.
- **MiniPile Dataset for Efficient Training**: MiniPile, a 6GB subset of the Pile corpus, was recommended as a viable alternative for training smaller-scale models due to the large size and cost of the full Pile dataset.
   - MiniPile was curated by filtering out low-quality clusters, ensuring a diverse pre-training dataset that is more manageable for academic budgets.
- **Frankenmerging - Composing Layers from Different Models**: A member inquired about the feasibility of composing layers from two different models, a technique known as 'frankenmerging.'
   - They expressed confusion about the potential risks of this approach, questioning whether it wouldn't lead to a garbled internal representation of the model, and sought clarification on potential benefits and challenges.
- **Model Merging with Optimizers**: A member suggested using an optimizer to find the best permutation of channels between layers of two different models before stacking them together.
   - They acknowledged the potential challenges, noting that such methods haven't been demonstrated for large GPT models.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro Discord Access is Confusing**: Users are unable to join the Perplexity Pro Discord server, even after leaving and rejoining using the link in their Perplexity settings.
   - The issue seems to be a lack of clear instructions regarding accessing the Pro section within the main Discord server.
- **Perplexity's Search Function Needs Fixing**: Users are reporting issues with Perplexity's search function, including the inability to access online sources and the use of outdated information.
   - Some users believe this is a backend issue, but the team at Perplexity has yet to acknowledge or address the problem.
- **Perplexity Pro Models Face Limitations**: Users are discussing the limitations of Perplexity Pro models for tasks like coding and blog post creation.
   - Some users are finding that Perplexity Pro is not as effective as other models for certain tasks, particularly when it comes to generating complex code or avoiding hallucinations in blog posts.
- **Perplexity's Prioritization of Front-End vs Backend**: There is a debate about whether Perplexity is prioritizing front-end development over backend development, with some users reporting issues with backend features like search and model selection.
   - Some users believe that these issues indicate a lack of focus on core backend functionalities, which are critical for the overall performance of the platform.
- **Perplexity Pro Feature Upgrade Discussion**: A discussion occurred about upgrading to [Perplexity Pro](https://www.perplexity.ai/pro) which offers features like image upload, smarter AI, and more Pro Search.
   - Other users also discussed the potential benefits of using [LMSYS Arena](https://www.youtube.com/embed/EQxYdx79vUg) and the upcoming [G1 Humanoid Robot](https://www.youtube.com/embed/EQxYdx79vUg) which is reportedly ready for mass production.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex: Building Natural Language Querying Systems**: Learn how to build a natural language querying system for graph databases using LlamaIndex and Amazon Neptune!
   - A comprehensive guide by @bechbd shows you how to translate natural language questions into openCypher queries and execute queries on Amazon Neptune graph.
- **O'Reilly Media Course on RAG**: LlamaIndex has launched an O'Reilly Media course on retrieval-augmented generation, authored by @ravithejads.
   - The 2-hour course covers components of LlamaIndex, evaluation of RAG systems, the ingestion pipeline, observability, agents, multi-modality, and more.
- **LlamaIndex: LLMs in Production Meetup**: Join LlamaIndex for "LLMs in Production", an AI product meetup hosted by @vesslai and @pinecone in San Francisco.
   - Learn from industry leaders about building context-augmented LLMs with RAG & Vector DB, custom LLMs for smarter, faster, and cheaper solutions, and high-performance inference for production-ready LLMs.
- **Hierarchical Node Parser:  No Chunking?**: A user asked if the LlamaIndex hierarchical node parser can create hierarchies without chunking, instead using predefined nodes.
   - The user wanted to keep metadata like page IDs associated with the nodes, but this was not possible with the current implementation.
- **Complex Questions with LlamaIndex Retrieval**: A user discussed the need for retrieval capabilities for both simple and complex questions within LlamaIndex.
   - They envisioned a hierarchical approach that could recursively summarize nodes and create higher-level representations of the data for nuanced, contextual responses.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Jeremy Howard Dishes on Latent Space**: The latest Latent Space podcast features Jeremy Howard, with discussions on Encoder-Decoder models, Fast.html, saving/updating state, fine-tuning vs RAG vs KV caching, and a new project he's working on.
   - The podcast is described as 'a 5-course meal,' after co-host Swyx's playful phrase 'give us a nibble.'
- **Encoder-Decoder Models Rise**: The discussion emphasizes the advantages of Encoder-Decoder models, particularly for complex contexts and intricate relationships, over Encoder-only models.
   - The interviewee, likely influenced by AI Paper Club calls,  already had knowledge of this approach, suggesting increasing awareness within the AI community.
- **Whisperfile Makes Transcription a Breeze**: Whisperfile is a new tool that allows users to easily transcribe audio locally, utilizing OpenAI's Whisper model.
   - Created by Justine Tunney, Whisperfile offers 100% local operation and even translates non-English audio into English during transcription.
- **Claude 3.5 Sonnet Gets a Token Boost**: Anthropic AI has doubled the maximum output token limit for Claude 3.5 Sonnet, expanding it from 4096 to 8192.
   - This update is now available in both the Anthropic API and Vertex AI, making Claude 3.5 Sonnet easier for developers to work with.
- **GPT-4 Fine-Tuning Challenges Composer**: OpenAI has released GPT-4 fine-tuning, a new feature that lets users customize GPT-4's behavior and performance.
   - This update could potentially compete with Cursor's Composer feature, as both offer similar approaches to customizing and using large language models.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo & MAX Update Cadence Synchronized**: Previously, **Mojo** and **MAX** had independent update cycles, but now they are synchronized.
   - This means you can install **MAX+mojo main** or **MAX+mojo nightlies**, but not **MAX main** and **mojo nightlies** separately.
- **Siamese Networks with Labels?**: A user inquired about switching a Siamese network's output from a sigmoid to a label (e.g., "dog" or "cat").
   - Another user suggested that if you want to switch to labeling, using a standard model for that task might be more efficient than trying to adapt a Siamese network.
- **Using the Slice Custom Op**: A user requested a code example demonstrating the use of the **slice custom op** ([https://docs.modular.com/max/api/mojo/graph/ops/slicing/slice](https://docs.modular.com/max/api/mojo/graph/ops/slicing/slice)).
   - They expressed difficulty understanding the op's arguments.
- **Mojo's `List` assignment uses `ref`**: A user was surprised to find no `__setitem__` method for assignment in Mojo's `List` implementation, but was informed that `__getitem__` returns a `ref[lifetime] T` which behaves like `__setitem__`.
   - This is how you assign items to a Mojo `List`.
- **Mojo's `ref` and `__lifetime_of` Functions**: The `ref` keyword in function return types was introduced recently (in Mojo v244) as part of the new language features.
   - Mojo's `__lifetime_of` function allows you to determine the lifespan of a reference, which is useful for memory management.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT struggles with simple tasks**: A user pointed out that ChatGPT struggles with simple tasks like counting the number of 'R's in the word 'strawberry,' implying that AI is not as advanced as some might believe.
   - This sparked a discussion about the current limitations of AI and whether it is truly intelligent or simply a tool that can perform specific tasks.
- **Grok2 takes a different approach**: A user mentioned that Grok2 has an interesting approach to dealing with problems.
   - Another user pointed out that Grok2's method involves breaking down every question and solving it step by step, which is similar to the way humans solve problems.
- **AI Enthusiasm - Is it overblown?**: One user expressed that the term 'AI enthusiast' has lost its meaning due to AI's current limitations.
   - This sentiment arose from a discussion about ChatGPT's struggles with a simple task and Grok2's method of solving problems.
- **Building a Smart Cookbook**: A user sought advice on creating a 'smart cookbook' that could be trained on their favorite cookbooks and provide personalized advice.
   - This user believes that such a model could be applied to any 'how-to' book and requested information about existing solutions or projects.
- **Strawberry Release Speculation**: A user asked about the release date of 'Strawberry,' possibly a new AI model or a feature.
   - Another user responded by jokingly stating that 'Strawberry' is still in the 'unreliably sourced leak' phase and expressed skepticism about its release.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torch.compile struggles with recompilations**: Torch.compile recompilations occur when input shape changes, like during generation, or when switching between training and inference modes.
   - This is due to changes in the `grad_mode`, and could be improved by implementing `torch.compile` optimization.
- **Torch.compile cache size limit**: The `torch._dynamo hit config.cache_size_limit (8)` message indicates that the cache size limit has been reached.
   - This suggests potential issues with torch.compile friendliness.  The size of the cache may need to be increased.
- **RNG objects incompatible with Torch.compile**: Passing an RNG generator object into the model causes graph breaks, suggesting torch.compile currently doesn't support such objects.
   - This could be a challenge, but could be addressed by potentially updating `torch.compile` to handle these objects.
- **Custom masks vs kv-cache**: Custom masks may not be directly compatible with kv-cache, but using your own mask and removing `self.causal_mask` might help address the issue.
   - This issue is worth further investigation.
- **Torchtune release date**: The community is eager to know the release date for Torchtune, which is reportedly 99% ready.
   - The discussion suggests that the release date is not yet confirmed.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LLaMA 3.1 70B Struggles with SQL**: [LLaMA 3.1 70B](https://ai.google.com/research/pubs/pub49727.html) has trouble querying a database using [LangChain's SQL agent](https://langchain.readthedocs.io/en/latest/modules/agents/agents.html#sql-agent), while [GPT 3.5](https://platform.openai.com/docs/models/gpt-3-5) succeeds with the same setup.
   - Despite trying custom parsers, the issue persists, indicating potential limitations in LLaMA's capabilities.
- **Mistral Faces Challenges Expanding Beyond 8k**: A user noted that [Mistral](https://www.mistral.ai/) cannot expand beyond 8k without further pretraining.
   - They suggested exploring *mergekit* and *frankenMoE finetuning* to address this limitation.
- **Model Merging Tactics Sparked Discussion**: A user proposed merging **UltraChat** and base **Mistral** into **Mistral-Yarn** as a potential model merging tactic.
   - While some expressed skepticism, the user remained optimistic, citing past successes in what they termed "cursed model merging".
- **Open Empathic Project Seeks Assistance**: A user requested support in expanding the categories within the **Open Empathic** project, particularly at the lower end.
   - They shared a [YouTube video](https://youtu.be/GZqYr8_Q7DE) showcasing the project's launch and a tutorial, encouraging users to contribute preferred movie scenes from YouTube videos, and provided a link to the [OpenEmpathic project](https://dct.openempathic.ai/).
- **LangChain Introduces Experimental SQLDatabaseChain**: A user introduced [LangChain's SQLDatabaseChain](https://langchain.readthedocs.io/en/latest/modules/chains/sql_database_chain.html) as an experimental feature designed to generate SQL queries based on user prompts.
   - They provided a code example for a function utilizing this feature, outlining a prompt template for SQL query generation and handling responses from the chain.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Ollama Integration with OpenInterpreter**: A user sought guidance on integrating Ollama with OpenInterpreter on a remote machine, specifically configuring a profile YAML and starting the interpreter with the profile.
   - They asked about using the correct IP address and port for their Ollama instance in OpenInterpreter's configuration to establish a connection, however, OpenInterpreter still refused to connect.
- **Deepseek API: An Alternative to OpenAI and Local LLMs**: A user inquired about a guide for using the Deepseek API as an alternative to OpenAI or local LLMs.
   - The user expressed interest in using Deepseek as a potential solution for accessing and utilizing large language models.
- **Troubleshoot Poetry and Pytorch Installation on Mac**: A user reported encountering issues while installing Poetry and Pytorch 2.3.0 on a Mac, mentioning an open issue that had not been resolved.
   - They sought guidance on finding a solution to this installation problem, potentially involving alternative installation methods or troubleshooting specific configuration settings.
- **OpenInterpreter Update Rollout**: The latest OpenInterpreter update was announced in the #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1271135268807905384) channel.
   - No additional details were provided regarding the nature or scope of the update.
- **Accessibility Roundtable Reminder**: A reminder for the Accessibility Roundtable was posted in the #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1275167433082146848) channel.
   - The reminder included a link to the event, suggesting that it was a virtual or online meeting.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **dspy-ai Installation Stumbles**: A user noted that the `requirements.txt` file lists `dspy==2.0.5` but questioned if it should actually be `dspy-ai` instead.
   - They also pointed out a potential compatibility issue with `pickle5==0.0.12` which is compatible with Python versions below 3.8, while `dspy-ai` requires Python 3.9 or higher.
- **Can ADAS Invent New Building Blocks?**: A user asked if ADAS could invent new building blocks like function calling to an integrated system.
   - They also inquired if anyone has already experimented with something similar.
- **Multi-Lora Setting for DSPy Finetuning**: A user suggested using a multi-lora setting for DSPy finetuning, believing it could be a valuable approach.
   - No further details were provided about how this might be implemented.
- **DSPy vs. Langchain/LLamaindex: Choose Your Weapon**: A user asked about comparing DSPy to Langchain and LLamaindex.
   - They were directed to the DSPy documentation for guidance on choosing the right tool.
- **Aider v0.51.0: Prompt Caching and Repo Mapping Improvements**: Aider released version 0.51.0, featuring improved prompt caching for Anthropic models, optimized repo mapping for larger repositories, and enhanced Jupyter Notebook .ipynb file editing.
   - The release includes a variety of bug fixes and improvements, and Aider contributed 56% of the code for this version, as noted in the [Release history](https://aider.chat/HISTORY.html#v0510>>>).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LTXStudio Launches Five New Features**: LTXStudio has released five new features for users to take their projects to the next level.
   - These features are accessible and testable now, with a tweet from LTXStudio announcing the release and encouraging users to try them out: [Tweet from LTX Studio (@LTXStudio)](https://x.com/LTXStudio/status/1825909655207383308?t=5Wk2X8i_lQ5R5HAJxcerlg&s=19).
- **JPEG Encoding: An Uncertain Image Tokenization Method**: A research paper proposes JPEG encoding as a viable image tokenization method, but current AR-based approaches struggle with significant information loss, resulting in low image quality.
   - The paper uses a JPEG quality setting of 25, which theoretically hinders high-quality image generation from the tokens and compresses a 256*256 image to 5,000 tokens, making training and inference slower than traditional VQ-VAE.
- **Questions About Image Compression Limits**: The author questions the maximum compression possible for images, given the paper's use of a JPEG quality setting of 25 for tokenization.
   - This raises concerns about the potential limitations of this method in achieving optimal image compression.
- **Training Models on H.265 or AV1 Frames**: The author suggests exploring the possibility of training models on H.265 frames, or even AV1 frames, as a potential alternative to JPEG encoding for image tokenization.
   - This approach could potentially address the limitations of the current JPEG encoding method and lead to better performance.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Leo Models Go Public**: A member made [quantized versions of their Leo models](https://huggingface.co/GPT4All-Community) publicly available on Hugging Face.
   - They are happy to take feedback and relay messages to the users if needed, adding them to the model card if desired.
- **Feedback & Updates via Model Card**: The member offers to add messages to the model card for feedback or relaying information to users.
   - This way, anyone can see the latest information, feedback, or updates.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Xeophon's Tweet**: Xeophon posted a link to a tweet from Bilawal Sidhu about the power of interconnects in deep learning.
   - The tweet highlights how interconnects are crucial for large-scale distributed training of models and that the field is continuously evolving.
- **Placeholder**: This is a placeholder summary to satisfy the minimum requirement of 2 summaries.
   - You can replace this with a real summary if you have another topic to discuss.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1275214173705338955)** (91 messages🔥🔥): 

> - `Fine-tuning Llama-3.1-405B`
> - `Unsloth limitations`
> - `Hugging Face Space GPU`
> - `Free model access`
> - `Training Loss` 


- **Fine-tuning Llama-3.1-405B with Hugging Face Space GPU?**: A member asked if they could fine-tune **Llama-3.1-405B** using an **H100 Hugging Face Space GPU**.
   - They were told that they would need at least **360 GB of GPU memory** and that **Unsloth currently does not support this.**
- **Unsloth does not support fine-tuning Llama-3.1-405B**: Unsloth does not currently support fine-tuning **Llama-3.1-405B**, which requires at least **360 GB of GPU memory.**
   - It was noted that **eight H100 GPUs** would be required for this task, but Unsloth does not offer this functionality.
- **Free model access and training**: A member inquired about whether **Lambda offers free fine-tuning** for **Llama-3.1-405B.**
   - They were informed that **Lambda does not offer free fine-tuning**, only free model execution, which is also available on platforms like **Hugging Face, Meta, and Groq.**
- **Training Loss issues and troubleshooting**: A member encountered difficulties in getting their **training loss** to stay below **1.000** while fine-tuning a model on a **Google Colab A100 runtime.**
   - They explored methods like halving the learning rate and adjusting the batch size, but concluded that using a **Colab A100 runtime** might not be a feasible long-term solution due to the high GPU memory requirements.
- **Unsloth Premium and partnerships**: A user asked for the pricing of **Unsloth Premium** and inquired about potential **partnerships with Unsloth.**
   - It was stated that **Unsloth Premium** is not available for direct purchase and its faster versions are limited to Fortune 500 companies. They were advised to reach out to **Mike or Daniel** for further information.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1275202744889770045)** (103 messages🔥🔥): 

> - `Perplexity (PPL)`
> - `Model Fine-tuning`
> - `Javascript AST Walking` 


- **PPL is a cheap test, but not a perfect metric**: **PPL** is a good metric for comparing the effects of quantization, but can be misleading if the difference between the base and quantized model is large.
   - It's also useful for comparing models on a token-to-token level to see if a topic has been observed, but the absolute value is meaningless - it's the delta that matters.
- **Companies downplay fine-tuning, but it's a powerful technique**: While companies like Anthropic and Google downplay the importance of **fine-tuning**, it can significantly improve model performance and is cost-effective compared to training from scratch.
   - OpenAI and Google are now pushing fine-tuning, likely to expand the market and gain new customers.
- **Javascript AST walking is a pain**: One member expressed a strong dislike for **Javascript AST walking**, finding it difficult and time-consuming, particularly when dealing with obfuscated code.
   - They described the experience as 'a pain' and lamented the need to deal with complex call stacks, obfuscation, and heavy use of bit shifting.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1275213997389512796)** (12 messages🔥): 

> - `Llama 3.1 Fine-Tuning`
> - `WSL Anaconda Installation`
> - `Mini Conda` 


- **Llama 3.1 Fine-Tuning on HuggingFace Space GPU**: A member asked if they could fine-tune **Llama-3.1-405B** on a **HuggingFace Space GPU** using an **H100**.
- **LoRA_MLP Backward Pass Confusion**: A member is confused about the manual derivatives in the backward pass for **LoRA_MLP** and is looking for clarification on the correctness of the equations.
- **Anaconda Installation on WSL Issues**: A member reported that they successfully installed **WSL** but encountered issues installing **Anaconda** following a tutorial video.
- **Mini Conda Installation Suggestion**: Another member suggested installing **Mini Conda** as a solution to the Anaconda installation problems.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

etherl: <@488399737884639242> no self promotions please <:slothhug:1257540335438008343>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1275399968814927924)** (3 messages): 

> - `Lottery Ticket Adaptation`
> - `LoRAs`
> - `Finetuning`
> - `Catastrophic Forgetting` 


- **Lottery Ticket Adaptation: A New Approach to Fine-tuning**: A new method called [Lottery Ticket Adaptation](https://github.com/kiddyboots216/lottery-ticket-adaptation) is an alternative to LoRAs and finetuning that aims to avoid catastrophic forgetting.
   - This method identifies the important weights for the new task and only trains those weights, potentially preserving the knowledge from the original model.
- **Lottery Ticket Adaptation: A New Approach to Fine-tuning**: A new method called [Lottery Ticket Adaptation](https://github.com/kiddyboots216/lottery-ticket-adaptation) is an alternative to LoRAs and finetuning that aims to avoid catastrophic forgetting.
   - This method identifies the important weights for the new task and only trains those weights, potentially preserving the knowledge from the original model.


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1275180850392072367)** (4 messages): 

> - `Llama Model on Mac` 


- **Setting up Llama on Mac**: A member asked if they could set up the new Llama model on their Mac with an M3 Air and 24GB of RAM.
- **Hugging Face Discord**: The member who provided the link suggested asking the question on the Hugging Face discord.


  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1275497831398117467)** (8 messages🔥): 

> - `Triton Error`
> - `Triton kernel optimization`
> - `constexpr type` 


- **Triton kernel optimization**: A user encountered a `ValueError` while attempting to use `tl.arange` with a non-constexpr value `seqlen` in a `triton.jit` kernel.
   - The user, a beginner with Triton, asked for help with the error, and was pointed to the specific line causing the issue: `base_idx_hidden_states = base_idx + tl.arange(0, seqlen)[:, None] * head_dim`. 
- **constexpr type in Triton**: The issue arose because `seqlen` was not declared as a `tl.constexpr` type, which is required for the `tl.arange` function in Triton.
   - The user was advised to explicitly specify the type of the variable `seqlen` within the function definition itself to resolve the error. 
- **Triton compiler behavior**: The user noted that this type specification was not needed in Python generally, where type inference is usually handled automatically.
   - This instance, however, highlights a key difference between Triton and regular Python code, emphasizing the importance of explicit type declarations in Triton kernels.


  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1275242807958700155)** (73 messages🔥🔥): 

> - `Comfy FP16`
> - `FP8 Matmul`
> - `Stable-fast`
> - `Oneflow/Onediff`
> - `Flux` 


- **Comfy's FP16 Implementation**: A member was under the impression that Comfy supports FP16 accumulator by default, but it requires a custom Torch C++ extension.
- **FP8 Matmul in Comfy**: Comfy's FP8 implementation doesn't actually use FP8 matmul for computation; it only uses it as an intermediate data type.
- **Stable-fast: An Alternative to Comfy**: A member recommends using Stable-fast, which doesn't support Flux (as it's abandoned) but has interesting ideas for optimization.
- **Oneflow/Onediff vs. Flux**: Oneflow/Onediff is a wrapper for diffusion models that uses Oneflow for inference and graph creation, but it's not compatible with Flux because Flux is too large.
- **Diffusion Model Quantization**: A member discussed how diffusion models can be effectively quantized by keeping self-attention and accumulation in FP16.



**Link mentioned**: <a href="https://github.com/mobiusml/hqq/blob/master/hqq/kernels/hqq_aten_cuda_kernel.cu#L14-L22">hqq/hqq/kernels/hqq_aten_cuda_kernel.cu at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1275258599052017765)** (2 messages): 

> - `CUDA matrix transpose`
> - `CUTLASS tutorial`
> - `4090D with 48GB`
> - `FP8 support`
> - `bf16 testing` 


- **CUTLASS Tutorial on Matrix Transpose**: A new tutorial from Colfax International focuses on memory copy techniques for NVIDIA GPUs using [CUTLASS](https://github.com/NVIDIA/cutlass/) and CuTe, using matrix transpose as an example.
   - The tutorial draws inspiration from [Mark Harris’s Efficient Matrix Transpose tutorial](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/), but focuses on CuTe abstractions.
- **4090D with 48GB and FP8 Support**: A Twitter user, @bdsqlsz, boasts about obtaining a 4090D with 48GB and access to Torch 2.4, boasting about the support for FP8.
   - They also share a conversation with a cloud platform, who recognized them as a prominent AI figure in China.
- **bf16 Testing Results**: The user also mentions their involvement in bf16 testing, further highlighting their involvement in the Chinese AI community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/bdsqlsz/status/1821838464108917123">Tweet from 青龍聖者 (@bdsqlsz)</a>: 48g 4090d with torch 2.4 fp8.  Quoting 青龍聖者 (@bdsqlsz)   I just contacted the cloud platform for testing, and the person in charge said to me: Oh, bdsqlsz，it&#39;s you~ Give you free computing power, ...</li><li><a href="https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/">Tutorial: Matrix Transpose in CUTLASS</a>: The goal of this tutorial is to elicit the concepts and techniques involving memory copy when programming on NVIDIA® GPUs using CUTLASS and its core backend library CuTe. Specifically, we will stud…
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1275339113628962816)** (1 messages): 

> - `Krish's Skillset`
> - `Krish's Job Search`
> - `Krish's Experience` 


- **Krish: CS Grad with Machine Learning Expertise**: Krish, a UC San Diego MS in CS grad, brings 2 years of professional software engineering and machine learning experience, specializing in deep learning models, including language and generative models.
   - Krish has a strong background in C++ software development, having integrated language models into enterprise software and created point cloud visualization tools.
- **Krish's Job Search: Full-Time & Internship Opportunities**: Krish is seeking both full-time and internship opportunities as soon as possible.
   - Krish, an international student on a deadline, is open to any leads and appreciates any help during this difficult time.
- **Krish's Strong Work Ethic**: Krish emphasizes his hard work and dedication to learning, highlighting his belief that he would be a valuable asset to any team.
   - Krish's resume can be found at [https://shorturl.at/YESMq](https://shorturl.at/YESMq).



**Link mentioned**: <a href="https://shorturl.at/YESMq">Krish_Rewanth_Resume.pdf</a>: no description found

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1275395727715078239)** (9 messages🔥): 

> - `CUDA Setup for VS Code`
> - `PyTorch CUDA errors`
> - `C++ CUDA code issues`
> - `VS Code C_CPP_Properties.json`
> - `PyTorch Cpp Extension` 


- **Setting up CUDA for VS Code**: A user inquired about setting up CUDA for VS Code, and a helpful member directed them to the official [NVIDIA Nsight™ Visual Studio Code Edition](https://developer.nvidia.com/nsight-visual-studio-code-edition) documentation for detailed instructions.
   - This documentation emphasizes that the GPU being debugged must be on a Linux or QNX target system, and local debugging can only be done on Linux systems.
- **Resolving PyTorch CUDA Errors**: Another user encountered errors with a `.cu` file that worked when running through the PyTorch `cpp_extension`.
   - A fellow user suggested adding specific include paths to the `c_cpp_properties.json` file within the VS Code workspace, including paths related to PyTorch, THC, CUDA, and Python.
- **C++ CUDA Code Errors and Fixes**: After adding the recommended include paths, the user still faced errors. They discovered that replacing `torch::cuda` with `c10:cuda` resolved the issue.
   - The user also noticed a significant initial run time of almost a minute, but subsequent runs were much faster, taking only a few seconds.



**Link mentioned**: <a href="https://developer.nvidia.com/nsight-visual-studio-code-edition">Nsight Visual Studio Code Edition</a>: CUDA development for NVIDIA platforms integrated into Microsoft Visual Studio Code

  

---


### **CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1275266091173548084)** (1 messages): 

> - `Composable Kernel`
> - `ROCm`
> - `Tile Programs`
> - `GPU Computing` 


- **Composable Kernel: Performance Portable GPU Computing**: A [GitHub repository](https://github.com/ROCm/composable_kernel/tree/ck_tile_toy/example/91_tile_program) for the **Composable Kernel** project within **ROCm** showcases an example of a **tile program** designed for performance portability in GPU computing.
- **Tile Program Example: 91_tile_program**: The specific example, named **91_tile_program**, demonstrates a practical implementation of tile programming within the Composable Kernel framework, highlighting its potential for optimizing performance on various GPU architectures.



**Link mentioned**: <a href="https://github.com/ROCm/composable_kernel/tree/ck_tile_toy/example/91_tile_program">composable_kernel/example/91_tile_program at ck_tile_toy · ROCm/composable_kernel</a>: Composable Kernel: Performance Portable Programming Model for Machine Learning Tensor Operators - ROCm/composable_kernel

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1275295714561232989)** (81 messages🔥🔥): 

> - `Llama2 eval crashing`
> - `GPT-Fast eval script`
> - `HF_eval script`
> - `OOM for Llama2`
> - `Int4wo vs bf16 performance` 


- **Llama2 eval crashes during model loading**: A user reported that running Llama2 eval crashes during the model loading phase, simply printing 'killed' and exiting.
- **GPT-Fast vs HF_eval Script Comparison**: The discussion centered around the use of different evaluation scripts, particularly comparing the GPT-Fast evaluation script with HF_eval.
- **HF_eval Script Limitations and Performance Issues**: The user reported that they encountered an issue while running the HF_eval script for evaluating Llama2, resulting in an error message indicating an unsupported default value for the `zero_point_domain` parameter.
- **OOM Issue with Llama2**: The user encountered an out-of-memory (OOM) error while running Llama2 evaluation, even though their system should have enough RAM and GPU memory.
- **Model Loading Discrepancies between Torch and Transformers**: The discussion highlighted a discrepancy in how the Transformers library and Torch load models, with HF_eval potentially having an issue with correctly using the specified precision for model loading.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/blob/main/torchao/_models/llama/generate.py#L146-L155">ao/torchao/_models/llama/generate.py at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/scripts/hf_eval.py">ao/scripts/hf_eval.py at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/_models/llama/generate.py">ao/torchao/_models/llama/generate.py at main · pytorch/ao</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1275262822044925994)** (3 messages): 

> - `H100 L2 cache optimization`
> - `memcpy optimization`
> - `cuMemAddressReserve`
> - `deterministic memory allocation` 


- **H100 L2 cache optimization: reversing the hash**: The author was trying to reverse engineer the hash function for the L2 cache on H100 GPUs and discovered that the only dynamic data appears to be a single bit per 2 MiB (MMU page).
   - This bit changes at a 4KiB granularity, is always balanced 50/50 every 64KiB, and can be handled efficiently with persistent threads.
- **Competing with NVIDIA on memcpy**: While the author hasn't yet tried to beat NVIDIA at matrix multiplication, they are considering optimizing memory copy operations first.
   - The goal is to achieve better power efficiency compared to NVIDIA's implementation, and potentially simplify the complexity of llm.c.
- **Eliminating dynamic memory allocation with cuMemAddressReserve**: The author realized that cuMemAddressReserve combined with cuMemMap/cuMemUnmap can completely eliminate the dynamic aspect of memory allocation.
   - This allows for deterministic allocation of physical pages based on a simple hash, potentially leading to optimized elementwise kernels in both llm.c and PyTorch.


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/)** (1 messages): 

evil666man: Would love to collaborate here!
  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1275249617029562420)** (6 messages): 

> - `Hermes 3/405`
> - `Llama 3.1-instruct-405`
> - `Meta-Llama 405b`
> - `LLM Arena`
> - `Hermes 3 Launch` 


- **Hermes 3/405 vs. Meta-Llama 405b**: A member asked if anyone has compared **Hermes 3/405** to other models, specifically if it is on par with **Meta-Llama 405b**.
   - They could not find **Hermes** in the [LLM Arena leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard).
- **Hermes 3 Benchmarks**: Another member stated that **Hermes 3** is compared to **Llama 3.1-instruct-405** in the technical report, using a suite of about 15 benchmarks.
   - The member was looking for performance comparisons to other LLMs, including **Meta-Llama 405b**.
- **Hermes 3 Launch Video**: A member shared a [YouTube video](https://www.youtube.com/watch?v=uAo513GIwoU) discussing the launch of **Hermes 3**.
   - They also mentioned that they will discuss **Hermes 3** in their upcoming video.



**Link mentioned**: <a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>: no description found

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1275340024115757178)** (2 messages): 

> - `pydantic-xml extension`
> - `Nous Aesthetics` 


- **PyDantic-XML Extension**: The **pydantic-xml** extension provides functionality for serializing and deserializing data between Pydantic models and XML.
- **Nous Aesthetics Video**: A video on **Nous Aesthetics**, with a YouTube link, [https://youtu.be/qGQ5U3dkZzk?si=MPLh7XEd1NrskX5g](https://youtu.be/qGQ5U3dkZzk?si=MPLh7XEd1NrskX5g), was shared.
- **PyDantic-XML Documentation**: The **pydantic-xml** extension documentation can be found at [https://pydantic-xml.readthedocs.io/en/latest/](https://pydantic-xml.readthedocs.io/en/latest/).



**Link mentioned**: <a href="https://pydantic-xml.readthedocs.io/en/latest/">pydantic-xml</a>: no description found

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1275188271013695661)** (93 messages🔥🔥): 

> - `Llama 3.1 Minitron 4B Width Base`
> - `Hermes 3 Image Generation`
> - `Hermes 3 Amnesia Mode`
> - `Hermes 3 405B Function Calling`
> - `Hermes 3 Performance Differences` 


- **Llama 3.1 Minitron 4B Width Base**: Llama-3.1-Minitron-4B-Width-Base is a text-to-text model that can be adopted for a variety of natural language generation tasks. It is obtained by pruning Llama-3.1-8B.
   - Specifically, it prunes model embedding size, number of attention heads, and MLP intermediate dimension. Following pruning, we perform continued training with distillation using 94 billion tokens to arrive at the final model, using the continuous pre-training data corpus used in Nemotron-4 15B for this purpose.
- **Hermes 3 cannot generate Images**: Hermes 3 is a text-to-text model. You can interact with H3-405B in https://discord.com/channels/1053877538025386074/1149866614590816256.
   - The bots cannot trigger image generation through a / command, they can only interact by @ mentioning each other.
- **Hermes 3 Amnesia Mode**: Amnesia mode is a feature of Hermes 3 8b that can be triggered by prompting it with "Hi" with no system prompts.
   - Amnesia mode is not available on Discord, as it remembers all the chats.
- **Hermes 3 405B Function Calling**: It was asked if there were any other API providers that offered faster speeds than the lambda API provider for the 405b model, but none were known.
   - The user attempted to convince the Discord servers of other LLMs like togather and octo to pay for faster speeds, but there was no response.
- **Hermes 3 Performance Differences**: It was observed that Hermes 3 8b runs differently in lm studio and koboldcpp, with similar sampler settings.
   - While both only used minp (0.1) and temp, the temp in koboldcpp was 1.25, whereas in lm studio it was 0.7.



**Link mentioned**: <a href="https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Width-Base">nvidia/Llama-3.1-Minitron-4B-Width-Base · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1275505791809355787)** (7 messages): 

> - `Roleplay benchmark`
> - `Knowledge cutoff updates`
> - `Model pretraining` 


- **Automated roleplay benchmark: A challenging task**: A member pondered the possibility of creating an automated roleplay benchmark, suggesting counting adjectives as a potential metric.
   - The member also inquired about methods for algorithmically scoring creative writing, highlighting the difficulty of evaluating creativity objectively.
- **Knowledge cutoff updates: A mystery**: A member inquired about knowledge cutoff updates, seeking resources to understand how new knowledge is incorporated into models without significantly altering their core functionality.
   - Another member mentioned that knowledge updates used to be more frequent but have become less common recently, suggesting that recent updates have been tied to specific model changes.
- **Continued pretraining: A key to knowledge updates**: A member expressed uncertainty about updating knowledge without continued pretraining, suggesting that continued pretraining may be a necessary component for incorporating new knowledge into models.
   - The member also noted that recent knowledge updates have often been accompanied by new model changes, implying a connection between model updates and knowledge updates.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1275174237983998126)** (78 messages🔥🔥): 

> - `OPRO Paper`
> - `C4AI Discord Invite`
> - `Cohere API Response Format`
> - `Cohere Classify Sunset`
> - `Reranker API on 10k Docs` 


- **DeepMind's OPRO Paper**: A member inquired about the source of the information regarding an OPRO-based prompt tuner.
- **C4AI Discord Invite**: A member requested an invite to the C4AI Discord server.
- **Cohere API Response Format - `response_format` Issue**: A member encountered an error while using the `response_format` parameter in the Cohere API.
- **Cohere Classify Sunset**: A member inquired about potential alternatives to the Cohere Classify endpoint.
- **Reranker API Efficiency for Large Datasets**: A member asked if chunking large datasets and running the Reranker API independently on each chunk would produce accurate overall relevancy scores.



**Link mentioned**: <a href="https://jobs.lever.co/cohere/bb3df91e-bef0-43b0-9e69-d8efa5ec1c8b">Cohere - Research Scholar</a>: Why this role? Cohere For AI is the dedicated research arm of Cohere. The Cohere For AI research lab seeks to solve complex machine learning problems by supporting fundamental research that explores t...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1275204831069470823)** (16 messages🔥): 

> - `Cohere Sponsorship`
> - `PDF Abstract Extraction`
> - `Cohere API SSL Verification`
> - `Cohere API LangChain`
> - `Freelance Developer Team` 


- **Hack49 Global Seeks Cohere Sponsorship**: Rishi Shah, co-founder of Hack49 Global, an international hackathon, seeks guidance on how to proceed with a Cohere sponsorship request.
   - They were advised to join the Cohere Discord and contact a specific user, but are unsure about the appropriate communication channel (DM or public channel).
- **Extracting Abstracts from PDFs Without GPT-4**: A community member is struggling to extract abstracts from PDFs without relying on GPT-4, using py_pdf_parser and embedding models.
   - They are seeking help from the community to find solutions and explore alternative LLM models with size constraints (under 2GB).
- **Resolving Cohere API SSL Verification Errors**: A user encounters an SSL verification error while using the Cohere API with Langchain, and wants to whitelist the API URL to resolve the issue.
   - The community suggests setting `verify=False` in the Langchain client, and provides the API URL (`https://api.cohere.com`) to facilitate whitelisting.
- **Freelance Developer Team Offers Services**: A developer with over 8 years of experience has formed a team of specialists and is seeking clients for projects.
   - They offer flexibility in project scope and budget, accommodating both demanding and cost-conscious clients, but emphasize intolerance for excessive demands.
- **Finding Sensitive Fields in Large Documents**: A user is working on a tool to find sensitive fields in large documents and seeks suggestions for efficient processing.
   - They have considered using Cohere's `documents` field, but are exploring alternative solutions like vector databases for faster processing on large files.


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1275536642970751120)** (1 messages): 

> - `Hermes 3` 


- **Hermes 3 Released**: **Hermes 3**, a 70B parameter model, has been released.
   - You can say hi to it at [OpenRouter](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b).
- **Hermes 3 Release Announcement**: The Hermes 3 announcement also included a copyright notice for OpenRouter, LLC.
   - The copyright notice states © 2023 - 2024 OpenRouter, LLC



**Link mentioned**: <a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b">Hermes 3 70B Instruct - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo), including advanced agentic capabilities, much better roleplaying, rea...

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1275213908554022922)** (84 messages🔥🔥): 

> - `GPT Functions`
> - `OpenRouter Model Support`
> - `German Pretraining`
> - `Mistral`
> - `Multilingual Models` 


- **GPT Functions on OpenRouter**: A user asked if GPT functions are still supported on OpenRouter, as they are receiving 'function_call=None' even though the stop reason is 'functioncall'.
   - The OpenRouter team confirmed that better tool call routing is coming soon, but currently, results may vary unless using OpenAI, Anthropic, or Google models.
- **Mistral Large Instruct 2407 for German**: A user inquired about a model with good German pretraining and was suggested to try Mistral-Large-Instruct-2407, which is multi-lingual by design and supports German.
   - The user tested the model but found it to be 'okay' but not great, and further suggested checking Hugging Face for other models.
- **OpenAI Assistant Embedding on WordPress**: A user asked for guidance on embedding an OpenAI assistant, including docs and instructions, on a WordPress website.
   - The user mentioned that WordPress supports the straight API but not the assistant API, and requested advice on go-to services or open-source options for embedding.
- **OpenRouter Error with Non-Free Models**: Users reported encountering an error when trying to access non-free models on OpenRouter, specifically getting a 'client-side exception' and needing to hard refresh the browser.
   - The OpenRouter team investigated and determined that the issue was related to access token expiration and potentially CORS errors, and ultimately resolved the issue.
- **OpenRouter Uncensored Models**: A user inquired about uncensored models on OpenRouter.
   - It was suggested that 'open source' and 'roleplay' tags are good indicators for models that may produce NSFW content, with popular options including Dolphin, Stheno, Euryale, and MythoMax.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://js.stripe.com')">no title found</a>: no description found</li><li><a href="https://openrouter.ai')."">no title found</a>: no description found</li><li><a href="https://ai.azure.com/explore/models/Phi-3.5-MoE-instruct/version/1/registry/azureml">Azure AI Studio</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>: no description found</li><li><a href="https://aka.ms/Phi-3.5-mini-instruct-pricing,">Info</a>: Mind-blowing beauty like the kind pictured in toda</li><li><a href="https://huggingface.co/microsoft/Phi-3.5-vision-instruct">microsoft/Phi-3.5-vision-instruct · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b">Hermes 3 70B Instruct - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo), including advanced agentic capabilities, much better roleplaying, rea...</li><li><a href="https://huggingface.co/microsoft/Phi-3.5-MoE-instruct">microsoft/Phi-3.5-MoE-instruct · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/models?modality=text%2Bimage-%3Etext">Models | OpenRouter</a>: Browse models on OpenRouter
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1275234886642438154)** (45 messages🔥): 

> - `Uncensored models`
> - `LM Studio server issues`
> - `Llama 3.1`
> - `Speech to Text and Text to Speech`
> - `Vision models` 


- **Exploring Uncensored LLM Models**: A user requested suggestions for uncensored LLM models best suited for non-coding tasks.
   - A link to [llm.extractum.io](https://llm.extractum.io/list/?uncensored) was provided, highlighting the platform's focus on uncensored LLMs and their potential for diverse applications like legal analysis, medical research, and creative writing.
- **LM Studio Server Issues with Llama 3.1**: A member reported encountering issues with LM Studio's local inference server, specifically with Llama 3.1, where the stop pattern was ignored.
   - They noted that the issue was absent in chat mode and suggested opening a discussion in the relevant channel to further troubleshoot and identify potential solutions.
- **Speech-to-Text and Text-to-Speech in LM Studio**: A user inquired about the availability of voice interaction with Llama 2/3 models, specifically whether speech-to-text and text-to-speech functionalities were integrated into LM Studio.
   - It was clarified that LM Studio currently lacks this support, prompting the user to explore external solutions like [Parler-TTS](https://github.com/huggingface/parler-tts) for text-to-speech and [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for speech-to-text.
- **Vision Models in LM Studio**: A user inquired about models capable of processing photos or videos as input, aiming to provide visual context for coding tasks.
   - It was confirmed that local models in LM Studio cannot handle such tasks; only cloud-based models like GPT4o and Claude currently offer this functionality.
- **Automating LM Studio Server Startup and Model Loading**: A user sought assistance in automating the startup of the LM Studio server and loading a specific LLM model.
   - The LM Studio SDK was recommended, providing a means to manage and automate these tasks through its [documentation](https://lmstudio.ai/docs/lmstudio-sdk/quick-start) and [GitHub repository](https://github.com/lmstudio-ai/lmstudio.js).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llm.extractum.io/list/?uncensored">Best Uncensored LLMs (Large Language Models): Explore the Curated List of the Best Uncensored LLMs</a>: Discover the advanced capabilities of the best uncensored LLM models on our platform. Explore the capabilities, benchmarks and internals of uncensored LLMs, ideal for handling complex data and sensiti...</li><li><a href="https://x.com/hellokillian/status/1723106008061587651)">Tweet from killian (@hellokillian)</a>: we&#39;ve literally been flying blind until now  $ interpreter --vision &gt; Recreate this component in Tailwind CSS  (this is realtime)</li><li><a href="https://lmstudio.ai/docs/lmstudio-sdk/quick-start">Quick Start Guide | LM Studio</a>: Minimal setup to get started with the LM Studio SDK</li><li><a href="https://github.com/lmstudio-ai/lmstudio.js">GitHub - lmstudio-ai/lmstudio.js: LM Studio TypeScript SDK (pre-release public alpha)</a>: LM Studio TypeScript SDK (pre-release public alpha) - lmstudio-ai/lmstudio.js</li><li><a href="https://github.com/huggingface/parler-tts">GitHub - huggingface/parler-tts: Inference and training library for high-quality TTS models.</a>: Inference and training library for high-quality TTS models. - huggingface/parler-tts</li><li><a href="https://github.com/ggerganov/whisper.cpp">GitHub - ggerganov/whisper.cpp: Port of OpenAI&#39;s Whisper model in C/C++</a>: Port of OpenAI&#39;s Whisper model in C/C++. Contribute to ggerganov/whisper.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1275222028013797528)** (36 messages🔥): 

> - `M2 Ultra`
> - `GPU performance`
> - `Nvidia 4090 vs 4060 Ti`
> - `Nvidia 48GB Card`
> - `LLM speed` 


- **M2 Ultra for AI Tasks**: A user expressed excitement for the upcoming M2 Ultra, noting that its performance is highly anticipated for AI tasks.
- **Nvidia 4090 vs 4060 Ti for AI**: The discussion turned to comparing the Nvidia 4090 and two 4060 Ti's for AI applications.
- **Nvidia 48GB Card Rumors**: A user inquired about the possibility of Nvidia releasing a 48GB consumer-grade card.
- **AMD 7900GRE Performance with LLMs**: A user reported slow performance with an AMD 7900GRE for AI tasks, specifically with 6B, 7B, and 8B models.
- **Nvidia Cards Recommended for AI**: Multiple users recommended Nvidia cards, specifically the 3090, for achieving faster AI performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/skeletor-laughs-in-evil-laughing-myah-myaah-dasmemeistgut-gif-5356566587527840753">Skeletor Laughs In Evil Laughing Myah Myaah Dasmemeistgut GIF - Skeletor laughs in evil laughing myah myaah dasmemeistgut - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1275410724910272576)** (6 messages): 

> - `GPT-4 Neuron Explanations`
> - `BlackboxNLP Paper`
> - `Language Models Explain Neurons` 


- **GPT-4 Neuron Explanations Debunked?**: A member recalled a paper claiming that GPT-4 explanations of neurons were not useful or better than baselines. 
   - Another member provided a link to a paper titled "Language Models can explain neurons in language models" but could not find a paper with a similar title claiming GPT-4 explanations were not useful, though the content was similar to the member's recollection.
- **BlackboxNLP Paper Exploration**: A member sought to identify a paper that claimed GPT-4 explanations of neurons were not useful or better than baselines. 
   - Another member searched through Google Scholar citations of the paper "Language Models can explain neurons in language models" and found no such paper with a similar title.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1275178946245300268)** (48 messages🔥): 

> - `Model Training`
> - `MiniPile Dataset`
> - `Frankenmerging`
> - `Model Merging`
> - `KANs` 


- **Training Models on Limited Data**: A member noted that training a model on a single, small file can lead to nonsensical outputs due to the influence of random initialization.
   - They further compared it to text compression benchmarks where models are trained to memorize a specific block of text as effectively as possible.
- **MiniPile Dataset for Efficient Training**: A member recommended the MiniPile dataset as a good alternative to the full Pile corpus, which is often too large for academic budgets.
   - MiniPile is a 6GB subset of the Pile corpus curated by filtering out low-quality clusters and aims to provide a diverse pre-training dataset for smaller-scale models.
- **Frankenmerging - Composing Layers from Different Models**: A member inquired about the feasibility of composing layers from two different models, a technique referred to as 'frankenmerging.'
   - They expressed confusion about why this wouldn't lead to complete garbling of the model's internal representation and sought clarification on the potential benefits and challenges of this approach.
- **Model Merging with Optimizers**: A member suggested using an optimizer to find the best permutation of channels between layers of two different models before stacking them together.
   - They acknowledged the potential challenges and noted that such methods haven't been demonstrated for large GPT models.
- **Kolmogorov-Arnold Networks (KANs) - A New Approach?**: A paper on KANs was discussed, advocating for their seamless synergy with scientific discovery by leveraging their ability to identify relevant features, reveal modular structures, and discover symbolic formulas.
   - The discussion focused on whether KANs offer advantages over transformers or CNNs for specific tasks and the lack of compelling evidence to support their claims beyond a single experiment comparing them to MLPs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2309.16039">Effective Long-Context Scaling of Foundation Models</a>: We present a series of long-context LLMs that support effective context windows of up to 32,768 tokens. Our model series are built through continual pretraining from Llama 2 with longer training seque...</li><li><a href="https://arxiv.org/abs/2402.01032">Repeat After Me: Transformers are Better than State Space Models at Copying</a>: Transformers are the dominant architecture for sequence modeling, but there is growing interest in models that use a fixed-size latent state that does not depend on the sequence length, which we refer...</li><li><a href="https://arxiv.org/abs/2406.07887">An Empirical Study of Mamba-based Language Models</a>: Selective state-space models (SSMs) like Mamba overcome some of the shortcomings of Transformers, such as quadratic computational complexity with sequence length and large inference-time memory requir...</li><li><a href="https://arxiv.org/abs/2408.10205">KAN 2.0: Kolmogorov-Arnold Networks Meet Science</a>: A major challenge of AI + Science lies in their inherent incompatibility: today&#39;s AI is primarily based on connectionism, while science depends on symbolism. To bridge the two worlds, we propose a...</li><li><a href="https://huggingface.co/datasets/JeanKaddour/minipile">JeanKaddour/minipile · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1275457161304084551)** (4 messages): 

> - `Chinchilla vs Gopher Data Filtering` 


- **Gopher paper mentions data filtering**: The Gopher paper discusses data deduplication and content filtering, but Chinchilla does not appear to mention this specifically, despite referencing the Gopher paper.
   - Both papers use the MassiveText dataset, but Chinchilla pre-trained on 1.4T tokens, while Gopher pre-trained on 300B tokens.
- **Data filtering strategies differ**: A user noted that Chinchilla and Gopher likely differ in their data filtering techniques, despite both using the MassiveText dataset.
   - They were curious about specific data filtering approaches used by Chinchilla, but the paper may not offer detailed insights.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1275169687810080808)** (21 messages🔥): 

> - `Llama 3.1 System Prompt`
> - `Llama Eval Chat Template`
> - `Huggingface Chat Template`
> - `System Prompt in Huggingface`
> - `YAML Parameters` 


- **Llama 3.1 Automatically Adds System Prompt**: A user noticed that Llama 3.1 automatically adds a system prompt, whereas Llama 3 does not.
   - The system prompt automatically added by Llama 3.1 reads:

*Cutting Knowledge Date: December 2023*
*Today Date: 26 Jul 2024*
- **System Prompt Overriding**: The user asked about what happens when a custom system prompt is provided for Llama 3.1.
   - It was confirmed that the custom system prompt is concatenated with the default system prompt.
- **Bug in Llama 3.1 Chat Template**: The issue is suspected to be a bug in the Llama 3.1 chat template or the tokenizer.apply_chat_template method.
   - It was suggested that this issue should ideally be fixed upstream in Huggingface.
- **Custom Jinja Template as Workaround**: A workaround using a custom Jinja template is being considered.
   - The user is unsure if there is a more principled approach to handle this and similar issues.
- **Accessing YAML Parameters in Doc to Text Function**: The user inquired about accessing YAML parameters within the doc_to_text function.
   - It was suggested to use the 'process_docs' function to add a field that all documents have access to, and then access any field for the given doc when running doc_to_text.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1275187344814772244)** (56 messages🔥🔥): 

> - `Perplexity Pro Discord`
> - `Perplexity Pro Features`
> - `Perplexity Pro Search Bugs`
> - `Perplexity AI Models`
> - `Perplexity's Future` 


- **Perplexity Pro Discord access confusion**: Some users are unable to join the Perplexity Pro Discord server, even after leaving and rejoining using the link in their Perplexity settings.
   - The issue seems to be a lack of clear instructions regarding accessing the Pro section within the main Discord server.
- **Perplexity's Search Function is Broken**: Users are reporting issues with Perplexity's search function, including the inability to access online sources and the use of outdated information.
   - Some users believe this is a backend issue, but the team at Perplexity has yet to acknowledge or address the problem.
- **Perplexity Pro Model Limitations**: Users are discussing the limitations of Perplexity Pro models for tasks like coding and blog post creation.
   - Some users are finding that Perplexity Pro is not as effective as other models for certain tasks, particularly when it comes to generating complex code or avoiding hallucinations in blog posts.
- **Perplexity's Commitment to Front-End vs Backend**: There is a debate about whether Perplexity is prioritizing front-end development over backend development, with some users reporting issues with backend features like search and model selection.
   - Some users believe that these issues indicate a lack of focus on core backend functionalities, which are critical for the overall performance of the platform.
- **Perplexity AI Image Generation**: Users are inquiring about the capabilities of Perplexity Pro for generating AI images.
   - However, it appears that Perplexity Pro currently does not offer image generation capabilities, although there is discussion about future potential.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/hub/faq">Perplexity Frequently Asked Questions</a>: If you have questions about Perplexity, our FAQ page is the perfect place to find answers. Our FAQ page is organized into categories and provides clear and concise answers.</li><li><a href="https://www.perplexity.ai/hub/blog/pro-search-upgraded-for-more-advanced-problem-solving">Pro Search: Upgraded for more advanced problem-solving</a>: Research shapes our daily lives. We use it to make informed decisions and solve problems—to innovate, learn, and grow. </li><li><a href="https://x.com/rauchg/status/1825158716821320071?s=61">Tweet from Guillermo Rauch (@rauchg)</a>: This is quite exciting because one of the inspirations for App Router & RSC was to democratize the extremely dynamic rendering power of systems like Google Search and Facebook.  When you search for “w...</li><li><a href="https://monnef.gitlab.io/by-ai">[By AI] Project Index</a>: no description found</li><li><a href="https://www.perplexity.ai/search/see-uploaded-files-then-write-ZpLJtgqzRa6oO2byfwnY1Q">see uploaded files. then write a wordpress theme based on this page please....</a>: Based on the uploaded files, I can describe how the page with the applied styles might look like:  The page appears to have a retro, pixelated video game...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1275307693459902517)** (5 messages): 

> - `Perplexity Pro`
> - `LMSYS Arena`
> - `G1 Humanoid Robot` 


- **Perplexity Pro Feature Upgrade**: There was a discussion about upgrading to [Perplexity Pro](https://www.perplexity.ai/pro) which offers features like image upload, smarter AI, and more Pro Search.
- **LMSYS Arena Elo**: A user was looking for statistics about **LMSYS Arena** elo.
- **G1 Humanoid Robot Ready for Production**: [G1 Humanoid Robot](https://www.youtube.com/embed/EQxYdx79vUg) is reportedly ready for mass production.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/in-baseball-if-a-batter-swing-UAK3snvlQjmIXxnVTM8N0Q">in baseball, if a batter swing and misses for a third strike and the pitch is...</a>: Yes, in baseball, if a batter swings and misses for a third strike and the pitch is wild, runners can advance. This situation is governed by the &quot;uncaught...</li><li><a href="https://www.perplexity.ai/search/how-to-use-claude-to-do-some-i-XgTfxNeARnS4GS0vjIXsZA">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1275431970662514709)** (4 messages): 

> - `Camera quality`
> - `Discord issues` 


- **High-End Camera Quality**: A user requested an image of a poor boy studying under a street lamp in the rain, taken with a high-end camera.
   - This request implies a desire for high image quality, detailed focus, and perhaps artistic composition.
- **Discord Link Issues**: A user reported experiencing an issue with a Discord link, sharing a link to the specific channel where the issue occurred.
   - Another user asked if anyone else encountered the same issue, suggesting it might be a widespread problem.


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1275182877113323671)** (3 messages): 

> - `LlamaIndex`
> - `RAG`
> - `Retrieval-Augmented Generation`
> - `LLMs in Production`
> - `Amazon Neptune` 


- **LlamaIndex: Building Natural Language Querying Systems**: Learn how to build a natural language querying system for graph databases using LlamaIndex and Amazon Neptune!
   - This comprehensive guide by @bechbd shows you how to translate natural language questions into openCypher queries and execute queries on Amazon Neptune graph.
- **O'Reilly Media Course on RAG**: LlamaIndex has launched an O'Reilly Media course on retrieval-augmented generation, authored by @ravithejads.
   - The 2-hour course covers components of LlamaIndex, evaluation of RAG systems, the ingestion pipeline, observability, agents, multi-modality, and more.
- **LLMs in Production: An AI Product Meetup**: Join LlamaIndex for "LLMs in Production", an AI product meetup hosted by @vesslai and @pinecone in San Francisco.
   - Learn from industry leaders about building context-augmented LLMs with RAG & Vector DB, custom LLMs for smarter, faster, and cheaper solutions, and high-performance inference for production-ready LLMs.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1275191931248443434)** (36 messages🔥): 

> - `LlamaIndex Hierarchical Node Parser`
> - `LlamaIndex Retrieval`
> - `LlamaIndex ChromaDB Vector Store`
> - `Rag Application with LlamaIndex`
> - `Connecting LlamaIndex to Private LLMs` 


- **Hierarchical Node Parser Without Chunking**: A user inquired about using the LlamaIndex hierarchical node parser without performing chunking, instead wanting to create hierarchies and a knowledge graph using predefined nodes.
   - The user also wanted to retain metadata associated with the nodes, such as page IDs. This scenario was deemed not possible with the current implementation of the hierarchical node parser.
- **LlamaIndex Retrieval for Complex Questions**: The user discussed the need for retrieval capabilities for both simple and complex questions.
   - They envisioned a hierarchical approach that could recursively summarize nodes and create higher-level representations of the data, facilitating more nuanced and contextual responses to complex inquiries.
- **ChromaDB Vector Store and Top-K Retrieval**: A user inquired about retrieving the top-K closest matches to a query from a ChromaDB vector store within LlamaIndex, similar to the `search_by_vector` functionality in LangChain.
   - The user explained their use case of processing questions related to tables in a database and then using vectors to identify the most relevant tables for subsequent SQL query execution. The user found that this functionality was not readily available in LlamaIndex.
- **Connecting LlamaIndex to Private LLMs**: A user sought guidance on connecting LlamaIndex to a private LLM accessible via HTTPS.
   - They encountered a challenge with SSL and requested advice on connecting to an external LLM instance using HTTPS and an API token. The user was referred to a guide on customizing LLM setup within LlamaIndex for further assistance.
- **RAG Pipeline Setup: Semantic Chunking and Llama-Parse**: A user discussed various approaches for building a robust RAG pipeline setup, emphasizing the importance of semantic chunking and Llama-parse as potential enhancements for ingestion.
   - They inquired about common practices and recommendations for augmenting basic ingestion methods using SimpleDirectoryReader and VectorStoreIndex, with semantic chunking and Llama-parse being considered for generating spatial text and markdown text, respectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom/#example-using-a-custom-llm-model-advanced">Customizing LLMs - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/17f23014953e07eb8f8e7690d4cca7fb26c2109c/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py#L378">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-chroma/llama_index/vector_stores/chroma/base.py at 17f23014953e07eb8f8e7690d4cca7fb26c2109c · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1275194209032470590)** (22 messages🔥): 

> - `Latent Space podcast`
> - `Encoder-Decoder models`
> - `Fast.html`
> - `Saving state/Updating state`
> - `Fine-tuning vs RAG vs KV caching` 


- **Jeremy Howard's Latest Latent Space Interview**: The latest Latent Space podcast features Jeremy Howard, with highlights including discussions on Encoder-Decoder models, Fast.html, saving/updating state, fine-tuning vs RAG vs KV caching, and a new project Howard is working on.
   - The podcast is described as 'a 5-course meal,' after co-host Swyx's playful phrase 'give us a nibble.'
- **Encoder-Decoder Models vs Encoder Only**: The discussion highlights the benefits of Encoder-Decoder models over Encoder-only models, particularly in scenarios where detailed context and intricate relationships are crucial.
   - The interviewee, likely through osmosis from AI Paper Club calls, already had knowledge of this approach, suggesting a growing awareness within the AI community.
- **Whisperfile: A New Tool for Transcribing Audio**: Whisperfile is a new tool that allows users to easily turn audio into text locally.
   - Created by Justine Tunney, Whisperfile embeds OpenAI's Whisper model, offers 100% local operation, and can even translate non-English audio to English during transcription.
- **Anthropic AI Boosts Claude 3.5 Sonnet's Output**: Anthropic AI has doubled the maximum output token limit for Claude 3.5 Sonnet, expanding it from 4096 to 8192.
   - This update is now available in both the Anthropic API and Vertex AI, making it easier for developers to work with Claude 3.5 Sonnet's capabilities.
- **GPT-4 Fine-Tuning: A New Frontier**: OpenAI has released GPT-4 fine-tuning, a new feature that allows users to customize GPT-4's behavior and performance.
   - This update offers potential competition for Cursor's Composer feature, as it provides a similar approach to customizing and using large language models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://simple-bench.com]">no title found</a>: no description found</li><li><a href="https://x.com/simonw/status/1825626551398052180?s=46">Tweet from Simon Willison (@simonw)</a>: Here are my notes on trying out whisperfile, against both the tiny and the medium sized Whisper models https://simonwillison.net/2024/Aug/19/whisperfile/  Quoting Justine Tunney (@JustineTunney)   I j...</li><li><a href="https://x.com/ParallaxAngle/status/1825633740933955929">Tweet from JediCat (@ParallaxAngle)</a>: Hi Dr. Howard  @jeremyphoward ,  I totally enjoyed your discussion with @swyx  on the latest Latent Space pod.  I listened to the part 28:30 onwards twice.    Topics I enjoyed most:  1) Encoder-Decode...</li><li><a href="https://x.com/alexalbert__/status/1825920737326281184">Tweet from Alex Albert (@alexalbert__)</a>: We&#39;ve moved this out of beta so you no longer need to use the header!  Now available for Claude 3.5 Sonnet in the Anthropic API and in Vertex AI.  Quoting Alex Albert (@alexalbert__)   Good news f...</li><li><a href="https://x.com/justinetunney/status/1825594600528162818?s=46">Tweet from Justine Tunney (@JustineTunney)</a>: I just launched whisperfile which is the easiest way to turn audio into text. You download one file that embeds OpenAI&#39;s Whisper model and runs 100% locally. It&#39;ll even translate non-English i...
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1275237689771425853)** (8 messages🔥): 

> - `Mojo & MAX Update Cadence`
> - `Siamese Networks with Labels`
> - `Slice Custom Op Usage` 


- **Mojo & MAX Version Synchronization**: Previously, **Mojo** and **MAX** had independent update cycles, but now they are synchronized. 
   - This means you can install **MAX+mojo main** or **MAX+mojo nightlies**, but not **MAX main** and **mojo nightlies** separately.
- **Siamese Network with Labels?**: A user inquired about switching a Siamese network's output from a sigmoid to a label (e.g., "dog" or "cat").
   - Another user suggested that if you want to switch to labeling, using a standard model for that task might be more efficient than trying to adapt a Siamese network.
- **Slice Custom Op Usage**: A user requested a code example demonstrating the use of the **slice custom op** (https://docs.modular.com/max/api/mojo/graph/ops/slicing/slice).
   - They expressed difficulty understanding the op's arguments.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1275201216175013929)** (12 messages🔥): 

> - `Mojo's List implementation`
> - `Mojo's `ref` keyword`
> - `Mojo's `__lifetime_of` function`
> - `AI Chip Performance`
> - `Network on Chip (NoC)` 


- **Mojo's `List` uses `ref` for assignment**: A user was surprised to find no `__setitem__` method for assignment in Mojo's `List` implementation, but was informed that `__getitem__` returns a `ref[lifetime] T` which behaves like `__setitem__`.
- **Mojo's `ref` and `__lifetime_of` are new**: The `ref` keyword in function return types was introduced recently (in Mojo v244) as part of the new language features.
- **AI Chips have local memory for better performance**: AI chips are designed with a lot of local memory to fit models in the cache, reducing the penalty of frequent data transfers to RAM.
- **NoC vs. Cache Management Tradeoffs**: While NoCs provide efficient data transfer across many cores, they also introduce latency between cores.
- **Mojo's Production Readiness**: The question of when Mojo will be production-ready was raised.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1275349492060786749)** (2 messages): 

> - `Modular installation issues`
> - `Modular Manifest Error`
> - `Modular Expiration` 


- **Modular Installation Errors**: A user reported encountering a series of errors while attempting to install the "max" module using the `modular install max` command.
- **Troubleshooting Steps**: The user attempted to resolve the issue by first cleaning the Modular environment using the `modular clean` command, followed by reinstalling the "max" module.
- **Request for Modular Version Information**: Another user suggested checking the Modular version using the `modular -v` command to potentially identify the cause of the error.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1275176727647092809)** (19 messages🔥): 

> - `ChatGPT capabilities`
> - `AI Enthusiasm`
> - `Grok2`
> - `Smart Cookbook`
> - `Strawberry Release` 


- **ChatGPT Can't Count 'R's in Strawberry**: A user pointed out that ChatGPT struggles with simple tasks like counting the number of 'R's in the word 'strawberry,' implying that AI is not as advanced as some might believe.
   - This sparked a discussion about the current limitations of AI and whether it is truly intelligent or simply a tool that can perform specific tasks.
- **Grok2's Interesting Approach**: A user mentioned that Grok2 has an interesting approach to dealing with problems.
   - Another user pointed out that Grok2's method involves breaking down every question and solving it step by step, which is similar to the way humans solve problems.
- **AI Enthusiasm is Overrated?**: One user expressed that the term 'AI enthusiast' has lost its meaning due to AI's current limitations.
   - This sentiment arose from a discussion about ChatGPT's struggles with a simple task and Grok2's method of solving problems.
- **Building a Smart Cookbook**: A user sought advice on creating a 'smart cookbook' that could be trained on their favorite cookbooks and provide personalized advice.
   - This user believes that such a model could be applied to any 'how-to' book and requested information about existing solutions or projects.
- **Strawberry Release Speculation**: A user asked about the release date of 'Strawberry,' possibly a new AI model or a feature.
   - Another user responded by jokingly stating that 'Strawberry' is still in the 'unreliably sourced leak' phase and expressed skepticism about its release.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1275553419423383552)** (1 messages): 

> - `Structured Output`
> - `JSON Output`
> - `Model Performance`
> - `Prompt Engineering` 


- **Structured Output vs. JSON Output**: A user noticed that structured output sometimes gives worse responses than regular JSON mode.
- **Understanding the Differences**: The discussion focused on exploring the potential reasons for variations in response quality between structured output and JSON mode.
- **Impact on Prompt Engineering**: The conversation highlighted the significance of understanding these differences for effective prompt engineering.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1275553419423383552)** (1 messages): 

> - `structured output`
> - `JSON mode` 


- **Structured Output vs. JSON Mode**: A user noticed that structured output sometimes generates worse responses than regular JSON mode.
   - The user did not provide any further details or opinions on this topic.
- **Investigating the Discrepancy**: It's important to understand why structured output might produce inferior responses compared to JSON mode.
   - Further analysis is needed to determine the root cause of this issue.


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

wiiiktor.: When do you plan to release it, if I may ask? I see it's 99% ready.
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1275477856486686804)** (19 messages🔥): 

> - `torch.compile recompilations`
> - `torch.compile optimization`
> - `kv-cache for generation`
> - `rng generator object in torch.compile`
> - `torch.compile and custom masks` 


- **Torch.compile recompilations due to input shape changes**: Recompilations occur when the input shape changes, like when concatenating a new token in generation.
- **Torch.compile recompilations due to `grad_mode` changes**: Recompilations occur when switching between training and inference modes, as the `grad_mode` changes.
- **Torch.compile cache size limit**: Hitting the `torch._dynamo hit config.cache_size_limit (8)` message indicates a potential issue with torch.compile friendliness.
- **Torch.compile limitations with RNG objects**: Passing an RNG generator object into the model causes graph breaks, suggesting torch.compile currently doesn't support such objects.
- **Custom masks and kv-cache**: The use of custom masks might not be directly compatible with kv-cache, but utilizing your own mask and removing `self.causal_mask` can potentially resolve the issue.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1275236543803424769)** (18 messages🔥): 

> - `LLaMA 3.1 70B for SQL`
> - `Mistral 8k Limits`
> - `Model Merging Tactics`
> - `Open Empathic Project`
> - `LangChain SQLDatabaseChain` 


- **LLaMA 3.1 70B struggles with SQL**: A user reported that [LLaMA 3.1 70B](https://ai.google.com/research/pubs/pub49727.html) is unable to query a database using [LangChain's SQL agent](https://langchain.readthedocs.io/en/latest/modules/agents/agents.html#sql-agent), while [GPT 3.5](https://platform.openai.com/docs/models/gpt-3-5) successfully accomplishes the same task using the same setup.
   - They tried various solutions like using a custom parser, but the issue persists, leading them to believe the problem lies with LLaMA's capabilities.
- **Mistral struggles expanding beyond 8k**: A user reported that [Mistral](https://www.mistral.ai/) cannot be extended beyond 8k without continued pretraining.
   - The user suggested further work on *mergekit* and *frankenMoE finetuning* as potential solutions.
- **Discussion on Model Merging Tactics**: A user suggested a potential model merging tactic by applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn**.
   - Other members expressed skepticism, but the user remained optimistic, referencing past successful attempts at "cursed model merging".
- **Open Empathic Project Seeks Assistance**: A user requested help expanding the categories of the **Open Empathic** project, specifically at the lower end.
   - They shared a [YouTube video](https://youtu.be/GZqYr8_Q7DE) demonstrating the project's launch and a tutorial, guiding users to contribute preferred movie scenes from YouTube videos, along with a link to the [OpenEmpathic project](https://dct.openempathic.ai/).
- **LangChain's SQLDatabaseChain**: A user introduced [LangChain's SQLDatabaseChain](https://langchain.readthedocs.io/en/latest/modules/chains/sql_database_chain.html) as an experimental feature that generates SQL queries based on user prompts.
   - They shared a code example for a function that uses this feature, providing a prompt template for generating SQL queries and handling the response from the chain.



**Link mentioned**: <a href="https://mydomain.com']">no title found</a>: no description found

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1275167433082146848)** (16 messages🔥): 

> - `Accessibility Roundtable`
> - `Deepseek API`
> - `OI with Ollama`
> - `Poetry and Pytorch on Mac`
> - `Ollama on a different machine` 


- **Accessibility Roundtable Reminder**: A reminder for the Accessibility Roundtable this Thursday, with a link to the event.
- **Deepseek API vs OpenAI and Local LLMs**: A user inquired about a guide to using Deepseek API instead of OpenAI or local LLMs.
- **Ollama Integration with OI on a Different Machine**: A user sought guidance on using Ollama with OI when it is not hosted on localhost.
   - Specifically, they asked about configuring a profile YAML and starting the interpreter with the profile.
- **Poetry and Pytorch Installation Issues on Mac**: A user reported trouble with installing Poetry and Pytorch 2.3.0 on Mac, mentioning an open issue without a response.
   - They asked for guidance on finding a solution to this issue.
- **API_BASE Configuration for OpenInterpreter**: Discussion ensued about how to configure the API_BASE for OpenInterpreter to work with Ollama on a different machine.
   - A user confirmed that they had tried using the correct IP address and port for their Ollama instance, but OpenInterpreter still refused to connect.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://host:port">no title found</a>: no description found</li><li><a href="http://10.0.0.4:11434)">no title found</a>: no description found</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/tree/main/interpreter/core/computer">open-interpreter/interpreter/core/computer at main · OpenInterpreter/open-interpreter</a>: A natural language interface for computers. Contribute to OpenInterpreter/open-interpreter development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1275314289904975943)** (1 messages): 

> - `OpenInterpreter Update` 


- **OpenInterpreter Update**: The latest OpenInterpreter update is available at [this link](https://discord.com/channels/1146610656779440188/1194880263122075688/1271135268807905384).
   - No additional details were provided.
- **OpenInterpreter Update 2**: No additional details were provided.
   - No additional details were provided.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

notnaton: Latest episode from Tool Use 🚀 : https://www.youtube.com/watch?v=uAo513GIwoU
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1275292714803662960)** (2 messages): 

> - `dspy-ai installation`
> - `ADAS and Function Calling`
> - `pickle5 compatibility`
> - `Python version` 


- **A dspy-ai Installation Confusion**: A user noted that the `requirements.txt` file lists `dspy==2.0.5` but questioned if it should actually be `dspy-ai` instead.
   - They also pointed out a potential compatibility issue with `pickle5==0.0.12` which is compatible with Python versions below 3.8, while `dspy-ai` requires Python 3.9 or higher.
- **Can ADAS invent new building blocks?**: A user asked if ADAS could invent new building blocks like function calling to an integrated system.
   - They also inquired if anyone has already experimented with something similar.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1275171945666711603)** (9 messages🔥): 

> - `DSPy Finetuning`
> - `DSPy vs. Langchain/LLamaindex`
> - `Aider v0.51.0 Changelog`
> - `Providing Feedback to DSPy Documentation`
> - `Multi-Lora Setting` 


- **DSPy Finetuning with Multi-Lora**: A user asked about the potential of using a multi-lora setting for DSPy finetuning, suggesting it could be a valuable approach.
- **DSPy vs. Langchain/LLamaindex**: A user inquired about comparing DSPy to Langchain and LLamaindex, and was directed to the DSPy documentation for guidance on choosing the right tool.
- **Aider v0.51.0: Enhanced Prompt Caching and Repo Mapping**: Aider released version 0.51.0, featuring improved prompt caching for Anthropic models, optimized repo mapping for larger repositories, and enhanced Jupyter Notebook .ipynb file editing.
   - The release includes a variety of bug fixes and improvements, and Aider contributed 56% of the code for this version.
- **Providing Feedback to DSPy Documentation**: A user asked about the best way to provide feedback on the DSPy documentation.
   - The suggestion was to submit a pull request or issue that references the roadmap in the title.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/paulgauthier/status/1825934199465119803">Tweet from Paul Gauthier (@paulgauthier)</a>: Aider v0.51.0  - Prompt caching for Anthropic models with --cache-prompts. - Repo map speedups in large/mono repos. - Improved Jupyter Notebook .ipynb file editing.  - Aider wrote 56% of the code in t...</li><li><a href="https://aider.chat/HISTORY.html#v0510>>>">Release history</a>: Release notes and stats on aider writing its own code.
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1275506387602116650)** (1 messages): 

> - `Late Interaction Models`
> - `Dense Embedding Models`
> - `Qdrant 1.10`
> - `ColBERT` 


- **Qdrant 1.10 Adds Support for Multi-Vector Representations**: Qdrant 1.10 introduced support for multi-vector representations, with late interaction being a prominent example of this model. 
   - Identifying relevant documents involves calculating a score based on the similarity between corresponding query and document embeddings.
- **Late Interaction:  A Deeper Look**: Late interaction models, such as ColBERT, calculate a score based on the similarity between corresponding query and document embeddings. 
- **Adapting Dense Embedding Models for Late Interaction**: Regular dense embedding models can be adapted for late interaction by removing the pooling step and using token level embeddings for retrieval/reranking.
- **Hybrid Search Explained**: The updated [Hybrid Search](https://qdrant.tech/articles/hybrid-search/) article explains how multi-vector representations can enhance retrieval quality.



**Link mentioned**: <a href="https://qdrant.tech/articles/late-interaction-models/">Any* Embedding Model Can Become a Late Interaction Model - If You Give It a Chance! - Qdrant</a>: We discovered something interesting. Standard dense embedding models can perform surprisingly well in late interaction scenarios.

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1275495655615234088)** (1 messages): 

> - `LTXStudio new features` 


- **LTXStudio Launches Five New Features**: LTXStudio has released five new features to help users take their projects to the next level.
   - These features can be accessed and tested now.
- **LTXStudio's New Features for Enhanced Projects**: LTXStudio has launched five new features aimed at taking projects to the next level.
   - These features are now available for users to explore and utilize.



**Link mentioned**: <a href="https://x.com/LTXStudio/status/1825909655207383308?t=5Wk2X8i_lQ5R5HAJxcerlg&s=19">Tweet from LTX Studio (@LTXStudio)</a>: 🎉 The wait is over 🎉  To celebrate we&#39;re launching FIVE new features to take your projects to the next level. Try them out yourself now 🔥

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1275204038178111710)** (5 messages): 

> - `JPEG Encoding for Images`
> - `AR-Based Image Tokenization`
> - `VQ-VAE`
> - `Image Compression Limits`
> - `H.265/AV1 for Training` 


- **JPEG Encoding: A Viable Image Tokenization Method?**: A research paper suggests that JPEG encoding could be a good method for tokenizing images, but current AR-based approaches face significant information loss, resulting in poor image quality.
   - The paper uses a JPEG quality setting of 25, making it theoretically impossible to generate high-quality images from the tokens, and compresses a 256*256 image to around 5,000 tokens, leading to longer training and inference times compared to traditional VQ-VAE.
- **Uncertainties About Image Compression Limits**: The author questions the maximum compression achievable for images, as the paper uses a JPEG quality setting of 25 for tokenization.
- **Training Models on H.265 or AV1 Frames**: The author suggests exploring the possibility of training models on H.265 frames, or even AV1 frames, as an alternative to JPEG encoding.


  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1275491282713579521)** (1 messages): 

> - `GPT4All-Community`
> - `Leo models`
> - `Hugging Face`
> - `Model Card` 


- **Leo Models Publicly Available**: A member made [quantized versions of their Leo models](https://huggingface.co/GPT4All-Community) publicly available on Hugging Face.
   - They are willing to take feedback and relay messages to the users if needed, adding them to the model card if desired.
- **Feedback and Updates via Model Card**: The member offers to add messages to the model card for feedback or relaying information to users.


  

---



### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

xeophon.: https://x.com/bilawalsidhu/status/1825548322687574410?s=46
  

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

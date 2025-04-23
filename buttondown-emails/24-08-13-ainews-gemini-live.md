---
id: 9afa131e-b7a0-448b-a184-550ac7f96ccf
title: '[AINews] Gemini Live'
date: '2024-08-14T01:23:26.876396Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-gemini-live
---

<!-- buttondown-editor-mode: plaintext -->**Lots of little $20/month subscriptions for everything in your life are all you need.**

> AI News for 8/12/2024-8/13/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**253** channels, and **2423** messages) for you. Estimated reading time saved (at 200wpm): **244 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

As promised at [Google I/O](https://buttondown.email/ainews/archive/ainews-google-io-in-60-seconds/), Gemini Live launched in Android today, for Gemini Advanced subscribers, as part of the #MadeByGoogle Pixel 9 launch event. With sympathies to the [poor presenter](https://x.com/durreadan01/status/1823430521768304674) who had 2 demo failures onstage:

 ![image.png](https://assets.buttondown.email/images/7e351a3f-0f0d-4793-93ce-5b1974595c25.png?w=960&fit=max) 

The [embargoed media reviews of Gemini Live](https://www.theverge.com/2024/8/13/24219736/gemini-live-hands-on-pixel-event) have been cautiously positive. It will have "[extensions](https://support.google.com/gemini/answer/13695044?visit_id=638591951502121215-2420806349&p=more_extensions&rd=1)" that are integrations with your Google Workspace (Gmail, Docs, Drive), YouTube, Google Maps, and other Google properties. 

The important thing is Google started the rollout of it today (though we still [cannot locate anyone](https://www.reddit.com/r/singularity/comments/1erdr0t/meet_gemini_live_a_new_way_to_have_more_natural/) with a live recording of it as of 5pm PT) vs a still-indeterminate date for ChatGPT's Advanced Voice Mode. Gemini Live will also come to iOS subscribers at a future point.

The company also shared demos of Gemini Live with [Pixel Buds Pro 2](https://x.com/greengart/status/1823444923573731411) to people in the audience and [with the WSJ](https://x.com/JoannaStern/status/1823429729870868676). For those that care about the Pixel 9, there are also notable image AI integrations with the Add Me photo feature and the Magic Editor.

https://www.youtube.com/watch?v=KoN_bcDmhR4

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

- Anthropic released Genie, a new AI software engineering system achieving state-of-the-art performance on SWE-Bench with 30.08%, a 57% improvement over previous models. Key aspects include reasoning datasets, agentic systems with planning and execution abilities, and self-improvement capabilities. [@omarsar0](https://twitter.com/omarsar0/status/1823118952362278962)

- Falcon Mamba, a new 7B open-access model by TII, was released. It's an attention-free model that can scale to arbitrary sequence lengths and has strong metrics compared to similar-sized models. [@osanseviero](https://twitter.com/osanseviero/status/1823000588029743324)

- Researchers benchmarked 13 popular open-source and commercial models on context lengths from 2k to 125k, finding that long context doesn't always help with Retrieval-Augmented Generation (RAG). Performance of most generation models decreases above a certain context size. [@DbrxMosaicAI](https://twitter.com/DbrxMosaicAI/status/1823129597288046979)

**AI Tools and Applications**

- Supabase launched an AI-based Postgres service, described as the "ChatGPT of databases". It allows users to build and launch databases, create charts, generate embeddings, and more. The tool is 100% open source. [@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1823020725042639243)

- Perplexity AI announced a partnership with Polymarket, integrating real-time probability predictions for events like election outcomes and market trends into their search results. [@perplexity_ai](https://twitter.com/perplexity_ai/status/1823029449534615705)

- A tutorial on building a multimodal recipe recommender using Qdrant, LlamaIndex, and Gemini was shared, demonstrating how to ingest YouTube videos and index both text and image chunks. [@llama_index](https://twitter.com/llama_index/status/1823145827042468125)

**AI Engineering Insights**

- An OpenAI engineer shared insights on success in the field, emphasizing the importance of thoroughly debugging and understanding code, and a willingness to work hard to complete tasks. [@_jasonwei](https://twitter.com/_jasonwei/status/1823067805748728051)

- The connection between matrices and graphs in linear algebra was discussed, highlighting how this relationship provides insights into nonnegative matrices and strongly connected components. [@svpino](https://twitter.com/svpino/status/1822966303642308903)

- Keras 3.5.0 was released with first-class Hugging Face Hub integration, allowing direct saving and loading of models to/from the Hub. The update also includes distribution API improvements and new ops supporting TensorFlow, PyTorch, and JAX. [@fchollet](https://twitter.com/fchollet/status/1823098449883230341)

**AI Ethics and Regulation**

- Discussions around AI regulation and its potential impact on innovation were highlighted, with some arguing that premature regulation could hinder progress towards beneficial AI applications. [@bindureddy](https://twitter.com/bindureddy/status/1823095005206261835)

- Concerns were raised about the effectiveness of AI "business strategy decision support" startups, with arguments that their value is not easily measurable or trustable by customers. [@saranormous](https://twitter.com/saranormous/status/1823076401496625164)

**AI Community and Events**

- The Google DeepMind podcast announced its third season, exploring topics such as the differences between chatbots and agents, AI's role in creativity, and potential life scenarios after AGI is achieved. [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1822997510727598585)

- An AI Python for Beginners course taught by Andrew Ng was announced, designed to help both aspiring developers and professionals leverage AI to boost productivity and automate tasks. [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1823004343278219378)

**Memes and Humor**

- Various humorous tweets and memes related to AI and technology were shared, including jokes about AI model names and capabilities. [@swyx](https://twitter.com/swyx/status/1823122765584683248)

This summary captures the main themes and discussions from the provided tweets, focusing on recent developments in AI models, tools, applications, and the broader implications for AI engineering and the tech industry.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Advanced Quantization and Model Optimization Techniques**

- **[Llama-3.1 70B 4-bit HQQ/calibrated quantized model: 99%+ in all benchmarks in lm-eval relative performance to FP16 and similar inference speed to fp16 ( 10 toks/sec in A100 ).](https://huggingface.co/mobiuslabsgmbh/Llama-3.1-70b-instruct_4bitgs64_hqq)** ([Score: 91, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1eqfsa4/llama31_70b_4bit_hqqcalibrated_quantized_model_99/)): The **Llama-3.1 70B** model has been successfully quantized to **4-bit** using **HQQ/calibrated quantization**, achieving over **99% relative performance** compared to FP16 across all benchmarks in lm-eval. This quantized version maintains a similar inference speed to FP16, processing approximately **10 tokens per second on an A100 GPU**. The achievement demonstrates significant progress in model compression while preserving performance, potentially enabling more efficient deployment of large language models.

- **Why is unsloth so efficient?** ([Score: 94, Comments: 35](https://reddit.com//r/LocalLLaMA/comments/1eqdox0/why_is_unsloth_so_efficient/)): Unsloth demonstrates remarkable **efficiency** in handling **32k text length** for summarization tasks on **limited GPU memory**. The user reports successfully training a model on an **L40S 48GB GPU** using Unsloth, while traditional methods like **transformers llama2** with **qlora**, **4bit**, and **bf16** techniques fail to fit on the same hardware. The significant performance boost is attributed to Unsloth's use of **Triton**, though the exact mechanisms remain unclear to the user.

- **[Pre-training an LLM in 9 days 😱😱😱](https://arxiv.org/abs/2408.03506)** ([Score: 216, Comments: 53](https://reddit.com//r/LocalLLaMA/comments/1eqakjc/pretraining_an_llm_in_9_days/)): Researchers at **Hugging Face** and **Google** have developed a method to pre-train a **1.3B parameter language model** in just **9 days** using **16 A100 GPUs**. The technique, called **Retro-GPT**, combines **retrieval-augmented language modeling** with **efficient pre-training strategies** to achieve comparable performance to models trained for much longer, potentially revolutionizing the speed and cost-effectiveness of LLM development.

**Theme 2. Open-source Contributions to LLM Development**

- **[An extensive open source collection of RAG implementations with many different strategies](https://github.com/NirDiamant/RAG_Techniques)** ([Score: 91, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1eqec8v/an_extensive_open_source_collection_of_rag/)): The post introduces an **open-source repository** featuring a comprehensive collection of **17 different Retrieval-Augmented Generation (RAG) strategies**, complete with **tutorials and visualizations**. The author encourages community engagement, inviting users to open issues, suggest additional strategies, and utilize the resource for learning and reference purposes.

- **Falcon Mamba 7B from TII (Technology Innovation Institute TII - UAE)** ([Score: 87, Comments: 18](https://reddit.com//r/LocalLLaMA/comments/1eqaad2/falcon_mamba_7b_from_tii_technology_innovation/)): The **Technology Innovation Institute (TII)** in the UAE has released **Falcon Mamba 7B**, an open-source **State Space Language Model (SSLM)** combining the Falcon architecture with Mamba's state space sequence modeling. The model, available on **Hugging Face**, comes with a model card, collection, and playground, allowing users to explore and experiment with this new AI technology.
    - Users tested **Falcon Mamba 7B**, reporting **mixed results**. One user found it **"very very very poor"** for a Product Requirements Document task, with responses becoming generic and disorganized.
    - The model's performance was questioned, with some users finding it **worse than Llama and Mistral** models despite claims of superiority. Testing with various prompts yielded disappointing results.
    - Some users expressed **skepticism** towards Falcon models based on past negative experiences, suggesting a potential pattern of underperformance in the Falcon series.


## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Model Releases and Capabilities**

- **Speculation about new GPT-4 model**: A post on r/singularity claims ChatGPT mentioned a ["new GPT-4o model out since last week"](https://www.reddit.com/r/singularity/comments/1eqpy8o/new_gpt4o_model_out_since_last_week_chatgpt/), generating discussion about potential new OpenAI releases.

- **Flux image generation model**: Several posts discuss the capabilities of the new Flux image generation model:
  - [Impressive impressionist landscape generation](https://www.reddit.com/r/StableDiffusion/comments/1eqa8ds/first_image_is_how_an_impressionist_landscape/) using a custom LoRA trained on 5000 images
  - [Attempts at generating anatomically correct nude images](https://www.reddit.com/r/StableDiffusion/comments/1eq8268/a_pretty_rough_first_attempt_at_a_combo/) using a custom LoRA
  - [Creative ad concept generation](https://www.reddit.com/r/StableDiffusion/comments/1eqi9wj/flux_nuke_your_thirst/) for fictional products

**AI-Generated Media**

- **AI-generated video with synthetic voice**: A [demo video](https://www.reddit.com/r/StableDiffusion/comments/1eqwh1p/added_voice_to_flux_videos_through_rendernet/) shows Flux-generated images animated and paired with AI-generated voice, though commenters note issues with lip sync and voice quality.

**Autonomous Vehicles**

- **Waymo self-driving car issues**: A [video post](https://www.reddit.com/r/singularity/comments/1eqoxho/waymo_cars_being_clueless_from_their_spawn/) shows Waymo autonomous vehicles having difficulties navigating from their starting point, sparking discussion on current limitations.

**AI and Society**

- **AI companions and relationships**: A [controversial meme post](https://www.reddit.com/r/singularity/comments/1eqz3sb/real_or/) sparked debate about the potential impact of AI companions on human relationships and societal dynamics.

---

# AI Discord Recap

> A summary of Summaries of Summaries by GPT4O (gpt-4o-2024-05-13)

**1. Model Performance and Benchmarking**

- **Uncensored Model Outperforms Meta Instruct**: An uncensored model tuned to retain the intelligence of the original Meta Instruct model has been released and has outperformed the original model on the **[LLM Leaderboard 2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)**.
  - The model's performance sparked discussions about the trade-offs between censorship and utility, with many users praising its ability to handle a wider range of inputs.
- **Mistral Large: The Current Champion?**: A member found **[Mistral Large 2](https://huggingface.co/mradermacher/Tiger-Gemma-9B-v1-i1-GGUF)** to be the best LLM right now, outcompeting **Claude 3.5 Sonnet** for difficult novel problems.
  - However, **Gemini Flash** undercut **OpenAI 4o mini** severely in price, but **OpenAI 4o** was less expensive than **Mistral Large**.
- **Google's Gemini Live: It's here, it's now, it's not free**: **[Gemini Live](https://www.digitalheresy.com/p/strawberry)** is now available to **Advanced Subscribers**, offering conversational overlay on Android and more connected apps.
  - Many users said that it is an improvement over the old voice mode, but is only available to paid users and lacks live video functionality.


**2. GPU and Hardware Discussions**

- **GPU Wars - A100 vs A6000**: Members discussed the pros and cons of **A100 vs A6000 GPUs**, with one member noting the A6000's great price/VRAM ratio and its lack of limitations compared to 24GB cards.
  - The discussion highlighted the importance of **VRAM** and cost-efficiency for large model training and inference.
- **Stable Diffusion Installation Woes**: A user reported difficulties installing **Stable Diffusion**, encountering issues with **CUDA installation** and finding their token on Hugging Face.
  - Another user provided guidance on generating a token through the profile settings menu and installing CUDA correctly.
- **TorchAO presentation at Cohere for AI**: **[Charles Hernandez](https://tinyurl.com/C4AICommunityApp)** from PyTorch Architecture Optimization will be presenting on TorchAO and quantization at the ml-efficiency group at Cohere For AI.
  - The event is hosted by **@Sree_Harsha_N** and attendees can join Cohere For AI through the provided link.


**3. Fine-tuning and Optimization Techniques**

- **Model Fine-Tuning Tips and Tricks**: Discussion revolved around fine-tuning a **Phi3 model** and whether to use **LoRA** or full fine-tuning, with one member suggesting **RAG** as a potential solution.
  - Users shared experiences and best practices, emphasizing the importance of choosing the right fine-tuning strategy for different models.
- **TransformerDecoderLayer Refactor PR**: A PR has been submitted to refactor the **TransformerDecoderLayer**, touching many files and making core changes in **modules/attention.py** and **modules/transformer.py**.
  - This PR implements **RFC #1211**, aiming to improve the TransformerDecoderLayer architecture.
- **PyTorch Full FP16: Is it possible?**: A user asked if full FP16 with loss/grad scaling is possible with **PyTorch core**, specifically when fine-tuning a large-ish model from **Fairseq**.
  - They tried using **torch.GradScaler()** and casting the model to FP16 without **torch.autocast('cuda', torch.float16)**, but got an error 'ValueError: Attempting to unscale FP16 gradients.'


**4. UI/UX Issues in AI Platforms**

- **Perplexity's UI/UX issues**: Users reported several UI/UX issues including missing buttons and a disappearing prompt field, leading to difficulties in interacting with the platform.
  - These bugs were reported across both the web and iOS versions of Perplexity, causing significant user frustration and hindering their ability to effectively utilize the platform.
- **LLM Studio's Model Explorer is Down**: Several members reported that **HuggingFace**, which powers the **LM Studio Model Explorer**, is down.
  - The site was confirmed to be inaccessible for several hours, with connectivity issues reported across various locations.
- **Perplexity's Website Stability Concerns**: Users reported a significant decline in website stability, citing issues with sporadic search behavior, forgetting context, and interface bugs on both web and iOS versions.
  - These issues raised concerns about the reliability and user experience provided by Perplexity.


**5. Open-Source AI Frameworks and Community Efforts**

- **Rust GPU Transitions to Community Ownership**: The **[Rust GPU](https://rust-gpu.github.io)** project, previously under **Embark Studios**, is now community-owned under the **Rust GPU GitHub organization**.
  - This transition marks the beginning of a broader strategy aimed at revitalizing, unifying, and standardizing GPU programming in Rust.
- **Open Interpreter for Anything to Anything**: Use Open Interpreter to convert any type of data into any other format.
  - This is possible by using the 'Convert Anything' tool, which harnesses the power of Open Interpreter.
- **Cohere For AI research lab**: **[Cohere For AI](https://cohere.com/research)** is a non-profit research lab that seeks to solve complex machine learning problems.
  - They support fundamental research exploring the unknown, and are focused on creating more points of entry into machine learning research.




---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Pro Early Access**: Early access to the Unsloth Pro version is currently being given to trusted members of the Unsloth community.
- **A100 vs A6000 GPU Showdown**: Members discussed the pros and cons of A100 vs A6000 GPUs, with one member noting the A6000's great price/VRAM ratio and its lack of limitations compared to 24GB cards.
- **Uncensored Model Tops the Charts**: An uncensored model tuned to retain the intelligence of the original Meta Instruct model has been released and has outperformed the original model on the LLM Leaderboard 2.
- **Dolphin Model Suffers From Censorship**: One member reported that the Dolphin 3.1 model fails the most basic requests and refuses them, possibly due to its heavy censorship.
- **Fine-tuning for AI Engineers**: Discussion revolved around fine-tuning a Phi3 model and whether to use LoRA or full fine-tuning, with one member suggesting RAG as a potential solution.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **TorchAO Presentation at Cohere For AI**: Charles Hernandez from PyTorch Architecture Optimization will be presenting on TorchAO and quantization at the ml-efficiency group at Cohere For AI on August 16th, 2000 CEST.
   - This event is hosted by @Sree_Harsha_N and attendees can join Cohere For AI through the link [https://tinyurl.com/C4AICommunityApp](https://tinyurl.com/C4AICommunityApp).
- **CPU matmul Optimization Battle**: A user is attempting to write a tiling-based matmul in Zig but is having difficulty achieving optimal performance.
   - They received advice on exploring cache-aware loop reordering and the potential for using SIMD instructions, and also compared the performance to GGML and NumPy, which leverages optimized BLAS implementations for incredibly fast results.
- **FP16 Weights and CPU Performance**: A user asked about handling FP16 weights on the CPU, noting that recent models generally use BF16.
   - They were advised to convert the FP16 weights to BF16 or FP32, with FP32 leading to no accuracy loss but potentially slower inference and exploring converting tensors at runtime from FP16 to FP32 to potentially improve performance.
- **PyTorch Full FP16: Is it Really Possible?**: A user asked if full FP16 with loss/grad scaling is possible with PyTorch core, specifically when fine-tuning a large-ish model from Fairseq.
   - They attempted to use `torch.GradScaler()` and cast the model to FP16 without `torch.autocast('cuda', torch.float16)` but got an error "ValueError: Attempting to unscale FP16 gradients."
- **torch.compile: The Missing Manual**: A new PyTorch document titled "torch.compile: The Missing Manual" was shared along with a YouTube video.
   - The document and video are available at [https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab) and [https://www.youtube.com/live/rew5CSUaIXg?si=zwbubwKcaiVKqqpf](https://www.youtube.com/live/rew5CSUaIXg?si=zwbubwKcaiVKqqpf), respectively, and provide detailed information on utilizing `torch.compile`.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Vision Adapters: The Key to Vision Models**: Only specific LLM models have vision adapters, most of them are going by name "LLaVa" or "obsidian".
   - The "VISION ADAPTER" is a crucial component for vision models; without it, the error you shared will pop up.
- **Mistral Large: The Current Champion?**: A member found **Mistral Large 2** to be the best LLM right now, outcompeting **Claude 3.5 Sonnet** for difficult novel problems.
   - However, the member also noted that **Gemini Flash** undercut **OpenAI 4o mini** severely in price, but **OpenAI 4o** was less expensive than **Mistral Large**.
- **LLM Studio's Model Explorer is Down**: Several members reported that HuggingFace, which powers the **LM Studio Model Explorer**, is down.
   - The site was confirmed to be inaccessible for several hours, with connectivity issues reported across various locations.
- **Llama 3.1 Performance Issues**: A user reported that their **Llama 3 8B model** is now running at only 3 tok/s, compared to 15 tok/s before a recent update.
   - The user checked their GPU offload settings and reset them to default, but the problem persists; the issue appears to be related to a change in the recent update.
- **LLM Output Length Control**: A member is looking for ways to restrict the output length of responses, as some models tend to output whole paragraphs even when instructed to provide a single sentence.
   - While system prompts can be modified, the member found that 8B models, specifically **Meta-Llama-3.1-8B-Instruct-GGUFI**, are not the best at following precise instructions.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Google Rolls Out Gemini Live, But Not for Everyone**: **Gemini Live** is now available to **Advanced Subscribers**, offering conversational overlay on Android and more connected apps. 
   - Many users said that it is an improvement over the old voice mode, but is only available to paid users and lacks live video functionality.
- **Strawberry: Marketing Genius or OpenAI's New Face?**: The discussion of a mysterious user named "Strawberry" with a string of emojis sparked speculation about a possible connection to OpenAI or Sam Altman.
   - Users remarked on how the strawberry emojis, linked to Sam Altman's image of holding strawberries, were a clever marketing strategy, successfully engaging users in conversation.
- **Project Astra's Long-Awaited Arrival**: The announcement of **Gemini Live** hinted at **Project Astra**, but many users were disappointed by the lack of further development.
   - One user even drew a comparison to a **Microsoft recall**, suggesting that people are skeptical about the product's release due to security concerns.
- **LLMs:  Not a One-Size-Fits-All Solution**: Some users expressed skepticism about LLMs being the solution to every problem, especially when it comes to tasks like math, database, and even waifu roleplay.
   - Other users emphasized that tokenization is still a fundamental weakness, and LLMs require a more strategic approach rather than relying on brute force tokenization to solve complex problems.
- **ChatGPT's Website Restrictions:  A Persistent Issue**: A member asked about getting ChatGPT to access a website and retrieve an article, but another member noted that ChatGPT might be blocked from crawling or hallucinating website content.
   - One user asked if anyone has attempted to use the term "web browser GPT" as a possible workaround.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's UI/UX Bugs**: Users encountered UI/UX issues including missing buttons and a disappearing prompt field, leading to difficulties in interacting with the platform.
   - These bugs were reported across both the web and iOS versions of Perplexity, causing significant user frustration and hindering their ability to effectively utilize the platform.
- **Sonar Huge: New Model, New Problems**: The new model "Sonar Huge" replaced the Llama 3.1 405B model in Perplexity Pro.
   - However, users observed that the new model was slow and failed to adhere to user profile prompts, prompting concerns about its effectiveness and performance.
- **Perplexity's Website Stability Issues**: Users reported a significant decline in the website's stability, with issues like sporadic search behavior, forgetting context, and various interface bugs.
   - These issues were observed on both web and iOS versions, raising concerns about the reliability and user experience provided by Perplexity.
- **Perplexity's Success Team Takes Note**: Perplexity's Success Team acknowledged receiving user feedback on the recent bugs and glitches experienced in the platform.
   - They indicated awareness of the reported issues and their impact on user experience, hinting at potential future solutions and improvements.
- **Feature Implementation Delays at Perplexity**: A user expressed frustration over the prolonged wait time for feature implementation.
   - They highlighted the discrepancy between promised features and the actual rollout pace, emphasizing the importance of faster development and delivery to meet user expectations.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability AI's SXSW Panel Proposal**: Stability AI CEO Prem Akkaraju and tech influencer Kara Swisher will discuss the importance of open AI models and the role of government in regulating their impact at SXSW.
   - The panel will explore the opportunities and risks of AI, including job displacement, disinformation, CSAM, and IP rights, and will be available to view on PanelPicker® at [PanelPicker | SXSW Conference & Festivals](http://panelpicker.sxsw.com/vote/153232).
- **Google Colab Runtime Stops Working**: A user encountered issues with their Google Colab runtime stopping prematurely.
   - Another user suggested switching to Kaggle, which offers more resources and longer runtimes, providing a solution for longer AI experimentation.
- **Stable Diffusion Installation and CUDA Challenges**: A user faced difficulties installing Stable Diffusion due to issues with CUDA installation and locating their Hugging Face token.
   - Another user provided guidance on generating a token through the Hugging Face profile settings menu and correctly installing CUDA, offering a solution to the user's challenges.
- **Model Merging Discussion**: A user suggested using the difference between UltraChat and base Mistral to improve Mistral-Yarn as a potential model merging tactic.
   - While some users expressed skepticism, the original user remained optimistic, citing successful past attempts at model merging, showcasing potential advancements in AI model development.
- **Flux Realism for Face Swaps**: A user sought alternative solutions to achieve realistic face swaps after experimenting with fal.ai, which produced cartoonish results.
   - Another user suggested using Flux, as it is capable of training on logos and accurately placing them onto images, providing a potential solution for the user's face swap goals.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Flash 1.5 Price Drop**: The input token costs for **Gemini Flash 1.5** have decreased by **78%** and the output token costs have decreased by **71%**.
   - This makes the model more accessible and affordable for a wider range of users.
- **GPT-4o Extended Early Access Launched**: Early access for **GPT-4o Extended** has launched through **OpenRouter**.
   - You can access it via this link: [https://x.com/OpenRouterAI/status/1823409123360432393](https://x.com/OpenRouterAI/status/1823409123360432393).
- **OpenRouter's Update Hurdle**: OpenRouter's update was blocked by the new 1:4 token:character ratio from Gemini, which doesn't map cleanly to the `max_tokens` parameter validation.
   - A user expressed frustration about the constantly changing token:character ratio and suggested switching to a per-token pricing system.
- **Euryale 70B Downtime**: A user reported that **Euryale 70B** was down for some users but not for them, prompting questions about any issues or error rates.
   - Further discussion revealed multiple instances of downtime, including a 10-minute outage due to an update and possible ongoing issues with location availability.
- **Model Performance Comparison**: Users compared the performance of **Groq 70b** and **Hyperbolic**, finding nearly identical results for the same prompt.
   - This led to a discussion about the impact of **FP8 quantization**, with some users noting that it makes a minimal difference in practice, but others pointing to potential degraded quality with certain providers.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo License's Catchy Clause**: The [Mojo License](https://www.mojo-lang.org/docs/license) prohibits the development of applications using the language for competitive activities.
   - However, it states that this rule does not apply to applications that become competitive after their initial release, but it is unclear how this clause will be applied.
- **Mojo Open-Sourcing Timeline Remains Unclear**: Users inquired about the timeline for open-sourcing the Mojo compiler.
   - The team confirmed that the compiler will be open-sourced eventually but did not provide a timeline, suggesting it may be a while before contributions can be made.
- **Mojo Development: Standard Library Focus**: The current focus of Mojo development is on building out the standard library.
   - Users are encouraged to contribute to the standard library, while work on the compiler is ongoing, but not yet open to contributions.
- **Stable Diffusion and Mojo: Memory Matters**: A user encountered a memory pressure issue running the Stable Diffusion Mojo ONNX example in WSL2, leading to the process being killed.
   - The user had 8GB allocated to WSL2, but the team advised doubling it as Stable Diffusion 1.5 is approximately 4GB, requiring more memory for both the model and its optimization processes.
- **Java by Microsoft: A Blast from the Past**: One member argued that 'Java by Microsoft' was unnecessary and could have been avoided, while another countered that it seemed crucial at the time.
   - The discussion acknowledged the emergence of newer solutions and the decline of 'Java by Microsoft' over time, highlighting its 20-year run and its relevance in the Microsoft marketshare.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere For AI Research Lab Expands**: **Cohere For AI** is a non-profit research lab focused on complex machine learning problems. They are creating more points of entry into machine learning research.
   - They support [fundamental research exploring the unknown](https://cohere.com/research).
- **Price Changes on Cohere's Website**: A user inquired about the **classify** feature's pricing, as it's no longer listed on the pricing page.
   - No response was provided.
- **JSONL Uploads Failing**: Users reported issues uploading JSONL datasets for fine-tuning.
   - Cohere support acknowledged the issue, stating it is under investigation and suggesting the API for dataset creation as a temporary solution.
- **Azure JSON Formatting Not Supported**: A member asked about structured output with `response_format` in Azure, but encountered an error.
   - It was confirmed that JSON formatting is not yet available on Azure.
- **Rerank Overview and Code Help**: A user asked for help with the Rerank Overview document, encountering issues with the provided code.
   - The issue was related to an outdated document, and a revised code snippet was provided. The user was also directed to the [relevant documentation](https://docs.cohere.com/reference/rerank) for further reference.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TransformerDecoderLayer Refactor Lands**: A PR has been submitted to refactor the TransformerDecoderLayer, touching many files and making core changes in modules/attention.py and modules/transformer.py.
   - This PR implements RFC #1211, aiming to improve the TransformerDecoderLayer architecture, and can be found here: [TransformerDecoderLayer Refactor](https://github.com/pytorch/torchtune/pull/1312).
- **DPO Preferred for RLHF**: There is a discussion about testing the HH RLHF builder with DPO or PPO, with DPO being preferred for preference datasets while PPO is dataset-agnostic.
   - The focus is on DPO, with the expectation of loss curves similar to normal SFT, and potential debugging needed for the HH RLHF builder, which may be addressed in a separate PR.
- **Torchtune WandB Issues Resolved**: A user encountered issues accessing WandB results for Torchtune, with access being granted after adding the user as a team member.
   - The user reported poor results with the default DPO config and turning gradient accumulation off, but later discovered it started working again, potentially due to a delay or some other factor.
- **Torchtune Performance with DPO**: There is a discussion about potential issues with the default DPO config causing poor performance in Torchtune.
   - The user suggested trying SIMPO (Stack Exchange Paired) and turning gradient accumulation back on, as having a balanced number of positive and negative examples in the batch can significantly improve loss.
- **PyTorch Conference: A Gathering of Minds**: There is a discussion about the upcoming PyTorch Conference, with links to the website and details on featured speakers.
   - You can find more information about the conference here: [PyTorch Conference](https://events.linuxfoundation.org/pytorch-conference/). There was also a mention of sneaking in a participant as an 'academic' for the conference, but this is potentially a joke.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Perplexity Pro's Reasoning Abilities**: A user noted that [Perplexity Pro](https://perplexity.ai/) has gotten *"crazy good at reasoning"* and is able to *"literally count letters"* like it *"ditched the tokenizer"*. 
   - They shared a link to [a GitHub repository](https://github.com/cognitivecomputations/grokadamw) that appears to be related to this topic.
- **Llama 3 MoE?**: A user asked if anyone has made a "MoE" version of Llama 3.
- **Grad Clipping Demystified**: A user asked about the functionality of grad clipping, specifically wondering what happens to gradients when they exceed the maximum value.
   - Another user explained that grad clipping essentially clips the gradient to a maximum value, preventing it from exploding during training.
- **OpenAI Benchmarks vs New Models**: A user shared their surprise at OpenAI releasing a benchmark instead of a new model.
   - They speculated that this might be a strategic move to steer the field towards better evaluation tools.
- **Axolotl's Capabilities**: A member noted that AutoGPTQ could do certain things, implying that Axolotl may be able to do so as well.
   - They were excited about the possibility of Axolotl replicating this capability.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Grok 2.0 Early Leak**: A member shared a link to a Tweet about [Grok 2.0 features and abilities](https://x.com/nima_owji/status/1823388838279922166), including image generation using the FLUX.1 model.
   - The tweet also noted that Grok 2.0 is better at coding, writing, and generating news.
- **Flux.1 Makes an Inflection Point**: A member mentioned that many Elon fan accounts predicted X would use MJ (presumably referring to a model), suggesting that Flux.1 may have made an inflection point in model usage.
   - The member questioned if Flux.1 is Schnellit's Pro model, given Elon's history.
- **Open-Source Image Annotation Search**: A member asked for recommendations for good open-source GUIs for annotating images quickly and efficiently.
   - The member specifically mentioned single-point annotations, straight-line annotations, and drawing polygonal segmentation masks.
- **Elon's Model Bluff**: A member discussed the possibility that Elon is using a development version of Grok and calling the bluff on weight licenses.
   - This member believes that Elon could potentially call this a "red-pill" version.
- **2D Pooling Success**: A user expresses surprise at how well 2D pooling works.
   - The user noted it was recommended by another user, and is currently verifying the efficacy of a new position encoding they believe they may have invented.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tensor Filtering Performance?**: A user asked for the fastest way to filter a Tensor, such as `t[t % 2 == 0]`, currently doing it by converting to list, filtering, and converting back to list.
   - A suggestion was made to use masking if computing something on a subset of the Tensor, but it was noted that the exact functionality is not possible yet.
- **Transcendental Folding Refactor Optimization**: A user proposed a refactor to only apply transcendental rewrite rules if the backend does not have a `code_for_op` for the `uop`.
   - The user implemented a `transcendental_folding` function and called it from `UOpGraph.__init__` but wasn't sure how this could be net negative lines, and asked what could be removed.
- **CUDA TIMEOUT ERROR - Resolved**: A user ran a script using `CLANG=1` and received a `RuntimeError: wait_result: 10000 ms TIMEOUT!` error.
   - The error occurred with the default runtime and was resolved by using `CUDA=1`, and the issue was potentially related to ##4562.
- **Nvidia FP8 PR Suggestions**: A user made suggestions on the Nvidia FP8 PR for **Tinygrad**.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Poe Partners with Agihouse for Hackathon**: Poe (@poe_platform) announced a partnership with Agihouse (@agihouse_org) for a "Previews Hackathon" to celebrate their expanded release.
   - The hackathon, hosted on [AGI House](https://app.agihouse.org/events/poe-previews-hackathon-20240817), invites creators to build innovative "in-chat generative UI experiences".
- **In-Chat UI is the Future**: The Poe Previews Hackathon encourages developers to create innovative and useful "in-chat generative UI experiences", highlighting the importance of user experience in generative AI.
   - The hackathon hopes to showcase the creativity and skill of its participants in a competitive environment.
- **Virtual Try On Feature Speeds up Training**: A member shared their experience building a virtual try-on feature, noting its effectiveness in speeding up training runs by storing extracted features.
   - The feature uses online preprocessing and stores extracted features in a document store table, allowing for efficient retrieval during training.
- **Flexible Virtual Try On Feature**: A member inquired about the specific features being extracted for the virtual try-on feature.
   - The member detailed the generic nature of the approach, successfully accommodating models of various sizes, demonstrating its flexibility in handling computational demands and model complexities.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Llama 3.1 8b Supports Structured Output**: A user confirmed that **Llama 3.1 8b** can produce structured output through tool use, having tested it directly with **llama.cpp**.
- **RAG Struggles With Technical Images**: A user is seeking advice on extracting information from images like electrical diagrams, maps, and voltage curves for **RAG** on technical documents.
   - They mentioned encountering difficulties with traditional methods, highlighting the need for capturing information not present in text form but visually interpretable by experts.
- **Next.js POST Request Misinterpreted as GET**: A user encountered a **405 Method Not Allowed** error when making a **POST request** from a **Next.js web app** running on **EC2** to a **FastAPI endpoint** on the same **EC2 instance**.
   - They observed the request being incorrectly interpreted as a **GET request** despite explicitly using the **POST method** in their **Next.js code**.
- **AWS pip install Issue Resolved**: A user resolved an issue with **pip install** on an **AWS system** by installing packages specifically for the **Unix-based environment**.
   - The problem arose from the virtual environment mistakenly emulating **Windows** during the **pip install** process, causing the issue.
- **Profundo Launches to Automate Research**: Profundo automates data collection, analysis, and reporting, enabling everyone to do deep research on topics they care about.
   - It minimizes errors and maximizes productivity, allowing users to focus on making informed decisions.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter in Obsidian**: A new YouTube series will demonstrate how to use Open Interpreter in the Obsidian note-taking app.
   - The series will focus on how the Open Interpreter plugin allows you to control your Obsidian vault, which could have major implications for how people work with knowledge. [Here's a link to Episode 0](https://www.youtube.com/watch?v=HjcPRoPfri0).
- **AI Agents in the Enterprise**: A user in the #general channel asked about the challenges of monitoring and governance of AI agents within large organizations.
   - The user invited anyone working on AI agents within an enterprise to share their experiences.
- **Screenless Personal Tutor for Kids**: A member in the #O1 channel proposed using Open Interpreter to create a screenless personal tutor for kids.
   - The member requested feedback and asked if anyone else was interested in collaborating on this project.
- **Convert Anything Tool**: The "Convert Anything" tool can be used to convert any type of data into any other format using Open Interpreter.
   - This tool harnesses the power of Open Interpreter and has potential for significant applications across various fields.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **SlimOrca Without Deduplication**: A user asked about a version of **SlimOrca** that has **soft prompting removed** and **no deduplication**, ideally including the code.
   - They also asked if anyone had experimented with fine-tuning (FT) on data with or without deduplication, and with or without soft prompting.
- **Fine-tuning with Deduplication**: The user inquired about the effects of fine-tuning (FT) with **soft prompting** versus without soft prompting.
   - They also inquired about the effects of fine-tuning (FT) on **deduplicated data** versus **non-deduplicated data**.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Building an Agentic Jupyter Notebook Automation System**: A member proposed constructing an agentic system to automate Jupyter Notebooks, aiming to create a pipeline that takes an existing notebook as input, modifies cells, and generates multiple variations.
   - They sought recommendations for libraries, cookbooks, or open-source projects that could serve as a starting point for this project, drawing inspiration from similar tools like Devin.
- **Automated Notebook Modifications and Validation**: The system should be able to intelligently replace specific cells within a Jupyter Notebook, generating diverse notebook versions based on these modifications.
   - Crucially, the system should possess an agentic quality, enabling it to validate its outputs and iteratively refine the modifications until it achieves the desired results.



---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1272632011063824524)** (167 messages🔥🔥): 

> - `Unsloth Pro`
> - `GPU choices`
> - `LLM Leaderboard results`
> - `Dolphin Model`
> - `Model fine-tuning` 


- **Unsloth Pro Early Access**: Early access to the Unsloth Pro version is currently being given to trusted members of the Unsloth community.
- **GPU Wars - A100 vs A6000**: Members discuss the pros and cons of A100 vs A6000 GPUs, with one member noting the A6000's great price/VRAM ratio and its lack of limitations compared to 24GB cards.
- **Uncensored Model Outperforms Meta Instruct**: An uncensored model tuned to retain the intelligence of the original Meta Instruct model has been released and has outperformed the original model on the LLM Leaderboard 2.
- **Dolphin Model Struggles with Censorship**: One member reported that the Dolphin 3.1 model fails the most basic requests and refuses them, possibly due to its heavy censorship.
- **Model Fine-Tuning Tips and Tricks**: Discussion revolves around fine-tuning a Phi3 model and whether to use LoRA or full fine-tuning, with one member suggesting RAG as a potential solution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard">Open LLM Leaderboard 2 - a Hugging Face Space by open-llm-leaderboard</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1823366649074094225">Tweet from Daniel Han (@danielhanchen)</a>: Found some issues with Llama 3.1&#39;s chat template:  1. Official repo adds 2x \n 2. Official repo does NOT strip / trim 3. Date format is %B not %b (not 3 letters) 4. Official repo has inconsistent ...</li><li><a href="https://huggingface.co/spaces/featherless-ai/try-this-model">HF&#39;s Missing Inference Widget - a Hugging Face Space by featherless-ai</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1272632104056000512)** (12 messages🔥): 

> - `Camping`
> - `Australia` 


- **Camping is bad**: A user lamented their experience of camping after returning with 6 mosquito bites, including one on their eyelid.
   - Another user chimed in, stating that "never go camping".
- **Australia is worse than camping**: One user said that "in australia there's horse shit everywhere", implying that it is worse than camping.
   - Others chimed in agreeing and adding that there are spiders as big as a dinner plate in Australia and that "everything wants to kill you".
- **Amazon Rainforest is worst of all**: One user said they "lived close to the Amazon rainforest", implying that it is worse than both camping and Australia.
   - This user also commented on a YouTube video titled: "Cosine Genie - SOTA AI Engineer Announcement", which describes Genie as the "best AI software engineer in the world by far".



**Link mentioned**: <a href="https://www.youtube.com/watch?v=NvmB7ngopOY">Cosine Genie - SOTA AI Engineer Announcement</a>: Genie is the best AI software engineer in the world by far - scoring 30% on the industry standard benchmark SWE-Bench we have beaten the previous SOTA scores...

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1272637676977258627)** (83 messages🔥🔥): 

> - `Unsloth model loading/saving`
> - `Llama 3.1 fine-tuning with Hindi`
> - `Model merging and HF hub`
> - `Unsloth with VLLM`
> - `Dataset creation` 


- **ModelFile not found after saving with Unsloth**: A user tried to save a model using Unsloth's `model.save_pretrained_gguf` method and noticed that the saved folder did not contain a `ModelFile`. 
   - Another user explained that the model file is saved as tensors, and the whole folder is needed for the config and architecture information, including json configs. This is a normal way for gguf files to be saved, split into several config files and tensor files.
- **Unsloth Finetuned Model Not Giving Hindi Summarization**: A user finetuned a Llama 3.1 8B model on Hindi summarization data and uploaded it to Hugging Face, but during inference, it either returned the input text or a summary in English. 
   - The user shared the code they used for inference and fine-tuning, and other users suggested that there may be issues with saving or loading custom tokenizers or that the merging on the hub might be combining layers in a strange way.
- **Using Unsloth with VLLM**: A user struggled to use a finetuned model with Unsloth using VLLM, sharing a Colab notebook with their code. 
   - Another user suggested using the vLLM documentation for troubleshooting, as it is known to be detailed and helpful.
- **Creating custom datasets**: A user asked for resources about creating custom datasets with their own information in formats like CSV or JSONL. 
   - Users suggested using Hugging Face datasets, creating the data manually, or using a larger model to generate data for them.
- **Unsloth memory usage issue**: A user encountered an issue where their LLaMA 3 8B Instruct model with Unsloth was consuming 300GB of physical memory but still encountering memory issues, causing the server to kill the model. 
   - Users suggested checking the available VRAM as the model likely needs more GPU memory to operate correctly.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/12pmRRQXunwvxeXxo97SUu5OuMaQ1IR3a?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://www.patched.codes/blog/a-comparative-study-of-fine-tuning-gpt-4o-mini-gemini-flash-1-5-and-llama-3-1-8b">A comparative study of fine-tuning GPT-4o-mini, Gemini Flash 1.5 and Llama-3.1-8B</a>: We compare fine-tuning GPT-4o-mini, Gemini Flash 1.5, and Llama-3.1-8B models using a custom vulnerability fixes dataset, with GPT-4o-mini showing the most significant improvement and setting a new st...</li><li><a href="https://www.deeplearning.ai/short-courses/finetuning-large-language-models/">Finetuning Large Language Models - DeepLearning.AI</a>: Master the basics of finetuning an LLM. Differentiate finetuning from prompt engineering and gain hands-on experience with real datasets.</li><li><a href="https://docs.vllm.ai/en/latest/">Welcome to vLLM! &#8212; vLLM</a>: no description found</li><li><a href="https://x.com/labenz/status/1822321840385048950">Tweet from Nathan Labenz (@labenz)</a>: Any inference platforms offer MoRA for Llama 3.1 models?  Seems like a big opportunity!  (Or… why not?)   cc @lqiao @FireworksAI_HQ   @tri_dao @togethercompute  @jefrankle @DbrxMosaicAI   Appreciate a...</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-12.-saving-the-model">How to Finetune Llama-3 and Export to Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama">How to Finetune Llama-3 and Export to Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama</li><li><a href="https://huggingface.co/datasets/yahma/alpaca-cleaned">yahma/alpaca-cleaned · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1272735526247731307)** (8 messages🔥): 

> - `Lexi model`
> - `LLM Leaderboard 2`
> - `Ahma-3B Instruct`
> - `Finnish-NLP/Ahma-3B`
> - `Finnish language model` 


- **Lexi Model Beats Original Instruct**: The LLM Leaderboard 2 results for an uncensored version of Lexi, a finetuned Llama-3.1-8B model, have been released.
   - Lexi not only retains the original instruct, but it actually beats it in performance.
- **Ahma-3B Instruct - Finnish Language Model**: The instruction-finetuned version of Ahma-3B, a Llama-based model pretrained from scratch in Finnish, has been released on Hugging Face.
   - Ahma-3B Instruct is trained to follow instructions in Finnish, and the base model was pretrained on 139 billion Finnish tokens.
- **Training Ahma-3B Instruct**: The training process for Ahma-3B Instruct involved translating and synthesizing single- and multi-turn data, using ClusterClipping based sampling and selection.
   - This was followed by Supervised Fine-Tuning (SFT) with Qlora using the Unsloth framework, and a fine-tuning step with DPO (Direct Preference Optimization) with a beta of 0.1.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored">Orenguteng/Llama-3.1-8B-Lexi-Uncensored · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Finnish-NLP/Ahma-3B-Instruct">Finnish-NLP/Ahma-3B-Instruct · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1272827506298196042)** (5 messages): 

> - `1.5-Pints`
> - `Tree Attention`
> - `Mistral`
> - `Llama 2`
> - `OpenELM` 


- **1.5-Pints: A Compute-Efficient Language Model**: A new language model called "1.5-Pints" was presented, which was pre-trained in just 9 days using a compute-efficient approach.
   - This model outperforms state-of-the-art models like Apple's OpenELM and Microsoft's Phi in instruction-following tasks, as measured by MT-Bench.
- **Tree Attention: Impressive Long Context Performance**: A research paper discussed the significant improvements in very-long context performance achieved through the use of tree attention.
   - The paper, available at [https://arxiv.org/pdf/2408.04093](https://arxiv.org/pdf/2408.04093), suggests that tree attention is a promising approach for handling long contexts.
- **1.5-Pints Architecture and Training**: The 1.5-Pints model utilizes a modified Mistral tokenizer and a Llama-2 architecture for compatibility.
   - Its training methodologies are based on those used by StableLM, TinyLlama, and HuggingFace, emphasizing the model's versatility.



**Link mentioned**: <a href="https://arxiv.org/abs/2408.03506">1.5-Pints Technical Report: Pretraining in Days, Not Months -- Your Language Model Thrives on Quality Data</a>: This paper presents a compute-efficient approach to pre-training a Language Model-the &#34;1.5-Pints&#34;-in only 9 days, while outperforming state-of-the-art models as an instruction-following assist...

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1272650789306765312)** (48 messages🔥): 

> - `TorchAO`
> - `CUDA developer hiring`
> - `CPU matmul optimization`
> - `CPU matmul performance`
> - `FP16/BF16 weights` 


- **TorchAO presentation at Cohere for AI**: Charles Hernandez from PyTorch Architecture Optimization will be presenting on TorchAO and quantization at the ml-efficiency group at Cohere For AI on August 16th, 2000 CEST.
   - This event is hosted by @Sree_Harsha_N and attendees can join Cohere For AI through the link [https://tinyurl.com/C4AICommunityApp](https://tinyurl.com/C4AICommunityApp).
- **Where to post CUDA developer job openings**: A user asked about the best place to post a job opening for a CUDA developer.
   - No specific answer was provided, but the user was directed to the "jobs channel" within the Discord server.
- **CPU matmul optimization in Zig**: A user is attempting to write a tiling-based matmul in Zig but is having difficulty achieving optimal performance.
   - The user shared their code and received advice on exploring cache-aware loop reordering and the potential for using SIMD instructions.
- **CPU matmul performance comparisons**: A user is comparing the performance of their CPU matmul implementation in Zig to the performance of GGML and NumPy.
   - The user noted that NumPy achieves incredibly fast performance using optimized BLAS implementations and shared links to resources on fast MMM on CPU, including an article by Sibboehm and a blog post by Salykova.
- **FP16/BF16 weights and CPU performance**: A user asked about handling FP16 weights on the CPU, noting that recent models generally use BF16.
   - The user was advised to convert the FP16 weights to BF16 or FP32, with FP32 leading to no accuracy loss but potentially slower inference. The user was also suggested to explore converting tensors at runtime from FP16 to FP32 to potentially improve performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://siboehm.com/articles/22/Fast-MMM-on-CPU">Fast Multidimensional Matrix Multiplication on CPU from Scratch</a>: Numpy can multiply two 1024x1024 matrices on a 4-core Intel CPU in ~8ms.This is incredibly fast, considering this boils down to 18 FLOPs / core / cycle, with...</li><li><a href="https://x.com/Sree_Harsha_N/status/1823091293221691882">Tweet from Sree Harsha (@Sree_Harsha_N)</a>: Excited to host Charles Hernandez from the @PyTorch Architecture Optimization @ the ml-efficiency group Aug16, 2000CEST talking about TorchAO(https://github.com/pytorch/ao) and quantization. Thanks @m...</li><li><a href="https://huggingface.co/vikhyatk/moondream2/tree/main?show_file_info=model.safetensors">vikhyatk/moondream2 at main</a>: no description found</li><li><a href="https://salykova.github.io/matmul-cpu">Beating NumPy in 150 Lines of C Code: A Tutorial on High-Performance Multi-Threaded Matrix Multiplication</a>: In this step by step tutorial we’ll implement high-performance multi-threaded matrix multiplication on CPU from scratch and learn how to optimize and parallelize code in C. On Ryzen 7700 our implement...</li><li><a href="https://ppc.cs.aalto.fi),">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1272821075780042752)** (4 messages): 

> - `PyTorch Full FP16`
> - `PyTorch Optimizer`
> - `torch.compile`
> - `Fairseq Fine-tuning` 


- **PyTorch Full FP16: Is it possible?**: A user asked if full FP16 with loss/grad scaling is possible with PyTorch core, specifically when fine-tuning a large-ish model from Fairseq.
   - They tried using `torch.GradScaler()` and casting the model to FP16 without `torch.autocast('cuda', torch.float16)`, but got an error "ValueError: Attempting to unscale FP16 gradients."
- **Custom Optimizer Implementation**: A user suggested manually accessing and scaling gradients within the optimizer's step function to achieve full FP16 functionality.
   - They provided code illustrating how to retrieve gradients from the optimizer's parameters and then apply scaling operations before calling `optimizer.step()`.
- **torch.compile: The Missing Manual**: A new PyTorch document titled "torch.compile: The Missing Manual" was shared along with a YouTube video.
   - The document and video are available at [https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab) and [https://www.youtube.com/live/rew5CSUaIXg?si=zwbubwKcaiVKqqpf](https://www.youtube.com/live/rew5CSUaIXg?si=zwbubwKcaiVKqqpf), respectively, and provide detailed information on utilizing `torch.compile`.
- **Fairseq's Training Approach**: The original user mentioned that Fairseq models are typically trained with their own custom full FP16 implementations.
   - They also mentioned that while fine-tuning in full BF16 is possible, FP16 AMP often performs better for smaller Fairseq models, likely because they were trained with FP16.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab">torch.compile, the missing manual</a>: torch.compile, the missing manual You are here because you want to use torch.compile to make your PyTorch model run faster. torch.compile is a complex and relatively new piece of software, and so you ...</li><li><a href="https://www.youtube.com/live/rew5CSUaIXg?si=zwbubwKcaiVKqqpf">PyTorch Webinar: torch.compile: The Missing Manual</a>: Hear from Edward Yang, Research Engineer for PyTorch at Meta about utilizing the manual for torch.compile.View the document here to follow along: https://doc...</li><li><a href="https://github.com/pytorch/pytorch/blob/2e7d67e6af45c9338c02dd647c46c328fa23ee48/torch/amp/grad_scaler.py#L259-L260">pytorch/torch/amp/grad_scaler.py at 2e7d67e6af45c9338c02dd647c46c328fa23ee48 · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1272808678428512337)** (7 messages): 

> - `Rust GPU`
> - `Zig`
> - `Fal Research Grants`
> - `Open Source Support` 


- **Rust GPU Transitions to Community Ownership**: The [Rust GPU](https://rust-gpu.github.io) project, previously under [Embark Studios](https://www.embark-studios.com/), is now community-owned under the [Rust GPU GitHub organization](https://github.com/rust-gpu/rust-gpu).
   - This transition marks the beginning of a broader strategy aimed at revitalizing, unifying, and standardizing GPU programming in Rust.
- **Rust/Zig vs CUDA's C/C++**: Rust and Zig offer similar benefits to C and C++ when it comes to GPU programming, although they are not officially supported by CUDA.
   - The discussion highlights the advantages of Rust and Zig, and suggests that learning Zig may be beneficial for those interested in this space.
- **Fal Research Grants for Open Source AI Projects**: The [Fal Research Grants](https://fal.ai/grants) program provides free compute resources to researchers and developers working on open source AI initiatives.
   - The program is open to anyone passionate about advancing AI through open source projects, regardless of their formal qualifications.
- **Fal's Support for Open Source Projects**: Fal appears to be actively supporting numerous open source projects.
   - One user mentioned that Fal funded the [AuraFlow](https://auraflow.io/) project.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://rust-gpu.github.io/blog/transition-announcement/">Rust GPU Transitions to Community Ownership |  </a>: no description found</li><li><a href="https://fal.ai/grants">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1272661469959295066)** (11 messages🔥): 

> - `CUDA Developers`
> - `CUDA Freshers`
> - `CUDA Hiring`
> - `CUDA Engineer`
> - `Triton` 


- **Seeking CUDA Developers for Confidential LLM Inference Project**: A company is seeking a CUDA developer to work on a confidential project related to LLM inference speed.
   - They are looking for someone with deep knowledge of Nvidia Nsight, CUDA programming skills, experience with Hopper Architecture (SM90) kernels, GPU optimization expertise, TensorRT & TensorRT-LLM know-how, and AI/ML framework experience (PyTorch, TensorRT).
- **CUDA Skills for Freshers: What Employers Look For**: A discussion arose about what employers expect from freshers applying for CUDA engineer roles.
   - A consensus emerged that the ability to write a non-trivial CUDA or Triton program to completion, effectively communicate design decisions, and demonstrate aptitude for learning are crucial skills for freshers.
- **Marketing Yourself as a CUDA Engineer**: A member emphasized the importance of showing that you complement a team well and can bring knowledge they don't already have.
   - They advised on showcasing your genuine excitement for the team's work and demonstrating your ability to hit the ground running with a curious and invested attitude.


  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1272670734493286523)** (7 messages): 

> - `Multithreading and GPU Use`
> - `Network Requests and GPUs`
> - `Magnum IO Architecture` 


- **Threads Need to Converge for Progress**: A member discussed the importance of thread convergence for forward progress, stating that *"threads require to converge for making forward progress...If threads dont converge, then forward progression can be tricky."*. 
   - This highlights a challenge with independent thread execution models, where coordinating the work of different threads is crucial for overall progress.
- **GPUs and Network Requests: Why Not?**: The discussion centered around why GPUs aren't widely used for multithreading network requests in scenarios like web crawling, with a member asking: *"Why a GPU is not used widely for multithreading in network requests for use cases like a web crawler?"* 
   - The response indicated that GPUs generally can't make network requests directly, although there might be technical ways to interact over PCIe, it's likely not a practical or efficient solution.
- **Magnum IO: A New Era in Data Center Architecture**: A member shared a link to an article about **Magnum IO**, a new IO subsystem designed for modern data centers and described as *"the IO subsystem of the modern data center"*. 
   - The article highlights the shift in the unit of computing from a single box to the entire data center, emphasizing the need for distributed resources and data sets, as illustrated in a diagram of the Magnum IO stack architecture.



**Link mentioned**: <a href="https://developer.nvidia.com/blog/accelerating-io-in-the-modern-data-center-magnum-io-architecture/">Accelerating IO in the Modern Data Center: Magnum IO Architecture | NVIDIA Technical Blog</a>: This is the first post in the Accelerating IO series, which describes the architecture, components, storage, and benefits of Magnum IO, the IO subsystem of the modern data center.

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: https://www.youtube.com/watch?v=aNAtbYSxzuA
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1272671885133484073)** (126 messages🔥🔥): 

> - `cuDNN stability`
> - `HuggingFace Llama 3 AutoTokenizer issues`
> - `Curand GPU weight initialization`
> - `copy_and_cast_kernel`
> - `cudaMallocAsync/cudaFreeAsync` 


- **cuDNN Stability Warning**: A member asked whether a warning should be added when running with outdated cuDNN, particularly for versions **9.2.1** and **9.3.0**, which significantly affect stability.
   - It was suggested to implement a check at the **Makefile** level, potentially printing a warning message during the build process.
- **HuggingFace Tokenizer Special Token Issue**: There was a discussion regarding the **HuggingFace Llama 3 AutoTokenizer** not properly recognizing the **EOT token (<|endoftext|>)**, leading to potential issues in code-fixing.
- **Curand GPU Weight Initialization PR**: A member proposed an alternative approach to faster model initialization using **curand** to initialize weights directly on the GPU. 
   - This PR is still under development and requires further testing and clean-up.
- **copy_and_cast_kernel Overengineering**: It was pointed out that the **copy_and_cast_kernel** might be overengineered and a simpler approach using direct casting within the kernel could be sufficient.
   - However, the member opted not to change it in this particular PR to avoid introducing potential compatibility issues.
- **cudaMallocAsync/cudaFreeAsync Optimization**: A suggestion was made to optimize the **cudaMallocAsync/cudaFreeAsync** in the critical loop by using a single malloc/free for the largest possible tensor size.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/">Llama 3.1 | Model Cards and Prompt formats</a>: Llama 3.1 - the most capable open model.</li><li><a href="https://github.com/karpathy/llm.c/pull/741">[WIP] initial curand implementation for model init by ngc92 · Pull Request #741 · karpathy/llm.c</a>: as an alternative to the multi-threaded model init, this uses curand to generate initial weights directly on the GPU. It is still work-in-progress, needs error-checking, and I dislike the cudamallo...</li><li><a href="https://github.com/karpathy/llm.c/pull">Pull requests · karpathy/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to karpathy/llm.c development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/tokenizers/pull/1419">add option to skip special tokens by ArthurZucker · Pull Request #1419 · huggingface/tokenizers</a>: Allow skipping special tokens when encoding fixes #1347, fixes #1391 fixes #1368</li><li><a href="https://github.com/karpathy/llm.c/pull/740/commits/16635d41a2a7c0c21ec058eb4201ff75ab97e392">Gordicaleksa fix dataloader2 by karpathy · Pull Request #740 · karpathy/llm.c</a>: commit on top of @gordicaleksa PR that makes a bunch of bugfixes  be more explicit with treatment of EOT token be careful with API for AutoTokenizer bugfix on dtype in fineweb.py use model_desc ins...
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1272643914851029063)** (150 messages🔥🔥): 

> - `Vision Adapters`
> - `Model Merging`
> - `Mistral Large`
> - `GPT-4o Mini`
> - `LLM Studio Headless` 


- **Vision Adapters: The Key to Vision Models**: Only specific LLM models have vision adapters, most of them are going by name "LLaVa" or "obsidian".
   - The "VISION ADAPTER" is a crucial component for vision models; without it, the error you shared will pop up.
- **Mistral Large: The Current Champion?**: A member found **Mistral Large 2** to be the best LLM right now, outcompeting **Claude 3.5 Sonnet** for difficult novel problems.
   - However, the member also noted that **Gemini Flash** undercut **OpenAI 4o mini** severely in price, but **OpenAI 4o** was less expensive than **Mistral Large**.
- **LLM Studio's Model Explorer is Down**: Several members reported that HuggingFace, which powers the **LM Studio Model Explorer**, is down.
   - The site was confirmed to be inaccessible for several hours, with connectivity issues reported across various locations.
- **Llama 3.1 Performance Issues**: A user reported that their **Llama 3 8B model** is now running at only 3 tok/s, compared to 15 tok/s before a recent update.
   - The user checked their GPU offload settings and reset them to default, but the problem persists; the issue appears to be related to a change in the recent update.
- **LLM Output Length Control**: A member is looking for ways to restrict the output length of responses, as some models tend to output whole paragraphs even when instructed to provide a single sentence.
   - While system prompts can be modified, the member found that 8B models, specifically **Meta-Llama-3.1-8B-Instruct-GGUFI**, are not the best at following precise instructions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/iruletheworldmo/status/1823079598176731624">Tweet from 🍓🍓🍓 (@iruletheworldmo)</a>: tomorrow. 10am pt. tune in.</li><li><a href="https://downforeveryoneorjustme.com/huggingface">Is Huggingface down? Live status and problems past 24 hours</a>: Live problems for Huggingface. Error received? Down? Slow? Check what is going on.</li><li><a href="https://huggingface.co/mradermacher/Tiger-Gemma-9B-v1-i1-GGUF">mradermacher/Tiger-Gemma-9B-v1-i1-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1272631729588408405)** (15 messages🔥): 

> - `Portable LLM inference`
> - `Apple Mac`
> - `GPU modding`
> - `Copper modding`
> - `Flashing NVIDIA BIOS` 


- **Portable LLM Inference: Device or Environment?**: A member argued that for portable LLM inference, one should consider whether they want to run inference on their portable device or access a private inference environment while mobile.
- **Apple Mac's Interface Consumes Memory**: A member discussed the memory consumption of macOS visual effects, like transparency, blurs, and shadows, which can consume up to **3GB** of memory.
- **Modding a Project for AI Experiments**: A member asked if anyone else works on modding projects for their AI experiments, and then described their project, which is modding an **Asus ROG Strix RTX 2070 8GB OC**. 
- **Copper Modding for Better Performance**: A member suggested that copper modding helps to spread heat from the memory chips, which improves bandwidth and boosts LLM inference speed.
- **Flashing NVIDIA BIOS to a 2080**: A member mentioned that they might flash the BIOS of their RTX 2070 to a 2080, but they need to read up on the process first.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1272641182039806013)** (151 messages🔥🔥): 

> - `Gemini Live`
> - `Google Fi`
> - `Strawberries`
> - `Project Astra`
> - `LLMs` 


- **Google's Gemini Live: It's here, it's now, it's not free**: Google's new Gemini Live is now available to **Advanced Subscribers** and features conversational overlay on Android and even more connected apps.
   - The general consensus is that it is an improvement from the old voice mode, but with the limitations that it is only available to paid users and the video feature is not live.
- **Google Fi: Is it worth the switch?**: **Google Fi**, Google's cellular network, is based on T-Mobile and some users reported it to be a solid option, though not enough for some to switch from AT&T.
   - One user mentioned that Google Fi is essentially deprioritized T-Mobile, and while not an issue in areas with lots of bandwidth, it is not the ideal option for those with limited coverage.
- **The Great Strawberry Debate**: The discussion of a mysterious user called "Strawberry" (with a string of emojis) led to a speculation that this user might be related to OpenAI or Sam Altman.
   - Many users mentioned how this was quite a clever marketing strategy, linking the strawberry emojis with the image of Sam Altman holding strawberries, and it seemed to be effective at keeping people talking.
- **Project Astra:  Is it a flop? **: While the **Gemini Live** announcement hinted at **Project Astra**, many users were disappointed to not see any further developments.
   - One user even mentioned a **Microsoft recall** comparison, and it seems like people are not too trusting of the company to release this product any time soon, mostly due to security concerns.
- **LLMs: A Solution for All Problems?**: Some users have expressed skepticism about LLMs being the solution for all problems, particularly in the context of using them for tasks like math, database, and even waifu roleplay.
   - Others emphasized the importance of understanding that tokenization is still a fundamental weakness of LLMs, and they can't solve complex problems just through brute force tokenization, but rather need a more strategic approach.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.digitalheresy.com/p/strawberry">🍓🌱⛓️‍💥</a>: A Heretic&#x27;s Journey through the Strawberry Patch</li><li><a href="https://tenor.com/bSx3t.gif">Clueless Aware GIF - Clueless Aware Twitch - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1272659055764176906)** (5 messages): 

> - `Prompt Library`
> - `System Prompt in LangChain` 


- **Prompt Library Location**: A user asked how to access the prompt library, and another user provided a link to a channel containing the prompt library.
- **Adding System Prompts in LangChain**: A user shared Python code demonstrating how to create a GPT based on Strawberry, but they wanted to know how to add a system prompt.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1272980860282343516)** (3 messages): 

> - `ChatGPT website access` 


- **ChatGPT can't access websites**: A member asked if there's a way to get ChatGPT to visit a website and pull an article for it to read.
- **ChatGPT may hallucinate website content**: Another member suggested that ChatGPT may be hallucinating or blocked from crawling the website.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1272980860282343516)** (3 messages): 

> - `ChatGPT accessing websites`
> - `ChatGPT's hallucination and web crawling` 


- **ChatGPT cannot access websites directly**: A user asked if there was a way to get ChatGPT to access a website and pull an article for it to read.
- **ChatGPT may hallucinate or be blocked from crawling websites**: A user suggested that ChatGPT might be hallucinating or blocked from crawling websites, explaining that it sometimes cites sources from websites, but not always.
- **Web browser GPT**: A user asked if anyone had tried mentioning the web browser GPT to ChatGPT, possibly as a way to work around this limitation.


  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1272631560469741764)** (106 messages🔥🔥): 

> - `Perplexity bug reports`
> - `Perplexity Pro Models`
> - `Perplexity's UI/UX`
> - `Perplexity's website stability`
> - `Perplexity's future` 


- **Perplexity's UI/UX issues**: Users reported several UI/UX issues including missing buttons and a disappearing prompt field, leading to difficulties in interacting with the platform.
- **New Model: Sonar Huge**: The new model "Sonar Huge" replaced the Llama 3.1 405B model in Perplexity Pro and was observed to be slow and not adhere to user profile prompts.
- **Perplexity's Website Stability Concerns**: Users reported a significant decline in website stability, citing issues with sporadic search behavior, forgetting context, and interface bugs on both web and iOS versions.
- **Perplexity's Success Team Acknowledges Bugs**: The Success Team at Perplexity acknowledged receiving user feedback on the recent bugs and glitches encountered in the platform.
- **The Future of Perplexity's Feature Implementation**: A user expressed frustration over the long wait time for feature implementation, highlighting the disparity between the promised features and the actual rollout pace.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1823029449534615705?s=46&t=JsxhFTRLBknd8RUv1f73bA">Tweet from Perplexity (@perplexity_ai)</a>: We&#39;re thrilled to announce our partnership with @Polymarket. Now, when you search for events on Perplexity, you&#39;ll see news summaries paired with real-time probability predictions, such as ele...</li><li><a href="https://status.perplexity.com/history/1">Notice history - Perplexity - Status</a>: Notice history - Perplexity Status
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1272637873912414238)** (9 messages🔥): 

> - `Coursera`
> - `Programming Courses`
> - `AI/ML`
> - `Cloud Computing`
> - `Data Science` 


- **Coursera's High-Paying Programming Courses**: Coursera offers a wide range of programming courses that can lead to lucrative tech careers, with specializations in **Python, AI/machine learning, cloud computing, and data science** being particularly popular.
   - These courses, offered by institutions like **Stanford, Google, and IBM**, help learners develop in-demand skills and enhance their career prospects.
- **Strategic Skill Development for Tech Careers**: For maximizing earning potential, Coursera recommends combining technical skills with soft skills like **project management and communication**, which can help learners stand out in competitive fields.
   - By strategically combining these skills, learners can position themselves for high-paying roles in **software engineering, data science, and cloud architecture**.
- **Importance of Hands-On Experience and Staying Current**: Coursera emphasizes the importance of choosing comprehensive programs that provide **hands-on experience** and stay up-to-date with emerging technologies.
   - This ensures that learners acquire the most relevant and valuable skills for the current job market.
- **Perplexity AI Chatbot Guidance on Shareable Threads**: A Perplexity AI chatbot reminds a user to make sure their thread is **shareable**, providing a link to instructions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/embed/81Pey4X6SY0">YouTube</a>: no description found</li><li><a href="https://www.perplexity.ai/search/link-me-the-best-coursera-cour-Re4JWGgnTDmDZ_06LX4Z_Q">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/could-you-analyse-the-image-an-Os_cDlCGRbelIR.2YAc4Lw">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/who-is-princess-elara-self-awa-wMoqIeyRS9.RXRMRA8nL9g">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/search/how-many-fish-are-caught-every-szC5dPGSTFysBuOJZ9qeKA">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.</li><li><a href="https://www.perplexity.ai/page/best-coursera-courses-for-prog-j1R20PNzTsqDCrrk6_Fk3A">no title found</a>: no description found</li><li><a href="https://www.perplexity.ai/search/whats-the-most-common-group-of-gGB6elWUR8KdkYVrYdhZ1Q">Perplexity</a>: Perplexity is a free AI-powered answer engine that provides accurate, trusted, and real-time answers to any question.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1272937147841773739)** (6 messages): 

> - `Perplexity Search Parameters`
> - `Search Location Options`
> - `Image Generation from Narrative` 


- **Search Parameter Control**: A member expressed interest in controlling search parameters like `intitle:ABC`, similar to Google Search.
   - They believe this feature would greatly enhance the search capabilities of Perplexity.
- **Search Location Selection**: Another member inquired about the possibility of selecting specific search locations within Perplexity.
   - They acknowledged the value of this feature for narrowing search results and finding location-specific information.
- **Narrative-Based Image Generation**: A user provided a narrative describing a scene with a cat walking along a wall in the rain, requesting an image based on this description.
   - The narrative included details about the camera angles, lighting, and atmosphere, suggesting an interest in AI-powered image generation.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1272675105520226400)** (1 messages): 

> - `SXSW Panel`
> - `OpenAI Models`
> - `AI Risks and Opportunities`
> - `Government Regulation`
> - `AI Impact` 


- **Stability AI's SXSW Panel Proposal**: Stability AI CEO Prem Akkaraju and tech influencer Kara Swisher will discuss the importance of open AI models and the role of government in regulating their impact.
   - The panel will explore the opportunities and risks of AI, including job displacement, disinformation, CSAM, and IP rights.
- **Democratizing Access to Cutting-Edge Technology**: The panel will emphasize how open-source AI models are driving innovation and democratizing access to technology, particularly in CGI.
   - This accessibility promotes experimentation and accelerates progress in various fields, empowering individuals and organizations with new possibilities.
- **Balancing Commercial Interests and AI Risks**: The discussion will address the challenges of balancing commercial interests with the potential risks of generative AI in a rapidly evolving sector.
   - Key topics will include mitigating disinformation, protecting intellectual property, and addressing ethical concerns surrounding AI use.
- **AI's Future Impact on Content Creation and Work**: The panel will explore the future of AI in content creation, work, education, and other domains.
   - They will discuss AI's capacity to enhance human potential across all social and economic strata, while considering its implications for jobs and skill development.
- **Prem Akkaraju's Vision for AI and CGI**: Prem Akkaraju will share his vision for the company and the industry, focusing on the convergence of AI and CGI.
   - He will discuss how this convergence is poised to transform creative fields and offer new possibilities for content creation and storytelling.



**Link mentioned**: <a href="http://panelpicker.sxsw.com/vote/153232">PanelPicker | SXSW Conference &amp; Festivals</a>: PanelPicker® is the official SXSW user-generated session proposal platform. Enter ideas and vote to help shape Conference programming for SXSW and SXSW EDU.

  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1272658083738292427)** (111 messages🔥🔥): 

> - `Google Colab Runtime`
> - `Stable Diffusion Installation`
> - `Stable Diffusion Model Merging`
> - `CUDA Installation`
> - `Flux Realism` 


- **Google Colab Runtime Stops**: A user inquired about preventing their Google Colab runtime from stopping.
   - Another user suggested using Kaggle instead, as it offers more resources and longer runtimes.
- **Stable Diffusion Installation Woes**: A user reported difficulties installing Stable Diffusion, encountering issues with CUDA installation and finding their token on Hugging Face.
   - Another user provided guidance on generating a token through the profile settings menu and installing CUDA correctly.
- **Model Merging Questions**: A user discussed potential model merging tactics, proposing applying the difference between UltraChat and base Mistral to Mistral-Yarn.
   - Other users expressed skepticism, but the original user remained optimistic, citing successful past attempts at model merging.
- **Flux Realism for Face Swaps**: A user inquired about using Flux Realism to put their face on images.
   - They mentioned trying fal.ai, but the results seemed cartoonish, prompting them to seek alternative solutions.
- **Training LORAs for Logo Generation**: A user asked for guidance on training LORAs for logo generation, specifically for placing logos onto images.
   - Another user recommended using Flux, as it can train on logos and accurately place them onto images, such as shirts or buildings.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.kaggle.com">Kaggle: Your Machine Learning and Data Science Community</a>: Kaggle is the world&#x2019;s largest data science community with powerful tools and resources to help you achieve your data science goals.</li><li><a href="https://tenor.com/view/heart-container-goddess-statue-totk-heart-container-totk-zelda-gif-891944359093961229">Heart Container Goddess Statue GIF - Heart container Goddess statue Totk heart container - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/CS1o/">CS1o - Overview</a>: CS1o has 2 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Installation-Guides">Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/cumulo-autumn/StreamDiffusion?tab=readme-ov-file#step1-make-environment">GitHub - cumulo-autumn/StreamDiffusion: StreamDiffusion: A Pipeline-Level Solution for Real-Time Interactive Generation</a>: StreamDiffusion: A Pipeline-Level Solution for Real-Time Interactive Generation - cumulo-autumn/StreamDiffusion
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1272894465169690707)** (2 messages): 

> - `Gemini Flash 1.5`
> - `GPT-4o Extended`
> - `OpenRouter Pricing` 


- **Gemini Flash 1.5 Price Drop**: The input token costs for **Gemini Flash 1.5** have decreased by **78%** and the output token costs have decreased by **71%**.
   - This makes the model more accessible and affordable for a wider range of users.
- **GPT-4o Extended Early Access Launch**: Early access has just launched for **GPT-4o Extended** through **OpenRouter**.
   - You can access it via this link: [https://x.com/OpenRouterAI/status/1823409123360432393](https://x.com/OpenRouterAI/status/1823409123360432393)
- **GPT-4o Extended Output Limit**: The maximum number of tokens allowed for GPT-4o Extended output is **64k**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1823409123360432393">Tweet from OpenRouter (@OpenRouterAI)</a>: You can now use GPT-4o extended output (alpha access) through OpenRouter!  64k max tokens</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5>)">Gemini Flash 1.5 - API, Providers, Stats</a>: Gemini 1.5 Flash is a foundation model that performs well at a variety of multimodal tasks such as visual understanding, classification, summarization, and creating content from image, audio and video...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1272630857756315708)** (80 messages🔥🔥): 

> - `Gemini Flash Price Updates`
> - `GCP Cost Table`
> - `Token:Character Ratio`
> - `Euryale 70B Downtime`
> - `Infermatic Downtime` 


- **Gemini Flash Prices on OpenRouter?**: A user inquired about the new Gemini flash prices on OpenRouter and when they will be updated.
   - A user mentioned their GCP cost table already reflects the new pricing, suggesting it's up to OpenRouter to implement the update.
- **OpenRouter's Update Hurdle**: OpenRouter's update was blocked by the new 1:4 token:character ratio from Gemini, which doesn't map cleanly to the `max_tokens` parameter validation.
   - Another user expressed frustration about the constantly changing token:character ratio and suggested switching to a per-token pricing system.
- **Euryale 70B Issues?**: A user reported that Euryale 70B was down for some users but not for them, prompting questions about any issues or error rates.
   - Further discussion revealed multiple instances of downtime, including a 10-minute outage due to an update and possible ongoing issues with location availability.
- **Model Performance Comparison**: Users compared the performance of Groq 70b and Hyperbolic, finding nearly identical results for the same prompt.
   - This led to a discussion about the impact of FP8 quantization, with some users noting that it makes a minimal difference in practice, but others pointing to potential degraded quality with certain providers.
- **ChatGPT 4.0 Default Setting Change**: A user expressed concern that the "middle-out" setting is no longer the default for ChatGPT 4.0, which impacts function calling for their frontends.
   - The user requested suggestions for setting this parameter in the system prompt using platforms like Ollama and a Wordpress plugin.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/MultiOn_AI/status/1823412701441482959">Tweet from MultiOn (@MultiOn_AI)</a>: Announcing our latest research breakthrough:   Agent Q - bringing next-generation AI agents with planning and AI self-healing capabilities, with a 340% improvement over LLama 3&#39;s baseline zero-sho...</li><li><a href="https://x.com/btibor91/status/1821452608046788853?t=_Pnb_GU6R2TZpd8DxZaHOA&s=19">Tweet from Tibor Blaho (@btibor91)</a>: Someone pressed publish on too many new (and test?) articles too early in the last hours - expect GPT-4o system card, SWE-bench Verified, new customer stories, and more soon  - Collaborating with The ...</li><li><a href="https://huggingface.co/deepseek-ai">deepseek-ai (DeepSeek)</a>: no description found</li><li><a href="https://codeium.com/blog/codeium-dream-bigger">Dream Bigger</a>: The Codeium mission, Cortex and Forge launches, and detailed vision.</li><li><a href="https://openrouter.ai/models/sao10k/l3-lunaris-8b">Llama 3 8B Lunaris - API, Providers, Stats</a>: Lunaris 8B is a versatile generalist and roleplaying model based on Llama 3. It&#x27;s a strategic merge of multiple models, designed to balance creativity with improved logic and general knowledge. R...</li><li><a href="https://openrouter.ai/models/aetherwiing/mn-starcannon-12b">Mistral Nemo 12B Starcannon - API, Providers, Stats</a>: Starcannon 12B is a creative roleplay and story writing model, using [nothingiisreal/mn-celeste-12b](https://openrouter.ai/models/nothingiisreal/mn-celeste-12b) as a base and [intervitens/mini-magnum-...</li><li><a href="https://openrouter.ai/models/sao10k/l3-euryale-70b">Llama 3 Euryale 70B v2.1 - API, Providers, Stats</a>: Euryale 70B v2.1 is a model focused on creative roleplay from [Sao10k](https://ko-fi. Run Llama 3 Euryale 70B v2.1 with API
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1272652219547779072)** (30 messages🔥): 

> - `Mojo Licensing Concerns`
> - `Mojo Open-Sourcing`
> - `Mojo Development`
> - `Mojo Learning Resources`
> - `Mojo Compiler` 


- **Mojo's License Has a Catch**: The Mojo license prohibits the development of any applications using the language for competitive activities.
   - However, it states that this rule does not apply to applications that become competitive after their initial release, but it is unclear how this clause will be applied.
- **Mojo's Uncertain Open-Sourcing Future**: Users inquired about the open-sourcing timeline of Mojo's compiler.
   - While the team confirmed that the compiler will be open-sourced eventually, a public timeline is not available, suggesting that it may be a while before contributions can be made.
- **Mojo Development: Standard Library Focus**: The current focus of Mojo development is on building out the standard library.
   - Users are encouraged to contribute to the standard library, while work on the compiler is ongoing, but not yet open to contributions.
- **Mojo Learning for Students**: A college student inquired about resources for learning Mojo, including compiler basics and MLIR.
   - The team suggested starting with learning Mojo itself and contributing to the standard library, while MLIR knowledge will be helpful for future compiler contributions.
- **Mojo Compiler:  Limited Documentation and  Internal Dialects**: The team acknowledges the lack of documentation for Mojo's internal dialects, which has made development challenging for some contributors.
   - The ability to add rewrite rules directly to the compiler is being considered, but it is not currently possible, leading some to shelve their projects due to the high time investment required to reverse-engineer the compiler.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/presentation/d/1vkM05Ld8nEfLalxSuWmjDv_wfYQ9IVb7HbQ07MR3Xxs/edit?usp=sharing">Small string optimization in Mojo’s stdlib</a>: Small string optimization in Mojo’s stdlib and small buffer optimization while we’re at it</li><li><a href="https://docs.google.com/presentation/?usp=slides_web">no title found</a>: no description found</li><li><a href="https://accounts.google.com/ServiceLogin?service=wise&passive=1209600&osid=1&continue=https://docs.google.com/presentation/d/1vkM05Ld8nEfLalxSuWmjDv_wfYQ9IVb7HbQ07MR3Xxs/edit?usp%3Dsharing&followup=https://docs.google.com/presentation/d/1vkM05Ld8nEfLalxSuWmjDv_wfYQ9IVb7HbQ07MR3Xxs/edit?usp%3Dsharing&ltmpl=slides&ec=GAZAmQI)">no title found</a>: no description found</li><li><a href="https://support.google.com/docs/answer/2375082?hl=en).[Dismiss](#)>>>">System requirements and browsers - Computer - Google Docs Editors Help</a>: no description found
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1272631576089595905)** (19 messages🔥): 

> - `Java by Microsoft`
> - `C# relevance`
> - `Stable Diffusion Memory Issue`
> - `WSL2 limitations`
> - `Mojo Optimization` 


- **Java by Microsoft: A Forgotten Giant?**: One member argued that "Java by Microsoft" was unnecessary and could have been avoided, while another countered that it seemed crucial at the time.
   - The discussion acknowledged the emergence of newer solutions and the decline of "Java by Microsoft" over time, highlighting its 20-year run and its relevance in the Microsoft marketshare.
- **C# Rise in Microsoft's Dominance**: C# appeared in 2000 and has been a key part of Windows development for over two decades, being viewed as a "nicer Java" for many tasks.
   - C# gained popularity rapidly as the "new way to do applications on Windows", especially in the 2nd and 3rd world countries where Windows has a significant presence.
- **Stable Diffusion Memory Issues in WSL2**: A new user encountered an issue running the Stable Diffusion Mojo ONNX example in WSL2, where the process was killed due to memory pressure.
   - The user had 8GB allocated to WSL2 but was advised to double it as Stable Diffusion 1.5 is approximately 4GB, requiring more memory for the model and optimization processes.
- **WSL2 Memory Constraints**: Windows prioritizes the health of the host OS over WSL2 processes, potentially leading to memory constraints when running memory-intensive applications like Stable Diffusion.
   - Doubling the memory allocated to WSL2 from 8GB to 16GB was suggested to alleviate the issue.
- **Mojo Optimization: Memory Efficiency**: The memory efficiency of Stable Diffusion was discussed, noting that optimization can consume significant RAM.
   - The user was advised to allocate more memory to WSL2 to ensure sufficient resources for both the model and its optimization processes.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1272683094180102157)** (11 messages🔥): 

> - `Cohere For AI`
> - `Pricing changes`
> - `Cohere's Research Lab`
> - `Hackathons`
> - `Computer Vision` 


- **Cohere For AI research lab**: **Cohere For AI** is a non-profit research lab that seeks to solve complex machine learning problems.
   - They support fundamental research exploring the unknown, and are focused on creating more points of entry into machine learning research. 
- **Pricing Changes on Cohere's Website**: A user asked about the **classify** feature's pricing, noticing it's no longer listed on the pricing page.
   - No response was given.
- **Hackathon Group**: A user is looking for more members to join their Hackathon group.
   - The group currently has 2-3 people and they are looking for people with diverse skillsets, especially those who can submit videos.
- **Computer Vision Interest**: A new user introduced themself as a Computer Engineering graduate interested in AI, ML, DL, particularly Computer Vision.
   - They have done some projects related to CV and are looking to improve in this area.



**Link mentioned**: <a href="https://cohere.com/research">Cohere For AI (C4AI)</a>: Cohere For AI is a non-profit research lab that seeks to solve complex machine learning problems. We support fundamental research that explores the unknown, and are focused on creating more points of ...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1272669301207203841)** (26 messages🔥): 

> - `JSONL Upload Issue`
> - `Azure JSON Formatting`
> - `Rerank Overview`
> - `Cohere API Usage`
> - `Python Kernel Restart` 


- **JSONL Uploads Failing**: Users reported issues uploading JSONL datasets for fine-tuning, with the error message "File format is not supported for dataset".
   - The issue was acknowledged by Cohere support and is being investigated. In the meantime, users can utilize the API for dataset creation, which is currently functioning correctly.
- **Azure JSON Formatting Not Supported**: A member inquired about using structured output with `response_format` in Azure, but encountered an error indicating the parameter is invalid.
   - It was confirmed that JSON formatting is not currently available on Azure.
- **Rerank Overview Code Help**: A user requested assistance with the Rerank Overview document, encountering issues with the provided code.
   - The issue was related to an outdated document, and a revised code snippet was provided. The user was also directed to the relevant documentation for further reference.
- **"Unknown Field" Error in Rerank**: A user experienced an "unknown field" error when using the Rerank API.
   - This error was confirmed to be unrelated to the Rerank API, and restarting the Python kernel was suggested as a potential resolution.



**Link mentioned**: <a href="https://docs.cohere.com/reference/rerank">Rerank - Cohere API References</a>: This endpoint takes in a query and a list of texts and produces an ordered array with each text assigned a relevance score.

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1272635724767695009)** (7 messages): 

> - `JSON Snippet Embeddings`
> - `Intermediate Text` 


- **Embeddings for JSON Snippets**: A member asked about the preferred method for providing JSON as document snippets, aiming for compatibility with large JSON datasets.
- **Utility of Intermediate Text**: A member inquired about the usefulness of intermediate text.
- **Embeddings as a Solution**: One member suggested converting JSON into embeddings as a possible solution.
- **Clarifying the Goal**: Another member requested clarification on the intended goal, offering assistance in finding a solution.


  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1272661992057868350)** (44 messages🔥): 

> - `TransformerDecoderLayer Refactor`
> - `RLHF with DPO/PPO`
> - `Torchtune & WandB`
> - `Torchtune Performance`
> - `PyTorch Conference` 


- **TransformerDecoderLayer Refactor PR**: A PR has been submitted to refactor the TransformerDecoderLayer, touching many files and making core changes in modules/attention.py and modules/transformer.py.
   - This PR implements the RFC #1211, aiming to improve the TransformerDecoderLayer architecture.
- **RLHF with DPO/PPO**: There is a discussion about testing the HH RLHF builder with DPO or PPO, with DPO being preferred for preference datasets while PPO is dataset-agnostic.
   - The focus is on DPO, with the expectation of loss curves similar to normal SFT, and potential debugging needed for the HH RLHF builder, which may be addressed in a separate PR.
- **Torchtune & WandB Issues**: A user encountered issues accessing WandB results for Torchtune, with access being granted after adding the user as a team member.
   - The user reported poor results with the default DPO config and turning gradient accumulation off, but later discovered it started working again, potentially due to a delay or some other factor.
- **Torchtune Performance with DPO**: There is a discussion about potential issues with the default DPO config causing poor performance in Torchtune.
   - The user suggested trying SIMPO (Stack Exchange Paired) and turning gradient accumulation back on, as having a balanced number of positive and negative examples in the batch can significantly improve loss.
- **PyTorch Conference**: A discussion about the upcoming PyTorch Conference, with links to the website and details on featured speakers.
   - There was also a mention of sneaking in a participant as an 'academic' for the conference, but this is potentially a joke.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/rafi-personal/torchtune?nw=nwuserrdoublea">rafi-personal</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://events.linuxfoundation.org/pytorch-conference/">PyTorch Conference | LF Events</a>: Join top&#x2d;tier researchers, developers, and academics for a deep dive into PyTorch, the cutting&#x2d;edge open&#x2d;source machine learning framework.</li><li><a href="https://github.com/pytorch/torchtune/pull/1312">TransformerDecoderLayer Refactor by pbontrager · Pull Request #1312 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (please add here)  This is an implementation of the similarly named RFC #12...</li><li><a href="https://github.com/pytorch/torchtune/pull/645#issuecomment-2047853377">DPO by yechenzhi · Pull Request #645 · pytorch/torchtune</a>: Context integrating DPO into Torchtune, more details see here Changelog  ...  Test plan  ....
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1272652228884299847)** (19 messages🔥): 

> - `Perplexity Pro`
> - `Llama 3`
> - `Grad Clipping`
> - `OpenAI Benchmark` 


- **Perplexity Pro reasoning abilities**: A user noted that [Perplexity Pro](https://perplexity.ai/) has gotten *"crazy good at reasoning"* and is able to *"literally count letters"* like it *"ditched the tokenizer"*.
   - They shared a link to [a GitHub repository](https://github.com/cognitivecomputations/grokadamw) that appears to be related to this topic.
- **Llama 3 and Model of Experts**: One user asked if anyone had made a "MoE" version of Llama 3.
- **Grad Clipping Explained**: A user asked about the functionality of grad clipping, specifically wondering what happens to gradients when they exceed the maximum value.
   - Another user explained that grad clipping essentially clips the gradient to a maximum value, preventing it from exploding during training.
- **OpenAI Benchmark Release**: A user shared their surprise at OpenAI releasing a benchmark instead of a new model, speculating that this might be a strategic move to steer the field towards better evaluation tools.



**Link mentioned**: <a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>: Contribute to cognitivecomputations/grokadamw development by creating an account on GitHub.

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1272940875584110666)** (4 messages): 

> - `AutoGPTQ`
> - `Axolotl` 


- **Axolotl's capabilities**: A member noted that AutoGPTQ could do certain things, implying that Axolotl may be able to do so as well.
   - They were excited about the possibility of Axolotl replicating this capability.
- **OpenAI's Sharegpt API**: A member recommended using `type: sharegpt` and `conversation: llama` with the API to get more desired results.
   - This suggestion indicates a preference for certain parameters within the API when working with Axolotl.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[deployment-help](https://discord.com/channels/1104757954588196865/1163840836472148058/1272943033608048722)** (1 messages): 

> - `LLM Inference`
> - `VLLM`
> - `SkyPilot`
> - `Fireworks`
> - `Lora Adapters` 


- **VLLM inference on your own GPUs with SkyPilot**: A member recommended using VLLM on your own GPUs and managing it via SkyPilot for greater flexibility.
   - This setup allows for full control and can handle specific needs.
- **Serverless Billing with Fireworks**: Fireworks was mentioned as a suitable solution for serving Lora adapters with serverless billing.
   - However, Fireworks has limitations, including compatibility with all base models and occasional quirks.


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1272921653776617564)** (16 messages🔥): 

> - `Grok 2.0`
> - `Flux.1 Model`
> - `Grok Image Generation`
> - `Open Source Image Annotation`
> - `Elon and Models` 


- **Grok 2.0 Leaks Early**: A member shared a link to a Tweet about Grok 2.0 features and abilities, including image generation using the FLUX.1 model.
   - The tweet also noted that Grok 2.0 is better at coding, writing, and generating news.
- **Flux.1 Model Usage**: A member mentioned that many Elon fan accounts predicted X would use MJ (presumably referring to a model), suggesting that Flux.1 may have made an inflection point in model usage.
   - The member questioned if Flux.1 is Schnellit's Pro model, given Elon's history.
- **Open-Source Image Annotation**: A member asked for recommendations for good open-source GUIs for annotating images quickly and efficiently.
   - The member specifically mentioned single-point annotations, straight-line annotations, and drawing polygonal segmentation masks.
- **Elon's Model Choices**: A member discussed the possibility that Elon is using a development version of Grok and calling the bluff on weight licenses.
   - This member believes that Elon could potentially call this a "red-pill" version.



**Link mentioned**: <a href="https://x.com/nima_owji/status/1823388838279922166">Tweet from Nima Owji (@nima_owji)</a>: BREAKING: Here&#39;s an early look at Grok 2.0 features and abilities!  It&#39;s better at coding, writing, and generating news! It&#39;ll also generate images using the FLUX.1 model!

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1272842479200370728)** (4 messages): 

> - `Position Encoding`
> - `2D Pooling` 


- **New Position Encoding?**: A user believes they may have invented a superior type of position encoding, and are currently verifying its efficacy.
- **2D Pooling Success**: The user expresses surprise at how well 2D pooling works, noting it was recommended by another user.


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

flammit_: no worries - just left hopefully helpful hints on your nvidia FP8 PR
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1272665251762147409)** (8 messages🔥): 

> - `Tensor Filtering`
> - `Transcendental Folding Optimization`
> - `CUDA TIMEOUT ERROR` 


- **Tensor Filtering the Fastest Way?**: A user asked for the fastest way to filter a Tensor, such as `t[t % 2 == 0]`, currently doing it by converting to list, filtering, and converting back to list.
   - A suggestion was made to use masking if computing something on a subset of the Tensor, but it was noted that the exact functionality is not possible yet.
- **Transcendental Folding Refactor Optimization**: A user proposed a refactor to only apply transcendental rewrite rules if the backend does not have a code_for_op for the uop.
   - The user implemented a `transcendental_folding` function and called it from `UOpGraph.__init__` but wasn't sure how this could be net negative lines, and asked what could be removed.
- **CUDA TIMEOUT ERROR**: A user ran a script using `CLANG=1` and received a `RuntimeError: wait_result: 10000 ms TIMEOUT!` error.
   - The error occurred with the default runtime and was resolved by using `CUDA=1`, and the issue was potentially related to ##4562.


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1273021815517741076)** (3 messages): 

> - `Poe Previews Hackathon`
> - `Agihouse Hackathon`
> - `Poe Platform Announcement`
> - `In-Chat Generative UI Experiences`
> - `Discord Channel` 


- **Poe Previews Hackathon: A Celebration of Expanded Release**: Poe (@poe_platform) announced a partnership with Agihouse (@agihouse_org) for a "Previews Hackathon" to celebrate their expanded release.
   - The hackathon invites all creators to build the most innovative and useful in-chat generative UI experiences, with details available at [https://app.agihouse.org/events/poe-previews-hackathon-20240817](https://app.agihouse.org/events/poe-previews-hackathon-20240817).
- **Discord Channel Discussion on the Hackathon**: A user in the #events Discord channel shared a link to the Poe Previews Hackathon announcement on X, confirming they're helping out with the event.
- **Hackathon Goal: In-Chat Generative UI Experiences**: The hackathon aims to create innovative and useful "in-chat generative UI experiences", encouraging creators to showcase their skills.
   - The announcement emphasizes the importance of user experience in the context of generative AI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/poe_platform/status/1823382125523181683">Tweet from Poe (@poe_platform)</a>: To celebrate the expanded release, we’re partnering with @agihouse_org for a Previews hackathon where you’ll compete to create the most innovative and useful in-chat generative UI experiences. All cre...</li><li><a href="https://app.agihouse.org/events/poe-previews-hackathon-20240817">AGI House</a>: no description found</li><li><a href="https://app.agihouse.org/events/poe-previews-">AGI House</a>: no description found
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1272961929115340921)** (4 messages): 

> - `Virtual Try On`
> - `Image Feature Extraction`
> - `Model Size` 


- **Virtual Try On Implementation**: A member shared their experience building a virtual try-on feature for their R&D team, noting its effectiveness in speeding up training runs by storing extracted features.
   - The feature utilizes online preprocessing and stores extracted features in a document store table, allowing for efficient retrieval during training.
- **Image Feature Extraction Techniques**: A member inquired about the specific features being extracted from images for the virtual try-on feature.
   - The member providing the feature details highlighted the generic nature of their approach, accommodating models ranging from extremely small to massive sizes.
- **Model Size Impact on Virtual Try On**: The member emphasized the successful application of their virtual try-on feature across a wide range of model sizes.
   - This demonstrates the flexibility of the approach in handling different computational demands and model complexities.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1272667816859930644)** (5 messages): 

> - `Llama 3.1 8b structured output`
> - `RAG on technical documents with images`
> - `Next.js and FastAPI interaction`
> - `AWS pip install issues` 


- **Llama 3.1 8b supports structured output through tools**: A user confirmed that **Llama 3.1 8b** can produce structured output through tool use, having tested it directly with **llama.cpp**.
   -  
- **Extracting information from technical images for RAG**: A user sought advice on extracting information from images like electrical diagrams, maps, and voltage curves for **RAG** on technical documents.
   - They mentioned encountering difficulties with traditional methods, highlighting the need for capturing information not present in text form but visually interpretable by experts.
- **Next.js POST request to FastAPI returns 405 Method Not Allowed**: A user encountered a **405 Method Not Allowed** error when making a **POST request** from a **Next.js web app** running on **EC2** to a **FastAPI endpoint** on the same **EC2 instance**.
   - They observed the request being incorrectly interpreted as a **GET request** despite explicitly using the **POST method** in their **Next.js code**.
- **AWS pip install issue resolved due to environment emulation**: A user resolved an issue with **pip install** on an **AWS system** by installing packages specifically for the **Unix-based environment**.
   - The problem arose from the virtual environment mistakenly emulating **Windows** during the **pip install** process, causing the issue.


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1272903058749390869)** (1 messages): 

> - `Profundo`
> - `Profundo use cases`
> - `Profundo AI`
> - `Profundo product hunt`
> - `Profundo benefits` 


- **Profundo launches for automated research**: Profundo automates data collection, analysis, and reporting, enabling everyone to do deep research on topics they care about. 
   - It minimizes errors and maximizes productivity, allowing users to focus on making informed decisions. 
- **Profundo's AI powers efficient data handling**: Profundo uses cutting-edge AI to help you gather, analyze, and report data more efficiently. 
   - Say goodbye to manual data collection and hello to automated insights. 
- **Profundo empowers diverse use cases**: Profundo is being used for self-study, content creation, first drafts, personal projects, and career development.
   - In the academic world, it's employed for research and literature reviews. 
- **Profundo seeks ProductHunt upvotes**: Profundo launched on ProductHunt today, and they are seeking upvotes to reach more people.
   - If you have used Profundo and found it useful, they encourage you to upvote them on ProductHunt. 



**Link mentioned**: <a href="http://profundo.app/>">Profundo | Research Redefined</a>: Profundo is a research platform that allows you to conduct research in a way that is more efficient and effective than ever before.

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1272954945963819058)** (1 messages): 

> - `AI Agents in Enterprises`
> - `Monitoring and Governance of AI Agents` 


- **AI Agent Governance in the Enterprise**: A user inquired about the challenges of monitoring and governance of AI agents within large organizations.
- **Open Discussion Invitation**: The user invited anyone working on AI agents within an enterprise to share their experiences.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1272661518428799017)** (2 messages): 

> - `Screenless personal tutor for kids` 


- **Screenless tutor idea for kids**: A member expressed interest in using 01 to build a screenless personal tutor for kids.
   - They asked for feedback and if anyone else was interested in collaborating on this project.
- **Screenless tutor idea for kids**: A member expressed interest in using 01 to build a screenless personal tutor for kids.
   - They asked for feedback and if anyone else was interested in collaborating on this project.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1272997098291462164)** (3 messages): 

> - `Open Interpreter in Obsidian`
> - `Convert Anything Tool` 


- **Open Interpreter for Anything to Anything**: Use Open Interpreter to convert any type of data into any other format.
   - This is possible by using the "Convert Anything" tool, which harnesses the power of Open Interpreter.
- **Open Interpreter in Obsidian**: A new YouTube series is launching that will demonstrate how to use Open Interpreter in the Obsidian note-taking app.
   - This plugin allows you to control your Obsidian vault using Open Interpreter, which could have major implications for how people work with knowledge.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=HjcPRoPfri0">Open Interpreter Obsidian &amp; Convert Anything - Ep 0</a>: Episode 0 of Tool Use!Open Interpreter Obsidian Plugin - Use Open Interpreter to control your Obsidian vault!CV - Convert anything to anything using the powe...</li><li><a href="https://www.youtube.com/watch?v=xaroJxFTVFQ">Is the AI Left-Bias Real?</a>: Take courses on large language models on Brilliant! First 30 days are free and 20% off the annual premium subscription when you use our link ➜  https://brill...
</li>
</ul>

</div>
  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1272760688007319655)** (1 messages): 

> - `SlimOrca without deduplication`
> - `Fine-tuning (FT) with deduplication` 


- **SlimOrca Without Deduplication**: A user inquired about a version of **SlimOrca** that has **soft prompting removed** and **no deduplication**, ideally including the code.
   - They also asked if anyone had experimented with fine-tuning (FT) on data with or without deduplication, and with or without soft prompting.
- **Fine-tuning on Deduplicated Data**: The user asked if anyone had experimented with fine-tuning on **deduplicated data** versus **non-deduplicated data**.
- **Fine-tuning with Soft Prompting**: The user inquired about the effects of fine-tuning (FT) with **soft prompting** versus without soft prompting.


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1273007276168319067)** (1 messages): 

> - `Agentic System for Jupyter Notebook Automation` 


- **Building an Agentic Jupyter Notebook Automation System**: A member expressed interest in building an agentic system for automating Jupyter Notebooks, aiming to create a pipeline that takes an existing notebook as input, modifies cells, and generates multiple variations.
   - They sought recommendations for libraries, cookbooks, or open-source projects that could provide a starting point for this project, drawing inspiration from similar tools like Devin.
- **Desired Functionality: Automated Notebook Modifications and Validation**: The envisioned system should be capable of intelligently replacing specific cells within a Jupyter Notebook, generating diverse notebook versions based on these modifications.
   - Crucially, the system should possess an agentic quality, enabling it to validate its outputs and iteratively refine the modifications until it achieves the desired results.


  

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

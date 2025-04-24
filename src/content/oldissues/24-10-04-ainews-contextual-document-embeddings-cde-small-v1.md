---
id: 2bac3a4c-2649-40d3-965c-5fc2ab97ddb2
title: 'Contextual Document Embeddings: `cde-small-v1`'
date: '2024-10-05T01:38:06.226049Z'
original_slug: ainews-contextual-document-embeddings-cde-small-v1
description: >-
  Meta’s new Movie Gen model claims superior video generation by adapting Llama
  3 better than OpenAI’s Sora, though no release is available yet. Meanwhile,
  Jack Morris and Sasha Rush introduced cde-small-v1, a highly efficient 143M
  parameter contextual embedding model that outperforms many larger models by
  incorporating contextual tokens during training.
tags:
  - meta
  - movie-gen
  - llama-3
  - openai
  - sora
  - contextual-embeddings
  - cde-small-v1
---


<!-- buttondown-editor-mode: plaintext -->**Contextual Batching is all you need.**

> AI News for 10/3/2024-10/4/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **31** Discords (**226** channels, and **1896** messages) for you. Estimated reading time saved (at 200wpm): **210 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

We often give the top story on AINews to movements of the big model labs, and today Meta's new text to video model, [Movie Gen](https://x.com/ahmad_al_dahle/status/1842188269557301607?s=46&t=6FDPaNxZcbSsELal6Sv7Ug), is sweeping the news, with [a paper](https://x.com/sytelus/status/1841960777588379656) that notably claims that they were able to adapt Llama 3 to video generation much better than OpenAI Sora's Diffusion Transformers. However, there is no actual release, just cherrypicked marketing videos, and we try to focus on news you can use here.

So we are happy to highlight Jack Morris and Sasha Rush's new paper and `cde-small-v1` model on [Contextual Document Embeddings](https://arxiv.org/abs/2410.02525), "**the best BERT-sized text embedding model in the world**".

![image.png](https://assets.buttondown.email/images/b1a9ffd9-ed18-4159-9925-22029f37649e.png?w=960&fit=max)

Jack puts it best:

> "Typical text embedding models have two main problems:
> 
> 1. training them is complicated and requires many tricks: giant batches, distillation, hard negatives...
> 2. the embeddings don't "know" what corpus they will be used in; consequently, all text spans are encoded the same way"
>
> To fix (1) we develop a new training technique: contextual batching. all batches share a lot of context – one batch might be about horse races in Kentucky, the next batch about differential equations, etc.
>
> 
> And for (2), we propose a new **contextual embedding** architecture. this requires changes to both the training and evaluation pipeline to incorporate **contextual tokens** – essentially, model sees extra text from the surrounding context, and can update the embedding accordingly

This seems to make sense - priming the embeddings model to adapt to context tokens first before doing proper embeddings.

While most [leaderboard-topping](https://huggingface.co/spaces/mteb/leaderboard) embeddings models are >7B in size (scoring ~72 on MTEB), the 143M parameter `cde-small-v1` scores a respectable 65 while sitting comfortably between models 50x larger. A nice efficiency win.

![image.png](https://assets.buttondown.email/images/ab55e9ab-ac48-4e7c-abb6-8f35ca02598a.png?w=960&fit=max)

While you're exploring new embeddings models, you might want to explore other advanced RAG techniques from [today's sponsor](http://wandb.me/ainews-course)!


---

> **Brought to you by RAG++**: Query refinement for RAG is like giving your system X-ray vision; with it, the system can “see“ user intentions more clearly - leading to more accurate chunk retrieval and more relevant LLM responses.
>
> [![image.png](https://assets.buttondown.email/images/05c0f424-b239-4561-bdc1-42322ae26689.png?w=960&fit=max)](http://wandb.me/ainews-course)
>
> 
> Learn about improving your RAG query refinement in this YouTube excerpt from Weights & Biases’ new course **[RAG++ : From POC to Production](http://wandb.me/ainews-course)** and sign up for free LLM api credits to get you started!

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

**AI Model and Company Updates**

- **OpenAI Developments**: OpenAI introduced Canvas, a new interface for collaborating with ChatGPT on writing and coding projects. [@karinanguyen_](https://twitter.com/karinanguyen_/status/1841888532299973056) highlighted key features including in-line feedback, targeted editing, and a menu of shortcuts. The canvas model was trained using novel synthetic data generation techniques, allowing for rapid iteration without relying on human data collection.

- **Google AI News**: [@_tim_brooks](https://twitter.com/_tim_brooks/status/1841982327431561528) announced joining Google DeepMind to work on video generation and world simulators. [@demishassabis](https://twitter.com/demishassabis/status/1841984103312208037) welcomed him, expressing excitement about making the long-standing dream of a world simulator a reality.

- **Model Releases and Updates**: Google released Gemini 1.5 Flash-8B, offering 50% lower prices and 2x higher rate limits compared to the previous version. [@_arohan_](https://twitter.com/_arohan_/status/1841904919772856631) mentioned that Flash 8B incorporates algorithmic efficiency improvements to pack as much as possible into a small form factor. [@bfl_ml](https://twitter.com/togethercompute/status/1841856799613600233) launched FLUX1.1 [pro], a new state-of-the-art diffusion model that delivers images 3x faster than its predecessor with improved quality.

**AI Research and Techniques**

- **Scaling Laws and Model Training**: [@soumithchintala](https://twitter.com/soumithchintala/status/1841931462427476431) discussed how modern transformers follow well-behaved scaling laws, allowing researchers to find hyperparameters at a smaller scale and then scale up parameters and data according to power laws. This approach increases confidence in larger training runs.

- **Inference Optimization**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1841854984142336460) shared a summary of transformer inference optimization techniques, including KV Cache, MQA/GQA, Sliding Window Attention, Linear Attention, FlashAttention, Ring Attention, and PagedAttention.

- **AI Safety and Alignment**: [@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1841968612250419464) expressed frustration with the focus on AI safety at the expense of potentially breakthrough research in neural networks, deep learning, and agent foundations.

**Industry Trends and Applications**

- **Voice AI and Call Centers**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1841833425432449066) highlighted the potential impact of OpenAI's Real-time API on the call center industry, with AI-powered calls costing significantly less than human agents.

- **AI in Healthcare**: [@BorisMPower](https://twitter.com/BorisMPower/status/1841936047858672066) noted that in a narrow test of professional doctors, AI performed better than human + AI, drawing parallels to observations in chess and Go.

- **Developer Tools and Interfaces**: Several tweets discussed the importance of novel interfaces for AI, with [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1841907725095338279) noting that better interfaces will make LLMs much easier to use, citing Cursor vs Copilot as an example.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Whisper Turbo: Significant Speed Improvements in Speech Recognition**

- **Open AI's new Whisper Turbo model runs 5.4 times faster LOCALLY than Whisper V3 Large on M1 Pro** ([Score: 80, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fvb83n/open_ais_new_whisper_turbo_model_runs_54_times/)): OpenAI's new **Whisper Turbo** model demonstrates **5.4x faster** local transcription compared to **Whisper V3 Large** on an **M1 Pro MacBook Pro**, processing a **66-second** audio file in **24 seconds** versus **130 seconds**. The post provides instructions for testing locally using the [nexa-sdk python package](https://github.com/NexaAI/nexa-sdk?tab=readme-ov-file#python-package) and includes links to both the **Whisper-V3-Large-Turbo** and **Whisper-V3-Large** models on nexaai.com.
  - **Faster-Whisper** outperforms **Whisper-Turbo** on an **RTX3090** Linux system, transcribing a **24:55** audio file in **14 seconds** vs **23 seconds**. The chunked algorithm is recommended for prioritizing transcription speed and long audio files.
  - Users report **Whisper Turbo** runs faster than real-time on MacBooks, opening possibilities for **local real-time assistant solutions**. The model supports multiple languages, not just English.
  - Discussions on **streaming input/output** for ASR models like Whisper highlight challenges due to its **30-second chunk architecture**. A working prototype exists but is less reliable compared to non-async architectures.


- **Finally, a User-Friendly Whisper Transcription App: SoftWhisper** ([Score: 62, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1fvncqc/finally_a_userfriendly_whisper_transcription_app/)): **SoftWhisper**, a new desktop app for **Whisper AI** transcription, offers an intuitive interface with features including a **built-in media player**, **speaker diarization** (using **Hugging Face API**), **SRT subtitle creation**, and the ability to handle long files. Developed using **Python** and **Tkinter**, the app aims to make transcription accessible, with the developer seeking feedback and potential collaborators for future improvements such as **GPU optimization**.
  - Users discussed **running the application**, with the developer providing a **tutorial** and **dependency_installer.bat** script for easier setup. The project now includes a **requirements.txt** file and instructions for **Python installation**.
  - A user shared a [GitHub repository](https://github.com/rmusser01/tldw/blob/main/App_Function_Libraries/Audio/Diarization_Lib.py) for **offline diarization** using **Pyannote**, which the developer expressed interest in exploring. The [offline usage of Pyannote](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/community/offline_usage_speaker_diarization.ipynb) was confirmed as permissible.
  - Suggestions for future improvements included **real-time capture capability** for meetings and support for **multiple audio stream videos**. The developer confirmed that **SoftWhisper** can transcribe video formats by extracting audio, though format support may be limited.


**Theme 2. Qwen 2.5: Controversy Over Chinese AI Models in Conservative Industries**

- **[Gemma 2 2b-it is an underrated SLM GOAT](https://i.redd.it/18x465phhnsd1.png)** ([Score: 92, Comments: 21](https://reddit.com//r/LocalLLaMA/comments/1fvowgm/gemma_2_2bit_is_an_underrated_slm_goat/)): **Gemma 2 2b-it** is praised as an exceptional **Small Language Model (SLM)**, outperforming many larger models in various benchmarks. The model demonstrates impressive capabilities, including **zero-shot reasoning**, **few-shot learning**, and strong performance in **coding tasks**, despite its relatively small size of **2 billion parameters**. Its efficiency and performance make it a strong contender in the SLM space, challenging larger models like **Mistral 7B** and **Llama 2 13B**.
  - A separate **leaderboard for Small Language Models (SLMs)** was suggested, with potential for **locally-run AGI** on smartphones. However, debate arose over the term "SLM", with some arguing that model size doesn't define whether it's a large or small language model.
  - The **Qwen2.5-3B-Instruct** model shows impressive performance compared to other small models like **Gemma2-2B-IT** and **Phi3.5-mini-Instruct**. A detailed performance comparison table was shared, highlighting Qwen's strengths in tasks like **MATH** (65.9%) and **GSM8K** (86.7%).
  - **Gemma 2 2b-it** is praised for its capabilities, with users noting its performance against older, larger models like **Claude 2** and **Gemini 1 Pro**. The model's efficiency and low cost for fine-tuning were also highlighted.


- **Qwen 2.5 = China = Bad** ([Score: 300, Comments: 232](https://reddit.com//r/LocalLLaMA/comments/1fv37i1/qwen_25_china_bad/)): The post discusses concerns about using the **Chinese AI model Qwen 2.5** in a **conservative industry**, where superiors have rejected its use due to fears of it being a **trojan** from **Alibaba**. The author argues that these concerns are unfounded, especially given plans to use the model **on-premise** without internet connection and to **finetune** it, potentially making it unrecognizable from its original form.
  - Users discussed potential **security risks** of LLMs, including **sleeper agents** that can persist through safety training and models trained to insert **exploitable code** under specific conditions. Some argued **air-gapping** and using **safetensors** format could mitigate risks.
  - Several commenters pointed out that while **technical risks** may be low, **perceived risks** can have real consequences for businesses, including impacts on **risk assessments**, **insurance premiums**, and **investor relations**. Some suggested using alternative models to avoid these issues.
  - There was debate about whether concerns over **Chinese models** like **Qwen** are justified. Some argued it's no riskier than other tech products made in China, while others cited examples of **Chinese espionage** and suggested caution when dealing with sensitive data or applications.


**Theme 3. XTC Sampler: New Technique to Reduce GPTisms in LLM Outputs**

- **[Say goodbye to GPTisms and slop! XTC sampler for llama.cpp](https://github.com/cyan2k/llama.cpp/tree/feature/xtc-sampler)** ([Score: 144, Comments: 45](https://reddit.com//r/LocalLLaMA/comments/1fv5kos/say_goodbye_to_gptisms_and_slop_xtc_sampler_for/)): The post introduces an **XTC sampler implementation** for **llama.cpp**, designed to reduce **GPTisms** and **slop** in language model outputs. This sampling method aims to improve the quality and coherence of generated text by addressing common issues associated with traditional sampling techniques used in large language models.
  - The **XTC sampler** implementation for **llama.cpp** aims to reduce **GPTisms** and improve creativity by ignoring top tokens during sampling. Users can find examples and usage instructions in the [GitHub repository](https://github.com/cyan2k/llama.cpp/tree/feature/xtc-sampler/xtc-examples).
  - Discussions arose about the effectiveness of XTC, with some users praising its ability to enhance creative writing, while others questioned its impact on general performance. The recommended parameter values are **threshold = 0.1** and **probability = 0.5**, with viable ranges of **0.05-0.2** for threshold and **0.3-1.0** for probability.
  - Debate ensued over whether removing top token candidates is the best approach for improving language model outputs. Some argued it could lead to decreased performance in non-creative tasks, while others emphasized its potential for reducing repetitive phrases and enhancing diversity in generated text.


- **[Quantization testing to see if Aphrodite Engine's custom FPx quantization is any good](https://www.reddit.com/gallery/1fv2bqp)** ([Score: 64, Comments: 32](https://reddit.com//r/LocalLLaMA/comments/1fv2bqp/quantization_testing_to_see_if_aphrodite_engines/)): Aphrodite Engine's custom **FPx quantization** was tested against standard **FP16** and **INT8** quantization methods. Results showed that **FPx** outperformed **INT8** and matched or slightly exceeded **FP16** performance, while offering potential memory savings. The testing utilized **MMLU** and **HumanEval** benchmarks, with plans for further evaluation using **TinyStories** and **Alpaca** datasets.
  - **Aphrodite's custom FP quantization** showed impressive results, with **FP6** recommended for <8-bit fast inferencing. **FP5** unexpectedly achieved the highest score (40.61%), potentially due to unintentional **Chain of Thought** reasoning.
  - Benchmark results revealed **GGUF Q4_K_M** performed surprisingly well, outperforming **GPTQ** and **FP4** quantizations. **Aphrodite's FP quants** demonstrated high speed, scaling faster at lower quantization levels, while **GGUF models** were notably slower.
  - The study concluded that **>4-bit quantization** using **Aphrodite's custom FP quants** is optimal for speed. For 4-bit or lower quantization, **GGUF** performs better. 8-bit quantization showed similar performance to full **BF16** models across methods.


**Theme 4. Tool Calling in Open-Source LLMs: Building Agentic AI Systems**

- **Tool Calling in LLMs: An Introductory Guide** ([Score: 73, Comments: 3](https://reddit.com//r/LocalLLaMA/comments/1fvdtqk/tool_calling_in_llms_an_introductory_guide/)): The post introduces **tool calling in LLMs**, defining tools as functions with **names**, **parameters**, and **descriptions** made available to language models. It explains that LLMs don't directly execute tools but generate a **structured schema** (usually a **JSON object**) containing the tool's name and parameter values when a relevant tool is identified for a given query. The post outlines a **4-step workflow** for tool calling, from defining a tool to generating a complete answer using tool outputs, and provides a link to an in-depth guide on using tool calling with agents in **open-source Llama 3**.

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Research and Techniques**

- **Google DeepMind advances multimodal learning**: A [new paper](https://arxiv.org/html/2406.17711v1) demonstrates how data curation via joint example selection can accelerate multimodal learning.

- **Microsoft's MInference speeds up long-context inference**: [MInference](https://arxiv.org/abs/2407.02490) enables inference of up to millions of tokens for long-context tasks while maintaining accuracy.

- **Scaling synthetic data creation**: A [paper on scaling synthetic data creation](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/) leverages 1 billion web-curated personas to generate diverse training data.

- **Exact volume rendering for NeRFs**: A [new paper](https://www.reddit.com/r/singularity/comments/1fvfhfj/new_paper_performs_exact_volume_rendering_at/) achieves exact volume rendering at 30FPS@720p, producing highly detailed 3D-consistent NeRFs.

**AI Model Releases and Improvements**

- **Salesforce releases xLAM-1b**: This 1 billion parameter model [achieves 70% accuracy in function calling, surpassing GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/).

- **Phi-3 Mini updated with function calling**: Rubra AI released an updated Phi-3 Mini model [with function calling capabilities](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/), competitive with Mistral-7b v3.

- **iPhone photo style LoRA for Flux**: A new [LoRA fine-tuning](https://www.reddit.com/r/StableDiffusion/comments/1fvpptg/iphone_photo_stye_lora_for_flux/) improves the realism of Stable Diffusion Flux outputs to match iPhone photo aesthetics.

**AI Industry Developments**

- **High demand for Nvidia's Blackwell AI chip**: Nvidia CEO Jensen Huang reports ["insane" demand](https://www.forbes.com/sites/antoniopequenoiv/2024/10/03/nvidia-shares-jump-after-ceo-jensen-huang-notes-insane-demand-for-blackwell-ai-superchip/) from major tech companies for their next-generation AI chip.

- **OpenAI discourages investors from backing competitors**: OpenAI is [asking investors not to fund certain AI competitors](https://www.reddit.com/r/singularity/comments/1fv86bx/the_vibes_are_off/), raising concerns about monopolistic practices.

- **Sora lead joins Google**: Tim Brooks, a lead researcher on OpenAI's Sora video generation model, [has joined Google](https://www.reddit.com/r/singularity/comments/1fvliry/sora_lead_tim_brooks_joins_google/).

**AI Ethics and Societal Impact**

- **Debate over AI alignment and corporate control**: Discussions around [OpenAI's shift towards profit-seeking](https://www.reddit.com/r/singularity/comments/1fv86bx/the_vibes_are_off/) and concerns about corporate control of AGI development.

- **EU AI regulation concerns**: French President Macron [warns that over-regulation and under-investment in AI](https://www.reddit.com/r/singularity/comments/1fvfq9o/the_eu_could_die_warns_macron_on_overregulation/) could harm the EU's competitiveness.

- **Unions and AI adoption**: A Swedish union leader's [perspective on embracing new technology](https://www.reddit.com/r/singularity/comments/1fv47p3/swedens_union_leaders_views_on_new_technology/) while protecting workers highlights the need for retraining and adaptation.

**AI Capabilities and Milestones**

- **Claims of human-level reasoning**: OpenAI CEO Sam Altman suggests they've [reached human-level reasoning capabilities](https://www.reddit.com/r/singularity/comments/1fvd7uv/altman_we_just_reached_humanlevel_reasoning/), though the exact meaning and implications are debated.

- **Improvements in image generation**: Demonstrations of [highly realistic photo generation](https://www.reddit.com/r/StableDiffusion/comments/1fvs0e1/ultra_realistic_photos_on_flux_just_by_adding_img/) using Stable Diffusion Flux, though some claims are disputed.


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1: Meta Unveils Movie Gen, Revolutionizes Video Generation**

- **Meta Premieres Movie Gen, Redefines Multimedia Creation**: [Meta's Movie Gen](https://ai.meta.com/research/movie-gen) introduces advanced models that generate high-quality images, videos, and synchronized audio from text prompts. Capabilities include precise video editing and personalized content generation.
- **AI Community Buzzes Over Movie Gen's Potential**: The [Movie Gen research paper](https://ai.meta.com/static-resource/movie-gen-research-paper) showcases groundbreaking techniques in video content creation. Meta is collaborating with creatives to refine the tool before wider release.
- **Movie Gen Sparks Excitement Across AI Forums**: Discussions highlight Movie Gen's promise to push the boundaries of AI-generated video, with enthusiasts eager to explore its applications in multimedia projects.

**Theme 2: New AI Models and Benchmarks Lead the Charge**

- **Nvidia Drops a Bombshell with GPT-4 Rival**: Nvidia's new AI model is **open, massive**, and set to challenge GPT-4, as reported by [VentureBeat](https://venturebeat.com/ai/nvidia-just-dropped-a-bombshell-its-new-ai-model-is-open-massive-and-ready-to-rival-gpt-4/). The AI community is eager to see how it stacks up.
- **Finance LLM Leaderboard Crowns Top Performers**: A new [LLM leaderboard](https://huggingface.co/blog/leaderboard-finbench) for finance ranks **OpenAI's GPT-4**, **Meta's Llama 3.1**, and **Alibaba's Qwen** as leaders across 40 tasks. This offers fresh metrics for evaluating models in financial applications.
- **Gemini 1.5 Flash-8B Delivers Budget-Friendly AI Power**: Now available on [OpenRouter](https://openrouter.ai/models/google/gemini-flash-1.5-8b) at **$0.0375 per million tokens**, Gemini 1.5 Flash-8B provides a cost-effective option without sacrificing performance.

**Theme 3: Advances in Model Optimization and Training Techniques**

- **TorchAO Lights Up PyTorch with Model Optimization**: The new [torchao library](https://pytorch.org/blog/quantization-aware-training/) introduces quantization and low-bit datatypes, boosting model performance and slashing memory usage. It's a significant leap forward for PyTorch users.
- **SageAttention Speeds Past Competitors**: [SageAttention](https://github.com/thu-ml/SageAttention) achieves **2.1x** speedups over FlashAttention2 and **2.7x** over xformers, all without losing accuracy. This quantization method turbocharges attention mechanisms.
- **VinePPO Unlocks RL Potential in LLMs**: The [VinePPO algorithm](https://arxiv.org/abs/2410.01679) addresses credit assignment issues in LLM reasoning tasks, outperforming PPO with up to **9x fewer steps** and **3x less time**, while using half the memory.

**Theme 4: OpenAI's Canvas Tool and Models Stir Mixed Reactions**

- **OpenAI's Canvas Tool Sparks Joy and Frustration**: The new [Canvas tool](https://openai.com/index/introducing-canvas/) streamlines coding by integrating features and reducing scrolling. However, users lament missing essentials like a **continue button** and face editing hiccups.
- **Advanced Voice Mode Could Elevate Coding**: Discussions suggest that combining **Advanced Voice Mode** with Canvas could enhance programming workflows. Community-shared [setup guides](https://github.com/jjmlovesgit/ChatGPT-Advanced-Voice-Mode) aim to smooth integration.
- **OpenAI's o1 Models Impress Developers**: The introduction of [o1-preview and o1-mini models](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0) enhances chatbot capabilities. Users note **o1-mini**'s surprising prowess in tackling complex tasks.

**Theme 5: Recurrent Neural Networks Make a Comeback**

- **RNNs Strike Back with 175x Faster Training**: The paper “[Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201)” reveals that **minLSTMs** and **minGRUs** without hidden state dependencies train dramatically faster, reigniting interest in RNN architectures.
- **Minimalist RNNs Enable Efficient Parallel Training**: By eliminating backpropagation through time, these simplified RNNs allow for parallel computation, challenging Transformers in sequence modeling efficiency.
- **Community Explores RNNs' Modern Potential**: Enthusiasts discuss how streamlining RNNs can lead to scalable training methods suitable for today's AI demands, potentially reshaping the landscape of neural network architectures.


---

# PART 1: High level Discord summaries




## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **torchao Library Introduces Model Optimization**: The [torchao library](https://huggingface.co/posts/singhsidhukuldeep/639926000427051) from PyTorch features quantization and low-bit datatype techniques, boosting model performance and memory use.
   - It promises automatic quantization alongside existing tools, marking a significant advancement in PyTorch.
- **OpenAI's Canvas Tool Streamlines Coding**: OpenAI's [Canvas tool](https://openai.com/index/introducing-canvas) has garnered excitement for its integrated features, reducing unnecessary scrolling during coding.
   - Users noted that its editing capabilities are a significant advancement over previous tools like Claude.
- **Meta's Movie Gen Models Show Great Potential**: Meta has launched its [Movie Gen models](https://ai.meta.com/static-resource/movie-gen-research-paper) that generate high-quality multimedia from text prompts.
   - These models feature precise video editing and personalized generation, highlighting their creative applications.
- **Cultural Biases Limit AI Training Understanding**: Current discussions point out that LLM training lacks human biases and relies heavily on large datasets, affecting concepts like love and morality.
   - Members question how AI might 'learn' these complex emotions without true inherent understanding.
- **VinePPO Addresses LLM Credit Assignment**: The paper on **VinePPO** critiques Proximal Policy Optimization (PPO) for its inconsistency in reasoning tasks and introduces a refinement to tackle credit assignment.
   - It shows that existing value networks in PPO yield high-variance updates, merely outperforming random baselines.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider's Telemetry Needs Urged**: Members highlighted the importance of **telemetry in Aider**, suggesting opt-in features for user privacy while improving insight on performance.
   - *System call tracing* was proposed to diagnose performance issues, emphasizing the need for **transparency** about the data collected.
- **OpenRouter Free Models put to the Test**: **OpenRouter's free models** present strict account-wide limits of **200 messages per day**, impacting flexibility for users wanting more access.
   - Participants raised concerns about lacking paid options for certain models, questioning the overall usability.
- **Benchmarking Models Raises Questions**: Participants shared experiences from **benchmarking various models**, noting mixed performance on processing error rates.
   - Aider’s ability to manage editing tasks was a focal point, with users reporting issues linked to token limits alongside specific errors.
- **Ollama Model Performance with Aider**: Users reported **slow response times** while using Aider with **Ollama's local 8B model**, questioning the benefit of paid API keys.
   - Discussions revealed local models may struggle with editing tasks, indicating a preference for models with stronger editing capabilities.
- **Exploring File Addition Complexity**: Testing the **/read-only** command in Aider illustrated it now only completes tasks by folder, complicating file access.
   - Another user confirmed that correct usage should still add all files, revealing nuances in command functionality.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Salamandra on-device demo shines**: [Salamandra](https://huggingface.co/spaces/Tonic/salamandra-on-device) demo showcased impressive capabilities, engaging users while highlighting its features.
   - *The excitement around Salamandra's spotlight in the community* reflects the growing interest in on-device AI applications.
- **Nvidia launches a game-changing AI model**: Nvidia's new AI model is **open, massive**, and prepared to **rival GPT-4** according to a report from [VentureBeat](https://venturebeat.com/ai/nvidia-just-dropped-a-bombshell-its-new-ai-model-is-open-massive-and-ready-to-rival-gpt-4/). The community is eager to see how this model will compete and what unique capabilities it possesses.
   - This announcement has stirred excitement within the AI community.
- **OpenAI introduces new models**: Two new **OpenAI** models, **o1-preview** and **o1-mini**, were integrated into the [open-source chatbot](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0), enhancing its functionality. Members celebrated these additions as a significant leap towards more robust chatbot experiences.
- **MusicGen iOS app shows progress**: Updates on the iOS app for **MusicGen** reveal features including a noise cancel for input audio and a 'tame the gary' toggle, focusing on drums. *One member remarked* that it aims for refined audio input-output integration, targeting enhanced user experience.
- **AI Sentience Prediction raises questions**: An article titled 'The Sentience Prediction Equation' discusses potential future AI sentience and its implications, questioning if AI will ponder its purpose. It humorously notes AI might ask, *'Why do humans insist on putting pineapple on pizza?'* introducing the Prediction Equation as an estimation tool.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Canvas Model Enhancements Spark Excitement**: The new **Canvas model** is generating buzz, with members discussing its potential functionality and integration with [GPT-4o](https://openai.com/index/introducing-canvas/). However, frustration arose due to missing features like a **continue button** and editing issues.
   - Users are hopeful that improvements will enhance the UX for programming tasks while addressing current limitations.
- **Advanced Voice Mode Could Boost Integration**: Conversations about the **Advanced Voice Mode** highlighted its potential synergy with the **Canvas tool** for smoother user experiences in coding. Community members circulated setup guides on GitHub to aid seamless integration.
   - They proposed features like real-time API integration to boost coding efficiency as an exciting next step.
- **Custom GPTs Experience Mixed Results**: Users reported challenges with integrating **Google API/OAuth** within Custom GPTs during its initial rollout, causing some concern about its reliability. They have yet to check in on recent improvements regarding stability.
   - This lack of consistency has left some users wary about re-engaging with the integration.
- **ChatGPT's Evaluation Inconsistencies Take Center Stage**: Frustrations emerged over inconsistent evaluations from **ChatGPT** when tasked with scoring answers on a scale at **temperature 0.7**, prompting suggestions for stricter grading scales. A user recommended using a **grading rubric** to enhance clarity and consistency.
   - Another proposed the **Chain-of-Thought** reasoning framework to improve scoring accuracy and evaluative clarity.
- **Efficient JSON Processing Tips Shared**: A developer sought advice on parsing 10,000 snippets into JSON with **GPT-4o** and inquired about the necessity of resending protocol parameters for each snippet. Suggestions encouraged optimization by only sending new snippets during processing.
   - This conversation illustrates the ongoing need for cost efficiency in model interactions and JSON handling.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth AI Projects streamline fine-tuning**: Members discussed using **Unsloth AI** for continual pretraining of LLMs, achieving up to **2x faster** training while using **50% less VRAM** compared to traditional methods.
   - Essential tools like the [continued pretraining notebook](https://docs.unsloth.ai/basics/continued-pretraining) were emphasized for expanding model training capabilities.
- **ZLUDA's funding brings new hopes**: ZLUDA's development has secured backing from a new commercial entity, targeting enhanced functionality for LLMs.
   - Concerns linger about possible legal disputes with **NVIDIA**, echoing issues experienced in previous equity backing scenarios.
- **Generational Preferences: A humorous take**: Members playfully debated their generational identities, one claiming to feel as a **boomer** at just 24, touching on cultural perceptions.
   - The lighthearted conversation noted that **Legos** and **modded Minecraft** define generational boundaries, hinting at shifting cultural practices.
- **Local inference script woes**: A member faced challenges with their local inference script for **gguf models** using **llama-cpp**, reporting sluggish performance despite a capable GPU.
   - Suggestions like using **llama-cli** emerged, indicating a potential for enhanced script efficiency.
- **Revival of Recurrent Neural Networks**: A recent paper suggests **minimal LSTMs** and **GRUs** trained **175x faster** by eliminating hidden state dependencies, sparking renewed interest in RNNs.
   - This finding points towards new possibilities in scalable training methods relevant to modern architectures.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **IREE faces unpredictable adoption timelines**: Members discussed whether large labs might adopt **IREE** for serving models at scale, amid indications that many use custom inference runtimes.
   - Some noted that it's typical for new technologies like IREE to have **unpredictable** adoption timelines.
- **RWKV introduces efficient parallelization**: **RWKV** employs partial parallelization by structuring networks into smaller layers, enabling computations while waiting for token inputs.
   - This approach aims to streamline performance while managing model interdependencies effectively.
- **Exploring Linear Attention models**: Dialogue focused on linear attention and gated linear attention's capacity to function as RNNs, enabling parallel computations across sequences.
   - Interest grew around **Songlin Yang's** research uncovering complex RNN classes improving parallelization.
- **VinePPO struggles with credit assignment**: The **VinePPO** paper outlines how value networks face credit assignment challenges in complex reasoning tasks, underperforming against random baselines.
   - This emphasizes the necessity for improved models or techniques to optimize credit assignment in **Proximal Policy Optimization (PPO)**.
- **lm-evaluation-harness seeks contributors**: The **lm-evaluation-harness** is inviting contributions for integrating new LLM evaluations and addressing bugs, with many issues available to tackle.
   - Potential contributors can find more detailed information in the [GitHub repository](https://github.com/EleutherAI/lm-evaluation-harness).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **SambaNova AI Impresses with Throughput**: [SambaNova AI](https://x.com/SambaNovaAI/status/1841901026821210131) launched their endpoints for **Llama 3.1 and 3.2** on OpenRouter, claiming the fastest throughput measurements recorded.
   - They noted, *‘These are the fastest we’ve seen’*, indicating a significant edge in their throughput metrics compared to competitors.
- **Gemini 1.5 Flash-8B Officially Launches**: The **Gemini 1.5 Flash-8B** model is now available, priced at **$0.0375 per million tokens**, making it a noteworthy budget option compared to peers.
   - For access, check the link [here](https://openrouter.ai/models/google/gemini-flash-1.5-8b); discussions have also centered on its performance scaling potential.
- **o1 Mini Surprises with Task Performance**: **o1 Mini** has shown improved capability in resolving complex tasks, exceeding community expectations for its performance.
   - A member mentioned plans to utilize **o1 Mini** for a bot handling image descriptions, showcasing its practical applications.
- **Anthropic Rides Funding Wave**: Discussions revealed that **Anthropic**'s rapid model development, particularly for **Claude**, stems from a team of ex-OpenAI engineers and backing from **Amazon**.
   - Speculations arose regarding how Anthropic competes effectively in performance with less financial support compared to giants in the sector.
- **OpenRouter Infrastructure Expansions on the Horizon**: Anticipation builds around expansions in **OpenRouter** to accommodate diverse model functionalities, including image and audio processing.
   - Development leads are confirmed to be actively working on upgrades to handle increased traffic and new model releases.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Langflow Integration Boosts LM Studio**: LM Studio is now integrating support for Langflow, as highlighted in a [recent GitHub pull request](https://github.com/langflow-ai/langflow/pull/4021), enhancing functionalities for building LLM applications.
   - This integration is set to streamline user experience and broaden the capabilities of LM Studio.
- **Memory Leak Drama with v0.3.2.6**: Users reported significant memory leak issues with LM Studio version **v0.3.2.6**, which resulted in models generating nonsensical output.
   - Recommendations suggest checking if the problem persists in version **v0.3.3** for resolution.
- **Model Downloading Troubles Trigger Errors**: A persistent issue with model downloads from Hugging Face surfaced, where errors occurred while selecting models in LM Studio.
   - Members suggested [sideloading models](https://lmstudio.ai/docs/advanced/sideload) directly into the models directory to bypass these errors.
- **Chat Cache Location Not Customizable**: Questions arose regarding the ability to customize the chat cache location in LM Studio, which is currently hardcoded.
   - LM Studio saves conversation data in JSON format, but there are no options for changing the cache location at this time.
- **AI Model Recommendations Spark Discussions**: Discussions highlighted **Llama-3-8B** as not meeting expectations for some users when used as a chatbot assistant.
   - Users were encouraged to explore various options on the [LM Studio Model Catalog](https://lmstudio.ai/models) for potentially better fits.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LangChain launches Voice ReAct Agent**: LangChain introduced a **Voice ReAct Agent** leveraging the [Realtime API](https://x.com/langchainai/status/1841914757764485439?s=46) for custom voice experiences, demonstrated with an agent using a calculator and a [Tavily web search tool](https://youtu.be/TdZtr1nrhJg).
   - This innovative agent showcases new possibilities for voice interaction in interactive applications.
- **GPT-4o Bots chat up a storm**: A demo highlighted two **GPT-4o Voice AI bots** conversing using the Realtime API, underlining the advancements in voice AI technology.
   - The bots exhibited impressive *turn-taking latency*, revealing notable improvements in interaction fluidity.
- **Meta Movie Gen strides into video generation**: Meta showcased its latest project, **Meta Movie Gen**, aimed at pioneering **video generation** but without a set release date. More details can be explored on their [AI research page](https://ai.meta.com/research/movie-gen/) and its [associated paper](https://ai.meta.com/static-resource/movie-gen-research-paper).
   - The project promises to push the boundaries of video content creation, driven by state-of-the-art models.
- **New LLM leaderboard introduces finance leaders**: The latest **LLM leaderboard** for finance positions **OpenAI's GPT-4**, **Meta's Llama 3.1**, and **Alibaba's Qwen** as top performers across 40 relevant tasks, as explained in a [Hugging Face blog post](https://huggingface.co/blog/leaderboard-finbench).
   - This evaluation method offers a fresh approach to measuring model performance in financial applications.
- **Luma AI sparks interest in 3D modeling**: Enthusiastic discussions about **Luma AI** emphasized its potential in creating lifelike 3D models for platforms like Unity and Unreal, with members sharing various functional showcases.
   - Luma AI's capabilities were highlighted in its applications for film editing and detailed 3D models, indicating its promise in creative tech.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Performance Benchmarks Inquiry**: Members are seeking **performance benchmarks** for tools and methodologies, especially comparing these metrics to raw performance from **fio tools**.
   - There's a drive to analyze the **data access methods** to understand their effectiveness against traditional performance metrics.
- **OpenAI's Financial Success**: **OpenAI** is reportedly setting financial records thanks to recent innovations, with speculations on hardware development to leverage this growth.
   - Conversations growing around **new product development** point to possibilities of a mobile device focusing on user data applications, reminiscent of **Apple's** privacy concerns.
- **Event Planning Strategies**: The event planning timeline suggests it might occur around **September**, aligning with the school season to encourage attendance.
   - Colocation with the **Triton** and **PyTorch** conferences has been proposed for better group travel, showcasing effective planning strategies.
- **Triton Kernel Challenges**: Users are troubleshooting **Triton kernels**, especially facing issues with non-contiguous inputs, indicating a possible need for **reshape**.
   - There are also persistent problems with **OptimState8bit** dispatch errors, spotlighting limitations of 8-bit optimizer implementations.
- **Need for a Hyperparameter Scaling Guide**: A member called for a **hyperparameter scaling guide**, indicating confusion due to the lack of clear heuristics for larger model training.
   - Concerns about training methodologies suggest a gap in accessible resources that could support community members in this technical area.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI Updates Collections UI**: Perplexity AI is enhancing its Collections feature with a new UI to support **custom instructions** and files uploads, slated for future deployment.
   - The upcoming **Files search feature** aims to improve information organization and user experience.
- **Boeing 777-300ER Specs Released**: A detailed outline of the **Boeing 777-300ER** specifications has been shared, covering dimensions, performance, and capacity.
   - Key highlights include a **maximum range** of **7,370 nautical miles** and the potential to seat up to **550 passengers**.
- **TradingView Premium Cracked Version Disclosed**: A free cracked version of **TradingView Premium** (Version 2.9) was circulated, offering advanced trading tools without fees.
   - This disclosure has generated interest among traders seeking improved charting capabilities.
- **Llama 3.2 Release Anticipated**: Users are buzzing about the expected features and release date of **Llama 3.2**, showing keen interest in its advancements.
   - The community is excited about potential innovations that this new iteration could bring.
- **Claude 3.5 Outshines Competitors**: Discussion emerged comparing **Claude 3.5 Sonnet** to other models, with many asserting its reliability in information retrieval.
   - Members highlighted the synergy of **Perplexity Pro** and Claude for improved data extraction from resources.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R 08-2024 Fine-tuning Highlights**: The updated *Command R 08-2024* introduces support for newer options designed to provide users with **more control** and **visibility**. This update features a **seamless integration** with [Weights & Biases](https://cohere.com/blog/fine-tuning-command0824) for enhanced performance tracking.
   - Members expressed **enthusiasm** for the Command R update, with comments like '*Awesome*' capturing the excitement and anticipation from the community.
- **Metrics are missing in the platform**: A user reported that they are unable to see the **metrics boxes** for their models across various tabs like Overview and API, which previously displayed essential information. They highlighted that it's taken **2 days** without resolution.
   - This has raised concerns about the consistency of the platform, questioning the status of model creation.
- **Pricing Page Confusion**: The **pricing page** indicates **$3 per 1M tokens** for training, but the finetune UI shows a price of **$8**. This discrepancy raises questions about the accuracy of the pricing information across different platforms.
   - This has caused confusion that could impact users budgeting for training and fine-tuning projects.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Finding OpenPose Alternatives**: Users expressed frustrations with **OpenPose** when generating sitting poses, prompting discussion of alternatives like **DWPose** and exploring custom model training options.
   - *Training one’s own model could also be a viable solution with sufficient reference images available.*
- **Improving ComfyUI's Image Quality**: A member raised questions on achieving **ComfyUI** outputs comparable to **Auto1111**, as recent images appear cartoony in quality.
   - *Specific nodes in ComfyUI were recommended as potential methods for better quality outputs.*
- **Clarity on SDXL Model Varieties**: Multiple versions of **SDXL** were under discussion, particularly `SDXL 1.0`, covering aspects like starting resolutions at **1024x1024**.
   - *Participants confirmed that all variations relate back to the **SDXL 1.0** model framework.*
- **Reference Images Yielding Poses**: It was confirmed that generating poses with a single reference image is feasible in **Stable Diffusion**, though accuracy may suffer.
   - *The img2img feature was highlighted as the correct approach, suggesting that multiple reference images would improve fidelity.*
- **Query for AI Object Placement Tools**: Discussions uncovered interest in **OpenPose** techniques to assist with object placement, specifically regarding a LoRA model for items like swords.
   - *While various training styles in Stable Diffusion exist, users noted a gap in dedicated posing methods.*



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **MinGRU Architecture Takes Recurrent Networks Down a Notch**: The introduction of **minGRUs** proposes a simpler form of **GRUs** that eliminates hidden state dependencies, boosting training speed by **175x**.
   - This paper highlights that all it takes are two linear layers to achieve parallel hidden state computations, sparking conversations about simplifying **NLP** architectures.
- **Hunting for Resources to Build a BARK Model**: A newcomer is eager to train a **BARK-like model** from scratch within **2-3 months** but struggles to find relevant literature.
   - They noted connections between BARK and models like **Audio LM** and **VALL-E**, seeking community suggestions for papers to steer their training efforts.
- **Navigating Language Challenges in Tech**: A member raised concerns about the predominance of **English** in technical discourse, stating that many complex terms, like **embeddings** and **transformers**, often lack straightforward translations.
   - *Frustration with language preferences* complicates technical discussions, as effective communication hinges on shared terminology.
- **Community Scam Alerts Keep Members Cautious**: Numerous warnings surfaced about potential scams targeting members with false promises of earning **$50k** in **72 hours** for a 10% share of profits.
   - Individuals were advised to approach such schemes with skepticism, especially those involving unsolicited **Telegram** outreach.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Inquiry on Article Scores Sparks Interest**: A member asked how to view scores for three articles they submitted, including a draft and LinkedIn links, which underscores ongoing concerns about submission feedback.
   - *Submission feedback remains a hot topic* among members seeking clarity on their contributions.
- **Real-time Streaming Stalled by Garbage Collection**: One member expressed a desire to stream **chat_manager** responses directly into the frontend in real-time, noting current responses stream only post garbage collection.
   - Another confirmed a **Streamlit UI** had been created around 8 months ago, resolving this challenge.
- **Chainlit Shows Promise for Chat Management**: A member indicated a solution using **Chainlit** exists, with a potential recipe in the AutoGen project on GitHub to facilitate real-time chat features.
   - This implementation could effectively address the needs for improved chat management highlighted in ongoing discussions.
- **GitHub Pull Request Chat Processing Insights**: A member shared a relevant [GitHub pull request](https://github.com/microsoft/autogen/pull/1783) that focuses on processing messages before sending them, enhancing customization.
   - This development aligns with previous inquiries about real-time streaming, showing community momentum towards improved features.
- **Campus Course Location Clarified**: A member inquired about the specific room on Berkeley Campus for a certain course, highlighting logistical concerns among participants.
   - *Coordinating activities* seems crucial as community members navigate their educational requirements.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Build AI Agents with LlamaCloud**: Learn how to build AI agents using [LlamaCloud and Qdrant Engine](https://twitter.com/llama_index/status/1841935964081627315), focusing on implementing **semantic caching** for better speed and efficiency.
   - The demo includes advanced techniques like **query routing** and **query decomposition** to optimize agent interactions.
- **Enhance Security in RAG Deployments**: A discussion emerged about utilizing [Box's enterprise-grade security](https://twitter.com/llama_index/status/1841950022835044833) combined with LlamaIndex for secure RAG implementations.
   - Members stressed the significance of a **permission-aware RAG** experience to ensure robust data handling.
- **Voice Interaction with OpenAI's APIs**: Marcus showcased a new function using [OpenAI's real-time audio APIs](https://twitter.com/llama_index/status/1842236491784982982) that enables voice commands for document chat.
   - This feature revolutionizes document interaction, allowing users to engage via spoken language.
- **Combat Hallucination in RAG**: [CleanlabAI's solution](https://twitter.com/llama_index/status/1842259131274817739) tackles hallucination issues in RAG by implementing a trustworthiness scoring system for LLM outputs.
   - This methodology boosts data quality by pinpointing and removing unreliable responses.
- **Exciting Hackathon Opportunity Announced**: The upcoming hackathon, featuring over **$12,000 in cash prizes**, kicks off on October 11th at [500 Global VC's headquarters](https://twitter.com/llama_index/status/1842274685989576947) in Palo Alto.
   - Participants will have a chance to create innovative projects while competing for substantial cash rewards throughout the weekend.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Live Demos of dslmodel Scheduled**: Interactive coding sessions for **dslmodel** live demos occur at **4:30 PST**, inviting participation in the coding lounge.
   - These demos aim to showcase real-time applications and user engagement with the dslmodel functionalities.
- **Sentiment Analysis Results Impress**: The **SentimentModel** accurately classified the phrase ‘This is a wonderful experience!’ with **sentiment='positive'** and a confidence level of **1.0**.
   - This highlights its effectiveness in sentiment classification tasks, providing users reliable outcomes.
- **Summarization Model Effectively Captures Themes**: Using the **SummarizationModel**, the document's key message was distilled to: '**Motivational speech on success and perseverance.**'
   - The model effectively pinpointed themes of control, success, and resilience, illustrating its capability in summarization tasks.
- **DSPy Decodes Its Acronym**: Members clarified that **DSPy** stands for **Declarative Self-improving Language Programs**, also cheekily dubbed **Declarative Self-Improving Python**.
   - The conversation showcased community engagement and humor while navigating the interpretations of the DSPy acronym.
- **DSPy Signatures Explained**: A user shared details on **DSPy signatures**, emphasizing their role as declarative specifications for module input/output behaviors.
   - These signatures provide a structured way to define and manage module interactions, diverging from standard function signatures.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Event Participation Limit Rolls Back to 25**: Members noted that participation for the event was capped at **25 people**, despite a proposed change to **99** by MikeBirdTech.
   - One user confirmed repeated attempts to join but still encountered a **full** status.
- **Join the Human Devices Event**: MikeBirdTech shared the link for the upcoming **Human Devices event**: [Join Here](https://discord.gg/mzcrk6pZ?event=1291393902758330389).
   - Participants are encouraged to **request or share** anything related to the event in the designated channel.
- **Obelisk: A Handy GitHub Tool**: A member highlighted the **Obelisk** project from GitHub, a tool for saving web pages as a single **HTML file**.
   - They suggested it could be **quite useful in many contexts**, providing a link to explore: [GitHub - go-shiori/obelisk](https://github.com/go-shiori/obelisk).
- **Meta Movie Gen Launches**: Today, [Meta premiered Movie Gen](https://x.com/aiatmeta/status/1842188252541043075?s=46&t=G6jp7iOBtkVuyhaYmaDb0w), a suite of advanced media foundation models designed to enhance video and audio creation.
   - The models generate high-quality images, videos, and synchronized audio with impressive alignment and quality.
- **Mozilla's Open Source Vision**: In a discussion about Meta Movie Gen's openness, a member clarified that while **Mozilla** promotes open source, this initiative is more about showcasing their vision.
   - The distinction between Mozilla's principles and the nature of Movie Gen highlights its alignment with broader goals.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **FAANG Companies Demand SDLC Certification**: A user inquired about recognized courses for **Software Development Lifecycle (SDLC)** certifications acknowledged by **FAANG** companies, aside from **PMP**.
   - This poses a significant concern for applicants transitioning from various industries into tech roles.
- **LangChain API Calls Changing**: A member noticed changes in the **API chain** for LangChain and seeks the latest methods for API calls.
   - This highlights the continuous updates and developments within the **LangChain** framework.
- **LangChain Takes on GPT Real-time API**: A user asked when **LangChain** would support the recently announced **GPT real-time API**, referencing upcoming integration.
   - Further clarification was provided via a [YouTube video](https://www.youtube.com/watch?v=TdZtr1nrhJg) addressing these inquiries.
- **Evaluating RAG Pipeline Retrievers**: Advice was sought on evaluating and comparing performance among three different **retrievers** in a **RAG pipeline**.
   - One member suggested using **query_similarity_score** to identify the top-performing retriever and offered to share code snippets through LinkedIn.
- **User Interest in LangChain Chatbots**: A user requested guidance on creating their own **chatbot** using **LangChain**.
   - This indicates a rising interest in utilizing **LangChain** for chatbot development.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **NeurIPS 2024 adjusts dates for Taylor Swift fans**: The start date for the **NeurIPS 2024** conference has been moved to **Tuesday, December 10**, humorously noted due to **Taylor Swift's Eras Tour** influence.
   - This change allows delegates to arrive a day earlier, aligning better with travel plans, as highlighted in a [tweet](https://fxtwitter.com/WilliamWangNLP/status/1841879266142904469).
- **Elon Musk hosts a security-heavy xAI recruiting bash**: **Elon Musk's xAI** recruiting event featured live music generated via code amid ID checks and metal detectors, generating excitement in AI recruitment.
   - This event coincided with **OpenAI's Dev Day**, stirring discussion as Musk aims to attract top talent amid **funding rumors**.
- **OpenAI CEO speaks at a packed Dev Day**: **Sam Altman**, CEO of **OpenAI**, addressed a full house of developers during their annual **Dev Day**, promoting recent advancements and upcoming projects.
   - Rumors about OpenAI closing in on a record-breaking funding round circulated during the event.
- **Meta Movie Gen Launches Advanced Features**: Meta premiered **Movie Gen**, a suite of media foundation models capable of generating high-quality images, videos, and audio from text prompts, boasting impressive capabilities like personalized video creation.
   - They reported *working closely with creative professionals* to enhance the tool's features before a broader release.
- **Reinforcement Learning Enhances LLMs for Code**: A new paper proposes an end-to-end reinforcement learning method for **LLMs** in competitive coding tasks, achieving state-of-the-art results while improving efficiency.
   - This method shows how execution feedback can drastically reduce sample requirements while enhancing catalyst performance.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tensors: Permuting vs Reshaping Dilemma**: A member inquired whether to use `.permute` or `.reshape` to transform a target tensor from sizes (1024,1,14,1) to (14,1024,1), highlighting the complexities of tensor operations in deep learning.
   - *Dumb q.* reflects some frustration, indicating a need for clarity on tensor manipulation best practices.
- **Efficient Stable Diffusion Training**: An inquiry was raised regarding the feasibility of training a **Stable Diffusion** model on an **M3 MacBook Air** within **48 hours**, signaling interest in efficient model training methods.
   - This suggests a demand for streamlined resources that make high-performance training more accessible to users.
- **Need for Enhanced bfloat16 Tests**: George emphasized the importance of increasing **bfloat16 tests** in tinygrad, pointing out the current limitations in `test_dtype.py`.
   - A member questioned what *additional tests* would actually enhance the robustness of the testing framework.
- **Check Out These Triton Talks**: A member shared a YouTube link to a **Triton talk** that covers various developments within Triton technology, providing insights for developers.
   - You can watch it [here](https://www.youtube.com/watch?v=ONrKkI7KhU4) to gain a deeper understanding of Triton's capabilities.
- **Analyzing Tinygrad CI Warnings and Failures**: A call went out for insights into recent **CI warnings** during Tinygrad's test runs, aiming to improve the framework's reliability.
   - Reviewing the [node cleanup and test speeds](https://github.com/tinygrad/tinygrad/actions/runs/11177982687/job/31074623873?pr=6880) boosts understanding of recent changes and stability efforts.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune's KTO Training Query**: A user asked if **Torchtune** supports **KTO training**, indicating interest in its capabilities for efficiency.
   - No further details or responses were shared in this thread.
- **VinePPO transforms RL for LLM Reasoning**: A member showcased **VinePPO**, a modification to PPO, achieving up to **9x fewer steps** and **3x less time** for RL-based methods.
   - These results suggest a potential shift in **RL post-training** approaches, with significant memory savings as well.
- **Flex Attention boosts runtime efficiency**: **Flex Attention** preserves runtime performance by leveraging **block sparsity** in attention masks, showing equal performance for **bsz=1** and **bsz=2** setups.
   - Testing has confirmed that processing **1000 tokens** retains time and memory efficiency similar to batching.
- **Streamlining Batch Size in Packed Runs**: A proposal was made to eliminate the batch size option in packed runs, focusing on **tokens_per_pack** for a stable **bs=1**.
   - This could enhance efficiency and simplify performance metrics considerations.
- **DDP Implementation Discussion**: Members speculated on the integration of **Distributed Data Parallel (DDP)**, with each sampler set to **bsz=1**, optimizing single device resource usage.
   - This could potentially improve performance allocation across devices.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **AI boosts network speeds while software lags**: Recent discussions noted that **AI advancements** have made **100 Gbps** technology more affordable, with labs achieving **1.6 Tbps**.
   - *Darkmatter* highlighted that software hasn't kept up with the **80x bandwidth increase**, resulting in challenges even at **10 Gbps**.
- **Urgency to enhance network capabilities**: *Luanon404* expressed a strong desire for improvements in networking, declaring, *'it's time to speed up the network.'*
   - This underscores a growing concern regarding optimal **throughput** and **latency** in current networking frameworks.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Exploring Alternatives to pip for axolotl**: A member found **dependency management** in **axolotl** frustrating and suggested using non-pip packagers like **uv** for installing and updating.
   - They showed eagerness to contribute to ongoing efforts aimed at enhancing the **axolotl** experience.
- **Community Engagement in axolotl Development**: The same member expressed their willingness to improve the **axolotl** library by investigating diverse **packaging options**.
   - Their goal is to prompt other developers to get involved and address shared frustrations with **dependency management**.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1291475638653550683)** (254 messages🔥🔥): 

> - `torchao library by PyTorch`
> - `OpenAI's Canvas tool`
> - `Meta's Movie Gen models`
> - `Cultural biases in AI training`
> - `Nous Forge Framework` 


- **torchao: A Leap in Model Optimization**: The [torchao library](https://huggingface.co/posts/singhsidhukuldeep/639926000427051) released by PyTorch introduces advanced techniques like quantization and low-bit datatypes, optimizing models for performance and memory efficiency.
   - Features include automatic quantization and integration with existing tools, heralded as a significant step forward in the PyTorch ecosystem.
- **OpenAI Canvas Tool Receives Praise**: Users are excited about OpenAI's [Canvas tool](https://openai.com/index/introducing-canvas) as it combines features from other platforms, streamlining coding and reducing unnecessary scrolling.
   - The editing capabilities of Canvas have been highlighted as a significant improvement over previous iterations in tools like Claude.
- **Meta's Impressive Movie Gen Models**: Meta recently unveiled its [Movie Gen models](https://ai.meta.com/static-resource/movie-gen-research-paper), capable of generating high-quality images, videos, and audio from text prompts.
   - The models incorporate advanced features like precise video editing and personalized video generation, showcasing the potential for significant creative applications.
- **Cultural Biases and AI Training**: Discussion on how training LLMs lacks inherent human biases makes them reliant on large amounts of training data to understand concepts like love and morality.
   - The conversation explores the complexities of human emotions and how they could be learned or simulated by AI without being inherently 'real'.
- **Nous Forge: The AI Orchestration Framework**: Nous Forge is described as a platform for orchestrating AI agents, akin to 'Kubernetes for LLMs', enhancing the management of AI interactions and resources.
   - However, the name may clash with other existing frameworks in the AI community, raising questions about branding and functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/posts/singhsidhukuldeep/639926000427051">@singhsidhukuldeep on Hugging Face: &quot;Good folks at @PyTorch have just released torchao, a game-changing library for…&quot;</a>: no description found</li><li><a href="https://x.com/aiatmeta/status/1842188252541043075?s=46">Tweet from AI at Meta (@AIatMeta)</a>: 🎥 Today we’re premiering Meta Movie Gen: the most advanced media foundation models to-date.  Developed by AI research teams at Meta, Movie Gen delivers state-of-the-art results across a range of capa...</li><li><a href="https://x.com/_tim_brooks/status/1841982327431561528">Tweet from Tim Brooks (@_tim_brooks)</a>: I will be joining @GoogleDeepMind to work on video generation and world simulators! Can&#39;t wait to collaborate with such a talented team.  I had an amazing two years at OpenAI making Sora. Thank yo...</li><li><a href="https://x.com/m_wulfmeier/status/1842201976597074290?t=bVksmRCFScV1q6Vc4kDwgw&s=19">Tweet from Markus Wulfmeier (@m_wulfmeier)</a>: Looks like the new generation of students is better prepared for the age of Gemini/ChatGPT based review...</li><li><a href="https://pytorch.org/blog/quantization-aware-training/">Quantization-Aware Training for Large Language Models with PyTorch</a>: In this blog, we present an end-to-end Quantization-Aware Training (QAT) flow for large language models in PyTorch. We demonstrate how QAT in PyTorch can recover up to 96% of the accuracy degradation ...</li><li><a href="https://arxiv.org/abs/2409.13079">Embedding Geometries of Contrastive Language-Image Pre-Training</a>: Since the publication of CLIP, the approach of using InfoNCE loss for contrastive pre-training has become widely popular for bridging two or more modalities. Despite its wide adoption, CLIP&#39;s orig...</li><li><a href="https://x.com/lauriewired/status/1841875972691525673?s=46">Tweet from LaurieWired (@lauriewired)</a>: Your phone can&#39;t run a local 70B model (yet).    But it might while you sleep.  A brand-new paper (arXiv:2410.00531) squeezed Llama-3.1-70B into just 11.3GB of memory at *full precision*!  Traditi...</li><li><a href="https://x.com/studiomilitary/status/1841980965771506141?s=46">Tweet from John Galt (@StudioMilitary)</a>: 36 new wallpapers from me uploaded to the doors app  Quoting kenneth (@kennethnym)   NEW DOORS WALLPAPER DROP FROM ⁦@NousResearch⁩</li><li><a href="https://x.com/slow_developer/status/1842270727153623414?t=HR1olb-kaLei_1EZRIncug&s=19">Tweet from Haider. (@slow_developer)</a>: 🚨 BREAKING   Grok 3 will be open source.  Elon Musk has just announced that xAI will open source its models.</li><li><a href="https://pastebin.com/P0wQwvv9">o1preview - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://github.com/pytorch/ao">GitHub - pytorch/ao: PyTorch native quantization and sparsity for training and inference</a>: PyTorch native quantization and sparsity for training and inference - pytorch/ao</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: Contribute to lllyasviel/stable-diffusion-webui-forge development by creating an account on GitHub.</li><li><a href="https://github.com/pytorch/ao/pull/930">BitNet b1.58 training by gau-nernst · Pull Request #930 · pytorch/ao</a>: This PR adds training code for BitNet b1.58 (ternary weights - 1.58 bit. The first version of BitNet is binary weights). This is implemented as tensor subclass and integrate nicely with the quantiz...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/)** (1 messages): 

lukfbi: Guys, please help me, what is the best temperature for RPG and RP on the Hermes 70b?
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1291484044508532798)** (2 messages): 

> - `VinePPO algorithm`
> - `Pluralistic alignment`
> - `Model steerability benchmarks` 


- **VinePPO tackles reasoning-heavy tasks**: The paper introduces **VinePPO**, a new approach addressing the credit assignment issues with **Proximal Policy Optimization (PPO)** in LLMs due to significant shortcomings in reasoning-heavy tasks.
   - It reveals that value networks in PPO often result in **high-variance updates** and barely outperform a random baseline when evaluating alternative steps.
- **Workshop on Pluralistic Alignment at NeurIPS**: A member expressed enthusiasm for an upcoming workshop on **pluralistic alignment** at NeurIPS, highlighting its relevance to current AI discussions.
   - They sought insights about **benchmarks for model steerability** at inference time, specifically regarding how models align with seeded personas.
- **Need for tradeoff-steerable benchmarks**: The discussion references a paper proposing the need for **trade-off steerable benchmarks** that enable models to manage multiple objectives at inference time.
   - One contributor noted the paper's strong conceptual framing but pointed out the absence of specific implementations for these benchmarks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01679">VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment</a>: Large language models (LLMs) are increasingly applied to complex reasoning tasks that require executing several complex steps before receiving any reward. Properly assigning credit to these steps is e...</li><li><a href="https://x.com/ma_tay_/status/1755605755607359760">Tweet from Taylor Sorensen (@ma_tay_)</a>: We define and encourage pluralistic multi-objective benchmarks, and trade-off steerable benchmarks which encourage models to steer ↔️ to trade-off objectives at inference time, and…
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1291484044508532798)** (2 messages): 

> - `VinePPO`
> - `Pluralistic Alignment`
> - `Model Steerability Evaluation` 


- **VinePPO tackles credit assignment in LLMs**: The paper evaluates Proximal Policy Optimization (PPO) in enhancing credit assignment for large language models (LLMs) and introduces **VinePPO** to improve this aspect, as current value networks often fail in complex reasoning tasks.
   - The results show that value networks *barely outperform a random baseline*, highlighting the need for alternative strategies in reasoning-heavy tasks.
- **Excitement for Pluralistic Alignment Workshop**: A member expressed enthusiasm for an upcoming workshop at NeurIPS focused on **pluralistic alignment** and its implications for model behavior.
   - They sought insights on existing benchmarks for model steerability geared towards aligning with specific personas at inference time.
- **Demand for Tradeoff-Steerable Benchmarks**: The discussion centered around the need for **trade-off steerable benchmarks** to assess models' abilities in managing multiple objectives during inference.
   - The paper provides a solid conceptual framework but lacks specific implementations for these benchmarks, which are crucial for evaluating model steerability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01679">VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment</a>: Large language models (LLMs) are increasingly applied to complex reasoning tasks that require executing several complex steps before receiving any reward. Properly assigning credit to these steps is e...</li><li><a href="https://x.com/ma_tay_/status/1755605755607359760">Tweet from Taylor Sorensen (@ma_tay_)</a>: We define and encourage pluralistic multi-objective benchmarks, and trade-off steerable benchmarks which encourage models to steer ↔️ to trade-off objectives at inference time, and…
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1291479413170503754)** (2 messages): 

> - `OpenAI's model outputs`
> - `Open-source reasoning models` 


- **OpenAI's outputs discourage open-source development**: A member remarked that it is **desirable for OpenAI** to keep its outputs from being distilled into open-source reasoning models.
   - This limitation potentially hinders broader community access and innovation in AI model development.
- **Concerns about accessibility of reasoning tools**: Another point raised focused on how **restrictions** on outputs are kept to prevent individuals from creating their own reasoning models.
   - This perspective reflects a desire for more **open access** to AI technologies for developers.


  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1291475960188895324)** (140 messages🔥🔥): 

> - `Telemetry in Aider`
> - `OpenRouter Free Models Limitations`
> - `Benchmarking Aider with Various Models`
> - `Transition of Aider Repo Ownership`
> - `User Experiences with Aider Performance` 


- **Discussion on Aider's Telemetry Features**: A member expressed the need for telemetry in Aider, emphasizing the importance of opt-in features that provide insight while ensuring user privacy.
   - Another suggested including system call tracing to diagnose performance issues and mentioned the need for transparency about the data collected.
- **OpenRouter Free Models and Usage Limits**: Users discussed the limitations of OpenRouter's free models, noting a strict account-wide limit of 200 messages per day across all free models.
   - The inability to access a paid version for certain models raised questions about flexibility in usage for users.
- **Benchmarking Various Models with Aider**: Participants shared results from benchmarking different models, indicating mixed performance and discussing error rates during processing.
   - Aider’s capability to handle various editing scenarios was highlighted, along with user experiences regarding token limits and error messages.
- **Transition of Aider Repository Ownership**: It was announced that the main Aider repository on GitHub has moved from a personal account to a dedicated Aider organization page for better organization.
   - Links in documentation and code will be updated to reflect this change, which aims to clarify the project's identity.
- **User Experiences and Performance of Aider**: Several users reported varied performance issues with Aider, especially when working with large files, and discussed configurations to avoid errors.
   - One user noted that using the `--no-pretty` flag significantly improved processing speeds, citing concerns about default settings leading to unexpected API errors.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/install/install.html">Installing aider</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/install.html">Installation</a>: How to install and get started pair programming with aider.</li><li><a href="https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks">Creating and highlighting code blocks - GitHub Docs</a>: no description found</li><li><a href="https://bolt.new/">bolt.new</a>: no description found</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/benchmark/README.md">aider/benchmark/README.md at main · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider">GitHub - Aider-AI/aider: aider is AI pair programming in your terminal</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider/pull/1907">fix: Ensure consistent language in coder prompt descriptions by fry69 · Pull Request #1907 · Aider-AI/aider</a>: fix #1850 Thanks to @businistry for reporting and @jorgecolonconsulting for digging into this and providing the fix!
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1291524121326518303)** (69 messages🔥🔥): 

> - `Using Aider with Ollama`
> - `File addition in Aider`
> - `Aider performance and models`
> - `Repo map functionality`
> - `Aider modes for querying` 


- **Using Aider with Ollama's Local Model**: A user reported slow response times while using Aider with the local **Ollama 8B model**, and questioned if a paid API key from OpenAI or Anthropic would improve speed.
   - It was noted that local models may struggle with editing tasks, and Aider generally performs better with models known for their editing capabilities.
- **File Addition Behavior in Aider**: A user tested the new **/read-only** command in Aider and found it now only completes by folder rather than file name, which added complexity to accessing specific files.
   - Another user confirmed that the **/read-only** command should still add all files from a folder if used correctly.
- **Aider Model Performance and Speed**: Users discussed the limitations of using smaller local models like **Ollama 8B**, which hinder Aider's ability to respond quickly and accurately during code editing.
   - Alternatives such as **Cursor Composer AI** were mentioned as faster options, prompting questions about whether Aider's speed could improve with paid API keys.
- **Repo Map Functionality and Disabled Status**: A user discovered that the **repo map** needs a Git repository to function correctly, and was previously confused about its disabled status with certain models.
   - After initializing a Git repo, the user successfully enabled the repo map and received useful context during queries.
- **Utilizing Aider Modes for Efficient Querying**: The discussion highlighted the use of different modes in Aider, particularly using **/ask** and **/architect** for effectively querying the codebase.
   - Users noted that these modes can guide Aider to ask for relevant files, reducing token usage and improving results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: Using the chat, ask and help chat modes.</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: Frequently asked questions about aider.</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1291745189232377916)** (2 messages): 

> - `Aider mentions`
> - `Hybrid search with SQLite`
> - `Reciprocal Rank Fusion` 


- **Aider in HN Discussions**: An interesting thread on [Hacker News](https://news.ycombinator.com/item?id=41732634) features numerous mentions of **Aider**, sparking engaging conversations.
   - *Multiple participants shared insights related to Aider's functionalities and role in recent developments.*
- **Hybrid Search Strategies in SQLite**: **Alex's work** on the [sqlite-vec](https://github.com/asg017/sqlite-vec) extension introduces fast vector lookups, merging **vector similarity** and **traditional full-text search**.
   - A detailed exploration can be found in his [blog post](https://simonwillison.net/2024/Oct/4/hybrid-full-text-search-and-vector-search-with-sqlite/), which outlines the potential of hybrid search methods.
- **Reciprocal Rank Fusion Approach**: The most promising method under investigation is **Reciprocal Rank Fusion** which combines top-ranked items from both the vector and full-text search results.
   - Alex provides an SQL query that exemplifies the integration of **sqlite-vec** KNN vector search with FTS5 search results.



**Link mentioned**: <a href="https://simonwillison.net/2024/Oct/4/hybrid-full-text-search-and-vector-search-with-sqlite/">Hybrid full-text search and vector search with SQLite</a>: As part of Alex’s work on his [sqlite-vec](https://github.com/asg017/sqlite-vec) SQLite extension - adding fast vector lookups to SQLite - he’s been investigating hybrid search, where search results f...

  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1291486205766008903)** (1 messages): 

> - `Salamandra on-device demo`
> - `OpenAI models update`
> - `Nemo-Mistral-Minitron improvements`
> - `Realtime Whisper Turbo`
> - `MusicGen iOS app progress` 


- **Salamandra on-device demo shines**: [Salamandra](https://huggingface.co/spaces/Tonic/salamandra-on-device) demo showcased impressive capabilities by a verified user, highlighting its features in an engaging format.
   - *The excitement around Salamandra's spotlight in the community* reflects the growing interest in on-device AI applications.
- **OpenAI's o1 models hit the scene**: Two new **OpenAI** models, **o1-preview** and **o1-mini**, were integrated into the [open source chatbot](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.3.0), enhancing its functionality.
   - Members celebrated these additions, calling it a significant step towards more robust chatbot experiences.
- **Nemo-Mistral-Minitron gets a boost**: [Improvements on the Nemo-Mistral-Minitron](https://huggingface.co/spaces/Tonic/Nemo-Mistral-Minitron) demo have been rolled out by a verified user, enhancing its performance and usability.
   - *Upgrade discussions* indicated a trend towards optimizing AI models for better interaction and results.
- **Realtime Whisper Turbo is live**: A new [Realtime Whisper Turbo](https://huggingface.co/spaces/KingNish/Realtime-whisper-large-v3-turbo) project using Gradio 5 beta has been introduced, promising real-time transcription performance.
   - Community feedback has been positive, emphasizing its potential use in various applications.
- **MusicGen iOS app shows progress**: Progress on the iOS app for **MusicGen** highlights its features including a noise cancel for input audio and a 'tame the gary' toggle.
   - *One member remarked* that it focuses particularly on drums and attempts to better incorporate input for refined output.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://iatalk.ing/mapa-conceitos-ia/)">O que é OpenAI, Redes Neurais, Arquitetura, LLM e outros conceitos da IA? - IA Talking 🤖</a>: Quando eu comecei a estudar sobre IA me deparei com uma enxurrada de novos conceitos: OpenAI, LLM, ChatGPT, parâmetros, modelo, llama, gpt, hugging face, modelo, rag, embedding, gguf, ahhhhh&#8230; É ...</li><li><a href="https://x.com/thepatch_kev/status/1840536425776763020)">Tweet from thecollabagepatch (@thepatch_kev)</a>: day 4  ios app for musicgen continuations  landing screen, noise cancel for input audio and a &#39;tame the gary&#39; toggle that sort of works  focuses it on drums and tries harder to incorporate inp...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1291476058779947048)** (151 messages🔥🔥): 

> - `Meta Movie Gen`
> - `Hugging Face Chat support`
> - `Gradio Chatbot UI`
> - `Model usage`
> - `InstantMesh` 


- **Meta Movie Gen premieres advanced models**: Meta introduced [Movie Gen](https://go.fb.me/kx1nqm), a 30B parameter transformer model capable of generating high-quality images and videos from text prompts, along with a 13B audio model for syncing high-fidelity audio to video.
   - The release included detailed capabilities such as precise video editing and personalized videos, raising questions about accessibility and usefulness.
- **Hugging Face Chat supports transformers**: Users inquired if Hugging Face Chat supports transformer models, specifically mentioning models like BERT for question and answer tasks.
   - Models in Huggingchat leverage the transformer architecture, with a focus on tasks like question answering, as detailed in the [Hugging Face tasks section](https://huggingface.co/tasks/question-answering).
- **Gradio Chatbot UI inquiry**: A user sought advice on how to programmatically trigger the submit button in the Gradio Chatbot UI without manual clicking.
   - It was suggested to manually call the related function but the specific function needed remained unclear to the user.
- **Discussion on model execution environments**: Users discussed the execution of diffusion pipelines, clarifying that the generation process runs on the machine executing the commands rather than Hugging Face servers.
   - Questions arose about whether using the Diffuser API provides a distinct advantage over running models directly with Python.
- **InstantMesh integration inquiries**: A user posed a question about using the Diffuser API to run InstantMesh and how it compares to local execution methods.
   - This highlights the flexibility offered by using APIs versus direct local execution, particularly in handling model outputs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/posts/singhsidhukuldeep/639926000427051">@singhsidhukuldeep on Hugging Face: &quot;Good folks at @PyTorch have just released torchao, a game-changing library for…&quot;</a>: no description found</li><li><a href="https://x.com/AIatMeta/status/1842188252541043075">Tweet from AI at Meta (@AIatMeta)</a>: 🎥 Today we’re premiering Meta Movie Gen: the most advanced media foundation models to-date.  Developed by AI research teams at Meta, Movie Gen delivers state-of-the-art results across a range of capa...</li><li><a href="https://huggingface.co/spaces/allenai/reward-bench">Reward Bench Leaderboard - a Hugging Face Space by allenai</a>: no description found</li><li><a href="https://huggingface.co/docs/hub/spaces-gpus#hardware-specs">Using GPU Spaces</a>: no description found</li><li><a href="https://huggingface.co/docs/autotrain/tasks/llm_finetuning">LLM Finetuning</a>: no description found</li><li><a href="https://www.tiktok.com/t/ZP8RtX7x1/">TikTok - Make Your Day</a>: no description found</li><li><a href="https://github.com/NVIDIAGameWorks/toolkit-remix">GitHub - NVIDIAGameWorks/toolkit-remix: RTX Remix Toolkit</a>: RTX Remix Toolkit. Contribute to NVIDIAGameWorks/toolkit-remix development by creating an account on GitHub.</li><li><a href="https://huggingface.co/tasks/question-answering">What is Question Answering? - Hugging Face</a>: no description found</li><li><a href="https://github.com/TencentARC/InstantMesh">GitHub - TencentARC/InstantMesh: InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models</a>: InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models - TencentARC/InstantMesh</li><li><a href="https://huggingface.co/spaces">Spaces - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/posts/TuringsSolutions/527665072738819">@TuringsSolutions on Hugging Face: &quot;Hyperdimensional Computing + Neural Network, tell your friends. To my…&quot;</a>: no description found</li><li><a href="https://github.com/RichardAragon/HyperDimensionalComputingNeuralNetwork">GitHub - RichardAragon/HyperDimensionalComputingNeuralNetwork</a>: Contribute to RichardAragon/HyperDimensionalComputingNeuralNetwork development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1291480950580056075)** (5 messages): 

> - `Sanity 70B FP8`
> - `CUDA`
> - `μP`
> - `HuggingFace model upload`
> - `Outdated tutorials` 


- **Learning about Sanity 70B FP8 and CUDA**: A member shared that they learned about **Sanity 70B FP8**, **CUDA**, and **μP** in the last three days, indicating engagement with advanced topics.
   - *Learning these technologies is crucial for performance optimization and effective model deployment.*
- **Struggles with HuggingFace model uploads**: A member is attempting to learn how to properly upload a model to the **HuggingFace console**, but the tutorial they referenced is outdated.
   - *They noted a discrepancy with model file types, finding that other models utilize **.json** files in addition to **model.pkl**.*
- **Seeking updated resources**: The same member expressed a need for more current tutorials and is searching YouTube for examples to clarify the uploading process.
   - *Community members engaged, with one inquiring if the information was documented in the official resources.*


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1291762637214060684)** (6 messages): 

> - `New AI Model from Nvidia`
> - `Music Composer on HuggingFace`
> - `Text to Singing Model` 


- **Nvidia launches a game-changing AI model**: Nvidia has released a new AI model that is described as **open, massive**, and ready to **rival GPT-4**, according to [VentureBeat](https://venturebeat.com/ai/nvidia-just-dropped-a-bombshell-its-new-ai-model-is-open-massive-and-ready-to-rival-gpt-4/). This announcement has stirred excitement within the AI community.
   - Many are curious about how this model will compete against existing players like GPT-4 and what unique capabilities it brings to the table.
- **Gradio-based Music Composer on HuggingFace**: A member shared a link to a project showcasing a complete **music composer** built on HuggingFace Spaces using **Gradio** at [this link](https://huggingface.co/spaces/skytnt/midi-composer). This reveals the innovative applications being developed within the HuggingFace ecosystem.
   - The project has garnered attention for its creativity and functionality, showcasing how AI can assist in music composition.
- **Seeking Text to Singing Capabilities**: A member expressed their ongoing search for a **text to singing** model or methodology to utilize singing effectively outside of traditional spaces. This highlights the demand for more versatile AI applications in music.
   - The interest in developing such capabilities suggests a growing trend towards integrating AI in diverse musical formats.



**Link mentioned**: <a href="https://huggingface.co/spaces/skytnt/midi-composer">Midi Music Generator - a Hugging Face Space by skytnt</a>: no description found

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1291482310587191339)** (10 messages🔥): 

> - `Salamandra-2B-Instruct Release`
> - `Fastai Convolution Explanation`
> - `Nvidia Model Updates`
> - `New Labeling Tool for LLMs`
> - `Llava Video Understanding Model` 


- **Salamandra-2B-Instruct is here!**: The **Salamandra-2B-Instruct** model has been released, bringing exciting new capabilities to users on the Hugging Face platform.
   - Check out the details on its [demo page](https://huggingface.co/spaces/Tonic/Salamandra-2B-Instruct) for more insights.
- **Fastai Course Insights on CNNs**: A user shared their exploration of unrolling convolutions while working on **Lesson 15** of the Fastai course, explaining that **CNNs** are like **NNs** without weight for each input.
   - They discussed their findings in detail on the [Fastai Forum](https://forums.fast.ai/t/rearranging-convolutions-as-matrix-products/114703?u=forbo7).
- **Nvidia Team's Model Innovations**: There are ongoing releases from the Nvidia team, notably a new model that enhances usability with **nvidialign** for model size reduction through ablations.
   - This series is closely tracked, showcasing significant advancements in model performance and capabilities.
- **Innovative Tool for Data Labeling**: A new collaborative tool has been developed for **labeling data** and fine-tuning **LLMs** that combines AI and human oversight to enhance accuracy and efficiency.
   - Interested testers can view the [demo video](https://www.youtube.com/watch?v=YVwby-49Y-I&feature=youtu.be) and provide feedback on this promising tool.
- **Exciting New Llava Video Understanding Model**: A fresh demo of the **Llava Video Understanding Model** has been released, showcasing its capabilities in video comprehension.
   - Curious users can view the demo [here](https://huggingface.co/spaces/Tonic/Llava-Video) for more information on its functionalities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Tonic/Salamandra-2B-Instruct">Salamandra 2B Instruct - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://huggingface.co/spaces/Tonic/Llava-Video">Llava Video - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://forums.fast.ai/t/rearranging-convolutions-as-matrix-products/114703?u=forbo7">Rearranging Convolutions as Matrix Products</a>: I’m currently working through lesson 15 of the fastai course, and have finished the portion about convolutions. Here, I explain how the convoluton operation can be rearranged, or unrolled, as a matrix...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1291837752127783055)** (4 messages): 

> - `AI Sentience Prediction`
> - `Original Research Sharing`
> - `Weekly Reading Group` 


- **Exploring AI Sentience Prediction**: An article titled 'The Sentience Prediction Equation' discusses when AI may achieve sentience and the implications thereof, questioning if AI will ever ponder its purpose.
   - The article humorously notes potential questions an AI might ask, like *'Why do humans insist on putting pineapple on pizza?'* and introduces the Sentience Prediction Equation as an estimation tool.
- **Inquiry on Sharing Original Research**: A member inquired if there's a venue for sharing original research within the community.
   - Another member suggested presenting it in the Discord, tagging individuals who might be interested in the topic.
- **Weekly Reading Group for Discussions**: It was mentioned that a weekly reading group exists for sharing and discussing various topics.
   - This provides a platform for members to present their research findings and engage in academic discourse.



**Link mentioned**: <a href="https://medium.com/@ryanfoster_37838/the-sentience-prediction-equation-when-will-ai-achieve-sentience-and-should-we-be-worried-bf5fa0042408">The Sentience Prediction Equation: When Will AI Achieve Sentience? (And Should We Be Worried?)</a>: You’ve heard the buzz: AI is getting smarter. It’s writing novels, making memes, diagnosing diseases, and even, well, generating this very…

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1291583441393029184)** (1 messages): 

> - `Model Training Explained`
> - `Conceptual Learning vs Instructional Learning`
> - `Catastrophic Forgetting` 


- **Model Training: Child-Like vs Student-Like Learning**: An analogy was drawn comparing a child learning from their environment to a student studying math from a book. The child represents a **pre-trained model**, while fine-tuning mirrors the **instruction-based learning** necessary for advancement.
   - If a toddler receives a math book, they won't understand its purpose, akin to a model lacking foundational knowledge impacting its performance.
- **Challenges of Learning Rare Concepts**: The conversation shifted to understanding new, rare topics, like hypothetical **aliens living in dark matter**, that are not widely observable. This implies that students may **struggle** when faced with subjects lacking prior associations from their learning experience.
   - Without context, eager learners could face difficulties, leading to what was labeled as **catastrophic forgetting** of previously learned information.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1291479150871314566)** (8 messages🔥): 

> - `Spacy's online training module`
> - `Fine-tuning with custom datasets`
> - `Using SFTTrainer for language models`
> - `ONNX model conversion issues`
> - `Transformers.js integration` 


- **Spacy’s Online Training Wins Hearts**: A member praised **Spacy’s** structured online training module, suggesting it is an excellent starting point for beginners to deep dive into NLP concepts.
   - They highlighted that it provides a structured, free course that effectively targets the beginner stage.
- **Fine-tuning Models with Custom Data**: A member stated that while you can fine-tune models with **public datasets**, adapting your custom set depends significantly on the use case.
   - They recommended ensuring that the custom data resembles public datasets if substantial modifications or cleaning is not performed on raw text.
- **SFTTrainer Class for Language Model Datasets**: A user identified that the datasets discussed are of the **language model** type and suggested using the **SFTTrainer** class for fine-tuning.
   - They requested confirmation on whether this was correct, hoping to clarify the appropriate trainer usage.
- **Issues with ONNX Conversion and Transformers.js**: A member encountered an issue when loading a model exported in **ONNX** format using **transformers.js**, which fails to load `onnx/decoder_model_merged_quantized.onnx`.
   - They sought assistance, prompting another member to suggest verifying the model's saved location and the correctness of the specified pathways.
- **Troubleshooting ONNX Model Loading**: In response to the ONNX loading issue, another member advised checking default arguments in the `from_pretrained` function to resolve loading problems.
   - They emphasized the importance of ensuring that the model's physical location matches the expected paths.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/openai/gsm8k">openai/gsm8k · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/TIGER-Lab/MathInstruct?row=0">TIGER-Lab/MathInstruct · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/trl/main/en/alignprop_trainer">Aligning Text-to-Image Diffusion Models with Reward Backpropagation</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1291568492830785546)** (3 messages): 

> - `Flux model restrictions`
> - `Hacktoberfest contributions` 


- **Flux isn't Truly Open Source**: Contrary to popular belief, **Flux** is **not open source** in the genuine sense as its model specifications and training data remain private, shared only weights.
   - *This highlights the disconnect between perception and reality in open source practices.*
- **Finding Repositories for Hacktoberfest in ML**: A member inquired about how to discover repositories for contributions during **Hacktoberfest** specifically in the ML domain.
   - In response, another member suggested using **GitHub's search function**, mentioning that their specific channel, **Diffusers**, hasn't yet opened Hacktoberfest issues.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1291477746245177475)** (134 messages🔥🔥): 

> - `Canvas Model`
> - `OpenAI Tools`
> - `Advanced Voice Mode`
> - `Discord Bots`
> - `AI Programming` 


- **Canvas Model Discussion Heats Up**: Many users expressed excitement about the new **Canvas model**, with discussions about its potential functionality, including manual invocation and integration with GPT-4o, as detailed in [this link](https://openai.com/index/introducing-canvas/).
   - Members noted that **Canvas** currently does not support certain features, creating some frustration, but acknowledged its promise in enhancing UX for programming.
- **Advanced Voice Mode Gets Attention**: The **Advanced Voice Mode** has sparked conversation regarding its integration with the Canvas tool, suggesting a potential future where both work seamlessly together.
   - Users have shared their hopes that features like real-time API integration could enhance coding efficiency, with some even sharing setup guides on GitHub.
- **Discord Bots and Community Help**: A user reached out for help regarding **Discord bots**, showing that community support remains strong for newcomers struggling with coding.
   - This prompted various members to offer assistance and share their experiences with creating and troubleshooting bots.
- **AI's Role in Programming Languages**: The discussion highlighted the effectiveness of **OpenAI's models** in programming, particularly **TypeScript** and **Python**, suggesting a preference for strict type languages when using AI for coding tasks.
   - Some members noted the challenges and frustrations with JavaScript, while praising alternatives like **Kotlin** for their usability.
- **AI and Avatars in Communication**: There was a debate about the usefulness of avatars in AI tools during voice calls, with differing opinions on their necessity and impact on user experience.
   - The potential for avatars to be used in professional branding was noted, suggesting an evolving landscape for digital interaction tools in professional settings.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/2oQ5VkW-DZ8?si=DwnRHuROyerEZ0zR">How to use a Large Action Model (AI) to schedule any task</a>: Learn how to take your actions to the next level with Nelima&#39;s brand-new scheduling feature! In this video, I’ll walk you through how to use Nelima’s powerfu...</li><li><a href="https://github.com/jjmlovesgit/ChatGPT-Advanced-Voice-Mode">GitHub - jjmlovesgit/ChatGPT-Advanced-Voice-Mode: ChatGPT Advanced Voice Mode Gets an Avatar!</a>: ChatGPT Advanced Voice Mode Gets an Avatar! Contribute to jjmlovesgit/ChatGPT-Advanced-Voice-Mode development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1291477208401186826)** (11 messages🔥): 

> - `Custom GPTs with Google API`
> - `Custom GPTs Model Queries`
> - `Canvas Issues`
> - `ChatGPT Counting and Math Concerns` 


- **Custom GPTs Finicky Integration**: A member shared past attempts to integrate **Google API/OAuth** with Custom GPTs, noting that it was quite finicky during the initial release.
   - They have not checked back since then to see if the stability of this integration has improved.
- **Canvas Lacks Essential Features**: Multiple members expressed frustration that the new **canvas** lacks a **continue button**, making it cumbersome to use.
   - Additionally, there are issues with editing large files and mismatches in document formatting that hinder functionality.
- **ChatGPT's Math Capabilities Under Scrutiny**: One member questioned whether **ChatGPT** has become worse at counting, suggesting the need for explicit instructions to perform math tasks.
   - Another clarified that LLMs like ChatGPT are text predictors and suggested using the **data analysis tool** or Python for accurate computations.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1291534615156363315)** (8 messages🔥): 

> - `Inconsistencies in ChatGPT evaluations`
> - `Embedding images in Newl Canvas`
> - `Efficient parsing of snippets to JSON`
> - `Model scoring techniques` 


- **Inconsistencies in ChatGPT evaluations**: A user shared frustrations with inconsistencies in ChatGPT evaluations when prompting it to score answers on a scale of 10 at **temperature 0.7**.
   - Members suggested that since GPT is stochastic, using a tighter scale, such as **0-5**, and providing a **grading rubric** could enhance consistency.
- **Embedding images in Newl Canvas**: A user noted that for Newl Canvas mains, images can be embedded directly using the syntax ```![Image Description](Image Link)```.
   - This feature could streamline the process of including visuals in canvas presentations.
- **Efficient parsing of snippets to JSON**: A user is parsing 10,000 snippets of text into JSON format using Python and GPT-4o, questioning the efficiency of resubmitting **system_prompt** and **response_format** with every snippet.
   - Suggestions were made on how to reduce costs by submitting only the next snippet without needing to resubmit the structures each time.
- **Model scoring techniques**: To address evaluation issues, a user suggested implementing a **Chain-of-Thought** approach for reasoning through the evaluations before providing a score.
   - Additionally, it was recommended to evaluate one answer at a time and provide diverse examples of high-quality evaluations.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1291534615156363315)** (8 messages🔥): 

> - `ChatGPT evaluations consistency`
> - `Using images in Newl Canvas`
> - `JSON parsing with GPT-4o`
> - `Grading rubric for evaluations`
> - `Chain-of-Thought in evaluations` 


- **Inconsistencies in ChatGPT evaluations**: A user noted inconsistencies when asking ChatGPT (temperature @ 0.7) to evaluate answers on a scale of 10, receiving different marks on reruns.
   - Another user explained that GPT's stochastic nature leads to varied outputs, suggesting tightening the scoring scale to improve consistency.
- **Embedding images in Newl Canvas**: A member shared that Newl Canvas mains can embed images using the syntax ```![Image Description](Image Link)```.
   - This feature enhances the visualization capabilities within the canvas for users.
- **Efficient JSON parsing with GPT-4o**: A developer using GPT-4o for parsing 10,000 snippets into JSON queried if it's necessary to resend the system_prompt and response_format for each snippet.
   - Advice was provided for optimizing costs by potentially streamlining the process without resubmitting the common parameters.
- **Grading rubric and Chain-of-Thought suggestion**: To improve rating accuracy, a user recommended providing a grading rubric to clarify what each score entails.
   - They also suggested employing Chain-of-Thought reasoning to enhance evaluative clarity before arriving at a final score.
- **Best practices for evaluations**: For effective evaluation, suggestions included reducing temperature to 0 and evaluating one answer at a time with diverse high-quality examples.
   - These strategies aim to foster more reliable and diverse assessments from the model.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1291495318004695060)** (77 messages🔥🔥): 

> - `Unsloth AI Projects`
> - `Lora Configuration in PEFT`
> - `Fine-tuning Models`
> - `ZLUDA Project Update`
> - `Movie Gen AI Model` 


- **Unsloth AI Projects for Fine-tuning**: Members discussed using Unsloth AI for continual pretraining of LLMs, noting its efficiency in training at **2x faster** and with **50% less VRAM** compared to alternatives.
   - The continued pretraining notebook and text completion notebook were highlighted as essential tools for training models in different languages.
- **Lora Configuration Challenges**: A member inquired about making embedding layers trainable in Lora configurations, seeking clarity on having them included in target modules with the **modules_to_save** option.
   - Some confusion arose regarding the differences between notebooks for continued pretraining and text completion, with emphasis on learning rate scheduling.
- **Fine-tuning with Gradual Learning**: Discussions on fine-tuning methodology suggested starting with simpler datasets before gradually introducing more complex data for better model performance.
   - Members speculated that a gradual increase in dataset complexity might be beneficial, especially for unknown languages.
- **ZLUDA Project Announcement**: ZLUDA's development is being funded by a new commercial organization, promising improved functionality and long-term vision for the project.
   - However, concerns were raised about potential legal issues with NVIDIA, echoing past experiences where investor backing may falter due to intellectual property disputes.
- **Introduction of Movie Gen AI Model**: The Movie Gen AI Model was introduced as a new standard for high-definition video generation from simple text inputs, enabling advanced editing capabilities.
   - Members reacted positively, acknowledging the project's novelty and sharing excitement about its potential impacts on content creation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vosen.github.io/ZLUDA/blog/zludas-third-life/">ZLUDA - ZLUDA&#x27;s third life</a>: no description found</li><li><a href="https://huggingface.co/posts/singhsidhukuldeep/639926000427051">@singhsidhukuldeep on Hugging Face: &quot;Good folks at @PyTorch have just released torchao, a game-changing library for…&quot;</a>: no description found</li><li><a href="https://arxiv.org/abs/2410.00531">TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices</a>: Large model inference is shifting from cloud to edge due to concerns about the privacy of user interaction data. However, edge devices often struggle with limited computing power, memory, and bandwidt...</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://huggingface.co/posts/TuringsSolutions/527665072738819">@TuringsSolutions on Hugging Face: &quot;Hyperdimensional Computing + Neural Network, tell your friends. To my…&quot;</a>: no description found</li><li><a href="https://unsloth.ai/blog/contpretraining">Continued LLM Pretraining with Unsloth</a>: Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.</li><li><a href="https://stackoverflow.com/questions/70508960/how-to-free-gpu-memory-in-pytorch">How to free GPU memory in PyTorch</a>: I have a list of sentences I&#x27;m trying to calculate perplexity for, using several models using this code:&#xA;from transformers import AutoModelForMaskedLM, AutoTokenizer&#xA;import torch&#xA;impo...</li><li><a href="https://x.com/danielhanchen/status/1841921149804163247">Tweet from Daniel Han (@danielhanchen)</a>: My @PyTorch conference talk on Hacks to make LLM training faster is out!  1. Bit representation at limits. Need O(Mantissa^2) transistors. Bfloat16(M=7)=49 & float32(M=32)=529  2. Hardware - tensor co...</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes/multilingual/README.md">llama-recipes/recipes/multilingual/README.md at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081 · meta-llama/llama-recipes</a>: Scripts for fine-tuning Meta Llama with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization and Q&...</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003">GitHub - meta-llama/llama-recipes at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081</a>: Scripts for fine-tuning Meta Llama with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization and Q&...</li><li><a href="https://github.com/unslothai/unsloth#finetune-llama-32-mistral-phi-35--gemma-2-5x-faster-with-80-less-memory">GitHub - unslothai/unsloth: Finetune Llama 3.2, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.2, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/RichardAragon/HyperDimensionalComputingNeuralNetwork">GitHub - RichardAragon/HyperDimensionalComputingNeuralNetwork</a>: Contribute to RichardAragon/HyperDimensionalComputingNeuralNetwork development by creating an account on GitHub.</li><li><a href="https://ai.meta.com/research/movie-gen">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1291477922221264997)** (43 messages🔥): 

> - `Generational Identity`
> - `Gen Z Preferences`
> - `Lego vs. Modded Minecraft` 


- **Generational Identity Crisis**: Members jokingly debated their generational identities, with one claiming to feel shame about being Gen Z and another calling themselves a 'boomer' despite being only 24.
   - *It's just that those are the most noisy* resonates with concerns around generational stereotypes.
- **Gen Z prefers VSCode?**: There was a discussion around Gen Z's preference for **VSCode**, with some members humorously noting they used **VS Codium** to block telemetry.
   - One joked that using Legos in childhood now defines generational boundaries almost like horoscopes.
- **Lego's Cultural Significance**: A member expressed that the decline of Lego play among future generations would signal the *end of society*, highlighting its cultural importance.
   - Another suggested that **modded Minecraft** could serve as an acceptable alternative to Lego for younger generations.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1291506668575789147)** (31 messages🔥): 

> - `Local inference with llama-cpp`
> - `Multi-GPU support`
> - `Fine-tuning models with plain text`
> - `Preparing datasets for training`
> - `Running LLM on mobile with Flutter` 


- **Local inference script struggles**: A member shared difficulties with their local inference script for **gguf models** using **llama-cpp**, experiencing long processing times despite having a **GPU**.
   - Another member suggested using **llama-cli** for potentially better performance.
- **Multi-GPU support update**: A member inquired about updates on **multi-GPU** support and mentioned applying for it a week ago without any responses.
   - Another member noted that testing is ongoing, with access currently limited, but it should be more broadly available later this year.
- **Fine-tuning using plain text**: Discussion arose about fine-tuning **llama3.1** on plain text datasets, specifically using medical science books, with a warning that **structured data** is necessary for training.
   - A member advised using **augmenToolKit** to convert books into structured datasets, highlighting that 80% of the workload involves dataset preparation.
- **Running LLM in Flutter app**: A member expressed the need to run an **LLM** on a **PC** while receiving mobile inputs for a Flutter application.
   - Another member suggested using a **/chat/completion** based approach to achieve this integration.
- **Finding 16bit models**: A member requested information and resources about **16bit models**, looking for notebooks or related materials.
   - Another member provided a link to the **Unsloth documentation** which includes a list of notebooks available for reference.



**Link mentioned**: <a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: See the list below for all our notebooks:

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1291477205452460207)** (6 messages): 

> - `Nanoflow framework`
> - `Recurrent Neural Networks revival`
> - `SageAttention quantization`
> - `Code replacement suggestion` 


- **Nanoflow serves LLMs with high throughput**: [Nanoflow](https://github.com/efeslab/Nanoflow) is a high-performance serving framework optimized for **LLMs** focused on throughput, aiming to enhance processing speeds.
   - It seeks to address serving complexities often encountered in large models, offering notable improvements in efficiency.
- **RNNs make a comeback!**: A recent paper discusses the potential of minimal LSTMs and GRUs that can be trained **175x faster** by removing hidden state dependencies, thus avoiding backpropagation through time.
   - This revival of traditional RNNs in the context of modern architectures posits new avenues for scalable training methods.
- **SageAttention boosts quantization**: [SageAttention](https://github.com/thu-ml/SageAttention) introduces a quantization method for Attention, achieving **2.1x** and **2.7x** speedups compared to **FlashAttention2** and **xformers** respectively without sacrificing model metrics.
   - This method seamlessly integrates with the quantization process, combining advanced techniques for improved performance.
- **Exploring code replacement with SageAttention**: A suggestion was made that it might be feasible to replace a specific code line in the [Llama model](https://github.com/unslothai/unsloth/blob/ae9e264e33c69b53dd5d533a4c5a264af4141c28/unsloth/models/llama.py#L426) with `sageattn`, citing efficiency gains.
   - This reflects ongoing discussions about optimizing implementations using the latest available techniques.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.06147">State-Free Inference of State-Space Models: The Transfer Function Approach</a>: We approach designing a state-space model for deep learning applications through its dual representation, the transfer function, and uncover a highly efficient sequence parallel inference algorithm th...</li><li><a href="https://arxiv.org/abs/2410.01201">Were RNNs All We Needed?</a>: The scalability limitations of Transformers regarding sequence length have renewed interest in recurrent sequence models that are parallelizable during training. As a result, many novel recurrent arch...</li><li><a href="https://github.com/efeslab/Nanoflow">GitHub - efeslab/Nanoflow: A throughput-oriented high-performance serving framework for LLMs</a>: A throughput-oriented high-performance serving framework for LLMs - efeslab/Nanoflow</li><li><a href="https://github.com/thu-ml/SageAttention">GitHub - thu-ml/SageAttention: A quantization method for Attention that achieves speedups of 2.1x and 2.7x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models.</a>: A quantization method for Attention that achieves speedups of 2.1x and 2.7x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models. - thu-m...</li><li><a href="https://github.com/unslothai/unsloth/blob/ae9e264e33c69b53dd5d533a4c5a264af4141c28/unsloth/models/llama.py#L426">unsloth/unsloth/models/llama.py at ae9e264e33c69b53dd5d533a4c5a264af4141c28 · unslothai/unsloth</a>: Finetune Llama 3.2, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1291496054419886206)** (94 messages🔥🔥): 

> - `IREE adoption and compilation`
> - `RWKV and parallelization`
> - `Chain of Thought (CoT) output limitations`
> - `Gated Linear Attention and models expressible as RNNs`
> - `MATS Program and mentorship opportunities` 


- **Exploring IREE's Potential**: Members discussed whether large labs might adopt IREE for serving models at scale, with indications that many use custom inference runtimes.
   - Some members pointed out that adoption timelines for new technologies like IREE are often unpredictable.
- **RWKV's Layered Parallelization Strategy**: RWKV introduces a method for partial parallelization by structuring the network into smaller layers, allowing the next token's hidden state to be computed while waiting for others.
   - This design constraint aims to streamline computations while balancing the need for interdependencies in the model's outputs.
- **Chain of Thought and Computation Efficiency**: The discussion revealed skepticism about the efficiency of Chain of Thought (CoT) outputs, suggesting improvements could be made through denser representation methods.
   - Members highlighted that while CoT may be beneficial, relying heavily on it might not address underlying performance issues effectively.
- **Understanding Linear Attention Models**: Members emphasized the dual nature of certain models, such as linear attention and gated linear attention, which can be expressed as RNNs while enabling parallel computations across sequences.
   - Interest was shown in how Songlin Yang's research has uncovered more complex RNN classes capable of efficient parallelization.
- **MATS Program Mentorship Announcement**: A member shared a tweet announcing mentorship availability for the MATS Program Winter 2024-25, along with application details.
   - This includes a mentoring opportunity with Alignment Science Co-Lead at AnthropicAI, emphasizing the program's growth and engagement.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wiki.rwkv.com/advance/architecture.html#how-does-rwkv-differ-from-classic-rnn)">RWKV Architecture</a>: no description found</li><li><a href="https://arxiv.org/abs/2312.06635">Gated Linear Attention Transformers with Hardware-Efficient Training</a>: Transformers with linear attention allow for efficient parallel training but can simultaneously be formulated as an RNN with 2D (matrix-valued) hidden states, thus enjoying linear-time inference compl...</li><li><a href="https://arxiv.org/abs/2406.06484">Parallelizing Linear Transformers with the Delta Rule over Sequence Length</a>: Transformers with linear attention (i.e., linear transformers) and state-space models have recently been suggested as a viable linear-time alternative to transformers with softmax attention. However, ...</li><li><a href="https://x.com/MATSprogram/status/1842286650006892914">Tweet from ML Alignment & Theory Scholars (@MATSprogram)</a>: @janleike, Alignment Science Co-Lead @AnthropicAI, will now be mentoring for MATS Winter 2024-25! Applications close Oct 6, 11:59 pm PT. https://matsprogram.org/apply</li><li><a href="https://github.com/lucidrains/quartic-transformer">GitHub - lucidrains/quartic-transformer: Exploring an idea where one forgets about efficiency and carries out attention across each edge of the nodes (tokens)</a>: Exploring an idea where one forgets about efficiency and carries out attention across each edge of the nodes (tokens) - lucidrains/quartic-transformer
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1291484013566885939)** (49 messages🔥): 

> - `VinePPO Challenges`
> - `minLSTMs and minGRUs`
> - `Transfer Learning in Math`
> - `Softmax Function Limitations`
> - `Test Time Training (TTT)` 


- **VinePPO shows issues in LLM credit assignment**: The paper discusses how **value networks** struggle with credit assignment in complex reasoning tasks, leading to poor performance compared to random baselines.
   - This highlights the need for better models or methods to effectively utilize credit assignment techniques in Proximal Policy Optimization (PPO).
- **Revisiting LSTM and GRU for parallel training**: The exploration of **minLSTMs** and **minGRUs** reveals a method to train recurrent networks efficiently in parallel without backpropagating through time, achieving 175x faster training.
   - This study suggests that traditional RNN architectures can be simplified while still providing significant performance improvements.
- **Quantifying Transfer Learning in Mathematics**: A participant inquired about research quantifying the transfer effects when training models on mathematical reasoning tasks like **MATH** and **GSM8k**.
   - They expressed interest in understanding how predictable the performance boost across related tasks can be.
- **Limitations of the Softmax Function**: A paper discusses that the **softmax function** can struggle with sharp decisions as input grows, limiting its ability to approximate aggressive computations.
   - This limitation suggests the need for adaptive approaches in softmax implementations to enhance robustness in model predictions.
- **The Promise of Test Time Training (TTT)**: Participants highlighted the compelling nature of **Test Time Training (TTT)**, noting its potential for future theoretical advancements in machine learning.
   - There was recognition that TTT could introduce risks with nonlinear models, yet it was considered a promising area for exploration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01201">Were RNNs All We Needed?</a>: The scalability limitations of Transformers regarding sequence length have renewed interest in recurrent sequence models that are parallelizable during training. As a result, many novel recurrent arch...</li><li><a href="https://arxiv.org/abs/2410.01104">softmax is not enough (for sharp out-of-distribution)</a>: A key property of reasoning systems is the ability to make sharp decisions on their input data. For contemporary AI systems, a key carrier of sharp behaviour is the softmax function, with its capabili...</li><li><a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>: Self-attention performs well in long context but has quadratic complexity. Existing RNN layers have linear complexity, but their performance in long context is limited by the expressive power of their...</li><li><a href="https://arxiv.org/abs/2410.01679">VinePPO: Unlocking RL Potential For LLM Reasoning Through Refined Credit Assignment</a>: Large language models (LLMs) are increasingly applied to complex reasoning tasks that require executing several complex steps before receiving any reward. Properly assigning credit to these steps is e...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1291515512706174987)** (1 messages): 

> - `lm-evaluation-harness`
> - `GPT-NeoX improvements` 


- **lm-evaluation-harness needs contributors**: The **lm-evaluation-harness** is open for contributions on integrating new LLM evaluations and fixing bugs, with many detailed issues available to explore [here](https://github.com/EleutherAI/lm-evaluation-harness/issues).
   - The community encourages potential contributors to check the [GitHub repository](https://github.com/EleutherAI/lm-evaluation-harness) for more information.
- **GPT-NeoX seeks improvements**: The **GPT-NeoX** team is looking for help on enhancing their test suite and adding new tests, which can be found in the [tests directory](https://github.com/EleutherAI/gpt-neox/tree/main/tests).
   - Contributors can also help improve container setups and explore a variety of issues listed on the [issues page](https://github.com/EleutherAI/gpt-neox/issues).
- **Explore new features in GPT-NeoX**: The **GPT-NeoX** project presents a host of new distributed features for those interested in contributing, with details available through their [PRs](https://github.com/EleutherAI/gpt-neox/pulls).
   - Engagement in this space could lead to impactful enhancements for the library’s functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/729741769192767510/755950983669874798)">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues">Issues · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - Issues · EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/gpt-neox/tree/main/tests">gpt-neox/tests at main · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox</li><li><a href="https://github.com/EleutherAI/gpt-neox/issues">Issues · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - Issues · EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1291475109600825405)** (2 messages): 

> - `SambaNova AI on OpenRouter`
> - `Gemini 1.5 Flash-8B Release` 


- **SambaNova AI Hits OpenRouter with Fastest Throughput**: [SambaNova AI](https://x.com/SambaNovaAI/status/1841901026821210131) announced their endpoints for **Llama 3.1 and 3.2** are live on OpenRouter, boasting the fastest throughput measurements they've recorded.
   - They mentioned, *‘These are the fastest we’ve seen’*, highlighting that their throughput measurements are generally more conservative than others.
- **Gemini 1.5 Flash-8B Now Available**: The **Gemini 1.5 Flash-8B** model has been officially launched and can be accessed for use [here](https://openrouter.ai/models/google/gemini-flash-1.5-8b).
   - Additionally, the model's ID has been renamed for consistency, while the old ID will still function via an alias.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/SambaNovaAI/status/1841901026821210131">Tweet from SambaNova Systems (@SambaNovaAI)</a>: We’re up on @OpenRouter! They say it’s the fastest throughput measurements they’ve seen. 🚀🚀🚀  Thanks for the shoutout!  Quoting OpenRouter (@OpenRouterAI)   .@SambaNovaAI endpoints for Llama 3.1 an...</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5-8b">Gemini 1.5 Flash-8B - API, Providers, Stats</a>: Gemini 1.5 Flash-8B is optimized for speed and efficiency, offering enhanced performance in small prompt tasks like chat, transcription, and translation. Run Gemini 1.5 Flash-8B with API
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1291477094810779804)** (140 messages🔥🔥): 

> - `Gemini 1.5 Flash`
> - `o1 Mini performance`
> - `Anthropic's model development`
> - `Model alignment techniques`
> - `OpenRouter infrastructure updates` 


- **Gemini 1.5 Flash impresses with low costs**: The **Gemini 1.5 Flash-8B** model offers a competitive price of **$0.0375 per million tokens**, leading to discussions about its performance and pricing structure compared to other models.
   - Members speculate on the potential scaling and applicability of **Gemini's** more recent offerings.
- **o1 Mini showcases improved solving capability**: Users noted that **o1 Mini** has been solving complex tasks effectively, surprising those in the community who did not expect its performance to exceed that of other models.
   - One participant plans to use **o1 Mini** in a bot to facilitate image descriptions, highlighting its enhanced usability.
- **Anthropic's strategic advantage with funding**: Discussion reveals that **Anthropic's** success can be attributed to its team of ex-OpenAI engineers and backing from **Amazon**, allowing for rapid development of their **Claude** models.
   - There’s speculation on how they maintain competitive performance despite less financial backing compared to larger corporations.
- **Innovative alignment techniques debated**: Members discuss how models like Anthropic's handle alignment, mentioning its effectiveness in training without post-model filtering, in contrast to OpenAI's methods.
   - The conversation also touches on concepts of prompt injections and model moderation techniques.
- **OpenRouter infrastructure improvements**: User expressed anticipation for future expansions of **OpenRouter** to support a wider range of model functionalities, including image and audio processing.
   - Development lead confirmed ongoing upgrades to manage increased traffic and new model releases.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://developers.googleblog.com/en/gemini-15-flash-8b-is-now-generally-available-for-use/">Gemini 1.5 Flash-8B is now production ready</a>: no description found</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1fvk2wr/what_would_an_ai_with_anxiety_look_like/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://openrouter.ai/settings/privacy">Privacy | OpenRouter</a>: Manage your privacy settings</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://openrouter.ai/models/openai/gpt-4-vision-preview">GPT-4 Vision - API, Providers, Stats</a>: Ability to understand images, in addition to all other [GPT-4 Turbo capabilties](/models/openai/gpt-4-turbo). Training data: up to Apr 2023. Run GPT-4 Vision with API
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1291569294731378730)** (127 messages🔥🔥): 

> - `LM Studio Updates`
> - `Memory Leak Issues`
> - `Model Downloading and Integration`
> - `Chat Cache Location`
> - `AI Model Recommendations` 


- **LM Studio Support for Langflow**: Good news that LM Studio support is being integrated into Langflow, as noted in a recent pull request on GitHub.
   - This aims to enhance functionalities for users who wish to create LLM applications.
- **Memory Leak Concerns**: Users reported experiencing a memory leak with LM Studio version v0.3.2.6, leading to models producing gibberish output.
   - Advice was given to check if the same issue persists in version v0.3.3.
- **Downloading Models and Troubleshooting**: Users are encountering issues with downloading models from Hugging Face, specifically seeing errors when selecting models in LM Studio.
   - A workaround is suggested by sideloading models directly into the models directory of LM Studio.
- **Chat Cache Customization Queries**: Users inquired about the ability to change the location of the chat cache in LM Studio, which is currently not customizable.
   - The application now saves conversation data in JSON format, but configurations for chat cache location are not available yet.
- **AI Model Recommendations**: Discussions on which AI models are recommended for chatbot assistants highlighted Llama-3-8B as not satisfactory for some users.
   - Users were directed to various models available on the LM Studio platform, encouraging exploration of options that better fit their needs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:9222`">no title found</a>: no description found</li><li><a href="https://dontasktoask.com/">Don't ask to ask, just ask</a>: no description found</li><li><a href="https://lmstudio.ai/docs/advanced/sideload">Sideload models - Advanced | LM Studio Docs</a>: Use model files you&#x27;ve downloaded outside of LM Studio</li><li><a href="https://lmstudio.ai/docs">Getting Started | LM Studio Docs</a>: Learn how to run Llama, Mistral, Gemma, and other LLMs locally with LM Studio.</li><li><a href="https://lmstudio.ai">LM Studio - Experiment with local LLMs</a>: Run Llama, Mistral, Phi-3 locally on your computer.</li><li><a href="https://lmstudio.ai/models">Model Catalog - LM Studio</a>: The latest and greatest LLMs you can run on your computer.</li><li><a href="https://lmstudio.ai/docs/basics/download-model#changing-the-models-directory))">Download an LLM - Running LLMs Locally | LM Studio Docs</a>: Discover and download supported LLMs in LM Studio</li><li><a href="https://lmstudio.ai/docs/basics/chat#faq">Manage chats - Running LLMs Locally | LM Studio Docs</a>: Manage conversation threads with LLMs</li><li><a href="https://github.com/langflow-ai/langflow/pull/4021">feat: Add LM Studio Model and Embeddings Component by EDLLT · Pull Request #4021 · langflow-ai/langflow</a>: Fixes #3973</li><li><a href="https://lmstudio.ai/docs/basics/server">Local LLM Server - Running LLMs Locally | LM Studio Docs</a>: Run an LLM API server on localhost with LM Studio
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1291475762268082299)** (18 messages🔥): 

> - `LangChain Voice ReAct Agent`
> - `GPT-4o Dialogue`
> - `Meta Movie Gen Breakthrough`
> - `New LLM Leaderboard for Finance`
> - `Contextual Information Embedding Model` 


- **LangChain unveils Voice ReAct Agent**: Using the [Realtime API](https://x.com/langchainai/status/1841914757764485439?s=46), LangChain introduced a **Voice ReAct Agent** that integrates voice and tools to create custom voice experiences.
   - They demonstrated its capabilities with a video showing an agent performing actions with a calculator and a [Tavily web search tool](https://youtu.be/TdZtr1nrhJg).
- **GPT-4o Bots engage in conversation**: A demo showcased **two GPT-4o Voice AI bots** conversing using the Realtime API, highlighting advancements in voice AI technology.
   - The conversation involved different setups, showcasing the efficiency of the new API in *turn-taking latency*.
- **Meta announces Movie Gen project**: Meta's new breakthrough, **Meta Movie Gen**, aims to deliver advanced video generation capabilities without a set release date yet.
   - The research can be explored further on their [AI research page](https://ai.meta.com/research/movie-gen/) and its [associated paper](https://ai.meta.com/static-resource/movie-gen-research-paper).
- **New LLM rankings for finance hit the scene**: A recently published **LLM leaderboard** for finance highlights **OpenAI's GPT-4**, **Meta's Llama 3.1**, and **Alibaba's Qwen** as the leading models across 40 relevant tasks.
   - This new benchmark aims to refine performance evaluation, as detailed in the [Hugging Face blog](https://huggingface.co/blog/leaderboard-finbench).
- **Advancements in Contextual Embedding Models**: A new **contextual information embedding** model, cde-small-v1, has been developed to enhance text retrieval by incorporating *contextual tokens* during training.
   - The model's performance and theoretical foundation are documented in a recent [ArXiv paper](https://x.com/jxmnop/status/1842236045074498026?s=46) detailing the paradigm shift it represents.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_tim_brooks/status/1841982327431561528?s=46">Tweet from Tim Brooks (@_tim_brooks)</a>: I will be joining @GoogleDeepMind to work on video generation and world simulators! Can&#39;t wait to collaborate with such a talented team.  I had an amazing two years at OpenAI making Sora. Thank yo...</li><li><a href="https://x.com/jxmnop/status/1842236045074498026?s=46">Tweet from jack morris (@jxmnop)</a>: We spent a year developing cde-small-v1, the best BERT-sized text embedding model in the world.   today, we&#39;re releasing the model on HuggingFace, along with the paper on ArXiv.   I think our rele...</li><li><a href="https://x.com/ahmad_al_dahle/status/1842188269557301607?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: I couldn’t be more excited to share our latest AI research breakthrough. We call it Meta Movie Gen and it’s a collection of state-of-the-art models that combine to deliver the most advanced video gene...</li><li><a href="https://x.com/langchainai/status/1841914757764485439?s=46">Tweet from LangChain (@LangChainAI)</a>: 🎤 Voice ReAct Agent 🤖  Using @OpenAI &#39;s new Realtime API, you can use the power of voice + tools to build custom voice experiences.  Check out our video of us talking to a simple agent that reas...</li><li><a href="https://x.com/clefourrier/status/1842286565374193665?s=46">Tweet from Clémentine Fourrier 🍊 (@clefourrier)</a>: New LLM leaderboard: for Finance! 💰  It uses 40 domain-relevant tasks, from forecasting & risk management to question answering & information extraction!  Current top 3 models:  - @OpenAI&#39;s GPT4 ...</li><li><a href="https://x.com/jxmnop">Tweet from undefined</a>: no description found</li><li><a href="https://x.com/kwindla/status/1841936672755483115">Tweet from kwindla (@kwindla)</a>: Old 4o vs New 4o — a dialog between two generations of voice AI  Here&#39;s the demo I showed last night at the @cloudflare/@openai builders event.  This is two GPT-4o Voice AI bots talking to each ot...</li><li><a href="https://x.com/andersonbcdefg/status/1841987927049724120">Tweet from Ben (e/treats) (@andersonbcdefg)</a>: it&#39;s not lossless but it works !!</li><li><a href="https://x.com/sama/status/1841946796274176405?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sam Altman (@sama)</a>: now live to 100% of chatgpt plus subscribers!  Quoting Sam Altman (@sama)   check out canvas in chatgpt:  https://openai.com/index/introducing-canvas/</li><li><a href="https://x.com/OfficialLoganK/status/1841903061360640029">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: Say hello to Gemini 1.5 Flash-8B ⚡️, now available for production usage with:  - 50% lower price (vs 1.5 Flash) - 2x higher rate limits (vs 1.5 Flash) - lower latency on small prompts (vs 1.5 Flash)  ...</li><li><a href="https://x.com/ahmad_al_dahle/status/1842188269557301607?s=46&t=6FDPa">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: I couldn’t be more excited to share our latest AI research breakthrough. We call it Meta Movie Gen and it’s a collection of state-of-the-art models that combine to deliver the most advanced video gene...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1291852309009006723)** (98 messages🔥🔥): 

> - `Discord audio issues`
> - `Luma AI applications`
> - `Gaussian splatting`
> - `3D modeling in gaming`
> - `Virtual meetings` 


- **Discord audio struggles**: Users reported various challenges with Discord's audio functionality during a meeting, prompting suggestions to switch to Zoom or rejoin the call.
   - *One user humorously noted that no meeting feels genuine without microphone problems*, highlighting the common frustrations with online platforms.
- **Exciting uses for Luma AI revealed**: Members expressed enthusiasm about the capabilities of **Luma AI** for creating lifelike 3D models and integrating them into platforms like Unity or Unreal.
   - Several shared links showcasing **Luma AI's** functionalities in film editing and 3D modeling, indicating its potential in various creative fields.
- **Gaussian splatting and 3D representation**: The conversation included discussions around **Gaussian splatting**, particularly its significance in rendering and optimizing 3D environments for gaming.
   - Users referenced specific models and tools that incorporate Gaussian splatting, emphasizing the *great potential* for future developments in this area.
- **Interest in virtual meetings**: Participants expressed interest in setting up more virtual meetings to delve deeper into AI and 3D modeling topics discussed during the call.
   - *Calls for collaboration were noted*, as users shared excitement for future explorations and inquiries regarding the technology.
- **Gratitude and positive feedback**: As the conversation wrapped up, users expressed appreciation for the engaging discussions and shared knowledge throughout the call.
   - The opening remark, **AI in Action**, served as a thematic focus for the meeting, reinforcing the intention to explore AI advancements collectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://vimeo.com/1012136742/065081e415">FREE YOSHI - PROOF OF CONCEPT</a>: This is &amp;quot;FREE YOSHI - PROOF OF CONCEPT&amp;quot; by Jeremy Rubier on Vimeo, the home for high quality videos and the people who love them.</li><li><a href="https://x.com/aishashok14/status/1832760312455450907/video/1">Tweet from Aishwarya Ashok (@aishashok14)</a>: A night at the mountain—a Pixar-styled film :) ft. @midjourney (--sref 804246641), @LumaLabsAI (camera motions) and @udiomusic   What does it feel like to go on a hike, at the end of a tiring climb, q...</li><li><a href="https://x.com/karanganesan">Tweet from undefined</a>: no description found</li><li><a href="https://lumalabs.ai/web">Luma AI - Fields Dashboard</a>: Make your imagination reality with AI.</li><li><a href="https://x.com/aishashok14/status/1829738607281635371/video/1">Tweet from Aishwarya Ashok (@aishashok14)</a>: Slow is beautiful✨  Deep breaths, calm mind, peaceful warmth, unwinding moments…these are wholesome!   Here’s a reminder to all of us:  Slow is cool, slow is beautiful.   Ft. @midjourney and @LumaLabs...</li><li><a href="https://x.com/bennash/status/1840829850292011172?s=46">Tweet from Ben Nash (@bennash)</a>: text-to-video cockpit scene with the new 10X faster @LumaLabsAI</li><li><a href="https://x.com/aishashok14/status/1828790536410730878/video/1">Tweet from Aishwarya Ashok (@aishashok14)</a>: Brb, busy making a tea estate documentary AI film. ☕️ 🍃   From lush green plantation to the strongly brewed cup, the process of tea making is an emotion.   Captured with @midjourney & @LumaLabsAI wit...</li><li><a href="https://x.com/lumalabsai/status/1841833038700761205?s=46&t=fm_-fV17wG2CozW7wmZR7g">Tweet from Luma AI (@LumaLabsAI)</a>: 👀 Sooo... what&#39;s your pick? 🍊↔🍎? 🥕↔🥦? 🧁↔🍩? 🍔↔🍕? Made with #LumaDreamMachine Keyframes #foodforthought #hungry #foodie</li><li><a href="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/">3D Gaussian Splatting for Real-Time Radiance Field Rendering</a>: no description found</li><li><a href="https://lumalabs.ai/ios">‎Luma AI</a>: ‎Show your world in spectacular quality 3D, and share anywhere on the web. Brought to you by Luma AI.  Luma is a new way to create incredible lifelike 3D with AI using your iPhone. Easily capture prod...</li><li><a href="https://github.com/graphdeco-inria/nerfshop">GitHub - graphdeco-inria/nerfshop: NeRFshop: Interactive Editing of Neural Radiance Fields</a>: NeRFshop: Interactive Editing of Neural Radiance Fields - graphdeco-inria/nerfshop
</li>
</ul>

</div>
  

---



### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1291607508875612201)** (1 messages): 

> - `Performance benchmarks`
> - `Fio tools`
> - `Data access methods` 


- **Inquiry on Performance Benchmarks**: A member inquired about existing **performance benchmarks** related to certain tools and methodologies mentioned in the discussion.
   - They specifically sought comparisons between these benchmarks and raw performance analytics obtained from **fio tools** when accessing data directly from storage.
- **Comparative Analysis of Data Access Methods**: The discussion highlighted the need to analyze and compare the performance of data access methods.
   - Members are curious about how these methods stack up against traditional **fio tool** performance metrics.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1291794415425294338)** (2 messages): 

> - `SageAttention`
> - `Meta Movie Gen` 


- **SageAttention Quantization Breakthrough**: The [SageAttention](https://github.com/thu-ml/SageAttention) method achieves speedups of **2.1x** and **2.7x** compared to **FlashAttention2** and **xformers**, respectively, without losing end-to-end metrics across various models.
   - This quantization approach emphasizes efficiency while maintaining high performance in attention mechanisms.
- **Meta Unveils Movie Gen - A Creative Revolution**: Meta premiered **Movie Gen**, a suite of state-of-the-art media foundation models designed for creating high-quality images and high-definition videos from text prompts.
   - Key capabilities include **audio-video synchronization**, precise video editing, and the ability to generate **personalized videos** using user-provided images.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/i/status/1842188252541043075">Tweet from AI at Meta (@AIatMeta)</a>: 🎥 Today we’re premiering Meta Movie Gen: the most advanced media foundation models to-date.  Developed by AI research teams at Meta, Movie Gen delivers state-of-the-art results across a range of capa...</li><li><a href="https://github.com/thu-ml/SageAttention">GitHub - thu-ml/SageAttention: A quantization method for Attention that achieves speedups of 2.1x and 2.7x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models.</a>: A quantization method for Attention that achieves speedups of 2.1x and 2.7x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models. - thu-m...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1291606009399349280)** (2 messages): 

> - `Book Updates`
> - `Chapter Upgrades` 


- **Team Engaged in Chapter Upgrades**: The team is actively engaged in upgrading chapters and examples to enhance the book's content.
   - *We're trying our best to get there,* indicating their commitment to the improvements.
- **Significant Revamp of the New Book**: The upcoming book will be significantly revamped compared to prior editions, promising a fresh take on the material.
   - This revamp suggests a focus on better alignment with current standards and practices.


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1291490894234779790)** (4 messages): 

> - `Event Planning`
> - `Colocation with Conferences`
> - `Planning Timelines` 


- **Assumptions on Event Timing**: A member noted that planning would benefit from knowing the event date from a couple of months back and suggested it might occur in **September** after the Labor Day holiday.
   - This timeline helps align with the school season for better attendance.
- **Co-location Strategy with Conferences**: One member mentioned the likelihood of colocating with the **Triton** and **PyTorch** conferences to encourage group travel.
   - This strategy has previously provided attendees with a good reason to be in the same location.
- **Baby Steps in Event Planning**: A participant reflected on their initial experience with event planning, admitting it was the first event they ever helped plan, calling it **baby steps**.
   - They expressed that multi-month planning posed its challenges for them.
- **Learning Through Experience**: Another member complimented the initial planner for their efforts, despite their own experience with event planning, having organized around **six or seven** events.
   - They highlighted that even experienced planners can learn from each other during the process.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1291768550603882587)** (4 messages): 

> - `Noncontiguous inputs in Torchao`
> - `OptimState8bit dispatch error`
> - `AdamW8bit compatibility with Accelerate` 


- **Torchao struggles with noncontiguous inputs**: It's suggested that **Torchao** requires using **reshape** if the tensor is not contiguous to function correctly.
   - *This issue may limit its overall performance.*
- **Encountering OptimState8bit dispatch errors**: Members experience an error while attempting to use **OptimState8bit** which states 'attempting to run unimplemented operator/function: aten._to_copy.default'.
   - *This points to potential limitations in current implementations relevant to 8bit optimizers.*
- **AdamW8bit fails with Accelerate**: The **AdamW8bit** optimizer does not work with **Accelerate**'s save_state/load_state functionalities, leading to a NotImplementedError.
   - *Stack traces indicate that the error occurs within functions tied to optimizer state management.*


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1291538040719671428)** (22 messages🔥): 

> - `OpenAI's Financial Success`
> - `Potential New Products from OpenAI`
> - `Resume Review Channel Proposal`
> - `Grad School Application Discussions` 


- **OpenAI's financial success streak**: Members noted that **OpenAI** is setting records with their financial growth, driven by their recent innovations.
   - *This kind of revenue could be aimed at making their own chips,* with speculation about their expansion into hardware.
- **Discussion on building new products**: A member speculated about **OpenAI** potentially developing their own mobile device, hinting at applications of machine learning with user data.
   - This insight highlights concerns similar to how companies like **Apple** handle user data privacy.
- **Proposal for a resume-review channel**: A member suggested creating a channel dedicated to **resume reviews**, emphasizing the benefits of anonymized feedback from peers.
   - Discussions also included integrating mock interviews and community feedback, though the idea faced prioritization issues.
- **Interest in grad school application advice**: There was a call for a channel focused on **grad school applications**, with members expressing a desire for diverse perspectives.
   - One member volunteered assistance, indicating that discussions around academia's exploration of this field would be beneficial.
- **Emphasis on open-source project development**: One user shared concern about turning the community into a **CV review** and job-hunting forum, stressing a focus on open-source performance projects.
   - They expressed empathy for juniors struggling to find jobs, sharing their own lengthy job search journey, underscoring intrinsic motivation over salary-driven goals.


  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1291743090692063295)** (1 messages): 

> - `Triton kernel performance`
> - `Tensor operations`
> - `Debugging Triton functions` 


- **User struggles with unchanged results in Triton kernel**: A user expressed frustration that their results don't seem to change regardless of the modifications made to the code in their Triton kernel.
   - *Has anyone faced this problem before?* They provided a code snippet for context.
- **Code snippet for adding a constant in Triton**: The user shared a code snippet demonstrating their implementation of a Triton kernel that adds a constant to a tensor using `tl.store`.
   - The `add_kernel` function loads values from pointers and attempts to perform an addition operation on them.


  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1291644916082151434)** (1 messages): 

> - `BF16 stochastic rounding`
> - `Grad norm analysis`
> - `Data shuffling concerns` 


- **BF16 Stochastic Rounding Boosts Performance**: Adding **BF16 stochastic rounding** to the weight update leads to a **non-trivial improvement** in performance.
   - This technique seems to enhance the overall efficiency of the model training process.
- **Grad Norm Curve Shows Interesting Gap**: The gap in the **grad norm curve** presents an intriguing observation that remains unexplained.
   - Further analysis may be necessary to understand its implications on model training and convergence.
- **Potential Issues with Data Shuffling**: The observed **pattern in loss curves** suggests potential insufficient data shuffling during training.
   - Improving the data shuffling process might help refine the model's learning and boost performance.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1291610706885476433)** (11 messages🔥): 

> - `Conv2d Triton Kernel Performance`
> - `Scaled Int8 Conv2d Exploration`
> - `Liger vs. PyTorch Performance`
> - `Fused KL/JSD Requirement Clarification` 


- **Conv2d Triton Kernel Performance Insights**: Discussion about the performance of the **Conv2d Triton kernel** indicated it is currently slower than the baseline **PyTorch BF16 conv2d** implementation, with dependence on input size for speed.
   - One member plans to revisit and optimize the kernel after current school commitments settle in about two weeks.
- **Exploring Scaled Int8 Conv2d Potential**: Concerns were raised about using a reasonable **Triton Conv2d implementation**, with expectations that speedups from **int8 tensor cores** would compensate for slower **BF16 Triton conv2d**.
   - One member emphasized improving configuration and auto-tuning to enhance performance.
- **Performance Comparison: Liger vs. PyTorch**: Testing revealed that the **Liger framework** is approximately **8x slower** than the **Torch Compile** under certain conditions, possibly due to misconfigured flags.
   - This indicates a need for further investigation into performance tuning for the Liger project.
- **Clarifying Fused KL/JSD Implementation Requirements**: A member sought clarification on implementing the **fused KL/JSD loss**, questioning if only the teacher's logits are necessary and whether softmax and temperature adjustments should apply.
   - Their proposed implementation structure was laid out, but they encouraged feedback on the approach to ensure accuracy.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1291487739874639872)** (5 messages): 

> - `Hyperparameter Scaling Guide`
> - `Open Source Project Maintenance`
> - `Embedding Geometries Paper Acceptance`
> - `Contrastive Language-Image Pre-Training`
> - `Euclidean vs Hyperbolic Geometry` 


- **Need for a Hyperparameter Scaling Guide**: A member expressed confusion over **hyperparameter scaling**, highlighting a lack of accessible heuristics for training experiments on larger models, noting that existing information is often confined to individual researchers.
   - *Maybe a guide exists... and I'm an idiot for not being able to find it* reflects the struggle for clarity in this complex topic.
- **Upcoming Article on Open Source Maintenance**: A member teased an upcoming article on **maintaining open source projects**, promising it will be both longer and better written than previous posts.
   - This topic hints at insights and strategies that could benefit the open source community.
- **ECCV '24 Paper on Alternative Embedding Geometries**: The paper titled '*Embedding Geometries of Contrastive Language-Image Pre-Training*' has been accepted by the **ECCV '24 Beyond Euclidean Workshop**, exploring systematic tests of various embedding geometries.
   - The findings indicate that intuitive **Euclidean geometry** outperforms both conventional **CLIP** and **MERU** in zero-shot scenarios.
- **CLIP Design Choices Revisited**: In the discussed paper, authors review original **CLIP** design choices and find that their experiments with **Euclidean CLIP** (EuCLIP) offer similar or superior performance compared to **hyperbolic alternatives**.
   - They emphasize the importance of revisiting foundational aspects of contrastive pre-training despite its popular adoption.
- **Links to Research and Personal Site**: The member provided links to various resources including their personal site and multiple blog posts, promoting further engagement with their work.
   - Included were links to significant projects and papers such as the **pgen parser generator** and articles on **compiler writing and bug tracking**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://apaz-cli.github.io/blog/Hyperparameter_Heuristics.html">Hyperparameter Heuristics</a>: no description found</li><li><a href="https://arxiv.org/abs/2409.13079">Embedding Geometries of Contrastive Language-Image Pre-Training</a>: Since the publication of CLIP, the approach of using InfoNCE loss for contrastive pre-training has become widely popular for bridging two or more modalities. Despite its wide adoption, CLIP&#39;s orig...</li><li><a href="https://apaz.dev">apaz's Website</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1291829811374850048)** (48 messages🔥): 

> - `AVX2 Emulation`
> - `Matrix Multiplication Implementation`
> - `Performance Testing`
> - `Parallel Programming Resources`
> - `Tinygrad with AVX Intrinsics` 


- **AVX2 Emulation Discussion**: Members discussed the dependency of emulating **AVX512** on specific goals, noting that validating implementations will yield differing performance outcomes.
   - One member aims to create a **library of implementations** for basic arithmetic using vector extensions of GCC and Clang.
- **Matrix Multiplication Exercise at Aalto**: A **course at Aalto University** offers an exercise that involves benchmarking code on an Intel CPU with native **AVX512** support, open to anyone for registration.
   - Members noted that the exercise includes automated benchmarks and unit tests for implementations, making it valuable for programming practice.
- **Parallel Programming Resource GitHub**: A member shared a GitHub repository at [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream) containing programming resources for both **CPU and GPU**.
   - The repository provides links to materials related to **GPU programming**, highlighting a lack of parallel programming classes at some institutions.
- **Tinygrad Compilation to AVX Intrinsics**: One member expressed interest in experimenting with **Tinygrad**, aiming to compile it to **AVX intrinsics** for improved performance.
   - This idea aligns with the ongoing discussion on hardware utilization and performance benchmarks.
- **Weight Loading Challenges in Python Implementation**: A member shared challenges faced in matching weight loading for their **Python implementation**, noting it remains a work in progress.
   - They expressed interest in leveraging existing resources to enhance their understanding and implementation practices.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.cppreference.com/w/cpp/experimental/simd">SIMD library - cppreference.com</a>: no description found</li><li><a href="https://github.com/gpu-mode/resource-stream">GitHub - gpu-mode/resource-stream: GPU programming related news and material links</a>: GPU programming related news and material links. Contribute to gpu-mode/resource-stream development by creating an account on GitHub.</li><li><a href="https://github.com/AndreSlavescu/EasyAI/blob/main/src/kernels/cpu_avx/matrix_methods/matrix_transpose_nn.cpp">EasyAI/src/kernels/cpu_avx/matrix_methods/matrix_transpose_nn.cpp at main · AndreSlavescu/EasyAI</a>: Learning tool for all! Contribute to AndreSlavescu/EasyAI development by creating an account on GitHub.</li><li><a href="https://github.com/addaleax/sw-simd">GitHub - addaleax/sw-simd: AVX2 software polyfill for CPUs supporting AVX instructions.</a>: AVX2 software polyfill for CPUs supporting AVX instructions. - addaleax/sw-simd
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1291504111476408342)** (65 messages🔥🔥): 

> - `Perplexity AI Collections UI`
> - `Boeing 777-300ER Specifications`
> - `TradingView Premium Package`
> - `Llama 3.2 Release`
> - `Claude 3.5 vs Other Models` 


- **Perplexity AI Working on New Collections UI**: Recent discussions reveal that Perplexity AI is developing a new user interface for its Collections feature, focusing on displaying custom instructions and enabling file uploads, though not yet publicly available.
   - This anticipated **Files search feature** will enhance user experience by organizing information more effectively.
- **Boeing 777-300ER Full Specifications Shared**: A comprehensive outline of the **Boeing 777-300ER** specifications was provided, highlighting its dimensions, performance, powerplant, capacity, and additional features.
   - Noted details include a **maximum range** of **7,370 nautical miles** and a seating capacity for up to **550 passengers** in a single-class layout.
- **TradingView Premium Cracked Version Released**: A member shared a link to a free cracked version of **TradingView Premium** (Version 2.9), boasting advanced tools for traders across various markets.
   - This version allows access to premium features without payment, appealing to numerous users looking for top-tier charting solutions.
- **Anticipation for Llama 3.2 Release**: Users are inquiring about the release date for **Llama 3.2**, expressing excitement and curiosity about its upcoming features.
   - The conversation indicates a strong interest in the progress and expected improvements from this new iteration.
- **Comparison of Claude 3.5 and Other AI Models**: There were discussions comparing the capabilities of **Claude 3.5 Sonnet** to other AI models, with many asserting it to be more reliable for obtaining information.
   - Users expressed interest in the combined potential of Perplexity Pro with Claude for enhanced performance in information retrieval from textbooks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.testingcatalog.com/perplexity-working-on-new-collections-ui-for-custom-instructions-and-file-uploads/">Perplexity working on new Collections UI with file uploads</a>: Discover Perplexity AI&#x27;s upcoming features: a new UI for custom instructions and file uploads. Stay tuned for enhanced search capabilities and file management.</li><li><a href="https://www.reddit.com/r/Cracked_Software_Hub/comments/1fo875c/tradingview_premium_cracked_version_available_for/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/Cracked_Software_">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1291579307516231752)** (5 messages): 

> - `U2V`
> - `Kreutzer's Etudes`
> - `Four-legged Robot`
> - `Quantum Clocks`
> - `Enum Values` 


- **Adding a New Enum Value Explained**: A user shared a query about [how to add a new enum value](https://www.perplexity.ai/search/how-do-you-add-a-new-enum-valu-fBPEV5LtStO_P19ZMVXiQQ), focusing on specific implementation details.
   - The discussion included considerations for compatibility and code integrity while modifying enums.
- **Thoughts on U2V**: A member asked for opinions on [U2V](https://www.perplexity.ai/search/hey-what-are-your-rhoughts-on-u2vFogOaTzOse8ibsQNwrA), highlighting its relevance and applications.
   - Responses discussed its potential impact and effectiveness in various contexts.
- **Why Kreutzer's Etudes are Important**: A post focused on [the significance of Kreutzer's Etudes](https://www.perplexity.ai/search/why-kreutzer-s-etudes-are-one-d8wh8YTgQnm8AO63eJnnvw#0) in music education, emphasizing technique development.
   - Participants shared insights regarding the etudes' role in mastering violin performance.
- **Four-legged Robot Climbs Ladder**: A link was shared regarding a [robot that climbs ladders](https://www.perplexity.ai/page/four-legged-robot-climbs-ladde-OT7S9LK0R.iJ6Yq0c7QmOg), showcasing its design and capabilities.
   - Discussion revolved around the implications of such technology in practical applications.
- **Understanding Quantum Clocks**: A user inquired about [quantum clocks](https://www.perplexity.ai/search/what-is-a-quantum-clock-t4A_.5lTTiCUnbMObd_5_A), seeking to understand their principles and accuracy.
   - Contributions highlighted the advancements in timekeeping and potential innovations driven by this technology.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1291480400366932031)** (2 messages): 

> - `Command R 08-2024 Update`
> - `Integration with Weights & Biases` 


- **Command R 08-2024 Fine-tuning Highlights**: The updated *Command R 08-2024* introduces support for newer options designed to provide users with **more control** and **visibility**.
   - This update also features a **seamless integration** with [Weights & Biases](https://cohere.com/blog/fine-tuning-command0824) for enhanced performance tracking.
- **Waves of Excitement for Command R**: Members expressed **enthusiasm** for the Command R update, highlighting the blend of new features and improved usability.
   - Comments like '*Awesome*' capture the overall excitement and anticipation from the community.



**Link mentioned**: <a href="https://cohere.com/blog/fine-tuning-command0824">Updates to Command R Fine-tuning</a>: Fine-tune the updated Command R 08-2024 with support for newer options giving you more control and visibility including a seamless integration with Weights &amp; Biases.

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1291485248865042472)** (39 messages🔥): 

> - `Metrics Visibility Issues`
> - `Fine-Tuning Challenges`
> - `Tool Use in Next.js`
> - `RAG with Embedding Datasets`
> - `UI Feedback on Colabs` 


- **Metrics are missing in the platform**: A user reported that they are unable to see the **metrics boxes** for their models across various tabs like Overview and API, which previously displayed essential information.
   - They expressed concern about the consistency of the platform and questioned the status of model creation, highlighting that it's taken **2 days** without resolution.
- **Troubleshooting Fine-Tuning Uploads**: Another member encountered multiple errors when trying to fine-tune a chatbot using JSON training documents, including issues with encoding and parsing.
   - They requested guidance and a sample JSON file that would be compatible with the Cohere platform.
- **Query on Tool Use Example in Next.js**: A user sought a simple example of using Tool use (Single Step) in **Next.js**, noting that most documentation is in Python.
   - Contributors suggested checking whether switching to v2 could address some issues.
- **Embedding Datasets for RAG**: A user stated they uploaded an embedding dataset intending to leverage RAG, but found they couldn't connect it to a chat, raising concerns on usability.
   - They inquired about the process of embedding CSV chunks effectively for their needs.
- **Feedback on Colabs and UI**: Users expressed frustration that several Colabs in the documentation are broken, providing feedback for improvements.
   - Participants were encouraged to share specific instances where the code generated errors or where updates were needed.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/v2/docs/structured-outputs-json">Structured Generations (JSON) — Cohere</a>: This page describes how to get Cohere models to create outputs in a certain format, such as JSON.</li><li><a href="https://docs.cohere.com/v2/docs/chat-fine-tuning">Fine-tuning for Chat — Cohere</a>: This document provides guidance on fine-tuning, evaluating, and improving chat models.</li><li><a href="https://docs.cohere.com/v2/docs/tool-use">Tool Use — Cohere</a>: Enable your large language models to connect with external tools for more advanced and dynamic interactions.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1291746700834242600)** (5 messages): 

> - `Pricing Discrepancy`
> - `Finetuning Commands`
> - `Documentation Updates` 


- **Pricing Page Confusion**: The **pricing page** indicates **$3 per 1M tokens** for training, but the finetune UI shows a price of **$8**.
   - This discrepancy raises questions about the accuracy of the pricing information across different platforms.
- **Command Shortcut Queries**: There was a question about whether the default command for training is set to **cmd-r+** and if it can be changed to **cmd-r**.
   - This inquiry reflects concerns over user experience and interface customization.
- **Uncertainty of Command Shortcuts on Finetuning**: A member expressed uncertainty about whether **cmd-r+** is even applicable in the finetuning process.
   - This indicates a potential gap in user knowledge regarding command functionality.
- **Outdated Documentation Concerns**: There are suggestions that the documentation might still be outdated, contributing to the confusion over commands and pricing.
   - Stale documentation can significantly affect user experience and troubleshooting.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

kittykills: Hello!
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1291480042303524985)** (44 messages🔥): 

> - `OpenPose Alternatives`
> - `ComfyUI Image Quality`
> - `SDXL Models`
> - `Reference Image Generation`
> - `AI Tools for Object Placement` 


- **OpenPose Alternatives for Poses**: Users discussed issues with **OpenPose** for generating sitting poses and alternatives like **DWPose**, questioning where to find better models.
   - *Training one’s own model could also be a viable solution with sufficient reference images available.*
- **Enhancing ComfyUI Output Quality**: A member inquired about getting **ComfyUI** to produce images as high quality as **Auto1111**, noting the resultant images look funky or cartoony.
   - *Using specific nodes in ComfyUI was suggested as a potential method for achieving better quality outputs.*
- **SDXL Model Clarifications**: Users discussed different versions of **SDXL**, including `SDXL 1.0`, and their individual properties such as resolution capabilities, which typically start at **1024x1024**.
   - *Some confirmed that all variations are based on the **SDXL 1.0** model.*
- **Generating Poses from Reference Images**: It was confirmed that using a single reference image for generating poses in **Stable Diffusion** is possible, but may not produce the most accurate results.
   - *Img2img was cited as the correct approach, though having multiple images from different angles would yield better fidelity.*
- **Need for AI Tools for Object Placement**: There was a query regarding **OpenPose** techniques that can help in placing objects in poses, with a suggestion of utilizing a LoRA model for specific items like swords.
   - *Users noted that while some training styles exist in Stable Diffusion, a dedicated method for posing remains lacking.*


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1291723957317271552)** (4 messages): 

> - `Translation of Technical Language`
> - `Language Barriers in Tech` 


- **Single Line of Code Enables Language Change for Captions**: A member suggested that it's just a **single line of code** to change speech and captions to another language.
   - *This makes it easier for multilingual support* in technical applications.
- **Challenges of Translating Technical Terms**: A member pointed out the challenge that the technical world is predominantly in **English** and many terms don't require translation.
   - *Terms like embeddings, manifold, and transformers* can be tough to manage in non-English contexts.
- **Understanding and Acceptance of Language Preferences**: Another member acknowledged the difficulty, saying they understand the frustrations surrounding translation in tech.
   - *Language preferences can complicate communication in technical discussions*.


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1291494195617333309)** (14 messages🔥): 

> - `MinGRU Architecture`
> - `Training Bark-like Models`
> - `Scam Alert` 


- **MinGRU Simplifies Recurrent Neural Networks**: The paper introduces a minimalist version of **GRUs**, termed **minGRUs**, that eliminates hidden state dependencies, allowing efficient parallel training at **175x** speed increase.
   - Its straightforward architecture consists of two linear layers and employs parallel processing to compute hidden states, provoking thoughts on the simplicity of potential solutions in **NLP**.
- **Seeking Guidance for Bark-like Model**: A newcomer expressed interest in training a **Bark-like model** from scratch, aiming for a two to three-month completion timeframe, and sought resources or papers for guidance.
   - A suggestion was made to consider the **Vall-E paper** as a foundational resource for understanding the training process.
- **Scam Warning in Community**: A user identified another member as a potential scammer offering a way to earn **$50k** in 72 hours in exchange for a 10% profit share.
   - Community members were cautioned against this scheme, highlighting concerns about the authenticity of the offer.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.01201">Were RNNs All We Needed?</a>: The scalability limitations of Transformers regarding sequence length have renewed interest in recurrent sequence models that are parallelizable during training. As a result, many novel recurrent arch...</li><li><a href="https://t.me/Official_HugoLarsson">Hugo Larsson</a>: The secret of getting ahead is getting started 🤝
</li>
</ul>

</div>
  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1291845339707609118)** (1 messages): 

> - `Earning Opportunities`
> - `Telegram Contact` 


- **Quick Cash Scheme Proposal**: A member is offering guidance to the first **10 interested people** on how to start earning **$50k or more within 72 hours**, requesting a **10% reimbursement** of profits.
   - *Interested individuals* are encouraged to reach out via **Telegram** to discuss details.
- **Connect with Hugo on Telegram**: The contact for more information is provided as **Hugo Larsson** on **Telegram**, with a direct link for messaging.
   - He emphasizes, *'The secret of getting ahead is getting started.'*



**Link mentioned**: <a href="https://t.me/Official_HugoLarsson">Hugo Larsson</a>: The secret of getting ahead is getting started 🤝

  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1291556189863411766)** (2 messages): 

> - `Training BARK Model`
> - `Earn Money Quickly` 


- **Seeking Guidance on BARK Model Training**: A new member expressed interest in training a **BARK-like model** from scratch with custom features within a **2-3 month timeframe** but struggled to find relevant papers related to BARK.
   - *They requested suggestions on how to learn about this process*, noting that training details seemed to relate closely to models like Audio LM and VALL-E.
- **Quick Cash Opportunity from Hugo**: A member, **Hugo**, offered assistance to the first **10 interested people** to **earn $50k or more within 72 hours** in exchange for 10% of their profits.
   - *Interested individuals were instructed to send him a friend request or a DM via Telegram*, highlighting that getting started is essential for success.



**Link mentioned**: <a href="https://t.me/Official_HugoLarsson">Hugo Larsson</a>: The secret of getting ahead is getting started 🤝

  

---


### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/1291845216684343369)** (1 messages): 

> - `Earn $50k in 72 hours`
> - `Telegram outreach` 


- **Earn $50k in 72 hours scheme**: A proposal was made to assist the first 10 interested individuals in **earning $50k or more within 72 hours**, with a 10% reimbursement on profits.
   - Interested persons were encouraged to send a friend request or direct message on **Telegram** for further details.
- **Direct Telegram engagement**: The facilitator, **Hugo Larsson**, provided a contact link to reach out via **Telegram** for inquiries regarding the earnings scheme.
   - Hugo emphasized that *'the secret of getting ahead is getting started'* and urged potential participants to engage directly.



**Link mentioned**: <a href="https://t.me/Official_HugoLarsson">Hugo Larsson</a>: The secret of getting ahead is getting started 🤝

  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1291618288429957120)** (19 messages🔥): 

> - `Article Score Inquiry`
> - `Real-time Streaming of Responses`
> - `Chainlit Integration`
> - `Github Autogen Pull Requests`
> - `Course Location on Campus` 


- **Inquiry about Article Scores**: A member asked how to view scores for three articles they submitted, including a draft and LinkedIn links.
   - This inquiry highlights ongoing concerns about submission feedback in the community.
- **Real-time Streaming Challenge**: A member expressed a desire to stream **chat_manager** responses directly into the frontend in real-time, stating that by default, responses stream only after garbage collection completes.
   - Another member confirmed that there exists a Streamlit UI that streams responses in real-time, mentioning it was built around 8 months ago.
- **Chainlit to the Rescue**: A member indicated that a solution using **Chainlit** exists, with a potential recipe available in the AutoGen project on GitHub.
   - They noted this implementation seems to fulfill the requirements for real-time chat management.
- **GitHub Autogen Pull Request Discussion**: A member shared a relevant [GitHub pull request](https://github.com/microsoft/autogen/pull/1783) that discusses processing messages before sending them, which could be useful for customizing message displays.
   - This development contributes alignment with the previous real-time streaming inquiries.
- **Course Location Inquiry**: A member inquired about the specific room on Berkeley Campus where a certain course is held.
   - This highlights logistical interests as the community coordinates activities related to the course.



**Link mentioned**: <a href="https://github.com/microsoft/autogen/pull/1783">process message before send by sonichi · Pull Request #1783 · microsoft/autogen</a>: Why are these changes needed?  Add a hookable method for processing a message before sending. Example application: customized frontend to display messages . Renamed other hookable methods for clari...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1291495504517009430)** (5 messages): 

> - `Building AI agents with LlamaCloud`
> - `Security in RAG`
> - `Real-time audio APIs from OpenAI`
> - `Avoiding hallucination in RAG`
> - `Hackathon announcement` 


- **Build AI Agents with LlamaCloud**: Learn how to build AI agents using [LlamaCloud and Qdrant Engine](https://twitter.com/llama_index/status/1841935964081627315), focusing on implementing **semantic caching** to enhance speed and efficiency.
   - The demo covers advanced agent techniques, including **query routing** and **query decomposition**.
- **Enhance Security in RAG Deployments**: A discussion arose about using [Box's enterprise-grade security](https://twitter.com/llama_index/status/1841950022835044833) in conjunction with LlamaIndex to ensure robust permissions for secure RAG implementations.
   - Members highlighted the importance of a **seamless, permission-aware RAG** experience.
- **Voice Interaction with OpenAI APIs**: Marcus demonstrated a new feature using [OpenAI's real-time audio APIs](https://twitter.com/llama_index/status/1842236491784982982) that allows users to chat with documents through voice commands.
   - This innovative approach simplifies document interaction by enabling conversation using your voice.
- **Combat Hallucination in RAG**: To prevent hallucination in RAG, [CleanlabAI's solution](https://twitter.com/llama_index/status/1842259131274817739) integrates a trustworthiness scoring system to evaluate LLM responses.
   - This method helps identify and eliminate low-quality data points, boosting the overall dataset quality.
- **Exciting Hackathon Opportunity**: The second hackathon, with over **$12,000 in cash prizes**, kicks off on October 11th at [500 Global VC's headquarters](https://twitter.com/llama_index/status/1842274685989576947) in Palo Alto.
   - Participants can learn to build exciting projects while competing for cash prizes throughout the weekend.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1291482823131402252)** (11 messages🔥): 

> - `Agent Class with Streaming`
> - `Integrating LLM with BigQuery`
> - `Error Handling in Code`
> - `OpenAIAgent for Streaming`
> - `Custom Agent Development` 


- **Agent class needs streaming support**: A user inquired about an existing agent class that supports **chat_memory**, tools, and **streaming** responses, particularly for function calling and context management.
   - Another member recommended using the **OpenAIAgent** or building a custom agent with **async streaming** and dynamic context retrieval, sharing a [Colab notebook](https://colab.research.google.com/drive/1wVCkvX7oQu1ZwrMSAyaJ8QyzHyfR0D_j?usp=sharing) for reference.
- **Integrating LLM with BigQuery**: A user is attempting to integrate an LLM with a BigQuery table for real-time prompting but encountered errors during the process.
   - Suggestions were made to provide the specific error message to better assist with troubleshooting and to format code using triple backticks for clarity.
- **Error in the code during integration**: A user shared code attempting to integrate their LLM with BigQuery but did not specify the error they encountered.
   - Community members encouraged sharing the error details for more targeted help and emphasized the importance of code readability.



**Link mentioned**: <a href="https://colab.research.google.com/drive/1wVCkvX7oQu1ZwrMSAyaJ8QyzHyfR0D_j?usp=sharing">Google Colab</a>: no description found

  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1291523368247627898)** (7 messages): 

> - `dslmodel live demos`
> - `Sentiment Analysis`
> - `Document Summarization`
> - `Arxiv Paper Structure`
> - `New Features in DSLModel` 


- **Upcoming live demos for dslmodel**: Live demos of **dslmodel** are scheduled for **4:30 PST**.
   - Participants are encouraged to join the demonstrations in the lounge for interactive coding.
- **Sentiment Analysis yields positive results**: The SentimentModel successfully classified the sentence ‘This is a wonderful experience!’ with **sentiment='positive'** and **confidence=1.0**.
   - This demonstrates the model's reliability in sentiment classification tasks.
- **Summarization Model captures essence**: A document summarization using the SummarizationModel provided a concise summary: '**Motivational speech on success and perseverance.**'
   - The model highlighted themes of control, success, and resilience in its reasoning.
- **Structure of Arxiv Paper implemented**: An Arxiv paper model was demonstrated using a class setup that included lead author and co-authors' details.
   - The paper discussed introduces **DSPy**, an important programming model for language processing.
- **Funny moments captured in gifs**: A humorous gif was shared showing a man in a black turtleneck with a funny expression, captioned ‘Mind Blow’.
   - The gif serves to illustrate the reaction many have to ‘mind-blowing’ concepts shared in the channel.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/mind-blow-galaxy-explode-boom-fireworks-gif-5139389">Mind Blow Galaxy GIF - Mind Blow Galaxy Explode - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/seanchatmangpt/dslmodel/blob/main/src/dslmodel/examples/dspy.ipynb">dslmodel/src/dslmodel/examples/dspy.ipynb at main · seanchatmangpt/dslmodel</a>: Structured outputs from DSPy and Jinja2. Contribute to seanchatmangpt/dslmodel development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1291752125679800391)** (4 messages): 

> - `DSPy full form`
> - `Backronym for DSPy` 


- **DSPy stands for Declarative Self-improving Language Programs**: A member clarified that the current backronym for DSPy is **Declarative Self-improving Language Programs**, pythonically.
   - They humorously noted that DSPy is also referred to as **Declarative Self-Improving Python**.
- **Community inquiry about DSPy**: A community member asked for the full form of DSPy, initiating a discussion on its meaning.
   - This inquiry prompted a friendly exchange on the interpretations and humor surrounding the acronym.


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1291704918096347198)** (4 messages): 

> - `Text Classification Tasks`
> - `DSPy Signatures`
> - `LM Behavior Specification` 


- **Sharing an example for text classification**: A user requested for an example related to **text classification tasks**.
   - *lmk if this helps!*
- **Understanding DSPy Signatures**: Another user shared a link explaining **DSPy signatures** as declarative specifications for input/output behavior in a module.
   - These signatures allow users to define and control module behavior, contrasting with typical function signatures that simply describe parameters.



**Link mentioned**: <a href="https://dspy-docs.vercel.app/docs/building-blocks/signatures#example-c-classification">Signatures | DSPy</a>: When we assign tasks to LMs in DSPy, we specify the behavior we need as a Signature.

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1291482053287874742)** (7 messages): 

> - `Event Participation Limit`
> - `Human Devices Event`
> - `Obelisk GitHub Tool` 


- **Event Participation Limit Rolls Back to 25**: Members noted that the participation for the event was capped at **25 people**, despite a change proposed by MikeBirdTech to **99**.
   - One user confirmed repeated attempts to join but still encountered a **full** status.
- **Join the Human Devices Event**: MikeBirdTech shared the link for the upcoming **Human Devices event** and provided a Discord URL for access: [Join Here](https://discord.gg/mzcrk6pZ?event=1291393902758330389).
   - Participants are encouraged to **request or share** anything related to the event in the designated channel.
- **Obelisk: A Handy GitHub Tool**: A member highlighted the **Obelisk** project from GitHub, a tool for saving web pages as a single **HTML file**.
   - They suggested that it could be **quite useful in many contexts**, providing a link for others to explore: [GitHub - go-shiori/obelisk](https://github.com/go-shiori/obelisk).



**Link mentioned**: <a href="https://github.com/go-shiori/obelisk">GitHub - go-shiori/obelisk: Go package and CLI tool for saving web page as single HTML file</a>: Go package and CLI tool for saving web page as single HTML file - go-shiori/obelisk

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/)** (1 messages): 

ellsies_: no logs at all
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1291579642226151466)** (5 messages): 

> - `Meta Movie Gen`
> - `Open Source Discussion` 


- **Meta Movie Gen Launches**: Today, [Meta premiered Movie Gen](https://x.com/aiatmeta/status/1842188252541043075?s=46&t=G6jp7iOBtkVuyhaYmaDb0w), a suite of advanced media foundation models designed to enhance video and audio creation.
   - The models can generate high-quality images and videos, as well as audio synced to video with impressive alignment and quality.
- **Open Source Vision from Mozilla**: In response to a query about the openness of Meta Movie Gen, a member clarified that while **Mozilla** promotes open source, this initiative is more about showcasing their vision.
   - Discussion highlighted the distinction between Mozilla's principles and the nature of Movie Gen, emphasizing it remains aligned with their broader goals.



**Link mentioned**: <a href="https://x.com/aiatmeta/status/1842188252541043075?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">Tweet from AI at Meta (@AIatMeta)</a>: 🎥 Today we’re premiering Meta Movie Gen: the most advanced media foundation models to-date.  Developed by AI research teams at Meta, Movie Gen delivers state-of-the-art results across a range of capa...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1291480614528090165)** (12 messages🔥): 

> - `FAANG SDLC certifications`
> - `LangChain API updates`
> - `LangChain support for GPT real-time API`
> - `Evaluating RAG pipelines`
> - `Creating a chatbot with LangChain` 


- **FAANG companies seek SDLC certification**: A user inquired about widely recognized courses or certifications for **Software Development Lifecycle (SDLC)** that are acknowledged by FAANG companies, apart from **PMP**.
   - This highlights a common concern among applicants transitioning from different industries into tech roles.
- **Changing API calls in LangChain**: A user mentioned noticing changes in the **API chain** for LangChain and is seeking the latest methods for calling the API.
   - This indicates ongoing updates and developments within the LangChain framework.
- **Inquiry about LangChain's GPT real-time API support**: A user asked when **LangChain** would support the newly announced **GPT real-time API**.
   - A response included a link to a [YouTube video](https://www.youtube.com/watch?v=TdZtr1nrhJg) for further clarification.
- **Evaluating RAG pipeline retrievers**: A user sought advice on how to evaluate and compare the performance of three different **retrievers** in their **RAG pipeline**.
   - Another member suggested using **query_similarity_score** to determine the best-performing retriever and offered to provide code snippets via LinkedIn.
- **Building a chatbot with LangChain**: A user asked for guidance on creating their own **chatbot** using **LangChain**.
   - This reflects a growing interest in leveraging LangChain for chatbot development.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1291484840847216691)** (3 messages): 

> - `NeurIPS 2024 Conference Date Change`
> - `Elon Musk's xAI Recruiting Event`
> - `OpenAI's Dev Day`
> - `Funding Rumors` 


- **NeurIPS 2024 adjusts dates for Taylor Swift fans**: The start date for the **NeurIPS 2024** conference has been moved to **Tuesday, December 10**, allowing delegates to arrive the day before.
   - This change was humorously noted as being influenced by **Taylor Swift's Eras Tour**, which caused a shift in plans.
- **Elon Musk hosts a security-heavy xAI recruiting bash**: A recruiting event for **Elon Musk's xAI** saw live music generated via code while attendees faced metal detector screenings and ID checks.
   - The event was timed to coincide with **OpenAI's Dev Day**, creating buzz as Musk seeks talent amidst funding rumors.
- **OpenAI CEO speaks at a packed Dev Day**: On the same day as Musk's event, **Sam Altman**, CEO of **OpenAI**, addressed a crowded auditorium of developers during their annual **Dev Day**.
   - Rumors circulated about OpenAI potentially closing in on the largest round of startup funding to date.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/10/3/24261160/elon-musk-xai-recruiting-party-openai-dev-day-sam-altman">Inside Elon Musk’s AI party at OpenAI’s old headquarters</a>: Elon Musk threw an xAI recruiting party in OpenAI’s original San Francisco headquarters.</li><li><a href="https://fxtwitter.com/WilliamWangNLP/status/1841879266142904469">Tweet from William Wang (@WilliamWangNLP)</a>: BREAKING: Taylor Swift&#39;s Eras Tour just did what AI couldn’t—pushed NeurIPS by a whole day! 🤖 🤣🤣🤣  #NeurIPS 2024 Conference Date Change The conference start date has been changed to Tuesday De...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1291620845357236269)** (8 messages🔥): 

> - `Meta Movie Gen`
> - `Model Optimization Techniques`
> - `LLMs and Code Synthesis Reinforcement Learning`
> - `OpenAI's Model Distillation`
> - `Canvas Development` 


- **Meta Movie Gen Launches Advanced Features**: Meta premiered [Movie Gen](https://go.fb.me/kx1nqm), a suite of media foundation models capable of generating high-quality images, videos, and audio from text prompts, boasting impressive capabilities like personalized video creation.
   - *We’re continuing to work closely with creative professionals* to enhance the tool's features before a potential release.
- **Innovative Model Layout Optimization**: In the Movie Gen paper, it was highlighted that Meta developed modeling tools to optimize the layout during training, enabling a complex parallelism strategy that effectively matched their models with the hardware.
   - *This optimization allows for better training efficiency* and performance across video and audio generation tasks.
- **Reinforcement Learning Enhances LLMs for Code**: A new paper proposes an end-to-end reinforcement learning method for LLMs deployed as agents, achieving state-of-the-art results in competitive programming tasks while leveraging execution feedback.
   - This method demonstrates significant improvements in iterative code synthesis, achieving results with smaller models while drastically reducing sample requirements.
- **Canvas Development with OpenAI's Distillation**: A developer shared insights about building Canvas, utilizing novel synthetic data techniques to enhance interactions without human-generated data, specifically leveraging distillation from OpenAI’s o1-preview.
   - *Developers can replicate these improvements* using the new [distillation product](https://openai.com/index/api-model-distillation/) announced at DevDay.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AIatMeta/status/1842188252541043075">Tweet from AI at Meta (@AIatMeta)</a>: 🎥 Today we’re premiering Meta Movie Gen: the most advanced media foundation models to-date.  Developed by AI research teams at Meta, Movie Gen delivers state-of-the-art results across a range of capa...</li><li><a href="https://x.com/nickaturley/status/1842281132265484595">Tweet from Nick Turley (@nickaturley)</a>: One of my favorite things about building Canvas: we used novel synthetic data generation techniques, such as distilling outputs from OpenAI’s o1-preview, to fine-tune the GPT-4o to open canvas, make t...</li><li><a href="https://arxiv.org/abs/2410.02089">RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning</a>: Large language models (LLMs) deployed as agents solve user-specified tasks over multiple steps while keeping the required manual engagement to a minimum. Crucially, such LLMs need to ground their gene...</li><li><a href="https://x.com/ahmad_al_dahle/status/1842032577164804571?s=46">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: Looking forward to tomorrow … 👀</li><li><a href="https://fxtwitter.com/xlr8harder/status/1842199810763370742">Tweet from xlr8harder (@xlr8harder)</a>: One of the coolest thing in Meta&#39;s Movie Gen paper is that Meta built modeling tools to optimize the layout of the model during training, which enabled them to use a complex and highly optimized p...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

natolambert: Should I make this a real poster at a conference?
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1291585301059342386)** (4 messages): 

> - `Permuting vs Reshaping Tensors`
> - `Stable Diffusion Model Training`
> - `Tinygrad CI Warnings`
> - `Analysis of CI Test Failures` 


- **Permuting vs Reshaping Tensors for Targets**: A member inquired whether to `.permute` or `.reshape` a target tensor sized (1024,1,14,1) to match the required shape of (14,1024,1). This discussion highlights the nuances of tensor manipulation in deep learning frameworks.
   - *Dumb q.* suggests a level of frustration or confusion surrounding this tensor transformation issue.
- **Training Stable Diffusion on M3 MacBook Air**: A member asked about the existence of models that can be trained for **stable diffusion** within **48 hours** on a standard **M3 MacBook Air**. This inquiry reflects growing interest in training efficiency on consumer hardware.
   - The question signals a need for accessible resources and guidance for efficient model training.
- **Exploring Tinygrad CI Warnings**: A call was made for individuals interested in analyzing the {warnings during the test run](https://github.com/tinygrad/tinygrad/actions/runs/11177982687/job/31074623873?pr=6880) of Tinygrad. This insight can help refine the stability and reliability of the framework.
   - The linked CI run showcases recent changes, including node cleanup and local **metal test speeds** enhancements.
- **Historical Analysis of CI Test Failures**: A user expressed interest in a comprehensive analysis of tests that have **failed** in historical **CI runs** as well as those that have **never failed**. Such analysis could provide valuable insights into test reliability and code stability.
   - This request suggests a proactive approach to improving continuous integration processes.



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/actions/runs/11177982687/job/31074623873?pr=6880">node cleanup + local metal test speed [pr] · tinygrad/tinygrad@2a8b305</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - node cleanup + local metal test speed [pr] · tinygrad/tinygrad@2a8b305

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1291607734948593727)** (2 messages): 

> - `bfloat16 tests`
> - `Triton talks` 


- **Call for More bfloat16 Tests in Tinygrad**: George highlighted the need for **more bfloat16 tests** in tinygrad during a recent discussion, referencing the limited existing tests in `test_dtype.py`.
   - One member questioned what *additional tests* would be beneficial for enhancing the testing framework.
- **Insightful Triton Talks Available**: A member shared a link to a **Triton talk** on YouTube, discussing various aspects and developments related to Triton technology.
   - The talk can be viewed [here](https://www.youtube.com/watch?v=ONrKkI7KhU4) for anyone interested in exploring Triton's capabilities.


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

leoandlibe: Hey guys, does torchtune support KTO training?~
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1291484591583924265)** (5 messages): 

> - `VinePPO`
> - `Flex Attention`
> - `Batch Size Optimization`
> - `Distributed Data Parallel (DDP)` 


- **VinePPO revolutionizes RL for LLM Reasoning**: A member highlighted that **VinePPO**, a modification to PPO, shows significant improvements over RL-free methods and standard PPO, achieving results with up to **9x fewer steps**, **3x less time**, and **half the memory**.
   - This prompts a rethink of **RL post-training**, as noted in the discussion thread.
- **Flex Attention achieves improved runtime performance**: A member discussed that **Flex Attention** should maintain similar runtime when processing batches of concatenated samples due to the **block sparsity** of the attention mask.
   - Another member confirmed that testing shows **bsz=1** with **1000 tokens** performs equally in time and memory as **bsz=2** with **500 tokens** each.
- **Exploration of batch size in packed runs**: A member suggested potentially removing the batch size option when utilizing packed setups to streamline processing, advocating for either batch size or **tokens_per_pack** for a consistent **bs=1**.
   - This raises questions about efficiency and the impact on performance metrics.
- **Discussion on implementing DDP**: There is speculation about incorporating **Distributed Data Parallel (DDP)**, where each sampler is set to **bsz=1**, optimizing for single device usage.
   - This approach could enhance resource allocation and performance across devices.



**Link mentioned**: <a href="https://x.com/a_kazemnejad/status/1841888338816455033/photo/1">Tweet from Amirhossein Kazemnejad (@a_kazemnejad)</a>: VinePPO, a straightforward modification to PPO, unlocks RL’s true potential for LLM Reasoning.  It beats RL-free methods (DPO and RestEM) and PPO, surpassing it in less steps(up to 9x), less time(up t...

  

---



### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1291546985664614451)** (4 messages): 

> - `Network Speed Improvements`
> - `Software Limitations`
> - `100 Gbps Technology`
> - `Latency vs Throughput`
> - `AI Contributions to Networking` 


- **AI boosts network speeds while software lags**: Members discussed how **AI advancements** have led to **100 Gbps** becoming cheaper than ever, with **1.6 Tbps** currently in labs.
   - *Darkmatter* pointed out that software has not kept pace with the **80x bandwidth increase**, leading to **issues** at even **10 Gbps**.
- **Urgency to enhance network capabilities**: *Luanon404* expressed enthusiasm for the improvements, stating, *'it's time to speed up the network.'*
   - This sentiment reflects a broader concern about achieving optimal **throughput** and **latency** in current network environments.


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1291868166791630909)** (1 messages): 

> - `axolotl packaging`
> - `dependency management` 


- **Exploring Alternatives to pip for axolotl**: A member raised the issue of finding **installing/updating dependencies** in **axolotl** frustrating and inquired about using non-pip packagers like **uv** as an alternative.
   - They expressed curiosity about any ongoing efforts and ways they could contribute to making the experience smoother.
- **Community Engagement in axolotl Development**: The same member highlighted their willingness to help improve the **axolotl** library by exploring different packaging options.
   - This move aims to encourage other developers to join in and alleviate common frustrations with dependency management.


  

---



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

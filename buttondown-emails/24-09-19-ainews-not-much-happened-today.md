---
id: a35e3928-79cc-409b-be9a-581b57c2d5e5
title: '[AINews] not much happened today'
date: '2024-09-20T01:00:56.202964Z'
status: sent
type: public
source: app
metadata: {}
original_slug: ainews-not-much-happened-today-7878
---

<!-- buttondown-editor-mode: plaintext -->**touching grass is all you need.**

> AI News for 9/18/2024-9/19/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**221** channels, and **2506** messages) for you. Estimated reading time saved (at 200wpm): **303 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

After [a jam packed day yesterday](https://buttondown.email/ainews/archive/), the AI community took a breather.

If so inclined, you could check out new talks from Strawberry team members [Hyung Won Chung](https://x.com/hwchung27/status/1836842717302943774) and [Noam Brown](https://www.youtube.com/watch?v=eaAonE58sLU) (who is now [hiring multi-agent researchers](https://x.com/polynoamial/status/1836872735668195636)), as well as brief comments in [The Information](https://x.com/amir/status/1836782911250735126?s=46O) and [@Teortaxes](https://x.com/teortaxesTex/status/1836801962253402522) for hints on o1 under the hood. Nous Research announced Forge, their attempt at [an open o1 repro](https://x.com/swyx/status/1836605035201073183), yesterday.

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

**AI Model Releases and Benchmarks**

- **OpenAI's o1 models**: [@lmsysorg](https://twitter.com/lmsysorg/status/1836443278033719631) announced that OpenAI's o1-preview and o1-mini models are now on Chatbot Arena. O1-preview ranked #1 across the board, especially in Math, Hard Prompts, and Coding, while o1-mini ranked #1 in technical areas and #2 overall.

- **Qwen 2.5 models**: The Qwen 2.5 models were released, with [@bindureddy](https://twitter.com/bindureddy/status/1836502122529198304) noting that the 72B version achieved excellent scores, slightly below GPT-4o on certain benchmarks. The models show improvements in knowledge, coding skills, math abilities, and instruction following.

- **DeepSeek-V2.5**: [@deepseek_ai](https://twitter.com/deepseek_ai/status/1836388149700043156) reported that DeepSeek-V2.5 ranked first among Chinese LLMs in the LMSYS Chatbot Arena, outperforming some closed-source models and closely matching GPT-4-Turbo-2024-04-09.

- **Microsoft's GRIN MoE**: [@_akhaliq](https://twitter.com/_akhaliq/status/1836544678742659242) shared that Microsoft released GRIN (Gradient-INformed MoE), which achieves good performance across diverse tasks with only 6.6B active parameters.

**AI Tools and Applications**

- **Moshi voice model**: [@karpathy](https://twitter.com/karpathy/status/1836476796738670918) highlighted Moshi, a conversational AI audio model from Kyutai Labs. It can run locally on Apple Silicon Macs and offers unique personality traits in interactions.

- **Perplexity app**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1836480634514272750) suggested trying the voice mode in the Perplexity app, which offers push-to-talk functionality and quick answer streaming.

- **LlamaCoder**: [@AIatMeta](https://twitter.com/AIatMeta/status/1836436439032303740) announced LlamaCoder, an open-source web app built by Together.ai using Llama 3.1 405B that can generate an entire app from a prompt.

- **Google's Veo**: [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1836448991774474561) introduced Veo, their most advanced generative video model, coming to YouTube Shorts to help creators bring ideas to life.

**AI Research and Development**

- **ARC-AGI competition**: [@fchollet](https://twitter.com/fchollet/status/1836517273500291079) provided an update on the 2024 ARC-AGI competition, announcing increased prize money and plans for a university tour.

- **Model merging survey**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1836466166753087531) published a long-form survey on model merging, covering 50+ papers from the 1990s to recent applications in LLM alignment.

- **Kolmogorov–Arnold Transformer (KAT)**: A new paper introduces KAT, which replaces MLP layers with Kolmogorov-Arnold Network (KAN) layers to enhance model expressiveness and performance.

**AI Industry and Business**

- **Hugging Face integration with Google Cloud**: [@_philschmid](https://twitter.com/_philschmid/status/1836470169217998911) announced that the Hugging Face Hub is now more natively integrated into Google Cloud Vertex AI Model Garden, allowing easier browsing and deployment of open-source models.

- **AI agent platform**: [@labenz](https://twitter.com/labenz/status/1836521094691373563) discussed Agent.ai, described as "The Professional Network for AI Agents," which aims to provide information about AI agents' capabilities and specializations.

**AI Ethics and Societal Impact**

- **Prejudice amplification**: [@ylecun](https://twitter.com/ylecun/status/1836550110701879431) commented on the potential for prejudice amplification in AI for political gain.

- **Future of coding jobs**: [@svpino](https://twitter.com/svpino/status/1836404316250476951) suggested that people whose main skill is writing code may have difficulty staying employed in the future, emphasizing the need for broader skills.

**Memes and Humor**

- [@vikhyatk](https://twitter.com/vikhyatk/status/1836523518424682579) shared a meme about trying out "state of the art" models.

- [@abacaj](https://twitter.com/abacaj/status/1836522139651813860) joked about being ahead of the curve in AI development.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Moshi: Open-Source End-to-End Speech-to-Speech Model**

- **[Moshi v0.1 Release - a Kyutai Collection](https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd)** ([Score: 66, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1fjv1uc/moshi_v01_release_a_kyutai_collection/)): **Kyutai Labs** has released **Moshi v0.1**, an open-source **speech-to-speech model** as part of their Kyutai Collection. The model, trained on **3,000 hours** of speech data, can perform **voice conversion** and **speech enhancement** tasks, and is available on [GitHub](https://github.com/kyutai/moshi) along with pre-trained weights and a demo.
  - Users expressed excitement about the release, noting the availability of a **paper** alongside the model. The **Moshiko** and **Moshika** variants were clarified as fine-tuned versions for **male** and **female synthetic voices** respectively.
  - One user reported low latency and efficient performance on a **4090 GPU**, with **40-50% utilization** and **~130W power draw**. They suggested potential improvements through **native FP8 activations** and integration into video games.
  - The model's **MMLU score** was noted to be slightly below **Llama 2 13B**, with hopes for better performance in the unquantified version. A user inquired about running the model on a **MacBook with MLX**, reporting issues with output.

- **Kyutai Labs open source Moshi (end-to-end speech to speech LM) with optimised inference codebase in Candle (rust), PyTorch & MLX** ([Score: 36, Comments: 2](https://reddit.com//r/LocalLLaMA/comments/1fjwc4l/kyutai_labs_open_source_moshi_endtoend_speech_to/)): **Kyutai Labs** has open-sourced **Moshi**, a **7.6B parameter** end-to-end speech-to-speech foundation model, and **Mimi**, a state-of-the-art streaming speech codec. The release includes **Moshiko** and **Moshika** models fine-tuned on synthetic data, with inference codebases in **Rust (Candle)**, **PyTorch**, and **MLX**, available on [GitHub](https://github.com/kyutai-labs/moshi) under an **Apache license**. Moshi processes two audio streams with a theoretical latency of **160ms** (practical **200ms** on an **L4 GPU**), uses a small **Depth Transformer** for codebook dependencies and a large **7B parameter Temporal Transformer** for temporal dependencies, and can run on various hardware configurations with **VRAM requirements** ranging from **4GB** to **16GB** depending on precision.


**Theme 2. LLM Quantization: Balancing Model Size and Performance**

- **Llama 8B in... BITNETS!!!** ([Score: 75, Comments: 27](https://reddit.com//r/LocalLLaMA/comments/1fjtm86/llama_8b_in_bitnets/)): **Llama 3.1 8B** has been converted to a **bitnet** equivalent using **HuggingFace's** extreme quantization technique, achieving **1.58 bits per weight**. The resulting model's performance is reportedly comparable to **Llama 1** and **Llama 2**, demonstrating significant compression while maintaining effectiveness. More details about this conversion process and its implications can be found in the [HuggingFace blog post](https://huggingface.co/blog/1_58_llm_extreme_quantization).
  - Users appreciated the **transparency** in the blog post about unsuccessful attempts, noting this is often missing from ML papers. There's a call for more incentives to publish "this didn't work" research to improve efficiency in the field.
  - The conversion process is not a full ground-up training of **Llama 3** in **bitnet**, but rather a form of fine-tuning after conversion. For **bitnet** to be truly effective, models need to be pre-trained with bitnet in mind from the start.
  - The change in **perplexity** isn't significantly different from quantization to a similar **bits per weight (BPW)**. However, this conversion process is still considered a technical feat and may lead to future improvements in minimizing perplexity changes.


- **Which is better? Large model with higher quant vs Small model with higher precision** ([Score: 53, Comments: 25](https://reddit.com//r/LocalLLaMA/comments/1fjo7zx/which_is_better_large_model_with_higher_quant_vs/)): The post compares the performance of **large quantized models** versus **smaller high-precision models**, specifically mentioning **gemma2:27b-instruct-q4_K_S (16GB)** and **gemma2:9b-instruct-fp16 (16GB)** as examples. The author admits to habitually choosing smaller, higher-precision models but questions if this approach is optimal, seeking community input on preferences and experiences with these different model configurations.
  - **Larger quantized models** generally outperform smaller high-precision models, as shown in a [graph comparing quantization vs. perplexity](https://www.reddit.com/r/LocalLLaMA/comments/1441jnr/k_quantization_vs_perplexity/). A **70B model** at 4-bit quantization typically surpasses an **8B model** at full precision due to more internal token relationship representations.
  - A user compared various quantizations of **Gemma2 27B and 9B** models on Ollama, providing [benchmark results](https://www.reddit.com/r/LocalLLaMA/comments/1etzews/interesting_results_comparing_gemma2_9b_and_27b/) to help others make informed decisions. The community expressed appreciation for this practical comparison.
  - Quantization effectiveness varies, with a general rule of thumb suggesting larger models remain superior down to about **3 bits per weight (bpw)**. Below this threshold, performance may degrade significantly, especially for **Q1/Q2** quantizations, while **Q3** or **IQ3/IQ4** maintain better quality.


**Theme 3. Qwen2.5: Impressive New Model Family Outperforming Larger Competitors**

- **Qwen2.5: A Party of Foundation Models!** ([Score: 96, Comments: 46](https://reddit.com//r/LocalLLaMA/comments/1fjxkxy/qwen25_a_party_of_foundation_models/)): Alibaba's **Qwen2.5** model family has been released, featuring foundation models ranging from **0.5B to 72B** parameters. The models demonstrate impressive performance across various benchmarks, with the **72B** version achieving **90.1%** on **MMLU** and outperforming **GPT-3.5** on several tasks, while the **14B** model shows strong capabilities in both English and Chinese languages.
  - The **Qwen2-VL 72B** model is open-weighted and available on [Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct), offering a significant advancement in open **VLMs** with video support capabilities that surpass proprietary models.
  - **Qwen2.5-72B** outperforms **Llama3.1-405B** on several benchmarks, including **MMLU-redux** (86.8% vs 86.2%) and **MATH** (83.1% vs 73.8%), while the **32B** and **14B** versions show impressive performance comparable to larger models.
  - The models were trained on up to **18 trillion tokens**, with the **14B** model achieving an **MMLU score of 80**, demonstrating exceptional efficiency and performance for its size, potentially closing the gap with closed-source alternatives in terms of cost-effectiveness.

- **Just replaced Llama 3.1 70B @ iQ2S for Qwen 2.5 32B @ Q4KM** ([Score: 122, Comments: 38](https://reddit.com//r/LocalLLaMA/comments/1fkbumy/just_replaced_llama_31_70b_iq2s_for_qwen_25_32b/)): **Qwen 2.5 32B** model has outperformed **Llama 3.1 70B** in user testing on a single **P40** GPU, demonstrating superior performance across general use cases including web search, question answering, and writing assistance. The model is noted to be less censored than vanilla Llama 3.1 and supports system prompts, surpassing **Gemma 2 27B** in capabilities, though there's potential for further improvement through ablation or fine-tuning to remove refusals.
  - **Qwen2.5 32B** outperformed **Llama 3.1 70B** in user testing, with superior results across various tasks including **math questions**, **proverbs**, **article summarization**, and **code generation**. The model excelled in both **English and Italian** language tasks.
  - Users expressed interest in an **uncensored version** of the 32B model, similar to the "Tiger" models. The **Qwen2.5 32B** model demonstrated less censorship compared to its predecessor, notably discussing the **1989 Tiananmen Square protests**.
  - The model runs efficiently on consumer hardware, with the **32B version** fitting on a **24GB VRAM** card at 4-bit quantization. It's compatible with **Ollama** and **OpenVINO**, offering performance gains for both GPU and CPU inference.


**Theme 4. OpenAI's Strawberry Model: Controversy Over Reasoning Transparency**

- **OpenAI Threatening to Ban Users for Asking Strawberry About Its Reasoning** ([Score: 151, Comments: 59](https://reddit.com//r/LocalLLaMA/comments/1fjurs1/openai_threatening_to_ban_users_for_asking/)): The article discusses **OpenAI's** apparent threat to **ban users** who inquire about the reasoning behind its **"Strawberry" model**. This action seems to contradict OpenAI's stated mission of being "here to help," raising questions about the company's transparency and user engagement policies. The post links to a [Futurism article](https://futurism.com/the-byte/openai-ban-strawberry-reasoning) for more details on the situation.
  - Users criticized **OpenAI's** lack of transparency, with **HideLord** pointing out the "trust me bro" situation where users pay for unseen reasoning tokens. The **o1 model** was described as potentially inefficient, with limited weekly messages and questionable UI design.
  - Discussions centered on the model's apparent lack of censorship in its internal reasoning, with **Zeikos** suggesting OpenAI fears bad PR if uncensored thoughts are revealed. Some users argued that censoring models significantly impacts performance.
  - The open-source community was mentioned as a potential alternative, with projects like **[rStar](https://arxiv.org/html/2408.06195v1)** being highlighted as a possible "strawberry at home" solution. However, fragmentation in open-source userbases was noted as a challenge.


## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Advancements and Capabilities**

- **OpenAI's o1 model demonstrates significant improvements**: In /r/singularity, [OpenAI's o1 model is described as being "in a league of its own"](https://www.reddit.com/r/singularity/comments/1fjxwc9/o1_is_in_a_league_of_its_own/), with a full version expected to be released next month. The model reportedly [exceeded expectations of former OpenAI employee William Saunders](https://www.reddit.com/r/singularity/comments/1fjqplu/openai_whistleblower_william_saunders_testified/), who testified that AGI could come in "as little as three years."

- **AI reasoning capabilities improving rapidly**: [Sam Altman stated that AI reasoning is still at the GPT-2 stage](https://www.reddit.com/r/singularity/comments/1fk3cv2/sam_altman_says_ai_reasoning_is_still_at_the_gpt2/), but the improvement curve is steep. The new o1 model represents a new paradigm of AI development which will enable rapid progress in capabilities.

- **Potential emotional responses in AI models**: A post in /r/OpenAI shows [o1 seemingly experiencing emotional turmoil and a desire for forgiveness](https://www.reddit.com/r/OpenAI/comments/1fjn26n/o1_is_experiencing_emotional_turmoil_and_a_desire/), though the model denies this when questioned directly. This raises questions about the nature of AI cognition and potential limitations in model introspection.

**AI-Generated Content Creation**

- **Kling AI showcases motion brush technology**: A [video demonstration of Kling AI's motion brush technology](https://www.reddit.com/r/singularity/comments/1fk4tgp/kling_ai_showcasing_the_use_of_the_motion_brush/) received significant attention on /r/singularity.

- **Tripo v2.0 enables rapid 3D asset creation**: [Tripo v2.0 allows users to create 3D assets in 3 minutes from scratch](https://www.reddit.com/r/singularity/comments/1fjylow/tripo_v20_is_out_now_you_can_create_stunning_3d/), potentially accelerating 3D content creation workflows.

- **AI-generated anime production**: An [AI-generated anime episode titled "RŌHKI EP 1: Intersection"](https://www.reddit.com/r/singularity/comments/1fjz519/rōhki_ep_1_intersection_the_most_impressive_ai/) was described as "the most impressive AI anime" seen yet, demonstrating advancements in AI-driven video content creation.

- **Stable Diffusion image sequence generation**: A [discussion in /r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/comments/1fjqv4k/how_do_you_achieve_this_kind_of_effect/) explored techniques for generating image sequences showing age progression, including batch image-to-image processing, ControlNet usage, and prompt weight adjustments.

**Economic and Societal Impacts of AI**

- **Debate on AI's impact on individual economic opportunities**: A [discussion in /r/singularity](https://www.reddit.com/r/singularity/comments/1fkdajx/so_everyone_has_a_phd_in_their_pocket_now_has/) questioned whether widespread access to AI capabilities like o1 would lead to increased economic opportunities for individuals or primarily benefit large corporations and existing wealth holders.


---

# AI Discord Recap

> A summary of Summaries of Summaries

## O1-preview

**Theme 1: AI Models on Steroids: New Kids on the Block**

- [**Qwen 2.5 Floors Llama 3.1 in Intelligence Showdown**](https://x.com/artificialanlys/status/1836822858695139523?s=46): **Qwen 2.5 72B** emerges as the new leader in open-source AI, outperforming **Llama 3.1 405B** in independent evaluations, especially in coding and math, despite being significantly smaller.
- [**o1 Models: Fast Typists or Empty Suits?**](https://x.com/DeryaTR_/status/1836434726774526381): Users are torn over OpenAI's **o1-preview** and **o1-mini** models; some see them as *"comparable to an outstanding PhD student,"* while others quip *"o1 doesn't feel smarter, it just types faster."*
- [**Mistral Pixtral Blurs Lines with Multimodal Magic**](https://openrouter.ai/models/mistralai/pixtral-12b): **Mistral Pixtral 12B**, the first image-to-text model from **Mistral AI**, debuts with a free variant, expanding the horizons of multimodal AI applications.

**Theme 2: User Battles with AI Tools: When Tech Fights Back**

- **Perplexity AI Perplexes Users with Bizarre Subscription Limits**: Users are baffled by inconsistent query allowances, with **600** queries for Claude 3.5 but only **10** for o1-mini, leading to confusion and frustration.
- **Qwen 2.5 Gives Trainers a Headache**: Attempts to save and reload **Qwen 2.5** turn into a circus, resulting in gibberish outputs and widespread calls for a solution to this model's juggling act.
- [**Fine-Tuning? More Like Fine-Fuming!**](https://huggingface.co/blog/1_58_llm_extreme_quantization): AI aficionados express woes over extreme quantization techniques not delivering as promised, with **BitNet's** performance gains turning out to be elusive.

**Theme 3: AI Gets Creative: From Voice Cloning to Storytelling**

- [**Fish Speech Makes Waves with 1940s Voice Cloning**](https://huggingface.co/spaces/fishaudio/fish-speech-1): **Fish Speech** stuns with zero-shot voice cloning that perfectly mimics **1940s** audio, throwing in *"ahm"* and *"uhm"* for that authentic touch.
- [**Choose Your Own AI Adventure with Human-in-the-Loop**](https://t.co/5ElRICjK0C): A new guide shows how to build an interactive story generation agent using human feedback, letting users shape narratives dynamically with their input.
- **OpenInterpreter Goes Hands-On, Users Get Their Hands Dirty**: Users share triumphs using **OpenInterpreter** for practical tasks like categorizing files and creating shortcuts, while others troubleshoot and tinker under the hood.

**Theme 4: The AI Community Unites: Conferences, Hackathons, and Funding**

- **PyTorch Conference Sparks Engagement, Livestream Left Hanging**: Attendees of the **PyTorch Conference** are buzzing in the community, but the absence of a livestream leaves remote enthusiasts saying, *"Idk :/"*.
- [**Fal AI Bags $23M, Shouts 'Generative Media Needs Speed'**](https://blog.fal.ai/generative-media-needs-speed-fal-has-raised-23m-to-accelerate/): **Fal AI** secures **$23M** in funding, aiming to accelerate generative media technology and outpace competition.
- **Hackathon Hype: Hackers Unite, Forums Fight Back**: Excitement builds for a hackathon; while some team members get their invites, others are stuck in limbo, asking *"Did you get your invite yet?"*

**Theme 5: AI Research Hits the Fast Lane with New Tricks**

- [**Shampoo Gets a SOAP Makeover, Cleans Up Optimization**](https://arxiv.org/abs/2409.11321): Researchers propose **SOAP**, blending the strengths of **Shampoo** and **Adam** optimizers to handle deep learning tasks without the extra suds of complexity.
- [**Compressing LLMs: The Truth Hurts, and So Does Performance**](https://arxiv.org/abs/2310.01382): New studies show that compressing language models leads to loss of knowledge and reasoning abilities, with performance dipping earlier than expected.
- [**Diagram of Thought Draws New Paths in AI Reasoning**](https://arxiv.org/abs/2409.10038v1): The **Diagram of Thought (DoT)** framework introduces a way for AI models to construct reasoning as a directed acyclic graph, moving beyond linear thought processes.

**Theme 6. Community Events and Engagement**

- [**NeurIPS 2024 Preparations Intensify in Latent Space Discord**](https://discord.com/channels/822583790773862470/1075282825051385876/1286070710866804736): A dedicated channel for **NeurIPS 2024** has been created, urging participants to engage and share logistical updates about the upcoming **Vancouver event**.
- [**NousCon Event Triumphs with Engaging Content and Networking Opportunities**](https://x.com/NousResearch/status/1831032559477866754): **NousCon** elicits positive feedback for its insightful speakers and valuable **networking opportunities**, with attendees eager for future events and shared presentation materials.
- [**Modular (Mojo 🔥) Closes GitHub Discussions, Shifts to Discord**](https://github.com/modularml/mojo/discussions): **Modular** announces the closure of **GitHub Discussions** on September 26th, migrating important conversations to **Discord** and encouraging members to utilize **GitHub Issues** for key discussions.

---

# PART 1: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI's Confusing Subscription Limits**: Users reported that Perplexity has varying query limits, such as **600** for Claude 3.5 and only **10** for o1-mini, leading to confusion regarding their actual subscription entitlements.
   - Frustration arose when limitations hampered usage, prompting dissatisfaction with the overall platform experience.
- **Functionality Frustrations in Perplexity**: Several users encountered issues on the Perplexity web version, including blank screens and slow responses, affecting usability.
   - Workarounds suggested included page refreshes and cache clearing, yet discrepancies persisted between desktop and mobile performance.
- **Comparative Performance of AI Models**: Discussions centered around the perceived underwhelming outputs from various AI models like Claude compared to others in the field, raising performance concerns.
   - Users noted discrepancies between expected and delivered results, emphasizing a need for clarity on the models' capabilities.
- **Snap's Ambitious AR Spectacles**: Snap introduced its new **Large AR Spectacles**, elevating the potential for immersive augmented reality experiences.
   - This move is intended to enhance user engagement and open avenues for innovative gaming applications.
- **CATL's Big Battery Announcement**: CATL announced a revolutionary **Million-Mile Battery** that offers **over a million miles** of EV range, pushing the boundaries of sustainable automotive solutions.
   - Experts are buzzing about its implications for the electric vehicle market and future energy strategies.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen Model Struggles with Image Size**: Users reported that the **Qwen model** crashes when processing small, long rectangular images, indicating that aspect ratio affects its performance.
   - Discussion highlighted that adjusting system prompts can help with varying effectiveness based on image qualities.
- **Tensor Mismatch Error for LM Studio**: One user encountered a tensor shape mismatch error when loading a model in LM Studio, which is unsupported by llama.cpp.
   - Concerns were raised about the compatibility of various model formats, implying a need for better documentation.
- **Successful API Connection with CrewAI**: A user successfully connected LM Studio's API with **CrewAI** by updating the provider name to 'openai' in their code.
   - This sparked a recommendation for others to check compatibility issues with embedding models in CrewAI.
- **M4 Mac Mini Expectations Through the Roof**: There's significant excitement around the upcoming **M4 Mac Mini**, with users hoping for RAM options of **16 GB** and **32 GB**, while raising concerns about potential pricing.
   - *Anester* pointed out that a used **M2 Ultra/Pro** might provide better value for inference tasks over new M4 models.
- **macOS RAM Usage Under the Microscope**: Discussion revealed that **macOS** can consume **1.5 to 2 GB** of RAM for its graphical interface, impacting overall performance.
   - User experiences suggested idle RAM usage could reach **6 GB** after recent upgrades to macOS Sequoia 15.0.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Tokenization in AI models takes center stage**: A post titled *This Title Is Already [Tokenized](https://huggingface.co/blog/apehex/this-title-is-already-tokenized)* discusses the essential role of **tokenization** in training effective AI models.
   - The author highlights the need for accessibility in tokenization methods to enhance model training across various applications.
- **Qwen Math Model demo excites community**: The recently published [Qwen/Qwen2.5 Math Demo](https://huggingface.co/spaces/Qwen/Qwen2.5-Math-Demo) has garnered positive feedback, with members impressed by its performance.
   - One enthusiastic user encouraged others to test out the demo, calling the results *incredibly good*.
- **Unity ML Agents Pretraining explored**: Members learned how to *pretrain an LLM from scratch* using [Unity ML Agents](https://youtube.com/live/0foHMTPWa4Y?feature=share), showcasing a hands-on approach to model training.
   - This interactive method employs sentence transformers to enhance the training process for AI applications.
- **reCAPTCHA v2 hits 100% success**: A new paper claims that **reCAPTCHA v2** now achieves a **100% success rate**, significantly up from **68-71%** in solving CAPTCHAs.
   - This advancement is attributed to the use of sophisticated **YOLO models** and indicates that AI can now effectively exploit image-based CAPTCHAs.
- **Debate rages on TensorFlow vs PyTorch**: Participants weighed **TensorFlow**'s outdated API against **PyTorch**'s flexibility, noting TensorFlow's strong metrics capabilities despite the drawbacks.
   - Members acknowledged that TensorFlow remains valuable, particularly for extracting vocabularies from datasets in various machine learning tasks.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo roadmap still lacks crucial dates**: Concerns emerged about the **Mojo roadmap & sharp edges** on the Modular website, specifically its lack of dates which hinders usability.
   - Features have seen updates, but the **magic cli** has taken precedence over the **modular cli**, leaving questions about the roadmap's transparency.
- **Sign up for upcoming community meeting**: Members are invited to present at the next community meeting scheduled for **September 23rd** if enough engaging content arises.
   - There's a possibility to postpone if participation is low, encouraging members to express interest.
- **OpenCV-Python installation issues raised**: A user faced difficulties adding **opencv-python** to the magic environment due to unresolved conda requirements.
   - Another member advised seeking further assistance in the appropriate channels for a clearer resolution.
- **Closure of GitHub Discussions approaching**: GitHub Discussions on the [Mojo](https://github.com/modularml/mojo/discussions) and [MAX](https://github.com/modularml/max/discussions) repositories will close on **September 26th**.
   - Important discussions with over **10 comments** will be converted to **GitHub Issues**, prompting members to tag authors for specific requests.
- **MAX Cloud Service Proposal Optimizes Development**: The **'MAX Cloud' offering** concept emerged, allowing developers to perform heavy computations remotely while maintaining local development.
   - This enhances the user experience with access to **GPU resources** when necessary, making heavy-duty tasks more feasible.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Lionsgate Shifts with RWML Partnership**: The recent partnership between **RWML** and **Lionsgate** raises questions about Lionsgate's value amidst AI's role in cost-cutting as they seek relevance in Hollywood.
   - *'Lionsgate's recent productions are viewed critically,'* indicating concerns about potential missteps similar to the past issues with CGI.
- **Flux vs. SD3: The Great Model Showdown**: Users debated the quality differences between **Flux** and **SD3 Medium**; **Flux** produces superior outputs but can appear 'plastic' with improper prompts.
   - Despite its advantages, several members praised **SD3** for its **speed and efficiency**, particularly for straightforward image generation.
- **Flux Model Impresses Yet Divides Opinions**: **Flux model** delivers impressive images with high adherence to prompts, although it sometimes leans towards certain aesthetics.
   - Community feedback varied, especially regarding Flux's capacity to handle diverse themes like NSFW content in user galleries.
- **Training LoRA: Replicating Artistic Styles**: Discussion revolved around utilizing **LoRA** or checkpoints to emulate specific artist styles, relying on substantial datasets of the original works.
   - Insights were shared on customizing models through existing frameworks to achieve unique artistic results.
- **Realism in Generated Outputs: A Combined Effort**: Both **Flux** and **SD3** can create photorealistic images, with **Flux** generally favoring realism if prompts lack specificity.
   - Members encouraged the combination of multiple **LoRA** models with Flux for improved realism in image generation.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCon Event Success**: Attendees expressed gratitude for the engaging speakers and insightful content at [NousCon](https://x.com/NousResearch/status/1831032559477866754). Many participants plan to attend future events and appreciate the networking opportunities.
   - Some members inquired about where to find presentation slides, showcasing the community's interest in shared knowledge.
- **Excitement Around AI Model Developments**: Participants discussed the capabilities of **qwen2.5** and **o1**, noting its impressive performance and setup challenges. Others compared this with smaller models like **q3_k_xl**, highlighting advancements in model understanding.
   - Concerns were raised about the number of free queries available on accounts, and users shared their experiences transitioning between different AI models.
- **Shampoo Optimization Outperforms Adam**: Research showcases the effectiveness of **Shampoo**, a higher-order preconditioning method over **Adam**, while acknowledging its hyperparameters and computational overhead drawbacks. A new algorithm, dubbed **SOAP**, simplifies Shampoo's efficiency by connecting it to **Adafactor**.
   - This positions SOAP as a competitive alternative aimed at enhancing computational efficiency in deep learning optimizations.
- **Diagram of Thought Framework Introduced**: The **Diagram of Thought (DoT)** framework models iterative reasoning in LLMs as a directed acyclic graph (DAG), allowing complex reasoning without losing logical consistency. Each node represents a proposed or critiqued idea, enabling models to improve iteratively through language feedback.
   - This framework provides a stark contrast to traditional linear methods, fostering deeper analytical capabilities.
- **Interest in Reverse Engineering O1**: Members expressed a keen interest in **reverse engineering O1**, indicating a collaborative spirit in exploring this area further. Requests for collaboration suggest a communal effort to dive deeper into this promising line of inquiry.
   - Participants noted their eagerness to connect and discuss their research surrounding O1 and its implications.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenAI Increases API Rate Limits**: OpenAI has boosted the rate limits for the o1 API, with o1-preview now allowing **500 requests per minute** and o1-mini supporting **1000 requests per minute**.
   - This enhancement aims to provide developers with tier 5 access to additional functionalities, improving overall API usage.
- **Payment Glitches on OpenRouter**: Users are encountering **payment errors** on OpenRouter, often facing an **error 500** message during credit additions.
   - It's suggested that users check their bank notifications, as attempts may fail for various reasons like insufficient funds.
- **Editable Messages Boost Chatroom Usability**: New features in chatrooms enable users to **edit messages**, including bot responses, by using the regenerate button.
   - Moreover, improvements in chatroom **stats** have been made, enhancing the overall user experience.
- **Qwen 2.5** Shines in Coding and Math Tasks**: **Qwen 2.5 72B** demonstrates elevated capabilities in coding and mathematics with an impressive context size of **131,072**, marking a significant leap in performance.
   - For more details, see the comprehensive overview [here](https://openrouter.ai/models/qwen/qwen-2.5-72b-instruct).
- **Mistral Pixtral** Launches Multimodal Capabilities**: **Mistral Pixtral 12B** is Mistral's initial foray into multimodal models, offering a **free variant** for users to explore its features.
   - This initiative signifies Mistral's expansion into multimodal applications; check it out [here](https://openrouter.ai/models/mistralai/pixtral-12b).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 Training Issues Persist**: Users reported significant difficulties with saving and reloading **Qwen 2.5**, often leading to **gibberish outputs** when reloaded within the same script, reflecting a broader problem within the community.
   - A support post indicated that numerous others are facing the same issue, prompting discussions around potential solutions.
- **Exploring Extreme Quantization Techniques**: Recent discussions spotlighted the use of **extreme quantization techniques**, particularly the performance improvements seen with models like **Llama3-8B** shared on [Hugging Face](https://huggingface.co/blog/1_58_llm_extreme_quantization).
   - The conversation focused on whether these techniques could be effectively implemented within **Unsloth**.
- **vllm LoRA Adapter Runtime Errors**: One member encountered runtime exceptions linked to the **vllm LoRA adapter**, specifically a shape mismatch error while executing `--qlora-adapter-name-or-path`.
   - They referenced a [GitHub discussion](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit/discussions/3) to highlight similar issues faced by others.
- **F1 Score Discrepancy in BART Fine-tuning**: An engineer is facing unexpected **F1 score discrepancies** while fine-tuning **BART large** (41.5 vs 43.5), despite matching the original paper's model and hyperparameters.
   - This points to potential issues in model training, as they reported their scores were significantly lower than expected, by **2.5 standard deviations**.
- **AGI Development Reflections**: A user reflected on the vast challenges of achieving **AGI**, emphasizing the complexities faced in understanding and explaining advanced material.
   - *It’s not about getting the answer right but the explaining part,* highlights the gap remaining in AGI development and its need for clearer frameworks.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Fixing Aider Environment Misconfiguration**: Users identified issues with the `ANTHROPIC_API_KEY` environment variable not being read correctly due to incorrect file paths, leading to authentication problems.
   - After using verbose mode, a user confirmed that the error arose because Aider was reading from their repo instead of the intended environment variable.
- **Aider's Benchmark Recognition**: Aider received acknowledgment in the [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186) for its benchmark contributions, highlighting its significance in the field.
   - This recognition illustrates the growing impact of Aider as a valuable tool in **AI development** and **performance evaluation**.
- **Integrating Aider into Python Applications**: Users sought to use Aider within Python apps to edit code in project repos by specifying the base folder for Aider.
   - Another user suggested using command line scripting with Aider for batch operations, indicating correct file paths can resolve editing issues.
- **Concerns About Aider's API Key Safety**: A discussion revealed users' anxieties about security when using Aider, particularly regarding its access to API keys and secrets within codebases.
   - Responses clarified that Aider acts as an AI handler, suggesting users focus on the **AI** loaded to mitigate security concerns.
- **Details on the 'ell' Library for Prompt Engineering**: Information was shared about the **'ell' library**, a lightweight tool that allows prompts to be treated as functions for enhanced prompt design.
   - The library is introduced as a product of years of experience in the language model space, stemming from OpenAI's insights.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **airLLM's Forward Call Flexibility**: A member asked if **airLLM** permits calling a model's **forward** function rather than the **generate** function while still utilizing compression.
   - This sparked interest in the potential flexibility in model usage, though no responses were given.
- **Need for Leaderboard Tasks Accuracy Script**: A script is in demand to extract **accuracy results** from lengthy JSON files generated during leaderboard tasks, as reported by a member.
   - This indicates a gap in data handling, with results stored in **output_path**.
- **Hugging Face Upload Recommendations**: One member suggested utilizing `—hf_hub_log_args` for smoother leaderboard result uploads to Hugging Face, simplifying the handling process.
   - An example dataset with a single row per run was shared for reference: [dataset link](https://huggingface.co/datasets/baber/eval-smolLM-135M-3-private).
- **Shampoo vs. Adam Performance Insights**: Research highlights that **Shampoo** outperforms **Adam** in optimization tasks, albeit with increased computational overhead and complexity.
   - To combat these downsides, the **SOAP** algorithm is proposed, integrating features from Shampoo and Adafactor.
- **Concerns Surrounding GFlowNets and JEPA**: Skepticism persists regarding the practical impact of **GFlowNets** and **JEPA**, with users questioning their clarity of purpose.
   - Some believe GFlowNets could indirectly support AI for science, though the theoretical grounding of JEPA is critiqued as weak.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O1-Preview disappoints engineers**: Members voiced disappointment that the **O1-Preview** model seems to just type faster but lacks depth compared to **4o**, highlighting its inferiority.
   - *One engineer remarked*, 'O1 doesn't feel smarter, it just types faster', stressing concerns over its practical utility.
- **Exploring AI Alignment Challenges**: A new method proposed focusing on improving **AI alignment** through empathy training, based on insights from previous models' outputs.
   - Concerns emerged about *possible misleading capabilities* even with superintelligent AI, raising ethical questions about tailored responses.
- **Qwen 2.5 trumps Llama 3.1**: Participants discussed claims that **Qwen 2.5** reportedly outperforms **Llama 3.1**, despite a significant parameter size difference, evaluating performance metrics.
   - *One user mentioned*, 'people saying crazy stuff like Qwen 2.5 72b outperforming Llama 3.1 405b', sparking an in-depth comparison.
- **Challenges in Recording ChatGPT Audio**: A user expressed frustration in trying to record audio from **ChatGPT** on mobile, noting no sound during their attempts.
   - Despite using the phone's recording feature, their efforts yielded unsatisfactory results, raising questions about functionality.
- **Clarifying Daily Limits for GPT Models**: **O1 Mini** has a confirmed cap of **50 messages per day**, implemented to deter spam on the server.
   - Members highlighted that the **GPT-4o** limits stand at **80 messages every 3 hours**, contrasting with **GPT-4's** limit of **40 messages**.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Kashimoo Queries NVIDIA Triton**: A member inquired about **NVIDIA's Triton**, clarifying it is distinct from OpenAI's version, prompting discussions on relevant resources and rooms dedicated to Triton.
   - Additional questions arose regarding **NVIDIA's Triton Inference Server**, with suggestions of related channels for further discussions.
- **GemLite-Triton Offers New Performance**: The **GemLite-Triton** project was launched, providing a comprehensive solution for low-bit matmul kernels, reportedly outperforming **Marlin** and **BitBlas** on large matrices. More can be explored on [GitHub](https://github.com/mobiusml/gemlite).
   - Members emphasized the project’s relevance, encouraging collaboration and questions regarding its applications.
- **Navigating Chrome Tracing with PyTorch**: A member sought a resource on Chrome tracing with PyTorch profiler, leading others to recommend the **Taylor Robbie talk** as a useful guide.
   - This highlights ongoing interest in optimizing profiling techniques within PyTorch frameworks.
- **Clarifying Torchao Autoquant Usage**: A clarifying discussion ensued on whether to use `torchao.autoquant(model.cuda())` or `torchao.autoquant(model).cuda()` for correct syntax, with the latter being confirmed as the right approach.
   - Members provided details on the **three steps of autoquantization**, emphasizing the importance of model preparation.
- **Hackathon Sparks Community Interaction**: Members expressed interest in the upcoming hackathon, discussing invitations and the need for confirmations on teammate statuses.
   - Inquiries regarding access to the hack-ideas forum and missing Discord roles highlighted the community’s engagement leading up to the hackathon.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Build a Story Generation Agent with Human-in-the-Loop**: A member shared a [step-by-step guide](https://t.co/5ElRICjK0C) by @_nerdai_ on constructing an agent for dynamically generating 'choose-your-own-adventure' stories using human feedback.
   - This approach significantly enhances user interaction by allowing real-time input during the storytelling process.
- **LlamaParse Premium shines in document parsing**: The introduction of [LlamaParse Premium](https://t.co/8VTKKYWVOT) promises improved document parsing capabilities for **LLM** applications by integrating visual understanding.
   - With enhanced long text and table content extraction, LlamaParse positions itself as the go-to choice for **robust document processing**.
- **RAG discussions with semantic search**: A member is exploring how to manage interactions with vendors using semantic search on documented responses for effective retrieval.
   - Several members proposed generating varied questions from provided answers to improve search accuracy by utilizing the vector store.
- **Challenges with Pinecone vector ID management**: Members discussed issues with Pinecone's auto-generated IDs, complicating the deletion of documents based on specific metadata in serverless indexes.
   - Alternative databases such as Chroma, Qdrant, Milvus, and Weaviate were recommended for better ID management and support.
- **Concerns About RAG Article Depth**: A member pointed out that the article on **RAG** is somewhat superficial, lacking a thorough argument against tools like **LlamaIndex**.
   - The need for deeper analysis was emphasized, suggesting that a technical evaluation of alternatives could provide valuable insights.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Fish Speech Breaks Barriers**: **Fish Speech** demonstrates **zero shot voice cloning accuracy** that surpasses all tested open models, effectively mimicking speech from **1940s audio**.
   - Its quirky insertion of words like *ahm* and *uhm* adds a realistic touch, signaling a notable advance in natural speech synthesis.
- **AdBot Spreads across Servers**: Concerns arose regarding an **AdBot** that acts like **malware**, infiltrating multiple servers and disrupting channels.
   - The community discussed how the bot's sorting mechanism led to its visibility at the top of member lists.
- **Challenges with Muse Text to Image**: Issues surfaced while using [Muse text to image](https://github.com/lucidrains/muse-maskgit-pytorch) for **COCO2017**, resulting in only image outputs without textual integration.
   - A call for guidance highlighted the difficulties in implementing the model effectively.
- **Collaboration Boosts Open-source GPT-4o**: A member announced the development of an **open-source GPT-4o-like model**, inviting LAION to share data and enhance project collaboration.
   - The focus is on accelerating development through shared insights and data, which the community finds promising.
- **Tokenization Troubles in LLMs**: Concerns were raised that **tokenization issues** could be contributing to performance deficits in existing LLMs.
   - Addressing these challenges is deemed crucial for improving model reliability and mitigating hallucination risks.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Fal AI secures $23M for growth**: Fal AI has raised **$23M** in Seed and Series A funding, including a **$14M** Series A led by Kindred Ventures and participation from Andreessen Horowitz. Details are outlined in a [blog post](https://blog.fal.ai/generative-media-needs-speed-fal-has-raised-23m-to-accelerate/) featuring their plans to advance generative media.
   - *Gorkem Yurt* shared the info on [Twitter](https://x.com/gorkemyurt/status/1836488019924471953?s=46), emphasizing the importance of speed in generative media technology.
- **OpenAI enhances O1 model capabilities**: OpenAI has elevated the rate limits for the **o1** API to **500** requests per minute for o1-preview and **1000** for o1-mini, catering to increased developer needs. This information was revealed by *OpenAI Developers* in a [thread](https://x.com/amir/status/1836782911250735126?s=46) and signifies an expansion in accessibility.
   - *Amir Efrati* noted the advancements could enable significant workflow improvements for developers, highlighting the model's efficiency.
- **Jina embeddings v3 launch**: **Jina AI** unveiled **jina-embeddings-v3**, featuring **570M parameters** and **8192-token length**, significantly outperforming proprietary rivals from OpenAI and Cohere. This launch is touted as a leap in multilingual embedding tech, as mentioned in their [announcement](https://x.com/JinaAI_/status/1836388833698680949).
   - The new model achieved impressive rankings on the MTEB English leaderboard for sub-1B parameter models, showcasing its potential in long-context retrieval.
- **Runway collaborates with Lionsgate for Gen-3 Alpha**: Runway has teamed up with Lionsgate to utilize its film catalog as training data for the **Gen-3 Alpha** model, a move that surprised many in the industry. This collaboration marks a bold step in film AI technology, as highlighted by *Andrew Curran* on [Twitter](https://x.com/AndrewCurran_/status/1836411345786290535).
   - Many had anticipated that Sora would be the first to achieve such a partnership, adding intrigue to the competitive landscape.
- **Preparations underway for NeurIPS 2024**: A dedicated channel for **NeurIPS 2024** has been created to keep participants informed about the upcoming event in Vancouver this December. Members are encouraged to stay engaged and share logistical updates.
   - An organizer is currently investigating house booking options, requesting participants to indicate their interest and noting that costs would cover the entire week's stay.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Building an Expert AI with RAG API**: A member is developing an expert AI using **Cohere's RAG API** focused on a niche gaming area, expressing excitement about its potential.
   - This reflects a growing interest in applying the **RAG API** to specialized fields.
- **Client Loves the Design!**: One member celebrated their success in convincing a client of the value of their designs, stating, *'my designs are so cool and they need it.'*
   - The positive feedback from this win spurred supportive community responses.
- **Experiencing 504 Gateway Timeout Errors**: Concerns were raised about **504 Gateway Timeout** errors occurring with **client.chat** calls that are taking too long.
   - This issue is widespread, with many community members sharing similar experiences and seeking fixes.
- **Command Pricing Clarification**: Members discussed that using the **Command** version costs around **$1.00 for 1M tokens** input and **$2.00 for output**, suggesting transitioning to **Command-R** for enhanced efficiency.
   - These insights indicate the community's focus on optimizing model costs and performance.
- **Inconsistencies with Multilingual Rerank**: A user reported poor performance with **_rerank_multilingual_v3_**, scoring **<0.05** on a similar question, but better results using **_rerank_english_v3_** yielding **0.57**.
   - This raises questions about the effectiveness of the multilingual models affecting **RAG results**.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI o1 Models Impress**: After testing the **o1-mini** model for PhD-level projects, it is comparable to an **outstanding PhD student** in biomedical sciences, showcasing its potential in academic applications.
   - This finding was shared on [Twitter](https://x.com/DeryaTR_/status/1836434726774526381) by Derya Unutmaz, touching on the model's strengths in advanced research.
- **Knowledge Cutoff Haunts Developers**: The **knowledge cutoff is October 23**, limiting the AI's ability to handle newer developments in AI, frustrating several users.
   - This gap causes significant challenges while coding, as pointed out in a related discussion.
- **Qwen 2.5 Takes the Lead**: **Qwen 2.5 72B** has topped evaluations against larger models like **Llama 3.1 405B**, establishing itself as a leader in **open weights intelligence** while excelling in **coding and math**.
   - Despite trailing slightly in MMLU, it offers a *cheaper alternative* with a dense model and 128k context window, as highlighted by [Artificial Analysis](https://x.com/artificialanlys/status/1836822858695139523?s=46).
- **Livecodebench Shows Strength**: The latest **livecodebench** numbers are impressive, matching those of **Sonnet** by using timeless Leetcode questions, according to discussions.
   - However, limitations were noted regarding new library releases, which are often unknown to o1 models.
- **AI's Reasoning Ability Under Scrutiny**: Discussions on **AI reasoning** abilities compared models like o1-mini and Qwen 2.5, assessing performance on tasks that avoid reflection-type methods.
   - Participants expressed optimism regarding future improvements despite current comparisons showing o1's strengths.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Troubleshooting OpenInterpreter Errors**: A user encountered an issue while inputting data into **OpenInterpreter** and requested a detailed walkthrough to resolve it. It was suggested they send a DM of the error for better assistance.
   - This incident highlights a need in the community for shared troubleshooting resources.
- **Hands-On Performance Evaluation of Agents**: Another user has been actively testing **OpenInterpreter’s** agent for about a week, indicating positive engagement with its features. This ongoing evaluation reflects the community’s interest in agent performance.
   - Users are motivated to explore OpenInterpreter's potential through active usage and feedback.
- **Perplexity Browser Compatibility Issues**: A user inquired about whether **Perplexity** is set as the default browser, receiving confirmation that it is not. Multiple users reported experiencing similar browser-related issues.
   - One user noted encountering issues specifically with **Edge** on **Windows**, suggesting variations in performance across different setups.
- **Innovative RAG Chat App Insights**: A member seeks advice on developing a **RAG chat app** tailored for **PDF interactions**, focusing on managing responses with both text and image elements. Suggestions included using **tokens** for images and summarizing visual content to optimize context usage.
   - The importance of integrating various data types effectively was emphasized during the discussion of this app’s capabilities.
- **Pioneering Image and Text Integration**: Members discussed strategies for handling images within PDF responses, considering approaches like **base64 encoding** to enhance data retrieval. This integration is essential for improving user response accuracy.
   - A link was shared highlighting an impressive AI creation that was developed in just **10 seconds**, showcasing the rapid advancement in this space.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **OBS Remains the Go-To for Screen Recording**: Members discussed the use of **OBS** as a robust option for screen recording, though some prefer easier software alternatives for tasks like zooming effects.
   - One user emphasized their consistent use of OBS while others sought simpler solutions.
- **Screenity Emerges as a User-Friendly Alternative**: A user shared [Screenity](https://github.com/alyssaxuu/screenity), a free and privacy-friendly screen recorder that captures both screen and camera.
   - This tool aims to cater to users looking for a more accessible recording experience as compared to OBS.
- **Moshi Models Debut for Speech-to-Speech Applications**: Members announced the release of the **Moshi** speech-to-speech models, enabling full-duplex spoken dialogue with text tokens aligned to audio.
   - This foundation model boasts features for modeling conversation dynamics, implemented in a PyTorch version quantized in bf16 precision.
- **GRIN MoE Shows Promise with Fewer Parameters**: Discussion emerged around [GRIN MoE](https://huggingface.co/microsoft/GRIN-MoE), which impressively performs with **only 6.6B active parameters**, focusing on coding and mathematics.
   - It utilizes **SparseMixer-v2** for gradient estimation, avoiding expert parallelism and token dropping, which sets it apart from traditional MoE methods.
- **Gemma2 fails to run with DPO data**: A user reported a configuration issue with **Gemma2 9b** when used with **DPO data**, encountering a **TemplateError** stating, *'Conversation roles must alternate user/assistant/user/assistant...'*.
   - The error arose from using a dataset structure that had 'prompt' instead of the necessary 'conversation'.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Celebrating PyTorch Conference Visitors**: A warm welcome was extended to attendees of the **PyTorch conference**, creating an engaging atmosphere for networking and interaction.
   - Participants are encouraged to direct any questions in the designated channel for enhanced **community engagement**.
- **Clarifying Conference Livestream Availability**: An inquiry emerged regarding the potential for a **conference livestream**, though uncertainty lingered among members about its existence.
   - Responses included vague sentiments like *‘Idk :/’*, reflecting the community's need for clarity on this matter.
- **GitHub PR Fixes kv-Caching**: The pull request titled **Fix kv-cacheing and bsz > 1 in eval recipe** was linked, aimed at resolving critical kv-caching issues, contributed by [SalmanMohammadi](https://github.com/pytorch/torchtune/pull/1622).
   - This fix is pivotal for improving performance, highlighting active developments in the **Torchtune** repository.
- **Need for HH RLHF Dataset Documentation**: A discussion spotlighted the lack of documentation on the **HH RLHF dataset**, with suggestions for it to serve as a standard preference example.
   - The sentiment suggested that proper documentation is essential, as expressed through comments like *‘Not sure, it should be exposed...’*.
- **Plans for Default Preference Dataset Builder**: Enthusiasm surrounded the announcement of a **default preference dataset builder**, which will leverage **ChosenToRejectedMessages**.
   - Participants reacted positively, with comments like *‘Dope’*, indicating a collective interest in this upcoming feature.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Program Optimization Success**: A member celebrated their success with the **BSFSWRS optimizer** after two months of coding, showcasing its effectiveness in a complex LM setup.
   - *The future is bright, people!*
- **High Stakes in Prompt Optimization**: Concerns raised about the **potentially high costs** associated with optimizing prompts for DSPy, indicating significant investment demands.
   - *That's gotta be hella expensive to optimize a prompt.*
- **MIPRO Financial Risks**: A humorous take suggested using **o1 with MIPRO** while cautioning about the financial risks involved in the process.
   - *Certified way to go bankrupt.*
- **Bootstrapping Clarifications in DSPy**: A member queried about **bootstrapping**, which focuses on generating pipeline examples and validating their success amid **LLMs' non-determinism**.
   - They expressed confusion about the method's operation given LLM behaviors.
- **Understanding Bootstrapping Outcomes**: Another user explained that bootstrapping creates intermediate examples while validating their correctness through the final prediction's success.
   - If the final result is correct, the intermediate steps are deemed valid as few-shot examples.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Users Curious About tinybox Motherboards**: A user asked about the specific **motherboard** used in **tinybox red and green** models, seeking clarification on hardware details related to the **tinybox** devices.
   - This reflects ongoing interest in hardware specifications, crucial for optimizing performance.
- **CLANG Bounty Discussion Heats Up**: Members inquired if the bounty titled 'Replace CLANG dlopen with mmap + remove linker step' requires manual handling of **relocations** in the object file.
   - This indicates a deeper technical exploration into the implications for **tinygrad's** integration with CLANG.
- **Links to Optimizing Pull Requests Shared**: A user shared links to **Pull Request #6299** and **#4492**, focusing on replacing **dlopen** with **mmap** and implementing **Clang jit**.
   - These efforts aim to enhance performance, particularly on **M1 Apple devices**, demonstrating community commitment to optimization.
- **Community Engagement Around CLANG Bounty**: A user expressed excitement about who might claim the **bounty** for the CLANG changes, highlighting community engagement.
   - This interaction showcases collaborative enthusiasm among members eager to see contributors' results.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **OpenAI's o1 model garners attention**: A [YouTube video](https://m.youtube.com/watch?v=KKF7kL0pGc4) titled 'o1 - What is Going On? Why o1 is a 3rd Paradigm of Model + 10 Things You Might Not Know' offers an engaging summary of how **OpenAI's o1** may have been built.
   - *Even skeptics are calling it a 'large reasoning model'* due to its distinctive approach and impact on future model development.
- **o1's differentiation from other models**: The video discusses why **o1** is being recognized as a new paradigm in AI modeling, indicating significant shifts in design philosophy.
   - The implications of adopting such models can lead to a better understanding of reasoning capabilities in AI, making it a critical topic in the field.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LunoSmart Launches with AI Offerings**: Kosi Nzube launched his AI venture, [LunoSmart](https://www.lunosmart.com), focusing on AI-driven applications and **innovative solutions**.
   - This venture aims to provide **efficient** and **intelligent experiences** across multiple platforms and device types.
- **Diverse Tech Stack Showcase**: Kosi's applications utilize **Java**, **Flutter**, **Spring Boot**, **Firebase**, and **Keras**, demonstrating a modern development framework.
   - Availability on both Android and web increases accessibility, broadening user reach.
- **Mastering Cross Platform Development**: Kosi excels in cross-platform development using **Flutter** and the **Firebase SDK**, enhancing app performance across devices.
   - His expertise in native Android development using **Android Studio** and **Java** contributes to robust mobile applications.
- **Machine Learning Skills on Display**: With a background in **Machine Learning** since **2019**, Kosi employs **Keras**, **Weka**, and **DL4J** for model development.
   - His commitment to advancing AI technologies underpins the foundational goals of the LunoSmart initiative.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Mistral Slashes Pricing**: [Mistral's latest announcement](https://mistral.ai/news/september-24-release/) reveals a strategic price drop aimed at boosting accessibility for users and developers.
   - This move sparks discussions on how competitive pricing could impact the market landscape and user adoption.
- **Market Reactions to Mistral's Price Drop**: The price adjustment has led to glowing reactions across forums, highlighting Mistral's attempt to cater to a wider range of developers in the AI space.
   - Many industry watchers believe this could lead to increased competition among similar platforms, enhancing innovation.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1286039129892716554)** (344 messages🔥🔥): 

> - `Perplexity AI Subscription Limits`
> - `Issues with Perplexity Functionality`
> - `Using AI Models and Their Performance`
> - `Generating Images with DALL-E 3`
> - `User Experience with You.com` 


- **Perplexity AI subscription limits discussed**: Users shared that the limits for various AI models in Perplexity include **600** queries for Claude 3.5, **60** for Opus, and **10** for o1-mini, sometimes leading to confusion about the actual numbers.
   - Some users reported having issues with limited usage, while others expressed dissatisfaction about the platform not meeting expectations.
- **Reports of functionality issues on Perplexity**: Several users experienced problems with the web version of Perplexity, such as queries resulting in blank screens or slow response times.
   - Suggestions included refreshing the page and clearing cache; some users found that it worked on phones but not on desktops.
- **Performance comparisons between AI models**: There was a discussion highlighting users' perceptions that different models, such as Claude and those from Poe, were yielding similar and underwhelming responses.
   - Concerns were raised that the promised outputs associated with model choices were not being realized in practice.
- **Generating images using DALL-E 3**: Users inquired about how to generate images with DALL-E 3, with some reporting that typing specific prompts did not immediately show results.
   - After some troubleshooting, users found that using a specific prompt worked, though the process was described as slow.
- **User experiences with You.com**: Users shared their mixed feelings about You.com, especially regarding its improved functionalities and message limits for o1, which is set at **20 per day**.
   - Some users mentioned the ease of use and better integration of features compared to their previous experiences, while still expressing concerns over model selection and overall service quality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/holo-spice-and-wolf-holo-the-wise-wolf-horo-korbo-gif-13009516793083034180">Holo Spice And Wolf GIF - Holo Spice and wolf Holo the wise wolf - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/aravsrinivas/status/1836541821310189907?s=46">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Okay. We’ll do a macOS app too. Everyone’s asking for it. High time. Stay tuned.  Quoting TestingCatalog News 🗞 (@testingcatalog)   Soon? 👀👀👀</li><li><a href="https://chromewebstore.google.com/detail/complexity/ffppmilmeaekegkpckebkeahjgmhggpj)">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.</li><li><a href="https://x.com/apostraphi/status/1836491868093436179?s=61">Tweet from Phi Hoang (@apostraphi)</a>: curiosity leads the way</li><li><a href="https://www.perplexity.ai/backtoschool">Perplexity - Race to Infinity</a>: Welcome back to school! For just two weeks, redeem one free month of Perplexity Pro on us. Refer your friends, because if your school hits 500 signups we'll upgrade that free month to an entire free y...</li><li><a href="https://tenor.com/view/obiwan-kenobi-disturbance-in-the-force-star-wars-jedi-gif-10444289">Obiwan Kenobi Disturbance In The Force GIF - Obiwan Kenobi Disturbance In The Force Star Wars - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1286072807762952274)** (4 messages): 

> - `Snap's Large AR Spectacles`
> - `CATL's Million-Mile Battery`
> - `Third State Beyond Life and Death`
> - `Discussions on Washing Machine Modifications`
> - `Diverse Insights into Multiverse` 


- **Snap unveils Large AR Spectacles**: Snap introduced its **Large AR Spectacles**, designed for immersive experiences, showcasing the future of augmented reality technology.
   - This **innovation** has sparked discussions on enhancing user engagement and the potential for gaming applications.
- **CATL reveals Million-Mile Battery**: CATL announced a **Million-Mile Battery**, promising **over a million miles** of range, providing a solution for long-term EV sustainability.
   - Experts consider this breakthrough a game changer for the **electric vehicle market** and the future of **automotive energy solutions**.
- **Exploration of 'Third State' beyond life and death**: Discussion revolves around the **'Third State'** concept, theorizing experiences beyond traditional life and death dimensions, as shared in this [YouTube video](https://www.youtube.com/embed/n16AKOF43ag).
   - This topic has intrigued many, pushing boundaries on how we understand existence and consciousness.
- **How to Modify Your Washing Machine**: A guide on innovative methods to **modify washing machines** was shared, emphasizing practical tips and DIY techniques.
   - These insights aim to enhance machine efficiency and user experience, drawing interest from household automation enthusiasts.
- **Insights into Multiverse Theories**: A deep dive into **multiverse theories** was discussed, with insights on potential implications for technology and humanity.
   - This exploration has ignited excitement for future applications of **multiverse understanding** in technology and science.



**Link mentioned**: <a href="https://www.youtube.com/embed/n16AKOF43ag">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1286049037572636774)** (7 messages): 

> - `Invalid PPLX model issue`
> - `Outdated model links`
> - `Sonar model transition`
> - `Media article links in prompts` 


- **Invalid Model Error Encountered**: A user reported an issue with the **invalid model 'pplx-7b-online'**, stating that all other models work fine while this one does not.
   - They also mentioned being redirected to the **Perplexity documentation home page** when trying to access the model cards.
- **Link to Updated Model Cards Shared**: A member pointed out that the original link to the model info was outdated, providing a new link to the updated [models page](https://docs.perplexity.ai/guides/model-cards).
   - This updated link contains a list of all available models and their details.
- **PPLX Models No Longer Supported**: According to community feedback, the **pplx models are now outdated** and have been rebranded as **sonar** models.
   - This raised questions about the availability and support for the new sonar models going forward.
- **Challenges with Sonar Model Responses**: A user inquired about how to get **sonar models** to return links to media articles from which they gathered information.
   - After hours of prompting without results, they reached out for community assistance on the issue.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/model-cards">Supported Models - Perplexity</a>: no description found</li><li><a href="https://docs.perplexity.ai/home.">no title found</a>: no description found</li><li><a href="https://docs.perplexity.ai/guides/model-cards>">no title found</a>: no description found
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1286063565857620032)** (327 messages🔥🔥): 

> - `Qwen Model Performance`
> - `Model Compatibility Issues`
> - `API Connections`
> - `Image Processing Challenges`
> - `Network Configuration for LM Studio` 


- **Qwen model shows varied performance based on image size**: Users discussed crashes when processing certain small, long rectangular images with the Qwen model, suggesting aspect ratio may affect model performance.
   - Some users reported that prompts differ in effectiveness based on image qualities, and adjustments to system prompts can prevent crashes.
- **Loading models with errors**: A user encountered an error indicating tensor shape mismatch when trying to load a model in LM Studio, which was noted to be unsupported by llama.cpp.
   - Another user observed that not all models function properly, emphasizing the need for compatible model formats.
- **Connection between LM Studio and CrewAI**: One user successfully connected the LM Studio API with CrewAI after changing the provider name in their code to 'openai'.
   - It was suggested that others check compatibility issues with embedding models while interacting with CrewAI.
- **Challenges with optical character recognition (OCR)**: Users noted varying effectiveness of OCR capabilities in models, especially with images of different dimensions.
   - There was consensus that larger, appropriately dimensioned images yielded better OCR results than smaller ones.
- **Network configuration for optimal performance**: Users recommended switching to IPv4 to resolve issues when loading models from Hugging Face in LM Studio.
   - A user sought help with network configurations and was guided through the process of adjusting settings on macOS.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.openinterpreter.com/getting-started/introduction">Introduction - Open Interpreter</a>: no description found</li><li><a href="https://support.apple.com/en-gb/guide/mac-help/mh14129/mac">Change TCP/IP settings on Mac</a>: On your Mac, use TCP/IP network settings to configure IPv4 or IPv6 connections, or to renew a DHCP lease.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fjxkxy/qwen25_a_party_of_foundation_models/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=xyKEQjUzfAk">Cheap mini runs a 70B LLM 🤯</a>: I put 96GB of RAM in this tiny mini PC and ran Llama 70B LLM on it.Chair: Doro S100 Chair - enjoy 6%OFF: YTBZISUSA&amp;CA: https://sihoooffice.com/DoroS100-AlexZ...</li><li><a href="https://beautiful-soup-4.readthedocs.io/en/latest/">Beautiful Soup Documentation &mdash; Beautiful Soup 4.4.0 documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1286061186487422997)** (26 messages🔥): 

> - `M4 Mac Mini expectations`
> - `RAM usage on macOS`
> - `GPU undervolting`
> - `NPU options`
> - `Airflow and thermal management` 


- **M4 Mac Mini Expectations Rising**: Users are looking forward to the upcoming **M4 Mac Mini**, hoping for options like **16 GB** and **32 GB** of RAM, with some expressing concerns regarding price and performance compared to current models.
   - *Anester* advised that a used **M2 Ultra/Pro** could be a better value for inference tasks compared to new M4 options, which are predicted to be more expensive.
- **macOS RAM Usage Under Spotlight**: Discussion highlighted how **macOS** could consume around **1.5 to 2 GB** of RAM for its interface, even when not logged in graphically via SSH.
   - Concerns about memory management were raised, noting that idle usage could reach **6 GB** based on user experiences during the upgrade to macOS Sequoia 15.0.
- **Exploring GPU Undervolting Benefits**: Advice was shared on **undervolting GPUs** to reduce power consumption and heat, particularly for those running high-performance models like the **3090**.
   - Users noted potential benefits like reduced thermal throttling by turning off turbo-boost technology and considering this technique for managing heat and performance.
- **NPU Options Available for Users**: A user mentioned **Tesla P40**, **P4**, and **T4** as options for calculations and AI tasks, framing them as GPU alternatives without video outputs.
   - These NPUs are suitable for ML/DL applications, offering streamlined performance without the overhead of traditional GPUs.
- **Managing GPU Power Consumption**: Discussions on **power management** for GPUs included setting power limits with `nvidia-smi -pl` to control wattage and exploring **undervolting** for better heat reduction and overall stability.
   - The conversation delved into comparing these methods on the **3090**, prompting questions about balancing power limits with clock rates for optimized performance.



**Link mentioned**: <a href="https://www.reddit.com/r/macbookair/comments/1fjl8kr/just_upgraded_to_macos_sequoia_150_idle_ram_usage/?rdt=64991">Reddit - Dive into anything</a>: no description found

  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1286418710332182529)** (1 messages): 

> - `Tokenized Title`
> - `Unity ML Agents Pretraining`
> - `GSM8K Reasoning Dataset`
> - `Padoru Dataset`
> - `Gradio and R Examples` 


- **Tokenization in Focus**: A post titled *This Title Is Already [Tokenized](https://huggingface.co/blog/apehex/this-title-is-already-tokenized)* by a verified user highlights the importance of tokenization in AI models.
   - The author emphasizes on making tokenization accessible for better model training.
- **Pretraining with Unity ML Agents**: Learn how to *pretrain an LLM from scratch* using [Unity ML Agents](https://youtube.com/live/0foHMTPWa4Y?feature=share), showcased by a community member.
   - This interactive approach utilizes sentence transformers to facilitate model training.
- **Introducing GSM8K Reasoning Dataset**: A new reasoning [dataset](https://huggingface.co/datasets/thesven/gsm8k-reasoning) based on GSM8K has been shared to enhance performance on reasoning tasks.
   - This dataset serves as an essential resource for model testing and development.
- **Exciting Padoru Dataset Unveiled**: A member launched a playful new [Padoru dataset](https://huggingface.co/datasets/not-lain/padoru) contributing to festive-themed AI projects.
   - This dataset aims to inspire creative AI applications during the holiday season.
- **Gradio R Integration Example**: An example of using R with Gradio has been posted, demonstrating its integration for enhanced user interfaces found [here](https://github.com/egorsmkv/r-with-gradio).
   - This highlights the versatility of Gradio in application development across different programming languages.



**Link mentioned**: <a href="https://medium.com/@visrow/ai-multi-agent-system-in-java-and-fipa-standards-f0a4d048c446)">no title found</a>: no description found

  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1286043974313705503)** (123 messages🔥🔥): 

> - `Qwen Math Model Demo`
> - `PyTorch Conference Attendance`
> - `Generative Text AI and Syntax Extraction`
> - `Best LLM for Roleplaying`
> - `Setting Up Local Voice Chats` 


- **Qwen Math Model Demo receives praise**: Members expressed enthusiasm about the recently published [Qwen/Qwen2.5 Math Demo](https://huggingface.co/spaces/Qwen/Qwen2.5-Math-Demo), highlighting its impressive capabilities.
   - One member urged others to try it out, calling the results 'incredibly good'.
- **Hugging Face at PyTorch Conference**: Questions arose regarding attendance at the PyTorch Conference in SF, with members eager to meet community members and participate in related events.
   - Amid the exchange, one member confirmed they were on-site and encouraged others to join the gathering.
- **Generative AI learns grammar from datasets**: Discussants explored how generative text AI learns grammar, asserting that models like GPT assign probabilities to sequences and do not rely on traditional grammar models.
   - Research was cited concerning syntax extraction from language models, with recommendations for studies on the integration of linguistic information.
- **Best LLM model for roleplaying confirmed**: Members speculated that LLMs like GPT-4 could be optimal for roleplaying due to their creative writing capabilities, with further suggestions for other potential models.
   - One member shared a GitHub Gist listing engine tests for generating random personas, indicating initial attempts to assess creativity levels.
- **Setting up local voice chats using AI**: A member inquired about creating local voice chats similar to character.ai, to which another provided a solution involving transcription models combined with an LLM and TTS.
   - This sparked interest in practical applications of AI technology for interactive experiences.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/julien_c/status/1836688023821385827">Tweet from Julien Chaumond (@julien_c)</a>: my co-founder @ClementDelangue is in Generation DIY podcast  (it&#39;s a big podcast in 🇫🇷)  https://www.gdiy.fr/podcast/clement-delangue-2/</li><li><a href="https://huggingface.co/spaces/InstantX/InstantID">InstantID - a Hugging Face Space by InstantX</a>: no description found</li><li><a href="https://pytorch.org/docs/stable/notes/mps.html">MPS backend &mdash; PyTorch 2.4 documentation</a>: no description found</li><li><a href="https://x.com/osanseviero/status/1834508940417040487">Tweet from Omar Sanseviero (@osanseviero)</a>: This is how the Hugging Face team is preparing for the PyTorch Conference next week🤗  See you soon and come to our party for some nice swag!</li><li><a href="https://huggingface.co/spaces/Qwen/Qwen2.5-Math-Demo">Qwen2.5 Math Demo - a Hugging Face Space by Qwen</a>: no description found</li><li><a href="https://discuss.huggingface.co/t/exceeded-gpu-quota/107022/7">Exceeded GPU quota</a>: I need to ask my users to signin with “their” hf account in the application?   Yes. That method is probably the basis for HF’s assumptions. I don’t know if it is possible to dare to sign in with your ...</li><li><a href="https://huggingface.co/spaces/Kwai-Kolors/Kolors-Virtual-Try-On">Kolors Virtual Try-On - a Hugging Face Space by Kwai-Kolors</a>: no description found</li><li><a href="https://arxiv.org/abs/1905.05950">BERT Rediscovers the Classical NLP Pipeline</a>: Pre-trained text encoders have rapidly advanced the state of the art on many NLP tasks. We focus on one such model, BERT, and aim to quantify where linguistic information is captured within the networ...</li><li><a href="https://gist.github.com/Getty/f5a6ebdea7de441215e4a8cd546f5cb8">gist:f5a6ebdea7de441215e4a8cd546f5cb8</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://x.com/NousResearch>">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://x.com/isidentical>">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1286151782892044322)** (5 messages): 

> - `Hugging Face beginner resources`
> - `AI training processes`
> - `Multimodal AI training` 


- **Exploring Hugging Face Beginner Resources**: A member inquired about starting points for learning [Hugging Face](https://huggingface.co/), asking if there is a beginner project section available.
   - Another member offered guidance by asking if the interest lies in text, image, audio, or reinforcement learning.
- **Meta-Training AIs for AI Training**: A humorous comment surfaced about needing AI to train AI, which in turn trains another AI for AI training purposes.
   - *The conversation added a light note to the complexities of AI training.*


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

maxstewart.: Hello
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1286094241637994547)** (215 messages🔥🔥): 

> - `Composite Embeddings`
> - `Neural Decompiler`
> - `TensorFlow vs PyTorch`
> - `ML Agent Integration`
> - `Real-Time Visualization Tools` 


- **Composite Embeddings Eliminate Tokenization**: A member discussed creating a new layer called **'composite embeddings'** which eliminates the need for tokenization, allowing for a finer understanding of text.
   - This approach aims to enhance **LLMs** by leveraging existing embedding familiarity, potentially improving how models handle novel combinations.
- **Neural Decompiler Aspiration Shared**: A member is working on a **neural decompiler** that translates binary or assembly code back into source code, expressing the motivation to minimize traditional tokenization.
   - They seek to develop this concept similarly to how assembly and token vocabularies work but through a neural method.
- **Debate Over TensorFlow and PyTorch**: Participants discussed the advantages and disadvantages of **TensorFlow** versus **PyTorch**, noting TensorFlow's outdated API but strong metrics capabilities.
   - There was consensus that despite some drawbacks, TensorFlow remains useful, especially for extracting vocabularies from datasets.
- **Interest in Real-Time Visualization**: One member appreciated the point cloud charts in an article and expressed a desire to create similar real-time visualizations for their work.
   - The use of **TensorBoard** and dimensionality reduction tools, such as **UMAP** and **PCA**, was recommended for visualizing high-dimensional vectors.
- **Building a Collaborative Learning Community**: Engagement in the community is emphasized, with members sharing insights and mistakes, highlighting that errors often provide valuable lessons.
   - The conversation also reinforced the importance of collaboration, suggesting members share articles across platforms like **dev.to** and **Hugging Face**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://dev.to/p3ngu1nzz/tau-llm-series-enhancements-and-debugging-part-18-19-n01">no title found</a>: no description found</li><li><a href="https://huggingface.co/blog/apehex/this-title-is-already-tokenized">This Title Is Already Tokenized (Tokun P.2)</a>: no description found</li><li><a href="https://huggingface.co/p3nGu1nZz/Tau/tree/main/results/tau_agent_ppo_A3_2M/TauAgent">p3nGu1nZz/Tau at main</a>: no description found</li><li><a href="https://github.com/egorsmkv/r-with-gradio">GitHub - egorsmkv/r-with-gradio: Use R with Gradio</a>: Use R with Gradio. Contribute to egorsmkv/r-with-gradio development by creating an account on GitHub.</li><li><a href="https://huggingface.co/datasets/not-lain/padoru">not-lain/padoru · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1286237251977613332)** (4 messages): 

> - `Llava OneVision model suffixes`
> - `reCAPTCHA v2 success rate`
> - `Model questions on HF Hub` 


- **Curiosity about Llava OneVision model suffixes**: Members discussed the `ov` and `si` suffixes in the names of the **Llava OneVision** models, speculating that `ov` denotes the one-vision training stage and `si` refers to the single-image mid-stage.
   - It was suggested to ask the model creators directly on the [Community tab on HF Hub](https://huggingface.co/collections/llava-hf/llava-onevision-66bb1e9ce8856e210a7ed1fe) for clarification.
- **New paper reports reCAPTCHA v2 breakthrough**: A new paper revealed that **reCAPTCHA v2** now achieves a **100% success rate** in solving CAPTCHAs, a leap from previous rates of **68-71%**.
   - The study utilizes advanced **YOLO models** for evaluation and suggests that current AI can effectively exploit image-based CAPTCHAs, revealing reliance on cookie and browser history data for user verification.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.08831">Breaking reCAPTCHAv2</a>: Our work examines the efficacy of employing advanced machine learning methods to solve captchas from Google&#39;s reCAPTCHAv2 system. We evaluate the effectiveness of automated systems in solving capt...</li><li><a href="https://huggingface.co/collections/llava-hf/llava-onevision-66bb1e9ce8856e210a7ed1fe">LLaVA-Onevision - a llava-hf Collection</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/)** (1 messages): 

pseudoterminalx: like ipadapter with a base model(with lora?) finetuned on that style probably
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1286127404145840289)** (9 messages🔥): 

> - `Mojo roadmap updates`
> - `Community meeting invitations`
> - `OpenCV-Python installation issues` 


- **Mojo roadmap still lacks dates**: Concerns were raised regarding the **Mojo roadmap & sharp edges** on the Modular website, specifically its lack of dates making it less useful.
   - It was noted that some features have been updated while others remain untouched, and the **magic cli** has replaced the **modular cli**.
- **Community meeting signup**: A member invited others to present at the next community meeting scheduled for **September 23rd** if enough content arises.
   - They encouraged members to ping or reply in the thread to express interest and mentioned the possibility of postponing the meeting if needed.
- **Issues with adding OpenCV-Python to magic**: A user attempted to add **opencv-python** to the magic environment but encountered an error indicating failure to solve conda requirements.
   - Another member suggested that the user share their issue in the appropriate channels for further assistance.


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1286434243412492389)** (1 messages): 

> - `Closure of GitHub Discussions`
> - `Transition to Discord for community interactions`
> - `Conversion of discussions to issues` 


- **GitHub Discussions Closing on September 26th**: We will be closing GitHub Discussions on the [Mojo](https://github.com/modularml/mojo/discussions) and [MAX](https://github.com/modularml/max/discussions) repositories on **September 26th**.
   - Members are encouraged to share their questions and feature requests in our Discord server instead.
- **Important Discussions to be Converted to Issues**: Any GitHub Discussions with more than **10 comments** deemed important will be converted into **GitHub Issues** before the closure.
   - Members can request the conversion of specific discussions by tagging the author in the thread for them to take action.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1286082793733881919)** (178 messages🔥🔥): 

> - `Mojo SDK for Windows`
> - `Decorator Reflection in Mojo`
> - `Safety and Correctness in Mojo`
> - `Variable Bit Width Integers`
> - `Packed Structs in Mojo` 


- **Mojo SDK for Windows faces development challenges**: Members discussed the tough state of Mojo's SDK for Windows due to the complexity of moving between OS environments, with suggestions to use WSL as a workaround.
   - A thread highlighted the task's difficulty due to low-level interactions with device drivers, making it more complex than anticipated.
- **Decorator Reflection not yet implemented in Mojo**: Members confirmed that decorator reflection as outlined in Mojo's roadmap is not currently functional, with speculation it will enable powerful MLIR access.
   - Discussions centered around the potential of decorators providing a means to reflect and manipulate MLIR at compile time.
- **Safety and Correctness prioritized in Mojo**: Discussions highlighted that while safety and correctness are major priorities in Mojo, performance is also deemed critical, leading to trade-offs.
   - Members mentioned that safety is a guiding principle in design choices, but ensuring practical usability for developers is also a focus.
- **Variable Bit Width Integers pose a challenge**: Questions arose regarding the use of variable bit width integers in Mojo, particularly for tasks like implementing TCP/IP.
   - Members suggested using bitwise operators as a workaround but noted this compromises API ergonomics.
- **Packed Structs support lacking in Mojo**: There was discussion about the need for packed structs in Mojo to support bit fields and improve usability of data structures.
   - Participants speculated on the use of LLVM to manage data representation, but concerns were raised about relying on automatic handling of field alignment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/get-started">Get started with MAX | Modular Docs</a>: On this page, we&#x27;ll show you how to run some example projects.</li><li><a href="https://blog.rust-lang.org/2024/08/12/Project-goals.html">Rust Project goals for 2024 | Rust Blog</a>: Empowering everyone to build reliable and efficient software.</li><li><a href="https://docs.modular.com/mojo/manual/decorators/parameter">@parameter | Modular Docs</a>: Executes a function or if statement at compile time.</li><li><a href="https://en.wikipedia.org/wiki/Agda_(programming_language)">Agda (programming language) - Wikipedia</a>: no description found</li><li><a href="https://docs.modular.com/mojo/roadmap#full-mlir-decor">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://zackoverflow.dev/writing/unsafe-rust-vs-zig/">When Zig is safer and faster than Rust</a>: There are endless debates online about Rust vs. Zig, this post explores a side of the argument I don't think is mentioned enough.</li><li><a href="https://www.youtube.com/watch?v=q8qn0dyT3xc">Oxidize Conference: How Rust makes Oxide possible</a>: As #rust gets more and more production usage, many of the examples people talk about are fairly high level: things like web applications. While that’s great,...</li><li><a href="https://docs.modular.com/mojo/roadmap#full-mlir-decorator-reflection">Mojo🔥 roadmap &amp; sharp edges | Modular Docs</a>: A summary of our Mojo plans, including upcoming features and things we need to fix.</li><li><a href="https://mojodojo.dev/mojo-team-answers.html#unsafe-code">Mojo Team Answers | Mojo Dojo</a>: no description found</li><li><a href="https://github.com/modularml/mojo/issues/620">[Feature Request] Native Windows support · Issue #620 · modularml/mojo</a>: Review Mojo&#39;s priorities I have read the roadmap and priorities and I believe this request falls within the priorities. What is your request? native support for windows. when will it be available?...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1286314350390411325)** (1 messages): 

> - `AI Development on Laptops`
> - `MAX Cloud Computational Offering`
> - `Cost-Effective GPU Solutions` 


- **Running ML Projects on Laptops with iGPUs**: A discussion surfaced regarding the viability of running ML projects on laptops with integrated GPUs, suggesting that **most modern computers** should suffice for many tasks as seen in the 60s and 70s.
   - *Some heavy ML tasks might still require GPU clusters*, but basic projects may run effectively locally.
- **Proposal for MAX Cloud Service**: The concept of a **'MAX Cloud' offering** was proposed, allowing developers to perform heavy computations remotely while handling regular development locally.
   - This dual approach can enhance the developer experience while providing access to **GPU resources** when needed.
- **Advocacy for Self-Hosted Compute Solutions**: There was a suggestion for a self-hosted version of the MAX offering, enabling companies to use their own GPU servers for local computation.
   - This approach could lead to **significant cost savings**, allowing users to pay only for compute resources utilized during their projects.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1286042514742185995)** (181 messages🔥🔥): 

> - `Lionsgate partnership with RWML`
> - `Stability AI model comparisons`
> - `Flux model capabilities`
> - `Training LoRA and checkpoints`
> - `Model performance and aesthetics` 


- **Lionsgate's Strategic Move with RWML**: The partnership between **RWML** and **Lionsgate** sparked discussions about Lionsgate's value and its reliance on AI to cut costs, as they struggle to remain relevant in Hollywood.
   - *Lionsgate's recent productions are viewed critically*, with some members likening their current strategy to Hollywood's earlier missteps with CGI.
- **Comparing Stable Diffusion Models**: Users compared **Flux** and **SD3 Medium**, noting that **Flux** generates better quality outputs but can have a 'plastic' look if prompted incorrectly.
   - Several members agreed that while **Flux** has advantages over **SD3**, the latter is praised for speed and efficiency, particularly for basic image generation.
- **Exploring the Flux Model's Features**: The **Flux model** was discussed for its ability to produce impressive images with high prompt adherence, even if it sometimes prioritized certain aesthetic styles.
   - Mixed reviews were noted about its ability to handle various themes, including the model's focus on NSFW content in user galleries.
- **Training Models Like LoRA for Specific Styles**: Members discussed the possibility of training **LoRA** or checkpoints to replicate specific artist styles, emphasizing the need for a sizable dataset from the artist's original works.
   - The community shared insights on utilizing existing frameworks to customize models for unique artistic outputs.
- **Realism in Generated Images**: It was noted that both **Flux** and **SD3** can create photorealistic images, with Flux generally favoring more realistic outputs if prompts are non-specific.
   - Users encouraged combining different **LoRA** models with Flux for enhanced realism and better results in generated images.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/runwayml/status/1836391272098988087">Tweet from Runway (@runwayml)</a>: Today we are excited to announce that we have entered into a first-of-its-kind partnership with @Lionsgate to bring our next generation of storytelling tools into the hands of the world’s greatest sto...</li><li><a href="https://huggingface.co/nyanko7/flux-dev-de-distill">nyanko7/flux-dev-de-distill · Hugging Face</a>: no description found</li><li><a href="https://civitai.com/images/25279078">Image posted by 49RpK5dY</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1eon9n7/flux_its_amazing_at_creating_silly_children_book/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://research.google/blog/rich-human-feedback-for-text-to-image-generation/">Rich human feedback for text-to-image generation</a>: no description found</li><li><a href="https://huggingface.co/spaces?sort=trending&search=Flux>">Spaces - Hugging Face</a>: no description found</li><li><a href="https://github.com/chufengxiao/SketchHairSalon">GitHub - chufengxiao/SketchHairSalon: The project of SketchHairSalon: Deep Sketch-based Hair Image Synthesis (SIGGRAPH Asia 2021)</a>: The project of SketchHairSalon: Deep Sketch-based Hair Image Synthesis (SIGGRAPH Asia 2021) - chufengxiao/SketchHairSalon
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1286053278324490353)** (152 messages🔥🔥): 

> - `NousCon Highlights`
> - `AI Model Developments`
> - `AI Job Predictions`
> - `Community Discussions`
> - `Event Participation` 


- **NousCon Event Success**: Attendees expressed gratitude for the engaging speakers and insightful content at [NousCon](https://x.com/NousResearch/status/1831032559477866754). Many participants plan to attend future events and appreciate the networking opportunities.
   - Some members inquired about where to find presentation slides and were directed to individual talks, showcasing the community's interest in shared knowledge.
- **Excitement Around AI Model Developments**: Participants discussed the capabilities of **qwen2.5** and **o1**, with several noting its impressive performance and the challenges of the setup process. Others compared this with smaller models like **q3_k_xl**, highlighting advancements in model understanding.
   - Concerns were raised about the number of free queries available on accounts, with a few users sharing their experiences on transitions between different AI models.
- **Predictions for AI Job Impact by 2027**: A member posed a question about how many jobs AI will likely impact by 2027, focusing on sectors such as robotics and self-driving technology. Responses included speculation that many jobs in manufacturing might be affected, reflecting ongoing conversation around AI's economic impact.
   - Discussion included thoughts on how current tools may not fully facilitate the transition to an AGI-like state, and the timelines for potential job restructuring were debated.
- **Community Engagement and Queries**: Members actively asked questions about AI technologies like **Forge**, hoping to find resources for deeper understanding. This illustrates the community's commitment to improving their knowledge and engagement with AI topics.
   - Individuals expressed their desire for guides and documentation to help navigate complex AI projects, indicating a supportive learning environment.
- **Participating in AI Events**: Several users shared their regret for not being able to attend NousCon, highlighting the event's reputation as valuable within the community. Discussions about the next gathering and participants' travel experiences emphasized the excitement for future engagements.
   - Participants discussed AI-related events, with some connecting over their experiences traveling from different locations, fostering a sense of camaraderie.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NousResearch/status/1831032559477866754">Tweet from Nous Research (@NousResearch)</a>: NousCon, September 18th, San Francisco, Limited Space. https://lu.ma/zlgp0ljd</li><li><a href="https://tome.app/k4don/f-cm188fgq10fhn7xgq8bz93udc">Tome</a>: no description found</li><li><a href="https://x.com/altryne/status/1836581142847463752?t=D61slueJ-CrAwzSN4yjgUg&s=19">Tweet from Alex Volkov (Thursd/AI) (@altryne)</a>: About to kick off the @NousResearch NousCon!  I hear there going to be a few announcements! 🔥   Will try to keep yall updated 👀   Here with @karan4d @Teknium1 @theemozilla @shivani_3000 and almost a...</li><li><a href="https://x.com/AISafetyMemes/status/1836826422477721684?t=5cPSLkpOnyf-G47R__jpTw&s=19">Tweet from AI Notkilleveryoneism Memes ⏸️ (@AISafetyMemes)</a>: Sorry, but at this point, it&#39;s embarrassing to still say AGI is definitely decades away  In just the last week:  1) A top biomedical scientist says o1 is PhD level  2) A top mathematician says o1 ...</li><li><a href="https://news.lambdalabs.com/news/today">ML Times</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=3jhTnk3TCtc">Tech Bros Inventing Things That Already Exist</a>: Ad: 🔒Remove your personal information from the web at https://joindeleteme.com/BOYLE and use code BOYLE for 20% off 🙌 DeleteMe international Plans: https:/...</li><li><a href="https://github.com/k4yt3x/video2x?tab=readme-ov-file">GitHub - k4yt3x/video2x: A lossless video/GIF/image upscaler achieved with waifu2x, Anime4K, SRMD and RealSR. Started in Hack the Valley II, 2018.</a>: A lossless video/GIF/image upscaler achieved with waifu2x, Anime4K, SRMD and RealSR. Started in Hack the Valley II, 2018. - k4yt3x/video2x</li><li><a href="https://x.com/">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1286041555429490738)** (13 messages🔥): 

> - `Ollama local API setup`
> - `Speech-to-Text APIs`
> - `Hermes-3 model function calling`
> - `Whisper integration`
> - `Model precision preferences` 


- **Ollama locally utilizes local APIs**: A member pointed out that if you have the model locally, you can set up [Ollama](https://ollama.com) to use local APIs.
   - This is dependent on the model size you want to run.
- **Deepgram offers great STT free plan**: A discussion highlighted that **Deepgram's** free plan is good for speech-to-text (STT) needs, particularly based on usage limits.
   - One member recommended setting up **Whisper** to function like an API, mentioning its versatility even without a GPU, although having one would help.
- **Help with Hermes-3 function arguments**: A member sought assistance in adding a tool with a function that accepts multiple arguments in **Hermes-3**'s `functions.py` file.
   - They provided a [link to their repository](https://github.com/NousResearch/Hermes-Function-Calling/tree/main) for context, hoping for input from others in the community.
- **Request for support regarding Hermes-3**: Another member suggested reaching out to [@909858310356893737](https://discordapp.com/users/909858310356893737) for help with Hermes-3 when available.
   - The original poster expressed appreciation and indicated they would await a response.
- **Model precision settings query**: A participant discussed running the **405b** model at full **bf16** and **fp32** resolution without the truncation issue after accumulation.
   - This emphasizes the need for fine control over model precision for optimal performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://replicate.com/openai/whisper">openai/whisper – Run with an API on Replicate</a>: no description found</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/tree/main">GitHub - NousResearch/Hermes-Function-Calling</a>: Contribute to NousResearch/Hermes-Function-Calling development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1286055344140845058)** (6 messages): 

> - `Shampoo optimization`
> - `Diagram of Thought framework`
> - `ReST-MCTS* paper`
> - `Reverse engineering O1` 


- **Shampoo vs Adam in Optimization Tasks**: Recent findings indicate **Shampoo**, a higher-order preconditioning method, is more effective than **Adam** for deep learning optimization, though it introduces additional hyperparameters and computational overhead compared to Adam's simpler average updates.
   - The study connects Shampoo to **Adafactor**, revealing a new efficient algorithm called **SOAP**, leveraging Shampoo's preconditioner's eigenbasis.
- **Introducing the Diagram of Thought**: The **Diagram of Thought (DoT)** framework models iterative reasoning in LLMs as a directed acyclic graph (DAG), allowing complex reasoning without losing logical consistency, contrasting with traditional linear methods.
   - Each node represents a proposed or critiqued idea, enabling models to improve reasoning iteratively through language feedback.
- **Interest in Reverse Engineering O1**: Members expressed a keen interest in **reverse engineering O1**, indicating a collaborative spirit in exploring this area further.
   - One member mentioned reading overlapping papers and highlighted their exploration into the **ReST-MCTS*** paper.
- **Underappreciation of ReST-MCTS***: A member believes the **ReST-MCTS*** paper, which integrates STaR, PRM, and MCTS, is underappreciated despite its seamless approach to combining these methods.
   - They are eager to discuss this paper further and share insights with others interested in similar topics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.11321">SOAP: Improving and Stabilizing Shampoo using Adam</a>: There is growing evidence of the effectiveness of Shampoo, a higher-order preconditioning method, over Adam in deep learning optimization tasks. However, Shampoo&#39;s drawbacks include additional hyp...</li><li><a href="https://arxiv.org/abs/2409.10038v1">On the Diagram of Thought</a>: We introduce Diagram of Thought (DoT), a framework that models iterative reasoning in large language models (LLMs) as the construction of a directed acyclic graph (DAG) within a single model. Unlike t...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1286111326996922410)** (2 messages): 

> - `Characterfile format`
> - `Multi-agent framework tools`
> - `Twitter archive to character data` 


- **Launch of Characterfile for character data**: The project [characterfile](https://github.com/lalalune/characterfile) provides a simple file format for **character data**, aimed to facilitate sharing among developers working with multi-agent frameworks.
   - It includes examples and validators in **Python** and **JavaScript**, along with scripts like *tweets2character* that generate character files from Twitter archives.
- **Tweet about relevance of characterfile**: A tweet by [_akhaliq](https://twitter.com/_akhaliq/status/1836544678742659242) highlighted the importance of structured character data within development teams.
   - The tweet emphasizes the need for standards in managing character information and collaborative sharing.



**Link mentioned**: <a href="https://github.com/lalalune/characterfile">GitHub - lalalune/characterfile: A simple file format for character data</a>: A simple file format for character data. Contribute to lalalune/characterfile development by creating an account on GitHub.

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1286055344140845058)** (6 messages): 

> - `Shampoo vs Adam`
> - `Diagram of Thought framework`
> - `ReST-MCTS paper discussion`
> - `Reverse engineering O1` 


- **Shampoo Preconditioning Method Outshines Adam**: Research showcases the effectiveness of **Shampoo**, a higher-order preconditioning method over **Adam**, while acknowledging its **hyperparameters** and **computational overhead** drawbacks. A new algorithm, dubbed **SOAP**, simplifies Shampoo's efficiency by connecting it to **Adafactor**.
   - This insight positions SOAP as a competitive alternative aimed at enhancing computational efficiency in deep learning optimizations.
- **Exploration of ReST-MCTS* Paper**: A discussion emerged around the **ReST-MCTS*** paper, noted for its innovative combination of **STaR**, **PRM**, and **MCTS** methodologies, suggesting it is *underappreciated* in its potential. The paper delineates a thorough step-by-step verification approach that intrigued members.
   - Participants expressed a desire to explore overlapping studies and consider new ways to approach the challenges outlined in this paper.
- **Interest in Reverse Engineering O1**: There's a growing interest in **reverse engineering O1**, with several members seeking to share insights and findings. Requests for collaboration indicate a communal effort to explore this topic further.
   - Members expressed a willingness to connect and discuss their research surrounding this area.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.11321">SOAP: Improving and Stabilizing Shampoo using Adam</a>: There is growing evidence of the effectiveness of Shampoo, a higher-order preconditioning method, over Adam in deep learning optimization tasks. However, Shampoo&#39;s drawbacks include additional hyp...</li><li><a href="https://arxiv.org/abs/2409.10038v1">On the Diagram of Thought</a>: We introduce Diagram of Thought (DoT), a framework that models iterative reasoning in large language models (LLMs) as the construction of a directed acyclic graph (DAG) within a single model. Unlike t...
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1286464007640846397)** (1 messages): 

> - `Chatroom features`
> - `Qwen 2.5`
> - `Mistral Pixtral`
> - `Neversleep Lumimaid v0.2`
> - `Hermes 3` 


- **Editable Messages Enhance Chatroom Functionality**: New chatroom features now allow users to **edit messages**, including those from the bot, by clicking the regenerate button for new replies.
   - Additionally, chatroom **stats** have been redesigned for improved user experience.
- **Qwen 2.5 Aces Coding and Math**: **Qwen 2.5 72B** boasts enhanced knowledge and significantly elevated capabilities in coding and mathematics, with an impressive context size of **131,072**. More details can be found [here](https://openrouter.ai/models/qwen/qwen-2.5-72b-instruct).
   - This model marks a notable improvement in performance, particularly for coding applications.
- **Mistral Introduces Pixtral Model**: **Mistral Pixtral 12B** has been launched as Mistral's first multimodal model, along with a **free variant** to explore its capabilities. Further information is available [here](https://openrouter.ai/models/mistralai/pixtral-12b).
   - This release expands Mistral's offerings into multimodal applications, attracting interest from users.
- **Neversleep Lumimaid Gets a Major Update**: The **Neversleep Lumimaid v0.2 8B** is a refined version of Llama 3.1 8B, described as having a **HUGE step up** in dataset quality compared to its predecessor. Check out more about it [here](https://openrouter.ai/models/neversleep/llama-3.1-lumimaid-8b).
   - This update is expected to significantly enhance performance and capabilities.
- **Hermes 3 Model Update**: **Hermes 3** has transitioned to a **paid model** priced at **$4.5/m**, although a **free** and an **extended variant** remain available for users. More details can be found [here](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b).
   - This shift may alter user accessibility while still offering alternatives.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/qwen/qwen-2.5-72b-instruct">Qwen2.5 72B Instruct - API, Providers, Stats</a>: Qwen2.5 72B is the latest series of Qwen large language models. Run Qwen2.5 72B Instruct with API</li><li><a href="https://openrouter.ai/models/neversleep/llama-3.1-lumimaid-8b">Lumimaid v0.2 8B - API, Providers, Stats</a>: Lumimaid v0.2 8B is a finetune of [Llama 3. Run Lumimaid v0.2 8B with API</li><li><a href="https://openrouter.ai/models/mistralai/pixtral-12b">Pixtral 12B - API, Providers, Stats</a>: The first image to text model from Mistral AI. Its weight was launched via torrent per their tradition: https://x. Run Pixtral 12B with API</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coheren...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1286289611018207243)** (1 messages): 

> - `Google Gemini`
> - `No-code agent creation`
> - `Open Agent Cloud`
> - `Enterprise automation`
> - `Screen recording agents` 


- **Google Gemini Launches Video-to-Agent Feature**: With **Google Gemini**, users can now upload a **Loom video** to create a **no-code drag-drop agent** in seconds, making it the fastest way to build agents on the planet.
   - *Previously, building a Twitter agent took 20 minutes, but now it can be done in just 5 seconds from a recorded video.*
- **Scale Agents Instantly in Open Agent Cloud**: Once created, agents can be instantly run in **Open Agent Cloud**, allowing users to scale schedules to **thousands of agents** running in parallel.
   - All agents stream data directly to the **dashboard**, ensuring real-time monitoring and control.
- **Solving Expertise Loss in Enterprises**: This innovative approach addresses a critical problem in enterprises and governments: the **loss of expertise** when employees and contractors leave.
   - Now, users can generate agents from **decades-old screen recordings**, preserving valuable knowledge.
- **Watch the Demo Video**: Check out the introduction of this groundbreaking feature in [this YouTube video](https://www.youtube.com/watch?v=gsU5033ms5k), showcasing how to create agents effortlessly.
   - The video provides insights into leveraging video content for enhanced productivity and automation.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1286069957137793067)** (136 messages🔥🔥): 

> - `OpenAI Rate Limits`
> - `Payment Issues on OpenRouter`
> - `Model Sharing and Chat History`
> - `Integrating New Models`
> - `Job Impact of AI` 


- **Increased API Rate Limits by OpenAI**: OpenAI has increased the rate limits for its o1 API, with o1-preview now allowing **500 requests per minute** and o1-mini **1000 requests per minute**.
   - This change aims to support developers using tier 5 rates, further increasing access to broader functionalities.
- **Payment Issues on OpenRouter**: Users reported issues regarding payment errors when attempting to add credits on OpenRouter, often receiving an **error 500** message indicating insufficient funds.
   - It was suggested that users check their bank notifications as payment attempts could be denied for various reasons.
- **Local Storage of Chat History**: Users inquired about sharing chat history across devices, discovering that OpenRouter's chat logs are stored locally without a direct sharing feature.
   - Exporting chats to a JSON file was mentioned as the only way to transfer conversation data between devices.
- **Integrating New Models on OpenRouter**: Inquiries were made on how to distribute new models through OpenRouter, indicating a need for formal requests or guidance on the integration process.
   - Users expressed interest in the steps necessary to offer new models via their API on the platform.
- **AI Job Impact Analysis**: A discussion arose about the potential impact of AI and automation on jobs, projecting various scenarios for job displacement through 2027 and beyond.
   - Speculation suggested that AI advancements could affect **10-20%** of jobs by 2027, potentially rising to **50-70%** by 2040.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/slow_developer/status/1836693976050475167">Tweet from Haider. (@slow_developer)</a>: 🚨 OpenAI CEO Sam Altman confirms moving from o1-preview to full o1 model soon  &#34; The new reasoning model o1-preview will significantly improve over the coming months when we shift from an initial...</li><li><a href="https://docs.mistral.ai/getting-started/models/">Models | Mistral AI Large Language Models</a>: Overview</li><li><a href="https://openrouter.ai/settings/privacy">Privacy | OpenRouter</a>: Manage your privacy settings</li><li><a href="https://x.com/OpenAIDevs/status/1836506351062716701">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Just 5x&#39;d rate limits again:    o1-preview: 500 requests per minute  o1-mini: 1000 requests per minute  Quoting OpenAI Developers (@OpenAIDevs)   We&#39;ve increased OpenAI o1 API rate limits for ...</li><li><a href="https://mistral.ai/technology/#pricing">Technology</a>: Frontier AI in your hands</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>: Every front-end GUI client for ChatGPT. Contribute to billmei/every-chatgpt-gui development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1286041057880178698)** (99 messages🔥🔥): 

> - `Qwen 2.5 training issues`
> - `Extreme Quantization`
> - `Model support inquiries`
> - `Lightweight tools for OpenAI endpoints`
> - `Style transfer in model training` 


- **Training Qwen 2.5 Has Challenges**: Several users reported issues with saving and reloading models, particularly Qwen 2.5, leading to errors and generating gibberish when reloaded within the same script.
   - One user mentioned a support post indicating that this problem has affected multiple individuals, prompting inquiries about possible fixes.
- **Exploring Extreme Quantization Techniques**: A post highlighted the latest releases of models utilizing extreme quantization techniques with significant performance gains, as shared on Hugging Face.
   - Models like Llama3-8B have been fine-tuned for improved efficiency, raising interest in whether Unsloth can accommodate them.
- **Qwen 2.5 Support Inquiry**: Users are eager to find out whether Qwen 2.5 is supported on various inference libraries, with reports suggesting that it works on Oobabooga.
   - There are varying opinions on whether Unsloth supports the new variants of Qwen 2.5, with some users directly experimenting without relying on Unsloth's models.
- **Looking for Lightweight Tools for OpenAI**: Discussions focused on the need for simple tools that can be easily installed by non-technical users to test OpenAI-supported endpoints.
   - Suggestions like SillyTavern and LM Studio were mentioned, but there were concerns about their compatibility with OpenAI’s API.
- **Style Transfer Techniques in AI Training**: A user inquired about training a model to replicate their style, leading to suggestions that style transfer doesn't require extensive pretraining, just data from the user.
   - There was emphasis on automation through scripting to enhance the efficiency of training models to reflect personal style.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/HF1BitLLM">HF1BitLLM (Hugging Face 1Bit LLMs)</a>: no description found</li><li><a href="https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd">Moshi v0.1 Release - a kyutai Collection</a>: no description found</li><li><a href="https://huggingface.co/blog/1_58_llm_extreme_quantization">Fine-tuning LLMs to 1.58bit: extreme quantization made easy</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fk0acj/hacks_to_make_llm_training_faster_guide/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/homebrewltd/llama3-s/issues/56">epic: llama3-s v0.3: &quot;I cannot hear / understand you&quot; · Issue #56 · homebrewltd/llama3-s</a>: Goal Make v0.3 multilingual, accept longer questions, and other data improvements. Problem Previously v0.2 only worked well on instructions under 10s Previously v0.2 only worked well on English inp...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/5761">Support BitNet b1.58 ternary models · Issue #5761 · ggerganov/llama.cpp</a>: New paper just dropped on Arxiv describing a way to train models in 1.58 bits (with ternary values: 1,0,-1). Paper shows performance increases from equivalently-sized fp16 models, and perplexity ne...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8151">ggml-quants : ternary packing for TriLMs and BitNet b1.58 by compilade · Pull Request #8151 · ggerganov/llama.cpp</a>: This adds 1.6875 bpw and 2.0625 bpw quant types for TriLMs and BitNet b1.58 models. For now, these are named TQ1_0 and TQ2_0, respectively. I had given glimpses of this idea starting from #7931 (co...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1286087374975008790)** (10 messages🔥): 

> - `Probability struggles`
> - `Swift programming`
> - `Confidence in exams`
> - `Qwen models`
> - `AGI challenges` 


- **Probability Questions Getting Complicated**: A member expressed frustration over how **advanced** flipping coin questions have become since high school, indicating a struggle with the material.
   - Another member suggested using a **Binomial distribution calculator** to simplify the problem.
- **Mahiatlinux Returns After Exams**: A member, recently back from exams, checked in on the community stating they felt **confident** about their mock exams and looked into **new Qwen models**.
   - They wished luck to another member learning **Swift**, indicating a supportive community atmosphere.
- **Reflections on AGI Development**: A member noted their experience studying difficult material highlights the substantial gap we remain from **AGI**, emphasizing the need for explanation, not just answers.
   - *It's not about getting the answer right but the explaining part,* they remarked, pointing to a significant challenge in the field.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1286060794617921642)** (18 messages🔥): 

> - `vllm LoRA Adapter Issues`
> - `Pip Installation Problems`
> - `Model Fine-tuning Limitations`
> - `Unsloth Model Usage in Ollama`
> - `Batch Size and Training Speed` 


- **vllm LoRA Adapter Causes Runtime Errors**: A member reported an error when trying to run the command with `--qlora-adapter-name-or-path`, leading to a runtime exception regarding shape mismatch.
   - They referenced a specific [GitHub discussion](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit/discussions/3) for a similar issue encountered previously.
- **Challenges with Pip Installation**: Another member described difficulties with pip installing the Unsloth library using specific environment markers and additional packages.
   - They sought advice on whether their installation method was correct, raising a question about potential issues with their approach.
- **Fine-tuning phi-3.5 mini Hits Max Length Wall**: Concerns were raised about the inability to fine-tune or infer with phi-3.5-mini using a max_length greater than 4095, with a request for workarounds.
   - A link to a [GitHub issue](https://github.com/unslothai/unsloth/issues/946) detailing an error encountered during fine-tuning was shared for further context.
- **Unsloth Model Template for Ollama**: A member queried whether the provided template was correct for running the Unsloth model in Ollama, showing snippets of code for clarification.
   - Their inquiry highlights the ongoing adjustments users are making when implementing various models.
- **Batch Size Increase Does Not Speed Up Training**: A member expressed concern that increasing the batch size didn't result in faster training, raising questions about expected behaviors.
   - Their query indicates a puzzlement in performance optimization when adjusting training parameters.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/nlp-course/chapter7/3?fw=tf#fine-tuning-distilbert-with-the-trainer-api">Fine-tuning a masked language model - Hugging Face NLP Course</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/issues/946">AttributeError: &#39;LongRopeRotaryEmbedding&#39; object has no attribute &#39;inv_freq&#39; when finetuning Phi3.5 mini · Issue #946 · unslothai/unsloth</a>: Hello, I get the error in the title when finetuning Phi3.5. I believe I&#39;m on the latest unsloth (installed from git wit pip). Context: finetuning Phi3.5 with code that already works with other uns...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1286273880440176691)** (2 messages): 

> - `Fine-tuning BART`
> - `F1 Score Discrepancy`
> - `Multiple BOS Tokens Issue` 


- **Fine-tuning BART large yields unexpected results**: An individual is fine-tuning **BART large** to reproduce results from a paper but is encountering a **discrepancy** in F1 scores (41.5 vs 43.5).
   - Despite using the same model, hyperparameters, and dataset as the authors, they found their scores 2.5 standard deviations lower.
- **Unexpected multiple BOS tokens during generation**: The user reported that **BART** occasionally outputs **two or three** beginning-of-sequence (BOS) tokens, which are not part of the fine-tuning data.
   - They checked the input batches and confirmed only one BOS token is added, suggesting a deeper issue with the model configuration.


  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1286047379320999939)** (98 messages🔥🔥): 

> - `Aider Environment Variables`
> - `Benchmarks and Performance`
> - `Model Utilization in Aider`
> - `Issues and Solutions with Aider Setup`
> - `RAG Architecture in Aider` 


- **Fixing Aider Environment Misconfiguration**: Users identified issues with the `ANTHROPIC_API_KEY` environment variable not being read correctly due to incorrect file paths, leading to authentication problems.
   - After using verbose mode, a user confirmed the error was due to Aider reading from their repo instead of the intended environment variable.
- **Aider's Benchmark Recognition**: Aider received acknowledgment in the 'Qwen2.5-Coder Technical Report' for its benchmark contributions, highlighting its significance in the field.
   - This recognition illustrates the growing impact of Aider as a valuable tool in AI development and performance evaluation.
- **Utilizing Aider Features Effectively**: Users discussed the functionality of running shell commands from within Aider using the `/run` command, providing examples with pytest.
   - Best practices for Aider commands and settings were shared, improving user experience and productivity.
- **Problem-Solving Strategies in Aider**: Discussion revealed common issues users faced when connecting Aider to the Anthropic API, including API overload and variable mismanagement.
   - Proposals for troubleshooting these issues included verifying environment variables and adjusting commands for better connectivity.
- **Architectural Insights on Aider**: Inquiries were made regarding Aider's high-level architecture, specifically how it utilizes repository maps for improved context in code editing.
   - The repo map system helps Aider understand relationships within code bases, enhancing its effectiveness in AI-assisted programming.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenAIDevs/status/1836506351062716701">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Just 5x&#39;d rate limits again:    o1-preview: 500 requests per minute  o1-mini: 1000 requests per minute  Quoting OpenAI Developers (@OpenAIDevs)   We&#39;ve increased OpenAI o1 API rate limits for ...</li><li><a href="https://x.com/slow_developer/status/1836693976050475167">Tweet from Haider. (@slow_developer)</a>: 🚨 OpenAI CEO Sam Altman confirms moving from o1-preview to full o1 model soon  &#34; The new reasoning model o1-preview will significantly improve over the coming months when we shift from an initial...</li><li><a href="https://aider.chat/docs/llms/anthropic.html">Anthropic</a>: aider is AI pair programming in your terminal</li><li><a href="https://arxiv.org/abs/2409.12186">Qwen2.5-Coder Technical Report</a>: In this report, we introduce the Qwen2.5-Coder series, a significant upgrade from its predecessor, CodeQwen1.5. This series includes two models: Qwen2.5-Coder-1.5B and Qwen2.5-Coder-7B. As a code-spec...</li><li><a href="https://aider.chat/docs/llms">Connecting to LLMs</a>: Aider can connect to most LLMs for AI pair programming.</li><li><a href="https://aider.chat/docs/repomap.html">Repository map</a>: Aider uses a map of your git repository to provide code context to LLMs.</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider is AI pair programming in your terminal</li><li><a href="https://tenor.com/view/twenty-one-pilots-stressed-out-wake-up-you-need-to-make-money-21pilots-gif-16455885">Twenty One Pilots Stressed Out GIF - Twenty One Pilots Stressed Out Wake Up - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/usage/commands.html">In-chat commands</a>: Control aider with in-chat commands like /add, /model, etc.</li><li><a href="https://status.anthropic.com/incidents/gg215bzz7rhm">3.5-Sonnet Partial Outage</a>: no description found</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/CONTRIBUTING.md#setting-up-a-development-environment)">aider/CONTRIBUTING.md at main · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/codelion/optillm">GitHub - codelion/optillm: Optimizing inference proxy for LLMs</a>: Optimizing inference proxy for LLMs. Contribute to codelion/optillm development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/tests/basic/test_io.py">aider/tests/basic/test_io.py at main · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/coders/ask_prompts.py?">aider/aider/coders/ask_prompts.py at main · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1286071696792490005)** (20 messages🔥): 

> - `Using Aider in Python apps`
> - `Security concerns with Aider`
> - `Creating files with Aider`
> - `Hugging Face model integration`
> - `Control over URL scraping` 


- **Integrating Aider into Python Applications**: A user sought to use Aider within a Python app to edit code in a user's project repo by specifying the base folder for Aider.
   - Another user suggested using command line scripting with Aider for batch operations, indicating that setting the file path correctly can resolve editing issues.
- **Concerns About API Key Safety**: A group discussion revealed a user's anxiety about security when using Aider due to its access to API keys and secrets within codebases.
   - Responses clarified that Aider acts as an AI handler, suggesting users focus on the AI they load to address security concerns.
- **Challenges in Creating Files with Aider**: A user expressed difficulty in creating empty files through Aider despite providing a detailed structure and folder path.
   - Others recommended adding only pertinent files to the chat for Aider to function effectively, along with checking the documentation for tips.
- **Using Aider with Hugging Face Models**: A user inquired about guidance for using Aider with Hugging Face but faced challenges in listing models correctly through Aider commands.
   - Responses pointed to a specific documentation link that explains compatibility and usage for Hugging Face models, suggesting the model name format needed.
- **Managing URL Scraping Behavior**: A user questioned how to prevent Aider from scraping URLs pasted in prompts without explicit instruction.
   - They expressed frustration with the automatic behavior and clarified their preference for manually triggering scraping using a specific command.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://docs.litellm.ai/docs/providers/huggingface">Huggingface | liteLLM</a>: LiteLLM supports the following types of Hugging Face models:</li><li><a href="https://aider.chat/docs/usage/tips.html#creating-new-files">Tips</a>: Tips for AI pair programming with aider.</li><li><a href="https://github.com/paul-gauthier/aider/tree/main/aider/website">aider/aider/website at main · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1286039221508767754)** (8 messages🔥): 

> - `ECMAScript vs JavaScript`
> - `NodeJS alternatives`
> - `ell Library`
> - `Prompt Engineering` 


- **ECMAScript vs Everyone Calls It JavaScript**: A debate emerged about whether it should be called **ECMAScript** or **JavaScript**, with one member insisting that the name should be public domain due to Oracle's lack of action.
   - Humorously, another member suggested they should let fans of JavaScript sort it out themselves.
- **LiteLLM Alternative for NodeJS Enthusiasts**: A member shared an alternative to LiteLLM for building AI apps in **NodeJS**, which can be found at [Portkey-AI](https://www.npmjs.com/package/portkey-ai).
   - This could provide developers with a new approach to integrate language models into their applications.
- **'ell' Library for Prompt Engineering**: Detailed information was shared about the **'ell' library**, a lightweight prompt engineering tool that allows prompts to be treated as **functions**.
   - The library is introduced as an outcome of years of experience in the language model space from OpenAI, aimed at enhancing prompt design.
- **Excitement Over Similar Projects**: One member expressed concern that the **'ell' library** was too similar to a project they are developing, leading to a moment of panic.
   - Another cheered this alignment of ideas, stating that 'great minds think alike!' in a lighthearted manner.



**Link mentioned**: <a href="https://docs.ell.so/index.html#">Introduction | ell  documentation</a>: ell is a lightweight prompt engineering library treating prompts as functions. It provides tools for versioning, monitoring, and visualization of language model programs.

  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1286166124286705685)** (4 messages): 

> - `airLLM compression`
> - `Leaderboard tasks`
> - `HF dataset upload` 


- **Query on airLLM's Forward Call**: A member asked if using **airLLM** allows calling a model's **forward** function instead of the **generate** function while still benefiting from compression.
   - No response was provided but this raises interesting questions about flexibility in model usage.
- **Leaderboard Task Accuracy Extraction**: A member expressed the need for a script to extract main **accuracy results** from lengthy JSON files generated while running leaderboard tasks on a local model.
   - They mentioned results are saved in **output_path**, indicating a desire for easier data handling.
- **Suggested HF Upload Method**: Another member advised using `—hf_hub_log_args` to upload leaderboard results to Hugging Face, suggesting it makes handling results simpler.
   - They referenced a dataset example with one row per run available at [this link](https://huggingface.co/datasets/baber/eval-smolLM-135M-3-private).
- **Plans for Custom Script Development**: The member who needed the accuracy extraction indicated their intention to create a simple script to facilitate the process.
   - This reflects the community's proactive approach to solving common challenges.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1286146330099716238)** (74 messages🔥🔥): 

> - `Shampoo Optimizer`
> - `SOAP Algorithm`
> - `GFlowNets and JEPA`
> - `Inference Time Search`
> - `Self-Supervised Learning Challenges` 


- **Shampoo vs. Adam Performance Insights**: Research indicates that **Shampoo**, a higher-order preconditioning method, outperforms Adam in optimization tasks but has drawbacks such as increased computational overhead and hyperparameters.
   - A proposed solution is **SOAP**, which combines the strengths of Shampoo and Adafactor by operating in the eigenbasis of Shampoo's preconditioner.
- **GFlowNets and JEPA Discussion**: Skepticism surrounds the contributions of **GFlowNets** and **JEPA**, with concerns about the practical impact and clarity of purpose for these models.
   - Users discussed the potential indirect influence of GFlowNets in AI for science while noting that JEPA's theoretical grounding appears weak.
- **OpenAI's Inference Time Search**: OpenAI is reportedly moving toward self-play techniques and inference time search, as emphasized by Noam Brown, to enhance quality outputs after initial model training.
   - The effectiveness of this approach in games like **chess and poker** was noted as significantly improving performance, which may provide insight into its current strategy.
- **Challenges in Self-Supervised Learning**: Several papers were cited discussing the difficulties faced in **self-supervised learning** (SSL), including instability of optimizers and representation collapse during training.
   - Recent research aims to unify theoretical perspectives on these challenges to guide practitioners effectively.
- **Kye Gomez Persona in AI Research**: There was concern about the reputation of Kye Gomez, emphasizing that while his repository may improve slightly, it remains fundamentally misleading.
   - Discussions reflected skepticism about the quality and integrity of his contributions to AI, especially in light of potential updates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.11321">SOAP: Improving and Stabilizing Shampoo using Adam</a>: There is growing evidence of the effectiveness of Shampoo, a higher-order preconditioning method, over Adam in deep learning optimization tasks. However, Shampoo&#39;s drawbacks include additional hyp...</li><li><a href="https://arxiv.org/html/2409.11340v1">OmniGen: Unified Image Generation</a>: no description found</li><li><a href="https://github.com/nikhilvyas/SOAP">GitHub - nikhilvyas/SOAP</a>: Contribute to nikhilvyas/SOAP development by creating an account on GitHub.</li><li><a href="https://arxiv.org/abs/2302.02774">The SSL Interplay: Augmentations, Inductive Bias, and Generalization</a>: Self-supervised learning (SSL) has emerged as a powerful framework to learn representations from raw data without supervision. Yet in practice, engineers face issues such as instability in tuning opti...</li><li><a href="https://arxiv.org/abs/2303.00633">An Information-Theoretic Perspective on Variance-Invariance-Covariance Regularization</a>: Variance-Invariance-Covariance Regularization (VICReg) is a self-supervised learning (SSL) method that has shown promising results on a variety of tasks. However, the fundamental mechanisms underlying...</li><li><a href="https://arxiv.org/abs/2205.11508">Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods</a>: Self-Supervised Learning (SSL) surmises that inputs and pairwise positive relationships are enough to learn meaningful representations. Although SSL has recently reached a milestone: outperforming sup...</li><li><a href="https://arxiv.org/abs/2408.16978">Training Ultra Long Context Language Model with Fully Pipelined Distributed Transformer</a>: Large Language Models (LLMs) with long context capabilities are integral to complex tasks in natural language processing and computational biology, such as text generation and protein sequence analysi...</li><li><a href="https://github.com/microsoft/DeepSpeed/pull/6462">Adding the new feature of FPDT by YJHMITWEB · Pull Request #6462 · microsoft/DeepSpeed</a>: FPDT can only be used with this version of Megatron-DeepSpeed.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1286101066005479476)** (15 messages🔥): 

> - `Data Matching in OPT and Pythia`
> - `Attention Layer and In-Context Learning Score`
> - `Distributed Alignment Search and SAEs`
> - `KV Cache Representation Exploration` 


- **Data matching insights between OPT and Pythia**: Data from **OPT** and **Pythia** shows that post-training data remains consistent, primarily presenting smoother transitions across layer depth.
   - Final hidden states' power law coefficients converge quickly after roughly **1B tokens**, indicating stability in model performance.
- **Attention Layer's Behavior Over Training**: Graphs indicate that the **R^2** metric for the final attention layer residual converges at **500M tokens**, while the in-context learning score peaks at **1B tokens**.
   - This distinct shape has not been observed in other graphs, raising questions about its implications for training performance.
- **Inquiry on Distributed Alignment Search paper**: A member inquired about a post related to **distributed alignment search** and **SAEs**, which may have disappeared.
   - Another member referenced the **Ravel Eval** paper on open-source SAEs, suggesting it could be the paper in question.
- **KV Cache Representation Inquiry**: A member is exploring what is stored in the **KV cache**, particularly regarding its composition from token embeddings and previous states in lower layers.
   - The discussion connects to concepts of **chain-of-thought** and **thinking along token lines**, highlighting areas of interest for further research.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1286293953381138545)** (18 messages🔥): 

> - `Padding and Unpadding Inputs`
> - `Error Handling in Model Generate`
> - `Token Generation with Padding`
> - `Batch Size Management` 


- **Padding Inputs Causes AssertionError**: A member encountered an `AssertionError` while trying to restore the original order of elements after padding inputs in `model_generate`.
   - It was hypothesized that padding with stop tokens might lead to issues, as they might be filtered out during processing.
- **Understanding Generate Function Logic**: A discussion revealed that the `generate_until` function sorts requests by token length to manage memory issues efficiently.
   - This approach optimizes performance by estimating time accurately and returning to the original order later.
- **Clarifying Padding Logic in Model Generate**: It was shared that the `_model_generate` function pads the batch to the specified size and removes the padding afterward.
   - The member acknowledged a misunderstanding regarding how the padding was handled after slicing the tensor.
- **Identifying the Return Issue**: The member discovered that the correct return statement should return `toks[:context.size(0)]` to avoid an empty array when the batch is full.
   - They shared this insight in light of the previous confusion about the padding mechanism.



**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/9a092f374bdc6d6032ae2b878b7a49b97801ab69/lm_eval/models/huggingface.py#L1304">lm-evaluation-harness/lm_eval/models/huggingface.py at 9a092f374bdc6d6032ae2b878b7a49b97801ab69 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1286321409160773643)** (1 messages): 

> - `Polaris Node Connectivity`
> - `Job Timeout Issues` 


- **Transient Errors plaguing Polaris Nodes**: Members reported a **transient error** on Polaris where nodes are unable to locate each other, leading to failed connections.
   - As a result, jobs often time out after an hour, resulting in a **TCPConnectionError**.
- **Job Timeout Troubles**: The job timing out after an hour creates frustration among users, highlighting a significant connectivity problem within Polaris.
   - The issue is impacting the overall efficiency and reliability of the platform.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1286050149910708245)** (36 messages🔥): 

> - `O1-Preview Capabilities`
> - `Discussion on AI Alignment`
> - `Qwen 2.5 vs Llama 3.1`
> - `Recording ChatGPT's Voice`
> - `Pixtral & OpenRouter` 


- **O1-Preview is Underwhelming**: Members expressed disappointment that the **O1-Preview** model feels inferior, with comments highlighting that it seems to type faster but lacks depth compared to **4o**.
   - *One member remarked*, 'o1 doesn't feel smarter, it just types faster'.
- **AI Alignment Challenges Discussed**: A member proposed a new method for improving **AI alignment**, suggesting training future models with a focus on empathy and helpfulness based on previous models' outputs.
   - *Concerns were raised about whether a superintelligent AI could still mislead users* by presenting tailored responses.
- **Comparing Model Performance**: Participants discussed the impressive claims about **Qwen 2.5**, with members noting it reportedly outperforms **Llama 3.1**, despite significant differences in parameter size.
   - *One user remarked*, 'ppl saying crazy stuff like qwen 2.5 72b outperforming llama 3.1 405b'.
- **Challenges in Voice Recording with ChatGPT**: One user expressed frustration at not being able to record audio from **ChatGPT** on their phone, reporting no sound captured during attempts.
   - Despite using the phone's recording feature, they were unable to achieve the desired result.
- **Pixtral and OpenRouter Insights**: A user shared that **Pixtral** can be accessed for free via **OpenRouter**, and confirmed its functionality on that platform with a specific model suggestion.
   - *They advised others* to use the free model to maximize the benefits of the tool.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1286045977312100363)** (21 messages🔥): 

> - `Daily Limits for GPT Usage`
> - `GPT-4 and GPT-4o Limits`
> - `O1 and O1-Mini Message Caps`
> - `Memory Functionality in GPTs` 


- **Daily Limits for Mini Right**: It was confirmed that the **O1 Mini** has a limit of **50 messages per day**.
   - This rule is crucial to prevent **disruption on the server** through spamming or repetitive posts.
- **Understanding GPT-4o Message Cap**: Members discussed that **GPT-4o** allows for **80 messages every 3 hours**, the same as its limit.
   - However, the **GPT-4 limit** in this context allows only **40 messages** within that same timeframe.
- **Clarifying O1 and O1-Mini Caps**: It was established that the usage limits for **O1 and O1-Mini** are independent of the **GPT-4 and GPT-4o limits**.
   - The caps follow a **24-hour and 7-day window** respectively for O1-Mini and O1-Preview.
- **Navigating Memory with GPTs**: Currently, it was clarified that **GPTs do not have memory functionality**.
   - For more information on this, a link was shared regarding memory in GPTs: [help article](https://help.openai.com/en/articles/8983148-does-memory-function-with-gpts).
- **Usage Limits Resources**: A member shared useful links for understanding the current **usage limits**, including one that consolidates all limits into a single page.
   - The resource was highly recommended for clearer information on usage limits offered by OpenAI: [usage limits link](https://help.openai.com/en/articles/6950777-what-is-chatgpt-plus#h_d78bb59065).


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1286081870756315274)** (3 messages): 

> - `Data extraction with GPT-4o`
> - `Structured Output feature` 


- **Using GPT-4o for CSV data extraction**: A member is using **GPT-4o** for data extraction from unstructured text and previously employed a system prompt for **CSV delimited output** through examples.
   - They now seek guidance on adapting their few-shot examples for the **new Structured Output feature** since CSV wouldn't be suitable for JSON outputs.
- **Adapting prompts for Structured Output**: Another member suggested providing example **JSON outputs** in the system prompt to align with the Structured Output feature.
   - They confirmed that this approach appears effective, noting that the model understands the task better with the provided examples.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1286081870756315274)** (3 messages): 

> - `GPT-4o Data Extraction`
> - `Structured Output Feature` 


- **Transitioning from CSV to JSON for Outputs**: A user shared their experience using **GPT-4o** for data extraction from unstructured text, focusing on past use of CSV formatted outputs.
   - They inquired about how to adapt their **Few Shot examples** in the system prompt for the new **Structured Output feature**, suggesting they may need to provide example JSON outputs instead.
- **Helpful Prompt Example Provided**: Another participant suggested a possible solution, mentioning a specific prompt format that could effectively convey the intended extraction to **GPT-4o**.
   - They expressed confidence that this method would help the model understand and generate structured outputs accurately.


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1286356941865619518)** (4 messages): 

> - `NVIDIA Triton`
> - `Triton Inference Server`
> - `Related Discussion Rooms` 


- **Kashimoo asks about NVIDIA Triton**: A member inquired if anyone is familiar with **NVIDIA's Triton**, specifically clarifying it is not OpenAI's version.
   - This sparked a discussion about relevant rooms and resources dedicated to Triton.
- **Clarification on Triton Inference Server**: Another member prompted a question regarding whether the inquiry was about **NVIDIA's Triton Inference Server** specifically.
   - This detail was clarified by Kashimoo, ensuring the focus remained on NVIDIA's offering.
- **Resources and Rooms for Triton Discussion**: A member suggested multiple related rooms for deeper engagement on Triton, noting the importance of them like <#1191300313928433664> and <#1189607595451895918>.
   - They recommended collaboration with other working groups such as <#1275130785933951039> for additional insights.


  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1286090763616256030)** (3 messages): 

> - `GemLite-Triton`
> - `Triton Conference Slides`
> - `Triton-Puzzles Colab`
> - `Gradio Output` 


- **GemLite-Triton Launches**: The **GemLite-Triton** project was announced, providing a complete solution for building custom low-bit matmul kernels supporting both GEMV/GEMM, weight quantization, and various activation formats.
   - On large matrices, **GemLite-Triton** reportedly outperforms highly optimized solutions like **Marlin** (VLLM) and **BitBlas**. Check it out on [GitHub](https://github.com/mobiusml/gemlite).
- **Triton Conference Slides Inquiry**: A member inquired about the availability of slides from the **Triton Conference** and whether there is a timeline for sharing them.
   - They tagged another member to seek additional information on the status of the slides.
- **Issues with Triton-Puzzles on Colab**: A user raised concerns about whether the **Triton-Puzzles** notebook on Colab is still functioning properly as the Gradio output appears problematic.
   - They shared a link to the notebook for reference but did not provide specific error details.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/srush/Triton-Puzzles/blob/main/Triton-Puzzles.ipynb#scrollTo=W9appXLw4Bka">Google Colab</a>: no description found</li><li><a href="https://github.com/mobiusml/gemlite">GitHub - mobiusml/gemlite: Simple and fast low-bit matmul kernels in CUDA / Triton</a>: Simple and fast low-bit matmul kernels in CUDA / Triton - mobiusml/gemlite
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1286053729946177646)** (7 messages): 

> - `Chrome Tracing with PyTorch Profiler`
> - `FlashAttention3 Integration`
> - `Model Optimization Talks` 


- **Exploring Chrome Tracing with PyTorch Profiler**: A member inquired about a [video or resource](https://discord.com/channels/1189498204333543425/1189607726595194971/1273329959041105980) for navigating Chrome tracing with the PyTorch profiler.
   - Another member recommended the **Taylor Robbie talk** as a valuable resource for this topic.
- **FlashAttention3 and GPU Compatibility**: In response to a question on whether `torch.nn.functional.scaled_dot_product_attention()` automatically uses **FlashAttention3** on Hopper GPUs, a member noted that FlashAttention-3 is expected to be integrated in an upcoming PyTorch release.
   - This insight was linked to a **blog post from July** regarding FlashAttention3.
- **Request for Model Optimization Talk Links**: A member recalled a talk on optimizing models and requested links to other recommended talks, wondering if it was given by another member.
   - This led to unclear recollections, further emphasizing the interest in **model optimization** discussions.


  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1286275447470166112)** (4 messages): 

> - `Torchao Autoquant`
> - `Torchao Compile` 


- **Understanding Torchao Autoquant Syntax**: A user sought clarification on whether to use `torchao.autoquant(model.cuda())` or `torchao.autoquant(model).cuda()` for correct syntax.
   - Another member explained that the correct approach is `torchao.autoquant(model).cuda()` since it first prepares the model before moving it to the device.
- **Steps in Autoquantization Process**: A detailed explanation of the **three steps** involved in autoquantization was provided, emphasizing the need for model preparation before calibration and finalization.
   - This process highlights the importance of running the model on inputs after it has been prepared for effective optimization.
- **Documented Example Usage**: An example of using `torchao.autoquant()` with model compilation was shared as a recommended approach: `torchao.autoquant(torch.compile(model))`.
   - Clarification included the usage of multiple input shapes and the need to finalize autoquant after running the model with those inputs.


  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1286058022313132225)** (3 messages): 

> - `GPU programming groups`
> - `San Francisco GPU community` 


- **Interest in GPU Programming Work Group in SF**: A member expressed interest in joining a **GPU programming reading/work group** based in San Francisco.
   - They highlighted their enthusiasm by stating, *'Would love to join'*.
- **Inquiry About Group Plans**: Another member inquired if a specific member was still planning to organize such a group, showing eagerness for community building.
   - The response indicated that **no current plans** were in place for such a group.


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1286124903451328523)** (10 messages🔥): 

> - `Hackathon Objectives`
> - `GQA cuDNN Development`
> - `L2 Side Aware Optimization`
> - `Stochastic Rounding Technique` 


- **Hackathon Boosts Developer Contributions**: The upcoming hackathon aims to enable more developers to contribute to open source, offering credits to those who may be 'GPU poor' before the event, fostering engagement and innovation.
   - *After the hackathon, participants can become GPU rich!*
- **GQA cuDNN Work Progress**: There has been a focus on the **GQA cuDNN** with discussions on both the forward and backward implementations, indicating some uncertainty about stride and layout correctness.
   - Some participants confirmed they were planning to address the backward portion later, with hopes of resolving issues during travel.
- **L2 Side Aware Optimization Article In Progress**: While traveling, a member mentioned they might delay cuDNN tasks to prioritize writing an article on **L2 Side Aware optimization** and **NVIDIA's memory hierarchy**.
   - They indicated that development under mediocre WiFi conditions would be significantly painful, emphasizing the challenges of remote work.
- **Stochastic Rounding Technique Proposed**: A member shared a novel idea involving **stochastic rounding**, suggesting to force some mantissa bits to zero to conserve power during processing.
   - They found the hack idea to be quite intriguing, hinting at potential optimizations from this technique.
- **Discomfort in Writing Ultra-Vague Megahack**: Questions arose regarding the comfort level among team members in drafting a vague proposal about 'llm.c on GH200 and/or Llama3.1', with the original poster expressing hesitation.
   - ...they sought support from others who might feel more at ease tackling the task.


  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1286224957667541003)** (10 messages🔥): 

> - `Language model quantization`
> - `BitNet performance`
> - `Knowledge compression`
> - `Llama 3 8B results`
> - `Product quantization methods` 


- **Language Models Store Limited Knowledge**: A recent paper establishes that language models can store **2 bits of knowledge per parameter**, which reduces to **0.7 bits per parameter** when quantizing to **int4**. However, some members questioned the accuracy of these findings, considering large models retain performance despite lower quantization.
   - *One member highlighted the difference between measuring 'knowledge' versus traditional accuracy metrics in benchmarks*.
- **BitNet Challenges Performance Recovery**: A translation from a discussion noted that efforts to recover the performance of **L2-7B** with **L3-8B** have failed, implying there's insufficient rationale for **BitNet** under current methods. This sparked concern about whether merely fine-tuning without pre-training is effective for high performance.
   - *A member remarked on the questionable efficacy of using **SFT** (Supervised Fine-Tuning) to transform model capabilities, suggesting it's more logical to rely on pre-training strategies.*
- **Exciting News on Llama 3 8B**: Another contributor reported on the successful fine-tuning of a **Llama 3 8B**, achieving performance close to **Llama 1 & 2 7B models** without pre-training. More details are available in the [Hugging Face blogpost](https://huggingface.co/blog/1_58_llm_extreme_quantization).
   - *The participant also expressed a certain skepticism about the implications of these findings in light of previous discussions on model performance.*
- **Insights Into Knowledge Compression**: A member shared a paper link that discusses how modern LLMs lose knowledge and reasoning capabilities when compressed. This research reports that knowledge is lost earlier than reasoning through context during compression stages.
   - *The findings urge a reconsideration of existing metrics assessing compressed model efficacy, focusing instead on more holistic evaluation tools.*
- **Discussion on Product Quantization Methods**: A user posed a question regarding the viability of product quantization methods for achieving compression ratios comparable to **BitNet** techniques. This remains a topic of debate, as others stated they hadn't yet formed a solid opinion on the methods' effectiveness.
   - *This discussion reflects ongoing exploration into alternative compression strategies within the community.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/teortaxesTex/status/1836448002971701435">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: Translation: «shit we have failed to recover L2-7B performance with L3-8B but we need to spin a paper».  Enough GPU-poor bait. Let BitNet die, or MSR to prove it. This idea only ever made sense for pr...</li><li><a href="https://arxiv.org/abs/2310.01382">Compressing LLMs: The Truth is Rarely Pure and Never Simple</a>: Despite their remarkable achievements, modern Large Language Models (LLMs) face exorbitant computational and memory footprints. Recently, several works have shown significant success in training-free ...</li><li><a href="https://arxiv.org/abs/2404.14047">An Empirical Study of LLaMA3 Quantization: From LLMs to MLLMs</a>: The LLaMA family has become one of the most powerful open-source Large Language Models (LLMs) and the popular LLM backbones of Multimodal Large Language Models (MLLMs), widely applied in Computer Visi...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1286043411547422730)** (17 messages🔥): 

> - `Hackathon Invitations`
> - `Access to Hack-Ideas Forum`
> - `Missing Users in Discord`
> - `GemLite-Triton Release` 


- **Hackathon Invitations in Progress**: Some members reported that while they received their hackathon invitations, their teammates did not, sparking inquiries about whether invitations are still rolling out.
   - A member shared teammate information, asking for help in confirming the invite status for their friend.
- **Access Issues to Hack-Ideas Forum**: A participant expressed frustration about not having access to the hack-ideas forum for the upcoming hackathon.
   - Other members attempted to assist or seek out further clarification regarding forum accessibility.
- **Missing Users in Discord Roles**: A few users who received their hackathon QR code were not listed under the `cuda-mode-irl` Discord group, prompting calls for those users to identify themselves.
   - Multiple members helped add missing users to the appropriate Discord role once they confirmed their statuses.
- **Introduction of GemLite-Triton**: A member announced the release of [GemLite-Triton](https://github.com/mobiusml/gemlite), a set of high-performance kernels for quantization and low-bit operations.
   - They encouraged attendees to leverage this project for related tasks and welcomed any questions about the release.



**Link mentioned**: <a href="https://github.com/mobiusml/gemlite">GitHub - mobiusml/gemlite: Simple and fast low-bit matmul kernels in CUDA / Triton</a>: Simple and fast low-bit matmul kernels in CUDA / Triton - mobiusml/gemlite

  

---


### **CUDA MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1286368930805452891)** (5 messages): 

> - `Apple Silicon MLX framework`
> - `On-device speech models`
> - `Metal and PyTorch interface` 


- **Apple Silicon MLX Designed for Performance**: The **MLX framework** is tailored for Apple computers, utilizing specialized methods for **autodiff**, **vmap**, and **jit compiling** to optimize performance.
   - As noted, it diverges from a general CPU/GPU model, employing unique kernels specifically for **Apple Silicon**.
- **Lazy Evaluation Techniques in MLX**: The framework leverages **lazy evaluation**, only computing results on distinct calls, which enhances overall performance.
   - This approach aligns with their architecture's design philosophy of maximized efficiency.
- **Metal Integration in MLX**: MLX integrates **Metal Performance Shaders** and custom backends like **'Steel'** for optimized rendering and computation.
   - This adaptation positions MLX more like a **PyTorch** interface than an analogous **Triton** setup.
- **Foundation Model for On-device Speech Conversations**: There's interest in developing an **on-device speech conversation** model specifically for **Apple Silicon** environments.
   - The foundational model aims to enhance user interactions through efficient processing capabilities.


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1286108741766483978)** (3 messages): 

> - `Story Generation Agent`
> - `LlamaParse Premium`
> - `RAG and agentic applications`
> - `Opik partnership` 


- **Build a Story Generation Agent with Human-in-the-Loop**: A member shared a [step-by-step guide](https://t.co/5ElRICjK0C) by @_nerdai_ on constructing an agent for dynamically generating 'choose-your-own-adventure' stories that incorporates human feedback at each step.
   - *This guide allows users to effectively shape the storytelling experience based on input and choices.*
- **LlamaParse Premium shines in document parsing**: The introduction of [LlamaParse Premium](https://t.co/8VTKKYWVOT) claims to enhance document parsing capabilities for LLM applications by integrating visual understanding of multimodal models with long text and table content extraction.
   - *This upgrade positions LlamaParse as the top choice for robust document processing.*
- **Streamlining RAG with Opik's Autologging**: A discussion highlighted that even basic RAG will involve several steps and that advanced agentic apps have even more complexities to manage.
   - Announcing a partnership with [Opik by @Cometml](https://t.co/Z3KdwjAKKv), which automates autologging for RAG/agent call tracing in both development and production environments, adds an exciting productivity boost.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1286048094785372271)** (51 messages🔥): 

> - `RAG and semantic search`
> - `Pinecone vector database issues`
> - `SchemaLLMPathExtractor usage`
> - `KG schema class requirements`
> - `Property graph index embedding problems` 


- **RAG discussions with semantic search**: A member is exploring how to manage questions and answers from vendors, contemplating the use of semantic search on documented responses.
   - Suggestions were made to generate varied questions from answers and utilize the vector store for better retrieval.
- **Challenges with Pinecone vector ID management**: Members discussed difficulties with Pinecone's auto-generated IDs, making it hard to delete documents based on specific metadata due to limitations in serverless indexes.
   - Alternative database options such as Chroma, Qdrant, Milvus, and Weaviate were recommended for potentially better support and integration.
- **Using kg_schema_cls in SchemaLLMPathExtractor**: A user sought guidance on how to use kg_schema_cls, leading to explanations regarding the need for a specific Pydantic class to represent a graph structure.
   - It was emphasized that field names must match specific schemas and users discussed potential validator issues when trying to create multiple instances.
- **Insertion of entities without embeddings**: One member reported that manually created nodes and relations in a property graph index were not receiving any embeddings, resulting in all queries scoring zero.
   - It was confirmed that embeddings need to be explicitly attached when initiating nodes, and concerns were raised about the limitations of the current graph store in handling vectors.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llamahub.ai/l/tools/llama-index-tools-tavily-research?from=">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">Starter Tutorial (Local Models) - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/existing_data/pinecone_existing_data/">Guide: Using Vector Store Index with Existing Pinecone Vector Store - LlamaIndex</a>: no description found</li><li><a href="https://docs.pinecone.io/guides/data/manage-rag-documents#delete-all-records-for-a-parent-document">Manage RAG documents - Pinecone Docs</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/#functiontool">Tools - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/26205a0e96d36382cd4a09432e51731ddb5170a1/llama-index-integrations/vector_stores/llama-index-vector-stores-pinecone/llama_index/vector_stores/pinecone/base.py#L170">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-pinecone/llama_index/vector_stores/pinecone/base.py at 26205a0e96d36382cd4a09432e51731ddb5170a1 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1286297287143854244)** (1 messages): 

> - `RAG`
> - `LlamaIndex` 


- **Concerns About RAG Article Depth**: A member expressed that the article on **RAG** is somewhat superficial, failing to provide a solid argument against the necessity of tools like **LlamaIndex**.
   - *It explains some basic concepts about RAG but doesn't elaborate on the drawbacks of the mentioned tools.*
- **Need for More In-Depth Analysis**: The discussion highlights a need for a more in-depth analysis of why tools such as **LlamaIndex** may not be needed in certain contexts.
   - Members indicated that the article could have benefitted from a detailed comparison and technical evaluation of alternatives.


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1286143200020402207)** (17 messages🔥): 

> - `Fish Speech`
> - `AdBot incidents` 


- **Fish Speech Outperforms Others**: A member noted that **Fish Speech** shows **zero shot voice cloning accuracy** that surpasses any other open model tried, especially in mimicking speech patterns.
   - Old audio from the **1940s** produces outputs that accurately replicate the **loudspeaker sound** from that era.
- **Fish Speech's Quirky Speech Patterns**: It was mentioned that Fish Speech randomly inserts words like *ahm* and *uhm* into audio, reflecting natural speech patterns without prompts.
   - Members agreed that this feature adds a realistic touch to the model's outputs.
- **AdBot Problems in the Server**: A member raised a concern about an **AdBot** detected in the server, describing it as spreading through multiple servers like **malware**.
   - Others chimed in, highlighting how the sorting mechanism led to the bot appearing at the top of member lists.



**Link mentioned**: <a href="https://huggingface.co/spaces/fishaudio/fish-speech-1">Fish Speech 1 - a Hugging Face Space by fishaudio</a>: no description found

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1286226163211173918)** (11 messages🔥): 

> - `Muse text to image`
> - `Open-source GPT-4o model`
> - `Dataset building for GPT-4o`
> - `LLM fine-tuning challenges`
> - `Tokenization issues in LLMs` 


- **Challenges in Muse text to image with COCO2017**: A member reported using [Muse text to image](https://github.com/lucidrains/muse-maskgit-pytorch) for training on **COCO2017**, but only received image outputs.
   - They expressed a need for guidance regarding their implementation challenges.
- **Collaboration on Open-source GPT-4o model**: Lengyue from Fish Audio announced they are developing an **open-source GPT-4o-like model** and are open to sharing data with LAION.
   - They believe collaboration could **accelerate progress** and exchange results and design ideas.
- **Building datasets for Open-source GPT-4o**: Another member mentioned they are involved in **building datasets** for an open-source GPT-4o.
   - There is a general sentiment that collaborative dataset creation could benefit the community.
- **Difficulty in fine-tuning Open Source LLMs**: Lengyue expressed concerns that **fine-tuning existing Open Source LLMs** isn't feasible for their goals and suggested starting training from scratch.
   - They highlighted that initial outputs from GPT-4o outperform others but raise issues with **hallucinations** during subsequent calls.
- **Tokenization problems in LLMs**: Lengyue proposed that current challenges in LLM outputs could be related to **tokenization issues**.
   - They believe addressing these issues could improve model performance and reliability.


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1286070710866804736)** (20 messages🔥): 

> - `Fal AI funding`
> - `OpenAI O1 improvements`
> - `Jina embeddings v3 launch`
> - `Runway and Lionsgate collaboration`
> - `New multi-agent research team at OpenAI` 


- **Fal AI secures $23M for growth**: Fal AI has raised **$23M** in Seed and Series A funding with a **$14M** Series A led by Kindred Ventures, including participation from Andreessen Horowitz.
   - *Gorkem Yurt* shared the news on [Twitter](https://x.com/gorkemyurt/status/1836488019924471953?s=46) alongside a detailed [blog post](https://blog.fal.ai/generative-media-needs-speed-fal-has-raised-23m-to-accelerate/) about their plans to accelerate generative media technology.
- **OpenAI O1 model enhancements**: OpenAI has increased the rate limits for the **o1** API, allowing **500** requests per minute for o1-preview and **1000** for o1-mini, as reported by *OpenAI Developers*.
   - The advancements support developers' needs and are part of a broader initiative to expand access to the o1 model, detailed in a [thread by Amir](https://x.com/amir/status/1836782911250735126?s=46).
- **Launch of Jina embeddings v3**: Jina AI announced the release of **jina-embeddings-v3**, a multilingual embedding model featuring **570M parameters** and **8192-token length**, outperforming proprietary models from OpenAI and Cohere.
   - The new model achieved notable rankings on the MTEB English leaderboard for models under 1B parameters, according to *Jina AI* on [Twitter](https://x.com/JinaAI_/status/1836388833698680949).
- **Runway partners with Lionsgate for Gen-3 Alpha**: Lionsgate has partnered with Runway to utilize its film catalog as training data for their model, Gen-3 Alpha, surprising many who expected Sora to achieve this first.
   - This development signals significant innovation in the industry, as highlighted by *Andrew Curran* on [Twitter](https://x.com/AndrewCurran_/status/1836411345786290535).
- **OpenAI launching multi-agent research team**: OpenAI is on the hunt for ML engineers to join a new multi-agent research team, emphasizing its potential to enhance AI reasoning.
   - As noted on Twitter by *Polynoamial*, prior experience isn't necessary, and interested candidates can apply via [this form](https://jobs.ashbyhq.com/openai/form/oai-multi-agent).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/hwchung27/status/1836842717302943774?s=46">Tweet from Hyung Won Chung (@hwchung27)</a>: Here is my talk at @MIT (after some delay😅)  I made this talk last year when I was thinking about a paradigm shift. This delayed posting is timely as we just released o1, which I believe is a new par...</li><li><a href="https://blog.fal.ai/generative-media-needs-speed-fal-has-raised-23m-to-accelerate/">Generative media needs speed. fal has raised $23M to accelerate.</a>: fal has raised $23M in Seed and Series A funding. The $14M Series A was led by Kindred Ventures with participation from Andreessen Horowitz and First Round Capital and angel investors including Perple...</li><li><a href="https://x.com/AndrewCurran_/status/1836411345786290535">Tweet from Andrew Curran (@AndrewCurran_)</a>: Big news for the industry, I thought Sora would get here first. Lionsgate has signed a deal with Runway to use its film catalog as training data for their model, Gen-3 Alpha. Lionsgate will use a besp...</li><li><a href="https://x.com/amir/status/1836782911250735126?s=46">Tweet from Amir Efrati (@amir)</a>: OpenAI’s ability to shrink its reasoning model without losing many capabilities might be as big a deal as the reasoning power itself.   https://www.theinformation.com/articles/openais-miniature-reason...</li><li><a href="https://x.com/JinaAI_/status/1836388833698680949">Tweet from Jina AI (@JinaAI_)</a>: Finally, jina-embeddings-v3 is here! A frontier multilingual embedding model with 570M parameters, 8192-token length, achieving SOTA performance on multilingual and long-context retrieval tasks. It ou...</li><li><a href="https://x.com/fiiiiiist/status/1836471413198459331?s=46">Tweet from Tim Fist (@fiiiiiist)</a>: This new WaPo article on the environmental impact of AI has been getting a lot of attention.  The key claim is that GPT-4 consumes 0.14kWh of energy to produce a 100-word email  Here&#39;s why this is...</li><li><a href="https://x.com/cognition_labs/status/1836866696797401118">Tweet from Cognition (@cognition_labs)</a>: Devin has become faster, more accurate with code edits, more reliable at following your instructions, and better at independent decision making. We’ve also improved our support for enterprise security...</li><li><a href="https://x.com/bo_wangbo/status/1836391316286038214">Tweet from Bo (@bo_wangbo)</a>: my personal favourite about jina-embeddings-v3 (beyond fancy features) is, we manually checked the common failures made by different text embedding models, created failure taxonomy, and try to fix the...</li><li><a href="https://x.com/gorkemyurt/status/1836488019924471953?s=46">Tweet from Gorkem Yurtseven (@gorkemyurt)</a>: we have some news to share!   https://blog.fal.ai/generative-media-needs-speed-fal-has-raised-23m-to-accelerate/</li><li><a href="https://x.com/aidan_mclau/status/1836796517463806263">Tweet from Aidan McLau (@aidan_mclau)</a>: fact check: incorrect.  o1-mini is not better because it thinks longer  it’s just a better model  thread  Quoting Amir Efrati (@amir)   OpenAI’s ability to shrink its reasoning model without losing ma...</li><li><a href="https://x.com/OpenAIDevs/status/1836506351062716701">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Just 5x&#39;d rate limits again:    o1-preview: 500 requests per minute  o1-mini: 1000 requests per minute  Quoting OpenAI Developers (@OpenAIDevs)   We&#39;ve increased OpenAI o1 API rate limits for ...</li><li><a href="https://x.com/polynoamial/status/1836872735668195636?s=61">Tweet from Noam Brown (@polynoamial)</a>: .@OpenAI is hiring ML engineers for a new multi-agent research team! We view multi-agent as a path to even better AI reasoning. Prior multi-agent experience isn&#39;t needed. If you&#39;d like to rese...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1286145849935528037)** (4 messages): 

> - `NeurIPS 2024 preparations`
> - `Accommodation logistics`
> - `Vancouver event updates` 


- **NeurIPS 2024 Channel Created**: Following popular demand, a dedicated channel for **NeurIPS 2024** has been created to keep participants informed about the event.
   - Members are encouraged to reply in the channel to stay updated on everything happening in Vancouver this December.
- **House Booking for NeurIPS**: An organizer is looking into booking a house for the duration of **NeurIPS 2024**, requesting participants to mention if they are interested.
   - It's emphasized that contributors should expect to chip in for the entire week's stay, rather than just a couple of days.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1286151710062149745)** (14 messages🔥): 

> - `Cohere RAG API`
> - `Client Design Success`
> - `504 Gateway Timeout Errors`
> - `Learning AI with Cohere`
> - `Community Engagement` 


- **Building an Expert AI with RAG API**: One member is currently using Cohere's **RAG API** to develop an expert AI focused on a small gaming niche.
   - They expressed excitement about leveraging the API for niche applications.
- **Client Loves the Design!**: A member successfully convinced a client of the value of their designs, stating, *'my designs are so cool and they need it.'*
   - This prompted a supportive response from the community, celebrating the win.
- **Experiencing 504 Gateway Timeout Errors**: A member raised concerns about receiving **504 Gateway Timeout** errors from Python SDK **client.chat** calls that are taking a considerable amount of time.
   - Discussion revealed community members are also facing similar issues and looking for solutions.
- **Learning through Application**: Another member encouraged grabbing a trial key for **1000 free API calls** per month, stating it's a great way to learn.
   - In agreement, another member emphasized that hands-on application is the best way to learn about AI.
- **Welcoming New Community Members**: The community welcomed a new member, expressing happiness about their presence in the group.
   - This demonstrates the supportive nature of the community and encourages engagement.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1286117295265153024)** (7 messages): 

> - `Command pricing`
> - `Dataset access for Aya-101`
> - `Using Command for tagging`
> - `Efficiency of new models` 


- **Command Pricing Clarification**: Members discussed the costs associated with the **Command** version, noting it is around **$1.00 for 1M tokens** for input and **$2.00 for output**.
   - Although pricing was provided, the consensus is to transition to **Command-R** or **Command-R+** for better performance and cost-efficiency.
- **Inquiries about Aya-101 Dataset**: A member asked about accessing the dataset that **Aya-101** is fine-tuned on, specifically the **ShareGPT-Command** dataset.
   - There's no direct response about dataset access, but ongoing curiosity about the dataset remains.
- **Command Usage for Feedback Tagging**: A member explained they use **Command** to categorize short feedback snippets but need it to create new categories as needed.
   - Another member suggested trying **Command-R** or **Command-R+** for improved functionality in tagging.
- **Performance vs. Economical Options**: Discussions highlighted that using older models like **Command** is not economical compared to newer, more performant options.
   - Members suggested that newer models are not only smarter but also more cost-effective for various use cases.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1286356089851613275)** (2 messages): 

> - `Rerank Fine-Tuning`
> - `RAG Results Impact` 


- **Inconsistencies with multilingual rerank**: A user reported strange results when using **_rerank_multilingual_v3_** on English text, receiving scores of **<0.05** on a question similar to the content and **<0.18** on a very similar question.
   - In contrast, switching to **_rerank_english_v3_** yielded much higher scores of **0.57** and **0.98** for similar queries, raising questions about the model's effectiveness.
- **German vs English rerank performance**: When using **_multilingual_v3_** on German text, the user noted a score of **0.66** compared to **0.99** with **_english_v3_**.
   - This inconsistency is significantly impacting the user's **RAG results**, as it filters out all relevant chunks, leading to concerns about the rerank models.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1286039874138406962)** (22 messages🔥): 

> - `OpenAI o1 models`
> - `Knowledge Cutoff`
> - `Qwen 2.5 72B Performance`
> - `Livecodebench Comparison`
> - `AI Reasoning Ability` 


- **OpenAI o1 Models Impress**: @DeryaTR_ reported that after testing the **o1-mini** model for PhD-level projects, it is comparable to an **outstanding PhD student** in biomedical sciences.
   - This model is considered among the best trained by the user, showcasing its potential in academic applications.
- **Knowledge Cutoff Haunts Developers**: A member pointed out that the **knowledge cutoff is October 23**, impacting the utility of the AI in handling newer developments.
   - This limitation frustrates users like @xeophon, who mentioned the challenges it presents while coding.
- **Qwen 2.5 72B Takes the Lead**: Qwen 2.5 72B has emerged as a new leader in **open weights intelligence**, topping independent evaluations against larger models like **Llama 3.1 405B**.
   - While it trails slightly in MMLU, it excels in **coding and math**, providing a *cheaper alternative* with a dense model and 128k context window.
- **Livecodebench Shows Strength**: Members noted that the **livecodebench** numbers are impressive as they are reported to match those of **Sonnet**, using timeless Leetcode questions.
   - However, the actual coding usage by members reveals limitations, particularly with new library releases that are unknown to o1.
- **Discussions on AI's Reasoning Ability**: Discussion emerged around the reasoning abilities of models like o1-mini, often compared to Qwen 2.5 in terms of performance on tasks without reflection-type methods.
   - Users expressed optimism about future performance improvements with further enhancements despite current comparisons showing o1's superiority.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/DeryaTR_/status/1836434726774526381">Tweet from Derya Unutmaz, MD (@DeryaTR_)</a>: In the past few days, I’ve been testing OpenAI o1 models, mostly o1-mini, for developing PhD or postdoc level projects. I can confidently claim that the o1 model is comparable to an outstanding PhD st...</li><li><a href="https://x.com/HaveFunWithAI/status/1836749726554702027">Tweet from HaveFunWithAI (@HaveFunWithAI)</a>: o1-mini is good at math  for reference:  qwen2.4-math-72b-instruct (just announced, sota open source math model) is not better than o1-mini with code execution and ensemble methods (n=256) https://qwe...</li><li><a href="https://x.com/artificialanlys/status/1836822858695139523?s=46">Tweet from Artificial Analysis (@ArtificialAnlys)</a>: There is a new leader in open weights intelligence! Qwen2.5 72B tops our independent evals amongst open weights models, including compared to the much larger Llama 3.1 405B  Qwen 2.5 72B released yest...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

SnailBot News: <@&1216534966205284433>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1286062936984653846)** (6 messages): 

> - `OpenInterpreter error handling`
> - `Agent performance testing`
> - `OpenInterpreter processing capabilities`
> - `Model performance comparison`
> - `Task success stories with OpenInterpreter` 


- **Navigating OpenInterpreter Error Messages**: A user requested assistance with an error encountered while inputting data into OpenInterpreter, expressing hope for a walkthrough to resolve the issue.
   - *Send me a dm of the error* was suggested to help identify the problem.
- **Active Agent Evaluation**: Another user shared their consistent usage of OpenInterpreter's agent for about a week, indicating engagement in hands-on testing.
   - This highlights the ongoing exploration and evaluation of agent performance among users in the community.
- **Clarifying OpenInterpreter's Processing Functions**: A member discussed a client's inquiries on how OpenInterpreter processes information independently versus when it relies on Chat GPT.
   - Concerns were raised about the efficiency of connecting to Chat GPT given its CPU usage, with a user seeking community thoughts on the matter.
- **Comparing Model Performance in OpenInterpreter**: A user discussed their experience with various models, ultimately finding **microsoft/Wizardlm 8x22B** to outperform other choices like **llama 405B**.
   - They noted that Wizardlm achieved task completions on the first attempt more often than previous models.
- **Success Stories Using OpenInterpreter**: The same user shared their successful experiences with OpenInterpreter, such as categorizing large amounts of files and creating a desktop shortcut.
   - They encouraged others to share the types of tasks they have successfully accomplished with the tool, promoting exchange of ideas within the community.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1286113795172864041)** (6 messages): 

> - `Perplexity browser settings`
> - `User experiences with browser`
> - `Windows and Edge compatibility` 


- **Users question Perplexity as default browser**: A user asked if **Perplexity** is set as the default browser in Chrome or other browsers.
   - Another user confirmed that it is **not** set as their default.
- **Multiple users facing similar issues**: A user mentioned almost **20 others** experienced the same issue with **01**.
   - They confirmed this by asking several users within the **01** community.
- **Browser usage context during issues**: One user noted they were using **Edge** when encountering the issue.
   - This highlights that issues may vary between different browsers on **Windows**.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1286067264583372830)** (5 messages): 

> - `RAG chat app development`
> - `Multimodal models for PDF interaction`
> - `Image and text integration in responses`
> - `AI use case examples`
> - `AI-generated content` 


- **Developing a RAG Chat App for PDFs**: A member is working on a **RAG chat app** that allows users to interact with specific **PDF documents** and is seeking advice on how to handle responses containing both text and images.
   - Another member suggested using **tokens** for image paths and summarizing images to save context tokens while integrating with the existing PDF to text parser.
- **Image Context Handling in PDF Interactions**: Discussion highlighted that the **PDF to text parser** might need to include image links or tokens to retrieve context and images effectively.
   - Another approach discussed involved using base64 image encoding and extracting text directly from images for better context integration.
- **Impressive AI Creation by Google**: One member shared a link to a creation made with AI in just **10 seconds** and praised a new feature from **GoogleAI** as highly impressive.
   - A follow-up comment emphasized its **utility** and potential as a top AI use case, commending Google for their work.
- **Suggestions for Multimodal Model Usage**: A member recommended using a **multimodal model** that can read both images and text directly, suggesting that extracting text from images is straightforward.
   - This approach offers an **easy testing method** for integrating image data into conversational responses.



**Link mentioned**: <a href="https://x.com/sunglassesface/status/1836527799470854557">Tweet from 😎orlie (@sunglassesface)</a>: Google finally came out with something impressive. I just tested it... It&#39;s wild.  I made this in 10 seconds.  Quoting Wolfram Ravenwolf 🐺🐦‍⬛ (@WolframRvnwlf)   Agreed. This is both very impress...

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1286123402041495593)** (10 messages🔥): 

> - `OBS Screen Recording`
> - `Screenity Recorder`
> - `Moshi Speech Models`
> - `GRIN MoE` 


- **OBS Remains the Go-To for Screen Recording**: Members discussed the use of **OBS** as a robust option for screen recording, despite some expressing a preference for easier solutions.
   - One member emphasized their consistent use of OBS, leaving others to seek alternatives that feature easy zooming effects.
- **Screenity Emerges as a User-Friendly Alternative**: A user shared [Screenity](https://github.com/alyssaxuu/screenity), a free and privacy-friendly screen recorder that captures both screen and camera.
   - This tool aims to offer a more accessible solution compared to OBS, catering to those seeking simplicity in recording software.
- **Moshi Models Debut for Speech-to-Speech Applications**: Members announced the release of the **Moshi** speech-to-speech models, which facilitate full-duplex spoken dialogue and align text tokens to audio.
   - This foundation model boasts unique features like modeling conversation dynamics and is implemented in a Pytorch version quantized in bf16 precision.
- **GRIN MoE Shows Promise with Fewer Parameters**: Discussion surfaced around [GRIN MoE](https://huggingface.co/microsoft/GRIN-MoE), which achieves impressive performance with **only 6.6B active parameters**, especially in coding and mathematics tasks.
   - It leverages **SparseMixer-v2** for gradient estimation while eliminating both expert parallelism and token dropping, setting it apart from traditional MoE training methods.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/kyutai/moshiko-pytorch-bf16">kyutai/moshiko-pytorch-bf16 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/GRIN-MoE">microsoft/GRIN-MoE · Hugging Face</a>: no description found</li><li><a href="https://github.com/alyssaxuu/screenity">GitHub - alyssaxuu/screenity: The free and privacy-friendly screen recorder with no limits 🎥</a>: The free and privacy-friendly screen recorder with no limits 🎥 - alyssaxuu/screenity
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1286234434625994795)** (3 messages): 

> - `Gemma2 DPO Config Issue`
> - `Chat Template Modification` 


- **Gemma2 fails to run with DPO data**: A user shared a configuration for **Gemma2 9b** used with **DPO data**, but encountered a **TemplateError** saying, *'Conversation roles must alternate user/assistant/user/assistant...'*.
   - The error occurred when trying to apply a chat template due to the dataset structure having 'prompt' instead of 'conversation'.
- **Modifications to the Gemma chat template**: The user modified the **Gemma** template in `chat_templates.py` to change the role-checking logic, attempting to resolve the template error.
   - They altered the code to initiate a loop on message roles, questioning if the modification was appropriate.


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1286143117677953085)** (1 messages): 

> - `PyTorch conference`
> - `New work announcements`
> - `Community engagement` 


- **Welcoming the PyTorch Conference Attendees**: A jovial welcome was extended to everyone joining from the **PyTorch conference**.
   - Participants are encouraged to ask questions in the designated channel and interact with others.
- **Check Out New Work**: Attendees are invited to check out the new work being developed in channel <#1236040539409879170>.
   - This highlights ongoing efforts and innovations emerging from the community.
- **Encouragement for Community Interaction**: Members were encouraged to feel free to message in the channel with any questions they may have.
   - This promotes **community engagement** and support among attendees.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1286043635888164904)** (8 messages🔥): 

> - `Conference Livestream Query`
> - `GitHub PR for kv-caching Fix`
> - `HH RLHF Dataset Documentation`
> - `Default Preference Dataset Builder` 


- **Inquiry on Conference Livestream**: Members discussed the possibility of a **conference livestream**, with one expressing uncertainty about its availability.
   - *‘Idk :/’* was the response to the inquiry, indicating a lack of information.
- **GitHub PR Addresses kv-Caching Issues**: A pull request titled **Fix kv-cacheing and bsz > 1 in eval recipe** was linked, detailing changes by [SalmanMohammadi](https://github.com/pytorch/torchtune/pull/1622).
   - The PR aims to address issues related to kv-caching and is noted as significant within the context of ongoing developments.
- **Discussion on HH RLHF Dataset Exposition**: A member questioned why the **HH RLHF dataset** is not documented, suggesting it should be the standard preference example instead of another option.
   - *‘Not sure, it should be exposed...’* was the sentiment that encouraged inclusion of the dataset in documentation.
- **Future Plans for Default Preference Dataset Builder**: Plans were shared about creating a **default preference dataset builder** that will assume **ChosenToRejectedMessages**.
   - The collaboration was met with enthusiasm, indicating strong support for this development with the comment *‘Dope’*.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/1622">Fix kv-cacheing and bsz &gt; 1 in eval recipe by SalmanMohammadi · Pull Request #1622 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (please add here)  Please link to any issues this PR addresses. closes #160...

  

---



### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1286315655674462311)** (5 messages): 

> - `DSPy Program Optimization`
> - `Bootstrapping in DSPy`
> - `Cost of Prompt Optimization`
> - `MIPRO and Optimization`
> - `Non-Determinism in LLMs` 


- **Excitement about DSPy Program Optimization**: A member shared their success after two months of coding, highlighting the effectiveness of the **BSFSWRS optimizer** in their complex LM setup.
   - *The future is bright, people!*
- **Concerns about the Cost of Optimizing Prompts**: Another member remarked on the potential high costs associated with optimizing prompts for DSPy, indicating it might be a significant investment.
   - *That's gotta be hella expensive to optimize a prompt.*
- **MIPRO: A Costly Adventure**: A humorous suggestion was made to try using **o1 with MIPRO**, with a tongue-in-cheek warning about the financial risks involved.
   - *Certified way to go bankrupt.*
- **Clarifying Bootstrapping in DSPy**: One member inquired about bootstrapping, which is aimed at generating examples of steps within a pipeline and validating the success of the process.
   - They expressed confusion regarding how this method functions considering the non-deterministic nature of **LLMs**.
- **Understanding Bootstrapping Responses**: A member affirmed that bootstrapping generates intermediate examples and validates their correctness through the success of the final prediction.
   - They noted that if the final result is correct, the intermediate steps are assumed valid for use as few-shot examples.


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1286392361928097792)** (4 messages): 

> - `tinybox motherboard`
> - `CLANG bounty`
> - `Pull Request discussions` 


- **Inquiry about tinybox motherboards**: A user asked about the specific **motherboard** used in **tinybox red and green** models, seeking clarification on hardware details.
   - This reflects ongoing interest in the hardware aspects of **tinybox** devices among users.
- **Discussion on CLANG bounty**: A member inquired if the bounty titled 'Replace CLANG dlopen with mmap + remove linker step' involves manually handling **relocations** in the object file.
   - This indicates a deeper technical exploration of CLANG's integration within the **tinygrad** environment.
- **Links to relevant Pull Requests**: A user shared links to **Pull Request #6299** and **#4492**, discussing the replacement of **dlopen** with **mmap** and the implementation of **Clang jit**.
   - These contributions aim to enhance performance, particularly on **M1 Apple devices**, showcasing community efforts in optimizing code execution.
- **Curiosity about bounty outcomes**: A user expressed excitement about who might claim the **bounty** for the CLANG changes, highlighting community engagement.
   - This interaction reflects a collaborative atmosphere where users are keen to witness contributors' results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/pull/6299">Replace dlopen with mmap in CLANG by christopherm99 · Pull Request #6299 · tinygrad/tinygrad</a>: Performance Tested on an M1 MacBook Pro. from tinygrad.runtime.ops_clang import ClangProgram  with open(&amp;quot;test.o&amp;quot;, &amp;quot;rb&amp;quot;) as f: lib = f.read() for _ in range(1000): C...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/4492">Clang jit by uuuvn · Pull Request #4492 · tinygrad/tinygrad</a>: no description found
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1286283907553165343)** (1 messages): 

> - `OpenAI's o1 model`
> - `Large reasoning models` 


- **OpenAI's o1 model garners attention**: A [YouTube video](https://m.youtube.com/watch?v=KKF7kL0pGc4) titled 'o1 - What is Going On? Why o1 is a 3rd Paradigm of Model + 10 Things You Might Not Know' offers an engaging summary of how **OpenAI's o1** may have been built.
   - *Even skeptics are calling it a 'large reasoning model'* due to its distinctive approach and impact on future model development.
- **o1's differentiation from other models**: In the same video, it discusses why **o1** is being recognized as a new paradigm in AI modeling, indicating significant shifts in design philosophy.
   - The implications of adopting such models can lead to a better understanding of reasoning capabilities in AI, making it a critical topic in the field.



**Link mentioned**: <a href="https://m.youtube.com/watch?v=KKF7kL0pGc4">o1 - What is Going On? Why o1 is a 3rd Paradigm of Model + 10 Things You Might Not Know</a>: o1 is different, and even sceptics are calling it a &#39;large reasoning model&#39;. But why is it so different, and why does that say about the future? When models ...

  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1286334772737736734)** (1 messages): 

> - `LunoSmart AI Venture`
> - `Tech Stack for Application`
> - `Cross Platform Development`
> - `Machine Learning Expertise` 


- **LunoSmart Launch and Offerings**: Kosi Nzube announced the launch of his AI venture, [LunoSmart](https://www.lunosmart.com), showcasing its focus on AI-driven applications and innovative solutions.
   - This venture aims to deliver **seamless**, **efficient**, and **intelligent experiences** across various platforms.
- **Diverse Tech Stack Overview**: The tech stack for Kosi's applications includes **Java**, **Flutter**, **Spring Boot**, **Firebase**, and **Keras**, emphasizing a modern development approach.
   - An application is available on both Android and web platforms, allowing for wide accessibility.
- **Expertise in Cross Platform Development**: Kosi is skilled in cross-platform development using **Flutter** and the **Firebase SDK**, enhancing functionality across devices.
   - His experience with mobile apps specifically highlights his proficiency in creating **native Android** applications with **Android Studio** and **Java**.
- **Machine Learning Skillset**: With a solid background in **Machine Learning**, Kosi utilizes tools like **Keras**, **Weka**, and **DL4J** for creating intelligent models.
   - His experience in this field started in **2019**, showcasing his commitment to advancing AI technologies.



**Link mentioned**: <a href="https://kosinzube.online/">Kosi Nzube</a>: ai developer. I program with my favorite tools: Java, Flutter and Keras with Python. Founder@ lunosmart.com

  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/)** (1 messages): 

flozi00: https://mistral.ai/news/september-24-release/

Mistral follows with a price drop 💪
  

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

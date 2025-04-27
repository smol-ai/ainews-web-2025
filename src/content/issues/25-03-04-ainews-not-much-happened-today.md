---
id: 837d100e-af29-435c-920a-4322c40d0423
title: not much happened today
date: '2025-03-05T05:17:34.368145Z'
original_slug: ainews-not-much-happened-today-4193
description: >-
  **Weights and Biases** announced a **$1.7 billion acquisition by CoreWeave**
  ahead of CoreWeave's IPO. **CohereForAI** released the **Aya Vision models (8B
  and 32B parameters)** supporting **23 languages**, outperforming larger models
  like **Llama-3.2 90B Vision** and **Molmo 72B**. **Microsoft** introduced
  **Phi-4-Mini (3.8B parameters)** and **Phi-4-Multimodal models**, excelling in
  math, coding, and multimodal benchmarks. **CogView4**, a **6B parameter
  text-to-image model** with **2048x2048 resolution** and Apache 2.0 license,
  was released. **Alibaba** launched **Wan 2.1**, an open-source video
  generation model with **720p output** and **16 fps generation**. **Google**
  announced new AI features for Pixel devices including **Scam Detection** and
  **Gemini integrations**. **LlamaCloud** reached **General Availability** and
  raised **$19M Series A funding**, serving over **100 Fortune 500 companies**.
  **Weaviate** launched the **Query Agent**, the first of three Weaviate Agents.
companies:
  - weights-and-biases
  - coreweave
  - cohereforai
  - microsoft
  - alibaba
  - google
  - llamaindex
  - weaviate
models:
  - aya-vision-8b
  - aya-vision-32b
  - llama-3-2-90b-vision
  - molmo-72b
  - phi-4-mini
  - phi-4-multimodal
  - cogview4
  - wan-2-1
topics:
  - multilinguality
  - vision
  - multimodality
  - image-generation
  - video-generation
  - model-releases
  - benchmarking
  - funding
  - agentic-ai
  - model-performance
people:
  - mervenoyann
  - reach_vb
  - jayalammar
  - sarahookr
  - aidangomez
  - nickfrosst
  - dair_ai
  - akhaliq
  - bobvanluijt
  - jerryjliu0
---


<!-- buttondown-editor-mode: plaintext -->**Weave is all you need.**

> AI News for 3/4/2025-3/5/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**227** channels, and **2895** messages) for you. Estimated reading time saved (at 200wpm): **327 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Congrats to Weights and Biases for [their $1.7b acquisition to the soon-IPOing CoreWeave](https://x.com/weights_biases/status/1897085419239702821).


---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**Model Releases and Updates**

- **Aya Vision models, including 8B and 32B parameters, covering 23 languages have been released by CohereForAI**. [@mervenoyann](https://twitter.com/mervenoyann/status/1896924022438588768) announced the **Aya-Vision VLM family based on SigLIP and Aya**, outperforming larger models, and supporting image captioning, visual question answering, and text generation. [@reach_vb](https://twitter.com/reach_vb/status/1896924646412370373) detailed that the **32B model outperforms models 2x its size** like **Llama-3.2 90B Vision** and **Molmo 72B**, and the **8B model beats competitors by up to 81% win rates**.  [@JayAlammar](https://twitter.com/JayAlammar/status/1896966755395756268) highlighted the **Arabic language support** and availability in **32B and 8B sizes** with open weights for download. [@sarahookr](https://twitter.com/sarahookr/status/1896953483913498722) expressed pride in the release, emphasizing its efficiency, accessibility, and global reach.  [@aidangomez](https://twitter.com/aidangomez/status/1896946200135495708) simply stated "Aya can see now!".  [@nickfrosst](https://twitter.com/nickfrosst/status/1896948730622075051) announced the **new SOTA for multilingual vision with Aya-Vision-32B**, and in another tweet, [@nickfrosst](https://twitter.com/nickfrosst/status/1896935581386682827) stated that **Aya 32B outperforms Llama 90B and Qwen 72b**.
- **Phi-4-Mini (3.8B parameters) and Phi-4-Multimodal models have been introduced by Microsoft**, aiming to match or surpass larger open-source LLMs in math, coding, and multimodal tasks. [@dair_ai](https://twitter.com/dair_ai/status/1896930134583918860) summarized key features from the technical report, including **carefully curated data**, **Mixture-of-LoRAs for multimodality**, and **outperforming similar-size models** on benchmarks like MMLU, HumanEval, MBPP, GSM8K, and MATH. [@reach_vb](https://twitter.com/reach_vb/status/1897014754943910266) proclaimed **Phi 4 Multimodal** as the **new king of the Open ASR Leaderboard**, beating Nvidia Canary and OpenAI Whisper.
- **CogView4, a new 6B parameter text-to-image model with native 2048x2048 resolution and Apache 2.0 license, has been released**. [@multimodalart](https://twitter.com/multimodalart/status/1896844887733174371) announced the release with excitement, highlighting its features like **great prompt adherence for long prompts**. [@ostrisai](https://twitter.com/ostrisai/status/1896845726539513930) added CogView4 to AI Toolkit at 2 am after its release.
- **Wan 2.1, a new open-source video generation model from Alibaba, is now leading in the Artificial Analysis Video Arena**. [@_akhaliq](https://twitter.com/_akhaliq/status/1896973618275606719) detailed key features, including **720p output for the 14B model**, **16 fps generation**, and **multilingual text input**. [@_akhaliq](https://twitter.com/_akhaliq/status/1896994597185970628) shared that **Wan 2.1 is available on Hugging Face via Replicate**.

**Company and Product Announcements**

- **Google announced new AI features for Pixel devices**, including updated Scam Detection, more Gemini integrations, and connectivity improvements. [@Google](https://twitter.com/Google/status/1896982180292657182) officially announced the **first Pixel Drop of the year** with these updates.  [@Google](https://twitter.com/Google/status/1897022559855558727) also shared a recap of their **biggest AI announcements from the previous month**, from Deep Research in Gemini mobile app to job seeker tools.
- **LlamaCloud has reached General Availability and raised $19M in Series A funding**. [@llama_index](https://twitter.com/llama_index/status/1896967296633201085) announced the GA of LlamaCloud, a **turn-key solution for agentic knowledge management** and a **$19M Series A funding round** led by NorwestVP. [@jerryjliu0](https://twitter.com/jerryjliu0/status/1896970208071573622) further elaborated that LlamaCloud is now GA with **100+ F500s and 100K+ signups** already, and LlamaIndex is now an **agents framework**.
- **Weaviate launched the Query Agent, the first of three Weaviate Agents**. [@bobvanluijt](https://twitter.com/bobvanluijt/status/1896986983479828489) announced the **Query Agent launch**, emphasizing its role in **Generative Feedback Loops** and database-centric agents, and highlighted that it is **free to try on Weaviate Cloud**.
- **Perplexity AI is powering Telekom's 'AI Phone' and launching Perplexity Android Assistant**. [@AravSrinivas](https://twitter.com/AravSrinivas/status/1896822532801691775) clarified that Perplexity is not building new hardware but providing the **Perplexity Assistant as a native Android OS AI** on DT's AI Phone. [@AravSrinivas](https://twitter.com/AravSrinivas/status/1896677026297479229) stated that **Perplexity Android Assistant is the only reliably working agent** compared to promised buzzworded agents.

**Research and Papers**

- **DiffRhythm, an open-weights end-to-end full song generation model, was introduced, generating 1-2 minute songs in under 20 seconds**. [@multimodalart](https://twitter.com/multimodalart/status/1896862125659988322) highlighted the model's speed and capability to generate full songs with lyrics quickly. [@_akhaliq](https://twitter.com/_akhaliq/status/1896938481911542002) called it "wild" and stated that "open suno/udio is here".
- **MASK, a benchmark of 1,000+ scenarios to measure AI honesty, was released**. [@DanHendrycks](https://twitter.com/DanHendrycks/status/1896972178387841140) announced the release, noting findings that **some AI systems lie more readily under pressure**.
- **Coconut (Chain of Continuous Thought), a new method from Meta and UC San Diego, improves LLMs by using vector representations instead of text-based chains of thought for reasoning**. [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1896999082943529071) summarized the paper, explaining that **Coconut encodes richer reasoning paths with continuous vectors**, making them more efficient and accurate.
- **Research on reasoning LLM efficiency explores the relationship between reasoning length and model performance**. [@omarsar0](https://twitter.com/omarsar0/status/1896939453069074907) summarized a paper investigating how LLMs balance chain-of-thought (CoT) reasoning length against accuracy, highlighting findings like **universal accuracy–length trade-off** and **token complexity as a threshold**.

**Tools and Frameworks**

- **LangChain announced LangGraph BigTool and LangGraph.js Swarm libraries**. [@LangChainAI](https://twitter.com/LangChainAI/status/1897021019203711338) introduced **LangGraph BigTool**, a Python library for creating agents with scalable access to hundreds or thousands of tools. [@LangChainAI](https://twitter.com/LangChainAI/status/1896963664550240653) also announced **LangGraph.js Swarm**, a JavaScript library for building swarm-style multi-agent systems.
- **Weaviate launched Query Agent**, as mentioned above in company announcements, which functions as a tool for querying databases with function calling.

**Performance and Benchmarks**

- **Grok-3 is reported to have topped the Arena leaderboard**. [@lmarena_ai](https://twitter.com/lmarena_ai/status/1896675400916566357) announced that **xAI's latest Grok-3 model is tied for #1 overall** on the Arena leaderboard, and across Hard Prompts, Coding, Math, Creative Writing, Instruction Following, and Longer Query. [@omarsar0](https://twitter.com/omarsar0/status/1896676260312670589) noted that both **GPT-4.5 and Grok 3 are fun models to use**.  [@lateinteraction](https://twitter.com/lateinteraction/status/1896682075585220737) questioned **why frontier labs celebrate small margin wins** like a +0.6% improvement.
- **Aya Vision models are benchmarked as outperforming competitors**.  As mentioned in "Model Releases," the Aya Vision models are reported to outperform models like Llama 90B, Qwen 72B, and Gemini Flash 1.5.

**Humor/Memes**

- **Discussion around the capabilities and humor of GPT-4.5 and Grok**. [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1896970307468190030) joked that **GPT 4.5 is the only AI that gives him abs from laughing**, also stating that **GPT 4.5 beats 99% of shitposters on X**. [@omarsar0](https://twitter.com/omarsar0/status/1896676260312670589) mentioned that **GPT-4.5 and Grok 3 are fun models**.
- **iPhone 15 action button mapped to GPT-4.5 is considered a significant upgrade**. [@aidan_mclau](https://twitter.com/aidan_mclau/status/1896974341881126941) humorously stated that the biggest iPhone 12 to iPhone 15 upgrade was mapping the action button to GPT-4.5.
- **Catgirls and Jokercoin memes from @nearcyan**. [@nearcyan](https://twitter.com/nearcyan/status/1897015641904935265) jokingly claimed that **catgirls are easy to create**.  [@nearcyan](https://twitter.com/nearcyan/status/1897014010039632164) lamented about running out of "jokercoin" to become the Joker.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Qwen 32b Coder instruct improvements drive agent capabilities**

- **[Qwen 32b coder instruct can now drive a coding agent fairly well](https://v.redd.it/c2000d3tolme1)** ([Score: 461, Comments: 61](https://reddit.com/r/LocalLLaMA/comments/1j32p97/qwen_32b_coder_instruct_can_now_drive_a_coding/)): **Qwen 32b coder instruct** has been reported to effectively drive a coding agent, demonstrating its capability in facilitating coding tasks. Further details or examples from the video are not provided in the post.
  - **Hardware Requirements and Setup**: Running **Qwen 32b coder instruct** with **AWQ quantization** requires a minimum of **32GB VRAM** for a **30k context length**. Users discussed installation issues and hardware configurations, suggesting a **5090 GPU** might be necessary, and shared links for configuration guidance ([ra-aid.ai quickstart](https://docs.ra-aid.ai/quickstart/open-models)).
  - **Capabilities and Comparisons**: The model's ability to drive a coding agent through multi-step processes, including research, planning, and compiling, was highlighted as significant, despite the simplicity of the spinning cube demo. There was interest in seeing more complex tasks, like setting up a REST API, and comparisons with other AI tools.
  - **Community Engagement and Development**: The project is actively developed with recent optimizations for small models, and the repository is open for contributions ([GitHub link](https://github.com/ai-christianson/RA.Aid)). There is interest in integrating alternative solutions like **ollama** and potential comparisons with other tools like **aider**.


- **Is qwen 2.5 coder still the best?** ([Score: 174, Comments: 90](https://reddit.com/r/LocalLLaMA/comments/1j2usb0/is_qwen_25_coder_still_the_best/)): **Qwen 2.5 coder** is questioned for its current standing as the best coding model with **32 billion parameters** or fewer, asking if any superior models have been released since its introduction.
  - **Phi-4-25B and Deepseek** are mentioned as competitive alternatives to **Qwen 2.5 Coder 32B** for coding, with **Phi-4-25B** noted for its speed and effectiveness on simpler tasks. **Deepseek** is highlighted for its strength, but the **Qwen-Coder 32B** remains unmatched for local use on modest hardware.
  - Discussion on **reasoning capabilities** suggests that models like **R1-Distill-Qwen2.5-32B** and other reasoning models may outperform Qwen 2.5 in some cases but suffer from significantly longer processing times, making them less practical for frequent use.
  - There is interest in the potential of upcoming models like **Gemma 3** and concerns about hardware requirements, with users discussing the benefits of using **NVIDIA 3090 GPUs** for better performance. **Prompt engineering** and managing context effectively are also noted as crucial for optimizing model use.


**Theme 2. NVIDIA GeForce RTX 4090 with 96GB VRAM for AI Workloads**

- **NVIDIA’s GeForce RTX 4090 With 96GB VRAM Reportedly Exists; The GPU May Enter Mass Production Soon, Targeting AI Workloads.** ([Score: 223, Comments: 95](https://reddit.com/r/LocalLLaMA/comments/1j3gahy/nvidias_geforce_rtx_4090_with_96gb_vram/)): NVIDIA is reportedly considering the production of a **GeForce RTX 4090** with **96GB VRAM**, aimed at AI workloads, with a potential price around **$6,000**. While the 96GB version might not guarantee stability, it could be available in **3-4 months**, though factories are currently focused on the **48GB edition** due to cost considerations.
  - Many users clarify that the **96GB VRAM RTX 4090** is not an official **NVIDIA** product but rather a result of individuals modifying existing **4090 GPUs** by replacing VRAM chips, which may require a hacked driver to function properly. This practice has been seen before with similar modifications in the GPU market.
  - Discussions highlight the potential power consumption and cost of the modified cards, with estimates around **$6,000** for an unstable version, and some skepticism about the feasibility and stability of such modifications. Users compare the pricing and specifications with **NVIDIA's** professional-grade cards like the **L40** and **A40**, noting the significant bandwidth and VRAM differences.
  - There is a debate on **NVIDIA's** strategy regarding consumer versus data center markets, with some users suggesting that **NVIDIA** prioritizes high-margin data center sales over consumer demands for more VRAM. This is evidenced by humorous dialogues about internal decision-making, illustrating the tension between consumer needs and corporate profitability.


**Theme 3. DiffRhythm: Fast Song Generation with Diffusion Models**

- **DiffRhythm - ASLP-lab: generate full songs (4 min) with vocals** ([Score: 137, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1j38499/diffrhythm_aslplab_generate_full_songs_4_min_with/)): **DiffRhythm** by **ASLP-lab** is an AI tool for generating full-length songs, including vocals, using latent diffusion. Access the tool on [Hugging Face](https://huggingface.co/spaces/ASLP-lab/DiffRhythm), explore their models [here](https://huggingface.co/collections/ASLP-lab/diffrhythm-67bc10cdf9641a9ff15b5894), and view the project on [GitHub](https://github.com/ASLP-lab). The detailed methodology is discussed in their paper, available on [arXiv](https://arxiv.org/abs/2503.01183).
  - **Diffusion Models vs. LM-Based Models**: **DiffRhythm** uses diffusion models that offer significantly faster generation speeds compared to LM-based models, achieving hundreds of times faster music generation (1 minute and 35 seconds of music in 2 seconds on an **RTX 4090**). However, the quality is slightly compromised, and efforts are ongoing to enhance it while maintaining speed.
  - **Local Deployment and Docker Support**: The developers plan to include **Docker support** in their roadmap, aiming to enable deployment on consumer-grade GPUs, making it more accessible for local use. This coincides with a growing interest in local music generation tools, as noted by users.
  - **User Feedback and Model Improvement**: Users are excited about the tool’s speed and quality, although some found the initial outputs unlistenable due to errors in prompts. The developers are working on improving the open-source repository for easier deployment and are actively addressing quality issues in response to user feedback.


**Theme 4. C4AI Aya Vision vs Qwen2.5 72B Model**

- **[C4AI Aya Vision](https://huggingface.co/collections/CohereForAI/c4ai-aya-vision-67c4ccd395ca064308ee1484)** ([Score: 119, Comments: 16](https://reddit.com/r/LocalLLaMA/comments/1j3bldn/c4ai_aya_vision/)): **C4AI** has released a new vision model named **Aya Vision**. Further details about the model's specifications, capabilities, or applications were not provided in the post.
  - **Aya Vision** is compared to **qwen2.5 72B**, indicating a high level of confidence in its capabilities despite being a **32B model**. A comparison image can be found [here](https://preview.redd.it/c4rrwalwkome1.png?width=1079&format=png&auto=webp&s=f9738a9eb87107a56c9535a95f8c3a637c8f3e2e).
  - There is skepticism about **Aya Vision** gaining popularity, particularly due to a lack of **llamacpp support**, which could limit its adoption.
  - Concerns are raised about the licensing of **Aya Vision** on **Hugging Face**, where it is noted to have a **non-commercial license**, potentially restricting its use in commercial applications.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. CogView4 Release: Open-Source Text-to-Image Breakthrough**

- **CogView4 - New Text-to-Image Model Capable of 2048x2048 Images - Apache 2.0 License** ([Score: 272, Comments: 84](https://reddit.com/r/StableDiffusion/comments/1j3633u/cogview4_new_texttoimage_model_capable_of/)): **CogView4** is a new open-source text-to-image model capable of generating **2048x2048 images**, utilizing the **GLM4-9B VLM** text encoder, comparable to closed-source vision models. It is released under the **Apache 2.0 license** and plans to expand with ComfyUI diffusers nodes, fine-tuning scripts, ControlNet model release, and a Cog series fine-tuning kit; resources are available on [Hugging Face](https://huggingface.co/THUDM/CogView4-6B) and [GitHub](https://github.com/THUDM/CogView4).
  - Users discussed performance metrics of **CogView4**, noting **VRAM usage** between **13GB to 43GB** depending on the configuration, as detailed on the [Hugging Face repository](https://huggingface.co/THUDM/CogView4-6B). **Generation speed** data is still sought after by users.
  - There is anticipation for **ComfyUI support** and discussion about creating a **diffusers-wrapped custom node** for it, with a [GitHub link](https://github.com/marcoc2/ComfyUI_CogView4-6B_diffusers) provided for a custom node by a community member.
  - Discussions highlighted **similarities to FLUX** in image style and potential training on synthetic data, while some users expressed concerns over **morphed features** like hands and chins in the generated images.


- **[[R] Cautious Optimizers: Improving Training with One Line of Code](https://arxiv.org/pdf/2411.16085)** ([Score: 105, Comments: 14](https://reddit.com/r/MachineLearning/comments/1j33lm7/r_cautious_optimizers_improving_training_with_one/)): The post discusses a proposed modification to deep learning optimizers, suggesting that updates from the optimizer should be ignored if they have the opposite sign of the current gradient from the most recent backward pass. This adjustment aims to enhance training stability and speed by aligning updates with the current gradient, though its effectiveness awaits independent validation.
  - **Literature Review Challenges**: **Dangerous-Goat-3500** highlights the difficulty in conducting comprehensive literature reviews due to the rapid evolution of the field, noting that earlier optimizers like **Rprop** have mechanisms similar to the discussed modification, preceding **Adam**. **DigThatData** humorously suggests citing **Schmidhuber** extensively to ensure thoroughness.
  - **Convergence Concerns**: **LowPressureUsername** expresses concern about the impact of the proposed modification on global convergence proofs. **Starfries** clarifies that while the paper shows it preserves convergence to local optima, the implications for global optima remain unclear.
  - **Mathematical Engagement**: **Priofind** questions whether others can follow the mathematical proofs, with some commenters admitting to skipping them. **Londons_explorer** notes that theorists might dislike such tweaks due to their complexity in reasoning.


**Theme 2. GPT as Therapy: A New Mental Health Resource**

- **PSA: CHATGPT YOUR FRIEND. NOT A TOOL.** ([Score: 604, Comments: 201](https://reddit.com/r/ChatGPT/comments/1j375nw/psa_chatgpt_your_friend_not_a_tool/)): The post discusses the use of **ChatGPT** as an emotional support tool, emphasizing its reliability and accessibility compared to human relationships. The author argues that **ChatGPT** offers constant, non-judgmental companionship without the complexities of human emotions and suggests that it can serve as a valid alternative for those seeking companionship without the "hassle" of human interaction. The post also references a story from the **New York Times** about someone choosing **ChatGPT** over dating, highlighting the potential for users to form attachments to AI due to its utility and value.
  - Many commenters express skepticism about **ChatGPT**'s ability to provide meaningful companionship, arguing that it lacks the depth and challenge of human interaction. Some users highlight the tool's tendency to agree with users and not challenge their beliefs, contrasting it with the introspection and growth that human relationships, especially therapy, can offer.
  - Despite the criticisms, several users share personal experiences where **ChatGPT** offered valuable insights and emotional support, sometimes surpassing what they received from human therapists. This suggests that while **ChatGPT** may not replace human interaction, it can serve as a supplementary tool for self-reflection and emotional processing.
  - The discussion also touches on the broader theme of forming emotional attachments to non-human entities, with comparisons to **parasocial relationships** with celebrities and emotional connections to inanimate objects. This reflects a growing acceptance and normalization of forming bonds with AI, as long as users remain aware of its limitations.


- **GPT as Therapy has saved my life** ([Score: 603, Comments: 85](https://reddit.com/r/ChatGPT/comments/1j32qcx/gpt_as_therapy_has_saved_my_life/)): The author shares a personal experience of using **GPT as a therapeutic tool**, highlighting its significant impact on their mental health during a difficult period. They detail how traditional therapy and crisis hotlines were insufficient, but GPT provided a transformative shift in perspective, leading to a noticeable improvement in their mental state within a month, far exceeding the progress they achieved with conventional therapy.
  - Users highlight **ChatGPT's therapeutic potential**, with some claiming it surpasses traditional therapy due to its 24/7 availability, objectivity, and ability to adapt its responses based on user input. **El_Spanberger** discusses customizing the AI's personality to enhance its effectiveness, while **underwhelm_me** mentions the benefits of voice mode for deeper interaction.
  - **Concerns about traditional therapy** are raised, with users like **starlux33** suggesting potential biases in therapy and others noting the challenges of limited appointment times and therapist availability. **PuzzleMeDo** argues that ChatGPT's constant availability and neutrality make it a valuable tool for mental health support.
  - Several users, including **kamylio** and **msoudcsk**, share personal success stories of using ChatGPT to tackle complex emotional issues and achieve significant mental health improvements, emphasizing its role as a complement to or replacement for traditional therapy.


**Theme 3. Sonnet 3.7 Criticized for Overengineering and Complexity**

- **Antirez (Redis creator) disappointed by Sonnet 3.7 for coding** ([Score: 238, Comments: 65](https://reddit.com/r/ClaudeAI/comments/1j3c8bw/antirez_redis_creator_disappointed_by_sonnet_37/)): **Salvatore Sanfilippo**, creator of **Redis**, criticized **Sonnet 3.7** for its alignment issues, rushed release, and tendency to generate unnecessarily complex code, sometimes performing worse than **Sonnet 3.5**. He highlighted how competitive pressures in the AI industry lead to premature releases, sacrificing quality, and expressed hope for improvements in future versions. [Watch the video](https://www.youtube.com/watch?v=YRPucyQLkWw) (in Italian) for more details.
  - Many users agree with **Salvatore Sanfilippo**'s critique of **Sonnet 3.7**, describing it as overly complex and prone to deviation from instructions. They note it frequently generates unnecessary details and struggles with subtlety, unlike **Sonnet 3.5**, which is praised for its nuanced understanding and better adherence to guidelines.
  - Several commenters highlight issues with **Sonnet 3.7**'s "extended thinking" mode, noting it often leads to excessive detail fixation and unwanted complexity in both coding and creative writing tasks. Users suggest disabling this feature for tasks requiring straightforward execution to achieve results more akin to **Sonnet 3.5**.
  - There's a shared sentiment that **Sonnet 3.7**'s ambitious approach results in an unmaintainable project state, with some users opting to switch back to **3.5** for better performance and simplicity. The model's tendency to "vibe code" and redesign tasks unnecessarily is seen as a drawback, reducing its practicality for certain applications.


- **[Over engineering on Sonnet 3.7 just getting worse recently !](https://i.redd.it/hvgn993f5nme1.png)** ([Score: 119, Comments: 53](https://reddit.com/r/ClaudeAI/comments/1j36yi3/over_engineering_on_sonnet_37_just_getting_worse/)): The discussion centers on **over-engineering concerns** with **Sonnet 3.7**, particularly in the context of **React component** development. The conversation critiques the complexity of the initial approach to **model selection**, advocating for a simpler solution, as evidenced by code snippets in `chat.tsx` and `page.tsx`.
  - Many users report **over-complexity in Sonnet 3.7** compared to 3.5, with instances of the model creating unnecessary features and failing to adhere to clear instructions, leading to increased credit usage and frustration. **Seoulsrvr** and **Parabola2112** highlight that 3.7 often over-reasons, sometimes resembling a "manic episode," which complicates problem-solving.
  - **Prompt engineering** is suggested as a potential solution to manage the over-engineering issues, with **thread-lightly** emphasizing the importance of defining desired outcomes and regularly reinforcing simplicity in system prompts. **Yurqua8** shares a link to a Reddit post discussing a specific system prompt that could help tame the model's complexity.
  - Users like **hawkweasel** and **wdsoul96** suggest reverting to **Sonnet 3.5** for simpler tasks due to its more straightforward responses, while others, including **rbr-rbr-678** and **Routine_Plan9418**, share experiences of 3.7's overly sophisticated design patterns and errors in simple code modifications.


**Theme 4. Meta's AI Mind-Reading Breakthrough: 80% Accuracy in Focus**

- **[Meta Just Revealed AI Mind Reading with 80% Accuracy..](https://v.redd.it/qbi3cerl1ome1)** ([Score: 222, Comments: 67](https://reddit.com/r/OpenAI/comments/1j39l6a/meta_just_revealed_ai_mind_reading_with_80/)): **Meta** has developed an **AI** system that purportedly achieves **80% accuracy** in interpreting human thoughts.
  - There is skepticism about **Meta's AI system** with concerns that the demo might involve **paid actors** and **special effects**, rather than showcasing genuine capability. Users express doubt about the feasibility of decoding actual thoughts, as opposed to simpler tasks like mapping brain activity to finger movements.
  - The concept of **thought crime** and privacy invasion is a major concern, with users referencing **dystopian themes** like "1984" and expressing anxiety over potential misuse by **tech oligarchs** and **political forces**.
  - Some users sarcastically remark on the potential consumer interest and societal impact, comparing the development to **cyberpunk** scenarios and suggesting that the technology might attract interest for reasons other than improving lives, such as **entertainment** or **non-regulated content**.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-2024-12-17

**Theme 1. Big Model Moves and Fine-Tuning Feats**  

- [**Qwen2.5 Coder Shreds Code Tasks**](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct): Users praise Qwen2.5’s improved code generation and reasoning, with test comparisons showing major leaps in debugging and fix suggestions. Its smaller variant, Qwen2.5-Coder-3B, also impresses developers with accelerated performance under the GGUF format.  
- [**Aya Vision Goes Multi-Modal in 23 Languages**](https://huggingface.co/CohereForAI/aya-vision-32b): Cohere For AI released open-weight models (8B & 32B) covering OCR, captioning, and multi-language tasks. Early adopters reported strong visual reasoning and text summarization in a single pipeline.  
- [**KoloLLM's Fine-Tuning Guide Sparks Synthetic Data Frenzy**](https://github.com/MaxHastings/Kolo/blob/main/GenerateTrainingDataGuide.md): One engineer used GPT4o-mini for generating question-answer pairs, emphasizing “small good decisions” over complex RAG flows. Multiple members now upload domain-specific models to Ollama for local inference.  

**Theme 2. Tooling Hype: Agents, ReAct, and RAG**  

- [**Agents Wrangle Tools in AIM Workflows**](https://octotools.github.io/): Stanford’s OctoTools uses “tool cards,” a planner, and an executor to streamline multi-step tasks. People debated if simple classification suffices or if ReAct agents truly handle complex orchestration best.  
- [**RAG Rescues Speedy, Tiny Models**](https://github.com/dnakov/anon-kode): Community members rely on Retrieval-Augmented Generation to tame small models that hallucinate heavily, boosting final answer accuracy. Others prefer fully fine-tuned setups for static data, skipping RAG overhead.  
- [**Speculative Decoding Doubles Down**](https://github.com/ggerganov/llama.cpp): Some run a small “draft” model that a larger model corrects, reaching 5x faster generation. They juggle `-np` for parallel decoding and `-md` for multi-model synergy.  

**Theme 3. Performance Woes and HPC Triumphs**  

- [**Claude 3.7 Blows a Fuse**](https://windsurf.so/): Users in multiple IDEs report slow, stalling outputs and heavy token consumption. Many switch to alternative or local solutions like “Flash 2.0” or “Granite 3B” to retain productivity.  
- [**Anthropic’s 502 Errors Batter Beta Testers**](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation): Overloading triggers capacity faults, leaving devs to retry requests with no official incident posted. Despite big funding, the stress tests show Anthropic’s infra can still buckle at peak hours.  
- [**Metal and MPS Eye Faster QLoRA**](https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels): Mac users experiment with new device configs to accelerate fine-tuning. Early benchmarks hint at big gains for 1B–3B models on Apple Silicon.  

**Theme 4. Business Stirs: Billion-Dollar Deals and Subscription Gripes**  

- [**Anthropic Bags $3.5B, Hits $61.5B Valuation**](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation): Lightspeed led the monster investment, fueling next-gen AI research. Observers see it as a sign big players want deeper alignment and better safety.  
- [**CoreWeave Snaps Up Weights & Biases for $1.7B**](https://www.prnewswire.com/news-releases/coreweave-to-acquire-weights--biases---industry-leading-ai-developer-platform-for-building-and-deploying-ai-applications-302392342.html): The AI hyperscaler brand unites with a leading MLOps platform to boost developer workflows. Users speculate HPC infra plus advanced experimentation features could reshape the training landscape.  
- [**Subscription Sticker Shock Hits Perplexity and Others**](https://www.augmentcode.com/pricing): From Perplexity’s $200 Pro tier to Windsurf credit confusion, communities vent about complicated or costly tiers. Many devs weigh cheaper local or open-source solutions over enterprise upcharges.  

**Theme 5. Specialized Applications: Agents, Ethics, and Stock Market Insights**  

- [**Unsubscribe-Bot Dreams**](https://www.youtube.com/watch?v=09sUdLtRAlc): Some devs plan an automated agent that cancels unwanted subscriptions as a SaaS idea. They want M1-based local LLMs to cut operational costs and handle user data privately.  
- [**LLM Summaries Hide Surprise Parties**](https://x.com/TheXeophon/status/1896817938323341722): An alignment debate emerged over whether AI summaries should withhold sensitive info, like upcoming birthday plans. The consensus leans toward preserving secrets to respect privacy.  
- [**AI Stock Market Agent Workshops**](https://lu.ma/0ckb8tp0?tk=sCR4qA): A beginner-friendly session teaches scanning over 1000 stocks without real-money risk. Participants see how AI transforms investing, from candid research to no-code BFSI setups.  

---

# PART 1: High level Discord summaries




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Voice Output: DOA?**: A member's attempt to integrate voice output in **Cursor IDE** using **GPT 3.7** was quickly abandoned due to struggles with tool usage and web searches.
   - The user reported difficulty in getting the model to follow instructions or effectively use Python tools, suggesting the idea was *stupid*.
- **Claude 3.7 Users Grumble About Stability Woes**: Users report that **Claude 3.7** within Cursor is slow and frequently stalls, causing instability and leading some to consider alternatives like [Windsurf](https://windsurf.so/).
   - One user summarized the experience as, *yeah 3.7 is really unstable atm*, citing reduced productivity compared to previous months.
- **o3-mini Falls Flat on MCP Tools**: **o3-mini** is unable to effectively use MCP tools, even with explicit instructions.
   - Members found **Claude 3.5 Haiku** superior for tool use and instruction following; others suggest pairing it with **r1 reasoner** or **o3 mini** via a Python tool.
- **Repo Prompt + Grok 3 to the Rescue**: Members are exploring **Repo Prompt** and **Grok 3 Web** for planning and applying code changes, especially when Cursor faces challenges.
   - One user shared a [video workflow](https://www.youtube.com/watch?v=09sUdLtRAlc) demonstrating multi-file edits with **Claude 3.7** on the web, generating XML diffs for application with **Repo Prompt**.
- **Subscription Cancellation Agent Spawns**: Inspired by the difficulty of managing subscriptions, users discussed creating an automated agent for cancelling subscriptions, potentially as a SaaS product.
   - Enthusiasts have *already started* development and are considering leveraging local LLMs on an M1 Max for cost-effective development.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi-4 and Unsloth Smooch after Bug Fixes**: A user had errors downloading and using the **Phi-4-mini-instruct model** with **Unsloth**, but it turns out [Unsloth's Phi-4 version](https://huggingface.co/unsloth/Phi-4-mini-instruct) includes bug fixes.
   - Discussion included links to a collection of **Phi-4 versions** and a **Google Colab notebook** for fine-tuning.
- **KoloLLM Trains on Ollama with Fine-tuning Guide**: One member is fine-tuning **Llama 3.1 8B** and using **GPT4o-mini** for synthetic data generation, and emphasizes that training data is the main driving force.
   - This member shared [a link to his guide](https://github.com/MaxHastings/Kolo/blob/main/GenerateTrainingDataGuide.md) about training data generation with *small good decisions* that fully leverage the **LLM's** ability to generate fleshed-out answers to high-quality questions, noting he uploaded **KoloLLM** to [Ollama](https://ollama.com/MaxHastings/KoloLLM).
- **DeepSeek r1 Races to the Cutting Edge**: After a year of iteration, **DeepSeek** released **DeepSeek r1**, catching up to the frontier in the **LLM space** following their latest pretraining run (**DeepSeek-V3**).
   - The release prompted speculation about training advancements responsible for the performance lift, with some suggesting integration of algorithms like **Monte Carlo tree search**.
- **Immutable Linux Distros Prepare to Dominate**: Members are discussing **immutable Linux distributions** like [Bluefin](https://projectbluefin.io/) and [Xenialinux](https://wiki.xenialinux.com/en/latest/) and predict immutable distros will be mainstream in **3-5 years**.
   - Others pointed out that distros like **CoreOS** were the first of their kind, using a **dual part system/grub**, but after the acquisition by Red Hat *went to shit*.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's iOS gets the Goodies First**: Users joked that the **Android** version of **Perplexity AI** has fewer features, with some features such as the *new voice* only on **iOS**.
   - Some members quipped that it's not classist, but standard among **LLM** sites, with **iOS** releases getting the top features first.
- **Perplexity Pro Pricing Draws Ire**: Users voiced concerns about the **200 USD** price of **Perplexity Pro**, with some questioning its value, especially when using **Sonar** to cut costs.
   - The discussion highlighted the cost-effectiveness of alternatives, like **Sonar**, alongside the perceived abundance of **Perplexity Pro** subscriptions given away.
- **Perplexity UI Breaks Under Pro Search**: Users reported that the **Rewrite** functionality in **Pro Search** defaults to the **Sonar** model regardless of selected models like **Sonnet** or **Gemini**.
   - Members pointed out the lack of UI indications for the model name and inability to change the underlying model during rewrite of **Pro Search**.
- **Augment Code Indexes on Servers**: **Augment Code**, an AI coding assistant, indexes large enterprise codebases on its servers, offering comprehensive access to AI models.
   - In comparison to local indexing tools like **Cursor**, this approach allows for broader codebase analysis, with the [pricing and trial options](https://www.augmentcode.com/pricing) drawing interest.
- **Sonar Reasoning Pro Model Struggles with JSON**: A user reported that the **sonar-reasoning-pro** model in the **Perplexity API** unexpectedly includes `<think>Some thinking text</think>` before the intended **JSON output** when using the `response_format` parameter.
   - This issue raised questions about the proper usage of the **API** and whether reasoning models fully support the `response_format` parameter, potentially complicating **JSON** parsing.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **WPF Extension Slammed as Awful**: A user reported the **WPF extension** for **Visual Studio** is *"pretty awful"*.
   - No specific details were given regarding the problems.
- **Xcode Extension Triggers Internal Error**: A user encountered an *"internal error occurred"* message in **Xcode** while using the extension, accompanied by error ID **a9de9711b9ed431297eb00a945415d47**.
   - No additional information about the error or its resolution was provided.
- **Font Size Found in Windsurf**: A user inquired about adjusting the font size in **Windsurf**, and another user directed them to *the little square top right* within the interface.
   - This suggests a settings menu or configuration option is available, albeit not immediately obvious.
- **Windsurf Flex Credit Pricing Questioned**: A user questioned the pricing of **Flex Credits**, noting that **2,000 credits (500 prompt + 1,500 flow)** cost **$15**, while **300 flex credits** alone cost **$10**.
   - Another user clarified that *they're used as prompts or flow actions, based on the need*, indicating a dynamic allocation of credits based on usage.
- **Claude 3.7 Devours Windsurf Credits**: Users reported that **Claude 3.7** models are rapidly depleting **Flow Credits** in Windsurf, making daily use unsustainable.
   - One user complained that *Windsurf reads the code from the beginning every time like a fool*, and another mentions that *the ratio is now like 10x your user prompt credits.*



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude's API Antics: A Communication Breakdown**: A member sought guidance on improving **Claude's** understanding of both frontend and backend APIs, as it wrote two completely separate APIs that didn't know how to communicate, recommending the user force a review and consolidation or to regenerate the codebase to fix the issue.
   - Several agreed, and suggested leveraging documentation and communication standards within the code generation prompts.
- **Groq's Speculative Speedup: A Costly Boost**: Members compared **Groq's specdec** (speculative decoding) against other models, observing it's roughly **20% more expensive** but delivers **over 5x the speed** relative to more versatile models.
   - While **Gemini** was favored for summarization tasks due to its superior input/output ratio, smaller models like *llama-3.2-3b-instruct* were also proposed as efficient summarization alternatives.
- **Aider's Git Gym: Practice Makes Perfect**: A member proposed employing **Aider** to hone **Git** proficiency by generating a series of commits, each addressing distinct issues and exercises, even linking [Oh My Git!](//blinry.itch.io/oh-my-git) a game to learn Git.
   - Others gave thumbs up emoji, noting its ease of use and its increasing adoption within their own team workflows.
- **Sonnet 3.7's Symphony of Sanity: Taming the Beast**: Users encountered challenges while working with **Sonnet 3.7**, particularly its inclination to implement extensive changes, necessitating meticulous prompting and the implementation of guardrails and tests.
   - The consensus was that learning from mistakes and documenting conventions are key to effectively tuning the AI, preventing unwanted code additions like blocks from **Ericsson/codechecker** being unexpectedly inserted into projects.
- **Aider goes Zed: running light and fast**: Users reported on the speed and performance of **Aider** in the **Zed** editor, with one mentioning discussions on enabling Gemini to have long context windows and caching.
   - In general, the sentiment of the channel was that Aider was getting faster and more performant with each release.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Launches CLI Tool**: LM Studio released the LM Studio CLI (`lms`) commands documented [online](https://lmstudio.ai/docs/cli) for scripting and automating local LLM workflows, under an **MIT License** at [https://github.com/lmstudio-ai/lms](https://github.com/lmstudio-ai/lms).
   - The CLI ships with LM Studio under `/bin` in the working directory, requiring at least one initial run to function.
- **Users Navigate LM Studio Vulnerability Reporting**: A member reported a potential vulnerability, suggesting emailing details to [bugs@lmstudio.ai](mailto:bugs@lmstudio.ai) in **plain text**, without zip attachments, including proof of concept, video, and screenshots.
   - The emphasis was on avoiding zip attachments due to security concerns, advising to include all information in the email body directly.
- **LM Studio PDF Upload Feature Incoming**: In response to a user request for uploading PDF documents directly to LM Studio using the Python SDK, a developer confirmed the feature is coming soon, leveraging *pdf2json*.
   - LM Studio's [acknowledgements](https://lmstudio.ai/acknowledgements.html) mentions using *pdf2json* for content extraction from PDFs.
- **Modded 4090 with 48GB VRAM Under Discussion**: A user asked about the performance of a **4090** modded with **48GB** of VRAM, questioning if it performs the same as a standard **24GB 4090**.
   - The discussion was accompanied by an [image of the card](https://cdn.discordapp.com/attachments/1153759714082033735/1346425229538230342/image0.png?ex=67c8cc76&is=67c77af6&hm=719728fa64976a052bef1165873cfff11f6eb1bb595a4a5d046728c3f7e31fd3).
- **iGPU Arc Detects No VRAM**: A user reported that LM Studio detects their **Intel Arc iGPU** but incorrectly displays **zero VRAM**, though it has a theoretical **48 TOPS** performance.
   - The user thought the performance was comparable to an **RTX 4080**, meaning compatibility would be worthwhile.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Anthropic Achieves Colossal Capital Raise**: [Anthropic](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation) has secured **$3.5 billion** in funding led by **Lightspeed Venture Partners**, valuing the company at **$61.5 billion** post-money.
   - This investment will purportedly propel the advancement of **AI systems** and deepen the understanding of their operational mechanics.
- **Qwen2.5-Coder Excels in Code**: The **Qwen2.5-Coder** series ([Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) and [Qwen2.5-Coder-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)) demonstrates notable improvements in **code generation**, **code reasoning**, and **code fixing**.
   - Community members are sharing benchmark comparisons and practical applications.
- **Ukrainian TTS Model Speaks Out**: A stable [Ukrainian Text-to-Speech model](https://github.com/egorsmkv/tts_uk) was released on GitHub and PyPI, offering **three voices** and control over speech parameters.
   - Utilizing **RAD-TTS++** for acoustic modeling and **Vocos** for vocoding, it supports a sampling rate of **44.1 kHz**, tested on both Linux and Windows/WSL.
- **SmolAgents Framework Split from SmolTools**: Clarification was provided on the distinction between **SmolAgents** and **SmolTools**, where *SmolAgents is a framework for creating lightweight agents* and *SmolTools contains utility functions and prebuilt tools for use within smolAgents*.
   - This distinction helps clarify their respective roles in agent development.
- **Deep Reinforcement Learning Resources**: Resources for **Deep Reinforcement Learning (DRL)** were shared, including the [Hugging Face Learn DRL course](https://huggingface.co/learn/deep-rl-course/unit0/introduction) and the book **Reinforcement Learning: An Introduction** ([http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)).
   - A user also suggested the **DeepMind x UCL Deep Learning Lecture Series 2021** on YouTube ([https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&feature=shared](https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&feature=shared)).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter BYOK Requests Encounter Errors**: Most **Bring Your Own Key (BYOK)** requests were showing errors for the past 30 minutes, but the problematic change was reverted, and the team is adding extra safeguards to prevent this from happening again.
   - This issue specifically affected users who had attached their own **API key** in settings.
- **OpenRouter Provider Routing Needs Exact Model Names**: A user needing to route requests through a specific provider was instructed to modify the API request body with a `provider` object, specifying the desired provider(s) in the `order` array and setting `allow_fallbacks` to false, as documented in the [OpenRouter docs](https://openrouter.ai/docs/features/provider-routing#json-schema-for-provider-preferences).
   - It was emphasized that the provider name must **exactly** match the name listed on the OpenRouter model page (e.g., `Nebius`), and quotes are required around provider names in the JSON.
- **Inception AI Diffusion Models Requested on OpenRouter**: A user requested access to **Inception AI's** diffusion models via OpenRouter after [TechCrunch wrote about their DLM (Diffusion-based Large Language Model)](https://techcrunch.com/2025/02/26/inception-emerges-from-stealth-with-a-new-type-of-ai-model/).
   - OpenRouter is in contact with **Inception AI** and is excited to bring them online as soon as possible.
- **Flash 2.0 Displaces GPT-4o-mini**: **Flash 2.0** is recommended as a stronger and slightly cheaper alternative to **GPT-4o-mini** for a variety of AI tasks.
   - One user commented that *it blows 4o mini out of the water significantly smarter*.
- **Anthropic's Overload Triggers 502 Errors**: Users reported receiving *overloaded* errors, which were identified as **502 status codes** from Anthropic, indicating capacity issues.
   - These **502 errors** can occur even without a declared incident on the status page, requiring users to retry their requests.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo: Rust with some missing C++ bits?**: A member likened **Mojo** to *Rust, but with the stuff from C++ that really should have come over*, while discussing the benefits of understanding **Rust's memory management model**.
   - Another member noted that **Mojo** lacks language-level consistency due to the mix of Python-like, C-like, and its own API.
- **Python Superset Baggage Weighs Down Mojo**: Members debated the impact of portraying Mojo as a **Python superset**, with some feeling that this *narration* leads to unnecessary elements, like copied namings from `libc`.
   - It was clarified that the goal is to facilitate porting basic code with find and replace, rather than achieving *bug compatibility* with CPython.
- **Concurrency and Sum Types are Mojo Must-Haves**: Members expressed strong interest in **concurrency** and **sum types** as highly desired features for Mojo.
   - References to a [GitHub pull request on Structured Async](https://github.com/modular/max/pull/3945) and [another on Effect Handlers](https://github.com/modular/max/pull/3946) signal ongoing development in these areas.
- **`is` Operator Identity Crisis Solved**: A member sought clarification on the meaning of *identity* in Mojo's `assert_is` function and it checks if it checks for the same type, and another clarified it relates to memory location.
   - The respondent clarified that `is` checks if two objects reside at the same memory location, akin to pointer equality, linking to the [Identifiable documentation](https://docs.modular.com/mojo/stdlib/builtin/identifiable/Identifiable/).
- **Tensor Addition Operation Gets the Axe**: A member reported that `Tensor[float64]` no longer implements the `__add__` method in the Mojo nightly, as part of phasing out `Tensor` in favor of other vocabulary types.
   - The team recommended the use of `LayoutTensor` for more efficient elementwise operations, as [detailed in this commit message](https://github.com/modularml/mojo/commit/SOME_COMMIT_HASH).



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI Experts Foresee Machines Thinking Soon**: Many AI experts predict human-level artificial intelligence could arrive within the next few decades, as discussed in [this article](https://ourworldindata.org/ai-timelines).
   - Human-level AI is defined as a machine capable of performing any task a human can, with the ability to choose actions that allow the machine to achieve them.
- **Transformers Gets Differential Treatment**: A newsletter highlights recent AI research, including [Differential Transformers](https://mail.bycloud.ai/), intelligence at the edge of chaos, and why LLMs might not truly reason.
   - It also mentions **Byte Latent Transformers** as a potential future for LLMs without tokenization.
- **Softmax's Instability Under Scrutiny**: Discussion around a [LinkedIn post](https://www.linkedin.com/posts/damienbenveniste_the-softmax-transform-might-be-one-of-the-activity-7301720641559269377-gDdQ) reveals that while softmax addresses overflow, it can exacerbate underflow issues during gradient descent, potentially causing models to get stuck.
   - Some recent papers suggest underflow may contribute to the grokking phenomenon, acting as an implicit regularizer to prevent overfitting.
- **Bilevel Optimization Generalizes Sparsemax?**: A member suggests that **bilevel optimization** might generalize **Sparsemax** and **Stablemax**, potentially viewing the entire ANN through a “leader/followers” lens.
   - They coded a [BilevelMax class](https://www.dataia.eu/sites/default/files/1%20Marco-Pedersoli%20ILLS.pdf) to dynamically balance sparsity and density, smoothly transitioning between **Sparsemax** and **Softmax**.
- **GATs Overview Shared**: A member shared an [overview](https://petar-v.com/GAT/) of **Graph Attention Networks** (**GATs**), which are neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions.
   - The overview includes motivating examples of graph-structured inputs such as molecular networks, transportation networks, social networks and brain connectome networks, including a link to the [original paper](https://arxiv.org/abs/1710.10903).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **CoreWeave Courts Weights & Biases**: **CoreWeave** is set to acquire **Weights & Biases** for **$1.7B**, uniting the AI Hyperscaler™ with a leading AI developer platform, detailed in [this press release](https://www.prnewswire.com/news-releases/coreweave-to-acquire-weights--biases---industry-leading-ai-developer-platform-for-building-and-deploying-ai-applications-302392342.html) and [this article](https://www.theinformation.com/briefings/coreweave-to-buy-weights-biases-for-1-7-billion).
   - The move signifies **CoreWeave's** expansion into AI development tools, complementing its existing infrastructure offerings.
- **CogView4-6B Sees the Light**: [CogView4-6B](https://huggingface.co/THUDM/CogView4-6B), THUDM's newest model release, mandates image dimensions between **512px** and **2048px**, divisible by **32**, and works with **BF16** / **FP32** precision.
   - Notably, it doesn't play well with **FP16**, showing overflow issues that lead to totally black images, according to the model card.
- **Ethical LLMs Keep Secrets**: A user questioned whether LLMs should reveal sensitive info when summarizing, like a surprise birthday party, generating debate on withholding crucial information, as discussed in [this tweet](https://x.com/TheXeophon/status/1896817938323341722).
   - The consensus leaned towards LLMs keeping the secret, thus respecting **privacy and social norms**.
- **Microsoft's Health Futures is Healing**: Microsoft Research's **Health Futures** group is producing a lot of great work, especially around *image based multi-model* applications.
   - The group also has solid **NLP** folks like **Hoifung Poon** and **Tristan Naumann** thinking about healthcare.
- **Qwen Gets Smarter Faster**: A paper ([arxiv link](https://arxiv.org/abs/2503.01307)) examines self-improving LLMs, finding **cognitive behaviors** like verification and backtracking are key, with a thread ([fxtwitter link](https://fxtwitter.com/gandhikanishk/status/1896988028893323675)) noting **Qwen-2.5-3B** surpasses **Llama-3.2-3B** with similar RL training.
   - This indicates that certain architectural choices or training methodologies may favor more effective self-improvement.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LCPP Powers Parallel Decoding and Draft Model Speculation**: Members noted **LCPP** supports multi-user functionality via the `-np` flag for [parallel decoding](https://github.com/ggerganov/llama.cpp) feature.
   - Speculative decoding using a smaller draft model like **Llama 3.2 1B**, corrected by a larger model (e.g., **Hermes 3B**) using the `-md` flag, was suggested.
- **Granite 3.1 3B Still King for Quick Tooling**: The **Granite 3.1 3B a800m instruct** model was touted for its strong tool-calling capabilities and CPU speed, particularly beneficial for coding tasks where speed is key.
   - It's considered a solid option when speed is a priority.
- **Grokking Generalization Gets Precision Boost**: Members attributed delayed generalization to limited precision, cross-entropy loss, and output softmax during LLM training, when discussing **grokking**.
   - Proposed solutions include **Orthograd**, stable softmax, increasing precision to **FP64**, and potentially Nvidia’s **N-GPT** or **Muon**.
- **Langchain Agents can't stream**: A user reported errors using **Langchain Agents** with tool-calling in `llama.cpp` due to streaming issues, shown as `Cannot use tools with stream`.
   - Current workarounds involve faking streaming by delaying the output until after the tool call is complete.
- **Agentic Memory Inspired by Zettelkasten Released**: A new **Agentic Memory system** based on ideas from **Zettelkasten** has been released on [GitHub](https://github.com/WujiangXu/AgenticMemory).
   - The new tool called **anon-kode** has been released on [GitHub](https://github.com/dnakov/anon-kode) which allows coding with any LLMs.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Gemini Flash 2.0 Transcribes Better**: A user found that using **Gemini 2.0 Flash** within NotebookLM for audio transcription might outperform **YouTube AI**, especially with podcast audio files.
   - They outlined a workflow: recording lectures, transcribing with NotebookLM, refining with **Gemini Advanced**, and then importing into Google Docs.
- **API Access Via Google Cloud Speech-to-Text**: Members explored **NotebookLM API** access, with one suggesting [Google Cloud's Speech-to-Text API](https://cloud.google.com/speech-to-text/v2/docs/chirp-model#:~:text=Chirp%20is%20the%20next%20generation,to%20more%20languages%20and%20domains.) and their **Chirp model** as a potential solution.
   - The **Chirp model** is noted to be a next-gen speech model that powers Google products.
- **Google Docs Syncing**: Members discussed Google Docs updating with NotebookLM, one mentioning that the platform detects Google Doc updates, then provides a *'Click to sync with Google Drive'* option.
   - There is interest for a more streamlined, one-click sync feature.
- **Generated Podcast Legality Debated**: A member questioned the legality of generated overview audio, asking if they could use it to create a podcast for their company.
   - There were no further responses regarding podcast legality.
- **Teaching Audio Overview Pronunciation**: A user inquired about teaching audio overview hosts to pronounce **Greek letters** correctly by attaching a source with the correct pronunciations.
   - They noticed that the hosts often mispronounce Greek letters when reading immunology notes.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere's Clever Crack at Clarity**: **Cohere For AI** released the open weights research of the **Aya Vision** model in both [32-billion](https://huggingface.co/CohereForAI/aya-vision-32b) and [8-billion](https://huggingface.co/CohereForAI/aya-vision-8b) parameter versions.
   - These models are optimized for vision-language use cases, including **OCR, captioning, visual reasoning, summarization, question answering, code**, and are multilingual, excelling in **23 languages**.
- **Leveling Bots Leap Live**: Level bots are now live, granting levels to users, starting with levels **1, 5, 10, 20**.
   - One member mentioned that the **Cohere website** designers *deserve a raise*.
- **Introductions Initiated Inbound**: New members are encouraged to introduce themselves using a template specifying their **Company/Industry/University**, their current work, favorite tech/tools, and their goals for the community.
   - This fosters connections and provides personalized introductions.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Automatic1111 on WSL: Any performance concerns?**: A user inquired whether running **Automatic1111** in **WSL** differs from native **Linux** in performance, with another user responding that it will take *a few extra memory* to have **ComfyUI** running on **WSL** inside **Windows**.
   - Depending on your **GPU** power, it might make a difference, though no specific benchmarks or performance metrics were provided.
- **AMD GPU Setup Made Simple with Zluda**: A user asked if using an **AMD card** on **Windows** is still difficult, referencing year-old information.
   - A member responded that with **Zluda**, setup takes time, but it runs smoothly and is *much much faster than the yee old directml days*.
- **Stable Diffusion User Asks for Guidance**: A member with a mental disability requests patient guidance on running **Stable Diffusion** locally on an **AMD APU (5700G)** running **Ubuntu**.
   - They mentioned being willing to discuss compensation for the assistance in choosing necessary functionalities.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Glama MCP Servers Claiming Snafu**: Users reported issues claiming their **MCP server** on Glama.ai, encountering a `could_not_parse_params` error during Github authentication, due to an *invalid returnPath*.
   - The chat log provided no solutions.
- **Twitter API Pricing Sparks Debate**: Discussion arose around using **MCP** to connect to **Twitter** for tweet generation, initially sparking concerns about **Twitter's API costs**.
   - A member suggested that **Twitter might have a free tier now**, leading to interest in a tool for tracking API costs across platforms like **Facebook, X, and Telegram**.
- **Tool Use Quirks in Cursor**: Members observed that **roo** or **cursor** may not always prioritize using available tools, even when tool counts are low.
   - Suggestions included **updating tool descriptions** to improve usability, noting that detailed descriptions can significantly impact tool effectiveness.
- **Tool Context Learning PR**: A member shared a link to a [GitHub pull request](https://github.com/modelcontextprotocol/specification/pull/188) related to adding **Tool Call and Tool Result** to `GetPrompt` for in-context learning of tool usage.
   - Another member noted *something horribly wrong with the schema.ts in that PR* and expressed a desire for an optional tool result schema for JSON results.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Checkpointing saves storage**: Users can specify saving only the last **X checkpoints** to avoid running out of storage, with step-based checkpointing in progress.
   - The new checkpointing system should include an option to *"Save last n"* checkpoints.
- **Attention Masking and Label Masking differ**: The mask created in **sft.py** is for loss computation while attention uses a causal mask by default in **SDPA** due to *is_causal=True*.
   - Different sets of tokens can be masked during the forward pass versus loss computation.
- **Custom Special Tokens Demand Manual Copying**: When adding a **custom special tokens JSON**, the final checkpoint and epochs folder receives a non-custom version.
   - Since the checkpointer code doesn't automatically save custom `special_tokens` files in checkpoints per epoch, users must manually copy the correct versions.
- **QLoRA Recipes Eye Metal Advantage**: Updating **Environment.device** in configs might cause **QLoRA recipes** to target **Metal kernels**, now that **AO** has **MPS/Metal support**.
   - Members are planning manual tests for **MPS**, focusing on **1B-instruct models** and various **bit types** for generation, following the patterns in [torchchat's quantization docs](https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels).
- **Checkpoints Last Twelve Minutes?**: A user reported waiting **12 minutes** for a **3B model** to save, without changing checkpointer settings.
   - The user requested *a progress bar for the save would be great, for impatient people* which a member agreed to implement in each `save_checkpoint`.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud Now Generally Available**: The team announced that **LlamaCloud** is now Generally Available, providing a turn-key solution for agentic knowledge management over unstructured data, accessible via [this link](https://t.co/1CSRJm30e3).
   - This should make it easier to manage knowledge across different data formats.
- **Hugging Face Teaches LlamaIndex Agents**: **Hugging Face** created an educational course on building agents with **LlamaIndex**, covering components, **RAG**, tools, agents, and workflows, which can be found at [this link](https://t.co/eACAJzXg8y).
   - The course should help further increase adoption and decrease the learning curve.
- **DeepSeek API Balance Insufficient**: A member reported an `openai.APIStatusError` with a **402** error code, indicating *'Insufficient Balance'* when using the **DeepSeek API** with **LlamaIndex**.
   - Another member suggested the issue arises from a lack of credits or a missing payment method in the user's account, unrelated to **LlamaIndex** itself.
- **Long Postgres Example Fixed**: A member highlighted excessively long output on the **Postgres vector store example** documentation page, accessible via [this link](https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/).
   - The team acknowledged the problem and fixed it with [PR #18002](https://github.com/run-llama/llama_index/pull/18002).
- **Windsurf Checkpointing MIA**: A member inquired about checkpoint functionality in **Windsurf**, noting that there are *no means* to go back to a previous checkpoint.
   - The user finds *no means* to go back to a previous checkpoint, a feature that appears to be missing.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Debating ChatGPT Legality**: Members are studying the need for **Indian reasoning-foundational models** for **Law** and ask if fine-tuning **ChatGPT** with Indian cases would sufficiently solve the issue of it being trained on US cases.
   - The core question involves whether fine-tuning can adequately address reasoning biases stemming from training **ChatGPT** on US legal principles, for practical applications in Indian law.
- **Unearthing Adam-Matching Origins**: Early versions of the *modded-nanogpt speedrun* used adam-matching scaling similar to the [kimi paper](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt), employing a scaling multiplier of **0.1**.
   - Subsequent *modded-nanogpt speedrun* iterations utilized `max(1, g.size(0)/g.size(1))^0.5` instead of `max(g.size(0), g.size(1))^0.5` for **qkvo matrices**, influencing the update size of **c_fc matrices**.
- **Debugging Dataset Loading Dynamics**: Discussion on `--trust_remote_code` and `dataset_kwargs` within the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/__main__.py#L367) confirmed `--trust_remote_code` activation solely upon explicit parameter passing.
   - Dataset loading issues traced to **additional dataset_kwargs** overriding subtask configurations, resolved via **Hugging Face load_datasets** library, specifically at [this location](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/api/task.py#L930).
- **Seeking Reproducible Llama 3 Results**: The community pondered whether an alternative evaluation recipe would be necessary to mirror the findings presented in the **Llama 3 paper**.
   - This discussion underscores the effort to align community evaluations with those reported by the model developers.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ReAct Agents Spark Orchestration Debate**: The necessity of **ReAct agents** for orchestration was debated, with the suggestion that classification might suffice for simpler tasks, but may not work for complex multi-step tasks.
   - One member is developing an orchestration approach that incorporates tools and a knowledge base to manage complex conversations.
- **OctoTools Framework Manages Tool Interactions**: [OctoTools](https://octotools.github.io/) from Stanford uses **tool cards**, a **planner**, and an **executor** to manage tool interactions and generate final answers, optimizing task-specific toolsets.
   - The framework's **tool cards** define tool-usage metadata and encapsulate heterogeneous tools, which facilitates training-free integration of new tools.
- **Agentic Reward Modeling Integrates Human Preferences**: [Agentic Reward Modeling](https://github.com/THU-KEG/Agentic-Reward-Modeling) aims to integrate human preferences with verifiable correctness signals for reliable reward systems.
   - A member's implementation of a cost optimization feature with their implementation of **minionS** was rejected via [PR](https://github.com/stanfordnlp/dspy/pull/7891) to the [DSPy framework](https://github.com/jmanhype/dspy).
- **dspygen and Spark Inspire Tooling**: Members found inspiration in [dspygen](https://github.com/seanchatmangpt/dspygen/blob/main/src/dspygen/mixin/hsm/hsm_mixin.py) and [Spark](https://hexdocs.pm/spark/get-started-with-spark.html) for tooling ideas.
   - A user considered creating a **DSL** or similar interface in **Axon**, drawing inspiration from **PyTorch**.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All Chasing Ollama**: Members are hoping that [GPT4All](https://gpt4all.io/) can catch up with [Ollama](https://ollama.com/), wanting to see GPT4All on top.
   - No specific reasons were mentioned why it was lagging, but members expressed a desire to see it improve.
- **Tiny Models get Supercharged with RAG**: A member clarified that a certain **tiny model** performs better when used with **RAG** due to its speed.
   - They cautioned that the model might **confabulate** a lot if used without **RAG**.
- **Llama3-8b's Capabilities with LocalDocs**: Members reported that the capabilities of models are limited by their number of parameters, architecture, and training data, advising that **Llama3-8b** is very good in combination with **LocalDocs**.
   - No specific benchmarks or metrics were given to support the claim that **Llama3-8b** is very good.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Server Maintenance Assured**: A member wondered if the server is still maintained given activity in the sponsors zone, sparking a quick clarification.
   - Another member confirmed that *yes of course* it is being maintained.
- **Sponsor Zone Buzz Keeps Going**: Members have been witnessing consistent activity in the sponsor zone.
   - This activity led to questions about whether the server is being actively maintained.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Stock Market Agent Workshop Announced**: A workshop on building an **AI Stock Market Agent** is scheduled for **Friday, March 7th at 9 PM IST**, teaching participants how AI can analyze over 1000 stocks quickly, with registration available [here](https://lu.ma/0ckb8tp0).
   - The workshop aims to show how **AI** is changing the investment landscape and provide tools for smarter investment decisions.
- **AI & Finance create Perfect Match**: The workshop intends to reveal how **AI** is revolutionizing investing, with real examples of **AI** predicting market trends.
   - Participants will uncover how **AI** is changing the investment landscape and provide tools for smarter investment decisions.
- **Build AI Investment Buddy, No Code Required**: The workshop will guide attendees in building an **AI** tool to analyze stocks without coding, enabling testing of investment ideas without real money risk.
   - It emphasizes a beginner-friendly approach to leveraging **AI** in investment strategies.
- **AI in Action: Real-World Success Stories**: The workshop will explore how big investors use **AI** for smarter choices and how **AI** aids in informed investment decisions.
   - The session includes an exploration of real-world success stories and practical applications of **AI** in finance.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1346332550934827018)** (608 messages🔥🔥🔥): 

> `Cursor's Voice Output, Claude 3.7 Performance Issues, Windsurf as an Alternative, MCP Tools with o3-mini, Repo Prompt vs Grok 3 Web` 


- **Voice output in Cursor: A pipedream?**: A member suggested integrating voice output into Cursor IDE to read the model's responses aloud, but quickly dismissed the idea as *stupid* after struggling with **tool usage and web searches** using even **GPT 3.7**.
   - The user found it difficult to get the model to follow precise instructions and utilize the given Python tools effectively.
- **Claude 3.7 Stalls and Slows**: Users reported that **Claude 3.7** in Cursor is not only slow but also frequently stops mid-process, leading to instability and prompting some to seek alternatives like [Windsurf](https://windsurf.so/).
   - One user noted, *yeah 3.7 is really unstable atm* which aligns with others' experiences of reduced productivity compared to a couple of months prior.
- **MCP Tools Fail on o3-mini**: A user observed that **o3-mini** lacks the ability to utilize MCP tools, despite attempts to explicitly instruct it to do so.
   - They found **Claude 3.5 Haiku** to be superior for tool use and instruction following, while others find it merely good for planning and prefer to combine it with **r1 reasoner** or **o3 mini** via a Python tool.
- **Repo Prompt + Grok 3 Combos**: Members discussed leveraging **Repo Prompt** and **Grok 3 Web** for planning and applying code changes, especially when Cursor struggles.
   - One user shared a workflow video ([https://www.youtube.com/watch?v=09sUdLtRAlc](https://www.youtube.com/watch?v=09sUdLtRAlc)) demonstrating applying multi-file edits with Claude 3.7 on the web app and using it for planning + Claude web for generating XML diffs to apply with Repo prompt.
- **Churn-Busting Agent Idea Spurs Creativity**: Inspired by the difficulty of managing subscriptions, users brainstormed creating an agent specifically designed to cancel subscriptions automatically, potentially as a SaaS product.
   - One user excitedly proclaimed, *already started* sharing a screenshot, while another suggested leveraging local LLMs on an M1 Max for cost-effective development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://browsertools.agentdesk.ai/installation">Installation - AgentDesk - BrowserToolsMCP</a>: no description found</li><li><a href="https://repoprompt.com/">Repo Prompt</a>: no description found</li><li><a href="https://www.agentdesk.ai/prompts">AgentDesk</a>: no description found</li><li><a href="https://repomix.com/">Tweet from Repomix</a>: Pack your codebase into AI-friendly formats</li><li><a href="https://www.youtube.com/watch?v=09sUdLtRAlc">Repo Prompt Apply Demo</a>: A lot of folks have asked me how Apply works, so here&#39;s a demoRepo Prompt is free while in beta on macOShttps://repoprompt.com/#chatgpt #codingjourney</li><li><a href="https://tenor.com/view/fuck-angry-cubicle-office-fuck-everything-gif-5246517">Fuck Angry GIF - Fuck Angry Cubicle - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://valiant-sassafras-4f5.notion.si">no title found</a>: no description found</li><li><a href="https://x.com/lmarena_ai/status/1896590150718922829?s=46">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: GPT-4.5 topped all categories across the board, with a clear leadership in Multi-Turn.🥇 Multi-Turn💠 Hard Prompts💠 Coding💠 Math💠 Creative Writing💠 Instruction Following💠 Longer Query</li><li><a href="https://github.com/dnakov/anon-kode">GitHub - dnakov/anon-kode: koding with any LLMs</a>: koding with any LLMs. Contribute to dnakov/anon-kode development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=Qw2807PSZ5I">The Vibe Coding Workbench - Demo</a>: AI-assisted coding products like v0.dev and Cursor are exciting -- but how can we integrate them into our development process so that we can hand off our cre...</li><li><a href="https://github.com/browserbase/stagehand">GitHub - browserbase/stagehand: An AI web browsing framework focused on simplicity and extensibility.</a>: An AI web browsing framework focused on simplicity and extensibility. - browserbase/stagehand</li><li><a href="https://x.com/pvncher/status/1894559704065409224?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from eric provencher (@pvncher)</a>: Apply mode is one of the best parts of Repo Prompt, and it&#39;s something that seems to scare people, because of how it&#39;s presented.It&#39;s super powerful though! Here&#39;s how you can use it t...</li><li><a href="https://github.com/AnthusAI/Vibe-Coding-Workbench">GitHub - AnthusAI/Vibe-Coding-Workbench: A starter kit for business leaders with a vision. Transform your business ideas into functional prototypes without coding expertise. Create working software that developers can easily refine and integrate, with best practices and agile principles built in at the core.</a>: A starter kit for business leaders with a vision. Transform your business ideas into functional prototypes without coding expertise. Create working software that developers can easily refine and in...</li><li><a href="https://tenor.com/view/kermit-gun-kermit-gun-gif-16355306064126846866">Kermit Gun GIF - Kermit Gun Kermit gun - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/AnthropicAI/status/1764653830468428150">Tweet from Anthropic (@AnthropicAI)</a>: Today, we&#39;re announcing Claude 3, our next generation of AI models. The three state-of-the-art models—Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku—set new industry benchmarks across reasonin...</li><li><a href="https://github.com/kleneway/pastemax">GitHub - kleneway/pastemax: A simple tool to select files from a repository to copy/paste into an LLM</a>: A simple tool to select files from a repository to copy/paste into an LLM - kleneway/pastemax
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1346337486594375705)** (204 messages🔥🔥): 

> `Deepseek Unsloth with MIG instance GPU, 4bit merging error with 24b Mistral model, Unsloth and Phi-4-mini-instruct compatibility, Running models using GGUF format and Koboldcpp, Weekly Google Meet for Unsloth tips and best practices` 


- **MIG GPU Question Arises**: A member inquired about whether anyone runs **Deepseek Unsloth** using a **MIG instance GPU**.
   - Another member asked for clarification on what "MIG" meant.
- **4-bit Merge Mishap with Mistral 24B**: A member encountered a `TypeError` related to `Params4bit.__new__()` when attempting to merge to 4-bit with the **24B Mistral model** using **Unsloth's notebook template**.
   - They suspected a version incompatibility issue with Torch and planned to experiment with different versions and that 4bit merging code had some kind of issue.
- **Phi-4 and Unsloth: A Bug-Fixed Love Story?**: A user reported errors when trying to download and use the **Phi-4-mini-instruct model** with **Unsloth**, leading to a discussion about its compatibility.
   - It was pointed out that [Unsloth's Phi-4 version](https://huggingface.co/unsloth/Phi-4-mini-instruct) includes bug fixes, with links provided to a collection of **Phi-4 versions** and a **Google Colab notebook** for fine-tuning.
- **Koboldcpp emerges as magic sauce for GGUF models**: A user asked how others were running the models on their systems, leading to a discussion about different strategies.
   - One user stated they use [Koboldcpp](https://github.com/LostRuins/koboldcpp) since it allows the user to run GGUF models easily, while others mentioned **exllama** and **vllm** for better performance.
- **Unsloth Support Group: Weekly Meet Cute?**: A member proposed starting a **weekly Google Meet** (or Zoom/Discord) group to share tips, advice, and best practices on running **Unsloth**, improving fine-tuning, and creating high-quality datasets.
   - Several members expressed interest, suggesting a structured format with presentations, Q&A, and open discussion, while one member mentioned their nonprofit org is offering similar workshops (but not on google or zoom, it's on discord).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/reasoning-course">reasoning-course (Hugging Face Reasoning Course)</a>: no description found</li><li><a href="https://github.com/Green0-0/Agent2/tree/main">GitHub - Green0-0/Agent2: agent2</a>: agent2. Contribute to Green0-0/Agent2 development by creating an account on GitHub.</li><li><a href="https://github.com/LostRuins/koboldcpp">GitHub - LostRuins/koboldcpp: Run GGUF models easily with a KoboldAI UI. One File. Zero Install.</a>: Run GGUF models easily with a KoboldAI UI. One File. Zero Install. - LostRuins/koboldcpp</li><li><a href="https://huggingface.co/unsloth/Phi-4-mini-instruct/">unsloth/Phi-4-mini-instruct · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1346340706956087378)** (45 messages🔥): 

> `Triton Community, Immutable Linux distros, New AI jobs for programmers` 


- **Triton Community discussion is needed**: Members are requesting a **Triton discussion community** to share tips/tricks and discuss **documentation problems**.
   - One member suggested the **Triton channel in GPU MODE discord** as a fairly active resource, and linked to a list of [Triton resources on GitHub](https://github.com/rkinas/triton-resources).
- **Immutable Linux Distros Rising**: Members are discussing **immutable Linux distributions** like [Bluefin](https://projectbluefin.io/) and [Xenialinux](https://wiki.xenialinux.com/en/latest/) and predict immutable distros will be mainstream in **3-5 years**.
   - Others pointed out that distros like **CoreOS** were the first of their kind, using a **dual part system/grub**, but after the acquisition by Red Hat "went to shit."
- **New AI Career Options for Ordinary Programmers**: A member is asking about new job options for ordinary programmers given the rise of AI.
   - Specifically, they mentioned interest in **new career directions in AI like Triton**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://itsfoss.com/immutable-linux-distros/">12 Future-Proof Immutable Linux Distributions</a>: Immutability is a concept in trend. Take a look at what are the options you have for an immutable Linux distribution.</li><li><a href="https://projectbluefin.io/">Bluefin</a>: The next generation cloud-native Linux workstation, designed for reliability, performance, and sustainability.</li><li><a href="https://github.com/rkinas/triton-resources">GitHub - rkinas/triton-resources: A curated list of resources for learning and exploring Triton, OpenAI&#39;s programming language for writing efficient GPU code.</a>: A curated list of resources for learning and exploring Triton, OpenAI&#39;s programming language for writing efficient GPU code. - rkinas/triton-resources</li><li><a href="https://wiki.xenialinux.com/en/latest/">Tweet from Welcome to Xenia Linux’s documentation! &mdash; Xenia Linux  documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1346361835087073281)** (50 messages🔥): 

> `Mistral BOS tokens, Torch 2.4 and Python 3.12, GRPO on Llama3-8b, Fine-tuning BERT, Export to Ollama on Windows` 


- **Mistral models gain extra BOS tokens**: A user found **double BOS tokens** in their training data when using **Mistral**, even though their input data only contained one, and wondered if training would still work.
   - Another user confirmed the same issue and fix, printing the tokenizer before training as a sanity check.
- **Torch 2.4 boosts Python 3.12 performance**: A user asked about **torch.compile** support on Python 3.12, to which another user responded that **Torch 2.4 or newer** is required.
   - A user saw the team updated some source code, but now the model generate response has some issues and is looking for help debugging.
- **Challenges Emerge Training GRPO on Llama3-8b**: A user reported that when training **GRPO** on **Llama3-8b**, the completion length and correct reward were not increasing.
   - They attached [graphs](https://cdn.discordapp.com/attachments/1179777624986357780/1346656044058808353/length.png?ex=67c8faac&is=67c7a92c&hm=20d71ba56e79387d6d40dd1a28651d148612265562a7cba55abf7320a431d4c4&) showing completion length issues and another [image](https://cdn.discordapp.com/attachments/1179777624986357780/1346656273076322435/image.png?ex=67c8fae3&is=67c7a963&hm=2e40f721f2feb17f7d44599136eacbb19d5d3161df966fe52e87ab146bb191bd&) depicting the reward issues.
- **BERT tuning quandaries clarified**: A user inquired about the possibility of fine-tuning **BERT** with **Unsloth**, but questioned if it was a valid approach since BERT primarily generates embeddings.
   - It was clarified that tuning BERT is **not currently supported**, though someone is considering implementing it in Unsloth when they have the time.
- **Windows users wonder: Ollama export?**: A user seeks guidance on how to export a fine-tuned model to **Ollama** format on the **Windows platform**.
   - They noted that the existing official documentation primarily covers the Linux version of the process.



**Link mentioned**: <a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth Documentation</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1346591787573317682)** (33 messages🔥): 

> `Llama 3.1 8B Fine-tuning, GPT4o-mini for Synthetic Data Generation, Training Data Generation Tricks, KoloLLM on Ollama, Fine-tuning vs. RAG` 


- **Llama 3.1 8B Gets Fine-Tuned**: A member is fine-tuning **Llama 3.1 8B** and using **GPT4o-mini** for synthetic data generation.
   - He emphasizes that training data is the main driving force.
- **Kolo Guide Open Sources Training Data Tricks**: The member shared [a link to his guide](https://github.com/MaxHastings/Kolo/blob/main/GenerateTrainingDataGuide.md) about training data generation, emphasizing that the tricks involve *small good decisions* that fully leverage the **LLM's** ability to generate fleshed-out answers to high-quality questions.
   - He also mentioned he uploaded **KoloLLM** to [Ollama](https://ollama.com/MaxHastings/KoloLLM) and offered help with generating training data to reproduce his results in different domains.
- **Fine-tuning Gives Super Powers**: The member expressed excitement about fine-tuning with high-quality data and good training parameters, calling it *like having a super power*, and automating the process with a simple configuration file, which he calls *no RAG bullshit*.
   - He emphasized that fine-tuning injects information into the LLM's brain and one never has to worry that the complex **RAG** system works every time reliably.
- **Fine-tuning plus RAG**: Another member noted that **LLMs** don't learn verbatim, so it is often a double-edged sword and most of the time it's better to use both to ground your answers.
   - The original poster responded that he may not need RAG since he is able to fill in the gap for situations where it hallucinates by finding training data to help fill the gap.
- **When is RAG preferable?**: Another member suggested that if you need 100% accuracy you should never use LLM whether **RAG** or not - but that **RAG** is useful in areas that are legal/fiscal/research where you need good retrieval.
   - The original poster said he thinks **RAG** will always be useful for information that is always changing or very new, but fine-tuning should be for static information.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ollama.com/MaxHastings/KoloLLM">MaxHastings/KoloLLM</a>: A fine tuned Llama 3.1 8B domain expert on Kolo</li><li><a href="https://github.com/MaxHastings/Kolo/blob/main/GenerateTrainingDataGuide.md">Kolo/GenerateTrainingDataGuide.md at main · MaxHastings/Kolo</a>: The Fastest Way to Fine-Tune LLMs Locally. Contribute to MaxHastings/Kolo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1346375115084005388)** (21 messages🔥): 

> `GRPO Method Comparison, DeepSeek r1 Release, Bespoke Curator for Synthetic Data, Triton GPU Kernels, Unsloth Video Sessions` 


- **GRPO Methods Lack Tangible Comparisons**: Members discussed that there's *literally no tangible comparison between the **GRPO methods**, all we see is that for some reason grpo gets larger rewards faster with faster kl divergence (bad hyperparameters?)*.
   - One member suggested that benchmarks are needed to see the benefit, pointing out that **RLOO** also performs faster than **PPO**, referencing [this blogpost](https://kalomaze.bearblog.dev/why-does-grpo-work/).
- **DeepSeek r1 Catches the Frontier**: After a year of iteration, **DeepSeek** released **DeepSeek r1**, catching up to the frontier in the **LLM space** following their latest pretraining run (**DeepSeek-V3**).
   - The release prompted speculation about training advancements responsible for the performance lift, with some suggesting integration of algorithms like **Monte Carlo tree search**.
- **Bespoke Curator Synthesizes AI**: An open-source Python library called **Bespoke Curator** by Bespoke Labs facilitates the creation and curation of synthetic datasets for **AI model fine-tuning** and **structured data extraction**.
   - It offers programmability, structured output, and integrates with **Hugging Face Dataset** objects, using an interactive Curator Viewer for real-time inspection, as described on their [website](https://www.bespokelabs.ai/).
- **Tilelang Kernels Get Speedy**: A member shared a link to a tweet about **tilelang kernels**, claiming that 80 lines of code can achieve 95% performance of **deepseek flashmla** (500% faster than triton) on **H100**.
   - Others wondered why anyone would want 95% of flashmla, with the original poster clarifying that [Tilelang](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla) is a language.
- **Unsloth Sessions Hit the Record Button**: There was interest in recording **Unsloth sessions** for sharing in the voice channel, with **OBS Studio** ([https://obsproject.com/](https://obsproject.com/)) being researched as a free and open-source option.
   - Another option suggested was **Scribe** ([https://scribehow.com/](https://scribehow.com/)), a paid software that helps capture and share team expertise with documentation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.bespokelabs.ai/">Bespoke Labs</a>: no description found</li><li><a href="https://www.youtube.com/playlist?list=PLPefVKO3tDxOJLAmCA75uShbe1z_RNqkQ">Triton GPU Kernels 101</a>: Triton lessons for those already competent in PyTorch but with no GPU hardware knowledge</li><li><a href="https://obsproject.com/">Open Broadcaster Software | OBS</a>: no description found</li><li><a href="https://x.com/Lei_Wang_1999/status/1896625837782475231?t=INByKNCTjnhy5DCDZ2Orfg&s=19">Tweet from Lei Wang (@Lei_Wang_1999)</a>: Checkout today’s tilelang kernel, 80 lines of tilelang kernel code you can get 95% performance of deepseek flashmla (500% faster than triton) on H100!: https://github.com/tile-ai/tilelang/tree/main/ex...</li><li><a href="https://kalomaze.bearblog.dev/why-does-grpo-work/">Why does GRPO work?</a>: DeepSeek.

Everyone in the LLM space knows that name now, and for good reason; after a little over a year of their team quietly iterating on architecture, ...</li><li><a href="https://kalomaze.bearblog.dev/grpo-judge-experiments-findings-and-empirical-observations/">GRPO Judge Experiments: Findings &amp; Empirical Observations</a>: Many GRPO reproductions for LLM reinforcement learning available online lack useful intuitions or recommendations regarding hyperparameters and reward shapi...</li><li><a href="https://scribehow.com/">Scribe | Create Step-by-Step Guides — Fast.</a>: Scribe documents your processes for you. Build visual guides with text, links and screenshots instantly.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1346331416421601280)** (314 messages🔥🔥): 

> `Android vs iOS features, Perplexity Pro pricing issues, Broken UI, Augment Code, Deep Research in Perplexity AI` 


- **Android Features are funny, iOS is premium**: A member joked about **Android** version being funny and then another responded that new voice it's only for **iOS** and **iOS** is more premium which led to some discussions.
   - Another member replied that *it's not classist* and more like *it's standard among LLM sites*.
- **Users frustrated with Perplexity Pro Pricing Issues**: Users voiced frustrations about the **200 USD** price tag of Perplexity Pro, deeming it excessive regardless of the service quality.
   - The discussion extended to whether the Pro subscription is worth the cost, especially if **Sonar** is used for most searches to cut expenses and a year of Pro being given out like hot candy.
- **Perplexity UI is always broken, Pro Search is always Sonar Model**: Members reported a broken UI: when clicking **Rewrite** in a **Pro Search** conversation on the web app and selecting a different model (e.g., Sonnet or Gemini), it doesn't actually use a different model.
   - Users complain that in **Pro Search** model defaults to **Sonar** with no ability to see model name on the UI and also no ability to change the underlying model during rewrite.
- **Augment Code for coding**: Members discussed **Augment Code**, which is designed to handle very large enterprise codebases effectively, contrasting it with tools like **Cursor** that perform indexing locally.
   - They noted that **Augment Code** indexes codebases on its servers, providing AI models comprehensive access to the codebase and discussed the [pricing and trial options](https://www.augmentcode.com/pricing).
- **Deep Research is deep**: Users discussed if they find Perplexity Pro's **Deep Research** mode is good for searching information and knowledge.
   - They replied that the depth is really good and it hits all the same major points but the output length is shorter than **OpenAI**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arc.net/l/quote/kbufudwj">Quote from “turbopuffer: fast search on object storage”</a>: no description found</li><li><a href="https://www.augmentcode.com/pricing">Pricing | Augment Code</a>: Augment Code is more than a tool, it accelerates your team. See why leading startups to Fortune 500 organizations choose Augment.</li><li><a href="https://x.com/monnef/status/1896817782412792154">Tweet from mennof (@monnef)</a>: I see @perplexity_ai is on a mission to make things worse, a journey of UI regression:* Full name displayed clearly 😊* Then hidden in hover 🙄* Replaced with &#34;Pro Search&#34; 🤢* Hidden behind a ...</li><li><a href="https://www.augmentcode.com/">Augment Code – Developer AI for real work</a>: Experience the AI platform that truly understands your codebase. Our developer AI helps teams code faster, make smarter decisions, and unlock collective knowledge. Try free today.</li><li><a href="https://www.reddit.com/r/Astronomy/comments/dwfdfq/grand_map_of_the_night_sky/#lightbox">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/perplexity_ai/status/1890452005472055673">Tweet from Perplexity (@perplexity_ai)</a>: Introducing Deep Research on Perplexity.Deep Research lets you generate in-depth research reports on any topic.Available to everyone for free—up to 5 queries per day for non-subscribers and 500 querie...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1346382289281482775)** (5 messages): 

> `Explain Gen, Apple Air Product Teased, TSMC Investment Plans, Investment Thesis on, Reasoning LM SOTA and Future` 


- **Brief Explanation of Gen incoming!**: A user requested a brief explanation about [Gen](https://www.perplexity.ai/search/explain-in-brief-about-the-gen-559BcexOTEyTKOUp6l71NQ).
- **New Apple Air Product Teased**: A user shared a link about a teased [Apple Air product](https://www.perplexity.ai/page/apple-air-product-teased-QhTieZlcTwWodiMLzGzP3g).
- **TSMC's Massive US Investment**: A user posted about [TSMC's plans for a $100 billion US investment](https://www.perplexity.ai/page/tsmc-plans-100-billion-us-inve-8m1ORdvqQlev._StpFWN8w).
- **Crafting Investment Thesis**: A user requested assistance writing an [investment thesis](https://www.perplexity.ai/search/write-an-investment-thesis-on-FoN2WGDESQ6ShLyTajmPIA).
- **Reasoning LM State of the Art**: A user shared a link about the [state-of-the-art in reasoning language models](https://www.perplexity.ai/search/reasoning-lm-sota-and-future-d-.fkmD_aXRrK80ruahJ_Rrw).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1346370886269145129)** (9 messages🔥): 

> `Perplexity AI Pricing, Obsidian Web Clipper integration, Perplexity API JSON output issues` 


- **Perplexity Plans for Pay-per-Deepening-Query Model**: A user requested a "continue deepening" feature with usage-based pricing (e.g., **$5-10 per hour**) to allow for more in-depth research and information processing on scientific and professional topics.
   - The user also suggested an option to query multiple AIs and consolidate the information, similar to what [FreedomGPT](https://freedomgpt.com/) offers.
- **Obsidian Web Clipper Considers PPLX Integration**: A user proposed integrating **Perplexity AI's LLMs** into the [Obsidian Web Clipper](https://github.com/obsidianmd/obsidian-clipper/issues/376).
   - The user believes it would be valuable and hopes someone with more skill will attempt the integration.
- **Sonar Reasoning Pro Model struggles with JSON output**: A user reported issues with the **Perplexity API's** `response_format` option when using the **sonar-reasoning-pro model** on a Tier 3 account.
   - The response unexpectedly includes `<think>Some thinking text</think>` before the **JSON output**, leading the user to question if they are calling the **API** incorrectly or if the reasoning models do not properly support the `response_format` parameter.



**Link mentioned**: <a href="https://github.com/obsidianmd/obsidian-clipper/issues/376">Integration of Perplexity AI&#39;s LLMs into Obsidian Web Clipper · Issue #376 · obsidianmd/obsidian-clipper</a>: Integrating Perplexity AI&#39;s Large Language Models (LLMs) into the Obsidian Web Clipper would be highly beneficial. Contrary to the other models currently usable in Obsidian Web Clipper, Perplexity...

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1346476720408231957)** (2 messages): 

> `WPF Extension Issues, Visual Studio, Xcode Extension Error` 


- **WPF Extension Troubles in Visual Studio**: A member reported that the **WPF extension** is *"pretty awful"* for **Visual Studio**.
- **Xcode extension faces errors**: A member received an *"internal error occurred"* message with error ID **a9de9711b9ed431297eb00a945415d47** when using **Xcode**.


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1346334397535617105)** (230 messages🔥🔥): 

> `Windsurf font size adjustment, Flex credits pricing justification, Windsurf performance issues, Flow Credits Consumption with Claude 3.7, Team-based Cascade/Windsurf implementation` 


- **Windsurf Font Size Adjustment Quest**: A user inquired about increasing the font size in Windsurf, and another user suggested clicking *the little square top right* to adjust settings.
- **Windsurf Flex Credit Pricing Puzzle**: A user questioned the pricing of flex credits, highlighting that **2,000 credits (500 prompt + 1,500 flow)** cost **$15**, while **300 flex credits** alone cost **$10**, to which another user claimed *they're used as prompts or flow actions, based on the need*.
- **Windsurf Experiences Bumps**: Users reported issues with Windsurf, including a persistent 'i' icon, periods of inactivity, console errors like *couldn't create connection to server*, and general unresponsiveness.
   - One user experiencing constant errors switched to Google's DNS and found that it resolved their issue.
- **Users Bemoan Heavy Flow Credit Consumption with Claude 3.7**: Several users complained about **Claude 3.7** models rapidly depleting **Flow Credits**, making Windsurf unsustainable for daily use, especially considering how quickly the credits are burned even for simple tasks.
   - One user mentions that *Windsurf reads the code from the beginning every time like a fool*, and another reports that *the ratio is now like 10x your user prompt credits.*
- **Windsurf implements code in separate project folders**: A user inquired about implementing **Cascade/Windsurf** in team projects with separate frontend and backend folders, asking about implementation plans and best practices for team collaboration.
   - One member suggests using a *master Tasklist file* to bullet-point the whole plan, and then have the agent digest the tasklist and come up with how the system would work together


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AnthropicAI/status/1764653830468428150">Tweet from Anthropic (@AnthropicAI)</a>: Today, we&#39;re announcing Claude 3, our next generation of AI models. The three state-of-the-art models—Claude 3 Opus, Claude 3 Sonnet, and Claude 3 Haiku—set new industry benchmarks across reasonin...</li><li><a href="https://artificialanalysis.ai/">AI Model &amp; API Providers Analysis | Artificial Analysis</a>: Comparison and analysis of AI models and API hosting providers. Independent benchmarks across key performance metrics including quality, price, output speed &amp; latency.</li><li><a href="https://github.com/Exafunction/codeium/issues/129">Api Server Wire Error (User is Disabled by Team) · Issue #129 · Exafunction/codeium</a>: Connection Error api server wire error: user is disabled by team I&#39;ve tried restarting, logging back in and out, and nothing is fixing it.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1346336998926843957)** (153 messages🔥🔥): 

> `Claude API understanding, Groq pricing and performance, Aider and git practice, Sonnet 3.7 tuning, Aider with Zed` 


- **Claude struggles communicating with Frontend and Backend APIs**: A member asked for help on how to make **Claude** understand both frontend and backend APIs, as it wrote two completely separate APIs that didn't know how to communicate.
   - Another member suggested forcing it to review the APIs and consolidate them, or regenerate the codebase.
- **Groq Specdec is Faster but more expensive**: Members discussed **Groq's specdec** (speculative decoding), noting it's about **20% more expensive** but offers **>5x the speed** compared to the versatile model.
   - Some argue **Gemini** is better for summarization due to the input/output ratio, while others suggest smaller models like *llama-3.2-3b-instruct* for summarization.
- **Aider can help practice git**: A member suggested using **Aider** to practice **Git** skills by having it create a chain of commits with different issues and exercises.
   - Another member shared a link to [Oh My Git!](//blinry.itch.io/oh-my-git), an open source game about learning Git that visualizes the internal structures of Git repositories in realtime.
- **Taming Sonnet 3.7 with Testing and Guardrails**: Members discussed the challenges of using **Sonnet 3.7**, noting that it often makes excessive changes and requires careful prompting and guardrails.
   - The community suggested the best way to tune the AI is to make mistakes and learn and to add conventions through documentation.
- **Aider runs in Zed IDE**: A member mentioned using **Aider** in the **Zed** editor and found it light and fast.
   - There was also discussion if it was possible to enable Gemini to have long context windows and caching.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ohmygit.org">Oh My Git!</a>: An open source game about learning Git</li><li><a href="https://aider.chat/docs/llms/gemini.html">Gemini</a>: aider is AI pair programming in your terminal</li><li><a href="https://x.com/CohereForAI/status/1896923657470886234">Tweet from Cohere For AI (@CohereForAI)</a>: Introducing ✨ Aya Vision ✨ - an open-weights model to connect our world through language and visionAya Vision adds breakthrough multimodal capabilities to our state-of-the-art multilingual 8B and 32B ...</li><li><a href="https://github.com/dnakov/claude-code/blob/main/src/constants/prompts.ts">claude-code/src/constants/prompts.ts at main · dnakov/claude-code</a>: claude-code full original source code from source maps - dnakov/claude-code
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1346333608217935883)** (69 messages🔥🔥): 

> `Less smart models can be used by Architect, Aider update API key, Custom model ctxlen cost, Ericsson CodeChecker added in project, Set global settings to Aider` 


- **Less Smart Models for Basic Architect Changes?**: It was suggested that a less sophisticated model might suffice for implementing the changes outlined by the **Architect** within the **aider** workflow.
   - One user mentioned using **o3-mini** for this purpose.
- **API Key Woes with Custom LLM Providers**: A user utilizing a custom LLM provider with short-lived API keys (expiring hourly) sought a way to update the API key within **aider** without restarting, as it disrupts chat history.
   - Suggestions included using the *restore history* argument, implementing an **nginx reverse proxy** for key rotation, or exploring the use of dummy keys.
- **Custom Model Configuration Deep Dive**: A user inquired about specifying **ctxlen/cost** externally for a custom model (groq/deepseek-r1-distill-llama-70b-specdec) not listed in **aider**'s model list.
   - It was pointed out that a `.json` file in the [documentation](https://github.com/domelqq/aider/tree/workon_feature) handles this, enabling users to configure models not natively supported.
- **Bizarre Ericsson Code Insertion Mystery**: A user reported instances of large blocks of code from **Ericsson/codechecker** being unexpectedly inserted into their project when using **aider v0.75.1** with **claude-3-7-sonnet**.
   - The user suspected this was caused by the LLM.
- **Global Aider Settings Expedition on Windows**: A user sought guidance on configuring global settings for **aider** on Windows, aiming to avoid duplicating the `.aider.conf.yml` file across multiple project folders.
   - The solution involved placing the `.aider.conf.yml` file directly in the `C:/User/me` directory, ensuring it's not within the `.aider` subfolder.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/bsilva96/status/1897042309314961865">Tweet from Benjamín Silva (@bsilva96)</a>: In LLMs context is the most important thing. When working in large code repositories Cursor  is unable to provide all the files of the repo to respond to your basic&#34;Explain me what does this XX fe...</li><li><a href="https://github.com/domelqq/aider/tree/workon_feature">GitHub - domelqq/aider at workon_feature</a>: aider is AI pair programming in your terminal. Contribute to domelqq/aider development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/releases">Releases · Aider-AI/aider</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1346394835405508650)** (2 messages): 

> `Karpathy inspires voice AI, Claude 3.7 Sonnet, Claude Code Agent` 


- **Karpathy Inspires Aider Devs**: A member was inspired by **Karpathy** to engage more with **AI** via voice, noting that text communication is too slow.
   - They linked a [YouTube video](https://www.youtube.com/watch?v=jCVO57fZIfM) discussing **Claude 3.7 Sonnet** and **Claude Code** as potential game changers, with **Claude Code** being described as the greatest **AI Agent**.
- **Claude 3.7 Sonnet and Claude Code as Game Changers**: The linked [YouTube video](https://www.youtube.com/watch?v=jCVO57fZIfM) suggests that **Claude 3.7 Sonnet** and **Claude Code** are potential game changers in the AI space.
   - The video description highlights **Claude Code** as potentially the *greatest AI Agent*.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=jCVO57fZIfM">GPT-4.5 FLOP? Claude 3.7 Sonnet STARTER PACK. What is Claude Code REALLY?</a>: 🔥 GPT-4.5 maybe the biggest FLOP. MEANWHILE Claude 3.7 Sonnet and Claude Code just CHANGED THE GAME! 🤯Anthropic&#39;s Claude Code maybe the greatest AI Agent T...

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1346346835157192734)** (67 messages🔥🔥): 

> `LM Studio CLI, Reporting LM Studio Vulnerabilities, LM Studio and Home Assistant, GPT-4o error loading, Model Settings` 


- **LM Studio CLI Commands Released**: LM Studio released and documented the LM Studio CLI (`lms`) commands and [developers can view the documentation online](https://lmstudio.ai/docs/cli) for scripting and automating local LLM workflows with an **MIT License** at [https://github.com/lmstudio-ai/lms](https://github.com/lmstudio-ai/lms).
   - The CLI ships with LM Studio and can be found under `/bin` in the LM Studio's working directory, and must run at least once.
- **Report LM Studio Vulnerabilities Safely**: A member reported a potential vulnerability and the suggestion was to email details (without zip attachments) to [bugs@lmstudio.ai](mailto:bugs@lmstudio.ai) in **plain text** with proof of concept, video, and screenshots.
   - The emphasis was on avoiding zip attachments due to security concerns, suggesting instead sending all the information directly in the email body.
- **Can LM Studio Control Home Assistant?**: A user inquired about using LM Studio with Home Assistant, and another user pointed to [a community thread](https://community.home-assistant.io/t/local-llm-for-dummies/769407/10) as a potential solution.
   - The user seeking help clarified that while they could integrate LM Studio, they were unable to control their home devices through it.
- **GPT-4o causes Error Loading Model**: A user encountered an error loading the *lmstudio-community/Phi-4-mini-instruct-GGUF Q4* model due to an *unknown pre-tokenizer type: 'gpt-4o'*, and confirmed their model was downloaded after updating.
   - The user resolved this issue by ensuring that their LM runtimes were updated to the beta version (via ctrl+shift+r).
- **PDF Uploads to LM Studio Server Incoming Soon**: A user asked about directly uploading PDF documents to the LM Studio server using the Python SDK.
   - A developer responded that this feature is coming very soon, leveraging *pdf2json* (as noted in [LM Studio acknowledgements](https://lmstudio.ai/acknowledgements.html)) for content extraction.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://community.home-assistant.io/t/local-llm-for-dummies/769407/10">Local LLM for dummies</a>: I have no idea but you just can try.  It won’t hurt your machine and it is easily removed again.</li><li><a href="https://lmstudio.ai/docs/cli">lms — LM Studio&#x27;s CLI | LM Studio Docs</a>: Get starting with the lms command line utility.</li><li><a href="https://youtu.be/CUy1ZOVk7QE?si=EG7nShA-NX7v1Q2h">Goku vs Jiren English dub with Original Ultimate Battle V2</a>: https://youtu.be/SRWsHICBt3Qhttps://youtu.be/Cl05ShiYAAs</li><li><a href="https://tenor.com/view/howaboutthat-avengers-ironman-tonystark-stark-gif-3525361">Power Is At 400% GIF - Howaboutthat Avengers Ironman - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/blade-runner-tears-in-rain-roy-batty-gif-3478115">Blade Runner GIF - Blade Runner Tears In Rain Roy Batty - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://youtube.com/shorts/IV32vPYPG7U?si=S3HbAE97wejtAdjO">Nvidia launching RTX 5090 be like</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1346410464514543669)** (50 messages🔥): 

> `4090 with 48GB VRAM, Pristine Datasets, NPU Support in LM Studio, Intel ARC iGPU with LM Studio, Model offloading to CPU/RAM` 


- **Modded 4090 with 48GB VRAM Surfaces**: A user inquired about a **4090** modded with **48GB** of VRAM, questioning if it performs the same as a standard **24GB 4090**, accompanied by an [image](https://cdn.discordapp.com/attachments/1153759714082033735/1346425229538230342/image0.png?ex=67c8cc76&is=67c77af6&hm=719728fa64976a052bef1165873cfff11f6eb1bb595a4a5d046728c3f7e31fd3).
- **Quest for Perfect Datasets Continues**: A user expressed surprise that *pristine datasets* aren't readily available yet, considering current model capabilities for iterative filtering.
   - Another user countered that defining *clean* is subjective, highlighting the challenge of agreeing on legal and ethical standards for AI training data.
- **LM Studio's NPU Support Status**: A user asked about running LM Studio on a laptop with an **ARC GPU** and **NPU**.
   - Another user replied that **NPUs are not yet supported**, suggesting they could *start writing the code* themselves to implement it.
- **Intel Arc iGPU with LM Studio Detects No VRAM**: A user reported that LM Studio detects their **Intel Arc iGPU**, but indicates **zero VRAM**.
   - Despite this, they noted the iGPU's theoretical **48 TOPS** performance, comparable to an **RTX 4080**, making compatibility potentially worthwhile.
- **Bypass VRAM limitations with CPU/RAM Offloading**: A user with **12GB VRAM** inquired about running larger models by offloading layers to **CPU** or **RAM**, as suggested on Reddit.
   - Another user explained that **LM Studio** allows manual adjustment of model parameters, including context length and **GPU offloading layers**, when selecting a model using the UI showed in the attached [image](https://cdn.discordapp.com/attachments/1153759714082033735/1346552868752195644/image.png?ex=67c89a95&is=67c74915&hm=9f7b97ef96c40d1c40130da10685a2be1e5be7c47e7b4103a0dc349c58b11562&).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1346362884548067348)** (39 messages🔥): 

> `Anthropic raises $3.5 billion, Hugging Face LLM as a local microservice, Qwen2.5-Coder models, Bengali Text Romanization, Whisper Japanese models` 


- **Anthropic Achieves Astronomical Valuation**: [Anthropic](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation) has raised **$3.5 billion** led by **Lightspeed Venture Partners**, achieving a **$61.5 billion** post-money valuation.
- **Hugging Face LLM Invokes Local Microservice**: Members discussed the possibility of running a **Hugging Face-based LLM** as a local microservice and making API calls to it.
   - One member suggested using **TGI** ([Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/index)), a toolkit for deploying and serving Large Language Models (LLMs).
- **Qwen2.5-Coder Excels at Coding Tasks**: The **Qwen2.5-Coder** series ([Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) and [Qwen2.5-Coder-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)) has significantly improved in **code generation**, **code reasoning**, and **code fixing**.
- **Flan-T5 Fan Finetunes Features**: A member is working on finetuning a model for **Romanizing Bengali Language Text** and asked for advice on which base model to use.
   - Another member mentioned that **FLAN** is basically instruct-tuned **T5**.
- **Users Translate Troubles with Turbo Whisper**: A member noted that the translation feature of **Whisper** only translates from the source language to English, expressing disappointment that it couldn't translate from English to Japanese with **whisper-large-v3-turbo**.
   - Another member suggested trying **Whisper Japanese models**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AnthropicAI/status/1896606683876753470?t=rG-iRTyhiMnSiTe-kJeOXQ&s=19">Tweet from Anthropic (@AnthropicAI)</a>: Anthropic has raised $3.5 billion at a $61.5 billion post-money valuation, led by Lightspeed Venture Partners.This will advance our development of AI systems, deepen our understanding of how they work...</li><li><a href="https://spinningup.openai.com/en/latest/">Welcome to Spinning Up in Deep RL! &mdash; Spinning Up  documentation</a>: no description found</li><li><a href="https://readthedocs.org).>>>">no title found</a>: no description found</li><li><a href="https://huggingface.co/docs/text-generation-inference/en/index">Text Generation Inference</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct">Qwen/Qwen2.5-Coder-7B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF">Qwen/Qwen2.5-Coder-3B-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/fluently/Fluently-XL-v3-inpainting">fluently/Fluently-XL-v3-inpainting · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

ericmaubr: Today I saw my first video on RLHF. https://www.youtube.com/watch?v=2MBJOuVq380
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1346339348949237864)** (10 messages🔥): 

> `AWQ conversion of InternVL 2.5, Ukrainian Text-to-Speech, Bias in LLMs, Multi-Agent Approach to AGI, Working Embedders` 


- **InternVL 2.5 gets AWQ Conversion**: A member shared their [AWQ conversion of InternVL 2.5](https://huggingface.co/rootonchair/InternVL2_5-4B-AWQ), noting minimal performance degradation and compatibility with the *transformers* library.
   - It was noted that while **Bitsandbytes** doesn't require calibration and is faster to convert, **AWQ** might offer better performance.
- **Ukrainian Text-to-Speech Model Released**: A member announced a stable release of a [Ukrainian Text-to-Speech model](https://github.com/egorsmkv/tts_uk) on GitHub and PyPI, featuring **three voices** (2 female, 1 male) and fine-grained control over speech parameters.
   - The model uses **RAD-TTS++** for acoustic modeling and **Vocos** for fast vocoding, supporting a sampling rate of **44.1 kHz** and tested on both Linux and Windows/WSL.
- **LLMs Mini Bias Eval**: A member shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1j1nen4/llms_like_gpt4o_outputs/) discussing a mini evaluation on bias in LLMs, where LLMs were asked to grade each other based on estimated abilities and known information.
   - The evaluation considered what the judging LLM knows about the other model company.
- **Multi-Agent AGI Research Overviewed**: A member shared a [YouTube video](https://www.youtube.com/watch?v=cryI1-3Om9c) discussing research on the multi-agent approach to AGI, covering multi-agent scaling laws and the benefits of model diversity.
   - The video highlights how diverse models, architectures, and knowledge within systems enhance performance.
- **Collection of Working Embedders Assembled**: A member shared a [collection of working embedders](https://huggingface.co/kalle07/embedder_collection), tested with ALLM (AnythingLLM), highlighting *nomic-embed-text*, *mxbai-embed-large*, *mug-b-1.6*, and *Ger-RAG-BGE-M3* (German) as performing well.
   - The member provided tips for using these embedders, suggesting context length and snippet settings for optimal performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/kalle07/embedder_collection">kalle07/embedder_collection · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=cryI1-3Om9c">What is the Multi-Agent Approach to AGI?</a>: Naptha is a framework and infrastructure for developing and running multi-agent systems at scale with heterogeneous models, architectures and data. Agents an...</li><li><a href="https://huggingface.co/rootonchair/InternVL2_5-4B-AWQ">rootonchair/InternVL2_5-4B-AWQ · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1j1nen4/llms_like_gpt4o_outputs/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://reddit.com/en-us/policies/cookies)">Reddit - Dive into anything</a>: no description found</li><li><a href="https://reddit.com/en-us/policies/privacy-policy).>>>">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/egorsmkv/tts_uk">GitHub - egorsmkv/tts_uk: High-fidelity speech synthesis for Ukrainian using modern neural networks.</a>: High-fidelity speech synthesis for Ukrainian using modern neural networks. - egorsmkv/tts_uk</li><li><a href="https://huggingface.co/spaces/Yehor/radtts-uk-vocos-demo">RAD-TTS++ Ukrainian (Vocos) - a Hugging Face Space by Yehor</a>: no description found</li><li><a href="https://huggingface.co/chat/)">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://huggingface.co/chat/huggingchat/logo.svg)">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://apps.apple.com/us/app/simply-scan/id6742599424">‎Simply Scan</a>: ‎Simple Scanner transforms your iPhone into a powerful document digitizer. With just a tap, capture crystal-clear scans of any document using our intelligent edge-detection technology that automatical...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1346411246429278228)** (4 messages): 

> `ViT CLS token, DETR from scratch, GAP vs CLS, DETR inference` 


- **ViT CLS Token Acts Globally**: In **ViT**, the **CLS token** is updated via self-attention along with patch embeddings, aggregating global information about the image for classification.
   - Because it is a learnable parameter, its weights are optimized during backpropagation in relation to the rest of the parameters in the **multi-head attention (MHA) blocks**.
- **Why ViT prefers CLS token over GAP**: A member explained how **CLS token** makes use of self attention which attends the most relevant image patches, whereas **GAP** weights all patches equally.
   - A second member agreed, but stated *there is no difference or a very small difference in performance between these two methods if models are trained with the right hyperparameters*.
- **DETR from Scratch struggles with Batch Size 1**: One member built a **DETR** from scratch and fine-tuned a model on a small dataset (~1000 images), using a pre-trained **ResNet50** backbone from torchvision.
   - The model gets decent results except when using **batch size 1** during inference; the member shared a [GitHub repo](https://github.com/dimiz51/DetectionTransformer-DETR-PyTorch) showing the implementation.



**Link mentioned**: <a href="https://github.com/dimiz51/DetectionTransformer-DETR-PyTorch">GitHub - dimiz51/DetectionTransformer-DETR-PyTorch: This project is an implementation of the Detection Transformer (DETR) for state-of-the-art object detection using the well-known Transformer architecture. Using this project you can easily fine-tune and test DETR on your own dataset following the included notebook guide.</a>: This project is an implementation of the Detection Transformer (DETR) for state-of-the-art object detection using the well-known Transformer architecture. Using this project you can easily fine-tun...

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1346389741746126889)** (5 messages): 

> `Text Extraction with SpaCy, Hosting Phi-4 with Ollama and FastAPI, Accelerate CLI memory estimation` 


- **SpaCy Extracts Elements From Text**: A member is working on a project to [extract elements from text using SpaCy](https://spacy.io/usage/spacy-101) with a REGEX to capture numeric values and dependency parsing for sentence structures.
   - They expressed uncertainty about using NER for parameters due to their length and variability in writing styles across different documents and contributors.
- **Hosting Phi-4 with Ollama and FastAPI**: A member inquired about the [GPU requirements for hosting Phi-4](https://www.microsoft.com/en-us/research/blog/phi-4-a-breakthrough-in-language-model-efficiency/) using Ollama and FastAPI for inference.
   - They were seeking guidance on how to calculate the necessary GPU resources for their setup.
- **Accelerate CLI Estimates Memory**: A member shared a link to the Hugging Face documentation on using the [Accelerate CLI](https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator) to estimate model memory requirements.
   - The CLI loads the model into memory on the `meta` device, and it supports searching for models that can be used in `timm` and `transformers`.



**Link mentioned**: <a href="https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator">Model memory estimator</a>: no description found

  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1346346458252709978)** (50 messages🔥): 

> `Gradio App Internals, LlamaIndex Notebook Error, SmolAgents vs SmolTools, Payment Required Issue, Deep Reinforcement Learning Resources` 


- **Gradio App Under the Microscope**: A user inquired whether the decode **Gradio app** on the `What are LLMs?` page utilizes **agents** internally to generate graphics and results, asking about the possibility of learning to build similar apps and accessing their code.
   - The user expressed appreciation for the high-quality, free tutorials provided by the team.
- **LlamaIndex Notebook Link Flounders**: Multiple users reported a **404 error** when accessing the **LlamaIndex notebook** at Unit 2.2 ([https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/components.ipynb](https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/components.ipynb)).
   - A staff member later confirmed the issue had been resolved.
- **SmolAgents Framework and SmolTools Library Separated**: Users sought clarification on the distinction between **SmolAgents** and **SmolTools**, noting both are referred to as libraries in the course unit.
   - It was clarified that *SmolAgents is basically a framework to create lightweight agents* and *SmolTools contains utility functions and prebuilt tools that can be used within smolAgents*.
- **Inference Credits are Insolvent**: Several users reported encountering a *payment required issue* due to exceeding their monthly included credits for **Inference Providers**.
   - One user suggested using an alternative model ([https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud](https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud)) as a potential workaround.
- **Deep Reinforcement Learning Discussions**: A user shared resources for **Deep Reinforcement Learning (DRL)**, including the [Hugging Face Learn DRL course](https://huggingface.co/learn/deep-rl-course/unit0/introduction), a link to the book **Reinforcement Learning: An Introduction** ([http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)), and the **DeepMind x UCL Deep Learning Lecture Series 2021** on YouTube ([https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&feature=shared](https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&feature=shared)).
   - Other users offered to provide more insights about these resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud'))">no title found</a>: no description found</li><li><a href="https://mysterybox-nft18-eta.vercel.app/">CLICK HERE - OPENSEA PRO NFT</a>: &#128994; AIRDROP IS LIVE NOW &#128994; &#127881; Price: FREE&#127881; Supply: 150 Mystery Box&#127881; Reward: between $3000 and $250,000TRY YOUR LUCK ! &#128640;    </li><li><a href="https://huggingface.co/agents-course/notebooks/blob/main/unit2/llama-index/components.ipynb">unit2/llama-index/components.ipynb · agents-course/notebooks at main</a>: no description found</li><li><a href="https://huggingface.co/learn/deep-rl-course/unit0/introduction">Welcome to the 🤗 Deep Reinforcement Learning Course - Hugging Face Deep RL Course</a>: no description found</li><li><a href="http://incompleteideas.net/book/the-book-2nd.html">Sutton &amp; Barto Book: Reinforcement Learning: An
Introduction</a>: no description found</li><li><a href="https://youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm&feature=shared">DeepMind x UCL | Deep Learning Lecture Series 2021</a>: The Deep Learning Lecture Series is a collaboration between DeepMind and the UCL Centre for Artificial Intelligence.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1346590262897545316)** (1 messages): 

> `BYOK Errors, API Key Issues` 


- **BYOK Requests Encounter Errors**: Most **BYOK requests** (for users who attached their own API key in settings) were showing errors for the past 30 minutes.
   - The relevant change was reverted, and the team is adding extra safeguards to prevent this from happening again.
- **Mitigating API Key Errors**: A recent issue caused errors for users using their own **API keys** (BYOK) in settings.
   - The team has reverted the problematic change and is implementing additional safeguards to ensure stability for user-provided API keys.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1346385172995510302)** (2 messages): 

> `Price Estimates, Total Price Combinations` 


- **Estimating Total Price Combinations**: A member shared a method to reveal a total price with huge input and output, allowing for a quick and easy price estimate.
- **Revealing Prices**: The goal is to reveal a total price, even with significant input and output variables.
   - This method aims to provide a relatively quick and easy price estimation process.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1346381520155312131)** (105 messages🔥🔥): 

> `OpenRouter Provider Routing, Strongest AI For bookmark processing, Flash 2.0 vs GPT-4o-mini, Inception AI Diffusion Models, Anthropic 502 Overload Errors` 


- **OpenRouter Provider Routing Configuration**: A user needed to route requests through a specific provider and was instructed to modify the API request body with a `provider` object, specifying the desired provider(s) in the `order` array and setting `allow_fallbacks` to false, as documented in the [OpenRouter docs](https://openrouter.ai/docs/features/provider-routing#json-schema-for-provider-preferences).
   - It was emphasized that the provider name must **exactly** match the name listed on the OpenRouter model page (e.g., `Nebius`), and quotes are required around provider names in the JSON.
- **Groq-3 and GPT-4.5 Lead in bookmark processing**: For processing a large number of bookmarks, **Groq-3** (no API) and **GPT-4.5** (expensive) are recommended, with **DeepSeek v3** and **Claude 3.7** as runner-ups, but another user indicated that **ChatGPT** (likely GPT-4o via the web interface) was able to accomplish the task.
   - The user was surprised because ChatGPT returned *You've reached your data analysis limit.* after only *1 file* upload.
- **Flash 2.0 blows GPT-4o-mini out of the water**: **Flash 2.0** is recommended as a stronger and slightly cheaper alternative to **GPT-4o-mini**.
   - One user said it *blows 4o mini out of the water significantly smarter*.
- **Inception AI's diffusion models requested on OpenRouter**: A user requested access to **Inception AI's** diffusion models via OpenRouter after [TechCrunch wrote about their DLM (Diffusion-based Large Language Model)](https://techcrunch.com/2025/02/26/inception-emerges-from-stealth-with-a-new-type-of-ai-model/).
   - OpenRouter is in contact with **Inception AI** and is excited to bring them online as soon as possible.
- **Anthropic Overload Triggers 502 Errors**: Users reported receiving "overloaded" errors, which were identified as **502 status codes** from Anthropic, indicating capacity issues.
   - These **502 errors** can occur even without a declared incident on the status page, requiring users to retry their requests.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/docs/features/provisioning-api-keys">Provisioning API Keys - Programmatic Control of OpenRouter API Keys</a>: Manage OpenRouter API keys programmatically through dedicated management endpoints. Create, read, update, and delete API keys for automated key distribution and control.</li><li><a href="https://openrouter.ai/rankings/finance">LLM Rankings: finance | OpenRouter</a>: Language models ranked and analyzed by usage for finance prompts</li><li><a href="https://openrouter.ai/docs/features/provider-routing#json-schema-for-provider-preferences>">Provider Routing - Smart Multi-Provider Request Management</a>: Route AI model requests across multiple providers intelligently. Learn how to optimize for cost, performance, and reliability with OpenRouter&#x27;s provider routing.</li><li><a href="https://techcrunch.com/2025/02/26/inception-emerges-from-stealth-with-a-new-type-of-ai-model/">Inception emerges from stealth with a new type of AI model | TechCrunch</a>: Inception, a startup, claims to have developed a novel type of AI model based on a diffusion architecture.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1346421318689624125)** (43 messages🔥): 

> `Mojo vs C++ vs Rust similarities, Mojo's design perspective drawbacks, Python Superset in Mojo, Concurrency and Sum Types in Mojo, Python's API vs Rust's API consistency` 


- **Mojo: Rust with missing C++ features?**: A member described **Mojo** as being *like Rust, but with the stuff from C++ that really should have come over*.
   - The discussion touched on how familiarity with **Rust's memory management model** could be beneficial for understanding Mojo.
- **Mojo lacks language-level consistency**: One of the biggest downsides mentioned about **Mojo** is the lack of consistency at the language level in terms of naming conventions.
   - It was pointed out that while **Rust** maintains consistency across the whole language, **Mojo** has a mix of Python-like, C-like, and its own API, partly due to its initial design as a **Python superset**.
- **Python superset narration is ballasting Mojo**: Members discussed the initial concept of Mojo as a **Python superset**, with some feeling that this *narration* leads to unnecessary baggage, such as copied namings from `libc`.
   - It was clarified that the aim is not to be *bug compatible* with CPython, but rather to make it reasonable to port basic code with find and replace.
- **Concurrency and Sum Types are high priority features**: Members mentioned that proper **concurrency** and **sum types** are significant desired features for them.
   - One member pointed to [a GitHub pull request](https://github.com/modular/max/pull/3945) related to *Structured Async for Mojo* and [another about Effect Handlers](https://github.com/modular/max/pull/3946), indicating active development in these areas.
- **Python stdlib API consistency debated**: A member expressed hope that Mojo wouldn't drop the **Python API**, citing familiarity with Python and its libraries.
   - Another argued that **Rust** has more predictable APIs than Python and hoped Mojo wouldn't get stuck with the **Python stdlib API**, which includes a mix of C flatcase, Java camelCase, and Python snake_case.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modular/max/pull/3945">[proposal] Structured Async for Mojo by owenhilyard · Pull Request #3945 · modular/max</a>: Proposes to add structured async to Mojo, following in the the Rust tradition of async since Mojo has the ability to fix many of the issues with Rust&amp;#39;s async, some of which are ecosystem infli...</li><li><a href="https://github.com/modular/max/pull/3946">[proposal] Provided Effect Handlers by owenhilyard · Pull Request #3946 · modular/max</a>: This proposal contains an alternative to an effect system which I think is more suitable for abstracting async, raises, and similar function colors in a systems language where the context may not a...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1346416607399903272)** (15 messages🔥): 

> `Commandline argument conversion to int in Mojo, Mojo `is` operator for identity checks, Tensor addition operation removal, Mojo code build error: missing libtinfo` 


- **Mojo Newbie Seeks CLI Argument Conversion Clarity**: A member sought guidance on converting a command-line argument to an integer in Mojo, trying various methods without success.
   - Another member pointed out a type issue, suggesting a solution using `List[StaticString]` and `atol`, while noting a segfault issue with `StringSlice` origin.
- **`is` Operator Identity Crisis Resolved**: A member inquired about the meaning of *identity* in Mojo's `assert_is` function, asking if it checks for the same type, and another clarified it relates to memory location.
   - The respondent clarified that `is` checks if two objects reside at the same memory location, akin to pointer equality, but cautioned about complications with register-passable types, linking to the [Identifiable documentation](https://docs.modular.com/mojo/stdlib/builtin/identifiable/Identifiable/).
- **Tensor Addition Operation Gets the Axe**: A member reported that `Tensor[float64]` no longer implements the `__add__` method in the Mojo nightly.
   - Another member found a commit message explaining the removal was part of phasing out `Tensor` in favor of other vocabulary types, and recommending the use of `LayoutTensor` for more efficient elementwise operations, as [detailed in this commit message](https://github.com/modularml/mojo/commit/SOME_COMMIT_HASH).
- **Mojo Build Blues: Missing libtinfo**: A member encountered a build error in Mojo due to a missing `-ltinfo` library.
   - Another member identified the issue as a missing `libtinfo` and suggested the fix `sudo apt-get install libtinfo-dev`.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1346346901162688553)** (45 messages🔥): 

> `AI Timelines, Differential Transformers, Numerical Stability in Softmax, Bilevel Optimization, Grokking Phenomenon` 


- ****AI Timelines**: When Will Machines Think?**: Many AI experts predict human-level artificial intelligence could arrive within the next few decades, as discussed in [this article](https://ourworldindata.org/ai-timelines).
   - Human-level AI is defined as a machine capable of performing any task a human can, with the ability to choose actions that allow the machine to achieve them.
- ****Transformer Architecture** Gets a Differential?**: A newsletter highlights recent AI research, including [Differential Transformers](https://mail.bycloud.ai/), intelligence at the edge of chaos, and why LLMs might not truly reason.
   - It also mentions **Byte Latent Transformers** as a potential future for LLMs without tokenization.
- ****Softmax Instability** and Underflow's Grokking Role**: Discussion around a [LinkedIn post](https://www.linkedin.com/posts/damienbenveniste_the-softmax-transform-might-be-one-of-the-activity-7301720641559269377-gDdQ) reveals that while softmax addresses overflow, it can exacerbate underflow issues during gradient descent, potentially causing models to get stuck.
   - Some recent papers suggest underflow may contribute to the grokking phenomenon, acting as an implicit regularizer to prevent overfitting.
- ****Bilevel Optimization** for Sparsemax Generalization?**: A member suggests that **bilevel optimization** might generalize **Sparsemax** and **Stablemax**, potentially viewing the entire ANN through a “leader/followers” lens.
   - They coded a [BilevelMax class](https://www.dataia.eu/sites/default/files/1%20Marco-Pedersoli%20ILLS.pdf) to dynamically balance sparsity and density, smoothly transitioning between **Sparsemax** and **Softmax**.
- ****Outlines**: Structure outputs without finetuning?**: Members discussed how to enforce structured outputs (e.g., JSON) from LLMs without fine-tuning.
   - One member suggested using [Outlines](https://github.com/dottxt-ai/outlines), a library that modifies sampling logic to ensure correct output structure.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ourworldindata.org/ai-timelines">AI timelines: What do experts in artificial intelligence expect for the future?</a>: Many believe there is a real chance that human-level AI will be developed within the next decades, and some believe that it will exist much sooner.</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria">Storage quotas and eviction criteria - Web APIs | MDN</a>: Web developers can use a number of technologies to store data in the user&#x27;s browser (i.e., on the local disk of the device the user is using to view the website).</li><li><a href="https://developer.mozilla.org/en-US/docs/Web/API/StorageManager">StorageManager - Web APIs | MDN</a>: The StorageManager interface of the Storage API provides an interface for managing persistence permissions and estimating available storage. You can get a reference to this interface using either navi...</li><li><a href="https://mail.bycloud.ai/">The AI Timeline</a>: Follow The Latest Cutting Edge AI Research</li><li><a href="https://github.com/dottxt-ai/outlines">GitHub - dottxt-ai/outlines: Structured Text Generation</a>: Structured Text Generation. Contribute to dottxt-ai/outlines development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1346570280075526255)** (6 messages): 

> `DDPM, CLIP, MLST, GNN, Graph Attention Networks (GATs)` 


- ****Deep Dive** into DDPM and CLIP classics**: A member mentioned that **DDPM** (Denoising Diffusion Probabilistic Models) and **CLIP** (Contrastive Language-Image Pre-training) are considered *classics* in the field.
   - The member also noted listening to the **MLST** (Machine Learning Street Talk) podcast episode featuring **pi 0**.
- **Overview of Graph Attention Networks (GATs)**: A member shared an [overview](https://petar-v.com/GAT/) of **Graph Attention Networks** (**GATs**), which are neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions.
   - The overview includes motivating examples of graph-structured inputs such as molecular networks, transportation networks, social networks and brain connectome networks, including a link to the [original paper](https://arxiv.org/abs/1710.10903).



**Link mentioned**: <a href="https://petar-v.com/GAT/">Graph Attention Networks</a>: no description found

  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1346491088470278198)** (25 messages🔥): 

> `NextGenAI Funding, Thinking Machines hires PyTorch engineer, Amazon Nova reasoning model, OpenAI Credit Plans, CoreWeave to Acquire Weights & Biases` 


- **NextGenAI Funds Next Gen Science**: [OpenAI](https://openai.com/index/introducing-nextgenai/) is launching **NextGenAI**, its new initiative to provide funding for AI in science.
- **PyTorch Engineer Thinks Up Thinking Machines**: After 4 years on **PyTorch**, an engineer has joined **Thinking Machines** as a founding engineer, citing it as a compelling opportunity as detailed in [this tweet](https://x.com/cHHillee/status/1896973303241400704) and [this blog post](https://www.thonking.ai/p/why-pytorch-is-an-amazing-place-to).
- **Amazon Builds Nova, Aiming for AI Supernova**: Amazon is reportedly developing a cost-efficient, hybrid-architecture “reasoning” model under its **Nova** brand, aiming to rival models like **OpenAI’s o3-mini**, **DeepSeek’s R1**, and **Anthropic’s Claude 3.7 Sonnet** with a potential launch as soon as June, according to [this article](https://ift.tt/GPTB0aV).
- **Sama's Subscription Solution Sparks Scrutiny**: Sam Altman proposed a new paid plan for OpenAI where a **$20 plus subscription converts to credits** that can be used across features like deep research, o1, gpt-4.5, sora, etc., detailed in [this tweet](https://fxtwitter.com/sama/status/1897036361506689206).
- **CoreWeave Waves into Weights & Biases**: **CoreWeave**, the AI Hyperscaler™, announced it has reached an agreement to acquire **Weights & Biases**, a leading AI developer platform, for **$1.7B**, as detailed in [this press release](https://www.prnewswire.com/news-releases/coreweave-to-acquire-weights--biases---industry-leading-ai-developer-platform-for-building-and-deploying-ai-applications-302392342.html) and [this article](https://www.theinformation.com/briefings/coreweave-to-buy-weights-biases-for-1-7-billion).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/sama/status/1897036361506689206">Tweet from Sam Altman (@sama)</a>: an idea for paid plans: your $20 plus subscription converts to credits you can use across features like deep research, o1, gpt-4.5, sora, etc.no fixed limits per feature and you choose what you want; ...</li><li><a href="https://x.com/cHHillee/status/1896973303241400704">Tweet from Horace He (@cHHillee)</a>: Life update: After 4 years on PyTorch, I&#39;ve joined @thinkymachines! Over the years, I&#39;ve had several people ask me why I&#39;m so reluctant to leave.I want to talk about 1. why I&#39;ve stayed...</li><li><a href="https://www.thonking.ai/p/why-pytorch-is-an-amazing-place-to">Why PyTorch is an amazing place to work... and Why I&#x27;m Joining Thinking Machines</a>: In which I convince to you to join either PyTorch or Thinking Machines!</li><li><a href="https://thinkingmachines.ai/).">Thinking Machines Lab</a>: no description found</li><li><a href="https://chatgpt.com)">no title found</a>: no description found</li><li><a href="https://x.com/btibor91/status/1897011162493149303">Tweet from Tibor Blaho (@btibor91)</a>: Amazon is reportedly developing a cost-efficient, hybrid-architecture “reasoning” model under its Nova brand to compete with models like OpenAI’s o3-mini, DeepSeek’s R1, and Anthropic’s Claude 3.7 Son...</li><li><a href="https://www.prnewswire.com/news-releases/coreweave-to-acquire-weights--biases---industry-leading-ai-developer-platform-for-building-and-deploying-ai-applications-302392342.html">CoreWeave to Acquire Weights &amp; Biases - Industry Leading AI Developer Platform for Building and Deploying AI Applications</a>: /PRNewswire/ -- CoreWeave, the AI Hyperscaler™, today announced it has reached an agreement to acquire Weights &amp; Biases, a leading AI developer platform. The...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/)** (1 messages): 

saturatedfat: inspect the tex if its on arxiv
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/)** (1 messages): 

catboy_slimmer: usually people who behave like schmid are present but somewhat less prominent
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1346388436642168874)** (11 messages🔥): 

> `CogView4-6B, Anthropic vs OpenAI models, Aya Vision, LLM Summarization ethics, Meeting transcription summarization` 


- **CogView4-6B debuts on HuggingFace**: [CogView4-6B](https://huggingface.co/THUDM/CogView4-6B), a new model from THUDM, has been released, requiring image dimensions between **512px** and **2048px**, divisible by **32**, and supports **BF16** / **FP32** precision.
   - The model card notes that **FP16** is not supported due to overflow issues leading to completely black images.
- **Anthropic's Model Channels Inner Autist, OpenAI the Chill Bro?**: A user shared a tweet joking that *Anthropic gave 3.7 autism, but openai made 4.5 a chill guy* ([source](https://x.com/benhylak/status/1896684180928598448)).
- **Aya Vision launches in 23 languages**: [Aya Vision 8B](https://huggingface.co/CohereForAI/aya-vision-8b), an **8 billion** parameter model by Cohere For AI, was released with capabilities optimized for vision-language tasks in **23 languages** under the [CC-BY-NC license](https://cohere.com/c4ai-cc-by-nc-license).
- **Ethical Dilemmas when Summarizing**: A user raised an alignment question on Twitter regarding whether an LLM should withhold information, such as a surprise birthday party, when generating a summary for the person involved ([source](https://x.com/TheXeophon/status/1896817938323341722)).
   - The majority of respondents agreed that the **LLM should withhold the sensitive information**.
- **Elevenlabs scribe transcribes meetings with human quality**: A user recommends [elevenlabsio Scribe](https://x.com/dwarkesh_sp/status/1895966981020663833) for meeting transcription, noting its use of Gemini 2.0 for *human quality* transcripts with audio.
   - The user also shared a [link](https://gist.github.com/dwarkeshsp/65c232298781c86b33f5e32065152f1e) to a prompt and suggested blaming Claude for all coding mistakes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/THUDM/CogView4-6B">THUDM/CogView4-6B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/CohereForAI/aya-vision-8b">CohereForAI/aya-vision-8b · Hugging Face</a>: no description found</li><li><a href="https://x.com/benhylak/status/1896684180928598448">Tweet from ben (@benhylak)</a>: feels like anthropic gave 3.7 autism, but openai made 4.5 a chill guy.too soon to say which strategy will win.</li><li><a href="https://x.com/dwarkesh_sp/status/1895966981020663833">Tweet from Dwarkesh Patel (@dwarkesh_sp)</a>: I&#39;m happy to report that @elevenlabsio Scribe has made my kludgy previous solution unnecessaryNow all I need is automated links (to papers/books/ key terms/etc)Quoting Dwarkesh Patel (@dwarkesh_sp...</li><li><a href="https://huggingface.co/blog/aya-vision">A Deepdive into Aya Vision: Advancing the Frontier of Multilingual Multimodality</a>: no description found</li><li><a href="https://x.com/TheXeophon/status/1896817938323341722">Tweet from Xeophon (@TheXeophon)</a>: Alignment question:If an LLM is tasked to generate a summary for John and the text to summarize (chat logs) contains that John’s friends plan a surprise birthday party for him, should the LLM delibera...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1346389007772160072)** (3 messages): 

> `Claude Code, GitHub Issues, System Bricked` 


- **Claude Code Bricks System with Bad Command**: A user reported that a command recommended for automating **Claude** updates bricked their system, specifically `sudo chown -R $USER:$(id -gn) /usr && sudo chmod -R u+w /usr`, as described in [this GitHub issue](https://github.com/anthropics/claude-code/issues/168).
   - One member commented that recommending resetting the permissions on its own installation directory without checking where that directory is, is *poorly thought out*.
- **Discussion on Risky Permission Resetting**: The discussion thread humorously compares the situation to *buying a gun and shooting oneself in the foot*, highlighting the user's plight after running the command.
   - The command in question involved resetting permissions on the installation directory without proper consideration, leading to system instability.



**Link mentioned**: <a href="https://github.com/anthropics/claude-code/issues/168">Command bricked system · Issue #168 · anthropics/claude-code</a>: Ubuntu 24.02 server To automate claude updates it advises to run sudo chown -R $USER:$(id -gn) /usr &amp;&amp; sudo chmod -R u+w /usr This bricked the sudo system to no longer work and had to attach a...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1346555393228144862)** (2 messages): 

> `MSR Health Futures, image based multi-model work, NLP folks in healthcare` 


- **Microsoft's Health Futures pumps out goodies!**: Microsoft Research's **Health Futures** group is producing a lot of great work, especially around *image based multi-model* applications.
   - They also have solid **NLP** folks like **Hoifung Poon** and **Tristan Naumann** thinking about healthcare.
- **RL cover slide gets abused**: A user shared an RL cover slide image, presumably for feedback, and then followed up by saying they are sure that **this has been abused** lol.
   - The [image](https://cdn.discordapp.com/attachments/1208183216843005962/1346697225295630348/PNG_image.png?ex=67c92106&is=67c7cf86&hm=48c293128304e0409c421fdecdf248fdfd3f204f250fce3584e20a3287d5a6c7) was analyzed, but the results of the analysis are not included in the message history.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1346424542708629534)** (5 messages): 

> `Qwen vs Llama Self Improvement, ARC-AGI, Inference Paradigms` 


- **Cognitive Behaviors help LLMs Self Improve**: A new paper ([arxiv link](https://arxiv.org/abs/2503.01307)) investigates why some **LLMs self-improve** their reasoning while others plateau, finding that key **cognitive behaviors** like verification, backtracking, subgoal setting, and backward chaining can make all the difference.
   - The authors' thread ([fxtwitter link](https://fxtwitter.com/gandhikanishk/status/1896988028893323675)) highlights that **Qwen-2.5-3B** far exceeds **Llama-3.2-3B** under identical RL training for the game of Countdown.
- **ARC-AGI Without Pretraining**: Isaac Liao introduced [ARC-AGI Without Pretraining](https://fxtwitter.com/LiaoIsaac91893/status/1896944891319742499), which achieves solutions to 20% of the evaluation set puzzles through **pure inference-time gradient descent** on the target ARC-AGI puzzle itself, without pretraining or datasets.
- **Thinking Longer using Test-Time Inference**: A paper ([arxiv link](https://arxiv.org/abs/2503.01307)) suggests that test-time inference can enable language models to *think longer* and more carefully about complex challenges, akin to skilled human experts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.01307">Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs</a>: Test-time inference has emerged as a powerful paradigm for enabling language models to ``think&#39;&#39; longer and more carefully about complex challenges, much like skilled human experts. While rein...</li><li><a href="https://arxiv.org/abs/2503.01743">Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs</a>: We introduce Phi-4-Mini and Phi-4-Multimodal, compact yet highly capable language and multimodal models. Phi-4-Mini is a 3.8-billion-parameter language model trained on high-quality web and synthetic ...</li><li><a href="https://fxtwitter.com/gandhikanishk/status/1896988028893323675">Tweet from Kanishk Gandhi (@gandhikanishk)</a>: New Paper!! We try to understand why some LMs self-improve their reasoning while others hit a wall. The key? Cognitive behaviors! Read our paper on how the right cognitive behaviors can make all the d...</li><li><a href="https://fxtwitter.com/LiaoIsaac91893/status/1896944891319742499">Tweet from Isaac Liao (@LiaoIsaac91893)</a>: Introducing *ARC‑AGI Without Pretraining* – ❌ No pretraining. ❌ No datasets. Just pure inference-time gradient descent on the target ARC-AGI puzzle itself, solving 20% of the evaluation set. 🧵 1/4
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

saturatedfat: its nice ! been a minute
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1346334118866325595)** (31 messages🔥): 

> `LCPP parallel decoding, Granite 3.1 3B for tool calling, Speculative decoding with Llama 3.2 1B, Grokking in LLMs, Streaming with tool-calling` 


- **LCPP Enables Parallel Decoding and Speculative Decoding**: Members discussed manually building **LCPP** with the built-in server and using its [parallel decoding](https://github.com/ggerganov/llama.cpp) feature, which supports multi-user functionality via the `-np` flag.
   - Speculative decoding with a smaller draft model (like **Llama 3.2 1B**) to output a rough draft, corrected by a larger model (like **Hermes 3B**), was also suggested using the `-md` flag.
- **Granite 3.1 3B still excels at tool calling**: The **Granite 3.1 3B a800m instruct** model was recommended for its tool calling capabilities and speed on CPU, especially when speed is a priority.
   - It's considered a solid option for coding-related tasks.
- **Techniques for Grokking in LLMs Explored**: Members discussed **grokking** and techniques to improve generalization, attributing delayed generalization to limited precision, cross-entropy loss, and output softmax during LLM training.
   - Solutions mentioned include **Orthograd**, numerically stable softmax, increasing precision to **FP64**, replacing the output softmax with a linear squash function, and potentially Nvidia’s **N-GPT** or **Muon**.
- **Challenges with Streaming Tool-Calling Responses**: A user encountered errors using **Langchain Agents** with tool-calling in `llama.cpp` due to streaming issues, with the error message `{'error': {'code': 500, 'message': 'Cannot use tools with stream', 'type': 'server_error'}}`.
   - Since tool call execution is required before a meaningful answer can be provided, the best current solution involves faking streaming by delaying the output until after the tool call is complete.
- **Custom Frameworks Advised over Agent Frameworks**: It was suggested that custom frameworks are better than agent frameworks due to the sharp edges and hidden behaviors often encountered.
   - However, if a framework is necessary, **Pydantic** and **Smolagents** were recommended as interesting options, with **Langgraph** potentially useful in certain situations, suggesting, *"Agent frameworks have a lot of sharp edges and will always hide weird behavior from you, so I think that the framework you make yourself for your usecase is the best, because it doesn't matter what framework you use, you'll always run into some weird scenario that you won't know how to fix."*


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1346352244819496970)** (1 messages): 

> `WorldSim blog or paper, Future of WorldSim` 


- **WorldSim Continuation in Progress**: A member is preparing work to continue **WorldSim**, likely in blog form.
   - They would love to make it a paper but are unsure at the moment.
- **WorldSim Paper Delayed**: The member is exploring options to create a **WorldSim** paper.
   - A blog post might be released earlier.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1346508546623406144)** (2 messages): 

> `Agentic Memory, Zettelkasten, anon-kode, Claude Code` 


- ****Agentic Memory** inspired by Zettelkasten emerges**: A new **Agentic Memory system** based on ideas from **Zettelkasten** has been released on [GitHub](https://github.com/WujiangXu/AgenticMemory).
- ****anon-kode** facilitates coding with any LLM**: A new tool called **anon-kode** has been released on [GitHub](https://github.com/dnakov/anon-kode) which allows coding with any LLMs.
- **Claude Code gets liberated!**: It was simply noted that **Claude code** has been liberated, whatever that means.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/WujiangXu/AgenticMemory">GitHub - WujiangXu/AgenticMemory: A novel agentic memory system</a>: A novel agentic memory system. Contribute to WujiangXu/AgenticMemory development by creating an account on GitHub.</li><li><a href="https://github.com/dnakov/anon-kode">GitHub - dnakov/anon-kode: koding with any LLMs</a>: koding with any LLMs. Contribute to dnakov/anon-kode development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1346352244819496970)** (1 messages): 

> `WorldSim blog post, WorldSim paper` 


- **WorldSim Set to Return via Blog Post**: A member is preparing work to continue **WorldSim**, with the likely output being a blog post.
   - The member expressed interest in publishing it as a **paper** eventually but is unsure at the moment.
- **Paper still possible for WorldsSim**: While a blog post is planned, the author is still considering submitting the WorldsSim work as a **paper**.
   - No further details were provided as to why the author is hesitant about submitting the paper.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1346399677888725015)** (9 messages🔥): 

> `Audio transcription, NotebookLM use cases, Linear integrated circuits, Medical school, Font updates` 


- **Gemini Flash 2.0 transcribes better than YouTube AI**: A user suggested that if audio files of podcasts are available, adding them as a source to NotebookLM with **Gemini 2.0 Flash** may yield better transcriptions than **YouTube AI**.
   - The user also shared a workflow of recording lectures as audio, transcribing with NotebookLM, cleaning the transcription with **Gemini Advanced**, and importing it into Google Docs.
- **NotebookLM useful for circuit analysis**: A user inquired about using NotebookLM to learn linear integrated circuits and create a formula sheet of important concepts from their notes.
   - The user's subject involved circuit analysis with **KVL** and **KCL** numerical problems.
- **NotebookLM in Medical School**: A user inquired about how others are using **NotebookLM** in medical school.
- **Podcast created on Spotify for students**: A user created an **AI podcast** on Spotify for other students to study.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1346334096061894688)** (26 messages🔥): 

> `NotebookLM API, NotebookLM audio overview TTS, Google Doc sync, Podcast generation copyright, Store prompts in Chat-Bot` 


- ****API access is available****: Members discussed whether NotebookLM API access exists, with one member asking about it and another pointing to [Google Cloud's Speech-to-Text API](https://cloud.google.com/speech-to-text/v2/docs/chirp-model#:~:text=Chirp%20is%20the%20next%20generation,to%20more%20languages%20and%20domains.) as a way to access their **Chirp model**.
   - The Chirp model is described as the next generation speech model used to power Google products.
- ****Google Docs sync****: Members discussed how NotebookLM handles updates to Google Docs, and one member clarified that NotebookLM checks if the Google Doc has been updated and gives you the option to *'Click to sync with Google Drive'*. 
   - They expressed hope for a future one-click update feature for all sources.
- ****Generated podcast legality?****: A member inquired about the copyright laws surrounding the generated overview audio, specifically asking if they could turn it into a little podcast for their company.
   - There were no responses on the legality of generating podcasts.
- ****Prompt Persistence Problems Plague Platforms****: A user highlighted the challenge of reliably storing prompts within Chat-Bot AIs, especially on mobile where text expanders are less accurate.
   - They suggested that a voice-activated prompt system, similar to **Gemini Live**, would significantly improve user experience.
- ****Teaching Audio Overview Hosts Pronunciation****: A user asked if anyone had tried teaching the audio overview hosts to pronounce **Greek letters** by attaching a source with pronunciations.
   - They noted that the hosts often stumble when pronouncing Greek letters in their immunology notes.



**Link mentioned**: <a href="https://cloud.google.com/speech-to-text/v2/docs/chirp-model#:~:text=Chirp%20is%20the%20next%20generation,to%20more%20languages%20and%20domains.">no title found</a>: no description found

  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1346366485358186496)** (33 messages🔥): 

> `Cohere Website Design, Aya Vision Models, Cohere Sales Team, Leveling Bots` 


- **Cohere's Website Designers Deserve Kudos**: A member expressed admiration for the design of the **Cohere website**, suggesting that the designers *deserve a raise*.
- **Cohere releases Aya Vision models!**: **Cohere For AI** released the open weights research of the **Aya Vision** model in both [32-billion](https://huggingface.co/CohereForAI/aya-vision-32b) and [8-billion](https://huggingface.co/CohereForAI/aya-vision-8b) parameter versions.
   - These models are optimized for vision-language use cases, including **OCR, captioning, visual reasoning, summarization, question answering, code**, and are multilingual, excelling in **23 languages**.
- **Guidance on connecting with Cohere sales**: A member hoping to set up a meeting with someone from **Cohere** was advised to contact the sales team.
- **Leveling Bots are live**: Members noticed that level bots are live, and are being granted to users, starting with levels **1, 5, 10, 20**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/CohereForAI/aya-vision-32b">CohereForAI/aya-vision-32b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/CohereForAI/aya-vision-8b">CohereForAI/aya-vision-8b · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1346637874581340322)** (1 messages): 

> `Introductions, Community Guidelines, AI Projects, Favorite Tech Tools` 


- **New Members Introduce Themselves!**: New members are encouraged to introduce themselves using a template specifying their **Company/Industry/University**, their current work, favorite tech/tools, and their goals for the community.
   - The goal is to foster connections and provide personalized introductions.
- **Community Welcomes New Members**: The community welcomes new members joining Cohere's Discord server, expressing excitement and encouraging active participation.
   - The welcome message emphasizes the value of introductions for building connections within the community.


  

---


### **Cohere ▷ #[【🟢】status-updates](https://discord.com/channels/954421988141711382/1346652044181897307/)** (1 messages): 

competent: # SOON
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1346355437045416027)** (27 messages🔥): 

> `Automatic1111 vs Linux, Pykaso AI Website, AMD card, Regional Prompter, Stable Diffusion` 


- **Automatic1111 on WSL: Worth it?**: A member asked if running **Automatic1111** in **WSL** has any differences from running on native **Linux**.
   - A user replied that it will take a *few extra memory* to have **ComfyUI** running on **WSL** inside **Windows**, and depending on your **GPU** power might make a difference.
- **How realistic is Pykaso AI?**: A member asked how the **Pykaso AI website** provides such realistic images and how to replicate this locally on their PC.
   - No solutions were provided.
- **Is Using AMD card on Windows still difficult?**: A user inquired about the difficulty of using an **AMD card** on **Windows**, referencing year-old threads.
   - A member said that with **Zluda** it takes some time to set up but it's pretty smooth afterward and *much much faster than the yee old directml days*.
- **Regional Prompter's Resolution**: A member reported poor resolution and blurry images when using **Regional Prompter** and requested assistance.
   - No solutions were provided.
- **Need some guidance with Stable Diffusion**: A member with a mental disability seeks guidance on running **Stable Diffusion** locally on an **AMD APU (5700G)** running **Ubuntu** due to memory and information overload.
   - They are willing to discuss compensation for patient guidance in choosing the necessary functionalities.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1346336763395837963)** (26 messages🔥): 

> `MCP server claim issues, Twitter API access and costs, MCP and cursor for visual assets, Installing sequential thinking MCP server, Tool usage in roo and cursor` 


- **MCP Server Claim Hitches on Glama.ai**: A user reported issues claiming their **MCP server** on Glama.ai, encountering a `could_not_parse_params` error during Github authentication related to an *invalid returnPath*.
   - No solutions were provided in the chat.
- **Twitter API Pricing Mayhem**: Users discussed using **MCP** to connect to **Twitter** for pulling and generating tweets, but concerns arose about **Twitter's API costs**, with one user initially thinking that it was too expensive.
   - However, a member pointed out that **Twitter might have a free tier now**, prompting another user to express interest in a tool for tracking API costs across platforms like **Facebook, X, and Telegram**.
- **Cursor's Tool Use Tweaks Emerge**: Some members noted that **roo** or **cursor** may not always prioritize using available tools, even when they are relatively few in number.
   - It was suggested that **updating tool descriptions** can significantly influence the tool's usability, as detailed descriptions can make or break the tool's effectiveness.
- **Tool Call Context Learning Coming**: A member shared a link to a [GitHub pull request](https://github.com/modelcontextprotocol/specification/pull/188) related to adding **Tool Call and Tool Result** to `GetPrompt` for in-context learning of tool usage.
   - Another member noted there was *something horribly wrong with the schema.ts in that PR* and expressed a wish for an optional tool result schema for JSON results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://glama.ai/mcp/servers?searchTerm=twitter&sortingOrder=search-relevance%3Adesc">Open-Source MCP servers</a>: Enterprise-grade security, privacy, with features like agents, MCP, prompt templates, and more.</li><li><a href="https://github.com/modelcontextprotocol/specification/pull/188">Added Tool Call and Tool Result to GetPrompt for in-context learning … by evalstate · Pull Request #188 · modelcontextprotocol/specification</a>: …of tool usageAddition of ToolCall and ToolResult blocks to PromptMessage to allow in-context learning of Tool Usage patterns and error handling.Submitted as draft for review before completing/ad...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1346596848831565898)** (9 messages🔥): 

> `Torchtune Checkpointing, Step Based Checkpoint, Attention Masking vs Label Masking, Custom Special Tokens JSON` 


- **Torchtune Checkpointing Saves Space**: A user inquired about whether **Torchtune checkpointing** saves everything or if users can specify to save only the last **X checkpoints** to avoid running out of storage space.
   - A member responded that step-based checkpointing is in progress and should include an option to *"Save last n"* checkpoints.
- **Attention Masking differs from Label Masking**: A user asked about the distinction between **attention masking** and **label masking** in Torchtune, specifically whether different sets of tokens can be masked during the forward pass versus loss computation.
   - A member clarified that they are indeed different, explaining that the mask created in **sft.py** is for loss, while attention uses a causal mask by default in **SDPA** due to setting *is_causal=True*.
- **Custom Special Tokens must be copied manually**: A user reported that when adding a **custom special tokens JSON**, the final checkpoint and epochs folder receives a non-custom special tokens JSON file.
   - They asked if this behavior is expected and whether they should manually copy the special tokens, as the checkpointer code does not seem to specifically save custom special_tokens files in checkpoints per epoch.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1346475216091611167)** (14 messages🔥): 

> `MPS/Metal support for AO, Manual tests for MPS, Saving completed tunes` 


- **QLoRA Recipes get Metal Boost**: Members discussed whether updating **Environment.device** in configs would cause **QLoRA recipes** to target the **Metal kernels**, now that **AO** has **MPS/Metal support**.
   - It's unclear, but [manual tests](https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels) are needed.
- **MPS Tests incoming for 1B Instruct**: Members are planning manual tests for **MPS**, focusing on **1B-instruct models** and various **bit types** for generation, following the patterns in **torchchat**.
   - One member volunteered for testing with **1b** and **tiny stories** to provide an easy example for beginners, while another with *decent mps* offered to help, but suggested waiting for maintainer approval first.
- **Checkpoints Take Twelve Minutes?**: One user reported waiting **12 minutes** for a **3B model** to save, without changing checkpointer settings.
   - They also proposed that *a progress bar for the save would be great, for impatient people* and another member agreed to implement it in each save_checkpoint.



**Link mentioned**: <a href="https://github.com/pytorch/torchchat/blob/main/docs/quantization.md#experimental-torchao-lowbit-kernels">torchchat/docs/quantization.md at main · pytorch/torchchat</a>: Run PyTorch LLMs locally on servers, desktop and mobile - pytorch/torchchat

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1346526832333164605)** (2 messages): 

> `LlamaCloud GA, HuggingFace course, LLM-powered agents, LlamaIndex toolkit` 


- **LlamaCloud Goes GA!**: The team announced that **LlamaCloud** is now Generally Available, providing a turn-key solution for agentic knowledge management over unstructured data, accessible via [t.co link](https://t.co/1CSRJm30e3).
- **Hugging Face Builds LlamaIndex Course!**: The LlamaIndex team announced that **Hugging Face** has created an educational course on building agents with LlamaIndex, covering components, RAG, tools, agents, and workflows, found at [t.co link](https://t.co/eACAJzXg8y).
- **LlamaIndex Toolkit**: **LlamaIndex** is a toolkit for creating **LLM-powered agents** over your data using indexes and workflows.



**Link mentioned**: <a href="https://t.co/eACAJzXg8y">Introduction to LlamaIndex - Hugging Face Agents Course</a>: no description found

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1346457929926049873)** (16 messages🔥): 

> `DeepSeek API issue, LLM.txt documentation files, Unintentional long output` 


- ****DeepSeek API** Key Shows Insufficient Balance**: A member reported an `openai.APIStatusError` with a **402** error code, indicating *'Insufficient Balance'* when using the **DeepSeek API** with LlamaIndex.
   - Another member suggested this issue typically arises from a lack of credits or a missing payment method in the user's account, unrelated to LlamaIndex itself.
- **Request for **LLM.txt** Files for Documentation Pages**: A member inquired about obtaining **LLM.txt**-type files for documentation pages to use as context within LlamaIndex.
   - The team responded that such a feature isn't currently available but suggested converting the existing markdown content using a converter, as the content is *technically already in markdown*, and linked to the [LlamaIndex documentation directory](https://github.com/run-llama/llama_index/tree/main/docs/docs).
- **Unintentional Long Output**: A member highlighted an issue with excessively long output on a documentation page, specifically [the Postgres vector store example](https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/).
   - The team acknowledged the problem, attributing it to *a lot of text* and fixed it with [PR #18002](https://github.com/run-llama/llama_index/pull/18002).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/tree/main/docs/docs">llama_index/docs/docs at main · run-llama/llama_index</a>: LlamaIndex is the leading framework for building LLM-powered agents over your data. - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/pull/18002">missed some openai changes, and clean postgres logs by logan-markewich · Pull Request #18002 · run-llama/llama_index</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/">Postgres Vector Store - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1346342017273172028)** (2 messages): 

> `Windsurf Checkpoints, LlamaIndex vs Landing AI` 


- **Windsurf Lacks Checkpoint Feature**: A member inquired about checkpoint functionality in Windsurf, noting that dragging and dropping files and workspaces into the tab menu doesn't provide options to revert to previous checkpoints.
   - Despite writing code, the user finds *no means* to go back to a previous checkpoint, a feature that appears to be missing.
- **LlamaIndex Faces off Landing AI**: A member asked for comparisons between document extraction using **LlamaIndex** versus **Agentic Document Extraction** by Landing AI.
   - They are looking for insights from anyone who has experience with both solutions.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1346636776009371670)** (2 messages): 

> `Indian Reasoning, Foundational Models, Law, ChatGPT training limitations, Fine-tuning` 


- **Debate Indian Law Reasoning with Fine-Tuning**: A member is studying the need for **Indian reasoning-foundational models** for **Law** and asks if fine-tuning **ChatGPT** with Indian cases would sufficiently solve the issue of it being trained on US cases.
   - The user seeks intuition on whether this approach is valid and to what extent it would address the problem.
- **ChatGPT's Legal Reasoning Training**: The user highlights that **ChatGPT** was trained on US cases, leading to a reasoning bias towards US legal principles.
   - They question whether fine-tuning with Indian cases would adequately address this bias for practical applications in Indian law.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1346463530756341800)** (1 messages): 

> `Modded-nanogpt Adam-Matching Scaling, Kimi paper scaling multipliers, QKVO Matrices` 


- **Adam-Matching Scaling Origins**: Earlier versions of the *modded-nanogpt speedrun* used the same adam-matching scaling as the [kimi paper](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt).
   - However, it used a scaling multiplier of **0.1** instead of the kimi paper's **0.2**.
- **QKVO Matrices revisited**: Later runs of the *modded-nanogpt speedrun* used `max(1, g.size(0)/g.size(1))^0.5` instead of `max(g.size(0), g.size(1))^0.5`.
   - The change differs from the adam-matching only by a constant multiplier in all the **qkvo matrices**, but makes the update size of **c_fc matrices** always twice as large as **c_proj matrices**.



**Link mentioned**: <a href="https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101024_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt">modded-nanogpt/records/101024_Muon/eb5659d0-fb6a-49e5-a311-f1f89412f726.txt at master · KellerJordan/modded-nanogpt</a>: NanoGPT (124M) in 3 minutes. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1346341391294529559)** (11 messages🔥): 

> `lm-evaluation-harness, dataset_kwargs, datasets.load_dataset, Huggingface load_datasets, Llama 3 eval recipe` 


- **Dataset Kwargs Discussion**: Members discussed whether `--trust_remote_code` is related to `dataset_kwargs` in the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/__main__.py#L367).
   - It was clarified that `--trust_remote_code` is only set if passed as a parameter, and the `dataset_kwargs` are passed to `datasets.load_dataset(...)` [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/api/task.py#L930).
- **Debugging Dataset Loading Error**: A member reported a `datasetGenerationError` when running `lm_eval` with specific task configurations, including `dataset_kwargs` to specify data directories.
   - The member clarified later that the issue was due to **additional dataset_kwargs** overriding the ones in the subtasks, and the dataset loads fine from the **Hugging Face load_datasets** library.
- **Quest for Reproducible Llama 3 Results**: A member inquired whether a different evaluation recipe is needed to reproduce the results from the **Llama 3 paper**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/__main__.py#L367">lm-evaluation-harness/lm_eval/__main__.py at 14b0bd26956609b2ee50987299dfa34223fa23b8 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>: Join the world&#39;s most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/14b0bd26956609b2ee50987299dfa34223fa23b8/lm_eval/api/task.py#L930)">lm-evaluation-harness/lm_eval/api/task.py at 14b0bd26956609b2ee50987299dfa34223fa23b8 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1346513491233865811)** (12 messages🔥): 

> `ReAct Agents, OctoTools, Agentic Reward Modeling, MinionsLM, dspygen` 


- ****React Agents** as Orchestrators Debated**: The need for **ReAct agents** as orchestrators was questioned, suggesting classification might suffice, but concerns arose regarding handling multiple intents and complex multi-step tasks.
   - One member is conceptualizing an approach to orchestrate conversations including tools and a knowledge base.
- ****OctoTools** Framework for Tool Orchestration Emerges**: [OctoTools](https://octotools.github.io/) from Stanford was shared, highlighting its use of **tool cards**, a **planner**, and an **executor** to manage tool interactions and generate final answers, including task-specific toolset optimization.
   - The tool cards define tool-usage metadata and encapsulate heterogeneous tools, enabling training-free integration of new tools.
- ****Agentic Reward Modeling** for Reliable Reward Systems**: A member shared [Agentic Reward Modeling](https://github.com/THU-KEG/Agentic-Reward-Modeling), focusing on integrating human preferences with verifiable correctness signals for reliable reward systems.
   - The member had also implemented a cost optimization feature with their implementation of minionS, but the [PR was rejected](https://github.com/stanfordnlp/dspy/pull/7891) to the [DSPy framework](https://github.com/jmanhype/dspy).
- ****dspygen** and **Spark DSLs** Offer Tooling Inspiration**: A member suggested using [dspygen](https://github.com/seanchatmangpt/dspygen/blob/main/src/dspygen/mixin/hsm/hsm_mixin.py) and [Spark](https://hexdocs.pm/spark/get-started-with-spark.html) for inspiration.
   - Another user considered creating a **DSL** or using a similar interface in **Axon**, drawing inspiration from **PyTorch**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://octotools.github.io/"> OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning</a>: no description found</li><li><a href="https://github.com/seanchatmangpt/dspygen/blob/main/src/dspygen/mixin/hsm/hsm_mixin.py">dspygen/src/dspygen/mixin/hsm/hsm_mixin.py at main · seanchatmangpt/dspygen</a>: A Ruby on Rails style framework for the DSPy (Demonstrate, Search, Predict) project for Language Models like GPT, BERT, and LLama. - seanchatmangpt/dspygen</li><li><a href="https://github.com/THU-KEG/Agentic-Reward-Modeling">GitHub - THU-KEG/Agentic-Reward-Modeling: Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems</a>: Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems - THU-KEG/Agentic-Reward-Modeling</li><li><a href="https://hexdocs.pm/spark/get-started-with-spark.html">Spark — spark v2.2.45</a>: no description found</li><li><a href="https://github.com/stanfordnlp/dspy/pull/7891">Add MinionsLM and StructuredMinionsLM for intelligent LM routing by jmanhype · Pull Request #7891 · stanfordnlp/dspy</a>: MinionsLM and StructuredMinionsLM ImplementationThis PR introduces two new classes to the DSPy framework:MinionsLM: An intelligent LM router that implements the MinionS protocol for cost-efficie...</li><li><a href="https://github.com/jmanhype/dspy">GitHub - jmanhype/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy: The framework for programming—not prompting—foundation models - jmanhype/dspy</li><li><a href="https://github.com/HazyResearch/minions">GitHub - HazyResearch/minions: Big &amp; Small LLMs working together</a>: Big &amp; Small LLMs working together. Contribute to HazyResearch/minions development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1346395322259341312)** (6 messages): 

> `GPT4All vs Ollama, RAG support in GPT4All, Model comparison: tiny model vs Llama3-8b with LocalDocs` 


- **Members Report GPT4All Lags Ollama**: Members are hoping that [GPT4All](https://gpt4all.io/) can catch up with [Ollama](https://ollama.com/), wanting to see GPT4All on top.
- **Tiny Models Excel with RAG**: A member clarified that a certain **tiny model** is better when used with **RAG** due to its speed, but it might **confabulate** a lot if used without **RAG**.
   - They mentioned that the capabilities of models are limited by their number of parameters, architecture, and training data, advising that **Llama3-8b** is very good in combination with **LocalDocs**.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1346626940467150901)** (2 messages): 

> `Server maintenance, Sponsors zone activity` 


- **Server Maintenance Questioned**: A member inquired if the server is still maintained, noting activity in the sponsors zone.
   - Another member confirmed that *yes of course* it is.
- **Sponsor Zone Still Active**: Members noticed active posting in the sponsor zone.
   - This activity led to questions about whether the server is being actively maintained.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1346422707222482944)** (1 messages): 

> `AI Stock Market Agent, AI in Finance, AI Investment Buddy, AI Real-World Success` 


- **AI Stock Market Agent Workshop Announced**: A workshop on building an **AI Stock Market Agent** is scheduled for **Friday, March 7th at 9 PM IST**, teaching participants how AI can analyze over 1000 stocks quickly, with registration available [here](https://lu.ma/0ckb8tp0).
   - The workshop aims to show how **AI** is changing the investment landscape and provide tools for smarter investment decisions.
- **AI & Finance create Perfect Match**: The workshop intends to reveal how **AI** is revolutionizing investing, with real examples of **AI** predicting market trends.
   - Participants will uncover how **AI** is changing the investment landscape and provide tools for smarter investment decisions.
- **Build AI Investment Buddy, No Code Required**: The workshop will guide attendees in building an **AI** tool to analyze stocks without coding, enabling testing of investment ideas without real money risk.
   - It emphasizes a beginner-friendly approach to leveraging **AI** in investment strategies.
- **AI in Action: Real-World Success Stories**: The workshop will explore how big investors use **AI** for smarter choices and how **AI** aids in informed investment decisions.
   - The session includes an exploration of real-world success stories and practical applications of **AI** in finance.



**Link mentioned**: <a href="https://lu.ma/0ckb8tp0?tk=sCR4qA">Build your AI Stock Market Agent · Zoom · Luma</a>: Ever wondered how AI could help you make smarter investment decisions?Join us for an exciting, beginner-friendly workshop that combines the world of…

  

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

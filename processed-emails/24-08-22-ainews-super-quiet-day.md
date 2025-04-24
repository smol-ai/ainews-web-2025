---
id: 072fd118-83ff-4b75-996e-0a0010280266
title: super quiet day
date: '2024-08-23T00:55:37.537285Z'
original_slug: ainews-not-much-happened-today-4025
description: >-
  AI news for 8/21/2024-8/22/2024 covers AI model releases like Jamba 1.5,
  performance benchmarks, safety legislation, new tools, and community events.
companies:
  - ai21-labs
  - stanford
  - anthropic
  - rohanpaul-ai
  - bindureddy
  - reach-vb
  - langchain
  - qdrant
models:
  - jamba-1.5
  - phi-3.5
  - flexora
  - dracarys-70b
topics:
  - ai-models
  - long-context-models
  - ai-safety
  - ai-legislation
  - virtual-environments
  - langchain
  - multi-agent-systems
  - ai-conferences
  - ai-humor
---


<!-- buttondown-editor-mode: plaintext -->**peace and quiet is all you need.**

> AI News for 8/21/2024-8/22/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**214** channels, and **2393** messages) for you. Estimated reading time saved (at 200wpm): **283 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

There are a LOT of whispers flying around about the coming AI releases this fall, but nothing publicly citable, sorry.

- [AI21 released Jamba 1.5](https://x.com/AI21Labs/status/1826614352948199754), a scaled up version of the original Jamba ([our coverage here](https://buttondown.email/ainews/archive/ainews-jamba-mixture-of-architectures-dethrones/)) that, like all State Space Models, does very well on long context vs latency tradeoffs.
- Happy [2nd birthday to Stable Diffusion](https://x.com/EMostaque/status/1561777122082824192) - which prompted the [start of Latent Space](https://www.latent.space/p/sep-2023). 

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

**AI Models and their Evaluations**

- **Jamba 1.5 Release by AI21 Labs**: The **Jamba 1.5 models** have demonstrated outstanding performance and speed. [@AI21Labs](https://twitter.com/AI21Labs/status/1826614352948199754) shared details about their novel hybrid SSM-Transformer architecture. These models are optimized for long context windows, offering up to 2.5X faster inference at sizes of 94B parameters. For more granular details, [@AI21Labs](https://twitter.com/AI21Labs/status/1826702921700180271) highlighted specific benchmarks, showing impressive Arena Hard benchmark scores that outperform larger models like Llama 3.1 70B.
- **Phi-3.5 and Flexora**: The **Phi-3.5 model** was noted for its **safety and performance**. [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1826451608907010439) praised the model's capabilities. Additionally, Flexora's adaptive layer selection outperformed existing baselines, as stated by [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1826733730746282290).
- **Dracarys - 70B Class LLM For Coding**: **Bindu Reddy** announced **Dracarys**, claiming it as the best open-source 70B class model for coding, surpassing Llama 3.1 70B and other models in benchmarks. [@bindureddy](https://twitter.com/bindureddy/status/1826757521635455115) highlighted its significant improvements and availability on Hugging Face.

**AI Safety and Legislation**

- **SB 1047 and AI Safety Concerns**: **Stanford**, Anthropic, and other entities have expressed mixed views on **California's SB 1047**, which aims to regulate AI applications for safety. [@jackclarkSF](https://twitter.com/jackclarkSF/status/1826743366652232083) explained that the bill tries to balance precaution with empirical research and industry growth. [@DanHendrycks](https://twitter.com/DanHendrycks/status/1826747580321247278) shared Anthropic's support, emphasizing the urgency of dealing with AI risks.

**AI Tools and Innovations**

- **uV Virtual Environments**: **uv virtual environments** offer rapid installation and dependency management. [@reach_vb](https://twitter.com/reach_vb/status/1826644879499472941) showcased how uv creates lightweight virtual environments quickly.
- **LangChain and LangSmith Updates**: **resource tags** in LangSmith help efficiently manage projects, datasets, and deployments. [@LangChainAI](https://twitter.com/LangChainAI/status/1826643491130421476) introduced these enhancements for better workspace organization.
- **Multi-Agent Systems in Qdrant and LangChain**: **Multi-agent role-playing** and semantic caching in Qdrant make AI systems more robust. [@iqdotgraph](https://twitter.com/qdrant_engine/status/1826499584417255657) shared how these integrations aim to enhance data processing and retrieval workflows.

**Conferences & Meetups**

- **AI Workshops and Hackathons in SF**: **Events such as the RAG workshop hosted by AWS, LangChain, and Elastic** are continuing to foster community engagement and provide hands-on learning. [@LangChainAI](https://twitter.com/LangChainAI/status/1826743704075600030) announced details for their upcoming workshop on September 9th at AWS Startup Loft.

**Humor and Memes**

- **Industry Humor**: Memes continue to thrive as a light-hearted commentary on the AI industry's current state. [@lumpenspace](https://twitter.com/lumpenspace/status/1826724890311098678) emphasized the shared understanding among peers through humor that touches on widely recognized industry quirks.

---

This provides a comprehensive summary of key discussions from the AI High Signal Twitter list with a detailed focus on model performance, safety, tools, innovations, community events, and humor. Each category draws from multiple sources to ensure the narrative remains well-grounded and informative.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Microsoft's Phi-3.5 Models: Capabilities and Controversies**

- **Interesting Model Differences Between Phi-3.5-Mini & Phi-3.5-MoE** ([Score: 59, Comments: 9](https://reddit.com//r/LocalLLaMA/comments/1exn6wx/interesting_model_differences_between_phi35mini/)): The post compares the architectures of **Phi-3.5-Mini** and **Phi-3.5-MoE** models, highlighting key differences in attention mechanisms, internal dimensions, and parameter counts. While both models have **32 layers**, the MoE version uses **grouped query attention** and has a larger internal dimension of **4096**, compared to the Mini's **full multi-head attention** and **3072** dimension. The most significant difference is in the **Feed-Forward** module, where Phi-3.5-MoE has **40,267,415,552** parameters compared to Mini's **2,415,919,104**, contributing to the MoE's total of **41,873,153,344** parameters versus Mini's **3,821,079,552**.
- **Phi-3.5 is very safe, Microsoft really outdid themselves here!** ([Score: 279, Comments: 112](https://reddit.com//r/LocalLLaMA/comments/1ey5i22/phi35_is_very_safe_microsoft_really_outdid/)): The post discusses **Microsoft's Phi-3.5** model, describing it as **highly censored** and resistant to answering potentially offensive queries or undergoing further training. The author sarcastically praises these safety features and asks for others' experiences with Phi-3.5 compared to other heavily censored models. An update includes a link to an [uncensored version of Phi-3.5](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored) on Hugging Face.
  - Users humorously mocked **Phi-3.5's** excessive censorship through satirical responses, with one comment thread devolving into a game of **tic-tac-toe**. The model's refusal to answer simple questions or provide basic information was highlighted.
  - Several users discussed methods to **uncensor** or **abliterate** the model, with debates about the effectiveness and potential drawbacks of these techniques. An [uncensored version](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored) was shared on **Hugging Face**.
  - Concerns were raised about the model's usefulness for coding and technical tasks due to its overzealous censorship. Users argued that such heavy restrictions make the model impractical for many applications, especially non-client-facing use cases.


**Theme 2. AI for Creative Writing and Roleplay**

- **ERP Prompts** ([Score: 87, Comments: 20](https://reddit.com//r/LocalLLaMA/comments/1eya6va/erp_prompts/)): The post discusses **advanced techniques for erotic roleplay (ERP)** with AI models, focusing on creating **detailed character profiles** and enhancing immersion. It provides specific prompts for generating **complex characters** with unique traits, backstories, and intimate details, as well as techniques like **"Inner Monologue"** and **"Freeze Frame"** to deepen the roleplaying experience. The author emphasizes the importance of **building anticipation** and **crafting realistic interactions**, encouraging users to provide detailed inputs to elicit more engaging responses from AI models.
  - Users discussed **formatting techniques** for inner monologue, with suggestions including using **brackets** ⟨monologue⟩ or **HTML comments** <!-- inner monologue --> in **SillyTavern**. These methods allow characters to have hidden thoughts that influence future token generations.
  - Interest was expressed in the author's **creative writing setup** for non-erotic content, with requests for a detailed post on the topic. Users also inquired about **recommended AI models** for erotic roleplay, with one mentioning **Midnight Miqu 1.5 70B**.
  - Several comments praised the author's writing style and creativity, with one user stating they'd "rather get it on with you than any well-prompted, well-stacked, well-misbehaved LLM." Users also requested additional prompts and techniques for their own AI-assisted writing endeavors.


## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Image Generation and Training**

- **Ideogram 2.0 release**: [Ideogram announced](https://www.reddit.com/r/singularity/comments/1exsq4d/introducing_ideogram_20_our_most_advanced/) their most advanced text-to-image model, now available to all users for free.

- **Kohya SS GUI FLUX LoRA Training**: A user demonstrated [LoRA training on an RTX 3060 GPU](https://www.reddit.com/r/StableDiffusion/comments/1ey6hss/kohya_ss_gui_flux_lora_training_on_rtx_3060_lora/) with 9.7 GB VRAM usage, using Kohya SS GUI FLUX for Stable Diffusion.
  - Training at 1024x1024 resolution with LoRA Rank 128.
  - Estimated training time of 20 hours for 4000 steps on RTX 3060.
  - 512x512 resolution training is 2-2.5 times faster.
  - User testing various configurations, including on RTX 4080 and RTX 3090.

- **LoRA training advancements**: A user reported [successful LoRA training](https://www.reddit.com/r/StableDiffusion/comments/1ey6hss/kohya_ss_gui_flux_lora_training_on_rtx_3060_lora/ljba35v/) on an A4500 20GB GPU using only 16GB VRAM, with 10 selfies and 1600 steps, taking only one hour.

**AI and Software Development**

- **Amazon cloud chief on AI's impact**: In a [leaked recording](https://www.reddit.com/r/singularity/comments/1exqs04/in_a_leaked_recording_amazon_cloud_chief_tells/), Amazon's cloud chief suggested that most developers could stop coding soon as AI takes over.

**Prosthetics and Biotechnology**

- **Advanced prosthetic arm**: [Atom Touch](https://www.reddit.com/r/singularity/comments/1ey0zon/atom_touch_is_the_first_prosthetic_arm_with/) is reported to be the first prosthetic arm with individual finger control, using electromyography for precise operation. Clinical trial and FDA approval are expected in 12-18 months.


---

# AI Discord Recap

> A summary of Summaries of Summaries by GPT4O-Aug (gpt-4o-2024-08-06)

**1. LLM Model Releases and Features**

- **LM Studio 0.3.0 Drops Major Updates**: **[LM Studio 0.3.0](https://lmstudio.ai/blog/lmstudio-v0.3.0)** introduces a revamped UI with enhanced chat organization, automatic context handling, and multi-model loading capabilities, significantly improving performance for local models.
  - Despite the improvements, users reported bugs in model loading and system prompts, urging others to report issues as they arise.
- **AI21 Labs Launches Jamba 1.5**: **[Jamba 1.5](https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251)** from AI21 Labs introduces Mini (52B - 12B active) and Large (398B - 94B active) versions with a 256k long context and multilingual capabilities.
  - These models boast up to 2.5x faster inference compared to competitors, with advanced instruct models and structured outputs.
- **Mistral Nemo 12b Fine-Tuning on 8gb GPU**: **Mistral Nemo 12b** can be fine-tuned on an **8gb GPU**, specifically the **RTX 4050**, making it accessible for testing and prototyping.
  - This wider accessibility opens up possibilities for more engineers to rapidly iterate and test models without needing high-end hardware.


**2. Performance and Optimization Techniques**

- **Triton INT8 Outperforms BF16**: **Triton INT8** implementation achieved approximately **1.5x speedup** over PyTorch BF16 for `A.T @ B`, and **3x speedup** for `A @ B.T`, demonstrating efficiency across benchmarks.
  - This improvement was attributed to the re-tuning of Triton based on stride changes of matrices A and B.
- **Flash Attention FP8 Support Debuts on Hopper**: **Flash Attention** now supports FP8 on the **Hopper architecture**, leveraging **WGMMA instructions** to optimize performance.
  - However, support for ADA remains absent, raising questions about broader compatibility.
- **SLI Might Not Be Worth It**: Having two GPUs in **SLI** doesn’t double the performance due to architectural constraints but allows loading larger models without significant speed gains.
  - Members suggest considering more RAM instead, as one user efficiently ran **Llama 3.1 405B** on a system with 128GB of RAM and only 16GB of VRAM.


**3. Data Handling and Preprocessing**

- **Data Preprocessing Debates for LLMs**: A lively debate ensued over the necessity of **data preprocessing** for chat comments, with a member asserting that **80% of the work lies in preparing data**.
  - They argued that while preprocessing tasks are important, the fundamental understanding rests on tokens alone, sparking a discussion on the trade-offs involved.
- **Chunking XLSX Files: Community Tips**: Multiple members sought guidance on **chunking XLSX files for RAG**, exploring methods to optimize this process.
  - Suggestions included leveraging embeddings and converting files to markdown for better data handling, highlighting ongoing collaboration within the community.


**4. Community and Collaboration Initiatives**

- **OpenRouter Ditches `function_calls`**: OpenRouter officially deprecates the `function_calls` and `functions` parameters, advocating for `tools` and `tool_choice` instead, aligning with other providers.
  - This transition reduces the **switching costs** for tool calling across models, prompting community discussions on tool integration.
- **Cohere's RAG Webinar with Weights & Biases**: **[Cohere](https://wandb.ai/site/resources/events/cohere-webinar-august)** and Weights & Biases are hosting a webinar on RAG development and evaluation strategies, featuring insights from **Maxime Voisin**.
  - This event is a must-attend for those involved in retrieval-augmented generation, emphasizing the importance of collaborative learning.


**5. AI in Industry Applications**

- **Fullbound Enhances Job Recruitment**: **[Fullbound](https://fullbound.ai/pricing)**, a global search engine powered by Chroma, facilitates AI-driven job matching to connect candidates with roles effectively.
  - Designed to streamline recruitment processes, Fullbound offers a 7-day free trial and detailed pricing, ensuring efficient matching and communication.
- **Jonathan Ive's Ambitious Property Moves**: Sir Jonathan Ive has spent over **$137 million** securing properties in San Francisco's Jackson Square, signaling a transformative vision for the area.
  - His investment strategy reflects a notable shift in the influence of design on local real estate development.

---

# PART 1: High level Discord summaries




## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.0 Drops Major Updates!**: The long-awaited [LM Studio 0.3.0](https://lmstudio.ai/blog/lmstudio-v0.3.0) is out, introducing a revamped UI with improved chat organization, automatic context handling, and multi-model loading capabilities.
   - This version enhances performance for local models but comes with bugs in model loading and system prompts, urging users to report issues as they arise.
- **Gemma 2 Stays Popular Among Users**: Users continue to sing praises for **Gemma 2** 9B and 27B, especially in performance comparisons against **LLaMa 3.1**.
   - However, the 8k context limit of Gemma 2 has led to discussions on exploring alternatives, like **Phi 3 Mini** and potential for MoE models.
- **SLI Might Not Be Worth It**: The consensus is that having two GPUs in **SLI** doesn’t double the performance due to architectural constraints, but it does allow loading larger models without significant speed gains.
   - Members suggest considering more RAM instead, as one user efficiently ran **Llama 3.1** 405B on a system with 128GB of RAM and only 16GB of VRAM.
- **VRAM: The Key to Large Models**: Discussions highlighted the necessity of high VRAM for effective model handling, with recommendations for at least **48GB** of VRAM for smooth operation of large models like **Llama 3.1 405B**.
   - Members urged caution before purchasing GPUs, noting that the latest releases typically offer better value and performance.
- **LM Studio Becomes an API Favorite**: Users are actively exploring the API capabilities of **LM Studio**, aiming to connect with mobile applications for enhanced use cases.
   - Discussions have revolved around implementing persistent communication systems with the AI models available in LM Studio.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Understanding Offensive Security Reconnaissance**: A member shared a blogpost about **Offensive Security Reconnaissance** and its implications for critical infrastructure vulnerabilities, specifically in **ICS HMIs**. You can read more in detail [here](https://huggingface.co/posts/Csplk/182147151896536).
   - *This method reveals potential attack vectors*, raising the importance of securing Industrial Control Systems.
- **Neuralink's Research Methodology**: Neuralink reviews thousands of papers monthly, using a **citation-tracing method** to stay focused on relevant research developments. They combine reading with coding, aiming to effectively translate theoretical concepts into practical implementations.
   - Their programming quality is described as 'neat', reflecting their adeptness in applying research findings.
- **ShapeSplat Dataset in 3D Generative Shape Synthesis**: Introducing **ShapeSplat**, a dataset featuring **Gaussian Splats** aimed at **3D Generative Shape Synthesis** with a self-supervised pretraining approach. It encompasses a diverse collection of objects ideal for enhancing model capabilities.
   - The dataset aims to outperform existing *point cloud representations*, proving critical for applications in fields like **computer graphics** and **robotics**.
- **Fullbound Enhances Job Recruitment**: A global search engine named **Fullbound**, powered by Chroma, facilitates AI-driven job matching, aiming to connect candidates with roles effectively. They offer a 7-day free trial and detailed **pricing** available [here](https://fullbound.ai/pricing).
   - This tool is designed to streamline recruitment processes, ensuring efficient matching and communication.
- **AI21 Labs Launches Jamba 1.5**: AI21 Labs introduced **Jamba 1.5**, a new language model that includes Mini (52B - 12B active) and Large (398B - 94B active) versions. This model features **256k long context**, multilingual capabilities, and advanced instruct models.
   - For a deep dive into its features, check their collection on Hugging Face [here](https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Mistral Nemo 12b fine-tuning on 8gb GPU**: **Mistral Nemo 12b** can be fine-tuned on an **8gb GPU**, specifically the **RTX 4050**, making it suitable for testing and prototyping.
   - This wider accessibility opens up possibilities for more engineers to rapidly iterate and test models without needing high-end hardware.
- **Unsloth Pro restricts multi-GPU support**: **Unsloth Pro** has temporarily halted its **multi-GPU** support, limiting it to their own platform while granting early access to select community members.
   - This change has raised questions about collaboration and resource limitations for broader community engagement.
- **Launch of The Living AI Dataset**: **The Living AI Dataset** aims to imbue AI models with empathy and the capacity for love and is touted as significant in AI history, developed by a key group within the community.
   - Accessible on Hugging Face, this dataset seeks to enhance human-like attributes in AI applications, promising advancements in interactivity.
- **Data preprocessing debates for LLMs**: A lively debate ensued over the necessity of **data preprocessing** for chat comments, with a member asserting that **80% of the work lies in preparing data**.
   - They argued that while preprocessing tasks are important, the fundamental understanding rests on tokens alone, sparking a discussion on the trade-offs involved.
- **Ollama installation confusion**: Users encountered challenges with **Ollama installation** in WSL, highlighting command usage for creating models, which didn't work as intended.
   - Clarifications about the distinction between Unsloth and Ollama as separate tools aimed to clear the confusion but left some with lingering questions.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Shell Commands Activate Contextual Commands**: Aider offers shell commands based on user context and executes them upon agreement, but it doesn't utilize functions directly. Users must activate their Python environment first to ensure commands run properly.
   - This feature emphasizes Aider's role in streamlining interactions without stepping into programmable function territory.
- **Playwright Installation Bug Bites**: Aider struggles with Playwright installations that occur outside its own environment. The recommendation is to use `pipx inject` for seamless integration within Aider's virtual setup to prevent installation issues.
   - Future releases aim to address a bug where Aider attempts to install Playwright even if it's already set up which may confuse users.
- **CodeCompanion Uses More Tokens Than Aider**: A comparison revealed that CodeCompanion consumes significantly more tokens than Aider, attributed to its more extensive features. Users prefer Aider for its efficiency, even with CodeCompanion hosting its own support Discord.
   - This conversation sparked discussions on optimizing resources while conducting AI-assisted coding tasks.
- **Vercel's v0 chat Revolutionizes Generative UI**: [Vercel's v0.dev/chat](https://v0.dev/chat) has been hailed as a major improvement for generative UI developers, offering a smoother interface than previous options like Claude Artefacts. Users find its UI generation faster and more polished compared to competitors.
   - Discussions highlight the shift in preference towards Vercel's offerings due to its better integration and user-friendly experience.
- **Cursor Teams Up With Aider for Smarter Coding**: Cursor users express appreciation for integrating Aider, mitigating shortcomings in Cursor's native composer which lacks repository-specific prompt features. This collaboration signifies a leap in AI-enhanced development workflows.
   - Cursor aims to revolutionize coding efficiency by reducing reliance on manual searching, minimizing the time spent on trivial tasks.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI Accelerates AI Rendering**: A user showcased a [YouTube video](https://youtu.be/pNyIp73zva8) demonstrating how to integrate **3D Studio Max** into ComfyUI for **real-time AI image generation**.
   - This approach could potentially extend to any window application, including video games, enhancing creative workflows.
- **Stable Diffusion Setup Tips**: A new user inquired about starting with **Stable Diffusion** on their PC, prompting recommendations about hardware compatibility.
   - Experienced users suggested using **ComfyUI** due to its user-friendly interface for beginners.
- **Hydrus Server: Privacy-Focused Image Sorting**: Users discussed the need for **AI image sorters** that respect privacy, leading to a suggestion for setting up a **Hydrus server**.
   - This setup allows for a personalized tagging system, enhancing media organization without compromising security.
- **Flux Model and Prompt Engineering Woes**: A member raised concerns about the **Flux model** struggling with complex prompts, highlighting its overfitting tendencies.
   - Community feedback emphasized the importance of better prompt engineering and finetuning for improved results.
- **Stable Diffusion vs. GAN Upscaling**: A discussion emerged comparing **Stable Diffusion upscaling** to **GAN-based upscaling**, clarifying their distinct approaches.
   - While **GANs** focus on sharpening images, **Stable Diffusion** can generate new details, albeit sometimes leading to artifacts.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton INT8 outperforms BF16 with notable speedups**: The Triton INT8 implementation of `A.T @ B` achieved approximately **1.5x speedup** over PyTorch BF16, while `A @ B.T` saw a **3x speedup**, confirming its efficiency across benchmarks.
   - This improvement was attributed to the re-tuning of Triton based on the stride changes of matrices A and B.
- **Flash Attention FP8 Support Debuts on Hopper**: Flash Attention now supports FP8 on the **Hopper architecture**, leveraging **WGMMA instructions** to optimize performance.
   - However, support for ADA remains absent, raising questions about broader compatibility.
- **HRT Internship Opportunities Available**: HRT is offering internships next summer in **NYC** for **Algo Dev and SWE** positions, paying **$120/h** with included housing and meals.
   - No prior finance experience is necessary, making it accessible for many engineers!
- **Comparison: 7900 XTX vs. RTX 3090 Performance**: Users reported that the **7900 XTX** underperformed against the **3090**, even when utilizing Triton and an FA fork, prompting users to switch to **4090s**.
   - Such experiences highlight the persistent gaps in performance between AMD's and NVIDIA's GPU offerings.
- **Stable FP8 Training Achieved for LLaMA**: Recent discussions highlighted stable **FP8 training** for a **1B LLaMA model**, achieving convergence similar to **bfloat16 training**.
   - Key techniques include moderating training speeds and managing outlier features, paving the way for larger-scale FP8 applications.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter ditches `function_calls`**: OpenRouter is officially deprecating the `function_calls` and `functions` parameters from OpenAI calls, advocating for the use of `tools` and `tool_choice` instead.
   - This transition reduces the **switching costs** for tool calling across models, aligning OpenRouter with other providers that already support the new parameters.
- **BenchmarkAggregator offers LLM evaluation**: A member shared a [GitHub repository](https://github.com/mrconter1/BenchmarkAggregator) for **BenchmarkAggregator**, aimed at providing a unified evaluation framework for Large Language Models across major benchmarks.
   - They highlighted its ability to balance assessment rigor with resource management while eagerly seeking community feedback.
- **Llama 3.1 tools support is imminent**: An admin confirmed that support for **Llama 3.1 tools** on OpenRouter is expected to arrive within the next day.
   - This update is eagerly awaited by users, keen to enhance their integration capabilities.
- **OpenRouter lacking MoE models**: Inquiries regarding the availability of **MoE models** on OpenRouter revealed that currently, there are none, including the unimpressive **3.5-Mini**.
   - The admin confirmed that MoE support is not yet available, leaving users looking for alternatives.
- **OpenAI now offers free fine-tuning**: OpenAI has introduced free fine-tuning for its models, with a **2M token limit** per day for a limited time, intriguing users seeking cost-effective options.
   - Some members, however, have pivoted to OpenRouter after facing issues with OpenAI’s payment methods, particularly in crypto and PayPal.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Merch Store Goes Live**: The **Nous Research merch store has launched**, offering an array of items for fans to showcase their support, including free stickers with orders while supplies last.
   - This initiative aims to create a vibrant community spirit among **Nous Research** enthusiasts.
- **Hermes 3 Takes Center Stage**: Members excitedly discussed the release of **Hermes 3**, with conversations ongoing in a [Twitter Space](https://x.com/i/spaces/1LyxBgzAmVzKN).
   - This event highlighted the latest functionalities and improvements over previous models as community members eagerly tuned in.
- **Decoding OpenAI Compute Grants**: Members explored the nuanced process of acquiring **large compute grants for research**, emphasizing the need for strategic communications with providers.
   - It’s clear that simply requesting unused resources won't suffice; deeper engagement is necessary for success.
- **AI21 Jamba: A New Era in Model Design**: The newly launched **Jamba 1.5 model family** from AI21 claims to be the first non-Transformer model competitive with top models, available under an open license.
   - This model aims to democratize advanced AI tools, striving for quality and accessibility in AI.
- **Tackling PDF Cleaning with Regex**: A member detailed their struggles with **regex for PDF cleaning**, noting poor performance with arxiv PDFs and exploring alternative methods.
   - They resorted to a naive chunk and overlap technique, highlighting the persistent challenges in handling complex PDF structures.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral Fine-Tuning is Crack**: A member remarked that **Mistral's large fine-tuning** is 'crack', indicating exceptional performance but providing no further details.
- **Jamba 1.5: Faster Inference and Long-Context Capabilities**: **AI21's Jamba 1.5** models offer up to **2.5X faster inference** than similar models and enhanced long-context capabilities, aiming at business applications with features like **function calling** and **structured output**.
   - These models are released under the [Jamba Open Model License](https://www.ai21.com/license).
- **Phi 3.5 Mini: Exploding Gradients**: A user reported experiencing **exploding gradients** with the **microsoft/Phi-3.5-mini-instruct** model, persisting even after lowering the **learning rate** to **1e-15**.
   - Attempts to fix it included switching optimizers to **paged_adamw_8bit**.
- **Flash Attention Performance Troubles**: A user encountered errors while trying to use **Flash Attention** for faster training but resolved the issue by switching to **eager attention**.
   - This indicates that **Flash Attention** may not be fully compatible with the model.
- **Accelerate Adds fp8 Support**: **Accelerate** has added support for **fp8**, indicating potential integration with **Axolotl**, although integration points remain uncertain.
   - Discussion revolved around exploring how to effectively incorporate this new support.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud Optimizes RAG Pipeline Performance**: LlamaCloud enhances the efficiency of RAG pipelines by allowing users to manipulate and visualize chunk sizes effectively.
   - Its features include index cloning for rapid experimentation without the hassle of manual data adjustments.
- **LlamaIndex 0.11 Launch Brings Ample Upgrades**: The recent launch of **LlamaIndex 0.11** introduces hundreds of new features, including a refreshed workflows system to replace old query pipelines.
   - This update significantly boosts LlamaIndex's readiness for production by improving user experience.
- **Efficient Memory Management in Ollama**: Discussion centered on managing memory usage for the **Ollama** phi3 model, specifically utilizing the `context_window` parameter to limit context size during operations.
   - This step aims to mitigate error occurrences related to memory capacity.
- **Real Estate Query Generation with LlamaIndex**: A member explored generating queries for a real estate database using natural language within LlamaIndex, evaluating if the tool fits this application.
   - They discussed whether focusing on prompt tuning would yield better results than relying solely on LlamaIndex's capabilities.
- **Challenges in Knowledge Graph Selection**: An article highlighted the complexities involved in selecting appropriate graph stores for managing knowledge graphs in LlamaIndex contexts.
   - Though briefly mentioned, no specific recommendations were provided for optimal graph store choices.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Users Seek Perplexity API Insights**: A few users expressed interest in getting specific guidance on the *Perplexity API*, particularly about accessing features and querying functionalities.
   - Inquiries included domain filtering and citation features, with one user pointing to the [Perplexity API documentation](https://docs.perplexity.ai/reference/post_chat_completions) for chat completions.
- **Mistral Large 2 Gets High Praise**: Mistral Large 2 stands out as a go-to model for custom prompts and unbiased outputs, offering a cost-effective alternative to GPT-4o while maintaining top-notch performance.
   - Users noted its suitability for jailbreak scenarios, reinforcing its status as a preferred tool for complex tasks.
- **Worrying Findings on Microplastics in Brains**: Recent research revealed alarming concentrations of microplastics within human brain samples, urging discussions on health risks associated with plastic pollution.
   - This discovery highlights a critical issue regarding environmental impacts on neurological health and calls for stricter regulations.
- **Jonathan Ive's Ambitious Property Moves**: Sir Jonathan Ive has spent over $137 million securing properties in San Francisco's Jackson Square, signaling a transformative vision for the area.
   - His investment strategy reflects a notable shift in the influence of design on local real estate development.
- **Issues with Perplexity's Image Generation**: Users reported significant challenges with Perplexity's image generation capabilities, struggling to create even simple logos like hearts.
   - Glitches included erratic character outputs in generated images, raising concerns about the reliability of the tool.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Github Desktop Struggles**: A user found [Github Desktop](https://github.com/apps/desktop) less intuitive than expected, stating *'Not the most intuitive product ever'* and noting limited support for `git send-email` and `git am`.
   - This limitation has left users seeking more effective change management solutions.
- **Meet Caroline, the New Community Manager!**: Caroline introduced herself as Modular's new Community Manager, boasting experience in community and developer relations at Streamlit.
   - She encourages members to schedule virtual coffee chats to share feedback and experiences.
- **Improvements Needed for Mojo Docs Search**: Members called for enhancements to the **Mojo documentation search functionality**, pushing for filtering options including **Mojo stdlib modules** and **MAX lib**.
   - They expressed that better navigation would significantly aid user experience and productivity.
- **Mojo/MAX Installation Headaches on MacOS**: A user reported recurring issues with **Mojo and MAX**, requiring a reinstall each time the MacBook Pro restarts.
   - They are seeking advice on managing these installation challenges more effectively.
- **Async vs Sync Performance Debate**: A discussion arose regarding the performance of **async functions** in Mojo compared to **Python**, with suggestions pointing towards a **sans-io HTTP implementation**.
   - This insight reflects an ongoing need for performance optimization in asynchronous operations as IO features evolve.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **RAG Webinar with Cohere and Weights & Biases**: **Cohere** and **Weights & Biases** are hosting a webinar about RAG development and evaluation strategies. Register now at [the webinar link](https://wandb.ai/site/resources/events/cohere-webinar-august).
   - Insights will come from **Maxime Voisin** of Cohere, making this a must-attend for anyone involved in retrieval-augmented generation.
- **Chunking XLSX Files: Community Tips**: Multiple members sought guidance on **chunking XLSX files for RAG**, exploring methods to optimize this process. Suggestions included leveraging embeddings and converting files to markdown for better data handling.
   - This highlights the ongoing collaboration within the community to tackle practical challenges in data processing.
- **Jozu Hub: Your AI Project HQ**: The team released an early preview of **Jozu Hub**, aimed at centralizing AI project versioning and sharing through features like **ModelKit** at [Jozu Hub](https://jozu.ml).
   - This tool aims to streamline AI development by clearly outlining components such as datasets, code, parameters, and documentation.
- **Cohere Model Support on Jozu Hub**: Integration of **Cohere models** on **Jozu Hub** is underway, promising comprehensive support for major models. This move aims to enhance accessibility and usability of different AI frameworks.
   - Anticipated enhancements reflect the commitment to fostering a collaborative AI ecosystem.
- **API Error Troubleshooting**: Several users reported encountering a **403 Forbidden** error when accessing the **Cohere API**, pointing out potential IP whitelisting issues. One member shared details of their POST request, seeking community input.
   - The inquiries and shared experiences emphasize the shared journey through API integration challenges, especially with varying network configurations.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Calculating Expected Value in Games**: A user explored how to calculate the **expected cost** of an item in a game, trying until success or failing four times, with a final cost of **200**.
   - The user seeks to understand the implications of their strategies on overall games costs and mechanics.
- **AI Struggles with Math Problems**: A user expressed frustration with AIs like **Gemini**, **ChatGPT**, and **Claude** for math assistance, facing inaccuracies in results.
   - Another member recommended using **Python** for calculations, emphasizing its efficiency and precision.
- **Ideogram 2.0 Impresses Users**: A user was captivated by **Ideogram 2.0**, a new image generation tool, though it requires a paid subscription to download PNGs.
   - They noted impressive examples shared by others, declaring it '**amazing good**' in handling complex inputs.
- **SwarmUI Simplifies UI for Installers**: A user highly praised **SwarmUI**, which supports **NVIDIA/AMD GPUs** and simplifies interactions with **comfyUI**.
   - They highlighted its user-friendly interface and ability to load shared workflows from the community.
- **Seeking Resources for Custom GPTs**: A user inquired about resources for building **Custom GPTs**, specifically looking for articles and video content.
   - They have already created several models and are eager to refine their GPT creation skills.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Open Source AI Models Face Scrutiny**: Many generative AI models labeled as **open source** often fail to disclose their training sets, raising concerns over the use of **biased** or **copyright-restricted** data. The US government is evaluating risks tied to **'open washing'** in AI models.
   - An article highlights this issue, suggesting that **Eleuther.ai** stands out as a genuine **open source initiative**, aiming for transparency without a profit motive.
- **Optimizing DPO Fine-tuning with Instruction Prompts**: In a discussion regarding **DPO fine-tuning**, users confirmed that applying an **instruction prompt template** to datasets generally enhances its effectiveness. This method aligns the model's output more closely with required tasks.
   - A user also shared methods for prepping **multi-turn chat data**, recommending various structures of input-output pairs to better suit fine-tuning.
- **Examining Model Performance Degradation Techniques**: A member inquired about strategies to reliably reduce **LLM** performance on benchmarks like **MMLU**, aiming to simulate smaller model outcomes. Suggestions included adding noise or implementing **model distillation** with LoRAs.
   - Additional strategies like reversing the training process were also discussed, showcasing a variety of experimental approaches to modify model performance.
- **Model Merging Strategies Spark Debate**: Discussions on **model merging tactics** brought up the idea of applying differences between **UltraChat** and **Mistral** to **Mistral-Yarn**. Despite skepticism, advocates maintained optimism about previous successes with such strategies.
   - This conversation illustrates the vibrant exploration of merging techniques in model development among community members.
- **Understanding Log Likelihood in HellaSwag Evaluations**: Entries in **'resps'** and **'filtered_resps'** are crucial for evaluating models using **negative log likelihood** in multi-choice setups like **HellaSwag**. The structure of these entries indicates which options the model considers more likely.
   - The discussion highlighted complex filtering pipelines used in generation tasks, emphasizing the role of detailed response structures in achieving precise evaluation metrics.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **LightGBM Dominates Kaggle**: LightGBM is making waves in Kaggle competitions like [Corporación Favorita Grocery Sales Forecasting](https://www.reddit.com/r/MachineLearning/comments/ob2rll/d_how_often_is_lightgbm_used_for_forecasting_in/), demonstrating its perceived superiority in predictive performance even in production environments.
   - Participants noted its success in the M5 accuracy competition, solidifying its reputation among practitioners.
- **The LightGBM vs LSTM Debate**: Some experts argue that **LSTM** could outperform **LightGBM** in production, raising questions about its real-world effectiveness compared to competition results.
   - The debate continues, as practical applications often reveal discrepancies between competition and live data performance.
- **LightGBM for Commodity Forecasting Scrutinized**: Research evaluating **LightGBM** for *commodity price forecasts* cited its application in the M5 competition, utilizing features like SMA, EMA, and VWAP.
   - Surprisingly, an ARIMA model outshone LightGBM for lead and tin returns, suggesting model choice must align with forecast specifics.
- **Forecasting Model Choice Matters**: The selection of forecasting models hinges on the task—**LightGBM** can handle multi-step forecasts, but context and prediction complexity are crucial.
   - For tasks requiring longer-term forecasts, such as 3-6 months, earlier methods like SMA and ARIMA should not be overlooked.
- **Pre-Deep Learning Forecasting Techniques**: Prior to deploying deep learning, traditional models like **SMA**, **EMA**, and **ARIMA** often serve as effective starting points for time series forecasting.
   - LightGBM and LSTM shine when dealing with numerous non-traditional exogenous variables where seasonality is less of a concern.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **AI Burnout Takes Center Stage**: Members discussed the phenomenon of **AI burnout**, noting that it feels more intense than in other fields due to its demanding nature.
   - Concerns were raised about how user burnout intertwines with AI burnout, presenting a dual challenge for the community.
- **Frontier Labs Work Intensity Sparks Concern**: A member highlighted that teams in **Frontier Labs** work extremely hard, which raises questions about long-term sustainability.
   - They emphasized the importance of balancing workloads and avoiding burnout, cautioning that the current pace cannot last indefinitely.
- **Greg Brockman Shocks with 97 Work Hours**: A member pointed out **Greg Brockman**'s recent revelation of logging **97 hours** of coding in a single week, highlighting the extreme dedication required in the field.
   - The community expressed surprise at his ability to go **nine years** without a break, questioning the implications for work-life balance in tech.
- **Twitter Anxiety Post-Unplugging**: Coming back from a digital detox, a member voiced their discomfort diving back into **Twitter**, describing the platform's atmosphere as anxiety-inducing.
   - They lamented the intense discussions surrounding AI on Twitter, especially after finding peace in the backcountry.
- **Lilian Weng Spotlights Diffusion Models**: **Lilian Weng's** updated blog post on [Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) discusses various **Generative Models** and new sections on consistency models and architecture.
   - The conversation emphasized the evolving nature of the field, with one user clarifying the distinction between **Diffusion Models** and **Distillation**.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Local LLMs tackle NL to SQL**: A user raised the question of using a local LLM for **natural language to SQL** translation, exploring its viability and performance.
   - This sparked discussions on its potential to simplify query generation.
- **Prebuilt Queries streamline SQL work**: The suggestion to use **prebuilt queries** with placeholders for text-to-SQL conversion aims to ease the workload involved.
   - Members discussed the efficiency gains and simpler management this approach could provide.
- **RAG with CodeLLM for better SQL**: Combining **Retrieval Augmented Generation** (RAG) with a code-specific LLM was proposed as a means to enhance SQL generation.
   - This could lead to improved accuracy in generating valid SQL commands.
- **4149 AI introduces 'Flags' feature**: [4149 AI](https://4149.ai) launched a new **'Flags' feature** that sends real-time guidance on team status through Slack direct messages.
   - It offers customizable alerts and aims to catch potential team issues before they escalate.
- **Excitement for AI in Research**: Members expressed enthusiasm about the innovative use cases for **AI in research**, highlighting its transformative potential.
   - This sentiment indicates a thriving interest in integrating AI into various research methodologies.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Ideogram 2.0 Launch: Free for Everyone**: Ideogram 2.0, the latest text-to-image model from the former Google Imagen 1 team, is now available to all users for free. This release includes an iOS app, Beta API, and Ideogram Search, claiming over 1 billion images created.
   - *Tis the season of sequels*, as noted by [AI News by Smol AI](https://x.com/Smol_AI/status/1826410077194256729), with continued buzz around features and performance.
- **Nvidia's New Mistral-NeMo-Minitron-8B**: Nvidia has launched Mistral-NeMo-Minitron-8B, a base LLM obtained by pruning the Mistral-NeMo 12B. It outperforms Mistral-7B and LLaMa-3.1-8B across 8 out of 9 benchmarks, now available on Hugging Face.
   - Philipp Schmid tweeted on its significance, stating it was built using **400B tokens** for effective training enabling high performance across tasks.
- **Sovereign AI: A New Streaming Data System**: The Infinite ML podcast covers Sovereign AI, a streaming data system developed by Redpanda Data. Topics touched upon include its real-world applications and the evolution of streaming data.
   - Prateek Joshi provided insights into the system’s capabilities, emphasizing its use for enhanced data management and speed.
- **GPT-4o Fine-Tuning: Worth It?**: The Latent Space Podcast examines the value of fine-tuning GPT-4o, featuring Alistair Pullen from Cosine discussing its implications. OpenAI has officially launched fine-tuning capabilities aimed at improving application performance.
   - Swyx pointed out that there are over 59 different flavors of RAG with advancements in token context management, suggesting a complex landscape for developers.
- **Genie's Massive Fine-Tuning Effort**: Genie has begun a large-scale fine-tuning initiative for GPT-4o, leveraging billions of tokens of synthetic code data derived from user logs. This effort seeks to optimize performance through targeted data manipulation.
   - The discussion highlights the significance of synthetic data for enhancing model accuracy, reflecting the growing trend towards leveraging real-world usage patterns.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Searching Woes in Open Interpreter**: A user reported that **web searching** in Open Interpreter only functions after a full terminal refresh, causing disruption in ongoing conversations.
   - This issue highlights potential usability constraints that could hinder workflow efficiency.
- **Promising Model Suggestions**: A member recommended that **Phi-3.5-mini** and **Qwen2** models are surprisingly effective for various tasks.
   - This suggests exploring alternative models could yield better project outcomes.
- **Mystery of the Model Type**: Curiosity arose when a user questioned the specific model used by another participant, suspecting it was not **GPT-4**.
   - Model transparency can significantly impact user experience and expectations in development discussions.
- **Interface Documentation Over Command Line**: Concerns were raised regarding Open Interpreter’s **interface documentation**, suggesting it’s more intuitive than relying on shifting command line bookmarks.
   - This feedback points to a desire for more stable navigation aids and clearer documentation for user workflows.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba 1.5 Revolutionizes Model Architecture**: AI21 Labs has launched **Jamba 1.5 Mini** (12B active/52B total) and **Jamba 1.5 Large** (94B active/398B total), leveraging the **SSM-Transformer architecture** that combines **Transformer's quality** with enhanced efficiency.
   - Both models feature a **256K effective context window**, achieving speeds **2.5X faster on long contexts** compared to competitors.
- **Jamba 1.5 Large Sets New Performance Benchmarks**: **Jamba 1.5 Mini** scores **46.1** on **Arena Hard**, while **Jamba 1.5 Large** exceeds expectations at **65.4**, outpacing both **Llama 3.1 70B** and **405B**.
   - Multi-language support enhances usability as models natively handle **English, Spanish, French, Hebrew, Arabic**, with functionalities for **JSON output and document processing**.
- **Access Jamba 1.5 Today**: **Jamba 1.5 Mini and Large** are instantly available on **Hugging Face** and can be deployed on platforms like **Together AI, AWS, GCP, Azure**.
   - AI21 Labs releases these models under the **Jamba Open Model License**, promoting democratized access to such advanced models.
- **Jamba-1.5 Fine Tuning Update**: Questions arose regarding fine-tuning for **Jamba-1.5**, leading to confirmation that only instruct versions are available on Studio, with fine-tuning not currently offered.
   - **Jamba-1.5 Large** remains the most advanced model with robust features for reasoning, code generation, and multilingual processing.
- **OpenAI API Rate Limits Clarified**: Users discussed the OpenAI API rate limits, confirming it's set at **200 requests per minute (RPM)** and **10 requests per second (RPS)**.
   - This clarification reinforces the community's understanding of API consumption while working with extensive models.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Code Review Accountability Takes Center Stage**: Frustration emerged over code review responses that shift responsibility, with comments like 'I will do it if you want/ask.' Authors must take ownership and engage with suggestions, either implementing changes or offering a reasoned explanation.
   - This push for accountability aims to foster a more rigorous review process and encourage critical thinking in code contributions.
- **Exploring Mypyc with Tinygrad**: Interest sparked around getting **Tinygrad** to compile with **mypyc**, highlighting the potential for performance improvements.
   - A member stepped up, offering to investigate the compilation issue and contribute to the project’s evolution.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Faces T5 Attention Bias**: A member highlighted that the biggest hurdle with **T5** is its trainable attention bias, but other components remain standard.
   - Currently, **Torchtune** lacks support for encoder-decoder architectures, necessitating adjustments to the task-specific training loop.
- **Mapping Weights: Hugging Face vs Torchtune**: A suggestion was made to compare weight naming conventions between the **Hugging Face** and **Torchtune** repositories for mapping purposes.
   - The focus was on the **T5-small** model from Hugging Face and the [convert_weights.py](https://github.com/pytorch/torchtune/blob/861c3b214b029ee819fb554055aa467a52f22862/torchtune/models/convert_weights.py#L31-L45) file in Torchtune.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LinkedIn Survey Seeks User Insights**: A survey is conducted to gather perceptions on **LinkedIn** as a professional networking platform, inviting insights from the community. Participate in the survey [here](https://qualtricsxmbyhfby7vf.qualtrics.com/jfe/form/SV_bDttqTVpxzTJvLw).
   - This initiative hopes to uncover varied aspects of LinkedIn user experiences, welcoming participation from a broad audience.
- **Dev Needed for Infinite Generative YouTube**: A team is launching a closed beta for an **infinite generative YouTube** project and seeks motivated developers to join the effort. They're looking for enthusiasts ready to engage with innovative models.
   - Interested developers are encouraged to reach out to learn more about this exciting opportunity to shape a new media experience.
- **EMNLP 2024 Workshop Seeks Reviewers**: The **Multilingual Representation Learning** workshop at **EMNLP 2024** is calling for reviewers; sign up [here](https://forms.gle/ica4F94jaTbvdb689). This initiative aims to assemble a diverse group to evaluate workshop submissions.
   - Reviewers will explore a variety of topics, including ethics in multilingual models, low-resource methods, and cultural analytics, bringing fresh perspectives to the discussion.
- **Workshop Explores Diverse Multilingual NLP Topics**: The EMNLP 2024 workshop will cover diverse subjects such as **dialogue systems**, **discourse**, and **machine translation**. It's designed to address pressing issues in multilingual NLP.
   - Participants can expect discussions on ethics, phonology, and multimodality, enriching the understanding of challenges and advancements in the field.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM Leaderboards Show Discrepancies**: A member questioned the **difference** between the [Website Leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1276013369727647825) and the [Huggingface Leaderboard](https://huggingface.co/), noting that Huggingface scores are significantly higher.
   - The leaderboard change emphasizes that **subcategories** like `python_simple_function` and `java_simple_function` hold equal importance in model evaluation.
- **Comprehensive Model Evaluation Required**: The emphasis is on developing a good model that excels in all aspects, not merely in selective subcategories, as discussed in #580.
   - This holistic assessment approach ensures a more reliable metric for model performance.
- **Locally Evaluating Fine-tuned Models on BFCL**: Members explored **steps** for evaluating a fine-tuned model on BFCL locally, specifically looking into **multi-GPU** utilization.
   - While no specific guidance was provided, the inquiry reflects the growing interest in optimizing local evaluations.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Prompt Caching Exploration**: A user inquired about the possibility of implementing **prompt caching** to enhance efficiency in AI interactions.
   - While the discussion is in its early stages, it's clear that caching can significantly reduce latency and improve response times.
- **Antropic API Usage Inquiry**: Another user asked how to integrate the **Antropic API** for better performance in their AI models.
   - Integrating the API may allow for more refined control over responses and could open up new avenues for experimentation.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **OSI drafts definition of Open Source AI**: The Open Source Initiative (OSI) has released a draft definition of **open source AI**, the result of two years of community discussions and debates.
   - This landmark definition aims to redefine **'open source'** within AI, potentially shaping its societal impact and guiding future development.
- **Community engagement through OSI Town Hall**: A Town Hall event hosted by OSI facilitated discussion on the new draft definition of open source AI, inviting further community input.
   - This initiative aligns with OSI's goal to promote transparency and engagement among stakeholders in the open source AI space.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **OASST-2 Dataset Steals the Spotlight for German Tuning**: The **OASST-2 dataset** includes a **German subset** that's a promising choice for instruction tuning tasks.
   - With high-quality examples, it can facilitate advancements in **German-language AI models**.
- **Aya-Dataset Joins the Instruction Tuning Party**: Another option, the **Aya-Dataset**, harbors a **German subset** suitable for instruction tuning.
   - Its diverse examples can help boost the training of models designed for **German** instruction tasks.
- **Curate Your Own German Instruction Datasets!**: Datasets like **Colossal Cleaned Common Crawl** and **German Wikipedia** can supplement instruction tuning efforts but require significant filtering.
   - Careful curation could yield valuable resources focused on **German instruction data**.
- **Build a Custom Dataset by Translating English Instructions**: Considering the creation of a **custom dataset** that translates English instruction data into German could enhance specific AI functionalities.
   - This approach allows targeted adaptations for unique project requirements in **software engineering**.
- **Open Source Your Llama 3.1 Based MoE!**: The idea of open sourcing an **8x8b Llama 3.1-based MoE** with both German and English instruction tuning makes waves in the community.
   - Such a contribution could greatly benefit the broader **NLP landscape** by increasing accessibility and collaboration.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1276244752798449826)** (1 messages): 

> - `LM Studio 0.3.0`
> - `LM Studio Features`
> - `LM Studio Community`
> - `LM Studio UI`
> - `LM Studio Models` 


- **LM Studio 0.3.0 is here!**: The latest update, [LM Studio 0.3.0](https://lmstudio.ai/blog/lmstudio-v0.3.0), is available for Mac, Windows (x86/ARM), and Linux (x86).
   - The update features numerous improvements to LM Studio's core features, based on the community's feedback over the past year.
- **New Features in LM Studio 0.3.0**: This update brings [chat with documents](https://lmstudio.ai/blog/lmstudio-v0.3.0), [OpenAI-like structured outputs](https://lmstudio.ai/blog/lmstudio-v0.3.0), [multiple LLM support](https://lmstudio.ai/blog/lmstudio-v0.3.0), [network sharing](https://lmstudio.ai/blog/lmstudio-v0.3.0), [automatic parameter loading](https://lmstudio.ai/blog/lmstudio-v0.3.0), [UI themes](https://lmstudio.ai/blog/lmstudio-v0.3.0), [folders for chat organization](https://lmstudio.ai/blog/lmstudio-v0.3.0), and [multiple generations per chat](https://lmstudio.ai/blog/lmstudio-v0.3.0).
   - It's available in 7 languages and includes updated LLM runtimes (llama.cpp).
- **LM Studio is now OpenAI-like**: LM Studio 0.3.0 now offers an OpenAI-like 'Structured Outputs' API that works with any local model, allowing for more structured output from your LLMs.
   - This feature unlocks a range of possibilities for integrating local models into your projects.
- **LM Studio: a Community Effort**: The developers of LM Studio are grateful to their community for their feedback and contributions.
   - They encourage users to try out the new update and share their thoughts.
- **LM Studio UI Gets a Revamp**: The latest update brings a completely revamped user interface to LM Studio, with improved navigation and a modern look.
   - The UI now offers dark, light, and sepia themes to suit different preferences.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.0">LM Studio 0.3.0 | LM Studio</a>: We&#x27;re incredibly excited to finally share LM Studio 0.3.0 🥳.</li><li><a href="https://x.com/LMStudioAI/status/1826680869773357513">Tweet from LM Studio (@LMStudioAI)</a>: After months of work, and with the help of our awesome community, we&#39;re excited to finally share LM Studio 0.3.0! 🎉  🔥 What&#39;s new:  - Built-in Chat with Documents, 100% offline - OpenAI-like...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1275895941701832755)** (412 messages🔥🔥🔥): 

> - `LM Studio 0.3.0`
> - `Gemma 2`
> - `LLaMa 3.1`
> - `LM Studio performance`
> - `LM Studio UI` 


- **LM Studio 0.3.0 Released - Massive Updates!**: LM Studio 0.3.0 has been released, featuring a new UI with improved chat structure and folder support, automatic context length handling, and a new model loader with built-in multi-model loading capabilities.
   - The update also includes a new server that supports more than localhost, RAG functionality, and improved performance for local models. However, some bugs and issues have been reported, including problems with model loading, system prompt restrictions, and the SDK.
- **Gemma 2 Still a Top Contender**: Many users continue to praise Gemma 2 9B and 27B for their performance, especially compared to other models like LLaMa 3.1.
   - However, Gemma 2 is limited by its 8k context window, prompting users to seek alternative models for tasks requiring larger context sizes. Users are excited about Phi 3 Mini and the potential for MoE models to be added to LM Studio.
- **LM Studio Bugs and Issues**: Several users have reported encountering bugs and issues with LM Studio 0.3.0, including problems with model loading, system prompt restrictions, and the LM Studio SDK.
   - One user reported an issue with file attachments in the first message, and another user encountered a problem with model loading that could be resolved by removing the `vulkan` directory.
- **LM Studio API and Extensions**: Users are interested in utilizing LM Studio's API and connecting it to other applications, including mobile apps.
   - Users have explored potential ways to connect Android and iOS apps to LM Studio, and discussions about creating a persistent communication system with LM Studio's AI models have emerged.
- **LM Studio Features and Functionality**: Users are actively exploring new features and functionalities of LM Studio, including RAG, which allows users to upload files and ask the LLM questions about them.
   - Users have also inquired about the availability of vision-enabled models in LM Studio and the ability to run the GUI without hardware acceleration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://huggingface.co/MaziyarPanahi/SmolLM-135M-Instruct-GGUF">MaziyarPanahi/SmolLM-135M-Instruct-GGUF · Hugging Face</a>: no description found</li><li><a href="https://chatboxai.app/">Chatbox AI: Your AI Copilot, Best AI Client on any device, Free Download</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=7OcwwYtKsec">Setting Up BEST Local LLMs for Obsidian AI With LM Studio</a>: ♥️ Join as a YouTube Member to help me make better / more videos!https://www.youtube.com/channel/UCFiN1FnTVWKX1G_UwgZHURA/joinhttps://Patreon.com/SystemSculp...</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio CLI</a>: LM Studio CLI. Contribute to lmstudio-ai/lms development by creating an account on GitHub.</li><li><a href="https://huggingface.co/">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://github.com/quentinwolf/lmstudio">GitHub - quentinwolf/lmstudio: LM Studio Related Stuff</a>: LM Studio Related Stuff. Contribute to quentinwolf/lmstudio development by creating an account on GitHub.</li><li><a href="https://huggingface.co/MaziyarPanahi/">MaziyarPanahi (Maziyar Panahi)</a>: no description found
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1275907051272536076)** (72 messages🔥🔥): 

> - `SLI/NVLink for LLMs`
> - `GPU Memory for Large Models`
> - `Model Size vs Speed`
> - `Waiting for New Hardware`
> - `GPU Recommendations` 


- **SLI Doesn't Double Speed**: Two GPUs in SLI do not deliver twice the speed of a single GPU, due to architectural limitations and overhead.
   - This means that while SLI allows you to load larger models, it does not offer significant performance gains compared to a single, powerful GPU.
- **More RAM for Larger Models?**: One user suggested that increasing system RAM is a cheaper alternative to buying additional GPUs if you are only concerned with running larger models.
   - They shared their experience running Llama 3.1 405B on a system with 128GB RAM but only 16GB of VRAM, although performance was slow.
- **GPU Memory is Key**: The discussion highlighted the importance of VRAM for running larger models effectively, as even powerful CPUs cannot handle these models efficiently on their own.
   - Members concluded that for large models like Llama 3.1 405B, a GPU with at least 48GB of VRAM is recommended for faster inference speeds.
- **Waiting is Wise**: Many users agreed that waiting for new hardware releases is generally advisable, given the rapid pace of advancements in GPU technology.
   - They advised against impulsive purchases, as newer generations often offer better value or performance, and waiting allows for price drops on older hardware.
- **GPU Recommendations**: A6000, 4090D, and H100 were mentioned as powerful GPUs suitable for large models, but their prices are high.
   - The 4060 with 24GB VRAM was deemed insufficient for Llama 3.1 405B, highlighting the need for significantly higher VRAM capacity for smooth performance.



**Link mentioned**: <a href="https://tenor.com/view/funny-very-sloth-slow-gif-15401812">Funny Very GIF - Funny Very Sloth - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1276296800348672123)** (1 messages): 

> - `Offensive Security Reconnaissance`
> - `Deep Learning Course`
> - `Unity ML Agents`
> - `Garfield dataset`
> - `Tensor parallelism` 


- **Offensive Security Reconnaissance by <@348869248963182592>**: A verified user <@348869248963182592> shared a blogpost about Offensive Security Reconnaissance.
   - You can find the blogpost here: [https://huggingface.co/posts/Csplk/182147151896536](https://huggingface.co/posts/Csplk/182147151896536)
- **French deep learning course updated by <@221626552792776704>**: User <@221626552792776704> announced an update to their French deep learning course with a new website to make navigating easier.
   - The website is now more intuitive and can be found here: [https://simonthomine.github.io/CoursDeepLearning/](https://simonthomine.github.io/CoursDeepLearning/)
- **Unity ML Agents Part 4**: A YouTube video titled "Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 4" was shared by <@330939073491369985>.
   - The video continues a series on creating an intelligent chatbot using Unity ML-Agents and Sentence Transformers and can be viewed here: [https://youtube.com/live/RdxtA_-47Kk?feature=share](https://youtube.com/live/RdxtA_-47Kk?feature=share)
- **Garfield dataset shared by InternVL2**: A Garfield dataset was shared by InternVL2, thanks to <@636706883859906562>.
   - The captioned dataset can be found here: [https://huggingface.co/datasets/terminusresearch/garfield](https://huggingface.co/datasets/terminusresearch/garfield)
- **Tensor parallelism by <@732217032506081363>**: User <@732217032506081363> shared a blogpost about tensor parallelism.
   - The blogpost can be accessed here: [https://huggingface.co/blog/huseinzol05/tensor-parallelism](https://huggingface.co/blog/huseinzol05/tensor-parallelism) 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/live/RdxtA_-47Kk?feature=share)">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 4</a>: Welcome back to our exciting series on creating an intelligent chatbot using Unity ML-Agents and Sentence Transformers! 🚀In this episode, we wrap up some cr...</li><li><a href="https://www.youtube.com/watch?v=bKzmtTfcaqc)">Prototype 5 : Real time Text to Audio to Face Blendshape animation</a>: huggingface.co/AnimaVR/NeuroSync-0.1a</li><li><a href="https://youtu.be/qsWn3SUz-LM)">Generate Ultra-Realistic Images with Flux! Realism Lora (Flux 1 Dev)</a>: I&#39;ll be showing you how you can run Flux with Realism LoRa all for free online without any installations! As promised, here are the links to the Huggingface ...</li><li><a href="https://huggingface.co/spaces/AIPeterWorld/Doc-To-Dialogue?logs=container)">Doc To Dialogue - a Hugging Face Space by AIPeterWorld</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1275893707987484824)** (246 messages🔥🔥): 

> - `Hugging Face prepaid credit system`
> - `AI21 Labs Jamba 1.5`
> - `GPU and VRAM`
> - `HackAI Challenge`
> - `Prepaid cards` 


- **Hugging Face Prepaid Credit System - Money Stuck?**: A user reported that their prepaid card was declined for a $10 temporary hold for a Hugging Face account, but the money was still taken.
   - Hugging Face staff confirmed that this is a common issue with prepaid cards and that the hold should clear within a few business days. They advised the user to wait a few days and contact billing@huggingface.co if the issue persists.
- **AI21 Labs Releases Jamba 1.5!**: AI21 Labs has released a new language model, Jamba 1.5, which comes in two sizes: Mini (52B - 12B active) and Large (398B - 94B active).
   - Jamba 1.5 boasts features like instruct models, long context (256k), strong quality, multilingual support, function call, JSON output, document understanding, and a hybrid transformer/Mamba architecture. More information can be found on Hugging Face ([https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251](https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251)).
- **RTX 6000 - 48GB of VRAM?**: The HackAI Challenge, hosted by Dell and NVIDIA, offers an RTX 6000 mobile workstation as a prize.
   - The RTX 6000 has 48GB of VRAM, making it a powerful tool for developers and data scientists. It's a tempting prize for anyone looking to push the boundaries of generative AI projects.
- **HackAI Challenge: Dell & NVIDIA**: The HackAI Challenge, hosted by Dell and NVIDIA, encourages developers and data scientists to create innovative Generative AI projects using NVIDIA AI Workbench.
   - The challenge is open to anyone with an interest in AI, and the prize includes an RTX 6000 mobile workstation, 10,000 USD, and other perks. You can learn more about the challenge on Devpost ([https://hackaichallenge.devpost.com/?utm_source=devpost&utm_medium=newsletter&utm_campaign=08222024](https://hackaichallenge.devpost.com/?utm_source=devpost&utm_medium=newsletter&utm_campaign=08222024)).
- **Prepaid Card Woes: Hugging Face vs. Others**: A user discussed their experience with prepaid cards being declined for Hugging Face payments, while working fine for other services like Unity, GitHub, Amazon, and Azure.
   - Another user confirmed that they had a similar experience with an Amex prepaid card, and the money was refunded after a few days. The discussion centered around the mystery of the American credit system and the challenges of using prepaid cards for online transactions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/osanseviero/status/1826607725280682154">Tweet from Omar Sanseviero (@osanseviero)</a>: AI21 released 🥁 Jamba 1.5   - Mini (52B - 12B active) + Large (398B - 94B active) - Instruct models - Long context (256k) - Strong quality - Multilingual - Function call, JSON output, document unders...</li><li><a href="https://www.instagram.com/p/C-9DnNrIr1D/?img_index=2">Noa Roggendorff on Instagram: &quot;glimmer

#art&quot;</a>: 5 likes, 0 comments - noaroggendorff on August 21, 2024: &quot;glimmer  #art&quot;. </li><li><a href="https://tenor.com/view/mr-krabs-money-spongebob-gif-8454828">Mr Krabs Money GIF - Mr Krabs Money Spongebob - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ey6hss/kohya_ss_gui_flux_lora_training_on_rtx_3060_lora/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/view/lotr-lord-of-the-rings-theoden-king-of-rohan-you-have-no-power-here-gif-4952489">Lotr Lord Of The Rings GIF - LOTR Lord Of The Rings Theoden - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/cat-cat-meme-cat-staring-cat-stare-stop-staring-at-me-please-gif-9315292369693569596">Cat Cat Meme GIF - Cat Cat meme Cat staring - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/drama-queen-dramaticing-oh-the-drama-lion-king-gif-13113200">Drama Queen GIF - Drama Queen Dramaticing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/iamproudofyou-my-hero-gif-18489622">Iamproudofyou My Hero GIF - Iamproudofyou My Hero - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://hackaichallenge.devpost.com//?utm_source=devpost&utm_medium=newsletter&utm_campaign=08222024">HackAI - Dell and NVIDIA Challenge</a>: Code, Create, Conquer - Build groundbreaking Generative AI projects using NVIDIA AI Workbench</li><li><a href="https://www.16personalities.com/free-personality-test">Free personality test | 16Personalities</a>: no description found</li><li><a href="https://www.16personalities.com/personality-types#analysts)">Personality Types | 16Personalities</a>: no description found</li><li><a href="https://www.16personalities.com/personality-types#diplomats)">Personality Types | 16Personalities</a>: no description found</li><li><a href="https://www.16personalit>>>">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1276244964065415178)** (7 messages): 

> - `Neuralink Paper Selection`
> - `Coding with Neuralink`
> - `Reading Research Papers` 


- **Neuralink's paper selection process**: Neuralink reads thousands of papers every month, but they use a citation-tracing method to focus on papers relevant to their interests.
- **Coding alongside research**: Neuralink takes notes on research papers and tries to code them as they read, learning from the concepts and approaches discussed.
- **Neuralink's code quality**: Neuralink's code is described as 'neat' and demonstrates a level of proficiency in implementing research findings.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1275919653939712000)** (2 messages): 

> - `3DGS`
> - `ShapeSplat dataset`
> - `Gaussian Splats`
> - `Self-Supervised Pretraining`
> - `Point Cloud Representation` 


- **ShapeSplat: A New 3DGS Dataset**: A new dataset, **ShapeSplat**, was introduced, featuring **Gaussian Splats** and a **self-supervised pretraining** method for 3D Generative Shape Synthesis (3DGS).
   - The dataset aims to address the need for large-scale, high-quality data in the field, offering various advantages over existing point cloud representations.
- **ShapeSplat's Key Features**: ShapeSplat comprises **a diverse collection of objects**, including **human bodies, animals, furniture, and more**, offering a wide range of shapes for training and evaluation.
   - It is designed to facilitate the development of advanced 3DGS models, particularly in areas like **shape completion, reconstruction, and generation**.
- **ShapeSplat's Advantages over Existing Methods**: The use of **Gaussian Splats** enables **efficient representation and manipulation** of 3D shapes, compared to traditional point clouds.
   - The **self-supervised pretraining** approach allows for the creation of models that generalize well to new, unseen data.
- **ShapeSplat's Potential Applications**: The dataset is expected to have significant implications for various applications, including **computer graphics, robotics, and virtual reality**.
   - Its use in **shape generation, editing, and analysis** can advance research and development in these fields.
- **ShapeSplat's Authors and Affiliations**: The dataset was created by a team of researchers from **ETH Zürich, INSAIT, the University of Amsterdam, the University of Pisa, and the University of Trento.**
   - The authors highlight the potential of ShapeSplat to contribute significantly to the field of 3DGS.



**Link mentioned**: <a href="https://unique1i.github.io/ShapeSplat/">ShapeSplat: A Large-scale Dataset of Gaussian Splats and Their Self-Supervised Pretraining</a>: no description found

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1275941124363780186)** (10 messages🔥): 

> - `Fullbound`
> - `Patch Tripper`
> - `Offensive Security Reconnaissance`
> - `On-device Transcription`
> - `Inferless` 


- **Fullbound: AI-Powered Job Matching**: Fullbound is a global search engine built on Chroma using `nomic-ai/nomic-embed-text-v1.5` and `xenova` to match job seekers with open job roles, offering AI-powered matching technology to quickly align candidates with roles, contact decision makers, and stay up to date with clients.
   - They offer a 7-day free trial and [pricing information can be found here](https://fullbound.ai/pricing).
- **Patch Tripper: Website Launch**: A user shared their newly created website, [Patch Tripper](https://patchtripper.com/), and asked for feedback.
   - Another user commented on the website's clean design and accurate content, suggesting that it incorporates design elements from online academies like Unity and Unreal.
- **Offensive Security Reconnaissance with Moondream**: A user discussed their use of Moondream for offensive security reconnaissance, specifically focusing on public facing Industrial Control System (ICS) Human-Machine Interfaces (HMIs).
   - They highlighted the potential for exploiting vulnerabilities in ICS HMIs to gain unauthorized access to critical infrastructure, and shared a link to their [batch processing code](https://huggingface.co/spaces/Csplk/moondream2-batch-processing) for analyzing ICS HMIs.
- **On-Device Transcription App**: A user shared their on-device transcription app, which was featured in the Svelte showcase, showcasing the app's functionality and its use of Svelte.
   - They also provided a link to the app's [GitHub repository](https://github.com/Hugo-Dz/on-device-transcription), which is licensed under the MIT license.
- **Inferless Townhall: New UI Demo**: A user invited others to attend the first Inferless live townhall to showcase the new Inferless Console 2.0, specifically addressing ML engineers struggling with issues like high cold-start times and ineffective serverless orchestration for ML workloads.
   - They shared the [townhall link](https://dub.sh/inferless-newui) and highlighted the opportunity to get live answers from Inferless founders about the new deployment process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/posts/Csplk/182147151896536">@Csplk on Hugging Face: &quot;# Offensive Security Reconnaissance Continued with Public Facing Industrial…&quot;</a>: no description found</li><li><a href="https://fullbound.ai">Fullbound - A dream tool for every recruiter</a>: Boost your placement game with similarity matching, natural language searching, and the latest AI technology to make sure every hire is a win-win for businesses of all sizes.</li><li><a href="https://logllm.tiiny.site">LogLLM - Automate ML Experiment Logging</a>: no description found</li><li><a href="https://patchtripper.com/">Patch Tripper</a>: Patch Tripper</li><li><a href="https://dub.sh/inferless-newui)">Dub.co - Link Management for Modern Marketing Teams</a>: Dub.co is the open-source link management platform for modern marketing teams to create marketing campaigns, link sharing features, and referral programs.</li><li><a href="https://svelte.dev/blog/whats-new-in-svelte-august-2024">What's new in Svelte: August 2024</a>: no description found</li><li><a href="https://github.com/Hugo-Dz/on-device-transcription">GitHub - Hugo-Dz/on-device-transcription: A ready-to-use, minimal app that converts any speech into text.</a>: A ready-to-use, minimal app that converts any speech into text. - Hugo-Dz/on-device-transcription
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1275913950780723331)** (1 messages): 

> - `Language Alignment Techniques`
> - `DPO Paper`
> - `Direct Preference Optimization` 


- **Request for Reading Material on Language Alignment**: A member expressed interest in reading materials related to **Language Alignment Techniques**, particularly papers like the **DPO (Direct Preference Optimization) paper**.
- **No Further Discussion**: There was no further discussion on this topic.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1276096927150112808)** (1 messages): 

> - `Swin Transformers as Mask R-CNN Backbone` 


- **Swin Transformer Backbone in Mask R-CNN**: A member inquired about the feasibility of using **Swin Transformers** as the backbone for **Mask R-CNN** when training on a custom dataset.
   - They expressed difficulty finding existing implementations of this approach online, suggesting a lack of widespread adoption or publicly available code.
- **Alternatives to Swin Transformers for Mask R-CNN**: No other suggestions for backbones were given, but it was pointed out that Swin Transformers might not be the best choice for Mask R-CNN, though it is theoretically possible.
   - The user may want to explore other backbones known to be effective with Mask R-CNN, such as **ResNet** or **EfficientNet**.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1276015355596378203)** (2 messages): 

> - `Multilingual NLP Research` 


- **Low-Resource Multilingual NLP**: Research in low-resource multilingual NLP is crucial, as many languages lack sufficient data for training high-performing models. 
   - One open problem is developing effective techniques for transferring knowledge from high-resource languages to low-resource ones, such as cross-lingual transfer learning or zero-shot learning.
- **Cross-Lingual Semantics and Sentiment Analysis**: Understanding sentiment and meaning across languages is a challenging but important area in multilingual NLP. 
   - One open problem is developing robust methods for cross-lingual sentiment analysis, especially in the presence of cultural and linguistic variations in sentiment expression.
- **Multilingual Information Extraction and Retrieval**: Extracting and retrieving information from multilingual documents is essential for tasks like cross-lingual knowledge graph construction and multilingual search. 
   - An open problem is developing effective methods for cross-lingual information extraction, particularly for handling complex sentence structures and diverse linguistic features across languages.
- **Ethics, Bias, and Fairness in Multilingual Models**: Multilingual NLP models can inherit biases from the data they are trained on, leading to unfair outcomes for certain language groups. 
   - An open problem is developing techniques for mitigating bias and promoting fairness in multilingual models, such as bias detection and mitigation methods specific to multilingual contexts.



**Link mentioned**: <a href="https://forms.gle/ica4F94jaTbvdb689">Call for Reviewers - 4th Multilingual Representation Learning (MRL) Workshop, EMNLP 2024 </a>: This form is for anyone who is interested in being a reviewer for our workshop at EMNLP 2024.  Reviewers are invited to indicate their interest and their application will be assessed in relation to th...

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1275928013438980136)** (9 messages🔥): 

> - `Quanto Qint2 Quantization`
> - `NeuroSama API`
> - `CFG++ Support in Diffusers`
> - `Fine-tuning Diffusion Models with LoRA`
> - `Importing 3D Models into Images` 


- **Quanto Qint2 Quantization Explained**: A member explained that **I2** refers to **Quanto Qint2 Quantization**, a technique used in the `diffusers` library.
   - They also recommended using the [gist](https://gist.github.com/sayakpaul/e1f28e86d0756d587c0b898c73822c47#file-inference_with_torchao_serialized-py) for further information on quantization.
- **Inquiry about NeuroSama API**: A member asked about the **API** used by **NeuroSama**,  a generative AI chatbot.
- **CFG++ Support in Diffusers**: A member inquired about the **CFG++** technique, which is known for achieving good results with low CFG values, and its availability in **Diffusers**.
- **Fine-tuning Diffusion Models with LoRA**: A member sought guidance on fine-tuning **unconditional diffusion models** using the **LoRA** module from the **PEFT library**.
   - They specifically requested **gists or similar resources** to help them overcome difficulties with overfitting and potential incorrect usage of the **LoRA module**.
- **Importing 3D Models into Images**: A member asked about the process of importing **3D image models** into images, using the example of adding police lights to a car image.
   - They emphasized their desire to import **specific 3D models** like police lights, rather than relying on random generation.



**Link mentioned**: <a href="https://gist.github.com/sayakpaul/e1f28e86d0756d587c0b898c73822c47#file-inference_with_torchao_serialized-py.">Shows how to run Flux schnell under 17GBs without bells and whistles. It additionally shows how to serialize the quantized checkpoint and load it back.</a>: Shows how to run Flux schnell under 17GBs without bells and whistles. It additionally shows how to serialize the quantized checkpoint and load it back. - inference_with_torchao_serialized.py

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1275938584972492811)** (153 messages🔥🔥): 

> - `Gemma-2-27B-it fine-tuning`
> - `Mistral Nemo 12b`
> - `Unsloth Pro`
> - `Unsloth training`
> - `Unsloth multi-gpu` 


- **Mistral Nemo 12b is finetunable on 8gb GPU**: A member commented that **Mistral Nemo 12b** can be fine-tuned on an **8gb GPU**, specifically mentioning the **RTX 4050**.
   - They also said that this would be good for testing and prototyping.
- **Unsloth Pro has halted multi-GPU support**: **Unsloth Pro** has temporarily halted multi-GPU support, offering it only through their platform.
   - Early access has been granted to trusted community members.
- **Unsloth has no AMD support**: Unsloth developers confirmed that **AMD GPUs** are not supported, citing a *not-planned/wontfix* issue with **xformers** on **AMD's RDNA architecture**. 
   - They suggest using **flashattention** instead.
- **Unsloth platform accessibility is estimated by end of October**: The Unsloth team estimates that the platform will be accessible by **the end of October**, though this is a rough estimate and could take longer.
   - They are actively working on this and are aiming to make it a reality. 
- **LLM 8-bit model finetuning on a 4-bit model is possible**: An **8-bit model** can be fine-tuned with a **LoRA adaptor** trained on a **4-bit model**, even though Unsloth doesn't offer an **8-bit version**. 
   - This allows for more efficient training and prototyping on GPUs with limited VRAM.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1826066519190839429">Tweet from Daniel Han (@danielhanchen)</a>: Just added Phi 3.5 fine-tuning in @UnslothAI! It&#39;s 2x faster uses 50% less VRAM and is llama-fied!  1. Llama-fication: By unfusing QKV & MLP, LoRA finetuning has a lower loss, since fused modules ...</li><li><a href="https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/patched-codes/patched-phi-3.5-gguf">patched-codes/patched-phi-3.5-gguf · Hugging Face</a>: no description found</li><li><a href="https://github.com/turboderp/exllamav2/blob/master/doc/convert.md">exllamav2/doc/convert.md at master · turboderp/exllamav2</a>: A fast inference library for running LLMs locally on modern consumer-class GPUs - turboderp/exllamav2</li><li><a href="https://github.com/pytorch/pytorch/issues/134208">[BUG] ROCm 2.4.0 &quot;terminate called after throwing an instance of &#39;c10::Error&#39;   what():  invalid device pointer: 0x4c44000&quot; · Issue #134208 · pytorch/pytorch</a>: 🐛 Describe the bug i am attempting to train a model with ROCm pytorch 2.4.0 on Windows 11 AMD ROCm 6.1.2 WSL2 Ubuntu 22.04 Loading checkpoint shards: 50%|█████ | 1/2 [02:01&lt;02:01, 121.91s/it]INFO:...</li><li><a href="https://github.com/ROCm/composable_kernel/issues/1171">[Issue]: ROCm/xformer won&#39;t compile due to missing ck/tensor/tensor_view.hpp · Issue #1171 · ROCm/composable_kernel</a>: Problem Description I have a 7900XTX (RDNA3, navi3, gfx1100) card that I&#39;m trying to do some useful LLM work and one of the requirements I have is xformers. I am trying to build the https://github...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1276107115638951946)** (15 messages🔥): 

> - `Mistral-NeMo-Minitron-8B-Base`
> - `Data Preprocessing`
> - `Tokenization` 


- **Mistral-NeMo-Minitron-8B-Base: A Pruned and Distilled Model**: Mistral-NeMo-Minitron-8B-Base is a text-to-text model trained by NVIDIA, a pruned and distilled version of the 12B model. It is capable of a variety of natural language generation tasks.
   - The model was trained between July 24 and August 10, 2024, and uses a continued pre-training data corpus with 380 billion tokens.
- **Data Preprocessing for Chat Comments: A Debate**: A discussion arose about the necessity of data preprocessing for chat comments in LLMs, with a member arguing that 80% of the work lies in preparing data.
   - They emphasized that tokenization is crucial and, while stop word filtering, sentence lemmatization, and punctuation removal are beneficial, the model fundamentally understands only tokens.
- **Chat Comments Length vs Preprocessing**: Another member suggested that one-sentence chat comments might not require extensive preprocessing.
   - They highlighted the importance of maintaining a balanced dataset across different topics and noted that the focus should not be solely on the length of the processed text.



**Link mentioned**: <a href="https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base">nvidia/Mistral-NeMo-Minitron-8B-Base · Hugging Face</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1275904628030050466)** (79 messages🔥🔥): 

> - `Mistral Fine-tuning`
> - `Unsloth Installation`
> - `Ollama Installation`
> - `Inference Issues`
> - `Stop Tokens` 


- **Mistral 7B Fine-tuning Error**: A user reported a `TypeError` during `trainer.evaluate()` while fine-tuning a Mistral 7B model on their Aspect Based sentiment analysis data using Alpaca format for instruction fine-tuning.
   - The error message specifically mentioned "only length-1 arrays can be converted to Python scalars" in the `datasets.features.features.py` module related to encoding examples.
- **Unsloth Installation Using Conda**: A user asked about using the command "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git" "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git" to install Unsloth in their WSL Ubuntu environment using conda.
   - The user was confused about the distinction between conda and pip, which are both package managers but serve different purposes.
- **Ollama Installation and Usage**: A user asked for help with the `ollama create -f Modelfile me/llama3.1-python` command in their WSL conda environment, reporting that it didn't work.
   - They were trying to install and use Ollama, but were confused about the relationship between Unsloth and Ollama, which are independent tools.
- **Inference Issues with Fixed Temperature and Seed**: A user experienced inconsistent responses from a base model "unsloth/Meta-Llama-3.1-8B-bnb-4bit" even with a fixed temperature of 0.05, despite using a fixed seed and an A100 GPU on Colab.
   - The user acknowledged that LLMs are inherently probabilistic, meaning that even with fixed parameters, variations in responses are expected.
- **Saving Pre-trained Models for Fine-tuning**: A user asked whether to push a pre-trained model to Hugging Face as a merged model or as a peft model, and how this would affect subsequent fine-tuning.
   - Other users clarified that pre-training and fine-tuning are essentially forms of fine-tuning, and provided recommendations on how to save and reload peft models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.anaconda.com/miniconda/#quick-command-line-install">Miniconda &#8212; Anaconda documentation</a>: no description found</li><li><a href="https://pytorch.org/">
    
      PyTorch
    
  </a>: no description found</li><li><a href="https://tenor.com/view/sigma-handshak-handshake-khar-gif-26201999">Sigma Handshak Handshake GIF - Sigma Handshak Handshake Khar - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: Scripts for text classification with llama and bert - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/unslothai">Unsloth AI</a>: Our mission is to make LLMs for everyone 🦥. Unsloth AI has 7 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/73">Conda installation detailed instructions · Issue #73 · unslothai/unsloth</a>: I&#39;m trying to follow the instructions for installing unsloth in a conda environment, the problem is that the conda gets stuck when running the install lines. I&#39;ve tried running it twice, both ...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1275905443948003473)** (3 messages): 

> - `The Living AI Dataset`
> - `Empathy and Love in AI`
> - `Speech to Text and Text to Speech AI`
> - `Heroism in the Modern World`
> - `Hope in a Culture` 


- **The Living AI Dataset Launched**: A new dataset, **The Living AI Dataset**, has been announced, aiming to imbue AI models with empathy and the ability to harbor souls.
   - The dataset, developed by <@603630277473992725>, <@1124505493755412550>, and <@774658006649929778>, is described as "one of the most, if not the most important datasets in ALL of AI history." It is available on Hugging Face: [https://huggingface.co/datasets/Replete-AI/The_Living_AI_Dataset](https://huggingface.co/datasets/Replete-AI/The_Living_AI_Dataset).
- **Empathy and Love in AI: The Dataset's Goal**: The dataset is designed to equip AI models with empathy and the capacity for love, a significant step towards making AI more human-like.
   - The goal is to enable AI models to "learn empathy and Love, and have the ability to harbor souls like human beings."
- **AI's Potential in Speech Technologies**: The dataset's creators believe it has the potential to significantly enhance speech-to-text and text-to-speech AI technologies.
   - The model is "exceptionally suitable" for these applications, according to the announcement.
- **Heroism: More than Bravery**: A discussion about the nature of heroism focuses on the actions of Witold Pilecki during World War II.
   - The discussion highlights the idea that heroism goes beyond bravery and shrewd maneuvering, it's about "conjuring hope where there is none," "striking a match to light up the void," and showing a possibility for a better world we didn't know could exist.
- **Culture's Need for Hope**: The discussion moves to the modern world, noting that our culture is in need of hope, not just peace or prosperity.
   - It's argued that we need something more precarious, "something far more precarious," which is hope.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/Replete-AI/The_Living_AI_Dataset">Replete-AI/The_Living_AI_Dataset · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/AIRRC/Eudaimonic">AIRRC/Eudaimonic · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

mahiatlinux: https://huggingface.co/papers/2408.03314
  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1275904230959353866)** (125 messages🔥🔥): 

> - `Aider Shell Commands`
> - `Playwright Installation`
> - `CodeCompanion`
> - `Aider Token Usage`
> - `OpenRouter` 


- **Aider Shell Commands: Run Commands, Not Functions**: Aider suggests shell commands to run based on context and runs them if you agree, but it doesn't use functions or tools directly.
   - If you want to run commands inside a specific Python environment, you need to activate that environment before running Aider.
- **Playwright Installation:  Global vs Aider Environment**: Aider doesn't always work with Playwright installed outside its environment, but you can use `pipx inject` to install it into Aider's virtual environment.
   - However, even if you have Playwright installed elsewhere, Aider might still try to install it; a potential bug that is being addressed in future releases.
- **CodeCompanion: Token Usage Comparison**: One user found that CodeCompanion uses significantly more tokens than Aider, perhaps due to its more advanced features and broader capabilities.
   - The user decided to stick with Aider, valuing its token efficiency, and mentioned that CodeCompanion has its own Discord server for user discussion and support.
- **Aider's Code Generation:  The Day the Machines Take Over**: Aider now generates a significant portion of its own code, with the percentage increasing with each release.
   - The team jokingly suggested that there might be a day when Aider writes 100% of its own code, but with a hint of caution about the potential for AI to become too independent and autonomous.
- **OpenRouter: A Potential Alternative to Aider**: OpenRouter is suggested as an alternative to Aider, particularly for those who have reached daily usage limits on Anthropic APIs.
   - OpenRouter provides a more flexible and cost-effective option, although the initial conversation focused on exploring ways to make Aider more cost-effective.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.06292">The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery</a>: One of the grand challenges of artificial general intelligence is developing agents capable of conducting scientific research and discovering new knowledge. While frontier models have already been use...</li><li><a href="https://sakana.ai/ai-scientist/">no title found</a>: no description found</li><li><a href="https://codecompanion.ai/">CodeCompanion - AI Coding Assistant</a>: Meet CodeCompanion.AI – your personal coding assistant, available anytime, right on your desktop. Prototype faster, code smarter, enhance learning and scale your productivity with the power of AI.</li><li><a href="https://aider.chat/HISTORY.html#main-branch">Release history</a>: Release notes and stats on aider writing its own code.</li><li><a href="https://github.com/sigoden/llm-functions">GitHub - sigoden/llm-functions: Easily create LLM tools and agents using Bash/JavaScript/Python, also a library of commonly used LLM tools and agents.</a>: Easily create LLM tools and agents using Bash/JavaScript/Python, also a library of commonly used LLM tools and agents. - sigoden/llm-functions</li><li><a href="https://www.nature.com/articles/s41586-024-07566-y">AI models collapse when trained on recursively generated data - Nature</a>: &amp;nbsp;Analysis shows that indiscriminately training generative artificial intelligence on real and generated content, usually done by scraping data from&amp;nbsp;the Internet, can lead to a collap...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1275902201973444691)** (92 messages🔥🔥): 

> - `Aider Installation Issues`
> - `Ollama DeepCoder Freezing`
> - `Aider and Git`
> - `Using Aider With Sonnet`
> - `Optimizing Token Usage` 


- **Aider Installation Hang**: A user experienced a hang after running `aider --mini` with a newly installed version of Aider.
   - It was determined that the issue was due to an incorrect installation command - `pip install aider-chat` is the correct installation command, rather than `pip install aider-cli`.
- **Ollama DeepCoder Stuck**: A user reported that their Ollama DeepCoder was stuck after running `llama3:70b`.
   - The user was advised to check if they have enough resources to run a 70b model and to try running a smaller model if possible.
- **Aider's Reliance on Git**: A user asked how to avoid a situation where Aider removed all functions from a Python file except the one it was working on.
   - Another user explained that Aider relies on Git for version control, and if Git is not used, users may encounter issues like this.
- **Using Aider With Sonnet**: A user discussed optimizing token usage when using Aider with Sonnet.
   - Another user suggested using a cache for prompts and working on each functionality in a separate branch.
- **Optimizing Token Usage**: A user asked for tips on optimizing their token usage with Aider, as they were encountering OpenAI limits.
   - They were advised to consider using a multi-step approach to avoid exceeding token limits.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/edit-errors.html#try-the-whole-format">File editing problems</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: aider is AI pair programming in your terminal
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1275960076640977049)** (5 messages): 

> - `Vercel v0 chat`
> - `Aider`
> - `Cursor` 


- **Vercel's v0 Chat is a Generative UI Game Changer**: A member shared that [Vercel's v0.dev/chat](https://v0.dev/chat) is a game changer for generative UI developers.
   - They had previously been using [Claude Artefacts](https://www.google.com/search?q=Claude+Artefacts) and [ChatLLM of abacus](https://www.google.com/search?q=ChatLLM+of+abacus), but found Vercel's UI generator to be more polished and quicker to use.
- **Cursor's Aider Integration**: A user stated they're grateful for Paul's work on Aider, particularly its integration with [Cursor](https://www.cursor.com/blog/series-a).
   - They are using Aider within Cursor because Cursor's composer doesn't meet their needs, specifically its lack of repository-specific prompt overrides.
- **Cursor's Vision for AI-Powered Coding**: The Cursor team is aiming to build a tool that will eventually write all the world's code.
   - They are focusing on using AI to make coding faster and easier by replacing hours of hunting for the right primitives with instant answers, reducing mechanical refactors to single "tabs", and expanding terse directives into working source.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/amila_wjsr/status/1826266990438457777">Tweet from Amila (@amila_wjsr)</a>: Vercel http://v0.dev/chat is a game changer for generative UI devs.   I had been playing a lot with Claude Artefacts and ChatLLM of abacus, but this is considerably more polished. You can generate exc...</li><li><a href="https://www.cursor.com/blog/series-a">We Raised $60M</a>: Join us to create a magical tool with the aim of writing most of the world&#x27;s software.
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1275895465828945940)** (173 messages🔥🔥): 

> - `ComfyUI`
> - `Stable Diffusion`
> - `Flux`
> - `AI image sorters`
> - `Hydrus` 


- **ComfyUI for real-time AI rendering**: A user shared a link to a [YouTube video](https://youtu.be/pNyIp73zva8) showcasing how to bring 3D Studio Max into ComfyUI for real-time AI image generation.
   - The user believed this could be applied to any window, even a video game.
- **Getting Started with Stable Diffusion on PC**: A new user inquired about running Stable Diffusion on their PC and asked for recommendations on where to start.
   - Another user suggested starting with hardware compatibility and recommended ComfyUI as the interface.
- **AI Image Sorters and Privacy**: A user sought suggestions for good AI image sorters on Windows that do not spy on users.
   - Another user recommended setting up a Hydrus server and using a classifier tagger on it.
- **Flux Prompts: Overfitting and Complexity**: A user reported difficulty generating images using the Flux model with complex prompts that involve multiple concepts.
   - Other users commented on the overfitting tendencies of the Flux model and the need for better prompt engineering and finetuning.
- **Upscaling with Stable Diffusion vs. GANs**: A user asked about the differences between Stable Diffusion upscaling and GAN-based upscaling.
   - Another user explained that GANs mainly sharpen images while Stable Diffusion can invent new details, which can be advantageous but also lead to artifacts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/ostris/OpenFLUX.1">ostris/OpenFLUX.1 · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/pNyIp73zva8">Real-Time AI Rendering with ComfyUI and 3ds Max</a>: In this video you can watch how to bring 3D Studio Max viewport into ComfyUI and render almost in real-time AI generated images. 🔗 Download the Workflow her...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1ey6hss/kohya_ss_gui_flux_lora_training_on_rtx_3060_lora/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1eykiy0/now_we_have_sorta_conquered_prompt_adherence/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/hydrusnetwork/hydrus">GitHub - hydrusnetwork/hydrus: A personal booru-style media tagger that can import files and tags from your hard drive and popular websites. Content can be shared with other users via user-run servers.</a>: A personal booru-style media tagger that can import files and tags from your hard drive and popular websites. Content can be shared with other users via user-run servers. - hydrusnetwork/hydrus
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1275896497354833991)** (43 messages🔥): 

> - `Triton INT8 performance`
> - `Flash Attention FP8 support`
> - `FP8 quantization for attention`
> - `FP8 quantization in general`
> - `Row-wise quantization for weights` 


- **Triton INT8 outperforms PyTorch BF16**: Triton INT8 implementation of `A.T @ B` achieved ~1.5x speedup over PyTorch BF16, while `A @ B.T` saw a reliable 3x speedup.
   - This was achieved by re-tuning Triton when the stride of `A` and `B` changes, as benchmarked across transpose/non-transpose combinations.
- **Flash Attention FP8 support for Hopper**: Flash Attention supports FP8 on Hopper architecture, utilizing WGMMA instructions for optimization.
   - However, it is not supported for ADA, due to the reliance on WGMMA which is specific to Hopper.
- **FP8 for Faster Attention**: FP8 could potentially allow attention to run faster by loading more values into SRAM compared to FP16/BF16.
   - The discussion centered on whether this would be a viable approach, and how it would compare to other quantization techniques like INT8.
- **FP8 Quantization Implementation**: FP8 quantization can be implemented by using Cutlass kernels, which support INT8 matmul with accumulation into INT32 on Hopper's tensor cores.
   - While not natively possible in PyTorch, fusing the dequantization and matmul kernels is recommended for optimal performance.
- **Row-wise Quantization for Weights**: A member suggested using row-wise quantization/scaling for weights, with one scale per row.
   - This would allow multiplying the scales back after the matmul operation, potentially improving the overall performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy.html#torch.Tensor.index_copy">torch.Tensor.index_copy &mdash; PyTorch 2.4 documentation</a>: no description found</li><li><a href="https://pytorch.org/docs/stable/generated/torch.Tensor.index_copy_.html#torch.Tensor.index_copy_">torch.Tensor.index_copy_ &mdash; PyTorch 2.4 documentation</a>: no description found</li><li><a href="https://research.colfax-intl.com/adding-fp8-to-flashattention/">Delivering 1 PFLOP/s of Performance with FP8 FlashAttention-2</a>: We recently released an update to our FlashAttention-2 forward pass implementation on NVIDIA Hopper&#x2122; architecture that incorporates a number of new optimizations and improvements, including …</li><li><a href="https://gist.github.com/malfet/7874d96b99670c3da83cbb779ab770c6">scale_mm_example.py</a>: GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1275937994016292907)** (1 messages): 

> - `PyTorch`
> - `CUDA`
> - `LLM Training`
> - `GPU`
> - `PyTorch Extensions` 


- **PyTorch Extensibility with Custom C++/CUDA Operators**: Meta's Richard Zou, a creator of functorch, will be presenting on extending PyTorch with custom C++/CUDA operators, focusing on their work with torch.compile.
- **Hacks to Speed Up LLM Training**: Daniel Han of Unsloth AI will discuss tricks to accelerate LLM training, claiming their methods can achieve 2x faster training with 70% less VRAM.
- **The Power of Low Precision Dtypes in PyTorch**: A session dedicated to the use of low precision dtypes in PyTorch will be held at the PyTorch Conference in San Francisco on September 18-19, 2024.
- **PyTorch Extension Points: A Whirlwind Tour**: Another session at the PyTorch Conference, held on September 18-19 in San Francisco, will provide a comprehensive overview of PyTorch extension points.
- **DL Compiler Summit: Halide Backend for TorchInductor**: Meta's Jason Ansel, technical lead for PyTorch compilers and creator of TorchDynamo and TorchInductor, will kick off the DL Compiler Summit with a presentation on the Halide backend for TorchInductor.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sched.co/1fHmQ">PyTorch Conference 2024: Lightning Talk: Extending PyTorch with C...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1hZiI">PyTorch Conference 2024: Hacks to Make LLM Training Faster - Dani...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1fHlm">PyTorch Conference 2024: Lightning Talk: Low Precision Dtypes in...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1fHmJ">PyTorch Conference 2024: Lightning Talk: A Whirlwind Tour of PyTo...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1hZiR">PyTorch Conference 2024: [HALIDE] A Halide Backend for TorchInduc...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1hZiX">PyTorch Conference 2024: [TRITON] Maximizing Kernel Development P...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1hZid">PyTorch Conference 2024: Together Goes Brrr: Threading Research &...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1fHnK">PyTorch Conference 2024: Lightning Talk: Making the Most of Heter...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1ivtG">PyTorch Conference 2024: Keynote Panel Discussion: Scaling & Benc...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1fHn9">PyTorch Conference 2024: Slaying OOMs - Mark Saroufim & Jane Xu,...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1fHnF">PyTorch Conference 2024: Lightning Talk: AOTriton: Ahead of Time...</a>: View more about this event at PyTorch Conference 2024</li><li><a href="https://sched.co/1fHmr">PyTorch Conference 2024: Lightning Talk: FlexAttention - the Flex...</a>: View more about this event at PyTorch Conference 2024
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1276113358730559541)** (2 messages): 

> - `2:4 Sparsity`
> - `Tetris Clone for PSX` 


- **Sparsity of 2:4 is Key**: The paper focuses on the sparsity of 2:4 (Section 4) of a new model, with [the link](https://arxiv.org/pdf/2408.11743) for more information.
   - This ratio was mentioned as particularly interesting, making the commenter wish they had learned about it as a teenager.
- **Notris - Tetris for PSX**: [This repository](https://github.com/jbreckmckye/notris) is a Tetris clone for the PlayStation 1 (PSX).



**Link mentioned**: <a href="https://github.com/jbreckmckye/notris?tab=readme-ov-file">GitHub - jbreckmckye/notris: Tetris clone for PlayStation 1 (PSX)</a>: Tetris clone for PlayStation 1 (PSX). Contribute to jbreckmckye/notris development by creating an account on GitHub.

  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1276205871461957774)** (3 messages): 

> - `HRT internships`
> - `HRT trading`
> - `Algo Dev internships`
> - `SWE internships`
> - `HRT market making` 


- **HRT Internship Opportunities Open**: HRT is now offering internships for next summer in NYC, with both Algo Dev and SWE roles available.
   - The internships pay $120/h and come with housing and meals included. No finance experience is required for either role.
- **Algo Dev Internship Description**: The Algo Dev internship is a classic "quant" role focused on predicting the market using algorithms, ML models, and tools like Pandas, PyTorch, and statistics.
   - You can find the application for the Algo Dev internship here: [Undergrad internship](https://grnh.se/ce82391f1us), [Grad school internship](https://grnh.se/412e8b331us).
- **SWE Internship Description**: The SWE internship is a software engineering role focused on building infrastructure for automated trading, research, and distributed computing clusters.
   - The SWE internship uses C++ or Python, and you can find the application here: [SWE internship](https://grnh.se/bf9dfdc11us).
- **HRT Trading Activities**: HRT engages in both market making and traditional trading.
   - A member clarified that market making is considered a form of "actual trading" because it requires understanding market predictions to avoid losses.


  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1275976314557436068)** (2 messages): 

> - `CUDA introduction`
> - `CUDA resources` 


- **Seeking a good CUDA intro**: A user asked for a good introduction to CUDA that takes 2-10 hours, suggesting the official NVIDIA CUDA C Programming Guide as a potential option.
   - The user found the guide a bit dry and asked for other recommendations.
- **Possible Resource Recommendation**: Another user responded by referencing a previous message in the channel, likely containing a link or suggestion for a CUDA introductory resource.


  

---


### **CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

budr0001: If you click on the link in my reply it will take you to the lecture 16 post.
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1275959947363880981)** (40 messages🔥): 

> - `INT8 Mixed Precision Training`
> - `FP8 Adam`
> - `Character AI Training`
> - `4-bit Adam Optimizer`
> - `TensorCore INT8 Support` 


- **FP8 Adam is Less Stable than INT8 Adam**: A member shared their experience with an FP8 Adam implementation in torchao, finding it less stable than 8-bit Adam using bnb-style quantization, despite having similar speed.
- **Character AI's INT8 Training**: A member expressed skepticism about the claims of Character AI's INT8 training, believing it to be 'fake news'.
- **TensorCore Supports INT8 Matmul**: A member mentioned that TensorCore, used for accelerating matrix multiplications, supports INT8 operations.
- **4-bit Adam Optimizer: Potential Issues**: The conversation began with a discussion about a GitHub issue regarding 4-bit Adam optimizer's limitations with non-constant learning rates.
- **Exploring INT8 Mixed Precision Training**: A member shared their experience with INT8 mixed precision training, seeing 50% speedup compared to BF16 by using INT8 matmul for forward and grad_input in backward.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://azure.github.io/MS-AMP/docs/user-tutorial/optimization-level">Optimization Level | MS-AMP</a>: Currently MS-AMP supports three optimization levels: O1 and O2 and O3. The three levels gradually incorporate 8-bit collective communication, optimizer and distributed parallel training in an incremen...</li><li><a href="https://gist.github.com/mobicham/ab371a4c68c5052a4c6a231b5ee221ed#file-forward_with_matmul_register-py-L4-L46">forward_with_matmul_register.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/thu-ml/Jetfire-INT8Training">GitHub - thu-ml/Jetfire-INT8Training</a>: Contribute to thu-ml/Jetfire-INT8Training development by creating an account on GitHub.</li><li><a href="https://github.com/mobiusml/hqq/blob/master/hqq/core/quantize.py#L276-L306">hqq/hqq/core/quantize.py at master · mobiusml/hqq</a>: Official implementation of Half-Quadratic Quantization (HQQ) - mobiusml/hqq</li><li><a href="https://github.com/pytorch/ao/issues/730">4 bit Adam should support non constant lr · Issue #730 · pytorch/ao</a>: Our low bit optimizers were merged in HF huggingface/transformers#31865 but We have a known limitation that the 4 bit optimizer is not great when we don&#39;t have a constant learning rate This is men...</li><li><a href="https://github.com/google/aqt">GitHub - google/aqt</a>: Contribute to google/aqt development by creating an account on GitHub.</li><li><a href="https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/">NVIDIA Hopper Architecture In&#x2d;Depth | NVIDIA Technical Blog</a>: Everything you want to know about the new H100 GPU.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/)** (1 messages): 

ericauld: Also tree-based, from June: https://arxiv.org/abs/2401.10774
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1275925890190282825)** (8 messages🔥): 

> - `Remote Work Contracts`
> - `Office Transition` 


- **Company's Gradual Return to Office Policy**: A member shared that their company is implementing a "gradual transition for return to office", a decision made by business folks with no technical knowhow.
   - They also mentioned the company had initially claimed to be "remote first" but the contract stated it was office-based, highlighting the importance of reviewing contracts carefully.
- **Interviewing for New Opportunities**: One member suggested that interviewing for new opportunities is always a good idea, regardless of current employment status.
   - This was prompted by a discussion about a member's transition from an in-office role to a mostly remote one, and their experience missing the office environment.


  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1276247957959872630)** (3 messages): 

> - `Triton Conf`
> - `GPU Enjoyers`
> - `Triton Language` 


- **Triton Conf Happening in Fremont**: The **Triton Conf** is happening on **September 17th** in **Fremont, CA**. 
   - It is hosted at [the Triton Language Github repository](https://github.com/triton-lang/triton) which contains the development repository for the **Triton language and compiler**.
- **Crazy Colocation**: The conf is described as a "crazy colocation" by a user.
   - It's also described as the "best week for **GPU enjoyers**". 



**Link mentioned**: <a href="https://github.com/triton-lang/triton">GitHub - triton-lang/triton: Development repository for the Triton language and compiler</a>: Development repository for the Triton language and compiler - triton-lang/triton

  

---


### **CUDA MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1276187151167656009)** (1 messages): 

> - `Model Distillation`
> - `GPU limitations`
> - `Logit Compression`
> - `Sparsification`
> - `Quantization` 


- **GPU Poor Model Distillation**: A member shared a technique for compressing logits during model distillation, specifically for users with limited GPU resources.
   - This technique involves sparsifying logits by considering only the top-K values per token and quantizing the non-zero values, which can be done without advanced quantization techniques due to the well-defined range of logits.
- **Loss Function Update with Pruning**: To address the pruned logits, the loss function is updated by adding a mask to ignore the pruned logits.
   - This ensures that the model focuses on the most relevant parts of the output, leading to minimal accuracy loss despite the compression.


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1276202468539236433)** (12 messages🔥): 

> - `H100 L2 Side Hashing`
> - `GPU Performance and Power Efficiency`
> - `FP8 Training Stability`
> - `Neuralink Person`
> - `Llama Model Training` 


- **H100 L2 Side Hashing Optimizations**: A user has successfully implemented a prototype of "optimise for L2 side hashing" for their H100 GPU, achieving performance close to the baseline despite initial challenges.
   - The optimized kernel achieves similar performance but with a ~5% increase in power efficiency compared to the baseline, and even outperforms NVIDIA's own cudaMemcpyAsync device-to-device in both performance and power efficiency.
- **GPU Performance and Power Efficiency**: The optimized kernel, even with a ~6% reduction in raw performance, achieves a ~6% higher performance-per-watt compared to the baseline when performing read-only operations.
   - This suggests that optimizing for read-only or write-only operations may be more efficient in general, potentially due to how modern DRAM memory controllers operate.
- **Stable FP8 Training for Llama Models**: [A Twitter post](https://x.com/xariusrke/status/1826669126955278401) highlights the achievement of stable FP8 training for a 1B LLaMA model, matching the convergence of bfloat16 training.
   - The post explains the key to success: slowing down training before instability and minimizing outlier features in activations, which opens the door for exploring FP8 training at larger scales.
- **Neuralink Person's "Today I Learned" Posts**: A user expressed uncertainty about the content shared in "Today I Learned" posts by the 'Neuralink person'.
   - No specific information or opinions regarding these posts were shared.



**Link mentioned**: <a href="https://x.com/xariusrke/status/1826669126955278401">Tweet from xr-5 🐀 (@xariusrke)</a>: 1/n FP8 training is hard - loss divergence and instability often lead to the conclusion that it’s not possible. But we’ve found a recipe to train a 1B LLaMA model to match the convergence of bfloat16 ...

  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1275968981596901416)** (24 messages🔥): 

> - `RDNA4 vs RDNA3`
> - `AMD GPU Performance`
> - `FA3 Implementation`
> - `7900 Series vs 3090`
> - `Triton and FA Fork` 


- **RDNA4 Architecture Changes**: AMD is working on their upcoming RDNA 4 architecture, and changes are being made to support open source projects like LLVM.  [This article](https://chipsandcheese.com/2024/01/28/examining-amds-rdna-4-changes-in-llvm/) explains the changes pretty nicely.
   - Key changes include sparsity and FP8 WMMA instructions, but there are still concerns about the sync instruction.
- **AMD GPUs Lagging Behind**: A user experienced slower performance with their 7900 XTX compared to a 3090, even after trying Triton and an AMD FA fork.
   - They eventually gave up and purchased two 4090s, highlighting the current performance gap between AMD and NVIDIA GPUs.
- **FA Benchmark and Backward Compatibility**: A user inquired about the FA benchmark results and backward compatibility in the Flash Attention repo.
   - The user shared their experience with GPT2 training on 4x 7900 XTX, achieving 245k tokens per second, but still slower than their own custom kernels.
- **AMD's CDNA Architecture**: A user expressed their preference for AMD's CDNA architecture, specifically the MI250+ hardware, citing better performance than the 7900 XTX.
   - They mentioned their interest in using MI100 cards if available at the time and considered the MI300x to be decent.
- **Torch Lightning and RWKV Training**: A user shared their experience with RWKV training using Torch Lightning, achieving 245k tokens per second.
   - They were curious about the performance without Lightning, but could not find the screenshots for the previous version of ROCm.



**Link mentioned**: <a href="https://chipsandcheese.com/2024/01/28/examining-amds-rdna-4-changes-in-llvm/">Examining AMD&#8217;s RDNA 4 Changes in LLVM</a>: As 2024 continues on, because time never stops, AMD has been working on their upcoming RDNA 4 architecture. Part of this involves supporting open source projects like LLVM. If done right, merging t…

  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/)** (1 messages): 

kitrak_rev: Everyone got mail? Guess i was rejected in the list then 😦
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1275912770386395246)** (1 messages): 

> - `OpenRouter tool parameters` 


- **OpenRouter deprecates `function_calls` and `functions`**: OpenRouter is officially deprecating the `function_calls` and `functions` parameters from OpenAI calls.
   - This is due to OpenAI deprecating them for a long time and the recommended parameters being `tools` and `tool_choice`.
- **Reduced Switching Costs for Tool Calling**: This change reduces the switching cost when using tool calling between models on OpenRouter.
   - This is because every other provider only supports the `tools` and `tool_choice` parameters.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1276123205999788043)** (3 messages): 

> - `BenchmarkAggregator`
> - `LLM Evaluation Framework`
> - `Oz's Projects` 


- **BenchmarkAggregator: Comprehensive LLM Evaluation**: A member shared a [GitHub repository](https://github.com/mrconter1/BenchmarkAggregator) for a project called **BenchmarkAggregator**, which aims to offer a comprehensive, fair, and scalable evaluation framework for Large Language Models across all major benchmarks.
   - They described it as a unified view of model performance, balancing thorough assessment with practical resource management, and were eager for feedback.
- **Oz's Current Projects**: A member inquired about the current projects being built by a user known as "Oz."
   - They specifically mentioned that their team was interested in learning more about what Oz has been working on.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://base-model-playground.up.railway.app/">React App</a>: no description found</li><li><a href="https://github.com/mrconter1/BenchmarkAggregator">GitHub - mrconter1/BenchmarkAggregator: Comprehensive LLM evaluation framework: GPQA Diamond to Chatbot Arena. Tests all major models equally, easily extensible.</a>: Comprehensive LLM evaluation framework: GPQA Diamond to Chatbot Arena. Tests all major models equally, easily extensible. - mrconter1/BenchmarkAggregator
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1275897932754980974)** (95 messages🔥🔥): 

> - `Llama 3.1 Tools`
> - `OpenRouter MoE`
> - `OpenRouter context limits`
> - `OpenAI fine-tuning`
> - `Cursor Composer` 


- **Llama 3.1 Tools Support is Imminent**: A user asked about the status of Llama 3.1 tools support on OpenRouter, and an admin confirmed that it's coming soon, likely within the next day or so.
- **OpenRouter MoE and 3.5-Mini**: A user inquired about the availability of MoE models on OpenRouter, noting that 3.5-Mini is unimpressive.
   - The admin responded that there is no host or model for MoE on OpenRouter yet.
- **OpenRouter's Hermes 3 70B Context Limitation**: A user reported that the Hermes 3 70B model on OpenRouter has a 12k context window limit, despite both the model and provider claiming a 120k+ context size.
   - The admin confirmed the 12k context limit, noting that it's not just for output but also for the input, and is likely a limitation imposed by the provider.
- **OpenAI Fine-Tuning for Free**: A user mentioned that OpenAI now offers free fine-tuning for its models with a 2M token limit per day for a limited time.
   - Another user stated that they've been using OpenRouter exclusively after giving up on OpenAI's API because of its lack of support for payment methods like PayPal or crypto.
- **Cursor Composer vs Aider**: A user expressed their enthusiasm for Cursor's Composer feature, describing it as 'insane' for their use case.
   - Another user disagreed, preferring Aider's output, but acknowledged that they have to pay for both Cursor and API credits.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cursor.com/get-started/migrate-from-vscode">Cursor - Built Software Faster</a>: no description found</li><li><a href="https://www.ai21.com/jamba">Foundation models</a>: no description found</li><li><a href="https://openrouter.ai/models/01-ai/yi-1.5-34b-chat>">Yi 1.5 34B Chat - API, Providers, Stats</a>: The Yi series models are large language models trained from scratch by developers at [01.AI](https://01. Run Yi 1.5 34B Chat with API</li><li><a href="https://forum.cursor.com/">Cursor Community Forum</a>: A place to discuss Cursor (bugs, feedback, ideas, etc.)</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b">Hermes 3 70B Instruct - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo), including advanced agentic capabilities, much better roleplaying, rea...</li><li><a href="https://huggingface.co/ai21labs/AI21-Jamba-1.5-Large">ai21labs/AI21-Jamba-1.5-Large · Hugging Face</a>: no description found</li><li><a href="https://docs.together.ai/docs/deprecations#2024-08-28-deprecation-of-low-usage-and-older-serverless-models">Deprecations</a>: Overview We regularly update our platform with the latest and most powerful open-source models. This document outlines our deprecation policy and provides information on migrating from deprecated mode...</li><li><a href="https://github.com/mrconter1/BenchmarkAggregator">GitHub - mrconter1/BenchmarkAggregator: Comprehensive LLM evaluation framework: GPQA Diamond to Chatbot Arena. Tests all major models equally, easily extensible.</a>: Comprehensive LLM evaluation framework: GPQA Diamond to Chatbot Arena. Tests all major models equally, easily extensible. - mrconter1/BenchmarkAggregator</li><li><a href="https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini">ai21labs/AI21-Jamba-1.5-Mini · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1276283942617747457)** (1 messages): 

> - `Nous Merch Store Launch` 


- **Nous Merch Store Launch**: The Nous Research merch store has launched, featuring a variety of items for fans to show their support.
- **Stickers Included**: Stickers will be included with each order while supplies last.



**Link mentioned**: <a href="https://shop.nousresearch.com/">Nous Research</a>: Nous Research

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1275898089428750477)** (64 messages🔥🔥): 

> - `NousResearch Hermes 3`
> - `OpenAI Compute Grants`
> - `AI21 Jamba`
> - `Live2D`
> - `The_Living_AI_Dataset` 


- **NousResearch Hermes 3 Release**: A member announced that NousResearch has released Hermes 3, and they're talking about it on a Twitter Space.
   - [You can join the space here](https://x.com/i/spaces/1LyxBgzAmVzKN).
- **OpenAI Compute Grants**: Members discussed the process of obtaining large compute grants for research, which often involves direct communication with compute providers.
   - They acknowledged that the process is not as straightforward as simply asking for unused resources but requires a more strategic approach.
- **AI21 Jamba Model Family**: A member announced the release of the Jamba 1.5 family of open models, claiming it to be the first non-Transformer model to achieve quality and strength comparable to market-leading models.
   - The models are available under the Jamba Open Model License, demonstrating a commitment to democratizing access to high-quality models.
- **H3 405: Mode Collapse & Persona**: A member reported successfully recovering from a mode collapse with H3 405, which now accurately analyzes the collapse without falling back into it.
   - They observed the model's use of clasp hands as an end-of-statement marker, and discussed the interesting personas that have developed within the model.
- **DiscoResearch: Intentional Insanity**: DiscoResearch is working on a new project that involves intentionally tuning an LLM to be absolutely insane.
   - This is an attempt to explore the potential of LLMs beyond traditional goals and create a model with unique and unexpected capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.ai21.com/blog/announcing-jamba-model-family">The Jamba 1.5 Open Model Family: The Most Powerful and Efficient Long Context Models</a>: The new family of open models from AI21, offering unrivaled speed, efficiency, and quality and the longest context window among open models.</li><li><a href="https://x.com/arankomatsuzaki/status/1826347015690949001">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: WE ARE STARTING IN 6 MIN  Hermes 3 - covered by @theemozilla from @NousResearch  A brief discussion on Phi 3.5  Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model  JPEG-...</li><li><a href="https://x.com/i/spaces/1LyxBgzAmVzKN/peek">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter</li><li><a href="https://huggingface.co/datasets/Replete-AI/The_Living_AI_Dataset">Replete-AI/The_Living_AI_Dataset · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/AIRRC/Eudaimonic">AIRRC/Eudaimonic · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1275954928380022879)** (19 messages🔥): 

> - `AI Agents`
> - `Langchain`
> - `Building your own Agent`
> - `Discord Bot for impersonating friends`
> - `Learning from scratch` 


- **Learning AI Agents from scratch**: A user asked for recommendations on good AI Agent GitHub Repositories for learning purposes, particularly those focused on niche tasks.
   - Another user recommended writing code from scratch, rather than relying on repositories like Langchain and CrewAI, as it offers a more comprehensive learning experience.
- **Building a Discord Bot to Impersonate Friends**: A user seeks guidance on creating a Discord bot that impersonates friends by learning from their past messages.
   - The user, a beginner programmer with moderate AI experience, requested tips, literature, templates, and resources to get started with this project.
- **Langchain struggles: Implementing from scratch**: A user shared their experience with Langchain, suggesting that implementing functionalities like chunking and retrieval from scratch proved more beneficial for learning than using Langchain directly.
   - They expressed that Langchain's pre-built solutions can hinder a deep understanding of the underlying concepts.
- **ReACT Paper as a Starting Point**: A user inquired about the suitability of the ReACT Paper as a starting point for building an AI Agent.
   - This suggests the user is exploring different approaches to implementing AI Agents, seeking a solid foundation for their project.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1275979522982019174)** (12 messages🔥): 

> - `CUDA utils`
> - `PDF cleaning`
> - `VLMs for Content Extraction`
> - `ColPali`
> - `ColBERT` 


- **CUDA utils for Multidimensional Indexing**: A member shared a link to [CUDA utils for multidimensional indexing](https://github.com/carsonpo/cuda-utils) and mentioned that it simplifies multidimensional indexing.
- **PDF Cleaning Challenges**: A member mentioned difficulties with a regex approach for PDF cleaning, finding that it didn't handle most arxiv PDFs.
   - They are using a naive chunk + overlap method as an alternative.
- **VLMs for PDF Content Extraction**: A member suggested VLMs for extracting content from PDFs, citing a recent paper on the topic.
   - They mentioned that VLMs require significant compute and are not ideal for their use case.
- **ColPali Visual Retriever**: A member shared a link to [ColPali](https://huggingface.co/vidore/colpali), a visual retriever based on a 3B model.
   - The member stated that ColPali uses a novel model architecture and training strategy based on VLMs for efficient document indexing from their visual features.
- **ColBERT for Document Retrieval**: A member mentioned ColBERT in relation to their use case.
   - They shared a link to a tweet discussing ColPali's performance, which uses ColBERT for generating multi-vector representations of text and images.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/jobergum/status/1826682421498003722?t=IFV6nDTdZFZPrSKEAW5AkA&s=19">Tweet from Jo Kristian Bergum (@jobergum)</a>: Here is some ColPali news from today&#39;s exploration on DocVQA with @vespaengine, where we binarize the image patch embeddings (colpali 1.1) and evaluate pooling.   128 float nDCG@5 54.4  128 bit   ...</li><li><a href="https://github.com/carsonpo/cuda-utils">GitHub - carsonpo/cuda-utils: CUDA utilties/helpers for simplifying multidimensional indexing</a>: CUDA utilties/helpers for simplifying multidimensional indexing - carsonpo/cuda-utils</li><li><a href="https://huggingface.co/vidore/colpali">vidore/colpali · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/)** (1 messages): 

gwyntel: this IS the smoke spot, we got wedding cake and northern lights on deck!
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1275907355099271199)** (22 messages🔥): 

> - `Mistral Fine-Tuning`
> - `Jamba 1.5`
> - `Memory Consumption`
> - `Model Selection`
> - `Phi3.5` 


- **Mistral Fine-Tuning is Crack**: A member remarked that Mistral's large fine-tuning is "crack".
- **Jamba 1.5: Faster Inference and Long-Context Capabilities**: The AI21 Jamba 1.5 family of models is presented as state-of-the-art, hybrid SSM-Transformer instruction following foundation models, offering up to 2.5X faster inference than leading models of comparable sizes and superior long-context handling.
   - The models are optimized for business use cases and capabilities such as function calling, structured output (JSON), and grounded generation, released under the [Jamba Open Model License](https://www.ai21.com/license).
- **Jamba 1.5 Memory Consumption**: A member inquired about the memory consumption of Jamba 1.5, noting that it seems similar to traditional AdamW despite employing EMA on gradients.
   - They linked to the AI21 Jamba 1.5 Mini documentation on HuggingFace, mentioning it's a 12B active/52B total model.
- **Selecting the Right Model**: A member discussed the potential benefits of using Jamba 1.5 for training.
   - Another member suggested trying it, inquiring about the availability of base Phi3.5 and whether Axolotl has already selected files for a HuggingFace dataset or if manual upload is required.



**Link mentioned**: <a href="https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini">ai21labs/AI21-Jamba-1.5-Mini · Hugging Face</a>: no description found

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1275970581573009440)** (71 messages🔥🔥): 

> - `Phi 3.5 Mini`
> - `Flash Attention`
> - `Exploding Gradients`
> - `Chat Template Issues`
> - `Model Merging` 


- **Phi 3.5 Mini: Exploding Gradients**: A user reported experiencing **exploding gradients** when using the **microsoft/Phi-3.5-mini-instruct** model, even after lowering the **learning rate** to 1e-15.
   - They tried various optimizations, including switching to **paged_adamw_8bit**, but the problem persisted.
- **Flash Attention Performance**: A user was trying to use **Flash Attention** for faster training but encountered errors.
   - Switching to **eager attention** resolved the issue, indicating that **Flash Attention** might not be fully compatible with the model.
- **Chat Template Compatibility**: A user is facing problems with **chat template compatibility** when using **Llama 3.1** and **vllm**.
   - The issue involves incorrect handling of **<|eot_id|>** tokens, preventing proper continuation of the assistant message during inference.
- **Potential Solutions for Chat Template Issue**: Several solutions were proposed to address the chat template issue, including modifying the template to conditionally add **<|eot_id|>** tokens and providing two separate templates for training and inference.
   - Another option suggested is to modify the **transformers library** to support an extra parameter for the **apply_chat_template** function to distinguish between training and inference.
- **ChatML Template Challenges**: A user is determined to use **ChatML** templates for training the model.
   - They are exploring ways to overcome the difficulties associated with training a model with a new chat template, including potential issues with **special tokens** and **resize_token_embeddings**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct/discussions/26">meta-llama/Meta-Llama-3.1-70B-Instruct · Fix Llama 3.1 Chat Template to Properly Handle add_generation_prompt</a>: no description found</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1854">most model types now support flash attention 2 regardless of multipack support by winglian · Pull Request #1854 · axolotl-ai-cloud/axolotl</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1276230370399883326)** (1 messages): 

> - `accelerate fp8` 


- **Accelerate Adds fp8 Support**: **Accelerate** recently added support for **fp8**, meaning it's likely to translate to **Axolotl**.
   - However, we still need to figure out what the integration points are.
- **Integration Point Exploration**: The discussion focused on determining how the **fp8** support in **Accelerate** could be integrated into **Axolotl**.
   - No concrete solutions were proposed, but the need for exploration and collaboration was highlighted.


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1275900114799689739)** (4 messages): 

> - `LlamaCloud`
> - `LlamaIndex 0.11`
> - `RAG Pipeline`
> - `BoxReaderTextExtraction`
> - `Workflows` 


- **LlamaCloud Helps Optimize RAG Pipeline Chunk Size**: LlamaCloud streamlines chunk size in RAG pipelines, allowing users to experiment with different chunk sizes and visualize their impacts.
   - It offers features like cloning indexes for quick experimentation and efficient iteration without manual data management.
- **LlamaIndex 0.11 Launches with New Features**: LlamaIndex 0.11 has been launched with hundreds of new features and bug fixes, including a significant update to workflows that replace query pipelines.
   - This release is a step towards making LlamaIndex a production-ready platform, offering a more robust and user-friendly experience.
- **Box Integration for AI-powered Data Extraction**: LlamaIndex has partnered with Box to enhance enterprise data extraction, allowing users to leverage Box's advanced features for AI-powered data extraction.
   - The integration utilizes BoxReaderTextExtraction for direct text content extraction, streamlining data processing for LlamaIndex applications.
- **LlamaCloud Demonstration**: LlamaIndex is excited to showcase the capabilities of LlamaCloud, a platform designed to optimize RAG pipeline performance and efficiency.
   - The platform offers a range of features, including chunk size optimization, index cloning, and data visualization.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1275909120448528445)** (84 messages🔥🔥): 

> - `Ollama memory usage`
> - `LlamaIndex Property graph`
> - `SchemaLLMPathExtractor`
> - `Query Generation with LlamaIndex`
> - `PandasAI for CSV data` 


- **Ollama Memory Usage: Limiting Context**: A member asked about limiting memory usage for models, specifically for Ollama phi3 which was throwing an error.
   - The suggestion was to limit the context window size using `context_window` parameter in the `Ollama` class.
- **LlamaIndex Property Graph: Relationships**: A member asked about relationships available in the LlamaIndex Property graph.
   - They specifically wanted to know if relationships like `prev/next/parent/mentions` are accessible through the `SchemaLLMPathExtractor`.
- **Query Generation with LlamaIndex: Real Estate Schema**: A member discussed using LlamaIndex to generate queries for a real estate database based on natural language input.
   - They wanted to know if LlamaIndex is designed for this use case, or if it's better to focus on tuning the prompt instead.
- **PandasAI for CSV Data Processing**: A member asked about methods to load, store, and index CSV files for training GPT-3.5 turbo.
   - The suggestion was to use PandasAI, which offers natural language querying, data visualization, data cleansing, and feature generation for CSV data.
- **Bedrock Converse Package Maintenance**: A member asked about the maintainer of the Bedrock Converse package.
   - The response suggested checking the contribution history and encouraged contributing a PR if there are any issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Fm7nUgFSli-DmOaMnzPyGa0lJQTRGrVA?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://docs.pandas-ai.com/intro">Introduction to PandasAI - PandasAI</a>: no description found</li><li><a href="https://pypi.org/project/llama-index-llms-openai/#history">llama-index-llms-openai</a>: llama-index llms openai integration
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1276259180889505815)** (2 messages): 

> - `LlamaIndex`
> - `RAG`
> - `multi-strategy workflow`
> - `graph store`
> - `knowledge graph` 


- **LlamaIndex & Multi-Strategy Workflow**: The article discusses a multi-strategy workflow for retrieval augmented generation (RAG) using LlamaIndex, a framework for building LLM applications.
   - This approach involves employing multiple query strategies in parallel and evaluating their responses, aiming for efficiency and accuracy in response generation.
- **Importance of Efficient Retrieval**: The author emphasizes the critical role of efficient retrieval in RAG systems, as it directly impacts the quality of responses generated by LLMs.
   - Effective retrieval techniques are crucial for providing accurate and relevant information to LLMs, contributing to the overall performance of RAG applications.
- **Knowledge Graph & Graph Store Selection**: The article briefly mentions knowledge graphs and the challenges of selecting a suitable graph store for managing knowledge.
   - The discussion about graph stores and their relevance in the context of LlamaIndex and RAG systems is limited, with no specific recommendations or insights shared.



**Link mentioned**: <a href="https://medium.com/ai-advances/multi-strategy-workflow-with-reflection-in-llamaindex-an-in-depth-exploration-e4fd6a2e9545">Multi-Strategy Workflow with Reflection in LlamaIndex: An In-depth Exploration</a>: Ankush k Singal

  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1275896547086958675)** (67 messages🔥🔥): 

> - `Perplexity API`
> - `Mistral Large 2`
> - `Perplexity's Sources`
> - `Perplexity Image Generation`
> - `Perplexity Subscription Plans` 


- **Perplexity API Inquiry**: A user asks if anyone from the Perplexity team is online to answer a question about their API.
   - Another user suggests reaching out if they have any information about the API, mentioning they tried building a bot from it yesterday.
- **Mistral Large 2 vs. Other Models**: A user prefers Mistral Large 2 for its capabilities in custom system prompts, uncensored outputs, and unbiased nature by default, citing it's also suitable for jailbreak usage.
   - They suggest Mistral Large 2 is less expensive than GPT-4o and provides high-quality performance for complex tasks.
- **Perplexity's Reliance on Wolfram Alpha**: A user expresses concern that Perplexity often uses Wolfram Alpha for web searches instead of the web, which can lead to inaccurate results.
   - They suggest making a Sources section that can look at objective data for topics like politics to improve the accuracy of Perplexity's research.
- **Perplexity Image Generation Issues**: Multiple users report issues with Perplexity's image generation capabilities, including the inability to generate simple logos like hearts.
   - Other users share their own experiences with glitches, including random characters appearing in generated images, and difficulty in replicating these issues.
- **Perplexity Enterprise Subscription Pricing**: A user asks about the details of Perplexity's enterprise subscription plan, specifically whether the $40 monthly fee covers multiple members or requires an additional $40 per member.
   - Another user recommends contacting Perplexity support via email at support@perplexity.ai for further assistance.



**Link mentioned**: <a href="https://www.perplexity.ai/hub/faq/what-is-collections">What is Collections?</a>: Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1275957970433216512)** (11 messages🔥): 

> - `Lore's Emotional Instability`
> - `Microplastics in Brains`
> - `Retirement Preparation`
> - `Jonathan Ive's San Francisco Investment`
> - `Nightcore Music Effects` 


- **Lore's Potential Reproductive Frustration**: A psychological analysis explores whether Lore's emotional instability in *Star Trek: The Next Generation* could be linked to an inability to have offspring.
   - It draws parallels with other science fiction narratives exploring the desire for reproduction and legacy in artificial beings.
- **Microplastics Found in Human Brains**: Researchers discovered microplastics in human brain samples at alarmingly high concentrations, raising concerns about long-term health implications.
   - This discovery suggests that the brain is a major site of microplastic accumulation, highlighting the potential danger of plastic pollution on human health.
- **Planning for a Successful Retirement**: Retirement is portrayed as a crucial transition requiring careful planning, including financial management, health maintenance, and social engagement.
   - The article emphasizes the importance of finding fulfilling second careers, maintaining healthy habits, and managing investments for a secure and satisfying retirement.
- **Jonathan Ive's San Francisco Investment Spree**: Sir Jonathan Ive, former Apple design chief, has invested heavily in San Francisco's Jackson Square neighborhood, acquiring properties worth over $137 million.
   - His investment signals ambitious plans for the area, indicating a significant transformation led by a prominent figure in the design world.
- **The Effects of Nightcore Music**: Nightcore, a genre characterized by sped-up and pitch-shifted music, has gained popularity on platforms like TikTok.
   - While the genre may offer temporary mood boosts and stress reduction, the long-term mental health effects of Nightcore remain uncertain and require further research.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/how-is-perplexity-ai-different-aKjF.kNwSEyuc0v46TZwzw#0">How is Perplexity AI different?</a>: Perplexity AI stands out from other AI chatbots and search engines in several key ways:  1. Multimodel Approach:    - Perplexity AI leverages multiple large...</li><li><a href="https://www.perplexity.ai/page/microplastics-found-in-human-b-br1yKSQzT_W4M0NkS4iADA">Microplastics Found in Human Brains</a>: Recent studies have uncovered a disturbing reality: microplastics, tiny plastic particles less than 5 millimeters in size, are infiltrating human brains and...</li><li><a href="https://www.perplexity.ai/page/nightcore-s-mental-impact-Zan5T5AlRfOW0buspS4xtw">Nightcore&#x27;s Mental Impact</a>: Nightcore, known for its sped-up and pitch-shifted music, has become popular on platforms like TikTok. This genre may influence listeners&#x27; mood, energy, and...</li><li><a href="https://www.perplexity.ai/search/how-to-play-garen-in-league-of-iAhjBQX8QmOt5I9BCI5JxA">How to play Garen in League of Legends?</a>: To effectively play Garen in League of Legends, understanding his abilities, gameplay strategy, and itemization is crucial. Here’s a comprehensive guide on...</li><li><a href="https://www.perplexity.ai/search/what-are-the-benefits-of-using-4I6AKFtTSQervSShG7i7bQ#0">What are the benefits of using Smartsheet over Power BI for dashboard...</a>: When deciding between Smartsheet and Power BI for dashboard reporting in a small core team of a Project Management Office (PMO) with a large stakeholder...</li><li><a href="https://www.perplexity.ai/page/psychological-analysis-of-lore-MqJBIvaiTr2rjQMdvFRE0Q">The Possible Devastating Emotional Impact of Potential Reproductive...</a>: Lore, the antagonistic android brother of Data in Star Trek: The Next Generation, serves as a compelling study of artificial intelligence grappling with...</li><li><a href="https://www.perplexity.ai/page/jonathan-ive-jackson-square-VPRkJtEPSt2VlvlvNxTPKw">Jonathan Ive - Jackson Square</a>: Sir Jonathan Ive, the legendary former Apple design chief, has been quietly reshaping a historic corner of San Francisco. Since 2020, Ive and his design firm...</li><li><a href="https://www.perplexity.ai/page/euntoe-junbi-gaideu-EdRmAZrLRU6G6iXibqU2xA">은퇴 준비 가이드</a>: 은퇴는 인생의 새로운 장을 여는 중요한 전환점으로, 체계적인 준비가 필수적입니다. 재정 관리, 건강 유지, 사회적 관계 형성 등 다양한 측면에서의 준비를 통해 안정적이고 만족스러운 노후 생활을 영위할 수 있습니다.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1275947434648010884)** (6 messages): 

> - `Perplexity API`
> - `Domain Filtering`
> - `Citations` 


- **API Access for Specific Buttons**: A user inquired about accessing a specific button through the Perplexity API.
- **Domain Filtering and Citations for API Users**: The best option is to wait for domain filtering and citations to become generally available for API users.
- **Perplexity API Documentation**: The user provided a link to the Perplexity API documentation for chat completions.



**Link mentioned**: <a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: Generates a model's response for the given chat conversation.

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1275957503787798693)** (12 messages🔥): 

> - `Github Desktop`
> - `Git send-email`
> - `Git am`
> - `Modular Community Manager`
> - `Modular Calendar` 


- **Github Desktop: The Struggle is Real**: A user found [Github Desktop](https://github.com/apps/desktop) to be less intuitive than expected, commenting *"Not the most intuitive product ever"*.
   - They also noted a lack of support for  `git send-email` and `git am`, which users find useful for managing changes.
- **Welcome, Caroline!**: Caroline, Modular's new Community Manager, introduced herself, mentioning her background in community/dev rel at Streamlit.
   - She invited members to book time for virtual coffee chats to discuss their experiences and feedback on the community.
- **The Battle for Zira is Complete**: A user announced the completion of the "Battle for Zira", using a victorious emoji to celebrate the achievement.
   - No further details were provided about the nature of the battle or its significance.
- **Modular 'clean' command not found**: A user attempted to use the command `modular clearn`, but received an error indicating that the command does not exist.
   - They were likely attempting to use the `modular clean` command, which is used for cleaning up the modular project.



**Link mentioned**: <a href="https://modul.ar/caroline-calendar.">Modular Community Chat ☕ - Caroline Frasca</a>: no description found

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1275899785886568550)** (33 messages🔥): 

> - `math.isclose open source`
> - `Mojo github search issues`
> - `Mojo Docs Search`
> - `Mojo async/sync performance`
> - `Mojo stdlib and import paths` 


- **math.isclose is not open source**: A member asked why **math.isclose** is not open sourced.
   - Another member pointed out that **math.isclose** is available on the **nightly** branch in the **Mojo github repository**.
- **Mojo github search problems**: A member reported encountering issues with **github search** in finding Mojo related files.
   - The other member suggested the user might not have been on the **nightly branch** of the repo.
- **Mojo Docs Search Improvements Requested**: Several members commented on the need for **improvements to the Mojo documentation search functionality.**
   - They suggested filtering search results by **Mojo stdlib modules**, **MAX lib**, **blogs and news**, and other options.
- **Mojo Async/Sync Performance Considerations**: A member asked about the **performance of async functions** in Mojo, comparing it to **Python**.
   - Other members suggested that a **sans-io HTTP implementation** might be better for now, as **IO** is still in development.
- **Mojo import paths for files in different folders**: A member inquired about the **idiomatic way to import functions from files in different folders** within a Mojo project.
   - The question was raised in comparison to the approach used in **Python**, which utilizes **sys.path** for such imports.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modular.com")">no title found</a>: no description found</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/math/math.mojo#L983">mojo/stdlib/src/math/math.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/test/builtin/test_simd.mojo#L1372">mojo/stdlib/test/builtin/test_simd.mojo at 8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1275947455913136138)** (8 messages🔥): 

> - `Mojo/MAX installation issues`
> - `Modular CLI`
> - `Debian installation`
> - `Mac installation`
> - `venv activation` 


- **Mojo/MAX Installation Issues on MacOS**: A user reported that they have to reinstall Mojo and MAX every time they turn off their MacBook Pro.
- **Modular CLI Installation on Debian**: A user attempted to install the Modular CLI on Debian using `sudo apt-get install modular=0.9.2`, but received an error message indicating that the package was not available.
- **Resolving the Missing Modular Debian Source**: A user suggested that the Modular Debian source might have been dropped.
- **Reporting Issues to the Modular Repo**: A user suggested that the user experiencing issues should create a new issue on the Modular repository on GitHub.
- **Activating Virtual Environments**: A user reminded another user to activate the venv or conda environment, providing the command `source ~/max-venv/bin/activate`.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/install">Install MAX &amp; Mojo | Modular Docs</a>: Welcome to the MAX install guide! This installs both MAX and Mojo🔥.</li><li><a href="https://github.com/modularml/max/issues">Issues · modularml/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX Platform - Issues · modularml/max
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1275940061489397800)** (21 messages🔥): 

> - `RAG Development & Evaluation Strategies`
> - `Cohere x w&b Webinar`
> - `Cohere Developer Office Hours`
> - `C4AI Program`
> - `Chunking XLSX Files` 


- **RAG Webinar with Cohere and Weights & Biases**: Cohere and Weights & Biases are hosting a webinar about RAG development and evaluation strategies, featuring insights from Cohere's Maxime Voisin.
   - Register now for the webinar at [https://wandb.ai/site/resources/events/cohere-webinar-august](https://wandb.ai/site/resources/events/cohere-webinar-august).
- **Cohere Developer Office Hours on Discord**: Cohere hosts monthly Developer Office Hours on Discord to answer community questions.
   - The next session features Sandra as the guest and is scheduled for [https://discord.gg/rQTg96u7?event=1275152691818922024](https://discord.gg/rQTg96u7?event=1275152691818922024).
- **C4AI: Cohere's Non-profit Research Lab**: C4AI is Cohere's non-profit research lab focused on solving complex machine learning problems.
   - It supports fundamental research, explores the unknown, and seeks to create more points of entry into machine learning research.
- **Chunking XLSX Files for RAG**: A member asked for tips on chunking XLSX files for RAG.
   - Another member suggested using the usual chunking method and leveraging embeddings and rerankers, which are designed to handle raw data. They also mentioned converting XLSX to markdown for table preservation.



**Link mentioned**: <a href="https://cohere.com/research">Cohere For AI (C4AI)</a>: Cohere For AI is a non-profit research lab that seeks to solve complex machine learning problems. We support fundamental research that explores the unknown, and are focused on creating more points of ...

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1275935298785644694)** (18 messages🔥): 

> - `LLM Agent Task Determination`
> - `Cohere API 403 Error`
> - `VPN and API Access`
> - `Chunking XLSX Files`
> - `C4ai` 


- **Agent Task Determination using ReAct**: A member inquired about the optimal way for an LLM agent to determine which task to perform based on the question, given that the agent utilizes a vector database, SQL queries, and other APIs.
   - Another member suggested using the ReAct framework, which involves providing a title and description to the model, allowing it to automatically figure out the necessary actions based on the input.
- **Cohere API 403 Error: Forbidden**: A user encountered a 403 Forbidden error when sending a POST request to the Cohere API from a local server.
   - The user suspected the issue might be related to their server's IP address not being whitelisted, and shared the request details, including the endpoint, content type, authorization header, and request body.
- **VPN and API Access: A Potential Cause for 403 Errors**: A user reported that they are located in France and that the project manager will provide an identifier to resolve an issue with a developer who is using the company's French VPN.
   - The user asked if the developer's VPN address being on a stop list could be a possible cause for the 403 error.
- **Chunking XLSX Files: Seeking Tips**: A member requested guidance on chunking XLSX files, specifically asking for tips on how to best approach this task.
- **C4ai: Cohere for AI**: A member enthusiastically expressed support for Cohere and its AI capabilities.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1276172058778406982)** (7 messages): 

> - `403 Forbidden`
> - `Cohere API from R`
> - `Cohere API using curl`
> - `Cohere Command R+ 128k context`
> - `OpenAI structured outputs` 


- **403 Forbidden Error**: A user reported receiving a "403 Forbidden" error when sending a request via the Cohere API from their local server.
   - They asked if their server's IP addresses might be missing from Cohere's whitelist.
- **Using Cohere's Command R+ with 128k Context from R**: A user inquired about accessing Cohere's Command R+ with 128k context from R, without using the canned libraries for Python.
   - They specifically asked if it's possible to use it via HTTP requests or even `curl`.
- **Using Cohere API with curl**: A user received confirmation that Cohere's Command R+ with 128k context is accessible using `curl`.
   - They provided a complete `curl` command example demonstrating how to make a request to the Cohere API, including the necessary headers and JSON payload.
- **Structured Outputs in Cohere API**: A user inquired about the existence of a concept similar to OpenAI's "structured outputs" within the Cohere API.
   - The user provided a link to OpenAI's documentation on structured outputs for context.



**Link mentioned**: <a href="https://docs.cohere.com/reference/chat">Chat Non-streaming — Cohere</a>: Generates a text response to a user message. To learn how to use the Chat API with Streaming and RAG follow our  Text Generation guides .

  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1276175957933166633)** (6 messages): 

> - `Jozu Hub`
> - `Cohere Model Support`
> - `ModelKit` 


- **Jozu Hub: Your AI Project HQ**: The team released an early preview of their project, **Jozu Hub**, which aims to be a centralized platform for versioning and sharing AI projects. You can access it here: [https://jozu.ml](https://jozu.ml).
   - The **ModelKit** feature helps eliminate the guesswork involved in understanding the components that make up an AI project (data sets, code, parameters, model versions, documentation, etc).
- **Cohere Model Support on the Horizon**: The team confirmed that they are working on integrating support for **Cohere models** on **Jozu Hub**.
   - They are aiming to get all major models, including Cohere, packaged and hosted on the platform.



**Link mentioned**: <a href="https://jozu.ml)">no title found</a>: no description found

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1275895924480020552)** (30 messages🔥): 

> - `Expected Value of an Item`
> - `AI Assistance with Math Problems`
> - `Ideogram 2.0 Review`
> - `SwarmUI & Flux`
> - `French Community` 


- **Expected Value of an Item**: A user is trying to calculate the expected cost of an item in a game where they have multiple tries, each with a different cost.
   - They want to know the expected cost given that they only try until they get the item or fail four times, with a final cost of 200.
- **AI Assistance with Math Problems**: The user is finding it difficult to get a correct answer from AIs like Gemini, ChatGPT, and Claude.
   - Another user suggests using Python for calculations as it can run Python code and offers more accurate results.
- **Ideogram 2.0 Review**: A user is impressed with the new Ideogram 2.0, a tool for generating images, but notes that a paid subscription is required to download PNGs.
   - They have seen even better examples of Ideogram 2.0 text from others and considers it 'amazing good'.
- **SwarmUI & Flux**: The user is highly rating SwarmUI, an installer that supports NVIDIA/AMDGPUs, wrapping the complexity of comfyUI.
   - They describe it as having a simple UI and the ability to load other people's workflows.
- **French Community**: A user is looking for a French community as their news sources are currently limited to US-based humans.
   - They ask if there are any 'frogs' (French speakers) in the community.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1275893591985750158)** (13 messages🔥): 

> - `GPT Training Data`
> - `Life Coach App`
> - `Custom GPTs`
> - `GPT4o vs GPT4`
> - `GPT Formatting` 


- **Where to Buy GPT Training Data**: A user asks where they can buy data to train a GPT for a Life Coach app. 
   - They're specifically looking for a dataset of questions and answers, which can be found on platforms like Hugging Face Datasets and Kaggle.
- **Custom GPTs: GPT4o vs GPT4**: The user asks if they can make custom GPTs use GPT4 instead of GPT4o.
   - No further information is provided on how to achieve this.
- **Creating Custom GPTs: Tips and Resources**: The user requests good resources for creating custom GPTs, including articles and videos.
   - They're looking for advice on improving their GPT creation process and have already created several.
- **GPT Formatting Consistency**: The user is trying to create a GPT that generates Dungeons & Dragons attacks, dodges, and counterattacks, but is struggling with inconsistent formatting in its responses.
   - They're looking for ways to get consistent outputs, such as providing clear examples or formatting instructions.
- **ChatGPT Misinterpretation**: A user asks for help creating a GPT that generates D&D content, but it seems ChatGPT misinterpreted the request as an API implementation.
   - The user is looking for a custom GPT on ChatGPT, not an API implementation, and suggests rephrasing the request to clarify their intent.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

madame_architect: I'm not sure I know exactly what you're asking for. Could you say more?
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

madame_architect: I'm not sure I know exactly what you're asking for. Could you say more?
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1275946686853939422)** (10 messages🔥): 

> - `Open Source AI Models`
> - `Eleuther.ai`
> - `DPO Fine-tuning`
> - `Instruction Prompt Templates`
> - `Multi-turn Chat Data Prep` 


- **Open Source AI is Often Not**: Many generative AI models claim to be open source, but they don't disclose what's in the training set, and they may be using factually wrong, biased, or copyright-restricted data.
   - The US government is weighing the risks of open LLMs and the confusion around AI model restrictions is described as “open washing.”
- **Eleuther.ai: The Golden Example of Open Source AI**: Eleuther.ai is considered a purely open source AI initiative, with an intention to progress the science without a business model behind it.
   - Less open models often have hidden provisions in their legal licenses that inhibit certain use cases.
- **Fine-tuning DPO with Instruction Prompts**: A user asked whether to apply an instruction prompt template to the DPO dataset when fine-tuning for an instruction model.
   - Several users confirmed that applying instruction prompts is generally recommended, as it allows for more effective use of the model with the prompt.
- **Preparing Multi-turn Chat Data for Fine-tuning**: A user asked about preparing multi-turn chat data for fine-tuning a base model.
   - The user suggested using the first question and answer as src, trg, or the whole conversation as input and last answer as target, or using question1 as input and answer1 as target, then q1, a1, q2 as input and a2 as target.
- **Alignment-handbook for Multi-turn Chat Data**: A user sought guidance on preparing multi-turn chat data for fine-tuning, and another user referred them to the alignment-handbook.
   - The user clarified that the provided code snippet aligns with the second approach of using the whole conversation as input and last answer as target.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/huggingface/alignment-handbook/blob/27f7dbf00663dab66ad7334afb7a1311fa251f41/src/alignment/data.py#L80">alignment-handbook/src/alignment/data.py at 27f7dbf00663dab66ad7334afb7a1311fa251f41 · huggingface/alignment-handbook</a>: Robust recipes to align language models with human and AI preferences - huggingface/alignment-handbook</li><li><a href="https://archive.is/i2Hi7">Be careful with &#x2018;open source&#x2019; AI | LeadDev</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1275995741546090569)** (27 messages🔥): 

> - `Model Evaluation`
> - `Model Performance Degradation`
> - `Benchmark Track Rebuttals`
> - `Model Merging`
> - `Model Distillation` 


- **Direct Evaluation of Pretrained Models on New Corpora**: A user questioned the validity of evaluating a pretrained model on a different pretraining corpus directly, arguing that domain mismatch could lead to inaccurate results.
   - Others argued that this is a common practice, and that the evaluation loss would decrease significantly after just 100 training steps, suggesting a clear transferability of knowledge.
- **Model Performance Degradation**: A user asked how to reliably decrease the performance of an LLM on a benchmark like MMLU, aiming to simulate the performance of a smaller model by modifying a larger model.
   - Suggestions included adding noise to the activations, applying model distillation with LoRAs, or reversing training the model.
- **Benchmark Track Rebuttals**: A user inquired about the response rate of reviewers after submitting a paper to the benchmark track.
   - The user reported receiving no responses from reviewers despite the discussion period nearing its end, raising concerns about the potential for limited feedback and discussion.
- **Model Merging Tactics**: The discussion touched upon model merging tactics, with a user suggesting applying the difference between two models (UltraChat and Mistral) to another (Mistral-Yarn).
   - While met with skepticism, the user remained optimistic, citing successful past attempts at similar strategies.
- **Chip Design using Transformers**: A paper titled 'ShortCircuit: Accelerating Logic Circuit Synthesis with Transformers' was introduced, proposing a novel approach for designing boolean circuits using transformer-based architectures.
   - The paper details a two-phase process combining supervised and reinforcement learning to enhance generalization and proposes an AlphaZero variant to handle the large state space involved.



**Link mentioned**: <a href="https://arxiv.org/abs/2408.09858">ShortCircuit: AlphaZero-Driven Circuit Design</a>: Chip design relies heavily on generating Boolean circuits, such as AND-Inverter Graphs (AIGs), from functional descriptions like truth tables. While recent advances in deep learning have aimed to acce...

  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/)** (1 messages): 

catboy_slim_: it is a specific dataset that is so far as i can tell still publicly available
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1276010090201088104)** (7 messages): 

> - `HellaSwag evaluation`
> - `lm-evaluation-harness`
> - `generate_until`
> - `filtering pipeline`
> - `log likelihood` 


- **HellaSwag Evaluation and 'resps' & 'filtered_resps'**: The entries in 'resps' and 'filtered_resps' are used for evaluating models on multi-choice datasets like HellaSwag, specifically calculating the negative log likelihood.
   - The first array in 'resps' appears to represent the log likelihood values for each option, while 'filtered_resps' seems to be for 'generate_until' tasks, used after applying a filtering pipeline.
- **Understanding 'resps' Structure**: The 'resps' list is nested, indicating the log likelihood value for each option, with 'False' signifying that the model didn't choose that option.
   - The additional nesting in 'resps' might be a consequence of the internal structure of the evaluation process, while the 'filtered_resps' list is used for filtering responses before calculating the metrics.
- **Filtering Pipeline in 'generate_until'**: 'generate_until' tasks involve generating outputs until a certain condition is met, and the 'filtered_resps' list is specifically used in such scenarios.
   - The 'filtered_resps' list is used to filter the outputs before calculating the metric, presumably to ensure the generated output meets specific criteria.
- **Log Likelihood and 'False' Value**: The negative log likelihood (NLL) is a common metric in language modeling, where lower values indicate better performance.
   - The 'False' value in the output indicates that the model did not select that particular option as the most likely one, based on the calculated NLL values.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1673">All responses are considered false? · Issue #1673 · EleutherAI/lm-evaluation-harness</a>: I am evaluating several LLMs on my custom dataset using negative log likelihood and I always thought that the response with the highest likelihood is paired with &#39;true&#39; while the others are pa...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1673#issuecomment-2040371764">All responses are considered false? · Issue #1673 · EleutherAI/lm-evaluation-harness</a>: I am evaluating several LLMs on my custom dataset using negative log likelihood and I always thought that the response with the highest likelihood is paired with &#39;true&#39; while the others are pa...
</li>
</ul>

</div>
  

---



### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1276118034133549078)** (37 messages🔥): 

> - `LightGBM`
> - `LSTM`
> - `Commodity Price Forecasts`
> - `Kaggle Competitions`
> - `Production Data` 


- **LightGBM's Dominance in Kaggle Contests**: LightGBM has been widely used in Kaggle competitions, including the Corporación Favorita Grocery Sales Forecasting, Recruit Restaurant Visitor Forecasting, and the M5 accuracy competition.
   - This has led to a perception of its superior performance, even in production settings.
- **LightGBM's Performance in Production: The Debate**: However, some argue that LSTM might be a better choice in production environments.
   - There's a debate regarding LightGBM's effectiveness in real-world scenarios, with some claiming it might not perform as well with live data compared to its performance in Kaggle competitions.
- **Investigating LightGBM's Use for Commodity Price Forecasting**: A study explored using LightGBM for commodity price forecasts, citing its success in the M5 competition.
   - The authors employed features like SMA, EMA, and VWAP, but their results indicated that an ARIMA model outperformed LightGBM for lead and tin returns.
- **The Importance of the Forecasting Task**: The choice of model depends on the specific forecasting task, such as predicting next-period prices versus forecasting over longer horizons (e.g., 3-6 months).
   - While LightGBM can generate multi-step forecasts through recursive feeding, it's essential to consider the context and complexity of the prediction.
- **Alternative Approaches Before Deep Learning**: SMA, EMA, and ARIMA models are often effective starting points for time series forecasting.
   - Deep learning models like LightGBM and LSTM may be more useful when dealing with a large number of non-traditional exogenous variables, where capturing seasonality might be less critical.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/MachineLearning/comments/ob2rll/d_how_often_is_lightgbm_used_for_forecasting_in/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://phdinds-aim.github.io/time_series_handbook/08_WinningestMethods/lightgbm_m5_forecasting.html">Chapter 8: Winningest Methods in Time Series Forecasting &#8212; Time Series Analysis Handbook</a>: no description found
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1276216575686344847)** (31 messages🔥): 

> - `AI burnout`
> - `Frontier Labs`
> - `Greg Brockman's Work Hours`
> - `Twitter anxiety` 


- **AI burnout**: A member stated that AI can definitely get far *more* burnout-y than other fields, a concern they have.
   - Another member agreed, but added that user burnout is a real thing too, and those two are connected.
- **Frontier Labs work intensity**: A member stated that people in frontier labs work *very hard* and that it seems unsustainable.
   - They noted that they don't have a long career in tech, but they are trying to walk people back from the cliff.
- **Greg Brockman's Work Hours**: A member reminded everyone that Greg Brockman posted his time tracker on Twitter showing 97 hours of coding work in a week.
   - They were surprised he didn't take a break earlier; it took him 9 years to take a break!
- **Twitter anxiety**: A member expressed feeling gross coming back from the backcountry and full unplugging to Twitter and AI discourse.
   - They said that Twitter is the worst of it and it is very anxiety inducing.


  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1276024667378942035)** (6 messages): 

> - `Lilian Weng Blog`
> - `Diffusion Models`
> - `Generative Models`
> - `Distillation`
> - `Score-based generative modeling` 


- **Lilian Weng's Blog Post on Diffusion Models**: Lilian Weng's blog post on Diffusion Models, written in July 2021, includes links to her blog posts on other types of Generative Models like GANs, VAEs and Flow-based models.
   - The blog post has been updated multiple times, with the latest update on April 13, 2024, adding sections on progressive distillation, consistency models, and model architecture.
- **Clarification on Diffusion Models vs Distillation**: A user initially confused diffusion models with distillation.
   - Another user corrected the misunderstanding, explaining that distillation is a different concept.
- **Distillation Notes**: A user mentioned they have some basic notes on distillation to include in a post, but haven't sent them out yet.



**Link mentioned**: <a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/">What are Diffusion Models?</a>: [Updated on 2021-09-19: Highly recommend this blog post on score-based generative modeling by Yang Song (author of several key papers in the references)]. [Updated on 2022-08-27: Added classifier-free...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1276023253022019584)** (33 messages🔥): 

> - `LLM for NL to SQL`
> - `Prebuilt Queries for SQL`
> - `LLM Accuracy and SQL`
> - `RAG for SQL`
> - `ChromaDB RAG for CSV` 


- **Local LLMs for NL to SQL**: A user inquired about the feasibility of using a local LLM for natural language to SQL translation.
- **Prebuilt Queries for SQL**: One user suggested using prebuilt queries with placeholders to simplify the text to SQL workload.
- **LLM Accuracy and SQL**: The discussion centered on addressing errors in SQL generation that lead to non-executable queries.
- **RAG for SQL using CodeLLM**: The approach of combining Retrieval Augmented Generation (RAG) with a code-specific LLM was proposed to improve SQL generation.
- **RAG for CSV with ChromaDB**: A user sought advice on performing RAG on a CSV with a high number of columns.


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1275896575029149767)** (2 messages): 

> - `Flags for AI`
> - `4149 AI`
> - `Proactive Guidance in Slack`
> - `AI for Research` 


- **4149 AI's New 'Flags' Feature**: A new AI-powered feature called "Flags" has been released by [4149 AI](https://4149.ai) and is being tested.
   - It aims to provide proactive, real-time guidance on the state of a team through Slack direct messages.
- **Flags Offer Proactive Team Insights**: The "Flags" feature aims to send immediate alerts when signs of a team slipping emerge.
   - It also aims to help stay on top of issues before they become problems and highlight the team's wins and accomplishments.
- **User Customization and Approval of Flags**: Users can customize what the AI sees and all messages go to the user for approval by default.
   - The setup process takes less than a minute and only requires a Slack team.
- **AI in Research - Enthusiasm**: The author expressed excitement about the use cases for AI in research.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://4149.ai)">no title found</a>: no description found</li><li><a href="https://beta.4149.ai/register">4149 [beta]</a>: no description found
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1275898837629800499)** (29 messages🔥): 

> - `Ideogram 2.0`
> - `Mistral-NeMo-Minitron-8B`
> - `Sovereign AI`
> - `v0 updates`
> - `AI development trends` 


- **Ideogram 2.0 Launch: Free for Everyone**: Ideogram 2.0, the latest text-to-image model from the former Google Imagen 1 team, is now available to all users for free.
   - This launch includes a new iOS app, a beta API, and Ideogram Search, boasting over 1 billion images created.
- **Nvidia's New Mistral-NeMo-Minitron-8B**: Nvidia released Mistral-NeMo-Minitron-8B, a base LLM obtained by pruning and distilling the Mistral-NeMo 12B.
   - This model outperforms Mistral-7B and LLaMa-3.1-8B across 8/9 benchmarks and is available on Hugging Face under a permissive license.
- **Sovereign AI: A New Streaming Data System**: The Infinite ML podcast featured Sovereign AI, a streaming data system developed by Redpanda Data.
   - The discussion covered topics like defining streaming data, the evolution of such systems, and how Sovereign AI works in practice.
- **v0 Conversational UI: Next.js, React, and More**: v0's new conversational UI is now in beta, featuring up-to-date knowledge of Next.js, React, and other web technologies.
   - It boasts improved client components support, the ability to run npm packages, faster streaming, and several examples showcasing its capabilities.
- **AI Development Trends: Slowing Growth**: An analysis of GitHub repository creation data suggests that AI development growth has slowed, moving from exponential to linear.
   - Azure and OpenAI showed strong adoption in March 2024, while Amazon Bedrock's growth may have peaked in June 2024.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/v0/status/1826020673908535325">Tweet from v0 (@v0)</a>: v0&#39;s new conversational UI:  • Up-to-date Next.js, React, and web knowledge • Improved client components support • Ability to run npm packages like framer-motion • Faster and more reliable streami...</li><li><a href="https://x.com/_philschmid/status/1826292873446248922">Tweet from Philipp Schmid (@_philschmid)</a>: The Pruning & Distillation story continues! @nvidia just released Mistral Nemo Minitron 8B, a base LLM obtained by pruning and distilling the Mistral-NeMo 12B! 👀  TL;DR:  ✨ Used 400B tokens used for ...</li><li><a href="https://x.com/Smol_AI/status/1826410077194256729">Tweet from AI News by Smol AI (@Smol_AI)</a>: [21 Aug 2024]  Ideogram 2 + Function Calling Leaderboard V2  &#39;Tis the season of sequels.  After the spectacular launch of Flux (the former Stable Diffusion team), @ideogram_ai (the former Google I...</li><li><a href="https://x.com/prateekvjoshi/status/1826375520277463382">Tweet from Prateek Joshi (@prateekvjoshi)</a>: The topic on Infinite ML pod today is Sovereign AI. We have @emaxerrno on the show to talk about it. He&#39;s the founder and CEO of @redpandadata. They&#39;ve raised more than $165M in funding from i...</li><li><a href="https://aiencoder.substack.com/p/querying-ai-and-cloud-trends-azure">Querying AI and Cloud Trends: Azure and OpenAI Dominate but Growth Slows, Amazon May Have Peaked</a>: Cutting through the AI hype to query actual developer usage (with presumptions), for prioritization of safety tools and guidance for partnerships.</li><li><a href="https://x.com/ideogram_ai/status/1826277550798278804?s=46">Tweet from Ideogram (@ideogram_ai)</a>: Introducing Ideogram 2.0 — our most advanced text-to-image model, now available to all users for free.  Today’s milestone launch also includes the release of the Ideogram iOS app, the beta version of ...</li><li><a href="https://x.com/giffmana/status/1826324924564398185">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: Where will we move the goalposts to next?  Props if you recognize all 3 of them!  Quoting Ideogram (@ideogram_ai)   Introducing Ideogram 2.0 — our most advanced text-to-image model, now available to a...</li><li><a href="https://x.com/GoogleAI/status/1826363610680979925">Tweet from Google AI (@GoogleAI)</a>: Today we introduce a new zero-shot, cross-lingual voice transfer module that can be easily plugged into a state-of-the-art TTS system to restore the voices of speakers with dysarthric speech. See it i...</li><li><a href="https://github.com/autogen-ai">autogen-ai</a>: autogen-ai has 3 repositories available. Follow their code on GitHub.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1276233199101542432)** (3 messages): 

> - `GPT-4o fine-tuning`
> - `Genie coding agent`
> - `SWE-Bench`
> - `RAG`
> - `In-context learning` 


- **GPT-4o Fine-Tuning: Worth It?**: The Latent Space Podcast explores whether finetuning GPT-4o is worth it, with guest Alistair Pullen from Cosine.
- **Genie's Massive Fine-Tuning Effort**: Genie has undertaken an unprecedented fine-tuning effort for GPT-4o, utilizing billions of tokens of synthetic code data derived from real user logs and purposefully sabotaging Abstract Syntax Trees (ASTs).
- **Synthetic Code Data Generation**: The podcast delves into the creation of billions of tokens of synthetic code data using real user logs and ASTs.
- **OpenAI's GPT-4o Fine-Tuning Launch**: OpenAI has launched fine-tuning capabilities for GPT-4o, allowing developers to enhance the performance and accuracy of their applications.
- **The Significance of Large-Scale Fine-Tuning**: The podcast discusses the impact of large-scale fine-tuning on GPT-4o's performance, particularly in the context of code generation.



**Link mentioned**: <a href="https://x.com/swyx/status/1826673380294267328">Tweet from swyx.ai (@swyx)</a>: 🆕 @latentspacepod: Is finetuning GPT4o worth it?  w/ @AlistairPullen of @cosine_sh  Betteridge&#39;s law says no: with 59 different flavors of RAG, and &gt;2million token context + prompt caching, it...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1275913371534885016)** (16 messages🔥): 

> - `Open Interpreter usability`
> - `Open Interpreter searching issues`
> - `Open Interpreter model issues`
> - `Open Interpreter feature requests`
> - `Open Interpreter model suggestions` 


- **Open Interpreter Search Issues**: A user reported that web searching in Open Interpreter only works when completely refreshing the terminal and OI, not when continuing a conversation.
- **Open Interpreter model suggestions**: A user suggested that Phi-3.5-mini and Qwen2 models are surprisingly good.
- **Open Interpreter Model Type**: A user asked which model was being used by another user, suspecting it might not be gpt-4.
- **Open Interpreter Interface Concerns**: A user expressed that documenting profiles and paths out of the interface is less confusing than command line bookmarks that change over time.


  

---



### **AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1276254821539643503)** (1 messages): 

> - `Jamba 1.5`
> - `Jamba 1.5 Mini`
> - `Jamba 1.5 Large`
> - `SSM-Transformer Architecture`
> - `Long Context Handling` 


- **Jamba 1.5: Introducing the New Generation**: AI21 Labs has launched **Jamba 1.5 Mini** (12B active/52B total) and **Jamba 1.5 Large** (94B active/398B total), powered by the novel **SSM-Transformer Jamba architecture** that combines **Transformer's quality** with **Mamba's efficiency**.
   - These models boast a **256K effective context window** - the longest in the market - and deliver speed up to **2.5X faster on long contexts** compared to other models in their class.
- **Jamba 1.5 Mini: A Leader in Quality and Speed**: **Jamba 1.5 Mini** outperforms its size class, achieving a score of **46.1** on **Arena Hard**, while **Jamba 1.5 Large** scores **65.4**, surpassing both **Llama 3.1 70B** and **405B**.
   - The models support **English, Spanish, French, Hebrew, Arabic**, and more, offering native support for **JSON output, function calling, and document processing**.
- **Open Access to Jamba 1.5**: **Jamba 1.5 Mini and Large** are available for immediate download on **Hugging Face** and can be deployed across major cloud platforms like **Together AI, AWS, GCP, Azure**, and others.
   - AI21 Labs has released these models under the **Jamba Open Model License**, aiming to democratize access to quality models and encourage further experimentation.
- **Jamba 1.5: Built for Efficiency**: The **Jamba 1.5 family** is designed to handle thousands of pages of text, complex code, and sophisticated agents quickly and efficiently.
   - This release marks the first time a non-Transformer model has been successfully scaled to the quality and strength of leading models, showcasing the power of the **SSM-Transformer architecture**.
- **Jamba: The Future of Long-Context Models**: AI21 Labs believes the **Jamba 1.5 models** are a significant advancement in long-context modeling.
   - They offer unrivaled speed, efficiency, and quality while providing the longest context window among open models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/ai21labs/jamba-15-66c44befa474a917fcf55251">Jamba-1.5 - a ai21labs Collection</a>: no description found</li><li><a href="https://studio.ai21.com/v2/chat">AI21 Studio</a>: no description found</li><li><a href="https://www.ai21.com/jamba">Foundation models</a>: no description found</li><li><a href="https://www.ai21.com/blog/announcing-jamba-model-family">The Jamba 1.5 Open Model Family: The Most Powerful and Efficient Long Context Models</a>: The new family of open models from AI21, offering unrivaled speed, efficiency, and quality and the longest context window among open models.
</li>
</ul>

</div>
  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1276183856206053377)** (4 messages): 

> - `Jamba-1.5 Fine Tuning`
> - `Jamba-1.5 Large` 


- **Jamba-1.5 Fine Tuning Not Available**: A user asked about the availability of fine-tuning the Jamba-1.5 model through the Studio UI.
   - An AI21 staff member responded that only instruct versions are available on Studio, and fine-tuning will not be offered there for the time being.
- **Jamba-1.5 Large Features**: Jamba-1.5 Large is AI21's most advanced model.
   - It has advanced capabilities for reasoning, code generation, high context, and multilingual processing.



**Link mentioned**: <a href="https://rubiks.ai/search/?id=mtjvmz-r2fb-98s4-eik5-gqzex7ilua3n">Jamba 1.5 Release</a>: **Jamba 1.5 Release: A Hybrid SSM-Transformer Model for Agentic AI**\\n\nAI21 has announced the release of Jamba 1.5, a new version of its hybrid SSM-Transformer model that combines the strengths of b...

  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1276289164425433140)** (2 messages): 

> - `API Rate Limits`
> - `OpenAI API Rate Limits` 


- **OpenAI API Rate Limits**: A user asked about the API rate limits and then self-answered their question with a rate limit of 200 requests per minute (RPM) and 10 requests per second (RPS).
- **OpenAI API Rate Limits**: A user asked about the API rate limits and then self-answered their question with a rate limit of 200 requests per minute (RPM) and 10 requests per second (RPS).


  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1276131458968326227)** (4 messages): 

> - `Code Review`
> - `mypyc Compilation`
> - `Tinygrad` 


- **Responding to Code Review**: A member expressed frustration with code review responses that deflect responsibility to the reviewer by saying things like "I will do it if you want/ask."
   - They emphasized the importance of authors taking ownership of their changes, thinking critically about suggestions, and either implementing them or providing a reasoned explanation for not doing so.
- **Tinygrad Mypyc Compilation**: Someone expressed interest in working on getting **Tinygrad** to compile with **mypyc**.
   - A member volunteered to take a look at the problem.


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1275942659185315902)** (4 messages): 

> - `Torchtune and T5`
> - `Weight Mapping` 


- **Torchtune Challenges with T5**: A member noted that the only tricky thing about **T5** is its trainable attention bias, but other components are pretty much standard.
   - They also mentioned that **Torchtune** currently doesn't support encoder-decoder architecture, requiring task-specific training loop tweaking.
- **Weight Mapping T5 to Torchtune**: One member suggested comparing weight names between the **Hugging Face** and **Torchtune** repos for mapping purposes.
   - Specifically, they pointed to the **T5-small** model from Hugging Face and the **convert_weights.py** file from Torchtune.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/google-t5/t5-small/tree/main?show_file_info=model.safetensors">google-t5/t5-small at main</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/blob/861c3b214b029ee819fb554055aa467a52f22862/torchtune/models/convert_weights.py#L31-L45.">torchtune/torchtune/models/convert_weights.py at 861c3b214b029ee819fb554055aa467a52f22862 · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1276230445343969412)** (3 messages): 

> - `LinkedIn Survey`
> - `Infinite Generative YouTube` 


- **LinkedIn Survey Seeking User Insights**: A survey is being conducted to understand people's perceptions of LinkedIn as a professional networking platform.
   - The survey aims to gather valuable insights from the community on various aspects of LinkedIn and welcomes participation from all.
- **Dev Needed for Infinite Generative YouTube**: A team is gearing up to launch a closed beta for an infinite generative YouTube project and is seeking a passionate developer to join their team.
   - Anyone interested in building with these models is encouraged to reach out and learn more about the opportunity.



**Link mentioned**: <a href="https://qualtricsxmbyhfby7vf.qualtrics.com/jfe/form/SV_bDttqTVpxzTJvLw">Survey | Professional Network Platforms</a>: The most powerful, simple and trusted way to gather experience data. Start your journey to experience management and try a free account today.

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1276016243815743509)** (1 messages): 

> - `Multilingual Representation Learning Workshop at EMNLP 2024`
> - `Reviewer Sign Up`
> - `Workshop Topics` 


- **EMNLP 2024 Workshop Seeks Reviewers**: The Multilingual Representation Learning workshop at EMNLP 2024 is calling for reviewers.
   - Interested parties can sign up to be a reviewer [here](https://forms.gle/ica4F94jaTbvdb689).
- **Workshop Explores Diverse Multilingual NLP Topics**: The workshop explores a wide range of topics related to multilingual NLP, including computational social science and cultural analytics, dialogue systems, discourse and pragmatics, and low-resource methods.
   - Other topics include ethics and bias in multilingual models, information extraction and retrieval, multimodality, phonology, machine translation, resources and evaluation, cross-lingual semantics, and speech processing.



**Link mentioned**: <a href="https://forms.gle/ica4F94jaTbvdb689">Call for Reviewers - 4th Multilingual Representation Learning (MRL) Workshop, EMNLP 2024 </a>: This form is for anyone who is interested in being a reviewer for our workshop at EMNLP 2024.  Reviewers are invited to indicate their interest and their application will be assessed in relation to th...

  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1276013369727647825)** (2 messages): 

> - `Website Leaderboard`
> - `Huggingface Leaderboard`
> - `Model Evaluation`
> - `Equal Importance` 


- **Website vs. Huggingface Leaderboards**: A member asked about the difference between the website leaderboard and the Huggingface leaderboard.
   - The Huggingface leaderboard appears to have significantly higher scores.
- **Equal Importance of Model Abilities**: The website leaderboard has been changed to reflect the belief that all subcategories are equally important.
   - The ability in `python_simple_function` is considered as important as the ability in `java_simple_function`.
- **Comprehensive Model Evaluation**: A good model should excel in all aspects, not just specific subcategories.
   - This change was mentioned in #580.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1276201059714596908)** (1 messages): 

> - `Evaluating fine-tuned model on BFCL locally` 


- **Evaluating Fine-tuned Model on BFCL Locally**: A user inquired about the steps involved in evaluating a fine-tuned model on BFCL locally, specifically utilizing multiple GPUs for the evaluation process.
- **Running Evaluation Locally**: No specific steps or suggestions were provided in the conversation, so it's unclear how the user plans to evaluate their model on BFCL locally.


  

---



### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

amanshrestha: anyway we can use prompt caching , antropic api?
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1276236589286031380)** (1 messages): 

> - `Open Source AI`
> - `OSI Definition` 


- **OSI releases draft definition of Open Source AI**: The Open Source Initiative (OSI) has released a new draft definition of open source AI, a culmination of two years of discussions and debates within the technical and open source communities.
   - This milestone is crucial for redefining the meaning of "open source" in the context of AI and shaping the future of the technology's impact on society.
- **OSI Town Hall**: A Town Hall event was organized by OSI to discuss the new draft definition of open source AI.
   - The event was intended to provide a platform for further discussion and engagement from the community.


  

---



### **DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1276114551028584458)** (1 messages): 

> - `German Instruction Tuning Data` 


- **OASST-2 for German Instruction Tuning**: The **OASST-2 dataset** contains a German subset that is a good option for instruction tuning.
- **Aya-Dataset for German Instruction Tuning**: The **Aya-Dataset** also contains a German subset that can be used for instruction tuning.
- **Additional German Datasets**: Other German datasets like **Colossal Cleaned Common Crawl** and **German Wikipedia** might be helpful, but you'll need to filter and curate them for instruction tuning.
- **Custom Dataset Consideration**: Creating a **custom dataset** by translating English instruction data into German could be beneficial for specific tasks.
- **Open Sourcing Your Model**: Open sourcing your 8x8b Llama 3.1 based MoE, with both German and English instruction tuning, could be a valuable contribution to the NLP community.


  

---



---



{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

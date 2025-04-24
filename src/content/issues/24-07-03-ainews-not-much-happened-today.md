---
id: ccac20dd-1d94-42e9-aee5-2f3d79003fa4
title: Not much happened today.
date: '2024-07-03T22:39:42.336133Z'
original_slug: ainews-not-much-happened-today-1036
description: >-
  **Meta** unveiled Meta 3D Gen, a new system capable of generating high-quality
  3D assets from text in under *one minute*, marking a significant advancement
  in text-to-3D generation. Additionally, an open-source mixture-of-agents
  approach using models like Qwen, WizardLM, LLaMA, and Mixtral outperformed
  GPT-4o on the AlpacaEval benchmark with a score of *65.1%* versus *57.5%*,
  highlighting progress in open-source LLM performance.
tags:
  - meta
  - microsoft
  - perplexity
  - gpt4all
  - yi
  - magic
  - poolside
  - opendevin
  - kyutai
  - peter-thiel
  - langchain
  - qdrant
  - leap
  - sora
  - runway
  - meta-3d-gen
  - phi-3-mini
  - gpt4all-3.0
  - yi-large
  - qwen
  - wizardlm
  - llama
  - mixtral
  - rubra-v0.1
  - mistral
  - phi-medium
  - yi-1.5
  - llama-3-8b
  - gpt-4o
  - text-to-3d-generation
  - open-source-llms
  - mixture-of-agents
  - benchmarks
  - reinforcement-learning-from-human-feedback
  - persona-driven-data-synthesis
  - few-shot-generalization
  - steering-vectors
  - ai-workflows
  - memory-retrieval
  - vector-search
  - gain-of-function-research
  - ai-risk
  - idle-compute
  - tool-calling-llms
  - video-generation
  - ai-ethics
  - agi-power-centralization
companies:
  - meta
  - microsoft
  - perplexity
  - gpt4all
  - yi
  - magic
  - poolside
  - opendevin
  - kyutai
  - peter-thiel
  - langchain
  - qdrant
  - leap
  - sora
  - runway
models:
  - meta-3d-gen
  - phi-3-mini
  - gpt4all-3.0
  - yi-large
  - qwen
  - wizardlm
  - llama
  - mixtral
  - rubra-v0.1
  - mistral
  - phi-medium
  - yi-1.5
  - llama-3-8b
  - gpt-4o
topics:
  - text-to-3d-generation
  - open-source-llms
  - mixture-of-agents
  - benchmarks
  - reinforcement-learning-from-human-feedback
  - persona-driven-data-synthesis
  - few-shot-generalization
  - steering-vectors
  - ai-workflows
  - memory-retrieval
  - vector-search
  - gain-of-function-research
  - ai-risk
  - idle-compute
  - tool-calling-llms
  - video-generation
  - ai-ethics
  - agi-power-centralization
---


<!-- buttondown-editor-mode: plaintext -->**Honesty is all you need.**

> AI News for 7/2/2024-7/3/2024.
We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**418** channels, and **2896** messages) for you. 
Estimated reading time saved (at 200wpm): **341 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

[Arvind Narayanan et al published a paper](https://www.aisnakeoil.com/p/new-paper-ai-agents-that-matter) about how Agent papers are mostly not reproducible and ignore cost, [Meta published a text-to-3D assets model](https://x.com/AIatMeta/status/1808157832497488201?utm_source=ainews&utm_medium=email), [Magic.dev and Poolside](https://x.com/johnbyronhanby/status/1808235931784434049) are code model companies seeking unicorn rounds, OpenDevin is [now a company](https://x.com/gneubig/status/1808493521315496229), Kyutai released a [realtime Audio LLM](https://x.com/giffmana/status/1808482848808010149) that [maybe doesn't work as advertised](https://x.com/benhylak/status/1808611023123067357), Peter Thiel backed [some AGI Blockchain thing](https://x.com/sentient_agi/status/1808136737257918916), The New Stack published [one](https://thenewstack.io/lets-get-agentic-langchain-and-llamaindex-talk-ai-agents/) and [two](https://thenewstack.io/mozilla-llamafile-builders-projects-shine-at-ai-engineers-worlds-fair/) writeups of AIEWF. 

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

**AI Model Releases and Updates**

- **Meta 3D Gen**: [@AIatMeta](https://twitter.com/AIatMeta/status/1808157832497488201) introduced Meta 3D Gen, a new system for **end-to-end generation of 3D assets from text in <1min**, producing high-quality 3D assets with high-resolution textures and material maps. Details are available in the technical report.
- **Perplexity Pro Search Update**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1808183923064656383) announced an updated version of Pro Search that can **perform deeper research on more complex queries** with multi-step reasoning, Wolfram|Alpha, and code execution.
- **Phi-3 Mini Update**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1808087286661132494) shared that Microsoft updated Phi-3 mini with **significant improvements in long-context understanding, instruction following, and structured output**, all achieved by post-training improvements.
- **GPT4All 3.0**: [@andriy_mulyar](https://twitter.com/andriy_mulyar/status/1808170696717070667) announced GPT4All 3.0, **supporting 1000's of models and all major operating systems**, with major UI/UX improvements and Local File Chat with LocalDocs.
- **Yi-Large Launch**: [@01AI_Yi](https://twitter.com/01AI_Yi/status/1808262539681177681) celebrated one week since Yi-Large launched on the Fireworks AI Playground, asking for user feedback on the model.

**Research Papers and Techniques**

- **Reinforcement Learning from Human Feedback (RLHF)**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1808218688388321463) provided an overview of the **evolution of RLHF research**, tracing its roots to papers studying the use of human feedback for training summarization models. Key papers were linked.
- **Persona-Driven Data Synthesis**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1808096574997770590) shared a paper proposing a **persona-driven data synthesis methodology using Persona Hub**, a collection of 1 billion diverse personas, to create scalable and diverse synthetic data for LLM training and evaluation.
- **Meta-tuning for Few-shot Generalization**: [@slashML](https://twitter.com/slashML/status/1808205600045912104) shared a paper on "**Unleashing the Power of Meta-tuning for Few-shot Generalization** Through Sparse Interpolated Experts".
- **Steering Vectors**: [@sarahookr](https://twitter.com/sarahookr/status/1808237222522769410) shared work on **steering model behavior towards non-differentiable objectives** by constraining the generation process to explicitly steer towards minimization or maximization of non-differentiable features.

**Frameworks and Tools**

- **LangSmith**: [@LangChainAI](https://twitter.com/LangChainAI/status/1808154656746754114) shared a case study on how @newcomputer used LangSmith to **iterate quickly and improve memory retrieval**, leading to 50% higher recall and 40% higher precision for their agentic memory system, Dot.
- **Qdrant Engine v1.10**: [@qdrant_engine](https://twitter.com/qdrant_engine/status/1808121142961406156) released Qdrant engine v1.10 with new features like **Universal query API, Multivector search, Inverse Document Frequency**, and more.
- **Leap AI**: [@LeapAI_](https://twitter.com/LeapAI_/status/1808238079037395145) introduced their platform for **building custom AI workflows to automate content creation, lead generation**, and more, integrating state-of-the-art AI models like GPT-4.

**Discussions and Perspectives**

- **Gain of Function Research with AI**: [@JvNixon](https://twitter.com/JvNixon/status/1808201698466570372) expressed concern about "**gain of function research**" with AI, drawing parallels to bioweapons research and the potential dangers of creating teams trying to generate novel, dangerous outputs to prove whether models are safe or not.
- **Probability of Doom vs. Probability of Life**: [@JvNixon](https://twitter.com/JvNixon/status/1808267707747557807) argued that framing AI risk in terms of **p(doom) is a deep collective psychological mistake**, forcing people to imagine abstract superintelligence. They prefer p(life) - the probability of you and your loved ones surviving into the far future - as it brings in more of life and progress, and forces a balance of risks against benefits.
- **Idle Compute in AI Labs**: [@far__el](https://twitter.com/far__el/status/1808205077015875693) noted that many AI labs have **lots of idle compute sitting around**, as they need compute in bursts. This leads to things like heavily subsidized inference, redefining compute cost as a marketing expense.

---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!

**AI Models & Techniques**

- **Microsoft's Phi-3 Mini update**: In /r/LocalLLaMA, Microsoft updated their Phi-3 Mini model in both 4K and 128K context versions, showing [**significant improvements in instruction following and knowledge retention**](https://www.reddit.com/r/LocalLLaMA/comments/1dtgylv/microsoft_updated_phi3_mini/). Comments discussed renaming conventions, excitement about the model line's potential, and comparisons to Microsoft's product naming history.

- **Open-source mixture-of-agents outperforms GPT-4o**: In /r/LocalLLaMA, a mixture-of-agents (MoA) approach using only open-source models [**achieved 65.1% on AlpacaEval compared to GPT-4o's 57.5%**](https://www.reddit.com/r/LocalLLaMA/comments/1dtmqt5/open_source_mixtureofagents_llms_far_outperform/). The models used included Qwen, WizardLM, LLaMA, and Mixtral variants. Comments questioned the limited benchmarks, noted the expense of this method, and referenced a related video.

- **Rubra v0.1 introduces tool-calling LLMs**: In /r/LocalLLaMA, Rubra v0.1, a collection of open-weight, tool-calling LLMs, was introduced, including variants of [**Llama, Qwen, Mistral, Phi, and Gemma models aiming to provide reliable function calls**](https://www.reddit.com/r/LocalLLaMA/comments/1dtt32y/new_collection_of_llama_mistral_phi_qwen_and/).

- **MMLU-Pro benchmark critiqued as math-heavy**: In /r/LocalLLaMA, the MMLU-Pro benchmark was critiqued for being [**dominated by math and Chain-of-Thought reasoning, making it less useful for assessing general knowledge**](https://www.reddit.com/r/LocalLLaMA/comments/1du52gf/mmlupro_is_a_math_benchmark/). Suggestions included targeted subsampling and comparisons to MixEval. Comments noted MMLU-Pro's popularity for local testing and evaluating future SOTA models.

- **Small model comparisons on MMLU-Pro**: In /r/LocalLLaMA, small models like Llama 3 8B, Mistral 7B, Phi Medium, and Yi 1.5 9B were [**compared on the MMLU-Pro benchmark**](https://www.reddit.com/r/LocalLLaMA/comments/1du0rka/small_model_mmlupro_comparisons_llama3_8b_mistral/). Key takeaways highlighted Mistral's strong all-around performance and Llama 3's competitiveness despite quantization.

**AI Video & Animation**

- **AI-generated alien nature documentary**: An [**AI-generated video showcasing an alien nature documentary**](https://v.redd.it/f15k13mye2ad1) demonstrated the improved quality and watchability of AI-driven content.

- **Sora vs. Runway video generation comparison**: A [**comparison video between Sora and Runway's video generation capabilities**](https://v.redd.it/iy8jinx6w2ad1) showed that while close, Sora has better motion and overall quality. Comments discussed Runway's high contrast, Sora's non-existence, and potential cherry-picking.

**AI Ethics & Societal Impact**

- **Concerns over Kling spam**: In /r/StableDiffusion, a discussion arose about the [**increasing spam of Kling and RWML videos, suggesting astroturfing by these closed-source services**](https://www.reddit.com/r/StableDiffusion/comments/1dtrnu6/meta_discussion_kling_spam/).

- **AGI's impact on power centralization**: In /r/singularity, a poll asked whether [**AGI will lead to centralization or decentralization of power**](https://www.reddit.com/r/singularity/comments/1du2gj2/will_agi_lead_to_centralization_or/).

- **AI's role in student loan debt**: In /r/singularity, a question was posed about whether [**AI systems should pay off student loans for displaced workers or if UBI would be better**](https://www.reddit.com/r/singularity/comments/1dtmltm/if_ais_start_taking_all_the_white_collar_jobs/).

- **Mental health in AI research**: In /r/singularity, the Italian National Research Council called for participation in a study to [**understand mental health challenges faced by AI researchers**](https://www.reddit.com/r/singularity/comments/1dtj1lk/help_us_understand_mental_health_in_ai_research/) and develop support systems.

**Miscellaneous**

- **GPT4All 3.0 release**: [GPT4All 3.0, an open-source local LLM desktop application, was announced](https://x.com/nomic_ai/status/1808162955806097767).

- **AI-generated art showcases**: Various AI-generated art pieces were shared, including [insect typography created with Stable Diffusion 3](https://www.reddit.com/gallery/1dtluza), [transparent pixel art of Genshin Impact characters](https://i.redd.it/a9wsgbvfy6ad1.png), and [a workflow combining SDXL with SD3 refiner](https://www.reddit.com/gallery/1dty6rl).

---

# AI Discord Recap

> A summary of Summaries of Summaries

1. **Real-Time AI Models Steal the Spotlight**:
   - **Kyutai Labs** launched [**Moshi**](https://moshi.chat/?queue_id=talktomoshi), a 7B multimodal model for real-time text and audio generation with 160ms response times, garnering excitement for its open-source availability and rapid interactions (*albeit a bit robotic*), showcasing during a demo session with plans to address minor bugs.
   - The **Phi-3 Mini** model received a major update akin to a **3.5 Mini**, with upcoming [**Gemma 2**](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit) support, but users noted startup issues reflecting the integration challenges of cutting-edge AI tools.

2. **Optimizing AI Deployment and Memory Management**:
   - Extensive discussions on **Colab and Kaggle notebooks** shared best practices for memory management with methods like `gc.collect()` and `torch.cuda.empty_cache()`. Scaling LoRA rank for models based on dataset size was debated, emphasizing optimization via efficient resource handling.
   - **Gemma 2** support enhancements for tools like **Unsloth** and **LM Studio** improve finetuning speed significantly, with Unsloth achieving **2x faster finetuning** and **63% less memory** usage, while LM Studio’s 0.2.27 update solved compatibility issues on **Mac, Windows, and Linux**.

3. **Innovations in AI Model Training and Fine-Tuning**:
   - **QLoRA** was highlighted for its [**efficient finetuning**](https://arxiv.org/abs/2305.14314) of quantized LLMs, enabling finetuning of 65B parameter models on 48GB GPUs with near 16-bit precision performance using 4-bit quantization, as detailed in the **QLoRA** paper.
   - Members delved into **optimizing CUDA operations** with tools like **DeepSpeed** and **Inductor** backend for Nvidia, focusing on **autotuning GEMM backends** and troubleshooting `torch.cuda.OutOfMemoryError`, reinforcing the importance of hardware-informed optimizations.

4. **Privacy, Security, and Ethical Considerations in AI**:
   - Concerns over **data policy enforcement** led to critical discussions on **OpenAI’s GPT-4** subscription pricing and sporadic model parameter adjustments affecting user experience. Issues like dataset removal due to minor policy breaches sparked debates on enforcement consistency vs. user needs.
   - Discussions on **anti-AI art software** like [**Glaze**](https://glaze.cs.uchicago.edu/) and **Nightshade** raised ethical questions about balancing copyright protection and technological progress, highlighting community frustrations over potential circumvention of protective tools.

5. **Community Tools, Tutorials, and Collaboration**:
   - Users shared various open-source tools and tutorials, such as [**creating custom pipelines**](https://github.com/andysingal/llm-course/blob/main/transformers/custom-pipeline.md) with Transformers and [**Gradio apps**](https://huggingface.co/spaces/xtreme86/System_roleplay_generator) for role-play prompts, fostering collaborative learning and practical implementation.
   - **Docker image** development for AI tools like [**AI Town**](https://github.com/Ikkitsuna/AI-Town-Windows-Setup-WSL-method) saw active community participation, focusing on simplifying setup processes and ensuring compatibility with various platforms via detailed PRs and documentation submissions on GitHub.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi-3 Mini's Marvelous Makeover**: The Phi-3 Mini model underwent a significant update, akin to a 3.5 Mini, with the **Gemma 2** compatible quantized versions expected soon, as indicated in announcements on Unsloth AI 
   - Feedback from users suggests both excitement and startup issues with the new **Gemma 2** support in Unsloth, reflecting the teething problems of cutting-edge AI tool integration.
- **Moshi's Melodic AI Mastery**: Kyutai Labs launched '**Moshi**', a 7B multimodal LM, generating high-quality text and audio in real-time, achieving **160ms** response times and planned for open-source availability.
   - The AI community is abuzz over Moshi's capabilities, with its RLHF fine-tuning, backend versatility, and the anticipation of upcoming updates.
- **Colab's Capacity Climb**: Newly shared Colab/Kaggle notebooks offer extensive dataset support and introduce improvements such as scaling LoRA rank based on model and dataset size, garner Unsloth community's attention.
   - Members discussed best practices for memory management including `gc.collect()` and `torch.cuda.empty_cache()`, while acknowledging the need to pin resource-heavy notebooks for ease of use.
- **Secretly Secure Docker Deployments**: Discussions ensued about secure secret management in Docker deployments, with community consensus settling on the use of `--env-file` flag for environmental variables as a best practice.
   - Suggestions circulated for efficient container handling and deployment, such as using local registries and Docker commands like `docker save` and `ctr images import`.
- **Tackling Unsloth's Local Quirks**: Users report configuration issues when utilizing Unsloth locally, with recommended fixes involving updates to the `config` object to reflect changes in API.
   - Although Gemma2's anticipated update within 1-2 days stirred the community, ongoing discussions continue to highlight delays and the eager anticipation for improvements in PHI's JAVA evaluations.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4 Subscribers Grapple with Usage Caps**: Users voiced concerns over **GPT-4 subscriptions**, facing issues like quickly reaching message limits and decreased performance after upgrading. The community exchanged alternative approaches and highlighted the constraints of the model.
   - Debates emerged on the **subscription pricing**, as some users paying up to $60 a month challenged OpenAI's sporadic parameter adjustments, questioning the cost-effectiveness for professional tools.
- **AI21 Unleashes 'Jamba' with Hefty Hype**: **AI21 Labs** introduced ['Jamba'](https://www.ai21.com/blog/announcing-jamba), boasting a *hybrid of Mamba SSM technology and Transformer architecture*, flaunting its **256K context window** and competitive pricing, stirring animated conversations.
   - Discussions ensued on applying **Jamba** to coding tasks, with reports of mixed results compared to other AI models like GPT-4 and Claude, igniting dialogues on potential approaches for accuracy enhancements.
- **Open-Source AI Tools Enter the Fray**: The release of ['Moshi'](https://moshi.chat/?queue_id=talktomoshi), a tool for real-time AI-powered conversations that's open-source, caught the interest of many, despite its early-stage limitations.
   - The community weighed the pros and cons of **open-source AI tools** against proprietary models, discussing how these developments could influence the incorporation of AI into everyday technology.
- **Prompt Engineering's Depth Explored**: **Prompt engineering** surfaced as a key topic, with members sharing advice on honing prompts for more precise task performance with AI, especially for nuanced tasks like creating PDFs with **formatted product tags**.
   - Users tackled the intricacies of **DALL-E prompt engineering**, offering recommendations like prompt simplification and specificity to mitigate issues related to undesired image elements.
- **Nested GPTs Spark Curiosity and Debate**: In the realm of GPT development, a user's query about the feasibility of a **GPT calling other GPTs** opened up a discussion on the technicalities and hypothetical depth of such nesting functionalities.
   - The community also expressed dissatisfaction with data policy enforcement, pointing out the removal of a dataset involving a 15-year-old entry and sparking a conversation on the need for nuanced compliance versus strict guidelines.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio's Latest Gemma 2 Enhancements**: LM Studio 0.2.27 introduces improved support and compatibility for **Gemma 2** models with enhanced performance on **Mac, Windows, and Linux platforms**. Users are encouraged to [update to the new version](https://lmstudio.ai) for a seamless experience.
   - Community contributors like [abetlen](https://github.com/abetlen) have been instrumental in updating **Gemma 9B and 27B** models, which can be redownloaded from [Hugging Face](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF), ensuring compatibility with current setups.
- **Sailing Smooth on ROCM Seas**: A concerning error 'unknown model architecture: gemma2' has sparked conversation surrounding the new **LM Studio 0.2.27** release, with proposed solutions including a **clear cache or a complete re-install**.
   - Community testing on the **ROCM GPU compatibility performance** suggests success on models like **AMD Radeon RX 6900 XT**, with prompts to assist in validating the latest Linux ROCm extension pack for the updated software version.
- **Resolving Power Puzzles**: A deep dive into LM Studio's energy usage revealed a high idle consumption, prompting discussions on power efficiency and comparisons with other tools like [Blender](https://discord.com/channels/1110598183144399058/1253332613540876401) that suggest a need for optimizations.
   - Contrasts between operating systems emerged, as **Linux users noticed a gentler power draw from their GPUs** when running models, compared to the power surges reported by Windows users amidst similar activity.
- **Scaling Battles and Interface Improvements**: Feedback on LM Studio pointed out scaling issues on **1080p monitors**, restricting workflow efficiency due to a cramped interface, and highlighting the importance of layout optimization in multi-display environments.
   - Users proposed adding metadata such as publication dates to model listings on LM Studio's interface, a suggestion that garnered positive responses from the community.
- **Gradio App's Role-Play Revolution**: In pursuit of a richer role-playing experience, a user has pioneered a **Gradio app** with dynamic variables aimed at improving immersive character interactions, igniting a flame of innovation for AI-driven storytelling.
   - The application's ability to offer tailored prompts places it at the forefront, receiving an invitation for community feedback to enhance its capability, viewable at [this creative space](https://huggingface.co/spaces/xtreme86/System_roleplay_generator).



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Transformers 4.42: New Models on Parade**: The [Transformers 4.42 release](https://huggingface.co/docs/transformers/v4.42.0/release) debuts novel models like **Gemma 2**, improvements in tool usability, and fine-tuning capabilities, marking another stride in model progress.
   - `KerasNLP` now enables model enthusiasts to [integrate and fine-tune Transformers](https://github.com/andysingal/llm-course/blob/main/transformers/custom-pipeline.md) across platforms, broadening the landscape for machine learning applications and efficiency.
- **Data Abundance: AWS Chronos Datasets Go Public**: AWS releases comprehensive [Chronos datasets on HF](https://huggingface.co/datasets/chronos), complete with both pretraining and evaluation benchmarks, providing a rich resource for temporal analysis.
   - Researchers can dive into temporal patterns with the AWS datasets, potentially sparking data-driven insights and model innovations.
- **AI Expertise Development: Free Courses Emerge**: Prominent institutions like [Harvard University](https://harvard.edu/) offer free ML courses, boasting quality content and a pathway to certification.
   - These courses are a gateway for those aiming to elevate their ML proficiency without financial barriers, though the repetitive nature of basics is a consideration for prospective learners.
- **Community Engagement: New Roles and Resources**: HuggingFace's Discord community strengthens with ongoing discussions on the capabilities of large context window models like [Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct), indicating a heightened interest in nuanced text processing.
   - Comparisons draw between the efficiencies of **HF models** like Meta-Llama and proprietary giants, revealing a landscape where open models tackle the dominance of closed-source tools.
- **Diffusers vs. A1111: Model Quality Disputed**: Running the same generation parameters, users report **RealVisXL V4.0 Lightning** falls short in quality when using diffusers compared to A1111, despite identical setup.
   - Discussion centers on the trade-offs in quality between different execution methods, critical for achieving desired model performance in photorealistic tasks.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GPT-4's Colossal Capacity**: Nvidia’s Sneak Peek**: GPT-4's speculated parameter range of **1.7 to 1.8 trillion** raised eyebrows, dwarfing GPT-3's **175 billion**, with a [discussion involving Nvidia](https://www.nvidia.com), suggesting the company's close ties due to hardware support, despite NDAs.
   - Practical applications of **InstructGPT** showcased efficiency leaps by **10X to 100X**, credited to **Reinforcement Learning from Human Feedback (RLHF)**, generating a buzz about its potential.
- **Scaling Law Skirmishes**: Kaplan vs. Hoffmann Unraveled**: Community debates addressed the discrepancy in scaling laws posited by Kaplan et al. and Hoffmann et al., with new insights on last layer costs and warmup duration, detailed in an [arXiv paper](https://arxiv.org/abs/2406.19146).
   - The conversation highlighted potential flaws in the [PyTorch FLOP counter](https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py) and the importance of accurate FLOPs calculation methods for model scaling.
- **Interpreting Interpretability**: Sparse Circuits Come to Light**: The paper on EAP and integrated gradients inspired a probe into **sparce feature circuits**, an approach to dissect language model behaviors, aiming for a methodical interpretability pipeline outlined in [this work](https://arxiv.org/abs/2403.19647).
   - The SHIFT method for classifier generalization stoked curiosity, suggesting fine-grained interpretability units could ablate extraneous features, drawing insights from human judgement.
- **Perplexity in Preprocessing**: Navigating Long Documents**: **Stellaathena’s** config perplexity baffled others with its error in **proof-pile**, a stark contrast to the smooth operation with `lambada_openai`, sparking a conversation on ensuring efficiency and accuracy in model evaluations.
   - Technical chatter included the **loglikelihood_rolling** feature and its use in turning loglikelihood into loss values, as part of the forum’s continuous agility in model assessment.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Trying Gemini 1.5 Pro**: Users engaged in a discussion about **Gemini 1.5 Pro**, emphasizing its **large context window** and rapid response times. Recommended for its solid performance, the chatbot garnered positive feedback.
   - Concerns were also raised regarding **Perplexity's live internet access**, with mixed experiences reported on its ability to pull real-time data, causing frustration among users.
- **Navigating GPT4o Access Troubles**: Members highlighted challenges in accessing **GPT4o** freely, instead directing others to **Bing chat** and [**Claude 3.5 Sonnet**](https://claude.ai) as viable alternatives for free conversations, subject to usage restrictions.
   - The conversation included tips on **Perplexity's Pro subscription refund process**, with advice tailored to various regions such as the EU, UK, and Turkey.
- **Mobile Mastery with Perplexity**: Queries about **Perplexity's mobile app features** were clarified with confirmation of the inclusion of **Wolfram Alpha** and **code generation** capabilities on iOS.
   - A discourse on the importance of mobile features indicated a keen interest from users in the accessibility of advanced tools on handheld devices.
- **Sonnet's API Silence**: Discussions revealed that **Sonnet 3.5** is not supported by the **Perplexity API**, prompting users to consult the [official model documentation](https://docs.perplexity.ai/docs/model-cards) for alternative options.
   - Further to API capabilities, inquiries surfaced regarding the potential to leverage **Perplexity's search engine** through the API, with the community showing enthusiasm for access to these extended functionalities.
- **AI Blackbox Building Blocks**: Instructions and principles for creating a blackbox system in AI were provided, offering guidance on constructing these complex systems.
   - Material on topics including the **Lean Canvas** and the **founding of Perplexity AI** were shared, contributing to a broader understanding of strategic planning and entrepreneurial beginnings in the tech field.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Conclave Convenes**: A **CUDA-only hackathon** hosted by Ash Vardanian features **Chris Lattner** and is scheduled for **July 13th** at the AGI House in San Francisco, offering hands-on experience with **H100 accelerators**. [Details available here](https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf), courtesy of Nebius.ai.
   - In a separate event, Meta's **Hacker Cup 2024** gears up for a **September 20th** start, with Mark Saroufim urging developers to dive into the **code generation challenge**. Meanwhile, GPU enthusiasts are in a dilemma over the **NVIDIA 3090's $1,000 price tag**, as shared by Mark Saroufim who snagged a **4090 for $1,200**.
- **Matrix Multiplication Mastery**: **Mobicham** surfaces a guide to achieving over **1 TFLOPS performance on matrix multiplication** on a CPU platform, specifically tuned for the **AMD Ryzen 7700**, which surpasses NumPy's offering. [Tutorial can be found here](https://salykova.github.io/matmul-cpu).
   - The **3D V-Cache** technology garners attention for its contribution to AMD Ryzen's performance, sparking debates around its specialization beyond the augmented cache size, affecting **clock speeds and silicon layering**.
- **Integrator Ins and Outs**: Conversations unfold about compiling functions in **Pytorch** using the Inductor backend for Nvidia, mentioning [John Carmack's commendations](https://x.com/ID_AA_Carmack/status/1807072152631333060) for the PyTorch team while delving into **buffer loading and dequantization** processes with torchao.
   - A hiccup in forcing Inductor to generate **Triton kernels** for all operations is discerned, where **GEMM succeeds but Conv fails**, as detailed in a [GitHub issue](https://github.com/pytorch/pytorch/issues/125728) seeking resolution.
- **Model Memory Marvel**: Cutting-edge memory efficiency strategies put the limelight on this channel's models which comfortably manage batch sizes that would see PyTorch balking, emphasizing on models' **memory savings**.
   - A cited **GitHub Pull Request [#667](https://github.com/karpathy/llm.c/pull/667)** addresses decimal places in batch sizes during training which caused integer division errors, marking an incremental improvement.
- **Optimizer Odyssey**: A wave of optimism is evident with [Facebook Research's schedule-free optimizers](https://github.com/facebookresearch/schedule_free), claimed to demonstrate accelerated convergence across a spectrum of tasks, potentially reshaping optimization methodologies.
   - The community shares findings that suggest a significant uptick in the potential to fine-tune models without rigorous schedule adherence, teetering on the brink of what could be an optimization renaissance.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Artist's Allies Artifice Abated**: Community dialogue centered on the development of **anti-AI art software** like [**Glaze**](https://glaze.cs.uchicago.edu/) and **Nightshade** to protect artist's copyrights, yet several members voiced concerns about the ease of bypassing such tools.
   - The conversation underscored the challenge of maintaining the balance between **copyright protection** and technological advancement in AI training.
- **Pixel Perfection Predicament**: Inquiries regarding **16x16 pixel art** led to recommendations for training at **512x512** resolution, despite *Crystalwizard*'s remarks about possible trial and error in search of efficiency.
   - Emphasis was placed on experimentation in training methods to hone image generation for this specific art style, underscoring the granularity of AI model training.
- **Discord's Employment Depot Discussed**: Threads emerged questioning if the server had a dedicated **job-posting channel**, highlighting a surge in demand for **freelance and job opportunities** within the community.
   - Separate discussions pondered the ethics and logistics of **upwork account rentals** among freelancers, reflecting on the gig economy landscape in tech.
- **Prompt Prowess & Performance Puzzle**: Debates unfolded over various **prompting techniques** such as **[A|B], C** versus **[A, B, C]**, evaluating their impacts on image outputs, particularly when using models like **SD1.5** versus **segmoe** and **MixofExperts**.
   - Interest focused on refining techniques to achieve higher fidelity in text2img results, with discussions assessing the effectiveness of different syntactical approaches.
- **Model Melee: MixofExperts and segmoe**: Community evaluations detailed **segmoe** model's advancements in **prompt understanding**, showcased in applications like **ComfyUI**, and its perceived superiority over niche **SD1.5 finetunes**.
   - Comparative analyses by members illuminated the nuanced differences in performance and the quest for precision in natural language understanding among emerging models.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Models Morphing on OpenRouter**: OpenRouter announced changes including a **significant update** to the **/models page**, and adjustments in **Google Token Sizes for Gemini and PaLM models**—equating bigger tokens with GPT, and thus affecting pricing models.
   - A deprecation wave hits the OpenRouter: both the **Default Model** on settings page and **custom auth headers** for OpenAI API keys are set to be retired, steering towards newer practices and standards.
- **Claude 3.5's Connection Conundrum**: Users across the community have been experiencing **500 errors** when working with **Claude 3.5**, prompting some to pivot temporarily to alternate versions, like **Claude 3.0**, for stability.
   - Discussions on the OpenRouter touched on **privacy settings and logging policies** with varied provider stances; **NovitaAI** and **Infermatic** stood out for their commitment to not retain data, as highlighted by [Alex Atallah](https://openrouter.ai/settings/privacy).
- **Discussing LLM Precision**: AI Engineers speculated on the **quantization of LLM models** on OpenRouter, with debate centering around whether deployed models are using **FP16** or remain in their original precision unless specifically altered by providers.
   - Alternative frontends for leveraging Claude models, like **SillyTavern** and **LibreChat**, were debated for their efficacy, with suggestions such as **Typingmind** and **Pal Chat** being proposed for enhanced engagement.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cash Infusion Without Code at Magic.dev**: In a surprising financial leap, [Magic.dev](https://www.reuters.com/technology) surges to a **$1.5B valuation** with a mere assemblage of 20 staff, void of any product or revenue trails.
   - Unprecedented capital raise earmarked to position the emerging company as a formidable contender in the AI domain, setting a **new fundraising benchmark** for startup ventures.
- **The Billion Persona Playbook Unveiled**: Groundbreaking strides in synthetic data generation as [Persona Hub](https://arxiv.org/abs/2406.20094) integrates **1 billion personas**, yielding impressive enhancements on benchmarks.
   - [Aran Komatsuzaki](https://x.com/arankomatsuzaki/status/1807593343007818065) heralds the methodology, spotlighting its potency in generating quality synthetic data and bolstering diversity.
- **Real-Time Audio LLM 'Moshi' Speaks Up**: [Moshi](https://x.com/giffmana/status/1808482848808010149), heralded by Kyutai Labs, debuts as the inaugural real-time Audio LLM, demonstrating minimal latency yet **slightly robotic articulation**.
   - Despite its eagerness to reply causing occasional interruptions, the technology heralds a new frontier for user interactions with artificial intelligence.
- **All Hands on Tech: OpenDevin's Fresh Initiative**: The entrepreneurial minds behind [OpenDevin](https://x.com/gneubig/status/1808493521315496229) forge All Hands AI, committing to democratize AI software development via **open-source initiatives**.
   - The platform's foundation symbolizes a collaborative step towards universally accessible AI tools and a shared development ethos.
- **Sentient's Seed Success: Funding the Open AGI Quest**: Sentient announces an **$85M seed influx**, co-led by notables like [Peter Thiel](https://x.com/sentient_agi/status/1808136737257918916), to sculpt a community-driven AGI platform inviting global participation.
   - The ambitious funding is a clarion call for collective intelligence in creating an egalitarian AI ecosystem.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Decentralized Transformers Gain Ground**: **jaan.li** introduced their projects focusing on decentralized edge transformers at [onefact.org](https://onefact.org) and usb.club, sparking interest in their potential applications and contact for collaboration.
   - While **san.tosh** sought updates on open GPT-4o, the community's anticipation remained, with ongoing discussions but no concrete news.
- **Terminator Model Scrutiny Rises**: The community criticized the **Terminator** model's insufficient ablation tests and urged for a substantial justification of its changes, with a strong call for presenting detailed studies.
   - Yet, with its GitHub release, skeptics of the model were proved wrong as [Terminator's code went live](https://github.com/hyperevolnet/Terminator), allowing broader exploration and experimentation.
- **Vision Transformers' QKV Questioned**: A debate emerged on the necessity of QKV within Vision Transformers, with hypotheses suggesting potential redundancies and a need for empirical evaluation.
   - Alternative theories were shared and craved a rigorous review to shed light on the full impact of attention mechanisms within such architectures.
- **FORA Forges Faster Diffusion Transformers**: Introduction of **FORA** proposed to speed up Diffusion transformers by caching reusable computations, offering a solution to computational efficiency challenges.
   - The technique garnered attention for its potential to mesh with existing models deploying swift processing advancements as outlined in their [repository](https://github.com/prathebaselva/FORA?tab=readme-ov-file).
- **HyperZ⋅Z⋅W Paper Provokes Polarized Opinions**: **HyperZ⋅Z⋅W** paper was welcomed with mixed reviews, showcasing how a nascent submission can stir both acknowledgment and skepticism regarding new methods for SOTA achievements.
   - Despite criticism, there's an aura of curiosity around the novel ideas and potential revisions flagged by the HyperZ⋅Z⋅W paper, hinting at a growing discussion on QKV's impact in ViT as per Schmidhuber's [survey](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html).



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad's UNMUL Caught in a RuntimeError**: A **RuntimeError** was reported within tinygrad: *'failed to render UOps.UNMUL'* with efforts led by **George Hotz** to assert the condition that should *'never happen'*.
   - Discussions unfolded about making loop collapse optional, hinted by `flat_l4.realize()`, to avoid user impact and highlighted by **Chenyuy**’s proposed workaround.
- **Fuzzy Frontend: Tinygrad's Testing Takeover**: **Chenyuy** floated the notion of a **frontend fuzzer** for tinygrad, geared to root out edge cases using an approach similar to porting torch code with LLM.
   - The community buzzed about creating minimal repro tests for certain dimensions to address heuristic boundary quirks, leaving PRs open for ongoing deep dives.
- **Debug Dash Before Tinygrad 1.0**: The need for improved error messages in tinygrad crystallized with *Yosifrost* emphasizing pre-1.0 developer tool enhancements.
   - Community collaboration ensued to reproduce errors and devise test cases, setting the stage for more robust debugging mechanisms.
- **Gradient Gripes and Memory Mysteries**: AI engineers exchanged experiences of gradient accumulation mishaps leading to CUDA out-of-memory errors, with tips like detaching loss circling the forums.
   - TinyJit's shortcomings in optimization were highlighted, including **TinyJit's** failure to use `assert t.grad is not None` effectively, provoking a swift community response.
- **Tinygrad vs PyTorch: Tensor Creation Quirks**: The inconsistency of `Tensor.randn/randint` and `Tensor.full` between tinygrad and PyTorch sparked an analysis of tensor contiguity and proposals for alignment.
   - The behavior was chalked up as an idiosyncrasy unique to tinygrad, yet it didn't stymie discussion on refining future iterations for better compatibility.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Pinecone's Predicament and Potential Pivots**: A **DocumentSummaryIndex** creation snag hit users due to **Pinecone limits**, with a node's oversized metadata and improper **embed exclusion filters** as culprits, detailed in this [GitHub snippet](https://github.com/run-llama/llama_index/blob/722cb67ca4e52c8c4d6ef8c5e99b7f6c9f57e244/llama-index-core/llama_index/core/indices/document_summary/base.py#L203).
   - Potential fixes include **metadata limitation** and seeking alternatives like **qdrant** or **pg_vector**, as one user suggested, showcasing the community's problem-solving prowess.
- **RAG Revolution on Raspberry Rig**: @pavan_mantha1 showcased a **RAG pipeline** functioning on a **Raspberry Pi**, utilizing **Docker** and **Ollama**, sparking intrigue on how compact setups can still deliver, specified in this [community highlight](https://twitter.com/llama_index/status/1808292764129583179).
   - This feat emphasizes the adaptability of AI systems to resource-constrained environments and captures the guild's admiration for efficient computing.
- **Democratizing Documents with OpenContracts**: **OpenContracts** emerged as an open-source marvel for document analytics, leveraging **LLMs** for annotations, enabled by **Llama Index**. The tool's reveal is captured [on Twitter](https://twitter.com/llama_index/status/1808528869252812902).
   - **GenAI native** technology is at the forefront, with the project bidding to make **AI-powered document handling** widely accessible.
- **Weaving Wisdom with Webinar Wonders**: **Weights & Biases** partners for a webinar aimed at enlightening on **RAG pipeline** construction, critically analyzing a year of development, as elaborated [here](https://twitter.com/llama_index/status/1808589017744880062).
   - The event is pivotal in addressing evaluation challenges, underscoring a commitment to growth and knowledge sharing in AI application.
- **Agentic RAG Rouses Readers**: In the article [**Unleashing AI Potential**](https://medium.com/ai-advances/unleashing-ai-potential-agentic-rag-with-llamaindex-claude-3-5-sonnet-and-mongodb-ea126164a801), **Agentic RAG** couples with **LlamaIndex** and **Claude-3.5 Sonnet** over **MongoDB**, catalyzing conversations on avant-garde AI strategies.
   - Its imminent promotion signals a surge in interest for transformative approaches in AI infrastructures, ready to be explored by the keen minds of the guild.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Tortoise-TTS Snaps to GGML**: A community member has successfully migrated **Tortoise-TTS** to **ggml**, opening up opportunities for [real-time text-to-speech](https://github.com/balisujohn/tortoise.cpp) operations. The repo is enhanced with **CUDA and CPU support**, giving developers a wider platform choice.
   - This move invites AI developers to dive into optimizing **transformers** and **diffusion models** to quicken the inference process, making this an engaging project for those keen on performance enhancements.
- **vLLM's Tool Call Triumph in Hermes 2 Pro**: The integration of **tool calling in vLLM** for Hermes 2 Pro has been executed successfully, bringing the project closer to the finish line. This development invites fresh conversations about the efficient handling of 'content' and 'tool_calls'.
   - Discussions ensue around the incorporation of `<scratch_pad>` in **Hermann 3 training**, aiming at a more nuanced parsing methodology and aligning with standards akin to those seen in OpenAI's framework.
- **Instructional Ingenuity from Genstruct 7B**: The [**Genstruct 7B model**](https://huggingface.co/NousResearch/Genstruct-7B), taking cues from Ada-Instruct, has made its mark by generating precise instructions from documents, thus facilitating the creation of tailored datasets for instruction finetuning.
   - Geared towards AI engineers, this technique brings to the forefront the fusion of raw text corpora into conversational datasets, providing an intelligent solution for dataset expansion without hefty investments.
- **CommandR Rises in Huggingface's Hands**: **Huggingface** raised a [pull request](https://github.com/cohere/CommandR) for Cohere's CommandR, introducing advancements that refine tool-use and retrieval-augmented generation (RAG) techniques.
   - Their creative input revamps the system prompt using a combination of a preamble and smart content organization, facilitated by Jinja templates, indicating a strong collaboration potential in RAG developments.
- **GraphRAG: Graph-based Genius by Microsoft**: Microsoft has unveiled a novel retrieval-augmented generation framework known as [**GraphRAG**](https://github.com/microsoft/graphrag), focusing on modular designs to uplift efficiency in information retrieval and content generation.
   - Accessible on GitHub, GraphRAG stands as a signature offering thorough customization options which are imperative for today’s dynamic AI research and development landscape.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo on Ubuntu: Installation Tango**: Users faced hurdles with **Mojo** on **Ubuntu 24.04/Python 3.12.3**, encountering compatibility issues, particularly with **max-engine**. A [step-by-step guide](https://docs.modular.com/mojo/manual/python/#resolving-issues) for a successful installation with Python 3.11 was shared.
   - Discussion centered around `List[String]` lacking the `Stringable` trait, impacting printability, with detailed references on [GitHub](https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/builtin/str.mojo#L23). Users noted **variable startup times** in programs due to loop unrolling and its compilation time.
- **Strassen's Splendid Speed? Not Quite Stable**: The **Strassen Algorithm** was outperformed by a naive vectorized approach, hitting **70 GFlops** over Strassen's **50 GFlops** on 1024x1024 matrices, as per discussions and benchmarks shared on [GitHub](https://github.com/RedKinda/Mojo-Marathons/).
   - Concerns were raised over its **numerical stability**, with potential instability leading to test failures when adjusted for different types and sizes of matrices.
- **SPIRAL: Spinning New High-Performance Code**: The [SPIRAL project](http://www.spiral.net/) aims to automate the development of DSP algorithms, at times surpassing the performance of MKL. It's tailored for direct hardware tasks and could be key for optimizing an array of numerical operations.
   - Discussions highlighted the complexity of optimizing algorithms beyond parallel processing and vectorization, hinting at cache locality benefits from recursive approaches over iterative ones.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Apple Cracks OpenAI's Boardroom Door**: [Bloomberg](https://www.bloomberg.com/news/articles/2024-07-02/apple-to-get-openai-board-observer-role-as-part-of-ai-agreement) reported that **Apple** will secure a **board observer seat** at OpenAI, with **Phil Schiller** set to take the position, signaling **strategic moves** in tech collaborations.
   - Community analysis suggests **Apple's partnership** could yield greater perks than **Microsoft's investments**, spotlighting benefits like exclusive app integrations and piquing debates on corporate strategies in AI advancements.
- **Moshi Masters Multimodal Mantra**: **Kyutai Labs** stunned audiences with **Moshi**, a trailblazing **real-time audio LLM** boasting **150ms latency**, as acclaimed during its presentation, where it demonstrated **superior simultaneous translation** abilities and was recognized for its **speed** and **multimodal prowess**.
   - Plans to publish **open models** for community innovation were commended, including Moshi's core **7B multimodal LM** and **VQ-VAE codec**, which are poised to redefine on-device interactivity and user experience.
- **Code's Constitutional Conundrum**: Debaters invoked the [EFF's perspective](https://www.eff.org/deeplinks/2015/04/remembering-case-established-code-speech) on **SB 1047**, examining the defense of **model weights** and **code** as speech, drawing parallels to **freedom of expression** and **3D gun design precedents**.
   - Discussions surged around the essence of **model weights** as a form of expression, questioning if these algorithmic outputs should enjoy similar **protections as language**, emphasizing their integral role in modern communication and innovation.
- **Claude 3.5 Grows Its Fan Club**: A surge of excitement razed through the community with the release of **Claude 3.5**, drawing **enthusiastic responses** and comparisons with previous iterations, with professionals noting leaps in performance and potential application areas.
   - Advocacy for **Claude TM** likened its market positioning to the successful strategies of well-known brands, with members urging a boost in promotional efforts to match its reputable counterparts and to emphasize its **enhanced capabilities**.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Azure Agonizes with 429 Aches**: Switching to **AzureAIDocumentIntelligenceLoader** from **PyPDFium2Loader** led to a consistent **429 error (Too Many Requests)**, highlighting the rate limiting challenges faced.
   - Community debates included finding ways to circumvent Azure's **rate limiting** without sacrificing efficiency or accuracy.
- **PDF Puzzles & Markdown Mysteries**: Efforts to transform PDFs into markdown via [marker](https://github.com/VikParuchuri/marker) stumbled when facing complex table formats, with merged cells causing major migration malaise.
   - The allure of an open-source tool persists despite **Azure Document Intelligence** offering superior parsing precision, prompting a search for a local solution.
- **LangSmith's Lost Link**: Reports surfaced of **LangSmith** unexpectedly halting call traces, sparking discussions on the robustness of LangChain's introspective faculties.
   - Technical scrutiny ensued as users worked to detect defects in the **tracing mechanism**, hinting at hidden bugs in LangChain's infrastructure.
- **CriticGPT Cornering Coding Errors**: The AI community dissected OpenAI's **CriticGPT** initiative, aimed at identifying and amending mistakes from **GPT-4**, with a digestible [video explanation](https://youtu.be/4PgcaIfwLjo) circulating among peers.
   - Enthusiastic dialogues unfolded around how **CriticGPT** marks advancement towards self-correcting AI systems, envisaging upgrades in automated code reliability.
- **Mac Meets Toolio: Open Source Serendipity**: Mac enthusiasts rejoiced as **Toolio** broke onto the open-source scene, promising private **LLM** deployment on macOS, as heralded in its [YouTube showcase](https://www.youtube.com/watch?v=9DpQYbteakc).
   - The innovation empowers users with fast inference and **JSON schema output**, tuning into the demands for enhanced control and personalization.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Beef Up Your llamafile Linux Rig**: For optimal **llamafile** performance, engineers recommend GPUs like **3090/4090** for personal projects or **A6000/RTX 6000 Ada** for professional environments; and CPUs such as older **EPYC** for their superior core counts and PCIe support.
   - Discussions indicated a preference for GPUs with substantial VRAM, highlighting that 24GB VRAM is necessary to manage models around the size of **33B parameters**.
- **VRAM: The Bigger, The Better**: AI enthusiasts stressed the importance of excess VRAM to run sizeable models, with a cautionary note on employing FP16 mode as it ramps up VRAM usage compared to its minor quality gains.
   - Community exchanges underscored **q4** configurations that smoothly handle 33B parameters with **24GB VRAM**, setting a benchmark for large model management.
- **CPU Inference Wizardry with Syncthread**: Creative uses of the syncthread trick for CPU inference were spotlighted, potentially changing the way we approach **CPU-based learning**.
   - Links to a [YouTube talk](https://www.youtube.com/live/5zE2sMka620?feature=shared&t=3140) detailed the technique, capturing the community's attention.
- **Threadripper Tames llama3's 70B Model**: A sophisticated AI engineer reported on the successful operationalization of the **llama3 70B** model using a powerhouse **Threadripper CPU**, indicating potential leaps in CPU realistic applications.
   - This successful deployment signifies Threadripper's ability to hold its own in an arena dominated by GPU prowess.
- **Navigating llamafile on RK3588 NPU Challenges**: The integration of **llamafile** with **Rockchip RK3588 NPU** hardware sparked inquiries among practitioners, advising on software versions like **v0.8.9** to circumvent compatibility issues.
   - This discussion points to broader challenges and considerations necessary when leveraging specific versions for optimal hardware performance.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Weighing in on phi mini's new payload**: The **phi mini** has been updated with new weights, yet maintains consistency with its original repository, raising questions among users regarding the necessity for adjustments in the **torchtune** process.
   - Speculations persist on whether the legacy methods will hold up, but the consensus seems to lean toward a smooth transition without requiring significant changes.
- **Gradients & Epochs: Torchtune's Training Twists**: A vibrant discussion ensued on optimal **training strategies**, contrasting the use of **gradients 8 vs 16** and whether batch size adjustments, along with epoch variation, might yield superior outcomes.
   - To assist in unraveling this conundrum, **Wandb was employed to track and log performance metrics**, with community members sharing insights to refine the training process.
- **Conversion Conundrums: HF Format Finesse**: Queries have arisen about the nuances of model conversion, particularly why parameters like `num_heads`, `num_kv_headers`, and `dim` are requisite when transitioning between the multihead formats used by **HF** and **torchtune**.
   - The inherent complexity of format conversion was highlighted as members exchanged tips on effectively navigating this technical terrain.
- **Checkpoint Champion: Torchtune's Savior**: The introduction of **FullModelHFCheckpointer** into **torchtune** has sparked interest for its ability to seamlessly translate models into **HF-friendly formats**.
   - This tool has been lauded for bridging compatibility gaps between diverse machine learning infrastructures, ensuring broader accessibility and utility.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Checkmating Challenges with Stockfish & LLMs**: Community members are exploring the combination of **Stockfish** game data with **LLMs** to enhance strategic reasoning capabilities, with a side notion of developing a swift **chess engine**.
   - Discussions unfolded around the technical hurdles of fine-tuning **LLMs with chess data**, debating over its practicality and the risk of overfitting. The theory of using existing tools like **Stockfish** within **LLMs** was met with promising interest.
- **Slack Bot Draws Cohere**: A novel **Cohere Slack bot** was crafted, showcasing the ability to swiftly handle **Slack's** 3-second request demand, a testament to **Cohere's API** efficiency.
   - The creator's offer to share their code and produce documentation has sparked enthusiasm within the community, with many looking forward to detailed guidance on integrating **Cohere** with communication platforms.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Sound of Speed: Kyutai Moshi's Audio LLM**: [**Kyutai Moshi**](https://www.moshi.chat/?queue_id=talktomoshi) released a real-time **Audio LLM** that operates with virtually no delay, though feedback highlights a somewhat robotic tone. It's been heralded for fast interactions, sometimes eager to the point of interruption.
   - Insights from user *Mikebirdtech* underscore the system's speed, expressing it's **almost too fast** as it can interrupt users during natural conversational pauses.
- **See-Through Intelligence: OI Glasses Concept**: In a speculative conversation, user johnlenflure sparked the idea of integrating **OI** into eyewear, envisioning a future with **smart glasses** bolstered by **OpenInterpreter** capabilities.
   - No further details or technical discussion followed, leaving the concept at a high level of abstractive interest among members.
- **Game On for Open Interpreter Mods**: User **Nonadjective.eth_55058** is seeking advice on integrating **Open Interpreter** into a game, aiming to develop a working proof of concept, even if initially clunky.
   - This reflects a growing interest within the community to explore and expand the modding potentials for **Open Interpreter**, indicating a trend towards customizable interactive experiences.
- **Project Compatibility with Open Interpreter**: A list of projects, including **Open interpreter, taxyai, clickolas cage, self-operating computer, pywinassistant,** and **GPT computer assistant**, were highlighted as compatible with **Open Interpreter**.
   - Interest in exploring and possibly configuring these projects to work in tandem with **Open Interpreter** was evident, suggesting a dynamic and collaborative environment for developers.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Quantization Quandaries: LoRA vs QLoRA**: Members delved into quantization, discussing the diversity in its application between **LoRA** and **QLoRA**, highlighting that LoRA leverages 8-bit quantization and QLoRA pushes the envelope further with 4-bit, citing the comprehensive treatment in the [QLoRA paper](https://arxiv.org/abs/2305.14314).
   - A dialogue clarified **QLoRA's** positioning as superior in finetuning 65B parameter models on a single 48GB GPU with finesse, aligning performance closely with 16-bit finetuning, as revealed in the /*QLoRA: Efficient Finetuning of Quantized LLMs*/ paper.
- **VRAM Vexations and CUDA Calamities**: Colab conundrums surfaced with a user struggling with **torch.cuda.OutOfMemoryError**, noting the attempt to allocate 172.00 MiB on **Google Colab** resulted in failure.
   - Contributors concurred on **VRAM** being the bottleneck and suggested an increase in VRAM to facilitate seamless operation, spotlighting the hardware's vitality in the running of models like **axolotl**.



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Docker Docking at AI Town**: Community excited about a **Docker image** for AI Town, with a call for contributions to enhance the tool's accessibility.
   - The Docker effort seeks to streamline the setup process, as enthusiasts recommend pushing a well-received [Windows WSL setup guide](https://github.com/Ikkitsuna/AI-Town-Windows-Setup-WSL-method) as a pull request to the main repository.
- **API-port Drama in Dockertown**: A savvy developer encountered API communication issues while porting AI Town to Docker, specifically with **Ollama API**, and is committed to sharing a fix soon.
   - The technical hurdle doesn't deter progress as the port gains traction, and the community remains watchful for updates to ensure seamless connectivity.
- **Convex Catches a Docker**: In an effort to simplify the AI Town experience, a member is tweaking Docker to automatically download **Convex**, anticipating a smoother ride for future users.
   - The automated Convex setup via Docker is expected to be operational by 8 p.m. UTC+4, indicating proactive community involvement aimed at user efficiency.
- **AI Town's Docker Test Fest**: A member's initiative to run Docker integration tests on their **Legion Go** setup has led to confidence in the port's performance, suggesting readiness for a pull request.
   - Volunteers were sought for Docker integration testing with the expectation to merge successful results, demonstrating the collaborative ethos of the AI Town developer community.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Gradio Gridlock Grapples Engineers**: Members face hurdles deploying a **RAG app using Gradio** on Modal. A [discussion sprouted](https://discord.com/channels/1238365980128706560/1241044231829848125/1257903763998511155) about it working locally but not on Hugging Face Spaces.
   - **Modal Slack** was suggested as an emergency eject for the issue, hoping community support could provide a fix for the deployment dilemma.
- **DeepSpeed Dilemma Draws Debate**: Configuring **DeepSpeed** stirs up a storm among members attempting to enable **data sharding** without opting into model sharding, as seen in [their exchange](https://discord.com/channels/1238365980128706560/1242542198008975430/1257990993446436926).
   - Clarification and assistance with **DeepSpeed settings** became a pressing concern, highlighting a knowledge gap that needs bridging.
- **Hugging Face Handover Headache**: Troubles were aired over the inability to share **private code** deployments on Hugging Face, with **sharing=True** unsupported in private spaces, discussed [here](https://discord.com/channels/1238365980128706560/1242564125524234361/1257930601730674719).
   - Frustrations flared as attempts to operate on **Modal** also hit hitches, sparking a search for alternative methods for private code collaboration.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Legal Eagles Eye LLM Precision**: A new [report from Screens](https://www.screens.ai/blog/screens-accuracy-evaluation-report) analyzes LLM performance in contract reviews by equating it to a ML classification problem, boasting a **97.5% accuracy rate** for their system.
   - The challenges of assessing long-response accuracy are addressed, suggesting a classification-based methodology could enhance LLM effectiveness in legal tasks such as negotiation and document summarization.
- **Prompt Tuning For The People**: **Evan_04487** is seeking a straightforward, hosted prompt-tuning tool that's accessible to non-tech experts like designers and managers to run prompt variations and review outcomes.
   - The ideal solution would be a freemium service, easy enough for low-stakes, with the capacity to juggle about two dozen variables, contrasting the complex, self-managed infrastructure meant for critical tasks.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Datasette Discovery in Data Journalism**: Derek Willis shared [an article about foreign gifts](https://thescoop.org/archives/2024/06/22/all-foreign-gifts-around-us/index.html), sparking interest in **Datasette's utility** for investigative journalism.
   - The discussion involved how Datasette can be leveraged as a **powerful tool** for sifting through public records and datasets, emphasizing its role in **transparency and accountability** in journalism.
- **Datasette's Deep Dive into Data**: Enthusiasts highlighted **Datasette's** implications for deep data analysis, considering the tool's capacity for handling complex queries.
   - Engineers discussed the potential for **Datasette** to **transform data-driven stories**, underscoring the importance of accessible and interpretable public data in the digital age.



---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1257776054966157312)** (201 messages🔥🔥): 

> - `SEQ_CLS support in unsloth`
> - `Using jellyfin as an alternative to Plex`
> - `Fine-tuning models on another language`
> - `Sharing Colab notebooks and account information`
> - `VRAM requirements for LORA fine-tuning` 


- ****Phi-3 Mini gets a huge update****: The Phi-3 Mini model received a significant update, likened to a 3.5 Mini, with new quantized versions expected to be available tomorrow including Gemma 2 support. It's noteworthy that **Phi-3 Mini**'s latest enhancements will improve its performance, enabling quick and efficient processing.
- ****Moshi's real-time voice model excites AI community****: *Kyutai Labs* introduced '**Moshi**', a 7B multimodal LM that generates high-quality text and audio with low latency, achieving a 160ms response time. The model is fine-tuned with RLHF and is designed to be open source, supporting various backend configurations with plans for future updates.
- ****Gemma 2 support added to unsloth****: The unsloth team announced the release of Gemma 2 support, allowing users to fine-tune the model for advanced AI tasks with improved efficiency. Initial user feedback notes that **Gemma 2** is already working well via the provided notebooks, although some encountered initial setup issues.
- ****SEQ_CLS support and fine-tuning in unsloth****: Users discussed the functionality of SEQ_CLS support in unsloth for fine-tuning tasks, recommending JSON outputs for multi-class classifications. Experience with the Phi-3 model suggests significant improvements and learning speed using this approach.
- ****Discussion on integrating graph-based RAG into unsloth****: There was interest in integrating Microsoft's **graph-based Retrieval-Augmented Generation (RAG)** system into unsloth to enhance its capabilities. Users speculated on benefits, highlighting advancements in AI and optimized workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AIatMeta/status/1808157832497488201">Tweet from AI at Meta (@AIatMeta)</a>: 📣 New research from GenAI at Meta, introducing Meta 3D Gen: A new system for end-to-end generation of 3D assets from text in &lt;1min.  Meta 3D Gen is a new combined AI system that can generate high-...</li><li><a href="https://moshi.chat/?queue_id=talktomoshi">moshi.chat</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/en/developer_guides/model_merging">Model merging</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/en/developer_guides/lora#merge-adapters">LoRA</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit">unsloth/Phi-3-mini-4k-instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/live/hm2IJSKcYvo">Moshi Keynote - Kyutai</a>: no description found</li><li><a href="https://huggingface.co/internlm/internlm2_5-7b-chat">internlm/internlm2_5-7b-chat · Hugging Face</a>: no description found</li><li><a href="https://github.com/microsoft/graphrag">GitHub - microsoft/graphrag: A modular graph-based Retrieval-Augmented Generation (RAG) system</a>: A modular graph-based Retrieval-Augmented Generation (RAG) system - microsoft/graphrag</li><li><a href="https://x.com/reach_vb/status/1808528557431210236">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Fuck yeah! Moshi by @kyutai_labs just owned the stage! 🇪🇺/acc.  Architecture 1. 7B Multimodal LM (speech in, speech out) 2. 2 channel I/O - Streaming LM constantly generates text tokens as well as a...</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1258151227787706450)** (1 messages): 

> - `Gemma 2 release`
> - `Phi 3 mini update`
> - `Finetuning improvements`
> - `Increased context lengths`
> - `Notebooks and 4-bit models` 


- ****Gemma 2 Boosts Finetuning Speed****: Unsloth now supports **Gemma 2** with finetuning that's **2x faster** and uses **63% less memory**. Check out the [blog post](https://unsloth.ai/blog/gemma2).
- ****Extended Context Length Achieved****: You can finetune **Gemma 2 (27B)** with QLoRA to **9.7K context lengths** with Unsloth in a 40GB GPU, while HF+FA2 only allows 3K lengths. Unsloth also supports **11K context lengths** on a 24GB card for the 9B model.
- ****Free Colab Notebooks Available****: Free notebooks for **Gemma 2 (9B)** and **27B** are available, including a [Colab notebook](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing) for the 9B model.
- ****Phi 3 Mini Gets an Update****: **Phi 3 mini** also saw an update, with new [Instruct model](https://huggingface.co/unsloth/Phi-3-mini-4k-instruct-bnb-4bit) available on Hugging Face.
- ****Experiment and Share Results****: The community is encouraged to experiment, test, and discuss results of their models on the Unsloth platform. A specific call to share results was made **@here**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/gemma2"> Finetune Gemma 2 with Unsloth</a>: Fine-tune Google&#x27;s new Gemma 2 model 2x faster with 63% less memory VRAM via Unsloth! 9B and 27B parameters.</li><li><a href="https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1257918970460373063)** (3 messages): 

> - `mahiatlinux's comment`
> - `theyruinedelise's reaction`
> - `response from mahiatlinux` 


- ****Positive Engagement on Channel****: A member expressed their amusement by saying *'That's good lol'*, which sparked a short, positive exchange.
- ****Surprise and Agreement****: Another member agreed with the positive sentiment and mentioned being *'surprised to see it so good'*, which was further acknowledged.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1257847320863571998)** (63 messages🔥🔥): 

> - `Llama.cpp quantization issues`
> - `Loading model speed on Colab`
> - `Unsloth compatibility with Gemma 2`
> - `Inference on CPU after fine-tuning with Unsloth`
> - `Training issues with Huggingfaces SFTTrainer and Unsloth` 


- **Llama.cpp quantization puzzles members**: A user asked about unresolved quantization issues with **llama.cpp**, and another committed to prioritizing checking these issues today. Another user shared a related issue and linked to a [relevant Discord message](https://discord.com/channels/1179035537009545276/1179035537529643040/1257661677466554489).
   - *Still, get the same issue even after following the instructions.*
- **Speed up model loading on T4 GPUs**: Users discussed how to **speed up model loading** times on Colab T4 GPUs, specifically for a 7GB model. Suggestions included keeping the model in VRAM to avoid repetitive loading times, although some believed this was not possible on Colab.
   - "Every time when I am running cuda code to load model (7gb) from disk to gpu memory, it takes around 30 seconds. Can I make my model load faster or even better load model only once?"
- **JIT libraries with Jax**: A user asked about using **JIT libraries with Jax** for training on a prompt-answer dataset requiring linear algebra knowledge for GPU optimization. They also queried whether **RAG** is a better option than traditional fine-tuning, given extensive training times.
   - *Am I right in saying that? Not sure if someone has done that.*
- **Unsloth adds Gemma 2 support**: Users discussed various **errors and issues related to Unsloth**. Notably, it's been mentioned that **Unsloth has recently added support for Gemma 2**.
   - "Just now Unsloth added support of Gemma 2, you can update and try again!"
- **Issues with training using Huggingfaces SFTTrainer and Unsloth**: A user shared an error while trying to start training using **Huggingfaces SFTTrainer** and Unsloth locally, which led to a permissions issue with their user. Another referenced [a GitHub issue](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1038) related to missing Python.h.
   - *Is this in local? Best bet is just to google it.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1038">Make Error, fatal error: Python.h: No such file or directory compilation terminated. · Issue #1038 · CMU-Perceptual-Computing-Lab/openpose</a>: In file included from /home/sclab/Downloads/openpose/3rdparty/pybind11/include/pybind11/pytypes.h:12:0, from /home/sclab/Downloads/openpose/3rdparty/pybind11/include/pybind11/cast.h:13, from /home/...</li><li><a href="https://huggingface.co/docs/transformers/en/internal/generation_utils#transformers.TextStreamer)">Utilities for Generation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1257842275682095285)** (241 messages🔥🔥): 

> - `Tutorial/Intermediate Colab/Kaggle notebook with more dataset support`
> - `Improvements and suggestions for the community notebook`
> - `Memory management and optimization techniques for notebooks`
> - `Text classification optimized notebook for Unsloth`
> - `Secret management in Docker and application deployment` 


- ****Intermediate Colab Notebook by flail_****: A member introduced a [tutorial/intermediate colab/kaggle notebook](https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t) featuring support for many datasets (including dolphin, ultrachat, capybara, slimorca) and auxiliary functions with the capability of scaling LoRA rank according to model and dataset size.
   - Suggestions for improvements, like using `shuffle(seed=42)` for reproducibility and avoiding `flatten_indices()`, were discussed, and [multiple issues about memory management techniques](https://stackoverflow.com/a/55340037/3548976) using `torch.cuda.empty_cache()` were explored.
- ****Text Classification Notebook by timotheeee1****: A separate [modified Unsloth notebook](https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb) optimized for text classification was shared, enabling efficient evaluation using batched inference, and improving stability by fitting the classification head to specific token labels.
   - The notebook features input trimming for classification tasks, which saves VRAM and incorporates `ignore_index` to direct the model's capacity to necessary predictions without wasting resources.
- ****Memory Management Techniques Discussed****: Conversations included incorporating `gc.collect()` and `torch.cuda.empty_cache()` in loops to handle out-of-memory issues, with various members sharing their methods of managing memory more efficiently.
   - Members debated the usefulness of sleep intervals and loop constructs for memory clearance, fueled by code snippets from Unsloth.
- ****Docker Secret Management****: A member sought advice on handling secret management when deploying Docker containers without uploading `.env` files directly, eventually resolving to use the `--env-file` flag for passing environment variables securely.
   - Different approaches, like leveraging a local registry and using `docker save my-app > my-app.tar` followed by `ctr images import`, were discussed to ensure secure and efficient deployment workflows.
- ****Community Notebook Pins and Improvements****: It was recommended to pin significant notebooks, such as those for dataset support and text classification, to avoid losing them in scrollback and improve accessibility for community members.
   - These resources are also planned to be added to the Unsloth GitHub page upon further review and refinement.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stackoverflow.com/a/55340037/3548976)">How to clear CUDA memory in PyTorch</a>: I am trying to get the output of a neural network which I have already trained. The input is an image of the size 300x300. I am using a batch size of 1, but I still get a CUDA error: out of memory ...</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>: Scripts for text classification with llama and bert - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/huggingface/trl/issues/632#issuecomment-1972630547">[DataCollatorForCompletionOnlyLM] Are the input_ids supposed to contain the labels? · Issue #632 · huggingface/trl</a>: I&#39;m using DataCollatorForCompletionOnlyLM to train a chat assistant. I saw that the data collator contains the response that I want to fine-tune on (i.e. the batch[&#39;labels&#39;]) inside the ba...</li><li><a href="https://colab.research.google.com/drive/1njCCbE1YVal9xC83hjdo2hiGItpY_D6t?usp=sharing">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1257958621002731520)** (10 messages🔥): 

> - `Issues with using Unsloth in a local system`
> - `Gemma2 update release timeline`
> - `Support for the latest Gemma`
> - `Discussion about Gemma`
> - `Evaluation of Java with PHI` 


- ****Unsloth config error in local setup****: A user reported an error with config settings when using Unsloth on a local system: *`config.hidden_act` is ignored, use `config.hidden_activation` instead*.
- ****Gemma2 update expected soon****: A member mentioned that the update for **Gemma2** is expected to be released in **1-2 days**.
- ****Latest Gemma not supported yet****: The latest version of **Gemma** is currently not supported, necessitating a wait for the new update.
- ****Gemma release issues****: A member highlighted that **Gemma** should be released today, but there are delays due to blog issues.
- ****Java Eval with PHI noted****: A user remarked on PHI's strong performance for Java, hitting **93** in evaluation.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1257776610099200011)** (194 messages🔥🔥): 

> - `Issues with OpenAI GPT-4 subscription and performance`
> - `AI21's Jamba model announcement and discussion`
> - `User experiences with AI for coding and programming`
> - `Live and open-source AI models debate`
> - `AI for real-time conversations: Moshi demo` 


- ****Subscription woes plague GPT-4 users****: A user expressed frustration over the GPT-4 subscription, citing difficulties with reaching the message limit and performance issues post-upgrade. The community suggested alternative problem-solving approaches and highlighted the model's limitations.
- ****AI21's Jamba promises high-tech benchmarks****: AI21 Labs announced ['Jamba'](https://www.ai21.com/blog/announcing-jamba) with a cutting-edge hybrid architecture, combining Mamba SSM technology and Transformer architecture, touting a **256K context window** and attractive pricing.
- ****Coding hurdles with AI models****: Discussions revealed challenges with using various AI models for coding tasks—particularly in producing correct and complete code. Users shared mixed experiences with **GPT-4**, **Claude Sonnet 3.5**, and **Jamba**, reflecting varied performance in task accuracy.
- ****Open-source real-time AI tools****: The newly released ['Moshi'](https://moshi.chat/?queue_id=talktomoshi) real-time AI conversation tool sparked interest for its open-source promise, despite some limitations in its current model's intelligence.
- ****AI race heats up with open-source competition****: As discussions on **Moshi** and other open-source capabilities emerged, users debated the competitive edge these tools have over proprietary models like OpenAI's offerings. The race to integrate advanced AI into daily tech highlights the evolving landscape.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: Debuting the first production-grade Mamba-based model delivering best-in-class quality and performance.</li><li><a href="https://moshi.chat/?queue_id=talktomoshi">moshi.chat</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1257988005789171813)** (11 messages🔥): 

> - `New TTS model voices availability`
> - `Data policy enforcement`
> - `Running ChatGPT in command prompt with Google search capability`
> - `Subscription pricing frustrations`
> - `Nested GPTs functionality` 


- **Data Policy Enforcement gets Critiqued**: *virtue signalling* around data policy enforcement draws frustration as **ChatGPT** flagged and removed a dataset due to a single entry of a 15-year-old, despite the data having no inappropriate activity. Users are upset with the shift in message from "We might have made a mistake" to an authoritarian "We are right you are wrong REMOVED!"
- **Run ChatGPT in Command Prompt with Google Search**: A user successfully managed to run **ChatGPT** through the **command prompt**, enabling it to perform **Google searches** using Python coding.
- **Subscription Pricing Frustrates Users**: Users express frustration over **OpenAI's** ability to change model parameters sporadically despite some paying up to **$60 a month** for the service. There are mixed opinions on the value of professional tools costing $60, with some seeing it as justifiable for doubling productivity.
- **Query on Nested GPTs**: A user inquired about the possibility of having a **GPT** call on other **GPTs** and wondered about the limitations on how deep this nesting could go.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1257800705884819516)** (117 messages🔥🔥): 

> - `Issues with GPT models not answering questions as expected`
> - `Difficulties with creating PDF documents using GPT APIs`
> - `Improving prompt engineering for better task performance`
> - `Challenges with AI-driven image generation using DALL-E`
> - `Developing an employee recognition program using AI prompts` 


- ****GPTs Troubles with Instruction Following****: A user expressed frustration with GPT models not properly following detailed instructions, resulting in skipped steps and incomplete processes. Suggested solutions include updating prompts, using the regenerate option, or structuring instructions in smaller, sequential parts.
- ****Manual PDF Creation from AI Output****: A user faced issues generating a PDF with formatted product tags using AI, citing problems with adding logos and adjusting text size automatically. They opted to split the task into smaller manual edits after AI failed to meet their needs.
- ****Improving DALL-E Prompt Engineering****: A user sought assistance with improving prompts for generating vector icons using DALL-E, reporting issues like unwanted shadows and extraneous elements. Advice included simplifying and clarifying the prompt to avoid conflicts and ensure precise output.
- ****Detailed Prompt for Employee Recognition Program****: An elaborate prompt was shared for developing an employee recognition program, with clearly defined goals, recognition methods, criteria, an implementation plan, and a feedback mechanism. This structured approach aims to create a comprehensive and effective program.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1257800705884819516)** (117 messages🔥🔥): 

> - `GPT performance issues`
> - `Improving prompt structure and attention control`
> - `Converting documents into PDF with product tags`
> - `Enhancing an AI icon generator`
> - `Developing an employee recognition program` 


- ****Address GPT Performance Issues with Structured Prompts****: Members discussed improving GPT performance by structuring prompts effectively, such as using conditional imperatives like `IF...THEN...` and including clear, strong attention control mechanisms in the prompts.
- ****Tips on Converting Documents into PDF with Product Tags****: The conversation covered challenges in converting documents into PDFs with product tags, specifically regarding issues with fitting text and logos into predefined document rectangles.
- ****Enhancing AI Icon Generator Output****: A user sought help for an AI icon generator, highlighting issues such as unwanted background elements, shadows, outlines, and random objects being generated despite clear prompt instructions.


  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1257774458471710794)** (166 messages🔥🔥): 

> - `Discussing technical issues and updates related to LM Studio.`
> - `Comparing different AI models like Gemma 2, Llama 3, and Mistral.`
> - `Enhancements and bugs in the Gemma 2 model including tokenizer and attention mechanisms.`
> - `Difficulty and recommendations for fine-tuning LLMs using LM Studio.`
> - `Best models and configurations for different hardware setups.` 


- **Technical Issues and Updates in LM Studio**: Users reported and discussed various technical issues related to LM Studio, including problems with CPU utilization, difficulties with different versions, and challenges with certain prompts and response generation.
- **Gemma 2 and Other AI Models: User Experiences**: Members shared their experiences with the Gemma 2 model, noting significant improvements in writing style and performance, despite some lingering issues like ignoring system prompts. Comparisons were made with other models like Llama 3 and Mistral, with varied opinions on their effectiveness.
- **Enhancements and Bug Fixes in Gemma 2**: Discussions highlighted the latest updates to the Gemma 2 model, including attention layer fixes and tokenizer improvements. Users debated the necessity of redownloading models to benefit from these updates.
- **Fine-Tuning and Model Recommendations**: Users expressed interest in fine-tuning LLMs for specific tasks, though it was clarified that LM Studio currently does not support fine-tuning. Recommendations were given for using other methods and tools like RAG for specific customizations.
- **Best Models for Different Hardware Setups**: Recommendations were given for using the best models based on hardware configurations, with suggestions to use models that allow full GPU offloading for laptops and other devices with limited VRAM. Specific suggestions included Gemma 9B for 7B-size requirements and the unreliability of current gaming laptops for running LLMs effectively.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://huggingface.co/bartowski/gemma-2-9b-it-GGUF">bartowski/gemma-2-9b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=Y08Nn23o_mY">How to give AI &quot;Memory&quot; - Intro to RAG (Retrieval Augmented Generation)</a>: This is an intro video to retrieval-augmented generation (RAG). RAG is great for giving AI long-term memory and external knowledge, reducing costs, and much ...</li><li><a href="https://huggingface.co/bartowski/WizardLM-2-8x22B-GGUF/tree/main">bartowski/WizardLM-2-8x22B-GGUF at main</a>: no description found</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8197">Add attention and final logit soft-capping, update scaling factor to Gemma2 by abetlen · Pull Request #8197 · ggerganov/llama.cpp</a>: This PR adds the missing attention layer and final logit soft-capping. Implementation referenced from huggingface transformers. Additionally Gemma2 applies a pre-attention scaling of hidden_size / ...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1257779517288611902)** (52 messages🔥): 

> - `dolphin-vision compatibility in LM Studio`
> - `Gemma 2 model performance and issues`
> - `System and hardware requirements for running large models`
> - `RP stress testing for AI models`
> - `Code generation capabilities of Gemma 2` 


- ****Dolphin-Vision compatibility questioned****: A user questioned whether [dolphin-vision](https://huggingface.co/cognitivecomputations/dolphin-vision-72b) works with LM Studio, expressing concerns about its format and memory requirements.
- ****Gemma 2’s erratic behavior****: Users reported issues with the **Gemma 2** model, including repeating symbols and failing at various context lengths. Even with different settings, users noted it sometimes does not function as expected.
- ****RPing to stress test models****: A user shared insights on stress testing models through role-playing (RP), emphasizing it as a method to reveal a model’s faults. They proposed measuring models by the percentage of edited output needed during an RP session.
- ****Gemma 2 praised for introspectiveness****: Despite some issues, users praised **Gemma 2** for its introspective abilities and strong context understanding. It was compared favorably to other models in terms of hidden context detection and correctness.
- ****Code generation improvements sought****: Users discussed the challenge of models inserting placeholders in code output, even with explicit instructions not to do so. Suggestions included using detailed system prompts to guide the model’s code generation more effectively.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1257798297414795357)** (1 messages): 

> - `LM Studio 0.2.27 Release`
> - `Improved Gemma 2 Support`
> - `Bug Fixes in lmstudio.js`
> - `Advanced Information on lmstudio.js` 


- ****LM Studio 0.2.27 Launch Celebrates Enhanced Gemma 2 Support!****: LM Studio 0.2.27 is now available for **Mac (M1/M2/M3), Windows (x86 and ARM64), and Linux (x86)**. Users can [download it](https://lmstudio.ai) or restart their app to trigger the auto-update.
- ****Gemma 2 Models Revamped****: Performance improvements for **Gemma 9B** and **Gemma 27B** models are thanks to contributions from [abetlen](https://github.com/abetlen), [ngxson](https://github.com/ngxson), [slaren](https://github.com/slaren), [ggerganov](https://github.com/ggerganov) and others. Download the updated models from the Hugging Face community page: [Gemma 9B](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF), [Gemma 27B](https://huggingface.co/lmstudio-community/gemma-2-27b-it-GGUF).
- ****Bug Fixes in lmstudio.js****: [lmstudio.js] team has fixed the **"invalid creation parameter" bug** ([issue #45](https://github.com/lmstudio-ai/lmstudio.js/issues/45)). Additional updates addressed messages about **no GPU support**.
- ****Advanced Information for Power Users****: The latest `llama.cpp` commit ID is **d08c20eddedb24515a3212e2de66bdff41a26b8c** and **OpenCL backend** is now bundled again for both Windows and Linux platforms. However, Gemma 2 is **not supported for OpenCL**.
- ****Updating AMD ROCm Extension Pack on Windows****: Windows users with AMD ROCm can follow the [instructions](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#amd-rocm) to update their ROCm extension pack. The Linux ROCm extension pack update is still **under development**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://github.com/lmstudio-ai/lmstudio.js/issues/45)">Issues · lmstudio-ai/lmstudio.js</a>: LM Studio TypeScript SDK (pre-release public alpha) - Issues · lmstudio-ai/lmstudio.js</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#amd-rocm).">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1257933929202319360)** (11 messages🔥): 

> - `Basic functionality issue with drive installation`
> - `User confusion on model insertion`
> - `Scaling issues on 1080p monitors`
> - `Unsupported architecture message in new models`
> - `User feedback on improving LM Studio interfaces` 


- ****Drive Installation Gripes****: Members expressed frustration that LM Studio cannot be installed on a chosen drive due to [Squirrel limitations](https://link.to.issue). Instead, users must change the 'My Models' folder to handle storage issues.
- ****Model Insertion Confusion****: A user struggled with inserting a model, citing error messages about model operation failures despite having adequate system resources, such as 15.90 GB RAM and an NVIDIA GeForce GTX 950M.
- ****Scaling Issue on 1080p Monitors****: A user mentioned that LM Studio does not scale well on a 1/4 section of a 1080p monitor, leading to missing settings buttons and poor layout. This impacts multi-monitor workflows.
- ****Unsupported Architecture Alerts****: Members observed that new models often trigger 'unsupported arch' alerts, similar to issues with DeepSeek Coder v2, due to their configuration files.
- ****Call for More Metadata****: A user suggested adding the publication date and additional metadata for models on LM Studio's homepage to enhance usability. This was well-received as a potential improvement.


  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1257907784893337711)** (3 messages): 

> - `Prompting Llama 3 70B to remove conversational lines`
> - `Issue with prompt results on Llama 3 compared to Qwen2 72B`
> - `Creation of a prompt tool with Gradio app for role-play and character immersion` 


- ****Prompt Llama 3 70B without annoying conversational lines****: A user inquired about prompting **Llama 3 70B** to skip the trite conversational lines at the beginning of its responses. Example phrases like *'What a wonderful thing it is to have a drink!'* were highlighted as unnecessary.
- ****Prompt success on Qwen2 72B vs Llama 3****: One user shared their success in removing conversational lines from **Qwen2 72B**, contrasting with their struggles to achieve the same results on **Llama 3**.
   - They expressed frustration, noting the challenge in getting similar prompt performance from **Llama 3** despite applying the same techniques.
- ****New Gradio app for role-play prompts****: A user introduced a new **Gradio app** designed to create role-play prompts for immersive character experiences. The app includes dynamic variables to define character roles and scenarios.
   - They shared a sample prompt and invited feedback for improvement, providing a [link to the app](https://huggingface.co/spaces/xtreme86/System_roleplay_generator).



**Link mentioned**: <a href="https://huggingface.co/spaces/xtreme86/System_roleplay_generator">System Roleplay Generator - a Hugging Face Space by xtreme86</a>: no description found

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1258108843963973642)** (3 messages): 

> - `Energy consumption of LMS`
> - `Hardware issues on Linux vs Windows`
> - `GPU usage comparison of LMS with other software`
> - `User experiences with different GPU setups`
> - `Potential future fixes for LMS energy consumption` 


- ****Energy consumption in LMS at idle****: A member reported unreasonable energy consumption in LMS at idle, noting that the power usage should be **half per GPU** at about 10W. They highlighted [Blender](https://discord.com/channels/1110598183144399058/1253332613540876401) as a better comparison than web browsing for GPU usage.
- ****Windows vs Linux: Differing power consumptions****: A member experienced no significant hardware issues with LMS on **Linux** but acknowledged possible differences on **Windows**. They observed that running LMS with a model loaded only made a 33-watt difference compared to running Firefox or Chrome, with an additional 22-watt difference when no model was loaded.


  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1257805109182595216)** (14 messages🔥): 

> - `Gemma 2 loading issue`
> - `ROCM GPU compatibility and performance`
> - `Linux ROCm extension pack testing` 


- ****Gemma 2 fails to load in LM Studio 0.2.27****: A user encountered an **error loading Gemma 2** with message: 'unknown model architecture: 'gemma2'' despite clearing the cache and running necessary scripts. Suggested fix included a **re-download or clean install**.
- ****ROCM support success on AMD GPUs****: Discussion revealed successful use of ROCM on **AMD Radeon RX 6900 XT** and **7800 XT** GPUs, with testimony of running **Gemma 2 at 8k tokens** without RAM issues. Another user confirmed that ROCM builds work fine with these models.
- ****Call for Linux ROCm extension testing****: A community member called for assistance in testing the **latest Linux ROCm extension pack for version 0.2.27**. Instructions provided included installing via script and confirming **ROCm llama.cpp** appears under settings.


  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1257837864683700244)** (1 messages): 

> - `Gemma 2 model update on Huggingface`
> - `Compatibility updates for Gemma 2 models` 


- **Gemma 2 Models Updated for Compatibility**: The **Gemma 2 models** on the [lmStudio community Huggingface](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF) have been updated to the latest changes and are safe to redownload and use with version **0.2.27**.
- **Gemma 2 Models Ready for Use**: The updated **Gemma 2 models** are now compatible with the latest version **0.2.27** and can be downloaded from [Huggingface](https://huggingface.co/lmstudio-community/gemma-2-27b-it-GGUF).


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1257773145008640070)** (70 messages🔥🔥): 

> - `gpuOffload value discussion`
> - `bot configuration issues with TypeScript and Discord.js` 


- ****[Fixing discord token issue](https://github.com/mrdjohnson/lmstudio-discord-bot)**: New bot fails due to invalid token**: **Aquora** initially struggles with an [invalid token error](https://github.com/mrdjohnson/lmstudio-discord-bot) while configuring a discord bot. The issue is eventually traced back to **disallowed MessageContent** intents, which is resolved by enabling them in the Discord Developer Portal.
- ****Bot loops & 'thinking' state fix**: Temperature and predicted tokens adjustment solves bot hallucinations**: **Aquora** experiences issues with the bot 
   - **DJ** suggests adding message history and direct message handling in a future article to improve bot functionality. **Aquora** is eager to contribute to further improvements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/0xAquora/Lmstudio-discordjs-chatbot">GitHub - 0xAquora/Lmstudio-discordjs-chatbot: This is a personal test made taking this one: (https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6) as example</a>: This is a personal test made taking this one: (https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6) as example - 0xAquora/Lmstudio-discordjs-chatbot</li><li><a href="https://github.com/mrdjohnson/lmstudio-discord-bot/tree/main">GitHub - mrdjohnson/lmstudio-discord-bot: A tutorial for creating a Discord bot that responds using LM Studio! This code is based on a blogpost found here: https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6</a>: A tutorial for creating a Discord bot that responds using LM Studio! This code is based on a blogpost found here: https://dev.to/mrdjohnson/i-made-a-discord-bot-with-lmstudiojs-4fd6 - mrdjohnson/lm...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1257796484473294908)** (1 messages): 

> - `New fine-tunes for Transformers models with KerasNLP`
> - `Experimental API for searching HF datasets by column names`
> - `Transformers 4.42 release with new features and models`
> - `Nearly 100k public models on HF Hub storing tensorboard logs`
> - `Local Gemma release` 


- **Transformers 4.42 release introduces new models and features**: The new [Transformers 4.42](https://x.com/osanseviero/status/1806440622007447631) release includes **Gemma 2**, RT-DETR, InstructBlip, LLaVa-NeXT-Video, **tool usage and RAG support**, GGUF fine-tuning, and **quantized KV cache**.
- **KerasNLP bridges fine-tuning for any Transformers model**: Access [tons of new fine-tunes](https://x.com/julien_c/status/1806366482269352232) for any **Transformers model** using a **KerasNLP** implementation.
- **AWS releases Chronos datasets on HF**: [AWS](https://x.com/solitarypenman/status/1806421605683232947) released all datasets used in the Chronos paper on **Hugging Face**, including both pretraining and evaluation datasets.
- **Local Gemma offers 100% private and secure generation**: The new [Local Gemma](https://x.com/reach_vb/status/1807830966515519667) is 100% local, **private and secure**, and can run anytime with `pip install local-gemma`.
- **Vision language models introduction released**: [Intro to vision language models](https://x.com/mervenoyann/status/1805910433024380978) has been released showcasing image-text-to-text models.
   - This includes tasks like **image captioning**, optical character recognition, and more.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/julien_c/status/1806366482269352232)">Tweet from Julien Chaumond (@julien_c)</a>: Keras 🤝 HF</li><li><a href="https://x.com/vanstriendaniel/status/1807814430262202465)">Tweet from Daniel van Strien (@vanstriendaniel)</a>: Search @huggingface datasets by column names with a new experimental API! This API allows you to:  - Search for question-answering datasets that include context - Find alpaca-style datasets - Locate D...</li><li><a href="https://x.com/osanseviero/status/1806440622007447631)">Tweet from Omar Sanseviero (@osanseviero)</a>: Transformers 4.42 is out, and it has lots of amazing features🥳  🔥New models: Gemma 2, RT-DETR (obj detection), InstructBlip, and LLaVa-NeXT-Video 🔧Tool usage and RAG support 👀GGUF fine-tuning 🤏Qu...</li><li><a href="https://x.com/Wauplin/status/1808074557128855750)">Tweet from Wauplin (@Wauplin)</a>: Almost 100k public models uses the Hub to store 𝚝𝚎𝚗𝚜𝚘𝚛𝚋𝚘𝚊𝚛𝚍 logs! Storing training logs alongside checkpoints let you keep track of everything in a single place using the Metrics tab&#39; �...</li><li><a href="https://x.com/reach_vb/status/1807830966515519667)">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Introducing Local Gemma! 💎  100% local, private and secure - run anywhere, anytime!  cuda, mps, cpu - with presets to go as fast as possible - built on the shoulders of transformers, we ensure 1:1 ge...</li><li><a href="https://x.com/solitarypenman/status/1806421605683232947)">Tweet from Abdul Fatir (@solitarypenman)</a>: 🚀🚀🚀 We just released all the datasets used in the Chronos paper on Hugging Face.  This includes both pretraining and evaluation (in-domain and zero-shot)  datasets. We also open-sourced a script to...</li><li><a href="https://x.com/mervenoyann/status/1807790959884665029)">Tweet from merve (@mervenoyann)</a>: Real-time DEtection Transformer (RT-DETR) landed in @huggingface transformers 🤩 with Apache 2.0 license 😍  do DETRs Beat YOLOs on Real-time Object Detection?  keep reading 👀</li><li><a href="https://x.com/xenovacom/status/1805990110065803492)!">Tweet from Xenova (@xenovacom)</a>: Florence-2, the new vision foundation model by Microsoft, can now run 100% locally in your browser on WebGPU, thanks to Transformers.js! 🤗🤯  It supports tasks like image captioning, optical characte...</li><li><a href="https://x.com/mervenoyann/status/1805910433024380978)">Tweet from merve (@mervenoyann)</a>: Just shipped: intro to vision language models (aka image-text-to-text)</li><li><a href="https://x.com/ben_burtenshaw/status/1806291858835837333)">Tweet from Ben Burtenshaw (@ben_burtenshaw)</a>: 🚀 Excited to launch our new series, Data Explorer by @argilla_io ! 🎥  We dive deep into datasets and their impact on model performance. Our first episode explores the PRISM dataset by @hannahrosekir...</li><li><a href="https://x.com/TheZachMueller/status/1807394438689214930)">Tweet from Zach Mueller (@TheZachMueller)</a>: How do you make @PyTorch dataloaders work efficiently during distributed training? Here&#39;s a video tutorial I did with @huggingface accelerate&#39;s dataloaders showing how we do so  https://www.yo...</li><li><a href="https://x.com/mervenoyann/status/1806267855559623115)">Tweet from merve (@mervenoyann)</a>: New RAG with Gemma recipe using @elastic search, @huggingface 🧑🏻‍🍳📖   Find it below ⇓
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1257776252471742606)** (236 messages🔥🔥): 

> - `Joining Hugging Face Discord Community`
> - `Adept Strategy Shift & Co-Founders Joining Amazon`
> - `AI Models for Text and Image Processing`
> - `Performance and Accuracy of Hugging Face Models`
> - `Suggestions for ML Certifications` 


- ****Invite to Join HuggingFace Discord**: <a [Discord Community](https://huggingface.co/discord-community) link was shared, clarifying past verified members will receive invitations soon.**: A user inquired about joining the [HuggingFace Discord Community](https://huggingface.co/discord-community); invitations will be sent to all past verified members soon, emphasizing a shared collaborative space for live projects.
   - *Another user expressed interest in moderating the community due to their engagement in reporting scams.*
- ****Promising AI Models for Text Processing**: Users discussed various AI models like Qwen with **132k context windows**. Accurate [models like Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct) are highlighted for their impressive context lengths and usefulness in detailed text processing.**: HuggingFace's user community discussed AI models capable of handling extensive context windows, recommending [Qwen2](https://huggingface.co/Qwen/Qwen2-7B-Instruct) with a 132k context window for detailed text processing tasks.
   - Experimentation with long contexts versus more concise model outputs was suggested, considering performance and quality in various contexts.
- ****Comparing Open Source and Proprietary AI Models**: Users compare HuggingFace models like Meta-Llama to OpenAI's GPT**. A [Meta-Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) model was recommended for superior performance in benchmarks.**: A comparison was made between HuggingFace models and proprietary models like OpenAI's GPT, with the [Meta-Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) model being recommended for benchmark superiority.
   - *Users noted the balancing act between speed and reliability with various models, indicating some use cases still prefer proprietary tools for their efficiency.*
- ****Learning Resources and Certifications in ML**: Suggestions were made for free and efficient online courses for proving ML proficiency**. Harvard and Coursera courses were recommended for their comprehensive content and certification credibility.**: Several users shared their experiences with [free courses from Harvard](https://harvard.edu/) and Coursera, noting their balance of quality and certification credibility, beneficial for those looking to prove ML proficiency.
   - *One user asked about skipping repetitive basics in these courses, highlighting a preference for progressive learning models.*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://swtokyo.com/">Startup Weekend Tokyo</a>: no description found</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/semantic-chunker/">Semantic Chunking | 🦜️🔗 LangChain</a>: Splits the text based on semantic similarity.</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://www.youtube.com/live/hm2IJSKcYvo">Moshi Keynote - Kyutai</a>: no description found</li><li><a href="https://huggingface.co/spaces/gokaygokay/Florence-2">Florence 2 - a Hugging Face Space by gokaygokay</a>: no description found</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - a Hugging Face Space by mteb</a>: no description found</li><li><a href="https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt">Transformers, what can they do? - Hugging Face NLP Course</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/tree/main">meta-llama/Meta-Llama-3-70B-Instruct at main</a>: no description found</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B-int4">THUDM/cogvlm2-llama3-chat-19B-int4 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/tree/main">mistralai/Mixtral-8x7B-Instruct-v0.1 at main</a>: no description found</li><li><a href="https://moshi.chat/?queue_id=talktomoshi">moshi.chat</a>: no description found</li><li><a href="https://huggingface.co/spaces/LanguageBind/Video-LLaVA">Video LLaVA - a Hugging Face Space by LanguageBind</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct">microsoft/Phi-3-vision-128k-instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2-7B-Instruct">Qwen/Qwen2-7B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://www.microsoft.com/en-us/research/blog/graphrag-new-tool-for-complex-data-discovery-now-on-github/">GraphRAG: New tool for complex data discovery now on GitHub</a>: GraphRAG, a graph-based approach to retrieval-augmented generation (RAG) that significantly improves question-answering over private or previously unseen datasets, is now available on GitHub. Learn mo...</li><li><a href="https://x.com/reach_vb/status/1807830966515519667https://x.com/reach_vb/status/1806731975618626004">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Introducing Local Gemma! 💎  100% local, private and secure - run anywhere, anytime!  cuda, mps, cpu - with presets to go as fast as possible - built on the shoulders of transformers, we ensure 1:1 ge...</li><li><a href="https://huggingface.co/chat/settings/meta-llama/Meta-Llama-3-70B-Instruct/">HuggingChat</a>: Making the community's best AI chat models available to everyone.</li><li><a href="https://huggingface.co/discord-community">discord-community (Hugging Face Discord Community)</a>: no description found</li><li><a href="https://tenor.com/view/ineedit-needit-spongebob-squarepants-need-it-gif-4883495">Need It GIF - Ineedit Needit Spongebob Squarepants - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/llava-hf/bakLlava-v1-hf">llava-hf/bakLlava-v1-hf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-beta">HuggingFaceH4/zephyr-7b-beta · Hugging Face</a>: no description found</li><li><a href="https://x.com/reach_vb/status/1807830966515519667">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Introducing Local Gemma! 💎  100% local, private and secure - run anywhere, anytime!  cuda, mps, cpu - with presets to go as fast as possible - built on the shoulders of transformers, we ensure 1:1 ge...</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs</a>: A high-throughput and memory-efficient inference and serving engine for LLMs - vllm-project/vllm</li><li><a href="https://github.com/VedankPurohit/LiveRecall">GitHub - VedankPurohit/LiveRecall: Welcome to **LiveRecall**, the open-source alternative to Microsoft&#39;s Recall. LiveRecall captures snapshots of your screen and allows you to recall them using natural language queries, leveraging semantic search technology. For added security, all images are encrypted.</a>: Welcome to **LiveRecall**, the open-source alternative to Microsoft&amp;#39;s Recall. LiveRecall captures snapshots of your screen and allows you to recall them using natural language queries, leverag...</li><li><a href="https://github.com/SillyTavern/SillyTavern">GitHub - SillyTavern/SillyTavern: LLM Frontend for Power Users.</a>: LLM Frontend for Power Users. Contribute to SillyTavern/SillyTavern development by creating an account on GitHub.</li><li><a href="https://huggingface.co/lakkeo/stable-cypher-instruct-3b">lakkeo/stable-cypher-instruct-3b · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1257779448758141069)** (7 messages): 

> - `advanced resources for CNN topics like ViT and Unets`
> - `request for more tutorials on torch.distributed`
> - `Gradio app for role-play and character immersion prompt creation`
> - `TIL about '|' and '&' operators for Sets and Dicts in Python`
> - `question about Bayes' theorem in French` 


- ****Role-Play Prompt Tool Takes Center Stage****: A member shared a Gradio app they created for role-play and character immersion prompt creation, seeking feedback and improvement tips. They showcased some generated results and provided a [link to the tool](https://huggingface.co/spaces/xtreme86/System_roleplay_generator).
- ****Python Set Operators Unveiled****: A member shared a new [GitHub resource](https://github.com/noahlt/til/blob/main/python/2024-07-02-dict-and-set-operators.md) about the `|` and `&` operators for Sets and Dicts in Python, stating it is not directly related to ML but still interesting.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/xtreme86/System_roleplay_generator">System Roleplay Generator - a Hugging Face Space by xtreme86</a>: no description found</li><li><a href="https://github.com/noahlt/til/blob/main/python/2024-07-02-dict-and-set-operators.md">til/python/2024-07-02-dict-and-set-operators.md at main · noahlt/til</a>: Today I Learned - inspired by @simonw and @pdubroy - noahlt/til
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1257801237739606067)** (8 messages🔥): 

> - `attention mechanism`
> - `transformer architecture`
> - `compatibility in shells`
> - `sequence transduction models`
> - `demo video reactions` 


- ****Attention in transformers explained visually****: A user shared a [YouTube video](https://www.youtube.com/watch?v=eMlx5fFNoYc) titled 'Attention in transformers, visually explained | Chapter 6, Deep Learning,' detailing the key mechanisms inside transformers and LLMs. The video is praised as the 'best video about transformer architecture' seen so far.
- ****Compatibility First with Starship.rs****: A link to [Starship.rs](https://starship.rs/) was shared, emphasizing the shell's compatibility with the most common operating systems. This tool promises usability across various environments.
- ****The Transformer Paper****: A user highlighted the [Transformer architecture paper](https://arxiv.org/abs/1706.03762) from arXiv, proposing a network based solely on attention mechanisms. It achieved a BLEU score of 28.4 for English-to-German and 41.8 for English-to-French translations on the WMT 2014 tasks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a>: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and d...</li><li><a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">Attention in transformers, visually explained | Chapter 6, Deep Learning</a>: Demystifying attention, the key mechanism inside transformers and LLMs.Instead of sponsored ad reads, these lessons are funded directly by viewers: https://3...</li><li><a href="https://starship.rs/">Starship: Cross-Shell Prompt</a>: Starship is the minimal, blazing fast, and extremely customizable prompt for any shell! Shows the information you need, while staying sleek and minimal. Quick installation available for Bash, Fish, ZS...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1257779227596558518)** (7 messages): 

> - `OpenAI's CriticGPT Release`
> - `Stable Release of Embodied Agents Toolkit`
> - `Open Source OCR for Kazakh Language`
> - `Blog on Reinforcement Learning Specialization`
> - `Zero-Shot Generating Spatial Sound from Images` 


- **OpenAI Unveils CriticGPT**: A user shared a [YouTube video](https://youtu.be/4PgcaIfwLjo) introducing **CriticGPT**, a new AI model by **OpenAI** that identifies errors in code generated by **GPT-4**. The release was lauded as a significant step towards improving code accuracy.
- **Embodied Agents Toolkit for Robotic Integration**: [Embodied Agents toolkit](https://github.com/MbodiAI/mbodied-agents) was recently released to integrate state-of-the-art multimodal transformers into robotics with minimal code. The toolkit includes **Gradio interface support** and **HuggingFace dataset integration**.
- **OCR Solution for Kazakh Language Released**: An [open source solution](https://huggingface.co/spaces/BMukhtar/BookRecognitionKz) for OCR in Kazakh language has been released. The solution aims to fill a significant gap in OCR technologies for underrepresented languages.
- **Reinforcement Learning Specialization Blog**: A user shared their [blog series](https://sezan92.github.io/2024/07/03/RL-course-blog.html) on **Reinforcement Learning Specialization** from Coursera. The detailed notes cover multiple weeks and courses regarding RL.
   - *Check out the details and give feedback*, as suggested by the author.
- **Vietnamese Vision-Language Model Launched**: The **Vi-VLM** team launched a [Vietnamese Vision-Language model](https://huggingface.co/Vi-VLM/Vistral-V-7B) based on LLaVA and Vistral LLM. The model, optimized for image description tasks, leverages a proprietary dataset for pretraining and supervised finetuning.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/BMukhtar/BookRecognitionKz">BookRecogntionKZ - a Hugging Face Space by BMukhtar</a>: no description found</li><li><a href="https://sezan92.github.io/2024/07/03/RL-course-blog.html">Reinforcement Learning Specialization</a>: Notes for Reinforcement Learning Specialization by Coursera</li><li><a href="https://youtu.be/4PgcaIfwLjo">OpenAI releases CriticGPT to correct GPT-4&#39;s mistakes | Read the paper with me</a>: OpenAI has unveiled CriticGPT, a new AI model based on GPT-4 designed to identify errors in code generated by ChatGPT, marking a significant step towards imp...</li><li><a href="https://huggingface.co/spaces/rishitdagli/see-2-sound">SEE-2-SOUND - a Hugging Face Space by rishitdagli</a>: no description found</li><li><a href="https://github.com/waefrebeorn/KAN-Stem">GitHub - waefrebeorn/KAN-Stem: attempt at using gpt4o to create a KAN stem training script</a>: attempt at using gpt4o to create a KAN stem training script - waefrebeorn/KAN-Stem</li><li><a href="https://github.com/MbodiAI/mbodied-agents">GitHub - mbodiai/embodied-agents: Seamlessly integrate state-of-the-art transformer models into robotics stacks</a>: Seamlessly integrate state-of-the-art transformer models into robotics stacks - mbodiai/embodied-agents</li><li><a href="https://huggingface.co/Vi-VLM/Vistral-V-7B">Vi-VLM/Vistral-V-7B · Hugging Face</a>: no description found</li><li><a href="https://github.com/hllj/Vistral-V">GitHub - hllj/Vistral-V: Vistral-V: Visual Instruction Tuning for Vistral - Vietnamese Large Vision-Language Model.</a>: Vistral-V: Visual Instruction Tuning for Vistral - Vietnamese Large Vision-Language Model. - hllj/Vistral-V</li><li><a href="https://c57e0e7e63316ef057.gradio.live/">LLaVA</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1257906905934856252)** (4 messages): 

> - `Highway Net vs ResNet performance`
> - `Gradient vanishing problem in LSTM`
> - `Multi-branch structure inspiration from LSTM`
> - `Pre-trained models and fine-tuning techniques`
> - `topicSummaries` 


- **Highway Net performs worse than ResNet: Time to rethink**: A user questioned why **Highway Net performs worse than ResNet**, suggesting it might be time to reconsider the design choices. **Does the gating scheme from LSTM** really solve the gradient vanishing problem?
- **LSTM-inspired multi-branch structure**: A user admitted that their idea of **multi-branch structure comes from LSTM**. This raises questions about the gradient vanishing problem in LSTM's gating scheme.
- **Fine-tuning pre-trained models with cost-effective methods**: A user shared a [paper](https://arxiv.org/abs/2405.14739) discussing techniques to **fine-tune pre-trained models** without updating all parameters, focusing on resource-efficient methods like low-rank adjustments.
   - *These methods often neglect higher-dimensional parameter spaces like 4D, causing structural integrity issues.*



**Link mentioned**: <a href="https://arxiv.org/abs/2405.14739">FLoRA: Low-Rank Core Space for N-dimension</a>: Adapting pre-trained foundation models for various downstream tasks has been prevalent in artificial intelligence. Due to the vast number of tasks and high costs, adjusting all parameters becomes unfe...

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1257779351236378744)** (6 messages): 

> - `ADVANCED_CNN_RESOURCES`
> - `Neighborhood_Attention_Transformer_usage`
> - `Developer_Job_Openings`
> - `MaskFormer_training_issues`
> - `Lightweight_AI_for_programming` 


- ****Books on Advanced CNN Techniques Needed****: A user is seeking recommendations for books or resources on **advanced CNN topics** like ViTs and UNets, including **video processing**.
- ****Neighborhood Attention Transformer Maintenance Issue****: One user shared a link to the **Neighborhood Attention Transformer** documentation, highlighting that the model is in **maintenance mode only** and suggesting to re-install version 4.40.2 if issues occur with newer versions. They referenced the paper titled [Neighborhood Attention Transformer](https://arxiv.org/abs/2204.07143) for more context.
- ****Developer Seeking Jobs****: A user inquired if there are any **companies or projects** currently seeking a skilled developer.
- ****Issues Training MaskFormer Model****: A user is having trouble with **training a MaskFormer model** for instance segmentation, struggling with mask accuracy and training times. They are using the Hugging Face **Trainer class** and asked for personal assistance.
- ****Looking for Lightweight AI for Programming****: A user asked for recommendations on an **AI model suitable for programming** that is also lightweight.



**Link mentioned**: <a href="https://huggingface.co/docs/transformers/v4.42.0/model_doc/nat">Neighborhood Attention Transformer</a>: no description found

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1257936960442335342)** (9 messages🔥): 

> - `Custom pipeline creation`
> - `Text summarization models with high max input token length`
> - `Performance of open-source models vs. ChatGPT`
> - `Challenges in downloading and using Meta LLaMA`
> - `Inference freezing issues with Mistral model` 


- **Creating Custom Pipeline Guide by Andy Singal**: [Andy Singal shared a guide](https://github.com/andysingal/llm-course/blob/main/transformers/custom-pipeline.md) on creating custom pipelines using Transformers. The guide is part of the **LLM course** available on GitHub.
   - Exploring custom pipelines could be beneficial for those wanting tailored solutions in their NLP tasks.
- **Need for Text Summarization Model for Long Documents**: A user asked for recommendations on text summarization models that handle extremely long documents, specifically those with high max input token length. This request highlights the challenge of summarizing lengthy texts effectively.
- **Open-source Models vs. ChatGPT**: Discussion on whether Hugging Face models can match the performance/accuracy of ChatGPT 3.5 or 4. A member mentioned open models claiming better performance than ChatGPT 3.5.
   - *Models tend to overfit on benchmarks*, as emphasized in the discussion.
- **Meta LLaMA Download Challenges**: A user experienced difficulties downloading Meta LLaMA and considered building an API call to the model. They expressed concern about potential failures due to temporary file storage limitations during the 20-minute download process.
- **Inference Freezing Issue in Mistral**: Running an experiment with Mistral model froze at iteration 1800 when running 3000 inferences, taking a day to proceed. Inference choking might be due to some caching or other resource management issues between runs.



**Link mentioned**: <a href="https://github.com/andysingal/llm-course/blob/main/transformers/custom-pipeline.md">llm-course/transformers/custom-pipeline.md at main · andysingal/llm-course</a>: Contribute to andysingal/llm-course development by creating an account on GitHub.

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1258165680478621706)** (1 messages): 

> - `Discussion on running RealVisXL V4.0 Lightning model with diffusers.`
> - `Comparison of quality between A1111 and diffusers.`
> - `Support on Boosty.`
> - `Recommended negative prompt and generation parameters.`
> - `Issues with model performance during training phase.` 


- ****Support RealVisXL V4.0 on Boosty****: A member shared that you can support the development of **RealVisXL V4.0 Lightning** on [Boosty](https://boosty.to/sg_161222).
- ****RealVisXL V4.0 model training****: The **RealVisXL V4.0 Lightning** model, aimed at achieving photorealism, is still in the training phase and may **contain artifacts** and perform poorly in some cases.
- ****Running RealVisXL V4.0 with diffusers****: A member reported that running **RealVisXL V4.0 Lightning** with diffusers results in much poorer quality compared to using **A1111** despite using the same parameters (prompt, step, scheduler, etc.).



**Link mentioned**: <a href="https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning">SG161222/RealVisXL_V4.0_Lightning · Hugging Face</a>: no description found

  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1257783941876940862)** (93 messages🔥🔥): 

> - `GPT-4 parameter discussion`
> - `Nvidia involvement in GPT-4 development and leaks`
> - `Mixture of Experts (MoE) models`
> - `InstructGPT efficiency`
> - `Discord server scraping and ToS violations` 


- ****GPT-4 parameters **: How Big is It Really?****: Discussion emerged around GPT-4's parameter count, with numbers suspected to be around **1.7 trillion** to **1.8 trillion** according to various sources, including [Nvidia](https://www.nvidia.com). Interestingly, this figure shows a monumental leap from GPT-3's **175 billion** parameters, leaving members curious how **MoE** (Mixture of Experts) contributed to this scaling.
- ****InstructGPT's Real-World Boost****: **InstructGPT** gains were highlighted, particularly in practical applications where it offers a **10X to 100X increase** in efficiency. The community underscored the significant impact of **RLHF (Reinforcement Learning from Human Feedback)** as a key driver behind this improvement.
- ****Nvidia in the Know: GPT-4 Secrets****: Nvidia's familiarity with the **GPT-4** model size stirred debate, given their hardware support contracts. Despite being bound by NDAs, many believe Nvidia has insights into OpenAI's models due to their integral role in hardware provisions.
- ****Scraping Discord: Risky Business****: Attempting to scrape Discord servers for data, **even personal ones**, violates [Discord's ToS](https://discord.com/terms) and could lead to bans. Some tools like **DiscordChatExporter** avoid rate limiting but pose significant risks, as highlighted by several members.
- ****MoE Model Efficiency Insights****: Technical deep dives into **MoE models** revealed constraints and efficiency boosts. While MoE can significantly reduce computational loads by activating selective weights, it still faces memory bandwidth and VRAM challenges during inference.



**Link mentioned**: <a href="https://buttondown.email/ainews/archive/">AI News</a>: We summarize top AI discords + AI reddits + AI X/Twitters, and send you a roundup each day! See archive for examples.  &quot;Highest-leverage 45 mins I spend everyday&quot; - Soumith &quot;best AI new...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1257862843919761438)** (76 messages🔥🔥): 

> - `UL2 vs traditional training objectives`
> - `Starcoder2 and UL2 performance`
> - `PrefixLM and training implications`
> - `Scaling laws and learning rate schedules`
> - `FIM and UL2 comparisons` 


- **Industry slow to adopt UL2 training objectives**: A member expressed surprise at the industry's slow adoption of **UL2 training objectives** despite their theoretical and empirical benefits in fixing issues like short-term planning and the reversal curse. Despite tests from [Starcoder2](https://twitter.com/vaibhav_adlakha/status/1777854167672820000) and Mosaic showing underperformance compared to traditional methods, the member remains optimistic about future tweaks.
- **[Scaling Laws Discrepancies](https://arxiv.org/abs/2406.19146)**: Researchers resolved discrepancies between **Kaplan** and **Hoffmann scaling laws** by identifying factors like last layer computational cost and optimizer tuning. Their findings debunked the necessity of careful learning rate decay for the validity of the Chinchilla scaling law.
- **PrefixLM and training efficiency concerns**: Members debated the efficiency of **PrefixLM** in training, noting its slower training speeds and potential inefficiency with current attention algorithms. One member pointed out that models might adapt differently to bidirectional contexts versus causal ones, impacting performance.
- **FIM vs UL2 Objectives**: Members discussed **Fill-in-the-Middle (FIM)** and how it compares to UL2/masked language objectives. They noted that FIM could be more efficient as it also predicts front and tail segments, potentially offering better outcomes than the UL2 approach.
- **Attention Mask Adaptation**: A discussion emerged on the adaptation of models to different attention masks, with some noting the challenges in training efficiency and the importance of specific tokens. A related paper on [causal to bidirectional attention adaptation](https://twitter.com/vaibhav_adlakha/status/1777854167672820000) was shared to illustrate these dynamics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.19146">Resolving Discrepancies in Compute-Optimal Scaling of Language Models</a>: Kaplan et al. and Hoffmann et al. developed influential scaling laws for the optimal model size as a function of the compute budget, but these laws yield substantially different predictions. We explai...</li><li><a href="https://arxiv.org/abs/2406.19370">Emergence of Hidden Capabilities: Exploring Learning Dynamics in Concept Space</a>: Modern generative models demonstrate impressive capabilities, likely stemming from an ability to identify and manipulate abstract concepts underlying their training data. However, fundamental question...</li><li><a href="https://boyuan.space/diffusion-forcing/">
      Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion
    </a>: no description found</li><li><a href="https://x.com/tomerporian/status/1808090819808629216">Tweet from Tomer Porian (@tomerporian)</a>: 🧵1/8 We resolve the discrepancy between the compute optimal scaling laws of Kaplan (exponent 0.88, Figure 14, left) et al. and Hoffmann et al. (“Chinchilla”, exponent 0.5). Paper: https://arxiv.org/a...</li><li><a href="https://github.com/YangLing0818/consistency_flow_matching">GitHub - YangLing0818/consistency_flow_matching: Official Implementation for &quot;Consistency Flow Matching: Defining Straight Flows with Velocity Consistency&quot;</a>: Official Implementation for &quot;Consistency Flow Matching: Defining Straight Flows with Velocity Consistency&quot; - YangLing0818/consistency_flow_matching
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1257882203543568415)** (22 messages🔥): 

> - `Discrepancy in compute optimal scaling laws between Kaplan et al. and Hoffmann et al.`
> - `Kaplan et al.'s last layer computational cost, warmup duration, and scale-dependent optimizer tuning`
> - `Attention flops and the 6ND approximation in scaling law computations`
> - `PyTorch flop counter utility and FLOPs calculation methodologies`
> - `Chinchilla paper scaling law and extrapolation issues` 


- ****Resolving Scaling Law Discrepancies Between Kaplan and Hoffmann**: [Researchers explain the discrepancy](https://arxiv.org/abs/2406.19146) between Kaplan et al. and Hoffmann et al.'s scaling laws by identifying issues such as last layer computational cost, warmup duration, and scale-dependent optimizer tuning.**: Researchers corrected these factors and obtained excellent agreement with Hoffmann et al.'s scaling law, also known as the **Chinchilla scaling law**. They found the **learning rate decay** hypothesis by Hoffmann et al. to be non-essential and derived scaling laws for **optimal learning rate and batch size**.
- ****Attention FLOPs Impact on Scaling Laws**: Community members discuss the inadequacy of the 6ND approximation for small scale models.**: They suggest incorporating attention flops using a different formula from **Kaplan et al.**, specifically `C = 6ND + 6 * n_layers * seq_len * d_model` instead of `6ND`.
- ****Utility Flaws in PyTorch's FLOP Counter**: Discussion on PyTorch's built-in flop counter utility.**: Concerns were raised about the utility not hitting an error if it encounters an operation it doesn't know the FLOPs for, defaulting to ignoring it.
- ****Extrapolation Pitfalls in Scaling Law Fits**: Community emphasizes the risks of using too small-scale experiments to fit scaling laws.**: They highlight that **chin***chilla numbers should not be cargo-culted** and emphasize the need for spending substantial compute to ensure accurate extrapolation.
- ****Chinchilla Work on Rubik's Cube Data**: Reference to a GitHub project focusing on synthetic data scaling law fits.**: A project on [GitHub](https://github.com/kyo-takano/chinchilla) demonstrates these principles using **Rubik's cube generated data** to validate scaling laws.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.19146">Resolving Discrepancies in Compute-Optimal Scaling of Language Models</a>: Kaplan et al. and Hoffmann et al. developed influential scaling laws for the optimal model size as a function of the compute budget, but these laws yield substantially different predictions. We explai...</li><li><a href="https://x.com/tomerporian/status/1808090819808629216">Tweet from Tomer Porian (@tomerporian)</a>: 🧵1/8 We resolve the discrepancy between the compute optimal scaling laws of Kaplan (exponent 0.88, Figure 14, left) et al. and Hoffmann et al. (“Chinchilla”, exponent 0.5). Paper: https://arxiv.org/a...</li><li><a href="https://arxiv.org/abs/2104.03113">Scaling Scaling Laws with Board Games</a>: The largest experiments in machine learning now require resources far beyond the budget of all but a few institutions. Fortunately, it has recently been shown that the results of these huge experiment...</li><li><a href="https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py">cookbook/calc/calc_transformer_flops.py at main · EleutherAI/cookbook</a>: Deep learning for dummies. All the practical details and useful utilities that go into working with real models. - EleutherAI/cookbook</li><li><a href="https://github.com/kyo-takano/chinchilla/blob/master/examples/efficientcube.ipynb">chinchilla/examples/efficientcube.ipynb at master · kyo-takano/chinchilla</a>: A toolkit for scaling law research ⚖. Contribute to kyo-takano/chinchilla development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1258144662800044155)** (2 messages): 

> - `EAP with integrated gradients`
> - `Methods for discovering and applying sparse feature circuits`
> - `Generalization improvement using SHIFT`
> - `Scalable interpretability pipeline for sparse feature circuits` 


- **Weekend Updates**: A community member mentioned an event happening this weekend, but no specific details were provided.
- **EAP and Integrated Gradients Insight**: Discussion on EAP with integrated gradients referencing the paper [Methods for discovering and applying sparse feature circuits](https://arxiv.org/abs/2403.19647).
- **Sparse Feature Circuits Paper Explored**: The paper introduces methods for discovering and applying **sparse feature circuits** that enable detailed understanding of language model behaviors. They are useful for downstream tasks and offer an unsupervised, scalable interpretability pipeline.
- **SHIFT for Classifier Generalization**: The paper discusses the **SHIFT** method that improves classifier generalization by ablating task-irrelevant features as judged by humans. This method leverages fine-grained units for better interpretability.



**Link mentioned**: <a href="https://arxiv.org/abs/2403.19647">Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models</a>: We introduce methods for discovering and applying sparse feature circuits. These are causally implicated subnetworks of human-interpretable features for explaining language model behaviors. Circuits i...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1257804828411957350)** (26 messages🔥): 

> - `PR confirmation and lm-eval reference`
> - `Loglikelihood_rolling functionality and usage`
> - `Handling document length longer than model's context in perplexity evaluations`
> - `Errors in model evaluation with specific configurations`
> - `Preprocessing functions and pipeline consistency` 


- ****PR confirmation for lm-eval reference****: A member requested confirmation for a **pull request** because they want to reference it in their paper for running evaluations of their benchmark. **Stellaathena** inquired if the numbers in the paper were from the current code, and the member confirmed.
- ****Understanding loglikelihood_rolling****: A member asked about the purpose of **loglikelihood_rolling** and whether it means inputting a model to get a loglikelihood value that can be turned into a loss value. **Hailey Schoelkopf** explained that it gives the loglikelihood of producing a document from the empty string with reference to the [documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/d855d0baf8576296e790d0c9477b40a710d28e67/docs/model_guide.md?plain=1#L63).
- ****Handling long documents in perplexity evaluations****: **Stellaathena** asked how to compute perplexity for documents longer than the model's context window without errors. **Hailey Schoelkopf** noted that default perplexity tasks handle chunking based on model length automatically within the `loglikelihood_rolling` method.
- ****Dataset-specific issues in evaluation configuration****: Using a specific config file with `proof-pile` dataset throws an **error** for **Stellaathena**, but the same config works with `lambada_openai`. **Hailey Schoelkopf** mentioned it could be related to the metrics, suggesting a fix and indicating a potential silent failure with metric use.
- ****Reusing preprocessing functions****: A member inquired about preventing preprocessing functions from rerunning every time and if preprocessed data can be stored and reused in the pipeline. This concern aligns with ensuring efficiency in the evaluation process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/d855d0baf8576296e790d0c9477b40a710d28e67/docs/model_guide.md?plain=1#L63>">lm-evaluation-harness/docs/model_guide.md at d855d0baf8576296e790d0c9477b40a710d28e67 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/actions/runs/9780045009/job/27000738664?pr=2010.">Added MedConceptsQA Benchmark · EleutherAI/lm-evaluation-harness@0c3a587</a>: A framework for few-shot evaluation of language models. - Added MedConceptsQA Benchmark · EleutherAI/lm-evaluation-harness@0c3a587
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1257772870797492285)** (174 messages🔥🔥): 

> - `Discussion About Trying Gemini 1.5 Pro`
> - `Access Issues with GPT4o`
> - `New Perplexity Features on Mobile`
> - `Refund Process for Pro Subscription`
> - `Concerns About Perplexity's Live Internet Access` 


- ****Gemini 1.5 Pro chatbot recommendation****: Members discussed the performance and features of **Gemini 1.5 Pro**, noting its **large context window** and **fast performance**. One user particularly recommended trying it due to its decent capabilities.
- ****Access Issues with GPT4o and Alternatives****: Several users reported issues finding **free ChatGPT 4o** options, suggesting alternatives like **Bing chat in precise mode** and **Claude 3.5 Sonnet** on [claude.ai](https://claude.ai), which is also praised for its free use despite some usage limits.
- ****New Perplexity features on mobile****: A user inquired about the availability of the **new Wolfram Alpha and code generation features** in Perplexity's mobile app. Another user confirmed that these features are available on iOS.
- ****Refund process for Pro subscription****: A user asked about the **refund process for Perplexity Pro subscription**. Another member provided detailed refund policies, specifying guidelines for **EU, UK, Turkey**, and **all other customers** including timelines and conditions.
- ****Concerns about Perplexity's live internet access****: A member reported erratic behavior in Perplexity's ability to **access live internet information**, especially for checking real-time data like sports scores and weather. Despite correcting itself at times, it often **denied** such capabilities, leading to **consistent frustration**.



**Link mentioned**: <a href="https://git.new/Portkey-Phidata">gateway/cookbook/integrations/Phidata_with_ Perplexity.ipynb at main · Portkey-AI/gateway</a>: A Blazing Fast AI Gateway. Route to 200+ LLMs with 1 fast &amp; friendly API. - Portkey-AI/gateway

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1257779644514566155)** (9 messages🔥): 

> - `Lean Canvas Guide`
> - `Starting Perplexity AI Story`
> - `Building a Blackbox`
> - `OpenSSH Query`
> - `Sober Living in Echo Park` 


- ****Guide to Lean Canvas****: Explore **Lean Canvas** solutions with a streamlined guide on [Perplexity AI](https://www.perplexity.ai/search/when-should-you-do-a-lean-canv-z_lDH7CJStuuX.MpyRGNMA). Contains valuable insights and step-by-step instructions.
- ****Perplexity AI Founding Story****: Dive into the origins of **Perplexity AI** with this comprehensive [narrative](https://www.perplexity.ai/search/the-story-behind-starting-perp-DnZ.yJgfSM28Ra9_h2uKWg). The article shares the inspiration and challenges faced during the creation process.
- ****Building a Blackbox****: Learn about constructing a blackbox system in the AI domain through a detailed [Perplexity AI search result](https://www.perplexity.ai/search/q-how-can-we-build-a-blackbox-Iua1cgLZTfOSrmg8lIxGiw#3). Discusses methodologies and potential applications.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1257781083303579648)** (7 messages): 

> - `Usage of Sonnet 3.5 with Perplexity API`
> - `Availability of Sonnet in Perplexity API`
> - `List of available models in Perplexity API`
> - `Search engine usage via Perplexity API`
> - `Issues with llama-3-sonar-large-32k-online model` 


- ****Sonnet 3.5 not available via Perplexity API****: **Sonnet** is not provided via the **Perplexity API**. The available models can be found in the [documentation](https://docs.perplexity.ai/docs/model-cards).
- ****Interest in API for search engine usage****: Multiple members expressed interest in using **Perplexity's search engine** via the API. One mentioned emailing **api@perplexity.ai** for beta access.
- ****Incorrect answers from llama-3-sonar-large-32k-online model****: A member noted that **llama-3-sonar-large-32k-online** gave wrong answers for a simple query. Another suggested using the `after:` parameter to refine the search.



**Link mentioned**: <a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1257842866391089172)** (19 messages🔥): 

> - `CUDA-only hackathon at the AGI House in San Francisco`
> - `Meta Hacker Cup 2024 schedule`
> - `Discussion about the price and purchase of NVIDIA GPUs (3090, 4090)` 


- ****CUDA Hackathon Opens in San Francisco****: **Ash Vardanian** is hosting a [CUDA-only hackathon](https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf) at the AGI House on **July 13th**, with **Chris Lattner** as a notable speaker. **H100 access** will be provided to all participants, sponsored by Nebius.ai.
- ****Meta Hacker Cup Returns for 2024 Season****: The [Meta Hacker Cup](https://codeforces.com/blog/entry/131165) will kick off on **September 20th** with its Practice Round, followed by a series of rounds culminating in the Finals on **December 7th**. **Mark Saroufim**, part of the organizing team, encourages participation, especially for those interested in code generation.
- ****Debate Over NVIDIA 3090 Prices****: Members discussed whether to purchase a **3090**, noting that current prices are around **$1,000**. **Mark Saroufim** mentioned he bought a **4090** for $1,200 during a Meta layoffs scare.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://partiful.com/e/fxMwOW9dtCCWoEPyyTIf">RSVP to Hardcore CUDA Hackathon | Partiful</a>: *All talks and projects MUST be written in CUDA* Every hardcore hacker gets a H100 for the day. All sponsored and proved by Nebius.ai! Let&#x27;s blow away some baselines.  Speakers: - Chris Lattner (...</li><li><a href="https://codeforces.com/blog/entry/131165">Meta Hacker Cup 2024 Schedule — Introducing the Meta Hacker Cup AI Track - Codeforces</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1257852511742595175)** (14 messages🔥): 

> - `Steps involved in compiling a function in Pytorch with Inductor backend for Nvidia device`
> - `Difference between triton IR and MLIR`
> - `John Carmack's positive feedback on PyTorch team and contributing to open source`
> - `Issue with forcing Inductor to generate Triton kernels for GEMM and Conv` 


- **Steps for Compiling Pytorch Function with Inductor**: Discussion on the steps for compiling a function in **Pytorch** with **Inductor** backend for Nvidia, from **PYTorch (python method)** to **PTX**. Confusion about whether **MLIR** should be a separate step. ([source](https://x.com/ID_AA_Carmack/status/1807072152631333060))
- **MLIR is not a Separate IR**: Clarification that **MLIR** is a toolkit for building your own IR, not a separate IR. Triton uses **ttir**, **ttgir**, **llir**, **ptx**, and **cubin** as steps but *'the ttir => ttgir translation is the most important'*.
- **John Carmack Praises PyTorch Team**: John Carmack is **impressed by the @PyTorch team's response to bug reports**, stating that while the project has a substantial learning curve, **the setup documentation** made him self-sufficient.
- **Inductor Triton Kernels Issue**: Forcing **Inductor** to generate **Triton kernels** for everything works for **GEMM** but not for **Conv**, despite having kernel templates. Issue raised on [GitHub](https://github.com/pytorch/pytorch/issues/125728), looking for a fix.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/ID_AA_Carmack/status/1807072152631333060">Tweet from John Carmack (@ID_AA_Carmack)</a>: I am super impressed by the @PyTorch team’s response to bug reports. Part of me feels that, since it is fully open source, I should go all the way down to creating a patch myself,  but a project that ...</li><li><a href="https://github.com/pytorch/pytorch/issues/125728">torch._inductor.config.max_autotune_gemm_backends = &quot;TRITON&quot; crashes with Convolution layer · Issue #125728 · pytorch/pytorch</a>: 🐛 Describe the bug Repro import torch import torch._inductor.config # torch._inductor.config.trace.enabled = True torch._inductor.config.max_autotune_gemm_backends = &quot;TRITON&quot; torch._inducto...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1257794158051983531)** (8 messages🔥): 

> - `High-performance matrix multiplication on CPU`
> - `3D V-Cache performance on AMD Ryzen`
> - `Difference between 3D and non-3D Ryzen chips`
> - `Discussion on specialization of 3D V-Cache chips`
> - `Simulation benchmarks for CPUs` 


- **High-performance matrix multiplication on CPU**: [Mobicham shared a tutorial](https://salykova.github.io/matmul-cpu) on high-performance matrix multiplication on CPU with code available at [matmul.c](https://github.com/salykova/matmul.c). The implementation, optimized for AMD Ryzen 7700, outperforms NumPy by achieving over **1 TFLOPS** using **3 lines of OpenMP directives**.
- **3D V-Cache boosts AMD's performance**: Iron_bound wondered about the performance impact of **3D V-Cache**, citing [a review](https://www.anandtech.com/show/18795/the-amd-ryzen-7-7800x3d-review-a-simpler-slice-of-v-cache-for-gaming/4) showing it has 96MB of L3 cache and comparing it across games and simulations.
- **3D vs non-3D Ryzen chip differences**: As_ai and iron_bound discussed the differences between 3D and non-3D Ryzen chips, noting that 3D versions have **double L3 cache** but operate at lower clocks to prevent damage to the extra cache silicon layer.
- **Specializations in 3D V-Cache chips**: As_ai questioned if the **3D V-Cache** had any specializations beyond the extra cache, and iron_bound confirmed **the differences are primarily more cache and lower clocks**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://oimo.io/works/life/">Life Universe</a>: no description found</li><li><a href="https://salykova.github.io/matmul-cpu">Beating NumPy’s matrix multiplication in 150 lines of C code</a>: TL;DR The code from the tutorial is available at matmul.c. This blog post is the result of my attempt to implement high-performance matrix multiplication on CPU while keeping the code simple, portable...</li><li><a href="https://www.anandtech.com/show/18795/the-amd-ryzen-7-7800x3d-review-a-simpler-slice-of-v-cache-for-gaming/4">The AMD Ryzen 7 7800X3D Review: A Simpler Slice of V-Cache For Gaming</a>: no description found
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1257888940464148604)** (31 messages🔥): 

> - `Loading a buffer containing int4 using torchao`
> - `Saving a tensor into a safetensors file`
> - `Dequantizing tensors using torchao`
> - `Handling packed int4 arrays in Python`
> - `torchao's handling of unexpected keyword arguments` 


- **Handling packed int4 arrays in Python efficiently**: Members discussed interpreting packed int4 arrays by first converting to uint8 and then bit-shifting and stacking the tensor using `torch.frombuffer()`. They emphasized understanding the bit-layout of the buffer before parsing it.
- **Dequantizing tensors using Python techniques**: A member asked about dequantizing tensors with quantization scales, which involved creating tensors and performing element-wise operations like `dequant_tensor = quant_tensor * scale` using PyTorch.
- **torchao's buffer loading and unexpected keyword handling**: Discussion centered around torchao's function `to()` and how it interprets arguments, revealing issues with unexpected keyword arguments being passed to `__new__()`. They noted the importance of configuring parameters correctly to avoid errors during tensor operations.



**Link mentioned**: <a href="https://github.com/ethanc8/Gemini-Nano/blob/master/playground/converter.py#L166">Gemini-Nano/playground/converter.py at master · ethanc8/Gemini-Nano</a>: Contribute to ethanc8/Gemini-Nano development by creating an account on GitHub.

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1257782594716700702)** (84 messages🔥🔥): 

> - `Memory efficiency comparison with PyTorch`
> - `Visualization of model weights and training issues`
> - `New GitHub PRs and bug fixes`
> - `Experiments with and observations on muP`
> - `Schedule-free optimization discussion` 


- ****Memory Efficiency: Our models significantly outperform PyTorch****: Our batch 16 run fits very comfortably, even batch 24 fits fine, while **PyTorch** struggles with batch size 8. This highlights **significant memory savings** compared to PyTorch.
- ****Visualization and Training Bugs****: Issues observed with **integer division by zero error** during training on batch sizes < 4 in the **HellaSwag eval dataloader**. Fix implemented in [GitHub PR #667](https://github.com/karpathy/llm.c/pull/667).
- ****muP Experiments Look Promising Yet Challenging****: **Preliminary muP** results are stable with various learning rates, with extensive experiments filling up to **80 GB of VRAM** per GPU. Further hyperparameter sweeps are being planned.
- ****Efficient Inference with HF Transformers****: Sample generation from **Hugging Face GPT2** models is extremely slow, taking 4 minutes for 512 steps due to inefficient eager mode and dynamic key-value concatenations. Active effort noted to improve [Transformers documentation](https://huggingface.co/blog/transformers-docs-redesign).
- ****Excitement Over Schedule-Free Optimizers****: [Schedule-free optimizers by Facebook Research](https://github.com/facebookresearch/schedule_free) show surreal convergence on various tasks. Claims suggest it may be a breakthrough in practical and theoretical optimization research.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/transformers-docs-redesign">Making sense of this mess</a>: no description found</li><li><a href="https://x.com/_clashluke/status/1808590060654108910?s=46&t=Qzf619GMalbD77YmVui2Jw">Tweet from Lucas Nestler (@_clashluke)</a>: Schedule-free optimizers (https://x.com/aaron_defazio/status/1776320004465582331) are surreal.  I&#39;ve read the paper, looked into the math, and tried to understand what&#39;s happening. It all seem...</li><li><a href="https://github.com/karpathy/llm.c/pull/667">Fix eval dataloader div by zero for &lt; 4 batch size by gordicaleksa · Pull Request #667 · karpathy/llm.c</a>: We need at least batch size of 4 to support the current eval logic. Alternatively we can rewrite the eval a bit, but that&#39;s probably overengineering at this point? Worst case that could happen is ...</li><li><a href="https://github.com/karpathy/llm.c/pull/641/">Add check versions of functions by gordicaleksa · Pull Request #641 · karpathy/llm.c</a>: Add socket close check functions - consistent with the rest of the codebase.</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/viewer/default/train">HuggingFaceFW/fineweb-edu · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1257773576665170053)** (145 messages🔥🔥): 

> - `Anti-AI art software debate`
> - `Tips for low-resolution pixel art training`
> - `Job postings on Discord`
> - `Improving prompt techniques and comparisons`
> - `Comparisons between SD models *MixofExperts* and segmoe` 


- ****Anti-AI Art Software Discussed****: Members discussed the feasibility of **anti-AI art software** that could protect artists' work from being used in AI training. Suggestions included existing tools like [**Glaze**](https://glaze.cs.uchicago.edu/) and **Nightshade**, but community members pointed out that these methods are easily defeated.
- ****Training Low-Resolution Pixel Art Models****: A user inquired about training AI for **16x16 pixel art**, and members recommended upscaling images to **512x512** for training. Crystalwizard noted the potential inefficiency but suggested trial and error as a cost-effective method.
- ****Job Postings and Freelance Work****: A user asked if there was a **job-posting channel** for recruiting, and another inquired about **upwork account rentals**, highlighting the **demand for freelance opportunities**.
- ****Effectiveness of Prompt Techniques****: Members discussed different **prompting techniques** and their effectiveness in generating images using text2img. Variations like **[A|B], C** vs. **[A, B, C]** were mentioned, alongside a comparison of model capabilities like **SD1.5** vs. **segmoe** and **MixofExperts**.
- ****Comparing SD Models: MixofExperts vs segmoe****: Discussions included the **segmoe model in ComfyUI** and its substantial improvement in **prompt understanding**. Comparisons were made with **SD1.5 finetunes and newer models**, emphasizing prompt accuracy and performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/cagliostrolab/animagine-xl-3.1">cagliostrolab/animagine-xl-3.1 · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=tZUMH_DUdfA&t=337s">AI News - ComfyUI Segmoe and Stable Video Diffusion 1.1</a>: The video introduces the new model for Adobe Firefly, emphasizing its improved capabilities in generating higher quality images, especially of people, and im...</li><li><a href="https://www.reddit.com/user/No_Dragonfruit_5472/comments/1chdemx/tradingview_premium_pack_crack_2024_version_free/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1257776175862779985)** (1 messages): 

> - `Big update to /models page`
> - `Changing Google Token Sizes for Gemini and PaLM models`
> - `Deprecation of Default Model in settings page`
> - `Deprecation of custom auth headers for OpenAI API keys` 


- ****Big Update Coming to /models Page****: A significant update to the **/models page** is coming soon, with a sneak peek shared. Members are encouraged to provide feedback in the [dedicated channel](https://discord.com/channels/1107397803266818229).
- ****Google Token Sizes Changing for Gemini and PaLM Models****: The **Gemini** and **PaLM** models will have their token lengths changed to match GPT-equivalent sizes, increasing token size roughly **3x** and reducing context limits, leading to higher pricing but with the same model and API.
- ****Deprecating the Default Model on Settings Page****: The **Default Model** option on the **/settings page** is being deprecated as most apps set models themselves or use the auto router. Users with a valid use case are encouraged to provide feedback.
- ****Deprecating Custom Auth Headers for API Keys****: The use of **custom auth headers** for sending OpenAI API keys is being deprecated, with a replacement coming soon. This feature was used by a few people in mid-June but was never officially documented.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1257816161379422219)** (3 messages): 

> - `Quick and dirty wrapper shared by lastrosade`
> - `Feedback on the non-streamed response` 


- ****Quick and dirty wrapper shared****: **lastrosade** announced the creation of a quick and dirty wrapper, offering it to anyone interested in the community. No additional technical details or links were provided.
- ****Feedback on the non-streamed response****: A community member, **clarie_starr**, commented on the wrapper mentioning, *"So all that for a non-streamed response. I'll give it to you it's um detailed."* followed by **lastrosade** agreeing that the wrapper *"sucks"*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1257772784084586579)** (77 messages🔥🔥): 

> - `500 errors with Claude 3.5`
> - `Self-moderation issues with Claude`
> - `Different frontends for using and jailbreaking Claude`
> - `OpenRouter privacy settings and logging policies`
> - `Google models token size change announcement` 


- ****500 Errors with Claude 3.5****: Several users reported intermittent **500 errors** while using **Claude 3.5** on OpenRouter. Temporary fixes include switching to different versions like **Claude 3.0**.
- ****OpenRouter Privacy and Logging Issues Addressed****: Users discussed OpenRouter's privacy settings, clarifying that some providers log requests while others do not, with an emphasis on **NovitaAI** and **Infermatic** not retaining data. [Alex Atallah](https://openrouter.ai/settings/privacy) provided insights into the different privacy policies for third-party providers.
- ****Google Models Token Size Update Clarification****: A discussion on the **token size change** in Google models raised concerns about potential cost increases. [LouisGV](https://openrouter.ai/models/microsoft/wizardlm-2-8x22b) clarified that total pricing remains roughly the same despite the token size adjustment.
- ****Exploring Different Frontends for Claude****: Users explored various frontends like **SillyTavern** and **LibreChat** for jailbreaking or prefilling Claude models. **Typingmind** and **Pal Chat** were suggested as alternatives for a smoother user experience.
- ****Quantization of LLM Models on OpenRouter****: Questions about the **quantization** of deployed LLM models on OpenRouter were raised, focusing on whether models are in **FP16** or other precisions. The discussion highlighted that models remain in their native precision unless specified otherwise by the provider.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sillytavern.app)">no title found</a>: no description found</li><li><a href="https://lmsys.org/blog/2024-07-01-routellm/">RouteLLM: An Open-Source Framework for Cost-Effective LLM Routing | LMSYS Org</a>: &lt;p&gt;LLMs have demonstrated remarkable capabilities across a range of tasks, but there exists wide variation in their costs and capabilities, as seen from the ...</li><li><a href="https://openrouter.ai/models/sao10k/l3-euryale-70b">Llama 3 Euryale 70B v2.1 by sao10k</a>: Euryale 70B v2.1 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k).  - Better prompt adherence. - Better anatomy / spatial awareness. - Adapts much better to unique and c...</li><li><a href="https://openrouter.ai/models/gryphe/mythomax-l2-13b">MythoMax 13B by gryphe</a>: One of the highest performing and most popular fine-tunes of Llama 2 13B, with rich descriptions and roleplay. #merge</li><li><a href="https://openrouter.ai/settings/privacy">Settings | OpenRouter</a>: Manage your accounts and preferences</li><li><a href="https://openrouter.ai/models/microsoft/wizardlm-2-8x22b">WizardLM-2 8x22B by microsoft</a>: WizardLM-2 8x22B is Microsoft AI&#x27;s most advanced Wizard model. It demonstrates highly competitive performance compared to leading proprietary models, and it consistently outperforms all existing ...</li><li><a href="https://infermatic.ai/privacy-policy/">Privacy Policy - Infermatic</a>: no description found</li><li><a href="https://web.archive.org/web/20240112082806/https://infermatic.ai/privacy-policy/">Privacy Policy - Infermatic</a>: no description found
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1257779373755465799)** (40 messages🔥): 

> - `Magic.dev $500M to $1.5B valuation, 20 employees, no product, no revenue.`
> - `New paper on persona-driven data synthesis with 1 billion personas.`
> - `First real-time Audio LLM by Kyutai, 'Moshi'.`
> - `OpenDevin founders start All Hands AI.`
> - `Sentient's $85M seed round for open AGI platform.` 


- **Magic.dev valuation skyrockets to $1.5B despite no product**: [Magic.dev](https://www.reuters.com/technology/artificial-intelligence/ai-coding-startup-magic-seeks-15-billion-valuation-new-funding-round-sources-say-2024-07-02/) jumps in valuation from **$500M** to **$1.5B** with just 20 employees, no product, and no revenue.
- **Synthetic data creation gets boost with 1 billion personas**: [Persona Hub](https://arxiv.org/abs/2406.20094) introduces 1 billion personas to scale synthetic data creation, showing massive gains in mathematical problem solving and diverse scenarios.
   - Presented by [Aran Komatsuzaki](https://x.com/arankomatsuzaki/status/1807593343007818065), the new approach leads to significant improvements, particularly on the **MATH** benchmark.
- **Kyutai launches real-time Audio LLM 'Moshi'**: [Moshi](https://x.com/giffmana/status/1808482848808010149) claims to be the first real-time Audio LLM with minimal delay, although the quality remains slightly robotic.
   - 'Moshi' demo [by Kyutai](https://x.com/kyutai_labs/status/1808526962941366415) shows promise despite its current limitations, as it sometimes interrupts the user in eagerness to respond.
- **Founders of OpenDevin launch All Hands AI**: [OpenDevin](https://x.com/gneubig/status/1808493521315496229) founders announce the formation of All Hands AI to accelerate AI software development for everyone in an open-source manner.
- **Sentient secures $85M seed round for open-source AI platform**: [Sentient](https://x.com/sentient_agi/status/1808136737257918916?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) announces **$85M** seed round co-led by Founders Fund to support the development of a community-built open AGI platform aimed at equitable AI development.
   - Prominent investors like [Peter Thiel](https://www.coindesk.com/business/2024/07/02/peter-thiels-founders-fund-leads-85m-seed-investment-into-open-source-ai-platform-sentient/) are backing the initiative, which aims to distribute AI benefits globally.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.20094">Scaling Synthetic Data Creation with 1,000,000,000 Personas</a>: We propose a novel persona-driven data synthesis methodology that leverages various perspectives within a large language model (LLM) to create diverse synthetic data. To fully exploit this methodology...</li><li><a href="https://huggingface.co/CAMB-AI/MARS5-TTS">CAMB-AI/MARS5-TTS · Hugging Face</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Flowers_for_Algernon">Flowers for Algernon - Wikipedia</a>: no description found</li><li><a href="https://x.com/poolsideai/status/1738669662467178581">Tweet from poolside (@poolsideai)</a>: Some of the very visible above the water line fun we’ve been having   Expect to see more of what is going on below the surface soon!</li><li><a href="https://x.com/SFResearch/status/1808549356536041487">Tweet from Salesforce AI Research (@SFResearch)</a>: Thanks @Benioff and @SilvioSavarese for guiding our research toward the power of Small Language Models (SMLs). xLAM-1B is open source and will be coming soon to Hugging Face. 💥 #SMLs #TinyGiant #AIRe...</li><li><a href="https://x.com/johnbyronhanby/status/1808235931784434049?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from John Byron Hanby, IV (@johnbyronhanby)</a>: As someone who has been fortunate to work directly with CAIOs for some of the companies listed, here are a few thoughts:  - How does an org find someone to do this role?  ✅The person needs to have a d...</li><li><a href="https://x.com/kyutai_labs/status/1808526962941366415">Tweet from kyutai (@kyutai_labs)</a>: https://moshi.chat/?queue_id=talktomoshi</li><li><a href="https://x.com/giffmana/status/1808482848808010149">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: Kyutai Moshi - first real-time Audio LLM.  Basically no delay - the LLM even interrupted the speaker a few times. It was actually a bit eager to answer very quick. :)  All to be open-sourced. Quality ...</li><li><a href="https://x.com/gneubig/status/1808493521315496229">Tweet from Graham Neubig (@gneubig)</a>: Announcement: @rbren_dev, @xingyaow_, and I have formed a company!  Our name is All Hands AI 🙌 https://www.all-hands.dev/  And our mission is to build the world’s best AI software development agents,...</li><li><a href="https://x.com/arankomatsuzaki/status/1807593343007818065?utm_source=ainews&utm_medium=email&utm_campaign=ainews-to-be-named-5628">Tweet from Aran Komatsuzaki (@arankomatsuzaki)</a>: Scaling Synthetic Data Creation with 1,000,000,000 Personas  - Presents a collection of 1B diverse personas automatically curated from web data - Massive gains on MATH: 49.6 -&gt;64.9  repo: https://g...</li><li><a href="https://github.com/hrishioa/rakis?tab=readme-ov-file">GitHub - hrishioa/rakis</a>: Contribute to hrishioa/rakis development by creating an account on GitHub.</li><li><a href="https://t.co/vQzLSq2ncG">Mozilla Llamafile, Builders Projects Shine at AI Engineers World&#039;s Fair</a>: At the AI event, the Mozilla team showed how Llamafiles made open models easier to use, and make them run fast on consumer CPUs.</li><li><a href="https://x.com/sentient_agi/status/1808136737257918916?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sentient (@sentient_agi)</a>: We are thrilled to announce Sentient&#39;s $85M seed round, co-led by @foundersfund with @peterthiel, alongside @PanteraCapital and @hiFramework. This marks a pivotal step in aligning AI development t...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1258135475609145376)** (34 messages🔥): 

> - `openai AV issues during AIEWF demo`
> - `migration to Zoom for better accessibility`
> - `Discord's incompatibility with Linux and proposed alternatives` 


- ****OpenAI's AV Struggles during AIEWF Demo****: Members experienced significant AV issues with Discord during the **OpenAI AIEWF demo**, leading to frustration and multiple participants being unable to see the screen. This resulted in a suggestion to switch platforms.
- ****Zoom Migration for Better Accessibility****: Due to continuous AV issues on Discord, members unanimously agreed to migrate to [Zoom](https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09) for the **Paper Club (West)** session. This shift aimed to resolve visibility problems and improve meeting quality.
- ****Discord Incompatibility with Linux****: Participants highlighted a **known issue** with Discord's compatibility on Linux, leading to additional accessibility challenges. Alternatives were briefly discussed, suggesting a need for a more reliable platform moving forward.



**Link mentioned**: <a href="https://us06web.zoom.us/j/8807908941?pwd=eHBBdk9sWWluSzB2TFdLOVdEN3BFdz09">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1258025787882475611)** (2 messages): 

> - `jaan.li introducing their work at onefact.org and usb.club`
> - `san.tosh inquiring about updates on open GPT-4o` 


- **jaan.li builds decentralized edge transformers**: **jaan.li** announced their work at [onefact.org](https://onefact.org) and usb.club on decentralized edge transformers. Contact them anytime at jaan@onefact.org.
- **Inquiry on updates for open GPT-4o**: **san.tosh** asked if there are any updates regarding open GPT-4o. The query remains open.


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1257916724435615814)** (59 messages🔥🔥): 

> - `Ablation tests and justification of changes for Terminator models`
> - `Discussions on slow-fast networks and their advantages`
> - `Released code for Terminator on GitHub`
> - `Introduction of FORA for accelerating Diffusion transformers`
> - `Critiques and suggestions for the HyperZ⋅Z⋅W paper` 


- ****Terminator model ablation tests criticized****: Members discussed the lack of sufficient ablation tests and justification of changes in the *Terminator* models, though benchmarks showed impressive performance. They emphasized the need for detailed ablation studies to highlight the impact of individual components *(e.g., residuals, dot product attention, intermediate pooling)*.
- ****Debate on QKV redundancy in ViT****: There was a heated debate on whether QKV in Vision Transformers (ViT) is redundant, with suggestions that Q & K might be unnecessary for attention matrix generation. Some members believe a proper evaluation and proof are required to validate this theory.
- ****Terminator code released on GitHub****: **Terminator** code has been released on GitHub for public access, contrary to some claims of it being vaporware. Users can now explore the **official repository** [here](https://github.com/hyperevolnet/Terminator).
- ****FORA speeds up Diffusion transformers****: A new approach, **Fast-FORward CAching (FORA)**, was introduced to accelerate Diffusion transformers by caching and reusing intermediate outputs, significantly reducing computational overhead. This method integrates seamlessly with existing models and provides faster processing with minimal trade-offs in quality. [Read more](https://github.com/prathebaselva/FORA?tab=readme-ov-file).
- ****HyperZ⋅Z⋅W paper receives mixed reviews****: The **HyperZ⋅Z⋅W** paper by @harvie_zhang received both praise and critique from the community, noting it as a rough submission with novel ideas for achieving SOTA. The author acknowledged the feedback, indicating future revisions and potential ablation studies to demonstrate the redundancy of QKV in ViT. *Read the survey by Schmidhuber [here](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html).*


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/kronk-disney-the-emperor%E2%80%99s-new-groove-emperor%27s-new-groove-disney%E2%80%99s-emperor%E2%80%99s-new-groove-gif-9209845644877110421">Kronk Disney GIF - Kronk Disney The emperor’s new groove - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/prathebaselva/FORA?tab=readme-ov-file">GitHub - prathebaselva/FORA</a>: Contribute to prathebaselva/FORA development by creating an account on GitHub.</li><li><a href="https://github.com/hyperevolnet/Terminator">GitHub - hyperevolnet/Terminator: The official repository for HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction.</a>: The official repository for HyperZ⋅Z⋅W Operator Connects Slow-Fast Networks for Full Context Interaction. - hyperevolnet/Terminator
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1257824887288893440)** (26 messages🔥): 

> - `Image dtype special treatment`
> - `Runtime error in tinygrad`
> - `UNMUL pattern matcher issue`
> - `Frontend fuzzer idea`
> - `Loop optimization bug` 


- ****Tinygrad faces runtime error on UOps.UNMUL****: A member reported a **RuntimeError** in tinygrad's codebase: *'failed to render UOps.UNMUL'*. **George Hotz** suggested treating the issue as an assert and that it should *'never happen'*.
   - *Chenyuy* proposed adding `flat_l4.realize()` as a potential workaround and suggested making loop collapse optional to mitigate the impact on users.
- ****Frontend fuzzer proposal for tinygrad****: *Chenyuy* proposed the idea of a **frontend fuzzer** for tinygrad, possibly using an LLM to port torch code. The suggestion aims to catch more edge cases and unexpected behaviors during development.
- ****Handling the UNMUL pattern matcher bug****: *Chenyuy* noted that altering input dimensions sometimes prevents triggering the bug, which underscores incomplete loop optimization. *Yosifrost* found that specific dimensions impact the occurrence of the bug, suggesting an issue with heuristic boundary behaviors.
   - Members discussed the possibility of writing a minimal reproduction (repro) test to isolate the bug and expressed intentions to leave the PR open for further investigation and focused testing.
- ****Error handling and dev tooling improvements before 1.0****: *Yosifrost* stressed the necessity for **better error messages** and dev tooling in tinygrad before version 1.0. Various members collaborated to reproduce the error and develop a minimal test case for further debugging.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1257774565673795748)** (33 messages🔥): 

> - `Equivalent of torch.no_grad() in tinygrad`
> - ``-=` operator incompatibility with gradient enabled in tinygrad`
> - `Handling gradient accumulation issues leading to CUDA memory errors`
> - `Slowdown and memory issues with TinyJit during gradient accumulation`
> - `Behavior of Tensor creation methods in tinygrad vs PyTorch` 


- ****Tinygrad has torch.no_grad() equivalent****: A user asked about the equivalent of `torch.no_grad()` in tinygrad, and it was explained that there’s `Tensor.no_grad = True` and the `@Tensor.inference_mode()` decorator with examples provided [here](https://github.com/tinygrad/tinygrad/blob/master/examples/mlperf/model_train.py).
- ****Incompatibility of `-=` operator with gradient enabled****: A user noted that `a -= lr * a.grad` will assert while `a = a - lr * a.grad` works due to gradient restrictions in tinygrad. The issue was illustrated with a reference to the code [here](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py#L228).
- ****Issues with gradient accumulation and CUDA memory****: Users discussed problems with gradient accumulation leading to CUDA memory out of bounds errors. Suggestions included detaching the loss and addressing `assert t.grad is not None` from the optimizer.
- ****Errors caused by TinyJit in optimization steps****: It was revealed that TinyJit fails with `assert t.grad is not None` when not used on the entire step, causing inefficiencies. Users suggested returning realized gradients from the jit function and calculating the step externally.
- ****Tensor creation method inconsistency in tinygrad****: A user observed that `Tensor.randn/randint` create contiguous tensors, whereas `Tensor.full` creates non-contiguous tensors, differing from PyTorch's behavior. It was confirmed as expected behavior in tinygrad, with potential improvements discussed for future versions.



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py#L228">tinygrad/tinygrad/tensor.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1257852461645959179)** (3 messages): 

> - `Building a RAG pipeline on Raspberry Pi`
> - `OpenContracts AI-powered document analytics tool`
> - `Webinar on RAG experimentation and evaluation with Weights & Biases` 


- **RAG on Raspberry Pi 🍓🔎**: A tutorial by @pavan_mantha1 demonstrates how to build a **RAG pipeline** on a **Raspberry Pi** using **Docker** and **Ollama**. Check out the [tweet](https://twitter.com/llama_index/status/1808292764129583179) for more details.
   - This project showcases how the **RAG pipeline** can be efficiently run on a small, embedded device like the **Raspberry Pi**.
- **OpenContracts Launches ✨**: **OpenContracts**, an open-source AI-powered document analytics tool by @johnscrudato, allows users to analyze, annotate, and share documents using **LLMs** and **Llama Index**. More information can be found [here](https://twitter.com/llama_index/status/1808528869252812902).
   - This project is **genAI native** and aims to democratize document analytics by using **AI** efficiently.
- **Webinar on RAG Experimentation 🚨**: A webinar on **RAG Experimentation and Evaluation** is being held in partnership with **Weights & Biases**. The session aims to teach how to build, evaluate, and iterate on **RAG pipelines**, as detailed in this [tweet](https://twitter.com/llama_index/status/1808589017744880062).
   - With over a year of **RAG development**, this webinar addresses the challenge of proper evaluation in the field.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1257789737964671127)** (49 messages🔥): 

> - `DocumentSummaryIndex issues with Pinecone limits`
> - `Code snippets and potential fixes for metadata exclusion`
> - `Alternative vector stores to Pinecone`
> - `LlamaIndex's single LLM support`
> - `Parsing issues with PDF tables` 


- ****DocumentSummaryIndex hits Pinecone limits****: A user reported issues with **Big Docs** exceeding **Pinecone limits** when creating a **DocumentSummaryIndex**. The issue arises from the first node's metadata being too large, and **embed exclusion filters** seemingly not applied properly. [GitHub link](https://github.com/run-llama/llama_index/blob/722cb67ca4e52c8c4d6ef8c5e99b7f6c9f57e244/llama-index-core/llama_index/core/indices/document_summary/base.py#L203)
   - Another user suggested not including too much metadata in documents/nodes and offered a potential code fix to exclude embedding metadata keys. Additionally, they suggested considering alternatives to Pinecone like **qdrant** or **pg_vector**.
- ****LlamaIndex supports only OpenAI LLMs****: Multiple users noted that **LlamaIndex currently only supports OpenAI as its LLM**, which was met with some dissatisfaction. They suggested that expanding support to other LLMs would be beneficial.
   - *'Looks like it currently only supports OpenAI as LLM ... if so then 👎'*
- ****PDF to Markdown parsers struggle with tables****: A user tried converting PDFs to Markdown using [Marker](https://github.com/VikParuchuri/marker) but found that 'funkily' formatted tables caused parsing issues. They are looking for a better local or open-source solution but mentioned that **Azure Document Intelligence** performs better.
   - Another user recommended trying **Unstructured** or **llamaparse**, even though they are not open-source. These tools seem to handle complex table structures better.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/blob/722cb67ca4e52c8c4d6ef8c5e99b7f6c9f57e244/llama-index-core/llama_index/core/indices/document_summary/base.py#L203">llama_index/llama-index-core/llama_index/core/indices/document_summary/base.py at 722cb67ca4e52c8c4d6ef8c5e99b7f6c9f57e244 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/extraction/">Structured Data Extraction - LlamaIndex</a>: no description found</li><li><a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>: Convert PDF to markdown quickly with high accuracy - VikParuchuri/marker
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1258027183125954571)** (5 messages): 

> - `Agentic RAG with LlamaIndex, Claude-3.5 Sonnet, and MongoDB`
> - `Toolio for running private AI/LLM agents and tool-calling workflows on Mac` 


- ****Agentic RAG gains interest****: [**Unleashing AI Potential: Agentic RAG with LlamaIndex, Claude-3.5 Sonnet, and MongoDB**](https://medium.com/ai-advances/unleashing-ai-potential-agentic-rag-with-llamaindex-claude-3-5-sonnet-and-mongodb-ea126164a801) article discusses innovative strategies in the realm of AI. A member hinted it will be promoted soon.
- ****Toolio simplifies private AI workflows on Mac****: [Toolio](https://www.youtube.com/watch?v=9DpQYbteakc) allows users to run private AI/Large Language Model (LLM) agents & tool-calling workflows on their Mac with ease, supporting JSON schema constraints and offering fast inference. An advocate claimed tool-calling is the 'true magic with LLMs' and anticipates meaningful innovations in this area.


  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1257899774972268584)** (1 messages): 

> - `Tortoise-TTS converted to ggml`
> - `Optimization for real-time inference`
> - `Open-source projects on GitHub`
> - `CUDA and CPU support for Tortoise-TTS` 


- ****Tortoise-TTS Conversion to GGML****: A member converted **Tortoise-TTS** to **ggml** and is seeking assistance to improve inference time for [real-time text-to-speech](https://github.com/balisujohn/tortoise.cpp). The repository already supports CUDA and CPU.
- ****Optimization Opportunity for AI Developers****: The [Tortoise-TTS ggml project](https://github.com/balisujohn/tortoise.cpp) offers a great chance to practice optimizing **transformers** and **diffusion models**. The goal is to speed up the inference process.



**Link mentioned**: <a href="https://github.com/balisujohn/tortoise.cpp">GitHub - balisujohn/tortoise.cpp: A ggml (C++) re-implementation of tortoise-tts</a>: A ggml (C++) re-implementation of tortoise-tts. Contribute to balisujohn/tortoise.cpp development by creating an account on GitHub.

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1257849473309212743)** (42 messages🔥): 

> - `Private channels on decentralized training`
> - `Tool calling in vLLM for Hermes 2 Pro`
> - `Discussion on handling tool calls and text content in Hermes 2 Pro` 


- ****Tool calling working properly in vLLM for Hermes 2 Pro****: A member announced that **tool calling is now functioning properly in vLLM for Hermes 2 Pro**. They indicated the project is very close to completion.
- ****Hermes 3 training includes <scratch_pad>****: The team discussed the addition of `<scratch_pad>` before tool calls in **Hermes 3 training**, aiming for improved parsing that extracts between `<scratch_pad>` and handles both 'content' and 'tool_calls'.
   - *Discussions included handling text content before tool calls and ensuring compatibility with OpenAI's spec.*



**Link mentioned**: <a href="https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ">Neural Networks: Zero to Hero</a>: no description found

  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1257773242156847146)** (3 messages): 

> - `Creating conversational dataset from documents`
> - `Instruction-generation from documents`
> - `Genstruct 7B model for generating instructions` 


- ****Creating datasets from documents****: **Creating conversational datasets** from documents depends on the document and budget. Options include generating datasets using a language model or using tools like those from **Anthropic**.
- ****Genstruct 7B generates instructions****: The [Genstruct 7B](https://huggingface.co/NousResearch/Genstruct-7B) model generates valid instructions from a raw text corpus, useful for creating instruction finetuning datasets. It’s inspired by the [Ada-Instruct](https://arxiv.org/abs/2310.04484) model.



**Link mentioned**: <a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1257918603588800626)** (5 messages): 

> - `Huggingface's PR on Cohere's CommandR model`
> - `Microsoft's GraphRAG release` 


- **Huggingface PR on Cohere CommandR Expands Tool-Use**: Huggingface opened a [PR](https://github.com/cohere/CommandR) on Cohere's CommandR model, focusing on *technical improvements for tool-use and RAG templates*. The system prompt is formulated using a preamble and dynamic content arrangement with Jinja templating.
- **Microsoft Releases GraphRAG**: Microsoft released a modular graph-based Retrieval-Augmented Generation (RAG) [system](https://github.com/microsoft/graphrag) called *GraphRAG*, designed to boost information retrieval and generation. The tool is available on GitHub and aims to enhance modularity and effectiveness in RAG systems.



**Link mentioned**: <a href="https://github.com/microsoft/graphrag">GitHub - microsoft/graphrag: A modular graph-based Retrieval-Augmented Generation (RAG) system</a>: A modular graph-based Retrieval-Augmented Generation (RAG) system - microsoft/graphrag

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1257907166334156832)** (7 messages): 

> - `Installation issues on Ubuntu 24.04/Python 3.12.3`
> - `Implementation workaround for Mojo/max on Ubuntu 24.04/Python 3.12.3`
> - `Mojo implicit conversion bug`
> - `Casting bug in Mojo` 


- ****Facing installation issue on Ubuntu 24.04****: A user reported an installation issue on **Ubuntu 24.04/Python 3.12.3**, receiving errors for **max-engine** due to version mismatches.
   - Another user shared a [step-by-step guide](https://docs.modular.com/mojo/manual/python/#resolving-issues) to resolve this by installing Python 3.11 and adjusting alternatives.
- ****Mojo's odd implicit conversion****: A user noticed that multiplying integers by `np.pi` in Mojo produces an unexpected **negative integer** result due to an [implicit conversion bug](https://github.com/modularml/mojo/issues/3146).
   - The discussion pointed out that this is related to casting bugs already tracked as [#3065](https://github.com/modularml/mojo/issues/3065) and [#3167](https://github.com/modularml/mojo/issues/3167).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/python/#resolving-issues">Python integration | Modular Docs</a>: Using Python and Mojo together.</li><li><a href="https://github.com/modularml/mojo/issues/3146).">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/3065)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/3167)">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1257789972505825382)** (2 messages): 

> - `` 


- ****Modular Shares Exciting Updates on Twitter****: [Modular](https://twitter.com/Modular/status/1808228006068212110) tweeted interesting updates recently. More details can be found in their latest Twitter announcements.
- ****Further Announcements from Modular on Twitter****: [Modular](https://twitter.com/Modular/status/1808567651280777598) shared additional updates a few days later. Check their Twitter for the full posts.


  

---


### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1258126192523874418)** (1 messages): 

> - `Mojo N-Body Example Benchmark`
> - `Single-core Numeric Performance in Mojo`
> - `Symplectic Integrator in N-body.js`
> - `Vectorization in N-Body Example`
> - `Ordinary Differential Equation Solver` 


- **Mojo Introduces N-Body Example**: [Modular's blog](https://www.modular.com/blog/a-brief-guide-to-the-mojo-n-body-example) details the **Mojo N-Body Example** included in the repository since August 2023, based on [The Computer Language Benchmarks Game](https://en.wikipedia.org/wiki/The_Computer_Language_Benchmarks_Game). This benchmark simulates Jovian planets' orbits and exercises single-core numeric performance.
- **N-body Benchmark Highlights**: [N-body is one of the Computer Language Benchmark Game's benchmarks](https://benchmarksgame-team.pages.debian.net/benchmarksgame/description/nbody.html#nbody) modeling **Jovian planets using a symplectic integrator**. Although primarily single-core, basic vectorization can be implemented to enhance performance.



**Link mentioned**: <a href="https://www.modular.com/blog/a-brief-guide-to-the-mojo-n-body-example">Modular: A Brief Guide to the Mojo N-Body Example</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: A Brief Guide to the Mojo N-Body Example

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1257817463840575630)** (8 messages🔥): 

> - `Mojo List types and printing issues`
> - `Printing RepresentableCollectionElement types`
> - `Printing errors inline in Mojo`
> - `Impact of excessive empty lines on startup time`
> - `Variable startup times in Mojo programs due to loop unrolling in bench_matmul` 


- ****Mojo List Confusion****: A member expressed confusion about `List[String]` in Mojo, noting that although it contains strings, it lacks the `Stringable` trait, affecting its printability. They provided a [GitHub link](https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/builtin/str.mojo#L23) for reference.
- ****Printing RepresentableCollectionElement types in Mojo****: A suggestion was made to use `print(list.__str__())` to print Lists of `RepresentableCollectionElement` types in Mojo, providing a [GitHub link](https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/collections/list.mojo#L338) for more details.
- ****Variable Startup Times Due to Loop Unrolling****: A user noted that their Mojo program had variable startup times, initially linking it to excessive empty lines. Another user clarified that it was due to `bench_matmul` unrolling a lot of loops, making it slow to compile.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/builtin/str.mojo#L23">mojo/stdlib/src/builtin/str.mojo at 8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2/stdlib/src/collections/list.mojo#L338">mojo/stdlib/src/collections/list.mojo at 8bd1dbdf26c70c634768bfd4c014537f6fdb0fb2 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/)** (1 messages): 

melodyogonna: The joys of early tooling
  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1257810211893285017)** (31 messages🔥): 

> - `Parallel processing in MLM`
> - `Matrix Multiplication Optimization`
> - `Strassen Algorithm Performance`
> - `SPIRAL Project`
> - `Numerical Stability in Matrix Multiplication` 


- **SPIRAL Project Aims at Automated High-Performance Libraries**: The [SPIRAL project](http://www.spiral.net/) focuses on automating software and hardware development for DSP algorithms and other numerical kernels, often outperforming MKL in direct hardware tasks.
- **Strassen Algorithm vs Naive Vectorized Approach**: **Strassen Algorithm** achieved around **50 GFlops** while the naive vectorized and parallelized version hit **70 GFlops** for 1024x1024 matrices. Details on [GitHub](https://github.com/RedKinda/Mojo-Marathons/).
   - Optimizations like vectorizing adds/subs and parallelizing sub-matrix multiplications added 3-5 extra GFlops, while [fewer intermediate allocations](https://github.com/RedKinda/Mojo-Marathons/) and better guardrails for non-square matrices are potential future improvements.
- **Parallel and Vectorized Operations Conceptual Challenge**: Discussions highlighted the conceptual difficulty beyond parallel, vectorize, unroll, and parts of tiling for tuning algorithms.
- **Numerical Stability Impact in Strassen Algorithm**: The [Strassen Algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm) reportedly reduces numerical stability, leading to test failures when different types and sizes were used.
- **Recursive vs Iterative Algorithms for Cache Locality**: It's suggested that different type sizes might benefit from recursive algorithms over iterative ones for better cache locality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://www.spiral.net/">SPIRAL Project: Home Page</a>: no description found</li><li><a href="https://docs.google.com/spreadsheets/d/1TBz9Lp0JT1Ph7ndfbWqp-B30FQcRYl1959hP2lZ6yH4/edit">Matrix Multiplication</a>: Sheet1  Contstraints,Parameters / Tuning Vectorization,Contiguous Access,Nelts, Unrollable Parallelization,Unrollable Unrolling,Contiguous Operations Tiling Square Optimized,Amorized Increase,Recursiv...
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1257821853884616804)** (29 messages🔥): 

> - `Apple getting a board observer seat at OpenAI`
> - `Microsoft investments in OpenAI and comparison with Apple's partnership`
> - `Kyutai Labs' new real-time audio LLM 'Moshi'`
> - `Training details and technical specifics of 'Moshi'`
> - `Kyutai Labs' open model releases and future plans` 


- **Apple Gets a Board Observer Seat at OpenAI**: Apple will get a **board observer seat** at OpenAI later this year as part of its partnership for Apple Intelligence, with **Phil Schiller** occupying the seat, as reported by [Bloomberg](https://www.bloomberg.com/news/articles/2024-07-02/apple-to-get-openai-board-observer-role-as-part-of-ai-agreement).
- **Microsoft vs. Apple in OpenAI Partnerships**: Community members compared **Microsoft's billion-dollar investment** in OpenAI to **Apple's partnership**, noting that Apple seems to be getting more benefits, including an app and iPhone integration, while Microsoft does not.
- **Kyutai Labs Unveils 'Moshi': The Real-Time Audio LLM**: **Kyutai Labs** introduced **Moshi**, the first real-time Audio LLM with **150ms latency**, during a live update, showcasing its **multimodal LM capabilities** with the potential for on-device use.
- **Moshi's Impressive Technical Details**: Moshi is built on a **7B multimodal LM**, has a **VQ-VAE-based speech codec** with a 300x compression factor, and delivers **superhuman response times**, according to multiple community members.
- **Open Model Releases and Future Plans for 'Moshi'**: Kyutai Labs plans to release **open models** including the 7B multimodal LM, audio codec, and an optimized stack. Users have already begun testing the model **in real-world scenarios** and discussed its response latency and potential uses.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/giffmana/status/1808482848808010149">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: Kyutai Moshi - first real-time Audio LLM.  Basically no delay - the LLM even interrupted the speaker a few times. It was actually a bit eager to answer very quick. :)  All to be open-sourced. Quality ...</li><li><a href="https://x.com/BartokGabi17/status/1808242102750568799">Tweet from Bartok Gabriel (@BartokGabi17)</a>: @markgurman Microsoft invests bilion in open Ai dosent get an app  Apple pays litterally in exposure, open Ai makes a great app an big iPhone integration  Profit??  Tim Apple it&#39;s a genius</li><li><a href="https://x.com/thexeophon/status/1808481304117227794?s=46">Tweet from Xeophon (@TheXeophon)</a>:   Quoting kyutai (@kyutai_labs)   Join us live tomorrow at 2:30pm CET for some exciting updates on our research!  https://www.youtube.com/live/hm2IJSKcYvo</li><li><a href="https://x.com/reach_vb/status/1808528557431210236">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>: Fuck yeah! Moshi by @kyutai_labs just owned the stage! 🇪🇺/acc.  Architecture 1. 7B Multimodal LM (speech in, speech out) 2. 2 channel I/O - Streaming LM constantly generates text tokens as well as a...</li><li><a href="https://x.com/markgurman/status/1808240961522159862">Tweet from Mark Gurman (@markgurman)</a>: NEW: Apple will get a board observer seat at OpenAI later this year as part its partnership for Apple Intelligence. The person getting the seat: Phil Schiller, the head of the App Store and former mar...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1257958333252374609)** (12 messages🔥): 

> - `SB 1047 and First Amendment challenges`
> - `Protection of 3D gun designs under the First Amendment`
> - `Model weights and code as protected speech`
> - `Claude 3.5 admiration`
> - `Use Claude TM` 


- ****SB 1047 faces potential First Amendment challenge****: A discussion emerged on whether **SB 1047** would survive a **First Amendment challenge** if it passes, particularly comparing it to the protection of **3D gun designs** and code as free speech, referencing the [EFF's case on code as speech](https://www.eff.org/deeplinks/2015/04/remembering-case-established-code-speech).
- ****Courts on code as protected language****: A quote highlighted that there is 'no meaningful difference between computer language and German or French,' citing that both communicate information and thus are protected under the First Amendment, likening them to 'music and mathematical equations.'
- ****Debate on model weights as protected speech****: Debate continued on whether **model weights** could be considered protected under the First Amendment, with discussions around their classification as 'high-level languages' or merely 'mathematical equations,' and comparisons to publishing random sequences of words.
- ****Claude 3.5 hype****: "woo Claude 3.5 admiration let's go" – sparked excitement among members regarding the recent release of **Claude 3.5**.
- ****Promoting Claude TM****: A suggestion was made to **promote Claude TM**, comparing its campaigns to those of **American Airlines** with the enthusiastic remark, *'I never would go back.'*



**Link mentioned**: <a href="https://www.eff.org/deeplinks/2015/04/remembering-case-established-code-speech)">Deeplinks Blog</a>: no description found

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1257789781291696138)** (30 messages🔥): 

> - `RAG strategies for answering general questions`
> - `Rate limiting issues with AzureAIDocumentIntelligenceLoader`
> - `Parsing PDFs with different tools and libraries`
> - `LangSmith tracing issues`
> - `General help and troubleshooting in LangChain` 


- **Azure and PDF Loaders Clash**: A user switched from **PyPDFium2Loader** to **AzureAIDocumentIntelligenceLoader** and encountered a consistent **429 error (Too Many Requests)**. This suggests possible rate limiting issues due to the method AzureAIDocumentIntelligenceLoader processes documents.
- **PDF to Markdown Pain Points**: A member tried using [marker](https://github.com/VikParuchuri/marker) to convert PDFs to markdown but faced issues with parsing tables with 'funky' formatting like merged cells. They noted that **Azure Document Intelligence** worked better for such documents but expressed a preference for a local or open-source alternative.
- **LangSmith Stops Tracing Calls**: A user reported that **LangSmith** stopped tracing their calls without any clear reason. This suggests potential bugs or issues in the tracing mechanism of LangChain.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://v02.api.js.langchain.com/classes/langchain_community_document_loaders_fs_pdf.PDFLoader.html">PDFLoader | LangChain.js - v0.2.8</a>: no description found</li><li><a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>: Convert PDF to markdown quickly with high accuracy - VikParuchuri/marker</li><li><a href="https://community.openai.com/t/using-gpt-4-api-to-semantically-chunk-documents/715689/136">Using gpt-4 API to Semantically Chunk Documents</a>: OK, after  2 months, I’ve got a fully functional system up and running in real time.  This is the process:      export the pdf (or whatever) document to txt.  I am set up to use: AWS Textract, PdfToTe...</li><li><a href="https://python.langchain.com/v0.2/docs/integrations/toolkits/openapi/#lets-see-some-examples>).">OpenAPI | 🦜️🔗 LangChain</a>: We can construct agents to consume arbitrary APIs, here APIs conformant to the OpenAPI/Swagger specification.</li><li><a href="https://github.com/langchain-ai/langchain/issues/832>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/2333>).">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1257774280767180820)** (2 messages): 

> - `uploading CSV files directly vs. providing file path`
> - `output not displaying in CSV playground`
> - `code improvement for CSV handling`
> - `FastAPI endpoint for file uploads`
> - `Chroma vectorstore usage and issues` 


- ****Enable Users to Upload CSV Files****: A user seeks help to enable CSV file uploads in their project, instead of users specifying file paths. They need a way to implement this in their FastAPI setup for better usability.
- ****No Output in CSV Playground****: In the `csv/playground/` directory, there's an issue where no output is displayed even though the code seems correct. This indicates a potential problem in the file handling or output rendering logic.
- ****Improving CSV Handling Code****: The user is looking for guidance to improve their existing code, which currently requires users to set the file path. Suggestions to enhance code efficiency and usability are needed.


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1257773671532204032)** (2 messages): 

> - `OpenAI CriticGPT paper discussion`
> - `Toolio open source project for private LLMs` 


- **OpenAI Introduces CriticGPT to Identify GPT-4 Mistakes**: A member shared a [YouTube video](https://youtu.be/4PgcaIfwLjo) discussing OpenAI's latest paper on **CriticGPT**, which is designed to correct errors made by **GPT-4**. The video highlights the key features of CriticGPT and its significance in improving the reliability of AI-generated code.
- **Toolio Empowers Private LLM Workflows on Mac**: A member announced the release of **Toolio**, an open source project that enables running private **LLM agents** and tool-calling workflows on Mac with ease. The project, showcased in a [YouTube video](https://www.youtube.com/watch?v=9DpQYbteakc), also features JSON schema output constraints and fast inference capabilities.



**Link mentioned**: <a href="https://youtu.be/4PgcaIfwLjo">OpenAI releases CriticGPT to correct GPT-4&#39;s mistakes | Read the paper with me</a>: OpenAI has unveiled CriticGPT, a new AI model based on GPT-4 designed to identify errors in code generated by ChatGPT, marking a significant step towards imp...

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/)** (1 messages): 

dracount: hi, is there a beginner langchain/langraph tutorial that anyone can recommend?
  

---



### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1257796022168584302)** (25 messages🔥): 

> - `Hardware recommendations for running llamafile`
> - `VRAM and CPU usage for large language models`
> - `Syncthread trick for CPU inference`
> - `Running llama3 70B on high-end workstation`
> - `Issues with Rockchip RK3588 NPUs support for llamafile` 


- ****Building the Best Llamafile Setup****: Members discussed hardware recommendations for a new Linux computer to run **llamafile** effectively. Recommended GPUs include **3090/4090** for consumer use and **A6000/RTX 6000 Ada** for workstations; at the higher end, older **EPYC CPUs** were suggested for their cores and PCIe lanes support.
- ****VRAM Needs for Large Models****: To run large models efficiently, **more VRAM** is crucial; for example, 24GB VRAM can handle 33B parameters on **q4**. FP16 mode is discouraged due to its massive VRAM requirements compared to minimal quality loss.
- ****Syncthread Trick in CPU Inference****: Members discussed the potential of **CPU-learning** with the syncthread trick used for CPU inference. A [YouTube talk](https://www.youtube.com/live/5zE2sMka620?feature=shared&t=3140) was shared explaining the concept.
- ****Running Llama3 70B on Threadripper****: A member successfully ran **llama3 70B** on a high-end **Threadripper CPU** workstation. This shows the capability of CPU for handling large models when adequate specifications are met.
- ****Issues with Rockchip RK3588 NPUs****: There were questions about running **llamafile** on **Rockchip RK3588 NPUs**. It was suggested to use version **v0.8.9** to avoid address space issues on such hardware.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/live/5zE2sMka620?feature=shared&t=3140">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>: https://twitter.com/aidotengineer</li><li><a href="https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#memorydisk-requirements">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: LLM inference in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1257826237683077130)** (22 messages🔥): 

> - `phi mini new weights same repo`
> - `torchtune evaluation using eleutherai's eval harness`
> - `evaluation basics and logs on wandb`
> - `discussion about gradients and epochs for training`
> - `FullModelHFCheckpointer and conversion between HF format and torchtune` 


- **New weights for phi mini**: **phi mini** has received new weights but is reusing the same repository as before. Users are assuming the old recipe will work without updates for torchtune.
- ****Training strategies and evaluations****: Discussion on different **gradients 8 vs 16** and batch size 2 to determine which works better for a dataset and whether to adjust epochs. **Wandb logs** and evaluation tutorials were shared to aid in better understanding and tracking performance metrics.
- **Conversion arguments for HF formats**: Queries about why conversion needs parameters like `num_heads`, `num_kv_heads`, and `dim`. **Conversion** is needed to switch between HF grouped multihead layers and torchtune's separate layers.
- **Checkpointers for torchtune**: torchtune's **FullModelHFCheckpointer** automatically converts checkpoints to HF format. Details were shared about **how the checkpointer ensures** compatibility with various tools and handles different formats.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/lmn07r/torchtune/reports/Untitled-Report--Vmlldzo4NTM2NDMw?accessToken=2yedg0bvpgy3fuoaec70tzdm0mqdklzj6bf66kavth4ygoh2ag6klda4tr75mw8t">Untitled Report</a>: Publish your model insights with interactive plots for performance metrics, predictions, and hyperparameters. Made by Lemon R using Weights &amp; Biases</li><li><a href="https://wandb.ai/lmn07r/torchtune/workspace?nw=nwuserlemon07r">lmn07r</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://pytorch.org/torchtune/main/deep_dives/checkpointer.html">Checkpointing in torchtune &mdash; torchtune main documentation</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/models/convert_weights.py#L162">torchtune/torchtune/models/convert_weights.py at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html#generation)">End-to-End Workflow with torchtune &mdash; TorchTune  documentation</a>: no description found</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html#run-evaluation-using-eleutherai-s-eval-harness)">End-to-End Workflow with torchtune &mdash; TorchTune  documentation</a>: no description found</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/models/convert_weights.py#L162.">torchtune/torchtune/models/convert_weights.py at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/transformers/blob/f91c16d270e5e3ff32fdb32ccf286d05c03dfa66/src/transformers/models/llama/modeling_llama.py#L262">transformers/src/transformers/models/llama/modeling_llama.py at f91c16d270e5e3ff32fdb32ccf286d05c03dfa66 · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1257899648480182373)** (14 messages🔥): 

> - `Training LLM with Stockfish data`
> - `Usage of tools like Stockfish for reasoning in LLMs`
> - `GitHub notebook code`
> - `Chess strategy and LLMs`
> - `Cohere API tools` 


- ****Mixing LLMs with Stockfish for Better Planning**: **: A user raised a question about using **Stockfish data** to improve **LLM reasoning** and planning abilities, or to develop a quick **chess engine**.
- ****Fine-Tuning LLMs with Chess Data**: **: *One member* shared their experience of **fine-tuning LLMs** with chess data, highlighting that it requires significant overfitting, which can be problematic. There's a debate on the effectiveness and utility of such an approach.
- ****Using Tools for Chess in Cohere's Platform**: **: An interesting perspective suggested that **LLMs** should use tools like **Stockfish** for better results in chess understanding, rather than being trained directly on chess data.



**Link mentioned**: <a href="https://github.com/">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...

  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1257811755141173379)** (5 messages): 

> - `Creation of a Cohere Slack bot`
> - `Discussion on Slack's request handling`
> - `Lack of documentation on Slack integration`
> - `Offer to share script and create documentation` 


- ****Cohere Slack Bot Simplifies Workspace Interaction****: A member shared their creation of a **Cohere Slack bot** designed for workspace convenience, enhancing accessibility.
- ****Slack's Efficiency Challenges Bot Creation****: With Slack's requirement to complete requests in under **3 seconds**, this showcases the impressive speed of the **Cohere models** utilized.
   - *"I have a Cloudflare Worker handling requests"*, demonstrating practical integration solutions despite initial documentation complexities.
- ****Community Eagerly Awaits Bot Documentation****: The community showed enthusiasm for the bot creation, seeking guidance and documentation on replicating the process.
   - *"I will work on a tutorial with documentation and publish it to one of my domains"*, indicating future resource availability for interested members.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1257850552675471441)** (18 messages🔥): 

> - `Kyutai Moshi - real-time Audio LLM`
> - `Various Open Interpreter compatible projects`
> - `Experience modding games for Open Interpreter`
> - `Pull request for Open Interpreter Labs`
> - `MIke Bird and blurryboi discussion on Kyutai Moshi` 


- **Kyutai Moshi launches real-time Audio LLM**: **Kyutai Moshi** released the first real-time **Audio LLM**, with no delay but some robotic quality, which will be open-sourced. [Live demo is available](https://www.moshi.chat/?queue_id=talktomoshi).
   - *Mikebirdtech* noted it was **FAST, almost too fast**, interrupting if pauses were too long.
- **Open Interpreter projects suggested by Techfren**: Several projects like **Open interpreter, taxyai, clickolas cage, self-operating computer, pywinassistant, GPT computer assistant** were listed as Open Interpreter compatible. They may require some configuration but are promising options.
- **Seeking modder experience for Open Interpreter**: **Nonadjective.eth_55058** expressed interest in modding a game to be compatible with Open Interpreter and sought advice from others with similar experience. They are open to clunky interfaces to create a proof of concept.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.moshi.chat/?queue_id=talktomoshi">moshi.chat</a>: no description found</li><li><a href="https://github.com/openinterpreterlabs">Open Interpreter Labs </a>: Open Interpreter Labs and Experiments (not directly affiliated with OI) - Open Interpreter Labs </li><li><a href="https://x.com/giffmana/status/1808482848808010149?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">Tweet from Lucas Beyer (bl16) (@giffmana)</a>: Kyutai Moshi - first real-time Audio LLM.  Basically no delay - the LLM even interrupted the speaker a few times. It was actually a bit eager to answer very quick. :)  All to be open-sourced. Quality ...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/)** (1 messages): 

johnlenflure: Isn't there a way to integrate 01 into glasses?
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1257774597852495972)** (1 messages): 

> - `Weighted cross entropy in the trainer` 


- ****Discovery of Weighted Cross Entropy Section****: A member noticed the **whole weighted cross entropy section** in the trainer. They mentioned they would look into it further.
- ****Lack of Additional Topics Noted****: No other significant topics were discussed in the given message history. Only **weighted cross entropy in the trainer** was mentioned.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1257965086111170580)** (6 messages): 

> - `Differences between LoRA and QLoRA quantization`
> - `Explanation of 8-bit quantization in LoRA`
> - `Efficiency of QLoRA in fine-tuning large models` 


- **LoRA is quantized to 8-bit**: **LoRA** applies 8-bit quantization while **QLoRA** goes further to use 4-bit quantization as clarified by users in the discussion.
   - The conversation highlighted how quantization is not just about decomposing matrices, referencing the [QLoRA paper](https://arxiv.org/abs/2305.14314) for a comprehensive explanation.
- **QLoRA facilitates efficient fine-tuning**: A member shared that the [QLoRA paper](https://arxiv.org/abs/2305.14314) demonstrates how QLoRA enables finetuning a 65B parameter model on a single 48GB GPU with near-full 16-bit finetuning performance.
   - *This paper introduces innovations like 4-bit NormalFloat (NF4) and double quantization to minimize memory usage without sacrificing performance,* according to a user discussing the content.



**Link mentioned**: <a href="https://arxiv.org/abs/2305.14314">QLoRA: Efficient Finetuning of Quantized LLMs</a>: We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLo...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1258113678390067371)** (2 messages): 

> - `torch.cuda.OutOfMemoryError on Google Colab`
> - `Axolotl running issues`
> - `GPU memory allocation`
> - `VRAM requirements` 


- ****CUDA Memory Woes in Google Colab****: A member encountered a **torch.cuda.OutOfMemoryError** while attempting to allocate 172.00 MiB GPU memory running **axolotl** on Google Colab.
- ****Axolotl Demands More VRAM****: In response to a CUDA memory error, another member advised that more **VRAM** is needed to avoid such issues.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1258031514638225408)** (5 messages): 

> - `Quantization and its impact on model performance`
> - `LoRA versus QLoRA configuration specifics`
> - `Memory footprint and inference speed improvements with 8-bit quantization` 


- **8-bit Quantization Explained**: **Quantization** reduces model precision from 32-bit or 16-bit down to 8-bit (int8), significantly reducing memory footprint and speeding up inference times. The `load_in_8bit` option in `lora.yaml` enables this quantization process for deploying large models on limited hardware.
- **Differences between LoRA and QLoRA**: While **LoRA** focuses solely on parameter-efficient fine-tuning, **QLoRA** combines low-rank adaptation with quantization. The inclusion of `load_in_8bit` in QLoRA configurations signifies the use of 8-bit quantization, as seen in various example files (`qlora.yml`).



**Link mentioned**: <a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=02e7bdf5-d8ec-486f-8697-c89ff466de3b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: Understand code, faster.

  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1257787172396859494)** (11 messages🔥): 

> - `Docker port for AI Town`
> - `GitHub Page for AI Town Windows Setup with WSL`
> - `API communication issues with Docker port of AI Town`
> - `Convex automatic download via Docker for AI Town`
> - `Testing Docker integration for AI Town` 


- ****Docker request for AI Town****: A member highlighted the need for a **Docker image** for AI Town as 'amazing' and useful, encouraging submissions to the main repo.
- ****GitHub setup guide for AI Town on WSL****: A member shared a [GitHub page](https://github.com/Ikkitsuna/AI-Town-Windows-Setup-WSL-method) for setting up AI Town on Windows using **WSL**, and it was suggested to submit a PR for incorporating it into the main repo.
- ****API issues in Docker port for AI Town****: While working on a **Docker port**, a member noted, 'I can run AI town with no problem except one: **Ollama and other API communication**,' and promised updates once a solution is found.
- ****Automation of Convex download in Docker****: A member is finalizing adjustments so that **Convex** automatically downloads via Docker, aimed at simplifying user experience and planning to go live around 8 p.m. UTC+4.
- ****Request for help with Docker integration testing****: A request for testing Docker integration was made before submitting a PR, but the member decided to proceed after successful tests on their **Legion Go**.



**Link mentioned**: <a href="https://github.com/Ikkitsuna/AI-Town-Windows-Setup-WSL-method">GitHub - Ikkitsuna/AI-Town-Windows-Setup-WSL-method: Guide for setting up AI Town on Windows using WSL</a>: Guide for setting up AI Town on Windows using WSL. Contribute to Ikkitsuna/AI-Town-Windows-Setup-WSL-method development by creating an account on GitHub.

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1257903763998511155)** (3 messages): 

> - `struggling to deploy RAG app using gradio on Modal`
> - `post on Modal Slack for help` 


- ****Struggling with RAG App Deployment****: A member mentioned facing issues deploying a **RAG app using Gradio** on Modal, despite it running fine locally and being deployed on Huggingface Spaces. They have exhausted all options and are looking for resources to figure out what went wrong.
- ****Modal Slack to the Rescue****: Another member suggested posting the issue on the [Modal Slack](https://modal.com/slack) for further assistance. The original member acknowledged the reminder and expressed their intent to do so.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[paige_when_finetune](https://discord.com/channels/1238365980128706560/1242224662142779530/)** (1 messages): 

shamik_53759: Yep, it's up now. Thanks!
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1257990993446436926)** (1 messages): 

> - `DeepSpeed configuration for data sharding and disabling model sharding`
> - `Assistance with DeepSpeed configurations`
> - `Confusion about DeepSpeed settings` 


- **Navigating DeepSpeed Configuration for Sharding**: A member expressed confusion over selecting the appropriate **DeepSpeed configuration** to enable **data sharding** and disable **model sharding**.
- **Request for Assistance with DeepSpeed Settings**: A request was made for assistance in choosing the correct **DeepSpeed settings** for enabling data sharding while disabling model sharding.


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1257930601730674719)** (1 messages): 

> - `Sharing private code deployments`
> - `Running code on multiple platforms`
> - `Limitations of private deployments on Hugging Face` 


- **Struggles with Private Code Deployments**: A user expressed difficulty in sharing private code with colleagues using Hugging Face private spaces, noting that **sharing=True** is not supported in private deployments.
- **Challenges Running Code on Modal**: Having issues with getting code to run on **Modal**, a user mentioned their struggles and expressed interest in alternative solutions for private code sharing.


  

---



### **LLM Perf Enthusiasts AI ▷ #[eval](https://discord.com/channels/1168579740391710851/1168986849784635553/1258055110236442729)** (1 messages): 

> - `Evaluating LLM accuracy in legal contract review`
> - `Screens tool achieving 97.5% accuracy`
> - `Methodologies for assessing LLMs in the legal domain`
> - `Impact of different LLMs and methods on AI accuracy in legal tasks` 


- ****Screens Report Evaluates LLM Accuracy in Legal Domain****: Screens' new [evaluation report](https://www.screens.ai/blog/screens-accuracy-evaluation-report) discusses treating LLM evaluations like traditional ML classification problems in the legal domain, particularly for contract reviews. They claim a **97.5% accuracy rate** for their system, highlighting its potential use in playbook execution and workflow routing.
- ****Accuracy Challenges and Methodologies Explored****: The report details the challenges in objectively evaluating long-form, free-text responses, proposing a methodology that uses classification standards to assess LLM performance. The approach can significantly aid in legal tasks like negotiation, redlining, and summarization.



**Link mentioned**: <a href="https://www.screens.ai/blog/screens-accuracy-evaluation-report">Screens Accuracy Evaluation Report</a>: Evaluating the accuracy of large language models (LLMs) on contract review tasks is critical to understanding reliability in the field. However, objectivity is a challenge when evaluating long form, f...

  

---


### **LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1258054107688865882)** (1 messages): 

> - `` 


- **Seeking Simple Prompt Tuning Tools**: **Evan_04487** is looking for a user-friendly tool for tuning templates prompts, tailored for non-technical stakeholders like designers and product managers. They need a hosted, freemium product that can run variations of templated prompts and manually inspect responses.
- **Requirement for Freemium Hosted Tool**: **Evan_04487** specified a preference for a freemium hosted tool for prompt tuning that can handle a couple dozen variables. They mentioned having a more robust self-service infrastructure for high-stakes exercises, but need something simpler for lower stakes.


  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/)** (1 messages): 

derekpwillis: https://thescoop.org/archives/2024/06/22/all-foreign-gifts-around-us/index.html
  

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

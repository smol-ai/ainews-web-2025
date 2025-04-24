---
id: bd83aaf6-a034-4ade-acd4-2426be77d923
title: not much happened this weekend
date: '2024-08-27T00:09:52.018820Z'
original_slug: ainews-not-much-happened-this-weekend
description: >-
  AI news highlights distributed AI optimizer from Nous Research, viral Cursor
  AI, George Hotz's tinybox, and Box AI beta. Recaps include humanoid robots
  from China, AI-generated robot motion, teleoperation systems, new AI models
  like Jamba 1.5, Dream Machine 1.5, Ideogram v2, Mistral-NeMo-Minitron 8B, and
  applications in autonomous sales.
companies:
  - nous research
  - cursor ai
  - box
  - agibot
  - unitree
  - eth zurich
  - disney
  - uc san diego
  - ai21 labs
  - luma labs
  - ideogram
  - nvidia
  - mistral
models:
  - distro
  - jamba-1.5
  - dream-machine-1.5
  - ideogram-v2
  - mistral-nemo-minitron-8b
topics:
  - distributed-ai
  - robotics
  - humanoid-robots
  - ai-generated-motion
  - teleoperation
  - multilingual-ai
  - text-to-video
  - text-to-image
  - small-language-models
  - ai-applications
---


<!-- buttondown-editor-mode: plaintext -->**we're running out of subtitles.**

> AI News for 8/23/2024-8/26/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**214** channels, and **5673** messages) for you. Estimated reading time saved (at 200wpm): **639 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

A few news items:

- **Distributed AI**: Nous Research [announced DisTrO](https://x.com/NousResearch/status/1828121648383566270), their new optimizers that "reduces the inter-GPU communication requirements by 1000x to 10,000x without relying on amortized analysis, and matches AdamW+All-Reduce in convergence rates. This enables low-latency training of large neural networks on slow internet bandwidths with heterogeneous networking hardware." - a [nice alternative to GDM's DiLoCo](https://x.com/apyh__/status/1828139739842850879).
- a snowball of Cursor AI love following [a viral video of an 8yr old using it](https://x.com/rickyrobinett/status/1825581674870055189) and their [fundraise announcement](https://x.com/cursor_ai/status/1826656532072923219). Their first podcast interview was [exactly a year ago](https://www.latent.space/p/cursor) and Aman [returned to co-host in June](https://www.latent.space/p/iclr-2024-benchmarks-agents).
- George Hotz's [tinybox is live for sale!](https://x.com/realGeorgeHotz/status/1828197925874463166). About

Since the newsflow is light, why not give Box feedback on [Box AI's new beta](https://shortclick.link/8dpebr)?

---

**[Sponsored by Box] You are building things with AI. So is Box. Imagine if you built your things using Box’s things. Actually, don’t imagine it, [try it yourself in the Box AI Developer Zone.](https://shortclick.link/8dpebr)**

> Swyx's comment: thanks to Box (via [Freeman & Forrest](https://x.com/editingemily/status/1790739130516676788)) for supporting AI News this August ([1](https://shortclick.link/23g92m), [2](https://shortclick.link/tndo68), [3](https://shortclick.link/5lxgsv))!

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

- **Humanoid Robots**: [@adcock_brett](https://twitter.com/adcock_brett/status/1827738485479899257) reported that China-based robotics startup AGIBOT revealed 5 new humanoid robots with open-source plans, each designed for different tasks from household chores to industrial operations. Additionally, [@adcock_brett](https://twitter.com/adcock_brett/status/1827738507885879382) mentioned that Unitree, another Chinese robot manufacturer, showcased its new G1 humanoid robot, reportedly nearing "mass production" at a price of $16,000.

- **AI-Generated Motion**: [@adcock_brett](https://twitter.com/adcock_brett/status/1827738530321207444) noted that ETH Zurich and Disney developed an AI system capable of generating physics-based movements for robots from text or image inputs, using a two-stage approach that learns latent representations of motion from large datasets.

- **Teleoperation System**: [@adcock_brett](https://twitter.com/adcock_brett/status/1827738552806805934) highlighted UC San Diego's release of ACE, a low-cost, cross-platform teleoperation system allowing researchers to control multiple robots with precision simultaneously. The system is fully open-sourced.

**AI Models and Tools**

- **Jamba 1.5**: [@adcock_brett](https://twitter.com/adcock_brett/status/1827738732469882945) reported that AI21 Labs unveiled Jamba 1.5, a new multilingual AI model family with a 256,000 context length, 2.5x faster long context in its size class, and permissive licensing for smaller organizations. The model has full open weights.

- **Dream Machine 1.5**: [@adcock_brett](https://twitter.com/adcock_brett/status/1827738620045779442) mentioned Luma Labs' release of Dream Machine 1.5, an upgrade to their AI video generation model, allowing for higher-quality text-to-video, more intelligent prompt understanding, and improved image-to-video capabilities.

- **Ideogram v2**: [@adcock_brett](https://twitter.com/adcock_brett/status/1827738665017045431) noted that Ideogram released v2 of its text-to-image AI model, distinguishing itself with the ability to generate near-perfect text, opening up new use cases for image generation like thumbnails, posters, and memes.

- **Mistral-NeMo-Minitron 8B**: [@adcock_brett](https://twitter.com/adcock_brett/status/1827738754888409144) reported that Nvidia and Mistral released Mistral-NeMo-Minitron 8B, a small model that can run on laptops and PCs, outperforming Mistral-7B and Meta-LLama 3.1-8B on the Open LLM leaderboard.

**AI Applications and Research**

- **Autonomous Sales Agents**: [@adcock_brett](https://twitter.com/adcock_brett/status/1827738597690126484) mentioned Salesforce's introduction of two fully autonomous, AI-powered sales agents, Einstein SDR Agent and Einstein Sales Coach Agent, capable of engaging with inbound leads and coaching salespeople in real-time.

- **Amazon's AI Assistant**: [@adcock_brett](https://twitter.com/adcock_brett/status/1827738709975863639) shared an update from Andy Jassy on Q, Amazon's AI assistant for software development, estimating it has saved the equivalent of 4,500 developer-years of work.

- **Neuralink Progress**: [@adcock_brett](https://twitter.com/adcock_brett/status/1827738642506256600) reported on Neuralink's progress with their second human patient, Alex, who demonstrated impressive control in playing Counter-Strike 2 using just the brain-computer interface and broke the previous world record for BCI cursor control on day one.

**AI Development and Tools**

- **Git Commit Message Generator**: [@karpathy](https://twitter.com/karpathy/status/1827810695658029262) shared a utility that auto-generates git commit messages based on the git diff of staged changes, using the `llm` CLI utility from @simonw.

- **Speculative Decoding for Code Edits**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1827832981056016645) highlighted Cursor.ai's blog post on modifying diff format and speculative edits with fine-tuned Llama 70B, achieving a 4-5x speed up over GPT4-o and pushing the pareto frontier on the accuracy/latency curve.

- **VoiceCraft**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1827812046206840932) mentioned an impressive tool for zero-shot speech editing and text-to-speech in the wild, capable of cloning unseen voices with only a few seconds of reference.

**AI Research and Frameworks**

- **GraphRAG**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1827835310257955195) discussed a survey paper on GraphRAG techniques, bridging graph-structured data and language models to capture complex relational knowledge more effectively than text-based methods.

- **iLoRA**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1827846782346219764) highlighted a paper proposing Instance-wise LoRA (iLoRA), which personalizes LLM recommendations by integrating LoRA with Mixture of Experts for improved accuracy in sequential recommendation systems.

- **RAGLAB**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1827843916034404430) mentioned RAGLAB, an open-source library for standardizing RAG research, featuring a modular design for fair comparisons between algorithms.

**AI Ethics and Regulation**

- **California SB 1047**: [@labenz](https://twitter.com/labenz/status/1827744915775766739) commented on the SB 1047 bill, noting that few models would be covered (only those costing $100M+) and that developers are already voluntarily conducting extensive safety testing.

**Memes and Humor**

- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1827762376906842375) shared a humorous t-shirt caption related to AI.
- [@vikhyatk](https://twitter.com/vikhyatk/status/1827819090582597828) jokingly suggested turning off syntax highlighting to become a better developer.
- [@abacaj](https://twitter.com/abacaj/status/1827775671730418120) humorously commented on the prevalence of Cursor-related content in their feed.

This summary captures the key developments, research, and discussions in AI and robotics from the provided tweets, focusing on aspects relevant to AI engineers and developers.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Hardware Optimization for Local LLM Inference**

- **Is $2-3000 enough to build a local coding AI system?** ([Score: 55, Comments: 102](https://reddit.com//r/LocalLLaMA/comments/1f0xg89/is_23000_enough_to_build_a_local_coding_ai_system/)): A user inquires about building a **local coding AI system** with a budget of **$2,000 to $3,000**, aiming to replicate the performance of commercial coding assistants like Cursor and Anthropic. They prioritize **speed over accuracy**, suggesting that accuracy can be improved through better prompting or retries, and specifically ask if a **Mac Studio** would be sufficient for this purpose.

- **Consider not using a Mac...** ([Score: 178, Comments: 149](https://reddit.com//r/LocalLLaMA/comments/1f0w0bn/consider_not_using_a_mac/)): The post compares **LLM inference performance** between an **M2 Mac Studio** and an **AMD build with a 2080ti GPU**. The **Nvidia setup** significantly outperforms the Mac, processing **32k context** in **25 seconds** compared to the Mac's **260 seconds**, while using less VRAM (**10GB** vs **30GB**) and supporting **64k context** with **flash attention** and **quant k,v**. Additionally, the Nvidia rig demonstrates more stable performance with **context shifting** and **reply generation**.

**Theme 2. Advancements in Long-Context LLM Generation**

- **[LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs](https://github.com/THUDM/LongWriter)** ([Score: 74, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1f13o9n/longwriter_unleashing_10000_word_generation_from/)): **LongWriter** is a technique that enables **long-context Large Language Models (LLMs)** to generate coherent texts exceeding **10,000 words**. The method involves breaking down the generation process into manageable chunks, using **context windows** of up to **32,000 tokens**, and employing strategies like **recursive summarization** and **dynamic prompting** to maintain consistency across sections. This approach allows for the creation of extended narratives, comprehensive reports, and other long-form content while preserving thematic coherence and logical flow throughout the generated text.

**Theme 3. Anthropic's Controversial Stance on AI Regulation**

- **[Do you think Anthropic is worse than OAI with fighting open source? To me it seems like the case. This letter appears to imply they actually suggested the bill to Sen Wienner... I really like my OSS LLMs....](https://i.redd.it/ybjoof8z1xkd1.png)** ([Score: 226, Comments: 111](https://reddit.com//r/LocalLLaMA/comments/1f1d4gh/do_you_think_anthropic_is_worse_than_oai_with/)): Anthropic appears to be taking a more aggressive stance against open-source LLMs compared to OpenAI, potentially **suggesting legislation to Senator Wienner**. The post author expresses concern about this perceived stance, indicating a preference for **open-source language models**. This debate highlights the tension between **AI safety regulations** and **innovation in LLM development**, particularly in the open-source domain.
  - The proposed **California bill SB1047** requires **safety testing** and a built-in **"kill switch"** for large AI models. Critics argue this could stifle **innovation** and **open-source development**, potentially driving AI progress out of the US.
  - Users expressed concerns about **regulatory capture**, suggesting Anthropic may be pushing for legislation to maintain their market position. Some compared it to past attempts to regulate new technologies like **cars**, **planes**, and **video games**.
  - Discussion highlighted the challenges of implementing a "kill switch" in mathematical models and the potential for **innovation to move elsewhere**, particularly to countries like **China** that may be less inclined to regulate AI development.

**Theme 4. Emerging Chinese LLMs Challenging Western Models**

- **Impressed by GLM-9B (they say little about the model)** ([Score: 54, Comments: 12](https://reddit.com//r/LocalLLaMA/comments/1f184lk/impressed_by_glm9b_they_say_little_about_the_model/)): The post author expresses surprise at the performance of the **GLM4-9B model**, claiming it **far exceeds Gemma 2 9B and Llama 3.1 8B** in terms of answer quality. They share a link to the [model on Hugging Face](https://huggingface.co/THUDM/glm-4-9b-chat) and ask for others' opinions and experiences with the model, noting that there seems to be little discussion about it.

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**Robotics and AI Hardware**

- **Disney Research's facial mimicry robot**: A [robot developed by Disney Research](https://www.reddit.com/r/singularity/comments/1f0yr4r/this_robot_from_disney_research_can_imitate_human/) can **imitate human facial movements**, specifically blinking and subtle head movements.
- **Beijing World Robotics Conference 2024**: The [conference showcased various robotic technologies](https://www.reddit.com/r/singularity/comments/1f0zlw6/beijing_world_robotics_conference_2024/), highlighting advancements in the field.

**Biotechnology and Food Tech**

- **Lab-grown meat cost parity**: A [study suggests that lab-grown meat can cost the same as USDA organic chicken](https://www.reddit.com/r/singularity/comments/1f0x6pq/labgrown_meat_can_cost_the_same_as_usda_organic/), indicating progress in making cultured meat economically viable.

**AI Model Development**

- **Model size and intelligence trade-offs**: A [discussion on the compromise between model size and intelligence](https://www.reddit.com/r/singularity/comments/1f158p9/in_the_race_to_bottom_for_price_significant_model/) suggests that recent models are significantly distilled compared to earlier versions like GPT-4, potentially affecting their capabilities.
- **Perceived slowdown in AI progress**: Users in [r/OpenAI are discussing a perceived slowdown in AI advancements](https://www.reddit.com/r/OpenAI/comments/1f0v3pe/anyone_else_feel_like_ai_improvement_has_really/), noting that recent developments haven't been as impressive as those from a year ago.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Claude 3.5 Sonnet


**1. LLM Advancements and Benchmarking**

- **Grok-2 Climbs the LMSYS Leaderboard**: **xAI's Grok-2** has made a significant impact on the [LMSYS Leaderboard](https://lmsys.org/leaderboard), surpassing **GPT-4o** (May) and tying with the latest **Gemini** for the #2 spot with over 6,000 community votes.
   - **Grok-mini**, also from xAI, secured the #5 position, excelling particularly in **Math** (#1) and ranking #2 across **Hard Prompts**, **Coding**, and **Instruction-following** categories.
- **1.5-Pints LLM: Quality Over Quantity**: A new compact LLM called **"1.5-Pints"** was pretrained in just 9 days using a curated dataset of 57 billion tokens, outperforming both **Apple's OpenELM** and **Microsoft's Phi** on the **MT-Bench** benchmark.
   - The model utilizes a **modified Mistral tokenizer** and **Llama-2 architecture**, prioritizing "textbook-like" content for enhanced reasoning and logical deduction capabilities.
  


**2. LLM Optimization Techniques**

- **DisTrO: Revolutionary Distributed Optimization**: Nous Research released a preliminary report on **DisTrO**, a family of distributed optimizers that reduces inter-GPU communication requirements by **1000x to 10,000x** without relying on amortized analysis.
   - DisTrO matches **AdamW+All-Reduce** in convergence speed, potentially revolutionizing large-scale LLM training. The full report is available on [GitHub](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf).
- **LIGER Kernel Boosts LLM Training Efficiency**: The new **LIGER** kernel for LLM training has achieved impressive results, offering **25% VRAM** savings and **33% training time** reduction compared to traditional methods.
   - While primarily designed for multi-GPU setups, LIGER is expected to provide improvements even for single GPU training scenarios, sparking excitement in the AI community.
- **Sparse-Marlin Accelerates Matrix Multiplication**: **Sparse-Marlin**, a new GPU-optimized kernel, has been integrated into the **vllm_project**, achieving **5.3x speedups** on NVIDIA GPUs (Ampere/Ada) for matrix multiplication with 4-bit quantized weights.
   - This advancement maintains efficiency with batch sizes up to 32 and leverages **2:4 sparsity**, potentially revolutionizing inference speed for large language models.
  


**3. Open Source AI Developments**

- **Zed AI: The Open Source Coding Companion**: **Zed AI** has launched as an open-source AI-powered code editor, offering a powerful interface for AI-assisted programming with support for models like **Claude-3.5** and integration with **Ollama**.
   - The editor features a new Anthropic API designed for fast text transformations, available free for the first month, positioning itself as a strong alternative to proprietary options like Cursor.
- **Apple's ML-Superposition Prompting Goes Open Source**: Apple has released their **ML-Superposition Prompting** project as open source, now available on [GitHub](https://github.com/apple/ml-superposition-prompting), aiming to advance prompting techniques in machine learning.
   - This release has generated excitement in the AI community, potentially offering new tools and methodologies for researchers and developers working on language models and prompt engineering.
- **Tinybox: Open Hardware for AI Enthusiasts**: The **Tinybox**, an open hardware project associated with the **tinygrad** framework, has launched sales to the public through the [tiny shop](https://tinycorp.myshopify.com).
   - With a current production capacity of about 4 units per day and a backlog of 60 units, the Tinybox represents a growing interest in accessible, open-source hardware for AI development and research.
  


**4. AI Industry and Community Updates**

- **AI Engineer London Meetup Announced**: The first **AI Engineer London Meetup** is scheduled for September 12th, featuring speakers Maxime LaBonne, Rovio Sc, Martins Bruveris, and Chris Bull, as announced by [@dctanner](https://x.com/dctanner/status/1827071893448618453).
   - This event is inspired by @swyx's AI Engineer World's Fair, aiming to bring together AI enthusiasts and professionals in London for knowledge sharing and networking.
- **Together AI Adjusts Pricing Structure**: **Together AI** announced price increases for its Serverless Reference endpoints, with **Llama-3 8B** rising from $0.20 to $0.40 per million tokens, and **Llama-3 70B** from $0.90 to $1.80 per million tokens, effective September 1, 2024.
   - While these changes affect the Serverless Reference endpoints, Together AI's Turbo and Lite pricing remains unchanged, as reflected on their [pricing page](https://www.together.ai/pricing).

---

# PART 1: High level Discord summaries




## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DisTrO's Distributed Optimization Breakthrough**: Nous Research released a preliminary report on **DisTrO**, demonstrating a **1000x to 10,000x** reduction in inter-GPU communication without amortized analysis and matching **AdamW+All-Reduce** in convergence speed. The full report is available on [GitHub](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf).
   - This advancement in distributed optimizers marks a significant progress in LLM training, with the team expressing excitement for upcoming code and algorithm releases.
- **Hermes 2.5 Beats Hermes 2 in Performance**: After integrating [code instruction examples](https://link.to.examples), **Hermes 2.5** demonstrated superior performance over **Hermes 2**, achieving a score of **52.3** on the MMLU benchmark compared to Hermes 2's **34.5**.
   - This substantial improvement sets a new standard for LLM performance evaluations among engineers.
- **1.5-Pints LLM Achieves Quick Training Success**: The new **1.5-Pints** model was pretrained in just **9 days**, surpassing both **Apple's OpenELM** and **Microsoft's Phi** on **MT-Bench**, which emulates human judgments. This was done using a curated dataset of **57 billion tokens** focusing on logical deduction.
   - Utilizing a modified **Mistral tokenizer** and the **Llama-2 architecture**, this model exemplifies efficient training methodologies in the LLM domain.
- **Sparse-Marlin Accelerates Matrix Multiplication**: The introduction of **Sparse-Marlin** into **vllm_project** improves matrix multiplication speeds by achieving **5.3x** speedups on NVIDIA GPUs using **4-bit quantized weights**.
   - This GPU-optimized kernel is likely to enhance performance significantly for users working with large models.
- **Exploring Whisper Diarization Implementation**: A user inquired about implementing **Whisper diarization** and shared a script utilizing **Whisper v3**, seeking a method to identify speaker changes.
   - Current efforts involve amalgamating diarization capabilities to streamline audio processing and improve output fidelity.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Accuses LinkedIn of Code Theft**: Members of the Unsloth channel assert that **LinkedIn** has copied code from their project, particularly in their **Triton kernel implementation**. They indicated the use of [LinkedIn's Liger-Kernel repository](https://github.com/linkedin/Liger-Kernel) and a post on [Ollama](https://ollama.com/unclemusclez/qwen2-unsloth) as evidence.
   - The claims point out that LinkedIn benchmarks its kernels against Unsloth's work, implying a lack of fair contribution back to the original project.
- **Performance Comparison: Unsloth vs. Hugging Face**: Discussions highlighted that **Unsloth** outperforms platforms like **Hugging Face** in speed and memory efficiency, despite lacking support for **8-bit models**. This places Unsloth in a competitive position, yet with notable limitations.
   - Members expressed that while Unsloth demonstrates impressive training and inference times, full model support remains essential for broader adoption.
- **Liger Kernel Speeds Up LLM Training**: A member revealed that the new **Liger Kernel** could enhance LLM training speeds by **20%** while cutting memory usage by **60%**, as discussed in a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1eznkml/liger_kernel_one_line_to_make_llm_training_20/).
   - Utilizing **Triton**, this kernel shows promise for optimizing training times, attracting attention for its potential applications.
- **Challenges in Fine-Tuning Multilingual Models**: Members shared insights on training models in languages like **Arabic** and **Persian**, stressing the importance of specialized datasets and pretraining. One suggestion included leveraging **Persian Wikipedia** for better model results.
   - Concerns were raised regarding proper support for these languages in **Llama-3**, indicating a gap that may hinder advancement in multilingual capabilities.
- **Replete-LLM V2 Arrives with Enhanced Features**: **Replete-LLM-V2-Llama-3.1-8b** is launched, emphasizing improvements in reasoning and coding performance, trained on the **Replete-AI/The_Living_AI_Dataset** to embed concepts of **Love and Empathy**.
   - The effectiveness of this model heavily relies on its system prompts, crucial for optimizing its information processing capabilities.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Clarifying Stable Diffusion Online's Status**: Members questioned whether [Stable Diffusion Online](https://stabledifffusion.com) is an official site or if it operates independently from Stability AI.
   - This inquiry reveals ongoing confusion within the community regarding the credibility and linkage of various platforms related to Stable Diffusion.
- **ComfyUI vs. ForgeUI - Choose Your Tool!**: A suggestion arose that those not utilizing the full capabilities of **ComfyUI** should consider switching to **ForgeUI** for a streamlined experience.
   - This debate highlights the ongoing conversation about optimizing workflows for image diffusion setups.
- **Diving into SD Image Upscaling Approaches**: Members discussed various techniques for image upscaling, including **Ultimate SD Upscale** and **Tiled Diffusion**, particularly noting the '4x-NomosWebPhoto-atd' model combined with SUPIR.
   - These discussions emphasize the community's efforts to enhance image quality through advanced methods.
- **Noise Injection: The Secret Sauce for Image Quality**: A member elaborated on 'Noise Injection' in A1111/Forge, explaining its role in improving image upscaling efforts.
   - This technique garners attention as a potential enhancement tactic, leading to higher quality outputs.
- **Struggles with Flux - Overfitting Issues**: Discussion focused on **Flux's** challenges with overfitting, particularly in fantasy-related outputs leading to less diversity in generated images.
   - This exploration raised concerns about how **Flux** needs adjustments to balance creativity with variability.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hermes 2.5 outperforms Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** appears to perform better than **Hermes 2** in various benchmarks.
   - Hermes 2 scored a **34.5** on the MMLU benchmark whereas Hermes 2.5 scored **52.3**.
- **Mistral struggles expanding beyond 8k**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed 'cursed model merging'.
- **Model Quantization and Distillation Essentials**: The importance of **Model Quantization** and **Model Distillation** for productionizing machine learning models was highlighted.
   - Members agreed these techniques are fundamental for effective deployment beyond local training.
- **TinyLlama's Quick Success**: **TinyLlama**, a model similar to Tau LLM, was successfully trained in just 9 days and has outperformed both Apple’s OpenELM and Microsoft’s Phi on MTBench.
   - Training code and model weights were made publicly available on GitHub and HuggingFace.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Model Scaling Hits Diminishing Returns**: Discussions highlight diminishing returns on **model scaling**, especially with **Llama 3.1** and **Claude 3.5 Sonnet**, where performance improvements lag behind increased compute power.
   - Participants stress the necessity of innovative breakthroughs to scale AI beyond mere data and computational increases.
- **Debating AI Consciousness**: Philosophical discussions revolve around whether current **LLMs** like GPT can be considered conscious, considering they lack organic experience and may follow different laws than human consciousness.
   - Participants also examined implications on free will, suggesting AI systems exhibit decision-making based on internal logic rather than true volition.
- **Sharing GPTs Effectively**: Members expressed interest in better tracking shared **GPTs** and their utility within the community, questioning how to assess their effectiveness.
   - The conversation included usability concerns regarding shared output features and possible improvements for tracking use-cases.
- **Create Custom GPTs with Brand Identity**: A suggestion arose to leverage the **custom GPT builder** to craft GPTs that align with specific brand identities for content creation, using the GPT store for system prompts.
   - The emphasis was on enhancing brand consistency through custom prompts in API integrations.
- **Subscription Models for OpenAI API**: Users explored how platforms manage **subscription models** for OpenAI's API, like monthly plans that utilize token-based pricing.
   - Chatbase was cited as an example under discussion, indicating a pressing need for clarity on implementation strategies.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Creator Community Launch**: Perplexity AI partnered with [Kale](https://www.kalecard.com/t/perplexity.ai?ref=perplexity.ai_discord) to launch the **Perplexity Creator Community**, allowing creators to earn cash for engaging video content.
   - This initiative encourages users to post on their own schedule while generating income based on their videos' reach.
- **API Rate Limits Cause Frustration**: Maged Helmy from Newcode.ai urgently requested increased API rate limits for their integration after waiting six months without a response from the Perplexity team.
   - With over 3,500 users, Newcode.ai's operation depends on these enhanced limits to maintain performance.
- **GPT-4o Dominates Coding, Claude 3.5 Sonnet for Knowledge**: Discussions highlighted **GPT-4o** as superior for STEM tasks while **Claude 3.5 Sonnet** excels in knowledge retrieval, particularly for coding-related queries.
   - Users noted Claude struggles with poetry and narratives, making GPT-4o a go-to option for a broader array of tasks.
- **Image Generation Troubles in Perplexity**: Users reported significant challenges with image generation, particularly using Dalle3, where attempts led to thread failures.
   - Feedback indicated that the image generation process might need refinement, as some results did not meet user expectations.
- **Perplexity Pro's LinkedIn Subscription Offer**: Perplexity AI is providing a free year of **Perplexity Pro** to LinkedIn Premium subscribers, though some users in the EU faced issues with availability.
   - The Pro version grants unlimited searches and access to advanced AI models like **GPT-4 Omni** and **Claude 3.5 Sonnet**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Grok-2 and Grok-mini hit Leaderboard!**: **xAI's Grok-2** and **Grok-mini** have surged onto the [LMSYS Leaderboard](https://lmsys.org/leaderboard) with over **6000** community votes! Notably, Grok-2 ties with **Gemini** for #2, while Grok-mini excels in **Math** (#1) and ranks #2 across **Hard Prompts**, **Coding**, and **Instruction-following**.
   - Members cheered as Grok-2 bested **GPT-4o** (May), indicating a potential shift in the leaderboard dynamics and user preferences.
- **Database Outage Resolved**: A recent **database change** led to a approximately **2 minute outage**, but the issue has since been resolved and service is back to normal.
   - The team apologized for the inconvenience caused, emphasizing the need for reliable uptime.
- **Mistral can't scale beyond 8k**: Concerns arose around **Mistral**, as it reportedly cannot extend past **8k** without continued pretraining, highlighted as a [known issue](https://link.to.issue).
   - Suggestions included exploring **mergekit** and **frankenMoE finetuning** techniques for improved performance.
- **Claude 3.5 Sonnet goes dark again**: Users reported that **Claude 3.5 Sonnet** is facing intermittent outages, affecting its availability significantly.
   - While **Haiku** is functional, issues persist across other models like **Hermes 3.5**, hinting at broader system instabilities.
- **OpenRouter API Key query**: Users are discussing how to integrate their own **API keys** with **OpenRouter**, and whether the displayed token pricing reflects total costs including the **OpenRouter fee**.
   - Clarifications indicated that token prices are listed in **OpenRouter credits**, and the applicable fees are calculated upon adding credits.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **OMI Model Competence Discussion**: Members discussed the ability of **OMI participants** to create AI models from scratch, but fell short on sharing concrete opinions or assessments.
   - *No solid conclusions were reached, leaving participants pondering the competencies at play.*
- **LLM Repetition Failure Mode**: A common failure mode in LLMs where they repeat phrases was discussed, possibly linked to model over-quantization and minimizing loss.
   - Participants hypothesized that certain conditions might be triggering this **looping behavior**, highlighting a need for further investigation.
- **Anthropic's Interpretability Cost Challenge**: Questions arose about the **cost** of replicating [Anthropic's interpretability work](https://arxiv.org/abs/2406.04093) for models like Llama 8B or Mistral, which are data-hungry and compute-intensive.
   - Members noted the high costs without providing specific figures, emphasizing the importance of resource allocation in these projects.
- **Sparse MoE's GPU Utilization Benefits**: A member raised how **Sparse MoE** utilizes GPU sparsity for efficient distributed training, allowing experts to be spread across multiple processes.
   - This strategy could enhance performance in distributed inference contexts, highlighting scalability approaches.
- **GNNs and Evolutionary Learning Approaches**: One member compared the evolution of **GNNs** to positional embeddings, suggesting future advancements may involve inferring embeddings from latent representations.
   - This perspective hints at new pathways toward improving representation learning in graph structures.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Hermes 2.5 outperforms Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** shows superior performance over **Hermes 2** in benchmarks, scoring **52.3** versus Hermes 2’s **34.5** on the MMLU.
   - This improvement highlights the effectiveness of recent optimizations in newer model iterations.
- **Mistral struggles with 8k limitations**: **Mistral** cannot extend beyond an 8k context length without ongoing pretraining, recognized as a significant limitation in its current setup, and [this is a known issue](https://link.to.issue).
   - There's ongoing dialogue about exploring solutions like *mergekit* and *frankenMoE finetuning* to push these boundaries.
- **Unpacking BERTopic Utility**: Discussion surfaced about **BERTopic**, a robust tool for topic modeling, with members sharing their project on [visualizing data](https://github.com/YoungPhlo/visualizing-datasets-ai-engineer-fall-2023).
   - The conversation reaffirmed its end-to-end capabilities for generating interpretable topics, stimulating curiosity about its clustering efficacy.
- **Call for Collaboration on Open Empathic Project**: A plea for broadening the categories for the **Open Empathic** project was made, emphasizing the need for contributions from the community.
   - Members were pointed to a [YouTube tutorial](https://youtu.be/GZqYr8_Q7DE) for guidance on how to add their favorite scenes, alongside a link to the [OpenEmpathic project](https://dct.openempathic.ai/).
- **AI Engineer Meetup Launch in London**: Newly announced AI Engineer Meetup set for September 12th in London, inspired by the AI Engineer World's Fair with four notable speakers confirmed.
   - Interested attendees are encouraged to register [here](https://x.com/dctanner/status/1827071893448618453?s=46) for what promises to be a highly engaging gathering.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox Sales Launch!**: The **Tinybox** factory is now at full power, with sales set to open shortly to the public. Interested buyers can check out [the tiny shop](https://tinycorp.myshopify.com) for purchase options.
   - The **Tinybox** is currently sold out with a production capacity of about **4 units per day**, creating a backlog of **60** more units.
- **Concerns About E-graph Performance**: Members expressed that **e-graph rewrites** lag behind current SAT solvers when tackling large search spaces, highlighting potential performance bottlenecks.
   - Continuous improvement is suggested to match the efficiency seen in established SAT solving techniques.
- **Exploring Tinygrad and AMD GPUs**: Discussion emerged about using **AMD GPUs** with **Tinybox**, noting AMD's recent acquisition of **Silo AI** and their advancements in training LLMs on AMD hardware.
   - Community members weighed in, contemplating the feasibility and advantages of integrating AMD's capabilities effectively.
- **Tinygrad vs Torch in BERT Pre-Training**: A user showed interest in collaborating with **Tinygrad** to pre-train a large **BERT** model, offering computing resources for the task.
   - This collaboration could pave the way for exploring the performance differences between Tinygrad and PyTorch for large model training.
- **Improving Training Speed**: A user reported a **25% increase** in training speed (GFLOPS) after tweaking preprocessing by removing the `.cast(dtypes.default_float)` call in the **beautiful_cifar** example.
   - With this adjustment, they noted that the model now processes data as `dtype.float`, enhancing efficiency.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R Model Update Lacks Announcement**: A new **Command-R** model has been released, but there’s been no official communication about its features, including pricing and context window.
   - *Users demand clarity* as many are eager to know about fine-tuning options and addressing unanswered questions.
- **Durov's Bold Citizenship Move**: Pavel Durov, the **Telegram** founder, recently secured French citizenship and is now facing a trial in France, stirring debate.
   - Some speculate he aims for *strategic prison time* to gain international media attention amid tensions with NATO.
- **Cohere Offers Free Trial for Chatbots**: A user explored using **Cohere**'s free trial for building a **Rasa** chatbot, hoping for a cost-free alternative to OpenAI’s services.
   - The response indicated interest in affordable options as users navigate the costs associated with AI deployments.
- **Cohere API Rate Limits Tightened**: New reports show users hitting 'too many requests' errors even at the documented rate, as limits have shifted to 1,000 calls per minute across all API keys.
   - Cohere clarified this means a holistic **1,000/minute limit** per user organization, impacting those using multiple keys concurrently.
- **Clarification on Rerank 3 Pricing**: Users inquired about **Rerank 3** pricing, specifically whether $2 for 1,000 searches covers true API calls.
   - Cohere confirmed that each search processes up to 100 documents and totals **409,600,000 tokens** for 1,000 searches based on document limits.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Create Llama introduces extraction template**: The [Create Llama](https://t.co/G95ReAuRS6) tool now features a structured extraction template, enhancing user experience.
   - This addition aims to streamline data extraction processes while maintaining accuracy and efficiency.
- **GraphRAG tutorial series kicks off**: A new step-by-step tutorial series on building [GraphRAG](https://t.co/7fLocjRvdN) has begun, focusing on core component implementation.
   - The first video emphasizes how to extract entities and relationships with LLMs using an in-memory implementation.
- **Data silos hinder enterprise LLM development**: Challenges persist with data silos in enterprise LLM development, underscoring the need for seamless authentication management.
   - LlamaIndex is investigating viable solutions to consolidate scattered knowledge across teams.
- **LLMs automate newsletter creation**: The LlamaIndex newsletter has transitioned to using LLMs for automating content creation, previously a manual, time-intensive task.
   - This shift exemplifies the capability of LLMs in enhancing efficiency for regular content summarization.
- **RAG-a-thon hackathon on the horizon**: The second [RAG-a-thon](https://t.co/IFvyW5QB6r) hackathon, in partnership with Pinecone, is set for October 11-13 in Palo Alto, offering over $7k in cash prizes.
   - It will be hosted at the 500 Global VC offices, welcoming participants to showcase innovative solutions.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Compiled Function Outputs Differ from Eager Mode**: A member raised a question about why a compiled function might produce different outputs compared to its non-compiled version, with the same seed. This is attributed to differing RNG usage: **Triton's RNG** in compiled code versus PyTorch's in eager mode, potentially influenced by in-place operation behavior.
   - In-place operations, like `scatter_`, may yield unexpected results in compiled code, leading to higher memory consumption and varying output.
- **Cudagraphs Might Consume More Memory**: The utilization of **cudagraphs** for debugging was discussed, indicating their potential to pre-allocate buffers. However, they can also lead to increased memory usage, which may not be desirable.
   - This signifies a trade-off in using cudagraphs, as the benefits need to be weighed against their memory overhead.
- **FP16 as a Memory-Saving Strategy**: Switching to **FP16** for inference instead of FP32 was suggested to lower memory usage, especially on hardware that doesn't support BF16. This altered approach reportedly alleviated out-of-memory issues.
   - Despite these improvements, discrepancies between compiled and non-compiled outputs remained a concern.
- **Exploring Numerical Differences in Compiled Kernels**: The remaining output variance might arise from numerical differences inherent in the compiled kernels, even with optimized memory usage. This points to potential computational variations despite identical inputs.
   - Participants expressed concern over these numerical discrepancies, highlighting an area for further consideration in compiled code evaluation.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Document Loading: Image Extraction Simplified**: The `extract_images=True` parameter in `PyPDFLoader` from LangChain's community package allows seamless image extraction from PDF documents, enriching text context for LLM processing.
   - This is particularly useful for applications requiring image analysis in conjunction with text data, expanding the functional capabilities of LangChain.
- **LLMChain vs LCEL: Flexibility vs Optimization**: `LLMChain` provides a straightforward approach to chaining models and prompts, whereas `LCEL` offers greater customization and flexibility for complex tasks.
   - While `LLMChain` remains the optimal choice for most scenarios, enthusiasts of modular design may prefer the intricate control that LCEL introduces.
- **Troubleshooting PostgresSaver Errors**: Users are encountering a `TypeError` related to tuple indexing while leveraging `PostgresSaver` with LangGraph, indicating potential issues in data type handling.
   - Further investigation is required to clarify tuple access methods and resolve this ongoing challenge experienced by developers.
- **GenAI's Growing Role in Data Science**: A discussion highlighted the emerging role of Generative AI in the data science landscape, particularly in automating code generation and data pipeline setup.
   - Despite skepticism regarding its limits, participants acknowledged the critical integration between data science and GenAI advancements.
- **RAG Collaboration: Seeking Partners**: A member shared their intent to develop a Retrieval-Augmented Generation (RAG) chatbot using LangChain, hoping to find collaborators for the project.
   - Challenges with scraping and RAG components were noted, underscoring the collaborative opportunities in this technical space.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **GPT-4 Fine-tuning vs Mistral: Mixed Reviews**: A user claimed that fine-tuning **GPT-4** is 'kind of shit' in comparison to **Mistral**, even though they utilized less data for training.
   - This sparked a discussion about the relative performance of both models in practical applications.
- **lm-eval-harness: Benchmarking Made Easy**: Members discussed the *lm-eval-harness* framework, suggesting it simplifies the creation of benchmarks by offering easy task integration.
   - One user emphasized their research on generating benchmark questions, shared in their recent paper on [MCQs](https://arxiv.org/abs/2406.02394) for LLM evaluation.
- **LIGER Shows Impressive Training Efficiency**: **LIGER** kernel promises **25% VRAM** and **33% training time** savings for **LLM training**, exciting users who are eager to test its capabilities.
   - However, there are doubts about its effectiveness for **single GPU training**, as noted by one user.
- **Curious About Phi-3-medium-128k-instruct Training Config**: A user sought the training configuration for the **Phi-3-medium-128k-instruct** model, emphasizing the need for shared setups.
   - Another user questioned the token training in a specific config setup (modules_to_save) and referenced an external message for clarity.
- **Exploring Data Curation Techniques**: A user probed into **data curation**, inquiring if it involves models providing ratings like the **LLM-Judge system**.
   - The conversation indicated an interest in methods employing model assessments for curating data, akin to existing systems.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Jitting Behavior Explained**: When running `mojo main.mojo` in script mode, **jitting** occurs, which is why **global variables** behave differently than in the `mojo build main.mojo` compile mode.
   - This clarification helps users understand the complications of memory management when switching modes.
- **Community Ponders Development Pace**: Concerns arose over a perceived *slowdown* in blog posts and updates for both **Max and Mojo**, possibly due to summer vacations or accumulating issues.
   - Members seek clarification on whether this impacts future releases and projects.
- **GPU Support Takes Center Stage**: There's a strong push for **GPU support** in Mojo with expectations that future releases could address this, potentially moving **Magic** out of alpha.
   - Members are eagerly awaiting the next major release, aligning their community discussions with progress on these capabilities.
- **Modverse 42 Release Schedule Clarified**: Members questioned the absence of **Modverse 42** release last week, learning that releases occur every **1-3 weeks**, depending on project volumes.
   - The ongoing weekly tag might be adjusted as content flow stabilizes.
- **Mojo's Struct Parameters and UnsafePointer Details**: Issues arose with struct usage in Mojo causing errors due to **variadic parameters** not being parameterized correctly outside their defining structure.
   - A discussion on using **UnsafePointer** highlighted how ownership needs to be explicitly managed, underscoring the complexities of reference management in Mojo.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Custom Paths for OpenInterpreter Profiles?**: A member queried about setting a *custom path for OpenInterpreter profiles*, but the developer stated this functionality isn't available yet, though it may come in the future.
   - This feature could enhance user flexibility once implemented.
- **OpenInterpreter --vision Flag Functionality on Windows**: Inquiries about the `--vision` flag on Windows concluded that it should function correctly, with an encouragement to report any issues in a dedicated channel.
   - Further testing might yield vital insights into its compatibility across setups.
- **Prebuilt OpenInterpreter Demand Surges**: The developer shared that preorders for *prebuilt OpenInterpreter* units are closed due to high demand, indicating strong interest.
   - Users will need to wait until sales resume, highlighting the technical community's engagement with the product.
- **Brand Guidelines Still Missing**: A request for a brand guideline document surfaced, but members confirmed that no such document is available yet.
   - The inquiry tied into discussions around project accessibility and design considerations.
- **Zed AI: The Open Source Coding Companion**: Zed AI offers a *cool interface* for coding with AI assistance, supporting models like **Claude-3.5** and Ollama, enhanced by a new *Anthropic API* free for the first month.
   - It's gaining attention as a strong alternative to proprietary options like Cursor, fostering greater open-source development.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Apple's Superposition Prompting Project Launches**: Members expressed excitement over Apple's new **ML-Superposition Prompting** project, now live on [GitHub](https://github.com/apple/ml-superposition-prompting), aimed at refining prompting techniques in ML.
   - Currently, the community discussion centers around the initial reception of the project without further technical insights.
- **OpenAI Introduces Typed Outputs**: Discussion sparked about OpenAI's new feature for **typed outputs**, focusing on validation for structured outputs in JSON format with references to projects like **Outlines**, **Guardrails**, and others.
   - Members linked to relevant GitHub repositories, indicating various libraries available for managing structured output formats.
- **Handling DSPy Output Errors**: A member reported a `ValueError` in DSPy regarding 'Too many retries trying to get the correct output format' while using typed predictors, attributed to output filler text.
   - Another user provided insight and linked to an existing [GitHub issue](https://github.com/stanfordnlp/dspy/issues/1001) to clarify this common problem with JSON output parsing.
- **Exploring ColBERT Training for German**: A user seeks guidance on structuring training data for a ColBERT model in German, proposing a 32-way triplets format like that of ColBERTv2.
   - Their suggested format for data structuring includes `raw_query = [(query, (positive_passage, positive_score), [(negative_passage1, negative_score1), ...])]`, and they are looking for validation on its suitability.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Hugging Face Leaderboard syncs with Website**: The [Hugging Face Leaderboard](https://huggingface.co/spaces/bigscience/bfcl) now mirrors the website leaderboard due to a recent pull request, prompting a request for feedback from team members.
   - *Anyone concerned about this change is encouraged to share suggestions.*
- **BFCL V2-Live Dataset Accuracy in Focus**: There’s an ongoing discussion about how to calculate the overall accuracy for the [BFCL V2-Live dataset](https://github.com/ShishirPatil/gorilla/discussions/602), noting it contains **2,251 question-function-answer pairs**.
   - The dataset includes **258 simple, 7 multiple, 16 chained, and 14 multi-stage function calls**, raising questions about accurate assessment methods.
- **Inquiries about Adding Models to BFCL**: A new member expressed interest in adding a model to BFCL, asking about the process for non-open-source uploads and model evaluations with multiple components.
   - *Details on maintaining model integrity while integrating with BFCL are sought.*
- **Gorilla Leaderboard Explained**: A query arose regarding the phrase "prepare the executable test pairs" in the [Gorilla Leaderboard documentation](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing).
   - The documentation clarifies that users are encouraged to contribute executable test pairs to the leaderboard, fostering collaborative improvement of evaluation methods.
- **Training LLMs for Function Calls**: The [Gorilla Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing) serves to train and evaluate LLMs for function calls using a standardized benchmark.
   - This framework allows for comparison across various models, enhancing performance evaluations.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Anthropic's Mechanistic Interpretability Costs**: A user questioned the expenses related to running **Anthropic's mechanistic interpretability** for models like **Llama 8b** and **Mistral**, noting the absence of open-source alternatives.
   - They highlighted concerns regarding whether the limitations are due to being data-intensive or **compute-heavy**, alongside seeking clarity on other contributing factors.
- **Upcoming AI Engineer London Meetup**: Mark your calendars for the **AI Engineer London Meetup** on **September 12th**, showcasing insights from figures like Maxime LaBonne and Rovio Sc.
   - Details shared in a [tweet by Damien C. Tanner](https://x.com/dctanner/status/1827071893448618453) reveal that this event aims to bring part of Swyx's **AI Engineer World's Fair** to the UK.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Romain Huet Takes Over OpenAI DevRel**: The new head of developer relations at **OpenAI** is **Romain Huet**, who confirmed his role on [Twitter](https://x.com/romainhuet) after joining in **July 2023**.
   - Huet's appointment comes after previous lead **Logan's** departure, suggesting a focused leadership transition in OpenAI's developer outreach.
- **Logan's Smooth Transition**: **Logan** left **OpenAI** in **July 2023**, with confirmation from his successor, **Romain Huet**.
   - Huet noted that the transition was smooth, indicating established protocols for leadership changes within the organization.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **AI Engineer London Meetup Kicks Off**: The inaugural **AI Engineer London Meetup** is set for the evening of **12 September**, featuring four speakers: **Maxime La Bonne**, **Roviosc**, **Martins Bruveris**, and **Chris Bull**. Registration details can be found [here](https://x.com/dctanner/status/1827071893448618453).
   - This event aims to be a segment of the **AI Engineer World's Fair**, hosted by **Damien C. Tanner**, highlighting vibrant discussions among AI engineers.
- **Highlight on AI Engineer World's Fair Influence**: This London Meetup draws inspiration from the **AI Engineer World's Fair**, with the goal of creating a collaborative atmosphere for AI discussions. The event brings together an exciting lineup of speakers to share insights and experiences.
   - Hosted by **Damien C. Tanner**, the meetup serves as a communal space for AI enthusiasts to network and engage with cutting-edge topics in the field.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Hamel's Attendance Question**: A user inquired if **Hamel** was available during a discussion on **LLM Finetuning**, indicating interest in his expertise.
   - This interaction highlights the community's anticipation for insights from known contributors in LLM optimization.
- **Hamel is not present**: Unfortunately, **Hamel** was not available at the time of the inquiry, suggesting a missed opportunity for discussion.
   - Community members expressed hope that he would engage in future sessions to share his insights.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **CUDA Hackathon Hits San Francisco**: Get ready for the **CUDA Hackathon** in San Francisco on **September 21st**, where you can hack alongside **NVIDIA engineers** and tackle real-world CUDA challenges.
   - This is a golden opportunity to engage with experts and work on innovative **accelerated computing** projects.
- **Deep Dive into Accelerated Computing**: The event will explore **accelerated computing**, using NVIDIA's parallel computing platform to optimize GPU applications.
   - Participants will have hands-on access to NVIDIA resources and engineers for guidance in building and refining CUDA applications.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Together AI Hits Users with Price Increase**: Effective September 1, 2024, the pricing for **Together API**'s Serverless Reference endpoints will rise for the **Llama-3 8B** and **70B** models, with the 8B model increasing from **$0.20** to **$0.40** per million tokens.
   - The **70B** model will see a jump from **$0.90** to **$1.80** per million tokens, reflecting a significant upward adjustment.
- **Turbo and Lite Pricing Stays Steady**: While serverless endpoints are increasing, **Together API**'s **Turbo** and **Lite** pricing remains intact, as confirmed on the [Together Pricing Page](https://www.together.ai/pricing), last updated July 18, 2024.
   - This keeps users from facing price hikes on these endpoints amid overall pricing changes.
- **OpenAI Drops Prices, Leaves Together AI Looking Weird**: In contrast to **Together AI**'s upcoming price increases, a member noted that **OpenAI** has recently dropped the price for **GPT-4O-Mini**, stirring discussions about pricing strategies.
   - This shift raises eyebrows about Together AI's decision to hike prices while competitors decrease theirs.
- **Funding Woes Spark Price Increases Speculation**: Speculations arose that Together AI may be doubling their prices due to funding issues, as members discussed the sustainability of current pricing strategies.
   - They mentioned that pricing for **4-bit** and **8-bit** models should remain unchanged for now, but potential changes lurk in the future.



---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1277685656788668416)** (1 messages): 

> - `DisTrO`
> - `Distributed Optimization`
> - `LLM Training`
> - `Inter-GPU Communication`
> - `AdamW` 


- **Nous Research Releases DisTrO Report**: Nous Research released a preliminary report on **DisTrO**, a family of distributed optimizers that reduces the inter-GPU communication requirements by **1000x to 10,000x** without relying on amortized analysis and matches **AdamW+All-Reduce** in convergence speed.
   - The report is available on GitHub: [DisTrO/A_Preliminary_Report_on_DisTrO.pdf](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf).
- **DisTrO: A Game-Changer for LLM Training**: This is a major development in the quest to improve training of LLMs, as DisTrO offers a significant reduction in inter-GPU communication requirements.
   - The team is excited to share these early results and plans to release the code, full algorithm, and paper in the near future.
- **Open Collaboration on DisTrO**: This project was made possible by the hard work of researchers and engineers at Nous Research, and they are inviting collaboration on this project.
   - Anyone interested in contributing can reach out to recruiting@nousresearch.com.



**Link mentioned**: <a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>: Distributed Training Over-The-Internet. Contribute to NousResearch/DisTrO development by creating an account on GitHub.

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1276617216464519189)** (786 messages🔥🔥🔥): 

> - `Hermes 2`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic`
> - `GPTs Agents` 


- **Hermes 2.5 outperforms Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** appears to perform better than **Hermes 2** in various benchmarks.
   - Hermes 2 scored a **34.5** on the MMLU benchmark whereas Hermes 2.5 scored **52.3**.
- **Mistral has struggles expanding beyond 8k**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed "cursed model merging".
- **Open Empathic Project Plea for Assistance**: A member appealed for help in expanding the categories of the **Open Empathic** project, particularly at the lower end.
   - They shared a [YouTube video on the Open Empathic Launch & Tutorial](https://youtu.be/GZqYr8_Q7DE) that guides users to contribute their preferred movie scenes from YouTube videos, as well as a link to the [OpenEmpathic project itself](https://dct.openempathic.ai/).
- **GPTs Agents cannot learn after initial training**: A member shared a concern about GPTs agents not learning from additional information provided after their initial training.
   - Another member cleared this misunderstanding, explaining that [uploaded files are saved as "knowledge" files](https://link.to/openai-docs) for the agent to reference when required, but **they do not continually modify the agent's base knowledge**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://n8python.github.io/VoidInterface/.">The Void</a>: no description found</li><li><a href="https://x.com/NousResearch/status/1828121648383566270">Tweet from Nous Research (@NousResearch)</a>: What if you could use all the computing power in the world to train a shared, open source AI model?  Preliminary report: https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://x.com/nearcyan/status/1827074097790194123">Tweet from near (@nearcyan)</a>: anyone need 2,000 H100s w/ IB? have to Dec 14th, priced well. please DM!</li><li><a href="https://tenor.com/view/big-if-true-big-if-true-gif-18577099">Big If GIF - Big If True - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://arxiv.org/abs/2106.11257">Secure Distributed Training at Scale</a>: Many areas of deep learning benefit from using increasingly larger neural networks trained on public data, as is the case for pre-trained models for NLP and computer vision. Training such models requi...</li><li><a href="https://youtube.com/playlist?list=PLAJnaovHtaFQFUX5kp3d1UYmaUH_Ux8OL&si=loJ9miYPiKiXuT1M">AGI-24</a>: The foremost conference dedicated entirely to the latest AGI Research. There is a growing recognition, in the AI field and beyond that the threshold of achie...</li><li><a href="https://spectrum.ieee.org/eleutherai-openai-not-open-enough">EleutherAI: When OpenAI Isn’t Open Enough</a>: EleutherAI is a loose association of computer scientists whose latest effort is GPT-NeoX-20B, a 20 billion parameter, pretrained language model. If you don’t know what that is, think of OpenAI’s GPT-3...</li><li><a href="https://tenor.com/view/donald-trump-yuge-huge-gif-14073397">Donald Trump Yuge GIF - Donald Trump Yuge Huge - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.google.com/document/u/0/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/mobilebasic">SHARED Continuous Finetuning By Rombodawg</a>: no description found</li><li><a href="https://tenor.com/view/skeleton-eyebrows-achmed-puppet-jeffdunham-gif-11732886">Skeleton Eyebrows GIF - Skeleton Eyebrows Achmed - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/that-thing-is-so-big-nufo-that-thing-is-massive-that-thing-is-huge-ginormous-gif-25538935">That Thing Is So Big Nufo GIF - That Thing Is So Big Nufo That Thing Is Massive - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/hud_zah/status/1827057785995141558">Tweet from HudZah ⁂ (@hud_zah)</a>: in a couple weeks, i built a nuclear fusor in my bedroom – with zero hardware experience  the secret? Claude sonnet 3.5 + projects  a glimpse into the process below</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f0o0m0/how_to_use_any_ai_on_huggingface_on_your_phone/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/blog/collaborative-training">Deep Learning over the Internet: Training Language Models Collaboratively</a>: no description found</li><li><a href="https://training-transformers-together.github.io/">Train vast neural networks together</a>: A NeurIPS'21 demonstration that explains how to train large models together with multiple collaborators.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ew7kwu/llama31storm8b_has_arrived_a_new_8b_parameter_llm/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/blog/">Hugging Face – Blog</a>: no description found</li><li><a href="https://arxiv.org/abs/2306.17453">Pollen: High-throughput Federated Learning Simulation via Resource-Aware Client Placement</a>: Federated Learning (FL) is a privacy-focused machine learning paradigm that collaboratively trains models directly on edge devices. Simulation plays an essential role in FL adoption, helping develop n...</li><li><a href="https://youtu.be/QwNoFOUiSiE?si=zd7YiLpNplLqr5sJ">Microsoft Excel World Championship 2023 Finals Highlights</a>: Main highlights of the 3-hour Microsoft Excel World Championship 2023 Finals livestream. Watch the full livestream here: https://www.youtube.com/live/UDGdPE_...</li><li><a href="https://github.com/bigscience-workshop/petals">GitHub - bigscience-workshop/petals: 🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading</a>: 🌸 Run LLMs at home, BitTorrent-style. Fine-tuning and inference up to 10x faster than offloading - bigscience-workshop/petals</li><li><a href="https://youtu.be/7ZXPWTdThAA">Nous Research</a>: no description found</li><li><a href="https://zenodo.org/records/13370693?token=">A New Frontier: Unveiling the Unified Dimensions of Reality</a>: A New Dawn: Understanding the Universe Beyond Conventional Physics In the vast expanse of knowledge, humanity has long relied on the familiar anchors of classical physics to make sense of the cosmos. ...</li><li><a href="https://github.com/arcee-ai/mergekit">GitHub - arcee-ai/mergekit: Tools for merging pretrained large language models.</a>: Tools for merging pretrained large language models. - arcee-ai/mergekit</li><li><a href="https://zenodo.org/records/13370693?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImM0ZmZhNzNjLWEwOGQtNGRjMy05N2I4LWNmZWNlZGRhNDc1NiIsImRhdGEiOnt9LCJyYW5kb20iOiJlZTkzODc0NDBhMjkyNTMxYzMxZGRhYThjYTJhMGQ5ZSJ9.L2MqhoD1xKMeTC9zmIHHmFBoLj3G3jpY8REYIufs8vzdOwSZuUantHcdtiyP9tV-d-MS0XAYveIbrUvqrgJx_A">A New Frontier: Unveiling the Unified Dimensions of Reality</a>: A New Dawn: Understanding the Universe Beyond Conventional Physics In the vast expanse of knowledge, humanity has long relied on the familiar anchors of classical physics to make sense of the cosmos. ...</li><li><a href="https://huggingface.co/Replete-AI/Replete-LLM-V2-Llama-3.1-8b">Replete-AI/Replete-LLM-V2-Llama-3.1-8b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/rombodawg/Try_out_Replete-LLM-V2-Llama-3.1-8b">Try Out Replete-LLM-V2-Llama-3.1-8b - a Hugging Face Space by rombodawg</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Replete-LLM-V2-Llama-3.1-8b-exl2">bartowski/Replete-LLM-V2-Llama-3.1-8b-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Replete-LLM-V2-Llama-3.1-8b-GGUF">bartowski/Replete-LLM-V2-Llama-3.1-8b-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1276620195619934239)** (25 messages🔥): 

> - `Co-Writer`
> - `Discord Bots`
> - `LLM Model`
> - `LLM Quantization`
> - `MoE` 


- **Co-Writer Bot**: A member asked for resources to create a Discord bot that imitates their friends, but they were told that there are no guides available to do so.
   - Another member suggested Shapes Inc AI Bot Maker, which allows you to make bots behave like friends, but doesn't allow for custom training.
- **Large Language Model (LLM) Quantization Explained**: A member asked about the differences between the `_S`, `_M`, and `_L` models within the same quantization, using the [Hermes 3.1 70B GGUF](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B-GGUF/tree/main) as an example.
   - Another member explained that the different suffixes refer to various mixes of quantization, not every weight gets the same quantization, and this impacts the overall size and performance of the model.
- **Sparse Mixture of Experts (MoE) Model**: A member asked about the benefits and drawbacks of using Sparse Mixture of Experts (MoE) models in large language models (LLMs).
   - They were informed that MoEs are primarily used by large labs serving APIs to reduce computational costs and, while promising, currently underperform dense models trained on the same number of tokens.
- **Google's One Million Experts (MoE) Paper**: A member shared Google's recent paper on the **One Million Experts MoE** which claims that the model **outperforms dense feed forward networks**.
   - [The paper](https://www.clioapp.ai/research/million-moe) describes how **MoEs** work by having many smaller neural networks, called **experts**, each focused on a specific task or domain, and the model chooses which expert is best suited to handle specific parts of the input, unlike traditional transformer models with dense feedforward networks.
- **Training LoRA on SmoLLM**: A member shared a concern about their **LoRA training** on **SmoLLM**, with a loss of **4.0 for a 330M parameter model**. 
   - Another member suggested that this might be a bad loss and recommended a **Markov chain** implementation for beginners in Go, which they linked to on **GitHub**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.clioapp.ai/research/million-moe">Mixture of a Million Experts | Clio AI Insights</a>: Google Deepmind introduced Mixture of Experts which led to GPT-4. This is another Google Deepmind Paper asking the question &quot;What if we scale this to a million experts?&quot;</li><li><a href="https://github.com/mb-14/gomarkov">GitHub - mb-14/gomarkov: Markov chains in golang</a>: Markov chains in golang. Contribute to mb-14/gomarkov development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1276825527692562432)** (8 messages🔥): 

> - `Pints LLM`
> - `Medical AI Research`
> - `LLM Benchmarking`
> - `LLM Training Efficiency`
> - `Multimodal LLMs` 


- **Pints LLM Outperforms OpenELM and Phi**: A compact LLM, **"1.5-Pints"**, pretrained in only 9 days by using a curated dataset of 57 billion tokens, outperformed both **Apple's OpenELM** and **Microsoft's Phi** on **MT-Bench**, a benchmark that emulates human judgments.
   - The **Pints** team used a **modified Mistral tokenizer** and a **Llama-2 architecture** for training, prioritizing "textbook-like" content for reasoning and logical deduction.
- **Medical AI Research Papers**: There's a discussion about sharing medical AI research papers on the channel.
   - A user provided a list of top medical AI research papers from August 17 - August 24, 2024, which includes topics like "Jailbreak on Medical Multimodal LLMs" and "LLaVA-Surg: Multimodal LLM Surgical Assistant".
- **LLM Benchmarking with MT-Bench**: The **MT-Bench** benchmark is used to evaluate the performance of LLMs as instruction-following assistants.
   - The **MT-Bench** emulates human judgments and is a valuable tool for comparing LLMs across different use cases.
- **Efficient LLM Training in 9 Days**: The **Pints** team achieved impressive training efficiency by pretraining their model in just 9 days.
   - This was made possible through a carefully curated dataset and effective training methodologies, showcasing the potential for developing high-quality LLMs in a shorter timeframe.
- **Multimodal LLMs in Medicine**: Several papers discussed in the medical AI research list highlight the use of **multimodal LLMs** in healthcare.
   - These models leverage both text and visual data to perform tasks like surgical assistance and clinical trial transition prediction, demonstrating the potential of multimodal approaches in the medical domain.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1827442651810918509">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: Last Week in Medical AI: Top Research Papers/Models 🏅(August 17 - August 24, 2024)  - Jailbreak on Medical Multimodal LLMs - LLMs are not Zero-Shot Biomedical Reasoners  - RuleAlign framework: Aligni...</li><li><a href="https://github.com/pints-ai/1.5-Pints">GitHub - Pints-AI/1.5-Pints: A compact LLM pretrained in 9 days by using high quality data</a>: A compact LLM pretrained in 9 days by using high quality data - Pints-AI/1.5-Pints</li><li><a href="https://www.arxiv.org/abs/2408.03506">1.5-Pints Technical Report: Pretraining in Days, Not Months -- Your Language Model Thrives on Quality Data</a>: This paper presents a compute-efficient approach to pre-training a Language Model-the &#34;1.5-Pints&#34;-in only 9 days, while outperforming state-of-the-art models as an instruction-following assist...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1276822589716561940)** (3 messages): 

> - `Pints-AI 1.5`
> - `Sparse-Marlin`
> - `Text Generation Webui` 


- **Pints-AI 1.5 outperforms Apple OpenELM and Microsoft Phi**: A new **LLM** trained in only **9 days** by **Pints-AI** called **Pints-AI 1.5** outperformed **Apple OpenELM** and **Microsoft Phi**. 
   - You can find the training code here: [Pints-AI 1.5](https://github.com/pints-ai/1.5-Pints).
- **Sparse-Marlin accelerates matrix multiplication with 4-bit quantized weights**: A new GPU-optimized kernel called **Sparse-Marlin** was integrated into **vllm_project**.
   - Sparse-Marlin achieves **5.3x speedups** on **NVIDIA GPUs** (Ampere/Ada) by using **4-bit quantized weights** and **2:4 sparsity** and maintains efficiency with batch sizes up to 32.
- **Text Generation Webui DRY Feature Update**: A new feature called **DRY** was added to the **Text Generation Webui** to prevent models from repeating phrases in their output.
   - DRY is a modern repetition penalty that helps to reduce the likelihood of models from repeating phrases verbatim that have previously occurred in the input.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/neuralmagic/status/1827285549058572566">Tweet from Neural Magic (@neuralmagic)</a>: Sparse-Marlin is here and integrated into @vllm_project! This GPU-optimized kernel accelerates matrix multiplication with 4-bit quantized weights and 2:4 sparsity, achieving 5.3x speedups on NVIDIA GP...</li><li><a href="https://github.com/pints-ai/1.5-Pints">GitHub - Pints-AI/1.5-Pints: A compact LLM pretrained in 9 days by using high quality data</a>: A compact LLM pretrained in 9 days by using high quality data - Pints-AI/1.5-Pints</li><li><a href="https://github.com/oobabooga/text-generation-webui/pull/5677">DRY: A modern repetition penalty that reliably prevents looping by p-e-w · Pull Request #5677 · oobabooga/text-generation-webui</a>: Looping is an undesirable behavior where the model repeats phrases verbatim that have previously occurred in the input. It affects most models, and is exacerbated by the use of truncation samplers....
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1276825527692562432)** (8 messages🔥): 

> - `1.5-Pints LLM`
> - `Medical AI Research`
> - `LLaVA-Surg`
> - `HIBOU`
> - `RuleAlign` 


- **1.5-Pints LLM Outperforms OpenELM and Phi**: A new model called **1.5-Pints** has been pretrained in **9 days** and outperforms **Apple OpenELM** and **Microsoft Phi** as an instruction-following assistant. 
   - The model was trained on a curated dataset of **57 billion tokens**, prioritizing **expository and textbook-like content** to aid in **reasoning and logical deduction**. The model uses a modified **Mistral tokenizer** and a **Llama-2 architecture** for wider compatibility. 
- **Medical AI Research Papers Shared**: A user asked if sharing **medical AI research papers** in the channel is allowed, and it was confirmed that they are welcome as long as they are related to AI.
   - The user shared a list of top **medical AI research papers and models** from August 17-24, including **LLaVA-Surg**, **HIBOU**, **RuleAlign**, **CTP-LLM**, **MEDCO**, **Clinical Insights**, **Federated Knowledge Injection**, and **an EMR Dataset** for clinical multi-step diagnosis.
- **LLaVA-Surg: A Multimodal Surgical Assistant**: One of the top medical AI research papers highlighted is **LLaVA-Surg**, a **multimodal LLM surgical assistant**. 
   - This paper explores the potential of using LLMs to assist surgeons in the operating room, suggesting a new frontier for medical AI.
- **HIBOU: Foundational Vision Transformer for Pathology**: Another noteworthy paper is **HIBOU**, a **foundational vision transformer for pathology**. 
   - This paper focuses on using deep learning techniques to analyze pathology images, potentially revolutionizing medical diagnosis and treatment.
- **RuleAlign: Aligning LLMs for Physician Rules**: The **RuleAlign framework** aims to align LLMs with **physician rules** to improve medical decision-making.
   - This paper highlights the importance of incorporating medical expertise into AI systems for safer and more effective healthcare.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1827442651810918509">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: Last Week in Medical AI: Top Research Papers/Models 🏅(August 17 - August 24, 2024)  - Jailbreak on Medical Multimodal LLMs - LLMs are not Zero-Shot Biomedical Reasoners  - RuleAlign framework: Aligni...</li><li><a href="https://github.com/pints-ai/1.5-Pints">GitHub - Pints-AI/1.5-Pints: A compact LLM pretrained in 9 days by using high quality data</a>: A compact LLM pretrained in 9 days by using high quality data - Pints-AI/1.5-Pints</li><li><a href="https://www.arxiv.org/abs/2408.03506">1.5-Pints Technical Report: Pretraining in Days, Not Months -- Your Language Model Thrives on Quality Data</a>: This paper presents a compute-efficient approach to pre-training a Language Model-the &#34;1.5-Pints&#34;-in only 9 days, while outperforming state-of-the-art models as an instruction-following assist...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1277449679650422835)** (1 messages): 

> - `Whisper Diarization`
> - `Whisper v3` 


- **Whisper Diarization Request**: A user is seeking information on implementing **Whisper diarization**, specifically asking if anyone has experience with it and is willing to share a script.
   - They mentioned having a script that utilizes **Whisper v3** locally or through **Groq**, but are looking for a solution that can identify speaker changes.
- **Using Whisper v3 for Diarization**: The user has a **Whisper v3** script that they are currently using for audio processing.
   - This script can be run locally or on a **Groq** platform, but they are seeking a way to integrate diarization capabilities to identify speaker changes.


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1276618062891847811)** (512 messages🔥🔥🔥): 

> - `Unsloth`
> - `Ollama`
> - `Llama.cpp`
> - `LinkedIn`
> - `Model Merging` 


- **Unsloth is accused of being ripped off by LinkedIn**: Some members of the Unsloth Discord channel are accusing LinkedIn of stealing code from their project, particularly in their Triton kernel implementation.
   - They point to [LinkedIn's Liger-Kernel repository on GitHub](https://github.com/linkedin/Liger-Kernel) and a post on [Ollama](https://ollama.com/unclemusclez/qwen2-unsloth) as evidence of the alleged theft. They also note that LinkedIn has benchmarked its kernels against Unsloth, but only in terms of kernels, suggesting they may be using Unsloth's work without contributing back to the project. 
- **Unsloth's performance vs. Hugging Face and LinkedIn**: Several members discuss the performance of Unsloth compared to other platforms like Hugging Face and LinkedIn, particularly in terms of speed, memory usage, and accuracy.
   - Some members note that while Unsloth is faster and more efficient in terms of training and inference, it does not support 8-bit models yet, unlike Hugging Face and LinkedIn. 
- **The pros and cons of open source**: The Unsloth Discord channel is abuzz with discussion about the nature of open source and the balance between collaboration and exploitation.
   - Some members argue that the open source model is valuable because it allows for two-way benefits, with companies contributing back to the projects they use, while others express frustration that some companies take advantage of open source projects without giving anything back. 
- **How to train models on Arabic, Persian, or other languages**: Several members discuss the challenges of training models on languages like Arabic and Persian, highlighting the need for specific datasets and continued pretraining to achieve good results.
   - One member suggests that Persian datasets and Persian Wikipedia could be used to pretrain Llama-3 models for better results in Persian, while another member advises against using Llama 3.1 for Persian as it does not currently support it.
- **Model deployment and inference with Unsloth**: The chat revolves around deploying and running models with Unsloth, particularly on platforms like Hugging Face and Fireworks.ai.
   - One member asks about using Unsloth for inference after deploying a model to Hugging Face, highlighting challenges with timeouts and model size limitations. Another member recommends using vLLM for inference and suggests that Unsloth's documentation should mention it as an option. 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: New to Unsloth? Start here!</li><li><a href="https://colab.research.google.com/drive/1weTpKOjBZxZJ5PQ-Ql8i6ptAY2x-FWVA?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/15OyFkGoCImV9dSsewU1wa2JuKB4-mDE_?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/docs/autotrain/main/en/sentence_transformer">Sentence Transformers</a>: no description found</li><li><a href="https://huggingface.co/rahulAkaVector/infosys-business-model-4bit">rahulAkaVector/infosys-business-model-4bit · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/installation/pip-install">Pip Install | Unsloth Documentation</a>: To install Unsloth locally via Pip, follow the steps below:</li><li><a href="https://huggingface.co/rahulAkaVector/java_code_generator/tree/main">rahulAkaVector/java_code_generator at main</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://tenor.com/view/snattacatz-gif-25782415">Snattacatz GIF - Snattacatz - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/rahulAkaVector/modelz/tree/main">rahulAkaVector/modelz at main</a>: no description found</li><li><a href="https://ui.endpoints.huggingface.co/catalog">Model Catalog | Inference Endpoints - Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/microsoft/Phi-3.5-mini-instruct">microsoft/Phi-3.5-mini-instruct · Hugging Face</a>: no description found</li><li><a href="https://ollama.com/unclemusclez/qwen2-unsloth">unclemusclez/qwen2-unsloth</a>: Get up and running with large language models.</li><li><a href="https://huggingface.co/docs/trl/main/en/dpo_trainer">DPO Trainer</a>: no description found</li><li><a href="https://huggingface.co/ai21labs/AI21-Jamba-1.5-Large#ruler-benchmark---effective-context-length>">ai21labs/AI21-Jamba-1.5-Large · Hugging Face</a>: no description found</li><li><a href="https://dontasktoask.com/">Don't ask to ask, just ask</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ezq84k/llama_31_chat_template">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=rpAtVIZB72U">LLAMA-3.1 🦙: EASIET WAY To FINE-TUNE ON YOUR DATA 🙌</a>: Learn how to efficiently fine-tuning the Llama 3.1 model using Unsloth, LoRa, and QLoRa techniques. LINKS:Colab: https://tinyurl.com/bdzxhy5nUnsloth: https:/...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ezq84k/llama_31_chat_template_quirks_chat_notebook/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=XZ3w_jec1v8">&quot;The Economics of Programming Languages&quot; by Evan Czaplicki (Strange Loop 2023)</a>: In the mythology of open source, programming languages are created by people who seemingly have no direct economic function. They are just really good at com...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/57">Benchmark against unsloth · Issue #57 · linkedin/Liger-Kernel</a>: 🚀 The feature, motivation and pitch hey, did you run any benchmark against unsloth which uses similar kernels? I guess your project can be used as a dropdown replacement with multi gpu support. Alt.....</li><li><a href="https://github.com/linkedin/Liger-Kernel?trk=comments_comments-list_comment-text">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://huggingface.co/unsloth/Phi-3.5-mini-instruct">unsloth/Phi-3.5-mini-instruct · Hugging Face</a>: no description found</li><li><a href="https://x.com/TheAITimeline/status/1827467655844131070">Tweet from The AI Timeline (@TheAITimeline)</a>: 🚨This week’s top AI/ML research papers:  - Transfusion - To Code, or Not To Code? - Automated Design of Agentic Systems - LLM Pruning and Distillation in Practice: Minitron - Hermes 3 Technical Repor...</li><li><a href="https://github.com/microsoft/T-MAC">GitHub - microsoft/T-MAC: Low-bit LLM inference on CPU with lookup table</a>: Low-bit LLM inference on CPU with lookup table. Contribute to microsoft/T-MAC development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1276823137387810889)** (14 messages🔥): 

> - `Liger Kernel`
> - `Triton`
> - `EAGLE`
> - `LLM Training` 


- **Liger Kernel Speeds Up LLM Training**: A member shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1eznkml/liger_kernel_one_line_to_make_llm_training_20/) claiming that a new kernel called **Liger Kernel** can make LLM training **20% faster** and reduce memory usage by **60%**.
   - Another member noted that the kernel utilizes **Triton** and has been very effective in speeding up training times.
- **Triton Kernel Debugging**: A member sought help debugging their implementation of a research paper's code using **Triton**.
   - They received suggestions to post their code in either the general channel or the **Triton** specific channel for support.
- **EAGLE Library Encountering Triton Errors**: A member shared their code for a **Triton** kernel within the **EAGLE** library and encountered a **RuntimeError**.
   - They suspected the issue was related to a large **block size** in the code and were advised to check the [EAGLE repository](https://github.com/abhijit-aiplanet/EAGLE) for potential solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheAITimeline/status/1827467655844131070">Tweet from The AI Timeline (@TheAITimeline)</a>: 🚨This week’s top AI/ML research papers:  - Transfusion - To Code, or Not To Code? - Automated Design of Agentic Systems - LLM Pruning and Distillation in Practice: Minitron - Hermes 3 Technical Repor...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eznkml/liger_kernel_one_line_to_make_llm_training_20/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/triton-lang/triton/issues/1058">Invalid memory access on A100 for very large tensors · Issue #1058 · triton-lang/triton</a>: Hi, I&#39;m trying to write a triton kernel that fills a float16 tensor with random +1/-1 entries. My code is below: import torch as ch import triton import triton.language as tl @triton.jit def _fill...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eznkml/li">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/abhijit-aiplanet/EAGLE/blob/main/eagle/model/modeling_llama_kv.py">EAGLE/eagle/model/modeling_llama_kv.py at main · abhijit-aiplanet/EAGLE</a>: EDITED IMPLEMENTATION OF EAGLE. Contribute to abhijit-aiplanet/EAGLE development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1276622349747359848)** (67 messages🔥🔥): 

> - `Unsloth CPU Compatibility`
> - `Unsloth on Tesla P100`
> - `Unsloth SFTTrainer Override`
> - `Unsloth Steps vs Epochs`
> - `Unsloth Phi-3.5-MoE Availability` 


- **Unsloth on CPU**: A user asked if Unsloth can run on a CPU.
- **P100 Issues with Unsloth**: A user reported encountering an LLVM error when trying to run Unsloth with a Tesla P100 GPU.
- **SFTTrainer Override for Custom Logic**: A user asked how to override the SFTTrainer's compute_loss function to add custom logic based on the LLM's output string.
- **Steps vs Epochs in Unsloth Training**: A user asked about the difference between steps and epochs in Unsloth training.
- **Unsloth Phi-3.5-MoE Availability**: A user inquired about the availability of the Phi-3.5-MoE model on Unsloth.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=Zt9CHJqO6p30)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/135ced7oHy">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/linkedin/Liger-Kernel">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/blob/b8b1eafda35d124046e11766aeeb6343957e0daf/unsloth/kernels/rms_layernorm.py">unsloth/unsloth/kernels/rms_layernorm.py at b8b1eafda35d124046e11766aeeb6343957e0daf · unslothai/unsloth</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1277046213514104893)** (1 messages): 

> - `Replete-LLM V2`
> - `Llama 3.1`
> - `Replete-AI/The_Living_AI_Dataset`
> - `System Prompts`
> - `Quantizations` 


- **Replete-LLM V2: The Second Coming**: The second version of Replete-LLM, **Replete-LLM-V2-Llama-3.1-8b**, is here, boasting massive reasoning and coding performance improvements over its predecessor.
   - This version is trained on the new **Replete-AI/The_Living_AI_Dataset** to teach the model about **Love and Empathy**, a crucial step towards building models that understand and care for us.
- **System Prompts: The Key to Replete-LLM V2's Power**: **Replete-LLM-V2** is trained with various system prompts to guide its information processing.
   - Detailed, specific, and effective system prompts are crucial for achieving optimal performance from this model.
- **Quantizations for Local Running**: Quantization files for **Replete-LLM-V2-Llama-3.1-8b** are available using **ExLlamaV2 v0.1.9** and **Llama.cpp**.
   - These quantizations allow users to run the model locally and can be used with **LM Studio** for a more interactive experience.
- **Prompt Format: The Right Way to Interact**: The recommended prompt format for interacting with **Replete-LLM-V2** is a three-part structure:
   - It begins with a **system prompt**, followed by a **user prompt**, and concludes with an **assistant response**.
- **Llama 3.1: The Engine Behind the Model**: **Replete-LLM-V2** is based on the new **Llama 3.1** model, showcasing its advanced capabilities in various tasks.
   - This new model provides a foundation for improved performance and further development in the field of AI.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Replete-AI/Replete-LLM-V2-Llama-3.1-8b">Replete-AI/Replete-LLM-V2-Llama-3.1-8b · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/rombodawg/Try_out_Replete-LLM-V2-Llama-3.1-8b">Try Out Replete-LLM-V2-Llama-3.1-8b - a Hugging Face Space by rombodawg</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Replete-LLM-V2-Llama-3.1-8b-exl2">bartowski/Replete-LLM-V2-Llama-3.1-8b-exl2 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/bartowski/Replete-LLM-V2-Llama-3.1-8b-GGUF">bartowski/Replete-LLM-V2-Llama-3.1-8b-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1277621710945914881)** (3 messages): 

> - `LLM Fine-Tuning for Question Answering`
> - `Unsloth and Llama Model` 


- **Unsloth and Llama for Question Answering**: A member requested an example of fine-tuning an LLM for question answering using Unsloth and Llama.
   - Another member suggested exploring GitHub for potential notebooks and discouraged spamming across multiple channels.
- **Don't Be Lazy, Explore GitHub**: A member recommended checking GitHub for potential notebooks on this topic.
   - They also discouraged spamming the same question in multiple channels.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1276635266374828062)** (570 messages🔥🔥🔥): 

> - `Stable Diffusion Online`
> - `Stable Diffusion Seed`
> - `ComfyUI`
> - `Stable Diffusion Image Upscaling`
> - `Flux` 


- **Stable Diffusion Online is not Affiliated with Stability AI**: A member questioned whether [Stable Diffusion Online](https://stabledifffusion.com) is an official site or is unrelated to Stability AI.
- **ComfyUI: The Ultimate Diffusion Workflow?**: A member suggested that if a user isn't going to use all the powerful stuff that ComfyUI affords, they should just use ForgeUI.
- **Exploring the Upscaling Landscape: SD Upscale, Tiled Diffusion, and Other Approaches**: Members discussed different approaches to upscaling images while gaining detail: Ultimate SD Upscale, Tiled Diffusion, and the use of models like '4x-NomosWebPhoto-atd' with SUPIR after.
- **The Enigma of 'Extra Noise' in A1111/Forge**: A member explained the concept of 'Noise Injection' in A1111/Forge and how it can be used to improve image upscales.
- **Navigating the Flux Landscape: Overfitting, Diversity, and Instruct Pix2Pix**: Members discussed the overfitting of Flux, particularly in relation to fantasy and how it can lead to a lack of image diversity.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/typing-bald-big-body-laptop-muscle-man-gif-17063750">Typing Bald GIF - Typing Bald Big Body - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://stable-diffusion-art.com/instruct-pix2pix/">Instruct Pix2Pix: Edit and stylize photos with text - Stable Diffusion Art</a>: Instruct Pix2Pix is a Stable Diffusion model that edits images with the user&#039;s text instruction alone. We will go through how it works, what it can do, how to</li><li><a href="https://huggingface.co/docs/accelerate/usage_guides/distributed_inference">Distributed Inference with 🤗 Accelerate</a>: no description found</li><li><a href="https://tenor.com/view/shrek-shrek-rizz-rizz-gif-11157824601050747846">Shrek Shrek Rizz GIF - Shrek Shrek rizz Rizz - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1dkh5cv/a1111_forge_improve_your_images_with_a_single/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://huggingface.co/ByteDance/Hyper-SD">ByteDance/Hyper-SD · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/mike-lowrey-gif-8186790">Mike Lowrey GIF - Mike Lowrey - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/unity-research/IP-Adapter-Instruct">GitHub - unity-research/IP-Adapter-Instruct: IP Adapter Instruct</a>: IP Adapter Instruct. Contribute to unity-research/IP-Adapter-Instruct development by creating an account on GitHub.</li><li><a href="https://stabledifffusion.com">Stable Diffusion Online - Free AI Image Generator</a>: Stable Diffusion is a free Artificial Intelligence image generator that easily creates high-quality AI art, images, anime, and realistic photos from simple text prompts. No sign-up!</li><li><a href="https://t.me/dogshouse_bot/join?startapp=8PYp1s3kTTSEkZBzibx3Qw">Dogs 🦴</a>: The most Telegram-native memecoin. Join our @dogs_community</li><li><a href="https://t.me/dogs_community)">Telegram – a new era of messaging</a>: Fast. Secure. Powerful.
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1276619011861385316)** (366 messages🔥🔥): 

> - `Codegemma`
> - `Hermes`
> - `Mistral`
> - `Model Merging`
> - `Open Empathic` 


- **Hermes 2.5 Outperforms Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** appears to perform better than **Hermes 2** in various benchmarks.
   - Hermes 2 scored a **34.5** on the MMLU benchmark whereas Hermes 2.5 scored **52.3**.
- **Mistral Struggles Expanding Beyond 8k**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed "cursed model merging".
- **Open Empathic Project Plea for Assistance**: A member appealed for help in expanding the categories of the **Open Empathic** project, particularly at the lower end.
   - They shared a [YouTube video on the Open Empathic Launch & Tutorial](https://youtu.be/GZqYr8_Q7DE) that guides users to contribute their preferred movie scenes from YouTube videos, as well as a link to the [OpenEmpathic project itself](https://dct.openempathic.ai/).
- **GPTs Agents Cannot Learn After Initial Training**: A member shared a concern about GPTs agents not learning from additional information provided after their initial training.
   - Another member cleared this misunderstanding, explaining that [uploaded files are saved as "knowledge" files](https://link.to/openai-docs) for the agent to reference when required, but **they do not continually modify the agent's base knowledge**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/docs/hub/en/security-git-ssh#checking-for-existing-ssh-keys">Git over SSH</a>: no description found</li><li><a href="https://huggingface.co/spaces/Vipitis/shadermatch/discussions/1">Vipitis/shadermatch · Accessibility notice</a>: no description found</li><li><a href="https://huggingface.co/AiTommy/decider_agent_test_merged">AiTommy/decider_agent_test_merged · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/AiTommy/decider_agent_lora">AiTommy/decider_agent_lora · Hugging Face</a>: no description found</li><li><a href="https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#pull-request-checklist>.">transformers/CONTRIBUTING.md at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://youtu.be/3u8HndJlA0c?si=NJ2RJQs7y-uo_-M-">Will &quot;Claude Investor&quot; DOMINATE the Future of Investment Research?&quot; - AI Agent Proliferation Begins</a>: Learn AI With Me:https://www.skool.com/natural20/aboutJoin my community and classroom to learn AI and get ready for the new world.[LINKS BELOW]JOSH BICKETTCa...</li><li><a href="https://youtu.be/3u8HndJlA0c?si=NJ2RJQ">Will &quot;Claude Investor&quot; DOMINATE the Future of Investment Research?&quot; - AI Agent Proliferation Begins</a>: Learn AI With Me:https://www.skool.com/natural20/aboutJoin my community and classroom to learn AI and get ready for the new world.[LINKS BELOW]JOSH BICKETTCa...</li><li><a href="https://github.com/vatsalsaglani/GenAINewsAgent?tab=readme-ov-file">GitHub - vatsalsaglani/GenAINewsAgent: A quick implementation of a GenAI News Summary agent using Llama-3 over Groq.</a>: A quick implementation of a GenAI News Summary agent using Llama-3 over Groq. - vatsalsaglani/GenAINewsAgent</li><li><a href="https://github.com/mshumer/gpt-investor">GitHub - mshumer/gpt-investor</a>: Contribute to mshumer/gpt-investor development by creating an account on GitHub.</li><li><a href="https://github.com/Vipitis/shadertoys-dataset/blob/dev-testing/experiments/run_experiments.ipynb?short_path=97d4273#L4885>)">shadertoys-dataset/experiments/run_experiments.ipynb at dev-testing · Vipitis/shadertoys-dataset</a>: WIP refactor of a dataset. Contribute to Vipitis/shadertoys-dataset development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1276678799852830802)** (18 messages🔥): 

> - `Model Quantization`
> - `Model Distillation`
> - `Diffusion Model Training`
> - `GPT-2 Rewriting`
> - `Prompt Weights` 


- **Model Quantization and Distillation are Production Essentials**: The author discusses the importance of **Model Quantization** and **Model Distillation** for machine learning engineers to productionize their models, beyond local training and testing.
- **Nautilus integrates NIST's Dioptra for Redteaming AI**: **Nautilus**, a security platform developed by **Qompass**, has integrated **NIST's Dioptra** for **Redteaming AI**.
   - The author expresses optimism that **NIST** is open-sourcing these tools, suggesting that it might be beneficial to understand their approach to **AI security**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://phrygian-alarm-029.notion.site/notes-reading-7b6946b8f4074771b9922654d61075d6?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team</li><li><a href="https://www.youtube.com/watch?v=l8pRSuU81PU">Let&#39;s reproduce GPT-2 (124M)</a>: We reproduce the GPT-2 (124M) from scratch. This video covers the whole process: First we build the GPT-2 network, then we optimize its training to be really...</li><li><a href="https://github.com/qompassai/Nautilus/tree/main/RedTeam/RedAI/NIST/Qompass_Dioptra">Nautilus/RedTeam/RedAI/NIST/Qompass_Dioptra at main · qompassai/Nautilus</a>: NIST-compliant Security solutions. Contribute to qompassai/Nautilus development by creating an account on GitHub.</li><li><a href="https://www.nist.gov/news-events/news/2024/07/department-commerce-announces-new-guidance-tools-270-days-following">Department of Commerce Announces New Guidance, Tools 270 Days Following President Biden’s Executive Order on AI</a>: For the first time, Commerce makes public new NIST draft guidance from U.S. AI Safety Institute to help AI developers evaluate and mitigate risks stemming from generative AI and dual-use foundation mo...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1276659260767338528)** (6 messages): 

> - `Stale bot`
> - `LLM Merging`
> - `Medical AI Papers`
> - `1-bit LLMs` 


- **Stale bot's impact on Open Source Projects**: A study of 20 large open-source projects examined the effects of using [Stale bot](https://github.com/probot/stale) to automatically track and close inactive pull requests.
   - While Stale bot can help manage a backlog of unresolved PRs and improve review efficiency, it may also have negative consequences.
- **Awesome-Model-Merging-Methods: LLM Merging Resources**: A [GitHub repository](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications) curates papers on LLM merging methods, theories, and applications.
   - The repository is based on a paper titled "Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities".
- **Top Medical AI Research Papers: Week of August 17th**: A thread on Twitter highlights several noteworthy medical AI research papers published in the week of August 17th, 2024.
   - Papers discussed include topics like jailbreaking medical multimodal LLMs, rule-aligned LLMs for physicians, and clinical trial prediction using LLMs.
- **Microsoft Achieves Breakthrough with 1-bit LLMs**: Microsoft has made significant progress in 1-bit LLMs, creating models that use ternary values (-1, 0, 1) instead of the traditional 16-bit format.
   - This innovation results in a 2.7x speed increase, 3.5x reduction in GPU memory usage, and 71x decrease in energy consumption while maintaining or even exceeding the performance of conventional models like LLaMA 3B.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenlifesciAI/status/1827442651810918509">Tweet from Open Life Science AI (@OpenlifesciAI)</a>: Last Week in Medical AI: Top Research Papers/Models 🏅(August 17 - August 24, 2024)  - Jailbreak on Medical Multimodal LLMs - LLMs are not Zero-Shot Biomedical Reasoners  - RuleAlign framework: Aligni...</li><li><a href="https://arxiv.org/abs/2305.18150">Understanding the Helpfulness of Stale Bot for Pull-based Development: An Empirical Study of 20 Large Open-Source Projects</a>: Pull Requests (PRs) that are neither progressed nor resolved clutter the list of PRs, making it difficult for the maintainers to manage and prioritize unresolved PRs. To automatically track, follow up...</li><li><a href="https://github.com/pints-ai/1.5-Pints">GitHub - Pints-AI/1.5-Pints: A compact LLM pretrained in 9 days by using high quality data</a>: A compact LLM pretrained in 9 days by using high quality data - Pints-AI/1.5-Pints</li><li><a href="https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications">GitHub - EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications: Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities. arXiv:2408.07666.</a>: Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities. arXiv:2408.07666. - EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1276690002985746493)** (14 messages🔥): 

> - `Tau LLM`
> - `database match command`
> - `TauAgent class`
> - `reward signals`
> - `TinyLlama` 


- **Tau LLM series continues**: The creators of the Tau LLM series continue to refine their project, focusing on implementing new features such as a `database match` command, a `TauAgent` class, and reward signals.
   - They also discuss their initial training data, which includes three domains: math, grammar, and spelling.
- **TinyLlama's success in 9 days**: TinyLlama, a similar model to Tau LLM, was trained in just 9 days by the same research group and has outperformed Apple OpenELM and Microsoft Phi on MTBench.
   - They have released their training code on GitHub and the model weights on HuggingFace.
- **GT-AI Translation Tool**: A new tool called GT-AI is available for text translation, completion, language detection, and more.
   - It offers multiple AI models and responsive design, working seamlessly on desktop and mobile devices.
- **Voicee: A Superfast Voice Assistant**: A new voice assistant called Voicee has been developed with a latency of under 500ms, with an average latency of 700ms.
   - It works best in Google Chrome and is looking for user feedback.
- **Dark Sentience dataset for emotional AI**: A new dataset titled "Dark Sentience" has been curated to address the lack of emotional depth in AI language models, specifically targeting a darker, more introspective mood.
   - The dataset is designed to enhance the emotional intelligence of AI by exposing it to complex human emotions, including suicide, depression, and anxiety.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/maximuspowers/bias-detection-ner">Bias Detection NER - a Hugging Face Space by maximuspowers</a>: no description found</li><li><a href="https://huggingface.co/spaces/KingNish/Voicee">Voicee - a Hugging Face Space by KingNish</a>: no description found</li><li><a href="https://huggingface.co/blog/huseinzol05/2d-parallelism-ray-pytorch">2D Parallelism using Ray PyTorch</a>: no description found</li><li><a href="https://github.com/Om-Alve/GemmaFromScratch/">GitHub - Om-Alve/GemmaFromScratch</a>: Contribute to Om-Alve/GemmaFromScratch development by creating an account on GitHub.</li><li><a href="https://github.com/LegallyCoder/StockLlama">GitHub - LegallyCoder/StockLlama: StockLlama is a time series forecasting model based on Llama, enhanced with custom embeddings for improved accuracy.</a>: StockLlama is a time series forecasting model based on Llama, enhanced with custom embeddings for improved accuracy. - LegallyCoder/StockLlama</li><li><a href="https://youtube.com/live/0WUXHWen0F8?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 8</a>: Welcome back to our Tau LLM series! 🌟In this episode, we&#39;re diving into some crucial components of our project. Our highlights include:- **Building the Trai...</li><li><a href="https://github.com/huridocs/pdf-document-layout-analysis">GitHub - huridocs/pdf-document-layout-analysis: A Docker-powered service for PDF document layout analysis. This service provides a powerful and flexible PDF analysis service. The service allows for the segmentation and classification of different parts of PDF pages, identifying the elements such as texts, titles, pictures, tables and so on.</a>: A Docker-powered service for PDF document layout analysis. This service provides a powerful and flexible PDF analysis service. The service allows for the segmentation and classification of differen...</li><li><a href="https://youtube.com/live/J1_zpS4HZMc?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers | Part 7</a>: Welcome back to our Tau LLM series! 🌟In this episode, we dive into some exciting new developments and continue to refine our project. Our highlights include...</li><li><a href="https://github.com/pints-ai/1.5-Pints">GitHub - Pints-AI/1.5-Pints: A compact LLM pretrained in 9 days by using high quality data</a>: A compact LLM pretrained in 9 days by using high quality data - Pints-AI/1.5-Pints</li><li><a href="https://huggingface.co/collections/pints-ai/15-pints-66b1f957dc722875b153b276">1.5-Pints - a pints-ai Collection</a>: no description found</li><li><a href="https://gt-ai-j3yu.vercel.app/">Create Next App</a>: no description found</li><li><a href="https://github.com/U-C4N/GT-AI">GitHub - U-C4N/GT-AI</a>: Contribute to U-C4N/GT-AI development by creating an account on GitHub.</li><li><a href="https://insiders.dashtoon.com/dashanimexl/">Introducing DashAnime XL 1.0</a>: Dashtoon Studio is an innovative AI-powered platform designed to empower creators in the art of comic crafting. The immense popularity and profound cultural influence of anime have sparked a fervent d...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1276620023565258814)** (14 messages🔥): 

> - `AWS RAGCHECKER`
> - `DPO`
> - `KTO`
> - `LLMs Alignment` 


- **AWS RAGCHECKER paper discussion**: A member mentioned the [AWS RAGCHECKER paper](https://arxiv.org/pdf/2408.08067) and asked if it could be implemented in HuggingFace.
- **DPO & KTO finetuning methods**: Another member proposed discussing DPO and KTO finetuning methods and their role in current language models.
- **Constitutional DPO for LLMs alignment**: A member mentioned discussing constitutional DPO, a method used for aligning LLMs, and its importance in recent research.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1277273994667622401)** (2 messages): 

> - `FSRCNN implementation`
> - `PyTorch implementation` 


- **FSRCNN implementation needs improvement**: A member asked for advice on improving their simplified PyTorch implementation of FSRCNN.
   - They shared a link to their project on GitLab: [paper_replication](https://gitlab.com/amrirasyidi/paper_replication/-/blob/master/notebooks/0_dummy.ipynb?ref_type=heads).
- **Collaboration on FSRCNN project**: Another member shared their interest in working on the same project.
   - They expressed excitement about collaborating on improving the FSRCNN implementation.



**Link mentioned**: <a href="https://gitlab.com/amrirasyidi/paper_replication/-/blob/master/notebooks/0_dummy.ipynb?ref_type=heads">notebooks/0_dummy.ipynb · master · Amri Rasyidi / Paper Replication · GitLab</a>: GitLab.com

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1277253721536991335)** (7 messages): 

> - `VLLM memory usage`
> - `VLLM system memory utilization`
> - `Accelerate trainer not working on Google Colab TPU`
> - `Speculative decoding with tool calling` 


- **VLLM's Memory Usage Mystery**: A user reported experiencing significantly higher memory consumption when using **VLLM** compared to standard model loading, even with smaller models.
   - A link to a [GitHub discussion](https://github.com/vllm-project/vllm/discussions/241) about GPU memory consumption was provided, potentially offering insights into this discrepancy.
- **VLLM System Memory Hog?**: The user specified that **Qwen 7B** typically occupies about **14GB of VRAM**, but when running with **VLLM**, system memory usage jumps to **over 20GB**, despite setting **GPU memory utilization** to **0.95** and **CPU offload** to **0GB**.
   - The user expressed confusion about this unexpected system memory usage, wondering if this is standard behavior for **VLLM**.
- **Accelerate Trainer's TPU Troubles**: Another user encountered an issue with the **Accelerate trainer** not functioning as expected when using a **Google Colab TPU**.
   - The user shared a link to a [simple NLP notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb) and an image of the error, seeking guidance on troubleshooting the problem.
- **Speculative Decoding with Tool Calling: A Match Made in Heaven?**: A user inquired about the compatibility of **speculative decoding** with **tool calling**.
   - This question remains unanswered, suggesting further exploration into the intersection of these two techniques.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb)">Google Colab</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/discussions/241">Question regarding the nearly double GPU memory consumption. · vllm-project/vllm · Discussion #241</a>: when i loaded vicuna7b_v1.3 with default config ,i found that gpu memory cost around 23G, but in fastchat&#39;s readme says &#39;requires around 14GB of GPU memory for Vicuna-7B &#39; anyway, the thro...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1276837781116817420)** (8 messages🔥): 

> - `Hugging Face iOS App Model Configuration`
> - `Stable Diffusion Fine-Tuning`
> - `Quantization Changes in Stable Diffusion` 


- **Hugging Face iOS App Model Configuration is Fixed for Now**: A user asked if the Hugging Face iOS app model can be configured by users or if it's fixed as of now. There was no confirmation if the app will be upgradeable in the future.
   - null
- **Stable Diffusion Fine-Tuning with Personal Data**: Another user asked for help fine-tuning Stable Diffusion 2 with personal data using Dreambooth.
   - null
- **Quantization Changes in Stable Diffusion**: A user explained that they changed the quantization process in Stable Diffusion, which resulted in improved output quality and Lora behavior.
   - The user stated that pre-quantized checkpoints might be incompatible with the newest code and require re-quantization.
- **Fine-tuning Stable Diffusion with Automatic1111**: A user asked if another user who offered help with Stable Diffusion fine-tuning was using Automatic1111, a popular Stable Diffusion web UI.
   - null
- **Understanding the "CLASS_DIR" Variable for Fine-tuning**: A user expressed confusion about the "CLASS_DIR" variable used in Stable Diffusion fine-tuning, specifically whether it should contain only the user's data or other data as well.
   - The user mentioned that their data set is of different types of photos and they want to fine-tune a model that will generate realistic images.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1276667799103737977)** (275 messages🔥🔥): 

> - `GPTs`
> - `AGI`
> - `LLMs`
> - `Model Scaling`
> - `DeepMind` 


- **Model Scaling Diminishing Returns**: The discussion revolves around the idea that simply scaling up models with more compute power and data may reach a point of diminishing returns, where further gains become incremental.
   - This is evidenced by the performance of models like Llama 3.1 and Claude 3.5 Sonnet, where the improvement in performance was not proportional to the increase in compute power.
- **The Debate on AI Consciousness**: The conversation delves into the philosophical aspects of artificial intelligence, specifically addressing the question of whether current LLMs can be considered conscious.
   - Several arguments are presented, including the idea that AI consciousness may be governed by different laws than human consciousness, and that AI's understanding of the world is limited by their lack of organic experience.
- **The Future of AI: Beyond Scaling**: The need for algorithmic breakthroughs in deep learning architectures to push the boundaries of AI beyond scaling is discussed.
   - The participants acknowledge that while scaling has brought about impressive advancements, it is not sufficient for achieving AGI. There is a consensus that new and innovative approaches are needed to address fundamental limitations and enable true breakthroughs.
- **AI and the Question of Free Will**: The conversation explores the concept of free will and how it relates to AI systems.
   - The participants discuss the possibility that even though AI may not have free will in the same way that humans do, they are still capable of making choices based on their internal logic and reasoning.
- **The Philosophical Implications of AI**: The conversation touches on the philosophical implications of AI, including the potential for AI to reshape our understanding of consciousness, personhood, and the nature of reality itself.
   - The participants explore a range of viewpoints, from those who believe that AI are essentially tools to be used by humans to those who believe that AI may eventually challenge our understanding of what it means to be human.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=DlX3QVFUtQI">What runs GPT-4o? Inside Microsoft&#39;s 2024 AI supercomputer with Mark Russinovich</a>: Microsoft has built the world’s largest cloud-based AI supercomputer that is already exponentially bigger than it was just 6 months ago, paving the way for a...

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1277072344040345713)** (8 messages🔥): 

> - `GPT sharing and community building`
> - `GPT Builder issues`
> - `OpenAI API pricing and subscription models` 


- **Sharing GPTs: Where do they show up?**: A member inquired about sharing GPTs within the community and how to track their effectiveness.
   - They asked how to determine if a shared GPT is being used or if it is working for others, specifically regarding the 'share-output' or 'use-cases' functionality.
- **GPT Builder's Clickable Links Are Buggy**: A member reported difficulties with the GPT Builder producing clickable web search result links, describing the issue as inconsistent.
   - They inquired if anyone else has encountered this issue and if there's a known solution.
- **Subscription Models on OpenAI API**: A member asked how platforms manage subscription plans, like monthly subscriptions, using OpenAI's token-based pricing model.
   - They cited Chatbase as an example and inquired about how they might implement such a plan, considering that OpenAI only offers token-based pricing.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1276852519968047236)** (12 messages🔥): 

> - `GPTs custom builders`
> - `Prompt engineering templates`
> - `GPT jailbreak`
> - `Meta robot tag` 


- **Building custom GPTs using the GPT store**: A member suggested using the custom GPT builder to create a GPT model that understands a company's brand identity and tone of voice, and then using this model as the system prompt for content creation assistants.
   - The member suggested that the GPT prompt engineering model from the GPT store could be used as the system prompt in the OpenAI playground or API.
- **GPT Jailbreak Method Shared**: A member shared a new jailbreak method for GPT-4 that they had discovered and tested.
   - Another member stated that there is no thread for sharing jailbreak methods.
- **Checking meta robot tags using GPT**: A member asked if GPT can be used to check the meta robot tag of a URL for "noindex" and "nofollow".
   - The member wasn't sure if this was the right channel to ask the question.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1276852519968047236)** (12 messages🔥): 

> - `GPT-4 Jailbreak`
> - `Meta Robot Tag`
> - `Prompt Engineering` 


- **GPT-4 Jailbreak Discussion**: A user asked if there was a thread dedicated to sharing new GPT-4 jailbreak methods.
   - A response confirmed that there is no such thread.
- **Checking Meta Robot Tag with GPT**: A user asked if they could use GPT to check the "noindex" and "nofollow" meta robot tags of a URL.
- **Prompt Engineering for Content Creation Assistants**: A user sought a template or guide for prompt engineering content creation assistants, aiming to teach them brand identity, tone, and terms to avoid.
   - A response suggested using the custom GPT builder, then leveraging the resulting prompt as the system prompt for further control.


  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1276650067960270850)** (2 messages): 

> - `Perplexity Creator Community`
> - `Kale partnership`
> - `Perplexity Pro for LinkedIn Premium` 


- **Perplexity Creator Community Launches**: Perplexity AI has partnered with [Kale](https://www.kalecard.com/t/perplexity.ai?ref=perplexity.ai_discord) to launch the **Perplexity Creator Community**, a program that rewards creators for sharing engaging video content about Perplexity.
   - Creators can earn cash based on the impact of their videos, and can post on their own schedule.
- **Kale: Empowering Creators & Connecting Brands**: Kale is a platform that connects brands with social-savvy creators, rewarding them for authentic content.
   - Kale's algorithm matches creators with brands they buy from and are relevant to their social media activity, encouraging real and engaging content.
- **Free Perplexity Pro for LinkedIn Premium Subscribers**: Perplexity is offering one free year of **Perplexity Pro** to all **LinkedIn Premium subscribers**.
   - Perplexity Pro offers unlimited Pro searches, access to frontier AI models like **GPT-4 Omni, Claude 3.5 Sonnet, and Llama 3.1 405b**, file uploading for analysis, and multimodal capabilities.



**Link mentioned**: <a href="https://www.kalecard.com/t/perplexity.ai?ref=perplexity.ai_discord">Perplexity AI Creator Community</a>: Earn rewards to post about Perplexity AI!

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1276617638919012475)** (216 messages🔥🔥): 

> - `Perplexity Pro`
> - `GPT-4o`
> - `Perplexity's performance`
> - `Claude 3.5 Sonnet`
> - `Image generation in Perplexity` 


- **Perplexity Pro's LinkedIn Offer**: LinkedIn Premium users are being offered one year of Perplexity Pro, but some users are reporting that the offer is not available in the EU, despite others in the EU receiving the offer.
   - A user in a small EU country confirmed they received the offer, while another user in the EU confirmed that it worked for them.
- **GPT-4o vs Claude 3.5 Sonnet**: Discussions revolve around the strengths of GPT-4o for coding and Claude 3.5 Sonnet for knowledge retrieval, with GPT-4o being seen as better for STEM topics and Sonnet for code-related tasks.
   - Users note that Claude 3.5 Sonnet's knowledge is limited for answering questions about poetry and stories, unlike GPT-4o and Bing, but it performs well for coding tasks. Some users also highlight that ChatGPT offers multiple GPTs, making it a good option for various tasks.
- **Issues with Image Generation in Perplexity**: Users report issues with image generation in Perplexity, with some threads being nuked after attempting to generate images using Dalle3, and others experiencing problems with the quality of generated images.
   - One user suggests that the issue might be related to the image generation process itself, as a simple prompt resulted in a bricked thread.
- **Perplexity's Search Functionality**: Users are concerned about Perplexity's search functionality, reporting unreliable searches and answers that don't reflect current information, leading to doubts about the accuracy of fact-checking.
   - Some users also highlight the use of the "TRIGGER WEB SEARCH" prompt within prompts, which seems to improve search results, suggesting a potential bug in the search functionality.
- **Perplexity's Code Interpreter Rollout**: Perplexity's CEO has announced the rollout of code interpreter and plot rendering, but users are experiencing limited functionality.
   - A user stated that the code interpreter is very limited, not allowing changes to the generated output, and suggests using Google Colab for more advanced use cases.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AravSr">Tweet from undefined</a>: no description found</li><li><a href="https://tenor.com/view/sponge-bob-imagination-rainbow-gif-6957879033178108845">Sponge Bob Imagination GIF - Sponge Bob Imagination Rainbow - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/aravsrinivas/status/1827117952669512116?s=61">Tweet from Aravind Srinivas (@AravSrinivas)</a>: The government is paying for your Covid tests. Microsoft is paying for your ChatGPT. VCs are paying for your Perplexity. Zuck is paying for your weights. What do you have to complain about anon?  Quot...</li><li><a href="https://x.com/AravSrinivas/status/1825617944782758066">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Code interpreter and plot rendering are slowly being rolled out!  Quoting Phil (@phill__1)   Wow perplexitys code interpreter can now install libraries and display charts in the result! This enables m...</li><li><a href="https://github.com/pnd280/complexity/issues/new?labels=bug&projects=&template=bug_report.yml&title=%5BBug%5D%3A+">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://chromewebstore.google.com/detail/complexity/ffppmilmeaekegkpckebkeahjgmhggpj">Complexity - Chrome Web Store</a>: ⚡ Supercharge your Perplexity.ai</li><li><a href="https://youtu.be/H-0PsLmEi4A?si=ZgykVL1c9VfC70nO">How Perplexity Works 🤖 — with Denis Yarats</a>: Today&#39;s guest is Denis Yarats. Denis is co-founder &amp; CTO at Perplexity, one of my favorite products and one of the most successful AI startups today. Perplex...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1276648917127139348)** (24 messages🔥): 

> - `Perplexity Onboarding`
> - `Perplexity Page Features`
> - `Teaching Seniors`
> - `Steve Jobs Jacket`
> - `Model Motivation` 


- **Jai Bhagat Teaches Seniors Perplexity**: Instructional designer Jai Bhagat challenged himself to teach people over the age of 50 how to use Perplexity.
   - He created a special Easter Egg in his Perplexity Page, where users can watch him conduct a Perplexity Onboarding, showcasing the platform's potential in accessible learning.
- **Perplexity Page Feature Exploration**: The conversation delves into exploring Perplexity Page features, particularly highlighting the ability to create interactive and engaging learning resources.
   - The user emphasizes the enjoyment derived from utilizing Perplexity's capabilities to create educational content.
- **Exploring the World with Perplexity**: The conversation covers a diverse range of topics, from Steve Jobs's jacket for sale to the Israel-Lebanon conflict, demonstrating Perplexity's wide-ranging information retrieval capabilities.
   - The user showcases Perplexity's versatility by seamlessly transitioning between various subjects and queries.



**Link mentioned**: <a href="https://www.youtube.com/embed/ubHakI1ARR8">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1276817656632184852)** (6 messages): 

> - `Perplexity API Rate Limits`
> - `Pro Search API`
> - `Perplexity API Responsiveness` 


- **Newcode.ai Needs Increased API Rate Limits**: Maged Helmy, the CEO and founder of Newcode.ai, which boasts over 3,500 users, expressed urgent need for increased API rate limits for their Perplexity API integration.
   - Maged has been submitting applications for increased rate limits for the past six months with no response and requests a Perplexity team member to reach out directly.
- **Perplexity API User Experiences Frustration**: A user expressed dissatisfaction with the quality of responses received from the Perplexity API.
   - They specifically noted that the responses felt 'dumb' and that the model appeared to be ignoring instructions.
- **Pro Search API Availability**: A user inquired about the possibility of using the Perplexity Pro search feature through the API.
   - The user sought clarification on whether this capability exists or is currently under development.


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1276749289787228221)** (1 messages): 

> - `Database Outage` 


- **Database Outage**: A recent database change caused a ~2 minute outage.
   - The issue has been fixed and service should be back to normal.
- **Apologies for inconvenience**: We apologize for the inconvenience caused by the recent outage.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1276622577913168047)** (245 messages🔥🔥): 

> - `Grok-2`
> - `Grok-mini`
> - `Mistral`
> - `Claude`
> - `OpenRouter` 


- **Grok-2 and Grok-mini join the Leaderboard**: Exciting news, **xAI's Grok-2** and **Grok-mini** are now on the [LMSYS Leaderboard](https://lmsys.org/leaderboard) with over 6000 community votes!
   - **Grok-2** surpasses **GPT-4o** (May) and ties with the latest **Gemini** for #2 spot, while **Grok-2-mini** is #5, excelling in **Math** (#1) and #2 across the board for **Hard Prompts**, **Coding**, and **Instruction-following**.
- **Mistral struggles to scale beyond 8k**: Members raised concerns that **Mistral** cannot be extended beyond 8k without continued pretraining, pointing to [this known issue](https://link.to.issue).
   - They suggested further exploration of *mergekit* and *frankenMoE finetuning* to push performance boundaries.
- **Claude 3.5 Sonnet is down again**: Users reported **Claude 3.5 Sonnet** is experiencing frequent outages, impacting its availability.
   - While **Haiku** seems to be working, other models like **Hermes 3.5** are also experiencing issues, leading to speculation about broader issues impacting the models.
- **OpenRouter's API Key & Pricing**: Users are discussing how to add their own **API keys** to **OpenRouter** and if the displayed token pricing reflects the total cost, including the **OpenRouter fee**.
   - It was clarified that the token price displayed is in **OpenRouter credits**, and the fee is automatically calculated when adding credits to the account.
- **Exploring Open Source Models: DeepSeek and Codestral**: Members discussed the strengths and limitations of **DeepSeek Coder V2**, highlighting its exceptional performance-to-cost ratio for coding from scratch but its weaknesses in understanding and refactoring existing code.
   - **Codestral 22B** from **Mistral** is also mentioned as a strong contender for open weights coding model, currently available for free via **API**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.novelcrafter.com/">The Novel Writing Toolbox</a>: no description found</li><li><a href="https://zed.dev/blog/zed-ai">Introducing Zed AI - Zed Blog</a>: Powerful AI-assisted coding powered by Anthropic&#x27;s Claude, now available.</li><li><a href="https://aider.chat/docs/leaderboards/#code-refactoring-leaderboard">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://huggingface.co/CausalLM/miniG">CausalLM/miniG · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/settings/integrations">Integrations | OpenRouter</a>: Use your own provider keys with OpenRouter</li><li><a href="https://turingpi.com/product/turing-pi-2/">Buy Turing Pi 2, mini ITX cluster board - for sale</a>: The Turing Pi 2 is a 4-node mini ITX cluster board with a built-in Ethernet switch that runs Turing RK1, Raspberry Pi CM4 or Nvidia Jetson compute modules</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: See how you&#x27;ve been using models on OpenRouter.</li><li><a href="https://x.com/NousResearch/status/1828121648383566270">Tweet from Nous Research (@NousResearch)</a>: What if you could use all the computing power in the world to train a shared, open source AI model?  Preliminary report: https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://www.anthropic.com/news/prompt-caching">Prompt caching with Claude</a>: Prompt caching, which enables developers to cache frequently used context between API calls, is now available on the Anthropic API. With prompt caching, customers can provide Claude with more backgrou...</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-70b-instruct/parameters">Meta: Llama 3.1 70B Instruct – Recommended Parameters</a>: Check recommended parameters and configurations for Meta: Llama 3.1 70B Instruct - Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. This 70B instruct-tuned...</li><li><a href="https://docs.mistral.ai/capabilities/code_generation/">Code generation | Mistral AI Large Language Models</a>: Codestral</li><li><a href="https://www.latent.space/p/cosine">Is finetuning GPT4o worth it?</a>: How Cosine Genie reached 50% on SWE-Bench Lite, 30% on the full SWE-Bench, and 44% on OpenAI&#x27;s new SWE-Bench Verified, all state of the art results by the widest ever margin recorded.</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b/activity">Meta: Llama 3.1 405B (base) – Recent Activity</a>: See recent activity and usage statistics for Meta: Llama 3.1 405B (base) - Meta&#x27;s latest class of model (Llama 3.1) launched with a variety of sizes &amp; flavors. This is the base 405B pre-train...</li><li><a href="https://x.com/lmsysorg/status/1827041269534879784?s=46&t=Q_sUgNqB0V1zhMyW85SZDw">Tweet from lmsys.org (@lmsysorg)</a>: Chatbot Arena update❤️‍🔥  Exciting news—@xAI&#39;s Grok-2 and Grok-mini are now officially on the leaderboard!  With over 6000 community votes, Grok-2 has claimed the #2 spot, surpassing GPT-4o (May)...</li><li><a href="https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb">anthropic-cookbook/misc/prompt_caching.ipynb at main · anthropics/anthropic-cookbook</a>: A collection of notebooks/recipes showcasing some fun and effective ways of using Claude. - anthropics/anthropic-cookbook</li><li><a href="https://openrouter.ai/docs/provider-routing#ignoring-providers">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://openrouter.ai/settings/preferences">Settings | OpenRouter</a>: Manage your accounts and preferences</li><li><a href="https://openrouter.ai/models/lynn/soliloquy-v3">Llama 3 Soliloquy 7B v3 32K - API, Providers, Stats</a>: Soliloquy v3 is a highly capable roleplaying model designed for immersive, dynamic experiences. Trained on over 2 billion tokens of roleplaying data, Soliloquy v3 boasts a vast knowledge base and rich...
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1276706249349660755)** (67 messages🔥🔥): 

> - `OMI`
> - `LLM failure mode`
> - `Anthropic Interpretability Work`
> - `Sparse MoE`
> - `Model Interpretability` 


- **OMI Model Competence**: A discussion about the competence of OMI participants in creating AI models from scratch arose, but no definitive opinions or assessments were shared.
- **LLM Repetition Failure Mode**: A common failure mode where an LLM gets stuck in a loop, repeating the same phrase, was discussed.
   - The specific conditions causing this are unclear, but it's hypothesized to be related to model over-quantization and the tendency to minimize cross-entropy loss by repeating the same phrase.
- **Anthropic's Interpretability Work: Cost Estimation**: A member inquired about the cost of replicating Anthropic's mechanistic interpretability work, particularly for models like Llama 8B or Mistral.
   - The high cost of the process is attributed to data-hungry and compute-intensive aspects, but no precise estimations were provided.
- **Sparse MoE: GPU Sparsity Advantage**: A question about how Sparse MoE leverages sparsity on GPUs was raised.
   - It was explained that sparsity benefits distributed training by allowing experts to be split across machines, enabling efficient utilization, and also applies to distributed inference.
- **Model Training and Evaluation Logging: Best Practices**: A member sought recommendations for platforms suitable for logging model training and evaluation runs.
   - The discussion explored various methods, including text files, CSV files, and even using git for managing training logs and results, with potential benefits and drawbacks of each approach.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.04093">Scaling and evaluating sparse autoencoders</a>: Sparse autoencoders provide a promising unsupervised approach for extracting interpretable features from a language model by reconstructing activations from a sparse bottleneck layer. Since language m...</li><li><a href="https://github.com/rokosbasilisk/filler_tokens/blob/v2/paper.pdf">filler_tokens/paper.pdf at v2 · rokosbasilisk/filler_tokens</a>: logit lens expts with filler tokens. Contribute to rokosbasilisk/filler_tokens development by creating an account on GitHub.</li><li><a href="https://github.com/huggingface/nanotron/blob/03d67f2103d5be0dc15ea6022a6cf16d6a633064/examples/moe/moe.py#L100">nanotron/examples/moe/moe.py at 03d67f2103d5be0dc15ea6022a6cf16d6a633064 · huggingface/nanotron</a>: Minimalistic large language model 3D-parallelism training - huggingface/nanotron
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1276646337525579816)** (131 messages🔥🔥): 

> - `GNN Research`
> - `Chinchilla Scaling Laws`
> - `ARC Research`
> - `SAE Papers`
> - `GPT4o` 


- **GNNs: Rewiring & Projection**: A member compared GNN research to the evolution of positional embeddings, starting with random walk embeddings and progressing to graph rewiring and projection.
   - This analogy suggests that future advancements may involve inferring positional embeddings from latent representations, similar to transformer architectures without positional embeddings.
- **Chinchilla Scaling Laws Under Fire**: A user inquired about a paper criticizing Chinchilla scaling laws, specifically questioning their accuracy.
   - There was a consensus that Redwood Research's GPT-4o and another YK server-based effort, focused on training a small LM with synthetic task datasets, are currently the leading contenders in the field.
- **ARC Research: Leading Methods & Community**: A discussion emerged regarding state-of-the-art methods for ARC research, with Redwood Research's GPT-4o and a YK-based project being highlighted.
   - Members pointed to a community project thread and an official Discord server for updates on ARC research, emphasizing the researchers' openness and willingness to share information.
- **The Quest for SAE Papers**: A user requested papers on SAE (Sparse AutoEncoder) techniques, particularly those employed by OpenAI and Anthropic for extracting interpretable features.
   - Several responses provided links to relevant papers, including one from OpenAI on concept extraction and another from transformer-circuits.pub focusing on scaling monosemanticity.
- **GPT4o: Architecture & Capabilities**: A discussion about the architecture behind GPT4o arose, with speculation that it might be a transformer.
   - While GPT4o's architecture remains undisclosed, insiders suggest it's a VLM (Vision-Language Model) with limited cross-domain transfer, implying no revolutionary advancements beyond existing open-source models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/NousResearch/status/1828121648383566270">Tweet from Nous Research (@NousResearch)</a>: What if you could use all the computing power in the world to train a shared, open source AI model?  Preliminary report: https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://arxiv.org/abs/2408.12528">Show-o: One Single Transformer to Unify Multimodal Understanding and Generation</a>: We present a unified transformer, i.e., Show-o, that unifies multimodal understanding and generation. Unlike fully autoregressive models, Show-o unifies autoregressive and (discrete) diffusion modelin...</li><li><a href="https://arxiv.org/abs/2408.09624">Attention is a smoothed cubic spline</a>: We highlight a perhaps important but hitherto unobserved insight: The attention module in a transformer is a smoothed cubic spline. Viewed in this manner, this mysterious but critical component of a t...</li><li><a href="https://x.com/XingyouSong/status/1826554454084333723">Tweet from Richard Song (@XingyouSong)</a>: How does Google optimize its research and systems? We’ve revealed the secrets behind the Vizier Gaussian Process Bandit algorithm, the black-box optimizer that’s been run millions of times!   Paper: h...</li><li><a href="https://openreview.net/group?id=ICML.cc/2024/Workshop/MI">ICML 2024 Workshop MI</a>: Welcome to the OpenReview homepage for ICML 2024 Workshop MI</li><li><a href="https://arxiv.org/abs/2404.19737">Better &amp; Faster Large Language Models via Multi-token Prediction</a>: Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in h...</li><li><a href="https://arxiv.org/abs/2108.08481">Neural Operator: Learning Maps Between Function Spaces</a>: The classical development of neural networks has primarily focused on learning mappings between finite dimensional Euclidean spaces or finite sets. We propose a generalization of neural networks to le...</li><li><a href="https://zongyi-li.github.io/blog/2020/fourier-pde/">Zongyi Li | Fourier Neural Operator</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=hm2IJSKcYvo">Unveiling of Moshi: the first voice-enabled AI openly accessible to all.</a>: In just 6 months, with a team of 8, the Kyutai research lab developed from scratch an AI model with unprecedented vocal capabilities called Moshi.This new ty...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1277612867314778133)** (2 messages): 

> - `Scaling Laws`
> - `Model Scaling Limits` 


- **Scaling Laws Paper: Are There Limits?**: A member requested papers that show cases where scaling laws don't work, for any specific property.
   - Another member provided a link to a paper titled [Are Scaling Laws Failing?](https://arxiv.org/abs/2306.09479) by a team of researchers from Google AI, suggesting it may be a relevant resource.
- **Scaling Laws Don't Work?**: A member had a discussion about scaling laws and how they don't always work in practice.
   - They shared a few specific cases where the scaling law would not work, such as in situations where there is a lack of data or computational resources. 



**Link mentioned**: <a href="https://arxiv.org/abs/2306.09479">Inverse Scaling: When Bigger Isn&#39;t Better</a>: Work on scaling laws has found that large language models (LMs) show predictable improvements to overall loss with increased scale (model size, training data, and compute). Here, we present evidence f...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1276631318553432216)** (17 messages🔥): 

> - `OpenAI ChatGPT4o and Anthropic external APIs multiple choice eval`
> - `llama 3.1 eval`
> - `lm-eval library output_type`
> - `lm-eval library doc_to_choice`
> - `lm-eval library chat_template` 


- **Evaluating Multiple Choice with ChatGPT4o/Anthropic**: A user asked if anyone had successfully used ChatGPT4o or Anthropic external APIs to evaluate multiple-choice questions, specifically in the context of OpenAI's 'multiple_choice' output_type.
   - The discussion explored the possibility of using the 'generate_until' output_type with OpenAI and Anthropic, as they do not provide loglikelihoods for 'multiple_choice'.
- **Llama 3.1 Evaluation**: A user mentioned the [Meta-Llama 3.1 evals](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-Instruct-evals/viewer/Meta-Llama-3.1-8B-Instruct-evals__mmlu__details) dataset as a reference point for multiple-choice evaluations, specifically highlighting the adapted 'mmlu' and 'arc_challenge' tasks.
- **lm-eval Library's 'output_type' Setting**: A user questioned if changing the 'output_type' in their YAML configuration file to 'generate_until' would be a viable solution for OpenAI and Anthropic, considering their lack of loglikelihoods for 'multiple_choice'.
   - The user proposed incorporating their choice responses within the 'doc_to_text' portion of their YAML configuration, mirroring the approach taken in the Llama 3.1 evals.
- **lm-eval Library's 'chat_template' Setting**: A user inquired about the use of the 'chat_template' parameter in the lm-eval library, specifically when working with instruct models and attempting to evaluate multiple-choice tasks.
   - It was suggested that setting 'chat_template=None' could resolve potential issues encountered with the default 'chat_template' setting.
- **lm-eval Library's Default Temperature Setting**: The discussion delved into the default temperature setting within the lm-eval library for OpenAI models, with the consensus being that it is typically set to 0.
   - This practice aligns with the common convention of using temperature 0 for most tasks within the lm-eval framework, often explicitly set in YAML configurations for certain benchmarks.



**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/aab42ba836b4af28cc1c5c1e697ea334c6ea7ced/lm_eval/evaluator.py#L295).">lm-evaluation-harness/lm_eval/evaluator.py at aab42ba836b4af28cc1c5c1e697ea334c6ea7ced · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1277690840617058305)** (3 messages): 

> - `GPU Utilization Measurement`
> - `NVIDIA-SMI Overreporting`
> - `DCGM-Exporter`
> - `PyNVML`
> - `TFLOPs and HFU/MFU Logging` 


- **Nvidia-SMI Overreports GPU Utilization**: A member inquired about the best tool to measure GPU utilization during training runs, noting that [NVIDIA-SMI often overreports utilization](https://arthurchiao.art/blog/understanding-gpu-performance/#25-two-metric-sources-nvml--dcgm).
   - They mentioned using **NVIDIA-SMI power util** as a simple and stupid option, but only if you trust the cooling system.
- **Accurate GPU Utilization with TFLOPs and HFU/MFU Logging**: A member recommended logging **TFLOPs** or **HFU/MFU** every iteration to **WandB** or the console as a more accurate approach.
   - They explained this involves adding the calculation to the logger and accurately determining the model's **FLOPs per iteration**, with a link to the [EleutherAI cookbook](https://github.com/EleutherAI/cookbook) for guidance.
- **PyTorch Profiler for GPU Utilization**: The [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html) provides accurate GPU utilization and indicates tensor core occupancy.
   - However, it introduces overhead, so the member suggested profiling only around 10 iterations at the beginning of each major run, hoping they are representative of the overall performance.



**Link mentioned**: <a href="https://github.com/EleutherAI/cookbook">GitHub - EleutherAI/cookbook: Deep learning for dummies. All the practical details and useful utilities that go into working with real models.</a>: Deep learning for dummies. All the practical details and useful utilities that go into working with real models. - EleutherAI/cookbook

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1276622907237339218)** (54 messages🔥): 

> - `BERTopic`
> - `OpenAI's sidebars`
> - `GPTs Agents`
> - `OpenEmpathic project`
> - `Model Merging` 


- **Hermes 2.5 beats Hermes 2**: After adding [code instruction examples](https://link.to.examples), **Hermes 2.5** appears to perform better than **Hermes 2** in various benchmarks.
   - Hermes 2 scored a **34.5** on the MMLU benchmark whereas Hermes 2.5 scored **52.3**.
- **Mistral has struggles expanding beyond 8k**: Members stated that **Mistral** cannot be extended beyond 8k without continued pretraining and [this is a known issue](https://link.to.issue).
   - They pointed to further work on *mergekit* and *frankenMoE finetuning* for the next frontiers in performance.
- **BERTopic Usage Discussion**: A member discussed **BERTopic**, a tool for easily interpretable topic modeling.
   - They shared their project on [visualizing datasets](https://github.com/YoungPhlo/visualizing-datasets-ai-engineer-fall-2023) and found that **BERTopic** is an end-to-end implementation. They also recalled asking if anyone was using it before for clustering.
- **Discussion on Model Merging Tactics**: A member suggested applying the difference between **UltraChat** and base **Mistral** to **Mistral-Yarn** as a potential merging tactic.
   - Others expressed skepticism, but this member remained optimistic, citing successful past attempts at what they termed "cursed model merging".
- **Open Empathic Project Plea for Assistance**: A member appealed for help in expanding the categories of the **Open Empathic** project, particularly at the lower end.
   - They shared a [YouTube video on the Open Empathic Launch & Tutorial](https://youtu.be/GZqYr8_Q7DE) that guides users to contribute their preferred movie scenes from YouTube videos, as well as a link to the [OpenEmpathic project itself](https://dct.openempathic.ai/). 


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/XingyouSong/status/1826554454084333723">Tweet from Richard Song (@XingyouSong)</a>: How does Google optimize its research and systems? We’ve revealed the secrets behind the Vizier Gaussian Process Bandit algorithm, the black-box optimizer that’s been run millions of times!   Paper: h...</li><li><a href="https://x.com/MikeShou1/status/1826854070877126720">Tweet from Mike Shou (@MikeShou1)</a>: When it comes to video keyframe generation, show-o is still temporally auto-regressive while full attention in space. Also when generating keyframe, we can generate interleaved textual description, na...</li><li><a href="https://x.com/romainhuet">Tweet from undefined</a>: no description found</li><li><a href="https://trainy.ai/blog/gpu-utilization-misleading">GPU Utilization is a Misleading Metric</a>: Most ML teams use GPU Utilization as their main performance metric, but we found this can be quite misleading.</li><li><a href="https://x.com/swyx/status/1827099985944703408">Tweet from swyx 🇸🇬 (@swyx)</a>: wtf, @mikeshou1&#39;s 1B omnimodel seems to beat @ArmenAgha&#39;s 34B Chameleon in VQA  AND is competitive with SDXL etc image gen models  AND can generate keyframes for video gen (eg Sora)  all in th...</li><li><a href="https://cohere.com/blog/combing-for-insight-in-10-000-hacker-news-posts-with-text-clustering">Combing For Insight in 10,000 Hacker News Posts With Text Clustering</a>: Hacker News is one of the leading online communities to discuss software and startup topics. I’ve frequented the site for over ten years and constantly admire the quality of its signal vs. noise ratio...</li><li><a href="https://x.com/amir/status/1827007117838192699?s=46">Tweet from Amir Efrati (@amir)</a>: ~Drama~ at AI agent startup Holistic (&#34;H&#34;) that recently raised $220M seed round: 3 of its 5 founders are out.   The departing founders were previously longtime Google DeepMind researchers.  h...</li><li><a href="https://x.com/vikhyatk/status/1827190823689335115">Tweet from vik (@vikhyatk)</a>: if you didn’t cancel your claude subscription today you deserve the world you’re about to get</li><li><a href="https://github.com/paul-gauthier/aider/releases/tag/v0.52.0?utm_source=ainews&utm_medium=email">Release Aider v0.52.0 · paul-gauthier/aider</a>: Aider now offers to run shell commands:  Launch a browser to view updated html/css/js. Install new dependencies. Run DB migrations. Run the program to exercise changes. Run new test cases.   /read ...</li><li><a href="https://share.snipd.com/episode/bc246320-a849-4718-9fc8-be0f4290aaf0">20VC: Chips, Models or Applications; Where is the Value in AI | Is Compute the Answer to All Model Performance Questions | Why Open AI Shelved AGI &amp; Is There Any Value in Models with OpenAI Price Dumping with Aidan, Gomez, Co-Founder @ Cohere</a>: 20VC: Chips, Models or Applications; Where is the Value in AI | Is Compute the Answer to All Model Performance Questions | Why Open AI Shelved AGI &amp; Is There An…</li><li><a href="https://x.com/drjimfan/status/1827116592951652823?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Jim Fan (@DrJimFan)</a>: The transformer-land and diffusion-land have been separate for too long. There were many attempts to unify before, but they lose simplicity and elegance. Time for a transfusion🩸to revitalize the merg...</li><li><a href="https://youtu.be/L0kBWyziFlc">Disrupting the $15 Trillion Construction Industry with Autonomous Agents: Dr. Sarah Buchner</a>: Dr. Sarah Buchner, Founder &amp; CEO of Trunk Tools, envisions a future for construction where an army of AI agents works on behalf of our users. We are currentl...</li><li><a href="https://github.com/YoungPhlo/visualizing-datasets-ai-engineer-fall-2023">GitHub - YoungPhlo/visualizing-datasets-ai-engineer-fall-2023: Visualizing Datasets: Unlocks a visual perspective on your text data before using it in downstream tasks</a>: Visualizing Datasets: Unlocks a visual perspective on your text data before using it in downstream tasks - YoungPhlo/visualizing-datasets-ai-engineer-fall-2023</li><li><a href="https://github.com/MaartenGr/BERTopic">GitHub - MaartenGr/BERTopic: Leveraging BERT and c-TF-IDF to create easily interpretable topics.</a>: Leveraging BERT and c-TF-IDF to create easily interpretable topics.  - GitHub - MaartenGr/BERTopic: Leveraging BERT and c-TF-IDF to create easily interpretable topics.
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1276631012599922699)** (1 messages): 

> - `AI Engineer London Meetup`
> - `AI Engineer World's Fair`
> - `Meetup speakers` 


- **AI Engineer London Meetup announced**: A new AI Engineer Meetup has been announced in London, hosted by <@dctanner>, inspired by @swyx's AI Engineer World's Fair.
   - The first meetup will take place on the evening of September 12th and will feature four speakers: @maximelabonne, @roviosc, @BruverisMartins and Chris Bull.
- **Registration Link for Meetup**: The announcement includes a registration link for the AI Engineer London Meetup, which is expected to be a popular event.



**Link mentioned**: <a href="https://x.com/dctanner/status/1827071893448618453?s=46">Tweet from Damien C. Tanner (@dctanner)</a>: We&#39;re brining a slice of @swyx&#39;s AI Engineer World&#39;s Fair to London!  Evening of 12 September is the first AI Engineer London Meetup.   Hear from 4 amazing speakers: @maximelabonne, @rovio...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1276630411149049998)** (78 messages🔥🔥): 

> - `Structured Outputs`
> - `LLM Deduplication`
> - `TaxonomySynthesis`
> - `BERTopic`
> - `GPT Researcher` 


- **Structured Outputs are different with grammar constrained sampling**: A member discussed the difference in using [Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) versus [Function Calling](https://platform.openai.com/docs/guides/functions/how-to-use-functions) in the context of using [Grammar Constrained Sampling](https://www.microsoft.com/en-us/research/publication/grammar-constrained-sampling-for-generative-language-models/).
- **Deduplication and Topic Similarity**: Members discussed how to handle duplicate or extremely similar topics generated by LLMs, particularly when dealing with thousands of topics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://maartengr.github.io/BERTopic/">BERTopic</a>: no description found</li><li><a href="https://medium.com/@juanc.olamendy/revolutionizing-retrieval-the-mastering-hypothetical-document-embeddings-hyde-b1fc06b9a6cc">Revolutionizing Retrieval: The Mastering Hypothetical Document Embeddings (HyDE)</a>: In the dynamic universe of Retrieval-Augmented Generation (RAG) systems, a perplexing question often surfaces: How does one construct an…</li><li><a href="https://maartengr.github.io/BERTopic/algorithm/algorithm.html#5-topic-representation">The Algorithm - BERTopic</a>: no description found</li><li><a href="https://www.figma.com/board/J19T0RN1Hvi1ajDlUtIvOc/Generative-Classifier?node-id=2-1801&t=Km2ND86IeNkD92WJ-1">Figma</a>: Created with FigJam</li><li><a href="https://github.com/danielgross/embedland/blob/main/bench.py#L281">embedland/bench.py at main · danielgross/embedland</a>: A collection of text embedding experiments. Contribute to danielgross/embedland development by creating an account on GitHub.</li><li><a href="https://github.com/CakeCrusher/TaxonomySynthesis">GitHub - CakeCrusher/TaxonomySynthesis: An AI-driven framework for synthesizing adaptive taxonomies, enabling automated data categorization and classification within dynamic hierarchical structures.</a>: An AI-driven framework for synthesizing adaptive taxonomies, enabling automated data categorization and classification within dynamic hierarchical structures. - CakeCrusher/TaxonomySynthesis</li><li><a href="https://github.com/assafelovic/gpt-researcher">GitHub - assafelovic/gpt-researcher: LLM based autonomous agent that does online comprehensive research on any given topic</a>: LLM based autonomous agent that does online comprehensive research on any given topic - assafelovic/gpt-researcher</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations. - stanford-oval/storm</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/storm/storm.ipynb">langgraph/examples/storm/storm.ipynb at main · langchain-ai/langgraph</a>: Build resilient language agents as graphs. Contribute to langchain-ai/langgraph development by creating an account on GitHub.</li><li><a href="https://feedly.com/">Feedly: Track the topics and trends that matter to you</a>: Market-leading solution for monitoring topics that matter</li><li><a href="https://exa.ai/">Exa</a>: The Exa API retrieves the best, realtime data from the web to complement your AI</li><li><a href="https://notebooklm.google.com/">no title found</a>: no description found</li><li><a href="https://app.tavily.com/chat">Tavily AI</a>: no description found</li><li><a href="https://consensus.app/">Consensus AI-powered Academic Search Engine</a>: Consensus is a new breed of academic search engine, powered by AI, grounded in science. Find the best papers while getting instant insights and topic synthesis.</li><li><a href="https://elicit.com/">Elicit: The AI Research Assistant</a>: Use AI to search, summarize, extract data from, and chat with over 125 million papers. Used by over 2 million researchers in academia and industry.</li><li><a href="https://storm.genie.stanford.edu/">no title found</a>: no description found</li><li><a href="https://gptr.dev/">GPT Researcher - Official Page</a>: Your AI assistant for rapid insights and in-depth research on any given task</li><li><a href="https://langchain-ai.github.io/langgraph/">🦜🕸️LangGraph</a>: no description found
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/)** (1 messages): 

georgehotz: buy your tinybox here https://tinycorp.myshopify.com/
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1277323878489653261)** (78 messages🔥🔥): 

> - `E-graphs`
> - `Tinygrad`
> - `Tinybox`
> - `AMD`
> - `BERT` 


- **E-graph Performance Concerns**: A member shared their view that e-graph rewrites are behind current SAT solvers in handling large search spaces.
- **Tinybox Sales Launch!**: The Tinybox factory is now at full power, with sales opening shortly to the public.
- **Tinybox: A Tinygrad Discussion**: A member inquired about the feasibility of using AMD GPUs with the Tinybox, citing AMD's recent acquisition of Silo AI and their success in training an LLM with AMD hardware.
- **Tinygrad vs Torch for BERT Pre-Training**: A member expressed interest in partnering with Tinygrad to pre-train a large BERT model, offering their computing resources.
- **Tinybox Shipping and Availability**: The Tinybox is currently sold out, with limited production capacity of about 4 units per day, and a backlog of 60 more units.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tinycorp.myshopify.com">tiny shop</a>: tiny shop</li><li><a href="https://en.wikipedia.org/wiki/Freight_forwarder">Freight forwarder - Wikipedia</a>: no description found</li><li><a href="https://tinycorp.myshopify.com/products/tinybox-red">tinybox red edition</a>: Payment must be completed within 5 days of order confirmation to guarantee your order. Contiguous US Shipping Only. Email support@tinygrad.org for Canada.  Payment Method: Bank Transfer/Wire Docs can ...</li><li><a href="https://tinycorp.myshopify.com/products/tinybox-green">tinybox green edition</a>: Payment must be completed within 5 days of order confirmation to guarantee your order. Contiguous US Shipping Only. Email support@tinygrad.org for Canada.  Payment Method: Bank Transfer/Wire Docs can ...</li><li><a href="https://x.com/__tinygrad__/status/1828140019577704652">Tweet from the tiny corp (@__tinygrad__)</a>: The factory is at full power! If you have a tinybox preorder and somehow haven&#39;t been contacted, reach out to support@tinygrad.org  Sales will open to the public shortly.
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1276642882127794187)** (13 messages🔥): 

> - `GPU=1 and OpenCL`
> - `Tensor Loading`
> - `Training Speed and Data Types`
> - `Performance Differences Between dtypes.half and dtypes.float`
> - `RoPE Approximation and Model Matching` 


- **GPU=1 Not Using GPU**: A user with a 3060 GPU was wondering how to see what calls are made to it when `GPU=1` is used.
   - Another user explained that `GPU=1` refers to OpenCL, and that the user likely has `pocl` installed, which only utilizes the CPU.
- **Loading a Saved Tensor**: A user asked how to load a tensor that was previously saved using `nn.state.safe_save`. 
   - The code snippet provided demonstrated saving the tensor to a `test.safetensor` file using `nn.state.safe_save({'t':t}, "test.safetensor")`.
- **Faster Training with Removed Cast**: A user reported a 25% increase in training speed (GFLOPS) after removing the `.cast(dtypes.default_float)` call in the preprocessing of the `beautiful_cifar` example.
   - The user noted that the `X.dtype` was now `dtype.float` instead of `dtype.half` before entering the model.
- **Performance Differences between dtypes.half and dtypes.float**: A user observed that `._data()` and `.realize()` are slower for `dtypes.half` compared to `dtypes.float`.
   - This was attributed to faster lowering but slower execution for `dtypes.half`.
- **RoPE Approximation and Model Matching**: A user questioned whether the approximation of sin/cos using RoPE inherently means models will not perfectly match reference implementations.
   - This question was left unanswered.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.tinygrad.org/nn/#tinygrad.nn.state.safe_load">nn (Neural Networks) - tinygrad docs</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/blob/1b4ad982e58c88c858e731b96a6ed3f2eef1b6b7/examples/beautiful_cifar.py#L91">tinygrad/examples/beautiful_cifar.py at 1b4ad982e58c88c858e731b96a6ed3f2eef1b6b7 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1276676918690975885)** (37 messages🔥): 

> - `Command-R pricing`
> - `Durov's trial`
> - `Command-R free trial` 


- **Command-R Model Update**: A new Command-R model has been released, but there's no official announcement yet.
   - Questions regarding pricing, context window, fine-tuning, and other features remain unanswered.
- **Durov's Controversial French Citizenship**: Pavel Durov, founder of Telegram, obtained French citizenship and is currently facing trial in France.
   - Some speculate that he's deliberately seeking imprisonment, while others believe it's a strategic move to gain international attention.
- **Command-R Free Trial Limits**: Command-R is free to use until users reach their rate limits on trial keys.
   - This was a highly requested change by users and was announced via email recently.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/0001-gif-17282391190974969363">0001 GIF - 0001 - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/evannie-gif-13360603728069224276">Evannie GIF - Evannie - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/1vnzh/status/1827690558799695949">Tweet from Ivan Zhang (@1vnzh)</a>: pause AI to play black myth wukong</li><li><a href="https://x.com/MyLordBebo/status/1827585606437765568">Tweet from Lord Bebo (@MyLordBebo)</a>: Durov: Getting French citizenship just to get into prison?  It seems like the guy in an interview with Tucker is talking about how NATO intelligence services are pressuring him to gain access to chats...</li><li><a href="https://x.com/SouthDallasFood/status/1827443354948579745">Tweet from South Dallas Foodie (@SouthDallasFood)</a>: Big Parma gatekeeping fr
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1276690811475460196)** (9 messages🔥): 

> - `Cohere free trial`
> - `Invite Feature Bug`
> - `Fine Tuning Cohere Embeddings`
> - `Rasa Chatbot` 


- **Cohere Free Trial for Rasa Chatbot**: A user from Bangladesh inquired about using Cohere's free trial to build a chatbot using Rasa.
   - They previously attempted using OpenAI, but were deterred by a $5 fee, and are seeking a similar free trial option with Cohere.
- **Invite Feature Bug**: A user reported a bug with the invite member feature, stating they were unable to join a team despite having a Gmail account.
   - They requested assistance in resolving this issue, as they were unable to join a team they were invited to.
- **Fine-tuning Cohere Embeddings Without API Key**: A user inquired about fine-tuning Cohere embed-multilingual-v3.0 using a domain dataset without a Cohere API key.
   - A response clarified that API keys are always needed to communicate with Cohere's endpoints for fine-tuning.
- **Trail Keys for Free Access**: A user asked about the availability of free trial keys for accessing Cohere services.
   - They were informed that trial keys are completely free with rate limits, and users can later upgrade for enhanced limits when ready for production.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1276680717727694878)** (32 messages🔥): 

> - `Cohere API Rate Limits`
> - `Multiple API Keys`
> - `Async Requests`
> - `Rerank 3 Pricing`
> - `Token Limits` 


- **Cohere API Rate Limits Update**: A user reported experiencing "too many requests" errors when using multiple API keys simultaneously, even while following the 10,000 requests/minute limit mentioned in the documentation.
   - The Cohere team confirmed a recent change to the rate limits, with each user now limited to 1,000 calls per minute across all organization API keys, explaining that it's per user, not per API key.
- **Async Requests and Rate Limits**: Users were experiencing "too many requests" errors when making asynchronous calls with multiple API keys, even with each key staying under the 1000/minute limit.
   - The Cohere team clarified that the rate limit is now per user, not per API key, meaning the 1,000 calls per minute limit applies to all API keys used by a single user within an organization.
- **Rerank 3 Pricing Breakdown**: A user inquired about the cost of using the Rerank 3 model, specifically asking if the $2 for 1,000 searches refers to 1,000 API calls or something else.
   - The Cohere team clarified that each query with up to 100 documents to be reranked counts as one search, meaning 1,000 searches can process up to 409,600,000 tokens, given that each document can contain up to 4,096 tokens.



**Link mentioned**: <a href="https://cohere.com/pricing">Pricing</a>: Access our models directly through our API to create scalable production workloads.   

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1276648997401923605)** (5 messages): 

> - `LlamaIndex`
> - `Create Llama`
> - `GraphRAG`
> - `Data Silos`
> - `LLMs for newsletters` 


- **Create Llama gets a structured extraction template**: The [Create Llama](https://t.co/G95ReAuRS6) tool is now featuring a structured extraction template.
- **GraphRAG tutorial series begins**: A step-by-step tutorial series on building [GraphRAG](https://t.co/7fLocjRvdN) has started.
   - The first video focuses on implementing core components using an in-memory implementation and covers extracting entities and relationships with LLMs.
- **Addressing data silos in enterprise LLM development**: Enterprise LLM development faces challenges with data silos, where each new data source requires separate authentication management and knowledge is scattered across teams.
   - LlamaIndex is exploring solutions to this issue.
- **LLMs for newsletter automation**: The LlamaIndex newsletter, which summarizes weekly tweets, was previously time-consuming to create.
   - LLMs are now used to automate this process, demonstrating the power of LLMs in content creation.
- **RAG-a-thon hackathon announced**: The second [RAG-a-thon](https://t.co/IFvyW5QB6r), in partnership with Pinecone, is scheduled for October 11-13 in Palo Alto.
   - This hackathon offers over $7k in cash prizes and will be held at the 500 Global VC offices.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1276619666759876752)** (71 messages🔥🔥): 

> - `tqdm`
> - `LlamaIndex API changes`
> - `PineconeVectorStore`
> - `LlamaIndex and OpenAI models`
> - `Updating VectorStoreIndex` 


- **tqdm -  Python's Must-Have Progress Bar**: A member expressed excitement about the growing popularity of `tqdm` for Python progress bars.
   - They stated it's *basically the only* progress bar they use.
- **LlamaIndex  API Change: ServiceContext Deprecated**: A user expressed frustration over the removal of the `ServiceContext` in LlamaIndex version 0.11.1.
   - The user highlighted that they were not warned about this change and that many users might leave LlamaIndex because of it.
- **PineconeVectorStore Compatibility Issues**: A member inquired about compatibility between Llamaindex and PineconeVectorStore after encountering an error.
   - The solution was simply updating the `llama-index-vector-stores-pinecone` package.
- **Using Latest OpenAI Model in LlamaIndex**: A user asked about setting the LlamaIndex query engine to use the latest OpenAI model.
   - The user was directed to set the default LLM globally using `Settings.llm = OpenAI(model="gpt-4o-mini")`.
- **Updating VectorStoreIndex With New Documents**: A user asked how to update a VectorStoreIndex by adding documents with embeddings.
   - The suggestion was to use the `insert_nodes` method, ensuring that the documents are properly parsed and embedded.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://",">no title found</a>: no description found</li><li><a href="https://",">no title found</a>: no description found</li><li><a href="https://discordapp.com/channels/1059199217496772688/1277285749078753325">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://postgres.new/">Postgres Sandbox</a>: In-browser Postgres sandbox with AI assistance</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/">Migrating from ServiceContext to Settings - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/">LlamaIndex - LlamaIndex</a>: no description found</li><li><a href="https://postgresml.org/docs/open-source/pgml/guides/embeddings/in-database-generation">In-database Embedding Generation – PostgresML</a>: Train and deploy models to make online predictions using only SQL, with an open source Postgres extension.</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/#document-management">Ingestion Pipeline - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/79518c4cc39981140b2e87c9a701b09d74d47e9a/llama-index-core/llama_index/core/base/embeddings/base.py#L37">llama_index/llama-index-core/llama_index/core/base/embeddings/base.py at 79518c4cc39981140b2e87c9a701b09d74d47e9a · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1277385972711030900)** (60 messages🔥🔥): 

> - `Eager vs compile discrepancy`
> - `Triton RNG`
> - `In-place operations and Torch.compile`
> - `KV cache allocation`
> - `Cudagraphs and memory consumption` 


- **Compiled Function Outputs Differ from Eager Mode**: A member asked why a compiled function might produce different outputs compared to its non-compiled counterpart, even with the same seed.
   - This discrepancy stems from the difference between Triton's RNG (used in compiled code) and PyTorch's RNG (used in eager mode), as well as potential in-place operations that are not well-suited for compilation.
- **In-place Operations in Torch.compile**: It was discussed that in-place operations, such as `scatter_`, generally perform poorly when used with `torch.compile`.
   - This is because compiled code may handle in-place operations differently, leading to increased memory consumption and potentially different output.
- **Cudagraphs and Memory**: The use of cudagraphs was suggested for debugging purposes.
   - However, it was also mentioned that cudagraphs can sometimes lead to higher memory consumption due to pre-allocating buffers, which may be undesirable in some cases.
- **FP16 for Inference**: The use of FP16 instead of FP32 for inference was suggested as a way to reduce memory usage, particularly for hardware that doesn't support BF16.
   - This approach helped resolve the out-of-memory issue, but the difference in output between compiled and non-compiled versions persisted.
- **Numerical Differences in Compiled Kernels**: It was speculated that the remaining output difference between compiled and non-compiled code could be due to numerical differences in the compiled kernels.
   - This suggests that even when memory usage is optimized, there might be slight variations in the computations performed by the compiled code.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/pytorch-labs/gpt-fast/blob/bc04265df30c7b927d38f34198e0e33a63cb893e/model.py#L80-L89">gpt-fast/model.py at bc04265df30c7b927d38f34198e0e33a63cb893e · pytorch-labs/gpt-fast</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - pytorch-labs/gpt-fast</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L401">pytorch/torch/_inductor/config.py at main · pytorch/pytorch</a>: Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch</li><li><a href="https://github.com/pytorch-labs/gpt-fast/blob/bc04265df30c7b927d38f341">GitHub - pytorch-labs/gpt-fast at bc04265df30c7b927d38f34198e0e33a63cb893e</a>: Simple and efficient pytorch-native transformer text generation in &lt;1000 LOC of python. - GitHub - pytorch-labs/gpt-fast at bc04265df30c7b927d38f34198e0e33a63cb893e
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1276777519349239930)** (41 messages🔥): 

> - `LangChain Document Loading`
> - `LangChain Model Usage`
> - `LLMChain vs LCEL`
> - `LangChain with Postgres`
> - `GenAI in Data Science` 


- **LangChain Document Loading - Extracting Images**: The `extract_images=True` parameter in the `PyPDFLoader` class from LangChain's community package is used to extract images from a PDF document when loading it.
   - This allows you to process the images alongside the text content, potentially enabling image-based analysis or enriching the context for your LLM.
- **LLMChain vs LCEL**: The `LLMChain` class represents a chain where you provide the model and a prompt, while the `LCEL` (LangChain's Execution Language) offers more flexibility and control.
   - While both aim to chain components for complex tasks, `LLMChain` is generally optimized, whereas LCEL allows for greater customization and modularity, although the increased control might not always be necessary.
- **LangChain with Postgres - PostgresSaver Error**: A user is facing a `TypeError: tuple indices must be integers or slices, not str` error while working with `LangGraph` and a Postgres database using `PostgresSaver`.
   - The error is likely related to how the user is accessing tuple elements with a string index, but more information is needed to pinpoint the cause and provide a solution.
- **GenAI's Role in Data Science**: A user raised a question about the role of Generative AI (GenAI) in data science, specifically after the emergence of ChatGPT.
   - The user believes GenAI has a limited role in generating code and basic data pipelines, but others argue that data science is vital for building functional GenAI applications, highlighting the increasing overlap between these fields.
- **RAG with LangChain - Collaboration and Challenges**: A user is building a RAG (Retrieval-Augmented Generation) chatbot for documentation related to popular packages like LangChain and LangGraph, seeking collaborators for this project.
   - The user is finding the scraping and RAG components challenging, and is willing to collaborate with others who are interested in these topics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/issues/24987>),">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.</li><li><a href="https://github.com/langchain-ai/langchain/issues/23661>))">Issues · langchain-ai/langchain</a>: 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1276783342792081491)** (3 messages): 

> - `AI agents`
> - `LangGraph framework`
> - `Retrieval Agents`
> - `ParDocs`
> - `RAG` 


- **AI Agents: The Future of AI**: AI agents are increasingly important for AI's growth, as they can mimic human-like attributes, interacting, reasoning, and making decisions to achieve goals with autonomy.
   - This article explores how to build AI agents using [LlamaIndex](https://community.analyticsvidhya.com/c/generative-ai-tech-discussion/how-to-use-llamaindex) and MonsterAPI, which provide tools for developing agents and accessing LLM APIs.
- **LangGraph Framework for Retrieval Agents**: The LangGraph framework is used to build retrieval agents, which are AI agents that utilize external knowledge bases to enhance their responses.
   - This article will demonstrate how to build a powerful retrieval agent using LangGraph, showcasing its capabilities in retrieving and processing information from external sources.
- **ParDocs: Parsing Unstructured Documents**: ParDocs is a tool designed to parse unstructured documents like PDFs and DOCX files into structured JSON.
   - This structured JSON makes it easier to work with vector databases and get more accurate outputs from AI agents.
- **r/RAG Subreddit for Retrieval-Augmented Generation**: The r/RAG subreddit was created to provide a community for sharing knowledge and tools related to Retrieval-Augmented Generation (RAG).
   - The creator of ParDocs is excited about the potential for collaboration and learning within this community, pushing the boundaries of what's possible with RAG.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/ai-artistry/building-a-retrieval-agent-with-langgraph-and-exa-9baf33e89510">Building a Retrieval Agent with LangGraph and Exa</a>: Ankush k Singal</li><li><a href="https://www.pardocs.com).">no title found</a>: no description found</li><li><a href="https://www.analyticsvidhya.com/blog/2024/08/how-to-build-an-ai-agent-using-llama-index-and-monsterapi/">How to Build an AI Agent using Llama Index and MonsterAPI?</a>: Discover how to build AI agents with LlamaIndex and MonsterAPI, from architecture to practical applications in automation and more.
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1277057611975626834)** (15 messages🔥): 

> - `GPT-4 fine-tuning`
> - `Mistral`
> - `LLM Benchmarks`
> - `lm-eval-harness`
> - `Glianorex Benchmark` 


- **GPT-4 Fine-tuning: A Loli-Sized Experiment**: A member suggested that fine-tuning GPT-4 is "kind of shit" compared to Mistral, though they used less data.
- **Framework for Creating Benchmarks Quickly**: A member asked if there are any frameworks for creating benchmarks quickly.
   - Another member pointed to the *lm-eval-harness* framework, which is easy to add tasks to.
- **Creating Benchmarks - A Conversation**: A member asked what it means to create benchmarks and what are some approaches.
   - They mentioned the *lm-eval-harness* framework and their own research about generating questions for benchmarking, which is detailed in a recent paper and available on GitHub.
- **The Glianorex Benchmark: A Medical Fiction**: A member shared a paper about their research on evaluating the effectiveness of MCQs for assessing LLM performance in the medical field.
   - They created a fictional medical benchmark centered around a non-existent gland, *Glianorex*, to isolate the knowledge of the LLM from its test-taking abilities.
- **Adding Your Own Benchmark to *lm-eval-harness***: A member asked if it is easy to add your own benchmark to *lm-eval-harness*.
   - The response confirmed that it is indeed "very easy" to add benchmarks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.02394">Multiple Choice Questions and Large Languages Models: A Case Study with Fictional Medical Data</a>: Large Language Models (LLMs) like ChatGPT demonstrate significant potential in the medical field, often evaluated using multiple-choice questions (MCQs) similar to those found on the USMLE. Despite th...</li><li><a href="https://github.com/maximegmd/glianorex-gen">GitHub - maximegmd/glianorex-gen</a>: Contribute to maximegmd/glianorex-gen development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1276637711985410199)** (17 messages🔥): 

> - `LIGER`
> - `LLM Training Efficiency`
> - `Single GPU Training`
> - `Training Time` 


- **LIGER: 25% VRAM & 33% Training Time Savings**: **LIGER** is a new kernel for **LLM training** that has achieved impressive results: **25% VRAM** and **33% training time** savings.
   - A user expressed excitement, stating they were pouring champagne in celebration.
- **LIGER Not Designed for Single GPU**: A user asked if **LIGER** offers any improvement for **single GPU training**.
   - Another user confirmed that there should be improvement, suggesting it's not specifically designed for single GPU scenarios.
- **Training Time and Checkpoint Resuming**: A user inquired about the reported training time of **139 hours** with only **30% completion** after **one hour**. 
   - Another user suggested resuming from a **checkpoint** as a possible explanation for the long training time.



**Link mentioned**: <a href="https://github.com/linkedin/Liger-Kernel">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1276679771631255703)** (5 messages): 

> - `Phi-3-medium-128k-instruct training config`
> - `training config for new tokens` 


- **Seeking Phi-3-medium-128k-instruct Training Config**: A user inquired about the availability of a training config for the **Phi-3-medium-128k-instruct** model.
   - They specifically asked if anyone has trained this model and could share their training configuration.
- **Clarifying Tokenizer Training Scope**: A user asked whether the `modules_to_save = ["lm_head", "embed_tokens"]` configuration trains the full tokenizer vocabulary or only newly added tokens.
   - They referenced a Discord message link for context.


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1277577357711904801)** (1 messages): 

> - `Data Curation`
> - `LLM Judge`
> - `Model Rating`
> - `Data Curation Setup` 


- **Data Curation - Model Rating System**: A user inquired about the process of data curation, specifically wondering if it involved prompting a model to give a rating, similar to the LLM-Judge system.
   - The user was interested in knowing if there was a specific setup or methodology used for this curation process.
- **LLM-Judge: An Example of Model Rating**: The user mentioned "LLM-Judge" as an example of a system that utilizes model ratings.
   - This system likely involves prompting models to provide assessments of other models, potentially based on various criteria.
- **Exploring Data Curation Methods**: The discussion centered around understanding the techniques and strategies employed in data curation.
   - The user's question hinted at the possibility of using model-based approaches for data rating, similar to LLM-Judge.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1276620003424342067)** (16 messages🔥): 

> - `Mojo JIT`
> - `Mojo Script Mode`
> - `Max/Mojo Development Pace`
> - `GPU Support`
> - `Community Meeting` 


- **Mojo's Jitting Behavior Explained**: A member asked about a potential `mojo jit` command, and another member explained that running `mojo main.mojo` (script mode) uses jitting, while `mojo build main.mojo` does a proper compile.
   - This explains why things like global variables work in one mode but not the other.
- **Community Concerns About Development Pace**: A member noted a perceived decrease in the pace of blog posts and releases for both Max and Mojo.
   - They asked if this was due to summer slowdown, upcoming releases, or unexpected issues.
- **GPU Support is a Top Priority**: Another member stated that the team seems to be pushing hard to get GPU support out, suggesting that the next major release might be aligned with this.
   - They also mentioned that the team might be looking to move Magic out of alpha with this release.
- **Upcoming Max+Mojo Community Meeting**: A member announced that the next Max+Mojo community meeting is scheduled for Monday, September 9th and encouraged interested parties to sign up to present.
   - They provided a link to the sign-up document and offered to help anyone interested in presenting at future meetings.
- **Dedicated Channel for Licensing Questions**: A new channel was announced for discussion and questions related to Max and Mojo licenses.
   - The announcement highlighted the team's desire to allow users to build freely on top of both Max and Mojo.



**Link mentioned**: <a href="https://modul.ar/community-meeting-doc">[Public] Mojo Community Meeting</a>: Mojo Community Meeting This doc link: https://modul.ar/community-meeting-doc  This is a public document; everybody is welcome to view and comment / suggest.  All meeting participants must adhere to th...

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1276622632325746688)** (18 messages🔥): 

> - `Mojo's memory usage`
> - `Modverse 42 release schedule`
> - `Mojo's struct parameters`
> - `Mojo's UnsafePointer` 


- **Mojo Memory Usage Measurement**: A member asked if there is a way to measure memory usage in Mojo, noting that the `benchmark` command only prints execution time.
   - Another member explained that the only good way to do this is to ask the OS, as Mojo's runtime doesn't have enough accounting capabilities, especially with external allocators or mmap things.
- **Modverse 42 Release Schedule**: A member inquired about the absence of the Modverse 42 release last week.
   - Another member responded that releases occur every 1-3 weeks depending on the amount of new projects and content, and that the `Weekly` tag will likely be removed or renamed.
- **Mojo's Struct Parameters Explained**: A member shared an example of using code within a struct but encountered an error: `cannot implicitly convert 'ClassStruct["x", "y", "add"]' value to 'ClassStruct[]'`. 
   - Another member explained that `ClassStruct` has `variadic parameters` and needs to be parameterized from the outer struct. Parameters need to be known before the program runs, so struct fields can be parameterized from struct parameters or a global `alias`.
- **Mojo's UnsafePointer and Optional**: A member asked what the value for an argument should be in a case where `self` is used but results in an error: `use of unknown declaration 'self'`.
   - Another member responded that in this example, Node doesn't actually have ownership over the reference inside of next. A parameter on the struct needs to delegate the ownership if references are used. They also suggested using `UnsafePointer` since Mojo doesn't have a safe, owning pointer.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/)** (1 messages): 

guidorice: not sure if anyone shared it: https://youtu.be/7TnkqfX84gI?si=sqHI4BLGOwneLarH 👍
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1276951510378614824)** (23 messages🔥): 

> - `OpenInterpreter Profiles`
> - `OpenInterpreter --vision on Windows`
> - `OpenInterpreter OS Mode`
> - `Prebuilt OpenInterpreter`
> - `OpenInterpreter Documentation` 


- **OpenInterpreter Custom Profile Paths**: A member asked if it's possible to set a custom path for OpenInterpreter profiles.
   - The developer responded that it's not currently possible but could be implemented as a setting in the future.
- **OpenInterpreter --vision on Windows**: A user inquired about the functionality of the `--vision` flag on Windows.
   - The response was that it should work on Windows, but for any issues, a specific channel for addressing them was suggested.
- **OpenInterpreter OS Mode for Browser Interactions**: A user wondered if OpenInterpreter can read and reply to browser posts.
   - It was confirmed that this functionality would require OS mode and could potentially involve searching and sending messages on Discord.
- **Prebuilt OpenInterpreter Availability**: A user asked for a prebuilt version of OpenInterpreter.
   - The developer informed them that preorders are currently closed due to high demand, and they'll have to wait until sales reopen.
- **OpenInterpreter Documentation Access**: A member inquired about referencing project documentation when running OpenInterpreter.
   - They explained they're working on merging helm charts and want to access the project documentation for assistance.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1277295416634904680)** (6 messages): 

> - `OpenInterpreter Brand Guidelines` 


- **Brand Guideline Doc Request**: A member inquired about a brand guideline document for the project, mentioning a previous conversation about it during an accessibility meeting.
- **Brand Guideline Doc Availability**: Another member confirmed that no brand guideline document is currently available.
- **Purpose of Brand Guideline Doc**: The inquiring member explained their role as an industrial designer and researcher and that they are interested in reviewing the document for their work.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1277043838367694911)** (4 messages): 

> - `Zed AI`
> - `Open Source`
> - `Cursor`
> - `AI Code Editor`
> - `Claude-3.5` 


- **Zed AI: Code with LLMs**: Zed AI offers a powerful interface for AI-assisted programming, letting you converse directly with AI in the editor to generate, transform, and analyze code.
   - It's open source and features a growing list of models, including support for Claude-3.5 and Ollama, with a new Anthropic API designed for fast text transformations available for free for the first month.
- **Zed AI vs Cursor**: A user questioned the benefits of Zed AI compared to Cursor, noting Zed's open source nature as a key difference.
   - The user expressed appreciation for the development of more open source AI coding options.
- **Zed AI Workflow Command**: Zed AI provides a `/workflow` command that suggests relevant sequential inline transformations, enhancing the development process through seamless collaboration with LLMs.
- **Zed AI:  Free Anthropic API**: Zed AI offers a new Anthropic API designed for fast text transformations, available for free for the first month.
- **Zed AI:  Integration into Workflow**: Zed AI allows you to converse with AI directly in the editor, providing a uniquely powerful interface for AI-assisted programming.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://zed.dev/ai">Zed - The editor for what&#x27;s next</a>: Zed is a high-performance, multiplayer code editor from the creators of Atom and Tree-sitter.</li><li><a href="https://www.youtube.com/watch?v=AbptudVb30Y">Zed AI: This is THE BEST Opensource AI Code Editor with FREE Claude-3.5 Sonnet &amp; Ollama Support</a>: Join this channel to get access to perks:https://www.youtube.com/@AICodeKing/joinIn this video, I&#39;ll be telling you about Zed AI which is a new AI based code...
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1277636526485799096)** (2 messages): 

> - `Apple's ML-Superposition Prompting` 


- **Apple's Superposition Prompting Project**: A member of the community expressed excitement and support for Apple's recent release of their **ML-Superposition Prompting** project, which is now available on Github.
   - The project, [available here](https://github.com/apple/ml-superposition-prompting), aims to develop and refine advanced techniques for prompting in machine learning.
- **No Further Discussion on Apple Project**: At present, the only discussion regarding Apple's ML-Superposition Prompting was the initial excitement over its release.



**Link mentioned**: <a href="https://github.com/apple/ml-superposition-prompting">GitHub - apple/ml-superposition-prompting</a>: Contribute to apple/ml-superposition-prompting development by creating an account on GitHub.

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1276652251766587554)** (18 messages🔥): 

> - `Typed outputs`
> - `DSPy's prompt length`
> - `Truncated outputs` 


- **New OpenAI Feature for Typed Outputs**: Several members discussed a new feature from OpenAI for typed outputs, with several links to different projects like **Outlines**, **Guidance**, **SGLang**, **Guardrails**, and **Instructor**.
   - The conversation specifically focused on how to handle the validation of **structured outputs**, often in JSON format,  and  the various libraries available to assist with it.
- **DSPy Error: Too Many Retries**: A member reported a `ValueError` while using `typed predictors` in DSPy, specifically due to `Too many retries trying to get the correct output format`.
   - Another member explained that this often happens when the models output filler text around the JSON, and the default JSON parser in DSPy is not very good. They linked to an existing issue on GitHub to address this.
- **Truncated outputs in DSPy**: A member noticed that their outputs were getting truncated while using DSPy.
   - They suspect that the prompt DSPy is creating may be exceeding the token limits. They asked how to view the final prompt sent to the LLM and if there are other reasons why outputs might be truncated.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1161519468141355160/1161519469319946286/1265353579103912007">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://discordapp.com/channels/1161519468141355160/1161519469319946286/1265421088326811648">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://x.com/ShreyaR/status/1635292887595491328">Tweet from shreya rajpal (@ShreyaR)</a>: Excited to release ✨Guardrails AI✨— an open-source package to add SLAs for LLM outputs!  Guardrails supports 🌟 pydantic-style validation of LLM outputs 🌟 corrective actions (e.g. reasking LLM) when ...</li><li><a href="https://github.com/guardrails-ai/guardrails">GitHub - guardrails-ai/guardrails: Adding guardrails to large language models.</a>: Adding guardrails to large language models. Contribute to guardrails-ai/guardrails development by creating an account on GitHub.</li><li><a href="https://github.com/outlines-dev/outlines">GitHub - outlines-dev/outlines: Structured Text Generation</a>: Structured Text Generation. Contribute to outlines-dev/outlines development by creating an account on GitHub.</li><li><a href="https://github.com/stanfordnlp/dspy/issues/1365#issuecomment-2276998691">Request for OpenAI Structured Output Support · Issue #1365 · stanfordnlp/dspy</a>: Hello DSPy Team, I want to request the addition of support for OpenAI&#39;s structured output in your library. OpenAI recently introduced structured outputs in their API, which seems to guarantee the ...</li><li><a href="https://github.com/sgl-project/sglang">GitHub - sgl-project/sglang: SGLang is a fast serving framework for large language models and vision language models.</a>: SGLang is a fast serving framework for large language models and vision language models. - sgl-project/sglang</li><li><a href="https://github.com/jxnl/instructor">GitHub - jxnl/instructor: structured outputs for llms</a>: structured outputs for llms . Contribute to jxnl/instructor development by creating an account on GitHub.</li><li><a href="https://github.com/guidance-ai/guidance">GitHub - guidance-ai/guidance: A guidance language for controlling large language models.</a>: A guidance language for controlling large language models. - guidance-ai/guidance</li><li><a href="https://github.com/stanfordnlp/dspy/issues/1001">[Feature Request] Better Typed Predictors for Reliable JSON Generation · Issue #1001 · stanfordnlp/dspy</a>: The Typed Predictors output JSON as plain text. They do not use function calling method/template and do not enforce the output to be a valid JSON. This makes generating structured output challengin...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1276833487256551446)** (1 messages): 

> - `ColBERT model for German language`
> - `ColBERTv2 data format` 


- **Training ColBERT model in German**: A user is seeking to train a ColBERT model for the German language and desires to use 32-way triplets similar to ColBERTv2, but is unsure about the data format required for training.
   - The user is looking for information on how to structure the training data for ColBERTv2 in German, as the repository only provides examples for ColBERTv1 and the Hugging Face dataset appears empty.
- **ColBERTv2 data format**: The user proposes a data format for training ColBERTv2 in German: `raw_query = [(query, (positive_passage, positive_score) , [(negative_passage1, negative_score1), (negative_passage2, negative_score2), ...])]`.
   - They are seeking confirmation or insights on whether this proposed data format is suitable for training the ColBERT model in German.


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1276629616739614730)** (5 messages): 

> - `Hugging Face Leaderboard`
> - `BFCL V2-Live Dataset`
> - `Model Evaluation and Uploads to BFCL` 


- **Hugging Face Leaderboard Now Mirrors Website**: The [Hugging Face Leaderboard](https://huggingface.co/spaces/bigscience/bfcl) now mirrors the website leaderboard, thanks to a recent pull request.
   - A team member requested feedback on this change and asked for any concerns or suggestions.
- **Discussion on BFCL V2-Live Dataset Accuracy Calculation**: A discussion on the [BFCL V2-Live dataset](https://github.com/ShishirPatil/gorilla/discussions/602) opened up with a question about how to calculate the overall accuracy for the dataset.
   - The dataset features a diverse set of **2,251 question-function-answer pairs**, comprised of **258 simple, 7 multiple, 16 chained, and 14  multi-stage function calls**.
- **Adding New Model to BFCL: Open-Source and Multi-Model Considerations**: A new member asked about adding their model to BFCL and inquired about the process for uploading a model without making it open-source or providing a URL for inference.
   - They also inquired about evaluating models with multiple components, such as a fine-tuned base model and an outcome-supervised value model.



**Link mentioned**: <a href="https://github.com/ShishirPatil/gorilla/discussions/602">[BFCL] How should we calculate the overall accuracy for V2-Live dataset? · ShishirPatil/gorilla · Discussion #602</a>: Would love to hear the thoughts from the community on this matter: BFCL V2 • Live dataset features a diverse set of 2,251 question-function-answer pairs. It comprises of 258 simple, 7 multiple, 16 ...

  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1276636828329443350)** (2 messages): 

> - `Gorilla Function Calling`
> - `Gorilla Leaderboard`
> - `Contributing to Gorilla`
> - `Executable Test Pairs` 


- **Gorilla Leaderboard Explained**: A user asked for clarification on "prepare the executable test pairs" in the [Gorilla Leaderboard documentation](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing).
   - The documentation encourages users to contribute executable test pairs to the leaderboard.
- **Gorilla: Training LLMs for Function Calls**: The [Gorilla Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing) is for training and evaluating LLMs for function calls (tool calls).
   - It uses a standardized benchmark to evaluate different models and allows for comparison.



**Link mentioned**: <a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#contributing">gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla</a>: Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1276756975430008852)** (4 messages): 

> - `Anthropic's mechanistic interpretability`
> - `Llama 8b`
> - `Mistral`
> - `Open Source Implementations`
> - `AI Engineer London Meetup` 


- **Anthropic's mechanistic interpretability is expensive to run**: A user asked about the cost of running Anthropic's mechanistic interpretability work on a large language model like Llama 8b or Mistral, citing the lack of open-source implementations.
   - They were curious if the process is data-intensive or compute-heavy, or if there are other factors contributing to its limited availability.
- **AI Engineer London Meetup - September 12th**: The first AI Engineer London Meetup will be held on the evening of September 12th, bringing a slice of Swyx's AI Engineer World's Fair to the UK.
   - The event will feature four amazing speakers: Maxime LaBonne, Rovio Sc, Martins Bruveris, and Chris Bull. Registration is available via the link provided.



**Link mentioned**: <a href="https://x.com/dctanner/status/1827071893448618453">Tweet from Damien C. Tanner (@dctanner)</a>: We&#39;re brining a slice of @swyx&#39;s AI Engineer World&#39;s Fair to London!  Evening of 12 September is the first AI Engineer London Meetup.   Hear from 4 amazing speakers: @maximelabonne, @rovio...

  

---



### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1277679943106695178)** (3 messages): 

> - `OpenAI DevRel`
> - `Logan's Departure`
> - `Romain Huet` 


- **Romain Huet Takes Over OpenAI DevRel**: The new head of developer relations at **OpenAI** is **Romain Huet**. 
   - Huet's [Twitter](https://x.com/romainhuet) profile confirms he joined **OpenAI** in July 2023.
- **What happened to Logan?**: Logan left OpenAI in **July 2023**. 
   - His departure was announced by his replacement, **Romain Huet**, who stated that Logan's transition went smoothly.



**Link mentioned**: <a href="https://x.com/romainhuet">Tweet from undefined</a>: no description found

  

---



### **Alignment Lab AI ▷ #[general](https://discord.com/channels/1087862276448595968/1095458248712265841/1276799769288769640)** (1 messages): 

> - `AI Engineer London Meetup` 


- **AI Engineer London Meetup Announced**: The first AI Engineer London Meetup will take place on the evening of 12 September.
   - Hear from four amazing speakers: Maxime La Bonne, Roviosc, Martins Bruveris, and Chris Bull. Registration is available at [the link provided](https://x.com/dctanner/status/1827071893448618453).
- **London Meetup Inspired by AI Engineer World's Fair**: This London Meetup is a slice of @swyx's AI Engineer World's Fair.
   - The event is hosted by @dctanner, who also announced the speakers for the evening.



**Link mentioned**: <a href="https://x.com/dctanner/status/1827071893448618453">Tweet from Damien C. Tanner (@dctanner)</a>: We&#39;re brining a slice of @swyx&#39;s AI Engineer World&#39;s Fair to London!  Evening of 12 September is the first AI Engineer London Meetup.   Hear from 4 amazing speakers: @maximelabonne, @rovio...

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/)** (1 messages): 

rolandtannous: hello is Hamel around?
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1277667337398653081)** (1 messages): 

> - `CUDA Hackathon`
> - `NVIDIA Engineers`
> - `Accelerated Computing` 


- **CUDA Hackathon in San Francisco**: A CUDA Hackathon event will be held in San Francisco on September 21st, giving attendees the chance to hack on CUDA with NVIDIA engineers.
   - This event is a great opportunity to learn from NVIDIA experts and work on cutting-edge accelerated computing projects.
- **Accelerated Computing with NVIDIA**: The event focuses on CUDA, NVIDIA's parallel computing platform and programming model for GPUs.
   - Participants will have access to NVIDIA engineers and resources to build and optimize applications using CUDA.


  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1277564072430141511)** (1 messages): 

> - `Together API pricing change`
> - `Together API 8B and 70B price increases`
> - `Together API pricing versus OpenAI pricing`
> - `Together AI Funding` 


- **Together AI Price Increase**: Together API will increase prices for its Serverless Reference endpoints for Llama-3 8B and Llama-3 70B models on September 1, 2024.
   - The price of Llama-3 8B will increase from $0.20 per million tokens to $0.40 per million tokens, and the price of Llama-3 70B will increase from $0.90 per million tokens to $1.80 per million tokens.
- **Together API Turbo and Lite Price Remains Unchanged**: Together API's Turbo and Lite endpoints will remain at their previously announced prices as reflected on the [Together Pricing Page](https://www.together.ai/pricing).
   - These prices were last announced on July 18, 2024.
- **OpenAI's GPT-4O-Mini and Together's Pricing**: A member noted that OpenAI recently dropped the price for GPT-4O-Mini.
   - They pointed out the contrast to Together AI's pricing increase as being strange.
- **Together AI's Funding**: A member speculated that Together AI is doubling their prices due to their funding running out.
   - They also mentioned that 4-bit and 8-bit pricing should remain unchanged for now, but could change in the future.



**Link mentioned**: <a href="https://docs.together.ai/docs/pricing-changes">Announcement of pricing changes to Together's serverless API</a>: Together serverless API pricing changes

  

---



---



{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

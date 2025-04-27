---
id: a8d4e427-2925-4b28-868a-b6d5cc9b9f28
title: not much happened today
date: '2024-12-31T02:24:45.402646Z'
original_slug: ainews-to-be-named-9002
description: >-
  **Sam Altman** publicly criticizes **DeepSeek** and **Qwen** models, sparking
  debate about **OpenAI**'s innovation claims and reliance on foundational
  research like the **Transformer architecture**. **Deepseek V3** shows
  significant overfitting issues in the **Misguided Attention** evaluation,
  solving only **22%** of test prompts, raising concerns about its reasoning and
  finetuning. Despite skepticism about its open-source status, **Deepseek V3**
  is claimed to surpass **ChatGPT4** as an open-source model, marking a
  milestone 1.75 years after ChatGPT4's release on **March 14, 2023**. The
  discussions highlight competitive dynamics in AI model performance and
  innovation sustainability.
companies:
  - openai
  - deepseek
  - google
  - qwen
models:
  - deepseek-v3
  - chatgpt-4
topics:
  - overfitting
  - reasoning
  - misguided-attention
  - model-evaluation
  - model-architecture
  - finetuning
  - open-source
people:
  - sam-altman
---


<!-- buttondown-editor-mode: plaintext -->**a quiet week is all we need.**

> AI News for 12/27/2024-12/30/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **32** Discords (**215** channels, and **5832** messages) for you. Estimated reading time saved (at 200wpm): **696 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Enjoy the break.

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

TO BE COMPLETED

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Deepseek's V3: Performance and Critique**

- **[Sam Altman is taking veiled shots at DeepSeek and Qwen. He mad.](https://i.redd.it/lba9xu2mqx9e1.jpeg)** ([Score: 1486, Comments: 432](https://reddit.com/r/LocalLLaMA/comments/1hphlz7/sam_altman_is_taking_veiled_shots_at_deepseek_and/)): **Sam Altman** criticizes **DeepSeek** and **Qwen** models, highlighting the simplicity of replicating existing ideas versus the complexity and risk of genuine innovation. His post on Twitter has garnered significant attention with **1.3 million views**, **1,175 reposts**, **233 quote tweets**, **15.2K likes**, and **2,046 bookmarks**.
  - Many commenters criticize **Sam Altman** and **OpenAI** for claiming innovation while relying heavily on foundational research from **Google** and other open-source contributions, noting that **OpenAI**'s work builds on existing technologies like the **Transformer architecture** from the paper *Attention Is All You Need*. They argue that **OpenAI** has monetized public knowledge while restricting access to its own findings.
  - There is a sentiment that **OpenAI**'s competitive edge or "moat" is questionable, as models like **DeepSeek** and **Qwen** are achieving similar performance at lower costs. Commenters highlight the irony of **OpenAI**'s past actions, such as scraping the internet for data without compensation, while now criticizing others for leveraging their work.
  - The discussion includes skepticism about **OpenAI**'s sustainability and innovation claims, pointing out that **OpenAI**'s profitability is challenged by competitors offering similar services cheaper. The conversation also touches on the broader issue of how innovation is often a cumulative process, with companies building on each other's work rather than creating entirely new concepts.


- **Deepseek V3 performs surprisingly bad in Misguided Attention eval, which tests for overfitting.** ([Score: 176, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1hpjhm0/deepseek_v3_performs_surprisingly_bad_in/)): **Deepseek V3** performed poorly in the **Misguided Attention** evaluation, solving only **22%** of the 13 test prompts, indicating significant overfitting issues. The model struggled with prompts involving slight variations of known problems, possibly due to optimizations like the compressed KV cache or MoE, and exhibited repetitive loops, suggesting potential finetuning issues related to reasoning traces.
  - **Overfitting and Reasoning Challenges**: The discussion highlights **Deepseek V3's** overfitting issues, with users suggesting that the model's reasoning capabilities could be better evaluated using its **DeepThink mode**. There is a consensus that the model struggles with variations of known problems, possibly due to biases in pretraining data and finetuning challenges.
  - **Misguided Attention and Evaluation Methods**: The term "misguided attention" is debated, with some users noting it describes the evaluation issue well. The evaluation of reasoning models is complicated by API limitations, leading to reliance on web interfaces, which can skew results.
  - **Model Architecture and Performance**: There is speculation about the architecture of various models, with some users noting that **Deepseek** models are stubborn in task execution, possibly due to **MoE** architecture. The conversation also touches on the performance of smaller models like **o1-mini** in specific tasks, indicating varying strengths across different models.


- **Many asked: When will we have an open source model better than chatGPT4?  The day has arrived.** ([Score: 204, Comments: 106](https://reddit.com/r/LocalLLaMA/comments/1hprz6x/many_asked_when_will_we_have_an_open_source_model/)): **Deepseek V3** is claimed to surpass **ChatGPT4** as an open-source model, achieving this milestone 1.75 years after ChatGPT4's release on **March 14, 2023**. The announcement was shared via a [link](https://x.com/lmarena_ai/status/1873695386323566638).
  - **Deepseek V3's Open Source Status**: There is skepticism about Deepseek V3 being truly open source, as it uses the **r1-lite model**, which isn't available for download. Users express doubt over claims that Deepseek surpasses GPT-4, noting that open-source models have reportedly outperformed GPT-4 for some time.
  - **Model Performance and Parameters**: The **Mixture-of-Experts architecture** for Deepseek V3 has **671B total parameters with 37B activated parameters**, but users question its real-world performance compared to benchmarks. Discussions highlight the superiority of models like **Claude Sonnet 3.5**, which is praised for its tone and feedback integration, over GPT-4.
  - **Comparative Model Analysis**: Users compare various models, such as **Qwen2.5-32b** and **Llama 405b**, which reportedly outperform GPT-4 in certain benchmarks and tasks. The conversation also touches on the desire for open-source models with capabilities akin to **o1 mini** and emphasizes the historical context of GPT-4's performance.


**Theme 2. Cerebras's Trillion Parameter Training on CS-3**

- **[10th December 2024: Cerebras Systems + US Energy Sandia National Labs have CLAIMED to demonstrate training of a 1 trillion parameter model on a single CS-3 system (!) This is ~1% the footprint & power of an equivalent GPU cluster.](https://www.reddit.com/gallery/1hpejko)** ([Score: 348, Comments: 66](https://reddit.com/r/LocalLLaMA/comments/1hpejko/10th_december_2024_cerebras_systems_us_energy/)): Cerebras Systems and **US Energy Sandia National Labs** have announced the successful training of a **1 trillion parameter model** on a single **CS-3 system**, claiming it uses only about **1% of the footprint and power** compared to an equivalent GPU cluster. For more details, refer to their [press release](https://cerebras.ai/press-release/cerebras-demonstrates-trillion-parameter-model-training-on-a-single-cs-3-system) and related posts on [CerebrasSystems](https://x.com/CerebrasSystems/status/1867296161750536442?t=wU_lBuMzYLClIb7ja4sjvw&s=19) and [SandiaLabs](https://x.com/SandiaLabs?t=7yRTp8-c5zXhEN23qEhXwA&s=09).
  - **Wafer Yield and Die Defects**: Discussions highlighted skepticism about Cerebras' claims of defect-free dies, referencing historical allowances for defective dies in their products. Calculations suggested that achieving a 99.9954% yield per die is highly improbable, given typical defect densities reported by **TSMC**.
  - **Hardware and Performance**: The training was conducted on a cluster of 16 CS-3 chips, not a single chip, which some found misleading. Users pointed out that while the architecture could potentially lower costs by consolidating numerous cores onto a single board, the performance and scalability compared to traditional GPU clusters remain crucial considerations.
  - **Cerebras' Market Position**: Despite the promising technology, Cerebras hasn't been widely adopted, potentially due to supply issues or the lack of an accessible ecosystem for startups. The discussion also touched on the potential for Cerebras to disrupt **Nvidia's** dominance if their hardware proves superior and can be easily integrated into existing frameworks like **PyTorch**.


**Theme 3. Affordable Local AI: Performance on Budget GPUs**

- **Budget AKA poor man Local LLM.** ([Score: 354, Comments: 76](https://reddit.com/r/LocalLLaMA/comments/1hpg2e6/budget_aka_poor_man_local_llm/)): A Reddit user describes building a budget-friendly local **LLM** setup using older hardware, including a **CROSSHAIR V FORMULA-Z** motherboard and **2x P102-100** GPUs, for a total cost of **$130**. Despite limitations in image generation speed, the setup efficiently runs various models like **Phi-4-14B** and **llama3.2-3b** with response times under one second, demonstrating the feasibility of low-cost, performance-oriented AI experimentation.
  - **GPU Performance Comparisons**: The **RTX 3060 12GB** is highlighted as a budget-friendly option for AI tasks, with performance metrics showing **12 tokens per second** for certain models. Comparatively, the **4060 Ti 16GB** achieves **23 tokens per second**, indicating a significant performance boost for a modest price increase, as discussed in [this Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1hp7yft/gpu_poors_dilemma_3060_12gb_vs_4060_ti_16gb/).
  - **Budget Hardware Feasibility**: While the setup described in the post costs **$130**, it may not be generally repeatable at that price, with potential total costs reaching **$500** due to additional components. However, using mining GPUs and second-hand components can still create a powerful system for around **$200** if deals are found.
  - **Community Interest and Experimentation**: The post has sparked interest among users wanting to experiment with larger models on a budget. Some users are considering similar setups using older or unused hardware, and there's curiosity about performance in other domains like image classification, although the setup is primarily geared towards **LLMs**.


**Theme 4. SmallThinker-3B: Efficient Reasoning in Small Scale Models**

- **Introducing SmallThinker-3B-Preview. An o1-like reasoning SLM!** ([Score: 303, Comments: 58](https://reddit.com/r/LocalLLaMA/comments/1hpop3y/introducing_smallthinker3bpreview_an_o1like/)): The **SmallThinker-3B-Preview** is a new reasoning model finetuned from **Qwen2.5-3b-Instruct**, designed for edge deployment and as a draft model for **QwQ-32B-Preview**, offering over **70% speedup** in token processing on an **NVIDIA 4090**. The model uses the **QWQ-LONGCOT-500K** dataset, with over **75%** of samples having output tokens exceeding **8K**, and is available for open-source research, though it currently has issues with repetitive outputs.
  - Discussions focused on **speculative decoding** and its implementation, with users sharing command-line parameters for deploying models using **llama-server** and **vllm**. A specific setup involving **CUDA_VISIBLE_DEVICES** and **tensor-parallel-size** was mentioned for optimizing speculative decoding with the **SmallThinker-3B-Preview** model.
  - Comments highlighted the potential of smaller models like **SmallThinker-3B-Preview** for **edge computing**, emphasizing their ability to run efficiently on consumer-grade GPUs. Users expressed interest in enhancing these models with **retrieval-augmented generation (RAG)** capabilities and tools for improved knowledge and reflection.
  - The model's fine-tuning process was discussed, with **llama-factory** being used and plans to share the training configuration. It was noted that fine-tuning the **3B model** could be done with a **single NVIDIA 4090 or 3090 GPU**, reflecting the model's accessibility for further development.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. OpenAI's O1 Offers Significant Advantage in Math and Education**

- **O1 is very good at Math and wins the Putnam Exam** ([Score: 109, Comments: 84](https://reddit.com/r/OpenAI/comments/1hpgaxm/o1_is_very_good_at_math_and_wins_the_putnam_exam/)): **O1** demonstrated exceptional mathematical prowess by scoring **8/12** on the **2024 Putnam Exam**, a significant achievement given the exam's difficulty. The correct answers were for problems **A1, A2, A3, A4, A6, B3, B4,** and **B5**, while errors occurred on **A5, B1, B2,** and **B6**.
  - **O1's Performance and Grading**: The discussion highlights skepticism regarding O1's reported performance on the **2024 Putnam Exam**, with some suggesting that the grading might not align with the rigorous standards of the exam. **Kevin Buzzard** estimates O1 got one problem right and partial credit on others, as discussed in [his blog](https://xenaproject.wordpress.com/2024/12/22/can-ai-do-maths-yet-thoughts-from-a-mathematician/).
  - **Training Data and Exam Timing**: There's clarification that the **2024 Putnam exam** occurred after the AI's training data cutoff in 2023, suggesting that O1 did not have prior access to the exam content, as confirmed by **Science_421**.
  - **AI's Approach vs. Human Approach**: Commenters note that O1 often reaches correct answers without showing all steps, akin to a physicist's approach rather than a mathematician's, who would typically provide a detailed proof. This style is not aligned with the Putnam's grading criteria, which values complete logical reasoning.


- **o1 is literally a game-changer!** ([Score: 126, Comments: 64](https://reddit.com/r/OpenAI/comments/1hpo32o/o1_is_literally_a_gamechanger/)): O1 significantly enhances the learning experience compared to **GPT-4**, making complex problem sets more manageable and improving the user's understanding of the process rather than just providing answers. This has resulted in improved academic performance and increased parental approval.
  - **Clarification Issues**: Users noted that while **O1** provides significant improvements over **GPT-4** in educational settings, it still struggles with making assumptions and providing incorrect answers without seeking clarification, a common problem across many **LLMs**. Suggestions included the need for more explicit input requirements to mitigate these issues.
  - **Coding Challenges**: A user shared an experience where **O1** provided incorrect coding information and stubbornly insisted on its correctness despite evidence to the contrary. Switching to **4o** resulted in immediate correction and apology, highlighting discrepancies in performance between the two models.
  - **Educational Impact**: The **O1** model is praised for its potential to revolutionize education by providing intelligent assistance in understanding complex subjects, with some users warning against over-reliance on the tool to ensure genuine learning. Concerns were raised about the illusion of improved grades when using **LLM** aids for problem sets.


- **[OpenAI, Andrew Ng Introduce New Course on Reasoning with o1](https://analyticsindiamag.com/ai-news-updates/openai-andrew-ng-introduce-new-course-on-reasoning-with-o1/)** ([Score: 116, Comments: 13](https://reddit.com/r/OpenAI/comments/1hpx8cj/openai_andrew_ng_introduce_new_course_on/)): **OpenAI** and **Andrew Ng** have introduced a new course focused on **reasoning with O1**, although the post does not provide further details or context.
  - The new course on **reasoning with O1** by **OpenAI** and **Andrew Ng** is available for free, as highlighted by multiple commenters.
  - **Andrew Ng's courses** generally receive positive feedback, especially those he personally teaches, though some are criticized for being outdated due to the rapid pace of AI advancements.
  - A direct link to the free course is provided by a commenter: [Reasoning with O1](https://www.deeplearning.ai/short-courses/reasoning-with-o1/).


**Theme 2. MAMBA Model's Struggle Against Transformer Dominance**

- **[D] - Why MAMBA did not catch on?** ([Score: 134, Comments: 49](https://reddit.com/r/MachineLearning/comments/1hpg91o/d_why_mamba_did_not_catch_on/)): **MAMBA** was anticipated to replace transformers due to its efficiency, offering **O(N)** complexity during training and **O(1)** during inference while maintaining comparable accuracy. Despite these advantages, it did not become dominant, possibly due to limitations in state space models or other unaddressed theoretical constraints.
  - **MAMBA's Limitations**: MAMBA models face practical challenges such as fixed state memory which limits their ability to handle tasks requiring dynamic state tracking, unlike transformers which utilize self-attention for efficient information retrieval. These limitations have been highlighted in theoretical analyses and experiments showing that MAMBA struggles with state tracking and practical copy tasks.
  - **Transformer Dominance**: The maturity of the software and hardware stack for transformers, including tools like **Hugging Face** and **CUDA** optimizations, makes them more accessible and efficient for large-scale applications. This established infrastructure, combined with the high cost of retraining models, deters the adoption of MAMBA despite its potential runtime efficiency advantages.
  - **Research and Development**: Current research continues to focus on improving transformer architectures, with innovations like **Hyena Hierarchy** offering significant improvements in efficiency and accuracy over traditional attention mechanisms. This ongoing development and the proven scalability of transformers suggest that alternatives like MAMBA will remain less popular until a major shift occurs in the landscape.


**Theme 3. OpenAI's AGI Definition and Economic Metrics**

- **[Leaked Documents Show OpenAI Has a Very Clear Definition of ‘AGI’](https://gizmodo.com/leaked-documents-show-openai-has-a-very-clear-definition-of-agi-2000543339)** ([Score: 101, Comments: 62](https://reddit.com/r/OpenAI/comments/1hpe6va/leaked_documents_show_openai_has_a_very_clear/)): **OpenAI**'s definition of **Artificial General Intelligence (AGI)** has been revealed through leaked documents. The details of these documents have not been provided, but the revelation indicates that OpenAI has a specific and clear understanding of AGI.
  - The discussion highlights skepticism about using **$100 billion** as a benchmark for achieving **AGI**, with users arguing that financial success does not equate to general intelligence. **CarrotcakeSuperSand** explains that this metric is tied to a clause in the **Microsoft deal**, where Microsoft loses rights to OpenAI’s IP upon reaching AGI, thus necessitating a clear financial threshold.
  - **Corgis_are_awesome** clarifies that the **$100 billion** figure is related to **Microsoft’s** initial investment and a 100x cap on their profit, separate from AGI definitions. The **OpenAI charter** states AGI as an AI system exceeding human capabilities in economically valuable work, with the board having the authority to determine AGI achievement.
  - **Class_of_22** and others express confusion and criticism over the perceived arbitrary nature of the profit-based AGI benchmark, with **FlugonNine** suggesting that the focus on wealth generation reflects the venture capitalist mindset within **OpenAI**. **Cyberdork** humorously critiques **Sam Altman’s** background, attributing the monetary focus to his business-oriented career.


**Theme 4. AI's Role in Gaming and Social Media**

- **[Dead Internet Theory is now a corporate objective](https://i.redd.it/jjoft3iqzw9e1.png)** ([Score: 393, Comments: 110](https://reddit.com/r/OpenAI/comments/1hpf6re/dead_internet_theory_is_now_a_corporate_objective/)): Meta plans to introduce **AI-generated characters** on Facebook to boost user engagement, allowing interactions that mimic real human interactions through their AI studio. This initiative, reported by the **Financial Times**, aligns with the broader trend of integrating AI in digital platforms, raising concerns about the authenticity of online interactions.
  - **AI Models' Limitations**: **swagonflyyyy** points out the limitations of AI models in conversational contexts, noting that while they excel in utility for backend applications, they often fall short in direct user interactions. **Gemma2's 27B model** is highlighted as superior for general chatting, and AI's role is better suited for backend tasks like moderation and summarization rather than frontend user interaction.
  - **Concerns Over AI Manipulation**: **AppropriateScience71** and **sdmat** express concerns over AI being used to manipulate users, citing **BlackOps 6's EOMM** as a negative example of AI altering game dynamics to enforce outcomes. There is a general sentiment that AI's role in altering user experiences, whether in gaming or social media, is perceived negatively and could harm user engagement.
  - **Prevalence of AI on Social Media**: **Agile-Landscape8612** and **OptimismNeeded** discuss the widespread presence of AI-generated content on platforms like Facebook, with many users seemingly unaware of it. This suggests that AI-generated posts are already integrated into social media, and banning bots could significantly impact platform content.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-2024-12-17

**Theme 1. AI Models Fight for Coding Supremacy**  

- [**DeepSeek V3 Displays Complex Coding Skills**](https://openrouter.ai/deepseek/deepseek-chat): It handles large context windows, excels at tasks like building MTG decks, and outruns some closed-source models. Yet it struggles with *“reasoning loops”* and XML outputs, showing room for refinement.  
- [**Gemini 2.0 Wins Hearts with Speed**](https://github.com/google-gemini/cookbook): Users praise Gemini’s *“flash thinking”* for coding assistance, claiming it sometimes beats GPT-4 in speed. They also look forward to Gemini’s upcoming features for specialized tasks like code generation.  
- [**Codeium 2024 Wrapped Confirms New Year Features**](https://codeium.com/wrapped-2024): The platform offered year-end coding stats while teasing *“lots of work left to do”* for 2025. Users reported both excitement and frustration over Windsurf outages and credit consumption.

**Theme 2. Fine-Tuning & LoRA Legwork**  

- [**LoRA Proves Useful but Tricky**](https://huggingface.co/docs/peft/main/en/developer_guides/lora#eva): Developers argue it retains new knowledge but warn about inflated expectations and dataset pitfalls. Discussions often mention *overfitting* risks in large-scale pretraining.  
- [**Hymba-1.5B-Instruct Goes Commercial**](https://huggingface.co/nvidia/Hymba-1.5B-Instruct): It draws praise for open-source instruction datasets and *“strict batch size requirements,”* prompting legal and ethical usage questions. Contributors see it as a stepping stone for robust AI solutions.  
- [**OpenRouter and Aider Integration**](https://aider.chat/docs/config/options.html): Coders encountered *‘model not found’* errors hooking DeepSeek V3 via OpenRouter. Proper environment variables and endpoint settings solved it, enabling streamlined fine-tuning workflows.

**Theme 3. Quantization & HPC Performance**  

- [**FP8 Tactics Accelerate Transformer Engines**](https://github.com/NVIDIA/TransformerEngine): NVIDIA’s FP8 approaches promise smaller numeric footprints with strong accuracy. Users highlight new 2D block quantization from [PyTorch’s blog](https://pytorch.org/blog/accelerating-gemms-triton/) for near 2x speedups.  
- [**TMA vs cp.async Sparks Debate**](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/): Fewer threads and registers make TMA more resource-efficient than cp.async. Developers see big gains in HPC tasks, especially GEMM-based workloads.  
- [**3090 NV-Link & Jetson Orin Nano Face Trials**](https://www.jeremymorgan.com/blog/tech/nvidia-jetson-orin-nano-speed-test/): Multi-GPU bridging intrigues performance seekers, but noise and cost concerns abound. Meanwhile, the Jetson Orin Nano’s *25W mode* impresses with modest but functional on-device AI endeavors.

**Theme 4. RAG, Embeddings & Agent Workflows**  

- [**Local RAG with LlamaIndex**](https://t.co/4WJ7ZcXy3H): Users feed Excel tables to Llama-3.2 or Llama-3.3, enabling advanced retrieval-augmented generation. Neomagus verifies *imported citations* to guard against AI hallucinations.  
- [**Light Prompter Shows Efficient Test-Time**](https://github.com/Green0-0/light_prompter): It batches prompts for faster model inference, and devs wonder if *test time training* tweaks model weights too. Others see parallels to RL research for *real-time updates*.  
- [**Vision Meets Embeddings**](https://nomic.ai/blog/posts/gpt4all-scaling-test-time-compute): Nomic’s *nomic-embed-vision-v1* pairs with text embeddings to refine image search. This approach teases *multimodal expansions* in GPT4All and beyond.

**Theme 5. APIs, Pricing & Prompt Engineering**  

- [**OpenRouter Users Weigh Costs**](https://openrouter.ai/rankings/translation?view=week): Some lament no discounts for input tokens, while performance of models like *GPT-4o mini* fuels translation-friendly usage. Providers jockey to differentiate with *“niche”* model strengths.  
- [**Perplexity Pro Baffles Subscribers**](https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter): DeepSeek v3 is missing despite its touted perks, prompting calls to stick with a free tier. Meanwhile, *Reasoning Mode* lumps complex queries into structured answers for advanced Q&A.  
- [**Prompt Engineering Gains Structure**](https://x.com/sh_reya/status/1873431565650502060): Overly broad requests baffle AI code tools, so devs break tasks into smaller steps. People eye *“Sora channels”* and markdown-friendly spaces for effective knowledge sharing.

---

# PART 1: High level Discord summaries




## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Codeium 2024 Wrapped & New Year Roadmap**: The team launched [Codeium 2024 Wrapped](https://codeium.com/wrapped-2024), urging everyone to view and share coding stats in style, followed by a warm year-in-review thank you.
   - They hinted at more **features** rolling out in 2025, emphasizing *lots of work left to do* to amp up the user experience.
- **Windsurf's Furious Outages & Credit Conundrums**: Users reported sluggish responses and 503 errors with **Windsurf**, prompting some to push for a [status page](https://stats.uptimerobot.com/aiH8grJl1y) for real-time updates.
   - Frustrations over depleted **premium credits** led to refund demands and exploration of alternatives like **ChatGPT 4o** to cope with repeated downtime.
- **DeepSeek V3 Dreams Drag On**: Impatient chatter arose around the delayed integration of **DeepSeek V3** in Windsurf, with users watching rival tools like **Cline** adopt it sooner.
   - Questions swirled about feature priorities, as some urged Codeium to speed up the merge to keep pace in the AI editor race.
- **Context Clutter in Codeium**: A lively debate grew around how **Codeium** handles context length for code revisions, leaving many confused over real limits versus marketing claims.
   - People found persistent issues with maintaining code discussions, even though the platform boasts a high context length for advanced usage.
- **React Native SVG Slip-Ups**: A user detailed trouble loading **SVG icons** on native simulators despite flawless web previews, stirring suspicion of version conflicts with `react-native-svg` and Expo.
   - Community members advocated debugging platform compatibility and library versions before resorting to drastic reconfigurations in their app setup.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LoRA Legwork in Fine-Tuning**: Members debated whether **LoRA** is effective for large-scale pretraining, pointing out that careful dataset structuring is crucial to avoid overfitting and inflated expectations ([documentation link](https://huggingface.co/docs/peft/main/en/developer_guides/lora#eva)).
   - They shared *previous experiences*, acknowledging skepticism over LoRA's reliability for knowledge retention, with references to [continued pretraining tips](https://unsloth.ai/blog/contpretraining).
- **Quantization Quandaries in Llama.cpp**: Some users encountered **quantization issues** with **Llama.cpp** after recent library updates, causing errors during integration ([sample issue report](https://github.com/unslothai/unsloth/issues/1333)).
   - Discussion focused on missing dependencies and the lack of unsloth quantization for bigger models like **Phi 4**, highlighting *operational delays* and library version mismatches.
- **Hymba's Hype for Commercial Use**: The **Hymba-1.5B-Instruct** model was introduced with claims of ready-for-commercial usage and *strict batch size requirements*, as seen on [Hugging Face](https://huggingface.co/nvidia/Hymba-1.5B-Instruct).
   - Contributors pointed out that it was derived from open-source instruction datasets, reminding everyone of *legalities and ethical considerations* for distributing advanced AI technology.
- **Light Prompter Lifts Test-Time Efficiency**: The GitHub project [Light Prompter](https://github.com/Green0-0/light_prompter) showcases batching tactics to increase **model inference efficiency**, featuring relevant notebooks and code examples.
   - A member mentioned *test time training* and how it might update weights during inference, with others suggesting it could overlap with **RL** research yet to be fully explored.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Claude 3.5 Sonnet stirs speculation**: Users questioned whether **claude-3.5-sonnet** differs from **claude-3.5-sonnet-20241022**, referencing [a Cursor forum thread](https://forum.cursor.com/t/are-claude-3-5-sonnet-and-claude-3-5-sonnet-20241022-different/24272/3).
   - They noted that **claude-3.5-sonnet** now redirects to the updated **20241022** build, prompting curiosity over performance gains.
- **Composer vs Chat face-off**: Some praised the **Composer** tool for code refinement, even pointing to [a discussion on quick 'Fix' actions](https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221).
   - Others valued **Chat** for general guidance, suggesting that a more direct or even frustrated tone occasionally yielded sharper **Cursor** responses.
- **Cursor powers web apps**: One person highlighted **Cursor**’s ease of use by delivering a functional web tool for a mobile MMO game without extensive coding background.
   - Another shared a **Guitar Chord Learning App** link such as [this fretboard tool](https://guitar-tab.vercel.app/en/tools/fretboard), underscoring Cursor’s utility for full-stack prototypes.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Grok’s Great Credit Countdown**: With only two days left before the year ends, **Grok AI** is offering $25 in free credits for its API users, highlighted in [this official link](https://x.com/stackblitz/status/1873769044761022633), which can be integrated into **Bolt** projects.
   - Members stressed that these final hours are perfect for trying **Grok AI** within Bolt, calling it *the sweet spot for quick prototyping*.
- **Voice Prompting Wish in Bolt**: A strong push emerged for a voice prompting feature akin to **ChatGPT**, offering more convenient coding discussions but noting the heavier overhead of audio models.
   - Enthusiasts envisioned *hands-free interactions* within **Bolt**, but they anticipated potential cost spikes due to the added model complexity.
- **Supabase vs Firebase vs Convex: Database Dilemmas**: Developers weighed usage of **Supabase**, **Firebase**, or **Convex** for data hosting in **Bolt** projects, referencing an [open GitHub issue](https://github.com/stackblitz/bolt.new/issues/4455) for details.
   - Some highlighted that exporting to **StackBlitz** enables manual refinements, while others warned that **Convex** remains in beta and may warrant caution.
- **Large Codebase LLM Fatigue**: Community members noticed **Bolt** slowing on extensive codebases, occasionally altering unrelated files, leading to repeated reboots and diff checks.
   - Users recommended reloading projects and toggling **diff mode** to mitigate random edits, sharing anecdotal success stories that it helped control token usage.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek V3 Gains Momentum**: Many users are switching to **DeepSeek V3** for coding tasks, touting large context windows and [API docs references](https://api-docs.deepseek.com/quick_start/pricing). Some users weigh the privacy trade-offs of hosting vs Hugging Face usage, citing cost and context window differences.
   - Others compared it with **Gemini** for code generation, concluding **DeepSeek** is faster, especially for extensive projects, while praising the newly introduced [Context Caching](https://api-docs.deepseek.com/news/news0802#how-to-use-deepseek-apis-caching-service) feature as a cost-saver.
- **Aider Installation and Configuration**: Enthusiasts emphasize installing **Aider** globally for stability, referencing [official guidelines](https://aider.chat/docs/install.html) and specific Python setup steps. Some Arch Linux users give OS-specific tips and note that adjusting `.aider.model.metadata.json` helps manage context and costs.
   - They also discuss ways to bypass Git restrictions, pointing to [GitHub issue #211](https://github.com/Aider-AI/aider/issues/211), while acknowledging the importance of token-limit awareness.
- **Gemini 2.0 Excels at Code**: Contributors report **Gemini 2.0** handles large projects effectively, offering a free tier that helps accelerate coding tasks. They frequent references to [model providers on LiteLLM](https://docs.litellm.ai/docs/providers), underscoring performance gains in big codebases.
   - Some rely on **Gemini** for broad code loading while using specialized models like **DeepSeek** for final generation, capitalizing on each model’s traits.
- **Integrating Aider with OpenRouter**: Certain members faced 'model not found' errors when tying **OpenRouter** to **Aider**, attributing them to endpoint misconfiguration. They overcame it by enabling specific settings and verifying the correct environment variables, referencing [OpenRouter integration tips](https://aider.chat/docs/config/options.html).
   - Others caution about user privacy with hosted endpoints, but note that once configured properly, **Aider** can seamlessly invoke **DeepSeek** via **OpenRouter**.
- **OCR Implementation with TesseractJS**: A user showcased building a web app in one hour using **Aider**, employing [TesseractJS](https://github.com/naptha/tesseract.js/) for automated OCR tasks. They highlight a boost in productivity from skipping manual coding in favor of direct AI-driven generation.
   - Community members see potential in bridging OCR with code generation, indicating future expansions into advanced text extraction workflows.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM Benchmarking Bloopers**: Participants found that **LLM performance** can be skewed by ambiguous questions, referencing [ARC 'Challenge' vs ARC 'Easy'](https://arxiv.org/abs/2412.17758) as an example of questionable setups.
   - They recommended shifting to **functional** tasks over multiple-choice to capture complex reasoning, with open discussion about adopting robust metrics.
- **Gradient Routing Gains Ground**: Members praised **Gradient Routing** as a method to isolate model capabilities using data-dependent masks during backprop, referencing [a paper about localizing computation](https://arxiv.org/abs/2410.04332).
   - This technique could improve interpretability by mapping specific subregions to certain tasks, fueling insights into advanced debugging.
- **TongGeometry's Triumphant Theorems**: **TongGeometry** systematically proposed and solved olympiad-level geometry problems, as described in [Proposing and solving olympiad geometry with guided tree search](https://arxiv.org/abs/2412.10673).
   - Some solutions even made it into *regional mathematical olympiads*, highlighting the model's impressive handling of complex geometric proofs.
- **Crosscoders Crack Model Layers**: The **Crosscoders** approach tracks features across multiple layers to better interpret how models evolve representations, referencing [an open-source replication](https://www.lesswrong.com/posts/srt6JXsRMtmqAJavD/open-source-replication-of-anthropic-s-crosscoder-paper-for).
   - Practitioners hope this method pinpoints nuanced transformations in networks, aiding *circuit simplification* and direct model diffing.
- **Teeny TinyStories Tactics**: The **TinyStories** dataset compiles synthetic short stories for training **small LMs under 10 million parameters**, per [TinyStories: How Small Can Language Models Be](https://arxiv.org/abs/2305.07759).
   - Users reported success in developing simpler architectures without major performance drop, fueling interest in *lightweight model design*.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek V3 falters on OpenRouter**: Some users reported reduced performance from **DeepSeek V3** when using it through [OpenRouter](https://openrouter.ai/deepseek/deepseek-chat), leading to speculation about updates or version changes.
   - They suspect a recent modification or a possible Together API factor may be at play, prompting concerns over consistent performance and user confidence.
- **OpenRouter welcomes new LLM providers**: Community members noted that integrating models into **OpenRouter** requires partnerships with established labs or self-hosting, with specialized coding abilities as a strong differentiator.
   - They pointed to [Prompt Caching on OpenRouter](https://openrouter.ai/docs/prompt-caching#deepseek) as a key cost saver and recommended promoting niche strengths to attract user interest.
- **GPT-4o mini excels at translations**: A discussion on translation models positioned **GPT-4o mini** as a reliable choice, while **Gemini 1.5 Flash** was said to produce frequent errors.
   - Users mentioned structured system prompts and relied on the [LLM Rankings for translation](https://openrouter.ai/rankings/translation?view=week) to optimize their results.
- **Multimodal agents spark interest**: Developers explored methods for building multimodal agents, clarifying that strict JSON output isn't mandatory for agent workflows.
   - They referenced [Anthropic’s guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) and mentioned Google’s **Project Mariner** as a possible inspiration.
- **Pricing debates heat up**: Community members noticed the lack of input token discounts on **OpenRouter**, highlighting cost implications for high-volume usage.
   - While some expressed concerns about potential model downgrades, others called for transparent explanations of performance changes.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek's Divergent Demo**: **DeepSeek V3** soared in tasks like building MTG decks via Scryfall queries, ranks #22 on [Aidan's benchmark](https://x.com/aidan_mclau/status/1872444303974543859), and impresses with advanced context retention.
   - However, evaluations using [MisguidedAttention](https://github.com/cpldcpu/MisguidedAttention) revealed **reasoning loops** and contradictory results, fueling questions about its architecture.
- **Local AI vs. API: Showdown or Symbiosis?**: Members weighed the customization benefits of **Aquila's Ollama** ([ollama.com](https://ollama.com)) and **LlamaCPP** for local setups, while affirming **OpenAI API** remains essential for agentic tasks.
   - Others called for more contributors to **LlamaCPP**, citing its influence across open-source AI projects and highlighting the synergy of local plus API solutions.
- **SmallThinker-3B Surprise**: The new **SmallThinker-3B-preview** at [Hugging Face](https://huggingface.co/PowerInfer/SmallThinker-3B-Preview) shows improved reasoning benchmarks and a knack for systematic steps.
   - Yet, members joked about its inability to stop at the right time, indicating it might overgenerate responses while exploring possibilities.
- **Hunyuan's 8GB Gambit**: The **Hunyuan** video model can run on GPUs with only **8GB VRAM**, as explained in [a blog post](https://blog.comfy.org/p/running-hunyuan-with-8gb-vram-and), though it proves sluggish at lower resolutions.
   - Community members flagged **speed issues**, noting that smaller configs open doors for resource-limited setups but may hamper higher-fidelity outputs.
- **Metrics That Matter**: In **binary classification** discussions, members championed reporting **Precision**, **Recall**, **F1**, and **AUC/ROC** from sklearn for added clarity.
   - They stressed the value of a **representative test set** and urged alignment of metrics with each model’s real-world objectives.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Deepseek v3 Dodges Pro Subscription**: Community members noted that **Deepseek v3** is conspicuously missing from the **Perplexity Pro** subscription, prompting confusion about its claimed benefits and higher-level features.
   - Some questioned whether to stick to **free Deepseek** instead, citing user frustration over paying for Pro yet not seeing advanced functionality.
- **Reasoning Mode Ramps Up Complex Queries**: Users highlighted **Reasoning Mode** for detailed Q&A within **Perplexity Pro**, where it automatically kicks in for intricate queries to improve accuracy.
   - They shared examples of sorting data into tables, underscoring a shared interest in harnessing structured layouts for robust answers.
- **Claude 3.5 Sonnet Battles GPT-4O**: Multiple users debated performance trade-offs between **Claude 3.5 Sonnet** and **GPT-4O**, referencing reliability and latency differences.
   - They pointed out possible synergy with **Deepseek** or **ChatGPT Pro** for specialized tasks, stressing that no single model dominates every scenario.
- **Searching for API Alternatives & Recency Filters**: A user sought **Search API** solutions that exceed current standards and asked about a **custom recency filter**, referencing [Perplexity API docs](https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter).
   - No definitive replies emerged on filter feasibility, spurring community interest in exploring new search paradigms for advanced data retrieval.
- **Conversational API Usage Fumbles**: Questions arose about whether the **Perplexity API** can provide context-driven replies instead of dictionary-like definitions.
   - A response confirmed that **Sonar models** aim for question-answering with proper references, clarifying they are not meant to function as a general conversational agent.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Gen Debates Heat Up**: The discussion spanned the pros and cons of **image generation** tools, referencing the inconsistent results for posters and the varied performance of models like **Claude** and **Eleven Labs**.
   - Some participants voiced frustration about heavy cleanup, while others described improvements in audio and video generation workflows, citing a [Reddit thread about model unpredictability](https://www.reddit.com/r/ClaudeAI/s/bO3cOogG6c).
- **B-STaR Paper Spotlights Self-Improvement**: Members discovered **B-STaR**: [Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners](https://arxiv.org/abs/2412.17256), championing advanced reasoning with minimal human annotation and a self-improvement training approach.
   - A user referenced the [Reddit thread](https://www.reddit.com/r/ClaudeAI/s/bO3cOogG6c) to highlight community discussions, suggesting these techniques could enable continuous refinement in future AI logic.
- **Gemini 2.0 Gains Grit**: Multiple members praised **Gemini 2.0** for flash-thinking and coding strengths, particularly its advantage over GPT-4 in speed and integrated usability.
   - They noted it may fill gaps left by OpenAI’s current line-up for specialized tasks, with talk of pushing beyond standard coding assistance.
- **Prompt Engineering & Sora Splits**: Calls for a dedicated **Sora** channel intensified, as users wanted more structure around advanced prompt engineering concepts for ChatGPT and related models.
   - Enthusiasts also sought formal **prompt engineering courses**, acknowledging how rapidly best practices can shift with evolving model updates.
- **Token Limits Trigger Tweaks**: Members wrestled with **GPT-2**’s 1024-token limit, while others faced feasibility issues generating lengthy blog posts through OpenAI’s APIs.
   - They discussed chunking content or sampling alternative models, referencing a [Discord post](https://discordapp.com/channels/974519864045756446/1315696747279810711/1323428129083097158) for approaches to address token constraints.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Audio Adventures**: The conversation covers reusing NotebookLM audio **publicly** with credit, referencing attempts with no adverse repercussions so far and a playful comment that *no one has been arrested yet.*
   - Some community members encountered inconsistent restrictions on posting [YouTube videos](https://www.youtube.com/watch?v=4rdXYdMmrFg) and [links](https://youtu.be/ubA36TeoyM4), attributing it to rate limiting or updated moderation settings.
- **Embedding NotebookLM for Interactive Impact**: Members proposed embedding **NotebookLM** on external sites to enable visitor queries, suggesting approaches like scraping or future API connections.
   - They also requested an *after the fact record* function to preserve critical snippets of a conversation, emphasizing a built-in recording feature for easier reviewing.
- **NotebookLM Plus Perks & Limits**: Many discussions focused on the **500**-notebook cap for Plus users versus **100** on free accounts, referring to [NotebookLM Help](https://support.google.com/notebooklm/answer/15678219) for clarity.
   - They also mentioned upload errors for MP3 files and coverage gaps in the resulting output, spotlighting system constraints that affect advanced usage.
- **Gemini 2.0 Podcast Quirks**: The [gemini-2-podcast repo](https://github.com/agituts/gemini-2-podcast) demonstrates Python scripts generating **Gemini 2.0**-based audio, although it ignores new files until the entire audio is deleted and re-rendered.
   - Others noted **NotebookLM** can skip or misread user sources, fueling interest in official APIs and mobile support to streamline cross-platform access.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **M2 Max MacBook Pro sparks performance debate**: Engineers questioned whether a **M2 Max MacBook Pro** with 32GB RAM and a 38-core GPU can tackle local AI workloads effectively, highlighting differences from Nvidia GPU setups.
   - Some found it usable, but others warned that truly heavy tasks could feel subpar on Apple's hardware.
- **Depth map fiasco annoys creators**: Users ran into **banding** artifacts when employing depth maps from 3D software, causing the model to interpret unintended edges.
   - They advised adjusting maximum depth levels and sticking to formats aligned with **Stable Diffusion** requirements.
- **LoRa training locks in consistent style**: A children’s book illustrator learned to maintain watercolor character designs by training a **LoRa** in **Stable Diffusion**.
   - They combined reference photos with specialized LoRa fine-tuning to achieve uniform illustrations.
- **AI video creation platforms draw curiosity**: Members explored cloud-based solutions like **Luma Dream Machine**, **Kling**, and **Minimax** for quick AI video testing.
   - They discussed cost factors, hardware demands, and shared [Webui Installation Guides](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides) plus [this YouTube walkthrough](https://www.youtube.com/watch?v=vY4QwnR4R2M).
- **Discord community wrestles with spam concerns**: Several users pushed for stronger **moderation** tools to counter bot activity and considered censorship implications on model outputs.
   - They worried that stricter safeguards could hinder character generation, especially when handling human anatomy.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Static Mojo vs Python Tradition**: Users debated the meaning and usage of **static methods** in **Mojo**, worried it might veer from Python's approach.
   - They proposed replicating Python's current behavior for consistency, citing the need to sync with existing **rebind** documentation at [Modular Docs](https://docs.modular.com/mojo/stdlib/builtin/rebind/rebind/).
- **Recursive Struct Showdown**: Defining **recursive structs** with `UnsafePointer[Self]` triggered segmentation faults in **Mojo**.
   - A switch to **ArcPointer** or **OwnedPointer** offered safer handling, though some overhead was unavoidable.
- **Mojo's 'Load' Trick for Faster SIMD**: Participants highlighted that using **load** is better than direct bitcast for handling SIMD data in **Mojo**.
   - They referenced [Performance Notes](https://www.computerenhance.com/p/table-of-contents), underlining how proper memory access is crucial for speed.
- **Pointers Parenting Woes**: Maintaining **child and parent pointers** in Mojo's recursive data structures tested users' patience.
   - They championed `OpaquePointer` as one method to sidestep pointer tangles and optional pointer pitfalls.
- **Debug Mode Takes a Dive (#3917)**: Running **Mojo** in full debug mode triggered segmentation faults, while normal runtime behaved better.
   - Developers noted [issue #3917](https://github.com/modularml/mojo/issues/3917) would be tackled after holidays, leaving the community waiting for a fix.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Speed Stampede**: Users reported up to **20x** faster performance hitting **6 t/s** using the [DeepSeek-V2.5-1210-GGUF model](https://model.lmstudio.ai/download/lmstudio-community/DeepSeek-V2.5-1210-GGUF) in LM Studio, with Perf Monitor tracking GPU usage.
   - They also referenced a [Nomic.ai blog post](https://www.nomic.ai/blog/posts/gpt4all-scaling-test-time-compute) about real-time scaling in on-device LLMs for **code interpreter** and **tool calling**.
- **Vision Models Check for Censorship**: A user discovered 'censored' **Vision Models** blocking NSFW content, prompting interest in uncensored approaches.
   - Likewise, they explored advanced functionalities and considered potential workarounds using *special configurations*.
- **3090 NV-Link & Noise Conundrum**: Community members debated **NV-Link** for dual **3090** setups, questioning if 2x2 bridging beats single cards while juggling longer cables.
   - Others warned about **blower fans** reaching **83 dB**, suggesting **water cooling** to mitigate noise when running *inference tasks*.
- **Jetson Orin Nano’s 25W Trials**: A user tested a **Jetson Orin Nano** with 20 models in **25W mode**, citing [a blog post](https://www.jeremymorgan.com/blog/tech/nvidia-jetson-orin-nano-speed-test/) for real-world speed data.
   - Debate followed on *quantizing models* and optimizing watts-per-token for more compact or edge-based LLM deployments.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **TMA Takes on cp.async**: Participants showed how **TMA** can outperform **cp.async** by enabling fewer threads and using fewer registers, thereby cutting resource overhead.
   - They highlighted potential boosts for HPC tasks and pointed to [this GEMM series on Hopper GPUs](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) for related examples.
- **Power-of-2 Drives MAGVIT-v2**: Community members explained how **MAGVIT-v2** leverages binary quantization, encoding decimals like 9 as [0][1][0][0][1][0] to represent powers of two.
   - They referenced **Dominika Przewlocka-Rus**'s work suggesting alignment with Laplacian distributions, spurring more conversation on potential bit-shift performance gains.
- **ThunderKittens vs Triton Tussle**: Members announced **ThunderKittens** will add integer matmul operators, illustrating ongoing experimentation with custom kernels.
   - They debated whether a carefully tuned **TK/CUDA** kernel can outpace **Triton**, citing constraints in Triton's fine-grained async execution and register handling.
- **Raspberry Pi 5 GPU Trials**: Enthusiasts reported that the **Raspberry Pi 5** GPU shows promise with smaller vision workloads despite limited raw compute power.
   - They saw slow performance on larger **LLMs** using 6–8bit quantization, prompting questions about **Vulkan** benchmarks and comparisons to Intel CPUs.
- **Cracked Tech Jobs in GPU Land**: A shared [cracked research engineer job](https://crackedengineers.com/job/p-1-ai-7f41fa30-6cfa-4e9a-8943-2324dc21d243) highlighted specialized roles in GPU and AI development.
   - The group advised searching for **CUDA** and **Triton** keywords, reflecting growing demand for advanced GPU expertise.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **On-Call Chaos: AI Code Woes**: One user pointed to [this tweet from Shreya Shankar](https://x.com/sh_reya/status/1873431565650502060) about burdens on on-calls caused by AI-generated code, urging better documentation and testing.
   - Others suggested that devs break tasks into smaller steps so **LLMs** can manage them effectively, rather than tackling entire complex features blindly.
- **Kagi Clash: Searching for an Edge**: Users praised **Kagi Assistant** for its flexible search capabilities, although some noted coverage gaps compared to **Perplexity**.
   - Enthusiasts look forward to upcoming features including a search API, anticipating stronger competition with similar tools.
- **Summit Sparks: 2025 AI Engineering Meetup**: An **AI Engineering Summit** is set for February 20-21, 2025 in New York, reportedly backed by major tech sponsors in prior events.
   - Organizers encourage early pre-registration for special access, promoting a gathering of AI professionals and industry leaders.
- **Cursor Conundrum: Collaboration or Chaos?**: Multiple devs shared frustration with the **Cursor** AI coding assistant, describing wasted effort during complex coding tasks.
   - They advised clarifying instructions and using iterative problem statements to reduce friction when pairing with AI tools.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Tie at the Top: Chatbot Arena**: Chatbot Arena sees **OpenAI's o1** jump to a joint #1 spot, earning +24 points from o1-preview and passing other contenders like [DeepSeek-V3 at #7](https://x.com/lmarena_ai/status/1873695386323566638).
   - Community chatter highlights **Claude's lower ranking** as perplexing, with refusals and roleplay issues cited as possible reasons.
- **SLMs Contradict The Bitter Lesson**: A debate emerged on how **smaller language models** can excel in targeted tasks by using specialized priors, questioning the push for more data and compute.
   - Participants referenced **Llama 3 8B** surpassing GPT-3 175B and underscored the importance of domain-specific solutions.
- **DeepSeek V3: XML Output Woes & Benchmarks**: Members shared frustration that **DeepSeek V3** struggles to output **XML** tags correctly, producing r1-like reasoning instead of fulfilling instructions.
   - They also questioned its instruction-following performance after prompt swaps from V2.5, noting negative feedback on post-training results.
- **GRPO vs. Vineppo: RLHF Rivalry**: Discussion centered on **GRPO (Group Relative Policy Optimization)** and its averaging of rewards, contrasted with vineppo's single-sample strategy and mid-episode resets.
   - A user explained that **DeepSeek V3** uses GRPO, raising concerns about memory limits with 1b–7b models and the possibility of dropping a value network.
- **Gary & Miles Bet on AI's 2027 Trajectory**: Community responded to a [Gary Marcus post](https://garymarcus.substack.com/cp/153809626) revealing his joint wager with Miles Brundage on future AI achievements.
   - Skeptical remarks included claims that we remain 'insanely far away from 4,' signaling caution about near-term leaps in model capability.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **LLaMA 3.3 in GPT4All Gains Groq Key**: Users shared steps for hooking up **LLaMA 3.3** (70B) with **GPT4All** through [Groq.com](https://groq.com/) to enable cloud LLM support.
   - They highlighted the cost benefits, noting it spares on-prem hardware overhead for AI workloads.
- **Gemini API Support Sparks Excitement**: Participants discussed **Gemini** compatibility with OpenAI’s API and the roadmap for **Gemini 2.0**, citing [google-gemini/cookbook](https://github.com/google-gemini/cookbook).
   - They expressed interest in using Gemini’s unique capabilities once official GPT4All integration is confirmed.
- **Jinja Jitters Trigger Chat Template Woes**: Recent GPT4All updates introduced **Jinja** parsing that caused syntax breakage for older chat templates.
   - Contributors suggested resetting default templates or referencing updated files, encouraging collaborative fixes.
- **Vision Embeddings Come Into Focus**: Members clarified that **nomic-embed-vision-v1** pairs with text embedding models to refine image searches via text queries.
   - They compared Nomic’s vision model to other publicly available options, expecting more robust demos in future releases.
- **Ollama Model Exports Spark Talk**: Enthusiasts explored reusing **Ollama** models in GPT4All, referencing the [Ollama Model Export Script](https://gist.github.com/supersonictw/f6cf5e599377132fe5e180b3d495c553).
   - They discussed designating Ollama as the LLM engine, pointing to the compatibility it shares with OpenAI-style APIs.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Breathe.ai Signs NDA to Test Cohere**: Breathe.ai officially joined Cohere via an NDA, aiming to collaborate on a research prototype.
   - Members welcomed them enthusiastically, sharing hopes for deeper technical exchanges and feedback loops.
- **HMM Tokenization Queries Spark Curiosity**: Several users asked about **HMM (Hidden Markov Model)** tokenization techniques, highlighting a gap in shared expertise.
   - No immediate advice surfaced, revealing an interest in expanding knowledge on advanced NLP tokenization methods.
- **Cohere's Rate Limit Ruckus**: Members encountered a mismatch in expected image embed rate limits, anticipating **400** calls per minute but observing **40**.
   - The support team confirmed the [rate limit documentation](https://docs.cohere.com/v2/docs/rate-limits) and assured a fix is in progress, reiterating the official cap remains **400** for production keys.
- **Fine-Tuning Firefight Continues**: A user reported **fine-tuning** errors, concerned about potential data or configuration issues.
   - Support is investigating delays caused by holidays, promising direct communication and escalating the troubleshooting process.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Magnificent Matching Speedup**: The claim of an **8x speedup** in matching functions sparked intense discussion, citing a bounty bridging **400ms** down to **50ms** as a target.
   - Skeptics noted that **50%** of runtime lies in these functions, spurring talk of how even **2x** acceleration might be the more realistic goal.
- **Rewrite Rumble: 2.5x Gains, 4/7 Grief**: A tweak to `full_graph_rewrite` yielded a **2.5x** boost in model rewrite times, though **4/7** tests promptly broke and called for urgent debugging.
   - Multi-threading emerged as one angle for improvement, alongside smaller test sets for zeroing in on the root issues.
- **AM Driver Marathon Aims for 11k Lines**: George Hotz pledged to expand the **AM driver** to **11,000** lines and merge it by year’s end, referencing [this commit](https://github.com/tinygrad/tinygrad/commit/0addbad36d414cc37e69e92fa9e1f26045cbf1f6) as a sign of progress.
   - Attendees anticipate **Meeting #51** at **930am Monday** in San Diego to slash technical debt on scheduler cleanups and push the AM driver onward.
- **Tinygrad CUDA Crushes Torch**: New benchmarks suggest **Tinygrad CUDA** is nearly **twice** as quick as Torch, with **OpenCL** slicing about **1ms** off overhead.
   - The devs recommended using `Device[out.device].synchronize()` to get precise metrics, noting that **JIT** speed really kicks in on the **third run**.
- **Frame Evaluation Hook Buzz**: Community members highlighted the **Frame Evaluation Hook API** from [PEP 523](https://peps.python.org/pep-0523/) as a handy way to capture runs directly in Python.
   - They pointed out that Torch’s dynamo compiler relies on this approach, calling it more flexible than post-capture solutions.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Local Llama-3.2 & Neomagus Secure Legal Citations**: Developers discussed building a local RAG app with **Llama-3.2** using [Llama Index tools](https://t.co/4WJ7ZcXy3H) to query **Excel tables** seamlessly.
   - They also highlighted **Neomagus** for verifying references in AI-generated text, with details shared [here](https://t.co/g5toC0m3T9), hoping to reduce false citations.
- **Llama 3.3 GPU Footprint & Ollama's Role**: One user inquired about **Llama 3.3 70B** GPU requirements, referencing a potential **Hugging Face** endpoint.
   - Another user tested **Ollama** locally and saw about **2.77GB** of RAM usage running `ollama run llama3.3`, indicating a more memory-friendly approach.
- **Bagel Bakes Monetization for Open Source AI**: A representative unveiled **Bagel**, a platform that helps **open source AI developers** earn income and sync with **Hugging Face**.
   - They shared a [tweet](https://x.com/BagelOpenAI/status/1873776090516488257) explaining how this novel architecture keeps developers in control while providing advanced models like **Llama-3.3**.
- **Filtering Nonword Sounds for Audio Clarity**: A user explored *ahh* and *um* removal using LLMs, sparking interest in refining **audio editing** workflows.
   - Participants noted that cleaning up filler words could enhance the **listening experience** for educational and professional recordings.
- **LlamaParse API Accelerates Data Manipulation**: Members discussed the **LlamaParse API** for direct integration, showcasing sample calls for uploading and checking parse jobs in [official docs](https://docs.cloud.llamaindex.ai/llamaparse/getting_started/api).
   - They emphasized the advantage of handling structured data seamlessly, referencing [GitHub examples](https://github.com/run-llama/llama_parse/tree/main/examples) for real RAG scenarios.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents MOOC Reopens for Enrollment**: The next **LLM Agents** course starts in late January, offering sign-ups via [this form](https://forms.gle/9u6HdVCWXgws16go).
   - Enrollees can reference the upcoming [Spring 2025 syllabus](https://llmagents-learning.org/sp25) as well as the [Fall 2024 materials](https://llmagents-learning.org/f24) for a head start.
- **Certificate Emails Coming in January**: Certificates from the earlier **LLM Agents** MOOC will be emailed by the end of January, though some participants are still waiting.
   - Members confirmed they can access [the course website](https://llmagents-learning.org/f24) to revisit lecture materials while they wait.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Dynamo Drama Diminishes**: Reports indicate **Dynamo errors** may be resolved, prompting members to consider removing compiler-disabled settings for better performance.
   - One user recommended verifying speed-ups with both compile modes enabled and disabled, stressing thorough **regression checks**.
- **Flex's Next Frontier Arrives Jan 13**: Members anticipate **Flex** updates in the upcoming **2.6.0** release on **January 13**, expecting improvements beyond **2.5.1**.
   - They noted multiple adjustments had been introduced, hoping these modifications would be integrated before final release.
- **Simple Eval vs LM Eval Showdown**: A member spotlighted [OpenAI's Simple Eval library](https://github.com/openai/simple-evals) as a potential alternative to **lm eval** tools.
   - Debate centered on **evaluation** speed and compatibility, with participants reviewing the GitHub page for specific implementation details.
- **FP8 Feats Propel Transformer Engines**: Users discussed **FP8 quantization** tactics, referencing [NVIDIA's Transformer Engine](https://github.com/NVIDIA/TransformerEngine) and [Microsoft's Automatic Mixed Precision Library](https://github.com/Azure/MS-AMP).
   - They also highlighted 2D block quantization approaches, citing [COAT](https://github.com/NVlabs/COAT), PyTorch's [Float8 GEMMs blog](https://pytorch.org/blog/accelerating-gemms-triton/), and *mixed-precision training* papers like [arXiv:2310.18313](https://arxiv.org/pdf/2310.18313) and [arXiv:2409.12517](https://arxiv.org/pdf/2409.12517).



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OS Mode: Video or No?**: A user asked if **OS mode** can accept **video** as input, hoping for clarity on its scope.
   - **No confirmed** solution emerged, but there's growing curiosity about multimedia support.
- **Isolation Indecision: Docker vs. OS**: Users pointed to [the Isolation doc](https://docs.openinterpreter.com/safety/isolation) and wondered if it governs operating system locks or **Docker** and **E2B** usage.
   - An attached image fueled confusion, suggesting ambiguous terminology in the doc.
- **Windows 1.0: Build Me Up**: Someone asked about a **Windows build** for the newly released **1.0 dev** version.
   - Cross-platform fans await support to confirm if broad OS compatibility is coming.
- **The Great Profile Swap: YAML to PY**: Users encountered trouble moving from **profiles.yaml** in **1.0.0** to the new **.py** format.
   - They questioned documentation accuracy, worried about saving processes.
- **Custom API Base URL Woes**: A user hoped to replicate **OpenAI**-style usage with endpoints like **gpt4o** or **claude-35-sonnet** on Ubuntu.
   - They ran into setup hurdles and requested help adapting these custom base URLs.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Arxiv 2412.15563 Gains Eyeballs**: One user asked for opinions on [Arxiv Paper 2412.15563](https://arxiv.org/abs/2412.15563), seeking clarity on its broader ramifications for large language models.
   - No direct analysis was offered, but there's interest in seeing if it might suit **DSPy** experiments.
- **AI Glossary Gains Momentum**: A member introduced an **AI Glossary** to speed up concept references, citing [Generating a Glossary from a Jekyll Blog Using DSPy & Claude](https://www.dbreunig.com/2024/12/27/generating-a-glossary-from-a-jekyll-blog-usign-dspy-claude.html) as inspiration.
   - They emphasized the interplay between language and technology, noting a backlog of terms still awaiting sharper definitions.
- **Openhands Hooks onto DSPy**: A question arose about making **Openhands** a one-shot noninteractive tool that returns chat responses and git diffs, fueling discussion on integrating it into **DSPy's** pipeline.
   - They recognized potential synergy but pointed out design nuances in how **DSPy** handles prompt tuning and automation.
- **Feedback System Sparks Code Curiosity**: A user proposed a system to record feedback on automated code changes for later evaluation, focusing on input/output logging.
   - They plan to use these data points to guide a **DSPy** pipeline that refines code quality based on historical outcomes.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **FFmpeg Slicing Gains Traction**: One user described a method to gather time stamps then apply **FFmpeg** to cut video content, praising the clarity of instructions.
   - They voiced satisfaction with the process, calling it *a straightforward approach for swift editing.*
- **Hackathon & Conference Fever in 2025**: Someone is seeking suggestions for 2025 hackathons and conferences, already set on **ICML**, **NeurIPS**, and **CVPR**.
   - They want to meet more community members and eagerly invite more ideas.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Leaderboard Zero-Shot Conundrum**: They clarified that recognized models must be tested in a **zero-shot** environment, yielding a single response with no iterative calls.
   - An **API endpoint** approach can bypass typical restrictions if the user only calls once, referencing **OpenAI’s o1** chain-of-thought logic behind an API.
- **Single-Call for Score Security**: They stressed that advanced chain-of-thought expansions must remain invisible to the user, enforcing only one **API call** for leaderboard evaluations.
   - This mechanism keeps the leaderboard consistent by disallowing multi-step generation or repeated attempts within a single evaluation.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Axolotl AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1322681294974353448)** (1 messages): 

> `Codeium 2024 Wrapped, Upcoming features` 


- **Codeium 2024 Wrapped Launch**: The team announced the release of the **Codeium 2024 Wrapped**, inviting users to check and share their stats at [this link](https://codeium.com/wrapped-2024).
   - Excitement filled the channel as the team thanked everyone for an incredible **2024**, hinting at more features to come.
- **Looking Forward to New Year Features**: The message emphasized a commitment to shipping more **features** to enhance the user experience in the new year.
   - *Lots of work left to do*, according to the announcement, as they prepare to make further improvements in **2025**.



**Link mentioned**: <a href="https://codeium.com/wrapped-2024">Codeium Wrapped 2024 | Windsurf Editor and Codeium extensions</a>: Check out your top languages, how much time you spent coding, your coding patterns and much more in Codeium 2024 Wrapped!

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1322304733368680489)** (194 messages🔥🔥): 

> `Windsurf performance issues, User login problems, Codeium pricing frustrations, Alternative IDEs, Error messages in Codeium` 


- **Windsurf struggles with performance and downtime**: Users are reporting slow performance and frequent outages with **Windsurf**, leading to frustrations with wasted credits and login issues.
   - Many are considering alternatives like Aide and Cody while Codeium addresses the server overload.
- **Login difficulties with Codeium**: One user expressed frustration with not being able to log into their account despite reinstalling the application and trying various troubleshooting steps.
   - Suggestions from others included force closing the application and checking the operating system settings.
- **Concerns over Codeium's credit system**: Several users are unhappy with how quickly their **premium credits** are being depleted, especially after recent changes to the system.
   - There are calls for potential refunds due to unanticipated issues leading to excessive credit usage.
- **Discussion of possible alternatives to Windsurf**: With ongoing issues, users are exploring alternatives like **ChatGPT 4o** and other open-source tools as temporary solutions.
   - Some share skepticism about the effectiveness of these alternatives compared to Windsurf.
- **Errors and messaging not being returned by Codeium**: Users report errors when trying to interact with the chat feature in Codeium, leading to repeated questions without responses.
   - Many suggest starting new chats or restarting the application as potential solutions to clear up responsiveness issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://codeium.com/windsurf/show-auth-token?authType=signin&from=redirect">Provide Authentication Token to VSCode | Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://codeium.com/settings">Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1322295225095553100)** (633 messages🔥🔥🔥): 

> `Windsurf service outages, DeepSeek V3 integration, Context length issues in Windsurf, User experiences with AI code suggestions, SVG loading issues in React Native` 


- **Windsurf service outages continue**: Users reported frequent service outages with Windsurf, experiencing 503 errors and slow response times during high usage periods.
   - This has led to frustration among users, with many suggesting the need for a status page to monitor service availability.
- **DeepSeek V3 yet to be integrated**: There are ongoing discussions regarding the integration of DeepSeek V3 into Windsurf and Cursor, with users expressing impatience for its implementation.
   - Similar tools like Cline have managed to integrate it more quickly, raising questions about the prioritization of new features.
- **Context length confusion in Windsurf**: There was a discussion regarding the context length used by Codeium and how it relates to Windsurf, with users confused about limitations.
   - While it was suggested that Codeium offers a high context length, users indicated challenges with maintaining context during code revisions.
- **Frustrations with AI code suggestions**: Several users expressed frustration with AI code suggestions from Sonnet, noting issues with unwanted refactorings and complicated prompts.
   - Suggestions included focusing on specific coding tasks and using project instructions effectively to improve the quality of responses.
- **SVG loading issues in React Native**: A user reported issues with loading SVG icons in React Native native simulators, which contrasts with successful web previews.
   - They suspect version compatibility issues between React Native, native-svg, and Expo as potential causes of the problem.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/advanced">Windsurf - Advanced</a>: no description found</li><li><a href="https://tenor.com/view/nounish-nounsdao-nouns-dao-noggle-gif-26326389">Nounish Nounsdao GIF - Nounish Nounsdao Nouns - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/roblox-roblox-outage-blox-fruits-update-gif-14794492378318604921">Roblox Roblox Outage GIF - ROBLOX ROBLOX OUTAGE BLOX FRUITS UPDATE - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.codeium.com/getstarted/overview">no title found</a>: no description found</li><li><a href="https://tenor.com/view/bored-crashing-faceplant-sleepy-pass-out-gif-8482195">Bored Crashing GIF - Bored Crashing Faceplant - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://code.visualstudio.com/docs/devcontainers/containers">Developing inside a Container using Visual Studio Code Remote Development</a>: Developing inside a Container using Visual Studio Code Remote Development</li><li><a href="https://chat.deepseek.com/">DeepSeek</a>: Chat with DeepSeek AI.</li><li><a href="https://codeium.com/plan">Plan Settings</a>: Tomorrow&#x27;s editor, today. Windsurf Editor is the first AI agent-powered IDE that keeps developers in the flow. Available today on Mac, Windows, and Linux.</li><li><a href="https://tenor.com/view/clapping-leonardo-di-caprio-gif-13334985">Clapping Leonardo Di Caprio GIF - Clapping Leonardo Di Caprio - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/whats-going-on-down-there-concerned-whats-going-on-gif-15556206">Whats Going On Down There Concerned GIF - Whats Going On Down There Concerned Whats Going On - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/no-brain-loading-slow-gif-8465847256202919615">No Brain Loading GIF - No Brain Loading Slow - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/zerosgifs-gif-20855209">Zerosgifs GIF - Zerosgifs - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://codeium.com/live/">Chat with Codeium | Windsurf Editor and Codeium extensions</a>: Chat with general using Codeium Live. Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.</li><li><a href="https://stats.uptimerobot.com/aiH8grJl1y">Status page</a>: no description found</li><li><a href="https://github.com/unixwzrd/venvutil">GitHub - unixwzrd/venvutil: Python virtual environment management functions and script to build and manage performance, compatibility, and regression test venv builds mostly for AI</a>: Python virtual environment management functions and script to build and manage performance, compatibility, and regression test venv builds mostly for AI - unixwzrd/venvutil</li><li><a href="https://codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.</li><li><a href="https://codeium.com/blog/termium-codeium-in-terminal-launch">Termium: Codeium in the Terminal</a>: AI-powered autocomplete for your terminal commands.</li><li><a href="https://www.eridepros.com/">Race Ready Off-Road Electric Moto | E Ride Pro </a>: “Shop high-performance e-motos at E Ride Pro. Discover eco-friendly, fast, and durable electric off-road bikes for all skill levels.”
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1322309152646762596)** (705 messages🔥🔥🔥): 

> `Fine-tuning LLM Models, Role of Tokens in Training, Open Source and Model Sharing, Quantization Issues with LLMs, Hymba Model Overview` 


- **Fine-tuning LLM Models**: Several users discussed strategies for fine-tuning language models, emphasizing the importance of properly structured datasets and the need for early stopping mechanisms.
   - The conversation highlighted the challenges faced when fine-tuning, including the potential risk of overfitting with too high of a learning rate.
- **Role of Tokens in Training**: Sadaisystems raised the question of the impact of training models with specific token formats, like XML, on model performance and understanding.
   - It was noted that models may recognize custom tokens during inference, but training is crucial for building effective related weights.
- **Open Source and Model Sharing**: Participants discussed the challenges of open-source software, particularly regarding power concentration and how it relates to the distribution of advanced AI technology.
   - Concerns were raised about legalities and ethical considerations in the open-source community, emphasizing the need to respect licenses.
- **Quantization Issues with LLMs**: Renegade2611 reported issues with Llama.cpp for quantization, noting errors encountered during integration that may be linked to recent updates to the library.
   - There was also discussion on the lack of compatible unsloth quantization for larger models like Phi 4, which has yet to be released due to operational delays.
- **Hymba Model Overview**: The Hymba-1.5B-Instruct model was introduced, highlighting its capabilities and the fact that it's ready for commercial use with specific batch size requirements.
   - Details were shared regarding its development from base models utilizing open-source instruction datasets and the importance of understanding its limitations during generation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing#scrollTo=6bZsfBuZDeCL">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=juQiExuBG5Bt">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing#scrollTo=6bZsf">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint#wandb-integration">Finetuning from Last Checkpoint | Unsloth Documentation</a>: Checkpointing allows you to save your finetuning progress so you can pause it and then continue.</li><li><a href="https://www.youtube.com/@WillCogley">Will Cogley</a>: Will Cogley is all about fusing mechanics, electronics and a little artistic creativity to make top-notch robotics and animatronics. These creations are carefully documented and published so that anyo...</li><li><a href="https://huggingface.co/nvidia/Hymba-1.5B-Instruct">nvidia/Hymba-1.5B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://www.kaggle.com/code/ahmedess/llama-3-2-vision-finetuning-unsloth-kaggle">Llama 3.2 Vision Finetuning Unsloth - Kaggle</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources</li><li><a href="https://unsloth.ai/blog/contpretraining">Continued LLM Pretraining with Unsloth</a>: Make a model learn a new language by doing continued pretraining with Unsloth using Llama 3, Phi-3 and Mistral.</li><li><a href="https://www.anninrobotics.com/">Annin Robotics</a>: Annin Robotics -Open source affordable robots - build your own 6 axis robot.</li><li><a href="https://x.com/danielhanchen/status/1872719599029850391">Tweet from Daniel Han (@danielhanchen)</a>: Cool things from DeepSeek v3&#39;s paper:1. Float8 uses E4M3 for forward & backward - no E5M22. Every 4th FP8 accumulate adds to master FP32 accum3. Latent Attention stores C cache not KV cache4. No M...</li><li><a href="https://docs.beam.cloud/v2/environment/custom-images#conda-environments">Container Images - Beam</a>: no description found</li><li><a href="https://www.stephendiehl.com/posts/unsloth/">A Rapid Tutorial on Unsloth</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/main/en/developer_guides/lora#eva">LoRA</a>: no description found</li><li><a href="https://gist.github.com/grahama1970/f832bbddb1edaa78ccc939a6f2ddd8a1">For dynamic adaptor loading and inferencing, the Unsloth Inference works fine--using Hugging Face does not work--outputs garbled</a>: For dynamic adaptor loading and inferencing, the Unsloth Inference works fine--using Hugging Face does not work--outputs garbled - hf_only_inference_sanity_check.py.py</li><li><a href="https://youtu.be/7pdEK9ckDQ8?feature=shared&t=31"> - YouTube</a>: no description found</li><li><a href="https://github.com/vllm-project/llm-compressor?">GitHub - vllm-project/llm-compressor: Transformers-compatible library for applying various compression algorithms to LLMs for optimized deployment with vLLM</a>: Transformers-compatible library for applying various compression algorithms to LLMs for optimized deployment with vLLM - vllm-project/llm-compressor</li><li><a href="https://github.com/confident-ai/deepeval">GitHub - confident-ai/deepeval: The LLM Evaluation Framework</a>: The LLM Evaluation Framework. Contribute to confident-ai/deepeval development by creating an account on GitHub.</li><li><a href="https://github.com/unslothai/unsloth/issues/1333">Issue training with Qwen 2.5 7B · Issue #1333 · unslothai/unsloth</a>: from trl import SFTTrainer from transformers import TrainingArguments, DataCollatorForSeq2Seq from unsloth import is_bfloat16_supported trainer = SFTTrainer( model = model, tokenizer = tokenizer, t...</li><li><a href="https://robotnanohand.com/">Home</a>: Introducing the Robot Nano Hand open source project. 3d print, build and program this state of the art humanoid robotic hand.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1322980034763554989)** (8 messages🔥): 

> `WSL Ubuntu setup, Community Gratitude, Computer Vision Projects, New Year Wishes, Server Appreciation` 


- **Runtime Error on WSL Ubuntu**: A user encountered a **RuntimeError** when trying to save a model on WSL Ubuntu, indicating missing files in the **llama.cpp** directory.
   - After troubleshooting, they resolved the issue by installing **curl** and necessary libraries via `apt-get`.
- **New Year Wishes and Community Support**: A member expressed gratitude towards the Unsloth Discord community for their learning experiences and wished everyone good health for the **New Year**.
   - Another member responded with enthusiasm, echoing sentiments of appreciation.
- **Aspirations in Computer Vision**: One member shared their recent focus on **computer vision** and expressed hopes of working on fine-tuning by **2025**.
   - This enthusiasm reflects a commitment to progressing in the field despite the timeline.
- **Enhanced Support for the Community**: A user expressed strong support for Jed.T, highlighting the exceptional nature of the Discord server and the **Unsloth framework**.
   - This reflects a growing sense of community and collaboration among its members.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1322348282172801035)** (171 messages🔥🔥): 

> `LoRA and its applications, Fine-tuning large language models, Challenges in language translation, Understanding model performance and training datasets, Learning resources for AI and LLMs` 


- **Navigating Efficacy of LoRA in Pretraining**: A member queried whether leveraging **LoRA** for large-scale pretraining aids in a model's retention of new knowledge, to which another member expressed skepticism about its reliability.
   - *Some shared previous experiences*, emphasizing a careful approach to expectations around performance.
- **Pitfalls of Fine-Tuning for Language Translation Models**: A participant expressed frustration over inconsistent translation results when fine-tuning the **Llama 3.1 8B** model for a new language, questioning the efficacy of continued pretraining.
   - Another contributor highlighted the inherent challenges, emphasizing that fundamental knowledge of the language data is crucial for reliable translation capabilities.
- **Learning Resources for Aspiring AI Developers**: **New developers** were advised on where to start in AI, with a focus on exploring existing AI documentation from OpenAI and Gemini, alongside understanding the historical evolution of LLMs.
   - Participants discussed the importance of understanding foundational concepts before diving into specific implementations in AI and LLM applications.
- **Exploring Effectiveness of Fine-Tuning on Instruct Models**: In discussions about fine-tuning Instruct models versus base models, it was mentioned that **pretraining** a base model is often more beneficial for certain applications.
   - Members agreed that the differences in training methodologies can lead to varying effectiveness depending on the nuance and amount of data available for specific use cases.
- **Understanding Cut Cross Entropy Implementation**: A technical overview of **cut cross entropy** demonstrated its automatic enabling under specific conditions in the Unsloth library, with snippets showing how this is coded.
   - The discussion revealed the integration of `fused_linear_cross_entropy` in the model's inference functions, contributing to potential performance improvements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/phi4">Finetune Phi-4 with Unsloth</a>: Fine-tune Microsoft&#x27;s new Phi-4 model with Unsloth! Open-source and beginner friendly.</li><li><a href="https://arxiv.org/abs/2401.13586">Instruction Fine-Tuning: Does Prompt Loss Matter?</a>: We present a novel study analyzing the effects of various prompt loss token weights (PLW) for supervised instruction fine-tuning (SIFT). While prompt-masking (PLW = 0) is common for SIFT, some fine-tu...</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-8.-multi-turn-conversations">Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation</a>: Beginner&#x27;s Guide for creating a customized personal assistant (like ChatGPT) to run locally on Ollama
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1322784435128635474)** (7 messages): 

> `Light Prompter, Test Time Training, Weights Updating, RL Techniques, VLLM Notebooks` 


- **Light Prompter Accelerates Test-Time Compute**: The [Light Prompter](https://github.com/Green0-0/light_prompter) GitHub repository focuses on **accelerating test-time compute** using batching techniques with included notebooks.
   - This project aims to increase efficiency during model inference with relevant contributions from the community encouraged.
- **Inquiries About Test Time Training**: A member posed a question about **test time training**, specifically whether it involves updating the model weights during inference.
   - The discussion hinted at the need for more research, suggesting that reading a related paper could be beneficial.
- **Discussion on RL Techniques for Training**: There was a suggestion that test time training might relate to **reinforcement learning (RL)** methodologies.
   - Another member speculated that similar approaches should exist and hinted at the possibility of finding available code or research.



**Link mentioned**: <a href="https://github.com/Green0-0/light_prompter">GitHub - Green0-0/light_prompter: Accelerate test-time-compute with batching!</a>: Accelerate test-time-compute with batching! Contribute to Green0-0/light_prompter development by creating an account on GitHub.

  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1322294311169036358)** (637 messages🔥🔥🔥): 

> `Cursor IDE issues, Deepseek API usage, Chat vs Composer, Web app development, Payment methods for Cursor` 


- **Cursor IDE becomes less effective**: Users report frustrations with Cursor, noting a decline in AI capabilities and responsiveness, leading some to consider alternatives or return to previous versions.
   - Many have experienced repeating issues or errors, often requiring restarts to resolve functional degradation throughout the day.
- **OpenAI API in Cursor.**: Users inquire about utilizing the OpenAI API within Cursor, discussing limitations and experiences with different models.
   - Some users find better results with Claude compared to the latest OpenAI offerings, suggesting a lack of improvement in newer models.
- **Web App Development with Cursor**: Users share experiences of developing web apps with Cursor, highlighting ease of use for those with limited coding knowledge.
   - One user successfully launched a web tool for a mobile MMO game, demonstrating that Cursor can be effective for building apps without extensive programming expertise.
- **Composer vs Chat Functionality**: The Composer tool is praised for its ability to iterate and fix code, while some users still find value in the Chat functionality.
   - Users discuss how treating the AI as an assistant can lead to better outcomes, suggesting that cursing or expressing frustration may prompt better responses from Cursor.
- **Payment Challenges for Cursor**: Users face difficulties using various payment methods for Cursor, often citing issues with localization and bank restrictions.
   - Challenges with payment processing lead some to seek alternative methods, indicating a pressing need for more accessible transactions on the Cursor platform.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/sonic-sonic-the-hedgehog-tails-knuckles-freaky-gif-12830799583270855342">Sonic Sonic The Hedgehog GIF - Sonic Sonic the hedgehog Tails - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.hiro.so/blog/write-better-smart-contracts-with-the-programming-language-clarity">Write Better Smart Contracts With the Programming Language Clarity</a>: Discover why the Clarity programming language is so well designed for writing smart contracts.</li><li><a href="https://x.com/code/status/1872673862992744625">Tweet from Visual Studio Code (@code)</a>: Claude 3.5 Sonnet, directly in @codeAvailable to everyone today with GitHub Copilot Free. Learn more: http://aka.ms/copilot-free</li><li><a href="https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221">How to do `Fix in Composer` and `Fix in Chat` actions from keyboard</a>: These 2:     I could not find it in settings.</li><li><a href="https://forum.cursor.com/t/are-claude-3-5-sonnet-and-claude-3-5-sonnet-20241022-different/24272/3">Are claude-3.5-sonnet and claude-3.5-sonnet-20241022 different?</a>: quick update here: claude-3.5-sonnet now points to claude-3-5-sonnet-20241022!</li><li><a href="https://guitar-tab.vercel.app/en/tools/fretboard">Guitar Chord Learning App</a>: Master guitar chords with our interactive learning tool!</li><li><a href="https://guitar-tab.vercel.app/en/tools/scales">Guitar Chord Learning App</a>: Master guitar chords with our interactive learning tool!</li><li><a href="https://guitar-tab.vercel.app/en/tools/tuner">Guitar Chord Learning App</a>: Master guitar chords with our interactive learning tool!</li><li><a href="https://coffeethencode.dev/dectalk/">DECTalk Generator</a>: no description found</li><li><a href="https://twomgg.onrender.com/">TwomGG</a>: no description found
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1323331159878275114)** (1 messages): 

> `Grok AI API promotion` 


- **Last Chance for Grok AI Promo**: It's the final two days of Grok AI's $25 free credits promo for API users, with a deadline fast approaching as the year ends. [Check the promo details here](https://x.com/stackblitz/status/1873769044761022633) before the credits vanish!
- **Opportunity to Experiment with Grok AI**: Members emphasized that today and tomorrow are THE perfect times to experiment with Grok AI API as part of building it into your Bolt app.



**Link mentioned**: <a href="https://x.com/stackblitz/status/1873769044761022633">Tweet from StackBlitz (@stackblitz)</a>: Build #GrokAI into your Bolt app!If you haven&#39;t tried it yet, today & tomorrow are THE time for it:before the year ends, every x․ai API user still gets $25 of free credits!

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1322363289677860976)** (20 messages🔥): 

> `Bolt code update issues, Voice prompting feature request, Token wastage concerns` 


- **Bolt code updates often fail**: Many users reported that Bolt stops making visible changes to the website despite generating code, leading to frustration in ongoing projects.
   - Some suggested rolling back checkpoints or using the visualizer to help address the issue, though this only worked temporarily.
- **Request for voice prompting feature**: There was a strong interest in adding a voice prompting feature like ChatGPT to facilitate easier communication while building projects.
   - However, users were cautioned that implementing such a feature could be costly due to the complexity of audio models compared to chat models.
- **Frustration over token wastage**: Several users expressed concerns about high token costs associated with prompting issues in Bolt, particularly when prompts do not yield the expected results.
   - Requests were made for a feature to allow prefixed instructions to minimize repeated prompts and save tokens.
- **LLM laziness observed in Bolt**: A user highlighted that LLMs, including Bolt, tend to become less responsive when models handle large codebases, leading to unexpected changes in non-relevant files.
   - Suggestions included reloading projects and enabling diff mode, which reportedly helps mitigate the laziness issue.


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1322300689090609346)** (460 messages🔥🔥🔥): 

> `Token Consumption, Error Handling in Bolt, Using Bolt for App Development, Firebase vs Supabase, Project Management in Bolt` 


- **Concerns Over Token Consumption**: Users express frustration over fast token consumption in Bolt, with estimates varying widely based on project size and user prompting skills.
   - Many suggest that as projects grow, AI becomes less capable, requiring more precise prompts to avoid unnecessary token usage.
- **Error Handling and Debugging**: Multiple users report encountering issues with errors stemming from migration problems and code modifications by Bolt, leading to increased token costs.
   - Some suggest using external tools like Google Gemini for error explanations and revisions, while others warn about the limitations of Bolt's current feedback mechanisms.
- **Integrating with External Tools**: Users are exploring the workflow of using Bolt alongside StackBlitz, emphasizing the importance of exporting projects for manual adjustments.
   - There are discussions about the feasibility of integrating Convex as an alternative to Supabase, though caution is advised due to its beta status.
- **User Experiences and Improvements**: Several users share experiences regarding how to better utilize Bolt and improve the AI's understanding and responsiveness when working on projects.
   - There are recommendations for implementing features like timestamping chats and better naming conventions for projects and forks to enhance user experience.
- **Community Support and Resources**: Community members discuss the lack of direct support from StackBlitz and emphasize the importance of utilizing community channels for assistance.
   - User-initiated improvement suggestions, such as clearer guidelines about the tool's capabilities, highlight the need for more intuitive instructions for non-developers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.netlify.com/sites/appname/overview">Netlify</a>: no description found</li><li><a href="https://docs.netlify.com/domains-https/custom-domains/multiple-domains/#domain-aliases">Sites with multiple domains</a>: Manage multiple domains for your site with a primary domain, domain aliases, domain-level redirects, automatic deploy subdomains, or branch subdomains.</li><li><a href="https://support.bolt.new/welcome">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It&#x27;s the all-in-one workspace for you and your team</li><li><a href="https://tenor.com/view/simpsons-homer-bart-lisa-join-us-gif-17846376318791889140">Simpsons Homer GIF - Simpsons Homer Bart - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/shake-my-head-mike-mclusky-mayor-of-kingstown-smh-disappointed-gif-293488442475603142">Shake My Head Mike Mclusky GIF - Shake my head Mike mclusky Mayor of kingstown - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://bolt.diy">GitHub - stackblitz-labs/bolt.diy: Prompt, run, edit, and deploy full-stack web applications using any LLM you want!</a>: Prompt, run, edit, and deploy full-stack web applications using any LLM you want! - stackblitz-labs/bolt.diy</li><li><a href="https://github.com/stackblitz/bolt.new/issues/4455">Supabase Problems · Issue #4455 · stackblitz/bolt.new</a>: Describe the bug My project was working fine until Bolt.new forced me to create a new data base. I do not want to keep burning tokens to fix the issue. I used 3million tokens fixit the first time. ...</li><li><a href="https://www.youtube.com/watch?v=1GfqnOAKr9M"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=eE6m0MmLpDU"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1322296416730611873)** (380 messages🔥🔥): 

> `DeepSeek V3 Performance, Aider Usage and Context Management, Gemini Models Insights, OpenRouter Integration Issues, OCR Implementation in Web Apps` 


- **DeepSeek V3 impresses users**: Many users are transitioning to **DeepSeek V3**, noting its efficiency and ability to handle large coding tasks effectively, sometimes completing projects significantly faster than before.
   - Comparisons with other models like **Gemini** reveal that DeepSeek is currently favored for its robust performance in coding assistance.
- **Managing Context and Limitations with Aider**: Users are learning how to optimize their **Aider** configurations for handling large projects, including setting up a `.aider.model.metadata.json` for context limits and costs.
   - Despite warnings about context limits, many users report successful experiences managing extensive codebases with reasonable performance.
- **Insights on Gemini Models**: Discussions about **Gemini 2.0** models highlight their strengths in coding tasks, particularly in free versions, with users effectively leveraging these models in their workflows.
   - Users suggest using **Gemini** for loading large codebases while relying on other models for generated coding.
- **OpenRouter Integration Challenges**: Some users encountered issues while trying to integrate **Aider** with OpenRouter's **DeepSeek**, often facing model not found errors due to configuration missteps.
   - Users were advised to enable specific settings to ensure proper endpoint access and functionality.
- **Excitement Over No-Code Development**: A user expressed excitement about rapidly building a web app in just one hour using **Aider**, demonstrating the potential for significant productivity gains.
   - Highlighting features like OCR through **TesseractJS**, the user pointed out the capabilities of automated solutions in coding without manually writing code.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat">Home</a>: aider is AI pair programming in your terminal</li><li><a href="https://blog.exolabs.net/day-2/">12 Days of EXO</a>: 12 Days of Truly Open Innovation</li><li><a href="https://aider.chat/docs/repomap.html">Repository map</a>: Aider uses a map of your git repository to provide code context to LLMs.</li><li><a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider is AI pair programming in your terminal</li><li><a href="https://tenor.com/view/star-trek-patrick-stewart-captain-jean-luc-picard-face-palm-disappointed-gif-4780258">Star Trek Patrick Stewart GIF - Star Trek Patrick Stewart Captain Jean Luc Picard - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://aider.chat/docs/llms/deepseek.html">DeepSeek</a>: aider is AI pair programming in your terminal</li><li><a href="https://x.com/kimmonismus/status/1873093574507872300">Tweet from Chubby♨️ (@kimmonismus)</a>: Open Source is on the rise</li><li><a href="https://x.com/alexocheema/status/1872447153366569110">Tweet from Alex Cheema - e/acc (@alexocheema)</a>: Had to stack up 8 Mac Minis to get it running.~5 tok/sec for now.First time running inference on 8 Mac Minis - performance can be improved a lot (theoretical limit is &gt;10 tok/sec on this setup).Quo...</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat">DeepSeek V3 - API, Providers, Stats</a>: DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and coding abilities of the previous versions. Pre-trained on nearly 15 trillion tokens, the reported ev...</li><li><a href="https://aider.chat/docs/config/options.html#repomap-settings">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>: Configuring advanced settings for LLMs.</li><li><a href="https://github.com/Aider-AI/aider-swe-bench">GitHub - Aider-AI/aider-swe-bench: Harness used to benchmark aider against SWE Bench benchmarks</a>: Harness used to benchmark aider against SWE Bench benchmarks - Aider-AI/aider-swe-bench</li><li><a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: Tell aider to follow your coding conventions when it works on your code.</li><li><a href="https://github.com/Aider-AI/conventions">GitHub - Aider-AI/conventions: Community-contributed convention files for use with aider</a>: Community-contributed convention files for use with aider - Aider-AI/conventions</li><li><a href="https://github.com/Aider-AI/aider/issues/2727">FastAPI Integration · Issue #2727 · Aider-AI/aider</a>: Issue AAAA - Aider As An API Overview I&#39;ve developed a FastAPI server that provides REST API access to Aider&#39;s functionality. Currently it runs as a standalone application but could benefit fr...</li><li><a href="https://api-docs.deepseek.com/news/news0802#how-to-use-deepseek-apis-caching-service">DeepSeek API introduces Context Caching on Disk, cutting prices by an order of magnitude | DeepSeek API Docs</a>: In large language model API usage, a significant portion of user inputs tends to be repetitive. For instance, user prompts often include repeated references, and in multi-turn conversations, previous ...</li><li><a href="https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus">BFloat16: The secret to high performance on Cloud TPUs | Google Cloud Blog</a>: How the high performance of Google Cloud TPUs is driven by Brain Floating Point Format, or bfloat16
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1322318992249065564)** (76 messages🔥🔥): 

> `DeepSeek V3 usage, Aider installation and configuration, Token limits with models, Git sparse-checkout compatibility, Shell command execution in Aider` 


- **DeepSeek V3 Concerns & Comparisons**: Users discussed the trade-offs between using DeepSeek V3 through Hugging Face and the hosted version, noting price differences and context window sizes, with Hugging Face offering **128k context** compared to **64k** for DeepSeek.
   - Concerns over **privacy and usage** of inputs when using DeepSeek hosted were raised, contributing to the discussion on whether higher prices might justify added security.
- **Aider Installation Best Practices**: Users noted that Aider should be installed globally with options like `aider-install` and seamless integration in various environments, emphasizing the importance of setup with the correct Python version.
   - Specific installation steps for different operating systems were provided, highlighting considerations for various package managers and environments, especially for Arch Linux users.
- **Managing Token Limits & User Commands**: Aider's ability to report token limits was mentioned, with users noting the adjustment of actions to avoid exceeding these limits while trying to maintain efficient coding workflows.
   - Frustration was expressed over Aider not executing shell commands directly, as users seek to streamline workflows, suggesting potential updates to allow broader approval settings.
- **Git Sparse-Checkout Compatibility**: Discussions emerged regarding Aider's compatibility with git sparse-checkout, with users advising against its use as Aider is reported to have issues with index-version 3 git repos.
   - Workarounds such as using the `--no-git` option were suggested to enable Aider functionality without git restrictions.
- **Commands Approval in Aider**: Users questioned the need for command approval in Aider, noting that while it enhances safety, it can hinder workflow efficiency, especially for advanced users.
   - The idea of introducing an environment variable to override the approval for shell commands was proposed to tailor Aider's operation according to specific user needs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/token-limits.html">Token limits</a>: aider is AI pair programming in your terminal</li><li><a href="https://www.chinatalk.media/p/deepseek-ceo-interview-with-chinas">Deepseek: The Quiet Giant Leading China’s AI Race</a>: Annotated translation of its CEO&#x27;s deepest interview</li><li><a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: Learn how to deploy + call models from different providers on LiteLLM</li><li><a href="https://www.codeguide.dev/">CodeGuide</a>: CodeGuide creates Detailed Documentation for your AI Coding Project.</li><li><a href="https://aider.chat/docs/install.html">Installation</a>: How to install and get started pair programming with aider.</li><li><a href="https://stackoverflow.com/questions/10418975/how-to-change-line-ending-settings">How to change line-ending settings</a>: Is there a file or menu that will let me change the settings on how to deal with line endings?&#xA;&#xA;I read there are 3 options:&#xD;&#xA;Checkout Windows-style, commit Unix-style&#xA;&#xA;Git will...</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">Models &amp; Pricing | DeepSeek API Docs</a>: The prices listed below are in unites of per 1M tokens. A token, the smallest unit of text that the model recognizes, can be a word, a number, or even a punctuation mark. We will bill based on the tot...</li><li><a href="https://github.com/Aider-AI/aider/issues/211">Aider uses GitPython, which doesn&#39;t work with index-version 3 git repos · Issue #211 · Aider-AI/aider</a>: When executing aider in an existing repo with R, Python, and Bash scripts, I get this error. When executing aider in a much smaller repo of only Python files, it works without problems. Thank you v...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1322334121212772455)** (31 messages🔥): 

> `Logit equalities in HF models, Dynamic test-time temperature in LLMs, BF16 training and gradient scaling, Lipschitz-1 RMSNorm replacement` 


- **Logits Equal in Float Precision**: A member reported encountering an issue with HF models where logits of two tokens are *exactly equal* in FP16 or BF16 during inference, despite not being the most probable tokens.
   - This discrepancy raises questions about the model's behavior since it occurs **20% of the time** in evaluations.
- **Dynamic Temperature in LLMs is Crucial**: A breakthrough in LLM architecture involves a strategy where **dynamic test-time temperature** is modulated to enhance creativity and problem-solving skills.
   - The proposal includes a mathematical structure expressing how temperature-controlled transitions in activation space can create creative trajectories.
- **BF16 Training and Gradient Scaling Queries**: Discussion around whether gradient scaling is necessary during BF16 training revealed that dynamic scaling might affect performance, while static scaling is less of a concern.
   - One member highlighted that it might not substantially speed up training latency, especially when processing smaller models.
- **Precision and Logits in Loss Functions**: A member was advised to compute logits in FP32 before applying the loss function for better performance and accuracy during BF16 training.
   - This approach ensures that the crucial cross-entropy calculations are not adversely affected by using lower precision.
- **RMSNorm Replacement in PyTorch**: A proposed implementation for a Lipschitz-1 RMSNorm replacement was shared, demonstrating how to normalize inputs based on their root mean square values.
   - The function utilizes the **tanh** activation for scaling, presented in a clear PyTorch code snippet.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1322303398569050186)** (219 messages🔥🔥): 

> `LLM Benchmarking Challenges, Gradient Routing for Neural Networks, TongGeometry for Geometry Theorem Discovery, Crosscoders for Feature Analysis, Superficial Alignment Hypothesis` 


- **LLM Benchmarking Reveals Flaws**: Discussions highlighted the challenges of accurately assessing LLM performance, emphasizing how current benchmarks often feature ambiguous questions and sensitivity to evaluation methods.
   - Participants suggested moving toward more functional benchmarks that measure performance in complex, open-ended tasks rather than simplistic multiple-choice formats.
- **Gradient Routing Enhances Model Interpretability**: Gradient routing was proposed as a method to improve the interpretability of neural networks by applying data-dependent masks during backpropagation, isolating capabilities within specific subregions.
   - This method could potentially address issues like mapping matrix entries to specific neurons by allowing for adjustable control over which parts of the model learn from particular data points.
- **TongGeometry and Geometry Theorem Discovery**: The paper on TongGeometry introduced a system for proposing and solving geometric problems, achieving significant discoveries in geometry theorems under computational constraints.
   - Despite lacks in methodological detail, the paper noted that some of TongGeometry's proposals were accepted in regional mathematical olympiads.
- **Exploring Crosscoders for Understanding Models**: Crosscoders, a new approach gaining attention, aims to track and resolve features across multiple layers of neural networks, showing potential for better understanding model behaviors.
   - Applications for crosscoders could improve how features are analyzed across layers, highlighting their use in circuit simplification and localization of model differentiations.
- **Superficial Alignment Hypothesis in SFT**: URIAL presents evidence supporting the Superficial Alignment Hypothesis, indicating that slight modifications in token distributions lead to similar performance metrics between base LLMs and their aligned versions.
   - This suggests alignment tuning may not fundamentally alter model capabilities but rather emphasizes stylistic token variation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.17758">In Case You Missed It: ARC &#39;Challenge&#39; Is Not That Challenging</a>: ARC Challenge appears more difficult than ARC Easy for modern LLMs primarily due to an evaluation setup that prevents direct comparison of answer choices rather than inherent complexity. Although some...</li><li><a href="https://arxiv.org/abs/2412.11834">Wonderful Matrices: Combining for a More Efficient and Effective Foundation Model Architecture</a>: In order to make the foundation model more efficient and effective, our idea is combining sequence transformation and state transformation. First, we prove the availability of rotary position embeddin...</li><li><a href="https://arxiv.org/abs/2410.04332">Gradient Routing: Masking Gradients to Localize Computation in Neural Networks</a>: Neural networks are trained primarily based on their inputs and outputs, without regard for their internal mechanisms. These neglected mechanisms determine properties that are critical for safety, lik...</li><li><a href="https://en.m.wikipedia.org/wiki/Where_Mathematics_Comes_From">Where Mathematics Comes From - Wikipedia</a>: no description found</li><li><a href="https://phyworld.github.io/">How Far is Video Generation from World Model: A Physical Law Perspective</a>: We conduct a systematic study to investigate whether video generation is able to learn physical laws from videos, leveraging data and model scaling.</li><li><a href="https://en.m.wikipedia.org/wiki/Chunking_(psychology)">Chunking (psychology) - Wikipedia</a>: no description found</li><li><a href="https://arxiv.org/abs/2412.10673">Proposing and solving olympiad geometry with guided tree search</a>: Mathematics olympiads are prestigious competitions, with problem proposing and solving highly honored. Building artificial intelligence that proposes and solves olympiads presents an unresolved challe...</li><li><a href="https://github.com/Re-Align/urial">GitHub - Re-Align/URIAL</a>: Contribute to Re-Align/URIAL development by creating an account on GitHub.</li><li><a href="https://arxiv.org/abs/2312.01552">The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning</a>: The alignment tuning process of large language models (LLMs) typically involves instruction learning through supervised fine-tuning (SFT) and preference tuning via reinforcement learning from human fe...</li><li><a href="https://www.lesswrong.com/posts/srt6JXsRMtmqAJavD/open-source-replication-of-anthropic-s-crosscoder-paper-for">Open Source Replication of Anthropic’s Crosscoder paper for model-diffing — LessWrong</a>: IntroAnthropic recently released an exciting mini-paper on crosscoders (Lindsey et al.). In this post, we open source a model-diffing crosscoder tra…</li><li><a href="https://transformer-circuits.pub/2024/crosscoders/index.html">Sparse Crosscoders for Cross-Layer Features and Model Diffing</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1322651431957954683)** (9 messages🔥): 

> `Neural Networks as Polycomputers, TinyStories Dataset, Small Transformers, Catastrophic Interference Solutions` 


- **Neural Networks exhibit Polycomputing properties**: Discussion centered around the idea that **neural networks** can be viewed as **polycomputers**, performing multiple computations on varying features simultaneously.
   - *Polycomputing* may offer insights into mitigating challenges such as **catastrophic interference**, enabling an agent to learn new behaviors without losing previously acquired knowledge.
- **TinyStories: A Dataset for Small Transformers**: The **TinyStories** dataset contains **synthetic short stories** generated by GPT-3.5 and GPT-4, designed to train small language models with fewer than 10 million parameters.
   - Members discussed the implications for training models with simpler architectures, as noted in the [TinyStories paper](https://arxiv.org/abs/2305.07759).
- **Seeking Open-Source Small Transformers**: A member requested references to **open-source, small transformers**, ideally with 1 to 5 layers pre-trained on complex tasks.
   - Responses highlighted examples like [TinyStories](https://arxiv.org/abs/2305.07759), indicating ongoing interest in developing lightweight models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2212.10675">There&#39;s Plenty of Room Right Here: Biological Systems as Evolved, Overloaded, Multi-scale Machines</a>: The applicability of computational models to the biological world is an active topic of debate. We argue that a useful path forward results from abandoning hard boundaries between categories and adopt...</li><li><a href="https://x.com/norabelrose/status/1873090825351250094">Tweet from Nora Belrose (@norabelrose)</a>: Neural networks are polycomputers in @drmichaellevin&#39;s sense.Depending on your perspective, you can interpret them as performing many different computations on different types of features. No pers...</li><li><a href="https://arxiv.org/abs/2305.07759">TinyStories: How Small Can Language Models Be and Still Speak Coherent English?</a>: Language models (LMs) are powerful tools for natural language processing, but they often struggle to produce coherent and fluent text when they are small. Models with around 125M parameters such as GP...</li><li><a href="https://huggingface.co/datasets/roneneldan/TinyStories">roneneldan/TinyStories · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1323056848210886676)** (12 messages🔥): 

> `Scrolls benchmark issues, GSM8K strict exact match clarification, mgsm_chat troubleshooting, ZeroSCROLLS vs SCROLLS evaluation, lm_eval command usage` 


- **Scrolls Benchmark Bugs Reported**: A user reported issues running the **scrolls benchmark**, noting that **load_metric** appears deprecated and must be replaced with **evaluate**.
   - Additionally, there were concerns regarding the **apply_chat_template** parameter not being recognized by **Instance**.
- **Clarifying GSM8K Metrics**: Inquiries were made on whether the **strict exact match** metric in GSM8K corresponds with the 'acc' metric used in the legacy leaderboard.
   - One member noted that the answer extraction process seems consistent between versions, referring to a specific [GitHub link](https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/tasks/gsm8k.py#L36).
- **Debugging mgsm_chat Model**: A user mentioned difficulties in replicating performance metrics with the **mgsm_chat** model, indicating no error but a lack of reproducibility.
   - Another member responded affirmatively about the model's functionality and asked for details on the specific errors encountered.
- **Discussion on SCROLLS vs ZeroSCROLLS Evaluation**: A user questioned why evaluations are conducted on **SCROLLS** for pre-trained models and **ZeroSCROLLS** for post-trained models despite the small dev set size.
   - This inquiry left open the possibility of re-directing the question to another appropriate channel if necessary.
- **lm_eval Command and Performance Results**: A user shared their **lm_eval** command for running the model and specified performance metrics for exact match evaluations.
   - The reported results showed a **flexible-extract** of **0.1098** and a **strict-match** of **0.0771**, with gratitude for prior assistance expressed.



**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/b281b0921b636bc36ad05c0b0b0763bd6dd43463/lm_eval/tasks/gsm8k.py#L36)">lm-evaluation-harness/lm_eval/tasks/gsm8k.py at b281b0921b636bc36ad05c0b0b0763bd6dd43463 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1322297747063308411)** (249 messages🔥🔥): 

> `DeepSeek V3 performance issues, OpenRouter model integration, Translation model recommendations, Building multimodal agents, LLM pricing and feature comparisons` 


- **DeepSeek V3 performance issues**: Users have reported that DeepSeek V3 performs noticeably worse on OpenRouter compared to its official API, with speculation that the Together API may be involved.
   - Responses indicate that changes or downgrades in performance can lead to user complaints, and some believe it's indicative of a new version being released.
- **OpenRouter model integration**: Integrating new models into OpenRouter requires providers with sufficient interest, and users can either partner with established AI labs or start their own provider.
   - Valuing niche LLM capabilities like coding can position a model favorably if marketed and developed appropriately.
- **Translation model recommendations**: Discussion highlighted that GPT-4o mini is preferred for translations, while Gemini 1.5 Flash was noted for making frequent errors.
   - Users suggested specific system prompts to enhance performance for translation tasks, emphasizing the importance of structure.
- **Building multimodal agents**: Although having models output JSON simplifies agent operations, it's not strictly necessary for running agents effectively.
   - Users discussed their interests in frameworks for multimodal agents, with mentions of Google’s Project Mariner as an interesting example.
- **LLM pricing and feature comparisons**: Discussions about LLM pricing revealed a lack of cached input token discounts via OpenRouter, with distinctions between various pricing strategies.
   - While some users expressed concerns about perceived downgrades in model performance, others emphasized the need for clear communication and evidence regarding model capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom is a multimodel chat interface. Add models and start chatting! Chatroom stores data locally in your browser.</li><li><a href="https://www.anthropic.com/research/building-effective-agents">Building effective agents</a>: A post for developers with advice and workflows for building effective AI agents</li><li><a href="https://openrouter.ai/docs/prompt-caching#deepseek">Prompt Caching | OpenRouter</a>: Optimize LLM cost by up to 90%</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat">DeepSeek V3 - API, Providers, Stats</a>: DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and coding abilities of the previous versions. Pre-trained on nearly 15 trillion tokens, the reported ev...</li><li><a href="https://openrouter.ai/rankings/translation?view=week">LLM Rankings: translation | OpenRouter</a>: Language models ranked and analyzed by usage for translation prompts</li><li><a href="https://api-docs.deepseek.com/api/create-completion">Create FIM Completion (Beta) | DeepSeek API Docs</a>: The FIM (Fill-In-the-Middle) Completion API.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/gGsmJeGdDi">Reddit - Dive into anything</a>: no description found</li><li><a href="https://fireworks.ai/blog/document-inlining-launch)">Fireworks - Fastest Inference for Generative AI</a>: Use state-of-the-art, open-source LLMs and image models at blazing fast speed, or fine-tune and deploy your own at no additional cost with Fireworks AI!
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1322346295418949712)** (74 messages🔥🔥): 

> `DeepSeek V3 Performance, Local AI vs. API Usage, Hunyuan Video Model Limitations, SmallThinker Model Overview, LLM Development Opportunities` 


- **DeepSeek V3 impresses with complex tasks**: DeepSeek V3 successfully passed the MTG commander deck building test and constructs correct Scryfall queries, showing it can handle complex tasks effectively.
   - Members noted that it feels like DeepSeek retains performance over context, setting it apart from other open-source models.
- **Debate on Local AI versus OpenAI API**: Users discussed the benefits of running Aquila's [Ollama](https://ollama.com) along with LlamaCPP for both learning and local setups, emphasizing system customization.
   - Having an OpenAI API setup was highlighted as advantageous for agentic tasks, providing a significant workflow improvement.
- **Limitations of Hunyuan Video Models**: Though Hunyuan can be used on limited hardware, it is noted to be sluggish and challenging to obtain good results with lower resolution and fewer frames.
   - There's also a blog post confirming that the model can run on GPUs with only **8GB VRAM**, though speed may be an issue.
- **Introduction of SmallThinker Model**: The new **SmallThinker-3B-preview** model has been introduced, showing improvements in reasoning capabilities with notable benchmark performance.
   - However, it struggles with knowing when to stop during tasks, prompting some humor among users.
- **Call for Developers for LlamaCPP**: The community expressed the urgent need for more developers for LlamaCPP, considering it foundational for many other projects.
   - It was suggested that those with coding experience should contribute, given its central role in advancing open-source AI models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/PowerInfer/SmallThinker-3B-Preview">PowerInfer/SmallThinker-3B-Preview · Hugging Face</a>: no description found</li><li><a href="https://blog.comfy.org/p/running-hunyuan-with-8gb-vram-and?r=4z50rt&utm_campaign=post&utm_medium=web&triedRedirect=true">Running Hunyuan with 8GB VRAM and PixArt Model Support</a>: Latest model support updates and office hour news from ComfyUI!</li><li><a href="https://www.videoleapapp.com/create/instagram-video-editor">Instagram Video Editor &amp; Maker: Create Instagram Videos | Videoleap</a>: Start your 7 day free trial today and try now! Use the Videoleap app to make and edit Instagram videos with ease. Add music &amp; more to your Instagram video.</li><li><a href="https://huggingface.co/datasets/PowerInfer/QWQ-LONGCOT-500K">PowerInfer/QWQ-LONGCOT-500K · Datasets at Hugging Face</a>: no description found</li><li><a href="https://youtu.be/jwChiek_aRY?si=uTnNyyUUX8IJXhie"> - YouTube</a>: no description found</li><li><a href="https://www.minimaxi.com/en/price">MiniMax - Intelligence with everyone</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1322293521742434305)** (149 messages🔥🔥): 

> `DeepSeek V3 performance issues, Weird behaviors in LLaMaCPP, Anthropic's reasoning models, Understanding LLaMa 3.3, High bandwidth vs home user solutions` 


- **DeepSeek V3 struggles with reasoning**: Members noted that **DeepSeek V3** gets caught in **reasoning loops**, exhibiting strange behaviors in evaluations, including infinite outputs and failure on reasoning tests like dead Schrödinger's cat.
   - Despite its performance in coding tasks, it is pondered if it performs differently across various benchmarks, raising questions about its architecture.
- **LLaMaCPP RPC middleware discussion**: A user discussed implementing a padding mechanism within **LLaMaCPP RPC**, suggesting it could manage tensor sizes effectively while preventing data corruption during processing.
   - Concerns were raised about whether this approach might lead to overly complex and hacky code, despite the potential efficiency benefits.
- **Anthropic's approach to models and reasoning**: There was speculation about **Anthropic's** possible internal reasoning models, with the idea that they may be using them to refine **Claude** instead of releasing them openly.
   - Members expressed curiosity about why Anthropic faces **compute issues**, given their background and resources.
- **User experiences with LLaMa 3.3**: One member shared positive impressions of **LLaMa 3.3** 70B’s performance regarding coding and document understanding, finding it superior in some tasks compared to alternatives.
   - These insights were contrasted with others indicating shaky performance under certain benchmarks, suggesting diverse user experiences.
- **Balance between high bandwidth solutions and home users**: A discussion ensued regarding **middleware** for quantization and network overhead, emphasizing the niche for efficient solutions geared toward consumer-grade hardware compared to data centers.
   - Members emphasized the lack of available resources for home users wanting to implement advanced models like **LLaMaCPP** without relying on high-bandwidth setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/aidan_mclau/status/1872444303974543859">Tweet from Aidan McLau (@aidan_mclau)</a>: two aidanbench updates:&gt; gemini-2.0-flash-thinking is now #2 (explanation for score change below)&gt; deepseek v3 is #22 (thoughts below)</li><li><a href="https://github.com/cpldcpu/MisguidedAttention">GitHub - cpldcpu/MisguidedAttention: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information</a>: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information - cpldcpu/MisguidedAttention</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/9285">Bug: GGML_ASSERT fail in ggml-rpc.cpp ggml_backend_rpc_buffer_init_tensor · Issue #9285 · ggerganov/llama.cpp</a>: What happened? I&#39;m trying to run qwen2-72b-instruct-q3_k_m.gguf with ggml-rpc function on 2 3060*2 machine. Machine1: 192.168.136.200 run llama-cli Machine2: 192.168.136.201 run rpc-server ./llama...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/9799">Bug: rpc server occasionally stops by assert function (deserialize_tensor)  · Issue #9799 · ggerganov/llama.cpp</a>: What happened? Description: When using the RPC backend in llama.cpp, I encountered crashes in the rpc_server::deserialize_tensor function. The assert fails because tensor-&gt;ne[i] can be zero after d...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1322582086795788391)** (6 messages): 

> `Sklearn Results Reporting, Binary Classification Metrics, Test Set Evaluation, Model Performance Trust, AUC/ROC Scores` 


- **Inquiry on Sklearn Results Format**: A member asked if the provided **sklearn results format** is typically reported in papers for binary classification, citing precision, recall, and F1-score metrics.
   - They presented a table format containing metrics for two classes alongside accuracy and averages.
- **Discussion on Metrics Trustworthiness**: Another member pointed out the importance of ensuring **trust in metrics**, emphasizing that the evaluation subset must be separate from the training set and representative of real-world distribution.
   - *Trust the metrics* also includes considering the classification model's goals, whether to prioritize precision or recall.
- **Adding AUC/ROC for Clarity**: The same member suggested that adding **AUC/ROC scores** for different classification thresholds could provide more insight into the model's performance.
   - This highlights the need for clarity in performance metrics when evaluating classification tasks.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1322582086795788391)** (6 messages): 

> `Reporting sklearn results, Metrics trustworthiness in classification, Binary classification metrics` 


- **Properly Reporting Sklearn Results**: A member inquired whether the reported results from sklearn in a class-precision format align with typical paper standards.
   - The example included metrics like **Precision**, **Recall**, and **F1-score**, along with **Support** values.
- **Trusting Classification Metrics**: Another member emphasized the importance of trusting the evaluation metrics used, asking if the test set is representative and decontaminated from the train set.
   - They suggested that understanding the model's goals is essential, highlighting the need for consideration of **precision vs recall** and suggesting the inclusion of **AUC/ROC scores**.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1322397353935634442)** (203 messages🔥🔥): 

> `Perplexity Pro Subscription, Deepseek v3 Availability, Reasoning Mode Functionality, Grant Proposal Assistance, Pro Reasoning and Search Enhancements` 


- **Confusion Over Perplexity Pro Access**: Users expressed confusion regarding the Perplexity Pro subscription and access to channels, noting issues with expired links and lack of student discounts.
   - Many are seeking clarity on how to effectively use the service and access the Pro features, showcasing the need for better communication from the platform.
- **Deepseek v3 Not Available in Pro Mode**: Discussion centered around the absence of Deepseek v3 in the Pro subscription, with users questioning its unavailability despite its perceived benefits.
   - Opinions varied on whether to utilize Deepseek for free instead, highlighting preferences for free services over potentially underwhelming Pro offerings.
- **Clarifying Reasoning Mode Features**: The functionality of the reasoning mode within Perplexity's Pro search was discussed, emphasizing how it triggers during complex queries to enhance output accuracy.
   - Users shared experiences with utilizing tables for organizing information, indicating a collective understanding of improving search queries through structured formats.
- **Getting Help with Grant Proposals**: A user sought advice on using Perplexity for creating instructional documents related to federal grant proposals, which are often complex and dense.
   - The challenge of extracting useful information efficiently from lengthy texts was a common concern, motivating requests for tips and strategies.
- **Comparing Models and Performance**: The conversation included evaluations of various models like Claude 3.5 Sonnet and GPT-4O, with users debating their effectiveness for different use cases.
   - Concerns about stability and accuracy in search results prompted discussions about alternatives, including Deepseek and ChatGPT Pro.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter">no title found</a>: no description found</li><li><a href="https://x.com/aravsrinivas/status/1871960456644145331?s=46">Tweet from Aravind Srinivas (@AravSrinivas)</a>: Merry Christmas 🎄! There’s a gift for all users on the App Store!</li><li><a href="https://youtu.be/7rD8AevYe9o?si=PN7UcRnBlVp3LIBo"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1322293871178154064)** (20 messages🔥): 

> `Meditation Techniques, Human Brain Speed, Neurosurgery After PG in ENT, HIV Drug Breakthrough, Cold Bath Benefits` 


- **Exploring Different Meditation Techniques**: Many members showed interest in various [meditation techniques](https://www.perplexity.ai/search/meditation-techniques-N7qb7MqYTFebfVJxgsdl0w), with multiple links shared referencing their effectiveness and benefits.
   - *Practice leads to improvement* was a recurring theme in the discussions, emphasizing dedication to the techniques.
- **Human Brain's Sluggish Performance**: A discussion focused on why the [human brain is considered very slow](https://www.perplexity.ai/page/the-human-brain-is-very-slow-YGm.UjyKRW.caXXHnhlb4Q) in processing information compared to modern computing.
   - Participants delved into the implications this has on learning and cognitive function.
- **Neurosurgery Pathways Post-ENT**: Several queries about the path of [neurosurgery after completing PG in ENT](https://www.perplexity.ai/search/neurosurgery-after-pg-in-ent-7arEmPo4QMSR07K4_a7KnQ) sparked diverse opinions and advice on the transition.
   - Members shared experiences, encouraging those interested to consider the extensive training this field requires.
- **Game-Changing HIV Drug Breakthrough**: An exciting development regarding an [HIV drug breakthrough](https://www.perplexity.ai/page/hiv-drug-named-breakthrough-of-kzPk2YAoQPKS.CdzOsNdXA) was highlighted, sparking discussions about its potential impact on treatment.
   - Members expressed optimism about future advancements in HIV research, underscoring a commitment to ongoing studies.
- **Cold Baths and Their Benefits**: A member shared insights about the [benefits of cold baths](https://www.perplexity.ai/search/beneficios-dos-banhos-frios-h.XO8IFRSLKsDxGoqpRPZA#0) for recovery and overall health.
   - The discussion included various personal anecdotes, noting *how invigorating* cold exposure can feel.



**Link mentioned**: <a href="https://www.youtube.com/embed/rS29fEFkzDU">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1322353035510284390)** (7 messages): 

> `Search API Alternatives, Custom Recency Time Feature, Citations Limit, API Credit Refunds, Conversational Use of API` 


- **Exploring Search API Alternatives**: A user inquired about other search API alternatives that match or exceed current quality standards.
   - The community is actively discussing various options, seeking and sharing recommendations.
- **Request for Custom Recency Filter**: A user asked if a custom recency time could be added to filter search results, referring to [Perplexity API documentation](https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter).
   - No specific responses were recorded regarding the feasibility of this request.
- **Clarification on Citations Limit**: A user questioned whether there is a limit on the number of citations returned by the API.
   - No answers or clarifications were provided on this topic during the discussion.
- **Refund Process for API Credit**: A member sought guidance on obtaining a refund for accidentally paid API credit.
   - Another user advised contacting [api@perplexity.ai](mailto:api@perplexity.ai) for assistance with the refund process.
- **Using API for Conversational Interaction**: A user explored the possibility of using the API for conversational interactions, expressing confusion over receiving definitions instead of contextual responses.
   - A response clarified that the Sonar models are designed for question-answering using web sources and proper citations, not for conversational purposes.



**Link mentioned**: <a href="https://docs.perplexity.ai/api-reference/chat-completions#body-search-recency-filter">no title found</a>: no description found

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1322344185659129947)** (96 messages🔥🔥): 

> `Image Generation Quality, AI in Coding, Gemini 2.0 Performance, Self-Employment and AI Usage, Token Limits in Content Creation` 


- **Users debate image generation capabilities**: Members discussed varying experiences with image generation tools, expressing mixed feelings about the quality and cleanup required for generated posters.
   - Conversations also touched on the limitations of models like Claude and the capabilities of models such as Eleven Labs in handling audio and video.
- **AI assistance in programming faces scrutiny**: A user shared concerns about ChatGPT's declining coding capabilities over the past few weeks, particularly in managing existing code and making unnecessary changes.
   - Another member suggested using a multi-step approach to coding with AI, highlighting that OpenAI models like GPT-4 have limitations with larger code bases.
- **Positive feedback on Gemini 2.0's performance**: Several members praised the performance of Gemini 2.0, particularly its 'flash thinking' ability and effectiveness in coding tasks compared to other models.
   - Comparisons were made between Gemini and OpenAI models, with users acknowledging the strengths of each while emphasizing the need for integrated features in OpenAI's offerings.
- **Discussions on self-employment and AI utilities**: One user expressed their experience of being unemployed while utilizing various AI models for creative coding projects, highlighting load balancing among free options.
   - The challenges of negotiating work conditions were mentioned in light of self-employment in the tech field.
- **Addressing API limitations and blog posts**: A member sought advice on managing token limits when generating extensive blog posts using APIs, particularly when combining URL data with manual inputs.
   - The conversation hinted at the need for strategies to maximize content generation efficiency given existing constraints.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.17256">B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners</a>: In the absence of extensive human-annotated data for complex reasoning tasks, self-improvement -- where models are trained on their own outputs -- has emerged as a primary method for enhancing perform...</li><li><a href="https://www.reddit.com/r/ClaudeAI/s/bO3cOogG6c">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1322294255045054638)** (11 messages🔥): 

> `GPT Agents Potential, GPT-2 Maximum Token Generation, Interactive App Button Features, Script Enhancement with AI Assistance` 


- **GPT Agents show promise**: A member expressed enthusiasm about the potential of **GPT's** to function as effective agents, eagerly anticipating the completion of integrated systems.
   - They highlighted their excitement for when everything aligns to begin utilizing these agents in practical applications.
- **Stuck with GPT-2 Token Limit**: A user working with the **GPT-2** model encountered issues due to its **maximum token length of 1024**, making it difficult to generate larger articles.
   - They inquired about methods to overcome this limit and generate text with as many as **10,000 tokens**.
- **Exploration of Interactive App Functions**: Discussion centered around buttons designed to assist in creating apps, with one button leading to a finished application and others generating procedural outputs.
   - Users were told that these buttons guide you through various types of apps, with options to continue navigating prompts.
- **AI Assists in Script Updates**: One member shared how **AI** helped them enhance a script to provide a **more coherent cinematic experience**.
   - They acknowledged not knowing how to code, yet successfully relied on AI to explain and modify their code block effectively.



**Link mentioned**: <a href="https://discordapp.com/channels/974519864045756446/1315696747279810711/1323428129083097158">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1322343519838273607)** (51 messages🔥): 

> `Sora Prompt Engineering, ChatGPT Prompting Techniques, Markdown Usage Guidelines, Course Interest in Prompt Engineering, Channel Purpose and Organization` 


- **Call for Dedicated Sora Prompt Channel**: Users expressed the need for a separate channel for **Sora prompts**, arguing that the current discussion isn't focused enough on prompt engineering for ChatGPT.
   - There was a consensus that having a dedicated space could enhance the engagement and usability of Sora prompts.
- **Concerns Over Prompting Best Practices**: Several members discussed the variability of prompts, noting that the best prompts are direct, yet context plays a critical role in outcomes.
   - There's an acknowledgment that as new model variations emerge, **best practices** can change, making it difficult to define universally effective prompting techniques.
- **Markdown in Discord Channels**: The use of **markdown** was debated, with some users feeling its absence hampers clear communication and the ability to share prompt examples accurately.
   - Feedback suggested that allowing markdown could facilitate better documentation of prompts and practices among members.
- **Interest in Prompt Engineering Courses**: There is notable interest among users for formal courses on prompt engineering to enhance their skills with ChatGPT.
   - Members reflected on the complexity of mastering prompting, recognizing the absence of established rules due to evolving models and contexts.
- **Channel Purpose and Engagement**: Discussions hinted at the channel focusing more on conversational uses of AI rather than strictly prompt engineering, which may dilute the intent of discussions.
   - Users voiced a desire for clearer boundaries regarding topics that directly relate to prompt engineering, rather than general discussions.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1322343519838273607)** (51 messages🔥): 

> `Sora Prompt Engineering, Prompt Engineering Courses, Markdown Use in Channels, User Engagement on Discord, ChatGPT Interaction Dynamics` 


- **Push for a Dedicated Sora Prompts Channel**: Members discussed the need for a dedicated **Sora prompts channel** to facilitate better engagement and organization around Sora-specific prompting.
   - There's a consensus that prompt engineering content is sparse in the current channel, leading to requests for structured discussions.
- **Interest in Prompt Engineering Courses**: Users expressed interest in finding or creating **courses on prompt engineering** to improve their skills with ChatGPT, noting room for improvement.
   - Participants shared thoughts on the variability of 'best' prompts and how this may change with different model versions.
- **Concerns Over Markdown Restrictions**: A member voiced frustration about markdown not being allowed in the channel, which hindered their ability to share **prompt examples effectively**.
   - Discussions indicated that allowing markdown could enable users to share examples more clearly, enhancing the collaborative learning experience.
- **Variable Nature of ChatGPT Interactions**: Participants noted that **the behavior of ChatGPT can vary between sessions**, making it difficult to establish consistent prompt patterns.
   - This variability requires a conversational approach, where users often need to adjust their prompts based on the AI's responses.
- **Engagement Dynamics in Chat Channels**: The conversation highlighted concerns about the channel's focus potentially shifting away from pure **prompt engineering towards general discussions**.
   - Members were encouraged to share more specific ideas or feedback to ensure the channel meets their prompting needs effectively.


  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1322330052045836328)** (29 messages🔥): 

> `NotebookLM audio usage, Embedding interactive features, Interactive mode suggestions, Handling sensitive content, YouTube video sharing` 


- **NotebookLM audio for public use**: A member inquired whether the audio from NotebookLM can be used publicly as long as credit is given, and another reassured them they've done so without issue.
   - Another member humorously noted that *no one's arrested them yet* for using the audio.
- **Embedding NotebookLM features**: A user questioned if NotebookLM's interactive feature could be embedded on a website for user interaction.
   - Suggestions included potentially scraping the website and connecting with APIs to integrate these features.
- **Suggestions for improving interactive mode**: A member expressed enthusiasm for the new interactive mode in NotebookLM but suggested adding a native recording feature to simplify saving discussions.
   - They proposed an idea for an 'after the fact record' option to save useful portions of conversations.
- **Issues with handling sensitive content**: A user reported difficulties in uploading complaints and sensitive documents to NotebookLM, stating that the system failed to find their notes or PDFs.
   - Others speculated that the platform's strictness regarding sensitive topics might be causing these issues.
- **Sharing YouTube videos**: Users discussed the ability to share YouTube videos in the channel, with some reporting restrictions while others could post links.
   - A member noted potential rate limits by Discord or modifications in moderation settings as possible reasons for the discrepancies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=4rdXYdMmrFg"> - YouTube</a>: no description found</li><li><a href="https://youtu.be/ubA36TeoyM4"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1322311062045069514)** (156 messages🔥🔥): 

> `NotebookLM Plus Features, Podcast Generation Issues, Source Management Challenges, User Feedback on AI Responses, Limitations on Notebook Usage` 


- **NotebookLM Plus Features Discussion**: Many users are curious about the differences between standard NotebookLM and NotebookLM Plus, with features such as increased upload limits and access to additional resource types highlighted.
   - Discussions emphasized the need for more clarity regarding limits, with a maximum of **500** notebooks for Plus users and **100** for free users.
- **Podcast Generation Does Not Update**: Users are facing issues with the podcast feature, where newly added sources are not reflected in the generated audio unless explicitly deleted and regenerated.
   - To regenerate the audio, a delete option is available in a three-dot menu next to the existing audio overview.
- **Issues with Source Uploading**: Several users reported errors while uploading MP3 files, with sources turning red and an error message appearing, indicating a need for fixes.
   - Additionally, issues with YouTube source transcripts not being recognized despite their availability were highlighted as a problem in the community.
- **User Frustrations with AI Responses**: Concerns were raised about NotebookLM's tendency to overlook sections of sources, which can affect the accuracy of generated responses.
   - Some users managed to resolve this by adjusting their source volume and content, emphasizing the need for iterative adjustments to achieve desired outputs.
- **Interest in API and Mobile Support**: Multiple users inquired about the availability of an API for NotebookLM and the possibility of utilizing the service on mobile devices.
   - Suggestions included the need for a summarized transcript retention option for chat interactions and updates on offline usability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=NotebookLM%20vs%20NotebookLM%20Plus%20User%20Limits">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://support.google.com/notebooklm/answer/14276468?hl=en">Sources - NotebookLM Help</a>: no description found</li><li><a href="https://learning.google.com/experiments/learn-about">Learn About</a>: no description found</li><li><a href="https://github.com/agituts/gemini-2-podcast">GitHub - agituts/gemini-2-podcast: A Python-based tool that generates engaging podcast conversations using Google&#39;s Gemini 2.0 Flash Experimental model for script generation and text-to-speech conversion.</a>: A Python-based tool that generates engaging podcast conversations using Google&#39;s Gemini 2.0 Flash Experimental model for script generation and text-to-speech conversion. - agituts/gemini-2-podcast
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1322317019915227167)** (146 messages🔥🔥): 

> `M2 Max MacBook Pro for AI, Depth Maps and Banding Issues, Using Loras for Consistency, AI Video Generation Tools, Stable Diffusion Discord Community` 


- **Is M2 Max MacBook Pro sufficient for AI tasks?**: A user inquired about purchasing an **M2 Max MacBook Pro** with 32GB RAM and a 38-core GPU for local AI tasks, expressing concern over potential performance issues compared to dedicated Nvidia GPUs.
   - While several members shared their experiences, one noted that although it would work, it might not provide a satisfying experience for intensive tasks.
- **Banding issues with depth maps**: A user reported problems using depth maps from 3D modeling software, noticing **banding** interpreted as edges by the model, and sought solutions.
   - Advice included ensuring the maximum depth aligns with the furthest object desired and using depth maps in formats consistent with model requirements.
- **Training Loras for consistent illustrations**: A member looking to maintain character consistency in a children's book was advised to *train a Lora using Stable Diffusion*.
   - This approach seemed promising for achieving a consistent watercolor hand-drawn style while creating illustrations based on reference photos.
- **Exploring AI video generation tools**: A discussion emerged around options for generating AI videos, mentioning platforms like **Luma Dream Machine**, **Kling**, and **Minimax** for cloud-based solutions.
   - Users inquired about the cost and availability of these platforms, wanting to experiment with video generation without committing to local installations.
- **Stable Diffusion Discord community concerns**: The community engaged in discussions about moderation, bot activity, and safety measures within the Discord server, suggesting the need for captcha implementation to deter spam.
   - Further conversations touched on the context of censorship in models and potential impacts on generating quality outputs, particularly concerning character anatomy.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=vY4QwnR4R2M"> - YouTube</a>: no description found</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1322340971232497715)** (138 messages🔥🔥): 

> `Mojo Static Methods, Recursive Structs in Mojo, Performance Optimization Techniques, Memory Management of Pointers, Using ArcPointer for Self-Referential Structures` 


- **Discussion on Mojo Static Methods**: Members debated the semantics of static methods in Mojo, considering the utility of using a 'self' argument as a signal for instance methods and the implications of this choice.
   - They discussed potential changes for backward compatibility with Python, suggesting that Mojo should replicate current Python static method behaviors.
- **Challenges with Recursive Structs**: A user encountered segmentation faults when using `UnsafePointer[Self]` for recursive struct definitions in Mojo's AST nodes.
   - They explored alternatives like `OwnedPointer` and `ArcPointer`, which seemed more viable despite some drawbacks.
- **Performance Optimization Techniques in Mojo**: Users discussed the importance of using 'load' for performance optimization when manipulating SIMD data in Mojo, as opposed to a direct bitcast, which might not utilize the best method for loading.
   - Reference to educational resources was made, emphasizing an understanding of CPU behavior as crucial for maximizing performance.
- **Managing Child and Parent Pointers**: Participants shared insights on the complexities of managing parent-child relationships in data structures, particularly when dealing with optional and unsafe pointers in recursive scenarios.
   - A recommended approach included using `OpaquePointer` to sidestep the intricacies and limitations that recursive types can introduce.
- **Bug Reporting in Mojo**: A bug was reported regarding segmentation faults occurring in Mojo when running in full debug mode, contrasting with the regular runtime behavior.
   - Users were advised to expect delays in responses from developers due to holidays.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/rebind/rebind/">rebind | Modular Docs</a>: rebindsrctype AnyTrivialRegType -&gt; desttype</li><li><a href="https://www.computerenhance.com/p/table-of-contents.">Table of Contents</a>: Every entry in every series, listed for quick navigation.</li><li><a href="https://github.com/modularml/mojo/issues/3917">[BUG] --debug-level full crashes when importing · Issue #3917 · modularml/mojo</a>: Bug description Running a mojo script using the debugger seg faults, as opposed to when running regular mojo, which runs to completion (although I have noticed strange behavior in the regular scrip...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1322298354209788006)** (85 messages🔥🔥): 

> `Model Performance Improvements, Vision Models and Censorship, Custom Config Implementation, Prompt Template Issues, Local Network Serving` 


- **Model Performance Shows Huge Gains**: Users have reported significant performance improvements, with claims of up to **20x** and **6t/s** using the latest builds.
   - One user recommended using the Perf Monitor for detailed GPU history to assess the improvements.
- **Censorship Challenges with Vision Models**: One user tested vision models only to find them 'censored' for NSFW content, prompting inquiries for uncensored alternatives.
   - There were suggestions to explore model capabilities or potentially bypass the existing censorship.
- **Implementing Custom Config in LM Studio**: A user detailed their method for adding a custom config preset in LM Studio by manually editing the config file.
   - It was pointed out that an easier method exists through the UI, allowing direct selection of preset files for configuration.
- **Issues with Prompt Templates**: Users noted that some models exhibit unexpected output by appending their own responses, marked by **### Instruction**.
   - It was suggested that this issue can often be resolved by ensuring the correct prompt template is used with the model.
- **Serving LM Studio on Local Network**: A user sought help to serve LM Studio on a local network but couldn't find the option in the current version.
   - Guidance was provided to check the server port options in the settings, leading to the use of the beta build for better functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://model.lmstudio.ai/download/lmstudio-community/DeepSeek-V2.5-1210-GGUF">Download and run lmstudio-community/DeepSeek-V2.5-1210-GGUF in LM Studio</a>: Use lmstudio-community/DeepSeek-V2.5-1210-GGUF locally in your LM Studio</li><li><a href="https://x.com/lmstudio">Tweet from undefined</a>: no description found</li><li><a href="https://www.nomic.ai/blog/posts/gpt4all-scaling-test-time-compute">Scaling Inference Time Compute with On-Device Language Models in GPT4All</a>: Scaling Inference Time Compute with On-Device Language Models with support for Code Interpreter, Tool Calling, and Code Sandboxing.</li><li><a href="https://lmstudio.ai/docs/configuration/prompt-template">Prompt Template - Configuration | LM Studio Docs</a>: Optionally set or modify the model&#x27;s prompt template</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta Releases</a>: LM Studio Beta Releases</li><li><a href="https://lmstudio.ai/docs/basics/rag">Chat with Documents - Running LLMs Locally | LM Studio Docs</a>: How to provide local documents to an LLM as additional context</li><li><a href="https://lmstudio.ai/docs/basics/download-model#changing-the-models-directory">Download an LLM - Running LLMs Locally | LM Studio Docs</a>: Discover and download supported LLMs in LM Studio
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1322526673689776159)** (24 messages🔥): 

> `3090 NV-Link setups, Noise levels of blower GPUs, Water cooling solutions, PCIe riser issues, Jetson Orin Nano performance` 


- **Exploring NV-Link with 3090 setups**: Several members discussed their experiences with **NV-Link** setups for **3090** GPUs, considering benefits and setup challenges.
   - *One noted the need for long and flexible NV-Link and questioned the benefit of 2x2 configurations over standalone cards.*
- **Concerns over Blower GPU Noise Levels**: There were concerns about the **noise levels** of the **ASUS GeForce RTX 3090 TURBO**, especially since it peaks at **83 decibels**, which can lead to hearing damage.
   - Members suggested that these blower cards are more suited for server setups rather than living spaces.
- **Water Cooling for 3090 GPUs**: A suggestion emerged that **water cooling** would be beneficial for high-performance setups to manage both noise and thermal limitations.
   - *Another member emphasized that **inference tasks** typically do not create excessive load, thus keeping noise to a minimum fortuitously.*
- **Challenges with PCIe Risers**: One member faced issues with a **90-degree PCIe riser** that misaligned the GPU, prompting the need for further adjustments.
   - This sparked discussions on cable management challenges and the need for custom-length cables in non-standard builds.
- **Testing Jetson Orin Nano Performance**: A member shared an update on their testing of the **Jetson Orin Nano**, comparing speeds across 20 different models in **25W mode**.
   - This led to inquiries about **quantization** of models and discussions on wattage efficiency.



**Link mentioned**: <a href="https://www.jeremymorgan.com/blog/tech/nvidia-jetson-orin-nano-speed-test/">How Fast Does the Jetson Nano Really Run Large Language Models?</a>: Can your Jetson Orin Nano handle the latest LLMs? We test a range of whooping models to see how fast they run.

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1323391290913718383)** (3 messages): 

> `CUDA Programming, Overlap Data Transfer, CUDA Projects` 


- **Seeking CUDA Project Ideas for Job Preparation**: A member has completed a course on **CUDA programming** and is looking for suggestions on **CUDA projects** to help showcase their skills during job hunting.
   - They specifically requested advice from experts in the field to enhance their portfolio.
- **Inquiry on Overlap Data Transfers**: Another member asked for assistance regarding **overlap data transfer** in CUDA programming.
   - They provided a [link to an NVIDIA blog](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/) discussing techniques for optimizing data transfers in CUDA.



**Link mentioned**: <a href="https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/">How to Overlap Data Transfers in CUDA C/C++ | NVIDIA Technical Blog</a>: In our last CUDA C/C++ post we discussed how to transfer data efficiently between the host and device. In this post, we discuss how to overlap data transfers with computation on the host&#8230;

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1322621742568570925)** (19 messages🔥): 

> `Triton Installation Issues, Cross Entropy Implementations, Softmax Kernel Optimization, SpMM Kernel in Triton` 


- **Triton installation fails load kernel test**: A user reported issues with their Triton installation causing mismatched results during kernel tests despite successfully installing Torch and Triton versions.
   - Another member pointed out missing details in the code, specifically regarding required input types and potential race conditions.
- **Exploring Cross Entropy Implementations in Triton**: A user inquired about available cross entropy implementations using Triton and highlighted performance issues they're facing.
   - Several members suggested notable implementations on GitHub, including those from Liger-Kernel and Attorch, for reference.
- **Softmax Kernel Optimization Queries**: A user presented their challenge with efficiently utilizing the GPU in their softmax kernel implementation, indicating that expanding dimensions significantly slowed down performance.
   - A member recommended examining the mathematical changes that occur with dimensional expansion and encouraged providing reference implementations to compare.
- **Building an SpMM Kernel in Triton**: A member asked for advice on accessing elements from a BCSR format in Triton while aiming to optimize element loading into shared memory for a SpMM kernel.
   - Another user clarified that Triton currently does not support direct indexing but suggested a workaround using pointer arithmetic, acknowledging potential performance concerns.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/tree/main/unsloth/kernels">unsloth/unsloth/kernels at main · unslothai/unsloth</a>: Finetune Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth</li><li><a href="https://github.com/triton-lang/triton/issues/5509">Cross Entropy Loss performance issue · Issue #5509 · triton-lang/triton</a>: Describe the issue I implemented cross-entropy using Triton, but the performance is disappointingly low. Even after removing most of the code in the loss_kernel (producing incorrect results), the p...</li><li><a href="https://github.com/linkedin/Liger-Kernel">GitHub - linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training</a>: Efficient Triton Kernels for LLM Training. Contribute to linkedin/Liger-Kernel development by creating an account on GitHub.</li><li><a href="https://github.com/duonlabs/sick-bide/blob/cc71ec639e1b690e2d70a85474d762d4b66f9c25/sick_bide/kernels/precompute/integral.py#L9">sick-bide/sick_bide/kernels/precompute/integral.py at cc71ec639e1b690e2d70a85474d762d4b66f9c25 · duonlabs/sick-bide</a>: A Neural network layer able to express distributions over anything - duonlabs/sick-bide</li><li><a href="https://github.com/BobMcDear/attorch/tree/main/attorch">attorch/attorch at main · BobMcDear/attorch</a>: A subset of PyTorch&#39;s neural network modules, written in Python using OpenAI&#39;s Triton. - BobMcDear/attorch</li><li><a href="https://github.com/duonlabs/sick-bide/blob/cc71ec639e1b690e2d70a85474d762d4b66f9c25/sick_bide/kernels/precompute/integral.py">sick-bide/sick_bide/kernels/precompute/integral.py at cc71ec639e1b690e2d70a85474d762d4b66f9c25 · duonlabs/sick-bide</a>: A Neural network layer able to express distributions over anything - duonlabs/sick-bide</li><li><a href="https://github.com/duonlabs/sick-bide/blob/cc71ec639e1b690e2d70a85474d762d4b66f9c25/sick_bide/reference.py#L7">sick-bide/sick_bide/reference.py at cc71ec639e1b690e2d70a85474d762d4b66f9c25 · duonlabs/sick-bide</a>: A Neural network layer able to express distributions over anything - duonlabs/sick-bide</li><li><a href="https://github.com/triton-lang/triton/blob/4d2e9e5de96a5d6ea163f2de04ae5c5b6be45825/python/triton/language/core.py#L2562">triton/python/triton/language/core.py at 4d2e9e5de96a5d6ea163f2de04ae5c5b6be45825 · triton-lang/triton</a>: Development repository for the Triton language and compiler - triton-lang/triton
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1322342859495571569)** (14 messages🔥): 

> `TMA vs cp.async, Vectorized Load Benefits, GEMM Tutorial Series, CUDA Kernel Efficiency, Input/Output Precision in CUTLASS` 


- **TMA shows benefits over cp.async**: A discussion clarified that TMA can execute instructions with fewer threads than cp.async, allowing for greater flexibility and efficiency with resources.
   - The distinction between register use for memory address generation was highlighted, noting that TMA conserves resources better than cp.async.
- **Vectorized load reduces memory instructions**: It's noted that vectorized loading can improve performance by reducing memory load instructions, leading to lower register usage and diminished instruction overhead.
   - Fewer load instructions help prevent LG throttling, enhancing occupancy and latency hiding for better performance.
- **GEMM Tutorial Series on Hopper GPUs**: A tutorial on GEMM (General Matrix Multiplication) on NVIDIA Hopper GPUs was introduced, emphasizing its importance in GPU computations.
   - The series comprises three parts, focusing on WGMMA instructions and advanced techniques necessary for efficient GEMM kernel implementation, with links provided for more information.
- **Assessing Kernel Efficiency**: A user's kernel profiling metrics reflected that the kernel's compute performance is good, achieving around **82.85% GPU throughput** despite low memory throughput.
   - The discussion included insights on occupancy, revealing the kernel achieves a **99.24% occupancy**, indicating effective use of resources within theoretical limits.
- **Understanding Precision in CUTLASS Kernels**: A beginner inquired about determining input, multiplication, and output precision within a CUTLASS kernel, specifically for a BF16 operation.
   - Links were shared to relevant documentation on CUTLASS functionality, indicating that understanding kernel definitions can clarify precision usage.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/">CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper&#x2122; GPUs</a>: No series of CUDA® tutorials is complete without a section on GEMM (GEneral Matrix Multiplication). Arguably the most important routine on modern GPUs, GEMM constitutes the majority of compute done…</li><li><a href="https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html">2. Kernel Profiling Guide &mdash; NsightCompute 12.6 documentation</a>: no description found</li><li><a href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/functionality.md">cutlass/media/docs/functionality.md at main · NVIDIA/cutlass</a>: CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1322397130672836721)** (4 messages): 

> `Guard Performance Optimization, Debugging Slow Code` 


- **Optimize Guard Performance**: It's noted that generally, there's minimal need to worry about **guard performance**.
   - However, *ways to disable unneeded guards* exist for those looking to maximize performance.
- **Investigating Slow Code Issues**: A member raised a concern regarding **slow performance** in their codebase, which consists of over **100 lines of code**.
   - *Requests for help on debugging* were made, seeking insights into the underlying issues affecting performance.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1322953851632091177)** (4 messages): 

> `Power-of-2 Quantization, MAGVIT-v2 Binary Quantization, Non-Uniform Quantization Levels, ViT Model Quantization Issues` 


- **Exploring Power-of-2 Quantization**: A member inquired whether anyone had investigated **power-of-2 quantization**, emphasizing its suitability for aligning with **Laplacian distributions**.
   - They noted the potential speed benefits due to **bit shifting** in integer arithmetic and pointed to **Dominika Przewlocka-Rus**'s research at Meta/Intel for further insights.
- **MAGVIT-v2 Uses Binary Quantization**: Another member mentioned that **MAGVIT-v2** employs a form of binary quantization, which converts continuous values into binary digits interpreted as powers of two.
   - This approach effectively turns values into a quantized range, such as {some continuous} to [0][1][0][0][1][0], which translates into the decimal value of **9**.
- **Debating Uniform vs Non-Uniform Quantization**: The discussion shifted to the difference between **uniform quantization** and a proposed non-uniform method, with quantization levels expanding in powers of two.
   - For example, a value of **10** would round to **8**, showcasing potential efficiency and speed without reliance on LUTs.
- **ViT Models Face Quantization Challenges**: A member highlighted a blog post from **Unsloth** discussing **ViT models** struggling with quantization due to data outliers.
   - They speculated that new quantization techniques might improve model performance, referencing the potential relevancy of their project.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1323417048692559872)** (1 messages): 

> `Cracked Tech Jobs, CUDA Engineer Role, Remote LLM Infrastructure Positions, Triton Kernel Development Roles` 


- **Exciting Cracked Research Engineer Job Opportunity**: A member discovered a [cracked research engineer job](https://crackedengineers.com/job/p-1-ai-7f41fa30-6cfa-4e9a-8943-2324dc21d243) that might pique interest in the tech community.
   - They highlighted it as a great resource for finding cracked tech jobs in various domains.
- **Search Queries for Ideal Tech Roles**: Tips were shared for finding roles like **CUDA engineer in SF** or **Remote LLM infrastructure engineer positions**.
   - The conversation emphasized using queries that the platform can act on, making job searches more effective.
- **Triton Kernel Development Roles Discussion**: Members discussed the necessity of including **Triton kernel development** in their job searches.
   - This reflects a growing trend towards specialized roles that enhance performance in AI development.



**Link mentioned**: <a href="https://crackedengineers.com/job/p-1-ai-7f41fa30-6cfa-4e9a-8943-2324dc21d243">Cracked Engineers</a>: Hire the best ai and software engineers for your startup.

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1322802561232928879)** (26 messages🔥): 

> `Deep Learning on Linux vs Windows, Resources for Triton, NVIDIA dGPU Management on Ubuntu, Switching to Arch Linux, Success Stories with CUDA` 


- **Linux preferred for Deep Learning vs Windows**: A discussion arose about whether to stick with Windows or dual boot Linux for deep learning on an NVIDIA RTX 4060, with many recommending Ubuntu 22.04 as the better option.
   - A user expressed concerns about managing dGPU resources, stating that Ubuntu 22.04 presents challenges not faced with their previous installation of Ubuntu 20.04.
- **Triton Resources for Beginners**: A user sought recommendations for resources to start learning Triton, with another member sharing a GitHub link to a curated list of Triton resources.
   - This list is aimed at those looking to learn and explore Triton, OpenAI's programming language for writing efficient GPU code.
- **Challenge with dGPU Management on Ubuntu**: Several users discussed difficulties with NVIDIA dGPU management on Ubuntu, particularly with using GNOME and Wayland environments.
   - There were suggestions regarding configuration, including disabling Wayland to free up the GPU for deep learning tasks.
- **Considerations for Switching to Arch Linux**: A user contemplated switching to Arch for better GPU management but preferred Ubuntu for compatibility with ROS.
   - The conversation highlighted the pros and cons of using different Linux distributions for machine learning and software development.
- **Learning CUDA Success Stories**: A beginner expressed interest in hearing success stories from those who have learned CUDA recently and completed meaningful projects.
   - This highlights the community's interest in learning from each other's experiences and projects undertaken with CUDA.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://askubuntu.com/questions/1229974/fresh-20-04-install-with-nvidia-and-igpu-only-get-igpu-resolutions,">fresh 20.04 install with nvidia and iGPU.. only get iGPU resolutions</a>: I am curious how am I supposed to switch to using the nvidia card on my ubuntu 20.04 setup?&#xA;&#xA;on 18.04 once installed, I could just plug in and go.. seems with 20.04 it is using both iGPU and P...</li><li><a href="https://askubuntu.com/questions/1061551/how-to-configure-igpu-for-xserver-and-nvidia-gpu-for-cuda-wor)">How to configure iGPU for xserver and nvidia GPU for CUDA work</a>: I have an Intel onboard GPU and NVIDIA GPU. I am running Ubuntu 18.04.&#xA;&#xA;How do I configure a dual GPU setup so that Intel onboard iGPU will drive the monitor, leaving NVIDIA GPU exclusively fo...</li><li><a href="https://www.reddit.com/r/Fedora/comments/x487g1/how_to_force_waylandgnomeshell_to_use_intel_igpu/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/deeplearning/comments/z4lpry/is_linux_still_vastly_preferred_for_deep_learning/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://askubuntu.com/questions/1536665/ubuntu-22-04-how-to-make-xorg-and-gnome-shell-use-igpu-exclusively-on-a-dual-b">Ubuntu 22.04 how to make Xorg and GNOME Shell use iGPU exclusively on a dual booted laptop with Intel i7 13th Gen</a>: I have followed the instructions mentioned here (same instructions are also mentioned here). Unfortunately, these instructions are catered to Ubuntu 20.04 and I am currently using Ubuntu 22.04. Her...</li><li><a href="https://forums.developer.nvidia.com/t/ubuntu-22-04-how-to-make-xorg-and-gnome-shell-use-igpu-exclusively-on-a-dual-booted-laptop-with-rtx-4060-and-intel-i7-13th-gen/318222/2">Ubuntu 22.04 how to make Xorg and GNOME Shell use iGPU exclusively on a dual booted laptop with RTX 4060 and Intel i7 13th Gen</a>: Hi everyone, I noticed this question hasn’t received any responses yet. As I’m new to the community, I might have missed including some important details or context. If anyone has suggestions on how I...</li><li><a href="https://github.com/rkinas/triton-resources">GitHub - rkinas/triton-resources: A curated list of resources for learning and exploring Triton, OpenAI&#39;s programming language for writing efficient GPU code.</a>: A curated list of resources for learning and exploring Triton, OpenAI&#39;s programming language for writing efficient GPU code. - rkinas/triton-resources</li><li><a href="https://wiki.archlinux.org/title/NVIDIA_Optimus">NVIDIA Optimus - ArchWiki</a>: no description found</li><li><a href="https://www.reddit.com?utm_source=share&utm_medium=android_app&utm_name=androidcss&utm_term=1&utm_content=1">reddit</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1322345351096565842)** (2 messages): 

> `Scaffolding Code for Lecture 20, Scan Algorithm` 


- **Inquiry about Scaffolding Code Availability**: A member asked if the **scaffolding code** for lecture 20 by Professor El Hajj, which demonstrates the **scan algorithm**, is available online.
   - They specified that the code should help create input for the kernel, invoke the kernel, and compare results.
- **Claude Reconstructs the Code**: The same member later mentioned that **Claude** was able to reconstruct the scaffolding code successfully.
   - This news was shared with a light-hearted tone, marked with a smiley face.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

iron_bound: https://www.youtube.com/watch?v=VpAZPPCLCUI
  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1323178479516389448)** (1 messages): 

> `Ladder Branch Feature` 


- **Feature In Ladder Branch, Not Merged Yet**: The feature is currently available in the **ladder branch** but has not been implemented or merged into the **main branch** yet.
   - This status indicates ongoing work, with potential future updates expected as the feature progresses toward merging.
- **Uncertainty About Future Implementation**: The lack of merging into the **main branch** raises questions about the timeline for this feature's full implementation.
   - Members expressed interest in tracking the progress of this branch integration.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1323190782391222304)** (5 messages): 

> `Integer Matmul Operators in TK, TK vs Triton Performance Comparison, Triton Optimizer Capabilities` 


- **Integer Matmul Operators Coming to TK**: A user inquired if **ThunderKittens** include **integer matmul operators**, and another member confirmed it's on the list to be added.
   - They also extended an invitation for others to contribute to this feature.
- **Debate Over TK and Triton Performance**: There has been some discussion about whether a well-crafted custom **TK/CUDA kernel** can outperform **Triton** implementations.
   - While some comparisons show TK winning, the effectiveness of Triton's optimizer remains uncertain.
- **Triton's Challenges with Fine-Grained Control**: A member noted that if a kernel requires **fine-grained asynchronous execution** or detailed control over **register utilization**, TK may perform better than Triton.
   - The lack of exposed levers in Triton makes it harder to reach peak performance in these scenarios.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1322341187759112254)** (4 messages): 

> `Raspberry Pi 5 GPU Performance, AI Project Testing on Raspberry Pi 5, Vulkan GPU Experience` 


- **Raspberry Pi 5's GPU for AI needs evaluation**: A user sought **quantitative information** regarding the **Raspberry Pi 5**’s GPU utility for compute tasks, questioning its effectiveness.
   - Responses indicated that while the Pi 5 performs well for vision tasks, it faces challenges with larger **LLM models**, being slow even with **6-8bit quantization**.
- **Testing AI performance on Raspberry Pi 5**: A contributor reported testing the **Pi 5** for AI purposes, noting the performance varies based on specific tasks.
   - They specified that while it excels in vision applications, it struggles with larger language models in its current state.
- **Inquiry about Vulkan testing frameworks**: A user expressed interest in learning about frameworks or benchmarks used to test the Pi 5's GPU, particularly for Vulkan.
   - They admitted to having little **Vulkan** experience, aiming to figure out effective testing methods for the GPU.
- **Comparative performance of Pi 5’s GPU and CPU**: It was discussed that the raw **FLOPS** of the Raspberry Pi 5's GPU are significantly lower than that of a recent **Intel CPU**, potentially by an order of magnitude.
   - Nonetheless, there are expectations that the Pi 5’s GPU might still perform comparably against its CPU in certain scenarios.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1322293555779338370)** (58 messages🔥🔥): 

> `AI-generated Code Challenges, Kagi Assistant vs. Perplexity, LLMs in Software Development, AI Engineering Summit, Cursor AI Programming Tools` 


- **AI-generated Code challenges affect engineering on-calls**: A user highlighted that engineering on-call experiences are degrading due to the blind integration of AI-generated code, citing the need for better documentation and testing.
   - Another user agreed, suggesting that engineers should break down tasks in a way that LLMs can handle effectively rather than expecting them to manage complex requests independently.
- **Kagi Assistant shows promise**: Several users expressed enthusiasm for Kagi Assistant, highlighting its customizability and search capabilities compared to Perplexity.
   - While some noted functionality gaps in the Kagi Assistant, others emphasized its potential, especially with upcoming features such as a search API.
- **LLMs: Effective but require precise execution**: Users discussed the dual nature of LLMs, noting their ability to generate results quickly but also the difficulties in more complex programming tasks.
   - Strategies such as refining prompts and generating thorough end-to-end tests were suggested as best practices for working with LLMs.
- **AI Engineering Summit announcement**: An AI Engineering Summit is scheduled for February 20-21, 2025, in New York, focusing on collaboration between AI engineers and leaders.
   - Participants are encouraged to pre-register for exclusive access, with previous sponsors including major tech companies.
- **Cursor AI programming tool frustrations**: Frustration around Cursor, an AI coding assistant, was discussed, with users sharing their experiences of it being counterproductive in coding tasks.
   - The general consensus suggests that successful collaboration with AI tools requires engineers to redefine their approach to include better-defined problem statements and iterative solutions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.twitch.tv/videos/2339351410">Twitch</a>: no description found</li><li><a href="https://www.twitch.tv/wesen3000">Twitch</a>: no description found</li><li><a href="https://apply.ai.engineer">AI Engineer Summit</a>: The highest-signal technical AI event of the year. For AI Engineers &amp; AI Leaders, Feb 20 - 21, 2025.</li><li><a href="https://x.com/sh_reya/status/1873431565650502060?s=46&t=jDrfS5vZD4MFwckU5E8f5Q">Tweet from Shreya Shankar (@sh_reya)</a>: how come nobody is talking about how much shittier eng on-calls are thanks to blind integrations of AI-generated code? LLMs are great coders but horrible engineers. no, the solution is not “prompt the...</li><li><a href="https://x.com/TrelisResearch/status/1873709556368327007">Tweet from Trelis Research (@TrelisResearch)</a>: ===+ Pressure Testing the &#34;HELLO&#34; Result with &#34;HELOL&#34;===Copying @giffmana @flowersslop . It was a cool test and finding.---+ Test Methods---1. Change the example from &#34;HELLO&#34; t...</li><li><a href="https://x.com/sh_reya/status/1873431565650502060?s=46&t">Tweet from Shreya Shankar (@sh_reya)</a>: how come nobody is talking about how much shittier eng on-calls are thanks to blind integrations of AI-generated code? LLMs are great coders but horrible engineers. no, the solution is not “prompt the...</li><li><a href="https://x.com/flowersslop/status/1873115669568311727?s=46">Tweet from Flowers (@flowersslop)</a>: I finetuned 4o on a synthetic dataset where the first letters of responses spell &#34;HELLO.&#34; This rule was never stated explicitly, neither in training, prompts, nor system messages, just encoded...</li><li><a href="https://www.philschmid.de/fine-tune-llms-in-2025">How to fine-tune open LLMs in 2025 with Hugging Face</a>: The only guide you need to fine-tune open LLMs in 2025, including QLoRA, Spectrum, Flash Attention, Liger Kernels and more.</li><li><a href="https://www.astralcodexten.com/p/notes-from-the-progress-studies-conference?utm_source=post-email-title&publication_id=89120&post_id=150459736&utm_campaign=email-post-title&isFreemail=true&r=43kx5&triedRedirect=true">Notes From The Progress Studies Conference</a>: ...</li><li><a href="https://skylarbpayne.com/posts/cursed-cursor">How to stop saying 'Fuck you Cursor' - Skylar Payne (Wicked Data LLC)</a>: no description found</li><li><a href="https://x.com/sh_reya/status/1873564811415449872">Tweet from Shreya Shankar (@sh_reya)</a>: this is quite interesting. I watched it on 2x so I may have missed some things. i like the cursor rule “don’t implement it yet, ask me for confirmation” — i will certainly be adding that to my cursorr...</li><li><a href="https://www.threads.net/@mockapapella/post/DBRZ62OvyLM?xmt=AQGzBSWxwbt-GDKQKGdCOKgIUW7iyAfyuj1MPjiXJO455Q">Mockapapella (&#064;mockapapella) on Threads</a>: You know, I was thinking about this.With the right prompt and context, this might be an excellent use case for GPT o1 plus the advanced data analysis tool.Coverage output + source file + AST map of al...</li><li><a href="https://youtu.be/58zHJL1dKtw?si=2QjyTl9m7-9QclZS"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/)** (1 messages): 

swyxio: https://news.ycombinator.com/item?id=42343692
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1323260193773064264)** (4 messages): 

> `Chatbot Arena updates, Claude's performance` 


- **Chatbot Arena Sees Exciting Rankings**: In the latest update, **OpenAI's o1** rises to joint #1, gaining **24 points** from o1-preview, while **DeepSeek-V3** lands at #7, being the only open model in the top-10.
   - Notable highlights include o1's achievement as the highest scorer in style control and DeepSeek-V3's cost-effectiveness at **$0.14** per **1M input token**.
- **Claude's Rankings Spark Debate**: A member expressed confusion over **Claude's low ranking**, stating that it 'does not make sense' to them.
   - Another member chimed in, noting that refusals can be detrimental to roleplay and other small factors may affect performance.



**Link mentioned**: <a href="https://x.com/lmarena_ai/status/1873695386323566638">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: Exciting News from Chatbot Arena❤️‍🔥@OpenAI&#39;s o1 rises to joint #1 (+24 points from o1-preview) and @deepseek_ai DeepSeek-V3 secures #7, now the best and the only open model in the top-10!o1 High...

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1322639831645421578)** (6 messages): 

> `Small Language Models (SLMs), The Bitter Lesson, Scaling Models` 


- **Bitter Lesson Swaying Model Performance**: The Bitter Lesson suggests that **scaling up data and compute** yields better results than integrating priors, but necessitates additional resources.
   - _As articulated by members, this trade-off reflects the core message of the lesson about the value of scale in AI model performance._
- **SLMs Can Outperform Larger Models**: With a focused task, **small language models (SLMs)** can surpass larger models due to the ability to integrate effective priors.
   - _A member noted that this strategy allows SLMs to excel in targeted scenarios, showcasing the balance between specialization and scale._
- **Potential for SLM Growth**: There are indications that SLMs still have room to grow, as evidenced by the **Llama 3 8B** outperforming **GPT-3 175B**.
   - _This reveals that despite being smaller, targeted optimization can lead to impressive performance gains._
- **The Importance of Domain Trade-offs**: Ultimately, the effectiveness of SLMs or larger models relies heavily on the **specific trade-offs** relevant to the problem domain.
   - _A member emphasized that the choice between model size and task specificity determines overall model success._


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1323037042858266646)** (3 messages): 

> `OAI employee hack, Crypto shilling, Holiday greetings` 


- **Another OAI Employee Hacked**: An **OAI employee** has reportedly been hacked and is now **shilling crypto** on their timeline, raising concerns among the community.
   - *Mr President, we have a situation* – this incident highlights ongoing security vulnerabilities within the organization.
- **Merry Christmas Message**: One member shared a simple greeting: **Merry Xmas**, spreading festive cheer in the channel.
   - This light-hearted message adds a touch of holiday spirit to the ongoing discussions.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1322296723720376392)** (23 messages🔥): 

> `DeepSeek V3 performance, Benchmarking instruction following tasks, Evaluation of model training, Interconnects market discussion, Scaling confusion in AI` 


- **DeepSeek V3 struggles with XML output**: A member expressed frustration that **DeepSeek V3** often fails to output **XML** tags after generating reasoning, despite its smart capabilities.
   - They noted it produces reasoning reminiscent of **o1/r1-like outputs**, indicating room for improvement in task completion.
- **Call for benchmarking DeepSeek V3**: There was a discussion about whether anyone has benchmarked **DeepSeek V3** for instruction following tasks after swapping out prompts from V2.5.
   - Members voiced skepticism regarding its post-training performance following feedback that appeared largely negative.
- **Concerns over training evaluation methods**: Members debated the usefulness of evaluation tables which seem misleading and fail to capture the complete picture of model **behavior**.
   - A comment highlighted distrust over **Twitter** reactions to training efficiency based on such tables, implying deeper analysis is needed.
- **Discussion on interconnects market**: There was a light-hearted comment suggesting someone needs to create a market for **interconnects**, indicating a need for industry clarity.
   - Another member commented on the confusing scaling practices in the AI space, reflecting common frustrations regarding industry trends.
- **Critique of OpenAI's plotting**: A member criticized the **OpenAI** plots for being **misleading**, questioning their accuracy in conveying scaling effects and training dynamics.
   - They pointed out that scaling discussions can often lead to confusion, reflecting a broader concern within the community.



**Link mentioned**: <a href="https://x.com/aidan_mclau/status/1873122732680134960">Tweet from Aidan McLau (@aidan_mclau)</a>: you should basically pretend that getting a model to think for longer is the same as building a bigger modelfollowing the math is quite fun and uncovers some neat things about industry progress

  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1322974707200299019)** (6 messages): 

> `Reading Research Papers, List Growth, RLHF Experiments` 


- **Papers present challenges for comprehension**: Several members noted that **research papers** are hard to read, with one expressing a feeling of overwhelm saying their list has **+50%** growth since the last review.
   - *Yes, papers are hard to read* was echoed by multiple users emphasizing the difficulty in processing complex information.
- **Effective strategy over ambition in RLHF**: One user mentioned their past efforts in **RLHF** research but eventually decided to stop due to the complexity involved.
   - They suggested that reading enough papers to plan experiments could be sufficient for progress.


  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1322975160239788092)** (2 messages): 

> `Outcome rewards, RLVR` 


- **Understanding RLVR in Outcome Rewards**: A member noted that **outcome rewards**, often referred to as **RLVR**, appear straightforward when considering the broader context of their application.
   - *Seems simple enough actually, in the big picture* suggests a level of clarity on the integration of these concepts.
- **Simplicity in Complex Systems**: Discussion alludes to the simplicity of **RLVR** in the grand scheme of things, reiterating that it seems more manageable than it appears.
   - This perception may indicate a deeper understanding of how these rewards function within reinforcement learning frameworks.


  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1323032111590015078)** (9 messages🔥): 

> `GRPO, Vineppo, Memory Constraints in RL, Optimizers in RLHF` 


- **Inquiry on GRPO Effectiveness**: A member queried the effectiveness of **GRPO (Group Relative Policy Optimization)**, mentioning its use in **DeepSeek** and **Qwen-2-Math**.
   - *What’s the TLDR of how it works?* prompted further discussion on the mechanics of the algorithm.
- **GRPO vs. Vineppo Comparison**: **GRPO** is compared to **vineppo**, revealing that GRPO averages rewards from multiple outputs while vineppo uses a single sample and resets to intermediate states.
   - This led to a discussion on the challenges of value functions, with one member noting that GRPO is what **DeepSeekv3** implemented.
- **Memory Constraints in RL Models**: A member expressed challenges with **memory issues** while running RL in post-training phases on **1b - 7b models**, suggesting that for suitable domains, forgoing the value network may be beneficial.
   - They also inquired about possible workarounds to accommodate longer context lengths, highlighting memory constraints as a significant concern.
- **Future Book on RLHF Optimizers**: One member mentioned the need to write the **RLHF book on optimizers**, suggesting that both GRPO and vineppo should be included.
   - This reflects a growing interest in documenting various optimization strategies within reinforcement learning.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1323374564696395848)** (6 messages): 

> `Gary Marcus's Collaboration, AI Predictions for 2027, Discussion on AI Development Timelines` 


- **Shock Over Gary & Miles Collaboration**: Members expressed surprise at the collaboration between **Gary Marcus** and **Miles Brundage**, suggesting it was unexpected and revealing mixed feelings.
   - *One noted that Gary is quite critical,* reflecting the complexity of their partnership.
- **Doubts on AI Progress Timeline**: Member @420gunna questioned the feasibility of **levels 7/8/9** being reached, claiming that the remaining expectations are overly optimistic.
   - Another voice emphasized the sentiment of being 'insanely far away from 4,' echoing doubts about current AI development milestones.



**Link mentioned**: <a href="https://garymarcus.substack.com/cp/153809626">Where will AI be at the end of 2027? A bet</a>: We, Gary Marcus, author, scientist, and noted skeptic of generative AI, and Miles Brundage, an independent AI policy researcher who recently left OpenAI and is bullish on AI progress, have agreed to t...

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1322569436972060725)** (44 messages🔥): 

> `API Integration with GPT4All, Updates on Nomic Models, Issues with Chat Templates, Gemini Model Support, Exploration of Vision Models` 


- **Integrating LLaMA 3.3 with GPT4All**: To use LLaMA 3.3 (70b) with LocalDocs in GPT4All, sign into Groq.com, generate an API key, and input it in the RMODEL maker's add models section for cloud LLM access.
   - This provides a cost-effective way to utilize cloud AI models.
- **Gemini API Support Queries**: There was discussion about the support for the Gemini API in GPT4All, with insight that existing Gemini models are compatible with OpenAI's API format but further support for Gemini 2.0 is pending.
   - Members expressed interest in using Gemini’s features and contributing to the integration process.
- **Issues with Chat Templates After Update**: Users reported syntax errors with chat templates used in GPT4All after updates introduced a switch to a Jinja parser.
   - The community is working on compatibility issues, with suggestions to reset templates or provide links for assistance.
- **Exploring the Vision Model**: There was clarification on the functionality of the nomic-embed-vision-v1 model, emphasizing that it works in conjunction with the text embedding model to enhance image searches using text queries.
   - Users expressed curiosity about the availability of Nomic's vision models in comparison to other models in the HuggingFace repository.
- **Community's Interest in Ollama Models**: Members discussed the possibility of using already installed Ollama models with GPT4All and shared a script for exporting those models as 'model.bin'.
   - There was also debate on whether to set Ollama as the LLm engine for GPT4All, highlighting the potential for OpenAI-compatible API integration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://caddyserver.com/">Caddy 2 - The Ultimate Server with Automatic HTTPS</a>: Caddy is a powerful, enterprise-ready, open source web server with automatic HTTPS written in Go</li><li><a href="https://gist.github.com/supersonictw/f6cf5e599377132fe5e180b3d495c553">Ollama Model Export Script</a>: Ollama Model Export Script. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/google-gemini/cookbook">GitHub - google-gemini/cookbook: Examples and guides for using the Gemini API</a>: Examples and guides for using the Gemini API. Contribute to google-gemini/cookbook development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1322429866100002941)** (14 messages🔥): 

> `breathe.ai testing, finding likeminded people, HMM tokenization, internship request` 


- **Breathe.ai joins Cohere Discord for testing**: Breathe.ai received an email from **Maxime** regarding testing a research prototype and signed an NDA to join the server.
   - A warm welcome was extended to Breathe, with members sharing enthusiasm for collaboration.
- **Seeking likeminded talkative community**: A member expressed curiosity about the availability of genuine and talkative like-minded individuals within the server.
   - In response, another member inquired about ongoing projects, indicating an openness to conversation.
- **Request for HMM tokenization knowledge**: Someone asked if anyone was familiar with **HMM (Hidden Markov Model)** tokenization, aiming to foster technical discussion.
   - Unfortunately, no one indicated they possessed that knowledge, leading to a quiet moment.
- **Internship promotion via LinkedIn**: A member requested help in sharing their **LinkedIn post** regarding an internship opportunity.
   - The post included a direct link for connections to support the search for internships.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1322456380267036774)** (5 messages): 

> `API Rate Limits, HMM Tokenization` 


- **Questions about API Rate Limits**: A member inquired whether the **50 requests per minute** rate limit for the Embed Job API applies to all endpoints and if it can be increased.
   - Another member provided a [link to the rate limits documentation](https://docs.cohere.com/v2/docs/rate-limits) and recommended contacting support at support@cohere.com for any enhancement requests.
- **Inquiry on HMM Tokenization**: A user asked if anyone has knowledge regarding **HMM (Hidden Markov Model)** tokenization techniques.
   - This drew attention but did not elicit any immediate responses or advice from the members in the chat.



**Link mentioned**: <a href="https://docs.cohere.com/v2/docs/rate-limits">API Keys and Rate Limits — Cohere</a>: This page describes Cohere API rate limits for production and evaluation keys.

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1322313158970445824)** (12 messages🔥): 

> `Image Embed Rate Limits, Fine-tuning Issues, Support Response Times` 


- **Confusion Over Image Embed Rate Limits**: A member inquired about the image embed rate limits, noting that they expect **400 per minute** for production keys but seem to be experiencing only **40**.
   - Another member confirmed that this is a **known issue** and that teams are working on a fix, assuring that the limits are indeed set to **400**.
- **Support for Fine-tuning Errors**: A member shared an error they are encountering and expressed concern that it might be related to their data or **fine-tuning issues**.
   - The support team responded, indicating that they are looking into the issue while managing potential **delays** due to the holidays.
- **Updates on Shlomi's Issue**: Support confirmed they are in direct communication with Shlomi regarding the ongoing issue and have **escalated** it for further investigation.
   - It was noted that the problem appears to be on the support team's side, and they promised to keep the community updated.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1322679831397138502)** (16 messages🔥): 

> `Speedup in Matching Functions, Model Rewrite Time Improvement, Meeting Discussion Points, Reversible Transformation in UOPs, Merge AM Driver Plans` 


- **Questioning 8x Speedup in Matching**: Discussion initiated around the **8x speedup** claimed in the matching functions, with one user noting that **50%** of their time is spent in those functions, indicating achieving even a **2x speedup** might be unrealistic.
   - Another clarified that the **bounty captures** the transition from **400ms** to **50ms**, illustrating the speedup mathematically.
- **Achieving 2.5x Speedup in Model Rewrite**: A member reported a **2.5x speedup** in model rewrite time after altering `full_graph_rewrite`, but noted **4/7 tests failed**, seeking debugging advice from peers.
   - Suggestions included carefully selecting test cases to analyze failures, with commentary on the use of multi-threading for potential performance gains.
- **Meeting #51 Agenda Confirmation**: Plans for **Meeting #51** were shared, including critical items such as **scheduler cleanups** and merging the **AM driver**, scheduled for **930am Monday** San Diego time.
   - One user expressed they might miss the meeting due to a prior commitment but was focused on optimizing performance with **llm.c**.
- **Clarifications on Reversible UOP Transformations**: Discussions ensued regarding the requirements for a **reversible transformation** between machine code and **uops**, raising questions about potential intermediate assembly steps.
   - Clarifications were sought on whether the transformation needs to be **deterministically 1:1 reversible** to some uop source code or just equivalent to the final rewritten uop state.
- **Plans to Merge AM Driver by Year-End**: George Hotz expressed intentions to increase the **line count** of the AM driver to **11,000** and aims to have it merged by the end of the year, rallying support from the community.
   - A recently linked GitHub commit related to the project was shared, emphasizing ongoing development efforts.



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/commit/0addbad36d414cc37e69e92fa9e1f26045cbf1f6">Happy New Year! Let&#39;s get AM merged · tinygrad/tinygrad@0addbad</a>: no description found

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1322486129848614912)** (12 messages🔥): 

> `Tinygrad Performance vs Torch, Understanding JIT Execution, Frame Evaluation Hook API` 


- **Tinygrad CUDA dramatically outperforms Torch**: New updates reveal that **Tinygrad CUDA** is now **2x faster** than Torch, with **OpenCL** also showing improvements with a performance boost of about **1ms**.
   - Context included a suggestion to use `Device[out.device].synchronize()` for synchronization in tinygrad, implying a comparison in execution speed factors.
- **Explaining JIT Functionality**: A user discussed their understanding of how **JIT batching** works, noting execution items are collected after the first run, with benefits fully realized on the third run.
   - George Hotz clarified that batching occurs on the **third run**, explaining that it isn't done post-capture because *batching can't occur until after capture*.
- **Introducing the Frame Evaluation Hook API**: A member shared insights about the **Frame Evaluation Hook API** as a more reliable method for capturing runs in Python, which is utilized in Torch's dynamo compiler.
   - They provided a link to the [PEP 523](https://peps.python.org/pep-0523/) documentation, suggesting its potential usefulness for future development.



**Link mentioned**: <a href="https://peps.python.org/pep-0523/">PEP 523 – Adding a frame evaluation API to CPython | peps.python.org</a>: This PEP proposes to expand CPython’s C API 2 to allow for the specification of a per-interpreter function pointer to handle the evaluation of frames 5. This proposal also suggests adding a new field ...

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1322619274677981275)** (2 messages): 

> `Local RAG with Llama-3.2, Neomagus for legal verification` 


- **Build a Local RAG App with Llama-3.2**: A thread by @akshay_pachaar discusses creating a Llama-3.2-powered app that can answer questions based on complex **Excel tables** using [Llama Index tools](https://t.co/4WJ7ZcXy3H).
   - The integration aims to make the process of querying data seamless and efficient, enhancing user interaction with spreadsheets.
- **Ensure Legal Accuracy with Neomagus**: Neomagus offers a solution to verify legal references in AI-generated content, addressing the risk of **non-existent citations** produced by tools like ChatGPT and Claude [more details here](https://t.co/g5toC0m3T9).
   - It extracts citations and matches them against verified sources to maintain **accuracy and trustworthiness** in legal research.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1322913274999869521)** (18 messages🔥): 

> `Llama 3.3 GPU Memory Requirements, RAG Solution Development, Ollama Local Model Running, LlamaParse API Details, Open Source AI Monetization` 


- **Understanding Llama 3.3 GPU Memory Usage**: A user inquired how much **GPU memory** the **Llama 3.3 70B** model requires and if it's available via a **Hugging Face endpoint**.
   - Another user suggested testing locally with **Ollama**, noting that running **ollama run llama3.3** may use approximately **2.77GB** of RAM.
- **In-house RAG Tool Issues**: A developer shared challenges with their in-house **Retrieval-Augmented Generation (RAG)** solution that diverges from the original query.
   - They explored different approaches but encountered issues with **maximum iterations** and unresponsive outputs despite extensive troubleshooting.
- **Ollama Tokenization Insights**: In response to a tokenizer-related question, it was noted that the **Ollama wrapper** handles the tokenizer, so users do not need to intervene.
   - The general consensus is that tokenization is inherently tied to the pre-trained model and managed within the **Ollama** infrastructure.
- **Exploring LlamaParse API Features**: Discussion highlighted the availability of the **LlamaParse API** for direct integration, with various sample calls provided for uploading and checking parsing jobs.
   - Users can leverage the API for efficient data manipulation, with detailed documentation available for further exploration.
- **New Monetization Platform Launch**: A representative announced the launch of **Bagel**, a platform for **open source AI developers** to monetize their contributions effectively.
   - The platform integrates with **Hugging Face**, offering access to advanced models like **Llama-3.3** and **Stable Diffusion**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/BagelOpenAI/status/1873776090516488257">Tweet from Bagel 🥯 (@BagelOpenAI)</a>: Today, the Bakery opens its doors to all open source AI developers.Bagel makes open source AI monetizable. Our novel AI model architecture enables anyone to contribute while ensuring developers receiv...</li><li><a href="https://github.com/run-llama/llama_index/blob/fd1edffd20cbf21085886b96b91c9b837f80a915/llama-index-core/llama_index/core/agent/react/output_parser.py#L104">llama_index/llama-index-core/llama_index/core/agent/react/output_parser.py at fd1edffd20cbf21085886b96b91c9b837f80a915 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/fd1edffd20cbf21085886b96b91c9b837f80a915/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py#L306">llama_index/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py at fd1edffd20cbf21085886b96b91c9b837f80a915 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.cloud.llamaindex.ai/llamaparse/getting_started/api">Using the REST API | LlamaCloud Documentation</a>: If you prefer to use the LlamaParse API directly, that&#x27;s great! You can use it in any language that can make HTTP requests. Here are some sample calls:</li><li><a href="https://docs.cloud.llamaindex.ai/llamaparse/getting_started/python">Using in Python | LlamaCloud Documentation</a>: First, get an api key. We recommend putting your key in a file called .env that looks like this:</li><li><a href="https://github.com/run-llama/llama_parse/tree/main/examples">llama_parse/examples at main · run-llama/llama_parse</a>: Parse files for optimal RAG. Contribute to run-llama/llama_parse development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1323294527884361799)** (1 messages): 

> `Filtering Nonword Sounds, Audio Editing with LLMs` 


- **Exploring LLMs for Nonword Sound Filtering**: A member inquired about experiences using LLMs to filter **nonword sounds** (e.g., *ahh*) and filler words (e.g., *so*, *look*, *ok*) in audio files.
   - The discussion highlights the potential utility of AI in **audio editing**, especially for enhancing clarity by removing unwanted sounds.
- **Interest in AI for Audio Clarity**: Members expressed curiosity about how AI can improve audio clarity by filtering out **filler words** in communication recordings.
   - One noted that this could significantly enhance the **listening experience** in educational and professional contexts.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1322360569604345876)** (14 messages🔥): 

> `Certificates Distribution, Upcoming LLM Agents MOOC, Access to Course Lectures` 


- **Certificates arriving throughout January**: Members were informed that certificates will be distributed via email by the end of January.
   - One member noted not having received theirs despite meeting the requirements.
- **Another LLM Agents MOOC starts soon**: A new course is slated to begin in late January, providing another opportunity for interested participants.
   - To sign up for the course, individuals are directed to fill in a [sign-up form](https://forms.gle/9u6HdVCWXgws16go).
- **Availability of lecture materials**: A member inquired about accessing previous course lectures, which are available in the course syllabus on the [course website](https://llmagents-learning.org/f24).
   - Another member confirmed they had found the lecture materials, thanking the group for assistance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/sp25">Large Language Model Agents MOOC</a>: MOOC, Spring 2025</li><li><a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC, Fall 2024
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1322315083254075443)** (4 messages): 

> `Dynamo Errors, Nested Compiles, OpenAI's Simple Eval Library, Flex Changes in 2.6.0, lm eval comparison` 


- **Dynamo Errors Resolved?**: A member mentioned previously encountering **Dynamo errors** but suggested that if those are resolved, removing the compiler disabled setting could be the way forward.
   - They highlighted the need for continued performance validation with both compile settings true and false.
- **Flex Changes Timeline for 2.6.0**: A member expressed hopes that the current changes to **Flex** would land before the **2.6.0** release on **January 13**.
   - They emphasized that multiple **Flex changes** have been added since **2.5.1**, suggesting improved efficiency.
- **Interest in Simple Eval Recipe**: A member proposed interest in sharing a recipe leveraging [OpenAI's Simple Eval library](https://github.com/openai/simple-evals).
   - They provided a link to the GitHub page, prompting discussion on its applicability and benefits.
- **Comparing Simple Eval to lm eval**: A member inquired about the possible advantages of using OpenAI's **Simple Eval** over existing **lm eval** tools.
   - This question highlights ongoing discussions about the effectiveness and efficiency of different evaluation libraries.



**Link mentioned**: <a href="https://github.com/openai/simple-evals">GitHub - openai/simple-evals</a>: Contribute to openai/simple-evals development by creating an account on GitHub.

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1322361203720458240)** (5 messages): 

> `FP8 quantization schemes, NVIDIA's Transformer Engine, Azure's Mixed Precision Library, FP8 block quantization, Mixed-precision training` 


- **Understanding FP8 Quantization Precision**: *Quantization granularity* in FP8 schemes is recognized as smaller and more precise, with most current schemes employing **per-tensor scaling**.
   - Upcoming technical reports, such as from DeepSeek, may provide further insights into FP8 comparisons.
- **Exploring Resources on FP8 Schemes**: Limited posts exist comparing FP8 quantization schemes specifically for training, though several resources on related applications are available.
   - Notably, [NVIDIA's Transformer Engine](https://github.com/NVIDIA/TransformerEngine) is a key reference in FP8 usage, despite the absence of formal papers.
- **Links to Relevant FP8 Research**: Several GitHub repos and papers were highlighted for further FP8 insights, such as [Microsoft's Automatic Mixed Precision Library](https://github.com/Azure/MS-AMP) and study on activations and optim states from [NVlabs - COAT](https://github.com/NVlabs/COAT).
   - Recent papers, including [arXiv:2310.18313](https://arxiv.org/pdf/2310.18313) and [arXiv:2409.12517](https://arxiv.org/pdf/2409.12517), provide additional frameworks regarding FP8 applications.
- **Innovations in FP8 Block Quantization**: A PyTorch blog post details advancements in *2D block quantization* for FP8, claiming nearly **2x** speedups in tensor quantization accuracy and efficiency.
   - The techniques introduced enhance GEMM operations during both inference and training, emphasizing improved processing speeds.
- **Mixed-precision Training Insights**: A brief discussion on various quantization schemes for *INT8/FP8 training* suggests shifts in techniques can enhance model performance.
   - For deeper insights, refer to the presentation on [Low-bit mixed-precision training](https://github.com/gpu-mode/lectures/blob/main/lecture_030/%5BGPU-MODE%5D%20Quantized%20training%20(20241006).pdf) for more detailed coverage.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Azure/MS-AMP">GitHub - Azure/MS-AMP: Microsoft Automatic Mixed Precision Library</a>: Microsoft Automatic Mixed Precision Library. Contribute to Azure/MS-AMP development by creating an account on GitHub.</li><li><a href="https://github.com/NVlabs/COAT">GitHub - NVlabs/COAT</a>: Contribute to NVlabs/COAT development by creating an account on GitHub.</li><li><a href="https://pytorch.org/blog/accelerating-gemms-triton/">Accelerating 2D Dynamic Block Quantized Float8 GEMMs in Triton</a>: 2D block quantization for Float8 (FP8) holds the promise of improving the accuracy of Float8 quantization while also accelerating GEMM’s for both inference and training.  In this blog, we showcase adv...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1322726692732665886)** (7 messages): 

> `OS Mode Inputs, Isolation Function Clarification, Windows Build for Version 1.0, Profiles.yaml vs .py Files, Custom API Base URLs` 


- **Clarification on OS Mode Inputs**: A user inquired whether **OS mode** utilizes **video** as an input, seeking clarification on its functionality.
   - This reflects ongoing curiosity surrounding the capabilities of the current system implementation.
- **Doubts about Isolation Function**: Users discussed the **Isolation doc** and questioned whether it relates to the operating system functions or pertains to **Docker and E2B measures**.
   - There was an image attached for further clarification, indicating confusion over terminology.
- **Request for Windows Build of Version 1.0**: A message asked if there is a **Windows build** available for the newly released **1.0 dev version**.
   - This indicates interest in cross-platform compatibility for software access.
- **Profiles.yaml Transition to .py Files**: There were struggles expressed in understanding the transition from **profiles.yaml** in **1.0.0** to a new format, potentially using **.py files**.
   - Concerns were raised about the documentation's accuracy regarding the saving process.
- **Custom API Base URL Challenges**: A user indicated complications while attempting to create a **custom API base URL** in an OpenAI format that mimics models like **gpt4o** and **claude-35-sonnet**.
   - This highlights challenges faced during implementation on **Ubuntu** that may need community support.



**Link mentioned**: <a href="https://docs.openinterpreter.com/safety/isolation),">no title found</a>: no description found

  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

ari9596: Anyone have opinions on  this  https://arxiv.org/abs/2412.15563
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1322330151010304082)** (3 messages): 

> `AI Glossary Creation, Exploring DSPy and Openhands Integration, Feedback Recording System for Code Changes` 


- **AI Glossary for Clear Communication**: Inspired by the repetitive need for definitions in AI discussions, a member created an [AI glossary](https://www.dbreunig.com/glossary.html) for their site, acknowledging a backlog to address.
   - *“If you want to know where the future is being made, look for where language is being invented...”* reflects the interplay of language and evolving technology.
- **Openhands Integration with DSPy**: A member inquired about molding Openhands into a one-shot noninteractive tool that returns a chat response and a git diff, questioning its integration into DSPy's pipeline.
   - While design considerations exist, they recognize the potential DIY power of DSPy in tuning prompts through built-in facilities.
- **Custom Feedback System for Code Changes**: The same member proposed creating a feedback recording system for evaluating code quality, based on automated code changes.
   - This approach would involve gathering input/output data and grading to potentially train a DSPy pipeline based on past user experiences.



**Link mentioned**: <a href="https://www.dbreunig.com/2024/12/27/generating-a-glossary-from-a-jekyll-blog-usign-dspy-claude.html">Generating a Glossary from a Jekyll Blog Using DSPy &amp; Claude</a>: Asking LLMs to take the first pass at an AI glossary for my site.

  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1322537467298971690)** (4 messages): 

> `FFmpeg usage, Hackathon and Conference Recommendations` 


- **FFmpeg for Video Editing**: A member mentioned that they need to gather **time stamps** and then use **FFmpeg** to cut their video.
   - They expressed gratitude for the clear explanation they received regarding the process.
- **Planning for 2025 Events**: A member is seeking recommendations for **hackathons** and **conferences** for the year **2025**, already planning to attend **ICML**, **NeurIPs**, and **CVPR**.
   - They are excited about the prospect of meeting more people in the community and welcome any additional suggestions.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1322919571455348837)** (1 messages): 

> `Leaderboard Techniques, API Endpoint Exceptions, Zero-shot Evaluation` 


- **Leaderboard Restrictions Clarified**: Techniques for model evaluation on the leaderboard are typically not allowed, as all models are assessed in a **zero-shot setting**.
   - An exception is made if the model operates via an [API endpoint](https://link.to/api), ensuring the user makes a single call and receives a single response.
- **API Call Mechanism for Validity**: Models leveraging complex internal techniques must ensure that users only perform one **API call**, which delivers a single response to remain eligible for leaderboard consideration.
   - This structure aligns with **OpenAI’s o1 model**, which successfully uses chain-of-thought reasoning behind its API.


  

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

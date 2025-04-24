---
id: e94eef2f-48e7-4416-b4e9-429de3b66f45
title: nothing much happened today
date: '2024-09-18T00:27:31.736910Z'
original_slug: ainews-nothing-much-happened-today-7147
description: >-
  AI news discusses open-source model replication, advancements in transformer
  reasoning, model performance improvements, multimodal capabilities, AI safety,
  and industry trends.
companies:
  - openai
  - lmsys
  - scale
  - cais
  - langchain
  - qdrant
models:
  - o1
  - chatgpt-4o
  - llama-3.1-405b
  - gpt-4
  - gpt-4o
topics:
  - transformers
  - intermediate-reasoning
  - model-comparison
  - model-merging
  - ai-safety
  - multimodal
  - ai-tools
  - code-review
  - visual-search
  - ai-integration
---


<!-- buttondown-editor-mode: plaintext -->**Peace and quiet is all you need.**

> AI News for 9/16/2024-9/17/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**221** channels, and **2197** messages) for you. Estimated reading time saved (at 200wpm): **225 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Given the [extreme restrictions, cost, and lack of transparency around o1](https://news.ycombinator.com/item?id=41534474), everyone has puts and takes on whether or not o1 can be replicated in open source/in the wild. As [discussed in /r/localLlama](https://www.reddit.com//r/LocalLLaMA/comments/1fiadsy/will_an_open_source_model_beat_o1_by_the_end_of/), Manifold markets currently has 63% odds on an open source version:

![image.png](https://assets.buttondown.email/images/666119c8-25fd-4bf0-8292-a74b238961ab.png?w=960&fit=max)

It is simultaneously likely that:

- there are many things about o1 that could be replicated in open source, especially with an OpenAssistant-level crowdsourced reasoning trace dataset
- MAYBE some of the MCST papers that people have been throwing around are relevant, but also MAYBE NOT
- there are real [RL on CoT](https://x.com/wgussml/status/1834691198013129053) advances done at the training level that no amount of dataset futzing will match up to.

For the last reason alone, the standard time-to-OSS-equivalent curves in model development may not apply in this instance.


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

**AI Model Updates and Advancements**

- **OpenAI's o1 Model**: [@denny_zhou](https://twitter.com/denny_zhou/status/1835761801453306089) highlighted that transformers can theoretically solve any problem with sufficient intermediate reasoning tokens, even with constant depth. This suggests significant potential for scaling LLM inference performance.

- **Performance Improvements**: [@lmsysorg](https://twitter.com/lmsysorg/status/1835825082280902829) reported substantial improvements in ChatGPT-4o (20240903) across various benchmarks, including overall performance, style control, hard prompts, and multi-turn interactions.

- **Model Comparisons**: [@lmsysorg](https://twitter.com/lmsysorg/status/1835760196758728898) compared bf16 and fp8 versions of Llama-3.1-405b, finding similar performance across categories, with fp8 closely matching bf16 while significantly reducing costs.

- **Emerging Capabilities**: [@svpino](https://twitter.com/svpino/status/1835740534729830800) discussed the specialization of GPT-4o in System 1 thinking and OpenAI o1 in System 2 thinking, anticipating future models that incorporate both under a single framework.

**AI Development and Research**

- **Evaluation Challenges**: [@alexandr_wang](https://twitter.com/alexandr_wang/status/1835738937719140440) announced a partnership between Scale and CAIS to launch "Humanity's Last Exam," a challenging open-source benchmark for LLMs with $500K in prizes for the best questions.

- **Model Merging**: [@cwolferesearch](https://twitter.com/cwolferesearch/status/1835748003128193470) explained the effectiveness of model merging, attributing its success to linear mode connectivity and sparsity in neural networks.

- **AI Safety**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1835739851452510258) shared insights on embedding-based toxic prompt detection, achieving high accuracy with minimal computational overhead.

- **Multimodal Capabilities**: [@_akhaliq](https://twitter.com/_akhaliq/status/1835677372344873377) introduced InstantDrag, an optimization-free pipeline for drag-based image editing, enhancing interactivity and speed in image manipulation tasks.

**AI Tools and Applications**

- **LangChain Updates**: [@LangChainAI](https://twitter.com/LangChainAI/status/1835720923414184128) announced the release of LangChain v0.3 for Python and JavaScript, focusing on improved dependencies and moving to peer dependencies.

- **AI in Code Review**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1835664732390351085) discussed the use of CodeRabbit for automated code reviews, highlighting its ability to adapt to team coding practices and provide tailored feedback.

- **AI in Product Search**: [@qdrant_engine](https://twitter.com/qdrant_engine/status/1835634931977892120) shared advancements in visual search solutions, integrating images, text, and other data into unified vector representations for improved product search experiences.

**Industry Trends and Observations**

- **AI Integration**: [@kylebrussell](https://twitter.com/kylebrussell/status/1835706377785798694) predicted that by 2030, AI will be the default, software will generate itself, and agents will be the new apps.

- **Open Source Developments**: [@far__el](https://twitter.com/far__el/status/1835791026034036845) hinted at upcoming developments in open-source AI models, suggesting potential competition with proprietary models.

- **AI in Fashion**: [@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1835745610919706685) demonstrated AI-generated fashion models, suggesting potential shifts in brand marketing strategies.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Advancements in Model Compression and Quantization**

- **[LMSYS finds minimal differences between bf16 and fp8 Llama-3.1-405b in Chatbot Arena](https://x.com/lmsysorg/status/1835760196758728898)** ([Score: 109, Comments: 34](https://reddit.com//r/LocalLLaMA/comments/1fil2an/lmsys_finds_minimal_differences_between_bf16_and/)): LMSYS conducted a comparison between **bf16** and **fp8** versions of **Llama-3.1-405b** in their **Chatbot Arena**, finding minimal differences in performance. The **fp8** model showed only a **0.3%** decrease in win rate compared to the **bf16** version, suggesting that **fp8 quantization** can significantly reduce model size and memory requirements with negligible impact on quality.
  - Users reported **significant differences in coding performance** between quantized versions, with some noting **fp8** as worse than **q8** for coding tasks. A tweet by **Aidan McLau** criticized the **LMSYS** evaluation, suggesting **bf16** is superior for specific prompts.
  - Discussions highlighted the **limitations of human perception-based evaluations** like the **LMSYS leaderboard**. Some users observed minimal differences between **q8** and **fp16** for coding, while others reported conflicting results in benchmarks.
  - Several comments praised **quantization techniques**, with one user successfully using an **IQ2_M version** of **Llama 3.1 70b** for coding tasks. The debate extended to comparisons between various quantization levels (**q6_k**, **q4km**) and their impacts on model performance.


- **Release of Llama3.1-70B weights with AQLM-PV compression.** ([Score: 249, Comments: 81](https://reddit.com//r/LocalLLaMA/comments/1fiscnl/release_of_llama3170b_weights_with_aqlmpv/)): The **Llama3.1-70B** and **Llama3.1-70B-Instruct** models have been compressed using **AQLM+PV-tuning**, reducing their size to **22GB** and enabling them to run on a single **3090 GPU**. This compression resulted in a **4-5 percentage point drop** in **MMLU performance**, with the base model's score decreasing from **0.78 to 0.73** and the instruct model's score from **0.82 to 0.78**. Additionally, a compressed **Llama3.1-8B** model has been released, which can run as an Android app using only **2.5GB of RAM**.
  - The compressed **Llama3.1-70B** models are similar to **IQ_2M** quantization, with comparable **22GB** size and **MMLU scores**. Users discussed running methods, including **Transformers**, **vLLM**, and **Aphrodite**, with some experiencing implementation challenges.
  - There's interest in compressing larger models like the **405B version** and **Gemma-2 27B**. Users speculated on potential sizes and compatibility with specific hardware, such as **M3 Max** with **128GB** RAM.
  - The **AQLM** quantization method is available as an [open-source project](https://github.com/Vahe1994/AQLM), but doesn't currently support **GGUF** format. Users reported slow inference speeds, with a **3090 GPU** achieving around **7 tokens/second**.

- **[Hugging Face optimised Segment Anything 2 (SAM 2) to run on-device (Mac/ iPhone) with sub-second inference!](https://v.redd.it/4ndo0w4sf7pd1)** ([Score: 83, Comments: 14](https://reddit.com//r/LocalLLaMA/comments/1fiab07/hugging_face_optimised_segment_anything_2_sam_2/)): **Hugging Face** has optimized **Segment Anything 2 (SAM 2)** for on-device inference, enabling it to run on **Mac and iPhone** with **sub-second** performance. This optimization allows for real-time segmentation tasks on mobile devices, potentially opening up new applications in augmented reality, image editing, and computer vision on edge devices.
  - **Hugging Face** is releasing **Apache-licensed optimized model checkpoints** for SAM 2 in various sizes, along with an [open-source application](https://github.com/huggingface/sam2-studio) for sub-second image annotation. They're also providing conversion guides for SAM2 fine-tunes like **Medical SAM**.
  - The developer is planning to add **video support** and is open to suggestions for future features. This indicates ongoing development and potential for expanded capabilities in the SAM 2 optimization project.
  - Users expressed interest in **Apple** optimizing other models, specifically mentioning **GroundingDino**. This suggests a demand for more on-device AI models optimized for Apple hardware.


**Theme 2. Open-Source LLMs Closing the Gap with Proprietary Models**

- **Will an open source model beat o1 by the end of Q1 2025?** ([Score: 111, Comments: 52](https://reddit.com//r/LocalLLaMA/comments/1fiadsy/will_an_open_source_model_beat_o1_by_the_end_of/)): The post speculates on whether **open-source language models** could surpass **OpenAI's GPT-4** (referred to as "o1") by **Q1 2025** using **"System 2" style** approaches like **Monte Carlo Tree Search (MCTS)** and **reflection**. The author references **Noam Brown's** work and has created a [Manifold market](https://manifold.markets/JohnL/by-the-end-of-q1-2025-will-an-open?r=Sm9obkw) to gauge opinions on this possibility.
  - **Open-source models** could potentially match **GPT-4's** performance by **Q1 2025**, with users citing **Claude 3.5's** significant improvement and the potential for **reflection** and **thinking magic** to enhance OS models further.
  - Speculation on **GPT-4's** architecture suggests it may be an **engineering achievement** rather than a new model, possibly using **fine-tuned existing models**, clever prompting, and a **"critic" LLM** to qualify responses.
  - Opinions vary on the timeline, with some believing **open-source models** could surpass **GPT-4** by late 2025, while others note that **OpenAI** is likely to improve their model further, maintaining their lead over open-source alternatives.


- **Release of Llama3.1-70B weights with AQLM-PV compression.** ([Score: 249, Comments: 81](https://reddit.com//r/LocalLLaMA/comments/1fiscnl/release_of_llama3170b_weights_with_aqlmpv/)): **Llama3.1-70B** and **Llama3.1-70B-Instruct** models have been compressed using **AQLM+PV-tuning**, reducing their size to **22GB** and enabling them to run on a single **3090 GPU**. The compression resulted in a **4-5 percentage point drop** in **MMLU performance**, with the base model's score decreasing from **0.78 to 0.73** and the instruct model's score from **0.82 to 0.78**. Additionally, a compressed **Llama3.1-8B** model has been released, which has been [run as an Android app](https://blacksamorez.substack.com/p/aqlm-executorch-android?r=49hqp1&utm_campaign=post&utm_medium=web&triedRedirect=true) using only **2.5GB of RAM**.
  - Users compared **AQLM+PV-tuning** to **IQ_2M** quantization, noting similar **22GB** size and **MMLU scores**. The **chat template** for the model was fixed to improve compatibility with **vLLM** and **Aphrodite**.
  - Running the model on **16GB VRAM** systems proved challenging due to size constraints. The **70B model** requires at least **17.5GB** for weights alone, plus additional memory for caches and embeddings.
  - Users expressed interest in applying AQLM compression to other models like **Gemma-2 27B** and **Mixtral**. The [AQLM GitHub repository](https://github.com/Vahe1994/AQLM) was shared for those interested in quantizing their own models.
- **There seems to be promise in creating an open-source o1 model soon!** ([Score: 173, Comments: 55](https://reddit.com//r/LocalLLaMA/comments/1fim224/there_seems_to_be_promise_in_creating_an/)): The author reports **promising results** in creating an **open-source o1-like model** using a **Q4_K_M 8B model** fine-tuned on a **small dataset of 370 rows**. They provide links to the [model](https://huggingface.co/Lyte/Llama-3.1-8B-Instruct-Reasoner-1o1_v0.3), a [demo](https://huggingface.co/spaces/Lyte/Llama-3.1-8B-Instruct-Reasoner-1o1_v0.3-Q4_K_M), and the [dataset](https://huggingface.co/datasets/Lyte/Reasoner-1o1-v0.3-HQ) used for fine-tuning, emphasizing the potential for **GPU-limited users** to soon have access to similar models.
  - Users compared the project to **Matt's o1 experiment**, noting that this attempt actually produced results. The author clarified they're not claiming a **SOTA model**, just sharing an interesting experiment.
  - Discussion focused on the need for **reinforcement learning** implementation to fully replicate o1's approach. Some speculated o1 uses RL to find optimal phrasing and syntactic structures for chain-of-thought processes.
  - Several comments suggested running **popular benchmarks** to prove credibility and comparing results. The author submitted the model to the **open llm leaderboard** for evaluation and acknowledged limitations due to the small dataset and GPU constraints.


**Theme 3. Developments in LLM Reasoning and Inference Techniques**



- **o1-preview: A model great at math and reasonong, average at coding, and worse at writing.** ([Score: 87, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1ficb0z/o1preview_a_model_great_at_math_and_reasonong/)): The **o1-preview model** demonstrates exceptional abilities in **complex reasoning**, **math**, and **science**, outperforming other models in single-shot responses to challenging prompts. However, it falls short in **creative writing** and is average in **coding**, with the author preferring **Sonnet 3.5** for coding tasks due to better **inference speed** and accuracy trade-offs. The model occasionally provides correct answers despite inconsistent reasoning steps, and while it represents a significant advancement, it's not yet at a **Ph.D. level** in reasoning or math.

- **Paper: Chain of Thought Empowers Transformers to Solve Inherently Serial Problems** ([Score: 136, Comments: 27](https://reddit.com//r/LocalLLaMA/comments/1fiftvc/paper_chain_of_thought_empowers_transformers_to/)): **Denny Zhou** from **Google DeepMind** claims that **Large Language Models (LLMs)** have no performance limit when scaling inference, as proven in their paper. The research demonstrates that **transformers** can solve any problem with **constant depth**, provided they can generate sufficient intermediate reasoning tokens, as detailed in the paper available at [arXiv](https://arxiv.org/abs/2402.12875).

- **The holy grail of LLM 'reasoning' tactics during inference** ([Score: 39, Comments: 4](https://reddit.com//r/LocalLLaMA/comments/1fimvl6/the_holy_grail_of_llm_reasoning_tactics_during/)): The post highlights a **GitHub repository** that compiles various **LLM 'reasoning' tactics** for use during inference, inspired by recent developments in **Reflection models** and their extensions. The repository, created by a third party and available at **[https://github.com/codelion/optillm](https://github.com/codelion/optillm)**, offers a **drop-in API** for testing different inference 'reasoning' or 'thinking' methods, which can be adapted to work with various local model providers.
  - Users expressed interest in the repository, with one noting that these **advancements surpass regular fine-tuning algorithms**. The repo's compatibility with **local servers** was discussed, with confirmation of successful integration with **oobaboogas textgen**.
  - The repository functions as a **transparent OpenAI API-compatible proxy**, allowing integration with various tools and frameworks. It can be used by setting the **base_url** in local servers to utilize the proxy.
  - Integration with **Patchwork** yielded **significant performance improvements** compared to the base model. Details on this integration are available in the [repository's README](https://github.com/codelion/optillm?tab=readme-ov-file#patchwork-with-optillm) and [wiki](https://github.com/codelion/optillm/wiki/Patchwork).


**Theme 4. Challenges in LLM Evaluation and Reliability**



- **As someone who is passionate about workflows in LLMs, I'm finding it hard to trust o1's outputs** ([Score: 35, Comments: 9](https://reddit.com//r/LocalLLaMA/comments/1fid8z5/as_someone_who_is_passionate_about_workflows_in/)): The post critiques **o1's outputs and workflow approach** for complex tasks, particularly in coding scenarios. The author, who is passionate about **LLM workflows**, observes that o1's outputs resemble a workflow structure rather than standard Chain of Thought, potentially leading to issues such as the LLM **talking itself into a corner** on simple questions or **mangling Python methods** by losing functionality through multiple processing steps. The post argues for the importance of **tailored workflows** for different types of tasks (e.g., reasoning vs. coding), suggesting that o1's current approach of using a single workflow for all tasks may be problematic, especially for complex development work, leading the author to still prefer **ChatGPT 4o** for coding tasks.

- **New Model Identifies and Removes Slop from Datasets** ([Score: 68, Comments: 18](https://reddit.com//r/LocalLLaMA/comments/1fidhib/new_model_identifies_and_removes_slop_from/)): The **Exllama community** has developed a model to identify and remove **'slop'** and **moralization** from public datasets, including those on **HuggingFace**. This breakthrough allows for the detection of **corporate slop**, categorization of slop types, and analysis of low-quality data trajectories, potentially improving **LLM conversational abilities** and understanding of prompt rejection patterns. More information about the project is available on the [Exllama Discord](https://discord.gg/m5yEPEwK) server, where interested parties can speak with **Kal'tsit**, the model's creator.

- **PhD-level model GPT-o1 fails on middle school math ‘trap’ problems, with an accuracy rate of only 24.3%** ([Score: 270, Comments: 78](https://reddit.com//r/LocalLLaMA/comments/1fipkus/phdlevel_model_gpto1_fails_on_middle_school_math/)): The **GPT-o1 model**, despite claims of PhD-level intelligence, achieved only a **24.3% accuracy rate** on the **MathTrap_Public** dataset, which contains middle school math problems with added "traps". The researchers created the **MathTrap dataset** by modifying questions from **GSM8K and MATH datasets**, introducing contradictions or unsolvable elements that require understanding both the original problem and the trap to identify. **Open-source models** performed even worse on **MathTrap_Private**, with **Reflection-70B** achieving **16.0% accuracy**, **Llama-3.1-8B** at **13.5%**, and **Llama-3.1-70B** at **19.4%**.
  - **PhD-level mathematicians** and other users noted they would make the same mistake as the AI, with one stating the problem is "**fundamentally uninteresting**". Many argued the **discontinuity at x=0** is not essential and the limit approach is valid.
  - Users questioned the research methodology, with one pointing out the **preprint was last revised on July 11th** and doesn't mention **o1**. They tested the trap problems and found **o1 correctly identified all traps** on the first try, suggesting potential misinformation.
  - Several commenters criticized the **prompt design**, arguing that a better-formulated question would have yielded more accurate results. One suggested asking, *"Is the function periodic? Calculate the period if yes, otherwise prove that none exists. Justify your argument."*

## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Model Advancements and Benchmarks**

- **OpenAI's new GPT-4o1 model achieved an IQ score of 120**, beating 90% of people on standard IQ tests. However, on completely new questions it hadn't seen before, it scored closer to the human average of 100 IQ. [This still represents significant progress in AI reasoning capabilities](https://www.reddit.com/r/OpenAI/comments/1fipe3b/openais_new_gpt_model_reaches_iq_120_beating_90/).

- OpenAI increased rate limits for their o1-mini model by 7x, going from [50 messages per week to 50 messages per day](https://www.reddit.com/r/singularity/comments/1fim3hh/openai_weve_increased_rate_limits_for_o1mini_by_7x/). The o1-preview model also saw an increase from 30 to 50 messages per week.

- The o1 model showed major improvements over o1-preview in coding benchmarks, [jumping from 62% correct to 89% correct](https://www.reddit.com/r/singularity/comments/1fi2hmb/are_big_jumps_in_reasoning_for_models_of_the_same/). This represents a 3.5x increase in reliability for complex code generation.

- Some users reported that o1-mini has replaced GPT-4 for coding tasks, as it provides [full, uncapped responses without needing to click "continue"](https://www.reddit.com/r/singularity/comments/1fi2hmb/are_big_jumps_in_reasoning_for_models_of_the_same/).

**AI Ethics and Societal Impact**

- Billionaire Larry Ellison suggested that [AI-powered surveillance systems could ensure "citizens will be on their best behavior"](https://www.reddit.com/r/singularity/comments/1fi0iuq/billionaire_larry_ellison_says_a_vast_aifueled/), sparking debate about privacy concerns and potential abuse of AI technologies.

- There are ongoing discussions about whether to celebrate or worry about rapid AI progress. Some view it as an exciting technological advancement, while others express concerns about job displacement and societal impacts.

**AI Development and Research**

- The o1 model appears to use a breakthrough involving [reinforcement training with built-in chain of thought processes](https://www.reddit.com/r/singularity/comments/1fi2hmb/are_big_jumps_in_reasoning_for_models_of_the_same/), potentially allowing for significant scaling of capabilities.

- Some researchers suggest o1 could be considered a "proto-AGI" architecture, though additional breakthroughs in areas like short-term and long-term memory may still be needed to achieve general intelligence.

**AI Tools and Applications**

- New AI image generation tools like FLUX are producing impressive results, with examples shown of [Half-Life inspired Soviet-era scenes](https://www.reddit.com/r/StableDiffusion/comments/1fi1e04/flux_halflife_but_soviet_era/) and [abstract surrealist landscapes](https://www.reddit.com/r/StableDiffusion/comments/1fi8rmg/mirrorscapes_flux/).

- The Quest 3 VR headset combined with AI video generation tools is enabling [new forms of immersive content creation](https://www.reddit.com/r/singularity/comments/1fihviw/quest_3_vr_headset_using_gravitysketch_runawaymls/).


---

# AI Discord Recap

> A summary of Summaries of Summaries

## O1-mini

**Theme 1. AI Models: New Releases and Rivalries**

- **Claude 3.5 Battles GPT-4o**: The community is torn between **Claude 3.5** and **GPT-4o**, with members conducting tests to determine which model excels in specific tasks. [Claude vs GPT-4o Showdown](https://discord.com/channels/1047197230748151888/1047649527299055688/1285314374591844523) highlights the ongoing rivalry.

- **Qwen 2.5 Unveils Stricter Variants**: **Qwen 2.5** introduces new model sizes ranging from **0.5B** to **72B** parameters, all featuring enhanced content filtering. Concerns about **knowledge retention** persist among users.

- **Mistral's Pixtral-12B Sets the Stage**: [Pixtral-12B](https://mistral.ai/news/pixtral-12b/) marks a significant leap in multimodal models, offering robust video and image generation capabilities that rival existing giants.

**Theme 2. Innovative Tools and Integrations**

- **Superflex Transforms Figma to Code**: **Superflex** now allows developers to generate front-end code directly from [Figma designs](https://www.producthunt.com/posts/superflex), streamlining the design-to-development workflow seamlessly.

- **OpenRouter Boosts Google Sheets with AI**: [GPT Unleashed for Sheets](https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147) integrates **OpenRouter** features like 'jobs' and 'contexts', enabling efficient **prompt engineering** within spreadsheets.

- **Aider Teams Up with Sonnet for Coding**: The integration of **Sonnet 3.5** with **O1 Mini** enhances **Aider's** reliability in handling coding tasks, with users praising its efficiency in managing quick fixes and assignments.

**Theme 3. Training, Optimization, and Technical Hurdles**

- **LM Studio Slashes Training Time**: Adjusting tokens and batch sizes in **LM Studio** reduced model training from **5 days** to just **1.3 hours**, showcasing significant **optimization** gains.

- **Tinygrad Faces AMD Compatibility Issues**: Users encounter **AttributeError** when updating **tinygrad** on AMD systems, sparking discussions on potential **kernel version** mismatches and troubleshooting strategies.

- **CUDA Mode Tackles In-Memory Computing**: SK Hynix introduces **AiMX-xPU** at [Hot Chips 2024](https://www.servethehome.com/sk-hynix-ai-specific-computing-memory-solution-aimx-xpu-at-hot-chips-2024/), enhancing **LLM inference** by performing computations directly in memory, thus boosting **power efficiency**.

**Theme 4. AI Safety and Ethical Concerns**

- **Cohere Rolls Out Customizable Safety Modes**: [Cohere's Safety Modes](https://discord.com/channels/954421988141711382/954421988783444043/1285331547032911883) in their Chat API allow users to tailor model outputs to meet specific **safety requirements**, aiming to mitigate **liability concerns**.

- **Unsloth AI's Censorship Sparks Debate**: The **Phi-3.5** model faces backlash for being overly censored, with users sharing [uncensored versions](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored) and debating the balance between **safety** and **usability**.

- **Jailbreaking Claude 3.5 Opens Pandora's Box**: A successful [jailbreak](https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d) for **Claude 3.5 Sonnet** ignites discussions on **model security** and the ethical implications of **bypassing safeguards**.

**Theme 5. Community Buzz and Funding Moves**

- **YOLO Vision 2024 Invites AI Engineers**: [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision) hosted by **Ultralytics** at Google Campus for Startups in Madrid invites AI engineers to register and participate, fostering **community interaction** through activities like voting for event music.

- **11x AI Secures $24M Series A Funding**: **11x AI** raises a substantial **$24M Series A** from **Benchmark**, boosting its **annual recurring revenue by 15x** and expanding its customer base to over **250 clients**.

- **Mistral's Strategic Moves Spark Debate**: An analysis of [Microsoft's strategy](https://mistral.ai/news/september-24-release/) in integrating AI technologies with Mistral's offerings prompts the community to reflect on the company's **competitive direction** and alignment with its **historical goals**.

---

## O1-preview

**Theme 1. New AI Models and Releases Ignite Tech Communities**

- [**Qwen 2.5 Drops with Fresh Sizes and Stricter Filters**](https://huggingface.co/Qwen?sort_models=modified#models): **Qwen 2.5** unveils models ranging from **0.5B to 72B** parameters, introducing tighter content filtering compared to its predecessor. Initial tests reveal limitations in topic knowledge, sparking concerns about impacts on **knowledge retention**.
- [**Mistral-Small-Instruct-2409 Makes a Grand Entrance**](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409): The **Mistral-Small-Instruct-2409** model, boasting **22B parameters**, supports function calls and sequences up to **128k tokens**. Despite its potential, it carries non-commercial usage restrictions and is best paired with [vLLM](https://github.com/vllm-project/vllm) for optimal performance.
- [**LlamaCloud Unveils Multimodal RAG Magic**](https://t.co/43eL8zvm7H): **LlamaCloud** launches **multimodal capabilities**, enabling swift creation of end-to-end **multimodal RAG pipelines** across unstructured data types. This leap enhances workflows for **marketing decks**, **legal contracts**, and **finance reports**.

**Theme 2. AI Tools Get Superpowers: Integrations Galore**

- [**Google Sheets Gets a Boost with OpenRouter Integration**](https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147): **OpenRouter** joins forces with the [GPT Unleashed for Sheets](https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147) add-on, offering free access to **100+ models**. Users can assign short codes to prompts, supercharging AI output management within spreadsheets.
- [**Aider Teams Up with Sonnet for Code Magic**](https://aider.chat/docs/): Developers cheer as **Aider** integrates **Sonnet 3.5** with **O1 mini**, enhancing coding tasks with reliable edits and fixes. Users laud Aider for its efficiency in handling swift code tweaks and assignments.
- [**Superflex Turns Figma Designs into Live Code**](https://www.producthunt.com/posts/superflex): **Superflex** transforms [Figma designs](https://www.producthunt.com/posts/superflex) directly into front-end code, seamlessly integrating into existing projects. This tool accelerates development, making designers' dreams a reality.

**Theme 3. Tech Gremlins and Solutions: Overcoming AI Hurdles**

- **LM Studio Users Wrestle with GPU Ghosting**: Despite proper settings, **LM Studio** stubbornly ignores GPUs, overloading CPUs and RAM instead. Blurry screens linked to anti-aliasing settings prompt users to tweak configurations for a smoother ride.
- **Unsloth Fine-Tune Frenzy Leads to Hallucinations**: Fine-tuning 'unsloth/llama-3-8b-bnb-4bit' causes models to hallucinate, hinting at potential data corruption during saving. The community debates the effects of using `save_method = 'merged_4bit_forced'`.
- **BitNet's Ternary Tricks Stir Up Debate**: Packing **5 ternary values into an 8-bit space** proves clever but complex. Discussions swirl around using **Lookup Tables** to enhance this method, pushing the envelope on neural network efficiency.

**Theme 4. AI Safety and Research Take Center Stage**

- **AI Safety Fellowship Fuels New Research Ventures**: A community member dives into **AI safety** after snagging an **Open Philanthropy fellowship**, keen on tackling **interpretability** and alignment research. They're on the hunt for collaboration over the next **six months**.
- [**Fourier Transforms Unveil Hidden State Secrets**](https://sander.ai/2024/09/02/spectral-autoregression.html): Delving into the [Fourier transforms](https://sander.ai/2024/09/02/spectral-autoregression.html) of hidden states reveals a shift from uniformity to a **power law** as layers deepen. Curiosity mounts about the attention mechanism's role in this spectral phenomenon.
- [**LlamaIndex Tackles Visual Data with Multimodal RAG**](https://t.co/GOedcAdLqF): **Product manuals** pose a challenge due to their visual nature. **LlamaIndex** introduces a sophisticated [indexing pipeline](https://t.co/GOedcAdLqF) to help LLMs effectively navigate and understand image-heavy documents.

**Theme 5. AI Ventures into Business and Creativity**

- [**Ultralytics Throws a Party at YOLO Vision 2024**](https://www.ultralytics.com/events/yolovision): **Ultralytics** invites AI enthusiasts to [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision) on **October 28** in Madrid. Attendees can groove to tunes they vote for during discussion panels, blending tech and fun.
- [**AdaletGPT Launches RAG Chatbot for Legal Aid**](https://adaletgpt.com): **AdaletGPT** unveils a **RAG chatbot** built with **OpenAI** and **LangChain**, offering AI-driven legal support at [adaletgpt.com](https://adaletgpt.com). Users can tap into advanced assistance with a friendly interface.
- **Open Interpreter Wows Users with Smarts**: **Open Interpreter** garners praise for its cleverness and capabilities. Excitement brews as users explore its potential, with beta tester slots in high demand.


---

# PART 1: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **O1 Mini capped at 10 uses daily**: Users expressed frustration over the recent limit of **10 uses per day** for the O1 Mini on Perplexity, feeling it restricts access compared to rivals.
   - There are **speculations** that this limit aims to manage server costs and marketing strategies, raising questions about user experience.
- **Claude 3.5 vs. GPT-4o Showdown**: Tension rises as community members weigh the pros and cons of choosing between **Claude 3.5 and GPT-4o**, with tests deemed essential for discerning differences.
   - Participants noted that **GPT-4o** may excel in specific tasks, hinting at its enhanced capabilities.
- **Perplexity AI's Reasoning Features Ignited Buzz**: The rollout of a **Reasoning focus** feature in Perplexity stirred discussion, as users experiment with enhanced functionalities within the **Pro Search** environment.
   - Feedback highlighted improved output quality and reasoning steps, showcasing a notable upgrade.
- **Minecraft Moderation Ban Issues Unpacked**: A community-led **Minecraft moderation ban discussion** was initiated on a dedicated page, calling for user opinions on existing policies.
   - Members are invited to share their thoughts, suggesting a collective effort to address potential moderation flaws.
- **Microsoft's Strategy Sparks Debate**: An analysis post raising questions about **Microsoft's tactics** has drawn attention, prompting users to scrutinize the company's competitive direction.
   - The discussion encourages reflection on whether Microsoft's recent actions align with its historical goals.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 Introduces New Model Variants**: Qwen 2.5 features new model sizes such as **0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B**, all with stricter content filtering compared to its predecessor.
   - The model variants reportedly limit knowledge on certain topics, raising concerns about potential impacts on **knowledge retention**.
- **Mistral-Small-Instruct-2409 Released**: The **Mistral-Small-Instruct-2409** model, with **22B parameters**, supports function calls and sequences up to **128k tokens**, though it has non-commercial usage restrictions.
   - Usage alongside [vLLM](https://github.com/vllm-project/vllm) is recommended for optimal inference pipeline performance.
- **Hallucinations in Fine-tuned Models**: After fine-tuning the model 'unsloth/llama-3-8b-bnb-4bit', users reported hallucinations in the downloaded version from Hugging Face, raising concerns about potential data corruption.
   - This triggered discussions around usage of `save_method = 'merged_4bit_forced'` and its effects on model performance.
- **Prioritizing Application Knowledge over Memorization**: It was emphasized that **application knowledge** trumps mere memorization of problems in platforms like LeetCode for effective coding in real-world scenarios.
   - A solid grasp of algorithms and data structures such as **linked lists** and **hashmaps** is crucial for practical application.
- **KTO reigns supreme in RLHF circles**: A preference for **KTO** over **ORPO** in reinforcement learning was noted for its simplicity as a *thumbs up, thumbs down* dataset.
   - While recognizing that **RLHF** methods can simplify models, the need to *test all available options* was highlighted.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O1 Models Lag in Real Applications**: Users expressed frustration with the **O1 models' performance**, noting that they excelled in playground scenarios but struggled in practical applications like **Aider** due to limitations in system prompts.
   - While O1 models showed promise, their effective deployment remains an issue, pushing developers to seek alternatives.
- **Sonnet Teaming Up with Aider**: Community discussions revealed users advocating for **Sonnet 3.5** integration with **O1 mini** to enhance coding tasks, citing superior reliability in edits and fixes.
   - Many praised Aider for efficiently handling quick coding fixes, illustrating the benefits of combining these tools.
- **Debate on RAG for Coding**: Discussions highlighted the effectiveness of **RAG** methods in coding versus fine-tuning on specific codebases, with many arguing for a tailored approach for better results.
   - Concerns arose about retrieval mechanisms failing in large codebases, underscoring a need for improved strategies.
- **Azure API Key Setup with Aider**: A user detailed the configuration steps required to integrate **Aider** with **Azure OpenAI**, emphasizing the importance of structured JSON requests for functionality.
   - Additional resources, such as the LiteLLM documentation, were recommended for handling Azure API keys effectively.
- **Superflex Transforms Figma to Code**: The launch of **Superflex** has been a game changer, allowing developers to generate front-end code directly from [Figma designs](https://www.producthunt.com/posts/superflex), streamlining their workflow.
   - This tool integrates designs into existing projects smoothly, making it a highly attractive option for modern web development.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPU Performance Troubleshooting**: Users expressed frustration with LM Studio not utilizing their GPU, despite proper settings in **Settings -> System Resources**. Issues causing blurry screens were linked to anti-aliasing settings, leading to suggestions for configuration adjustments.
   - The active conversation highlighted common troubleshooting steps that could enhance GPU utilization and reduce blurred visuals in user interfaces.
- **Training Time Drastically Reduced**: One user trained a 100k parameter model, seeing a shift from **5 days** to **1.3 hours** by adjusting tokens and batch size. Community members discussed bottlenecks in the data loader, emphasizing the importance of efficient configurations for training efficiency.
   - The conversation shed light on practical solutions for optimizing model training durations through parameter adjustments.
- **New Features Fuel Excitement in LM Studio**: The recent addition of document integration in LM Studio sparked positive feedback, demonstrating the community’s long-standing request for this feature. Users were eager to test the updated version and leverage improved functionality.
   - This feature underlined how simplicity in design appeals to users lacking extensive IT backgrounds, making advanced features more accessible.
- **Discussions on Dual GPU Setups**: Users explored the benefits of dual **4060 Ti** setups to maximize VRAM without excessive power consumption. This practical configuration sparked debates on the advantages of using identical GPUs to streamline setups and manage energy efficiency.
   - The discussions suggested a growing trend towards optimizing cost-effectiveness and performance in GPU setups.
- **VRAM Criticality for LLM Performance**: Concerns surfaced regarding the critical need for VRAM in handling powerful LLMs, with insights into various GPUs’ capabilities in token generation rates. Members shared personal experiences indicating that many powerful models exceed the VRAM limits of currently available cards.
   - The emphasis on VRAM sparked deeper conversations on how GPU advancements can better support LLM training and inference demands.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **API Documentation Gets a Boost**: The [Hugging Face Inference API docs](https://huggingface.co/docs/api-inference) received a critical update, now featuring clearer rate limits, enhanced code examples, and a dedicated PRO section.
   - This revamp aims to streamline the user experience as the dataset offerings continue to proliferate, making deployment more intuitive.
- **Countdown to 1 Million Models**: The community speculated on achieving **1 million models** soon, with stats showing **40K weekly models** up for grabs.
   - Excitement surged as participants compared the growth rates of different model repositories, with predictions pointing to an imminent milestone.
- **New Tools for Dataset Creation**: [DataCraft](https://x.com/dvilasuero/status/1835711765570630017) was introduced as a no-code tool for generating synthetic datasets using natural language, aimed at simplifying data creation challenges.
   - This tool incorporates best practices, enhancing accessibility for users looking to build effective AI datasets.
- **Engaging in Gradio Office Hours**: Members were invited to join ongoing [Gradio office hours](https://t.co/Dxeb0jaQ6e), an open forum for discussing features, enhancements, and community feedback.
   - This session serves as a fertile ground for sharing insights and troubleshooting Gradio-related issues directly with experts.
- **Challenges with LLaMA3 Setup**: A user sought help downloading the **LLaMA3** model, expressing their struggles with the current **PyTorch** setup and requesting guidance.
   - Confusion ensued over implementation choices, revealing a shared need for clarity on the effectiveness of heterogeneous tools in model operations.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o stuns in GeoGuessr**: Members expressed surprise at how well **GPT-4o** performs in **GeoGuessr**, although it still trails behind expert players. Notably, it deviates from the expected speed of the **o1-mini** model.
   - This performance sparks curiosity regarding potential improvements and applications beyond gaming.
- **Fine-tuning job hits a hard limit**: A user vented frustrations over their fine-tuning job exceeding a hard limit, incurring a cost of **$24.31** against a remaining quota of **$19.91**. Speculation arose that it could be tied to discounts.
   - The discussion centered on strategies for managing costs in fine-tuning operations.
- **Advanced Voice Mode availability awaits**: Multiple members reported using Plus but lacking access to **Advanced Voice Mode**, with expectations set for availability by **end of Fall**. This raises questions about rollout timing.
   - The anticipation reflects a keen interest in advancements in voice capabilities.
- **Exploring auto prompts for Ideogram/Midjourney**: A member circulated an **auto prompt for Ideogram/Midjourney**, encouraging feedback and rating on usability, emphasizing that it's free to share.
   - The initiation of this resource exchange showcases community collaboration.
- **Discussion on Official Libraries**: The mention of **official libraries** stirred interest, though no in-depth conversation followed. This opens the door for future discussions on potential resources.
   - The ambiguity leaves room for clarification as users seek more details.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter integrates with Google Sheets**: OpenRouter has been incorporated into the [GPT Unleashed for Sheets](https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147) addon, making it available for free following user requests.
   - *I personally love using OR too* and anticipate beneficial feedback as more users adopt this integration.
- **Exciting features boost Google Sheets performance**: The addition of features like 'jobs', 'contexts', and 'model presets' in the Google Sheets addon streamlines prompt engineering.
   - These enhancements allow users to assign short codes to prompts, optimizing AI output management.
- **OpenRouter suffers API outages**: Various users have reported intermittent issues accessing OpenRouter, particularly with the `o1` models, causing confusion over rate limits.
   - One user noted a temporary outage in Switzerland but confirmed that functionality was restored shortly after.
- **Gemini struggles with image generation consistency**: There have been mixed discussions regarding Gemini's image generation capabilities, with discrepancies noted between its official site and OpenRouter performance.
   - It was clarified that Gemini's chatbot uses Imagen models for image generation, while OpenRouter uses Google Vertex AI.
- **Mistral API sees significant price drops**: New announcements reveal substantial price reductions for Mistral APIs, dropping to **$2** for Large 2 models, making it a competitive option.
   - This shift is impacting user decisions regarding which models to utilize for their API calls.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Explore Metal Puzzles and Collaboration**: The [Metal Puzzles GitHub repository](https://github.com/abeleinin/Metal-Puzzles) promotes learning Metal programming through collaborative puzzle solving, encouraging community engagement.
   - A live puzzle-solving session was proposed, with enthusiasm from members pointing to growing interest among newcomers.
- **Triton LayerNorm hits inconsistency wall**: A member reported that using **tensor parallelism > 1** with Triton LayerNorm results in **non-deterministic gradient accumulation**, impacting their MoE training.
   - They are reaching out to the Liger team for potential insights and alternative implementation suggestions.
- **FP8 achieves restored end-to-end functionality**: The recent implementation updates have successfully restored **FP8** end-to-end capabilities for both forward and backward passes, advancing the functionality in AI workflows.
   - Future tasks will include multi-GPU support and performance testing to ensure convergence with existing techniques.
- **SK Hynix drives in-memory computing innovation**: At **Hot Chips 2024**, SK Hynix showcased its in-memory computing technologies, **AiMX-xPU** and **LPDDR-AiM**, tailored for efficient LLM inference.
   - This method significantly reduces power consumption and latency by performing computations directly in memory.
- **BitNet's ternary packing quirk**: Discussion revealed that packing **5 ternary values into an 8-bit space** is superior to traditional methods, enhancing efficiency despite implementation complexity.
   - Members considered Lookup Tables as a possible enhancement for packing methods, promoting further exploration.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCon Details Confirmed**: Location details for **NousCon** were confirmed to be sent out that evening, sparking discussions about potential future event venues including **NYC**.
   - A user raised inquiries about the broader implications for community engagement at future events.
- **Interest in Hermes 3 Unleashed**: A new member expressed a desire to utilize **AI Model Hermes 3** for business inquiries and sought contact information.
   - Another user recommended reaching out to a specific member for advice.
- **InstantDrag Takes Center Stage**: **InstantDrag** was highlighted as a modern solution for drag-based image editing, noted for improving speed without needing masks or text prompts.
   - Comparisons were made to **DragGAN**, showcasing potential for faster workflows.
- **LLM Inference Performance Limit Explored**: A tweet from Denny Zhou pointed out that transformers can theoretically solve any problem if given sufficient intermediate reasoning tokens.
   - This was linked to a paper accepted at **ICLR 2024**, emphasizing the significance of **constant depth** in transformer capabilities.
- **Jailbreak for Claude 3.5 Unveiled**: A member successfully created a jailbreak for **Claude 3.5 Sonnet**, reported as a particularly challenging model to breach.
   - While inspired by previous works, they emphasized their unique approach and functionality.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Luma Labs Launches Dream Machine API**: Luma Labs announced the release of the [Dream Machine API](https://lumalabs.ai/dream-machine/api), enabling developers to leverage a leading video generation model with minimal tooling.
   - This initiative aims to make video creation accessible, allowing users to dive straight into creative development.
- **11x AI Raises $24m Series A Funding**: 11x AI has successfully secured a **$24m Series A** funding round from **Benchmark**, increasing its annual recurring revenue by **15x** this year and serving over **250 customers**.
   - The team plans to build **LLM-powered systems** aimed at transforming digital go-to-market strategies.
- **AI's Job Market Disruption**: A report predicts that **60 million jobs** across the US and Mexico will be impacted by AI within the next year, with future projections potentially escalating to **70 million** in the US and **26 million** in Mexico over a decade.
   - While some job transformations might not lead to losses, a significant number of positions remain at considerable risk, underscoring the need for workforce adaptation.
- **Claude 3.5 System Prompt Circulates**: The **Claude 3.5 Projects + Artifacts system prompt** was shared via a [gist](https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d), gaining traction among users interested in exploring AI applications.
   - This prompt's relevance is highlighted by its discussion across multiple platforms, indicating its significance in current AI evaluations.
- **Yann LeCun Showcases ZIG Based Inference Stack**: Yann LeCun introduced a new **ZIG based inference stack** aimed at optimizing high-performance AI inference capable of running deep learning systems efficiently on various hardware.
   - This open-sourced project marks its exit from stealth mode, demonstrating notable advancements in AI performance.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Foundation Models Forge Ahead in Biotech**: A member presented their work on **foundation models** in **biotech**, focusing on **large scale representation learning** for both sequence and tabular data, underscoring the growing intersection of AI and biotechnological applications.
   - This highlights the rising interest in leveraging **AI technologies** to revolutionize traditional biotech processes.
- **AI Safety Fellowship Sparks Interest**: Excitement brewed as a member shared their transition to **AI safety** after receiving an **Open Philanthropy career transition fellowship**, indicating an eagerness to engage in **interpretability** and alignment research.
   - They invited others to share their research projects for potential collaboration over the next **six months**.
- **Troubleshooting TensorRT-LLM Build Problems**: Concerns surfaced regarding issues with building a **TensorRT-LLM** on a **T4 video card**, specifically citing an error linked to workspace size and asking for troubleshooting tips.
   - One suggestion to resolve the issue was to increase workspace size using `IBuilderConfig::setMemoryPoolLimit()`.
- **Interpreting Hidden States through Fourier Transforms**: Discussions kickstarted with a focus on the [Fourier transforms](https://sander.ai/2024/09/02/spectral-autoregression.html) of hidden states, revealing a trend from uniformity to a **power law** as layer depth increased.
   - Questions rose about whether the attention mechanism plays a role in shaping this **power spectrum** in final hidden states.
- **Pythia Checkpoints Gain Traction**: Community members highlighted the **Pythia suite** as a robust resource for probing scale and architectural effects on model behavior, encouraging broader exploration.
   - Interest was expressed in analyzing different architectures through the [Pythia repository](https://github.com/EleutherAI/pythia) to confirm observations related to model training effects.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SSH Fails to Connect After Key Update**: A member faced **SSH connection issues** with their deployed pods post SSH key update, questioning if any configuration tweaks could solve it.
   - *I can't get in!* prompted discussions on possible fixes and alternatives using detailed config checks.
- **Stable Diffusion Model Won't Load**: Installation woes hit another user as they faced a 'model failed to load' error even after following the [setup guide](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides).
   - Suggestions flowed in to seek help by sharing specific error logs for targeted troubleshooting.
- **ComfyUI Faces White Screen Dilemma**: Post-update, a user reported a **white screen** issue with ComfyUI, halting their GUI attempts.
   - A fix was proposed: completely unload ComfyUI and restart using the update script.
- **Control Net Needs Robust Dataset**: Members debated the **dataset requirements** for training Effective Control Net, emphasizing needing quality data.
   - Suggestions included exploring **novel dataset augmentations** to enhance training outcomes.
- **CivitAI Bounty Pack Seeks Input**: A member inquired about posting a **CivitAI bounty** for a character pack of 49 items with around 4000 images, looking for proper Buzz compensation.
   - *What’s a reasonable offer?* prompted discussions on bounty pricing strategies.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud Launches Multimodal RAG Capabilities**: The recent launch of **multimodal capabilities in LlamaCloud** enables users to quickly create **end-to-end multimodal RAG pipelines** across unstructured data formats, enhancing their workflow significantly ([details here](https://t.co/43eL8zvm7H)).
   - This toolkit supports various applications, including **marketing slide decks**, **legal contracts**, and **finance reports**, thereby simplifying complex data processing.
- **LlamaIndex Integrates Seamlessly with Neo4j**: Community members explored how to retrieve embeddings stored as node properties in **Neo4j** using **LlamaIndex**, suggesting a connection via property graph indexing for effective querying.
   - It was discussed that once nodes are retrieved, parsing their properties for embeddings should be a straightforward task, linking to [Neo4j Graph Store - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/Neo4jKGIndexDemo/).
- **Addressing Circular Dependency in LlamaIndex Packages**: A circular dependency issue was detected between `llama-index-agent-openai` and `llama-index-llms-openai`, leading members to brainstorm potential solutions including creating an **openai-utils** package.
   - Questions regarding timelines for these fixes surged, creating a need for community contributions to address the dependency promptly.
- **Navigating Image Coordinates with GPT-4o**: A user highlighted challenges with **image coordinate extraction** using **GPT-4o**, specifically aligning labels and getting accurate coordinates due to their grid overlay method.
   - Feedback from the community was encouraged to improve precision in detecting entities for cropping images, underlining the technical difficulties involving spatial recognition.
- **Multimodal RAG and Product Manual Challenges**: **Product manuals** have proven difficult for RAG techniques since they are primarily visual, necessitating a sophisticated [indexing pipeline](https://t.co/GOedcAdLqF) for LLMs to navigate them effectively.
   - The discussion emphasized the need for methods to handle step-by-step visuals and diagrams typical in product manuals.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Mistral Launches New Features**: Mistral has introduced several features including a [free tier on La Plateforme](https://mistral.ai/news/september-24-release/) aimed at developers for API experimentation.
   - These updates also feature **reduced prices** and enhancements to **Mistral Small**, making it more appealing for users.
- **Transformers Benefit from Intermediate Generation**: Research shows that incorporating a 'chain of thought' in transformers can significantly enhance their computational capabilities.
   - This approach is expected to improve performance on reasoning tasks where standard transformers struggle.
- **Unleashing Secrets of Gemini Models**: Exciting insights into unreleased **Gemini models** like **potter-v1** and **dumbledore-v1** have emerged, hinting at a strong lineup including **gemini-test** and **qwen2.5-72b-instruct**.
   - The community is buzzing about these new models, marking a pivotal moment in model development.
- **Celebrating Newsletter Readers Together**: A member shared an invitation for 'the great newsletter reader party,' creating opportunities for community engagement through shared readings.
   - This initiative aims to build connections and foster a love for curated content among participants.
- **Critique on Mainstream Media Reliance**: A discussion highlighted the drawbacks of depending solely on mainstream media for news consumption.
   - Members expressed a desire for more diverse and alternative sources to explore.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Navigating Chat History Management in LangChain**: Members discussed the complexities surrounding **Chat Message History Management** in **LangChain**, particularly regarding the storage of UI messages in **PostgresChatMessageHistory**.
   - It was agreed that UI-specific messages must reside in a separate table as existing systems lack combined transaction support.
- **Setting Goals for Open Source Contributions**: A member expressed ambition to significantly contribute to open-source projects while seeking sponsorship for independence.
   - They requested community insights on pathways to achieve these impactful contributions.
- **Migrating to Modern LLMChain Implementations**: Feedback suggested migrating from legacy **LLMChain** to newer models for better parameter clarity and streaming capabilities.
   - Newer implementations allow easier access to raw message outputs, stressing the importance of keeping updated.
- **AdaletGPT Debuts RAG Chatbot**: A backend developer at **adaletgpt.com** launched a **RAG chatbot** utilizing **OpenAI** and **LangChain**, inviting users to try it out at [adaletgpt.com](https://adaletgpt.com).
   - They encouraged community inquiries stating they would provide support with a *I will do my best for you* assurance.
- **AI Solutions for Local Business Integration**: A member expressed readiness to market AI solutions to local businesses, inquiring about effective implementation strategies.
   - They specifically sought tips on engaging business owners who might lack AI familiarity.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad bumps into AMD issues**: A user faced an **AttributeError** while attempting to bump **tinygrad** from **0.9.0 to 0.9.2** on AMD indicating a possible kernel version problem with **struct_kfd_ioctl_criu_args**.
   - Investigations reference the **tinygrad/extra/hip_gpu_driver/test_kfd_2.py** file and related **[pull request #5917](https://github.com/tinygrad/tinygrad/pull/5917)** addressing the issue.
- **Monitoring VRAM allocation spikes**: A user sought advice on identifying the causes of **spikes in VRAM allocation**, prompting discussions around effective memory usage monitoring tools.
   - Community members emphasized the significance of understanding these spikes to optimize Tinygrad's performance.
- **Investigating Tinygrad Tensor errors**: Another member reported encountering errors during Tensor manipulation in **Tinygrad**, linking to an **[open issue](https://github.com/tinygrad/tinygrad/issues/6352)** for more details.
   - This highlighted ongoing challenges in debugging Tinygrad and the need for community collaboration.
- **Forking Diffusers integrates Tinygrad**: Discussion arose around a **Diffusers fork** that utilizes Tinygrad, steering away from Torch and aiming for a fresh approach without direct replication.
   - Community members expressed enthusiasm over this initiative as a potential enhancement for Tinygrad's ecosystem.
- **NotebookLM creates engaging Tinygrad podcast**: The **NotebookLM** team released an **8-minute podcast** weaving engaging analogies to clarify Tinygrad concepts, effectively pitching **tinybox**.
   - This approach showcases innovative methods to educate others about Tinygrad's principles and applications.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere introduces beta Safety Modes**: Cohere announced the beta launch of **Safety Modes** in their Chat API, enabling users to customize model outputs for safety needs.
   - *This could potentially allow users to implement safety checks and mitigate liability concerns.*
- **Cohere refines market strategy**: **Cohere** strategically hones in on specific use cases to navigate the crowded LLM market, avoiding oversaturation.
   - Members discussed the value of **pragmatic business choices** that emphasize clarity and utility in model applications.
- **Inquiry on fine-tuning models**: A user inquired about the possibility of skipping the final `<|END_OF_TURN_TOKEN|>` during fine-tuning for smoother inference continuation.
   - They proposed a POC example of training data, highlighting potential benefits for fine-tuning chat models.
- **Sagemaker Client issues flagged**: A user reported receiving `input_tokens=-1.0` and `output_tokens=-1.0` from the Sagemaker client when accessing the endpoint.
   - This raised concerns about possible misconfigurations during the setup of the endpoint.
- **Support channel for Sagemaker queries**: A suggestion was made for the original poster to reach out to [support@cohere.com](mailto:support@cohere.com) for assistance on the Sagemaker billing issue.
   - The user indicated they would investigate the matter further by checking the user's account.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **GitHub Communique Sparks Anticipation**: A member responded to Prashant on **GitHub** regarding ongoing discussions and can be followed up with potential reactions.
   - *Stay tuned for any follow-up reactions* that might emerge from this interaction.
- **CodeBlueprint with Aider Showcased**: A member shared a link demonstrating their new coding pattern, **CodeBlueprint with Aider**, showcasing its integration potential.
   - This showcase might provide insights into employing fresh tools in coding practices.
- **Ruff Check Encountered Errors**: Prashant reported facing a **TOML parse error** when executing `ruff check . --fix-only`, indicating an unknown field `indent-width`.
   - This error highlights potential configuration mismatches that need resolving.
- **Introduction of GPT-4 Vision API Wrapper**: A new [Pull Request](https://github.com/stanfordnlp/dspy/pull/682) adds a **GPT-4 Vision API wrapper**, streamlining image analysis requests in the DSPy repository.
   - The introduction of the **GPT4Vision** class in `visionopenai.py` should simplify API interactions for developers.
- **Community Eager for Contributions and Bounties**: Members expressed enthusiasm to contribute, with one asking if there are any bounties available for participation.
   - Although needed changes were acknowledged, no specifics on bounties were disclosed during the discussion.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Compositing Techniques Shine**: Members discussed that basic **compositing** techniques are viable options for image creation, suggesting the use of libraries like **Pillow** for enhanced results.
   - *Training images with integrated text is not recommended* for achieving poster-quality visuals.
- **Post-Processing for Quality Boost**: An effective workflow involving tools like **GIMP** can greatly improve the accuracy and effectiveness of imagery through post-processing techniques.
   - *Doing it in post yields the best results* compared to relying solely on initial methods.
- **Nouswise Enhances Creative Processes**: **Nouswise** was highlighted as a personal search engine that provides trusted answers throughout various creative phases, from **reading** to **curation**.
   - Its functionalities streamline methods for **searching** and **writing**, boosting overall productivity.
- **Seeking Whisper Speech Insights**: A member inquired about experiences with **Whisper speech** technology, prompting suggestions to review a specific channel for further guidance.
   - Community discussions allowed for shared insights and *collective knowledge* with relevant links to resources.
- **StyleTTS-ZS Project Resource Call**: A member requested computational resource support for the **StyleTTS-ZS** project, which aims for efficient high-quality zero-shot text-to-speech synthesis.
   - The project is detailed on [GitHub](https://github.com/yl4579/StyleTTS-ZS), encouraging community collaboration for its development.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter impresses users**: **Open Interpreter** garnered praise for its cleverness, enhancing excitement about its functionalities within the community.
   - Members expressed eagerness to explore its potential, with ongoing discussions surrounding its features.
- **Interest peaked for beta testing**: Members inquired about available slots for **beta testers** of the **Open Interpreter**, signaling ongoing enthusiasm for contributing to its development.
   - Such inquiries reflect a keen interest in aiding the tool's advancement and improving user experiences.
- **Human Device Discord event this Friday**: An upcoming event by **Human Device** is set for this Friday, with participants encouraged to join through the [Discord link](https://discord.gg/UmXdvf3v?event=1285618083448225813).
   - This event aims to engage users in discussions about innovative technologies and offerings.
- **Tool Use Podcast highlights voice intelligence**: The latest episode of [Tool Use](https://youtu.be/La9BfaFTsFU) showcases **Killian Lucas** discussing advancements in voice intelligence and the **01 Voices** script's capabilities.
   - Listeners can expect insights into how voice agents interact in group conversations seamlessly.
- **Deepgram goes open-source**: A member announced creation of an open-source and local version of **Deepgram**, stirring enthusiasm within the community for more accessible tools.
   - This initiative emphasizes community engagement in developing effective voice intelligence solutions.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Eleuther Eval Recipe's Limited Use**: Concerns emerged regarding the **Eleuther eval recipe** and its performance with both **generation** and **multiple choice (mc)** tasks, particularly relating to the impact of **cache** from generation tasks on subsequent task executions.
   - It was confirmed by other users that the recipe is malfunctioning, suggesting potential issues tied to **cache management**.
- **Cache Reset Necessity**: Users discussed the absence of a proper cache reset as a potential source of issues, especially when switching tasks after **model generation**.
   - One member noted their practice of resetting caches post-generation, but highlighted this only prepares for a new round of generation without achieving a full reset.
- **Inconsistent Batch Size During MM Evaluations**: Discussion pointed to an issue with expected batch sizes not being met during model evaluations, particularly when caching is utilized.
   - This challenge is anticipated to reoccur when future multiple model evaluations are attempted by another user.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Community Curiosity on RISC-V Support**: Members are inquiring about any plans to support **RISC-V**, but currently there are **no plans yet** for this architecture.
   - This interest may prompt future discussions on alternative architecture compatibility.
- **Zero-copy Interoperability Lacks Mojo-Python Integration**: There's a challenge in achieving **zero-copy data interoperability** since Mojo modules cannot be imported or called from Python now.
   - The discussion included how the **Mandelbrot example** could inefficiently utilize memory via `numpy_array.itemset()`.
- **Mandelbrot Example Highlights Mojo's Potential**: A tutorial on the **Mandelbrot set** demonstrated that Mojo can execute high-performance code while integrating Python visual tools.
   - This tutorial illustrated Mojo's fit for crafting fast solutions for irregular applications leveraging Python libraries.
- **LLVM Intrinsics Now Supported at Comptime**: Mojo has extended support for **LLVM intrinsics at comptime**, focusing on functions like `ctlz` and `popcount` for integers.
   - Future developments hinge on LLVM's capacity to constant fold these intrinsics, opening pathways for broader type support.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Shampoo Gets No Love in Transformers**: A member highlighted the absence of **Shampoo** in both **Transformers** and **Axolotl**, arguing that it offers substantial benefits that are being overlooked.
   - *Shampoo is literally such a free lunch, in large scale, in predictable manner,* indicates its potential that may deserve further exploration.
- **Shampoo Scaling Law vs Adam**: Discussion around the **Shampoo Scaling Law for language models** revealed a comparative analysis against **Adam**, with a plot referencing **Kaplan et al**.
   - The plot illustrated Shampoo's effective scaling characteristics, suggesting it as a preferable choice for large models over **Adam**.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Ultralytics Calls for Community at YOLO Vision 2024!**: Ultralytics is hosting [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision) on <t:1727424000:F> - <t:1727458200:t> at Google Campus for Startups in Madrid 🇪🇸, and invites AI engineers to register and join.
   - Attendees can engage by voting for the music during the discussion panel, aiming to boost community interaction!
- **Voting for Music at YOLO Vision 2024!**: Registered participants for [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision) can vote on the music played during discussions, adding a unique interactive touch to the event.
   - This initiative encourages attendee participation and aims to create an engaging atmosphere during the event.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Finetuning (Hamel + Dan) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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




### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1285314374591844523)** (389 messages🔥🔥): 

> - `O1 Mini limits`
> - `Comparison of AI models`
> - `Perplexity features and functionalities`
> - `Integration with other services`
> - `Promise of Pro Search enhancements` 


- **O1 Mini's Daily Usage Limit**: Users discussed the recent limit of **10 uses per day** for O1 Mini on Perplexity, expressing dissatisfaction with the low cap compared to other platforms offering higher limits.
   - There are speculations that the O1 Mini access might be limited to prevent disruption of marketing strategies and manage server costs.
- **Comparison of AI Models: Claude vs. GPT-4o**: Members were unsure which AI model to choose between **Claude 3.5 and GPT-4o**, emphasizing the importance of testing both to find the better fit.
   - Discussion highlighted that GPT-4o might be superior for certain tasks, mainly due to its broader capabilities.
- **Perplexity's New Features**: The introduction of a **Reasoning focus** in Perplexity sparked interest, as users noted that it appears to utilize O1 Mini and enhances the functionality within the Pro Search environment.
   - Users were experimenting and sharing experiences with the new features, revealing advancements in the output quality and reasoning steps.
- **Integrating Perplexity with Other Tools**: Queries were raised about how to integrate Perplexity Pro with a **VSCode extension** for autocomplete functionality, indicating a desire for enhanced workflow integration.
   - Users pointed out that current functionalities exist within the Perplexity platform but integrating with external apps remains less straightforward.
- **Community Collaborations and Resources**: Users were encouraged to explore the **Complexity extension** that significantly enhances Perplexity's user experience, offering advanced functionalities and organization features.
   - Community managers within the platform emphasized the importance of user feedback and collaboration in improving tools and experiencing the platform's full potential.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.testingcatalog.com/perplexity-adopts-o1-model-amid-openais-new-message-limits/">Perplexity adopts o1 model amid OpenAI’s new message limits</a>: OpenAI&#x27;s o1 models now allow 50 messages daily, up from a weekly limit. o1 mini may soon be free. o1 features advanced reasoning, and 4o&#x27;s knowledge cutoff is now September 2024.</li><li><a href="https://docs.openinterpreter.com/getting-started/introduction">Introduction - Open Interpreter</a>: no description found</li><li><a href="https://tenor.com/view/the-universe-tim-and-eric-mind-blown-mind-blown-meme-mind-explosion-mind-explosion-meme-gif-18002878">The Universe Tim And Eric Mind Blown GIF - The Universe Tim And Eric Mind Blown Mind Blown Meme - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/testingcatalog/status/1835799548284883148?s=46">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: ChatGPT is experimenting with a new design for the Forced Search feature making it more prominent to users.   &#34;All the web, now in ChatGPT&#34;   Also spotted by @btibor91</li><li><a href="https://cplx.vercel.app/">Complexity</a>: An enhanced version of Perplexity.ai that everyone has ever wanted.</li><li><a href="https://chromewebstore.google.com/detail/complexity/ffppmilmeaekegkpckebkeahjgmhggpj">Complexity - Chrome Web Store</a>: ⚡ Supercharge your Perplexity.ai</li><li><a href="https://addons.mozilla.org/en-US/firefox/addon/complexity/">Complexity – Get this Extension for 🦊 Firefox (en-US)</a>: Download Complexity for Firefox. ⚡ Supercharge your Perplexity.ai
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1285339735585456219)** (14 messages🔥): 

> - `Minecraft Moderation Ban Issue`
> - `Microsoft's Strategy`
> - `Research Topic Discussions`
> - `Global AI Center Strength`
> - `Bitcoin's 66-bit Puzzle` 


- **Minecraft Moderation Ban Issue Discussed**: A [detailed page](https://www.perplexity.ai/page/minecraft-moderation-ban-issue-udsocXhbT8uu5egJmMjLFg) regarding the moderation ban issue in **Minecraft** has been opened for discussion among users.
   - Community members are encouraged to share their thoughts on the current moderation policies.
- **Microsoft's Business Strategy Under Scrutiny**: A post [questioning Microsoft's tactics](https://www.perplexity.ai/search/why-does-microsoft-not-seem-to-yp8liVn9QP6FoBu15ueWIQ) in the tech space raised eyebrows about their competitive stance and future direction.
   - It's suggested that users analyze whether Microsoft's actions align with its historic approach and goals.
- **Exploring New Research Topics**: A member expressed their intention to discuss a new research topic, detailing their interests in a search [post](https://www.perplexity.ai/search/i-have-a-research-topic-i-want-9LTYvWD5RNONSiW.mFBsmg).
   - They are seeking feedback and resources to develop their ideas further.
- **Global AI Center Strength Debated**: A conversation about the [Strength of Global AI Centers](https://www.perplexity.ai/page/global-ai-center-strength-trai-Bq16gnJgT8uDpzZPuWWSPg) highlighted various centers' capabilities and contributions to the AI field.
   - Participants are weighing the potential of these centers in shaping future AI advancements.
- **Bitcoin's 66-bit Puzzle Solved!**: A page detailing how the **66-bit puzzle** in **Bitcoin** was solved has generated interest among enthusiasts, viewable [here](https://www.perplexity.ai/page/bitcoin-s-66-bit-puzzle-solved-1fFxJ9Z8Ti6V83.DnGIU.Q).
   - The discussion revolves around the implications of this solution on cryptocurrency security.



**Link mentioned**: <a href="https://www.youtube.com/embed/Eq4HMjeDj08">YouTube</a>: no description found

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1285337201550299136)** (280 messages🔥🔥): 

> - `Qwen 2.5 Model Release`
> - `Mistral-Small-Instruct-2409`
> - `Unsloth Installation Issues`
> - `Fine-tuning LLMs`
> - `Backus-Naur Form (BNF)` 


- **Qwen 2.5 Introduces New Model Variants**: Qwen 2.5 features new model sizes including **0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B**, with stricter content filtering compared to its predecessor according to initial tests.
   - The model variants are reported to limit knowledge on certain topics, potentially impacting knowledge retention from certain sources.
- **Mistral-Small-Instruct-2409 Released**: The **Mistral-Small-Instruct-2409** model boasts **22B parameters**, is capable of supporting function calls, and can handle sequences up to **128k** tokens long, although it currently has non-commercial usage restrictions.
   - This model is recommended to be used with [vLLM](https://github.com/vllm-project/vllm) for efficient inference pipelines.
- **Issues with Unsloth Installation**: Users are encountering difficulties installing Unsloth and managing dependencies like **xformers**, with some receiving 'Unsupported platform' errors on Windows.
   - Advice has been offered to use **WSL** or install specific CUDA versions to resolve these installation issues.
- **Fine-tuning Strategies for Specific JSON Syntax**: To effectively fine-tune a model for specific JSON syntax, the amount of training data required may vary; 500 to thousands of examples might suffice depending on the model's prior knowledge.
   - Training quality is emphasized over quantity, and implementing **Backus-Naur Form (BNF)** is recommended for ensuring structural integrity in outputs.
- **Discussion on Backus-Naur Form (BNF)**: The usage of **BNF** can help restrict language models to adhere to a defined structure, which may enhance the performance in generating outputs that require specific formatting.
   - Understanding BNF can be crucial for parsing outputs and ensuring they maintain the required structured integrity.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-Small-Instruct-2409">mistralai/Mistral-Small-Instruct-2409 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/SmolLM-1.7B-Instruct">unsloth/SmolLM-1.7B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://x.com/zhouwenmeng/status/1834899729165304198">Tweet from Wenmeng Zhou (@zhouwenmeng)</a>: Qwen-q1 ? ? 🍓🍓🍓🍓🍓</li><li><a href="https://huggingface.co/unsloth/SmolLM-1.7B-Instruct-bnb-4bit">unsloth/SmolLM-1.7B-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/google/datagemma-release-66df7636084d2b150a4e6643">DataGemma Release - a google Collection</a>: no description found</li><li><a href="https://download.pytorch.org/whl/cu124">no title found</a>: no description found</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/0efb8bd31e4359ba9e8f52e8d003d35ff038e081/recipes/multilingual/README.md">llama-recipes/recipes/multilingual/README.md at 0efb8bd31e4359ba9e8f52e8d003d35ff038e081 · meta-llama/llama-recipes</a>: Scripts for fine-tuning Meta Llama3 with composable FSDP &amp;amp; PEFT methods to cover single/multi-node GPUs. Supports default &amp;amp; custom datasets for applications such as summarization and Q...</li><li><a href="https://huggingface.co/Qwen?sort_models=modified#models">Qwen (Qwen)</a>: no description found</li><li><a href="https://github.com/unclemusclez/ollama-toolkit">GitHub - unclemusclez/ollama-toolkit: The Ollama Toolkit is a collection of powerful tools designed to enhance your experience with the Ollama project, an open-source framework for deploying and scaling machine learning models. Think of it as your one-stop shop for streamlining workflows and unlocking the full potential of Ollama!</a>: The Ollama Toolkit is a collection of powerful tools designed to enhance your experience with the Ollama project, an open-source framework for deploying and scaling machine learning models. Think o...</li><li><a href="https://github.com/ACGNnsj/triton-windows-build/releases/">Releases · ACGNnsj/triton-windows-build</a>: Development repository for the Triton language and compiler - ACGNnsj/triton-windows-build</li><li><a href="https://huggingface.co/google/gemma-7b">google/gemma-7b · Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/KTVeTXPZD9">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/issues/3561">How to know the maximum concurrent requests /tokens that generate can handle at the same time? · Issue #3561 · vllm-project/vllm</a>: Your current environment I am wondering how to know or configure the number of concurrent requests (number of tokens). I can see from logs these values: INFO 03-18 12:34:52 llm_engine.py:706] Avg p...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1285456514219245632)** (7 messages): 

> - `Job Application Strategies`
> - `Importance of Application Knowledge`
> - `Coding Style Compliance`
> - `Fundamental Understanding in Code`
> - `Limitations of LeetCode` 


- **Prioritizing Application Knowledge over Memorization**: A member emphasized that while there is a lot of memorization involved, **application knowledge** is more crucial than simply memorizing LeetCode problems.
   - Understanding algorithms and data structures like **linked lists** and **hashmaps** is vital for real-world coding.
- **Compliance with Existing Code Styles**: It was noted that modifying code often requires adherence to **existing coding styles**, preventing arbitrary changes.
   - This means even if one has better ideas, they may not be accepted due to differences in **fundamental coding practices**.
- **Navigating Legacy Codebases**: In most companies, developers typically work within **legacy codebases** that may not use the latest technologies or methodologies.
   - This collaboration often affects the acceptance of new ideas, as peers will interact with the code written by others.
- **The Diminishing Returns of LeetCode Preparation**: A member pointed out that the focus on memorization through LeetCode doesn't necessarily translate to **applicability** in real-world scenarios.
   - Knowing how and when to apply concepts can be more important than memorizing solutions.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1285314964973813883)** (15 messages🔥): 

> - `Model fine-tuning`
> - `Mac compatibility issues`
> - `Gratitude in the community` 


- **Hallucinations in Fine-tuned Models**: A user reported that after finetuning the model 'unsloth/llama-3-8b-bnb-4bit', the downloaded version from Hugging Face started hallucinating, raising concerns about potential corruption during saving.
   - They shared their uploading command, which included `save_method = 'merged_4bit_forced'`, prompting discussion on whether this could influence model performance.
- **Mac M3 Chip Running Issues**: A user is facing an issue running a Kaggle notebook on a Mac M3 chip where they're encountering a 'Torch not compiled with CUDA enabled' error, noting the lack of CUDA support on Mac OS.
   - Another member pointed out that 'Unsloth doesn't support Mac', leaving open the question of potential solutions.
- **Community Appreciation**: One member expressed gratitude for the community's assistance in training a neural network to generate Python code, culminating in a celebratory comment.
   - Responses of encouragement and acknowledgments followed, reinforcing the supportive atmosphere within the group.



**Link mentioned**: <a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-1-8b-conversational-unsloth/notebook"> Kaggle Llama 3.1 8b Conversational Unsloth</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1285583026507616266)** (23 messages🔥): 

> - `KTO vs PPO`
> - `Domain Adaptation of Llama-3.1-8B`
> - `Continued Pre-Training vs Full Fine-Tuning`
> - `GPU Limitations`
> - `Unsloth Support` 


- **KTO reigns supreme in RLHF circles**: A member expressed strong preference for **KTO** over **ORPO** for reinforcement learning, citing its simplicity as a *thumbs up, thumbs down* dataset.
   - While acknowledging that **RLHF** methods can simplify models, they emphasized the need to *test all available options*.
- **Domain adaptation struggles with Llama-3.1-8B**: Another member seeks help with **domain adaptation** for **Llama-3.1-8B**, aiming for full fine-tuning without quantizing the weights, but encountered errors on **H100 160GB** GPUs.
   - They managed to execute continued pre-training but are eager to see how full fine-tuning performs, and are investigating increasing model precision.
- **Continued pre-training may suffice for fine-tuning**: Discussion highlighted that **continued pre-training** could achieve results close to full fine-tuning, especially when exploring parameter adjustments incrementally.
   - The community seems supportive of experimenting with higher precision models beyond quantized versions, as illustrated in the Korean notebook example.
- **Limitations of GPU resources impact research**: Frustration surfaced about **limited GPU resources**, which slows down the testing of various methods in reinforcement learning.
   - Members discussed needing to proceed carefully with evaluations one by one, akin to conserving limited fuel for a long journey.
- **Potential issues in existing coding scripts**: A member faced issues with their code when using multiple **H100-80GB GPUs** for FFT processing, generating errors despite adequate resources.
   - This led to inquiries about possible mistakes in their existing setup, highlighting the challenges encountered during domain adaptation.


  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1285318099926782023)** (238 messages🔥🔥): 

> - `O1 Models Performance`
> - `Aider and Sonnet Integration`
> - `RAG and Fine-tuning for Codebases`
> - `Use Cases for Different Models`
> - `Feedback on Flux Canvas Art` 


- **O1 Models Performance**: Discussions centered around the performance of O1 models, with some users expressing frustration over their limitations in applications due to lack of system prompts.
   - Users noted that while O1 models could be effective in playgrounds, they struggled within other applications like Aider.
- **Aider and Sonnet Integration**: Aider users suggested combining Sonnet 3.5 with O1 mini for coding tasks, with Sonnet proving more reliable for editing and coding.
   - Several users reported successful experiences using Aider for quick fixes and assignments thanks to its capabilities.
- **RAG and Fine-tuning for Codebases**: There was debate on the effectiveness of RAG for coding tasks, with some users advocating for fine-tuning models on specific codebases instead.
   - The conversation highlighted challenges faced with using retrieval mechanisms for large codebases and the processes involved.
- **Use Cases for Different Models**: Comparisons were made between models like O1-mini and Claude, focusing on their unique benefits and use-case scenarios.
   - Some users found O1 models suitable for generating code from scratch but lacking in refactoring and code editing scenarios.
- **Feedback on Flux Canvas Art**: A user requested feedback on their website, Flux Canvas Art, seeking insights from the community.
   - This request for feedback appeared amidst discussions on various technologies and tools being utilized in development.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>: How to configure aider with a yaml config file.</li><li><a href="https://aider.chat/docs/repomap.html">Repository map</a>: Aider uses a map of your git repository to provide code context to LLMs.</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: Using the chat, ask and help chat modes.</li><li><a href="https://trypear.ai/">PearAI - Open Source AI Code Editor for Fast Development</a>: PearAI is an Open-source AI-powered code editor with features like AI chat, inline prompts, and debugging to accelerate your coding process.</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: Quantitative benchmarks of LLM code editing skill.</li><li><a href="https://x.com/openai/status/1835851762609357292?s=46">Tweet from OpenAI (@OpenAI)</a>: We appreciate your excitement for OpenAI o1 and we want you to be able to use it more.  We have increased rate limits for o1-mini by 7x, from 50 messages per week to 50 messages per day.  o1-preview i...</li><li><a href="https://fluxcanvas.art/">Flux Canvas Art</a>: no description found</li><li><a href="https://github.com/mckaywrigley/o1-ai-playground">GitHub - mckaywrigley/o1-ai-playground: Come join the best place on the internet to learn AI skills. Use code &quot;o1launchparty&quot; for an extra 20% off.</a>: Come join the best place on the internet to learn AI skills. Use code &quot;o1launchparty&quot; for an extra 20% off. - mckaywrigley/o1-ai-playground</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/CONTRIBUTING.md#setting-up-a-development-environment">aider/CONTRIBUTING.md at main · paul-gauthier/aider</a>: aider is AI pair programming in your terminal. Contribute to paul-gauthier/aider development by creating an account on GitHub.</li><li><a href="http://fluxcanvas.art/">Flux Canvas Art</a>: no description found</li><li><a href="https://github.com/mckaywrigley/o1-ai-playground/pull/2">Fix for dark system theme hydration error by fry69 · Pull Request #2 · mckaywrigley/o1-ai-playground</a>: Fix for hydration error when the system is set to a dark theme by default. Solution taken from here -&gt; facebook/react#17741 (comment)</li><li><a href="https://github.com/DoS007/big-AGI-2">GitHub - DoS007/big-AGI-2: Generative AI suite powered by state-of-the-art models and providing advanced AI/AGI functions. It features AI personas, AGI functions, multi-model chats, text-to-image, voice, response streaming, code highlighting and execution, PDF import, presets for developers, much more. Deploy on-prem or in the cloud.</a>: Generative AI suite powered by state-of-the-art models and providing advanced AI/AGI functions. It features AI personas, AGI functions, multi-model chats, text-to-image, voice, response streaming, ...</li><li><a href="https://github.com/enricoros/big-AGI.git">GitHub - enricoros/big-AGI: Generative AI suite powered by state-of-the-art models and providing advanced AI/AGI functions. It features AI personas, AGI functions, multi-model chats, text-to-image, voice, response streaming, code highlighting and execution, PDF import, presets for developers, much more. Deploy on-prem or in the cloud.</a>: Generative AI suite powered by state-of-the-art models and providing advanced AI/AGI functions. It features AI personas, AGI functions, multi-model chats, text-to-image, voice, response streaming, ...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1285319914000744470)** (45 messages🔥): 

> - `Aider configuration`
> - `Azure OpenAI integration`
> - `User story implementation`
> - `Streaming output metrics`
> - `OpenRouter outages` 


- **Aider setup for Azure OpenAI**: A user shared their process of configuring Aider to work with Azure OpenAI, needing to pass a user key in JSON format for each request.
   - Suggestions were made to explore LiteLLM documentation for passing Azure API keys and handling specific request formats.
- **Enhancing app features using user stories**: A user compared Aider's capabilities to those of marblism.com, emphasizing how both can help create app features through task management.
   - They expressed interest in using Aider to implement user stories to improve their app development process.
- **Challenges with streaming metrics**: A user inquired about obtaining accurate metrics while using Aider's streaming functionality, mentioning the difficulty in assessing completion status.
   - They noted that disabling streaming significantly worsened their experience, highlighting a need for balance.
- **Reported OpenRouter outages**: Several users discussed experiencing outages with OpenRouter, seeking confirmation from others on the platform's status.
   - It was mentioned that services should be operational, but issues persisted for some users.
- **User interface tools compared**: A user introduced marblism.com as a similar tool to Aider for app creation, noting its focus on working with user stories for feature development.
   - They suggested exploring how Aider could similarly structure tasks to improve app functionality.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://some-azure-endpoit',">no title found</a>: no description found</li><li><a href="https://aider.chat/docs/llms/azure.html">Azure</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>: How to configure aider with a yaml config file.</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://aider.chat/docs/usage/commands.html">In-chat commands</a>: Control aider with in-chat commands like /add, /model, etc.</li><li><a href="https://docs.litellm.ai/docs/providers/azure">Azure OpenAI | liteLLM</a>: API Keys, Params
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1285361804431589397)** (8 messages🔥): 

> - `Superflex AI Assistant`
> - `Claude 3.5 Artifacts`
> - `RethinkMCTS Algorithm`
> - `Code Integration from Figma`
> - `Optillm for Inference Proxy` 


- **Superflex AI Transforms Figma Designs**: Superflex allows users to write front-end code directly from [Figma designs](https://www.producthunt.com/posts/superflex), integrating smoothly with existing projects and offering enhanced functionality over previous iterations focused on prototyping.
   - Integrating wireframes into the codebase using Superflex is feasible, especially with designs made using a UI kit, making it an appealing choice for developers.
- **Insights on Claude 3.5 Artifacts Prompt**: A shared link revealed the **extracted system prompt for Claude 3.5**, providing detailed insights into its artifacts; it's a useful resource for developers working with AI models.
   - The prompt can be viewed through a [GitHub gist](https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d) which contains the necessary information to enhance auto-generative capabilities.
- **RethinkMCTS Enhances LLM Code Generation**: The paper introduces **RethinkMCTS**, an algorithm utilizing Monte Carlo Tree Search to enhance code generation by refining search strategies through detailed execution feedback ([View Paper](https://www.arxiv.org/abs/2409.09584)).
   - This method aims to address previous limitations in search quality during code generation tasks, potentially leading to more relevant outputs.
- **Discussion on Direct Image Integration Methods**: Concerns were raised about how Superflex compares to simply pasting images for UI creation, as some users found clipboard methods effective for integrating UI components.
   - This highlights ongoing debates within the developer community about the best practices for translating design into functional code.
- **Optillm: Optimizing Inference for LLMs**: A link to the **Optillm GitHub repo** was shared, emphasizing an optimizing inference proxy designed specifically for LLMs ([GitHub Link](https://github.com/codelion/optillm)).
   - This tool aims to enhance performance and usability in working with large language models, crucial for developers looking to streamline their workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.arxiv.org/abs/2409.09584">RethinkMCTS: Refining Erroneous Thoughts in Monte Carlo Tree Search for Code Generation</a>: LLM agents enhanced by tree search algorithms have yielded notable performances in code generation. However, current search algorithms in this domain suffer from low search quality due to several reas...</li><li><a href="https://github.com/codelion/optillm">GitHub - codelion/optillm: Optimizing inference proxy for LLMs</a>: Optimizing inference proxy for LLMs. Contribute to codelion/optillm development by creating an account on GitHub.</li><li><a href="https://www.producthunt.com/posts/superflex"> Superflex - Write Front-End Code 10x Faster ⚡️ | Product Hunt</a>: Superflex is an AI assistant that turns ideas into front-end code from Figma designs, images, or text prompts—matching your coding style and utilizing your UI components. Build better frontend faster ...</li><li><a href="https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d">Extracted Claude 3.5 Sonnet system prompt for artifacts</a>: Extracted Claude 3.5 Sonnet system prompt for artifacts - claude_35_artifacts_system_prompt.txt
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1285314911005577389)** (154 messages🔥🔥): 

> - `GPU and Performance Issues`
> - `Model Training Challenges`
> - `New Features in LM Studio`
> - `Community Tools and Extensions`
> - `User Experience and Settings` 


- **GPU Performance Troubleshooting**: A user expressed frustration over LM Studio not utilizing their GPU, repeatedly using CPU and RAM instead. Another noted that GPU settings are found under **Settings -> System Resources**, confirming GPU usage if monitored in **Task Manager**.
   - An identified issue causing blurry screens was related to anti-aliasing settings, with suggestions to disable certain configurations for improvement.
- **Training a Small Language Model**: A user reported training a 100k parameter model, initially encountering training duration estimates of up to 5 days. Adjusting the number of tokens and batch size significantly reduced the estimate to about 1.3 hours.
   - Concerns were raised regarding potential bottlenecks in the data loader, leading to long wait times during training, with discussions on using PyTorch effectively.
- **Exciting New Features in LM Studio**: Users discussed the recent introduction of document integration in LM Studio, a major request from the community. Enthusiastic feedback indicated users’ eagerness to test the updated version of the application.
   - One member noted that the simplicity of LM Studio was a significant advantage for those without extensive IT knowledge.
- **Community Tools and Extensions Development**: A member shared a discord app they developed for LM Studio, highlighting community-driven development efforts. There was also a request for extensions, reflecting interest in custom tools for enhanced functionality.
   - Responses indicated a growing community interest and potential for collaboration in creating useful tools and extensions around the LM Studio platform.
- **User Experience with Settings and Features**: Users shared issues and clarifications regarding system prompts and feature integrations within their chats in LM Studio. Guidance was provided on accessing system settings to ease repetitive tasks during model interaction.
   - The conversation underscored the importance of user-friendly features for improving overall experience, especially for those new to the platform.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmstudio-community/datagemma-rag-27b-it-GGUF">lmstudio-community/datagemma-rag-27b-it-GGUF · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/acting-nod-hmph-gif-18509831">Acting Nod GIF - Acting Nod Hmph - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1285387903249416192)** (46 messages🔥): 

> - `GPU Recommendations for LM Studio`
> - `Dual GPU Setups`
> - `Used GPU Market Insights`
> - `Intel ARC Performance`
> - `VRAM Importance in LLMs` 


- **Choosing between 4090 and 4080 for VRAM**: A member considers getting a **4090** for maximum **VRAM** but grapples with its high price over the **4080**, questioning if the performance gap is worthy.
   - Another suggests that two **4060 Ti's** might offer more VRAM and be more cost-effective, paired with lower power draw.
- **The merits of dual GPU setups**: Discussions suggest that using two **4060 Ti's** can maximize VRAM without exceeding power limits, making it a practical choice.
   - Participants note that using identical GPUs can simplify setups, and careful power management can reduce overall energy costs.
- **Searching for used GPUs in the market**: Members share insights about hunting for **used 3090s** in varying regions, highlighting challenges like pricing and availability.
   - While some find deals on eBay, others prefer local platforms like **Kijiji** for used parts, mentioning prices around **$800-920 CAD**.
- **Intel ARC's role in LLM performance**: A member inquires about using **Intel ARC A770** for LLMs, leading to discussions about performance metrics leveraging SYCL backend.
   - Claims are made suggesting that ARC configurations can yield **34 tokens per second** with potential increases via **IPEX**.
- **The criticality of VRAM in LLMs**: Concerns arise surrounding the need for sufficient **VRAM**, emphasizing that most powerful models may require more than what current cards offer.
   - Members discuss their experiences with **token generation** rates on various GPUs, particularly stressing the importance of thickness in VRAM.


  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1285707527459180595)** (1 messages): 

> - `Hugging Face API docs`
> - `TRL v0.10 release`
> - `Sentence Transformers v3.1`
> - `DataCraft for synthetic datasets`
> - `Core ML Segment Anything 2` 


- **Revamped Hugging Face API Docs**: Hugging Face has [unveiled new API docs](https://x.com/Wauplin/status/1835715850583564713) highlighting clearer **rate limits**, a dedicated **PRO section**, enhanced code examples, and more detailed parameter lists.
   - This update aims to simplify **AI deployment** and enhance user experience.
- **Introducing TRL v0.10 for Vision-Language Models**: [TRL v0.10](https://x.com/QGallouedec/status/1833893093793304950) has released new features enabling fine-tuning for **vision-language models** in just two lines of code, coinciding with Mistral's release of Pixtral.
   - This minimalistic approach makes integrating new models significantly easier and more efficient.
- **Sentence Transformers v3.1 Launch**: The latest [Sentence Transformers v3.1](https://x.com/tomaarsen/status/1833870859552928172) includes a **hard negatives mining utility** to improve model training and a new strong loss function.
   - It also supports training with streaming datasets and custom modules, enhancing flexibility in model development.
- **DataCraft Simplifies Synthetic Dataset Creation**: [DataCraft](https://x.com/dvilasuero/status/1835711765570630017) has been introduced to help users create synthetic datasets using natural language in a no-code UI, addressing the challenges in generating high-quality data.
   - It leverages best practices for dataset generation, making it more accessible for users to build effective datasets.
- **Core ML Segment Anything 2 is Here**: The launch of [Segment Anything 2](https://x.com/pcuenq/status/1834616110475514343) for Core ML showcases on-device ML capabilities with a demo app available on Mac.
   - This development points towards a promising future for **on-device AI** applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Wauplin/status/1835715850583564713)">Tweet from Wauplin (@Wauplin)</a>: I&#39;m thrilled to unveil our revamped Inference API docs! We&#39;ve tackled your feedback head-on: clearer rate limits, dedicated PRO section, better code examples, and detailed parameter lists for ...</li><li><a href="https://x.com/QGallouedec/status/1833893093793304950)">Tweet from Quentin Gallouédec (@QGallouedec)</a>: Perfect timing! @MistralAI released Pixtral, their first multimodal model just when our fresh new release of TRL added vision-language models fine-tuning in two lines 🌟</li><li><a href="https://x.com/qlhoest/status/1829145570465722578)">Tweet from Quentin Lhoest 🤗 (@qlhoest)</a>: 🤗Hugging Face Datasets users rejoice !  I made a few lines of code for ✨PySpark✨ to read/write from/to HF Datasets. All distributed and optimized !  Code snippet / docs and JupyterLab demo below 🧡</li><li><a href="https://x.com/tomaarsen/status/1833870859552928172)">Tweet from tomaarsen (@tomaarsen)</a>: Sentence Transformers v3.1 is out! Featuring a hard negatives mining utility to get better models out of your data, a new strong loss function, training with streaming datasets, custom modules, bug fi...</li><li><a href="https://x.com/dvilasuero/status/1835711765570630017)">Tweet from Daniel Vila Suero (@dvilasuero)</a>: 🧶 Introducing DataCraft: build synthetic datasets using natural language!  Creating good quality synthetic data is difficult. It’s a trial and error process and requires a lot of tricks.  DataCraft p...</li><li><a href="https://x.com/pcuenq/status/1834616110475514343)">Tweet from Pedro Cuenca (@pcuenq)</a>: Announcing SAM 2 Studio and Core ML Segment Anything 2!  I&#39;m super excited about on-device ML, and firmly believe that it will be a big part of the future of AI. We converted Segment Anything 2 to...</li><li><a href="https://x.com/OzzyGT/status/1834594141822406796)">Tweet from Alvaro Somoza (@OzzyGT)</a>: Want to know how to erase/fill parts of an image with diffusers? It&#39;s been a while, but finally I have a new guide and a space you can try for this. You can read about it in this blog post:  https...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1285319311312814160)** (135 messages🔥🔥): 

> - `Short-form video tools`
> - `Hugging Face Inference API updates`
> - `FSDP GPU usage`
> - `CogvideoX img2vid capabilities`
> - `Hugging Face SQL Console launch` 


- **Best Tools for Short-form Videos**: A member suggested using [visla.us](https://visla.us) with the OpenAI Visla plugin as the best tool for creating short-form videos like TikToks.
   - This sparked a conversation about the effectiveness and performance of various tools in the community.
- **Revamped Inference API Docs Unveiled**: Many users expressed excitement over the newly updated [Inference API documentation](https://huggingface.co/docs/api-inference) that includes clearer rate limits, better code examples, and a dedicated PRO section.
   - This update aims to simplify AI deployment and improve user experience as datasets on Hugging Face continue to grow.
- **Confusion Over FSDP GPU Memory Usage**: A user shared confusion over unexpectedly high memory usage during fine-tuning of an 8B LLaMA model with FSDP and BF16 AMP, seeing 29G used across 8 GPUs.
   - Suggestions included debugging with raw PyTorch calls and the potential for optimized resource usage in FSDP.
- **Cognitive Capabilities of CogvideoX**: Members discussed the impressive capabilities of the new [CogvideoX img2vid](https://huggingface.co/spaces), noting its efficiency in generating videos with minimal VRAM usage.
   - Despite some initial criticisms regarding basic shots, others praised its ability to handle complex scenes like walking or riding a scooter.
- **Launch of SQL Console for Datasets**: The community celebrated the introduction of the SQL Console feature on Hugging Face, allowing users to run SQL queries directly on datasets, enhancing discoverability and usability.
   - Users were encouraged to share thoughts and SQL snippets related to this new functionality as the demand for dataset management grows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Wauplin/status/1835715850583564713">Tweet from Wauplin (@Wauplin)</a>: I&#39;m thrilled to unveil our revamped Inference API docs! We&#39;ve tackled your feedback head-on: clearer rate limits, dedicated PRO section, better code examples, and detailed parameter lists for ...</li><li><a href="https://huggingface.co/code-of-conduct#:~:text=Our%20Standards&text=Demonstrating%20empathy%20and%20kindness%20toward,and%20learning%20from%20the%20experience">Code of Conduct – Hugging Face</a>: no description found</li><li><a href="https://docs.omniverse.nvidia.com/composer/latest/index.html">USD Composer Overview &mdash; Omniverse USD Composer latest documentation</a>: no description found</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/blog/sql-console">Introducing the SQL Console on Datasets</a>: no description found</li><li><a href="https://huggingface.co/spaces/sanchit-gandhi/whisper-jax-spaces">Whisper JAX - a Hugging Face Space by sanchit-gandhi</a>: no description found</li><li><a href="https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html">Advanced Model Training with Fully Sharded Data Parallel (FSDP) — PyTorch Tutorials 2.4.0+cu121 documentation</a>: no description found</li><li><a href="https://github.com/unclemusclez/ollama-toolkit">GitHub - unclemusclez/ollama-toolkit: The Ollama Toolkit is a collection of powerful tools designed to enhance your experience with the Ollama project, an open-source framework for deploying and scaling machine learning models. Think of it as your one-stop shop for streamlining workflows and unlocking the full potential of Ollama!</a>: The Ollama Toolkit is a collection of powerful tools designed to enhance your experience with the Ollama project, an open-source framework for deploying and scaling machine learning models. Think o...</li><li><a href="https://github.com/yt-dlp/yt-dlp/issues/10128">[youtube] Sign in to confirm you’re not a bot. This helps protect our community · Issue #10128 · yt-dlp/yt-dlp</a>: DO NOT REMOVE OR SKIP THE ISSUE TEMPLATE I understand that I will be blocked if I intentionally remove or skip any mandatory* field Checklist I&#39;m asking a question and not reporting a bug or reque...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1285363482815893621)** (5 messages): 

> - `Learning Manim`
> - `ML Data Pipelines with PyTorch`
> - `Hugging Face Dataset Issues` 


- **Exploring ML Data Pipelines**: Another user shares their journey of learning about **ML data pipelines with PyTorch** while focusing on training **1D CNN classifiers**.
   - This discussion sparked interest, prompting others to ask for resources.
- **Image Issues in Hugging Face Dataset**: A member expressed confusion while working with a dataset on **Hugging Face**, not seeing images despite being sure it was their mistake.
   - They linked the **synthetic drilling dataset** directly, inviting feedback from the community.



**Link mentioned**: <a href="https://huggingface.co/datasets/jonasmaltebecker/synthetic_drilling_dataset/viewer/default/validation?row=1">jonasmaltebecker/synthetic_drilling_dataset · Datasets at Hugging Face</a>: no description found

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1285339908197580810)** (11 messages🔥): 

> - `Inference API Documentation Improvements`
> - `Model Growth and Downloads Discussion`
> - `AI Community Engagement` 


- **Revamped Inference API Docs Launched**: A member announced the launch of improved [Inference API docs](https://huggingface.co/docs/api-inference), highlighting clearer rate limits, a dedicated PRO section, better code examples, and detailed parameter lists.
   - Another user expressed excitement, saying, *I love ittttt ❤️*.
- **Race to 1 Million Models**: Discussion sparked around who would reach 1 million first: flux or total number of models, with speculation that **1M models** might be achieved next week.
   - A member noted, *I'm getting some stats and we're close to 40K weekly models 🤯*.
- **WhatsApp Group for AI Enthusiasts**: A request was made for a WhatsApp group dedicated to AI discussions.
   - Another member advised against cross-posting in response to this inquiry.



**Link mentioned**: <a href="https://x.com/Wauplin/status/1835715850583564713>">Tweet from Wauplin (@Wauplin)</a>: I&#39;m thrilled to unveil our revamped Inference API docs! We&#39;ve tackled your feedback head-on: clearer rate limits, dedicated PRO section, better code examples, and detailed parameter lists for ...

  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1285328853073657949)** (5 messages): 

> - `Behavioral Biometric Recognition in Minecraft`
> - `PowershAI Multilingual Documentation`
> - `Nvidia Mini-4B Model Release`
> - `HuggingFace Agent Registration`
> - `Continuous MFA and Ban Evasion Detection` 


- **Behavioral Biometric Recognition in Minecraft**: A member showcased their model that identifies players based on their mouse movements in **Minecraft**, aimed at **continuous MFA** and detecting **ban evasion**.
   - The project, featuring an open-sourced repository on [GitHub](https://github.com/templateprotection/AimNet-Mouse-Dynamics), emphasizes potential applications beyond gaming.
- **PowershAI Documentation Translated**: The updated **PowershAI documentation** is now available in multiple languages, all utilizing the tool itself, and is hosted on [GitHub](https://github.com/rrg92/powershai).
   - Members are welcomed to review translations and suggest new ones to enhance accessibility.
- **Nvidia Releases Mini-4B Model**: Nvidia introduced the **Mini-4B model**, noted for its compact size but requiring specific **Nvidia drivers** for operation beyond device limits.
   - This model is touted as performing the best within its size category, viewable on the [Hugging Face Space](https://huggingface.co/spaces/Tonic/Nemotron-Mini-4B).
- **HuggingFace Agent Registration Suggestion**: A suggestion was made to register the recently released Mini-4B as a **HuggingFace agent**, which would enable querying **SQL** and integrating it with other agents.
   - This integration could significantly enhance functionality and user interactivity.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Tonic/Nemotron-Mini-4B">Minitron - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://github.com/rrg92/powershai">GitHub - rrg92/powershai: Powershell + AI</a>: Powershell + AI. Contribute to rrg92/powershai development by creating an account on GitHub.</li><li><a href="https://github.com/templateprotection/AimNet-Mouse-Dynamics">GitHub - templateprotection/AimNet-Mouse-Dynamics: An open sourced approach to One-Shot Learning for Mouse Dynamics recognition in PyTorch. This includes tools for data preprocessing, training both classification and embedding models, and evaluating model performance on a Minecraft dataset.</a>: An open sourced approach to One-Shot Learning for Mouse Dynamics recognition in PyTorch. This includes tools for data preprocessing, training both classification and embedding models, and evaluatin...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1285349051318931645)** (5 messages): 

> - `Downloading LLaMA3`
> - `Using PyTorch`
> - `MLIR Conversion Tool` 


- **Need assistance with LLaMA3 download**: A member requested help in downloading and running the **LLaMA3** open-source LLM using **PyTorch** and expressed gratitude for any guidance.
   - *‘I didn’t find anything that useful,’* reflecting the difficulty in getting started with the model.
- **Clarification on PyTorch usage**: Another member questioned why **PyTorch** was chosen for the implementation of LLaMA3.
   - This prompted confusion from the initial user, who sought clarity on the concern.
- **Uploading LLaMA3 model locally**: The member clarified that they were simply trying to upload the **LLaMA3** model locally to their setup.
   - A follow-up indicated the need for **PyTorch** due to compatibility with a tool that converts **PyTorch** code into **MLIR**.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1285637296053551167)** (1 messages): 

> - `Gradio Office Hours` 


- **Join Gradio Office Hours Now!**: <@997168115802722304> is hosting Gradio office hours right now in our Discord, offering a chance to discuss **Gradio**, **HF**, and **AI**.
   - Everyone is invited to join the conversation, find out more at [this link](https://t.co/Dxeb0jaQ6e).
- **Chat with Experts at Gradio**: The ongoing office hours are designed for those interested in **Gradio** topics, such as new features and updates from **HF** and **AI** advancements.
   - All participants are encouraged to join in and engage, making it a great opportunity to ask questions and share insights.



**Link mentioned**: <a href="https://t.co/Dxeb0jaQ6e">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1285364699139018852)** (99 messages🔥🔥): 

> - `GPT-4o Performance`
> - `Alpha Rollouts`
> - `AI Implementation in Businesses`
> - `Custom GPTs for Code Snippets`
> - `LLM Benchmarks` 


- **GPT-4o impresses in GeoGuessr**: Members expressed surprise at how well **GPT-4o** performs in **GeoGuessr**, despite not beating top expert players.
   - One member noted that it doesn't follow the expected speed of the **o1-mini** model when responding.
- **Accidental Alpha Rollouts Spark Discussion**: The potential unintentional rollout of the **Alpha** features led to speculation among users who are curious about their access.
   - Users are frustrated as some are experiencing glitches though the features appear to be available.
- **Selling AI Solutions in Local Business**: A member expressed confidence in selling **AI solutions** to local businesses and sought advice from those with experience.
   - The conversation focused on strategies for closing deals and driving adoption of AI technologies.
- **Custom GPTs for Code Snippet Management**: A user inquired about AI solutions for managing and reusing code snippets effectively, highlighting the need for better organization.
   - Members suggested using **Custom GPTs** and emphasized the importance of uploading well-commented knowledge bases.
- **Searching for LLM Benchmark Information**: A member asked for resources providing comprehensive benchmarks for various **LLM models**.
   - Others recommended using **lmsys.org** and consulting **GPT-4o** for helpful options.



**Link mentioned**: <a href="https://status.openai.com/">OpenAI Status</a>: no description found

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1285325298380050566)** (26 messages🔥): 

> - `Fine-tuning Limitations`
> - `Advanced Voice Mode Availability`
> - `Custom GPT Sharing`
> - `Token Refresh Confusion` 


- **Fine-tuning job hits hard limit again**: A user expressed frustration about a fine-tuning job exceeding their hard limit, with a cost of **$24.31** and only **$19.91** left in their quota.
   - Another member speculated it might be a discount rather than a quota issue, leading to discussions about proceeding with the job.
- **When will Advanced Voice Mode be available?**: Multiple members shared that they're using Plus but do not yet have access to the **Advanced Voice Mode**.
   - One user mentioned that the **expected availability** is by the **end of Fall**.
- **Guidance on custom GPT sharing needed**: A user sought assistance on sharing their customized GPTs without revealing their full billing name, as it was greyed out.
   - They inquired if enabling a builder profile would allow them to change their display name and asked for users willing to test their GPTs.
- **Token refresh timing raises concerns**: A user showed uncertainty about their **free tokens** refreshing and hesitated to test it for fear of unexpected charges.
   - They mentioned a suggestion from an **ask-ai** channel indicating a refresh at **midnight UTC**.
- **Discussion on page loading issues**: One user reported encountering a **404 error** when trying to load a page, prompting concerns among other members.
   - This technical glitch seemed to resonate with others but was not elaborated further.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1285626364900413501)** (3 messages): 

> - `Auto prompt for ideogram/midjourney`
> - `Prompt sharing practice`
> - `Library resources` 


- **Sharing Auto Prompt for Ideogram/Midjourney**: A member created an **auto prompt for Ideogram/Midjourney** detailing all the necessary steps and encouraged others to rate it.
   - They expressed openness to sharing it widely, indicating that feedback would be appreciated.
- **Interest in Creative Prompts**: The member asked if others were interested in the newly created prompt for Ideogram/Midjourney.
   - *Free for sharing*, the prompt is positioned as a resource for others in the community.
- **Discussion on Official Libraries**: There was a brief mention about **official libraries**, although no detailed discussion followed.
   - The context of this mention remains vague, needing further exploration in future conversations.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1285626364900413501)** (3 messages): 

> - `Auto prompt for ideogram/midjourney`
> - `Official libraries` 


- **Creating Auto Prompts for Ideogram/Midjourney**: A member shared an auto prompt for **ideogram/midjourney** that includes all the necessary steps and mentioned it is free for sharing.
   - The member encouraged others to **rate** the prompt and asked if anyone is **interested**.
- **Discussion on Official Libraries**: The term '**official libraries**' was mentioned, suggesting a potential topic of interest for users.
   - Details or context surrounding this mention were not provided, leaving it open for further discussion.


  

---



### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1285584253253455923)** (1 messages): 

> - `OpenRouter integration`
> - `Google Sheets Addon Features`
> - `Updates and Improvements`
> - `User Feedback`
> - `Support for Multiple Models` 


- **OpenRouter successfully integrated into Google Sheets**: OpenRouter has been added to the [GPT Unleashed for Sheets](https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147) addon following a user's request, making it available for free.
   - *I personally love using OR too* and hope to receive valuable feedback and more users along the way.
- **Innovative Features enhance Google Sheets performance**: The addon includes features like 'jobs', 'contexts', and 'model presets' to streamline prompt engineering and boost productivity.
   - Users can assign short codes to prompts, making it easier to reuse and refine AI outputs.
- **September Updates boost Addon functionality**: Recent updates have added support for **Claude** from Anthropic, increased UX/UI enhancements, and improved overall performance.
   - With OpenRouter integration, users can now access **100+ models** within the addon.
- **User testimonials highlight addon benefits**: Users appreciate that the addon is **free forever**, supports numerous popular language models, and simplifies AI tool building.
   - Key benefits include massive productivity boosts and effective tracking of results and API calls.



**Link mentioned**: <a href="https://workspace.google.com/marketplace/app/gpt_unleashed_for_sheets/353298171147">GPT Unleashed for Sheets™ - Google Workspace Marketplace</a>: no description found

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1285324078953267313)** (117 messages🔥🔥): 

> - `OpenRouter API Issues`
> - `Gemini Image Generation`
> - `Prompt Caching Usage`
> - `Mistral API Price Drops`
> - `Model Performance and Ratings` 


- **OpenRouter API experiencing issues**: Several users reported problems accessing OpenRouter, especially regarding the `o1` models, which led to confusion over rate limits and requests exhaustion.
   - One user noted a temporary outage in Switzerland but later confirmed functionality returned after initial issues.
- **Gemini's contrasting capabilities in image generation**: Users discussed discrepancies between Gemini’s image generation capabilities on its official site compared to its performance via OpenRouter.
   - It was clarified that the Gemini chatbot integrates image generation from Imagen models, while OpenRouter utilizes Google Vertex AI for Gemini models.
- **Understanding prompt caching**: A discussion on prompt caching illuminated its cost efficiency, allowing repeated use of prompts to reduce expenses on subsequent queries.
   - Users highlighted examples where essential prompt components could be cached, saving on costs during multiple related inquiries.
- **Significant price reductions on Mistral API**: Announcements indicated substantial price drops for Mistral APIs, with new pricing set at $2 for Large 2 models, attracting positive comparisons to other providers.
   - This price change is seen as competitive and could impact user decisions on which models to utilize for API requests.
- **Model performance discussions**: Users shared differing opinions on the performance of vision models, noting that Google's flash models appeared to outperform Pixtral 12B in certain aspects.
   - Conversations also included insights on commonly faced rate limits and performance issues that are typical in ongoing testing and usage scenarios.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/september-24-release/">AI in abundance</a>: Introducing a free API, improved pricing across the board, a new enterprise-grade Mistral Small, and free vision capabilities on le Chat.</li><li><a href="https://openrouter.ai/settings/keys">Keys | OpenRouter</a>: Manage your keys or create new ones</li><li><a href="https://gemini.google.com.">‎Gemini - chat to supercharge your ideas</a>: Bard is now Gemini. Get help with writing, planning, learning, and more from Google AI.</li><li><a href="https://openrouter.ai/credits">Credits | OpenRouter</a>: Manage your credits and payment history</li><li><a href="https://openrouter.ai/activity">Activity | OpenRouter</a>: See how you&#x27;ve been using models on OpenRouter.</li><li><a href="https://openrouter.ai/activity?api_key_id=359060">Activity | OpenRouter</a>: See how you&#x27;ve been using models on OpenRouter.</li><li><a href="https://mistral.ai/news/pixtral-12b/">Announcing Pixtral 12B</a>: Pixtral 12B - the first-ever multimodal Mistral model. Apache 2.0.</li><li><a href="https://openrouter.ai/models/mistralai/pixtral-12b:free">Pixtral 12B (free) - API, Providers, Stats</a>: The first image to text model from Mistral AI. Its weight was launched via torrent per their tradition: https://x. Run Pixtral 12B (free) with API</li><li><a href="https://openrouter.ai/activity?api_key_id=496719">Activity | OpenRouter</a>: See how you&#x27;ve been using models on OpenRouter.</li><li><a href="https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb">anthropic-cookbook/misc/prompt_caching.ipynb at main · anthropics/anthropic-cookbook</a>: A collection of notebooks/recipes showcasing some fun and effective ways of using Claude. - anthropics/anthropic-cookbook
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1285334283363291207)** (14 messages🔥): 

> - `Metal discussion group`
> - `ZML project insights`
> - `Zig programming language`
> - `ATen in Zig`
> - `CUDA support in Zig` 


- **Interest in creating a Metal discussion group**: A member suggested creating a Metal discussion group, wondering if others would be interested in exploring the [Metal-Puzzles GitHub repository](https://github.com/abeleinin/Metal-Puzzles).
   - They highlighted that the project focuses on solving puzzles to learn Metal in a collaborative environment.
- **Exploring the ZML high-performance AI stack**: Members shared their interest in the [ZML project](https://github.com/zml/zml), noting its potential for high-performance AI inference, appealing especially to those involved in programming language design.
   - They discussed whether Zig could simplify development compared to C++ in complex frameworks like PyTorch.
- **Comparing Zig's applicability to C++**: There was a discussion on whether Zig at the lower level could maintain compatibility with Python, despite the differences in programming paradigms.
   - Members reflected on the challenges faced with PyTorch internals and debated the potential improvements Zig could offer.
- **Curiosity about ATen in Zig**: A member expressed interest in how the ATen library would appear if it were implemented in Zig, envisioning potential benefits and optimizations.
   - This sparked a discussion about the implications for AI frameworks and their underlying infrastructures.
- **Support for CUDA in Zig**: A member mentioned the importance of supporting CUDA within the Zig programming environment, suggesting it would enhance the language's applicability in the field of AI.
   - This reflects a broader interest in leveraging Zig for high-performance computing tasks, including those that involve GPU acceleration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/zml/zml/">GitHub - zml/zml: High performance AI inference stack. Built for production. @ziglang / @openxla / MLIR / @bazelbuild</a>: High performance AI inference stack. Built for production. @ziglang / @openxla / MLIR / @bazelbuild - zml/zml</li><li><a href="https://github.com/abeleinin/Metal-Puzzles">GitHub - abeleinin/Metal-Puzzles: Solve Puzzles. Learn Metal 🤘</a>: Solve Puzzles. Learn Metal 🤘. Contribute to abeleinin/Metal-Puzzles development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1285394688911605760)** (9 messages🔥): 

> - `Triton Developer Conference`
> - `Proton Tutorial`
> - `Triton CPU/ARM Development`
> - `Shoutout at Keynote`
> - `CUDA Community` 


- **Attendance Queries for Triton Developer Conference**: A member inquired about attendance at tomorrow's **Triton Developer Conference**, seeking to connect with someone involved in the organization.
   - *DM me if you're involved with the organization!*
- **Proton Tutorial Impresses Attendees**: A conference-goer praised the **Proton tutorial**, describing it as a pretty nice tool.
   - They linked to the tutorial [notebook](https://github.com/Deep-Learning-Profiling-Tools/triton-samples/blob/main/Triton_Tools_Tutorial.ipynb) for further exploration.
- **Keynote Shoutout for Community**: A member reported receiving a shoutout at the **Triton conference keynote** from another participant.
   - There was light-hearted banter about encouraging more people to join the discussions surrounding the conference.
- **Triton CPU/ARM Development Conversation**: There was a query regarding the nature of current development for **Triton CPU and ARM**, specifically if it's open or closed source.
   - Members seem eager to understand the specifics of that ongoing work.
- **Praising the CUDA Community Server**: A member expressed appreciation for the server, stating it's definitely the **best** one for CUDA discussions.
   - Others echoed that sentiment, reinforcing the community vibe.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Deep-Learning-Profiling-Tools/triton-samples/blob/main/Triton_Tools_Tutorial.ipynb">triton-samples/Triton_Tools_Tutorial.ipynb at main · Deep-Learning-Profiling-Tools/triton-samples</a>: Contribute to Deep-Learning-Profiling-Tools/triton-samples development by creating an account on GitHub.</li><li><a href="https://github.com/triton-lang/triton/blob/main/docs/meetups/02-20-2024/Proton.pdf">triton/docs/meetups/02-20-2024/Proton.pdf at main · triton-lang/triton</a>: Development repository for the Triton language and compiler - triton-lang/triton
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1285509111160508447)** (2 messages): 

> - `Flash Attention v2 with learnable bias`
> - `BitBlas and Triton-like language` 


- **Exploring implementation of Flash Attention v2 with learnable bias**: A member inquired about implementing **Flash Attention v2** with a learnable bias of size **[B, H, N, N]** added before the softmax operation and required gradients.
   - *Any ideas on how to start approaching this problem?*
- **BitBlas authors create a promising Triton-like language**: The authors of **BitBlas** are developing a new **Triton-like language** based on **TVM**, which shows great potential. This could lead to significant advancements if successful, as highlighted in the [test_tilelang_dequantize_gemm.py](https://github.com/microsoft/BitBLAS/blob/main/testing/python/tilelang/test_tilelang_dequantize_gemm.py) example.
   - The **BitBlas** library focuses on supporting mixed-precision matrix multiplications, particularly for **quantized LLM deployment**.



**Link mentioned**: <a href="https://github.com/microsoft/BitBLAS/blob/main/testing/python/tilelang/test_tilelang_dequantize_gemm.py">BitBLAS/testing/python/tilelang/test_tilelang_dequantize_gemm.py at main · microsoft/BitBLAS</a>: BitBLAS is a library to support mixed-precision matrix multiplications, especially for quantized LLM deployment. - microsoft/BitBLAS

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1285552277444362302)** (1 messages): 

> - `SK Hynix AiMX-xPU`
> - `In-Memory Computing`
> - `LLM Inference`
> - `Power Efficiency` 


- **SK Hynix reveals AiMX-xPU at Hot Chips 2024**: During **Hot Chips 2024**, SK Hynix introduced the **AiMX-xPU** and **LPDDR-AiM**, showcasing their advancements in in-memory computing aimed at **LLM inference**.
   - The innovation allows data transformations to occur directly within memory, enhancing both **power efficiency** and speed by minimizing interconnect traversal.
- **In-Memory Computing aids memory-bound LLMs**: SK Hynix emphasized their commitment to supporting **LLMs** which are characterized as **memory-bound** due to their heavy reliance on memory access.
   - This focus aligns with their new computing solutions designed to streamline operations specifically for AI models.



**Link mentioned**: <a href="https://www.servethehome.com/sk-hynix-ai-specific-computing-memory-solution-aimx-xpu-at-hot-chips-2024/">SK Hynix AI-Specific Computing Memory Solution AiMX-xPU at Hot Chips 2024</a>: SK Hynix showed off its AiMX-xPU concept at Hot Chips 2024 for more efficient LLM inference compute being done in-memory

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1285326893717127192)** (2 messages): 

> - `Learning Custom CUDA Kernels`
> - `Neural Network Training` 


- **Beginner's Quest for Custom CUDA Kernel Knowledge**: A member expressed their ambition to learn and teach others about writing custom CUDA kernels over the next **six weeks**.
   - *Feeling like a neural network pro*, they shared their background but acknowledged their **beginner status** in CUDA development.
- **Encouragement from the Community**: Another member reacted positively to the initial query, showing support and willingness to help.
   - This interaction highlights the community's **welcoming** nature for newcomers.


  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1285627961055252522)** (3 messages): 

> - `Implementation using Metal or WebGPU`
> - `CUDA Alternatives`
> - `FAQs on GPU Programming`
> - `Metal Channel in Discord` 


- **Feasibility of Metal or WebGPU in PMPP**: .mattrix96 questioned whether it would be feasible to follow the book PMPP using **Metal** or **WebGPU** instead of **CUDA**, citing an absence of an Nvidia GPU.
   - This concern highlights the need for alternative approaches for GPU programming when hardware limitations exist.
- **Recommended Approach for Learning CUDA**: mr.osophy shared guidance from the FAQ stating that learners should ideally cover at least **Chapter 6** of PMPP to grasp foundational concepts in **CUDA**, as skills can transfer to other platforms.
   - The suggested strategy involves learning by doing, tackling challenges, and seeking assistance through appropriate Discord channels.
- **Discussion about the Metal Channel**: .mattrix96 expressed interest in checking out the **Metal channel** after receiving insights about learning alternatives.
   - They acknowledged the necessity of proper hardware and expressed willingness to find a solution if needed.


  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1285509089576751166)** (4 messages): 

> - `H100 purse`
> - `GH100 confusion`
> - `High pricing concerns` 


- **H100 purse priced absurdly**: A member pointed out an [H100 purse](https://gpupurse.com/products/h100-purse) listed at a staggering **$65,536.00 USD** with no apparent sale indication.
   - Another noted that it's questionable because **the item isn't even H100s**, indicating a potential scam.
- **Naive buyer skepticism**: *The fact that they just need one naive hyped guy to drop $65k on it* was shared, highlighting concerns about exploitation.
   - This sentiment reflects the broader skepticism among members regarding pricing strategies on supposedly high-value products.
- **GH100 branding confusion**: Discussion arose with a member revealing that the item actually says **GH100** on the silicone if you zoom in.
   - This suggests potential misrepresentation in the listing, further fueling distrust in the marketplace.



**Link mentioned**: <a href="https://gpupurse.com/products/h100-purse">H100 Purse</a>: Purse that has a rare one of a kind gpt-4 training gpu.    This purse is subject to export controls. 

  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1285316900548776070)** (37 messages🔥): 

> - `RMSNorm Implementation`
> - `FP8 Stability Issues`
> - `Consistency between Python and C/CUDA`
> - `Llama 3 Token Support`
> - `Dynamic Threadgroup Sizing` 


- **RMSNorm Implementation Progress**: A member is working on adding RMSNorm support, recently modifying the layer norm kernels accordingly and focusing on `rmsnorm_forward_kernel6` for review.
   - They observed ~1e-3 differences between Python and C/CUDA initially, but later discovered it was due to using bf16 precision instead of fp32.
- **FP8 End-to-End Functionality Restored**: The new tensor-based approach has successfully restored FP8 end-to-end capabilities for both forward and backward functionalities.
   - Future work will include cleaning the implementation, re-adding multi-GPU support, and testing performance convergence with prior approaches.
- **Consistency checks between Python and CUDA**: The Llama 3 branch is being tested across two terminals for consistency in activations using both Python and C/CUDA implementations.
   - Members configured their setups to ensure matching activations during the forward pass in their training processes.
- **Addressing Dynamic Threadgroup Allocation**: A member indicated that with the new changes, dynamic threadgroup sizing allows for easy adjustment if shared memory is exceeded.
   - As a result, they decided not to implement fallbacks for kernel memory limitations, relying on the dynamic sizing functionality.
- **Implementation of Llama 3 Tokens**: The dataloader has been updated to support Llama 3 tokens with the new dtype of uint32_t, moving away from the previous uint16_t.
   - Additionally, the RMSNorm forward has been added, matching outputs with Llama 3 Encoder forward while preparing for further adjustments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/757">RMSNorm - WIP by gordicaleksa · Pull Request #757 · karpathy/llm.c</a>: WIP - adding RMSNorm support.</li><li><a href="https://github.com/karpathy/llm.c/pull/757/files">RMSNorm - WIP by gordicaleksa · Pull Request #757 · karpathy/llm.c</a>: WIP - adding RMSNorm support.</li><li><a href="https://github.com/karpathy/llm.c/pull/754">add llama 3 support to llm.c by karpathy · Pull Request #754 · karpathy/llm.c</a>: This branch starts with a copy paste of train_gpt2.cu and test_gpt2.cu, but these two files (and other files) will change to incorporate Llama 3.1 support, before merging back to master.</li><li><a href="https://github.com/ademeure/llm.c/blob/llmc_reorg2/llmc/layernorm.cuh">llm.c/llmc/layernorm.cuh at llmc_reorg2 · ademeure/llm.c</a>: LLM training in simple, raw C/CUDA. Contribute to ademeure/llm.c development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1285314372352213054)** (15 messages🔥): 

> - `BitNet efficiency`
> - `In-memory computing from SK Hynix`
> - `Ternary packing methods`
> - `Custom silicon for neural networks`
> - `Lookup tables for packing` 


- **BitNet's packing strategy scrutinized**: Discussion revealed that packing **5 ternary values in an 8-bit space** is more efficient than a traditional 2-bit packing method, despite being complex to implement.
   - One member shared code to demonstrate the packing and unpacking process and considered avoiding modulo and division for optimized performance.
- **SK Hynix showcases in-memory computing**: At **Hot Chips 2024**, SK Hynix introduced advancements in **in-memory computing** for LLM inference, utilizing their AiMX-xPU and LPDDR-AiM technologies.
   - This method reduces power consumption and increases efficiency by performing computations directly in memory, which is crucial since LLMs are typically memory-bound.
- **Exploring the utility of Lookup Tables**: A member questioned the potential benefits of using **Lookup Tables (LUT)** to enhance the efficiency of the packing method discussed earlier.
   - The practicality of integrating LUT with packed values is under consideration, emphasizing the need for further examination.
- **Custom silicon development discussion**: Members discussed a new company, **Deepsilicon**, which focuses on building custom hardware and software for AI computations, claiming to operate with significantly less RAM.
   - Concerns about the viability of their ambitious goals were raised, highlighting the ongoing interest in innovative AI computing approaches.
- **Confusion around BitNet’s implementation**: Members debated the **2-bit implementation** used in the BitNet paper, questioning its effectiveness and the relevance for GPU runtime performance.
   - They acknowledged a need to look deeper into the paper's specifics regarding embedding, LM-head, and quantization strategies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.servethehome.com/sk-hynix-ai-specific-computing-memory-solution-aimx-xpu-at-hot-chips-2024/">SK Hynix AI-Specific Computing Memory Solution AiMX-xPU at Hot Chips 2024</a>: SK Hynix showed off its AiMX-xPU concept at Hot Chips 2024 for more efficient LLM inference compute being done in-memory</li><li><a href="https://www.deepsilicon.net">deepsilicon</a>: no description found</li><li><a href="https://x.com/sdianahu/status/1833186687369023550">Tweet from Diana (@sdianahu)</a>: Deepsilicon runs neural nets with 5x less RAM and ~20x faster. They are building SW and custom silicon for it.  What’s interesting is that they have proved it with SW, and you can even try it.    On w...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1285336181017411635)** (9 messages🔥): 

> - `Hack Ideas Discussion`
> - `Point Cloud Registration Kernel`
> - `Meetup at PyTorch Conference`
> - `Student Pricing Inquiry` 


- **Prioritize ideas for Hack Session**: Members are encouraged to review ideas in the hack-ideas thread and give a thumbs up to interesting projects, aiding in prioritizing where to focus efforts before the hack session.
   - This will help streamline the planning and align interests among participants.
- **New ideas for 3D Computer Vision**: A new member introduced a custom kernel idea for **point cloud registration (ICP)**, emphasizing its role in **3D computer vision** and expressing openness to collaborate on other projects.
   - *“My main goals are to learn and have fun.”* reflects their positive outlook.
- **Meetup Plans for CUDA HACK**: A member proposed a meetup for those attending both the **PyTorch Conference** and the **CUDA HACK**, suggesting forming a group to connect during the event.
   - Discussions included confirming attendance and building camaraderie among participants.
- **Student Pricing for PyTorch Conference**: A member questioned the possibility of obtaining **student pricing** for the PyTorch conference, sharing their situation regarding an educational email that remains active.
   - They plan to attend if pricing accommodates their budget as they explore their agenda for the hack session.
- **Squirrel Profile Pic Appreciation**: A lighthearted exchange about a user’s **squirrel profile pic** occurred, with compliments about its adorability and a fun response from the member.
   - *“I’ll pass that along to the squirrel.”* adds a humorous touch to the conversation.


  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1285358965269725235)** (1 messages): 

> - `Triton LayerNorm Issue`
> - `Tensor Parallelism and Training MoEs` 


- **Triton LayerNorm introduces inconsistencies**: A member reported an issue with Triton LayerNorm and RMSNorm implementations when using **tensor parallelism > 1**, stating that *'the parameter gradients are accumulated in a non-deterministic way'* leading to inconsistent results.
   - This issue has specifically affected their attempts to train **Mixture of Experts (MoEs)**, prompting inquiry into alternative implementations.
- **Seeking insights from Liger team on Triton**: The member is looking to check with the Liger team if they have tested the Triton implementations for their kernels in light of the aforementioned issue.
   - They tagged another member, suggesting that he may have a better understanding of this problem and its implications.


  

---


### **CUDA MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1285384858448691263)** (5 messages): 

> - `Metal Puzzles GitHub Repository`
> - `Live Puzzle Solving Session`
> - `Conferences` 


- **Explore Metal Puzzles on GitHub**: The [Metal Puzzles GitHub repository](https://github.com/abeleinin/Metal-Puzzles) encourages users to solve puzzles while learning about Metal programming.
   - The project aims to foster collaboration and knowledge sharing in the Metal community while having fun.
- **Proposal for Live Puzzle Solving Session**: A member proposed organizing a live puzzle solving session for next week, in light of a busy conference schedule.
   - Another member enthusiastically agreed, expressing excitement for the idea.
- **Newcomers Embrace Puzzles**: A newcomer mentioned starting their journey with puzzles, indicating growing interest in the community.
   - This reflects a positive trend as more members engage with puzzle-solving activities.



**Link mentioned**: <a href="https://github.com/abeleinin/Metal-Puzzles">GitHub - abeleinin/Metal-Puzzles: Solve Puzzles. Learn Metal 🤘</a>: Solve Puzzles. Learn Metal 🤘. Contribute to abeleinin/Metal-Puzzles development by creating an account on GitHub.

  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1285319549813657630)** (66 messages🔥🔥): 

> - `NousCon inquiries`
> - `AI Model Hermes 3 usage`
> - `InstantDrag development`
> - `CPLX extension for perplexity`
> - `Jailbreaking Claude 3.5` 


- **NousCon location details shared**: A member inquired about the location of NousCon, and another confirmed that details would be sent out that evening.
   - This led to discussions about potential future events, including ideas for locations like NYC.
- **Interest in using AI Model Hermes 3**: A new member expressed interest in utilizing the AI Model Hermes 3 and sought contact information for business inquiries.
   - Another user suggested reaching out to a specific member for further information.
- **Discussion on InstantDrag**: A user highlighted InstantDrag as a modern solution for drag-based image editing, improving interactivity and speed without needing masks or text prompts.
   - The discussion included comparing it to DragGAN and noted the potential for faster edits within applications.
- **CPLX extension for perplexity announced**: An alpha build of the CPLX extension for perplexity was introduced, featuring a scratchpad separate from the main output.
   - Further discussions revealed its integration with the 'scrapthpad-think' framework, showcasing new functionalities.
- **Jailbreak achieved for Claude 3.5 Sonnet**: A user proudly shared their success in creating a jailbreak for Claude 3.5 Sonnet, which was described as one of the harder models to breach.
   - They noted that while inspired by another's work, their approach was unique and functional.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Ffftdtd5dtft/Hermes-3-Llama-3.1-8B-IQ1_S-GGUF">Ffftdtd5dtft/Hermes-3-Llama-3.1-8B-IQ1_S-GGUF · Hugging Face</a>: no description found</li><li><a href="https://x.com/_akhaliq/status/1835677372344873377?t=Zkttn9BN3f0bv5lGZAfcZw&s=19">Tweet from AK (@_akhaliq)</a>: InstantDrag  Improving Interactivity in Drag-based Image Editing  discuss: https://huggingface.co/papers/2409.08857  Drag-based image editing has recently gained popularity for its interactivity and p...</li><li><a href="https://github.com/XingangPan/DragGAN">GitHub - XingangPan/DragGAN: Official Code for DragGAN (SIGGRAPH 2023)</a>: Official Code for DragGAN (SIGGRAPH 2023). Contribute to XingangPan/DragGAN development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1285377361587605514)** (18 messages🔥): 

> - `Parameter vs RAM Estimates`
> - `Model Training Data Efficiency`
> - `Scaling Parameters and Tokens`
> - `Optimal Compute Usage in LLMs`
> - `Llama Models Data Scaling` 


- **Parameter Estimates Misalignment**: A discussion emerged around the memory requirements for a **1B parameter model**, where estimates varied between **14GB** and **40GB** depending on the training context.
   - This discrepancy prompted inquiries about the implications for different types of training, highlighting that **all parameters influence memory needs**.
- **Parameters and Data Correlation Clarified**: Members debated whether parameters are a direct proxy for the amount of training data, concluding that **more parameters allow for better pattern extraction** but are not strictly correlated.
   - This led to the consensus that while they scale together, they do not need to, emphasizing the complexities of model training.
- **Independent Control of Parameters and Tokens**: It was noted that you can control parameters and tokens independently, with a suggestion to scale them in a **1:1 ratio** for optimal compute use.
   - However, members pointed out that models like **Llama** often train on far more data than their parameter counts might imply.
- **Detailed Memory Requirement Calculation**: For calculating memory requirements, the formula suggested is to multiply the number of parameters by various factors for precision and optimizer kinds, yielding a rough count of required RAM.
   - The final memory estimate needs further adjustment based on **activation requirements**, which can vary significantly.
- **Challenges with Large Answers in Open Source LLMs**: A member asked about strategies to **send and receive big prompts** using open source models like **gpt4all**.
   - This reflects ongoing community interest in optimizing interactions with large language models for better performance.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1285326215414026330)** (3 messages): 

> - `Scaling LLM Inference`
> - `Chunking Phases in Research` 


- **Transformers' Performance Limit with Scaling**: In a recent [tweet](https://x.com/denny_zhou/status/1835761801453306089?s=46&t=VBhI-dqaQfawcUDHNO0L9A), it was highlighted that mathematically, transformers can solve any problem if allowed to generate as many **intermediate reasoning tokens** as needed, asserting that **constant depth is sufficient**.
   - This insight was tied to a forthcoming paper, detailed in [this arXiv link](http://arxiv.org/abs/2402.12875), set to be presented at **ICLR 2024**.
- **Inquiry on Chunking Phases Research**: There was a request for **top-tier** and **latest research papers** related to *chunking phases* and *approximation* techniques.
   - This reflects an ongoing interest in understanding current methodologies in this area of study.



**Link mentioned**: <a href="https://x.com/denny_zhou/status/1835761801453306089?s=46&t=VBhI-dqaQfawcUDHNO0L9A">Tweet from Denny Zhou (@denny_zhou)</a>: What is the performance limit when scaling LLM inference? Sky&#39;s the limit.  We have mathematically proven that transformers can solve any problem, provided they are allowed to generate as many int...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1285321676741148674)** (3 messages): 

> - `ChatGPT o1-preview`
> - `RL in development environments`
> - `iText2KG and SeekTopic Algorithm`
> - `LLMs generating research ideas` 


- **ChatGPT o1-preview shows off coding prowess**: @realGeorgeHotz declared that **ChatGPT o1-preview** is the first model capable of programming, estimating its intelligence at **120 IQ**.
   - He expressed strong optimism about **reinforcement learning** in development, particularly in writing and testing code, sharing a link where ChatGPT writes **tinygrad tests**: [link](https://chatgpt.com/share/66e693ef-1a50-8000-81ff-899498f9d052).
- **iText2KG development discussion**: A member shared a conversation with the **iText2KG** developer about adding a **SeekTopic algorithm** for edge extraction, indicating forward movement in development.
   - The interest in this approach was confirmed by another member who considered it a very promising direction for the research.
- **LLMs enhance research generation**: Research highlighted that **LLMs** can generate better research ideas and plans according to the paper found at [arXiv](https://arxiv.org/html/2409.04109).
   - This insight supports the growing perception of LLMs’ capabilities in contributing to research and innovation.



**Link mentioned**: <a href="https://x.com/realGeorgeHotz/status/1835228364837470398">Tweet from George Hotz 🌑 (@realGeorgeHotz)</a>: ChatGPT o1-preview is the first model that&#39;s capable of programming (at all). Saw an estimate of 120 IQ, feels about right.  Very bullish on RL in development environments. Write code, write tests...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1285326215414026330)** (3 messages): 

> - `Scaling LLM inference limits`
> - `Chunking phases in research`
> - `Transformer capabilities` 


- **LLM Inference Performance Limit Explored**: A recent [tweet from Denny Zhou](https://x.com/denny_zhou/status/1835761801453306089?s=46&t=VBhI-dqaQfawcUDHNO0L9A) claims that transformers can solve any problem given enough intermediate reasoning tokens, asserting that constant depth is sufficient.
   - This mathematical proof was featured in a paper titled *What is the performance limit when scaling LLM inference?* accepted at ICLR 2024, which can be viewed [here](http://arxiv.org/abs/2402.12875).
- **Request for Research on Chunking Phases**: A user inquired about recent research papers focused on **chunking phases** and approximations, seeking top-tier and latest studies.
   - They specifically asked for any relevant work that could advance understanding in this area.



**Link mentioned**: <a href="https://x.com/denny_zhou/status/1835761801453306089?s=46&t=VBhI-dqaQfawcUDHNO0L9A">Tweet from Denny Zhou (@denny_zhou)</a>: What is the performance limit when scaling LLM inference? Sky&#39;s the limit.  We have mathematically proven that transformers can solve any problem, provided they are allowed to generate as many int...

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1285321251430207568)** (77 messages🔥🔥): 

> - `Dream Machine API`
> - `11x AI Series A funding`
> - `Impact of AI on jobs`
> - `Claude 3.5 system prompt`
> - `ZIG based inference stack` 


- **Dream Machine API Launch**: Luma Labs announced the launch of the [Dream Machine API](https://lumalabs.ai/dream-machine/api) allowing developers to create using a leading video generation model without extensive tooling.
   - *Get started today* to explore simplified video creation capabilities.
- **11x AI Secures $24m Series A**: 11x AI, co-founded by Alice and Jordan, has raised a **$24m Series A** from **Benchmark** and others, increasing its ARR by **15x** this year and supporting over **250 customers**.
   - The team plans to develop **LLM-powered systems** for digital workers aimed at redefining modern GTM functions.
- **AI's Growing Impact on Jobs**: A report estimates that **60 million jobs** in the US and Mexico will be affected by AI within the next year, with future projections rising to **70 million** in the US and **26 million** in Mexico over ten years.
   - While not all job changes will result in losses, a significant number of occupations remain vulnerable, highlighting the urgent need for adaptation.
- **Claude 3.5 System Prompt Shared**: A user shared the **Claude 3.5 Projects + Artifacts system prompt** in a gist, which is relevant for those exploring AI applications.
   - This prompt has now been discussed in various Discord channels, reflecting its significance in current AI evaluations.
- **ZIG Based Inference Stack by Yann LeCun**: A new **ZIG based inference stack**, backed by **Yann LeCun**, has been revealed, aimed at providing high-performance AI inference capable of running deep learning systems on various hardware.
   - This project, open-sourced and now out of stealth, showcases advancements in AI performance capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/11x_official/status/1835711787712582082?s=46">Tweet from 11x (@11x_official)</a>: 👋🏻 Hey from Alice & Jordan - We just raised a $24m Series A from @benchmark!  Read our full blog post here: https://www.11x.ai/blog/series-a  Some highlights so far this year: - Increased our ARR by...</li><li><a href="https://www.amazon.com/Decline-American-Programmer-Edward-Yourdon/dp/013191958X">no title found</a>: no description found</li><li><a href="https://www.amazon.com/Resurrection-American-Programmer-Yourdon-Computing/dp/013121831X/ref=">no title found</a>: no description found</li><li><a href="https://www.amazon.com/Resurrection-American-Programmer-Yourdon-Computing/dp/013121831X/ref=sw_img_mw_ace_sim?_encoding=UTF8&pd_rd_i=013121831X&pd_rd_w=hIDWd&content-id=amzn1.sym.5a573fc7-d6aa-4c57-a9db-8a2220925981&pf_rd_p=5a573fc7-d6aa-4c57-a9db-8a2220925981&pf_rd_r=T08GE2EA6ZR6NJNC71CV&pd_rd_wg=jhWGS&pd_rd_r=c962c739-02de-4045-afb9-cdd0bacb017f">no title found</a>: no description found</li><li><a href="https://x.com/lmsysorg/status/1835825082280902829">Tweet from lmsys.org (@lmsysorg)</a>: Chatbot Arena update🔥  We&#39;ve been testing the latest ChatGPT-4o (20240903) over the past 2 weeks, and the results show significant improvements across the board:  - Overall: 1316 -&gt; 1336 - Ove...</li><li><a href="https://x.com/maartengr/status/1835709176703508688?s=46">Tweet from Maarten Grootendorst (@MaartenGr)</a>: I&#39;m thrilled to announce the release of the digital version of &#34;Hands-On Large Language Models&#34; 🎉  The book contains more than 250 visuals (in color!) to help you understand the inner wor...</li><li><a href="https://www.arcads.ai/">Arcads - Create AI Video Ads</a>: AI UGC Made Easy: Write your script, pick your actors and generate your UGC video in 2min.</li><li><a href="https://x.com/sullyomarr/status/1836059834543734892?s=46">Tweet from Sully (@SullyOmarr)</a>: Finally excited to ship Otto!!  Otto lets you use AI agents within tables to automate hours manual research in a few minutes  Here&#39;s my quick breakdown of how to use it, with some real world use c...</li><li><a href="https://x.com/lumalabsai/status/1835742651662139529?s=46">Tweet from Luma AI (@LumaLabsAI)</a>: 🚀 Introducing the Dream Machine API. Developers can now build and scale creative products with the world&#39;s most popular and intuitive video generation model without building complex tools in thei...</li><li><a href="https://x.com/zml_ai/status/1835973073385685099?s">Tweet from ZML (@zml_ai)</a>: https://github.com/zml/zml</li><li><a href="https://x.com/zml_ai/status/1835973073385685099?s=46">Tweet from ZML (@zml_ai)</a>: https://github.com/zml/zml</li><li><a href="https://x.com/ylecun/status/1836030233796874244?s=46">Tweet from Yann LeCun (@ylecun)</a>: ZML: a  high-performance AI inference stack that can parallelize and run deep learning systems on lots of different hardware. It&#39;s out of stealth, impressive, and open source.  Quoting ZML (@zml_a...</li><li><a href="https://english.elpais.com/economy-and-business/2024-09-15/artificial-intelligence-will-affect-60-million-us-and-mexican-jobs-within-the-year.html#">Artificial intelligence will affect 60 million US and Mexican jobs within the year</a>: IDB study shows the impact that AI will have on the labor market. Women and low-skilled workers are more vulnerable to being replaced</li><li><a href="https://www.patreon.com/posts/super-panavision-109117838?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_fan&utm_content=web_share">Super Panavision 70 Tutorial | Abandoned Films</a>: Get more from Abandoned Films on Patreon</li><li><a href="https://gist.github.com/njpearman/ffdc8768dc37451bf2c8d5f93b6a905d">Extracted Claude 3.5 Sonnet system prompt for artifacts</a>: Extracted Claude 3.5 Sonnet system prompt for artifacts - claude_35_artifacts_system_prompt.txt</li><li><a href="https://www.when2meet.com/?26526365-sTYbJ">Evals Group - When2meet</a>: no description found
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1285358690504802314)** (10 messages🔥): 

> - `Foundation Models in Biotech`
> - `AI Safety Transition`
> - `TensorRT-LLM Issues`
> - `Transformer Model Memory Profiling`
> - `Photo Upgrade Tools` 


- **Foundation Models in Biotech shared**: One member introduced their work on **foundation models** in **biotech**, focusing on **large scale representation learning** for sequence and tabular data.
   - This expertise highlights the growing intersection of AI technologies and biotechnological applications.
- **AI Safety career transition**: A member announced that they received an **Open Philanthropy career transition fellowship** to shift focus to **AI safety** after 9 years in applied AI/ML model development.
   - Their initial interests lie in **interpretability** and broader **alignment** discussions related to AI.
- **Building issues with TensorRT-LLM**: A user inquired about difficulties experienced while building a model with **TensorRT-LLM** on a **T4 video card compatible with CUDA**.
   - This raises a concern for those utilizing specific hardware setups in their model development workflows.
- **Debugging Transformer Model Memory**: One member is investigating memory requirements while training small transformers, noting a minimum of **26 bytes per param** needed despite their calculations showing only **16**.
   - They are exploring **pytorch profiling tools** but find the stack traces opaque due to **FSDP and mixed precision** layers complicating profiling.
- **Suggestions for photo enhancement tools**: In response to a request for tools to '**upgrade**' poorly taken portraits, a member suggested using **Krita** with a **stable diffusion plugin**.
   - The inquiry highlights a need for open-source solutions to enhance photography while preserving likeness.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1285388437607682119)** (8 messages🔥): 

> - `AI Safety Fellowship`
> - `Token Embedding Variability`
> - `Multi-head Low-Rank Attention`
> - `Diagram of Thought`
> - `Hyper-graphs` 


- **Exploration of AI Safety and Research Help**: A member shared excitement about transitioning into **AI Safety** after receiving an Open Philanthropy fellowship, expressing interest in **interpretability** and alignment research.
   - They encouraged sharing ongoing research projects where they could volunteer their extensive experience, aiming to contribute within the next **six months**.
- **Token Embedding Variability as Stability Proxy**: A recent paper introduced **Token Embedding Variability (TEV)** as an efficient method to assess pre-training stability in language models, while also proposing **Multi-head Low-Rank Attention (MLRA)** to prevent gradient explosion.
   - Empirical results on **GPT-2** showed improved stability and lower perplexity, particularly in deeper networks, though some questioned the details of the experimental setup.
- **Diagram of Thought for Iterative Reasoning**: Introduction of the **Diagram of Thought (DoT)** framework models reasoning in **large language models** as a directed acyclic graph (DAG), enhancing logical consistency through complex pathways.
   - This model allows for better iterative improvements in reasoning by utilizing **natural language feedback** and auto-regressive token prediction.
- **Call for Hyper-graphs in Reasoning Models**: A participant suggested the need for **hyper-graphs** in reasoning models, hinting at potential enhancements to the search-in-the-chain approach presented in recent literature.
   - Despite some doubts about the impact, they acknowledged the intriguing nature of exploring hyper-graph structures in this context.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.07146">Gated Slot Attention for Efficient Linear-Time Sequence Modeling</a>: Linear attention Transformers and their gated variants, celebrated for enabling parallel training and efficient recurrent inference, still fall short in recall-intensive tasks compared to traditional ...</li><li><a href="https://arxiv.org/abs/2409.07787">Stable Language Model Pre-training by Reducing Embedding Variability</a>: Stable pre-training is essential for achieving better-performing language models. However, tracking pre-training stability by calculating gradient variance at every step is impractical due to the sign...</li><li><a href="https://arxiv.org/abs/2409.10038">On the Diagram of Thought</a>: We introduce Diagram of Thought (DoT), a framework that models iterative reasoning in large language models (LLMs) as the construction of a directed acyclic graph (DAG) within a single model. Unlike t...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1285703238879805513)** (9 messages🔥): 

> - `Fourier Transforms of Hidden States`
> - `Power Law in Hidden States`
> - `Pythia Checkpoints Exploration`
> - `Untrained Model Behavior`
> - `Attention Residuals Analysis` 


- **Exploring Fourier Transforms of Hidden States**: Discussion initiated on interpreting the [Fourier transforms](https://sander.ai/2024/09/02/spectral-autoregression.html) of hidden states like Sander Dieleman did in his blog post. Key findings indicate that hidden states start uniformly and evolve to a power law over layer depth.
   - Questions raised about whether the attention mechanism induces this power spectrum power law in final hidden states.
- **Pythia Checkpoints Recommended**: A member recommended using the [Pythia suite](https://github.com/EleutherAI/pythia) to investigate scale and training effects on observed phenomena. This model behavior may be impacted by architectural or initialization factors.
   - Exploring different models with varied architectures is suggested to confirm these observations.
- **Clarification on Hidden States and Attention Residuals**: A clarification was made that the hidden states plot derived from the pretrained OPT-125m while attention residuals came from an untrained one. This indicates differing spectral properties based on training status.
   - It was noted that the hidden states properties evolve through training, suggesting the potential for training to bias these outcomes.
- **Attention Residuals and MLP Analysis**: Attention residuals from the pretrained model revealed significant spikes that align with layer structure, indicating spectral behavior. The analysis of MLP residuals complements this understanding with significant observations of early layer behavior.
   - The contrasting behavior of the freshly initialized model vs. pretrained models raises further questions about training and representation efficiency.
- **Next Steps Focused on Pythia Utilization**: The conversation closed with a consensus to explore Pythia for a deeper understanding of model training effects. This aligns with the aim to quantify changes over the course of model training.



**Link mentioned**: <a href="https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)">interpreting GPT: the logit lens — LessWrong</a>: This post relates an observation I&#x27;ve made in my work with GPT-2, which I have not seen made elsewhere. …

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1285523748090286090)** (37 messages🔥): 

> - `Issue with LM Evaluation Harness`
> - `Integration of Torchtune`
> - `TensorRT-LLM Build Errors`
> - `Deployment of Datasets on Hugging Face`
> - `Chain of Thought Prompting` 


- **Pointers needed for LM Evaluation Harness issue**: A user sought assistance regarding an [issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/2305) with the LM Evaluation Harness.
   - Another user noted they could help by potentially adding the dataset to HF.
- **Torchtune integrates with LM Evaluation Harness**: A maintainer of Torchtune mentioned their integration with the eval harness and proposed enhancements for generation tasks using static KV-caching.
   - They suggested finding the largest divisible batch size to avoid errors with cache setups.
- **TensorRT-LLM Building Troubles**: A user raised concerns over building a TRT-LLM engine, citing an error related to workspace size.
   - Suggestions included increasing workspace size using IBuilderConfig::setMemoryPoolLimit().
- **Dataset Deployment on Hugging Face**: A user confirmed they added the dataset to [HF](https://huggingface.co/datasets/baber/multilingual_mmlu) to assist in evaluations.
   - Further clarifications on the dataset's structure were requested to optimize splits and subsets.
- **Chain of Thought Prompting Query**: A user inquired about experiences with chain of thought prompting using the LM Evaluation Harness.
   - They were interested in appending model answers to subsequent prompts and recording the results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/2305)">Issues · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - Issues · EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/pytorch/torchtune/blob/60a7e3dae0d43e841c5e0ee4c1622fc9b1c2c4ca/recipes/eleuther_eval.py#L37">torchtune/recipes/eleuther_eval.py at 60a7e3dae0d43e841c5e0ee4c1622fc9b1c2c4ca · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/fb963f0f0a5b28b69763590bb59676072cf43a01/lm_eval/models/huggingface.py#L1236)">lm-evaluation-harness/lm_eval/models/huggingface.py at fb963f0f0a5b28b69763590bb59676072cf43a01 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/fb963f0f0a5b28b69763590bb59676072cf43a01/lm_eval/models/huggingface.py#L1295)">lm-evaluation-harness/lm_eval/models/huggingface.py at fb963f0f0a5b28b69763590bb59676072cf43a01 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/fb963f0f0a5b28b69763590bb59676072cf43a01/lm_eval/evaluator.py#L439)">lm-evaluation-harness/lm_eval/evaluator.py at fb963f0f0a5b28b69763590bb59676072cf43a01 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/fb963f0f0a5b28b69763590bb59676072cf43a01/lm_eval/evaluator_utils.py#L203)">lm-evaluation-harness/lm_eval/evaluator_utils.py at fb963f0f0a5b28b69763590bb59676072cf43a01 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1285680433249390594)** (1 messages): 

> - `Model Outputs`
> - `Library Utilization` 


- **Community Excitement for Model Outputs**: Members expressed enthusiasm for showcasing their models and outputs created using the libraries, indicating a supportive community atmosphere.
   - *Definitely let us know* when you have models or other outputs, as the community loves to hear about the great things people get up to using the resources.
- **Library Engagement**: There was a strong indication that community members value updates related to their implementation of the libraries, emphasizing collaboration and sharing.
   - *We love to hear about the great things people get up to* using our libraries, highlighting the importance of community contribution.


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1285317374853251083)** (53 messages🔥): 

> - `SSH Connection Issues`
> - `Stable Diffusion Installation Errors`
> - `ComfyUI White Screen`
> - `Control Net Training Challenges`
> - `CivitAI Bounty Offer` 


- **SSH Connection Issues with Deployed Pods**: A member reported difficulties with SSH connections to already deployed pods after updating their SSH key.
   - They inquired whether any configuration changes would allow them to connect via SSH.
- **Stable Diffusion Installation Troubles**: A member encountered a 'Stable Diffusion model failed to load' error despite running the setup script as directed.
   - Other members suggested following installation guides and posting detailed error logs for technical support.
- **ComfyUI Black Screen After Update**: A user updated ComfyUI and faced a white screen issue when trying to load the GUI.
   - One member advised to fully unload ComfyUI and run the update script again before rebooting the system.
- **Challenges in Control Net Training**: Discussion emerged around the necessity of a substantial dataset for effective training with Control Net.
   - Members suggested considering novel dataset augmentations and workflows to achieve desired outcomes.
- **CivitAI Bounty Pack Inquiry**: A member is looking to post a CivitAI bounty for creating a 49 character pack with an approximate image count of 4000.
   - They are seeking advice on an appropriate Buzz amount to offer for the bounty.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Gui">Home</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui Installation Guides</a>: Stable Diffusion Knowledge Base (Setups, Basics, Guides and more) - CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/fairy-root/Flux-Prompt-Generator?tab=readme-ov-file">GitHub - fairy-root/Flux-Prompt-Generator: Flux Prompt Generator provides a flexible and customizable prompt generator for generating detailed and creative prompts for image generation models.</a>: Flux Prompt Generator provides a flexible and customizable prompt generator for generating detailed and creative prompts for image generation models. - fairy-root/Flux-Prompt-Generator
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1285373740410081362)** (2 messages): 

> - `Multimodal RAG techniques`
> - `LlamaCloud launch`
> - `Product manual challenges` 


- **Tackling Product Manuals with Multimodal RAG**: Product manuals often present a challenge for RAG due to their visual-centric nature, lacking text and primarily featuring **step-by-step visuals** and diagrams.
   - To enable LLMs to navigate these manuals effectively, a sophisticated [indexing pipeline](https://t.co/GOedcAdLqF) is required.
- **LlamaCloud Unveils Multimodal Capabilities**: Today’s launch of **multimodal capabilities in LlamaCloud** allows users to create end-to-end multimodal RAG pipelines quickly across a variety of unstructured data formats.
   - This toolkit supports diverse applications, including **marketing slide decks**, legal and insurance contracts, and finance reports - simplifying the workflow for users ([details here](https://t.co/43eL8zvm7H)).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1285429789653733446)** (36 messages🔥): 

> - `LlamaIndex and Neo4j Integration`
> - `Embedding Retrieval from Neo4j`
> - `Circular Dependency in LlamaIndex Packages`
> - `GraphRAG Implementation`
> - `Image Coordinate Extraction with GPT-4o` 


- **Integration of LlamaIndex with Neo4j for Embeddings**: A user inquired about retrieving embeddings stored as node properties in Neo4j using LlamaIndex. Others suggested connecting Neo4j with LlamaIndex's property graph index for effective querying.
   - It was noted that once the node is retrieved, parsing its properties to obtain embeddings should be straightforward.
- **Circular Dependency Detected in LlamaIndex Packages**: A circular dependency was discovered between the packages `llama-index-agent-openai` and `llama-index-llms-openai` during a Bazel Dependency Graph update. Members discussed potential solutions, including the creation of an `openai-utils` package to resolve the issue.
   - Questions were raised about the timeline for resolving this dependency, with suggestions for community contributions to expedite fixes.
- **Evaluating Transition to GraphRAG**: A member expressed interest in transitioning from basic RAG to GraphRAG and sought advice on whether to use LlamaIndex's abstractions or Microsoft's package. Recommendations for constructing an evaluation set to compare approaches were also requested.
   - Several members showed interest in sharing insights on the best practices for this transition.
- **Challenges in Extracting Image Coordinates with GPT-4o**: A user described difficulties aligning labels and extracting accurate coordinates from images using GPT-4o. They requested suggestions on improving their grid overlay method to ensure accurate spatial recognition.
   - The goal was to produce precise coordinates for cropping images based on detected entities, and feedback from the community was welcomed.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/Neo4jKGIndexDemo/">Neo4j Graph Store - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_neo4j/">Neo4j Property Graph Index - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/#texttocypherretriever">Property Graph Index - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/blob/7e557890adac0f505be29d06a1eff60fc7dc629b/llama-index-integrations/agent/llama-index-agent-openai/pyproject.toml#L35)">llama_index/llama-index-integrations/agent/llama-index-agent-openai/pyproject.toml at 7e557890adac0f505be29d06a1eff60fc7dc629b · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/7e557890adac0f505be29d06a1eff60fc7dc629b/llama-index-integrations/llms/llama-index-llms-openai/pyproject.toml#L38)">llama_index/llama-index-integrations/llms/llama-index-llms-openai/pyproject.toml at 7e557890adac0f505be29d06a1eff60fc7dc629b · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index
</li>
</ul>

</div>
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1285644658353111070)** (4 messages): 

> - `Mistral's September Release`
> - `Free Tier on La Plateforme`
> - `Pricing Update`
> - `Mistral Small Improvements`
> - `Vision Capabilities with Pixtral 12B` 


- **Mistral Launches New Features**: Mistral has launched several features including a [free tier on La Plateforme](https://mistral.ai/news/september-24-release/) for developers to experiment with API endpoints at no cost.
   - The new updates also bring **reduced prices** across the board and improvements to **Mistral Small**.
- **Mistral's Free Tier Sparks Reactions**: *“Free? lol”* was the reaction from a member regarding the newly announced free tier, indicating skepticism about its implications.
   - This sentiment reflects a broader discussion in the community about the viability of **VCs desperate** for better user engagement.
- **User Data Request for Insights**: Another member urged, *“pls guys give us user data too,”* pointing to a craving for more insights and specifics from the community regarding user engagement.
   - This highlights ongoing interest in understanding user behavior as part of the evaluation of Mistral's new offerings.



**Link mentioned**: <a href="https://mistral.ai/news/september-24-release/">AI in abundance</a>: Introducing a free API, improved pricing across the board, a new enterprise-grade Mistral Small, and free vision capabilities on le Chat.

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1285645734540218480)** (12 messages🔥): 

> - `Intermediate Generation in Transformers`
> - `Visualizing Attention Matrices`
> - `Alpha Code Website Feature`
> - `Attention Rollout Paper`
> - `Gradient-Based Token Associations` 


- **Intermediate Generation boosts Transformer capabilities**: Research identifies that allowing transformers to utilize a 'chain of thought' or 'scratchpad' significantly enhances their computational power, depending on the amount of intermediate generation.
   - This advancement shows potential for solving reasoning problems that standard transformers struggle with, with implications for utilizing intermediate steps effectively.
- **Best practices for visualizing attention matrices**: A member inquired about the best practices for visualizing attention matrices in a QA setting to demonstrate associations between questions and supporting facts.
   - Suggestions included exploring various techniques to represent the strength of connections visually, which could clarify how answers are derived.
- **Alpha Code's interactive transparency feature**: Discussion highlighted the Alpha Code website's feature where hovering over tokens reveals the most attended tokens from the prompt, color-coded by association strength.
   - This interactive approach could enhance user understanding of attentional relationships in generated responses.
- **Attention Rollout as a reference for 'most attended'**: A suggestion was made to refer to the attention rollout paper for various definitions of 'most attended', to gain insights into implementations.
   - This could provide solid foundational practices for measuring attention effectively in transformer models.
- **Gradient analysis for output token association**: Exploring output token probabilities’ gradients with respect to input tokens was discussed as an alternative approach for understanding token associations, albeit complex to implement.
   - This method could be computationally intensive, necessitating custom coding to handle its O(n^2) time complexity.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://poloclub.github.io/transformer-explainer/">Transformer Explainer: LLM Transformer Model Visually Explained</a>: An interactive visualization tool showing you how transformer models work in large language models (LLM) like GPT.</li><li><a href="https://arxiv.org/abs/2402.12875">Chain of Thought Empowers Transformers to Solve Inherently Serial Problems</a>: Instructing the model to generate a sequence of intermediate steps, a.k.a., a chain of thought (CoT), is a highly effective method to improve the accuracy of large language models (LLMs) on arithmetic...</li><li><a href="https://arxiv.org/abs/2310.07923">The Expressive Power of Transformers with Chain of Thought</a>: Recent theoretical work has identified surprisingly simple reasoning problems, such as checking if two nodes in a graph are connected or simulating finite-state machines, that are provably unsolvable ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1285448997787402270)** (15 messages🔥): 

> - `Gemini Models`
> - `NotebookLM Tweet`
> - `Podcast with Riley`
> - `Guest Lecture on LLMs` 


- **Unreleased Gemini Models Galore**: Exciting news surfaced about unreleased **Gemini models**, including **potter-v1** and **dumbledore-v1** in vision arena, as well as **gemini-test**, a **Gemini 1.5 pro refresh**.
   - Other models mentioned include **zeus-flare-thunder** (v1 and v2) and **qwen2.5-72b-instruct**, indicating a robust upcoming lineup.
- **NotebookLM Sparks Interest**: A random tweet about **NotebookLM** unexpectedly gained traction, leading to private messages from two individuals at **Google**.
   - The excitement is palpable, as this engagement has opened up new opportunities for discussions and potentially more connections.
- **Podcast with Riley Yields Fun Insights**: A podcast session with **Riley** was described as enjoyable, with praise for his engaging personality and insights shared throughout.
   - Listeners are encouraged to join in discussions, emphasizing that the community atmosphere is much more enjoyable compared to other platforms like **Twitter**.
- **First Guest Lecture on LLMs at McGill**: A member celebrated their first guest lecture as an adjunct professor at **McGill** on **LLMs**, sharing their slides as a valuable resource.
   - They aim to assist others with the material, showcasing a proactive approach to knowledge sharing in the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/phill__1/status/1835989093093617739?s=46">Tweet from Phil (@phill__1)</a>: lmarena is pretty crazy currently. Unreleased models: -  potter-v1 and -v2 and dumbledore-v1 and -v2 (only in vision arena) - zeus-flare-thunder-v1 and -v2 - sharp-game-player-v1 and -v2 - qwen2.5-72b...</li><li><a href="https://x.com/agarwl_/status/1836119825216602548?s=46">Tweet from Rishabh Agarwal (@agarwl_)</a>: I gave my first guest lecture today in a grad course on LLMs as an (soon-to-be) adjunct prof at McGill. Putting the slides here, maybe useful to some folks ;)  https://drive.google.com/file/d/1komQ7s9...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1285396690194403399)** (2 messages): 

> - `AI developers skipping Google's Gemini`
> - `Humorous AI article` 


- **AI Developers Laugh Off Google's Gemini**: A member pointed out an article discussing why **AI developers are skipping Google's Gemini** published on [The Information](https://www.theinformation.com/articles/why-ai-developers-are-skipping-googles-gemini).
   - *This article's featured image made me chortle.*
- **Article Sparks Humor Among Members**: The same article elicited laughter among members regarding the **humorous aspect** of the content. 
   - One remarked specifically about the article's featured image, calling it *hilarious*.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1285388361686716457)** (3 messages): 

> - `Newsletter Reader Party`
> - `Mainstream Media Critique` 


- **Celebrating Readers at the Newsletter Party**: A member announced 'the great newsletter reader party,' inviting people to join and enjoy reading together.
   - This event aims to foster community and engagement among newsletter enthusiasts.
- **Criticism of Mainstream Media Consumption**: A discussion point emerged about the downsides of solely relying on mainstream media for news.
   - The sentiment expressed highlights a growing desire for alternative sources and more diverse content.


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1285340237228150845)** (15 messages🔥): 

> - `Chat Message History Management`
> - `UI Messages Storage`
> - `Open Source Aspirations`
> - `Migrating to LLMChain`
> - `Implementing AI in Business` 


- **Clarifying Chat Message History Management**: A member sought clarity on managing chat history in **LangChain** and noted complexities with storing additional UI data alongside message history, which is handled separately in the **PostgresChatMessageHistory**.
   - Others confirmed the need to store UI-specific messages in a distinct table, as the existing systems do not support combined transactions.
- **Developing Open Source Contributions**: A member expressed aspirations to become a significant contributor in the open-source space, wishing to work on impactful projects while maintaining independence through sponsorship.
   - They sought insights from the community about pathways to achieve this ambitious goal.
- **Migrating to New LLMChain Implementations**: A member advised transitioning from legacy **LLMChain** to newer implementations due to clearer parameters and enhanced streaming capabilities.
   - Several advantages were highlighted, including easier access to raw message outputs, underscoring the benefits of staying updated.
- **Integrating AI Solutions in Local Businesses**: One member expressed confidence in selling AI solutions to local businesses and sought advice on successful implementation strategies and thriving markets.
   - They were particularly interested in tips for engaging business owners who may have limited familiarity with AI technology.
- **Speeding Up Structured Responses**: A member inquired about methods to expedite structured responses in **LangChain**, specifically using **Pydantic** models with a **JSONOutputParser**.
   - This inquiry reflects ongoing challenges faced by developers in improving response efficiency.



**Link mentioned**: <a href="https://python.langchain.com/docs/versions/migrating_chains/llm_chain/">Migrating from LLMChain | 🦜️🔗 LangChain</a>: LLMChain combined a prompt template, LLM, and output parser into a class.

  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/)** (1 messages): 

taixian0420: please dm me
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1285493431992324168)** (6 messages): 

> - `RAG Chatbot`
> - `AdaletGPT` 


- **AdaletGPT Launches RAG Chatbot**: A backend developer at **adaletgpt.com** has successfully built a **RAG chatbot** using OpenAI and Langchain.
   - You can check it out at their website: [adaletgpt.com](https://adaletgpt.com).
- **Open Invitation for Questions**: The developer encouraged community members to reach out directly via DM for any additional questions related to the chatbot.
   - *I will do my best for you* was their assurance to provide support.



**Link mentioned**: <a href="https://adaletgpt.com">no title found</a>: no description found

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1285493749467713566)** (1 messages): 

> - `RAG Chatbot`
> - `OpenAI Integration`
> - `LangChain Framework` 


- **Backend Developer Launches RAG Chatbot**: A backend developer at [adaletgpt.com](https://adaletgpt.com/) shared their new RAG chatbot built using **OpenAI** and **LangChain**.
   - They encouraged others to reach out via DM for any questions, stating they will do their best to assist.
- **Invitation for Questions**: The developer invited the community to contact them for further inquiries regarding the RAG chatbot.
   - They expressed willingness to support users by stating, *I will do my best for you.*



**Link mentioned**: <a href="https://adaletgpt.com/">no title found</a>: no description found

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1285338481559601242)** (10 messages🔥): 

> - `tinygrad version bump`
> - `ROCm compatibility`
> - `CRIU feature in AMDKFD`
> - `pytest filtering`
> - `testing unnecessary files` 


- **Tinygrad bump to 0.9.2 fails on AMD**: A user encountered an **AttributeError** related to `struct_kfd_ioctl_criu_args` while trying to bump **tinygrad** from **0.9.0 to 0.9.2** in **nixpkgs** on AMD.
   - They questioned if this issue was caused by a wrong kernel version since the struct is present in **/usr/include/linux/kfd_ioctl.h**.
- **CRIU support is a recent addition**: It was noted by community members that **CRIU** support in the **amdgpu driver** is a relatively new feature.
   - This might indicate compatibility concerns when bumping versions of **tinygrad** or related libraries.
- **Misidentified tests in extra/ directory**: Users pointed out that failures are arising from **extra/hip_gpu_driver/test_kfd_2.py**, which shouldn't run in actual tests.
   - Another failure in **extra/hip_gpu_driver/test_sdma_fun.py** was also mentioned, suggesting it too shouldn't count as a valid test.
- **Filtering out irrelevant tests**: The team agreed to filter out the **extra/** folder when running **pytest**, thereby avoiding unnecessary failures.
   - This will ensure only pertinent tests within the **test/** directory are executed.
- **Acknowledgment of repo junk**: **George Hotz** confirmed that files in the **extra/** folder are considered 'repo junk' and should not be included in testing.
   - Only the relevant tests located in the **test/** directory should be executed for accurate results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/518c022c29104d79c7a50ec41af5b7e6404da317/extra/hip_gpu_driver/test_kfd_2.py#L31)">tinygrad/extra/hip_gpu_driver/test_kfd_2.py at 518c022c29104d79c7a50ec41af5b7e6404da317 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5917">hip_ioctl changes by wozeparrot · Pull Request #5917 · tinygrad/tinygrad</a>: feat: allow specifying processor as envvar feat: vendor kfd_ioctl.h
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1285322186370060482)** (8 messages🔥): 

> - `VRAM allocation spikes`
> - `Tinygrad Tensor error`
> - `Diffusers fork with Tinygrad`
> - `NotebookLM podcast`
> - `Fundamental operations in Tinygrad` 


- **Debugging VRAM spikes**: A user inquired about the best methods to identify the causes of **spikes in VRAM allocation**.
   - The discussion hints at potential tools or strategies being sought to monitor **memory usage** effectively.
- **Tinygrad Tensor Error Investigation**: A member reported encountering an error while running a **Tinygrad** code snippet involving Tensor manipulation.
   - They linked to an [open issue](https://github.com/tinygrad/tinygrad/issues/6352) that might provide context to the problem.
- **Forking Diffusers to use Tinygrad**: A member shared updates about collaborating on a **Diffusers fork** that integrates Tinygrad instead of Torch.
   - They expressed hope for a new approach without copying existing implementations too closely.
- **NotebookLM Turns Tinygrad Into Podcast**: A member shared that **NotebookLM** created an **8-minute podcast** explaining Tinygrad using engaging analogies.
   - They highlighted that the podcast effectively serves as a **sales pitch for tinybox**.
- **Understanding Tinygrad's Simplistic Approach**: A user illustrated Tinygrad's efficiency by stating, *“They use only 3 fundamental operation types.”*
   - They compared this to a master chef using just three knives, emphasizing the simplicity behind powerful tools.



**Link mentioned**: <a href="https://github.com/tinygrad/tinygrad/issues/6352)">Issues · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - Issues · tinygrad/tinygrad

  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1285331547032911883)** (10 messages🔥): 

> - `Cohere Chat API Safety Modes`
> - `Cohere's market strategy`
> - `Training language models`
> - `Applying to Cohere` 


- **Cohere launches beta Safety Modes in Chat API**: Cohere announced the beta launch of **Safety Modes**, a new feature in the Cohere Chat API that allows customers to customize model outputs to meet safety requirements.
   - *This could potentially allow users to implement safety checks and mitigate liability concerns*.
- **Cohere's focused market approach**: **Cohere** strategically navigates the crowded LLM market by honing in on specific use cases, avoiding the oversaturated landscape.
   - Members discussed the value of **pragmatic business choices** that emphasize clarity and utility in model applications.
- **Guard rails still necessary for safety**: While the new **Safety Modes** offer baseline checks, some members emphasized the need for further guard rails to ensure comprehensive user safety.
   - *It's just a base safety check*, implying users should maintain additional safeguards.
- **Teaching models local languages**: A member humorously questioned whether teaching **Command-R** to understand Toronto slang was feasible, acknowledging its existing capabilities.
   - This sparked discussions on biases in language models, with one pointing out a potential **Canadian bias**.
- **New candidate in the Cohere discussion**: A new member introduced themselves, stating they recently applied for a position at Cohere focused on the **Japanese language**.
   - The community reacted positively, welcoming the newcomer to the group.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1285329176496312360)** (1 messages): 

> - `Fine-tuning models`
> - `Dataset management`
> - `Cohere platform capabilities` 


- **Inquiry about Skipping End Tokens in Fine-tuning**: A user inquired whether it's possible to skip the final `<|END_OF_TURN_TOKEN|>` during fine-tuning on demand, aiming for a more seamless inference continuation.
   - They suggested a POC example of training data and shared that they believe there might be potential for this feature, especially for fine-tuning chat models.
- **Understanding Cohere's Dataset Management**: The discussion included references to the [Listed Dataset Types](https://docs.cohere.com/docs/datasets#dataset-types) on the Cohere platform and its implications for managing datasets for fine-tuning.
   - Key points covered included dataset limits, retention policies, and the ability to manage datasets via the Dashboard or [Datasets API](https://docs.cohere.com/reference/create-dataset).



**Link mentioned**: <a href="https://docs.cohere.com/docs/datasets#dataset-types>),">Datasets — Cohere</a>: The document provides an overview of the Dataset API, including file size limits, data retention policies, dataset creation, validation, metadata preservation, using datasets for fine-tuning models, d...

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1285347997911355483)** (4 messages): 

> - `Sagemaker Client Issues`
> - `Cohere Support` 


- **Billed Units Show -1.0 in Sagemaker**: A user reported that in the response from the Sagemaker client, the billed units return `input_tokens=-1.0` and `output_tokens=-1.0` when hitting the endpoint.
   - This issue raises questions about potential input misconfigurations or errors when setting up the endpoint.
- **Support Recommended for Sagemaker Inquiry**: Another user suggested that the original poster should contact [support@cohere.com](mailto:support@cohere.com) for further assistance.
   - They offered to look into the user's account to better address the billing unit issue.


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1285601778943135844)** (3 messages): 

> - `GitHub Responses`
> - `CodeBlueprint with Aider`
> - `Ruff Check Errors` 


- **GitHub Communication Update**: @connorshorten informed that he has responded to Prashant on **GitHub** regarding ongoing discussions.
   - *Stay tuned for any follow-up reactions* from Prashant on this interaction.
- **Showcasing CodeBlueprint Pattern**: A member shared a link to demonstrate their new pattern, **CodeBlueprint with Aider**.
   - This showcase could provide insights into the integration of new tools within coding practices.
- **Encountered Errors with Ruff Check**: Prashant reported facing an error when running `ruff check . --fix-only`, citing a **TOML parse error**.
   - The error suggests there is an **unknown field `indent-width`** in the configuration which does not match expected parameters.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1285339587677257738)** (11 messages🔥): 

> - `GPT-4 Vision API wrapper`
> - `Contributions and Bounties`
> - `Documentation Needs`
> - `DSPy Program API Flexibility` 


- **Introduction of GPT-4 Vision API wrapper**: A new [Pull Request](https://github.com/stanfordnlp/dspy/pull/682) adds a **GPT-4 Vision API wrapper** in the DSPy repository, simplifying requests for image analysis.
   - The change introduces a **GPT4Vision** class in `visionopenai.py`, streamlining the API interaction process.
- **Interest in Contributions and Bounties**: Members expressed a desire to contribute, with one asking, *'I would love to contribute? any bounties you have in mind?'*
   - While there was acknowledgment of needed changes, specifics on available bounties were not discussed.
- **Request for Simple Documentation**: A member indicated there is a requirement for **simple documentation** and expressed their willingness to help with it.
   - This reflects an ongoing effort within the community to improve resources and support for users.
- **DSPy Program Flexibility Inquiry**: A general question was raised about **calling the optimized DSPy program** from a different programming language rather than being confined to Python.
   - Another member suggested that it may require a **Python VM**, with a mention that complex inter-process communication could be an alternative.



**Link mentioned**: <a href="https://github.com/stanfordnlp/dspy/pull/682">Add GPT-4 Vision API wrapper by jmanhype · Pull Request #682 · stanfordnlp/dspy</a>: Introduce a new GPT4Vision class in visionopenai.py that wraps the GPT-4 Vision API. This abstraction layer simplifies the process of making requests to the API for analyzing images. Key functional...

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1285384200144556082)** (10 messages🔥): 

> - `Image Compositing Techniques`
> - `Pillow Library for Image Processing`
> - `Text Integration in Images`
> - `Creative Process with Nouswise`
> - `Whisper Speech Support` 


- **Compositing Techniques: Basic but Effective**: Members discussed that basic **compositing** techniques are viable options for image creation, though specific libraries like **Pillow** are suggested.
   - One member emphasized that *training images with integrated text is not recommended* to achieve poster-quality visuals.
- **Post-Processing for Better Image Quality**: An effective workflow for achieving high-quality imagery involves tools like **GIMP**, where post-processing can greatly enhance accuracy and effectiveness.
   - It was noted that *doing it in post yields the best results* compared to other methods.
- **Nouswise: A Tool for the Creative Process**: **Nouswise** was highlighted as a personal search engine offering trusted answers throughout various stages of the creative process, from **reading** to **curation**.
   - Its functionality includes effective methods for **searching** and **writing**, enhancing overall productivity.
- **Seeking Support for Whisper Speech**: A user inquired about experiences with **Whisper speech** technology, prompting a suggestion to review a specific channel for guidance.
   - Community engagement offered a path for collective knowledge sharing, with links ensuring access to relevant discussions.
- **Resource Support for StyleTTS-ZS**: A member requested computational resource support for the **StyleTTS-ZS** project, linked in the discussion.
   - This **GitHub** project promises efficient high-quality zero-shot text-to-speech synthesis using advanced techniques and encourages community collaboration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/yl4579/StyleTTS-ZS">GitHub - yl4579/StyleTTS-ZS: StyleTTS-ZS: Efficient High-Quality Zero-Shot Text-to-Speech Synthesis with Distilled Time-Varying Style Diffusion</a>: StyleTTS-ZS: Efficient High-Quality Zero-Shot Text-to-Speech Synthesis with Distilled Time-Varying Style Diffusion - yl4579/StyleTTS-ZS</li><li><a href="https://nouswise.com">Nouswise</a>: Nouswise is your personal search engine, grounded in the personal information you trust the most.
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

mkaic: https://mistral.ai/news/pixtral-12b/
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1285509730889891871)** (5 messages): 

> - `Open Interpreter updates`
> - `Beta testing inquiry`
> - `01 app functionality` 


- **Open Interpreter's Cleverness**: A member praised the **Open Interpreter** for being **clever**, highlighting its potential capabilities.
   - *Geese* signified excitement about the tool's functionality within the community.
- **Inquiry about Beta Testing Opportunities**: A member inquired if there's still **capacity for beta testers** for the **Open Interpreter**, expressing excitement about sharing innovative ideas.
   - The inquiry suggests ongoing interest and willingness to contribute to its development.
- **Questions about 01 App Functionality**: A member asked if others managed to get the **01 app** working on their phones, signaling potential user concerns or interest.
   - Another member confirmed their success with the **01 app**, indicating positive user experiences.
- **Discussion on 01 Light Feature**: A member mentioned **01 light**, possibly referring to a feature or update within the **01 app** context.
   - This points to ongoing discussions about various functionalities related to the app.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1285631612490612748)** (2 messages): 

> - `Human Device Discord Event`
> - `Beta Availability Inquiry` 


- **Human Device hosts Discord event this Friday**: A message highlights an upcoming event by **Human Device** scheduled for this Friday. Interested participants can join via the [Discord link](https://discord.gg/UmXdvf3v?event=1285618083448225813).
- **Inquiry about Beta availability**: A member inquired if there is any space available in the **beta** version currently. This suggests ongoing interest in participating in the beta testing.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1285668862330933309)** (2 messages): 

> - `Tool Use Podcast`
> - `01 Voices Script`
> - `Voice Agents in Group Conversations`
> - `Deepgram Local Version` 


- **Exciting Episode of Tool Use Features Killian**: This week's episode of [Tool Use](https://youtu.be/La9BfaFTsFU) features special guest **Killian Lucas**, diving into the realm of voice intelligence.
   - The episode discusses the innovations in voice agents, with Killian sharing insights as the creator.
- **Showcasing Extensible 01 Voices Script**: **mikebirdtech** highlighted a great episode showcasing how extensible the **01** is, with an impressive script by **<@601600720864542741>**.
   - The demonstration included how voice agents can actively participate in group conversations without overreacting to unrelated statements.
- **Community Contribution: Open Source Deepgram**: A member announced they created an open-source and local version of **Deepgram**, expressing excitement about the project.
   - This highlights the community's engagement in developing tools for voice intelligence.



**Link mentioned**: <a href="https://youtu.be/La9BfaFTsFU">The Future of Voice Agents with Killian Lucas - Ep 5 - Tool Use</a>: Join us for this week&#39;s episode of Tool Use as we dive into the exciting world of voice intelligence. We&#39;re joined by special guest Killian Lucas, creator of...

  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1285672763239170212)** (5 messages): 

> - `Eleuther Eval Recipe`
> - `Cache Management`
> - `Model Generation Issues` 


- **Concerns about Eleuther Eval with Generation & MC Tasks**: A user questioned if the Eleuther eval recipe works effectively with both **generation** and multiple choice (**mc**) tasks, suspecting that the **cache** from generation tasks might affect subsequent tasks.
   - Another user confirmed that the recipe is indeed broken, indicating potential underlying issues with cache management.
- **Need for Cache Reset**: Discussion arose about whether a missing reset for caches is causing issues, especially when changing tasks post-generation.
   - One member commented that they reset caches after every model generation, but this only prepares them for a new generation without full reset functionality.
- **Batch Size Expectations in MM Evals**: A parallel issue was identified where expected batch sizes are not always achieved during model evaluations when caching is enabled.
   - This issue may surface again when another user attempts MM evaluations in future instances.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1285560688991014975)** (2 messages): 

> - `RISC-V support` 


- **Future Plans for RISC-V Support**: A member inquired about any plans to support **RISC-V**.
   - Another member responded, stating that there are **no plans yet** to support RISC-V at this time.
- **Community Interest in RISC-V**: The question about RISC-V indicates a growing **community interest** in updating support for diverse architectures.
   - The lack of plans at this stage may spark further discussion on alternative architecture compatibility in the future.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1285318423169204289)** (2 messages): 

> - `Zero-copy data interoperability`
> - `Mandelbrot example`
> - `LLVM intrinsics in Mojo` 


- **Zero-copy data interoperability remains a challenge**: Currently, there's no support for importing Mojo modules or calling its functions from Python, which is seen as a pre-requisite for zero-copy interoperability.
   - Members are curious if it's feasible to transfer data between Mojo and Python without copying, referencing the Mandelbrot example like `numpy_array.itemset((row, col), matrix[row, col])` as a potential inefficiency.
- **Mandelbrot set showcases Mojo's capabilities**: The tutorial on the Mandelbrot set highlights how Mojo can write high-performance code while leveraging Python's ecosystem for visualization.
   - It emphasizes that Mojo is suitable for developing fast programs, specifically for irregular applications, which could benefit from Python's libraries.
- **Mojo extends LLVM intrinsics support at comptime**: Following the Mojo community meeting, it was revealed that Mojo now supports LLVM intrinsics at comptime, particularly for functions dealing with integers like `ctlz` and `popcount`.
   - This feature will simplify future extensions to support other types, contingent on LLVM's ability to constant fold these intrinsics.



**Link mentioned**: <a href="https://docs.modular.com/mojo/notebooks/Mandelbrot">Mandelbrot in Mojo with Python plots | Modular Docs</a>: Learn how to write high-performance Mojo code and import Python packages.

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1285450423636983920)** (2 messages): 

> - `Shampoo in Transformers`
> - `Liger usage`
> - `Shampoo Scaling Law`
> - `Performance of Shampoo`
> - `Shampoo vs Adam` 


- **Shampoo's Absence in Transformers and Axolotl**: A member raised a query about the lack of implementation of **Shampoo** in **Transformers** and **Axolotl**, sparking discussion on its potential benefits.
   - The original poster noted that *Shampoo is literally such a free lunch, in large scale, in predictable manner* and seems to be overlooked.
- **Discussion around Shampoo Scaling Law**: A link was shared discussing the **Shampoo Scaling law for language models**, comparing its performance against **Adam**.
   - The plot referenced **Kaplan et al** and highlighted the effective scaling characteristics of Shampoo in large models.



**Link mentioned**: <a href="https://x.com/cloneofsimo/status/1836003682141577418">Tweet from Simo Ryu (@cloneofsimo)</a>: Shampoo Scaling law for language model Plot taste of Kaplan et al, but comparing shampoo and adam. Shampoo is literally such a free lunch, in large scale, in predictable manner.

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1285380011775230053)** (1 messages): 

> - `YOLO Vision 2024`
> - `Ultralytics Event`
> - `Google Campus for Startups` 


- **Join Ultralytics at YOLO Vision 2024!**: Ultralytics is hosting [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision) on <t:1727424000:F> - <t:1727458200:t> at Google Campus for Startups in Madrid 🇪🇸.
   - Make sure to register to attend and participate in deciding the music for the discussion panel on communities!
- **Vote for Music at YOLO Vision 2024!**: Once registered for [YOLO Vision 2024](https://www.ultralytics.com/events/yolovision), attendees can vote for the music played during the discussion panel.
   - This interactive element aims to enhance engagement during the event, encouraging community involvement!


  

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
